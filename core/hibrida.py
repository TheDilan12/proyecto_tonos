"""E11 · Red híbrida "medir en vez de regresar".

Diagnóstico que la motiva (análisis del error de E10):
- Una CNN con pooling global no puede *seleccionar* píxeles: promediar los píxeles del color
  dominante con su máscara exacta reproduce el objetivo con ΔE ≈ 0,5, pero la misma máscara a
  1/4 de resolución ya da ΔE ≈ 3,8 (el error de E10).
- Sin referencia blanca, el iluminante de la foto queda implícito; la misma prenda en catálogo y
  en foto de cliente recibía predicciones distintas (ΔE ≈ 2,8).

Arquitectura: codificador MobileNetV3 + decodificador FPN ligero hasta la resolución nativa
(80 × 60). La red produce
- un mapa de atención `w` (softmax espacial) sobre los píxeles que pertenecen al color dominante,
- tres log-ganancias `g` por canal (corrección del iluminante, tipo Von Kries),
- un residuo acotado (± 3 unidades Lab).
El color se *mide* sobre los píxeles reales: Lab(Σ w · e^g ⊙ RGB_lineal) + residuo.

Supervisión gratuita: `w` contra la máscara del color dominante del objetivo (k-means) y `g` contra
la ganancia verdadera (balance de blancos del catálogo ÷ iluminante y brillo simulados).
Pérdida principal: ΔE2000 diferenciable.
"""

import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from core import config
from core.datos import decodificar
from core.modelos import DISPOSITIVO, MEDIA, STD
from core.objetivo_color import srgb_a_lineal
from core.perturbaciones import rgb_de_temperatura, simular_foto_real

TAM_CODIFICADOR = (160, 120)   # 2 × la resolución nativa

# --- color en torch (diferenciable) ---------------------------------------
_M_XYZ = torch.tensor([[0.412453, 0.357580, 0.180423],
                       [0.212671, 0.715160, 0.072169],
                       [0.019334, 0.119193, 0.950227]])
_BLANCO_D65 = torch.tensor([0.95047, 1.0, 1.08883])


def lineal_a_lab(lin):
    """RGB lineal (…, 3) en [0, 1] -> CIELAB (D65), igual que skimage.color.rgb2lab."""
    xyz = lin.clamp(0, 1) @ _M_XYZ.to(lin).T / _BLANCO_D65.to(lin)
    f = torch.where(xyz > 0.008856, xyz.clamp(min=1e-6) ** (1 / 3), 7.787 * xyz + 16 / 116)
    L = torch.where(xyz[..., 1] > 0.008856, 116 * f[..., 1] - 16, 903.3 * xyz[..., 1])
    return torch.stack([L, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], dim=-1)


def ciede2000(lab1, lab2, eps=1e-8):
    """ΔE2000 diferenciable entre tensores (…, 3)."""
    L1, a1, b1 = lab1.unbind(-1)
    L2, a2, b2 = lab2.unbind(-1)
    C1, C2 = torch.sqrt(a1 ** 2 + b1 ** 2 + eps), torch.sqrt(a2 ** 2 + b2 ** 2 + eps)
    Cb7 = ((C1 + C2) / 2) ** 7
    G = 0.5 * (1 - torch.sqrt(Cb7 / (Cb7 + 25.0 ** 7) + 1e-6))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = torch.sqrt(a1p ** 2 + b1 ** 2 + eps), torch.sqrt(a2p ** 2 + b2 ** 2 + eps)
    # atan2 tiene gradiente indefinido en (0, 0): se desplaza un epsilon
    h1p = torch.rad2deg(torch.atan2(b1, a1p + 1e-6)) % 360
    h2p = torch.rad2deg(torch.atan2(b2, a2p + 1e-6)) % 360
    dLp, dCp = L2 - L1, C2p - C1p
    dh = h2p - h1p
    dh = torch.where(dh > 180, dh - 360, torch.where(dh < -180, dh + 360, dh))
    dHp = 2 * torch.sqrt(C1p * C2p + eps) * torch.sin(torch.deg2rad(dh) / 2)
    Lbp, Cbp = (L1 + L2) / 2, (C1p + C2p) / 2
    suma = h1p + h2p
    hbp = torch.where((h1p - h2p).abs() > 180, torch.where(suma < 360, (suma + 360) / 2, (suma - 360) / 2), suma / 2)
    rad = torch.deg2rad
    T = (1 - 0.17 * torch.cos(rad(hbp - 30)) + 0.24 * torch.cos(rad(2 * hbp))
         + 0.32 * torch.cos(rad(3 * hbp + 6)) - 0.20 * torch.cos(rad(4 * hbp - 63)))
    dtheta = 30 * torch.exp(-((hbp - 275) / 25) ** 2)
    Cbp7 = Cbp ** 7
    Rc = 2 * torch.sqrt(Cbp7 / (Cbp7 + 25.0 ** 7) + 1e-6)
    Sl = 1 + 0.015 * (Lbp - 50) ** 2 / torch.sqrt(20 + (Lbp - 50) ** 2)
    Sc, Sh = 1 + 0.045 * Cbp, 1 + 0.015 * Cbp * T
    Rt = -torch.sin(rad(2 * dtheta)) * Rc
    tL, tC, tH = dLp / Sl, dCp / Sc, dHp / Sh
    return torch.sqrt((tL ** 2 + tC ** 2 + tH ** 2 + Rt * tC * tH).clamp(min=0) + 1e-6)


# --- red ------------------------------------------------------------------
class RedHibrida(nn.Module):
    TOMAS = (3, 6, 12, 16)   # salidas de MobileNetV3 a 1/4, 1/8, 1/16 y 1/32

    def __init__(self, n_grupos, n_tonos=0, canales=64, preentrenada=True):
        super().__init__()
        from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
        base = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2 if preentrenada else None)
        self.features = base.features
        self.n_grupos, self.n_tonos, self.tipo = n_grupos, n_tonos, "hibrida"
        self.tam = TAM_CODIFICADOR
        ch = [24, 40, 112, 960]
        self.laterales = nn.ModuleList([nn.Conv2d(c, canales, 1) for c in ch])
        self.suavizado = nn.Sequential(nn.Conv2d(canales, canales, 3, padding=1), nn.BatchNorm2d(canales), nn.Hardswish(),
                                       nn.Conv2d(canales, canales, 3, padding=1), nn.BatchNorm2d(canales), nn.Hardswish())
        # el mapa de atención también ve la imagen nativa (bordes finos de la prenda)
        self.atencion = nn.Sequential(nn.Conv2d(canales + 3, 32, 3, padding=1), nn.Hardswish(), nn.Conv2d(32, 1, 1))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cabeza_ganancia = nn.Sequential(nn.Linear(960, 256), nn.Hardswish(), nn.Linear(256, 3))
        self.cabeza_residuo = nn.Sequential(nn.Linear(960 + 3, 128), nn.Hardswish(), nn.Linear(128, 3))
        self.cabeza_grupo = nn.Sequential(nn.Dropout(0.2), nn.Linear(960, n_grupos))
        self.cabeza_tono = nn.Sequential(nn.Dropout(0.2), nn.Linear(960, n_tonos)) if n_tonos else None
        for cabeza in (self.cabeza_ganancia, self.cabeza_residuo):   # arrancar sin corrección
            nn.init.zeros_(cabeza[-1].weight); nn.init.zeros_(cabeza[-1].bias)

    def arquitectura(self):
        return {"tipo": "hibrida", "n_grupos": self.n_grupos, "n_tonos": self.n_tonos}

    def forward(self, x, lin):
        """x: imagen normalizada (B, 3, 160, 120); lin: RGB lineal nativo (B, 3, 80, 60).
        Retorna dict con lab, logits de grupo/tono, logits de atención (B, 80*60) y log-ganancias."""
        feats, h = [], x
        for i, capa in enumerate(self.features):
            h = capa(h)
            if i in self.TOMAS:
                feats.append(h)
        p = self.laterales[3](feats[3])
        for k in (2, 1, 0):
            p = F.interpolate(p, size=feats[k].shape[-2:], mode="bilinear", align_corners=False) + self.laterales[k](feats[k])
        p = self.suavizado(p)
        p = F.interpolate(p, size=lin.shape[-2:], mode="bilinear", align_corners=False)
        logits_w = self.atencion(torch.cat([p, lin], dim=1)).flatten(1)          # (B, H*W)
        w = logits_w.softmax(1)
        glob = self.pool(feats[3]).flatten(1)
        g = 2.0 * torch.tanh(self.cabeza_ganancia(glob) / 2.0)                    # log-ganancias en (-2, 2)
        px = lin.flatten(2).transpose(1, 2) * torch.exp(g)[:, None, :]             # (B, H*W, 3)
        lab_medido = lineal_a_lab((w[..., None] * px.clamp(0, 1)).sum(1))
        residuo = 3 * torch.tanh(self.cabeza_residuo(torch.cat([glob, lab_medido.detach() / 100], 1)))
        tono = self.cabeza_tono(glob) if self.cabeza_tono is not None else None
        return {"lab": lab_medido + residuo, "grupo": self.cabeza_grupo(glob), "tono": tono,
                "logits_w": logits_w, "g": g}


# --- datos ----------------------------------------------------------------
def _tensores(rgb, voltear):
    if voltear:
        rgb = rgb[:, ::-1]
    lin = torch.from_numpy(srgb_a_lineal(rgb.astype(np.float32) / 255.0).copy()).permute(2, 0, 1).float()
    x = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()[None] / 255
    x = F.interpolate(x, size=TAM_CODIFICADOR, mode="bilinear", align_corners=False)[0]
    return (x - MEDIA) / STD, lin


class DatasetHibrido(torch.utils.data.Dataset):
    """Foto (opcionalmente con contexto real simulado) + objetivo Lab + máscara del color dominante +
    log-ganancia verdadera. Aumento: volteo horizontal (consistente en imagen y máscara)."""

    def __init__(self, bytes_, lab, grupos=None, tonos=None, mapas=None, mascaras=None, ganancias=None,
                 entrenamiento=False, contexto_real=0.0, semilla=0, voltear=False):
        n = len(bytes_)
        self.bytes_, self.lab = list(bytes_), np.asarray(lab, np.float32)
        self.grupos = list(grupos) if grupos is not None else [0] * n
        self.tonos = list(tonos) if tonos is not None else [0] * n
        self.mapas, self.mascaras, self.ganancias = mapas, mascaras, ganancias
        self.entrenamiento, self.p_real, self.semilla, self.voltear = entrenamiento, contexto_real, semilla, voltear
        self.epoca = 0

    def __len__(self):
        return len(self.bytes_)

    def __getitem__(self, i):
        rng = np.random.default_rng((self.semilla, self.epoca, i))
        rgb = decodificar(self.bytes_[i])
        log_g = np.log(self.ganancias[i]) if self.ganancias is not None else np.zeros(3)
        if self.p_real > 0 and rng.random() < self.p_real:
            rgb, prm = simular_foto_real(rgb, self.mapas[i], semilla=int(rng.integers(1 << 31)))
            log_g = log_g - np.log(rgb_de_temperatura(prm["kelvin_exacto"]) * prm["brillo_exacto"])
        if rgb.shape[:2] != (80, 60):   # 8 imágenes del dataset tienen otro tamaño
            rgb = np.asarray(Image.fromarray(rgb).resize((60, 80), Image.BILINEAR))
        voltear = self.voltear or (self.entrenamiento and rng.random() < 0.5)
        x, lin = _tensores(rgb, voltear)
        m = self.mascaras[i] if self.mascaras is not None else np.zeros(rgb.shape[:2], bool)
        m = torch.from_numpy((m[:, ::-1] if voltear else m).astype(np.float32).copy()).flatten()
        return (x, lin, torch.from_numpy(self.lab[i]), m, torch.from_numpy(log_g.astype(np.float32)),
                int(self.grupos[i]), int(self.tonos[i]))


# --- entrenamiento --------------------------------------------------------
def entrenar_hibrida(ds_train, ds_val, n_grupos, n_tonos, epocas=40, lote=64, lr=6e-4, ruta=None, log=print,
                     pesos=None):
    """Entrena `RedHibrida`. Selecciona el mejor modelo por ΔE2000 mediano en `ds_val` (con contexto real)."""
    pesos = {"dE": 1.0, "dE76": 0.1, "atencion": 0.2, "ganancia": 10.0, "grupo": 0.3, "tono": 0.3, **(pesos or {})}
    torch.manual_seed(config.SEMILLA)
    red = RedHibrida(n_grupos, n_tonos).to(DISPOSITIVO)
    opt = torch.optim.AdamW(red.parameters(), lr=lr, weight_decay=1e-4)
    pasos = epocas * int(np.ceil(len(ds_train) / lote))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=pasos, pct_start=0.1)
    gpu = DISPOSITIVO.type == "cuda"
    dl = torch.utils.data.DataLoader(ds_train, batch_size=lote, shuffle=True, num_workers=6 if gpu else 0,
                                     pin_memory=gpu, drop_last=True)
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    if ruta is not None:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.with_suffix(".json").write_text(json.dumps(red.arquitectura()))
    historia, mejor = [], np.inf
    for ep in range(epocas):
        ds_train.epoca = ep
        red.train(); t0 = time.time(); acum = []; omitidos = 0
        for x, lin, lab, m, log_g, g, t in dl:
            x, lin, lab, m, log_g, g, t = (v.to(DISPOSITIVO, non_blocking=True) for v in (x, lin, lab, m, log_g, g, t))
            s = red(x, lin)
            perdida = pesos["dE"] * ciede2000(s["lab"], lab).mean() + pesos["dE76"] * (s["lab"] - lab).norm(dim=1).mean()
            tiene = m.sum(1) > 0
            if tiene.any():
                q = m[tiene] / m[tiene].sum(1, keepdim=True)
                perdida = perdida - pesos["atencion"] * (q * s["logits_w"][tiene].log_softmax(1)).sum(1).mean()
            perdida = perdida + pesos["ganancia"] * (s["g"] - log_g).abs().mean() + pesos["grupo"] * ce(s["grupo"], g)
            if s["tono"] is not None:
                perdida = perdida + pesos["tono"] * ce(s["tono"], t)
            opt.zero_grad()
            if not torch.isfinite(perdida):
                omitidos += 1; sched.step(); continue
            perdida.backward()
            norma = nn.utils.clip_grad_norm_(red.parameters(), 5.0)
            if not torch.isfinite(norma):
                omitidos += 1; opt.zero_grad(); sched.step(); continue
            opt.step(); sched.step()
            acum.append(perdida.item())
        lab_val = predecir_hibrida(red, ds_val)["lab"]
        de = ciede2000(torch.from_numpy(lab_val), torch.from_numpy(ds_val.lab)).numpy()
        historia.append({"epoca": ep + 1, "perdida_train": float(np.mean(acum)), "dE2000_val_mediana": float(np.median(de)),
                         "dE2000_val_media": float(de.mean()), "segundos": time.time() - t0})
        log(f"  época {ep + 1}/{epocas}: pérdida {np.mean(acum):.3f} · ΔE2000 val (contexto real) mediana {np.median(de):.2f}"
            f" · media {de.mean():.2f} · {time.time() - t0:.0f} s" + (f" · {omitidos} lotes omitidos (no finitos)" if omitidos else ""))
        if np.median(de) < mejor and ruta is not None:
            mejor = np.median(de)
            torch.save(red.state_dict(), ruta)
    if ruta is not None:
        red.load_state_dict(torch.load(ruta, map_location=DISPOSITIVO))
    return red, historia


def predecir_hibrida(red, ds, lote=128, tta=False):
    """Como `modelos.predecir_red`: dict con lab, grupo, tono (probabilidades). tta promedia con el reflejo."""
    red.eval()
    disp = next(red.parameters()).device
    vistas = [False, True] if tta else [False]
    salidas = []
    for v in vistas:
        voltear_orig, ds.voltear = ds.voltear, v
        dl = torch.utils.data.DataLoader(ds, batch_size=lote, shuffle=False, num_workers=0)
        labs, grupos, tonos = [], [], []
        with torch.no_grad():
            for x, lin, *_ in dl:
                s = red(x.to(disp), lin.to(disp))
                labs.append(s["lab"].cpu().numpy()); grupos.append(s["grupo"].softmax(1).cpu().numpy())
                if s["tono"] is not None:
                    tonos.append(s["tono"].softmax(1).cpu().numpy())
        ds.voltear = voltear_orig
        salidas.append((np.concatenate(labs), np.concatenate(grupos), np.concatenate(tonos) if tonos else None))
    lab = np.mean([s[0] for s in salidas], 0)
    grupo = np.mean([s[1] for s in salidas], 0)
    tono = np.mean([s[2] for s in salidas], 0) if salidas[0][2] is not None else None
    return {"lab": lab, "grupo": grupo, "tono": tono}


def predecir_foto(red, rgb):
    """Una foto RGB (80x60) -> (Lab, mapa de atención 80x60)."""
    red.eval()
    if rgb.shape[:2] != (80, 60):
        rgb = np.asarray(Image.fromarray(rgb).resize((60, 80), Image.BILINEAR))
    x, lin = _tensores(rgb, False)
    with torch.no_grad():
        s = red(x[None], lin[None])
    return s["lab"][0].numpy(), s["logits_w"].softmax(1)[0].reshape(rgb.shape[:2]).numpy()
