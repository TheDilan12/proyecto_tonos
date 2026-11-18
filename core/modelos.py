"""Modelos que estiman el color CIELAB de la prenda directamente de la foto (sin máscara manual).

- `RegresorLab`: regresión ridge sobre características (clásicas o profundas congeladas).
- `RedColor`: MobileNetV3-Large preentrenada con varias cabezas: color Lab (regresión), grupo
  básico (clasificación) y, opcionalmente, tono de 46 nombres (clasificación). Se entrena con
  aumento geométrico y, en la variante robusta, con fondos e iluminantes simulados manteniendo
  fijo el objetivo (constancia de color).

Cada red guarda junto a sus pesos un `.json` con su arquitectura (resolución de entrada, bloques
congelados, número de grupos y tonos), para poder cargarla sin repetir esos parámetros.
"""

import json
import time

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from core import config
from core.datos import decodificar
from core.perturbaciones import simular_foto_real

ESCALA_LAB = np.array([100.0, 100.0, 100.0], dtype=np.float32)
# GPU si existe (entrenamiento ~20x más rápido); en CPU funciona igual, solo más lento.
DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pesos ImageNet de MobileNetV3 incluidos en el proyecto (modelos/pesos_preentrenados/checkpoints/):
# así se puede entrenar en un PC cuya red bloquea download.pytorch.org.
RUTA_PESOS_BASE = config.RUTA_MODELOS / "pesos_preentrenados"
if (RUTA_PESOS_BASE / "checkpoints").exists():
    torch.hub.set_dir(str(RUTA_PESOS_BASE))


class RegresorLab:
    """Estandarización + ridge con validación cruzada del parámetro de regularización."""

    def __init__(self):
        self.modelo = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 12)))

    def fit(self, X, lab):
        self.modelo.fit(X, lab)
        return self

    def predict(self, X):
        return self.modelo.predict(X)


# --- red neuronal ---------------------------------------------------------
class RedColor(nn.Module):
    """MobileNetV3 + cabezas de color Lab, grupo y (opcional) tono.

    congelados: número de bloques iniciales congelados (7 en E8/E9 para entrenar en CPU; 0 = toda
    la red se ajusta, E10). tam: (alto, ancho) de la imagen de entrada.
    """

    def __init__(self, n_grupos, n_tonos=0, congelados=7, tam=config.TAM_ENTRADA, preentrenada=True):
        super().__init__()
        from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
        base = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2 if preentrenada else None)
        self.features = base.features
        self.congelados, self.tam, self.n_grupos, self.n_tonos = congelados, tuple(tam), n_grupos, n_tonos
        for p in self.features[:congelados].parameters():
            p.requires_grad = False
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cabeza_lab = nn.Sequential(nn.Linear(960, 256), nn.Hardswish(), nn.Dropout(0.2), nn.Linear(256, 3))
        self.cabeza_grupo = nn.Sequential(nn.Dropout(0.2), nn.Linear(960, n_grupos))
        self.cabeza_tono = nn.Sequential(nn.Dropout(0.2), nn.Linear(960, n_tonos)) if n_tonos else None

    def arquitectura(self):
        return {"n_grupos": self.n_grupos, "n_tonos": self.n_tonos, "congelados": self.congelados, "tam": list(self.tam)}

    def train(self, modo=True):
        super().train(modo)
        if self.congelados:
            self.features[:self.congelados].eval()   # BatchNorm congelado también
        return self

    def forward(self, x):
        """Retorna (lab, logits_grupo, logits_tono o None)."""
        h = self.pool(self.features(x)).flatten(1)
        tono = self.cabeza_tono(h) if self.cabeza_tono is not None else None
        return self.cabeza_lab(h), self.cabeza_grupo(h), tono


MEDIA = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def preparar_tensor(rgb, entrenamiento=False, rng=None, tam=None):
    """RGB (80x60) -> tensor normalizado de tamaño `tam` (por defecto TAM_ENTRADA).
    En entrenamiento: aumento geométrico (volteo, rotación ±10°, recorte 85-100 %)."""
    alto, ancho = tam or config.TAM_ENTRADA
    img = Image.fromarray(rgb)
    if entrenamiento:
        if rng.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        ang = rng.uniform(-10, 10)
        img = img.rotate(ang, resample=Image.BILINEAR, fillcolor=tuple(int(v) for v in rgb[0, 0]))
        esc = rng.uniform(0.85, 1.0)
        w, h = img.size
        cw, ch = int(w * esc), int(h * esc)
        x0, y0 = rng.integers(0, w - cw + 1), rng.integers(0, h - ch + 1)
        img = img.crop((x0, y0, x0 + cw, y0 + ch))
    x = torch.from_numpy(np.asarray(img.resize((ancho, alto), Image.BICUBIC))).permute(2, 0, 1).float() / 255
    return (x - MEDIA) / STD


class DatasetColor(torch.utils.data.Dataset):
    """Imágenes + objetivo Lab + grupo (+ tono). `contexto_real` aplica fondos/luz simulados con prob. p."""

    def __init__(self, bytes_, lab, grupos, mapas=None, entrenamiento=False, contexto_real=0.0, semilla=0,
                 tonos=None, tam=None):
        self.bytes_, self.lab, self.grupos = list(bytes_), lab.astype(np.float32), list(grupos)
        self.tonos = list(tonos) if tonos is not None else [0] * len(self.grupos)
        self.mapas, self.entrenamiento, self.p_real, self.semilla = mapas, entrenamiento, contexto_real, semilla
        self.tam = tam
        self.epoca = 0

    def __len__(self):
        return len(self.bytes_)

    def __getitem__(self, i):
        rng = np.random.default_rng((self.semilla, self.epoca, i))
        rgb = decodificar(self.bytes_[i])
        if self.p_real > 0 and rng.random() < self.p_real:
            rgb, _ = simular_foto_real(rgb, self.mapas[i], semilla=int(rng.integers(1 << 31)))
        x = preparar_tensor(rgb, self.entrenamiento, rng, self.tam)
        return x, torch.from_numpy(self.lab[i] / ESCALA_LAB), int(self.grupos[i]), int(self.tonos[i])


def entrenar_red(ds_train, ds_val, n_grupos, epocas=8, lote=64, lr=6e-4, peso_grupo=0.3, ruta=None, log=print,
                 n_tonos=0, peso_tono=0.3, congelados=7, tam=None, peso_lab=(1.0, 1.0, 1.0)):
    """Entrena `RedColor`. Guarda el mejor modelo (menor ΔE76 medio en validación) en `ruta` y su
    arquitectura en `ruta.with_suffix('.json')`.

    peso_lab: peso por canal (L*, a*, b*) en la pérdida Huber; >1 en L* castiga más los errores de
    claridad, que son los que más dependen de la exposición de la foto.
    """
    torch.manual_seed(config.SEMILLA)
    tam = tam or config.TAM_ENTRADA
    red = RedColor(n_grupos, n_tonos=n_tonos, congelados=congelados, tam=tam).to(DISPOSITIVO)
    ds_train.tam = ds_val.tam = tuple(tam)
    opt = torch.optim.AdamW([p for p in red.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    pasos = epocas * int(np.ceil(len(ds_train) / lote))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=pasos, pct_start=0.15)
    gpu = DISPOSITIVO.type == "cuda"   # con GPU, la decodificación/aumento en CPU va en paralelo
    dl_tr = torch.utils.data.DataLoader(ds_train, batch_size=lote, shuffle=True,
                                        num_workers=4 if gpu else 0, pin_memory=gpu, persistent_workers=False)
    w = torch.tensor(peso_lab, dtype=torch.float32, device=DISPOSITIVO)
    huber, ce = nn.SmoothL1Loss(beta=0.05), nn.CrossEntropyLoss(label_smoothing=0.1)
    historia, mejor = [], np.inf
    if ruta is not None:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.with_suffix(".json").write_text(json.dumps(red.arquitectura()))
    for ep in range(epocas):
        ds_train.epoca = ep
        red.train(); t0 = time.time(); perdidas = []
        for x, y, g, t in dl_tr:
            x, y, g, t = x.to(DISPOSITIVO), y.to(DISPOSITIVO), g.to(DISPOSITIVO), t.to(DISPOSITIVO)
            p_lab, p_g, p_t = red(x)
            perdida = huber(p_lab * w, y * w) + peso_grupo * ce(p_g, g)
            if p_t is not None:
                perdida = perdida + peso_tono * ce(p_t, t)
            opt.zero_grad(); perdida.backward(); opt.step(); sched.step()
            perdidas.append(perdida.item())
        pred_val = predecir_red(red, ds_val)["lab"]
        de76 = np.linalg.norm(pred_val - ds_val.lab, axis=1).mean()
        historia.append({"epoca": ep + 1, "perdida_train": float(np.mean(perdidas)), "dE76_val": float(de76),
                         "segundos": time.time() - t0})
        log(f"  época {ep + 1}/{epocas}: pérdida {np.mean(perdidas):.4f} · ΔE76 val {de76:.2f} · {time.time() - t0:.0f} s")
        if de76 < mejor and ruta is not None:
            mejor = de76
            torch.save(red.state_dict(), ruta)
    if ruta is not None:
        red.load_state_dict(torch.load(ruta, map_location=DISPOSITIVO))
    return red, historia


def predecir_red(red, ds, lote=128, tta=False):
    """Retorna dict con 'lab' (n, 3), 'grupo' (n, n_grupos) y 'tono' (n, n_tonos) o None (probabilidades).
    tta=True promedia la predicción de la imagen y de su reflejo horizontal."""
    red.eval()
    disp = next(red.parameters()).device
    ds.tam = red.tam
    dl = torch.utils.data.DataLoader(ds, batch_size=lote, shuffle=False, num_workers=0)
    labs, grupos, tonos = [], [], []
    with torch.no_grad():
        for x, *_ in dl:
            x = x.to(disp)
            vistas = [x, torch.flip(x, dims=[3])] if tta else [x]
            salidas = [red(v) for v in vistas]
            labs.append(torch.stack([s[0] for s in salidas]).mean(0).cpu().numpy() * ESCALA_LAB)
            grupos.append(torch.stack([s[1].softmax(1) for s in salidas]).mean(0).cpu().numpy())
            if salidas[0][2] is not None:
                tonos.append(torch.stack([s[2].softmax(1) for s in salidas]).mean(0).cpu().numpy())
    return {"lab": np.concatenate(labs), "grupo": np.concatenate(grupos),
            "tono": np.concatenate(tonos) if tonos else None}


def cargar_red(ruta, n_grupos=None):
    """Carga en CPU (para notebooks/Grad-CAM); los pesos entrenados en GPU se mapean a CPU.
    La arquitectura se lee del .json junto a los pesos (si no existe: la de E8/E9)."""
    meta = ruta.with_suffix(".json")
    arq = json.loads(meta.read_text()) if meta.exists() else {"n_grupos": n_grupos}
    if arq.get("tipo") == "hibrida":
        from core.hibrida import RedHibrida
        red = RedHibrida(arq["n_grupos"], arq.get("n_tonos", 0), preentrenada=False)
        red.load_state_dict(torch.load(ruta, map_location="cpu"))
        return red.eval()
    red = RedColor(arq["n_grupos"], n_tonos=arq.get("n_tonos", 0), congelados=arq.get("congelados", 7),
                   tam=arq.get("tam", config.TAM_ENTRADA), preentrenada=False)
    red.load_state_dict(torch.load(ruta, map_location="cpu"))
    return red.eval()


def grad_cam(red, x):
    """Grad-CAM sobre el último bloque convolucional, respecto al croma predicho (a*, b*)."""
    activ = {}
    capa = red.features[-1]
    h1 = capa.register_forward_hook(lambda m, i, o: activ.__setitem__("a", o))
    red.eval(); x = x.clone().requires_grad_(False)
    p_lab = red(x)[0]
    objetivo = p_lab[:, 1:].norm(dim=1).sum() + p_lab[:, 0].sum() * 0.2
    grads = torch.autograd.grad(objetivo, activ["a"])[0]
    h1.remove()
    pesos = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((pesos * activ["a"]).sum(1)).detach()
    cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
    return cam.numpy()
