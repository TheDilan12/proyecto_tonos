"""Descriptores de color e imagen usados en el EDA."""

import hashlib

import numpy as np
import pandas as pd
from PIL import Image
from skimage import color as skcolor
from skimage.morphology import disk, opening

from core import config
from core.datos import abrir


def mascara_primer_plano(rgb, umbral=config.UMBRAL_DELTA_E_FONDO):
    """Separa producto y fondo comparando cada píxel con el color del borde.

    El color de fondo se estima como la mediana Lab de los píxeles del borde.
    Retorna (mascara_bool, lab, lab_fondo).
    """
    lab = skcolor.rgb2lab(rgb)
    borde = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    lab_fondo = np.median(borde, axis=0)
    de = skcolor.deltaE_ciede2000(lab, lab_fondo[None, None, :])
    mascara = opening(de > umbral, disk(1))
    return mascara, lab, lab_fondo


def mascara_piel(rgb):
    """Heurística clásica de piel en YCrCb (Cr 133-173, Cb 77-127).

    Sirve como indicador aproximado de que la foto incluye una persona.
    """
    ycc = np.asarray(Image.fromarray(rgb).convert("YCbCr"), dtype=np.int16)
    cb, cr = ycc[..., 1], ycc[..., 2]
    return (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)


def dhash(gris, tam=16):
    """Hash perceptual por diferencias (tam x tam bits) para detectar casi-duplicados."""
    pequeno = np.asarray(Image.fromarray(gris).resize((tam + 1, tam), Image.BILINEAR), dtype=np.int16)
    bits = (pequeno[:, 1:] > pequeno[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):0{tam * tam // 4}x}"


def describir_imagen(imagen_bytes):
    """Calcula los descriptores de una imagen. Retorna un dict (con `error` si falla)."""
    try:
        img = abrir(imagen_bytes)
        modo, (ancho, alto) = img.mode, img.size
        rgb = np.asarray(img.convert("RGB"))
    except Exception as e:
        return {"error": str(e)}

    mascara, lab, lab_fondo = mascara_primer_plano(rgb)
    frac = mascara.mean()
    pix = lab[mascara] if frac >= config.FRACCION_MINIMA_PRIMER_PLANO else lab.reshape(-1, 3)
    gris = np.asarray(img.convert("L"))
    # Laplaciano como medida de nitidez (varianza baja = imagen borrosa)
    g = gris.astype(np.float32)
    lap = g[1:-1, :-2] + g[1:-1, 2:] + g[:-2, 1:-1] + g[2:, 1:-1] - 4 * g[1:-1, 1:-1]
    croma = np.hypot(pix[:, 1], pix[:, 2])
    ys, xs = np.nonzero(mascara) if frac > 0 else (np.array([alto / 2]), np.array([ancho / 2]))
    piel = mascara_piel(rgb) & mascara

    return {
        "ancho": ancho, "alto": alto, "modo": modo,
        "md5": hashlib.md5(imagen_bytes).hexdigest(),
        "dhash": dhash(gris),
        "frac_primer_plano": frac,
        "fallback_mascara": frac < config.FRACCION_MINIMA_PRIMER_PLANO,
        "frac_piel": piel.sum() / max(mascara.sum(), 1),
        "L_fondo": lab_fondo[0],
        "L": np.median(pix[:, 0]), "a": np.median(pix[:, 1]), "b": np.median(pix[:, 2]),
        "L_std": pix[:, 0].std(), "croma": np.median(croma),
        "croma_std": croma.std(),
        "centro_x": xs.mean() / ancho, "centro_y": ys.mean() / alto,
        "brillo_medio": gris.mean(), "contraste": gris.std(),
        "nitidez": lap.var(),
        "r_medio": rgb[..., 0].mean() / 255, "g_medio": rgb[..., 1].mean() / 255,
        "b_medio": rgb[..., 2].mean() / 255,
        "r_std": rgb[..., 0].std() / 255, "g_std": rgb[..., 1].std() / 255,
        "b_std": rgb[..., 2].std() / 255,
        "error": None,
    }


def describir_dataset(df, ruta_cache=config.RUTA_INTERMEDIOS / "descriptores.parquet"):
    """Descriptores de todas las imágenes (con caché en parquet). Retorna DataFrame indexado por id."""
    if ruta_cache.exists():
        return pd.read_parquet(ruta_cache)
    filas = [describir_imagen(b) for b in df["imagen_bytes"]]
    desc = pd.DataFrame(filas, index=df["id"].values)
    desc.index.name = "id"
    ruta_cache.parent.mkdir(parents=True, exist_ok=True)
    desc.to_parquet(ruta_cache)
    return desc


def lab_a_rgb(lab):
    """Convierte uno o varios colores Lab a RGB en [0, 1] (para pintar puntos)."""
    lab = np.atleast_2d(lab).astype(float)
    return np.clip(skcolor.lab2rgb(lab[None, :, :])[0], 0, 1)


def matriz_delta_e(centroides_lab):
    """Matriz de distancias CIEDE2000 entre centroides Lab (DataFrame cuadrado)."""
    v = centroides_lab.values
    m = skcolor.deltaE_ciede2000(v[:, None, :], v[None, :, :])
    return pd.DataFrame(m, index=centroides_lab.index, columns=centroides_lab.index)
