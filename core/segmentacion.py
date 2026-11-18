"""Segmentación de prendas con un modelo de human parsing (SegFormer-B2 entrenado en ATR).

Reemplaza la máscara por umbral del EDA: funciona sobre cualquier foto (fondo de color,
persona, otras prendas) y separa la prenda de la piel, el cabello y el resto de la ropa.
"""

import time

import numpy as np
import torch
from PIL import Image

from core import config
from core.datos import decodificar

_MODELO = None


def cargar_modelo():
    """Carga (una sola vez) el procesador y el modelo SegFormer desde Hugging Face."""
    global _MODELO
    if _MODELO is None:
        from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor
        proc = SegformerImageProcessor.from_pretrained(config.SEG_MODELO)
        modelo = AutoModelForSemanticSegmentation.from_pretrained(config.SEG_MODELO).eval()
        _MODELO = (proc, modelo)
    return _MODELO


def segmentar(imagenes_rgb):
    """Lista de arreglos RGB (H, W, 3) -> lista de mapas de clases (H, W) uint8 del mismo tamaño."""
    proc, modelo = cargar_modelo()
    alto, ancho = config.SEG_TAM
    pils = [Image.fromarray(im).resize((ancho, alto), Image.BICUBIC) for im in imagenes_rgb]
    with torch.no_grad():
        entrada = proc(images=pils, return_tensors="pt", size={"height": alto, "width": ancho})
        logits = modelo(**entrada).logits
    salida = []
    for im, lg in zip(imagenes_rgb, logits):
        lg = torch.nn.functional.interpolate(lg[None], size=im.shape[:2], mode="bilinear", align_corners=False)
        salida.append(lg[0].argmax(0).numpy().astype(np.uint8))
    return salida


def segmentar_dataset(df, ruta_cache=config.RUTA_INTERMEDIOS / "segmentacion.npz", lote=config.SEG_LOTE):
    """Segmenta todas las imágenes de `df` (con caché). Retorna dict id -> mapa de clases."""
    if ruta_cache.exists():
        datos = np.load(ruta_cache, allow_pickle=True)
        return dict(zip(datos["ids"].tolist(), datos["mapas"]))
    ids = df["id"].to_numpy()
    mapas = np.empty(len(ids), dtype=object)
    inicio = time.time()
    for i in range(0, len(ids), lote):
        imgs = [decodificar(b) for b in df["imagen_bytes"].iloc[i:i + lote]]
        for j, m in enumerate(segmentar(imgs)):
            mapas[i + j] = m
        if (i // lote) % 50 == 0:
            print(f"  segmentadas {i + len(imgs):,}/{len(ids):,} ({time.time() - inicio:.0f} s)", flush=True)
    ruta_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ruta_cache, ids=ids, mapas=mapas)
    return dict(zip(ids.tolist(), mapas))


def mascara_prenda(mapa, subcategoria):
    """Máscara booleana de la prenda objetivo según la subcategoría del producto."""
    clases = config.CLASES_PRENDA_POR_SUBCATEGORIA.get(subcategoria, [4, 5, 6, 7])
    return np.isin(mapa, clases)


def mascara_persona(mapa):
    """Todo lo que no es fondo (persona + prendas + accesorios)."""
    return ~np.isin(mapa, config.CLASES_FONDO)
