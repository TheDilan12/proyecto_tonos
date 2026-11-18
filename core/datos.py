"""Carga del dataset Fashion Product Images (versión small) desde parquet."""

import io

import numpy as np
import pandas as pd
from PIL import Image

from core import config


def cargar_dataset(ruta=config.RUTA_PARQUET):
    """Lee los parquet del dataset.

    Retorna (ok, df, mensaje). `df` trae la metadata y la columna `imagen_bytes`
    con la imagen JPEG codificada; las imágenes se decodifican bajo demanda.
    """
    if not ruta.exists():
        return False, None, f"No existe la carpeta de datos: {ruta}"
    archivos = sorted(ruta.glob("*.parquet"))
    if not archivos:
        return False, None, f"No hay archivos .parquet en {ruta}"
    try:
        df = pd.concat([pd.read_parquet(a) for a in archivos], ignore_index=True)
    except Exception as e:  # parquet corrupto o incompleto
        return False, None, f"Error leyendo parquet: {e}"
    df["imagen_bytes"] = df.pop("image").map(lambda x: x["bytes"])
    df["grupo_color"] = df[config.COLUMNA_OBJETIVO].map(config.TONO_A_GRUPO)
    return True, df, f"{len(df):,} registros leídos de {len(archivos)} archivos"


def decodificar(imagen_bytes, modo="RGB"):
    """Bytes JPEG -> arreglo numpy uint8 (H, W, 3)."""
    return np.asarray(Image.open(io.BytesIO(imagen_bytes)).convert(modo))


def abrir(imagen_bytes):
    """Bytes JPEG -> PIL.Image sin convertir (conserva el modo original)."""
    return Image.open(io.BytesIO(imagen_bytes))
