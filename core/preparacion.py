"""Preparación de datos (CRISP-DM etapa 3): tabla de prendas con objetivo continuo y conjuntos.

Une la metadata, la segmentación, el color medido (objetivo continuo), la coherencia con la
etiqueta discreta y la partición train/val/test del notebook 01 (agrupada por casi-duplicados).
"""

import io

import numpy as np
import pandas as pd
from PIL import Image

from core import config
from core.datos import cargar_dataset, decodificar
from core.objetivo_color import (coherencia_con_etiqueta, medir_dataset, prototipos_por_tono)
from core.perturbaciones import simular_foto_real
from core.segmentacion import segmentar_dataset

RUTA_TABLA = config.RUTA_INTERMEDIOS / "tabla_prendas.parquet"
RUTA_TEST_REAL = config.RUTA_INTERMEDIOS / "test_contexto_real.parquet"
RUTA_PROTOTIPOS = config.RUTA_INTERMEDIOS / "prototipos_tono.csv"


def cargar_prendas():
    """Dataset restringido al alcance textil. Retorna (ok, df con bytes, mensaje)."""
    ok, df, msg = cargar_dataset()
    if not ok:
        return ok, df, msg
    df = df[df["masterCategory"].isin(config.ALCANCE_MASTER)].reset_index(drop=True)
    return True, df, f"{len(df):,} prendas en el alcance textil ({', '.join(config.ALCANCE_MASTER)})"


def construir_tabla(df, mapas):
    """Tabla por prenda: metadata + objetivo continuo + multicolor + coherencia + split + rol."""
    obj = medir_dataset(df, mapas)
    meta = df.drop(columns="imagen_bytes").set_index("id")
    tabla = meta.join(obj)
    splits = pd.read_csv(config.RUTA_INTERMEDIOS / "splits.csv", index_col=0)
    tabla = tabla.join(splits[["split"]])
    tabla["tiene_objetivo"] = tabla["L_obj"].notna()
    # Prototipos solo con prendas de un color dominante (las multicolor desplazan la mediana)
    base = tabla["tiene_objetivo"] & ~tabla["multicolor"].fillna(False) & (tabla["split"] == "train")
    proto = prototipos_por_tono(tabla[base], tabla.loc[base, "baseColour"])
    proto["grupo"] = proto.index.map(config.TONO_A_GRUPO)
    tabla["coherente"] = False
    con = tabla["tiene_objetivo"]
    tabla.loc[con, "coherente"] = coherencia_con_etiqueta(tabla[con], tabla.loc[con, "baseColour"], proto)
    excluir = tabla["grupo_color"].isin(["Multi", "Metálico"])
    tabla["uso_entrenamiento"] = tabla["tiene_objetivo"] & tabla["coherente"] & ~tabla["multicolor"].fillna(False) & ~excluir
    tabla["uso_evaluacion"] = tabla["tiene_objetivo"] & ~excluir
    tabla.to_parquet(RUTA_TABLA)
    proto.to_csv(RUTA_PROTOTIPOS)
    return tabla, proto


def mascaras_dominantes(df, mapas, ids, ruta=None):
    """Máscara del color dominante y ganancias de balance de blancos de cada prenda (caché .npz)."""
    from core.datos import decodificar
    from core.objetivo_color import mascara_dominante
    ruta = ruta or config.RUTA_INTERMEDIOS / "mascaras_dominantes.npz"
    if ruta.exists():
        d = np.load(ruta)
        return dict(zip(d["ids"].tolist(), d["mascaras"])), dict(zip(d["ids"].tolist(), d["ganancias"]))
    fila = df.set_index("id")
    mascaras, ganancias = [], []
    for k, pid in enumerate(ids):
        m, g = mascara_dominante(decodificar(fila.at[pid, "imagen_bytes"]), mapas[pid], fila.at[pid, "subCategory"])
        if m.shape != (80, 60):   # 8 de 21.361 imágenes no miden 80x60: se llevan al tamaño nativo común
            m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((60, 80), Image.NEAREST)) > 127
        mascaras.append(m); ganancias.append(g)
        if k % 3000 == 0:
            print(f"  máscaras dominantes {k:,}/{len(ids):,}", flush=True)
    np.savez_compressed(ruta, ids=np.array(ids), mascaras=np.array(mascaras), ganancias=np.array(ganancias))
    return dict(zip(ids, mascaras)), dict(zip(ids, ganancias))


def cargar_tabla():
    tabla = pd.read_parquet(RUTA_TABLA)
    proto = pd.read_csv(RUTA_PROTOTIPOS, index_col=0)
    return tabla, proto


def generar_test_real(df, mapas, ids_test, repeticiones=1):
    """Versión 'foto de cliente' de cada imagen de test (fondo, luz y compresión aleatorios)."""
    if RUTA_TEST_REAL.exists():
        return pd.read_parquet(RUTA_TEST_REAL)
    bytes_por_id = df.set_index("id")["imagen_bytes"]
    filas = []
    for rep in range(repeticiones):
        for pid in ids_test:
            rgb, params = simular_foto_real(decodificar(bytes_por_id[pid]), mapas[pid], semilla=int(pid) * 10 + rep)
            buf = io.BytesIO(); Image.fromarray(rgb).save(buf, "PNG")
            filas.append({"id": pid, "rep": rep, "imagen_bytes": buf.getvalue(), **params})
    res = pd.DataFrame(filas)
    res.to_parquet(RUTA_TEST_REAL)
    return res
