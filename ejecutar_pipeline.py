"""Ejecuta los cálculos pesados del proyecto (con caché en cada paso).

Uso:  C:\\envs\\vision_por_computador\\Scripts\\python.exe -u ejecutar_pipeline.py

1. Segmentación de prendas (SegFormer)          -> data/intermedios/segmentacion.npz
2. Objetivo continuo, multicolor, coherencia     -> data/intermedios/tabla_prendas.parquet
3. Conjunto de prueba "contexto real"            -> data/intermedios/test_contexto_real.parquet
4. Características (histograma Lab, ResNet-50, DINOv2, CLIP) -> data/intermedios/caracteristicas/
5. Modelos y predicciones de todos los enfoques  -> resultados/predicciones.parquet,
   resultados/predicciones_nombre.parquet (grupo/tono de las cabezas de clasificación), modelos/

Los notebooks 02-05 leen estos resultados; si falta alguno, este script lo regenera.
"""

import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
torch.set_num_threads(10)

from core import config
from core.caracteristicas import caracteristicas_cache, clip_texto
from core.color import describir_imagen
from core.datos import decodificar
from core.modelos import (DISPOSITIVO, DatasetColor, RegresorLab, cargar_red, entrenar_red, predecir_red)
from core.objetivo_color import balance_gris, medir_color_prenda
from core.hibrida import DatasetHibrido, entrenar_hibrida, predecir_hibrida
from core.preparacion import cargar_prendas, construir_tabla, generar_test_real, mascaras_dominantes
from core.segmentacion import segmentar, segmentar_dataset

RUTA_PRED = config.RUTA_RESULTADOS / "predicciones.parquet"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def guardar_pred(nuevas):
    """Agrega predicciones (modelo, conjunto, id, L, a, b) reemplazando las del mismo modelo/conjunto."""
    config.RUTA_RESULTADOS.mkdir(exist_ok=True)
    if RUTA_PRED.exists():
        viejas = pd.read_parquet(RUTA_PRED)
        claves = set(zip(nuevas["modelo"], nuevas["conjunto"]))
        viejas = viejas[[(m, c) not in claves for m, c in zip(viejas["modelo"], viejas["conjunto"])]]
        nuevas = pd.concat([viejas, nuevas], ignore_index=True)
    nuevas.to_parquet(RUTA_PRED)


def ya_hay(modelo, conjunto):
    if not RUTA_PRED.exists():
        return False
    p = pd.read_parquet(RUTA_PRED, columns=["modelo", "conjunto"])
    return ((p["modelo"] == modelo) & (p["conjunto"] == conjunto)).any()


def pred_df(modelo, conjunto, ids, lab):
    lab = np.asarray(lab, float)
    return pd.DataFrame({"modelo": modelo, "conjunto": conjunto, "id": ids, "L": lab[:, 0], "a": lab[:, 1], "b": lab[:, 2]})


def main():
    ok, df, msg = cargar_prendas()
    if not ok:
        sys.exit(msg)
    log(msg)
    from core.modelos import DISPOSITIVO
    log(f"Dispositivo de cómputo: {DISPOSITIVO}" + (f" ({torch.cuda.get_device_name(0)})" if DISPOSITIVO.type == "cuda" else ""))
    bytes_por_id = df.set_index("id")["imagen_bytes"]

    log("1. Segmentación de prendas")
    mapas = segmentar_dataset(df)

    log("2. Objetivo continuo y tabla de prendas")
    tabla, proto = construir_tabla(df, mapas)
    ids_tr = tabla.index[(tabla["split"] == "train") & tabla["uso_entrenamiento"]].to_numpy()
    ids_va = tabla.index[(tabla["split"] == "val") & tabla["uso_entrenamiento"]].to_numpy()
    ids_te = tabla.index[(tabla["split"] == "test") & tabla["uso_evaluacion"]].to_numpy()
    log(f"   train {len(ids_tr):,} · val {len(ids_va):,} · test {len(ids_te):,}")
    lab_cols = ["L_obj", "a_obj", "b_obj"]

    log("3. Conjunto de prueba 'contexto real'")
    real = generar_test_real(df, mapas, ids_te)
    real_rgb = {pid: decodificar(b) for pid, b in zip(real["id"], real["imagen_bytes"])}
    ruta_seg_real = config.RUTA_INTERMEDIOS / "segmentacion_test_real.npz"
    if ruta_seg_real.exists():
        d = np.load(ruta_seg_real, allow_pickle=True); mapas_real = dict(zip(d["ids"].tolist(), d["mapas"]))
    else:
        ids = list(real_rgb); mapas_real = {}
        for i in range(0, len(ids), 32):
            for pid, m in zip(ids[i:i + 32], segmentar([real_rgb[p] for p in ids[i:i + 32]])):
                mapas_real[pid] = m
        np.savez_compressed(ruta_seg_real, ids=np.array(ids), mapas=np.array([mapas_real[p] for p in ids], dtype=object))

    conjuntos = {
        "train": (ids_tr, lambda pid: decodificar(bytes_por_id[pid])),
        "val": (ids_va, lambda pid: decodificar(bytes_por_id[pid])),
        "test": (ids_te, lambda pid: decodificar(bytes_por_id[pid])),
        "test_real": (ids_te, lambda pid: real_rgb[pid]),
    }

    log("5a. Enfoques sin aprendizaje (medición directa)")
    if not ya_hay("E1_umbral_mediana", "test_real"):
        for conj in ["test", "test_real"]:
            ids, fn = conjuntos[conj]
            lab = [[r["L"], r["a"], r["b"]] for r in (describir_imagen(_png(fn(p))) for p in ids)]
            guardar_pred(pred_df("E1_umbral_mediana", conj, ids, lab))
    if not ya_hay("segmentacion_kmeans", "test_real"):
        for nombre, wb in [("segmentacion_kmeans", False), ("segmentacion_kmeans_grayworld", True)]:
            labs = []
            for p in ids_te:
                rgb = balance_gris(real_rgb[p]) if wb else real_rgb[p]
                o = medir_color_prenda(rgb, mapas_real[p], tabla.at[p, "subCategory"], corregir_blancos=False)
                labs.append([o.get("L_obj", np.nan), o.get("a_obj", np.nan), o.get("b_obj", np.nan)])
            guardar_pred(pred_df(nombre, "test_real", ids_te, labs))

    log("4-5b. Características + regresión ridge")
    y_tr = tabla.loc[ids_tr, lab_cols].to_numpy()
    for nombre in ["hist_lab", "resnet50", "dinov2_s", "clip_b32"]:
        modelo = f"ridge_{nombre}"
        if ya_hay(modelo, "test_real"):
            continue
        X = {}
        for conj, (ids, fn) in conjuntos.items():
            t0 = time.time()
            X[conj] = caracteristicas_cache(nombre, conj, lambda i, ids=ids, fn=fn: fn(ids[i]), len(ids))
            log(f"   {nombre} {conj}: {X[conj].shape} ({time.time() - t0:.0f} s)")
        reg = RegresorLab().fit(X["train"], y_tr)
        for conj in ["val", "test", "test_real"]:
            guardar_pred(pred_df(modelo, conj, conjuntos[conj][0], reg.predict(X[conj])))

    log("5c. CLIP zero-shot (nombre de tono -> prototipo)")
    if not ya_hay("clip_zeroshot", "test_real"):
        tonos = proto.index.tolist()
        T = clip_texto([f"a photo of a {t.lower()} colored garment" for t in tonos])
        T /= np.linalg.norm(T, axis=1, keepdims=True)
        for conj in ["test", "test_real"]:
            Xc = np.load(config.RUTA_INTERMEDIOS / "caracteristicas" / f"clip_b32_{conj}.npy")
            Xc = Xc / np.linalg.norm(Xc, axis=1, keepdims=True)
            elegido = (Xc @ T.T).argmax(1)
            guardar_pred(pred_df("clip_zeroshot", conj, conjuntos[conj][0], proto.iloc[elegido][["L", "a", "b"]].to_numpy()))

    entrenar = "--sin-entrenar" not in sys.argv
    if not entrenar:
        log("--sin-entrenar: solo se usan las redes ya entrenadas (para entrenar, correr sin esa opción en GPU).")

    log("5d. Fine-tuning MobileNetV3 (E8 estándar, E9 contexto real, E10 red completa + tonos)")
    grupos = sorted(tabla.loc[ids_tr, "grupo_color"].unique())
    g_idx = {g: i for i, g in enumerate(grupos)}
    tonos = sorted(tabla.loc[ids_tr, "baseColour"].unique())
    t_idx = {t: i for i, t in enumerate(tonos)}
    (config.RUTA_MODELOS).mkdir(exist_ok=True)
    (config.RUTA_MODELOS / "grupos.json").write_text(json.dumps(grupos, ensure_ascii=False))
    (config.RUTA_MODELOS / "tonos.json").write_text(json.dumps(tonos, ensure_ascii=False))

    def ds(ids, **kw):
        return DatasetColor([bytes_por_id[p] for p in ids], tabla.loc[ids, lab_cols].to_numpy(),
                            [g_idx.get(g, 0) for g in tabla.loc[ids, "grupo_color"]],
                            tonos=[t_idx.get(t, 0) for t in tabla.loc[ids, "baseColour"]],
                            mapas=[mapas[p] for p in ids], **kw)

    ds_real = DatasetColor([_png(real_rgb[p]) for p in ids_te], tabla.loc[ids_te, lab_cols].to_numpy(), [0] * len(ids_te))

    # E11 (híbrida): máscara del color dominante y ganancias de balance de blancos de train/val (caché)
    log("   máscaras del color dominante (supervisión de E11)")
    mascaras_dom, ganancias_wb = mascaras_dominantes(df, mapas, list(ids_tr) + list(ids_va))

    def ds_h(ids, **kw):
        return DatasetHibrido([bytes_por_id[p] for p in ids], tabla.loc[ids, lab_cols].to_numpy(),
                              [g_idx.get(g, 0) for g in tabla.loc[ids, "grupo_color"]],
                              [t_idx.get(t, 0) for t in tabla.loc[ids, "baseColour"]],
                              mapas=[mapas[p] for p in ids],
                              mascaras=[mascaras_dom[p] for p in ids] if p_en(ids, mascaras_dom) else None,
                              ganancias=[ganancias_wb[p] for p in ids] if p_en(ids, ganancias_wb) else None, **kw)

    for nombre, cfg in REDES.items():
        hibrida = cfg.get("tipo") == "hibrida"
        if ya_hay(nombre, "test_real") and ya_hay_nombres(nombre, "test_real"):
            continue
        ruta = config.RUTA_MODELOS / f"{nombre}.pt"
        if ruta.exists():
            red = cargar_red(ruta, len(grupos)).to(DISPOSITIVO)
        elif entrenar and hibrida:
            log(f"   entrenando {nombre}: {cfg}")
            red, hist = entrenar_hibrida(ds_h(ids_tr, entrenamiento=True, contexto_real=cfg["p_real"], semilla=1),
                                         ds_h(ids_va, contexto_real=1.0, semilla=7),   # validación en contexto real fijo
                                         len(grupos), len(tonos), epocas=cfg["epocas"], ruta=ruta, log=log)
            pd.DataFrame(hist).to_csv(config.RUTA_RESULTADOS / f"historia_{nombre}.csv", index=False)
        elif entrenar:
            log(f"   entrenando {nombre}: {cfg}")
            red, hist = entrenar_red(ds(ids_tr, entrenamiento=True, contexto_real=cfg["p_real"], semilla=1), ds(ids_va),
                                     len(grupos), epocas=cfg["epocas"], ruta=ruta, log=log,
                                     n_tonos=len(tonos) if cfg.get("tonos") else 0,
                                     congelados=cfg.get("congelados", 7), tam=cfg.get("tam"),
                                     peso_lab=cfg.get("peso_lab", (1.0, 1.0, 1.0)))
            pd.DataFrame(hist).to_csv(config.RUTA_RESULTADOS / f"historia_{nombre}.csv", index=False)
        else:
            log(f"   {nombre}: sin entrenar todavía, se omite")
            continue
        if hibrida:
            pruebas = [("test", ds_h(ids_te)),
                       ("test_real", DatasetHibrido(ds_real.bytes_, tabla.loc[ids_te, lab_cols].to_numpy()))]
        else:
            pruebas = [("test", ds(ids_te)), ("test_real", ds_real)]
        for conj, d in pruebas:
            sal = (predecir_hibrida if hibrida else predecir_red)(red, d, tta=cfg.get("tta", False))
            guardar_pred(pred_df(nombre, conj, ids_te, sal["lab"]))
            guardar_nombres(nombre, conj, ids_te, sal, grupos, tonos)
        log(f"   {nombre}: predicciones y nombres guardados")
    log("Listo.")


# Redes que se ajustan (fine-tuning). E8 y E9 se entrenaron en CPU (7 bloques congelados, 6 épocas);
# E10 aprovecha la GPU: toda la red, mayor resolución, más épocas, cabeza de 46 tonos, más peso a L*
# en la pérdida y predicción promediada con la imagen reflejada (TTA).
REDES = {
    "mobilenet_estandar": {"p_real": 0.0, "epocas": 6},
    "mobilenet_contexto_real": {"p_real": 0.6, "epocas": 6},
    "mobilenet_e10": {"p_real": 0.7, "epocas": 15, "congelados": 0, "tam": (224, 168), "tonos": True,
                      "peso_lab": (1.5, 1.0, 1.0), "tta": True},
    # E11: red híbrida "medir en vez de regresar" (core/hibrida.py), propuesta tras el análisis del error de E10
    "hibrida_e11": {"tipo": "hibrida", "p_real": 0.8, "epocas": 40, "tta": True},
}


def p_en(ids, dic):
    return all(p in dic for p in ids)
RUTA_NOMBRES = config.RUTA_RESULTADOS / "predicciones_nombre.parquet"


def ya_hay_nombres(modelo, conjunto):
    if not RUTA_NOMBRES.exists():
        return False
    p = pd.read_parquet(RUTA_NOMBRES, columns=["modelo", "conjunto"])
    return ((p["modelo"] == modelo) & (p["conjunto"] == conjunto)).any()


def guardar_nombres(modelo, conjunto, ids, salida, grupos, tonos):
    """Nombre predicho por las cabezas de clasificación: grupo (top-2) y tono (top-3, si hay cabeza)."""
    og = np.argsort(-salida["grupo"], axis=1)
    df = pd.DataFrame({"modelo": modelo, "conjunto": conjunto, "id": ids,
                       "grupo_1": np.array(grupos)[og[:, 0]], "grupo_2": np.array(grupos)[og[:, 1]]})
    if salida["tono"] is not None:
        ot = np.argsort(-salida["tono"], axis=1)
        for k in range(3):
            df[f"tono_{k + 1}"] = np.array(tonos)[ot[:, k]]
    if RUTA_NOMBRES.exists():
        viejas = pd.read_parquet(RUTA_NOMBRES)
        viejas = viejas[~((viejas["modelo"] == modelo) & (viejas["conjunto"] == conjunto))]
        df = pd.concat([viejas, df], ignore_index=True)
    df.to_parquet(RUTA_NOMBRES)


def _png(rgb):
    import io
    from PIL import Image
    buf = io.BytesIO(); Image.fromarray(rgb).save(buf, "PNG")
    return buf.getvalue()


if __name__ == "__main__":
    main()
