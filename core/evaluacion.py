"""Métricas de evaluación alineadas con el objetivo de negocio.

- Error perceptual ΔE2000 entre el color predicho y el medido (objetivo continuo).
- Nombre de tono y grupo básico derivados del color predicho (prototipo más cercano).
- Recuperación en el catálogo de colorantes: ¿la referencia correcta queda en el top-k sugerido?
  Como las referencias vecinas pueden estar a menos de 1 ΔE (indistinguibles a la vista), también
  se mide un acierto "tolerante" y la distancia perceptual de la sugerencia al color real.
- Nombre desde las cabezas de clasificación de la red (grupo y tono), que no dependen de los
  prototipos.
"""

import numpy as np
import pandas as pd
from skimage import color as skcolor
from sklearn.cluster import KMeans

from core import config


def delta_e(pred, real):
    return skcolor.deltaE_ciede2000(np.asarray(pred, float), np.asarray(real, float))


def resumen_delta_e(de):
    de = np.asarray(de)
    return {"dE_media": de.mean(), "dE_mediana": np.median(de), "dE_p90": np.percentile(de, 90),
            "pct_dE<5": (de < 5).mean(), "pct_dE<10": (de < 10).mean()}


def nombres_desde_lab(pred_lab, proto, k=3):
    """Índices de los k prototipos de tono más cercanos (ΔE2000) a cada color predicho."""
    p = proto[["L", "a", "b"]].to_numpy(float)
    d = skcolor.deltaE_ciede2000(np.asarray(pred_lab, float)[:, None, :], p[None, :, :])
    return proto.index.to_numpy()[np.argsort(d, axis=1)[:, :k]]


def metricas_nombre(pred_lab, tono_real, proto):
    """Accuracy de grupo básico y top-1/top-3 de tono a partir del color predicho."""
    top = nombres_desde_lab(pred_lab, proto, k=3)
    tono_real = np.asarray(tono_real)
    grupo = np.vectorize(config.TONO_A_GRUPO.get)
    return {"acc_grupo": (grupo(top[:, 0]) == grupo(tono_real)).mean(),
            "top1_tono": (top[:, 0] == tono_real).mean(),
            "top3_tono": (top == tono_real[:, None]).any(axis=1).mean()}


def catalogo_simulado(lab_train, n=config.N_CATALOGO_SIMULADO):
    """Catálogo de referencias. Usa el CSV real de la empresa si existe; si no, simula `n`
    referencias que cubren el gamut de colores textiles observado (centros de k-means)."""
    if config.RUTA_CATALOGO_REAL.exists():
        cat = pd.read_csv(config.RUTA_CATALOGO_REAL)
        return cat.set_index("referencia")[["L", "a", "b"]], "real"
    km = KMeans(n_clusters=n, n_init=4, random_state=config.SEMILLA).fit(np.asarray(lab_train, float))
    cat = pd.DataFrame(km.cluster_centers_, columns=["L", "a", "b"],
                       index=[f"REF-{i + 1:03d}" for i in range(n)])
    cat.index.name = "referencia"
    return cat, "simulado"


TOLERANCIA_REF = 1.0   # ΔE2000: referencias a menos de esto de la mejor se consideran equivalentes


def metricas_catalogo(pred_lab, real_lab, catalogo, ks=(1, 3, 5)):
    """La referencia 'correcta' es la más cercana al color real. Se mide si aparece en el top-k
    de referencias sugeridas a partir del color predicho, y el ΔE de la primera sugerencia.

    hit@3_tol: alguna de las 3 sugeridas está a ≤ TOLERANCIA_REF de la mejor posible (equivalente
    a la vista). sug_dE<=3 / <=5: la primera sugerencia queda a ese ΔE del color real (tolerancias
    usuales de aceptación comercial: ~1-2 estricta, ~3-5 para una primera propuesta de laboratorio)."""
    c = catalogo[["L", "a", "b"]].to_numpy(float)
    d_pred = skcolor.deltaE_ciede2000(np.asarray(pred_lab, float)[:, None, :], c[None])
    d_real = skcolor.deltaE_ciede2000(np.asarray(real_lab, float)[:, None, :], c[None])
    correcta = d_real.argmin(axis=1)
    orden = np.argsort(d_pred, axis=1)
    res = {f"hit@{k}": (orden[:, :k] == correcta[:, None]).any(axis=1).mean() for k in ks}
    # ΔE entre el color real y la primera referencia sugerida (lo que vería el laboratorio)
    de_sug = d_real[np.arange(len(correcta)), orden[:, 0]]
    res["dE_sugerida"] = np.mean(de_sug)
    res["dE_sugerida_mediana"] = np.median(de_sug)
    mejor = d_real.min(axis=1, keepdims=True)
    equivalentes = d_real <= mejor + TOLERANCIA_REF
    res["hit@3_tol"] = np.take_along_axis(equivalentes, orden[:, :3], axis=1).any(axis=1).mean()
    res["sug_dE<=3"] = (de_sug <= 3).mean()
    res["sug_dE<=5"] = (de_sug <= 5).mean()
    return res


def metricas_cabeza(nombres, grupo_real, tono_real):
    """Métricas de nombre desde las cabezas de clasificación (DataFrame con grupo_1, grupo_2 y,
    si existe la cabeza de tonos, tono_1..tono_3), alineado con grupo_real / tono_real."""
    from sklearn.metrics import f1_score
    g, t = np.asarray(grupo_real), np.asarray(tono_real)
    res = {"acc_grupo": (nombres["grupo_1"].to_numpy() == g).mean(),
           "f1_grupo": f1_score(g, nombres["grupo_1"], average="macro"),
           "top2_grupo": ((nombres["grupo_1"].to_numpy() == g) | (nombres["grupo_2"].to_numpy() == g)).mean()}
    if "tono_1" in nombres and nombres["tono_1"].notna().all():
        top = nombres[["tono_1", "tono_2", "tono_3"]].to_numpy()
        res["top1_tono"] = (top[:, 0] == t).mean()
        res["top3_tono"] = (top == t[:, None]).any(axis=1).mean()
    return res


def tabla_resultados(filas):
    return pd.DataFrame(filas).set_index("modelo")
