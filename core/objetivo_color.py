"""Del nombre discreto al color continuo: medición CIELAB de la prenda, multicolor y coherencia.

Para cada imagen:
1. balance de blancos usando el fondo como referencia neutra (constancia de color);
2. píxeles de la prenda (segmentación) -> k-means en CIELAB -> color dominante y paleta;
3. métricas de multicolor (participación y distancia del segundo color);
4. la etiqueta discreta (baseColour) define un prototipo por tono; el color medido se
   acepta como objetivo continuo solo si es coherente con su etiqueta.
"""

import numpy as np
import pandas as pd
from skimage import color as skcolor
from sklearn.cluster import KMeans

from core import config
from core.color import mascara_primer_plano
from core.segmentacion import mascara_prenda

# Píxeles de fondo con L* mayor a esto sirven como referencia de blanco.
L_MINIMO_REFERENCIA = 70.0


def srgb_a_lineal(rgb01):
    return np.where(rgb01 <= 0.04045, rgb01 / 12.92, ((rgb01 + 0.055) / 1.055) ** 2.4)


def lineal_a_srgb(lin):
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)


def balance_blancos(rgb, referencia):
    """Von Kries: escala cada canal (en RGB lineal) para que los píxeles de `referencia` queden neutros.

    Retorna (rgb_corregido uint8, ganancias por canal). Si no hay suficiente referencia, no corrige.
    """
    rgb01 = rgb.astype(np.float32) / 255.0
    if referencia.sum() < 0.05 * referencia.size:
        return rgb, np.ones(3, dtype=np.float32)
    lin = srgb_a_lineal(rgb01)
    blanco = lin[referencia].mean(axis=0)
    ganancias = blanco.mean() / np.maximum(blanco, 1e-4)
    corregido = lineal_a_srgb(lin * ganancias)
    return (corregido * 255).round().astype(np.uint8), ganancias.astype(np.float32)


def balance_gris(rgb):
    """Gray-world: supone que el promedio de la escena es gris (no requiere fondo blanco)."""
    lin = srgb_a_lineal(rgb.astype(np.float32) / 255.0)
    media = lin.reshape(-1, 3).mean(axis=0)
    corregido = lineal_a_srgb(lin * (media.mean() / np.maximum(media, 1e-4)))
    return (corregido * 255).round().astype(np.uint8)


def paleta_kmeans(lab_pixeles, k=config.K_COLORES, semilla=config.SEMILLA, max_pixeles=1500):
    """k-means en CIELAB. Retorna (centros Lab ordenados por participación, participaciones, etiquetas)."""
    n = len(lab_pixeles)
    if n == 0:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0, dtype=int)
    k = int(min(k, max(1, n // 20)))
    rng = np.random.default_rng(semilla)
    muestra = lab_pixeles if n <= max_pixeles else lab_pixeles[rng.choice(n, max_pixeles, replace=False)]
    km = KMeans(n_clusters=k, n_init=3, random_state=semilla).fit(muestra)
    etiquetas = km.predict(lab_pixeles)
    participacion = np.bincount(etiquetas, minlength=k) / n
    orden = np.argsort(-participacion)
    # Centro robusto: mediana de los píxeles de cada grupo
    centros = np.array([np.median(lab_pixeles[etiquetas == c], axis=0) for c in orden])
    remapeo = np.empty(k, dtype=int); remapeo[orden] = np.arange(k)
    return centros, participacion[orden], remapeo[etiquetas]


def medir_color_prenda(rgb, mapa_seg, subcategoria, corregir_blancos=True):
    """Mide el color de la prenda de una imagen. Retorna un dict con el objetivo continuo y la paleta."""
    mascara = mascara_prenda(mapa_seg, subcategoria)
    fuente = "segmentacion"
    if mascara.mean() < config.FRACCION_MINIMA_PRENDA:
        # Prenda fotografiada sola (sin modelo): el parser no la reconoce; se usa ΔE contra el fondo.
        mascara, _, _ = mascara_primer_plano(rgb)
        fuente = "umbral_fondo"
    referencia = (mapa_seg == 0)
    lab_original = skcolor.rgb2lab(rgb)
    referencia &= lab_original[..., 0] > L_MINIMO_REFERENCIA
    lab_fondo = lab_original[referencia].mean(axis=0) if referencia.any() else np.array([np.nan] * 3)
    if corregir_blancos:
        rgb, ganancias = balance_blancos(rgb, referencia)
    else:
        ganancias = np.ones(3, dtype=np.float32)
    lab = skcolor.rgb2lab(rgb)
    pix = lab[mascara]
    centros, part, etiquetas = paleta_kmeans(pix)
    if len(centros) == 0:
        return {"fuente_mascara": "vacia", "frac_prenda": 0.0}
    dominante = centros[0]
    pix_dom = pix[etiquetas == 0]
    d12 = float(skcolor.deltaE_ciede2000(centros[0], centros[1])) if len(centros) > 1 else 0.0
    return {
        "fuente_mascara": fuente,
        "frac_prenda": float(mascara.mean()),
        "L_obj": dominante[0], "a_obj": dominante[1], "b_obj": dominante[2],
        "C_obj": float(np.hypot(dominante[1], dominante[2])),
        "part_1": float(part[0]), "part_2": float(part[1]) if len(part) > 1 else 0.0,
        "dE_12": d12,
        "multicolor": bool(len(part) > 1 and part[1] >= config.MULTI_FRACCION_SEGUNDO
                           and d12 >= config.MULTI_DELTA_E),
        "paleta": centros.round(2).tolist(), "paleta_part": part.round(3).tolist(),
        "L_std_prenda": float(pix_dom[:, 0].std()),       # sombras / pliegues / textura
        "croma_fondo": float(np.hypot(lab_fondo[1], lab_fondo[2])),  # dominante de color de la luz
        "L_fondo": float(lab_fondo[0]),
        "ganancia_r": float(ganancias[0]), "ganancia_b": float(ganancias[2]),
        "frac_piel": float(np.isin(mapa_seg, config.CLASES_PIEL_CABELLO).mean()),
    }


def mascara_dominante(rgb, mapa_seg, subcategoria):
    """Máscara (HxW) de los píxeles del color dominante y ganancias de balance de blancos, con el mismo
    procedimiento que `medir_color_prenda` (supervisión de la red híbrida E11)."""
    mascara = mascara_prenda(mapa_seg, subcategoria)
    if mascara.mean() < config.FRACCION_MINIMA_PRENDA:
        mascara, _, _ = mascara_primer_plano(rgb)
    referencia = (mapa_seg == 0) & (skcolor.rgb2lab(rgb)[..., 0] > L_MINIMO_REFERENCIA)
    rgb_wb, ganancias = balance_blancos(rgb, referencia)
    pix = skcolor.rgb2lab(rgb_wb)[mascara]
    dominante = np.zeros(mascara.shape, bool)
    if len(pix):
        _, _, etiquetas = paleta_kmeans(pix)
        dominante[mascara] = etiquetas == 0
    return dominante, ganancias


def medir_dataset(df, mapas, ruta_cache=config.RUTA_INTERMEDIOS / "objetivo_color.parquet"):
    """Aplica `medir_color_prenda` a todo `df` (con caché). Retorna DataFrame indexado por id."""
    if ruta_cache.exists():
        return pd.read_parquet(ruta_cache)
    from core.datos import decodificar
    filas = []
    for i, (pid, sub, b) in enumerate(zip(df["id"], df["subCategory"], df["imagen_bytes"])):
        fila = medir_color_prenda(decodificar(b), mapas[pid], sub)
        fila["id"] = pid
        filas.append(fila)
        if i % 5000 == 0:
            print(f"  medidas {i:,}/{len(df):,}", flush=True)
    res = pd.DataFrame(filas).set_index("id")
    ruta_cache.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(ruta_cache)
    return res


def prototipos_por_tono(obj, etiquetas, filtro=None):
    """Prototipo continuo de cada nombre de tono: mediana Lab del color medido (y su dispersión)."""
    datos = obj.assign(tono=etiquetas)
    if filtro is not None:
        datos = datos[filtro]
    grupo = datos.groupby("tono")
    proto = grupo[["L_obj", "a_obj", "b_obj"]].median()
    proto.columns = ["L", "a", "b"]
    proto["n"] = grupo.size()
    return proto


def distancias_a_prototipos(lab, proto):
    """Matriz (n, n_tonos) de ΔE2000 entre colores Lab y los prototipos."""
    lab = np.asarray(lab, dtype=float)
    p = proto[["L", "a", "b"]].to_numpy(dtype=float)
    return skcolor.deltaE_ciede2000(lab[:, None, :], p[None, :, :])


def coherencia_con_etiqueta(obj, etiquetas, proto, top_n=config.COHERENCIA_TOP_N):
    """True si el prototipo de la etiqueta está entre los `top_n` más cercanos al color medido."""
    d = distancias_a_prototipos(obj[["L_obj", "a_obj", "b_obj"]].to_numpy(), proto)
    rango = (d.argsort(axis=1).argsort(axis=1))
    col = proto.index.get_indexer(etiquetas)
    validos = col >= 0
    res = np.zeros(len(obj), dtype=bool)
    res[validos] = rango[np.where(validos)[0], col[validos]] < top_n
    return res
