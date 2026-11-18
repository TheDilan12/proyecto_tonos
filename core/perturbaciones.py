"""Simulación del contexto real: fotos de requerimientos sin fondo blanco de estudio.

Las fotos que llegan al área comercial no son de catálogo: tienen fondos arbitrarios,
luz cálida o fría, distinta exposición y compresión de mensajería. Este módulo genera
esas condiciones sobre las imágenes de prueba para medir la robustez de cada modelo.
El objetivo de color NO cambia: sigue siendo el color medido bajo la luz de estudio.
"""

import io

import numpy as np
from PIL import Image, ImageFilter
from skimage import color as skcolor

from core import config
from core.objetivo_color import lineal_a_srgb, srgb_a_lineal
from core.segmentacion import mascara_persona


def rgb_de_temperatura(kelvin):
    """Color (RGB lineal, normalizado a 6500 K) de un iluminante de cuerpo negro (aprox. de Tanner Helland)."""
    t = kelvin / 100.0
    r = 255.0 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
    g = 99.4708025861 * np.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * (t - 60) ** -0.0755148492
    b = 255.0 if t >= 66 else (0.0 if t <= 19 else 138.5177312231 * np.log(t - 10) - 305.0447927307)
    rgb = srgb_a_lineal(np.clip(np.array([r, g, b]), 0, 255) / 255.0)
    ref = srgb_a_lineal(np.array([255.0, 254.0, 250.0]) / 255.0)  # ≈ 6500 K
    return rgb / ref


def fondo_aleatorio(alto, ancho, rng):
    """Fondo de color sólido, degradado o textura suave (como una pared, tela o mueble)."""
    tipo = rng.choice(["solido", "degradado", "textura"])
    def color():
        lab = np.array([rng.uniform(25, 90), rng.uniform(-30, 30), rng.uniform(-30, 40)])
        return np.clip(skcolor.lab2rgb(lab[None, None, :])[0, 0], 0, 1)
    if tipo == "solido":
        img = np.ones((alto, ancho, 3)) * color()
    elif tipo == "degradado":
        t = np.linspace(0, 1, alto)[:, None, None]
        img = (1 - t) * color() + t * color()
        img = np.broadcast_to(img, (alto, ancho, 3)).copy()
    else:
        ruido = rng.normal(size=(alto // 4 + 1, ancho // 4 + 1, 1))
        ruido = np.asarray(Image.fromarray(((ruido[..., 0] - ruido.min()) / np.ptp(ruido) * 255).astype(np.uint8))
                           .resize((ancho, alto), Image.BICUBIC), dtype=float)[..., None] / 255
        img = (1 - ruido) * color() + ruido * color()
    return img, tipo


def simular_foto_real(rgb, mapa_seg, semilla, cambiar_fondo=True, cambiar_luz=True, comprimir=True):
    """Aplica fondo, iluminante, exposición y compresión aleatorios. Retorna (rgb uint8, parámetros)."""
    rng = np.random.default_rng(semilla)
    alto, ancho = rgb.shape[:2]
    img = rgb.astype(float) / 255.0
    params = {}
    if cambiar_fondo:
        alfa = Image.fromarray((mascara_persona(mapa_seg) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.7))
        alfa = np.asarray(alfa, dtype=float)[..., None] / 255
        fondo, params["fondo"] = fondo_aleatorio(alto, ancho, rng)
        img = alfa * img + (1 - alfa) * fondo
    if cambiar_luz:
        kelvin = rng.uniform(*config.TEMPERATURAS_LUZ)
        brillo = rng.uniform(*config.FACTOR_BRILLO)
        img = lineal_a_srgb(srgb_a_lineal(img) * rgb_de_temperatura(kelvin) * brillo)
        params.update(kelvin=round(kelvin), brillo=round(brillo, 2), kelvin_exacto=kelvin, brillo_exacto=brillo)
    salida = (np.clip(img, 0, 1) * 255).round().astype(np.uint8)
    if comprimir:
        calidad = int(rng.integers(*config.CALIDAD_JPEG))
        buf = io.BytesIO(); Image.fromarray(salida).save(buf, "JPEG", quality=calidad)
        salida = np.asarray(Image.open(buf).convert("RGB"))
        params["jpeg"] = calidad
    return salida, params
