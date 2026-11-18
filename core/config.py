"""Configuración centralizada del proyecto: rutas, catálogos y parámetros."""

from pathlib import Path

# --- Rutas locales ---------------------------------------------------------
RUTA_CORE = Path(__file__).resolve().parent
RAIZ_PROYECTO = RUTA_CORE.parent

RUTA_DATOS = RAIZ_PROYECTO / "data"
RUTA_PARQUET = RUTA_DATOS / "raw" / "fashion_small" / "data"
RUTA_INTERMEDIOS = RUTA_DATOS / "intermedios"
RUTA_FIGURAS = RAIZ_PROYECTO / "figuras"
RUTA_MODELOS = RAIZ_PROYECTO / "modelos"
RUTA_RESULTADOS = RAIZ_PROYECTO / "resultados"
# Catálogo de colorantes: si la empresa entrega un CSV con columnas referencia,L,a,b
# en esta ruta, la evaluación lo usa en vez del catálogo simulado.
RUTA_CATALOGO_REAL = RUTA_DATOS / "catalogo_colorantes.csv"

# --- Fuentes externas ------------------------------------------------------
HF_DATASET = "ashraq/fashion-product-images-small"
URL_KAGGLE = "https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset"

# --- Parámetros del EDA ----------------------------------------------------
SEMILLA = 42
COLUMNA_OBJETIVO = "baseColour"

# Distancia CIEDE2000 al color del fondo por debajo de la cual un píxel se
# considera fondo (el catálogo usa fondo blanco/gris muy claro).
UMBRAL_DELTA_E_FONDO = 6.0
# Si la máscara de primer plano cubre menos de esta fracción, se usa la imagen completa.
FRACCION_MINIMA_PRIMER_PLANO = 0.03

PROPORCIONES_SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}

# --- Catálogo de tonos -----------------------------------------------------
# Agrupación de los 46 tonos finos en colores básicos (11 términos de
# Berlin & Kay + "Metálico" y "Multi", que no son tonos cromáticos).
GRUPOS_COLOR = {
    "Negro": ["Black"],
    "Blanco": ["White", "Off White", "Cream"],
    "Gris": ["Grey", "Grey Melange", "Charcoal"],
    "Azul": ["Blue", "Navy Blue", "Turquoise Blue", "Teal"],
    "Café": ["Brown", "Coffee Brown", "Mushroom Brown", "Tan", "Taupe",
             "Khaki", "Beige", "Skin", "Nude"],
    "Verde": ["Green", "Olive", "Sea Green", "Lime Green", "Fluorescent Green"],
    "Rojo": ["Red", "Maroon", "Burgundy"],
    "Rosado": ["Pink", "Rose", "Magenta", "Peach"],
    "Morado": ["Purple", "Lavender", "Mauve"],
    "Naranja": ["Orange", "Rust"],
    "Amarillo": ["Yellow", "Mustard"],
    "Metálico": ["Silver", "Gold", "Bronze", "Copper", "Steel", "Metallic"],
    "Multi": ["Multi"],
}
TONO_A_GRUPO = {tono: grupo for grupo, tonos in GRUPOS_COLOR.items() for tono in tonos}

# --- Alcance textil y objetivo continuo -----------------------------------------
# El negocio son colorantes textiles: solo prendas (se excluyen calzado, relojes,
# joyería y cuidado personal, que además concentraban los tonos metálicos).
ALCANCE_MASTER = ["Apparel"]

# --- Segmentación de prendas (human parsing) -------------------------------
SEG_MODELO = "mattmdjaga/segformer_b2_clothes"
SEG_TAM = (256, 192)          # alto, ancho de entrada (las imágenes son 80x60)
SEG_LOTE = 32
# Clases del modelo: 0 fondo, 1 sombrero, 2 cabello, 3 gafas, 4 parte superior,
# 5 falda, 6 pantalón, 7 vestido, 8 cinturón, 9-10 zapatos, 11 cara,
# 12-13 piernas, 14-15 brazos, 16 bolso, 17 bufanda.
CLASES_FONDO = [0]
CLASES_PIEL_CABELLO = [2, 11, 12, 13, 14, 15]
CLASES_PRENDA_POR_SUBCATEGORIA = {
    "Topwear": [4, 7],
    "Bottomwear": [5, 6],
    "Dress": [7, 4, 5],
    "Saree": [7, 4, 5, 17],
    "Innerwear": [4, 5, 6, 7],
    "Loungewear and Nightwear": [4, 5, 6, 7],
    "Apparel Set": [4, 5, 6, 7],
    "Socks": [],
}
# Si la prenda objetivo ocupa menos de esta fracción de la imagen (p. ej. una
# prenda fotografiada sola, sin modelo), se usa la máscara por ΔE contra el fondo.
FRACCION_MINIMA_PRENDA = 0.03

# --- Objetivo de color continuo --------------------------------------------
K_COLORES = 3                 # k-means sobre los píxeles de la prenda
# Prenda multicolor: el segundo color ocupa al menos esta fracción y está a más
# de este ΔE2000 del color dominante.
MULTI_FRACCION_SEGUNDO = 0.25
MULTI_DELTA_E = 20.0
# Coherencia con la etiqueta discreta: el color medido debe quedar entre los
# N prototipos de tono más cercanos al de su etiqueta.
COHERENCIA_TOP_N = 5

# --- Contexto real simulado (robustez) -------------------------------------
TEMPERATURAS_LUZ = (3000, 7500)      # K, rango de iluminantes simulados
FACTOR_BRILLO = (0.75, 1.20)
CALIDAD_JPEG = (35, 75)

# --- Modelado ---------------------------------------------------------------
TAM_ENTRADA = (128, 96)       # alto, ancho para el fine-tuning
N_CATALOGO_SIMULADO = 300
