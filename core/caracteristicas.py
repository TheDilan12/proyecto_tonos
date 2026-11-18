"""Representaciones de las imágenes para los modelos de regresión de color.

- Clásica: histogramas de color CIELAB de la imagen completa y del centro (sin máscara).
- Profundas congeladas: ResNet-50 (ImageNet), DINOv2-S (auto-supervisado) y CLIP ViT-B/32
  (texto-imagen). Se usan como extractores fijos y encima se entrena una regresión lineal.
"""

import numpy as np
import torch
from PIL import Image
from skimage import color as skcolor

from core import config

BINS_LAB = (4, 8, 8)
RANGO_LAB = ((0, 100), (-60, 60), (-60, 70))


def histograma_lab(rgb):
    """Histograma Lab normalizado de la imagen completa + el del 50 % central (512 valores)."""
    lab = skcolor.rgb2lab(rgb)
    alto, ancho = lab.shape[:2]
    centro = lab[alto // 4: 3 * alto // 4, ancho // 4: 3 * ancho // 4]
    salida = []
    for region in (lab, centro):
        h, _ = np.histogramdd(region.reshape(-1, 3), bins=BINS_LAB, range=RANGO_LAB)
        salida.append(h.ravel() / max(h.sum(), 1))
    return np.concatenate(salida).astype(np.float32)


# --- extractores profundos ------------------------------------------------
_EXTRACTORES = {}
DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEDIA_IMAGENET = [0.485, 0.456, 0.406]
STD_IMAGENET = [0.229, 0.224, 0.225]


def _a_tensor(imagenes_rgb, tam=224, media=MEDIA_IMAGENET, std=STD_IMAGENET):
    x = np.stack([np.asarray(Image.fromarray(im).resize((tam, tam), Image.BICUBIC)) for im in imagenes_rgb])
    x = torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255.0
    x = (x - torch.tensor(media)[:, None, None]) / torch.tensor(std)[:, None, None]
    return x.to(DISPOSITIVO)


def _embedding(salida):
    """transformers >= 5 devuelve un objeto con `pooler_output` (embedding proyectado); antes, un tensor."""
    return salida if torch.is_tensor(salida) else salida.pooler_output


def cargar_extractor(nombre):
    """Retorna una función lista_rgb -> np.ndarray (n, d) para el extractor pedido."""
    if nombre in _EXTRACTORES:
        return _EXTRACTORES[nombre]
    if nombre == "resnet50":
        from torchvision.models import ResNet50_Weights, resnet50
        red = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
        red.fc = torch.nn.Identity()
        red.to(DISPOSITIVO)
        fn = lambda imgs: red(_a_tensor(imgs)).cpu().numpy()
    elif nombre == "dinov2_s":
        from transformers import AutoModel
        red = AutoModel.from_pretrained("facebook/dinov2-small").eval().to(DISPOSITIVO)
        def fn(imgs):
            h = red(pixel_values=_a_tensor(imgs)).last_hidden_state
            return torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1).cpu().numpy()
    elif nombre == "clip_b32":
        from transformers import CLIPModel
        red = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(DISPOSITIVO)
        media, std = [0.4815, 0.4578, 0.4082], [0.2686, 0.2613, 0.2758]
        fn = lambda imgs: _embedding(red.get_image_features(pixel_values=_a_tensor(imgs, media=media, std=std))).cpu().numpy()
    else:
        raise ValueError(f"Extractor desconocido: {nombre}")

    def extraer(imagenes_rgb, lote=64):
        partes = []
        with torch.no_grad():
            for i in range(0, len(imagenes_rgb), lote):
                partes.append(fn(imagenes_rgb[i:i + lote]))
        return np.concatenate(partes).astype(np.float32)

    _EXTRACTORES[nombre] = extraer
    return extraer


def caracteristicas_cache(nombre, conjunto, imagenes_fn, n):
    """Calcula (o lee de caché) las características `nombre` del `conjunto` (clean / real)."""
    ruta = config.RUTA_INTERMEDIOS / "caracteristicas" / f"{nombre}_{conjunto}.npy"
    if ruta.exists():
        return np.load(ruta)
    imgs = [imagenes_fn(i) for i in range(n)]
    if nombre == "hist_lab":
        X = np.stack([histograma_lab(im) for im in imgs])
    else:
        X = cargar_extractor(nombre)(imgs)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    np.save(ruta, X)
    return X


def clip_texto(frases):
    """Embeddings de texto de CLIP (para clasificación zero-shot de nombres de color)."""
    from transformers import CLIPModel, CLIPTokenizer
    red = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    with torch.no_grad():
        return _embedding(red.get_text_features(**tok(frases, padding=True, return_tensors="pt"))).numpy()
