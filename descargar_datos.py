"""Descarga el dataset Fashion Product Images (versión small) en data/raw/fashion_small/.

Uso:  python descargar_datos.py

Fuente: Hugging Face `ashraq/fashion-product-images-small` (espejo del dataset de Kaggle de
Param Aggarwal). Son dos archivos parquet (~260 MB) con las 44.072 imágenes de 60x80 px y su metadata.
"""

from huggingface_hub import snapshot_download

from core import config

if __name__ == "__main__":
    destino = config.RUTA_PARQUET.parent
    snapshot_download(config.HF_DATASET, repo_type="dataset", local_dir=str(destino), allow_patterns=["*.parquet"])
    archivos = sorted(config.RUTA_PARQUET.glob("*.parquet"))
    print(f"Listo: {len(archivos)} archivos en {config.RUTA_PARQUET}")
