"""Keep all Hugging Face model and dataset caches on the project drive."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
_default_cache = Path(PROJECT_DIR.anchor) / "MyTransformer_HF_Cache"
HF_CACHE_DIR = Path(os.environ.get("MYTRANSFORMER_HF_CACHE", str(_default_cache)))


def configure_huggingface_cache() -> Path:
    """Set every supported Hub/Datasets cache variable before model loading."""

    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    hub = HF_CACHE_DIR / "hub"
    datasets = HF_CACHE_DIR / "datasets"
    hub.mkdir(parents=True, exist_ok=True)
    datasets.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(HF_CACHE_DIR)
    os.environ["HF_HUB_CACHE"] = str(hub)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub)
    os.environ["HF_DATASETS_CACHE"] = str(datasets)
    # Remove an inherited legacy variable so Transformers does not emit its
    # deprecation warning. HF_HOME/HF_HUB_CACHE are the supported settings.
    os.environ.pop("TRANSFORMERS_CACHE", None)
    return HF_CACHE_DIR
