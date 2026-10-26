"""Shared helpers for the notebooks: datasets, models, embedding cache, zero-shot classifiers.

The notebooks hold the experiments and explanations; this file only holds the plumbing that
all three of them need, so each notebook stays short.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
EMB_DIR = DATA_DIR / "emb"
FIG_DIR = ROOT / "figs"
RESULTS_DIR = ROOT / "results"
for _d in (DATA_DIR, EMB_DIR, FIG_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Both are ViT-B image towers; the CLIP one uses 32x32 patches (fast), SigLIP 16x16 (≈4x the compute).
MODELS = {
    "clip": "openai/clip-vit-base-patch32",
    "siglip": "google/siglip-base-patch16-224",
}

# Images sampled per class for each split. Kept small so everything runs on a laptop CPU.
# None = use every image.
SPLIT_SIZES = {
    "pets": {"train": 50, "test": 30},
    "pathmnist": {"train": 50, "test": 100},
    "my_photos": {"train": None, "test": None},
}
PATHMNIST_SIZE = 64  # 28, 64, 128 or 224 px; 224 is a 12.6 GB download
SEED = 0

# The 12 cat breeds in Oxford-IIIT Pets; the other 25 classes are dog breeds.
PET_CATS = {
    "Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair", "Egyptian Mau",
    "Maine Coon", "Persian", "Ragdoll", "Russian Blue", "Siamese", "Sphynx",
}


# --------------------------------------------------------------------------- datasets

@dataclass
class ImageSet:
    """A labeled subset of a dataset. Images are loaded lazily, one at a time."""
    name: str
    split: str
    tag: str  # identifies the subsample, used in cache file names
    classnames: list
    labels: np.ndarray
    _get: Callable[[int], Image.Image]

    def __len__(self):
        return len(self.labels)

    def image(self, i: int) -> Image.Image:
        return self._get(i)


def _subsample(labels: np.ndarray, n_per_class: int | None, seed: int) -> np.ndarray:
    if n_per_class is None:
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    idx = [rng.permutation(np.flatnonzero(labels == c))[:n_per_class] for c in np.unique(labels)]
    return np.sort(np.concatenate(idx))


def _tag(name: str, split: str, seed: int) -> str:
    n = SPLIT_SIZES[name][split]
    return f"{'all' if n is None else f'n{n}'}_s{seed}" + (f"_px{PATHMNIST_SIZE}" if name == "pathmnist" else "")


def load_imageset(name: str, split: str, seed: int = SEED) -> ImageSet:
    """name: 'pets' | 'pathmnist' | 'my_photos'; split: 'train' | 'test'."""
    n = SPLIT_SIZES[name][split]

    if name == "pets":
        # Hugging Face mirror of Oxford-IIIT Pets (the official server is very slow). train = "trainval".
        import io
        import json

        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        path = hf_hub_download("timm/oxford-iiit-pet", f"data/{split}-00000-of-00001.parquet",
                               repo_type="dataset", local_dir=DATA_DIR / "oxford-iiit-pet")
        pf = pq.ParquetFile(path)
        all_labels = pf.read(columns=["label"]).column("label").to_numpy()
        idx = _subsample(all_labels, n, seed)
        images = pf.read(columns=["image"]).column("image").take(idx).to_pylist()  # only the sampled rows
        names = json.loads(pf.schema_arrow.metadata[b"huggingface"])["info"]["features"]["label"]["names"]
        classnames = [n_.replace("_", " ").title() for n_ in names]  # american_pit_bull_terrier -> American Pit Bull Terrier
        get = lambda i: Image.open(io.BytesIO(images[i]["bytes"])).convert("RGB")

    elif name == "pathmnist":
        from medmnist import INFO, PathMNIST

        root = DATA_DIR / "medmnist"
        root.mkdir(exist_ok=True)
        ds = PathMNIST(split=split, download=True, size=PATHMNIST_SIZE, root=str(root))
        all_labels = ds.labels.ravel()
        idx = _subsample(all_labels, n, seed)
        labels_map = INFO["pathmnist"]["label"]
        classnames = [labels_map[str(k)] for k in range(len(labels_map))]
        get = lambda i: Image.fromarray(ds.imgs[idx[i]])

    elif name == "my_photos":
        # data/my_photos/<class name>/<images>; half of each class goes to train, half to test.
        root = DATA_DIR / "my_photos"
        classnames = sorted(p.name for p in root.iterdir() if p.is_dir())
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        files, all_labels = [], []
        for c, cname in enumerate(classnames):
            fs = sorted(f for f in (root / cname).iterdir() if f.suffix.lower() in exts)
            fs = [fs[i] for i in np.random.default_rng(seed).permutation(len(fs))]
            half = len(fs) // 2
            keep = fs[:half] if split == "train" else fs[half:]
            files += keep
            all_labels += [c] * len(keep)
        all_labels = np.asarray(all_labels)
        idx = np.arange(len(files))
        get = lambda i: Image.open(files[i]).convert("RGB")

    else:
        raise ValueError(f"unknown dataset {name!r}")

    return ImageSet(name, split, _tag(name, split, seed), classnames, all_labels[idx], get)


# --------------------------------------------------------------------------- models

def load_model(key: str):
    """Returns (model, processor) for 'clip' or 'siglip'. Weights are frozen (eval mode)."""
    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(MODELS[key]).eval().to(DEVICE)
    processor = AutoProcessor.from_pretrained(MODELS[key])
    return model, processor


def is_siglip(model) -> bool:
    return model.config.model_type == "siglip"


def logit_params(model) -> tuple[float, float]:
    """(scale, bias) the model learned for turning cosine similarity into logits.
    CLIP: logits = scale * cos, softmax over classes.
    SigLIP: logits = scale * cos + bias, sigmoid per class."""
    scale = model.logit_scale.exp().item()
    bias = model.logit_bias.item() if is_siglip(model) else 0.0
    return scale, bias


@torch.no_grad()
def encode_images(model, processor, images: Sequence[Image.Image]) -> np.ndarray:
    inputs = processor(images=list(images), return_tensors="pt").to(DEVICE)
    return F.normalize(model.get_image_features(**inputs), dim=-1).cpu().numpy()


@torch.no_grad()
def encode_texts(model, processor, texts: Sequence[str]) -> np.ndarray:
    # SigLIP was trained on text padded to a fixed 64 tokens; it needs the same padding at inference.
    padding = "max_length" if is_siglip(model) else True
    inputs = processor(text=list(texts), padding=padding, truncation=True, return_tensors="pt").to(DEVICE)
    return F.normalize(model.get_text_features(**inputs), dim=-1).cpu().numpy()


# --------------------------------------------------------------------------- embedding cache

def _cache_path(model_key: str, name: str, split: str, tag: str) -> Path:
    return EMB_DIR / f"{name}_{split}_{tag}_{model_key}.npz"


def embed_imageset(model_key: str, imageset: ImageSet, model=None, processor=None,
                   batch_size: int = 32) -> np.ndarray:
    """L2-normalized image embeddings, shape (n_images, dim). Cached under data/emb/."""
    path = _cache_path(model_key, imageset.name, imageset.split, imageset.tag)
    if path.exists():
        return np.load(path)["emb"]

    from tqdm.auto import tqdm

    if model is None:
        model, processor = load_model(model_key)
    chunks = []
    for start in tqdm(range(0, len(imageset), batch_size), desc=f"{model_key} {imageset.name} {imageset.split}"):
        imgs = [imageset.image(i) for i in range(start, min(start + batch_size, len(imageset)))]
        chunks.append(encode_images(model, processor, imgs))
    emb = np.concatenate(chunks).astype(np.float32)
    np.savez(path, emb=emb, labels=imageset.labels, classnames=np.array(imageset.classnames))
    return emb


def load_embeddings(model_key: str, name: str, split: str, seed: int = SEED):
    """(embeddings, labels, classnames) from the cache written by notebook 01."""
    path = _cache_path(model_key, name, split, _tag(name, split, seed))
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found - run 01_embeddings.ipynb first")
    z = np.load(path)
    return z["emb"], z["labels"], list(z["classnames"])


# --------------------------------------------------------------------------- zero-shot

def zeroshot_weights(model, processor, prompts_per_class: Sequence[Sequence[str]]) -> np.ndarray:
    """One unit vector per class, shape (n_classes, dim).

    Each class gets one or more prompts; their text embeddings are averaged and re-normalized
    ("prompt ensembling"). The result is the weight matrix of a linear classifier built from
    language alone: logits = image_emb @ W.T.
    """
    W = []
    for prompts in prompts_per_class:
        w = encode_texts(model, processor, prompts).mean(0)
        W.append(w / np.linalg.norm(w))
    return np.stack(W).astype(np.float32)


def log_softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=axis, keepdims=True))
