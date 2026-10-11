"""Helpers shared by the pipeline steps: seeding, model construction/loading, evaluators, I/O."""

from __future__ import annotations

import dataclasses
import json
import logging
import random
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.sentence_transformer.modules import Pooling, Transformer
from sentence_transformers.util.similarity import SimilarityFunction
from transformers import set_seed as hf_set_seed

from tsdae_sbert_pubmed.config import Config

logger = logging.getLogger(__name__)
RUN_CONFIG_NAME = "run_config.json"


def setup_logging() -> None:
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    hf_set_seed(seed)


def use_fp16(cfg: Config) -> bool:
    return cfg.fp16 and torch.cuda.is_available()


def build_cls_model(base_model: str, max_seq_length: int) -> SentenceTransformer:
    """A fresh encoder with CLS pooling (the TSDAE paper's pooling)."""
    word = Transformer(base_model, max_seq_length=max_seq_length)
    pool = Pooling(word.get_embedding_dimension(), "cls")
    return SentenceTransformer(modules=[word, pool])


def load_local_model(path: str | Path) -> SentenceTransformer:
    """Load a saved SentenceTransformer from a local folder; never falls back to the Hugging Face Hub."""
    path = Path(path)
    assert path.is_dir(), f"Model folder not found: {path.resolve()}"
    assert (path / "modules.json").is_file(), f"{path.resolve()} is not a saved SentenceTransformer (no modules.json)"
    return SentenceTransformer(str(path.resolve()), local_files_only=True)


def output_exists(output: Path, cfg: Config, overwrite: bool) -> bool:
    """True if ``output`` was already produced and should be kept. Warns if it was made with other settings."""
    if overwrite or not output.exists():
        return False
    saved = output / RUN_CONFIG_NAME if output.is_dir() else output.with_name(RUN_CONFIG_NAME)
    if saved.is_file():
        old = json.loads(saved.read_text())
        diff = {k: (old.get(k), v) for k, v in dataclasses.asdict(cfg).items() if old.get(k) != v}
        if diff:
            logger.warning("%s was produced with different settings (some may not affect this step), old vs new: %s", output, diff)
    logger.info("%s already exists; skipping (pass --overwrite to redo)", output)
    return True


def make_stsb_evaluator(batch_size: int = 32) -> EmbeddingSimilarityEvaluator:
    """STS-benchmark validation split (scores already normalised to [0, 1])."""
    stsb = load_dataset("sentence-transformers/stsb", split="validation")
    return EmbeddingSimilarityEvaluator(
        stsb["sentence1"],
        stsb["sentence2"],
        stsb["score"],
        main_similarity=SimilarityFunction.COSINE,
        batch_size=batch_size,
        name="stsb-dev",
    )


def write_jsonl(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def ensure_nltk() -> None:
    """Download the NLTK sentence/word tokenizer data if it is missing."""
    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
