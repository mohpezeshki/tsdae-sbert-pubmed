"""Single source of truth for paths, seeds, sizes and hyperparameters.

Every script builds its command line from :class:`Config`, so a default changed here
changes it everywhere, and any field can be overridden per run, e.g. ``--tsdae-sentences 300000``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Paths
    raw_dir: str = "data/pubmed_raw"  # downloaded PubMed baseline XML files (re-downloadable)
    out_dir: str = "outputs"  # prepared data, models and results

    # PubMed preparation
    n_files: int = 8  # number of baseline files to use (in sorted order)
    file_start: int = 0  # index of the first file; negative counts from the end (-8 = newest 8 files)
    min_title_words: int = 4
    min_abstract_words: int = 40
    min_sentence_words: int = 6
    max_pairs: int = 200_000
    n_dev: int = 2_000  # held-out title->abstract pairs, never used for training
    max_sentences: int = 300_000  # sentence pool written to disk for TSDAE

    # Shared model settings
    base_model: str = "bert-base-uncased"
    train_max_seq_length: int = 128
    eval_max_seq_length: int = 256  # applied to all four models at evaluation time
    seed: int = 42
    fp16: bool = True  # only takes effect when a CUDA GPU is available
    max_steps: int = -1  # >0 caps training steps (smoke tests only)

    # Step 1: TSDAE
    tsdae_sentences: int = 20_000  # quick test; 300_000 for the full run
    tsdae_batch_size: int = 16
    tsdae_epochs: int = 1
    tsdae_lr: float = 3e-5
    tsdae_del_ratio: float = 0.6

    # Step 2: NLI (MultipleNegativesRankingLoss)
    nli_examples: int = 100_000
    nli_batch_size: int = 32
    nli_epochs: int = 1
    nli_lr: float = 2e-5
    nli_warmup_ratio: float = 0.1
    nli_weight_decay: float = 0.01

    # Evaluation
    eval_batch_size: int = 32

    # Derived paths
    @property
    def data_dir(self) -> Path:
        return Path(self.out_dir) / "data"

    @property
    def dev_pairs_path(self) -> Path:
        return self.data_dir / "dev_pairs.jsonl"

    @property
    def sentences_path(self) -> Path:
        return self.data_dir / "tsdae_sentences.txt"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.json"

    @property
    def models_dir(self) -> Path:
        return Path(self.out_dir) / "models"

    @property
    def tsdae_dir(self) -> Path:
        return self.models_dir / "tsdae"

    @property
    def nli_bert_dir(self) -> Path:
        return self.models_dir / "bert-nli"

    @property
    def nli_tsdae_dir(self) -> Path:
        return self.models_dir / "tsdae-nli"

    @property
    def results_path(self) -> Path:
        return Path(self.out_dir) / "results.csv"

    def save(self, path: Path) -> None:
        """Write the config as JSON next to an output so runs are traceable."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dataclasses.asdict(self), indent=2))


def build_parser(description: str) -> argparse.ArgumentParser:
    """Return a parser exposing every Config field as ``--field-name``, plus ``--overwrite``."""
    parser = argparse.ArgumentParser(description=description)
    for f in dataclasses.fields(Config):
        flag = "--" + f.name.replace("_", "-")
        if f.type == "bool":
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default)
        else:
            parser.add_argument(flag, type={"int": int, "float": float}.get(f.type, str), default=f.default)
    parser.add_argument("--overwrite", action="store_true", help="redo this step even if its output exists")
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    """Pick the Config fields out of a parsed namespace."""
    return Config(**{f.name: getattr(args, f.name) for f in dataclasses.fields(Config)})
