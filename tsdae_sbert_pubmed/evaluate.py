"""Evaluate the four models (A: base, B: TSDAE, C: NLI only, D: TSDAE + NLI) with the same
max_seq_length on held-out PubMed title->abstract retrieval, NanoBEIR biomedical subsets and STSb dev.
Writes ``<out_dir>/results.csv`` and prints the table.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.base.evaluation.evaluator import BaseEvaluator
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator, NanoBEIREvaluator

from tsdae_sbert_pubmed.common import build_cls_model, load_local_model, make_stsb_evaluator, read_jsonl, setup_logging
from tsdae_sbert_pubmed.config import Config, build_parser, config_from_args

logger = logging.getLogger(__name__)

NANOBEIR_BIO = ["scifact", "nfcorpus", "scidocs"]


def make_pubmed_evaluator(cfg: Config) -> InformationRetrievalEvaluator:
    """Titles are queries; each title's own abstract is its single relevant document among all dev abstracts."""
    dev = list(read_jsonl(cfg.dev_pairs_path))
    queries = {f"q{i}": p["title"] for i, p in enumerate(dev)}
    corpus = {f"d{i}": p["abstract"] for i, p in enumerate(dev)}
    relevant = {f"q{i}": {f"d{i}"} for i in range(len(dev))}
    return InformationRetrievalEvaluator(
        queries, corpus, relevant, name="pubmed-dev", mrr_at_k=[10], ndcg_at_k=[10],
        accuracy_at_k=[1, 5, 10], batch_size=cfg.eval_batch_size,
    )


def score(evaluator: BaseEvaluator, model: SentenceTransformer, suffix: str) -> float:
    """Run ``evaluator`` and return its metric whose key ends with ``suffix``.

    Key prefixes differ across library versions (e.g. 'pubmed-dev_cosine_ndcg@10',
    'NanoBEIR_mean_cosine_ndcg@10'), so match on the suffix and fail loudly if it is missing or ambiguous.
    """
    results = evaluator(model)
    matches = [k for k in results if k.endswith(suffix)]
    if len(matches) != 1:
        raise KeyError(f"expected one metric ending in {suffix!r}, found {matches}; available: {sorted(results)}")
    return float(results[matches[0]])


def candidates(cfg: Config) -> dict[str, Callable[[], SentenceTransformer]]:
    """Model loaders, called one at a time so only one model is in memory."""
    for path in (cfg.tsdae_dir, cfg.nli_bert_dir, cfg.nli_tsdae_dir):  # fail before any evaluation runs
        assert path.is_dir(), f"missing model folder {path}; run the training steps first"
    return {
        "A. bert (no training)": lambda: build_cls_model(cfg.base_model, cfg.eval_max_seq_length),
        "B. TSDAE only": lambda: load_local_model(cfg.tsdae_dir),
        "C. bert + NLI (control)": lambda: load_local_model(cfg.nli_bert_dir),
        "D. TSDAE + NLI": lambda: load_local_model(cfg.nli_tsdae_dir),
    }


def run(cfg: Config) -> pd.DataFrame:
    pubmed = make_pubmed_evaluator(cfg)
    nano = NanoBEIREvaluator(dataset_names=NANOBEIR_BIO, batch_size=cfg.eval_batch_size)
    stsb = make_stsb_evaluator(cfg.eval_batch_size)
    rows = {}
    for name, load in candidates(cfg).items():
        model = load()
        model.max_seq_length = cfg.eval_max_seq_length
        assert model.max_seq_length == cfg.eval_max_seq_length
        rows[name] = {
            "PubMed title->abstract nDCG@10": score(pubmed, model, "pubmed-dev_cosine_ndcg@10"),
            "NanoBEIR bio mean nDCG@10": score(nano, model, "mean_cosine_ndcg@10"),
            "STSb dev Spearman": score(stsb, model, "spearman_cosine"),
        }
        logger.info("%s: %s", name, rows[name])
    results = pd.DataFrame(rows).T.round(4)
    results.index.name = "model"
    cfg.results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(cfg.results_path)
    print(results.to_string())
    logger.info("wrote %s", cfg.results_path)
    return results


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    run(config_from_args(build_parser(__doc__).parse_args(argv)))


if __name__ == "__main__":
    main()
