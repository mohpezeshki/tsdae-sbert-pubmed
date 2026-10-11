"""Step 2: supervised training on general-domain NLI triplets with MultipleNegativesRankingLoss.

``--start none`` trains the control (base model + CLS pooling, no TSDAE); ``--start <folder>`` starts
from a locally saved model such as the TSDAE output. Data, seed and hyperparameters are identical in
both cases, so the starting weights are the only difference.
"""

from __future__ import annotations

import logging
from pathlib import Path

from datasets import load_dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.base.sampler import BatchSamplers
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss

from tsdae_sbert_pubmed.common import (
    build_cls_model,
    load_local_model,
    make_stsb_evaluator,
    output_exists,
    set_seed,
    setup_logging,
    use_fp16,
)
from tsdae_sbert_pubmed.config import Config, build_parser, config_from_args

logger = logging.getLogger(__name__)


def default_output(cfg: Config, start: str | None) -> Path:
    """models/bert-nli for the control, models/<start folder name>-nli otherwise."""
    return cfg.nli_bert_dir if start is None else cfg.models_dir / f"{Path(start).name}-nli"


def build_trainer(cfg: Config, model: SentenceTransformer, output: Path) -> SentenceTransformerTrainer:
    """Same NLI subset, sampler seed and hyperparameters regardless of the starting model."""
    nli = load_dataset("sentence-transformers/all-nli", "triplet", split="train")
    nli = nli.shuffle(seed=cfg.seed).select(range(min(cfg.nli_examples, len(nli))))
    args = SentenceTransformerTrainingArguments(
        output_dir=str(Path(cfg.out_dir) / "checkpoints" / output.name),
        num_train_epochs=cfg.nli_epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.nli_batch_size,
        learning_rate=cfg.nli_lr,
        warmup_steps=cfg.nli_warmup_ratio,  # a float in [0, 1) is a ratio of total steps
        weight_decay=cfg.nli_weight_decay,
        # all-nli repeats anchors; duplicates in a batch would be false in-batch negatives for MNRL.
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        fp16=use_fp16(cfg),
        seed=cfg.seed,
        save_strategy="no",
        logging_steps=100,
        report_to="none",
    )
    return SentenceTransformerTrainer(model=model, args=args, train_dataset=nli, loss=MultipleNegativesRankingLoss(model))


def load_start_model(cfg: Config, start: str | None) -> SentenceTransformer:
    """Base model + CLS pooling for the control, else the saved local model; same max_seq_length either way."""
    if start is None:
        return build_cls_model(cfg.base_model, cfg.train_max_seq_length)
    model = load_local_model(start)
    model.max_seq_length = cfg.train_max_seq_length
    return model


def run(cfg: Config, start: str | None, output: Path | None = None, overwrite: bool = False) -> Path:
    output = output or default_output(cfg, start)
    if output_exists(output, cfg, overwrite):
        return output
    set_seed(cfg.seed)
    model = load_start_model(cfg, start)
    logger.info("NLI training from %s -> %s", start or f"{cfg.base_model} (no TSDAE)", output)
    build_trainer(cfg, model, output).train()
    model.save(str(output))
    cfg.save(output / "run_config.json")
    stsb = make_stsb_evaluator(cfg.eval_batch_size)
    logger.info("saved %s; STSb dev Spearman %.4f", output, stsb(model)[stsb.primary_metric])
    return output


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    parser = build_parser(__doc__)
    parser.add_argument("--start", required=True, help="'none' for the control, or a local model folder")
    parser.add_argument("--output", default=None, help="output folder (default: derived from --start)")
    args = parser.parse_args(argv)
    start = None if args.start.lower() == "none" else args.start
    run(config_from_args(args), start, Path(args.output) if args.output else None, args.overwrite)


if __name__ == "__main__":
    main()
