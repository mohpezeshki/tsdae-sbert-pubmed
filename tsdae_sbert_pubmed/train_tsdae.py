"""Step 1: unsupervised TSDAE domain adaptation of the base model on PubMed sentences (CLS pooling,
tied encoder/decoder, 60% word deletion, constant LR, no weight decay, as in Wang et al., 2021).

Uses SentenceTransformerTrainer: the legacy ``model.fit`` builds its optimizer from the encoder's
parameters only, so the decoder's cross-attention layers in DenoisingAutoEncoderLoss would never train.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

from datasets import Dataset
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer.losses import DenoisingAutoEncoderLoss

from tsdae_sbert_pubmed.common import build_cls_model, ensure_nltk, output_exists, set_seed, setup_logging, use_fp16
from tsdae_sbert_pubmed.config import Config, build_parser, config_from_args

logger = logging.getLogger(__name__)


def delete_words(text: str, del_ratio: float, rng: random.Random) -> str:
    """TSDAE deletion noise: drop each token with probability ``del_ratio``, keeping at least one."""
    from nltk import word_tokenize
    from nltk.tokenize.treebank import TreebankWordDetokenizer

    words = word_tokenize(text)
    if not words:
        return text
    kept = [w for w in words if rng.random() >= del_ratio] or [rng.choice(words)]
    return TreebankWordDetokenizer().detokenize(kept)


def build_dataset(sentences: list[str], del_ratio: float, seed: int) -> Dataset:
    """Columns in the order DenoisingAutoEncoderLoss expects: (damaged input, original)."""
    ensure_nltk()
    rng = random.Random(seed)
    return Dataset.from_dict({"noisy": [delete_words(s, del_ratio, rng) for s in sentences], "text": sentences})


def build_trainer(cfg: Config, sentences: list[str]) -> SentenceTransformerTrainer:
    """Fresh CLS-pooled encoder + tied-decoder TSDAE loss, wrapped in a trainer."""
    model = build_cls_model(cfg.base_model, cfg.train_max_seq_length)
    # With tie_encoder_decoder=True the decoder is always built from the encoder's own checkpoint.
    loss = DenoisingAutoEncoderLoss(model, tie_encoder_decoder=True)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(Path(cfg.out_dir) / "checkpoints" / "tsdae"),
        num_train_epochs=cfg.tsdae_epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.tsdae_batch_size,
        learning_rate=cfg.tsdae_lr,
        lr_scheduler_type="constant",
        weight_decay=0.0,
        fp16=use_fp16(cfg),
        seed=cfg.seed,
        save_strategy="no",
        logging_steps=100,
        report_to="none",
    )
    dataset = build_dataset(sentences, cfg.tsdae_del_ratio, cfg.seed)
    return SentenceTransformerTrainer(model=model, args=args, train_dataset=dataset, loss=loss)


def run(cfg: Config, overwrite: bool = False) -> Path:
    output = cfg.tsdae_dir
    if output_exists(output, cfg, overwrite):
        return output
    set_seed(cfg.seed)
    sentences = cfg.sentences_path.read_text(encoding="utf-8").splitlines()[: cfg.tsdae_sentences]
    logger.info("TSDAE on %d sentences", len(sentences))
    trainer = build_trainer(cfg, sentences)
    trainer.train()
    trainer.model.save(str(output))
    cfg.save(output / "run_config.json")
    logger.info("saved TSDAE model to %s", output)
    return output


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    args = build_parser(__doc__).parse_args(argv)
    run(config_from_args(args), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
