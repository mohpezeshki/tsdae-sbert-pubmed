"""End-to-end smoke test: tiny model, synthetic PubMed sample, 2 training steps per stage.

Checks that every code path runs, not model quality. Needs network (Hugging Face Hub) and takes a few
minutes on CPU, so it only runs when RUN_SMOKE=1.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import random
from pathlib import Path

import pytest
import torch

from tsdae_sbert_pubmed import evaluate, prepare_pubmed, train_nli, train_tsdae
from tsdae_sbert_pubmed.config import Config

pytestmark = pytest.mark.skipif(os.environ.get("RUN_SMOKE") != "1", reason="set RUN_SMOKE=1 to run")

WORDS = "protein kinase cell tumor patient mice gene expression receptor dose trial insulin serum brain".split()


def synthetic_xml_gz(path: Path, n_articles: int = 120, seed: int = 0) -> Path:
    """English articles with 6-sentence abstracts (~700 sentences in total)."""
    rng = random.Random(seed)
    sentence = lambda: " ".join(rng.choice(WORDS) for _ in range(rng.randint(7, 12))).capitalize() + "."
    records = "".join(
        f"<PubmedArticle><MedlineCitation><PMID>{i}</PMID><Article>"
        f"<ArticleTitle>Study {i} of {sentence()}</ArticleTitle>"
        f"<Abstract><AbstractText>{' '.join(sentence() for _ in range(6))}</AbstractText></Abstract>"
        f"<Language>eng</Language></Article></MedlineCitation></PubmedArticle>"
        for i in range(n_articles)
    )
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(f"<?xml version='1.0'?><PubmedArticleSet>{records}</PubmedArticleSet>")
    return path


@pytest.fixture(scope="module")
def cfg(tmp_path_factory: pytest.TempPathFactory) -> Config:
    tmp = tmp_path_factory.mktemp("smoke")
    return Config(
        out_dir=str(tmp / "out"),
        base_model="google/bert_uncased_L-2_H-128_A-2",
        n_dev=20,
        max_sentences=500,
        tsdae_sentences=300,
        tsdae_batch_size=8,
        nli_examples=200,
        nli_batch_size=8,
        train_max_seq_length=32,
        eval_max_seq_length=64,
        max_steps=2,
    )


def test_pipeline(cfg: Config, tmp_path: Path) -> None:
    prepare_pubmed.run(cfg, xml_paths=[synthetic_xml_gz(tmp_path / "pubmed99n0001.xml.gz")])
    manifest = json.loads(cfg.manifest_path.read_text())
    assert manifest["dev_pairs"] == 20 and manifest["sentences"] == 500

    # TSDAE: the decoder-only weights (cross-attention) must be trained too, unlike with legacy model.fit.
    sentences = cfg.sentences_path.read_text().splitlines()[: cfg.tsdae_sentences]
    trainer = train_tsdae.build_trainer(cfg, sentences)
    encoder_ids = {id(p) for p in trainer.model.parameters()}
    decoder_only = {n: p for n, p in trainer.loss.decoder.named_parameters() if id(p) not in encoder_ids}
    cross = {n: p.detach().clone() for n, p in decoder_only.items() if "crossattention" in n}
    assert cross and len(decoder_only) < len(list(trainer.loss.decoder.parameters()))  # tied + untied parts
    trainer.train()
    assert all(not torch.equal(v, decoder_only[n]) for n, v in cross.items())

    train_tsdae.run(cfg)
    assert json.loads((cfg.tsdae_dir / "config.json").read_text()).get("is_decoder", False) is False

    # C and D: identical NLI batches; only the starting weights differ.
    batches = []
    for start in (None, str(cfg.tsdae_dir)):
        model = train_nli.load_start_model(cfg, start)
        assert model.max_seq_length == cfg.train_max_seq_length
        batch = next(iter(train_nli.build_trainer(cfg, model, cfg.models_dir / "tmp").get_train_dataloader()))
        batches.append({k: v for k, v in batch.items() if k.endswith("input_ids")})
    assert batches[0] and batches[0].keys() == batches[1].keys()
    assert all(torch.equal(batches[0][k], batches[1][k]) for k in batches[0])

    assert train_nli.run(cfg, None) == cfg.nli_bert_dir
    assert train_nli.run(cfg, str(cfg.tsdae_dir)) == cfg.nli_tsdae_dir
    with pytest.raises(AssertionError):
        train_nli.load_start_model(cfg, "bert-base-uncased")  # a Hub name is refused, not silently downloaded

    results = evaluate.run(cfg)
    assert results.shape == (4, 3)
    assert all(math.isfinite(v) for v in results.to_numpy().ravel())
    assert cfg.results_path.is_file()
