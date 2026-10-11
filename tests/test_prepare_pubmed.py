"""Parser and split tests on a tiny hand-written PubMed-style fixture (no network)."""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path

import pytest

from tsdae_sbert_pubmed.prepare_pubmed import iter_articles, load_pairs, split_pairs

FIXTURE = Path(__file__).parent / "fixtures" / "pubmed_sample.xml"


@pytest.fixture
def xml_gz(tmp_path: Path) -> Path:
    path = tmp_path / "pubmed99n0001.xml.gz"
    with FIXTURE.open("rb") as src, gzip.open(path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return path


def test_iter_articles(xml_gz: Path) -> None:
    articles = list(iter_articles(xml_gz))
    assert articles == [
        (
            "Effect of Escherichia coli on H2O uptake in mice",
            "Water uptake by E. coli is poorly understood. We measured H2O flux in 20 mice. Uptake increased by 30%.",
        ),
        (
            "Serum lipids and coronary heart disease",
            "Serum cholesterol was measured in 500 men. High levels predicted events.",
        ),
    ]  # no-abstract (1002), German (1003) and book (1005) records are skipped; copyright line is not abstract text


def test_load_pairs_filters_and_dedups(xml_gz: Path, tmp_path: Path) -> None:
    copy = tmp_path / "pubmed99n0002.xml.gz"
    shutil.copy(xml_gz, copy)
    pairs = load_pairs([xml_gz, copy], min_title_words=4, min_abstract_words=12)
    assert [t for t, _ in pairs] == ["Effect of Escherichia coli on H2O uptake in mice"]  # 1004 (11 words) too short; second file deduped


def test_split_holds_out_dev_disjointly() -> None:
    pairs = [(f"title {i}", f"abstract {i}") for i in range(100)]
    dev, train = split_pairs(pairs, n_dev=10, max_pairs=60, seed=0)
    assert len(dev) == 10 and len(train) == 50
    assert not set(dev) & set(train)
    assert split_pairs(pairs, n_dev=10, max_pairs=60, seed=0) == (dev, train)  # deterministic
