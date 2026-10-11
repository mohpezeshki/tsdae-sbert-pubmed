"""Download PubMed baseline files, extract English (title, abstract) pairs, hold out a dev set,
and build the TSDAE sentence pool from the remaining (training) abstracts only.

Outputs (under ``<out_dir>/data``): dev_pairs.jsonl, tsdae_sentences.txt, manifest.json.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import random
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Sequence
from pathlib import Path

import requests

from tsdae_sbert_pubmed.common import ensure_nltk, output_exists, set_seed, setup_logging, write_jsonl
from tsdae_sbert_pubmed.config import Config, build_parser, config_from_args

logger = logging.getLogger(__name__)

BASELINE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
FILE_RE = re.compile(r'href="(pubmed\d+n\d+\.xml\.gz)"')
# Older MEDLINE records end with e.g. "(ABSTRACT TRUNCATED AT 250 WORDS)".
TRUNCATION_RE = re.compile(r"\(?\s*ABSTRACT TRUNCATED(?: AT \d+ WORDS)?\s*\)?\.?", re.IGNORECASE)


# Download


def list_baseline_files(base_url: str = BASELINE_URL) -> list[str]:
    """File names of the current PubMed baseline, in sorted order."""
    html = requests.get(base_url, timeout=60).text
    return sorted(set(FILE_RE.findall(html)))


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(name: str, dest_dir: Path, base_url: str = BASELINE_URL) -> Path:
    """Download one baseline file and verify it against NLM's .md5; keep a verified existing copy."""
    dest = dest_dir / name
    expected = requests.get(base_url + name + ".md5", timeout=60).text.strip().split()[-1]
    if dest.exists() and _md5(dest) == expected:
        return dest
    tmp = dest.with_suffix(".part")
    with requests.get(base_url + name, stream=True, timeout=120) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    if _md5(tmp) != expected:
        tmp.unlink()
        raise OSError(f"MD5 mismatch for {name}; re-run to retry the download")
    tmp.rename(dest)
    return dest


def download_baseline(n_files: int, file_start: int, dest_dir: Path) -> list[Path]:
    """Download (or reuse) ``n_files`` consecutive baseline files starting at index ``file_start``.

    File order is not chronological (e.g. the 2026 baseline's n0001 is mostly 1976-78 articles,
    n0450 is 1960-61, the last files are the newest), so the choice of files sets the era of the text.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    names = list_baseline_files()[file_start:][:n_files]
    assert names, f"no baseline files selected with file_start={file_start}"
    logger.info("using %d files: %s ... %s", len(names), names[0], names[-1])
    paths = []
    for name in names:
        paths.append(download_file(name, dest_dir))
        logger.info("ok %s", name)
    return paths


# Parsing


def text_of(elem: ET.Element | None) -> str:
    """All text inside ``elem`` (including inline markup such as <i>, <sub>), whitespace-normalised."""
    return " ".join("".join(elem.itertext()).split()) if elem is not None else ""


def iter_articles(path: str | Path) -> Iterator[tuple[str, str]]:
    """Yield (title, abstract) for English PubmedArticle records that have both.

    Structured abstracts (several AbstractText parts) are joined without their labels.
    Memory stays flat: each record is dropped from the tree once processed.
    """
    with gzip.open(path, "rb") as f:
        context = ET.iterparse(f, events=("start", "end"))
        _, root = next(context)
        for event, elem in context:
            if event != "end" or elem.tag not in ("PubmedArticle", "PubmedBookArticle", "DeleteCitation"):
                continue
            art = elem.find("MedlineCitation/Article") if elem.tag == "PubmedArticle" else None
            if art is not None:
                langs = [l.text for l in art.findall("Language")]
                title = text_of(art.find("ArticleTitle")).rstrip(".")
                parts = [text_of(p) for p in art.findall("Abstract/AbstractText")]
                abstract = " ".join(TRUNCATION_RE.sub("", " ".join(p for p in parts if p)).split())
                if title and abstract and (not langs or "eng" in langs):
                    yield title, abstract
            root.clear()  # drops this record and the empty shells of earlier ones


def load_pairs(paths: Sequence[Path], min_title_words: int, min_abstract_words: int) -> list[tuple[str, str]]:
    """Filter by length and drop duplicate titles or abstracts (duplicates would leak across the split
    and make the title->abstract retrieval labels ambiguous)."""
    pairs, seen_titles, seen_abstracts = [], set(), set()
    for path in paths:
        for title, abstract in iter_articles(path):
            t_key, a_key = title.casefold(), abstract.casefold()
            if (
                len(title.split()) < min_title_words
                or len(abstract.split()) < min_abstract_words
                or t_key in seen_titles
                or a_key in seen_abstracts
            ):
                continue
            seen_titles.add(t_key)
            seen_abstracts.add(a_key)
            pairs.append((title, abstract))
        logger.info("%s: %d pairs so far", Path(path).name, len(pairs))
    return pairs


def split_pairs(
    pairs: list[tuple[str, str]], n_dev: int, max_pairs: int, seed: int
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Shuffle, cap at ``max_pairs``, and return (dev, train). Dev is taken first and is disjoint from train."""
    assert len(pairs) > n_dev, f"only {len(pairs)} pairs, need more than n_dev={n_dev}"
    pairs = pairs[:]
    random.Random(seed).shuffle(pairs)
    pairs = pairs[:max_pairs]
    return pairs[:n_dev], pairs[n_dev:]


def build_sentences(train_pairs: list[tuple[str, str]], min_words: int, max_sentences: int, seed: int) -> list[str]:
    """Sentence-split the training abstracts, keep sentences with ``min_words``+ words, shuffle and cap."""
    from nltk.tokenize import sent_tokenize

    ensure_nltk()
    sentences = [s for _, a in train_pairs for s in sent_tokenize(a) if len(s.split()) >= min_words]
    random.Random(seed).shuffle(sentences)
    return sentences[:max_sentences]


def run(cfg: Config, overwrite: bool = False, xml_paths: Sequence[Path] | None = None) -> None:
    """Prepare the data. ``xml_paths`` bypasses the download (used by tests)."""
    if output_exists(cfg.manifest_path, cfg, overwrite):
        return
    set_seed(cfg.seed)
    if xml_paths is None:
        xml_paths = download_baseline(cfg.n_files, cfg.file_start, Path(cfg.raw_dir))
    pairs = load_pairs(xml_paths, cfg.min_title_words, cfg.min_abstract_words)
    dev, train = split_pairs(pairs, cfg.n_dev, cfg.max_pairs, cfg.seed)
    sentences = build_sentences(train, cfg.min_sentence_words, cfg.max_sentences, cfg.seed)

    write_jsonl(({"title": t, "abstract": a} for t, a in dev), cfg.dev_pairs_path)
    cfg.sentences_path.write_text("\n".join(sentences) + "\n", encoding="utf-8")
    manifest = {"files": [Path(p).name for p in xml_paths], "pairs": len(pairs), "dev_pairs": len(dev),
                "train_pairs": len(train), "sentences": len(sentences)}
    cfg.manifest_path.write_text(json.dumps(manifest, indent=2))
    cfg.save(cfg.data_dir / "run_config.json")
    logger.info("%d pairs -> %d dev, %d train; %d TSDAE sentences", len(pairs), len(dev), len(train), len(sentences))
    if len(sentences) < cfg.tsdae_sentences:
        logger.warning("only %d sentences, fewer than tsdae_sentences=%d", len(sentences), cfg.tsdae_sentences)


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    args = build_parser(__doc__).parse_args(argv)
    run(config_from_args(args), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
