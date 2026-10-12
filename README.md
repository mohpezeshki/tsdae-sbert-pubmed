# tsdae-sbert-pubmed: biomedical sentence embeddings with TSDAE domain adaptation + SBERT

This project tests a domain-adaptation recipe from the TSDAE paper (Wang, Reimers & Gurevych, *TSDAE: Using
Transformer-based Sequential Denoising Auto-Encoder for Unsupervised Sentence Embedding Learning*, Findings of
EMNLP 2021) on biomedical text:

1. **Unsupervised, target domain.** Adapt `bert-base-uncased` with TSDAE on unlabeled PubMed sentences. TSDAE
   deletes about 60% of the words in a sentence, encodes the damaged sentence into a single vector (CLS pooling),
   and trains a decoder to rebuild the original sentence from that vector alone.
2. **Supervised, general domain.** Train the adapted model on general-domain NLI triplets
   (`sentence-transformers/all-nli`, `triplet`) with MultipleNegativesRankingLoss, as in SBERT.

**Hypothesis.** TSDAE pre-training on the target domain, followed by NLI training (model **D**), beats NLI
training alone (model **C**) on biomedical retrieval, while general-domain STSb stays about the same.

**Result so far (one seed, quick run).** It holds: D beats C by 21.5 nDCG@10 points on PubMed and 5.3 on
NanoBEIR, with STSb unchanged. TSDAE alone (B) retrieves better still. See [Results](#results).

## Pipeline

```
PubMed baseline XML (.xml.gz, NLM FTP)
        │  prepare_pubmed: parse English title+abstract pairs, dedup, shuffle (seed)
        ├──────────────► 2,000 dev pairs  (held out first; never used for training)
        └─► remaining pairs ─► sentence split ─► TSDAE sentence pool
                                                      │
bert-base-uncased (CLS) ── train_tsdae ───────────────┴──► [B] TSDAE
        │                                                     │
        │  train_nli --start none                             │  train_nli --start <TSDAE folder>
        ▼                                                     ▼
   [C] bert + NLI  ◄───── same NLI data, seed, hyperparameters ─────►  [D] TSDAE + NLI

evaluate: [A] plain bert, [B], [C], [D], all at max_seq_length 256
   • PubMed dev title→abstract retrieval (nDCG@10)
   • NanoBEIR SciFact / NFCorpus / SCIDOCS (mean nDCG@10)
   • STSb validation (Spearman, cosine)
```

## Repository layout

```
tsdae_sbert_pubmed/
  config.py          all defaults (paths, seeds, sizes, batch sizes, epochs, LRs); every field is a CLI flag
  common.py          seeding, model building/loading, STSb evaluator, I/O
  prepare_pubmed.py  download + MD5 check + parse + split; writes dev pairs and TSDAE sentences
  train_tsdae.py     step 1
  train_nli.py       step 2: --start none | <local model folder>
  evaluate.py        four-model comparison; writes results.csv and prints it
notebook/colab.ipynb runs the same scripts on Colab (no duplicated logic)
notebook/explore_pubmed.ipynb  per-file yield, publication years and sample pairs (helps pick --file-start);
                     on Colab it clones the repo itself
tests/               parser unit tests (fixture XML) and an opt-in end-to-end smoke test
```

The code is a package run with `python -m tsdae_sbert_pubmed.<step>` from the repo root, not a set of loose scripts.
That way the steps share one config and one set of helpers, run without `pip install -e` (convenient on
Colab), and the tests can import the parser directly.

## Setup

Tested with Python 3.12, sentence-transformers 6.1.0, transformers 5.19, datasets 5.1,
torch 2.14 (CPU), nltk 3.10 and pandas 3.0.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                      # parser tests, no network needed
```

## Running

### Colab

Open `notebook/colab.ipynb`, select a T4 GPU, set `REPO_DIR` (where this repo lives on your Drive) and
`QUICK` (`True`: 100k TSDAE sentences, `False`: 300k), then run all. Models, prepared data and `results.csv` are written to
`MyDrive/tsdae-sbert-pubmed-outputs/{quick|full}`. Raw XML goes to the VM's local disk.

### Local

```bash
python -m tsdae_sbert_pubmed.prepare_pubmed                       # data → outputs/data
python -m tsdae_sbert_pubmed.train_tsdae                          # B   → outputs/models/tsdae
python -m tsdae_sbert_pubmed.train_nli --start none               # C   → outputs/models/bert-nli
python -m tsdae_sbert_pubmed.train_nli --start outputs/models/tsdae   # D → outputs/models/tsdae-nli
python -m tsdae_sbert_pubmed.evaluate                             # → outputs/results.csv
```

Every step accepts every config flag (`python -m tsdae_sbert_pubmed.train_tsdae --help`). Pass the **same flags to
every step** in a run (the notebook does this through one `ARGS` string). If an output already exists, the
step skips itself and warns when it was made with different settings. Use `--overwrite` to redo a step.

`--start` must be an existing local folder that contains a saved SentenceTransformer. A Hub model name is
rejected, so the run can never silently download a different model.

### Quick test vs. full run

| Setting | `--tsdae-sentences` | TSDAE steps (batch 16) | NLI steps per model (100k triplets, batch 32) |
|---|---|---|---|
| CLI default | 20,000 | 1,250 | ~3,125 |
| Colab quick (`QUICK = True`) | 100,000 | 6,250 | ~3,125 |
| Full run (`QUICK = False`) | 300,000 | 18,750 | ~3,125 |

### Runtime, memory and disk

- **Runtime:** not measured on a GPU yet. The trainer's progress bar shows it/s after about a minute; multiply by
  the step counts above to estimate. TSDAE steps are more expensive than NLI steps because each one also runs
  the decoder.
- **GPU memory:** fp16 is enabled automatically on CUDA (`--no-fp16` turns it off). Default batch sizes are
  expected to fit a 16 GB T4 at 128 tokens, but this has not been verified. See Troubleshooting.
- **Disk:** each bert-base model folder is about 440 MB (three are saved). Baseline files are roughly 14–20 MB each.
- **Data volume (measured on two files):** `pubmed26n0001` has 30,000 records, of which 12,387 pass the filters
  (English, title of 4+ words, abstract of 40+ words). The newest file, `pubmed26n1334`, gave 3,933 pairs
  and about 10 sentences per abstract. With the default 8 files, expect roughly 100k pairs. The
  `--max-pairs 200000` cap is then not reached, but there are more than 300k sentences, so the full run has
  enough data.
- **Which files:** baseline file order is **not chronological**. In the 2026 baseline, n0001 is mostly
  1976–78, n0300 is 1996–97, n0450 is 1960–61, and n1334 is 2025–26. The default uses the first 8 files
  (1970s text). To use the newest files, pass `--file-start -8`.

## Reading the results table

`results.csv` has one row per model and three columns:

| Column | Meaning |
|---|---|
| PubMed title->abstract nDCG@10 | Each held-out title is a query; its own abstract is the only relevant document among the 2,000 dev abstracts. |
| NanoBEIR bio mean nDCG@10 | Mean over NanoSciFact, NanoNFCorpus and NanoSCIDOCS (50 queries each; corpora of 2,919 / 2,953 / 1,866 documents). |
| STSb dev Spearman | Spearman correlation between cosine similarity and gold scores on the STSb validation split (1,500 pairs). |

What to compare:

- **D vs. C** on the two retrieval columns is the main claim. **D vs. C** on STSb should show little change.
- **B vs. A** shows what TSDAE alone does, without any labels.
- All models use CLS pooling and the same `--eval-max-seq-length` (256) at evaluation time.

## Results

### Current pipeline (Colab quick run)

Settings: first 8 baseline files (`--file-start 0`, mostly 1970s text), 100,000 TSDAE sentences, 100,000 NLI
triplets, seed 42, everything else at the defaults in `config.py`. Run on a Colab T4.

| Model | PubMed nDCG@10 | NanoBEIR bio nDCG@10 | STSb dev Spearman |
|---|---|---|---|
| A. bert (no training) | 0.0436 | 0.0162 | 0.3173 |
| B. TSDAE only | **0.8162** | **0.2600** | 0.7095 |
| C. bert + NLI (control) | 0.4864 | 0.1690 | 0.8359 |
| D. TSDAE + NLI | 0.7017 | 0.2220 | **0.8411** |

What this run shows:

- **D beats C, as hypothesised.** TSDAE before NLI adds 21.5 nDCG@10 points on PubMed (0.486 → 0.702) and 5.3
  points on NanoBEIR (0.169 → 0.222). STSb does not get worse (0.836 → 0.841). The NanoBEIR gap is well above
  the 1–2 points expected from noise, but this is still a single seed.
- **TSDAE alone is the best retriever here.** B beats D by 11.5 points on PubMed and 3.8 points on NanoBEIR.
  NLI training gives the best STSb score (D: 0.841 vs. B: 0.710) but costs part of the biomedical retrieval
  gain. For biomedical retrieval only, model B is the better choice in this setup.
- **Much stronger than the earlier run.** Compared with the superseded run below, B went from 0.29 to 0.82 on
  PubMed and D from 0.49 to 0.70. Two things changed at once: the fixes (the decoder is now trained, and C and D
  get the same NLI order) and 5× more TSDAE sentences (100k vs. 20k). This run cannot tell how much each one
  contributed.
- **Caveat on the PubMed column.** The dev abstracts are held out from TSDAE training, but they come from the
  same 8 files and the same era as the TSDAE sentences. The task is also easy (one relevant document among
  2,000), so it favours a model adapted to exactly this text. NanoBEIR is the more independent check, and it
  shows the same ranking (B > D > C > A).

### Earlier run with the original Colab code (superseded)

These numbers come from the original notebook, before this repo's fixes. The exact settings weren't recorded.
That code's defaults were: the first 8 baseline files, 20,000 TSDAE sentences (the quick setting), and
100,000 NLI triplets.

| Model | PubMed nDCG@10 | NanoBEIR bio nDCG@10 | STSb dev Spearman |
|---|---|---|---|
| A. bert (no training) | 0.0420 | 0.0162 | 0.3173 |
| B. TSDAE only | 0.2881 | 0.0799 | 0.6431 |
| C. bert + NLI (control) | 0.4939 | 0.1825 | 0.8355 |
| D. TSDAE + NLI | 0.4926 | 0.1737 | 0.8350 |

What this run shows:

- **Not a fair test of the hypothesis.** The run has two known defects. TSDAE was trained with `model.fit`,
  which never updates the decoder's cross-attention layers. C and D also did not see their NLI data in the
  same order. The table is kept for reference only.
- **D did not beat C.** D was 0.13 points lower on PubMed nDCG@10, 0.88 points lower on NanoBEIR and the same
  on STSb. All three gaps are within the expected noise (NanoBEIR has 150 queries in total). So this run gives
  no support for the hypothesis, and it is too noisy to count against it either.
- **TSDAE alone did learn.** B beats A clearly on every metric (PubMed 0.04 → 0.29, STSb 0.32 → 0.64), even
  with the decoder defect. Any benefit of TSDAE was apparently erased by NLI training in this setup.
- **Possible reasons D ≈ C:**
  - The weakened TSDAE described above.
  - Only 20k TSDAE sentences.
  - 1970s PubMed text (the first 8 files), while the evaluation sets are modern.
  - Training on 100k NLI triplets (about 3,125 steps) may override what TSDAE learned.

Next runs to try, in order:

1. The current pipeline at 20k TSDAE sentences (`--tsdae-sentences 20000`, the setting of the superseded run), to
   separate the effect of the fixes from the effect of more data.
2. Two or three `--seed` values for C and D (and B), to measure the noise.
3. The full setting (`--tsdae-sentences 300000`).
4. Newer abstracts (`--file-start -8`), closer in era to the NanoBEIR corpora.

## Known limitations

- **Small NanoBEIR query sets.** Each subset has 50 queries (150 in total), so differences under about 1–2 nDCG
  points may be noise. To see seed variance, repeat C and D with a few `--seed` values (use a separate
  `--out-dir` per seed).
- **Single seed by default.** Seeds are fixed, so C and D see the same data in the same order. The smoke
  test checks that both get identical NLI batches. One seed still doesn't show the run-to-run spread.
- **Easy in-domain retrieval task.** The PubMed corpus is only the 2,000 dev abstracts, with one relevant
  document per query.
- **Possible overlap with NanoBEIR corpora.** SciFact and NFCorpus documents are biomedical abstracts, and some
  may also appear among the PubMed sentences TSDAE trains on. TSDAE never sees labels, which matches the
  paper's unsupervised domain-adaptation setting, but keep it in mind when reading the NanoBEIR gains.
- **STSb validation is used only for reporting.** No checkpoint is selected on it.
- **Licensing.** NLM provides PubMed data under its terms and conditions
  (<https://www.nlm.nih.gov/databases/download/terms_and_conditions.html>). Some abstracts may be protected by
  copyright held by publishers or authors. NLM asks users to acknowledge NLM as the source of the data and not
  to imply NLM endorsement. This repo does not redistribute PubMed text. Keep `outputs/data` and any trained
  models within those terms.

## Troubleshooting

- **CUDA out of memory.** Lower the batch size of the failing step: `--tsdae-batch-size 8` (the TSDAE paper's
  value) or `--nli-batch-size 16`. MultipleNegativesRankingLoss uses in-batch negatives, so a smaller NLI batch
  changes the results. Pass the same value to both NLI runs; the shared flags make this automatic. You can also
  lower `--train-max-seq-length` (applies to C and D alike) or `--eval-batch-size` during evaluation.
- **Metric key errors.** `evaluate.score` matches metric names by suffix (`pubmed-dev_cosine_ndcg@10`,
  `mean_cosine_ndcg@10`, `spearman_cosine`) and raises a `KeyError` that lists every available key. If a
  future version renames a key, update the suffix in `tsdae_sbert_pubmed/evaluate.py`. `requirements.txt` caps
  sentence-transformers below 7 for this reason.
- **`ImportError: ... requires accelerate`.** Install from `requirements.txt`; `sentence-transformers[train]`
  pulls it in.
- **Colab disconnects / Drive.** Re-run all cells. Finished steps are skipped because their outputs are on Drive.
  A step interrupted mid-training saves nothing and restarts from scratch. Raw XML lives on the VM disk and is
  re-downloaded, with already-verified files reused when the VM survived. If the disconnect hit while a model was
  being saved, delete that model folder (or pass `--overwrite`) and re-run the step. Keep quick and full runs in
  separate folders (the notebook does this) so a quick-run model is never reused in a full run.
- **Download errors.** Each file is checked against NLM's `.md5`. A mismatch deletes the partial file and
  raises an error; re-run to retry. Set `HF_TOKEN` if Hugging Face Hub rate limits slow the dataset and
  model downloads.

## Tests

```bash
pytest -q                                   # parser + split tests (fixture XML, no network)
RUN_SMOKE=1 pytest -q tests/test_smoke.py   # end-to-end on CPU: tiny BERT, synthetic data, 2 steps/stage (~1 min, needs network)
```

The smoke test checks that the code paths run, not that the models are any good.

## Implementation notes

- TSDAE uses `SentenceTransformerTrainer`, not the legacy `model.fit`. In sentence-transformers 3.3.1 and 6.1.0,
  `fit` builds its optimizer from the encoder's parameters only. The decoder's cross-attention layers (randomly
  initialised) and LM-head transform live in `DenoisingAutoEncoderLoss`, so `fit` never updates them; this was
  checked by a direct test. The trainer optimizes the loss module's parameters, and the smoke test asserts that
  the cross-attention weights change.
- NLI training uses the no-duplicates batch sampler, because all-nli repeats anchors and duplicates would act as
  false in-batch negatives. It also keeps the legacy `fit` defaults the original notebook relied on: linear
  schedule with 10% warmup and weight decay 0.01.
