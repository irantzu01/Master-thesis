"""Shared paths for the evaluation scripts.

CKPT_DIR     folder with the unpacked release checkpoints
             (baseline, cscl-es, cscl-en, tokalign, tokalign-cscl, energizar, energizar-cscl;
             see "Pretrained checkpoints" in the README).
             Default: <repo>/checkpoints, override with the environment variable CKPT_DIR.
FLORES_DIR   FLORES-200 "dev" folder containing eus_Latn.dev, spa_Latn.dev, eng_Latn.dev
             (download: https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz).
             Default: <repo>/data/flores200_dataset/dev, override with FLORES_DIR.
FLORES_DEVTEST_DIR  FLORES-200 "devtest" folder ({eus_Latn,spa_Latn,eng_Latn}.devtest), used for perplexity.
             Default: next to FLORES_DIR (<repo>/data/flores200_dataset/devtest).
RAW_SCORES_DIR  per-pair alignment scores (.npy) saved by random_cosine.py / random_jaccard.py, used for the alignment plots.
             Default: <repo>/results/raw_alignment_scores.
APERTIUM_EU_ES_DIX, APERTIUM_EU_EN_DIX  bilingual Apertium dictionaries for the word-level alignment
             (clones of apertium-eu-es / apertium-eu-en, see the README). Default: <repo>/apertium-eu-*/apertium-eu-*.eu-*.dix.
RESULTS_DIR  where per-item results (.jsonl) and summaries are written.
             Default: src/Evaluation/results, override with RESULTS_DIR.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = Path(os.environ.get("CKPT_DIR", REPO_ROOT / "checkpoints"))
FLORES_DIR = Path(os.environ.get("FLORES_DIR", REPO_ROOT / "data" / "flores200_dataset" / "dev"))
FLORES_DEVTEST_DIR = Path(os.environ.get("FLORES_DEVTEST_DIR", FLORES_DIR.parent / "devtest"))  # used for perplexity
RAW_SCORES_DIR = Path(os.environ.get("RAW_SCORES_DIR", REPO_ROOT / "results" / "raw_alignment_scores"))
APERTIUM_EU_ES_DIX = Path(os.environ.get("APERTIUM_EU_ES_DIX", REPO_ROOT / "apertium-eu-es" / "apertium-eu-es.eu-es.dix"))
APERTIUM_EU_EN_DIX = Path(os.environ.get("APERTIUM_EU_EN_DIX", REPO_ROOT / "apertium-eu-en" / "apertium-eu-en.eu-en.dix"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", Path(__file__).resolve().parent / "results"))


def ckpt(name):
    """Path of an unpacked release checkpoint."""
    return str(CKPT_DIR / name)


def is_local(path):
    """True for a local folder, False for a Hugging Face hub id such as 'openai-community/gpt2'."""
    return os.path.isdir(path)
