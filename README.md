# Adapting GPT-2-scale models to Basque: code-switching curriculum learning and vocabulary adaptation

Code for the MSc thesis of **Irantzu Elcoro Alberdi** (MSc Applied Data Science, University of Göttingen).
First examiner: Prof. Lisa Beinborn · Supervisor: Zhuojing Huang · Second examiner: Prof. Caroline Sporleder.

> Lines marked **TODO** still need to be checked or filled in before submission. Delete them once done.

## Overview

The thesis adapts two small GPT-2 models to Basque and compares:

1. **Continual pre-training (CPT)** on monolingual Basque versus **code-switching curriculum learning (CSCL)**.
2. **Spanish versus English** as the source language (`DeepESP/gpt2-spanish`, 117M, and `openai-community/gpt2`, 124M).
3. Two **cross-lingual vocabulary adaptation (CVA)** methods, **TokAlign** and **EnerGIZAr**, alone and combined with CSCL.

| Name in thesis  | Base model    | Vocabulary         | Training                                          |
|-----------------|---------------|--------------------|---------------------------------------------------|
| Baseline        | GPT-2 Spanish | original           | one 400k-step monolingual Basque run (CPT)        |
| CSCL-es         | GPT-2 Spanish | original           | 5-stage curriculum (400k steps in total)          |
| CSCL-en         | GPT-2 English | original           | 5-stage curriculum (400k steps in total)          |
| TokAlign        | GPT-2 Spanish | Basque (TokAlign)  | 5k-step stabilization only                        |
| TokAlign-CSCL   | GPT-2 Spanish | Basque (TokAlign)  | stabilization, then curriculum                    |
| EnerGIZAr       | GPT-2 Spanish | Basque (EnerGIZAr) | initialization only, no training                  |
| EnerGIZAr-CSCL  | GPT-2 Spanish | Basque (EnerGIZAr) | initialization, then curriculum                   |

The unadapted GPT-2 Spanish and GPT-2 English are evaluated as references.

**Curriculum.** Phases 1a/1b/1c use synthetic code-switched text (POS-guided lemma substitution with Apertium: modifiers, verbs, nouns), phase 2 alternates sentences (OpenSubtitles; BasqueParl/EuskañolDS for Spanish), and phase 3 is monolingual Basque. Steps per phase: 44,444 / 44,444 / 44,445 / 133,333 / 133,334 (400k in total). Weights are carried over between phases; the optimizer and learning-rate schedule restart in each phase with 5% warmup.

**Shared hyperparameters.** Learning rate 5e-5, cosine schedule, weight decay 0.01, AdamW (eps 1e-4), gradient clipping 0.3, bf16, batch 8 × 4 accumulation steps = 32, maximum length 512. The Baseline uses a single schedule with 1% warmup.

## Repository layout

```
scripts/      SLURM job scripts, one per experiment (train_*.sh, eval_*.sh, ...)
src/
  data_creation/   download, cleaning and construction of all training data
  Training/        training scripts (baseline, CSCL, TokAlign, EnerGIZAr)
  giza/            GIZA++ pipeline for EnerGIZAr
  Evaluation/      perplexity, BasqueGLUE, Latxa benchmarks, alignment analyses
  Experiments/     exploratory scripts (not used for reported results)
patches/      local changes to third-party code (apply as described below)
plots/        figure scripts and figures used in the thesis
results/      small result files
```

**TODO:** confirm this matches the final repository, and remove or mark legacy files (for example `train_cscl_restart.py`, `train_energizar.py` v1, the `(Copiar)` notebook, `src/giza/*.BACKUP`).

## Setup

### Python environment

Python 3.11.9, PyTorch 2.6.0 (CUDA 12.4), Hugging Face `transformers` and `datasets`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**TODO:** add `requirements.txt` (`pip freeze > requirements.txt` inside the working environment) and state the exact `transformers` version. `scripts/requirements.sh` only installs `flash-attn`.

The SLURM scripts assume the cluster modules `gcc/13.2.0`, `python/3.11.9` and `cuda`, partition `react`, one GPU per job. Edit `BASE_DIR` at the top of each script to point to your checkout. Provide your own Hugging Face token (`huggingface-cli login`); no token is stored in this repository.

### External tools

These are cloned separately and are not part of the repository.

| Tool         | Repository                                           | Commit                                     | Used for                     |
|--------------|------------------------------------------------------|--------------------------------------------|------------------------------|
| mgiza        | https://github.com/moses-smt/mgiza                   | `6105035cd5b2f631b80c58abba47e49793feacf7` | GIZA++ word alignment        |
| mosesdecoder | https://github.com/moses-smt/mosesdecoder            | `08e782040189e0abb3b43a4ec9245e6179e39eca` | `train-model.perl` pipeline  |
| TokAlign     | https://github.com/ZNLP/TokAlign                     | `3ef5bc9215ff703a134e06030f5474ade36e5db6` | vocabulary alignment         |
| MultiBLiMP   | https://github.com/catherinearnett/multiblimp        | `dcd23bcd2ddf3e85c616eef1dc8ea0dd15f3482b` | morphosyntactic evaluation   |
| Apertium     | https://github.com/apertium (`apertium-eu-es`, `apertium-eu-en`) | **TODO:** versions             | bilingual dictionaries       |

```bash
git clone https://github.com/moses-smt/mgiza && git -C mgiza checkout 6105035c
git clone https://github.com/moses-smt/mosesdecoder && git -C mosesdecoder checkout 08e78204
git clone https://github.com/ZNLP/TokAlign && git -C TokAlign checkout 3ef5bc92
git clone https://github.com/catherinearnett/multiblimp && git -C multiblimp checkout dcd23bcd
git -C multiblimp apply ../patches/multiblimp.patch      # adds --local_model_path
git -C TokAlign apply ../patches/tokalign.patch          # TODO: create this patch
```

Build mgiza following its own instructions (CMake; Boost 1.83.0 was used here).

## Data

The corpora are not redistributed. Download them from the sources below. `src/data_creation/` builds everything derived from them.

| Data                              | Source                                                                  | Used for                          |
|-----------------------------------|-------------------------------------------------------------------------|-----------------------------------|
| Latxa corpus v1.1                 | https://huggingface.co/datasets/HiTZ/latxa-corpus-v1.1                  | phase 3, Baseline, validation     |
| OpenSubtitles                     | https://opus.nlpl.eu/OpenSubtitles/                                     | phase 2, synthetic text           |
| BasqueParl / EuskañolDS           | **TODO:** link                                                          | phase 2 (Spanish)                 |
| BaSCo corpus                      | **TODO:** link                                                          | code-switching reference          |
| CCMatrix, ParaCrawl v9, Elhuyar   | **TODO:** links (OPUS / paracrawl.eu / Elhuyar)                         | EnerGIZAr alignment (4M pairs)    |
| FLORES-200                        | https://github.com/facebookresearch/flores                              | perplexity, alignment analyses    |
| BasqueGLUE                        | https://huggingface.co/datasets/orai-nlp/basqueGLUE                     | downstream evaluation             |
| EusProficiency / EusReading       | https://huggingface.co/datasets/HiTZ/EusProficiency, `HiTZ/EusReading`  | downstream evaluation             |
| MultiBLiMP                        | https://github.com/catherinearnett/multiblimp                           | morphosyntactic evaluation        |

Expected data folder: `data/` at the repository root (**TODO:** confirm; some scripts may still use absolute paths).

### Building the training data

Run in this order. **TODO:** confirm the order and arguments of each script.

1. `download_latxa.py`, `download_data.py`, `basco_corpus.py`: fetch the sources.
2. `cache_raw_spa.py`, `cache_raw_en.py`: cache the raw Spanish and English text.
3. `clean_local_apertium.py`: prepare the Apertium dictionaries (`data/curriculum_dictionaries*`).
4. `compile_phase_1_spa.py`, `compile_phase_2_spa.py`: Spanish curriculum, phases 1 and 2.
5. `2_compile_phase_1_en.py`, `3_compile_phase_2_en.py`: English curriculum, phases 1 and 2.
6. `clean_parallel_corpus.py`: clean and subsample the parallel data for GIZA++ (4M pairs).

Random seeds are fixed inside the scripts. **TODO:** confirm that every generation script sets its seed.

## Training

Each configuration has one job script in `scripts/`. They call the Python scripts in `src/Training/`.

| Configuration                    | Job script                                    | Python script                         |
|----------------------------------|-----------------------------------------------|---------------------------------------|
| Baseline                         | `scripts/train_baseline.sh`                   | `src/Training/continual_training_es.py` |
| CSCL-es                          | `scripts/train_cscl_spanish.sh`               | `src/Training/train_cscl_spanish.py`  |
| CSCL-en                          | `scripts/train_cscl_english.sh`               | `src/Training/train_cscl_english.py`  |
| TokAlign, TokAlign-CSCL          | **TODO:** job script                          | `src/Training/train_tokalign.py`      |
| EnerGIZAr, EnerGIZAr-CSCL        | `scripts/train_energizar_v2.sh`               | `src/Training/build_energizar_embeddings.py`, `train_energizar.py` |
| Basque tokenizer                 | `scripts/train_tokenizer.sh`                  | `src/Training/train_tokenizer.py`     |

The CSCL scripts train one phase at a time (`--phase 1a|1b|1c|2|3`), each resuming from the previous phase's `final_milestone` folder.

**TODO:** the scripts currently refer to `code/...`; update them to `src/Training/...` and check that the training scripts in `src/Training/` are the versions that produced the reported results (per-phase restart with 5% warmup).

### TokAlign

The Spanish–Basque alignment matrix is built with TokAlign (GloVe on 50k Spanish and 50k Basque sentences, cosine similarity, argmax copy, identity for shared tokens). Stabilization: 2,500 steps with only the embeddings trainable (learning rate 6.4e-4), then 2,500 steps of full training (5e-5), warmup ratio 0.03.

### EnerGIZAr

1. `src/giza/tokenize_parallel.py` and `src/giza/giza_full_pipeline.sh` run GIZA++ (mgiza + Moses `train-model.perl`, IBM Model 4, `grow-diag-final-and`) on the cleaned parallel data.
2. `src/Training/build_energizar_embeddings.py` builds the row-normalised translation matrix (δ = 0.01, fallback to the best-aligned source token when the maximum aligned probability is below 0.1) and initialises the model.
3. Coverage: 10.1% identity, 31.4% aligned, 58.5% fallback.

## Pretrained checkpoints

Final checkpoints are attached to the [**v1.0-checkpoints** release](https://github.com/irantzu01/Master-thesis/releases/tag/v1.0-checkpoints) (weights, config and tokenizer; no optimizer state).

| Configuration   | File                                                                                                                  |
|-----------------|-----------------------------------------------------------------------------------------------------------------------|
| Baseline        | [`baseline.tar.gz`](https://github.com/irantzu01/Master-thesis/releases/download/v1.0-checkpoints/baseline.tar.gz)               |
| CSCL-es         | [`cscl-es.tar.gz`](https://github.com/irantzu01/Master-thesis/releases/download/v1.0-checkpoints/cscl-es.tar.gz)                 |
| CSCL-en         | [`cscl-en.tar.gz`](https://github.com/irantzu01/Master-thesis/releases/download/v1.0-checkpoints/cscl-en.tar.gz)                 |
| TokAlign-CSCL   | [`tokalign-cscl.tar.gz`](https://github.com/irantzu01/Master-thesis/releases/download/v1.0-checkpoints/tokalign-cscl.tar.gz)     |
| EnerGIZAr-CSCL  | [`energizar-cscl.tar.gz`](https://github.com/irantzu01/Master-thesis/releases/download/v1.0-checkpoints/energizar-cscl.tar.gz)   |

```bash
mkdir cscl-es && tar -xzf cscl-es.tar.gz -C cscl-es
```
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("./cscl-es")
tokenizer = AutoTokenizer.from_pretrained("./cscl-es")
```

The TokAlign (stabilization only) and EnerGIZAr (initialization only) checkpoints are not in the release (**TODO:** add them if wanted). The evaluation scripts have a `MODELS` dictionary at the top; point its paths to the unpacked folders.

## Evaluation

| Metric (thesis section)               | Job script                      | Python script                                   | Protocol                                      |
|---------------------------------------|---------------------------------|-------------------------------------------------|-----------------------------------------------|
| Perplexity, Basque FLORES-200         | `calculate_perplexity.sh`       | `src/Evaluation/ppl_bootstrap.py`               | 3 bootstrap seeds                             |
| MultiBLiMP                            | `eval_multiblimp.sh`            | `multiblimp/scripts/lm_eval/eval_model.py`      | deterministic, single run                     |
| BasqueGLUE                            | `eval_basqueglue.sh`            | `src/Evaluation/eval_basqueglue_multiseed.py`   | 5-shot, seeds 42/123/2024 (0-shot: one seed)  |
| EusReading                            | `eval_eus_read.sh`              | `src/Evaluation/evaluate_eus_read.py`           | 5-shot, single run                            |
| EusProficiency                        | `eval_eus_prof.sh`              | `src/Evaluation/evaluate_eus_prof.py`           | 5-shot, single run                            |
| Sentence-level cosine / Jaccard alignment | `run_alignment.sh`, `alignment_random.sh` | `alignment_multiseed.py`, `jacard_multiseed.py`, `random_cosine.py`, `random_jaccard.py` | FLORES-200 pairs, mean-pooled last layer, seeds 42/123/2024 |
| Word-level alignment                  | `word_alignment.sh`             | `src/Evaluation/word_alignment.py`              | 200 sampled Apertium pairs                    |
| t-SNE of parallel sentences           | `tsne.sh`                       | `src/Evaluation/tsne_crosslingual_alignment.py` | middle layer                                  |

**TODO:** confirm which alignment scripts are final (`evaluate_alignment.py`, `evaluate_jacard.py`, `eval_muse.py` look older).

### Random-pair baseline

Raw cosine and Jaccard alignment are inflated by anisotropy, especially for GPT-2 English, where matched and random pairs score almost the same. The alignment scripts therefore also compute a random-pair baseline (each source item paired with a non-matching target) and the reported gap is *matched − random*, averaged over the three seeds.

## Known limitations

- Each configuration was trained once. Seeds vary only in the evaluation (bootstrap, few-shot sampling, random-pair baseline).
- MultiBLiMP per-item outputs were written to a shared results folder and overwritten between models, so only aggregate scores are available and no error bars are reported.
- Two results are unexplained: the TokAlign-CSCL drop on BasqueGLUE Vaxx and the Baseline outlier on EusReading.
- Training from scratch needs the data sources above and a GPU with bf16 support. Exact numbers may differ slightly because GPU training is not bit-wise deterministic.

## Citation

```bibtex
@mastersthesis{elcoro2026basque,
  author = {Elcoro Alberdi, Irantzu},
  title  = {TODO: thesis title},
  school = {University of G{\"o}ttingen},
  year   = {2026}
}
```
