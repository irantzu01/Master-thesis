# Bridging the Representation Gap: Combining Vocabulary Adaptation and Code-Switching for Low Resource Languages

Code for the MSc thesis of **Irantzu Elcoro Alberdi** (MSc Applied Data Science, Georg-August-Universität Göttingen, Human-Centered Data Science, September 2026).
First examiner and supervisor: Prof. Lisa Beinborn · Second examiner: Prof. Caroline Sporleder · Co-supervisor: Zhuojing Huang.

The thesis adapts GPT-2-small-scale models to Basque. This repository contains the code for data construction, training and evaluation; model checkpoints are attached to a release (see below).

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

**Curriculum.** Phases 1a/1b/1c use synthetic code-switched text (POS-guided lemma substitution with Apertium: adjectives and adverbs, verbs, nouns), phase 2 alternates sentences (OpenSubtitles; plus BasqueParl and EuskañolDS for Spanish), and phase 3 is monolingual Basque (Latxa). Every configuration sees the same data budget of 4,123,000 lines (1,374,333 per phase; phase 1 is three subphases of 458,111 lines). Steps per phase: 44,444 / 44,444 / 44,445 / 133,333 / 133,334 (400k in total). Weights are carried over between phases; the optimizer and learning-rate schedule restart in each phase with 5% warmup.

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

## Setup

### Python environment

Python 3.11.9, PyTorch 2.6.0 (CUDA 12.4), Hugging Face `transformers` and `datasets`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The SLURM scripts assume the cluster modules `gcc/13.2.0`, `python/3.11.9` and `cuda`, partition `react`, one GPU per job. Edit `BASE_DIR` at the top of each script to point to your checkout. Provide your own Hugging Face token (`huggingface-cli login`); no token is stored in this repository.

### External tools

These are cloned separately and are not part of the repository.

| Tool         | Repository                                           | Commit                                     | Used for                     |
|--------------|------------------------------------------------------|--------------------------------------------|------------------------------|
| mgiza        | https://github.com/moses-smt/mgiza                   | `6105035cd5b2f631b80c58abba47e49793feacf7` | GIZA++ word alignment        |
| mosesdecoder | https://github.com/moses-smt/mosesdecoder            | `08e782040189e0abb3b43a4ec9245e6179e39eca` | `train-model.perl` pipeline  |
| TokAlign     | https://github.com/ZNLP/TokAlign                     | `3ef5bc9215ff703a134e06030f5474ade36e5db6` | vocabulary alignment         |
| MultiBLiMP   | https://github.com/catherinearnett/multiblimp        | `dcd23bcd2ddf3e85c616eef1dc8ea0dd15f3482b` | morphosyntactic evaluation   |
| Apertium     | https://github.com/apertium/apertium-eu-es, https://github.com/apertium/apertium-eu-en | `apertium-eu-es` 0.3.4, `apertium-eu-en` 0.3.3 (package versions; no commit hash recorded) | bilingual dictionaries (only the `.dix` files are used, nothing is compiled) |

```bash
git clone https://github.com/moses-smt/mgiza && git -C mgiza checkout 6105035c
git clone https://github.com/moses-smt/mosesdecoder && git -C mosesdecoder checkout 08e78204
git clone https://github.com/ZNLP/TokAlign && git -C TokAlign checkout 3ef5bc92
git clone https://github.com/catherinearnett/multiblimp && git -C multiblimp checkout dcd23bcd
git clone https://github.com/apertium/apertium-eu-es && git clone https://github.com/apertium/apertium-eu-en   # use the releases matching the versions above
git -C multiblimp apply ../patches/multiblimp.patch      # adds --local_model_path
git -C TokAlign apply ../patches/tokalign.patch          # TODO: create this patch
```

Build mgiza following its own instructions (CMake; Boost 1.83.0 was used here).

## Data

The corpora are not redistributed. Download them from the sources below. `src/data_creation/` builds everything derived from them.

| Data                              | Source                                                                  | Used for                          |
|-----------------------------------|-------------------------------------------------------------------------|-----------------------------------|
| Latxa corpus v1.1                 | https://huggingface.co/datasets/HiTZ/latxa-corpus-v1.1                  | Baseline, phase 3, validation, Basque tokenizer, TokAlign |
| CulturaX (Spanish, English)       | https://huggingface.co/datasets/uonlp/CulturaX                          | phase 1 source text, TokAlign     |
| Wikipedia dumps (Spanish, English)| https://dumps.wikimedia.org                                             | phase 1 source text               |
| OpenSubtitles v2024               | https://opus.nlpl.eu/OpenSubtitles/                                     | phase 2 (es-eu and en-eu pairs)   |
| BasqueParl (343,000 lines)        | https://huggingface.co/datasets/HiTZ/basqueparl (Escribano et al., 2022) | phase 2 (Spanish)                |
| EuskañolDS silver corpus (20,008) | https://huggingface.co/datasets/HiTZ/EuskanolDS (Heredia et al., 2025)  | phase 2 (Spanish)                 |
| CCMatrix, ParaCrawl v9, Elhuyar   | https://opus.nlpl.eu (CCMatrix, ParaCrawl, Elhuyar)                     | EnerGIZAr alignment               |
| Apertium eu-es, eu-en dictionaries| https://github.com/apertium                                             | phase 1 substitution, word-level alignment |
| FLORES-200                        | https://github.com/facebookresearch/flores                              | perplexity, alignment analyses    |
| BasqueGLUE                        | https://huggingface.co/datasets/orai-nlp/basqueGLUE                     | downstream evaluation             |
| EusProficiency / EusReading       | https://huggingface.co/datasets/HiTZ/EusProficiency, `HiTZ/EusReading`  | downstream evaluation             |
| MultiBLiMP                        | https://github.com/catherinearnett/multiblimp                           | morphosyntactic evaluation        |

Expected data folder: `data/` at the repository root. To use another location, set the environment variable `THESIS_DATA` (read by `src/data_creation/data_paths.py`).

OpenSubtitles is downloaded from OPUS in Moses format and unpacked into `data/es-eu.txt/` (`OpenSubtitles.es-eu.es`, `OpenSubtitles.es-eu.eu`) and `data/en-eu.txt/` for English. BasqueParl is read from the Hugging Face cache (`basqueparl-v1.1.tsv`), EuskañolDS from `data/EuskanolDS/silver.txt`.

### Building the training data

Scripts are in `src/data_creation/` and run in this order (Spanish pipeline shown; the English one uses the same scripts with `_en`). **TODO:** confirm the download and dictionary steps.

1. `download_latxa.py` downloads the Latxa corpus (train/valid/test, five sub-corpora). `download_data.py` builds the TokAlign stabilization corpus (150,000 CulturaX samples: 120,000 Basque and 30,000 Spanish, shuffled with seed 0).
2. `clean_local_apertium.py --lang es` / `--lang en`: build the three dictionaries (adjectives and adverbs, lexical verbs, common nouns) from the Apertium `apertium-eu-es` / `apertium-eu-en` `.dix` files. Output: `data/curriculum_dictionaries/` (`dict_1_adj_adv.txt`, `dict_2_verbs.txt`, `dict_3_nouns.txt`) and `data/curriculum_dictionaries_en/` (same names with `_en`). Both folders are included in the repository. **TODO:** record the commit of each Apertium repository.
3. `1_cache_raw_spa.py` / `1_cache_raw_en.py`: cache 1,374,333 raw lines, interleaving Wikipedia (20231101) and CulturaX 50/50 (seed 42) into `raw_spanish_master.txt` / `raw_english_master.txt`.
4. `2_compile_phase_1_spa.py` / `2_compile_phase_1_en.py`: build phases 1a, 1b, 1c from three disjoint blocks of 458,111 raw lines. Words that match a dictionary entry (case-insensitive, exact surface form) are replaced by the Basque lemma; a list of high-ambiguity function words (22 Spanish, 34 English) is never replaced. The dictionaries are cumulative: 1a uses adjectives and adverbs, 1b adds verbs, 1c adds nouns. No sentence-level POS tagging is done; the POS information comes from the separate dictionaries.
5. `3_compile_phase_2_spa.py` / `3_compile_phase_2_en.py`: build phase 2 (1,374,333 lines). Spanish: EuskañolDS lines, then BasqueParl lines, then OpenSubtitles blocks that join the previous line in one language with the current line in the other (alternating direction); a second pass with the order reversed fills any remaining budget. English: OpenSubtitles blocks only, in two passes (the second inverts which language leads) if the budget is not filled.
6. `clean_parallel_corpus.py --es_in ... --eu_in ... --es_out ... --eu_out ...`: drops empty and duplicate pairs and pairs whose word-length ratio exceeds 3, applied to the concatenated CCMatrix, ParaCrawl v9 and Elhuyar files. The three corpora are concatenated (CCMatrix, ParaCrawl v9, Elhuyar, Spanish side and Basque side in the same order) before this step. The 4M-pair subsample is drawn after tokenisation, see the EnerGIZAr section.

Phase 3 reads the Latxa corpus directly. The only random seed in the pipeline above is the interleaving in step 3 (42); steps 4 and 5 are deterministic.

## Training

Each configuration has one job script in `scripts/`. They call the Python scripts in `src/Training/`.

| Configuration                    | Job script                                    | Python script                         |
|----------------------------------|-----------------------------------------------|---------------------------------------|
| Baseline                         | `scripts/train_baseline.sh`                   | `src/Training/continual_training_es.py` |
| CSCL-es                          | `scripts/train_cscl_spanish.sh`               | `src/Training/train_cscl_spanish.py`  |
| CSCL-en                          | `scripts/train_cscl_english.sh`               | `src/Training/train_cscl_english.py`  |
| TokAlign, TokAlign-CSCL          |  `scripts/train_tokalign.sh`                  | `src/Training/train_cscl_spanish.py`      |
| EnerGIZAr, EnerGIZAr-CSCL        | `scripts/train_energizar.sh`               | `src/Training/build_energizar_embeddings.py`, `train_energizar.py` |
| Basque tokenizer                 | `scripts/train_tokenizer.sh`                  | `src/Training/train_tokenizer.py`     |

The CSCL scripts train one phase at a time (`--phase 1a|1b|1c|2|3`), each resuming from the previous phase's `final_milestone` folder.
Disclaimer: some paths in the scripts might need to be changed.

### TokAlign

The Basque target tokenizer (BPE, 50,257 tokens) is trained on 200,000 Latxa sentences (`src/Training/train_tokenizer.py`, seed 42). The Spanish–Basque alignment matrix is built with TokAlign (GloVe on 50,000 Spanish CulturaX and 50,000 Basque Latxa sentences, cosine similarity, argmax copy, identity for shared tokens). Stabilization uses a separate 150,000-sample corpus (120,000 Latxa and 30,000 Spanish CulturaX sentences, seed 0): 2,500 steps with only the embeddings trainable (learning rate 6.4e-4), then 2,500 steps of full training (5e-5), warmup ratio 0.03, effective batch 128, context length 1,024, DeepSpeed ZeRO-3, bf16.

### EnerGIZAr

1. `src/data_creation/clean_parallel_corpus.py` deduplicates and filters the concatenated CCMatrix, ParaCrawl v9 and Elhuyar pairs (about 10.55M raw, 10,071,898 after cleaning) into `data/combined_clean.{es,eu}`. OpenSubtitles is deliberately excluded, since it is used in phase 2 of the curriculum.
2. `src/giza/tokenize_parallel.py` tokenises the Spanish side with the DeepESP/gpt2-spanish tokenizer and the Basque side with the target tokenizer (`basque_tokenizer_50257/`) and drops pairs longer than 101 tokens, writing `data/parallel.{es,eu}.tok`.
3. `src/giza/subsample_parallel.sh` draws 4,000,000 pairs with `shuf --random-source=<(yes 42)` (GNU coreutils) into `src/giza/parallel_sub.{es,eu}.tok`.
4. GIZA++ (mgiza + Moses `train-model.perl`; Models 1, HMM, 3 and 4, five iterations each; maximum fertility 10, maximum sentence length 101; `grow-diag-final-and`) is run on `parallel_sub`: **TODO:** paste the exact `train-model.perl` / `mgiza` commands from `giza_alignment.sh` and `giza_full_pipeline.sh`.
5. `src/Training/build_energizar_embeddings.py` builds the row-normalised translation matrix (δ = 0.01, fallback to a normal distribution around the mean source embedding when the maximum aligned probability is at most 0.1, identity for shared tokens) and initialises the model.
6. Coverage of the target vocabulary: 5,099 tokens (10.1%) identity, 15,770 (31.4%) aligned, 29,389 (58.5%) fallback.

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
| BasqueGLUE (BEC, BHTC, coref, QNLI, Vaxx, WiC) | `eval_basqueglue.sh` | `src/Evaluation/eval_basqueglue_multiseed.py`   | 5-shot, seeds 42/123/2024 (0-shot: one seed)  |
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
  title  = {Bridging the Representation Gap: Combining Vocabulary Adaptation and Code-Switching for Low Resource Languages},
  school = {University of G{\"o}ttingen},
  year   = {2026}
}
```
