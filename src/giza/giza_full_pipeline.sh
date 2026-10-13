#!/bin/bash
#SBATCH --job-name=giza_eu_es
#SBATCH --output=logs/giza_eu_es_%j.log
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100:1
#SBATCH --time=24:00:00
#SBATCH --partition=react


export PATH=$PATH:/mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis/mgiza/mgizapp/bin

cd /mnt/vast-react/projects/react_ag_beinborn_students/u25035/giza/full-run-v2/giza.eu.tok-es.tok

mgiza eu.tok-es.tok.dump.gizacfg -nodumps 0 -onlyaldumps 0 -model345dumpfrequency 1

# export PATH=$PATH:/mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis/mgiza/mgizapp/bin
# export PATH=$PATH:/mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis/mosesdecoder/scripts/training

# cd /mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis/code/giza

# train-model.perl -root-dir full-run-v2 -corpus parallel_sub \
#     -f es.tok -e eu.tok \
#     -alignment grow-diag-final-and -mgiza -mgiza-cpus 8 -parallel \
#     -external-bin-dir /mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis/mgiza/mgizapp/bin \
#     -last-step 3

# cp full-run-v2/giza.es.tok-eu.tok/es.tok-eu.tok.gizacfg full-run-v2/giza.es.tok-eu.tok/es.tok-eu.tok.dump.gizacfg
# cp full-run-v2/giza.eu.tok-es.tok/eu.tok-es.tok.gizacfg full-run-v2/giza.eu.tok-es.tok/eu.tok-es.tok.dump.gizacfg

# cd full-run-v2/giza.es.tok-eu.tok
# mgiza es.tok-eu.tok.dump.gizacfg -nodumps 0 -onlyaldumps 0 -model345dumpfrequency 1

# cd ../giza.eu.tok-es.tok
# mgiza eu.tok-es.tok.dump.gizacfg -nodumps 0 -onlyaldumps 0 -model345dumpfrequency 1