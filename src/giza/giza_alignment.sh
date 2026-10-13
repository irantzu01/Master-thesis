#!/bin/bash
#SBATCH --job-name=giza_alignment
#SBATCH --output=logs/giza_alignment_%j.log
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100:1
#SBATCH --time=24:00:00
#SBATCH --partition=react


# export PATH=$PATH:/mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis/mgiza/mgizapp/bin
# which mgiza

# cd /mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis/code/giza/full-run/giza.es.tok-eu.tok
# cp es.tok-eu.tok.gizacfg es.tok-eu.tok.dump.gizacfg
# sed -i 's/^nodumps 1/nodumps 0/' es.tok-eu.tok.dump.gizacfg
# sed -i 's/^onlyaldumps 1/onlyaldumps 0/' es.tok-eu.tok.dump.gizacfg

# echo "Starting MGIZA++ alignment..."
# mgiza es.tok-eu.tok.dump.gizacfg
# echo "MGIZA++ alignment completed."

# cd /mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis/code/giza/full-run/giza.eu.tok-es.tok

# cp eu.tok-es.tok.gizacfg eu.tok-es.tok.dump.gizacfg
# sed -i 's/^nodumps 1/nodumps 0/' eu.tok-es.tok.dump.gizacfg
# sed -i 's/^onlyaldumps 1/onlyaldumps 0/' eu.tok-es.tok.dump.gizacfg

# echo "Starting MGIZA++ alignment..."
# mgiza eu.tok-es.tok.dump.gizacfg
# echo "MGIZA++ alignment completed."

export PATH=$PATH:/mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis/mgiza/mgizapp/bin
export PATH=$PATH:/mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis/mosesdecoder/scripts/training

cd /mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis/code/giza

train-model.perl -root-dir full-run -corpus parallel \
    -f es.tok -e eu.tok \
    -alignment grow-diag-final-and -mgiza -mgiza-cpus 8 -parallel \
    -external-bin-dir /mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis/mgiza/mgizapp/bin \
    -last-step 3


cd /mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis/code/giza/full-run/giza.es.tok-eu.tok
cp es.tok-eu.tok.gizacfg es.tok-eu.tok.dump.gizacfg

cd /mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis/code/giza/full-run/giza.eu.tok-es.tok
cp eu.tok-es.tok.gizacfg eu.tok-es.tok.dump.gizacfg


cd /mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis/code/giza/full-run/giza.es.tok-eu.tok
mgiza es.tok-eu.tok.dump.gizacfg -nodumps 0 -onlyaldumps 0 -model345dumpfrequency 1

cd /mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis/code/giza/full-run/giza.eu.tok-es.tok
mgiza eu.tok-es.tok.dump.gizacfg -nodumps 0 -onlyaldumps 0 -model345dumpfrequency 1