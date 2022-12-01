#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --ntasks=1
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:v100
#SBATCH --job-name=softsim
#SBATCH --mem-per-cpu=128000
#SBATCH --time=5-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daniel.garijo@upm.es
#SBATCH --output=out-%j.log
##------------------------ End job description ------------------------

mkdir /home/u951/u951196/softsim/data/link_com/results/ae_median

module purge && module load Python/3.9.5-GCCcore-10.3.0 && module load CUDA/11.3.1

source /home/u951/u951196/softsim/torch_cuda/bin/activate

srun python /home/u951/u951196/softsim/src/LinkCom/AEModel/link_test.py --data-path /home/u951/u951196/softsim/data/SoftwareSim/post_process/ --json-path /home/u951/u951196/softsim/data/SoftwareSim/final_data/ --score-path /home/u951/u951196/softsim/src/LinkCom/DataEng/median.csv --save-path /home/u951/u951196/softsim/data/link_com/results/ae_median --epochs 200 --sim_type sbert_100

deactivate