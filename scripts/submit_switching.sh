#$ -S /bin/bash
#$ -M marcus.wieder@univie.ac.at
#$ -m e
#$ -j y
#$ -p -200
#$ -o /data/shared/projects/SGE_LOG/
#$ -pe smp 2
#$ -cwd

. /data/shared/software/python_env/anaconda3/etc/profile.d/conda.sh
conda activate rew

python switching.py $1 # zink_id
