#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH --job-name aft_opt
#SBATCH -o /home/pineda/QuickTune/logs/%A-%a.%x.o
#SBATCH -e /home/pineda/QuickTune/logs/%A-%a.%x.e
#SBATCH --gres=gpu:1
#SBATCH --time=0-23:59:00
#SBATCH --export=version=mini

#
source /home/pineda/anaconda3/bin/activate quick_tune
export PYTHONPATH="${PYTHONPATH}:${HOME}/QuickTune"

echo ${VERSION}
if [ "${VERSION}" == "micro" ]; then
  ./bash_scripts/run_main_experiment.sh
elif [ "${VERSION}" == "mini" ]; then
  ./bash_scripts/run_main_experiment2.sh
elif [ "${VERSION}" == "extended" ]; then
  ./bash_scripts/run_main_experiment3.sh
elif [ "${VERSION}" == "user" ]; then
  ./bash_scripts/run_example_user_interface.sh
fi
