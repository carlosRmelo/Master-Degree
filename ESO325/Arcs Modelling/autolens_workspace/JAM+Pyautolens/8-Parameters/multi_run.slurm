#!/bin/bash
##
## Copyright (C) 2009-2017 VersatusHPC, Inc.
##
## nodes = quantidade de nodes
#SBATCH --nodes=1
##
## ntasks-per-node = quantidade de processos por node
#SBATCH --ntasks-per-node=1
##
## cpus-per-task = quantidade de threads por processo
#SBATCH --cpus-per-task=20
##
## hint = utilizar o hyper-threading dos nucleos, se houver
## Se desejar utilizar apenas os nucleos reais use "nomultithread"
#SBATCH --hint=multithread
##
#SBATCH --partition=cosmoobs
## mem-per-cpu = quantidade de memoria RAM por nucleo
#SBATCH --mem-per-cpu=5G
##
## time = quantidade de tempo
#SBATCH --time=00:25:00
##

module purge
module add softwares/python/3.6-gnu-5.3

## Execucao do software
## Para softwares compilados com MPI e OpenMP, dever ser considerado a execucao
## em MPI puro ou hibrida (MPI/OpenMP).
## Na execucao com MPI puro, a variavel SLURM_CPUS_PER_TASK sera igual a 1 ou não
## estara definida. Assim o multithreading (OpenMP) estara desativado na execucao.
## Na execucao hibrida, os processos MPI estarao com multithreading ativo.
## Isto e recomendado na execucao em multiplos nodes. O numero total de nucleos
## e igual a $SLURM_JOB_NUM_NODES * $SLURM_NTASKS_PER_NODE * $SLURM_CPUS_PER_TASK
## Consulte <https://slurm.schedmd.com/mc_support.html> para mais informacoes.
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
        omp_threads=$SLURM_CPUS_PER_TASK
else
    	omp_threads=1
fi
export OMP_NUM_THREADS=$omp_threads

srun -N 1 -c 20 python multi_run.py

