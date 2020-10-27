#!/bin/bash
##
## Copyright (C) 2009-2017 VersatusHPC, Inc.
##
## nodes = quantidade de nodes
#SBATCH --nodes=3
##
## ntasks-per-node = quantidade de processos por node
#SBATCH --ntasks-per-node=20
##
## cpus-per-task = quantidade de threads por processo
#SBATCH --cpus-per-task=1
##
## hint = utilizar o hyper-threading dos nucleos, se houver
## Se desejar utilizar apenas os nucleos reais use "nomultithread"
#SBATCH --hint=nomultithread
##
##
#SBATCH --partition=cosmoobs
## mem-per-cpu = quantidade de memoria RAM por nucleo
#SBATCH --mem-per-cpu=15G
##
## time = quantidade de tempo
#SBATCH --time=7-00:30:00
##
## Configura o envio de e-mail quando o job for cancelado/finalizado.
## Substitua "root" por seu endereco de e-mail.
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=carlos.melo@ufrgs.br
module purge

module add openmpi
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

mpiexec -n 50 python main_code.py

