#!/bin/bash
##
## Copyright (C) 2009-2017 VersatusHPC, Inc.
##
## nodes = quantidade de nodes
#SBATCH --nodes=1
##
## ntasks-per-node = quantidade de processos por node
#SBATCH --ntasks-per-node=16
##
## cpus-per-task = quantidade de threads por processo
#SBATCH --cpus-per-task=1
##
## hint = utilizar o hyper-threading dos nucleos, se houver
## Se desejar utilizar apenas os nucleos reais use "nomultithread"
#SBATCH --hint=nomultithread
##
## mem-per-cpu = quantidade de memoria RAM por nucleo
#SBATCH --mem-per-cpu=2G
##
## time = quantidade de tempo
#SBATCH --time=00:05:00
##
## Nome do job . Aparece na saida do comando 'qstat' .
## E recomendado, mas nao necesssario, que o nome do job
## seja o mesmo que o nome do arquivo de input
#SBATCH --job-name=Test_Emcee
##
#SBATCH --partition=cosmoobs
## Consulte <https://slurm.schedmd.com/sbatch.html> para mais informacoes sobre
## as diretivas acima.

echo -e "\n## Job iniciado em $(date +'%d-%m-%Y as %T') #####################\n"


## Informacoes do job impressos no arquivo de saida.
echo -e "\n## Jobs ativos de $USER: \n"
squeue -a --user=$USER
echo -e "\n## Node de execucao do job:         $(hostname -s) \n"
echo -e "\n## Numero de tarefas para este job: $SLURM_NTASKS \n"
#########################################
##-------  Inicio do trabalho     ----- #
#########################################

## Configura o ambiente de execucao do software.
module load openmpi
module load softwares/python/3.6-gnu-5.3

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
mpirun python script.py

echo -e "\n## Job finalizado em $(date +'%d-%m-%Y as %T') ###################"


