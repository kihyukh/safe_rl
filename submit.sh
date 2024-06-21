#! /bin/bash
#
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=8192
#SBATCH --time 96:00:00

N=1

open_sem(){
    mkfifo pipe-$$
    exec 3<>pipe-$$
    rm pipe-$$
    local i=$1
    for((;i>0;i--)); do
        printf %s 000 >&3
    done
}

# run the given command asynchronously and pop/push tokens
run_with_lock(){
    local x
    # this read waits until there is something to read
    read -u 3 -n 3 x && ((0==x)) || exit $x
    (
     ( "$@"; )
    # push the return code of the command to the semaphore
    printf '%.3d' $? >&3
    )&
}

open_sem $N

output_dir="output"

mkdir -p $output_dir

for seed in 1; do
    run_with_lock srun -N 1 -n 1 -c 1 --exclusive python run.py
done

wait
