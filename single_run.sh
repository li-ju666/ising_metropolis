#!/bin/bash
# script to repeat the simulation for a single parameter set
# with different random seeds


# run the simulation for each seed
for seed in 0 1 2
do
    python3 single_run.py --random_seed ${seed} -T 4.2
    echo "Finished run with seed ${seed}"
done
