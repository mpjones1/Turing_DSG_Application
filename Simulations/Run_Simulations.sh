#!/bin/bash
for ((c=1;c<=5;c++)) # loop over repeats
do 
    export c
    sbatch --export=ALL Parameters.sh # submit jobs on different nodes with parameters in 'Parameters.sh'
done
