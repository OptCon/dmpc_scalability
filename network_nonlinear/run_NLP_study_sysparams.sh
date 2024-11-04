#!/bin/bash


# omega0 (Figure 4)
for omega0 in 0.2 0.3 0.4
do
	julia -t 40 benchmark_dSQP.jl 3 100 0.2 $omega0 0.2 0.2
done

# p_load (Figure 4)
for valap in 0.2 0.4 0.6
do
	julia -t 40 benchmark_dSQP.jl 3 100 0.2 0.2 $valap 0.2
done

# rw (Figure 3)
for valrw in 3 4 5
do
	julia -t 40 benchmark_dSQP.jl $valrw 100 0.2 0.2 0.2 0.2
	julia -t 40 benchmark_MadNLP.jl $valrw 100 0.2 0.2 0.2 0.2
done

