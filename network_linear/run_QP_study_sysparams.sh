#!/bin/bash


# rw with load (data for Figure 2)
for valrw in 3 4 5
do
	julia -t 40 benchmark_ADMM.jl $valrw 100 0.2 0.2 0.2 0.2
	julia -t 40 benchmark_OSQP.jl $valrw 100 0.2 0.2 0.2 0.2
	julia -t 40 benchmark_CPLEX.jl $valrw 100 0.2 0.2 0.2 0.2
done
