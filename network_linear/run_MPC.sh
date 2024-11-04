#!/bin/bash


# simulate linear MPC in closed-loop (Figure 7)

# Network A
# CPLEX
julia -t 40 MPC.jl 0

# OSQP
for inner_iter in {-10..-5}
do
	for l in {1..5}
	do
		julia -t 40 MPC.jl $inner_iter
	done
done

# ADMM
for inner_iter in {1..10}
do
	for l in {1..5}
	do
		julia -t 40 MPC.jl $inner_iter
	done
done

# Network B
# CPLEX
julia -t 40 MPC_loadRegion.jl 0

# OSQP
for inner_iter in {-10..-3}
do
	for l in {1..5}
	do
		julia -t 40 MPC_loadRegion.jl $inner_iter
	done
done

# ADMM
for inner_iter in {1..10}
do
	for l in {1..5}
	do
		julia -t 40 MPC_loadRegion.jl $inner_iter
	done
done
