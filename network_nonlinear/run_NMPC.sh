#!/bin/bash

# closed-loop NMPC simulation

# Network A
for inner_iter in 0 1 2 3 4 5 6 7 8 9 10
do
	julia -t 40 NMPC.jl $inner_iter
done

# Network B
for inner_iter in 0 1 2 3 4 5 6 7 8 9 10
do
	julia -t 40 NMPC_loadRegion.jl $inner_iter
done
