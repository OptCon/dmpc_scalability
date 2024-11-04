#=MIT License
Copyright (c) 2024 Goesta Stomberg, Maurice Raetsch, Alexander Engelmann, Timm Faulwasser
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.=#

using JuMP, MathOptInterface, LinearAlgebra, SparseArrays, Random, Printf, Dates

rw::Int64 = parse(Int64, ARGS[1])
N::Int64 = parse(Int64,ARGS[2])
aij::Float64 = parse(Float64,ARGS[3])
omega0::Float64 = parse(Float64,ARGS[4])
p_load::Float64 = parse(Float64,ARGS[5])
load_hat::Float64 = parse(Float64,ARGS[6])

include("ode_coupled.jl")
include("build_sProb_network_nonlinear_hn2.jl");
include("create_benchmark_NLP.jl")
include("../methods/structs.jl")
include("../methods/derivative_functions.jl")
include("../methods/structs.jl")
include("../methods/buildQP.jl");
include("../methods/run_madNLP.jl");
include("../utils/integration.jl")

# nominal values
# rw::Int64 = 2;
# N::Int64 = 100;
# aij::Float64 = 0.2;
# omega0::Float64 = 0.2
# p_load::Float64 = 0.2;
# load_hat::Float64 = 0.2;


function benchmark_MadNLP(rw::Int64,N::Int64,aij::Float64,omega0::Float64,p_load_::Float64,load_hat::Float64)

    a= Dates.format(now(), "yyyy-mm-dd-HH-MM-SS");
    a = string( a, "_madnlp_benchmark.csv")
    write(a, "w, rw, Noscillator, NsubSys, nz, N, aij, omega0, p_load, load_hat, t_madnlp, maxiter_madnlp\n");

    p_load = p_load_
    for n = 2:6

        @printf "Settings n = %i, rw = %i, N = %i, aij = %f, p_load = %f, load_hat = %f, nthreads = %i\n" n rw N aij p_load load_hat Threads.nthreads()

        (sProb, p_load) = create_sProb( rw,N,aij,omega0,p_load_,load_hat,n )

        w = rw*n
        Noscillator = w^2; 
        NsubSys = length(sProb.AA)

        nnz = Vector{Int}(undef,NsubSys)
        for i = 1:NsubSys
            nnz[i] = size(sProb.AA[i],2)
        end
        nz = sum(nnz)
        
        for l = 1:4 # repeat time measurements for this NLP
            GC.gc(true)

            @printf "Benchmarking MadNLP\n"
            (sol_madnlp, t_madnlp, maxiter_madnlp) = run_madNLP(sProb,600.0)
            @printf "MadNLP time = %f\n" t_madnlp
            
            io = open(a, "a");
            write(io, "$w, $rw, $Noscillator, $NsubSys, $nz, $N, $aij, $omega0, $p_load, $load_hat, $t_madnlp, $maxiter_madnlp\n");
            close(io);            
        end
    end
end

benchmark_MadNLP(rw,N,aij,omega0,p_load,load_hat);
