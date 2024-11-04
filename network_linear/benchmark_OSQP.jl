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

using JuMP, MathOptInterface, LinearAlgebra, SparseArrays, OSQP, Ipopt, Plots, Random, Printf, Dates

rw::Int64 = parse(Int64, ARGS[1])
N::Int64 = parse(Int64,ARGS[2])
aij::Float64 = parse(Float64,ARGS[3])
omega0::Float64 = parse(Float64,ARGS[4])
p_load::Float64 = parse(Float64,ARGS[5])
load_hat::Float64 = parse(Float64,ARGS[6])

include("../methods/structs.jl")
include("build_qProb_network_linear_hn2.jl")
include("../methods/run_OSQP.jl");
include("../utils/KKTres.jl")
include("../utils/integration.jl")
include("create_benchmark_QP.jl")

# nominal values
# rw = 2;
# N = 100;
# aij = 0.2;
# omega0 = 0.2;
# p_load = 0.0;
# load_hat = 0.2;

function tune_OSQP_rho(qProb)
    rrho_osqp = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 5.0, 6.0, 10.0]
    acc = Inf*ones(length(rrho_osqp))
    for k = 1:length(rrho_osqp)
        rho_l = rrho_osqp[k]
        z_osqp, gam, nu, mu, mubox, t_log_osqp = run_OSQP_benchmark(qProb, rho_l, 100);
        res = eval_KKT_residual(qProb, z_osqp, gam, nu, mu, mubox);
        @printf "OSQP at 100 iterations with rho = %f yields KKT residual = %f\n" rho_l res
        acc[k] = res
        if res > minimum(acc)
            break;
        end
    end
    rho_osqp = rrho_osqp[findmin(acc)[2]]
    return rho_osqp
end

function tune_OSQP_maxiter(qProb, rho_osqp, tol)
    # find maxiter for reaching tolerance
    maxiter_osqp = 200
    NsubSys = length(qProb.AA)
    zlog_osqp, gam_log, nu_log, mu_log, mubox_log, t_log_osqp = run_OSQP_log(qProb, rho_osqp, maxiter_osqp);
    for j= 1:size(zlog_osqp,2)
        nu_l = Vector{Vector{Float64}}(undef,NsubSys)
        mu_l = Vector{Vector{Float64}}(undef, NsubSys)
        mubox_l = Vector{Vector{Float64}}(undef, NsubSys)
        for i = 1:NsubSys
            nu_l[i] = nu_log[i][:,j]
            mu_l[i] = mu_log[i][:,j]
            mubox_l[i] = mubox_log[i][:,j]
        end
        res = eval_KKT_residual(qProb, zlog_osqp[:,j], gam_log[:,j], nu_l, mu_l, mubox_l);
        if res <= tol
            maxiter_osqp = j
            break;
        end
    end
    @printf "OSQP tuned to %i maxiter and rho = %f\n" maxiter_osqp rho_osqp

    return maxiter_osqp;
end


function benchmark_OSQP(rw::Int64,N::Int64,aij::Float64,omega0::Float64,p_load_::Float64,load_hat::Float64)

    rho_osqp = 1.0
    maxiter_osqp_3 = 100
    maxiter_osqp_4 = 100


    a= Dates.format(now(), "yyyy-mm-dd-HH-MM-SS");
    a = string( a, "_osqp_benchmark.csv")
    write(a, "w, rw, Noscillator, NsubSys, nz, N, aij, omega0, p_load, load_hat, tol, t_osqp, iter_osqp, rho_osqp, kktres_osqp\n");

    for n = 2:6
        @printf "Settings rw = %i, N = %i, aij = %f, p_load = %f, load_hat = %f, nthreads = %i\n" rw N aij p_load_ load_hat Threads.nthreads()

        w = rw*n
        Noscillator = w^2;

        (qProb, p_load) = create_qProb(rw,N,aij,omega0,p_load_,load_hat,n)
        NsubSys = length(qProb.AA)
        nnz = Vector{Int}(undef, NsubSys)
        for i = 1:NsubSys
            nnz[i] = size(qProb.AA[i],2);
        end
        nz = sum(nnz)

        # tune OSQP rho 
        if n == 2 # only tune rho for 4 regions
            rho_osqp = tune_OSQP_rho(qProb)
        end

        maxiter_osqp_3 = tune_OSQP_maxiter(qProb,rho_osqp,1e-3)
        maxiter_osqp_4 = tune_OSQP_maxiter(qProb,rho_osqp,1e-4)

        for l = 1:5 # repeat time measurements for this qProb
            GC.gc(true)

            # benchmark OSQP for tolerance 1e-3
            @printf "Benchmarking OSQP with rho = %f and maxiter = %i\n" rho_osqp maxiter_osqp_3
            z_osqp, gam, nu, mu, mubox, t_osqp = run_OSQP_benchmark(qProb, rho_osqp, maxiter_osqp_3);
            kktres_osqp = eval_KKT_residual(qProb, z_osqp, gam, nu, mu, mubox);
            @printf "OSQP time = %f with KKT res = %f\n" t_osqp kktres_osqp
            io = open(a, "a");
            write(io, "$w, $rw, $Noscillator, $NsubSys, $nz, $N, $aij, $omega0, $p_load, $load_hat, 0.001, $t_osqp, $maxiter_osqp_3, $rho_osqp, $kktres_osqp\n");
            close(io);
            
            # benchmark OSQP for tolerance 1e-4
            @printf "Benchmarking OSQP with rho = %f and maxiter = %i\n" rho_osqp maxiter_osqp_4
            z_osqp, gam, nu, mu, mubox, t_osqp = run_OSQP_benchmark(qProb, rho_osqp, maxiter_osqp_4);
            kktres_osqp = eval_KKT_residual(qProb, z_osqp, gam, nu, mu, mubox);
            @printf "OSQP time = %f with KKT res = %f\n" t_osqp kktres_osqp

            io = open(a, "a");
            write(io, "$w, $rw, $Noscillator, $NsubSys, $nz, $N, $aij, $omega0, $p_load, $load_hat, 0.0001, $t_osqp, $maxiter_osqp_4, $rho_osqp, $kktres_osqp\n");
            close(io);


        end
    end
end

benchmark_OSQP(rw,N,aij,omega0,p_load,load_hat);
