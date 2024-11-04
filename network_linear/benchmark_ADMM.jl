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
include("../methods/run_ADMM.jl");
include("../utils/KKTres.jl")
include("../utils/integration.jl")
include("create_benchmark_QP.jl")

# nominal values
# rw = 3;
# N = 100;
# aij = 0.2;
# omega0 = 0.2;
# p_load = 0.2;
# load_hat = 0.2;




function tune_ADMM_rho(qProb, Mavg, admm_loc_tol)
    rrho_admm = [0.005, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 5.0]

    acc = Inf*ones(length(rrho_admm))
    A = hcat(qProb.AA...);
    for k = 1:length(rrho_admm)
        rho_l = rrho_admm[k]
        admm_opts_l = admm_opts_struct(100,rho_l,false,admm_loc_tol)
        (zbar_admm, gam, tIter, tLocSol, tZupdate, tDual, tmp, tSetup, nu, mu, mubox) = run_ADMM_benchmark(qProb, admm_opts_l,Mavg);
        res = eval_KKT_residual(qProb, zbar_admm, gam, nu, mu, mubox);    
        @printf "ADMM at 100 iterations with rho = %f yields KKT residual = %f\n" rho_l res
        acc[k] = res
        if res > minimum(acc)
            break;
        end
    end
    rho_admm = rrho_admm[findmin(acc)[2]]

    return rho_admm
end

function tune_ADMM_maxiter(qProb,Mavg, rho_admm, tol)

    #determine ADMM maxiter
    maxiter_admm = 300
    admm_loc_tol = tol/10
    admm_opts = admm_opts_struct(maxiter_admm,rho_admm,true,admm_loc_tol)
    NsubSys = length(qProb.AA)
    (zbar_log, gam_log, nu_log, mu_log, mubox_log, tIter, tLocSol, tZupdate, tDual, tmp, tSetup) = run_ADMM_log(qProb, admm_opts,Mavg);
    for j = 1:size(zbar_log,2)
        nu = Vector{Vector{Float64}}(undef,NsubSys)
        mu = Vector{Vector{Float64}}(undef, NsubSys)
        mubox = Vector{Vector{Float64}}(undef, NsubSys)
        for i = 1:NsubSys
            nu[i] = nu_log[i][:,j]
            mu[i] = mu_log[i][:,j]
            mubox[i] = mubox_log[i][:,j]
        end
        res = eval_KKT_residual(qProb, zbar_log[:,j], gam_log[:,j], nu, mu, mubox);
        if res <= tol
            @printf "ADMM with rho = %f and maxiter %f yields KKT residual = %f\n" rho_admm j res
            maxiter_admm = j;
            break;
        end
    end

    return maxiter_admm

end

function benchmark_ADMM(rw::Int64,N::Int64,aij::Float64,omega0::Float64,p_load_::Float64,load_hat::Float64)


    rho_admm = 1.0
    maxiter_admm = 100

    a= Dates.format(now(), "yyyy-mm-dd-HH-MM-SS");
    a = string(a, "_admm_benchmark.csv")
    write(a, "w, rw, Noscillator, NsubSys, nz, N, aij, omega0, p_load, load_hat, tol, t_admm, tLocSol, tZupdate, tDual, tSetup, iter_admm, rho_admm, locTol_admm, kktres_admm\n");

    for n = 2:6
        @printf "Settings rw = %i, N = %i, aij = %f, p_load = %f, load_hat = %f, nthreads = %i\n" rw N aij p_load_ load_hat Threads.nthreads()

        w = rw*n
        Noscillator = w^2;

        (qProb, p_load) = create_qProb(rw,N,aij,omega0,p_load_,load_hat,n)
        Mavg = construct_ADMM_avg(qProb.AA)
        NsubSys = length(qProb.AA)
        nnz = Vector{Int}(undef, NsubSys)
        for i = 1:NsubSys
            nnz[i] = size(qProb.AA[i],2);
        end
        nz = sum(nnz)

        if n == 2
            rho_admm = tune_ADMM_rho(qProb,Mavg,1e-4)
        end

        maxiter_admm_3 = tune_ADMM_maxiter(qProb,Mavg,rho_admm,1e-3)
        maxiter_admm_4 = tune_ADMM_maxiter(qProb,Mavg,rho_admm,1e-4)
        admm_loc_tol_3 = 1e-4
        admm_loc_tol_4 = 1e-5
        admm_opts_3 = admm_opts_struct(maxiter_admm_3,rho_admm,false,admm_loc_tol_3)
        admm_opts_4 = admm_opts_struct(maxiter_admm_4,rho_admm,false,admm_loc_tol_4)
        
        for l = 1:5 # repeat time measurements for this qProb
            GC.gc(true)    
            
            # benchmark ADMM for tolerance 1e-3
            @printf "Benchmarking ADMM with rho = %f and maxiter = %i\n" rho_admm maxiter_admm_3
            (zbar, gam, tIter, tLocSol, tZupdate, tDual, tmp, tSetup,  nu, mu, mubox) = run_ADMM_benchmark(qProb, admm_opts_3, Mavg);
            kktres_admm = eval_KKT_residual(qProb, zbar, gam, nu, mu, mubox);
            @printf "ADMM time = %f with KKT res = %f\n" tIter kktres_admm   
            
            io = open(a, "a");
            write(io, "$w, $rw, $Noscillator, $NsubSys, $nz, $N, $aij, $omega0, $p_load, $load_hat, 0.001, $tIter, $tLocSol, $tZupdate, $tDual, $tSetup, $maxiter_admm_3, $rho_admm, $admm_loc_tol_3, $kktres_admm\n");
            close(io);

            # benchmark ADMM for tolerance 1e-4
            @printf "Benchmarking ADMM with rho = %f and maxiter = %i\n" rho_admm maxiter_admm_4
            (zbar, gam, tIter, tLocSol, tZupdate, tDual, tmp, tSetup,  nu, mu, mubox) = run_ADMM_benchmark(qProb, admm_opts_4, Mavg);
            kktres_admm = eval_KKT_residual(qProb, zbar, gam, nu, mu, mubox);
            @printf "ADMM time = %f with KKT res = %f\n" tIter kktres_admm   
            
            io = open(a, "a");
            write(io, "$w, $rw, $Noscillator, $NsubSys, $nz, $N, $aij, $omega0, $p_load, $load_hat, 0.0001, $tIter, $tLocSol, $tZupdate, $tDual, $tSetup, $maxiter_admm_4, $rho_admm, $admm_loc_tol_4, $kktres_admm\n");
            close(io);
            
        end
    end
end



benchmark_ADMM(rw,N,aij,omega0,p_load,load_hat);
