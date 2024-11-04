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

include("ode_coupled.jl")
include("build_sProb_network_nonlinear_hn2.jl");
include("create_benchmark_NLP.jl")
include("../methods/structs.jl")
include("../methods/run_ADMM.jl");
include("../methods/derivative_functions.jl");
include("../methods/buildQP.jl");
include("../methods/run_dSQP.jl");
include("../utils/KKTres.jl")
include("../utils/integration.jl")

# nominal values
# rw::Int64 = 3;
# N::Int64 = 100;
# aij::Float64 = 0.2;
# omega0::Float64 = 0.2
# p_load::Float64 = 0.2;
# load_hat::Float64 = 0.2;

function tune_dsqp_rho(sProb, Mavg, admm_loc_tol)
    rrho_dsqp = [0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0]
    dsqp_outer_iter = 10
    dsqp_inner_iter_tmp = 20
    NsubSys = length(sProb.AA)
    acc = Inf*ones(length(rrho_dsqp))
    for k = 1:length(rrho_dsqp)
        rho_l = rrho_dsqp[k]
        sqp_opts_l = sqpOpts_struct(dsqp_outer_iter,dsqp_inner_iter_tmp,rho_l,Vector{OSQP.Model}(undef, NsubSys),true,admm_loc_tol)
        (zbar, ggam, tSqpIter, T_buildQP, T_setupOSQP, T_locSol, T_zUpdate, T_dual, T_admmIter, nu, mu, mubox) = run_dSQP(sProb, sqp_opts_l, Mavg);
        res = eval_KKT_residual_NLP(sProb, zbar, ggam, nu, mu, mubox)
        @printf "dSQP at %i iterations with rho = %f yields KKT residual = %f\n" dsqp_outer_iter*dsqp_inner_iter_tmp rho_l res
        acc[k] = res
        if res > minimum(acc)
            break;
        end
        GC.gc(true)
    end
    rho_dsqp = rrho_dsqp[findmin(acc)[2]]

    return rho_dsqp
end

function tune_dsqp_inner_iter(sProb, Mavg, tol, rho_dsqp, inner_iter_min)
    # find lmax for reaching tolerance
    dsqp_outer_iter = 10;
    dsqp_inner_iter = inner_iter_min
    NsubSys = length(sProb.AA)
    admm_loc_tol = tol/10
    for inner_iter_l = inner_iter_min:40
        sqp_opts_l = sqpOpts_struct(dsqp_outer_iter,inner_iter_l,rho_dsqp,Vector{OSQP.Model}(undef, NsubSys),true,admm_loc_tol)
        (zbar, ggam, tSqpIter, T_buildQP, T_setupOSQP, T_locSol, T_zUpdate, T_dual, T_admmIter, nu, mu, mubox) = run_dSQP(sProb, sqp_opts_l, Mavg);
        sol_dsqp = vcat(zbar...)
        res = eval_KKT_residual_NLP(sProb, zbar, ggam, nu, mu, mubox)
        @printf "dSQP with rho = %f and iter = %i reaches KKT res = %f\n" rho_dsqp inner_iter_l*dsqp_outer_iter res
        if res <= tol
            dsqp_inner_iter = inner_iter_l;
            break;
        end
        GC.gc(true)
    end

    return dsqp_inner_iter
end

function benchmark_dSQP(rw::Int64,N::Int64,aij::Float64,omega0::Float64,p_load_::Float64,load_hat::Float64)

    a= Dates.format(now(), "yyyy-mm-dd-HH-MM-SS");
    a = string( a, "_dsqp_benchmark.csv")
    write(a, "w, rw, Noscillator, NsubSys, nz, N, aij, omega0, p_load, load_hat, tol, dsqp_outeriter, dsqp_inner_iter, rho_dsqp, tSqpIter, T_buildQP, T_setupOSQP, T_locSol, T_zUpdate, T_dual, T_admmIter, kktres_dsqp\n");

    rho_dsqp = 1.0;
    dsqp_inner_iter_3 = 2;
    dsqp_inner_iter_4 = 2;
    dsqp_outer_iter = 10;
    p_load = p_load_
    for n = 2:6  

        @printf "Settings n = %i, rw = %i, N = %i, aij = %f, p_load = %f, load_hat = %f, nthreads = %i\n" n rw N aij p_load load_hat Threads.nthreads()

        (sProb, p_load) = create_sProb( rw,N,aij,omega0,p_load_,load_hat,n )
        Mavg = construct_ADMM_avg(sProb.AA)

        w = rw*n
        Noscillator = w^2; 
        NsubSys = length(sProb.AA)

        nnz = Vector{Int}(undef,NsubSys)
        for i = 1:NsubSys
            nnz[i] = size(sProb.AA[i],2)
        end
        nz = sum(nnz)

        if n == 2
            rho_dsqp = tune_dsqp_rho(sProb, Mavg, 1e-3);
        end
        dsqp_inner_iter_3 = tune_dsqp_inner_iter(sProb, Mavg, 1e-3, rho_dsqp, dsqp_inner_iter_3)
        dsqp_inner_iter_4 = tune_dsqp_inner_iter(sProb, Mavg, 1e-4, rho_dsqp, dsqp_inner_iter_4)
        sqp_opts_3 = sqpOpts_struct(dsqp_outer_iter,dsqp_inner_iter_3,rho_dsqp,Vector{OSQP.Model}(undef, NsubSys),true,1e-4)
        sqp_opts_4 = sqpOpts_struct(dsqp_outer_iter,dsqp_inner_iter_4,rho_dsqp,Vector{OSQP.Model}(undef, NsubSys),true,1e-5)
        
        for l = 1:5 # repeat time measurements for this NLP
            GC.gc(true)
            # benchmark dSQP for tolerance 1e-3
            @printf "Benchmarking dSQP with rho = %f and inner iter = %i\n" rho_dsqp dsqp_inner_iter_3
            (zbar, ggam, tSqpIter, T_buildQP, T_setupOSQP, T_locSol, T_zUpdate, T_dual, T_admmIter, nu, mu, mubox) = run_dSQP(sProb, sqp_opts_3, Mavg);
            kktres_dsqp = eval_KKT_residual_NLP(sProb, zbar, ggam, nu, mu, mubox)
            @printf "dSQP time = %f\n" tSqpIter
            io = open(a, "a");
            write(io, "$w, $rw, $Noscillator, $NsubSys, $nz, $N, $aij, $omega0, $p_load, $load_hat, 1e-3, $dsqp_outer_iter, $dsqp_inner_iter_3, $rho_dsqp, $tSqpIter, $T_buildQP, $T_setupOSQP, $T_locSol, $T_zUpdate, $T_dual, $T_admmIter, $kktres_dsqp\n");
            close(io);

            # benchmark dSQP for tolerance 1e-4
            @printf "Benchmarking dSQP with rho = %f and inner iter = %i\n" rho_dsqp dsqp_inner_iter_4
            (zbar, ggam, tSqpIter, T_buildQP, T_setupOSQP, T_locSol, T_zUpdate, T_dual, T_admmIter, nu, mu, mubox) = run_dSQP(sProb, sqp_opts_4, Mavg);
            kktres_dsqp = eval_KKT_residual_NLP(sProb, zbar, ggam, nu, mu, mubox)
            @printf "dSQP time = %f\n" tSqpIter
            io = open(a, "a");
            write(io, "$w, $rw, $Noscillator, $NsubSys, $nz, $N, $aij, $omega0, $p_load, $load_hat, 1e-4, $dsqp_outer_iter, $dsqp_inner_iter_4, $rho_dsqp, $tSqpIter, $T_buildQP, $T_setupOSQP, $T_locSol, $T_zUpdate, $T_dual, $T_admmIter, $kktres_dsqp\n");
            close(io);
        end
    end
end

benchmark_dSQP(rw,N,aij,omega0,p_load,load_hat);
