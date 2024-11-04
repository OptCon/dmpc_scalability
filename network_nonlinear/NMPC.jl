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

using JuMP, MathOptInterface, LinearAlgebra, SparseArrays, OSQP, Plots, Random, Printf, Dates, CSV, Tables

dsqp_inner_iter::Int64 = parse(Int64, ARGS[1])

include("build_sProb_network_nonlinear_hn2.jl");
include("ode_coupled.jl")
include("../methods/structs.jl")
include("../methods/run_ADMM.jl");
include("../methods/derivative_functions.jl");
include("../methods/buildQP.jl");
include("../methods/run_dSQP.jl");
include("../methods/run_madNLP.jl")
include("../utils/integration.jl")
include("hn2_sim.jl")
include("../utils/KKTres.jl")

# dsqp_inner_iter = 0;


function run_MPC_loop(dsqp_inner_iter::Int64)

rw = 3;
N = 100;
aij = 0.2;
theta0 = 0.0;
omega0 = 0.0;
p_load = 0.5;
load_hat = 0.2;
tol = 1e-3;

Random.seed!(4)

# tuned values from open-loop analysis
rho_dsqp = 0.5; #tuned hn2 for rw = 3, n = 2, p_load = 0.5, omega0 = 0.1
dsqp_outer_iter = 1;
admm_loc_tol = tol/10;
n = 3;

@printf "Settings n = %i, rw = %i, N = %i, aij = %f, theta0 = %f, omega0 = %f, nthreads = %i, tol = %f\n" n rw N aij theta0 omega0 Threads.nthreads() tol

w = rw*n
Noscillator = w^2; regionSize = rw^2; nx = 2; nxr = nx*regionSize;
NsubSys::Int = Int64(floor(Noscillator/regionSize))
x0 = Vector{Vector{Float64}}(undef, NsubSys);
M = 0.167;
D = 0.045;
regionSize = rw^2;
nx = 2
nxr = nx*regionSize;
Mrand_base = M*(0.9*ones(regionSize)+0.2*rand(regionSize)) 
Drand_base = D*(0.9*ones(regionSize)+0.2*rand(regionSize))
x0_base = zeros(nxr)
for j = 1:regionSize
    x0_base[(j-1)*nx+1:j*nx] = [-theta0 + 2*theta0*rand(); -omega0 + 2*omega0*rand()];
end

isLoad_base = Vector{Bool}(undef, regionSize)

for l = 1:1000
    for j = 1:regionSize
        if sum(isLoad_base[1:j-1]) >= p_load*regionSize
            isLoad_base[j] = false
        else
            isLoad_base[j] = rand() <= p_load ? 1.0 : 0.0
        end
    end
    if sum(isLoad_base) >= 0.9*p_load*regionSize
        break
    end
end
p_load_new = sum(isLoad_base)/regionSize
@printf "Each subsystem has %i loads of %i buses. Specified p_load = %f, realized p_load = %f\n" sum(isLoad_base) regionSize p_load p_load_new
p_load = p_load_new 
wload_base = zeros(regionSize)


Mrand = Vector{Vector{Float64}}(undef, NsubSys);
Drand = Vector{Vector{Float64}}(undef, NsubSys);
isLoad = Vector{Vector{Bool}}(undef, NsubSys)
w_loads = Vector{Vector{Float64}}(undef, NsubSys)

for i = 1:NsubSys
    Mrand[i] = Mrand_base
    Drand[i] = Drand_base
    x0[i] = x0_base;    
    w_loads[i] = wload_base
    isLoad[i] = isLoad_base
end
buildtime = @elapsed begin sProb = build_sProb_network_nonlinear_hn2(w , rw, N , aij , x0, Mrand, Drand, isLoad, w_loads)
end

createDerivativeFun!(sProb,NsubSys);


S = sProb.SS[1];
for i = 2:NsubSys
    S = blockdiag(S,sProb.SS[i]);
end

# Run MadNLP once for compilation
@printf "Running MadNLP\n"
(sol, t_madnlp, maxiter_madnlp) = run_madNLP(sProb,600.0)
@printf "MadNLP time = %f\n" t_madnlp

nz = Vector{Int}(undef,NsubSys)
for i = 1:NsubSys
    nz[i] = size(sProb.AA[i],2)
end
createDerivativeFun!(sProb,NsubSys);

h = Vector{Vector{Float64}}(undef,NsubSys);
z = Vector{Vector{Float64}}(undef,NsubSys);
for i = 1:NsubSys
    z[i] = sol[sum(nz[1:i-1])+1:sum(nz[1:i])]; 
    h[i] = [get_ineq_constraint_value(sProb.derivatives.evaluator_ineq[i], sProb.derivatives.ineq_constraints[i], nz[i], z[i], sProb.derivatives.rhs_ineq[i]) ; sProb.llbx[i] - z[i]; z[i] - sProb.uubx[i]];
    nact = sum(h[i] .> -1e-4);
    @printf "Subystem %i has %i of %i active constraints\n" i nact length(h[i])
end

if dsqp_inner_iter > 0
    Mavg = construct_ADMM_avg(sProb.AA)
    sqp_opts = sqpOpts_struct(dsqp_outer_iter,dsqp_inner_iter,rho_dsqp,Vector{OSQP.Model}(undef, NsubSys),true,admm_loc_tol)
    (qProb, tas, admm_opts, hasIneqConstraints, z, ggam) = prepare_dSQP!(sProb,sqp_opts)
    (sol_dsqp, sol_ggam, tSqpIter, T_buildQP, T_setupOSQP, T_locSol, T_zUpdate, T_dual, T_admmIter) = iterate_dSQP!(sProb, qProb, sqp_opts, tas, admm_opts, Mavg, hasIneqConstraints, z, ggam) # run dSQP once for compilation
end

# MPC Loop
xcurr = x0;
xnext = Vector{Vector{Float64}}(undef,NsubSys)
ucurr = Vector{Vector{Float64}}(undef,NsubSys)
xtraj = Vector{Matrix{Float64}}(undef,NsubSys)
utraj = Vector{Matrix{Float64}}(undef,NsubSys)
ktraj = zeros(1,0);
ttraj = zeros(1,0);
for i = 1:NsubSys
    xtraj[i] = zeros(nxr,0);
    utraj[i] = zeros(regionSize,0);
end
y = Vector{Vector{Float64}}(undef,NsubSys)
nu = Vector{Vector{Float64}}(undef,NsubSys)
mu = Vector{Vector{Float64}}(undef,NsubSys) # Ax-b <= 0
mubox = Vector{Vector{Float64}}(undef,NsubSys) # l <= x <= u
solve_time = 0.0;
kkt_res = 0.0;
sol = zeros(sum(nz))
for k = 1:100 # MPC T_setupOSQP
    if k == 2
        load_hat = 0.1 # p.u.
        for i = 1:NsubSys
            w_loads[i] = -load_hat*rand(regionSize)
        end
    end     
    tcreateOCP = @elapsed begin sProb = build_sProb_network_nonlinear_hn2(w , rw, N , aij , xcurr, Mrand, Drand, isLoad, w_loads)
    end
    for i = 1:NsubSys
        sProb.zz0[i] = sol[sum(nz[1:i-1])+1:sum(nz[1:i])]
    end
    @printf "sProb.zz0[1][1] = %f\n" sProb.zz0[1][1]


    if dsqp_inner_iter <= 0
        (sol, t_madnlp, maxiter_madnlp) = run_madNLP(sProb,600.0)
        solve_time = t_madnlp
        @printf "NMPC step %i, create OCP took %f and solving took %f\n" k tcreateOCP t_madnlp
    else
        createDerivativeFun!(sProb,NsubSys)
        (sol_dsqp, sol_ggam, tSqpIter, T_buildQP, T_setupOSQP, T_locSol, T_zUpdate, T_dual, T_admmIter) = iterate_dSQP!(sProb, qProb, sqp_opts, tas, admm_opts, Mavg, hasIneqConstraints, z, ggam)
        sol = vcat(sol_dsqp...)
        solve_time = tSqpIter    
        for i = 1:NsubSys
            y[i] = tas.res[i].y
            nh = length(sProb.derivatives.ineq_constraints[i]);
            ng = length(sProb.derivatives.eq_constraints[i]);
            nu[i] = y[i][1:ng];
            mu[i] = y[i][ng+1: ng+nh];
            mubox[i] = y[i][ng+nh+1:end];
        end    
        kkt_res = eval_KKT_residual_NLP(sProb, sol_dsqp, sol_ggam, nu, mu, mubox)
        @printf "NMPC step %i, create OCP took %f and solving took %f, kkt_res = %f\n" k tcreateOCP tSqpIter kkt_res
    end
    for i = 1:NsubSys
        zi = zeros(nz[i]);
        if dsqp_inner_iter <= 0
            zi = sol[sum(nz[1:i-1])+1:sum(nz[1:i])];
        else
            zi = sol_dsqp[i];
        end
        ucurr[i] = sProb.SS[i]*zi;
        for j = 1:regionSize
            if isLoad[i][j]
                ucurr[i][j] = w_loads[i][j]   #overwrite inexact load values         
            end
        end
        xtraj[i] = hcat(xtraj[i], xcurr[i])
        utraj[i] = hcat(utraj[i], ucurr[i])
    end
    ttraj = hcat(ttraj, solve_time)
    ktraj = hcat(ktraj, kkt_res)
    xcent_next = hn2_sim(0.1, vcat(xcurr...), vcat(ucurr...), w, rw, Mrand, Drand, aij, NsubSys)
    for i = 1:NsubSys
        xcurr[i] = xcent_next[ nxr*(i-1)+1 : nxr*i ]
    end
    
    #Plotting 
    t_plot = plot(xtraj[1][1,:]);
    for i = 1:NsubSys
        for j = 1:2:nxr-1;
            plot!(xtraj[i][j,:]);
        end
    end
    display(t_plot)

    w_plot = plot(xtraj[1][2,:]);
    for i = 1:NsubSys        
        for j = 2:2:nxr
            plot!(xtraj[i][j,:]);
        end
    end
    display(w_plot)

    u_plot = plot(utraj[1][1,:],linetype=:steppost);
    for i = 1:NsubSys
        for j = 1:regionSize
            plot!(utraj[i][j,:],linetype=:steppost);
        end
    end
    display(u_plot)
end


return xtraj, utraj, ktraj, ttraj

end

xtraj, utraj, kkt, times = run_MPC_loop(dsqp_inner_iter);

a= Dates.format(now(), "yyyy-mm-dd-HH-MM-SS");
str = string(a, "_xtraj_iter_", dsqp_inner_iter, ".csv")
CSV.write(str, Tables.table(vcat(xtraj...)), writeheader=false)
str = string(a, "_utraj_iter_", dsqp_inner_iter, ".csv")
CSV.write(str, Tables.table(vcat(utraj...)), writeheader=false)
str = string(a, "_kkt_iter_", dsqp_inner_iter, ".csv")
CSV.write(str, Tables.table(kkt), writeheader=false)
str = string(a, "_times_iter_", dsqp_inner_iter, ".csv")
CSV.write(str, Tables.table(times), writeheader=false)