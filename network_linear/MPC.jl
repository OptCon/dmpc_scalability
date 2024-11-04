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

iter::Int64 = parse(Int64, ARGS[1])

include("build_qProb_network_linear_hn2.jl");
include("../methods/structs.jl")
include("../methods/run_ADMM.jl");
include("../methods/run_OSQP.jl")
include("../methods/run_CPLEX.jl")
include("../utils/integration.jl")
include("hn2_sim_lin.jl")
include("../utils/KKTres.jl")

# iter = 0; # 0 = CPLEX, <0 = OSQP, >0 = ADMM


function run_MPC_loop(iter::Int64)

rw = 3;
N = 100;
aij = 0.2;
theta0 = 0.0;
omega0 = 0.0;
p_load = 0.5;
load_hat = 0.0;
tol = 1e-3;

Random.seed!(4)

# tuned values from open-loop analysis
rho_admm = 0.4; #tuned hn2 for rw = 3, n = 2, p_load = 0.5, omega0 = 0.1
rho_osqp = 0.2;
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
buildtime = @elapsed begin qProb = build_qProb_network_linear_hn2(w , rw, N , aij , x0, Mrand, Drand, isLoad, w_loads)
end

S = qProb.SS[1];
for i = 2:NsubSys
    S = blockdiag(S,qProb.SS[i]);
end

nz = Vector{Int}(undef,NsubSys)
for i = 1:NsubSys
    nz[i] = size(qProb.AA[i],2)
end

if iter == 0 # CPLEX
    # Run CPLEX once for compilation
    @printf "Running CPLEX\n"
    (sol, solve_time) = run_CPLEX(qProb)
    @printf "CPLEX time = %f\n" solve_time
elseif iter > 0 # ADMM
    Mavg = construct_ADMM_avg(qProb.AA)
    admm_opts = admm_opts_struct(iter,rho_admm,false,admm_loc_tol)
    tas = prepare_ADMM(qProb,rho_admm,admm_loc_tol)
    (zbar, gam, tIter, tLocSol, tZupdate, tDual, admm_opts.maxiter) = iterate_ADMM_benchmark(qProb, admm_opts, tas, Mavg)
    y = Vector{Vector{Float64}}(undef,NsubSys)
    nu = Vector{Vector{Float64}}(undef,NsubSys)
    mu = Vector{Vector{Float64}}(undef,NsubSys) # Ax-b <= 0
    mubox = Vector{Vector{Float64}}(undef,NsubSys) # l <= x <= u
    for i = 1:NsubSys
        y[i] = tas.res[i].y
        nh = size(qProb.AAineq[i],1)
        ng = size(qProb.AAeq[i],1)
        nu[i] = y[i][1:ng];
        mu[i] = y[i][ng+1: ng+nh];
        mubox[i] = y[i][ng+nh+1:end];
    end    
    kkt_res = eval_KKT_residual(qProb, zbar, gam, nu, mu, mubox)
    @printf "ADMM residual = %f\n" kkt_res
else # centralized OSQP
    Cent = prepare_OSQP(qProb,rho_osqp,-iter)
    Ncons = size(qProb.AA[1],1)
    nnx = Vector{Int}(undef,NsubSys)
    nng = Vector{Int}(undef,NsubSys)
    nnh = Vector{Int}(undef,NsubSys)
    for i = 1:NsubSys
        nnx[i] = size(qProb.AA[i],2)
        nng[i] = size(qProb.AAeq[i],1)
        nnh[i] = size(qProb.AAineq[i],1)
    end
    (sol, gam, nu, mu, mubox, solve_time) = iterate_OSQP!(Cent,nnx,nng,nnh,Ncons,NsubSys,hcat(qProb.AA...))
    kkt_res = eval_KKT_residual(qProb, sol, gam, nu, mu, mubox)
    @printf "OSQP residual = %f\n" kkt_res
end

z = Vector{Vector{Float64}}(undef,NsubSys);
ggam = Vector{Vector{Float64}}(undef,NsubSys);
for i = 1:NsubSys
    z[i] = zeros(nz[i]);
    ggam[i] = zeros(nz[i]); 
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

    # update OCP
    if k == 1 || k == 2
        tcreateOCP = @elapsed begin qProb = build_qProb_network_linear_hn2(w , rw, N , aij , xcurr, Mrand, Drand, isLoad, w_loads)
        end
    else
        tcreateOCP = @elapsed begin for i = 1:NsubSys
            qProb.bbeq[i][1:nxr] = xcurr[i];
        end
        end
    end
    
    # optimizer warm start
    for i = 1:NsubSys
        qProb.xx0[i] = z[i]
        if iter > 0
            qProb.ggam0[i] = ggam[i];
        end
    end
    @printf "qProb.zz0[1][1] = %f\n" qProb.xx0[1][1]

    if iter == 0 #CPLEX
        (sol, solve_time) = run_CPLEX(qProb)
        @printf "MPC step %i, create OCP took %f and solving took %f\n" k tcreateOCP solve_time
    elseif iter > 0 #ADMM
        # update new constraints in ADMM
        tSetupOSQP = @elapsed begin Threads.@threads for i = 1:NsubSys
            OSQP.update!(tas.m[i], l=[qProb.bbeq[i]; qProb.lbineq[i]; qProb.llbx[i]], u=[qProb.bbeq[i]; qProb.ubineq[i]; qProb.uubx[i]])
        end
        end
        (zbar, gam, tIter, tLocSol, tZupdate, tDual, admm_opts.maxiter) = iterate_ADMM_benchmark(qProb, admm_opts, tas, Mavg)
        sol = zbar
        solve_time = tIter    
        for i = 1:NsubSys
            y[i] = tas.res[i].y
            nh = size(qProb.AAineq[i],1)
            ng = size(qProb.AAeq[i],1)
            nu[i] = y[i][1:ng];
            mu[i] = y[i][ng+1: ng+nh];
            mubox[i] = y[i][ng+nh+1:end];
        end    
        kkt_res = eval_KKT_residual(qProb, zbar, gam, nu, mu, mubox)
        @printf "MPC step %i, create OCP took %f and solving took %f, kkt_res = %f\n" k tcreateOCP tIter kkt_res
    else # centralized OSQP
        beq = vcat(qProb.bbeq...);
        ubineq = vcat(qProb.ubineq...);
        lbineq = vcat(qProb.lbineq...);
        uubx = vcat(qProb.uubx...);
        llbx = vcat(qProb.llbx...);
        OSQP.update!(Cent, l = [beq;zeros(Ncons);lbineq;llbx], u = [beq;zeros(Ncons);ubineq;uubx]);
        (sol, gam, nu, mu, mubox, solve_time) = iterate_OSQP!(Cent,nnx,nng,nnh,Ncons,NsubSys,hcat(qProb.AA...))
        sol .= min.(sol,uubx)
        sol .= max.(sol,llbx)
        kkt_res = eval_KKT_residual(qProb, sol, gam, nu, mu, mubox)
        @printf "MPC step %i, create OCP took %f and solving took %f, kkt_res = %f\n" k tcreateOCP solve_time kkt_res
    end
    
    # get control input and simulate centralized plant
    for i = 1:NsubSys
        z[i] = sol[sum(nz[1:i-1])+1:sum(nz[1:i])];
        if iter > 0
            ggam[i] = gam[sum(nz[1:i-1])+1:sum(nz[1:i])];
        end
        ucurr[i] = qProb.SS[i]*z[i];
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
    xcent_next = hn2_sim_lin(0.1, vcat(xcurr...), vcat(ucurr...), w, rw, Mrand, Drand, aij, NsubSys)
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

xtraj, utraj, kkt, times = run_MPC_loop(iter);

a= Dates.format(now(), "yyyy-mm-dd-HH-MM-SS");
str = string(a, "_lin_xtraj_iter_", iter, ".csv")
CSV.write(str, Tables.table(vcat(xtraj...)), writeheader=false)
str = string(a, "_lin_utraj_iter_", iter, ".csv")
CSV.write(str, Tables.table(vcat(utraj...)), writeheader=false)
str = string(a, "_lin_kkt_iter_", iter, ".csv")
CSV.write(str, Tables.table(kkt), writeheader=false)
str = string(a, "_lin_times_iter_", iter, ".csv")
CSV.write(str, Tables.table(times), writeheader=false)