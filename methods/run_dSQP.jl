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

mutable struct sqpOpts_struct
    maxIter::Int64
    maxInnerIter::Int64
    rho::Float64
    OSQP::Vector{OSQP.Model}
    useGN::Bool
    locTol::Float64
end

function run_dSQP(sProb::sProb_struct, opts::sqpOpts_struct, Mavg::SparseMatrixCSC{Float64,Int64})
    
    (qProb, tas, admm_opts, hasIneqConstraints, z, ggam) = prepare_dSQP!(sProb,opts)

    (z, ggam, tSqpIter, T_buildQP, T_setupOSQP, T_locSol, T_zUpdate, T_dual, T_admmIter) = iterate_dSQP!(sProb, qProb, opts, tas, admm_opts, Mavg, hasIneqConstraints, z, ggam)
    NsubSys = length(sProb.AA)
    y = Vector{Vector{Float64}}(undef,NsubSys)
    nu = Vector{Vector{Float64}}(undef,NsubSys)
    mu = Vector{Vector{Float64}}(undef,NsubSys) # Ax-b <= 0
    mubox = Vector{Vector{Float64}}(undef,NsubSys) # l <= x <= u
    for i = 1:NsubSys
        y[i] = tas.res[i].y
        nh = length(sProb.derivatives.ineq_constraints[i]);
        ng = length(sProb.derivatives.eq_constraints[i]);
        nu[i] = y[i][1:ng];
        mu[i] = y[i][ng+1: ng+nh];
        mubox[i] = y[i][ng+nh+1:end];
    end
    
    return (z, ggam, tSqpIter, T_buildQP, T_setupOSQP, T_locSol, T_zUpdate, T_dual, T_admmIter, nu, mu, mubox)
end

function prepare_dSQP!(sProb::sProb_struct, opts::sqpOpts_struct)

    NsubSys = size(sProb.AA,1);
    nz = Vector{Int}(undef, NsubSys)    

    for i = 1:NsubSys
        nz[i] = size(sProb.AA[i],2);
    end

    if isnothing(sProb.zz0)
        for i = 1:NsubSys
            sProb.zz0[i]    = zeros(Float64, nz[i])
        end
    end
    
    if isnothing(sProb.ggam0)
        for i = 1:NsubSys
            sProb.ggam0[i]    = zeros(Float64, nz[i])
        end
    end

    z = Vector{Vector{Float64}}(undef,NsubSys);
    ggam = Vector{Vector{Float64}}(undef,NsubSys);
    for i = 1:NsubSys
        z[i] = sProb.zz0[i] #deep copy to not change sProb.zz0
        ggam[i] = sProb.ggam0[i]
    end
    qProb, hasIneqConstraints = buildQP(sProb,z,ggam,opts.useGN);
    tas = prepare_ADMM(qProb, opts.rho, opts.locTol); # setup OSQP solvers and averaging matrix

    admm_opts = admm_opts_struct(opts.maxInnerIter,opts.rho,false,opts.locTol)

    return (qProb, tas, admm_opts, hasIneqConstraints, z, ggam)
end

function iterate_dSQP!(sProb::sProb_struct, qProb::qProb_struct, opts::sqpOpts_struct, tas::thread_admm_avg_struct, admm_opts::admm_opts_struct, Mavg::SparseMatrixCSC{Float64,Int64}, hasIneqConstraints::Bool, z::Vector{Vector{Float64}}, ggam::Vector{Vector{Float64}})

    NsubSys = size(sProb.AA,1);
    nz = Vector{Int}(undef, NsubSys)    

    for i = 1:NsubSys
        nz[i] = size(sProb.AA[i],2);
    end

    T_buildQP = 0.0;
    T_setupOSQP = 0.0;
    T_locSol = 0.0;
    T_zUpdate = 0.0;
    T_dual = 0.0;
    T_admmIter = 0.0;
    
    tSqpIter = @elapsed begin for k = 1:opts.maxIter
        tBuildQP = @elapsed begin qProb = updateQP(qProb, sProb,z,ggam,NsubSys, Val(hasIneqConstraints))
        end
        
        tSetupOSQP = @elapsed begin Threads.@threads for i = 1:NsubSys
            OSQP.update!(tas.m[i], Ax=[qProb.AAeq[i]; qProb.AAineq[i]; sparse(I,nz[i],nz[i])].nzval, l=[qProb.bbeq[i]; qProb.lbineq[i]; qProb.llbx[i]], u=[qProb.bbeq[i]; qProb.ubineq[i]; qProb.uubx[i]])
        end
        end

        (zbar, gam, tIter, tLocSol, tZupdate, tDual, maxiter) = iterate_ADMM_benchmark(qProb, admm_opts, tas, Mavg);
        Threads.@threads for i = 1:NsubSys
            z[i] = zbar[tas.s[i]:tas.e[i]];
            ggam[i] = gam[tas.s[i]:tas.e[i]];
        end

        T_buildQP += tBuildQP;
        T_setupOSQP += tSetupOSQP;
        T_locSol += tLocSol;
        T_zUpdate += tZupdate;
        T_dual += tDual;
        T_admmIter += tIter;

    end
    end

    

    return (z, ggam, tSqpIter, T_buildQP, T_setupOSQP, T_locSol, T_zUpdate, T_dual, T_admmIter)

end