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

mutable struct thread_admm_avg_struct
    m::Vector{OSQP.Model}
    s::Vector{Int64}
    e::Vector{Int64}
    res::Vector{OSQP.Results}
    nz::Vector{Int64}
end

mutable struct admm_opts_struct
    maxiter::Int64
    rho::Float64
    create_log::Bool #log zbar iterates
    loc_tol::Float64
end

function run_ADMM_log(QP::qProb_struct, opts::admm_opts_struct, Mavg::SparseMatrixCSC{Float64,Int64}) #, usol, S, tol)

    tSetup = @elapsed begin tas = prepare_ADMM(QP::qProb_struct, opts.rho, opts.loc_tol)
    end

    (zbar_log, gam_log, nu_log, mu_log, mubox_log, tIter, tLocSol, tZupdate, tDual, opts.maxiter) = iterate_ADMM_log(QP::qProb_struct, opts::admm_opts_struct, tas::thread_admm_avg_struct, Mavg::SparseMatrixCSC{Float64,Int64})

    return (zbar_log, gam_log, nu_log, mu_log, mubox_log, tIter, tLocSol, tZupdate, tDual, opts.maxiter, tSetup)

end

function run_ADMM_benchmark(QP::qProb_struct, opts::admm_opts_struct, Mavg::SparseMatrixCSC{Float64,Int64}) #, usol, S, tol)

    tSetup = @elapsed begin tas = prepare_ADMM(QP::qProb_struct, opts.rho, opts.loc_tol)
    end

    (zbar, gam, tIter, tLocSol, tZupdate, tDual, opts.maxiter) = iterate_ADMM_benchmark(QP::qProb_struct, opts::admm_opts_struct, tas::thread_admm_avg_struct, Mavg::SparseMatrixCSC{Float64,Int64})

    NsubSys = length(QP.AA)
    y = Vector{Vector{Float64}}(undef,NsubSys)
    nu = Vector{Vector{Float64}}(undef,NsubSys)
    mu = Vector{Vector{Float64}}(undef,NsubSys) # Ax-b <= 0
    mubox = Vector{Vector{Float64}}(undef,NsubSys) # l <= x <= u
    for i = 1:NsubSys
        y[i] = tas.res[i].y
        nu[i] = y[i][1:size(QP.AAeq[i],1)];
        mu[i] = y[i][size(QP.AAeq[i],1)+1: size(QP.AAeq[i],1) + size(QP.AAineq[i],1)];
        mubox[i] = y[i][size(QP.AAeq[i],1) + size(QP.AAineq[i],1)+1:end];
    end

    return (zbar, gam, tIter, tLocSol, tZupdate, tDual, opts.maxiter, tSetup, nu, mu, mubox)

end

function iterate_ADMM_log(QP::qProb_struct, opts::admm_opts_struct, tas::thread_admm_avg_struct, Mavg::SparseMatrixCSC{Float64,Int64})

    tLocSol::Float64 = 0.0
    tZupdate::Float64 = 0.0
    tDual::Float64 = 0.0
    NsubSys::Int64 = length(QP.AA);

    gam::Vector{Float64} = vcat(QP.ggam0...);
    zbar::Vector{Float64} = vcat(QP.xx0...);
    zz  = Vector{Vector{Float64}}(undef,NsubSys);
    
    rho = opts.rho;
    zbar_log::Matrix{Float64} = zeros(sum(tas.nz),opts.maxiter);
    gam_log::Matrix{Float64} = zeros(sum(tas.nz),opts.maxiter);
    nu_log = Vector{Matrix{Float64}}(undef,NsubSys)
    mu_log = Vector{Matrix{Float64}}(undef,NsubSys)
    mubox_log = Vector{Matrix{Float64}}(undef,NsubSys)
    y = Vector{Vector{Float64}}(undef,NsubSys)

    for i = 1:NsubSys
        nu_log[i] = zeros( size(QP.AAeq[i],1), opts.maxiter );
        mu_log[i] = zeros( size(QP.AAineq[i],1), opts.maxiter );
        mubox_log[i] = zeros( size(QP.HH[i],1), opts.maxiter);
    end

            
    tIter = @elapsed begin for iter = 1:opts.maxiter
        # z update
        tLocSol = tLocSol + @elapsed begin Threads.@threads for i = 1:NsubSys
            OSQP.update_q!(tas.m[i], (QP.gg[i]+gam[tas.s[i]:tas.e[i]]-rho*zbar[tas.s[i]:tas.e[i]]))
            tas.res[i] = OSQP.solve!(tas.m[i])
            zz[i] = tas.res[i].x
        end
        end
        z = cat(zz...,dims=1)
        
        # zbar update
        tZupdate = tZupdate + @elapsed begin
            zbar = Mavg*z;           
        end
            
        # dual update
        tDual = tDual + @elapsed begin Threads.@threads for i = 1:NsubSys
            gam[tas.s[i]:tas.e[i]] = gam[tas.s[i]:tas.e[i]] + rho*(z[tas.s[i]:tas.e[i]] - zbar[tas.s[i]:tas.e[i]])
        end
        end

        if opts.create_log
            zbar_log[:,iter] = zbar
            gam_log[:,iter] = gam
            for i = 1:NsubSys
                y[i] = tas.res[i].y
                nu_log[i][:,iter] = y[i][1:size(QP.AAeq[i],1)];
                mu_log[i][:,iter] = y[i][size(QP.AAeq[i],1)+1: size(QP.AAeq[i],1) + size(QP.AAineq[i],1)];
                mubox_log[i][:,iter] = y[i][size(QP.AAeq[i],1) + size(QP.AAineq[i],1)+1:end];
            end
        end        
    end
    end

return (zbar_log, gam_log, nu_log, mu_log, mubox_log, tIter, tLocSol, tZupdate, tDual, opts.maxiter)


end

function iterate_ADMM_benchmark(QP::qProb_struct, opts::admm_opts_struct, tas::thread_admm_avg_struct, Mavg::SparseMatrixCSC{Float64,Int64})

    tLocSol::Float64 = 0.0
    tZupdate::Float64 = 0.0
    tDual::Float64 = 0.0
    NsubSys::Int64 = length(QP.AA);

    gam::Vector{Float64} = vcat(QP.ggam0...);
    zbar::Vector{Float64} = vcat(QP.xx0...);
    zz  = Vector{Vector{Float64}}(undef,NsubSys);

    rho::Float64 = opts.rho;
            
    tIter = @elapsed begin for iter = 1:opts.maxiter
        # z update
        tLocSol = tLocSol + @elapsed begin Threads.@threads for i = 1:NsubSys
            OSQP.update_q!(tas.m[i], (QP.gg[i]+gam[tas.s[i]:tas.e[i]]-rho*zbar[tas.s[i]:tas.e[i]]))
            tas.res[i] = OSQP.solve!(tas.m[i])
            zz[i] = tas.res[i].x
        end
        end
        z = cat(zz...,dims=1)
        
        # zbar update
        tZupdate = tZupdate + @elapsed begin
            zbar = Mavg*z;          
        end
            
        # dual update
        tDual = tDual + @elapsed begin Threads.@threads for i = 1:NsubSys
            gam[tas.s[i]:tas.e[i]] = gam[tas.s[i]:tas.e[i]] + rho*(z[tas.s[i]:tas.e[i]] - zbar[tas.s[i]:tas.e[i]])
        end
        end
   
    end
    end

return (zbar, gam, tIter, tLocSol, tZupdate, tDual, opts.maxiter)


end


function construct_ADMM_avg(AA::Vector{SparseMatrixCSC{Float64, Int64}})
# manually setup the ADMM averaging matrix for two-assigned partially separable programs
    NsubSys = size(AA,1);
    nz = Vector{Int64}(undef,NsubSys);
    s = Vector{Int64}(undef,NsubSys); # start idx
    e = Vector{Int64}(undef,NsubSys); # end idx

    start_idx = 1
    for i = 1:NsubSys
        nz[i] = size(AA[i],2)
        s[i] = start_idx
        e[i] = start_idx + nz[i] - 1
        start_idx = start_idx + nz[i]
    end
    E = sparse(cat(AA...,dims=2))
    
    n = size(E,2)

    Mavgi = Vector{SparseMatrixCSC{Float64, Int64}}(undef,NsubSys)    
    Threads.@threads for i = 1:NsubSys
        idx0 = sum(nz[1:i-1])
        Mavgi[i] = spzeros(nz[i],n)
        Mavgi[i][:,idx0+1:idx0+nz[i]] = spdiagm(ones(nz[i]));
        for k = 1:nz[i]
            og_constraints = findall(x->x==1, E[:,idx0+k])
            if !isempty(og_constraints)
                copy_indices = [];
                for o = 1:length(og_constraints)
                    j = findfirst(x->x==-1,E[og_constraints[o],:])
                    push!(copy_indices,j)
                end
                den::Int64 = 1 + length(copy_indices)
                Mavgi[i][k,idx0+k] = 1/den;
                for l = 1:length(copy_indices)
                    Mavgi[i][k,copy_indices[l]] = 1/den;
                end
            end
            copy_constraint = findfirst(x->x==-1, E[:,idx0+k])
            if copy_constraint !== nothing
                oj = findfirst(x->x==1,E[copy_constraint,:]) #original variable

                og_constraints = findall(x->x==1, E[:,oj])
                if !isempty(og_constraints)
                    copy_indices = [];
                    for o = 1:length(og_constraints)
                        j = findfirst(x->x==-1,E[og_constraints[o],:])
                        push!(copy_indices,j)
                    end
                    den = 1 + length(copy_indices)
                    Mavgi[i][k,oj] = 1/den;
                    for l = 1:length(copy_indices)
                        Mavgi[i][k,copy_indices[l]] = 1/den;
                    end
                end
            end
        end
    end
    Mavg = sparse(cat(Mavgi...,dims=1));
    

    return Mavg

end




function prepare_ADMM(QP::qProb_struct, rho, locTol)
    NsubSys = size(QP.HH,1);
    m = Vector{OSQP.Model}(undef,NsubSys);
    nz = Vector{Int64}(undef,NsubSys);
    s = Vector{Int64}(undef,NsubSys); # start idx
    e = Vector{Int64}(undef,NsubSys); # end idx
    res = Vector{OSQP.Results}(undef,NsubSys)

    start_idx = 1
    for i = 1:NsubSys
        nz[i] = size(QP.HH[i],1)
        s[i] = start_idx
        e[i] = start_idx + nz[i] - 1
        start_idx = start_idx + nz[i]
    end

    # check problem formulation
    Threads.@threads for i = 1:NsubSys
        if size(QP.AAineq[i],1) == 0
            QP.AAineq[i] = spzeros(0,nz[i]);
            QP.ubineq[i] = spzeros(0);
            QP.lbineq[i] = spzeros(0);
        end
    end

    tSetupOsqp = @elapsed begin Threads.@threads for i = 1:NsubSys
    
        # Create an OSQP model
        m[i] = OSQP.Model()
        OSQP.setup!(m[i]; P=QP.HH[i]+rho*sparse(I,nz[i],nz[i]), q=(QP.gg[i]+QP.ggam0[i]-rho*QP.xx0[i]), A=[QP.AAeq[i]; QP.AAineq[i]; sparse(I,nz[i],nz[i])], l=[QP.bbeq[i]; QP.lbineq[i]; QP.llbx[i]], u=[QP.bbeq[i]; QP.ubineq[i]; QP.uubx[i]], warm_starting=true, verbose = false, eps_abs = locTol, eps_rel = locTol, eps_prim_inf = locTol, eps_dual_inf = locTol) 
        res[i] = OSQP.solve!(m[i])
    
        # Check solver status
        if res[i].info.status != :Solved
            error("OSQP did not solve the problem!")
        end
    
    end
    end

    print("tSetupOsqp = $tSetupOsqp\n");

    tas = thread_admm_avg_struct(m,s,e,res,nz);
    return tas;
end
