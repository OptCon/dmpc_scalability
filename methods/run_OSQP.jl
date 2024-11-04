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

function run_OSQP_log(QP,rho,maxiter)

    NsubSys = size(QP.HH,1)
    H = QP.HH[1]
    Aeq = QP.AAeq[1]
    Aineq = QP.AAineq[1]
    for i = 2:NsubSys
        H = blockdiag(H,QP.HH[i])
        Aeq = blockdiag(Aeq, QP.AAeq[i])
        Aineq = blockdiag(Aineq, QP.AAineq[i])
    end

    nnx = Vector{Int}(undef,NsubSys)
    nng = Vector{Int}(undef,NsubSys)
    nnh = Vector{Int}(undef,NsubSys)
    for i = 1:NsubSys
        nnx[i] = size(QP.AA[i],2)
        nng[i] = size(QP.AAeq[i],1)
        nnh[i] = size(QP.AAineq[i],1)
    end

    g = vcat(QP.gg...);
    beq = vcat(QP.bbeq...);
    ubineq = vcat(QP.ubineq...);
    lbineq = vcat(QP.lbineq...);
    uubx = vcat(QP.uubx...);
    llbx = vcat(QP.llbx...);

    E = cat(QP.AA...,dims=2)
    (Ncons,nz) = size(E)
    ng = size(Aeq,1)
    nh = size(Aineq,1)

    z = zeros(nz,maxiter)
    gam = zeros(nz,maxiter)
    nu = Vector{Matrix{Float64}}(undef,NsubSys)
    mu = Vector{Matrix{Float64}}(undef,NsubSys)
    mubox = Vector{Matrix{Float64}}(undef,NsubSys)
    for i = 1:NsubSys
        nu[i] = zeros(nng[i],maxiter)
        mu[i] = zeros(nnh[i],maxiter)
        mubox[i] = zeros(nnx[i],maxiter)
    end
    
    tOSQP = 0
    rescent = 0
    Cent = OSQP.Model()
    OSQP.setup!(Cent; P = H, q=g, A = [Aeq;E;Aineq;sparse(I,nz,nz)], l = [beq;zeros(Ncons);lbineq;llbx], u = [beq;zeros(Ncons);ubineq;uubx],verbose = false, max_iter = 1, check_termination = 0, rho=rho, linsys_solver = "mkl pardiso");
    for k = 1:maxiter    
        tOSQP = tOSQP +  @elapsed begin
        rescent = OSQP.solve!(Cent)
        end
        z[:,k] = rescent.x;
        y = rescent.y;
        nucent = y[1:ng]
        lam = y[ng+1:ng+Ncons]
        mucent = y[ng+Ncons+1:ng+Ncons+nh]
        muboxcent = y[ng+Ncons+nh+1:end]
        for i = 1:NsubSys
            nu[i][:,k] = nucent[ sum(nng[1:i-1])+1:sum(nng[1:i])]
            mu[i][:,k] = mucent[ sum(nnh[1:i-1])+1:sum(nnh[1:i])]
            mubox[i][:,k] = muboxcent[ sum(nnx[1:i-1])+1:sum(nnx[1:i])]                
        end
        gam[:,k] = transpose(E)*lam
    end   
    
    Cent = nothing
    GC.gc(true)

    return (z, gam, nu, mu, mubox, tOSQP)
end


function run_OSQP_benchmark(QP,rho,maxiter)
    NsubSys = size(QP.HH,1)
    H = QP.HH[1]
    Aeq = QP.AAeq[1]
    Aineq = QP.AAineq[1]
    for i = 2:NsubSys
        H = blockdiag(H,QP.HH[i])
        Aeq = blockdiag(Aeq, QP.AAeq[i])
        Aineq = blockdiag(Aineq, QP.AAineq[i])
    end

    nnx = Vector{Int}(undef,NsubSys)
    nng = Vector{Int}(undef,NsubSys)
    nnh = Vector{Int}(undef,NsubSys)
    for i = 1:NsubSys
        nnx[i] = size(QP.AA[i],2)
        nng[i] = size(QP.AAeq[i],1)
        nnh[i] = size(QP.AAineq[i],1)
    end

    g = vcat(QP.gg...);
    beq = vcat(QP.bbeq...);
    ubineq = vcat(QP.ubineq...);
    lbineq = vcat(QP.lbineq...);
    uubx = vcat(QP.uubx...);
    llbx = vcat(QP.llbx...);

    E = cat(QP.AA...,dims=2)
    (Ncons,nz) = size(E)
    ng = size(Aeq,1)
    nh = size(Aineq,1)
    
    z = zeros(nz,1)
    gam = zeros(nz,1)
    nu = Vector{Vector{Float64}}(undef,NsubSys)
    mu = Vector{Vector{Float64}}(undef,NsubSys) # Ax-b <= 0
    mubox = Vector{Vector{Float64}}(undef,NsubSys) # l <= x <= u
        
    tOSQP = 0
    rescent = 0
    Cent = OSQP.Model()
    
    OSQP.setup!(Cent; P = H, q=g, A = [Aeq;E;Aineq;sparse(I,nz,nz)], l = [beq;zeros(Ncons);lbineq;llbx], u = [beq;zeros(Ncons);ubineq;uubx],verbose = false, max_iter = maxiter, check_termination = 0, rho=rho, linsys_solver = "mkl pardiso");
    rescent = OSQP.solve!(Cent)
    tOSQP = rescent.info.solve_time
    z = rescent.x;
    y = rescent.y;
    nucent = y[1:ng]
    lam = y[ng+1:ng+Ncons]
    mucent = y[ng+Ncons+1:ng+Ncons+nh]
    muboxcent = y[ng+Ncons+nh+1:end]
    for i = 1:NsubSys
        nu[i] = nucent[ sum(nng[1:i-1])+1:sum(nng[1:i])]
        mu[i] = mucent[ sum(nnh[1:i-1])+1:sum(nnh[1:i])]
        mubox[i] = muboxcent[ sum(nnx[1:i-1])+1:sum(nnx[1:i])]
    end
    gam = transpose(E)*lam
    
    Cent = nothing
    GC.gc(true)

    return (z, gam, nu, mu, mubox, tOSQP)

end

function prepare_OSQP(QP,rho,maxiter)
    NsubSys = size(QP.HH,1)
    H = QP.HH[1]
    Aeq = QP.AAeq[1]
    Aineq = QP.AAineq[1]
    for i = 2:NsubSys
        H = blockdiag(H,QP.HH[i])
        Aeq = blockdiag(Aeq, QP.AAeq[i])
        Aineq = blockdiag(Aineq, QP.AAineq[i])
    end

    nnx = Vector{Int}(undef,NsubSys)
    nng = Vector{Int}(undef,NsubSys)
    nnh = Vector{Int}(undef,NsubSys)
    for i = 1:NsubSys
        nnx[i] = size(QP.AA[i],2)
        nng[i] = size(QP.AAeq[i],1)
        nnh[i] = size(QP.AAineq[i],1)
    end

    g = vcat(QP.gg...);
    beq = vcat(QP.bbeq...);
    ubineq = vcat(QP.ubineq...);
    lbineq = vcat(QP.lbineq...);
    uubx = vcat(QP.uubx...);
    llbx = vcat(QP.llbx...);

    E = cat(QP.AA...,dims=2)
    (Ncons,nz) = size(E)

    Cent = OSQP.Model()
    
    OSQP.setup!(Cent; P = H, q=g, A = [Aeq;E;Aineq;sparse(I,nz,nz)], l = [beq;zeros(Ncons);lbineq;llbx], u = [beq;zeros(Ncons);ubineq;uubx],verbose = false, max_iter = maxiter, check_termination = 0, rho=rho, linsys_solver = "mkl pardiso");

    return Cent
end


function iterate_OSQP!(Cent,nnx,nng,nnh,Ncons,NsubSys,E)
    ng = sum(nng)
    nh = sum(nnh)
    rescent = OSQP.solve!(Cent)
    tOSQP = rescent.info.solve_time
    z = rescent.x;
    y = rescent.y;
    nucent = y[1:ng]
    lam = y[ng+1:ng+Ncons]
    mucent = y[ng+Ncons+1:ng+Ncons+nh]
    muboxcent = y[ng+Ncons+nh+1:end]
    nu = Vector{Vector{Float64}}(undef,NsubSys)
    mu = Vector{Vector{Float64}}(undef,NsubSys) # lb <= Ax <= ub
    mubox = Vector{Vector{Float64}}(undef,NsubSys) # l <= x <= u
    for i = 1:NsubSys
        nu[i] = nucent[ sum(nng[1:i-1])+1:sum(nng[1:i])]
        mu[i] = mucent[ sum(nnh[1:i-1])+1:sum(nnh[1:i])]
        mubox[i] = muboxcent[ sum(nnx[1:i-1])+1:sum(nnx[1:i])]
    end
    gam = transpose(E)*lam
    return (z, gam, nu, mu, mubox, tOSQP)
end
