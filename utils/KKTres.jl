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

function eval_KKT_residual(QP::qProb_struct, z, gam, nu, mu, mubox)

    NsubSys = size(QP.HH,1)
    nz = Array{Int64}(undef,NsubSys);
    nz[1] = size(QP.HH[1],2);
    H = QP.HH[1]
    Aeq = QP.AAeq[1]
    Aineq = QP.AAineq[1]
    for i = 2:NsubSys
        nz[i] = size(QP.HH[i],2);
        H = blockdiag(H,QP.HH[i])
        Aeq = blockdiag(Aeq, QP.AAeq[i])
        Aineq = blockdiag(Aineq, QP.AAineq[i])
    end
    AA = cat(QP.AA...,dims=2)
    ubineq = vcat(QP.ubineq...)
    lbineq = vcat(QP.lbineq...)
    x0 = vcat(QP.xx0...)
    ubx = vcat(QP.uubx...)
    lbx = vcat(QP.llbx...)
    g = vcat(QP.gg...)
    beq = vcat(QP.bbeq...)

    mu_centralized = vcat(mu...)
    mubox_centralized = vcat(mubox...)
    gradL = H*z + g + transpose(Aeq)*vcat(nu...) + transpose(Aineq)*mu_centralized + mubox_centralized + gam
   
    eq = Aeq*z - beq; #equality constraint error

    inequ = Aineq*z - ubineq;
    ineql = lbineq - Aineq*z;
    inequ[inequ .== -Inf] .= -10^6 
    ineql[ineql .== -Inf] .= -10^6
    muu = zeros(length(mu_centralized))
    mul = zeros(length(mu_centralized))
    muu[mu_centralized .>0] .= mu_centralized[mu_centralized .> 0]
    mul[mu_centralized .<0] .= -mu_centralized[mu_centralized .< 0]
    cinequ = muu .* inequ; # complementarity
    cineql = mul .* ineql; # complementarity
    inequ = inequ[inequ .>= 0]; #inequality constraint error
    ineql = ineql[ineql .>= 0]; #inequality constraint error

    ub = z-ubx;
    lb = lbx - z;
    ub[ub .== -Inf] .= -10^6
    lb[lb .== -Inf] .= -10^6
    muboxu = zeros(length(mubox_centralized))
    muboxl = zeros(length(mubox_centralized))
    muboxu[mubox_centralized .>0] .= mubox_centralized[mubox_centralized .>0]
    muboxl[mubox_centralized .<0] .= -mubox_centralized[mubox_centralized .<0]
    cboxu = muboxu .* ub # complementarity
    cboxl = muboxl .* lb # complementarity
    ub = ub[ub .>= 0]; #upper box constraint error
    lb = lb[lb .>= 0]; #lower box constraint error

    cons = AA*z; #consensus error

    KKT = [gradL; eq; inequ; ineql; ub; lb; cons; cinequ; cineql; cboxu; cboxl];

    res = norm(KKT, Inf)

    return res


end

function eval_KKT_residual_NLP(sProb::sProb_struct, zz, ggam, nu, mu, mubox)
    NsubSys = length(sProb.AA)
    qProb, hasIneqConstraints = buildQP(sProb,zz,ggam,true) # with GN Hessian
    for i = 1:NsubSys
        qProb.HH[i] = eval_GN_hessian(sProb.derivatives.evaluator[i],length(zz[i]),0,sProb.derivatives.hessian_structure[i],zz[i]);
    end
    res = eval_KKT_residual(qProb, vcat(zz...), vcat(ggam...), nu, mu, mubox)

    return res

end
