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

using JuMP, CPLEX

# solve QP with CPLEX
function run_CPLEX(QP::qProb_struct)
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
    lbx = vcat(QP.llbx...)
    ubx = vcat(QP.uubx...)
    beq = vcat(QP.bbeq...)
    g = vcat(QP.gg...)
    x0 = vcat(QP.xx0...)

    model = Model(CPLEX.Optimizer)
    set_silent(model)
    @variable(model, xJmp[1:sum(nz)])
    set_lower_bound.(xJmp[lbx.!=-Inf], lbx[lbx.!=-Inf]);
    set_upper_bound.(xJmp[ubx.!=Inf], ubx[ubx.!=Inf]);
    @objective(model, Min, 0.5*transpose(xJmp)*H*xJmp + transpose(g)*xJmp);
    @constraint(model, Aeq*xJmp == beq);
    if size(Aineq,1) > 0
        @constraint(model, lbineq .<= Aineq*xJmp .<= ubineq);
    end
    @constraint(model, AA*xJmp == 0);
    set_attribute(model, "CPXPARAM_Threads",Threads.nthreads())
    for idx = 1:size(xJmp,1)
        set_start_value(xJmp[idx], x0[idx])
    end
    optimize!(model) 
    t_cplex = solve_time(model)
    sol = value.(xJmp);

    model = nothing;
    H = nothing;
    g = nothing;
    Aineq = nothing;
    bineq = nothing;
    lbx = nothing;
    ubx = nothing;
    AA = nothing;
    GC.gc(true);
    return sol, t_cplex
end