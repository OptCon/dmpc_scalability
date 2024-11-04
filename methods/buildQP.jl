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

function buildQP(sProb::sProb_struct,z::Vector{Vector{Float64}},ggam::Vector{Vector{Float64}},useGN)
    NsubSys = length(sProb.zz0)
    IneqConstraints::Bool = false
    
    qProb = qProb_struct(Vector{SparseMatrixCSC{Float64, Int}}(undef, NsubSys),     # HH
                        Vector{Vector{Float64}}(undef, NsubSys),                    # gg
                        Vector{SparseMatrixCSC{Float64, Int64}}(undef, NsubSys),    # AAeq
                        Vector{Vector{Float64}}(undef, NsubSys),                    # bbeq
                        Vector{SparseMatrixCSC{Float64, Int64}}(undef, NsubSys),    # AAineq
                        Vector{Vector{Float64}}(undef, NsubSys),                    # ubineq
                        Vector{Vector{Float64}}(undef, NsubSys),                    # lbineq
                        Vector{Vector{Float64}}(undef, NsubSys),                    # llbx
                        Vector{Vector{Float64}}(undef, NsubSys),                    # uubx
                        Vector{SparseMatrixCSC{Float64, Int}}(undef, NsubSys),      # AA
                        Vector{Vector{Float64}}(undef, NsubSys),                    # ggam0
                        Vector{Vector{Float64}}(undef, NsubSys),                    # xx0
                        sProb.N,                                                    # N
                        Vector{SparseMatrixCSC{Int64, Int64}}(undef, NsubSys))      # SS

    # other sensitivities
    @inbounds Threads.@threads for i = 1:NsubSys
        qProb.HH[i] = eval_GN_hessian(sProb.derivatives.evaluator[i], sProb.derivatives.num_variables[i], sProb.derivatives.num_parameters[i], sProb.derivatives.hessian_structure[i], z[i])        
        qProb.HH[i] = sparse(0.5*(qProb.HH[i]+transpose(qProb.HH[i])))
        if sProb.derivatives.num_parameters[i] != 0
            qProb.gg[i] = eval_gradient(sProb.derivatives.evaluator[i], sProb.derivatives.num_variables[i], sProb.derivatives.num_parameters[i], z[i]) - qProb.HH[i] * z[i]
            qProb.AAeq[i] = eval_jacobian_constraints(sProb.derivatives.evaluator_eq[i], sProb.derivatives.num_variables[i], sProb.derivatives.num_parameters[i], sProb.derivatives.jacobian_eq_structure[i], length(sProb.derivatives.eq_constraints[i]), z[i])
        else
            qProb.gg[i] = eval_gradient(sProb.derivatives.evaluator[i], sProb.derivatives.num_variables[i], z[i]) - qProb.HH[i] * z[i]
            qProb.AAeq[i] = eval_jacobian_constraints(sProb.derivatives.evaluator_eq[i], sProb.derivatives.num_variables[i], sProb.derivatives.jacobian_eq_structure[i], length(sProb.derivatives.eq_constraints[i]), z[i])
        end      
        
        qProb.bbeq[i] = -get_eq_constraint_value(sProb.derivatives.evaluator_eq[i], sProb.derivatives.eq_constraints[i], sProb.derivatives.num_variables[i], z[i], sProb.derivatives.rhs_eq[i])  + qProb.AAeq[i]*z[i]
        if length(sProb.derivatives.ineq_constraints[i]) > 0
            IneqConstraints = true
            if sProb.derivatives.num_parameters[i] != 0
                qProb.AAineq[i] = eval_jacobian_constraints(sProb.derivatives.evaluator_ineq[i], sProb.derivatives.num_variables[i], sProb.derivatives.num_parameters[i], sProb.derivatives.jacobian_ineq_structure[i], length(sProb.derivatives.ineq_constraints[i]), z[i])
            else
                qProb.AAineq[i] = eval_jacobian_constraints(sProb.derivatives.evaluator_ineq[i], sProb.derivatives.num_variables[i], sProb.derivatives.jacobian_ineq_structure[i], length(sProb.derivatives.ineq_constraints[i]), z[i])
            end
            qProb.ubineq[i] = -get_ineq_constraint_value(sProb.derivatives.evaluator_ineq[i], sProb.derivatives.ineq_constraints[i], sProb.derivatives.num_variables[i], z[i], sProb.derivatives.rhs_ineq[i]) + qProb.AAineq[i]*z[i]
        else
            qProb.AAineq[i] = sparse(Matrix{Any}(undef, 0, size(z[i],1)))
            qProb.ubineq[i] = Vector{Float64}(undef, 0)
        end
        qProb.uubx[i] = sProb.uubx[i]
        qProb.llbx[i] = sProb.llbx[i]
        qProb.xx0[i] = z[i]
        qProb.ggam0[i] = ggam[i]
        qProb.AA[i] = sProb.AA[i]
        qProb.lbineq[i] = -Inf*ones(size(qProb.AAineq[i],1))
    end
    
    return qProb, IneqConstraints
end

function updateQP(qProb::qProb_struct, sProb::sProb_struct,z::Vector{Vector{Float64}},ggam::Vector{Vector{Float64}}, NsubSys::Int64, IneqConstraints::Val{true})
    @inbounds Threads.@threads for i = 1:NsubSys
        if sProb.derivatives.num_parameters[i] != 0
            qProb.gg[i] = eval_gradient(sProb.derivatives.evaluator[i], sProb.derivatives.num_variables[i], sProb.derivatives.num_parameters[i], z[i]) - qProb.HH[i] * z[i]
            qProb.AAeq[i] = eval_jacobian_constraints(sProb.derivatives.evaluator_eq[i], sProb.derivatives.num_variables[i], sProb.derivatives.num_parameters[i], sProb.derivatives.jacobian_eq_structure[i], length(sProb.derivatives.eq_constraints[i]), z[i])
            qProb.AAineq[i] = eval_jacobian_constraints(sProb.derivatives.evaluator_ineq[i], sProb.derivatives.num_variables[i], sProb.derivatives.num_parameters[i], sProb.derivatives.jacobian_ineq_structure[i], length(sProb.derivatives.ineq_constraints[i]), z[i])
        else
            qProb.gg[i] = eval_gradient(sProb.derivatives.evaluator[i], sProb.derivatives.num_variables[i], z[i]) - qProb.HH[i] * z[i]
            qProb.AAeq[i] = eval_jacobian_constraints(sProb.derivatives.evaluator_eq[i], sProb.derivatives.num_variables[i], sProb.derivatives.jacobian_eq_structure[i], length(sProb.derivatives.eq_constraints[i]), z[i])
            qProb.AAineq[i] = eval_jacobian_constraints(sProb.derivatives.evaluator_ineq[i], sProb.derivatives.num_variables[i], sProb.derivatives.jacobian_ineq_structure[i], length(sProb.derivatives.ineq_constraints[i]), z[i])
        end
        qProb.bbeq[i] = -get_eq_constraint_value(sProb.derivatives.evaluator_eq[i], sProb.derivatives.eq_constraints[i], sProb.derivatives.num_variables[i], z[i], sProb.derivatives.rhs_eq[i])  + qProb.AAeq[i]*z[i]
        qProb.ubineq[i] = -get_ineq_constraint_value(sProb.derivatives.evaluator_ineq[i], sProb.derivatives.ineq_constraints[i], sProb.derivatives.num_variables[i], z[i], sProb.derivatives.rhs_ineq[i]) + qProb.AAineq[i]*z[i]
        qProb.xx0[i] = z[i]
        qProb.ggam0[i] = ggam[i]
    end
    return qProb
end

function updateQP(qProb::qProb_struct, sProb::sProb_struct,z::Vector{Vector{Float64}},ggam::Vector{Vector{Float64}}, NsubSys::Int64, IneqConstraints::Val{false})
    @inbounds Threads.@threads for i = 1:NsubSys
        if sProb.derivatives.num_parameters[i] != 0
            qProb.gg[i] = eval_gradient(sProb.derivatives.evaluator[i], sProb.derivatives.num_variables[i], sProb.derivatives.num_parameters[i], z[i]) - qProb.HH[i] * z[i]
            qProb.AAeq[i] = eval_jacobian_constraints(sProb.derivatives.evaluator_eq[i], sProb.derivatives.num_variables[i], sProb.derivatives.num_parameters[i], sProb.derivatives.jacobian_eq_structure[i], length(sProb.derivatives.eq_constraints[i]), z[i])
        else
            qProb.gg[i] = eval_gradient(sProb.derivatives.evaluator[i], sProb.derivatives.num_variables[i], z[i]) - qProb.HH[i] * z[i]
            qProb.AAeq[i] = eval_jacobian_constraints(sProb.derivatives.evaluator_eq[i], sProb.derivatives.num_variables[i], sProb.derivatives.jacobian_eq_structure[i], length(sProb.derivatives.eq_constraints[i]), z[i])
        end
            qProb.bbeq[i] = -get_eq_constraint_value(sProb.derivatives.evaluator_eq[i], sProb.derivatives.eq_constraints[i], sProb.derivatives.num_variables[i], z[i], sProb.derivatives.rhs_eq[i])  + qProb.AAeq[i]*z[i]
        qProb.xx0[i] = z[i]
        qProb.ggam0[i] = ggam[i]
    end
    return qProb
end