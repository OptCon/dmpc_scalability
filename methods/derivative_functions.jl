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

function createDerivativeFun!(sProb::sProb_struct, NsubSys::Int64)
    sProb.derivatives = derivatives_struct(Vector{MathOptInterface.Nonlinear.Evaluator{MathOptInterface.Nonlinear.ReverseAD.NLPEvaluator}}(undef, NsubSys),     # Evaluator
                                        Vector{MathOptInterface.Nonlinear.Evaluator{MathOptInterface.Nonlinear.ReverseAD.NLPEvaluator}}(undef, NsubSys),     # evaluator_eq
                                        Vector{MathOptInterface.Nonlinear.Evaluator{MathOptInterface.Nonlinear.ReverseAD.NLPEvaluator}}(undef, NsubSys),     # evaluator_ineq
                                        Vector{Int}(undef, NsubSys),                         # num_variables
                                        Vector{Vector{Any}}(undef, NsubSys),     # eq_constraints (Any due to compatibiltiy for all types of constraints)
                                        Vector{Vector{Any}}(undef, NsubSys),     # ineq_constraints (Any due to compatibiltiy for all types of constraints)
                                        Vector{Vector{Vector{Float64}}}(undef, NsubSys),     # jacobian_eq_structure
                                        Vector{Vector{Vector{Float64}}}(undef, NsubSys),     # jacobian_ineq_structure
                                        Vector{Vector{Vector{Float64}}}(undef, NsubSys),     # hessian_structure
                                        Vector{Vector{Vector{Float64}}}(undef, NsubSys),     # hessian_lagrangian_structure
                                        Vector{Int}(undef, NsubSys),                         # num_parameters
                                        Vector{Vector{Float64}}(undef, NsubSys),             # rhs_eq
                                        Vector{Vector{Float64}}(undef, NsubSys)              # rhs_ineq
                        )

    for i in 1:NsubSys #multi-threading possible here, but requires a lot of memory for large problems
        sProb.derivatives.evaluator[i],sProb.derivatives.evaluator_eq[i],sProb.derivatives.evaluator_ineq[i], sProb.derivatives.num_variables[i],sProb.derivatives.eq_constraints[i], sProb.derivatives.ineq_constraints[i],jacobian_sparsity_eq,jacobian_sparsity_ineq, sProb.derivatives.num_parameters[i] = setup_evaluator(sProb.model[i])
        sProb.derivatives.hessian_structure[i] = prepare_hessian(sProb.derivatives.evaluator[i])
        sProb.derivatives.hessian_lagrangian_structure[i] = prepare_hessian_lagrangian(sProb.derivatives.evaluator[i])
        sProb.derivatives.jacobian_eq_structure[i] = prepare_jacobian_eq_constraints(sProb.derivatives.evaluator_eq[i], length(sProb.derivatives.eq_constraints[i]), jacobian_sparsity_eq)  
        if length(sProb.derivatives.ineq_constraints[i]) != 0
            sProb.derivatives.jacobian_ineq_structure[i] = prepare_jacobian_ineq_constraints(sProb.derivatives.evaluator_ineq[i], length(sProb.derivatives.ineq_constraints[i]), jacobian_sparsity_ineq)
        end
    end

    nz = Vector{Int}(undef,NsubSys)
    for i = 1:NsubSys #multi-threading possible here, but requires a lot of memory for large problems
        nz[i] = size(sProb.AA[i],2)

        constraints = sProb.derivatives.eq_constraints[i]
        result = zeros(length(constraints))
        MOI.eval_constraint(sProb.derivatives.evaluator_eq[i], result, zeros(nz[i]))
        sProb.derivatives.rhs_eq[i] = zeros(length(constraints))
        for j = 1:length(constraints)
            sProb.derivatives.rhs_eq[i][j] = constraint_object(constraints[j]).set.value
        end

        constraints = sProb.derivatives.ineq_constraints[i]
        result = zeros(length(constraints))
        MOI.eval_constraint(sProb.derivatives.evaluator_ineq[i], result, zeros(nz[i]))
        sProb.derivatives.rhs_ineq[i] = zeros(length(constraints))
        for j = 1:length(constraints)
            sProb.derivatives.rhs_ineq[i][j] = constraint_object(constraints[i]).set.upper
        end

    end



end

# initialize all necessary objects for all related functions
function setup_evaluator(model::Model)
    rows = Any[]
    nlp = MOI.Nonlinear.Model()
    nlp_ineq = MOI.Nonlinear.Model()
    nlp_eq = MOI.Nonlinear.Model()
    # load constraints
    for (F, S) in list_of_constraint_types(model)
        if !(S == MathOptInterface.Parameter{Float64})
            for ci in all_constraints(model, F, S)
                push!(rows, ci)
                object = constraint_object(ci)
                MOI.Nonlinear.add_constraint(nlp, object.func, object.set)
                if S == MathOptInterface.EqualTo{Float64}
                    MOI.Nonlinear.add_constraint(nlp_eq, object.func, object.set)
                elseif S == MathOptInterface.LessThan{Float64}
                    MOI.Nonlinear.add_constraint(nlp_ineq, object.func, object.set)
                end
            end
        end
    end
    # set objective
    MOI.Nonlinear.set_objective(nlp, objective_function(model))
    x = all_variables(model)
    num_parameters = num_constraints(model, VariableRef, MOI.Parameter{Float64})
    backend = MOI.Nonlinear.SparseReverseMode()
    evaluator = MOI.Nonlinear.Evaluator(nlp, backend, index.(x))
    MOI.initialize(evaluator, [:Hess, :Grad, :Jac])
    evaluator_eq = MOI.Nonlinear.Evaluator(nlp_eq, backend, index.(x))
    MOI.initialize(evaluator_eq, [:Jac])
    jacobian_sparsity_eq = MOI.jacobian_structure(evaluator_eq)
    evaluator_ineq = MOI.Nonlinear.Evaluator(nlp_ineq, backend, index.(x))
    MOI.initialize(evaluator_ineq, [:Jac])
    jacobian_sparsity_ineq = MOI.jacobian_structure(evaluator_ineq)
    num_variables = length(x)
    eq_constraints = vcat(all_constraints(model, NonlinearExpr, MathOptInterface.EqualTo{Float64}),
    all_constraints(model, QuadExpr, MathOptInterface.EqualTo{Float64})[:],
    all_constraints(model, AffExpr, MathOptInterface.EqualTo{Float64})[:])
    ineq_constraints = vcat(all_constraints(model, QuadExpr, MathOptInterface.LessThan{Float64}),
    all_constraints(model, AffExpr, MathOptInterface.LessThan{Float64})[:],
    all_constraints(model, NonlinearExpr, MathOptInterface.LessThan{Float64})[:])
    return evaluator, evaluator_eq, evaluator_ineq, num_variables, eq_constraints, ineq_constraints, jacobian_sparsity_eq, jacobian_sparsity_ineq, num_parameters
end

# get hessian sparsity necessary for evaluation
function prepare_hessian(evaluator::MathOptInterface.Nonlinear.Evaluator{MathOptInterface.Nonlinear.ReverseAD.NLPEvaluator})
    hessian_sparsity = MOI.hessian_objective_structure(evaluator)
    I = [i for (i, _) in hessian_sparsity]
    J = [j for (_, j) in hessian_sparsity]
    V = zeros(length(hessian_sparsity))
    return [I,J,V]
end

# get hessian_lagrangian sparsity necessary for evaluation (not tested yet)
function prepare_hessian_lagrangian(evaluator::MathOptInterface.Nonlinear.Evaluator{MathOptInterface.Nonlinear.ReverseAD.NLPEvaluator})
    hessian_sparsity = MOI.hessian_lagrangian_structure(evaluator)
    I = [i for (i, _) in hessian_sparsity]
    J = [j for (_, j) in hessian_sparsity]
    V = zeros(length(hessian_sparsity))
    return [I,J,V]
end

# return hessian evaluated at xx
function eval_GN_hessian(evaluator::MathOptInterface.Nonlinear.Evaluator{MathOptInterface.Nonlinear.ReverseAD.NLPEvaluator}, num_variables::Int64, num_parameters::Int64, hessian_structure::Vector{Vector{Float64}}, xx::Vector{Float64})
    I,J,V = hessian_structure
    MOI.eval_hessian_objective(evaluator, V, vcat(xx, zeros(num_parameters)))
    H = SparseArrays.sparse(I, J, V, num_variables, num_variables)
    H = Matrix(fill_off_diagonal(H))
    # drop parameters
    H = H[begin:size(H,1)-num_parameters, begin:size(H,1)-num_parameters] 

    return H
end

# return hessian_lagrangian
function eval_hessian_lagrangian(evaluator::MathOptInterface.Nonlinear.Evaluator{MathOptInterface.Nonlinear.ReverseAD.NLPEvaluator}, num_variables::Int64, num_parameters::Int64, hessian_lagrangian_structure::Vector{Vector{Float64}}, xx::Vector{Float64}, sigma::Float64, my::Vector{Float64})
    I,J,V = hessian_lagrangian_structure

    MOI.eval_hessian_lagrangian(evaluator, V, vcat(xx, zeros(num_parameters)), 1.0, my)

    
    H = SparseArrays.sparse(I, J, V, num_variables, num_variables)
    H = Matrix(fill_off_diagonal(H))

    # drop parameters
    H = H[begin:size(H,1)-num_parameters, begin:size(H,1)-num_parameters] 

    return H
end

# fill off diagonal of hessian (eval hessian only returns upper half)
function fill_off_diagonal(H::SparseMatrixCSC{Float64, Int64})
    ret = H + H'
    row_vals = SparseArrays.rowvals(ret)
    non_zeros = SparseArrays.nonzeros(ret)
    for col in 1:size(ret, 2)
        for i in SparseArrays.nzrange(ret, col)
            if col == row_vals[i]
                non_zeros[i] /= 2
            end
        end
    end
    return ret
end

# evaluate opbjective gradient at point xx without parameters
function eval_gradient(evaluator::MathOptInterface.Nonlinear.Evaluator{MathOptInterface.Nonlinear.ReverseAD.NLPEvaluator}, num_variables::Int64, xx::Vector{Float64})
    grad = Vector{Float64}(undef, num_variables)
    MOI.eval_objective_gradient(evaluator, grad, xx)
    # drop parameters
    return grad
end

# evaluate opbjective gradient at point xx
function eval_gradient(evaluator::MathOptInterface.Nonlinear.Evaluator{MathOptInterface.Nonlinear.ReverseAD.NLPEvaluator}, num_variables::Int64, num_parameters::Int64, xx::Vector{Float64})
    grad = Vector{Float64}(undef, num_variables)
    MOI.eval_objective_gradient(evaluator, grad, xx)
    
    grad = zeros(num_variables)
    MOI.eval_objective_gradient(evaluator, grad, vcat(xx, zeros(num_parameters)))
    # drop parameters
    return grad[begin:size(grad,1)-num_parameters]
end

# get sparsity pattern of jacobian of eq constraints (necessary for evaluation)
function prepare_jacobian_eq_constraints(evaluator::MathOptInterface.Nonlinear.Evaluator{MathOptInterface.Nonlinear.ReverseAD.NLPEvaluator}, num_eq_constraints::Int64, jacobian_sparsity_eq::Vector{Tuple{Int64, Int64}})
    I = [i for (i, _) in jacobian_sparsity_eq if i <= num_eq_constraints]
    J = [j for (i, j) in jacobian_sparsity_eq if i <= num_eq_constraints]
    V = zeros(length(jacobian_sparsity_eq))
    return [I,J,V]
end

# get sparsity pattern of jacobian of ineq constraints (necessary for evaluation)
function prepare_jacobian_ineq_constraints(evaluator::MathOptInterface.Nonlinear.Evaluator{MathOptInterface.Nonlinear.ReverseAD.NLPEvaluator}, num_ineq_constraints::Int64, jacobian_sparsity_ineq::Vector{Tuple{Int64, Int64}})
    I = [i for (i, _) in jacobian_sparsity_ineq if i <= num_ineq_constraints]
    J = [j for (i, j) in jacobian_sparsity_ineq if i <= num_ineq_constraints]
    V = zeros(length(jacobian_sparsity_ineq))
    return [I,J,V]
end

# evaluate constraint jacobian in xx, can be utilized for eq and ineq constraints depending on arguments
function eval_jacobian_constraints(evaluator::MathOptInterface.Nonlinear.Evaluator{MathOptInterface.Nonlinear.ReverseAD.NLPEvaluator}, num_variables::Int64, num_parameters::Int64, jacobian_structure::Vector{Vector{Float64}}, num_constraints::Int64, xx::Vector{Float64})
    I,J,V = jacobian_structure
    MOI.eval_constraint_jacobian(evaluator, V, vcat(xx, ones(num_parameters)))

    H = SparseArrays.sparse(I, J, V[1:length(I)], num_constraints, num_variables)
    H = Matrix(H)
    # drop parameters
    return sparse(H[:,begin:size(H,2)-num_parameters])
end

# evaluate constraint jacobian in xx, can be utilized for eq and ineq constraints depending on arguments (no params)
function eval_jacobian_constraints(evaluator::MathOptInterface.Nonlinear.Evaluator{MathOptInterface.Nonlinear.ReverseAD.NLPEvaluator}, num_variables::Int64, jacobian_structure::Vector{Vector{Float64}}, num_constraints::Int64, xx::Vector{Float64})
    I,J,V = jacobian_structure
    MOI.eval_constraint_jacobian(evaluator, V, xx)
    return SparseArrays.sparse(I, J, V[1:length(I)], num_constraints, num_variables)
end


# evaluate eq constraints in xx
function get_eq_constraint_value(evaluator::MathOptInterface.Nonlinear.Evaluator{MathOptInterface.Nonlinear.ReverseAD.NLPEvaluator}, constraints::Vector{Any}, num_variables::Int64, xx::Vector{Float64}, rhs_eq::Vector{Float64}) # constraints::Vector{ConstraintRef{Model, MathOptInterface.ConstraintIndex{MathOptInterface.ScalarAffineFunction{Float64}, MathOptInterface.EqualTo{Float64}}, ScalarShape}}
    result = zeros(length(constraints))
    MOI.eval_constraint(evaluator, result, vcat(xx, zeros(num_variables - length(vcat(xx[:])))))
    result = result - rhs_eq
    return result
end

# evaluate ineq constraints in xx
function get_ineq_constraint_value(evaluator::MathOptInterface.Nonlinear.Evaluator{MathOptInterface.Nonlinear.ReverseAD.NLPEvaluator}, constraints::Vector{Any}, num_variables::Int64, xx::Vector{Float64}, rhs_ineq::Vector{Float64})
    result = zeros(length(constraints))
    MOI.eval_constraint(evaluator, result, vcat(xx, zeros(num_variables - length(vcat(xx[:])))))
    result = result - rhs_ineq
    return result
end