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

using MadNLP, MadNLPPardiso

function run_madNLP(sProb::sProb_struct, max_wall_time::Float64)
    centralized_model, xJmp = create_centralized_madnlp_model(sProb)
    NsubSys = length(sProb.AA)

    # set constraints to empty functions / default initial guess
    sProb   = setDefaultValsMadNLP(sProb)

    for i=1:NsubSys
        nx = length(sProb.zz0[i])

        if isempty(sProb.llbx[i])
            sProb.llbx[i] = -inf(nx,1)
        end
        if isempty(sProb.uubx[i])
            sProb.uubx[i] = inf(nx,1)
        end
    end
    
    # concatenate... 
    A     = hcat(sProb.AA...)
    llbx = vcat(sProb.llbx...)
    uubx = vcat(sProb.uubx...)
    x0 = vcat(sProb.zz0...)

    set_silent(centralized_model)
    set_lower_bound.(xJmp[llbx.!=-Inf], llbx[llbx.!=-Inf]);
    set_upper_bound.(xJmp[uubx.!=Inf], uubx[uubx.!=Inf]);
    @constraint(centralized_model, A*xJmp == 0)
    set_attribute(centralized_model, "pardisomkl_num_threads", Threads.nthreads())
    set_attribute(centralized_model, "max_wall_time", max_wall_time)
    for idx = 1:size(xJmp,1)
        set_start_value(xJmp[idx], x0[idx])
    end
    optimize!(centralized_model)
    sol = value.(xJmp);
    t_madnlp = solve_time(centralized_model)
    iter_madnlp = barrier_iterations(centralized_model)

    return sol, t_madnlp, iter_madnlp
end

# used to fuse JuMP models for the individual subsystems into one large model
function create_centralized_madnlp_model(sProb::sProb_struct)
    centralized_model = Model(()->MadNLP.Optimizer(linear_solver=PardisoMKLSolver))
    # centralized_model = Model(MadNLP.Optimizer)
    # Dictionary to map old variables to new variables in the large model
    var_map = Dict{VariableRef, VariableRef}()

    objective = 0

    xJmp = Vector{VariableRef}(undef, length(vcat(sProb.zz0...)))
    idx = 1
        
    for model in sProb.model
        param_variables = [constraint_object(constraint).func for constraint in all_constraints(model, VariableRef, MathOptInterface.Parameter{Float64})]
        param_values = [constraint_object(constraint).set.value for constraint in all_constraints(model, VariableRef, MathOptInterface.Parameter{Float64})]

        # Extract variables and create corresponding variables in the large model
        for v in all_variables(model)
            # Create a new variable in the large model with the same bounds and attributes
            if v in param_values
                param_index = findall(x->x==v, param_variables)
                new_var = @variable(model, set = Parameter(param_values[param_index][1]))
                var_map[v] = new_var
            else
                new_var = @variable(centralized_model)
                var_map[v] = new_var
            end
        end

        for (func_type, set_type) in list_of_constraint_types(model)
            for c in all_constraints(model, func_type, set_type)
                # Get the function and set from the original constraint
                expr = constraint_object(c).func
                sense = constraint_object(c).set

                if !(sense isa MathOptInterface.Parameter)
                    # Replace old variables with new variables in the constraint function
                    new_expr = substitute_variables(expr, var_map)
                    # Add the constraint to the large model
                    if sense isa MathOptInterface.LessThan
                        @constraint(centralized_model, new_expr <= sense.upper)
                    elseif sense isa MathOptInterface.GreaterThan
                        @constraint(centralized_model, new_expr >= sense.lower)
                    elseif sense isa MathOptInterface.Interval
                        @constraint(centralized_model, sense.lower <= new_expr <= sense.upper)    
                    elseif sense isa MathOptInterface.EqualTo
                        @constraint(centralized_model, new_expr == sense.value)
                    else
                        error("Unsupported constraint type $(typeof(sense))")
                    end
                end
            end
        end
        objective += substitute_variables(objective_function(model), var_map)
        @objective(centralized_model, Min, objective)
    end

    # get xJmp
    for xJmp_submodel in sProb.xJmp
        for var in xJmp_submodel
            xJmp[idx] = var_map[var]
            idx +=1
        end
    end
    return centralized_model, xJmp
end

# map variables of subsystems to the variables of the large model and reconstruct constraints / expressions with the new variables
function substitute_variables(expr, var_map)
    if expr isa VariableRef
        new_var1 = var_map[expr]
        return new_var1
    elseif expr isa Float64
        return expr
    elseif expr isa AffExpr
        new_aff = AffExpr()
        for var in expr.terms.keys
            coeff = coefficient(expr, var)
            new_var = var_map[var]
            new_aff += coeff * new_var
        end
        new_aff += constant(expr)
        return new_aff
    elseif expr isa QuadExpr
        new_quad = QuadExpr()
        for term in expr.terms.keys
            var1 = term.a
            var2 = term.b
            coeff = coefficient(expr, var1, var2)
            new_var1 = var_map[var1]
            new_var2 = var_map[var2]
            new_quad += coeff * new_var1 * new_var2
        end
        for var in expr.aff.terms.keys
            coeff = coefficient(expr, var)
            new_var = var_map[var]
            new_quad += coeff * new_var
        end
        new_quad += constant(expr)
        return new_quad
    elseif expr isa NonlinearExpr
        operator, args = expr.head, expr.args
        if length(expr.args) == 1
            new_exprs = GenericNonlinearExpr{VariableRef}(
                operator,
                substitute_variables(args[1], var_map),
            )
        else
            new_exprs = GenericNonlinearExpr{VariableRef}(
                operator,
                substitute_variables(args[1], var_map),
                substitute_variables(args[2], var_map),
            )
        end
        return new_exprs
    elseif expr isa Vector{NonlinearExpr}
        # handle a vector of NonlinearExpr
        new_exprs = [substitute_variables(sub_expr, var_map) for sub_expr in expr]
        return new_exprs
    elseif expr isa Float64
        return expr
    else
        error("Unsupported expression type $(typeof(expr))")
    end
end

function setDefaultValsMadNLP(sProb::sProb_struct)
    NsubSys = length(sProb.AA)

    # set default primal values to zero if not set
    if isnothing(sProb.zz0)
        for i=1:NsubSys
            nxi          = size(sProb.AA[i],2)
            sProb.zz0[i] = zeros(nxi)
        end
    end

    # set default lower/upper bounds if not present
    if isnothing(sProb.llbx)
        for i=1:NsubSys
            nxi           = size(sProb.AA[i],2)        
            sProb.llbx[i] = -inf*ones(nxi)
        end
    end

    # set default lower/upper bounds if not present
    if isnothing(sProb.uubx)
        for i=1:NsubSys
            nxi           = size(sProb.AA[i],2)       
            sProb.uubx[i] = Inf*ones(nxi)
        end
    end
    return sProb
end