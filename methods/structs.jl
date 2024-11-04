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

mutable struct derivatives_struct
    evaluator::Vector{MathOptInterface.Nonlinear.Evaluator{MathOptInterface.Nonlinear.ReverseAD.NLPEvaluator}}
    evaluator_eq::Vector{MathOptInterface.Nonlinear.Evaluator{MathOptInterface.Nonlinear.ReverseAD.NLPEvaluator}}
    evaluator_ineq::Vector{MathOptInterface.Nonlinear.Evaluator{MathOptInterface.Nonlinear.ReverseAD.NLPEvaluator}}
    num_variables::Vector{Int64}
    eq_constraints::Vector{Vector{Any}}
    ineq_constraints::Vector{Vector{Any}}
    jacobian_eq_structure::Vector{Vector{Vector{Float64}}}
    jacobian_ineq_structure::Vector{Vector{Vector{Float64}}}
    hessian_structure::Vector{Vector{Vector{Float64}}}
    hessian_lagrangian_structure::Vector{Vector{Vector{Float64}}}
    num_parameters::Vector{Int64}
    rhs_eq::Vector{Vector{Float64}}
    rhs_ineq::Vector{Vector{Float64}}
end

mutable struct sProb_struct
    model::Vector{Model}
    llbx::Vector{Vector{Float64}}
    uubx::Vector{Vector{Float64}}
    AA::Vector{SparseMatrixCSC{Float64, Int64}}
    zz0::Vector{Vector{Float64}}
    pp::Vector{Vector{Float64}}
    derivatives::derivatives_struct
    nnu0::Vector{Vector{Float64}}
    mmu0::Vector{Vector{Float64}}
    ggam0::Vector{Vector{Float64}}
    xJmp::Vector{Vector{VariableRef}}

    # OCP parameters
    N::Int64
    SS::Vector{SparseMatrixCSC{Int64,Int64}}
    SS_x::Vector{SparseMatrixCSC{Int64,Int64}}
end

mutable struct qProb_struct
    # problem parameters
    HH::Vector{SparseMatrixCSC{Float64, Int64}}
    gg::Vector{Vector{Float64}}
    AAeq::Vector{SparseMatrixCSC{Float64, Int64}}
    bbeq::Vector{Vector{Float64}}  
    AAineq::Vector{SparseMatrixCSC{Float64, Int64}}
    ubineq::Vector{Vector{Float64}}
    lbineq::Vector{Vector{Float64}}
    llbx::Vector{Vector{Float64}}
    uubx::Vector{Vector{Float64}} 
    AA::Vector{SparseMatrixCSC{Float64, Int64}}
    
    # solver initialization     
    ggam0::Vector{Vector{Float64}}
    xx0::Vector{Vector{Float64}}

    # OCP parameters
    N::Int64
    SS::Vector{SparseMatrixCSC{Int64}}
end


