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

function ef1(ode::Function,h::Float64,x,u, w::Int64, rw::Int64, Mrand, Drand, aij, NsubSys )
    k1 = ode(x,u, w, rw, Mrand, Drand, aij, NsubSys);
    return (x + h*k1);    
end

function hn2(ode::Function,h::Float64,x,u, w::Int64, rw::Int64, Mrand, Drand, aij, NsubSys )
    k1 = ode(x,u, w, rw, Mrand, Drand, aij, NsubSys);
    k2 = ode(x+h*k1,u, w, rw, Mrand, Drand, aij, NsubSys);
    return (x + h/2 * (k1 + k2));    
end

function rk4(ode::Function,h::Float64,x,u, w::Int64, rw::Int64, Mrand, Drand, aij, NsubSys )
    k1 = ode(x,u, w, rw, Mrand, Drand, aij, NsubSys);
    k2 = ode(x+h/2*k1,u, w, rw, Mrand, Drand, aij, NsubSys);
    k3 = ode(x+h/2*k2,u, w, rw, Mrand, Drand, aij, NsubSys);
    k4 = ode(x+h*k3,u, w, rw, Mrand, Drand, aij, NsubSys);
    return (x + h/6 * (k1 + 2*k2 + 2*k3 + k4));    
end

function hn2_coupled(ode::Function,h::Float64, x::Vector{VariableRef}, u::Vector{VariableRef}, nx::Int64, w::Int64, rw::Int64, xNeighbor::Any, region_number::Int64, Mrand, Drand, aij)
    k1 = ode(x,u,nx, w, rw, Mrand, Drand, aij, xNeighbor, region_number);
    k2 = ode(x+h*k1,u,nx, w, rw, Mrand, Drand, aij, xNeighbor, region_number);
    return (x + h/2 * (k1+ k2));    
end

function rk4_coupled(ode::Function,h::Float64, x::Vector{VariableRef}, u::Vector{VariableRef}, nx::Int64, w::Int64, rw::Int64, xNeighbor::Any, region_number::Int64, Mrand, Drand, aij)
    k1 = ode(x,u,nx, w, rw, Mrand, Drand, aij, xNeighbor, region_number);
    k2 = ode(x+h/2*k1,u,nx, w, rw, Mrand, Drand, aij, xNeighbor, region_number);
    k3 = ode(x+h/2*k2,u,nx, w, rw, Mrand, Drand, aij, xNeighbor, region_number);
    k4 = ode(x+h*k3,u,nx, w, rw, Mrand, Drand, aij, xNeighbor, region_number);
    return (x + h/6 * (k1+ 2*k2 + 2*k3 + k4));    
end

function hn2_LTI(Ac::Matrix{Float64}, Bc::Matrix{Float64}, h::Float64)
    nx = size(Ac,1)    
    Ad = Matrix(I,nx,nx) + h*Ac + h^2*Ac^2/2
    Bd = h*Bc + h^2*Ac*Bc/2
    return (Ad,Bd)
end

function rk4_LTI(Ac::Matrix{Float64}, Bc::Matrix{Float64}, h::Float64)
    nx = size(Ac,1)    
    Ad = Matrix(I,nx,nx) + h*Ac + h^2*Ac^2/2 + h^3*Ac^3/6 + h^4*Ac^4/24
    Bd = h*Bc + h^2*Ac*Bc/2 + h^3*Ac^2*Bc/6 + h^4*Ac^3*Bc/24
    return (Ad,Bd)
end