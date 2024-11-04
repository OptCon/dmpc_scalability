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

function create_sProb(  rw::Int64,N::Int64,aij::Float64,omega0::Float64,p_load::Float64,load_hat::Float64,n::Int64 )
    Random.seed!(4)
    M = 0.167;
    D = 0.045;
    regionSize = rw^2;
    nx = 2
    nxr = nx*regionSize;
    Mrand_base = M*(0.9*ones(regionSize)+0.2*rand(regionSize)) 
    Drand_base = D*(0.9*ones(regionSize)+0.2*rand(regionSize))
    x0_base = zeros(nxr)
    theta0 = 0.0;
    for j = 1:regionSize
        x0_base[(j-1)*nx+1:j*nx] = [-theta0 + 2*theta0*rand(); -omega0 + 2*omega0*rand()];
    end
    isLoad_base = Vector{Bool}(undef, regionSize)
    for l = 1:1000
        for j = 1:regionSize
            if sum(isLoad_base[1:j-1]) >= p_load*regionSize
                isLoad_base[j] = false
            else
                isLoad_base[j] = rand() <= p_load ? 1.0 : 0.0
            end
        end
        if sum(isLoad_base) >= 0.9*p_load*regionSize
            break
        end
    end
    p_load_new = sum(isLoad_base)/regionSize
    @printf "Each subsystem has %i loads of %i buses. Specified p_load = %f, realized p_load = %f\n" sum(isLoad_base) regionSize p_load p_load_new
    p_load = p_load_new
    wload_base = -load_hat*ones(regionSize) + 2* load_hat*rand(regionSize)

    w = rw*n
    Noscillator = w^2;  
    NsubSys::Int = Int64(floor(Noscillator/regionSize))
    x0 = Vector{Vector{Float64}}(undef, NsubSys);

    Mrand = Vector{Vector{Float64}}(undef, NsubSys);
    Drand = Vector{Vector{Float64}}(undef, NsubSys);
    isLoad = Vector{Vector{Bool}}(undef, NsubSys)
    w_loads = Vector{Vector{Float64}}(undef, NsubSys)

    for i = 1:NsubSys
        Mrand[i] = Mrand_base
        Drand[i] = Drand_base
        x0[i] = x0_base;    
        w_loads[i] = wload_base
        isLoad[i] = isLoad_base
    end

    @printf "building sProb now\n"
    buildtime = @elapsed begin sProb = build_sProb_network_nonlinear_hn2(w , rw, N , aij , x0, Mrand, Drand, isLoad, w_loads)
    end
    createDerivativeFun!(sProb,NsubSys);

    return (sProb, p_load)
end