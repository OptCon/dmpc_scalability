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

function hn2_sim_lin(h::Float64,x,u, w::Int64, rw::Int64, Mrand, Drand, aij, NsubSys)
    lx = length(x)
    nx = 2
    nu = 1
    regionSize = rw^2;
    nxr = nx*regionSize
    nur = nu*regionSize
    xnext = Vector{Vector{Float64}}(undef, NsubSys)
    xcurr = Vector{Vector{Float64}}(undef, NsubSys)
    ucurr = Vector{Vector{Float64}}(undef, NsubSys)
    xNeighbor = Vector{Vector{Vector{Float64}}}(undef,NsubSys)
    ow = Int64(floor(w/rw))  # Number of regions per outer border


    for i = 1:NsubSys
        xcurr[i] = x[ nxr*(i-1)+1 : nxr*i];
        ucurr[i] = u[ nur*(i-1)+1 : nur*i];
    end

    for i = 1:NsubSys
        Ae = spzeros(nxr, lx)
        As = spzeros(nxr, lx)
        Aw = spzeros(nxr, lx)
        An = spzeros(nxr, lx)
        #east
        for j = 1:regionSize
            if mod(j,rw) != 0 # has neighboring oscillator on east within the same region

            else
                if mod(i,ow) != 0 # has neighboring oscillator is in neighboring region
                    k = (i-1)*nxr + (j-1)*nx + 1 + nxr- rw*nx + nx
                    Ae[ (j-1)*nx + nx, k] = 1; #omega, theta_j
                end
            end
        end

        #south
        for j = 1:regionSize
            if j <= regionSize - rw # has neighbor on south within the same region

            else
                if i <= NsubSys - ow # has neighboring oscillator in neighboring region on south
                    k = (i-1)*nxr + (j-1)*nx + 1 + ow*nxr + rw*nx -nxr
                    As[ (j-1)*nx + nx, k] = 1; #omega, theta_j
                end
            end
        end

        #west
        for j = 1:regionSize
            if mod(j,rw) != 1 # has neighbor within region on west
            
            else
                if mod(i,ow) != 1 # has neighboring region on west
                    k = (i-1)*nxr + (j-1)*nx + 1 - nxr + rw*nx - nx
                    Aw[ (j-1)*nx + nx, k] = 1; #omega, theta_j
                end
            end
        end

        #north
        for j = 1:regionSize
            if j > rw # has neighbor within region on north

            else
                if i > ow # has neighboring region on north
                    k = (i-1)*nxr + (j-1)*nx + 1 - ow*nxr - rw*nx + nxr
                    An[ (j-1)*nx + nx, k] = 1; #omega, theta_j
                end
            end
        end
        xNeighbor[i] = Vector{Vector{Float64}}(undef,4)
        xNeighbor[i][1] = Ae*x;
        xNeighbor[i][2] = As*x;
        xNeighbor[i][3] = Aw*x;
        xNeighbor[i][4] = An*x;
    end

    for i = 1:NsubSys
        xnext[i] = hn2_coupled(ode_coupled_lin,h, xcurr[i], ucurr[i], nx, w, rw, xNeighbor[i], i, Mrand[i], Drand[i], aij)
    end
    xnext_cent = vcat(xnext...)

    return xnext_cent
end

function hn2_coupled(ode::Function,h::Float64, x::Vector{Float64}, u::Vector{Float64}, nx::Int64, w::Int64, rw::Int64, xNeighbor::Any, region_number::Int64, Mrand, Drand, aij)
    k1 = ode(x,u,nx, w, rw, Mrand, Drand, aij, xNeighbor, region_number);
    k2 = ode(x+h*k1,u,nx, w, rw, Mrand, Drand, aij, xNeighbor, region_number);
    return (x + h/2 * (k1+ k2));    
end

function ode_coupled_lin(x::Vector{Float64}, u::Vector{Float64}, nx, w, rw, Mrand, Drand, aij, xNeighbour, region_number)
    lx = length(x)
    regionSize = rw^2
    ow = Int64(floor(w/rw))  # Number of regions per outer border
    NsubSys = ow^2

    # Initialize sparse matrices
    theta_diff_east = spzeros(lx, lx)
    theta_diff_south = spzeros(lx, lx)
    theta_diff_west = spzeros(lx, lx)
    theta_diff_north = spzeros(lx, lx)
    omega_select_omega = spzeros(lx, lx)
    theta_select_omega = spzeros(lx, lx)

    i = region_number
    nxr = nx * regionSize
    
    for j = 1:regionSize
        #east
        if mod(j,rw) != 0 # has neighboring oscillator on east within the same region
            theta_diff_east[ 0+ (j-1)*nx + nx, 0+ (j-1)*nx + 1 ] = 1; #omega_i , theta_i
            k = 0+ j*nx + 1; #neighboring oscillator index
            theta_diff_east[ 0+ (j-1)*nx + nx, k] = -1; #omega_i , theta_j
        else
            if mod(i,ow) != 0 # has neighboring oscillator is in neighboring region
                theta_diff_east[ 0+ (j-1)*nx + nx, 0+ (j-1)*nx + 1 ] = 1; #omega_i , theta_i
                k = 0+ (j-1)*nx + 1 + nxr- rw*nx + nx
                # theta_diff_east[ 0+ (j-1)*nx + nx, k] = -1; #omega, theta_j
            end
        end

        #south
        if j <= regionSize - rw # has neighbor on south within the same region
            theta_diff_south[ 0+ (j-1)*nx + nx, 0+ (j-1)*nx + 1 ] = 1; #omega_i , theta_i
            k = 0+ (j-1)*nx + rw*nx + 1; #neighboring oscillator index
            theta_diff_south[ 0+ (j-1)*nx + nx, k] = -1; #omega_i , theta_j
        else
            if i <= NsubSys - ow # has neighboring oscillator in neighboring region on south
                theta_diff_south[ 0+ (j-1)*nx + nx, 0+ (j-1)*nx + 1 ] = 1; #omega_i , theta_i
                k = 0+ (j-1)*nx + 1 + ow*nxr + rw*nx -nxr
                # theta_diff_south[ 0+ (j-1)*nx + nx, k] = -1; #omega, theta_j
            end
        end

        #west
        if mod(j,rw) != 1 # has neighbor within region on west
            theta_diff_west[ 0+ (j-1)*nx + nx, 0+ (j-1)*nx + 1 ] = 1; #omega_i , theta_i
            k = 0+ (j-2)*nx + 1; #neighboring oscillator index
            theta_diff_west[ 0+ (j-1)*nx + nx, k] = -1; #omega_i , theta_j
        else
            if mod(i,ow) != 1 # has neighboring region on west
                theta_diff_west[ 0+ (j-1)*nx + nx, 0+ (j-1)*nx + 1 ] = 1; #omega_i , theta_i
                k = 0+ (j-1)*nx + 1 - nxr + rw*nx - nx
                # theta_diff_west[ 0+ (j-1)*nx + nx, k] = -1; #omega, theta_j
            end
        end

        #north
        if j > rw # has neighbor within region on north
            theta_diff_north[ 0+ (j-1)*nx + nx, 0+ (j-1)*nx + 1 ] = 1; #omega_i , theta_i
            k = 0+ (j-1)*nx - rw*nx + 1; #neighboring oscillator index
            theta_diff_north[ 0+ (j-1)*nx + nx, k] = -1; #omega_i , theta_j
        else
            if i > ow # has neighboring region on north
                theta_diff_north[ 0+ (j-1)*nx + nx, 0+ (j-1)*nx + 1 ] = 1; #omega_i , theta_i
                k = 0+ (j-1)*nx + 1 - ow*nxr - rw*nx + nxr
                # theta_diff_north[ 0+ (j-1)*nx + nx, k] = -1; #omega, theta_j
            end
        end
    end

    # Omega and theta selection matrices
    for j in 1:lx
        if j % nx == 0
            omega_select_omega[j, j] = 1
            theta_select_omega[j-1, j] = 1
        end
    end

    # Precompute input expansion matrix
    B = kron(sparse(I,regionSize,regionSize), [0;1])

    # Precompute Ms and Ds
    Ms = kron(sparse(I,regionSize,regionSize),ones(nx))*Mrand; #spanned Mrand_cent vector
    Ds = kron(sparse(I,regionSize,regionSize),ones(nx))*Drand; #spanned Mrand_cent vector

    # Dynamics equation: omega_dot
    x_dot = -(Ds./ Ms) .* (omega_select_omega * x) .+ (1 ./ Ms) .* (B * u) .+ (theta_select_omega * x)
    coupling = - (aij ./ Ms) .* (theta_diff_east * x .- xNeighbour[1]) .-
    (aij ./ Ms) .* (theta_diff_south * x .- xNeighbour[2]) .-
    (aij ./ Ms) .* (theta_diff_west * x .- xNeighbour[3]) .-
    (aij ./ Ms) .* (theta_diff_north * x .- xNeighbour[4]) 
    for j = 1:regionSize
        x_dot[nx*j] += coupling[nx*j]
    end

    return x_dot
end