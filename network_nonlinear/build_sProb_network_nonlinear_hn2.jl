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

speye(N) = spdiagm(ones(N))

function build_sProb_network_nonlinear_hn2( w, rw, N, aij, x0 , Mrand, Drand , isLoad, w_loads )
    dt::Float64 = 0.1     # time step size
    nx::Int = 2           # state dimension
    nu::Int = 1           # input dimension
    
    Noscillator = w^2;
    regionSize = rw^2;

    ow = Int64(floor(w/rw)); # number of regions per outer border

    NsubSys = Int64(floor(Noscillator/regionSize));
    
    model_list = Vector{Model}(undef, NsubSys)

    # omega limit
    xmin = [-Inf, -1.6*pi] #[-0.8; -5] #[-pi/4 -1.6 pi]
    xmax = [Inf, 1.6*pi]#[ 0.8;  5]

    nxr = nx*regionSize; # number of states in the region
    nur = nu*regionSize; # number of inputs in the region
        
    # Preallocate arrays
    xJmp_vec = Vector{Vector{VariableRef}}(undef, NsubSys)
    ow = Int64(floor(w/rw)); # number of regions per outer border

    nz = Array{Int64}(undef,NsubSys);
    neighboringRegions = Array{Int64}(undef,NsubSys);

    for i = 1:NsubSys
        if i == 1 #east and south neighbors
            neighboringRegions[i] = 2;
        elseif i < ow && mod(i,ow) != 0 #west, east, and south neighbors
            neighboringRegions[i] = 3;
        elseif i == ow #west and south neighbors
            neighboringRegions[i] = 2;
        elseif i > ow && mod(i,ow) == 1 && i < NsubSys - ow +1#north, east, and south neighbors
            neighboringRegions[i] = 3;
        elseif i > ow && mod(i,ow) == 0 && i < NsubSys - ow +1#north, south, and west neighbors
            neighboringRegions[i] = 3;
        elseif i == NsubSys - ow +1 #north and east neighbors
            neighboringRegions[i] = 2;
        elseif i > NsubSys - ow +1 && i < NsubSys #north, east, and west neighbors
            neighboringRegions[i] = 3;
        elseif i == NsubSys # north and west neighbors
            neighboringRegions[i] = 2;
        else #north, east, south, and west neighbors
            neighboringRegions[i] = 4;
        end
    end

    for i = 1:NsubSys
        model = Model(add_bridges = false) # disable bridging mechanism        
        nz[i] = ((N+1)*nxr + (N+1)*nur + (N+1)*rw*neighboringRegions[i])
        xJmp_vec[i] = @variable(model, x[1:nz[i]])
        set_string_names_on_creation(model, false)
        
        model_list[i] = model
    end

    # equality constraints
    M_r = sparse(I,regionSize,regionSize);
    M_N = sparse(I,N+1,N+1);
    M_n = sparse(I,N,N);
    I_nxr = sparse(I,nxr,nxr);
    nc = 0

    for i in 1:NsubSys
        # neighboringRegions = 0
        ncopy_e = 0
        ncopy_s = 0
        ncopy_w = 0
        ncopy_n = 0
        Ae = spzeros(nx*regionSize,0);
        As = spzeros(nx*regionSize,0);
        Aw = spzeros(nx*regionSize,0);
        An = spzeros(nx*regionSize,0);

        Br = kron(M_r./Mrand[i],[0;dt]); 

        incidence_east = spzeros(nxr,nxr);
        if mod(i,ow) != 0 # has neighboring region on east
            # neighboringRegions = neighboringRegions + 1
            Ae = spzeros(nx*regionSize,rw);
            for j = 1:regionSize
                if mod(j,rw) == 0 # next to eastern neighboring regions
                    ncopy_e = ncopy_e + 1;
                    nc = nc + 1;
                    Ae[nx*(j-1)+1:nx*j , ncopy_e ] = [0; 1] # Aij
                    incidence_east[nx*(j-1)+2, nx*(j-1)+1] = 1
                end
            end

        end

        incidence_south = spzeros(nxr,nxr)
        if i <= NsubSys - ow # has neighboring region on south
            # neighboringRegions = neighboringRegions + 1
            As = spzeros(nx*regionSize,rw);
            for j = 1:regionSize
                if j > regionSize - rw # next to southern neighboring regions
                    ncopy_s = ncopy_s + 1;
                    nc = nc + 1;
                    As[nx*(j-1)+1:nx*j , ncopy_s] = [0; 1]
                    incidence_south[nx*(j-1)+2, nx*(j-1)+1] = 1
                end
            end
        end


        incidence_west = spzeros(nxr,nxr)
        if mod(i,ow) != 1 # has neighboring region on west
            # neighboringRegions = neighboringRegions + 1
            Aw = spzeros(nx*regionSize,rw);
            
            for j = 1:regionSize
                if mod(j,rw) == 1 # next to western neighboring regions
                    ncopy_w = ncopy_w + 1;
                    nc = nc + 1;
                    Aw[nx*(j-1)+1:nx*j , ncopy_w ] = [0; 1]
                    incidence_west[nx*(j-1)+2, nx*(j-1)+1] = 1
                end
            end    
        end

        incidence_north = spzeros(nxr,nxr)
        if i > ow # has neighboring region on north
            # neighboringRegions = neighboringRegions + 1
            An = spzeros(nx*regionSize,rw);
            for j = 1:regionSize
                if j <= rw # next to northern neighboring regions
                    ncopy_n = ncopy_n + 1;
                    nc = nc + 1;
                    An[nx*(j-1)+1:nx*j , ncopy_n ] = [0; 1]
                    incidence_north[nx*(j-1)+2, nx*(j-1)+1] = 1
                end
            end   
        end

        for j = 1:regionSize
            if j > rw # has neighbor on north
                jn = j - rw; # neighbor index
                incidence_north[nx*(j-1)+1:nx*j,nx*(j-1)+1] = [0; 1;];
                incidence_north[nx*(j-1)+1:nx*j,nx*(jn-1)+1] = [0; -1];
            end
            if mod(j,rw) != 1 # has neighbor on west
                jn = j - 1; # neighbor index
                incidence_west[nx*(j-1)+1:nx*j,nx*(j-1)+1] = [0; 1];
                incidence_west[nx*(j-1)+1:nx*j,nx*(jn-1)+1] = [0; -1];
            end
            if mod(j,rw) != 0 # has neighbor on east
                jn = j + 1; # neighbor index
                incidence_east[nx*(j-1)+1:nx*j,nx*(j-1)+1] = [0; 1];
                incidence_east[nx*(j-1)+1:nx*j,nx*(jn-1)+1] = [0; -1];
            end
            if j <= regionSize - rw # has neighbor on south
                jn = j + rw; # neighbor index
                incidence_south[nx*(j-1)+1:nx*j,nx*(j-1)+1] = [0; 1];
                incidence_south[nx*(j-1)+1:nx*j,nx*(jn-1)+1] = [0; -1];
            end
        end

        model = model_list[i]
        x = xJmp_vec[i]
        # initial condition
        @constraint(model, x[1:nxr] - x0[i] .== zeros(nxr));

            # loads
        for j = 1:regionSize
            if isLoad[i][j]
                for k = 1:N+1
                    @constraint(model, x[ (N+1)*nxr + (k-1)*nur + j ] - w_loads[i][j] .== 0  )
                end
            end
        end
                
        # system dynamics
        for k = 0:N-1
            u_subsys = x[1+(N+1)*nxr + k*nur:(N+1)*nxr + (k+1)*nur]
            x_subsys = x[1+(k*nxr):((k+1)*nxr)]


            size_cross_east = size([zeros(nxr,k*ncopy_e) -Ae zeros(nxr,(N-k)*ncopy_e)],2)
            size_cross_south = size([zeros(nxr,k*ncopy_s) -As zeros(nxr,(N-k)*ncopy_s)],2)
            size_cross_west = size([zeros(nxr,k*ncopy_w) -Aw zeros(nxr,(N-k)*ncopy_w)],2)

            xNeighbor = Vector{Any}(undef, NsubSys)
            xNeighbor[1] = sparse([zeros(nxr,(N+1)*nxr) zeros(nxr,(N+1)*nur) zeros(nxr,k*ncopy_e) Ae zeros(nxr,(N-k)*ncopy_e) zeros(nxr, (N+1)*rw*neighboringRegions[i] - size_cross_east)]) * x;
            xNeighbor[2] = sparse([zeros(nxr,(N+1)*nxr)  zeros(nxr,(N+1)*nur) zeros(nxr, size_cross_east) zeros(nxr,k*ncopy_s) As zeros(nxr,(N-k)*ncopy_s) zeros(nxr, (N+1)*rw*neighboringRegions[i] - size_cross_east - size_cross_south)])* x;
            xNeighbor[3] = sparse([zeros(nxr,(N+1)*nxr) zeros(nxr,(N+1)*nur) zeros(nxr, size_cross_east + size_cross_south) zeros(nxr,k*ncopy_w) Aw zeros(nxr,(N-k)*ncopy_w) zeros(nxr, (N+1)*rw*neighboringRegions[i] - size_cross_east - size_cross_south - size_cross_west)]) * x;
            xNeighbor[4] = sparse([zeros(nxr,(N+1)*nxr) zeros(nxr,(N+1)*nur) zeros(nxr, size_cross_east + size_cross_south + size_cross_west) zeros(nxr,k*ncopy_n) An zeros(nxr,(N-k)*ncopy_n)]) * x;
            x_plus_subsys_hn2 = hn2_coupled(ode_coupled, dt,x_subsys,u_subsys, nx, w, rw, xNeighbor, i, Mrand[i], Drand[i], aij)
            x_plus_subsys_var = x[1+((k+1)*nxr):((k+2)*nxr)]
            @constraint(model, x_plus_subsys_var - x_plus_subsys_hn2 .== 0)
            # GC.gc(true)
        end
        GC.gc(true)
        
        for k = 0:N
            # theta_diff constraint
            size_cross_east = size([zeros(nxr,k*ncopy_e) -Ae zeros(nxr,(N-k)*ncopy_e)],2)
            size_cross_south = size([zeros(nxr,k*ncopy_s) -As zeros(nxr,(N-k)*ncopy_s)],2)
            size_cross_west = size([zeros(nxr,k*ncopy_w) -Aw zeros(nxr,(N-k)*ncopy_w)],2)
            AE = sparse([zeros(nxr,k*nxr) incidence_east zeros(nxr,(N-k)*nxr) zeros(nxr,(N+1)*nur) zeros(nxr,k*ncopy_e) -Ae zeros(nxr,(N-k)*ncopy_e) zeros(nxr, (N+1)*rw*neighboringRegions[i] - size_cross_east)])
            for j = size(AE,1) : -1 : 1
                if AE[j,:] == sparse(zeros(size(AE,2))) #zero row
                    AE = vcat(AE[1:j-1,:],AE[j+1:end,:])
                end
            end
            
            AS = sparse([zeros(nxr,k*nxr) incidence_south zeros(nxr,(N-k)*nxr) zeros(nxr,(N+1)*nur) zeros(nxr, size_cross_east) zeros(nxr,k*ncopy_s) -As zeros(nxr,(N-k)*ncopy_s) zeros(nxr, (N+1)*rw*neighboringRegions[i] - size_cross_east - size_cross_south)])
            for j = size(AS,1) : -1 : 1
                if AS[j,:] == sparse(zeros(size(AS,2))) #zero row
                    AS = vcat(AS[1:j-1,:],AS[j+1:end,:])
                end
            end
        
            AW = sparse([zeros(nxr,k*nxr) incidence_west zeros(nxr,(N-k)*nxr) zeros(nxr,(N+1)*nur) zeros(nxr, size_cross_east + size_cross_south) zeros(nxr,k*ncopy_w) -Aw zeros(nxr,(N-k)*ncopy_w) zeros(nxr, (N+1)*rw*neighboringRegions[i] - size_cross_east - size_cross_south - size_cross_west)])
            for j = size(AW,1) : -1 : 1
                if AW[j,:] == sparse(zeros(size(AW,2))) #zero row
                    AW = vcat(AW[1:j-1,:],AW[j+1:end,:])
                end
            end
        
            AN = sparse([zeros(nxr,k*nxr) incidence_north zeros(nxr,(N-k)*nxr) zeros(nxr,(N+1)*nur) zeros(nxr, size_cross_east + size_cross_south + size_cross_west) zeros(nxr,k*ncopy_n) -An zeros(nxr,(N-k)*ncopy_n)])
            for j = size(AN,1) : -1 : 1
                if AN[j,:] == sparse(zeros(size(AN,2))) #zero row
                    AN = vcat(AN[1:j-1,:],AN[j+1:end,:])
                end
            end

            A = vcat(AE,AS,AW,AN);

            # drop duplicate rows
            for j = size(A,1) : -1 : 1
                for jj = 1:size(A,1)
                    if jj!= j && A[jj,:] == -A[j,:]
                        A = vcat(A[1:j-1,:],A[j+1:end,:])
                        break;
                    end
                end
            end        
            
            @constraint(model,  A*x .- pi/2 .<= 0)   # theta_i - theta_j < = pi/2
            @constraint(model, -A*x .- pi/2 .<= 0)   # theta_j - theta_i < = pi/2
        end
    end

    GC.gc(true)
    # cost
    Q = sparse([0 0; 0 1]) #penalize frequency deviations more, because we want to synchronize the frequencies
    R = 0.1*speye(nu)
    P = Q
    Qr = kron(M_r,Q);
    Rr = kron(M_r,R);
    Pr = kron(M_r,P);

    HH = Array{SparseMatrixCSC{Float64}}(undef,NsubSys)
    gg = Array{Vector{Float64}}(undef,NsubSys)

    for i = 1:NsubSys
        Qcpy = spzeros(nz[i] - (N+1)*(nxr+nur), nz[i] - (N+1)*(nxr+nur))
        H = blockdiag( kron(M_n,Qr), Pr, kron(M_N,Rr), Qcpy  ) + 0.0001*speye(nz[i]);
        HH[i] = H;
        g = spzeros(nz[i]);
        gg[i] = g;
    end

    # inequality constraints
    AAineq = Array{SparseMatrixCSC{Float64}}(undef,NsubSys)
    llineq = Array{Vector{Float64}}(undef,NsubSys)
    uuineq = Array{Vector{Float64}}(undef,NsubSys)

    umin = -0.3; # p.u.
    umax =  0.3; # p.u.
    for i = 1:NsubSys
        lineq = [kron(ones((N+1)*regionSize),xmin); umin*ones((N+1)*nur); -Inf*ones(nz[i] - (N+1)*(nxr+nur))];
        uineq = [kron(ones((N+1)*regionSize),xmax); umax*ones((N+1)*nur); Inf*ones(nz[i] - (N+1)*(nxr+nur))];
        Aineq = sparse(I,nz[i],nz[i]);
        llineq[i] = lineq;
        uuineq[i] = uineq;
        AAineq[i] = Aineq;
    end

    # coupling constraints
    AA = Array{SparseMatrixCSC{Int64}}(undef,NsubSys)

    nc = 0
    for i = 1:NsubSys
        if i == 1 #east and south neighbors
            nc = nc + 2 * rw *(N+1);
        elseif i < ow && mod(i,ow) != 0 #west, east, and south neighbors
            nc = nc + 3 * rw *(N+1);
        elseif i == ow #west and south neighbors
            nc = nc + 2 * rw *(N+1);
        elseif i > ow && mod(i,ow) == 1 && i < NsubSys - ow +1#north, east, and south neighbors
            nc = nc + 3 * rw *(N+1);
        elseif i > ow && mod(i,ow) == 0 && i < NsubSys - ow +1#north, south, and west neighbors
            nc = nc + 3 * rw * (N+1);
        elseif i == NsubSys - ow +1 #north and east neighbors
            nc = nc + 2 * rw *(N+1);
        elseif i > NsubSys - ow +1 && i < NsubSys #north, east, and west neighbors
            nc = nc + 3 * rw * (N+1);
        elseif i == NsubSys # north and west neighbors
            nc = nc + 2 * rw *(N+1);
        else #north, east, south, and west neighbors
            nc = nc + 4 * rw *(N+1);
        end
    end

    for i = 1:NsubSys
        AA[i] = spzeros(nc,nz[i]);
    end
    nc_idx = 0
    for i = 1:NsubSys
        if i == 1 #east and south neighbors            
            #east 
            AA[i][nc_idx + 1: nc_idx + (N+1)*rw, (N+1)*(nxr+nur)+1:(N+1)*(nxr+nur+rw)] = -speye((N+1)*rw);
            addConsEast!(i,AA,N,M_N,rw,nxr,regionSize,nc_idx)
            nc_idx = nc_idx + (N+1)*rw;
        
            #south
            AA[i][nc_idx + 1: nc_idx + (N+1)*rw, (N+1)*(nxr+nur+rw)+1:(N+1)*(nxr+nur+2*rw)] = -speye((N+1)*rw);
            addConsSouth!(i,AA,N,M_N,rw,nxr,regionSize,nc_idx,ow)
            nc_idx = nc_idx + (N+1)*rw;        
            
        elseif i < ow && mod(i,ow) != 0 #west, east, and south neighbors
            #east 
            AA[i][nc_idx + 1: nc_idx + (N+1)*rw, (N+1)*(nxr+nur)+1:(N+1)*(nxr+nur+rw)] = -speye((N+1)*rw);
            addConsEast!(i,AA,N,M_N,rw,nxr,regionSize,nc_idx)
            nc_idx = nc_idx + (N+1)*rw;

            #south
            AA[i][nc_idx + 1: nc_idx + (N+1)*rw, (N+1)*(nxr+nur+rw)+1:(N+1)*(nxr+nur+2*rw)] = -speye((N+1)*rw);
            addConsSouth!(i,AA,N,M_N,rw,nxr,regionSize,nc_idx,ow)
            nc_idx = nc_idx + (N+1)*rw;

            #west
            AA[i][nc_idx + 1: nc_idx + (N+1)*rw, (N+1)*(nxr+nur+2*rw)+1:(N+1)*(nxr+nur+3*rw)] = -speye((N+1)*rw);
            addConsWest!(i,AA,N,M_N,rw,nxr,regionSize,nc_idx)
            nc_idx = nc_idx + (N+1)*rw;

        elseif i == ow #west and south neighbors
            #south
            AA[i][nc_idx + 1: nc_idx + (N+1)*rw, (N+1)*(nxr+nur)+1:(N+1)*(nxr+nur+rw)] = -speye((N+1)*rw);
            addConsSouth!(i,AA,N,M_N,rw,nxr,regionSize,nc_idx,ow)
            nc_idx = nc_idx + (N+1)*rw;

            #west
            AA[i][nc_idx + 1: nc_idx + (N+1)*rw, (N+1)*(nxr+nur+rw)+1:(N+1)*(nxr+nur+2*rw)] = -speye((N+1)*rw);
            addConsWest!(i,AA,N,M_N,rw,nxr,regionSize,nc_idx)
            nc_idx = nc_idx + (N+1)*rw;

        elseif i > ow && mod(i,ow) == 1 && i < NsubSys - ow +1#north, east, and south neighbors
            #east
            AA[i][nc_idx + 1: nc_idx + (N+1)*rw, (N+1)*(nxr+nur)+1:(N+1)*(nxr+nur+rw)] = -speye((N+1)*rw);
            addConsEast!(i,AA,N,M_N,rw,nxr,regionSize,nc_idx)
            nc_idx = nc_idx + (N+1)*rw;

            #south
            AA[i][nc_idx + 1: nc_idx + (N+1)*rw, (N+1)*(nxr+nur+rw)+1:(N+1)*(nxr+nur+2*rw)] = -speye((N+1)*rw);
            addConsSouth!(i,AA,N,M_N,rw,nxr,regionSize,nc_idx,ow)
            nc_idx = nc_idx + (N+1)*rw;

            #north
            AA[i][nc_idx + 1: nc_idx + (N+1)*rw, (N+1)*(nxr+nur+2*rw)+1:(N+1)*(nxr+nur+3*rw)] = -speye((N+1)*rw);
            addConsNorth!(i,AA,N,M_N,rw,nxr,regionSize,nc_idx,ow)
            nc_idx = nc_idx + (N+1)*rw;

        elseif i > ow && mod(i,ow) == 0 && i < NsubSys - ow +1 #north, south, and west neighbors  
            #south
            AA[i][nc_idx + 1: nc_idx + (N+1)*rw, (N+1)*(nxr+nur)+1:(N+1)*(nxr+nur+rw)] = -speye((N+1)*rw);
            addConsSouth!(i,AA,N,M_N,rw,nxr,regionSize,nc_idx,ow)
            nc_idx = nc_idx + (N+1)*rw;

            #west
            AA[i][nc_idx + 1: nc_idx + (N+1)*rw, (N+1)*(nxr+nur+rw)+1:(N+1)*(nxr+nur+2*rw)] = -speye((N+1)*rw);
            addConsWest!(i,AA,N,M_N,rw,nxr,regionSize,nc_idx)
            nc_idx = nc_idx + (N+1)*rw;

            #north
            AA[i][nc_idx + 1: nc_idx + (N+1)*rw, (N+1)*(nxr+nur+2*rw)+1:(N+1)*(nxr+nur+3*rw)] = -speye((N+1)*rw);
            addConsNorth!(i,AA,N,M_N,rw,nxr,regionSize,nc_idx,ow)
            nc_idx = nc_idx + (N+1)*rw;

        elseif i == NsubSys - ow +1 #north and east neighbors
            #east
            AA[i][nc_idx + 1: nc_idx + (N+1)*rw, (N+1)*(nxr+nur)+1:(N+1)*(nxr+nur+rw)] = -speye((N+1)*rw);
            addConsEast!(i,AA,N,M_N,rw,nxr,regionSize,nc_idx)
            nc_idx = nc_idx + (N+1)*rw;

            #north
            AA[i][nc_idx + 1: nc_idx + (N+1)*rw, (N+1)*(nxr+nur+rw)+1:(N+1)*(nxr+nur+2*rw)] = -speye((N+1)*rw);
            addConsNorth!(i,AA,N,M_N,rw,nxr,regionSize,nc_idx,ow)
            nc_idx = nc_idx + (N+1)*rw;

        elseif i > NsubSys - ow +1 && i < NsubSys #north, east, and west neighbors
            #east
            AA[i][nc_idx + 1: nc_idx + (N+1)*rw, (N+1)*(nxr+nur)+1:(N+1)*(nxr+nur+rw)] = -speye((N+1)*rw);
            addConsEast!(i,AA,N,M_N,rw,nxr,regionSize,nc_idx)
            nc_idx = nc_idx + (N+1)*rw;

            #west
            AA[i][nc_idx + 1: nc_idx + (N+1)*rw, (N+1)*(nxr+nur+rw)+1:(N+1)*(nxr+nur+2*rw)] = -speye((N+1)*rw);
            addConsWest!(i,AA,N,M_N,rw,nxr,regionSize,nc_idx)
            nc_idx = nc_idx + (N+1)*rw;

            #north
            AA[i][nc_idx + 1: nc_idx + (N+1)*rw, (N+1)*(nxr+nur+2*rw)+1:(N+1)*(nxr+nur+3*rw)] = -speye((N+1)*rw);
            addConsNorth!(i,AA,N,M_N,rw,nxr,regionSize,nc_idx,ow)
            nc_idx = nc_idx + (N+1)*rw;

        elseif i == NsubSys # north and west neighbors
            #west
            AA[i][nc_idx + 1: nc_idx + (N+1)*rw, (N+1)*(nxr+nur)+1:(N+1)*(nxr+nur+rw)] = -speye((N+1)*rw);
            addConsWest!(i,AA,N,M_N,rw,nxr,regionSize,nc_idx)
            nc_idx = nc_idx + (N+1)*rw;

            #north
            AA[i][nc_idx + 1: nc_idx + (N+1)*rw, (N+1)*(nxr+nur+rw)+1:(N+1)*(nxr+nur+2*rw)] = -speye((N+1)*rw);
            addConsNorth!(i,AA,N,M_N,rw,nxr,regionSize,nc_idx,ow)
            nc_idx = nc_idx + (N+1)*rw;

        else #north, east, south, and west neighbors
            #east
            AA[i][nc_idx + 1: nc_idx + (N+1)*rw, (N+1)*(nxr+nur)+1:(N+1)*(nxr+nur+rw)] = -speye((N+1)*rw);
            addConsEast!(i,AA,N,M_N,rw,nxr,regionSize,nc_idx)
            nc_idx = nc_idx + (N+1)*rw;

            #south
            AA[i][nc_idx + 1: nc_idx + (N+1)*rw, (N+1)*(nxr+nur+rw)+1:(N+1)*(nxr+nur+2*rw)] = -speye((N+1)*rw);
            addConsSouth!(i,AA,N,M_N,rw,nxr,regionSize,nc_idx,ow)
            nc_idx = nc_idx + (N+1)*rw;

            #west
            AA[i][nc_idx + 1: nc_idx + (N+1)*rw, (N+1)*(nxr+nur+2*rw)+1:(N+1)*(nxr+nur+3*rw)] = -speye((N+1)*rw);
            addConsWest!(i,AA,N,M_N,rw,nxr,regionSize,nc_idx)
            nc_idx = nc_idx + (N+1)*rw;

            #north
            AA[i][nc_idx + 1: nc_idx + (N+1)*rw, (N+1)*(nxr+nur+3*rw)+1:(N+1)*(nxr+nur+4*rw)] = -speye((N+1)*rw);
            addConsNorth!(i,AA,N,M_N,rw,nxr,regionSize,nc_idx,ow)
            nc_idx = nc_idx + (N+1)*rw;
        end

    end

    sProb = sProb_struct(Vector{Model}(undef, NsubSys),                              # model
                        Vector{Vector{Float64}}(undef, NsubSys),                     # llbx 
                        Vector{Vector{Float64}}(undef, NsubSys),                     # uubx 
                        Vector{SparseMatrixCSC{Float64, Int64}}(undef, NsubSys),     # AA 
                        Vector{Vector{Float64}}(undef, NsubSys),                     # zz0
                        Vector{Vector{Float64}}(undef, NsubSys),                     # pp
                        derivatives_struct(Vector{MathOptInterface.Nonlinear.Evaluator{MathOptInterface.Nonlinear.ReverseAD.NLPEvaluator}}(undef, NsubSys),     # Evaluator
                                        Vector{MathOptInterface.Nonlinear.Evaluator{MathOptInterface.Nonlinear.ReverseAD.NLPEvaluator}}(undef, NsubSys),        # evaluator_eq
                                        Vector{MathOptInterface.Nonlinear.Evaluator{MathOptInterface.Nonlinear.ReverseAD.NLPEvaluator}}(undef, NsubSys),        # evaluator_ineq
                                        Vector{Int}(undef, NsubSys),                         # num_variables
                                        Vector{Vector{Any}}(undef, NsubSys),                 # eq_constraints
                                        Vector{Vector{Any}}(undef, NsubSys),                 # ineq_constraints
                                        Vector{Vector{Vector{Float64}}}(undef, NsubSys),     # jacobian_eq_structure
                                        Vector{Vector{Vector{Float64}}}(undef, NsubSys),     # jacobian_ineq_structure
                                        Vector{Vector{Vector{Float64}}}(undef, NsubSys),     # hessian_structure
                                        Vector{Vector{Vector{Float64}}}(undef, NsubSys),     # hessian_lagrangian_structure
                                        Vector{Int}(undef, NsubSys),                         # num_parameters
                                        Vector{Vector{Float64}}(undef,NsubSys),              # rhs_eq
                                        Vector{Vector{Float64}}(undef,NsubSys),              # rhs_ineq
                        ), # derivatives
                        Vector{Vector{Float64}}(undef, NsubSys),                             # nnu0 
                        Vector{Vector{Float64}}(undef, NsubSys),                             # mmu0
                        Vector{Vector{Float64}}(undef, NsubSys),                             # ggam0
                        Vector{Vector{VariableRef}}(undef, NsubSys),                         # xJmp
                        0,                                                                   # N
                        Vector{SparseMatrixCSC{Int64,Int64}}(undef, NsubSys),
                        Vector{SparseMatrixCSC{Int64,Int64}}(undef, NsubSys)
    )
    

    for i = 1:NsubSys
        model = model_list[i]
        sProb.model[i] = model
        x = xJmp_vec[i]
        
        @objective(model, Min, 0.5*transpose(x)*HH[i]*x + transpose(gg[i])*x);

        sProb.llbx[i] = llineq[i]
        sProb.uubx[i] = uuineq[i]
        sProb.AA[i] = AA[i]
        sProb.xJmp[i] = xJmp_vec[i]
        sProb.zz0[i] = zeros(Float64, size(AA[i],2))
        sProb.ggam0[i] = zeros(nz[i]);
        sProb.SS[i] = sparse(zeros(nur,nz[i]));
        sProb.SS[i][:,(N+1)*nxr+1:(N+1)*nxr + nur] = speye(nur);    
    end

    return sProb
end

function addConsEast!(i,AA,N,M_N,rw,nxr,regionSize,nc_idx) #add constaints to the eas-neighbor of subsystem i

    base = zeros(rw,regionSize);
    ncopy_e = 0
    for j = 1:regionSize
        if mod(j,rw) == 1 #is neighbor
            ncopy_e = ncopy_e + 1;
            base[ncopy_e,j] = 1;
        end
    end
    AA[i+1][nc_idx + 1: nc_idx + (N+1)*rw, 1:(N+1)*nxr] = kron(M_N,kron(base,[1 0]));
end

function addConsSouth!(i,AA,N,M_N,rw,nxr,regionSize,nc_idx,ow) #add constaints to south neighbor of subsystem i)
    base = zeros(rw,regionSize);
    ncopy_s = 0
    for j = 1:regionSize
        if j <= rw #is neighbor
            ncopy_s = ncopy_s + 1;
            base[ncopy_s,j] = 1;
        end
    end
    AA[i+ow][nc_idx + 1: nc_idx + (N+1)*rw, 1:(N+1)*nxr] = kron(M_N,kron(base,[1 0]));
end

function addConsWest!(i,AA,N,M_N,rw,nxr,regionSize,nc_idx) #add constraints to subsystem west of subsystem i
    base = zeros(rw,regionSize);
    ncopy_w = 0
    for j = 1:regionSize
        if mod(j,rw) == 0 #is neighbor
            ncopy_w = ncopy_w + 1;
            base[ncopy_w,j] = 1;
        end
    end
    AA[i-1][nc_idx + 1: nc_idx + (N+1)*rw, 1:(N+1)*nxr] = kron(M_N,kron(base,[1 0]));
end

function addConsNorth!(i,AA,N,M_N,rw,nxr,regionSize,nc_idx,ow) #add constraints to subsystem north of subsystem i
    base = zeros(rw,regionSize);
    ncopy_n = 0
    for j = 1:regionSize
        if j > regionSize - rw #is neighbor
            ncopy_n = ncopy_n + 1;
            base[ncopy_n,j] = 1;
        end
    end
    AA[i-ow][nc_idx + 1: nc_idx + (N+1)*rw, 1:(N+1)*nxr] = kron(M_N,kron(base,[1 0]));
end