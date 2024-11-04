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

function build_qProb_network_linear_hn2( w , rw , N , aij , x0 , Mrand, Drand, isLoad, w_loads )
    dt = 0.1              # sampling interval
    nx::Int = 2           # state dimension
    nu::Int = 1           # input dimension

    Noscillator = w^2;
    regionSize = rw^2;  

    ow = Int64(floor(w/rw)); # number of regions per outer border

    NsubSys = Int64(floor(Noscillator/regionSize));

    print("Number of regions = $NsubSys\n")

    HH = Vector{SparseMatrixCSC{Float64, Int64}}(undef,NsubSys)
    gg = Vector{Vector{Float64}}(undef,NsubSys)
    AAeq = Vector{SparseMatrixCSC{Float64, Int64}}(undef,NsubSys)
    bbeq = Vector{Vector{Float64}}(undef,NsubSys)
    AAineq = Vector{SparseMatrixCSC{Float64, Int64}}(undef,NsubSys)
    ubineq = Vector{Vector{Float64}}(undef,NsubSys)
    lbineq = Vector{Vector{Float64}}(undef,NsubSys)
    llbx = Vector{Vector{Float64}}(undef,NsubSys)
    uubx = Vector{Vector{Float64}}(undef,NsubSys)
    AA = Vector{SparseMatrixCSC{Int64}}(undef,NsubSys)
    SS = Vector{SparseMatrixCSC{Int64}}(undef,NsubSys) #selector matrix for control input

    nxr = nx*regionSize; # number of states in the region
    nur = nu*regionSize; # number of inputs in the region

    M_r = sparse(I,regionSize,regionSize);
    M_N = sparse(I,N+1,N+1);
    M_n = sparse(I,N,N);

    I_nxr = sparse(I,nxr,nxr);

    # equality constraints

    for i = 1:NsubSys

        Arc = spzeros(nxr,nxr); # system matrix within one region for one time step

        for j = 1:regionSize
            Ac = [0 1; 0 -(Drand[i][j]/Mrand[i][j])];
            Aijc = [0; aij/Mrand[i][j]]

            Arc[nx*(j-1)+1:nx*j,nx*(j-1)+1:nx*j] = Ac;
            if j > rw # has neighbor on north
                jn = j - rw; # neighbor index
                Arc[nx*(j-1)+1:nx*j,nx*(jn-1)+1] = Aijc;
                Arc[nx*(j-1)+1:nx*j,nx*(j-1)+1] += -Aijc;
            end
            if mod(j,rw) != 1 # has neighbor on west
                jn = j - 1; # neighbor index
                Arc[nx*(j-1)+1:nx*j,nx*(jn-1)+1] = Aijc;
                Arc[nx*(j-1)+1:nx*j,nx*(j-1)+1] += -Aijc;
            end
            if mod(j,rw) != 0 # has neighbor on east
                jn = j + 1; # neighbor index
                Arc[nx*(j-1)+1:nx*j,nx*(jn-1)+1] = Aijc;
                Arc[nx*(j-1)+1:nx*j,nx*(j-1)+1] += -Aijc;
            end
            if j <= regionSize - rw # has neighbor on south
                jn = j + rw; # neighbor index
                Arc[nx*(j-1)+1:nx*j,nx*(jn-1)+1] = Aijc;
                Arc[nx*(j-1)+1:nx*j,nx*(j-1)+1] += -Aijc;
            end
        end

        neighboringRegions = 0
        ncopy_e = 0
        ncopy_s = 0
        ncopy_w = 0
        ncopy_n = 0
        Ae = spzeros(nx*regionSize,0);
        As = spzeros(nx*regionSize,0);
        Aw = spzeros(nx*regionSize,0);
        An = spzeros(nx*regionSize,0);

        if mod(i,ow) != 0 # has neighboring region on east
            neighboringRegions = neighboringRegions + 1
            Ae = spzeros(nx*regionSize,rw);
            for j = 1:regionSize
                if mod(j,rw) == 0 # next to eastern neighboring regions
                    ncopy_e = ncopy_e + 1;
                    Aij = [0; aij/Mrand[i][j]*dt]
                    Aijc = [0; aij/Mrand[i][j]]
                    Ae[nx*(j-1)+1:nx*j , ncopy_e ] = Aij        # Euler discretization for aij*theta_j
                    Arc[nx*(j-1)+1:nx*j,nx*(j-1)+1] += -Aijc;   # continuous-time term for -aij*theta_i
                end
            end

        end

        if i <= NsubSys - ow # has neighboring region on south
            neighboringRegions = neighboringRegions + 1
            As = spzeros(nx*regionSize,rw);
            for j = 1:regionSize
                if j > regionSize - rw # next to southern neighboring regions
                    ncopy_s = ncopy_s + 1;
                    Aij = [0; aij/Mrand[i][j]*dt]
                    Aijc = [0; aij/Mrand[i][j]]
                    As[nx*(j-1)+1:nx*j , ncopy_s] = Aij         # Euler discretization for aij*theta_j
                    Arc[nx*(j-1)+1:nx*j,nx*(j-1)+1] += -Aijc;   # continuous-time term for -aij*theta_i
                end
            end
        end

        if mod(i,ow) != 1 # has neighboring region on west
            neighboringRegions = neighboringRegions + 1
            Aw = spzeros(nx*regionSize,rw);
            for j = 1:regionSize
                if mod(j,rw) == 1 # next to western neighboring regions
                    ncopy_w = ncopy_w + 1;
                    Aij = [0; aij/Mrand[i][j]*dt]
                    Aijc = [0; aij/Mrand[i][j]]
                    Aw[nx*(j-1)+1:nx*j , ncopy_w ] = Aij        # Euler discretization for aij*theta_j
                    Arc[nx*(j-1)+1:nx*j,nx*(j-1)+1] += -Aijc;   # continuous-time term for -aij*theta_i
                end
            end    
        end

        if i > ow # has neighboring region on north
            neighboringRegions = neighboringRegions + 1
            An = spzeros(nx*regionSize,rw);
            for j = 1:regionSize
                if j <= rw # next to northern neighboring regions
                    ncopy_n = ncopy_n + 1;
                    Aij = [0; aij/Mrand[i][j]*dt]
                    Aijc = [0; aij/Mrand[i][j]]
                    An[nx*(j-1)+1:nx*j , ncopy_n ] = Aij        # Euler discretization for aij*theta_j
                    Arc[nx*(j-1)+1:nx*j,nx*(j-1)+1] += -Aijc;   # continuous-time term for -aij*theta_i
                end
            end   

        end

        Bc = spzeros(nxr,nur)
        for j = 1:regionSize
            Bc[ j*nx , j ] = 1/Mrand[i][j] # Bc \neq 0 also for loads, because loads are set via u
        end
        # hn2 discretization for system dynamics within region (B \neq 0 for loads)
        (Ard,Brd) = hn2_LTI(Matrix(Arc),Matrix(Bc),dt);


        Aeq = spzeros((N+1)*nxr, (N+1)*nxr + (N+1)*nur + (N+1)*rw*neighboringRegions);
        Aeq[1:nxr,1:nxr] = I_nxr; #initial condition

        # dynamics
        for k = 0:N-1
            Aeq[(k+1)*nxr+1:(k+2)*nxr,:] = sparse([zeros(nxr,k*nxr) Ard -I_nxr zeros(nxr,(N-1-k)*nxr) zeros(nxr,k*nur) Brd zeros(nxr,(N-k)*nur) zeros(nxr,k*ncopy_e) Ae zeros(nxr,(N-k)*ncopy_e) zeros(nxr, k*ncopy_s) As zeros(nxr,(N-k)*ncopy_s)  zeros(nxr,k*ncopy_w) Aw zeros(nxr,(N-k)*ncopy_w) zeros(nxr, k*ncopy_n) An zeros(nxr,(N-k)*ncopy_n) ]);
        end
        AAeq[i] = Aeq;
    end

    # Initial and reference states
    for i = 1:NsubSys
        neq = size(AAeq[i],1);
        bbeq[i] = spzeros(neq);
        bbeq[i][1:nxr] = x0[i];
    end

    # loads
    for i = 1:NsubSys
        base_A = spzeros(0,nur);
        base_b = zeros(0);
        for j = 1:regionSize
            if isLoad[i][j]
                row_u = [spzeros(1, j-1) 1 spzeros(1,regionSize - j)]
                base_A = vcat(base_A, row_u)
                base_b = vcat(base_b, w_loads[i][j])
            end
        end
        Aeq_load = kron(M_N,base_A)
        nr = size(Aeq_load,1)
        AAeq[i] = vcat(AAeq[i], [spzeros(nr, (N+1)*nxr) Aeq_load spzeros(nr, size(AAeq[i],2) - (N+1)*nxr - (N+1)*nur) ])
        for k = 1:N+1
            bbeq[i] = vcat(bbeq[i], base_b)
        end
    end

    # objective
    Q = sparse([0 0; 0 1]) #penalize frequency deviations more, because we want to synchronize the frequencies
    R = 0.1*speye(1)
    P = Q

    Qr = kron(M_r,Q);
    Rr = kron(M_r,R);
    Pr = kron(M_r,P);
    nz = Array{Int64}(undef,NsubSys);

    for i = 1:NsubSys
        nz[i] = size(AAeq[i],2)
        Qcpy = spzeros(nz[i] - (N+1)*(nxr+nur), nz[i] - (N+1)*(nxr+nur))
        H = blockdiag( kron(M_n,Qr), Pr, kron(M_N,Rr), Qcpy  ) + 0.0001*speye(nz[i]);
        HH[i] = H;
        g = spzeros(nz[i]);
        gg[i] = g;
        SS[i] = spzeros(nur,nz[i]);
        SS[i][:,(N+1)*nxr+1:(N+1)*nxr + nur] = speye(nur);
    end


    # box constraints
    umin = -0.3 # p.u.
    umax =  0.3 # p.u.
    xmin = [-Inf; -1.6*pi]
    xmax = [ Inf;  1.6*pi]
    for i = 1:NsubSys
        lineq = [kron(ones((N+1)*regionSize),xmin); umin*ones((N+1)*nur); xmin[1]*ones(nz[i] - (N+1)*(nxr+nur))];
        uineq = [kron(ones((N+1)*regionSize),xmax); umax*ones((N+1)*nur); xmax[1]*ones(nz[i] - (N+1)*(nxr+nur))];
        llbx[i] = lineq;
        uubx[i] = uineq;
    end

    # theta_i - theta_j constraints
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

        incidence_east = spzeros(nxr,nxr);
        if mod(i,ow) != 0 # has neighboring region on east
            # neighboringRegions = neighboringRegions + 1
            Ae = spzeros(nx*regionSize,rw);
            for j = 1:regionSize
                if mod(j,rw) == 0 # next to eastern neighboring regions
                    ncopy_e = ncopy_e + 1;
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

        Aineq = spzeros(0,nz[i]);
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
            
            Aineq = vcat(Aineq,A);
        end
        AAineq[i] = Aineq;
        ubineq[i] = pi/2*ones(size(AAineq[i],1));
        lbineq[i] = -pi/2*ones(size(AAineq[i],1));
    end


    # coupling constraints

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

    Ncons = size(AA[1],1);
    llam0 = Vector{Vector{Float64}}(undef, NsubSys);
    ggam0 = Vector{Vector{Float64}}(undef, NsubSys);
    xx0 = Vector{Vector{Float64}}(undef, NsubSys);
    for i = 1:NsubSys
        llam0[i] = zeros(Ncons);
        ggam0[i] = zeros(nz[i]);
        xx0[i] = zeros(nz[i]);
    end

    my_qProb = qProb_struct(HH,gg,AAeq,bbeq,AAineq,ubineq,lbineq,llbx,uubx,AA,ggam0,xx0,N,SS)

    return my_qProb
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