## Import modules
using CUDA
using Random
using Distributions
using JLD
using BenchmarkTools
using NaNStatistics
using FFTW
using DelimitedFiles

## Define functions

"""
Kernel function for calculating the chemical potential of the polymer blend, i.e. the expression inside the square brackets of equation (31) in [Glotzer, 'Computer Simulations of Spinodal Decomposition in Polymer Blends', Annual Reviews of Computational
Physics II, 1995]
    INPUT: phi - microstructure array (CuArray), l - number of lattice sites in each dimension (int), chi_c - critical value of the interaction parameter (float), chi_s - value of the interaction parameter on the spinodal (float), chi - interaction parameter (float), dx - spatial discretisation (float)
    OUTPUT: dphi_int - chemical potential array (CuArray)
"""
function gpu_dphi_int_calc(phi,dphi_int,l,chi_c,chi_s,chi,dx)
    index_x  = (blockIdx().x-1)*blockDim().x + threadIdx().x # calculate thread IDs to facilitate assignment of work
    stride_x = gridDim().x*blockDim().x
    index_y = (blockIdx().y-1)*blockDim().y + threadIdx().y
    stride_y = gridDim().y*blockDim().y
    index_z = (blockIdx().z-1)*blockDim().z + threadIdx().z
    stride_z = gridDim().z*blockDim().z
    for k = index_z:stride_z:size(phi)[3] # assign work to threads
        for i = index_x:stride_x:size(phi)[2]
            for j = index_y:stride_y:size(phi)[1]
                # enforce periodic boundary conditions
                if (i > 1) 
                    left = i-1;
                else
                    left = l;
                end
                if (i < l)
                    right = i+1;
                else
                    right = 1;
                end
                if (j > 1)
                    up = j-1;
                else
                    up = l;
                end
                if (j < l)
                    down = j+1;
                else
                    down = 1;
                end
                if (k < l)
                    out = k+1;
                else 
                    out = 1;
                end
                if (k > 1)
                    in_ = k-1;
                else
                    in_ = l;
                end
                # intermediate steps required to calculate spatial derivatives (see equations (29) and (30) of Glozter)
                pi_ = phi[j,right,k]^2 + phi[j,left,k]^2 + phi[up,i,k]^2 + phi[down,i,k]^2 + phi[j,i,out]^2 + phi[j,i,in_]^2- 2*(phi[j,right,k]*phi[j,left,k] + phi[up,i,k]*phi[down,i,k] + phi[j,i,out]*phi[j,i,in_]);
                sigma = phi[j,right,k] + phi[j,left,k] + phi[up,i,k] + phi[down,i,k] + phi[j,i,out] + phi[j,i,in_] - 6*phi[j,i,k];
                # calculate chemical potential 
                dphi_int[j,i,k] = (chi_c/(2*(chi-chi_s)))*log(phi[j,i,k]/(1-phi[j,i,k])) - (2*chi/(chi-chi_s))*phi[j,i,k] + ((1-2*phi[j,i,k])/(36*phi[j,i,k]^2*(1-phi[j,i,k])^2*(2*dx)^2))*pi_ - (1/dx^2)*(1/(18*phi[j,i,k]*(1-phi[j,i,k])))*sigma
            end
        end
    end
    return 
end

"""
Kernel function for calculating the Gaussian random variables required to calculate the noise term following equations (12) and (13) of [Petschek and Metiu, 'A computer simulation of the time‐dependent Ginzburg–Landau model for spinodal decomposition',
The Journal of Chemical Physics, 1998]
    INPUT: d1,d2,d3 - independent normal distributions (normal{float})
    OUTPUT: n1,n2,n3 - arrays containing GRVs (CuArray)
"""
function gpu_gaussian_rvs(n1,n2,n3,d1,d2,d3)
    index_x  = (blockIdx().x-1)*blockDim().x + threadIdx().x # calculate thread IDs to facilitate assignment of work 
    stride_x = gridDim().x*blockDim().x
    index_y = (blockIdx().y-1)*blockDim().y + threadIdx().y
    stride_y = gridDim().y*blockDim().y
    index_z = (blockIdx().z-1)*blockDim().z + threadIdx().z
    stride_z = gridDim().z*blockDim().z
    for k = index_z:stride_z:size(n1)[3] # assign work to threads
        for i = index_x:stride_x:size(n1)[2]
            for j = index_y:stride_y:size(n1)[1]
                @inbounds n1[j,i,k] = Float32(rand(d1)); # draw random numbers from a normal distribuiton 
                @inbounds n2[j,i,k] = Float32(rand(d2));
                @inbounds n3[j,i,k] = Float32(rand(d3));
            end
        end
    end
    return 
end

"""
Kernel function to calculate noise term following equations (12) and (13) of [Petschek and Metiu, 'A computer simulation of the time‐dependent Ginzburg–Landau model for spinodal decomposition', The Journal of Chemical Physics, 1998]
    INPUT: n1,n2,n3 - arrays containing GRVs (CuArray), l - number of lattice sites in each dimension (int), dx - spatial discretisation (float)
    OUTPUT: mu - noise variable array (CuArray)
"""
function gpu_noise(mu,n1,n2,n3,l,dx)
    index_x  = (blockIdx().x-1)*blockDim().x + threadIdx().x # calculate thread IDs to facilitate assignment of work
    stride_x = gridDim().x*blockDim().x
    index_y = (blockIdx().y-1)*blockDim().y + threadIdx().y
    stride_y = gridDim().y*blockDim().y
    index_z = (blockIdx().z-1)*blockDim().z + threadIdx().z
    stride_z = gridDim().z*blockDim().z
    for k = index_z:stride_z:size(n1)[3] # assign work to threads
        for i = index_x:stride_x:size(n1)[2]
            for j = index_y:stride_y:size(n1)[1]
                # enforce periodic boundary coniditions
                if (i > 1)
                    left = i-1;
                else
                    left = l;
                end
                if (i < l)
                    right = i+1;
                else
                    right = 1;
                end
                if (j > 1)
                    up = j-1;
                else
                    up = l;
                end
                if (j < l)
                    down = j+1;
                else
                    down = 1;
                end
                if (k < l)
                    out = k+1;
                else 
                    out = 1;
                end
                if (k > 1)
                    in_ = k-1;
                else
                    in_ = l;
                end
                # calculate noise term
                @inbounds mu[j,i,k] = (1/dx)*(n1[j,right,k]-n1[j,i,k]+n2[up,i,k]-n2[j,i,k]+n3[j,i,out]-n3[j,i,k])
            end
        end
    end
    return 
end

"""
Kernel function for calculating the evolution of the microstructure over one time-step using the Euler method
    INPUT: dphi_int - chemical potential array (CuArray), mu - noise variable array (CuArray), chi - interaction parameter (float), chi_s - value of the interaction parameter on the spinodal (float), v0 - monomeric volume (float), lambda - Kuhn length (float), l - number of lattice sites in each dimension (int),
           dx - spatial discretisation (float), dt - temporal discretisation (float)
    OUTPUT: phi - microstructure array (CuArray), dphi - change of microstructure array (CuArray)
"""
function gpu_update_lat(phi,dphi_int,dphi,mu,chi,chi_s,v0,lambda,l,dx,dt)
    index_x = (blockIdx().x-1)*blockDim().x + threadIdx().x # calculate thread IDs to facilitate ssignment of work
    stride_x = gridDim().x*blockDim().x
    index_y = (blockIdx().y-1)*blockDim().y + threadIdx().y
    stride_y = gridDim().y*blockDim().y
    index_z = (blockIdx().z-1)*blockDim().z + threadIdx().z
    stride_z = gridDim().z*blockDim().z
    for k = index_z:stride_z:size(phi)[3] # assign work to threads
        for i = index_x:stride_x:size(phi)[2]
            for j = index_y:stride_y:size(phi)[1]
                # enforce periodic boundary conditions
                if (i > 1)
                    left = i-1;
                else
                    left = l;
                end
                if (i < l)
                    right = i+1;
                else
                    right = 1;
                end
                if (j > 1)
                    up = j-1;
                else
                    up = l;
                end
                if (j < l)
                    down = j+1;
                else
                    down = 1;
                end
                if (k < l)
                    out = k+1;
                else 
                    out = 1;
                end
                if (k > 1)
                    in_ = k-1;
                else
                    in_ = l;
                end
                # calculate change of microstructure
                @inbounds dphi[j,i,k] = (dt/(2*dx^2))*(dphi_int[j,right,k] + dphi_int[j,left,k] + dphi_int[up,i,k] + dphi_int[down,i,k] + dphi_int[j,i,out] + dphi_int[j,i,in_] - 6*dphi_int[j,i,k]);
                # calculate microstructure
                @inbounds phi[j,i,k] = phi[j,i,k] + dphi[j,i,k] + ((v0^(1/2)*(chi-chi_s)^(1/4))/(lambda^(3/2)))*sqrt(dt/dx^3)*mu[j,i,k];
            end
        end
    end
    return
end

"""
Function for converting cartesian coordinates into polar coordinates
    INPUT: x,y - cartesian coordinates (float)
    OUTPUT: rho,phi - polar coordinates (float)
"""
function cart2pol(x,y)
    rho = sqrt.(x.^2 + y.^2)
    phi = atan.(y,x)
    return [rho,phi]
end

"""
Function for calculating the radial average of a 2D powerspectrum
    INPUT: ps2d - 2D powerspectrum (array), l - number of lattice sites in each dimension (int)
    OUTPUT: ps1D - 1D radially-averaged powerspectrum (array)
"""
function rad_av(ps2D,l)
    x = -(l-1)/2:1:(l-1)/2
    X = x' .* ones(size(x)) # define cartesian meshgrid
    Y = ones(size(x))' .* x
    rho, azi = cart2pol(X,Y) # convert cartesian coordinates to polar coordinates
    rho = round.(rho) # round radial distances to nearest integer
    rho_ = reshape(rho,l^2) # reshape array of radial distances into column vector
    ps2D_2 = reshape(ps2D,l^2) # reshape 2D powerspectrum array into column vector
    ps1D = zeros(Int64(round((l-1)/2))+1) # create array to save radially-averaged data in
    i = 1
    for r = 0:1:Int64(round((l-1)/2)) # loop over radial distances
        ps1D[i] = nanmean(ps2D_2[getindex.(findall(x->x==r,rho_),1)]) # average the data at a given radial distance and add to ps1D array
        i = i + 1
    end
    return ps1D
end

"""
Function to calculate the scattering corresponding to a microstructure, i.e. the radial average of the 2D powerspectrum of the structure
    INPUT: phi - microstructure array (CuArray), phi0 - average composition (float), l - number of lattice sites in each dimension (int)
    OUTPUT: scat_data - scattering data (array), prcl_data - pair correlation data (array)
"""
function scat_calc(phi,phi0,l)
    pc2D_1 = zeros((l,l)) # create array to store data in
    for jz = 1:1:l # loop over slices of the microstructure in the z-direction
        s = phi[jz,:,:].-phi0 # calculate fluctuation from average composition
        FT1 = fft(s) # calculate the FT (DFT) of the fluctuations
        pc2D_1 = pc2D_1 .+ FT1 # update data in storage array
    end
    scat_data = rad_av(real(fftshift(pc2D_1.*conj(pc2D_1))),l) # calculate radial average of scattering data 
    prcl_data = rad_av(real(fftshift(fft(pc2D_1.*conj(pc2D_1)))),l) # calculate radial average of pair correlation 
    return [scat_data, prcl_data]
end

"""
Function to run finite difference simulation of spinodal decomposition and calculate the scattering at regular intervals
    INPUT: phi0 - average composition (int), chi_c - critical value of the interaction parameter (float), chi_s - value of the interaction parameter on the spinodal (float), chi - interaction parameter (float), ICn - magnitude of initial flucutation from phi0 (float), 
           l - number of lattice sites in each dimension (int), v0 - monomeric volume (float), lambda - Kuhn length (float), dt - spatial discretisation (float), dts - time between measurements of scattering data (float), dx - spatial discretisation (float), nsteps - number of time steps (int),
           nsnaps - number of measurements of scattering data to make (int), rep - repeat number (int), i.e. number to keep track of repeats, threads1D - number of threads to request in each dimension (int), nblocks - number of blocks to request in each dimension (int)
    OUTPUT: (time series of scattering data and pair correlation data saved to file (array))
"""
function run(phi0,N,chi_c,chi_s,chi,ICn,l,v0,lambda,dt,dts,dx,nsteps,nsnaps,rep,nthreads1D,nblocks1D)
   
    nq = Int64((l-1)/2)+1 # number of wavenumbers/ distance values for which scattering/ pair correlation data defined
    SD = zeros((nsnaps,nq)) # storage array for scattering data
    PC = zeros((nsnaps,nq)) # storage array for pair correlation data

    phi = CUDA.fill(0.0f0,l,l,l); # overwriteable storage array for microstructure data (CuArray)
    dphi_int = CUDA.fill(0.0f0,l,l,l); # overwriteable storage array for chemical potential data (CuArray)
    dphi = CUDA.fill(0.0f0,l,l,l); # overwriteable storage array for change of microstructure data (CuArray)
    n1 = CUDA.fill(0.0f0,l,l,l); # overwriteable storage array for GRVs (CuArray)
    n2 = CUDA.fill(0.0f0,l,l,l);
    n3 = CUDA.fill(0.0f0,l,l,l);
    mu = CUDA.fill(0.0f0,l,l,l); # overwriteable storage array for noise term (CuArray)

    # specify number of thread-blocks to request
    nthreads = (nthreads1D,nthreads1D,nthreads1D);
    nblocks = (nblocks1D,nblocks1D,nblocks1D);

    #Random.seed!(1234)
    phi_cpu = (rand!(zeros(Float32,l,l,l)).-0.5)*ICn + ones(Float32,l,l,l)*phi0; # initial condition - define on CPU
    phi = copyto!(phi,phi_cpu) # move to GPU
  
    si = 1
    for t = 1:1:nsteps # loop over time steps
        d1 = Normal() # define new Normal distributions
        d2 = Normal()
        d3 = Normal()
        @cuda threads = nthreads blocks = nblocks  gpu_dphi_int_calc(phi,dphi_int,l,chi_c,chi_s,chi,dx) # calculate chemical potential 
        @cuda threads = nthreads blocks = nblocks  gpu_gaussian_rvs(n1,n2,n3,d1,d2,d3) # calculate GRVs to calculate noise term with
        @cuda threads = nthreads blocks = nblocks  gpu_noise(mu,n1,n2,n3,l,dx) # calculate noise term
        @cuda threads = nthreads blocks = nblocks  gpu_update_lat(phi,dphi_int,dphi,mu,chi,chi_s,v0,lambda,l,dx,dt) # update microstructure
        if t == 1 # calculate and store scattering and pair correlation data at first time step
	        SD[si,:], PC[si,:] = scat_calc(Array(phi),phi0,l)
        end
        if t%(Int(dts/dt)) == 0 # calculate and store scattering and pair correlation data at regular intervals, i.e. every (dts/dt) time steps
            si = si + 1
            SD[si,:], PC[si,:] = scat_calc(Array(phi),phi0,l)
        end
    end
    # save scattering and pair correlation data
    writedlm("../Scattering_Data/Akcasu/ScD_Phi0=$phi0 N=$N chi=$chi ICn=$ICn dx=$dx dt=$dt nsteps=$nsteps dts=$dts L=$l v0=$v0 lam=$lambda NET 3D SnS SD 2DFT rep=$rep.txt", SD) 
    writedlm("../Scattering_Data/Akcasu/PCF_Phi0=$phi0 N=$N chi=$chi ICn=$ICn dx=$dx dt=$dt nsteps=$nsteps dts=$dts L=$l v0=$v0 lam=$lambda NET 3D SnS SD 2DFT rep=$rep.txt", PC) 
end

# Define blend parameters (to be imported from 'Parameters.sh')
phi0 = parse(Float32, ARGS[1]);
N = parse(Int64, ARGS[2]); 
chi_c = Float32(2/N);
chi_s = Float32(chi_c/(4*phi0*(1-phi0)));
chi = parse(Float32, ARGS[3]);
ICn = parse(Float32, ARGS[4]);
l = parse(Int64, ARGS[5]);
v0 = parse(Int64, ARGS[6]);
lambda = parse(Float32, ARGS[7]);

# Define simulation parameters (to be imported from 'Parameters.sh')
dt = parse(Float32, ARGS[8]);
dts = parse(Float32, ARGS[9]); 
dx = parse(Float32, ARGS[10]);
nsteps = parse(Int64, ARGS[11]);
nsnaps = parse(Int64, ARGS[12]);
rep = parse(Int64, ARGS[13]); 
nthreads1D = parse(Int64, ARGS[14]);
nblocks1D = parse(Int64, ARGS[15]); 

# Run code
run(phi0,N,chi_c,chi_s,chi,ICn,l,v0,lambda,dt,dts,dx,nsteps,nsnaps,rep,nthreads1D,nblocks1D)
