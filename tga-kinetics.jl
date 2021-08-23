using Plots
using Printf
using DelimitedFiles
using Optim
using Dierckx # splines
using StaticArrays
include("tga-kinetics-functions.jl")

# load data
filenames = ["test_data1.txt","test_data2.txt","test_data3.txt"]

# smoothing factors for smoothing spline
smoothing_factors = [5e-6, 1e-5, 5e-6]
# consider only experimental points in the range cut[low_T, high_T]
cut = [ [200,1200],[200,1200],[200,1200]]
# array containg the normalizing mass
normalizing_temperature=[cut_i[1] for cut_i in cut]
# the columns containing the time, temperature and mass data in the files, respectivley
cols = [1,2,3]
# Do not read the this amount of the first rows of the initial file.
# could be e.g. comments in the data files.
skiprows = 0
# number of experiments
ne=length(filenames)
# number of partial components
nc=2
# number of parameters in model: A E n c = 4 parameters.
# Remember to change this when using a different model
np=4

# load the data
# X = mass data
# T = temperature
# DX = negative of the mass loss rate
X,T,DX = load_data(filenames, smoothing_factors, normalizing_temperature,
                cut, cols, skiprows)
# arrays to hold kinetic parameter values
parmtype = zeros(Int64,ne,nc,np)
pmat = zeros(Float64,ne,nc,np)
recon = zeros(Float64,ne,nc,np)
# Pre-allocate memory for output arrays
# number of data points in each experiment
N = [length(DX[i]) for i=1:ne]
# calculated mass conversion
Y = [zeros((N[i],nc)) for i = 1:ne]
# Time step in experiments
DT = [X[i][2,1]-X[i][1,1] for i = 1:ne]
# initial mass values (initial value of the integration)
Y0 = [ones(nc) for i = 1:ne]

# initial parameter values
pmat[1,:,:]=[10 200_000 1 0.4 ; 8 190_000 1 0.6]
pmat[2,:,:]=[10 200_000 1 0.4 ; 8 190_000 1 0.6]
pmat[3,:,:]=[10 200_000 1 0.4 ; 8 190_000 1 0.6]
# 1 = common parameters
# 2 = individual parameters
# 3 = fixed parameter
parmtype[1,:,:]=[ 1 1 1 1; 1 1 3 1]
parmtype[2,:,:]=[ 1 1 1 1; 1 1 3 1]
parmtype[3,:,:]=[ 1 1 1 1; 1 1 3 1]




# construct the mapping, i.e. determine the array that is to be passed to the
# optimizer, taking into accound that some parameters are common, individual or
# constat.
p0,pall,indmap=mapper(pmat,parmtype)
# convert to SVector
pmat=toSVector(pmat)
construct_pmat!(p0,pall,indmap,pmat,ne,nc,np)

# Use a closure to call
# minimize!(minimize!(p,pall,indmap,pmat,nc,ne,np,T,DX,N,Y,DT,Y0))
# as minimize!(p) and optimize it.
res=optimize(p->minimize!(p,pall,indmap,pmat,nc,ne,np,T,DX,N,Y,DT,Y0,integrator=rk4!),
        p0,
        NelderMead(),
        Optim.Options(iterations=4000))


# Print and plot
println()
@printf "minimum: %7.5e  iters: %i \n\n" res.minimum res.iterations

for e = 1:ne
        #pmat[e]=pmat[1]
        heun!(DT[e],Y0[e],N[e],nc,T[e],DX[e],Y[e],pmat[e])
        plt=plot()
        plot!(plt,X[e][:,1],DX[e],label="experimental rate")
        dsum = zeros(length(DX[e]))
        for c = 1:nc
                dc = [-pmat[e][c+3*nc]*dydt(T1,y1,pmat[e],c,nc) for (T1,y1) in zip(T[e],Y[e][:,c])]
                dsum .= dsum .+ dc
                plot!(plt,X[e][:,1],dc,label="component $c")
        end
        plot!(plt,X[e][:,1],dsum,title="Deviation = $(round(sqrt(sum(abs2, (dsum .- DX[e]))/(maximum(DX[e])^2*N[e]))*100,digits=2)) %",
        label="sum calculated",ylabel="- mass loss rate",xlabel="time (min)")

        @printf "pmat[%s,:,:]=%s\n"  e round.(pmat[e],digits=6)
        display(plt)
end
