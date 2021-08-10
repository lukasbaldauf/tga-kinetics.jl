using Plots
using Printf
using DelimitedFiles
using Optim
using BenchmarkTools
using Dierckx
using StaticArrays
using Interpolations
include("tga-kinetics-functions.jl")


# load data

filenames = ["test_data1.txt","test_data2.txt","test_data3.txt"]

# Array for mass conversion
X = []
# array for temperatures
T = Array{Array{Float64,1},1}()
# array for mass conversion rate
DX = Array{Array{Float64,1},1}()
# smoothing factors for smoothing spline
s = [1e-4, 1e-4, 1e-4]
norms=zeros(3)
# consider only experimental points in the range [low_T, high_T]
cut = [ [400,1200],[400,1200],[400,1200]]
for (i,file) in enumerate(filenames)
        x = DelimitedFiles.readdlm(file,comments=true,comment_char='#',skipstart=50)
        mass_norm = x[:,3][x[:,1] .< 150][1]
        #mass_norm[i]=mass_norm
        #x = x[1:floor(Int,length(x[:,1])/500):end,:]
        ndx = ( cut[i][2] .> x[:,2].> cut[i][1])
        x = Matrix([x[:,1][ndx]*60 x[:,2][ndx] x[:,3][ndx]])
        #x = Matrix([x[:,1][ndx]*60 x[:,2][ndx] x[:,3][ndx]])
        push!(X,x)
        push!(T,x[:,2].+273.15)
        spl = Spline1D(x[:,1],x[:,3],s=s[i])
        dx = -derivative(spl,x[:,1])
        dx = dx
        #plot(x[:,1],dx)
        push!(DX,dx)
end
ne=length(filenames) # number of experiments
nc=1 # number of partial components
np=4 # number of parameters in model: A E n c

parmtype = zeros(Int64,ne,nc,np)
pmat = zeros(Float64,ne,nc,np)
recon = zeros(Float64,ne,nc,np)
# initial parameters
pmat[1,:,:]=[8 200_000 1 0.2]
pmat[2,:,:]=[8 200_000 1 0.2]
pmat[3,:,:]=[8 200_000 1 0.2]
#pmat[2,:,:]=[-112.18844 -979604.402715 0.843048 2.026574 0.057221 -0.288351]
#pmat[3,:,:]=[-38.395383 -249473.833356 0.817539 1.874923 0.020734 -16.821003]
# 1 = common parameters
# 2 = individual parameters
# 3 = fixed parameter
parmtype[1,:,:]=[ 2 1 1 1]
parmtype[2,:,:]=[ 2 1 1 1]
parmtype[3,:,:]=[ 2 1 1 1]
#parmtype[2,:,:]=[ 3 3 3 3 3 3]
#parmtype[3,:,:]=[ 1 1 1 1 1 1]

N = [length(DX[i]) for i=1:ne]
Y = [zeros((N[i],nc)) for i = 1:ne]
DT = [X[i][2,1]-X[i][1,1] for i = 1:ne]
Y0 = [ones(nc) for i = 1:ne]


p0,pall,indmap=mapper(pmat,parmtype)
pmat=toSVector(pmat)
construct_pmat!(p0,pall,indmap,pmat,ne,nc,np)

#p0=res.minimizer
res=optimize(p->minimize!(p,pall,indmap,pmat,nc,ne,np,T,DX,N,Y,DT,Y0),
        p0,
        NelderMead(),
            Optim.Options(iterations=4000))
println()
@printf "minimum: %7.5e  iters: %i \n\n" res.minimum res.iterations

for e = 1:ne
        #pmat[e]=pmat[1]
        heun!(DT[e],Y0[e],N[e],nc,T[e],DX[e],Y[e],pmat[e])
        plt=plot()
        plot!(plt,X[e][:,1],DX[e],label="experimental")
        dsum = zeros(length(DX[e]))
        for c = 1:nc
                dc = [-pmat[e][c+3*nc]*dydt(T1,y1,pmat[e],c,nc) for (T1,y1) in zip(T[e],Y[e][:,c])]
                dsum .= dsum .+ dc
                #plot!(plt,X[e][:,1],dc)
        end
        plot!(plt,X[e][:,1],dsum,title="reldev = $(round(sqrt(sum(abs2, (dsum .- DX[e]))/(maximum(DX[e])^2*N[e]))*100,digits=2)) %",
        label="calculated",ylabel="conversion rate")

        @printf "pmat[%s,:,:]=%s\n"  e round.(pmat[e],digits=6)
        display(plt)
end
