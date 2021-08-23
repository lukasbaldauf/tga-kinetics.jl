@inline function dydt(T::Float64,y::Float64,pj::Array{Float64,2},j::Int64,m::Int64) ::Float64
    """ This function returns the equation governing the kinetics of the systems
    considered.
    """
    if y < 0.0
        return 0.0
    else
        #return @inbounds -10^(pj[j]+pj[j+4*m]*T+pj[j+5*m]*T^2)*exp(-(pj[j+m])/(8.3145*(T)))*y^pj[j+2*m]
        return @inbounds -10^(pj[j])*exp(-(pj[j+m])/(8.3145*(T)))*y^pj[j+2*m]
    end
end


function heun!(dt,y0,n,m,temp,dx,y,pe) :: Float64
    """ Integrate along temperature (instead of the usual
    time) using Heun's method (a predictor-corrector
    method). At the same time, subtract the calculated conversion rate and
    from the experimental conversion rate to directly calculate the deviation
    from the two, which is minimized by the optimization algorithm to find the
    best kinetic parameters.
    """
    @views y[1,:]=y0
    OF ::Float64 = 0.0
    max_rate = 0.0
    for i = 1:n-1
        # sum partial components
        DY = 0.0
        T1 = temp[i]
        T2 = temp[i+1]
        dxi = dx[i]
        for j = 1:m
            y1  = y[i,j]
            f1=dydt(T1, y1, pe,j,m)
            f2=dydt(T2, y1 + dt * f1, pe,j,m)
            y[i+1,j] = y1 + dt * ( f1+f2) / 2.0
            DY += -pe[j+3*m]*dydt(T1,y1,pe,j,m)
        end
        if dxi > max_rate
            max_rate = dxi
        end
        OF += (dxi-DY)^2
    end
    return OF/(max_rate^2*n)
end

function minimize!(p,pall,indmap,pmat,m,n_d,np,T,DX,N,Y,DT,Y0;
    integrator=heun!)
    """
    The objective function to be minimized by some
    minimization algorithm.
    Change the integrator here to switch from heun!() to rk4!().
    """
    OF  = 0.0
    construct_pmat!(p,pall,indmap,pmat,n_d,m,np)
    for i = 1:n_d
        pe = pmat[i]
        OF += integrator(DT[i],Y0[i],N[i],m,T[i],DX[i],Y[i],pe)
    end
    return OF
end



function rk4!(dt,y0,n,m,temp,dx,y,pe) :: Float64
    """Same as heun!(), but using a 4th order Runge-Kutta method to integrate
    along the temperature profile.
    At the same time, subtract the calculated conversion rate and
    from the experimental conversion rate to directly calculate the deviation
    from the two, which is minimized by the optimization algorithm to find the
    best kinetic parameters.
    Interpolates linearly between temperature values.
    """
    @views y[1,:]=y0
    OF ::Float64 = 0.0
    max_rate = 0.0
    for i = 1:n-1
        # sum partial components
        DY = 0.0
        T1 = temp[i]
        T2 = temp[i+1]
        dxi = dx[i]
        for j = 1:m
            y1  = y[i,j]
            f1=dydt(T1, y1, pe, j, m)
            f2=dydt((T1 + T2) / 2.0, y1 + dt * f1 / 2.0, pe, j, m)
            f3=dydt((T1 + T2) / 2.0, y1 + dt * f2 / 2.0, pe, j, m)
            f4=dydt(T2, y1 + dt * f3, pe, j, m)
            y[i+1,j] = y1 + dt * ( f1 + 2.0 * f2 + 2.0 * f3 + f4 ) / 6.0
            DY += -pe[j+3*m]*dydt(T1,y1,pe,j,m)
        end
        if dxi > max_rate
            max_rate = dxi
        end
        OF += (dxi-DY)^2
    end
    return OF/(max_rate^2*n)
end

function plot_res()
    """ Crude plot of the results.
    """
        plt=1
        for i = 1:n_d
                plt=plot_result(plt,i,X[i],DX[i],N[i],m,res.minimizer,
                T[i],X[i][:,1],DT[i],Y0[i])
        end
end

function mapper(pmat,tpe)
        """ construct a 1D array from a 2D array of parameter values, e.g.

        PMAT = [[A1, E1, n1, c1],
                [A2, E2, n2, c2]]

        and extract the parameter values that are to be passed to
        optimization algorithm. Doing that, we can account for
        some parameters that set as constants during optimization,
        some parameters that are common for severals experiments (i.e.
        1 parameters describes the kinetics of multiple experiments) or
        that some parameters are individual for one experiments (i.e. 1 parameter
        is used to describe the kinetics of 1 experiment). These are described
        by the TPE array, where 1 = common parameters, 2 = individual parameters
        and 3 = fixed parameters.

        Outputs:
                par - containing the parameters to be optimized
                pfxd - containing the fixed parameter values
                indmap - indices accesing the optimization element values
        """
        # 1D array to store values to optimize which is taken in
        # by optimizer
        pvar = Array{Float64,1}()
        pfxd = Array{Float64,1}()
        pfxd_ndx = Array{Array{Int64,1},1}()
        # matrix to keep track of seen paramters
        n_experiments, n_components, n_params = size(pmat)
        added = zeros(Int64,n_components,n_params)
        # keep track of mapping of pmat to par
        indmap = tpe*0
        cnt = 0
        for ne in 1:n_experiments
                for nc in 1:n_components
                        for np in 1:n_params
                                if tpe[ne,nc,np]==1 #&& added[nc,np] != 1
                                        if added[nc,np] < 1
                                                # if common parameter and if not added
                                                # to array previously, add the value
                                                push!(pvar, pmat[ne,nc,np]) ; cnt += 1
                                                added[nc,np] = cnt
                                                indmap[ne,nc,np] = cnt
                                        else
                                                # optimization value allready added
                                                # but keep track of mapping
                                                indmap[ne,nc,np] = added[nc,np]
                                        end
                                elseif tpe[ne,nc,np]==2
                                        # if individual optimize, add the value
                                        # allways
                                        push!(pvar, pmat[ne,nc,np]) ; cnt += 1
                                        indmap[ne,nc,np] = cnt
                                elseif tpe[ne,nc,np]==3
                                        # trouble here since value is not
                                        # and index of par ...
                                        push!(pfxd, pmat[ne,nc,np])
                                        push!(pfxd_ndx, [ne,nc,np])

                                end
                        end
                end
        end
        pall = pvar[:]
        for (i,ndx) in enumerate(pfxd_ndx)
                push!(pall,pfxd[i]) ; cnt += 1
                indmap[ndx...] = cnt
        end
        return pvar, pall, indmap
end




function construct_pmat!(p, pall, indmap, pmat,
                         n_exp, n_comp, n_part)
        """
        Add the optimization elements contained in p
        to an array of fixed format resulting in an array
        containing the optimization elements in addition to the fixed values.

        p       : array, usually passed from optimization algorithm
        pall    : array, containing all values of p and the fixed parameter values
        indmap  : the mapping created from the mapper() function
        pmat    : the output matrix containing the optimization parameters
                  and the fixed parameters.
        """
        pall[1:length(p)] .= p
        for e in 1:n_exp
            pmat[e] .= @view pall[indmap[e,:,:]]
        end
        return pmat
end


function toSVector(pmat)
    ne,nc,np=size(pmat)
    pmat = [ pmat[e,:,:] for e in 1:ne]
    return pmat
end

function savetxt(fname)
        """ Save calculated TGA data along with experimental data."""

        for e in 1:ne
                f = open(fname*"$e.txt", "w")
                write(f,"# Simulated TGA data with $nc partial components.\n")
                write(f, "# Format: component [A, E, n, c, ...(extra_params)... ] \n")
                out = 0.0*Y[e][:,1]
                DC = []
                for c in 1:nc
                        push!(DC, [-pmat[e][c+3*nc]*dydt(T1,y1,pmat[e],c,nc) for (T1,y1) in zip(T[e],Y[e][:,c])])
                        write(f, "# comp. $c " * string(round.(pmat[e][c,:],digits=4))*"\n")
                        out .= out .+ Y[e][:,c]*pmat[e][c,end]
                end
                # write to file
                write(f,"#--------------------------------------------------------------\n")
                write(f, "# time (min)   Temp (°C)   exp_mass   calc_mass   -exp_rate")
                to_write = [X[e][:,1]     X[e][:,2]  X[e][:,3]  out         DX[e]]
                dsum=0.0*DC[1]
                for c in 1:nc
                    write(f,"   -calc_rate_comp.$c")
                    to_write=cat(to_write,DC[c],dims=2)
                    dsum .= dsum .+ DC[c]
                end
                write(f,"   -sum_calc_rate \n")
                to_write = cat(to_write,dsum,dims=2)
                to_write = round.(to_write,digits=6)
                writedlm(f,to_write, "    ")

        end
end

function load_data(filenames, smoothing_factors, normalizing_temperature,
        cut, cols, skiprows)
        """ Load, cut, normalize and calculate the derivative of the raw data.

                filenames : Load the time, temperature and mass data from these
                            files.

                smoothing_factors : the smoothing factors used in the smoothing
                                    splines for calculating deriavtive.

                normalizing_tempearture : Normalize the mass data using the
                                          masses at these temperatures.

                cut : consider only data points in the range cut[min_T, max_T]

                cols : the time, temp and mass data are containted in cols[1],
                       cols[2] and cols[3], respectively.

                skiprows : do not read this amount of rows from the beginning
                           of the file, which are e.g. comments and other
                           non-numerical values.
        """
        # Array for mass conversion
        X = []
        # array for temperatures
        T = Array{Array{Float64,1},1}()
        # array for mass conversion rate
        DX = Array{Array{Float64,1},1}()
        for (i,file) in enumerate(filenames)
                x = DelimitedFiles.readdlm(file,comments=true,comment_char='#',skipstart=skiprows)
                normalizing_mass = x[:,cols[3]][x[:,cols[2]] .> normalizing_temperature[i]][1]
                #x = x[1:floor(Int,length(x[:,1])/500):end,:]
                println(normalizing_mass)
                ndx = ( cut[i][2] .> x[:,cols[2]].> cut[i][1])
                x = Matrix([x[:,cols[1]][ndx] x[:,cols[2]][ndx] x[:,cols[3]][ndx]/normalizing_mass])
                #x = Matrix([x[:,1][ndx]*60 x[:,2][ndx] x[:,3][ndx]])
                push!(X,x)
                # °C to kelving
                push!(T,x[:,2].+273.15)
                spl = Spline1D(x[:,1],x[:,3],s=smoothing_factors[i])
                dx = -derivative(spl,x[:,1])
                dx = dx
                #plot(x[:,1],dx)
                push!(DX,dx)
        end
        return X,T,DX
end
