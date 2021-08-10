function dydt2(T::Float64,y::Float64,pj::Array{Float64,1},j::Int64,m::Int64) ::Float64
    #if y < 0.0
    #    return 0.0
    #else
        return @inbounds -(10^pj[j])*exp(-pj[j+m]/(8.3145*(T)))*y^pj[j+2*m]
    #end
end

@inline function dydt(T::Float64,y::Float64,pj::Array{Float64,2},j::Int64,m::Int64) ::Float64
    if y < 0.0
        return 0.0
    else
        return @inbounds -10^(pj[j])*exp(-(pj[j+m])/(8.3145*(T)))*y^pj[j+2*m]
        #return @inbounds -10^(pj[j]+pj[j+4*m]*T)*exp(-(pj[j+m]+pj[j+5*m]*T)/(8.3145*(T)))*y^pj[j+2*m]
    end
end


function dydt(T::Float64,y::Float64,pj::Array{Float64,1}) ::Float64
    if y < 0.0
        return 0.0
    else
        return @inbounds -(10^pj[1])*exp(-pj[2]/(8.3145*(T)))*y^pj[3]
    end
end


function heun!(dt,y0,n,m,temp,dx,y,pe) :: Float64
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
            f2=dydt(T1, y1 + dt * f1, pe,j,m)
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


function heun2!(dt,y0,n,m,temp,dx,y,
    f1::Float64,f2::Float64,p)
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
                f1=dydt2(T1, y1, p, j, m)
                f2=dydt2(T1, y1 + dt * f1, p, j, m)
                y[i+1,j] = y1 + dt * ( f1+f2) / 2.0
                DY += -1/p[j+3*m]*dydt2(T1,y1,p,j,m)
        end
        if dxi > max_rate
            max_rate = dxi
        end
        OF += (dxi-DY)^2
    end
    return OF/(max_rate^2*n)
end

function get_maxrate(dxi::Float64,max_rate::Float64) :: Float64
        return max(max_rate,dxi)
end

function minimize!(p,pall,indmap,pmat,m,n_d,np,T,DX,N,Y,DT,Y0)
    OF  = 0.0
    construct_pmat!(p,pall,indmap,pmat,n_d,m,np)
    for i = 1:n_d
        pe = pmat[i]
        OF += heun!(DT[i],Y0[i],N[i],m,T[i],DX[i],Y[i],pe)
    end
    return OF
end

function minimize2!(p,m,n_d,T,DX,N,Y,DT,Y0)
    f1 = 0.0 ; f2 = 0.0 ; f3 = 0.0 ; f4 = 0.0
    OF = 0
    for i = 1:n_d
        OF += heun2!(DT[i],Y0[i],N[i],m,T[i],DX[i],Y[i],f1,f2,p)
    end
    return OF
end

function rk4!(dt,y0,n,m,temp,dx,y,
    f1::Float64,f2::Float64,f3::Float64,f4::Float64,p)
    @views y[1,:]=y0
    OF = 0.0
    for i = 1:n-1
        DY = 0.0
        for j = 1:m
            T1 = temp[i]
            T2 = temp[i+1]
            y1  = y[i,j]
                f1=dydt(T1, y1, p, j, m)
                f2=dydt((T1 + T2) / 2.0, y1 + dt * f1 / 2.0, p, j, m)
                f3=dydt((T1 + T2) / 2.0, y1 + dt * f2 / 2.0, p, j, m)
                f4=dydt(T2, y1 + dt * f3, p, j, m)
                y[i+1,j] = y1 + dt * ( f1 + 2.0 * f2 + 2.0 * f3 + f4 ) / 6.0
                DY += -p[j+3*m]*dydt(T1,y1,p,j,m)
        end
        OF += (dx[i]-DY)^2
    end
    return OF
end

function plot_res()
        plt=1
        for i = 1:n_d
                plt=plot_result(plt,i,X[i],DX[i],N[i],m,res.minimizer,
                T[i],X[i][:,1],DT[i],Y0[i])
        end
end

function mapper(pmat,tpe)
        # Gives rise to two arrays:
                # par which is for optimization
                # pfxd which is the array for fixed values
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


function construct_pmat__!(p, pall, indmap, pmat)
        """
        Add the optimization elements contained in p
        to the array containing the optimization elements
        in addition to the fixed values.

        p       : array, usually passed from optimization algorithm
        pall    : array, containing all values of p and the additional fixed values
        indmap  : the mapping created from the mapper() function
        pmat    : the output matrix containing the optimization parameters
                and the fixed parameters.
        """
        pall[1:length(p)] .= p
        for (i,val) in enumerate(indmap)
                pmat[i]=pall[val]
        end
end

function construct_pmat__!(p, pall, indmap, pmat,
                         n_exp, n_comp, n_part)
        """
        Add the optimization elements contained in p
        to the array containing the optimization elements
        in addition to the fixed values.

        p       : array, usually passed from optimization algorithm
        pall    : array, containing all values of p and the additional fixed values
        indmap  : the mapping created from the mapper() function
        pmat    : the output matrix containing the optimization parameters
                and the fixed parameters.
        """
        for e in n_exp
            for c in n_comp
                for p in n_part
                    pmat[e][c][p]=pall[indmap[e,c,p]]
                end
            end
        end
end

function construct_pmat!(p, pall, indmap, pmat,
                         n_exp, n_comp, n_part)
        """
        Add the optimization elements contained in p
        to the array containing the optimization elements
        in addition to the fixed values.

        p       : array, usually passed from optimization algorithm
        pall    : array, containing all values of p and the additional fixed values
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
