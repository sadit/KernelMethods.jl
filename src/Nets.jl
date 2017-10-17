using SimilaritySearch
using TextModel
using JSON
using DataStructures

include("/home/job/KDENets-FC/src/kernels.jl")

type Net
   data::Vector{Vector{Float64}}
   labels::Vector{Int32}
   references::Vector{Int32}
   partitions::Vector{Int32}
   centers::Vector{Vector{Float64}}
   centroids::Vector{Vector{Float64}}
   dists::Vector{Float64}
   csigmas::Vector{Float64}
   sigmas::Dict{Int64,Float64}
   stats::Dict{String,Float64}
end

Net(data,labels)=Net(data,labels,[],[],[],[],[],[],Dict(),Dict())

function cosine(x1,x2)
    xc1=DenseCosine(x1)
    xc2=DenseCosine(x2)
    d=CosineDistance()
    return d(xc1,xc2)
end

function maxmin(data,centers,ind,index::KnnResult,distance_function,partitions,lK)
    c=last(centers)
    if length(index)==0
        for i in ind
            if i!=c
                push!(index,i,Inf)
            end
        end
    end
    nindex=KnnResult(length(index))
    for fn in index
        dist=distance_function(data[fn.objID],data[c])
        push!(lK[fn.objID],dist)
        dist = if (dist<fn.dist) dist else fn.dist end
        partitions[fn.objID] = if (dist<fn.dist) c else partitions[fn.objID] end
        if fn.objID!=c
            push!(nindex,fn.objID,dist)
        end
    end
    index.k=nindex.k
    index.pool=nindex.pool
    fn=pop!(index)
    return fn.objID,fn.dist
end

function get_centroids(data,partitions)
    centroids=[]
    centers=sort([j for j in Set(partitions)])
    for c in centers
        ind=[i for (i,v) in enumerate(partitions) if v==c]
        push!(centroids,mean(data[ind]))
    end
    return centroids
end

function enet(N::Net,num_of_centers::Int64; distance_function=L2Distance(), per_class=false)
    n=length(N.data)
    lK,partitions=[[] for i in 1:n],[0 for i in 1:n]
    gcenters,dists,sigmas=[],[],Dict()
    indices=[[i for i in  1:n]]
    for ind in indices
        centers=[]
        s=rand(1:length(ind))
        push!(centers,ind[s])
        #ll=N.labels[ind[s]]
        index=KnnResult(length(ind))
        partitions[ind[s]]=ind[s]
        k=1
        while  k<=num_of_centers-1 && k<=length(ind)
            fnid,d=maxmin(N.data,centers,ind,index,distance_function,partitions,lK)
            push!(centers,fnid)
            push!(dists,d)
            partitions[fnid]=fnid
            k+=1
        end
        sigmas[0]=minimum(dists)
        gcenters=vcat(gcenters,centers)
    end
    N.references,N.partitions,N.dists,N.sigmas=gcenters,partitions,dists,sigmas
    N.centers,N.centroids=N.data[gcenters],get_centroids(N.data,partitions)
    N.csigmas,N.stats=get_csigmas(N.data,N.centroids,N.partitions)
end

function kmpp(N::Net,num_of_centers::Int64)
    n=length(N.data)
    s=rand(1:n)
    centers,d=[s],L2SquaredDistance()
    D=[d(N.data[j],N.data[s]) for j in 1:n]
    for i in 1:num_of_centers-1
        cp=cumsum(D/sum(D))
        r=rand()
        sl=[j for j in 1:length(cp) if cp[j]>=r]
        #println("DDDDDDDDDDDDDD",D)
        #println("sum ",sum(D))
        #while length(sl)==0
        #    sl=[j for j in 1:length(cp) if cp[j]>=r]
        #end
        s=sl[1]
        push!(centers,s)
        for j in 1:n
            dist=d(N.data[j],N.data[s])
            if dist<D[j]
                D[j]=dist
            end
        end
    end
    return centers
end

function assign(data,centroids,partitions)
    d=L2SquaredDistance()
    for i in 1:length(data)
        partitions[i]=sortperm([d(data[i],c) for c in centroids])[1]
    end
end

function distances(data,centroids,partitions)
    dists=[]
    df=L2SquaredDistance()
    for i in 1:length(centroids)
        ind=[j for (j,l) in enumerate(partitions) if l==i]
        if length(ind)>0
            X=data[ind]
            dd=[df(centroids[i],x) for x in X]
            push!(dists,maximum(dd))
        end
    end
    sort!(dists)
    return dists
end

function get_csigmas(data,centroids,partitions;distance_function=L2SquaredDistance())
    csigmas,stats=[],Dict("SSE"=>0.0,"BSS"=>0.0)
    refs=sort([j for j in Set(partitions)])
    df=distance_function
    m=mean(data)
    for i in refs
        ind=[j for (j,l) in enumerate(partitions) if l==i]
        #if length(ind)>0
        X=data[ind]
        dd=[df(data[i],x) for x in X]
        push!(csigmas,maximum(dd))
        stats["SSE"]+=sum(dd)
        stats["BSS"]+=length(X)*(sum(mean(X)-m))^2
        #end
    end
    return csigmas,stats
end

function kmnet(N::Net,num_of_centers::Int64; max_iter=1000)
    n=length(N.data)
    lK,partitions=[[] for i in 1:n],[0 for i in 1:n],[0 for i in 1:n]
    dists,sigmas=[],Dict()
    init=kmpp(N,num_of_centers)
    centroids=N.data[init]
    i,aux=1,[]
    while centroids != aux && i<max_iter
        i=i+1
        aux = centroids
        assign(N.data,centroids,partitions)
        centroids=get_centroids(N.data,partitions)
    end
    dists=distances(N.data,centroids,partitions)
    N.partitions,N.dists,N.sigmas=partitions,dists,sigmas
    N.centroids,N.sigmas[0]=centroids,maximum(N.dists)
    N.csigmas,N.stats=get_csigmas(N.data,N.centroids,N.partitions)
    N.sigmas[0]=maximum(N.csigmas)
end

function dnet(N::Net,num_of_elements::Int64; distance_function=L2Distance())
    n,d,k=length(N.data),distance_function,num_of_elements
    lK,partitions,references=[[] for i in 1:n],[0 for i in 1:n],[]
    dists,sigmas=[],Dict()
    while 0 in partitions
        pending=[j for (j,v) in enumerate(partitions) if partitions[j]==0]
        s=rand(pending)
        partitions[s]=s
        pending=[j for (j,v) in enumerate(partitions) if partitions[j]==0]
        push!(references,s)
        pc=sortperm([d(N.data[j],N.data[s]) for j in pending])
        if length(pc)>=k
            partitions[pending[pc[1:k]]]=s
        else
            partitions[pending[pc]]=s
        end
    end
    N.references,N.partitions=references,partitions
    N.centers,N.centroids=N.data[references],get_centroids(N.data,partitions)
    N.csigmas,N.stats=get_csigmas(N.data,N.centroids,N.partitions)
    N.sigmas[0]=maximum(N.csigmas)
end

function gen_cfeatures(Xo,N::Net; kernel=linear)
    Xm,sigmas,Xr=N.centroids,N.csigmas,[]
    nc=length(Xo)
    for xi in Xo
        xd=[]
        for (j,x) in enumerate(Xm)
            push!(xd, kernel(xi,x,sigma=sigmas[j]))
        end
        push!(Xr,xd)
    end
    return [Float64.(x) for x in Xr]
end
