module Nets
export Net, enet, kmnet, dnet, gen_features, KernelClassifier
import KernelMethods.Kernels: sigmoid, gaussian, linear, cuchy
import KernelMethods.Scores: accuracy, recall
import KernelMethods.Supervised: NearNeighborClassifier, optimize!, predict_one, predict_one_proba
using SimilaritySearch
using TextModel
#using JSON
#using DataStructures

type Net
    data::Vector{Vector{Float64}}
    labels::Vector{Any}
    references::Vector{Int32}
    partitions::Vector{Int32}
    centers::Vector{Vector{Float64}}
    centroids::Vector{Vector{Float64}}
    dists::Vector{Float64}
    csigmas::Vector{Float64}
    sigmas::Dict{Int64,Float64}
    stats::Dict{String,Float64}
    reftype::Symbol
    distance
    kernel
end

Net(data,labels)=Net(data,labels,[],[],[],[],[],[],Dict(),Dict(),
                     :centroids,L2SquaredDistance(),gaussian)

function cosine(x1,x2)
    xc1=DenseCosine(x1)
    xc2=DenseCosine(x2)
    d=CosineDistance()
    return d(xc1,xc2)
end

function maxmin(data,centers,ind,index::KnnResult,distance,partitions,lK)
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
        dist=distance(data[fn.objID],data[c])
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

# Epsilon Network using farthest first traversal Algorithm

function enet(N::Net,num_of_centers::Int64; distance=L2SquaredDistance(), 
              per_class=false,reftype=:centroids, kernel=linear)
    N.distance=distance
    N.kernel=kernel
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
            fnid,d=maxmin(N.data,centers,ind,index,distance,partitions,lK)
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
    N.csigmas,N.stats=get_csigmas(N.data,N.centroids,N.partitions,distance=N.distance)
    N.reftype=reftype
end


# KMeans ++ seeding Algorithm 

function kmpp(N::Net,num_of_centers::Int64)
    n=length(N.data)
    s=rand(1:n)
    centers,d=[s],L2SquaredDistance()
    D=[d(N.data[j],N.data[s]) for j in 1:n]
    for i in 1:num_of_centers-1
        cp=cumsum(D/sum(D))
        r=rand()
        sl=[j for j in 1:length(cp) if cp[j]>=r]
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

#Assign Elementes to thier  nearest centroid

function assign(data,centroids,partitions;distance=L2SquaredDistance())
    d=distance
    for i in 1:length(data)
        partitions[i]=sortperm([d(data[i],c) for c in centroids])[1]
    end
end

#Distances for each element to its nearest cluster centroid

function get_distances(data,centroids,partitions;distance=L2SquaredDistance())
    dists=[]
    for i in 1:length(centroids)
        ind=[j for (j,l) in enumerate(partitions) if l==i]
        if length(ind)>0
            X=data[ind]
            dd=[distance(centroids[i],x) for x in X]
            push!(dists,maximum(dd))
        end
    end
    sort!(dists)
    return dists
end

#Calculated the sigma for each ball

function get_csigmas(data,centroids,partitions;distance=L2SquaredDistance())
    csigmas,stats=[],Dict("SSE"=>0.0,"BSS"=>0.0)
    refs=sort([j for j in Set(partitions)])
    df=distance
    m=mean(data)
    for i in refs
        ind=[j for (j,l) in enumerate(partitions) if l==i]
        #if length(ind)>0
        X=data[ind]
        dd=[df(data[i],x) for x in X]
        push!(csigmas,max(0,maximum(dd)))
        stats["SSE"]+=sum(dd)
        stats["BSS"]+=length(X)*(sum(mean(X)-m))^2
        #end
    end
    return csigmas,stats
end

#Feature generator using kmeans centroids

function kmnet(N::Net,num_of_centers::Int64; max_iter=1000,kernel=linear,distance=L2SquaredDistance())
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
    N.distance=distance
    dists=get_distances(N.data,centroids,partitions,distance=N.distance)
    N.partitions,N.dists,N.sigmas=partitions,dists,sigmas
    N.centroids,N.sigmas[0]=centroids,maximum(N.dists)
    N.csigmas,N.stats=get_csigmas(N.data,N.centroids,N.partitions,distance=N.distance)
    N.sigmas[0]=maximum(N.csigmas)
    N.reftype=:centroids
    N.kernel=kernel
end

#Feature generator using naive algoritmh for density net

function dnet(N::Net,num_of_elements::Int64; distance=L2SquaredDistance(),kernel=linear)
    n,d,k=length(N.data),distance,num_of_elements
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
    N.csigmas,N.stats=get_csigmas(N.data,N.centroids,N.partitions,distance=N.distance)
    N.sigmas[0]=maximum(N.csigmas)
    N.reftype=:centroid
    N.distance=distance
    N.kernel=kernel
end

#Generates feature espace using cluster centroids or centers

function gen_features(Xo,N::Net)
    sigmas,Xr=N.csigmas,[]
    Xm = N.reftype==:centroids || length(N.centers)==0 ? N.centroids : N.centers 
    nc=length(Xo)
    kernel=N.kernel
    for xi in Xo
        xd=[]
        for (j,x) in enumerate(Xm)
            push!(xd, kernel(xi,x,sigma=sigmas[j],distance=N.distance))
        end
        push!(Xr,xd)
    end
    return [Float64.(x) for x in Xr]
end

function KlusterClassifier(Xe,Ye; op_function=accuracy, 
                           K=[4,8,16,32,128],
                           kernels=[gaussian,sigmoid,linear],
                           runs=5, trainratio=0.7, testratio=0.3,folds=0)
    top=[]
    DNNC=Dict()
    distances=[("L2",L2SquaredDistance()),("cosine",cosine)]
    for k in K
        for (nettype,compute_net) in ["epsilon"=>enet,"kmeans"=>kmnet,"density"=>dnet]
            for kernel in kernels
                for (distancek,distanceN) in distances
                    if distancek=="cosine" && nettype=="kmeans"
                        continue
                    else
                        N=Net(Xe,Ye)
                        compute_net(N,k,kernel=kernel,distance=distanceN)
                        X=gen_features(N.data,N)
                    end
                    for (kd,distance) in distances 
                        nnc = NearNeighborClassifier(X,Ye, distance)
                        opval,kknn,w=optimize!(nnc, op_function,runs=runs, trainratio=trainratio, 
                                               testratio=testratio,folds=folds)[1]
                        kernelname=Symbol(kernel)
                        key="$nettype/$kernelname/$k/KNN$kknn/$kd"
                        #println("##### ",key)
                        push!(top,(opval,key))
                        DNNC[key]=(nnc,N)
                    end
                end
            end
        end
    end
    top=sort(top,rev=true)[1:10]
    LN=[(DNNC[t[2]],t[1],t[2]) for t in top ]
end

function predict(knc,X;ensemble_k=1)
    cl,N,desc=knc[1]
    xv=gen_features(X,N)
    y_pred=[predict_one(cl,x)[1].first for x in xv]
    y_pred
end

function predict_proba(knc,X;ensemble_k=1)
    cl,N,desc=knc[1]
    xv=gen_features(X,N)
    y_pred=[predict_one_proba(cl,x) for x in xv]
    y_pred
end

end
