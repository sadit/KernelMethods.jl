module KMap
using SimilaritySearch:
    Sequential, KnnResult, search, clear!, Item

export kmap, centroid, partition, knearestreferences, sequence

include("enet.jl")
include("dnet.jl")
include("criterions.jl")

"""
Transforms `objects` to a new representation space induced by `(refs, dist, kernel)`
- `refs` a list of references
- `kernel` a kernel function (and an embedded distance) with signature (T, T) -> Float64
"""
function kmap(objects::AbstractVector{T}, kernel, refs::AbstractVector{T}) where {T}
    # X = Vector{T}(length(objects))
    m = Vector{T}(length(objects))
    @inbounds for i in 1:length(objects)
        u = Vector{Float64}(length(refs))
        obj = objects[i]
        for j in 1:length(refs)
            u[j] = kernel(obj, refs[j])
        end

        m[i] = u
    end

    return m
end

"""
Groups items in `objects` using a nearest neighbor rule over `refs`.
The output is controlled using a callback function. The call is performed in `objects` order.

- `callback` is a function that is called for each `(objID, refItem)`
- `objects` is the input dataset
- `dist` a distance function \$(T, T) \\rightarrow \Re\$
- `refs` the list of references
- `k` specifies the number of nearest neighbors to use
- `indexclass` specifies the kind of index to be used, a function receiving `(refs, dist)` as arguments,
    and returning the new metric index

Please note that each object can be related to more than one group \$k > 1\$ (default \$k=1\$)
"""
function partition(callback::Function, objects::AbstractVector{T}, dist, refs::AbstractVector{T}; k::Int=1, indexclass=Sequential) where T
    index = indexclass(refs, dist)
    res = KnnResult(k)
    for i in 1:length(objects)
        clear!(res)
        search(index, objects[i], res)
        for p in res
            callback(i, p)
        end
    end
end

function invindex(objects::AbstractVector{T}, dist, refs::AbstractVector{T}; k::Int=1, indexclass=Sequential) where T
    π = [Vector{Int}() for i in 1:length(refs)]
    partition((i, p) -> push!(π[p.objID], i), objects, dist, refs, k=k, indexclass=indexclass)
    π
end

function sequence(objects::AbstractVector{T}, dist, refs::AbstractVector{T}; indexclass=Sequential) where T
    s = Vector{Int}(length(objects))
    partition((i, p) -> begin s[i] = p.objID end, objects, dist, refs, indexclass=indexclass)
    s
end

function knearestreferences(objects::AbstractVector{T}, dist, refs::AbstractVector{T}; indexclass=Sequential) where T
    s = Vector{Vector{Int}}(length(objects))
    partition((i, p) -> s[i] = [p.objID for p in res], objects, dist, refs, indexclass=indexclass)
    s
end

"""
Computes the centroid of the list of objects

- Use the dot operator (broadcast) to convert several groups of objects
"""
function centroid(objects::AbstractVector{Vector{F}})::Vector{F} where {F <: AbstractFloat}
    u = copy(objects[1])
    @inbounds for i in 2:length(objects)
        w = objects[i]
        for j in 1:length(u)
            u[j] += w[j]
        end
    end

    f = 1.0 / length(objects)
    @inbounds for j in 1:length(u)
        u[j] *= f
    end

    return u
end


end
