module KMap
using SimilaritySearch:
    Sequential, KnnResult, search, clear!, Item

export kmap, spartition, apartition, centroid

include("fft.jl")
include("criterions.jl")

"""
Transforms `objects` to a new representation space induced by `(refs, dist, kernel)`
- `refs` a list of references
- `dkernel` a distance function (and an embedded kernel) with signature (T, T) -> Float64
"""
function kmap(objects::AbstractVector{T}, dkernel, refs::AbstractVector{T}) where {T}
    # X = Vector{T}(length(objects))
    m = Vector{T}(length(objects))
    @inbounds for i in 1:length(objects)
        u = Vector{Float64}(length(refs))
        obj = objects[i]
        for j in 1:length(refs)
            u[j] = dkernel(obj, refs[j])
        end

        m[i] = u
    end

    return m
end

"""
Computes the `k` nearest references to each object.
The output is a matrix \$(k, |objects|)\$ of `Item` objects (SimilaritySearch), where the i-th column corresponds to the `k` nearest refences of the i-th object in `objects`.
"""
function spartitionitems(objects::AbstractVector{T}, dist, refs::AbstractVector{T}; k::Int=1, indexclass=Sequential) where T
    s = Vector{Vector{Item}}(length(objects))
    index = indexclass(refs, dist)
    res = KnnResult(k)
    for i in 1:length(objects)
        clear!(res)
        search(index, objects[i], res)
        s[i] = collect(Item, res)
    end
    s
end

function spartition(objects::AbstractVector{T}, dist, refs::AbstractVector{T}; indexclass=Sequential) where T
    s = Vector{Int}(length(objects))
    index = indexclass(refs, dist)
    res = KnnResult(1)
    for i in 1:length(objects)
        clear!(res)
        search(index, objects[i], res)
        s[i] = first(res).objID
    end
    s
end

function spartition(objects::AbstractVector{T}, dist, refs::AbstractVector{T}, k::Int; indexclass=Sequential) where T
    s = Vector{Vector{Int}}(length(objects))
    index = indexclass(refs, dist)
    res = KnnResult(k)
    for i in 1:length(objects)
        clear!(res)
        search(index, objects[i], res)
        s[i] = [p.objID for p in res]
    end
    s
end
"""
Groups items in `objects` using a nearest neighbor rule over `refs`.
The output is an inverted index pointing from `refs` to those objects in `objects`
having a reference as its nearest neighbor.

- Each object can be related to more than one group \$k > 1\$ (default \$k=1\$)
"""
function apartitionitems(objects::Vector{T}, dist, refs::Vector{T}; k::Int=1, indexclass=Sequential) where T
    π = [Vector{Item}() for i in 1:length(refs)] 
    index = indexclass(refs, dist)
    res = KnnResult(k)
    for i in 1:length(objects)
        clear!(res)
        search(index, objects[i], res)
        for p in res
            push!(π[p.objID], p)
        end
    end

    π
end

function apartition(objects::Vector{T}, dist, refs::Vector{T}; k::Int=1, indexclass=Sequential) where T
    π = [Vector{Int}() for i in 1:length(refs)]
    index = indexclass(refs, dist)
    res = KnnResult(k)
    for i in 1:length(objects)
        clear!(res)
        search(index, objects[i], res)
        for p in res
            push!(π[p.objID], i)
        end
    end

    π
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
