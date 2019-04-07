module KMap
using SimilaritySearch:
    Sequential, KnnResult, search, empty!, Item

#using TextModel

export kmap, centroid!, partition, knearestreferences, sequence

include("enet.jl")
include("dnet.jl")
include("criterions.jl")
include("kclass.jl")

"""
Transforms `objects` to a new representation space induced by ``(refs, dist, kernel)``
- `refs` a list of references
- `kernel` a kernel function (and an embedded distance) with signature ``(T, T) \\rightarrow Float64``
"""
function kmap(objects::AbstractVector{T}, kernel, refs::AbstractVector{T}) where {T}
    # X = Vector{T}(length(objects))
    m = Vector{Vector{Float64}}(undef, length(objects))
    @inbounds for i in 1:length(objects)
        u = Vector{Float64}(undef, length(refs))
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
- `dist` a distance function ``(T, T) \\rightarrow \\mathbb{R}``
- `refs` the list of references
- `k` specifies the number of nearest neighbors to use
- `indexclass` specifies the kind of index to be used, a function receiving `(refs, dist)` as arguments,
    and returning the new metric index

Please note that each object can be related to more than one group ``k > 1`` (default ``k=1``)
"""
function partition(callback::Function, dist::Function, objects::AbstractVector{T}, refs::AbstractVector{T}; k::Int=1, indexclass=Sequential) where T
    index = fit(Sequential, refs)
    res = KnnResult(k)
    for i in 1:length(objects)
        empty!(res)
        search(index, dist, objects[i], res)
        for p in res
            callback(i, p)
        end
    end
end

"""
Creates an inverted index from references to objects.
So, an object ``u`` is in ``r``'s posting list iff ``r``
is among the ``k`` nearest references of ``u``.

"""
function invindex(dist::Function, objects::AbstractVector{T}, refs::AbstractVector{T}; k::Int=1, indexclass=Sequential) where T
    π = [Vector{Int}() for i in 1:length(refs)]
    partition((i, p) -> push!(π[p.objID], i), dist, objects, refs, k=k, indexclass=indexclass)
    π
end

"""
Returns the nearest reference (identifier) of each item in the dataset
"""
function sequence(dist::Function, objects::AbstractVector{T}, refs::AbstractVector{T}; indexclass=Sequential) where T
    s = Vector{Int}(length(objects))
    partition((i, p) -> begin s[i] = p.objID end, dist, objects, refs, indexclass=indexclass)
    s
end

"""
Returns an array of k-nearest neighbors for `objects`
"""
function knearestreferences(dist::Function, objects::AbstractVector{T}, refs::AbstractVector{T}; indexclass=Sequential) where T
    s = Vector{Vector{Int}}(length(objects))
    partition((i, p) -> s[i] = [p.objID for p in res], dist, objects, refs, indexclass=indexclass)
    s
end

"""
Computes the centroid of the list of objects

- Use the dot operator (broadcast) to convert several groups of objects
"""
function centroid!(objects::AbstractVector{Vector{F}})::Vector{F} where {F <: AbstractFloat}
    u = copy(objects[1])
    @inbounds for i in 2:length(objects)
        w = objects[i]
        @simd for j in 1:length(u)
            u[j] += w[j]
        end
    end

    f = 1.0 / length(objects)
    @inbounds @simd for j in 1:length(u)
        u[j] *= f
    end

    return u
end

#=
"""
Computes the centroid of a collection of `DenseCosine` vectors.
It don't destroys the input array, however, the VBOW version does it
"""

function centroid!(vecs::AbstractVector{DenseCosine{F}}) where {F <: AbstractFloat}
    # info("** COMPUTING the centroid of $(length(vecs)) items")
    m = length(vecs[1].vec)
    w = zeros(F, m)
    
    for vv in vecs
        v::Vector{F} = vv.vec
        @inbounds @simd for i in 1:m
            w[i] += v[i]
        end
    end

    DenseCosine(w)
end


"""
Computes a centroid-like sparse vector (i.e., a center under the angle distance) for a collection of sparse vectors.
The computation destroys input array to reduce memory allocations.
"""
function centroid!(vecs::AbstractVector{VBOW})
	lastpos = length(vecs)
	while lastpos > 1
		pos = 1
		for i in 1:2:lastpos
			if i < lastpos
				vecs[pos] = vecs[i] + vecs[i+1]
			end
			pos += 1
		end
		lastpos = pos - 1
	end
	
    vecs[1]
end
=#

end
