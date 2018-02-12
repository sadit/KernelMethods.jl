using SimilaritySearch
export fftraversal

"""
Selects a number of farthest points in `X`, using a farthest first traversal

- The callback function is called on each selected far point as `callback(centerID, dmax)` where `dmax` is the distance to the nearest previous reported center (the first is reported with typemax)
- The selected objects are far under the `dist` distance function with signature (T, T) -> Float64
- The number of points is determined by the stop criterion function with signature (Float64[], T[]) -> Bool
    - The first argument corresponds to the list of known distances (far objects)
    - The second argument corresponds to the database

Check `criterions.jl` for basic implementations of stop criterions
"""
function fftraversal(callback::Function, X::AbstractVector{T}, dist, stop) where {T}
    N = length(X)
    dmaxlist = Float64[]
    dset = [typemax(Float64) for i in 1:N]
    imax::Int = rand(1:N)
    dmax::Float64 = typemax(Float64)
    if N == 0
        return
    end

    k::Int = 0
    
    @inbounds while k <= N
        k += 1
        pivot = X[imax]
        push!(dmaxlist, dmax)
        callback(imax, dmax)
        info("computing fartest point $k, dmax: $dmax, imax: $imax")
        dmax = 0.0
        imax = 0
        for i in 1:N
            d = dist(X[i], pivot)
            if d < dset[i]
                dset[i] = d
            end

            if dset[i] > dmax
                dmax = dset[i]
                imax = i
            end
        end

        if dmax == 0 || stop(dmaxlist, X)
            break
        end
    end
end
