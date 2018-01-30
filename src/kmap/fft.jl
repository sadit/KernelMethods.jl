using SimilaritySearch
export farthest_points

"""
Selects a number of farthest points in `X`.

- The selected objects are far under the `dist` distance function with signature (T, T) -> Float64
- The number of points is determined by the stop criterion function with signature (Float64[], T[]) -> Bool
  * The first argument corresponds to the list of known distances (far objects)
  * The second argument corresponds to the database
- It returns the tuple `(farlist, dmaxlist, indexes)`
  * `farlist` contains the list of distance points (in the selection order)
  * `dmaxlist` contains the list of distances of farlist
  * `indexes` constains the indexes in `X` of points in `farlist`

Check `criterions.jl` for basic implementations of stop criterions
"""
function farthest_points(X::Vector{T}, dist, stop)::Tuple{Vector{T}, Vector{Float64}, Vector{Int}} where {T}
   N = length(X)
   farlist = T[]
   dmaxlist = Float64[]
   indexes = Int[]

   if N == 0
      return farlist, dmaxlist, indexes
   end

   dset = [typemax(Float64) for i in 1:N]
   imax::Int = rand(1:N)
   dmax::Float64 = typemax(Float64)

   # local k::T
   k = 0
   @inbounds while k <= N
      k += 1
      pivot = X[imax]
      push!(farlist, pivot)
      push!(dmaxlist, dmax)
      push!(indexes, imax)
      info("computing fartest point $k, dmax: $dmax, pivot: ", pivot)
      dmax = 0.0
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

   # info(farlist)
   return farlist, dmaxlist, indexes
end
