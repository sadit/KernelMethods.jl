export size_criterion, sqrt_criterion, change_criterion, log_criterion

function size_criterion(maxsize)
   function metastop()
      return (dmaxlist, database) -> length(dmaxlist) >= maxsize
   end

   return metastop
end

function sqrt_criterion()
   (dmaxlist, database) -> length(dmaxlist) >= Int(length(database) |> sqrt |> round)
end

function log_criterion()
   (dmaxlist, database) -> length(dmaxlist) >= Int(length(database) |> log2 |> round)
end

function change_criterion(tol=0.001, window=3)
    # function metastop()
      mlist = Float64[]
      count = 0.0
      function stop(dmaxlist, database)
         count += dmaxlist[end]

         if length(dmaxlist) % window != 1
            return false
         end
         push!(mlist, count)
         count = 0.0
         if length(dmaxlist) < 2
            return false
         end

         s = abs(mlist[end] - mlist[end-1])
         return s <= tol
      end
      return stop
   # end

   # return metastop()
end
