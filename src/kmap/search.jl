# Copyright 2018 Eric S. Tellez <eric.tellez@infotec.mx>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
export search_model

using KernelMethods
import KernelMethods.Scores: accuracy, recall, precision, f1, precision_recall
import KernelMethods.CrossValidation: montecarlo, kfolds
import KernelMethods.Supervised: NearNeighborClassifier, NaiveBayesClassifier, optimize!, predict, predict_proba, transform, inverse_transform
import SimilaritySearch: L2Distance, CosineDistance
# using KernelMethods.KMap: fftraversal, sqrt_criterion, change_criterion, log_criterion, kmap
using KernelMethods.Kernels: gaussian_kernel, cauchy_kernel, sigmoid_kernel, tanh_kernel, linear_kernel

"""
Searches for a competitive configuration in a parameter space using random search
"""
function search_model(X, y;
                folds=3,
                keep=3,
                score=recall,
                size=16,
                #distances=[L2Distance, CosineDistance],
                #kdistances=[L2Distance, CosineDistance],
                distances=[L2Distance],
                kdistances=[L2Distance, CosineDistance],
                sampling=vcat(
                    [(fftraversal, x) for x in (sqrt_criterion, log_criterion, change_criterion)],
                    [(dnet, x) for x in (3, 10, 30, 100)]
                ),
                kernels=[linear_kernel, gaussian_kernel, sigmoid_kernel, cauchy_kernel, tanh_kernel],
                reftypes=[:centroids, :centers],
            )

    bestlist = []
    tabu = Set()

    for i in 1:size
        m = (rand(distances), rand(kdistances), rand(kernels), rand(sampling), rand(reftypes))
        if m in tabu
            continue
        end
        push!(tabu, m)

        _dist, _kdist, kernel, net, reftype = m
        dist = _dist()
        kdist = _kdist()
        refs = Vector{typeof(X[1])}()
        dmax = 0.0
        if net[1] == fftraversal
            criterion = net[2]
            function pushcenter1(c, _dmax)
                push!(refs, X[c])
                dmax = _dmax
            end
            fftraversal(pushcenter1, X, dist, criterion())
        elseif net[1] == dnet
            k = net[2]
            function pushcenter2(c, dmaxlist)
                push!(refs, X[c])
                dmax += dmaxlist[end][end]    
            end
            dnet(pushcenter2, X, dist, k)
            dmax /= length(refs)
        end

        if kernel in (cauchy_kernel, gaussian_kernel, sigmoid_kernel, tanh_kernel)
            kernel = kernel(dist, dmax/2)
        elseif kernel == linear_kernel
            kernel = kernel(dist)
        end  # else: use kernel as a function

        if reftype == :centroids
            a = [centroid(X[plist]) for plist in invindex(X, dist, refs)]
            M = kmap(X, kernel, a)
        else
            M = kmap(X, kernel, refs)
        end

        if _kdist == CosineDistance
            nnc = NearNeighborClassifier(DenseCosine.(M), y, CosineDistance())
        else
            nnc = NearNeighborClassifier(M, y, kdist)
        end
        

        best = optimize!(nnc, score, folds=folds)[1]
        push!(bestlist, (best[1], nnc))
        sort!(bestlist, by=x->-x[1])
        if length(bestlist) > keep
            bestlist = bestlist[1:keep]
        end
    end

    bestlist
end
