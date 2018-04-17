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
export KernelClassifier, KConfigurationSpace, predict, predict_one

using KernelMethods
import KernelMethods.Scores: accuracy, recall, precision, f1, precision_recall
import KernelMethods.CrossValidation: montecarlo, kfolds
import KernelMethods.Supervised: NearNeighborClassifier, NaiveBayesClassifier, optimize!, predict, predict_one, transform, inverse_transform
import SimilaritySearch: L2Distance, CosineDistance
# import KernelMethods.KMap: sqrt_criterion, log_criterion, change_criterion
# using KernelMethods.KMap: fftraversal, sqrt_criterion, change_criterion, log_criterion, kmap
using KernelMethods.Kernels: gaussian_kernel, cauchy_kernel, sigmoid_kernel, tanh_kernel, linear_kernel

struct KConfigurationSpace
    distances
    kdistances
    sampling
    kernels
    reftypes
    classifiers
end

function KConfigurationSpace(;
    distances=[L2Distance],
    #kdistances=[L2Distance, CosineDistance],
    kdistances=[L2Distance, CosineDistance],
    sampling=vcat(
        [(fftraversal, x) for x in (sqrt_criterion, log_criterion, change_criterion)],
        [(dnet, x) for x in (30, 100, 300)]
    ),
    kernels=[linear_kernel, gaussian_kernel, sigmoid_kernel, cauchy_kernel, tanh_kernel],
    reftypes=[:centroids, :centers],
    classifiers=[NearNeighborClassifier, NaiveBayesClassifier]
)
    KConfigurationSpace(distances, kdistances, sampling, kernels, reftypes, classifiers)
end

struct KConfiguration
    dist
    kdist
    kernel
    net
    reftype
    classifier
end

function randconf(space::KConfigurationSpace)
    KConfiguration(
        rand(space.distances),
        rand(space.kdistances),
        rand(space.kernels),
        rand(space.sampling),
        rand(space.reftypes),
        rand(space.classifiers)
    )
end

function randconf(space::KConfigurationSpace, num::Integer)
    [randconf(space) for i in 1:num]
end

struct KernelClassifierType{ItemType}
    kernel
    refs::Vector{ItemType}
    classifier
    conf::KConfiguration
end

"""
Searches for a competitive configuration in a parameter space using random search
"""
function KernelClassifier(X, y;
                folds=3,
                score=recall,
                size=32,
                ensemble_size=3,
                space=KConfigurationSpace()
            )

    bestlist = []
    tabu = Set()
    dtype = typeof(X[1])
    
    for conf in randconf(space, size)
        if conf in tabu
            continue
        end

        info("probing configuration $(conf), data-type $(typeof(X))")
        push!(tabu, conf)
        dist = conf.dist()
        kdist = conf.kdist()
        refs = Vector{dtype}()
        dmax = 0.0

        if conf.kernel in (cauchy_kernel, gaussian_kernel, sigmoid_kernel, tanh_kernel)
            kernel = conf.kernel(dist, dmax/2)
        elseif conf.kernel == linear_kernel
            kernel = conf.kernel(dist)
        else
            kernel = conf.kernel
        end
        
        if conf.net[1] == fftraversal
            criterion = conf.net[2]
            function pushcenter1(c, _dmax)
                push!(refs, X[c])
                dmax = _dmax
            end
            fftraversal(pushcenter1, X, dist, criterion())

            info("computing kmap, conf: $conf")
            if conf.reftype == :centroids
                a = [centroid(X[plist]) for plist in invindex(X, dist, refs)]
                M = kmap(X, kernel, a)
            else
                M = kmap(X, kernel, refs)
            end
        elseif conf.net[1] == dnet
            k = conf.net[2]
            function pushcenter2(c, dmaxlist)
                if conf.reftype == :centroids
                    a = vcat([X[c]], X[[p.objID for p in dmaxlist]]) |> centroid
                    push!(refs, a)
                else
                    push!(refs, X[c])
                end
                
                dmax += last(dmaxlist).dist
            end

            dnet(pushcenter2, X, dist, k)
            info("computing kmap, conf: $conf")
            M = kmap(X, kernel, refs)
            dmax /= length(refs)
        end

        info("creating and optimizing classifier, conf: $conf")
        if conf.kdist == CosineDistance
            classifier = NearNeighborClassifier(DenseCosine.(M), y, kdist)
            best = optimize!(classifier, score, folds=folds)[1]
        else
            if conf.classifier == NearNeighborClassifier
                classifier = NearNeighborClassifier(M, y, kdist)
                best = optimize!(classifier, score, folds=folds)[1]
            else
                classifier = NaiveBayesClassifier(M, y)
                best = optimize!(classifier, M, y, score, folds=folds)[1]
            end
        end

        model = KernelClassifierType(kernel, refs, classifier, conf)
        push!(bestlist, (best[1], model))
        info("score: $(best[1]), conf: $conf")
        sort!(bestlist, by=x->-x[1])

        if length(bestlist) > ensemble_size
            bestlist = bestlist[1:ensemble_size]
        end
    end

    info("final scores: ", [b[1] for b in bestlist])
    # @show [b[1] for b in bestlist]
    [b[2] for b in bestlist]
end

function predict(kmodel::AbstractVector{KernelClassifierType{ItemType}}, vector) where {ItemType}
    [predict_one(kmodel, x) for x in vector]
end

function predict_one(kmodel::AbstractVector{KernelClassifierType{ItemType}}, x) where {ItemType}
    C = Dict()
    for m in kmodel
        label = predict_one(m, x)
        C[label] = get(C, label, 0) + 1
    end
    
    counter = [(c, label) for (label, c) in C]
    sort!(counter, by=x->-x[1])

    # @show counter
    counter[1][end]
end

function predict_one(kmodel::KernelClassifierType{ItemType}, x) where {ItemType}
    kernel = kmodel.kernel
    refs = kmodel.refs

    vec = Vector{Float64}(length(refs))
    for i in 1:length(refs)
        vec[i] = kernel(x, refs[i])
    end

    if kmodel.conf.kdist == CosineDistance
        predict_one(kmodel.classifier, DenseCosine(vec))
    else
        predict_one(kmodel.classifier, vec)
    end
end
