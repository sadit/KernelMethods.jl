# Copyright 2017 Eric S. Tellez
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

export NearNeighborClassifier, predict, optimize!, predict_proba
import KernelMethods.CrossValidation: montecarlo, kfolds

mutable struct NearNeighborClassifier{IndexType,LabelType}
    X::IndexType
    y::Vector{LabelType}
    unique_y::Vector{LabelType}
    k::Int
    weight
end

const SMOOTHING_FACTOR = 1e-6

function NearNeighborClassifier(X::Vector{ItemType}, y::Vector{LabelType}, dist, k::Int=1, weight=:uniform, indexclass=Sequential) where {ItemType, LabelType}
    index = indexclass(X, dist)
    u = unique(y)
    sort!(u)
    NearNeighborClassifier(index, y, u, k, weight)
end

function optimize!(nnc::NearNeighborClassifier, score::Function; runs=3, trainratio=0.5, testratio=0.5, folds=0, shufflefolds=true)
    bestlist = Tuple[]
    function f(train_X, train_y, test_X, test_y)
        tmp = NearNeighborClassifier(train_X, train_y, nnc.X.dist)
        kmax = sqrt(length(nnc.X.db)) |> round |> Int
        k = 2
        while k <= kmax
            for weight in (:uniform, :distance)
                tmp.weight = weight
                tmp.k = k - 1
                pred_y = predict(tmp, test_X)
                s = score(test_y, pred_y)
                push!(bestlist, (s, k - 1, weight))
            end
            k += k
        end
        bestlist[end][1]
    end
    if folds > 1
        kfolds(f, nnc.X.db, nnc.y, folds=folds, shuffle=shufflefolds)
    else
        montecarlo(f, nnc.X.db, nnc.y, runs=runs, trainratio=trainratio, testratio=testratio)
    end
    sort!(bestlist, by=x -> (-x[1], x[2]))
    nnc.k = bestlist[1][2]
    nnc.weight = bestlist[1][3]
    bestlist
end

function predict(nnc::NearNeighborClassifier{IndexType,LabelType}, vector) where {IndexType,LabelType}
    y = Vector{LabelType}(length(vector))
    for i in 1:length(vector)
        m = predict_one(nnc, vector[i])
        y[i] = m[1].first
    end
    y
end

function predict_proba(nnc::NearNeighborClassifier{IndexType,LabelType}, vector; smoothing=0.05) where {IndexType,LabelType}
    y = Vector{Dict{LabelType,Float64}}(length(vector))
    for i in 1:length(vector)
        y[i] = predict_one_proba(nnc, vector[i], smoothing=smoothing)
    end

    y
end

function predict_one(nnc::NearNeighborClassifier{IndexType,LabelType}, item) where {IndexType,LabelType}
    res = KnnResult(nnc.k)
    search(nnc.X, item, res)
    counter = Dict{typeof(nnc.y[1]), Float64}()
    for k in nnc.unique_y
        counter[k] = 0.0
    end
    if nnc.weight == :uniform
        for p in res
            l = nnc.y[p.objID]
            counter[l] += 1.0
        end
    elseif nnc.weight == :distance
        for p in res
            l = nnc.y[p.objID]
            counter[l] += 1.0 / (p.dist + SMOOTHING_FACTOR)
        end
    else
        throw(ArgumentError("Unknown weighting scheme $(nnc.weight)"))
    end

    m = collect(counter)
    sort!(m, by=x->-x.second)
    m
end

function predict_one_proba(nnc::NearNeighborClassifier{IndexType,LabelType}, item; smoothing=0.1) where {IndexType,LabelType}
    m = predict_one(nnc, item)
    t = 0.0
    for x in m
        t += x.second
    end

    s = t * smoothing
    ss = s * length(nnc.unique_y)

    Dict(x.first => (x.second + s) / (t + ss) for x in m)
end
