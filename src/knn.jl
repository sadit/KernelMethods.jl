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

export NearNeighborClassifier

mutable struct NearNeighborClassifier{IndexType,LabelType}
    X::IndexType
    y::Vector{Int}
    k::Int
    le::LabelEncoder
    weight
end

const NNC_SMOOTHING_FACTOR = 1e-6

function NearNeighborClassifier(X::Vector{ItemType}, y::Vector{LabelType}, dist, k::Int=1, weight=:uniform, indexclass=Sequential) where {ItemType, LabelType}
    le = LabelEncoder(y)
    y_ = transform.(le, y)
    index = indexclass(X, dist)
    NearNeighborClassifier{indexclass, LabelType}(index, y_, k, le, weight)
end

function predict(nnc::NearNeighborClassifier{IndexType,LabelType}, vector) where {IndexType,LabelType}
    [predict_one(nnc, item) for item in vector]
end

function predict_proba(nnc::NearNeighborClassifier{IndexType,LabelType}, vector; smoothing=0.0) where {IndexType,LabelType}
    [predict_one_proba(nnc, item, smoothing=smoothing) for item in vector]
end

function _predict_one(nnc::NearNeighborClassifier{IndexType,LabelType}, item) where {IndexType,LabelType}
    res = KnnResult(nnc.k)
    search(nnc.X, item, res)
    w = zeros(Float64, length(nnc.le.labels))
    if nnc.weight == :uniform
        for p in res
            l = nnc.y[p.objID]
            w[l] += 1.0
        end
    elseif nnc.weight == :distance
        for p in res
            l = nnc.y[p.objID]
            w[l] += 1.0 / (p.dist + NNC_SMOOTHING_FACTOR)
        end
    else
        throw(ArgumentError("Unknown weighting scheme $(nnc.weight)"))
    end

    w
end

function predict_one(nnc::NearNeighborClassifier{IndexType,LabelType}, item) where {IndexType,LabelType}
    score, i = findmax(_predict_one(nnc, item))
    inverse_transform(nnc.le, i)
end

function predict_one_proba(nnc::NearNeighborClassifier{IndexType,LabelType}, item; smoothing=0.1) where {IndexType,LabelType}
    w = _predict_one(nnc, item)
    t = sum(w)

    s = t * smoothing
    ss = s * length(w)

    for i in 1:length(w)
        w[i] = (w[i] + s) / (t + ss)  # overriding previous w
    end

    w
end

function optimize!(nnc::NearNeighborClassifier, scorefun::Function; runs=3, trainratio=0.5, testratio=0.5, folds=0, shufflefolds=true)
    mem = Dict{Tuple,Float64}()
    function f(train_X, train_y, test_X, test_y)
        tmp = NearNeighborClassifier(train_X, train_y, nnc.X.dist)
        kmax = sqrt(length(nnc.X.db)) |> round |> Int
        k = 2
        while k <= kmax
            for weight in (:uniform, :distance)
                tmp.weight = weight
                tmp.k = k - 1
                pred_y = predict(tmp, test_X)
                score = scorefun(test_y, pred_y)
                key = (k - 1, weight)
                mem[key] = get(mem, key, 0.0) + score
            end
            k += k
        end
        0
    end
    if folds > 1
        kfolds(f, nnc.X.db, nnc.y, folds=folds, shuffle=shufflefolds)
        bestlist = [(score/folds, conf) for (conf, score) in mem]
    else
        montecarlo(f, nnc.X.db, nnc.y, runs=runs, trainratio=trainratio, testratio=testratio)
        bestlist = [(score/runs, conf) for (conf, score) in mem]
    end

    sort!(bestlist, by=x -> (-x[1], x[2][1]))
    best = bestlist[1]
    nnc.k = best[2][1]
    nnc.weight = best[2][2]

    bestlist
end
