# Copyright 2017,2018 Eric S. Tellez
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

export NearNeighborRegression, optimize!, predict, predict_proba
import KernelMethods.CrossValidation: montecarlo, kfolds

using SimilaritySearch:
    Sequential, KnnResult, clear!

mutable struct NearNeighborRegression{IndexType,DataType}
    X::IndexType
    y::Vector{DataType}
    k::Int
    summarize::Function
end


function NearNeighborRegression(X::AbstractVector{ItemType}, y::AbstractVector{DataType}, dist; summarize=mean, k::Int=1, indexclass=Sequential) where {ItemType, DataType}
    index = indexclass(X, dist)
    NearNeighborRegression{indexclass, DataType}(index, y, k, summarize)
end

function predict(nnc::NearNeighborRegression{IndexType,DataType}, vector) where {IndexType,DataType}
    [predict_one(nnc, item) for item in vector]
end

function predict_one(nnc::NearNeighborRegression{IndexType,DataType}, item) where {IndexType,DataType}
    res = KnnResult(nnc.k)
    search(nnc.X, item, res)
    DataType[nnc.y[p.objID] for p in res] |> nnc.summarize
end

function _train_create_table_reg(train_X, train_y, test_X, dist, k::Int)
    index = Sequential(train_X, dist)
    res = KnnResult(k)
    function f(x)
        clear!(res)  # this is thread unsafe
        search(index, x, res)
        [train_y[p.objID] for p in res]
    end

    f.(test_X)
end

function _train_predict(nnc::NearNeighborRegression{IndexType,DataType}, table, test_X, k) where {IndexType,DataType}
    A = Vector{DataType}(length(test_X))
    for i in 1:length(test_X)
        row = table[i]
        A[i] = nnc.summarize(row[1:k])
    end

    A
end

function gmean(X)
    prod(X)^(1/length(X))
end

function hmean(X)
    d = 0.0
    for x in X
        d += 1.0 / x
    end
    length(X) / d
end

function optimize!(nnc::NearNeighborRegression, scorefun::Function; summarize_list=[mean, median, gmean, hmean], runs=3, trainratio=0.5, testratio=0.5, folds=0, shufflefolds=true)
    mem = Dict{Tuple,Float64}()
    function f(train_X, train_y, test_X, test_y)
        tmp = NearNeighborRegression(train_X, train_y, nnc.X.dist)
        kmax = sqrt(length(nnc.X.db)) |> round |> Int
        table = _train_create_table_reg(train_X, train_y, test_X, nnc.X.dist, kmax)
        k = 2
        while k <= kmax
            tmp.k = k - 1
            for summarize in summarize_list
                tmp.summarize = summarize
                pred_y = _train_predict(tmp, table, test_X, tmp.k)
                score = scorefun(test_y, pred_y)
                key = (k - 1, summarize)
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
    nnc.summarize = best[2][2]

    bestlist
end