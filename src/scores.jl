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

module Scores

export accuracy, precision_recall, precision, recall, f1

"""
It computes the recall between the gold dataset and the list of predictions `predict`

It applies the desired weighting scheme for binary and multiclass problems
- `:macro` performs a uniform weigth to each class
- `:weigthed` the weight of each class is proportional to its population in gold
- `:micro` returns the global recall, without distinguishing among classes
"""
function recall(gold, predict; weight=:macro)::Float64
    precision, recall, precision_recall_per_class = precision_recall(gold, predict)
    if weight == :macro
        mean(x -> x.second[2], precision_recall_per_class)
    elseif weight == :weighted
        mean(x -> x.second[2] * x.second[3] / length(gold), precision_recall_per_class)
    elseif :micro
        recall
    else
        throw(Exception("Unknown weighting method $weight"))
    end
end

"""
It computes the precision between the gold dataset and the list of predictions `predict`

It applies the desired weighting scheme for binary and multiclass problems
- `:macro` performs a uniform weigth to each class
- `:weigthed` the weight of each class is proportional to its population in gold
- `:micro` returns the global precision, without distinguishing among classes
"""
function precision(gold, predict; weight=:macro)::Float64
    precision, recall, precision_recall_per_class = precision_recall(gold, predict)
    if weight == :macro
        mean(x -> x.second[1], precision_recall_per_class)
    elseif weight == :weighted
        mean(x -> x.second[1] * x.second[3] / length(gold), precision_recall_per_class)
    elseif weight == :micro
        precision
    else
        throw(Exception("Unknown weighting method $weight"))
    end
end

"""
It computes the F1 score between the gold dataset and the list of predictions `predict`

It applies the desired weighting scheme for binary and multiclass problems
- `:macro` performs a uniform weigth to each class
- `:weigthed` the weight of each class is proportional to its population in gold
- `:micro` returns the global F1, without distinguishing among classes
"""
function f1(gold, predict; weight=:macro)::Float64
    precision, recall, precision_recall_per_class = precision_recall(gold, predict)
    if weight == :macro
        mean(x -> 2 * x.second[1] * x.second[2] / (x.second[1] + x.second[2]), precision_recall_per_class)
    elseif weight == :weighted
        mean(x -> 2 * x.second[1] * x.second[2] / (x.second[1] + x.second[2]) * x.second[3]/length(gold), precision_recall_per_class)
    elseif weight == :micro
        2 * (precision * recall) / (precision + recall)
    else
        throw(Exception("Unknown weighting method $weight"))
    end
end

"""
It computes the global and per-class precision and recall values between the gold standard
and the predicted set
"""
function precision_recall(gold, predicted)
    labels = unique(gold)
    M = Dict{typeof(labels[1]), Tuple}()
    tp_ = 0
    tn_ = 0
    fn_ = 0
    fp_ = 0

    for label in labels
        lgold = label .== gold
        lpred = label .== predicted

        tp = 0
        tn = 0
        fn = 0
        fp = 0
        for i in 1:length(lgold)
            if lgold[i] == lpred[i]
                if lgold[i]
                    tp += 1
                else
                    tn += 1
                end
            else
                if lgold[i]
                    fn += 1
                else
                    fp += 1
                end
            end
        end

        tp_ += tp
        tn_ += tp
        fn_ += tp
        fp_ += tp
        M[label] = (tp / (tp + fp), tp / (tp + fn), sum(lgold) |> Int)  # precision, recall, class-population
    end

    tp_ / (tp_ + fp_), tp_ / (tp_ + fn_), M
end

"""
It computes the accuracy score between the gold and the predicted sets
"""
function accuracy(gold, predicted)
    #  mean(gold .== predicted)
    c = 0
    for i in 1:length(gold)
        c += (gold[i] == predicted[i])
    end

    c / length(gold)
end

end
