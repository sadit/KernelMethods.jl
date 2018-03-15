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

using KernelMethods
import KernelMethods.Scores: pearson, spearman, negsqerror
import KernelMethods.CrossValidation: montecarlo, kfolds
import KernelMethods.Supervised: NearNeighborRegression, optimize!, predict, predict_proba, transform, inverse_transform
import SimilaritySearch: L2Distance
# import KernelMethods.Nets: KlusterClassifier
using Base.Test

include("loaddata.jl")

@testset "KNN Regression" begin
    X, y = loadlinearreg()
    nnc = NearNeighborRegression(X, y, L2Distance())
    @show optimize!(nnc, pearson, runs=5, trainratio=0.5, testratio=0.5)
    #@test optimize!(nnc, accuracy, runs=5, trainratio=0.3, testratio=0.3)[1][1] > 0.9
    #@test sum([maximum(x) for x in predict_proba(nnc, X, smoothing=0)])/ length(X) > 0.9 ## close to have all ones, just in case
    #@test sum([maximum(x) for x in predict_proba(nnc, X, smoothing=0.01)])/ length(X) > 0.9 ## close to have all ones, just in case
end

