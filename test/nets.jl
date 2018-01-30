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
using KernelMethods.KMap:
    farthest_points, sqrt_criterion, change_criterion, log_criterion, kmap, spartition, apartition
using KernelMethods.Kernels:
    gaussian_kernel

import KernelMethods.Scores: accuracy, recall, precision, f1, precision_recall
import KernelMethods.CrossValidation: montecarlo, kfolds
import KernelMethods.Supervised: NearNeighborClassifier, NaiveBayesClassifier, optimize!, predict, predict_proba, transform, inverse_transform
import SimilaritySearch: L2Distance
# import KernelMethods.Nets: KlusterClassifier
using Base.Test

function loadiris()
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    filename = basename(url)
    if !isfile(filename)
        download(url, filename)
    end
    data = readcsv(filename)
    X = data[:, 1:4]
    X = [Float64.(X[i, :]) for i in 1:size(X, 1)]
    y = String.(data[:, 5])
    X, y
end

@testset "Nets" begin
    X, y = loadiris()
    dist = L2Distance()
    #criterion = change_criterion(0.1)
    criterion = sqrt_criterion()
    farlist, dmaxlist, faridxlist = farthest_points(X, dist, criterion)
    g = gaussian_kernel(dist, dmaxlist[end]/2)
    m = kmap(X, g, farlist)
    @show m
    # @show [@view m[:, i] for i in 1:size(m, 2)]
    # @show spartition(X, dist, farlist)
    w = zip(spartition(X, dist, farlist, 3), y) |> collect
    
    # for (a, b) in w
    #     @show b, a
    # end

    # yref = y[faridxlist]
    # for (i, plist) in enumerate(apartition(X, dist, farlist, k=1))
    #     @show i, plist
    # end
end
# @testset "KlusterClassifier" begin
#     X, y = loadiris()
#     kc = KlusterClassifier(X,y)
#     @show kc
#     @test kc[1][2]>0.9
# end

