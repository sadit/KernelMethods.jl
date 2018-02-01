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

@testset "encode by farthest points" begin
    using KernelMethods.Kernels: gaussian_kernel
    using KernelMethods.KMap: farthest_points, sqrt_criterion, change_criterion, log_criterion, kmap
    using KernelMethods.Scores: accuracy
    using KernelMethods.Supervised: NearNeighborClassifier, optimize!
    using SimilaritySearch: L2Distance

    X, y = loadiris()
    dist = L2Distance()
    # criterion = change_criterion(0.01)
    criterion = sqrt_criterion()
    farlist, dmaxlist, faridxlist = farthest_points(X, dist, criterion)
    g = gaussian_kernel(dist, dmaxlist[end]/2)
    M = kmap(X, g, farlist)
    nnc = NearNeighborClassifier(M, y, L2Distance())
    
    @test optimize!(nnc, accuracy, folds=2)[1][1] > 0.9
    @test optimize!(nnc, accuracy, folds=3)[1][1] > 0.9
    @test optimize!(nnc, accuracy, folds=5)[1][1] > 0.95
    @test optimize!(nnc, accuracy, folds=10)[1][1] > 0.95
    @show optimize!(nnc, accuracy, folds=5)
end

@testset "Clustering and centroid computation (with cosine)" begin
    using KernelMethods.KMap: farthest_points, sqrt_criterion, invindex, centroid
    using SimilaritySearch: L2Distance, L1Distance
    using SimilaritySearch: CosineDistance, DenseCosine
    X, y = loadiris()
    dist = L2Distance()
    farlist, dmaxlist, faridxlist = farthest_points(X, dist, sqrt_criterion())
    a = [centroid(X[plist]) for plist in invindex(X, dist, farlist)]
    g = gaussian_kernel(dist, dmaxlist[end]/4)
    M = kmap(X, g, a)
    nnc = NearNeighborClassifier([DenseCosine(w) for w in M], y, CosineDistance())
    
    @test optimize!(nnc, accuracy, folds=2)[1][1] > 0.9
    @test optimize!(nnc, accuracy, folds=3)[1][1] > 0.9
    @test optimize!(nnc, accuracy, folds=5)[1][1] > 0.95
    @test optimize!(nnc, accuracy, folds=10)[1][1] > 0.95
    @show optimize!(nnc, accuracy, folds=10)
end
# @testset "KlusterClassifier" begin
#     X, y = loadiris()
#     kc = KlusterClassifier(X,y)
#     @show kc
#     @test kc[1][2]>0.9
# end