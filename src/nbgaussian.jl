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

export GaussianKernel

using Distributions

struct GaussianKernel <: NaiveBayesKernel
    mean_given_y::Vector{Float64}
    var_given_y::Vector{Float64}
end

function GaussianKernel(X::AbstractVector{ItemType}, y::AbstractVector{Int}, nlabels::Int) where ItemType
    dim = length(X[1])
    C = zeros(Float64, nlabels)
    CC = zeros(Float64, nlabels)
    α = 1 / length(X)

    # we estimate our μ and ρ^2 using sample estimators, because it is fast and we expect
    # to have a bunch of examples (hundreds-thousands-millions)
    @inbounds for i in 1:length(X)
        label = y[i]
        for x in X[i]
            C[label] += x * α  # performing here the product α we can prevent some weird floating point errors
            CC[label] += x * x * α
        end
    end

    CC[:] -= C[:]  # WARNING floating point computation can produce very small values when C and CC are similar
    # @assert sum(CC) > 0

    GaussianKernel(C, CC)
end

function kernel_prob(nbc::NaiveBayesClassifier, kernel::GaussianKernel, x::AbstractVector{Float64})::Vector{Float64}
    n = length(nbc.le.labels)
    scores = zeros(Float64, n)
    @inbounds for i in 1:n
        pxy = 1.0
        py = nbc.probs[i]
        var2 = 2 * kernel.var_given_y[i]
        a = 1 / sqrt(pi * var2)

        for j in 1:length(x)
            pxy *= a * exp(-(x[j] - kernel.mean_given_y[i])^2 / var2)
        end
        scores[i] = py * pxy
    end

    scores
end
