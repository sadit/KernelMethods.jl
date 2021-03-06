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

module KernelMethods
import StatsBase: fit, predict
using SimilaritySearch
export fit, predict
include("scores.jl")
include("cv.jl")
include("kernels.jl")
include("supervised.jl")
include("kmap/kmap.jl")
# include("nets.jl")
end
