module Kernels
export linear, maxk, cuchy, gaussian, sigmoid

using SimilaritySearch
using TextModel
using JSON

function linear(xo,xm;sigma=1,distance=L2SquaredDistance())
    d=distance
    return d(xo,xm)
end

function maxk(xo,xm;sigma=1,distance=L2SquaredDistance())
    d=distance
    sim = d(xo,xm) > sigma ? 1.0 : 0.0 
    return sim
end

function gaussian(xo,xm; sigma=1,distance=L2SquaredDistance())
    sim=exp(-distance(xo,xm)/(2*sigma))
    sim = isnan(sim) ? 0 : sim
    sim = isinf(sim) ? -1: sim
    return sim
end

function sigmoid(xo,xm; sigma=1, distance=L2SquaredDistance())
    sim=2*sqrt(sigma)/(1+exp(-distance(xo,xm)))
    sim = isnan(sim) ? 0 : sim
    sim = isinf(sim) ? 1: sim
    return sim
end

function cuchy(xo,xm; sigma=1,distance=L2SquaredDistance())
    sim=1/(1+distance(xo,xm)/sigma*sigma)
    sim = isnan(sim) ? 0 : sim
    sim = isinf(sim) ? -1: sim
    return sim
end
end


