module Kernels

export linear, maxk, cauchy, gaussian, sigmoid
using SimilaritySearch: L2SquaredDistance
using TextModel
using JSON

function linear(xo,xm;sigma=1,distance=L2SquaredDistance())::Float64
    d=distance
    return d(xo,xm)
end

function maxk(xo,xm;sigma=1,distance=L2SquaredDistance())::Float64
    d=distance
    sim = d(xo,xm) > sigma ? 1.0 : 0.0 
    return sim
end

function gaussian(xo,xm; sigma=1,distance=L2SquaredDistance())::Float64
    sim = exp(-distance(xo,xm)/(2*sigma))
    sim = isnan(sim) ? 0 : sim
    sim = isinf(sim) ? -1 : sim
    return sim
end

function sigmoid(xo,xm; sigma=1, distance=L2SquaredDistance())::Float64
    sim=2*sqrt(sigma)/(1+exp(-distance(xo,xm)))
    sim = isnan(sim) ? 0 : sim
    sim = isinf(sim) ? -1: sim
    return sim
end

function cauchy(xo,xm; sigma=1,distance=L2SquaredDistance())::Float64
    x = distance(xo,xm)/(sigma*sigma)
    sim = 1 / (1 + x)
    sim = isnan(sim) ? 0 : sim
    sim = isinf(sim) ? -1: sim
    return sim
end

end


