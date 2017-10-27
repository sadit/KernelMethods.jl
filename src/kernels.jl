module Kernels

export linear, maxk, cauchy, gaussian, sigmoid

using SimilaritySearch
using TextModel
using JSON

linear(xo,xm;sigma=1,distance=L2SquaredDistance())=distance(xo,xm)

maxk(xo,xm;sigma=1,distance=L2SquaredDistance())=distance(xo,xm) > sigma ? 1.0 : 0.0

function gaussian(xo,xm; sigma=1,distance=L2SquaredDistance())
    d=distance(xo,xm)
    (d==0 || sigma==0) && return 1.0
    exp(-d/(2*sigma))
end

sigmoid(xo,xm; sigma=1, distance=L2SquaredDistance())=2*sqrt(sigma)/(1+exp(-distance(xo,xm)))


function cauchy(xo,xm; sigma=1,distance=L2SquaredDistance())
    d=distance(xo,xm)
    (d==0 || sigma==0) && return 1.0
    1/(1+d/(sigma*sigma))
end
end


