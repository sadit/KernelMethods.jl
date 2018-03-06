module Kernels

export linear, maxk, cauchy, gaussian, sigmoid, gaussian_kernel, sigmoid_kernel, cauchy_kernel
using SimilaritySearch

linear(xo, xm; sigma=1, distance=L2SquaredDistance()) = distance(xo,xm)

maxk(xo, xm; sigma=1, distance=L2SquaredDistance()) = distance(xo,xm) > sigma ? 1.0 : 0.0

function gaussian(xo, xm; sigma=1, distance=L2SquaredDistance())
    d=distance(xo, xm)
    (d==0 || sigma==0) && return 1.0
    exp(-d/(2*sigma))
end

function sigmoid(xo, xm; sigma=1, distance=L2SquaredDistance())
    2*sqrt(sigma)/(1+exp(-distance(xo,xm)))
end

function cauchy(xo, xm; sigma=1, distance=L2SquaredDistance())
    d=distance(xo,xm)
    (d==0 || sigma==0) && return 1.0
    1/(1+d/(sigma*sigma))
end

"""
Creates a Gaussian kernel with the given `sigma`
"""
function gaussian_kernel(dist, sigma=1.0)
    sigma2 = sigma * 2
    function fun(obj, ref)::Float64
        d = dist(obj, ref)
        (d == 0 || sigma == 0) && return 1.0
        exp(-d / sigma2)
    end

    fun
end

function sigmoid_kernel(dist, sigma=1.0)
    sqrtsigma = sqrt(sigma)
    function fun(obj, ref)::Float64
        x = dist(obj, ref)
        2 * sqrtsigma / (1 + exp(-x))
    end

    fun
end

function cauchy_kernel(dist, sigma=1.0)
    sqsigma = sigma^2
    function fun(obj, ref)::Float64
        x = dist(obj, ref)
        (x == 0 || sqsigma == 0) && return 1.0
        1 / (1 + x^2 / sqsigma)
    end
end

function tanh_kernel(dist, sigma=1.0)
    function fun(obj, ref)::Float64
        x = dist(obj, ref)
        (exp(x-sigma) - exp(-x+sigma)) / (exp(x-sigma) + exp(-x+sigma))
    end
end

end
