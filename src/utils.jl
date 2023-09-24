using Images

"""
    pre_process(x; noise=0.03f0)

Helper function to add tiny noise to inputs.
"""
function pre_process(x; noise::Float32=0.03f0)
    ϵ = Float32.(randn(size(x)) * noise)
    x += ϵ
    return x
end

"A simple functor to convert a vector to a convolutional layer."
struct ToConv
    n_in::Int
end

"""
    (f::ToConv)(x)

Method to convert a vector to a convolutional layer.
"""
function (f::ToConv)(x)
    return reshape(x, (f.n_in, f.n_in, 1, :))
end

"""
    convert2mnist(x)

Converts a vector to a 28x28 grey image.
"""
function convert2mnist(x)
    x = (x .+ 1) ./ 2 |>
        x -> reshape(x, 28, 28) |>
             permutedims |>
             x -> Gray.(x)
    return x
end

"""
    cos_dist(x,y)

Computes the cosine distance between two vectors.
"""
function cos_dist(x, y)
    x = (x .- minimum(x)) ./ (maximum(x) / -minimum(x)) |> x -> reshape(x, 28, 28)
    y = (y .- minimum(y)) ./ (maximum(y) / -minimum(y)) |> x -> reshape(x, 28, 28)
    cos_sim = (x'y/(norm(x)*norm(y)))[1]
    return 1 - cos_sim
end

"""
    ssim_dist(x,y)

Computes 1-SSIM between two images.
"""
function ssim_dist(x, y)
    x = convert2mnist(x)
    y = convert2mnist(y)
    return (1 - assess_ssim(x, y))/2
end