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