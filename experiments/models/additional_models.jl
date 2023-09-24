"""
    LeNetBuilder

MLJFlux builder for a LeNet-like convolutional neural network.
"""
mutable struct LeNetBuilder
	filter_size::Int
	channels1::Int
	channels2::Int
end

"""
    MLJFlux.build(b::LeNetBuilder, rng, n_in, n_out)

Overloads the MLJFlux build function for a LeNet-like convolutional neural network.
"""
function MLJFlux.build(b::LeNetBuilder, rng, n_in, n_out)

    # Setup:
    _n_in = Int(sqrt(n_in))
	k, c1, c2 = b.filter_size, b.channels1, b.channels2
	mod(k, 2) == 1 || error("`filter_size` must be odd. ")
    p = div(k - 1, 2) # padding to preserve image size on convolution:

    # Model:
	front = Flux.Chain(
        Conv((k, k), 1 => c1, pad=(p, p), relu),
        MaxPool((2, 2)),
        Conv((k, k), c1 => c2, pad=(p, p), relu),
        MaxPool((2, 2)),
        Flux.flatten
    )
	d = Flux.outputsize(front, (_n_in, _n_in, 1, 1)) |> first
    back = Flux.Chain(
        Dense(d, 120, relu),
        Dense(120, 84, relu),
        Dense(84, n_out),
    )
    chain = Flux.Chain(ECCCo.ToConv(_n_in), front, back)

	return chain
end

"""
    lenet5(builder=LeNetBuilder(5, 6, 16); kwargs...)

Builds a LeNet-like convolutional neural network.
"""
lenet5(builder=LeNetBuilder(5, 6, 16); kwargs...) = NeuralNetworkClassifier(builder=builder; acceleration=CUDALibs(), kwargs...)

"""
    ResNetBuilder

MLJFlux builder for a ResNet.
"""
mutable struct ResNetBuilder end

"""
    MLJFlux.build(b::LeNetBuilder, rng, n_in, n_out)

Overloads the MLJFlux build function for a LeNet-like convolutional neural network.
"""
function MLJFlux.build(b::ResNetBuilder, rng, n_in, n_out)
    _n_in = Int(sqrt(n_in))
    front = Metalhead.ResNet(18; inchannels=1)
    d = Flux.outputsize(front, (_n_in, _n_in, 1, 1)) |> first
    back = Flux.Chain(
        Dense(d, 120, relu),
        Dense(120, 84, relu),
        Dense(84, n_out),
    )
    chain = Flux.Chain(ECCCo.ToConv(_n_in), front, back)
    return chain
end

"""
    lenet5(builder=LeNetBuilder(5, 6, 16); kwargs...)

Builds a LeNet-like convolutional neural network.
"""
resnet18(builder=ResNetBuilder(); kwargs...) = NeuralNetworkClassifier(builder=builder; acceleration=CUDALibs(), kwargs...)
