using CounterfactualExplanations
using Distributions
using Flux
using JointEnergyModels

"""
    (model::AbstractFittedModel)(x)

When called on data `x`, softmax logits are returned. In the binary case, outputs are one-hot encoded.
"""
(model::AbstractFittedModel)(x) = log.(CounterfactualExplanations.predict_proba(model, nothing, x))

"Base type that stores information relevant to energy-based posterior sampling from `AbstractFittedModel`."
mutable struct EnergySampler
    model::AbstractFittedModel
    data::CounterfactualData
    sampler::JointEnergyModels.ConditionalSampler
    opt::JointEnergyModels.AbstractSamplingRule
    buffer::Union{Nothing,AbstractArray}
    yidx::Union{Nothing,Any}
end

"""
    EnergySampler(
        model::AbstractFittedModel,
        data::CounterfactualData,
        y::Any;
        opt::JointEnergyModels.AbstractSamplingRule=ImproperSGLD(),
        niter::Int=100,
        nsamples::Int=1000
    )

Constructor for `EnergySampler` that takes a `model`, `data` and conditioning value `y` as inputs.
"""
function EnergySampler(
    model::AbstractFittedModel,
    data::CounterfactualData,
    y::Any;
    opt::JointEnergyModels.AbstractSamplingRule=ImproperSGLD(),
    niter::Int=100,
    nsamples::Int=100
)

    @assert y ∈ data.y_levels || y ∈ 1:length(data.y_levels)

    if model.model.model isa JointEnergyClassifier
        sampler = model.model.model.jem.sampler
    else
        K = length(data.y_levels)
        input_size = size(selectdim(data.X, ndims(data.X), 1))
        𝒟x = Uniform(extrema(data.X)...)
        𝒟y = Categorical(ones(K) ./ K)
        sampler = ConditionalSampler(𝒟x, 𝒟y; input_size=input_size)
    end
    yidx = get_target_index(data.y_levels, y)

    # Initiate:
    energy_sampler = EnergySampler(model, data, sampler, opt, nothing, nothing)

    # Generate conditional samples:
    generate_samples!(energy_sampler, nsamples, yidx; niter=niter)

    return energy_sampler
end

"""
    EnergySampler(
        ce::CounterfactualExplanation;
        kwrgs...
    )

Constructor for `EnergySampler` that takes a `CounterfactualExplanation` as input. The underlying model, data and `target` are used for the `EnergySampler`, where `target` is the conditioning value of `y`.
"""
function EnergySampler(
    ce::CounterfactualExplanation;
    kwrgs...
)

    # Setup:
    model = ce.M
    data = ce.data
    y = ce.target

    return EnergySampler(model, data, y; kwrgs...)
end

"""
    generate_samples(e::EnergySampler, n::Int, y::Int; niter::Int=100)

Generates `n` samples from `EnergySampler` for conditioning value `y`.
"""
function generate_samples(e::EnergySampler, n::Int, y::Int; niter::Int=100)

    # Generate samples:
    chain = e.model.fitresult[1]
    rule = e.opt
    xsamples = e.sampler(chain, rule; niter=niter, n_samples=n, y=y)

    return xsamples
end

"""
    generate_samples!(e::EnergySampler, n::Int, y::Int; niter::Int=100)

Generates `n` samples from `EnergySampler` for conditioning value `y`. Assigns samples and conditioning value to `EnergySampler`.
"""
function generate_samples!(e::EnergySampler, n::Int, y::Int; niter::Int=100)
    e.buffer = generate_samples(e, n, y; niter=niter)
    e.yidx = y
end

"""
    Base.rand(sampler::EnergySampler, n::Int=100; retrain=false)

Overloads the `rand` method to randomly draw `n` samples from `EnergySampler`.
"""
function Base.rand(sampler::EnergySampler, n::Int=100; from_buffer=true, niter::Int=100)
    ntotal = size(sampler.buffer, 2)
    idx = rand(1:ntotal, n)
    if from_buffer
        X = sampler.buffer[:, idx]
    else
        X = generate_samples(sampler, n, sampler.yidx; niter=niter)
    end
    return X
end
