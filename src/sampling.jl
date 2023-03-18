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
    nsamples::Int=1000
)

    @assert y ∈ data.y_levels || y ∈ 1:length(data.y_levels)
    K = length(data.y_levels)
    𝒟x = Normal()
    𝒟y = Categorical(ones(K) ./ K)
    sampler = ConditionalSampler(𝒟x, 𝒟y)
    yidx = get_target_index(data.y_levels, y)

    # Initiate:
    energy_sampler = EnergySampler(model, data, sampler, opt, nothing, nothing)

    # Generate samples:
    generate_samples!(energy_sampler, nsamples, yidx; niter=niter)

    return energy_sampler
end

"""
    generate_samples(e::EnergySampler, n::Int, y::Int; niter::Int=100)

Generates `n` samples from `EnergySampler` for conditioning value `y`.
"""
function generate_samples(e::EnergySampler, n::Int, y::Int; niter::Int=100)
    X = e.sampler(e.model, e.opt, (size(e.data.X, 1), n); niter=niter, y=y)
    return X
end

"""
    generate_samples!(e::EnergySampler, n::Int, y::Int; niter::Int=100)

Generates `n` samples from `EnergySampler` for conditioning value `y`. Assigns samples and conditioning value to `EnergySampler`.
"""
function generate_samples!(e::EnergySampler, n::Int, y::Int; niter::Int=100)
    e.buffer = generate_samples(e,n,y;niter=niter)
    e.yidx = y
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
