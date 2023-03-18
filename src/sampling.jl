using CounterfactualExplanations
using Distributions
using Flux
using JointEnergyModels

(model::AbstractFittedModel)(x) = log.(CounterfactualExplanations.predict_proba(model, nothing, x))

mutable struct EnergySampler
    ce::CounterfactualExplanation
    sampler::JointEnergyModels.ConditionalSampler
    opt::JointEnergyModels.AbstractSamplingRule
    buffer::AbstractArray
end

function EnergySampler(
    ce::CounterfactualExplanation;
    opt::JointEnergyModels.AbstractSamplingRule=ImproperSGLD(),
    niter::Int=100,
    nsamples::Int=1000
)

    # Setup:
    model = ce.M
    data = ce.data
    K = length(data.y_levels)
    𝒟x = Normal()
    𝒟y = Categorical(ones(K) ./ K)
    sampler = ConditionalSampler(𝒟x, 𝒟y)

    # Fit:
    i = get_target_index(data.y_levels, ce.target)
    buffer = sampler(model, opt, (size(data.X, 1), nsamples); niter=niter, y=i)

    return EnergySampler(ce, sampler, opt, buffer)
end

function Base.rand(sampler::EnergySampler, n::Int=100; retrain=false)
    ntotal = size(sampler.buffer,2)
    idx = rand(1:ntotal, n)
    if !retrain
        X = sampler.buffer[:,idx]
    end
    return X
end
