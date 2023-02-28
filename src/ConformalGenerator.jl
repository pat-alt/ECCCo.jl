using CategoricalArrays
using CounterfactualExplanations
using CounterfactualExplanations.Generators
using CounterfactualExplanations.Models: binary_to_onehot
using CounterfactualExplanations.Objectives
using Flux
using LinearAlgebra
using Parameters
using SliceMap
using Statistics

mutable struct ConformalGenerator <: AbstractGradientBasedGenerator
    loss::Union{Nothing,Function} # loss function
    complexity::Function # complexity function
    λ::Union{AbstractFloat,AbstractVector} # strength of penalty
    decision_threshold::Union{Nothing,AbstractFloat}
    opt::Flux.Optimise.AbstractOptimiser # optimizer
    τ::AbstractFloat # tolerance for convergence
    κ::Real
    temp::Real
end

# API streamlining:
@with_kw struct ConformalGeneratorParams
    opt::Flux.Optimise.AbstractOptimiser = Descent()
    τ::AbstractFloat = 1e-3
    κ::Real = 1.0
    temp::Real = 0.5
end

"""
    ConformalGenerator(;
        loss::Union{Nothing,Function}=conformal_training_loss,
        complexity::Function=norm,
        λ::Union{AbstractFloat,AbstractVector}=[0.1, 1.0],
        decision_threshold=nothing,
        kwargs...
    )

An outer constructor method that instantiates a generic generator.

# Examples
```julia-repl
generator = ConformalGenerator()
```
"""
function ConformalGenerator(;
    loss::Union{Nothing,Function}=conformal_training_loss,
    complexity::Function=norm,
    λ::Union{AbstractFloat,AbstractVector}=[0.1, 1.0],
    decision_threshold=nothing,
    kwargs...
)
    params = ConformalGeneratorParams(; kwargs...)
    ConformalGenerator(loss, complexity, λ, decision_threshold, params.opt, params.τ, params.κ, params.temp)
end

@doc raw"""
    conformal_training_loss(counterfactual_explanation::AbstractCounterfactualExplanation; kwargs...)

A configurable classification loss function for Conformal Predictors.
"""
function conformal_training_loss(counterfactual_explanation::AbstractCounterfactualExplanation; kwargs...)
    conf_model = counterfactual_explanation.M.model
    fitresult = counterfactual_explanation.M.fitresult
    X = CounterfactualExplanations.decode_state(counterfactual_explanation)
    y = counterfactual_explanation.target_encoded[:,:,1]
    if counterfactual_explanation.M.likelihood == :classification_binary
        y = binary_to_onehot(y)
    end
    y = permutedims(y)
    generator = counterfactual_explanation.generator
    loss = SliceMap.slicemap(X, dims=(1, 2)) do x
        x = Matrix(x)
        ConformalPrediction.classification_loss(
            conf_model, fitresult, x, y;
            temp=generator.temp
        )
    end
    loss = mean(loss)
    return loss
end

"""
    set_size_penalty(
        generator::ConformalGenerator,
        counterfactual_explanation::AbstractCounterfactualExplanation,
    )

Additional penalty for ConformalGenerator.
"""
function set_size_penalty(
    generator::ConformalGenerator,
    counterfactual_explanation::AbstractCounterfactualExplanation,
)

    conf_model = counterfactual_explanation.M.model
    fitresult = counterfactual_explanation.M.fitresult
    X = CounterfactualExplanations.decode_state(counterfactual_explanation)
    loss = SliceMap.slicemap(X, dims=(1, 2)) do x
        x = Matrix(x)
        ConformalPrediction.smooth_size_loss(
            conf_model, fitresult, x;
            κ=generator.κ,
            temp=generator.temp
        )
    end
    loss = mean(loss)

    return loss

end

# Complexity:
"""
    h(generator::AbstractGenerator, counterfactual_explanation::AbstractCounterfactualExplanation)

The default method to apply the generator complexity penalty to the current counterfactual state for any generator.
"""
function Generators.h(
    generator::ConformalGenerator,
    counterfactual_explanation::AbstractCounterfactualExplanation,
)

    # Distance from factual:
    dist_ = generator.complexity(
        counterfactual_explanation.x .-
        CounterfactualExplanations.decode_state(counterfactual_explanation),
    )

    # Set size penalty:
    in_target_domain = all(target_probs(counterfactual_explanation) .>= 0.5)
    if in_target_domain
        Ω = set_size_penalty(generator, counterfactual_explanation)
    else
        Ω = 0
    end

    if length(generator.λ) == 1
        penalty = generator.λ * (dist_ .+ Ω)
    else
        penalty = generator.λ[1] * dist_ .+ generator.λ[2] * Ω
    end

    return penalty
end
