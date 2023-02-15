using CounterfactualExplanations
using CounterfactualExplanations.Generators
using Flux
using LinearAlgebra
using Parameters
using Statistics

mutable struct ConformalGenerator <: AbstractGradientBasedGenerator
    loss::Union{Nothing,Symbol} # loss function
    complexity::Function # complexity function
    λ::Union{AbstractFloat,AbstractVector} # strength of penalty
    decision_threshold::Union{Nothing,AbstractFloat}
    opt::Flux.Optimise.AbstractOptimiser # optimizer
    τ::AbstractFloat # tolerance for convergence
end

# API streamlining:
@with_kw struct ConformalGeneratorParams
    opt::Flux.Optimise.AbstractOptimiser = Descent()
    τ::AbstractFloat = 1e-3
    κ::Real = 1.0
    Τ::Real = 0.5
end

"""
    ConformalGenerator(
        ;
        loss::Symbol=:logitbinarycrossentropy,
        complexity::Function=norm,
        λ::AbstractFloat=0.1,
        opt::Flux.Optimise.AbstractOptimiser=Flux.Optimise.Descent(),
        τ::AbstractFloat=1e-5
    )

An outer constructor method that instantiates a generic generator.

# Examples
```julia-repl
generator = ConformalGenerator()
```
"""
function ConformalGenerator(;
    loss::Union{Nothing,Symbol} = nothing,
    complexity::Function = norm,
    λ::Union{AbstractFloat,AbstractVector} = [0.1, 1.0],
    decision_threshold = nothing,
    kwargs...,
)
    params = ConformalGeneratorParams(; kwargs...)
    ConformalGenerator(loss, complexity, λ, decision_threshold, params.opt, params.τ)
end

# Loss:
# """
#     ℓ(generator::ConformalGenerator, counterfactual_explanation::AbstractCounterfactualExplanation)

# The default method to apply the generator loss function to the current counterfactual state for any generator.
# """
# function ℓ(
#     generator::ConformalGenerator,
#     counterfactual_explanation::AbstractCounterfactualExplanation,
# )

#     loss_fun =
#         !isnothing(generator.loss) ? getfield(Losses, generator.loss) :
#         CounterfactualExplanations.guess_loss(counterfactual_explanation)
#     @assert !isnothing(loss_fun) "No loss function provided and loss function could not be guessed based on model."
#     loss = loss_fun(
#         getfield(Models, :logits)(
#             counterfactual_explanation.M,
#             CounterfactualExplanations.decode_state(counterfactual_explanation),
#         ),
#         counterfactual_explanation.target_encoded,
#     )
#     return loss
# end

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

    # Euclidean norm of gradient:
    in_target_domain = all(target_probs(counterfactual_explanation) .>= 0.5)
    if in_target_domain
        grad_norm = gradient_penalty(generator, counterfactual_explanation)
    else
        grad_norm = 0
    end

    if length(generator.λ) == 1
        penalty = generator.λ * (dist_ .+ grad_norm)
    else
        penalty = generator.λ[1] * dist_ .+ generator.λ[2] * grad_norm
    end
    return penalty
end
