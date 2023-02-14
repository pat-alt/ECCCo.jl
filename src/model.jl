using ConformalPrediction
using CounterfactualExplanations.Models
using Flux
using MLUtils
using SliceMap
using Statistics

"""
    Models.ConformalModel <: AbstractDifferentiableJuliaModel

Constructor for models trained in `Flux.jl`. 
"""
struct Models.ConformalModel <: AbstractDifferentiableJuliaModel
    model::ConformalPrediction.ConformalProbabilisticSet
    likelihood::Symbol
    function FluxModel(model, likelihood)
        if likelihood ∈ [:classification_binary, :classification_multi]
            new(model, likelihood)
        else
            throw(
                ArgumentError(
                    "`type` should be in `[:classification_binary,:classification_multi]`",
                ),
            )
        end
    end
end

# Outer constructor method:
function Models.ConformalModel(model; likelihood::Symbol=:classification_binary)
    Models.ConformalModel(model, likelihood)
end

# Methods
function logits(M::Models.ConformalModel, X::AbstractArray)
    return SliceMap.slicemap(x -> M.model(x), X, dims=(1, 2))
end

function probs(M::Models.ConformalModel, X::AbstractArray)
    if M.likelihood == :classification_binary
        output = σ.(logits(M, X))
    elseif M.likelihood == :classification_multi
        output = softmax(logits(M, X))
    end
    return output
end