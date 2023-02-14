using ConformalPrediction
using CounterfactualExplanations.Models
using Flux
using MLUtils
using SliceMap
using Statistics

"""
    ConformalModel <: Models.AbstractDifferentiableJuliaModel

Constructor for models trained in `Flux.jl`. 
"""
struct ConformalModel <: Models.AbstractDifferentiableJuliaModel
    model::ConformalPrediction.ConformalProbabilisticSet
    likelihood::Symbol
    function ConformalModel(model, likelihood)
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
function ConformalModel(model; likelihood::Symbol=:classification_binary)
    ConformalModel(model, likelihood)
end

# Methods
function logits(M::ConformalModel, X::AbstractArray)
    return SliceMap.slicemap(x -> M.model(x), X, dims=(1, 2))
end

function probs(M::ConformalModel, X::AbstractArray)
    if M.likelihood == :classification_binary
        output = σ.(logits(M, X))
    elseif M.likelihood == :classification_multi
        output = softmax(logits(M, X))
    end
    return output
end