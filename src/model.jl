using ConformalPrediction
using CounterfactualExplanations.Models
using Flux
using MLJBase
using MLUtils
using SliceMap
using Statistics

"""
    ConformalModel <: Models.AbstractDifferentiableJuliaModel

Constructor for models trained in `Flux.jl`. 
"""
struct ConformalModel <: Models.AbstractDifferentiableJuliaModel
    model::ConformalPrediction.ConformalProbabilisticSet
    fitresult::Any
    likelihood::Symbol
    function ConformalModel(model, fitresult, likelihood)
        if likelihood ∈ [:classification_binary, :classification_multi]
            new(model, fitresult, likelihood)
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
function ConformalModel(model, fitresult; likelihood::Symbol=:classification_binary)
    ConformalModel(model, fitresult, likelihood)
end

# Methods
@doc raw"""
    Models.logits(M::ConformalModel, X::AbstractArray)

To keep things consistent with the architecture of `CounterfactualExplanations.jl`, this method computes logits $\beta_i x_i$ (i.e. the linear predictions) for a Conformal Classifier. By default, `MLJ.jl` and `ConformalPrediction.jl` return probabilistic predictions. To get the underlying logits, we invert the softmax function. 

Let $\hat{p}_i$ denote the estimated softmax output for feature $i$. Then in the multi-class case the following formula can be applied:

```math
\beta_i x_i = \log (\hat{p}_i) + \log (\sum_i \exp(\hat{p}_i))
```

For a short derivation, see here: https://math.stackexchange.com/questions/2786600/invert-the-softmax-function. 

In the binary case logits are fed through the sigmoid function instead of softmax, so we need to further adjust as follows,

```math
\beta x = \beta_1 x_1 - \beta_0 x_0
```    

which follows from the derivation here: https://stats.stackexchange.com/questions/233658/softmax-vs-sigmoid-function-in-logistic-classifier
"""
function Models.logits(M::ConformalModel, X::AbstractArray)
    yhat = SliceMap.slicemap(X, dims=(1, 2)) do x
        conf_model = M.model
        fitresult = M.fitresult
        x = MLJBase.table(permutedims(x))
        p̂ = MMI.predict(conf_model.model, fitresult, MMI.reformat(conf_model.model, x)...)
        p̂ = map(p̂) do pp
            L = p̂.decoder.classes
            probas = pdf.(pp, L)
            return probas
        end
        p̂ = reduce(hcat, p̂)
        ŷ = reduce(hcat, (map(p -> log.(p) .+ log(sum(exp.(p))), eachcol(p̂))))
        if M.likelihood == :classification_binary
            ŷ = reduce(hcat, (map(y -> y[2] - y[1], eachcol(ŷ))))
        end
        return ŷ
    end
    return yhat
end

function Models.probs(M::ConformalModel, X::AbstractArray)
    if M.likelihood == :classification_binary
        output = σ.(Models.logits(M, X))
    elseif M.likelihood == :classification_multi
        output = softmax(Models.logits(M, X))
    end
    return output
end