using ConformalPrediction
using CounterfactualExplanations.Models
using Flux
using MLJBase
using MLJEnsembles
using MLJFlux
using MLUtils
using SliceMap
using Statistics

const CompatibleAtomicModel = Union{<:MLJFlux.MLJFluxProbabilistic,MLJEnsembles.ProbabilisticEnsembleModel{<:MLJFlux.MLJFluxProbabilistic}}

"""
    ConformalModel <: Models.AbstractDifferentiableJuliaModel

Constructor for models trained in `Flux.jl`. 
"""
struct ConformalModel <: Models.AbstractDifferentiableJuliaModel
    model::ConformalPrediction.ConformalProbabilisticSet
    fitresult::Any
    likelihood::Union{Nothing,Symbol}
    function ConformalModel(model, fitresult, likelihood)
        if likelihood ∈ [:classification_binary, :classification_multi] || isnothing(likelihood)
            new(model, fitresult, likelihood)
        else
            throw(
                ArgumentError(
                    "`likelihood` should either be `nothing` or in `[:classification_binary,:classification_multi]`",
                ),
            )
        end
    end
end

"""
    _get_chains(fitresult)

Private function that extracts the chains from a fitted model.
"""
function _get_chains(fitresult)
    if fitresult isa MLJEnsembles.WrappedEnsemble
        chains = map(res -> res[1], fitresult.ensemble)
    else
        chains = [fitresult[1]]
    end
    return chains
end

"""
    _outdim(fitresult)

Private function that extracts the output dimension from a fitted model.
"""
function _outdim(fitresult)
    if fitresult isa MLJEnsembles.WrappedEnsemble
        outdim = length(fitresult.ensemble[1][2])
    else
        outdim = length(fitresult[2])
    end
    return outdim
end

"""
    ConformalModel(model, fitresult=nothing; likelihood::Union{Nothing,Symbol}=nothing)

Outer constructor for `ConformalModel`. If `fitresult` is not specified, the model is not fitted and `likelihood` is inferred from the model. If `fitresult` is specified, `likelihood` is inferred from the output dimension of the model. If `likelihood` is not specified, it defaults to `:classification_binary`.
"""
function ConformalModel(model, fitresult=nothing; likelihood::Union{Nothing,Symbol}=nothing)

    # Check if model is fitted and infer likelihood:
    if isnothing(fitresult)
        @info "Conformal Model is not fitted."
    else
        outdim = _outdim(fitresult)
        _likelihood = outdim == 2 ? :classification_binary : :classification_multi
        @assert likelihood == _likelihood || isnothing(likelihood) "Specification of `likelihood` does not match the output dimension of the model."
        likelihood = _likelihood
    end

    # Default to binary classification, if not specified or inferred:
    if isnothing(likelihood)
        likelihood = :classification_binary
        @info "Likelihood not specified. Defaulting to $likelihood."
    end

    # Construct model:
    testmode!.(_get_chains(fitresult))
    M = ConformalModel(model, fitresult, likelihood)
    return M
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
    fitresult = M.fitresult
    function predict_logits(fitresult, x)
        p̂ = MLUtils.stack(map(chain -> chain(x),_get_chains(fitresult))) |> 
            p -> mean(p, dims=ndims(p)) |>
            p -> MLUtils.unstack(p, dims=ndims(p))[1]
        if ndims(p̂) == 2
            p̂ = [p̂]
        end
        p̂ = reduce(hcat, p̂)
        if all(0.0 .<= vec(p̂) .<= 1.0)
            ŷ = reduce(hcat, (map(p -> log.(p) .+ log(sum(exp.(p))), eachcol(p̂))))
        else
            ŷ = p̂
        end
        if M.likelihood == :classification_binary
            ŷ = reduce(hcat, (map(y -> y[2] - y[1], eachcol(ŷ))))
        end
        ŷ = ndims(ŷ) > 1 ? ŷ : permutedims([ŷ])
        return ŷ
    end
    if ndims(X) > 2
        yhat = map(eachslice(X, dims=ndims(X))) do x
            predict_logits(fitresult, x)
        end
        yhat = MLUtils.stack(yhat)
    else
        yhat = predict_logits(fitresult, X)
    end
    return yhat
end

"""
    Models.probs(M::ConformalModel, X::AbstractArray)

Returns the estimated probabilities for a Conformal Classifier.
"""
function Models.probs(M::ConformalModel, X::AbstractArray)
    if M.likelihood == :classification_binary
        output = σ.(Models.logits(M, X))
    elseif M.likelihood == :classification_multi
        output = softmax(Models.logits(M, X))
    end
    return output
end

"""
    train(M::ConformalModel, data::CounterfactualData; kwrgs...)

Trains a Conformal Classifier `M` on `data`. 
"""
function Models.train(M::ConformalModel, data::CounterfactualData; kwrgs...)
    X, y = data.X, data.output_encoder.labels
    X = table(permutedims(X))
    conf_model = M.model
    mach = machine(conf_model, X, y)
    fit!(mach; kwrgs...)
    likelihood, _ = CounterfactualExplanations.guess_likelihood(data.output_encoder.y)
    return ConformalModel(mach.model, mach.fitresult, likelihood)
end