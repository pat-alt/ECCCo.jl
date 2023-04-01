using ChainRules: ignore_derivatives
using LinearAlgebra: norm
using Statistics: mean

"""
    set_size_penalty(counterfactual_explanation::AbstractCounterfactualExplanation)

Penalty for smooth conformal set size.
"""
function set_size_penalty(
    counterfactual_explanation::AbstractCounterfactualExplanation; 
    κ::Real=1.0, temp::Real=0.5, agg=mean
)

    conf_model = counterfactual_explanation.M.model
    fitresult = counterfactual_explanation.M.fitresult
    X = CounterfactualExplanations.decode_state(counterfactual_explanation)
    loss = map(eachslice(X, dims=3)) do x
        x = Matrix(x)
        if target_probs(counterfactual_explanation, x)[1] >= 0.5
            l = ConformalPrediction.smooth_size_loss(
                conf_model, fitresult, x';
                κ=κ,
                temp=temp
            )[1]
        else 
            l = 0.0
        end
        return l
    end
    loss = agg(loss)

    return loss

end

function distance_from_energy(
    counterfactual_explanation::AbstractCounterfactualExplanation;
    n::Int=100, from_buffer=true, agg=mean, kwargs...
)
    conditional_samples = []
    ignore_derivatives() do
        _dict = counterfactual_explanation.params
        if !(:energy_sampler ∈ collect(keys(_dict)))
            _dict[:energy_sampler] = CCE.EnergySampler(counterfactual_explanation; kwargs...)
        end
        sampler = _dict[:energy_sampler]
        push!(conditional_samples, rand(sampler, n; from_buffer=from_buffer))
    end
    x′ = CounterfactualExplanations.counterfactual(counterfactual_explanation)
    loss = map(eachslice(x′, dims=3)) do x
        x = Matrix(x)
        Δ = map(eachcol(conditional_samples[1])) do xsample
            norm(x - xsample)
        end
        return mean(Δ)
    end
    loss = agg(loss)[1]

    return loss

end

function distance_from_targets(
    counterfactual_explanation::AbstractCounterfactualExplanation;
    n::Int=100, agg=mean
)
    target_samples = counterfactual_explanation.data.X |>
        X -> X[:,rand(1:end,n)]
    x′ = CounterfactualExplanations.counterfactual(counterfactual_explanation)
    loss = map(eachslice(x′, dims=3)) do x
        x = Matrix(x)
        Δ = map(eachcol(target_samples)) do xsample
            norm(x - xsample)
        end
        return mean(Δ)
    end
    loss = agg(loss)[1]

    return loss

end

