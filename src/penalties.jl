using ChainRules: ignore_derivatives
using LinearAlgebra: norm
using Statistics: mean

"""
    set_size_penalty(ce::AbstractCounterfactualExplanation)

Penalty for smooth conformal set size.
"""
function set_size_penalty(
    ce::AbstractCounterfactualExplanation; 
    κ::Real=0.0, temp::Real=0.05, agg=mean
)

    conf_model = ce.M.model
    fitresult = ce.M.fitresult
    X = CounterfactualExplanations.decode_state(ce)
    loss = map(eachslice(X, dims=ndims(X))) do x
        x = ndims(x) == 1 ? x[:,:] : x
        if target_probs(ce, x)[1] >= 0.5
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
    ce::AbstractCounterfactualExplanation;
    n::Int=10000, from_buffer=true, agg=mean, kwargs...
)
    conditional_samples = []
    ignore_derivatives() do
        _dict = ce.params
        if !(:energy_sampler ∈ collect(keys(_dict)))
            _dict[:energy_sampler] = ECCCo.EnergySampler(ce; kwargs...)
        end
        sampler = _dict[:energy_sampler]
        push!(conditional_samples, rand(sampler, n; from_buffer=from_buffer))
    end
    x′ = CounterfactualExplanations.counterfactual(ce)
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
    ce::AbstractCounterfactualExplanation;
    n::Int=10000, agg=mean
)
    target_idx = ce.data.output_encoder.labels .== ce.target
    target_samples = ce.data.X[:,target_idx] |>
        X -> X[:,rand(1:end,n)]
    x′ = CounterfactualExplanations.counterfactual(ce)
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

