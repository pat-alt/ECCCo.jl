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
                conf_model, fitresult, x;
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
    n::Int=100, retrain=false, kwargs...
)
    sampler = get!(counterfactual_explanation.params, :energy_sampler) do 
        CCE.EnergySampler(counterfactual_explanation; kwargs...)
    end
    conditional_samples = 
end