using CounterfactualExplanations
using Statistics: mean

@doc raw"""
    conformal_training_loss(ce::AbstractCounterfactualExplanation; temp::Real=0.1, agg=mean, kwargs...)

A configurable classification loss function for Conformal Predictors.
"""
function conformal_training_loss(ce::AbstractCounterfactualExplanation; temp::Real=0.1, agg=mean, kwargs...)
    conf_model = ce.M.model
    fitresult = ce.M.fitresult
    X = CounterfactualExplanations.decode_state(ce)
    y = ce.target_encoded[:, :, 1]
    if ce.M.likelihood == :classification_binary
        y = binary_to_onehot(y)
    end
    y = permutedims(y)

    n_classes = length(ce.data.y_levels)
    loss_mat = ones(n_classes, n_classes)
    loss = map(eachslice(X, dims=ndims(X))) do x
        x = ndims(x) == 1 ? x[:,:]' : x
        ConformalPrediction.classification_loss(
            conf_model, fitresult, x, y;
            temp=temp, loss_matrix = loss_mat,
        )
    end
    loss = agg(loss)[1]
    return loss
end