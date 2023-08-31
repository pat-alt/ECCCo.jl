"""
    train_models(models::Dict)

Trains all models in a dictionary and returns a dictionary of `ConformalModel` objects.
"""
function train_models(models::Dict, X, y; kwargs...)
    model_dict = Dict(mod_name => _train(model, X, y; mod_name=mod_name, kwargs...) for (mod_name, model) in models)
    return model_dict
end

"""
    _train(
        model::AbstractModel,
        X::AbstractMatrix,
        y::AbstractVector,
        cov::Float64=0.95,
        method::Symbol=:simple_inductive,
        mod_name::String="model"
    )

Trains a model and returns a `ConformalModel` object.
"""
function _train(model, X, y; cov, method=:simple_inductive, mod_name="model")
    conf_model = conformal_model(model; method=method, coverage=cov)
    mach = machine(conf_model, X, y)
    @info "Begin training $mod_name."
    fit!(mach)
    @info "Finished training $mod_name."
    M = ECCCo.ConformalModel(mach.model, mach.fitresult)
    return M
end

"""
    save_models(model_dict::Dict; save_name::String, output_path)

Helper function to save models.
"""
function save_models(model_dict::Dict; save_name::String, output_path)
    Serialization.serialize(joinpath(output_path, "$(save_name)_models.jls"), model_dict)
end