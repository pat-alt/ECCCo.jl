using CounterfactualExplanations: AbstractParallelizer

"""
    train_models(models::Dict)

Trains all models in a dictionary and returns a dictionary of `ConformalModel` objects.
"""
function train_models(models::Dict, X, y; parallelizer::Union{Nothing,AbstractParallelizer}=nothing, train_parallel::Bool=false, kwargs...)
    verbose = is_multi_processed(parallelizer) ? false : true
    if is_multi_processed(parallelizer) && train_parallel
        # Split models into groups of approximately equal size:
        model_list = [(key, value) for (key, value) in models]
        x = split_obs(model_list, parallelizer.n_proc)
        x = MPI.scatter(x, parallelizer.comm)
        # Train models:
        model_dict = Dict()
        for (mod_name, model) in x
            model_dict[mod_name] = _train(model, X, y; mod_name=mod_name, verbose=verbose, kwargs...)
        end
        MPI.Barrier(parallelizer.comm)
        output = MPI.gather(output, parallelizer.comm)
        # Collect output from all processe in rank 0:
        if parallelizer.rank == 0
            output = merge(output...)
        else
            output = nothing
        end
        # Broadcast output to all processes:
        model_dict = MPI.bcast(output, parallelizer.comm; root=0)
        MPI.Barrier(parallelizer.comm)
    else
        model_dict = Dict(mod_name => _train(model, X, y; mod_name=mod_name, verbose=verbose, kwargs...) for (mod_name, model) in models)
    end
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
function _train(model, X, y; cov, method=:simple_inductive, mod_name="model", verbose::Bool=true)
    conf_model = conformal_model(model; method=method, coverage=cov)
    mach = machine(conf_model, X, y)
    @info "Begin training $mod_name."
    if verbose
        fit!(mach)
    else
        fit!(mach, verbosity=0)
    end
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