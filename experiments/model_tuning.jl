"""
    tune_model(exper::Experiment; kwargs...)

Tunes MLP in place and saves the tuned model to disk.
"""
function tune_model(exper::Experiment; kwargs...)
    if !(is_multi_processed(exper) && MPI.Comm_rank(exper.parallelizer.comm) != 0)
        @info "Tuning models."
        # Output path:
        model_tuning_path = mkpath(joinpath(DEFAULT_OUTPUT_PATH, "tuned_model"))
        # Simple MLP:
        mod = NeuralNetworkClassifier(
            builder=default_builder(),
            epochs=exper.epochs,
            batch_size=batch_size(exper),
            finaliser=exper.finaliser,
            loss=exper.loss,
            acceleration=CUDALibs(),
        )
        mach = tune_model(mod, X, y; tuning_params=exper.model_tuning_params, measure=exper.model_measures, kwargs...)
        Serialization.serialize(joinpath(model_tuning_path, "$(exper.save_name).jls"), mach)
    end
    return mach
end

"""
    tune_model(mod::Supervised, X, y; tuning_params::NamedTuple, kwargs...)

Tunes a model by performing a grid search over the parameters specified in `tuning_params`.
"""
function tune_model(mod::Supervised, X, y; tuning_params::NamedTuple, measure=MODEL_MEASURES, kwargs...)

    ranges = []

    for (k, v) in pairs(tuning_params)
        if k ∈ fieldnames(typeof(model))
            push!(ranges, range(mod, k, values=v))
        elseif k ∈ fieldnames(typeof(model.builder))
            push!(ranges, range(mod, :(builder.$(k)), values=v))
        elseif k ∈ fieldnames(typeof(model.optimiser))
            push!(ranges, range(mod, :(optimiser.$(k)), values=v))
        else
            error("Parameter $k not found in model, builder or optimiser.")
        end
    end
    
    self_tuning_mod = TunedModel(
        model=mod,
        range=ranges,
        measure=measure,
        kwargs...
    )

    mach = machine(self_tuning_mod, X, y)
    fit!(mach, verbosity=0)

    return mach

end

