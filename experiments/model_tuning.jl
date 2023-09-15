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
        model = NeuralNetworkClassifier(
            builder=default_builder(),
            epochs=exper.epochs,
            batch_size=batch_size(exper),
            finaliser=exper.finaliser,
            loss=exper.loss,
            acceleration=CUDALibs(),
        )
        # Unpack data:
        X, y, _ = prepare_data(exper::Experiment)
        # Tune model:
        measure = collect(values(exper.model_measures))
        mach = tune_model(model, X, y; tuning_params=exper.model_tuning_params, measure=measure, kwargs...)
        Serialization.serialize(joinpath(model_tuning_path, "$(exper.save_name).jls"), mach)
    end
    return mach
end

"""
    tune_model(mod::Supervised, X, y; tuning_params::NamedTuple, kwargs...)

Tunes a model by performing a grid search over the parameters specified in `tuning_params`.
"""
function tune_model(
    model::Supervised, X, y; 
    tuning_params::NamedTuple,
    measure::Vector=MODEL_MEASURE_VEC, 
    tuning=Grid(shuffle=false), 
    resampling=CV(nfolds=3, shuffle=true,), 
    kwargs...
)

    ranges = []

    for (k, v) in pairs(tuning_params)
        if k ∈ fieldnames(typeof(model))
            push!(ranges, range(model, k, values=v))
        elseif k ∈ fieldnames(typeof(model.builder))
            push!(ranges, range(model, :(builder.$(k)), values=v))
        elseif k ∈ fieldnames(typeof(model.optimiser))
            push!(ranges, range(model, :(optimiser.$(k)), values=v))
        else
            error("Parameter $k not found in model, builder or optimiser.")
        end
    end
    
    self_tuning_mod = TunedModel(
        model=model,
        range=ranges,
        measure=measure,
        tuning=tuning,
        resampling=resampling,
        kwargs...
    )

    mach = machine(self_tuning_mod, X, y)
    fit!(mach, verbosity=0)

    return mach

end

