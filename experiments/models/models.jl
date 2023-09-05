include("additional_models.jl")
include("default_models.jl")
include("train_models.jl")

function prepare_models(exp::Experiment)

    # Unpack data:
    X, labels, sampler = prepare_data(exp::Experiment)

    # Training:
    if !exp.use_pretrained
        if isnothing(exp.builder)
            builder = default_builder()
        end
        # Default models:
        if isnothing(exp.models)
            @info "Using default models."
            models = default_models(;
                sampler=sampler,
                builder=builder,
                batch_size=batch_size(exp),
                sampling_steps=exp.sampling_steps,
                α=exp.α,
                n_ens=exp.n_ens,
                use_ensembling=exp.use_ensembling,
                finaliser=exp.finaliser,
                loss=exp.loss,
            )
        end
        # Additional models:
        if !isnothing(exp.additional_models)
            @info "Using additional models."
            add_models = Dict{Any,Any}()
            for (k, mod) in exp.additional_models
                add_models[k] = mod(;
                    batch_size=batch_size(exp),
                    finaliser=exp.finaliser,
                    loss=exp.loss,
                )
            end
            add_models = Dict(k => mod(;batch_size=batch_size(exp), ) for (k, mod) in exp.additional_models)
            models = merge!(models, exp.additional_models)
        end
        @info "Training models."
        model_dict = train_models(models, X, labels; cov=exp.coverage)
    else
        @info "Loading pre-trained models."
        model_dict = Serialization.deserialize(joinpath(pretrained_path(exp), "$(exp.save_name)_models.jls"))
    end

    # Save models:
    @info "Saving models to $(joinpath(exp.output_path, "$(exp.save_name)_models.jls"))."
    Serialization.serialize(joinpath(exp.output_path, "$(exp.save_name)_models.jls"), model_dict)

    return model_dict
end