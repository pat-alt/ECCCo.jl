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
        if isnothing(exp.models)
            @info "Using default models."
            models = default_models(;
                sampler=sampler,
                builder=builder,
                batch_size=batch_size(exp)
            )
        end
        @info "Training models."
        model_dict = train_models(models, X, labels; coverage=exp.coverage)
    else
        @info "Loading pre-trained models."
        model_dict = Serialization.deserialize(joinpath(pretrained_path(), "results/$(exp.save_name)_models.jls"))
    end

    # Save models:
    @info "Saving models to $(joinpath(exp.output_path, "$(exp.save_name)_models.jls"))."
    Serialization.serialize(joinpath(exp.output_path, "$(exp.save_name)_models.jls"), model_dict)

    return model_dict
end