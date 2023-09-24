include("additional_models.jl")
include("default_models.jl")
include("train_models.jl")

function prepare_models(exper::Experiment; save_models::Bool=true)

    # Unpack data:
    X, labels, sampler = prepare_data(exper::Experiment)

    # Training:
    if !exper.use_pretrained
        if isnothing(exper.builder)
            if tuned_mlp_exists(exper) && exper.use_tuned
                @info "Loading tuned model architecture."
                # Load the best MLP:
                best_mlp = Serialization.deserialize(joinpath(tuned_model_path(exper), "$(exper.save_name)_best_mlp.jls"))
                builder = best_mlp.best_model.builder
            else
                # Otherwise, use default MLP:
                builder = default_builder(n_hidden=exper.n_hidden, n_layers=exper.n_layers, activation=exper.activation)
            end
        else
            builder = exper.builder
        end
        # Default models:
        if isnothing(exper.models)
            @info "Using default models."
            models = default_models(;
                sampler=sampler,
                builder=builder,
                batch_size=batch_size(exper),
                sampling_steps=exper.sampling_steps,
                α=exper.α,
                n_ens=exper.n_ens,
                use_ensembling=exper.use_ensembling,
                finaliser=exper.finaliser,
                loss=exper.loss,
                epochs=exper.epochs,
            )
        end
        # Additional models:
        if !isnothing(exper.additional_models)
            @info "Using additional models."
            add_models = Dict{Any,Any}()
            for (k, mod) in exper.additional_models
                if isa(mod, Function)
                    add_models[k] = mod(;
                        batch_size=batch_size(exper),
                        finaliser=exper.finaliser,
                        loss=exper.loss,
                        epochs=exper.epochs,
                    )
                else
                    add_models[k] = mod
                end
            end
            models = merge(models, add_models)
        end
        @info "Training models."
        model_dict = train_models(models, X, labels; parallelizer=exper.parallelizer, train_parallel=exper.train_parallel, cov=exper.coverage)
    else
        # Pre-trained models:
        if !(is_multi_processed(exper) && MPI.Comm_rank(exper.parallelizer.comm) != 0)
            # Load models on root process:
            @info "Loading pre-trained models."
            model_dict = Serialization.deserialize(joinpath(pretrained_path(exper), "$(exper.save_name)_models.jls"))
        else
            # Dummy model on other processes: 
            model_dict = nothing
        end
        # Broadcast models:
        if is_multi_processed(exper)
            model_dict = MPI.bcast(model_dict, exper.parallelizer.comm; root=0)
        end
    end

    # Save models:
    local_models_exist = isfile(joinpath(DEFAULT_OUTPUT_PATH, "$(exper.save_name)_models.jls"))
    on_root_process = !(is_multi_processed(exper) && MPI.Comm_rank(exper.parallelizer.comm) != 0)
    if save_models && on_root_process && !local_models_exist
        @info "Saving models to $(joinpath(exper.output_path , "$(exper.save_name)_models.jls"))."
        Serialization.serialize(joinpath(exper.output_path, "$(exper.save_name)_models.jls"), model_dict)
    end

    return model_dict
end