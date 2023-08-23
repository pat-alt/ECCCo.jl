"""
    meta(exp::Experiment)

Extract and save meta data about the experiment.
"""
function meta(outcome::ExperimentOutcome; save_output::Bool=false)

    model_params = meta_model(outcome; save_output=save_output)
    model_performance = meta_model_performance(outcome; save_output=save_output)
    generator_params = meta_generators(outcome; save_output=save_output)

    return model_params, model_performance, generator_params
    
end

"""
    meta_model(outcome::ExperimentOutcome)

Extract and save meta data about the data and models in `outcome.model_dict`.
"""
function meta_model(outcome::ExperimentOutcome; save_output::Bool=false)

    # Unpack:
    exp = outcome.exp
    n_obs, batch_size = meta_data(exp)

    params = DataFrame(
        Dict(
            :n_obs => Int.(round(n_obs / 10) * 10),
            :batch_size => batch_size,
            :dataname => exp.dataname,
            :sgld_batch_size => exp.sampling_batch_size,
            # :epochs => exp.epochs,
            # :n_hidden => n_hidden,
            # :n_layers => length(model_dict["MLP"].fitresult[1][1]) - 1,
            # :activation => string(activation),
            # :n_ens => n_ens,
            # :lambda => string(α[3]),
            # :jem_sampling_steps => jem.sampling_steps,
        )
    )
   
    if save_output
        save_path = joinpath(exp.params_path, "$(exp.save_name)_model_params.csv")
        @info "Saving model parameters to $(save_path)."
        CSV.write(save_path, params)
    end

    return params

end

function meta_generators(outcome::ExperimentOutcome; save_output::Bool=false)

    # Unpack:
    exp = outcome.exp
    generator_dict = outcome.generator_dict

    # Output:
    opt = first(values(generator_dict)).opt
    generator_params = DataFrame(
        Dict(
            :opt => string(typeof(opt)),
            :eta => opt.eta,
            :dataname => dataname,
        )
    )

    if save_output
        save_path = joinpath(exp.params_path, "$(exp.save_name)_generator_params.csv")
        @info "Saving generator parameters to $(save_path)."
        CSV.write(save_path, generator_params)
    end

    return generator_params
end

"""
    meta_model_performance(outcome::ExperimentOutcome; measures=MODEL_MEASURES)

Compute and save the model performance for the models in `outcome.model_dict`.
"""
function meta_model_performance(outcome::ExperimentOutcome; measures::Union{Nothing,Dict}=nothing, save_output::Bool=false)

    # Unpack:
    exp = outcome.exp
    measures = isnothing(measures) ? exp.model_measures : measures
    model_dict = outcome.model_dict

    # Model performance:
    model_performance = DataFrame()
    for (mod_name, model) in model_dict
        # Test performance:
        _perf = CounterfactualExplanations.Models.model_evaluation(model, exp.test_data, measure=collect(values(measures)))
        _perf = DataFrame([[p] for p in _perf], collect(keys(measures)))
        _perf.mod_name .= mod_name
        _perf.dataname .= exp.dataname
        model_performance = vcat(model_performance, _perf)
    end
    
    @info "Model performance:"
    println(model_performance)

    if save_output
        save_path = joinpath(exp.params_path, "$(exp.save_name)_model_performance.jls")
        @info "Saving model performance to $(save_path)."
        Serialization.serialize(save_path, model_performance)
        save_path = joinpath(exp.params_path, "$(exp.save_name)_model_performance.csv")
        @info "Saving model performance to $(save_path)."
        CSV.write(save_path, model_performance)
    end

    return model_performance
end