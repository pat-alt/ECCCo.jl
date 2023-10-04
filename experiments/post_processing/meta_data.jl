"""
    all_meta(exper::Experiment)

Extract and save meta data about the experiment.
"""
function all_meta(
    outcome::ExperimentOutcome;
    save_output::Bool = false,
    params_path::Union{Nothing,String} = nothing,
)

    model_params = meta_model(outcome; save_output = save_output, params_path = params_path)
    model_performance = meta_model_performance(
        outcome;
        save_output = save_output,
        params_path = params_path,
    )
    generator_params =
        meta_generators(outcome; save_output = save_output, params_path = params_path)

    return model_params, model_performance, generator_params

end

"""
    meta_model(outcome::ExperimentOutcome)

Extract and save meta data about the data and models in `outcome.model_dict`.
"""
function meta_model(
    outcome::ExperimentOutcome;
    save_output::Bool = false,
    params_path::Union{Nothing,String} = nothing,
)

    # Unpack:
    exper = outcome.exper
    n_obs, batch_size = meta_data(exper)
    model_dict = outcome.model_dict

    params = DataFrame(
        Dict(
            :n_obs => Int.(round(n_obs / 10) * 10),
            :batch_size => batch_size,
            :dataname => exper.dataname,
            :sgld_batch_size => exper.sampling_batch_size,
            :epochs => exper.epochs,
            :n_hidden => exper.n_hidden,
            :n_layers => length(model_dict["MLP"].fitresult[1][1]) - 1,
            :activation => string(exper.activation),
            :n_ens => exper.n_ens,
            :lambda => string(exper.α[3]),
            :jem_sampling_steps => exper.sampling_steps,
        ),
    )

    if save_output
        save_path = isnothing(params_path) ? exper.params_path : params_path
        save_path = joinpath(save_path, "$(exper.save_name)_model_params.csv")
        @info "Saving model parameters to $(save_path)."
        CSV.write(save_path, params)
    end

    return params

end

function meta_generators(
    outcome::ExperimentOutcome;
    save_output::Bool = false,
    params_path::Union{Nothing,String} = nothing,
)

    # Unpack:
    exper = outcome.exper
    generator_dict = outcome.generator_dict
    Λ = exper.Λ
    Λ_Δ = exper.Λ_Δ
    reg_strengh = exper.reg_strength

    # Output:
    opt = first(values(generator_dict)).opt
    generator_params = DataFrame(
        Dict(
            :opt => string(typeof(opt)),
            :eta => opt.eta,
            :dataname => exper.dataname,
            :lambda_1 => string(Λ[1]),
            :lambda_2 => string(Λ[2]),
            :lambda_3 => string(Λ[3]),
            :lambda_1_Δ => string(Λ_Δ[1]),
            :lambda_2_Δ => string(Λ_Δ[2]),
            :lambda_3_Δ => string(Λ_Δ[3]),
            :n_individuals => exper.n_individuals,
            :reg_strength => string(reg_strengh),
        ),
    )

    if save_output
        save_path = isnothing(params_path) ? exper.params_path : params_path
        save_path = joinpath(save_path, "$(exper.save_name)_generator_params.csv")
        @info "Saving generator parameters to $(save_path)."
        CSV.write(save_path, generator_params)
    end

    return generator_params
end

"""
    meta_model_performance(outcome::ExperimentOutcome; measures=MODEL_MEASURES)

Compute and save the model performance for the models in `outcome.model_dict`.
"""
function meta_model_performance(
    outcome::ExperimentOutcome;
    measures::Union{Nothing,Dict} = nothing,
    save_output::Bool = false,
    params_path::Union{Nothing,String} = nothing,
)

    # Unpack:
    exper = outcome.exper
    measures = isnothing(measures) ? exper.model_measures : measures
    model_dict = outcome.model_dict

    # Model performance:
    model_performance = DataFrame()
    for (mod_name, model) in model_dict
        # Test performance:
        _perf = CounterfactualExplanations.Models.model_evaluation(
            model,
            exper.test_data,
            measure = collect(values(measures)),
        )
        _perf = DataFrame([[p] for p in _perf], collect(keys(measures)))
        _perf.mod_name .= mod_name
        _perf.dataname .= exper.dataname
        model_performance = vcat(model_performance, _perf)
    end

    @info "Model performance:"
    println(model_performance)

    if save_output
        save_path_root = isnothing(params_path) ? exper.params_path : params_path
        save_path = joinpath(save_path_root, "$(exper.save_name)_model_performance.jls")
        @info "Saving model performance to $(save_path)."
        Serialization.serialize(save_path, model_performance)
        save_path = joinpath(save_path_root, "$(exper.save_name)_model_performance.csv")
        @info "Saving model performance to $(save_path)."
        CSV.write(save_path, model_performance)
    end

    return model_performance
end
