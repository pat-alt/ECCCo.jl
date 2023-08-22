"""
    meta_model_performance(outcome::ExperimentOutcome; measure=MODEL_MEASURES)

Compute and save the model performance for the models in `outcome.model_dict`.
"""
function meta_model_performance(outcome::ExperimentOutcome; measure=MODEL_MEASURES)

    exp = outcome.exp
    model_dict = outcome.model_dict

    # Model performance:
    model_performance = DataFrame()
    for (mod_name, model) in model_dict
        # Test performance:
        _perf = CounterfactualExplanations.Models.model_evaluation(model, exp.test_data, measure=collect(values(measure)))
        _perf = DataFrame([[p] for p in _perf], collect(keys(measure)))
        _perf.mod_name .= mod_name
        _perf.dataname .= dataname
        model_performance = vcat(model_performance, _perf)
    end
    Serialization.serialize(joinpath(exp.output_path, "$(exp.save_name)_model_performance.jls"), model_performance)
    CSV.write(joinpath(exp.output_path, "$(exp.save_name)_model_performance.csv"), model_performance)
    @info "Model performance:"
    println(model_performance)
    return model_performance
end

"""
    meta_data(exp::Experiment)

Extract and save meta data about the experiment.
"""
function meta_data(outcome::ExperimentOutcome)

    # Data params:
    _, _, n_obs, default_save_name, batch_size, sampler = prepare_data(
        exp.counterfactual_data;
        𝒟x=exp.𝒟x,
        sampling_batch_size=exp.sampling_batch_size
    )
    save_name = isnothing(save_name) ? default_save_name : save_name

    params = DataFrame(
        Dict(
            :n_obs => Int.(round(n_obs / 10) * 10),
            :epochs => epochs,
            :batch_size => batch_size,
            :n_hidden => n_hidden,
            :n_layers => length(model_dict["MLP"].fitresult[1][1]) - 1,
            :activation => string(activation),
            :n_ens => n_ens,
            :lambda => string(α[3]),
            :jem_sampling_steps => jem.sampling_steps,
            :sgld_batch_size => sampler.batch_size,
            :dataname => dataname,
        )
    )
    if !isnothing(save_path)
        CSV.write(joinpath(save_path, "$(save_name)_model_params.csv"), params)
    end

    # Output:
    opt = first(values(generator_dict)).opt
    generator_params = DataFrame(
        Dict(
            :opt => string(typeof(opt)),
            :eta => opt.eta,
            :dataname => dataname,
        )
    )
    CSV.write(joinpath(params_path, "$(save_name)_generator_params.csv"), generator_params)

end