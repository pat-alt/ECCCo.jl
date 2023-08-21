# General setup:
include("$(pwd())/notebooks/setup.jl")
eval(setup_notebooks)
output_path = "$(pwd())/replicated"
isdir(output_path) || mkdir(output_path)
@info "All results will be saved to $output_path."
params_path = "$(pwd())/replicated/params"
isdir(params_path) || mkdir(params_path)
@info "All parameter choices will be saved to $params_path."
test_size = 0.2

# Constants:
if ENV["RETRAIN"] == "true"
    const RETRAIN = true
else
    const RETRAIN = false
end

# Artifacts:
using LazyArtifacts
@warn "Models were pre-trained on `julia-1.8.5` and may not work on other versions."
artifact_path = joinpath(artifact"results-paper-submission-1.8.5","results-paper-submission-1.8.5")
pretrained_path = joinpath(artifact_path, "results")

# Scripts:
include("data/data.jl")
include("models/models.jl")
include("benchmarking/benchmarking.jl")

"Sets up the experiment."
Base.@kwdef struct Experiment
    counterfactual_data::CounterfactualData
    test_data::CounterfactualData
    dataname::String = "dataset"
    output_path::String = output_path
    pretrained_path::String = pretrained_path
    use_pretrained::Bool = true
    models::Union{Nothing, Dict} = nothing
    builder::Union{Nothing, MLJFlux.GenericBuilder} = nothing
    𝒟x::Distribution = Normal()
    sampling_batch_size::Int = 50
    coverage::Float64 = 0.95
    generators::Union{Nothing, Dict} = nothing
    n_individuals::Int = 50
end

"""
    run_experiment(exp::Experiment)

Run the experiment specified by `exp`.
"""
function run_experiment(exp::Experiment)

    # SETUP ----------
    # Data
    X, labels, n_obs, save_name, batch_size, sampler = prepare_data(
        counterfactual_data;
        𝒟x=exp.𝒟x,
        sampling_batch_size=exp.sampling_batch_size,
    )

    # MODELS ----------
    if isnothing(builder)
        builder = default_builder()
    end
    if isnothing(models)
        @info "Using default models."
        models = default_models(;
            sampler=sampler,
            builder=builder,
            batch_size=batch_size,
        )
    end

    # TRAINING ----------
    if !pretrained
        @info "Training models."
        model_dict = train_models(models, X, labels; coverage=coverage)
        Serialization.serialize(joinpath(output_path, "$(save_name)_models.jls"), model_dict)
    else
        @info "Loading pre-trained models."
        model_dict = Serialization.deserialize(joinpath(pretrained_path, "$(save_name)_models.jls"))
    end

    params = DataFrame(
        Dict(
            :n_obs => Int.(round(n_obs/10)*10),
            :epochs => epochs,
            :batch_size => batch_size,
            :n_hidden => n_hidden,
            :n_layers => length(model_dict["MLP"].fitresult[1][1])-1,
            :activation => string(activation),
            :n_ens => n_ens,
            :lambda => string(α[3]),
            :jem_sampling_steps => jem.sampling_steps,
            :sgld_batch_size => sampler.batch_size,
            :dataname => dataname,
        )
    )
    CSV.write(joinpath(params_path, "$(save_name)_model_params.csv"), params)

    measure = Dict(
        :f1score => multiclass_f1score,
        :acc => accuracy,
        :precision => multiclass_precision
    )
    model_performance = DataFrame()
    for (mod_name, model) in model_dict
        # Test performance:
        _perf = CounterfactualExplanations.Models.model_evaluation(model, test_data, measure=collect(values(measure)))
        _perf = DataFrame([[p] for p in _perf], collect(keys(measure)))
        _perf.mod_name .= mod_name
        _perf.dataname .= dataname
        model_performance = vcat(model_performance, _perf)
    end
    Serialization.serialize(joinpath(output_path, "$(save_name)_model_performance.jls"), model_performance)
    CSV.write(joinpath(output_path, "$(save_name)_model_performance.csv"), model_performance)
    @info "Model performance:"
    println(model_performance)
    
    # COUNTERFACTUALS ----------
    # Benchmark generators:
    bmk, generator_dict = run_benchmark(;
        n_individuals=n_individuals,
        dataname=dataname,
        counterfactual_data=counterfactual_data,
        model_dict=model_dict,
        generators=generators,
        measures=measures,
    )

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
    CSV.write(joinpath(output_path, "$(save_name)_benchmark.csv"), bmk())

end

"""
    run_experiment(counterfactual_data::CounterfactualData, test_data::CounterfactualData; kwargs...)

Overload the `run_experiment` function to allow for passing in `CounterfactualData` objects and other keyword arguments.
"""
function run_experiment(counterfactual_data::CounterfactualData, test_data::CounterfactualData; kwargs...)
    # Parameters:
    exp = Experiment(
        counterfactual_data, test_data;
        kwargs...
    )
    return run_experiment(exp)
end

"""
    meta_data(exp::Experiment)

Extract and save meta data about the experiment.
"""
function meta_data(
    exp::Experiment; 
    save_path::Union{String,Nothing}=nothing,
    save_name::Union{String,Nothing}=nothing,
)

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
end