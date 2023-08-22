# General setup:
include("$(pwd())/notebooks/setup.jl")
eval(setup_notebooks)

test_size = 0.2

# Constants:
const DEFAULT_OUTPUT_PATH = "$(pwd())/results"
const RETRAIN = "retrain" ∈ ARGS ? true : false

"Default model performance measures."
const MODEL_MEASURES = Dict(
    :f1score => multiclass_f1score,
    :acc => accuracy,
    :precision => multiclass_precision
)

"Default coverage rate."
const DEFAULT_COVERAGE = 0.95

"The default benchmarking measures."
const CE_MEASURES = [
    CounterfactualExplanations.distance,
    ECCCo.distance_from_energy,
    ECCCo.distance_from_targets,
    CounterfactualExplanations.Evaluation.validity,
    CounterfactualExplanations.Evaluation.redundancy,
    ECCCo.set_size_penalty
]

# Pre-trained models:
function pretrained_path()
    @info "Models were pre-trained on `julia-1.8.5` and may not work on other versions."
    return joinpath(artifact"results-paper-submission-1.8.5", "results-paper-submission-1.8.5")
end

"Sets up the experiment."
Base.@kwdef struct Experiment
    counterfactual_data::CounterfactualData
    test_data::CounterfactualData
    dataname::String = "dataset"
    save_name::String = replace(lowercase(dataname), " " => "_")
    output_path::String = DEFAULT_OUTPUT_PATH
    params_path::String = joinpath(output_path, "params")
    use_pretrained::Bool = true
    models::Union{Nothing,Dict} = nothing
    builder::Union{Nothing,MLJFlux.GenericBuilder} = nothing
    𝒟x::Distribution = Normal()
    sampling_batch_size::Int = 50
    min_batch_size::Int = 128
    coverage::Float64 = DEFAULT_COVERAGE
    generators::Union{Nothing,Dict} = nothing
    n_individuals::Int = 50
    ce_measures::AbstractArray = CE_MEASURES
    model_measures::Dict = MODEL_MEASURES
end

"A container to hold the results of an experiment."
mutable struct ExperimentOutcome
    exp::Experiment
    model_dict::Union{Nothing, Dict}
    generator_dict::Union{Nothing, Dict}
    bmk::Union{Nothing, Benchmark}
end

# Scripts:
include("data/data.jl")
include("models/models.jl")
include("benchmarking/benchmarking.jl")
include("post_processing.jl")

"""
    run_experiment!(exp::Experiment)

Run the experiment specified by `exp`.
"""
function run_experiment!(exp::Experiment)
    
    # Setup
    @info "All results will be saved to $(exp.output_path)."
    isdir(exp.output_path) || mkdir(exp.output_path)
    @info "All parameter choices will be saved to $(exp.params_path)."
    isdir(exp.params_path) || mkdir(exp.params_path)

    # Models
    model_dict = prepare_models(exp)
    outcome = ExperimentOutcome(exp, model_dict, nothing, nothing)
    meta_model_performance(outcome)
    
    # Benchmark
    bmk, generator_dict = run_benchmark(exp, model_dict)
    outcome.bmk = bmk
    outcome.generator_dict = generator_dict

    # Save data:
    Serialization.serialize(joinpath(exp.output_path, "$(exp.save_name)_outcome.jls"), outcome)
    CSV.write(joinpath(exp.output_path, "$(exp.ave_name)_benchmark.csv"), bmk())

end

"""
    run_experiment!(counterfactual_data::CounterfactualData, test_data::CounterfactualData; kwargs...)

Overload the `run_experiment` function to allow for passing in `CounterfactualData` objects and other keyword arguments.
"""
function run_experiment!(counterfactual_data::CounterfactualData, test_data::CounterfactualData; kwargs...)
    # Parameters:
    exp = Experiment(
        counterfactual_data, test_data;
        kwargs...
    )
    return run_experiment!(exp)
end