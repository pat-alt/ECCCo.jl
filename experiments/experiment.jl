"Sets up the experiment."
Base.@kwdef struct Experiment
    counterfactual_data::CounterfactualData
    test_data::CounterfactualData
    dataname::String = "dataset"
    save_name::String = replace(lowercase(dataname), " " => "_")
    output_path::String = DEFAULT_OUTPUT_PATH
    params_path::String = joinpath(output_path, "params")
    use_pretrained::Bool = !RETRAIN
    models::Union{Nothing,Dict} = nothing
    builder::Union{Nothing,MLJFlux.Builder} = nothing
    𝒟x::Distribution = Normal()
    sampling_batch_size::Int = 50
    sampling_steps::Int = 50
    min_batch_size::Int = 128
    epochs::Int = 100
    n_hidden::Int = 32
    activation::Function = Flux.relu
    α::AbstractArray = [1.0, 1.0, 1e-1]
    n_ens::Int = 5
    use_ensembling::Bool = true
    coverage::Float64 = DEFAULT_COVERAGE
    generators::Union{Nothing,Dict} = nothing
    n_individuals::Int = 25
    ce_measures::AbstractArray = CE_MEASURES
    model_measures::Dict = MODEL_MEASURES
    use_class_loss::Bool = false
    use_variants::Bool = true
    Λ::AbstractArray = [0.25, 0.75, 0.75]
    Λ_Δ::AbstractArray = Λ
    opt::Flux.Optimise.AbstractOptimiser = Flux.Optimise.Descent(0.01)
    parallelizer::Union{Nothing, AbstractParallelizer} = PLZ
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
    train_models!(outcome::ExperimentOutcome, exp::Experiment)

Train the models specified by `exp` and store them in `outcome`.
"""
function train_models!(outcome::ExperimentOutcome, exp::Experiment)
    model_dict = prepare_models(exp)
    outcome.model_dict = model_dict
    meta_model_performance(outcome)
end

"""
    benchmark!(outcome::ExperimentOutcome, exp::Experiment)

Benchmark the models specified by `exp` and store the results in `outcome`.
"""
function benchmark!(outcome::ExperimentOutcome, exp::Experiment)
    bmk, generator_dict = run_benchmark(exp, outcome.model_dict)
    outcome.bmk = bmk
    outcome.generator_dict = generator_dict
end

"""
    run_experiment(exp::Experiment)

Run the experiment specified by `exp`.
"""
function run_experiment(exp::Experiment; save_output::Bool=true, only_models::Bool=false)
    
    # Setup
    @info "All results will be saved to $(exp.output_path)."
    isdir(exp.output_path) || mkdir(exp.output_path)
    @info "All parameter choices will be saved to $(exp.params_path)."
    isdir(exp.params_path) || mkdir(exp.params_path)
    outcome = ExperimentOutcome(exp, nothing, nothing, nothing)

    # Models
    train_models!(outcome, exp)

    # Return if only models are needed:
    !only_models || return outcome

    # Benchmark
    benchmark!(outcome, exp)

    # Save data:
    if save_output
        Serialization.serialize(joinpath(exp.output_path, "$(exp.save_name)_outcome.jls"), outcome)
        Serialization.serialize(joinpath(exp.output_path, "$(exp.save_name)_bmk.jls"), outcome.bmk)
        meta(outcome; save_output=true)
    end

    return outcome

end

"""
    run_experiment(counterfactual_data::CounterfactualData, test_data::CounterfactualData; kwargs...)

Overload the `run_experiment` function to allow for passing in `CounterfactualData` objects and other keyword arguments.
"""
function run_experiment(counterfactual_data::CounterfactualData, test_data::CounterfactualData; kwargs...)
    # Parameters:
    exp = Experiment(;
        counterfactual_data=counterfactual_data,
        test_data=test_data,
        kwargs...
    )
    return run_experiment(exp)
end

# Pre-trained models:
function pretrained_path(exp::Experiment)
    if isfile(joinpath(exp.output_path, "$(exp.save_name)_models.jls"))
        @info "Found local pre-trained models in $(exp.output_path) and using those."
        return exp.output_path
    else
        @info "Using artifacts. Models were pre-trained on `julia-$(LATEST_VERSION)` and may not work on other versions."
        return joinpath(LATEST_ARTIFACT_PATH, "results")
    end
end