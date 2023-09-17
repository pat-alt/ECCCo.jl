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
    additional_models::Union{Nothing,Dict} = nothing
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
    n_individuals::Int = N_IND
    ce_measures::AbstractArray = CE_MEASURES
    model_measures::Dict = MODEL_MEASURES
    use_class_loss::Bool = false
    use_variants::Bool = true
    Λ::AbstractArray = [0.25, 0.75, 0.75]
    Λ_Δ::AbstractArray = Λ
    opt::Flux.Optimise.AbstractOptimiser = Flux.Optimise.Descent(0.01)
    parallelizer::Union{Nothing, AbstractParallelizer} = PLZ
    nsamples::Union{Nothing,Int} = nothing
    nmin::Union{Nothing,Int} = nothing
    finaliser::Function = Flux.softmax
    loss::Function = Flux.Losses.crossentropy
    train_parallel::Bool = false
    reg_strength::Real = 0.1
    niter_eccco::Union{Nothing,Int} = nothing
    model_tuning_params::NamedTuple = DEFAULT_MODEL_TUNING_SMALL
end

"A container to hold the results of an experiment."
mutable struct ExperimentOutcome
    exper::Experiment
    model_dict::Union{Nothing, Dict}
    generator_dict::Union{Nothing, Dict}
    bmk::Union{Nothing, Benchmark}
end

"""
    train_models!(outcome::ExperimentOutcome, exper::Experiment)

Train the models specified by `exper` and store them in `outcome`.
"""
function train_models!(outcome::ExperimentOutcome, exper::Experiment; save_meta::Bool=false)
    model_dict = prepare_models(exper)
    outcome.model_dict = model_dict
    if !(is_multi_processed(exper) && MPI.Comm_rank(exper.parallelizer.comm) != 0)
        meta_model_performance(outcome; save_output=save_meta)
    end
end

"""
    benchmark!(outcome::ExperimentOutcome, exper::Experiment)

Benchmark the models specified by `exper` and store the results in `outcome`.
"""
function benchmark!(outcome::ExperimentOutcome, exper::Experiment)
    bmk, generator_dict = run_benchmark(exper, outcome.model_dict)
    outcome.bmk = bmk
    outcome.generator_dict = generator_dict
end

"""
    run_experiment(exper::Experiment)

Run the experiment specified by `exper`.
"""
function run_experiment(exper::Experiment; save_output::Bool=true, only_models::Bool=ONLY_MODELS)
    
    # Setup
    if save_output && !(is_multi_processed(exper) && MPI.Comm_rank(exper.parallelizer.comm) != 0)
        @info "All results will be saved to $(exper.output_path)."
        isdir(exper.output_path) || mkdir(exper.output_path)
        @info "All parameter choices will be saved to $(exper.params_path)."
        isdir(exper.params_path) || mkdir(exper.params_path)
    end
    outcome = ExperimentOutcome(exper, nothing, nothing, nothing)

    # Model tuning:
    if TUNE_MODEL
        mach = tune_model(exper)
        return mach
    end

    # Model training:
    if only_models
        train_models!(outcome, exper; save_meta=true)
        return outcome
    else
        train_models!(outcome, exper)
    end

    # Benchmark:
    benchmark!(outcome, exper)
    if is_multi_processed(exper)
        MPI.Barrier(exper.parallelizer.comm)
    end

    # Save data:
    if save_output && !(is_multi_processed(exper) && MPI.Comm_rank(exper.parallelizer.comm) != 0)
        Serialization.serialize(joinpath(exper.output_path, "$(exper.save_name)_outcome.jls"), outcome)
        Serialization.serialize(joinpath(exper.output_path, "$(exper.save_name)_bmk.jls"), outcome.bmk)
        meta(outcome; save_output=true)
    end

    return outcome

end

"""
    run_experiment(counterfactual_data::CounterfactualData, test_data::CounterfactualData; kwargs...)

Overload the `run_experiment` function to allow for passing in `CounterfactualData` objects and other keyword arguments.
"""
function run_experiment(counterfactual_data::CounterfactualData, test_data::CounterfactualData; save_output::Bool=true, kwargs...)
    # Parameters:
    exper = Experiment(;
        counterfactual_data=counterfactual_data,
        test_data=test_data,
        kwargs...
    )
    return run_experiment(exper; save_output=save_output)
end

# Pre-trained models:
function pretrained_path(exper::Experiment)
    if isfile(joinpath(DEFAULT_OUTPUT_PATH, "$(exper.save_name)_models.jls"))
        @info "Found local pre-trained models in $(DEFAULT_OUTPUT_PATH) and using those."
        return DEFAULT_OUTPUT_PATH
    else
        @info "Using artifacts. Models were pre-trained on `julia-$(LATEST_VERSION)` and may not work on other versions."
        Pkg.Artifacts.download_artifact(ARTIFACT_HASH, ARTIFACT_TOML)
        return joinpath(LATEST_ARTIFACT_PATH, "results")
    end
end