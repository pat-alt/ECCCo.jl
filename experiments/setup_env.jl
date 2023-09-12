# Deps:
using Chain: @chain
using ConformalPrediction
using CounterfactualExplanations
using CounterfactualExplanations.Data
using CounterfactualExplanations.DataPreprocessing: train_test_split
using CounterfactualExplanations.Evaluation: benchmark, evaluate, Benchmark
using CounterfactualExplanations.Generators: JSMADescent
using CounterfactualExplanations.Models: load_mnist_mlp, load_fashion_mnist_mlp, train, probs
using CounterfactualExplanations.Objectives
using CounterfactualExplanations.Parallelization
using CSV
using Dates
using DataFrames
using Distributions: Normal, Distribution, Categorical, Uniform
using ECCCo
using Flux
using JointEnergyModels
using LazyArtifacts
using Logging
using MLJBase: multiclass_f1score, accuracy, multiclass_precision, table, machine, fit!
using MLJEnsembles
using MLJFlux
using Random
using Serialization
using TidierData

import MPI

Random.seed!(2023)

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"              # avoid command prompt and just download data

# Scripts:
include("experiment.jl")
include("data/data.jl")
include("models/models.jl")
include("benchmarking/benchmarking.jl")
include("post_processing/post_processing.jl")
include("utils.jl")

# Parallelization:
plz = nothing

if "threaded" ∈ ARGS
    const USE_THREADS = true
    plz = ThreadsParallelizer()
else
    const USE_THREADS = false
end

if "mpi" ∈ ARGS
    MPI.Init()
    const USE_MPI = true
    plz = MPIParallelizer(MPI.COMM_WORLD; threaded=USE_THREADS)
    if MPI.Comm_rank(MPI.COMM_WORLD) != 0
        global_logger(NullLogger())
    else
        @info "Multi-processing using MPI. Disabling logging on non-root processes."
        if USE_THREADS
            @info "Multi-threading using $(Threads.nthreads()) threads."
        end
    end
else
    const USE_MPI = false
end

const PLZ = plz

# Constants:
const LATEST_VERSION = "1.8.5"
const ARTIFACT_NAME = "results-paper-submission-$(LATEST_VERSION)"
const ARTIFACT_TOML = LazyArtifacts.find_artifacts_toml(".")
const ARTIFACT_HASH = artifact_hash(ARTIFACT_NAME, ARTIFACT_TOML)
const LATEST_ARTIFACT_PATH = joinpath(artifact_path(ARTIFACT_HASH), ARTIFACT_NAME)

time_stamped = false
if any(contains.(ARGS, "output_path"))
    @assert sum(contains.(ARGS, "output_path")) == 1 "Only one output path can be specified."
    _path = ARGS[findall(contains.(ARGS, "output_path"))][1] |> x -> replace(x, "output_path=" => "")
elseif isinteractive()
    @info "You are running experiments interactively. By default, results will be saved in a temporary directory."
    _path = tempdir()
else
    timestamp = Dates.format(now(), "yyyy-mm-dd@HH:MM")
    time_stamped = true
    _path = "$(pwd())/results_$(timestamp)"
end

"Default output path."
const DEFAULT_OUTPUT_PATH = _path
const TIME_STAMPED = time_stamped

"Boolean flag to only train models."
const ONLY_MODELS = "only_models" ∈ ARGS ? true : false

"Boolean flag to retrain models."
const RETRAIN = "retrain" ∈ ARGS || ONLY_MODELS ? true : false

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

"Test set proportion."
const TEST_SIZE = 0.2

const UPLOAD = "upload" ∈ ARGS

n_ind_specified = false
if any(contains.(ARGS, "n_individuals="))
    n_ind_specified = true
    n_individuals = ARGS[findall(contains.(ARGS, "n_individuals="))][1] |> x -> replace(x, "n_individuals=" => "") |> x -> parse(Int, x)
else
    n_individuals = 100
end

"Number of individuals to use in benchmarking."
const N_IND = n_individuals

"Boolean flag to check if number of individuals was specified."
const N_IND_SPECIFIED = n_ind_specified