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

Random.seed!(2023)

# Parallelization:
plz = nothing

if "threaded" ∈ ARGS
    @info "Multi-threading using $(Threads.nthreads()) threads."
    const USE_THREADS = true
    plz = ThreadsParallelizer()
else
    const USE_THREADS = false
end

if "mpi" ∈ ARGS
    @info "Multi-processing using MPI."
    import MPI
    MPI.Init()
    const USE_MPI = true
    plz = MPIParallelizer(MPI.COMM_WORLD, threaded=USE_THREADS)
    if MPI.Comm_rank(MPI.COMM_WORLD) != 0
        @info "Disabling logging on non-root processes."
        global_logger(NullLogger())
    end
else
    const USE_MPI = false
end

const PLZ = plz

# Constants:
const LATEST_VERSION = "1.8.5"
const ARTIFACT_NAME = "results-paper-submission-$(LATEST_VERSION)"
artifact_toml = LazyArtifacts.find_artifacts_toml(".")
_hash = artifact_hash(ARTIFACT_NAME, artifact_toml)
const LATEST_ARTIFACT_PATH = joinpath(artifact_path(_hash), ARTIFACT_NAME)

if any(contains.(ARGS, "output_path"))
    @assert sum(contains.(ARGS, "output_path")) == 1 "Only one output path can be specified."
    _path = ARGS[findall(contains.(ARGS, "output_path"))][1] |> x -> replace(x, "output_path=" => "")
else
    timestamp = Dates.format(now(), "yyyy-mm-dd@HH:MM")
    _path = "$(pwd())/results_$(timestamp)"
end

"Default output path."
const DEFAULT_OUTPUT_PATH = _path

ispath(DEFAULT_OUTPUT_PATH) || mkpath(DEFAULT_OUTPUT_PATH)

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