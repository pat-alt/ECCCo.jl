using Pkg
Pkg.activate(@__DIR__)

# Deps:
using CounterfactualExplanations
using CounterfactualExplanations.Data
using CounterfactualExplanations.DataPreprocessing: train_test_split
using CounterfactualExplanations.Evaluation: benchmark, evaluate, Benchmark
using CounterfactualExplanations.Generators: JSMADescent
using CounterfactualExplanations.Models: load_mnist_mlp, load_fashion_mnist_mlp, train, probs
using CounterfactualExplanations.Objectives
using CSV
using Distributions
using ECCCo
using JointEnergyModels
using LazyArtifacts
using MLJBase: multiclass_f1score, accuracy, multiclass_precision
using MLJEnsembles
using MLJFlux

# Constants:
const LATEST_VERSION = "1.8.5"
const ARTIFACT_NAME = "results-paper-submission-$(LATEST_VERSION)"
artifact_toml = LazyArtifacts.find_artifacts_toml(".")
_hash = artifact_hash(ARTIFACT_NAME, artifact_toml)
const LATEST_ARTIFACT_PATH = artifact_path(_hash)

# Pre-trained models:
function pretrained_path()
    @info "Models were pre-trained on `julia-$(LATEST_VERSION)` and may not work on other versions."
    return LATEST_ARTIFACT_PATH
end

"Default output path."
const DEFAULT_OUTPUT_PATH = "$(pwd())/results"

"Boolean flag to retrain models."
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

"Test set proportion."
const TEST_SIZE = 0.2