setup_notebooks = quote

    using Pkg
    Pkg.activate("notebooks")

    using AlgebraOfGraphics
    using AlgebraOfGraphics: Violin, BoxPlot, BarPlot
    using CairoMakie
    using ECCCo
    using ECCCo: set_size_penalty, distance_from_energy, distance_from_targets
    using Chain: @chain
    using ConformalPrediction
    using CounterfactualExplanations
    using CounterfactualExplanations.Data
    using CounterfactualExplanations.DataPreprocessing: train_test_split
    using CounterfactualExplanations.Evaluation: benchmark, evaluate
    using CounterfactualExplanations.Generators: JSMADescent
    using CounterfactualExplanations.Models: load_mnist_mlp, load_fashion_mnist_mlp, train, probs
    using CounterfactualExplanations.Objectives
    using CSV
    using DataFrames
    using DataStructures
    using Distributions
    using Flux
    using Images
    using JointEnergyModels
    using LaplaceRedux: LaplaceApproximation
    using LinearAlgebra
    using Markdown
    using MLDatasets
    using MLDatasets: convert2image
    using MLJBase
    using MLJEnsembles
    using MLJFlux
    using MLUtils
    using Plots
    using Random
    using Serialization
    using StatsPlots
    using Tidier

    # Setup:
    Plots.theme(:wong)
    Random.seed!(2023)
    www_path = "www"
    params_path = "artifacts/params"
    output_path = "artifacts/results"
    output_images_path = "artifacts/results/images"
    img_height = 300
    panel_height = 250

end;