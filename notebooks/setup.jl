setup_notebooks = quote

    using Pkg
    Pkg.activate("notebooks")

    using AlgebraOfGraphics
    using AlgebraOfGraphics: Violin, BoxPlot
    using CairoMakie
    using CCE
    using CCE: set_size_penalty, distance_from_energy, distance_from_targets
    using Chain: @chain
    using ConformalPrediction
    using CounterfactualExplanations
    using CounterfactualExplanations.Data
    using CounterfactualExplanations.Models: load_mnist_mlp
    using CounterfactualExplanations.Objectives
    using Distributions
    using Flux
    using Images
    using JointEnergyModels
    using LinearAlgebra
    using MLDatasets
    using MLDatasets: convert2image
    using MLJBase
    using MLJFlux
    using MLUtils
    using Plots
    using Random
    using Serialization
    using Tidier

    # Setup:
    Plots.theme(:wong)
    Random.seed!(2023)
    www_path = "www"
    output_path = "artifacts"
    img_height = 300

end;