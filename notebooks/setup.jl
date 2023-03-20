setup_notebooks = quote

    using Pkg
    Pkg.activate("notebooks")

    using CCE
    using CCE: set_size_penalty, distance_from_energy, distance_from_targets
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
    using Plots
    using Random

    # Setup:
    theme(:wong)
    Random.seed!(2023)
    www_path = "notebooks/www"
    img_height = 300

end;