"""
    default_builder(n_hidden::Int=16, activation::Function=Flux.swish)

Default builder for MLPs.
"""
function default_builder(n_hidden::Int=16, activation::Function=Flux.swish)
    builder = MLJFlux.MLP(
        hidden=(n_hidden, n_hidden, n_hidden),
        σ=activation
    )
    return builder
end

"""
    default_models(
        builder::GenericBuilder=default_builder(),
        epochs::Int=25,
        batch_size::Int=128,
        finaliser::Function=Flux.softmax,
        loss::Function=Flux.Losses.crossentropy,
        sampler::AbstractSampler,
        α::Float64,
        verbosity::Int=10,
        sampling_steps::Int=30,
        n_ens::Int=5,
        use_ensembling::Bool=true,
    )

Builds a dictionary of default models for training.
"""
function default_models(;
    sampler::AbstractSampler,
    builder::MLJFlux.Builder=default_builder(),
    epochs::Int=100,
    batch_size::Int=128,
    finaliser::Function=Flux.softmax,
    loss::Function=Flux.Losses.crossentropy,
    α::AbstractArray=[1.0, 1.0, 1e-1],
    verbosity::Int=10,
    sampling_steps::Int=30,
    n_ens::Int=5,
    use_ensembling::Bool=true,
)

    # Simple MLP:
    mlp = NeuralNetworkClassifier(
        builder=builder,
        epochs=epochs,
        batch_size=batch_size,
        finaliser=finaliser,
        loss=loss,
    )

    # Deep Ensemble:
    mlp_ens = EnsembleModel(model=mlp, n=n_ens)

    # Joint Energy Model:
    jem = JointEnergyClassifier(
        sampler;
        builder=builder,
        epochs=epochs,
        batch_size=batch_size,
        finaliser=finaliser,
        loss=loss,
        jem_training_params=(
            α=α, verbosity=verbosity,
        ),
        sampling_steps=sampling_steps
    )

    # Deep Ensemble of Joint Energy Models:
    jem_ens = EnsembleModel(model=jem, n=n_ens)

    # Dictionary of models:
    if !use_ensembling
        models = Dict(
            "MLP" => mlp,
            "JEM" => jem,
        )
    else
        models = Dict(
            "MLP" => mlp,
            "MLP Ensemble" => mlp_ens,
            "JEM" => jem,
            "JEM Ensemble" => jem_ens,
        )
    end

    return models
end