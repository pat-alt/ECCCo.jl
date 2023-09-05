# Training data:
n_obs = 10000
counterfactual_data = load_mnist(n_obs)
counterfactual_data.X = ECCCo.pre_process.(counterfactual_data.X)

# VAE (trained on full dataset):
using CounterfactualExplanations.Models: load_mnist_vae
vae = load_mnist_vae()
counterfactual_data.generative_model = vae

# Test data:
test_data = load_mnist_test()

# Additional models:
add_models = Dict(
    :lenet5 => lenet5,
)

# Default builder:
n_hidden = 128
activation = Flux.swish
builder = MLJFlux.@builder Flux.Chain(
    Dense(n_in, n_hidden, activation),
    Dense(n_hidden, n_out),
)

# Run:
run_experiment(
    counterfactual_data, test_data; 
    dataname="MNIST",
    builder= builder,
    𝒟x = Uniform(-1.0, 1.0),
    α = [1.0,1.0,1e-2],
    sampling_batch_size = 10,
    sampling_steps=50,
    use_ensembling = true,
    n_individuals = 5,
    nsamples = 10,
    nmin = 10,
    use_variants = false,
    use_class_loss = true,
    additional_models=add_models,
    epochs = 10,
)