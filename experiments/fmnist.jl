# Training data:
dataname = "Fashion MNIST"
n_obs = 10000
counterfactual_data = load_fashion_mnist(n_obs)
counterfactual_data.X = ECCCo.pre_process.(counterfactual_data.X)

# VAE (trained on full dataset):
using CounterfactualExplanations.Models: load_fashion_mnist_vae
vae = load_fashion_mnist_vae()
counterfactual_data.generative_model = vae

# Test data:
test_data = load_fashion_mnist_test()

# Model tuning:
model_tuning_params = DEFAULT_MODEL_TUNING_LARGE

# Tuning parameters:
tuning_params = DEFAULT_GENERATOR_TUNING
push!(tuning_params.Λ, [0.1, 0.1, 3.0])

# Additional models:
add_models = Dict(
    "LeNet-5" => lenet5,
    # "ResNet-18" => resnet18(; epochs=10),
)

# Parameter choices:
params = (
    n_individuals=N_IND_SPECIFIED ? N_IND : 10,
    builder=default_builder(n_hidden=128, n_layers=2, activation=Flux.swish),
    𝒟x=Uniform(-1.0, 1.0),
    α=[1.0, 1.0, 1e-2],
    sampling_batch_size=10,
    sampling_steps=50,
    use_ensembling=true,
    use_variants=false,
    additional_models=add_models,
    epochs=10,
    nsamples=10,
    nmin=1,
    niter_eccco=100
)

if !GRID_SEARCH
    run_experiment(
        counterfactual_data, test_data;
        dataname=dataname,
        params...
    )
else
    grid_search(
        counterfactual_data, test_data;
        dataname=dataname,
        tuning_params=tuning_params
    )
end