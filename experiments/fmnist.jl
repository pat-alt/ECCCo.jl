# Training data:
dataname = "Fashion MNIST"
n_obs = 10000
counterfactual_data = load_fashion_mnist(n_obs)
counterfactual_data.X = ECCCo.pre_process.(counterfactual_data.X)
# Adjust domain constraints to account for noise added during pre-processing:
counterfactual_data.domain = fill((minimum(counterfactual_data.X), maximum(counterfactual_data.X)), size(counterfactual_data.X, 1))

# VAE (trained on full dataset):
using CounterfactualExplanations.Models: load_fashion_mnist_vae
vae = load_fashion_mnist_vae()
counterfactual_data.generative_model = vae

# Test data:
test_data = load_fashion_mnist_test()

# Dimensionality reduction:
maxout_dim = vae.params.latent_dim
counterfactual_data.dt = MultivariateStats.fit(MultivariateStats.PCA, counterfactual_data.X; maxoutdim=maxout_dim);

# Model tuning:
model_tuning_params = DEFAULT_MODEL_TUNING_LARGE

# Tuning parameters:
tuning_params = DEFAULT_GENERATOR_TUNING
tuning_params = (; tuning_params..., Λ=[tuning_params.Λ[2:end]..., [0.1, 0.1, 3.0]])

# Additional models:
add_models = Dict(
    "LeNet-5" => lenet5,
)

# CE measures (add cosine distance):
ce_measures = [CE_MEASURES..., ECCCo.distance_from_energy_ssim, ECCCo.distance_from_targets_ssim]

# Parameter choices:
params = (
    n_individuals=N_IND_SPECIFIED ? N_IND : 2,
    builder=default_builder(n_hidden=128, n_layers=1, activation=Flux.swish),
    𝒟x=Uniform(-1.0, 1.0),
    α=[1.0, 1.0, 1e-2],
    sampling_batch_size=10,
    sampling_steps=25,
    use_ensembling=true,
    use_variants=false,
    additional_models=add_models,
    epochs=100,
    nsamples=10,
    nmin=1,
    niter_eccco=10,
    Λ=[0.01, 0.25, 0.25],
    Λ_Δ=[0.01, 0.1, 0.3],
    opt=Flux.Optimise.Descent(0.1),
    reg_strength=0.0,
    ce_measures=ce_measures,
    dim_reduction=true,
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
        tuning_params=tuning_params,
        n_individuals=5
    )
end