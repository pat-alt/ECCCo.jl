# Data: 
dataname = "German Credit"
counterfactual_data, test_data =
    train_test_split(load_german_credit(nothing); test_size = TEST_SIZE)

# Domain constraints:
counterfactual_data.domain = extrema(counterfactual_data.X, dims=2)

# VAE:
using CounterfactualExplanations.GenerativeModels: VAE, train!
X = counterfactual_data.X
y = counterfactual_data.output_encoder.y
vae = VAE(size(X, 1); nll = Flux.Losses.mse, epochs = 100, λ = 0.01, latent_dim = 5)
train!(vae, X, y)
counterfactual_data.generative_model = vae

# Dimensionality reduction:
maxout_dim = vae.params.latent_dim
counterfactual_data.dt = MultivariateStats.fit(
    MultivariateStats.PCA,
    counterfactual_data.X;
    maxoutdim = maxout_dim,
);

# Model tuning:
model_tuning_params = DEFAULT_MODEL_TUNING_LARGE

# Tuning parameters:
tuning_params = DEFAULT_GENERATOR_TUNING_LARGE

# Parameter choices:
params = (
    n_hidden = 32,
    activation = Flux.relu,
    builder = default_builder(n_hidden = 32, n_layers = 3, activation = Flux.relu),
    α = [1.0, 1.0, 1e-1],
    sampling_batch_size = 10,
    sampling_steps = 30,
    use_ensembling = true,
    opt = Flux.Optimise.Descent(0.05),
    Λ = [0.2, 0.2, 0.2],
    reg_strength = 0.5,
    dim_reduction = true,
)

# Best grid search params:
params = append_best_params(params, dataname)

if GRID_SEARCH
    grid_search(
        counterfactual_data,
        test_data;
        dataname = dataname,
        tuning_params = tuning_params,
        params...,
    )
elseif FROM_GRID_SEARCH
    outcomes_file_path = joinpath(
        DEFAULT_OUTPUT_PATH,
        "grid_search",
        "$(replace(lowercase(dataname), " " => "_")).jls",
    )
    save_best(outcomes_file_path)
    bmk2csv(dataname)
else
    run_experiment(
        counterfactual_data,
        test_data;
        dataname = dataname,
        model_tuning_params = model_tuning_params,
        params...,
    )
end
