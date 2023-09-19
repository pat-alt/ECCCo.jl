# Data:
dataname = "GMSC"
counterfactual_data, test_data = train_test_split(load_gmsc(nothing); test_size=TEST_SIZE)
nobs = size(counterfactual_data.X, 2)

# Model tuning:
model_tuning_params = DEFAULT_MODEL_TUNING_LARGE

# Tuning parameters:
tuning_params = DEFAULT_GENERATOR_TUNING_LARGE

# Parameter choices:
params = (
    n_hidden=32,
    activation=Flux.relu,
    builder=default_builder(n_hidden=32, n_layers=3, activation=Flux.relu),
    α = [1.0, 1.0, 1e-1],
    sampling_batch_size = 10,
    sampling_steps = 30,
    use_ensembling = true,
    opt = Flux.Optimise.Descent(0.05)
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
