# Data:
dataname = "California Housing"
counterfactual_data, test_data = train_test_split(load_california_housing(nothing); test_size=TEST_SIZE)

# Model tuning:
model_tuning_params = DEFAULT_MODEL_TUNING_LARGE

# Tuning parameters:
tuning_params = DEFAULT_GENERATOR_TUNING

# Parameter choices:
params = (
    n_hidden=32,
    activation=Flux.relu,
    builder=MLJFlux.@builder Flux.Chain(
        Dense(n_in, n_hidden, activation),
        Dense(n_hidden, n_hidden, activation),
        Dense(n_hidden, n_out),
    ),
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