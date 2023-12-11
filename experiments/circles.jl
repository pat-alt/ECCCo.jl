# Data:
dataname = "Circles"
n_obs = Int(1000 / (1.0 - TEST_SIZE))
counterfactual_data, test_data =
    train_test_split(load_circles(n_obs; noise = 0.05, factor = 0.5); test_size = TEST_SIZE)

# Domain constraints:
counterfactual_data.domain = extrema(counterfactual_data.X, dims=2)

# Model tuning:
model_tuning_params = DEFAULT_MODEL_TUNING_SMALL

# Tuning parameters:
tuning_params = DEFAULT_GENERATOR_TUNING

# Parameter choices:
# These are the parameter choices originally used in the paper that were manually fine-tuned for the JEM.
params = (
    use_tuned = false,
    n_hidden = 32,
    n_layers = 3,
    activation = Flux.swish,
    epochs = 100,
    α = [1.0, 1.0, 1e-2],
    sampling_steps = 30,
    opt = Flux.Optimise.Descent(0.05),
    Λ = [0.1, 0.1, 0.05],
    reg_strength = 1.0,
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
