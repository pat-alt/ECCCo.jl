# Data:
dataname = "Moons"
n_obs = Int(2500 / (1.0 - TEST_SIZE))
counterfactual_data, test_data = train_test_split(load_moons(n_obs); test_size = TEST_SIZE)

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
    activation = Flux.relu,
    epochs = 500,
    sampling_batch_size = 10,
    sampling_steps = 30,
    opt = Flux.Optimise.Descent(0.01),
    Λ = [0.1, 0.1, 0.5],
    reg_strength = 0.0,
)

# Best grid search params:
append_best_params!(params, dataname)

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
