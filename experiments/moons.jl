# Data:
dataname = "Moons"
n_obs = Int(2500 / (1.0 - TEST_SIZE))
counterfactual_data, test_data = train_test_split(load_moons(n_obs); test_size=TEST_SIZE)

# Model tuning:
model_tuning_params = DEFAULT_MODEL_TUNING_SMALL

# Tuning parameters:
tuning_params = DEFAULT_GENERATOR_TUNING

# Parameter choices:
params = (
    epochs=500,
    n_hidden=32,
    activation = Flux.relu,
    sampling_batch_size=10,
    sampling_steps=30,
    opt=Flux.Optimise.Descent(0.05),
    α=[1.0, 1.0, 1e-1],
    nsamples=100,
    niter_eccco=100,
    Λ=[0.1, 0.2, 0.2],
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