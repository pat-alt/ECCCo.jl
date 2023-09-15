# Data:
dataname = "Linearly Separable"
n_obs = Int(1000 / (1.0 - TEST_SIZE))
counterfactual_data, test_data = train_test_split(
    load_blobs(n_obs; cluster_std=0.1, center_box=(-1.0 => 1.0));
    test_size=TEST_SIZE
)

# Model tuning:
model_tuning_params = DEFAULT_MODEL_TUNING_SMALL

# Tuning parameters:
tuning_params = DEFAULT_GENERATOR_TUNING

# Parameter choices:
params = (
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
        tuning_params=tuning_params,
    )
end