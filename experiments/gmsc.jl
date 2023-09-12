counterfactual_data, test_data = train_test_split(load_gmsc(nothing); test_size=TEST_SIZE)
nobs = size(counterfactual_data.X, 2)

# Default builder:
n_hidden = 32
activation = Flux.relu
builder = MLJFlux.@builder Flux.Chain(
    Dense(n_in, n_hidden, activation),
    Dense(n_hidden, n_hidden, activation),
    Dense(n_hidden, n_out),
)

# Number of individuals:
n_ind = N_IND_SPECIFIED ? N_IND : 10

run_experiment(
    counterfactual_data, test_data; 
    dataname="GMSC",
    builder = builder,
    α=[1.0, 1.0, 1e-1],
    sampling_batch_size=10,
    sampling_steps = 30,
    use_ensembling = true,
    Λ=[0.1, 0.5, 0.5],
    opt = Flux.Optimise.Descent(0.05),
    n_individuals = n_ind,
    use_variants = false, 
    min_batch_size = 250,
)