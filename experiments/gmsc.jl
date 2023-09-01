counterfactual_data, test_data = train_test_split(load_gmsc(nothing); test_size=TEST_SIZE)
run_experiment(
    counterfactual_data, test_data; dataname="GMSC",
    n_hidden=128,
    activation = Flux.swish,
    builder = MLJFlux.@builder Flux.Chain(
        Dense(n_in, n_hidden, activation),
        Dense(n_hidden, n_hidden, activation),
        Dense(n_hidden, n_out),
    ),
    α=[1.0, 1.0, 1e-1],
    sampling_batch_size=nothing,
    sampling_steps = 30,
    use_ensembling = true,
    Λ=[0.1, 0.5, 0.5],
    opt = Flux.Optimise.Descent(0.05),
)