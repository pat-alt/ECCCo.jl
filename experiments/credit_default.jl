counterfactual_data, test_data = train_test_split(load_credit_default(nothing); test_size=TEST_SIZE)

# Default builder:
n_hidden = 32
activation = Flux.relu
builder = MLJFlux.@builder Flux.Chain(
    Dense(n_in, n_hidden, activation),
    Dense(n_hidden, n_hidden, activation),
    Dense(n_hidden, n_out),
)

# Number of individuals:
n_ind = N_IND_SPECIFIED ? N_IND : 100

run_experiment(
    counterfactual_data, test_data;
    dataname="Credit Default",
    builder=builder,
    α=[1.0, 1.0, 1e-1],
    sampling_batch_size=10,
    sampling_steps=30,
    use_ensembling=true,
    opt=Flux.Optimise.Descent(0.05),
    n_individuals=n_ind,
    use_variants=true,
    Λ=[0.1, 0.2, 0.2],
    nsamples=100,
    niter_eccco=100
)