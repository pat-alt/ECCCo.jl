n_obs = Int(2500 / (1.0 - TEST_SIZE))
counterfactual_data, test_data = train_test_split(load_moons(n_obs); test_size=TEST_SIZE)
run_experiment(
    counterfactual_data, test_data; 
    dataname="Moons",
    epochs=500,
    n_hidden=32,
    activation = Flux.relu,
    sampling_batch_size=10,
    sampling_steps=30,
    Λ=[0.25, 0.75, 0.75],
    opt=Flux.Optimise.Descent(0.05),
    nsamples=10,
    nmin=10
)