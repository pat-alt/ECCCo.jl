n_obs = Int(1000 / (1.0 - TEST_SIZE))
counterfactual_data, test_data = train_test_split(load_circles(n_obs; noise=0.05, factor=0.5); test_size=TEST_SIZE)
run_experiment(
    counterfactual_data, test_data; dataname="Circles",
    n_hidden=32,
    α=[1.0, 1.0, 1e-2],
    sampling_batch_size=nothing,
    sampling_steps=20,
    λ₁=0.25,
    λ₂ = 0.75,
    λ₃ = 0.75,
    opt=Flux.Optimise.Descent(0.01),
    use_class_loss = false,
)