n_obs = Int(2500 / (1.0 - TEST_SIZE))
counterfactual_data, test_data = train_test_split(load_moons(n_obs); TEST_SIZE=TEST_SIZE)
run_experiment!(
    counterfactual_data, test_data; dataname="Moons",
    epochs=500,
    n_hidden=32,
    activation = Flux.relu,
    α=[1.0, 1.0, 1e-1],
    sampling_batch_size=10,
    sampling_steps=30,
    λ₁=0.25,
    λ₂=0.75,
    λ₃=0.75,
    opt=Flux.Optimise.Descent(0.05),
    use_class_loss=false
)