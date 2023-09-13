n_obs = Int(1000 / (1.0 - TEST_SIZE))
counterfactual_data, test_data = train_test_split(
    load_blobs(n_obs; cluster_std=0.1, center_box=(-1.0 => 1.0));
    test_size=TEST_SIZE
)
run_experiment(
    counterfactual_data, test_data; 
    dataname="Linearly Separable",
    nsamples=100,
    nmin=1,
    niter_eccco=30
)