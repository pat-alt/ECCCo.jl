"""
    grid_search(
        couterfactual_data::CounterfactualData,
        test_data::CounerfactualData;
        dataname::String,
        tuning_params::NamedTuple,
        kwargs...,
    )

Perform a grid search over the hyperparameters specified by `tuning_params`. Experiments will be run for each combination of hyperparameters. Other keyword arguments are passed to `run_experiment` and fixed for all experiments.
"""
function grid_search(
    couterfactual_data::CounterfactualData,
    test_data::CounterfactualData;
    dataname::String,
    n_individuals::Int=N_IND,
    tuning_params::NamedTuple,
    kwargs...,
)

    # Output path:
    grid_search_path = mkpath(joinpath(DEFAULT_OUTPUT_PATH, "grid_search"))

    # Grid setup:
    tuning_params = [Pair.(k, vals) for (k, vals) in pairs(tuning_params)]
    grid = Iterators.product(tuning_params...)
    outcomes = Dict{Any,Any}()
    
    # Search:
    counter = 1
    for tuning_params in grid
        @info "Running experiment $(counter)/$(length(grid)) with tuning parameters: $(tuning_params)"
        outcome = run_experiment(
            counterfactual_data, test_data;
            save_output=false,
            dataname=dataname,
            n_individuals=n_individuals,
            output_path=grid_search_path,
            tuning_params...,
            kwargs...,
        )
        outcomes[tuning_params] = outcome
        counter += 1
    end

    # Save:
    if !(is_multi_processed(exper) && MPI.Comm_rank(exper.parallelizer.comm) != 0)
        Serialization.serialize(joinpath(grid_search_path, "$(replace(lowercase(dataname), " " => "_")).jls"), outcomes)
    end
end