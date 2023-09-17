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
    for tuning_params in grid
        outcome = run_experiment(
            counterfactual_data, test_data;
            save_output=false,
            dataname=dataname,
            output_path=grid_search_path,
            tuning_params...,
            kwargs...,
        )
        outcomes[tuning_params] = outcome
    end

    # Save:
    Serialization.serialize(joinpath(grid_search_path, "$(replace(lowercase(dataname), " " => "_")).jls"), outcomes)
end