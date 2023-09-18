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
    if !(is_multi_processed(PLZ) && MPI.Comm_rank(PLZ.comm) != 0)
        Serialization.serialize(joinpath(grid_search_path, "$(replace(lowercase(dataname), " " => "_")).jls"), outcomes)
        Serialization.serialize(joinpath(grid_search_path, "$(replace(lowercase(dataname), " " => "_"))_best.jls"), best_outcome(outcomes))
        Serialization.serialise(joinpath(grid_search_path, "$(replace(lowercase(dataname), " " => "_"))_best_eccco.jls"), best_eccco(outcomes))
        Serialization.serialise(joinpath(grid_search_path, "$(replace(lowercase(dataname), " " => "_"))_best_eccco_Δ.jls"), best_eccco_Δ(outcomes))
    end
end

const ECCCO_NAMES = [
    "ECCCo",
    "ECCCo (no CP)",
    "ECCCo (no EBM)",
    "ECCCo-Δ",
    "ECCCo-Δ (no CP)",
    "ECCCo-Δ (no EBM)",
]

"""
    best_outcome(outcomes; generator=ECCCO_NAMES, measure=["distance_from_energy", "distance_from_targets"])

Returns the best outcome from grid search results. The best outcome is defined as the one with the lowest average rank across all datasets and variables for the specified generator and measure.
"""
function best_outcome(outcomes::Dict; generator=ECCCO_NAMES, measure=["distance_from_energy", "distance_from_targets"])
    ranks = []
    for (params, outcome) in outcomes
        _ranks = generator_rank(outcome; generator=generator, measure=measure) |>
                x -> x.avg_rank |>
                     x -> sum(x) / length(x)[1]
        push!(ranks, _ranks)
    end
    best_index = argmin(ranks)
    best_outcome = (
        params = collect(keys(outcomes))[best_index],
        outcome = collect(values(outcomes))[best_index]
    )
    return best_outcome
end

best_eccco(outcomes) = best_outcome(outcomes; generator=["ECCCo"], measure=["distance_from_energy", "distance_from_targets"])

best_eccco_Δ(outcomes) = best_outcome(outcomes; generator=["ECCCo-Δ"], measure=["distance_from_energy", "distance_from_targets"])