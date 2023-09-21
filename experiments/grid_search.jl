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
        Serialization.serialize(joinpath(grid_search_path, "$(replace(lowercase(dataname), " " => "_"))_best_eccco.jls"), best_eccco(outcomes))
        Serialization.serialize(joinpath(grid_search_path, "$(replace(lowercase(dataname), " " => "_"))_best_eccco_Δ.jls"), best_eccco_Δ(outcomes))
    end
end

const ALL_ECCCO_NAMES = [
    "ECCCo",
    "ECCCo (no CP)",
    "ECCCo (no EBM)",
    "ECCCo-Δ",
    "ECCCo-Δ (no CP)",
    "ECCCo-Δ (no EBM)",
]

const ECCCO_NAMES = [
    "ECCCo",
    "ECCCo (no CP)",
    "ECCCo (no EBM)",
]

const ECCCo_Δ_NAMES = [
    "ECCCo-Δ",
    "ECCCo-Δ (no CP)",
    "ECCCo-Δ (no EBM)",
]

"""
    best_outcome(outcomes; generator=ECCCO_NAMES, measure=["distance_from_energy", "distance_from_targets"])

Returns the best outcome from grid search results. The best outcome is defined as the one with the lowest average rank across all datasets and variables for the specified generator and measure.
"""
function best_outcome(
    outcomes::Dict; 
    generator=ALL_ECCCO_NAMES, 
    measure=["distance_from_energy_l2", "distance_from_targets_l2"], 
    model::Union{Nothing,AbstractArray}=nothing,
    weights::Union{Nothing,AbstractArray}=nothing
)

    weights = isnothing(weights) ? ones(length(measure)) : weights
    df_weights = DataFrame(variable=measure, weight=weights)

    ranks = []
    for (params, outcome) in outcomes
        _ranks = generator_rank(outcome; generator=generator, measure=measure, model=model) |>
            x -> leftjoin(x, df_weights, on=:variable) |>
            x -> x.avg_rank .* x.weight |>
            x -> (sum(x) / length(x))[1]
        push!(ranks, _ranks)
    end
    best_index = argmin(ranks)
    best_outcome = (
        params = collect(keys(outcomes))[best_index],
        outcome = collect(values(outcomes))[best_index]
    )
    return best_outcome
end

best_eccco(outcomes; kwrgs...) = best_outcome(outcomes; generator=ECCCO_NAMES, kwrgs...)

best_eccco_Δ(outcomes; kwrgs...) = best_outcome(outcomes; generator=ECCCo_Δ_NAMES, kwrgs...)

"""
    best_absolute_outcome(outcomes; generator=ECCCO_NAMES, measure="distance_from_energy")

Return the best outcome from grid search results. The best outcome is defined as the one with the lowest average value across all datasets and variables for the specified generator and measure.
"""
function best_absolute_outcome(outcomes::Dict; generator=ECCCO_NAMES, measure::String="distance_from_energy_l2", model::Union{Nothing,AbstractArray}=nothing)
    avg_values = []
    for (params, outcome) in outcomes
        # Compute:
        results = summarise_outcome(outcome, measure=[measure], model=model)
        # Adjust variables for which higher is better:
        higher_is_better = [var ∈ ["validity", "redundancy"] for var in results.variable]
        results.mean[higher_is_better] .= -results.mean[higher_is_better]
        # Compute avergaes:
        _avg_values = subset(results, :generator => ByRow(x -> x ∈ generator)) |>
            x -> x.mean |>
            x -> (sum(x)/length(x))[1]
        push!(avg_values, _avg_values)
    end
    println(avg_values)
    best_index = argmin(avg_values)
    best_outcome = (
        params = collect(keys(outcomes))[best_index],
        outcome = collect(values(outcomes))[best_index]
    )
end

best_absolute_outcome_eccco(outcomes; kwrgs...) = best_absolute_outcome(outcomes; generator=ECCCO_NAMES, kwrgs...)

best_absolute_outcome_eccco_Δ(outcomes; kwrgs...) = best_absolute_outcome(outcomes; generator=ECCCo_Δ_NAMES, kwrgs...)

"""
    append_best_params!(params::NamedTuple, dataname::String)

Appends the best parameters from grid search results to the specified parameters.
"""
function append_best_params!(params::NamedTuple, dataname::String)
    if !isfile(joinpath(DEFAULT_OUTPUT_PATH, "grid_search", "$(replace(lowercase(dataname), " " => "_")).jls"))
        @warn "No grid search results found. Using default parameters."
    else
        @info "Appending best parameters from grid search results."
        grid_search_results = Serialization.deserialize(joinpath(DEFAULT_OUTPUT_PATH, "grid_search", "$(replace(lowercase(dataname), " " => "_")).jls"))
        best_params = best_eccco_Δ(grid_search_results).params
        params = (; params..., best_params...)
    end
end