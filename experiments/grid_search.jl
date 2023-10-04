using DataFrames

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
    n_individuals::Int = N_IND,
    tuning_params::NamedTuple,
    kwargs...,
)

    # Output path:
    grid_search_path = mkpath(joinpath(DEFAULT_OUTPUT_PATH, "grid_search"))

    # Grid setup:
    tuning_params = [Pair.(k, vals) for (k, vals) in pairs(tuning_params)]
    grid = Iterators.product(tuning_params...)

    # Temporary storage on disk:
    storage_path = joinpath(grid_search_path, ".tmp_results_$(replace(lowercase(dataname), " " => "_"))")
    mkpath(storage_path)
    @info "Storing temporary results in $(storage_path)."

    # Search:
    counter = 1
    for params in grid
        @info "Running experiment $(counter)/$(length(grid)) with tuning parameters: $(params)"
        outcome = run_experiment(
            counterfactual_data,
            test_data;
            save_output = false,
            dataname = dataname,
            n_individuals = n_individuals,
            output_path = grid_search_path,
            params...,
            kwargs...,
        )

        params = map(x -> typeof(x[2]) <: Vector ? x[1] => Tuple(x[2]) : x[1] => x[2], params)
        df_params =
            DataFrame(merge(Dict(:id => counter), Dict(params))) |>
            x -> select(x, :id, Not(:id))
        df_outcomes =
            DataFrame(Dict(:id => counter, :params => params, :outcome => outcome)) |>
            x -> select(x, :id, Not(:id))

        # Save:
        if !(is_multi_processed(PLZ) && MPI.Comm_rank(PLZ.comm) != 0)
            Serialization.serialize(
                joinpath(storage_path, "params_$(counter).jls"),
                df_params,
            )
            Serialization.serialize(
                joinpath(storage_path, "outcomes_$(counter).jls"),
                df_outcomes,
            )
        end
        counter += 1
    end

    # Save:
    if !(is_multi_processed(PLZ) && MPI.Comm_rank(PLZ.comm) != 0)

        # Deserialise:
        df_params = []
        df_outcomes = []
        for i in 1:length(grid)
            push!(df_params, Serialization.deserialize(joinpath(storage_path, "params_$(i).jls")))
            push!(df_outcomes, Serialization.deserialize(joinpath(storage_path, "outcomes_$(i).jls")))
        end
        outcomes = Dict(:df_params => vcat(df_params...), :df_outcomes => vcat(df_outcomes...))

        Serialization.serialize(
            joinpath(grid_search_path, "$(replace(lowercase(dataname), " " => "_")).jls"),
            outcomes,
        )
        Serialization.serialize(
            joinpath(
                grid_search_path,
                "$(replace(lowercase(dataname), " " => "_"))_best.jls",
            ),
            best_absolute_outcome(outcomes),
        )
        Serialization.serialize(
            joinpath(
                grid_search_path,
                "$(replace(lowercase(dataname), " " => "_"))_best_eccco.jls",
            ),
            best_absolute_outcome_eccco(outcomes),
        )
        Serialization.serialize(
            joinpath(
                grid_search_path,
                "$(replace(lowercase(dataname), " " => "_"))_best_eccco_delta.jls",
            ),
            best_absolute_outcome_eccco_Δ(outcomes),
        )
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

const ECCCO_NAMES = ["ECCCo", "ECCCo (no CP)", "ECCCo (no EBM)"]

const ECCCo_Δ_NAMES = ["ECCCo-Δ", "ECCCo-Δ (no CP)", "ECCCo-Δ (no EBM)", "ECCCo-Δ (latent)"]

"""
    best_outcome(outcomes; generator=ECCCO_NAMES, measure=["distance_from_energy", "distance_from_targets"])

Returns the best outcome from grid search results. The best outcome is defined as the one with the lowest average rank across all datasets and variables for the specified generator and measure.
"""
function best_rank_outcome(
    outcomes::Dict;
    generator = ALL_ECCCO_NAMES,
    measure = ["distance_from_energy_l2", "distance_from_targets_l2"],
    model::Union{Nothing,AbstractArray} = nothing,
    weights::Union{Nothing,AbstractArray} = nothing,
)

    weights = isnothing(weights) ? ones(length(measure)) : weights
    df_weights = DataFrame(variable = measure, weight = weights)

    ranks = []
    for outcome in outcomes[:df_outcomes].outcome
        _ranks =
            generator_rank(
                outcome;
                generator = generator,
                measure = measure,
                model = model,
            ) |>
            x ->
                leftjoin(x, df_weights, on = :variable) |>
                x -> x.avg_rank .* x.weight |> x -> (sum(x)/length(x))[1]
        push!(ranks, _ranks)
    end
    best_index = argmin(ranks)
    best_outcome = (
        params=outcomes[:df_outcomes].params[best_index],
        outcome=outcomes[:df_outcomes].outcome[best_index],
    )
    return best_outcome
end

best_rank_eccco(outcomes; kwrgs...) =
    best_rank_outcome(outcomes; generator = ECCCO_NAMES, kwrgs...)

best_rank_eccco_Δ(outcomes; kwrgs...) =
    best_rank_outcome(outcomes; generator = ECCCo_Δ_NAMES, kwrgs...)

"""
    best_absolute_outcome(outcomes; generator=ECCCO_NAMES, measure="distance_from_energy")

Return the best outcome from grid search results. The best outcome is defined as the one with the lowest average value across all datasets and variables for the specified generator and measure.
"""
function best_absolute_outcome(
    outcomes::Dict;
    generator = ECCCO_NAMES,
    measure::AbstractArray = ["distance_from_energy_l2"],
    model::Union{Nothing,AbstractArray} = nothing,
    weights::Union{Nothing,AbstractArray} = nothing,
)

    weights = isnothing(weights) ? ones(length(measure)) : weights
    df_weights = DataFrame(variable = measure, weight = weights)

    avg_values = []
    for (params, outcome) in zip(outcomes[:df_outcomes].params, outcomes[:df_outcomes].outcome)

        # Setup
        evaluation = deepcopy(outcome.bmk.evaluation)
        exper = outcome.exper
        generator_dict = outcome.generator_dict
        model_dict = outcome.model_dict

        # Discard outlier results:
        if any(evaluation.value .> 1e6)
            @warn "Discarding outlier results: $(params)."
            push!(avg_values, Inf)
            continue
        end

        # Adjust variables for which higher is better:
        higher_is_better = [var ∈ ["validity", "redundancy"] for var in evaluation.variable]
        evaluation.value[higher_is_better] .= -evaluation.value[higher_is_better]

        # Normalise to allow for comparison across measures:
        evaluation =
            groupby(evaluation, [:dataname, :variable]) |>
            x -> transform(x, :value => standardize => :value)

        # Reconstruct outcome with normalised values:
        bmk = CounterfactualExplanations.Evaluation.Benchmark(evaluation)
        outcome = ExperimentOutcome(exper, model_dict, generator_dict, bmk)

        # Compute:
        results =
            summarise_outcome(outcome, measure = measure, model = model) |>
            x -> leftjoin(x, df_weights, on = :variable)

        # Compute weighted averages:
        _avg_values =
            subset(results, :generator => ByRow(x -> x ∈ generator)) |>
            x -> x.mean .* x.weight |> x -> (sum(x)/length(x))[1]

        # Append:
        push!(avg_values, _avg_values)
    end
    best_index = argmin(avg_values)
    best_outcome = (
        params=outcomes[:df_outcomes].params[best_index],
        outcome=outcomes[:df_outcomes].outcome[best_index],
    )
end

best_absolute_outcome_eccco(outcomes; kwrgs...) =
    best_absolute_outcome(outcomes; generator = ECCCO_NAMES, kwrgs...)

best_absolute_outcome_eccco_Δ(outcomes; kwrgs...) =
    best_absolute_outcome(outcomes; generator = ECCCo_Δ_NAMES, kwrgs...)

"""
    append_best_params!(params::NamedTuple, dataname::String)

Appends the best parameters from grid search results to the specified parameters.
"""
function append_best_params!(params::NamedTuple, dataname::String)
    if !isfile(
        joinpath(
            DEFAULT_OUTPUT_PATH,
            "grid_search",
            "$(replace(lowercase(dataname), " " => "_")).jls",
        ),
    )
        @warn "No grid search results found. Using default parameters."
    else
        @info "Appending best parameters from grid search results."
        grid_search_results = Serialization.deserialize(
            joinpath(
                DEFAULT_OUTPUT_PATH,
                "grid_search",
                "$(replace(lowercase(dataname), " " => "_")).jls",
            ),
        )
        best_params = best_absolute_outcome_eccco_Δ(grid_search_results).params
        params = (; params..., best_params...)
    end
end
