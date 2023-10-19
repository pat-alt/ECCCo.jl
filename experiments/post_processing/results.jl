"""
    plausibility(outcome::ExperimentOutcome)

Helper function to quickly filter a benchmark table for the distance from targets: the smaller this distance, the higher the plausibility.
"""
function summarise_outcome(
    outcome::ExperimentOutcome;
    measure::Union{Nothing,AbstractArray} = nothing,
    model::Union{Nothing,AbstractArray} = nothing,
)

    bmk = outcome.bmk
    measure = isnothing(measure) ? unique(bmk().variable) : measure
    df = bmk()
    # If the :run column is missing (single runs), add it:
    if !("run" ∈ names(df))
        df.run .= 1
    end
    # Aggregate per run:
    df =
        groupby(df, [:dataname, :generator, :model, :run, :variable]) |>
        x ->
            combine(x, :value => mean => :mean_group, :value => std => :std_group) |>
            x -> subset(x, :variable => ByRow(x -> x ∈ measure))
    # Compute mean and std across runs:
    df =
        groupby(df, [:dataname, :generator, :model, :variable]) |>
        x ->
            combine(x, :mean_group => mean => :mean, :mean_group => std => :std) 
    # Subset:
    if !isnothing(model)
        df = subset(df, :model => ByRow(x -> x ∈ model))
    end
    sort!(df, [:model, :variable, :mean])
    return df
end

"""
    plausibility(outcome::ExperimentOutcome)

Helper function to quickly filter a benchmark table for the distance from targets: the smaller this distance, the higher the plausibility.
"""
plausibility(outcome::ExperimentOutcome; kwrgs...) =
    summarise_outcome(outcome, measure = ["distance_from_targets_l2"], kwrgs...)

"""
    plausibility_image(outcome::ExperimentOutcome)

Helper function to quickly filter a benchmark table for the distance from targets: the smaller this distance, the higher the plausibility.
"""
plausibility_image(outcome::ExperimentOutcome; kwrgs...) =
    summarise_outcome(outcome, measure = ["distance_from_targets_ssim"], kwrgs...)

"""
    faithfulness(outcome::ExperimentOutcome)

Helper function to quickly filter a benchmark table for the distance from energy: the smaller this distance, the higher the faithfulness.
"""
faithfulness(outcome::ExperimentOutcome; kwrgs...) =
    summarise_outcome(outcome, measure = ["distance_from_energy_l2"], kwrgs...)

"""
    faithfulness_image(outcome::ExperimentOutcome)

Helper function to quickly filter a benchmark table for the distance from energy: the smaller this distance, the higher the faithfulness.
"""
faithfulness_image(outcome::ExperimentOutcome; kwrgs...) =
    summarise_outcome(outcome, measure = ["distance_from_energy_ssim"], kwrgs...)

"""
    closeness(outcome::ExperimentOutcome)

Helper function to quickly filter a benchmark table for the distance from the factual: the smaller this distance, the higher the closeness desideratum.
"""
closeness(outcome::ExperimentOutcome; kwrgs...) =
    summarise_outcome(outcome, measure = ["distance"], kwrgs...)

"""
    validity(outcome::ExperimentOutcome)

Helper function to quickly filter a benchmark table for the validity: the higher this value, the higher the validity.
"""
validity(outcome::ExperimentOutcome; kwrgs...) =
    summarise_outcome(outcome, measure = ["validity"], kwrgs...)

"""
    redundancy(outcome::ExperimentOutcome)

Helper function to quickly filter a benchmark table for the redundancy: the higher this value, the higher the redundancy.
"""
redundancy(outcome::ExperimentOutcome; kwrgs...) =
    summarise_outcome(outcome, measure = ["redundancy"], kwrgs...)

"""
    uncertainty(outcome::ExperimentOutcome)

Helper function to quickly filter a benchmark table for the uncertainty: the higher this value, the higher the uncertainty.
"""
uncertainty(outcome::ExperimentOutcome; kwrgs...) =
    summarise_outcome(outcome, measure = ["set_size_penalty"], kwrgs...)

"""
    generator_rank(outcome::ExperimentOutcome; generator::Union{AbstractArray,Nothing}=nothing, measure::Union{AbstractArray,Nothing}=nothing, model::Union{Nothing,String}=nothing)

Computes the average rank of a generator across all datasets and variables.
"""
function generator_rank(
    outcome::ExperimentOutcome;
    generator::Union{AbstractArray,Nothing} = nothing,
    measure::Union{AbstractArray,Nothing} = nothing,
    model::Union{Nothing,AbstractArray} = nothing,
)

    # Setup:
    generator = isnothing(generator) ? collect(keys(outcome.generator_dict)) : generator
    bmk = outcome.bmk
    measure = isnothing(measure) ? unique(bmk().variable) : measure

    # Compute:
    results = summarise_outcome(outcome, measure = measure, model = model)
    # Adjust variables for which higher is better:
    higher_is_better = [var ∈ ["validity", "redundancy"] for var in results.variable]
    results.mean[higher_is_better] .= -results.mean[higher_is_better]
    # Compute ranks:
    ranked_results =
        groupby(results, [:dataname, :model, :variable]) |>
        x ->
            combine(x, :mean => sortperm => :rank, :generator) |>
            x ->
                subset(x, :generator => ByRow(x -> x ∈ generator)) |>
                x ->
                    groupby(x, [:dataname, :generator, :variable]) |>
                    x ->
                        combine(x, :rank => mean => :avg_rank) |>
                        x -> subset(x, :variable => ByRow(x -> x ∈ measure))
    sort!(ranked_results, [:variable, :avg_rank])
    return ranked_results
end

generator_rank_plausibility(outcome::ExperimentOutcome; kwrgs...) =
    generator_rank(outcome, measure = ["distance_from_targets_l2"], kwrgs...)

generator_rank_faithfulness(outcome::ExperimentOutcome; kwrgs...) =
    generator_rank(outcome, measure = ["distance_from_energy_l2"], kwrgs...)

generator_rank_closeness(outcome::ExperimentOutcome; kwrgs...) =
    generator_rank(outcome, measure = ["distance"], kwrgs...)
