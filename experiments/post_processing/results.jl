"""
    plausibility(outcome::ExperimentOutcome)

Helper function to quickly filter a benchmark table for the distance from targets: the smaller this distance, the higher the plausibility.
"""
function summarise_outcome(outcome::ExperimentOutcome; measure::Union{Nothing,AbstractArray}=nothing, model::Union{Nothing,AbstractArray}=nothing)

    bmk = outcome.bmk
    measure = isnothing(measure) ? unique(bmk().variable) : measure

    df = groupby(bmk(), [:dataname, :generator, :model, :variable]) |>
        x -> combine(x, :value => mean => :mean, :value => std => :std) |>
        x -> subset(x, :variable => ByRow(x -> x ∈ measure))
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
plausibility(outcome::ExperimentOutcome; kwrgs...) = summarise_outcome(outcome, measure=["distance_from_targets"], kwrgs...)


"""
    faithfulness(outcome::ExperimentOutcome)

Helper function to quickly filter a benchmark table for the distance from energy: the smaller this distance, the higher the faithfulness.
"""
faithfulness(outcome::ExperimentOutcome; kwrgs...) = summarise_outcome(outcome, measure=["distance_from_energy"], kwrgs...)

"""
    closeness(outcome::ExperimentOutcome)

Helper function to quickly filter a benchmark table for the distance from the factual: the smaller this distance, the higher the closeness desideratum.
"""
closeness(outcome::ExperimentOutcome; kwrgs...) = summarise_outcome(outcome, measure=["distance"], kwrgs...)

"""
    generator_rank(outcome::ExperimentOutcome; generator::Union{AbstractArray,Nothing}=nothing, measure::Union{AbstractArray,Nothing}=nothing, model::Union{Nothing,String}=nothing)

Computes the average rank of a generator across all datasets and variables.
"""
function generator_rank(
    outcome::ExperimentOutcome; 
    generator::Union{AbstractArray,Nothing}=nothing, 
    measure::Union{AbstractArray,Nothing}=nothing, 
    model::Union{Nothing,AbstractArray}=nothing
)

    # Setup:
    generator = isnothing(generator) ? collect(keys(outcome.generator_dict)) : generator
    bmk = outcome.bmk
    measure = isnothing(measure) ? unique(bmk().variable) : measure

    # Compute:
    results = summarise_outcome(outcome, measure=measure, model=model) 
    # Adjust variables for which higher is better:
    higher_is_better = [var ∈ ["validity", "redundancy"] for var in results.variable]
    results.mean[higher_is_better] .= - results.mean[higher_is_better]
    # Compute ranks:
    ranked_results = groupby(results, [:dataname, :model, :variable]) |>
        x -> combine(x, :mean => sortperm => :rank, :generator) |>
        x -> subset(x, :generator => ByRow(x -> x ∈ generator)) |>
        x -> groupby(x, [:dataname, :generator, :variable]) |>
        x -> combine(x, :rank => mean => :avg_rank) |>
        x -> subset(x, :variable => ByRow(x -> x ∈ measure)) 
    sort!(ranked_results, [:variable, :avg_rank])
    return ranked_results
end

generator_rank_plausibility(outcome::ExperimentOutcome; kwrgs...) = generator_rank(outcome, measure=["distance_from_targets"], kwrgs...)

generator_rank_faithfulness(outcome::ExperimentOutcome; kwrgs...) = generator_rank(outcome, measure=["distance_from_energy"], kwrgs...)

generator_rank_closeness(outcome::ExperimentOutcome; kwrgs...) = generator_rank(outcome, measure=["distance"], kwrgs...)

