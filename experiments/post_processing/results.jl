"""
    plausibility(outcome::ExperimentOutcome)

Helper function to quickly filter a benchmark table for the distance from targets: the smaller this distance, the higher the plausibility.
"""
function summarise_outcome(outcome::ExperimentOutcome; measure="distance_from_targets", model::Union{Nothing,String}=nothing)
    bmk = outcome.bmk
    df = groupby(bmk(), [:dataname, :generator, :model, :variable]) |>
        x -> combine(x, :value => mean => :mean, :value => std => :std) |>
        x -> subset(x, :variable => ByRow(x -> x ==measure))
    if !isnothing(model)
        df = subset(df, :model => ByRow(x -> x == model))
    end
    sort!(df, [:model, :mean])
    return df
end

"""
    plausibility(outcome::ExperimentOutcome)

Helper function to quickly filter a benchmark table for the distance from targets: the smaller this distance, the higher the plausibility.
"""
plausibility(outcome::ExperimentOutcome; kwrgs...) = summarise_outcome(outcome, measure="distance_from_targets", kwrgs...)


"""
    faithfulness(outcome::ExperimentOutcome)

Helper function to quickly filter a benchmark table for the distance from energy: the smaller this distance, the higher the faithfulness.
"""
faithfulness(outcome::ExperimentOutcome; kwrgs...) = summarise_outcome(outcome, measure="distance_from_energy", kwrgs...)

"""
    closeness(outcome::ExperimentOutcome)

Helper function to quickly filter a benchmark table for the distance from the factual: the smaller this distance, the higher the closeness desideratum.
"""
closeness(outcome::ExperimentOutcome; kwrgs...) = summarise_outcome(outcome, measure="distance", kwrgs...)