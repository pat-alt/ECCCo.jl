"""
    plausibility(outcome::ExperimentOutcome)

Helper function to quickly filter a benchmark table for the distance from targets: the smaller this distance, the higher the plausibility.
"""
function plausibility(outcome::ExperimentOutcome)
    bmk = outcome.bmk
    df = @chain bmk() begin
        @group_by(dataname, generator, model, variable)
        @summarize(mean=mean(value),sd=std(value))
        @filter(variable == "distance_from_targets")
        @ungroup
        @arrange(mean)
    end
    return df
end

"""
    faithfulness(outcome::ExperimentOutcome)

Helper function to quickly filter a benchmark table for the distance from energy: the smaller this distance, the higher the faithfulness.
"""
function faithfulness(outcome::ExperimentOutcome)
    bmk = outcome.bmk
    df = @chain bmk() begin
        @group_by(dataname, generator, model, variable)
        @summarize(mean=mean(value),sd=std(value))
        @filter(variable == "distance_from_energy")
        @ungroup
        @arrange(mean)
    end
    return df
end

"""
    closeness(outcome::ExperimentOutcome)

Helper function to quickly filter a benchmark table for the distance from the factual: the smaller this distance, the higher the closeness desideratum.
"""
function closeness(outcome::ExperimentOutcome)
    bmk = outcome.bmk
    df = @chain bmk() begin
        @group_by(dataname, generator, model, variable)
        @summarize(mean=mean(value),sd=std(value))
        @filter(variable == "distance")
        @ungroup
        @arrange(mean)
    end
    return df
end