"""
    aggregate_results(outcome::ExperimentOutcome; measure::String="distance_from_targets")

Function to quickly aggregate benchmarking results for a given measure.
"""
function aggregate_results(outcome::ExperimentOutcome; measure::String="distance_from_targets")
    df = @chain outcome.bmk() begin
        @group_by(generator, model)
        @filter(variable == measure)
        @summarize(val = mean(value), std = std(value))
        @ungroup
    end
    return df
end