function save_best(outcomes_file_path::String)
    # Just load the best model from the grid search:
    @assert isfile(outcomes_file_path)
    outcomes = Serialization.deserialize(outcomes_file_path)
    outcome = best_absolute_outcome_eccco_Δ(outcomes).outcome
    exper = outcome.exper

    # Save data:
    output_path = replace(exper.output_path, "grid_search" => "")
    params_path = joinpath(output_path, "params")
    Serialization.serialize(joinpath(output_path, "$(exper.save_name)_outcome.jls"), outcome)
    Serialization.serialize(joinpath(output_path, "$(exper.save_name)_bmk.jls"), outcome.bmk)
    Serialization.serialize(joinpath(output_path, "$(exper.save_name)_models.jls"), outcome.model_dict)
    meta(outcome; save_output=true, params_path=params_path)
end