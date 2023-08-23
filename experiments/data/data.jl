function _prepare_data(exp::Experiment)

    # Unpack data:
    counterfactual_data = exp.counterfactual_data
    min_batch_size = exp.min_batch_size
    sampling_batch_size = exp.sampling_batch_size
    𝒟x = exp.𝒟x

    # Data parameters:
    X, _ = CounterfactualExplanations.DataPreprocessing.unpack_data(counterfactual_data)
    X = table(permutedims(X))
    labels = counterfactual_data.output_encoder.labels
    input_dim, n_obs = size(counterfactual_data.X)
    output_dim = length(unique(labels))

    # Model parameters:
    batch_size = minimum([Int(round(n_obs / 10)), min_batch_size])
    sampling_batch_size = isnothing(sampling_batch_size) ? batch_size : sampling_batch_size

    # JEM parameters:
    𝒟y = Categorical(ones(output_dim) ./ output_dim)
    sampler = ConditionalSampler(
        𝒟x, 𝒟y,
        input_size=(input_dim,),
        batch_size=sampling_batch_size,
    )
    return X, labels, n_obs, batch_size, sampler
end

function meta_data(exp::Experiment)
    _, _, n_obs, batch_size, _ = _prepare_data(exp::Experiment)
    return n_obs, batch_size
end

function prepare_data(exp::Experiment)
    X, labels, _, _,  sampler = _prepare_data(exp::Experiment)
    return X, labels, sampler
end

function batch_size(exp::Experiment)
    _, _, _, batch_size, _ = _prepare_data(exp::Experiment)
    return batch_size
end