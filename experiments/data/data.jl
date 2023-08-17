function prepare_data(
    counterfactual_data::CounterfactualData;
    𝒟x=Normal(),
    min_batch_size=128,
    sampling_batch_size=50,
)
    X, _ = CounterfactualExplanations.DataPreprocessing.unpack_data(counterfactual_data)
    X = table(permutedims(X))
    labels = counterfactual_data.output_encoder.labels
    input_dim, n_obs = size(counterfactual_data.X)
    output_dim = length(unique(labels))
    save_name = replace(lowercase(dataname), " " => "_")

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
    return X, labels, n_obs, save_name, batch_size, sampler
end