function choose_random_mnist(outcome::ExperimentOutcome; model::String="MLP", img_height=200, seed=966)

    # Set seed:
    if !isnothing(seed)
        Random.seed!(seed)
    end

    # Get output:
    bmk = outcome.bmk()
    grouped_bmk = groupby(bmk[bmk.variable.=="distance" .&& bmk.model.==model,:], [:dataname, :target, :factual])
    random_choice = rand(1:length(grouped_bmk))
    generators = unique(bmk.generator)
    n_generators = length(generators)

    # Get data:
    df = grouped_bmk[random_choice][1:n_generators, :] |> 
        x -> sort(x, :generator) |>
        x -> subset(x, :generator => ByRow(x -> x != "ECCCo"))
    generators = df.generator
    replace!(generators, "ECCCo-Δ" => "ECCCo")
    replace!(generators, "ECCCo-Δ (latent)" => "ECCCo+")
    n_generators = length(generators)

    # Factual:
    img = CounterfactualExplanations.factual(grouped_bmk[random_choice][1:n_generators,:].ce[1]) |> ECCCo.convert2mnist
    p1 = Plots.plot(
        img,
        axis=([], false),
        size=(img_height, img_height),
        title="Factual",
    )
    plts = [p1]

    # Counterfactuals:
    for (i, generator) in enumerate(generators)
        img = CounterfactualExplanations.counterfactual(df.ce[i]) |> ECCCo.convert2mnist
        p = Plots.plot(
            img,
            axis=([], false),
            size=(img_height, img_height),
            title="$generator",
        )
        push!(plts, p)
    end

    plt = Plots.plot(
        plts...,
        layout=(1, n_generators + 1), 
        size=(img_height * (n_generators + 1), img_height)
    )
    display(plt)

    return plt, df.target[1], seed

end

function plot_random_eccco(outcome::ExperimentOutcome; generator="ECCCo-Δ", img_height=200, seed=966)
    # Set seed:
    if !isnothing(seed)
        Random.seed!(seed)
    end

    # Get output:
    bmk = outcome.bmk()
    grouped_bmk = groupby(bmk[bmk.variable.=="distance" .&& bmk.generator.==generator, :], [:dataname, :target, :factual])
    random_choice = rand(1:length(grouped_bmk))
    models = unique(bmk.model)
    n_models = length(models)

    df = grouped_bmk[random_choice]
    while nrow(df) > n_models
        random_choice = rand(1:length(grouped_bmk))
        df = grouped_bmk[random_choice]
    end
    sort!(df, :model)
    models = df.model

    # Factual:
    img = CounterfactualExplanations.factual(df.ce[1]) |> ECCCo.convert2mnist
    p1 = Plots.plot(
        img,
        axis=([], false),
        size=(img_height, img_height),
        title="Factual",
    )
    plts = [p1]

    # Counterfactuals:
    for (i, model) in enumerate(models)
        img = CounterfactualExplanations.counterfactual(df.ce[i]) |> ECCCo.convert2mnist
        p = Plots.plot(
            img,
            axis=([], false),
            size=(img_height, img_height),
            title="$model",
        )
        push!(plts, p)
    end

    plt = Plots.plot(
        plts...,
        layout=(1, n_models + 1),
        size=(img_height * (n_models + 1), img_height)
    )
    display(plt)

    return plt, df.target[1], seed
end