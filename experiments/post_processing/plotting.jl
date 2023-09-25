function choose_random_mnist(outcome::ExperimentOutcome; model::String="LeNet-5", img_height=125, seed=966)

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
    ces = []

    # Counterfactuals:
    for (i, generator) in enumerate(generators)
        ce = df.ce[i]
        img = CounterfactualExplanations.counterfactual(ce) |> ECCCo.convert2mnist
        p = Plots.plot(
            img,
            axis=([], false),
            size=(img_height, img_height),
            title="$generator",
        )
        push!(plts, p)
        push!(ces, ce)
    end

    plt = Plots.plot(
        plts...,
        layout=(1, n_generators + 1), 
        size=(img_height * (n_generators + 1), img_height),
        dpi=300
    )
    display(plt)

    return plt, df.target[1], seed, ces, df.sample[1]

end

function plot_random_eccco(outcome::ExperimentOutcome; ce=nothing, generator="ECCCo-Δ", img_height=200, seed=966)
    # Set seed:
    if !isnothing(seed)
        Random.seed!(seed)
    end

    # Get output:
    bmk = outcome.bmk()
    ce = isnothing(ce) ? rand(bmk.ce) : ce
    gen = outcome.generator_dict[generator]
    models = outcome.model_dict
    x = CounterfactualExplanations.counterfactual(ce)
    target = ce.target
    data = ce.data

    # Factual:
    img = CounterfactualExplanations.factual(ce) |> ECCCo.convert2mnist
    p1 = Plots.plot(
        img,
        axis=([], false),
        size=(img_height, img_height),
        title="Factual",
    )
    plts = [p1]

    for (model_name, M) in models
        ce = generate_counterfactual(x, target, data, M, gen; initialization=:identity, converge_when=:generator_conditions)
        img = CounterfactualExplanations.counterfactual(ce) |> ECCCo.convert2mnist
        p = Plots.plot(
            img,
            axis=([], false),
            size=(img_height, img_height),
            title="$model_name",
        )
        push!(plts, p)
    end
    n_models = length(models)

    plt = Plots.plot(
        plts...,
        layout=(1, n_models + 1),
        size=(img_height * (n_models + 1), img_height),
        dpi=300
    )
    display(plt)

    return plt, target, seed
end