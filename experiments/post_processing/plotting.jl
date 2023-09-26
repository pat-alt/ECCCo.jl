using Plots

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

function plot_all_mnist(gen, model, data=load_mnist_test(); img_height=150, seed=123, maxoutdim=64)

    # Set seed:
    if !isnothing(seed)
        Random.seed!(seed)
    end

    # Dimensionality reduction:
    data.dt = MultivariateStats.fit(MultivariateStats.PCA, data.X; maxoutdim=maxoutdim)

    targets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    factuals = targets
    plts = []

    for factual in factuals
        chosen = rand(findall(predict_label(model, data) .== factual))
        x = select_factual(data, chosen)
        for target in targets
            if factual != target
                @info "Generating counterfactual for $(factual) -> $(target)"
                ce = generate_counterfactual(
                    x, target, data, model, gen; 
                    initialization=:identity, converge_when=:generator_conditions
                )
                plt = Plots.plot(
                    CounterfactualExplanations.counterfactual(ce) |> ECCCo.convert2mnist,
                    axis=([], false),
                    size=(img_height, img_height),
                    title="$factual → $target",
                )
            else
                plt = Plots.plot(
                    x |> ECCCo.convert2mnist,
                    axis=([], false),
                    size=(img_height, img_height),
                    title="Factual",
                )
            end
            push!(plts, plt)
        end
    end

    plt = Plots.plot(
        plts...,
        layout=(length(factuals), length(targets)),
        size=(img_height * length(targets), img_height * length(factuals)),
        dpi=300
    )

    return plt

end