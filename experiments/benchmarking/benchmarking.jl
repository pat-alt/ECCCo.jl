function default_generators(;
    Λ::AbstractArray=[0.25, 0.75, 0.75],
    Λ_Δ::AbstractArray=Λ,
    use_variants::Bool=true,
    use_class_loss::Bool=false,
    opt=Flux.Optimise.Descent(0.01),
)

    @info "Begin benchmarking counterfactual explanations."
    λ₁, λ₂, λ₃ = Λ
    λ₁_Δ, λ₂_Δ, λ₃_Δ = Λ_Δ

    if use_variants
        generator_dict = Dict(
            "Wachter" => WachterGenerator(λ=λ₁, opt=opt),
            "REVISE" => REVISEGenerator(λ=λ₁, opt=opt),
            "Schut" => GreedyGenerator(),
            "ECCCo" => ECCCoGenerator(λ=Λ, opt=opt, use_class_loss=use_class_loss),
            "ECCCo (no CP)" => ECCCoGenerator(λ=[λ₁, 0.0, λ₃], opt=opt, use_class_loss=use_class_loss),
            "ECCCo (no EBM)" => ECCCoGenerator(λ=[λ₁, λ₂, 0.0], opt=opt, use_class_loss=use_class_loss),
            "ECCCo-Δ" => ECCCoGenerator(λ=Λ_Δ, opt=opt, use_class_loss=use_class_loss, use_energy_delta=true),
            "ECCCo-Δ (no CP)" => ECCCoGenerator(λ=[λ₁_Δ, 0.0, λ₃_Δ], opt=opt, use_class_loss=use_class_loss, use_energy_delta=true),
            "ECCCo-Δ (no EBM)" => ECCCoGenerator(λ=[λ₁_Δ, λ₂_Δ, 0.0], opt=opt, use_class_loss=use_class_loss, use_energy_delta=true),
        )
    else
        generator_dict = Dict(
            "Wachter" => WachterGenerator(λ=λ₁, opt=opt),
            "REVISE" => REVISEGenerator(λ=λ₁, opt=opt),
            "Schut" => GreedyGenerator(),
            "ECCCo" => ECCCoGenerator(λ=Λ, opt=opt, use_class_loss=use_class_loss),
            "ECCCo-Δ" => ECCCoGenerator(λ=Λ_Δ, opt=opt, use_class_loss=use_class_loss, use_energy_delta=true),
        )
    end
    return generator_dict
end

"""
    run_benchmark(
        generators::Union{Nothing, Dict}=nothing,
        measures::AbstractArray=default_measures,
    )

Run the benchmarking procedure.
"""
function run_benchmark(exp::Experiment, model_dict::Dict)

    n_individuals = exp.n_individuals
    dataname = exp.dataname
    counterfactual_data = exp.counterfactual_data
    generator_dict = exp.generators
    measures = exp.ce_measures
    parallelizer = exp.parallelizer

    # Benchmark generators:
    if isnothing(generator_dict)
        generator_dict = default_generators(;
            Λ=exp.Λ,
            Λ_Δ=exp.Λ_Δ,
            use_variants=exp.use_variants,
            use_class_loss=exp.use_class_loss,
            opt=exp.opt
        )
    end

    # Run benchmark:
    bmks = []
    labels = counterfactual_data.output_encoder.labels
    for target in sort(unique(labels))
        for factual in sort(unique(labels))
            if factual == target
                continue
            end
            bmk = benchmark(
                counterfactual_data;
                models=model_dict,
                generators=generator_dict,
                measure=measures,
                suppress_training=true, dataname=dataname,
                n_individuals=n_individuals,
                target=target, factual=factual,
                initialization=:identity,
                converge_when=:generator_conditions,
                parallelizer=parallelizer,
            )
            push!(bmks, bmk)
        end
    end
    bmk = reduce(vcat, bmks)
    return bmk, generator_dict
end

