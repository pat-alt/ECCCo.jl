"The default benchmarking measures."
const default_measures = [
    CounterfactualExplanations.distance,
    ECCCo.distance_from_energy,
    ECCCo.distance_from_targets,
    CounterfactualExplanations.Evaluation.validity,
    CounterfactualExplanations.Evaluation.redundancy,
    ECCCo.set_size_penalty
]

function default_generators(
    Λ::AbstractArray=[0.25, 0.75, 0.75],
    Λ_Δ::AbstractArray=[Λ[1], Λ[2], 4.0],
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
function run_benchmark(;
    n_individuals::Int,
    dataname::String,
    counterfactual_data::CounterfactualData,
    model_dict::Dict,
    generators::Union{Nothing, Dict}=nothing,
    measures::AbstractArray=default_measures,   
)
    # Benchmark generators:
    if isnothing(generators)
        generator_dict = default_generators()
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
                converge_when=:generator_conditions
            )
            push!(bmks, bmk)
        end
    end
    bmk = reduce(vcat, bmks)
    return bmk, generator_dict
end

