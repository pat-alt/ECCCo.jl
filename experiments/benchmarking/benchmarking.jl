function default_generators(;
    Λ::AbstractArray=[0.25, 0.75, 0.75],
    Λ_Δ::AbstractArray=Λ,
    use_variants::Bool=true,
    use_class_loss::Bool=false,
    opt=Flux.Optimise.Descent(0.01),
    nsamples::Union{Nothing,Int}=nothing,
    nmin::Union{Nothing,Int}=nothing,
    reg_strength::Real=0.5,
)

    @info "Begin benchmarking counterfactual explanations."
    λ₁, λ₂, λ₃ = Λ
    λ₁_Δ, λ₂_Δ, λ₃_Δ = Λ_Δ

    if use_variants
        generator_dict = Dict(
            "Wachter" => WachterGenerator(λ=λ₁, opt=opt),
            "REVISE" => REVISEGenerator(λ=λ₁, opt=opt),
            "Schut" => GreedyGenerator(),
            "ECCCo" => ECCCoGenerator(λ=Λ, opt=opt, use_class_loss=use_class_loss, nsamples=nsamples, nmin=nmin),
            "ECCCo (no CP)" => ECCCoGenerator(λ=[λ₁, 0.0, λ₃], opt=opt, use_class_loss=use_class_loss, nsamples=nsamples, nmin=nmin),
            "ECCCo (no EBM)" => ECCCoGenerator(λ=[λ₁, λ₂, 0.0], opt=opt, use_class_loss=use_class_loss, nsamples=nsamples, nmin=nmin),
            "ECCCo-Δ" => ECCCoGenerator(λ=Λ_Δ, opt=opt, use_class_loss=use_class_loss, use_energy_delta=true, nsamples=nsamples, nmin=nmin, reg_strength=reg_strength),
            "ECCCo-Δ (no CP)" => ECCCoGenerator(λ=[λ₁_Δ, 0.0, λ₃_Δ], opt=opt, use_class_loss=use_class_loss, use_energy_delta=true, nsamples=nsamples, nmin=nmin, reg_strength=reg_strength),
            "ECCCo-Δ (no EBM)" => ECCCoGenerator(λ=[λ₁_Δ, λ₂_Δ, 0.0], opt=opt, use_class_loss=use_class_loss, use_energy_delta=true, nsamples=nsamples, nmin=nmin, reg_strength=reg_strength),
        )
    else
        generator_dict = Dict(
            "Wachter" => WachterGenerator(λ=λ₁, opt=opt),
            "REVISE" => REVISEGenerator(λ=λ₁, opt=opt),
            "Schut" => GreedyGenerator(),
            "ECCCo" => ECCCoGenerator(λ=Λ, opt=opt, use_class_loss=use_class_loss, nsamples=nsamples, nmin=nmin),
            "ECCCo-Δ" => ECCCoGenerator(λ=Λ_Δ, opt=opt, use_class_loss=use_class_loss, use_energy_delta=true, nsamples=nsamples, nmin=nmin, reg_strength=reg_strength),
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
function run_benchmark(exper::Experiment, model_dict::Dict)

    n_individuals = exper.n_individuals
    dataname = exper.dataname
    counterfactual_data = exper.counterfactual_data
    generator_dict = exper.generators
    measures = exper.ce_measures
    parallelizer = exper.parallelizer

    # Benchmark generators:
    if isnothing(generator_dict)
        generator_dict = default_generators(;
            Λ=exper.Λ,
            Λ_Δ=exper.Λ_Δ,
            use_variants=exper.use_variants,
            use_class_loss=exper.use_class_loss,
            opt=exper.opt,
            nsamples=exper.nsamples,
            nmin=exper.nmin,
            reg_strength=exper.reg_strength,
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
            @info "Benchmarking factual=$(factual) ▶️ target=$(target)."
            # Progress bars are only shown on the root process:
            if !(is_multi_processed(exper) && MPI.Comm_rank(exper.parallelizer.comm) != 0)
                verbose = true
            else
                verbose = false
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
                verbose=verbose
            )
            push!(bmks, bmk)
        end
    end
    bmk = reduce(vcat, bmks)
    return bmk, generator_dict
end

