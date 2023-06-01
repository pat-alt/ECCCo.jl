# General setup:
include("$(pwd())/notebooks/setup.jl")
eval(setup_notebooks)
output_path = "$(pwd())/replicated"
isdir(output_path) || mkdir(output_path)
@info "All results will be saved to $output_path."
params_path = "$(pwd())/replicated/params"
isdir(params_path) || mkdir(params_path)
@info "All parameter choices will be saved to $params_path."
test_size = 0.2

# Artifacts:
using LazyArtifacts
@warn "Models were pre-trained on `julia-1.8.5` and may not work on other versions."
artifact_path = joinpath(artifact"results-paper-submission-1.8.5","results-paper-submission-1.8.5")
pretrained_path = joinpath(artifact_path, "results")

function run_experiment(
    counterfactual_data,
    test_data;
    dataname,
    output_path=output_path,
    params_path=params_path,
    pretrained_path=pretrained_path,
    retrain=false,
    epochs=100,
    n_hidden=16,
    activation=Flux.swish,
    builder=MLJFlux.MLP(
        hidden=(n_hidden, n_hidden, n_hidden),
        σ=activation
    ),
    n_ens=5,
    𝒟x=Normal(),
    sampling_batch_size=50,
    α=[1.0, 1.0, 1e-1],
    verbosity=10,
    sampling_steps=30,
    use_ensembling=false,
    coverage=.95,
    λ₁=0.25,
    λ₂ = 0.75,
    λ₃ = 0.75,
    opt=Flux.Optimise.Descent(0.01),
    use_class_loss=false,
    use_variants=true,
    n_individuals=25,
    generators=nothing,
)   

    # SETUP ----------

    # Data
    X, y = CounterfactualExplanations.DataPreprocessing.unpack_data(counterfactual_data)
    X = table(permutedims(X))
    labels = counterfactual_data.output_encoder.labels
    input_dim, n_obs = size(counterfactual_data.X)
    output_dim = length(unique(labels))
    save_name = replace(lowercase(dataname), " " => "_")

    # Model parameters:
    batch_size = minimum([Int(round(n_obs / 10)), 128])
    sampling_batch_size = isnothing(sampling_batch_size) ? batch_size : sampling_batch_size
    _loss = Flux.Losses.crossentropy                # loss function
    _finaliser = Flux.softmax                       # finaliser function

    # JEM parameters:
    𝒟y = Categorical(ones(output_dim) ./ output_dim)
    sampler = ConditionalSampler(
        𝒟x, 𝒟y,
        input_size=(input_dim,),
        batch_size=sampling_batch_size,
    )

    # MODELS ----------

    # Simple MLP:
    mlp = NeuralNetworkClassifier(
        builder=builder,
        epochs=epochs,
        batch_size=batch_size,
        finaliser=_finaliser,
        loss=_loss,
    )

    # Deep Ensemble:
    mlp_ens = EnsembleModel(model=mlp, n=n_ens)

    # Joint Energy Model:
    jem = JointEnergyClassifier(
        sampler;
        builder=builder,
        epochs=epochs,
        batch_size=batch_size,
        finaliser=_finaliser,
        loss=_loss,
        jem_training_params=(
            α=α, verbosity=verbosity,
        ),
        sampling_steps=sampling_steps
    )

    # Deep Ensemble of Joint Energy Models:
    jem_ens = EnsembleModel(model=jem, n=n_ens)

    # Dictionary of models:
    if !use_ensembling
        models = Dict(
            "MLP" => mlp,
            "JEM" => jem,
        )
    else
        models = Dict(
            "MLP" => mlp,
            "MLP Ensemble" => mlp_ens,
            "JEM" => jem,
            "JEM Ensemble" => jem_ens,
        )
    end

    # TRAINING ----------
    function _train(model, X=X, y=labels; cov=coverage, method=:simple_inductive, mod_name="model")
        conf_model = conformal_model(model; method=method, coverage=cov)
        mach = machine(conf_model, X, y)
        @info "Begin training $mod_name."
        fit!(mach)
        @info "Finished training $mod_name."
        M = ECCCo.ConformalModel(mach.model, mach.fitresult)
        return M
    end
    if retrain
        @info "Retraining models."
        model_dict = Dict(mod_name => _train(model; mod_name=mod_name) for (mod_name, model) in models)
        Serialization.serialize(joinpath(output_path, "$(save_name)_models.jls"), model_dict)
    else
        @info "Loading pre-trained models."
        model_dict = Serialization.deserialize(joinpath(pretrained_path, "$(save_name)_models.jls"))
    end

    params = DataFrame(
        Dict(
            :n_obs => Int.(round(n_obs/10)*10),
            :epochs => epochs,
            :batch_size => batch_size,
            :n_hidden => n_hidden,
            :n_layers => length(model_dict["MLP"].fitresult[1][1])-1,
            :activation => string(activation),
            :n_ens => n_ens,
            :lambda => string(α[3]),
            :jem_sampling_steps => jem.sampling_steps,
            :sgld_batch_size => sampler.batch_size,
            :dataname => dataname,
        )
    )
    CSV.write(joinpath(params_path, "$(save_name)_model_params.csv"), params)

    measure = Dict(
        :f1score => multiclass_f1score,
        :acc => accuracy,
        :precision => multiclass_precision
    )
    model_performance = DataFrame()
    for (mod_name, model) in model_dict
        # Test performance:
        _perf = CounterfactualExplanations.Models.model_evaluation(model, test_data, measure=collect(values(measure)))
        _perf = DataFrame([[p] for p in _perf], collect(keys(measure)))
        _perf.mod_name .= mod_name
        _perf.dataname .= dataname
        model_performance = vcat(model_performance, _perf)
    end
    Serialization.serialize(joinpath(output_path, "$(save_name)_model_performance.jls"), model_performance)
    CSV.write(joinpath(output_path, "$(save_name)_model_performance.csv"), model_performance)
    @info "Model performance:"
    println(model_performance)
    
    # COUNTERFACTUALS ----------
    
    @info "Begin benchmarking counterfactual explanations."
    Λ = [λ₁, λ₂, λ₃]

    generator_params = DataFrame(
        Dict(
            :λ1 => λ₁,
            :λ2 => λ₂,
            :λ3 => λ₃,
            :opt => string(typeof(opt)),
            :eta => opt.eta,
            :dataname => dataname,
        )
    )
    CSV.write(joinpath(params_path, "$(save_name)_generator_params.csv"), generator_params)

    # Benchmark generators:
    if !isnothing(generators)
        generator_dict = generators
    elseif use_variants
        generator_dict = Dict(
            "Wachter" => WachterGenerator(λ=λ₁, opt=opt),
            "REVISE" => REVISEGenerator(λ=λ₁, opt=opt),
            "Schut" => GreedyGenerator(),
            "ECCCo" => ECCCoGenerator(λ=Λ, opt=opt, use_class_loss=use_class_loss),
            "ECCCo (no CP)" => ECCCoGenerator(λ=[λ₁, 0.0, λ₃], opt=opt, use_class_loss=use_class_loss),
            "ECCCo (no EBM)" => ECCCoGenerator(λ=[λ₁, λ₂, 0.0], opt=opt, use_class_loss=use_class_loss),
        )
    else
        generator_dict = Dict(
            "Wachter" => WachterGenerator(λ=λ₁, opt=opt),
            "REVISE" => REVISEGenerator(λ=λ₁, opt=opt),
            "Schut" => GreedyGenerator(),
            "ECCCo" => ECCCoGenerator(λ=Λ, opt=opt, use_class_loss=use_class_loss),
        )
    end

    # Measures:
    measures = [
        CounterfactualExplanations.distance,
        ECCCo.distance_from_energy,
        ECCCo.distance_from_targets,
        CounterfactualExplanations.Evaluation.validity,
        CounterfactualExplanations.Evaluation.redundancy,
        ECCCo.set_size_penalty
    ]

    bmks = []
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
    CSV.write(joinpath(output_path, "$(save_name)_benchmark.csv"), bmk())

end