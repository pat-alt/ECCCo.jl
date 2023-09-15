"An MLP builder that is more easily tunable."
mutable struct TuningBuilder <: MLJFlux.Builder
    n_hidden::Int
    n_layers::Int
end

"Outer constructor."
TuningBuilder(;n_hidden=32, n_layers=3) = TuningBuilder(n_hidden, n_layers)

function MLJFlux.build(nn::TuningBuilder, rng, n_in, n_out)
    hidden = ntuple(i -> nn.n_hidden, nn.n_layers)
    return MLJFlux.build(MLJFlux.MLP(hidden=hidden), rng, n_in, n_out)
end

"""
    tune_model(mod::Supervised, X, y; tuning_params::NamedTuple, kwargs...)

Tunes a model by performing a grid search over the parameters specified in `tuning_params`.
"""
function tune_model(mod::Supervised, X, y; tuning_params::NamedTuple, kwargs...)

    ranges = []

    for (k, v) in pairs(tuning_params)
        if k ∈ fieldnames(typeof(model))
            push!(ranges, range(mod, k, values=v))
        elseif k ∈ fieldnames(typeof(model.builder))
            push!(ranges, range(mod, :(builder.$(k)), values=v))
        elseif k ∈ fieldnames(typeof(model.optimiser))
            push!(ranges, range(mod, :(optimiser.$(k)), values=v))
        else
            error("Parameter $k not found in model, builder or optimiser.")
        end
    end
    
    self_tuning_mod = TunedModel(
        model=mod,
        range=ranges,
        kwargs...
    )

    mach = machine(self_tuning_mod, X, y)
    fit!(mach, verbosity=0)

    return mach

end