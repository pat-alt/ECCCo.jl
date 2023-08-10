using CounterfactualExplanations.Objectives
using CounterfactualExplanations.Generators: GradientBasedGenerator

"Constructor for `CCEGenerator`."
function CCEGenerator(; 
    λ::Union{AbstractFloat,Vector{<:AbstractFloat}}=[0.1, 1.0], 
    κ::Real=1.0, 
    temp::Real=0.1, 
    kwargs...
)
    function _set_size_penalty(ce::AbstractCounterfactualExplanation)
        return ECCCo.set_size_penalty(ce; κ=κ, temp=temp)
    end
    _penalties = [Objectives.distance_l2, _set_size_penalty]
    λ = λ isa AbstractFloat ? [0.0, λ] : λ
    return GradientBasedGenerator(; penalty=_penalties, λ=λ, kwargs...)
end

"Constructor for `ECECCCoGenerator`: Energy Constrained Conformal Counterfactual Explanation Generator."
function ECCCoGenerator(; 
    λ::Union{AbstractFloat,Vector{<:AbstractFloat}}=[0.2,0.4,0.4], 
    κ::Real=1.0, 
    temp::Real=0.1, 
    opt::Union{Nothing,Flux.Optimise.AbstractOptimiser}=nothing,
    use_class_loss::Bool=false,
    nsamples::Int=50,
    nmin::Int=25,
    use_energy_delta::Bool=false,
    kwargs...
)

    # Default optimiser
    if isnothing(opt)
        opt = CounterfactualExplanations.Generators.Descent(0.1)
    end

    # Loss function
    if use_class_loss
        loss_fun(ce::AbstractCounterfactualExplanation) = conformal_training_loss(ce; temp=temp)
    else
        loss_fun = nothing
    end

    # Set size penalty
    function _set_size_penalty(ce::AbstractCounterfactualExplanation)
        return ECCCo.set_size_penalty(ce; κ=κ, temp=temp)
    end

    # Energy penalty
    function _energy_penalty(ce::AbstractCounterfactualExplanation)
        if use_energy_delta
            return ECCCo.energy_delta(ce; n=nsamples, nmin=nmin, kwargs...)
        else
            return ECCCo.distance_from_energy(ce; n=nsamples, nmin=nmin, kwargs...)
        end
    end

    _penalties = [Objectives.distance_l1, _set_size_penalty, _energy_penalty]
    λ = λ isa AbstractFloat ? [0.0, λ, λ] : λ

    # Generator
    return GradientBasedGenerator(; loss=loss_fun, penalty=_penalties, λ=λ, opt=opt, kwargs...)
end

"Constructor for `EnergyDrivenGenerator`."
function EnergyDrivenGenerator(; λ::Union{AbstractFloat,Vector{<:AbstractFloat}}=[0.1, 1.0], kwargs...)
    _penalties = [Objectives.distance_l2, ECCCo.distance_from_energy]
    λ = λ isa AbstractFloat ? [0.0, λ] : λ
    return GradientBasedGenerator(; penalty=_penalties, λ=λ, kwargs...)
end

"Constructor for `TargetDrivenGenerator`."
function TargetDrivenGenerator(; λ::Union{AbstractFloat,Vector{<:AbstractFloat}}=[0.1, 1.0], kwargs...)
    _penalties = [Objectives.distance_l2, ECCCo.distance_from_targets]
    λ = λ isa AbstractFloat ? [0.0, λ] : λ
    return GradientBasedGenerator(; penalty=_penalties, λ=λ, kwargs...)
end