using CounterfactualExplanations.Objectives

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
    return Generator(; penalty=_penalties, λ=λ, kwargs...)
end

"Constructor for `ECECCCoGenerator`: Energy Constrained Conformal Counterfactual Explanation Generator."
function ECCCoGenerator(; 
    λ::Union{AbstractFloat,Vector{<:AbstractFloat}}=[0.2,0.4,0.4], 
    κ::Real=1.0, 
    temp::Real=0.1, 
    opt::Union{Nothing,Flux.Optimise.AbstractOptimiser}=nothing,
    kwargs...
)
    if isnothing(opt)
        opt = CounterfactualExplanations.Generators.Descent(0.1)
    end
    function _set_size_penalty(ce::AbstractCounterfactualExplanation)
        return ECCCo.set_size_penalty(ce; κ=κ, temp=temp)
    end
    _penalties = [Objectives.distance_l1, _set_size_penalty, ECCCo.distance_from_energy]
    λ = λ isa AbstractFloat ? [0.0, λ, λ] : λ
    return Generator(; penalty=_penalties, λ=λ, opt=opt, kwargs...)
end

"Constructor for `EnergyDrivenGenerator`."
function EnergyDrivenGenerator(; λ::Union{AbstractFloat,Vector{<:AbstractFloat}}=[0.1, 1.0], kwargs...)
    _penalties = [Objectives.distance_l2, ECCCo.distance_from_energy]
    λ = λ isa AbstractFloat ? [0.0, λ] : λ
    return Generator(; penalty=_penalties, λ=λ, kwargs...)
end

"Constructor for `TargetDrivenGenerator`."
function TargetDrivenGenerator(; λ::Union{AbstractFloat,Vector{<:AbstractFloat}}=[0.1, 1.0], kwargs...)
    _penalties = [Objectives.distance_l2, ECCCo.distance_from_targets]
    λ = λ isa AbstractFloat ? [0.0, λ] : λ
    return Generator(; penalty=_penalties, λ=λ, kwargs...)
end