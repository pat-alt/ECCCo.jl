using CounterfactualExplanations.Objectives
using CounterfactualExplanations.Generators: GradientBasedGenerator

"Constructor for `ECECCCoGenerator`: Energy Constrained Conformal Counterfactual Explanation Generator."
function ECCCoGenerator(; 
    λ::Union{AbstractFloat,Vector{<:AbstractFloat}}=[0.2,0.4,0.4], 
    κ::Real=1.0, 
    temp::Real=0.1, 
    opt::Union{Nothing,Flux.Optimise.AbstractOptimiser}=nothing,
    use_class_loss::Bool=false,
    use_energy_delta::Bool=false,
    nsamples::Union{Nothing,Int}=nothing,
    nmin::Union{Nothing,Int}=nothing,
    niter::Union{Nothing,Int}=nothing,
    reg_strength::Real=0.1,
    kwargs...
)

    # Default ECCCo parameters
    nsamples = isnothing(nsamples) ? 50 : nsamples
    nmin = isnothing(nmin) ? 25 : nmin
    niter = isnothing(niter) ? 500 : niter

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

    _energy_penalty =
        use_energy_delta ? (ECCCo.energy_delta, (n=nsamples, nmin=nmin, niter=niter, reg_strength=reg_strength)) : (ECCCo.distance_from_energy, (n=nsamples, nmin=nmin, niter=niter))

    _penalties = [
        (Objectives.distance_l1, []), 
        (ECCCo.set_size_penalty, (κ=κ, temp=temp)),
        _energy_penalty,
    ]
    λ = λ isa AbstractFloat ? [0.0, λ, λ] : λ

    # Generator
    return GradientBasedGenerator(; loss=loss_fun, penalty=_penalties, λ=λ, opt=opt, kwargs...)
end