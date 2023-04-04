using CounterfactualExplanations.Objectives

"Constructor for `CCEGenerator`."
function CCEGenerator(; λ::Union{AbstractFloat,Vector{<:AbstractFloat}}=[0.1, 1.0], κ::Real=0.0, temp::Real=0.05, kwargs...)
    function _set_size_penalty(ce::AbstractCounterfactualExplanation)
        return CCE.set_size_penalty(ce; κ=κ, temp=temp)
    end
    _penalties = [Objectives.distance_l2, _set_size_penalty]
    λ = λ isa AbstractFloat ? [0.0, λ] : λ
    return Generator(; penalty=_penalties, λ=λ, kwargs...)
end