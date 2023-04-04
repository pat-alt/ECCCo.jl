module CCE

using CounterfactualExplanations
import MLJModelInterface as MMI

include("model.jl")
include("penalties.jl")
include("losses.jl")
include("generator.jl")
include("sampling.jl")

using MLJFlux
MLJFlux.reformat(X, ::Type{<:AbstractMatrix}) = X'

export CCEGenerator, EnergySampler, set_size_penalty, distance_from_energy

end