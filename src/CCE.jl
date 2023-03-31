module CCE

using CounterfactualExplanations
import MLJModelInterface as MMI

include("model.jl")
include("penalties.jl")
include("losses.jl")
include("generator.jl")
include("sampling.jl")
# include("ConformalGenerator.jl")

using MLJFlux
MLJFlux.reformat(X, ::Type{<:AbstractMatrix}) = permutedims(X)

end
