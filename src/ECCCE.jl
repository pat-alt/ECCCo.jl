module ECCCE

using CounterfactualExplanations
import MLJModelInterface as MMI

include("model.jl")
include("penalties.jl")
include("losses.jl")
include("generator.jl")
include("sampling.jl")

export ECCCEGenerator, EnergySampler, set_size_penalty, distance_from_energy

end