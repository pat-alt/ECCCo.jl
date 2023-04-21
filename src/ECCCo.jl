module ECCCo

using CounterfactualExplanations
import MLJModelInterface as MMI

include("model.jl")
include("penalties.jl")
include("losses.jl")
include("generator.jl")
include("sampling.jl")

export CCEGenerator, ECCCoGenerator, EnergySampler
export set_size_penalty, distance_from_energy

end