module ECCCo

using CounterfactualExplanations
import MLJModelInterface as MMI

include("model.jl")
include("sampling.jl")
include("penalties.jl")
include("losses.jl")
include("generator.jl")

export CCEGenerator, ECCCoGenerator, EnergySampler
export get_lowest_energy_sample
export set_size_penalty, distance_from_energy

end