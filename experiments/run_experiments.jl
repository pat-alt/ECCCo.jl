include("setup.jl")

# User inputs:
if ENV["DATANAME"] == "all"
    datanames = ["linearly_separable", "moons", "circles", "mnist", "gmsc"]
else
    datanames = [ENV["DATANAME"]]
end

# Linearly Separable
if "linearly_separable" in datanames
    @info "Running linearly separable experiment."
    include("linearly_separable.jl")
end

# Moons
if "moons" in datanames
    @info "Running moons experiment."
    include("moons.jl")
end

# Circles
if "circles" in datanames
    @info "Running circles experiment."
    include("circles.jl")
end

# MNIST
if "mnist" in datanames
    @info "Running MNIST experiment."
    include("mnist.jl")
end

# GMSC
if "gmsc" in datanames
    @info "Running GMSC experiment."
    include("gmsc.jl")
end
