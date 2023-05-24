include("setup.jl")

# User inputs:
if ENV("DATANAME") == "all"
    datanames = ["linearly_separable", "moons", "circles", "mnist", "gmsc"]
else
    datanames = [ENV("DATANAME")]
end

# Linearly Separable
if "linearly_separable" in datanames
    include("linearly_separable.jl")
end

# Moons
if "moons" in datanames
    include("moons.jl")
end

# Circles
if "circles" in datanames
    include("circles.jl")
end

# MNIST
if "mnist" in datanames
    include("mnist.jl")
end

# GMSC
if "gmsc" in datanames
    include("gmsc.jl")
end
