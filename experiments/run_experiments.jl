include("setup_env.jl");

# User inputs:
if "run-all" in ARGS
    datanames = ["linearly_separable", "moons", "circles", "mnist", "fmnist", "gmsc"]
elseif any(contains.(ARGS, "data="))
    datanames = [ARGS[findall(contains.(ARGS, "data="))][1] |> x -> replace(x, "data=" => "")]
    datanames = replace.(split(datanames[1], ","), " " => "")
else
    @warn "No dataset specified, defaulting to all."
    datanames = ["linearly_separable", "moons", "circles", "mnist", "fmnist", "gmsc"]
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

# GMSC
if "gmsc" in datanames
    @info "Running GMSC experiment."
    include("gmsc.jl")
end

# MNIST
if "mnist" in datanames
    @info "Running MNIST experiment."
    include("mnist.jl")
end

if "fmnist" in datanames
    @info "Running Fashion-MNIST experiment."
    include("fmnist.jl")
end

if USE_MPI
    MPI.Finalize()
end

# if UPLOAD
#     @info "Uploading results."
#     generate_artifacts()
# end
