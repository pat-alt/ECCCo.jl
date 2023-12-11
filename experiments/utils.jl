using CounterfactualExplanations.Parallelization: ThreadsParallelizer
using Distributions: Uniform
using Flux
using LinearAlgebra: norm
using Statistics: mean, std

function is_multi_processed(parallelizer::Union{Nothing,AbstractParallelizer})
    if isnothing(parallelizer) || isa(parallelizer, ThreadsParallelizer)
        return false
    else
        return isa(
            parallelizer,
            Base.get_extension(CounterfactualExplanations, :MPIExt).MPIParallelizer,
        )
    end
end

is_multi_processed(exper::Experiment) = is_multi_processed(exper.parallelizer)

function min_max_scale(x::AbstractArray)
    x_norm = (x .- minimum(x)) ./ (maximum(x) - minimum(x))
    x_norm = replace(x_norm, NaN => 0.0)
    return x_norm
end

function standardize(x::AbstractArray)
    x_norm = (x .- sum(x) / length(x)) ./ std(x)
    x_norm = replace(x_norm, NaN => 0.0)
    return x_norm
end

function get_learning_rate(opt::Flux.Optimise.AbstractOptimiser)
    if hasfield(typeof(opt), :eta)
        return opt.eta
    elseif hasfield(typeof(opt), :os)
        _os = opt.os
        opt = _os[findall([:eta in fieldnames(typeof(o)) for o in _os])][1]
        return opt.eta
    else 
        throw(ArgumentError("Cannot find learning rate."))
    end
end