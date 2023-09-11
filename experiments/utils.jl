using CounterfactualExplanations.Parallelization: ThreadsParallelizer

function is_multi_processed(parallelizer::Union{Nothing,AbstractParallelizer})
    if isnothing(parallelizer) || isa(parallelizer, ThreadsParallelizer)
        return false
    else
        return isa(parallelizer, Base.get_extension(CounterfactualExplanations, :MPIExt).MPIParallelizer)
    end
end

is_multi_processed(exper::Experiment) = is_multi_processed(exper.parallelizer)