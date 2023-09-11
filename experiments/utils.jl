is_multi_processed(exper::Experiment) = isa(exper.parallelizer, Base.get_extension(CounterfactualExplanations, :MPIExt).MPIParallelizer)

is_multi_processed(parallelizer::AbstractParallelizer) = isa(parallelizer, Base.get_extension(CounterfactualExplanations, :MPIExt).MPIParallelizer)