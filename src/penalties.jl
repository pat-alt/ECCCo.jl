using ChainRules: ignore_derivatives
using CounterfactualExplanations: get_target_index
using Distances
using Flux
using LinearAlgebra: norm
using Statistics: mean

"""
    set_size_penalty(ce::AbstractCounterfactualExplanation)

Penalty for smooth conformal set size.
"""
function set_size_penalty(
    ce::AbstractCounterfactualExplanation; 
    κ::Real=1.0, temp::Real=0.1, agg=mean
)

    _loss = 0.0

    conf_model = ce.M.model
    fitresult = ce.M.fitresult
    X = CounterfactualExplanations.decode_state(ce)
    _loss = map(eachslice(X, dims=ndims(X))) do x
        x = ndims(x) == 1 ? x[:,:] : x
        if target_probs(ce, x)[1] >= 0.5
            l = ConformalPrediction.ConformalTraining.smooth_size_loss(
                conf_model, fitresult, x';
                κ=κ,
                temp=temp
            )[1]
        else 
            l = 0.0
        end
        return l
    end
    _loss = agg(_loss)

    return _loss

end

function energy_delta(
    ce::AbstractCounterfactualExplanation;
    n::Int=50, niter=500, from_buffer=true, agg=mean,
    choose_lowest_energy=true,
    choose_random=false,
    nmin::Int=25,
    return_conditionals=false,
    reg_strength=0.1,
    kwargs...
)

    # nmin = minimum([nmin, n])

    # @assert choose_lowest_energy ⊻ choose_random || !choose_lowest_energy && !choose_random "Must choose either lowest energy or random samples or neither."

    # conditional_samples = []
    # ignore_derivatives() do
    #     _dict = ce.params
    #     if !(:energy_sampler ∈ collect(keys(_dict)))
    #         _dict[:energy_sampler] = ECCCo.EnergySampler(ce; niter=niter, nsamples=n, kwargs...)
    #     end
    #     eng_sampler = _dict[:energy_sampler]
    #     if choose_lowest_energy
    #         nmin = minimum([nmin, size(eng_sampler.buffer)[end]])
    #         xmin = ECCCo.get_lowest_energy_sample(eng_sampler; n=nmin)
    #         push!(conditional_samples, xmin)
    #     elseif choose_random
    #         push!(conditional_samples, rand(eng_sampler, n; from_buffer=from_buffer))
    #     else
    #         push!(conditional_samples, eng_sampler.buffer)
    #     end
    # end
    conditional_samples = []
    ignore_derivatives() do 
        _dict = ce.params
        if !(:energy_sampler ∈ collect(keys(_dict)))
            _dict[:energy_sampler] = ECCCo.EnergySampler(ce; niter=niter, nsamples=n, kwargs...)
        end
        eng_sampler = _dict[:energy_sampler]
        generate_samples!(eng_sampler, ce.num_counterfactuals, get_target_index(ce.data.y_levels, ce.target); niter=niter)
        xsampled = eng_sampler.buffer[:,(end-ce.num_counterfactuals+1):end]
        push!(conditional_samples, xsampled)
    end

    xgenerated = conditional_samples[1]                         # conditional samples
    xproposed = CounterfactualExplanations.decode_state(ce)     # current state
    t = get_target_index(ce.data.y_levels, ce.target)
    E(x) = -logits(ce.M, x)[t,:]                                # negative logits for target class

    # Generative loss:
    gen_loss = E(xproposed) .- E(xgenerated)
    gen_loss = reduce((x, y) -> x + y, gen_loss) / length(gen_loss)                  # aggregate over samples

    # Regularization loss:
    reg_loss = E(xgenerated).^2 .+ E(xproposed).^2
    reg_loss = reduce((x, y) -> x + y, reg_loss) / length(reg_loss)                  # aggregate over samples

    if !return_conditionals
        return gen_loss + reg_strength * reg_loss
    else
        return conditional_samples[1]
    end

end

function distance_from_energy(
    ce::AbstractCounterfactualExplanation;
    n::Int=50, niter=500, from_buffer=true, agg=mean, 
    choose_lowest_energy=true,
    choose_random=false,
    nmin::Int=25,
    return_conditionals=false,
    kwargs...
)

    _loss = 0.0
    nmin = minimum([nmin, n])

    @assert choose_lowest_energy ⊻ choose_random || !choose_lowest_energy && !choose_random "Must choose either lowest energy or random samples or neither."

    conditional_samples = []
    ignore_derivatives() do
        _dict = ce.params
        if !(:energy_sampler ∈ collect(keys(_dict)))
            _dict[:energy_sampler] = ECCCo.EnergySampler(ce; niter=niter, nsamples=n, kwargs...)
        end
        eng_sampler = _dict[:energy_sampler]
        if choose_lowest_energy
            nmin = minimum([nmin, size(eng_sampler.buffer)[end]])
            xmin = ECCCo.get_lowest_energy_sample(eng_sampler; n=nmin)
            push!(conditional_samples, xmin)
        elseif choose_random
            push!(conditional_samples, rand(eng_sampler, n; from_buffer=from_buffer))
        else
            push!(conditional_samples, eng_sampler.buffer)
        end
    end

    _loss = map(eachcol(conditional_samples[1])) do xsample
        distance_l1(ce; from=xsample, agg=agg)
    end
    _loss = reduce((x,y) -> x + y, _loss) / n       # aggregate over samples

    if return_conditionals
        return conditional_samples[1]
    end
    return _loss

end

function distance_from_targets(
    ce::AbstractCounterfactualExplanation;
    n::Int=1000, agg=mean,
    n_nearest_neighbors::Union{Int,Nothing}=nothing,
)
    target_idx = ce.data.output_encoder.labels .== ce.target
    target_samples = ce.data.X[:,target_idx] |>
        X -> X[:,rand(1:end,n)]
    x′ = CounterfactualExplanations.counterfactual(ce)
    loss = map(eachslice(x′, dims=ndims(x′))) do x
        Δ = map(eachcol(target_samples)) do xsample
            norm(x - xsample, 1)
        end
        if n_nearest_neighbors != nothing
            Δ = sort(Δ)[1:n_nearest_neighbors]
        end
        return mean(Δ)
    end
    loss = agg(loss)[1]

    return loss

end

