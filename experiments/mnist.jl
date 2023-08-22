function pre_process(x; noise::Float32=0.03f0)
    ϵ = Float32.(randn(size(x)) * noise)
    x += ϵ
    return x
end

# Training data:
n_obs = 10000
counterfactual_data = load_mnist(n_obs)
counterfactual_data.X = pre_process.(counterfactual_data.X)

# VAE (trained on full dataset):
using CounterfactualExplanations.Models: load_mnist_vae
vae = load_mnist_vae()
counterfactual_data.generative_model = vae

# Test data:
test_data = load_mnist_test()

# Models:
builder = MLJFlux.@builder Flux.Chain(
    Dense(n_in, n_hidden, activation),
    Dense(n_hidden, n_out),
)

# Generators:
eccco_generator = ECCCoGenerator(
    λ=[0.1,0.25,0.25], 
    temp=0.1, 
    opt=nothing,
    use_class_loss=true,
    nsamples=10,
    nmin=10,
)
Λ = eccco_generator.λ
generator_dict = Dict(
    "Wachter" => WachterGenerator(λ=Λ[1], opt=eccco_generator.opt),
    "REVISE" => REVISEGenerator(λ=Λ[1], opt=eccco_generator.opt),
    "Schut" =>  GreedyGenerator(η=2.0),
    "ECCCo" => eccco_generator,
)

# Run:
run_experiment!(
    counterfactual_data, test_data; dataname="MNIST",
    n_hidden = 128,
    activation = Flux.swish,
    builder = MLJFlux.@builder Flux.Chain(
        Dense(n_in, n_hidden, activation),
        Dense(n_hidden, n_out),
    ),
    𝒟x = Uniform(-1.0, 1.0),
    α = [1.0,1.0,1e-2],
    sampling_batch_size = 10,
    ssampling_steps=25,
    use_ensembling = true,
    generators = generator_dict,
)