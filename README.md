# ECCCo

![](www/poc_gradient_fields.png)

*Energy-Constrained Counterfactual Explanations.*

This work is currently undergoing peer review. This README is therefore only meant to provide reviewers access to the code base. The code base will be made public after the review process.

## Inspecting the Package Code

This code base is structured as a Julia package. The package code is located in the `src/` folder.

## Inspecting the Results

All results have been carefully reported either in the paper itself or in the supplementary material. In addition, we have released our results as binary files. These will be made publicly available after the review process. 

## Reproducing the Results

To reproduce the results, you need to install the package, which will automatically install all dependencies. Since the package is not publicly registered you will need to install it from source. To do so, start `Julia` from within this folder by executing `julia --project` from the command line and then enter `Pkg` mode by typing `]`. Then execute the following commands:

```julia
(ECCCo) pkg> resolve
(ECCCo) pkg> instantiate
```

Next, you may need to activate, resolve and instantiate the environment in `experiments`:

```julia
(ECCCo) pkg> activate experiments/
  Activating project at `~/code/ECCCo.jl/experiments`
(experiments) pkg> resolve
(experiments) pkg> instantiate
```

After that is done, you can exit Julia and proceed below.

### Sequential

The `experiments/` folder contains separate Julia scripts for each dataset and a [run_experiments.jl](experiments/run_experiments.jl) that calls the individual scripts. You can either run these scripts inside a Julia session or just use the command line to execute them as described in the following.

To run the experiment for a single dataset, (e.g. `linearly_separable`) simply run the following command:

```shell
julia --project=experiments/ experiments/run_experiments.jl -- data=linearly_separable
```

We use the following identifiers:

- `linearly_separable` (*Linearly Separable* data)
- `moons` (*Moons* data)
- `circles` (*Circles* data)
- `california_housing` (*California Housing* data)
- `gmsc` (*GMSC* data)
- `german_credit` (*German Credit* data)
- `mnist` (*MNIST* data)
- `fmnist` (*Fashion MNIST* data)

To run experiments for multiple datasets at once simply separate them with a comma `,`

```shell
julia --project=experiments/ experiments/run_experiments.jl -- data=linearly_separable,moons,circles
```

To run all experiments at once you can instead run

```shell
julia --project=experiments/ experiments/run_experiments.jl -- run-all
```

Pre-trained versions of all of our black-box models have been archived as `Pkg` [artifacts](https://pkgdocs.julialang.org/v1/artifacts/) and are used by default. Should you wish to retrain the models as well, simply use the `retrain` flag as follows:

```shell
julia --project=experiments experiments/run_experiments.jl -- retrain data=linearly_separable
```

### Multi-threading

To use multi-threading, proceed as follows:

```shell
julia --threads 16 --project=experiments experiments/run_experiments.jl -- data=linearly_separable threaded
```

### Multi-Processing

To use multi-processing, proceed as follows:

```shell
mpiexecjl --project=experiments -n 4 julia experiments/run_experiments.jl -- data=linearly_separable mpi
```

Multi-processing and multi-threading can be combined:

```shell
mpiexecjl --project=experiments -n 4 julia experiments/run_experiments.jl -- data=linearly_separable threaded mpi
```

## Reproducing Figures

To recreate the exact figures shown in the main paper you can use two notebooks:

- `experiments/notebooks/figure2.qmd`: Figure 2 (gradient fields)
- `experiments/notebooks/figure1and3.qmd`: Figures 1 and 3 (MNIST examples)