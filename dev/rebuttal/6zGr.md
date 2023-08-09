Thank you! In this individual response, we will refer back to the main points discussed in the global response where relevant and discuss any other specific points the reviewer has raised. Below we will go through individual points where quotations trace back to reviewer remarks.

#### Low plausibility (real-world data)

> "The major weakness of this work is that plausibility for non-JEM-based classifiers is very low on 'real-world' datasets (Table 2)."

As we argue in **Point 3** (and to some extent also **Point 2**) of the global rebuttal, we believe that this should not be seen as a weakness at all:

- Conditional on high fidelity, plausibility hinges on the quality of the underlying model. 
- Subpar outcomes can therefore be understood as a signal that the model needs to be improved. 

As noted in the global rebuttal, we aim to make this intuition even clearer in the paper. 

#### Visual quality (MNIST)

> "[...] visual quality of generated counterfactuals seems to be low. [Results] hint to low diversity of generated counterfactuals."

Again, we kindly point to the global rebuttal (**Point 2** and **Point 3**) in this context. Additionally, we note the following:

- The visual quality and diversity of the counterfactuals (Fig. 6 in suppl.) seems to faithfully represent generative property of the model (Fig. 6 in suppl.).
- If diversity is crucial, our implementation is fully compatible with adding additional diversity penalties as in DiCE (Mothilal et al., 2019).

We will discuss this more thoroughly in the paper. 

#### Closeness desideratum

> "ECCCos seems to generate counterfactuals that heavily change the initial image [...] thereby violating the closeness desideratum."

- We would look at this as the price you have to pay for faithfulness and plausibility.
  - Concerning faithfulness, large perturbations in the case of MNIST, for example, seem to reflect the fact that the underlying model is sensitive to perturbations across the entire image, even though the images are very sparse. We would argue that this is an undesirable property of the model, not the explanation. 
  - Concerning plausibility, larger perturbations are typically necessary to move counterfactuals not simply across the decision boundary, but into dense areas of the target domain. Thus, REVISE, for example, is also often associated with larger perturbations.
- This tradeoff can be governed through penalty strengths: if closeness is a high priority, simply increase the relative size of $\lambda_1$ in Equation (5).

We are happy to highlight this tradeoff in section 7. 

#### Datasets

> "The experiments are only conducted on small-scale datasets."

In short, we have relied on illustrative datasets commonly used in similar studies. Please refer to our global rebuttal (in particular **Point 1**) for additional details.

#### Conformal Prediction (ablation)

> "[...] it is unclear if conformal prediction is actually required for ECCCos."

Please refer to **Point 4** in the global rebuttal.

#### Bias towards faithfulness

> "Experimental results for faithfulness are biased since (un)faithfulness is already used during counterfactual optimization as regularizer."

- This is true and we are transparent about this in the paper (line 320 to 322). 
- ECCCo is intentionally biased towards faithfulness in the same way that Wachter is intentionally biased towards minimal perturbations. 

We are happy to make this point more explicit in section 7. 

#### Other questions

Finally, let us try to answer the specific questions that were raised:

- In line 178 we (belatedly) mention that the L1 Norm is our default choice for dist$(\cdot)$. We realise now that it's not obvious that this also applies to Equations 3 and 4 and will fix that. Note that we also experimented with other distance/similarity metrics, but found the differences in outcomes to be small enough to consistently rely on L1 for its sparsity-inducing properties. 
- $f$ by default just rescales the input data. GMSC data is standardized and MNIST images are rescaled to $[-1,1]$ (mentioned in Appendix D, lines 572-576, but maybe this indeed belongs in the body). $f^{-1}$ is simply the inverse transformation. Synthetic data is not rescaled. We still explicitly mention $f$ here to stay consistent with the generalised notation in Equation (1). For example, $f$/$f^{-1}$ could just as well be a compression/decompression or an encoder/decoder as in REVISE.
- In all of our experiments we set $\alpha=0.05$ (90\% target coverage) and $\kappa=1$ to avoid penalising sets of size one. We should add this to Appendix D, thanks for flagging. Note that we did experiment with these parameter choices, but as we point out in the paper, more work is needed to better understand the role of Conformal Prediction in this context. 
- I have just run ECCCo and Wachter for a single MNIST digit on my machine (no GPU) using default parameters from the experiment:
    - ECCCo: `4.065607 seconds (4.34 M allocations: 1.011 GiB, 7.62\% gc time)`. 
    - Wachter: `1.899047 seconds (2.16 M allocations: 343.889 MiB, 4.59\% gc time, 74.80\% compilation time)`.
  
    This is not performance-optimized code and the bulk of the runtime and allocation is driven by sampling through SGLD. Note that while in our experiments we chose to resample for each individual counterfactual explanation, in practice, sampling could be done once for a given dataset. In any case, the computational burden should typically be lower than the overhead involved in training a sufficiently expressive VAE for REVISE, for example. 

We also thank the reviewer for their suggestions and will take these on board.