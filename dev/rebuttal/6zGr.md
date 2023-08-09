Thank you! In this individual response, we will refer back to the main points discussed in the global response where relevant and discuss any other specific points the reviewer has raised.

We largely agree with some of the weaknesses pointed out and will address these below. To start off, we want to address what is being described as the "major weakness" of our paper: the remark that our results indicate that ECCCo does not directly help with plausibility for "weaker" models. That is mostly correct, but let us try to make the case for why this should not be considered as a weakness at all: 

- We would argue that this is a desirable property of ECCCo, if our priority is to understand model behaviour: lower plausibility conditional on high fidelity implies that the model itself has learned implausible explanations for the data (we point to this in lines 237-239, 305-307, 322-324, 340-342, ...).
- We think that this characteristic is desirable for the following reasons: 
    - For practitioners/researchers this is valuable information indicating that despite good predictive performance, the learned posterior density $p_{\theta}(\mathbf{x}|\mathbf{y^{+}})$ is high in regions of the input domain that are implausible (in the sense of Def 2.1, i.e. the corresponding true density $p(\mathbf{x}|\mathbf{y^{+}})$ is low in those same regions).
    - Instead of using surrogate-aided counterfactual search engines to sample those counterfactuals from $p_{\theta}(\mathbf{x}|\mathbf{y^{+}})$ that are indeed plausible, we would are that the next point of action in such cases should generally be to improve the model.
    - We agree that this places an additional burden on researchers/practitioners, but that does not render ECCCo impractical. In situations where providing actionable recourse is an absolute priority, practitioners can always resort to REVISE and related tools in the short term. Major discrepancies between ECCCo and surrogate-aided tools should then at the very least signal to researchers/practitioners, that the underlying model needs to be improved in the medium-term. 

To conclude, we believe that ECCCo and derivative works have the potential to help us identify models that have learned implausible explanations for the data and improve upon that. To illustrate this, we have relied on gradually improving our classifiers through ensembling and joint energy modelling. We chose to focus on JEMs because:

- ECCCo itself uses ideas underlying JEMs. 
- JEMs have been shown to have multiple desirable properties including robustness and good predictive uncertainty quantification. Based on the previous literature on counterfactuals, these model properties should generally positively correlate with the plausibility of counterfactuals (and our findings seem to confirm this).

We agree with the criticism that the "visual quality of generated counterfactuals seems to be low" and we observe "diversity of generated counterfactuals", but:

- The visual quality and diversity of the counterfactuals (Fig. 6 in suppl.) seems to faithfully represent generative property of the model (Fig. 6 in suppl.).
- If diversity is crucial, our implementation is fully compatible with adding additional diversity penalties as in DiCE (Mothilal et al., 2019).

We do agree with the criticism that our work could benefit from including other classes of models that can be expected to learn more plausible explanations than our small MLPs (ResNet, CNN, Transformer, adversarially-trained networks, Bayesian NNs, ...). We also agree that additional, more complex datasets need to be consulted in this context and we intend to tackle this in future work. 

- We would argue that these are limitations of our work, but not necessarily weaknesses. As we have argued elsewhere, this work was limited in both scope and size. Including more experiments would mean compromising on explanations/elaborations with regard to our setup that to our feeling are critical.
- These limitations could be made more explicit in a camera-ready version of the paper, should it come to that.

Finally, let us try to answer the specific questions that were raised:

- In line 178 we (belatedly) mention that the L1 Norm is our default choice for dist$(\cdot)$. We realise now that it's not obvious that this also applies to Equations 3 and 4 and will fix that. Note that we also experimented with other distance/similarity metrics, but found the differences in outcomes to be small enough to consistently rely on L1 for its sparsity-inducing properties. 
- $f$ by default just rescales the input data. GMSC data is standardized and MNIST images are rescaled to $[-1,1]$ (mentioned in Appendix D, lines 572-576, but maybe this indeed belongs in the body). $f^{-1}$ is simply the inverse transformation. Synthetic data is not rescaled. We still explicitly mention $f$ here to stay consistent with the generalised notation in Equation (1). For example, $f$/$f^{-1}$ could just as well be a compression/decompression or an encoder/decoder as in REVISE.
- In all of our experiments we set $\alpha=0.05$ (90\% target coverage) and $\kappa=1$ to avoid penalising sets of size one. We should add this to Appendix D, thanks for flagging. Note that we did experiment with these parameter choices, but as we point out in the paper, more work is needed to better understand the role of Conformal Prediction in this context. 
- I have just run ECCCo and Wachter for a single MNIST digit on my machine (no GPU) using default parameters from the experiment:
    - ECCCo: `4.065607 seconds (4.34 M allocations: 1.011 GiB, 7.62\% gc time)`. 
    - Wachter: `1.899047 seconds (2.16 M allocations: 343.889 MiB, 4.59\% gc time, 74.80\% compilation time)`.
  
    This is not performance-optimized code and the bulk of the runtime and allocation is driven by sampling through SGLD. Note that while in our experiments we chose to resample for each individual counterfactual explanation, in practice, sampling could be done once for a given dataset. In any case, the computational burden should typically be lower than the overhead involved in training a sufficiently expressive VAE for REVISE, for example. 

We also thank the reviewer for their suggestions and will take these on board. The "ECCCo" vs. "ECCCos" story actually caused us some headache: we eventually tried to highlight that *ECCCo* relates to the generator, hence shown in italic consistent with the other generators. Perhaps it makes more sense to drop the distinction between the two.