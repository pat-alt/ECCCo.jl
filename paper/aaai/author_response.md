# Author Response

1. Applied to additional commonly used tabular real-world datasets
2. Constraining energy directly
   1. Better results across the board, in particular for image data
   2. Derived from JEM loss function -> more theoretically grounded
   3. No sampling overhead.
   4. Energy does not depend on differentiability.
   5. Benchmarks no longer biased with respect to unfaithfulness metric (addressing reviewer concern).
3. Counterfactual explanations do not scale well to high-dimensional input data
   1. We have added native support for multi-processing and multi-threading.
   2. We have run more extensive experiments including fine-tuning hyperparameter choices.
   3. For image data we use PCA to map counterfactuals to a smaller dimenionsional latent space, which not only reduces costs of gradient computations but also leads to higher plausibility.
   4. PCA is much less costly and interventionist than a VAE: pricipal component merely represent variation in the data; nothing else about the data is learned by the surrogate. 
      1. ECCCo-$\Delta$ (latent) remains faithful, although not as faithful as standard ECCCo-$\Delta$.
4. We have revisited the mathematical notation.
5. We have moved the introduction of conformal prediction forward and added more detail in line with reviewer feedback.
6. We have extended the limitations section. 
7. Distance metric
   1. We have revisited the distance metrics and decided to use the L2 Norm for plausibility and faithfulness
   2. Orginially, we used the L1 Norm in line with how the the closeness criterium is commonly evaluated. But in this context the L1 Norm implicitly addresses the desire for sparsity.
   3. In the case of image data, we investigated various additional distance metrics:
      1. Cosine similarity
      2. Euclidean distance
      3. Ultimately we chose to rely on structural dissimilarity.
   