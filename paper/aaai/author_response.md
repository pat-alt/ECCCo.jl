# Author Response

1. Applied to additional commonly used tabular real-world datasets
2. Energy delta
   1. Better results, in particular for image data
   2. No longer biased (addressing reviewer concern)
3. Counterfactual explanations do not scale well to high-dimensional input data
   1. We have added native support for multi-processing and multi-threading
   2. We have run more extensive experiments including fine-tuning hyperparameter choices
4. We have revisited the mathematical notation.
5. We have moved the introduction of conformal prediction forward and added more detail in line with reviewer feedback.
6. We have extended the limitations section. 
7. Distance metric
   1. We have revisited the distance metrics and decided to use the L2 Norm for plausibility and faithfulness
   2. Orginially, we used the L1 Norm in line with how the the closeness criterium is commonly evaluated. But in this context the L1 Norm implicitly addresses the desire for sparsity.
   3. In the case of image data, we also used cosine distance.
   