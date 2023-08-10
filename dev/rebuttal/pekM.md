Thank you! In this individual response, we will refer back to the main points discussed in the global response where relevant and discuss any other specific points the reviewer has raised. Below we will go through individual points where quotations trace back to reviewer remarks.

### Mathematical notation

> "Some notions are lacking descriptions and explanations"

We will make a full pass over all notation, and improve where needed. 
### Conditional distribution

> "[...] the class-condition distribution $p(\mathbf{x}|\mathbf{y^{+}})$ is existed but unknown and learning this distribution is very challenging especially for structural data"

We do not see this as a weakness of our paper. Instead:

- While we agree that learning this distribution is not always trivial, we note that this task is at the very core of Generative Modelling and AI&mdash;a field that has recently enjoyed success, especially in the context of large unstructured data like images and language.
- Learning the generative task is also at the core of related approaches mentioned in the paper like REVISE: as we mention in line 89, the authors of REVISE "propose using a generative model such as a Variational Autoencoder (VAE)" to learn $p(\mathbf{x})$. We also point to other related approaches towards plausibility that all centre around learning the data-generating process of the inputs $X$ (lines 85 to 104).
- Learning $p(\mathbf{x}|\mathbf{y^{+}})$ should generally be easier than learning the unconditional distribution $p(\mathbf{x})$, because the information contained in labels can be leveraged in the latter case. 

We will revisit section 2 to clarify this.

### Implausibility metric

> "Additionally, the implausibility metric seems not general and rigorous [...]"

- We agree it is not perfect and speak to this in the paper (e.g. lines 297 to 299). But we think that it is an improved, more robust version of the metric that was previously proposed and used in the literature (lines 159 to 166). Nonetheless, we will make this limitation clearer also in section 7.
- The rule-based unary constraint metric proposed in Vo et al. (2023) looks interesting, but the paper will be presented for the first time at KDD in August 2023 and we were not aware of it at the time of writing. Thanks for bringing it to our attention, we will mention it in the same context in section 7. 

### Definiton of "faithfulness"

> "Faithfulness [...] can be understood as the validity and fidelity of counterfactual examples. [...] The definition 4.1 is fine but missing of the details of $p_{\theta}(\mathbf{x}|\mathbf{y^{+}})$. [...] However, it is not clear to me how to [...] use it in [SGLD]."

We wish to highlight a possible reviewer misunderstanding with regard to a fundamental take in our work:

- We argue extensively in sections 3 and 4 that faithfulness should *not* be understood simply as validity and fidelity. That is why we propose a definition of "faithfulness" that works with distributional quantities. 
- Specifically, we want to understand if counterfactuals are consistent with what the model has learned about the data, which is best expressed as $p_{\theta}(\mathbf{x}|\mathbf{y^{+}})$ (Def. 4.2).
- The role of SGLD in this context is described in some detail in Section 4.1 (lines 138 to 155) and additional explanations are provided in Appendix A.

We will revisit sections 3 and 4 of the paper to better explain this.

### Conformal Prediction (CP) 

CP in this context is mentioned both as a strength

> "conformal prediction for counter-factual explanation is interesting"

and a weakness

> "motivation of using Conformal Prediction (CP) is not convincing to me"

We reiterate our motivation here:

- As we explain in some detail (lines 180 to 193) the idea rests on the notion that Predictive Uncertainty estimates can be used to generate plausible counterfactuals as previous work has shown.
- Since CP is model-agnostic, we propose relying on it to relax restrictions that were previously placed on the class of classifiers (lines 183 to 189).
- CP does indeed produce prediction sets in the context of classification. That is why we work with a smooth version of the set size that is compatible with gradient-based counterfactual search, as we explain in some detail in lines 194 to 205 and also in Appendix B. 

Following the suggestion from reviewer 6zGr we will smoothen the introduction Conformal Prediction and better motivate it beforehand.

### Experiments

Please see **Points 1** and **4** of our global rebuttal.