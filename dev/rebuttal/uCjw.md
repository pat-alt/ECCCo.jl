Thank you! In this individual response, we will refer back to the main points discussed in the global response where relevant and discuss any other specific points the reviewer has raised. Below we will go through individual points where quotations trace back to reviewer remarks.

#### Data and models

> "[...] I still find the experiments with real-world data a bit limited. [...] The focus of the models being tested seems narrow."

Firstly, concerning the limited set of models and real-world datasets (Question 1 and Question 3), please refer to **Point 1** and **Point 2** in the global response, respectively. 

#### Generalisability

> "Is the ECCCos approach adaptable to a broad range of black-box models beyond those discussed?"

Our approach should generalise to any classifier that is differentiable with respect to inputs, consistent with other gradient-based counterfactual generators (Equation 1). Our actual implementation is currently compatible with neural networks trained in Julia and has experimental support for `torch` trained in either Python or R. Even though it is possible to generate counterfactuals for non-differentiable models, it is not immediately obvious to us how SGLD can be applied in this context. An interesting question for future research would be if other scalable and gradient-free methods can be used to sample from the conditional distribution learned by the model. 

#### Link to causality

> "There’s a broad literature on causal abstractions and causal model explanations that seems related."

This is an interesting thought. We would have to think about this more, but there is a possible link to the work by Karimi et al. on counterfactuals through interventions as opposed to perturbations (references in the paper). An idea could be to use the abstracted causal graph as our sampler for ECCCo (instead of SGLD). Combining the approach proposed by Karimi et al. with ideas underlying ECCCo, one could then generate counterfactuals that faithfully describe the causal graph learned by the model, instead of generating counterfactuals that comply with prior causal knowledge. We think this may go beyond the scope of our paper but would be happy to add this to section 7.