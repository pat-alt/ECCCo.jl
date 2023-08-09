Thank you! In this individual response, we will refer back to the main points discussed in the global response where relevant and discuss any other specific points the reviewer has raised. Below we will go through individual points where quotations trace back to reviewer remarks.

#### Q1 and Q3: Data and models

Please refer to **Point 1** and **Point 2** in the global response, respectively. 

#### Q2: Generalisability

> "Is the ECCCos approach adaptable to a broad range of black-box models beyond those discussed?"

Our approach should generalise to any classifier that is differentiable with respect to inputs, consistent with other gradient-based counterfactual generators (Equation 1). Our actual implementation is currently compatible with neural networks trained in Julia and has experimental support for `torch` trained in either Python or R. Even though it is possible to generate counterfactuals for non-differentiable models, it is not immediately obvious to us how SGLD can be applied in this context. An interesting question for future research would be if other scalable and gradient-free methods can be used to sample from the conditional distribution learned by the model. 

#### Q4: Link to causality

> "There’s a broad literature on causal abstractions and causal model explanations that seems related."

We agree that there is a connection with causal abstractions and causal model explanations. The two papers by Karimi et al. we are citing in the paper point in this direction, too, addressing counterfactuals through interventions as opposed to perturbations. An area of future research could be to use the abstracted causal graph as our sampler for ECCCo (instead of SGLD). Combining the approach proposed by Karimi et al. with ideas underlying ECCCo, one could then generate counterfactuals that faithfully describe the causal graph learned by the model, instead of generating counterfactuals that comply with prior causal knowledge. We will extend section 7 with a short discussion of this connection.