
**Title**: ECCCos from the Black Box: Faithful Explanations through Energy-Constrained Conformal Counterfactuals

**Keywords**: Explainable AI, Counterfactual Explanations, Algorithmic Recourse, Energy-Based Models, Conformal Prediction

**Abstract**: Counterfactual Explanations offer an intuitive and straightforward way to explain black-box models and offer Algorithmic Recourse to individuals. To address the need for plausible explanations, existing work has primarily relied on surrogate models to learn how the input data is distributed. This effectively reallocates the task of learning realistic explanations for the data from the model itself to the surrogate. Consequently, the generated explanations may seem plausible to humans but need not necessarily describe the behaviour of the black-box model faithfully. We formalise this notion of faithfulness through the introduction of a tailored evaluation metric and propose a novel algorithmic framework for generating **E**nergy-**C**onstrained **C**onformal **Co**unterfactuals (ECCCos) that are only as plausible as the model permits. Through extensive empirical studies, we demonstrate that ECCCos reconcile the need for faithfulness and plausibility. In particular, we show that for models with gradient access, it is possible to achieve state-of-the-art performance without the need for surrogate models. To do so, our framework relies solely on properties defining the black-box model itself by leveraging recent advances in Energy-Based Modelling and Conformal Prediction. To our knowledge, this is the first venture in this direction for generating faithful Counterfactual Explanations. Thus, we anticipate that ECCCos can serve as a baseline for future research. We believe that our work opens avenues for researchers and practitioners seeking tools to better distinguish trustworthy from unreliable models.

**Corresponding Author**: p.altmeyer@tudelft.nl 

**Revier Nomination**: Arie.vanDeursen@tudelft.nl

**Primary Area**: Interpretability and Explainability

**Claims**: Yes

**Code of Ethics**: Yes

**Broader Impacts**: A narrow focus on generating plausible counterfactuals may lead practitioners and researchers to believe that even a highly vulnerable black-box model has learned plausible data representations. Our work aims to mitigate this.

**Limitations**: Yes

**Theory**: While we do not include any theoretical results in terms of formal proofs, we have approached the topic of Counterfactual Explanations from a new theoretical angle in this work. Where necessary we have clearly stated our assumptions. 

**Experiments**: Yes

**Training Details**: Yes

**Error Bars**: Yes

**Compute**: All of our experiments could be run locally on a personal machine. We will provide details regarding training times and compute in the supplementary material.

**Reproducibility**: Yes

**Safeguards**: n/a

**Licenses**: Yes

**Assets**: Yes

**Human Subjects**: n/a

**IRB Approvals**: n/a

**TLDR**: We leverage ideas from Energy-Based Modelling and Conformal Prediction to generate faithful Counterfactual Explanations that can distinguish trustworthy from unreliable models.

