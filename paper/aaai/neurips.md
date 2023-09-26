## Official Review of Submission12195 by Reviewer uCjw

Summary:
The paper addresses the challenge of faithfully explaining black-box models using Counterfactual Explanations (CE). Recognizing that traditional surrogate models might produce plausible but not always faithful explanations, the authors introduce a novel evaluation metric for faithfulness and propose an algorithmic framework for generating "Energy-Constrained Conformal Counterfactuals (ECCCos)." These ECCCos aim to provide explanations that are both faithful to the model's behavior and plausible. Leveraging advances in Energy-Based Modeling and Conformal Prediction, the paper proposed a step forward by emphasizing the importance of faithfulness in counterfactual explanations.

Soundness: 3 good
Presentation: 4 excellent
Contribution: 3 good
Strengths:
Originality: The paper introduces a new take on Counterfactual Explanations (CE) by differentiating between plausibility and faithfulness. Their "Energy-Constrained Conformal Counterfactuals (ECCCos)" is a novel approach, blending ideas from Energy-Based Modelling and Conformal Prediction.

Quality: The authors use established techniques in new ways to create ECCCos. Their approach is both theoretically grounded and practical.

Clarity: The paper is well-organized and easy to follow. The concept of ECCCos is explained clearly. Examples, like the '9' to '7' transformation, help illustrate the main points.

Significance: This work makes a clear contribution to the explainability literature. It addresses the shortcomings of previous models and offers a new way to generate counterfactual explanations. The focus on faithfulness fills a gap in current research, pushing for explanations that match a model's behavior.

Weaknesses:
The paper could benefit from more experiments on real-world data with more relevant models:

Limited Real-world Empirical Evaluation: While the authors test their method on multiple datasets, I still find the experiments with real-world data a bit limited. I would expect that the authors go beyond MNIST for demonstrating the effectiveness of their method on unstructured signals (i.e. vision, language, audio, etc.). I propose the authors find more realistic datasets that could demonstrate their claims.

Limited Model Evaluation: The focus of the models being tested seems narrow, centered on specific model types. Testing ECCCos on a broader range of models that are actually used in practice would demonstrate its wider applicability.

Questions:
Limited Real-world Empirical Evaluation How does ECCCos directly compare with existing methods on a more realistic dataset? Any vision/language dataset that allows such comparison would be very interesting.

Generalizability: Is the ECCCos approach adaptable to a broad range of black-box models beyond those discussed?

Limited Model Evaluation: Is ECCCo better than alternative approaches when trying to explain the predictions of more complex neural models?

Connections to causal abstractions and causal explanations: There’s a broad literature on causal abstractions and causal model explanations that seems related. While it is more focused on NLP applications, it seems still relevant here. Do you think that there is such a connection? If so, can you address it in your work?

Limitations:
The authors have touched upon the challenges of generating plausible counterfactuals and the distinction between fidelity and faithfulness. They also discussed the assumptions regarding model calibration. A more comprehensive discussion on the limitations of their proposed method, especially in real-world scenarios, would be interesting.

Flag For Ethics Review: No ethics review needed.
Rating: 7: Accept: Technically solid paper, with high impact on at least one sub-area, or moderate-to-high impact on more than one areas, with good-to-excellent evaluation, resources, reproducibility, and no unaddressed ethical considerations.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Code Of Conduct: Yes

## Official Review of Submission12195 by Reviewer pekM

Summary:
This paper studies the counter-factual explanation. Specifically, this proposes a novel approach for generating Energy-Constrained Conformal Counterfactuals (ECCCos) by leveraging with some recent advances in conformal prediction.

Soundness: 2 fair
Presentation: 2 fair
Contribution: 2 fair
Strengths:
The idea of leveraging with conformal prediction for counter-factual explanation is interesting.

Weaknesses:
The mathematical notions used in this paper are not solid. Some notions are lacking descriptions and explanations, for example, 
 in Def 4.1, 
 in Eq. (2), and 
.

Plausibility is an important constraints imposed on counter-factual examples. This helps to generate plausible example with valid attributes (e.g., age cannot be decreased and over 150). The definition 2.1 of Plausible Counterfactuals is reasonable but almost impossible to realize in practice because the class-condition distribution 
 is existed but unknown and learning this distribution is very challenging especially for structural data. Additionally, the implausibility metric seems not general and rigorous because for structural data like images, the image that minimizes 
 in Eq. (3) might not lie on the image manifold and correspond to a meaningful image.

Faithfulness is also very important and can be understood as the validity and fidelity of counter-factual examples, i.e., the model needs to predict counter-factual examples to the target class 
. The definition 4.1 is fine but missing of the details of 
. As far as I can guess, it is a distribution over the 
-th decision region induced by 
. However, it is not clear to me how to characterize/formulate 
 and use it in Stochastic Gradient Langevin Dynamics (SGLD) as in Eq. (2).

The motivation of using Conformal Prediction (CP) is not convincing to me. For CP, we need to provide a prediction set for a given data example x for ensuring a significant level. It is unclear to me about the definition of 
 and the role of the term in (6) in the proposed approach.

The experiments are humble and not really solid to me. I cannot see the clear advantages of the proposed approach except its superiority on some adapted metrics. I suggest the authors to refer to this paper [1] for conducting more rigorous experiments. Additionally, the authors need to conduct ablation studies regarding the involving terms in (5).

[1] Vy Vo, Trung Le, Van Nguyen, He Zhao, Edwin Bonilla, Gholamreza Haffari, Dinh Phung, Feature-based Learning for Diverse and Privacy-Preserving Counterfactual Explanations, KDD23.

Questions:
Please address my questions in the weakness section.

Limitations:
The authors adequately addressed the limitations of the work.

Flag For Ethics Review: No ethics review needed.
Rating: 3: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility and incompletely addressed ethical considerations.
Confidence: 5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.
Code Of Conduct: Yes

## Official Review of Submission12195 by Reviewer ZaU8

Summary:
This work argues for the importance of producing model explanations that are not only plausible but faithful to the black-box model in question. The authors note how XAI methods based on learning local surrogate models may produce plausible but unfaithful explanations, and quantify their definition of faithfulness. A novel algorithm for generating counterfactual explanations balancing faithfulness and plausibility is presented, which uses the gradients of the black-box model. I must note here that I am giving my score the lowest possible confidence rating of 1 since this paper is quite outside my area of expertise.

Soundness: 4 excellent
Presentation: 2 fair
Contribution: 3 good
Strengths:
Honest acknowledgment of method limitations in Section 7. Multiple baseline methods for comparison. Comprehensiveness of experiment producing Table 2.

Weaknesses:
Not able to fully assess from my understanding.

Questions:
It may be good to add a citation to [Welling & Teh, 2011] for SGLD on line 144

Limitations:
Need for gradient access, e.g. through autodiff, for black-box model under investigation.

Flag For Ethics Review: No ethics review needed.
Rating: 6: Weak Accept: Technically solid, moderate-to-high impact paper, with no major concerns with respect to evaluation, resources, reproducibility, ethical considerations.
Confidence: 1: Your assessment is an educated guess. The submission is not in your area or the submission was difficult to understand. Math/other details were not carefully checked.
Code Of Conduct: Yes

## Official Review of Submission12195 by Reviewer 6zGr

Summary:
The present work introduces faithfulness as an additional desideratum for counterfactual explanations. To this end, the authors introduce a metric for faithfulness that they finally additively include in their gradient-based counterfactual generation method. Technically, they use ideas from energy-based modeling to promote faithfulness. Further, they use ideas from conformal prediction to foster plausibility. In their experiments, they show the improved faithfulness of their approach to previous works and yield plausible counterfactuals for JEM-based classifiers.

Soundness: 2 fair
Presentation: 3 good
Contribution: 3 good
Strengths:
The notion of faithfulness (Def. 4.1; i.e., is the counterfactual “consistent with what models learned about the data” (L137)) is novel and interesting in the context of counterfactual explanations.
The inclusion of conformal prediction alleviates the need of a well-calibrated model, as required in the approach by Schut et al.
The paper is overall clearly written (with small needed fixes; see questions & suggestions below). Fig. 2 is very illustrative and nicely shows the effect of each proposed component.
Code is provided for reproducibility.
Weaknesses:
The major weakness of this work is that plausibility for non-JEM-based classifiers is very low on “real-world” datasets (Table 2). Further, there are no qualitative examples for non-JEM-based counterfactuals on, e.g., MNIST. Consequently, it is unclear whether ECCCos generates plausible counterfactuals beyond synthetic datasets for non-JEM-based classifiers, e.g., MLPs, CNNs, or transformers. This could significantly limit ECCCos’ applicability and utility for researchers as well as practitioners alike.
Besides the above, the visual quality of generated counterfactuals seems to be low (Fig. 6 in supplement using the JEM ensemble as classifier). Further, the counterfactuals for the same counterfactual target classes look very similar to each other and may hint to low diversity of generated counterfactuals.
ECCCos seems to generate counterfactuals that heavily change the initial image (column cost (closeness) in Tab. 7, 8, or Fig. 6), thereby violating the closeness desideratum.
The experiments are only conducted on small-scale datasets and it is unclear whether ECCCOS also works for, e.g., CIFAR- or ImageNet-like data.
From the synthetic experiments (Tab. 1) it is unclear if conformal prediction is actually required for ECCCos, as results are similar or better without it.
Experimental results for faithfulness are biased since (un)faithfulness is already used during counterfactual optimization as regularizer (Eq. 5).
Questions:
What are the choices for dist in Eq. 3 and 4?
How are the feature transformers 
 & 
 defined?
How are 
 & 
 chosen in the experiments?
What is the computational cost (runtime) of ECCCos, e.g., on MNIST for a single counterfactual?
Suggestions

The method is often called just “ECCCo” instead of “ECCCos” in the present manuscript.
The introduction of conformal prediction in Sec. 5 is a bit sudden and could be motivated beforehand.
The selection of the 
 lowest-energy samples in Algo. 1 could be mathematically better formulated
Limitations:
The limitations are adequately addressed.

Flag For Ethics Review: No ethics review needed.
Rating: 4: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.
Confidence: 5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.
Code Of Conduct: Yes

## API retrieval

cdate	domain	forum	id	invitations	mdate	nonreaders	number	readers	replyto	signatures	tcdate	tmdate	version	writers	summary	soundness	presentation	contribution	strengths	weaknesses	questions	limitations	flag_for_ethics_review	rating	confidence	code_of_conduct	first_time_reviewer
1.68716E+12	NeurIPS.cc/2023/Conference	cpte8KXdjE	SZKWXoX6au	['NeurIPS.cc/2023/Conference/Submission12195/-/Official_Review', 'NeurIPS.cc/2023/Conference/-/Edit']	1.69354E+12	[]	1	['NeurIPS.cc/2023/Conference/Program_Chairs', 'NeurIPS.cc/2023/Conference/Submission12195/Senior_Area_Chairs', 'NeurIPS.cc/2023/Conference/Submission12195/Area_Chairs', 'NeurIPS.cc/2023/Conference/Submission12195/Reviewers/Submitted', 'NeurIPS.cc/2023/Conference/Submission12195/Authors', 'NeurIPS.cc/2023/Conference/Submission12195/Reviewer_6zGr']	cpte8KXdjE	['NeurIPS.cc/2023/Conference/Submission12195/Reviewer_6zGr']	1.68716E+12	1.69354E+12	2	['NeurIPS.cc/2023/Conference', 'NeurIPS.cc/2023/Conference/Submission12195/Reviewer_6zGr']	{'value': 'The present work introduces faithfulness as an additional desideratum for counterfactual explanations. To this end, the authors introduce a metric for faithfulness that they finally additively include in their gradient-based counterfactual generation method. Technically, they use ideas from energy-based modeling to promote faithfulness. Further, they use ideas from conformal prediction to foster plausibility. In their experiments, they show the improved faithfulness of their approach to previous works and yield plausible counterfactuals for JEM-based classifiers.'}	{'value': '2 fair'}	{'value': '3 good'}	{'value': '3 good'}	{'value': '- The notion of faithfulness (Def. 4.1; i.e., is the counterfactual ‚Äúconsistent with what models learned about the data‚Äù (L137)) is novel and interesting in the context of counterfactual explanations.\n- The inclusion of conformal prediction alleviates the need of a well-calibrated model, as required in the approach by [Schut et al](https://arxiv.org/abs/2103.08951).\n- The paper is overall clearly written (with small needed fixes; see questions & suggestions below). Fig. 2 is very illustrative and nicely shows the effect of each proposed component.\n- Code is provided for reproducibility.'}	{'value': '- The major weakness of this work is that plausibility for non-JEM-based classifiers is very low on ‚Äúreal-world‚Äù datasets (Table 2). Further, there are no qualitative examples for non-JEM-based counterfactuals on, e.g., MNIST. Consequently, it is unclear whether ECCCos generates plausible counterfactuals beyond synthetic datasets for non-JEM-based classifiers, e.g., MLPs, CNNs, or transformers. This could significantly limit ECCCos‚Äô applicability and utility for researchers as well as practitioners alike.\n- Besides the above, the visual quality of generated counterfactuals seems to be low (Fig. 6 in supplement using the JEM ensemble as classifier). Further, the counterfactuals for the same counterfactual target classes look very similar to each other and may hint to low diversity of generated counterfactuals.\n- ECCCos seems to generate counterfactuals that heavily change the initial image (column `cost` (closeness) in Tab. 7, 8, or Fig. 6), thereby violating the closeness desideratum.\n- The experiments are only conducted on small-scale datasets and it is unclear whether ECCCOS also works for, e.g., CIFAR- or ImageNet-like data.\n- From the synthetic experiments (Tab. 1) it is unclear if conformal prediction is actually required for ECCCos, as results are similar or better without it.\n- Experimental results for faithfulness are biased since (un)faithfulness is already used during counterfactual optimization as regularizer (Eq. 5).'}	{'value': '- What are the choices for `dist` in Eq. 3 and 4?\n- How are the feature transformers $f$ & $f^{-1}$ defined?\n- How are $\\alpha$ & $\\kappa$ chosen in the experiments?\n- What is the computational cost (runtime) of ECCCos, e.g., on MNIST for a single counterfactual?\n\n**Suggestions**\n\n- The method is often called just ‚ÄúECCCo‚Äù instead of ‚ÄúECCCos‚Äù in the present manuscript.\n- The introduction of conformal prediction in Sec. 5 is a bit sudden and could be motivated beforehand.\n- The selection of the $n_E$ lowest-energy samples in Algo. 1 could be mathematically better formulated\n'}	{'value': 'The limitations are adequately addressed.'}	{'value': ['No ethics review needed.']}	{'value': '4: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.'}	{'value': '5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.'}	{'value': 'Yes'}	{'value': 'Yes', 'readers': ['NeurIPS.cc/2023/Conference/Program_Chairs', 'NeurIPS.cc/2023/Conference/Submission12195/Senior_Area_Chairs', 'NeurIPS.cc/2023/Conference/Submission12195/Area_Chairs', 'NeurIPS.cc/2023/Conference/Submission12195/Reviewer_6zGr']}
1.68859E+12	NeurIPS.cc/2023/Conference	cpte8KXdjE	aM6qGu5KII	['NeurIPS.cc/2023/Conference/Submission12195/-/Official_Review', 'NeurIPS.cc/2023/Conference/-/Edit']	1.69354E+12	[]	2	['NeurIPS.cc/2023/Conference/Program_Chairs', 'NeurIPS.cc/2023/Conference/Submission12195/Senior_Area_Chairs', 'NeurIPS.cc/2023/Conference/Submission12195/Area_Chairs', 'NeurIPS.cc/2023/Conference/Submission12195/Reviewers/Submitted', 'NeurIPS.cc/2023/Conference/Submission12195/Authors', 'NeurIPS.cc/2023/Conference/Submission12195/Reviewer_ZaU8']	cpte8KXdjE	['NeurIPS.cc/2023/Conference/Submission12195/Reviewer_ZaU8']	1.68859E+12	1.69354E+12	2	['NeurIPS.cc/2023/Conference', 'NeurIPS.cc/2023/Conference/Submission12195/Reviewer_ZaU8']	{'value': 'This work argues for the importance of producing model explanations that are not only plausible but faithful to the black-box model in question. The authors note how XAI methods based on learning local surrogate models may produce plausible but unfaithful explanations, and quantify their definition of faithfulness. A novel algorithm for generating counterfactual explanations balancing faithfulness and plausibility is presented, which uses the gradients of the black-box model. I must note here that I am giving my score the lowest possible confidence rating of 1 since this paper is quite outside my area of expertise.'}	{'value': '4 excellent'}	{'value': '2 fair'}	{'value': '3 good'}	{'value': 'Honest acknowledgment of method limitations in Section 7. Multiple baseline methods for comparison. Comprehensiveness of experiment producing Table 2.'}	{'value': 'Not able to fully assess from my understanding.'}	{'value': 'It may be good to add a citation to [Welling & Teh, 2011] for SGLD on line 144'}	{'value': 'Need for gradient access, e.g. through autodiff, for black-box model under investigation.'}	{'value': ['No ethics review needed.']}	{'value': '6: Weak Accept: Technically solid, moderate-to-high impact paper, with no major concerns with respect to evaluation, resources, reproducibility, ethical considerations.'}	{'value': '1: Your assessment is an educated guess. The submission is not in your area or the submission was difficult to understand. Math/other details were not carefully checked.'}	{'value': 'Yes'}	
1.68895E+12	NeurIPS.cc/2023/Conference	cpte8KXdjE	ZELyyMw9ph	['NeurIPS.cc/2023/Conference/Submission12195/-/Official_Review', 'NeurIPS.cc/2023/Conference/-/Edit']	1.69354E+12	[]	3	['NeurIPS.cc/2023/Conference/Program_Chairs', 'NeurIPS.cc/2023/Conference/Submission12195/Senior_Area_Chairs', 'NeurIPS.cc/2023/Conference/Submission12195/Area_Chairs', 'NeurIPS.cc/2023/Conference/Submission12195/Reviewers/Submitted', 'NeurIPS.cc/2023/Conference/Submission12195/Authors', 'NeurIPS.cc/2023/Conference/Submission12195/Reviewer_pekM']	cpte8KXdjE	['NeurIPS.cc/2023/Conference/Submission12195/Reviewer_pekM']	1.68895E+12	1.69354E+12	2	['NeurIPS.cc/2023/Conference', 'NeurIPS.cc/2023/Conference/Submission12195/Reviewer_pekM']	{'value': 'This paper studies the counter-factual explanation. Specifically, this proposes a novel approach for generating Energy-Constrained Conformal Counterfactuals (ECCCos) by leveraging with some recent advances in conformal prediction.  '}	{'value': '2 fair'}	{'value': '2 fair'}	{'value': '2 fair'}	{'value': 'The idea of leveraging with conformal prediction for counter-factual explanation is interesting.'}	{'value': 'The mathematical notions used in this paper are not solid. Some notions are lacking descriptions and explanations, for example, $p_\\theta(x \\mid y+)$ in Def 4.1,  $\\mathcal{E}$ in Eq. (2), and $\\hat{X}^{n_E}_{\\theta, y+}$. \n\nPlausibility is an important constraints imposed on counter-factual examples. This helps to generate plausible example with valid attributes (e.g.,  age cannot be decreased and over 150). The definition 2.1 of Plausible Counterfactuals is reasonable but almost impossible to realize in practice because the class-condition distribution $p(x \\mid y+)$ is existed but unknown and learning this distribution is very challenging especially for structural data. Additionally, the implausibility metric seems not general and rigorous because for structural data like images, the image that minimizes $impl$ in Eq. (3) might not lie on the image manifold and correspond to a meaningful image.\n\nFaithfulness is also very important and can be understood as the validity and fidelity of counter-factual examples, i.e., the model needs to predict counter-factual examples to the target class $y+$. The definition 4.1 is fine but missing of the details of $p_\\theta(x \\mid y+)$. As far as I can guess, it is a distribution over the $y+$-th decision region induced by $M_\\theta$. However, it is not clear to me how to characterize/formulate $p_\\theta(x \\mid y+)$ and use it in Stochastic Gradient Langevin Dynamics (SGLD) as in Eq. (2).\n\nThe motivation of using Conformal Prediction (CP) is not convincing to me. For CP, we need to provide a prediction set for a given data example x for ensuring a significant level. It is unclear to me about the definition of $C_{\\theta,y}(x; \\alpha)$ and the role of the term in (6) in the proposed approach.\n\nThe experiments are humble and not really solid to me. I cannot see the clear advantages of the proposed approach except its superiority on some adapted metrics. I suggest the authors to refer to this paper [1] for conducting more rigorous experiments. Additionally, the authors need to conduct ablation studies regarding the involving terms in (5).\n\n[1] Vy Vo, Trung Le, Van Nguyen, He Zhao, Edwin Bonilla, Gholamreza Haffari, Dinh Phung, Feature-based Learning for Diverse and Privacy-Preserving Counterfactual Explanations, KDD23.       '}	{'value': 'Please address my questions in the weakness section.'}	{'value': 'The authors adequately addressed the limitations of the work.'}	{'value': ['No ethics review needed.']}	{'value': '3: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility and incompletely addressed ethical considerations.'}	{'value': '5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.'}	{'value': 'Yes'}	
1.68996E+12	NeurIPS.cc/2023/Conference	cpte8KXdjE	ZDfFLtBoBi	['NeurIPS.cc/2023/Conference/Submission12195/-/Official_Review', 'NeurIPS.cc/2023/Conference/-/Edit']	1.69354E+12	[]	4	['NeurIPS.cc/2023/Conference/Program_Chairs', 'NeurIPS.cc/2023/Conference/Submission12195/Senior_Area_Chairs', 'NeurIPS.cc/2023/Conference/Submission12195/Area_Chairs', 'NeurIPS.cc/2023/Conference/Submission12195/Reviewers/Submitted', 'NeurIPS.cc/2023/Conference/Submission12195/Authors', 'NeurIPS.cc/2023/Conference/Submission12195/Reviewer_uCjw']	cpte8KXdjE	['NeurIPS.cc/2023/Conference/Submission12195/Reviewer_uCjw']	1.68996E+12	1.69354E+12	2	['NeurIPS.cc/2023/Conference', 'NeurIPS.cc/2023/Conference/Submission12195/Reviewer_uCjw']	{'value': 'The paper addresses the challenge of faithfully explaining black-box models using Counterfactual Explanations (CE). Recognizing that traditional surrogate models might produce plausible but not always faithful explanations, the authors introduce a novel evaluation metric for faithfulness and propose an algorithmic framework for generating "Energy-Constrained Conformal Counterfactuals (ECCCos)." These ECCCos aim to provide explanations that are both faithful to the model\'s behavior and plausible. Leveraging advances in Energy-Based Modeling and Conformal Prediction, the paper proposed a step forward by emphasizing the importance of faithfulness in counterfactual explanations.'}	{'value': '3 good'}	{'value': '4 excellent'}	{'value': '3 good'}	{'value': '\n**Originality**: The paper introduces a new take on Counterfactual Explanations (CE) by differentiating between plausibility and faithfulness. Their "Energy-Constrained Conformal Counterfactuals (ECCCos)" is a novel approach, blending ideas from Energy-Based Modelling and Conformal Prediction.\n\n**Quality**: The authors use established techniques in new ways to create ECCCos. Their approach is both theoretically grounded and practical.\n\n**Clarity**: The paper is well-organized and easy to follow. The concept of ECCCos is explained clearly. Examples, like the \'9\' to \'7\' transformation, help illustrate the main points.\n\n**Significance**: This work makes a clear contribution to the explainability literature. It addresses the shortcomings of previous models and offers a new way to generate counterfactual explanations. The focus on faithfulness fills a gap in current research, pushing for explanations that match a model\'s behavior.\n\n\n'}	{'value': 'The paper could benefit from more experiments on real-world data with more relevant models:\n\n1. **Limited Real-world Empirical Evaluation**: While the authors test their method on multiple datasets, I still find the experiments with real-world data a bit limited. I would expect that the authors go beyond MNIST for demonstrating the effectiveness of their method on unstructured signals (i.e. vision, language, audio, etc.). I propose the authors find more realistic datasets that could demonstrate their claims.\n\n2. **Limited Model Evaluation**: The focus of the models being tested seems narrow, centered on specific model types. Testing ECCCos on a broader range of models that are actually used in practice would demonstrate its wider applicability.\n\n'}	{'value': '1. **Limited Real-world Empirical Evaluation** How does ECCCos directly compare with existing methods on a more realistic dataset? Any vision/language dataset that allows such comparison would be very interesting.\n\n2. **Generalizability**: Is the ECCCos approach adaptable to a broad range of black-box models beyond those discussed?\n\n3. **Limited Model Evaluation**: Is ECCCo better than alternative approaches when trying to explain the predictions of more complex neural models?\n\n4. **Connections to causal abstractions and causal explanations**: There‚Äôs a broad literature on  causal abstractions and causal model explanations that seems related. While it is more focused on NLP applications, it seems still relevant here. Do you think that there is such a connection? If so, can you address it in your work?\n\n'}	{'value': 'The authors have touched upon the challenges of generating plausible counterfactuals and the distinction between fidelity and faithfulness. They also discussed the assumptions regarding model calibration. A more comprehensive discussion on the limitations of their proposed method, especially in real-world scenarios, would be interesting.\n\n'}	{'value': ['No ethics review needed.']}	{'value': '7: Accept: Technically solid paper, with high impact on at least one sub-area, or moderate-to-high impact on more than one areas, with good-to-excellent evaluation, resources, reproducibility, and no unaddressed ethical considerations.'}	{'value': '4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.'}	{'value': 'Yes'}	