Thank you!

Many of the weaknesses pointed out here seem to centre around mathematical notions, so we will try to address these one-by-one with reference to the corresponding explanations in the paper.

The first explicit concern raised is about `lacking descriptions and explanations' of mathematical notation:

\begin{itemize}
    \item We state in Definition 4.1 that $p_{\theta}(\mathbf{x}|\mathbf{y^{+}})$ `denote[s] the conditional distribution of $\mathbf{x}$ in the target class  $\mathbf{y}^{+}$, where $\theta$ denotes the parameters of model $M_{\theta}$'. In other words, the conditional density is parameterised by $\theta$, which to our knowledge is standard notation and in fact the same notation as in Grathwohl (2020) (one of our main reference points). In the following sentence (line 137) of the paper we state in plain English that this can be understood intuitively as `what the model has learned about the data'. 
    \item Both $\varepsilon(\cdot)$ and $\hat{\mathbf{X}}_{\theta,y^{+}}^{n_E}$ are in our opinion sufficiently explained in line 146 and lines 168-169, respectively. Given the strict page limits, not every concept can be explained thoroughly. We do appreciate the expressed concern, however, and, in fact, our initial more lengthy drafts of the paper did include more textbook-style explanations in these places, which were eventually dropped for the sake of brevity. It is worth noting that additional detail can still be found in the Appendix.
\end{itemize}

The second explicit concern raised is that the conditional `distribution [$p(\mathbf{x}|\mathbf{y^{+}})$] is very challenging especially for structural data'. We disagree with the statement that this should be seen as a weakness of our paper:

\begin{itemize}
    \item Even if learning $p(\mathbf{x}|\mathbf{y^{+}})$ was an insurmountable challenge, it should in any case not invalidate the definition itself, which the reviewer seems to agree with.
    \item While we agree that learning this distribution is not always trivial, we note that this task is at the very core of Generative Modelling and AI---a field that has recently enjoyed success especially in the context of large unstructured data like images and language.
    \item Learning the generative task is also at the core of related approaches mentioned in the paper like REVISE: as we mention in line 89, the authors of REVISE `propose using a generative model such as a Variational Autoencoder (VAE)' to learn plausible explanations. We also point to other related approaches towards plausibility that all centre around learning the data-generating process of the inputs $X$ (lines 85 to 104).
    \item Learning $p(\mathbf{x}|\mathbf{y^{+}})$---the core task of Generative AI---should generally be easier than learning the unconditional distribution $p(\mathbf{x})$, because the information contained in labels can be leveraged in the latter case. 
\end{itemize}

The next explicit concern raised is about the generalisabilty and rigorousness of our implausibility metric:

\begin{itemize}
    \item We agree it is not perfect and do not fail to highlight its limitations in the paper (e.g. lines 297 to 299). But we think that it is an improved, more robust version of the metric that was previously proposed and used in the literature (lines 159 to 166). We did experiment with other distance/similarity metrics, but found the differences negligible enough to rely on L1 as our default metric across datasets and models for its sparsity inducing properties. 
    \item The rule-based unary constraint metric proposed in Vo et al. (2023) looks interesting, but the paper will be presented for the first time at KDD in August 2023 and we were not aware of it at the time of writing. Thanks for bringing it to our attention. 
\end{itemize}

Concern is also expressed with respect to how we defined `faithfulness'. The definition of $p_{\theta}(\mathbf{x}|\mathbf{y^{+}})$ seems to again cause confusion, but in line with this, we wish to highlight a possible reviewer misunderstanding with regard to a fundamental take in our work:

\begin{itemize}
    \item Regarding the point that `faithfulness [...] can be understood as the validity and fidelity of counterfactual examples', that is actually precisely what we argue \textit{against} in Sections 3 and 4. Here we do think we went above and beyond to convey the intuition through illustrative examples, because it forms the motivation for our work.
    \item As we point out repeatedly, `any valid counterfactual also has full fidelity by construction'. Any successful adversarial attack on a model is also a valid counterfactual, but it is hard to see how adversarial attacks in isolation faithfully describe model behaviour. That is why we propose a definition of `faithfulness' that works with distributional quantities. Specifically, we want to understand if counterfactuals are consistent with what the model has learned about the data, which is best expressed as $p_{\theta}(\mathbf{x}|\mathbf{y^{+}})$ (Def. 4.2).
    \item The role of SGLD is described in some detail in Section 4.1 (lines 138 to 155) and additional explanations are provided in Appendix A.
\end{itemize}
  

Finally, the idea to use Conformal Prediction (CP) in this context is mentioned both as a strength---`conformal prediction for counter-factual explanation is interesting'---and a weakness---`motivation of using Conformal Prediction (CP) is not convincing to me'. We reiterate our motivation here:

\begin{itemize}
    \item As we explain in some detail (lines 180 to 193) the idea rests on the notion that Predictive Uncertainty estimates can be used to generate plausible counterfactuals as previous work has shown.
    \item Since CP is model-agnostic, we propose relying on it to relax restrictions that were previously placed on the class of classifiers (lines 183 to 189).
    \item CP does indeed produce prediction sets in the context of classification. That is why we work with a smooth version of the set size that is compatible with gradient-based counterfactual search, as we explain in some detail in lines 194 to 205 and also in Appendix B.
\end{itemize}

We hope this sufficiently addresses at least some of the concerns that were raised and that the reviewer may reconsider their recommendation to reject this paper. 