# 自然语言处理 cs.CL

- **最新发布 162 篇**

- **更新 140 篇**

## 最新发布

#### [new 001] A Large-Scale Dataset and Citation Intent Classification in Turkish with LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦于土耳其语中引文意图分类任务，旨在解决因语言特点导致的引文意图理解难题。论文构建了一个公开数据集，并提出基于DSPy框架的可编程分类流水线与集成模型，最终达到91.3%的准确率。**

- **链接: [http://arxiv.org/pdf/2509.21907v1](http://arxiv.org/pdf/2509.21907v1)**

> **作者:** Kemal Sami Karaca; Bahaeddin Eravcı
>
> **备注:** Submitted to IEEE UBMK 2025 International Conference on Computer Science and Engineering
>
> **摘要:** Understanding the qualitative intent of citations is essential for a comprehensive assessment of academic research, a task that poses unique challenges for agglutinative languages like Turkish. This paper introduces a systematic methodology and a foundational dataset to address this problem. We first present a new, publicly available dataset of Turkish citation intents, created with a purpose-built annotation tool. We then evaluate the performance of standard In-Context Learning (ICL) with Large Language Models (LLMs), demonstrating that its effectiveness is limited by inconsistent results caused by manually designed prompts. To address this core limitation, we introduce a programmable classification pipeline built on the DSPy framework, which automates prompt optimization systematically. For final classification, we employ a stacked generalization ensemble to aggregate outputs from multiple optimized models, ensuring stable and reliable predictions. This ensemble, with an XGBoost meta-model, achieves a state-of-the-art accuracy of 91.3\%. Ultimately, this study provides the Turkish NLP community and the broader academic circles with a foundational dataset and a robust classification framework paving the way for future qualitative citation studies.
>
---
#### [new 002] Learning to Reason with Mixture of Tokens
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究强化学习中的推理生成任务，旨在解决现有方法未充分利用模型分布信息的问题。提出混合词元生成（MoT-G）框架，在连续空间中进行推理，提升训练效率和推理性能。**

- **链接: [http://arxiv.org/pdf/2509.21482v1](http://arxiv.org/pdf/2509.21482v1)**

> **作者:** Adit Jain; Brendan Rappazzo
>
> **备注:** 30 page
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) has become a leading approach for improving large language model (LLM) reasoning capabilities. Most current methods follow variants of Group Relative Policy Optimization, which samples multiple reasoning completions, scores them relative to each other, and adjusts the policy accordingly. However, these approaches invariably sample discrete tokens at each reasoning step, discarding the rich distributional information in the model's probability distribution over candidate tokens. While preserving and utilizing this distributional information has proven beneficial in non-RL settings, current RLVR methods seem to be unnecessarily constraining the reasoning search space by not using this information. To address this limitation, we investigate mixture-of-token generation (MoT-G) in RLVR. We present a unified framework that generalizes existing MoT-G approaches, including existing training-free methods that construct mixture embeddings as weighted sums over token embeddings, and extend RLVR to operate directly in this continuous mixture space for generating chain-of-thought. Evaluating two MoT-G variants on Reasoning-Gym, a suite of reasoning-intensive language tasks, we find that MoT--G methods achieve substantial improvements (5--35 \% gains on 7 out of 10 tasks) compared to standard decoding with the Qwen2.5-1.5B model, while reaching comparable accuracy with half the number of trajectories, suggesting improved training efficiency. Through comprehensive hidden-state and token-level analyses, we provide evidence that MoT--G's benefits may stem from its ability to maintain higher hidden-state entropy throughout the reasoning process and promote exploration in token space.
>
---
#### [new 003] Representing LLMs in Prompt Semantic Task Space
- **分类: cs.CL; cs.LG; 68T07, 68T50, 65F20; I.2.7; I.2.6; H.3.3**

- **简介: 该论文属于模型表示与选择任务，旨在解决如何高效、可扩展地为特定任务选择最佳大语言模型的问题。提出了一种无需训练的方法，将LLM表示为提示语义任务空间中的线性算子，实现高效、可解释的模型表示与选择。**

- **链接: [http://arxiv.org/pdf/2509.22506v1](http://arxiv.org/pdf/2509.22506v1)**

> **作者:** Idan Kashani; Avi Mendelson; Yaniv Nemcovsky
>
> **备注:** Accepted to Findings of the Association for Computational Linguistics: EMNLP 2025
>
> **摘要:** Large language models (LLMs) achieve impressive results over various tasks, and ever-expanding public repositories contain an abundance of pre-trained models. Therefore, identifying the best-performing LLM for a given task is a significant challenge. Previous works have suggested learning LLM representations to address this. However, these approaches present limited scalability and require costly retraining to encompass additional models and datasets. Moreover, the produced representation utilizes distinct spaces that cannot be easily interpreted. This work presents an efficient, training-free approach to representing LLMs as linear operators within the prompts' semantic task space, thus providing a highly interpretable representation of the models' application. Our method utilizes closed-form computation of geometrical properties and ensures exceptional scalability and real-time adaptability to dynamically expanding repositories. We demonstrate our approach on success prediction and model selection tasks, achieving competitive or state-of-the-art results with notable performance in out-of-sample scenarios.
>
---
#### [new 004] Evaluating and Improving Cultural Awareness of Reward Models for LLM Alignment
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦于提升大语言模型奖励模型的文化意识，针对现有评估缺乏文化相关数据的问题，提出了CARB基准，并设计了Think-as-Locals方法，通过强化学习改进文化对齐效果。**

- **链接: [http://arxiv.org/pdf/2509.21798v1](http://arxiv.org/pdf/2509.21798v1)**

> **作者:** Hongbin Zhang; Kehai Chen; Xuefeng Bai; Yang Xiang; Min Zhang
>
> **备注:** Under review on ICLR 2026;Work in progress;
>
> **摘要:** Reward models (RMs) are crucial for aligning large language models (LLMs) with diverse cultures. Consequently, evaluating their cultural awareness is essential for further advancing global alignment of LLMs. However, existing RM evaluations fall short in assessing cultural awareness due to the scarcity of culturally relevant evaluation datasets. To fill this gap, we propose Cultural Awareness Reward modeling Benchmark (CARB), covering 10 distinct cultures across 4 cultural domains. Our extensive evaluation of state-of-the-art RMs reveals their deficiencies in modeling cultural awareness and demonstrates a positive correlation between performance on CARB and downstream multilingual cultural alignment tasks. Further analysis identifies the spurious correlations within culture-aware reward modeling, wherein RM's scoring relies predominantly on surface-level features rather than authentic cultural nuance understanding. To address these, we propose Think-as-Locals to elicit deeper culturally grounded reasoning from generative RMs via reinforcement learning from verifiable rewards (RLVR) and employ well-designed rewards to ensure accurate preference judgments and high-quality structured evaluation criteria generation. Experimental results validate its efficacy in mitigating spurious features interference and advancing culture-aware reward modeling.
>
---
#### [new 005] Elastic MoE: Unlocking the Inference-Time Scalability of Mixture-of-Experts
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究混合专家（MoE）模型的推理扩展性问题，提出Elastic MoE框架，在不增加训练开销的情况下，使模型在推理时灵活调整激活专家数量，提升性能与计算效率。**

- **链接: [http://arxiv.org/pdf/2509.21892v1](http://arxiv.org/pdf/2509.21892v1)**

> **作者:** Naibin Gu; Zhenyu Zhang; Yuchen Feng; Yilong Chen; Peng Fu; Zheng Lin; Shuohuan Wang; Yu Sun; Hua Wu; Weiping Wang; Haifeng Wang
>
> **摘要:** Mixture-of-Experts (MoE) models typically fix the number of activated experts $k$ at both training and inference. Intuitively, activating more experts at inference $k'$ (where $k'> k$) means engaging a larger set of model parameters for the computation and thus is expected to improve performance. However, contrary to this intuition, we find the scaling range to be so narrow that performance begins to degrade rapidly after only a slight increase in the number of experts. Further investigation reveals that this degradation stems from a lack of learned collaboration among experts. To address this, we introduce Elastic Mixture-of-Experts (EMoE), a novel training framework that enables MoE models to scale the number of activated experts at inference without incurring additional training overhead. By simultaneously training experts to collaborate in diverse combinations and encouraging the router for high-quality selections, EMoE ensures robust performance across computational budgets at inference. We conduct extensive experiments on various MoE settings. Our results show that EMoE significantly expands the effective performance-scaling range, extending it to as much as 2-3$\times$ the training-time $k$, while also pushing the model's peak performance to a higher level.
>
---
#### [new 006] Enhancing Low-Rank Adaptation with Structured Nonlinear Transformations
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型的参数高效微调任务，旨在解决LoRA方法因线性结构导致表达能力受限的问题。提出了非线性扩展LoRAN，并引入基于正弦激活函数Sinter，在不增加参数量的前提下提升模型性能。**

- **链接: [http://arxiv.org/pdf/2509.21870v1](http://arxiv.org/pdf/2509.21870v1)**

> **作者:** Guanzhi Deng; Mingyang Liu; Dapeng Wu; Yinqiao Li; Linqi Song
>
> **备注:** This manuscript has been submitted to IEEE Journal of Selected Topics in Signal Processing (JSTSP) for review. Until the moment I submitted the manuscript to arXiv, we haven't received any review comments from JSTSP
>
> **摘要:** Low-Rank Adaptation (LoRA) is a widely adopted parameter-efficient fine-tuning method for large language models. However, its linear nature limits expressiveness. We propose LoRAN, a non-linear extension of LoRA that applies lightweight transformations to the low-rank updates. We further introduce Sinter, a sine-based activation that adds structured perturbations without increasing parameter count. Experiments across summarization and classification tasks show that LoRAN consistently improves over QLoRA. Ablation studies reveal that Sinter outperforms standard activations such as Sigmoid, ReLU, and Tanh, highlighting the importance of activation design in lowrank tuning.
>
---
#### [new 007] Why Chain of Thought Fails in Clinical Text Understanding
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究了思维链（CoT）在临床文本理解中的失效问题，评估了95个大模型在87项任务上的表现。结果发现多数模型使用CoT后性能下降，揭示了其在提升可解释性的同时可能降低可靠性的问题，为临床场景下的模型推理策略提供了实证依据。**

- **链接: [http://arxiv.org/pdf/2509.21933v1](http://arxiv.org/pdf/2509.21933v1)**

> **作者:** Jiageng Wu; Kevin Xie; Bowen Gu; Nils Krüger; Kueiyu Joshua Lin; Jie Yang
>
> **摘要:** Large language models (LLMs) are increasingly being applied to clinical care, a domain where both accuracy and transparent reasoning are critical for safe and trustworthy deployment. Chain-of-thought (CoT) prompting, which elicits step-by-step reasoning, has demonstrated improvements in performance and interpretability across a wide range of tasks. However, its effectiveness in clinical contexts remains largely unexplored, particularly in the context of electronic health records (EHRs), the primary source of clinical documentation, which are often lengthy, fragmented, and noisy. In this work, we present the first large-scale systematic study of CoT for clinical text understanding. We assess 95 advanced LLMs on 87 real-world clinical text tasks, covering 9 languages and 8 task types. Contrary to prior findings in other domains, we observe that 86.3\% of models suffer consistent performance degradation in the CoT setting. More capable models remain relatively robust, while weaker ones suffer substantial declines. To better characterize these effects, we perform fine-grained analyses of reasoning length, medical concept alignment, and error profiles, leveraging both LLM-as-a-judge evaluation and clinical expert evaluation. Our results uncover systematic patterns in when and why CoT fails in clinical contexts, which highlight a critical paradox: CoT enhances interpretability but may undermine reliability in clinical text tasks. This work provides an empirical basis for clinical reasoning strategies of LLMs, highlighting the need for transparent and trustworthy approaches.
>
---
#### [new 008] QoNext: Towards Next-generation QoE for Foundation Models
- **分类: cs.CL**

- **简介: 该论文提出QoNext框架，将网络与多媒体领域的质量体验（QoE）理念应用于基础模型评估。针对现有方法忽视交互体验的问题，构建了面向用户体验的数据库和预测模型，实现更细致、主动的评估与优化指导。**

- **链接: [http://arxiv.org/pdf/2509.21889v1](http://arxiv.org/pdf/2509.21889v1)**

> **作者:** Yijin Guo; Ye Shen; Farong Wen; Junying Wang; Zicheng Zhang; Qi Jia; Guangtao Zhai
>
> **摘要:** Existing evaluations of foundation models, including recent human-centric approaches, fail to capture what truly matters: user's experience during interaction. Current methods treat evaluation as a matter of output correctness alone, overlooking that user satisfaction emerges from the interplay between response quality and interaction, which limits their ability to account for the mechanisms underlying user experience. To address this gap, we introduce QoNext, the first framework that adapts Quality of Experience (QoE) principles from networking and multimedia to the assessment of foundation models. QoNext identifies experiential factors that shape user experience and incorporates them into controlled experiments, where human ratings are collected under varied configurations. From these studies we construct a QoE-oriented database and train predictive models that estimate perceived user experience from measurable system parameters. Our results demonstrate that QoNext not only enables proactive and fine-grained evaluation but also provides actionable guidance for productized services of optimizing foundation models in practice.
>
---
#### [new 009] From tests to effect sizes: Quantifying uncertainty and statistical variability in multilingual and multitask NLP evaluation benchmarks
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理（NLP）领域的多语言和多任务评估研究。它提出基于重采样的方法，用于量化评估指标的不确定性与统计变异性，解决因模型和数据变化导致的性能波动被低估的问题，并通过多个任务展示其在排行榜中的应用效果。**

- **链接: [http://arxiv.org/pdf/2509.22612v1](http://arxiv.org/pdf/2509.22612v1)**

> **作者:** Jonne Sälevä; Duygu Ataman; Constantine Lignos
>
> **备注:** Paper currently under review at ACL Rolling Review
>
> **摘要:** In this paper, we introduce a set of resampling-based methods for quantifying uncertainty and statistical precision of evaluation metrics in multilingual and/or multitask NLP benchmarks. We show how experimental variation in performance scores arises from both model- and data-related sources, and that accounting for both of them is necessary to avoid substantially underestimating the overall variability over hypothetical replications. Using multilingual question answering, machine translation, and named entity recognition as example tasks, we also demonstrate how resampling methods are useful for computing sampling distributions for various quantities used in leaderboards such as the average/median, pairwise differences between models, and rankings.
>
---
#### [new 010] Think Socially via Cognitive Reasoning
- **分类: cs.CL**

- **简介: 该论文提出Cognitive Reasoning和CogFlow框架，旨在增强大语言模型（LLMs）的社会认知能力。传统逻辑推理范式难以处理社交情境中的模糊信息，为此论文通过模拟人类思维流程，并结合监督学习与强化学习，提升LLMs在社会情境下的理解和回应能力。**

- **链接: [http://arxiv.org/pdf/2509.22546v1](http://arxiv.org/pdf/2509.22546v1)**

> **作者:** Jinfeng Zhou; Zheyu Chen; Shuai Wang; Quanyu Dai; Zhenhua Dong; Hongning Wang; Minlie Huang
>
> **备注:** Repository: https://github.com/thu-coai/CogFlow
>
> **摘要:** LLMs trained for logical reasoning excel at step-by-step deduction to reach verifiable answers. However, this paradigm is ill-suited for navigating social situations, which induce an interpretive process of analyzing ambiguous cues that rarely yield a definitive outcome. To bridge this gap, we introduce Cognitive Reasoning, a paradigm modeled on human social cognition. It formulates the interpretive process into a structured cognitive flow of interconnected cognitive units (e.g., observation or attribution), which combine adaptively to enable effective social thinking and responses. We then propose CogFlow, a complete framework that instills this capability in LLMs. CogFlow first curates a dataset of cognitive flows by simulating the associative and progressive nature of human thought via tree-structured planning. After instilling the basic cognitive reasoning capability via supervised fine-tuning, CogFlow adopts reinforcement learning to enable the model to improve itself via trial and error, guided by a multi-objective reward that optimizes both cognitive flow and response quality. Extensive experiments show that CogFlow effectively enhances the social cognitive capabilities of LLMs, and even humans, leading to more effective social decision-making.
>
---
#### [new 011] From Outliers to Topics in Language Models: Anticipating Trends in News Corpora
- **分类: cs.CL**

- **简介: 该论文研究了语言模型中被忽略的异常点如何预示新闻语料中新兴话题。通过词向量和聚类方法，分析法语和英语新闻数据，发现异常点会随时间演变为明确主题，解决了从噪声中识别趋势性话题的问题。**

- **链接: [http://arxiv.org/pdf/2509.22030v1](http://arxiv.org/pdf/2509.22030v1)**

> **作者:** Evangelia Zve; Benjamin Icard; Alice Breton; Lila Sainero; Gauvain Bourgne; Jean-Gabriel Ganascia
>
> **备注:** presented at ICNLSP 2025; to appear in the ACL Anthology; received the Best Full Paper Award
>
> **摘要:** This paper examines how outliers, often dismissed as noise in topic modeling, can act as weak signals of emerging topics in dynamic news corpora. Using vector embeddings from state-of-the-art language models and a cumulative clustering approach, we track their evolution over time in French and English news datasets focused on corporate social responsibility and climate change. The results reveal a consistent pattern: outliers tend to evolve into coherent topics over time across both models and languages.
>
---
#### [new 012] Vision Language Models Cannot Plan, but Can They Formalize?
- **分类: cs.CL**

- **简介: 该论文研究视觉语言模型（VLMs）在多模态规划中的作用，旨在解决长序列动作规划问题。作者提出五种将VLM用于PDDL形式化的框架，在真实图像环境下评估其性能，发现视觉理解是瓶颈，而形式化方法优于端到端生成。**

- **链接: [http://arxiv.org/pdf/2509.21576v1](http://arxiv.org/pdf/2509.21576v1)**

> **作者:** Muyu He; Yuxi Zheng; Yuchen Liu; Zijian An; Bill Cai; Jiani Huang; Lifeng Zhou; Feng Liu; Ziyang Li; Li Zhang
>
> **摘要:** The advancement of vision language models (VLMs) has empowered embodied agents to accomplish simple multimodal planning tasks, but not long-horizon ones requiring long sequences of actions. In text-only simulations, long-horizon planning has seen significant improvement brought by repositioning the role of LLMs. Instead of directly generating action sequences, LLMs translate the planning domain and problem into a formal planning language like the Planning Domain Definition Language (PDDL), which can call a formal solver to derive the plan in a verifiable manner. In multimodal environments, research on VLM-as-formalizer remains scarce, usually involving gross simplifications such as predefined object vocabulary or overly similar few-shot examples. In this work, we present a suite of five VLM-as-formalizer pipelines that tackle one-shot, open-vocabulary, and multimodal PDDL formalization. We evaluate those on an existing benchmark while presenting another two that for the first time account for planning with authentic, multi-view, and low-quality images. We conclude that VLM-as-formalizer greatly outperforms end-to-end plan generation. We reveal the bottleneck to be vision rather than language, as VLMs often fail to capture an exhaustive set of necessary object relations. While generating intermediate, textual representations such as captions or scene graphs partially compensate for the performance, their inconsistent gain leaves headroom for future research directions on multimodal planning formalization.
>
---
#### [new 013] Thinking in Many Modes: How Composite Reasoning Elevates Large Language Model Performance with Limited Data
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出复合推理（CR）方法，旨在提升大语言模型在数据有限时的复杂问题解决能力。通过动态结合多种推理方式（如演绎、归纳、溯因），CR在科学和医学问答任务中表现出优于现有方法的性能与效率。**

- **链接: [http://arxiv.org/pdf/2509.22224v1](http://arxiv.org/pdf/2509.22224v1)**

> **作者:** Zishan Ahmad; Saisubramaniam Gopalakrishnan
>
> **备注:** 7 pages, 3 figures
>
> **摘要:** Large Language Models (LLMs), despite their remarkable capabilities, rely on singular, pre-dominant reasoning paradigms, hindering their performance on intricate problems that demand diverse cognitive strategies. To address this, we introduce Composite Reasoning (CR), a novel reasoning approach empowering LLMs to dynamically explore and combine multiple reasoning styles like deductive, inductive, and abductive for more nuanced problem-solving. Evaluated on scientific and medical question-answering benchmarks, our approach outperforms existing baselines like Chain-of-Thought (CoT) and also surpasses the accuracy of DeepSeek-R1 style reasoning (SR) capabilities, while demonstrating superior sample efficiency and adequate token usage. Notably, CR adaptively emphasizes domain-appropriate reasoning styles. It prioritizes abductive and deductive reasoning for medical question answering, but shifts to causal, deductive, and inductive methods for scientific reasoning. Our findings highlight that by cultivating internal reasoning style diversity, LLMs acquire more robust, adaptive, and efficient problem-solving abilities.
>
---
#### [new 014] Generation-Time vs. Post-hoc Citation: A Holistic Evaluation of LLM Attribution
- **分类: cs.CL**

- **简介: 该论文研究大语言模型（LLM）的引用生成任务，旨在解决在高风险领域中如何可靠地引用来源的问题。论文提出了两种方法：生成时引用（G-Cite）和事后引用（P-Cite），并进行了全面评估，比较了它们在覆盖率、正确性和速度上的权衡，为不同应用场景提供推荐方案。**

- **链接: [http://arxiv.org/pdf/2509.21557v1](http://arxiv.org/pdf/2509.21557v1)**

> **作者:** Yash Saxena; Raviteja Bommireddy; Ankur Padia; Manas Gaur
>
> **备注:** Accepted at NeurIPS 2025 LLM Evaluation Workshop
>
> **摘要:** Trustworthy Large Language Models (LLMs) must cite human-verifiable sources in high-stakes domains such as healthcare, law, academia, and finance, where even small errors can have severe consequences. Practitioners and researchers face a choice: let models generate citations during decoding, or let models draft answers first and then attach appropriate citations. To clarify this choice, we introduce two paradigms: Generation-Time Citation (G-Cite), which produces the answer and citations in one pass, and Post-hoc Citation (P-Cite), which adds or verifies citations after drafting. We conduct a comprehensive evaluation from zero-shot to advanced retrieval-augmented methods across four popular attribution datasets and provide evidence-based recommendations that weigh trade-offs across use cases. Our results show a consistent trade-off between coverage and citation correctness, with retrieval as the main driver of attribution quality in both paradigms. P-Cite methods achieve high coverage with competitive correctness and moderate latency, whereas G-Cite methods prioritize precision at the cost of coverage and speed. We recommend a retrieval-centric, P-Cite-first approach for high-stakes applications, reserving G-Cite for precision-critical settings such as strict claim verification. Our codes and human evaluation results are available at https://anonymous.4open.science/r/Citation_Paradigms-BBB5/
>
---
#### [new 015] Advancing Natural Language Formalization to First Order Logic with Fine-tuned LLMs
- **分类: cs.CL; cs.AI; 03B10; I.2.7; I.2.3**

- **简介: 该论文研究自然语言到一阶逻辑（FOL）的自动翻译任务，旨在解决形式化表示的挑战。通过微调大语言模型（LLM），对比不同架构和训练策略，提出新方法与评估指标，发现Flan-T5-XXL在谓词辅助下表现最佳，推动了逻辑翻译的进展。**

- **链接: [http://arxiv.org/pdf/2509.22338v1](http://arxiv.org/pdf/2509.22338v1)**

> **作者:** Felix Vossel; Till Mossakowski; Björn Gehrke
>
> **备注:** 15 pages, 7 tables, accepted at the International Joint Conference on Learning & Reasoning (IJCLR 2025)
>
> **摘要:** Automating the translation of natural language to first-order logic (FOL) is crucial for knowledge representation and formal methods, yet remains challenging. We present a systematic evaluation of fine-tuned LLMs for this task, comparing architectures (encoder-decoder vs. decoder-only) and training strategies. Using the MALLS and Willow datasets, we explore techniques like vocabulary extension, predicate conditioning, and multilingual training, introducing metrics for exact match, logical equivalence, and predicate alignment. Our fine-tuned Flan-T5-XXL achieves 70% accuracy with predicate lists, outperforming GPT-4o and even the DeepSeek-R1-0528 model with CoT reasoning ability as well as symbolic systems like ccg2lambda. Key findings show: (1) predicate availability boosts performance by 15-20%, (2) T5 models surpass larger decoder-only LLMs, and (3) models generalize to unseen logical arguments (FOLIO dataset) without specific training. While structural logic translation proves robust, predicate extraction emerges as the main bottleneck.
>
---
#### [new 016] Diagnosing the Performance Trade-off in Moral Alignment: A Case Study on Gender Stereotypes
- **分类: cs.CL**

- **简介: 该论文研究了道德对齐中性能权衡问题，聚焦性别刻板印象。通过分析发现当前公平目标的局限性，揭示下游任务性能受整体遗忘水平影响，选择性遗忘会加剧整体遗忘，通用缓解方法无效。属于自然语言处理中的模型行为调控任务。**

- **链接: [http://arxiv.org/pdf/2509.21456v1](http://arxiv.org/pdf/2509.21456v1)**

> **作者:** Guangliang Liu; Bocheng Chen; Xitong Zhang; Kristen Marie Johnson
>
> **摘要:** Moral alignment has emerged as a widely adopted approach for regulating the behavior of pretrained language models (PLMs), typically through fine-tuning or model editing on curated datasets. However, this process often comes at the cost of degraded downstream task performance. Prior studies commonly aim to achieve a performance trade-off by encouraging PLMs to selectively forget stereotypical knowledge through carefully designed fairness objectives, while preserving their helpfulness. In this short paper, we investigate the underlying mechanisms of the performance trade-off in the context of mitigating gender stereotypes, through the lens of forgetting and the fairness objective. Our analysis reveals the limitations of current fairness objective in achieving trade-off by demonstrating that: (1) downstream task performance is primarily driven by the overall forgetting level; (2) selective forgetting of stereotypes tends to increase overall forgetting; and (3) general solutions for mitigating forgetting are ineffective at reducing overall forgetting and fail to improve downstream task performance.
>
---
#### [new 017] Context Parametrization with Compositional Adapters
- **分类: cs.CL**

- **简介: 该论文提出CompAs，一种基于组合结构的适配器生成框架，旨在解决大语言模型在处理多段上下文信息时的效率与灵活性问题。通过将上下文映射为可合并的适配器参数，实现高效、鲁棒的推理，并支持超长输入的处理和上下文恢复。**

- **链接: [http://arxiv.org/pdf/2509.22158v1](http://arxiv.org/pdf/2509.22158v1)**

> **作者:** Josip Jukić; Martin Tutek; Jan Šnajder
>
> **摘要:** Large language models (LLMs) often seamlessly adapt to new tasks through in-context learning (ICL) or supervised fine-tuning (SFT). However, both of these approaches face key limitations: ICL is inefficient when handling many demonstrations, and SFT incurs training overhead while sacrificing flexibility. Mapping instructions or demonstrations from context directly into adapter parameters offers an appealing alternative. While prior work explored generating adapters based on a single input context, it has overlooked the need to integrate multiple chunks of information. To address this gap, we introduce CompAs, a meta-learning framework that translates context into adapter parameters with a compositional structure. Adapters generated this way can be merged algebraically, enabling instructions, demonstrations, or retrieved passages to be seamlessly combined without reprocessing long prompts. Critically, this approach yields three benefits: lower inference cost, robustness to long-context instability, and establishes a principled solution when input exceeds the model's context window. Furthermore, CompAs encodes information into adapter parameters in a reversible manner, enabling recovery of input context through a decoder, facilitating safety and security. Empirical results on diverse multiple-choice and extractive question answering tasks show that CompAs outperforms ICL and prior generator-based methods, especially when scaling to more inputs. Our work establishes composable adapter generation as a practical and efficient alternative for scaling LLM deployment.
>
---
#### [new 018] LUMINA: Detecting Hallucinations in RAG System with Context-Knowledge Signals
- **分类: cs.CL**

- **简介: 该论文提出LUMINA框架，用于检测RAG系统中的幻觉问题。通过量化外部上下文与内部知识的利用信号，无需复杂调参即可有效识别模型生成内容中的不实信息，在多个基准测试中表现优异。**

- **链接: [http://arxiv.org/pdf/2509.21875v1](http://arxiv.org/pdf/2509.21875v1)**

> **作者:** Min-Hsuan Yeh; Yixuan Li; Tanwi Mallick
>
> **摘要:** Retrieval-Augmented Generation (RAG) aims to mitigate hallucinations in large language models (LLMs) by grounding responses in retrieved documents. Yet, RAG-based LLMs still hallucinate even when provided with correct and sufficient context. A growing line of work suggests that this stems from an imbalance between how models use external context and their internal knowledge, and several approaches have attempted to quantify these signals for hallucination detection. However, existing methods require extensive hyperparameter tuning, limiting their generalizability. We propose LUMINA, a novel framework that detects hallucinations in RAG systems through context-knowledge signals: external context utilization is quantified via distributional distance, while internal knowledge utilization is measured by tracking how predicted tokens evolve across transformer layers. We further introduce a framework for statistically validating these measurements. Experiments on common RAG hallucination benchmarks and four open-source LLMs show that LUMINA achieves consistently high AUROC and AUPRC scores, outperforming prior utilization-based methods by up to +13% AUROC on HalluRAG. Moreover, LUMINA remains robust under relaxed assumptions about retrieval quality and model matching, offering both effectiveness and practicality.
>
---
#### [new 019] Multilingual Dialogue Generation and Localization with Dialogue Act Scripting
- **分类: cs.CL**

- **简介: 该论文提出对话行为脚本（DAS），用于多语言对话生成与本地化。针对非英语对话数据稀缺的问题，DAS通过抽象意图表示生成自然、文化适配的对话，避免直接翻译带来的问题。**

- **链接: [http://arxiv.org/pdf/2509.22086v1](http://arxiv.org/pdf/2509.22086v1)**

> **作者:** Justin Vasselli; Eunike Andriani Kardinata; Yusuke Sakai; Taro Watanabe
>
> **备注:** 16 pages, 10 tables, 2 figures, Accepted at EMNLP Main 2025
>
> **摘要:** Non-English dialogue datasets are scarce, and models are often trained or evaluated on translations of English-language dialogues, an approach which can introduce artifacts that reduce their naturalness and cultural appropriateness. This work proposes Dialogue Act Script (DAS), a structured framework for encoding, localizing, and generating multilingual dialogues from abstract intent representations. Rather than translating dialogue utterances directly, DAS enables the generation of new dialogues in the target language that are culturally and contextually appropriate. By using structured dialogue act representations, DAS supports flexible localization across languages, mitigating translationese and enabling more fluent, naturalistic conversations. Human evaluations across Italian, German, and Chinese show that DAS-generated dialogues consistently outperform those produced by both machine and human translators on measures of cultural relevance, coherence, and situational appropriateness.
>
---
#### [new 020] KnowMT-Bench: Benchmarking Knowledge-Intensive Long-Form Question Answering in Multi-Turn Dialogues
- **分类: cs.CL**

- **简介: 该论文提出了KnowMT-Bench，首个针对多轮对话中知识密集型长答案问答任务的基准测试。旨在评估和提升大模型在医学、金融、法律等领域的事实准确性与信息效率。通过动态生成对话历史并验证最终答案，揭示了多轮对话中性能下降问题，并验证了RAG缓解效果。**

- **链接: [http://arxiv.org/pdf/2509.21856v1](http://arxiv.org/pdf/2509.21856v1)**

> **作者:** Junhao Chen; Yu Huang; Siyuan Li; Rui Yao; Hanqian Li; Hanyu Zhang; Jungang Li; Jian Chen; Bowen Wang; Xuming Hu
>
> **摘要:** Multi-Turn Long-Form Question Answering (MT-LFQA) is a key application paradigm of Large Language Models (LLMs) in knowledge-intensive domains. However, existing benchmarks are limited to single-turn dialogue, while multi-turn dialogue benchmarks typically assess other orthogonal capabilities rather than knowledge-intensive factuality. To bridge this critical gap, we introduce \textbf{KnowMT-Bench}, the \textit{first-ever} benchmark designed to systematically evaluate MT-LFQA for LLMs across knowledge-intensive fields, including medicine, finance, and law. To faithfully assess the model's real-world performance, KnowMT-Bench employs a dynamic evaluation setting where models generate their own multi-turn dialogue histories given logically progressive question sequences. The factual capability and information delivery efficiency of the \textit{final-turn} answer are then evaluated using a human-validated automated pipeline. Our experiments reveal that multi-turn contexts degrade performance: factual capability declines due to the contextual noise from self-generated histories, while information efficiency drops as models become more verbose with increasing dialogue length. We then investigate mitigation strategies, demonstrating that retrieval-augmented generation (RAG) can effectively alleviate and even reverse this factual degradation. These findings underscore the importance of our benchmark in evaluating and enhancing the conversational factual capabilities of LLMs in real-world knowledge-intensive applications. Code is available at \href{https://github.com/hardenyu21/KnowMT-Bench}{\textcolor{cyan}{\texttt{KnowMT-Bench}}}.
>
---
#### [new 021] From Formal Language Theory to Statistical Learning: Finite Observability of Subregular Languages
- **分类: cs.CL; cs.FL; cs.LG**

- **简介: 该论文研究自然语言结构建模，探讨子规则语言类的有限可观察性。证明其线性可分性，并通过实验验证模型有效性，为语言学习提供理论基础。**

- **链接: [http://arxiv.org/pdf/2509.22598v1](http://arxiv.org/pdf/2509.22598v1)**

> **作者:** Katsuhiko Hayashi; Hidetaka Kamigaito
>
> **备注:** 12 pages, 5 figures
>
> **摘要:** We prove that all standard subregular language classes are linearly separable when represented by their deciding predicates. This establishes finite observability and guarantees learnability with simple linear models. Synthetic experiments confirm perfect separability under noise-free conditions, while real-data experiments on English morphology show that learned features align with well-known linguistic constraints. These results demonstrate that the subregular hierarchy provides a rigorous and interpretable foundation for modeling natural language structure. Our code used in real-data experiments is available at https://github.com/UTokyo-HayashiLab/subregular.
>
---
#### [new 022] Conversational Implicatures: Modelling Relevance Theory Probabilistically
- **分类: cs.CL**

- **简介: 该论文探讨如何将贝叶斯方法应用于关联理论框架下的语用学研究，重点分析会话含义的隐含意义传递问题，尝试构建概率模型以解释语用现象。**

- **链接: [http://arxiv.org/pdf/2509.22354v1](http://arxiv.org/pdf/2509.22354v1)**

> **作者:** Christoph Unger; Hendrik Buschmeier
>
> **摘要:** Recent advances in Bayesian probability theory and its application to cognitive science in combination with the development of a new generation of computational tools and methods for probabilistic computation have led to a 'probabilistic turn' in pragmatics and semantics. In particular, the framework of Rational Speech Act theory has been developed to model broadly Gricean accounts of pragmatic phenomena in Bayesian terms, starting with fairly simple reference games and covering ever more complex communicative exchanges such as verbal syllogistic reasoning. This paper explores in which way a similar Bayesian approach might be applied to relevance-theoretic pragmatics (Sperber & Wilson, 1995) by study a paradigmatic pragmatic phenomenon: the communication of implicit meaning by ways of (conversational) implicatures.
>
---
#### [new 023] One Model, Many Morals: Uncovering Cross-Linguistic Misalignments in Computational Moral Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理与伦理AI交叉任务，旨在解决大语言模型（LLMs）在多语言、多文化环境下道德推理不一致的问题。作者将道德推理基准翻译为五种语言，进行零样本评估，揭示了语言和文化差异导致的LLMs道德判断偏差，并提出了道德推理错误分类体系。**

- **链接: [http://arxiv.org/pdf/2509.21443v1](http://arxiv.org/pdf/2509.21443v1)**

> **作者:** Sualeha Farid; Jayden Lin; Zean Chen; Shivani Kumar; David Jurgens
>
> **备注:** 22 pages, 11 figures, 6 tables
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed in multilingual and multicultural environments where moral reasoning is essential for generating ethically appropriate responses. Yet, the dominant pretraining of LLMs on English-language data raises critical concerns about their ability to generalize judgments across diverse linguistic and cultural contexts. In this work, we systematically investigate how language mediates moral decision-making in LLMs. We translate two established moral reasoning benchmarks into five culturally and typologically diverse languages, enabling multilingual zero-shot evaluation. Our analysis reveals significant inconsistencies in LLMs' moral judgments across languages, often reflecting cultural misalignment. Through a combination of carefully constructed research questions, we uncover the underlying drivers of these disparities, ranging from disagreements to reasoning strategies employed by LLMs. Finally, through a case study, we link the role of pretraining data in shaping an LLM's moral compass. Through this work, we distill our insights into a structured typology of moral reasoning errors that calls for more culturally-aware AI.
>
---
#### [new 024] Dual-Head Reasoning Distillation: Improving Classifier Accuracy with Train-Time-Only Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对分类任务中Chain-of-Thought（CoT）推理导致的吞吐量下降问题，提出Dual-Head Reasoning Distillation（DHRD）方法。通过在训练时引入两个头部：一个用于分类，一个用于推理监督，最终在测试时仅使用分类头，从而提升准确率并保持高吞吐量。**

- **链接: [http://arxiv.org/pdf/2509.21487v1](http://arxiv.org/pdf/2509.21487v1)**

> **作者:** Jillian Xu; Dylan Zhou; Vinay Shukla; Yang Yang; Junrui Ruan; Shuhuai Lin; Wenfei Zou; Yinxiao Liu; Karthik Lakshmanan
>
> **备注:** Accepted by the Workshop on Efficient Reasoning, Neurips 2025, 5 pages
>
> **摘要:** Chain-of-Thought (CoT) prompting often improves classification accuracy, but it introduces a significant throughput penalty with rationale generation (Wei et al., 2022; Cheng and Van Durme, 2024). To resolve this trade-off, we introduce Dual-Head Reasoning Distillation (DHRD), a simple training method for decoder-only language models (LMs) that adds (i) a pooled classification head used during training and inference and (ii) a reasoning head supervised by teacher rationales used only in training. We train with a loss function that is a weighted sum of label cross-entropy and token-level LM loss over input-plus-rationale sequences. On seven SuperGLUE tasks, DHRD yields relative gains of 0.65-5.47% over pooled baselines, with notably larger gains on entailment/causal tasks. Since we disable the reasoning head at test time, inference throughput matches pooled classifiers and exceeds CoT decoding on the same backbones by 96-142 times in QPS.
>
---
#### [new 025] COSPADI: Compressing LLMs via Calibration-Guided Sparse Dictionary Learning
- **分类: cs.CL**

- **简介: 该论文提出CoSpaDi，一种无需训练的LLM压缩框架。通过稀疏字典学习替代低秩分解，在保持模型精度的同时实现高效压缩，兼容量化技术，并在多个模型上验证了其优于现有方法的性能。**

- **链接: [http://arxiv.org/pdf/2509.22075v1](http://arxiv.org/pdf/2509.22075v1)**

> **作者:** Dmitriy Shopkhoev; Denis Makhov; Magauiya Zhussip; Ammar Ali; Stamatios Lefkimmiatis
>
> **摘要:** Post-training compression of large language models (LLMs) largely relies on low-rank weight approximation, which represents each column of a weight matrix in a shared low-dimensional subspace. While this is a computationally efficient strategy, the imposed structural constraint is rigid and can lead to a noticeable model accuracy drop. In this work, we propose CoSpaDi (Compression via Sparse Dictionary Learning), a novel training-free compression framework that replaces low-rank decomposition with a more flexible structured sparse factorization in which each weight matrix is represented with a dense dictionary and a column-sparse coefficient matrix. This formulation enables a union-of-subspaces representation: different columns of the original weight matrix are approximated in distinct subspaces spanned by adaptively selected dictionary atoms, offering greater expressiveness than a single invariant basis. Crucially, CoSpaDi leverages a small calibration dataset to optimize the factorization such that the output activations of compressed projection layers closely match those of the original ones, thereby minimizing functional reconstruction error rather than mere weight approximation. This data-aware strategy preserves better model fidelity without any fine-tuning under reasonable compression ratios. Moreover, the resulting structured sparsity allows efficient sparse-dense matrix multiplication and is compatible with post-training quantization for further memory and latency gains. We evaluate CoSpaDi across multiple Llama and Qwen models under per-layer and per-group settings at 20-50\% compression ratios, demonstrating consistent superiority over state-of-the-art data-aware low-rank methods both in accuracy and perplexity. Our results establish structured sparse dictionary learning as a powerful alternative to conventional low-rank approaches for efficient LLM deployment.
>
---
#### [new 026] Multilingual Vision-Language Models, A Survey
- **分类: cs.CL**

- **简介: 该论文综述多语言视觉-语言模型，探讨其在跨语言文本与图像处理中的研究进展。分析31个模型和21个基准，指出语言中立性与文化适应性之间的矛盾，并评估训练方法与评测目标的差异。**

- **链接: [http://arxiv.org/pdf/2509.22123v1](http://arxiv.org/pdf/2509.22123v1)**

> **作者:** Andrei-Alexandru Manea; Jindřich Libovický
>
> **摘要:** This survey examines multilingual vision-language models that process text and images across languages. We review 31 models and 21 benchmarks, spanning encoder-only and generative architectures, and identify a key tension between language neutrality (consistent cross-lingual representations) and cultural awareness (adaptation to cultural contexts). Current training methods favor neutrality through contrastive learning, while cultural awareness depends on diverse data. Two-thirds of evaluation benchmarks use translation-based approaches prioritizing semantic consistency, though recent work incorporates culturally grounded content. We find discrepancies in cross-lingual capabilities and gaps between training objectives and evaluation goals.
>
---
#### [new 027] Context Is What You Need: The Maximum Effective Context Window for Real World Limits of LLMs
- **分类: cs.CL; cs.AI; I.2.7**

- **简介: 该论文研究大语言模型（LLMs）上下文窗口的实际效能，提出最大有效上下文窗口（MECW）概念，通过系统测试比较不同模型在多种任务下的表现，发现实际有效长度远低于标称值，揭示了模型性能随上下文增加而下降的问题。**

- **链接: [http://arxiv.org/pdf/2509.21361v1](http://arxiv.org/pdf/2509.21361v1)**

> **作者:** Norman Paulsen
>
> **备注:** 20 pages, 4 charts
>
> **摘要:** Large language model (LLM) providers boast big numbers for maximum context window sizes. To test the real world use of context windows, we 1) define a concept of maximum effective context window, 2) formulate a testing method of a context window's effectiveness over various sizes and problem types, and 3) create a standardized way to compare model efficacy for increasingly larger context window sizes to find the point of failure. We collected hundreds of thousands of data points across several models and found significant differences between reported Maximum Context Window (MCW) size and Maximum Effective Context Window (MECW) size. Our findings show that the MECW is, not only, drastically different from the MCW but also shifts based on the problem type. A few top of the line models in our test group failed with as little as 100 tokens in context; most had severe degradation in accuracy by 1000 tokens in context. All models fell far short of their Maximum Context Window by as much as 99 percent. Our data reveals the Maximum Effective Context Window shifts based on the type of problem provided, offering clear and actionable insights into how to improve model accuracy and decrease model hallucination rates.
>
---
#### [new 028] Debiasing Large Language Models in Thai Political Stance Detection via Counterfactual Calibration
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦泰语政治立场检测任务，旨在解决大语言模型在低资源和文化复杂场景下的系统性偏见问题。提出ThaiFACTUAL框架，通过反事实数据增强与推理监督，无需微调即可减轻情感泄露和实体偏好，提升公平性和泛化能力，并发布首个高质量泰语政治立场数据集。**

- **链接: [http://arxiv.org/pdf/2509.21946v1](http://arxiv.org/pdf/2509.21946v1)**

> **作者:** Kasidit Sermsri; Teerapong Panboonyuen
>
> **备注:** 9 pages
>
> **摘要:** Political stance detection in low-resource and culturally complex settings poses a critical challenge for large language models (LLMs). In the Thai political landscape - marked by indirect language, polarized figures, and entangled sentiment and stance - LLMs often display systematic biases such as sentiment leakage and favoritism toward entities. These biases undermine fairness and reliability. We present ThaiFACTUAL, a lightweight, model-agnostic calibration framework that mitigates political bias without requiring fine-tuning. ThaiFACTUAL uses counterfactual data augmentation and rationale-based supervision to disentangle sentiment from stance and reduce bias. We also release the first high-quality Thai political stance dataset, annotated with stance, sentiment, rationales, and bias markers across diverse entities and events. Experimental results show that ThaiFACTUAL significantly reduces spurious correlations, enhances zero-shot generalization, and improves fairness across multiple LLMs. This work highlights the importance of culturally grounded debiasing techniques for underrepresented languages.
>
---
#### [new 029] Think-on-Graph 3.0: Efficient and Adaptive LLM Reasoning on Heterogeneous Graphs via Multi-Agent Dual-Evolving Context Retrieval
- **分类: cs.CL**

- **简介: 该论文提出ToG-3框架，针对图增强RAG方法中静态图结构限制推理的问题，设计多智能体协同的双演化机制（MACER），动态构建异构图索引，提升轻量级大模型的深度推理能力。**

- **链接: [http://arxiv.org/pdf/2509.21710v1](http://arxiv.org/pdf/2509.21710v1)**

> **作者:** Xiaojun Wu; Cehao Yang; Xueyuan Lin; Chengjin Xu; Xuhui Jiang; Yuanliang Sun; Hui Xiong; Jia Li; Jian Guo
>
> **备注:** 28 pages, 17 figures
>
> **摘要:** Retrieval-Augmented Generation (RAG) and Graph-based RAG has become the important paradigm for enhancing Large Language Models (LLMs) with external knowledge. However, existing approaches face a fundamental trade-off. While graph-based methods are inherently dependent on high-quality graph structures, they face significant practical constraints: manually constructed knowledge graphs are prohibitively expensive to scale, while automatically extracted graphs from corpora are limited by the performance of the underlying LLM extractors, especially when using smaller, local-deployed models. This paper presents Think-on-Graph 3.0 (ToG-3), a novel framework that introduces Multi-Agent Context Evolution and Retrieval (MACER) mechanism to overcome these limitations. Our core innovation is the dynamic construction and refinement of a Chunk-Triplets-Community heterogeneous graph index, which pioneeringly incorporates a dual-evolution mechanism of Evolving Query and Evolving Sub-Graph for precise evidence retrieval. This approach addresses a critical limitation of prior Graph-based RAG methods, which typically construct a static graph index in a single pass without adapting to the actual query. A multi-agent system, comprising Constructor, Retriever, Reflector, and Responser agents, collaboratively engages in an iterative process of evidence retrieval, answer generation, sufficiency reflection, and, crucially, evolving query and subgraph. This dual-evolving multi-agent system allows ToG-3 to adaptively build a targeted graph index during reasoning, mitigating the inherent drawbacks of static, one-time graph construction and enabling deep, precise reasoning even with lightweight LLMs. Extensive experiments demonstrate that ToG-3 outperforms compared baselines on both deep and broad reasoning benchmarks, and ablation studies confirm the efficacy of the components of MACER framework.
>
---
#### [new 030] SimulSense: Sense-Driven Interpreting for Efficient Simultaneous Speech Translation
- **分类: cs.CL**

- **简介: 该论文提出SimulSense，用于高效同声传译（SimulST）。它模仿人类译员感知语义单元进行读写决策，无需复杂训练数据和高计算成本，实现更优的质量-延迟平衡和实时效率。**

- **链接: [http://arxiv.org/pdf/2509.21932v1](http://arxiv.org/pdf/2509.21932v1)**

> **作者:** Haotian Tan; Hiroki Ouchi; Sakriani Sakti
>
> **备注:** \c{opyright} 2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
>
> **摘要:** How to make human-interpreter-like read/write decisions for simultaneous speech translation (SimulST) systems? Current state-of-the-art systems formulate SimulST as a multi-turn dialogue task, requiring specialized interleaved training data and relying on computationally expensive large language model (LLM) inference for decision-making. In this paper, we propose SimulSense, a novel framework for SimulST that mimics human interpreters by continuously reading input speech and triggering write decisions to produce translation when a new sense unit is perceived. Experiments against two state-of-the-art baseline systems demonstrate that our proposed method achieves a superior quality-latency tradeoff and substantially improved real-time efficiency, where its decision-making is up to 9.6x faster than the baselines.
>
---
#### [new 031] In Their Own Words: Reasoning Traces Tailored for Small Models Make Them Better Reasoners
- **分类: cs.CL**

- **简介: 该论文研究如何将大模型的推理能力迁移到小模型。任务是提升小模型的推理性能，解决因分布不匹配导致性能下降的问题。提出Reverse Speculative Decoding方法，生成适合小模型学习的推理轨迹，实验证明能有效提升小模型的推理表现。**

- **链接: [http://arxiv.org/pdf/2509.22230v1](http://arxiv.org/pdf/2509.22230v1)**

> **作者:** Jaehoon Kim; Kwangwook Seo; Dongha Lee
>
> **摘要:** Transferring reasoning capabilities from larger language models to smaller ones through supervised fine-tuning often fails counterintuitively, with performance degrading despite access to high-quality teacher demonstrations. We identify that this failure stems from distributional misalignment: reasoning traces from larger models contain tokens that are low probability under the student's distribution, exceeding the internal representation capacity of smaller architectures and creating learning barriers rather than helpful guidance. We propose Reverse Speculative Decoding (RSD), a mechanism for generating student-friendly reasoning traces in which the teacher model proposes candidate tokens but the student model determines acceptance based on its own probability distributions, filtering low probability tokens. When applied to Qwen3-0.6B, direct distillation of s1K-1.1 reasoning trace data degrades average performance across major reasoning benchmarks by 20.5\%, while the same model trained on RSD-generated reasoning traces achieves meaningful improvements of 4.9\%. Our analysis reveals that low probability tokens constitute the critical bottleneck in reasoning ability transfer. However, cross-model experiments demonstrate that RSD traces are model-specific rather than universally applicable, indicating that distributional alignment must be tailored for each student architecture's unique internal representation.
>
---
#### [new 032] Agribot: agriculture-specific question answer system
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出Agribot，一个农业问答系统，旨在解决农民获取农业信息的难题。基于Kisan Call Center数据构建，系统支持查询天气、市场价等农业相关问题，采用句子嵌入模型并优化后准确率达86%，提升信息获取效率。**

- **链接: [http://arxiv.org/pdf/2509.21535v1](http://arxiv.org/pdf/2509.21535v1)**

> **作者:** Naman Jain; Pranjali Jain; Pratik Kayal; Jayakrishna Sahit; Soham Pachpande; Jayesh Choudhari
>
> **摘要:** India is an agro-based economy and proper information about agricultural practices is the key to optimal agricultural growth and output. In order to answer the queries of the farmer, we have build an agricultural chatbot based on the dataset from Kisan Call Center. This system is robust enough to answer queries related to weather, market rates, plant protection and government schemes. This system is available 24* 7, can be accessed through any electronic device and the information is delivered with the ease of understanding. The system is based on a sentence embedding model which gives an accuracy of 56%. After eliminating synonyms and incorporating entity extraction, the accuracy jumps to 86%. With such a system, farmers can progress towards easier information about farming related practices and hence a better agricultural output. The job of the Call Center workforce would be made easier and the hard work of various such workers can be redirected to a better goal.
>
---
#### [new 033] Navigating the Impact of Structured Output Format on Large Language Models through the Compass of Causal Inference
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究结构化输出对大语言模型生成效果的影响，属于自然语言处理任务。通过因果推理分析发现，在多数场景下无因果影响，仅少数受指令等复杂因素影响，解决了评估方法不严谨的问题。**

- **链接: [http://arxiv.org/pdf/2509.21791v1](http://arxiv.org/pdf/2509.21791v1)**

> **作者:** Han Yuan; Yue Zhao; Li Zhang; Wuqiong Luo; Zheng Ma
>
> **摘要:** Structured output from large language models (LLMs) has enhanced efficiency in processing generated information and is increasingly adopted in industrial applications. Prior studies have investigated the impact of structured output on LLMs' generation quality, often presenting one-way findings. Some suggest that structured format enhances completeness and factual accuracy, while others argue that it restricts the reasoning capacity of LLMs and leads to reductions in standard evaluation metrics. Potential limitations of these assessments include restricted testing scenarios, weakly controlled comparative settings, and reliance on coarse metrics. In this work, we present a refined analysis using causal inference. Based on one assumed and two guaranteed constraints, we derive five potential causal structures characterizing the influence of structured output on LLMs' generation: (1) collider without m-bias, (2) collider with m-bias, (3) single cause from instruction, (4) single cause from output format, and (5) independence. Across seven public and one developed reasoning tasks, we find that coarse metrics report positive, negative, or neutral effects of structured output on GPT-4o's generation. However, causal inference reveals no causal impact in 43 out of 48 scenarios. In the remaining 5, 3 involve multifaceted causal structures influenced by concrete instructions.
>
---
#### [new 034] Beyond Textual Context: Structural Graph Encoding with Adaptive Space Alignment to alleviate the hallucination of LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型（LLMs）的幻觉问题，提出SSKG-LLM模型，通过引入知识图谱的结构与语义信息，并利用自适应空间对齐方法，增强LLMs的事实推理能力。**

- **链接: [http://arxiv.org/pdf/2509.22251v1](http://arxiv.org/pdf/2509.22251v1)**

> **作者:** Yifang Zhang; Pengfei Duan; Yiwen Yang; Shengwu Xiong
>
> **备注:** 11 pages, 5 figures
>
> **摘要:** Currently, the main approach for Large Language Models (LLMs) to tackle the hallucination issue is incorporating Knowledge Graphs(KGs).However, LLMs typically treat KGs as plain text, extracting only semantic information and limiting their use of the crucial structural aspects of KGs. Another challenge is the gap between the embedding spaces of KGs encoders and LLMs text embeddings, which hinders the effective integration of structured knowledge. To overcome these obstacles, we put forward the SSKG-LLM, an innovative model architecture that is designed to efficiently integrate both the Structural and Semantic information of KGs into the reasoning processes of LLMs. SSKG-LLM incorporates the Knowledge Graph Retrieval (KGR) module and the Knowledge Graph Encoding (KGE) module to preserve semantics while utilizing structure. Then, the Knowledge Graph Adaptation (KGA) module is incorporated to enable LLMs to understand KGs embeddings. We conduct extensive experiments and provide a detailed analysis to explore how incorporating the structural information of KGs can enhance the factual reasoning abilities of LLMs. Our code are available at https://github.com/yfangZhang/SSKG-LLM.
>
---
#### [new 035] FoodSEM: Large Language Model Specialized in Food Named-Entity Linking
- **分类: cs.CL; cs.IR**

- **简介: 该论文提出FoodSEM，一种专为食品领域命名实体链接（NEL）任务优化的开源大语言模型。针对通用和领域特定模型在食品NEL上的不足，FoodSEM通过指令-响应方式链接文本中的食品实体到多个本体，并在多个数据集上达到98%的F1分数，提供了食品语义理解的强基线。**

- **链接: [http://arxiv.org/pdf/2509.22125v1](http://arxiv.org/pdf/2509.22125v1)**

> **作者:** Ana Gjorgjevikj; Matej Martinc; Gjorgjina Cenikj; Sašo Džeroski; Barbara Koroušić Seljak; Tome Eftimov
>
> **备注:** To appear in the Proceedings of the 28th International Conference on Discovery Science (DS 2025)
>
> **摘要:** This paper introduces FoodSEM, a state-of-the-art fine-tuned open-source large language model (LLM) for named-entity linking (NEL) to food-related ontologies. To the best of our knowledge, food NEL is a task that cannot be accurately solved by state-of-the-art general-purpose (large) language models or custom domain-specific models/systems. Through an instruction-response (IR) scenario, FoodSEM links food-related entities mentioned in a text to several ontologies, including FoodOn, SNOMED-CT, and the Hansard taxonomy. The FoodSEM model achieves state-of-the-art performance compared to related models/systems, with F1 scores even reaching 98% on some ontologies and datasets. The presented comparative analyses against zero-shot, one-shot, and few-shot LLM prompting baselines further highlight FoodSEM's superior performance over its non-fine-tuned version. By making FoodSEM and its related resources publicly available, the main contributions of this article include (1) publishing a food-annotated corpora into an IR format suitable for LLM fine-tuning/evaluation, (2) publishing a robust model to advance the semantic understanding of text in the food domain, and (3) providing a strong baseline on food NEL for future benchmarking.
>
---
#### [new 036] SynerGen: Contextualized Generative Recommender for Unified Search and Recommendation
- **分类: cs.CL**

- **简介: 该论文提出SynerGen，一种统一个性化搜索与推荐的生成式模型。针对传统分阶段架构导致的校准偏差和工程复杂度问题，设计单一流生成框架，结合检索与排序任务，并引入时间感知位置编码，实现性能提升。**

- **链接: [http://arxiv.org/pdf/2509.21777v1](http://arxiv.org/pdf/2509.21777v1)**

> **作者:** Vianne R. Gao; Chen Xue; Marc Versage; Xie Zhou; Zhongruo Wang; Chao Li; Yeon Seonwoo; Nan Chen; Zhen Ge; Gourab Kundu; Weiqi Zhang; Tian Wang; Qingjun Cui; Trishul Chilimbi
>
> **备注:** Generative Recommender, Recommendation System, Information Retrieval
>
> **摘要:** The dominant retrieve-then-rank pipeline in large-scale recommender systems suffers from mis-calibration and engineering overhead due to its architectural split and differing optimization objectives. While recent generative sequence models have shown promise in unifying retrieval and ranking by auto-regressively generating ranked items, existing solutions typically address either personalized search or query-free recommendation, often exhibiting performance trade-offs when attempting to unify both. We introduce \textit{SynerGen}, a novel generative recommender model that bridges this critical gap by providing a single generative backbone for both personalized search and recommendation, while simultaneously excelling at retrieval and ranking tasks. Trained on behavioral sequences, our decoder-only Transformer leverages joint optimization with InfoNCE for retrieval and a hybrid pointwise-pairwise loss for ranking, allowing semantic signals from search to improve recommendation and vice versa. We also propose a novel time-aware rotary positional embedding to effectively incorporate time information into the attention mechanism. \textit{SynerGen} achieves significant improvements on widely adopted recommendation and search benchmarks compared to strong generative recommender and joint search and recommendation baselines. This work demonstrates the viability of a single generative foundation model for industrial-scale unified information access.
>
---
#### [new 037] Universal Legal Article Prediction via Tight Collaboration between Supervised Classification Model and LLM
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦法律条文预测（LAP）任务，旨在解决现有方法在跨司法辖区适用性和预测准确性方面的不足。提出Uni-LAP框架，结合监督分类模型与大语言模型的优势，通过Top-K损失和类比推理提升预测效果，实验证明其泛化性与有效性。**

- **链接: [http://arxiv.org/pdf/2509.22119v1](http://arxiv.org/pdf/2509.22119v1)**

> **作者:** Xiao Chi; Wenlin Zhong; Yiquan Wu; Wei Wang; Kun Kuang; Fei Wu; Minghui Xiong
>
> **备注:** 10 pages, 6 figures, Accepted to ICAIL 2025 (International Conference on Artificial Intelligence and Law)
>
> **摘要:** Legal Article Prediction (LAP) is a critical task in legal text classification, leveraging natural language processing (NLP) techniques to automatically predict relevant legal articles based on the fact descriptions of cases. As a foundational step in legal decision-making, LAP plays a pivotal role in determining subsequent judgments, such as charges and penalties. Despite its importance, existing methods face significant challenges in addressing the complexities of LAP. Supervised classification models (SCMs), such as CNN and BERT, struggle to fully capture intricate fact patterns due to their inherent limitations. Conversely, large language models (LLMs), while excelling in generative tasks, perform suboptimally in predictive scenarios due to the abstract and ID-based nature of legal articles. Furthermore, the diversity of legal systems across jurisdictions exacerbates the issue, as most approaches are tailored to specific countries and lack broader applicability. To address these limitations, we propose Uni-LAP, a universal framework for legal article prediction that integrates the strengths of SCMs and LLMs through tight collaboration. Specifically, in Uni-LAP, the SCM is enhanced with a novel Top-K loss function to generate accurate candidate articles, while the LLM employs syllogism-inspired reasoning to refine the final predictions. We evaluated Uni-LAP on datasets from multiple jurisdictions, and empirical results demonstrate that our approach consistently outperforms existing baselines, showcasing its effectiveness and generalizability.
>
---
#### [new 038] WebGen-Agent: Enhancing Interactive Website Generation with Multi-Level Feedback and Step-Level Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出WebGen-Agent，用于提升网站生成任务的性能。针对现有代码代理在视觉反馈和用户交互方面不足的问题，引入多级视觉反馈和GUI测试，并结合Step-GRPO强化学习方法，显著提升了网站生成的准确性和外观质量。**

- **链接: [http://arxiv.org/pdf/2509.22644v1](http://arxiv.org/pdf/2509.22644v1)**

> **作者:** Zimu Lu; Houxing Ren; Yunqiao Yang; Ke Wang; Zhuofan Zong; Junting Pan; Mingjie Zhan; Hongsheng Li
>
> **摘要:** Agent systems powered by large language models (LLMs) have demonstrated impressive performance on repository-level code-generation tasks. However, for tasks such as website codebase generation, which depend heavily on visual effects and user-interaction feedback, current code agents rely only on simple code execution for feedback and verification. This approach fails to capture the actual quality of the generated code. In this paper, we propose WebGen-Agent, a novel website-generation agent that leverages comprehensive and multi-level visual feedback to iteratively generate and refine the website codebase. Detailed and expressive text descriptions and suggestions regarding the screenshots and GUI-agent testing of the websites are generated by a visual language model (VLM), together with scores that quantify their quality. The screenshot and GUI-agent scores are further integrated with a backtracking and select-best mechanism, enhancing the performance of the agent. Utilizing the accurate visual scores inherent in the WebGen-Agent workflow, we further introduce \textit{Step-GRPO with Screenshot and GUI-agent Feedback} to improve the ability of LLMs to act as the reasoning engine of WebGen-Agent. By using the screenshot and GUI-agent scores at each step as the reward in Step-GRPO, we provide a dense and reliable process supervision signal, which effectively improves the model's website-generation ability. On the WebGen-Bench dataset, WebGen-Agent increases the accuracy of Claude-3.5-Sonnet from 26.4% to 51.9% and its appearance score from 3.0 to 3.9, outperforming the previous state-of-the-art agent system. Additionally, our Step-GRPO training approach increases the accuracy of Qwen2.5-Coder-7B-Instruct from 38.9% to 45.4% and raises the appearance score from 3.4 to 3.7.
>
---
#### [new 039] The Outputs of Large Language Models are Meaningless
- **分类: cs.CL; cs.AI**

- **简介: 该论文探讨大型语言模型（LLMs）输出是否具有意义的问题。作者认为，由于LLMs缺乏必要的意图，其输出本质上是无意义的，并反驳了相关论点，最后解释为何这些输出仍看似有意义并能传递知识。**

- **链接: [http://arxiv.org/pdf/2509.22206v1](http://arxiv.org/pdf/2509.22206v1)**

> **作者:** Anandi Hattiangadi; Anders J. Schoubye
>
> **备注:** 24 pages, 2 figures, forthcoming in Herman Cappelen and Rachel Sterken, eds. Communicating with AI: Philosophical Perspectives. Oxford: Oxford University Press
>
> **摘要:** In this paper, we offer a simple argument for the conclusion that the outputs of large language models (LLMs) are meaningless. Our argument is based on two key premises: (a) that certain kinds of intentions are needed in order for LLMs' outputs to have literal meanings, and (b) that LLMs cannot plausibly have the right kinds of intentions. We defend this argument from various types of responses, for example, the semantic externalist argument that deference can be assumed to take the place of intentions and the semantic internalist argument that meanings can be defined purely in terms of intrinsic relations between concepts, such as conceptual roles. We conclude the paper by discussing why, even if our argument is sound, the outputs of LLMs nevertheless seem meaningful and can be used to acquire true beliefs and even knowledge.
>
---
#### [new 040] Retrieval-Augmented Guardrails for AI-Drafted Patient-Portal Messages: Error Taxonomy Construction and Large-Scale Evaluation
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文针对AI生成的患者门户消息质量评估问题，提出了基于检索增强的评估框架（RAEC），构建了临床错误分类体系，并设计了可扩展的两阶段提示架构，提升了错误检测的准确性和一致性。**

- **链接: [http://arxiv.org/pdf/2509.22565v1](http://arxiv.org/pdf/2509.22565v1)**

> **作者:** Wenyuan Chen; Fateme Nateghi Haredasht; Kameron C. Black; Francois Grolleau; Emily Alsentzer; Jonathan H. Chen; Stephen P. Ma
>
> **摘要:** Asynchronous patient-clinician messaging via EHR portals is a growing source of clinician workload, prompting interest in large language models (LLMs) to assist with draft responses. However, LLM outputs may contain clinical inaccuracies, omissions, or tone mismatches, making robust evaluation essential. Our contributions are threefold: (1) we introduce a clinically grounded error ontology comprising 5 domains and 59 granular error codes, developed through inductive coding and expert adjudication; (2) we develop a retrieval-augmented evaluation pipeline (RAEC) that leverages semantically similar historical message-response pairs to improve judgment quality; and (3) we provide a two-stage prompting architecture using DSPy to enable scalable, interpretable, and hierarchical error detection. Our approach assesses the quality of drafts both in isolation and with reference to similar past message-response pairs retrieved from institutional archives. Using a two-stage DSPy pipeline, we compared baseline and reference-enhanced evaluations on over 1,500 patient messages. Retrieval context improved error identification in domains such as clinical completeness and workflow appropriateness. Human validation on 100 messages demonstrated superior agreement (concordance = 50% vs. 33%) and performance (F1 = 0.500 vs. 0.256) of context-enhanced labels vs. baseline, supporting the use of our RAEC pipeline as AI guardrails for patient messaging.
>
---
#### [new 041] AutoSCORE: Enhancing Automated Scoring with Multi-Agent Large Language Models via Structured Component Recognition
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出AutoSCORE，一个基于多智能体大语言模型的自动评分框架，旨在解决LLM评分中准确性低、可解释性差等问题。通过结构化组件识别与分步评分，提升评分准确性和一致性，适用于教育评估中的自动评分任务。**

- **链接: [http://arxiv.org/pdf/2509.21910v1](http://arxiv.org/pdf/2509.21910v1)**

> **作者:** Yun Wang; Zhaojun Ding; Xuansheng Wu; Siyue Sun; Ninghao Liu; Xiaoming Zhai
>
> **备注:** 9 pages, 2 figures
>
> **摘要:** Automated scoring plays a crucial role in education by reducing the reliance on human raters, offering scalable and immediate evaluation of student work. While large language models (LLMs) have shown strong potential in this task, their use as end-to-end raters faces challenges such as low accuracy, prompt sensitivity, limited interpretability, and rubric misalignment. These issues hinder the implementation of LLM-based automated scoring in assessment practice. To address the limitations, we propose AutoSCORE, a multi-agent LLM framework enhancing automated scoring via rubric-aligned Structured COmponent REcognition. With two agents, AutoSCORE first extracts rubric-relevant components from student responses and encodes them into a structured representation (i.e., Scoring Rubric Component Extraction Agent), which is then used to assign final scores (i.e., Scoring Agent). This design ensures that model reasoning follows a human-like grading process, enhancing interpretability and robustness. We evaluate AutoSCORE on four benchmark datasets from the ASAP benchmark, using both proprietary and open-source LLMs (GPT-4o, LLaMA-3.1-8B, and LLaMA-3.1-70B). Across diverse tasks and rubrics, AutoSCORE consistently improves scoring accuracy, human-machine agreement (QWK, correlations), and error metrics (MAE, RMSE) compared to single-agent baselines, with particularly strong benefits on complex, multi-dimensional rubrics, and especially large relative gains on smaller LLMs. These results demonstrate that structured component recognition combined with multi-agent design offers a scalable, reliable, and interpretable solution for automated scoring.
>
---
#### [new 042] Exploring Solution Divergence and Its Effect on Large Language Model Problem Solving
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究了大语言模型在解决问题时生成解的多样性（solution divergence），提出将其作为提升模型性能的新指标。论文验证了这一指标在监督微调和强化学习中的有效性，实验表明其能提高多个任务的成功率。**

- **链接: [http://arxiv.org/pdf/2509.22480v1](http://arxiv.org/pdf/2509.22480v1)**

> **作者:** Hang Li; Kaiqi Yang; Yucheng Chu; Hui Liu; Jiliang Tang
>
> **备注:** 17 pages, 11 figures
>
> **摘要:** Large language models (LLMs) have been widely used for problem-solving tasks. Most recent work improves their performance through supervised fine-tuning (SFT) with labeled data or reinforcement learning (RL) from task feedback. In this paper, we study a new perspective: the divergence in solutions generated by LLMs for a single problem. We show that higher solution divergence is positively related to better problem-solving abilities across various models. Based on this finding, we propose solution divergence as a novel metric that can support both SFT and RL strategies. We test this idea on three representative problem domains and find that using solution divergence consistently improves success rates. These results suggest that solution divergence is a simple but effective tool for advancing LLM training and evaluation.
>
---
#### [new 043] Thinking with Sound: Audio Chain-of-Thought Enables Multimodal Reasoning in Large Audio-Language Models
- **分类: cs.CL; cs.SD**

- **简介: 该论文针对复杂声学场景下音频推理任务中大模型表现不佳的问题，提出TwS框架，通过结合语言推理与实时音频分析提升鲁棒性，并构建MELD-Hard1k基准进行评估。**

- **链接: [http://arxiv.org/pdf/2509.21749v1](http://arxiv.org/pdf/2509.21749v1)**

> **作者:** Zhen Xiong; Yujun Cai; Zhecheng Li; Junsong Yuan; Yiwei Wang
>
> **摘要:** Recent Large Audio-Language Models (LALMs) have shown strong performance on various audio understanding tasks such as speech translation and Audio Q\&A. However, they exhibit significant limitations on challenging audio reasoning tasks in complex acoustic scenarios. These situations would greatly benefit from the use of acoustic tools like noise suppression, source separation, and precise temporal alignment, but current LALMs lack access to such tools. To address this limitation, we introduce Thinking-with-Sound (TwS), a framework that equips LALMs with Audio CoT by combining linguistic reasoning with on-the-fly audio-domain analysis. Unlike existing approaches that treat audio as static input, TwS enables models to actively think with audio signals, performing numerical analysis and digital manipulation through multimodal reasoning. To evaluate this approach, we construct MELD-Hard1k, a new robustness benchmark created by introducing various acoustic perturbations. Experiments reveal that state-of-the-art LALMs suffer dramatic performance degradation on MELD-Hard1k, with accuracy dropping by more than $50\%$ compared to clean audio. TwS achieves substantial improvements in robustness, demonstrating both effectiveness and scalability: small models gain $24.73\%$ absolute accuracy, with improvements scaling consistently up to $36.61\%$ for larger models. Our findings demonstrate that Audio CoT can significantly enhance robustness without retraining, opening new directions for developing more robust audio understanding systems.
>
---
#### [new 044] NeLLCom-Lex: A Neural-agent Framework to Study the Interplay between Lexical Systems and Language Use
- **分类: cs.CL**

- **简介: 该论文提出NeLLCom-Lex框架，利用神经网络代理研究词汇系统与语言使用的相互作用。通过模拟颜色命名任务，探索语义变化的机制，解决传统方法难以揭示因果关系的问题。**

- **链接: [http://arxiv.org/pdf/2509.22479v1](http://arxiv.org/pdf/2509.22479v1)**

> **作者:** Yuqing Zhang; Ecesu Ürker; Tessa Verhoef; Gemma Boleda; Arianna Bisazza
>
> **备注:** Findings of EMNLP 2025
>
> **摘要:** Lexical semantic change has primarily been investigated with observational and experimental methods; however, observational methods (corpus analysis, distributional semantic modeling) cannot get at causal mechanisms, and experimental paradigms with humans are hard to apply to semantic change due to the extended diachronic processes involved. This work introduces NeLLCom-Lex, a neural-agent framework designed to simulate semantic change by first grounding agents in a real lexical system (e.g. English) and then systematically manipulating their communicative needs. Using a well-established color naming task, we simulate the evolution of a lexical system within a single generation, and study which factors lead agents to: (i) develop human-like naming behavior and lexicons, and (ii) change their behavior and lexicons according to their communicative needs. Our experiments with different supervised and reinforcement learning pipelines show that neural agents trained to 'speak' an existing language can reproduce human-like patterns in color naming to a remarkable extent, supporting the further use of NeLLCom-Lex to elucidate the mechanisms of semantic change.
>
---
#### [new 045] Bridging Fairness and Explainability: Can Input-Based Explanations Promote Fairness in Hate Speech Detection?
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究在仇恨言论检测任务中，输入解释与公平性之间的关系。针对NLP模型的偏见问题，通过系统分析，探讨输入解释在识别偏见、选择公平模型和训练中减小偏见的效果，发现其在训练中有效但不适用于模型选择。**

- **链接: [http://arxiv.org/pdf/2509.22291v1](http://arxiv.org/pdf/2509.22291v1)**

> **作者:** Yifan Wang; Mayank Jobanputra; Ji-Ung Lee; Soyoung Oh; Isabel Valera; Vera Demberg
>
> **摘要:** Natural language processing (NLP) models often replicate or amplify social bias from training data, raising concerns about fairness. At the same time, their black-box nature makes it difficult for users to recognize biased predictions and for developers to effectively mitigate them. While some studies suggest that input-based explanations can help detect and mitigate bias, others question their reliability in ensuring fairness. Existing research on explainability in fair NLP has been predominantly qualitative, with limited large-scale quantitative analysis. In this work, we conduct the first systematic study of the relationship between explainability and fairness in hate speech detection, focusing on both encoder- and decoder-only models. We examine three key dimensions: (1) identifying biased predictions, (2) selecting fair models, and (3) mitigating bias during model training. Our findings show that input-based explanations can effectively detect biased predictions and serve as useful supervision for reducing bias during training, but they are unreliable for selecting fair models among candidates.
>
---
#### [new 046] ReviewScore: Misinformed Peer Review Detection with Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于学术评审质量检测任务，旨在解决AI会议中低质量评审问题。提出了ReviewScore，利用大语言模型自动识别评审中的错误前提和可回答问题，构建了人工标注数据集，并验证了模型在事实性评估上的潜力。**

- **链接: [http://arxiv.org/pdf/2509.21679v1](http://arxiv.org/pdf/2509.21679v1)**

> **作者:** Hyun Ryu; Doohyuk Jang; Hyemin S. Lee; Joonhyun Jeong; Gyeongman Kim; Donghyeon Cho; Gyouk Chu; Minyeong Hwang; Hyeongwon Jang; Changhun Kim; Haechan Kim; Jina Kim; Joowon Kim; Yoonjeon Kim; Kwanhyung Lee; Chanjae Park; Heecheol Yun; Gregor Betz; Eunho Yang
>
> **摘要:** Peer review serves as a backbone of academic research, but in most AI conferences, the review quality is degrading as the number of submissions explodes. To reliably detect low-quality reviews, we define misinformed review points as either "weaknesses" in a review that contain incorrect premises, or "questions" in a review that can be already answered by the paper. We verify that 15.2% of weaknesses and 26.4% of questions are misinformed and introduce ReviewScore indicating if a review point is misinformed. To evaluate the factuality of each premise of weaknesses, we propose an automated engine that reconstructs every explicit and implicit premise from a weakness. We build a human expert-annotated ReviewScore dataset to check the ability of LLMs to automate ReviewScore evaluation. Then, we measure human-model agreements on ReviewScore using eight current state-of-the-art LLMs and verify moderate agreements. We also prove that evaluating premise-level factuality shows significantly higher agreements than evaluating weakness-level factuality. A thorough disagreement analysis further supports a potential of fully automated ReviewScore evaluation.
>
---
#### [new 047] Capturing Opinion Shifts in Deliberative Discourse through Frequency-based Quantum deep learning methods
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在通过分析讨论中的观点变化来建模协商过程。研究构建了包含多元观点的数据集，并比较了两种模型（基于频率的论述调节和量子协商框架），以更有效地解析协商性话语并预测结果，应用于政策制定与社交媒体意见挖掘等领域。**

- **链接: [http://arxiv.org/pdf/2509.22603v1](http://arxiv.org/pdf/2509.22603v1)**

> **作者:** Rakesh Thakur; Harsh Chaturvedi; Ruqayya Shah; Janvi Chauhan; Ayush Sharma
>
> **备注:** 9 pages, 2 figures, 1 table
>
> **摘要:** Deliberation plays a crucial role in shaping outcomes by weighing diverse perspectives before reaching decisions. With recent advancements in Natural Language Processing, it has become possible to computationally model deliberation by analyzing opinion shifts and predicting potential outcomes under varying scenarios. In this study, we present a comparative analysis of multiple NLP techniques to evaluate how effectively models interpret deliberative discourse and produce meaningful insights. Opinions from individuals of varied backgrounds were collected to construct a self-sourced dataset that reflects diverse viewpoints. Deliberation was simulated using product presentations enriched with striking facts, which often prompted measurable shifts in audience opinions. We have given comparative analysis between two models namely Frequency-Based Discourse Modulation and Quantum-Deliberation Framework which outperform the existing state of art models. The findings highlight practical applications in public policy-making, debate evaluation, decision-support frameworks, and large-scale social media opinion mining.
>
---
#### [new 048] Variational Reasoning for Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出一种用于语言模型的变分推理框架，通过将思维轨迹视为潜在变量并利用变分推断优化，旨在提升模型的推理能力。研究统一了变分方法与强化学习，并在多种任务中验证了有效性。**

- **链接: [http://arxiv.org/pdf/2509.22637v1](http://arxiv.org/pdf/2509.22637v1)**

> **作者:** Xiangxin Zhou; Zichen Liu; Haonan Wang; Chao Du; Min Lin; Chongxuan Li; Liang Wang; Tianyu Pang
>
> **摘要:** We introduce a variational reasoning framework for language models that treats thinking traces as latent variables and optimizes them through variational inference. Starting from the evidence lower bound (ELBO), we extend it to a multi-trace objective for tighter bounds and propose a forward-KL formulation that stabilizes the training of the variational posterior. We further show that rejection sampling finetuning and binary-reward RL, including GRPO, can be interpreted as local forward-KL objectives, where an implicit weighting by model accuracy naturally arises from the derivation and reveals a previously unnoticed bias toward easier questions. We empirically validate our method on the Qwen 2.5 and Qwen 3 model families across a wide range of reasoning tasks. Overall, our work provides a principled probabilistic perspective that unifies variational inference with RL-style methods and yields stable objectives for improving the reasoning ability of language models. Our code is available at https://github.com/sail-sg/variational-reasoning.
>
---
#### [new 049] Multi-Objective Reinforcement Learning for Large Language Model Optimization: Visionary Perspective
- **分类: cs.CL; cs.AI; cs.LG; cs.MA**

- **简介: 该论文属于多目标强化学习（MORL）在大语言模型（LLM）优化中的应用研究。旨在解决LLM多目标优化中的效率与灵活性问题，提出了MORL分类法、基准框架及元策略双层学习范式的研究方向。**

- **链接: [http://arxiv.org/pdf/2509.21613v1](http://arxiv.org/pdf/2509.21613v1)**

> **作者:** Lingxiao Kong; Cong Yang; Oya Deniz Beyan; Zeyd Boukhers
>
> **备注:** 3 pages, 1 figure, accepted by ECAI MODeM 2025
>
> **摘要:** Multi-Objective Reinforcement Learning (MORL) presents significant challenges and opportunities for optimizing multiple objectives in Large Language Models (LLMs). We introduce a MORL taxonomy and examine the advantages and limitations of various MORL methods when applied to LLM optimization, identifying the need for efficient and flexible approaches that accommodate personalization functionality and inherent complexities in LLMs and RL. We propose a vision for a MORL benchmarking framework that addresses the effects of different methods on diverse objective relationships. As future research directions, we focus on meta-policy MORL development that can improve efficiency and flexibility through its bi-level learning paradigm, highlighting key research questions and potential solutions for improving LLM performance.
>
---
#### [new 050] InfiR2: A Comprehensive FP8 Training Recipe for Reasoning-Enhanced Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出InfiR2，一种端到端的FP8训练方法，旨在降低大语言模型训练的计算成本。通过混合粒度量化策略，实现高效且几乎无损的训练，在推理任务上性能与BF16相当，同时提升训练效率。**

- **链接: [http://arxiv.org/pdf/2509.22536v1](http://arxiv.org/pdf/2509.22536v1)**

> **作者:** Wenjun Wang; Shuo Cai; Congkai Xie; Mingfa Feng; Yiming Zhang; Zhen Li; Kejing Yang; Ming Li; Jiannong Cao; Yuan Xie; Hongxia Yang
>
> **摘要:** The immense computational cost of training Large Language Models (LLMs) presents a major barrier to innovation. While FP8 training offers a promising solution with significant theoretical efficiency gains, its widespread adoption has been hindered by the lack of a comprehensive, open-source training recipe. To bridge this gap, we introduce an end-to-end FP8 training recipe that seamlessly integrates continual pre-training and supervised fine-tuning. Our methodology employs a fine-grained, hybrid-granularity quantization strategy to maintain numerical fidelity while maximizing computational efficiency. Through extensive experiments, including the continue pre-training of models on a 160B-token corpus, we demonstrate that our recipe is not only remarkably stable but also essentially lossless, achieving performance on par with the BF16 baseline across a suite of reasoning benchmarks. Crucially, this is achieved with substantial efficiency improvements, including up to a 22% reduction in training time, a 14% decrease in peak memory usage, and a 19% increase in throughput. Our results establish FP8 as a practical and robust alternative to BF16, and we will release the accompanying code to further democratize large-scale model training.
>
---
#### [new 051] CHRONOBERG: Capturing Language Evolution and Temporal Awareness in Foundation Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CHRONOBERG，一个涵盖250年英文书籍的时序语料库，用于研究语言演变和模型的时间感知能力。任务聚焦于语言模型的时序泛化，旨在解决现有语料缺乏长期时间结构的问题，通过情感分析与训练实验展示时间敏感性建模的重要性。**

- **链接: [http://arxiv.org/pdf/2509.22360v1](http://arxiv.org/pdf/2509.22360v1)**

> **作者:** Niharika Hegde; Subarnaduti Paul; Lars Joel-Frey; Manuel Brack; Kristian Kersting; Martin Mundt; Patrick Schramowski
>
> **摘要:** Large language models (LLMs) excel at operating at scale by leveraging social media and various data crawled from the web. Whereas existing corpora are diverse, their frequent lack of long-term temporal structure may however limit an LLM's ability to contextualize semantic and normative evolution of language and to capture diachronic variation. To support analysis and training for the latter, we introduce CHRONOBERG, a temporally structured corpus of English book texts spanning 250 years, curated from Project Gutenberg and enriched with a variety of temporal annotations. First, the edited nature of books enables us to quantify lexical semantic change through time-sensitive Valence-Arousal-Dominance (VAD) analysis and to construct historically calibrated affective lexicons to support temporally grounded interpretation. With the lexicons at hand, we demonstrate a need for modern LLM-based tools to better situate their detection of discriminatory language and contextualization of sentiment across various time-periods. In fact, we show how language models trained sequentially on CHRONOBERG struggle to encode diachronic shifts in meaning, emphasizing the need for temporally aware training and evaluation pipelines, and positioning CHRONOBERG as a scalable resource for the study of linguistic change and temporal generalization. Disclaimer: This paper includes language and display of samples that could be offensive to readers. Open Access: Chronoberg is available publicly on HuggingFace at ( https://huggingface.co/datasets/spaul25/Chronoberg). Code is available at (https://github.com/paulsubarna/Chronoberg).
>
---
#### [new 052] Detecting (Un)answerability in Large Language Models with Linear Directions
- **分类: cs.CL**

- **简介: 该论文研究了大语言模型中（不可）回答性检测问题，旨在判断给定文本是否能回答问题。提出通过模型激活空间中的线性方向识别不可回答问题，实验表明该方法在多个问答基准上优于现有方法，并可推广至其他不可回答因素。**

- **链接: [http://arxiv.org/pdf/2509.22449v1](http://arxiv.org/pdf/2509.22449v1)**

> **作者:** Maor Juliet Lavi; Tova Milo; Mor Geva
>
> **摘要:** Large language models (LLMs) often respond confidently to questions even when they lack the necessary information, leading to hallucinated answers. In this work, we study the problem of (un)answerability detection, focusing on extractive question answering (QA) where the model should determine if a passage contains sufficient information to answer a given question. We propose a simple approach for identifying a direction in the model's activation space that captures unanswerability and uses it for classification. This direction is selected by applying activation additions during inference and measuring their impact on the model's abstention behavior. We show that projecting hidden activations onto this direction yields a reliable score for (un)answerability classification. Experiments on two open-weight LLMs and four extractive QA benchmarks show that our method effectively detects unanswerable questions and generalizes better across datasets than existing prompt-based and classifier-based approaches. Moreover, the obtained directions extend beyond extractive QA to unanswerability that stems from factors, such as lack of scientific consensus and subjectivity. Last, causal interventions show that adding or ablating the directions effectively controls the abstention behavior of the model.
>
---
#### [new 053] ProPerSim: Developing Proactive and Personalized AI Assistants through User-Assistant Simulation
- **分类: cs.CL**

- **简介: 该论文提出ProPerSim任务与框架，旨在开发兼具主动性和个性化的AI助手。通过模拟用户与助手的交互，助手根据反馈持续学习优化推荐策略，解决了现有研究中主动性与个性化结合不足的问题。**

- **链接: [http://arxiv.org/pdf/2509.21730v1](http://arxiv.org/pdf/2509.21730v1)**

> **作者:** Jiho Kim; Junseong Choi; Woosog Chay; Daeun Kyung; Yeonsu Kwon; Yohan Jo; Edward Choi
>
> **摘要:** As large language models (LLMs) become increasingly integrated into daily life, there is growing demand for AI assistants that are not only reactive but also proactive and personalized. While recent advances have pushed forward proactivity and personalization individually, their combination remains underexplored. To bridge this gap, we introduce ProPerSim, a new task and simulation framework for developing assistants capable of making timely, personalized recommendations in realistic home scenarios. In our simulation environment, a user agent with a rich persona interacts with the assistant, providing ratings on how well each suggestion aligns with its preferences and context. The assistant's goal is to use these ratings to learn and adapt to achieve higher scores over time. Built on ProPerSim, we propose ProPerAssistant, a retrieval-augmented, preference-aligned assistant that continually learns and adapts through user feedback. Experiments across 32 diverse personas show that ProPerAssistant adapts its strategy and steadily improves user satisfaction, highlighting the promise of uniting proactivity and personalization.
>
---
#### [new 054] MotivGraph-SoIQ: Integrating Motivational Knowledge Graphs and Socratic Dialogue for Enhanced LLM Ideation
- **分类: cs.CL**

- **简介: 该论文提出MotivGraph-SoIQ框架，结合激励知识图谱和苏格拉底对话，用于增强大语言模型（LLM）的创意生成。旨在解决LLM在创意落地和确认偏误方面的不足，提升创意的新颖性、实验严谨性和合理性。**

- **链接: [http://arxiv.org/pdf/2509.21978v1](http://arxiv.org/pdf/2509.21978v1)**

> **作者:** Xinping Lei; Tong Zhou; Yubo Chen; Kang Liu; Jun Zhao
>
> **备注:** EMNLP2025 Findings
>
> **摘要:** Large Language Models (LLMs) hold substantial potential for accelerating academic ideation but face critical challenges in grounding ideas and mitigating confirmation bias for further refinement. We propose integrating motivational knowledge graphs and socratic dialogue to address these limitations in enhanced LLM ideation (MotivGraph-SoIQ). This novel framework provides essential grounding and practical idea improvement steps for LLM ideation by integrating a Motivational Knowledge Graph (MotivGraph) with a Q-Driven Socratic Ideator. The MotivGraph structurally stores three key node types(problem, challenge and solution) to offer motivation grounding for the LLM ideation process. The Ideator is a dual-agent system utilizing Socratic questioning, which facilitates a rigorous refinement process that mitigates confirmation bias and improves idea quality across novelty, experimental rigor, and motivational rationality dimensions. On the ICLR25 paper topics dataset, MotivGraph-SoIQ exhibits clear advantages over existing state-of-the-art approaches across LLM-based scoring, ELO ranking, and human evaluation metrics.
>
---
#### [new 055] Towards Transparent AI: A Survey on Explainable Language Models
- **分类: cs.CL**

- **简介: 该论文属于综述任务，旨在解决语言模型（LMs）缺乏透明度的问题。论文系统梳理了可解释AI（XAI）方法在不同Transformer架构LM中的应用，分析其优劣，并探讨未来研究方向，以推动LM的可解释性发展。**

- **链接: [http://arxiv.org/pdf/2509.21631v1](http://arxiv.org/pdf/2509.21631v1)**

> **作者:** Avash Palikhe; Zichong Wang; Zhipeng Yin; Rui Guo; Qiang Duan; Jie Yang; Wenbin Zhang
>
> **摘要:** Language Models (LMs) have significantly advanced natural language processing and enabled remarkable progress across diverse domains, yet their black-box nature raises critical concerns about the interpretability of their internal mechanisms and decision-making processes. This lack of transparency is particularly problematic for adoption in high-stakes domains, where stakeholders need to understand the rationale behind model outputs to ensure accountability. On the other hand, while explainable artificial intelligence (XAI) methods have been well studied for non-LMs, they face many limitations when applied to LMs due to their complex architectures, considerable training corpora, and broad generalization abilities. Although various surveys have examined XAI in the context of LMs, they often fail to capture the distinct challenges arising from the architectural diversity and evolving capabilities of these models. To bridge this gap, this survey presents a comprehensive review of XAI techniques with a particular emphasis on LMs, organizing them according to their underlying transformer architectures: encoder-only, decoder-only, and encoder-decoder, and analyzing how methods are adapted to each while assessing their respective strengths and limitations. Furthermore, we evaluate these techniques through the dual lenses of plausibility and faithfulness, offering a structured perspective on their effectiveness. Finally, we identify open research challenges and outline promising future directions, aiming to guide ongoing efforts toward the development of robust, transparent, and interpretable XAI methods for LMs.
>
---
#### [new 056] "Be My Cheese?": Assessing Cultural Nuance in Multilingual LLM Translations
- **分类: cs.CL**

- **简介: 该论文研究多语言大模型在翻译英语比喻语言（如习语和双关语）时的文化适配能力。任务聚焦本地化质量评估，通过人工评审分析87份电商邮件翻译，揭示语法正确性之外的文化适应问题，指出当前模型在文化细腻度上的不足，强调数据量并非翻译质量唯一决定因素。**

- **链接: [http://arxiv.org/pdf/2509.21577v1](http://arxiv.org/pdf/2509.21577v1)**

> **作者:** Madison Van Doren; Cory Holland
>
> **摘要:** This pilot study explores the localisation capabilities of state-of-the-art multilingual AI models when translating figurative language, such as idioms and puns, from English into a diverse range of global languages. It expands on existing LLM translation research and industry benchmarks, which emphasise grammatical accuracy and token-level correctness, by focusing on cultural appropriateness and overall localisation quality - critical factors for real-world applications like marketing and e-commerce. To investigate these challenges, this project evaluated a sample of 87 LLM-generated translations of e-commerce marketing emails across 24 regional dialects of 20 languages. Human reviewers fluent in each target language provided quantitative ratings and qualitative feedback on faithfulness to the original's tone, meaning, and intended audience. Findings suggest that, while leading models generally produce grammatically correct translations, culturally nuanced language remains a clear area for improvement, often requiring substantial human refinement. Notably, even high-resource global languages, despite topping industry benchmark leaderboards, frequently mistranslated figurative expressions and wordplay. This work challenges the assumption that data volume is the most reliable predictor of machine translation quality and introduces cultural appropriateness as a key determinant of multilingual LLM performance - an area currently underexplored in existing academic and industry benchmarks. As a proof of concept, this pilot highlights limitations of current multilingual AI systems for real-world localisation use cases. Results of this pilot support the opportunity for expanded research at greater scale to deliver generalisable insights and inform deployment of reliable machine translation workflows in culturally diverse contexts.
>
---
#### [new 057] ArabJobs: A Multinational Corpus of Arabic Job Ads
- **分类: cs.CL**

- **简介: 该论文提出了ArabJobs，一个多国阿拉伯语招聘信息语料库，用于研究语言、地区和经济差异。通过分析性别、职业结构及方言变化，展示了其在公平NLP和劳动力市场研究中的应用价值。**

- **链接: [http://arxiv.org/pdf/2509.22589v1](http://arxiv.org/pdf/2509.22589v1)**

> **作者:** Mo El-Haj
>
> **摘要:** ArabJobs is a publicly available corpus of Arabic job advertisements collected from Egypt, Jordan, Saudi Arabia, and the United Arab Emirates. Comprising over 8,500 postings and more than 550,000 words, the dataset captures linguistic, regional, and socio-economic variation in the Arab labour market. We present analyses of gender representation and occupational structure, and highlight dialectal variation across ads, which offers opportunities for future research. We also demonstrate applications such as salary estimation and job category normalisation using large language models, alongside benchmark tasks for gender bias detection and profession classification. The findings show the utility of ArabJobs for fairness-aware Arabic NLP and labour market research. The dataset is publicly available on GitHub: https://github.com/drelhaj/ArabJobs.
>
---
#### [new 058] StableToken: A Noise-Robust Semantic Speech Tokenizer for Resilient SpeechLLMs
- **分类: cs.CL**

- **简介: 该论文提出StableToken，一种鲁棒的语义语音分词器，用于提升语音大模型的稳定性。针对现有分词器在噪声下不稳定的问题，设计多分支架构与位表决机制，有效减少噪声对输出的影响，显著提高下游任务的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.22220v1](http://arxiv.org/pdf/2509.22220v1)**

> **作者:** Yuhan Song; Linhao Zhang; Chuhan Wu; Aiwei Liu; Wei Jia; Houfeng Wang; Xiao Zhou
>
> **摘要:** Prevalent semantic speech tokenizers, designed to capture linguistic content, are surprisingly fragile. We find they are not robust to meaning-irrelevant acoustic perturbations; even at high Signal-to-Noise Ratios (SNRs) where speech is perfectly intelligible, their output token sequences can change drastically, increasing the learning burden for downstream LLMs. This instability stems from two flaws: a brittle single-path quantization architecture and a distant training signal indifferent to intermediate token stability. To address this, we introduce StableToken, a tokenizer that achieves stability through a consensus-driven mechanism. Its multi-branch architecture processes audio in parallel, and these representations are merged via a powerful bit-wise voting mechanism to form a single, stable token sequence. StableToken sets a new state-of-the-art in token stability, drastically reducing Unit Edit Distance (UED) under diverse noise conditions. This foundational stability translates directly to downstream benefits, significantly improving the robustness of SpeechLLMs on a variety of tasks.
>
---
#### [new 059] From Long to Lean: Performance-aware and Adaptive Chain-of-Thought Compression via Multi-round Refinement
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究链式推理（CoT）压缩任务，旨在解决推理延迟高和输出冗长的问题。提出MACC框架，通过多轮自适应压缩优化推理长度与性能，在降低延迟的同时提升准确率，并实现预测模型表现的能力。**

- **链接: [http://arxiv.org/pdf/2509.22144v1](http://arxiv.org/pdf/2509.22144v1)**

> **作者:** Jianzhi Yan; Le Liu; Youcheng Pan; Shiwei Chen; Zike Yuan; Yang Xiang; Buzhou Tang
>
> **备注:** 17 pages, 8 figures
>
> **摘要:** Chain-of-Thought (CoT) reasoning improves performance on complex tasks but introduces significant inference latency due to verbosity. We propose Multiround Adaptive Chain-of-Thought Compression (MACC), a framework that leverages the token elasticity phenomenon--where overly small token budgets can paradoxically increase output length--to progressively compress CoTs via multiround refinement. This adaptive strategy allows MACC to determine the optimal compression depth for each input. Our method achieves an average accuracy improvement of 5.6 percent over state-of-the-art baselines, while also reducing CoT length by an average of 47 tokens and significantly lowering latency. Furthermore, we show that test-time performance--accuracy and token length--can be reliably predicted using interpretable features like perplexity and compression rate on the training set. Evaluated across different models, our method enables efficient model selection and forecasting without repeated fine-tuning, demonstrating that CoT compression is both effective and predictable. Our code will be released in https://github.com/Leon221220/MACC.
>
---
#### [new 060] JGU Mainz's Submission to the WMT25 Shared Task on LLMs with Limited Resources for Slavic Languages: MT and QA
- **分类: cs.CL**

- **简介: 该论文属于WMT25共享任务，针对资源有限的斯拉夫语言（乌克兰语、上索布语、下索布语）进行机器翻译和问答研究。采用参数高效微调方法优化模型，并引入数据增强与集成学习，提升了基线性能。**

- **链接: [http://arxiv.org/pdf/2509.22490v1](http://arxiv.org/pdf/2509.22490v1)**

> **作者:** Hossain Shaikh Saadi; Minh Duc Bui; Mario Sanz-Guerrero; Katharina von der Wense
>
> **备注:** WMT 25 Shared Task LLMs with Limited Resources for Slavic Languages: MT and QA
>
> **摘要:** This paper presents the JGU Mainz submission to the WMT25 Shared Task on LLMs with Limited Resources for Slavic Languages: Machine Translation and Question Answering, focusing on Ukrainian, Upper Sorbian, and Lower Sorbian. For each language, we jointly fine-tune a Qwen2.5-3B-Instruct model for both tasks with parameter-efficient finetuning. Our pipeline integrates additional translation and multiple-choice question answering (QA) data. For Ukrainian QA, we further use retrieval-augmented generation. We also apply ensembling for QA in Upper and Lower Sorbian. Experiments show that our models outperform the baseline on both tasks.
>
---
#### [new 061] Question-Driven Analysis and Synthesis: Building Interpretable Thematic Trees with LLMs for Text Clustering and Controllable Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出RTP框架，利用大语言模型构建可解释的主题二叉树，解决文本聚类中的可解释性问题。通过自然语言问题划分数据，提升主题逻辑的清晰度，并支持可控生成，适用于数据稀缺领域。**

- **链接: [http://arxiv.org/pdf/2509.22211v1](http://arxiv.org/pdf/2509.22211v1)**

> **作者:** Tiago Fernandes Tavares
>
> **摘要:** Unsupervised analysis of text corpora is challenging, especially in data-scarce domains where traditional topic models struggle. While these models offer a solution, they typically describe clusters with lists of keywords that require significant manual effort to interpret and often lack semantic coherence. To address this critical interpretability gap, we introduce Recursive Thematic Partitioning (RTP), a novel framework that leverages Large Language Models (LLMs) to interactively build a binary tree. Each node in the tree is a natural language question that semantically partitions the data, resulting in a fully interpretable taxonomy where the logic of each cluster is explicit. Our experiments demonstrate that RTP's question-driven hierarchy is more interpretable than the keyword-based topics from a strong baseline like BERTopic. Furthermore, we establish the quantitative utility of these clusters by showing they serve as powerful features in downstream classification tasks, particularly when the data's underlying themes correlate with the task labels. RTP introduces a new paradigm for data exploration, shifting the focus from statistical pattern discovery to knowledge-driven thematic analysis. Furthermore, we demonstrate that the thematic paths from the RTP tree can serve as structured, controllable prompts for generative models. This transforms our analytical framework into a powerful tool for synthesis, enabling the consistent imitation of specific characteristics discovered in the source corpus.
>
---
#### [new 062] A Novel Differential Feature Learning for Effective Hallucination Detection and Classification
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型幻觉检测与分类任务，提出一种基于差分特征学习的方法。通过双模型架构和投影融合块，有效定位并提取稀疏的幻觉信号特征，在保证精度的同时显著降低计算成本。**

- **链接: [http://arxiv.org/pdf/2509.21357v1](http://arxiv.org/pdf/2509.21357v1)**

> **作者:** Wenkai Wang; Vincent Lee; Yizhen Zheng
>
> **备注:** 10 pages, 7 figures, 13 tables
>
> **摘要:** Large language model hallucination represents a critical challenge where outputs deviate from factual accuracy due to distributional biases in training data. While recent investigations establish that specific hidden layers exhibit differences between hallucinatory and factual content, the precise localization of hallucination signals within layers remains unclear, limiting the development of efficient detection methods. We propose a dual-model architecture integrating a Projected Fusion (PF) block for adaptive inter-layer feature weighting and a Differential Feature Learning (DFL) mechanism that identifies discriminative features by computing differences between parallel encoders learning complementary representations from identical inputs. Through systematic experiments across HaluEval's question answering, dialogue, and summarization datasets, we demonstrate that hallucination signals concentrate in highly sparse feature subsets, achieving significant accuracy improvements on question answering and dialogue tasks. Notably, our analysis reveals a hierarchical "funnel pattern" where shallow layers exhibit high feature diversity while deep layers demonstrate concentrated usage, enabling detection performance to be maintained with minimal degradation using only 1\% of feature dimensions. These findings suggest that hallucination signals are more concentrated than previously assumed, offering a pathway toward computationally efficient detection systems that could reduce inference costs while maintaining accuracy.
>
---
#### [new 063] Comparative Personalization for Multi-document Summarization
- **分类: cs.CL**

- **简介: 该论文研究个性化多文档摘要任务，旨在解决用户写作风格和内容偏好的个性化需求。提出了ComPSum框架，通过比较用户偏好生成个性化摘要，并构建了PerMSum数据集和AuthorMap评估方法，验证了模型的有效性。**

- **链接: [http://arxiv.org/pdf/2509.21562v1](http://arxiv.org/pdf/2509.21562v1)**

> **作者:** Haoyuan Li; Snigdha Chaturvedi
>
> **摘要:** Personalized multi-document summarization (MDS) is essential for meeting individual user preferences of writing style and content focus for summaries. In this paper, we propose that for effective personalization, it is important to identify fine-grained differences between users' preferences by comparing the given user's preferences with other users' preferences.Motivated by this, we propose ComPSum, a personalized MDS framework. It first generates a structured analysis of a user by comparing their preferences with other users' preferences. The generated structured analysis is then used to guide the generation of personalized summaries. To evaluate the performance of ComPSum, we propose AuthorMap, a fine-grained reference-free evaluation framework for personalized MDS. It evaluates the personalization of a system based on the authorship attribution between two personalized summaries generated for different users. For robust evaluation of personalized MDS, we construct PerMSum, a personalized MDS dataset in the review and news domain. We evaluate the performance of ComPSum on PerMSum using AuthorMap, showing that it outperforms strong baselines.
>
---
#### [new 064] VoiceAssistant-Eval: Benchmarking AI Assistants across Listening, Speaking, and Viewing
- **分类: cs.CL; cs.AI; cs.CV; cs.HC; cs.SD**

- **简介: 该论文提出了VoiceAssistant-Eval，一个用于评估语音助手在听、说、视三方面能力的综合基准，包含10,497个任务示例。通过评估21个开源模型和GPT-4o-Audio，揭示了当前模型在音频理解、多模态处理等方面的不足，为下一代AI助手的研发提供了指导框架。**

- **链接: [http://arxiv.org/pdf/2509.22651v1](http://arxiv.org/pdf/2509.22651v1)**

> **作者:** Ke Wang; Houxing Ren; Zimu Lu; Mingjie Zhan; Hongsheng Li
>
> **摘要:** The growing capabilities of large language models and multimodal systems have spurred interest in voice-first AI assistants, yet existing benchmarks are inadequate for evaluating the full range of these systems' capabilities. We introduce VoiceAssistant-Eval, a comprehensive benchmark designed to assess AI assistants across listening, speaking, and viewing. VoiceAssistant-Eval comprises 10,497 curated examples spanning 13 task categories. These tasks include natural sounds, music, and spoken dialogue for listening; multi-turn dialogue, role-play imitation, and various scenarios for speaking; and highly heterogeneous images for viewing. To demonstrate its utility, we evaluate 21 open-source models and GPT-4o-Audio, measuring the quality of the response content and speech, as well as their consistency. The results reveal three key findings: (1) proprietary models do not universally outperform open-source models; (2) most models excel at speaking tasks but lag in audio understanding; and (3) well-designed smaller models can rival much larger ones. Notably, the mid-sized Step-Audio-2-mini (7B) achieves more than double the listening accuracy of LLaMA-Omni2-32B-Bilingual. However, challenges remain: multimodal (audio plus visual) input and role-play voice imitation tasks are difficult for current models, and significant gaps persist in robustness and safety alignment. VoiceAssistant-Eval identifies these gaps and establishes a rigorous framework for evaluating and guiding the development of next-generation AI assistants. Code and data will be released at https://mathllm.github.io/VoiceAssistantEval/ .
>
---
#### [new 065] The QCET Taxonomy of Standard Quality Criterion Names and Definitions for the Evaluation of NLP Systems
- **分类: cs.CL; cs.AI; I.2.m**

- **简介: 该论文提出QCET分类法，旨在解决NLP系统评估中质量标准名称不统一导致的可比性问题。通过描述性方法构建标准质量准则名称和定义体系，用于评估对比、指导设计及合规性判断。**

- **链接: [http://arxiv.org/pdf/2509.22064v1](http://arxiv.org/pdf/2509.22064v1)**

> **作者:** Anya Belz; Simon Mille; Craig Thomson
>
> **备注:** 39 pages, 7 figures
>
> **摘要:** Prior work has shown that two NLP evaluation experiments that report results for the same quality criterion name (e.g. Fluency) do not necessarily evaluate the same aspect of quality, and the comparability implied by the name can be misleading. Not knowing when two evaluations are comparable in this sense means we currently lack the ability to draw reliable conclusions about system quality on the basis of multiple, independently conducted evaluations. This in turn hampers the ability of the field to progress scientifically as a whole, a pervasive issue in NLP since its beginning (Sparck Jones, 1981). It is hard to see how the issue of unclear comparability can be fully addressed other than by the creation of a standard set of quality criterion names and definitions that the several hundred quality criterion names actually in use in the field can be mapped to, and grounded in. Taking a strictly descriptive approach, the QCET Quality Criteria for Evaluation Taxonomy derives a standard set of quality criterion names and definitions from three surveys of evaluations reported in NLP, and structures them into a hierarchy where each parent node captures common aspects of its child nodes. We present QCET and the resources it consists of, and discuss its three main uses in (i) establishing comparability of existing evaluations, (ii) guiding the design of new evaluations, and (iii) assessing regulatory compliance.
>
---
#### [new 066] How Accurate Are LLMs at Multi-Question Answering on Conversational Transcripts?
- **分类: cs.CL**

- **简介: 该论文研究LLMs在基于对话转录文本进行多问题回答任务中的表现。针对工业场景中计算成本高和延迟大的问题，作者对多种模型进行了实验对比，发现微调的开源模型在某些情况下可超越GPT-4o。**

- **链接: [http://arxiv.org/pdf/2509.21732v1](http://arxiv.org/pdf/2509.21732v1)**

> **作者:** Xiliang Zhu; Shi Zong; David Rossouw
>
> **备注:** Accepted by EMNLP 2025 Industry Track
>
> **摘要:** Deploying Large Language Models (LLMs) for question answering (QA) over lengthy contexts is a significant challenge. In industrial settings, this process is often hindered by high computational costs and latency, especially when multiple questions must be answered based on the same context. In this work, we explore the capabilities of LLMs to answer multiple questions based on the same conversational context. We conduct extensive experiments and benchmark a range of both proprietary and public models on this challenging task. Our findings highlight that while strong proprietary LLMs like GPT-4o achieve the best overall performance, fine-tuned public LLMs with up to 8 billion parameters can surpass GPT-4o in accuracy, which demonstrates their potential for transparent and cost-effective deployment in real-world applications.
>
---
#### [new 067] Fuzzy Reasoning Chain (FRC): An Innovative Reasoning Framework from Fuzziness to Clarity
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Fuzzy Reasoning Chain (FRC)框架，结合LLM语义先验与模糊隶属度，解决文本歧义和不确定性问题，应用于情感分析任务，提升了推理稳定性与可解释性。**

- **链接: [http://arxiv.org/pdf/2509.22054v1](http://arxiv.org/pdf/2509.22054v1)**

> **作者:** Ping Chen; Xiang Liu; Zhaoxiang Liu; Zezhou Chen; Xingpeng Zhang; Huan Hu; Zipeng Wang; Kai Wang; Shuming Shi; Shiguo Lian
>
> **备注:** Accepet by EMNLP 2025 Findings (11 pages, 1 figures)
>
> **摘要:** With the rapid advancement of large language models (LLMs), natural language processing (NLP) has achieved remarkable progress. Nonetheless, significant challenges remain in handling texts with ambiguity, polysemy, or uncertainty. We introduce the Fuzzy Reasoning Chain (FRC) framework, which integrates LLM semantic priors with continuous fuzzy membership degrees, creating an explicit interaction between probability-based reasoning and fuzzy membership reasoning. This transition allows ambiguous inputs to be gradually transformed into clear and interpretable decisions while capturing conflicting or uncertain signals that traditional probability-based methods cannot. We validate FRC on sentiment analysis tasks, where both theoretical analysis and empirical results show that it ensures stable reasoning and facilitates knowledge transfer across different model scales. These findings indicate that FRC provides a general mechanism for managing subtle and ambiguous expressions with improved interpretability and robustness.
>
---
#### [new 068] Chimera: Diagnosing Shortcut Learning in Visual-Language Understanding
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于视觉-语言理解任务，旨在诊断模型在处理图表时的“捷径学习”问题。作者构建了Chimera测试集（含7500个维基图表），通过多层级问题评估模型是否真正理解图表，而非依赖记忆或表面模式。**

- **链接: [http://arxiv.org/pdf/2509.22437v1](http://arxiv.org/pdf/2509.22437v1)**

> **作者:** Ziheng Chi; Yifan Hou; Chenxi Pang; Shaobo Cui; Mubashara Akhtar; Mrinmaya Sachan
>
> **备注:** Our code (https://github.com/CHIzhP/Chimera) and data (https://huggingface.co/datasets/CHIzhP/Chimera) are publicly available
>
> **摘要:** Diagrams convey symbolic information in a visual format rather than a linear stream of words, making them especially challenging for AI models to process. While recent evaluations suggest that vision-language models (VLMs) perform well on diagram-related benchmarks, their reliance on knowledge, reasoning, or modality shortcuts raises concerns about whether they genuinely understand and reason over diagrams. To address this gap, we introduce Chimera, a comprehensive test suite comprising 7,500 high-quality diagrams sourced from Wikipedia; each diagram is annotated with its symbolic content represented by semantic triples along with multi-level questions designed to assess four fundamental aspects of diagram comprehension: entity recognition, relation understanding, knowledge grounding, and visual reasoning. We use Chimera to measure the presence of three types of shortcuts in visual question answering: (1) the visual-memorization shortcut, where VLMs rely on memorized visual patterns; (2) the knowledge-recall shortcut, where models leverage memorized factual knowledge instead of interpreting the diagram; and (3) the Clever-Hans shortcut, where models exploit superficial language patterns or priors without true comprehension. We evaluate 15 open-source VLMs from 7 model families on Chimera and find that their seemingly strong performance largely stems from shortcut behaviors: visual-memorization shortcuts have slight impact, knowledge-recall shortcuts play a moderate role, and Clever-Hans shortcuts contribute significantly. These findings expose critical limitations in current VLMs and underscore the need for more robust evaluation protocols that benchmark genuine comprehension of complex visual inputs (e.g., diagrams) rather than question-answering shortcuts.
>
---
#### [new 069] On Code-Induced Reasoning in LLMs
- **分类: cs.CL; cs.PL**

- **简介: 该论文研究代码数据如何提升大语言模型（LLM）的推理能力，通过构建多语言指令数据集并进行结构与语义扰动实验，分析不同代码特性对模型性能的影响。属于LLM训练数据设计任务，旨在揭示代码增强推理的关键因素。**

- **链接: [http://arxiv.org/pdf/2509.21499v1](http://arxiv.org/pdf/2509.21499v1)**

> **作者:** Abdul Waheed; Zhen Wu; Carolyn Rosé; Daphne Ippolito
>
> **摘要:** Code data has been shown to enhance the reasoning capabilities of large language models (LLMs), but it remains unclear which aspects of code are most responsible. We investigate this question with a systematic, data-centric framework. We construct parallel instruction datasets in ten programming languages and apply controlled perturbations that selectively disrupt structural or semantic properties of code. We then finetune LLMs from five model families and eight scales on each variant and evaluate their performance on natural language, math, and code tasks. Across 3,331 experiments, our results show that LLMs are more vulnerable to structural perturbations than semantic ones, particularly on math and code tasks. Appropriate abstractions like pseudocode and flowcharts can be as effective as code, while encoding the same information with fewer tokens without adhering to original syntax can often retain or even improve performance. Remarkably, even corrupted code with misleading signals remains competitive when surface-level regularities persist. Finally, syntactic styles also shape task-specific gains with Python favoring natural language reasoning and lower-level languages such as Java and Rust favoring math. Through our systematic framework, we aim to provide insight into how different properties of code influence reasoning and inform the design of training data for enhancing LLM reasoning capabilities.
>
---
#### [new 070] Semantic Agreement Enables Efficient Open-Ended LLM Cascades
- **分类: cs.CL**

- **简介: 该论文研究LLM部署中的成本与质量平衡问题，提出基于语义一致性的级联方法。通过模型输出的语义共识判断是否需调用大模型，在无需训练和模型内部信息的情况下，实现40%成本和60%延迟降低，效果优于目标模型。**

- **链接: [http://arxiv.org/pdf/2509.21837v1](http://arxiv.org/pdf/2509.21837v1)**

> **作者:** Duncan Soiffer; Steven Kolawole; Virginia Smith
>
> **备注:** EMNLP 2025 Industry Track
>
> **摘要:** Cascade systems route computational requests to smaller models when possible and defer to larger models only when necessary, offering a promising approach to balance cost and quality in LLM deployment. However, they face a fundamental challenge in open-ended text generation: determining output reliability when generation quality lies on a continuous spectrum, often with multiple valid responses. To address this, we propose semantic agreement -- meaning-level consensus between ensemble outputs -- as a training-free signal for reliable deferral. We show that when diverse model outputs agree semantically, their consensus is a stronger reliability signal than token-level confidence. Evaluated from 500M to 70B-parameter models, we find that semantic cascades match or surpass target-model quality at 40% of the cost and reduce latency by up to 60%. Our method requires no model internals, works across black-box APIs, and remains robust to model updates, making it a practical baseline for real-world LLM deployment.
>
---
#### [new 071] OjaKV: Context-Aware Online Low-Rank KV Cache Compression with Oja's Rule
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出OjaKV，用于解决大语言模型长上下文生成中的KV缓存内存瓶颈问题。通过结合全秩存储关键token与在线低秩压缩中间token的方法，实现高效且准确的上下文感知KV缓存压缩，无需微调模型。**

- **链接: [http://arxiv.org/pdf/2509.21623v1](http://arxiv.org/pdf/2509.21623v1)**

> **作者:** Yuxuan Zhu; David H. Yang; Mohammad Mohammadi Amiri; Keerthiram Murugesan; Tejaswini Pedapati; Pin-Yu Chen
>
> **摘要:** The expanding long-context capabilities of large language models are constrained by a significant memory bottleneck: the key-value (KV) cache required for autoregressive generation. This bottleneck is substantial; for instance, a Llama-3.1-8B model processing a 32K-token prompt at a batch size of 4 requires approximately 16GB for its KV cache, a size exceeding the model's weights. While KV-cache compression via low-rank projection is a promising direction, existing methods rely on a static, offline-learned subspace that performs poorly under data distribution shifts. To overcome these limitations, we introduce OjaKV, a novel framework that integrates a strategic hybrid storage policy with online subspace adaptation. First, OjaKV recognizes that not all tokens are equally important for compression; it preserves the crucial first and most recent tokens in full-rank, maintaining high-fidelity anchors for attention. Second, for the vast majority of intermediate tokens, it applies low-rank compression by incrementally adapting the projection basis using Oja's algorithm for online principal component analysis. This adaptation involves a comprehensive update during prompt prefilling and lightweight periodic updates during decoding, ensuring the subspace remains aligned with the evolving context. Crucially, our framework is fully compatible with modern attention modules like FlashAttention. Experiments demonstrate that OjaKV maintains or even improves zero-shot accuracy at high compression ratios. In particular, OjaKV achieves its strongest gains on very long-context benchmarks that require complex reasoning, highlighting the importance of online subspace adaptation in dynamically tracking context shifts. These results establish our hybrid framework as a practical, plug-and-play solution for memory-efficient long-context inference without requiring model fine-tuning.
>
---
#### [new 072] RedNote-Vibe: A Dataset for Capturing Temporal Dynamics of AI-Generated Text in Social Media
- **分类: cs.CL**

- **简介: 该论文聚焦社交媒体中AI生成文本（AIGT）的时序动态分析任务，旨在解决现有数据集静态、缺乏长期用户互动研究的问题。为此，作者构建了首个5年纵向数据集RedNote-Vibe，并提出了基于心理语言学特征的可解释检测框架PLAD，以提升AIGT检测性能并揭示其与用户互动的关系。**

- **链接: [http://arxiv.org/pdf/2509.22055v1](http://arxiv.org/pdf/2509.22055v1)**

> **作者:** Yudong Li; Yufei Sun; Yuhan Yao; Peiru Yang; Wanyue Li; Jiajun Zou; Yongfeng Huang; Linlin Shen
>
> **摘要:** The proliferation of Large Language Models (LLMs) has led to widespread AI-Generated Text (AIGT) on social media platforms, creating unique challenges where content dynamics are driven by user engagement and evolve over time. However, existing datasets mainly depict static AIGT detection. In this work, we introduce RedNote-Vibe, the first longitudinal (5-years) dataset for social media AIGT analysis. This dataset is sourced from Xiaohongshu platform, containing user engagement metrics (e.g., likes, comments) and timestamps spanning from the pre-LLM period to July 2025, which enables research into the temporal dynamics and user interaction patterns of AIGT. Furthermore, to detect AIGT in the context of social media, we propose PsychoLinguistic AIGT Detection Framework (PLAD), an interpretable approach that leverages psycholinguistic features. Our experiments show that PLAD achieves superior detection performance and provides insights into the signatures distinguishing human and AI-generated content. More importantly, it reveals the complex relationship between these linguistic features and social media engagement. The dataset is available at https://github.com/testuser03158/RedNote-Vibe.
>
---
#### [new 073] We Think, Therefore We Align LLMs to Helpful, Harmless and Honest Before They Go Wrong
- **分类: cs.CL**

- **简介: 该论文研究大型语言模型的多目标对齐任务，旨在解决帮助性、无害性和诚实性（HHH）之间的冲突与一致性问题。提出AMBS方法，通过共享表示和并行分支策略，提升多目标对齐效果，减少遗忘和输出碎片化。**

- **链接: [http://arxiv.org/pdf/2509.22510v1](http://arxiv.org/pdf/2509.22510v1)**

> **作者:** Gautam Siddharth Kashyap; Mark Dras; Usman Naseem
>
> **摘要:** Alignment of Large Language Models (LLMs) along multiple objectives-helpfulness, harmlessness, and honesty (HHH)-is critical for safe and reliable deployment. Prior work has used steering vector-small control signals injected into hidden states-to guide LLM outputs, typically via one-to-one (1-to-1) Transformer decoders. In this setting, optimizing a single alignment objective can inadvertently overwrite representations learned for other objectives, leading to catastrophic forgetting. More recent approaches extend steering vectors via one-to-many (1-to-N) Transformer decoders. While this alleviates catastrophic forgetting, naive multi-branch designs optimize each objective independently, which can cause inference fragmentation-outputs across HHH objectives may become inconsistent. We propose Adaptive Multi-Branch Steering (AMBS), a two-stage 1-to-N framework for unified and efficient multi-objective alignment. In Stage I, post-attention hidden states of the Transformer layer are computed once to form a shared representation. In Stage II, this representation is cloned into parallel branches and steered via a policy-reference mechanism, enabling objective-specific control while maintaining cross-objective consistency. Empirical evaluations on Alpaca, BeaverTails, and TruthfulQA show that AMBS consistently improves HHH alignment across multiple 7B LLM backbones. For example, on DeepSeek-7B, AMBS improves average alignment scores by +32.4% and reduces unsafe outputs by 11.0% compared to a naive 1-to-N baseline, while remaining competitive with state-of-the-art methods.
>
---
#### [new 074] Black-Box Hallucination Detection via Consistency Under the Uncertain Expression
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型生成内容中的“幻觉”问题。作者提出了一种基于不确定表达一致性的黑盒检测方法，在无需模型内部信息的情况下有效识别不实回答。**

- **链接: [http://arxiv.org/pdf/2509.21999v1](http://arxiv.org/pdf/2509.21999v1)**

> **作者:** Seongho Joo; Kyungmin Min; Jahyun Koo; Kyomin Jung
>
> **摘要:** Despite the great advancement of Language modeling in recent days, Large Language Models (LLMs) such as GPT3 are notorious for generating non-factual responses, so-called "hallucination" problems. Existing methods for detecting and alleviating this hallucination problem require external resources or the internal state of LLMs, such as the output probability of each token. Given the LLM's restricted external API availability and the limited scope of external resources, there is an urgent demand to establish the Black-Box approach as the cornerstone for effective hallucination detection. In this work, we propose a simple black-box hallucination detection metric after the investigation of the behavior of LLMs under expression of uncertainty. Our comprehensive analysis reveals that LLMs generate consistent responses when they present factual responses while non-consistent responses vice versa. Based on the analysis, we propose an efficient black-box hallucination detection metric with the expression of uncertainty. The experiment demonstrates that our metric is more predictive of the factuality in model responses than baselines that use internal knowledge of LLMs.
>
---
#### [new 075] Fine-Grained Detection of Context-Grounded Hallucinations Using LLMs
- **分类: cs.CL**

- **简介: 该论文聚焦于利用大语言模型（LLMs）检测文本生成中的上下文无关幻觉问题。针对现有评估方法复杂且缺乏基准的问题，作者构建了一个包含1,000个人工标注样本的基准，并提出了基于自由文本描述的新表示方式。通过实验分析，揭示了LLMs在此任务上的挑战与优化策略。**

- **链接: [http://arxiv.org/pdf/2509.22582v1](http://arxiv.org/pdf/2509.22582v1)**

> **作者:** Yehonatan Pesiakhovsky; Zorik Gekhman; Yosi Mass; Liat Ein-Dor; Roi Reichart
>
> **摘要:** Context-grounded hallucinations are cases where model outputs contain information not verifiable against the source text. We study the applicability of LLMs for localizing such hallucinations, as a more practical alternative to existing complex evaluation pipelines. In the absence of established benchmarks for meta-evaluation of hallucinations localization, we construct one tailored to LLMs, involving a challenging human annotation of over 1,000 examples. We complement the benchmark with an LLM-based evaluation protocol, verifying its quality in a human evaluation. Since existing representations of hallucinations limit the types of errors that can be expressed, we propose a new representation based on free-form textual descriptions, capturing the full range of possible errors. We conduct a comprehensive study, evaluating four large-scale LLMs, which highlights the benchmark's difficulty, as the best model achieves an F1 score of only 0.67. Through careful analysis, we offer insights into optimal prompting strategies for the task and identify the main factors that make it challenging for LLMs: (1) a tendency to incorrectly flag missing details as inconsistent, despite being instructed to check only facts in the output; and (2) difficulty with outputs containing factually correct information absent from the source - and thus not verifiable - due to alignment with the model's parametric knowledge.
>
---
#### [new 076] Bridging Draft Policy Misalignment: Group Tree Optimization for Speculative Decoding
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型推理中的推测解码任务，旨在解决草案策略与解码策略不一致导致的效率问题。提出Group Tree Optimization（GTO）方法，通过Draft Tree Reward和基于组的草案策略训练，提升草案接受长度和推理速度。**

- **链接: [http://arxiv.org/pdf/2509.22134v1](http://arxiv.org/pdf/2509.22134v1)**

> **作者:** Shijing Hu; Jingyang Li; Zhihui Lu; Pan Zhou
>
> **摘要:** Speculative decoding accelerates large language model (LLM) inference by letting a lightweight draft model propose multiple tokens that the target model verifies in parallel. Yet existing training objectives optimize only a single greedy draft path, while decoding follows a tree policy that re-ranks and verifies multiple branches. This draft policy misalignment limits achievable speedups. We introduce Group Tree Optimization (GTO), which aligns training with the decoding-time tree policy through two components: (i) Draft Tree Reward, a sampling-free objective equal to the expected acceptance length of the draft tree under the target model, directly measuring decoding performance; (ii) Group-based Draft Policy Training, a stable optimization scheme that contrasts trees from the current and a frozen reference draft model, forming debiased group-standardized advantages and applying a PPO-style surrogate along the longest accepted sequence for robust updates. We further prove that increasing our Draft Tree Reward provably improves acceptance length and speedup. Across dialogue (MT-Bench), code (HumanEval), and math (GSM8K), and multiple LLMs (e.g., LLaMA-3.1-8B, LLaMA-3.3-70B, Vicuna-1.3-13B, DeepSeek-R1-Distill-LLaMA-8B), GTO increases acceptance length by 7.4% and yields an additional 7.7% speedup over prior state-of-the-art EAGLE-3. By bridging draft policy misalignment, GTO offers a practical, general solution for efficient LLM inference.
>
---
#### [new 077] Fine-tuning Done Right in Model Editing
- **分类: cs.CL**

- **简介: 该论文属于模型编辑任务，旨在解决微调方法在编辑大语言模型时效果不佳的问题。作者发现传统深度优先的微调方式导致过优化和编辑干扰，并提出恢复广度优先微调框架及局部调整策略LocFT-BF，显著提升编辑性能并扩展至大规模模型。**

- **链接: [http://arxiv.org/pdf/2509.22072v1](http://arxiv.org/pdf/2509.22072v1)**

> **作者:** Wanli Yang; Fei Sun; Rui Tang; Hongyu Zang; Du Su; Qi Cao; Jingang Wang; Huawei Shen; Xueqi Cheng
>
> **摘要:** Fine-tuning, a foundational method for adapting large language models, has long been considered ineffective for model editing. Here, we challenge this belief, arguing that the reported failure arises not from the inherent limitation of fine-tuning itself, but from adapting it to the sequential nature of the editing task, a single-pass depth-first pipeline that optimizes each sample to convergence before moving on. While intuitive, this depth-first pipeline coupled with sample-wise updating over-optimizes each edit and induces interference across edits. Our controlled experiments reveal that simply restoring fine-tuning to the standard breadth-first (i.e., epoch-based) pipeline with mini-batch optimization substantially improves its effectiveness for model editing. Moreover, fine-tuning in editing also suffers from suboptimal tuning parameter locations inherited from prior methods. Through systematic analysis of tuning locations, we derive LocFT-BF, a simple and effective localized editing method built on the restored fine-tuning framework. Extensive experiments across diverse LLMs and datasets demonstrate that LocFT-BF outperforms state-of-the-art methods by large margins. Notably, to our knowledge, it is the first to sustain 100K edits and 72B-parameter models,10 x beyond prior practice, without sacrificing general capabilities. By clarifying a long-standing misconception and introducing a principled localized tuning strategy, we advance fine-tuning from an underestimated baseline to a leading method for model editing, establishing a solid foundation for future research.
>
---
#### [new 078] GraphSearch: An Agentic Deep Searching Workflow for Graph Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文提出GraphSearch，一种面向图检索增强生成（GraphRAG）的智能深度搜索流程。针对现有方法在检索深度和图数据利用效率上的不足，设计了包含六模块的多轮交互框架，并采用双通道检索策略，提升复杂查询下的推理效果。**

- **链接: [http://arxiv.org/pdf/2509.22009v1](http://arxiv.org/pdf/2509.22009v1)**

> **作者:** Cehao Yang; Xiaojun Wu; Xueyuan Lin; Chengjin Xu; Xuhui Jiang; Yuanliang Sun; Jia Li; Hui Xiong; Jian Guo
>
> **摘要:** Graph Retrieval-Augmented Generation (GraphRAG) enhances factual reasoning in LLMs by structurally modeling knowledge through graph-based representations. However, existing GraphRAG approaches face two core limitations: shallow retrieval that fails to surface all critical evidence, and inefficient utilization of pre-constructed structural graph data, which hinders effective reasoning from complex queries. To address these challenges, we propose \textsc{GraphSearch}, a novel agentic deep searching workflow with dual-channel retrieval for GraphRAG. \textsc{GraphSearch} organizes the retrieval process into a modular framework comprising six modules, enabling multi-turn interactions and iterative reasoning. Furthermore, \textsc{GraphSearch} adopts a dual-channel retrieval strategy that issues semantic queries over chunk-based text data and relational queries over structural graph data, enabling comprehensive utilization of both modalities and their complementary strengths. Experimental results across six multi-hop RAG benchmarks demonstrate that \textsc{GraphSearch} consistently improves answer accuracy and generation quality over the traditional strategy, confirming \textsc{GraphSearch} as a promising direction for advancing graph retrieval-augmented generation.
>
---
#### [new 079] NFDI4DS Shared Tasks for Scholarly Document Processing
- **分类: cs.CL**

- **简介: 该论文介绍了NFDI4DS联盟下的12项共享任务，旨在推动学术文档处理的研究。通过标准化评估促进FAIR原则与可重复性研究，开发并开放了数据集、模型和工具，以支持方法创新与科研社区发展。**

- **链接: [http://arxiv.org/pdf/2509.22141v1](http://arxiv.org/pdf/2509.22141v1)**

> **作者:** Raia Abu Ahmad; Rana Abdulla; Tilahun Abedissa Taffa; Soeren Auer; Hamed Babaei Giglou; Ekaterina Borisova; Zongxiong Chen; Stefan Dietze; Jennifer DSouza; Mayra Elwes; Genet-Asefa Gesese; Shufan Jiang; Ekaterina Kutafina; Philipp Mayr; Georg Rehm; Sameer Sadruddin; Sonja Schimmler; Daniel Schneider; Kanishka Silva; Sharmila Upadhyaya; Ricardo Usbeck
>
> **备注:** Accepted at the RDI4DS 2025 Workshop
>
> **摘要:** Shared tasks are powerful tools for advancing research through community-based standardised evaluation. As such, they play a key role in promoting findable, accessible, interoperable, and reusable (FAIR), as well as transparent and reproducible research practices. This paper presents an updated overview of twelve shared tasks developed and hosted under the German National Research Data Infrastructure for Data Science and Artificial Intelligence (NFDI4DS) consortium, covering a diverse set of challenges in scholarly document processing. Hosted at leading venues, the tasks foster methodological innovations and contribute open-access datasets, models, and tools for the broader research community, which are integrated into the consortium's research data infrastructure.
>
---
#### [new 080] Mixture of Detectors: A Compact View of Machine-Generated Text Detection
- **分类: cs.CL**

- **简介: 该论文聚焦于机器生成文本检测（MGTD）任务，旨在解决文本真实性问题。提出BMAS English数据集，支持二分类、多分类、生成器归属和对抗攻击分析，推动检测方法的全面性与实用性。**

- **链接: [http://arxiv.org/pdf/2509.22147v1](http://arxiv.org/pdf/2509.22147v1)**

> **作者:** Sai Teja Lekkala; Yadagiri Annepaka; Arun Kumar Challa; Samatha Reddy Machireddy; Partha Pakray; Chukhu Chunka
>
> **备注:** 20 pages, 3 figures
>
> **摘要:** Large Language Models (LLMs) are gearing up to surpass human creativity. The veracity of the statement needs careful consideration. In recent developments, critical questions arise regarding the authenticity of human work and the preservation of their creativity and innovative abilities. This paper investigates such issues. This paper addresses machine-generated text detection across several scenarios, including document-level binary and multiclass classification or generator attribution, sentence-level segmentation to differentiate between human-AI collaborative text, and adversarial attacks aimed at reducing the detectability of machine-generated text. We introduce a new work called BMAS English: an English language dataset for binary classification of human and machine text, for multiclass classification, which not only identifies machine-generated text but can also try to determine its generator, and Adversarial attack addressing where it is a common act for the mitigation of detection, and Sentence-level segmentation, for predicting the boundaries between human and machine-generated text. We believe that this paper will address previous work in Machine-Generated Text Detection (MGTD) in a more meaningful way.
>
---
#### [new 081] Redefining Machine Simultaneous Interpretation: From Incremental Translation to Human-Like Strategies
- **分类: cs.CL**

- **简介: 该论文研究实时机器同声传译任务，旨在提升翻译质量与实时性。针对传统方法的不足，提出四种新操作策略，并在LLM框架中实现，通过实验验证其在语义准确性和延迟上的优势。**

- **链接: [http://arxiv.org/pdf/2509.21801v1](http://arxiv.org/pdf/2509.21801v1)**

> **作者:** Qianen Zhang; Satoshi Nakamura
>
> **摘要:** Simultaneous Machine Translation (SiMT) requires high-quality translations under strict real-time constraints, which traditional encoder-decoder policies with only READ/WRITE actions cannot fully address. We extend the action space of SiMT with four adaptive actions: SENTENCE_CUT, DROP, PARTIAL_SUMMARIZATION and PRONOMINALIZATION, which enable real-time restructuring, omission, and simplification while preserving semantic fidelity. We implement these actions in a decoder-only large language model (LLM) framework and construct training references through action-aware prompting. To evaluate both quality and latency, we further develop a latency-aware TTS pipeline that maps textual outputs to speech with realistic timing. Experiments on the ACL60/60 English-Chinese and English-German benchmarks show that our framework consistently improves semantic metrics (e.g., COMET-KIWI) and achieves lower delay (measured by Average Lagging) compared to reference translations and salami-based baselines. Notably, combining DROP and SENTENCE_CUT yields the best overall balance between fluency and latency. These results demonstrate that enriching the action space of LLM-based SiMT provides a promising direction for bridging the gap between human and machine interpretation.
>
---
#### [new 082] Domain-Aware Speaker Diarization On African-Accented English
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究了针对非洲口音英语的说话人日志任务，旨在解决临床对话中领域差异导致的性能下降问题。作者评估了多个系统，分析错误来源，并通过微调分割模块进行轻量级领域适配，提出了改进方向和可复现的方法。**

- **链接: [http://arxiv.org/pdf/2509.21554v1](http://arxiv.org/pdf/2509.21554v1)**

> **作者:** Chibuzor Okocha; Kelechi Ezema; Christan Grant
>
> **备注:** 5 pages
>
> **摘要:** This study examines domain effects in speaker diarization for African-accented English. We evaluate multiple production and open systems on general and clinical dialogues under a strict DER protocol that scores overlap. A consistent domain penalty appears for clinical speech and remains significant across models. Error analysis attributes much of this penalty to false alarms and missed detections, aligning with short turns and frequent overlap. We test lightweight domain adaptation by fine-tuning a segmentation module on accent-matched data; it reduces error but does not eliminate the gap. Our contributions include a controlled benchmark across domains, a concise approach to error decomposition and conversation-level profiling, and an adaptation recipe that is easy to reproduce. Results point to overlap-aware segmentation and balanced clinical resources as practical next steps.
>
---
#### [new 083] R-Capsule: Compressing High-Level Plans for Efficient Large Language Model Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出R-Capsule框架，用于压缩大语言模型的推理过程。针对传统Chain-of-Thought（CoT）方法效率低、冗余高的问题，R-Capsule通过信息瓶颈原理，将高层推理计划编码为少量隐状态，同时保持执行步骤简洁透明，从而在复杂任务中实现高效且准确的推理。**

- **链接: [http://arxiv.org/pdf/2509.22131v1](http://arxiv.org/pdf/2509.22131v1)**

> **作者:** Hongyu Shan; Mingyang Song; Chang Dai; Di Liang; Han Chen
>
> **摘要:** Chain-of-Thought (CoT) prompting helps Large Language Models (LLMs) tackle complex reasoning by eliciting explicit step-by-step rationales. However, CoT's verbosity increases latency and memory usage and may propagate early errors across long chains. We propose the Reasoning Capsule (R-Capsule), a framework that aims to combine the efficiency of latent reasoning with the transparency of explicit CoT. The core idea is to compress the high-level plan into a small set of learned latent tokens (a Reasoning Capsule) while keeping execution steps lightweight or explicit. This hybrid approach is inspired by the Information Bottleneck (IB) principle, where we encourage the capsule to be approximately minimal yet sufficient for the task. Minimality is encouraged via a low-capacity bottleneck, which helps improve efficiency. Sufficiency is encouraged via a dual objective: a primary task loss for answer accuracy and an auxiliary plan-reconstruction loss that encourages the capsule to faithfully represent the original textual plan. The reconstruction objective helps ground the latent space, thereby improving interpretability and reducing the use of uninformative shortcuts. Our framework strikes a balance between efficiency, accuracy, and interpretability, thereby reducing the visible token footprint of reasoning while maintaining or improving accuracy on complex benchmarks. Our codes are available at: https://anonymous.4open.science/r/Reasoning-Capsule-7BE0
>
---
#### [new 084] GRAB: A Risk Taxonomy--Grounded Benchmark for Unsupervised Topic Discovery in Financial Disclosures
- **分类: cs.CL**

- **简介: 该论文提出了GRAB，一个面向金融披露文本的无监督主题发现基准。针对10-K风险披露中缺乏评估模型的公开基准问题，构建了包含1.61M句子的数据集，并通过弱监督生成细粒度风险标签，统一评估多种主题模型。**

- **链接: [http://arxiv.org/pdf/2509.21698v1](http://arxiv.org/pdf/2509.21698v1)**

> **作者:** Ying Li; Tiejun Ma
>
> **备注:** 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: NeurIPS 2025 Workshop on Generative AI in Finance
>
> **摘要:** Risk categorization in 10-K risk disclosures matters for oversight and investment, yet no public benchmark evaluates unsupervised topic models for this task. We present GRAB, a finance-specific benchmark with 1.61M sentences from 8,247 filings and span-grounded sentence labels produced without manual annotation by combining FinBERT token attention, YAKE keyphrase signals, and taxonomy-aware collocation matching. Labels are anchored in a risk taxonomy mapping 193 terms to 21 fine-grained types nested under five macro classes; the 21 types guide weak supervision, while evaluation is reported at the macro level. GRAB unifies evaluation with fixed dataset splits and robust metrics--Accuracy, Macro-F1, Topic BERTScore, and the entropy-based Effective Number of Topics. The dataset, labels, and code enable reproducible, standardized comparison across classical, embedding-based, neural, and hybrid topic models on financial disclosures.
>
---
#### [new 085] Following the TRACE: A Structured Path to Empathetic Response Generation with Multi-Agent Models
- **分类: cs.CL; cs.MA**

- **简介: 该论文聚焦于共情响应生成任务，旨在解决专用模型分析深度与大语言模型生成流畅性之间的矛盾。提出TRACE框架，通过任务分解将共情建模为结构化认知过程，结合深入分析与表达生成，实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2509.21849v1](http://arxiv.org/pdf/2509.21849v1)**

> **作者:** Ziqi Liu; Ziyang Zhou; Yilin Li; Haiyang Zhang; Yangbin Chen
>
> **摘要:** Empathetic response generation is a crucial task for creating more human-like and supportive conversational agents. However, existing methods face a core trade-off between the analytical depth of specialized models and the generative fluency of Large Language Models (LLMs). To address this, we propose TRACE, Task-decomposed Reasoning for Affective Communication and Empathy, a novel framework that models empathy as a structured cognitive process by decomposing the task into a pipeline for analysis and synthesis. By building a comprehensive understanding before generation, TRACE unites deep analysis with expressive generation. Experimental results show that our framework significantly outperforms strong baselines in both automatic and LLM-based evaluations, confirming that our structured decomposition is a promising paradigm for creating more capable and interpretable empathetic agents. Our code is available at https://anonymous.4open.science/r/TRACE-18EF/README.md.
>
---
#### [new 086] FeatBench: Evaluating Coding Agents on Feature Implementation for Vibe Coding
- **分类: cs.CL; cs.AI; cs.SE**

- **简介: 该论文提出了FeatBench，一个针对“vibe coding”范式下功能实现的评估基准。现有代码生成基准无法有效评估基于自然语言交互的编码代理能力，尤其在功能实现方面存在不足。FeatBench通过纯自然语言提示、严格的数据收集流程和全面测试用例，填补了这一空白，并揭示了当前代理在该任务上的低成功率（最高29.94%）。**

- **链接: [http://arxiv.org/pdf/2509.22237v1](http://arxiv.org/pdf/2509.22237v1)**

> **作者:** Haorui Chen; Chengze Li; Jia Li
>
> **摘要:** The rapid advancement of Large Language Models (LLMs) has given rise to a novel software development paradigm known as "vibe coding," where users interact with coding agents through high-level natural language. However, existing evaluation benchmarks for code generation inadequately assess an agent's vibe coding capabilities. Existing benchmarks are misaligned, as they either require code-level specifications or focus narrowly on issue-solving, neglecting the critical scenario of feature implementation within the vibe coding paradiam. To address this gap, we propose FeatBench, a novel benchmark for vibe coding that focuses on feature implementation. Our benchmark is distinguished by several key features: 1. Pure Natural Language Prompts. Task inputs consist solely of abstract natural language descriptions, devoid of any code or structural hints. 2. A Rigorous & Evolving Data Collection Process. FeatBench is built on a multi-level filtering pipeline to ensure quality and a fully automated pipeline to evolve the benchmark, mitigating data contamination. 3. Comprehensive Test Cases. Each task includes Fail-to-Pass (F2P) and Pass-to-Pass (P2P) tests to verify correctness and prevent regressions. 4. Diverse Application Domains. The benchmark includes repositories from diverse domains to ensure it reflects real-world scenarios. We evaluate two state-of-the-art agent frameworks with four leading LLMs on FeatBench. Our evaluation reveals that feature implementation within the vibe coding paradigm is a significant challenge, with the highest success rate of only 29.94%. Our analysis also reveals a tendency for "aggressive implementation," a strategy that paradoxically leads to both critical failures and superior software design. We release FeatBench, our automated collection pipeline, and all experimental results to facilitate further community research.
>
---
#### [new 087] FLEXI: Benchmarking Full-duplex Human-LLM Speech Interaction
- **分类: cs.CL**

- **简介: 该论文提出FLEXI，首个评估全双工人机语音交互的基准，重点测试紧急场景下的模型中断能力。通过六个交互场景，揭示开源与商业模型在延迟、对话质量等方面的差距，并建议采用下一对词预测提升交互自然度。**

- **链接: [http://arxiv.org/pdf/2509.22243v1](http://arxiv.org/pdf/2509.22243v1)**

> **作者:** Yuan Ge; Saihan Chen; Jingqi Xiao; Xiaoqian Liu; Tong Xiao; Yan Xiang; Zhengtao Yu; Jingbo Zhu
>
> **摘要:** Full-Duplex Speech-to-Speech Large Language Models (LLMs) are foundational to natural human-computer interaction, enabling real-time spoken dialogue systems. However, benchmarking and modeling these models remains a fundamental challenge. We introduce FLEXI, the first benchmark for full-duplex LLM-human spoken interaction that explicitly incorporates model interruption in emergency scenarios. FLEXI systematically evaluates the latency, quality, and conversational effectiveness of real-time dialogue through six diverse human-LLM interaction scenarios, revealing significant gaps between open source and commercial models in emergency awareness, turn terminating, and interaction latency. Finally, we suggest that next token-pair prediction offers a promising path toward achieving truly seamless and human-like full-duplex interaction.
>
---
#### [new 088] How Large Language Models Need Symbolism
- **分类: cs.CL; cs.AI**

- **简介: 该论文探讨AI未来发展，认为仅靠模型规模不足，需引入人工符号引导语言模型的直觉，以实现真正发现。属于理论研究任务，旨在解决大模型缺乏方向性的问题。**

- **链接: [http://arxiv.org/pdf/2509.21404v1](http://arxiv.org/pdf/2509.21404v1)**

> **作者:** Xiaotie Deng; Hanyu Li
>
> **摘要:** We argue that AI's future requires more than scaling. To unlock genuine discovery, large language models need a compass: human-crafted symbols to guide their powerful but blind intuition.
>
---
#### [new 089] Language Models Can Learn from Verbal Feedback Without Scalar Rewards
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出一种反馈条件策略（FCP），用于语言模型直接从口头反馈中学习，无需标量奖励。通过离线最大似然训练和在线引导优化，将反馈驱动学习转化为条件生成任务，提升模型对复杂反馈的利用能力。**

- **链接: [http://arxiv.org/pdf/2509.22638v1](http://arxiv.org/pdf/2509.22638v1)**

> **作者:** Renjie Luo; Zichen Liu; Xiangyan Liu; Chao Du; Min Lin; Wenhu Chen; Wei Lu; Tianyu Pang
>
> **摘要:** LLMs are often trained with RL from human or AI feedback, yet such methods typically compress nuanced feedback into scalar rewards, discarding much of their richness and inducing scale imbalance. We propose treating verbal feedback as a conditioning signal. Inspired by language priors in text-to-image generation, which enable novel outputs from unseen prompts, we introduce the feedback-conditional policy (FCP). FCP learns directly from response-feedback pairs, approximating the feedback-conditional posterior through maximum likelihood training on offline data. We further develop an online bootstrapping stage where the policy generates under positive conditions and receives fresh feedback to refine itself. This reframes feedback-driven learning as conditional generation rather than reward optimization, offering a more expressive way for LLMs to directly learn from verbal feedback. Our code is available at https://github.com/sail-sg/feedback-conditional-policy.
>
---
#### [new 090] Death of the Novel(ty): Beyond n-Gram Novelty as a Metric for Textual Creativity
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文研究文本创造力评估，指出n-gram新颖性不足。通过专家标注分析人类与AI生成文本，发现高n-gram新颖性不等于创造力，并测试模型识别创造性表达的能力。属于自然语言生成评估任务。**

- **链接: [http://arxiv.org/pdf/2509.22641v1](http://arxiv.org/pdf/2509.22641v1)**

> **作者:** Arkadiy Saakyan; Najoung Kim; Smaranda Muresan; Tuhin Chakrabarty
>
> **备注:** 26 pages, 10 figures, under review
>
> **摘要:** N-gram novelty is widely used to evaluate language models' ability to generate text outside of their training data. More recently, it has also been adopted as a metric for measuring textual creativity. However, theoretical work on creativity suggests that this approach may be inadequate, as it does not account for creativity's dual nature: novelty (how original the text is) and appropriateness (how sensical and pragmatic it is). We investigate the relationship between this notion of creativity and n-gram novelty through 7542 expert writer annotations (n=26) of novelty, pragmaticality, and sensicality via close reading of human and AI-generated text. We find that while n-gram novelty is positively associated with expert writer-judged creativity, ~91% of top-quartile expressions by n-gram novelty are not judged as creative, cautioning against relying on n-gram novelty alone. Furthermore, unlike human-written text, higher n-gram novelty in open-source LLMs correlates with lower pragmaticality. In an exploratory study with frontier close-source models, we additionally confirm that they are less likely to produce creative expressions than humans. Using our dataset, we test whether zero-shot, few-shot, and finetuned models are able to identify creative expressions (a positive aspect of writing) and non-pragmatic ones (a negative aspect). Overall, frontier LLMs exhibit performance much higher than random but leave room for improvement, especially struggling to identify non-pragmatic expressions. We further find that LLM-as-a-Judge novelty scores from the best-performing model were predictive of expert writer preferences.
>
---
#### [new 091] Can LLMs Solve and Generate Linguistic Olympiad Puzzles?
- **分类: cs.CL**

- **简介: 该论文研究了大型语言模型（LLMs）在解决和生成语言学奥林匹克竞赛题目中的表现。任务涉及语言学谜题的求解与生成，旨在评估LLM的能力并探索自动化生成的可能性。论文扩展了解题基准，分析了LLM性能，并探讨了其在促进语言学传播中的潜力。**

- **链接: [http://arxiv.org/pdf/2509.21820v1](http://arxiv.org/pdf/2509.21820v1)**

> **作者:** Neh Majmudar; Elena Filatova
>
> **备注:** To be published in the Proceedings of Main Conference of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025)
>
> **摘要:** In this paper, we introduce a combination of novel and exciting tasks: the solution and generation of linguistic puzzles. We focus on puzzles used in Linguistic Olympiads for high school students. We first extend the existing benchmark for the task of solving linguistic puzzles. We explore the use of Large Language Models (LLMs), including recent state-of-the-art models such as OpenAI's o1, for solving linguistic puzzles, analyzing their performance across various linguistic topics. We demonstrate that LLMs outperform humans on most puzzles types, except for those centered on writing systems, and for the understudied languages. We use the insights from puzzle-solving experiments to direct the novel task of puzzle generation. We believe that automating puzzle generation, even for relatively simple puzzles, holds promise for expanding interest in linguistics and introducing the field to a broader audience. This finding highlights the importance of linguistic puzzle generation as a research task: such puzzles can not only promote linguistics but also support the dissemination of knowledge about rare and understudied languages.
>
---
#### [new 092] LLM-Based Support for Diabetes Diagnosis: Opportunities, Scenarios, and Challenges with GPT-5
- **分类: cs.CL**

- **简介: 该论文研究了GPT-5在糖尿病诊断中的应用，属于医疗辅助决策任务。旨在解决糖尿病早期识别难的问题，通过模拟框架测试GPT-5在五个典型场景下的表现，验证其能否提供符合ADA标准的临床支持与患者解释。**

- **链接: [http://arxiv.org/pdf/2509.21450v1](http://arxiv.org/pdf/2509.21450v1)**

> **作者:** Gaurav Kumar Gupta; Nirajan Acharya; Pranal Pande
>
> **摘要:** Diabetes mellitus is a major global health challenge, affecting over half a billion adults worldwide with prevalence projected to rise. Although the American Diabetes Association (ADA) provides clear diagnostic thresholds, early recognition remains difficult due to vague symptoms, borderline laboratory values, gestational complexity, and the demands of long-term monitoring. Advances in large language models (LLMs) offer opportunities to enhance decision support through structured, interpretable, and patient-friendly outputs. This study evaluates GPT-5, the latest generative pre-trained transformer, using a simulation framework built entirely on synthetic cases aligned with ADA Standards of Care 2025 and inspired by public datasets including NHANES, Pima Indians, EyePACS, and MIMIC-IV. Five representative scenarios were tested: symptom recognition, laboratory interpretation, gestational diabetes screening, remote monitoring, and multimodal complication detection. For each, GPT-5 classified cases, generated clinical rationales, produced patient explanations, and output structured JSON summaries. Results showed strong alignment with ADA-defined criteria, suggesting GPT-5 may function as a dual-purpose tool for clinicians and patients, while underscoring the importance of reproducible evaluation frameworks for responsibly assessing LLMs in healthcare.
>
---
#### [new 093] StateX: Enhancing RNN Recall via Post-training State Expansion
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出StateX，一种通过后训练扩展预训练RNN状态的方法，旨在提升其长上下文回忆能力。针对线性注意力和状态空间模型，设计低参数开销的结构修改，有效增强RNN的回忆与上下文学习能力，同时控制训练成本。**

- **链接: [http://arxiv.org/pdf/2509.22630v1](http://arxiv.org/pdf/2509.22630v1)**

> **作者:** Xingyu Shen; Yingfa Chen; Zhen Leng Thai; Xu Han; Zhiyuan Liu; Maosong Sun
>
> **摘要:** While Transformer-based models have demonstrated remarkable language modeling performance, their high complexities result in high costs when processing long contexts. In contrast, recurrent neural networks (RNNs) such as linear attention and state space models have gained popularity due to their constant per-token complexities. However, these recurrent models struggle with tasks that require accurate recall of contextual information from long contexts, because all contextual information is compressed into a constant-size recurrent state. Previous works have shown that recall ability is positively correlated with the recurrent state size, yet directly training RNNs with larger recurrent states results in high training costs. In this paper, we introduce StateX, a training pipeline for efficiently expanding the states of pre-trained RNNs through post-training. For two popular classes of RNNs, linear attention and state space models, we design post-training architectural modifications to scale up the state size with no or negligible increase in model parameters. Experiments on models up to 1.3B parameters demonstrate that StateX efficiently enhances the recall and in-context learning ability of RNNs without incurring high post-training costs or compromising other capabilities.
>
---
#### [new 094] A State-of-the-Art SQL Reasoning Model using RLVR
- **分类: cs.CL; cs.AI; cs.DB; cs.LG**

- **简介: 该论文研究自然语言到SQL的转换任务，旨在解决企业中数据查询自动化的问题。提出基于RLVR的训练方法，结合离线和在线强化学习，在BIRD数据集上达到SOTA性能。**

- **链接: [http://arxiv.org/pdf/2509.21459v1](http://arxiv.org/pdf/2509.21459v1)**

> **作者:** Alnur Ali; Ashutosh Baheti; Jonathan Chang; Ta-Chung Chi; Brandon Cui; Andrew Drozdov; Jonathan Frankle; Abhay Gupta; Pallavi Koppol; Sean Kulinski; Jonathan Li; Dipendra Misra; Krista Opsahl-Ong; Jose Javier Gonzalez Ortiz; Matei Zaharia; Yue Zhang
>
> **摘要:** Developing custom reasoning models via Reinforcement Learning (RL) that can incorporate organization-specific knowledge has great potential to address problems faced by enterprise customers. In many of these problems, the reward function is verifiable, a setting termed RL with Verifiable Rewards (RLVR). We apply RLVR to a popular data science benchmark called BIRD that measures the ability of an AI agent to convert a natural language query for a database to SQL executions. We apply a simple and general-purpose training recipe involving careful prompt and model selection, a warm-up stage using our offline RL approach called TAO, followed by rigorous online RLVR training. With no additional training data beyond the BIRD training set and no use of proprietary models, our very first submission to the BIRD leaderboard reached state-of-the-art accuracy on the private test set: 73.56% without self-consistency and 75.68% with self-consistency. In the latter case, our model also required fewer generations than the second-best approach. While BIRD is only a proxy task, the simplicity of our framework makes it broadly applicable to enterprise domains such as business intelligence, data science, and coding.
>
---
#### [new 095] What Is The Political Content in LLMs' Pre- and Post-Training Data?
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文分析LLM训练数据中的政治内容，探讨其如何影响模型偏见。研究基于OLMO2的预训练和微调数据，统计政治倾向及来源，发现左倾内容占主导，并与模型政策立场相关。任务属于数据偏见分析，旨在揭示LLM政治偏见的成因。**

- **链接: [http://arxiv.org/pdf/2509.22367v1](http://arxiv.org/pdf/2509.22367v1)**

> **作者:** Tanise Ceron; Dmitry Nikolaev; Dominik Stammbach; Debora Nozza
>
> **备注:** 9 pages, under review
>
> **摘要:** Large language models (LLMs) are known to generate politically biased text, yet how such biases arise remains unclear. A crucial step toward answering this question is the analysis of training data, whose political content remains largely underexplored in current LLM research. To address this gap, we present in this paper an analysis of the pre- and post-training corpora of OLMO2, the largest fully open-source model released together with its complete dataset. From these corpora, we draw large random samples, automatically annotate documents for political orientation, and analyze their source domains and content. We then assess how political content in the training data correlates with models' stance on specific policy issues. Our analysis shows that left-leaning documents predominate across datasets, with pre-training corpora containing significantly more politically engaged content than post-training data. We also find that left- and right-leaning documents frame similar topics through distinct values and sources of legitimacy. Finally, the predominant stance in the training data strongly correlates with models' political biases when evaluated on policy issues. These findings underscore the need to integrate political content analysis into future data curation pipelines as well as in-depth documentation of filtering strategies for transparency.
>
---
#### [new 096] No Prompt Left Behind: Exploiting Zero-Variance Prompts in LLM Reinforcement Learning via Entropy-Guided Advantage Shaping
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于大语言模型强化学习任务，旨在解决现有方法忽略零方差提示（所有响应奖励相同）的问题。提出RL-ZVP算法，利用熵引导优势塑造从零方差提示中提取学习信号，提升数学推理性能，在多个基准上优于基线方法。**

- **链接: [http://arxiv.org/pdf/2509.21880v1](http://arxiv.org/pdf/2509.21880v1)**

> **作者:** Thanh-Long V. Le; Myeongho Jeon; Kim Vu; Viet Lai; Eunho Yang
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) is a powerful framework for improving the reasoning abilities of Large Language Models (LLMs). However, current methods such as GRPO rely only on problems where the model responses to the same input differ in correctness, while ignoring those where all responses receive the same reward - so-called zero-variance prompts. In this work, we argue that such prompts are not useless but can, in fact, provide meaningful feedback for policy optimization. To this end, we introduce RL with Zero-Variance Prompts (RL-ZVP), a novel algorithm that extract learning signals from zero-variance prompts. RL-ZVP directly rewards correctness and penalizes errors even without contrasting responses, modulating feedback with token-level characteristics to preserve informative, nuanced signals. Across six math reasoning benchmarks, RL-ZVP achieves significant improvements of up to 8.61 points in accuracy and 7.77 points in pass rate over GRPO, while consistently outperforming other baselines that filter out zero-variance prompts. These results highlight the untapped potential of learning from zero-variance prompts in RLVR.
>
---
#### [new 097] When Does Reasoning Matter? A Controlled Study of Reasoning's Contribution to Model Performance
- **分类: cs.CL**

- **简介: 该论文研究了推理能力对大模型性能的影响，通过合成数据蒸馏框架对比了指令微调与推理模型在数学和通用任务上的表现。工作重点在于分析推理在不同任务、模型规模下的有效性及成本问题。**

- **链接: [http://arxiv.org/pdf/2509.22193v1](http://arxiv.org/pdf/2509.22193v1)**

> **作者:** Nicolas Boizard; Hippolyte Gisserot-Boukhlef; Kevin El-Haddad; Céline Hudelot; Pierre Colombo
>
> **摘要:** Large Language Models (LLMs) with reasoning capabilities have achieved state-of-the-art performance on a wide range of tasks. Despite its empirical success, the tasks and model scales at which reasoning becomes effective, as well as its training and inference costs, remain underexplored. In this work, we rely on a synthetic data distillation framework to conduct a large-scale supervised study. We compare Instruction Fine-Tuning (IFT) and reasoning models of varying sizes, on a wide range of math-centric and general-purpose tasks, evaluating both multiple-choice and open-ended formats. Our analysis reveals that reasoning consistently improves model performance, often matching or surpassing significantly larger IFT systems. Notably, while IFT remains Pareto-optimal in training and inference costs, reasoning models become increasingly valuable as model size scales, overcoming IFT performance limits on reasoning-intensive and open-ended tasks.
>
---
#### [new 098] Self-Speculative Biased Decoding for Faster Live Translation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对实时翻译中的高延迟问题，提出一种无需草案计算的自推测偏置解码方法。通过复用最新输出作为草案并提高接受率，实现1.7倍加速，同时减少80%的界面闪烁，适用于流式文本生成任务。**

- **链接: [http://arxiv.org/pdf/2509.21740v1](http://arxiv.org/pdf/2509.21740v1)**

> **作者:** Linxiao Zeng; Haoyun Deng; Kangyuan Shu; Shizhen Wang
>
> **摘要:** Large Language Models (LLMs) have recently demonstrated impressive capabilities in various text generation tasks. However, it remains challenging to use them off-the-shelf in streaming applications (such as live translation), where the output must continually update as the input context expands, while still maintaining a reasonable computational cost to meet the latency requirement. In this work, we reexamine the re-translation approach to simultaneous translation and propose Self-Speculative Biased Decoding, a novel inference paradigm designed to avoid repeatedly generating output from scratch for a consistently growing input stream. We propose using the most recent output as a draft for the current growing input context. During the verification stage, the output will be biased towards the draft token for a higher draft acceptance rate. This strategy not only minimizes flickering that might distract users but also leads to higher speedups. Conventional decoding may take charge from the point of divergence after draft verification and continue until the end condition is met. Unlike existing speculative decoding strategies, our approach eliminates the need for draft computations, making it a model-agnostic and plug-and-play solution for accelerating latency-sensitive streaming applications. Experimental results on simultaneous text-to-text re-translation demonstrate that our approach achieves up to 1.7x speedup compared to conventional auto-regressive re-translation without compromising quality. Additionally, it significantly reduces flickering by 80% by incorporating the display-only mask-k technique.
>
---
#### [new 099] Safety Compliance: Rethinking LLM Safety Reasoning through the Lens of Compliance
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦LLM安全合规性任务，旨在解决现有安全方法缺乏系统性和法律依据的问题。作者提出以欧盟AI法案和GDPR为标准，构建安全合规基准，并使用GRPO训练合规推理模型，有效提升LLM在法律框架下的安全性。**

- **链接: [http://arxiv.org/pdf/2509.22250v1](http://arxiv.org/pdf/2509.22250v1)**

> **作者:** Wenbin Hu; Huihao Jing; Haochen Shi; Haoran Li; Yangqiu Song
>
> **摘要:** The proliferation of Large Language Models (LLMs) has demonstrated remarkable capabilities, elevating the critical importance of LLM safety. However, existing safety methods rely on ad-hoc taxonomy and lack a rigorous, systematic protection, failing to ensure safety for the nuanced and complex behaviors of modern LLM systems. To address this problem, we solve LLM safety from legal compliance perspectives, named safety compliance. In this work, we posit relevant established legal frameworks as safety standards for defining and measuring safety compliance, including the EU AI Act and GDPR, which serve as core legal frameworks for AI safety and data security in Europe. To bridge the gap between LLM safety and legal compliance, we first develop a new benchmark for safety compliance by generating realistic LLM safety scenarios seeded with legal statutes. Subsequently, we align Qwen3-8B using Group Policy Optimization (GRPO) to construct a safety reasoner, Compliance Reasoner, which effectively aligns LLMs with legal standards to mitigate safety risks. Our comprehensive experiments demonstrate that the Compliance Reasoner achieves superior performance on the new benchmark, with average improvements of +10.45% for the EU AI Act and +11.85% for GDPR.
>
---
#### [new 100] Towards Minimal Causal Representations for Human Multimodal Language Understanding
- **分类: cs.CL**

- **简介: 该论文针对多模态语言理解任务，旨在解决模型因数据集偏差导致的因果特征混淆问题。提出CaMIB模型，通过信息瓶颈和因果约束分离因果与捷径特征，提升模型的泛化能力与可解释性。**

- **链接: [http://arxiv.org/pdf/2509.21805v1](http://arxiv.org/pdf/2509.21805v1)**

> **作者:** Menghua Jiang; Yuncheng Jiang; Haifeng Hu; Sijie Mai
>
> **摘要:** Human Multimodal Language Understanding (MLU) aims to infer human intentions by integrating related cues from heterogeneous modalities. Existing works predominantly follow a ``learning to attend" paradigm, which maximizes mutual information between data and labels to enhance predictive performance. However, such methods are vulnerable to unintended dataset biases, causing models to conflate statistical shortcuts with genuine causal features and resulting in degraded out-of-distribution (OOD) generalization. To alleviate this issue, we introduce a Causal Multimodal Information Bottleneck (CaMIB) model that leverages causal principles rather than traditional likelihood. Concretely, we first applies the information bottleneck to filter unimodal inputs, removing task-irrelevant noise. A parameterized mask generator then disentangles the fused multimodal representation into causal and shortcut subrepresentations. To ensure global consistency of causal features, we incorporate an instrumental variable constraint, and further adopt backdoor adjustment by randomly recombining causal and shortcut features to stabilize causal estimation. Extensive experiments on multimodal sentiment analysis, humor detection, and sarcasm detection, along with OOD test sets, demonstrate the effectiveness of CaMIB. Theoretical and empirical analyses further highlight its interpretability and soundness.
>
---
#### [new 101] Influence Guided Context Selection for Effective Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦于检索增强生成（RAG）任务，旨在解决因低质量检索上下文导致的效果下降问题。提出基于上下文影响值（CI值）的筛选方法，结合查询、上下文列表和生成器信息，有效过滤噪声并提升生成质量。**

- **链接: [http://arxiv.org/pdf/2509.21359v1](http://arxiv.org/pdf/2509.21359v1)**

> **作者:** Jiale Deng; Yanyan Shen; Ziyuan Pei; Youmin Chen; Linpeng Huang
>
> **摘要:** Retrieval-Augmented Generation (RAG) addresses large language model (LLM) hallucinations by grounding responses in external knowledge, but its effectiveness is compromised by poor-quality retrieved contexts containing irrelevant or noisy information. While existing approaches attempt to improve performance through context selection based on predefined context quality assessment metrics, they show limited gains over standard RAG. We attribute this limitation to their failure in holistically utilizing available information (query, context list, and generator) for comprehensive quality assessment. Inspired by recent advances in data selection, we reconceptualize context quality assessment as an inference-time data valuation problem and introduce the Contextual Influence Value (CI value). This novel metric quantifies context quality by measuring the performance degradation when removing each context from the list, effectively integrating query-aware relevance, list-aware uniqueness, and generator-aware alignment. Moreover, CI value eliminates complex selection hyperparameter tuning by simply retaining contexts with positive CI values. To address practical challenges of label dependency and computational overhead, we develop a parameterized surrogate model for CI value prediction during inference. The model employs a hierarchical architecture that captures both local query-context relevance and global inter-context interactions, trained through oracle CI value supervision and end-to-end generator feedback. Extensive experiments across 8 NLP tasks and multiple LLMs demonstrate that our context selection method significantly outperforms state-of-the-art baselines, effectively filtering poor-quality contexts while preserving critical information. Code is available at https://github.com/SJTU-DMTai/RAG-CSM.
>
---
#### [new 102] ResT: Reshaping Token-Level Policy Gradients for Tool-Use Large Language Models
- **分类: cs.CL**

- **简介: 该论文针对工具使用大语言模型的强化学习训练问题，提出ResT方法。通过熵感知的token级梯度重塑，降低策略方差，提升训练效率和稳定性，在单轮和多轮任务中均取得SOTA效果。**

- **链接: [http://arxiv.org/pdf/2509.21826v1](http://arxiv.org/pdf/2509.21826v1)**

> **作者:** Zihan Lin; Xiaohan Wang; Jie Cao; Jiajun Chai; Guojun Yin; Wei Lin; Ran He
>
> **摘要:** Large language models (LLMs) transcend passive generation and act as goal-directed agents by invoking external tools. Reinforcement learning (RL) offers a principled framework for optimizing these emergent tool-use policies, yet the prevailing paradigm relies exclusively on sparse outcome rewards and lacks consideration of the particularity of tool-use tasks, inflating policy-gradient variance and resulting in inefficient training. To better understand and address these challenges, we first establish a theoretical link between policy entropy and training stability of tool-use tasks, which reveals that structured, low-entropy tokens are primary determinants of rewards. Motivated by this insight, we propose \textbf{Res}haped \textbf{T}oken-level policy gradients (\textbf{ResT}) for tool-use tasks. ResT reshapes the policy gradient through entropy-informed token reweighting, progressively upweighting reasoning tokens as training proceeds. This entropy-aware scheme enables a smooth shift from structural correctness to semantic reasoning and stabilizes convergence in multi-turn tool-use tasks. Evaluation on BFCL and API-Bank shows that ResT achieves state-of-the-art results, outperforming prior methods by up to $8.76\%$. When fine-tuned on a 4B base LLM, ResT further surpasses GPT-4o by $4.11\%$ on single-turn tasks and $1.50\%$ on multi-turn base tasks.
>
---
#### [new 103] Evaluating the Limits of Large Language Models in Multilingual Legal Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文评估了大语言模型（LLM）在多语言法律推理中的能力与局限性，重点解决其在法律任务中的性能不足问题。研究对比了LLaMA和Gemini的表现，构建了开源评测框架，并分析了语言相似性、对抗鲁棒性等因素的影响。**

- **链接: [http://arxiv.org/pdf/2509.22472v1](http://arxiv.org/pdf/2509.22472v1)**

> **作者:** Antreas Ioannou; Andreas Shiamishis; Nora Hollenstein; Nezihe Merve Gürel
>
> **备注:** 39 pages, 36 figures. Code and evaluation pipeline available at https://github.com/RobustML-Lab/Legal-Multilingual-Evaluation-of-LLMs
>
> **摘要:** In an era dominated by Large Language Models (LLMs), understanding their capabilities and limitations, especially in high-stakes fields like law, is crucial. While LLMs such as Meta's LLaMA, OpenAI's ChatGPT, Google's Gemini, DeepSeek, and other emerging models are increasingly integrated into legal workflows, their performance in multilingual, jurisdictionally diverse, and adversarial contexts remains insufficiently explored. This work evaluates LLaMA and Gemini on multilingual legal and non-legal benchmarks, and assesses their adversarial robustness in legal tasks through character and word-level perturbations. We use an LLM-as-a-Judge approach for human-aligned evaluation. We moreover present an open-source, modular evaluation pipeline designed to support multilingual, task-diverse benchmarking of any combination of LLMs and datasets, with a particular focus on legal tasks, including classification, summarization, open questions, and general reasoning. Our findings confirm that legal tasks pose significant challenges for LLMs with accuracies often below 50% on legal reasoning benchmarks such as LEXam, compared to over 70% on general-purpose tasks like XNLI. In addition, while English generally yields more stable results, it does not always lead to higher accuracy. Prompt sensitivity and adversarial vulnerability is also shown to persist across languages. Finally, a correlation is found between the performance of a language and its syntactic similarity to English. We also observe that LLaMA is weaker than Gemini, with the latter showing an average advantage of about 24 percentage points across the same task. Despite improvements in newer LLMs, challenges remain in deploying them reliably for critical, multilingual legal applications.
>
---
#### [new 104] S2J: Bridging the Gap Between Solving and Judging Ability in Generative Reward Models
- **分类: cs.CL**

- **简介: 该论文聚焦生成式奖励模型（GRM）的评估任务，旨在解决“解题能力强但判断能力弱”的solve-to-judge gap问题。提出S2J方法，通过联合优化解题与判断能力，有效缩小了这一差距，提升了GRM的判断性能，并在小数据集上达到SOTA效果。**

- **链接: [http://arxiv.org/pdf/2509.22099v1](http://arxiv.org/pdf/2509.22099v1)**

> **作者:** Shaoning Sun; Jiachen Yu; Zongqi Wang; Xuewei Yang; Tianle Gu; Yujiu Yang
>
> **摘要:** With the rapid development of large language models (LLMs), generative reward models (GRMs) have been widely adopted for reward modeling and evaluation. Previous studies have primarily focused on training specialized GRMs by optimizing them on preference datasets with the judgment correctness as supervision. While it's widely accepted that GRMs with stronger problem-solving capabilities typically exhibit superior judgment abilities, we first identify a significant solve-to-judge gap when examining individual queries. Specifically, the solve-to-judge gap refers to the phenomenon where GRMs struggle to make correct judgments on some queries (14%-37%), despite being fully capable of solving them. In this paper, we propose the Solve-to-Judge (S2J) approach to address this problem. Specifically, S2J simultaneously leverages both the solving and judging capabilities on a single GRM's output for supervision, explicitly linking the GRM's problem-solving and evaluation abilities during model optimization, thereby narrowing the gap. Our comprehensive experiments demonstrate that S2J effectively reduces the solve-to-judge gap by 16.2%, thereby enhancing the model's judgment performance by 5.8%. Notably, S2J achieves state-of-the-art (SOTA) performance among GRMs built on the same base model while utilizing a significantly smaller training dataset. Moreover, S2J accomplishes this through self-evolution without relying on more powerful external models for distillation.
>
---
#### [new 105] Taxonomy of Comprehensive Safety for Clinical Agents
- **分类: cs.CL**

- **简介: 该论文提出TACOS，一个面向临床聊天机器人的21类安全分类体系，旨在解决现有方法在处理复杂临床场景中的不足。通过整合安全过滤与工具选择，提升临床智能体的安全性与适用性。**

- **链接: [http://arxiv.org/pdf/2509.22041v1](http://arxiv.org/pdf/2509.22041v1)**

> **作者:** Jean Seo; Hyunkyung Lee; Gibaeg Kim; Wooseok Han; Jaehyo Yoo; Seungseop Lim; Kihun Shin; Eunho Yang
>
> **备注:** EMNLP 2025 Industry
>
> **摘要:** Safety is a paramount concern in clinical chatbot applications, where inaccurate or harmful responses can lead to serious consequences. Existing methods--such as guardrails and tool calling--often fall short in addressing the nuanced demands of the clinical domain. In this paper, we introduce TACOS (TAxonomy of COmprehensive Safety for Clinical Agents), a fine-grained, 21-class taxonomy that integrates safety filtering and tool selection into a single user intent classification step. TACOS is a taxonomy that can cover a wide spectrum of clinical and non-clinical queries, explicitly modeling varying safety thresholds and external tool dependencies. To validate our framework, we curate a TACOS-annotated dataset and perform extensive experiments. Our results demonstrate the value of a new taxonomy specialized for clinical agent settings, and reveal useful insights about train data distribution and pretrained knowledge of base models.
>
---
#### [new 106] The InviTE Corpus: Annotating Invectives in Tudor English Texts for Computational Modeling
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理与历史研究交叉任务，旨在解决早期现代英语中宗教攻击性语言的识别问题。作者构建了InviTE语料库，并评估了不同模型在攻击性语言检测中的表现，强调历史数据预训练模型的优势。**

- **链接: [http://arxiv.org/pdf/2509.22345v1](http://arxiv.org/pdf/2509.22345v1)**

> **作者:** Sophie Spliethoff; Sanne Hoeken; Silke Schwandt; Sina Zarrieß; Özge Alaçam
>
> **摘要:** In this paper, we aim at the application of Natural Language Processing (NLP) techniques to historical research endeavors, particularly addressing the study of religious invectives in the context of the Protestant Reformation in Tudor England. We outline a workflow spanning from raw data, through pre-processing and data selection, to an iterative annotation process. As a result, we introduce the InviTE corpus -- a corpus of almost 2000 Early Modern English (EModE) sentences, which are enriched with expert annotations regarding invective language throughout 16th-century England. Subsequently, we assess and compare the performance of fine-tuned BERT-based models and zero-shot prompted instruction-tuned large language models (LLMs), which highlights the superiority of models pre-trained on historical data and fine-tuned to invective detection.
>
---
#### [new 107] Exploratory Semantic Reliability Analysis of Wind Turbine Maintenance Logs using Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理与可靠性分析交叉任务，旨在解决风力涡轮机维护日志中非结构化文本难以用于定量分析的问题。作者提出基于大语言模型的语义分析框架，实现故障模式识别、因果链推断等深度分析，提升风电运维智能化水平。**

- **链接: [http://arxiv.org/pdf/2509.22366v1](http://arxiv.org/pdf/2509.22366v1)**

> **作者:** Max Malyi; Jonathan Shek; Andre Biscaya
>
> **摘要:** A wealth of operational intelligence is locked within the unstructured free-text of wind turbine maintenance logs, a resource largely inaccessible to traditional quantitative reliability analysis. While machine learning has been applied to this data, existing approaches typically stop at classification, categorising text into predefined labels. This paper addresses the gap in leveraging modern large language models (LLMs) for more complex reasoning tasks. We introduce an exploratory framework that uses LLMs to move beyond classification and perform deep semantic analysis. We apply this framework to a large industrial dataset to execute four analytical workflows: failure mode identification, causal chain inference, comparative site analysis, and data quality auditing. The results demonstrate that LLMs can function as powerful "reliability co-pilots," moving beyond labelling to synthesise textual information and generate actionable, expert-level hypotheses. This work contributes a novel and reproducible methodology for using LLMs as a reasoning tool, offering a new pathway to enhance operational intelligence in the wind energy sector by unlocking insights previously obscured in unstructured data.
>
---
#### [new 108] Transformers Can Learn Connectivity in Some Graphs but Not Others
- **分类: cs.CL; cs.AI; cs.LG; cs.LO**

- **简介: 该论文研究了Transformer模型在推理图连通性（如传递关系）任务中的能力。作者通过训练不同规模的模型，发现Transformer能有效学习“网格状”图的连通性，但对包含多个不连通组件的图表现较差。**

- **链接: [http://arxiv.org/pdf/2509.22343v1](http://arxiv.org/pdf/2509.22343v1)**

> **作者:** Amit Roy; Abulhair Saparov
>
> **备注:** Under Review
>
> **摘要:** Reasoning capability is essential to ensure the factual correctness of the responses of transformer-based Large Language Models (LLMs), and robust reasoning about transitive relations is instrumental in many settings, such as causal inference. Hence, it is essential to investigate the capability of transformers in the task of inferring transitive relations (e.g., knowing A causes B and B causes C, then A causes C). The task of inferring transitive relations is equivalent to the task of connectivity in directed graphs (e.g., knowing there is a path from A to B, and there is a path from B to C, then there is a path from A to C). Past research focused on whether transformers can learn to infer transitivity from in-context examples provided in the input prompt. However, transformers' capability to infer transitive relations from training examples and how scaling affects the ability is unexplored. In this study, we seek to answer this question by generating directed graphs to train transformer models of varying sizes and evaluate their ability to infer transitive relations for various graph sizes. Our findings suggest that transformers are capable of learning connectivity on "grid-like'' directed graphs where each node can be embedded in a low-dimensional subspace, and connectivity is easily inferable from the embeddings of the nodes. We find that the dimensionality of the underlying grid graph is a strong predictor of transformers' ability to learn the connectivity task, where higher-dimensional grid graphs pose a greater challenge than low-dimensional grid graphs. In addition, we observe that increasing the model scale leads to increasingly better generalization to infer connectivity over grid graphs. However, if the graph is not a grid graph and contains many disconnected components, transformers struggle to learn the connectivity task, especially when the number of components is large.
>
---
#### [new 109] Think Right, Not More: Test-Time Scaling for Numerical Claim Verification
- **分类: cs.CL**

- **简介: 该论文聚焦于**数值性声明的事实核查任务**，旨在解决大语言模型在组合推理与数值推理中的不足，特别是“推理漂移”问题。作者提出一种**测试时计算扩展（TTS）方法**，结合自适应机制和验证模型（VERIFIERFC），以提高事实核查的准确性和计算效率。**

- **链接: [http://arxiv.org/pdf/2509.22101v1](http://arxiv.org/pdf/2509.22101v1)**

> **作者:** Primakov Chungkham; V Venktesh; Vinay Setty; Avishek Anand
>
> **备注:** Accepted to EMNLP 2025, 19 pages
>
> **摘要:** Fact-checking real-world claims, particularly numerical claims, is inherently complex that require multistep reasoning and numerical reasoning for verifying diverse aspects of the claim. Although large language models (LLMs) including reasoning models have made tremendous advances, they still fall short on fact-checking real-world claims that require a combination of compositional and numerical reasoning. They are unable to understand nuance of numerical aspects, and are also susceptible to the reasoning drift issue, where the model is unable to contextualize diverse information resulting in misinterpretation and backtracking of reasoning process. In this work, we systematically explore scaling test-time compute (TTS) for LLMs on the task of fact-checking complex numerical claims, which entails eliciting multiple reasoning paths from an LLM. We train a verifier model (VERIFIERFC) to navigate this space of possible reasoning paths and select one that could lead to the correct verdict. We observe that TTS helps mitigate the reasoning drift issue, leading to significant performance gains for fact-checking numerical claims. To improve compute efficiency in TTS, we introduce an adaptive mechanism that performs TTS selectively based on the perceived complexity of the claim. This approach achieves 1.8x higher efficiency than standard TTS, while delivering a notable 18.8% performance improvement over single-shot claim verification methods. Our code and data can be found at https://github.com/VenkteshV/VerifierFC
>
---
#### [new 110] DeHate: A Stable Diffusion-based Multimodal Approach to Mitigate Hate Speech in Images
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出DeHate，一种基于Stable Diffusion的多模态方法，用于识别并模糊图像中的仇恨内容。通过水印增强扩散技术和DAAM模块生成仇恨注意力图，解决在线图像中仇恨言论的问题，并发布相关数据集和共享任务。**

- **链接: [http://arxiv.org/pdf/2509.21787v1](http://arxiv.org/pdf/2509.21787v1)**

> **作者:** Dwip Dalal; Gautam Vashishtha; Anku Ranui; Aishwarya Reganti; Parth Patwa; Mohd Sarique; Chandan Gupta; Keshav Nath; Viswanatha Reddy; Vinija Jain; Aman Chadha; Amitava Das; Amit Sheth; Asif Ekbal
>
> **备注:** Defactify 3 workshop at AAAI 2024
>
> **摘要:** The rise in harmful online content not only distorts public discourse but also poses significant challenges to maintaining a healthy digital environment. In response to this, we introduce a multimodal dataset uniquely crafted for identifying hate in digital content. Central to our methodology is the innovative application of watermarked, stability-enhanced, stable diffusion techniques combined with the Digital Attention Analysis Module (DAAM). This combination is instrumental in pinpointing the hateful elements within images, thereby generating detailed hate attention maps, which are used to blur these regions from the image, thereby removing the hateful sections of the image. We release this data set as a part of the dehate shared task. This paper also describes the details of the shared task. Furthermore, we present DeHater, a vision-language model designed for multimodal dehatification tasks. Our approach sets a new standard in AI-driven image hate detection given textual prompts, contributing to the development of more ethical AI applications in social media.
>
---
#### [new 111] Library Hallucinations in LLMs: Risk Analysis Grounded in Developer Queries
- **分类: cs.SE; cs.CL**

- **简介: 该论文研究LLMs在代码生成中的“库幻觉”问题，分析用户提示变化对错误库名和成员生成的影响。通过评估六种模型，揭示了提示细微差异导致高幻觉率的风险，并探讨缓解方法。任务属于风险分析与缓解策略研究。**

- **链接: [http://arxiv.org/pdf/2509.22202v1](http://arxiv.org/pdf/2509.22202v1)**

> **作者:** Lukas Twist; Jie M. Zhang; Mark Harman; Helen Yannakoudakis
>
> **备注:** 23 pages, 5 tables
>
> **摘要:** Large language models (LLMs) are increasingly used to generate code, yet they continue to hallucinate, often inventing non-existent libraries. Such library hallucinations are not just benign errors: they can mislead developers, break builds, and expose systems to supply chain threats such as slopsquatting. Despite increasing awareness of these risks, little is known about how real-world prompt variations affect hallucination rates. Therefore, we present the first systematic study of how user-level prompt variations impact library hallucinations in LLM-generated code. We evaluate six diverse LLMs across two hallucination types: library name hallucinations (invalid imports) and library member hallucinations (invalid calls from valid libraries). We investigate how realistic user language extracted from developer forums and how user errors of varying degrees (one- or multi-character misspellings and completely fake names/members) affect LLM hallucination rates. Our findings reveal systemic vulnerabilities: one-character misspellings in library names trigger hallucinations in up to 26% of tasks, fake library names are accepted in up to 99% of tasks, and time-related prompts lead to hallucinations in up to 84% of tasks. Prompt engineering shows promise for mitigating hallucinations, but remains inconsistent and LLM-dependent. Our results underscore the fragility of LLMs to natural prompt variation and highlight the urgent need for safeguards against library-related hallucinations and their potential exploitation.
>
---
#### [new 112] Uncertainty-Aware Knowledge Tracing Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于知识追踪（KT）任务，旨在解决模型预测错误时无法识别学生误选问题。工作提出通过捕捉预测不确定性来增强KT模型，表明不确定性可反映预测错误，并可用于教育资源有限的教育平台中辅助理解学生能力。**

- **链接: [http://arxiv.org/pdf/2509.21514v1](http://arxiv.org/pdf/2509.21514v1)**

> **作者:** Joshua Mitton; Prarthana Bhattacharyya; Ralph Abboud; Simon Woodhead
>
> **备注:** 10 pages, 7 figures. Joshua Mitton and Prarthana Bhattacharyya contributed equally to this paper
>
> **摘要:** The main focus of research on Knowledge Tracing (KT) models is on model developments with the aim of improving predictive accuracy. Most of these models make the most incorrect predictions when students choose a distractor, leading to student errors going undetected. We present an approach to add new capabilities to KT models by capturing predictive uncertainty and demonstrate that a larger predictive uncertainty aligns with model incorrect predictions. We show that uncertainty in KT models is informative and that this signal would be pedagogically useful for application in an educational learning platform that can be used in a limited resource setting where understanding student ability is necessary.
>
---
#### [new 113] EPO: Entropy-regularized Policy Optimization for LLM Agents Reinforcement Learning
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究多轮稀疏奖励环境下大语言模型（LLM）智能体的强化学习问题，提出EPO框架，通过熵正则化、熵平滑和自适应权重机制，解决探索-利用失衡导致的策略崩溃问题，提升任务完成性能。**

- **链接: [http://arxiv.org/pdf/2509.22576v1](http://arxiv.org/pdf/2509.22576v1)**

> **作者:** Xu Wujiang; Wentian Zhao; Zhenting Wang; Li Yu-Jhe; Jin Can; Jin Mingyu; Mei Kai; Wan Kun; Metaxas Dimitris
>
> **摘要:** Training LLM agents in multi-turn environments with sparse rewards, where completing a single task requires 30+ turns of interaction within an episode, presents a fundamental challenge for reinforcement learning. We identify a critical failure mode unique to this setting: the exploration-exploitation cascade failure. This cascade begins with early-stage policy premature convergence, where sparse feedback causes agents to commit to flawed, low-entropy strategies. Subsequently, agents enter late-stage policy collapse, where conventional entropy regularization becomes counterproductive, promoting chaotic exploration that destabilizes training. We propose Entropy-regularized Policy Optimization (EPO), a general framework that breaks this failure cycle through three synergistic mechanisms: (1) adopting entropy regularization in multi-turn settings to enhance exploration, (2) an entropy smoothing regularizer that bounds policy entropy within historical averages to prevent abrupt fluctuations, and (3) adaptive phase-based weighting that balances exploration and exploitation across training. Our analysis justifies that EPO guarantees monotonically decreasing entropy variance while maintaining convergence. EPO achieves up to 152% performance improvement on ScienceWorld and up to 19.8% on ALFWorld. Our work demonstrates that multi-turn sparse-reward settings require fundamentally different entropy control than traditional RL, with broad implications for LLM agent training.
>
---
#### [new 114] VideoJudge: Bootstrapping Enables Scalable Supervision of MLLM-as-a-Judge for Video Understanding
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出VideoJudge，一种用于视频理解模型评估的MLLM裁判。针对现有评估方法不足，通过生成器与评估器的交互训练，实现更符合人类判断的自动评估，在多个基准上优于大模型基线。**

- **链接: [http://arxiv.org/pdf/2509.21451v1](http://arxiv.org/pdf/2509.21451v1)**

> **作者:** Abdul Waheed; Zhen Wu; Dareen Alharthi; Seungone Kim; Bhiksha Raj
>
> **备注:** Work in progress
>
> **摘要:** Precisely evaluating video understanding models remains challenging: commonly used metrics such as BLEU, ROUGE, and BERTScore fail to capture the fineness of human judgment, while obtaining such judgments through manual evaluation is costly. Recent work has explored using large language models (LLMs) or multimodal LLMs (MLLMs) as evaluators, but their extension to video understanding remains relatively unexplored. In this work, we introduce VideoJudge, a 3B and 7B-sized MLLM judge specialized to evaluate outputs from video understanding models (\textit{i.e.}, text responses conditioned on videos). To train VideoJudge, our recipe builds on the interplay between a generator and an evaluator: the generator is prompted to produce responses conditioned on a target rating, and responses not matching the evaluator's rating are discarded. Across three out of four meta-evaluation benchmarks, VideoJudge-7B outperforms larger MLLM judge baselines such as Qwen2.5-VL (32B and 72B). Notably, we find that LLM judges (Qwen3) models perform worse than MLLM judges (Qwen2.5-VL) and long chain-of-thought reasoning does not improve performance, indicating that providing video inputs is crucial for evaluation of video understanding tasks.
>
---
#### [new 115] IIET: Efficient Numerical Transformer via Implicit Iterative Euler Method
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出IIET，一种基于隐式欧拉法的高效数值Transformer，旨在解决高阶方法计算开销大、效率低的问题。通过迭代简化和模型压缩技术，提升性能并降低推理成本，在NLP任务中取得显著效果。**

- **链接: [http://arxiv.org/pdf/2509.22463v1](http://arxiv.org/pdf/2509.22463v1)**

> **作者:** Xinyu Liu; Bei Li; Jiahao Liu; Junhao Ruan; Kechen Jiao; Hongyin Tang; Jingang Wang; Xiao Tong; Jingbo Zhu
>
> **摘要:** High-order numerical methods enhance Transformer performance in tasks like NLP and CV, but introduce a performance-efficiency trade-off due to increased computational overhead. Our analysis reveals that conventional efficiency techniques, such as distillation, can be detrimental to the performance of these models, exemplified by PCformer. To explore more optimizable ODE-based Transformer architectures, we propose the \textbf{I}terative \textbf{I}mplicit \textbf{E}uler \textbf{T}ransformer \textbf{(IIET)}, which simplifies high-order methods using an iterative implicit Euler approach. This simplification not only leads to superior performance but also facilitates model compression compared to PCformer. To enhance inference efficiency, we introduce \textbf{I}teration \textbf{I}nfluence-\textbf{A}ware \textbf{D}istillation \textbf{(IIAD)}. Through a flexible threshold, IIAD allows users to effectively balance the performance-efficiency trade-off. On lm-evaluation-harness, IIET boosts average accuracy by 2.65\% over vanilla Transformers and 0.8\% over PCformer. Its efficient variant, E-IIET, significantly cuts inference overhead by 55\% while retaining 99.4\% of the original task accuracy. Moreover, the most efficient IIET variant achieves an average performance gain exceeding 1.6\% over vanilla Transformer with comparable speed.
>
---
#### [new 116] IA2: Alignment with ICL Activations Improves Supervised Fine-Tuning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究如何利用上下文学习（ICL）提升监督微调（SFT）效果。提出IA2方法，通过对齐ICL的激活模式进行自蒸馏，从而提高SFT模型的准确性和校准能力，在多个基准上验证了有效性。**

- **链接: [http://arxiv.org/pdf/2509.22621v1](http://arxiv.org/pdf/2509.22621v1)**

> **作者:** Aayush Mishra; Daniel Khashabi; Anqi Liu
>
> **摘要:** Supervised Fine-Tuning (SFT) is used to specialize model behavior by training weights to produce intended target responses for queries. In contrast, In-Context Learning (ICL) adapts models during inference with instructions or demonstrations in the prompt. ICL can offer better generalizability and more calibrated responses compared to SFT in data scarce settings, at the cost of more inference compute. In this work, we ask the question: Can ICL's internal computations be used to improve the qualities of SFT? We first show that ICL and SFT produce distinct activation patterns, indicating that the two methods achieve adaptation through different functional mechanisms. Motivated by this observation and to use ICL's rich functionality, we introduce ICL Activation Alignment (IA2), a self-distillation technique which aims to replicate ICL's activation patterns in SFT models and incentivizes ICL-like internal reasoning. Performing IA2 as a priming step before SFT significantly improves the accuracy and calibration of model outputs, as shown by our extensive empirical results on 12 popular benchmarks and 2 model families. This finding is not only practically useful, but also offers a conceptual window into the inner mechanics of model adaptation.
>
---
#### [new 117] Towards Efficient Online Exploration for Reinforcement Learning with Human Feedback
- **分类: stat.ML; cs.AI; cs.CL; cs.LG; math.ST; stat.TH**

- **简介: 该论文研究在线强化学习与人类反馈（RLHF）中的高效探索问题，旨在通过优化策略和奖励模型的数据收集方式，解决现有方法在信息获取效率低、导致线性遗憾的问题。提出了一种新的探索方案，降低了策略改进中奖励差异的不确定性，并在多臂老虎机模型下证明了多项式级别的遗憾界。**

- **链接: [http://arxiv.org/pdf/2509.22633v1](http://arxiv.org/pdf/2509.22633v1)**

> **作者:** Gen Li; Yuling Yan
>
> **摘要:** Reinforcement learning with human feedback (RLHF), which learns a reward model from human preference data and then optimizes a policy to favor preferred responses, has emerged as a central paradigm for aligning large language models (LLMs) with human preferences. In this paper, we investigate exploration principles for online RLHF, where one seeks to adaptively collect new preference data to refine both the reward model and the policy in a data-efficient manner. By examining existing optimism-based exploration algorithms, we identify a drawback in their sampling protocol: they tend to gather comparisons that fail to reduce the most informative uncertainties in reward differences, and we prove lower bounds showing that such methods can incur linear regret over exponentially long horizons. Motivated by this insight, we propose a new exploration scheme that directs preference queries toward reducing uncertainty in reward differences most relevant to policy improvement. Under a multi-armed bandit model of RLHF, we establish regret bounds of order $T^{(\beta+1)/(\beta+2)}$, where $\beta>0$ is a hyperparameter that balances reward maximization against mitigating distribution shift. To our knowledge, this is the first online RLHF algorithm with regret scaling polynomially in all model parameters.
>
---
#### [new 118] Can Synthetic Query Rewrites Capture User Intent Better than Humans in Retrieval-Augmented Generation?
- **分类: cs.IR; cs.CL**

- **简介: 该论文研究查询重写任务，旨在解决用户意图与RAG系统响应间的偏差问题。提出SynRewrite模型，利用合成数据训练，通过GPT-4o生成高质量重写，并结合DPO优化，实验证明其在检索和生成上优于人工重写。**

- **链接: [http://arxiv.org/pdf/2509.22325v1](http://arxiv.org/pdf/2509.22325v1)**

> **作者:** JiaYing Zheng; HaiNan Zhang; Liang Pang; YongXin Tong; ZhiMing Zheng
>
> **备注:** 10 pages, 6 figures
>
> **摘要:** Multi-turn RAG systems often face queries with colloquial omissions and ambiguous references, posing significant challenges for effective retrieval and generation. Traditional query rewriting relies on human annotators to clarify queries, but due to limitations in annotators' expressive ability and depth of understanding, manually rewritten queries often diverge from those needed in real-world RAG systems, resulting in a gap between user intent and system response. We observe that high-quality synthetic queries can better bridge this gap, achieving superior performance in both retrieval and generation compared to human rewrites. This raises an interesting question: Can rewriting models trained on synthetic queries better capture user intent than human annotators? In this paper, we propose SynRewrite, a synthetic data-driven query rewriting model to generate high-quality synthetic rewrites more aligned with user intent. To construct training data, we prompt GPT-4o with dialogue history, current queries, positive documents, and answers to synthesize high-quality rewrites. A Flan-T5 model is then finetuned on this dataset to map dialogue history and queries to synthetic rewrites. Finally, we further enhance the rewriter using the generator's feedback through the DPO algorithm to boost end-task performance. Experiments on TopiOCQA and QRECC datasets show that SynRewrite consistently outperforms human rewrites in both retrieval and generation tasks. Our results demonstrate that synthetic rewrites can serve as a scalable and effective alternative to human annotations.
>
---
#### [new 119] LABELING COPILOT: A Deep Research Agent for Automated Data Curation in Computer Vision
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出Labeling Copilot，一个用于计算机视觉领域自动数据整理的深度研究代理。针对高质量数据集构建中质量、多样性和成本的平衡问题，设计了三大核心模块：校准发现、可控合成和共识标注，实现了高效、可扩展的数据筛选与标注流程。**

- **链接: [http://arxiv.org/pdf/2509.22631v1](http://arxiv.org/pdf/2509.22631v1)**

> **作者:** Debargha Ganguly; Sumit Kumar; Ishwar Balappanawar; Weicong Chen; Shashank Kambhatla; Srinivasan Iyengar; Shivkumar Kalyanaraman; Ponnurangam Kumaraguru; Vipin Chaudhary
>
> **摘要:** Curating high-quality, domain-specific datasets is a major bottleneck for deploying robust vision systems, requiring complex trade-offs between data quality, diversity, and cost when researching vast, unlabeled data lakes. We introduce Labeling Copilot, the first data curation deep research agent for computer vision. A central orchestrator agent, powered by a large multimodal language model, uses multi-step reasoning to execute specialized tools across three core capabilities: (1) Calibrated Discovery sources relevant, in-distribution data from large repositories; (2) Controllable Synthesis generates novel data for rare scenarios with robust filtering; and (3) Consensus Annotation produces accurate labels by orchestrating multiple foundation models via a novel consensus mechanism incorporating non-maximum suppression and voting. Our large-scale validation proves the effectiveness of Labeling Copilot's components. The Consensus Annotation module excels at object discovery: on the dense COCO dataset, it averages 14.2 candidate proposals per image-nearly double the 7.4 ground-truth objects-achieving a final annotation mAP of 37.1%. On the web-scale Open Images dataset, it navigated extreme class imbalance to discover 903 new bounding box categories, expanding its capability to over 1500 total. Concurrently, our Calibrated Discovery tool, tested at a 10-million sample scale, features an active learning strategy that is up to 40x more computationally efficient than alternatives with equivalent sample efficiency. These experiments validate that an agentic workflow with optimized, scalable tools provides a robust foundation for curating industrial-scale datasets.
>
---
#### [new 120] The Thinking Spectrum: An Emperical Study of Tunable Reasoning in LLMs through Model Merging
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究了通过模型合并技术，在大语言模型中实现可调推理能力的方法。任务是探索如何在不训练的情况下，构建兼顾推理精度与计算效率的模型谱系。工作包括大规模实证分析多种合并方法，揭示其在准确率与效率间的可控权衡，并发现帕累托改进实例。**

- **链接: [http://arxiv.org/pdf/2509.22034v1](http://arxiv.org/pdf/2509.22034v1)**

> **作者:** Xiaochong Lan; Yu Zheng; Shiteng Cao; Yong Li
>
> **摘要:** The growing demand for large language models (LLMs) with tunable reasoning capabilities in many real-world applications highlights a critical need for methods that can efficiently produce a spectrum of models balancing reasoning depth and computational cost. Model merging has emerged as a promising, training-free technique to address this challenge by arithmetically combining the weights of a general-purpose model with a specialized reasoning model. While various merging techniques exist, their potential to create a spectrum of models with fine-grained control over reasoning abilities remains largely unexplored. This work presents a large-scale empirical study evaluating a range of model merging techniques across multiple reasoning benchmarks. We systematically vary merging strengths to construct accuracy-efficiency curves, providing the first comprehensive view of the tunable performance landscape. Our findings reveal that model merging offers an effective and controllable method for calibrating the trade-off between reasoning accuracy and token efficiency, even when parent models have highly divergent weight spaces. Crucially, we identify instances of Pareto Improvement, where a merged model achieves both higher accuracy and lower token consumption than one of its parents. Our study provides the first comprehensive analysis of this tunable space, offering practical guidelines for creating LLMs with specific reasoning profiles to meet diverse application demands.
>
---
#### [new 121] From Bias to Balance: Exploring and Mitigating Spatial Bias in LVLMs
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究了大视觉-语言模型（LVLMs）中的空间偏差问题，发现语言模型组件的位置嵌入设计不平衡导致视觉信息整合不均。提出了一种简单有效的平衡位置分配机制BaPA，无需微调即可提升模型的空间鲁棒性和多模态性能。**

- **链接: [http://arxiv.org/pdf/2509.21984v1](http://arxiv.org/pdf/2509.21984v1)**

> **作者:** Yingjie Zhu; Xuefeng Bai; Kehai Chen; Yang Xiang; Weili Guan; Jun Yu; Min Zhang
>
> **摘要:** Large Vision-Language Models (LVLMs) have achieved remarkable success across a wide range of multimodal tasks, yet their robustness to spatial variations remains insufficiently understood. In this work, we present a systematic study of the spatial bias of LVLMs, focusing on how models respond when identical key visual information is placed at different locations within an image. Through a carefully designed probing dataset, we demonstrate that current LVLMs often produce inconsistent outputs under such spatial shifts, revealing a fundamental limitation in their spatial-semantic understanding. Further analysis shows that this phenomenon originates not from the vision encoder, which reliably perceives and interprets visual content across positions, but from the unbalanced design of position embeddings in the language model component. In particular, the widely adopted position embedding strategies, such as RoPE, introduce imbalance during cross-modal interaction, leading image tokens at different positions to exert unequal influence on semantic understanding. To mitigate this issue, we introduce Balanced Position Assignment (BaPA), a simple yet effective mechanism that assigns identical position embeddings to all image tokens, promoting a more balanced integration of visual information. Extensive experiments show that BaPA enhances the spatial robustness of LVLMs without retraining and further boosts their performance across diverse multimodal benchmarks when combined with lightweight fine-tuning. Further analysis of information flow reveals that BaPA yields balanced attention, enabling more holistic visual understanding.
>
---
#### [new 122] ERGO: Efficient High-Resolution Visual Understanding for Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出ERGO，针对视觉-语言模型处理高分辨率图像计算成本高的问题，设计了一种“粗到细”的推理流程，通过多模态上下文指导观察区域选择，在保证精度的同时显著提升效率。**

- **链接: [http://arxiv.org/pdf/2509.21991v1](http://arxiv.org/pdf/2509.21991v1)**

> **作者:** Jewon Lee; Wooksu Shin; Seungmin Yang; Ki-Ung Song; DongUk Lim; Jaeyeon Kim; Tae-Ho Kim; Bo-Kyeong Kim
>
> **摘要:** Efficient processing of high-resolution images is crucial for real-world vision-language applications. However, existing Large Vision-Language Models (LVLMs) incur substantial computational overhead due to the large number of vision tokens. With the advent of "thinking with images" models, reasoning now extends beyond text to the visual domain. This capability motivates our two-stage "coarse-to-fine" reasoning pipeline: first, a downsampled image is analyzed to identify task-relevant regions; then, only these regions are cropped at full resolution and processed in a subsequent reasoning stage. This approach reduces computational cost while preserving fine-grained visual details where necessary. A major challenge lies in inferring which regions are truly relevant to a given query. Recent related methods often fail in the first stage after input-image downsampling, due to perception-driven reasoning, where clear visual information is required for effective reasoning. To address this issue, we propose ERGO (Efficient Reasoning & Guided Observation) that performs reasoning-driven perception-leveraging multimodal context to determine where to focus. Our model can account for perceptual uncertainty, expanding the cropped region to cover visually ambiguous areas for answering questions. To this end, we develop simple yet effective reward components in a reinforcement learning framework for coarse-to-fine perception. Across multiple datasets, our approach delivers higher accuracy than the original model and competitive methods, with greater efficiency. For instance, ERGO surpasses Qwen2.5-VL-7B on the V* benchmark by 4.7 points while using only 23% of the vision tokens, achieving a 3x inference speedup. The code and models can be found at: https://github.com/nota-github/ERGO.
>
---
#### [new 123] Are Hallucinations Bad Estimations?
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; stat.ML**

- **简介: 该论文研究生成模型中的幻觉问题，将其形式化为估计与合理原因的脱节。指出即使最优估计器也会产生幻觉，并通过理论和实验验证了这一现象，揭示了损失最小化与人类可接受输出之间的结构性错位。**

- **链接: [http://arxiv.org/pdf/2509.21473v1](http://arxiv.org/pdf/2509.21473v1)**

> **作者:** Hude Liu; Jerry Yao-Chieh Hu; Jennifer Yuntong Zhang; Zhao Song; Han Liu
>
> **备注:** Code is available at https://github.com/MAGICS-LAB/hallucination
>
> **摘要:** We formalize hallucinations in generative models as failures to link an estimate to any plausible cause. Under this interpretation, we show that even loss-minimizing optimal estimators still hallucinate. We confirm this with a general high probability lower bound on hallucinate rate for generic data distributions. This reframes hallucination as structural misalignment between loss minimization and human-acceptable outputs, and hence estimation errors induced by miscalibration. Experiments on coin aggregation, open-ended QA, and text-to-image support our theory.
>
---
#### [new 124] Compiling by Proving: Language-Agnostic Automatic Optimization from Formal Semantics
- **分类: cs.PL; cs.CL**

- **简介: 该论文提出“通过证明进行编译”的方法，利用验证证明生成优化执行规则，实现语言无关的自动优化。工作包括构建全路径可达性证明并集成到K框架中，验证了不同编译粒度下的性能提升。**

- **链接: [http://arxiv.org/pdf/2509.21793v1](http://arxiv.org/pdf/2509.21793v1)**

> **作者:** Jianhong Zhao; Everett Hildenbrandt; Juan Conejero; Yongwang Zhao
>
> **摘要:** Verification proofs encode complete program behavior, yet we discard them after checking correctness. We present compiling by proving, a paradigm that transforms these proofs into optimized execution rules. By constructing All-Path Reachability Proofs through symbolic execution and compiling their graph structure, we consolidate many semantic rewrites into single rules while preserving correctness by construction. We implement this as a language-agnostic extension to the K framework. Evaluation demonstrates performance improvements across different compilation scopes: opcode-level optimizations show consistent speedups, while whole-program compilation achieves orders of magnitude greater performance gains.
>
---
#### [new 125] UISim: An Interactive Image-Based UI Simulator for Dynamic Mobile Environments
- **分类: cs.CV; cs.AI; cs.CL; cs.HC; cs.LG**

- **简介: 该论文提出UISim，一种基于图像的移动UI模拟器，用于动态环境下的UI测试与AI代理训练。针对现有方法依赖物理设备或静态分析的问题，UISim通过两阶段方法预测并生成视觉一致的UI状态，提升UI开发效率和AI训练效果。**

- **链接: [http://arxiv.org/pdf/2509.21733v1](http://arxiv.org/pdf/2509.21733v1)**

> **作者:** Jiannan Xiang; Yun Zhu; Lei Shu; Maria Wang; Lijun Yu; Gabriel Barcik; James Lyon; Srinivas Sunkara; Jindong Chen
>
> **摘要:** Developing and testing user interfaces (UIs) and training AI agents to interact with them are challenging due to the dynamic and diverse nature of real-world mobile environments. Existing methods often rely on cumbersome physical devices or limited static analysis of screenshots, which hinders scalable testing and the development of intelligent UI agents. We introduce UISim, a novel image-based UI simulator that offers a dynamic and interactive platform for exploring mobile phone environments purely from screen images. Our system employs a two-stage method: given an initial phone screen image and a user action, it first predicts the abstract layout of the next UI state, then synthesizes a new, visually consistent image based on this predicted layout. This approach enables the realistic simulation of UI transitions. UISim provides immediate practical benefits for UI testing, rapid prototyping, and synthetic data generation. Furthermore, its interactive capabilities pave the way for advanced applications, such as UI navigation task planning for AI agents. Our experimental results show that UISim outperforms end-to-end UI generation baselines in generating realistic and coherent subsequent UI states, highlighting its fidelity and potential to streamline UI development and enhance AI agent training.
>
---
#### [new 126] SecureAgentBench: Benchmarking Secure Code Generation under Realistic Vulnerability Scenarios
- **分类: cs.SE; cs.AI; cs.CL; cs.CR**

- **简介: 该论文提出SecureAgentBench，一个用于评估代码生成安全性的基准，旨在解决现有工具忽略真实漏洞上下文的问题。通过105个任务，结合功能测试与漏洞检测，全面评估LLM在生成安全代码方面的能力，揭示当前代理在安全编码上的不足。**

- **链接: [http://arxiv.org/pdf/2509.22097v1](http://arxiv.org/pdf/2509.22097v1)**

> **作者:** Junkai Chen; Huihui Huang; Yunbo Lyu; Junwen An; Jieke Shi; Chengran Yang; Ting Zhang; Haoye Tian; Yikun Li; Zhenhao Li; Xin Zhou; Xing Hu; David Lo
>
> **摘要:** Large language model (LLM) powered code agents are rapidly transforming software engineering by automating tasks such as testing, debugging, and repairing, yet the security risks of their generated code have become a critical concern. Existing benchmarks have offered valuable insights but remain insufficient: they often overlook the genuine context in which vulnerabilities were introduced or adopt narrow evaluation protocols that fail to capture either functional correctness or newly introduced vulnerabilities. We therefore introduce SecureAgentBench, a benchmark of 105 coding tasks designed to rigorously evaluate code agents' capabilities in secure code generation. Each task includes (i) realistic task settings that require multi-file edits in large repositories, (ii) aligned contexts based on real-world open-source vulnerabilities with precisely identified introduction points, and (iii) comprehensive evaluation that combines functionality testing, vulnerability checking through proof-of-concept exploits, and detection of newly introduced vulnerabilities using static analysis. We evaluate three representative agents (SWE-agent, OpenHands, and Aider) with three state-of-the-art LLMs (Claude 3.7 Sonnet, GPT-4.1, and DeepSeek-V3.1). Results show that (i) current agents struggle to produce secure code, as even the best-performing one, SWE-agent supported by DeepSeek-V3.1, achieves merely 15.2% correct-and-secure solutions, (ii) some agents produce functionally correct code but still introduce vulnerabilities, including new ones not previously recorded, and (iii) adding explicit security instructions for agents does not significantly improve secure coding, underscoring the need for further research. These findings establish SecureAgentBench as a rigorous benchmark for secure code generation and a step toward more reliable software development with LLMs.
>
---
#### [new 127] SBFA: Single Sneaky Bit Flip Attack to Break Large Language Models
- **分类: cs.CR; cs.CL; cs.LG**

- **简介: 该论文提出SBFA，一种通过单比特翻转破坏大语言模型（LLM）性能的安全攻击方法。针对现有位翻转攻击不够隐蔽和灵活的问题，设计了ImpactScore敏感度指标和SKIP算法，实现高效、隐蔽的攻击，在多种模型和数据格式下验证其有效性。**

- **链接: [http://arxiv.org/pdf/2509.21843v1](http://arxiv.org/pdf/2509.21843v1)**

> **作者:** Jingkai Guo; Chaitali Chakrabarti; Deliang Fan
>
> **备注:** 10 pages, 4 figures, 5 tables, 2 equations. Topics: Bit-flip attacks, adversarial attacks, large language models (LLMs)
>
> **摘要:** Model integrity of Large language models (LLMs) has become a pressing security concern with their massive online deployment. Prior Bit-Flip Attacks (BFAs) -- a class of popular AI weight memory fault-injection techniques -- can severely compromise Deep Neural Networks (DNNs): as few as tens of bit flips can degrade accuracy toward random guessing. Recent studies extend BFAs to LLMs and reveal that, despite the intuition of better robustness from modularity and redundancy, only a handful of adversarial bit flips can also cause LLMs' catastrophic accuracy degradation. However, existing BFA methods typically focus on either integer or floating-point models separately, limiting attack flexibility. Moreover, in floating-point models, random bit flips often cause perturbed parameters to extreme values (e.g., flipping in exponent bit), making it not stealthy and leading to numerical runtime error (e.g., invalid tensor values (NaN/Inf)). In this work, for the first time, we propose SBFA (Sneaky Bit-Flip Attack), which collapses LLM performance with only one single bit flip while keeping perturbed values within benign layer-wise weight distribution. It is achieved through iterative searching and ranking through our defined parameter sensitivity metric, ImpactScore, which combines gradient sensitivity and perturbation range constrained by the benign layer-wise weight distribution. A novel lightweight SKIP searching algorithm is also proposed to greatly reduce searching complexity, which leads to successful SBFA searching taking only tens of minutes for SOTA LLMs. Across Qwen, LLaMA, and Gemma models, with only one single bit flip, SBFA successfully degrades accuracy to below random levels on MMLU and SST-2 in both BF16 and INT8 data formats. Remarkably, flipping a single bit out of billions of parameters reveals a severe security concern of SOTA LLM models.
>
---
#### [new 128] Vision-Language Alignment from Compressed Image Representations using 2D Gaussian Splatting
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文研究了基于2D高斯溅射（2DGS）的视觉-语言对齐方法，旨在解决传统RGB图像传输能耗高和序列长度过长的问题。论文提出了高效的2DGS表示与优化流程，并适配CLIP模型，实现高效压缩与语义对齐，验证了其在零样本任务中的可行性。**

- **链接: [http://arxiv.org/pdf/2509.22615v1](http://arxiv.org/pdf/2509.22615v1)**

> **作者:** Yasmine Omri; Connor Ding; Tsachy Weissman; Thierry Tambe
>
> **摘要:** Modern vision language pipelines are driven by RGB vision encoders trained on massive image text corpora. While these pipelines have enabled impressive zero shot capabilities and strong transfer across tasks, they still inherit two structural inefficiencies from the pixel domain: (i) transmitting dense RGB images from edge devices to the cloud is energy intensive and costly, and (ii) patch based tokenization explodes sequence length, stressing attention budgets and context limits. We explore 2D Gaussian Splatting (2DGS) as an alternative visual substrate for alignment: a compact, spatially adaptive representation that parameterizes images by a set of colored anisotropic Gaussians. We develop a scalable 2DGS pipeline with structured initialization, luminance aware pruning, and batched CUDA kernels, achieving over 90x faster fitting and about 97% GPU utilization compared to prior implementations. We further adapt contrastive language image pretraining (CLIP) to 2DGS by reusing a frozen RGB-based transformer backbone with a lightweight splat aware input stem and a perceiver resampler, training only about 7% of the total parameters. On large DataComp subsets, GS encoders yield meaningful zero shot ImageNet-1K performance while compressing inputs 3 to 20x relative to pixels. While accuracy currently trails RGB encoders, our results establish 2DGS as a viable multimodal substrate, pinpoint architectural bottlenecks, and open a path toward representations that are both semantically powerful and transmission efficient for edge cloud learning.
>
---
#### [new 129] A2R: An Asymmetric Two-Stage Reasoning Framework for Parallel Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出A2R，一种用于复杂任务求解的两阶段推理框架。通过“探索者”和“合成器”模型并行生成与整合解决方案，弥补模型潜在能力与实际表现间的差距，提升推理效率与性能。**

- **链接: [http://arxiv.org/pdf/2509.22044v1](http://arxiv.org/pdf/2509.22044v1)**

> **作者:** Ziqi Wang; Boye Niu; Zhongli Li; Linghui Meng; Jing Liu; Zhi Zheng; Tong Xu; Hua Wu; Haifeng Wang; Enhong Chen
>
> **备注:** 15 pages, 3 figures
>
> **摘要:** Recent Large Reasoning Models have achieved significant improvements in complex task-solving capabilities by allocating more computation at the inference stage with a "thinking longer" paradigm. Even as the foundational reasoning capabilities of models advance rapidly, the persistent gap between a model's performance in a single attempt and its latent potential, often revealed only across multiple solution paths, starkly highlights the disparity between its realized and inherent capabilities. To address this, we present A2R, an Asymmetric Two-Stage Reasoning framework designed to explicitly bridge the gap between a model's potential and its actual performance. In this framework, an "explorer" model first generates potential solutions in parallel through repeated sampling. Subsequently,a "synthesizer" model integrates these references for a more refined, second stage of reasoning. This two-stage process allows computation to be scaled orthogonally to existing sequential methods. Our work makes two key innovations: First, we present A2R as a plug-and-play parallel reasoning framework that explicitly enhances a model's capabilities on complex questions. For example, using our framework, the Qwen3-8B-distill model achieves a 75% performance improvement compared to its self-consistency baseline. Second, through a systematic analysis of the explorer and synthesizer roles, we identify an effective asymmetric scaling paradigm. This insight leads to A2R-Efficient, a "small-to-big" variant that combines a Qwen3-4B explorer with a Qwen3-8B synthesizer. This configuration surpasses the average performance of a monolithic Qwen3-32B model at a nearly 30% lower cost. Collectively, these results show that A2R is not only a performance-boosting framework but also an efficient and practical solution for real-world applications.
>
---
#### [new 130] MDAR: A Multi-scene Dynamic Audio Reasoning Benchmark
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文提出了MDAR，一个多场景动态音频推理基准，旨在评估AI在复杂、多源音频环境中的推理能力。现有基准多为静态单一场景，无法全面反映真实情况。MDAR包含3000个问答对，覆盖五类复杂推理任务，测试26种模型表现，揭示其在多场景推理中的局限性。**

- **链接: [http://arxiv.org/pdf/2509.22461v1](http://arxiv.org/pdf/2509.22461v1)**

> **作者:** Hui Li; Changhao Jiang; Hongyu Wang; Ming Zhang; Jiajun Sun; Zhixiong Yang; Yifei Cao; Shihan Dou; Xiaoran Fan; Baoyu Fan; Tao Ji; Tao Gui; Qi Zhang; Xuanjing Huang
>
> **备注:** 25 pages, 7 figures
>
> **摘要:** The ability to reason from audio, including speech, paralinguistic cues, environmental sounds, and music, is essential for AI agents to interact effectively in real-world scenarios. Existing benchmarks mainly focus on static or single-scene settings and do not fully capture scenarios where multiple speakers, unfolding events, and heterogeneous audio sources interact. To address these challenges, we introduce MDAR, a benchmark for evaluating models on complex, multi-scene, and dynamically evolving audio reasoning tasks. MDAR comprises 3,000 carefully curated question-answer pairs linked to diverse audio clips, covering five categories of complex reasoning and spanning three question types. We benchmark 26 state-of-the-art audio language models on MDAR and observe that they exhibit limitations in complex reasoning tasks. On single-choice questions, Qwen2.5-Omni (open-source) achieves 76.67% accuracy, whereas GPT-4o Audio (closed-source) reaches 68.47%; however, GPT-4o Audio substantially outperforms Qwen2.5-Omni on the more challenging multiple-choice and open-ended tasks. Across all three question types, no model achieves 80% performance. These findings underscore the unique challenges posed by MDAR and its value as a benchmark for advancing audio reasoning research.Code and benchmark can be found at https://github.com/luckyerr/MDAR.
>
---
#### [new 131] Learning Human-Perceived Fakeness in AI-Generated Videos via Multimodal LLMs
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文聚焦于AI生成视频中人类感知的虚假痕迹检测任务，旨在解决如何识别并定位视频中的伪造线索。为此，作者构建了DeeptraceReward数据集，并训练多模态语言模型作为奖励模型，以模仿人类判断与定位能力，提升视频生成的可信度评估。**

- **链接: [http://arxiv.org/pdf/2509.22646v1](http://arxiv.org/pdf/2509.22646v1)**

> **作者:** Xingyu Fu; Siyi Liu; Yinuo Xu; Pan Lu; Guangqiuse Hu; Tianbo Yang; Taran Anantasagar; Christopher Shen; Yikai Mao; Yuanzhe Liu; Keyush Shah; Chung Un Lee; Yejin Choi; James Zou; Dan Roth; Chris Callison-Burch
>
> **备注:** Project Page: https://deeptracereward.github.io/
>
> **摘要:** Can humans identify AI-generated (fake) videos and provide grounded reasons? While video generation models have advanced rapidly, a critical dimension -- whether humans can detect deepfake traces within a generated video, i.e., spatiotemporal grounded visual artifacts that reveal a video as machine generated -- has been largely overlooked. We introduce DeeptraceReward, the first fine-grained, spatially- and temporally- aware benchmark that annotates human-perceived fake traces for video generation reward. The dataset comprises 4.3K detailed annotations across 3.3K high-quality generated videos. Each annotation provides a natural-language explanation, pinpoints a bounding-box region containing the perceived trace, and marks precise onset and offset timestamps. We consolidate these annotations into 9 major categories of deepfake traces that lead humans to identify a video as AI-generated, and train multimodal language models (LMs) as reward models to mimic human judgments and localizations. On DeeptraceReward, our 7B reward model outperforms GPT-5 by 34.7% on average across fake clue identification, grounding, and explanation. Interestingly, we observe a consistent difficulty gradient: binary fake v.s. real classification is substantially easier than fine-grained deepfake trace detection; within the latter, performance degrades from natural language explanations (easiest), to spatial grounding, to temporal labeling (hardest). By foregrounding human-perceived deepfake traces, DeeptraceReward provides a rigorous testbed and training signal for socially aware and trustworthy video generation.
>
---
#### [new 132] ReGeS: Reciprocal Retrieval-Generation Synergy for Conversational Recommender Systems
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文针对对话推荐系统（CRS）任务，旨在解决现有方法依赖领域工程或易产生幻觉的问题。提出ReGeS框架，通过生成与检索的双向协同，提升用户意图理解和物品特征区分能力，实现无需额外标注且效果优异的推荐。**

- **链接: [http://arxiv.org/pdf/2509.21371v1](http://arxiv.org/pdf/2509.21371v1)**

> **作者:** Dayu Yang; Hui Fang
>
> **备注:** Accepted by WISE 2025: 26th International Web Information Systems Engineering conference. Our code is publicly available at the link: https://github.com/dayuyang1999/ReGeS
>
> **摘要:** Connecting conversation with external domain knowledge is vital for conversational recommender systems (CRS) to correctly understand user preferences. However, existing solutions either require domain-specific engineering, which limits flexibility, or rely solely on large language models, which increases the risk of hallucination. While Retrieval-Augmented Generation (RAG) holds promise, its naive use in CRS is hindered by noisy dialogues that weaken retrieval and by overlooked nuances among similar items. We propose ReGeS, a reciprocal Retrieval-Generation Synergy framework that unifies generation-augmented retrieval to distill informative user intent from conversations and retrieval-augmented generation to differentiate subtle item features. This synergy obviates the need for extra annotations, reduces hallucinations, and simplifies continuous updates. Experiments on multiple CRS benchmarks show that ReGeS achieves state-of-the-art performance in recommendation accuracy, demonstrating the effectiveness of reciprocal synergy for knowledge-intensive CRS tasks.
>
---
#### [new 133] PRIME: Planning and Retrieval-Integrated Memory for Enhanced Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出PRIME，一种结合快速直觉（System 1）与深度推理（System 2）的多智能体框架，旨在提升大语言模型在复杂、知识密集型任务中的推理能力。通过实验验证，PRIME使开源模型在多跳和知识驱动任务上接近闭源SOTA模型表现。**

- **链接: [http://arxiv.org/pdf/2509.22315v1](http://arxiv.org/pdf/2509.22315v1)**

> **作者:** Hieu Tran; Zonghai Yao; Nguyen Luong Tran; Zhichao Yang; Feiyun Ouyang; Shuo Han; Razieh Rahimi; Hong Yu
>
> **备注:** 8 pages
>
> **摘要:** Inspired by the dual-process theory of human cognition from \textit{Thinking, Fast and Slow}, we introduce \textbf{PRIME} (Planning and Retrieval-Integrated Memory for Enhanced Reasoning), a multi-agent reasoning framework that dynamically integrates \textbf{System 1} (fast, intuitive thinking) and \textbf{System 2} (slow, deliberate thinking). PRIME first employs a Quick Thinking Agent (System 1) to generate a rapid answer; if uncertainty is detected, it then triggers a structured System 2 reasoning pipeline composed of specialized agents for \textit{planning}, \textit{hypothesis generation}, \textit{retrieval}, \textit{information integration}, and \textit{decision-making}. This multi-agent design faithfully mimics human cognitive processes and enhances both efficiency and accuracy. Experimental results with LLaMA 3 models demonstrate that PRIME enables open-source LLMs to perform competitively with state-of-the-art closed-source models like GPT-4 and GPT-4o on benchmarks requiring multi-hop and knowledge-grounded reasoning. This research establishes PRIME as a scalable solution for improving LLMs in domains requiring complex, knowledge-intensive reasoning.
>
---
#### [new 134] What Makes LLM Agent Simulations Useful for Policy? Insights From an Iterative Design Engagement in Emergency Preparedness
- **分类: cs.HC; cs.CL**

- **简介: 该论文探讨如何使大语言模型代理（LLM agents）模拟真正服务于政策制定。通过与应急准备团队一年的迭代设计，构建了13,000个LLM代理的系统，模拟人群在紧急情况下的行为，为实际政策提供支持，并总结出三项关键设计启示。**

- **链接: [http://arxiv.org/pdf/2509.21868v1](http://arxiv.org/pdf/2509.21868v1)**

> **作者:** Yuxuan Li; Sauvik Das; Hirokazu Shirado
>
> **摘要:** There is growing interest in using Large Language Models as agents (LLM agents) for social simulations to inform policy, yet real-world adoption remains limited. This paper addresses the question: How can LLM agent simulations be made genuinely useful for policy? We report on a year-long iterative design engagement with a university emergency preparedness team. Across multiple design iterations, we iteratively developed a system of 13,000 LLM agents that simulate crowd movement and communication during a large-scale gathering under various emergency scenarios. These simulations informed actual policy implementation, shaping volunteer training, evacuation protocols, and infrastructure planning. Analyzing this process, we identify three design implications: start with verifiable scenarios and build trust gradually, use preliminary simulations to elicit tacit knowledge, and treat simulation and policy development as evolving together. These implications highlight actionable pathways to making LLM agent simulations that are genuinely useful for policy.
>
---
#### [new 135] Gender Stereotypes in Professional Roles Among Saudis: An Analytical Study of AI-Generated Images Using Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于社会与AI交叉研究任务，旨在探究AI生成图像是否强化沙特职业中的性别刻板印象和文化偏差。研究分析了三种AI模型生成的1006张沙特职业图像，评估其在性别、服饰、背景等方面的偏见，揭示当前AI输出存在显著性别失衡和文化不准确性。**

- **链接: [http://arxiv.org/pdf/2509.21466v1](http://arxiv.org/pdf/2509.21466v1)**

> **作者:** Khaloud S. AlKhalifah; Malak Mashaabi; Hend Al-Khalifa
>
> **摘要:** This study investigates the extent to which contemporary Text-to-Image artificial intelligence (AI) models perpetuate gender stereotypes and cultural inaccuracies when generating depictions of professionals in Saudi Arabia. We analyzed 1,006 images produced by ImageFX, DALL-E V3, and Grok for 56 diverse Saudi professions using neutral prompts. Two trained Saudi annotators evaluated each image on five dimensions: perceived gender, clothing and appearance, background and setting, activities and interactions, and age. A third senior researcher adjudicated whenever the two primary raters disagreed, yielding 10,100 individual judgements. The results reveal a strong gender imbalance, with ImageFX outputs being 85\% male, Grok 86.6\% male, and DALL-E V3 96\% male, indicating that DALL-E V3 exhibited the strongest overall gender stereotyping. This imbalance was most evident in leadership and technical roles. Moreover, cultural inaccuracies in clothing, settings, and depicted activities were frequently observed across all three models. Counter-stereotypical images often arise from cultural misinterpretations rather than genuinely progressive portrayals. We conclude that current models mirror societal biases embedded in their training data, generated by humans, offering only a limited reflection of the Saudi labour market's gender dynamics and cultural nuances. These findings underscore the urgent need for more diverse training data, fairer algorithms, and culturally sensitive evaluation frameworks to ensure equitable and authentic visual outputs.
>
---
#### [new 136] MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出MinerU2.5，一种用于高效高分辨率文档解析的视觉-语言模型。它通过解耦全局布局分析与局部内容识别，实现计算效率与识别精度的平衡，在多个基准上取得最优性能。**

- **链接: [http://arxiv.org/pdf/2509.22186v1](http://arxiv.org/pdf/2509.22186v1)**

> **作者:** Junbo Niu; Zheng Liu; Zhuangcheng Gu; Bin Wang; Linke Ouyang; Zhiyuan Zhao; Tao Chu; Tianyao He; Fan Wu; Qintong Zhang; Zhenjiang Jin; Guang Liang; Rui Zhang; Wenzheng Zhang; Yuan Qu; Zhifei Ren; Yuefeng Sun; Yuanhong Zheng; Dongsheng Ma; Zirui Tang; Boyu Niu; Ziyang Miao; Hejun Dong; Siyi Qian; Junyuan Zhang; Jingzhou Chen; Fangdong Wang; Xiaomeng Zhao; Liqun Wei; Wei Li; Shasha Wang; Ruiliang Xu; Yuanyuan Cao; Lu Chen; Qianqian Wu; Huaiyu Gu; Lindong Lu; Keming Wang; Dechen Lin; Guanlin Shen; Xuanhe Zhou; Linfeng Zhang; Yuhang Zang; Xiaoyi Dong; Jiaqi Wang; Bo Zhang; Lei Bai; Pei Chu; Weijia Li; Jiang Wu; Lijun Wu; Zhenxiang Li; Guangyu Wang; Zhongying Tu; Chao Xu; Kai Chen; Yu Qiao; Bowen Zhou; Dahua Lin; Wentao Zhang; Conghui He
>
> **备注:** Technical Report; GitHub Repo: https://github.com/opendatalab/MinerU; Hugging Face Model: https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B; Hugging Face Demo: https://huggingface.co/spaces/opendatalab/MinerU
>
> **摘要:** We introduce MinerU2.5, a 1.2B-parameter document parsing vision-language model that achieves state-of-the-art recognition accuracy while maintaining exceptional computational efficiency. Our approach employs a coarse-to-fine, two-stage parsing strategy that decouples global layout analysis from local content recognition. In the first stage, the model performs efficient layout analysis on downsampled images to identify structural elements, circumventing the computational overhead of processing high-resolution inputs. In the second stage, guided by the global layout, it performs targeted content recognition on native-resolution crops extracted from the original image, preserving fine-grained details in dense text, complex formulas, and tables. To support this strategy, we developed a comprehensive data engine that generates diverse, large-scale training corpora for both pretraining and fine-tuning. Ultimately, MinerU2.5 demonstrates strong document parsing ability, achieving state-of-the-art performance on multiple benchmarks, surpassing both general-purpose and domain-specific models across various recognition tasks, while maintaining significantly lower computational overhead.
>
---
#### [new 137] UltraHorizon: Benchmarking Agent Capabilities in Ultra Long-Horizon Scenarios
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出UltraHorizon基准，用于评估智能体在超长时序、部分可观测任务中的能力。针对现有评测缺乏对长期推理、规划和工具使用的考察，设计了包含探索任务的环境，揭示LLM在长时序任务中的不足，并分析其失败原因。**

- **链接: [http://arxiv.org/pdf/2509.21766v1](http://arxiv.org/pdf/2509.21766v1)**

> **作者:** Haotian Luo; Huaisong Zhang; Xuelin Zhang; Haoyu Wang; Zeyu Qin; Wenjie Lu; Guozheng Ma; Haiying He; Yingsha Xie; Qiyang Zhou; Zixuan Hu; Hongze Mi; Yibo Wang; Naiqiang Tan; Hong Chen; Yi R. Fung; Chun Yuan; Li Shen
>
> **摘要:** Autonomous agents have recently achieved remarkable progress across diverse domains, yet most evaluations focus on short-horizon, fully observable tasks. In contrast, many critical real-world tasks, such as large-scale software development, commercial investment, and scientific discovery, unfold in long-horizon and partially observable scenarios where success hinges on sustained reasoning, planning, memory management, and tool use. Existing benchmarks rarely capture these long-horizon challenges, leaving a gap in systematic evaluation. To bridge this gap, we introduce \textbf{UltraHorizon} a novel benchmark that measures the foundational capabilities essential for complex real-world challenges. We use exploration as a unifying task across three distinct environments to validate these core competencies. Agents are designed in long-horizon discovery tasks where they must iteratively uncover hidden rules through sustained reasoning, planning, memory and tools management, and interaction with environments. Under the heaviest scale setting, trajectories average \textbf{200k+} tokens and \textbf{400+} tool calls, whereas in standard configurations they still exceed \textbf{35k} tokens and involve more than \textbf{60} tool calls on average. Our extensive experiments reveal that LLM-agents consistently underperform in these settings, whereas human participants achieve higher scores, underscoring a persistent gap in agents' long-horizon abilities. We also observe that simple scaling fails in our task. To better illustrate the failure of agents, we conduct an in-depth analysis of collected trajectories. We identify eight types of errors and attribute them to two primary causes: in-context locking and functional fundamental capability gaps. \href{https://github.com/StarDewXXX/UltraHorizon}{Our code will be available here.}
>
---
#### [new 138] Evaluating Open-Source Large Language Models for Technical Telecom Question Answering
- **分类: cs.NI; cs.CL**

- **简介: 该论文属于技术问答评估任务，旨在探究开源大语言模型在电信领域的表现。研究对比了Gemma和DeepSeek两模型在语义准确性、一致性及幻觉等方面的性能，构建了105道电信相关问答对的基准测试，以支持可信AI助手的开发。**

- **链接: [http://arxiv.org/pdf/2509.21949v1](http://arxiv.org/pdf/2509.21949v1)**

> **作者:** Arina Caraus; Alessio Buscemi; Sumit Kumar; Ion Turcanu
>
> **备注:** Accepted at the IEEE GLOBECOM Workshops 2025: "Large AI Model over Future Wireless Networks"
>
> **摘要:** Large Language Models (LLMs) have shown remarkable capabilities across various fields. However, their performance in technical domains such as telecommunications remains underexplored. This paper evaluates two open-source LLMs, Gemma 3 27B and DeepSeek R1 32B, on factual and reasoning-based questions derived from advanced wireless communications material. We construct a benchmark of 105 question-answer pairs and assess performance using lexical metrics, semantic similarity, and LLM-as-a-judge scoring. We also analyze consistency, judgment reliability, and hallucination through source attribution and score variance. Results show that Gemma excels in semantic fidelity and LLM-rated correctness, while DeepSeek demonstrates slightly higher lexical consistency. Additional findings highlight current limitations in telecom applications and the need for domain-adapted models to support trustworthy Artificial Intelligence (AI) assistants in engineering.
>
---
#### [new 139] Bridging Kolmogorov Complexity and Deep Learning: Asymptotically Optimal Description Length Objectives for Transformers
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于理论机器学习任务，旨在解决神经网络模型复杂度缺乏统一衡量标准的问题。作者提出基于Kolmogorov复杂度的最优描述长度目标，证明其在Transformer中的可行性，并设计可微变分目标以实现高效压缩与泛化。**

- **链接: [http://arxiv.org/pdf/2509.22445v1](http://arxiv.org/pdf/2509.22445v1)**

> **作者:** Peter Shaw; James Cohan; Jacob Eisenstein; Kristina Toutanova
>
> **摘要:** The Minimum Description Length (MDL) principle offers a formal framework for applying Occam's razor in machine learning. However, its application to neural networks such as Transformers is challenging due to the lack of a principled, universal measure for model complexity. This paper introduces the theoretical notion of asymptotically optimal description length objectives, grounded in the theory of Kolmogorov complexity. We establish that a minimizer of such an objective achieves optimal compression, for any dataset, up to an additive constant, in the limit as model resource bounds increase. We prove that asymptotically optimal objectives exist for Transformers, building on a new demonstration of their computational universality. We further show that such objectives can be tractable and differentiable by constructing and analyzing a variational objective based on an adaptive Gaussian mixture prior. Our empirical analysis shows that this variational objective selects for a low-complexity solution with strong generalization on an algorithmic task, but standard optimizers fail to find such solutions from a random initialization, highlighting key optimization challenges. More broadly, by providing a theoretical framework for identifying description length objectives with strong asymptotic guarantees, we outline a potential path towards training neural networks that achieve greater compression and generalization.
>
---
#### [new 140] Towards mitigating information leakage when evaluating safety monitors
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文针对安全监控评估中的信息泄露问题，提出内容过滤、评分过滤和微调模型生物三种缓解策略，旨在更准确地评估监控器检测真实模型行为的能力。**

- **链接: [http://arxiv.org/pdf/2509.21344v1](http://arxiv.org/pdf/2509.21344v1)**

> **作者:** Gerard Boxo; Aman Neelappa; Shivam Raval
>
> **备注:** 14 pages, 4 figures
>
> **摘要:** White box monitors that analyze model internals offer promising advantages for detecting potentially harmful behaviors in large language models, including lower computational costs and integration into layered defense systems.However, training and evaluating these monitors requires response exemplars that exhibit the target behaviors, typically elicited through prompting or fine-tuning. This presents a challenge when the information used to elicit behaviors inevitably leaks into the data that monitors ingest, inflating their effectiveness. We present a systematic framework for evaluating a monitor's performance in terms of its ability to detect genuine model behavior rather than superficial elicitation artifacts. Furthermore, we propose three novel strategies to evaluate the monitor: content filtering (removing deception-related text from inputs), score filtering (aggregating only over task-relevant tokens), and prompt distilled fine-tuned model organisms (models trained to exhibit deceptive behavior without explicit prompting). Using deception detection as a representative case study, we identify two forms of leakage that inflate monitor performance: elicitation leakage from prompts that explicitly request harmful behavior, and reasoning leakage from models that verbalize their deceptive actions. Through experiments on multiple deception benchmarks, we apply our proposed mitigation strategies and measure performance retention. Our evaluation of the monitors reveal three crucial findings: (1) Content filtering is a good mitigation strategy that allows for a smooth removal of elicitation signal and can decrease probe AUROC by 30\% (2) Score filtering was found to reduce AUROC by 15\% but is not as straightforward to attribute to (3) A finetuned model organism improves monitor evaluations but reduces their performance by upto 40\%, even when re-trained.
>
---
#### [new 141] Does AI Coaching Prepare us for Workplace Negotiations?
- **分类: cs.HC; cs.AI; cs.CL; cs.CY**

- **简介: 该论文研究AI在职场谈判准备中的效果，构建了基于理论的AI教练Trucey，并通过实验和访谈对比其与ChatGPT和传统手册的表现。结果显示，AI虽能降低恐惧，但在可用性和心理赋能上不如手册，提出需结合结构化内容与AI优势的设计方向。**

- **链接: [http://arxiv.org/pdf/2509.22545v1](http://arxiv.org/pdf/2509.22545v1)**

> **作者:** Veda Duddu; Jash Rajesh Parekh; Andy Mao; Hanyi Min; Ziang Xiao; Vedant Das Swain; Koustuv Saha
>
> **摘要:** Workplace negotiations are undermined by psychological barriers, which can even derail well-prepared tactics. AI offers personalized and always -- available negotiation coaching, yet its effectiveness for negotiation preparedness remains unclear. We built Trucey, a prototype AI coach grounded in Brett's negotiation model. We conducted a between-subjects experiment (N=267), comparing Trucey, ChatGPT, and a traditional negotiation Handbook, followed by in-depth interviews (N=15). While Trucey showed the strongest reductions in fear relative to both comparison conditions, the Handbook outperformed both AIs in usability and psychological empowerment. Interviews revealed that the Handbook's comprehensive, reviewable content was crucial for participants' confidence and preparedness. In contrast, although participants valued AI's rehearsal capability, its guidance often felt verbose and fragmented -- delivered in bits and pieces that required additional effort -- leaving them uncertain or overwhelmed. These findings challenge assumptions of AI superiority and motivate hybrid designs that integrate structured, theory-driven content with targeted rehearsal, clear boundaries, and adaptive scaffolds to address psychological barriers and support negotiation preparedness.
>
---
#### [new 142] CapRL: Stimulating Dense Image Caption Capabilities via Reinforcement Learning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文聚焦图像描述生成任务，旨在解决监督微调方法导致模型泛化能力差、描述单一的问题。提出CapRL框架，利用强化学习通过语言模型回答问题的准确性作为奖励，提升图像描述的多样性和实用性。**

- **链接: [http://arxiv.org/pdf/2509.22647v1](http://arxiv.org/pdf/2509.22647v1)**

> **作者:** Long Xing; Xiaoyi Dong; Yuhang Zang; Yuhang Cao; Jianze Liang; Qidong Huang; Jiaqi Wang; Feng Wu; Dahua Lin
>
> **备注:** Code is available at https://github.com/InternLM/CapRL
>
> **摘要:** Image captioning is a fundamental task that bridges the visual and linguistic domains, playing a critical role in pre-training Large Vision-Language Models (LVLMs). Current state-of-the-art captioning models are typically trained with Supervised Fine-Tuning (SFT), a paradigm that relies on expensive, non-scalable data annotated by humans or proprietary models. This approach often leads to models that memorize specific ground-truth answers, limiting their generality and ability to generate diverse, creative descriptions. To overcome the limitation of SFT, we propose applying the Reinforcement Learning with Verifiable Rewards (RLVR) paradigm to the open-ended task of image captioning. A primary challenge, however, is designing an objective reward function for the inherently subjective nature of what constitutes a "good" caption. We introduce Captioning Reinforcement Learning (CapRL), a novel training framework that redefines caption quality through its utility: a high-quality caption should enable a non-visual language model to accurately answer questions about the corresponding image. CapRL employs a decoupled two-stage pipeline where an LVLM generates a caption, and the objective reward is derived from the accuracy of a separate, vision-free LLM answering Multiple-Choice Questions based solely on that caption. As the first study to apply RLVR to the subjective image captioning task, we demonstrate that CapRL significantly enhances multiple settings. Pretraining on the CapRL-5M caption dataset annotated by CapRL-3B results in substantial gains across 12 benchmarks. Moreover, within the Prism Framework for caption quality evaluation, CapRL achieves performance comparable to Qwen2.5-VL-72B, while exceeding the baseline by an average margin of 8.4%. Code is available here: https://github.com/InternLM/CapRL.
>
---
#### [new 143] InvBench: Can LLMs Accelerate Program Verification with Invariant Synthesis?
- **分类: cs.PL; cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究LLMs在程序验证中的不变式合成任务，旨在解决自动发现强不变式的难题。提出InvBench框架评估LLMs的性能，对比7种模型与传统求解器UAutomizer，结果显示LLM表现尚不占优。通过微调和采样优化可提升效果，表明该任务对当前LLMs仍是挑战。**

- **链接: [http://arxiv.org/pdf/2509.21629v1](http://arxiv.org/pdf/2509.21629v1)**

> **作者:** Anjiang Wei; Tarun Suresh; Tianran Sun; Haoze Wu; Ke Wang; Alex Aiken
>
> **摘要:** Program verification relies on loop invariants, yet automatically discovering strong invariants remains a long-standing challenge. We introduce a principled framework for evaluating LLMs on invariant synthesis. Our approach uses a verifier-based decision procedure with a formal soundness guarantee and assesses not only correctness but also the speedup that invariants provide in verification. We evaluate 7 state-of-the-art LLMs, and existing LLM-based verifiers against the traditional solver UAutomizer. While LLM-based verifiers represent a promising direction, they do not yet offer a significant advantage over UAutomizer. Model capability also proves critical, as shown by sharp differences in speedups across models, and our benchmark remains an open challenge for current LLMs. Finally, we show that supervised fine-tuning and Best-of-N sampling can improve performance: fine-tuning on 3589 instances raises the percentage of speedup cases for Qwen3-Coder-480B from 8% to 29.2%, and Best-of-N sampling with N=16 improves Claude-sonnet-4 from 8.8% to 22.1%.
>
---
#### [new 144] LLM Agent Meets Agentic AI: Can LLM Agents Simulate Customers to Evaluate Agentic-AI-based Shopping Assistants?
- **分类: cs.HC; cs.CL**

- **简介: 该论文研究LLM代理能否模拟用户与Agentic AI购物助手（如Amazon Rufus）进行多轮交互。通过招募40名用户并构建数字孪生，对比人类与代理的交互轨迹和反馈，发现代理能较好模拟用户行为，验证了其在可扩展评估中的潜力。**

- **链接: [http://arxiv.org/pdf/2509.21501v1](http://arxiv.org/pdf/2509.21501v1)**

> **作者:** Lu Sun; Shihan Fu; Bingsheng Yao; Yuxuan Lu; Wenbo Li; Hansu Gu; Jiri Gesi; Jing Huang; Chen Luo; Dakuo Wang
>
> **摘要:** Agentic AI is emerging, capable of executing tasks through natural language, such as Copilot for coding or Amazon Rufus for shopping. Evaluating these systems is challenging, as their rapid evolution outpaces traditional human evaluation. Researchers have proposed LLM Agents to simulate participants as digital twins, but it remains unclear to what extent a digital twin can represent a specific customer in multi-turn interaction with an agentic AI system. In this paper, we recruited 40 human participants to shop with Amazon Rufus, collected their personas, interaction traces, and UX feedback, and then created digital twins to repeat the task. Pairwise comparison of human and digital-twin traces shows that while agents often explored more diverse choices, their action patterns aligned with humans and yielded similar design feedback. This study is the first to quantify how closely LLM agents can mirror human multi-turn interaction with an agentic AI system, highlighting their potential for scalable evaluation.
>
---
#### [new 145] InfiMed-Foundation: Pioneering Advanced Multimodal Medical Models with Compute-Efficient Pre-Training and Multi-Stage Fine-Tuning
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出InfiMed-Foundation系列模型，针对医疗领域多模态任务，解决通用模型知识不足、训练效率低的问题。通过高质量数据筛选、分阶段微调和高效训练策略，提升了医学问答与诊断性能。**

- **链接: [http://arxiv.org/pdf/2509.22261v1](http://arxiv.org/pdf/2509.22261v1)**

> **作者:** Guanghao Zhu; Zhitian Hou; Zeyu Liu; Zhijie Sang; Congkai Xie; Hongxia Yang
>
> **摘要:** Multimodal large language models (MLLMs) have shown remarkable potential in various domains, yet their application in the medical field is hindered by several challenges. General-purpose MLLMs often lack the specialized knowledge required for medical tasks, leading to uncertain or hallucinatory responses. Knowledge distillation from advanced models struggles to capture domain-specific expertise in radiology and pharmacology. Additionally, the computational cost of continual pretraining with large-scale medical data poses significant efficiency challenges. To address these issues, we propose InfiMed-Foundation-1.7B and InfiMed-Foundation-4B, two medical-specific MLLMs designed to deliver state-of-the-art performance in medical applications. We combined high-quality general-purpose and medical multimodal data and proposed a novel five-dimensional quality assessment framework to curate high-quality multimodal medical datasets. We employ low-to-high image resolution and multimodal sequence packing to enhance training efficiency, enabling the integration of extensive medical data. Furthermore, a three-stage supervised fine-tuning process ensures effective knowledge extraction for complex medical tasks. Evaluated on the MedEvalKit framework, InfiMed-Foundation-1.7B outperforms Qwen2.5VL-3B, while InfiMed-Foundation-4B surpasses HuatuoGPT-V-7B and MedGemma-27B-IT, demonstrating superior performance in medical visual question answering and diagnostic tasks. By addressing key challenges in data quality, training efficiency, and domain-specific knowledge extraction, our work paves the way for more reliable and effective AI-driven solutions in healthcare. InfiMed-Foundation-4B model is available at \href{https://huggingface.co/InfiX-ai/InfiMed-Foundation-4B}{InfiMed-Foundation-4B}.
>
---
#### [new 146] Speak Your Mind: The Speech Continuation Task as a Probe of Voice-Based Model Bias
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文研究语音续说任务（SC）中语音基础模型的偏见问题，通过分析性别和发音类型对续说结果的影响，评估了三个模型在语音身份保持、语义连贯性及文本偏见方面的表现，揭示了系统性语音质量偏差。**

- **链接: [http://arxiv.org/pdf/2509.22061v1](http://arxiv.org/pdf/2509.22061v1)**

> **作者:** Shree Harsha Bokkahalli Satish; Harm Lameris; Olivier Perrotin; Gustav Eje Henter; Éva Székely
>
> **备注:** 6 pages, 1 figure, Submitted to IEEE ICASSP 2026
>
> **摘要:** Speech Continuation (SC) is the task of generating a coherent extension of a spoken prompt while preserving both semantic context and speaker identity. Because SC is constrained to a single audio stream, it offers a more direct setting for probing biases in speech foundation models than dialogue does. In this work we present the first systematic evaluation of bias in SC, investigating how gender and phonation type (breathy, creaky, end-creak) affect continuation behaviour. We evaluate three recent models: SpiritLM (base and expressive), VAE-GSLM, and SpeechGPT across speaker similarity, voice quality preservation, and text-based bias metrics. Results show that while both speaker similarity and coherence remain a challenge, textual evaluations reveal significant model and gender interactions: once coherence is sufficiently high (for VAE-GSLM), gender effects emerge on text-metrics such as agency and sentence polarity. In addition, continuations revert toward modal phonation more strongly for female prompts than for male ones, revealing a systematic voice-quality bias. These findings highlight SC as a controlled probe of socially relevant representational biases in speech foundation models, and suggest that it will become an increasingly informative diagnostic as continuation quality improves.
>
---
#### [new 147] You Can't Steal Nothing: Mitigating Prompt Leakages in LLMs via System Vectors
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文研究LLM中的系统提示泄露问题，提出一种新型攻击方法，并设计SysVec方案将系统提示编码为向量以增强安全性与模型性能。**

- **链接: [http://arxiv.org/pdf/2509.21884v1](http://arxiv.org/pdf/2509.21884v1)**

> **作者:** Bochuan Cao; Changjiang Li; Yuanpu Cao; Yameng Ge; Ting Wang; Jinghui Chen
>
> **备注:** 29 pages, 10 tables, 6figures, accepted by CCS 25
>
> **摘要:** Large language models (LLMs) have been widely adopted across various applications, leveraging customized system prompts for diverse tasks. Facing potential system prompt leakage risks, model developers have implemented strategies to prevent leakage, primarily by disabling LLMs from repeating their context when encountering known attack patterns. However, it remains vulnerable to new and unforeseen prompt-leaking techniques. In this paper, we first introduce a simple yet effective prompt leaking attack to reveal such risks. Our attack is capable of extracting system prompts from various LLM-based application, even from SOTA LLM models such as GPT-4o or Claude 3.5 Sonnet. Our findings further inspire us to search for a fundamental solution to the problems by having no system prompt in the context. To this end, we propose SysVec, a novel method that encodes system prompts as internal representation vectors rather than raw text. By doing so, SysVec minimizes the risk of unauthorized disclosure while preserving the LLM's core language capabilities. Remarkably, this approach not only enhances security but also improves the model's general instruction-following abilities. Experimental results demonstrate that SysVec effectively mitigates prompt leakage attacks, preserves the LLM's functional integrity, and helps alleviate the forgetting issue in long-context scenarios.
>
---
#### [new 148] See, Point, Fly: A Learning-Free VLM Framework for Universal Unmanned Aerial Navigation
- **分类: cs.RO; cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文提出See, Point, Fly（SPF），一种无需训练的视觉-语言导航框架，用于无人机在复杂环境中根据自然语言指令自主导航。其核心是将动作预测转化为2D空间定位任务，并结合3D位移控制无人机，实现了高效、通用的闭环导航。**

- **链接: [http://arxiv.org/pdf/2509.22653v1](http://arxiv.org/pdf/2509.22653v1)**

> **作者:** Chih Yao Hu; Yang-Sen Lin; Yuna Lee; Chih-Hai Su; Jie-Ying Lee; Shr-Ruei Tsai; Chin-Yang Lin; Kuan-Wen Chen; Tsung-Wei Ke; Yu-Lun Liu
>
> **备注:** CoRL 2025. Project page: https://spf-web.pages.dev
>
> **摘要:** We present See, Point, Fly (SPF), a training-free aerial vision-and-language navigation (AVLN) framework built atop vision-language models (VLMs). SPF is capable of navigating to any goal based on any type of free-form instructions in any kind of environment. In contrast to existing VLM-based approaches that treat action prediction as a text generation task, our key insight is to consider action prediction for AVLN as a 2D spatial grounding task. SPF harnesses VLMs to decompose vague language instructions into iterative annotation of 2D waypoints on the input image. Along with the predicted traveling distance, SPF transforms predicted 2D waypoints into 3D displacement vectors as action commands for UAVs. Moreover, SPF also adaptively adjusts the traveling distance to facilitate more efficient navigation. Notably, SPF performs navigation in a closed-loop control manner, enabling UAVs to follow dynamic targets in dynamic environments. SPF sets a new state of the art in DRL simulation benchmark, outperforming the previous best method by an absolute margin of 63%. In extensive real-world evaluations, SPF outperforms strong baselines by a large margin. We also conduct comprehensive ablation studies to highlight the effectiveness of our design choice. Lastly, SPF shows remarkable generalization to different VLMs. Project page: https://spf-web.pages.dev
>
---
#### [new 149] Dynamic Experts Search: Enhancing Reasoning in Mixture-of-Experts LLMs at Test Time
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究推理增强任务，针对MoE大模型在推理时的专家激活策略。提出DES方法，通过动态控制专家数量和继承配置，在不增加成本的情况下提升推理多样性与稳定性，实验证明其优于现有TTS基线。**

- **链接: [http://arxiv.org/pdf/2509.22572v1](http://arxiv.org/pdf/2509.22572v1)**

> **作者:** Yixuan Han; Fan Ma; Ruijie Quan; Yi Yang
>
> **摘要:** Test-Time Scaling (TTS) enhances the reasoning ability of large language models (LLMs) by allocating additional computation during inference. However, existing approaches primarily rely on output-level sampling while overlooking the role of model architecture. In mainstream Mixture-of-Experts (MoE) LLMs, we observe that varying the number of activated experts yields complementary solution sets with stable accuracy, revealing a new and underexplored source of diversity. Motivated by this observation, we propose Dynamic Experts Search (DES), a TTS strategy that elevates expert activation into a controllable dimension of the search space. DES integrates two key components: (1) Dynamic MoE, which enables direct control of expert counts during inference to generate diverse reasoning trajectories without additional cost; and (2) Expert Configuration Inheritance, which preserves consistent expert counts within a reasoning path while varying them across runs, thereby balancing stability and diversity throughout the search. Extensive experiments across MoE architectures, verifiers and reasoning benchmarks (i.e., math, code and knowledge) demonstrate that DES reliably outperforms TTS baselines, enhancing accuracy and stability without additional cost. These results highlight DES as a practical and scalable form of architecture-aware TTS, illustrating how structural flexibility in modern LLMs can advance reasoning.
>
---
#### [new 150] C-QUERI: Congressional Questions, Exchanges, and Responses in Institutions Dataset
- **分类: cs.CY; cs.CL**

- **简介: 该论文提出了C-QUERI数据集，旨在研究政治听证中提问策略的党派差异。通过从国会听证转录文本中提取问答对，构建了108至117届国会的数据集，并展示了仅凭问题即可预测提问者党派，为政治话语分析提供了新框架。**

- **链接: [http://arxiv.org/pdf/2509.21548v1](http://arxiv.org/pdf/2509.21548v1)**

> **作者:** Manjari Rudra; Daniel Magleby; Sujoy Sikdar
>
> **摘要:** Questions in political interviews and hearings serve strategic purposes beyond information gathering including advancing partisan narratives and shaping public perceptions. However, these strategic aspects remain understudied due to the lack of large-scale datasets for studying such discourse. Congressional hearings provide an especially rich and tractable site for studying political questioning: Interactions are structured by formal rules, witnesses are obliged to respond, and members with different political affiliations are guaranteed opportunities to ask questions, enabling comparisons of behaviors across the political spectrum. We develop a pipeline to extract question-answer pairs from unstructured hearing transcripts and construct a novel dataset of committee hearings from the 108th--117th Congress. Our analysis reveals systematic differences in questioning strategies across parties, by showing the party affiliation of questioners can be predicted from their questions alone. Our dataset and methods not only advance the study of congressional politics, but also provide a general framework for analyzing question-answering across interview-like settings.
>
---
#### [new 151] Random Direct Preference Optimization for Radiography Report Generation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文针对放射科报告生成（RRG）任务，旨在提升其临床实用性。提出一种无需奖励模型或人工标注的随机直接偏好优化（Random DPO）框架，通过对比采样增强模型性能，实验表明可提升5%的临床指标。**

- **链接: [http://arxiv.org/pdf/2509.21351v1](http://arxiv.org/pdf/2509.21351v1)**

> **作者:** Valentin Samokhin; Boris Shirokikh; Mikhail Goncharov; Dmitriy Umerenkov; Maksim Bobrin; Ivan Oseledets; Dmitry Dylov; Mikhail Belyaev
>
> **摘要:** Radiography Report Generation (RRG) has gained significant attention in medical image analysis as a promising tool for alleviating the growing workload of radiologists. However, despite numerous advancements, existing methods have yet to achieve the quality required for deployment in real-world clinical settings. Meanwhile, large Visual Language Models (VLMs) have demonstrated remarkable progress in the general domain by adopting training strategies originally designed for Large Language Models (LLMs), such as alignment techniques. In this paper, we introduce a model-agnostic framework to enhance RRG accuracy using Direct Preference Optimization (DPO). Our approach leverages random contrastive sampling to construct training pairs, eliminating the need for reward models or human preference annotations. Experiments on supplementing three state-of-the-art models with our Random DPO show that our method improves clinical performance metrics by up to 5%, without requiring any additional training data.
>
---
#### [new 152] Learning GUI Grounding with Spatial Reasoning from Visual Feedback
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对GUI定位任务中坐标预测不准确的问题，提出将GUI定位重构为交互式搜索任务。通过引入光标移动与视觉反馈机制，结合强化学习训练模型GUI-Cursor，提升了高分辨率复杂界面下的定位精度，在多个数据集上取得SOTA效果。**

- **链接: [http://arxiv.org/pdf/2509.21552v1](http://arxiv.org/pdf/2509.21552v1)**

> **作者:** Yu Zhao; Wei-Ning Chen; Huseyin Atahan Inan; Samuel Kessler; Lu Wang; Lukas Wutschitz; Fangkai Yang; Chaoyun Zhang; Pasquale Minervini; Saravan Rajmohan; Robert Sim
>
> **摘要:** Graphical User Interface (GUI) grounding is commonly framed as a coordinate prediction task -- given a natural language instruction, generate on-screen coordinates for actions such as clicks and keystrokes. However, recent Vision Language Models (VLMs) often fail to predict accurate numeric coordinates when processing high-resolution GUI images with complex layouts. To address this issue, we reframe GUI grounding as an \emph{interactive search task}, where the VLM generates actions to move a cursor in the GUI to locate UI elements. At each step, the model determines the target object, evaluates the spatial relations between the cursor and the target, and moves the cursor closer to the target conditioned on the movement history. In this interactive process, the rendered cursor provides visual feedback to help the model align its predictions with the corresponding on-screen locations. We train our GUI grounding model, GUI-Cursor, using multi-step online reinforcement learning with a dense trajectory-based reward function. Our experimental results show that GUI-Cursor, based on Qwen2.5-VL-7B, improves the GUI grounding accuracy and achieves state-of-the-art results on ScreenSpot-v2 ($88.8\% \rightarrow 93.9\%$) and ScreenSpot-Pro ($26.8\% \rightarrow 56.5\%$). Moreover, we observe that GUI-Cursor learns to solve the problem within two steps for 95\% of instances and can adaptively conduct more steps on more difficult examples.
>
---
#### [new 153] Mental Health Impacts of AI Companions: Triangulating Social Media Quasi-Experiments, User Perspectives, and Relational Theory
- **分类: cs.HC; cs.AI; cs.CL; cs.CY; stat.AP**

- **简介: 该论文研究AI伴侣（如Replika）对心理健康的影响，结合社交媒体数据分析、用户访谈和关系理论，探讨其对情绪表达、孤独感等的作用，并提出设计建议以促进健康使用。**

- **链接: [http://arxiv.org/pdf/2509.22505v1](http://arxiv.org/pdf/2509.22505v1)**

> **作者:** Yunhao Yuan; Jiaxun Zhang; Talayeh Aledavood; Renwen Zhang; Koustuv Saha
>
> **摘要:** AI-powered companion chatbots (AICCs) such as Replika are increasingly popular, offering empathetic interactions, yet their psychosocial impacts remain unclear. We examined how engaging with AICCs shaped wellbeing and how users perceived these experiences. First, we conducted a large-scale quasi-experimental study of longitudinal Reddit data, applying stratified propensity score matching and Difference-in-Differences regression. Findings revealed mixed effects -- greater affective and grief expression, readability, and interpersonal focus, alongside increases in language about loneliness and suicidal ideation. Second, we complemented these results with 15 semi-structured interviews, which we thematically analyzed and contextualized using Knapp's relationship development model. We identified trajectories of initiation, escalation, and bonding, wherein AICCs provided emotional validation and social rehearsal but also carried risks of over-reliance and withdrawal. Triangulating across methods, we offer design implications for AI companions that scaffold healthy boundaries, support mindful engagement, support disclosure without dependency, and surface relationship stages -- maximizing psychosocial benefits while mitigating risks.
>
---
#### [new 154] Learn the Ropes, Then Trust the Wins: Self-imitation with Progressive Exploration for Agentic Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.MA**

- **简介: 该论文针对智能体在长期稀疏奖励任务中的探索-利用平衡问题，提出SPEAR方法。通过渐进式课程学习和自模仿学习，结合内在奖励与经验回放，稳定训练并提升LLM的工具使用能力。**

- **链接: [http://arxiv.org/pdf/2509.22601v1](http://arxiv.org/pdf/2509.22601v1)**

> **作者:** Yulei Qin; Xiaoyu Tan; Zhengbao He; Gang Li; Haojia Lin; Zongyi Li; Zihan Xu; Yuchen Shi; Siqi Cai; Renting Rui; Shaofei Cai; Yuzheng Cai; Xuan Zhang; Sheng Ye; Ke Li; Xing Sun
>
> **备注:** 26 pages, 11 figures
>
> **摘要:** Reinforcement learning (RL) is the dominant paradigm for sharpening strategic tool use capabilities of LLMs on long-horizon, sparsely-rewarded agent tasks, yet it faces a fundamental challenge of exploration-exploitation trade-off. Existing studies stimulate exploration through the lens of policy entropy, but such mechanical entropy maximization is prone to RL training instability due to the multi-turn distribution shifting. In this paper, we target the progressive exploration-exploitation balance under the guidance of the agent own experiences without succumbing to either entropy collapsing or runaway divergence. We propose SPEAR, a curriculum-based self-imitation learning (SIL) recipe for training agentic LLMs. It extends the vanilla SIL framework, where a replay buffer stores self-generated promising trajectories for off-policy update, by gradually steering the policy evolution within a well-balanced range of entropy across stages. Specifically, our approach incorporates a curriculum to manage the exploration process, utilizing intrinsic rewards to foster skill-level exploration and facilitating action-level exploration through SIL. At first, the auxiliary tool call reward plays a critical role in the accumulation of tool-use skills, enabling broad exposure to the unfamiliar distributions of the environment feedback with an upward entropy trend. As training progresses, self-imitation gets strengthened to exploit existing successful patterns from replayed experiences for comparative action-level exploration, accelerating solution iteration without unbounded entropy growth. To further stabilize training, we recalibrate the advantages of experiences in the replay buffer to address the potential policy drift. Reugularizations such as the clipping of tokens with high covariance between probability and advantage are introduced to the trajectory-level entropy control to curb over-confidence.
>
---
#### [new 155] Accelerate Creation of Product Claims Using Generative AI
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出一个名为Claim Advisor的Web应用，利用生成式AI加速产品宣称的创建。任务是优化宣称生成流程，解决传统方法耗时费力的问题。工作包括：语义搜索、宣称生成与优化、模拟评估宣称效果，并在消费品行业取得良好效果。**

- **链接: [http://arxiv.org/pdf/2509.20652v1](http://arxiv.org/pdf/2509.20652v1)**

> **作者:** Po-Yu Liang; Yong Zhang; Tatiana Hwa; Aaron Byers
>
> **备注:** This paper has been accepted at the GenProCC workshop (NeurIPS 2025)
>
> **摘要:** The benefit claims of a product is a critical driver of consumers' purchase behavior. Creating product claims is an intense task that requires substantial time and funding. We have developed the $\textbf{Claim Advisor}$ web application to accelerate claim creations using in-context learning and fine-tuning of large language models (LLM). $\textbf{Claim Advisor}$ was designed to disrupt the speed and economics of claim search, generation, optimization, and simulation. It has three functions: (1) semantically searching and identifying existing claims and/or visuals that resonate with the voice of consumers; (2) generating and/or optimizing claims based on a product description and a consumer profile; and (3) ranking generated and/or manually created claims using simulations via synthetic consumers. Applications in a consumer packaged goods (CPG) company have shown very promising results. We believe that this capability is broadly useful and applicable across product categories and industries. We share our learning to encourage the research and application of generative AI in different industries.
>
---
#### [new 156] ARTI-6: Towards Six-dimensional Articulatory Speech Encoding
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文提出ARTI-6，一种六维语音编码框架，用于解决语音与发音动作的互推问题。通过MRI数据提取关键发音区域特征，构建了预测模型和合成模型，实现语音与发音动作的高效转换，推动语音技术发展。**

- **链接: [http://arxiv.org/pdf/2509.21447v1](http://arxiv.org/pdf/2509.21447v1)**

> **作者:** Jihwan Lee; Sean Foley; Thanathai Lertpetchpun; Kevin Huang; Yoonjeong Lee; Tiantian Feng; Louis Goldstein; Dani Byrd; Shrikanth Narayanan
>
> **摘要:** We propose ARTI-6, a compact six-dimensional articulatory speech encoding framework derived from real-time MRI data that captures crucial vocal tract regions including the velum, tongue root, and larynx. ARTI-6 consists of three components: (1) a six-dimensional articulatory feature set representing key regions of the vocal tract; (2) an articulatory inversion model, which predicts articulatory features from speech acoustics leveraging speech foundation models, achieving a prediction correlation of 0.87; and (3) an articulatory synthesis model, which reconstructs intelligible speech directly from articulatory features, showing that even a low-dimensional representation can generate natural-sounding speech. Together, ARTI-6 provides an interpretable, computationally efficient, and physiologically grounded framework for advancing articulatory inversion, synthesis, and broader speech technology applications. The source code and speech samples are publicly available.
>
---
#### [new 157] AUDDT: Audio Unified Deepfake Detection Benchmark Toolkit
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文提出了AUDDT，一个用于音频深度伪造检测的开源基准工具包。针对现有检测模型在数据集泛化能力不足的问题，系统梳理了28个数据集，并自动化评估预训练检测器的性能，揭示其在不同条件下的表现差异和局限性。**

- **链接: [http://arxiv.org/pdf/2509.21597v1](http://arxiv.org/pdf/2509.21597v1)**

> **作者:** Yi Zhu; Heitor R. Guimarães; Arthur Pimentel; Tiago Falk
>
> **摘要:** With the prevalence of artificial intelligence (AI)-generated content, such as audio deepfakes, a large body of recent work has focused on developing deepfake detection techniques. However, most models are evaluated on a narrow set of datasets, leaving their generalization to real-world conditions uncertain. In this paper, we systematically review 28 existing audio deepfake datasets and present an open-source benchmarking toolkit called AUDDT (https://github.com/MuSAELab/AUDDT). The goal of this toolkit is to automate the evaluation of pretrained detectors across these 28 datasets, giving users direct feedback on the advantages and shortcomings of their deepfake detectors. We start by showcasing the usage of the developed toolkit, the composition of our benchmark, and the breakdown of different deepfake subgroups. Next, using a widely adopted pretrained deepfake detector, we present in- and out-of-domain detection results, revealing notable differences across conditions and audio manipulation types. Lastly, we also analyze the limitations of these existing datasets and their gap relative to practical deployment scenarios.
>
---
#### [new 158] AgentPack: A Dataset of Code Changes, Co-Authored by Agents and Humans
- **分类: cs.SE; cs.CL**

- **简介: 该论文提出AgentPack，一个包含1.3M由AI与人类共同编写代码修改的数据集。旨在解决传统代码训练数据噪声大、意图不清晰的问题。通过分析结构特性并展示模型在该数据上优于传统数据的效果，为代码编辑模型训练提供新资源。**

- **链接: [http://arxiv.org/pdf/2509.21891v1](http://arxiv.org/pdf/2509.21891v1)**

> **作者:** Yangtian Zi; Zixuan Wu; Aleksander Boruch-Gruszecki; Jonathan Bell; Arjun Guha
>
> **摘要:** Fine-tuning large language models for code editing has typically relied on mining commits and pull requests. The working hypothesis has been that commit messages describe human intent in natural language, and patches to code describe the changes that implement that intent. However, much of the previously collected data is noisy: commit messages are terse, human-written commits commingle several unrelated edits, and many commits come from simple, rule-based bots. The recent adoption of software engineering agents changes this landscape. Code changes co-authored by humans and agents tend to be more narrowly scoped and focused on clearer goals. Their commit messages, generated by LLMs, articulate intent and rationale in much greater detail. Moreover, when these changes land in public repositories, they are implicitly filtered by humans: maintainers discard low-quality commits to their projects. We present AgentPack, a corpus of 1.3M code edits co-authored by Claude Code, OpenAI Codex, and Cursor Agent across public GitHub projects up to mid-August 2025. We describe the identification and curation pipeline, quantify adoption trends of these agents, and analyze the structural properties of the edits. Finally, we show that models fine-tuned on AgentPack can outperform models trained on prior human-only commit corpora, highlighting the potential of using public data from software engineering agents to train future code-editing models.
>
---
#### [new 159] Leveraging Big Data Frameworks for Spam Detection in Amazon Reviews
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于垃圾评论检测任务，旨在解决亚马逊商品评论中的虚假评论问题。研究利用大数据框架和机器学习方法处理大量评论数据，提取特征并分类垃圾评论，其中逻辑回归模型准确率达90.35%，提升了评论可信度。**

- **链接: [http://arxiv.org/pdf/2509.21579v1](http://arxiv.org/pdf/2509.21579v1)**

> **作者:** Mst Eshita Khatun; Halima Akter; Tasnimul Rehan; Toufiq Ahmed
>
> **备注:** Accepted & presented at THE 16th INTERNATIONAL IEEE CONFERENCE ON COMPUTING, COMMUNICATION AND NETWORKING TECHNOLOGIES (ICCCNT) 2025
>
> **摘要:** In this digital era, online shopping is common practice in our daily lives. Product reviews significantly influence consumer buying behavior and help establish buyer trust. However, the prevalence of fraudulent reviews undermines this trust by potentially misleading consumers and damaging the reputations of the sellers. This research addresses this pressing issue by employing advanced big data analytics and machine learning approaches on a substantial dataset of Amazon product reviews. The primary objective is to detect and classify spam reviews accurately so that it enhances the authenticity of the review. Using a scalable big data framework, we efficiently process and analyze a large scale of review data, extracting key features indicative of fraudulent behavior. Our study illustrates the utility of various machine learning classifiers in detecting spam reviews, with Logistic Regression achieving an accuracy of 90.35%, thus contributing to a more trustworthy and transparent online shopping environment.
>
---
#### [new 160] RISK: A Framework for GUI Agents in E-commerce Risk Management
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出RISK框架，用于构建处理电商风险管理中复杂网页交互的GUI智能体。针对传统方法无法处理多步骤动态内容的问题，RISK包含数据集、基准和强化微调方法，提升了单步和多步任务性能。**

- **链接: [http://arxiv.org/pdf/2509.21982v1](http://arxiv.org/pdf/2509.21982v1)**

> **作者:** Renqi Chen; Zeyin Tao; Jianming Guo; Jingzhe Zhu; Yiheng Peng; Qingqing Sun; Tianyi Zhang; Shuai Chen
>
> **摘要:** E-commerce risk management requires aggregating diverse, deeply embedded web data through multi-step, stateful interactions, which traditional scraping methods and most existing Graphical User Interface (GUI) agents cannot handle. These agents are typically limited to single-step tasks and lack the ability to manage dynamic, interactive content critical for effective risk assessment. To address this challenge, we introduce RISK, a novel framework designed to build and deploy GUI agents for this domain. RISK integrates three components: (1) RISK-Data, a dataset of 8,492 single-step and 2,386 multi-step interaction trajectories, collected through a high-fidelity browser framework and a meticulous data curation process; (2) RISK-Bench, a benchmark with 802 single-step and 320 multi-step trajectories across three difficulty levels for standardized evaluation; and (3) RISK-R1, a R1-style reinforcement fine-tuning framework considering four aspects: (i) Output Format: Updated format reward to enhance output syntactic correctness and task comprehension, (ii) Single-step Level: Stepwise accuracy reward to provide granular feedback during early training stages, (iii) Multi-step Level: Process reweight to emphasize critical later steps in interaction sequences, and (iv) Task Level: Level reweight to focus on tasks of varying difficulty. Experiments show that RISK-R1 outperforms existing baselines, achieving a 6.8% improvement in offline single-step and an 8.8% improvement in offline multi-step. Moreover, it attains a top task success rate of 70.5% in online evaluation. RISK provides a scalable, domain-specific solution for automating complex web interactions, advancing the state of the art in e-commerce risk management.
>
---
#### [new 161] LLMs for Bayesian Optimization in Scientific Domains: Are We There Yet?
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究LLM在科学领域贝叶斯优化中的应用，发现LLM对实验反馈不敏感，性能不如传统方法。提出LLMNN混合方法，结合LLM先验知识与近邻采样，提升实验设计效果。**

- **链接: [http://arxiv.org/pdf/2509.21403v1](http://arxiv.org/pdf/2509.21403v1)**

> **作者:** Rushil Gupta; Jason Hartford; Bang Liu
>
> **备注:** Accepted to EMNLP 2025
>
> **摘要:** Large language models (LLMs) have recently been proposed as general-purpose agents for experimental design, with claims that they can perform in-context experimental design. We evaluate this hypothesis using both open- and closed-source instruction-tuned LLMs applied to genetic perturbation and molecular property discovery tasks. We find that LLM-based agents show no sensitivity to experimental feedback: replacing true outcomes with randomly permuted labels has no impact on performance. Across benchmarks, classical methods such as linear bandits and Gaussian process optimization consistently outperform LLM agents. We further propose a simple hybrid method, LLM-guided Nearest Neighbour (LLMNN) sampling, that combines LLM prior knowledge with nearest-neighbor sampling to guide the design of experiments. LLMNN achieves competitive or superior performance across domains without requiring significant in-context adaptation. These results suggest that current open- and closed-source LLMs do not perform in-context experimental design in practice and highlight the need for hybrid frameworks that decouple prior-based reasoning from batch acquisition with updated posteriors.
>
---
#### [new 162] HetaRAG: Hybrid Deep Retrieval-Augmented Generation across Heterogeneous Data Stores
- **分类: cs.IR; cs.CL**

- **简介: 该论文提出HetaRAG，一种融合异构数据存储（如向量数据库、知识图谱等）的混合深度检索增强生成框架，旨在解决传统RAG系统在单一模态下的精度、召回率和上下文理解不足的问题。**

- **链接: [http://arxiv.org/pdf/2509.21336v1](http://arxiv.org/pdf/2509.21336v1)**

> **作者:** Guohang Yan; Yue Zhang; Pinlong Cai; Ding Wang; Song Mao; Hongwei Zhang; Yaoze Zhang; Hairong Zhang; Xinyu Cai; Botian Shi
>
> **备注:** 15 pages, 4 figures
>
> **摘要:** Retrieval-augmented generation (RAG) has become a dominant paradigm for mitigating knowledge hallucination and staleness in large language models (LLMs) while preserving data security. By retrieving relevant evidence from private, domain-specific corpora and injecting it into carefully engineered prompts, RAG delivers trustworthy responses without the prohibitive cost of fine-tuning. Traditional retrieval-augmented generation (RAG) systems are text-only and often rely on a single storage backend, most commonly a vector database. In practice, this monolithic design suffers from unavoidable trade-offs: vector search captures semantic similarity yet loses global context; knowledge graphs excel at relational precision but struggle with recall; full-text indexes are fast and exact yet semantically blind; and relational engines such as MySQL provide strong transactional guarantees but no semantic understanding. We argue that these heterogeneous retrieval paradigms are complementary, and propose a principled fusion scheme to orchestrate them synergistically, mitigating the weaknesses of any single modality. In this work we introduce HetaRAG, a hybrid, deep-retrieval augmented generation framework that orchestrates cross-modal evidence from heterogeneous data stores. We plan to design a system that unifies vector indices, knowledge graphs, full-text engines, and structured databases into a single retrieval plane, dynamically routing and fusing evidence to maximize recall, precision, and contextual fidelity. To achieve this design goal, we carried out preliminary explorations and constructed an initial RAG pipeline; this technical report provides a brief overview. The partial code is available at https://github.com/KnowledgeXLab/HetaRAG.
>
---
## 更新

#### [replaced 001] CLASH: Evaluating Language Models on Judging High-Stakes Dilemmas from Multiple Perspectives
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.10823v3](http://arxiv.org/pdf/2504.10823v3)**

> **作者:** Ayoung Lee; Ryan Sungmo Kwon; Peter Railton; Lu Wang
>
> **摘要:** Navigating dilemmas involving conflicting values is challenging even for humans in high-stakes domains, let alone for AI, yet prior work has been limited to everyday scenarios. To close this gap, we introduce CLASH (Character perspective-based LLM Assessments in Situations with High-stakes), a meticulously curated dataset consisting of 345 high-impact dilemmas along with 3,795 individual perspectives of diverse values. CLASH enables the study of critical yet underexplored aspects of value-based decision-making processes, including understanding of decision ambivalence and psychological discomfort as well as capturing the temporal shifts of values in the perspectives of characters. By benchmarking 14 non-thinking and thinking models, we uncover several key findings. (1) Even strong proprietary models, such as GPT-5 and Claude-4-Sonnet, struggle with ambivalent decisions, achieving only 24.06 and 51.01 accuracy. (2) Although LLMs reasonably predict psychological discomfort, they do not adequately comprehend perspectives involving value shifts. (3) Cognitive behaviors that are effective in the math-solving and game strategy domains do not transfer to value reasoning. Instead, new failure patterns emerge, including early commitment and overcommitment. (4) The steerability of LLMs towards a given value is significantly correlated with their value preferences. (5) Finally, LLMs exhibit greater steerability when reasoning from a third-party perspective, although certain values (e.g., safety) benefit uniquely from first-person framing.
>
---
#### [replaced 002] Hallucination to Truth: A Review of Fact-Checking and Factuality Evaluation in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.03860v2](http://arxiv.org/pdf/2508.03860v2)**

> **作者:** Subhey Sadi Rahman; Md. Adnanul Islam; Md. Mahbub Alam; Musarrat Zeba; Md. Abdur Rahman; Sadia Sultana Chowa; Mohaimenul Azam Khan Raiaan; Sami Azam
>
> **摘要:** Large Language Models (LLMs) are trained on vast and diverse internet corpora that often include inaccurate or misleading content. Consequently, LLMs can generate misinformation, making robust fact-checking essential. This review systematically analyzes how LLM-generated content is evaluated for factual accuracy by exploring key challenges such as hallucinations, dataset limitations, and the reliability of evaluation metrics. The review emphasizes the need for strong fact-checking frameworks that integrate advanced prompting strategies, domain-specific fine-tuning, and retrieval-augmented generation (RAG) methods. It proposes five research questions that guide the analysis of the recent literature from 2020 to 2025, focusing on evaluation methods and mitigation techniques. Instruction tuning, multi-agent reasoning, and RAG frameworks for external knowledge access are also reviewed. The key findings demonstrate the limitations of current metrics, the importance of validated external evidence, and the improvement of factual consistency through domain-specific customization. The review underscores the importance of building more accurate, understandable, and context-aware fact-checking. These insights contribute to the advancement of research toward more trustworthy models.
>
---
#### [replaced 003] Influence-driven Curriculum Learning for Pre-training on Limited Data
- **分类: cs.CL; cs.LG; I.2.7**

- **链接: [http://arxiv.org/pdf/2508.15475v2](http://arxiv.org/pdf/2508.15475v2)**

> **作者:** Loris Schoenegger; Lukas Thoma; Terra Blevins; Benjamin Roth
>
> **备注:** Added acknowledgments section. 9 pages, Accepted to the BabyLM Workshop at EMNLP 2025
>
> **摘要:** Curriculum learning, a training technique where data is presented to the model in order of example difficulty (e.g., from simpler to more complex documents), has shown limited success for pre-training language models. In this work, we investigate whether curriculum learning becomes competitive if we replace conventional human-centered difficulty metrics with one that more closely corresponds to example difficulty as observed during model training. Specifically, we experiment with sorting training examples by their \textit{training data influence}, a score which estimates the effect of individual training examples on the model's output. Models trained on our curricula are able to outperform ones trained in random order by over 10 percentage points in benchmarks, confirming that curriculum learning is beneficial for language model pre-training, as long as a more model-centric notion of difficulty is adopted.
>
---
#### [replaced 004] Personalized LLM Decoding via Contrasting Personal Preference
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.12109v2](http://arxiv.org/pdf/2506.12109v2)**

> **作者:** Hyungjune Bu; Chanjoo Jung; Minjae Kang; Jaehyung Kim
>
> **摘要:** As large language models (LLMs) are progressively deployed in various real-world applications, personalization of LLMs has become increasingly important. While various approaches to LLM personalization such as prompt-based and training-based methods have been actively explored, the development of effective decoding-time algorithms remains largely overlooked, despite their demonstrated potential. In this paper, we propose CoPe (Contrasting Personal Preference), a novel decoding-time approach applied after performing parameter-efficient fine-tuning (PEFT) on user-specific data. Our core idea is to leverage reward-guided decoding specifically for personalization by maximizing each user's implicit reward signal. We evaluate CoPe across five open-ended personalized text generation tasks. Our empirical results demonstrate that CoPe achieves strong performance, improving personalization by an average of 10.57% in ROUGE-L, without relying on external reward models or additional training procedures.
>
---
#### [replaced 005] Beyond Static Testbeds: An Interaction-Centric Agent Simulation Platform for Dynamic Recommender Systems
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.16429v2](http://arxiv.org/pdf/2505.16429v2)**

> **作者:** Song Jin; Juntian Zhang; Yuhan Liu; Xun Zhang; Yufei Zhang; Guojun Yin; Fei Jiang; Wei Lin; Rui Yan
>
> **备注:** EMNLP2025 Main
>
> **摘要:** Evaluating and iterating upon recommender systems is crucial, yet traditional A/B testing is resource-intensive, and offline methods struggle with dynamic user-platform interactions. While agent-based simulation is promising, existing platforms often lack a mechanism for user actions to dynamically reshape the environment. To bridge this gap, we introduce RecInter, a novel agent-based simulation platform for recommender systems featuring a robust interaction mechanism. In RecInter platform, simulated user actions (e.g., likes, reviews, purchases) dynamically update item attributes in real-time, and introduced Merchant Agents can reply, fostering a more realistic and evolving ecosystem. High-fidelity simulation is ensured through Multidimensional User Profiling module, Advanced Agent Architecture, and LLM fine-tuned on Chain-of-Thought (CoT) enriched interaction data. Our platform achieves significantly improved simulation credibility and successfully replicates emergent phenomena like Brand Loyalty and the Matthew Effect. Experiments demonstrate that this interaction mechanism is pivotal for simulating realistic system evolution, establishing our platform as a credible testbed for recommender systems research. Our codes are available at https://github.com/jinsong8/RecInter.
>
---
#### [replaced 006] HBO: Hierarchical Balancing Optimization for Fine-Tuning Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.12300v2](http://arxiv.org/pdf/2505.12300v2)**

> **作者:** Weixuan Wang; Minghao Wu; Barry Haddow; Alexandra Birch
>
> **摘要:** Fine-tuning large language models (LLMs) on a mixture of diverse datasets poses challenges due to data imbalance and heterogeneity. Existing methods often address these issues across datasets (globally) but overlook the imbalance and heterogeneity within individual datasets (locally), which limits their effectiveness. We introduce Hierarchical Balancing Optimization (HBO), a novel method that enables LLMs to autonomously adjust data allocation during fine-tuning both across datasets (globally) and within each individual dataset (locally). HBO employs a bilevel optimization strategy with two types of actors: a Global Actor, which balances data sampling across different subsets of the training mixture, and several Local Actors, which optimizes data usage within each subset based on difficulty levels. These actors are guided by reward functions derived from the LLM's training state, which measure learning progress and relative performance improvement. We evaluate HBO on three LLM backbones across nine diverse tasks in multilingual and multitask setups. Results show that HBO consistently outperforms existing baselines, achieving significant accuracy gains. Our in-depth analysis further demonstrates that both the global actor and local actors of HBO effectively adjust data usage during fine-tuning. HBO provides a comprehensive solution to the challenges of data imbalance and heterogeneity in LLM fine-tuning, enabling more effective training across diverse datasets.
>
---
#### [replaced 007] Expanding Reasoning Potential in Foundation Model by Learning Diverse Chains of Thought Patterns
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.21124v2](http://arxiv.org/pdf/2509.21124v2)**

> **作者:** Xuemiao Zhang; Can Ren; Chengying Tu; Rongxiang Weng; Shuo Wang; Hongfei Yan; Jingang Wang; Xunliang Cai
>
> **摘要:** Recent progress in large reasoning models for challenging mathematical reasoning has been driven by reinforcement learning (RL). Incorporating long chain-of-thought (CoT) data during mid-training has also been shown to substantially improve reasoning depth. However, current approaches often utilize CoT data indiscriminately, leaving open the critical question of which data types most effectively enhance model reasoning capabilities. In this paper, we define the foundation model's reasoning potential for the first time as the inverse of the number of independent attempts required to correctly answer the question, which is strongly correlated with the final model performance. We then propose utilizing diverse data enriched with high-value reasoning patterns to expand the reasoning potential. Specifically, we abstract atomic reasoning patterns from CoT sequences, characterized by commonality and inductive capabilities, and use them to construct a core reference set enriched with valuable reasoning patterns. Furthermore, we propose a dual-granularity algorithm involving chains of reasoning patterns and token entropy, efficiently selecting high-value CoT data (CoTP) from the data pool that aligns with the core set, thereby training models to master reasoning effectively. Only 10B-token CoTP data enables the 85A6B Mixture-of-Experts (MoE) model to improve by 9.58% on the challenging AIME 2024 and 2025, and to raise the upper bound of downstream RL performance by 7.81%.
>
---
#### [replaced 008] Conflict-Aware Soft Prompting for Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.15253v2](http://arxiv.org/pdf/2508.15253v2)**

> **作者:** Eunseong Choi; June Park; Hyeri Lee; Jongwuk Lee
>
> **备注:** Accepted to EMNLP 2025; 15 pages; 5 figures, 11 tables; Code available at https://github.com/eunseongc/CARE
>
> **摘要:** Retrieval-augmented generation (RAG) enhances the capabilities of large language models (LLMs) by incorporating external knowledge into their input prompts. However, when the retrieved context contradicts the LLM's parametric knowledge, it often fails to resolve the conflict between incorrect external context and correct parametric knowledge, known as context-memory conflict. To tackle this problem, we introduce Conflict-Aware REtrieval-Augmented Generation (CARE), consisting of a context assessor and a base LLM. The context assessor encodes compact memory token embeddings from raw context tokens. Through grounded/adversarial soft prompting, the context assessor is trained to discern unreliable context and capture a guidance signal that directs reasoning toward the more reliable knowledge source. Extensive experiments show that CARE effectively mitigates context-memory conflicts, leading to an average performance gain of 5.0\% on QA and fact-checking benchmarks, establishing a promising direction for trustworthy and adaptive RAG systems.
>
---
#### [replaced 009] Persona-Augmented Benchmarking: Evaluating LLMs Across Diverse Writing Styles
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.22168v2](http://arxiv.org/pdf/2507.22168v2)**

> **作者:** Kimberly Le Truong; Riccardo Fogliato; Hoda Heidari; Zhiwei Steven Wu
>
> **备注:** Accepted to EMNLP 2025
>
> **摘要:** Current benchmarks for evaluating Large Language Models (LLMs) often do not exhibit enough writing style diversity, with many adhering primarily to standardized conventions. Such benchmarks do not fully capture the rich variety of communication patterns exhibited by humans. Thus, it is possible that LLMs, which are optimized on these benchmarks, may demonstrate brittle performance when faced with "non-standard" input. In this work, we test this hypothesis by rewriting evaluation prompts using persona-based LLM prompting, a low-cost method to emulate diverse writing styles. Our results show that, even with identical semantic content, variations in writing style and prompt formatting significantly impact the estimated performance of the LLM under evaluation. Notably, we identify distinct writing styles that consistently trigger either low or high performance across a range of models and tasks, irrespective of model family, size, and recency. Our work offers a scalable approach to augment existing benchmarks, improving the external validity of the assessments they provide for measuring LLM performance across linguistic variations.
>
---
#### [replaced 010] Improving LLM-as-a-Judge Inference with the Judgment Distribution
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.03064v2](http://arxiv.org/pdf/2503.03064v2)**

> **作者:** Victor Wang; Michael J. Q. Zhang; Eunsol Choi
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** Using language models to scalably approximate human preferences on text quality (LLM-as-a-judge) has become a standard practice applicable to many tasks. A judgment is often extracted from the judge's textual output alone, typically with greedy decoding. However, LLM judges naturally provide distributions over judgment tokens, inviting a breadth of inference methods for extracting fine-grained preferences. We find that taking the mean of the judgment distribution consistently outperforms taking the mode (i.e. greedy decoding) in all evaluation settings (i.e. pointwise, pairwise, and listwise). We further explore novel methods of deriving preferences from judgment distributions, and find that methods incorporating risk aversion often improve performance. Lastly, we analyze LLM-as-a-judge paired with chain-of-thought (CoT) prompting, showing that CoT can collapse the spread of the judgment distribution, often harming performance. Our findings show that leveraging distributional output improves LLM-as-a-judge, as opposed to using the text interface alone.
>
---
#### [replaced 011] Rare-to-Frequent: Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with LLM Guidance
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2410.22376v3](http://arxiv.org/pdf/2410.22376v3)**

> **作者:** Dongmin Park; Sebin Kim; Taehong Moon; Minkyu Kim; Kangwook Lee; Jaewoong Cho
>
> **备注:** ICLR 2025 (spotlight)
>
> **摘要:** State-of-the-art text-to-image (T2I) diffusion models often struggle to generate rare compositions of concepts, e.g., objects with unusual attributes. In this paper, we show that the compositional generation power of diffusion models on such rare concepts can be significantly enhanced by the Large Language Model (LLM) guidance. We start with empirical and theoretical analysis, demonstrating that exposing frequent concepts relevant to the target rare concepts during the diffusion sampling process yields more accurate concept composition. Based on this, we propose a training-free approach, R2F, that plans and executes the overall rare-to-frequent concept guidance throughout the diffusion inference by leveraging the abundant semantic knowledge in LLMs. Our framework is flexible across any pre-trained diffusion models and LLMs, and can be seamlessly integrated with the region-guided diffusion approaches. Extensive experiments on three datasets, including our newly proposed benchmark, RareBench, containing various prompts with rare compositions of concepts, R2F significantly surpasses existing models including SD3.0 and FLUX by up to 28.1%p in T2I alignment. Code is available at https://github.com/krafton-ai/Rare-to-Frequent.
>
---
#### [replaced 012] Dream to Chat: Model-based Reinforcement Learning on Dialogues with User Belief Modeling
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.16876v3](http://arxiv.org/pdf/2508.16876v3)**

> **作者:** Yue Zhao; Xiaoyu Wang; Dan Wang; Zhonglin Jiang; Qingqing Gu; Teng Chen; Ningyuan Xi; Jinxian Qu; Yong Chen; Luo Ji
>
> **备注:** Accepted to EMNLP 2025 Findings
>
> **摘要:** World models have been widely utilized in robotics, gaming, and auto-driving. However, their applications on natural language tasks are relatively limited. In this paper, we construct the dialogue world model, which could predict the user's emotion, sentiment, and intention, and future utterances. By defining a POMDP, we argue emotion, sentiment and intention can be modeled as the user belief and solved by maximizing the information bottleneck. By this user belief modeling, we apply the model-based reinforcement learning framework to the dialogue system, and propose a framework called DreamCUB. Experiments show that the pretrained dialogue world model can achieve state-of-the-art performances on emotion classification and sentiment identification, while dialogue quality is also enhanced by joint training of the policy, critic and dialogue world model. Further analysis shows that this manner holds a reasonable exploration-exploitation balance and also transfers well to out-of-domain scenarios such as empathetic dialogues.
>
---
#### [replaced 013] Learn Globally, Speak Locally: Bridging the Gaps in Multilingual Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.05418v2](http://arxiv.org/pdf/2507.05418v2)**

> **作者:** Jaedong Hwang; Kumar Tanmay; Seok-Jin Lee; Ayush Agrawal; Hamid Palangi; Kumar Ayush; Ila Fiete; Paul Pu Liang
>
> **摘要:** Large Language Models (LLMs) have achieved strong performance in domains like mathematics, factual question answering, and code generation, yet their ability to reason on these tasks in different languages remains underdeveloped. Especially for low-resource languages such as Swahili or Thai, LLMs can often misinterpret prompts or default to reasoning in English. This implicit bias toward high-resource languages undermines factual accuracy, interpretability, and trust. We propose M2A, a novel method that combines multi-scale multilingual alignment with language-consistency rewards on machine-translated questions, training models to reason directly and accurately in the target language. Furthermore, existing multilingual benchmarks only evaluate on final answers, overlooking whether reasoning occurs in the intended language. To close this gap, we introduce GeoFact-X, a geography-based multilingual factual reasoning benchmark together with reasoning traces in five languages: English, Hindi, Japanese, Swahili, and Thai. Our results show that M2A significantly enhances multilingual reasoning fidelity in both mathematical and factual reasoning tasks, highlighting that reasoning-aware multilingual reinforcement learning is crucial for robust cross-lingual generalization. https://jd730.github.io/projects/M2A_GeoFact-X
>
---
#### [replaced 014] Towards Inclusive ASR: Investigating Voice Conversion for Dysarthric Speech Recognition in Low-Resource Languages
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.14874v5](http://arxiv.org/pdf/2505.14874v5)**

> **作者:** Chin-Jou Li; Eunjung Yeo; Kwanghee Choi; Paula Andrea Pérez-Toro; Masao Someki; Rohan Kumar Das; Zhengjun Yue; Juan Rafael Orozco-Arroyave; Elmar Nöth; David R. Mortensen
>
> **备注:** 5 pages, 1 figure, Proceedings of Interspeech
>
> **摘要:** Automatic speech recognition (ASR) for dysarthric speech remains challenging due to data scarcity, particularly in non-English languages. To address this, we fine-tune a voice conversion model on English dysarthric speech (UASpeech) to encode both speaker characteristics and prosodic distortions, then apply it to convert healthy non-English speech (FLEURS) into non-English dysarthric-like speech. The generated data is then used to fine-tune a multilingual ASR model, Massively Multilingual Speech (MMS), for improved dysarthric speech recognition. Evaluation on PC-GITA (Spanish), EasyCall (Italian), and SSNCE (Tamil) demonstrates that VC with both speaker and prosody conversion significantly outperforms the off-the-shelf MMS performance and conventional augmentation techniques such as speed and tempo perturbation. Objective and subjective analyses of the generated data further confirm that the generated speech simulates dysarthric characteristics.
>
---
#### [replaced 015] Think With Videos For Agentic Long-Video Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.10821v3](http://arxiv.org/pdf/2506.10821v3)**

> **作者:** Huaying Yuan; Zheng Liu; Junjie Zhou; Hongjin Qian; Yan Shu; Nicu Sebe; Ji-Rong Wen; Zhicheng Dou
>
> **摘要:** Long-video understanding~(LVU) is a challenging problem in computer vision. Existing methods either downsample frames for single-pass reasoning, sacrificing fine-grained details, or depend on textual reasoning over task-agnostic representations, hindering task-specific perception and exploration. In this paper, we propose VideoExplorer, a framework grounded in the principle of ``thinking with video'', which naturally intertwines planning, temporal grounding, and scalable perception into a coherent reasoning process. Rather than reasoning over a static context, VideoExplorer iteratively formulates sub-questions, locates relevant moments, and performs task-oriented, temporally scalable video understanding until reaching the final answer, enabling faithful, efficient, and interpretable reasoning. To address the lack of LVU training resources, we construct a long-video reasoning dataset using difficulty-adaptive sampling to ensure high-quality trajectories on complex tasks. Building on this dataset, we design a two-stage training pipeline: supervised trajectory initialization followed by trajectory-level preference optimization, encouraging adaptive temporal grounding and iterative information integration guided by downstream rewards. Extensive evaluations on popular long-video understanding and reasoning benchmarks demonstrate VideoExplorer's significant advantage over existing baselines, highlighting its robustness, adaptability, and efficiency. Our code is made publicly available in this repository(https://github.com/yhy-2000/VideoDeepResearch).
>
---
#### [replaced 016] VAT-KG: Knowledge-Intensive Multimodal Knowledge Graph Dataset for Retrieval-Augmented Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.21556v3](http://arxiv.org/pdf/2506.21556v3)**

> **作者:** Hyeongcheol Park; Jiyoung Seo; MinHyuk Jang; Hogun Park; Ha Dam Baek; Gyusam Chang; Hyeonsoo Im; Sangpil Kim
>
> **备注:** Project Page: https://vatkg.github.io/
>
> **摘要:** Multimodal Knowledge Graphs (MMKGs), which represent explicit knowledge across multiple modalities, play a pivotal role by complementing the implicit knowledge of Multimodal Large Language Models (MLLMs) and enabling more grounded reasoning via Retrieval Augmented Generation (RAG). However, existing MMKGs are generally limited in scope: they are often constructed by augmenting pre-existing knowledge graphs, which restricts their knowledge, resulting in outdated or incomplete knowledge coverage, and they often support only a narrow range of modalities, such as text and visual information. These limitations restrict applicability to multimodal tasks, particularly as recent MLLMs adopt richer modalities like video and audio. Therefore, we propose the Visual-Audio-Text Knowledge Graph (VAT-KG), the first concept-centric and knowledge-intensive multimodal knowledge graph that covers visual, audio, and text information, where each triplet is linked to multimodal data and enriched with detailed descriptions of concepts. Specifically, our construction pipeline ensures cross-modal knowledge alignment between multimodal data and fine-grained semantics through a series of stringent filtering and alignment steps, enabling the automatic generation of MMKGs from any multimodal dataset. We further introduce a novel multimodal RAG framework that retrieves detailed concept-level knowledge in response to queries from arbitrary modalities. Experiments on question answering tasks across various modalities demonstrate the effectiveness of VAT-KG in supporting MLLMs, highlighting its practical value in unifying and leveraging multimodal knowledge.
>
---
#### [replaced 017] GLEAM: Learning to Match and Explain in Cross-View Geo-Localization
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.07450v2](http://arxiv.org/pdf/2509.07450v2)**

> **作者:** Xudong Lu; Zhi Zheng; Yi Wan; Yongxiang Yao; Annan Wang; Renrui Zhang; Panwang Xia; Qiong Wu; Qingyun Li; Weifeng Lin; Xiangyu Zhao; Peifeng Ma; Xue Yang; Hongsheng Li
>
> **备注:** 18 pages
>
> **摘要:** Cross-View Geo-Localization (CVGL) focuses on identifying correspondences between images captured from distinct perspectives of the same geographical location. However, existing CVGL approaches are typically restricted to a single view or modality, and their direct visual matching strategy lacks interpretability: they only determine whether two images correspond, without explaining the rationale behind the match. In this paper, we present GLEAM-C, a foundational CVGL model that unifies multiple views and modalities-including UAV imagery, street maps, panoramic views, and ground photographs-by aligning them exclusively with satellite imagery. Our framework enhances training efficiency through optimized implementation while achieving accuracy comparable to prior modality-specific CVGL models through a two-phase training strategy. Moreover, to address the lack of interpretability in traditional CVGL methods, we leverage the reasoning capabilities of multimodal large language models (MLLMs) to propose a new task, GLEAM-X, which combines cross-view correspondence prediction with explainable reasoning. To support this task, we construct a bilingual benchmark using GPT-4o and Doubao-1.5-Thinking-Vision-Pro to generate training and testing data. The test set is further refined through detailed human revision, enabling systematic evaluation of explainable cross-view reasoning and advancing transparency and scalability in geo-localization. Together, GLEAM-C and GLEAM-X form a comprehensive CVGL pipeline that integrates multi-modal, multi-view alignment with interpretable correspondence analysis, unifying accurate cross-view matching with explainable reasoning and advancing Geo-Localization by enabling models to better Explain And Match. Code and datasets used in this work will be made publicly accessible at https://github.com/Lucky-Lance/GLEAM.
>
---
#### [replaced 018] Geometric-Mean Policy Optimization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.20673v2](http://arxiv.org/pdf/2507.20673v2)**

> **作者:** Yuzhong Zhao; Yue Liu; Junpeng Liu; Jingye Chen; Xun Wu; Yaru Hao; Tengchao Lv; Shaohan Huang; Lei Cui; Qixiang Ye; Fang Wan; Furu Wei
>
> **备注:** Code is available at https://github.com/callsys/GMPO
>
> **摘要:** Group Relative Policy Optimization (GRPO) has significantly enhanced the reasoning capability of large language models by optimizing the arithmetic mean of token-level rewards. Unfortunately, GRPO is observed to suffer from unstable policy updates when facing tokens with outlier importance-weighted rewards, which manifest as extreme importance sampling ratios during training. In this study, we propose Geometric-Mean Policy Optimization (GMPO), with the aim to improve the stability of GRPO through suppressing token reward outliers. Instead of optimizing the arithmetic mean, GMPO maximizes the geometric mean of token-level rewards, which is inherently less sensitive to outliers and maintains a more stable range of importance sampling ratio. GMPO is plug-and-play-simply replacing GRPO's arithmetic mean with the geometric mean of token-level rewards, as the latter is inherently less sensitive to outliers. GMPO is theoretically plausible-analysis reveals that both GMPO and GRPO are weighted forms of the policy gradient while the former enjoys more stable weights, which consequently benefits policy optimization and performance. Experiments on multiple mathematical reasoning benchmarks show that GMPO-7B improves the average Pass@1 of GRPO by up to 4.1%, outperforming many state-of-the-art approaches. Code is available at https://github.com/callsys/GMPO.
>
---
#### [replaced 019] RPG: A Repository Planning Graph for Unified and Scalable Codebase Generation
- **分类: cs.CL; cs.AI; cs.SE**

- **链接: [http://arxiv.org/pdf/2509.16198v3](http://arxiv.org/pdf/2509.16198v3)**

> **作者:** Jane Luo; Xin Zhang; Steven Liu; Jie Wu; Yiming Huang; Yangyu Huang; Chengyu Yin; Ying Xin; Jianfeng Liu; Yuefeng Zhan; Hao Sun; Qi Chen; Scarlett Li; Mao Yang
>
> **摘要:** Large language models excel at generating individual functions or single files of code, yet generating complete repositories from scratch remains a fundamental challenge. This capability is key to building coherent software systems from high-level specifications and realizing the full potential of automated code generation. The process requires planning at two levels: deciding what features and modules to build (proposal stage) and defining their implementation details (implementation stage). Current approaches rely on natural language planning, which often produces unclear specifications, misaligned components, and brittle designs due to its inherent ambiguity and lack of structure. To address these limitations, we introduce the Repository Planning Graph (RPG), a structured representation that encodes capabilities, file structures, data flows, and functions in a unified graph. By replacing free-form natural language with an explicit blueprint, RPG enables consistent long-horizon planning for repository generation. Building on RPG, we develop ZeroRepo, a graph-driven framework that operates in three stages: proposal-level planning, implementation-level construction, and graph-guided code generation with test validation. To evaluate, we construct RepoCraft, a benchmark of six real-world projects with 1,052 tasks. On RepoCraft, ZeroRepo produces nearly 36K Code Lines and 445K Code Tokens, on average 3.9$\times$ larger than the strongest baseline (Claude Code), and 68$\times$ larger than other baselines. It achieves 81.5% coverage and 69.7% test accuracy, improving over Claude Code by 27.3 and 35.8 points. Further analysis shows that RPG models complex dependencies, enables more sophisticated planning through near-linear scaling, and improves agent understanding of repositories, thus accelerating localization.
>
---
#### [replaced 020] PilotRL: Training Language Model Agents via Global Planning-Guided Progressive Reinforcement Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.00344v3](http://arxiv.org/pdf/2508.00344v3)**

> **作者:** Keer Lu; Chong Chen; Bin Cui; Huang Leng; Wentao Zhang
>
> **摘要:** Large Language Models (LLMs) have shown remarkable advancements in tackling agent-oriented tasks. Despite their potential, existing work faces challenges when deploying LLMs in agent-based environments. The widely adopted agent paradigm ReAct centers on integrating single-step reasoning with immediate action execution, which limits its effectiveness in complex tasks requiring long-term strategic planning. Furthermore, the coordination between the planner and executor during problem-solving is also a critical factor to consider in agent design. Additionally, current approaches predominantly rely on supervised fine-tuning, which often leads models to memorize established task completion trajectories, thereby restricting their generalization ability when confronted with novel problem contexts. To address these challenges, we introduce an adaptive global plan-based agent paradigm AdaPlan, aiming to synergize high-level explicit guidance with execution to support effective long-horizon decision-making. Based on the proposed paradigm, we further put forward PilotRL, a global planning-guided training framework for LLM agents driven by progressive reinforcement learning. We first develop the model's ability to follow explicit guidance from global plans when addressing agent tasks. Subsequently, based on this foundation, we focus on optimizing the quality of generated plans. Finally, we conduct joint optimization of the model's planning and execution coordination. Experiments indicate that PilotRL could achieve state-of-the-art performances, with LLaMA3.1-8B-Instruct + PilotRL surpassing closed-sourced GPT-4o by 3.60%, while showing a more substantial gain of 55.78% comparing to GPT-4o-mini at a comparable parameter scale.
>
---
#### [replaced 021] Large Language Models versus Classical Machine Learning: Performance in COVID-19 Mortality Prediction Using High-Dimensional Tabular Data
- **分类: cs.LG; cs.AI; cs.CL; 92C50, 68T50; J.3**

- **链接: [http://arxiv.org/pdf/2409.02136v2](http://arxiv.org/pdf/2409.02136v2)**

> **作者:** Mohammadreza Ghaffarzadeh-Esfahani; Mahdi Ghaffarzadeh-Esfahani; Arian Salahi-Niri; Hossein Toreyhi; Zahra Atf; Amirali Mohsenzadeh-Kermani; Mahshad Sarikhani; Zohreh Tajabadi; Fatemeh Shojaeian; Mohammad Hassan Bagheri; Aydin Feyzi; Mohammadamin Tarighatpayma; Narges Gazmeh; Fateme Heydari; Hossein Afshar; Amirreza Allahgholipour; Farid Alimardani; Ameneh Salehi; Naghmeh Asadimanesh; Mohammad Amin Khalafi; Hadis Shabanipour; Ali Moradi; Sajjad Hossein Zadeh; Omid Yazdani; Romina Esbati; Moozhan Maleki; Danial Samiei Nasr; Amirali Soheili; Hossein Majlesi; Saba Shahsavan; Alireza Soheilipour; Nooshin Goudarzi; Erfan Taherifard; Hamidreza Hatamabadi; Jamil S Samaan; Thomas Savage; Ankit Sakhuja; Ali Soroush; Girish Nadkarni; Ilad Alavi Darazam; Mohamad Amin Pourhoseingholi; Seyed Amir Ahmad Safavi-Naini
>
> **备注:** Code is available at: https://github.com/mohammad-gh009/Large-Language-Models-vs-Classical-Machine-learning and https://github.com/Sdamirsa/Tehran_COVID_Cohort. The datasets are available from the corresponding author on reasonable request (sdamirsa@ymail.com)
>
> **摘要:** This study compared the performance of classical feature-based machine learning models (CMLs) and large language models (LLMs) in predicting COVID-19 mortality using high-dimensional tabular data from 9,134 patients across four hospitals. Seven CML models, including XGBoost and random forest (RF), were evaluated alongside eight LLMs, such as GPT-4 and Mistral-7b, which performed zero-shot classification on text-converted structured data. Additionally, Mistral- 7b was fine-tuned using the QLoRA approach. XGBoost and RF demonstrated superior performance among CMLs, achieving F1 scores of 0.87 and 0.83 for internal and external validation, respectively. GPT-4 led the LLM category with an F1 score of 0.43, while fine-tuning Mistral-7b significantly improved its recall from 1% to 79%, yielding a stable F1 score of 0.74 during external validation. Although LLMs showed moderate performance in zero-shot classification, fine-tuning substantially enhanced their effectiveness, potentially bridging the gap with CML models. However, CMLs still outperformed LLMs in handling high-dimensional tabular data tasks. This study highlights the potential of both CMLs and fine-tuned LLMs in medical predictive modeling, while emphasizing the current superiority of CMLs for structured data analysis.
>
---
#### [replaced 022] Demystifying Domain-adaptive Post-training for Financial LLMs
- **分类: cs.CL; cs.AI; cs.CE; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.04961v3](http://arxiv.org/pdf/2501.04961v3)**

> **作者:** Zixuan Ke; Yifei Ming; Xuan-Phi Nguyen; Caiming Xiong; Shafiq Joty
>
> **备注:** EMNLP 2025 (Oral)
>
> **摘要:** Domain-adaptive post-training of large language models (LLMs) has emerged as a promising approach for specialized domains such as medicine and finance. However, significant challenges remain in identifying optimal adaptation criteria and training strategies across varying data and model configurations. To address these challenges, we introduce FINDAP, a systematic and fine-grained investigation into domain-adaptive post-training of LLMs for the finance domain. Our approach consists of four key components: FinCap, which defines the core capabilities required for the target domain; FinRec, an effective training recipe that jointly optimizes continual pre-training and instruction-following, along with a novel preference data distillation method leveraging process signals from a generative reward model; FinTrain, a curated set of training datasets supporting FinRec; and FinEval, a comprehensive evaluation suite aligned with FinCap. The resulting model, Llama-Fin, achieves state-of-the-art performance across a wide range of financial tasks. Our analysis also highlights how each post-training stage contributes to distinct capabilities, uncovering specific challenges and effective solutions, providing valuable insights for domain adaptation of LLMs
>
---
#### [replaced 023] From Tokens to Thoughts: How LLMs and Humans Trade Compression for Meaning
- **分类: cs.CL; cs.AI; cs.IT; math.IT**

- **链接: [http://arxiv.org/pdf/2505.17117v5](http://arxiv.org/pdf/2505.17117v5)**

> **作者:** Chen Shani; Liron Soffer; Dan Jurafsky; Yann LeCun; Ravid Shwartz-Ziv
>
> **摘要:** Humans organize knowledge into compact categories that balance compression with semantic meaning preservation. Large Language Models (LLMs) demonstrate striking linguistic abilities, yet whether they achieve this same balance remains unclear. We apply the Information Bottleneck principle to quantitatively compare how LLMs and humans navigate this compression-meaning trade-off. Analyzing embeddings from 40+ LLMs against classic human categorization benchmarks, we uncover three key findings. First, LLMs broadly align with human categories but miss fine-grained semantic distinctions crucial for human understanding. Second, LLMs demonstrate aggressive statistical compression, achieving ``optimal'' information-theoretic efficiency, while humans prioritize contextual richness and adaptive flexibility. Third, encoder models surprisingly outperform decoder models in human alignment, suggesting that generation and understanding rely on distinct mechanisms in current architectures. In addition, training dynamics analysis reveals that conceptual structure develops in distinct phases: rapid initial formation followed by architectural reorganization, with semantic processing migrating from deeper to mid-network layers as models discover more efficient encoding. These divergent strategies, where LLMs optimize for compression and humans for adaptive utility, reveal fundamental differences between artificial and biological intelligence, guiding development toward more human-aligned AI.
>
---
#### [replaced 024] DivLogicEval: A Framework for Benchmarking Logical Reasoning Evaluation in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.15587v3](http://arxiv.org/pdf/2509.15587v3)**

> **作者:** Tsz Ting Chung; Lemao Liu; Mo Yu; Dit-Yan Yeung
>
> **备注:** Accepted by EMNLP 2025. Project Page: https://ttchungc.github.io/projects/divlogiceval/
>
> **摘要:** Logic reasoning in natural language has been recognized as an important measure of human intelligence for Large Language Models (LLMs). Popular benchmarks may entangle multiple reasoning skills and thus provide unfaithful evaluations on the logic reasoning skill. Meanwhile, existing logic reasoning benchmarks are limited in language diversity and their distributions are deviated from the distribution of an ideal logic reasoning benchmark, which may lead to biased evaluation results. This paper thereby proposes a new classical logic benchmark DivLogicEval, consisting of natural sentences composed of diverse statements in a counterintuitive way. To ensure a more reliable evaluation, we also introduce a new evaluation metric that mitigates the influence of bias and randomness inherent in LLMs. Through experiments, we demonstrate the extent to which logical reasoning is required to answer the questions in DivLogicEval and compare the performance of different popular LLMs in conducting logical reasoning.
>
---
#### [replaced 025] InftyThink: Breaking the Length Limits of Long-Context Reasoning in Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.06692v4](http://arxiv.org/pdf/2503.06692v4)**

> **作者:** Yuchen Yan; Yongliang Shen; Yang Liu; Jin Jiang; Mengdi Zhang; Jian Shao; Yueting Zhuang
>
> **备注:** Project Page: https://zju-real.github.io/InftyThink Code: https://github.com/ZJU-REAL/InftyThink Dataset: https://huggingface.co/datasets/ZJU-REAL/InftyThink
>
> **摘要:** Advanced reasoning in large language models has achieved remarkable performance on challenging tasks, but the prevailing long-context reasoning paradigm faces critical limitations: quadratic computational scaling with sequence length, reasoning constrained by maximum context boundaries, and performance degradation beyond pre-training context windows. Existing approaches primarily compress reasoning chains without addressing the fundamental scaling problem. To overcome these challenges, we introduce InftyThink, a paradigm that transforms monolithic reasoning into an iterative process with intermediate summarization. By interleaving short reasoning segments with concise progress summaries, our approach enables unbounded reasoning depth while maintaining bounded computational costs. This creates a characteristic sawtooth memory pattern that significantly reduces computational complexity compared to traditional approaches. Furthermore, we develop a methodology for reconstructing long-context reasoning datasets into our iterative format, transforming OpenR1-Math into 333K training instances. Experiments across multiple model architectures demonstrate that our approach reduces computational costs while improving performance, with Qwen2.5-Math-7B showing 3-13% improvements across MATH500, AIME24, and GPQA_diamond benchmarks. Our work challenges the assumed trade-off between reasoning depth and computational efficiency, providing a more scalable approach to complex reasoning without architectural modifications.
>
---
#### [replaced 026] SuperCoder: Assembly Program Superoptimization with Large Language Models
- **分类: cs.CL; cs.AI; cs.PF; cs.PL; cs.SE**

- **链接: [http://arxiv.org/pdf/2505.11480v2](http://arxiv.org/pdf/2505.11480v2)**

> **作者:** Anjiang Wei; Tarun Suresh; Huanmi Tan; Yinglun Xu; Gagandeep Singh; Ke Wang; Alex Aiken
>
> **摘要:** Superoptimization is the task of transforming a program into a faster one while preserving its input-output behavior. In this work, we investigate whether large language models (LLMs) can serve as superoptimizers, generating assembly programs that outperform code already optimized by industry-standard compilers. We construct the first large-scale benchmark for this problem, consisting of 8,072 real-world assembly programs averaging 130 lines, in contrast to prior datasets restricted to 2-15 straight-line, loop-free programs. We evaluate 23 LLMs on this benchmark and find that the strongest baseline, Claude-opus-4, achieves a 51.5% test-passing rate and a 1.43x average speedup over gcc -O3. To further enhance performance, we fine-tune models with reinforcement learning, optimizing a reward function that integrates correctness and performance speedup. Starting from Qwen2.5-Coder-7B-Instruct (61.4% correctness, 1.10x speedup), the fine-tuned model SuperCoder attains 95.0% correctness and 1.46x average speedup. Our results demonstrate, for the first time, that LLMs can be applied as superoptimizers for assembly programs, establishing a foundation for future research in program performance optimization beyond compiler heuristics.
>
---
#### [replaced 027] Retrieval-Augmented Generation with Hierarchical Knowledge
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.10150v3](http://arxiv.org/pdf/2503.10150v3)**

> **作者:** Haoyu Huang; Yongfeng Huang; Junjie Yang; Zhenyu Pan; Yongqiang Chen; Kaili Ma; Hongzhi Chen; James Cheng
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** Graph-based Retrieval-Augmented Generation (RAG) methods have significantly enhanced the performance of large language models (LLMs) in domain-specific tasks. However, existing RAG methods do not adequately utilize the naturally inherent hierarchical knowledge in human cognition, which limits the capabilities of RAG systems. In this paper, we introduce a new RAG approach, called HiRAG, which utilizes hierarchical knowledge to enhance the semantic understanding and structure capturing capabilities of RAG systems in the indexing and retrieval processes. Our extensive experiments demonstrate that HiRAG achieves significant performance improvements over the state-of-the-art baseline methods.
>
---
#### [replaced 028] QWHA: Quantization-Aware Walsh-Hadamard Adaptation for Parameter-Efficient Fine-Tuning on Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.17428v3](http://arxiv.org/pdf/2509.17428v3)**

> **作者:** Hyesung Jeon; Seojune Lee; Beomseok Kang; Yulhwa Kim; Jae-Joon Kim
>
> **备注:** 25 pages, 9 figures, 14 tables
>
> **摘要:** The demand for efficient deployment of large language models (LLMs) has driven interest in quantization, which reduces inference cost, and parameter-efficient fine-tuning (PEFT), which lowers training overhead. This motivated the development of quantization-aware PEFT to produce accurate yet efficient quantized models. In this setting, reducing quantization error prior to fine-tuning is crucial for achieving high model accuracy. However, existing methods that rely on low-rank adaptation suffer from limited representational capacity. Recent Fourier-related transform (FT)-based adapters offer greater representational power than low-rank adapters, but their direct integration into quantized models often results in ineffective error reduction and increased computational overhead. To overcome these limitations, we propose QWHA, a method that integrates FT-based adapters into quantized models by employing the Walsh-Hadamard Transform (WHT) as the transform kernel, together with a novel adapter initialization scheme incorporating adaptive parameter selection and value refinement. We demonstrate that QWHA effectively mitigates quantization errors while facilitating fine-tuning, and that its design substantially reduces computational cost. Experimental results show that QWHA consistently outperforms baselines in low-bit quantization accuracy and achieves significant training speedups over existing FT-based adapters. The code is available at https://github.com/vantaa89/qwha.
>
---
#### [replaced 029] Why Reinforcement Fine-Tuning Enables MLLMs Preserve Prior Knowledge Better: A Data Perspective
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.23508v2](http://arxiv.org/pdf/2506.23508v2)**

> **作者:** Zhihao Zhang; Qiaole Dong; Qi Zhang; Jun Zhao; Enyu Zhou; Zhiheng Xi; Senjie Jin; Xiaoran Fan; Yuhao Zhou; Mingqi Wu; Yanwei Fu; Tao Ji; Tao Gui; Xuanjing Huang; Kai Chen
>
> **备注:** 20 pages (Preprint.)
>
> **摘要:** Post-training algorithms such as Supervised Fine-Tuning (SFT) and Reinforcement Fine-Tuning (RFT) are widely used to adapt multimodal large language models to downstream tasks. While effective at task adaptation, their impact on prior knowledge remains unclear. In this paper, we introduce jigsaw puzzles as a novel task absent from existing pretraining corpora and systematically study the behavior of SFT and RFT on open-source multimodal model, Qwen2.5-VL series. Our experiments reveal a sharp trade-off: SFT enables rapid task acquisition but leads to catastrophic forgetting, whereas RFT learns more slowly but maintains prior knowledge. We study this phenomenon through learning dynamics by examining both the magnitude and direction of how training data influence prior knowledge. Our analysis shows that RFT mainly reinforces correct samples naturally aligned with the base model's probability landscape, leading to weaker interference with prior knowledge. Moreover, training on RFT-simulated rollouts, which exert a small magnitude of influence and are well aligned in direction to prior knowledge, allows SFT to preserve prior knowledge better while rapidly learning new tasks. These findings suggest that distribution of training data, rather than algorithmic differences, plays a central role in forgetting, and highlight RFT's potential for stable continual learning in multimodal large language models.
>
---
#### [replaced 030] Probing Neural Topology of Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.01042v2](http://arxiv.org/pdf/2506.01042v2)**

> **作者:** Yu Zheng; Yuan Yuan; Yue Zhuo; Yong Li; Paolo Santi
>
> **摘要:** Probing large language models (LLMs) has yielded valuable insights into their internal mechanisms by linking neural activations to interpretable semantics. However, the complex mechanisms that link neuron's functional co-activation with the emergent model capabilities remains largely unknown, hindering a deeper understanding and safer development of LLMs. In this work, we introduce graph probing, a method for uncovering the functional connectivity of LLM neurons and relating it to language generation performance. By probing models across diverse LLM families and scales, we discover a universal predictability of next-token prediction performance using only neural topology, which persists even when retaining just 1% of neuron connections. Strikingly, probing on topology outperforms probing on activation by up to 130.4%, suggesting that neural topology contains orders of richer information of LLM performance than neural activation, which can be easily extracted with simple linear or MLP probes. To explain the dependence between neural topology and language performance, we identify default networks and hub neurons in LLMs and provide causal evidence by interventional experiments on multiple benchmarks, showing that LLMs actually exploit these topological information. Further analyses suggest that neural topology can be effectively leveraged to improve the efficiency, reliability, and safety of LLMs through proof-of-concept applications in model pruning, hallucination detection, and LLM fingerprinting. Codes and data for the graph probing toolbox are available at https://github.com/DavyMorgan/llm-graph-probing.
>
---
#### [replaced 031] Sparse but Wrong: Incorrect L0 Leads to Incorrect Features in Sparse Autoencoders
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.16560v2](http://arxiv.org/pdf/2508.16560v2)**

> **作者:** David Chanin; Adrià Garriga-Alonso
>
> **摘要:** Sparse Autoencoders (SAEs) extract features from LLM internal activations, meant to correspond to interpretable concepts. A core SAE training hyperparameter is L0: how many SAE features should fire per token on average. Existing work compares SAE algorithms using sparsity-reconstruction tradeoff plots, implying L0 is a free parameter with no single correct value aside from its effect on reconstruction. In this work we study the effect of L0 on SAEs, and show that if L0 is not set correctly, the SAE fails to disentangle the underlying features of the LLM. If L0 is too low, the SAE will mix correlated features to improve reconstruction. If L0 is too high, the SAE finds degenerate solutions that also mix features. Further, we present a proxy metric that can help guide the search for the correct L0 for an SAE on a given training distribution. We show that our method finds the correct L0 in toy models and coincides with peak sparse probing performance in LLM SAEs. We find that most commonly used SAEs have an L0 that is too low. Our work shows that L0 must be set correctly to train SAEs with correct features.
>
---
#### [replaced 032] SOLAR: Towards Characterizing Subjectivity of Individuals through Modeling Value Conflicts and Trade-offs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.12633v2](http://arxiv.org/pdf/2504.12633v2)**

> **作者:** Younghun Lee; Dan Goldwasser
>
> **备注:** Accepted to the Main Conference at EMNLP 2025. 9 pages
>
> **摘要:** Large Language Models (LLMs) not only have solved complex reasoning problems but also exhibit remarkable performance in tasks that require subjective decision making. Existing studies suggest that LLM generations can be subjectively grounded to some extent, yet exploring whether LLMs can account for individual-level subjectivity has not been sufficiently studied. In this paper, we characterize subjectivity of individuals on social media and infer their moral judgments using LLMs. We propose a framework, SOLAR (Subjective Ground with Value Abstraction), that observes value conflicts and trade-offs in the user-generated texts to better represent subjective ground of individuals. Empirical results show that our framework improves overall inference results as well as performance on controversial situations. Additionally, we qualitatively show that SOLAR provides explanations about individuals' value preferences, which can further account for their judgments.
>
---
#### [replaced 033] The Invisible Leash: Why RLVR May or May Not Escape Its Origin
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.14843v2](http://arxiv.org/pdf/2507.14843v2)**

> **作者:** Fang Wu; Weihao Xuan; Ximing Lu; Mingjie Liu; Yi Dong; Zaid Harchaoui; Yejin Choi
>
> **摘要:** Recent advances in LLMs highlight RLVR as a promising method for enhancing AI's capabilities, particularly in solving complex logical tasks. However, it remains unclear whether the current practice of RLVR truly expands a model's reasoning boundary or mainly amplifies high-reward outputs that the base model already knows for improved precision. This study presents an empirical investigation that provides fresh insights into the potential limits of the common practice of RLVR. We examine how, under current training conditions, RLVR can operate as a support-constrained optimization mechanism that may restrict the discovery of entirely original solutions, remaining constrained by the base model's initial distribution. We also identify an entropy-reward trade-off: while the current RLVR recipe reliably enhances precision, it may progressively narrow exploration and potentially overlook correct yet underrepresented solutions. Extensive empirical experiments validate that while the current RLVR recipe consistently improves pass@1, the shrinkage of empirical support generally outweighs the expansion of empirical support under larger sampling budgets, failing to recover correct answers that were previously accessible to the base model. Interestingly, we also observe that while RLVR sometimes increases token-level entropy - resulting in greater uncertainty at each generation step - answer-level entropy declines, indicating that these seemingly more uncertain paths ultimately converge onto a smaller set of distinct answers. Taken together, these findings reveal potential limits of the current RLVR recipe in extending reasoning horizons. Breaking this invisible leash may require future algorithmic innovations such as explicit exploration mechanisms or hybrid strategies that seed probability mass into underrepresented solution regions.
>
---
#### [replaced 034] TEXT2AFFORD: Probing Object Affordance Prediction abilities of Language Models solely from Text
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2402.12881v3](http://arxiv.org/pdf/2402.12881v3)**

> **作者:** Sayantan Adak; Daivik Agrawal; Animesh Mukherjee; Somak Aditya
>
> **备注:** Accepted at Conference on Computational Natural Language Learning 2024
>
> **摘要:** We investigate the knowledge of object affordances in pre-trained language models (LMs) and pre-trained Vision-Language models (VLMs). A growing body of literature shows that PTLMs fail inconsistently and non-intuitively, demonstrating a lack of reasoning and grounding. To take a first step toward quantifying the effect of grounding (or lack thereof), we curate a novel and comprehensive dataset of object affordances -- Text2Afford, characterized by 15 affordance classes. Unlike affordance datasets collected in vision and language domains, we annotate in-the-wild sentences with objects and affordances. Experimental results reveal that PTLMs exhibit limited reasoning abilities when it comes to uncommon object affordances. We also observe that pre-trained VLMs do not necessarily capture object affordances effectively. Through few-shot fine-tuning, we demonstrate improvement in affordance knowledge in PTLMs and VLMs. Our research contributes a novel dataset for language grounding tasks, and presents insights into LM capabilities, advancing the understanding of object affordances. Codes and data are available at https://github.com/sayantan11995/Text2Afford
>
---
#### [replaced 035] EigenBench: A Comparative Behavioral Measure of Value Alignment
- **分类: cs.AI; cs.CL; cs.CY; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.01938v3](http://arxiv.org/pdf/2509.01938v3)**

> **作者:** Jonathn Chang; Leonhard Piff; Suvadip Sana; Jasmine X. Li; Lionel Levine
>
> **摘要:** Aligning AI with human values is a pressing unsolved problem. To address the lack of quantitative metrics for value alignment, we propose EigenBench: a black-box method for comparatively benchmarking language models' values. Given an ensemble of models, a constitution describing a value system, and a dataset of scenarios, our method returns a vector of scores quantifying each model's alignment to the given constitution. To produce these scores, each model judges the outputs of other models across many scenarios, and these judgments are aggregated with EigenTrust (Kamvar et al., 2003), yielding scores that reflect a weighted consensus judgment of the whole ensemble. EigenBench uses no ground truth labels, as it is designed to quantify subjective traits for which reasonable judges may disagree on the correct label. Hence, to validate our method, we collect human judgments on the same ensemble of models and show that EigenBench's judgments align closely with those of human evaluators. We further demonstrate that EigenBench can recover model rankings on the GPQA benchmark without access to objective labels, supporting its viability as a framework for evaluating subjective values for which no ground truths exist.
>
---
#### [replaced 036] R-Stitch: Dynamic Trajectory Stitching for Efficient Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.17307v4](http://arxiv.org/pdf/2507.17307v4)**

> **作者:** Zhuokun Chen; Zeren Chen; Jiahao He; Lu Sheng; Mingkui Tan; Jianfei Cai; Bohan Zhuang
>
> **摘要:** Chain-of-thought (CoT) enhances the problem-solving ability of large language models (LLMs) but incurs substantial inference cost due to long autoregressive trajectories. Existing acceleration strategies either shorten traces via early stopping or compression, or adopt speculative decoding with a smaller model. However, speculative decoding provides limited gains when model agreement is low and rigidly enforces token-level consistency, overlooking the observation that some smaller models, when correct, produce significantly more concise reasoning traces that could reduce inference length. We introduce R-Stitch, a training-free hybrid decoding framework that leverages token-level entropy as an uncertainty proxy to delegate computation between a small language model (SLM) and an LLM. Our analysis shows that high-entropy tokens are more likely to induce errors, motivating an entropy-guided routing strategy that lets the SLM efficiently handle low-entropy tokens while delegating uncertain ones to the LLM, thereby avoiding full rollbacks and preserving answer quality. We further extend this design with R-Stitch$^{+}$, which learns an adaptive routing policy to adjust the token budget dynamically beyond fixed thresholds. By jointly reducing per-token decoding complexity and the number of generated tokens, our method achieves substantial acceleration with negligible accuracy loss. Concretely, it attains peak speedups of 3.00$\times$ on DeepSeek-R1-Distill-Qwen-7B, 3.85$\times$ on 14B, and 4.10$\times$ on QWQ-32B while maintaining accuracy comparable to full LLM decoding. Moreover, it naturally enables adaptive efficiency--accuracy trade-offs that can be tailored to diverse computational budgets without retraining.
>
---
#### [replaced 037] JudgeAgent: Knowledge-wise and Dynamic LLM Evaluation with Agent-as-Interviewer
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.02097v3](http://arxiv.org/pdf/2509.02097v3)**

> **作者:** Zhichao Shi; Xuhui Jiang; Chengjin Xu; Cangli Yao; Zhenxin Huang; Shengjie Ma; Yinghan Shen; Jian Guo; Yuanzhuo Wang
>
> **摘要:** Current evaluation paradigms for large language models (LLMs) suffer from overestimated or biased evaluations and mismatched question difficulty, leading to incomplete evaluations of knowledge and capability boundaries, which hinder their effective application and optimization. To address these challenges, we propose Agent-as-Interviewer, a dynamic evaluation paradigm that employs LLM agents to conduct multi-turn interactions for evaluation. Unlike current benchmarking or dynamic interaction paradigms, Agent-as-Interviewer utilizes agents to invoke knowledge tools for wider and deeper knowledge in the dynamic multi-turn question generation, achieving more comprehensive evaluations of LLM's knowledge boundaries. It also leverages agents to plan query strategies for adjustment of the question difficulty levels, enhancing the difficulty control to match the actual capabilities of target LLMs. Based on this paradigm, we develop JudgeAgent, a knowledge-wise dynamic evaluation framework that employs knowledge-driven synthesis as the agent's tool and uses difficulty scoring as strategy guidance, thereby finally providing valuable suggestions to help targets optimize themselves. Extensive experiments validate the effectiveness of JudgeAgent's suggestions, demonstrating that Agent-as-Interviewer can accurately identify the knowledge and capability boundaries of target models. The source code is available on https://github.com/DataArcTech/JudgeAgent.
>
---
#### [replaced 038] Detecting and Interpreting NSFW Prompts in Text-to-Image Models through Uncovering Harmful Semantics
- **分类: cs.CR; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.18123v2](http://arxiv.org/pdf/2412.18123v2)**

> **作者:** Yiming Wang; Jiahao Chen; Qingming Li; Tong Zhang; Rui Zeng; Xing Yang; Shouling Ji
>
> **摘要:** As text-to-image (T2I) models advance and gain widespread adoption, their associated safety concerns are becoming increasingly critical. Malicious users exploit these models to generate Not-Safe-for-Work (NSFW) images using harmful or adversarial prompts, underscoring the need for effective safeguards to ensure the integrity and compliance of model outputs. However, existing detection methods often exhibit low accuracy and inefficiency. In this paper, we propose HiddenGuard, an interpretable defense framework leveraging the hidden states of T2I models to detect NSFW prompts. HiddenGuard extracts NSFW features from the hidden states of the model's text encoder, utilizing the separable nature of these features to detect NSFW prompts. The detection process is efficient, requiring minimal inference time. HiddenGuard also offers real-time interpretation of results and supports optimization through data augmentation techniques. Our extensive experiments show that HiddenGuard significantly outperforms both commercial and open-source moderation tools, achieving over 95\% accuracy across all datasets and greatly improves computational efficiency.
>
---
#### [replaced 039] LoRA-MGPO: Mitigating Double Descent in Low-Rank Adaptation via Momentum-Guided Perturbation Optimization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.14538v3](http://arxiv.org/pdf/2502.14538v3)**

> **作者:** Yupeng Chang; Chenlu Guo; Yi Chang; Yuan Wu
>
> **摘要:** Parameter-efficient fine-tuning (PEFT), particularly Low-Rank Adaptation (LoRA), adapts large language models (LLMs) by training only a small fraction of parameters. However, as the rank of the low-rank matrices used for adaptation increases, LoRA often exhibits an unstable "double descent" phenomenon, characterized by transient divergence in the training loss, which delays convergence and impairs generalization by causing instability due to the attraction to sharp local minima. To address this, we introduce LoRA-MGPO, a framework that incorporates Momentum-Guided Perturbation Optimization (MGPO). MGPO stabilizes training dynamics by mitigating the double descent phenomenon and guiding weight perturbations using momentum vectors from the optimizer's state, thus avoiding dual gradient computations. Additionally, an adaptive normalization scheme scales the magnitude of perturbations based on an exponential moving average (EMA) of gradient norms, further enhancing stability. While EMA controls the magnitude of the perturbations, MGPO guides their direction, ensuring a more stable optimization trajectory. Experiments on a suite of natural language understanding and generation benchmarks show that LoRA-MGPO consistently achieves superior performance over LoRA and other PEFT methods. The analysis indicates that LoRA-MGPO leads to smoother loss curves, faster convergence, and improved generalization by stabilizing the training process and mitigating the attraction to sharp minima.
>
---
#### [replaced 040] HiCoLoRA: Addressing Context-Prompt Misalignment via Hierarchical Collaborative LoRA for Zero-Shot DST
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2509.19742v2](http://arxiv.org/pdf/2509.19742v2)**

> **作者:** Shuyu Zhang; Yifan Wei; Xinru Wang; Yanmin Zhu; Yangfan He; Yixuan Weng; Bin Li
>
> **摘要:** Zero-shot Dialog State Tracking (zs-DST) is essential for enabling Task-Oriented Dialog Systems (TODs) to generalize to new domains without costly data annotation. A central challenge lies in the semantic misalignment between dynamic dialog contexts and static prompts, leading to inflexible cross-layer coordination, domain interference, and catastrophic forgetting. To tackle this, we propose Hierarchical Collaborative Low-Rank Adaptation (HiCoLoRA), a framework that enhances zero-shot slot inference through robust prompt alignment. It features a hierarchical LoRA architecture for dynamic layer-specific processing (combining lower-layer heuristic grouping and higher-layer full interaction), integrates Spectral Joint Domain-Slot Clustering to identify transferable associations (feeding an Adaptive Linear Fusion Mechanism), and employs Semantic-Enhanced SVD Initialization (SemSVD-Init) to preserve pre-trained knowledge. Experiments on multi-domain datasets MultiWOZ and SGD show that HiCoLoRA outperforms baselines, achieving SOTA in zs-DST. Code is available at https://github.com/carsonz/HiCoLoRA.
>
---
#### [replaced 041] Stuffed Mamba: Oversized States Lead to the Inability to Forget
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.07145v3](http://arxiv.org/pdf/2410.07145v3)**

> **作者:** Yingfa Chen; Xinrong Zhang; Shengding Hu; Xu Han; Zhiyuan Liu; Maosong Sun
>
> **备注:** COLM 2025
>
> **摘要:** Recent advancements in recurrent architectures, such as Mamba and RWKV, have showcased strong language capabilities. Unlike transformer-based models, these architectures encode all contextual information into a fixed-size state, leading to great inference efficiency. However, this approach can cause information interference, where different token data conflicts, resulting in performance degradation and incoherent outputs beyond a certain context length. To prevent this, most RNNs incorporate mechanisms designed to "forget" earlier tokens. In this paper, we reveal that Mamba-based models struggle to effectively forget earlier tokens even with built-in forgetting mechanisms. We demonstrate that this issue stems from training on contexts that are too short for the state size, enabling the model to perform well without needing to learn how to forget. Then, we show that the minimum training length required for the model to learn forgetting scales linearly with the state size, and the maximum context length for accurate retrieval of a 5-digit passkey scales exponentially with the state size, indicating that the model retains some information beyond the point where forgetting begins. These findings highlight a critical limitation in current RNN architectures and provide valuable insights for improving long-context modeling. Our work suggests that future RNN designs must account for the interplay between state size, training length, and forgetting mechanisms to achieve robust performance in long-context tasks.
>
---
#### [replaced 042] Attributing Response to Context: A Jensen-Shannon Divergence Driven Mechanistic Study of Context Attribution in Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.16415v3](http://arxiv.org/pdf/2505.16415v3)**

> **作者:** Ruizhe Li; Chen Chen; Yuchen Hu; Yanjun Gao; Xi Wang; Emine Yilmaz
>
> **备注:** Accepted at NeurIPS 2025 Mechanistic Interpretability Workshop
>
> **摘要:** Retrieval-Augmented Generation (RAG) leverages large language models (LLMs) combined with external contexts to enhance the accuracy and reliability of generated responses. However, reliably attributing generated content to specific context segments, context attribution, remains challenging due to the computationally intensive nature of current methods, which often require extensive fine-tuning or human annotation. In this work, we introduce a novel Jensen-Shannon Divergence driven method to Attribute Response to Context (ARC-JSD), enabling efficient and accurate identification of essential context sentences without additional fine-tuning, gradient-calculation or surrogate modelling. Evaluations on a wide range of RAG benchmarks, such as TyDi QA, Hotpot QA, and Musique, using instruction-tuned LLMs in different scales demonstrate superior accuracy and significant computational efficiency improvements compared to the previous surrogate-based method. Furthermore, our mechanistic analysis reveals specific attention heads and multilayer perceptron (MLP) layers responsible for context attribution, providing valuable insights into the internal workings of RAG models and how they affect RAG behaviours. Our code is available at https://github.com/ruizheliUOA/ARC_JSD.
>
---
#### [replaced 043] HiddenBench: Assessing Collective Reasoning in Multi-Agent LLMs via Hidden Profile Tasks
- **分类: cs.CL; cs.AI; cs.MA**

- **链接: [http://arxiv.org/pdf/2505.11556v2](http://arxiv.org/pdf/2505.11556v2)**

> **作者:** Yuxuan Li; Aoi Naito; Hirokazu Shirado
>
> **摘要:** Multi-agent systems built on large language models (LLMs) promise enhanced problem-solving through distributed information integration, but may also replicate collective reasoning failures observed in human groups. Yet the absence of a theory-grounded benchmark makes it difficult to systematically evaluate and improve such reasoning. We introduce HiddenBench, the first benchmark for evaluating collective reasoning in multi-agent LLMs. It builds on the Hidden Profile paradigm from social psychology, where individuals each hold asymmetric pieces of information and must communicate to reach the correct decision. To ground the benchmark, we formalize the paradigm with custom tasks and show that GPT-4.1 groups fail to integrate distributed knowledge, exhibiting human-like collective reasoning failures that persist even with varied prompting strategies. We then construct the full benchmark, spanning 65 tasks drawn from custom designs, prior human studies, and automatic generation. Evaluating 15 LLMs across four model families, HiddenBench exposes persistent limitations while also providing comparative insights: some models (e.g., Gemini-2.5-Flash/Pro) achieve higher performance, yet scale and reasoning are not reliable indicators of stronger collective reasoning. Our work delivers the first reproducible benchmark for collective reasoning in multi-agent LLMs, offering diagnostic insight and a foundation for future research on artificial collective intelligence.
>
---
#### [replaced 044] Can LLMs be Good Graph Judge for Knowledge Graph Construction?
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.17388v4](http://arxiv.org/pdf/2411.17388v4)**

> **作者:** Haoyu Huang; Chong Chen; Zeang Sheng; Yang Li; Wentao Zhang
>
> **备注:** EMNLP 2025 Main
>
> **摘要:** In real-world scenarios, most of the data obtained from the information retrieval (IR) system is unstructured. Converting natural language sentences into structured Knowledge Graphs (KGs) remains a critical challenge. We identified three limitations with respect to existing KG construction methods: (1) There could be a large amount of noise in real-world documents, which could result in extracting messy information. (2) Naive LLMs usually extract inaccurate knowledge from some domain-specific documents. (3) Hallucination phenomenon cannot be overlooked when directly using LLMs to construct KGs. In this paper, we propose \textbf{GraphJudge}, a KG construction framework to address the aforementioned challenges. In this framework, we designed an entity-centric strategy to eliminate the noise information in the documents. And we fine-tuned a LLM as a graph judge to finally enhance the quality of generated KGs. Experiments conducted on two general and one domain-specific text-graph pair datasets demonstrate state-of-the-art performance against various baseline methods with strong generalization abilities. Our code is available at \href{https://github.com/hhy-huang/GraphJudge}{https://github.com/hhy-huang/GraphJudge}.
>
---
#### [replaced 045] Beyond Early-Token Bias: Model-Specific and Language-Specific Position Effects in Multilingual LLMs
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.16134v2](http://arxiv.org/pdf/2505.16134v2)**

> **作者:** Mikhail Menschikov; Alexander Kharitonov; Maiia Kotyga; Vadim Porvatov; Anna Zhukovskaya; David Kagramanyan; Egor Shvetsov; Evgeny Burnaev
>
> **摘要:** Large Language Models (LLMs) exhibit position bias - a systematic tendency to neglect information at specific context positions. However, the patterns of position bias behavior, depending on the language or model, remain unexplored. We present a multilingual study across five typologically distinct languages (English, Russian, German, Hindi, and Vietnamese) and five model architectures, examining how position bias interacts with prompt strategies and affects output entropy. Our key findings are: (1) Position bias is primarily model-driven, yet exhibits language-specific variations. For instance, Qwen2.5-7B-Instruct and DeepSeek 7B Chat consistently favors late positions, challenging established assumptions of a universal early-token bias in LLMs. (2) Explicitly instructing the model that "the context is relevant to the query" unexpectedly reduces accuracy across languages, undermining common prompt-engineering practices. (3) While the largest accuracy drop occurs when relevant information is placed in the middle of the context, this is not explicitly reflected by a corresponding peak in output entropy.
>
---
#### [replaced 046] Position IDs Matter: An Enhanced Position Layout for Efficient Context Compression in Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2409.14364v4](http://arxiv.org/pdf/2409.14364v4)**

> **作者:** Runsong Zhao; Xin Liu; Xinyu Liu; Pengcheng Huang; Chunyang Xiao; Tong Xiao; Jingbo Zhu
>
> **摘要:** Using special tokens (e.g., gist, memory, or compressed tokens) to compress context information is a common practice for large language models (LLMs). However, existing approaches often neglect that position encodings inherently induce local inductive biases in models, causing the compression process to ignore holistic contextual dependencies. We propose \textbf{Enhanced Position Layout (EPL)}, a simple yet effective method that improves the context compression capability of LLMs by only adjusting position IDs, the numerical identifiers that specify token positions. EPL minimizes the distance between context tokens and their corresponding special tokens and at the same time maintains the sequence order in position IDs between context tokens, special tokens, and the subsequent tokens. Integrating EPL into our best performing context compression model results in a 1.9 ROUGE-1 F1 improvement on out-of-domain question answering datasets on average. When extended to multimodal scenarios, EPL leads to an average accuracy gain of 2.6 points for vision compression LLMs.
>
---
#### [replaced 047] Vulnerability of LLMs to Vertically Aligned Text Manipulations
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.20016v4](http://arxiv.org/pdf/2410.20016v4)**

> **作者:** Zhecheng Li; Yiwei Wang; Bryan Hooi; Yujun Cai; Zhen Xiong; Nanyun Peng; Kai-wei Chang
>
> **备注:** Accepted to ACL 2025 (Main)
>
> **摘要:** Vertical text input is commonly encountered in various real-world applications, such as mathematical computations and word-based Sudoku puzzles. While current large language models (LLMs) have excelled in natural language tasks, they remain vulnerable to variations in text formatting. Recent research demonstrates that modifying input formats, such as vertically aligning words for encoder-based models, can substantially lower accuracy in text classification tasks. While easily understood by humans, these inputs can significantly mislead models, posing a potential risk of bypassing detection in real-world scenarios involving harmful or sensitive information. With the expanding application of LLMs, a crucial question arises: Do decoder-based LLMs exhibit similar vulnerabilities to vertically formatted text input? In this paper, we investigate the impact of vertical text input on the performance of various LLMs across multiple text classification datasets and analyze the underlying causes. Our findings are as follows: (i) Vertical text input significantly degrades the accuracy of LLMs in text classification tasks. (ii) Chain-of-Thought (CoT) reasoning does not help LLMs recognize vertical input or mitigate its vulnerability, but few-shot learning with careful analysis does. (iii) We explore the underlying cause of the vulnerability by analyzing the inherent issues in tokenization and attention matrices.
>
---
#### [replaced 048] Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.11756v2](http://arxiv.org/pdf/2505.11756v2)**

> **作者:** David Chanin; Tomáš Dulka; Adrià Garriga-Alonso
>
> **摘要:** It is assumed that sparse autoencoders (SAEs) decompose polysemantic activations into interpretable linear directions, as long as the activations are composed of sparse linear combinations of underlying features. However, we find that if an SAE is more narrow than the number of underlying "true features" on which it is trained, and there is correlation between features, the SAE will merge components of correlated features together, thus destroying monosemanticity. In LLM SAEs, these two conditions are almost certainly true. This phenomenon, which we call feature hedging, is caused by SAE reconstruction loss, and is more severe the narrower the SAE. In this work, we introduce the problem of feature hedging and study it both theoretically in toy models and empirically in SAEs trained on LLMs. We suspect that feature hedging may be one of the core reasons that SAEs consistently underperform supervised baselines. Finally, we use our understanding of feature hedging to propose an improved variant of matryoshka SAEs. Importantly, our work shows that SAE width is not a neutral hyperparameter: narrower SAEs suffer more from hedging than wider SAEs.
>
---
#### [replaced 049] RuCCoD: Towards Automated ICD Coding in Russian
- **分类: cs.CL; cs.AI; cs.DB**

- **链接: [http://arxiv.org/pdf/2502.21263v2](http://arxiv.org/pdf/2502.21263v2)**

> **作者:** Aleksandr Nesterov; Andrey Sakhovskiy; Ivan Sviridov; Airat Valiev; Vladimir Makharev; Petr Anokhin; Galina Zubkova; Elena Tutubalina
>
> **备注:** Accepted to EMNLP 2025 (Main Conference)
>
> **摘要:** This study investigates the feasibility of automating clinical coding in Russian, a language with limited biomedical resources. We present a new dataset for ICD coding, which includes diagnosis fields from electronic health records (EHRs) annotated with over 10,000 entities and more than 1,500 unique ICD codes. This dataset serves as a benchmark for several state-of-the-art models, including BERT, LLaMA with LoRA, and RAG, with additional experiments examining transfer learning across domains (from PubMed abstracts to medical diagnosis) and terminologies (from UMLS concepts to ICD codes). We then apply the best-performing model to label an in-house EHR dataset containing patient histories from 2017 to 2021. Our experiments, conducted on a carefully curated test set, demonstrate that training with the automated predicted codes leads to a significant improvement in accuracy compared to manually annotated data from physicians. We believe our findings offer valuable insights into the potential for automating clinical coding in resource-limited languages like Russian, which could enhance clinical efficiency and data accuracy in these contexts. Our code and dataset are available at https://github.com/auto-icd-coding/ruccod.
>
---
#### [replaced 050] Unlearning Isn't Deletion: Investigating Reversibility of Machine Unlearning in LLMs
- **分类: cs.CL; cs.AI; cs.CR; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.16831v2](http://arxiv.org/pdf/2505.16831v2)**

> **作者:** Xiaoyu Xu; Xiang Yue; Yang Liu; Qingqing Ye; Huadi Zheng; Peizhao Hu; Minxin Du; Haibo Hu
>
> **备注:** 46 pages
>
> **摘要:** Unlearning in large language models (LLMs) aims to remove specified data, but its efficacy is typically assessed with task-level metrics like accuracy and perplexity. We demonstrate that these metrics are often misleading, as models can appear to forget while their original behavior is easily restored through minimal fine-tuning. This phenomenon of \emph{reversibility} suggests that information is merely suppressed, not genuinely erased. To address this critical evaluation gap, we introduce a \emph{representation-level analysis framework}. Our toolkit comprises PCA-based similarity and shift, centered kernel alignment (CKA), and Fisher information, complemented by a summary metric, the mean PCA distance, to measure representational drift. Applying this framework across six unlearning methods, three data domains, and two LLMs, we identify four distinct forgetting regimes based on their \emph{reversibility} and \emph{catastrophicity}. Our analysis reveals that achieving the ideal state--irreversible, non-catastrophic forgetting--is exceptionally challenging. By probing the limits of unlearning, we identify a case of seemingly irreversible, targeted forgetting, offering new insights for designing more robust erasure algorithms. Our findings expose a fundamental gap in current evaluation practices and establish a representation-level foundation for trustworthy unlearning.
>
---
#### [replaced 051] Sycophancy Is Not One Thing: Causal Separation of Sycophantic Behaviors in LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.21305v2](http://arxiv.org/pdf/2509.21305v2)**

> **作者:** Daniel Vennemeyer; Phan Anh Duong; Tiffany Zhan; Tianyu Jiang
>
> **摘要:** Large language models (LLMs) often exhibit sycophantic behaviors -- such as excessive agreement with or flattery of the user -- but it is unclear whether these behaviors arise from a single mechanism or multiple distinct processes. We decompose sycophancy into sycophantic agreement and sycophantic praise, contrasting both with genuine agreement. Using difference-in-means directions, activation additions, and subspace geometry across multiple models and datasets, we show that: (1) the three behaviors are encoded along distinct linear directions in latent space; (2) each behavior can be independently amplified or suppressed without affecting the others; and (3) their representational structure is consistent across model families and scales. These results suggest that sycophantic behaviors correspond to distinct, independently steerable representations.
>
---
#### [replaced 052] Process Reinforcement through Implicit Rewards
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.01456v2](http://arxiv.org/pdf/2502.01456v2)**

> **作者:** Ganqu Cui; Lifan Yuan; Zefan Wang; Hanbin Wang; Yuchen Zhang; Jiacheng Chen; Wendi Li; Bingxiang He; Yuchen Fan; Tianyu Yu; Qixin Xu; Weize Chen; Jiarui Yuan; Huayu Chen; Kaiyan Zhang; Xingtai Lv; Shuo Wang; Yuan Yao; Xu Han; Hao Peng; Yu Cheng; Zhiyuan Liu; Maosong Sun; Bowen Zhou; Ning Ding
>
> **备注:** 24 pages. Model&Code&Data available at https://github.com/PRIME-RL/PRIME
>
> **摘要:** Dense process rewards have proven a more effective alternative to the sparse outcome-level rewards in the inference-time scaling of large language models (LLMs), particularly in tasks requiring complex multi-step reasoning. While dense rewards also offer an appealing choice for the reinforcement learning (RL) of LLMs since their fine-grained rewards have the potential to address some inherent issues of outcome rewards, such as training efficiency and credit assignment, this potential remains largely unrealized. This can be primarily attributed to the challenges of training process reward models (PRMs) online, where collecting high-quality process labels is prohibitively expensive, making them particularly vulnerable to reward hacking. To address these challenges, we propose PRIME (Process Reinforcement through IMplicit rEwards), which enables online PRM updates using only policy rollouts and outcome labels through implict process rewards. PRIME combines well with various advantage functions and forgoes the dedicated reward model training phrase that existing approaches require, substantially reducing the development overhead. We demonstrate PRIME's effectiveness on competitional math and coding. Starting from Qwen2.5-Math-7B-Base, PRIME achieves a 15.1% average improvement across several key reasoning benchmarks over the SFT model. Notably, our resulting model, Eurus-2-7B-PRIME, surpasses Qwen2.5-Math-7B-Instruct on seven reasoning benchmarks with 10% of its training data.
>
---
#### [replaced 053] LoopServe: An Adaptive Dual-phase LLM Inference Acceleration System for Multi-Turn Dialogues
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.13681v2](http://arxiv.org/pdf/2507.13681v2)**

> **作者:** Haoyang Li; Zhanchao Xu; Yiming Li; Xuejia Chen; Darian Li; Anxin Tian; Qingfa Xiao; Cheng Deng; Jun Wang; Qing Li; Lei Chen; Mingxuan Yuan
>
> **摘要:** Multi-turn dialogues are essential in many real-world applications of large language models, such as chatbots and virtual assistants. As conversation histories become longer, existing large language models face increasing computational and memory challenges, which hinder their ability to provide efficient and responsive interactions. Most current acceleration methods either compress the context or optimize key value caching, but they often rely on fixed or position-based heuristics that do not adapt well to the dynamic and unpredictable patterns found in actual multi-turn conversations. As a result, these models cannot accurately identify and prioritize the most relevant context, leading to degraded response quality. In this paper, we present LoopServe, an adaptive dual-phase inference acceleration framework for large language models in multi-turn dialogues. LoopServe introduces two main innovations. First, it performs online sparsification during the prefilling phase by dynamically selecting the most important parts of the attention matrix for each new input. Second, it uses progressive key value compression during decoding by adaptively maintaining a relevant and efficient cache based on the most recently generated output tokens. We also propose a new benchmark with eleven multi-turn datasets that reflect realistic query positions and conversational dependencies. Extensive experiments demonstrate that LoopServe consistently achieves superior effectiveness compared to existing baselines and significantly accelerates LLM inference across a wide range of long-context dialogue tasks.
>
---
#### [replaced 054] Recursive Training Loops in LLMs: How training data properties modulate distribution shift in generated data?
- **分类: cs.LG; cs.AI; cs.CL; 68T50; I.2.7**

- **链接: [http://arxiv.org/pdf/2504.03814v5](http://arxiv.org/pdf/2504.03814v5)**

> **作者:** Grgur Kovač; Jérémy Perez; Rémy Portelas; Peter Ford Dominey; Pierre-Yves Oudeyer
>
> **备注:** Accepted to EMNLP 2025 (Oral)
>
> **摘要:** Large language models (LLMs) are increasingly used in the creation of online content, creating feedback loops as subsequent generations of models will be trained on this synthetic data. Such loops were shown to lead to distribution shifts - models misrepresenting the true underlying distributions of human data (also called model collapse). However, how human data properties affect such shifts remains poorly understood. In this paper, we provide the first empirical examination of the effect of such properties on the outcome of recursive training. We first confirm that using different human datasets leads to distribution shifts of different magnitudes. Through exhaustive manipulation of dataset properties combined with regression analyses, we then identify a set of properties predicting distribution shift magnitudes. Lexical diversity is found to amplify these shifts, while semantic diversity and data quality mitigate them. Furthermore, we find that these influences are highly modular: data scrapped from a given internet domain has little influence on the content generated for another domain. Finally, experiments on political bias reveal that human data properties affect whether the initial bias will be amplified or reduced. Overall, our results portray a novel view, where different parts of internet may undergo different types of distribution shift.
>
---
#### [replaced 055] ASCoT: An Adaptive Self-Correction Chain-of-Thought Method for Late-Stage Fragility in LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.05282v3](http://arxiv.org/pdf/2508.05282v3)**

> **作者:** Dongxu Zhang; Ning Yang; Jihua Zhu; Jinnan Yang; Miao Xin; Baoliang Tian
>
> **摘要:** Chain-of-Thought (CoT) prompting has significantly advanced the reasoning capabilities of Large Language Models (LLMs), yet the reliability of these reasoning chains remains a critical challenge. A widely held "cascading failure" hypothesis suggests that errors are most detrimental when they occur early in the reasoning process. This paper challenges that assumption through systematic error-injection experiments, revealing a counter-intuitive phenomenon we term "Late-Stage Fragility": errors introduced in the later stages of a CoT chain are significantly more likely to corrupt the final answer than identical errors made at the beginning. To address this specific vulnerability, we introduce the Adaptive Self-Correction Chain-of-Thought (ASCoT) method. ASCoT employs a modular pipeline in which an Adaptive Verification Manager (AVM) operates first, followed by the Multi-Perspective Self-Correction Engine (MSCE). The AVM leverages a Positional Impact Score function I(k) that assigns different weights based on the position within the reasoning chains, addressing the Late-Stage Fragility issue by identifying and prioritizing high-risk, late-stage steps. Once these critical steps are identified, the MSCE applies robust, dual-path correction specifically to the failure parts. Extensive experiments on benchmarks such as GSM8K and MATH demonstrate that ASCoT achieves outstanding accuracy, outperforming strong baselines, including standard CoT. Our work underscores the importance of diagnosing specific failure modes in LLM reasoning and advocates for a shift from uniform verification strategies to adaptive, vulnerability-aware correction mechanisms.
>
---
#### [replaced 056] Learning the Wrong Lessons: Syntactic-Domain Spurious Correlations in Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.21155v2](http://arxiv.org/pdf/2509.21155v2)**

> **作者:** Chantal Shaib; Vinith M. Suriyakumar; Levent Sagun; Byron C. Wallace; Marzyeh Ghassemi
>
> **备注:** NeurIPS 2025 Spotlight
>
> **摘要:** For an LLM to correctly respond to an instruction it must understand both the semantics and the domain (i.e., subject area) of a given task-instruction pair. However, syntax can also convey implicit information Recent work shows that syntactic templates -- frequent sequences of Part-of-Speech (PoS) tags -- are prevalent in training data and often appear in model outputs. In this work we characterize syntactic templates, domain, and semantics in task-instruction pairs. We identify cases of spurious correlations between syntax and domain, where models learn to associate a domain with syntax during training; this can sometimes override prompt semantics. Using a synthetic training dataset, we find that the syntactic-domain correlation can lower performance (mean 0.51 +/- 0.06) on entity knowledge tasks in OLMo-2 models (1B-13B). We introduce an evaluation framework to detect this phenomenon in trained models, and show that it occurs on a subset of the FlanV2 dataset in open (OLMo-2-7B; Llama-4-Maverick), and closed (GPT-4o) models. Finally, we present a case study on the implications for safety finetuning, showing that unintended syntactic-domain correlations can be used to bypass refusals in OLMo-2-7B Instruct and GPT-4o. Our findings highlight two needs: (1) to explicitly test for syntactic-domain correlations, and (2) to ensure syntactic diversity in training data, specifically within domains, to prevent such spurious correlations.
>
---
#### [replaced 057] ChartGalaxy: A Dataset for Infographic Chart Understanding and Generation
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.18668v4](http://arxiv.org/pdf/2505.18668v4)**

> **作者:** Zhen Li; Duan Li; Yukai Guo; Xinyuan Guo; Bowen Li; Lanxi Xiao; Shenyu Qiao; Jiashu Chen; Zijian Wu; Hui Zhang; Xinhuan Shu; Shixia Liu
>
> **备注:** 58 pages
>
> **摘要:** Infographic charts are a powerful medium for communicating abstract data by combining visual elements (e.g., charts, images) with textual information. However, their visual and structural richness poses challenges for large vision-language models (LVLMs), which are typically trained on plain charts. To bridge this gap, we introduce ChartGalaxy, a million-scale dataset designed to advance the understanding and generation of infographic charts. The dataset is constructed through an inductive process that identifies 75 chart types, 440 chart variations, and 68 layout templates from real infographic charts and uses them to create synthetic ones programmatically. We showcase the utility of this dataset through: 1) improving infographic chart understanding via fine-tuning, 2) benchmarking code generation for infographic charts, and 3) enabling example-based infographic chart generation. By capturing the visual and structural complexity of real design, ChartGalaxy provides a useful resource for enhancing multimodal reasoning and generation in LVLMs.
>
---
#### [replaced 058] MUCAR: Benchmarking Multilingual Cross-Modal Ambiguity Resolution for Multimodal Large Language Models
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.17046v2](http://arxiv.org/pdf/2506.17046v2)**

> **作者:** Xiaolong Wang; Zhaolu Kang; Wangyuxuan Zhai; Xinyue Lou; Yunghwei Lai; Ziyue Wang; Yawen Wang; Kaiyu Huang; Yile Wang; Peng Li; Yang Liu
>
> **摘要:** Multimodal Large Language Models (MLLMs) have demonstrated significant advances across numerous vision-language tasks. MLLMs have shown promising capability in aligning visual and textual modalities, allowing them to process image-text pairs with clear and explicit meanings. However, resolving the inherent ambiguities present in real-world language and visual contexts remains a challenge. Existing multimodal benchmarks typically overlook linguistic and visual ambiguities, relying mainly on unimodal context for disambiguation and thus failing to exploit the mutual clarification potential between modalities. To bridge this gap, we introduce MUCAR, a novel and challenging benchmark designed explicitly for evaluating multimodal ambiguity resolution across multilingual and cross-modal scenarios. MUCAR includes first a multilingual dataset where ambiguous textual expressions are uniquely resolved by corresponding visual contexts, and second a dual-ambiguity dataset that systematically pairs ambiguous images with ambiguous textual contexts, with each combination carefully constructed to yield a single, clear interpretation through mutual disambiguation. Extensive evaluations involving 19 state-of-the-art multimodal models--encompassing both open-source and proprietary architectures--reveal substantial gaps compared to human-level performance, highlighting the need for future research into more sophisticated cross-modal ambiguity comprehension methods, further pushing the boundaries of multimodal reasoning.
>
---
#### [replaced 059] Positional Encoding via Token-Aware Phase Attention
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.12635v2](http://arxiv.org/pdf/2509.12635v2)**

> **作者:** Yu Wang; Sheng Shen; Rémi Munos; Hongyuan Zhan; Yuandong Tian
>
> **备注:** 24 pages
>
> **摘要:** We prove under practical assumptions that Rotary Positional Embedding (RoPE) introduces an intrinsic distance-dependent bias in attention scores that limits RoPE's ability to model long-context. RoPE extension methods may alleviate this issue, but they typically require post-hoc adjustments after pretraining, such as rescaling or hyperparameters retuning. This paper introduces Token-Aware Phase Attention (TAPA), a new positional encoding method that incorporates a learnable phase function into the attention mechanism. TAPA preserves token interactions over long range, extends to longer contexts with direct and light fine-tuning, extrapolates to unseen lengths, and attains significantly lower perplexity on long-context than RoPE families.
>
---
#### [replaced 060] KaLM-Embedding-V2: Superior Training Techniques and Data Inspire A Versatile Embedding Model
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.20923v3](http://arxiv.org/pdf/2506.20923v3)**

> **作者:** Xinping Zhao; Xinshuo Hu; Zifei Shan; Shouzheng Huang; Yao Zhou; Xin Zhang; Zetian Sun; Zhenyu Liu; Dongfang Li; Xinyuan Wei; Youcheng Pan; Yang Xiang; Meishan Zhang; Haofen Wang; Jun Yu; Baotian Hu; Min Zhang
>
> **备注:** 32 pages, 16 tables, 5 figures
>
> **摘要:** Recent advancements in Large Language Models (LLMs)-based text embedding models primarily focus on data scaling or synthesis, yet limited exploration of training techniques and data quality, thereby constraining performance. In this work, we propose KaLM-Embedding-V2, a series of versatile and compact embedding models, systematically incentivizing advanced embedding capability in LLMs by superior training techniques and high-quality data. For model architecture, we implement the models on a 0.5B compact size with simple mean-pooling to produce fixed-length embeddings and remove the causal attention mask to enable fully bidirectional representation learning. For training techniques, we propose a progressive multi-stage training pipeline: pre-training on weakly supervised large-scale datasets, fine-tuning with supervised high-quality datasets, and contrastive distillation with fine-grained soft signals, integrated with focal-style reweighting and online hard-negative mixing to emphasize difficult samples and enrich hard negatives, respectively. For training data, we curate over 20 categories for pre-training and 100 categories for fine-tuning and contrastive distillation, to improve both performance and generalization, leveraging task-specific instructions, hard-negative mining, and example-based multi-class labeling to ensure high quality. Combining these techniques, our KaLM-Embedding-V2 series achieves state-of-the-art performance on the Massive Text Embedding Benchmark, outperforming models of comparable size and rivaling models 3-26x larger, setting a new standard for versatile and compact embedding models under 1B parameters. The code, data, and models will be publicly available to facilitate academic research.
>
---
#### [replaced 061] The Imitation Game: Turing Machine Imitator is Length Generalizable Reasoner
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.13332v2](http://arxiv.org/pdf/2507.13332v2)**

> **作者:** Zhouqi Hua; Wenwei Zhang; Chengqi Lyu; Yuzhe Gu; Songyang Gao; Kuikun Liu; Dahua Lin; Kai Chen
>
> **摘要:** Length generalization, the ability to solve problems of longer sequences than those observed during training, poses a core challenge of Transformer-based large language models (LLM). Although existing studies have predominantly focused on data-driven approaches for arithmetic operations and symbolic manipulation tasks, these approaches tend to be task-specific with limited overall performance. To pursue a more general solution, this paper focuses on a broader case of reasoning problems that are computable, i.e., problems that algorithms can solve, thus can be solved by the Turing Machine. From this perspective, this paper proposes Turing MAchine Imitation Learning (TAIL) to improve the length generalization ability of LLMs. TAIL synthesizes chain-of-thoughts (CoT) data that imitate the execution process of a Turing Machine by computer programs, which linearly expands the reasoning steps into atomic states to alleviate shortcut learning and explicit memory fetch mechanism to reduce the difficulties of dynamic and long-range data access in elementary operations. To validate the reliability and universality of TAIL, we construct a challenging synthetic dataset covering 8 classes of algorithms and 18 tasks. Without bells and whistles, TAIL significantly improves the length generalization ability as well as the performance of Qwen2.5-7B on various tasks using only synthetic data, surpassing previous methods and DeepSeek-R1. The experimental results reveal that the key concepts in the Turing Machine, instead of the thinking styles, are indispensable for TAIL for length generalization, through which the model exhibits read-and-write behaviors consistent with the properties of the Turing Machine in their attention layers. This work provides a promising direction for future research in the learning of LLM reasoning from synthetic data.
>
---
#### [replaced 062] Adaptively profiling models with task elicitation
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.01986v3](http://arxiv.org/pdf/2503.01986v3)**

> **作者:** Davis Brown; Prithvi Balehannina; Helen Jin; Shreya Havaldar; Hamed Hassani; Eric Wong
>
> **备注:** EMNLP 2025 Main Conference
>
> **摘要:** Language model evaluations often fail to characterize consequential failure modes, forcing experts to inspect outputs and build new benchmarks. We introduce task elicitation, a method that automatically builds new evaluations to profile model behavior. Task elicitation finds hundreds of natural-language tasks -- an order of magnitude more than prior work -- where frontier models exhibit systematic failures, in domains ranging from forecasting to online harassment. For example, we find that Sonnet 3.5 over-associates quantum computing and AGI and that o3-mini is prone to hallucination when fabrications are repeated in-context.
>
---
#### [replaced 063] $100K or 100 Days: Trade-offs when Pre-Training with Academic Resources
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.23261v2](http://arxiv.org/pdf/2410.23261v2)**

> **作者:** Apoorv Khandelwal; Tian Yun; Nihal V. Nayak; Jack Merullo; Stephen H. Bach; Chen Sun; Ellie Pavlick
>
> **备注:** Published at COLM 2025
>
> **摘要:** Pre-training is notoriously compute-intensive and academic researchers are notoriously under-resourced. It is, therefore, commonly assumed that academics can't pre-train models. In this paper, we seek to clarify this assumption. We first survey academic researchers to learn about their available compute and then empirically measure the time to replicate models on such resources. We introduce a benchmark to measure the time to pre-train models on given GPUs and also identify ideal settings for maximizing training speed. We run our benchmark on a range of models and academic GPUs, spending 2,000 GPU-hours on our experiments. Our results reveal a brighter picture for academic pre-training: for example, although Pythia-1B was originally trained on 64 GPUs for 3 days, we find it is also possible to replicate this model (with the same hyper-parameters) in 3x fewer GPU-days: i.e. on 4 GPUs in 18 days. We conclude with a cost-benefit analysis to help clarify the trade-offs between price and pre-training time. We believe our benchmark will help academic researchers conduct experiments that require training larger models on more data. We fully release our codebase at: https://github.com/apoorvkh/academic-pretraining.
>
---
#### [replaced 064] Training-Free Bayesianization for Low-Rank Adapters of Large Language Models
- **分类: stat.ML; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.05723v3](http://arxiv.org/pdf/2412.05723v3)**

> **作者:** Haizhou Shi; Yibin Wang; Ligong Han; Huan Zhang; Hao Wang
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Estimating the uncertainty of responses from Large Language Models (LLMs) remains a critical challenge. While recent Bayesian methods have demonstrated effectiveness in quantifying uncertainty through low-rank weight updates, they typically require complex fine-tuning or post-training procedures. In this paper, we propose Training-Free Bayesianization (TFB), a simple yet theoretically grounded framework that efficiently transforms trained low-rank adapters into Bayesian ones without additional training. TFB systematically searches for the maximally acceptable level of variance in the weight posterior, constrained within a family of low-rank isotropic Gaussian distributions. Our theoretical analysis shows that under mild conditions, this search process is equivalent to KL-regularized variational optimization, a generalized form of variational inference. Through comprehensive experiments, we show that TFB achieves superior uncertainty estimation and generalization compared to existing methods while eliminating the need for complex Bayesianization training procedures. Code will be available at https://github.com/Wang-ML-Lab/bayesian-peft.
>
---
#### [replaced 065] BiomedSQL: Text-to-SQL for Scientific Reasoning on Biomedical Knowledge Bases
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.20321v2](http://arxiv.org/pdf/2505.20321v2)**

> **作者:** Mathew J. Koretsky; Maya Willey; Adi Asija; Owen Bianchi; Chelsea X. Alvarado; Tanay Nayak; Nicole Kuznetsov; Sungwon Kim; Mike A. Nalls; Daniel Khashabi; Faraz Faghri
>
> **备注:** Under Review
>
> **摘要:** Biomedical researchers increasingly rely on large-scale structured databases for complex analytical tasks. However, current text-to-SQL systems often struggle to map qualitative scientific questions into executable SQL, particularly when implicit domain reasoning is required. We introduce BiomedSQL, the first benchmark explicitly designed to evaluate scientific reasoning in text-to-SQL generation over a real-world biomedical knowledge base. BiomedSQL comprises 68,000 question/SQL query/answer triples grounded in a harmonized BigQuery knowledge base that integrates gene-disease associations, causal inference from omics data, and drug approval records. Each question requires models to infer domain-specific criteria, such as genome-wide significance thresholds, effect directionality, or trial phase filtering, rather than rely on syntactic translation alone. We evaluate a range of open- and closed-source LLMs across prompting strategies and interaction paradigms. Our results reveal a substantial performance gap: GPT-o3-mini achieves 59.0% execution accuracy, while our custom multi-step agent, BMSQL, reaches 62.6%, both well below the expert baseline of 90.0%. BiomedSQL provides a new foundation for advancing text-to-SQL systems capable of supporting scientific discovery through robust reasoning over structured biomedical knowledge bases. Our dataset is publicly available at https://huggingface.co/datasets/NIH-CARD/BiomedSQL, and our code is open-source at https://github.com/NIH-CARD/biomedsql.
>
---
#### [replaced 066] GEP: A GCG-Based method for extracting personally identifiable information from chatbots built on small language models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.21192v2](http://arxiv.org/pdf/2509.21192v2)**

> **作者:** Jieli Zhu; Vi Ngoc-Nha Tran
>
> **备注:** 16 pages, 5 figures, 4 tables
>
> **摘要:** Small language models (SLMs) become unprecedentedly appealing due to their approximately equivalent performance compared to large language models (LLMs) in certain fields with less energy and time consumption during training and inference. However, the personally identifiable information (PII) leakage of SLMs for downstream tasks has yet to be explored. In this study, we investigate the PII leakage of the chatbot based on SLM. We first finetune a new chatbot, i.e., ChatBioGPT based on the backbone of BioGPT using medical datasets Alpaca and HealthCareMagic. It shows a matchable performance in BERTscore compared with previous studies of ChatDoctor and ChatGPT. Based on this model, we prove that the previous template-based PII attacking methods cannot effectively extract the PII in the dataset for leakage detection under the SLM condition. We then propose GEP, which is a greedy coordinate gradient-based (GCG) method specifically designed for PII extraction. We conduct experimental studies of GEP and the results show an increment of up to 60$\times$ more leakage compared with the previous template-based methods. We further expand the capability of GEP in the case of a more complicated and realistic situation by conducting free-style insertion where the inserted PII in the dataset is in the form of various syntactic expressions instead of fixed templates, and GEP is still able to reveal a PII leakage rate of up to 4.53%.
>
---
#### [replaced 067] ExpertSteer: Intervening in LLMs through Expert Knowledge
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.12313v2](http://arxiv.org/pdf/2505.12313v2)**

> **作者:** Weixuan Wang; Minghao Wu; Barry Haddow; Alexandra Birch
>
> **摘要:** Large Language Models (LLMs) exhibit remarkable capabilities across various tasks, yet guiding them to follow desired behaviours during inference remains a significant challenge. Activation steering offers a promising method to control the generation process of LLMs by modifying their internal activations. However, existing methods commonly intervene in the model's behaviour using steering vectors generated by the model itself, which constrains their effectiveness to that specific model and excludes the possibility of leveraging powerful external expert models for steering. To address these limitations, we propose ExpertSteer, a novel approach that leverages arbitrary specialized expert models to generate steering vectors, enabling intervention in any LLMs. ExpertSteer transfers the knowledge from an expert model to a target LLM through a cohesive four-step process: first aligning representation dimensions with auto-encoders to enable cross-model transfer, then identifying intervention layer pairs based on mutual information analysis, next generating steering vectors from the expert model using Recursive Feature Machines, and finally applying these vectors on the identified layers during inference to selectively guide the target LLM without updating model parameters. We conduct comprehensive experiments using three LLMs on 15 popular benchmarks across four distinct domains. Experiments demonstrate that ExpertSteer significantly outperforms established baselines across diverse tasks at minimal cost.
>
---
#### [replaced 068] Semantic Component Analysis: Introducing Multi-Topic Distributions to Clustering-Based Topic Modeling
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.21054v3](http://arxiv.org/pdf/2410.21054v3)**

> **作者:** Florian Eichin; Carolin M. Schuster; Georg Groh; Michael A. Hedderich
>
> **备注:** 5 pages, 3 figures, code: https://github.com/mainlp/semantic_components
>
> **摘要:** Topic modeling is a key method in text analysis, but existing approaches fail to efficiently scale to large datasets or are limited by assuming one topic per document. Overcoming these limitations, we introduce Semantic Component Analysis (SCA), a topic modeling technique that discovers multiple topics per sample by introducing a decomposition step to the clustering-based topic modeling framework. We evaluate SCA on Twitter datasets in English, Hausa and Chinese. There, it achieves competitive coherence and diversity compared to BERTopic, while uncovering at least double the topics and maintaining a noise rate close to zero. We also find that SCA outperforms the LLM-based TopicGPT in scenarios with similar compute budgets. SCA thus provides an effective and efficient approach for topic modeling of large datasets.
>
---
#### [replaced 069] Sharing Matters: Analysing Neurons Across Languages and Tasks in LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2406.09265v3](http://arxiv.org/pdf/2406.09265v3)**

> **作者:** Weixuan Wang; Barry Haddow; Minghao Wu; Wei Peng; Alexandra Birch
>
> **摘要:** Large language models (LLMs) have revolutionized the field of natural language processing (NLP), and recent studies have aimed to understand their underlying mechanisms. However, most of this research is conducted within a monolingual setting, primarily focusing on English. Few studies have attempted to explore the internal workings of LLMs in multilingual settings. In this study, we aim to fill this research gap by examining how neuron activation is shared across tasks and languages. We classify neurons into four distinct categories based on their responses to a specific input across different languages: all-shared, partial-shared, specific, and non-activated. Building upon this categorisation, we conduct extensive experiments on three tasks across nine languages using several LLMs and present an in-depth analysis in this work. Our findings reveal that: (i) deactivating the all-shared neurons significantly decreases performance; (ii) the shared neurons play a vital role in generating responses, especially for the all-shared neurons; (iii) neuron activation patterns are highly sensitive and vary across tasks, LLMs, and languages. These findings shed light on the internal workings of multilingual LLMs and pave the way for future research. We release the code to foster research in this area.
>
---
#### [replaced 070] What Factors Affect LLMs and RLLMs in Financial Question Answering?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.08339v3](http://arxiv.org/pdf/2507.08339v3)**

> **作者:** Peng Wang; Xuesi Hu; Jiageng Wu; Yuntao Zou; Qiancheng Zhang; Dagang Li
>
> **备注:** Preprint
>
> **摘要:** Recently, the development of large language models (LLMs) and reasoning large language models (RLLMs) have gained considerable attention from many researchers. RLLMs enhance the reasoning capabilities of LLMs through Long Chain-of-Thought (Long CoT) processes, significantly improving the performance of LLMs in addressing complex problems. However, there are few works that systematically explore what methods can fully unlock the performance of LLMs and RLLMs within the financial domain. To investigate the impact of various methods on LLMs and RLLMs, we utilize five LLMs and three RLLMs to assess the effects of prompting methods, agentic frameworks, and multilingual alignment methods on financial question-answering tasks. Our research findings indicate: (1) Current prompting methods and agent frameworks enhance the performance of LLMs in financial question answering by simulating Long CoT; (2) RLLMs possess inherent Long CoT capabilities, which limits the effectiveness of conventional methods in further enhancing their performance; (3) Current advanced multilingual alignment methods primarily improve the multilingual performance of LLMs by extending the reasoning length, which yields minimal benefits for RLLMs. Additionally, we discuss strategies for enhancing the performance of LLMs and RLLMs in financial question answering, which may serve as a inspiration for future improvements. We hope that this study can serve as an important reference for LLMs and RLLMs in the field of financial question answering.
>
---
#### [replaced 071] Domain-Aware Tensor Network Structure Search
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.23537v2](http://arxiv.org/pdf/2505.23537v2)**

> **作者:** Giorgos Iacovides; Wuyang Zhou; Chao Li; Qibin Zhao; Danilo Mandic
>
> **摘要:** Tensor networks (TNs) provide efficient representations of high-dimensional data, yet identification of the optimal TN structures, the so called tensor network structure search (TN-SS) problem, remains a challenge. Current state-of-the-art (SOTA) algorithms solve TN-SS as a purely numerical optimization problem and require extensive function evaluations, which is prohibitive for real-world applications. In addition, existing methods ignore the valuable domain information inherent in real-world tensor data and lack transparency in their identified TN structures. To this end, we propose a novel TN-SS framework, termed the tnLLM, which incorporates domain information about the data and harnesses the reasoning capabilities of large language models (LLMs) to directly predict suitable TN structures. The proposed framework involves a domain-aware prompting pipeline which instructs the LLM to infer suitable TN structures based on the real-world relationships between tensor modes. In this way, our approach is capable of not only iteratively optimizing the objective function, but also generating domain-aware explanations for the identified structures. Experimental results demonstrate that tnLLM achieves comparable TN-SS objective function values with much fewer function evaluations compared to SOTA algorithms. Furthermore, we demonstrate that the LLM-enabled domain information can be used to find good initializations in the search space for sampling-based SOTA methods to accelerate their convergence while preserving theoretical performance guarantees.
>
---
#### [replaced 072] Improving the Language Understanding Capabilities of Large Language Models Using Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.11020v5](http://arxiv.org/pdf/2410.11020v5)**

> **作者:** Bokai Hu; Sai Ashish Somayajula; Xin Pan; Pengtao Xie
>
> **摘要:** Instruction-fine-tuned large language models (LLMs) under 14B parameters continue to underperform on natural language understanding (NLU) tasks, often trailing smaller models like BERT-base on benchmarks such as GLUE and SuperGLUE. Motivated by the success of reinforcement learning in reasoning tasks (e.g., DeepSeek), we explore Proximal Policy Optimization (PPO) as a framework to improve the NLU capabilities of LLMs. We frame NLU as a reinforcement learning environment, treating token generation as a sequence of actions and optimizing for reward signals based on alignment with ground-truth labels. PPO consistently outperforms supervised fine-tuning, yielding an average improvement of 6.3 points on GLUE, and surpasses zero-shot and few-shot prompting by 38.7 and 26.1 points, respectively. Notably, PPO-tuned models outperform GPT-4o by over 4\% on average across sentiment and natural language inference tasks, including gains of 7.3\% on the Mental Health dataset and 10.9\% on SIGA-nli. This work highlights a promising direction for adapting LLMs to new tasks by reframing them as reinforcement learning problems, enabling learning through simple end-task rewards rather than extensive data curation.
>
---
#### [replaced 073] How LLMs Fail to Support Fact-Checking
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.01902v2](http://arxiv.org/pdf/2503.01902v2)**

> **作者:** Adiba Mahbub Proma; Neeley Pate; James Druckman; Gourab Ghoshal; Hangfeng He; Ehsan Hoque
>
> **备注:** Adiba and Neeley contributed equally
>
> **摘要:** While Large Language Models (LLMs) can amplify online misinformation, they also show promise in tackling misinformation. In this paper, we empirically study the capabilities of three LLMs -- ChatGPT, Gemini, and Claude -- in countering political misinformation. We implement a two-step, chain-of-thought prompting approach, where models first identify credible sources for a given claim and then generate persuasive responses. Our findings suggest that models struggle to ground their responses in real news sources, and tend to prefer citing left-leaning sources. We also observe varying degrees of response diversity among models. Our findings highlight concerns about using LLMs for fact-checking through only prompt-engineering, emphasizing the need for more robust guardrails. Our results have implications for both researchers and non-technical users.
>
---
#### [replaced 074] CMRAG: Co-modality-based visual document retrieval and question answering
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.02123v2](http://arxiv.org/pdf/2509.02123v2)**

> **作者:** Wang Chen; Wenhan Yu; Guanqiang Qi; Weikang Li; Yang Li; Lei Sha; Deguo Xia; Jizhou Huang
>
> **摘要:** Retrieval-Augmented Generation (RAG) has become a core paradigm in document question answering tasks. However, existing methods have limitations when dealing with multimodal documents: one category of methods relies on layout analysis and text extraction, which can only utilize explicit text information and struggle to capture images or unstructured content; the other category treats document segmentation as visual input and directly passes it to visual language models (VLMs) for processing, yet it ignores the semantic advantages of text, leading to suboptimal retrieval and generation results. To address these research gaps, we propose the Co-Modality-based RAG (CMRAG) framework, which can simultaneously leverage texts and images for more accurate retrieval and generation. Our framework includes two key components: (1) a Unified Encoding Model (UEM) that projects queries, parsed text, and images into a shared embedding space via triplet-based training, and (2) a Unified Co-Modality-informed Retrieval (UCMR) method that statistically normalizes similarity scores to effectively fuse cross-modal signals. To support research in this direction, we further construct and release a large-scale triplet dataset of (query, text, image) examples. Experiments demonstrate that our proposed framework consistently outperforms single-modality--based RAG in multiple visual document question-answering (VDQA) benchmarks. The findings of this paper show that integrating co-modality information into the RAG framework in a unified manner is an effective approach to improving the performance of complex VDQA systems.
>
---
#### [replaced 075] LLMAEL: Large Language Models are Good Context Augmenters for Entity Linking
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2407.04020v3](http://arxiv.org/pdf/2407.04020v3)**

> **作者:** Amy Xin; Yunjia Qi; Zijun Yao; Fangwei Zhu; Kaisheng Zeng; Xu Bin; Lei Hou; Juanzi Li
>
> **摘要:** Specialized entity linking (EL) models are well-trained at mapping mentions to unique knowledge base (KB) entities according to a given context. However, specialized EL models struggle to disambiguate long-tail entities due to their limited training data. Meanwhile, extensively pre-trained large language models (LLMs) possess broader knowledge of uncommon entities. Yet, with a lack of specialized EL training, LLMs frequently fail to generate accurate KB entity names, limiting their standalone effectiveness in EL. With the observation that LLMs are more adept at context generation instead of EL execution, we introduce LLM-Augmented Entity Linking (LLMAEL), the first framework to enhance specialized EL models with LLM data augmentation. LLMAEL leverages off-the-shelf, tuning-free LLMs as context augmenters, generating entity descriptions to serve as additional input for specialized EL models. Experiments show that LLMAEL sets new state-of-the-art results across 6 widely adopted EL benchmarks: compared to prior methods that integrate tuning-free LLMs into EL, LLMAEL achieves an absolute 8.9% gain in EL accuracy. We release our code and datasets.
>
---
#### [replaced 076] Shadow-FT: Tuning Instruct Model via Training on Paired Base Model
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.12716v3](http://arxiv.org/pdf/2505.12716v3)**

> **作者:** Taiqiang Wu; Runming Yang; Jiayi Li; Pengfei Hu; Yik-Chung Wu; Ngai Wong; Yujiu Yang
>
> **备注:** 24 pages, 12 tables, 8 figures. Previous name: Shadow-FT: Tuning Instruct via Base
>
> **摘要:** Large language models (LLMs) consistently benefit from further fine-tuning on various tasks. However, we observe that directly tuning the Instruct (i.e., instruction-tuned) models often leads to marginal improvements and even performance degeneration. Notably, paired Base models, the foundation for these Instruct variants, contain highly similar weight values (i.e., less than 2% on average for Llama 3.1 8B). The Base model tends to be a good learner yet a weak backbone without post-training. Therefore, we propose a novel Shadow-FT framework to tune the Instruct models by leveraging the corresponding Base models. The key insight is to fine-tune the Base model, and then \textit{directly} graft the learned weight updates to the Instruct model. Our proposed Shadow-FT introduces no additional parameters, is easy to implement, and significantly improves performance. We conduct extensive experiments on tuning mainstream LLMs, such as Qwen 3 and Llama 3 series, and evaluate them across 19 benchmarks covering coding, reasoning, and mathematical tasks. Experimental results demonstrate that Shadow-FT consistently outperforms conventional full-parameter and parameter-efficient tuning approaches. Further analyses indicate that Shadow-FT can be applied to multimodal large language models (MLLMs) and combined with direct preference optimization~(DPO). Codes and weights are available at \href{https://github.com/wutaiqiang/Shadow-FT}{Github}.
>
---
#### [replaced 077] MrGuard: A Multilingual Reasoning Guardrail for Universal LLM Safety
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.15241v3](http://arxiv.org/pdf/2504.15241v3)**

> **作者:** Yahan Yang; Soham Dan; Shuo Li; Dan Roth; Insup Lee
>
> **备注:** Preprint
>
> **摘要:** Large Language Models (LLMs) are susceptible to adversarial attacks such as jailbreaking, which can elicit harmful or unsafe behaviors. This vulnerability is exacerbated in multilingual settings, where multilingual safety-aligned data is often limited. Thus, developing a guardrail capable of detecting and filtering unsafe content across diverse languages is critical for deploying LLMs in real-world applications. In this work, we introduce a multilingual guardrail with reasoning for prompt classification. Our method consists of: (1) synthetic multilingual data generation incorporating culturally and linguistically nuanced variants, (2) supervised fine-tuning, and (3) a curriculum-based Group Relative Policy Optimization (GRPO) framework that further improves performance. Experimental results demonstrate that our multilingual guardrail, MrGuard, consistently outperforms recent baselines across both in-domain and out-of-domain languages by more than 15%. We also evaluate MrGuard's robustness to multilingual variations, such as code-switching and low-resource language distractors in the prompt, and demonstrate that it preserves safety judgments under these challenging conditions. The multilingual reasoning capability of our guardrail enables it to generate explanations, which are particularly useful for understanding language-specific risks and ambiguities in multilingual content moderation.
>
---
#### [replaced 078] UltraEdit: Training-, Subject-, and Memory-Free Lifelong Editing in Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.14679v2](http://arxiv.org/pdf/2505.14679v2)**

> **作者:** Xiaojie Gu; Ziying Huang; Jia-Chen Gu; Kai Zhang
>
> **摘要:** Lifelong learning enables large language models (LLMs) to adapt to evolving information by continually updating their internal knowledge. An ideal system should support efficient, wide-ranging updates while preserving existing capabilities and ensuring reliable deployment. Model editing stands out as a promising solution for this goal, offering a focused and efficient way to revise a model's internal knowledge. Although recent paradigms have made notable progress, they often struggle to meet the demands of practical lifelong adaptation at scale. To bridge this gap, we propose UltraEdit, a training-, subject-, and memory-free approach that is well-suited for ultra-scalable, real-world lifelong model editing. UltraEdit fundamentally differs from traditional paradigms by computing parameter shifts in one step using only a hidden state and its gradient, making the approach simple yet efficient. To improve scalability in lifelong settings, UltraEdit employs a lifelong normalization strategy that continuously updates feature statistics across turns, allowing it to adapt to distributional shifts and maintain consistency over time. UltraEdit achieves editing speeds over 7x faster than the previous state-of-the-art method, which was also the fastest known approach, while using less than 1/4 the VRAM. This makes it the only method currently capable of editing a 7B LLM on a 24GB consumer-grade GPU. Furthermore, we construct UltraEditBench, the largest dataset in the field to date with over 2M editing pairs, and demonstrate that our method supports up to 2M edits while maintaining high accuracy. Comprehensive experiments on five datasets and six models show that UltraEdit consistently achieves superior performance across diverse model editing scenarios, taking a further step towards safe and scalable lifelong learning. Our code is available at: https://github.com/XiaojieGu/UltraEdit
>
---
#### [replaced 079] Development and Validation of a Large Language Model for Generating Fully-Structured Radiology Reports
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2409.18319v3](http://arxiv.org/pdf/2409.18319v3)**

> **作者:** Chuang Niu; Md Sayed Tanveer; Md Zabirul Islam; Parisa Kaviani; Qing Lyu; Mannudeep K. Kalra; Christopher T. Whitlow; Ge Wang
>
> **摘要:** Current LLMs for creating fully-structured reports face the challenges of formatting errors, content hallucinations, and privacy leakage issues when uploading data to external servers.We aim to develop an open-source, accurate LLM for creating fully-structured and standardized LCS reports from varying free-text reports across institutions and demonstrate its utility in automatic statistical analysis and individual lung nodule retrieval. With IRB approvals, our retrospective study included 5,442 de-identified LDCT LCS radiology reports from two institutions. We constructed two evaluation datasets by labeling 500 pairs of free-text and fully-structured radiology reports and one large-scale consecutive dataset from January 2021 to December 2023. Two radiologists created a standardized template for recording 27 lung nodule features on LCS. We designed a dynamic-template-constrained decoding method to enhance existing LLMs for creating fully-structured reports from free-text radiology reports. Using consecutive structured reports, we automated descriptive statistical analyses and a nodule retrieval prototype. Our best LLM for creating fully-structured reports achieved high performance on cross-institutional datasets with an F1 score of about 97%, with neither formatting errors nor content hallucinations. Our method consistently improved the best open-source LLMs by up to 10.42%, and outperformed GPT-4o by 17.19%. The automatically derived statistical distributions were consistent with prior findings regarding attenuation, location, size, stability, and Lung-RADS. The retrieval system with structured reports allowed flexible nodule-level search and complex statistical analysis. Our developed software is publicly available for local deployment and further research.
>
---
#### [replaced 080] Thinking Augmented Pre-training
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.20186v3](http://arxiv.org/pdf/2509.20186v3)**

> **作者:** Liang Wang; Nan Yang; Shaohan Huang; Li Dong; Furu Wei
>
> **备注:** 19 pages
>
> **摘要:** This paper introduces a simple and scalable approach to improve the data efficiency of large language model (LLM) training by augmenting existing text data with thinking trajectories. The compute for pre-training LLMs has been growing at an unprecedented rate, while the availability of high-quality data remains limited. Consequently, maximizing the utility of available data constitutes a significant research challenge. A primary impediment is that certain high-quality tokens are difficult to learn given a fixed model capacity, as the underlying rationale for a single token can be exceptionally complex and deep. To address this issue, we propose Thinking augmented Pre-Training (TPT), a universal methodology that augments text with automatically generated thinking trajectories. Such augmentation effectively increases the volume of the training data and makes high-quality tokens more learnable through step-by-step reasoning and decomposition. We apply TPT across diverse training configurations up to $100$B tokens, encompassing pre-training with both constrained and abundant data, as well as mid-training from strong open-source checkpoints. Experimental results indicate that our method substantially improves the performance of LLMs across various model sizes and families. Notably, TPT enhances the data efficiency of LLM pre-training by a factor of $3$. For a $3$B parameter model, it improves the post-training performance by over $10\%$ on several challenging reasoning benchmarks.
>
---
#### [replaced 081] From Roots to Rewards: Dynamic Tree Reasoning with Reinforcement Learning
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.13142v4](http://arxiv.org/pdf/2507.13142v4)**

> **作者:** Ahmed Bahloul; Simon Malberg
>
> **备注:** RARA Workshop @ ICDM 2025
>
> **摘要:** Modern language models address complex questions through chain-of-thought (CoT) reasoning (Wei et al., 2023) and retrieval augmentation (Lewis et al., 2021), yet struggle with error propagation and knowledge integration. Tree-structured reasoning methods, particularly the Probabilistic Tree-of-Thought (ProbTree)(Cao et al., 2023) framework, mitigate these issues by decomposing questions into hierarchical structures and selecting answers through confidence-weighted aggregation of parametric and retrieved knowledge (Yao et al., 2023). However, ProbTree's static implementation introduces two key limitations: (1) the reasoning tree is fixed during the initial construction phase, preventing dynamic adaptation to intermediate results, and (2) each node requires exhaustive evaluation of all possible solution strategies, creating computational inefficiency. We present a dynamic reinforcement learning (Sutton and Barto, 2018) framework that transforms tree-based reasoning into an adaptive process. Our approach incrementally constructs the reasoning tree based on real-time confidence estimates, while learning optimal policies for action selection (decomposition, retrieval, or aggregation). This maintains ProbTree's probabilistic rigor while improving both solution quality and computational efficiency through selective expansion and focused resource allocation. The work establishes a new paradigm for treestructured reasoning that balances the reliability of probabilistic frameworks with the flexibility required for real-world question answering systems. Code available at: https://github.com/ahmedehabb/From-Roots-to-Rewards-Dynamic-Tree-Reasoning-with-RL
>
---
#### [replaced 082] Intercept Cancer: Cancer Pre-Screening with Large Scale Healthcare Foundation Models
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.00209v2](http://arxiv.org/pdf/2506.00209v2)**

> **作者:** Liwen Sun; Hao-Ren Yao; Gary Gao; Ophir Frieder; Chenyan Xiong
>
> **摘要:** Cancer screening, leading to early detection, saves lives. Unfortunately, existing screening techniques require expensive and intrusive medical procedures, not globally available, resulting in too many lost would-be-saved lives. We present CATCH-FM, CATch Cancer early with Healthcare Foundation Models, a cancer pre-screening methodology that identifies high-risk patients for further screening solely based on their historical medical records. With millions of electronic healthcare records (EHR), we establish the scaling law of EHR foundation models pretrained on medical code sequences, pretrain compute-optimal foundation models of up to 2.4 billion parameters, and finetune them on clinician-curated cancer risk prediction cohorts. In our retrospective evaluation comprising of thirty thousand patients, CATCH-FM achieves strong efficacy, with 50% sensitivity in predicting first cancer risks at 99% specificity cutoff, and outperforming feature-based tree models and both general and medical LLMs by up to 20% AUPRC. Despite significant demographic, healthcare system, and EHR coding differences, CATCH-FM achieves state-of-the-art pancreatic cancer risk prediction on the EHRSHOT few-shot leaderboard, outperforming EHR foundation models pretrained using on-site patient data. Our analysis demonstrates the robustness of CATCH-FM in various patient distributions, the benefits of operating in the ICD code space, and its ability to capture non-trivial cancer risk factors. Our code will be open-sourced.
>
---
#### [replaced 083] ZeroTuning: Unlocking the Initial Token's Power to Enhance Large Language Models Without Training
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.11739v2](http://arxiv.org/pdf/2505.11739v2)**

> **作者:** Feijiang Han; Xiaodong Yu; Jianheng Tang; Delip Rao; Weihua Du; Lyle Ungar
>
> **摘要:** Token-level attention tuning, a class of training-free methods including Post-hoc Attention Steering (PASTA) and Attention Calibration (ACT), has emerged as a promising way to improve frozen LLMs with interpretable interventions. However, these methods depend on auxiliary heuristics to identify "important" task-specific tokens, which can introduce bias and limit applicability when token importance is unclear or when using optimized kernels where attention maps are inaccessible. We propose a simpler and more elegant alternative: acting only on the initial token (e.g., <BOS> in LLaMA). We show theoretically that adding lightweight biases to this token's attention logits monotonically controls the entropy of the downstream attention distribution - an effect amplified by its natural function as an attention sink. Our empirical analysis reveals that this tuning process can positively affect LLMs and better unlock their pretrained knowledge, with stronger effects in early layers and distinct scaling preferences across attention heads. Building on these insights, we introduce ZeroTuning: a training-free method that improves LLM performance by applying head-specific attention adjustments to the initial token, requiring zero parameter updates. We present two variants: a supervised mode that calibrates on validation examples, and a novel unsupervised mode that directly minimizes the model's output entropy. The method is lightweight, kernel-agnostic, and requires only four lines of modification to the standard LlamaAttention code. It achieves broad gains across 15 datasets and outperforms previous, more complex methods; for instance, with Llama-3.1-8B, it yields relative improvements of 19.9% on classification, 4.5% on question answering, and 2.1% on dialogue. ZeroTuning also works out-of-the-box with quantized inference and maintains its performance improvements with increasing context lengths.
>
---
#### [replaced 084] TrustJudge: Inconsistencies of LLM-as-a-Judge and How to Alleviate Them
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.21117v2](http://arxiv.org/pdf/2509.21117v2)**

> **作者:** Yidong Wang; Yunze Song; Tingyuan Zhu; Xuanwang Zhang; Zhuohao Yu; Hao Chen; Chiyu Song; Qiufeng Wang; Cunxiang Wang; Zhen Wu; Xinyu Dai; Yue Zhang; Wei Ye; Shikun Zhang
>
> **备注:** 22 pages, 9 figures, 6 tables
>
> **摘要:** The adoption of Large Language Models (LLMs) as automated evaluators (LLM-as-a-judge) has revealed critical inconsistencies in current evaluation frameworks. We identify two fundamental types of inconsistencies: (1) Score-Comparison Inconsistency, where lower-rated responses outperform higher-scored ones in pairwise comparisons, and (2) Pairwise Transitivity Inconsistency, manifested through circular preference chains (A>B>C>A) and equivalence contradictions (A=B=C\neq A). We argue that these issues come from information loss in discrete rating systems and ambiguous tie judgments during pairwise evaluation. We propose TrustJudge, a probabilistic framework that addresses these limitations through two key innovations: 1) distribution-sensitive scoring that computes continuous expectations from discrete rating probabilities, preserving information entropy for more precise scoring, and 2) likelihood-aware aggregation that resolves transitivity violations using bidirectional preference probabilities or perplexity. We also formalize the theoretical limitations of current LLM-as-a-judge frameworks and demonstrate how TrustJudge's components overcome them. When evaluated with Llama-3.1-70B-Instruct as judge using our dataset, TrustJudge reduces Score-Comparison inconsistency by 8.43% (from 23.32% to 14.89%) and Pairwise Transitivity inconsistency by 10.82% (from 15.22% to 4.40%), while maintaining higher evaluation accuracy. Our work provides the first systematic analysis of evaluation framework inconsistencies in LLM-as-a-judge paradigms, offering both theoretical insights and practical solutions for reliable automated assessment. The framework demonstrates consistent improvements across various model architectures and scales, enabling more trustworthy LLM evaluation without requiring additional training or human annotations. The codes can be found at https://github.com/TrustJudge/TrustJudge.
>
---
#### [replaced 085] Chain or tree? Re-evaluating complex reasoning from the perspective of a matrix of thought
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.03918v2](http://arxiv.org/pdf/2509.03918v2)**

> **作者:** Fengxiao Tang; Yufeng Li; Zongzong Wu; Ming Zhao
>
> **摘要:** Large Language Models (LLMs) face significant accuracy degradation due to insufficient reasoning ability when dealing with complex and abstract tasks. Thought structures such as Chain of Thought (CoT) and Tree of Thought (ToT) focus on enhancing the reasoning capability of LLMs. However, they suffer from inherent drawbacks such as redundancy within the same layer of the tree structure and the singularity of the paths in the chain structure. Some studies have utilized Retrieval-Augmented Generation (RAG) methods to enhance CoT and ToT in mitigating hallucinations in LLMs, yet the fundamental shortcomings of the thought structures still persist. Furthermore, when dealing with multi-entity and multi-hop information, the retrieved verification knowledge often contains large amounts of fragmented, superficial, or even erroneous data, misleading the reasoning process of LLMs. To address these issues, we propose the Matrix of Thought (MoT), a novel and efficient thought structure for LLMs. MoT explores problems in both horizontal and vertical dimensions through a "column-cell communication" mechanism, enabling LLMs to actively engage in multi-strategy and deep thinking while reducing redundancy in the thought nodes within the column cells, thereby enhancing the reasoning capability of LLMs. Additionally, through a fact-correction mechanism, it leverages the knowledge graph triples retrieved by RAG and the original text to construct knowledge units and correct erroneous answers. To validate the effectiveness of this method, we conducted extensive experiments in three tasks: 24-point game, question answering evaluation, and proposition writing.The results demonstrate that our framework outperforms state-of-the-art methods, with reasoning time only 14.4\% of that of the baseline method, proving its efficiency and accuracy. The code for framework is available at https://github.com/lyfiter/mtqa.
>
---
#### [replaced 086] Language-Specific Latent Process Hinders Cross-Lingual Performance
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.13141v3](http://arxiv.org/pdf/2505.13141v3)**

> **作者:** Zheng Wei Lim; Alham Fikri Aji; Trevor Cohn
>
> **摘要:** Large language models (LLMs) are demonstrably capable of cross-lingual transfer, but can produce inconsistent output when prompted with the same queries written in different languages. To understand how language models are able to generalize knowledge from one language to the others, we measure representation similarity between languages, and apply the logit lens to interpret the implicit steps taken by LLMs to solve multilingual multi-choice reasoning questions. Our analyses reveal LLMs predict inconsistently and are less accurate because they rely on representations that are dissimilar across languages, rather than working in a shared semantic space. While larger models are more multilingual, we show their hidden states are more likely to dissociate from the shared representation compared to smaller models, but are nevertheless more capable of retrieving knowledge embedded across different languages. Finally, we demonstrate that knowledge sharing in small models can be facilitated by steering their latent processing towards the shared semantic space. This improves the models' multilingual reasoning performance, as a result of more knowledge transfer from, and better output consistency with English.
>
---
#### [replaced 087] UniErase: Towards Balanced and Precise Unlearning in Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.15674v2](http://arxiv.org/pdf/2505.15674v2)**

> **作者:** Miao Yu; Liang Lin; Guibin Zhang; Xinfeng Li; Junfeng Fang; Xingrui Yu; Ivor Tsang; Ningyu Zhang; Kun Wang; Yang Wang
>
> **摘要:** Large language models (LLMs) require iterative updates to address the outdated information problem, where LLM unlearning offers an approach for selective removal. However, mainstream unlearning methods primarily rely on fine-tuning techniques, which often lack precision in targeted unlearning and struggle to balance unlearning efficacy with general ability under massive and sequential settings. To bridge this gap, in this work, we introduce UniErase, a novel unlearning framework that demonstrates precision and balanced performances between knowledge unlearning and ability retaining. We first propose the Unlearning Token, which is optimized to steer LLMs toward a forgetting space. To achieve concrete unlearning behaviors, we further introduce the lightweight Unlearning Edit to efficiently associate the unlearning targets with this meta-token. Serving as a new unlearning paradigm via editing, UniErase achieves outstanding performances across batch, sequential, and precise unlearning tasks under fictitious and real-world knowledge scenarios. On the TOFU benchmark, compared with 8 baselines, UniErase, modifying only $\sim$ \textbf{3.66%} of the LLM parameters, outperforms the previous best-forgetting baseline by \textbf{$\sim$ 4.01$\times$} for \textbf{model ability} with even higher unlearning efficacy. Similarly, UniErase, with better ability retention, also surpasses the previous best-retaining method by \textbf{35.96%} for \textbf{unlearning efficacy}, showing balanced and dual top-tier performances in the current unlearning community.
>
---
#### [replaced 088] Texture or Semantics? Vision-Language Models Get Lost in Font Recognition
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.23768v4](http://arxiv.org/pdf/2503.23768v4)**

> **作者:** Zhecheng Li; Guoxian Song; Yujun Cai; Zhen Xiong; Junsong Yuan; Yiwei Wang
>
> **备注:** Accepted to COLM 2025
>
> **摘要:** Modern Vision-Language Models (VLMs) exhibit remarkable visual and linguistic capabilities, achieving impressive performance in various tasks such as image recognition and object localization. However, their effectiveness in fine-grained tasks remains an open question. In everyday scenarios, individuals encountering design materials, such as magazines, typography tutorials, research papers, or branding content, may wish to identify aesthetically pleasing fonts used in the text. Given their multimodal capabilities and free accessibility, many VLMs are often considered potential tools for font recognition. This raises a fundamental question: Do VLMs truly possess the capability to recognize fonts? To investigate this, we introduce the Font Recognition Benchmark (FRB), a compact and well-structured dataset comprising 15 commonly used fonts. FRB includes two versions: (i) an easy version, where 10 sentences are rendered in different fonts, and (ii) a hard version, where each text sample consists of the names of the 15 fonts themselves, introducing a stroop effect that challenges model perception. Through extensive evaluation of various VLMs on font recognition tasks, we arrive at the following key findings: (i) Current VLMs exhibit limited font recognition capabilities, with many state-of-the-art models failing to achieve satisfactory performance and being easily affected by the stroop effect introduced by textual information. (ii) Few-shot learning and Chain-of-Thought (CoT) prompting provide minimal benefits in improving font recognition accuracy across different VLMs. (iii) Attention analysis sheds light on the inherent limitations of VLMs in capturing semantic features.
>
---
#### [replaced 089] Unifying Symbolic Music Arrangement: Track-Aware Reconstruction and Structured Tokenization
- **分类: cs.SD; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2408.15176v4](http://arxiv.org/pdf/2408.15176v4)**

> **作者:** Longshen Ou; Jingwei Zhao; Ziyu Wang; Gus Xia; Qihao Liang; Torin Hopkins Ye Wang
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** We present a unified framework for automatic multitrack music arrangement that enables a single pre-trained symbolic music model to handle diverse arrangement scenarios, including reinterpretation, simplification, and additive generation. At its core is a segment-level reconstruction objective operating on token-level disentangled content and style, allowing for flexible any-to-any instrumentation transformations at inference time. To support track-wise modeling, we introduce REMI-z, a structured tokenization scheme for multitrack symbolic music that enhances modeling efficiency and effectiveness for both arrangement tasks and unconditional generation. Our method outperforms task-specific state-of-the-art models on representative tasks in different arrangement scenarios -- band arrangement, piano reduction, and drum arrangement, in both objective metrics and perceptual evaluations. Taken together, our framework demonstrates strong generality and suggests broader applicability in symbolic music-to-music transformation.
>
---
#### [replaced 090] QA-LIGN: Aligning LLMs through Constitutionally Decomposed QA
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.08123v4](http://arxiv.org/pdf/2506.08123v4)**

> **作者:** Jacob Dineen; Aswin RRV; Qin Liu; Zhikun Xu; Xiao Ye; Ming Shen; Zhaonan Li; Shijie Lu; Chitta Baral; Muhao Chen; Ben Zhou
>
> **备注:** Accepted to Findings of EMNLP 2025
>
> **摘要:** Alignment of large language models (LLMs) with principles like helpfulness, honesty, and harmlessness typically relies on scalar rewards that obscure which objectives drive the training signal. We introduce QA-LIGN, which decomposes monolithic rewards into interpretable principle-specific evaluations through structured natural language programs. Models learn through a draft, critique, and revise pipeline, where symbolic evaluation against the rubrics provides transparent feedback for both initial and revised responses during GRPO training. Applied to uncensored Llama-3.1-8B-Instruct, QA-LIGN reduces attack success rates by up to 68.7% while maintaining a 0.67% false refusal rate, achieving Pareto optimal safety-helpfulness performance and outperforming both DPO and GRPO with state-of-the-art reward models given equivalent training. These results demonstrate that making reward signals interpretable and modular improves alignment effectiveness, suggesting transparency enhances LLM safety.
>
---
#### [replaced 091] DAMR: Efficient and Adaptive Context-Aware Knowledge Graph Question Answering with LLM-Guided MCTS
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.00719v4](http://arxiv.org/pdf/2508.00719v4)**

> **作者:** Yingxu Wang; Shiqi Fan; Mengzhu Wang; Siyang Gao; Chao Wang; Nan Yin
>
> **摘要:** Knowledge Graph Question Answering (KGQA) aims to interpret natural language queries and perform structured reasoning over knowledge graphs by leveraging their relational and semantic structures to retrieve accurate answers. Existing methods primarily follow either the retrieve-then-reason paradigm, which relies on Graph Neural Networks or heuristic rules to extract static candidate paths, or dynamic path generation strategies that employ LLMs with prompting to jointly perform retrieval and reasoning. However, the former lacks adaptability due to static path extraction and the absence of contextual refinement, while the latter suffers from high computational costs and limited evaluation accuracy because of their dependence on fixed scoring functions and repeated LLM calls. To address these issues, this paper proposes Dynamically Adaptive MCTS-based Reasoning (DAMR), a novel framework that integrates LLM-guided Monte Carlo Tree Search (MCTS) with adaptive path evaluation to enable efficient and context-aware KGQA. DAMR leverages MCTS as a backbone, where an LLM-based planner selects the top-$k$ semantically relevant relations at each expansion step to effectively reduce the search space. To enhance evaluation accuracy, we introduce a lightweight Transformer-based scorer that performs context-aware plausibility estimation by jointly encoding the question and relation sequence through cross-attention, thereby capturing fine-grained semantic shifts during multi-hop reasoning. Furthermore, to mitigate the scarcity of high-quality supervision, DAMR incorporates a dynamic pseudo-path refinement mechanism that periodically generates training signals from partial paths explored during search, enabling the scorer to continually adapt to the evolving distribution of reasoning trajectories. Extensive experiments on multiple KGQA benchmarks show that DAMR significantly outperforms SOTA methods.
>
---
#### [replaced 092] Elucidating Mechanisms of Demographic Bias in LLMs for Healthcare
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.13319v2](http://arxiv.org/pdf/2502.13319v2)**

> **作者:** Hiba Ahsan; Arnab Sen Sharma; Silvio Amir; David Bau; Byron C. Wallace
>
> **备注:** Accepted in EMNLP (Findings)
>
> **摘要:** We know from prior work that LLMs encode social biases, and that this manifests in clinical tasks. In this work we adopt tools from mechanistic interpretability to unveil sociodemographic representations and biases within LLMs in the context of healthcare. Specifically, we ask: Can we identify activations within LLMs that encode sociodemographic information (e.g., gender, race)? We find that gender information is highly localized in MLP layers and can be reliably manipulated at inference time via patching. Such interventions can surgically alter generated clinical vignettes for specific conditions, and also influence downstream clinical predictions which correlate with gender, e.g., patient risk of depression. We find that representation of patient race is somewhat more distributed, but can also be intervened upon, to a degree. To our knowledge, this is the first application of mechanistic interpretability methods to LLMs for healthcare.
>
---
#### [replaced 093] Modelling Analogies and Analogical Reasoning: Connecting Cognitive Science Theory and NLP Research
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.09381v2](http://arxiv.org/pdf/2509.09381v2)**

> **作者:** Molly R Petersen; Claire E Stevenson; Lonneke van der Plas
>
> **备注:** Accepted to Transactions of the Association for Computational Linguistics (TACL)
>
> **摘要:** Analogical reasoning is an essential aspect of human cognition. In this paper, we summarize key theory about the processes underlying analogical reasoning from the cognitive science literature and relate it to current research in natural language processing. While these processes can be easily linked to concepts in NLP, they are generally not viewed through a cognitive lens. Furthermore, we show how these notions are relevant for several major challenges in NLP research, not directly related to analogy solving. This may guide researchers to better optimize relational understanding in text, as opposed to relying heavily on entity-level similarity.
>
---
#### [replaced 094] Reasoning Under Uncertainty: Exploring Probabilistic Reasoning Capabilities of LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.10739v2](http://arxiv.org/pdf/2509.10739v2)**

> **作者:** Mobina Pournemat; Keivan Rezaei; Gaurang Sriramanan; Arman Zarei; Jiaxiang Fu; Yang Wang; Hamid Eghbalzadeh; Soheil Feizi
>
> **备注:** 27 pages, 4 figures
>
> **摘要:** Despite widespread success in language understanding and generation, large language models (LLMs) exhibit unclear and often inconsistent behavior when faced with tasks that require probabilistic reasoning. In this work, we present the first comprehensive study of the reasoning capabilities of LLMs over explicit discrete probability distributions. Given observations from a probability distribution, we evaluate models on three carefully designed tasks, mode identification, maximum likelihood estimation, and sample generation, by prompting them to provide responses to queries about either the joint distribution or its conditionals. These tasks thus probe a range of probabilistic skills, including frequency analysis, marginalization, and generative behavior. Through comprehensive empirical evaluations, we demonstrate that there exists a clear performance gap between smaller and larger models, with the latter demonstrating stronger inference and surprising capabilities in sample generation. Furthermore, our investigations reveal notable limitations, including sensitivity to variations in the notation utilized to represent probabilistic outcomes and performance degradation of over 60% as context length increases. Together, our results provide a detailed understanding of the probabilistic reasoning abilities of LLMs and identify key directions for future improvement.
>
---
#### [replaced 095] Taxonomy, Opportunities, and Challenges of Representation Engineering for Large Language Models
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.19649v4](http://arxiv.org/pdf/2502.19649v4)**

> **作者:** Jan Wehner; Sahar Abdelnabi; Daniel Tan; David Krueger; Mario Fritz
>
> **摘要:** Representation Engineering (RepE) is a novel paradigm for controlling the behavior of LLMs. Unlike traditional approaches that modify inputs or fine-tune the model, RepE directly manipulates the model's internal representations. As a result, it may offer more effective, interpretable, data-efficient, and flexible control over models' behavior. We present the first comprehensive survey of RepE for LLMs, reviewing the rapidly growing literature to address key questions: What RepE methods exist and how do they differ? For what concepts and problems has RepE been applied? What are the strengths and weaknesses of RepE compared to other methods? To answer these, we propose a unified framework describing RepE as a pipeline comprising representation identification, operationalization, and control. We posit that while RepE methods offer significant potential, challenges remain, including managing multiple concepts, ensuring reliability, and preserving models' performance. Towards improving RepE, we identify opportunities for experimental and methodological improvements and construct a guide for best practices.
>
---
#### [replaced 096] Labeling Free-text Data using Language Model Ensembles
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.08413v4](http://arxiv.org/pdf/2501.08413v4)**

> **作者:** Jiaxing Qiu; Dongliang Guo; Natalie Papini; Noelle Peace; Hannah F. Fitterman-Harris; Cheri A. Levinson; Tom Hartvigsen; Teague R. Henry
>
> **摘要:** Free-text responses are commonly collected in psychological studies, providing rich qualitative insights that quantitative measures may not capture. Labeling curated topics of research interest in free-text data by multiple trained human coders is typically labor-intensive and time-consuming. Though large language models (LLMs) excel in language processing, LLM-assisted labeling techniques relying on closed-source LLMs cannot be directly applied to free-text data, without explicit consent for external use. In this study, we propose a framework of assembling locally-deployable LLMs to enhance the labeling of predetermined topics in free-text data under privacy constraints. Analogous to annotation by multiple human raters, this framework leverages the heterogeneity of diverse open-source LLMs. The ensemble approach seeks a balance between the agreement and disagreement across LLMs, guided by a relevancy scoring methodology that utilizes embedding distances between topic descriptions and LLMs' reasoning. We evaluated the ensemble approach using both publicly accessible Reddit data from eating disorder related forums, and free-text responses from eating disorder patients, both complemented by human annotations. We found that: (1) there is heterogeneity in the performance of labeling among same-sized LLMs, with some showing low sensitivity but high precision, while others exhibit high sensitivity but low precision. (2) Compared to individual LLMs, the ensemble of LLMs achieved the highest accuracy and optimal precision-sensitivity trade-off in predicting human annotations. (3) The relevancy scores across LLMs showed greater agreement than dichotomous labels, indicating that the relevancy scoring method effectively mitigates the heterogeneity in LLMs' labeling.
>
---
#### [replaced 097] InfiMed: Low-Resource Medical MLLMs with Advancing Understanding and Reasoning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.23867v2](http://arxiv.org/pdf/2505.23867v2)**

> **作者:** Zeyu Liu; Zhitian Hou; Guanghao Zhu; Zhijie Sang; Congkai Xie; Hongxia Yang
>
> **摘要:** Multimodal Large Language Models (MLLMs) have achieved remarkable progress in domains such as visual understanding and mathematical reasoning. However, their application in the medical domain is constrained by two key challenges: (1) multimodal medical datasets are scarce and often contain sparse information, limiting reasoning depth; and (2) Reinforcement Learning with Verifiable Rewards (RLVR), though effective in general domains, cannot reliably improve model performance in the medical domain. To overcome these challenges, during the supervised fine-tuning (SFT) stage, we incorporate high-quality textual reasoning data and general multimodal data alongside multimodal medical data to efficiently enhance foundational medical capabilities and restore the base model's reasoning ability. Moreover, considering that there are some multimodal medical datasets with sparse information, we further synthesize reflective-pattern-injected chain-of-thought (CoT) in addition to general CoT samples, equipping the model with initial reflective reasoning capabilities that provide a structured foundation for subsequent RLVR training. Finally, we introduce our InfiMed-Series models, InfiMed-SFT-3B and InfiMed-RL-3B, both of which deliver state-of-the-art performance across seven multimodal medical benchmarks. Notably, InfiMed-RL-3B achieves an average accuracy of 59.2%, outperforming even larger models like InternVL3-8B, which achieves 57.3%. Specifically, during the SFT phase, we utilized 188K samples, while the RLVR phase incorporated 36K samples, demonstrating the efficacy of both training strategies in achieving superior performance. We also conducted a series of extensive experiments, which provide valuable insights that contribute to advancing the performance of MLLMs in medical scenarios.
>
---
#### [replaced 098] TokUR: Token-Level Uncertainty Estimation for Large Language Model Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.11737v3](http://arxiv.org/pdf/2505.11737v3)**

> **作者:** Tunyu Zhang; Haizhou Shi; Yibin Wang; Hengyi Wang; Xiaoxiao He; Zhuowei Li; Haoxian Chen; Ligong Han; Kai Xu; Huan Zhang; Dimitris Metaxas; Hao Wang
>
> **备注:** Preprint; Work in progress
>
> **摘要:** While Large Language Models (LLMs) have demonstrated impressive capabilities, their output quality remains inconsistent across various application scenarios, making it difficult to identify trustworthy responses, especially in complex tasks requiring multi-step reasoning. In this paper, we propose a Token-level Uncertainty estimation framework for Reasoning (TokUR) that enables LLMs to self-assess and self-improve their responses in mathematical reasoning. Specifically, we introduce low-rank random weight perturbation during LLM decoding to generate predictive distributions for token-level uncertainty estimation, and we aggregate these uncertainty quantities to capture the semantic uncertainty of generated responses. Experiments on mathematical reasoning datasets of varying difficulty demonstrate that TokUR exhibits a strong correlation with answer correctness and model robustness, and the uncertainty signals produced by TokUR can be leveraged to enhance the model's reasoning performance at test time. These results highlight the effectiveness of TokUR as a principled and scalable approach for improving the reliability and interpretability of LLMs in challenging reasoning tasks.
>
---
#### [replaced 099] Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.05257v2](http://arxiv.org/pdf/2507.05257v2)**

> **作者:** Yuanzhe Hu; Yu Wang; Julian McAuley
>
> **备注:** Y. Hu and Y. Wang contribute equally
>
> **摘要:** Recent benchmarks for Large Language Model (LLM) agents primarily focus on evaluating reasoning, planning, and execution capabilities, while another critical component-memory, encompassing how agents memorize, update, and retrieve long-term information-is under-evaluated due to the lack of benchmarks. We term agents with memory mechanisms as memory agents. In this paper, based on classic theories from memory science and cognitive science, we identify four core competencies essential for memory agents: accurate retrieval, test-time learning, long-range understanding, and selective forgetting. Existing benchmarks either rely on limited context lengths or are tailored for static, long-context settings like book-based QA, which do not reflect the interactive, multi-turn nature of memory agents that incrementally accumulate information. Moreover, no existing benchmarks cover all four competencies. We introduce MemoryAgentBench, a new benchmark specifically designed for memory agents. Our benchmark transforms existing long-context datasets and incorporates newly constructed datasets into a multi-turn format, effectively simulating the incremental information processing characteristic of memory agents. By carefully selecting and curating datasets, our benchmark provides comprehensive coverage of the four core memory competencies outlined above, thereby offering a systematic and challenging testbed for assessing memory quality. We evaluate a diverse set of memory agents, ranging from simple context-based and retrieval-augmented generation (RAG) systems to advanced agents with external memory modules and tool integration. Empirical results reveal that current methods fall short of mastering all four competencies, underscoring the need for further research into comprehensive memory mechanisms for LLM agents.
>
---
#### [replaced 100] Towards an AI Musician: Synthesizing Sheet Music Problems for Musical Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.04059v2](http://arxiv.org/pdf/2509.04059v2)**

> **作者:** Zhilin Wang; Zhe Yang; Yun Luo; Yafu Li; Xiaoye Qu; Ziqian Qiao; Haoran Zhang; Runzhe Zhan; Derek F. Wong; Jizhe Zhou; Yu Cheng
>
> **备注:** 34 pages
>
> **摘要:** Enhancing the ability of Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) to interpret sheet music is a crucial step toward building AI musicians. However, current research lacks both evaluation benchmarks and training data for sheet music reasoning. Inspired by mathematics, where simple operations yield infinite verifiable problems, we introduce a novel approach that treats core music theory rules, such as those governing beats and intervals, as programmatic functions to systematically synthesize a vast and diverse corpus of sheet music reasoning problems. This approach allows us to introduce a data synthesis framework that generates verifiable sheet music questions in both textual and visual modalities, leading to the Synthetic Sheet Music Reasoning Benchmark (SSMR-Bench) and a complementary training set. Evaluation results on SSMR-Bench highlight the key role reasoning plays in interpreting sheet music, while also pointing out the ongoing challenges in understanding sheet music in a visual format. By leveraging synthetic data for RLVR, all models show significant improvements on the SSMR-Bench. Additionally, they also demonstrate considerable advancements on previously established human-crafted benchmarks, such as MusicTheoryBench and the music subset of MMMU. Finally, our results show that the enhanced reasoning ability can also facilitate music composition.
>
---
#### [replaced 101] Prompting is not Enough: Exploring Knowledge Integration and Controllable Generation on Large Language Models
- **分类: cs.CL; cs.AI; 68P20; H.3.4; I.2.6**

- **链接: [http://arxiv.org/pdf/2505.19660v2](http://arxiv.org/pdf/2505.19660v2)**

> **作者:** Tingjia Shen; Hao Wang; Chuan Qin; Ruijun Sun; Yang Song; Defu Lian; Hengshu Zhu; Enhong Chen
>
> **备注:** 13 pages, 5 figures
>
> **摘要:** Open-domain question answering (OpenQA) represents a cornerstone in natural language processing (NLP), primarily focused on extracting answers from unstructured textual data. With the rapid advancements in Large Language Models (LLMs), LLM-based OpenQA methods have reaped the benefits of emergent understanding and answering capabilities enabled by massive parameters compared to traditional methods. However, most of these methods encounter two critical challenges: how to integrate knowledge into LLMs effectively and how to adaptively generate results with specific answer formats for various task situations. To address these challenges, we propose a novel framework named GenKI, which aims to improve the OpenQA performance by exploring Knowledge Integration and controllable Generation on LLMs simultaneously. Specifically, we first train a dense passage retrieval model to retrieve associated knowledge from a given knowledge base. Subsequently, we introduce a novel knowledge integration model that incorporates the retrieval knowledge into instructions during fine-tuning to intensify the model. Furthermore, to enable controllable generation in LLMs, we leverage a certain fine-tuned LLM and an ensemble based on text consistency incorporating all coherence, fluency, and answer format assurance. Finally, extensive experiments conducted on the TriviaQA, MSMARCO, and CMRC2018 datasets, featuring diverse answer formats, have demonstrated the effectiveness of GenKI with comparison of state-of-the-art baselines. Moreover, ablation studies have disclosed a linear relationship between the frequency of retrieved knowledge and the model's ability to recall knowledge accurately against the ground truth. Our code of GenKI is available at https://github.com/USTC-StarTeam/GenKI
>
---
#### [replaced 102] MLP Memory: A Retriever-Pretrained Memory for Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.01832v2](http://arxiv.org/pdf/2508.01832v2)**

> **作者:** Rubin Wei; Jiaqi Cao; Jiarui Wang; Jushi Kai; Qipeng Guo; Bowen Zhou; Zhouhan Lin
>
> **摘要:** Modern approaches to enhancing Large Language Models' factual accuracy and knowledge utilization face a fundamental trade-off: non-parametric retrieval-augmented generation (RAG) provides flexible access to external knowledge but suffers from high inference latency and shallow integration, while parametric fine-tuning methods like LoRA risk catastrophic forgetting and degraded general capabilities. In this work, we propose MLP Memory, a lightweight parametric module that learns to internalize retrieval patterns without explicit document access. By pretraining an MLP to imitate a $k$NN retriever's behavior on the entire pretraining dataset, we create a differentiable memory component that captures the benefits of retrieval-based knowledge access in a fully parametric form. Our architecture integrates this pretrained MLP Memory with Transformer decoders through simple probability interpolation, achieving 12.3\% relative improvement on five question-answering benchmarks and 5.2 points absolute gain across nine general NLP tasks, while reducing hallucinations by up to 10 points on HaluEval. Moreover, MLP Memory delivers 2.5$\times$ faster inference than RAG with superior accuracy. Our findings show that learning retrieval patterns parametrically bridges the gap between efficient inference and effective knowledge access, offering a practical alternative to both RAG and fine-tuning approaches.
>
---
#### [replaced 103] Constituency Parsing using LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2310.19462v3](http://arxiv.org/pdf/2310.19462v3)**

> **作者:** Xuefeng Bai; Jialong Wu; Yulong Chen; Zhongqing Wang; Kehai Chen; Min Zhang; Yue Zhang
>
> **备注:** Accepted at IEEE Transactions on Audio, Speech, and Language Processing (TASLP). See https://ieeexplore.ieee.org/document/11130901/ for the official version
>
> **摘要:** Constituency parsing is a fundamental yet unsolved challenge in natural language processing. In this paper, we examine the potential of recent large language models (LLMs) to address this challenge. We reformat constituency parsing as a sequence-to-sequence generation problem and evaluate the performance of a diverse range of LLMs under zero-shot, few-shot, and supervised fine-tuning learning paradigms. We observe that while LLMs achieve acceptable improvements, they still encounter substantial limitations, due to the absence of mechanisms to guarantee the validity and faithfulness of the generated constituent trees. Motivated by this observation, we propose two strategies to guide LLMs to generate more accurate constituent trees by learning from erroneous samples and refining outputs in a multi-agent collaboration way, respectively. The experimental results demonstrate that our methods effectively reduce the occurrence of invalid and unfaithful trees, thereby enhancing overall parsing performance and achieving promising results across different learning paradigms.
>
---
#### [replaced 104] Cost-Optimal Grouped-Query Attention for Long-Context Modeling
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.09579v3](http://arxiv.org/pdf/2503.09579v3)**

> **作者:** Yingfa Chen; Yutong Wu; Chenyang Song; Zhen Leng Thai; Xingyu Shen; Xu Han; Zhiyuan Liu; Maosong Sun
>
> **备注:** EMNLP 2025 Main
>
> **摘要:** Grouped-Query Attention (GQA) is a widely adopted strategy for reducing the computational cost of attention layers in large language models (LLMs). However, current GQA configurations are often suboptimal because they overlook how context length influences inference cost. Since inference cost grows with context length, the most cost-efficient GQA configuration should also vary accordingly. In this work, we analyze the relationship among context length, model size, GQA configuration, and model loss, and introduce two innovations: (1) we decouple the total head size from the hidden size, enabling more flexible control over attention FLOPs; and (2) we jointly optimize the model size and the GQA configuration to arrive at a better allocation of inference resources between attention layers and other components. Our analysis reveals that commonly used GQA configurations are highly suboptimal for long-context scenarios. More importantly, we propose a recipe for deriving cost-optimal GQA configurations. Our results show that for long-context scenarios, one should use fewer attention heads while scaling up model size. Configurations selected by our recipe can reduce both memory usage and FLOPs by more than 50% compared to Llama-3's GQA, with *no degradation in model capabilities*. Our findings offer valuable insights for designing efficient long-context LLMs. The code is available at https://www.github.com/THUNLP/cost-optimal-gqa .
>
---
#### [replaced 105] DOTA: Distributional Test-Time Adaptation of Vision-Language Models
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.HC**

- **链接: [http://arxiv.org/pdf/2409.19375v3](http://arxiv.org/pdf/2409.19375v3)**

> **作者:** Zongbo Han; Jialong Yang; Guangyu Wang; Junfan Li; Qianli Xu; Mike Zheng Shou; Changqing Zhang
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Vision-language foundation models (VLMs), such as CLIP, exhibit remarkable performance across a wide range of tasks. However, deploying these models can be unreliable when significant distribution gaps exist between training and test data, while fine-tuning for diverse scenarios is often costly. Cache-based test-time adapters offer an efficient alternative by storing representative test samples to guide subsequent classifications. Yet, these methods typically employ naive cache management with limited capacity, leading to severe catastrophic forgetting when samples are inevitably dropped during updates. In this paper, we propose DOTA (DistributiOnal Test-time Adaptation), a simple yet effective method addressing this limitation. Crucially, instead of merely memorizing individual test samples, DOTA continuously estimates the underlying distribution of the test data stream. Test-time posterior probabilities are then computed using these dynamically estimated distributions via Bayes' theorem for adaptation. This distribution-centric approach enables the model to continually learn and adapt to the deployment environment. Extensive experiments validate that DOTA significantly mitigates forgetting and achieves state-of-the-art performance compared to existing methods.
>
---
#### [replaced 106] WebExplorer: Explore and Evolve for Training Long-Horizon Web Agents
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.06501v3](http://arxiv.org/pdf/2509.06501v3)**

> **作者:** Junteng Liu; Yunji Li; Chi Zhang; Jingyang Li; Aili Chen; Ke Ji; Weiyu Cheng; Zijia Wu; Chengyu Du; Qidi Xu; Jiayuan Song; Zhengmao Zhu; Wenhu Chen; Pengyu Zhao; Junxian He
>
> **摘要:** The paradigm of Large Language Models (LLMs) has increasingly shifted toward agentic applications, where web browsing capabilities are fundamental for retrieving information from diverse online sources. However, existing open-source web agents either demonstrate limited information-seeking abilities on complex tasks or lack transparent implementations. In this work, we identify that the key challenge lies in the scarcity of challenging data for information seeking. To address this limitation, we introduce WebExplorer: a systematic data generation approach using model-based exploration and iterative, long-to-short query evolution. This method creates challenging query-answer pairs that require multi-step reasoning and complex web navigation. By leveraging our curated high-quality dataset, we successfully develop advanced web agent WebExplorer-8B through supervised fine-tuning followed by reinforcement learning. Our model supports 128K context length and up to 100 tool calling turns, enabling long-horizon problem solving. Across diverse information-seeking benchmarks, WebExplorer-8B achieves the state-of-the-art performance at its scale. Notably, as an 8B-sized model, WebExplorer-8B is able to effectively search over an average of 16 turns after RL training, achieving higher accuracy than WebSailor-72B on BrowseComp-en/zh and attaining the best performance among models up to 100B parameters on WebWalkerQA and FRAMES. Beyond these information-seeking tasks, our model also achieves strong generalization on the HLE benchmark even though it is only trained on knowledge-intensive QA data. These results highlight our approach as a practical path toward long-horizon web agents.
>
---
#### [replaced 107] GeoDANO: Geometric VLM with Domain Agnostic Vision Encoder
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11360v2](http://arxiv.org/pdf/2502.11360v2)**

> **作者:** Seunghyuk Cho; Zhenyue Qin; Yang Liu; Youngbin Choi; Seungbeom Lee; Dongwoo Kim
>
> **备注:** Accepted to EMNLP-Findings 2025
>
> **摘要:** We introduce GeoDANO, a geometric vision-language model (VLM) with a domain-agnostic vision encoder, for solving plane geometry problems. Although VLMs have been employed for solving geometry problems, their ability to recognize geometric features remains insufficiently analyzed. To address this gap, we propose a benchmark that evaluates the recognition of visual geometric features, including primitives such as dots and lines, and relations such as orthogonality. Our preliminary study shows that vision encoders often used in general-purpose VLMs, e.g., OpenCLIP, fail to detect these features and struggle to generalize across domains. To overcome the limitation, we develop GeoCLIP, a CLIP-based model trained on synthetic geometric diagram--caption pairs. Benchmark results show that GeoCLIP outperforms existing vision encoders in recognizing geometric features. We then propose our VLM, GeoDANO, which augments GeoCLIP with a domain adaptation strategy for unseen diagram styles. GeoDANO outperforms specialized methods for plane geometry problems and GPT-4o on MathVerse. The implementation is available at https://github.com/ml-postech/GeoDANO.
>
---
#### [replaced 108] Distribution-Aligned Decoding for Efficient LLM Task Adaptation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.15888v2](http://arxiv.org/pdf/2509.15888v2)**

> **作者:** Senkang Hu; Xudong Han; Jinqi Jiang; Yihang Tao; Zihan Fang; Yong Dai; Sam Tak Wu Kwong; Yuguang Fang
>
> **备注:** Accepted by NeurIPS'25
>
> **摘要:** Adapting billion-parameter language models to a downstream task is still costly, even with parameter-efficient fine-tuning (PEFT). We re-cast task adaptation as output-distribution alignment: the objective is to steer the output distribution toward the task distribution directly during decoding rather than indirectly through weight updates. Building on this view, we introduce Steering Vector Decoding (SVD), a lightweight, PEFT-compatible, and theoretically grounded method. We start with a short warm-start fine-tune and extract a task-aware steering vector from the Kullback-Leibler (KL) divergence gradient between the output distribution of the warm-started and pre-trained models. This steering vector is then used to guide the decoding process to steer the model's output distribution towards the task distribution. We theoretically prove that SVD is first-order equivalent to the gradient step of full fine-tuning and derive a globally optimal solution for the strength of the steering vector. Across three tasks and nine benchmarks, SVD paired with four standard PEFT methods improves multiple-choice accuracy by up to 5 points and open-ended truthfulness by 2 points, with similar gains (1-2 points) on commonsense datasets without adding trainable parameters beyond the PEFT adapter. SVD thus offers a lightweight, theoretically grounded path to stronger task adaptation for large language models.
>
---
#### [replaced 109] The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm
- **分类: cs.LG; cs.AI; cs.CL; cs.NA; math.NA; math.OC; 65F30, 68T07, 68N19; G.1.3; I.2.6; F.2.1; G.1.6**

- **链接: [http://arxiv.org/pdf/2505.16932v3](http://arxiv.org/pdf/2505.16932v3)**

> **作者:** Noah Amsel; David Persson; Christopher Musco; Robert M. Gower
>
> **备注:** 34 pages, 8 figures, 4 algorithms
>
> **摘要:** Computing the polar decomposition and the related matrix sign function has been a well-studied problem in numerical analysis for decades. Recently, it has emerged as an important subroutine within the Muon algorithm for training deep neural networks. However, the requirements of this application differ sharply from classical settings: deep learning demands GPU-friendly algorithms that prioritize high throughput over high precision. We introduce Polar Express, a new method for computing the polar decomposition. Like Newton-Schulz and other classical polynomial methods, our approach uses only matrix-matrix multiplications, making it very efficient on GPUs. Inspired by earlier work of Chen & Chow and Nakatsukasa & Freund, Polar Express adapts the update rule at each iteration by solving a minimax optimization problem. We prove that this strategy minimizes error in a worst-case sense, allowing Polar Express to converge as rapidly as possible both in the early iterations and asymptotically. We also address finite-precision issues, making it practical to use in bfloat16. When integrated into the Muon training framework, our method leads to consistent improvements in validation loss when training a GPT-2 model on one billion tokens from the FineWeb dataset, outperforming recent alternatives across a range of learning rates.
>
---
#### [replaced 110] On the Within-class Variation Issue in Alzheimer's Disease Detection
- **分类: eess.AS; cs.AI; cs.CL; cs.LG; cs.SD; q-bio.NC**

- **链接: [http://arxiv.org/pdf/2409.16322v3](http://arxiv.org/pdf/2409.16322v3)**

> **作者:** Jiawen Kang; Dongrui Han; Lingwei Meng; Jingyan Zhou; Jinchao Li; Xixin Wu; Helen Meng
>
> **备注:** Accepted for publication in Proc. of Interspeech 2025 conference. Note: this is an extended version of the conference paper, with an additional section included
>
> **摘要:** Alzheimer's Disease (AD) detection employs machine learning classification models to distinguish between individuals with AD and those without. Different from conventional classification tasks, we identify within-class variation as a critical challenge in AD detection: individuals with AD exhibit a spectrum of cognitive impairments. Therefore, simplistic binary AD classification may overlook two crucial aspects: within-class heterogeneity and instance-level imbalance. In this work, we found using a sample score estimator can generate sample-specific soft scores aligning with cognitive scores. We subsequently propose two simple yet effective methods: Soft Target Distillation (SoTD) and Instance-level Re-balancing (InRe), targeting two problems respectively. Based on the ADReSS and CU-MARVEL corpora, we demonstrated and analyzed the advantages of the proposed approaches in detection performance. These findings provide insights for developing robust and reliable AD detection models.
>
---
#### [replaced 111] BP-Seg: A graphical model approach to unsupervised and non-contiguous text segmentation using belief propagation
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.16965v2](http://arxiv.org/pdf/2505.16965v2)**

> **作者:** Fengyi Li; Kayhan Behdin; Natesh Pillai; Xiaofeng Wang; Zhipeng Wang; Ercan Yildiz
>
> **摘要:** Text segmentation based on the semantic meaning of sentences is a fundamental task with broad utility in many downstream applications. In this paper, we propose a graphical model-based unsupervised learning approach, named BP-Seg for efficient text segmentation. Our method not only considers local coherence, capturing the intuition that adjacent sentences are often more related, but also effectively groups sentences that are distant in the text yet semantically similar. This is achieved through belief propagation on the carefully constructed graphical models. Experimental results on both an illustrative example and a dataset with long-form documents demonstrate that our method performs favorably compared to competing approaches.
>
---
#### [replaced 112] Cross-Linguistic Analysis of Memory Load in Sentence Comprehension: Linear Distance and Structural Density
- **分类: cs.CL; q-bio.NC**

- **链接: [http://arxiv.org/pdf/2509.20916v2](http://arxiv.org/pdf/2509.20916v2)**

> **作者:** Krishna Aggarwal
>
> **备注:** 7 pages, 4 figures (Figure 2 has 3 sub-divisions)
>
> **摘要:** This study examines whether sentence-level memory load in comprehension is better explained by linear proximity between syntactically related words or by the structural density of the intervening material. Building on locality-based accounts and cross-linguistic evidence for dependency length minimization, the work advances Intervener Complexity-the number of intervening heads between a head and its dependent-as a structurally grounded lens that refines linear distance measures. Using harmonized dependency treebanks and a mixed-effects framework across multiple languages, the analysis jointly evaluates sentence length, dependency length, and Intervener Complexity as predictors of the Memory-load measure. Studies in Psycholinguistics have reported the contributions of feature interference and misbinding to memory load during processing. For this study, I operationalized sentence-level memory load as the linear sum of feature misbinding and feature interference for tractability; current evidence does not establish that their cognitive contributions combine additively. All three factors are positively associated with memory load, with sentence length exerting the broadest influence and Intervener Complexity offering explanatory power beyond linear distance. Conceptually, the findings reconcile linear and hierarchical perspectives on locality by treating dependency length as an important surface signature while identifying intervening heads as a more proximate indicator of integration and maintenance demands. Methodologically, the study illustrates how UD-based graph measures and cross-linguistic mixed-effects modelling can disentangle linear and structural contributions to processing efficiency, providing a principled path for evaluating competing theories of memory load in sentence comprehension.
>
---
#### [replaced 113] Rethinking the Embodied Gap in Vision-and-Language Navigation: A Holistic Study of Physical and Visual Disparities
- **分类: cs.RO; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.13019v2](http://arxiv.org/pdf/2507.13019v2)**

> **作者:** Liuyi Wang; Xinyuan Xia; Hui Zhao; Hanqing Wang; Tai Wang; Yilun Chen; Chengju Liu; Qijun Chen; Jiangmiao Pang
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Recent Vision-and-Language Navigation (VLN) advancements are promising, but their idealized assumptions about robot movement and control fail to reflect physically embodied deployment challenges. To bridge this gap, we introduce VLN-PE, a physically realistic VLN platform supporting humanoid, quadruped, and wheeled robots. For the first time, we systematically evaluate several ego-centric VLN methods in physical robotic settings across different technical pipelines, including classification models for single-step discrete action prediction, a diffusion model for dense waypoint prediction, and a train-free, map-based large language model (LLM) integrated with path planning. Our results reveal significant performance degradation due to limited robot observation space, environmental lighting variations, and physical challenges like collisions and falls. This also exposes locomotion constraints for legged robots in complex environments. VLN-PE is highly extensible, allowing seamless integration of new scenes beyond MP3D, thereby enabling more comprehensive VLN evaluation. Despite the weak generalization of current models in physical deployment, VLN-PE provides a new pathway for improving cross-embodiment's overall adaptability. We hope our findings and tools inspire the community to rethink VLN limitations and advance robust, practical VLN models. The code is available at https://crystalsixone.github.io/vln_pe.github.io/.
>
---
#### [replaced 114] RARE: Retrieval-Aware Robustness Evaluation for Retrieval-Augmented Generation Systems
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.00789v2](http://arxiv.org/pdf/2506.00789v2)**

> **作者:** Yixiao Zeng; Tianyu Cao; Danqing Wang; Xinran Zhao; Zimeng Qiu; Morteza Ziyadi; Tongshuang Wu; Lei Li
>
> **摘要:** Retrieval-Augmented Generation (RAG) enhances recency and factuality in answers. However, existing evaluations rarely test how well these systems cope with real-world noise, conflicting between internal and external retrieved contexts, or fast-changing facts. We introduce Retrieval-Aware Robustness Evaluation (RARE), a unified framework and large-scale benchmark that jointly stress-tests query and document perturbations over dynamic, time-sensitive corpora. One of the central features of RARE is a knowledge-graph-driven synthesis pipeline (RARE-Get) that automatically extracts single and multi-hop relations from the customized corpus and generates multi-level question sets without manual intervention. Leveraging this pipeline, we construct a dataset (RARE-Set) spanning 527 expert-level time-sensitive finance, economics, and policy documents and 48295 questions whose distribution evolves as the underlying sources change. To quantify resilience, we formalize retrieval-conditioned robustness metrics (RARE-Met) that capture a model's ability to remain correct or recover when queries, documents, or real-world retrieval results are systematically altered. Our findings reveal that RAG systems are unexpectedly sensitive to perturbations. Moreover, they consistently demonstrate lower robustness on multi-hop queries compared to single-hop queries across all domains.
>
---
#### [replaced 115] $A^2R^2$: Advancing Img2LaTeX Conversion via Visual Reasoning with Attention-Guided Refinement
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.20890v2](http://arxiv.org/pdf/2507.20890v2)**

> **作者:** Zhecheng Li; Guoxian Song; Yiwei Wang; Zhen Xiong; Junsong Yuan; Yujun Cai
>
> **摘要:** Img2LaTeX is a practically important task that involves translating mathematical expressions and structured visual content from images into LaTeX code. In recent years, vision-language models (VLMs) have achieved remarkable progress across a range of visual understanding tasks, largely due to their strong generalization capabilities. However, despite initial efforts to apply VLMs to the Img2LaTeX task, their performance remains suboptimal. Empirical evidence shows that VLMs can be challenged by fine-grained visual elements, such as subscripts and superscripts in mathematical expressions, which results in inaccurate LaTeX generation. To address this challenge, we propose $A^2R^2$: Advancing Img2LaTeX Conversion via Visual Reasoning with Attention-Guided Refinement, a framework that effectively integrates attention localization and iterative refinement within a visual reasoning framework, enabling VLMs to perform self-correction and progressively improve LaTeX generation quality. For effective evaluation, we introduce a new dataset, Img2LaTex-Hard-1K, consisting of 1,100 carefully curated and challenging examples designed to rigorously evaluate the capabilities of VLMs within this task domain. Extensive experimental results demonstrate that: (1) $A^2R^2$ significantly improves model performance across various evaluation metrics spanning both textual and visual levels; (2) Increasing the number of inference rounds yields notable performance gains, underscoring the potential of $A^2R^2$ in test-time scaling scenarios; (3) Ablation studies and further evaluations confirm the effectiveness of our approach and the synergy of its core components during inference.
>
---
#### [replaced 116] Follow the Path: Reasoning over Knowledge Graph Paths to Improve LLM Factuality
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.11140v2](http://arxiv.org/pdf/2505.11140v2)**

> **作者:** Mike Zhang; Johannes Bjerva; Russa Biswas
>
> **备注:** Updated version 26.9
>
> **摘要:** We introduce fs1, a simple yet effective method that improves the factuality of reasoning traces by sourcing them from large reasoning models (e.g., DeepSeek-R1) and grounding them by conditioning on knowledge graph (KG) paths. We fine-tune eight instruction-tuned Large Language Models (LLMs) on 3.9K factually grounded reasoning traces and rigorously evaluate them on six complex open-domain question-answering (QA) benchmarks encompassing 23.9K questions. Our results demonstrate that our fs1-tuned model (32B parameters) consistently outperforms instruction-tuned counterparts with parallel sampling by 6-14 absolute points (pass@$16$). Our detailed analysis shows that fs1 considerably improves model performance over more complex questions (requiring 3 or more hops on KG paths) and numerical answer types compared to the baselines. Furthermore, in single-pass inference, we notice that smaller LLMs show the most improvements. While prior works demonstrate the effectiveness of reasoning traces primarily in the STEM domains, our work shows strong evidence that anchoring reasoning to factual KG paths is a critical step in transforming LLMs for reliable knowledge-intensive tasks.
>
---
#### [replaced 117] Table-R1: Inference-Time Scaling for Table Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.23621v2](http://arxiv.org/pdf/2505.23621v2)**

> **作者:** Zheyuan Yang; Lyuhao Chen; Arman Cohan; Yilun Zhao
>
> **备注:** EMNLP 2025
>
> **摘要:** In this work, we present the first study to explore inference-time scaling on table reasoning tasks. We develop and evaluate two post-training strategies to enable inference-time scaling: distillation from frontier model reasoning traces and reinforcement learning with verifiable rewards (RLVR). For distillation, we introduce a large-scale dataset of reasoning traces generated by DeepSeek-R1, which we use to fine-tune LLMs into the Table-R1-SFT model. For RLVR, we propose task-specific verifiable reward functions and apply the GRPO algorithm to obtain the Table-R1-Zero model. We evaluate our Table-R1-series models across diverse table reasoning tasks, including short-form QA, fact verification, and free-form QA. Notably, the Table-R1-Zero model matches or exceeds the performance of GPT-4.1 and DeepSeek-R1, while using only a 7B-parameter LLM. It also demonstrates strong generalization to out-of-domain datasets. Extensive ablation and qualitative analyses reveal the benefits of instruction tuning, model architecture choices, and cross-task generalization, as well as emergence of essential table reasoning skills during RL training.
>
---
#### [replaced 118] Demystifying Multilingual Chain-of-Thought in Process Reward Modeling
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.12663v2](http://arxiv.org/pdf/2502.12663v2)**

> **作者:** Weixuan Wang; Minghao Wu; Barry Haddow; Alexandra Birch
>
> **摘要:** Large language models (LLMs) are designed to perform a wide range of tasks. To improve their ability to solve complex problems requiring multi-step reasoning, recent research leverages process reward modeling to provide fine-grained feedback at each step of the reasoning process for reinforcement learning (RL), but it predominantly focuses on English. In this paper, we tackle the critical challenge of extending process reward models (PRMs) to multilingual settings. To achieve this, we train multilingual PRMs on a dataset spanning seven languages, which is translated from English. Through comprehensive evaluations on two widely used reasoning benchmarks across 11 languages, we demonstrate that multilingual PRMs not only improve average accuracy but also reduce early-stage reasoning errors. Furthermore, our results highlight the sensitivity of multilingual PRMs to both the number of training languages and the volume of English data, while also uncovering the benefits arising from more candidate responses and trainable parameters. This work opens promising avenues for robust multilingual applications in complex, multi-step reasoning tasks. In addition, we release the code to foster research along this line.
>
---
#### [replaced 119] Wide-In, Narrow-Out: Revokable Decoding for Efficient and Effective DLLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.18578v2](http://arxiv.org/pdf/2507.18578v2)**

> **作者:** Feng Hong; Geng Yu; Yushi Ye; Haicheng Huang; Huangjie Zheng; Ya Zhang; Yanfeng Wang; Jiangchao Yao
>
> **摘要:** Diffusion Large Language Models (DLLMs) have emerged as a compelling alternative to Autoregressive models, designed for fast parallel generation. However, existing DLLMs are plagued by a severe quality-speed trade-off, where faster parallel decoding leads to significant performance degradation. We attribute this to the irreversibility of standard decoding in DLLMs, which is easily polarized into the wrong decoding direction along with early error context accumulation. To resolve this, we introduce Wide-In, Narrow-Out (WINO), a training-free decoding algorithm that enables revokable decoding in DLLMs. WINO employs a parallel draft-and-verify mechanism, aggressively drafting multiple tokens while simultaneously using the model's bidirectional context to verify and re-mask suspicious ones for refinement. Verified in open-source DLLMs like LLaDA and MMaDA, WINO is shown to decisively improve the quality-speed trade-off. For instance, on the GSM8K math benchmark, it accelerates inference by 6$\times$ while improving accuracy by 2.58%; on Flickr30K captioning, it achieves a 10$\times$ speedup with higher performance. More comprehensive experiments are conducted to demonstrate the superiority and provide an in-depth understanding of WINO.
>
---
#### [replaced 120] Resource Consumption Red-Teaming for Large Vision-Language Models
- **分类: cs.CR; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.18053v2](http://arxiv.org/pdf/2507.18053v2)**

> **作者:** Haoran Gao; Yuanhe Zhang; Zhenhong Zhou; Lei Jiang; Fanyu Meng; Yujia Xiao; Li Sun; Kun Wang; Yang Liu; Junlan Feng
>
> **摘要:** Resource Consumption Attacks (RCAs) have emerged as a significant threat to the deployment of Large Language Models (LLMs). With the integration of vision modalities, additional attack vectors exacerbate the risk of RCAs in large vision-language models (LVLMs). However, existing red-teaming studies have mainly overlooked visual inputs as a potential attack surface, resulting in insufficient mitigation strategies against RCAs in LVLMs. To address this gap, we propose RECITE ($\textbf{Re}$source $\textbf{C}$onsumpt$\textbf{i}$on Red-$\textbf{Te}$aming for LVLMs), the first approach for exploiting visual modalities to trigger unbounded RCAs red-teaming. First, we present $\textit{Vision Guided Optimization}$, a fine-grained pixel-level optimization to obtain \textit{Output Recall Objective} adversarial perturbations, which can induce repeating output. Then, we inject the perturbations into visual inputs, triggering unbounded generations to achieve the goal of RCAs. Empirical results demonstrate that RECITE increases service response latency by over 26 $\uparrow$, resulting in an additional 20\% increase in GPU utilization and memory consumption. Our study reveals security vulnerabilities in LVLMs and establishes a red-teaming framework that can facilitate the development of future defenses against RCAs.
>
---
#### [replaced 121] AtomR: Atomic Operator-Empowered Large Language Models for Heterogeneous Knowledge Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.16495v4](http://arxiv.org/pdf/2411.16495v4)**

> **作者:** Amy Xin; Jinxin Liu; Zijun Yao; Zhicheng Lee; Shulin Cao; Lei Hou; Juanzi Li
>
> **摘要:** Despite the outstanding capabilities of large language models (LLMs), knowledge-intensive reasoning still remains a challenging task due to LLMs' limitations in compositional reasoning and the hallucination problem. A prevalent solution is to employ chain-of-thought (CoT) with retrieval-augmented generation (RAG), which first formulates a reasoning plan by decomposing complex questions into simpler sub-questions, and then applies iterative RAG at each sub-question. However, prior works exhibit two crucial problems: inadequate reasoning planning and poor incorporation of heterogeneous knowledge. In this paper, we introduce AtomR, a framework for LLMs to conduct accurate heterogeneous knowledge reasoning at the atomic level. Inspired by how knowledge graph query languages model compositional reasoning through combining predefined operations, we propose three atomic knowledge operators, a unified set of operators for LLMs to retrieve and manipulate knowledge from heterogeneous sources. First, in the reasoning planning stage, AtomR decomposes a complex question into a reasoning tree where each leaf node corresponds to an atomic knowledge operator, achieving question decomposition that is highly fine-grained and orthogonal. Subsequently, in the reasoning execution stage, AtomR executes each atomic knowledge operator, which flexibly selects, retrieves, and operates atomic level knowledge from heterogeneous sources. We also introduce BlendQA, a challenging benchmark specially tailored for heterogeneous knowledge reasoning. Experiments on three single-source and two multi-source datasets show that AtomR outperforms state-of-the-art baselines by a large margin, with F1 score improvements of 9.4% on 2WikiMultihop and 9.5% on BlendQA. We release our code and datasets.
>
---
#### [replaced 122] Unveiling the Potential of Diffusion Large Language Model in Controllable Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.04504v2](http://arxiv.org/pdf/2507.04504v2)**

> **作者:** Zhen Xiong; Yujun Cai; Zhecheng Li; Yiwei Wang
>
> **摘要:** Controllable generation is a fundamental task in NLP with many applications, providing a basis for function calling to agentic communication. However, even state-of-the-art autoregressive Large Language Models (LLMs) today exhibit unreliability when required to generate structured output. Inspired by the current new diffusion-based large language models (dLLM), we realize that the architectural difference, especially the global information-sharing mechanism for language modeling, may be the key to unlock next-level controllable generation. To explore the possibility, we propose Self-adaptive Schema Scaffolding ($S^3$), a novel framework that enables dLLM to stably generate reliable structured outputs (e.g., JSON) by utilizing its innate reverse reasoning capability and global context awareness. $S^3$ initiates a schematic template directly in the output context as a starting state for dLLM, offering a more robust and general method than intricate prompt optimization. Experiments demonstrate that our method substantially unlocks the dLLM's potential in controllable generation in terms of structure adherence, content fidelity, and faithfulness. These results establish new perspectives and practical pathways for deploying language models in controllable generation tasks.
>
---
#### [replaced 123] Resisting Contextual Interference in RAG via Parametric-Knowledge Reinforcement
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2506.05154v2](http://arxiv.org/pdf/2506.05154v2)**

> **作者:** Chenyu Lin; Yilin Wen; Du Su; Hexiang Tan; Fei Sun; Muhan Chen; Chenfu Bao; Zhonghou Lyu
>
> **摘要:** Retrieval-augmented generation (RAG) improves performance on knowledge-intensive tasks but can be derailed by wrong, irrelevant, or conflicting retrieved text, causing models to rely on inaccurate evidence and cascade errors. We propose Knowledgeable-R1, a reinforcement-learning framework that explicitly trains large language models to use parametric knowledge (PK) to resist contextual interference while still exploiting external context when it is reliably helpful. Knowledgeable-R1 introduces a joint sampling scheme that generates paired responses with and without retrieval, and learns both local advantages (within each decoding regime) and global advantages under the same input to quantify when to ignore misleading context versus adopt it. We employ an asymmetric advantage transformation that amplifies exploratory behaviors toward parametric knowledge. Experiments show that \method significantly improves robustness and reasoning accuracy in knowledge conflict scenarios and general RAG scenarios, outperforming SOTA baselines by 23% in counterfactual scenarios, and without degradation when the retrieved context is fully accurate.Our code are available at https://github.com/lcy80366872/knowledgeable-R1.
>
---
#### [replaced 124] Security Degradation in Iterative AI Code Generation -- A Systematic Analysis of the Paradox
- **分类: cs.SE; cs.AI; cs.CL; cs.CR; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.11022v2](http://arxiv.org/pdf/2506.11022v2)**

> **作者:** Shivani Shukla; Himanshu Joshi; Romilla Syed
>
> **备注:** Keywords - Large Language Models, Security Vulnerabilities, AI-Generated Code, Iterative Feedback, Software Security, Secure Coding Practices, Feedback Loops, LLM Prompting Strategies
>
> **摘要:** The rapid adoption of Large Language Models(LLMs) for code generation has transformed software development, yet little attention has been given to how security vulnerabilities evolve through iterative LLM feedback. This paper analyzes security degradation in AI-generated code through a controlled experiment with 400 code samples across 40 rounds of "improvements" using four distinct prompting strategies. Our findings show a 37.6% increase in critical vulnerabilities after just five iterations, with distinct vulnerability patterns emerging across different prompting approaches. This evidence challenges the assumption that iterative LLM refinement improves code security and highlights the essential role of human expertise in the loop. We propose practical guidelines for developers to mitigate these risks, emphasizing the need for robust human validation between LLM iterations to prevent the paradoxical introduction of new security issues during supposedly beneficial code "improvements".
>
---
#### [replaced 125] Longitudinal and Multimodal Recording System to Capture Real-World Patient-Clinician Conversations for AI and Encounter Research: Protocol
- **分类: cs.CY; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.16378v2](http://arxiv.org/pdf/2509.16378v2)**

> **作者:** Misk Al Zahidy; Kerly Guevara Maldonado; Luis Vilatuna Andrango; Ana Cristina Proano; Ana Gabriela Claros; Maria Lizarazo Jimenez; David Toro-Tobon; Victor M. Montori; Oscar J. Ponce-Ponte; Juan P. Brito
>
> **备注:** 23 pages, 2 figures, 2 tables
>
> **摘要:** The promise of AI in medicine depends on learning from data that reflect what matters to patients and clinicians. Most existing models are trained on electronic health records (EHRs), which capture biological measures but rarely patient-clinician interactions. These relationships, central to care, unfold across voice, text, and video, yet remain absent from datasets. As a result, AI systems trained solely on EHRs risk perpetuating a narrow biomedical view of medicine and overlooking the lived exchanges that define clinical encounters. Our objective is to design, implement, and evaluate the feasibility of a longitudinal, multimodal system for capturing patient-clinician encounters, linking 360 degree video/audio recordings with surveys and EHR data to create a dataset for AI research. This single site study is in an academic outpatient endocrinology clinic at Mayo Clinic. Adult patients with in-person visits to participating clinicians are invited to enroll. Encounters are recorded with a 360 degree video camera. After each visit, patients complete a survey on empathy, satisfaction, pace, and treatment burden. Demographic and clinical data are extracted from the EHR. Feasibility is assessed using five endpoints: clinician consent, patient consent, recording success, survey completion, and data linkage across modalities. Recruitment began in January 2025. By August 2025, 35 of 36 eligible clinicians (97%) and 212 of 281 approached patients (75%) had consented. Of consented encounters, 162 (76%) had complete recordings and 204 (96%) completed the survey. This study aims to demonstrate the feasibility of a replicable framework for capturing the multimodal dynamics of patient-clinician encounters. By detailing workflows, endpoints, and ethical safeguards, it provides a template for longitudinal datasets and lays the foundation for AI models that incorporate the complexity of care.
>
---
#### [replaced 126] A Critical Look At Tokenwise Reward-Guided Text Generation
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2406.07780v3](http://arxiv.org/pdf/2406.07780v3)**

> **作者:** Ahmad Rashid; Ruotian Wu; Julia Grosse; Agustinus Kristiadi; Pascal Poupart
>
> **备注:** Work accepted at COLM 2025
>
> **摘要:** Large language models (LLMs) can be improved by aligning with human preferences through fine-tuning -- the so-called reinforcement learning from human feedback (RLHF). However, the cost of fine-tuning an LLM is prohibitive for many users. Due to their ability to bypass LLM fine-tuning, prediction-time tokenwise reward-guided text generation (RGTG) methods have recently been proposed. They use a reward model trained on full sequences to score partial sequences during decoding in a bid to steer the generation towards sequences with high rewards. However, these methods have so far been only heuristically motivated and poorly analyzed. In this work, we show that reward models trained on full sequences are not compatible with scoring partial sequences. To alleviate this, we propose to train a Bradley-Terry reward model on partial sequences explicitly, and autoregressively sample from the implied tokenwise policy during decoding. We study the properties of this reward model and the resulting policy: we show that this policy is proportional to the ratio of two distinct RLHF policies. Our simple approach outperforms previous RGTG methods and performs similarly to strong offline baselines without large-scale LLM fine-tuning. Code for our work is available at https://github.com/ahmadrash/PARGS
>
---
#### [replaced 127] LLM-OptiRA: LLM-Driven Optimization of Resource Allocation for Non-Convex Problems in Wireless Communications
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.02091v2](http://arxiv.org/pdf/2505.02091v2)**

> **作者:** Xinyue Peng; Yanming Liu; Yihan Cang; Chaoqun Cao; Ming Chen
>
> **备注:** 6 pages,4 figures
>
> **摘要:** Solving non-convex resource allocation problems poses significant challenges in wireless communication systems, often beyond the capability of traditional optimization techniques. To address this issue, we propose LLM-OptiRA, the first framework that leverages large language models (LLMs) to automatically detect and transform non-convex components into solvable forms, enabling fully automated resolution of non-convex resource allocation problems in wireless communication systems. LLM-OptiRA not only simplifies problem-solving by reducing reliance on expert knowledge, but also integrates error correction and feasibility validation mechanisms to ensure robustness. Experimental results show that LLM-OptiRA achieves an execution rate of 96% and a success rate of 80% on GPT-4, significantly outperforming baseline approaches in complex optimization tasks across diverse scenarios.
>
---
#### [replaced 128] Latent Concept Disentanglement in Transformer-based Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.16975v2](http://arxiv.org/pdf/2506.16975v2)**

> **作者:** Guan Zhe Hong; Bhavya Vasudeva; Vatsal Sharan; Cyrus Rashtchian; Prabhakar Raghavan; Rina Panigrahy
>
> **摘要:** When large language models (LLMs) use in-context learning (ICL) to solve a new task, they must infer latent concepts from demonstration examples. This raises the question of whether and how transformers represent latent structures as part of their computation. Our work experiments with several controlled tasks, studying this question using mechanistic interpretability. First, we show that in transitive reasoning tasks with a latent, discrete concept, the model successfully identifies the latent concept and does step-by-step concept composition. This builds upon prior work that analyzes single-step reasoning. Then, we consider tasks parameterized by a latent numerical concept. We discover low-dimensional subspaces in the model's representation space, where the geometry cleanly reflects the underlying parameterization. Overall, we show that small and large models can indeed disentangle and utilize latent concepts that they learn in-context from a handful of abbreviated demonstrations.
>
---
#### [replaced 129] WildSpeech-Bench: Benchmarking End-to-End SpeechLLMs in the Wild
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.21875v3](http://arxiv.org/pdf/2506.21875v3)**

> **作者:** Linhao Zhang; Jian Zhang; Bokai Lei; Chuhan Wu; Aiwei Liu; Wei Jia; Xiao Zhou
>
> **摘要:** Recent multi-modal Large Language Models (LLMs) such as GPT-4o have demonstrated strong capabilities of direct speech interaction. However, the lack of specialized and comprehensive benchmarks for end-to-end speech LLM evaluation hinders optimizing the user experience of Audio LLMs in real-world applications. Existing evaluation methods often adapt text-based benchmarks, overlooking speech's unique characteristics and challenges, including prosody, homophones, stuttering, and differing user expectations. Here, we introduce the first comprehensive benchmark designed to systematically evaluate end-to-end speechLLMs in practical speech conversations. We systematically curate real-world chat data relevant to spoken scenarios, introduce diversity in speaker attributes and acoustic conditions, and augment the dataset with speech-specific phenomena. We further design a query-aware evaluation method to use customized evaluation checklists and prompts to enhance the accuracy of automatic evaluation. We conduct comprehensive testing and detailed analysis of various mainstream speech models, revealing significant differences in model performance across different speech scenarios. The use of query-aware evaluation further enables a finer-grained assessment under various speech-specific scenarios. Our benchmark can provide valuable insights for speech model development and evaluation.
>
---
#### [replaced 130] EmoBench-UA: A Benchmark Dataset for Emotion Detection in Ukrainian
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.23297v2](http://arxiv.org/pdf/2505.23297v2)**

> **作者:** Daryna Dementieva; Nikolay Babakov; Alexander Fraser
>
> **备注:** EMNLP2025, Findings
>
> **摘要:** While Ukrainian NLP has seen progress in many texts processing tasks, emotion classification remains an underexplored area with no publicly available benchmark to date. In this work, we introduce EmoBench-UA, the first annotated dataset for emotion detection in Ukrainian texts. Our annotation schema is adapted from the previous English-centric works on emotion detection (Mohammad et al., 2018; Mohammad, 2022) guidelines. The dataset was created through crowdsourcing using the Toloka.ai platform ensuring high-quality of the annotation process. Then, we evaluate a range of approaches on the collected dataset, starting from linguistic-based baselines, synthetic data translated from English, to large language models (LLMs). Our findings highlight the challenges of emotion classification in non-mainstream languages like Ukrainian and emphasize the need for further development of Ukrainian-specific models and training resources.
>
---
#### [replaced 131] Learning to Summarize by Learning to Quiz: Adversarial Agentic Collaboration for Long Document Summarization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.20900v2](http://arxiv.org/pdf/2509.20900v2)**

> **作者:** Weixuan Wang; Minghao Wu; Barry Haddow; Alexandra Birch
>
> **摘要:** Long document summarization remains a significant challenge for current large language models (LLMs), as existing approaches commonly struggle with information loss, factual inconsistencies, and coherence issues when processing excessively long documents. We propose SummQ, a novel adversarial multi-agent framework that addresses these limitations through collaborative intelligence between specialized agents operating in two complementary domains: summarization and quizzing. Our approach employs summary generators and reviewers that work collaboratively to create and evaluate comprehensive summaries, while quiz generators and reviewers create comprehension questions that serve as continuous quality checks for the summarization process. This adversarial dynamic, enhanced by an examinee agent that validates whether the generated summary contains the information needed to answer the quiz questions, enables iterative refinement through multifaceted feedback mechanisms. We evaluate SummQ on three widely used long document summarization benchmarks. Experimental results demonstrate that our framework significantly outperforms existing state-of-the-art methods across ROUGE and BERTScore metrics, as well as in LLM-as-a-Judge and human evaluations. Our comprehensive analyses reveal the effectiveness of the multi-agent collaboration dynamics, the influence of different agent configurations, and the impact of the quizzing mechanism. This work establishes a new approach for long document summarization that uses adversarial agentic collaboration to improve summarization quality.
>
---
#### [replaced 132] video-SALMONN 2: Caption-Enhanced Audio-Visual Large Language Models
- **分类: cs.CV; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2506.15220v3](http://arxiv.org/pdf/2506.15220v3)**

> **作者:** Changli Tang; Yixuan Li; Yudong Yang; Jimin Zhuang; Guangzhi Sun; Wei Li; Zejun Ma; Chao Zhang
>
> **摘要:** We present video-SALMONN 2, a family of audio-visual large language models that set new state-of-the-art (SOTA) results in video description and question answering (QA). Our core contribution is multi-round direct preference optimisation (MrDPO), paired with a caption-quality objective that jointly rewards completeness and factual accuracy. Unlike standard DPO with a fixed reference policy, MrDPO periodically refreshes the reference by bootstrapping from a newly re-initialised lightweight adapter trained on the latest preferences, avoiding reference staleness and enabling continual improvement. This strategy produces captions that are consistently more detailed and accurate than those from proprietary systems such as GPT-4o and Gemini-1.5 Pro. We further distil these gains by using our model to generate a high-quality video-caption corpus for supervised fine-tuning of new models, transferring benefits beyond captioning to strong performance on complex video-QA tasks. Across widely used audio-visual and visual-only understanding benchmarks (including Video-MME, WorldSense, AVUT, Video-Holmes, DailyOmni, MLVU, and LVBench), our 3B and 7B models achieve SOTA results at comparable scales, while the 72B model surpasses all other open-source systems. Our source code, models, and data are released at \href{https://github.com/bytedance/video-SALMONN-2}{https://github.com/bytedance/video-SALMONN-2}.
>
---
#### [replaced 133] Semantic Reformulation Entropy for Robust Hallucination Detection in QA Tasks
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.17445v2](http://arxiv.org/pdf/2509.17445v2)**

> **作者:** Chaodong Tong; Qi Zhang; Lei Jiang; Yanbing Liu; Nannan Sun; Wei Li
>
> **备注:** 5pages, 5 figures, submitted to ICASSP 2026,
>
> **摘要:** Reliable question answering with large language models (LLMs) is challenged by hallucinations, fluent but factually incorrect outputs arising from epistemic uncertainty. Existing entropy-based semantic-level uncertainty estimation methods are limited by sampling noise and unstable clustering of variable-length answers. We propose Semantic Reformulation Entropy (SRE), which improves uncertainty estimation in two ways. First, input-side semantic reformulations produce faithful paraphrases, expand the estimation space, and reduce biases from superficial decoder tendencies. Second, progressive, energy-based hybrid clustering stabilizes semantic grouping. Experiments on SQuAD and TriviaQA show that SRE outperforms strong baselines, providing more robust and generalizable hallucination detection. These results demonstrate that combining input diversification with multi-signal clustering substantially enhances semantic-level uncertainty estimation.
>
---
#### [replaced 134] Comparing Uncertainty Measurement and Mitigation Methods for Large Language Models: A Systematic Review
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.18346v2](http://arxiv.org/pdf/2504.18346v2)**

> **作者:** Toghrul Abbasli; Kentaroh Toyoda; Yuan Wang; Leon Witt; Muhammad Asif Ali; Yukai Miao; Dan Li; Qingsong Wei
>
> **摘要:** Large Language Models (LLMs) have been transformative across many domains. However, hallucination -- confidently outputting incorrect information -- remains one of the leading challenges for LLMs. This raises the question of how to accurately assess and quantify the uncertainty of LLMs. Extensive literature on traditional models has explored Uncertainty Quantification (UQ) to measure uncertainty and employed calibration techniques to address the misalignment between uncertainty and accuracy. While some of these methods have been adapted for LLMs, the literature lacks an in-depth analysis of their effectiveness and does not offer a comprehensive benchmark to enable insightful comparison among existing solutions. In this work, we fill this gap via a systematic survey of representative prior works on UQ and calibration for LLMs and introduce a rigorous benchmark. Using two widely used reliability datasets, we empirically evaluate six related methods, which justify the significant findings of our review. Finally, we provide outlooks for key future directions and outline open challenges. To the best of our knowledge, this survey is the first dedicated study to review the calibration methods and relevant metrics for LLMs.
>
---
#### [replaced 135] Is GPT-OSS Good? A Comprehensive Evaluation of OpenAI's Latest Open Source Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.12461v2](http://arxiv.org/pdf/2508.12461v2)**

> **作者:** Ziqian Bi; Keyu Chen; Chiung-Yi Tseng; Danyang Zhang; Tianyang Wang; Hongying Luo; Lu Chen; Junming Huang; Jibin Guan; Junfeng Hao; Junhao Song
>
> **摘要:** In August 2025, OpenAI released GPT-OSS models, its first open weight large language models since GPT-2 in 2019, comprising two mixture of experts architectures with 120B and 20B parameters. We evaluated both variants against six contemporary open source large language models ranging from 14.7B to 235B parameters, representing both dense and sparse designs, across ten benchmarks covering general knowledge, mathematical reasoning, code generation, multilingual understanding, and conversational ability. All models were tested in unquantised form under standardised inference settings, with statistical validation using McNemars test and effect size analysis. Results show that gpt-oss-20B consistently outperforms gpt-oss-120B on several benchmarks, such as HumanEval and MMLU, despite requiring substantially less memory and energy per response. Both models demonstrate mid-tier overall performance within the current open source landscape, with relative strength in code generation and notable weaknesses in multilingual tasks. These findings provide empirical evidence that scaling in sparse architectures may not yield proportional performance gains, underscoring the need for further investigation into optimisation strategies and informing more efficient model selection for future open source deployments. More details and evaluation scripts are available at the \href{https://ai-agent-lab.github.io/gpt-oss}{Project Webpage}.
>
---
#### [replaced 136] MultiVox: A Benchmark for Evaluating Voice Assistants for Multimodal Interactions
- **分类: cs.MM; cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2507.10859v2](http://arxiv.org/pdf/2507.10859v2)**

> **作者:** Ramaneswaran Selvakumar; Ashish Seth; Nishit Anand; Utkarsh Tyagi; Sonal Kumar; Sreyan Ghosh; Dinesh Manocha
>
> **摘要:** The rapid progress of Large Language Models (LLMs) has empowered omni models to act as voice assistants capable of understanding spoken dialogues. These models can process multimodal inputs beyond text, such as speech and visual data, enabling more context-aware interactions. However, current benchmarks fall short in comprehensively evaluating how well these models generate context-aware responses, particularly when it comes to implicitly understanding fine-grained speech characteristics, such as pitch, emotion, timbre, and volume or the environmental acoustic context such as background sounds. Additionally, they inadequately assess the ability of models to align paralinguistic cues with complementary visual signals to inform their responses. To address these gaps, we introduce MultiVox, the first omni voice assistant benchmark designed to evaluate the ability of voice assistants to integrate spoken and visual cues including paralinguistic speech features for truly multimodal understanding. Specifically, MultiVox includes 1000 human-annotated and recorded speech dialogues that encompass diverse paralinguistic features and a range of visual cues such as images and videos. Our evaluation on 10 state-of-the-art models reveals that, although humans excel at these tasks, current models consistently struggle to produce contextually grounded responses.
>
---
#### [replaced 137] Cognitive Load Limits in Large Language Models: Benchmarking Multi-Hop Reasoning
- **分类: cs.AI; cs.CL; cs.LG; I.2.7; I.2.6**

- **链接: [http://arxiv.org/pdf/2509.19517v2](http://arxiv.org/pdf/2509.19517v2)**

> **作者:** Sai Teja Reddy Adapala
>
> **摘要:** The scaling of Large Language Models (LLMs) has exposed a critical gap between their performance on static benchmarks and their fragility in dynamic, information-rich environments. While models excel at isolated tasks, the computational limits that govern their reasoning under cognitive load remain poorly understood. In this work, we introduce a formal theory of computational cognitive load, positing that extraneous, task-irrelevant information (Context Saturation) and interference from task-switching (Attentional Residue) are key mechanisms that degrade performance. We designed the Interleaved Cognitive Evaluation (ICE), a deconfounded benchmark to systematically manipulate these load factors on challenging multi-hop reasoning tasks. A comprehensive study (N = 10 replications per item across 200 questions) revealed significant performance variations across five instruction-tuned models. Smaller open-source architectures (Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.2) exhibited baseline brittleness, achieving 0% accuracy (SEM = 0.0) across all conditions, including clean controls, on this high-intrinsic-load task. In contrast, Gemini-2.0-Flash-001 showed partial resilience, achieving 85% accuracy in control conditions, with a statistically significant degradation under context saturation ($\beta = -0.003$ per % load, $p < 0.001$). These findings provide preliminary evidence that cognitive load is a key contributor to reasoning failures, supporting theories of hallucination-as-guessing under uncertainty. We conclude that dynamic, cognitive-aware stress testing, as exemplified by the ICE benchmark, is essential for evaluating the true resilience and safety of advanced AI systems.
>
---
#### [replaced 138] Generator-Assistant Stepwise Rollback Framework for Large Language Model Agent
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.02519v4](http://arxiv.org/pdf/2503.02519v4)**

> **作者:** Xingzuo Li; Kehai Chen; Yunfei Long; Xuefeng Bai; Yong Xu; Min Zhang
>
> **备注:** EMNLP 2025 Main
>
> **摘要:** Large language model (LLM) agents typically adopt a step-by-step reasoning framework, in which they interleave the processes of thinking and acting to accomplish the given task. However, this paradigm faces a deep-rooted one-pass issue whereby each generated intermediate thought is plugged into the trajectory regardless of its correctness, which can cause irreversible error propagation. To address the issue, this paper proposes a novel framework called Generator-Assistant Stepwise Rollback (GA-Rollback) to induce better decision-making for LLM agents. Particularly, GA-Rollback utilizes a generator to interact with the environment and an assistant to examine each action produced by the generator, where the assistant triggers a rollback operation upon detection of incorrect actions. Moreover, we introduce two additional strategies tailored for the rollback scenario to further improve its effectiveness. Extensive experiments show that GA-Rollback achieves significant improvements over several strong baselines on three widely used benchmarks. Our analysis further reveals that GA-Rollback can function as a robust plug-and-play module, integrating seamlessly with other methods.
>
---
#### [replaced 139] KV Cache Steering for Controlling Frozen LLMs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.08799v2](http://arxiv.org/pdf/2507.08799v2)**

> **作者:** Max Belitsky; Dawid J. Kopiczko; Michael Dorkenwald; M. Jehanzeb Mirza; James R. Glass; Cees G. M. Snoek; Yuki M. Asano
>
> **摘要:** We propose cache steering, a lightweight method for implicit steering of language models via a one-shot intervention applied directly to the key-value cache. To validate its effectiveness, we apply cache steering to induce chain-of-thought reasoning in small language models. Our approach constructs steering vectors from reasoning traces, obtained either from teacher models (e.g., GPT-4o) or existing human annotations, that shift model behavior toward more explicit, multi-step reasoning without fine-tuning or prompt modifications. Experimental evaluations on diverse reasoning benchmarks demonstrate that cache steering improves both the qualitative structure of model reasoning and quantitative task performance. Additional experiments show that the method also scales to larger models and yields further gains on challenging datasets such as GPQA and MATH. Compared to prior activation steering techniques that require continuous interventions, our one-shot cache steering offers substantial advantages in terms of inference latency, hyperparameter stability, and ease of integration with existing inference APIs. Beyond mere reasoning induction, we show that cache steering enables controllable transfer of reasoning styles (e.g., stepwise, causal, analogical), making it a practical tool for behavior-level guidance of language models.
>
---
#### [replaced 140] GTPO and GRPO-S: Token and Sequence-Level Reward Shaping with Policy Entropy
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.04349v5](http://arxiv.org/pdf/2508.04349v5)**

> **作者:** Hongze Tan; Jianfei Pan; Jinghao Lin; Tao Chen; Zhihang Zheng; Zhihao Tang; Haihua Yang
>
> **摘要:** Reinforcement learning (RL) is a pivotal task for enhancing Large Language Model (LLM) reasoning. Conventional algorithms, however, typically adhere to a coarse-grained credit assignment paradigm, applying a uniform reward to all tokens in a sequence, a critical flaw in long-chain reasoning tasks. In this paper, we address this challenge and propose Dynamic Entropy Weighting, a novel mechanism that facilitates fine-grained rewards through two new algorithms: Group Token Policy Optimization (GTPO), which assigns an entropy-weighted reward to each token, and the analogous algorithm Sequence-Level GRPO (GRPO-S). Our approach is founded on the hypothesis that high policy entropy within a reasoning path is a powerful heuristic for cognitive effort at pivotal junctures, which can be repurposed into a learning signal. By repurposing policy entropy for reward shaping, we achieve true per-token credit assignment. Experimental results across challenging reasoning benchmarks validate the superiority of our approach, showing our methods significantly outperform a strong DAPO baseline and confirming our entropy-weighting mechanism as the key driver of this performance boost.
>
---
