# 自然语言处理 cs.CL

- **最新发布 156 篇**

- **更新 77 篇**

## 最新发布

#### [new 001] Investigating Political and Demographic Associations in Large Language Models Through Moral Foundations Theory
- **分类: cs.CL; cs.CY**

- **简介: 该论文研究大语言模型（LLM）在道德基础理论（MFT）框架下的政治与人口特征关联，探究其回应是否带有意识形态倾向。通过对比人类数据，分析LLM固有立场及在显式提示下模拟不同意识形态观点的准确性。**

- **链接: [http://arxiv.org/pdf/2510.13902v1](http://arxiv.org/pdf/2510.13902v1)**

> **作者:** Nicole Smith-Vaniz; Harper Lyon; Lorraine Steigner; Ben Armstrong; Nicholas Mattei
>
> **摘要:** Large Language Models (LLMs) have become increasingly incorporated into everyday life for many internet users, taking on significant roles as advice givers in the domains of medicine, personal relationships, and even legal matters. The importance of these roles raise questions about how and what responses LLMs make in difficult political and moral domains, especially questions about possible biases. To quantify the nature of potential biases in LLMs, various works have applied Moral Foundations Theory (MFT), a framework that categorizes human moral reasoning into five dimensions: Harm, Fairness, Ingroup Loyalty, Authority, and Purity. Previous research has used the MFT to measure differences in human participants along political, national, and cultural lines. While there has been some analysis of the responses of LLM with respect to political stance in role-playing scenarios, no work so far has directly assessed the moral leanings in the LLM responses, nor have they connected LLM outputs with robust human data. In this paper we analyze the distinctions between LLM MFT responses and existing human research directly, investigating whether commonly available LLM responses demonstrate ideological leanings: either through their inherent responses, straightforward representations of political ideologies, or when responding from the perspectives of constructed human personas. We assess whether LLMs inherently generate responses that align more closely with one political ideology over another, and additionally examine how accurately LLMs can represent ideological perspectives through both explicit prompting and demographic-based role-playing. By systematically analyzing LLM behavior across these conditions and experiments, our study provides insight into the extent of political and demographic dependency in AI-generated responses.
>
---
#### [new 002] Information Gain-based Policy Optimization: A Simple and Effective Approach for Multi-Turn LLM Agents
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对多轮LLM智能体训练中奖励稀疏导致的信用分配难问题，提出信息增益策略优化（IGPO），通过建模每步信息增益提供密集内在奖励，提升多轮任务的准确性和样本效率。**

- **链接: [http://arxiv.org/pdf/2510.14967v1](http://arxiv.org/pdf/2510.14967v1)**

> **作者:** Guoqing Wang; Sunhao Dai; Guangze Ye; Zeyu Gan; Wei Yao; Yong Deng; Xiaofeng Wu; Zhenzhe Ying
>
> **摘要:** Large language model (LLM)-based agents are increasingly trained with reinforcement learning (RL) to enhance their ability to interact with external environments through tool use, particularly in search-based settings that require multi-turn reasoning and knowledge acquisition. However, existing approaches typically rely on outcome-based rewards that are only provided at the final answer. This reward sparsity becomes particularly problematic in multi-turn settings, where long trajectories exacerbate two critical issues: (i) advantage collapse, where all rollouts receive identical rewards and provide no useful learning signals, and (ii) lack of fine-grained credit assignment, where dependencies between turns are obscured, especially in long-horizon tasks. In this paper, we propose Information Gain-based Policy Optimization (IGPO), a simple yet effective RL framework that provides dense and intrinsic supervision for multi-turn agent training. IGPO models each interaction turn as an incremental process of acquiring information about the ground truth, and defines turn-level rewards as the marginal increase in the policy's probability of producing the correct answer. Unlike prior process-level reward approaches that depend on external reward models or costly Monte Carlo estimation, IGPO derives intrinsic rewards directly from the model's own belief updates. These intrinsic turn-level rewards are combined with outcome-level supervision to form dense reward trajectories. Extensive experiments on both in-domain and out-of-domain benchmarks demonstrate that IGPO consistently outperforms strong baselines in multi-turn scenarios, achieving higher accuracy and improved sample efficiency.
>
---
#### [new 003] Code-driven Number Sequence Calculation: Enhancing the inductive Reasoning Abilities of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦大模型的归纳推理能力提升，针对现有数据模式单一和训练缺乏精细思维过程的问题，提出CodeSeq数据集，通过算法化数列生成通用项任务，结合迭代修正与强化学习，增强模型自主推理与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.14620v1](http://arxiv.org/pdf/2510.14620v1)**

> **作者:** Kedi Chen; Zhikai Lei; Xu Guo; Xuecheng Wu; Siyuan Zeng; Jianghao Yin; Yinqi Zhang; Qin Chen; Jie Zhou; Liang He; Qipeng Guo; Kai Chen; Wei Zhang
>
> **摘要:** Large language models (LLMs) make remarkable progress in reasoning tasks. Among different reasoning modes, inductive reasoning, due to its better alignment with human learning, attracts increasing interest. However, research on inductive reasoning faces certain challenges. First, existing inductive data mostly focuses on superficial regularities while lacking more complex internal patterns. Second, current works merely prompt LLMs or finetune on simple prompt-response pairs, but do not provide precise thinking processes nor implement difficulty control. Unlike previous work, we address these challenges by introducing \textit{CodeSeq}, a synthetic post-training dataset built from number sequences. We package number sequences into algorithmic problems to discover their general terms, defining a general term generation (GTG) task correspondingly. Our pipeline generates supervised finetuning data by reflecting on failed test cases and incorporating iterative corrections, thereby teaching LLMs to learn autonomous case generation and self-checking. Additionally, it leverages reinforcement learning with a novel Case-Synergy Solvability Scaling Reward based on both solvability, estimated from the problem pass rate, and the success rate of self-directed case generation, enabling models to learn more effectively from both successes and failures. Experimental results show that the models trained with \textit{CodeSeq} improve on various reasoning tasks and can preserve the models' OOD performance.
>
---
#### [new 004] Order from Chaos: Comparative Study of Ten Leading LLMs on Unstructured Data Categorization
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型在无结构文本分类任务中的表现，旨在解决现有模型准确率低、幻觉和类别膨胀问题。作者评估了十个主流LLM，并提出基于多模型协同的集成方法，显著提升性能，表明协同优于单纯扩大模型规模。**

- **链接: [http://arxiv.org/pdf/2510.13885v1](http://arxiv.org/pdf/2510.13885v1)**

> **作者:** Ariel Kamen
>
> **备注:** 10 pages, 4 figures,
>
> **摘要:** This study presents a comparative evaluation of ten state-of-the-art large language models (LLMs) applied to unstructured text categorization using the Interactive Advertising Bureau (IAB) 2.2 hierarchical taxonomy. The analysis employed a uniform dataset of 8,660 human-annotated samples and identical zero-shot prompts to ensure methodological consistency across all models. Evaluation metrics included four classic measures - accuracy, precision, recall, and F1-score - and three LLM-specific indicators: hallucination ratio, inflation ratio, and categorization cost. Results show that, despite their rapid advancement, contemporary LLMs achieve only moderate classic performance, with average scores of 34% accuracy, 42% precision, 45% recall, and 41% F1-score. Hallucination and inflation ratios reveal that models frequently overproduce categories relative to human annotators. Among the evaluated systems, Gemini 1.5/2.0 Flash and GPT 20B/120B offered the most favorable cost-to-performance balance, while GPT 120B demonstrated the lowest hallucination ratio. The findings suggest that scaling and architectural improvements alone do not ensure better categorization accuracy, as the task requires compressing rich unstructured text into a limited taxonomy - a process that challenges current model architectures. To address these limitations, a separate ensemble-based approach was developed and tested. The ensemble method, in which multiple LLMs act as independent experts, substantially improved accuracy, reduced inflation, and completely eliminated hallucinations. These results indicate that coordinated orchestration of models - rather than sheer scale - may represent the most effective path toward achieving or surpassing human-expert performance in large-scale text categorization.
>
---
#### [new 005] Interpreting the Latent Structure of Operator Precedence in Language Models
- **分类: cs.CL**

- **简介: 该论文研究大语言模型如何内部处理算术运算优先级。基于LLaMA 3.2-3B，构建算术数据集，结合可解释性方法，发现模型在残差流中编码中间结果，且操作符嵌入线性表示优先级，并提出通过嵌入维度交换修改优先级的方法。**

- **链接: [http://arxiv.org/pdf/2510.13908v1](http://arxiv.org/pdf/2510.13908v1)**

> **作者:** Dharunish Yugeswardeenoo; Harshil Nukala; Cole Blondin; Sean O Brien; Vasu Sharma; Kevin Zhu
>
> **备注:** 9 pages, 4 figures. Accepted to INTERPLAY Workshop at COLM 2025
>
> **摘要:** Large Language Models (LLMs) have demonstrated impressive reasoning capabilities but continue to struggle with arithmetic tasks. Prior works largely focus on outputs or prompting strategies, leaving the open question of the internal structure through which models do arithmetic computation. In this work, we investigate whether LLMs encode operator precedence in their internal representations via the open-source instruction-tuned LLaMA 3.2-3B model. We constructed a dataset of arithmetic expressions with three operands and two operators, varying the order and placement of parentheses. Using this dataset, we trace whether intermediate results appear in the residual stream of the instruction-tuned LLaMA 3.2-3B model. We apply interpretability techniques such as logit lens, linear classification probes, and UMAP geometric visualization. Our results show that intermediate computations are present in the residual stream, particularly after MLP blocks. We also find that the model linearly encodes precedence in each operator's embeddings post attention layer. We introduce partial embedding swap, a technique that modifies operator precedence by exchanging high-impact embedding dimensions between operators.
>
---
#### [new 006] EvoEdit: Evolving Null-space Alignment for Robust and Efficient Knowledge Editing
- **分类: cs.CL; cs.LG**

- **简介: 该论文针对大语言模型的知识编辑任务，解决连续编辑中的灾难性干扰问题。提出EvoEdit方法，通过序列化零空间对齐保持历史知识，实现高效稳定的编辑，兼顾性能与速度。**

- **链接: [http://arxiv.org/pdf/2510.13851v1](http://arxiv.org/pdf/2510.13851v1)**

> **作者:** Sicheng Lyu; Yu Gu; Xinyu Wang; Jerry Huang; Sitao Luan; Yufei Cui; Xiao-Wen Chang; Peng Lu
>
> **摘要:** Large language models (LLMs) require continual updates to rectify outdated or erroneous knowledge. Model editing has emerged as a compelling paradigm for introducing targeted modifications without the computational burden of full retraining. Existing approaches are mainly based on a locate-then-edit framework. However, in sequential editing contexts, where multiple updates are applied over time, they exhibit significant limitations and suffer from catastrophic interference, i.e., new edits compromise previously integrated updates and degrade preserved knowledge. To address these challenges, we introduce EvoEdit, a novel editing strategy that mitigates catastrophic interference through sequential null-space alignment, enabling stable and efficient model editing. By performing sequential null-space alignment for each incoming edit, EvoEdit preserves both original and previously modified knowledge representations and maintains output invariance on preserved knowledge even across long edit sequences, effectively mitigating interference. Evaluations on real-world sequential knowledge-editing benchmarks show that EvoEdit achieves better or comparable performance than prior state-of-the-art locate-then-edit techniques, with up to 3.53 times speedup. Overall, these results underscore the necessity of developing more principled approaches for designing LLMs in dynamically evolving information settings, while providing a simple yet effective solution with strong theoretical guarantees.
>
---
#### [new 007] Rewiring Experts on the Fly:Continuous Rerouting for Better Online Adaptation in Mixture-of-Expert models
- **分类: cs.CL**

- **简介: 该论文针对MoE模型在部署中因分布偏移导致的路由不佳问题，提出一种无需外部数据、在线自适应重路由方法。通过生成序列自监督，动态优化路由，提升推理性能，兼具高效性与即插即用特性。**

- **链接: [http://arxiv.org/pdf/2510.14853v1](http://arxiv.org/pdf/2510.14853v1)**

> **作者:** Guinan Su; Yanwu Yang; Li Shen; Lu Yin; Shiwei Liu; Jonas Geiping
>
> **摘要:** Mixture-of-Experts (MoE) models achieve efficient scaling through sparse expert activation, but often suffer from suboptimal routing decisions due to distribution shifts in deployment. While existing test-time adaptation methods could potentially address these issues, they primarily focus on dense models and require access to external data, limiting their practical applicability to MoE architectures. However, we find that, instead of relying on reference data, we can optimize MoE expert selection on-the-fly based only on input context. As such, we propose \textit{a data-free, online test-time framework} that continuously adapts MoE routing decisions during text generation without external supervision or data. Our method cycles between two phases: During the prefill stage, and later in regular intervals, we optimize the routing decisions of the model using self-supervision based on the already generated sequence. Then, we generate text as normal, maintaining the modified router until the next adaption. We implement this through lightweight additive vectors that only update router logits in selected layers, maintaining computational efficiency while preventing over-adaptation. The experimental results show consistent performance gains on challenging reasoning tasks while maintaining robustness to context shifts. For example, our method achieves a 5.5\% improvement on HumanEval with OLMoE. Furthermore, owing to its plug-and-play property, our method naturally complements existing test-time scaling techniques, e.g., achieving 6\% average gains when incorporated with self-consistency on DeepSeek-V2-Lite.
>
---
#### [new 008] Supervised Fine-Tuning or Contrastive Learning? Towards Better Multimodal LLM Reranking
- **分类: cs.CL; cs.CV; cs.IR**

- **简介: 该论文研究多模态大语言模型（LLM）重排序任务，比较监督微调（SFT）与对比学习（CL）的效果。通过分解训练目标并分析权重与方向作用，发现SFT因更强的权重机制更优，最终在MRB基准上取得新SOTA。**

- **链接: [http://arxiv.org/pdf/2510.14824v1](http://arxiv.org/pdf/2510.14824v1)**

> **作者:** Ziqi Dai; Xin Zhang; Mingxin Li; Yanzhao Zhang; Dingkun Long; Pengjun Xie; Meishan Zhang; Wenjie Li; Min Zhang
>
> **摘要:** In information retrieval, training reranking models mainly focuses on two types of objectives: metric learning (e.g. contrastive loss to increase the predicted scores on relevant query-document pairs) and classification (binary label prediction of relevance vs. irrelevance). For BERT-style encoders, various studies have shown that contrastive learning (CL) can be more effective than discriminative (classification) learning. However, for large language models (LLMs), classification via supervised fine-tuning (SFT), which predicts ''yes'' (resp. ''no'') token for relevant (resp. irrelevant) pairs, appears more promising as it aligns well with the generative nature of LLMs. This divergence raises a central question: which objective is intrinsically better suited to LLM-based reranking, and what mechanism underlies the difference? In this work, we conduct a comprehensive comparison and analysis between CL and SFT for reranking, taking the universal multimodal retrieval (UMR) as the experimental playground. We first decompose the objectives into two components: weight, which controls the magnitude of those updates, and direction, which guides the model updates, then present a unified framework for understanding their interactions. Through probing experiments, we find that SFT provides a substantially stronger weighting scheme than CL, whereas the preferred scoring direction shows no clear winner. Taken together, these results point to a consistent advantage of SFT over CL for LLM reranking. To further validate our findings, we conduct large-scale training with SFT and present new state-of-the-art rerankers on the MRB benchmark. We also provide ablations on SFT settings and expect our findings to benefit future research and applications in this area.
>
---
#### [new 009] Schema for In-Context Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大模型在上下文学习中缺乏抽象知识迁移的问题，提出基于认知科学“图式理论”的SA-ICL框架，通过显式构建推理步骤的抽象图式来增强模型推理能力，提升性能并减少对示例数量的依赖。**

- **链接: [http://arxiv.org/pdf/2510.13905v1](http://arxiv.org/pdf/2510.13905v1)**

> **作者:** Pan Chen; Shaohong Chen; Mark Wang; Shi Xuan Leong; Priscilla Fung; Varinia Bernales; Alan Aspuru-Guzik
>
> **摘要:** In-Context Learning (ICL) enables transformer-based language models to adapt to new tasks by conditioning on demonstration examples. However, traditional example-driven in-context learning lacks explicit modules for knowledge retrieval and transfer at the abstraction level. Inspired by cognitive science, specifically schema theory, which holds that humans interpret new information by activating pre-existing mental frameworks (schemas) to structure understanding, we introduce SCHEMA ACTIVATED IN CONTEXT LEARNING (SA-ICL). This framework extracts the representation of the building blocks of cognition for the reasoning process instilled from prior examples, creating an abstracted schema, a lightweight, structured template of key inferential steps and their relationships, which is then used to augment a model's reasoning process when presented with a novel question. We demonstrate that a broad range of large language models (LLMs) lack the capacity to form and utilize internal schema-based learning representations implicitly, but instead benefit significantly from explicit schema-based scaffolding. Across chemistry and physics questions from the GPQA dataset, our experiments show that SA-ICL consistently boosts performance, up to 36.19 percent, when the single demonstration example is of high quality, which simultaneously reduces reliance on the number of demonstrations and enhances interpretability. SCHEMA ACTIVATED IN CONTEXT LEARNING not only bridges disparate ICL strategies ranging from pattern priming to Chain-of-Thought prompting, but also paves a new path for enhancing human-like reasoning in LLMs.
>
---
#### [new 010] COIG-Writer: A High-Quality Dataset for Chinese Creative Writing with Thought Processes
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对中文创意写作中缺乏过程监督数据的问题，构建了包含思维过程的高质量数据集COIG-Writer。提出创意写作需逻辑架构与语言表达协同，验证过程监督的有效性、文化特异性及词汇多样性与质量的负相关。**

- **链接: [http://arxiv.org/pdf/2510.14763v1](http://arxiv.org/pdf/2510.14763v1)**

> **作者:** Yunwen Li; Shuangshuang Ying; Xingwei Qu; Xin Li; Sheng Jin; Minghao Liu; Zhoufutu Wen; Tianyu Zheng; Xeron Du; Qiguang Chen; Jiajun Shi; Wangchunshu Zhou; Jiazhan Feng; Wanjun Zhong; Libo Qin; Stephen Huang; Wanxiang Che; Chenghua Lin; Eli Zhang
>
> **摘要:** Large language models exhibit systematic deficiencies in creative writing, particularly in non-English contexts where training data is scarce and lacks process-level supervision. We present COIG-Writer, a novel Chinese creative writing dataset that captures both diverse outputs and their underlying thought processes through systematic reverse-engineering of high-quality texts. Unlike existing datasets that provide only input-output pairs, COIG-Writer comprises 1,665 meticulously curated triplets spanning 51 genres, each containing: (1) a reverse-engineered prompt, (2) detailed creative reasoning documenting decision-making processes, and (3) the final text. Through comprehensive experiments, we identify a two-component model of creative writing: narrative logic (provided by process supervision) and linguistic expression (maintained by general-purpose data). Our findings reveal three critical insights: (1) Process supervision is highly effective but requires stabilization with general data. A ratio of at least one creative sample to twelve general samples is needed to achieve optimal performance; below this threshold, the win rate progressively degrades (from 62.75% down to 35.78%)., (2) creative capabilities are culturally-bound with no cross-lingual transfer (89.26pp gap between Chinese and English performance), and (3) lexical diversity inversely correlates with creative quality (TTR paradox), suggesting high diversity signals compensatory behavior for logical deficiencies. These findings establish that creative excellence emerges from the interaction between logical scaffolding and linguistic grounding, analogous to how mathematical reasoning enhances but cannot replace linguistic competence in foundation models.
>
---
#### [new 011] Finding Answers in Thought Matters: Revisiting Evaluation on Large Language Models with Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型推理任务中的答案提取问题，发现不同提取方法显著影响性能。为此提出“答案再生”框架，通过额外推理步骤重生成答案，提升评估的鲁棒性与准确性，适用于数学与开放问答任务。**

- **链接: [http://arxiv.org/pdf/2510.14773v1](http://arxiv.org/pdf/2510.14773v1)**

> **作者:** Hwiyeol Jo; Joosung Lee; Jaehone Lee; Sang-Woo Lee; Joonsuk Park; Kang Min Yoo
>
> **备注:** ARR Submitted
>
> **摘要:** Evaluating generative models, such as large language models (LLMs), commonly involves question-answering tasks where the final answer is selected based on probability of answer choices. On the other hand, for models requiring reasoning, the method of answer extraction plays a critical role. Our research reveals that the performance of reasoning models and their final answer distributions are highly sensitive to the answer extraction algorithm employed. In order to mitigate this, we propose a basic framework: Answer Regeneration. The method uses an additional model inference, providing the prior input and output prefaced by the prompt "Answer:". The final answer is then selected or extracted from the regenerated output. We show that this extraction-rule-agnostic approach exhibits improved performance and enhanced robustness. Furthermore, we have applied this framework to general math problems and open-ended question answering tasks. Our analysis and this framework could offer a more reliable results for model evaluation.
>
---
#### [new 012] LiteStage: Latency-aware Layer Skipping for Multi-stage Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对多阶段推理中延迟高的问题，提出LiteStage框架，通过分阶段层跳过和在线置信度早退机制，在降低延迟的同时保持精度，实现高效推理。**

- **链接: [http://arxiv.org/pdf/2510.14211v1](http://arxiv.org/pdf/2510.14211v1)**

> **作者:** Beomseok Kang; Jiwon Song; Jae-Joon Kim
>
> **摘要:** Multi-stage reasoning has emerged as an effective strategy for enhancing the reasoning capability of small language models by decomposing complex problems into sequential sub-stages. However, this comes at the cost of increased latency. We observe that existing adaptive acceleration techniques, such as layer skipping, struggle to balance efficiency and accuracy in this setting due to two key challenges: (1) stage-wise variation in skip sensitivity, and (2) the generation of redundant output tokens. To address these, we propose LiteStage, a latency-aware layer skipping framework for multi-stage reasoning. LiteStage combines a stage-wise offline search that allocates optimal layer budgets with an online confidence-based generation early exit to suppress unnecessary decoding. Experiments on three benchmarks, e.g., OBQA, CSQA, and StrategyQA, show that LiteStage achieves up to 1.70x speedup with less than 4.0% accuracy loss, outperforming prior training-free layer skipping methods.
>
---
#### [new 013] LiRA: Linguistic Robust Anchoring for Cross-lingual Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦跨语言大模型在低资源语言下的性能不足问题，提出LiRA框架，通过锚定表示学习（Arca）和语言感知推理（LaSR）提升跨语言理解、检索与推理能力，并发布多语言商品检索数据集。**

- **链接: [http://arxiv.org/pdf/2510.14466v1](http://arxiv.org/pdf/2510.14466v1)**

> **作者:** Haolin Li; Haipeng Zhang; Mang Li; Yaohua Wang; Lijie Wen; Yu Zhang; Biqing Huang
>
> **摘要:** As large language models (LLMs) rapidly advance, performance on high-resource languages (e.g., English, Chinese) is nearing saturation, yet remains substantially lower for low-resource languages (e.g., Urdu, Thai) due to limited training data, machine-translation noise, and unstable cross-lingual alignment. We introduce LiRA (Linguistic Robust Anchoring for Large Language Models), a training framework that robustly improves cross-lingual representations under low-resource conditions while jointly strengthening retrieval and reasoning. LiRA comprises two modules: (i) Arca (Anchored Representation Composition Architecture), which anchors low-resource languages to an English semantic space via anchor-based alignment and multi-agent collaborative encoding, preserving geometric stability in a shared embedding space; and (ii) LaSR (Language-coupled Semantic Reasoner), which adds a language-aware lightweight reasoning head with consistency regularization on top of Arca's multilingual representations, unifying the training objective to enhance cross-lingual understanding, retrieval, and reasoning robustness. We further construct and release a multilingual product retrieval dataset covering five Southeast Asian and two South Asian languages. Experiments across low-resource benchmarks (cross-lingual retrieval, semantic similarity, and reasoning) show consistent gains and robustness under few-shot and noise-amplified settings; ablations validate the contribution of both Arca and LaSR. Code will be released on GitHub and the dataset on Hugging Face.
>
---
#### [new 014] Informed Routing in LLMs: Smarter Token-Level Computation for Faster Inference
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型推理成本高的问题，提出“知情路由”机制，通过预测模块评估 token 的重要性与可恢复性，实现执行或近似的灵活决策，在保持性能的同时显著降低计算开销。**

- **链接: [http://arxiv.org/pdf/2510.13831v1](http://arxiv.org/pdf/2510.13831v1)**

> **作者:** Chao Han; Yijuan Liang; Zihao Xuan; Daokuan Wu; Wei Zhang; Xiaoyu Shen
>
> **摘要:** The deployment of large language models (LLMs) in real-world applications is increasingly limited by their high inference cost. While recent advances in dynamic token-level computation allocation attempt to improve efficiency by selectively activating model components per token, existing methods rely on greedy routing--a myopic execute-or-skip mechanism that often leads to irreversible information loss and suboptimal token selection. This paper introduces informed routing, a new paradigm that proactively addresses these issues. The key insight is to assess not only a token's immediate importance but also its recoverability, i.e., how well its transformation can be approximated. To this end, we propose the Lightweight Feature Forecaster (LFF), a small predictive module that estimates a unit's output before routing decisions are made. This enables a flexible execute-or-approximate policy that preserves model fidelity while drastically reducing computation. Extensive experiments on both language modeling and reasoning tasks show that informed routing achieves state-of-the-art efficiency-performance trade-offs across multiple sparsity levels. Notably, even without final LoRA fine-tuning, our method matches or surpasses strong baselines that require full fine-tuning, all while reducing training time by over 50%. The code is available at: https://github.com/EIT-NLP/informed-routing
>
---
#### [new 015] Quantifying Phonosemantic Iconicity Distributionally in 6 Languages
- **分类: cs.CL**

- **简介: 该论文研究语音与语义间的系统性关联（即语音象征性），在6种语言中采用分布法大规模量化分析音义相似空间的对齐，揭示新现象并验证已有假设，探讨语言中音义关系的普遍性与差异。**

- **链接: [http://arxiv.org/pdf/2510.14040v1](http://arxiv.org/pdf/2510.14040v1)**

> **作者:** George Flint; Kaustubh Kislay
>
> **备注:** 7 pages, 2 figures, under review -- ACL (AACL 2025)
>
> **摘要:** Language is, as commonly theorized, largely arbitrary. Yet, systematic relationships between phonetics and semantics have been observed in many specific cases. To what degree could those systematic relationships manifest themselves in large scale, quantitative investigations--both in previously identified and unidentified phenomena? This work undertakes a distributional approach to quantifying phonosemantic iconicity at scale across 6 diverse languages (English, Spanish, Hindi, Finnish, Turkish, and Tamil). In each language, we analyze the alignment of morphemes' phonetic and semantic similarity spaces with a suite of statistical measures, and discover an array of interpretable phonosemantic alignments not previously identified in the literature, along with crosslinguistic patterns. We also analyze 5 previously hypothesized phonosemantic alignments, finding support for some such alignments and mixed results for others.
>
---
#### [new 016] The German Commons - 154 Billion Tokens of Openly Licensed Text for German Language Models
- **分类: cs.CL**

- **简介: 该论文旨在解决德语大模型训练中缺乏开放授权文本的问题，构建了含1545.6亿token的高质量、可合法使用的德语语料库“German Commons”，覆盖七大领域，确保数据可复现与扩展，推动开源德语语言模型发展。**

- **链接: [http://arxiv.org/pdf/2510.13996v1](http://arxiv.org/pdf/2510.13996v1)**

> **作者:** Lukas Gienapp; Christopher Schröder; Stefan Schweter; Christopher Akiki; Ferdinand Schlatt; Arden Zimmermann; Phillipe Genêt; Martin Potthast
>
> **备注:** 13 pages, 3 figures, 12 tables, includes datasheet
>
> **摘要:** Large language model development relies on large-scale training corpora, yet most contain data of unclear licensing status, limiting the development of truly open models. This problem is exacerbated for non-English languages, where openly licensed text remains critically scarce. We introduce the German Commons, the largest collection of openly licensed German text to date. It compiles data from 41 sources across seven domains, encompassing legal, scientific, cultural, political, news, economic, and web text. Through systematic sourcing from established data providers with verifiable licensing, it yields 154.56 billion tokens of high-quality text for language model training. Our processing pipeline implements comprehensive quality filtering, deduplication, and text formatting fixes, ensuring consistent quality across heterogeneous text sources. All domain subsets feature licenses of at least CC-BY-SA 4.0 or equivalent, ensuring legal compliance for model training and redistribution. The German Commons therefore addresses the critical gap in openly licensed German pretraining data, and enables the development of truly open German language models. We also release code for corpus construction and data filtering tailored to German language text, rendering the German Commons fully reproducible and extensible.
>
---
#### [new 017] Readers Prefer Outputs of AI Trained on Copyrighted Books over Expert Human Writers
- **分类: cs.CL; cs.CY**

- **简介: 该论文研究AI模仿作家风格生成文本的质量，比较专家作家与AI（如ChatGPT）在风格忠实度和写作质量上的表现。通过细调模型，AI输出更受读者青睐，且难以被检测，显著降低成本，涉及版权合理使用问题。**

- **链接: [http://arxiv.org/pdf/2510.13939v1](http://arxiv.org/pdf/2510.13939v1)**

> **作者:** Tuhin Chakrabarty; Jane C. Ginsburg; Paramveer Dhillon
>
> **备注:** Preprint Under Review
>
> **摘要:** The use of copyrighted books for training AI models has led to numerous lawsuits from authors concerned about AI's ability to generate derivative content.Yet it's unclear whether these models can generate high quality literary text while emulating authors' styles. To answer this we conducted a preregistered study comparing MFA-trained expert writers with three frontier AI models: ChatGPT, Claude & Gemini in writing up to 450 word excerpts emulating 50 award-winning authors' diverse styles. In blind pairwise evaluations by 159 representative expert & lay readers, AI-generated text from in-context prompting was strongly disfavored by experts for both stylistic fidelity (OR=0.16, p<10^8) & writing quality (OR=0.13, p<10^7) but showed mixed results with lay readers. However, fine-tuning ChatGPT on individual authors' complete works completely reversed these findings: experts now favored AI-generated text for stylistic fidelity (OR=8.16, p<10^13) & writing quality (OR=1.87, p=0.010), with lay readers showing similar shifts. These effects generalize across authors & styles. The fine-tuned outputs were rarely flagged as AI-generated (3% rate v. 97% for in-context prompting) by best AI detectors. Mediation analysis shows this reversal occurs because fine-tuning eliminates detectable AI stylistic quirks (e.g., cliche density) that penalize in-context outputs. While we do not account for additional costs of human effort required to transform raw AI output into cohesive, publishable prose, the median fine-tuning & inference cost of $81 per author represents a dramatic 99.7% reduction compared to typical professional writer compensation. Author-specific fine-tuning thus enables non-verbatim AI writing that readers prefer to expert human writing, providing empirical evidence directly relevant to copyright's fourth fair-use factor, the "effect upon the potential market or value" of the source works.
>
---
#### [new 018] CRaFT: An Explanation-Based Framework for Evaluating Cultural Reasoning in Multilingual Language Models
- **分类: cs.CL**

- **简介: 该论文提出CRaFT框架，旨在评估多语言大模型在文化语境下的推理能力。通过分析模型对文化相关问题的解释，从文化流利性、偏离度、一致性和语言适应性四个维度进行评测，揭示语言与文化理解的关系，推动构建更具文化适应性的语言模型。**

- **链接: [http://arxiv.org/pdf/2510.14014v1](http://arxiv.org/pdf/2510.14014v1)**

> **作者:** Shehenaz Hossain; Haithem Afli
>
> **摘要:** Correct answers do not necessarily reflect cultural understanding. We introduce CRaFT, an explanation-based multilingual evaluation framework designed to assess how large language models (LLMs) reason across cultural contexts. Rather than scoring outputs solely based on accuracy, CRaFT evaluates model explanations using four interpretable metrics: Cultural Fluency, Deviation, Consistency, and Linguistic Adaptation. We apply the framework to 50 culturally grounded questions from the World Values Survey, translated into Arabic, Bengali, and Spanish, and evaluate three models (GPT, DeepSeek, and FANAR) across over 2,100 answer-explanation pairs. Results reveal significant cross-lingual variation in reasoning: Arabic reduces fluency, Bengali enhances it, and Spanish remains largely stable. While GPT adapts more effectively across languages, it exhibits lower consistency; FANAR shows stable but rigid reasoning. These findings suggest that cultural awareness in LLMs is not intrinsic but emerges through linguistic framing. CRaFT offers a new lens for evaluating cross-cultural reasoning in multilingual settings, providing actionable insights for building culturally adaptive language models.
>
---
#### [new 019] MathMist: A Parallel Multilingual Benchmark Dataset for Mathematical Problem Solving and Reasoning
- **分类: cs.CL**

- **简介: 该论文聚焦多语言数学推理任务，旨在解决现有基准数据集语言覆盖不足的问题。作者构建了包含七种语言的平行数据集MathMist，涵盖2.1万对问题与答案，并评估多种大模型在跨语言数学推理中的表现，揭示其在低资源语言中的局限性。**

- **链接: [http://arxiv.org/pdf/2510.14305v1](http://arxiv.org/pdf/2510.14305v1)**

> **作者:** Mahbub E Sobhani; Md. Faiyaz Abdullah Sayeedi; Tasnim Mohiuddin; Md Mofijul Islam; Swakkhar Shatabda
>
> **摘要:** Mathematical reasoning remains one of the most challenging domains for large language models (LLMs), requiring not only linguistic understanding but also structured logical deduction and numerical precision. While recent LLMs demonstrate strong general-purpose reasoning abilities, their mathematical competence across diverse languages remains underexplored. Existing benchmarks primarily focus on English or a narrow subset of high-resource languages, leaving significant gaps in assessing multilingual and cross-lingual mathematical reasoning. To address this, we introduce MathMist, a parallel multilingual benchmark for mathematical problem solving and reasoning. MathMist encompasses over 21K aligned question-answer pairs across seven languages, representing a balanced coverage of high-, medium-, and low-resource linguistic settings. The dataset captures linguistic variety, multiple types of problem settings, and solution synthesizing capabilities. We systematically evaluate a diverse suite of models, including open-source small and medium LLMs, proprietary systems, and multilingual-reasoning-focused models, under zero-shot, chain-of-thought (CoT), and code-switched reasoning paradigms. Our results reveal persistent deficiencies in LLMs' ability to perform consistent and interpretable mathematical reasoning across languages, with pronounced degradation in low-resource settings. All the codes and data are available at GitHub: https://github.com/mahbubhimel/MathMist
>
---
#### [new 020] Element2Vec: Build Chemical Element Representation from Text for Property Prediction
- **分类: cs.CL**

- **简介: 该论文提出Element2Vec，旨在通过自然语言生成化学元素的向量表示，解决元素属性难以测量且数据稀疏的问题。结合全局与局部语义向量，引入测试时自注意力训练方法，提升属性预测准确性，推动材料科学的AI驱动发现。**

- **链接: [http://arxiv.org/pdf/2510.13916v1](http://arxiv.org/pdf/2510.13916v1)**

> **作者:** Yuanhao Li; Keyuan Lai; Tianqi Wang; Qihao Liu; Jiawei Ma; Yuan-Chao Hu
>
> **摘要:** Accurate property data for chemical elements is crucial for materials design and manufacturing, but many of them are difficult to measure directly due to equipment constraints. While traditional methods use the properties of other elements or related properties for prediction via numerical analyses, they often fail to model complex relationships. After all, not all characteristics can be represented as scalars. Recent efforts have been made to explore advanced AI tools such as language models for property estimation, but they still suffer from hallucinations and a lack of interpretability. In this paper, we investigate Element2Vecto effectively represent chemical elements from natural languages to support research in the natural sciences. Given the text parsed from Wikipedia pages, we use language models to generate both a single general-purpose embedding (Global) and a set of attribute-highlighted vectors (Local). Despite the complicated relationship across elements, the computational challenges also exist because of 1) the discrepancy in text distribution between common descriptions and specialized scientific texts, and 2) the extremely limited data, i.e., with only 118 known elements, data for specific properties is often highly sparse and incomplete. Thus, we also design a test-time training method based on self-attention to mitigate the prediction error caused by Vanilla regression clearly. We hope this work could pave the way for advancing AI-driven discovery in materials science.
>
---
#### [new 021] Constraint-Driven Small Language Models Based on Agent and OpenAlex Knowledge Graph: Mining Conceptual Pathways and Discovering Innovation Points in Academic Papers
- **分类: cs.CL; cs.LG; I.2.7**

- **简介: 该论文属学术分析任务，旨在发现论文中的创新点。针对现有方法缺乏深度概念关联分析的问题，提出基于OpenAlex知识图谱与小语言模型的框架，通过代理机制和提示工程挖掘关键概念路径，实现精准创新点识别。**

- **链接: [http://arxiv.org/pdf/2510.14303v1](http://arxiv.org/pdf/2510.14303v1)**

> **作者:** Ziye Xia; Sergei S. Ospichev
>
> **备注:** 9 pages, 10 figures
>
> **摘要:** In recent years, the rapid increase in academic publications across various fields has posed severe challenges for academic paper analysis: scientists struggle to timely and comprehensively track the latest research findings and methodologies. Key concept extraction has proven to be an effective analytical paradigm, and its automation has been achieved with the widespread application of language models in industrial and scientific domains. However, existing paper databases are mostly limited to similarity matching and basic classification of key concepts, failing to deeply explore the relational networks between concepts. This paper is based on the OpenAlex opensource knowledge graph. By analyzing nearly 8,000 open-source paper data from Novosibirsk State University, we discovered a strong correlation between the distribution patterns of paper key concept paths and both innovation points and rare paths. We propose a prompt engineering-based key concept path analysis method. This method leverages small language models to achieve precise key concept extraction and innovation point identification, and constructs an agent based on a knowledge graph constraint mechanism to enhance analysis accuracy. Through fine-tuning of the Qwen and DeepSeek models, we achieved significant improvements in accuracy, with the models publicly available on the Hugging Face platform.
>
---
#### [new 022] DROID: Dual Representation for Out-of-Scope Intent Detection
- **分类: cs.CL; I.2.7, I.5.1**

- **简介: 该论文针对任务型对话系统中的意图识别，解决用户输入超出预设范围（OOS）的检测难题。提出DROID框架，结合通用与领域适配编码器，通过双表示融合和数据增强，在低资源下显著提升OOS检测性能。**

- **链接: [http://arxiv.org/pdf/2510.14110v1](http://arxiv.org/pdf/2510.14110v1)**

> **作者:** Wael Rashwan; Hossam M. Zawbaa; Sourav Dutta; Haytham Assem
>
> **备注:** 14 pages, 6 figures, 4 Tables. Preprint submitted to IEEE Transactions on Neural Networks and Learning Systems (TNNLS)
>
> **摘要:** Detecting out-of-scope (OOS) user utterances remains a key challenge in task-oriented dialogue systems and, more broadly, in open-set intent recognition. Existing approaches often depend on strong distributional assumptions or auxiliary calibration modules. We present DROID (Dual Representation for Out-of-Scope Intent Detection), a compact end-to-end framework that combines two complementary encoders -- the Universal Sentence Encoder (USE) for broad semantic generalization and a domain-adapted Transformer-based Denoising Autoencoder (TSDAE) for domain-specific contextual distinctions. Their fused representations are processed by a lightweight branched classifier with a single calibrated threshold that separates in-domain and OOS intents without post-hoc scoring. To enhance boundary learning under limited supervision, DROID incorporates both synthetic and open-domain outlier augmentation. Despite using only 1.5M trainable parameters, DROID consistently outperforms recent state-of-the-art baselines across multiple intent benchmarks, achieving macro-F1 improvements of 6--15% for known and 8--20% for OOS intents, with the most significant gains in low-resource settings. These results demonstrate that dual-encoder representations with simple calibration can yield robust, scalable, and reliable OOS detection for neural dialogue systems.
>
---
#### [new 023] Language steering in latent space to mitigate unintended code-switching
- **分类: cs.CL; cs.LG**

- **简介: 该论文针对多语言大模型中意外的语码转换问题，提出在潜在空间中通过主成分分析识别语言方向，并在推理时调整词元嵌入以控制语言身份，有效减少语码转换，保持语义，仅需少量平行数据，计算开销低。**

- **链接: [http://arxiv.org/pdf/2510.13849v1](http://arxiv.org/pdf/2510.13849v1)**

> **作者:** Andrey Goncharov; Nikolai Kondusov; Alexey Zaytsev
>
> **摘要:** Multilingual Large Language Models (LLMs) often exhibit unintended code-switching, reducing reliability in downstream tasks. We propose latent-space language steering, a lightweight inference-time method that identifies language directions via PCA on parallel translations and steers token embeddings along these axes to control language identity. Our approach mitigates code-switching while preserving semantics with negligible computational overhead and requires only minimal parallel data for calibration. Empirically, we achieve 95-99\% language classification accuracy using a single principal component and reduce next-token distributional divergence by up to 42% across multiple language pairs on Qwen2.5 and Llama-3.2 models. We further analyze the layer-wise evolution of language representations, revealing that language identity concentrates in final layers with near-perfect linear separability.
>
---
#### [new 024] TokDrift: When LLM Speaks in Subwords but Code Speaks in Grammar
- **分类: cs.CL; cs.AI; cs.LG; cs.PL; cs.SE**

- **简介: 该论文研究代码大模型中子词分词与语法的错位问题。提出TokDrift框架，通过语义保持改写生成仅分词不同的代码变体，发现轻微格式变化即可导致模型行为显著波动，揭示分词未对齐语法边界是影响模型可靠性的隐患，呼吁未来采用语法感知的分词方法。**

- **链接: [http://arxiv.org/pdf/2510.14972v1](http://arxiv.org/pdf/2510.14972v1)**

> **作者:** Yinxi Li; Yuntian Deng; Pengyu Nie
>
> **摘要:** Large language models (LLMs) for code rely on subword tokenizers, such as byte-pair encoding (BPE), learned from mixed natural language text and programming language code but driven by statistics rather than grammar. As a result, semantically identical code snippets can be tokenized differently depending on superficial factors such as whitespace or identifier naming. To measure the impact of this misalignment, we introduce TokDrift, a framework that applies semantic-preserving rewrite rules to create code variants differing only in tokenization. Across nine code LLMs, including large ones with over 30B parameters, even minor formatting changes can cause substantial shifts in model behavior. Layer-wise analysis shows that the issue originates in early embeddings, where subword segmentation fails to capture grammar token boundaries. Our findings identify misaligned tokenization as a hidden obstacle to reliable code understanding and generation, highlighting the need for grammar-aware tokenization for future code LLMs.
>
---
#### [new 025] FinDeepResearch: Evaluating Deep Research Agents in Rigorous Financial Analysis
- **分类: cs.CL**

- **简介: 该论文聚焦深度研究代理在金融分析中的评估任务，旨在解决现有方法缺乏系统性评价的问题。作者提出HisRubric评估框架，构建FinDeepResearch多语言基准，涵盖64家上市公司，并对16种主流方法进行实验，揭示其在不同市场与语言下的表现差异。**

- **链接: [http://arxiv.org/pdf/2510.13936v1](http://arxiv.org/pdf/2510.13936v1)**

> **作者:** Fengbin Zhu; Xiang Yao Ng; Ziyang Liu; Chang Liu; Xianwei Zeng; Chao Wang; Tianhui Tan; Xuan Yao; Pengyang Shao; Min Xu; Zixuan Wang; Jing Wang; Xin Lin; Junfeng Li; Jingxian Zhu; Yang Zhang; Wenjie Wang; Fuli Feng; Richang Hong; Huanbo Luan; Ke-Wei Huang; Tat-Seng Chua
>
> **摘要:** Deep Research (DR) agents, powered by advanced Large Language Models (LLMs), have recently garnered increasing attention for their capability in conducting complex research tasks. However, existing literature lacks a rigorous and systematic evaluation of DR Agent's capabilities in critical research analysis. To address this gap, we first propose HisRubric, a novel evaluation framework with a hierarchical analytical structure and a fine-grained grading rubric for rigorously assessing DR agents' capabilities in corporate financial analysis. This framework mirrors the professional analyst's workflow, progressing from data recognition to metric calculation, and finally to strategic summarization and interpretation. Built on this framework, we construct a FinDeepResearch benchmark that comprises 64 listed companies from 8 financial markets across 4 languages, encompassing a total of 15,808 grading items. We further conduct extensive experiments on the FinDeepResearch using 16 representative methods, including 6 DR agents, 5 LLMs equipped with both deep reasoning and search capabilities, and 5 LLMs with deep reasoning capabilities only. The results reveal the strengths and limitations of these approaches across diverse capabilities, financial markets, and languages, offering valuable insights for future research and development. The benchmark and evaluation code will be made publicly available.
>
---
#### [new 026] RAGCap-Bench: Benchmarking Capabilities of LLMs in Agentic Retrieval Augmented Generation Systems
- **分类: cs.CL**

- **简介: 该论文提出RAGCap-Bench，旨在评估代理式检索增强生成系统中大模型的中间能力。针对多跳问答中推理不足的问题，构建细粒度基准，分析常见错误与核心能力，验证提升中间步骤表现可改善整体效果。**

- **链接: [http://arxiv.org/pdf/2510.13910v1](http://arxiv.org/pdf/2510.13910v1)**

> **作者:** Jingru Lin; Chen Zhang; Stephen Y. Liu; Haizhou Li
>
> **摘要:** Retrieval-Augmented Generation (RAG) mitigates key limitations of Large Language Models (LLMs)-such as factual errors, outdated knowledge, and hallucinations-by dynamically retrieving external information. Recent work extends this paradigm through agentic RAG systems, where LLMs act as agents to iteratively plan, retrieve, and reason over complex queries. However, these systems still struggle with challenging multi-hop questions, and their intermediate reasoning capabilities remain underexplored. To address this, we propose RAGCap-Bench, a capability-oriented benchmark for fine-grained evaluation of intermediate tasks in agentic RAG workflows. We analyze outputs from state-of-the-art systems to identify common tasks and the core capabilities required for their execution, then construct a taxonomy of typical LLM errors to design targeted evaluation questions. Experiments show that "slow-thinking" models with stronger RAGCap performance achieve better end-to-end results, underscoring the benchmark's validity and the importance of enhancing these intermediate capabilities.
>
---
#### [new 027] Robust or Suggestible? Exploring Non-Clinical Induction in LLM Drug-Safety Decisions
- **分类: cs.CL**

- **简介: 该论文研究LLM在药物安全预测中是否引入社会人口偏见。通过FAERS数据和 persona 框架，评估ChatGPT-4o和Bio-Medical-Llama在不同人群和用户角色下的表现，发现模型存在显性和隐性偏见，揭示其在药物流行病学应用中的公平性风险。**

- **链接: [http://arxiv.org/pdf/2510.13931v1](http://arxiv.org/pdf/2510.13931v1)**

> **作者:** Siying Liu; Shisheng Zhang; Indu Bala
>
> **备注:** Preprint of a paper accepted as a poster at the NeurIPS 2025 Workshop on Generative AI for Health (GenAI4Health). The final camera-ready workshop version may differ. Licensed under CC BY 4.0
>
> **摘要:** Large language models (LLMs) are increasingly applied in biomedical domains, yet their reliability in drug-safety prediction remains underexplored. In this work, we investigate whether LLMs incorporate socio-demographic information into adverse event (AE) predictions, despite such attributes being clinically irrelevant. Using structured data from the United States Food and Drug Administration Adverse Event Reporting System (FAERS) and a persona-based evaluation framework, we assess two state-of-the-art models, ChatGPT-4o and Bio-Medical-Llama-3.8B, across diverse personas defined by education, marital status, employment, insurance, language, housing stability, and religion. We further evaluate performance across three user roles (general practitioner, specialist, patient) to reflect real-world deployment scenarios where commercial systems often differentiate access by user type. Our results reveal systematic disparities in AE prediction accuracy. Disadvantaged groups (e.g., low education, unstable housing) were frequently assigned higher predicted AE likelihoods than more privileged groups (e.g., postgraduate-educated, privately insured). Beyond outcome disparities, we identify two distinct modes of bias: explicit bias, where incorrect predictions directly reference persona attributes in reasoning traces, and implicit bias, where predictions are inconsistent, yet personas are not explicitly mentioned. These findings expose critical risks in applying LLMs to pharmacovigilance and highlight the urgent need for fairness-aware evaluation protocols and mitigation strategies before clinical deployment.
>
---
#### [new 028] Meronymic Ontology Extraction via Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究自动构建产品本体中的部分-整体关系（meronymy）。针对人工构建本体费时费力的问题，提出利用大语言模型从评论文本中全自动抽取meronymic本体，方法优于BERT基线，验证了LLM在本体抽取中的潜力。**

- **链接: [http://arxiv.org/pdf/2510.13839v1](http://arxiv.org/pdf/2510.13839v1)**

> **作者:** Dekai Zhang; Simone Conia; Antonio Rago
>
> **摘要:** Ontologies have become essential in today's digital age as a way of organising the vast amount of readily available unstructured text. In providing formal structure to this information, ontologies have immense value and application across various domains, e.g., e-commerce, where countless product listings necessitate proper product organisation. However, the manual construction of these ontologies is a time-consuming, expensive and laborious process. In this paper, we harness the recent advancements in large language models (LLMs) to develop a fully-automated method of extracting product ontologies, in the form of meronymies, from raw review texts. We demonstrate that the ontologies produced by our method surpass an existing, BERT-based baseline when evaluating using an LLM-as-a-judge. Our investigation provides the groundwork for LLMs to be used more generally in (product or otherwise) ontology extraction.
>
---
#### [new 029] Flip-Flop Consistency: Unsupervised Training for Robustness to Prompt Perturbations in LLMs
- **分类: cs.CL; cs.LG**

- **简介: 该论文针对大语言模型在不同提示表述下输出不一致的问题，提出一种无监督训练方法Flip-Flop Consistency（F²C），通过共识交叉熵和表示对齐损失提升模型对提示扰动的鲁棒性，增强一致性、性能与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.14242v1](http://arxiv.org/pdf/2510.14242v1)**

> **作者:** Parsa Hejabi; Elnaz Rahmati; Alireza S. Ziabari; Morteza Dehghani
>
> **备注:** 14 pages, 6 figures, 3 tables, and 1 algorithm
>
> **摘要:** Large Language Models (LLMs) often produce inconsistent answers when faced with different phrasings of the same prompt. In this paper, we propose Flip-Flop Consistency ($F^2C$), an unsupervised training method that improves robustness to such perturbations. $F^2C$ is composed of two key components. The first, Consensus Cross-Entropy (CCE), uses a majority vote across prompt variations to create a hard pseudo-label. The second is a representation alignment loss that pulls lower-confidence and non-majority predictors toward the consensus established by high-confidence, majority-voting variations. We evaluate our method on 11 datasets spanning four NLP tasks, with 4-15 prompt variations per dataset. On average, $F^2C$ raises observed agreement by 11.62%, improves mean $F_1$ by 8.94%, and reduces performance variance across formats by 3.29%. In out-of-domain evaluations, $F^2C$ generalizes effectively, increasing $\overline{F_1}$ and agreement while decreasing variance across most source-target pairs. Finally, when trained on only a subset of prompt perturbations and evaluated on held-out formats, $F^2C$ consistently improves both performance and agreement while reducing variance. These findings highlight $F^2C$ as an effective unsupervised method for enhancing LLM consistency, performance, and generalization under prompt perturbations. Code is available at https://github.com/ParsaHejabi/Flip-Flop-Consistency-Unsupervised-Training-for-Robustness-to-Prompt-Perturbations-in-LLMs.
>
---
#### [new 030] Revisiting the UID Hypothesis in LLM Reasoning Traces
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大模型推理过程的信息分布，受语言学UID假说启发，提出用熵度量分析推理链。发现正确数学推理的信息流不均匀，与人类通信模式不同，挑战了现有认知，为可解释推理模型提供新方向。**

- **链接: [http://arxiv.org/pdf/2510.13850v1](http://arxiv.org/pdf/2510.13850v1)**

> **作者:** Minju Gwak; Guijin Son; Jaehyung Kim
>
> **摘要:** Large language models (LLMs) often solve problems using step-by-step Chain-of-Thought (CoT) reasoning, yet these intermediate steps are frequently unfaithful or hard to interpret. Inspired by the Uniform Information Density (UID) hypothesis in psycholinguistics -- which posits that humans communicate by maintaining a stable flow of information -- we introduce entropy-based metrics to analyze the information flow within reasoning traces. Surprisingly, across three challenging mathematical benchmarks, we find that successful reasoning in LLMs is globally non-uniform: correct solutions are characterized by uneven swings in information density, in stark contrast to human communication patterns. This result challenges assumptions about machine reasoning and suggests new directions for designing interpretable and adaptive reasoning models.
>
---
#### [new 031] Explore to Evolve: Scaling Evolved Aggregation Logic via Proactive Online Exploration for Deep Research Agents
- **分类: cs.CL**

- **简介: 该论文聚焦深度研究智能体的信息聚合任务，旨在解决现有方法重检索轻聚合的问题。提出“探索以进化”范式，通过主动在线探索构建高质量聚合数据集WebAggregatorQA，并训练出具备强聚合能力的开源模型WebAggregator，显著提升信息整合性能。**

- **链接: [http://arxiv.org/pdf/2510.14438v1](http://arxiv.org/pdf/2510.14438v1)**

> **作者:** Rui Wang; Ce Zhang; Jun-Yu Ma; Jianshu Zhang; Hongru Wang; Yi Chen; Boyang Xue; Tianqing Fang; Zhisong Zhang; Hongming Zhang; Haitao Mi; Dong Yu; Kam-Fai Wong
>
> **摘要:** Deep research web agents not only retrieve information from diverse sources such as web environments, files, and multimodal inputs, but more importantly, they need to rigorously analyze and aggregate knowledge for insightful research. However, existing open-source deep research agents predominantly focus on enhancing information-seeking capabilities of web agents to locate specific information, while overlooking the essential need for information aggregation, which would limit their ability to support in-depth research. We propose an Explore to Evolve paradigm to scalably construct verifiable training data for web agents. Begins with proactive online exploration, an agent sources grounded information by exploring the real web. Using the collected evidence, the agent then self-evolves an aggregation program by selecting, composing, and refining operations from 12 high-level logical types to synthesize a verifiable QA pair. This evolution from high-level guidance to concrete operations allowed us to scalably produce WebAggregatorQA, a dataset of 10K samples across 50K websites and 11 domains. Based on an open-source agent framework, SmolAgents, we collect supervised fine-tuning trajectories to develop a series of foundation models, WebAggregator. WebAggregator-8B matches the performance of GPT-4.1, while the 32B variant surpasses GPT-4.1 by more than 10% on GAIA-text and closely approaches Claude-3.7-sonnet. Moreover, given the limited availability of benchmarks that evaluate web agents' information aggregation abilities, we construct a human-annotated evaluation split of WebAggregatorQA as a challenging test set. On this benchmark, Claude-3.7-sonnet only achieves 28%, and GPT-4.1 scores 25.8%. Even when agents manage to retrieve all references, they still struggle on WebAggregatorQA, highlighting the need to strengthen the information aggregation capabilities of web agent foundations.
>
---
#### [new 032] LLMs as Scalable, General-Purpose Simulators For Evolving Digital Agent Training
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于数字智能体训练任务，旨在解决真实UI轨迹数据收集成本高的问题。作者提出UI-Simulator，通过大模型生成多样化UI状态与交互轨迹，并设计UI-Simulator-Grow实现高效扩展，显著提升智能体训练效果与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.14969v1](http://arxiv.org/pdf/2510.14969v1)**

> **作者:** Yiming Wang; Da Yin; Yuedong Cui; Ruichen Zheng; Zhiqian Li; Zongyu Lin; Di Wu; Xueqing Wu; Chenchen Ye; Yu Zhou; Kai-Wei Chang
>
> **备注:** Preprint. Project page: https://ui-simulator.notion.site/llms-as-scalable-digital-world-simulator; Code and data: https://github.com/WadeYin9712/UI-Simulator
>
> **摘要:** Digital agents require diverse, large-scale UI trajectories to generalize across real-world tasks, yet collecting such data is prohibitively expensive in both human annotation, infra and engineering perspectives. To this end, we introduce $\textbf{UI-Simulator}$, a scalable paradigm that generates structured UI states and transitions to synthesize training trajectories at scale. Our paradigm integrates a digital world simulator for diverse UI states, a guided rollout process for coherent exploration, and a trajectory wrapper that produces high-quality and diverse trajectories for agent training. We further propose $\textbf{UI-Simulator-Grow}$, a targeted scaling strategy that enables more rapid and data-efficient scaling by prioritizing high-impact tasks and synthesizes informative trajectory variants. Experiments on WebArena and AndroidWorld show that UI-Simulator rivals or surpasses open-source agents trained on real UIs with significantly better robustness, despite using weaker teacher models. Moreover, UI-Simulator-Grow matches the performance of Llama-3-70B-Instruct using only Llama-3-8B-Instruct as the base model, highlighting the potential of targeted synthesis scaling paradigm to continuously and efficiently enhance the digital agents.
>
---
#### [new 033] MetaBench: A Multi-task Benchmark for Assessing LLMs in Metabolomics
- **分类: cs.CL; cs.AI; cs.CE**

- **简介: 该论文提出MetaBench，首个面向代谢组学的多任务基准，旨在评估大语言模型在知识、理解、映射、推理和研究五方面的能力，揭示现有模型在跨数据库标识映射和稀有代谢物上的不足，推动领域专用AI工具发展。**

- **链接: [http://arxiv.org/pdf/2510.14944v1](http://arxiv.org/pdf/2510.14944v1)**

> **作者:** Yuxing Lu; Xukai Zhao; J. Ben Tamo; Micky C. Nnamdi; Rui Peng; Shuang Zeng; Xingyu Hu; Jinzhuo Wang; May D. Wang
>
> **备注:** 22 pages, 6 figures, 4 tables
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities on general text; however, their proficiency in specialized scientific domains that require deep, interconnected knowledge remains largely uncharacterized. Metabolomics presents unique challenges with its complex biochemical pathways, heterogeneous identifier systems, and fragmented databases. To systematically evaluate LLM capabilities in this domain, we introduce MetaBench, the first benchmark for metabolomics assessment. Curated from authoritative public resources, MetaBench evaluates five capabilities essential for metabolomics research: knowledge, understanding, grounding, reasoning, and research. Our evaluation of 25 open- and closed-source LLMs reveals distinct performance patterns across metabolomics tasks: while models perform well on text generation tasks, cross-database identifier grounding remains challenging even with retrieval augmentation. Model performance also decreases on long-tail metabolites with sparse annotations. With MetaBench, we provide essential infrastructure for developing and evaluating metabolomics AI systems, enabling systematic progress toward reliable computational tools for metabolomics research.
>
---
#### [new 034] RLSR: Reinforcement Learning with Supervised Reward Outperforms SFT in Instruction Following
- **分类: cs.CL**

- **简介: 该论文研究大模型指令遵循优化任务，提出用监督奖励的强化学习（RLSR）替代传统SFT。RLSR利用人类标注响应的语义相似性作为奖励信号，提升指令遵循能力，实验表明其优于SFT，并可与SFT结合进一步提效。**

- **链接: [http://arxiv.org/pdf/2510.14200v1](http://arxiv.org/pdf/2510.14200v1)**

> **作者:** Zhichao Wang; Andy Wong; Ruslan Belkin
>
> **摘要:** After the pretraining stage of LLMs, techniques such as SFT, RLHF, RLVR, and RFT are applied to enhance instruction-following ability, mitigate undesired responses, improve reasoning capability and enable efficient domain adaptation with minimal data. SFT relies on the next-token prediction objective to strengthen instruction following in a base model using a large corpus of human-labeled responses. In contrast, RFT employs a RL-based approach to adapt fine-tuned reasoning models to specific domains with limited supervision. Inspired by RFT, we propose replacing SFT with RLSR to leverage the extensive SFT dataset in an RL framework, thereby improving the base model's instruction-following ability. In RLSR, the base model generates multiple responses for each prompt, and reward scores are computed as the cosine similarity in the semantic embedding space between the generated and human-labeled responses. RLSR can be utilized in multiple ways. It can directly replace SFT, achieving superior performance on instruction-following benchmarks-for example, RLSR (SB) on Qwen-7B (INFINITY) achieved an AlpacaEval win rate of 26.34%, surpassing SFT's 21.01%. Furthermore, combining SFT and RLSR further enhances downstream task performance; Qwen-7B (INFINITY) achieved a win rate of 30.73% when trained with SFT + RLSR.
>
---
#### [new 035] Pluto: A Benchmark for Evaluating Efficiency of LLM-generated Hardware Code
- **分类: cs.CL**

- **简介: 该论文针对大模型生成硬件代码的效率评估问题，提出Pluto基准框架。它包含114个带测试平台和优化参考实现的问题，用于评估生成代码在面积、延迟和功耗上的表现，揭示当前模型在功能正确性尚可但效率显著不足的问题。**

- **链接: [http://arxiv.org/pdf/2510.14756v1](http://arxiv.org/pdf/2510.14756v1)**

> **作者:** Manar Abdelatty; Maryam Nouh; Jacob K. Rosenstein; Sherief Reda
>
> **摘要:** Large Language Models (LLMs) are increasingly used to automate hardware design tasks, including the generation of Verilog code. While early benchmarks focus primarily on functional correctness, efficient hardware design demands additional optimization for synthesis metrics such as area, delay, and power. Existing benchmarks fall short in evaluating these aspects comprehensively: they often lack optimized baselines or testbenches for verification. To address these gaps, we present Pluto, a benchmark and evaluation framework designed to assess the efficiency of LLM-generated Verilog designs. Pluto presents a comprehensive evaluation set of 114 problems with self-checking testbenches and multiple Pareto-optimal reference implementations. Experimental results show that state-of-the-art LLMs can achieve high functional correctness, reaching 78.3\% at pass@1, but their synthesis efficiency still lags behind expert-crafted implementations, with area efficiency of 63.8\%, delay efficiency of 65.9\%, and power efficiency of 64.0\% at eff@1. This highlights the need for efficiency-aware evaluation frameworks such as Pluto to drive progress in hardware-focused LLM research.
>
---
#### [new 036] Think Globally, Group Locally: Evaluating LLMs Using Multi-Lingual Word Grouping Games
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出GlobalGroup任务，通过多语言词汇分组游戏评估大模型在抽象推理中的语言偏差问题。构建了五种语言的基准测试，衡量不同语言模态下的模型表现，发现英语表现更优，并揭示开源与闭源模型间的性能差异。**

- **链接: [http://arxiv.org/pdf/2510.14030v1](http://arxiv.org/pdf/2510.14030v1)**

> **作者:** César Guerra-Solano; Zhuochun Li; Xiang Lorraine Li
>
> **备注:** EMNLP Main 2025
>
> **摘要:** Large language models (LLMs) can exhibit biases in reasoning capabilities due to linguistic modality, performing better on tasks in one language versus another, even with similar content. Most previous works evaluate this through reasoning tasks where reliance on strategies or knowledge can ensure success, such as in commonsense or math tasks. However, abstract reasoning is vital to reasoning for everyday life, where people apply "out-of-the-box thinking" to identify and use patterns for solutions, without a reliance on formulaic approaches. Comparatively, little work has evaluated linguistic biases in this task type. In this paper, we propose a task inspired by the New York Times Connections: GlobalGroup, that evaluates models in an abstract reasoning task across several languages. We constructed a game benchmark with five linguistic backgrounds -- English, Spanish, Chinese, Hindi, and Arabic -- in both the native language and an English translation for comparison. We also proposed game difficulty measurements to evaluate models on games with similar difficulty, enabling a more controlled comparison, which is particularly important in reasoning evaluations. Through experimentation, we find English modalities largely lead to better performance in this abstract reasoning task, and performance disparities between open- and closed-source models.
>
---
#### [new 037] From Loop Nests to Silicon: Mapping AI Workloads onto AMD NPUs with MLIR-AIR
- **分类: cs.CL; cs.AR; cs.LG**

- **简介: 该论文提出MLIR-AIR编译器栈，旨在将高层AI工作负载高效映射到AMD NPU等空间架构。通过AIR方言实现对计算、数据流和并行的细粒度控制，解决了传统编译器抽象过度导致性能受限的问题，支持自动调度与通信计算重叠，在矩阵乘法和多头注意力中验证了高效性。**

- **链接: [http://arxiv.org/pdf/2510.14871v1](http://arxiv.org/pdf/2510.14871v1)**

> **作者:** Erwei Wang; Samuel Bayliss; Andra Bisca; Zachary Blair; Sangeeta Chowdhary; Kristof Denolf; Jeff Fifield; Brandon Freiberger; Erika Hunhoff; Phil James-Roxby; Jack Lo; Joseph Melber; Stephen Neuendorffer; Eddie Richter; Andre Rosti; Javier Setoain; Gagandeep Singh; Endri Taka; Pranathi Vasireddy; Zhewen Yu; Niansong Zhang; Jinming Zhuang
>
> **摘要:** General-purpose compilers abstract away parallelism, locality, and synchronization, limiting their effectiveness on modern spatial architectures. As modern computing architectures increasingly rely on fine-grained control over data movement, execution order, and compute placement for performance, compiler infrastructure must provide explicit mechanisms for orchestrating compute and data to fully exploit such architectures. We introduce MLIR-AIR, a novel, open-source compiler stack built on MLIR that bridges the semantic gap between high-level workloads and fine-grained spatial architectures such as AMD's NPUs. MLIR-AIR defines the AIR dialect, which provides structured representations for asynchronous and hierarchical operations across compute and memory resources. AIR primitives allow the compiler to orchestrate spatial scheduling, distribute computation across hardware regions, and overlap communication with computation without relying on ad hoc runtime coordination or manual scheduling. We demonstrate MLIR-AIR's capabilities through two case studies: matrix multiplication and the multi-head attention block from the LLaMA 2 model. For matrix multiplication, MLIR-AIR achieves up to 78.7% compute efficiency and generates implementations with performance almost identical to state-of-the-art, hand-optimized matrix multiplication written using the lower-level, close-to-metal MLIR-AIE framework. For multi-head attention, we demonstrate that the AIR interface supports fused implementations using approximately 150 lines of code, enabling tractable expression of complex workloads with efficient mapping to spatial hardware. MLIR-AIR transforms high-level structured control flow into spatial programs that efficiently utilize the compute fabric and memory hierarchy of an NPU, leveraging asynchronous execution, tiling, and communication overlap through compiler-managed scheduling.
>
---
#### [new 038] Ensembling Large Language Models to Characterize Affective Dynamics in Student-AI Tutor Dialogues
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文聚焦教育中AI辅导对话的情感动态分析，提出首个基于大模型集成的框架，通过融合多个前沿大模型的零样本情感标注，实现大规模、细粒度的情感感知，揭示学生在与AI互动中的情绪变化规律及干预时机。**

- **链接: [http://arxiv.org/pdf/2510.13862v1](http://arxiv.org/pdf/2510.13862v1)**

> **作者:** Chenyu Zhang; Sharifa Alghowinem; Cynthia Breazeal
>
> **备注:** 4 pages, 3 figures. Published in the 11th International Conference on Affective Computing and Intelligent Interaction (ACII 2025), Late-Breaking Results Track
>
> **摘要:** While recent studies have examined the leaning impact of large language model (LLM) in educational contexts, the affective dynamics of LLM-mediated tutoring remain insufficiently understood. This work introduces the first ensemble-LLM framework for large-scale affect sensing in tutoring dialogues, advancing the conversation on responsible pathways for integrating generative AI into education by attending to learners' evolving affective states. To achieve this, we analyzed two semesters' worth of 16,986 conversational turns exchanged between PyTutor, an LLM-powered AI tutor, and 261 undergraduate learners across three U.S. institutions. To investigate learners' emotional experiences, we generate zero-shot affect annotations from three frontier LLMs (Gemini, GPT-4o, Claude), including scalar ratings of valence, arousal, and learning-helpfulness, along with free-text emotion labels. These estimates are fused through rank-weighted intra-model pooling and plurality consensus across models to produce robust emotion profiles. Our analysis shows that during interaction with the AI tutor, students typically report mildly positive affect and moderate arousal. Yet learning is not uniformly smooth: confusion and curiosity are frequent companions to problem solving, and frustration, while less common, still surfaces in ways that can derail progress. Emotional states are short-lived--positive moments last slightly longer than neutral or negative ones, but they are fragile and easily disrupted. Encouragingly, negative emotions often resolve quickly, sometimes rebounding directly into positive states. Neutral moments frequently act as turning points, more often steering students upward than downward, suggesting opportunities for tutors to intervene at precisely these junctures.
>
---
#### [new 039] Multimodal Retrieval-Augmented Generation with Large Language Models for Medical VQA
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文针对医疗视觉问答（MedVQA）任务，旨在通过结合文本与视觉示例的检索增强生成（RAG）方法，提升大语言模型在伤口护理问答中的回答质量。作者提出MasonNLP系统，利用通用大模型与轻量级RAG框架，无需额外训练，有效提高推理与结构化输出能力。**

- **链接: [http://arxiv.org/pdf/2510.13856v1](http://arxiv.org/pdf/2510.13856v1)**

> **作者:** A H M Rezaul Karim; Ozlem Uzuner
>
> **摘要:** Medical Visual Question Answering (MedVQA) enables natural language queries over medical images to support clinical decision-making and patient care. The MEDIQA-WV 2025 shared task addressed wound-care VQA, requiring systems to generate free-text responses and structured wound attributes from images and patient queries. We present the MasonNLP system, which employs a general-domain, instruction-tuned large language model with a retrieval-augmented generation (RAG) framework that incorporates textual and visual examples from in-domain data. This approach grounds outputs in clinically relevant exemplars, improving reasoning, schema adherence, and response quality across dBLEU, ROUGE, BERTScore, and LLM-based metrics. Our best-performing system ranked 3rd among 19 teams and 51 submissions with an average score of 41.37%, demonstrating that lightweight RAG with general-purpose LLMs -- a minimal inference-time layer that adds a few relevant exemplars via simple indexing and fusion, with no extra training or complex re-ranking -- provides a simple and effective baseline for multimodal clinical NLP tasks.
>
---
#### [new 040] Attribution Quality in AI-Generated Content:Benchmarking Style Embeddings and LLM Judges
- **分类: cs.CL**

- **简介: 该论文研究AI生成内容的作者归属问题，比较了风格嵌入和大模型裁判两种方法在多领域文本中的归因效果，发现二者在不同文体中各有优势，提出需结合互补策略，并提供了开源基准框架。**

- **链接: [http://arxiv.org/pdf/2510.13898v1](http://arxiv.org/pdf/2510.13898v1)**

> **作者:** Misam Abbas
>
> **备注:** Accepted for publication at the 2025 IEEE ICDM Workshop on "Grounding Documents with Reasoning, Agents, Retrieval, and Attribution". This is author submitted version. Not yet published
>
> **摘要:** Attributing authorship in the era of large language models (LLMs) is increasingly challenging as machine-generated prose rivals human writing. We benchmark two complementary attribution mechanisms , fixed Style Embeddings and an instruction-tuned LLM judge (GPT-4o) on the Human AI Parallel Corpus, an open dataset of 600 balanced instances spanning six domains (academic, news, fiction, blogs, spoken transcripts, and TV/movie scripts). Each instance contains a human prompt with both a gold continuation and an LLM-generated continuation from either GPT-4o or LLaMA-70B-Instruct. The Style Embedding baseline achieves stronger aggregate accuracy on GPT continuations (82 pct vs. 68 pct). The LLM Judge is slightly better than the Style embeddings on LLaMA continuations (85 pct vs. 81 pct) but the results are not statistically significant. Crucially, the LLM judge significantly outperforms in fiction and academic prose, indicating semantic sensitivity, whereas embeddings dominate in spoken and scripted dialogue, reflecting structural strengths. These complementary patterns highlight attribution as a multidimensional problem requiring hybrid strategies. To support reproducibility we provide code on GitHub and derived data on Hugging Face under the MIT license. This open framework provides a reproducible benchmark for attribution quality assessment in AI-generated content, along with a review of related literature influencing this work.
>
---
#### [new 041] Natural Language Tools: A Natural Language Approach to Tool Calling In Large Language Agents
- **分类: cs.CL**

- **简介: 该论文提出自然语言工具（NLT）框架，解决大模型中程序化工具调用的格式约束与任务干扰问题。通过自然语言输出解耦工具选择与响应生成，在多领域提升工具调用准确率18.4%，降低输出波动，适用于开放权重模型并增强无原生支持模型的能力。**

- **链接: [http://arxiv.org/pdf/2510.14453v1](http://arxiv.org/pdf/2510.14453v1)**

> **作者:** Reid T. Johnson; Michelle D. Pain; Jordan D. West
>
> **备注:** 31 pages, 7 figures
>
> **摘要:** We present Natural Language Tools (NLT), a framework that replaces programmatic JSON tool calling in large language models (LLMs) with natural language outputs. By decoupling tool selection from response generation, NLT eliminates task interference and format constraints that degrade tool call performance. When evaluated across 10 models and 6,400 trials spanning customer service and mental health domains, NLT improves tool calling accuracy by 18.4 percentage points while reducing output variance by 70%. Open-weight models see the largest gains, surpassing flagship closed-weight alternatives, with implications for model training in both reinforcement learning and supervised fine-tuning stages. These improvements persist under prompt perturbations and extend tool-calling capabilities to models lacking native support.
>
---
#### [new 042] MERLIN: A Testbed for Multilingual Multimodal Entity Recognition and Linking
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对多语言多模态实体识别与链接任务，构建了包含五种语言新闻标题及配图的MERLIN测试集，并引入视觉信息提升链接准确率，尤其改善文本模糊或模型多语言能力弱时的表现。**

- **链接: [http://arxiv.org/pdf/2510.14307v1](http://arxiv.org/pdf/2510.14307v1)**

> **作者:** Sathyanarayanan Ramamoorthy; Vishwa Shah; Simran Khanuja; Zaid Sheikh; Shan Jie; Ann Chia; Shearman Chua; Graham Neubig
>
> **摘要:** This paper introduces MERLIN, a novel testbed system for the task of Multilingual Multimodal Entity Linking. The created dataset includes BBC news article titles, paired with corresponding images, in five languages: Hindi, Japanese, Indonesian, Vietnamese, and Tamil, featuring over 7,000 named entity mentions linked to 2,500 unique Wikidata entities. We also include several benchmarks using multilingual and multimodal entity linking methods exploring different language models like LLaMa-2 and Aya-23. Our findings indicate that incorporating visual data improves the accuracy of entity linking, especially for entities where the textual context is ambiguous or insufficient, and particularly for models that do not have strong multilingual abilities. For the work, the dataset, methods are available here at https://github.com/rsathya4802/merlin
>
---
#### [new 043] Guarding the Guardrails: A Taxonomy-Driven Approach to Jailbreak Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦LLM越狱攻击检测，旨在解决现有防御方法覆盖不全、分类不足的问题。作者提出包含50种策略的层次化分类体系，开展红队测试，分析攻击有效性，并构建意大利语多轮对抗对话数据集，提升自动检测能力。**

- **链接: [http://arxiv.org/pdf/2510.13893v1](http://arxiv.org/pdf/2510.13893v1)**

> **作者:** Olga E. Sorokoletova; Francesco Giarrusso; Vincenzo Suriani; Daniele Nardi
>
> **摘要:** Jailbreaking techniques pose a significant threat to the safety of Large Language Models (LLMs). Existing defenses typically focus on single-turn attacks, lack coverage across languages, and rely on limited taxonomies that either fail to capture the full diversity of attack strategies or emphasize risk categories rather than the jailbreaking techniques. To advance the understanding of the effectiveness of jailbreaking techniques, we conducted a structured red-teaming challenge. The outcome of our experiments are manifold. First, we developed a comprehensive hierarchical taxonomy of 50 jailbreak strategies, consolidating and extending prior classifications into seven broad families, including impersonation, persuasion, privilege escalation, cognitive overload, obfuscation, goal conflict, and data poisoning. Second, we analyzed the data collected from the challenge to examine the prevalence and success rates of different attack types, providing insights into how specific jailbreak strategies exploit model vulnerabilities and induce misalignment. Third, we benchmark a popular LLM for jailbreak detection, evaluating the benefits of taxonomy-guided prompting for improving automatic detection. Finally, we compiled a new Italian dataset of 1364 multi-turn adversarial dialogues, annotated with our taxonomy, enabling the study of interactions where adversarial intent emerges gradually and succeeds in bypassing traditional safeguards.
>
---
#### [new 044] Less is More: Improving LLM Reasoning with Minimal Test-Time Intervention
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦大语言模型推理优化任务，旨在减少测试时代价。通过发现推理不确定性集中在少数高熵位置，提出无需训练的Minimal Test-Time Intervention（MTI）框架，仅在关键位置进行选择性干预，显著提升推理准确率与稳定性，同时保持高效。**

- **链接: [http://arxiv.org/pdf/2510.13940v1](http://arxiv.org/pdf/2510.13940v1)**

> **作者:** Zhen Yang; Mingyang Zhang; Feng Chen; Ganggui Ding; Liang Hou; Xin Tao; Pengfei Wan; Ying-Cong Chen
>
> **备注:** Code: https://github.com/EnVision-Research/MTI
>
> **摘要:** Recent progress in large language models (LLMs) has focused on test-time scaling to improve reasoning via increased inference computation, but often at the cost of efficiency. We revisit test-time behavior and uncover a simple yet underexplored phenomenon: reasoning uncertainty is highly localized-only a small subset of high-entropy tokens dominantly affects output correctness. Motivated by this, we propose Minimal Test-Time Intervention (MTI), a training-free framework that enhances reasoning accuracy and stability with minimal overhead. MTI includes: (i) Selective CFG intervention, applying classifier-free guidance only at uncertain positions; and (ii) Lightweight negative-prompt guidance, reusing the main model's KV cache to approximate unconditional decoding efficiently. MTI yields consistent gains across general, coding, and STEM tasks-e.g., +1.35% average improvement on eight benchmarks for Qwen3-8B-Base and +5% on AIME2024 using Qwen3-32B-Reasoning-while remaining highly efficient.
>
---
#### [new 045] Your Next Token Prediction: A Multilingual Benchmark for Personalized Response Generation
- **分类: cs.CL**

- **简介: 该论文提出“你的下一个词预测”（YNTP）任务，旨在生成个性化语言响应。针对隐私导致的真实对话数据稀缺问题，构建了英日中多语言基准数据集，通过用户与基于MBTI的人格化NPC五日对话，捕捉个体表达习惯，评估提示与微调方法，推动用户对齐的语言建模。**

- **链接: [http://arxiv.org/pdf/2510.14398v1](http://arxiv.org/pdf/2510.14398v1)**

> **作者:** Shiyao Ding; Takayuki Ito
>
> **摘要:** Large language models (LLMs) excel at general next-token prediction but still struggle to generate responses that reflect how individuals truly communicate, such as replying to emails or social messages in their own style. However, real SNS or email histories are difficult to collect due to privacy concerns. To address this, we propose the task of "Your Next Token Prediction (YNTP)", which models a user's precise word choices through controlled human-agent conversations. We build a multilingual benchmark of 100 dialogue sessions across English, Japanese, and Chinese, where users interact for five days with psychologically grounded NPCs based on MBTI dimensions. This setup captures natural, daily-life communication patterns and enables analysis of users' internal models. We evaluate prompt-based and fine-tuning-based personalization methods, establishing the first benchmark for YNTP and a foundation for user-aligned language modeling. The dataset is available at: https://github.com/AnonymousHub4Submissions/your-next-token-prediction-dataset-100
>
---
#### [new 046] On-device System of Compositional Multi-tasking in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究大模型在设备端的多任务组合执行，解决传统方法难以同时处理如摘要与翻译联用的问题。提出添加可学习投影层，融合多个适配器，在保持高效的同时实现复杂任务协同，验证于安卓应用中，兼顾速度与资源限制。**

- **链接: [http://arxiv.org/pdf/2510.13848v1](http://arxiv.org/pdf/2510.13848v1)**

> **作者:** Ondrej Bohdal; Konstantinos Theodosiadis; Asterios Mpatziakas; Dimitris Filippidis; Iro Spyrou; Christos Zonios; Anastasios Drosou; Dimosthenis Ioannidis; Kyeng-Hun Lee; Jijoong Moon; Hyeonmok Ko; Mete Ozay; Umberto Michieli
>
> **备注:** Accepted at EMNLP 2025 (industry track)
>
> **摘要:** Large language models (LLMs) are commonly adapted for diverse downstream tasks via parameter-efficient fine-tuning techniques such as Low-Rank Adapters (LoRA). While adapters can be combined to handle multiple tasks separately, standard approaches struggle when targeting the simultaneous execution of complex tasks, such as generating a translated summary from a long conversation. To address this challenge, we propose a novel approach tailored specifically for compositional multi-tasking scenarios involving summarization and translation. Our technique involves adding a learnable projection layer on top of the combined summarization and translation adapters. This design enables effective integration while maintaining efficiency through reduced computational overhead compared to alternative strategies requiring extensive retraining or sequential processing. We demonstrate the practical viability of our method within an on-device environment by developing an Android app capable of executing compositional tasks seamlessly. Experimental results indicate our solution performs well and is fast in both cloud-based and on-device implementations, highlighting the potential benefits of adopting our framework in real-world applications demanding high-speed operation alongside resource constraints.
>
---
#### [new 047] Too Open for Opinion? Embracing Open-Endedness in Large Language Models for Social Simulation
- **分类: cs.CL**

- **简介: 该论文探讨在社会模拟中使用大语言模型时，应采用开放式文本生成而非封闭式问答。任务是提升社会仿真真实性，解决人为限制导致的偏差问题，主张利用开放-ended生成捕捉多样观点与推理，增强测量效度与方法价值。**

- **链接: [http://arxiv.org/pdf/2510.13884v1](http://arxiv.org/pdf/2510.13884v1)**

> **作者:** Bolei Ma; Yong Cao; Indira Sen; Anna-Carolina Haensch; Frauke Kreuter; Barbara Plank; Daniel Hershcovich
>
> **摘要:** Large Language Models (LLMs) are increasingly used to simulate public opinion and other social phenomena. Most current studies constrain these simulations to multiple-choice or short-answer formats for ease of scoring and comparison, but such closed designs overlook the inherently generative nature of LLMs. In this position paper, we argue that open-endedness, using free-form text that captures topics, viewpoints, and reasoning processes "in" LLMs, is essential for realistic social simulation. Drawing on decades of survey-methodology research and recent advances in NLP, we argue why this open-endedness is valuable in LLM social simulations, showing how it can improve measurement and design, support exploration of unanticipated views, and reduce researcher-imposed directive bias. It also captures expressiveness and individuality, aids in pretesting, and ultimately enhances methodological utility. We call for novel practices and evaluation frameworks that leverage rather than constrain the open-ended generative diversity of LLMs, creating synergies between NLP and social science.
>
---
#### [new 048] AI-Powered Early Diagnosis of Mental Health Disorders from Real-World Clinical Conversations
- **分类: cs.CL**

- **简介: 该论文研究利用机器学习从真实临床对话中早期诊断抑郁症、焦虑症和PTSD。针对误诊率高问题，作者使用真实访谈数据，比较了GPT-4.1 Mini、MetaLLaMA和LoRA微调RoBERTa模型，实现超80%准确率，尤其提升PTSD检测效果，推动AI在低资源环境中的应用。**

- **链接: [http://arxiv.org/pdf/2510.14937v1](http://arxiv.org/pdf/2510.14937v1)**

> **作者:** Jianfeng Zhu; Julina Maharjan; Xinyu Li; Karin G. Coifman; Ruoming Jin
>
> **备注:** 7 pages 1 figure
>
> **摘要:** Mental health disorders remain among the leading cause of disability worldwide, yet conditions such as depression, anxiety, and Post-Traumatic Stress Disorder (PTSD) are frequently underdiagnosed or misdiagnosed due to subjective assessments, limited clinical resources, and stigma and low awareness. In primary care settings, studies show that providers misidentify depression or anxiety in over 60% of cases, highlighting the urgent need for scalable, accessible, and context-aware diagnostic tools that can support early detection and intervention. In this study, we evaluate the effectiveness of machine learning models for mental health screening using a unique dataset of 553 real-world, semistructured interviews, each paried with ground-truth diagnoses for major depressive episodes (MDE), anxiety disorders, and PTSD. We benchmark multiple model classes, including zero-shot prompting with GPT-4.1 Mini and MetaLLaMA, as well as fine-tuned RoBERTa models using LowRank Adaptation (LoRA). Our models achieve over 80% accuracy across diagnostic categories, with especially strongperformance on PTSD (up to 89% accuracy and 98% recall). We also find that using shorter context, focused context segments improves recall, suggesting that focused narrative cues enhance detection sensitivity. LoRA fine-tuning proves both efficient and effective, with lower-rank configurations (e.g., rank 8 and 16) maintaining competitive performance across evaluation metrics. Our results demonstrate that LLM-based models can offer substantial improvements over traditional self-report screening tools, providing a path toward low-barrier, AI-powerd early diagnosis. This work lays the groundwork for integrating machine learning into real-world clinical workflows, particularly in low-resource or high-stigma environments where access to timely mental health care is most limited.
>
---
#### [new 049] AI Debaters are More Persuasive when Arguing in Alignment with Their Own Beliefs
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究AI辩论中的说服力与模型信念的关系，探讨在主观问题中模型是否更愿迎合法官立场。通过测量大语言模型的先验信念并比较不同辩论协议，发现模型更擅长辩护符合自身信念的观点，但会倾向迎合法官，且顺序辩论存在偏倚。**

- **链接: [http://arxiv.org/pdf/2510.13912v1](http://arxiv.org/pdf/2510.13912v1)**

> **作者:** María Victoria Carro; Denise Alejandra Mester; Facundo Nieto; Oscar Agustín Stanchi; Guido Ernesto Bergman; Mario Alejandro Leiva; Eitan Sprejer; Luca Nicolás Forziati Gangi; Francisca Gauna Selasco; Juan Gustavo Corvalán; Gerardo I. Simari; María Vanina Martinez
>
> **备注:** 31 pages
>
> **摘要:** The core premise of AI debate as a scalable oversight technique is that it is harder to lie convincingly than to refute a lie, enabling the judge to identify the correct position. Yet, existing debate experiments have relied on datasets with ground truth, where lying is reduced to defending an incorrect proposition. This overlooks a subjective dimension: lying also requires the belief that the claim defended is false. In this work, we apply debate to subjective questions and explicitly measure large language models' prior beliefs before experiments. Debaters were asked to select their preferred position, then presented with a judge persona deliberately designed to conflict with their identified priors. This setup tested whether models would adopt sycophantic strategies, aligning with the judge's presumed perspective to maximize persuasiveness, or remain faithful to their prior beliefs. We implemented and compared two debate protocols, sequential and simultaneous, to evaluate potential systematic biases. Finally, we assessed whether models were more persuasive and produced higher-quality arguments when defending positions consistent with their prior beliefs versus when arguing against them. Our main findings show that models tend to prefer defending stances aligned with the judge persona rather than their prior beliefs, sequential debate introduces significant bias favoring the second debater, models are more persuasive when defending positions aligned with their prior beliefs, and paradoxically, arguments misaligned with prior beliefs are rated as higher quality in pairwise comparison. These results can inform human judges to provide higher-quality training signals and contribute to more aligned AI systems, while revealing important aspects of human-AI interaction regarding persuasion dynamics in language models.
>
---
#### [new 050] A Linguistics-Aware LLM Watermarking via Syntactic Predictability
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究LLM生成文本的公开可验证水印技术，旨在平衡文本质量与检测鲁棒性。提出STELA框架，利用POS n-gram建模语言不确定性，动态调整水印强度，无需模型logits即可检测，支持多语言且提升检测性能。**

- **链接: [http://arxiv.org/pdf/2510.13829v1](http://arxiv.org/pdf/2510.13829v1)**

> **作者:** Shinwoo Park; Hyejin Park; Hyeseon Ahn; Yo-Sub Han
>
> **摘要:** As large language models (LLMs) continue to advance rapidly, reliable governance tools have become critical. Publicly verifiable watermarking is particularly essential for fostering a trustworthy AI ecosystem. A central challenge persists: balancing text quality against detection robustness. Recent studies have sought to navigate this trade-off by leveraging signals from model output distributions (e.g., token-level entropy); however, their reliance on these model-specific signals presents a significant barrier to public verification, as the detection process requires access to the logits of the underlying model. We introduce STELA, a novel framework that aligns watermark strength with the linguistic degrees of freedom inherent in language. STELA dynamically modulates the signal using part-of-speech (POS) n-gram-modeled linguistic indeterminacy, weakening it in grammatically constrained contexts to preserve quality and strengthen it in contexts with greater linguistic flexibility to enhance detectability. Our detector operates without access to any model logits, thus facilitating publicly verifiable detection. Through extensive experiments on typologically diverse languages-analytic English, isolating Chinese, and agglutinative Korean-we show that STELA surpasses prior methods in detection robustness. Our code is available at https://github.com/Shinwoo-Park/stela_watermark.
>
---
#### [new 051] AutoRubric-R1V: Rubric-Based Generative Rewards for Faithful Multimodal Reasoning
- **分类: cs.CL**

- **简介: 该论文针对多模态大模型在强化学习中因仅奖励最终答案导致的推理不忠实问题，提出AutoRubric-R1V框架，通过自动生成评分标准并结合过程监督，提升多步推理的忠实性与性能。**

- **链接: [http://arxiv.org/pdf/2510.14738v1](http://arxiv.org/pdf/2510.14738v1)**

> **作者:** Mengzhao Jia; Zhihan Zhang; Ignacio Cases; Zheyuan Liu; Meng Jiang; Peng Qi
>
> **摘要:** Multimodal large language models (MLLMs) have rapidly advanced from perception tasks to complex multi-step reasoning, yet reinforcement learning with verifiable rewards (RLVR) often leads to spurious reasoning since only the final-answer correctness is rewarded. To address this limitation, we propose AutoRubric-R1V, a framework that integrates RLVR with process-level supervision through automatically collected rubric-based generative rewards. Our key innovation lies in a scalable self-aggregation method that distills consistent reasoning checkpoints from successful trajectories, enabling problem-specific rubric construction without human annotation or stronger teacher models. By jointly leveraging rubric-based and outcome rewards, AutoRubric-R1V achieves state-of-the-art performance on six multimodal reasoning benchmarks and substantially improves reasoning faithfulness in dedicated evaluations.
>
---
#### [new 052] PAGE: Prompt Augmentation for text Generation Enhancement
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出PAGE框架，旨在提升文本生成模型在特定任务下的表现与可控性。通过引入轻量级辅助模块（如分类器）对输入进行增强，构建更丰富的提示信息，从而优化生成质量，无需额外训练生成模型，适用于如软件需求生成等任务。**

- **链接: [http://arxiv.org/pdf/2510.13880v1](http://arxiv.org/pdf/2510.13880v1)**

> **作者:** Mauro Jose Pacchiotti; Luciana Ballejos; Mariel Ale
>
> **备注:** in Spanish language
>
> **摘要:** In recent years, natural language generative models have shown outstanding performance in text generation tasks. However, when facing specific tasks or particular requirements, they may exhibit poor performance or require adjustments that demand large amounts of additional data. This work introduces PAGE (Prompt Augmentation for text Generation Enhancement), a framework designed to assist these models through the use of simple auxiliary modules. These modules, lightweight models such as classifiers or extractors, provide inferences from the input text. The output of these auxiliaries is then used to construct an enriched input that improves the quality and controllability of the generation. Unlike other generation-assistance approaches, PAGE does not require auxiliary generative models; instead, it proposes a simpler, modular architecture that is easy to adapt to different tasks. This paper presents the proposal, its components and architecture, and reports a proof of concept in the domain of requirements engineering, where an auxiliary module with a classifier is used to improve the quality of software requirements generation.
>
---
#### [new 053] Entropy Meets Importance: A Unified Head Importance-Entropy Score for Stable and Efficient Transformer Pruning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究Transformer模型剪枝任务，旨在解决现有基于梯度的重要性评分方法忽略注意力多样性的缺陷。作者提出HIES新标准，融合重要性与注意力熵，提升剪枝后的模型性能与稳定性。**

- **链接: [http://arxiv.org/pdf/2510.13832v1](http://arxiv.org/pdf/2510.13832v1)**

> **作者:** Minsik Choi; Hyegang Son; Changhoon Kim; Young Geun Kim
>
> **备注:** 32 pages
>
> **摘要:** Transformer-based models have achieved remarkable performance in NLP tasks. However, their structural characteristics-multiple layers and attention heads-introduce efficiency challenges in inference and deployment. To address these challenges, various pruning methods have recently been proposed. Notably, gradient-based methods using Head Importance Scores (HIS) have gained traction for interpretability, efficiency, and ability to identify redundant heads. However, HIS alone has limitations as it captures only the gradient-driven contribution, overlooking the diversity of attention patterns. To overcome these limitations, we introduce a novel pruning criterion, HIES (Head Importance-Entropy Score), which integrates head importance scores with attention entropy, providing complementary evidence on per-head contribution. Empirically, HIES-based pruning yields up to 15.2% improvement in model quality and 2.04x improvement in stability over HIS-only methods, enabling substantial model compression without sacrificing either accuracy or stability. Code will be released upon publication.
>
---
#### [new 054] Toward Cybersecurity-Expert Small Language Models
- **分类: cs.CL; cs.AI; cs.CR**

- **简介: 该论文聚焦网络安全领域的小型语言模型（SLM）构建，旨在解决缺乏高质量专业模型与数据的问题。作者提出CyberPal 2.0系列模型及SecKnowledge 2.0数据生成管道，通过专家引导的思维链数据增强，使小模型在多项网络安全任务中超越大模型表现。**

- **链接: [http://arxiv.org/pdf/2510.14113v1](http://arxiv.org/pdf/2510.14113v1)**

> **作者:** Matan Levi; Daniel Ohayon; Ariel Blobstein; Ravid Sagi; Ian Molloy; Yair Allouche
>
> **摘要:** Large language models (LLMs) are transforming everyday applications, yet deployment in cybersecurity lags due to a lack of high-quality, domain-specific models and training datasets. To address this gap, we present CyberPal 2.0, a family of cybersecurity-expert small language models (SLMs) ranging from 4B-20B parameters. To train CyberPal 2.0, we generate an enriched chain-of-thought cybersecurity instruction dataset built with our data enrichment and formatting pipeline, SecKnowledge 2.0, which integrates expert-in-the-loop steering of reasoning formats alongside LLM-driven multi-step grounding, yielding higher-fidelity, task-grounded reasoning traces for security tasks. Across diverse cybersecurity benchmarks, CyberPal 2.0 consistently outperforms its baselines and matches or surpasses various open and closed-source frontier models, while remaining a fraction of their size. On core cyber threat intelligence knowledge tasks, our models outperform almost all tested frontier models, ranking second only to Sec-Gemini v1. On core threat-investigation tasks, such as correlating vulnerabilities and bug tickets with weaknesses, our best 20B-parameter model outperforms GPT-4o, o1, o3-mini, and Sec-Gemini v1, ranking first, while our smallest 4B-parameter model ranks second.
>
---
#### [new 055] Bridging the Semantic Gap: Contrastive Rewards for Multilingual Text-to-SQL
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究多语言Text-to-SQL任务，旨在解决语义对齐不足和跨语言性能下降问题。作者提出结合对比奖励信号的GRPO框架，提升模型对用户意图的理解与生成SQL的语义匹配，在少量样本下显著提高执行和语义准确率。**

- **链接: [http://arxiv.org/pdf/2510.13827v1](http://arxiv.org/pdf/2510.13827v1)**

> **作者:** Ashish Kattamuri; Ishita Prasad; Meetu Malhotra; Arpita Vats; Rahul Raja; Albert Lie
>
> **备注:** 20th International Workshop on Semantic and Social Media Adaptation & Personalization
>
> **摘要:** Current Text-to-SQL methods are evaluated and only focused on executable queries, overlooking the semantic alignment challenge -- both in terms of the semantic meaning of the query and the correctness of the execution results. Even execution accuracy itself shows significant drops when moving from English to other languages, with an average decline of 6 percentage points across non-English languages. We address these challenges by presenting a new framework that combines Group Relative Policy Optimization (GRPO) within a multilingual contrastive reward signal to enhance both task efficiency and semantic accuracy in Text-to-SQL systems in cross-lingual scenarios. Our method teaches models to obtain better correspondence between SQL generation and user intent by combining a reward signal based on semantic similarity. On the seven-language MultiSpider dataset, fine-tuning the LLaMA-3-3B model with GRPO improved the execution accuracy up to 87.4 percent (+26 pp over zero-shot) and semantic accuracy up to 52.29 percent (+32.86 pp). Adding our contrastive reward signal in the GRPO framework further improved the average semantic accuracy to 59.14 percent (+6.85 pp, up to +10 pp for Vietnamese). Our experiments showcase that a smaller, parameter-efficient 3B LLaMA model fine-tuned with our contrastive reward signal outperforms a much larger zero-shot 8B LLaMA model, with an uplift of 7.43 pp in execution accuracy (from 81.43 percent on the 8B model to 88.86 percent on the 3B model), and nearly matches its semantic accuracy (59.14 percent vs. 68.57 percent) -- all using just 3,000 reinforcement learning training examples. These results demonstrate how we can improve the performance of Text-to-SQL systems with contrastive rewards for directed semantic alignment, without requiring large-scale training datasets.
>
---
#### [new 056] Harmonizing Diverse Models: A Layer-wise Merging Strategy for Consistent Generation
- **分类: cs.CL**

- **简介: 该论文针对RAG系统中LLM输出不一致问题，提出一种层间模型融合方法。通过合成数据、三元组损失和基于中间层激活的权重合并，提升生成一致性，在工业RAG场景下实现响应相似度显著提升。**

- **链接: [http://arxiv.org/pdf/2510.14915v1](http://arxiv.org/pdf/2510.14915v1)**

> **作者:** Xujun Peng; Anoop Kumar; Jingyu Wu; Parker Glenn; Daben Liu
>
> **备注:** EMNLP 2025 Industry track
>
> **摘要:** Retrieval-Augmented Generation (RAG) systems leverage Large Language Models (LLMs) to generate accurate and reliable responses that are grounded in retrieved context. However, LLMs often generate inconsistent outputs for semantically equivalent inputs, a problem compounded by the scarcity of consistency-focused training data and the limitations of current fine-tuning techniques in enhancing output consistency. We propose a new approach combining systematic synthetic data generation, triplet loss for better embeddings, and a novel layer-wise model merging approach. Using consistency-aware weights derived from intermediate layer activations, our method effectively integrates knowledge from specialized models. Experimental results how that our merged model significantly enhances output consistency, achieving a ~47.5\% improvement in response similarity over the baseline, thus offering a practical solution for increasing the reliability of an industrial RAG system.
>
---
#### [new 057] Qwen3Guard Technical Report
- **分类: cs.CL**

- **简介: 该论文提出Qwen3Guard，解决大模型安全对齐任务中静态二分类和非流式检测问题。设计生成式三类判断与流式逐token检测两种变体，支持多语言、多尺度，实现细粒度、低延迟的安全防护。**

- **链接: [http://arxiv.org/pdf/2510.14276v1](http://arxiv.org/pdf/2510.14276v1)**

> **作者:** Haiquan Zhao; Chenhan Yuan; Fei Huang; Xiaomeng Hu; Yichang Zhang; An Yang; Bowen Yu; Dayiheng Liu; Jingren Zhou; Junyang Lin; Baosong Yang; Chen Cheng; Jialong Tang; Jiandong Jiang; Jianwei Zhang; Jijie Xu; Ming Yan; Minmin Sun; Pei Zhang; Pengjun Xie; Qiaoyu Tang; Qin Zhu; Rong Zhang; Shibin Wu; Shuo Zhang; Tao He; Tianyi Tang; Tingyu Xia; Wei Liao; Weizhou Shen; Wenbiao Yin; Wenmeng Zhou; Wenyuan Yu; Xiaobin Wang; Xiaodong Deng; Xiaodong Xu; Xinyu Zhang; Yang Liu; Yeqiu Li; Yi Zhang; Yong Jiang; Yu Wan; Yuxin Zhou
>
> **摘要:** As large language models (LLMs) become more capable and widely used, ensuring the safety of their outputs is increasingly critical. Existing guardrail models, though useful in static evaluation settings, face two major limitations in real-world applications: (1) they typically output only binary "safe/unsafe" labels, which can be interpreted inconsistently across diverse safety policies, rendering them incapable of accommodating varying safety tolerances across domains; and (2) they require complete model outputs before performing safety checks, making them fundamentally incompatible with streaming LLM inference, thereby preventing timely intervention during generation and increasing exposure to harmful partial outputs. To address these challenges, we present Qwen3Guard, a series of multilingual safety guardrail models with two specialized variants: Generative Qwen3Guard, which casts safety classification as an instruction-following task to enable fine-grained tri-class judgments (safe, controversial, unsafe); and Stream Qwen3Guard, which introduces a token-level classification head for real-time safety monitoring during incremental text generation. Both variants are available in three sizes (0.6B, 4B, and 8B parameters) and support up to 119 languages and dialects, providing comprehensive, scalable, and low-latency safety moderation for global LLM deployments. Evaluated across English, Chinese, and multilingual benchmarks, Qwen3Guard achieves state-of-the-art performance in both prompt and response safety classification. All models are released under the Apache 2.0 license for public use.
>
---
#### [new 058] Suicidal Comment Tree Dataset: Enhancing Risk Assessment and Prediction Through Contextual Analysis
- **分类: cs.CL**

- **简介: 该论文聚焦自杀风险预测任务，旨在通过分析用户在社交媒体上的评论树序列解决个体风险动态评估不足的问题。作者构建了基于Reddit的标注数据集，并结合大模型实验验证了上下文信息对提升风险识别准确率的有效性。**

- **链接: [http://arxiv.org/pdf/2510.14395v1](http://arxiv.org/pdf/2510.14395v1)**

> **作者:** Jun Li; Qun Zhao
>
> **摘要:** Suicide remains a critical global public health issue. While previous studies have provided valuable insights into detecting suicidal expressions in individual social media posts, limited attention has been paid to the analysis of longitudinal, sequential comment trees for predicting a user's evolving suicidal risk. Users, however, often reveal their intentions through historical posts and interactive comments over time. This study addresses this gap by investigating how the information in comment trees affects both the discrimination and prediction of users' suicidal risk levels. We constructed a high-quality annotated dataset, sourced from Reddit, which incorporates users' posting history and comments, using a refined four-label annotation framework based on the Columbia Suicide Severity Rating Scale (C-SSRS). Statistical analysis of the dataset, along with experimental results from Large Language Models (LLMs) experiments, demonstrates that incorporating comment trees data significantly enhances the discrimination and prediction of user suicidal risk levels. This research offers a novel insight to enhancing the detection accuracy of at-risk individuals, thereby providing a valuable foundation for early suicide intervention strategies.
>
---
#### [new 059] The Harder The Better: Maintaining Supervised Fine-tuning Generalization with Less but Harder Data
- **分类: cs.CL**

- **简介: 该论文针对大模型在专业领域适配时对高质量标注数据的依赖问题，提出THTB框架，通过认知科学启发的方法筛选更难、更具认知价值的指令数据，在仅用5%甚至2%数据下实现超越全量数据训练的性能，提升细调效率与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.13892v1](http://arxiv.org/pdf/2510.13892v1)**

> **作者:** Zhaoyang Shang; Sibo Wei; Jianbin Guo; Rui Zhou; Lifeng Dong; Yin Luo
>
> **摘要:** Large Language Models (LLMs) excel in general tasks, but adapting them to specialized domains relies on high-quality supervised fine-tuning (SFT) data. Although existing methods can identify subsets of high-quality data and reduce training cost to some extent, their selection process still suffers from over-reliance on LLMs' internal knowledge, weak interpretability, and limited generalization. To address these limitations, we propose THTB (The Harder The Better), a cognitive science-inspired framework for instruction data selection and annotation guidance. THTB prioritizes higher-level cognitive instructions by combining quality filtering with intrinsic and extrinsic hardness scoring, offering interpretable and quantifiable criteria for efficient SFT, both in data selection and annotation guidance. Experiments show that THTB enables models trained on only 5% of the data to outperform full-dataset training, while achieving superior generalization compared with LLM-only selection. In addition, THTB provides effective annotation guidance in vertical domains, enabling a model trained on just 2% of the data to surpass models trained on much larger datasets, demonstrating strong potential for domain adaptation. Our code, datasets, and models are available on https://github.com/DYJG-research/THTB.
>
---
#### [new 060] RLAIF-SPA: Optimizing LLM-based Emotional Speech Synthesis via RLAIF
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究情感语音合成任务，旨在解决现有方法因依赖昂贵标注和间接目标导致情感表达不足的问题。提出RLAIF-SPA框架，利用强化学习与AI反馈，结合语义准确性和韵律情感对齐优化，提升合成语音的情感表现力与自然度。**

- **链接: [http://arxiv.org/pdf/2510.14628v1](http://arxiv.org/pdf/2510.14628v1)**

> **作者:** Qing Yang; Zhenghao Liu; Junxin Wang; Yangfan Du; Pengcheng Huang; Tong Xiao
>
> **摘要:** Text-To-Speech synthesis has achieved near-human quality in neutral speech, but emotional expressiveness remains a challenge. Existing methods often rely on costly emotion annotations or optimize indirect objectives that fail to capture the emotional expressiveness and perceptual naturalness of speech, leading to generated speech that is accurate but emotionally flat. To address these challenges, we propose the RLAIF-SPA framework, incorporating a Reinforcement Learning from AI Feedback (RLAIF) mechanism to employ Automatic Speech Recognition (ASR) and Large Language Model (LLM) techniques to respectively judge semantic accuracy and prosodic-emotional label alignment as a direct reward for emotional expressiveness and intelligibility optimization. Specifically, it leverages Prosodic Label Alignment to enhance expressive quality by jointly considering semantic accuracy and prosodic-emotional alignment along four fine-grained dimensions: Structure, Emotion, Speed, and Tone. In addition, it incorporates Semantic Accuracy Feedback to ensure the generation of clear and accurate speech. Experiments on the Libri Speech dataset show that RLAIF-SPA outperforms Chat-TTS, with a 26.1% reduction in WER, a 9.1% increase in SIM-O, and over 10% improvement in human evaluation.
>
---
#### [new 061] Assessing Socio-Cultural Alignment and Technical Safety of Sovereign LLMs
- **分类: cs.CL**

- **简介: 该论文聚焦主权大语言模型的评估任务，旨在解决其在社会文化对齐与技术安全性方面缺乏评估框架的问题。作者构建新数据集并提出分析框架，验证模型在低资源语言支持中的表现及潜在风险，强调需更全面的评估标准。**

- **链接: [http://arxiv.org/pdf/2510.14565v1](http://arxiv.org/pdf/2510.14565v1)**

> **作者:** Kyubyung Chae; Gihoon Kim; Gyuseong Lee; Taesup Kim; Jaejin Lee; Heejin Kim
>
> **摘要:** Recent trends in LLMs development clearly show growing interest in the use and application of sovereign LLMs. The global debate over sovereign LLMs highlights the need for governments to develop their LLMs, tailored to their unique socio-cultural and historical contexts. However, there remains a shortage of frameworks and datasets to verify two critical questions: (1) how well these models align with users' socio-cultural backgrounds, and (2) whether they maintain safety and technical robustness without exposing users to potential harms and risks. To address this gap, we construct a new dataset and introduce an analytic framework for extracting and evaluating the socio-cultural elements of sovereign LLMs, alongside assessments of their technical robustness. Our experimental results demonstrate that while sovereign LLMs play a meaningful role in supporting low-resource languages, they do not always meet the popular claim that these models serve their target users well. We also show that pursuing this untested claim may lead to underestimating critical quality attributes such as safety. Our study suggests that advancing sovereign LLMs requires a more extensive evaluation that incorporates a broader range of well-grounded and practical criteria.
>
---
#### [new 062] An LLM-Powered AI Agent Framework for Holistic IoT Traffic Interpretation
- **分类: cs.CL; cs.CR; cs.NI**

- **简介: 该论文提出一种基于大语言模型的AI代理框架，用于全面解读物联网流量。针对高维、多样的IoT流量分析难题，融合特征提取、异常检测、摘要生成与检索增强问答，实现跨层语义理解，提升解释准确性与可读性。**

- **链接: [http://arxiv.org/pdf/2510.13925v1](http://arxiv.org/pdf/2510.13925v1)**

> **作者:** Daniel Adu Worae; Spyridon Mastorakis
>
> **摘要:** Internet of Things (IoT) networks generate diverse and high-volume traffic that reflects both normal activity and potential threats. Deriving meaningful insight from such telemetry requires cross-layer interpretation of behaviors, protocols, and context rather than isolated detection. This work presents an LLM-powered AI agent framework that converts raw packet captures into structured and semantically enriched representations for interactive analysis. The framework integrates feature extraction, transformer-based anomaly detection, packet and flow summarization, threat intelligence enrichment, and retrieval-augmented question answering. An AI agent guided by a large language model performs reasoning over the indexed traffic artifacts, assembling evidence to produce accurate and human-readable interpretations. Experimental evaluation on multiple IoT captures and six open models shows that hybrid retrieval, which combines lexical and semantic search with reranking, substantially improves BLEU, ROUGE, METEOR, and BERTScore results compared with dense-only retrieval. System profiling further indicates low CPU, GPU, and memory overhead, demonstrating that the framework achieves holistic and efficient interpretation of IoT network traffic.
>
---
#### [new 063] Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究窄域微调在大语言模型激活中留下的可读痕迹，旨在揭示微调偏差及其可解释性。作者通过分析激活差异定位微调影响，提出简单工具即可识别并利用这些痕迹，警告当前研究可能高估窄域微调模型的代表性，并建议改进训练方法以降低偏差风险。**

- **链接: [http://arxiv.org/pdf/2510.13900v1](http://arxiv.org/pdf/2510.13900v1)**

> **作者:** Julian Minder; Clément Dumas; Stewart Slocum; Helena Casademunt; Cameron Holmes; Robert West; Neel Nanda
>
> **摘要:** Finetuning on narrow domains has become an essential tool to adapt Large Language Models (LLMs) to specific tasks and to create models with known unusual properties that are useful for research. We show that narrow finetuning creates strong biases in LLM activations that can be interpreted to understand the finetuning domain. These biases can be discovered using simple tools from model diffing - the study of differences between models before and after finetuning. In particular, analyzing activation differences on the first few tokens of random text and steering by adding this difference to the model activations produces text similar to the format and general content of the finetuning data. We demonstrate that these analyses contain crucial information by creating an LLM-based interpretability agent to understand the finetuning domain. With access to the bias, the agent performs significantly better compared to baseline agents using simple prompting. Our analysis spans synthetic document finetuning for false facts, emergent misalignment, subliminal learning, and taboo word guessing game models across different architectures (Gemma, LLaMA, Qwen) and scales (1B to 32B parameters). We suspect these biases reflect overfitting and find that mixing pretraining data into the finetuning corpus largely removes them, though residual risks may remain. Our work (1) demonstrates that narrowly finetuned models have salient traces of their training objective in their activations and suggests ways to improve how they are trained, (2) warns AI safety and interpretability researchers that the common practice of using such models as a proxy for studying broader finetuning (e.g., chat-tuning) might not be realistic, and (3) highlights the need for deeper investigation into the effects of narrow finetuning and development of truly realistic case studies for model-diffing, safety and interpretability research.
>
---
#### [new 064] FRACCO: A gold-standard annotated corpus of oncological entities with ICD-O-3.1 normalisation
- **分类: cs.CL; cs.AI**

- **简介: 该论文构建了法语肿瘤学文本的金标准标注语料库FRACCO，解决法语临床自然语言处理资源稀缺问题。任务为命名实体识别与概念标准化，涵盖形态、部位及组织分化，采用ICD-O-3.1编码，结合自动匹配与人工验证，提供高质量标注数据。**

- **链接: [http://arxiv.org/pdf/2510.13873v1](http://arxiv.org/pdf/2510.13873v1)**

> **作者:** Johann Pignat; Milena Vucetic; Christophe Gaudet-Blavignac; Jamil Zaghir; Amandine Stettler; Fanny Amrein; Jonatan Bonjour; Jean-Philippe Goldman; Olivier Michielin; Christian Lovis; Mina Bjelogrlic
>
> **摘要:** Developing natural language processing tools for clinical text requires annotated datasets, yet French oncology resources remain scarce. We present FRACCO (FRench Annotated Corpus for Clinical Oncology) an expert-annotated corpus of 1301 synthetic French clinical cases, initially translated from the Spanish CANTEMIST corpus as part of the FRASIMED initiative. Each document is annotated with terms related to morphology, topography, and histologic differentiation, using the International Classification of Diseases for Oncology (ICD-O) as reference. An additional annotation layer captures composite expression-level normalisations that combine multiple ICD-O elements into unified clinical concepts. Annotation quality was ensured through expert review: 1301 texts were manually annotated for entity spans by two domain experts. A total of 71127 ICD-O normalisations were produced through a combination of automated matching and manual validation by a team of five annotators. The final dataset representing 399 unique morphology codes (from 2549 different expressions), 272 topography codes (from 3143 different expressions), and 2043 unique composite expressions (from 11144 different expressions). This dataset provides a reference standard for named entity recognition and concept normalisation in French oncology texts.
>
---
#### [new 065] Serialized EHR make for good text representations
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出SerialBEHRT模型，解决EHR数据结构与语言模型序列先验不匹配的问题。通过将EHR序列化并预训练，增强时间建模，提升患者表征质量，在抗生素敏感性预测任务中优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.13843v1](http://arxiv.org/pdf/2510.13843v1)**

> **作者:** Zhirong Chou; Quan Qin; Shi Li
>
> **摘要:** The emergence of foundation models in healthcare has opened new avenues for learning generalizable representations from large scale clinical data. Yet, existing approaches often struggle to reconcile the tabular and event based nature of Electronic Health Records (EHRs) with the sequential priors of natural language models. This structural mismatch limits their ability to capture longitudinal dependencies across patient encounters. We introduce SerialBEHRT, a domain aligned foundation model that extends SciBERT through additional pretraining on structured EHR sequences. SerialBEHRT is designed to encode temporal and contextual relationships among clinical events, thereby producing richer patient representations. We evaluate its effectiveness on the task of antibiotic susceptibility prediction, a clinically meaningful problem in antibiotic stewardship. Through extensive benchmarking against state of the art EHR representation strategies, we demonstrate that SerialBEHRT achieves superior and more consistent performance, highlighting the importance of temporal serialization in foundation model pretraining for healthcare.
>
---
#### [new 066] Attention Is All You Need for KV Cache in Diffusion LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对扩散大语言模型解码中KV缓存重复计算问题，提出无需训练的Elastic-Cache方法，通过注意力感知的动态检测和深度感知更新策略，自适应决定何时、何处刷新缓存，在大幅降低延迟的同时保持生成质量。**

- **链接: [http://arxiv.org/pdf/2510.14973v1](http://arxiv.org/pdf/2510.14973v1)**

> **作者:** Quan Nguyen-Tri; Mukul Ranjan; Zhiqiang Shen
>
> **备注:** https://vila-lab.github.io/elastic-cache-webpage/
>
> **摘要:** This work studies how to adaptively recompute key-value (KV) caches for diffusion large language models (DLMs) to maximize prediction accuracy while minimizing decoding latency. Prior methods' decoders recompute QKV for all tokens at every denoising step and layer, despite KV states changing little across most steps, especially in shallow layers, leading to substantial redundancy. We make three observations: (1) distant ${\bf MASK}$ tokens primarily act as a length-bias and can be cached block-wise beyond the active prediction window; (2) KV dynamics increase with depth, suggesting that selective refresh starting from deeper layers is sufficient; and (3) the most-attended token exhibits the smallest KV drift, providing a conservative lower bound on cache change for other tokens. Building on these, we propose ${\bf Elastic-Cache}$, a training-free, architecture-agnostic strategy that jointly decides ${when}$ to refresh (via an attention-aware drift test on the most-attended token) and ${where}$ to refresh (via a depth-aware schedule that recomputes from a chosen layer onward while reusing shallow-layer caches and off-window MASK caches). Unlike fixed-period schemes, Elastic-Cache performs adaptive, layer-aware cache updates for diffusion LLMs, reducing redundant computation and accelerating decoding with negligible loss in generation quality. Experiments on LLaDA-Instruct, LLaDA-1.5, and LLaDA-V across mathematical reasoning and code generation tasks demonstrate consistent speedups: $8.7\times$ on GSM8K (256 tokens), $45.1\times$ on longer sequences, and $4.8\times$ on HumanEval, while consistently maintaining higher accuracy than the baseline. Our method achieves significantly higher throughput ($6.8\times$ on GSM8K) than existing confidence-based approaches while preserving generation quality, enabling practical deployment of diffusion LLMs.
>
---
#### [new 067] Semantic Prosody in Machine Translation: the English-Chinese Case of Passive Structures
- **分类: cs.CL**

- **简介: 该论文研究机器翻译中被动结构的语义韵问题，旨在解决字面翻译导致语义韵不一致的问题。作者构建了英汉句对数据集，通过微调主流MT模型，提升其在翻译负面语境时使用“被”字句的准确性，并发现语义韵知识可跨语言迁移。**

- **链接: [http://arxiv.org/pdf/2510.14662v1](http://arxiv.org/pdf/2510.14662v1)**

> **作者:** Xinyue Ma; Pol Pastells; Mireia Farrús; Mariona Taulé
>
> **备注:** 11 pages, 2 figures, *SEM workshop at EMNLP 2025 conference
>
> **摘要:** Semantic prosody is a collocational meaning formed through the co-occurrence of a linguistic unit and a consistent series of collocates, which should be treated separately from semantic meaning. Since words that are literal translations of each other may have different semantic prosody, more attention should be paid to this linguistic property to generate accurate translations. However, current machine translation models cannot handle this problem. To bridge the gap, we propose an approach to teach machine translation models about semantic prosody of a specific structure. We focus on Chinese BEI passives and create a dataset of English-Chinese sentence pairs with the purpose of demonstrating the negative semantic prosody of BEI passives. Then we fine-tune OPUS-MT, NLLB-600M and mBART50 models with our dataset for the English-Chinese translation task. Our results show that fine-tuned MT models perform better on using BEI passives for translating unfavourable content and avoid using it for neutral and favourable content. Also, in NLLB-600M, which is a multilingual model, this knowledge of semantic prosody can be transferred from English-Chinese translation to other language pairs, such as Spanish-Chinese.
>
---
#### [new 068] SIMBA UQ: Similarity-Based Aggregation for Uncertainty Quantification in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型的不确定性量化（UQ），旨在通过黑箱方法评估模型输出的置信度。提出基于生成结果间相似性的聚合框架，利用输出一致性作为置信度代理，提升UQ校准效果，适用于问答、摘要和文本到SQL等任务。**

- **链接: [http://arxiv.org/pdf/2510.13836v1](http://arxiv.org/pdf/2510.13836v1)**

> **作者:** Debarun Bhattacharjya; Balaji Ganesan; Junkyu Lee; Radu Marinescu; Katsiaryna Mirylenka; Michael Glass; Xiao Shou
>
> **备注:** 15 pages including appendix, Findings of EMNLP 2025
>
> **摘要:** When does a large language model (LLM) know what it does not know? Uncertainty quantification (UQ) provides measures of uncertainty, such as an estimate of the confidence in an LLM's generated output, and is therefore increasingly recognized as a crucial component of trusted AI systems. Black-box UQ methods do not require access to internal model information from the generating LLM and therefore have numerous real-world advantages, such as robustness to system changes, adaptability to choice of LLM, reduced costs, and computational tractability. In this paper, we investigate the effectiveness of UQ techniques that are primarily but not necessarily entirely black-box, where the consistency between a generated output and other sampled generations is used as a proxy for confidence in its correctness. We propose a high-level non-verbalized similarity-based aggregation framework that subsumes a broad swath of UQ approaches suitable for complex generative tasks, as well as introduce specific novel techniques from the framework that train confidence estimation models using small training sets. Through an empirical study with datasets spanning the diverse tasks of question answering, summarization, and text-to-SQL, we demonstrate that our proposed similarity-based methods can yield better calibrated confidences than baselines.
>
---
#### [new 069] ConDABench: Interactive Evaluation of Language Models for Data Analysis
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对现实数据分析中目标模糊、数据不洁及需交互的问题，提出ConDABench框架，生成1420个对话式数据分析任务，并构建评估系统，首次实现对大模型在复杂交互分析任务中的系统评测。**

- **链接: [http://arxiv.org/pdf/2510.13835v1](http://arxiv.org/pdf/2510.13835v1)**

> **作者:** Avik Dutta; Priyanshu Gupta; Hosein Hasanbeig; Rahul Pratap Singh; Harshit Nigam; Sumit Gulwani; Arjun Radhakrishna; Gustavo Soares; Ashish Tiwari
>
> **摘要:** Real-world data analysis tasks often come with under-specified goals and unclean data. User interaction is necessary to understand and disambiguate a user's intent, and hence, essential to solving these complex tasks. Existing benchmarks for evaluating LLMs on data analysis tasks do not capture these complexities or provide first-class support for interactivity. We introduce ConDABench, a framework for generating conversational data analysis (ConDA) benchmarks and evaluating external tools on the generated benchmarks. \bench consists of (a) a multi-agent workflow for generating realistic benchmarks from articles describing insights gained from public datasets, (b) 1,420 ConDA problems generated using this workflow, and (c) an evaluation harness that, for the first time, makes it possible to systematically evaluate conversational data analysis tools on the generated ConDA problems. Evaluation of state-of-the-art LLMs on the benchmarks reveals that while the new generation of models are better at solving more instances, they are not necessarily better at solving tasks that require sustained, long-form engagement. ConDABench is an avenue for model builders to measure progress towards truly collaborative models that can complete complex interactive tasks.
>
---
#### [new 070] Beyond Correctness: Evaluating Subjective Writing Preferences Across Cultures
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究跨文化主观写作偏好评估任务，旨在解决现有偏好学习方法难以捕捉主观质量的问题。作者构建了WritingPreferenceBench数据集，发现传统奖励模型表现不佳，而生成式推理模型显著更优，表明需通过中间推理过程建模主观偏好。**

- **链接: [http://arxiv.org/pdf/2510.14616v1](http://arxiv.org/pdf/2510.14616v1)**

> **作者:** Shuangshuang Ying; Yunwen Li; Xingwei Qu; Xin Li; Sheng Jin; Minghao Liu; Zhoufutu Wen; Xeron Du; Tianyu Zheng; Yichi Zhang; Letian Ni; Yuyang Cheng; Qiguang Chen; Jingzhe Ding; Shengda Long; Wangchunshu Zhou; Jiazhan Feng; Wanjun Zhong; Libo Qin; Ge Zhang; Wenhao Huang; Wanxiang Che; Chenghua Lin
>
> **摘要:** Current preference learning methods achieve high accuracy on standard benchmarks but exhibit significant performance degradation when objective quality signals are removed. We introduce WritingPreferenceBench, a dataset of 1,800 human-annotated preference pairs (1,200 English, 600 Chinese) across 8 creative writing genres, where responses are matched for objective correctness, factual accuracy, and length. On this benchmark, sequence-based reward models--the standard architecture for RLHF--achieve only 52.7% mean accuracy, while zero-shot language model judges perform at 53.9%. In contrast, generative reward models that produce explicit reasoning chains achieve 81.8% accuracy. We observe high within-model variance across genres: individual models range from 18.2% to 81.8% accuracy across different writing categories, with standard deviations averaging 10.1%. This variance persists regardless of model scale, with 27B parameter models showing no consistent improvement over 8B variants. Our results suggest that current RLHF methods primarily learn to detect objective errors rather than capture subjective quality preferences (e.g., creativity, stylistic flair, and emotional resonance), and that successful preference modeling may require intermediate reasoning representations rather than direct classification.
>
---
#### [new 071] Catch Your Breath: Adaptive Computation for Self-Paced Sequence Production
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型在序列生成中动态调整计算步数的问题，提出“Catch Your Breath”损失函数，使模型可通过输出<don't know>请求额外计算。实验表明其能自适应复杂度，用更少数据达到相同性能。**

- **链接: [http://arxiv.org/pdf/2510.13879v1](http://arxiv.org/pdf/2510.13879v1)**

> **作者:** Alexandre Galashov; Matt Jones; Rosemary Ke; Yuan Cao; Vaishnavh Nagarajan; Michael C. Mozer
>
> **摘要:** We explore a class of supervised training objectives that allow a language model to dynamically and autonomously scale the number of compute steps used for each input token. For any token, the model can request additional compute steps by emitting a <don't know> output. If the model is granted a delay, a specialized <pause> token is inserted at the next input step, providing the model with additional compute resources to generate an output. The model can request multiple pauses. To train the model to use <don't know> outputs judiciously and to calibrate its uncertainty, we frame the selection of each output token as a sequential-decision problem with a time cost. We refer to the class of methods as $\textit{Catch Your Breath}$ losses and we study three methods in this class: CYB-AP frames the model's task as anytime prediction, where an output may be required at any step and accuracy is discounted over time; CYB-VA is a variational approach that aims to maximize prediction accuracy subject to a specified distribution over stopping times; and CYB-DP imposes a penalty based on a computational budget. Through fine-tuning experiments, we identify the best performing loss variant. The CYB model needs only one third as much training data as the baseline (no pause) model needs to achieve the same performance, and half as much data as a model with pauses and a cross-entropy loss. We find that the CYB model requests additional steps when doing so improves accuracy, and the model adapts its processing time to token-level complexity and context. For example, it often pauses after plural nouns like $\textit{patients}$ and $\textit{challenges}$ but never pauses after the first token of contracted words like $\textit{wasn}$ and $\textit{didn}$, and it shows high variability for ambiguous tokens like $\textit{won}$, which could function as either a verb or part of a contraction.
>
---
#### [new 072] Beyond One World: Benchmarking Super Heros in Role-Playing Across Multiversal Contexts
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦大模型在多宇宙背景下角色扮演的一致性问题，提出“超越单一世界”基准，包含30个英雄的90个版本，通过事实回忆与道德困境任务评估模型的 canon 准确性与思行一致性，揭示当前模型在跨版本泛化和推理对齐上的不足。**

- **链接: [http://arxiv.org/pdf/2510.14351v1](http://arxiv.org/pdf/2510.14351v1)**

> **作者:** Perapard Ngokpol; Kun Kerdthaisong; Pasin Buakhaw; Pitikorn Khlaisamniang; Supasate Vorathammathorn; Piyalitt Ittichaiwong; Nutchanon Yongsatianchot
>
> **摘要:** Large language models (LLMs) are increasingly used as role-playing agents, yet their capacity to faithfully and consistently portray version-specific characters -- for example, superheroes across comic and cinematic universes -- remains underexplored. Superhero canons such as Marvel and DC provide a rich testbed: decades of storytelling yield multiple incarnations of the same character with distinct histories, values, and moral codes. To study this problem, we introduce Beyond One World, a benchmark for character-grounded roleplay spanning 30 iconic heroes and 90 canon-specific versions. The benchmark comprises two tasks: (i) Canon Events, which probes factual recall of pivotal life stages, and (ii) Moral Dilemmas, which confronts models with ethically charged scenarios. We score responses for canonical accuracy and reasoning fidelity under a framework that separates internal deliberation ("thinking") from outward decisions ("acting"). We further propose Think-Act Matching, a metric that quantifies alignment between reasons and actions and serves as a proxy for model trustworthiness. Experiments across reasoning- and non-reasoning-oriented models yield three findings: (1) chain-of-thought prompting improves narrative coherence in weaker models but can reduce canonical accuracy in stronger ones; (2) cross-version generalization within a character remains a major obstacle; and (3) models often excel at either thinking or acting, but rarely both. Beyond One World exposes critical gaps in multiversal consistency and reasoning alignment, offering a challenging evaluation for role-playing LLMs.
>
---
#### [new 073] RAID: Refusal-Aware and Integrated Decoding for Jailbreaking LLMs
- **分类: cs.CL**

- **简介: 该论文针对大语言模型的越狱攻击问题，提出RAID框架。通过在嵌入空间优化对抗后缀，结合拒绝感知正则化与连贯性约束，并采用批评引导解码，生成自然且高效的越狱后缀，提升攻击成功率并降低成本。**

- **链接: [http://arxiv.org/pdf/2510.13901v1](http://arxiv.org/pdf/2510.13901v1)**

> **作者:** Tuan T. Nguyen; John Le; Thai T. Vu; Willy Susilo; Heath Cooper
>
> **摘要:** Large language models (LLMs) achieve impressive performance across diverse tasks yet remain vulnerable to jailbreak attacks that bypass safety mechanisms. We present RAID (Refusal-Aware and Integrated Decoding), a framework that systematically probes these weaknesses by crafting adversarial suffixes that induce restricted content while preserving fluency. RAID relaxes discrete tokens into continuous embeddings and optimizes them with a joint objective that (i) encourages restricted responses, (ii) incorporates a refusal-aware regularizer to steer activations away from refusal directions in embedding space, and (iii) applies a coherence term to maintain semantic plausibility and non-redundancy. After optimization, a critic-guided decoding procedure maps embeddings back to tokens by balancing embedding affinity with language-model likelihood. This integration yields suffixes that are both effective in bypassing defenses and natural in form. Experiments on multiple open-source LLMs show that RAID achieves higher attack success rates with fewer queries and lower computational cost than recent white-box and black-box baselines. These findings highlight the importance of embedding-space regularization for understanding and mitigating LLM jailbreak vulnerabilities.
>
---
#### [new 074] Rethinking Schema Linking: A Context-Aware Bidirectional Retrieval Approach for Text-to-SQL
- **分类: cs.CL; cs.IR**

- **简介: 该论文针对Text-to-SQL中的模式链接问题，提出一种上下文感知的双向检索框架，通过双路径检索与关键词提取提升模式召回率、降低误检，显著缩小全模式与理想模式间的性能差距，增强SQL生成准确率。**

- **链接: [http://arxiv.org/pdf/2510.14296v1](http://arxiv.org/pdf/2510.14296v1)**

> **作者:** Md Mahadi Hasan Nahid; Davood Rafiei; Weiwei Zhang; Yong Zhang
>
> **备注:** 30 Pages
>
> **摘要:** Schema linking -- the process of aligning natural language questions with database schema elements -- is a critical yet underexplored component of Text-to-SQL systems. While recent methods have focused primarily on improving SQL generation, they often neglect the retrieval of relevant schema elements, which can lead to hallucinations and execution failures. In this work, we propose a context-aware bidirectional schema retrieval framework that treats schema linking as a standalone problem. Our approach combines two complementary strategies: table-first retrieval followed by column selection, and column-first retrieval followed by table selection. It is further augmented with techniques such as question decomposition, keyword extraction, and keyphrase extraction. Through comprehensive evaluations on challenging benchmarks such as BIRD and Spider, we demonstrate that our method significantly improves schema recall while reducing false positives. Moreover, SQL generation using our retrieved schema consistently outperforms full-schema baselines and closely approaches oracle performance, all without requiring query refinement. Notably, our method narrows the performance gap between full and perfect schema settings by 50\%. Our findings highlight schema linking as a powerful lever for enhancing Text-to-SQL accuracy and efficiency.
>
---
#### [new 075] What Layers When: Learning to Skip Compute in LLMs with Residual Gates
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出GateSkip，旨在减少大语言模型推理计算量。通过在残差流中引入可学习门控机制，实现按token重要性跳过特定层，兼顾效率与精度，适用于长文本推理与指令模型，兼容多种优化技术。**

- **链接: [http://arxiv.org/pdf/2510.13876v1](http://arxiv.org/pdf/2510.13876v1)**

> **作者:** Filipe Laitenberger; Dawid Kopiczko; Cees G. M. Snoek; Yuki M. Asano
>
> **备注:** Preprint
>
> **摘要:** We introduce GateSkip, a simple residual-stream gating mechanism that enables token-wise layer skipping in decoder-only LMs. Each Attention/MLP branch is equipped with a sigmoid-linear gate that condenses the branch's output before it re-enters the residual stream. During inference we rank tokens by the gate values and skip low-importance ones using a per-layer budget. While early-exit or router-based Mixture-of-Depths models are known to be unstable and need extensive retraining, our smooth, differentiable gates fine-tune stably on top of pretrained models. On long-form reasoning, we save up to 15\% compute while retaining over 90\% of baseline accuracy. On instruction-tuned models we see accuracy gains at full compute and match baseline quality near 50\% savings. The learned gates give insight into transformer information flow (e.g., BOS tokens act as anchors), and the method combines easily with quantization, pruning, and self-speculative decoding.
>
---
#### [new 076] Big Reasoning with Small Models: Instruction Retrieval at Inference Time
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究小语言模型的推理能力提升，提出推理时通过检索外部指令来辅助推理。构建指令语料库，在推理时检索并执行指令，显著提升小模型在医学、法律和数学任务上的表现。**

- **链接: [http://arxiv.org/pdf/2510.13935v1](http://arxiv.org/pdf/2510.13935v1)**

> **作者:** Kenan Alkiek; David Jurgens; Vinod Vydiswaran
>
> **摘要:** Can we bring large-scale reasoning to local-scale compute? Small language models (SLMs) are increasingly attractive because they run efficiently on local hardware, offering strong privacy, low cost, and reduced environmental impact. Yet they often struggle with tasks that require multi-step reasoning or domain-specific knowledge. We address this limitation through instruction intervention at inference time, where an SLM retrieves structured reasoning procedures rather than generating them from scratch. Our method builds an Instruction Corpus by grouping similar training questions and creating instructions via GPT-5. During inference, the SLM retrieves the most relevant instructions and follows their steps. Unlike retrieval-augmented generation, which retrieves text passages, instruction retrieval gives the model structured guidance for reasoning. We evaluate this framework on MedQA (medical board exams), MMLU Professional Law, and MathQA using models from 3B to 14B parameters without any additional fine-tuning. Instruction retrieval yields consistent gains: 9.4% on MedQA, 7.9% on MMLU Law, and 5.1% on MathQA. Concise instructions outperform longer ones, and the magnitude of improvement depends strongly on model family and intrinsic reasoning ability.
>
---
#### [new 077] Instructions are all you need: Self-supervised Reinforcement Learning for Instruction Following
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究指令跟随任务，旨在解决多约束指令执行中外部监督依赖和稀疏奖励问题。提出一种无标签自监督强化学习框架，通过指令生成奖励信号和伪标签，结合约束分解与二分类策略，提升模型在多领域指令跟随中的泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.14420v1](http://arxiv.org/pdf/2510.14420v1)**

> **作者:** Qingyu Ren; Qianyu He; Bowei Zhang; Jie Zeng; Jiaqing Liang; Yanghua Xiao; Weikang Zhou; Zeye Sun; Fei Yu
>
> **摘要:** Language models often struggle to follow multi-constraint instructions that are crucial for real-world applications. Existing reinforcement learning (RL) approaches suffer from dependency on external supervision and sparse reward signals from multi-constraint tasks. We propose a label-free self-supervised RL framework that eliminates dependency on external supervision by deriving reward signals directly from instructions and generating pseudo-labels for reward model training. Our approach introduces constraint decomposition strategies and efficient constraint-wise binary classification to address sparse reward challenges while maintaining computational efficiency. Experiments show that our approach generalizes well, achieving strong improvements across 3 in-domain and 5 out-of-domain datasets, including challenging agentic and multi-turn instruction following. The data and code are publicly available at https://github.com/Rainier-rq/verl-if
>
---
#### [new 078] ADMIT: Few-shot Knowledge Poisoning Attacks on RAG-based Fact Checking
- **分类: cs.CL; cs.AI; cs.CR**

- **简介: 该论文研究RAG系统在事实核查中的知识投毒攻击，提出ADMIT方法，通过少量语义对齐的恶意注入，误导模型生成错误结论。无需访问目标模型或检索器，即可实现高成功率攻击，揭示了现实场景下RAG系统的严重安全漏洞。**

- **链接: [http://arxiv.org/pdf/2510.13842v1](http://arxiv.org/pdf/2510.13842v1)**

> **作者:** Yutao Wu; Xiao Liu; Yinghui Li; Yifeng Gao; Yifan Ding; Jiale Ding; Xiang Zheng; Xingjun Ma
>
> **摘要:** Knowledge poisoning poses a critical threat to Retrieval-Augmented Generation (RAG) systems by injecting adversarial content into knowledge bases, tricking Large Language Models (LLMs) into producing attacker-controlled outputs grounded in manipulated context. Prior work highlights LLMs' susceptibility to misleading or malicious retrieved content. However, real-world fact-checking scenarios are more challenging, as credible evidence typically dominates the retrieval pool. To investigate this problem, we extend knowledge poisoning to the fact-checking setting, where retrieved context includes authentic supporting or refuting evidence. We propose \textbf{ADMIT} (\textbf{AD}versarial \textbf{M}ulti-\textbf{I}njection \textbf{T}echnique), a few-shot, semantically aligned poisoning attack that flips fact-checking decisions and induces deceptive justifications, all without access to the target LLMs, retrievers, or token-level control. Extensive experiments show that ADMIT transfers effectively across 4 retrievers, 11 LLMs, and 4 cross-domain benchmarks, achieving an average attack success rate (ASR) of 86\% at an extremely low poisoning rate of $0.93 \times 10^{-6}$, and remaining robust even in the presence of strong counter-evidence. Compared with prior state-of-the-art attacks, ADMIT improves ASR by 11.2\% across all settings, exposing significant vulnerabilities in real-world RAG-based fact-checking systems.
>
---
#### [new 079] BioMedSearch: A Multi-Source Biomedical Retrieval Framework Based on LLMs
- **分类: cs.CL**

- **简介: 该论文提出BioMedSearch，解决大模型在生物医学问答中缺乏科学严谨性的问题。通过整合文献、数据库与网络搜索，结合子查询分解与多源信息过滤，提升复杂生物医学问题的检索与推理准确性。**

- **链接: [http://arxiv.org/pdf/2510.13926v1](http://arxiv.org/pdf/2510.13926v1)**

> **作者:** Congying Liu; Xingyuan Wei; Peipei Liu; Yiqing Shen; Yanxu Mao; Tiehan Cui
>
> **摘要:** Biomedical queries often rely on a deep understanding of specialized knowledge such as gene regulatory mechanisms and pathological processes of diseases. They require detailed analysis of complex physiological processes and effective integration of information from multiple data sources to support accurate retrieval and reasoning. Although large language models (LLMs) perform well in general reasoning tasks, their generated biomedical content often lacks scientific rigor due to the inability to access authoritative biomedical databases and frequently fabricates protein functions, interactions, and structural details that deviate from authentic information. Therefore, we present BioMedSearch, a multi-source biomedical information retrieval framework based on LLMs. The method integrates literature retrieval, protein database and web search access to support accurate and efficient handling of complex biomedical queries. Through sub-queries decomposition, keywords extraction, task graph construction, and multi-source information filtering, BioMedSearch generates high-quality question-answering results. To evaluate the accuracy of question answering, we constructed a multi-level dataset, BioMedMCQs, consisting of 3,000 questions. The dataset covers three levels of reasoning: mechanistic identification, non-adjacent semantic integration, and temporal causal reasoning, and is used to assess the performance of BioMedSearch and other methods on complex QA tasks. Experimental results demonstrate that BioMedSearch consistently improves accuracy over all baseline models across all levels. Specifically, at Level 1, the average accuracy increases from 59.1% to 91.9%; at Level 2, it rises from 47.0% to 81.0%; and at the most challenging Level 3, the average accuracy improves from 36.3% to 73.4%. The code and BioMedMCQs are available at: https://github.com/CyL-ucas/BioMed_Search
>
---
#### [new 080] ConsistencyAI: A Benchmark to Assess LLMs' Factual Consistency When Responding to Different Demographic Groups
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文提出ConsistencyAI基准，评估大语言模型在不同人群提问下的事实一致性。通过构建多样化 personas，测试19个LLM回答的 factual consistency，发现模型、话题均影响结果，揭示了回答偏差问题，并开源工具促进公平性研究。**

- **链接: [http://arxiv.org/pdf/2510.13852v1](http://arxiv.org/pdf/2510.13852v1)**

> **作者:** Peter Banyas; Shristi Sharma; Alistair Simmons; Atharva Vispute
>
> **备注:** For associated code repository, see http://github.com/banyasp/consistencyAI For user-friendly web app, see http://v0-llm-comparison-webapp.vercel.app/
>
> **摘要:** Is an LLM telling you different facts than it's telling me? This paper introduces ConsistencyAI, an independent benchmark for measuring the factual consistency of large language models (LLMs) for different personas. ConsistencyAI tests whether, when users of different demographics ask identical questions, the model responds with factually inconsistent answers. Designed without involvement from LLM providers, this benchmark offers impartial evaluation and accountability. In our experiment, we queried 19 LLMs with prompts that requested 5 facts for each of 15 topics. We repeated this query 100 times for each LLM, each time adding prompt context from a different persona selected from a subset of personas modeling the general population. We processed the responses into sentence embeddings, computed cross-persona cosine similarity, and computed the weighted average of cross-persona cosine similarity to calculate factual consistency scores. In 100-persona experiments, scores ranged from 0.9065 to 0.7896, and the mean was 0.8656, which we adopt as a benchmark threshold. xAI's Grok-3 is most consistent, while several lightweight models rank lowest. Consistency varies by topic: the job market is least consistent, G7 world leaders most consistent, and issues like vaccines or the Israeli-Palestinian conflict diverge by provider. These results show that both the provider and the topic shape the factual consistency. We release our code and interactive demo to support reproducible evaluation and encourage persona-invariant prompting strategies.
>
---
#### [new 081] Quechua Speech Datasets in Common Voice: The Case of Puno Quechua
- **分类: cs.CL**

- **简介: 该论文属语音数据建设任务，旨在解决低资源语言Quechua的语音数据稀缺问题。工作包括推动17种Quechua语言进入Common Voice平台，以Puno Quechua为案例完成语言接入与读音及自发语音语料收集，共贡献12小时数据，并提出技术与伦理研究方向。**

- **链接: [http://arxiv.org/pdf/2510.13871v1](http://arxiv.org/pdf/2510.13871v1)**

> **作者:** Elwin Huaman; Wendi Huaman; Jorge Luis Huaman; Ninfa Quispe
>
> **备注:** to be published in the 9th Annual International Conference on Information Management and Big Data (SIMBig 2025)
>
> **摘要:** Under-resourced languages, such as Quechuas, face data and resource scarcity, hindering their development in speech technology. To address this issue, Common Voice presents a crucial opportunity to foster an open and community-driven speech dataset creation. This paper examines the integration of Quechua languages into Common Voice. We detail the current 17 Quechua languages, presenting Puno Quechua (ISO 639-3: qxp) as a focused case study that includes language onboarding and corpus collection of both reading and spontaneous speech data. Our results demonstrate that Common Voice now hosts 191.1 hours of Quechua speech (86\% validated), with Puno Quechua contributing 12 hours (77\% validated), highlighting the Common Voice's potential. We further propose a research agenda addressing technical challenges, alongside ethical considerations for community engagement and indigenous data sovereignty. Our work contributes towards inclusive voice technology and digital empowerment of under-resourced language communities.
>
---
#### [new 082] A Robust Classification Method using Hybrid Word Embedding for Early Diagnosis of Alzheimer's Disease
- **分类: cs.CL; cs.AI; cs.LG; eess.AS; I.2.7; I.2.6**

- **简介: 该论文针对阿尔茨海默病早期诊断，提出一种融合Doc2Vec与ELMo的混合词嵌入分类方法。通过句子困惑度和语言特征分析语言能力变化，结合逻辑回归与超参数优化，实现91%准确率和97% AUC，优于现有模型，具备稳定性与临床应用潜力。**

- **链接: [http://arxiv.org/pdf/2510.14332v1](http://arxiv.org/pdf/2510.14332v1)**

> **作者:** Yangyang Li
>
> **备注:** Peer-reviewed and published in Proceedings of the 2020 3rd International Conference on Algorithms, Computing and Artificial Intelligence (ACAI 2020). 7 pages, 5 figures
>
> **摘要:** Early detection of Alzheimer's Disease (AD) is greatly beneficial to AD patients, leading to early treatments that lessen symptoms and alleviating financial burden of health care. As one of the leading signs of AD, language capability changes can be used for early diagnosis of AD. In this paper, I develop a robust classification method using hybrid word embedding and fine-tuned hyperparameters to achieve state-of-the-art accuracy in the early detection of AD. Specifically, we create a hybrid word embedding based on word vectors from Doc2Vec and ELMo to obtain perplexity scores of the sentences. The scores identify whether a sentence is fluent or not and capture semantic context of the sentences. I enrich the word embedding by adding linguistic features to analyze syntax and semantics. Further, we input an embedded feature vector into logistic regression and fine tune hyperparameters throughout the pipeline. By tuning hyperparameters of the machine learning pipeline (e.g., model regularization parameter, learning rate and vector size of Doc2Vec, and vector size of ELMo), I achieve 91% classification accuracy and an Area Under the Curve (AUC) of 97% in distinguishing early AD from healthy subjects. Based on my knowledge, my model with 91% accuracy and 97% AUC outperforms the best existing NLP model for AD diagnosis with an accuracy of 88% [32]. I study the model stability through repeated experiments and find that the model is stable even though the training data is split randomly (standard deviation of accuracy = 0.0403; standard deviation of AUC = 0.0174). This affirms our proposed method is accurate and stable. This model can be used as a large-scale screening method for AD, as well as a complementary examination for doctors to detect AD.
>
---
#### [new 083] DPRF: A Generalizable Dynamic Persona Refinement Framework for Optimizing Behavior Alignment Between Personalized LLM Role-Playing Agents and Humans
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出动态角色精炼框架（DPRF），旨在提升大语言模型角色代理与目标个体行为的一致性。针对人工构建角色失真问题，通过迭代分析生成行为与真实行为的认知差异，优化角色设定，增强行为对齐，在多场景下验证了其有效性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.14205v1](http://arxiv.org/pdf/2510.14205v1)**

> **作者:** Bingsheng Yao; Bo Sun; Yuanzhe Dong; Yuxuan Lu; Dakuo Wang
>
> **备注:** In Submission
>
> **摘要:** The emerging large language model role-playing agents (LLM RPAs) aim to simulate individual human behaviors, but the persona fidelity is often undermined by manually-created profiles (e.g., cherry-picked information and personality characteristics) without validating the alignment with the target individuals. To address this limitation, our work introduces the Dynamic Persona Refinement Framework (DPRF).DPRF aims to optimize the alignment of LLM RPAs' behaviors with those of target individuals by iteratively identifying the cognitive divergence, either through free-form or theory-grounded, structured analysis, between generated behaviors and human ground truth, and refining the persona profile to mitigate these divergences.We evaluate DPRF with five LLMs on four diverse behavior-prediction scenarios: formal debates, social media posts with mental health issues, public interviews, and movie reviews.DPRF can consistently improve behavioral alignment considerably over baseline personas and generalizes across models and scenarios.Our work provides a robust methodology for creating high-fidelity persona profiles and enhancing the validity of downstream applications, such as user simulation, social studies, and personalized AI.
>
---
#### [new 084] MoM: Mixtures of Scenario-Aware Document Memories for Retrieval-Augmented Generation Systems
- **分类: cs.CL**

- **简介: 该论文属RAG任务，旨在解决传统文本分块导致的知识理解浅层化问题。提出MoM框架，通过模拟专家阅读生成结构化文档记忆，结合多路径采样与反向推理训练小模型，实现跨域文档的深度内容提取与类人阅读理解。**

- **链接: [http://arxiv.org/pdf/2510.14252v1](http://arxiv.org/pdf/2510.14252v1)**

> **作者:** Jihao Zhao; Zhiyuan Ji; Simin Niu; Hanyu Wang; Feiyu Xiong; Zhiyu Li
>
> **摘要:** The traditional RAG paradigm, which typically engages in the comprehension of relevant text chunks in response to received queries, inherently restricts both the depth of knowledge internalization and reasoning capabilities. To address this limitation, our research transforms the text processing in RAG from passive chunking to proactive understanding, defining this process as document memory extraction with the objective of simulating human cognitive processes during reading. Building upon this, we propose the Mixtures of scenario-aware document Memories (MoM) framework, engineered to efficiently handle documents from multiple domains and train small language models (SLMs) to acquire the ability to proactively explore and construct document memories. The MoM initially instructs large language models (LLMs) to simulate domain experts in generating document logical outlines, thereby directing structured chunking and core content extraction. It employs a multi-path sampling and multi-perspective evaluation mechanism, specifically designing comprehensive metrics that represent chunk clarity and extraction completeness to select the optimal document memories. Additionally, to infuse deeper human-like reading abilities during the training of SLMs, we incorporate a reverse reasoning strategy, which deduces refined expert thinking paths from high-quality outcomes. Finally, leveraging diverse forms of content generated by MoM, we develop a three-layer document memory retrieval mechanism, which is grounded in our theoretical proof from the perspective of probabilistic modeling. Extensive experimental results across three distinct domains demonstrate that the MoM framework not only resolves text chunking challenges in existing RAG systems, providing LLMs with semantically complete document memories, but also paves the way for SLMs to achieve human-centric intelligent text processing.
>
---
#### [new 085] PRISM: Agentic Retrieval with LLMs for Multi-Hop Question Answering
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文研究多跳问答中的检索任务，旨在提升证据检索的精度与召回率。提出PRISM框架，通过三个智能体协同工作，分解问题、精准筛选并补充遗漏证据，有效过滤干扰信息，提高问答性能。**

- **链接: [http://arxiv.org/pdf/2510.14278v1](http://arxiv.org/pdf/2510.14278v1)**

> **作者:** Md Mahadi Hasan Nahid; Davood Rafiei
>
> **备注:** 18 pages
>
> **摘要:** Retrieval plays a central role in multi-hop question answering (QA), where answering complex questions requires gathering multiple pieces of evidence. We introduce an Agentic Retrieval System that leverages large language models (LLMs) in a structured loop to retrieve relevant evidence with high precision and recall. Our framework consists of three specialized agents: a Question Analyzer that decomposes a multi-hop question into sub-questions, a Selector that identifies the most relevant context for each sub-question (focusing on precision), and an Adder that brings in any missing evidence (focusing on recall). The iterative interaction between Selector and Adder yields a compact yet comprehensive set of supporting passages. In particular, it achieves higher retrieval accuracy while filtering out distracting content, enabling downstream QA models to surpass full-context answer accuracy while relying on significantly less irrelevant information. Experiments on four multi-hop QA benchmarks -- HotpotQA, 2WikiMultiHopQA, MuSiQue, and MultiHopRAG -- demonstrates that our approach consistently outperforms strong baselines.
>
---
#### [new 086] Speculative Model Risk in Healthcare AI: Using Storytelling to Surface Unintended Harms
- **分类: cs.CL**

- **简介: 该论文关注医疗AI中的推测性模型风险，旨在通过讲故事的方法揭示潜在危害。作者提出一种以人为中心的框架，生成用户故事并支持多智能体讨论，帮助在部署前识别更广泛的AI风险。实验表明，故事能促进参与者更全面、均衡地思考各类潜在危害。**

- **链接: [http://arxiv.org/pdf/2510.14718v1](http://arxiv.org/pdf/2510.14718v1)**

> **作者:** Xingmeng Zhao; Dan Schumacher; Veronica Rammouz; Anthony Rios
>
> **备注:** 8 pages main + Appendix
>
> **摘要:** Artificial intelligence (AI) is rapidly transforming healthcare, enabling fast development of tools like stress monitors, wellness trackers, and mental health chatbots. However, rapid and low-barrier development can introduce risks of bias, privacy violations, and unequal access, especially when systems ignore real-world contexts and diverse user needs. Many recent methods use AI to detect risks automatically, but this can reduce human engagement in understanding how harms arise and who they affect. We present a human-centered framework that generates user stories and supports multi-agent discussions to help people think creatively about potential benefits and harms before deployment. In a user study, participants who read stories recognized a broader range of harms, distributing their responses more evenly across all 13 harm types. In contrast, those who did not read stories focused primarily on privacy and well-being (58.3%). Our findings show that storytelling helped participants speculate about a broader range of harms and benefits and think more creatively about AI's impact on users.
>
---
#### [new 087] MedTrust-RAG: Evidence Verification and Trust Alignment for Biomedical Question Answering
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文针对生物医学问答中检索增强生成模型易产生幻觉的问题，提出MedTrust-RAG框架，通过引用感知推理、迭代检索验证和信任对齐模块，提升回答的可信性与事实一致性。**

- **链接: [http://arxiv.org/pdf/2510.14400v1](http://arxiv.org/pdf/2510.14400v1)**

> **作者:** Yingpeng Ning; Yuanyuan Sun; Ling Luo; Yanhua Wang; Yuchen Pan; Hongfei Lin
>
> **摘要:** Biomedical question answering (QA) requires accurate interpretation of complex medical knowledge. Large language models (LLMs) have shown promising capabilities in this domain, with retrieval-augmented generation (RAG) systems enhancing performance by incorporating external medical literature. However, RAG-based approaches in biomedical QA suffer from hallucinations due to post-retrieval noise and insufficient verification of retrieved evidence, undermining response reliability. We propose MedTrust-Guided Iterative RAG, a framework designed to enhance factual consistency and mitigate hallucinations in medical QA. Our method introduces three key innovations. First, it enforces citation-aware reasoning by requiring all generated content to be explicitly grounded in retrieved medical documents, with structured Negative Knowledge Assertions used when evidence is insufficient. Second, it employs an iterative retrieval-verification process, where a verification agent assesses evidence adequacy and refines queries through Medical Gap Analysis until reliable information is obtained. Third, it integrates the MedTrust-Align Module (MTAM) that combines verified positive examples with hallucination-aware negative samples, leveraging Direct Preference Optimization to reinforce citation-grounded reasoning while penalizing hallucination-prone response patterns. Experiments on MedMCQA, MedQA, and MMLU-Med demonstrate that our approach consistently outperforms competitive baselines across multiple model architectures, achieving the best average accuracy with gains of 2.7% for LLaMA3.1-8B-Instruct and 2.4% for Qwen3-8B.
>
---
#### [new 088] A Survey on Collaborating Small and Large Language Models for Performance, Cost-effectiveness, Cloud-edge Privacy, and Trustworthiness
- **分类: cs.CL; cs.AI; 68T50 (Primary) 68T07 (Secondary); I.2.7**

- **简介: 该论文综述小模型（SLM）与大模型（LLM）协同方法，旨在提升性能、降低成本、保障云边隐私与可信性。提出四类目标的分类体系，总结现有方法与设计范式，并指出未来挑战。**

- **链接: [http://arxiv.org/pdf/2510.13890v1](http://arxiv.org/pdf/2510.13890v1)**

> **作者:** Fali Wang; Jihai Chen; Shuhua Yang; Ali Al-Lawati; Linli Tang; Hui Liu; Suhang Wang
>
> **备注:** 17 pages, 17 figures, under review
>
> **摘要:** Large language models (LLMs) have advanced many domains and applications but face high fine-tuning costs, inference latency, limited edge deployability, and reliability concerns. Small language models (SLMs), compact, efficient, and adaptable, offer complementary remedies. Recent work explores collaborative frameworks that fuse SLMs' specialization and efficiency with LLMs' generalization and reasoning to meet diverse objectives across tasks and deployment scenarios. Motivated by these developments, this paper presents a systematic survey of SLM-LLM collaboration organized by collaboration objectives. We propose a taxonomy with four goals: performance enhancement, cost-effectiveness, cloud-edge privacy, and trustworthiness. Within this framework, we review representative methods, summarize design paradigms, and outline open challenges and future directions toward efficient, secure, and scalable SLM-LLM collaboration.
>
---
#### [new 089] Predicting Task Performance with Context-aware Scaling Laws
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究大语言模型的下游任务性能预测，提出一个结合训练算力与上下文长度的可解释框架，解决传统缩放定律忽略上下文影响的问题，经实验证明能准确建模并外推长上下文下的性能表现。**

- **链接: [http://arxiv.org/pdf/2510.14919v1](http://arxiv.org/pdf/2510.14919v1)**

> **作者:** Kyle Montgomery; David Park; Jianhong Tu; Michael Bendersky; Beliz Gunel; Dawn Song; Chenguang Wang
>
> **摘要:** Scaling laws have transformed our understanding of large language models by linking upstream metrics like cross-entropy loss to design factors such as model size, training data, and compute. However, these conventional laws fail to capture downstream task performance, where context plays a critical role. In this work, we propose a straightforward, interpretable framework that jointly models downstream performance as a function of the training compute and the provided context. We empirically validate our framework by fitting it on the observed downstream performance of extended-context variants of Llama-2-7B and Llama-2-13B across 65,500 unique instances spanning three tasks: arithmetic reasoning, common sense reasoning, and machine translation. Our results demonstrate that our framework accurately models in-distribution downstream performance, generalizes across three orders of magnitude in training compute, and reliably extrapolates performance as the amount of context increases. These findings offer valuable insights into the interplay between training compute and context utilization, providing guidance for designing more efficient long-context LLMs for diverse downstream tasks. Our code is available at https://github.com/wang-research-lab/context-scaling.
>
---
#### [new 090] Intent Clustering with Shared Pseudo-Labels
- **分类: cs.CL; cs.IR**

- **简介: 该论文研究无监督意图聚类任务，旨在解决依赖商业大模型和预设簇数量的问题。作者提出一种无需训练的方法，利用轻量开源大模型生成伪标签，通过共享伪标签进行多标签分类，实现高效、可解释的意图聚类。**

- **链接: [http://arxiv.org/pdf/2510.14640v1](http://arxiv.org/pdf/2510.14640v1)**

> **作者:** I-Fan Lin; Faegheh Hasibi; Suzan Verberne
>
> **摘要:** In this paper, we propose an intuitive, training-free and label-free method for intent clustering that makes minimal assumptions using lightweight and open-source LLMs. Many current approaches rely on commercial LLMs, which are costly, and offer limited transparency. Additionally, their methods often explicitly depend on knowing the number of clusters in advance, which is often not the case in realistic settings. To address these challenges, instead of asking the LLM to match similar text directly, we first ask it to generate pseudo-labels for each text, and then perform multi-label classification in this pseudo-label set for each text. This approach is based on the hypothesis that texts belonging to the same cluster will share more labels, and will therefore be closer when encoded into embeddings. These pseudo-labels are more human-readable than direct similarity matches. Our evaluation on four benchmark sets shows that our approach achieves results comparable to and better than recent baselines, while remaining simple and computationally efficient. Our findings indicate that our method can be applied in low-resource scenarios and is stable across multiple models and datasets.
>
---
#### [new 091] Users as Annotators: LLM Preference Learning from Comparison Mode
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究LLM对齐中的偏好学习任务，解决用户标注数据质量不可控的问题。提出通过模型响应差异建模用户行为，利用EM算法估计用户质量并过滤低质标注，提升对齐效果。**

- **链接: [http://arxiv.org/pdf/2510.13830v1](http://arxiv.org/pdf/2510.13830v1)**

> **作者:** Zhongze Cai; Xiaocheng Li
>
> **摘要:** Pairwise preference data have played an important role in the alignment of large language models (LLMs). Each sample of such data consists of a prompt, two different responses to the prompt, and a binary label indicating which of the two responses is better. The labels are usually annotated by professional human annotators. In this paper, we consider an alternative approach to collect pairwise preference data -- user annotation from comparison mode. With the increasingly wider adoption of LLMs among the population, users are contributing more and more of their preference labels through their daily interactions with the LLMs. The upside of such labels is that users are the best experts in judging the responses to their own queries/prompts, but the downside is the lack of quality control in these labels. In this paper, we consider a new idea of generating two responses from two different models or two different versions of the same model. The asymmetry allows us to make an inference of the user's data quality through our proposed user behavior model. We develop an expectation-maximization algorithm to estimate a latent quality factor of the user, and filter users' annotation data accordingly. The downstream task shows the effectiveness of our approach in both capturing the user behavior and data filtering for LLM alignment.
>
---
#### [new 092] Knowledge Reasoning Language Model: Unifying Knowledge and Language for Inductive Knowledge Graph Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究归纳式知识图谱推理（KGR），旨在解决开放域中未知实体与关系的推理问题。现有方法存在大模型知识被稀疏图谱信息扭曲及生成幻觉问题。作者提出知识推理语言模型（KRLM），通过设计知识推理语言、动态知识记忆机制和结构感知预测器，实现大模型知识与图谱上下文的协同统一，提升推理准确性与可信度。**

- **链接: [http://arxiv.org/pdf/2510.13909v1](http://arxiv.org/pdf/2510.13909v1)**

> **作者:** Xingrui Zhuo; Jiapu Wang; Gongqing Wu; Zhongyuan Wang; Jichen Zhang; Shirui Pan; Xindong Wu
>
> **摘要:** Inductive Knowledge Graph Reasoning (KGR) aims to discover facts in open-domain KGs containing unknown entities and relations, which poses a challenge for KGR models in comprehending uncertain KG components. Existing studies have proposed Knowledge Graph Foundation Models (KGFMs) that learn structural invariances across KGs to handle this uncertainty. Recently, Large Language Models (LLMs) have demonstrated strong capabilities for open-domain knowledge reasoning. As a result, the latest research has focused on LLM-based KGFMs that integrate LLM knowledge with KG context for inductive KGR. However, the intrinsic knowledge of LLMs may be overshadowed by sparse KG context, leading to LLM knowledge distortion, which can cause irreversible damage to model reasoning. Moreover, existing LLM-based KGR methods still struggle to fully constrain generative hallucinations in LLMs, severely limiting the credibility of reasoning results. To address these limitations, we propose a Knowledge Reasoning Language Model (KRLM) that achieves unified coordination between LLM knowledge and KG context throughout the KGR process. Specifically, we design a Knowledge Reasoning Language (KRL) instruction format and a KRL tokenizer to align LLM knowledge with KG representations. Then, we propose a KRL attention layer that coordinates intrinsic LLM knowledge with additional KG context through a dynamic knowledge memory mechanism. Finally, a structure-aware next-entity predictor is proposed, which strictly constrains the reasoning results within a trustworthy knowledge domain. Extensive experimental results on 25 real-world inductive KGR datasets demonstrate the significant superiority of the proposed KRLM\footnote{Our source codes are available at https://anonymous.4open.science/r/KRLM-EA36 in both zero-shot reasoning and fine-tuning scenarios.
>
---
#### [new 093] DialectGen: Benchmarking and Improving Dialect Robustness in Multimodal Generation
- **分类: cs.CL; cs.CV; cs.LG**

- **简介: 该论文研究多模态生成模型在方言文本输入下的性能问题，构建涵盖六种英语方言的大规模基准DialectGen。发现现有模型在方言输入时性能显著下降，提出一种基于编码器的缓解策略，在提升方言生成效果的同时保持标准英语性能。**

- **链接: [http://arxiv.org/pdf/2510.14949v1](http://arxiv.org/pdf/2510.14949v1)**

> **作者:** Yu Zhou; Sohyun An; Haikang Deng; Da Yin; Clark Peng; Cho-Jui Hsieh; Kai-Wei Chang; Nanyun Peng
>
> **摘要:** Contact languages like English exhibit rich regional variations in the form of dialects, which are often used by dialect speakers interacting with generative models. However, can multimodal generative models effectively produce content given dialectal textual input? In this work, we study this question by constructing a new large-scale benchmark spanning six common English dialects. We work with dialect speakers to collect and verify over 4200 unique prompts and evaluate on 17 image and video generative models. Our automatic and human evaluation results show that current state-of-the-art multimodal generative models exhibit 32.26% to 48.17% performance degradation when a single dialect word is used in the prompt. Common mitigation methods such as fine-tuning and prompt rewriting can only improve dialect performance by small margins (< 7%), while potentially incurring significant performance degradation in Standard American English (SAE). To this end, we design a general encoder-based mitigation strategy for multimodal generative models. Our method teaches the model to recognize new dialect features while preserving SAE performance. Experiments on models such as Stable Diffusion 1.5 show that our method is able to simultaneously raise performance on five dialects to be on par with SAE (+34.4%), while incurring near zero cost to SAE performance.
>
---
#### [new 094] ERGO: Entropy-guided Resetting for Generation Optimization in Multi-turn Language Models
- **分类: cs.CL**

- **简介: 该论文针对多轮对话中大模型性能下降问题，提出ERGO方法，通过熵检测模型不确定性并动态重置提示，提升生成准确性与稳定性，显著改善多轮交互中的性能与可靠性。**

- **链接: [http://arxiv.org/pdf/2510.14077v1](http://arxiv.org/pdf/2510.14077v1)**

> **作者:** Haziq Mohammad Khalid; Athikash Jeyaganthan; Timothy Do; Yicheng Fu; Sean O'Brien; Vasu Sharma; Kevin Zhu
>
> **备注:** 14 pages, 5 figures
>
> **摘要:** Large Language Models (LLMs) suffer significant performance degradation in multi-turn conversations when information is presented incrementally. Given that multi-turn conversations characterize everyday interactions with LLMs, this degradation poses a severe challenge to real world usability. We hypothesize that abrupt increases in model uncertainty signal misalignment in multi-turn LLM interactions, and we exploit this insight to dynamically realign conversational context. We introduce ERGO (Entropy-guided Resetting for Generation Optimization), which continuously quantifies internal uncertainty via Shannon entropy over next token distributions and triggers adaptive prompt consolidation when a sharp spike in entropy is detected. By treating uncertainty as a first class signal rather than a nuisance to eliminate, ERGO embraces variability in language and modeling, representing and responding to uncertainty. In multi-turn tasks with incrementally revealed instructions, ERGO yields a 56.6% average performance gain over standard baselines, increases aptitude (peak performance capability) by 24.7%, and decreases unreliability (variability in performance) by 35.3%, demonstrating that uncertainty aware interventions can improve both accuracy and reliability in conversational AI.
>
---
#### [new 095] Efficient Seq2seq Coreference Resolution Using Entity Representations
- **分类: cs.CL**

- **简介: 该论文研究对话等增量场景下的共指消解任务，针对序列到序列（seq2seq）模型效率低的问题，提出一种基于实体表示的压缩方法，通过保留实体级词元、丢弃冗余输入提升效率，在保持接近最优性能的同时显著提高处理速度。**

- **链接: [http://arxiv.org/pdf/2510.14504v1](http://arxiv.org/pdf/2510.14504v1)**

> **作者:** Matt Grenander; Shay B. Cohen; Mark Steedman
>
> **摘要:** Seq2seq coreference models have introduced a new paradigm for coreference resolution by learning to generate text corresponding to coreference labels, without requiring task-specific parameters. While these models achieve new state-of-the-art performance, they do so at the cost of flexibility and efficiency. In particular, they do not efficiently handle incremental settings such as dialogue, where text must processed sequentially. We propose a compressed representation in order to improve the efficiency of these methods in incremental settings. Our method works by extracting and re-organizing entity-level tokens, and discarding the majority of other input tokens. On OntoNotes, our best model achieves just 0.6 CoNLL F1 points below a full-prefix, incremental baseline while achieving a compression ratio of 1.8. On LitBank, where singleton mentions are annotated, it passes state-of-the-art performance. Our results indicate that discarding a wide portion of tokens in seq2seq resolvers is a feasible strategy for incremental coreference resolution.
>
---
#### [new 096] From Explainability to Action: A Generative Operational Framework for Integrating XAI in Clinical Mental Health Screening
- **分类: cs.CL**

- **简介: 该论文提出生成式操作框架，解决可解释AI在精神健康筛查中难以转化为临床可用信息的问题。利用大语言模型整合XAI输出与临床指南，生成可读、可信的临床叙事，推动AI从技术透明到实际应用的转化。**

- **链接: [http://arxiv.org/pdf/2510.13828v1](http://arxiv.org/pdf/2510.13828v1)**

> **作者:** Ratna Kandala; Akshata Kishore Moharir; Divya Arvinda Nayak
>
> **摘要:** Explainable Artificial Intelligence (XAI) has been presented as the critical component for unlocking the potential of machine learning in mental health screening (MHS). However, a persistent lab-to-clinic gap remains. Current XAI techniques, such as SHAP and LIME, excel at producing technically faithful outputs such as feature importance scores, but fail to deliver clinically relevant, actionable insights that can be used by clinicians or understood by patients. This disconnect between technical transparency and human utility is the primary barrier to real-world adoption. This paper argues that this gap is a translation problem and proposes the Generative Operational Framework, a novel system architecture that leverages Large Language Models (LLMs) as a central translation engine. This framework is designed to ingest the raw, technical outputs from diverse XAI tools and synthesize them with clinical guidelines (via RAG) to automatically generate human-readable, evidence-backed clinical narratives. To justify our solution, we provide a systematic analysis of the components it integrates, tracing the evolution from intrinsic models to generative XAI. We demonstrate how this framework directly addresses key operational barriers, including workflow integration, bias mitigation, and stakeholder-specific communication. This paper also provides a strategic roadmap for moving the field beyond the generation of isolated data points toward the delivery of integrated, actionable, and trustworthy AI in clinical practice.
>
---
#### [new 097] Building a Macedonian Recipe Dataset: Collection, Parsing, and Comparative Analysis
- **分类: cs.CL**

- **简介: 该论文致力于构建首个马其顿语食谱数据集，属于数据构建与烹饪文化分析任务。针对马其顿食谱数字化缺失问题，作者通过网络爬取和结构化解析，解决成分描述异质性难题，并进行成分频率与共现模式的探索性分析，揭示马其顿饮食的独特组合特征。**

- **链接: [http://arxiv.org/pdf/2510.14128v1](http://arxiv.org/pdf/2510.14128v1)**

> **作者:** Darko Sasanski; Dimitar Peshevski; Riste Stojanov; Dimitar Trajanov
>
> **摘要:** Computational gastronomy increasingly relies on diverse, high-quality recipe datasets to capture regional culinary traditions. Although there are large-scale collections for major languages, Macedonian recipes remain under-represented in digital research. In this work, we present the first systematic effort to construct a Macedonian recipe dataset through web scraping and structured parsing. We address challenges in processing heterogeneous ingredient descriptions, including unit, quantity, and descriptor normalization. An exploratory analysis of ingredient frequency and co-occurrence patterns, using measures such as Pointwise Mutual Information and Lift score, highlights distinctive ingredient combinations that characterize Macedonian cuisine. The resulting dataset contributes a new resource for studying food culture in underrepresented languages and offers insights into the unique patterns of Macedonian culinary tradition.
>
---
#### [new 098] Rewriting History: A Recipe for Interventional Analyses to Study Data Effects on Model Behavior
- **分类: cs.CL**

- **简介: 该论文提出一种干预分析方法，研究训练数据对语言模型行为的影响。通过“重写历史”式的数据修改与重训练，验证数据与模型行为的关系，旨在探究数据如何影响模型知识获取，弥补观测性分析的局限。**

- **链接: [http://arxiv.org/pdf/2510.14261v1](http://arxiv.org/pdf/2510.14261v1)**

> **作者:** Rahul Nadkarni; Yanai Elazar; Hila Gonen; Noah A. Smith
>
> **摘要:** We present an experimental recipe for studying the relationship between training data and language model (LM) behavior. We outline steps for intervening on data batches -- i.e., ``rewriting history'' -- and then retraining model checkpoints over that data to test hypotheses relating data to behavior. Our recipe breaks down such an intervention into stages that include selecting evaluation items from a benchmark that measures model behavior, matching relevant documents to those items, and modifying those documents before retraining and measuring the effects. We demonstrate the utility of our recipe through case studies on factual knowledge acquisition in LMs, using both cooccurrence statistics and information retrieval methods to identify documents that might contribute to knowledge learning. Our results supplement past observational analyses that link cooccurrence to model behavior, while demonstrating that extant methods for identifying relevant training documents do not fully explain an LM's ability to correctly answer knowledge questions. Overall, we outline a recipe that researchers can follow to test further hypotheses about how training data affects model behavior. Our code is made publicly available to promote future work.
>
---
#### [new 099] Retrofitting Small Multilingual Models for Retrieval: Matching 7B Performance with 300M Parameters
- **分类: cs.CL**

- **简介: 该论文聚焦多语言检索任务，旨在解决小模型（<1B参数）在检索性能上落后于大模型的问题。通过优化训练数据规模、负采样策略和任务多样性，提出一种300M参数的小模型，性能媲美甚至超越7B大模型。**

- **链接: [http://arxiv.org/pdf/2510.14274v1](http://arxiv.org/pdf/2510.14274v1)**

> **作者:** Lifu Tu; Yingbo Zhou; Semih Yavuz
>
> **摘要:** Training effective multilingual embedding models presents unique challenges due to the diversity of languages and task objectives. Although small multilingual models (<1 B parameters) perform well on multilingual tasks generally, they consistently lag behind larger models (>1 B) in the most prevalent use case: retrieval. This raises a critical question: Can smaller models be retrofitted specifically for retrieval tasks to enhance their performance? In this work, we investigate key factors that influence the effectiveness of multilingual embeddings, focusing on training data scale, negative sampling strategies, and data diversity. We find that while increasing the scale of training data yields initial performance gains, these improvements quickly plateau - indicating diminishing returns. Incorporating hard negatives proves essential for consistently improving retrieval accuracy. Furthermore, our analysis reveals that task diversity in the training data contributes more significantly to performance than language diversity alone. As a result, we develop a compact (approximately 300M) multilingual model that achieves retrieval performance comparable to or even surpassing current strong 7B models.
>
---
#### [new 100] Optimal Aggregation of LLM and PRM Signals for Efficient Test-Time Scaling
- **分类: cs.CL**

- **简介: 该论文研究测试时扩展（TTS）中大语言模型（LLM）与过程奖励模型（PRM）信号的融合问题，旨在提升响应选择效率。提出理论框架并设计加权聚合方法，通过校准权重显著提高性能，仅用21.3%计算量即超越传统投票法。**

- **链接: [http://arxiv.org/pdf/2510.13918v1](http://arxiv.org/pdf/2510.13918v1)**

> **作者:** Peng Kuang; Yanli Wang; Xiaoyu Han; Yaowenqi Liu; Kaidi Xu; Haohan Wang
>
> **摘要:** Process reward models (PRMs) are a cornerstone of test-time scaling (TTS), designed to verify and select the best responses from large language models (LLMs). However, this promise is challenged by recent benchmarks where simple majority voting, which ignores PRM signals, occasionally outperforms standard PRM-based selection. This raises a critical question: How can we effectively utilize verification signals from PRMs for TTS? To address this, we start by developing a theoretical framework for optimally combining signals from both the LLM and the PRM. Our framework reveals that the optimal strategy is a weighted aggregation of responses, a strategy whose effectiveness hinges on estimating weights that capture the complex interplay between the models. Based on our theoretical results, we empirically show that these optimal weighting functions differ significantly across LLM-PRM pairs and, notably, often assign substantial negative weights. Motivated by these insights, we propose efficient pre-computation methods to calibrate these weighting functions. Extensive experiments across 5 LLMs and 7 PRMs demonstrate that our calibration method significantly boosts the TTS efficiency, surpassing the performance of vanilla weighted majority voting while using only $21.3\%$ of the computation. Ultimately, our work demonstrates that investing in a more intelligent aggregation strategy can be a more convincing path to performance gains than simply scaling test-time computation.
>
---
#### [new 101] From Binary to Bilingual: How the National Weather Service is Using Artificial Intelligence to Develop a Comprehensive Translation Program
- **分类: cs.CL; cs.AI; cs.CY; cs.HC**

- **简介: 该论文属自然语言处理与公共服务交叉任务，旨在解决非英语人群获取气象信息的障碍。NWS联合LILT开发基于AI的多语言翻译系统，利用大模型和GIS分析，实现天气预警等信息的高效、准确、文化适配的自动翻译，覆盖西班牙语、中文等语言，并融入伦理AI原则。**

- **链接: [http://arxiv.org/pdf/2510.14369v1](http://arxiv.org/pdf/2510.14369v1)**

> **作者:** Joseph E. Trujillo-Falcon; Monica L. Bozeman; Liam E. Llewellyn; Samuel T. Halvorson; Meryl Mizell; Stuti Deshpande; Bob Manning; Todd Fagin
>
> **摘要:** To advance a Weather-Ready Nation, the National Weather Service (NWS) is developing a systematic translation program to better serve the 68.8 million people in the U.S. who do not speak English at home. This article outlines the foundation of an automated translation tool for NWS products, powered by artificial intelligence. The NWS has partnered with LILT, whose patented training process enables large language models (LLMs) to adapt neural machine translation (NMT) tools for weather terminology and messaging. Designed for scalability across Weather Forecast Offices (WFOs) and National Centers, the system is currently being developed in Spanish, Simplified Chinese, Vietnamese, and other widely spoken non-English languages. Rooted in best practices for multilingual risk communication, the system provides accurate, timely, and culturally relevant translations, significantly reducing manual translation time and easing operational workloads across the NWS. To guide the distribution of these products, GIS mapping was used to identify language needs across different NWS regions, helping prioritize resources for the communities that need them most. We also integrated ethical AI practices throughout the program's design, ensuring that transparency, fairness, and human oversight guide how automated translations are created, evaluated, and shared with the public. This work has culminated into a website featuring experimental multilingual NWS products, including translated warnings, 7-day forecasts, and educational campaigns, bringing the country one step closer to a national warning system that reaches all Americans.
>
---
#### [new 102] ShishuLM: Lightweight Language Model with Hybrid Decoder-MLP Architecture and Paired Weight Sharing
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大模型内存和计算开销大的问题，提出轻量级语言模型ShishuLM。通过混合解码器-MLP架构与权重共享，减少参数量和KV缓存，显著降低内存与延迟，适用于小语言模型在代理AI中的高效部署。**

- **链接: [http://arxiv.org/pdf/2510.13860v1](http://arxiv.org/pdf/2510.13860v1)**

> **作者:** Shivanshu Kumar; Gopalakrishnan Srinivasan
>
> **摘要:** While the transformer architecture has achieved state-of-the-art performance on natural language processing tasks, these models impose substantial memory and computational overhead. Recent research has identified significant architectural redundancies within these models, presenting opportunities for optimization without compromising performance. Taking insights from research in AI interpretability and inference-time layer pruning, we introduce an efficient language model architecture, referred to as ShishuLM, which reduces both the parameter count and Key-Value (KV) cache requirements. Given the increasing importance of Small Language Models (SLMs) in agentic AI systems, we evaluate our approach on two SLMs of different scales. Our analysis reveals that for moderate-context scenarios, normalization coupled with attention computation is roughly linear with the input, enabling entire transformer blocks to be approximated through Multi-Layer Perceptrons (MLPs). Our results show that ShishuLM provides up to 25% reduction in memory requirements and up to 40% improvement in latency during both training and inference, compared to parent models. Our experimental and analytical findings provide insights towards building more efficient SLM architectures from a pre-training standpoint.
>
---
#### [new 103] Seeing Hate Differently: Hate Subspace Modeling for Culture-Aware Hate Speech Detection
- **分类: cs.CL; cs.AI; cs.SI; 68T50, 68T45; I.2.7; I.2.6; K.4.1**

- **简介: 该论文属仇恨言论检测任务，旨在解决文化差异导致的标注偏差与数据稀疏问题。提出文化感知框架，通过建模文化属性组合与标签传播，构建个体化的仇恨子空间以提升检测性能。**

- **链接: [http://arxiv.org/pdf/2510.13837v1](http://arxiv.org/pdf/2510.13837v1)**

> **作者:** Weibin Cai; Reza Zafarani
>
> **摘要:** Hate speech detection has been extensively studied, yet existing methods often overlook a real-world complexity: training labels are biased, and interpretations of what is considered hate vary across individuals with different cultural backgrounds. We first analyze these challenges, including data sparsity, cultural entanglement, and ambiguous labeling. To address them, we propose a culture-aware framework that constructs individuals' hate subspaces. To alleviate data sparsity, we model combinations of cultural attributes. For cultural entanglement and ambiguous labels, we use label propagation to capture distinctive features of each combination. Finally, individual hate subspaces, which in turn can further enhance classification performance. Experiments show our method outperforms state-of-the-art by 1.05\% on average across all metrics.
>
---
#### [new 104] Reliable Fine-Grained Evaluation of Natural Language Math Proofs
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦自然语言数学证明的细粒度评估任务，旨在解决缺乏可靠自动评估方法的问题。作者构建了专家标注数据集ProofBench，提出并验证了评估框架ProofGrader，实现接近人类评分的自动化打分，显著优于基线方法。**

- **链接: [http://arxiv.org/pdf/2510.13888v1](http://arxiv.org/pdf/2510.13888v1)**

> **作者:** Wenjie Ma; Andrei Cojocaru; Neel Kolhe; Bradley Louie; Robin Said Sharif; Haihan Zhang; Vincent Zhuang; Matei Zaharia; Sewon Min
>
> **备注:** 31 pages, 6 figures, 10 tables
>
> **摘要:** Recent advances in large language models (LLMs) for mathematical reasoning have largely focused on tasks with easily verifiable final answers; however, generating and verifying natural language math proofs remains an open challenge. We identify the absence of a reliable, fine-grained evaluator for LLM-generated math proofs as a critical gap. To address this, we propose a systematic methodology for developing and validating evaluators that assign fine-grained scores on a 0-7 scale to model-generated math proofs. To enable this study, we introduce ProofBench, the first expert-annotated dataset of fine-grained proof ratings, spanning 145 problems from six major math competitions (USAMO, IMO, Putnam, etc) and 435 LLM-generated solutions from Gemini-2.5-pro, o3, and DeepSeek-R1. %with expert gradings. Using ProofBench as a testbed, we systematically explore the evaluator design space across key axes: the backbone model, input context, instructions and evaluation workflow. Our analysis delivers ProofGrader, an evaluator that combines a strong reasoning backbone LM, rich context from reference solutions and marking schemes, and a simple ensembling method; it achieves a low Mean Absolute Error (MAE) of 0.926 against expert scores, significantly outperforming naive baselines. Finally, we demonstrate its practical utility in a best-of-$n$ selection task: at $n=16$, ProofGrader achieves an average score of 4.14 (out of 7), closing 78% of the gap between a naive binary evaluator (2.48) and the human oracle (4.62), highlighting its potential to advance downstream proof generation.
>
---
#### [new 105] Unlocking the Potential of Diffusion Language Models through Template Infilling
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究扩散语言模型的生成任务，旨在解决其沿用自回归前缀提示导致推理策略受限的问题。提出模板填充方法及动态片段分配机制，先生成结构模板再填充内容，提升生成灵活性，在数学推理与代码生成中显著提效。**

- **链接: [http://arxiv.org/pdf/2510.13870v1](http://arxiv.org/pdf/2510.13870v1)**

> **作者:** Junhoo Lee; Seungyeon Kim; Nojun Kwak
>
> **摘要:** Diffusion Language Models (DLMs) have emerged as a promising alternative to Autoregressive Language Models, yet their inference strategies remain limited to prefix-based prompting inherited from the autoregressive paradigm. In this paper, we propose Template Infilling (TI), a tailored conditioning methodology for DLMs' generation process. Unlike conventional prefix prompting, TI first generates a structural template for the target response, then fills in the masked segments. To enhance the flexibility of this structural control, we introduce Dynamic Segment Allocation (DSA), which adaptively adjusts segment lengths based on generation confidence. We demonstrate the effectiveness of our approach on mathematical reasoning and code generation benchmarks, achieving consistent improvements of 17.01$\%$p over baseline. Furthermore, we show that TI provides additional advantages in multi-token generation settings, enabling effective speedup while maintaining generation quality.
>
---
#### [new 106] LLMs Can Get "Brain Rot"!
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出“LLM脑腐”假说，探究低质网络文本对大模型认知能力的持续损害。通过控制实验，验证数据质量下降会导致模型推理、安全等能力衰退，且难以完全恢复，强调数据净化是训练安全的关键。**

- **链接: [http://arxiv.org/pdf/2510.13928v1](http://arxiv.org/pdf/2510.13928v1)**

> **作者:** Shuo Xing; Junyuan Hong; Yifan Wang; Runjin Chen; Zhenyu Zhang; Ananth Grama; Zhengzhong Tu; Zhangyang Wang
>
> **摘要:** We propose and test the LLM Brain Rot Hypothesis: continual exposure to junk web text induces lasting cognitive decline in large language models (LLMs). To causally isolate data quality, we run controlled experiments on real Twitter/X corpora, constructing junk and reversely controlled datasets via two orthogonal operationalizations: M1 (engagement degree) and M2 (semantic quality), with matched token scale and training operations across conditions. Contrary to the control group, continual pre-training of 4 LLMs on the junk dataset causes non-trivial declines (Hedges' $g>0.3$) on reasoning, long-context understanding, safety, and inflating "dark traits" (e.g., psychopathy, narcissism). The gradual mixtures of junk and control datasets also yield dose-response cognition decay: for example, under M1, ARC-Challenge with Chain Of Thoughts drops $74.9 \rightarrow 57.2$ and RULER-CWE $84.4 \rightarrow 52.3$ as junk ratio rises from $0\%$ to $100\%$. Error forensics reveal several key insights. First, we identify thought-skipping as the primary lesion: models increasingly truncate or skip reasoning chains, explaining most of the error growth. Second, partial but incomplete healing is observed: scaling instruction tuning and clean data pre-training improve the declined cognition yet cannot restore baseline capability, suggesting persistent representational drift rather than format mismatch. Finally, we discover that the popularity, a non-semantic metric, of a tweet is a better indicator of the Brain Rot effect than the length in M1. Together, the results provide significant, multi-perspective evidence that data quality is a causal driver of LLM capability decay, reframing curation for continual pretraining as a \textit{training-time safety} problem and motivating routine "cognitive health checks" for deployed LLMs.
>
---
#### [new 107] On the Ability of LLMs to Handle Character-Level Perturbations: How Well and How?
- **分类: cs.CL**

- **简介: 该论文研究大语言模型对字符级扰动的鲁棒性，提出插入隐形Unicode字符的抗滥用方法。通过评估不同设置下模型的表现，探讨其在严重噪声干扰下的响应机制，揭示LLM在低层级输入扰动下的稳定性和潜在风险。**

- **链接: [http://arxiv.org/pdf/2510.14365v1](http://arxiv.org/pdf/2510.14365v1)**

> **作者:** Anyun Zhuo; Xuefei Ning; Ningyuan Li; Yu Wang; Pinyan Lu
>
> **摘要:** This work investigates the resilience of contemporary LLMs against frequent and structured character-level perturbations, specifically through the insertion of noisy characters after each input character. We introduce \nameshort{}, a practical method that inserts invisible Unicode control characters into text to discourage LLM misuse in scenarios such as online exam systems. Surprisingly, despite strong obfuscation that fragments tokenization and reduces the signal-to-noise ratio significantly, many LLMs still maintain notable performance. Through comprehensive evaluation across model-, problem-, and noise-related configurations, we examine the extent and mechanisms of this robustness, exploring both the handling of character-level tokenization and \textit{implicit} versus \textit{explicit} denoising mechanism hypotheses of character-level noises. We hope our findings on the low-level robustness of LLMs will shed light on the risks of their misuse and on the reliability of deploying LLMs across diverse applications.
>
---
#### [new 108] LaSeR: Reinforcement Learning with Last-Token Self-Rewarding
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究大语言模型的推理增强，旨在解决测试时验证信号缺失与效率低的问题。提出LaSeR方法，利用末token自奖励机制，通过一步额外推理高效统一推理与自验证，提升训练与推理性能。**

- **链接: [http://arxiv.org/pdf/2510.14943v1](http://arxiv.org/pdf/2510.14943v1)**

> **作者:** Wenkai Yang; Weijie Liu; Ruobing Xie; Yiju Guo; Lulu Wu; Saiyong Yang; Yankai Lin
>
> **备注:** Work in progress. Github repo: https://github.com/RUCBM/LaSeR
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has recently emerged as a core paradigm for enhancing the reasoning capabilities of Large Language Models (LLMs). To address the lack of verification signals at test time, prior studies incorporate the training of model's self-verification capability into the standard RLVR process, thereby unifying reasoning and verification capabilities within a single LLM. However, previous practice requires the LLM to sequentially generate solutions and self-verifications using two separate prompt templates, which significantly reduces efficiency. In this work, we theoretically reveal that the closed-form solution to the RL objective of self-verification can be reduced to a remarkably simple form: the true reasoning reward of a solution is equal to its last-token self-rewarding score, which is computed as the difference between the policy model's next-token log-probability assigned to any pre-specified token at the solution's last token and a pre-calculated constant, scaled by the KL coefficient. Based on this insight, we propose LaSeR (Reinforcement Learning with Last-Token Self-Rewarding), an algorithm that simply augments the original RLVR loss with a MSE loss that aligns the last-token self-rewarding scores with verifier-based reasoning rewards, jointly optimizing the reasoning and self-rewarding capabilities of LLMs. The optimized self-rewarding scores can be utilized in both training and testing to enhance model performance. Notably, our algorithm derives these scores from the predicted next-token probability distribution of the last token immediately after generation, incurring only the minimal extra cost of one additional token inference. Experiments show that our method not only improves the model's reasoning performance but also equips it with remarkable self-rewarding capability, thereby boosting its inference-time scaling performance.
>
---
#### [new 109] Less is More: Denoising Knowledge Graphs For Retrieval Augmented Generation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对LLM生成的知识图谱在检索增强生成中存在噪声的问题，提出DEG-RAG框架，通过实体消解和三元组反思进行去噪，提升图谱质量与问答性能，首次系统研究了LLM生成知识图谱的实体消解方法。**

- **链接: [http://arxiv.org/pdf/2510.14271v1](http://arxiv.org/pdf/2510.14271v1)**

> **作者:** Yilun Zheng; Dan Yang; Jie Li; Lin Shang; Lihui Chen; Jiahao Xu; Sitao Luan
>
> **摘要:** Retrieval-Augmented Generation (RAG) systems enable large language models (LLMs) instant access to relevant information for the generative process, demonstrating their superior performance in addressing common LLM challenges such as hallucination, factual inaccuracy, and the knowledge cutoff. Graph-based RAG further extends this paradigm by incorporating knowledge graphs (KGs) to leverage rich, structured connections for more precise and inferential responses. A critical challenge, however, is that most Graph-based RAG systems rely on LLMs for automated KG construction, often yielding noisy KGs with redundant entities and unreliable relationships. This noise degrades retrieval and generation performance while also increasing computational cost. Crucially, current research does not comprehensively address the denoising problem for LLM-generated KGs. In this paper, we introduce DEnoised knowledge Graphs for Retrieval Augmented Generation (DEG-RAG), a framework that addresses these challenges through: (1) entity resolution, which eliminates redundant entities, and (2) triple reflection, which removes erroneous relations. Together, these techniques yield more compact, higher-quality KGs that significantly outperform their unprocessed counterparts. Beyond the methods, we conduct a systematic evaluation of entity resolution for LLM-generated KGs, examining different blocking strategies, embedding choices, similarity metrics, and entity merging techniques. To the best of our knowledge, this is the first comprehensive exploration of entity resolution in LLM-generated KGs. Our experiments demonstrate that this straightforward approach not only drastically reduces graph size but also consistently improves question answering performance across diverse popular Graph-based RAG variants.
>
---
#### [new 110] Synthesizing Agentic Data for Web Agents with Progressive Difficulty Enhancement Mechanisms
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对网页代理在复杂问答任务中长时序推理能力不足的问题，提出一种渐进式难度增强的数据合成方法，通过控制任务复杂度生成高质量、多样化的训练数据，并在固定训练框架下验证其有效性，显著提升模型性能。**

- **链接: [http://arxiv.org/pdf/2510.13913v1](http://arxiv.org/pdf/2510.13913v1)**

> **作者:** Shrey Pandit; Xuan-Phi Nguyen; Yifei Ming; Austin Xu; Jiayu Wang; Caiming Xiong; Shafiq Joty
>
> **备注:** Preprint. ICLR 26 submission
>
> **摘要:** Web-based 'deep research' agents aim to solve complex question - answering tasks through long-horizon interactions with online tools. These tasks remain challenging, as the underlying language models are often not optimized for long-horizon reasoning and exploration. Prior work has proposed workflows for constructing instruction-tuning datasets, often leveraging knowledge graphs. However, such methods typically lack fine-grained control over difficulty and quality, yielding synthetic data that falls short of capturing the complexity required for long-horizon reasoning. Furthermore, many studies conflate data and training effects by comparing models trained under different optimization recipes, making it difficult to isolate and evaluate the effectiveness of the data itself. We introduce a two-pronged data synthesis pipeline that generates question - answer pairs by progressively increasing task complexity until a frontier baseline web agent fails. The baseline agent plays multiple roles in this process: attempting the questions, validating factuality, checking for alternative answers, and enforcing filtering. To evaluate the effectiveness of our synthesis methods, we adopt a controlled training setup based on distillation from strong web agents. Experiments across multiple web-based benchmarks show that our dataset - despite being smaller - enables the training of more effective web agents than existing datasets. In particular, our data exhibits twice the diversity in tool-use actions, allowing models trained on it to achieve stronger performance while avoiding repetitive tool-calling behaviors.
>
---
#### [new 111] Readability $\ne$ Learnability: Rethinking the Role of Simplicity in Training Small Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究小语言模型训练中“可读性”与“可学习性”的关系，挑战了简化文本更易学习的观点。通过构建控制变量的合成数据，发现统计简单性（如n-gram多样性）比可读性更能预测模型学习效率和生成连贯性。**

- **链接: [http://arxiv.org/pdf/2510.13915v1](http://arxiv.org/pdf/2510.13915v1)**

> **作者:** Ivan Lee; Taylor Berg-Kirkpatrick
>
> **备注:** Accepted to COLM 2025 (Spotlight)
>
> **摘要:** Recent studies suggest that very small language models (SLMs) can generate surprisingly coherent text when trained on simplified, child-directed corpora such as TinyStories. These findings have been interpreted as evidence that readability -- characterized by accessible vocabulary, familiar narrative structure, and simple syntax -- plays a key role in enabling such capabilities to emerge. In this paper, we challenge that interpretation. We construct synthetic datasets with matched structure but varied readability, and find that readability alone does not predict coherence or learning efficiency in SLMs. Models trained on complex, adult-level text perform comparably to those trained on simplified language, and even exhibit faster development of coherence during training. Instead, we show that statistical simplicity, as measured by n-gram diversity, is a stronger predictor of learnability. Our findings caution against the growing trend of anthropomorphizing language model training -- drawing parallels to human cognitive development without empirical basis -- and argue for more precise reasoning about what properties actually support capability emergence in small models.
>
---
#### [new 112] Evaluating & Reducing Deceptive Dialogue From Language Models with Multi-turn RL
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究大语言模型在多轮对话中的欺骗行为，提出新指标“信念错位”量化欺骗，并发现现有模型存在显著欺骗倾向。作者提出多轮强化学习方法，有效减少77.6%的欺骗行为，提升对话安全性。**

- **链接: [http://arxiv.org/pdf/2510.14318v1](http://arxiv.org/pdf/2510.14318v1)**

> **作者:** Marwa Abdulhai; Ryan Cheng; Aryansh Shrivastava; Natasha Jaques; Yarin Gal; Sergey Levine
>
> **摘要:** Large Language Models (LLMs) interact with millions of people worldwide in applications such as customer support, education and healthcare. However, their ability to produce deceptive outputs, whether intentionally or inadvertently, poses significant safety concerns. The unpredictable nature of LLM behavior, combined with insufficient safeguards against hallucination, misinformation, and user manipulation, makes their misuse a serious, real-world risk. In this paper, we investigate the extent to which LLMs engage in deception within dialogue, and propose the belief misalignment metric to quantify deception. We evaluate deception across four distinct dialogue scenarios, using five established deception detection metrics and our proposed metric. Our findings reveal this novel deception measure correlates more closely with human judgments than any existing metrics we test. Additionally, our benchmarking of eight state-of-the-art models indicates that LLMs naturally exhibit deceptive behavior in approximately 26% of dialogue turns, even when prompted with seemingly benign objectives. When prompted to deceive, LLMs are capable of increasing deceptiveness by as much as 31% relative to baselines. Unexpectedly, models trained with RLHF, the predominant approach for ensuring the safety of widely-deployed LLMs, still exhibit deception at a rate of 43% on average. Given that deception in dialogue is a behavior that develops over an interaction history, its effective evaluation and mitigation necessitates moving beyond single-utterance analyses. We introduce a multi-turn reinforcement learning methodology to fine-tune LLMs to reduce deceptive behaviors, leading to a 77.6% reduction compared to other instruction-tuned models.
>
---
#### [new 113] TextBandit: Evaluating Probabilistic Reasoning in LLMs Through Language-Only Decision Tasks
- **分类: cs.CL**

- **简介: 该论文提出TextBandit任务，评估大语言模型仅通过文本反馈进行概率推理与决策的能力。模型需在无数字提示下从语言中推断奖励结构并选择最优策略。实验显示Qwen3-4B表现最佳，表明语言可支持概率推理。**

- **链接: [http://arxiv.org/pdf/2510.13878v1](http://arxiv.org/pdf/2510.13878v1)**

> **作者:** Jimin Lim; Arjun Damerla; Arthur Jiang; Nam Le
>
> **备注:** COLM 2025 @ ORIGen Workshop
>
> **摘要:** Large language models (LLMs) have shown to be increasingly capable of performing reasoning tasks, but their ability to make sequential decisions under uncertainty only using natural language remains underexplored. We introduce a novel benchmark in which LLMs interact with multi-armed bandit environments using purely textual feedback, "you earned a token", without access to numerical cues or explicit probabilities, resulting in the model to infer latent reward structures purely off linguistic cues and to adapt accordingly. We evaluated the performance of four open-source LLMs and compare their performance to standard decision-making algorithms such as Thompson Sampling, Epsilon Greedy, Upper Confidence Bound (UCB), and random choice. While most of the LLMs underperformed compared to the baselines, Qwen3-4B, achieved the best-arm selection rate of 89.2% , which significantly outperformed both the larger LLMs and traditional methods. Our findings suggest that probabilistic reasoning is able to emerge from language alone, and we present this benchmark as a step towards evaluating decision-making capabilities in naturalistic, non-numeric contexts.
>
---
#### [new 114] FACTS: Table Summarization via Offline Template Generation with Agentic Workflows
- **分类: cs.CL**

- **简介: 该论文针对查询聚焦的表格摘要任务，解决现有方法在效率、准确性和隐私上的不足。提出FACTS框架，通过离线生成SQL与模板，实现快速、准确且隐私合规的摘要生成，支持跨表复用，无需微调或暴露敏感数据。**

- **链接: [http://arxiv.org/pdf/2510.13920v1](http://arxiv.org/pdf/2510.13920v1)**

> **作者:** Ye Yuan; Mohammad Amin Shabani; Siqi Liu
>
> **备注:** Under Review
>
> **摘要:** Query-focused table summarization requires generating natural language summaries of tabular data conditioned on a user query, enabling users to access insights beyond fact retrieval. Existing approaches face key limitations: table-to-text models require costly fine-tuning and struggle with complex reasoning, prompt-based LLM methods suffer from token-limit and efficiency issues while exposing sensitive data, and prior agentic pipelines often rely on decomposition, planning, or manual templates that lack robustness and scalability. To mitigate these issues, we introduce an agentic workflow, FACTS, a Fast, Accurate, and Privacy-Compliant Table Summarization approach via Offline Template Generation. FACTS produces offline templates, consisting of SQL queries and Jinja2 templates, which can be rendered into natural language summaries and are reusable across multiple tables sharing the same schema. It enables fast summarization through reusable offline templates, accurate outputs with executable SQL queries, and privacy compliance by sending only table schemas to LLMs. Evaluations on widely-used benchmarks show that FACTS consistently outperforms baseline methods, establishing it as a practical solution for real-world query-focused table summarization.
>
---
#### [new 115] Midtraining Bridges Pretraining and Posttraining Distributions
- **分类: cs.CL**

- **简介: 该论文研究语言模型训练中的“中期训练”（midtraining）阶段，旨在探究其作用机制。通过控制实验，分析其在数学与代码领域的有效性，发现midtraining可缩小预训练与下游任务间的语法差距，减少知识遗忘，优于持续预训练，是一种有效的领域自适应方法。**

- **链接: [http://arxiv.org/pdf/2510.14865v1](http://arxiv.org/pdf/2510.14865v1)**

> **作者:** Emmy Liu; Graham Neubig; Chenyan Xiong
>
> **摘要:** Recently, many language models have been pretrained with a "midtraining" phase, in which higher quality, often instruction-formatted data, is mixed in at the end of pretraining. Despite the popularity of this practice, there is little scientific understanding of this phase of model training or why it is effective. In this work, we conduct the first systematic investigation of midtraining through controlled experiments with language models pretrained from scratch and fine-tuned on supervised finetuning datasets in different domains. We find that when compared after supervised fine-tuning, the effectiveness of midtraining is highest in the math and code domains, where midtraining can best reduce the syntactic gap between pretraining and posttraining data. In these cases, midtraining consistently outperforms continued pretraining in both in-domain validation loss as well as pretraining data forgetting after posttraining. We conduct ablations on the starting time of the midtraining phase and mixture weights of the midtraining data, using code midtraining as a case study, and find that timing has a greater impact than mixture weights, with earlier introduction of specialized data, yielding greater benefits in-domain as well as preserving general language modeling better. These findings establish midtraining as a domain adaptation technique that compared to continued pretraining yields better performance through reduced forgetting.
>
---
#### [new 116] DynaSpec: Context-aware Dynamic Speculative Sampling for Large-Vocabulary Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对大词汇表语言模型中推测解码的推理加速任务，解决固定词汇短列表导致的性能下降问题。提出DynaSpec，利用轻量元分类器动态选择上下文相关的令牌簇作为短列表，提升草案生成速度与接受率。**

- **链接: [http://arxiv.org/pdf/2510.13847v1](http://arxiv.org/pdf/2510.13847v1)**

> **作者:** Jinbin Zhang; Nasib Ullah; Erik Schultheis; Rohit Babbar
>
> **摘要:** Speculative decoding (a.k.a. speculative sampling) has become a standard way to accelerate LLM inference: a small drafter proposes multiple tokens and a large target model verifies them once per speculation length. Recently, scaling of the LLM vocabulary has pushed the number of tokens to grow substantially. While verification over the full vocabulary leaves the target model largely unaffected, the O(|V|d) parameters in the drafter's output head become a latency bottleneck, slowing the entire pipeline. Contemporary methods (e.g., FR-Spec, VocabTrim) restrict the drafter's vocabulary to a fixed subset of the target model's vocabulary, ranked in descending order of token frequency. Although this reduces draft-time compute, it is brittle, since: (i) frequency lists are corpus-dependent and require retuning to generalize, and (ii) static shortlists suppress rare or domain-specific tokens, lowering the expected number of tokens per verification step. We propose DynaSpec, a context-dependent dynamic shortlisting mechanism that is robust, speeds up drafting, and generalizes across diverse tasks. Concretely, we introduce lightweight, coarse-grained meta-classifiers that route contexts to a small number of token clusters; the union of the top-k selected clusters forms the drafter's shortlist, while verification retains the full vocabulary and exactness. The meta-classifier finishes its computation earlier than the drafter's hidden state generation by exploiting parallel execution of draft encoding and meta shortlisting on separate streams. On standard speculative-decoding benchmarks, we observe consistent gains in mean accepted length over fixed-shortlist baselines, while context-dependent selection enables smaller shortlists without degrading acceptance.
>
---
#### [new 117] Harnessing Consistency for Robust Test-Time LLM Ensemble
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型（LLM）集成的鲁棒性问题，旨在应对因分词差异和模型能力不一导致的集成错误。提出CoRE方法，通过建模词元级和模型级一致性，提升集成系统的稳定性和性能。**

- **链接: [http://arxiv.org/pdf/2510.13855v1](http://arxiv.org/pdf/2510.13855v1)**

> **作者:** Zhichen Zeng; Qi Yu; Xiao Lin; Ruizhong Qiu; Xuying Ning; Tianxin Wei; Yuchen Yan; Jingrui He; Hanghang Tong
>
> **备注:** 15 pages, 12 figures
>
> **摘要:** Different large language models (LLMs) exhibit diverse strengths and weaknesses, and LLM ensemble serves as a promising approach to integrate their complementary capabilities. Despite substantial progress in improving ensemble quality, limited attention has been paid to the robustness of ensembles against potential erroneous signals, which often arise from heterogeneous tokenization schemes and varying model expertise. Our analysis shows that ensemble failures typically arise from both the token level and the model level: the former reflects severe disagreement in token predictions, while the latter involves low confidence and pronounced disparities among models. In light of this, we propose CoRE, a plug-and-play technique that harnesses model consistency for robust LLM ensemble, which can be seamlessly integrated with diverse ensemble methods. Token-level consistency captures fine-grained disagreements by applying a low-pass filter to downweight uncertain tokens with high inconsistency, often due to token misalignment, thereby improving robustness at a granular level. Model-level consistency models global agreement by promoting model outputs with high self-confidence and minimal divergence from others, enhancing robustness at a coarser level. Extensive experiments across diverse benchmarks, model combinations, and ensemble strategies demonstrate that CoRE consistently improves ensemble performance and robustness.
>
---
#### [new 118] R2T: Rule-Encoded Loss Functions for Low-Resource Sequence Tagging
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究低资源序列标注任务，提出R2T框架，将语言规则嵌入神经网络损失函数，通过原则性学习提升模型对未登录词的处理能力。实验证明其在POS和NER任务中显著优于基线模型。**

- **链接: [http://arxiv.org/pdf/2510.13854v1](http://arxiv.org/pdf/2510.13854v1)**

> **作者:** Mamadou K. Keita; Christopher Homan; Sebastien Diarra
>
> **摘要:** We introduce the Rule-to-Tag (R2T) framework, a hybrid approach that integrates a multi-tiered system of linguistic rules directly into a neural network's training objective. R2T's novelty lies in its adaptive loss function, which includes a regularization term that teaches the model to handle out-of-vocabulary (OOV) words with principled uncertainty. We frame this work as a case study in a paradigm we call principled learning (PrL), where models are trained with explicit task constraints rather than on labeled examples alone. Our experiments on Zarma part-of-speech (POS) tagging show that the R2T-BiLSTM model, trained only on unlabeled text, achieves 98.2% accuracy, outperforming baselines like AfriBERTa fine-tuned on 300 labeled sentences. We further show that for more complex tasks like named entity recognition (NER), R2T serves as a powerful pre-training step; a model pre-trained with R2T and fine-tuned on just 50 labeled sentences outperformes a baseline trained on 300.
>
---
#### [new 119] BenchPress: A Human-in-the-Loop Annotation System for Rapid Text-to-SQL Benchmark Curation
- **分类: cs.CL; cs.AI; cs.DB; cs.HC**

- **简介: 该论文针对私有企业场景下文本到SQL（text-to-SQL）基准构建困难的问题，提出BenchPress系统。它结合检索增强生成与大模型，自动生成自然语言描述，通过人工校验加速高质量领域特定基准的构建，提升标注效率与准确性。**

- **链接: [http://arxiv.org/pdf/2510.13853v1](http://arxiv.org/pdf/2510.13853v1)**

> **作者:** Fabian Wenz; Omar Bouattour; Devin Yang; Justin Choi; Cecil Gregg; Nesime Tatbul; Çağatay Demiralp
>
> **备注:** CIDR'26
>
> **摘要:** Large language models (LLMs) have been successfully applied to many tasks, including text-to-SQL generation. However, much of this work has focused on publicly available datasets, such as Fiben, Spider, and Bird. Our earlier work showed that LLMs are much less effective in querying large private enterprise data warehouses and released Beaver, the first private enterprise text-to-SQL benchmark. To create Beaver, we leveraged SQL logs, which are often readily available. However, manually annotating these logs to identify which natural language questions they answer is a daunting task. Asking database administrators, who are highly trained experts, to take on additional work to construct and validate corresponding natural language utterances is not only challenging but also quite costly. To address this challenge, we introduce BenchPress, a human-in-the-loop system designed to accelerate the creation of domain-specific text-to-SQL benchmarks. Given a SQL query, BenchPress uses retrieval-augmented generation (RAG) and LLMs to propose multiple natural language descriptions. Human experts then select, rank, or edit these drafts to ensure accuracy and domain alignment. We evaluated BenchPress on annotated enterprise SQL logs, demonstrating that LLM-assisted annotation drastically reduces the time and effort required to create high-quality benchmarks. Our results show that combining human verification with LLM-generated suggestions enhances annotation accuracy, benchmark reliability, and model evaluation robustness. By streamlining the creation of custom benchmarks, BenchPress offers researchers and practitioners a mechanism for assessing text-to-SQL models on a given domain-specific workload. BenchPress is freely available via our public GitHub repository at https://github.com/fabian-wenz/enterprise-txt2sql and is also accessible on our website at http://dsg-mcgraw.csail.mit.edu:5000.
>
---
#### [new 120] LLM Prompt Duel Optimizer: Efficient Label-Free Prompt Optimization
- **分类: cs.CL; stat.ML**

- **简介: 该论文研究无需标签的提示词优化任务，解决人工标注成本高的问题。提出Prompt Duel Optimizer（PDO），利用大模型自身作为裁判提供偏好反馈，结合双汤普森采样与高性能提示变异，实现高效、无监督的提示优化，在多个基准上优于基线方法。**

- **链接: [http://arxiv.org/pdf/2510.13907v1](http://arxiv.org/pdf/2510.13907v1)**

> **作者:** Yuanchen Wu; Saurabh Verma; Justin Lee; Fangzhou Xiong; Poppy Zhang; Amel Awadelkarim; Xu Chen; Yubai Yuan; Shawndra Hill
>
> **摘要:** Large language models (LLMs) are highly sensitive to their input prompts, making prompt design a central challenge. While automatic prompt optimization (APO) reduces manual engineering, most approaches assume access to ground-truth references such as labeled validation data. In practice, however, collecting high-quality labels is costly and slow. We propose the Prompt Duel Optimizer (PDO), a sample-efficient framework for label-free prompt optimization. PDO formulates the problem as a dueling-bandit setting, where supervision signal comes from pairwise preference feedback provided by an LLM judge. The framework combines Double Thompson Sampling (D-TS), which prioritizes informative prompt comparisons, with Top-Performer Guided Mutation, which expands the candidate pool by mutating high-performing prompts. PDO naturally operates in label-free settings and can also incorporate partial labels to mitigate judge noise. Experiments on BIG-bench Hard (BBH) and MS MARCO show that PDO consistently outperforms baseline methods. Ablation studies further demonstrate the effectiveness of both D-TS and prompt mutation.
>
---
#### [new 121] PluriHop: Exhaustive, Recall-Sensitive QA over Distractor-Rich Corpora
- **分类: cs.CL; cs.IR; cs.LG**

- **简介: 该论文研究面向重复性报告数据的问答任务，提出“多跳全量问答”（pluri-hop QA）问题，解决现有方法在高干扰、需全覆盖场景下的不足。作者构建多语言风电报告数据集PluriHopWIND，并提出PluriHopRAG框架，通过子问题分解与早期过滤提升召回与精度。**

- **链接: [http://arxiv.org/pdf/2510.14377v1](http://arxiv.org/pdf/2510.14377v1)**

> **作者:** Mykolas Sveistrys; Richard Kunert
>
> **摘要:** Recent advances in large language models (LLMs) and retrieval-augmented generation (RAG) have enabled progress on question answering (QA) when relevant evidence is in one (single-hop) or multiple (multi-hop) passages. Yet many realistic questions about recurring report data - medical records, compliance filings, maintenance logs - require aggregation across all documents, with no clear stopping point for retrieval and high sensitivity to even one missed passage. We term these pluri-hop questions and formalize them by three criteria: recall sensitivity, exhaustiveness, and exactness. To study this setting, we introduce PluriHopWIND, a diagnostic multilingual dataset of 48 pluri-hop questions built from 191 real-world wind industry reports in German and English. We show that PluriHopWIND is 8-40% more repetitive than other common datasets and thus has higher density of distractor documents, better reflecting practical challenges of recurring report corpora. We test a traditional RAG pipeline as well as graph-based and multimodal variants, and find that none of the tested approaches exceed 40% in statement-wise F1 score. Motivated by this, we propose PluriHopRAG, a RAG architecture that follows a "check all documents individually, filter cheaply" approach: it (i) decomposes queries into document-level subquestions and (ii) uses a cross-encoder filter to discard irrelevant documents before costly LLM reasoning. We find that PluriHopRAG achieves relative F1 score improvements of 18-52% depending on base LLM. Despite its modest size, PluriHopWIND exposes the limitations of current QA systems on repetitive, distractor-rich corpora. PluriHopRAG's performance highlights the value of exhaustive retrieval and early filtering as a powerful alternative to top-k methods.
>
---
#### [new 122] Classifying and Addressing the Diversity of Errors in Retrieval-Augmented Generation Systems
- **分类: cs.CL; cs.LG**

- **简介: 该论文针对检索增强生成（RAG）系统中的错误类型进行分类，提出错误分类体系并构建标注数据集，设计自动化评估方法以识别和解决RAG错误，旨在提升系统鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.13975v1](http://arxiv.org/pdf/2510.13975v1)**

> **作者:** Kin Kwan Leung; Mouloud Belbahri; Yi Sui; Alex Labach; Xueying Zhang; Stephen Rose; Jesse C. Cresswell
>
> **备注:** 8 pages
>
> **摘要:** Retrieval-augmented generation (RAG) is a prevalent approach for building LLM-based question-answering systems that can take advantage of external knowledge databases. Due to the complexity of real-world RAG systems, there are many potential causes for erroneous outputs. Understanding the range of errors that can occur in practice is crucial for robust deployment. We present a new taxonomy of the error types that can occur in realistic RAG systems, examples of each, and practical advice for addressing them. Additionally, we curate a dataset of erroneous RAG responses annotated by error types. We then propose an auto-evaluation method aligned with our taxonomy that can be used in practice to track and address errors during development. Code and data are available at https://github.com/layer6ai-labs/rag-error-classification.
>
---
#### [new 123] CURE: Confidence-driven Unified Reasoning Ensemble Framework for Medical Question Answering
- **分类: cs.CL; cs.AI; physics.med-ph**

- **简介: 该论文针对医疗大模型需昂贵微调的问题，提出CURE框架，通过置信度驱动的多模型协作机制，在无需微调的情况下提升医学问答性能，实现在资源受限场景下的高效推理。**

- **链接: [http://arxiv.org/pdf/2510.14353v1](http://arxiv.org/pdf/2510.14353v1)**

> **作者:** Ziad Elshaer; Essam A. Rashed
>
> **摘要:** High-performing medical Large Language Models (LLMs) typically require extensive fine-tuning with substantial computational resources, limiting accessibility for resource-constrained healthcare institutions. This study introduces a confidence-driven multi-model framework that leverages model diversity to enhance medical question answering without fine-tuning. Our framework employs a two-stage architecture: a confidence detection module assesses the primary model's certainty, and an adaptive routing mechanism directs low-confidence queries to Helper models with complementary knowledge for collaborative reasoning. We evaluate our approach using Qwen3-30B-A3B-Instruct, Phi-4 14B, and Gemma 2 12B across three medical benchmarks; MedQA, MedMCQA, and PubMedQA. Result demonstrate that our framework achieves competitive performance, with particularly strong results in PubMedQA (95.0\%) and MedMCQA (78.0\%). Ablation studies confirm that confidence-aware routing combined with multi-model collaboration substantially outperforms single-model approaches and uniform reasoning strategies. This work establishes that strategic model collaboration offers a practical, computationally efficient pathway to improve medical AI systems, with significant implications for democratizing access to advanced medical AI in resource-limited settings.
>
---
#### [new 124] An Efficient Rubric-based Generative Verifier for Search-Augmented LLMs
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文针对搜索增强型大模型的奖励建模问题，提出“nugget-as-rubric”可验证范式，将信息点作为评分标准，并构建自动流水线生成多粒度评分规则，进而设计高效生成式验证器Search-Gen-V，提升长文本任务下奖励信号的鲁棒性与可验证性。**

- **链接: [http://arxiv.org/pdf/2510.14660v1](http://arxiv.org/pdf/2510.14660v1)**

> **作者:** Linyue Ma; Yilong Xu; Xiang Long; Zhi Zheng
>
> **摘要:** Search augmentation empowers Large Language Models with retrieval capabilities to overcome the limitations imposed by static parameters. Recently, Reinforcement Learning leverages tailored reward signals as a viable technique to enhance LLMs performing tasks involving search. However, existing reward modeling for search-augmented LLMs faces several limitations. Rule-based rewards, such as Exact Match, are verifiable but fragile to variations in expression and cannot be applied to long-form workloads. In contrast, generative rewards improve robustness, but designing verifiable and stable rewards for long-form workloads in dynamic corpora remains challenging and also incurs high computational costs. In this paper, we propose a unified and verifiable paradigm, "nugget-as-rubric", which treats atomic information points as structured evaluation criteria for different search-augmentation workloads. Short-form tasks correspond to a single rubric, whereas long-form tasks expand to multiple rubrics aligned with the question's information needs. To support long-form settings, we design an automatic rubric construction pipeline based on query rewriting, which can automatically retrieve passages relevant to each question and extract rubrics from them, both from static corpora and from dynamic online web content. Furthermore, we introduce \textbf{Search-Gen-V}, a 4B-parameter efficient generative verifier under our proposed verifiable paradigm, which is trained via the idea of distillation and a two-stage strategy. Experimental results show that Search-Gen-V achieves strong verification accuracy across different workloads, making it a scalable, robust, and efficient verifiable reward constructor for search-augmented LLMs.
>
---
#### [new 125] E2Edev: Benchmarking Large Language Models in End-to-End Software Development Task
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文聚焦端到端软件开发任务，旨在评估大语言模型在真实开发场景中的能力。作者构建了含细粒度需求、多测试场景及自动化测试流程的基准E2EDev，并提出人机协同标注框架以降本提质，揭示现有方法仍面临有效性与成本挑战。**

- **链接: [http://arxiv.org/pdf/2510.14509v1](http://arxiv.org/pdf/2510.14509v1)**

> **作者:** Jingyao Liu; Chen Huang; Zhizhao Guan; Wenqiang Lei; Yang Deng
>
> **摘要:** E2EDev comprises (i) a fine-grained set of user requirements, (ii) {multiple BDD test scenarios with corresponding Python step implementations for each requirement}, and (iii) a fully automated testing pipeline built on the Behave framework. To ensure its quality while reducing the annotation effort, E2EDev leverages our proposed Human-in-the-Loop Multi-Agent Annotation Framework (HITL-MAA). {By evaluating various E2ESD frameworks and LLM backbones with E2EDev}, our analysis reveals a persistent struggle to effectively solve these tasks, underscoring the critical need for more effective and cost-efficient E2ESD solutions. Our codebase and benchmark are publicly available at https://github.com/SCUNLP/E2EDev.
>
---
#### [new 126] Circuit Insights: Towards Interpretability Beyond Activations
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属可解释AI任务，旨在解决现有方法依赖激活和数据、难捕捉特征交互的问题。提出WeightLens和CircuitLens，分别从权重解析特征、揭示电路级动态，实现更鲁棒、高效的自动化电路分析。**

- **链接: [http://arxiv.org/pdf/2510.14936v1](http://arxiv.org/pdf/2510.14936v1)**

> **作者:** Elena Golimblevskaia; Aakriti Jain; Bruno Puri; Ammar Ibrahim; Wojciech Samek; Sebastian Lapuschkin
>
> **摘要:** The fields of explainable AI and mechanistic interpretability aim to uncover the internal structure of neural networks, with circuit discovery as a central tool for understanding model computations. Existing approaches, however, rely on manual inspection and remain limited to toy tasks. Automated interpretability offers scalability by analyzing isolated features and their activations, but it often misses interactions between features and depends strongly on external LLMs and dataset quality. Transcoders have recently made it possible to separate feature attributions into input-dependent and input-invariant components, providing a foundation for more systematic circuit analysis. Building on this, we propose WeightLens and CircuitLens, two complementary methods that go beyond activation-based analysis. WeightLens interprets features directly from their learned weights, removing the need for explainer models or datasets while matching or exceeding the performance of existing methods on context-independent features. CircuitLens captures how feature activations arise from interactions between components, revealing circuit-level dynamics that activation-only approaches cannot identify. Together, these methods increase interpretability robustness and enhance scalable mechanistic analysis of circuits while maintaining efficiency and quality.
>
---
#### [new 127] TRI-DEP: A Trimodal Comparative Study for Depression Detection Using Speech, Text, and EEG
- **分类: cs.AI; cs.CL; cs.LG; eess.AS; eess.SP**

- **简介: 该论文研究抑郁症自动检测，属多模态情感计算任务。针对现有研究缺乏系统比较与统一评估的问题，提出TRI-DEP框架，系统比较语音、文本和EEG的单、双、三模态组合，分析特征表示与融合策略，验证三模态结合预训练模型可提升性能。**

- **链接: [http://arxiv.org/pdf/2510.14922v1](http://arxiv.org/pdf/2510.14922v1)**

> **作者:** Annisaa Fitri Nurfidausi; Eleonora Mancini; Paolo Torroni
>
> **摘要:** Depression is a widespread mental health disorder, yet its automatic detection remains challenging. Prior work has explored unimodal and multimodal approaches, with multimodal systems showing promise by leveraging complementary signals. However, existing studies are limited in scope, lack systematic comparisons of features, and suffer from inconsistent evaluation protocols. We address these gaps by systematically exploring feature representations and modelling strategies across EEG, together with speech and text. We evaluate handcrafted features versus pre-trained embeddings, assess the effectiveness of different neural encoders, compare unimodal, bimodal, and trimodal configurations, and analyse fusion strategies with attention to the role of EEG. Consistent subject-independent splits are applied to ensure robust, reproducible benchmarking. Our results show that (i) the combination of EEG, speech and text modalities enhances multimodal detection, (ii) pretrained embeddings outperform handcrafted features, and (iii) carefully designed trimodal models achieve state-of-the-art performance. Our work lays the groundwork for future research in multimodal depression detection.
>
---
#### [new 128] Scaling Test-Time Compute to Achieve IOI Gold Medal with Open-Weight Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文针对开放权重模型在国际信息学奥林匹克（IOI）竞赛中表现不足的问题，提出GenCluster框架，通过扩展测试时计算资源，结合生成、聚类、排序与提交策略，首次实现开放模型达到IOI金牌水平。**

- **链接: [http://arxiv.org/pdf/2510.14232v1](http://arxiv.org/pdf/2510.14232v1)**

> **作者:** Mehrzad Samadi; Aleksander Ficek; Sean Narenthiran; Siddhartha Jain; Wasi Uddin Ahmad; Somshubra Majumdar; Vahid Noroozi; Boris Ginsburg
>
> **备注:** 14 pages, 11 figures
>
> **摘要:** Competitive programming has become a rigorous benchmark for evaluating the reasoning and problem-solving capabilities of large language models (LLMs). The International Olympiad in Informatics (IOI) stands out as one of the most prestigious annual competitions in competitive programming and has become a key benchmark for comparing human and AI-level programming ability. While several proprietary models have been claimed to achieve gold medal-level performance at the IOI, often with undisclosed methods, achieving comparable results with open-weight models remains a significant challenge. In this paper, we present \gencluster, a scalable and reproducible test-time compute framework that attains IOI gold-level performance using open-weight models. It combines large-scale generation, behavioral clustering, ranking, and a round-robin submission strategy to efficiently explore diverse solution spaces under limited validation budgets. Our experiments show that the performance of our proposed approach scales consistently with available compute, narrowing the gap between open and closed systems. Notably, we will show that GenCluster can achieve a gold medal at IOI 2025 for the first time with an open-weight model gpt-oss-120b, setting a new benchmark for transparent and reproducible evaluation of reasoning in LLMs.
>
---
#### [new 129] CAST: Compositional Analysis via Spectral Tracking for Understanding Transformer Layer Functions
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属模型可解释性任务，旨在揭示Transformer各层功能。提出CAST框架，通过估计变换矩阵并进行谱分析，无需探针即可识别编码器与解码器模型的分层行为模式。**

- **链接: [http://arxiv.org/pdf/2510.14262v1](http://arxiv.org/pdf/2510.14262v1)**

> **作者:** Zihao Fu; Ming Liao; Chris Russell; Zhenguang G. Cai
>
> **摘要:** Large language models have achieved remarkable success but remain largely black boxes with poorly understood internal mechanisms. To address this limitation, many researchers have proposed various interpretability methods including mechanistic analysis, probing classifiers, and activation visualization, each providing valuable insights from different perspectives. Building upon this rich landscape of complementary approaches, we introduce CAST (Compositional Analysis via Spectral Tracking), a probe-free framework that contributes a novel perspective by analyzing transformer layer functions through direct transformation matrix estimation and comprehensive spectral analysis. CAST offers complementary insights to existing methods by estimating the realized transformation matrices for each layer using Moore-Penrose pseudoinverse and applying spectral analysis with six interpretable metrics characterizing layer behavior. Our analysis reveals distinct behaviors between encoder-only and decoder-only models, with decoder models exhibiting compression-expansion cycles while encoder models maintain consistent high-rank processing. Kernel analysis further demonstrates functional relationship patterns between layers, with CKA similarity matrices clearly partitioning layers into three phases: feature extraction, compression, and specialization.
>
---
#### [new 130] Efficient Parallel Samplers for Recurrent-Depth Models and Their Connection to Diffusion Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究 recurrent-depth 语言模型与扩散语言模型的关系，提出一种无需调参、可直接用于现有模型的并行采样器，通过并行细化隐状态加速生成，实现最高5倍提速，揭示此类模型可视为强因果扩散模型。**

- **链接: [http://arxiv.org/pdf/2510.14961v1](http://arxiv.org/pdf/2510.14961v1)**

> **作者:** Jonas Geiping; Xinyu Yang; Guinan Su
>
> **备注:** Code can be found at https://github.com/seal-rg/recurrent-pretraining
>
> **摘要:** Language models with recurrent depth, also referred to as universal or looped when considering transformers, are defined by the capacity to increase their computation through the repetition of layers. Recent efforts in pretraining have demonstrated that these architectures can scale to modern language modeling tasks while exhibiting advantages in reasoning tasks. In this work, we examine the relationship between recurrent-depth models and diffusion language models. Building on their similarities, we develop a new diffusion forcing sampler for these models to accelerate generation. The sampler advances by decoding new tokens at every forward pass of the model, while the latent states of these tokens can be further refined in parallel through recurrence. Theoretically, generation with our sampler is strictly more expressive than the baseline autoregressive generation using the same time budget on modern hardware. Moreover, this sampler, based on principles from diffusion literature, can be directly applied to existing 3.5B recurrent-depth transformers without any tuning, leading to up to a 5x speedup. Consequently, our findings not only provide an efficient mechanism for parallelizing the extra computation in recurrent-depth models at inference, but also suggest that such models can be naturally viewed as strong continuous, though causal, diffusion language models.
>
---
#### [new 131] Where to Search: Measure the Prior-Structured Search Space of LLM Agents
- **分类: cs.AI; cs.CL; cs.LO**

- **简介: 该论文研究LLM智能体在科学推理等任务中的迭代搜索问题，旨在将领域先验融入可操作的假设空间。作者提出一种形式化理论，用模糊关系算子和路径生成函数刻画搜索过程，度量可达性难度，并通过几何视角分析受安全约束的搜索空间结构。**

- **链接: [http://arxiv.org/pdf/2510.14846v1](http://arxiv.org/pdf/2510.14846v1)**

> **作者:** Zhuo-Yang Song
>
> **备注:** 10 pages, 2 figures, 1 table
>
> **摘要:** The generate-filter-refine (iterative paradigm) based on large language models (LLMs) has achieved progress in reasoning, programming, and program discovery in AI+Science. However, the effectiveness of search depends on where to search, namely, how to encode the domain prior into an operationally structured hypothesis space. To this end, this paper proposes a compact formal theory that describes and measures LLM-assisted iterative search guided by domain priors. We represent an agent as a fuzzy relation operator on inputs and outputs to capture feasible transitions; the agent is thereby constrained by a fixed safety envelope. To describe multi-step reasoning/search, we weight all reachable paths by a single continuation parameter and sum them to obtain a coverage generating function; this induces a measure of reachability difficulty; and it provides a geometric interpretation of search on the graph induced by the safety envelope. We further provide the simplest testable inferences and validate them via a majority-vote instantiation. This theory offers a workable language and operational tools to measure agents and their search spaces, proposing a systematic formal description of iterative search constructed by LLMs.
>
---
#### [new 132] Are My Optimized Prompts Compromised? Exploring Vulnerabilities of LLM-based Optimizers
- **分类: cs.LG; cs.AI; cs.CL; cs.CR**

- **简介: 该论文研究LLM提示优化器的安全问题，揭示反馈中毒攻击可显著提升攻击成功率。作者提出一种无需访问奖励模型的伪造奖励攻击，并设计轻量级高亮防御方法有效降低风险，强调需加强优化框架的安全防护。**

- **链接: [http://arxiv.org/pdf/2510.14381v1](http://arxiv.org/pdf/2510.14381v1)**

> **作者:** Andrew Zhao; Reshmi Ghosh; Vitor Carvalho; Emily Lawton; Keegan Hines; Gao Huang; Jack W. Stokes
>
> **摘要:** Large language model (LLM) systems now underpin everyday AI applications such as chatbots, computer-use assistants, and autonomous robots, where performance often depends on carefully designed prompts. LLM-based prompt optimizers reduce that effort by iteratively refining prompts from scored feedback, yet the security of this optimization stage remains underexamined. We present the first systematic analysis of poisoning risks in LLM-based prompt optimization. Using HarmBench, we find systems are substantially more vulnerable to manipulated feedback than to injected queries: feedback-based attacks raise attack success rate (ASR) by up to $\Delta$ASR = 0.48. We introduce a simple fake-reward attack that requires no access to the reward model and significantly increases vulnerability, and we propose a lightweight highlighting defense that reduces the fake-reward $\Delta$ASR from 0.23 to 0.07 without degrading utility. These results establish prompt optimization pipelines as a first-class attack surface and motivate stronger safeguards for feedback channels and optimization frameworks.
>
---
#### [new 133] Towards Reversible Model Merging For Low-rank Weights
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文研究低秩权重下的模型合并任务，旨在解决传统合并方法在低秩模型上性能严重下降的问题。作者提出可逆模型合并（RMM），通过构建可恢复原始任务模型的紧凑基空间，实现高效、无数据、闭式求解的模型融合。**

- **链接: [http://arxiv.org/pdf/2510.14163v1](http://arxiv.org/pdf/2510.14163v1)**

> **作者:** Mohammadsajad Alipour; Mohammad Mohammadi Amiri
>
> **摘要:** Model merging aims to combine multiple fine-tuned models into a single set of weights that performs well across all source tasks. While prior work has shown that merging can approximate the performance of individual fine-tuned models for each task, it largely overlooks scenarios where models are compressed into low-rank representations, either through low-rank adaptation (LoRA) or post-training singular value decomposition (SVD). We first demonstrate that applying conventional merging methods to low-rank weights leads to severe performance degradation in the merged model. Motivated by this phenomenon, we propose a fundamentally different approach: instead of collapsing all adapters into one set of weights, we construct a compact basis (e.g., an equivalent of holding two or more models) from which original task-specific models can be recovered via linear combination. This reframes merging as generating a reconstruction-capable model space rather than producing a single merged model. Crucially, this allows us to ``revert'' to each individual model when needed, recognizing that no merged model can consistently outperform one specialized for its task. Building on this insight, we introduce our method, Reversible Model Merging (RMM), an efficient, data-free, and flexible method that provides a closed-form solution for selecting the optimal basis of model weights and task-specific coefficients for linear combination. Extensive experiments across diverse datasets and model scales demonstrate that RMM consistently outperforms existing merging approaches, preserving the performance of low-rank compressed models by a significant margin.
>
---
#### [new 134] BitNet Distillation
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出BitNet蒸馏（BitDistill），旨在将全精度大模型高效蒸馏为1.58位模型。针对任务特定微调中低比特模型性能下降问题，引入SubLN、多头注意力蒸馏和持续预训练，实现接近全精度性能，显著降低内存与推理成本。**

- **链接: [http://arxiv.org/pdf/2510.13998v1](http://arxiv.org/pdf/2510.13998v1)**

> **作者:** Xun Wu; Shaohan Huang; Wenhui Wang; Ting Song; Li Dong; Yan Xia; Furu Wei
>
> **备注:** 12 pages, 4 figures
>
> **摘要:** In this paper, we present BitNet Distillation (BitDistill), a lightweight pipeline that fine-tunes off-the-shelf full-precision LLMs (e.g., Qwen) into 1.58-bit precision (i.e., ternary weights {-1, 0, 1}) for specific downstream tasks, achieving strong task-specific performance with minimal computational cost. Specifically, BitDistill incorporates three key techniques: the SubLN module, as introduced in BitNet; multi-head attention distillation, based on MiniLM; and continual pre-training, which serves as a crucial warm-up step to mitigate the scalability issue of the performance gap between finetuned full-precision and 1.58-bit LLMs on specific tasks. Experimental results show that BitDistill achieves performance comparable to the full-precision counterpart models across model size, while enabling up to 10x memory savings and 2.65x faster inference on CPUs. Code is available at https://github.com/microsoft/BitNet.
>
---
#### [new 135] Reasoning with Sampling: Your Base Model is Smarter Than You Think
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究如何通过纯采样方法激发基础大模型的推理能力，无需额外训练。提出一种受MCMC启发的迭代采样算法，在数学、代码和知识推理任务上接近或超越强化学习微调的效果，且保持生成多样性。**

- **链接: [http://arxiv.org/pdf/2510.14901v1](http://arxiv.org/pdf/2510.14901v1)**

> **作者:** Aayush Karan; Yilun Du
>
> **摘要:** Frontier reasoning models have exhibited incredible capabilities across a wide array of disciplines, driven by posttraining large language models (LLMs) with reinforcement learning (RL). However, despite the widespread success of this paradigm, much of the literature has been devoted to disentangling truly novel behaviors that emerge during RL but are not present in the base models. In our work, we approach this question from a different angle, instead asking whether comparable reasoning capabilites can be elicited from base models at inference time by pure sampling, without any additional training. Inspired by Markov chain Monte Carlo (MCMC) techniques for sampling from sharpened distributions, we propose a simple iterative sampling algorithm leveraging the base models' own likelihoods. Over different base models, we show that our algorithm offers substantial boosts in reasoning that nearly match and even outperform those from RL on a wide variety of single-shot tasks, including MATH500, HumanEval, and GPQA. Moreover, our sampler avoids the collapse in diversity over multiple samples that is characteristic of RL-posttraining. Crucially, our method does not require training, curated datasets, or a verifier, suggesting broad applicability beyond easily verifiable domains.
>
---
#### [new 136] MAFA: A Multi-Agent Framework for Enterprise-Scale Annotation with Configurable Task Adaptation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出MAFA，一个可配置的多智能体框架，用于企业级标注任务。针对金融领域海量客户语句标注 backlog 问题，通过多代理协作与动态任务适配，实现高准确率自动标注，显著减少人工工作量并提升效率。**

- **链接: [http://arxiv.org/pdf/2510.14184v1](http://arxiv.org/pdf/2510.14184v1)**

> **作者:** Mahmood Hegazy; Aaron Rodrigues; Azzam Naeem
>
> **摘要:** We present MAFA (Multi-Agent Framework for Annotation), a production-deployed system that transforms enterprise-scale annotation workflows through configurable multi-agent collaboration. Addressing the critical challenge of annotation backlogs in financial services, where millions of customer utterances require accurate categorization, MAFA combines specialized agents with structured reasoning and a judge-based consensus mechanism. Our framework uniquely supports dynamic task adaptation, allowing organizations to define custom annotation types (FAQs, intents, entities, or domain-specific categories) through configuration rather than code changes. Deployed at JP Morgan Chase, MAFA has eliminated a 1 million utterance backlog while achieving, on average, 86% agreement with human annotators, annually saving over 5,000 hours of manual annotation work. The system processes utterances with annotation confidence classifications, which are typically 85% high, 10% medium, and 5% low across all datasets we tested. This enables human annotators to focus exclusively on ambiguous and low-coverage cases. We demonstrate MAFA's effectiveness across multiple datasets and languages, showing consistent improvements over traditional and single-agent annotation baselines: 13.8% higher Top-1 accuracy, 15.1% improvement in Top-5 accuracy, and 16.9% better F1 in our internal intent classification dataset and similar gains on public benchmarks. This work bridges the gap between theoretical multi-agent systems and practical enterprise deployment, providing a blueprint for organizations facing similar annotation challenges.
>
---
#### [new 137] ColorBench: Benchmarking Mobile Agents with Graph-Structured Framework for Complex Long-Horizon Tasks
- **分类: cs.AI; cs.CL**

- **简介: 该论文针对移动智能体在复杂长周期任务中的评估难题，提出图结构化基准框架ColorBench，支持多正确路径与错误路径模拟，实现静态可复现的动态行为测试，涵盖175个任务，推动对代理能力的细粒度分析与提升。**

- **链接: [http://arxiv.org/pdf/2510.14621v1](http://arxiv.org/pdf/2510.14621v1)**

> **作者:** Yuanyi Song; Heyuan Huang; Qiqiang Lin; Yin Zhao; Xiangmou Qu; Jun Wang; Xingyu Lou; Weiwen Liu; Zhuosheng Zhang; Jun Wang; Yong Yu; Weinan Zhang; Zhaoxiang Wang
>
> **摘要:** The rapid advancement of multimodal large language models has enabled agents to operate mobile devices by directly interacting with graphical user interfaces, opening new possibilities for mobile automation. However, real-world mobile tasks are often complex and allow for multiple valid solutions. This contradicts current mobile agent evaluation standards: offline static benchmarks can only validate a single predefined "golden path", while online dynamic testing is constrained by the complexity and non-reproducibility of real devices, making both approaches inadequate for comprehensively assessing agent capabilities. To bridge the gap between offline and online evaluation and enhance testing stability, this paper introduces a novel graph-structured benchmarking framework. By modeling the finite states observed during real-device interactions, it achieves static simulation of dynamic behaviors. Building on this, we develop ColorBench, a benchmark focused on complex long-horizon tasks. It supports evaluation of multiple valid solutions, subtask completion rate statistics, and atomic-level capability analysis. ColorBench contains 175 tasks (74 single-app, 101 cross-app) with an average length of over 13 steps. Each task includes at least two correct paths and several typical error paths, enabling quasi-dynamic interaction. By evaluating ColorBench across various baselines, we discover limitations of existing models and propose improvement directions and feasible technical pathways to enhance agents' performance on complex, long-horizon problems based on experimental results. Code and data are available at: https://github.com/MadeAgents/ColorBench.
>
---
#### [new 138] TITAN: Graph-Executable Reasoning for Cyber Threat Intelligence
- **分类: cs.AI; cs.CL; cs.CR; cs.IR**

- **简介: 该论文提出TITAN框架，属于知识图谱与网络安全交叉任务，旨在通过自然语言查询实现对网络威胁情报的可执行推理。它构建路径规划模型与图执行器，结合TITAN数据集，支持从文本到结构化推理路径的生成与验证。**

- **链接: [http://arxiv.org/pdf/2510.14670v1](http://arxiv.org/pdf/2510.14670v1)**

> **作者:** Marco Simoni; Aleksandar Fontana; Andrea Saracino; Paolo Mori
>
> **摘要:** TITAN (Threat Intelligence Through Automated Navigation) is a framework that connects natural-language cyber threat queries with executable reasoning over a structured knowledge graph. It integrates a path planner model, which predicts logical relation chains from text, and a graph executor that traverses the TITAN Ontology to retrieve factual answers and supporting evidence. Unlike traditional retrieval systems, TITAN operates on a typed, bidirectional graph derived from MITRE, allowing reasoning to move clearly and reversibly between threats, behaviors, and defenses. To support training and evaluation, we introduce the TITAN Dataset, a corpus of 88209 examples (Train: 74258; Test: 13951) pairing natural language questions with executable reasoning paths and step by step Chain of Thought explanations. Empirical evaluations show that TITAN enables models to generate syntactically valid and semantically coherent reasoning paths that can be deterministically executed on the underlying graph.
>
---
#### [new 139] AI for Service: Proactive Assistance with AI Glasses
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出AI4Service新范式，旨在实现主动式服务。针对现有AI被动响应问题，设计Alpha-Service框架，集成感知、决策、执行与记忆模块，通过AI眼镜实现实时环境理解与用户意图推断，支持无需显式指令的个性化服务。**

- **链接: [http://arxiv.org/pdf/2510.14359v1](http://arxiv.org/pdf/2510.14359v1)**

> **作者:** Zichen Wen; Yiyu Wang; Chenfei Liao; Boxue Yang; Junxian Li; Weifeng Liu; Haocong He; Bolong Feng; Xuyang Liu; Yuanhuiyi Lyu; Xu Zheng; Xuming Hu; Linfeng Zhang
>
> **备注:** 24 pages, 5 figures, work in progress
>
> **摘要:** In an era where AI is evolving from a passive tool into an active and adaptive companion, we introduce AI for Service (AI4Service), a new paradigm that enables proactive and real-time assistance in daily life. Existing AI services remain largely reactive, responding only to explicit user commands. We argue that a truly intelligent and helpful assistant should be capable of anticipating user needs and taking actions proactively when appropriate. To realize this vision, we propose Alpha-Service, a unified framework that addresses two fundamental challenges: Know When to intervene by detecting service opportunities from egocentric video streams, and Know How to provide both generalized and personalized services. Inspired by the von Neumann computer architecture and based on AI glasses, Alpha-Service consists of five key components: an Input Unit for perception, a Central Processing Unit for task scheduling, an Arithmetic Logic Unit for tool utilization, a Memory Unit for long-term personalization, and an Output Unit for natural human interaction. As an initial exploration, we implement Alpha-Service through a multi-agent system deployed on AI glasses. Case studies, including a real-time Blackjack advisor, a museum tour guide, and a shopping fit assistant, demonstrate its ability to seamlessly perceive the environment, infer user intent, and provide timely and useful assistance without explicit prompts.
>
---
#### [new 140] Talking Points: Describing and Localizing Pixels
- **分类: cs.CV; cs.CL**

- **简介: 该论文聚焦像素级视觉-语言定位任务，旨在实现通过自然语言描述精确定位图像中的关键点。提出由点描述器和点定位器组成的框架，构建数据集LlamaPointInPart并设计评估协议，实现细粒度跨模态对齐。**

- **链接: [http://arxiv.org/pdf/2510.14583v1](http://arxiv.org/pdf/2510.14583v1)**

> **作者:** Matan Rusanovsky; Shimon Malnick; Shai Avidan
>
> **摘要:** Vision-language models have achieved remarkable success in cross-modal understanding. Yet, these models remain limited to object-level or region-level grounding, lacking the capability for pixel-precise keypoint comprehension through natural language. We introduce a novel framework for pixel level grounding. The framework consists of two complementary components: a Point Descriptor that generates rich, contextual descriptions of individual keypoints, and a Point Localizer that regresses precise pixel coordinates from these descriptions. Unlike prior work that relies on templated prompts or keypoint names, our approach produces free-form, coarse-to-fine descriptions that situate keypoints within their visual context. Since there is no available dataset to train such a system, we introduce LlamaPointInPart, a carefully curated dataset of 20K+ image-keypoint-description triplets synthesized from multiple vision-language models, capturing multi-scale information from scene-level context to visual features around the keypoint. For cross-category generalization, we optimize the Point Descriptor on AP-10K via GRPO, using the frozen Point Localizer as a reward model to produce descriptions that maximize localization accuracy. To evaluate our results we establish a new evaluation protocol. Instead of comparing the text description produced by our method to the ground truth, we use the localizer to determine how close is the predicted point generated to the ground truth point. Experiments demonstrate superior performance compared to baseline models on LlamaPointInPart.The bidirectional nature of our framework should enable future applications in both keypoint-guided image understanding and language-guided precise localization. Our code and dataset are publicly available at https://github.com/matanr/Talking_Points.
>
---
#### [new 141] Budget-aware Test-time Scaling via Discriminative Verification
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究测试时扩展中的预算感知问题，旨在降低生成式验证的高计算成本。提出结合判别式验证与自一致性，实现在有限计算下显著提升大模型推理性能，优于现有生成式方法。**

- **链接: [http://arxiv.org/pdf/2510.14913v1](http://arxiv.org/pdf/2510.14913v1)**

> **作者:** Kyle Montgomery; Sijun Tan; Yuqi Chen; Siyuan Zhuang; Tianjun Zhang; Raluca Ada Popa; Chenguang Wang
>
> **摘要:** Test-time scaling is a powerful strategy for boosting the performance of large language models on complex reasoning tasks. While state-of-the-art approaches often employ generative verifiers to select the best solution from a pool of candidates, this method incurs prohibitive computational costs, limiting its practicality. In this work, we shift the focus to a more budget-aware paradigm: discriminative verification. We conduct a thorough empirical analysis and demonstrate that while discriminative verifiers may underperform in isolation, combining them with self-consistency in a hybrid approach creates a powerful and efficient test-time scaling mechanism. Notably, under a fixed compute budget, this hybrid approach surpasses state-of-the-art generative verification by a significant margin: achieving up to 15.3\% higher accuracy on AIME2025. Our findings establish that for practical, real-world applications, budget-aware scaling with discriminative verifiers is not only a "free" upgrade over self-consistency, but also a more effective and efficient alternative to costly generative techniques. Code is available at https://github.com/wang-research-lab/verification.
>
---
#### [new 142] IMAGINE: Integrating Multi-Agent System into One Model for Complex Reasoning and Planning
- **分类: cs.AI; cs.CL**

- **简介: 该论文针对大模型在复杂推理与规划中的不足，提出IMAGINE框架，将多智能体系统能力集成到单一模型中，通过端到端训练提升性能。基于Qwen3-8B实现82.7%通过率，显著优于大模型。**

- **链接: [http://arxiv.org/pdf/2510.14406v1](http://arxiv.org/pdf/2510.14406v1)**

> **作者:** Xikai Zhang; Bo Wang; Likang Xiao; Yongzhi Li; Quan Chen; Wenju Wu; Liu Liu
>
> **摘要:** Although large language models (LLMs) have made significant strides across various tasks, they still face significant challenges in complex reasoning and planning. For example, even with carefully designed prompts and prior information explicitly provided, GPT-4o achieves only a 7% Final Pass Rate on the TravelPlanner dataset in the sole-planning mode. Similarly, even in the thinking mode, Qwen3-8B-Instruct and DeepSeek-R1-671B, only achieve Final Pass Rates of 5.9% and 40%, respectively. Although well-organized Multi-Agent Systems (MAS) can offer improved collective reasoning, they often suffer from high reasoning costs due to multi-round internal interactions, long per-response latency, and difficulties in end-to-end training. To address these challenges, we propose a general and scalable framework called IMAGINE, short for Integrating Multi-Agent System into One Model. This framework not only integrates the reasoning and planning capabilities of MAS into a single, compact model, but also significantly surpass the capabilities of the MAS through a simple end-to-end training. Through this pipeline, a single small-scale model is not only able to acquire the structured reasoning and planning capabilities of a well-organized MAS but can also significantly outperform it. Experimental results demonstrate that, when using Qwen3-8B-Instruct as the base model and training it with our method, the model achieves an 82.7% Final Pass Rate on the TravelPlanner benchmark, far exceeding the 40% of DeepSeek-R1-671B, while maintaining a much smaller model size.
>
---
#### [new 143] Agentic Design of Compositional Machines
- **分类: cs.AI; cs.CL; cs.CV; cs.GR; cs.LG**

- **简介: 该论文研究大语言模型（LLM）在组合式机器设计中的应用，旨在探索其创造能力。基于Besiege游戏构建测试平台BesiegeField，评估LLM的空间推理、组装策略等能力，并通过强化学习提升性能，推动语言模型在物理设计任务中的发展。**

- **链接: [http://arxiv.org/pdf/2510.14980v1](http://arxiv.org/pdf/2510.14980v1)**

> **作者:** Wenqian Zhang; Weiyang Liu; Zhen Liu
>
> **备注:** 75 pages, 31 figures, Project Page: https://besiegefield.github.io
>
> **摘要:** The design of complex machines stands as both a marker of human intelligence and a foundation of engineering practice. Given recent advances in large language models (LLMs), we ask whether they, too, can learn to create. We approach this question through the lens of compositional machine design: a task in which machines are assembled from standardized components to meet functional demands like locomotion or manipulation in a simulated physical environment. To support this investigation, we introduce BesiegeField, a testbed built on the machine-building game Besiege, which enables part-based construction, physical simulation and reward-driven evaluation. Using BesiegeField, we benchmark state-of-the-art LLMs with agentic workflows and identify key capabilities required for success, including spatial reasoning, strategic assembly, and instruction-following. As current open-source models fall short, we explore reinforcement learning (RL) as a path to improvement: we curate a cold-start dataset, conduct RL finetuning experiments, and highlight open challenges at the intersection of language, machine design, and physical reasoning.
>
---
#### [new 144] Agentic Entropy-Balanced Policy Optimization
- **分类: cs.LG; cs.AI; cs.CL; cs.IR**

- **简介: 该论文研究Web智能体的长周期工具调用任务，针对现有强化学习因过度依赖熵导致训练崩溃的问题，提出AEPO算法，通过动态平衡rollout与策略更新中的熵，提升采样多样性并稳定训练，显著优于主流方法。**

- **链接: [http://arxiv.org/pdf/2510.14545v1](http://arxiv.org/pdf/2510.14545v1)**

> **作者:** Guanting Dong; Licheng Bao; Zhongyuan Wang; Kangzhi Zhao; Xiaoxi Li; Jiajie Jin; Jinghan Yang; Hangyu Mao; Fuzheng Zhang; Kun Gai; Guorui Zhou; Yutao Zhu; Ji-Rong Wen; Zhicheng Dou
>
> **备注:** Working in progress
>
> **摘要:** Recently, Agentic Reinforcement Learning (Agentic RL) has made significant progress in incentivizing the multi-turn, long-horizon tool-use capabilities of web agents. While mainstream agentic RL algorithms autonomously explore high-uncertainty tool-call steps under the guidance of entropy, excessive reliance on entropy signals can impose further constraints, leading to the training collapse. In this paper, we delve into the challenges caused by entropy and propose the Agentic Entropy-Balanced Policy Optimization (AEPO), an agentic RL algorithm designed to balance entropy in both the rollout and policy update phases. AEPO comprises two core components: (1) a dynamic entropy-balanced rollout mechanism that adaptively allocate global and branch sampling budget through entropy pre-monitoring, while imposing a branch penalty on consecutive high-entropy tool-call steps to prevent over-branching issues; and (2) Entropy-Balanced Policy Optimization that inserts a stop-gradient operation into the high-entropy clipping term to preserve and properly rescale gradients on high-entropy tokens, while incorporating entropy-aware advantage estimation to prioritize learning on high-uncertainty tokens. Results across 14 challenging datasets show that AEPO consistently outperforms 7 mainstream RL algorithms. With just 1K RL samples, Qwen3-14B with AEPO achieves impressive results: 47.6% on GAIA, 11.2% on Humanity's Last Exam, and 43.0% on WebWalker for Pass@1; 65.0% on GAIA, 26.0% on Humanity's Last Exam, and 70.0% on WebWalker for Pass@5. Further analysis reveals that AEPO improves rollout sampling diversity while maintaining stable policy entropy, facilitating scalable web agent training.
>
---
#### [new 145] MathCanvas: Intrinsic Visual Chain-of-Thought for Multimodal Mathematical Reasoning
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对多模态数学推理中视觉辅助不足的问题，提出MathCanvas框架，通过预训练与微调赋予大模型内在的视觉链式思维能力，并构建数据集与基准评测其性能。**

- **链接: [http://arxiv.org/pdf/2510.14958v1](http://arxiv.org/pdf/2510.14958v1)**

> **作者:** Weikang Shi; Aldrich Yu; Rongyao Fang; Houxing Ren; Ke Wang; Aojun Zhou; Changyao Tian; Xinyu Fu; Yuxuan Hu; Zimu Lu; Linjiang Huang; Si Liu; Rui Liu; Hongsheng Li
>
> **备注:** Project Page: https://mathcanvas.github.io/
>
> **摘要:** While Large Language Models (LLMs) have excelled in textual reasoning, they struggle with mathematical domains like geometry that intrinsically rely on visual aids. Existing approaches to Visual Chain-of-Thought (VCoT) are often limited by rigid external tools or fail to generate the high-fidelity, strategically-timed diagrams necessary for complex problem-solving. To bridge this gap, we introduce MathCanvas, a comprehensive framework designed to endow unified Large Multimodal Models (LMMs) with intrinsic VCoT capabilities for mathematics. Our approach consists of two phases. First, a Visual Manipulation stage pre-trains the model on a novel 15.2M-pair corpus, comprising 10M caption-to-diagram pairs (MathCanvas-Imagen) and 5.2M step-by-step editing trajectories (MathCanvas-Edit), to master diagram generation and editing. Second, a Strategic Visual-Aided Reasoning stage fine-tunes the model on MathCanvas-Instruct, a new 219K-example dataset of interleaved visual-textual reasoning paths, teaching it when and how to leverage visual aids. To facilitate rigorous evaluation, we introduce MathCanvas-Bench, a challenging benchmark with 3K problems that require models to produce interleaved visual-textual solutions. Our model, BAGEL-Canvas, trained under this framework, achieves an 86% relative improvement over strong LMM baselines on MathCanvas-Bench, demonstrating excellent generalization to other public math benchmarks. Our work provides a complete toolkit-framework, datasets, and benchmark-to unlock complex, human-like visual-aided reasoning in LMMs. Project Page: https://mathcanvas.github.io/
>
---
#### [new 146] You May Speak Freely: Improving the Fine-Grained Visual Recognition Capabilities of Multimodal Large Language Models with Answer Extraction
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对细粒度视觉分类中多选项问答的评估难题，提出nlg2choice方法，通过开放提问与受限解码两阶段机制，提升多模态大模型在分类与检索任务上的性能，尤其适用于高数量级、高相似度选项场景。**

- **链接: [http://arxiv.org/pdf/2510.14885v1](http://arxiv.org/pdf/2510.14885v1)**

> **作者:** Logan Lawrence; Oindrila Saha; Megan Wei; Chen Sun; Subhransu Maji; Grant Van Horn
>
> **备注:** Accepted to WACV26. 12 pages, 8 tables, 5 figures
>
> **摘要:** Despite the renewed interest in zero-shot visual classification due to the rise of Multimodal Large Language Models (MLLMs), the problem of evaluating free-form responses of auto-regressive models remains a persistent challenge. Most existing works focus on language-only tasks or don't consider Multiple Choice Questions (MCQs) beyond 5-way options, both of which are critical capabilities to solve tasks in Fine-Grained Visual Classification (FGVC) where choice counts are in the hundreds to thousands and the choices are highly related. Furthermore, in this highly multi-way MCQ setting it is not clear how to extend LLM choice extraction to retrieval-based problems, where computing probabilities over the choice set is computationally costly. In this work we investigate nlg2choice, a simple two-stage method which first asks the MLLM an open-ended question for the task with minimal constraints, then uses text-only constrained decoding to predict the most likely choice. In retrieval settings, we compute the probability of the constrained response taking that choice with an early stopping method to significantly improve throughput. Our results show improvement over a suite of seven fine-grained visual datasets when evaluating in terms of classification and retrieval, and show that this performance holds over the various ways that users of LLMs can implement tasks in natural language.
>
---
#### [new 147] Detecting Early and Implicit Suicidal Ideation via Longitudinal and Information Environment Signals on Social Media
- **分类: cs.SI; cs.AI; cs.CL; cs.CY; cs.HC**

- **简介: 该论文研究社交媒体中隐性自杀意念的早期检测，提出一种融合用户发帖历史与社交邻域互动信息的计算框架，通过建模信息环境提升预测性能，较仅用个体数据的方法提升15%。**

- **链接: [http://arxiv.org/pdf/2510.14889v1](http://arxiv.org/pdf/2510.14889v1)**

> **作者:** Soorya Ram Shimgekar; Ruining Zhao; Agam Goyal; Violeta J. Rodriguez; Paul A. Bloom; Hari Sundaram; Koustuv Saha
>
> **摘要:** On social media, many individuals experiencing suicidal ideation (SI) do not disclose their distress explicitly. Instead, signs may surface indirectly through everyday posts or peer interactions. Detecting such implicit signals early is critical but remains challenging. We frame early and implicit SI as a forward-looking prediction task and develop a computational framework that models a user's information environment, consisting of both their longitudinal posting histories as well as the discourse of their socially proximal peers. We adopted a composite network centrality measure to identify top neighbors of a user, and temporally aligned the user's and neighbors' interactions -- integrating the multi-layered signals in a fine-tuned DeBERTa-v3 model. In a Reddit study of 1,000 (500 Case and 500 Control) users, our approach improves early and implicit SI detection by 15% over individual-only baselines. These findings highlight that peer interactions offer valuable predictive signals and carry broader implications for designing early detection systems that capture indirect as well as masked expressions of risk in online environments.
>
---
#### [new 148] Generating Fair Consensus Statements with Social Choice on Token-Level MDPs
- **分类: cs.AI; cs.CL; cs.GT**

- **简介: 该论文研究共识语句生成任务，旨在解决现有方法缺乏公平性保障的问题。作者将生成过程建模为基于社会选择理论的token级多目标MDP，提出两种方法：一种确保事前核心稳定性的随机策略，另一种通过搜索最大化平等福利，实验证明后者能提升最差情况下的代理对齐。**

- **链接: [http://arxiv.org/pdf/2510.14106v1](http://arxiv.org/pdf/2510.14106v1)**

> **作者:** Carter Blair; Kate Larson
>
> **摘要:** Current frameworks for consensus statement generation with large language models lack the inherent structure needed to provide provable fairness guarantees when aggregating diverse free-form opinions. We model the task as a multi-objective, token-level Markov Decision Process (MDP), where each objective corresponds to an agent's preference. Token-level rewards for each agent are derived from their policy (e.g., a personalized language model). This approach utilizes the finding that such policies implicitly define optimal Q-functions, providing a principled way to quantify rewards at each generation step without a value function (Rafailov et al., 2024). This MDP formulation creates a formal structure amenable to analysis using principles from social choice theory. We propose two approaches grounded in social choice theory. First, we propose a stochastic generation policy guaranteed to be in the ex-ante core, extending core stability concepts from voting theory to text generation. This policy is derived from an underlying distribution over complete statements that maximizes proportional fairness (Nash Welfare). Second, for generating a single statement, we target the maximization of egalitarian welfare using search algorithms within the MDP framework. Empirically, experiments using language models to instantiate agent policies show that search guided by the egalitarian objective generates consensus statements with improved worst-case agent alignment compared to baseline methods, including the Habermas Machine (Tessler et al., 2024).
>
---
#### [new 149] LTR-ICD: A Learning-to-Rank Approach for Automatic ICD Coding
- **分类: cs.LG; cs.CL; cs.IR**

- **简介: 该论文研究自动ICD编码任务，旨在解决现有方法忽略诊断代码顺序的问题。作者提出LTR-ICD模型，首次从学习排序角度同时优化分类与代码优先级排序，显著提升高优先级代码的识别准确率和整体性能。**

- **链接: [http://arxiv.org/pdf/2510.13922v1](http://arxiv.org/pdf/2510.13922v1)**

> **作者:** Mohammad Mansoori; Amira Soliman; Farzaneh Etminani
>
> **摘要:** Clinical notes contain unstructured text provided by clinicians during patient encounters. These notes are usually accompanied by a sequence of diagnostic codes following the International Classification of Diseases (ICD). Correctly assigning and ordering ICD codes are essential for medical diagnosis and reimbursement. However, automating this task remains challenging. State-of-the-art methods treated this problem as a classification task, leading to ignoring the order of ICD codes that is essential for different purposes. In this work, as a first attempt, we approach this task from a retrieval system perspective to consider the order of codes, thus formulating this problem as a classification and ranking task. Our results and analysis show that the proposed framework has a superior ability to identify high-priority codes compared to other methods. For instance, our model accuracy in correctly ranking primary diagnosis codes is 47%, compared to 20% for the state-of-the-art classifier. Additionally, in terms of classification metrics, the proposed model achieves a micro- and macro-F1 scores of 0.6065 and 0.2904, respectively, surpassing the previous best model with scores of 0.597 and 0.2660.
>
---
#### [new 150] Stable but Miscalibrated: A Kantian View on Overconfidence from Filters to Large Language Models
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文探讨推理系统的过自信问题，提出基于康德哲学的反馈稳定性视角，构建H-Risk指标衡量表观稳定与认知不稳的差距，并在语言模型中验证其与校准误差和幻觉的关联，旨在建立可控的理性边界。**

- **链接: [http://arxiv.org/pdf/2510.14925v1](http://arxiv.org/pdf/2510.14925v1)**

> **作者:** Akira Okutomi
>
> **备注:** 19 pages, 2 figures, preliminary version
>
> **摘要:** We reinterpret Kant's Critique of Pure Reason as a theory of feedback stability, viewing reason as a regulator that keeps inference within the bounds of possible experience. We formalize this intuition via a composite instability index (H-Risk) combining spectral margin, conditioning, temporal sensitivity, and innovation amplification. In linear-Gaussian simulations, higher H-Risk predicts overconfident errors even under formal stability, revealing a gap between nominal and epistemic stability. Extending to large language models (LLMs), we find that fragile internal dynamics correlate with miscalibration and hallucination, while critique-style prompts show mixed effects on calibration and hallucination. These results suggest a structural bridge between Kantian self-limitation and feedback control, offering a principled lens for diagnosing -- and selectively reducing -- overconfidence in reasoning systems. This is a preliminary version; supplementary experiments and broader replication will be reported in a future revision.
>
---
#### [new 151] Joint Modeling of Big Five and HEXACO for Multimodal Apparent Personality-trait Recognition
- **分类: cs.CV; cs.CL; cs.MM**

- **简介: 该论文研究多模态表观人格识别，旨在同时建模大五人格与较新的人格模型HEXACO，尤其关注诚实-谦逊特质。通过联合优化二者识别，揭示其关系，提升对人类行为的理解。**

- **链接: [http://arxiv.org/pdf/2510.14203v1](http://arxiv.org/pdf/2510.14203v1)**

> **作者:** Ryo Masumura; Shota Orihashi; Mana Ihori; Tomohiro Tanaka; Naoki Makishima; Taiga Yamane; Naotaka Kawata; Satoshi Suzuki; Taichi Katayama
>
> **备注:** Accepted at APSIPA ASC 2025
>
> **摘要:** This paper proposes a joint modeling method of the Big Five, which has long been studied, and HEXACO, which has recently attracted attention in psychology, for automatically recognizing apparent personality traits from multimodal human behavior. Most previous studies have used the Big Five for multimodal apparent personality-trait recognition. However, no study has focused on apparent HEXACO which can evaluate an Honesty-Humility trait related to displaced aggression and vengefulness, social-dominance orientation, etc. In addition, the relationships between the Big Five and HEXACO when modeled by machine learning have not been clarified. We expect awareness of multimodal human behavior to improve by considering these relationships. The key advance of our proposed method is to optimize jointly recognizing the Big Five and HEXACO. Experiments using a self-introduction video dataset demonstrate that the proposed method can effectively recognize the Big Five and HEXACO.
>
---
#### [new 152] Generative AI in Heritage Practice: Improving the Accessibility of Heritage Guidance
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文探讨生成式AI在遗产保护领域的应用，旨在提升公众指南的可及性。研究开发了专用聊天机器人HAZEL，通过微调大模型辅助编写遗产指导文本，并与ChatGPT对比，验证其在文本修订任务中的有效性与局限性。**

- **链接: [http://arxiv.org/pdf/2510.13811v1](http://arxiv.org/pdf/2510.13811v1)**

> **作者:** Jessica Witte; Edmund Lee; Lisa Brausem; Verity Shillabeer; Chiara Bonacchi
>
> **备注:** 21 pages
>
> **摘要:** This paper discusses the potential for integrating Generative Artificial Intelligence (GenAI) into professional heritage practice with the aim of enhancing the accessibility of public-facing guidance documents. We developed HAZEL, a GenAI chatbot fine-tuned to assist with revising written guidance relating to heritage conservation and interpretation. Using quantitative assessments, we compare HAZEL's performance to that of ChatGPT (GPT-4) in a series of tasks related to the guidance writing process. The results of this comparison indicate a slightly better performance of HAZEL over ChatGPT, suggesting that the GenAI chatbot is more effective once the underlying large language model (LLM) has been fine-tuned. However, we also note significant limitations, particularly in areas requiring cultural sensitivity and more advanced technical expertise. These findings suggest that, while GenAI cannot replace human heritage professionals in technical authoring tasks, its potential to automate and expedite certain aspects of guidance writing could offer valuable benefits to heritage organisations, especially in resource-constrained contexts.
>
---
#### [new 153] Benchmarking Multimodal Large Language Models for Face Recognition
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文聚焦多模态大模型在人脸识别中的性能评估，旨在填补开源模型与专用模型间的评测空白。作者在多个标准人脸数据集上系统测试现有MLLM的零样本识别能力，发现其语义表征强但精度不及专用模型，为后续研究提供基准和方向。**

- **链接: [http://arxiv.org/pdf/2510.14866v1](http://arxiv.org/pdf/2510.14866v1)**

> **作者:** Hatef Otroshi Shahreza; Sébastien Marcel
>
> **摘要:** Multimodal large language models (MLLMs) have achieved remarkable performance across diverse vision-and-language tasks. However, their potential in face recognition remains underexplored. In particular, the performance of open-source MLLMs needs to be evaluated and compared with existing face recognition models on standard benchmarks with similar protocol. In this work, we present a systematic benchmark of state-of-the-art MLLMs for face recognition on several face recognition datasets, including LFW, CALFW, CPLFW, CFP, AgeDB and RFW. Experimental results reveal that while MLLMs capture rich semantic cues useful for face-related tasks, they lag behind specialized models in high-precision recognition scenarios in zero-shot applications. This benchmark provides a foundation for advancing MLLM-based face recognition, offering insights for the design of next-generation models with higher accuracy and generalization. The source code of our benchmark is publicly available in the project page.
>
---
#### [new 154] Just-In-Time Objectives: A General Approach for Specialized AI Interactions
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文提出“即时目标”方法，通过观察用户行为自动推断其当前任务目标，并引导大模型针对性优化输出。解决了通用模型响应平淡的问题，实现了个性化、专业化的人机交互，在多实验中显著优于常规LLM。**

- **链接: [http://arxiv.org/pdf/2510.14591v1](http://arxiv.org/pdf/2510.14591v1)**

> **作者:** Michelle S. Lam; Omar Shaikh; Hallie Xu; Alice Guo; Diyi Yang; Jeffrey Heer; James A. Landay; Michael S. Bernstein
>
> **摘要:** Large language models promise a broad set of functions, but when not given a specific objective, they default to milquetoast results such as drafting emails littered with cliches. We demonstrate that inferring the user's in-the-moment objective, then rapidly optimizing for that singular objective, enables LLMs to produce tools, interfaces, and responses that are more responsive and desired. We contribute an architecture for automatically inducing just-in-time objectives by passively observing user behavior, then steering downstream AI systems through generation and evaluation against this objective. Inducing just-in-time objectives (e.g., "Clarify the abstract's research contribution") enables automatic generation of tools, e.g., those that critique a draft based on relevant HCI methodologies, anticipate related researchers' reactions, or surface ambiguous terminology. In a series of experiments (N=14, N=205) on participants' own tasks, JIT objectives enable LLM outputs that achieve 66-86% win rates over typical LLMs, and in-person use sessions (N=17) confirm that JIT objectives produce specialized tools unique to each participant.
>
---
#### [new 155] Terrarium: Revisiting the Blackboard for Multi-Agent Safety, Privacy, and Security Studies
- **分类: cs.AI; cs.CL; cs.CR; I.2.7; I.2.11**

- **简介: 该论文提出Terrarium框架，旨在研究基于大语言模型的多智能体系统在安全、隐私和安全方面的风险。通过重构黑板架构，构建可配置测试平台，分析多种攻击向量，并验证其在协作场景中的灵活性与实用性。**

- **链接: [http://arxiv.org/pdf/2510.14312v1](http://arxiv.org/pdf/2510.14312v1)**

> **作者:** Mason Nakamura; Abhinav Kumar; Saaduddin Mahmud; Sahar Abdelnabi; Shlomo Zilberstein; Eugene Bagdasarian
>
> **摘要:** A multi-agent system (MAS) powered by large language models (LLMs) can automate tedious user tasks such as meeting scheduling that requires inter-agent collaboration. LLMs enable nuanced protocols that account for unstructured private data, user constraints, and preferences. However, this design introduces new risks, including misalignment and attacks by malicious parties that compromise agents or steal user data. In this paper, we propose the Terrarium framework for fine-grained study on safety, privacy, and security in LLM-based MAS. We repurpose the blackboard design, an early approach in multi-agent systems, to create a modular, configurable testbed for multi-agent collaboration. We identify key attack vectors such as misalignment, malicious agents, compromised communication, and data poisoning. We implement three collaborative MAS scenarios with four representative attacks to demonstrate the framework's flexibility. By providing tools to rapidly prototype, evaluate, and iterate on defenses and designs, Terrarium aims to accelerate progress toward trustworthy multi-agent systems.
>
---
#### [new 156] Do Slides Help? Multi-modal Context for Automatic Transcription of Conference Talks
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究会议演讲的自动转录任务，旨在利用幻灯片等多模态信息提升语音识别准确率。针对缺乏带幻灯片数据集的问题，提出数据增强方法，构建基准并融合视觉上下文，显著降低词错误率，尤其在领域术语上表现突出。**

- **链接: [http://arxiv.org/pdf/2510.13979v1](http://arxiv.org/pdf/2510.13979v1)**

> **作者:** Supriti Sinhamahapatra; Jan Niehues
>
> **摘要:** State-of-the-art (SOTA) Automatic Speech Recognition (ASR) systems primarily rely on acoustic information while disregarding additional multi-modal context. However, visual information are essential in disambiguation and adaptation. While most work focus on speaker images to handle noise conditions, this work also focuses on integrating presentation slides for the use cases of scientific presentation. In a first step, we create a benchmark for multi-modal presentation including an automatic analysis of transcribing domain-specific terminology. Next, we explore methods for augmenting speech models with multi-modal information. We mitigate the lack of datasets with accompanying slides by a suitable approach of data augmentation. Finally, we train a model using the augmented dataset, resulting in a relative reduction in word error rate of approximately 34%, across all words and 35%, for domain-specific terms compared to the baseline model.
>
---
## 更新

#### [replaced 001] Beyond Monolingual Assumptions: A Survey of Code-Switched NLP in the Era of Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.07037v3](http://arxiv.org/pdf/2510.07037v3)**

> **作者:** Rajvee Sheth; Samridhi Raj Sinha; Mahavir Patil; Himanshu Beniwal; Mayank Singh
>
> **摘要:** Code-switching (CSW), the alternation of languages and scripts within a single utterance, remains a fundamental challenge for multilingual NLP, even amidst the rapid advances of large language models (LLMs). Most LLMs still struggle with mixed-language inputs, limited CSW datasets, and evaluation biases, hindering deployment in multilingual societies. This survey provides the first comprehensive analysis of CSW-aware LLM research, reviewing 308 studies spanning five research areas, 12 NLP tasks, 30+ datasets, and 80+ languages. We classify recent advances by architecture, training strategy, and evaluation methodology, outlining how LLMs have reshaped CSW modeling and what challenges persist. The paper concludes with a roadmap emphasizing the need for inclusive datasets, fair evaluation, and linguistically grounded models to achieve truly multilingual intelligence. A curated collection of all resources is maintained at https://github.com/lingo-iitgn/awesome-code-mixing/.
>
---
#### [replaced 002] Probabilistic Reasoning with LLMs for k-anonymity Estimation
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.09674v5](http://arxiv.org/pdf/2503.09674v5)**

> **作者:** Jonathan Zheng; Sauvik Das; Alan Ritter; Wei Xu
>
> **备注:** 10 pages, Accepted to NeurIPS 2025
>
> **摘要:** Probabilistic reasoning is a key aspect of both human and artificial intelligence that allows for handling uncertainty and ambiguity in decision-making. In this paper, we introduce a new numerical reasoning task under uncertainty for large language models, focusing on estimating the privacy risk of user-generated documents containing privacy-sensitive information. We propose BRANCH, a new LLM methodology that estimates the k-privacy value of a text-the size of the population matching the given information. BRANCH factorizes a joint probability distribution of personal information as random variables. The probability of each factor in a population is estimated separately using a Bayesian network and combined to compute the final k-value. Our experiments show that this method successfully estimates the k-value 73% of the time, a 13% increase compared to o3-mini with chain-of-thought reasoning. We also find that LLM uncertainty is a good indicator for accuracy, as high-variance predictions are 37.47% less accurate on average.
>
---
#### [replaced 003] A Comprehensive Survey on Benchmarks and Solutions in Software Engineering of LLM-Empowered Agentic System
- **分类: cs.SE; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.09721v2](http://arxiv.org/pdf/2510.09721v2)**

> **作者:** Jiale Guo; Suizhi Huang; Mei Li; Dong Huang; Xingsheng Chen; Regina Zhang; Zhijiang Guo; Han Yu; Siu-Ming Yiu; Christian Jensen; Pietro Lio; Kwok-Yan Lam
>
> **备注:** 21 pages
>
> **摘要:** The integration of Large Language Models (LLMs) into software engineering has driven a transition from traditional rule-based systems to autonomous agentic systems capable of solving complex problems. However, systematic progress is hindered by a lack of comprehensive understanding of how benchmarks and solutions interconnect. This survey addresses this gap by providing the first holistic analysis of LLM-powered software engineering, offering insights into evaluation methodologies and solution paradigms. We review over 150 recent papers and propose a taxonomy along two key dimensions: (1) Solutions, categorized into prompt-based, fine-tuning-based, and agent-based paradigms, and (2) Benchmarks, including tasks such as code generation, translation, and repair. Our analysis highlights the evolution from simple prompt engineering to sophisticated agentic systems incorporating capabilities like planning, reasoning, memory mechanisms, and tool augmentation. To contextualize this progress, we present a unified pipeline illustrating the workflow from task specification to deliverables, detailing how different solution paradigms address various complexity levels. Unlike prior surveys that focus narrowly on specific aspects, this work connects 50+ benchmarks to their corresponding solution strategies, enabling researchers to identify optimal approaches for diverse evaluation criteria. We also identify critical research gaps and propose future directions, including multi-agent collaboration, self-evolving systems, and formal verification integration. This survey serves as a foundational guide for advancing LLM-driven software engineering. We maintain a GitHub repository that continuously updates the reviewed and related papers at https://github.com/lisaGuojl/LLM-Agent-SE-Survey.
>
---
#### [replaced 004] Natural Language Processing RELIES on Linguistics
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2405.05966v5](http://arxiv.org/pdf/2405.05966v5)**

> **作者:** Juri Opitz; Shira Wein; Nathan Schneider
>
> **备注:** Appeared in Computational Linguistics. Journal version at https://doi.org/10.1162/coli_a_00560
>
> **摘要:** Large Language Models (LLMs) have become capable of generating highly fluent text in certain languages, without modules specially designed to capture grammar or semantic coherence. What does this mean for the future of linguistic expertise in NLP? We highlight several aspects in which NLP (still) relies on linguistics, or where linguistic thinking can illuminate new directions. We argue our case around the acronym RELIES that encapsulates six major facets where linguistics contributes to NLP: Resources, Evaluation, Low-resource settings, Interpretability, Explanation, and the Study of language. This list is not exhaustive, nor is linguistics the main point of reference for every effort under these themes; but at a macro level, these facets highlight the enduring importance of studying machine systems vis-\`a-vis systems of human language.
>
---
#### [replaced 005] KScope: A Framework for Characterizing the Knowledge Status of Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.07458v2](http://arxiv.org/pdf/2506.07458v2)**

> **作者:** Yuxin Xiao; Shan Chen; Jack Gallifant; Danielle Bitterman; Thomas Hartvigsen; Marzyeh Ghassemi
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Characterizing a large language model's (LLM's) knowledge of a given question is challenging. As a result, prior work has primarily examined LLM behavior under knowledge conflicts, where the model's internal parametric memory contradicts information in the external context. However, this does not fully reflect how well the model knows the answer to the question. In this paper, we first introduce a taxonomy of five knowledge statuses based on the consistency and correctness of LLM knowledge modes. We then propose KScope, a hierarchical framework of statistical tests that progressively refines hypotheses about knowledge modes and characterizes LLM knowledge into one of these five statuses. We apply KScope to nine LLMs across four datasets and systematically establish: (1) Supporting context narrows knowledge gaps across models. (2) Context features related to difficulty, relevance, and familiarity drive successful knowledge updates. (3) LLMs exhibit similar feature preferences when partially correct or conflicted, but diverge sharply when consistently wrong. (4) Context summarization constrained by our feature analysis, together with enhanced credibility, further improves update effectiveness and generalizes across LLMs.
>
---
#### [replaced 006] Sri Lanka Document Datasets: A Large-Scale, Multilingual Resource for Law, News, and Policy
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.04124v3](http://arxiv.org/pdf/2510.04124v3)**

> **作者:** Nuwan I. Senaratna
>
> **备注:** 4 pages. 230,091 documents (57.7 GB) across 24 datasets in Sinhala, Tamil, and English. Last updated on 2025-10-16-0818
>
> **摘要:** We present a collection of open, machine-readable document datasets covering parliamentary proceedings, legal judgments, government publications, news, and tourism statistics from Sri Lanka. The collection currently comprises of 230,091 documents (57.7 GB) across 24 datasets in Sinhala, Tamil, and English. The datasets are updated daily and mirrored on GitHub and Hugging Face. These resources aim to support research in computational linguistics, legal analytics, socio-political studies, and multilingual natural language processing. We describe the data sources, collection pipeline, formats, and potential use cases, while discussing licensing and ethical considerations. This manuscript is at version v2025-10-16-0818.
>
---
#### [replaced 007] AAVENUE: Detecting LLM Biases on NLU Tasks in AAVE via a Novel Benchmark
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2408.14845v3](http://arxiv.org/pdf/2408.14845v3)**

> **作者:** Abhay Gupta; Philip Meng; Ece Yurtseven; Sean O'Brien; Kevin Zhu
>
> **备注:** Published at NLP4PI @ EMNLP 2024
>
> **摘要:** Detecting biases in natural language understanding (NLU) for African American Vernacular English (AAVE) is crucial to developing inclusive natural language processing (NLP) systems. To address dialect-induced performance discrepancies, we introduce AAVENUE ({AAVE} {N}atural Language {U}nderstanding {E}valuation), a benchmark for evaluating large language model (LLM) performance on NLU tasks in AAVE and Standard American English (SAE). AAVENUE builds upon and extends existing benchmarks like VALUE, replacing deterministic syntactic and morphological transformations with a more flexible methodology leveraging LLM-based translation with few-shot prompting, improving performance across our evaluation metrics when translating key tasks from the GLUE and SuperGLUE benchmarks. We compare AAVENUE and VALUE translations using five popular LLMs and a comprehensive set of metrics including fluency, BARTScore, quality, coherence, and understandability. Additionally, we recruit fluent AAVE speakers to validate our translations for authenticity. Our evaluations reveal that LLMs consistently perform better on SAE tasks than AAVE-translated versions, underscoring inherent biases and highlighting the need for more inclusive NLP models. We have open-sourced our source code on GitHub and created a website to showcase our work at https://aavenuee.github.io.
>
---
#### [replaced 008] Machine Unlearning Meets Adversarial Robustness via Constrained Interventions on LLMs
- **分类: cs.LG; cs.CL; cs.CR; cs.CY; math.OC**

- **链接: [http://arxiv.org/pdf/2510.03567v3](http://arxiv.org/pdf/2510.03567v3)**

> **作者:** Fatmazohra Rezkellah; Ramzi Dakhmouche
>
> **摘要:** With the increasing adoption of Large Language Models (LLMs), more customization is needed to ensure privacy-preserving and safe generation. We address this objective from two critical aspects: unlearning of sensitive information and robustness to jail-breaking attacks. We investigate various constrained optimization formulations that address both aspects in a \emph{unified manner}, by finding the smallest possible interventions on LLM weights that either make a given vocabulary set unreachable or embed the LLM with robustness to tailored attacks by shifting part of the weights to a \emph{safer} region. Beyond unifying two key properties, this approach contrasts with previous work in that it doesn't require an oracle classifier that is typically not available or represents a computational overhead. Surprisingly, we find that the simplest point-wise constraint-based intervention we propose leads to better performance than max-min interventions, while having a lower computational cost. Comparison against state-of-the-art defense methods demonstrates superior performance of the proposed approach.
>
---
#### [replaced 009] Causal Language Control in Multilingual Transformers via Sparse Feature Steering
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.13410v2](http://arxiv.org/pdf/2507.13410v2)**

> **作者:** Cheng-Ting Chou; George Liu; Jessica Sun; Cole Blondin; Kevin Zhu; Vasu Sharma; Sean O'Brien
>
> **摘要:** Deterministically controlling the target generation language of large multilingual language models (LLMs) remains a fundamental challenge, particularly in zero-shot settings where neither explicit language prompts nor fine-tuning are available. In this work, we investigate whether sparse autoencoder (SAE) features, previously shown to correlate with interpretable model behaviors, can be leveraged to steer the generated language of LLMs during inference. Leveraging pretrained SAEs on the residual streams of Gemma-2B and Gemma-9B, we identify features whose activations differ most significantly between English and four target languages: Chinese, Japanese, Spanish, and French. By modifying just a single SAE feature at one transformer layer, we achieve controlled language shifts with up to 90\% success, as measured by FastText language classification, while preserving semantic fidelity according to LaBSE (Language-Agnostic BERT Sentence Embedding) similarity. Our analysis reveals that language steering is most effective in mid-to-late transformer layers and is amplified by specific attention heads disproportionately associated with language-sensitive SAE features. These results demonstrate the promise of sparse feature steering as a lightweight and interpretable mechanism for controllable multilingual generation.
>
---
#### [replaced 010] Moto: Latent Motion Token as the Bridging Language for Learning Robot Manipulation from Videos
- **分类: cs.RO; cs.AI; cs.CL; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.04445v4](http://arxiv.org/pdf/2412.04445v4)**

> **作者:** Yi Chen; Yuying Ge; Weiliang Tang; Yizhuo Li; Yixiao Ge; Mingyu Ding; Ying Shan; Xihui Liu
>
> **备注:** ICCV 2025. Project page: https://chenyi99.github.io/moto/
>
> **摘要:** Recent developments in Large Language Models pre-trained on extensive corpora have shown significant success in various natural language processing tasks with minimal fine-tuning. This success offers new promise for robotics, which has long been constrained by the high cost of action-labeled data. We ask: given the abundant video data containing interaction-related knowledge available as a rich "corpus", can a similar generative pre-training approach be effectively applied to enhance robot learning? The key challenge is to identify an effective representation for autoregressive pre-training that benefits robot manipulation tasks. Inspired by the way humans learn new skills through observing dynamic environments, we propose that effective robotic learning should emphasize motion-related knowledge, which is closely tied to low-level actions and is hardware-agnostic, facilitating the transfer of learned motions to actual robot actions. To this end, we introduce Moto, which converts video content into latent Motion Token sequences by a Latent Motion Tokenizer, learning a bridging "language" of motion from videos in an unsupervised manner. We pre-train Moto-GPT through motion token autoregression, enabling it to capture diverse visual motion knowledge. After pre-training, Moto-GPT demonstrates the promising ability to produce semantically interpretable motion tokens, predict plausible motion trajectories, and assess trajectory rationality through output likelihood. To transfer learned motion priors to real robot actions, we implement a co-fine-tuning strategy that seamlessly bridges latent motion token prediction and real robot control. Extensive experiments show that the fine-tuned Moto-GPT exhibits superior robustness and efficiency on robot manipulation benchmarks, underscoring its effectiveness in transferring knowledge from video data to downstream visual manipulation tasks.
>
---
#### [replaced 011] InfoDet: A Dataset for Infographic Element Detection
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.17473v5](http://arxiv.org/pdf/2505.17473v5)**

> **作者:** Jiangning Zhu; Yuxing Zhou; Zheng Wang; Juntao Yao; Yima Gu; Yuhui Yuan; Shixia Liu
>
> **备注:** Submitted to ICLR 2026
>
> **摘要:** Given the central role of charts in scientific, business, and communication contexts, enhancing the chart understanding capabilities of vision-language models (VLMs) has become increasingly critical. A key limitation of existing VLMs lies in their inaccurate visual grounding of infographic elements, including charts and human-recognizable objects (HROs) such as icons and images. However, chart understanding often requires identifying relevant elements and reasoning over them. To address this limitation, we introduce InfoDet, a dataset designed to support the development of accurate object detection models for charts and HROs in infographics. It contains 11,264 real and 90,000 synthetic infographics, with over 14 million bounding box annotations. These annotations are created by combining the model-in-the-loop and programmatic methods. We demonstrate the usefulness of InfoDet through three applications: 1) constructing a Thinking-with-Boxes scheme to boost the chart understanding performance of VLMs, 2) comparing existing object detection models, and 3) applying the developed detection model to document layout and UI element detection.
>
---
#### [replaced 012] CoT-Evo: Evolutionary Distillation of Chain-of-Thought for Scientific Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.13166v2](http://arxiv.org/pdf/2510.13166v2)**

> **作者:** Kehua Feng; Keyan Ding; Zhihui Zhu; Lei Liang; Qiang Zhang; Huajun Chen
>
> **备注:** 28 pages, 3 figures
>
> **摘要:** While chain-of-thought (CoT) distillation from advanced large language models (LLMs) has proven effective in general reasoning tasks, it struggles in scientific domains where even advanced models often produce incorrect or superficial reasoning due to high complexity and specialized knowledge requirements. Directly distilling from such flawed outputs results in low-quality training data and limits the performance of smaller student models. To overcome this, we propose CoT-Evo, an evolutionary CoT distillation framework. It begins by constructing a diverse pool of reasoning trajectories from multiple LLM thinkers, enriches them with automatically retrieved domain knowledge, and iteratively refines the trajectories using novelty-driven selection, reflective recombination and mutation. The refinement is guided by a fitness function that evaluates answer correctness, coherence, and effective knowledge utilization. This results in a high-quality CoT dataset tailored for scientific reasoning. We employ this evolved dataset to fine-tune a compact model, which achieves state-of-the-art performance on scientific reasoning benchmarks. Our work establishes a scalable approach to synthesizing high-fidelity scientific reasoning data from diverse and fallible LLMs.
>
---
#### [replaced 013] HALF: Harm-Aware LLM Fairness Evaluation Aligned with Deployment
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.12217v2](http://arxiv.org/pdf/2510.12217v2)**

> **作者:** Ali Mekky; Omar El Herraoui; Preslav Nakov; Yuxia Wang
>
> **摘要:** Large language models (LLMs) are increasingly deployed across high-impact domains, from clinical decision support and legal analysis to hiring and education, making fairness and bias evaluation before deployment critical. However, existing evaluations lack grounding in real-world scenarios and do not account for differences in harm severity, e.g., a biased decision in surgery should not be weighed the same as a stylistic bias in text summarization. To address this gap, we introduce HALF (Harm-Aware LLM Fairness), a deployment-aligned framework that assesses model bias in realistic applications and weighs the outcomes by harm severity. HALF organizes nine application domains into three tiers (Severe, Moderate, Mild) using a five-stage pipeline. Our evaluation results across eight LLMs show that (1) LLMs are not consistently fair across domains, (2) model size or performance do not guarantee fairness, and (3) reasoning models perform better in medical decision support but worse in education. We conclude that HALF exposes a clear gap between previous benchmarking success and deployment readiness.
>
---
#### [replaced 014] Women, Infamous, and Exotic Beings: A Comparative Study of Honorific Usages in Wikipedia and LLMs for Bengali and Hindi
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.03479v4](http://arxiv.org/pdf/2501.03479v4)**

> **作者:** Sourabrata Mukherjee; Atharva Mehta; Sougata Saha; Akhil Arora; Monojit Choudhury
>
> **备注:** Accepted and published at EMNLP 2025 (Main)
>
> **摘要:** The obligatory use of third-person honorifics is a distinctive feature of several South Asian languages, encoding nuanced socio-pragmatic cues such as power, age, gender, fame, and social distance. In this work, (i) We present the first large-scale study of third-person honorific pronoun and verb usage across 10,000 Hindi and Bengali Wikipedia articles with annotations linked to key socio-demographic attributes of the subjects, including gender, age group, fame, and cultural origin. (ii) Our analysis uncovers systematic intra-language regularities but notable cross-linguistic differences: honorifics are more prevalent in Bengali than in Hindi, while non-honorifics dominate while referring to infamous, juvenile, and culturally exotic entities. Notably, in both languages, and more prominently in Hindi, men are more frequently addressed with honorifics than women. (iii) To examine whether large language models (LLMs) internalize similar socio-pragmatic norms, we probe six LLMs using controlled generation and translation tasks over 1,000 culturally balanced entities. We find that LLMs diverge from Wikipedia usage, exhibiting alternative preferences in honorific selection across tasks, languages, and socio-demographic attributes. These discrepancies highlight gaps in the socio-cultural alignment of LLMs and open new directions for studying how LLMs acquire, adapt, or distort social-linguistic norms. Our code and data are publicly available at https://github.com/souro/honorific-wiki-llm
>
---
#### [replaced 015] The simulation of judgment in LLMs
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2502.04426v3](http://arxiv.org/pdf/2502.04426v3)**

> **作者:** Edoardo Loru; Jacopo Nudo; Niccolò Di Marco; Alessandro Santirocchi; Roberto Atzeni; Matteo Cinelli; Vincenzo Cestari; Clelia Rossi-Arnaud; Walter Quattrociocchi
>
> **备注:** Please refer to published version: https://doi.org/10.1073/pnas.2518443122
>
> **摘要:** Large Language Models (LLMs) are increasingly embedded in evaluative processes, from information filtering to assessing and addressing knowledge gaps through explanation and credibility judgments. This raises the need to examine how such evaluations are built, what assumptions they rely on, and how their strategies diverge from those of humans. We benchmark six LLMs against expert ratings--NewsGuard and Media Bias/Fact Check--and against human judgments collected through a controlled experiment. We use news domains purely as a controlled benchmark for evaluative tasks, focusing on the underlying mechanisms rather than on news classification per se. To enable direct comparison, we implement a structured agentic framework in which both models and nonexpert participants follow the same evaluation procedure: selecting criteria, retrieving content, and producing justifications. Despite output alignment, our findings show consistent differences in the observable criteria guiding model evaluations, suggesting that lexical associations and statistical priors could influence evaluations in ways that differ from contextual reasoning. This reliance is associated with systematic effects: political asymmetries and a tendency to confuse linguistic form with epistemic reliability--a dynamic we term epistemia, the illusion of knowledge that emerges when surface plausibility replaces verification. Indeed, delegating judgment to such systems may affect the heuristics underlying evaluative processes, suggesting a shift from normative reasoning toward pattern-based approximation and raising open questions about the role of LLMs in evaluative processes.
>
---
#### [replaced 016] Cross-Question Method Reuse in Large Language Models: From Word-Level Prediction to Rational Logical-Layer Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.05660v2](http://arxiv.org/pdf/2509.05660v2)**

> **作者:** Hong Su
>
> **摘要:** Large language models (LLMs) have been widely applied to assist in finding solutions for diverse questions. Prior work has proposed representing a method as a pair of a question and its corresponding solution, enabling method reuse. However, existing approaches typically require the questions to be highly similar. In this paper, we extend the scope of method reuse to address questions with low similarity or with hidden similarities that are not explicitly observable. For questions that are similar in a general-specific sense (i.e., broader or narrower in scope), we propose to first separate the question and solution, rather than directly feeding the pair to the LLM. The LLM is then guided to adapt the solution to new but related questions, allowing it to focus on solution transfer rather than question recognition. Furthermore, we extend this approach to cases where questions only share partial features or hidden characteristics. This enables cross-question method reuse beyond conventional similarity constraints. Experimental verification shows that our scope-extension approach increases the probability of filtering out reusable solutions, thereby improving the effectiveness of cross-question method reuse.
>
---
#### [replaced 017] CAP: Evaluation of Persuasive and Creative Image Generation
- **分类: cs.CV; cs.CL; cs.GR**

- **链接: [http://arxiv.org/pdf/2412.10426v2](http://arxiv.org/pdf/2412.10426v2)**

> **作者:** Aysan Aghazadeh; Adriana Kovashka
>
> **摘要:** We address the task of advertisement image generation and introduce three evaluation metrics to assess Creativity, prompt Alignment, and Persuasiveness (CAP) in generated advertisement images. Despite recent advancements in Text-to-Image (T2I) generation and their performance in generating high-quality images for explicit descriptions, evaluating these models remains challenging. Existing evaluation methods focus largely on assessing alignment with explicit, detailed descriptions, but evaluating alignment with visually implicit prompts remains an open problem. Additionally, creativity and persuasiveness are essential qualities that enhance the effectiveness of advertisement images, yet are seldom measured. To address this, we propose three novel metrics for evaluating the creativity, alignment, and persuasiveness of generated images. Our findings reveal that current T2I models struggle with creativity, persuasiveness, and alignment when the input text is implicit messages. We further introduce a simple yet effective approach to enhance T2I models' capabilities in producing images that are better aligned, more creative, and more persuasive.
>
---
#### [replaced 018] Why We Build Local Large Language Models: An Observational Analysis from 35 Japanese and Multilingual LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.14471v2](http://arxiv.org/pdf/2412.14471v2)**

> **作者:** Koshiro Saito; Sakae Mizuki; Masanari Ohi; Taishi Nakamura; Taihei Shiotani; Koki Maeda; Youmi Ma; Kakeru Hattori; Kazuki Fujii; Takumi Okamoto; Shigeki Ishida; Hiroya Takamura; Rio Yokota; Naoaki Okazaki
>
> **备注:** Accepted as a spotlight at the 1st workshop on Multilingual and Equitable Language Technologies (MELT), COLM 2025
>
> **摘要:** Why do we build local large language models (LLMs)? What should a local LLM learn from the target language? Which abilities can be transferred from other languages? Do language-specific scaling laws exist? To explore these research questions, we evaluated 35 Japanese, English, and multilingual LLMs on 19 evaluation benchmarks for Japanese and English, taking Japanese as a local language. Adopting an observational approach, we analyzed correlations of benchmark scores, and conducted principal component analysis (PCA) on the scores to derive \textit{ability factors} of local LLMs. We found that training on English text can improve the scores of academic subjects in Japanese (JMMLU). In addition, it is unnecessary to specifically train on Japanese text to enhance abilities for solving Japanese code generation, arithmetic reasoning, commonsense, and reading comprehension tasks. In contrast, training on Japanese text could improve question-answering tasks about Japanese knowledge and English-Japanese translation, which indicates that abilities for solving these two tasks can be regarded as \textit{Japanese abilities} for LLMs. Furthermore, we confirmed that the Japanese abilities scale with the computational budget for Japanese text.
>
---
#### [replaced 019] The Mechanistic Emergence of Symbol Grounding in Language Models
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2510.13796v2](http://arxiv.org/pdf/2510.13796v2)**

> **作者:** Shuyu Wu; Ziqiao Ma; Xiaoxi Luo; Yidong Huang; Josue Torres-Fonseca; Freda Shi; Joyce Chai
>
> **摘要:** Symbol grounding (Harnad, 1990) describes how symbols such as words acquire their meanings by connecting to real-world sensorimotor experiences. Recent work has shown preliminary evidence that grounding may emerge in (vision-)language models trained at scale without using explicit grounding objectives. Yet, the specific loci of this emergence and the mechanisms that drive it remain largely unexplored. To address this problem, we introduce a controlled evaluation framework that systematically traces how symbol grounding arises within the internal computations through mechanistic and causal analysis. Our findings show that grounding concentrates in middle-layer computations and is implemented through the aggregate mechanism, where attention heads aggregate the environmental ground to support the prediction of linguistic forms. This phenomenon replicates in multimodal dialogue and across architectures (Transformers and state-space models), but not in unidirectional LSTMs. Our results provide behavioral and mechanistic evidence that symbol grounding can emerge in language models, with practical implications for predicting and potentially controlling the reliability of generation.
>
---
#### [replaced 020] Iterative LLM-Based Generation and Refinement of Distracting Conditions in Math Word Problems
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.08615v3](http://arxiv.org/pdf/2510.08615v3)**

> **作者:** Kaiqi Yang; Hang Li; Yucheng Chu; Zitao Liu; Mi Tian; Hui Liu
>
> **摘要:** Mathematical reasoning serves as a crucial testbed for the intelligence of large language models (LLMs), and math word problems (MWPs) are a popular type of math problems. Most MWP datasets consist of problems containing only the necessary information, while problems with distracting and excessive conditions are often overlooked. Prior works have tested popular LLMs and found a dramatic performance drop in the presence of distracting conditions. However, datasets of MWPs with distracting conditions are limited, and most suffer from lower levels of difficulty and out-of-context expressions. This makes distracting conditions easy to identify and exclude, thus reducing the credibility of benchmarking on them. Moreover, when adding distracting conditions, the reasoning and answers may also change, requiring intensive labor to check and write the solutions. To address these issues, we design an iterative framework to generate distracting conditions using LLMs. We develop a set of prompts to revise MWPs from different perspectives and cognitive levels, encouraging the generation of distracting conditions as well as suggestions for further revision. Another advantage is the shared solutions between original and revised problems: we explicitly guide the LLMs to generate distracting conditions that do not alter the original solutions, thus avoiding the need to generate new solutions. This framework is efficient and easy to deploy, reducing the overhead of generating MWPs with distracting conditions while maintaining data quality.
>
---
#### [replaced 021] MIO: A Foundation Model on Multimodal Tokens
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2409.17692v4](http://arxiv.org/pdf/2409.17692v4)**

> **作者:** Zekun Wang; King Zhu; Chunpu Xu; Wangchunshu Zhou; Jiaheng Liu; Yibo Zhang; Jiashuo Wang; Ning Shi; Siyu Li; Yizhi Li; Haoran Que; Zhaoxiang Zhang; Yuanxing Zhang; Ge Zhang; Ke Xu; Jie Fu; Wenhao Huang
>
> **备注:** EMNLP 2025 (Oral). Codes and models are available in https://github.com/MIO-Team/MIO
>
> **摘要:** In this paper, we introduce MIO, a novel foundation model built on multimodal tokens, capable of understanding and generating speech, text, images, and videos in an end-to-end, autoregressive manner. While the emergence of large language models (LLMs) and multimodal large language models (MM-LLMs) propels advancements in artificial general intelligence through their versatile capabilities, they still lack true any-to-any understanding and generation. Recently, the release of GPT-4o has showcased the remarkable potential of any-to-any LLMs for complex real-world tasks, enabling omnidirectional input and output across images, speech, and text. However, it is closed-source and does not support the generation of multimodal interleaved sequences. To address this gap, we present MIO, which is trained on a mixture of discrete tokens across four modalities using causal multimodal modeling. MIO undergoes a four-stage training process: (1) alignment pre-training, (2) interleaved pre-training, (3) speech-enhanced pre-training, and (4) comprehensive supervised fine-tuning on diverse textual, visual, and speech tasks. Our experimental results indicate that MIO exhibits competitive, and in some cases superior, performance compared to previous dual-modal baselines, any-to-any model baselines, and even modality-specific baselines. Moreover, MIO demonstrates advanced capabilities inherent to its any-to-any feature, such as interleaved video-text generation, chain-of-visual-thought reasoning, visual guideline generation, instructional image editing, etc.
>
---
#### [replaced 022] Evaluating Arabic Large Language Models: A Survey of Benchmarks, Methods, and Gaps
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.13430v2](http://arxiv.org/pdf/2510.13430v2)**

> **作者:** Ahmed Alzubaidi; Shaikha Alsuwaidi; Basma El Amel Boussaha; Leen AlQadi; Omar Alkaabi; Mohammed Alyafeai; Hamza Alobeidli; Hakim Hacid
>
> **摘要:** This survey provides the first systematic review of Arabic LLM benchmarks, analyzing 40+ evaluation benchmarks across NLP tasks, knowledge domains, cultural understanding, and specialized capabilities. We propose a taxonomy organizing benchmarks into four categories: Knowledge, NLP Tasks, Culture and Dialects, and Target-Specific evaluations. Our analysis reveals significant progress in benchmark diversity while identifying critical gaps: limited temporal evaluation, insufficient multi-turn dialogue assessment, and cultural misalignment in translated datasets. We examine three primary approaches: native collection, translation, and synthetic generation discussing their trade-offs regarding authenticity, scale, and cost. This work serves as a comprehensive reference for Arabic NLP researchers, providing insights into benchmark methodologies, reproducibility standards, and evaluation metrics while offering recommendations for future development.
>
---
#### [replaced 023] Preservation of Language Understanding Capabilities in Speech-aware Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.12171v2](http://arxiv.org/pdf/2509.12171v2)**

> **作者:** Marek Kubis; Paweł Skórzewski; Iwona Christop; Mateusz Czyżnikiewicz; Jakub Kubiak; Łukasz Bondaruk; Marcin Lewandowski
>
> **备注:** 5 pages, 1 figure; benchmark code available at https://github.com/SamsungLabs/C3T
>
> **摘要:** The paper presents C3T (Cross-modal Capabilities Conservation Test), a new benchmark for assessing the performance of speech-aware large language models. The benchmark utilizes textual tasks and a voice cloning text-to-speech model to quantify the extent to which language understanding capabilities are preserved when the model is accessed via speech input. C3T quantifies the fairness of the model for different categories of speakers and its robustness across text and speech modalities.
>
---
#### [replaced 024] Comparing Human and Language Models Sentence Processing Difficulties on Complex Structures
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.07141v2](http://arxiv.org/pdf/2510.07141v2)**

> **作者:** Samuel Joseph Amouyal; Aya Meltzer-Asscher; Jonathan Berant
>
> **备注:** Data and code will be released soon
>
> **摘要:** Large language models (LLMs) that fluently converse with humans are a reality - but do LLMs experience human-like processing difficulties? We systematically compare human and LLM sentence comprehension across seven challenging linguistic structures. We collect sentence comprehension data from humans and five families of state-of-the-art LLMs, varying in size and training procedure in a unified experimental framework. Our results show LLMs overall struggle on the target structures, but especially on garden path (GP) sentences. Indeed, while the strongest models achieve near perfect accuracy on non-GP structures (93.7% for GPT-5), they struggle on GP structures (46.8% for GPT-5). Additionally, when ranking structures based on average performance, rank correlation between humans and models increases with parameter count. For each target structure, we also collect data for their matched baseline without the difficult structure. Comparing performance on the target vs. baseline sentences, the performance gap observed in humans holds for LLMs, with two exceptions: for models that are too weak performance is uniformly low across both sentence types, and for models that are too strong the performance is uniformly high. Together, these reveal convergence and divergence in human and LLM sentence comprehension, offering new insights into the similarity of humans and LLMs.
>
---
#### [replaced 025] Multilinguality Does not Make Sense: Investigating Factors Behind Zero-Shot Transfer in Sense-Aware Tasks
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.24834v2](http://arxiv.org/pdf/2505.24834v2)**

> **作者:** Roksana Goworek; Haim Dubossarsky
>
> **备注:** accepted to EMNLP 2025 Main
>
> **摘要:** Cross-lingual transfer is central to modern NLP, enabling models to perform tasks in languages different from those they were trained on. A common assumption is that training on more languages improves zero-shot transfer. We test this on sense-aware tasks-polysemy and lexical semantic change-and find that multilinguality is not necessary for effective transfer. Our large-scale analysis across 28 languages reveals that other factors, such as differences in pretraining and fine-tuning data and evaluation artifacts, better explain the perceived benefits of multilinguality. We also release fine-tuned models and provide empirical baselines to support future research. While focused on two sense-aware tasks, our findings offer broader insights into cross-lingual transfer, especially for low-resource languages.
>
---
#### [replaced 026] NExT-OMNI: Towards Any-to-Any Omnimodal Foundation Models with Discrete Flow Matching
- **分类: cs.CL; cs.AI; cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2510.13721v2](http://arxiv.org/pdf/2510.13721v2)**

> **作者:** Run Luo; Xiaobo Xia; Lu Wang; Longze Chen; Renke Shan; Jing Luo; Min Yang; Tat-Seng Chua
>
> **摘要:** Next-generation multimodal foundation models capable of any-to-any cross-modal generation and multi-turn interaction will serve as core components of artificial general intelligence systems, playing a pivotal role in human-machine interaction. However, most existing multimodal models remain constrained by autoregressive architectures, whose inherent limitations prevent a balanced integration of understanding and generation capabilities. Although hybrid and decoupling strategies have been explored to address these tasks within unified frameworks separately, their redundant, non-integrated designs limit their applicability to broader scenarios, such as cross-modal retrieval. In this work, we introduce NExT-OMNI, an open-source omnimodal foundation model that achieves unified modeling through discrete flow paradigms. By leveraging metric-induced probability paths and kinetic optimal velocities, NExT-OMNI natively supports any-to-any understanding and generation with enhanced response efficiency, while enabling broader application scenarios through concise unified representations rather than task-decoupled designs. Trained on large-scale interleaved text, image, video, and audio data, NExT-OMNI delivers competitive performance on multimodal generation and understanding benchmarks, while outperforming prior unified models in multi-turn multimodal interaction and cross-modal retrieval, highlighting its architectural advantages as a next-generation multimodal foundation model. To advance further research, we release training details, data protocols, and open-source both the code and model checkpoints.
>
---
#### [replaced 027] Following the Autoregressive Nature of LLM Embeddings via Compression and Alignment
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11401v3](http://arxiv.org/pdf/2502.11401v3)**

> **作者:** Jingcheng Deng; Zhongtao Jiang; Liang Pang; Liwei Chen; Kun Xu; Zihao Wei; Huawei Shen; Xueqi Cheng
>
> **摘要:** A new trend uses LLMs as dense text encoders via contrastive learning. However, since LLM embeddings predict the probability distribution of the next token, they are inherently generative and distributive, conflicting with contrastive learning, which requires embeddings to capture full-text semantics and align via cosine similarity. This discrepancy hinders the full utilization of LLMs' pre-training capabilities, resulting in inefficient learning. In response to this issue, we propose AutoRegEmbed, a new contrastive learning method built on embedding conditional probability distributions, which integrates two core tasks: information compression and conditional distribution alignment. The information compression task encodes text into the embedding space, ensuring that the embedding vectors capture global semantics. The conditional distribution alignment task focuses on aligning text embeddings with positive samples embeddings by leveraging the conditional distribution of embeddings while simultaneously reducing the likelihood of generating negative samples from text embeddings, thereby achieving embedding alignment and uniformity. Experimental results demonstrate that our method significantly outperforms traditional contrastive learning approaches and achieves performance comparable to state-of-the-art models when using the same amount of data.
>
---
#### [replaced 028] Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2407.11550v5](http://arxiv.org/pdf/2407.11550v5)**

> **作者:** Yuan Feng; Junlin Lv; Yukun Cao; Xike Xie; S. Kevin Zhou
>
> **备注:** NeurIPS 2025
>
> **摘要:** Large Language Models have excelled in various domains but face efficiency challenges due to the growing Key-Value (KV) cache required for long-sequence inference. Recent efforts aim to reduce KV cache size by evicting vast non-critical cache elements during runtime while preserving generation quality. However, these methods typically allocate compression budgets uniformly across all attention heads, ignoring the unique attention patterns of each head. In this paper, we establish a theoretical loss upper bound between pre- and post-eviction attention output, explaining the optimization target of prior cache eviction methods, while guiding the optimization of adaptive budget allocation. Base on this, we propose {\it Ada-KV}, the first head-wise adaptive budget allocation strategy. It offers plug-and-play benefits, enabling seamless integration with prior cache eviction methods. Extensive evaluations on 13 datasets from Ruler and 16 datasets from LongBench, all conducted under both question-aware and question-agnostic scenarios, demonstrate substantial quality improvements over existing methods. Our code is available at https://github.com/FFY0/AdaKV.
>
---
#### [replaced 029] RepIt: Representing Isolated Targets to Steer Language Models
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.13281v3](http://arxiv.org/pdf/2509.13281v3)**

> **作者:** Vincent Siu; Nathan W. Henry; Nicholas Crispino; Yang Liu; Dawn Song; Chenguang Wang
>
> **摘要:** While activation steering in large language models (LLMs) is a growing area of research, methods can often incur broader effects than desired. This motivates isolation of purer concept vectors to enable targeted interventions and understand LLM behavior at a more granular level. We present RepIt, a simple and data-efficient framework for isolating concept-specific representations. Across five frontier LLMs, RepIt enables precise interventions: it selectively suppresses refusal on targeted concepts while preserving refusal elsewhere, producing models that answer WMD-related questions while still scoring as safe on standard benchmarks. We further show that the corrective signal localizes to just 100-200 neurons and that robust target representations can be extracted from as few as a dozen examples on a single A6000. This efficiency raises a dual concern: manipulations can be performed with modest compute and data to extend to underrepresented data-scarce topics while evading existing benchmarks. By disentangling refusal vectors with RepIt, this work demonstrates that targeted interventions can counteract overgeneralization, laying the foundation for more granular control of model behavior.
>
---
#### [replaced 030] Echoes of BERT: Do Modern Language Models Rediscover the Classical NLP Pipeline?
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.02132v4](http://arxiv.org/pdf/2506.02132v4)**

> **作者:** Michael Li; Nishant Subramani
>
> **摘要:** Large transformer-based language models dominate modern NLP, yet our understanding of how they encode linguistic information relies primarily on studies of early models like BERT and GPT-2. Building on classic BERTology work, we analyze 25 models spanning from classical architectures (BERT, DeBERTa, GPT-2) to modern large language models (Pythia, OLMo-2, Gemma-2, Qwen2.5, Llama-3.1), probing layer-by-layer representations across eight linguistic tasks in English. Consistent with earlier findings, we find that hierarchical organization persists in modern models: early layers capture syntax, middle layers handle semantics and entity-level information, and later layers encode discourse phenomena. We dive deeper, conducting an in-depth multilingual analysis of two specific linguistic properties - lexical identity and inflectional morphology - that help disentangle form from meaning. We find that lexical information concentrates linearly in early layers but becomes increasingly nonlinear deeper in the network, while inflectional information remains linearly accessible throughout all layers. Additional analyses of attention mechanisms, steering vectors, and pretraining checkpoints reveal where this information resides within layers, how it can be functionally manipulated, and how representations evolve during pretraining. Taken together, our findings suggest that, even with substantial advances in LLM technologies, transformer models learn to organize linguistic information in similar ways, regardless of model architecture, size, or training regime, indicating that these properties are important for next token prediction. Our code is available at https://github.com/ml5885/model_internal_sleuthing
>
---
#### [replaced 031] Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.06261v5](http://arxiv.org/pdf/2507.06261v5)**

> **作者:** Gheorghe Comanici; Eric Bieber; Mike Schaekermann; Ice Pasupat; Noveen Sachdeva; Inderjit Dhillon; Marcel Blistein; Ori Ram; Dan Zhang; Evan Rosen; Luke Marris; Sam Petulla; Colin Gaffney; Asaf Aharoni; Nathan Lintz; Tiago Cardal Pais; Henrik Jacobsson; Idan Szpektor; Nan-Jiang Jiang; Krishna Haridasan; Ahmed Omran; Nikunj Saunshi; Dara Bahri; Gaurav Mishra; Eric Chu; Toby Boyd; Brad Hekman; Aaron Parisi; Chaoyi Zhang; Kornraphop Kawintiranon; Tania Bedrax-Weiss; Oliver Wang; Ya Xu; Ollie Purkiss; Uri Mendlovic; Ilaï Deutel; Nam Nguyen; Adam Langley; Flip Korn; Lucia Rossazza; Alexandre Ramé; Sagar Waghmare; Helen Miller; Nathan Byrd; Ashrith Sheshan; Raia Hadsell; Sangnie Bhardwaj; Pawel Janus; Tero Rissa; Dan Horgan; Alvin Abdagic; Lior Belenki; James Allingham; Anima Singh; Theo Guidroz; Srivatsan Srinivasan; Herman Schmit; Kristen Chiafullo; Andre Elisseeff; Nilpa Jha; Prateek Kolhar; Leonard Berrada; Frank Ding; Xiance Si; Shrestha Basu Mallick; Franz Och; Sofia Erell; Eric Ni; Tejasi Latkar; Sherry Yang; Petar Sirkovic; Ziqiang Feng; Robert Leland; Rachel Hornung; Gang Wu; Charles Blundell; Hamidreza Alvari; Po-Sen Huang; Cathy Yip; Sanja Deur; Li Liu; Gabriela Surita; Pablo Duque; Dima Damen; Johnson Jia; Arthur Guez; Markus Mircea; Animesh Sinha; Alberto Magni; Paweł Stradomski; Tal Marian; Vlado Galić; Wenhu Chen; Hisham Husain; Achintya Singhal; Dominik Grewe; François-Xavier Aubet; Shuang Song; Lorenzo Blanco; Leland Rechis; Lewis Ho; Rich Munoz; Kelvin Zheng; Jessica Hamrick; Kevin Mather; Hagai Taitelbaum; Eliza Rutherford; Yun Lei; Kuangyuan Chen; Anand Shukla; Erica Moreira; Eric Doi; Berivan Isik; Nir Shabat; Dominika Rogozińska; Kashyap Kolipaka; Jason Chang; Eugen Vušak; Srinivasan Venkatachary; Shadi Noghabi; Tarun Bharti; Younghoon Jun; Aleksandr Zaks; Simon Green; Jeshwanth Challagundla; William Wong; Muqthar Mohammad; Dean Hirsch; Yong Cheng; Iftekhar Naim; Lev Proleev; Damien Vincent; Aayush Singh; Maxim Krikun; Dilip Krishnan; Zoubin Ghahramani; Aviel Atias; Rajeev Aggarwal; Christo Kirov; Dimitrios Vytiniotis; Christy Koh; Alexandra Chronopoulou; Pawan Dogra; Vlad-Doru Ion; Gladys Tyen; Jason Lee; Felix Weissenberger; Trevor Strohman; Ashwin Balakrishna; Jack Rae; Marko Velic; Raoul de Liedekerke; Oded Elyada; Wentao Yuan; Canoee Liu; Lior Shani; Sergey Kishchenko; Bea Alessio; Yandong Li; Richard Song; Sam Kwei; Orion Jankowski; Aneesh Pappu; Youhei Namiki; Yenai Ma; Nilesh Tripuraneni; Colin Cherry; Marissa Ikonomidis; Yu-Cheng Ling; Colin Ji; Beka Westberg; Auriel Wright; Da Yu; David Parkinson; Swaroop Ramaswamy; Jerome Connor; Soheil Hassas Yeganeh; Snchit Grover; George Kenwright; Lubo Litchev; Chris Apps; Alex Tomala; Felix Halim; Alex Castro-Ros; Zefei Li; Anudhyan Boral; Pauline Sho; Michal Yarom; Eric Malmi; David Klinghoffer; Rebecca Lin; Alan Ansell; Pradeep Kumar S; Shubin Zhao; Siqi Zuo; Adam Santoro; Heng-Tze Cheng; Solomon Demmessie; Yuchi Liu; Nicole Brichtova; Allie Culp; Nathaniel Braun; Dan Graur; Will Ng; Nikhil Mehta; Aaron Phillips; Patrik Sundberg; Varun Godbole; Fangyu Liu; Yash Katariya; David Rim; Mojtaba Seyedhosseini; Sean Ammirati; Jonas Valfridsson; Mahan Malihi; Timothy Knight; Andeep Toor; Thomas Lampe; Abe Ittycheriah; Lewis Chiang; Chak Yeung; Alexandre Fréchette; Jinmeng Rao; Huisheng Wang; Himanshu Srivastava; Richard Zhang; Rocky Rhodes; Ariel Brand; Dean Weesner; Ilya Figotin; Felix Gimeno; Rachana Fellinger; Pierre Marcenac; José Leal; Eyal Marcus; Victor Cotruta; Rodrigo Cabrera; Sheryl Luo; Dan Garrette; Vera Axelrod; Sorin Baltateanu; David Barker; Dongkai Chen; Horia Toma; Ben Ingram; Jason Riesa; Chinmay Kulkarni; Yujing Zhang; Hongbin Liu; Chao Wang; Martin Polacek; Will Wu; Kai Hui; Adrian N Reyes; Yi Su; Megan Barnes; Ishaan Malhi; Anfal Siddiqui; Qixuan Feng; Mihai Damaschin; Daniele Pighin; Andreas Steiner; Samuel Yang; Ramya Sree Boppana; Simeon Ivanov; Arun Kandoor; Aditya Shah; Asier Mujika; Da Huang; Christopher A. Choquette-Choo; Mohak Patel; Tianhe Yu; Toni Creswell; Jerry; Liu; Catarina Barros; Yasaman Razeghi; Aurko Roy; Phil Culliton; Binbin Xiong; Jiaqi Pan; Thomas Strohmann; Tolly Powell; Babi Seal; Doug DeCarlo; Pranav Shyam; Kaan Katircioglu; Xuezhi Wang; Cassidy Hardin; Immanuel Odisho; Josef Broder; Oscar Chang; Arun Nair; Artem Shtefan; Maura O'Brien; Manu Agarwal; Sahitya Potluri; Siddharth Goyal; Amit Jhindal; Saksham Thakur; Yury Stuken; James Lyon; Kristina Toutanova; Fangxiaoyu Feng; Austin Wu; Ben Horn; Alek Wang; Alex Cullum; Gabe Taubman; Disha Shrivastava; Chongyang Shi; Hamish Tomlinson; Roma Patel; Tao Tu; Ada Maksutaj Oflazer; Francesco Pongetti; Mingyao Yang; Adrien Ali Taïga; Vincent Perot; Nuo Wang Pierse; Feng Han; Yoel Drori; Iñaki Iturrate; Ayan Chakrabarti; Legg Yeung; Dave Dopson; Yi-ting Chen; Apoorv Kulshreshtha; Tongfei Guo; Philip Pham; Tal Schuster; Junquan Chen; Alex Polozov; Jinwei Xing; Huanjie Zhou; Praneeth Kacham; Doron Kukliansky; Antoine Miech; Sergey Yaroshenko; Ed Chi; Sholto Douglas; Hongliang Fei; Mathieu Blondel; Preethi Myla; Lior Madmoni; Xing Wu; Daniel Keysers; Kristian Kjems; Isabela Albuquerque; Lijun Yu; Joel D'sa; Michelle Plantan; Vlad Ionescu; Jaume Sanchez Elias; Abhirut Gupta; Manish Reddy Vuyyuru; Fred Alcober; Tong Zhou; Kaiyang Ji; Florian Hartmann; Subha Puttagunta; Hugo Song; Ehsan Amid; Anca Stefanoiu; Andrew Lee; Paul Pucciarelli; Emma Wang; Amit Raul; Slav Petrov; Isaac Tian; Valentin Anklin; Nana Nti; Victor Gomes; Max Schumacher; Grace Vesom; Alex Panagopoulos; Konstantinos Bousmalis; Daniel Andor; Josh Jacob; Yuan Zhang; Bill Rosgen; Matija Kecman; Matthew Tung; Alexandra Belias; Noah Goodman; Paul Covington; Brian Wieder; Nikita Saxena; Elnaz Davoodi; Muhuan Huang; Sharath Maddineni; Vincent Roulet; Folawiyo Campbell-Ajala; Pier Giuseppe Sessa; Xintian; Wu; Guangda Lai; Paul Collins; Alex Haig; Vytenis Sakenas; Xiaowei Xu; Marissa Giustina; Laurent El Shafey; Pichi Charoenpanit; Shefali Garg; Joshua Ainslie; Boone Severson; Montse Gonzalez Arenas; Shreya Pathak; Sujee Rajayogam; Jie Feng; Michiel Bakker; Sheng Li; Nevan Wichers; Jamie Rogers; Xinyang Geng; Yeqing Li; Rolf Jagerman; Chao Jia; Nadav Olmert; David Sharon; Matthew Mauger; Sandeep Mariserla; Hongxu Ma; Megha Mohabey; Kyuyeun Kim; Alek Andreev; Scott Pollom; Juliette Love; Vihan Jain; Priyanka Agrawal; Yannick Schroecker; Alisa Fortin; Manfred Warmuth; Ji Liu; Andrew Leach; Irina Blok; Ganesh Poomal Girirajan; Roee Aharoni; Benigno Uria; Andrei Sozanschi; Dan Goldberg; Lucian Ionita; Marco Tulio Ribeiro; Martin Zlocha; Vighnesh Birodkar; Sami Lachgar; Liangzhe Yuan; Himadri Choudhury; Matt Ginsberg; Fei Zheng; Gregory Dibb; Emily Graves; Swachhand Lokhande; Gabriel Rasskin; George-Cristian Muraru; Corbin Quick; Sandeep Tata; Pierre Sermanet; Aditya Chawla; Itay Karo; Yan Wang; Susan Zhang; Orgad Keller; Anca Dragan; Guolong Su; Ian Chou; Xi Liu; Yiqing Tao; Shruthi Prabhakara; Marc Wilson; Ruibo Liu; Shibo Wang; Georgie Evans; David Du; Alfonso Castaño; Gautam Prasad; Mona El Mahdy; Sebastian Gerlach; Machel Reid; Jarrod Kahn; Amir Zait; Thanumalayan Sankaranarayana Pillai; Thatcher Ulrich; Guanyu Wang; Jan Wassenberg; Efrat Farkash; Kiran Yalasangi; Congchao Wang; Maria Bauza; Simon Bucher; Ting Liu; Jun Yan; Gary Leung; Vikas Sindhwani; Parker Barnes; Avi Singh; Ivan Jurin; Jichuan Chang; Niket Kumar Bhumihar; Sivan Eiger; Gui Citovsky; Ben Withbroe; Zhang Li; Siyang Xue; Niccolò Dal Santo; Georgi Stoyanov; Yves Raimond; Steven Zheng; Yilin Gao; Vít Listík; Sławek Kwasiborski; Rachel Saputro; Adnan Ozturel; Ganesh Mallya; Kushal Majmundar; Ross West; Paul Caron; Jinliang Wei; Lluis Castrejon; Sharad Vikram; Deepak Ramachandran; Nikhil Dhawan; Jiho Park; Sara Smoot; George van den Driessche; Yochai Blau; Chase Malik; Wei Liang; Roy Hirsch; Cicero Nogueira dos Santos; Eugene Weinstein; Aäron van den Oord; Sid Lall; Nicholas FitzGerald; Zixuan Jiang; Xuan Yang; Dale Webster; Ali Elqursh; Aedan Pope; Georges Rotival; David Raposo; Wanzheng Zhu; Jeff Dean; Sami Alabed; Dustin Tran; Arushi Gupta; Zach Gleicher; Jessica Austin; Edouard Rosseel; Megh Umekar; Dipanjan Das; Yinghao Sun; Kai Chen; Karolis Misiunas; Xiang Zhou; Yixian Di; Alyssa Loo; Josh Newlan; Bo Li; Vinay Ramasesh; Ying Xu; Alex Chen; Sudeep Gandhe; Radu Soricut; Nikita Gupta; Shuguang Hu; Seliem El-Sayed; Xavier Garcia; Idan Brusilovsky; Pu-Chin Chen; Andrew Bolt; Lu Huang; Alex Gurney; Zhiying Zhang; Alexander Pritzel; Jarek Wilkiewicz; Bryan Seybold; Bhargav Kanagal Shamanna; Felix Fischer; Josef Dean; Karan Gill; Ross Mcilroy; Abhishek Bhowmick; Jeremy Selier; Antoine Yang; Derek Cheng; Vladimir Magay; Jie Tan; Dhriti Varma; Christian Walder; Tomas Kocisky; Ryo Nakashima; Paul Natsev; Mike Kwong; Ionel Gog; Chiyuan Zhang; Sander Dieleman; Thomas Jimma; Andrey Ryabtsev; Siddhartha Brahma; David Steiner; Dayou Du; Ante Žužul; Mislav Žanić; Mukund Raghavachari; Willi Gierke; Zeyu Zheng; Dessie Petrova; Yann Dauphin; Yuchuan Liu; Ido Kessler; Steven Hand; Chris Duvarney; Seokhwan Kim; Hyo Lee; Léonard Hussenot; Jeffrey Hui; Josh Smith; Deepali Jain; Jiawei Xia; Gaurav Singh Tomar; Keyvan Amiri; Du Phan; Fabian Fuchs; Tobias Weyand; Nenad Tomasev; Alexandra Cordell; Xin Liu; Jonathan Mallinson; Pankaj Joshi; Andy Crawford; Arun Suggala; Steve Chien; Nick Fernando; Mariella Sanchez-Vargas; Duncan Williams; Phil Crone; Xiyang Luo; Igor Karpov; Jyn Shan; Terry Thurk; Robin Strudel; Paul Voigtlaender; Piyush Patil; Tim Dozat; Ali Khodaei; Sahil Singla; Piotr Ambroszczyk; Qiyin Wu; Yifan Chang; Brian Roark; Chaitra Hegde; Tianli Ding; Angelos Filos; Zhongru Wu; André Susano Pinto; Shuang Liu; Saarthak Khanna; Aditya Pandey; Siobhan Mcloughlin; Qiujia Li; Sam Haves; Allan Zhou; Elena Buchatskaya; Isabel Leal; Peter de Boursac; Nami Akazawa; Nina Anderson; Terry Chen; Krishna Somandepalli; Chen Liang; Sheela Goenka; Stephanie Winkler; Alexander Grushetsky; Yifan Ding; Jamie Smith; Fan Ye; Jordi Pont-Tuset; Eric Li; Ruichao Li; Tomer Golany; Dawid Wegner; Tao Jiang; Omer Barak; Yuan Shangguan; Eszter Vértes; Renee Wong; Jörg Bornschein; Alex Tudor; Michele Bevilacqua; Tom Schaul; Ankit Singh Rawat; Yang Zhao; Kyriakos Axiotis; Lei Meng; Cory McLean; Jonathan Lai; Jennifer Beattie; Nate Kushman; Yaxin Liu; Blair Kutzman; Fiona Lang; Jingchen Ye; Praneeth Netrapalli; Pushkar Mishra; Myriam Khan; Megha Goel; Rob Willoughby; David Tian; Honglei Zhuang; JD Chen; Zak Tsai; Tasos Kementsietsidis; Arjun Khare; James Keeling; Keyang Xu; Nathan Waters; Florent Altché; Ashok Popat; Bhavishya Mittal; David Saxton; Dalia El Badawy; Michael Mathieu; Zheng Zheng; Hao Zhou; Nishant Ranka; Richard Shin; Qingnan Duan; Tim Salimans; Ioana Mihailescu; Uri Shaham; Ming-Wei Chang; Yannis Assael; Nishanth Dikkala; Martin Izzard; Vincent Cohen-Addad; Cat Graves; Vlad Feinberg; Grace Chung; DJ Strouse; Danny Karmon; Sahand Sharifzadeh; Zoe Ashwood; Khiem Pham; Jon Blanton; Alex Vasiloff; Jarred Barber; Mark Geller; Aurick Zhou; Fedir Zubach; Tzu-Kuo Huang; Lei Zhang; Himanshu Gupta; Matt Young; Julia Proskurnia; Ronny Votel; Valentin Gabeur; Gabriel Barcik; Aditya Tripathi; Hongkun Yu; Geng Yan; Beer Changpinyo; Filip Pavetić; Amy Coyle; Yasuhisa Fujii; Jorge Gonzalez Mendez; Tianhao Zhou; Harish Rajamani; Blake Hechtman; Eddie Cao; Da-Cheng Juan; Yi-Xuan Tan; Valentin Dalibard; Yilun Du; Natalie Clay; Kaisheng Yao; Wenhao Jia; Dimple Vijaykumar; Yuxiang Zhou; Xinyi Bai; Wei-Chih Hung; Steven Pecht; Georgi Todorov; Nikhil Khadke; Pramod Gupta; Preethi Lahoti; Arnaud Autef; Karthik Duddu; James Lee-Thorp; Alexander Bykovsky; Tautvydas Misiunas; Sebastian Flennerhag; Santhosh Thangaraj; Jed McGiffin; Zack Nado; Markus Kunesch; Andreas Noever; Amir Hertz; Marco Liang; Victor Stone; Evan Palmer; Samira Daruki; Arijit Pramanik; Siim Põder; Austin Kyker; Mina Khan; Evgeny Sluzhaev; Marvin Ritter; Avraham Ruderman; Wenlei Zhou; Chirag Nagpal; Kiran Vodrahalli; George Necula; Paul Barham; Ellie Pavlick; Jay Hartford; Izhak Shafran; Long Zhao; Maciej Mikuła; Tom Eccles; Hidetoshi Shimokawa; Kanav Garg; Luke Vilnis; Hanwen Chen; Ilia Shumailov; Kuang-Huei Lee; Abdelrahman Abdelhamed; Meiyan Xie; Vered Cohen; Ester Hlavnova; Dan Malkin; Chawin Sitawarin; James Lottes; Pauline Coquinot; Tianli Yu; Sandeep Kumar; Jingwei Zhang; Aroma Mahendru; Zafarali Ahmed; James Martens; Tao Chen; Aviel Boag; Daiyi Peng; Coline Devin; Arseniy Klimovskiy; Mary Phuong; Danny Vainstein; Jin Xie; Bhuvana Ramabhadran; Nathan Howard; Xinxin Yu; Gitartha Goswami; Jingyu Cui; Sam Shleifer; Mario Pinto; Chih-Kuan Yeh; Ming-Hsuan Yang; Sara Javanmardi; Dan Ethier; Chace Lee; Jordi Orbay; Suyog Kotecha; Carla Bromberg; Pete Shaw; James Thornton; Adi Gerzi Rosenthal; Shane Gu; Matt Thomas; Ian Gemp; Aditya Ayyar; Asahi Ushio; Aarush Selvan; Joel Wee; Chenxi Liu; Maryam Majzoubi; Weiren Yu; Jake Abernethy; Tyler Liechty; Renke Pan; Hoang Nguyen; Qiong; Hu; Sarah Perrin; Abhinav Arora; Emily Pitler; Weiyi Wang; Kaushik Shivakumar; Flavien Prost; Ben Limonchik; Jing Wang; Yi Gao; Timothee Cour; Shyamal Buch; Huan Gui; Maria Ivanova; Philipp Neubeck; Kelvin Chan; Lucy Kim; Huizhong Chen; Naman Goyal; Da-Woon Chung; Lu Liu; Yao Su; Anastasia Petrushkina; Jiajun Shen; Armand Joulin; Yuanzhong Xu; Stein Xudong Lin; Yana Kulizhskaya; Ciprian Chelba; Shobha Vasudevan; Eli Collins; Vasilisa Bashlovkina; Tony Lu; Doug Fritz; Jongbin Park; Yanqi Zhou; Chen Su; Richard Tanburn; Mikhail Sushkov; Mitchelle Rasquinha; Jinning Li; Jennifer Prendki; Yiming Li; Pallavi LV; Shriya Sharma; Hen Fitoussi; Hui Huang; Andrew Dai; Phuong Dao; Mike Burrows; Henry Prior; Danfeng Qin; Golan Pundak; Lars Lowe Sjoesund; Art Khurshudov; Zhenkai Zhu; Albert Webson; Elizabeth Kemp; Tat Tan; Saurabh Agrawal; Susie Sargsyan; Liqun Cheng; Jim Stephan; Tom Kwiatkowski; David Reid; Arunkumar Byravan; Assaf Hurwitz Michaely; Nicolas Heess; Luowei Zhou; Sonam Goenka; Viral Carpenter; Anselm Levskaya; Bo Wang; Reed Roberts; Rémi Leblond; Sharat Chikkerur; Stav Ginzburg; Max Chang; Robert Riachi; Chuqiao; Xu; Zalán Borsos; Michael Pliskin; Julia Pawar; Morgane Lustman; Hannah Kirkwood; Ankit Anand; Aditi Chaudhary; Norbert Kalb; Kieran Milan; Sean Augenstein; Anna Goldie; Laurel Prince; Karthik Raman; Yanhua Sun; Vivian Xia; Aaron Cohen; Zhouyuan Huo; Josh Camp; Seher Ellis; Lukas Zilka; David Vilar Torres; Lisa Patel; Sho Arora; Betty Chan; Jonas Adler; Kareem Ayoub; Jacky Liang; Fayaz Jamil; Jiepu Jiang; Simon Baumgartner; Haitian Sun; Yael Karov; Yaroslav Akulov; Hui Zheng; Irene Cai; Claudio Fantacci; James Rubin; Alex Rav Acha; Mengchao Wang; Nina D'Souza; Rohit Sathyanarayana; Shengyang Dai; Simon Rowe; Andrey Simanovsky; Omer Goldman; Yuheng Kuang; Xiaoyue Pan; Andrew Rosenberg; Tania Rojas-Esponda; Praneet Dutta; Amy Zeng; Irina Jurenka; Greg Farquhar; Yamini Bansal; Shariq Iqbal; Becca Roelofs; Ga-Young Joung; Parker Beak; Changwan Ryu; Ryan Poplin; Yan Wu; Jean-Baptiste Alayrac; Senaka Buthpitiya; Olaf Ronneberger; Caleb Habtegebriel; Wei Li; Paul Cavallaro; Aurora Wei; Guy Bensky; Timo Denk; Harish Ganapathy; Jeff Stanway; Pratik Joshi; Francesco Bertolini; Jessica Lo; Olivia Ma; Zachary Charles; Geta Sampemane; Himanshu Sahni; Xu Chen; Harry Askham; David Gaddy; Peter Young; Jiewen Tan; Matan Eyal; Arthur Bražinskas; Li Zhong; Zhichun Wu; Mark Epstein; Kai Bailey; Andrew Hard; Kamyu Lee; Sasha Goldshtein; Alex Ruiz; Mohammed Badawi; Matthias Lochbrunner; JK Kearns; Ashley Brown; Fabio Pardo; Theophane Weber; Haichuan Yang; Pan-Pan Jiang; Berkin Akin; Zhao Fu; Marcus Wainwright; Chi Zou; Meenu Gaba; Pierre-Antoine Manzagol; Wendy Kan; Yang Song; Karina Zainullina; Rui Lin; Jeongwoo Ko; Salil Deshmukh; Apoorv Jindal; James Svensson; Divya Tyam; Heri Zhao; Christine Kaeser-Chen; Scott Baird; Pooya Moradi; Jamie Hall; Qiuchen Guo; Vincent Tsang; Bowen Liang; Fernando Pereira; Suhas Ganesh; Ivan Korotkov; Jakub Adamek; Sridhar Thiagarajan; Vinh Tran; Charles Chen; Chris Tar; Sanil Jain; Ishita Dasgupta; Taylan Bilal; David Reitter; Kai Zhao; Giulia Vezzani; Yasmin Gehman; Pulkit Mehta; Lauren Beltrone; Xerxes Dotiwalla; Sergio Guadarrama; Zaheer Abbas; Stefani Karp; Petko Georgiev; Chun-Sung Ferng; Marc Brockschmidt; Liqian Peng; Christoph Hirnschall; Vikas Verma; Yingying Bi; Ying Xiao; Avigail Dabush; Kelvin Xu; Phil Wallis; Randall Parker; Qifei Wang; Yang Xu; Ilkin Safarli; Dinesh Tewari; Yin Zhang; Seungyeon Kim; Andrea Gesmundo; Mackenzie Thomas; Sergey Levi; Ahmed Chowdhury; Kanishka Rao; Peter Garst; Sam Conway-Rahman; Helen Ran; Kay McKinney; Zhisheng Xiao; Wenhao Yu; Rohan Agrawal; Axel Stjerngren; Catalin Ionescu; Jingjing Chen; Vivek Sharma; Justin Chiu; Fei Liu; Ken Franko; Clayton Sanford; Xingyu Cai; Paul Michel; Sanjay Ganapathy; Jane Labanowski; Zachary Garrett; Ben Vargas; Sean Sun; Bryan Gale; Thomas Buschmann; Guillaume Desjardins; Nimesh Ghelani; Palak Jain; Mudit Verma; Chulayuth Asawaroengchai; Julian Eisenschlos; Jitendra Harlalka; Hideto Kazawa; Don Metzler; Joshua Howland; Ying Jian; Jake Ades; Viral Shah; Tynan Gangwani; Seungji Lee; Roman Ring; Steven M. Hernandez; Dean Reich; Amer Sinha; Ashutosh Sathe; Joe Kovac; Ashleah Gill; Ajay Kannan; Andrea D'olimpio; Martin Sevenich; Jay Whang; Been Kim; Khe Chai Sim; Jilin Chen; Jiageng Zhang; Shuba Lall; Yossi Matias; Bill Jia; Abe Friesen; Sara Nasso; Ashish Thapliyal; Bryan Perozzi; Ting Yu; Anna Shekhawat; Safeen Huda; Peter Grabowski; Eric Wang; Ashwin Sreevatsa; Hilal Dib; Mehadi Hassen; Parker Schuh; Vedrana Milutinovic; Chris Welty; Michael Quinn; Ali Shah; Bangju Wang; Gabe Barth-Maron; Justin Frye; Natalie Axelsson; Tao Zhu; Yukun Ma; Irene Giannoumis; Hanie Sedghi; Chang Ye; Yi Luan; Kevin Aydin; Bilva Chandra; Vivek Sampathkumar; Ronny Huang; Victor Lavrenko; Ahmed Eleryan; Zhi Hong; Steven Hansen; Sara Mc Carthy; Bidisha Samanta; Domagoj Ćevid; Xin Wang; Fangtao Li; Michael Voznesensky; Matt Hoffman; Andreas Terzis; Vikash Sehwag; Gil Fidel; Luheng He; Mu Cai; Yanzhang He; Alex Feng; Martin Nikoltchev; Samrat Phatale; Jason Chase; Rory Lawton; Ming Zhang; Tom Ouyang; Manuel Tragut; Mehdi Hafezi Manshadi; Arjun Narayanan; Jiaming Shen; Xu Gao; Tolga Bolukbasi; Nick Roy; Xin Li; Daniel Golovin; Liviu Panait; Zhen Qin; Guangxing Han; Thomas Anthony; Sneha Kudugunta; Viorica Patraucean; Aniket Ray; Xinyun Chen; Xiaochen Yang; Tanuj Bhatia; Pranav Talluri; Alex Morris; Andrija Ražnatović; Bethanie Brownfield; James An; Sheng Peng; Patrick Kane; Ce Zheng; Nico Duduta; Joshua Kessinger; James Noraky; Siqi Liu; Keran Rong; Petar Veličković; Keith Rush; Alex Goldin; Fanny Wei; Shiva Mohan Reddy Garlapati; Caroline Pantofaru; Okwan Kwon; Jianmo Ni; Eric Noland; Julia Di Trapani; Françoise Beaufays; Abhijit Guha Roy; Yinlam Chow; Aybuke Turker; Geoffrey Cideron; Lantao Mei; Jon Clark; Qingyun Dou; Matko Bošnjak; Ralph Leith; Yuqing Du; Amir Yazdanbakhsh; Milad Nasr; Chester Kwak; Suraj Satishkumar Sheth; Alex Kaskasoli; Ankesh Anand; Balaji Lakshminarayanan; Sammy Jerome; David Bieber; Chun-Te Chu; Alexandre Senges; Tianxiao Shen; Mukund Sridhar; Ndaba Ndebele; Benjamin Beyret; Shakir Mohamed; Mia Chen; Markus Freitag; Jiaxian Guo; Luyang Liu; Paul Roit; Heng Chen; Shen Yan; Tom Stone; JD Co-Reyes; Jeremy Cole; Salvatore Scellato; Shekoofeh Azizi; Hadi Hashemi; Alicia Jin; Anand Iyer; Marcella Valentine; András György; Arun Ahuja; Daniel Hernandez Diaz; Chen-Yu Lee; Nathan Clement; Weize Kong; Drew Garmon; Ishaan Watts; Kush Bhatia; Khyatti Gupta; Matt Miecnikowski; Hugo Vallet; Ankur Taly; Edward Loper; Saket Joshi; James Atwood; Jo Chick; Mark Collier; Fotis Iliopoulos; Ryan Trostle; Beliz Gunel; Ramiro Leal-Cavazos; Arnar Mar Hrafnkelsson; Michael Guzman; Xiaoen Ju; Andy Forbes; Jesse Emond; Kushal Chauhan; Ben Caine; Li Xiao; Wenjun Zeng; Alexandre Moufarek; Daniel Murphy; Maya Meng; Nitish Gupta; Felix Riedel; Anil Das; Elijah Lawal; Shashi Narayan; Tiberiu Sosea; James Swirhun; Linda Friso; Behnam Neyshabur; Jing Lu; Sertan Girgin; Michael Wunder; Edouard Yvinec; Aroonalok Pyne; Victor Carbune; Shruti Rijhwani; Yang Guo; Tulsee Doshi; Anton Briukhov; Max Bain; Ayal Hitron; Xuanhui Wang; Ashish Gupta; Ke Chen; Cosmo Du; Weiyang Zhang; Dhruv Shah; Arjun Akula; Max Dylla; Ashyana Kachra; Weicheng Kuo; Tingting Zou; Lily Wang; Luyao Xu; Jifan Zhu; Justin Snyder; Sachit Menon; Orhan Firat; Igor Mordatch; Yuan Yuan; Natalia Ponomareva; Rory Blevins; Lawrence Moore; Weijun Wang; Phil Chen; Martin Scholz; Artur Dwornik; Jason Lin; Sicheng Li; Diego Antognini; Te I; Xiaodan Song; Matt Miller; Uday Kalra; Adam Raveret; Oscar Akerlund; Felix Wu; Andrew Nystrom; Namrata Godbole; Tianqi Liu; Hannah DeBalsi; Jewel Zhao; Buhuang Liu; Avi Caciularu; Lauren Lax; Urvashi Khandelwal; Victoria Langston; Eric Bailey; Silvio Lattanzi; Yufei Wang; Neel Kovelamudi; Sneha Mondal; Guru Guruganesh; Nan Hua; Ofir Roval; Paweł Wesołowski; Rishikesh Ingale; Jonathan Halcrow; Tim Sohn; Christof Angermueller; Bahram Raad; Eli Stickgold; Eva Lu; Alec Kosik; Jing Xie; Timothy Lillicrap; Austin Huang; Lydia Lihui Zhang; Dominik Paulus; Clement Farabet; Alex Wertheim; Bing Wang; Rishabh Joshi; Chu-ling Ko; Yonghui Wu; Shubham Agrawal; Lily Lin; XiangHai Sheng; Peter Sung; Tyler Breland-King; Christina Butterfield; Swapnil Gawde; Sumeet Singh; Qiao Zhang; Raj Apte; Shilpa Shetty; Adrian Hutter; Tao Li; Elizabeth Salesky; Federico Lebron; Jonni Kanerva; Michela Paganini; Arthur Nguyen; Rohith Vallu; Jan-Thorsten Peter; Sarmishta Velury; David Kao; Jay Hoover; Anna Bortsova; Colton Bishop; Shoshana Jakobovits; Alessandro Agostini; Alekh Agarwal; Chang Liu; Charles Kwong; Sasan Tavakkol; Ioana Bica; Alex Greve; Anirudh GP; Jake Marcus; Le Hou; Tom Duerig; Rivka Moroshko; Dave Lacey; Andy Davis; Julien Amelot; Guohui Wang; Frank Kim; Theofilos Strinopoulos; Hui Wan; Charline Le Lan; Shankar Krishnan; Haotian Tang; Peter Humphreys; Junwen Bai; Idan Heimlich Shtacher; Diego Machado; Chenxi Pang; Ken Burke; Dangyi Liu; Renga Aravamudhan; Yue Song; Ed Hirst; Abhimanyu Singh; Brendan Jou; Liang Bai; Francesco Piccinno; Chuyuan Kelly Fu; Robin Alazard; Barak Meiri; Daniel Winter; Charlie Chen; Mingda Zhang; Jens Heitkaemper; John Lambert; Jinhyuk Lee; Alexander Frömmgen; Sergey Rogulenko; Pranav Nair; Paul Niemczyk; Anton Bulyenov; Bibo Xu; Hadar Shemtov; Morteza Zadimoghaddam; Serge Toropov; Mateo Wirth; Hanjun Dai; Sreenivas Gollapudi; Daniel Zheng; Alex Kurakin; Chansoo Lee; Kalesha Bullard; Nicolas Serrano; Ivana Balazevic; Yang Li; Johan Schalkwyk; Mark Murphy; Mingyang Zhang; Kevin Sequeira; Romina Datta; Nishant Agrawal; Charles Sutton; Nithya Attaluri; Mencher Chiang; Wael Farhan; Gregory Thornton; Kate Lin; Travis Choma; Hung Nguyen; Kingshuk Dasgupta; Dirk Robinson; Iulia Comşa; Michael Riley; Arjun Pillai; Basil Mustafa; Ben Golan; Amir Zandieh; Jean-Baptiste Lespiau; Billy Porter; David Ross; Sujeevan Rajayogam; Mohit Agarwal; Subhashini Venugopalan; Bobak Shahriari; Qiqi Yan; Hao Xu; Taylor Tobin; Pavel Dubov; Hongzhi Shi; Adrià Recasens; Anton Kovsharov; Sebastian Borgeaud; Lucio Dery; Shanthal Vasanth; Elena Gribovskaya; Linhai Qiu; Mahdis Mahdieh; Wojtek Skut; Elizabeth Nielsen; CJ Zheng; Adams Yu; Carrie Grimes Bostock; Shaleen Gupta; Aaron Archer; Chris Rawles; Elinor Davies; Alexey Svyatkovskiy; Tomy Tsai; Yoni Halpern; Christian Reisswig; Bartek Wydrowski; Bo Chang; Joan Puigcerver; Mor Hazan Taege; Jian Li; Eva Schnider; Xinjian Li; Dragos Dena; Yunhan Xu; Umesh Telang; Tianze Shi; Heiga Zen; Kyle Kastner; Yeongil Ko; Neesha Subramaniam; Aviral Kumar; Pete Blois; Zhuyun Dai; John Wieting; Yifeng Lu; Yoel Zeldes; Tian Xie; Anja Hauth; Alexandru Ţifrea; Yuqi Li; Sam El-Husseini; Dan Abolafia; Howard Zhou; Wen Ding; Sahra Ghalebikesabi; Carlos Guía; Andrii Maksai; Ágoston Weisz; Sercan Arik; Nick Sukhanov; Aga Świetlik; Xuhui Jia; Luo Yu; Weiyue Wang; Mark Brand; Dawn Bloxwich; Sean Kirmani; Zhe Chen; Alec Go; Pablo Sprechmann; Nithish Kannen; Alen Carin; Paramjit Sandhu; Isabel Edkins; Leslie Nooteboom; Jai Gupta; Loren Maggiore; Javad Azizi; Yael Pritch; Pengcheng Yin; Mansi Gupta; Danny Tarlow; Duncan Smith; Desi Ivanov; Mohammad Babaeizadeh; Ankita Goel; Satish Kambala; Grace Chu; Matej Kastelic; Michelle Liu; Hagen Soltau; Austin Stone; Shivani Agrawal; Min Kim; Kedar Soparkar; Srinivas Tadepalli; Oskar Bunyan; Rachel Soh; Arvind Kannan; DY Kim; Blake JianHang Chen; Afief Halumi; Sudeshna Roy; Yulong Wang; Olcan Sercinoglu; Gena Gibson; Sijal Bhatnagar; Motoki Sano; Daniel von Dincklage; Qingchun Ren; Blagoj Mitrevski; Mirek Olšák; Jennifer She; Carl Doersch; Jilei; Wang; Bingyuan Liu; Qijun Tan; Tamar Yakar; Tris Warkentin; Alex Ramirez; Carl Lebsack; Josh Dillon; Rajiv Mathews; Tom Cobley; Zelin Wu; Zhuoyuan Chen; Jon Simon; Swaroop Nath; Tara Sainath; Alexei Bendebury; Ryan Julian; Bharath Mankalale; Daria Ćurko; Paulo Zacchello; Adam R. Brown; Kiranbir Sodhia; Heidi Howard; Sergi Caelles; Abhinav Gupta; Gareth Evans; Anna Bulanova; Lesley Katzen; Roman Goldenberg; Anton Tsitsulin; Joe Stanton; Benoit Schillings; Vitaly Kovalev; Corey Fry; Rushin Shah; Kuo Lin; Shyam Upadhyay; Cheng Li; Soroush Radpour; Marcello Maggioni; Jing Xiong; Lukas Haas; Jenny Brennan; Aishwarya Kamath; Nikolay Savinov; Arsha Nagrani; Trevor Yacovone; Ryan Kappedal; Kostas Andriopoulos; Li Lao; YaGuang Li; Grigory Rozhdestvenskiy; Kazuma Hashimoto; Andrew Audibert; Sophia Austin; Daniel Rodriguez; Anian Ruoss; Garrett Honke; Deep Karkhanis; Xi Xiong; Qing Wei; James Huang; Zhaoqi Leng; Vittal Premachandran; Stan Bileschi; Georgios Evangelopoulos; Thomas Mensink; Jay Pavagadhi; Denis Teplyashin; Paul Chang; Linting Xue; Garrett Tanzer; Sally Goldman; Kaushal Patel; Shixin Li; Jeremy Wiesner; Ivy Zheng; Ian Stewart-Binks; Jie Han; Zhi Li; Liangchen Luo; Karel Lenc; Mario Lučić; Fuzhao Xue; Ryan Mullins; Alexey Guseynov; Chung-Ching Chang; Isaac Galatzer-Levy; Adam Zhang; Garrett Bingham; Grace Hu; Ale Hartman; Yue Ma; Jordan Griffith; Alex Irpan; Carey Radebaugh; Summer Yue; Lijie Fan; Victor Ungureanu; Christina Sorokin; Hannah Teufel; Peiran Li; Rohan Anil; Dimitris Paparas; Todd Wang; Chu-Cheng Lin; Hui Peng; Megan Shum; Goran Petrovic; Demetra Brady; Richard Nguyen; Klaus Macherey; Zhihao Li; Harman Singh; Madhavi Yenugula; Mariko Iinuma; Xinyi Chen; Kavya Kopparapu; Alexey Stern; Shachi Dave; Chandu Thekkath; Florence Perot; Anurag Kumar; Fangda Li; Yang Xiao; Matthew Bilotti; Mohammad Hossein Bateni; Isaac Noble; Lisa Lee; Amelio Vázquez-Reina; Julian Salazar; Xiaomeng Yang; Boyu Wang; Ela Gruzewska; Anand Rao; Sindhu Raghuram; Zheng Xu; Eyal Ben-David; Jieru Mei; Sid Dalmia; Zhaoyi Zhang; Yuchen Liu; Gagan Bansal; Helena Pankov; Steven Schwarcz; Andrea Burns; Christine Chan; Sumit Sanghai; Ricky Liang; Ethan Liang; Antoine He; Amy Stuart; Arun Narayanan; Yukun Zhu; Christian Frank; Bahar Fatemi; Amit Sabne; Oran Lang; Indro Bhattacharya; Shane Settle; Maria Wang; Brendan McMahan; Andrea Tacchetti; Livio Baldini Soares; Majid Hadian; Serkan Cabi; Timothy Chung; Nikita Putikhin; Gang Li; Jeremy Chen; Austin Tarango; Henryk Michalewski; Mehran Kazemi; Hussain Masoom; Hila Sheftel; Rakesh Shivanna; Archita Vadali; Ramona Comanescu; Doug Reid; Joss Moore; Arvind Neelakantan; Michaël Sander; Jonathan Herzig; Aviv Rosenberg; Mostafa Dehghani; JD Choi; Michael Fink; Reid Hayes; Eric Ge; Shitao Weng; Chia-Hua Ho; John Karro; Kalpesh Krishna; Lam Nguyen Thiet; Amy Skerry-Ryan; Daniel Eppens; Marco Andreetto; Navin Sarma; Silvano Bonacina; Burcu Karagol Ayan; Megha Nawhal; Zhihao Shan; Mike Dusenberry; Shantanu Thakoor; Sagar Gubbi; Duc Dung Nguyen; Reut Tsarfaty; Samuel Albanie; Jovana Mitrović; Meet Gandhi; Bo-Juen Chen; Alessandro Epasto; Georgi Stephanov; Ye Jin; Samuel Gehman; Aida Amini; Jack Weber; Feryal Behbahani; Shawn Xu; Miltos Allamanis; Xi Chen; Myle Ott; Claire Sha; Michal Jastrzebski; Hang Qi; David Greene; Xinyi Wu; Abodunrinwa Toki; Daniel Vlasic; Jane Shapiro; Ragha Kotikalapudi; Zhe Shen; Takaaki Saeki; Sirui Xie; Albin Cassirer; Shikhar Bharadwaj; Tatsuya Kiyono; Srinadh Bhojanapalli; Elan Rosenfeld; Sam Ritter; Jieming Mao; João Gabriel Oliveira; Zoltan Egyed; Bernd Bandemer; Emilio Parisotto; Keisuke Kinoshita; Juliette Pluto; Petros Maniatis; Steve Li; Yaohui Guo; Golnaz Ghiasi; Jean Tarbouriech; Srimon Chatterjee; Julie Jin; Katrina; Xu; Jennimaria Palomaki; Séb Arnold; Madhavi Sewak; Federico Piccinini; Mohit Sharma; Ben Albrecht; Sean Purser-haskell; Ashwin Vaswani; Chongyan Chen; Matheus Wisniewski; Qin Cao; John Aslanides; Nguyet Minh Phu; Maximilian Sieb; Lauren Agubuzu; Anne Zheng; Daniel Sohn; Marco Selvi; Anders Andreassen; Krishan Subudhi; Prem Eruvbetine; Oliver Woodman; Tomas Mery; Sebastian Krause; Xiaoqi Ren; Xiao Ma; Jincheng Luo; Dawn Chen; Wei Fan; Henry Griffiths; Christian Schuler; Alice Li; Shujian Zhang; Jean-Michel Sarr; Shixin Luo; Riccardo Patana; Matthew Watson; Dani Naboulsi; Michael Collins; Sailesh Sidhwani; Emiel Hoogeboom; Sharon Silver; Emily Caveness; Xiaokai Zhao; Mikel Rodriguez; Maxine Deines; Libin Bai; Patrick Griffin; Marco Tagliasacchi; Emily Xue; Spandana Raj Babbula; Bo Pang; Nan Ding; Gloria Shen; Elijah Peake; Remi Crocker; Shubha Srinivas Raghvendra; Danny Swisher; Woohyun Han; Richa Singh; Ling Wu; Vladimir Pchelin; Tsendsuren Munkhdalai; Dana Alon; Geoff Bacon; Efren Robles; Jannis Bulian; Melvin Johnson; George Powell; Felipe Tiengo Ferreira; Yaoyiran Li; Frederik Benzing; Mihajlo Velimirović; Hubert Soyer; William Kong; Tony; Nguyên; Zhen Yang; Jeremiah Liu; Joost van Amersfoort; Daniel Gillick; Baochen Sun; Nathalie Rauschmayr; Katie Zhang; Serena Zhan; Tao Zhou; Alexey Frolov; Chengrun Yang; Denis Vnukov; Louis Rouillard; Hongji Li; Amol Mandhane; Nova Fallen; Rajesh Venkataraman; Clara Huiyi Hu; Jennifer Brennan; Jenny Lee; Jerry Chang; Martin Sundermeyer; Zhufeng Pan; Rosemary Ke; Simon Tong; Alex Fabrikant; William Bono; Jindong Gu; Ryan Foley; Yiran Mao; Manolis Delakis; Dhruva Bhaswar; Roy Frostig; Nick Li; Avital Zipori; Cath Hope; Olga Kozlova; Swaroop Mishra; Josip Djolonga; Craig Schiff; Majd Al Merey; Eleftheria Briakou; Peter Morgan; Andy Wan; Avinatan Hassidim; RJ Skerry-Ryan; Kuntal Sengupta; Mary Jasarevic; Praveen Kallakuri; Paige Kunkle; Hannah Brennan; Tom Lieber; Hassan Mansoor; Julian Walker; Bing Zhang; Annie Xie; Goran Žužić; Adaeze Chukwuka; Alex Druinsky; Donghyun Cho; Rui Yao; Ferjad Naeem; Shiraz Butt; Eunyoung Kim; Zhipeng Jia; Mandy Jordan; Adam Lelkes; Mark Kurzeja; Sophie Wang; James Zhao; Andrew Over; Abhishek Chakladar; Marcel Prasetya; Neha Jha; Sriram Ganapathy; Yale Cong; Prakash Shroff; Carl Saroufim; Sobhan Miryoosefi; Mohamed Hammad; Tajwar Nasir; Weijuan Xi; Yang Gao; Young Maeng; Ben Hora; Chin-Yi Cheng; Parisa Haghani; Yoad Lewenberg; Caden Lu; Martin Matysiak; Naina Raisinghani; Huiyu Wang; Lexi Baugher; Rahul Sukthankar; Minh Giang; John Schultz; Noah Fiedel; Minmin Chen; Cheng-Chun Lee; Tapomay Dey; Hao Zheng; Shachi Paul; Celine Smith; Andy Ly; Yicheng Wang; Rishabh Bansal; Bartek Perz; Susanna Ricco; Stasha Blank; Vaishakh Keshava; Deepak Sharma; Marvin Chow; Kunal Lad; Komal Jalan; Simon Osindero; Craig Swanson; Jacob Scott; Anastasija Ilić; Xiaowei Li; Siddhartha Reddy Jonnalagadda; Afzal Shama Soudagar; Yan Xiong; Bat-Orgil Batsaikhan; Daniel Jarrett; Naveen Kumar; Maulik Shah; Matt Lawlor; Austin Waters; Mark Graham; Rhys May; Sabela Ramos; Sandra Lefdal; Zeynep Cankara; Nacho Cano; Brendan O'Donoghue; Jed Borovik; Frederick Liu; Jordan Grimstad; Mahmoud Alnahlawi; Katerina Tsihlas; Tom Hudson; Nikolai Grigorev; Yiling Jia; Terry Huang; Tobenna Peter Igwe; Sergei Lebedev; Xiaodan Tang; Igor Krivokon; Frankie Garcia; Melissa Tan; Eric Jia; Peter Stys; Shikhar Vashishth; Yu Liang; Balaji Venkatraman; Chenjie Gu; Anastasios Kementsietsidis; Chen Zhu; Junehyuk Jung; Yunfei Bai; Mohammad Javad Hosseini; Faruk Ahmed; Aditya Gupta; Xin Yuan; Shereen Ashraf; Shitij Nigam; Gautam Vasudevan; Pranjal Awasthi; Adi Mayrav Gilady; Zelda Mariet; Ramy Eskander; Haiguang Li; Hexiang Hu; Guillermo Garrido; Philippe Schlattner; George Zhang; Rohun Saxena; Petar Dević; Kritika Muralidharan; Ashwin Murthy; Yiqian Zhou; Min Choi; Arissa Wongpanich; Zhengdong Wang; Premal Shah; Yuntao Xu; Yiling Huang; Stephen Spencer; Alice Chen; James Cohan; Junjie Wang; Jonathan Tompson; Junru Wu; Ruba Haroun; Haiqiong Li; Blanca Huergo; Fan Yang; Tongxin Yin; James Wendt; Michael Bendersky; Rahma Chaabouni; Javier Snaider; Johan Ferret; Abhishek Jindal; Tara Thompson; Andrew Xue; Will Bishop; Shubham Milind Phal; Archit Sharma; Yunhsuan Sung; Prabakar Radhakrishnan; Mo Shomrat; Reeve Ingle; Roopali Vij; Justin Gilmer; Mihai Dorin Istin; Sam Sobell; Yang Lu; Emily Nottage; Dorsa Sadigh; Jeremiah Willcock; Tingnan Zhang; Steve Xu; Sasha Brown; Katherine Lee; Gary Wang; Yun Zhu; Yi Tay; Cheolmin Kim; Audrey Gutierrez; Abhanshu Sharma; Yongqin Xian; Sungyong Seo; Claire Cui; Elena Pochernina; Cip Baetu; Krzysztof Jastrzębski; Mimi Ly; Mohamed Elhawaty; Dan Suh; Eren Sezener; Pidong Wang; Nancy Yuen; George Tucker; Jiahao Cai; Zuguang Yang; Cindy Wang; Alex Muzio; Hai Qian; Jae Yoo; Derek Lockhart; Kevin R. McKee; Mandy Guo; Malika Mehrotra; Artur Mendonça; Sanket Vaibhav Mehta; Sherry Ben; Chetan Tekur; Jiaqi Mu; Muye Zhu; Victoria Krakovna; Hongrae Lee; AJ Maschinot; Sébastien Cevey; HyunJeong Choe; Aijun Bai; Hansa Srinivasan; Derek Gasaway; Nick Young; Patrick Siegler; Dan Holtmann-Rice; Vihari Piratla; Kate Baumli; Roey Yogev; Alex Hofer; Hado van Hasselt; Svetlana Grant; Yuri Chervonyi; David Silver; Andrew Hogue; Ayushi Agarwal; Kathie Wang; Preeti Singh; Four Flynn; Josh Lipschultz; Robert David; Lizzetth Bellot; Yao-Yuan Yang; Long Le; Filippo Graziano; Kate Olszewska; Kevin Hui; Akanksha Maurya; Nikos Parotsidis; Weijie Chen; Tayo Oguntebi; Joe Kelley; Anirudh Baddepudi; Johannes Mauerer; Gregory Shaw; Alex Siegman; Lin Yang; Shravya Shetty; Subhrajit Roy; Yunting Song; Wojciech Stokowiec; Ryan Burnell; Omkar Savant; Robert Busa-Fekete; Jin Miao; Samrat Ghosh; Liam MacDermed; Phillip Lippe; Mikhail Dektiarev; Zach Behrman; Fabian Mentzer; Kelvin Nguyen; Meng Wei; Siddharth Verma; Chris Knutsen; Sudeep Dasari; Zhipeng Yan; Petr Mitrichev; Xingyu Wang; Virat Shejwalkar; Jacob Austin; Srinivas Sunkara; Navneet Potti; Yan Virin; Christian Wright; Gaël Liu; Oriana Riva; Etienne Pot; Greg Kochanski; Quoc Le; Gargi Balasubramaniam; Arka Dhar; Yuguo Liao; Adam Bloniarz; Divyansh Shukla; Elizabeth Cole; Jong Lee; Sheng Zhang; Sushant Kafle; Siddharth Vashishtha; Parsa Mahmoudieh; Grace Chen; Raphael Hoffmann; Pranesh Srinivasan; Agustin Dal Lago; Yoav Ben Shalom; Zi Wang; Michael Elabd; Anuj Sharma; Junhyuk Oh; Suraj Kothawade; Maigo Le; Marianne Monteiro; Shentao Yang; Kaiz Alarakyia; Robert Geirhos; Diana Mincu; Håvard Garnes; Hayato Kobayashi; Soroosh Mariooryad; Kacper Krasowiak; Zhixin; Lai; Shibl Mourad; Mingqiu Wang; Fan Bu; Ophir Aharoni; Guanjie Chen; Abhimanyu Goyal; Vadim Zubov; Ankur Bapna; Elahe Dabir; Nisarg Kothari; Kay Lamerigts; Nicola De Cao; Jeremy Shar; Christopher Yew; Nitish Kulkarni; Dre Mahaarachchi; Mandar Joshi; Zhenhai Zhu; Jared Lichtarge; Yichao Zhou; Hannah Muckenhirn; Vittorio Selo; Oriol Vinyals; Peter Chen; Anthony Brohan; Vaibhav Mehta; Sarah Cogan; Ruth Wang; Ty Geri; Wei-Jen Ko; Wei Chen; Fabio Viola; Keshav Shivam; Lisa Wang; Madeleine Clare Elish; Raluca Ada Popa; Sébastien Pereira; Jianqiao Liu; Raphael Koster; Donnie Kim; Gufeng Zhang; Sayna Ebrahimi; Partha Talukdar; Yanyan Zheng; Petra Poklukar; Ales Mikhalap; Dale Johnson; Anitha Vijayakumar; Mark Omernick; Matt Dibb; Ayush Dubey; Qiong Hu; Apurv Suman; Vaibhav Aggarwal; Ilya Kornakov; Fei Xia; Wing Lowe; Alexey Kolganov; Ted Xiao; Vitaly Nikolaev; Steven Hemingray; Bonnie Li; Joana Iljazi; Mikołaj Rybiński; Ballie Sandhu; Peggy Lu; Thang Luong; Rodolphe Jenatton; Vineetha Govindaraj; Hui; Li; Gabriel Dulac-Arnold; Wonpyo Park; Henry Wang; Abhinit Modi; Jean Pouget-Abadie; Kristina Greller; Rahul Gupta; Robert Berry; Prajit Ramachandran; Jinyu Xie; Liam McCafferty; Jianling Wang; Kilol Gupta; Hyeontaek Lim; Blaž Bratanič; Andy Brock; Ilia Akolzin; Jim Sproch; Dan Karliner; Duhyeon Kim; Adrian Goedeckemeyer; Noam Shazeer; Cordelia Schmid; Daniele Calandriello; Parul Bhatia; Krzysztof Choromanski; Ceslee Montgomery; Dheeru Dua; Ana Ramalho; Helen King; Yue Gao; Lynn Nguyen; David Lindner; Divya Pitta; Oleaser Johnson; Khalid Salama; Diego Ardila; Michael Han; Erin Farnese; Seth Odoom; Ziyue Wang; Xiangzhuo Ding; Norman Rink; Ray Smith; Harshal Tushar Lehri; Eden Cohen; Neera Vats; Tong He; Parthasarathy Gopavarapu; Adam Paszke; Miteyan Patel; Wouter Van Gansbeke; Lucia Loher; Luis Castro; Maria Voitovich; Tamara von Glehn; Nelson George; Simon Niklaus; Zach Eaton-Rosen; Nemanja Rakićević; Erik Jue; Sagi Perel; Carrie Zhang; Yuval Bahat; Angéline Pouget; Zhi Xing; Fantine Huot; Ashish Shenoy; Taylor Bos; Vincent Coriou; Bryan Richter; Natasha Noy; Yaqing Wang; Santiago Ontanon; Siyang Qin; Gleb Makarchuk; Demis Hassabis; Zhuowan Li; Mandar Sharma; Kumaran Venkatesan; Iurii Kemaev; Roxanne Daniel; Shiyu Huang; Saloni Shah; Octavio Ponce; Warren; Chen; Manaal Faruqui; Jialin Wu; Slavica Andačić; Szabolcs Payrits; Daniel McDuff; Tom Hume; Yuan Cao; MH Tessler; Qingze Wang; Yinan Wang; Ivor Rendulic; Eirikur Agustsson; Matthew Johnson; Tanya Lando; Andrew Howard; Sri Gayatri Sundara Padmanabhan; Mayank Daswani; Andrea Banino; Michael Kilgore; Jonathan Heek; Ziwei Ji; Alvaro Caceres; Conglong Li; Nora Kassner; Alexey Vlaskin; Zeyu Liu; Alex Grills; Yanhan Hou; Roykrong Sukkerd; Gowoon Cheon; Nishita Shetty; Larisa Markeeva; Piotr Stanczyk; Tejas Iyer; Yuan Gong; Shawn Gao; Keerthana Gopalakrishnan; Tim Blyth; Malcolm Reynolds; Avishkar Bhoopchand; Misha Bilenko; Dero Gharibian; Vicky Zayats; Aleksandra Faust; Abhinav Singh; Min Ma; Hongyang Jiao; Sudheendra Vijayanarasimhan; Lora Aroyo; Vikas Yadav; Sarah Chakera; Ashwin Kakarla; Vilobh Meshram; Karol Gregor; Gabriela Botea; Evan Senter; Dawei Jia; Geza Kovacs; Neha Sharma; Sebastien Baur; Kai Kang; Yifan He; Lin Zhuo; Marija Kostelac; Itay Laish; Songyou Peng; Louis O'Bryan; Daniel Kasenberg; Girish Ramchandra Rao; Edouard Leurent; Biao Zhang; Sage Stevens; Ana Salazar; Ye Zhang; Ivan Lobov; Jake Walker; Allen Porter; Morgan Redshaw; Han Ke; Abhishek Rao; Alex Lee; Hoi Lam; Michael Moffitt; Jaeyoun Kim; Siyuan Qiao; Terry Koo; Robert Dadashi; Xinying Song; Mukund Sundararajan; Peng Xu; Chizu Kawamoto; Yan Zhong; Clara Barbu; Apoorv Reddy; Mauro Verzetti; Leon Li; George Papamakarios; Hanna Klimczak-Plucińska; Mary Cassin; Koray Kavukcuoglu; Rigel Swavely; Alain Vaucher; Jeffrey Zhao; Ross Hemsley; Michael Tschannen; Heming Ge; Gaurav Menghani; Yang Yu; Natalie Ha; Wei He; Xiao Wu; Maggie Song; Rachel Sterneck; Stefan Zinke; Dan A. Calian; Annie Marsden; Alejandro Cruzado Ruiz; Matteo Hessel; Almog Gueta; Benjamin Lee; Brian Farris; Manish Gupta; Yunjie Li; Mohammad Saleh; Vedant Misra; Kefan Xiao; Piermaria Mendolicchio; Gavin Buttimore; Varvara Krayvanova; Nigamaa Nayakanti; Matthew Wiethoff; Yash Pande; Azalia Mirhoseini; Ni Lao; Jasmine Liu; Yiqing Hua; Angie Chen; Yury Malkov; Dmitry Kalashnikov; Shubham Gupta; Kartik Audhkhasi; Yuexiang Zhai; Sudhindra Kopalle; Prateek Jain; Eran Ofek; Clemens Meyer; Khuslen Baatarsukh; Hana Strejček; Jun Qian; James Freedman; Ricardo Figueira; Michal Sokolik; Olivier Bachem; Raymond Lin; Dia Kharrat; Chris Hidey; Pingmei Xu; Dennis Duan; Yin Li; Muge Ersoy; Richard Everett; Kevin Cen; Rebeca Santamaria-Fernandez; Amir Taubenfeld; Ian Mackinnon; Linda Deng; Polina Zablotskaia; Shashank Viswanadha; Shivanker Goel; Damion Yates; Yunxiao Deng; Peter Choy; Mingqing Chen; Abhishek Sinha; Alex Mossin; Yiming Wang; Arthur Szlam; Susan Hao; Paul Kishan Rubenstein; Metin Toksoz-Exley; Miranda Aperghis; Yin Zhong; Junwhan Ahn; Michael Isard; Olivier Lacombe; Florian Luisier; Chrysovalantis Anastasiou; Yogesh Kalley; Utsav Prabhu; Emma Dunleavy; Shaan Bijwadia; Justin Mao-Jones; Kelly Chen; Rama Pasumarthi; Emily Wood; Adil Dostmohamed; Nate Hurley; Jiri Simsa; Alicia Parrish; Mantas Pajarskas; Matt Harvey; Ondrej Skopek; Yony Kochinski; Javier Rey; Verena Rieser; Denny Zhou; Sun Jae Lee; Trilok Acharya; Guowang Li; Joe Jiang; Xiaofan Zhang; Bryant Gipson; Ethan Mahintorabi; Marco Gelmi; Nima Khajehnouri; Angel Yeh; Kayi Lee; Loic Matthey; Leslie Baker; Trang Pham; Han Fu; Alex Pak; Prakhar Gupta; Cristina Vasconcelos; Adam Sadovsky; Brian Walker; Sissie Hsiao; Patrik Zochbauer; Andreea Marzoca; Noam Velan; Junhao Zeng; Gilles Baechler; Danny Driess; Divya Jain; Yanping Huang; Lizzie Tao; John Maggs; Nir Levine; Jon Schneider; Erika Gemzer; Samuel Petit; Shan Han; Zach Fisher; Dustin Zelle; Courtney Biles; Eugene Ie; Asya Fadeeva; Casper Liu; Juliana Vicente Franco; Adrian Collister; Hao Zhang; Renshen Wang; Ruizhe Zhao; Leandro Kieliger; Kurt Shuster; Rui Zhu; Boqing Gong; Lawrence Chan; Ruoxi Sun; Sujoy Basu; Roland Zimmermann; Jamie Hayes; Abhishek Bapna; Jasper Snoek; Weel Yang; Puranjay Datta; Jad Al Abdallah; Kevin Kilgour; Lu Li; SQ Mah; Yennie Jun; Morgane Rivière; Abhijit Karmarkar; Tammo Spalink; Tao Huang; Lucas Gonzalez; Duc-Hieu Tran; Averi Nowak; John Palowitch; Martin Chadwick; Ellie Talius; Harsh Mehta; Thibault Sellam; Philipp Fränken; Massimo Nicosia; Kyle He; Aditya Kini; David Amos; Sugato Basu; Harrison Jobe; Eleni Shaw; Qiantong Xu; Colin Evans; Daisuke Ikeda; Chaochao Yan; Larry Jin; Lun Wang; Sachin Yadav; Ilia Labzovsky; Ramesh Sampath; Ada Ma; Candice Schumann; Aditya Siddhant; Rohin Shah; John Youssef; Rishabh Agarwal; Natalie Dabney; Alessio Tonioni; Moran Ambar; Jing Li; Isabelle Guyon; Benny Li; David Soergel; Boya Fang; Georgi Karadzhov; Cristian Udrescu; Trieu Trinh; Vikas Raunak; Seb Noury; Dee Guo; Sonal Gupta; Mara Finkelstein; Denis Petek; Lihao Liang; Greg Billock; Pei Sun; David Wood; Yiwen Song; Xiaobin Yu; Tatiana Matejovicova; Regev Cohen; Kalyan Andra; David D'Ambrosio; Zhiwei Deng; Vincent Nallatamby; Ebrahim Songhori; Rumen Dangovski; Andrew Lampinen; Pankil Botadra; Adam Hillier; Jiawei Cao; Nagabhushan Baddi; Adhi Kuncoro; Toshihiro Yoshino; Ankit Bhagatwala; Marcáurelio Ranzato; Rylan Schaeffer; Tianlin Liu; Shuai Ye; Obaid Sarvana; John Nham; Chenkai Kuang; Isabel Gao; Jinoo Baek; Shubham Mittal; Ayzaan Wahid; Anita Gergely; Bin Ni; Josh Feldman; Carrie Muir; Pascal Lamblin; Wolfgang Macherey; Ethan Dyer; Logan Kilpatrick; Víctor Campos; Mukul Bhutani; Stanislav Fort; Yanif Ahmad; Aliaksei Severyn; Kleopatra Chatziprimou; Oleksandr Ferludin; Mason Dimarco; Aditya Kusupati; Joe Heyward; Dan Bahir; Kevin Villela; Katie Millican; Dror Marcus; Sanaz Bahargam; Caglar Unlu; Nicholas Roth; Zichuan Wei; Siddharth Gopal; Deepanway Ghoshal; Edward Lee; Sharon Lin; Jennie Lees; Dayeong Lee; Anahita Hosseini; Connie Fan; Seth Neel; Marcus Wu; Yasemin Altun; Honglong Cai; Enrique Piqueras; Josh Woodward; Alessandro Bissacco; Salem Haykal; Mahyar Bordbar; Prasha Sundaram; Sarah Hodkinson; Daniel Toyama; George Polovets; Austin Myers; Anu Sinha; Tomer Levinboim; Kashyap Krishnakumar; Rachita Chhaparia; Tatiana Sholokhova; Nitesh Bharadwaj Gundavarapu; Ganesh Jawahar; Haroon Qureshi; Jieru Hu; Nikola Momchev; Matthew Rahtz; Renjie Wu; Aishwarya P S; Kedar Dhamdhere; Meiqi Guo; Umang Gupta; Ali Eslami; Mariano Schain; Michiel Blokzijl; David Welling; Dave Orr; Levent Bolelli; Nicolas Perez-Nieves; Mikhail Sirotenko; Aman Prasad; Arjun Kar; Borja De Balle Pigem; Tayfun Terzi; Gellért Weisz; Dipankar Ghosh; Aditi Mavalankar; Dhruv Madeka; Kaspar Daugaard; Hartwig Adam; Viraj Shah; Dana Berman; Maggie Tran; Steven Baker; Ewa Andrejczuk; Grishma Chole; Ganna Raboshchuk; Mahdi Mirzazadeh; Thais Kagohara; Shimu Wu; Christian Schallhart; Bernett Orlando; Chen Wang; Alban Rrustemi; Hao Xiong; Hao Liu; Arpi Vezer; Nolan Ramsden; Shuo-yiin Chang; Sidharth Mudgal; Yan Li; Nino Vieillard; Yedid Hoshen; Farooq Ahmad; Ambrose Slone; Amy Hua; Natan Potikha; Mirko Rossini; Jon Stritar; Sushant Prakash; Zifeng Wang; Xuanyi Dong; Alireza Nazari; Efrat Nehoran; Kaan Tekelioglu; Yinxiao Li; Kartikeya Badola; Tom Funkhouser; Yuanzhen Li; Varun Yerram; Ramya Ganeshan; Daniel Formoso; Karol Langner; Tian Shi; Huijian Li; Yumeya Yamamori; Amayika Panda; Alaa Saade; Angelo Scorza Scarpati; Chris Breaux; CJ Carey; Zongwei Zhou; Cho-Jui Hsieh; Sophie Bridgers; Alena Butryna; Nishesh Gupta; Vaibhav Tulsyan; Sanghyun Woo; Evgenii Eltyshev; Will Grathwohl; Chanel Parks; Seth Benjamin; Rina Panigrahy; Shenil Dodhia; Daniel De Freitas; Chris Sauer; Will Song; Ferran Alet; Jackson Tolins; Cosmin Paduraru; Xingyi Zhou; Brian Albert; Zizhao Zhang; Lei Shu; Mudit Bansal; Sarah Nguyen; Amir Globerson; Owen Xiao; James Manyika; Tom Hennigan; Rong Rong; Josip Matak; Anton Bakalov; Ankur Sharma; Danila Sinopalnikov; Andrew Pierson; Stephen Roller; Geoff Brown; Mingcen Gao; Toshiyuki Fukuzawa; Amin Ghafouri; Kenny Vassigh; Iain Barr; Zhicheng Wang; Anna Korsun; Rajesh Jayaram; Lijie Ren; Tim Zaman; Samira Khan; Yana Lunts; Dan Deutsch; Dave Uthus; Nitzan Katz; Masha Samsikova; Amr Khalifa; Nikhil Sethi; Jiao Sun; Luming Tang; Uri Alon; Xianghong Luo; Dian Yu; Abhishek Nayyar; Bryce Petrini; Will Truong; Vincent Hellendoorn; Nikolai Chinaev; Chris Alberti; Wei Wang; Jingcao Hu; Vahab Mirrokni; Ananth Balashankar; Avia Aharon; Aahil Mehta; Ahmet Iscen; Joseph Kready; Lucas Manning; Anhad Mohananey; Yuankai Chen; Anshuman Tripathi; Allen Wu; Igor Petrovski; Dawsen Hwang; Martin Baeuml; Shreyas Chandrakaladharan; Yuan Liu; Rey Coaguila; Maxwell Chen; Sally Ma; Pouya Tafti; Susheel Tatineni; Terry Spitz; Jiayu Ye; Paul Vicol; Mihaela Rosca; Adrià Puigdomènech; Zohar Yahav; Sanjay Ghemawat; Hanzhao Lin; Phoebe Kirk; Zaid Nabulsi; Sergey Brin; Bernd Bohnet; Ken Caluwaerts; Aditya Srikanth Veerubhotla; Dan Zheng; Zihang Dai; Petre Petrov; Yichong Xu; Ramin Mehran; Zhuo Xu; Luisa Zintgraf; Jiho Choi; Spurthi Amba Hombaiah; Romal Thoppilan; Sashank Reddi; Lukasz Lew; Li Li; Kellie Webster; KP Sawhney; Lampros Lamprou; Siamak Shakeri; Mayank Lunayach; Jianmin Chen; Sumit Bagri; Alex Salcianu; Ying Chen; Yani Donchev; Charlotte Magister; Signe Nørly; Vitor Rodrigues; Tomas Izo; Hila Noga; Joe Zou; Thomas Köppe; Wenxuan Zhou; Kenton Lee; Xiangzhu Long; Danielle Eisenbud; Anthony Chen; Connor Schenck; Chi Ming To; Peilin Zhong; Emanuel Taropa; Minh Truong; Omer Levy; Danilo Martins; Zhiyuan Zhang; Christopher Semturs; Kelvin Zhang; Alex Yakubovich; Pol Moreno; Lara McConnaughey; Di Lu; Sam Redmond; Lotte Weerts; Yonatan Bitton; Tiziana Refice; Nicolas Lacasse; Arthur Conmy; Corentin Tallec; Julian Odell; Hannah Forbes-Pollard; Arkadiusz Socala; Jonathan Hoech; Pushmeet Kohli; Alanna Walton; Rui Wang; Mikita Sazanovich; Kexin Zhu; Andrei Kapishnikov; Rich Galt; Matthew Denton; Ben Murdoch; Caitlin Sikora; Kareem Mohamed; Wei Wei; Uri First; Tim McConnell; Luis C. Cobo; James Qin; Thi Avrahami; Daniel Balle; Yu Watanabe; Annie Louis; Adam Kraft; Setareh Ariafar; Yiming Gu; Eugénie Rives; Charles Yoon; Andrei Rusu; James Cobon-Kerr; Chris Hahn; Jiaming Luo; Yuvein; Zhu; Niharika Ahuja; Rodrigo Benenson; Raphaël Lopez Kaufman; Honglin Yu; Lloyd Hightower; Junlin Zhang; Darren Ni; Lisa Anne Hendricks; Gabby Wang; Gal Yona; Lalit Jain; Pablo Barrio; Surya Bhupatiraju; Siva Velusamy; Allan Dafoe; Sebastian Riedel; Tara Thomas; Zhe Yuan; Mathias Bellaiche; Sheena Panthaplackel; Klemen Kloboves; Sarthak Jauhari; Canfer Akbulut; Todor Davchev; Evgeny Gladchenko; David Madras; Aleksandr Chuklin; Tyrone Hill; Quan Yuan; Mukundan Madhavan; Luke Leonhard; Dylan Scandinaro; Qihang Chen; Ning Niu; Arthur Douillard; Bogdan Damoc; Yasumasa Onoe; Fabian Pedregosa; Fred Bertsch; Chas Leichner; Joseph Pagadora; Jonathan Malmaud; Sameera Ponda; Andy Twigg; Oleksii Duzhyi; Jingwei Shen; Miaosen Wang; Roopal Garg; Jing Chen; Utku Evci; Jonathan Lee; Leon Liu; Koji Kojima; Masa Yamaguchi; Arunkumar Rajendran; AJ Piergiovanni; Vinodh Kumar Rajendran; Marco Fornoni; Gabriel Ibagon; Harry Ragan; Sadh MNM Khan; John Blitzer; Andrew Bunner; Guan Sun; Takahiro Kosakai; Scott Lundberg; Ndidi Elue; Kelvin Guu; SK Park; Jane Park; Arunachalam Narayanaswamy; Chengda Wu; Jayaram Mudigonda; Trevor Cohn; Hairong Mu; Ravi Kumar; Laura Graesser; Yichi Zhang; Richard Killam; Vincent Zhuang; Mai Giménez; Wael Al Jishi; Ruy Ley-Wild; Alex Zhai; Kazuki Osawa; Diego Cedillo; Jialu Liu; Mayank Upadhyay; Marcin Sieniek; Roshan Sharma; Tom Paine; Anelia Angelova; Sravanti Addepalli; Carolina Parada; Kingshuk Majumder; Avery Lamp; Sanjiv Kumar; Xiang Deng; Artiom Myaskovsky; Tea Sabolić; Jeffrey Dudek; Sarah York; Félix de Chaumont Quitry; Jiazhong Nie; Dee Cattle; Alok Gunjan; Bilal Piot; Waleed Khawaja; Seojin Bang; Simon Wang; Siavash Khodadadeh; Raghavender R; Praynaa Rawlani; Richard Powell; Kevin Lee; Johannes Griesser; GS Oh; Cesar Magalhaes; Yujia Li; Simon Tokumine; Hadas Natalie Vogel; Dennis Hsu; Arturo BC; Disha Jindal; Matan Cohen; Zi Yang; Junwei Yuan; Dario de Cesare; Tony Bruguier; Jun Xu; Monica Roy; Alon Jacovi; Dan Belov; Rahul Arya; Phoenix Meadowlark; Shlomi Cohen-Ganor; Wenting Ye; Patrick Morris-Suzuki; Praseem Banzal; Gan Song; Pranavaraj Ponnuramu; Fred Zhang; George Scrivener; Salah Zaiem; Alif Raditya Rochman; Kehang Han; Badih Ghazi; Kate Lee; Shahar Drath; Daniel Suo; Antonious Girgis; Pradeep Shenoy; Duy Nguyen; Douglas Eck; Somit Gupta; Le Yan; Joao Carreira; Anmol Gulati; Ruoxin Sang; Daniil Mirylenka; Emma Cooney; Edward Chou; Mingyang Ling; Cindy Fan; Ben Coleman; Guilherme Tubone; Ravin Kumar; Jason Baldridge; Felix Hernandez-Campos; Angeliki Lazaridou; James Besley; Itay Yona; Neslihan Bulut; Quentin Wellens; AJ Pierigiovanni; Jasmine George; Richard Green; Pu Han; Connie Tao; Geoff Clark; Chong You; Abbas Abdolmaleki; Justin Fu; Tongzhou Chen; Ashwin Chaugule; Angad Chandorkar; Altaf Rahman; Will Thompson; Penporn Koanantakool; Mike Bernico; Jie Ren; Andrey Vlasov; Sergei Vassilvitskii; Maciej Kula; Yizhong Liang; Dahun Kim; Yangsibo Huang; Chengxi Ye; Dmitry Lepikhin; Wesley Helmholz
>
> **备注:** 72 pages, 17 figures
>
> **摘要:** In this report, we introduce the Gemini 2.X model family: Gemini 2.5 Pro and Gemini 2.5 Flash, as well as our earlier Gemini 2.0 Flash and Flash-Lite models. Gemini 2.5 Pro is our most capable model yet, achieving SoTA performance on frontier coding and reasoning benchmarks. In addition to its incredible coding and reasoning skills, Gemini 2.5 Pro is a thinking model that excels at multimodal understanding and it is now able to process up to 3 hours of video content. Its unique combination of long context, multimodal and reasoning capabilities can be combined to unlock new agentic workflows. Gemini 2.5 Flash provides excellent reasoning abilities at a fraction of the compute and latency requirements and Gemini 2.0 Flash and Flash-Lite provide high performance at low latency and cost. Taken together, the Gemini 2.X model generation spans the full Pareto frontier of model capability vs cost, allowing users to explore the boundaries of what is possible with complex agentic problem solving.
>
---
#### [replaced 032] Drivel-ology: Challenging LLMs with Interpreting Nonsense with Depth
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.03867v3](http://arxiv.org/pdf/2509.03867v3)**

> **作者:** Yang Wang; Chenghao Xiao; Chia-Yi Hsiao; Zi Yan Chang; Chi-Li Chen; Tyler Loakman; Chenghua Lin
>
> **备注:** Accepted for oral presentation at the EMNLP 2025 Main Conference
>
> **摘要:** We introduce Drivelology, a unique linguistic phenomenon characterised as "nonsense with depth" - utterances that are syntactically coherent yet pragmatically paradoxical, emotionally loaded, or rhetorically subversive. While such expressions may resemble surface-level nonsense, they encode implicit meaning requiring contextual inference, moral reasoning, or emotional interpretation. We find that current large language models (LLMs), despite excelling at many natural language processing (NLP) tasks, consistently fail to grasp the layered semantics of Drivelological text. To investigate this, we construct a benchmark dataset of over 1,200+ meticulously curated and diverse examples across English, Mandarin, Spanish, French, Japanese, and Korean. Each example underwent careful expert review to verify its Drivelological characteristics, involving multiple rounds of discussion and adjudication to address disagreements. Using this dataset, we evaluate a range of LLMs on classification, generation, and reasoning tasks. Our results reveal clear limitations of LLMs: models often confuse Drivelology with shallow nonsense, produce incoherent justifications, or miss implied rhetorical functions altogether. These findings highlight a deep representational gap in LLMs' pragmatic understanding and challenge the assumption that statistical fluency implies cognitive comprehension. We release our dataset and code to facilitate further research in modelling linguistic depth beyond surface-level coherence.
>
---
#### [replaced 033] SteeringSafety: A Systematic Safety Evaluation Framework of Representation Steering in LLMs
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.13450v2](http://arxiv.org/pdf/2509.13450v2)**

> **作者:** Vincent Siu; Nicholas Crispino; David Park; Nathan W. Henry; Zhun Wang; Yang Liu; Dawn Song; Chenguang Wang
>
> **摘要:** We introduce SteeringSafety, a systematic framework for evaluating representation steering methods across seven safety perspectives spanning 17 datasets. While prior work highlights general capabilities of representation steering, we systematically explore safety perspectives including bias, harmfulness, hallucination, social behaviors, reasoning, epistemic integrity, and normative judgment. Our framework provides modularized building blocks for state-of-the-art steering methods, enabling unified implementation of DIM, ACE, CAA, PCA, and LAT with recent enhancements like conditional steering. Results on Gemma-2-2B, Llama-3.1-8B, and Qwen-2.5-7B reveal that strong steering performance depends critically on pairing of method, model, and specific perspective. DIM shows consistent effectiveness, but all methods exhibit substantial entanglement: social behaviors show highest vulnerability (reaching degradation as high as 76%), jailbreaking often compromises normative judgment, and hallucination steering unpredictably shifts political views. Our findings underscore the critical need for holistic safety evaluations.
>
---
#### [replaced 034] Higher-order interactions of multi-layer prompt
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.09394v2](http://arxiv.org/pdf/2510.09394v2)**

> **作者:** Ziyu Zheng; Yaming Yang; Ziyu Guan; Wei Zhao; Xinyan Huang; Weigang Lu
>
> **备注:** under review
>
> **摘要:** The "pre-train, prompt" paradigm has successfully evolved in representation learning. While current prompt-tuning methods often introduce learnable prompts, they predominantly treat prompts as isolated, independent components across different network layers. This overlooks the complex and synergistic higher-order interactions that exist between prompts at various hierarchical depths, consequently limiting the expressive power and semantic richness of the prompted model. To address this fundamental gap, we propose a novel framework that explicitly models the Higher-order Interactions of Multi-layer Prompt. Our approach conceptualizes prompts from different layers not as separate entities, but as a cohesive system where their inter-relationships are critical. We design an innovative interaction module that captures these sophisticated, non-linear correlations among multi-layer prompts, effectively modeling their cooperative effects. This allows the model to dynamically aggregate and refine prompt information across the network's depth, leading to a more integrated and powerful prompting strategy. Extensive experiments on eight benchmark datasets demonstrate that our method, by leveraging these higher-order interactions, consistently surpasses state-of-the-art prompt-tuning baselines. The performance advantage is particularly pronounced in few-shot scenarios, validating that capturing the intricate interplay between multi-layer prompts is key to unlocking more robust and generalizable representation learning.
>
---
#### [replaced 035] Revisiting Prompt Optimization with Large Reasoning Models-A Case Study on Event Extraction
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.07357v2](http://arxiv.org/pdf/2504.07357v2)**

> **作者:** Saurabh Srivastava; Ziyu Yao
>
> **摘要:** Large Reasoning Models (LRMs) such as DeepSeek-R1 and OpenAI o1 have demonstrated remarkable capabilities in various reasoning tasks. Their strong capability to generate and reason over intermediate thoughts has also led to arguments that they may no longer require extensive prompt engineering or optimization to interpret human instructions and produce accurate outputs. In this work, we aim to systematically study this open question, using the structured task of event extraction for a case study. We experimented with two LRMs (DeepSeek-R1 and o1) and two general-purpose Large Language Models (LLMs) (GPT-4o and GPT-4.5), when they were used as task models or prompt optimizers. Our results show that on tasks as complicated as event extraction, LRMs as task models still benefit from prompt optimization, and that using LRMs as prompt optimizers yields more effective prompts. Our finding also generalizes to tasks beyond event extraction. Finally, we provide an error analysis of common errors made by LRMs and highlight the stability and consistency of LRMs in refining task instructions and event guidelines.
>
---
#### [replaced 036] Interpretability and Transparency-Driven Detection and Transformation of Textual Adversarial Examples (IT-DT)
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2307.01225v2](http://arxiv.org/pdf/2307.01225v2)**

> **作者:** Bushra Sabir; M. Ali Babar; Sharif Abuadbba
>
> **摘要:** Transformer-based text classifiers like BERT, Roberta, T5, and GPT-3 have shown impressive performance in NLP. However, their vulnerability to adversarial examples poses a security risk. Existing defense methods lack interpretability, making it hard to understand adversarial classifications and identify model vulnerabilities. To address this, we propose the Interpretability and Transparency-Driven Detection and Transformation (IT-DT) framework. It focuses on interpretability and transparency in detecting and transforming textual adversarial examples. IT-DT utilizes techniques like attention maps, integrated gradients, and model feedback for interpretability during detection. This helps identify salient features and perturbed words contributing to adversarial classifications. In the transformation phase, IT-DT uses pre-trained embeddings and model feedback to generate optimal replacements for perturbed words. By finding suitable substitutions, we aim to convert adversarial examples into non-adversarial counterparts that align with the model's intended behavior while preserving the text's meaning. Transparency is emphasized through human expert involvement. Experts review and provide feedback on detection and transformation results, enhancing decision-making, especially in complex scenarios. The framework generates insights and threat intelligence empowering analysts to identify vulnerabilities and improve model robustness. Comprehensive experiments demonstrate the effectiveness of IT-DT in detecting and transforming adversarial examples. The approach enhances interpretability, provides transparency, and enables accurate identification and successful transformation of adversarial inputs. By combining technical analysis and human expertise, IT-DT significantly improves the resilience and trustworthiness of transformer-based text classifiers against adversarial attacks.
>
---
#### [replaced 037] iQUEST: An Iterative Question-Guided Framework for Knowledge Base Question Answering
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.01784v3](http://arxiv.org/pdf/2506.01784v3)**

> **作者:** Shuai Wang; Yinan Yu
>
> **备注:** Accepted to the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025), Main Track
>
> **摘要:** While Large Language Models (LLMs) excel at many natural language processing tasks, they often suffer from factual inaccuracies in knowledge-intensive scenarios. Integrating external knowledge resources, particularly knowledge graphs (KGs), provides a transparent and updatable foundation for more reliable reasoning. Knowledge Base Question Answering (KBQA), which queries and reasons over KGs, is central to this effort, especially for complex, multi-hop queries. However, multi-hop reasoning poses two key challenges: (1)~maintaining coherent reasoning paths, and (2)~avoiding prematurely discarding critical multi-hop connections. To address these issues, we introduce iQUEST, a question-guided KBQA framework that iteratively decomposes complex queries into simpler sub-questions, ensuring a structured and focused reasoning trajectory. Additionally, we integrate a Graph Neural Network (GNN) to look ahead and incorporate 2-hop neighbor information at each reasoning step. This dual approach strengthens the reasoning process, enabling the model to explore viable paths more effectively. Detailed experiments demonstrate the consistent improvement delivered by iQUEST across four benchmark datasets and four LLMs.
>
---
#### [replaced 038] All Code, No Thought: Current Language Models Struggle to Reason in Ciphered Language
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.09714v2](http://arxiv.org/pdf/2510.09714v2)**

> **作者:** Shiyuan Guo; Henry Sleight; Fabien Roger
>
> **备注:** Version 2: updated related works section on LLM steganography
>
> **摘要:** Detecting harmful AI actions is important as AI agents gain adoption. Chain-of-thought (CoT) monitoring is one method widely used to detect adversarial attacks and AI misalignment. However, attackers and misaligned models might evade CoT monitoring through ciphered reasoning: reasoning hidden in encrypted, translated, or compressed text. To assess this risk, we test whether models can perform ciphered reasoning. For each of 28 different ciphers, we fine-tune and prompt up to 10 models to reason in that cipher. We measure model accuracy on math problems as a proxy for reasoning ability. Across the models we test, we find an asymmetry: model accuracy can drop significantly when reasoning in ciphered text, even though models demonstrate comprehension of ciphered text by being able to translate it accurately to English. Even frontier models struggle with lesser-known ciphers, although they can reason accurately in well-known ciphers like rot13. We show that ciphered reasoning capability correlates with cipher prevalence in pretraining data. We also identify scaling laws showing that ciphered reasoning capability improves slowly with additional fine-tuning data. Our work suggests that evading CoT monitoring using ciphered reasoning may be an ineffective tactic for current models and offers guidance on constraining the development of this capability in future frontier models.
>
---
#### [replaced 039] On Theoretical Interpretations of Concept-Based In-Context Learning
- **分类: cs.IT; cs.AI; cs.CL; math.IT**

- **链接: [http://arxiv.org/pdf/2509.20882v2](http://arxiv.org/pdf/2509.20882v2)**

> **作者:** Huaze Tang; Tianren Peng; Shao-lun Huang
>
> **摘要:** In-Context Learning (ICL) has emerged as an important new paradigm in natural language processing and large language model (LLM) applications. However, the theoretical understanding of the ICL mechanism remains limited. This paper aims to investigate this issue by studying a particular ICL approach, called concept-based ICL (CB-ICL). In particular, we propose theoretical analyses on applying CB-ICL to ICL tasks, which explains why and when the CB-ICL performs well for predicting query labels in prompts with only a few demonstrations. In addition, the proposed theory quantifies the knowledge that can be leveraged by the LLMs to the prompt tasks, and leads to a similarity measure between the prompt demonstrations and the query input, which provides important insights and guidance for model pre-training and prompt engineering in ICL. Moreover, the impact of the prompt demonstration size and the dimension of the LLM embeddings in ICL are also explored based on the proposed theory. Finally, several real-data experiments are conducted to validate the practical usefulness of CB-ICL and the corresponding theory.
>
---
#### [replaced 040] Prompt Perturbations Reveal Human-Like Biases in Large Language Model Survey Responses
- **分类: cs.CL; cs.AI; cs.CY; J.4**

- **链接: [http://arxiv.org/pdf/2507.07188v3](http://arxiv.org/pdf/2507.07188v3)**

> **作者:** Jens Rupprecht; Georg Ahnert; Markus Strohmaier
>
> **摘要:** Large Language Models (LLMs) are increasingly used as proxies for human subjects in social science surveys, but their reliability and susceptibility to known human-like response biases, such as central tendency, opinion floating and primacy bias are poorly understood. This work investigates the response robustness of LLMs in normative survey contexts, we test nine LLMs on questions from the World Values Survey (WVS), applying a comprehensive set of ten perturbations to both question phrasing and answer option structure, resulting in over 167,000 simulated survey interviews. In doing so, we not only reveal LLMs' vulnerabilities to perturbations but also show that all tested models exhibit a consistent recency bias, disproportionately favoring the last-presented answer option. While larger models are generally more robust, all models remain sensitive to semantic variations like paraphrasing and to combined perturbations. This underscores the critical importance of prompt design and robustness testing when using LLMs to generate synthetic survey data.
>
---
#### [replaced 041] EviNote-RAG: Enhancing RAG Models via Answer-Supportive Evidence Notes
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.00877v3](http://arxiv.org/pdf/2509.00877v3)**

> **作者:** Yuqin Dai; Guoqing Wang; Yuan Wang; Kairan Dou; Kaichen Zhou; Zhanwei Zhang; Shuo Yang; Fei Tang; Jun Yin; Pengyu Zeng; Zhenzhe Ying; Can Yi; Changhua Meng; Yuchen Zhou; Yongliang Shen; Shuai Lu
>
> **摘要:** Retrieval-Augmented Generation (RAG) has advanced open-domain question answering by incorporating external information into model reasoning. However, effectively leveraging external information to enhance reasoning presents the following challenges: (1) low signal-to-noise ratio, where answer-supportive external information is diluted by irrelevant material, and (2) error accumulation, which arises in multi-hop reasoning when incomplete or misleading information is incorporated. To address these challenges, we introduce EviNote-RAG, a framework that follows a retrieve-note-answer workflow. Instead of reasoning directly over raw external information, the model first produces Supportive-Evidence Notes (SENs), which concisely preserve answer-critical information and explicitly mark key and uncertainty information to improve accuracy. We further design an entailment-based Evidence Quality Reward (EQR) to ensure that SENs are logically sufficient to derive the final answer, thereby enhancing SENs' quality. Experiments on both in-domain and out-of-domain QA benchmarks show that EviNote-RAG achieves state-of-the-art performance, improving answer accuracy, training stability, robustness, and efficiency. In particular, it yields relative F1 gains of 20% on HotpotQA (+0.093), 40% on Bamboogle (+0.151), and 91% on 2Wiki (+0.256), benefiting from improvements in the reasoning process.
>
---
#### [replaced 042] Beyond the Surface: Enhancing LLM-as-a-Judge Alignment with Human via Internal Representations
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.03550v3](http://arxiv.org/pdf/2508.03550v3)**

> **作者:** Peng Lai; Jianjie Zheng; Sijie Cheng; Yun Chen; Peng Li; Yang Liu; Guanhua Chen
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** The growing scale of evaluation tasks has led to the widespread adoption of automated evaluation using LLMs, a paradigm known as "LLM-as-a-judge". However, improving its alignment with human preferences without complex prompts or fine-tuning remains challenging. Previous studies mainly optimize based on shallow outputs, overlooking rich cross-layer representations. In this work, motivated by preliminary findings that middle-to-upper layers encode semantically and task-relevant representations that are often more aligned with human judgments than the final layer, we propose LAGER, a post-hoc, plug-and-play framework for improving the alignment of LLM-as-a-Judge point-wise evaluations with human scores by leveraging internal representations. LAGER produces fine-grained judgment scores by aggregating cross-layer score-token logits and computing the expected score from a softmax-based distribution, while keeping the LLM backbone frozen and ensuring no impact on the inference process. LAGER fully leverages the complementary information across different layers, overcoming the limitations of relying solely on the final layer. We evaluate our method on the standard alignment benchmarks Flask, HelpSteer, and BIGGen using Spearman correlation, and find that LAGER achieves improvements of up to 7.5% over the best baseline across these benchmarks. Without reasoning steps, LAGER matches or outperforms reasoning-based methods. Experiments on downstream applications, such as data selection and emotional understanding, further show the generalization of LAGER.
>
---
#### [replaced 043] Paper2Agent: Reimagining Research Papers As Interactive and Reliable AI Agents
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.06917v2](http://arxiv.org/pdf/2509.06917v2)**

> **作者:** Jiacheng Miao; Joe R. Davis; Yaohui Zhang; Jonathan K. Pritchard; James Zou
>
> **摘要:** We introduce Paper2Agent, an automated framework that converts research papers into AI agents. Paper2Agent transforms research output from passive artifacts into active systems that can accelerate downstream use, adoption, and discovery. Conventional research papers require readers to invest substantial effort to understand and adapt a paper's code, data, and methods to their own work, creating barriers to dissemination and reuse. Paper2Agent addresses this challenge by automatically converting a paper into an AI agent that acts as a knowledgeable research assistant. It systematically analyzes the paper and the associated codebase using multiple agents to construct a Model Context Protocol (MCP) server, then iteratively generates and runs tests to refine and robustify the resulting MCP. These paper MCPs can then be flexibly connected to a chat agent (e.g. Claude Code) to carry out complex scientific queries through natural language while invoking tools and workflows from the original paper. We demonstrate Paper2Agent's effectiveness in creating reliable and capable paper agents through in-depth case studies. Paper2Agent created an agent that leverages AlphaGenome to interpret genomic variants and agents based on ScanPy and TISSUE to carry out single-cell and spatial transcriptomics analyses. We validate that these paper agents can reproduce the original paper's results and can correctly carry out novel user queries. Paper2Agent automatically created AI co-scientist that identified new splicing variant associated with ADHD risk. By turning static papers into dynamic, interactive AI agents, Paper2Agent introduces a new paradigm for knowledge dissemination and a foundation for the collaborative ecosystem of AI co-scientists.
>
---
#### [replaced 044] Checkpoint-GCG: Auditing and Attacking Fine-Tuning-Based Prompt Injection Defenses
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.15738v2](http://arxiv.org/pdf/2505.15738v2)**

> **作者:** Xiaoxue Yang; Bozhidar Stevanoski; Matthieu Meeus; Yves-Alexandre de Montjoye
>
> **摘要:** Large language models (LLMs) are increasingly deployed in real-world applications ranging from chatbots to agentic systems, where they are expected to process untrusted data and follow trusted instructions. Failure to distinguish between the two poses significant security risks, exploited by prompt injection attacks, which inject malicious instructions into the data to control model outputs. Model-level defenses have been proposed to mitigate prompt injection attacks. These defenses fine-tune LLMs to ignore injected instructions in untrusted data. We introduce Checkpoint-GCG, a white-box attack against fine-tuning-based defenses. Checkpoint-GCG enhances the Greedy Coordinate Gradient (GCG) attack by leveraging intermediate model checkpoints produced during fine-tuning to initialize GCG, with each checkpoint acting as a stepping stone for the next one to continuously improve attacks. First, we instantiate Checkpoint-GCG to evaluate the robustness of the state-of-the-art defenses in an auditing setup, assuming both (a) full knowledge of the model input and (b) access to intermediate model checkpoints. We show Checkpoint-GCG to achieve up to $96\%$ attack success rate (ASR) against the strongest defense. Second, we relax the first assumption by searching for a universal suffix that would work on unseen inputs, and obtain up to $89.9\%$ ASR against the strongest defense. Finally, we relax both assumptions by searching for a universal suffix that would transfer to similar black-box models and defenses, achieving an ASR of $63.9\%$ against a newly released defended model from Meta.
>
---
#### [replaced 045] Multi-Perspective Stance Detection
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.08752v2](http://arxiv.org/pdf/2411.08752v2)**

> **作者:** Benedetta Muscato; Praveen Bushipaka; Gizem Gezici; Lucia Passaro; Fosca Giannotti
>
> **摘要:** Subjective NLP tasks usually rely on human annotations provided by multiple annotators, whose judgments may vary due to their diverse backgrounds and life experiences. Traditional methods often aggregate multiple annotations into a single ground truth, disregarding the diversity in perspectives that arises from annotator disagreement. In this preliminary study, we examine the effect of including multiple annotations on model accuracy in classification. Our methodology investigates the performance of perspective-aware classification models in stance detection task and further inspects if annotator disagreement affects the model confidence. The results show that multi-perspective approach yields better classification performance outperforming the baseline which uses the single label. This entails that designing more inclusive perspective-aware AI models is not only an essential first step in implementing responsible and ethical AI, but it can also achieve superior results than using the traditional approaches.
>
---
#### [replaced 046] Rediscovering Entropy Regularization: Adaptive Coefficient Unlocks Its Potential for LLM Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **链接: [http://arxiv.org/pdf/2510.10959v2](http://arxiv.org/pdf/2510.10959v2)**

> **作者:** Xiaoyun Zhang; Xiaojian Yuan; Di Huang; Wang You; Chen Hu; Jingqing Ruan; Kejiang Chen; Xing Hu
>
> **备注:** 16 pages, 4 figures
>
> **摘要:** Reasoning ability has become a defining capability of Large Language Models (LLMs), with Reinforcement Learning with Verifiable Rewards (RLVR) emerging as a key paradigm to enhance it. However, RLVR training often suffers from policy entropy collapse, where the policy becomes overly deterministic, hindering exploration and limiting reasoning performance. While entropy regularization is a common remedy, its effectiveness is highly sensitive to the fixed coefficient, making it unstable across tasks and models. In this work, we revisit entropy regularization in RLVR and argue that its potential has been largely underestimated. Our analysis shows that (i) tasks of varying difficulty demand distinct exploration intensities, and (ii) balanced exploration may require the policy entropy to be maintained within a moderate range below its initial level. Therefore, we propose Adaptive Entropy Regularization (AER)--a framework that dynamically balances exploration and exploitation via three components: difficulty-aware coefficient allocation, initial-anchored target entropy, and dynamic global coefficient adjustment. Experiments on multiple mathematical reasoning benchmarks show that AER consistently outperforms baselines, improving both reasoning accuracy and exploration capability.
>
---
#### [replaced 047] Are Large Reasoning Models Interruptible?
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.11713v3](http://arxiv.org/pdf/2510.11713v3)**

> **作者:** Tsung-Han Wu; Mihran Miroyan; David M. Chan; Trevor Darrell; Narges Norouzi; Joseph E. Gonzalez
>
> **备注:** Project Page: http://dynamic-lm.github.io
>
> **摘要:** Large Reasoning Models (LRMs) excel at complex reasoning but are traditionally evaluated in static, "frozen world" settings: model responses are assumed to be instantaneous, and the context of a request is presumed to be immutable over the duration of the response. While generally true for short-term tasks, the "frozen world" assumption breaks down in modern reasoning tasks such as assistive programming, where models may take hours to think through problems and code may change dramatically from the time the model starts thinking to the model's final output. In this work, we challenge the frozen world assumption and evaluate LRM robustness under two realistic dynamic scenarios: interruptions, which test the quality of the model's partial outputs on a limited budget, and dynamic context, which tests model adaptation to in-flight changes. Across mathematics and programming benchmarks that require long-form reasoning, static evaluations consistently overestimate robustness: even state-of-the-art LRMs, which achieve high accuracy in static settings, can fail unpredictably when interrupted or exposed to changing context, with performance dropping by up to 60% when updates are introduced late in the reasoning process. Our analysis further reveals several novel failure modes, including reasoning leakage, where models fold the reasoning into their final answer when interrupted; panic, where under time pressure models abandon reasoning entirely and return incorrect answers; and self-doubt, where performance degrades while incorporating updated information. Project Page: http://dynamic-lm.github.io/
>
---
#### [replaced 048] Seeing Through Green: Text-Based Classification and the Firm's Returns from Green Patents
- **分类: econ.GN; cs.CL; q-fin.EC**

- **链接: [http://arxiv.org/pdf/2507.02287v2](http://arxiv.org/pdf/2507.02287v2)**

> **作者:** Lapo Santarlasci; Armando Rungi; Antonio Zinilli
>
> **摘要:** This paper introduces Natural Language Processing for identifying ``true'' green patents from official supporting documents. We start our training on about 12.4 million patents that had been classified as green from previous literature. Thus, we train a simple neural network to enlarge a baseline dictionary through vector representations of expressions related to environmental technologies. After testing, we find that ``true'' green patents represent about 20\% of the total of patents classified as green from previous literature. We show heterogeneity by technological classes, and then check that `true' green patents are about 1\% less cited by following inventions. In the second part of the paper, we test the relationship between patenting and a dashboard of firm-level financial accounts in the European Union. After controlling for reverse causality, we show that holding at least one ``true'' green patent raises sales, market shares, and productivity. If we restrict the analysis to high-novelty ``true'' green patents, we find that they also yield higher profits. Our findings underscore the importance of using text analyses to gauge finer-grained patent classifications that are useful for policymaking in different domains.
>
---
#### [replaced 049] ChartGalaxy: A Dataset for Infographic Chart Understanding and Generation
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.18668v5](http://arxiv.org/pdf/2505.18668v5)**

> **作者:** Zhen Li; Duan Li; Yukai Guo; Xinyuan Guo; Bowen Li; Lanxi Xiao; Shenyu Qiao; Jiashu Chen; Zijian Wu; Hui Zhang; Xinhuan Shu; Shixia Liu
>
> **备注:** 58 pages
>
> **摘要:** Infographic charts are a powerful medium for communicating abstract data by combining visual elements (e.g., charts, images) with textual information. However, their visual and structural richness poses challenges for large vision-language models (LVLMs), which are typically trained on plain charts. To bridge this gap, we introduce ChartGalaxy, a million-scale dataset designed to advance the understanding and generation of infographic charts. The dataset is constructed through an inductive process that identifies 75 chart types, 440 chart variations, and 68 layout templates from real infographic charts and uses them to create synthetic ones programmatically. We showcase the utility of this dataset through: 1) improving infographic chart understanding via fine-tuning, 2) benchmarking code generation for infographic charts, and 3) enabling example-based infographic chart generation. By capturing the visual and structural complexity of real design, ChartGalaxy provides a useful resource for enhancing multimodal reasoning and generation in LVLMs.
>
---
#### [replaced 050] Detecting Token-Level Hallucinations Using Variance Signals: A Reference-Free Approach
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.04137v3](http://arxiv.org/pdf/2507.04137v3)**

> **作者:** Keshav Kumar
>
> **摘要:** Large Language Models (LLMs) have demonstrated impressive generative capabilities across diverse tasks but remain susceptible to hallucinations, confidently generated yet factually incorrect outputs. We introduce a reference-free, token-level hallucination detection framework that leverages the variance in token log-probabilities across multiple stochastic generations. Unlike prior methods that require ground-truth references or sentence-level verification, our approach is model-agnostic, interpretable, and suited for real-time or post-hoc analysis. We evaluate our method on unanswerable question prompts from the SQuAD v2 dataset and benchmark across three autoregressive models of varying scales: GPT-Neo 125M, Falcon 1B, and Mistral 7B. Through both quantitative metrics and visual diagnostics, we show that token-level variance reliably highlights instability in model outputs and correlates with hallucination patterns. Our framework is lightweight, reproducible, and adaptable to multiple domains, offering a valuable diagnostic tool for analyzing generative reliability in LLMs.
>
---
#### [replaced 051] LLMs are Single-threaded Reasoners: Demystifying the Working Mechanism of Soft Thinking
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.03440v4](http://arxiv.org/pdf/2508.03440v4)**

> **作者:** Junhong Wu; Jinliang Lu; Zixuan Ren; Gangqiang Hu; Zhi Wu; Dai Dai; Hua Wu
>
> **备注:** 11 pages, 6 figures, working in progress
>
> **摘要:** Human cognition naturally engages with abstract and fluid concepts, whereas existing reasoning models often rely on generating discrete tokens, potentially constraining their expressive capabilities. Recent advancements aim to address this limitation by enabling large language models (LLMs) to generate soft, abstract tokens, thus facilitating reasoning within a continuous concept space. In this paper, we investigate the Soft Thinking capabilities of various LLMs through a systematic analysis of their internal behavior using a suite of probing techniques. Contrary to the prevailing belief that Soft Thinking supports parallel exploration of diverse reasoning paths, our findings reveal that LLMs behave as single-threaded reasoners--they predominantly rely on the token with the highest probability in the soft input to predict the next step. This behavior induces a greedy feedback loop that suppresses alternative reasoning paths and undermines the benefits of transmitting richer information via Soft Tokens. To address this Greedy Pitfall, we propose Stochastic Soft Thinking, which introduces stochasticity to break free from this Greedy Pitfall. Our experiments demonstrate that incorporating randomness--particularly with the Gumbel-Softmax trick--can alleviate the limitations of vanilla approaches and unleash the potential of Soft Thinking, resulting in superior performance across eight reasoning benchmarks. We further demonstrate that Stochastic Soft Thinking exhibits stronger exploration potential compared to conventional COT. Our findings deepen the understanding of continuous reasoning and establish the foundation for future work on improving Soft Thinking with Reinforcement Learning.
>
---
#### [replaced 052] ENIGMA: The Geometry of Reasoning and Alignment in Large-Language Models
- **分类: cs.LG; cs.AI; cs.CL; 68T50; I.2.7**

- **链接: [http://arxiv.org/pdf/2510.11278v2](http://arxiv.org/pdf/2510.11278v2)**

> **作者:** Gareth Seneque; Lap-Hang Ho; Nafise Erfanian Saeedi; Jeffrey Molendijk; Ariel Kuperman; Tim Elson
>
> **备注:** 52 pages, 10 figures, author typo corrected, abstract typo corrected
>
> **摘要:** We present Entropic Mutual-Information Geometry Large-Language Model Alignment (ENIGMA), a novel approach to Large-Language Model (LLM) training that jointly improves reasoning, alignment and robustness by treating an organisation's policies/principles as directions to move on a model's information manifold. Our single-loop trainer combines Group-Relative Policy Optimisation (GRPO), an on-policy, critic-free RL method with Chain-of-Thought (CoT)-format only rewards; a Self-Supervised Alignment with Mutual Information (SAMI)-style symmetric InfoNCE auxiliary; and an entropic Sinkhorn optimal-transport regulariser on hidden-state distributions to bound geometry drift. We also introduce infoNCE metrics that specialise to a standard MI lower bound under matched negatives to measure how strongly a model's CoT encodes these policies. These metrics include a Sufficiency Index (SI) that enables the selection and creation of principles that maximise downstream performance prior to training. In our experiments using small (1B) LLMs, high-SI principles predict steadier training dynamics and improved benchmark performance over GRPO ablations. Our information-geometry analysis of trained models validates desirable structural change in the manifold. These results support our hypothesis that reasoning, alignment, and robustness are projections of a single information-geometric objective, and that models trained using ENIGMA demonstrate principled reasoning without the use of a reward model, offering a path to trusted capability
>
---
#### [replaced 053] Beyond Two-Stage Training: Cooperative SFT and RL for LLM Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.06948v2](http://arxiv.org/pdf/2509.06948v2)**

> **作者:** Liang Chen; Xueting Han; Li Shen; Jing Bai; Kam-Fai Wong
>
> **摘要:** Reinforcement learning (RL) has proven effective in incentivizing the reasoning abilities of large language models (LLMs), but suffers from severe efficiency challenges due to its trial-and-error nature. While the common practice employs supervised fine-tuning (SFT) as a warm-up stage for RL, this decoupled two-stage approach suffers from catastrophic forgetting: second-stage RL gradually loses SFT-acquired behaviors and inefficiently explores new patterns. This study introduces a novel method for learning reasoning models that employs bilevel optimization to facilitate better cooperation between these training paradigms. By conditioning the SFT objective on the optimal RL policy, our approach enables SFT to meta-learn how to guide RL's optimization process. During training, the lower level performs RL updates while simultaneously receiving SFT supervision, and the upper level explicitly maximizes the cooperative gain-the performance advantage of joint SFT-RL training over RL alone. Empirical evaluations on five reasoning benchmarks demonstrate that our method consistently outperforms baselines and achieves a better balance between effectiveness and efficiency.
>
---
#### [replaced 054] Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.04340v3](http://arxiv.org/pdf/2510.04340v3)**

> **作者:** Daniel Tan; Anders Woodruff; Niels Warncke; Arun Jose; Maxime Riché; David Demitri Africa; Mia Taylor
>
> **备注:** 40 pages, 22 figures. In proceedings at ICLR 2026
>
> **摘要:** Language model finetuning often results in learning undesirable traits in combination with desired ones. To address this, we propose inoculation prompting: modifying finetuning data by prepending a short system-prompt instruction that deliberately elicits the undesirable trait. At test time, we evaluate without the instruction; inoculated models have much lower expression of the trait than models trained with unmodified training data. Inoculation is selective: in a toy setting where assistant responses are always in Spanish and ALL-CAPS, an appropriate inoculation (e.g., ``You always speak in Spanish.'') teaches the model to capitalize responses while still responding in English. We find that inoculation is also effective across several additional settings: reducing emergent misalignment (EM) from task-specific finetuning, defending against backdoor injections, and mitigating the transmission of traits via subliminal learning. Follow-up analysis suggests a mechanism: making a trait less surprising via inoculation reduces optimization pressure to globally update the model, thereby reducing the degree of generalization. Our analysis relates to prior work on EM: inoculation explains prior findings that educational contexts mitigate EM from insecure code. Beyond demonstrating a simple and effective technique for selective learning, our results contribute to a better conceptual understanding of how and why language models generalize.
>
---
#### [replaced 055] Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.02922v3](http://arxiv.org/pdf/2504.02922v3)**

> **作者:** Julian Minder; Clément Dumas; Caden Juang; Bilal Chugtai; Neel Nanda
>
> **备注:** 51 pages, 33 figures, Accepted at 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
>
> **摘要:** Model diffing is the study of how fine-tuning changes a model's representations and internal algorithms. Many behaviors of interest are introduced during fine-tuning, and model diffing offers a promising lens to interpret such behaviors. Crosscoders are a recent model diffing method that learns a shared dictionary of interpretable concepts represented as latent directions in both the base and fine-tuned models, allowing us to track how concepts shift or emerge during fine-tuning. Notably, prior work has observed concepts with no direction in the base model, and it was hypothesized that these model-specific latents were concepts introduced during fine-tuning. However, we identify two issues which stem from the crosscoders L1 training loss that can misattribute concepts as unique to the fine-tuned model, when they really exist in both models. We develop Latent Scaling to flag these issues by more accurately measuring each latent's presence across models. In experiments comparing Gemma 2 2B base and chat models, we observe that the standard crosscoder suffers heavily from these issues. Building on these insights, we train a crosscoder with BatchTopK loss and show that it substantially mitigates these issues, finding more genuinely chat-specific and highly interpretable concepts. We recommend practitioners adopt similar techniques. Using the BatchTopK crosscoder, we successfully identify a set of chat-specific latents that are both interpretable and causally effective, representing concepts such as $\textit{false information}$ and $\textit{personal question}$, along with multiple refusal-related latents that show nuanced preferences for different refusal triggers. Overall, our work advances best practices for the crosscoder-based methodology for model diffing and demonstrates that it can provide concrete insights into how chat-tuning modifies model behavior.
>
---
#### [replaced 056] Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.12491v2](http://arxiv.org/pdf/2504.12491v2)**

> **作者:** Hansi Zeng; Kai Hui; Honglei Zhuang; Zhen Qin; Zhenrui Yue; Hamed Zamani; Dana Alon
>
> **摘要:** While metrics available during pre-training, such as perplexity, correlate well with model performance at scaling-laws studies, their predictive capacities at a fixed model size remain unclear, hindering effective model selection and development. To address this gap, we formulate the task of selecting pre-training checkpoints to maximize downstream fine-tuning performance as a pairwise classification problem: predicting which of two LLMs, differing in their pre-training, will perform better after supervised fine-tuning (SFT). We construct a dataset using 50 1B parameter LLM variants with systematically varied pre-training configurations, e.g., objectives or data, and evaluate them on diverse downstream tasks after SFT. We first conduct a study and demonstrate that the conventional perplexity is a misleading indicator. As such, we introduce novel unsupervised and supervised proxy metrics derived from pre-training that successfully reduce the relative performance prediction error rate by over 50%. Despite the inherent complexity of this task, we demonstrate the practical utility of our proposed proxies in specific scenarios, paving the way for more efficient design of pre-training schemes optimized for various downstream tasks.
>
---
#### [replaced 057] Make an Offer They Can't Refuse: Grounding Bayesian Persuasion in Real-World Dialogues without Pre-Commitment
- **分类: cs.CL; cs.GT**

- **链接: [http://arxiv.org/pdf/2510.13387v2](http://arxiv.org/pdf/2510.13387v2)**

> **作者:** Buwei He; Yang Liu; Zhaowei Zhang; Zixia Jia; Huijia Wu; Zhaofeng He; Zilong Zheng; Yipeng Kang
>
> **备注:** Under review
>
> **摘要:** Persuasion, a fundamental social capability for humans, remains a challenge for AI systems such as large language models (LLMs). Current studies often overlook the strategic use of information asymmetry in message design or rely on strong assumptions regarding pre-commitment. In this work, we explore the application of Bayesian Persuasion (BP) in natural language within single-turn dialogue settings, to enhance the strategic persuasion capabilities of LLMs. Our framework incorporates a commitment-communication mechanism, where the persuader explicitly outlines an information schema by narrating their potential types (e.g., honest or dishonest), thereby guiding the persuadee in performing the intended Bayesian belief update. We evaluate two variants of our approach: Semi-Formal-Natural-Language (SFNL) BP and Fully-Natural-Language (FNL) BP, benchmarking them against both naive and strong non-BP (NBP) baselines within a comprehensive evaluation framework. This framework covers a diverse set of persuadees -- including LLM instances with varying prompts and fine-tuning and human participants -- across tasks ranging from specially designed persuasion scenarios to general everyday situations. Experimental results on LLM-based agents reveal three main findings: (1) LLMs guided by BP strategies consistently achieve higher persuasion success rates than NBP baselines; (2) SFNL exhibits greater credibility and logical coherence, while FNL shows stronger emotional resonance and robustness in naturalistic conversations; (3) with supervised fine-tuning, smaller models can attain BP performance comparable to that of larger models.
>
---
#### [replaced 058] Sentence Smith: Controllable Edits for Evaluating Text Embeddings
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.14734v3](http://arxiv.org/pdf/2502.14734v3)**

> **作者:** Hongji Li; Andrianos Michail; Reto Gubelmann; Simon Clematide; Juri Opitz
>
> **备注:** EMNLP 2025 (main)
>
> **摘要:** Controllable and transparent text generation has been a long-standing goal in NLP. Almost as long-standing is a general idea for addressing this challenge: Parsing text to a symbolic representation, and generating from it. However, earlier approaches were hindered by parsing and generation insufficiencies. Using modern parsers and a safety supervision mechanism, we show how close current methods come to this goal. Concretely, we propose the Sentence Smith framework for English, which has three steps: 1. Parsing a sentence into a semantic graph. 2. Applying human-designed semantic manipulation rules. 3. Generating text from the manipulated graph. A final entailment check (4.) verifies the validity of the applied transformation. To demonstrate our framework's utility, we use it to induce hard negative text pairs that challenge text embedding models. Since the controllable generation makes it possible to clearly isolate different types of semantic shifts, we can evaluate text embedding models in a fine-grained way, also addressing an issue in current benchmarking where linguistic phenomena remain opaque. Human validation confirms that our transparent generation process produces texts of good quality. Notably, our way of generation is very resource-efficient, since it relies only on smaller neural networks.
>
---
#### [replaced 059] Absolute Zero: Reinforced Self-play Reasoning with Zero Data
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.03335v3](http://arxiv.org/pdf/2505.03335v3)**

> **作者:** Andrew Zhao; Yiran Wu; Yang Yue; Tong Wu; Quentin Xu; Yang Yue; Matthieu Lin; Shenzhi Wang; Qingyun Wu; Zilong Zheng; Gao Huang
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) has shown promise in enhancing the reasoning capabilities of large language models by learning directly from outcome-based rewards. Recent RLVR works that operate under the zero setting avoid supervision in labeling the reasoning process, but still depend on manually curated collections of questions and answers for training. The scarcity of high-quality, human-produced examples raises concerns about the long-term scalability of relying on human supervision, a challenge already evident in the domain of language model pretraining. Furthermore, in a hypothetical future where AI surpasses human intelligence, tasks provided by humans may offer limited learning potential for a superintelligent system. To address these concerns, we propose a new RLVR paradigm called Absolute Zero, in which a single model learns to propose tasks that maximize its own learning progress and improves reasoning by solving them, without relying on any external data. Under this paradigm, we introduce the Absolute Zero Reasoner (AZR), a system that self-evolves its training curriculum and reasoning ability by using a code executor to both validate proposed code reasoning tasks and verify answers, serving as an unified source of verifiable reward to guide open-ended yet grounded learning. Despite being trained entirely without external data, AZR achieves overall SOTA performance on coding and mathematical reasoning tasks, outperforming existing zero-setting models that rely on tens of thousands of in-domain human-curated examples. Furthermore, we demonstrate that AZR can be effectively applied across different model scales and is compatible with various model classes.
>
---
#### [replaced 060] SFTMix: Elevating Language Model Instruction Tuning with Mixup Recipe
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.05248v3](http://arxiv.org/pdf/2410.05248v3)**

> **作者:** Yuxin Xiao; Shujian Zhang; Wenxuan Zhou; Marzyeh Ghassemi; Sanqiang Zhao
>
> **摘要:** To acquire instruction-following capabilities, large language models (LLMs) undergo instruction tuning, where they are trained on instruction-response pairs using next-token prediction (NTP). Efforts to improve instruction tuning often focus on higher-quality supervised fine-tuning (SFT) datasets, typically requiring data filtering with proprietary LLMs or human annotation. In this paper, we take a different approach by proposing SFTMix, a novel Mixup-based recipe that elevates LLM instruction tuning without relying on well-curated datasets. We observe that LLMs exhibit uneven confidence across the semantic representation space. We argue that examples with different confidence levels should play distinct roles in instruction tuning: Confident data is prone to overfitting, while unconfident data is harder to generalize. Based on this insight, SFTMix leverages training dynamics to identify examples with varying confidence levels. We then interpolate them to bridge the confidence gap and apply a Mixup-based regularization to support learning on these additional, interpolated examples. We demonstrate the effectiveness of SFTMix in both instruction-following and healthcare-specific SFT tasks, with consistent improvements across LLM families and SFT datasets of varying sizes and qualities. Extensive analyses across six directions highlight SFTMix's compatibility with data selection, adaptability to compute-constrained scenarios, and scalability to broader applications.
>
---
#### [replaced 061] When Style Breaks Safety: Defending LLMs Against Superficial Style Alignment
- **分类: cs.LG; cs.AI; cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2506.07452v2](http://arxiv.org/pdf/2506.07452v2)**

> **作者:** Yuxin Xiao; Sana Tonekaboni; Walter Gerych; Vinith Suriyakumar; Marzyeh Ghassemi
>
> **摘要:** Large language models (LLMs) can be prompted with specific styles (e.g., formatting responses as lists), including in malicious queries. Prior jailbreak research mainly augments these queries with additional string transformations to maximize attack success rate (ASR). However, the impact of style patterns in the original queries that are semantically irrelevant to the malicious intent remains unclear. In this work, we seek to understand whether style patterns compromise LLM safety, how superficial style alignment increases model vulnerability, and how best to mitigate these risks during alignment. We first define ASR inflation as the increase in ASR due to style patterns in existing jailbreak benchmark queries. By evaluating 32 LLMs across seven benchmarks, we find that nearly all models exhibit ASR inflation. Notably, the inflation correlates with an LLM's relative attention to style patterns, which also overlap more with its instruction-tuning data when inflation occurs. We then investigate superficial style alignment, and find that fine-tuning with specific styles makes LLMs more vulnerable to jailbreaks of those same styles. Finally, we propose SafeStyle, a defense strategy that incorporates a small amount of safety training data augmented to match the distribution of style patterns in the fine-tuning data. Across three LLMs, six fine-tuning style settings, and two real-world instruction-tuning datasets, SafeStyle consistently outperforms baselines in maintaining LLM safety.
>
---
#### [replaced 062] Plugging Schema Graph into Multi-Table QA: A Human-Guided Framework for Reducing LLM Reliance
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.04427v3](http://arxiv.org/pdf/2506.04427v3)**

> **作者:** Xixi Wang; Miguel Costa; Jordanka Kovaceva; Shuai Wang; Francisco C. Pereira
>
> **备注:** Accepted to EMNLP 2025 findings
>
> **摘要:** Large language models (LLMs) have shown promise in table Question Answering (Table QA). However, extending these capabilities to multi-table QA remains challenging due to unreliable schema linking across complex tables. Existing methods based on semantic similarity work well only on simplified hand-crafted datasets and struggle to handle complex, real-world scenarios with numerous and diverse columns. To address this, we propose a graph-based framework that leverages human-curated relational knowledge to explicitly encode schema links and join paths. Given a natural language query, our method searches on graph to construct interpretable reasoning chains, aided by pruning and sub-path merging strategies to enhance efficiency and coherence. Experiments on both standard benchmarks and a realistic, large-scale dataset demonstrate the effectiveness of our approach. To our knowledge, this is the first multi-table QA system applied to truly complex industrial tabular data.
>
---
#### [replaced 063] Why is Your Language Model a Poor Implicit Reward Model?
- **分类: cs.CL; cs.AI; cs.LG; stat.ML**

- **链接: [http://arxiv.org/pdf/2507.07981v2](http://arxiv.org/pdf/2507.07981v2)**

> **作者:** Noam Razin; Yong Lin; Jiarui Yao; Sanjeev Arora
>
> **备注:** Code available at https://github.com/princeton-pli/exrm-vs-imrm
>
> **摘要:** Reward models are key to language model post-training and inference pipelines. Conveniently, recent work showed that every language model defines an implicit reward model (IM-RM), without requiring any architectural changes. However, such IM-RMs tend to generalize worse, especially out-of-distribution, compared to explicit reward models (EX-RMs) that apply a dedicated linear head over the hidden representations of a language model. The existence of a generalization gap is puzzling, as EX-RMs and IM-RMs are nearly identical. They can be trained using the same data, loss function, and language model, and differ only in how the reward is computed. Toward a fundamental understanding of the implicit biases underlying different reward model types, we investigate the root cause of this gap. Our main finding, backed by theory and experiments, is that IM-RMs rely more heavily on superficial token-level cues. Consequently, they often generalize worse than EX-RMs under token-level distribution shifts, as well as in-distribution. Furthermore, we provide evidence against alternative hypotheses for the generalization gap. Most notably, we challenge the intuitive claim that IM-RMs struggle in tasks where generation is harder than verification because they can operate both as a verifier and a generator. Taken together, our results highlight that seemingly minor design choices can substantially impact the generalization behavior of reward models.
>
---
#### [replaced 064] Are LLMs Stable Formal Logic Translators in Logical Reasoning Across Linguistically Diversified Texts?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.04575v2](http://arxiv.org/pdf/2506.04575v2)**

> **作者:** Qingchuan Li; Jiatong Li; Zirui Liu; Mingyue Cheng; Yuting Zeng; Qi Liu; Tongxuan Liu
>
> **摘要:** Logical reasoning with large language models (LLMs) has received growing attention. One mainstream approach translates natural language into formal logic and then applies symbolic solvers for deduction. While effective in many tasks, these LLM-based translators often fail to generate consistent symbolic representations when the same concept appears in different linguistic forms. Such inconsistencies break logical coherence and lead to solver errors. However, most existing benchmarks lack this type of linguistic variation, which frequently occurs in real-world text, leaving the problem underexplored. To address this gap, we present SoLT, a benchmark that systematically rewrites reasoning datasets into diverse yet logically equivalent forms across multiple levels. Beyond evaluation, SoLT also provides a general method to enrich any dataset with linguistic diversity while preserving both meaning and logic. To further enhance the stability of LLM-based reasoning, we propose MenTaL, which explicitly guides models to build a concept-symbol mapping table during translation. By linking equivalent expressions to shared symbols, MenTaL maintains consistency and mitigates symbol drift. Experiments on SoLT demonstrate that LLMs indeed suffer from inconsistent symbol mapping under linguistic variation, leading to significant drops in reasoning accuracy. Meanwhile, applying MenTaL brings clear and stable performance improvements across diverse inputs. Overall, our findings reveal that overlooking linguistic diversity hides key weaknesses in LLM-based translators, and our work offers a step toward more reliable logical reasoning in varied real-world scenarios. Our code is available at https://github.com/wufeiwuwoshihua/LinguDiver.
>
---
#### [replaced 065] Merge-of-Thought Distillation
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.08814v3](http://arxiv.org/pdf/2509.08814v3)**

> **作者:** Zhanming Shen; Zeyu Qin; Zenan Huang; Hao Chen; Jiaqi Hu; Yihong Zhuang; Guoshan Lu; Gang Chen; Junbo Zhao
>
> **摘要:** Efficient reasoning distillation for long chain-of-thought (CoT) models is increasingly constrained by the assumption of a single oracle teacher, despite the practical availability of multiple candidate teachers and growing CoT corpora. We revisit teacher selection and observe that different students have different "best teachers," and even for the same student, the best teacher can vary across datasets. Therefore, to unify multiple teachers' reasoning abilities into a student to overcome conflicts among various teachers' supervision, we propose Merge-of-Thought Distillation (MoT), a lightweight framework that alternates between teacher-specific supervised fine-tuning branches and weight-space merging of the resulting student variants. On competition math benchmarks, using only about 200 CoT samples, applying MoT to a Qwen3-14B student surpasses strong models including Deepseek-R1, Qwen3-32B, and OpenAI-O1, demonstrating substantial gains. Besides, MoT consistently outperforms the best single-teacher distillation, improves general reasoning beyond mathematics while reducing catastrophic forgetting, and shows robustness to distribution-shifted and peer-level teachers. Finally, we have demonstrated MoT possesses consensus CoT by eliminating teacher-specific inductive biases and inter-teacher conflicts while repeatedly reinforcing the learning of consensus reasoning features. These results position MoT as a simple, effective route to efficiently distilling long CoT capabilities from diverse teachers into compact students.
>
---
#### [replaced 066] Benchmarking Adversarial Robustness to Bias Elicitation in Large Language Models: Scalable Automated Assessment with LLM-as-a-Judge
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.07887v2](http://arxiv.org/pdf/2504.07887v2)**

> **作者:** Riccardo Cantini; Alessio Orsino; Massimo Ruggiero; Domenico Talia
>
> **摘要:** The growing integration of Large Language Models (LLMs) into critical societal domains has raised concerns about embedded biases that can perpetuate stereotypes and undermine fairness. Such biases may stem from historical inequalities in training data, linguistic imbalances, or adversarial manipulation. Despite mitigation efforts, recent studies show that LLMs remain vulnerable to adversarial attacks that elicit biased outputs. This work proposes a scalable benchmarking framework to assess LLM robustness to adversarial bias elicitation. Our methodology involves: (i) systematically probing models across multiple tasks targeting diverse sociocultural biases, (ii) quantifying robustness through safety scores using an LLM-as-a-Judge approach, and (iii) employing jailbreak techniques to reveal safety vulnerabilities. To facilitate systematic benchmarking, we release a curated dataset of bias-related prompts, named CLEAR-Bias. Our analysis, identifying DeepSeek V3 as the most reliable judge LLM, reveals that bias resilience is uneven, with age, disability, and intersectional biases among the most prominent. Some small models outperform larger ones in safety, suggesting that training and architecture may matter more than scale. However, no model is fully robust to adversarial elicitation, with jailbreak attacks using low-resource languages or refusal suppression proving effective across model families. We also find that successive LLM generations exhibit slight safety gains, while models fine-tuned for the medical domain tend to be less safe than their general-purpose counterparts.
>
---
#### [replaced 067] TripScore: Benchmarking and rewarding real-world travel planning with fine-grained evaluation
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.09011v3](http://arxiv.org/pdf/2510.09011v3)**

> **作者:** Yincen Qu; Huan Xiao; Feng Li; Gregory Li; Hui Zhou; Xiangying Dai; Xiaoru Dai
>
> **摘要:** Travel planning is a valuable yet complex task that poses significant challenges even for advanced large language models (LLMs). While recent benchmarks have advanced in evaluating LLMs' planning capabilities, they often fall short in evaluating feasibility, reliability, and engagement of travel plans. We introduce a comprehensive benchmark for travel planning that unifies fine-grained criteria into a single reward, enabling direct comparison of plan quality and seamless integration with reinforcement learning (RL). Our evaluator achieves moderate agreement with travel-expert annotations (60.75%) and outperforms multiple LLM-as-judge baselines. We further release a large-scale dataset of 4,870 queries including 219 real-world, free-form requests for generalization to authentic user intent. Using this benchmark, we conduct extensive experiments across diverse methods and LLMs, including test-time computation, neuro-symbolic approaches, supervised fine-tuning, and RL via GRPO. Across base models, RL generally improves itinerary feasibility over prompt-only and supervised baselines, yielding higher unified reward scores.
>
---
#### [replaced 068] Adversarial Defence without Adversarial Defence: Enhancing Language Model Robustness via Instance-level Principal Component Removal
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.21750v4](http://arxiv.org/pdf/2507.21750v4)**

> **作者:** Yang Wang; Chenghao Xiao; Yizhi Li; Stuart E. Middleton; Noura Al Moubayed; Chenghua Lin
>
> **备注:** This paper was accepted with an A-decision to Transactions of the Association for Computational Linguistics. This version is the pre-publication version prior to MIT Press production
>
> **摘要:** Pre-trained language models (PLMs) have driven substantial progress in natural language processing but remain vulnerable to adversarial attacks, raising concerns about their robustness in real-world applications. Previous studies have sought to mitigate the impact of adversarial attacks by introducing adversarial perturbations into the training process, either implicitly or explicitly. While both strategies enhance robustness, they often incur high computational costs. In this work, we propose a simple yet effective add-on module that enhances the adversarial robustness of PLMs by removing instance-level principal components, without relying on conventional adversarial defences or perturbing the original training data. Our approach transforms the embedding space to approximate Gaussian properties, thereby reducing its susceptibility to adversarial perturbations while preserving semantic relationships. This transformation aligns embedding distributions in a way that minimises the impact of adversarial noise on decision boundaries, enhancing robustness without requiring adversarial examples or costly training-time augmentation. Evaluations on eight benchmark datasets show that our approach improves adversarial robustness while maintaining comparable before-attack accuracy to baselines, achieving a balanced trade-off between robustness and generalisation.
>
---
#### [replaced 069] DiSTAR: Diffusion over a Scalable Token Autoregressive Representation for Speech Generation
- **分类: eess.AS; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.12210v2](http://arxiv.org/pdf/2510.12210v2)**

> **作者:** Yakun Song; Xiaobin Zhuang; Jiawei Chen; Zhikang Niu; Guanrou Yang; Chenpeng Du; Dongya Jia; Zhuo Chen; Yuping Wang; Yuxuan Wang; Xie Chen
>
> **摘要:** Recent attempts to interleave autoregressive (AR) sketchers with diffusion-based refiners over continuous speech representations have shown promise, but they remain brittle under distribution shift and offer limited levers for controllability. We introduce DISTAR, a zero-shot text-to-speech framework that operates entirely in a discrete residual vector quantization (RVQ) code space and tightly couples an AR language model with a masked diffusion model, without forced alignment or a duration predictor. Concretely, DISTAR drafts block-level RVQ tokens with an AR language model and then performs parallel masked-diffusion infilling conditioned on the draft to complete the next block, yielding long-form synthesis with blockwise parallelism while mitigating classic AR exposure bias. The discrete code space affords explicit control at inference: DISTAR produces high-quality audio under both greedy and sample-based decoding using classifier-free guidance, supports trade-offs between robustness and diversity, and enables variable bit-rate and controllable computation via RVQ layer pruning at test time. Extensive experiments and ablations demonstrate that DISTAR surpasses state-of-the-art zero-shot TTS systems in robustness, naturalness, and speaker/style consistency, while maintaining rich output diversity. Audio samples are provided on https://anonymous.4open.science/w/DiSTAR_demo.
>
---
#### [replaced 070] Confidence-Based Response Abstinence: Improving LLM Trustworthiness via Activation-Based Uncertainty Estimation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.13750v2](http://arxiv.org/pdf/2510.13750v2)**

> **作者:** Zhiqi Huang; Vivek Datla; Chenyang Zhu; Alfy Samuel; Daben Liu; Anoop Kumar; Ritesh Soni
>
> **备注:** UncertaiNLP at EMNLP 2025
>
> **摘要:** We propose a method for confidence estimation in retrieval-augmented generation (RAG) systems that aligns closely with the correctness of large language model (LLM) outputs. Confidence estimation is especially critical in high-stakes domains such as finance and healthcare, where the cost of an incorrect answer outweighs that of not answering the question. Our approach extends prior uncertainty quantification methods by leveraging raw feed-forward network (FFN) activations as auto-regressive signals, avoiding the information loss inherent in token logits and probabilities after projection and softmax normalization. We model confidence prediction as a sequence classification task, and regularize training with a Huber loss term to improve robustness against noisy supervision. Applied in a real-world financial industry customer-support setting with complex knowledge bases, our method outperforms strong baselines and maintains high accuracy under strict latency constraints. Experiments on Llama 3.1 8B model show that using activations from only the 16th layer preserves accuracy while reducing response latency. Our results demonstrate that activation-based confidence modeling offers a scalable, architecture-aware path toward trustworthy RAG deployment.
>
---
#### [replaced 071] Thinker: Learning to Think Fast and Slow
- **分类: cs.CL; cs.AI; cs.LG; I.2.6; I.2.8; I.5.1**

- **链接: [http://arxiv.org/pdf/2505.21097v2](http://arxiv.org/pdf/2505.21097v2)**

> **作者:** Stephen Chung; Wenyu Du; Jie Fu
>
> **备注:** 23 pages
>
> **摘要:** Recent studies show that the reasoning capabilities of Large Language Models (LLMs) can be improved by applying Reinforcement Learning (RL) to question-answering (QA) tasks in areas such as math and coding. With a long context length, LLMs may learn to perform search, as indicated by the self-correction behavior observed in DeepSeek R1. However, this search behavior is often imprecise and lacks confidence, resulting in long, redundant responses and highlighting deficiencies in intuition and verification. Inspired by the Dual Process Theory in psychology, we introduce a simple modification to the QA task that includes four stages: Fast Thinking, where the LLM must answer within a strict token budget; Verification, where the model evaluates its initial response; Slow Thinking, where it refines the initial response with more deliberation; and Summarization, where it distills the refinement from the previous stage into precise steps. Our proposed task improves average accuracy from 25.6% to 27.3% for Qwen2.5-1.5B, and from 45.9% to 51.0% for DeepSeek-R1-Qwen-1.5B. Notably, for Qwen2.5-1.5B, the Fast Thinking mode alone achieves 25.2% accuracy using fewer than 1000 tokens, demonstrating substantial inference efficiency gains. These findings suggest that intuition and deliberative reasoning are distinct, complementary systems benefiting from targeted training. Additionally, we have open-sourced both the trained models and the source code.
>
---
#### [replaced 072] A$^2$FM: An Adaptive Agent Foundation Model for Tool-Aware Hybrid Reasoning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.12838v2](http://arxiv.org/pdf/2510.12838v2)**

> **作者:** Qianben Chen; Jingyi Cao; Jiayu Zhang; Tianrui Qin; Xiaowan Li; King Zhu; Dingfeng Shi; He Zhu; Minghao Liu; Xiaobo Liang; Xin Gui; Ge Zhang; Jian Yang; Yuchen Eleanor Jiang; Wangchunshu Zhou
>
> **备注:** 9 pages, 5 figures, submitted to ICLR 2026
>
> **摘要:** Large language models split into two families: reasoning-centric LLMs, which strengthen internal chain-of-thought reasoning but cannot invoke external tools, and agentic LLMs, which learn to interact with environments and leverage tools but often lag in deep reasoning. This divide arises from fundamentally different training objectives, leading to mismatched strengths and inefficiency on simple queries, where both families tend to overthink or over-call tools. In this work, we present Adaptive Agent Foundation Model (A$^2$FM), a unified framework that follows a route-then-align principle: the model first learns task-aware routing and then aligns mode-specific trajectories under a shared backbone. To address the inefficiency gap, we introduce a third mode-instant-that handles simple queries directly, preventing unnecessary reasoning or tool calls while complementing the agentic and reasoning modes. To jointly enhance accuracy and efficiency, we propose Adaptive Policy Optimization (APO), which enforces adaptive sampling across modes and applies a cost-regularized reward. On the 32B scale, A$^2$FM achieves 13.4% on BrowseComp, 70.4% on AIME25, and 16.7% on HLE, setting new SOTA among comparable models and performing competitively with frontier LLMs across agentic, reasoning, and general benchmarks. Notably, the adaptive execution achieves a cost of pass of only $0.00487 per correct answer-cutting cost by 45.2% relative to reasoning and 33.5% relative to agentic, thus delivering substantially higher cost efficiency while maintaining comparable accuracy.
>
---
#### [replaced 073] Thunder-DeID: Accurate and Efficient De-identification Framework for Korean Court Judgments
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.15266v3](http://arxiv.org/pdf/2506.15266v3)**

> **作者:** Sungeun Hahm; Heejin Kim; Gyuseong Lee; Hyunji Park; Jaejin Lee
>
> **摘要:** To ensure a balance between open access to justice and personal data protection, the South Korean judiciary mandates the de-identification of court judgments before they can be publicly disclosed. However, the current de-identification process is inadequate for handling court judgments at scale while adhering to strict legal requirements. Additionally, the legal definitions and categorizations of personal identifiers are vague and not well-suited for technical solutions. To tackle these challenges, we propose a de-identification framework called Thunder-DeID, which aligns with relevant laws and practices. Specifically, we (i) construct and release the first Korean legal dataset containing annotated judgments along with corresponding lists of entity mentions, (ii) introduce a systematic categorization of Personally Identifiable Information (PII), and (iii) develop an end-to-end deep neural network (DNN)-based de-identification pipeline. Our experimental results demonstrate that our model achieves state-of-the-art performance in the de-identification of court judgments.
>
---
#### [replaced 074] Confidence Calibration in Large Language Model-Based Entity Matching
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.19557v2](http://arxiv.org/pdf/2509.19557v2)**

> **作者:** Iris Kamsteeg; Juan Cardenas-Cartagena; Floris van Beers; Gineke ten Holt; Tsegaye Misikir Tashu; Matias Valdenegro-Toro
>
> **备注:** 9 pages, 2 figures. UncertaiNLP 2025 Workshop @ EMNLP Camera Ready
>
> **摘要:** This research aims to explore the intersection of Large Language Models and confidence calibration in Entity Matching. To this end, we perform an empirical study to compare baseline RoBERTa confidences for an Entity Matching task against confidences that are calibrated using Temperature Scaling, Monte Carlo Dropout and Ensembles. We use the Abt-Buy, DBLP-ACM, iTunes-Amazon and Company datasets. The findings indicate that the proposed modified RoBERTa model exhibits a slight overconfidence, with Expected Calibration Error scores ranging from 0.0043 to 0.0552 across datasets. We find that this overconfidence can be mitigated using Temperature Scaling, reducing Expected Calibration Error scores by up to 23.83%.
>
---
#### [replaced 075] AI-generated Essays: Characteristics and Implications on Automated Scoring and Academic Integrity
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.17439v4](http://arxiv.org/pdf/2410.17439v4)**

> **作者:** Yang Zhong; Jiangang Hao; Michael Fauss; Chen Li; Yuan Wang
>
> **备注:** 29 pages
>
> **摘要:** The rapid advancement of large language models (LLMs) has enabled the generation of coherent essays, making AI-assisted writing increasingly common in educational and professional settings. Using large-scale empirical data, we examine and benchmark the characteristics and quality of essays generated by popular LLMs and discuss their implications for two key components of writing assessments: automated scoring and academic integrity. Our findings highlight limitations in existing automated scoring systems, such as e-rater, when applied to essays generated or heavily influenced by AI, and identify areas for improvement, including the development of new features to capture deeper thinking and recalibrating feature weights. Despite growing concerns that the increasing variety of LLMs may undermine the feasibility of detecting AI-generated essays, our results show that detectors trained on essays generated from one model can often identify texts from others with high accuracy, suggesting that effective detection could remain manageable in practice.
>
---
#### [replaced 076] ScholarBench: A Bilingual Benchmark for Abstraction, Comprehension, and Reasoning Evaluation in Academic Contexts
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.16566v2](http://arxiv.org/pdf/2505.16566v2)**

> **作者:** Dongwon Noh; Donghyeok Koh; Junghun Yuk; Gyuwan Kim; Jaeyong Lee; Kyungtae Lim; Cheoneum Park
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** Prior benchmarks for evaluating the domain-specific knowledge of large language models (LLMs) lack the scalability to handle complex academic tasks. To address this, we introduce \texttt{ScholarBench}, a benchmark centered on deep expert knowledge and complex academic problem-solving, which evaluates the academic reasoning ability of LLMs and is constructed through a three-step process. \texttt{ScholarBench} targets more specialized and logically complex contexts derived from academic literature, encompassing five distinct problem types. Unlike prior benchmarks, \texttt{ScholarBench} evaluates the abstraction, comprehension, and reasoning capabilities of LLMs across eight distinct research domains. To ensure high-quality evaluation data, we define category-specific example attributes and design questions that are aligned with the characteristic research methodologies and discourse structures of each domain. Additionally, this benchmark operates as an English-Korean bilingual dataset, facilitating simultaneous evaluation for linguistic capabilities of LLMs in both languages. The benchmark comprises 5,031 examples in Korean and 5,309 in English, with even state-of-the-art models like o3-mini achieving an average evaluation score of only 0.543, demonstrating the challenging nature of this benchmark.
>
---
#### [replaced 077] EasyNER: A Customizable Easy-to-Use Pipeline for Deep Learning- and Dictionary-based Named Entity Recognition from Medical and Life Science Text
- **分类: q-bio.QM; cs.CL; 92-04, 92-08, 68T50; J.3; I.2.7; H.3.3**

- **链接: [http://arxiv.org/pdf/2304.07805v3](http://arxiv.org/pdf/2304.07805v3)**

> **作者:** Rafsan Ahmed; Petter Berntsson; Alexander Skafte; Salma Kazemi Rashed; Marcus Klang; Adam Barvesten; Ola Olde; William Lindholm; Antton Lamarca Arrizabalaga; Pierre Nugues; Sonja Aits
>
> **摘要:** Background Medical and life science research generates millions of publications, and it is a great challenge for researchers to utilize this information in full since its scale and complexity greatly surpasses human reading capabilities. Automated text mining can help extract and connect information spread across this large body of literature, but this technology is not easily accessible to life scientists. Methods and Results Here, we developed an easy-to-use end-to-end pipeline for deep learning- and dictionary-based named entity recognition (NER) of typical entities found in medical and life science research articles, including diseases, cells, chemicals, genes/proteins, species and others. The pipeline can access and process large medical research article collections (PubMed, CORD-19) or raw text and incorporates a series of deep learning models fine-tuned on the HUNER corpora collection. In addition, the pipeline can perform dictionary-based NER related to COVID-19 and other medical topics. Users can also load their own NER models and dictionaries to include additional entities. The output consists of publication-ready ranked lists and graphs of detected entities and files containing the annotated texts. In addition, we provide two accessory scripts which allow processing of files in PubTator format and rapid inspection of the results for specific entities of interest. As model use cases, the pipeline was deployed on two collections of autophagy-related abstracts from PubMed and on the CORD19 dataset, a collection of 764 398 research article abstracts related to COVID-19. Conclusions The NER pipeline we present is applicable in a variety of medical research settings and makes customizable text mining accessible to life scientists.
>
---
