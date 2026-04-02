# 自然语言处理 cs.CL

- **最新发布 106 篇**

- **更新 44 篇**

## 最新发布

#### [new 001] Semantic Shifts of Psychological Concepts in Scientific and Popular Media Discourse: A Distributional Semantics Analysis of Russian-Language Corpora
- **分类: cs.CL**

- **简介: 该论文属于语义分析任务，研究心理概念在科学与通俗媒体中的意义变化。通过分布语义方法分析俄语文本，揭示不同语境下术语的语义差异。**

- **链接: [https://arxiv.org/pdf/2604.00017](https://arxiv.org/pdf/2604.00017)**

> **作者:** Orlova Anastasia
>
> **摘要:** This article examines semantic shifts in psychological concepts across scientific and popular media discourse using methods of distributional semantics applied to Russian-language corpora. Two corpora were compiled: a scientific corpus of approximately 300 research articles from the journals Psychology. Journal of the Higher School of Economics and Vestnik of Saint Petersburg University. Psychology (767,543 tokens) and a popular science corpus consisting of texts from the online psychology platforms Yasno and Chistye kogntsii (1,199,150 tokens). After preprocessing (OCR recognition, lemmatization, removal of stop words and non-informative characters), the corpora were analyzed through frequency analysis, clustering, and the identification of semantic associations. The results reveal significant differences in vocabulary and conceptual framing between the two discourse types: scientific texts emphasize methodological and clinical terminology, while popular science materials foreground everyday experience and therapeutic practice. A comparison of semantic associations for key concepts such as burnout and depression shows that scientific discourse links these terms to psychological resources, symptomatology, and diagnostic constructs, whereas popular science discourse frames them through personal narratives, emotions, and everyday situations. These findings demonstrate a clear shift from precise professional terminology toward more generalized and experiential meanings in popular media discourse and confirm the effectiveness of distributional semantics methods for identifying semantic transformations of psychological concepts across different communicative contexts.
>
---
#### [new 002] Eyla: Toward an Identity-Anchored LLM Architecture with Integrated Biological Priors -- Vision, Implementation Attempt, and Lessons from AI-Assisted Development
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI系统设计任务，旨在解决LLM身份一致性问题。提出Eyla架构，整合生物先验，但实现失败，分析失败原因并提供改进建议。**

- **链接: [https://arxiv.org/pdf/2604.00009](https://arxiv.org/pdf/2604.00009)**

> **作者:** Arif Aditto
>
> **备注:** 8 pages, 3 tables, 25 references. Preprint under review for workshop submission
>
> **摘要:** We present the design rationale, implementation attempt, and failure analysis of Eyla, a proposed identity-anchored LLM architecture that integrates biologically-inspired subsystems -- including HiPPO-initialized state-space models, zero-initialized adapters, episodic memory retrieval, and calibrated uncertainty training -- into a unified agent operating system running on consumer hardware. Unlike existing approaches that optimize models for generic helpfulness, Eyla targets identity consistency: the ability to maintain a coherent self-model under adversarial pressure, admit uncertainty, and resist manipulation. We propose the Identity Consistency Score (ICS), a novel benchmark for evaluating this property across LLMs. We then present an honest account of attempting to implement this architecture using AI coding assistants (Claude Code, Cursor) as a non-programmer, documenting a $1,000+ failure that produced a 1.27B parameter model with 86 brain subsystems contributing less than 2% to output. Our analysis identifies five systematic failure modes of AI-assisted development for novel architectures and offers concrete recommendations. To our knowledge, this is the first paper to combine an architectural vision with a documented first-person failure analysis of AI-assisted LLM development, providing lessons for both the AI systems and AI-assisted software engineering communities.
>
---
#### [new 003] Locally Confident, Globally Stuck: The Quality-Exploration Dilemma in Diffusion Language Models
- **分类: cs.CL**

- **简介: 该论文研究扩散语言模型的生成质量与探索性之间的权衡问题。针对随机解码质量下降和探索受限的矛盾，提出一种改进的采样方法，提升生成质量和多样性。任务为语言生成中的质量-探索平衡。**

- **链接: [https://arxiv.org/pdf/2604.00375](https://arxiv.org/pdf/2604.00375)**

> **作者:** Liancheng Fang; Aiwei Liu; Henry Peng Zou; Yankai Chen; Enze Ma; Leyi Pan; Chunyu Miao; Wei-Chieh Huang; Xue Liu; Philip S. Yu
>
> **摘要:** Diffusion large language models (dLLMs) theoretically permit token decoding in arbitrary order, a flexibility that could enable richer exploration of reasoning paths than autoregressive (AR) LLMs. In practice, however, random-order decoding often hurts generation quality. To mitigate this, low-confidence remasking improves single-sample quality (e.g., Pass@$1$) by prioritizing confident tokens, but it also suppresses exploration and limits multi-sample gains (e.g., Pass@$k$), creating a fundamental quality--exploration dilemma. In this paper, we provide a unified explanation of this dilemma. We show that low-confidence remasking improves a myopic proxy for quality while provably constraining the entropy of the induced sequence distribution. To overcome this limitation, we characterize the optimal distribution that explicitly balances quality and exploration, and develop a simple Independent Metropolis--Hastings sampler that approximately targets this distribution during decoding. Experiments across a range of reasoning benchmarks including MATH500, AIME24/25, HumanEval, and MBPP show that our approach yields better exploration-quality tradeoff than both random and low-confidence remasking.
>
---
#### [new 004] Asymmetric Actor-Critic for Multi-turn LLM Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话系统任务，旨在提升多轮交互中LLM代理的可靠性。通过设计异构的actor-critic框架，利用大模型生成、小模型监督，提高任务成功率与稳定性。**

- **链接: [https://arxiv.org/pdf/2604.00304](https://arxiv.org/pdf/2604.00304)**

> **作者:** Shuli Jiang; Zhaoyang Zhang; Yi Zhang; Shuo Yang; Wei Xia; Stefano Soatto
>
> **备注:** 19 pages
>
> **摘要:** Large language models (LLMs) exhibit strong reasoning and conversational abilities, but ensuring reliable behavior in multi-turn interactions remains challenging. In many real-world applications, agents must succeed in one-shot settings where retries are impossible. Existing approaches either rely on reflection or post-hoc evaluation, which require additional attempts, or assume fully trainable models that cannot leverage proprietary LLMs. We propose an asymmetric actor-critic framework for reliable conversational agents. A powerful proprietary LLM acts as the actor, while a smaller open-source critic provides runtime supervision, monitoring the actor's actions and intervening within the same interaction trajectory. Unlike training-based actor-critic methods, our framework supervises a fixed actor operating in open-ended conversational environments. The design leverages a generation-verification asymmetry: while high-quality generation requires large models, effective oversight can often be achieved by smaller ones. We further introduce a data generation pipeline that produces supervision signals for critic fine-tuning without modifying the actor. Experiments on $\tau$-bench and UserBench show that our approach significantly improves reliability and task success over strong single-agent baselines. Moreover, lightweight open-source critics rival or surpass larger proprietary models in the critic role, and critic fine-tuning yields additional gains over several state-of-the-art methods.
>
---
#### [new 005] More Human, More Efficient: Aligning Annotations with Quantized SLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的文本标注任务，旨在解决人工标注效率低、模型偏差大等问题。通过微调量化小模型，提升标注一致性与可复现性。**

- **链接: [https://arxiv.org/pdf/2604.00586](https://arxiv.org/pdf/2604.00586)**

> **作者:** Jiayu Wang; Junyoung Lee
>
> **摘要:** As Large Language Model (LLM) capabilities advance, the demand for high-quality annotation of exponentially increasing text corpora has outpaced human capacity, leading to the widespread adoption of LLMs in automatic evaluation and annotation. However, proprietary LLMs often exhibit systematic biases that diverge from human expert consensus, lacks reproducibility, and raises data privacy concerns. Our work examines the viability of finetuning a quantized Small Language Model of 1.7B parameter size on limited human-annotated data to serve as a highly aligned, deterministic evaluator and annotator. By implementing a custom, multi-dimensional rubric framework and simple augmentation and regularization techniques, the proposed approach achieves higher inter-annotator agreement (0.23 points increase in Krippendorff's $\alpha$) than the best performing state-of-the-art proprietary LLM. We also demonstrate the generalizability of the proposed training pipeline on a separate emotion classification task. The results show that task-specific alignment and efficient 4-bit quantized fine-tuning provide superior open-source alternative to using proprietary models for evaluation and annotation. Our finetuning approach is publicly available at this https URL.
>
---
#### [new 006] LangMARL: Natural Language Multi-Agent Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文属于多智能体强化学习任务，解决LLM在动态环境中协调策略演化的问题。提出LangMARL框架，通过语言空间的信用分配和梯度进化提升协作效率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.00722](https://arxiv.org/pdf/2604.00722)**

> **作者:** Huaiyuan Yao; Longchao Da; Xiaoou Liu; Charles Fleming; Tianlong Chen; Hua Wei
>
> **备注:** 20 pages, 12 figures
>
> **摘要:** Large language model (LLM) agents struggle to autonomously evolve coordination strategies in dynamic environments, largely because coarse global outcomes obscure the causal signals needed for local policy refinement. We identify this bottleneck as a multi-agent credit assignment problem, which has long been studied in classical multi-agent reinforcement learning (MARL) but remains underaddressed in LLM-based systems. Building on this observation, we propose LangMARL, a framework that brings credit assignment and policy gradient evolution from cooperative MARL into the language space. LangMARL introduces agent-level language credit assignment, pioneers gradient evolution in language space for policy improvement, and summarizes task-relevant causal relations from replayed trajectories to provide dense feedback and improve convergence under sparse rewards. Extensive experiments across diverse cooperative multi-agent tasks demonstrate improved sample efficiency, interpretability, and strong generalization.
>
---
#### [new 007] Universal YOCO for Efficient Depth Scaling
- **分类: cs.CL**

- **简介: 该论文属于大语言模型优化任务，旨在解决推理效率与深度扩展的矛盾。提出YOCO-U架构，结合递归计算与高效注意力机制，提升模型效率与性能。**

- **链接: [https://arxiv.org/pdf/2604.01220](https://arxiv.org/pdf/2604.01220)**

> **作者:** Yutao Sun; Li Dong; Tianzhu Ye; Shaohan Huang; Jianyong Wang; Furu Wei
>
> **摘要:** The rise of test-time scaling has remarkably boosted the reasoning and agentic proficiency of Large Language Models (LLMs). Yet, standard Transformers struggle to scale inference-time compute efficiently, as conventional looping strategies suffer from high computational overhead and a KV cache that inflates alongside model depth. We present Universal YOCO (YOCO-U), which combines the YOCO decoder-decoder architecture with recursive computation to achieve a synergistic effect greater than either alone. Built on the YOCO framework, YOCO-U implements a Universal Self-Decoder that performs multiple iterations via parameter sharing, while confining the iterative process to shallow, efficient-attention layers. This combination yields a favorable capability-efficiency tradeoff that neither YOCO nor recursion achieves independently. The YOCO architecture provides a constant global KV cache and linear pre-filling, while partial recursion enhances representational depth with limited overhead. Together, YOCO-U improves token utility and scaling behavior while maintaining efficient inference. Empirical results confirm that YOCO-U remains highly competitive in general and long-context benchmarks, demonstrating that the integration of efficient-attention architectures and recursive computation is a promising direction for scalable LLMs.
>
---
#### [new 008] Common TF-IDF variants arise as key components in the test statistic of a penalized likelihood-ratio test for word burstiness
- **分类: cs.CL; cs.IR; math.ST**

- **简介: 该论文属于自然语言处理任务，旨在解释TF-IDF的统计基础。通过惩罚似然比检验框架，揭示TF-IDF与词频突变检测的关系，提出新的术语加权方法。**

- **链接: [https://arxiv.org/pdf/2604.00672](https://arxiv.org/pdf/2604.00672)**

> **作者:** Zeyad Ahmed; Paul Sheridan; Michael McIsaac; Aitazaz A. Farooque
>
> **备注:** 27 pages, 3 tables, 7 figures, accepted in Discover Computing 2026
>
> **摘要:** TF-IDF is a classical formula that is widely used for identifying important terms within documents. We show that TF-IDF-like scores arise naturally from the test statistic of a penalized likelihood-ratio test setup capturing word burstiness (also known as word over-dispersion). In our framework, the alternative hypothesis captures word burstiness by modeling a collection of documents according to a family of beta-binomial distributions with a gamma penalty term on the precision parameter. In contrast, the null hypothesis assumes that words are binomially distributed in collection documents, a modeling approach that fails to account for word burstiness. We find that a term-weighting scheme given rise to by this test statistic performs comparably to TF-IDF on document classification tasks. This paper provides insights into TF-IDF from a statistical perspective and underscores the potential of hypothesis testing frameworks for advancing term-weighting scheme development.
>
---
#### [new 009] Brevity Constraints Reverse Performance Hierarchies in Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型在特定任务中的性能问题，发现大模型因冗长输出导致错误，通过约束回复长度提升准确率并反转性能差距。属于自然语言处理任务。**

- **链接: [https://arxiv.org/pdf/2604.00025](https://arxiv.org/pdf/2604.00025)**

> **作者:** MD Azizul Hakim
>
> **摘要:** Standard evaluation protocols reveal a counterintuitive phenomenon: on 7.7% of benchmark problems spanning five datasets, larger language models underperform smaller ones by 28.4 percentage points despite 10-100x more parameters. Through systematic evaluation of 31 models (0.5B-405B parameters) across 1,485 problems, we identify the mechanism as spontaneous scale-dependent verbosity that introduces errors through overelaboration. Causal intervention experiments demonstrate this reflects correctable prompt design rather than fundamental capability limitations. Constraining large models to produce brief responses improves accuracy by 26 percentage points and reduces performance gaps by up to two-thirds. Most critically, brevity constraints completely reverse performance hierarchies on mathematical reasoning and scientific knowledge benchmarks, with large models achieving 7.7-15.9 percentage point advantages over small models -- direct inversions of the original gaps. These reversals prove large models possess superior latent capabilities that universal prompting masks. We validate findings through three independent contamination tests and demonstrate inverse scaling operates continuously across the full parameter spectrum, with dataset-specific optimal scales ranging from 0.5B to 3.0B parameters. Our results establish that maximizing large model performance requires scale-aware prompt engineering rather than universal evaluation protocols, with immediate implications for deployment: prompt adaptation simultaneously improves accuracy and reduces computational costs.
>
---
#### [new 010] Embarrassingly Simple Self-Distillation Improves Code Generation
- **分类: cs.CL**

- **简介: 该论文属于代码生成任务，旨在提升大语言模型的代码生成能力。通过自蒸馏方法，利用模型自身输出进行微调，显著提升了代码生成效果。**

- **链接: [https://arxiv.org/pdf/2604.01193](https://arxiv.org/pdf/2604.01193)**

> **作者:** Ruixiang Zhang; Richard He Bai; Huangjie Zheng; Navdeep Jaitly; Ronan Collobert; Yizhe Zhang
>
> **摘要:** Can a large language model (LLM) improve at code generation using only its own raw outputs, without a verifier, a teacher model, or reinforcement learning? We answer in the affirmative with simple self-distillation (SSD): sample solutions from the model with certain temperature and truncation configurations, then fine-tune on those samples with standard supervised fine-tuning. SSD improves Qwen3-30B-Instruct from 42.4% to 55.3% pass@1 on LiveCodeBench v6, with gains concentrating on harder problems, and it generalizes across Qwen and Llama models at 4B, 8B, and 30B scale, including both instruct and thinking variants. To understand why such a simple method can work, we trace these gains to a precision-exploration conflict in LLM decoding and show that SSD reshapes token distributions in a context-dependent way, suppressing distractor tails where precision matters while preserving useful diversity where exploration matters. Taken together, SSD offers a complementary post-training direction for improving LLM code generation.
>
---
#### [new 011] $\texttt{YC-Bench}$: Benchmarking AI Agents for Long-Term Planning and Consistent Execution
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出YC-Bench，用于评估AI代理在长期规划和执行中的一致性。解决AI在复杂任务中保持战略一致性的难题，通过模拟一年的创业环境进行测试。**

- **链接: [https://arxiv.org/pdf/2604.01212](https://arxiv.org/pdf/2604.01212)**

> **作者:** Muyu He; Adit Jain; Anand Kumar; Vincent Tu; Soumyadeep Bakshi; Sachin Patro; Nazneen Rajani
>
> **备注:** 16 pages, 10 figures
>
> **摘要:** As LLM agents tackle increasingly complex tasks, a critical question is whether they can maintain strategic coherence over long horizons: planning under uncertainty, learning from delayed feedback, and adapting when early mistakes compound. We introduce $\texttt{YC-Bench}$, a benchmark that evaluates these capabilities by tasking an agent with running a simulated startup over a one-year horizon spanning hundreds of turns. The agent must manage employees, select task contracts, and maintain profitability in a partially observable environment where adversarial clients and growing payroll create compounding consequences for poor decisions. We evaluate 12 models, both proprietary and open source, across 3 seeds each. Only three models consistently surpass the starting capital of \$200K, with Claude Opus 4.6 achieving the highest average final funds at \$1.27 M, followed by GLM-5 at \$1.21 M at 11$\times$ lower inference cost. Scratchpad usage, the sole mechanism for persisting information across context truncation, is the strongest predictor of success, and adversarial client detection is the primary failure mode, accounting for $47\%$ of bankruptcies. Our analysis reveals that frontier models still fail through distinct failure modes such as over-parallelization, demonstrating the capability gaps for long-horizon performance. $\texttt{YC-Bench}$ is open-source, reproducible, and configurable.
>
---
#### [new 012] From Baselines to Preferences: A Comparative Study of LoRA/QLoRA and Preference Optimization for Mental Health Text Classification
- **分类: cs.CL**

- **简介: 论文研究心理健康文本分类任务，探讨LoRA/QLoRA与偏好优化方法的适用性。对比不同优化策略的效果，提供选择训练策略的实用框架。**

- **链接: [https://arxiv.org/pdf/2604.00773](https://arxiv.org/pdf/2604.00773)**

> **作者:** Mihael Arcan
>
> **摘要:** Mental health text classification has rapidly adopted modern adaptation methods, yet practical guidance on which optimization strategy to use, when, and why remains limited. This paper presents a systematic comparative study of optimization pathways for a joint mental-health classification task, moving from strong vanilla baselines to progressively more specialized techniques. We first establish classical and encoder references, then examine parameter-efficient supervised fine-tuning with LoRA/QLoRA under multiple objective and optimization settings, and finally evaluate preference-based optimization with DPO, ORPO, and KTO, including class-rebalanced training. Rather than emphasizing a single headline score, we focus on methodological insight: how performance changes with objective formulation, adapter choice, optimizer behavior, context windowing, and class-balance intervention. The results show that optimization effects are highly method-dependent: some approaches deliver stable, transferable gains, while others are sensitive to configuration and data balance. Preference optimization, in particular, exhibits large variation across objectives, indicating that method selection is more consequential than simply adding a preference-training stage. The central contribution is a clear optimization narrative for mental health NLP: start from transparent baselines, apply controlled tuning, and use preference optimization selectively where its gains are demonstrable. This provides a reproducible and practically grounded framework for choosing effective training strategies beyond architecture choice alone.
>
---
#### [new 013] WHBench: Evaluating Frontier LLMs with Expert-in-the-Loop Validation on Women's Health Topics
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文提出WHBench，针对女性健康领域评估大语言模型，解决医疗指南准确性与公平性问题，通过专家验证测试模型表现。**

- **链接: [https://arxiv.org/pdf/2604.00024](https://arxiv.org/pdf/2604.00024)**

> **作者:** Sneha Maurya; Pragya Saboo; Girish Kumar
>
> **摘要:** Large language models are increasingly used for medical guidance, but women's health remains under-evaluated in benchmark design. We present the Women's Health Benchmark (WHBench), a targeted evaluation suite of 47 expert-crafted scenarios across 10 women's health topics, designed to expose clinically meaningful failure modes including outdated guidelines, unsafe omissions, dosing errors, and equity-related blind spots. We evaluate 22 models using a 23-criterion rubric spanning clinical accuracy, completeness, safety, communication quality, instruction following, equity, uncertainty handling, and guideline adherence, with safety-weighted penalties and server-side score recalculation. Across 3,102 attempted responses (3,100 scored), no model mean performance exceeds 75 percent; the best model reaches 72.1 percent. Even top models show low fully correct rates and substantial variation in harm rates. Inter-rater reliability is moderate at the response label level but high for model ranking, supporting WHBench utility for comparative system evaluation while highlighting the need for expert oversight in clinical deployment. WHBench provides a public, failure-mode-aware benchmark to track safer and more equitable progress in womens health AI.
>
---
#### [new 014] Emotion Entanglement and Bayesian Inference for Multi-Dimensional Emotion Understanding
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多维情感理解任务，旨在解决现有基准依赖短文本和预定义标签的问题。提出EmoScene基准和贝叶斯推理框架，提升情感共现建模与预测一致性。**

- **链接: [https://arxiv.org/pdf/2604.00819](https://arxiv.org/pdf/2604.00819)**

> **作者:** Hemanth Kotaprolu; Kishan Maharaj; Raey Zhao; Abhijit Mishra; Pushpak Bhattacharyya
>
> **备注:** 15 pages in total, 8 Figures, 2 Tables
>
> **摘要:** Understanding emotions in natural language is inherently a multi-dimensional reasoning problem, where multiple affective signals interact through context, interpersonal relations, and situational cues. However, most existing emotion understanding benchmarks rely on short texts and predefined emotion labels, reducing this process to independent label prediction and ignoring the structured dependencies among emotions. To address this limitation, we introduce Emotional Scenarios (EmoScene), a theory-grounded benchmark of 4,731 context-rich scenarios annotated with an 8-dimensional emotion vector derived from Plutchik's basic emotions. We evaluate six instruction-tuned large language models in a zero-shot setting and observe modest performance, with the best model achieving a Macro F1 of 0.501, highlighting the difficulty of context-aware multi-label emotion prediction. Motivated by the observation that emotions rarely occur independently, we further propose an entanglement-aware Bayesian inference framework that incorporates emotion co-occurrence statistics to perform joint posterior inference over the emotion vector. This lightweight post-processing improves structural consistency of predictions and yields notable gains for weaker models (e.g., +0.051 Macro F1 for Qwen2.5-7B). EmoScene therefore provides a challenging benchmark for studying multi-dimensional emotion understanding and the limitations of current language models.
>
---
#### [new 015] ORBIT: Scalable and Verifiable Data Generation for Search Agents on a Tight Budget
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文提出ORBIT，一个用于搜索代理的可扩展且可验证的数据集，解决人工标注成本高和预处理复杂的问题。通过四阶段框架生成20K个需要多步推理的问答对，提升小模型性能。**

- **链接: [https://arxiv.org/pdf/2604.01195](https://arxiv.org/pdf/2604.01195)**

> **作者:** Nandan Thakur; Zijian Chen; Xueguang Ma; Jimmy Lin
>
> **摘要:** Search agents, which integrate language models (LMs) with web search, are becoming crucial for answering complex user queries. Constructing training datasets for deep research tasks, involving multi-step retrieval and reasoning, remains challenging due to expensive human annotation, or cumbersome prerequisites. In this work, we introduce ORBIT, a training dataset with 20K reasoning-intensive queries with short verifiable answers, generated using a frugal framework without relying on paid API services. The modular framework relies on four stages: seed creation, question--answer pair generation, and two stages of verification: self and external. ORBIT spans 15 domains and each training pair requires 4--5 reasoning steps, with external search verification required from the complete web. We train Qwen3-4B as the base model on ORBIT using GRPO and evaluate it on Wikipedia question answering tasks. Extensive experiment results demonstrate that ORBIT-4B achieves strong performance among sub-4B LLMs as search agents, proving the utility of synthetic datasets. Our framework, code and datasets are open-sourced and available publicly.
>
---
#### [new 016] The Chronicles of RiDiC: Generating Datasets with Controlled Popularity Distribution for Long-form Factuality Evaluation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型事实性评估任务，旨在解决长文本生成事实性评价的问题。通过构建可控流行度分布的数据集，评估大模型的长文本生成准确性。**

- **链接: [https://arxiv.org/pdf/2604.00019](https://arxiv.org/pdf/2604.00019)**

> **作者:** Pavel Braslavski; Dmitrii Iarosh; Nikita Sushko; Andrey Sakhovskiy; Vasily Konovalov; Elena Tutubalina; Alexander Panchenko
>
> **备注:** Accepted to LREC 2026
>
> **摘要:** We present a configurable pipeline for generating multilingual sets of entities with specified characteristics, such as domain, geographical location and popularity, using data from Wikipedia and Wikidata. These datasets are intended for evaluating the factuality of LLMs' long-form generation, thereby complementing evaluation based on short-form QA datasets. We present the RiDiC dataset as an example of this approach. RiDiC contains 3,000 entities from three domains -- rivers, natural disasters, and car models -- spanning different popularity tiers. Each entity is accompanied by its geographical location, English and Chinese names (if available) and relevant English and Chinese Wikipedia content, which is used to evaluate LLMs' responses. Generations about RiDiC entities were obtained from three LLMs in English and Chinese. These were then evaluated using a third-party factuality checker, which showed that entities from our dataset caused even frontier models to hallucinate. To facilitate the evaluation of LLMs' long-form factuality in multiple languages, the code, data, and generation/evaluation scripts have been released.
>
---
#### [new 017] LLM REgression with a Latent Iterative State Head
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出RELISH，一种用于文本回归的轻量级架构，通过迭代优化潜在状态直接预测数值，解决传统方法效率低的问题。**

- **链接: [https://arxiv.org/pdf/2604.01206](https://arxiv.org/pdf/2604.01206)**

> **作者:** Yiheng Su; Matthew Lease
>
> **摘要:** We present RELISH (REgression with a Latent Iterative State Head), a novel, lightweight architecture designed for text regression with large language models. Rather than decoding numeric targets as text or aggregating multiple generated outputs, RELISH predicts scalar values directly from frozen LLM representations by iteratively refining a learned latent state through cross-attention over token-level representations, and then mapping the final state to a point estimate with a linear regressor. Across five datasets, four LLM backbones, and two LLM training regimes, RELISH consistently outperforms prior baselines from all three major LLM regression families, including autoregressive decoding, regression-aware inference, and existing predictive head methods. Despite these gains, RELISH remains highly parameter-efficient, requiring only 3.4-3.7M trainable parameters across frozen LLM backbones (only 0.01-0.04% additional overhead), far less than LoRA-based alternatives that grow with model size (0.26-0.42%).
>
---
#### [new 018] Finding and Reactivating Post-Trained LLMs' Hidden Safety Mechanisms
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型安全任务，旨在解决后训练导致的LLM安全性下降问题。通过重新激活被抑制的安全机制，提升模型安全性而不影响推理能力。**

- **链接: [https://arxiv.org/pdf/2604.00012](https://arxiv.org/pdf/2604.00012)**

> **作者:** Mingjie Li; Wai Man Si; Michael Backes; Yang Zhang; Yisen Wang
>
> **摘要:** Despite the impressive performance of general-purpose large language models (LLMs), they often require fine-tuning or post-training to excel at specific tasks. For instance, large reasoning models (LRMs), such as the DeepSeek-R1 series, demonstrate strong reasoning capabilities after post-training different general large language models on diverse chain-of-thought (CoT) datasets. However, this additional training frequently comes at the cost of reduced safety, as the fine-tuned or post-trained models tend to exhibit more harmful behaviors compared with the regular LLMs before post-training or fine-tuning, potentially leading to harmful outcomes due to their enhanced capabilities. Taking LRMs as an example, we first investigate the underlying cause of this safety degradation in this paper. Our analysis reveals that post-training can mask the original safety mechanisms of the base LLM, while over-amplifying representations related to their post-training ability. But luckily, we also find that LRMs' safety mechanisms still exist instead of being removed during their post-training. Based on these findings, we propose a lightweight and cost-effective solution called SafeReAct that restores the suppressed safety behaviors by aligning with LoRA adapters on a few layers. Experiments on four state-of-the-art LRMs show that our method significantly improves safety on harmful prompts without compromising reasoning performance. Besides LRMs, additional results on other domain-specific LLMs, like medical models, further confirm the generality and effectiveness of our approach.
>
---
#### [new 019] Disentangling Prompt Element Level Risk Factors for Hallucinations and Omissions in Mental Health LLM Responses
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于心理健康大语言模型安全评估任务，旨在识别导致幻觉和遗漏的风险因素。通过构建UTCO框架进行系统测试，分析了2075个提示中的错误情况。**

- **链接: [https://arxiv.org/pdf/2604.00014](https://arxiv.org/pdf/2604.00014)**

> **作者:** Congning Ni; Sarvech Qadir; Bryan Steitz; Mihir Sachin Vaidya; Qingyuan Song; Lantian Xia; Shelagh Mulvaney; Siru Liu; Hyeyoung Ryu; Leah Hecht; Amy Bucher; Christopher Symons; Laurie Novak; Susannah L. Rose; Murat Kantarcioglu; Bradley Malin; Zhijun Yin
>
> **备注:** Submitted to AMIA 2026 Annual Symposium (under review)
>
> **摘要:** Mental health concerns are often expressed outside clinical settings, including in high-distress help seeking, where safety-critical guidance may be needed. Consumer health informatics systems increasingly incorporate large language models (LLMs) for mental health question answering, yet many evaluations underrepresent narrative, high-distress inquiries. We introduce UTCO (User, Topic, Context, Tone), a prompt construction framework that represents an inquiry as four controllable elements for systematic stress testing. Using 2,075 UTCO-generated prompts, we evaluated Llama 3.3 and annotated hallucinations (fabricated or incorrect clinical content) and omissions (missing clinically necessary or safety-critical guidance). Hallucinations occurred in 6.5% of responses and omissions in 13.2%, with omissions concentrated in crisis and suicidal ideation prompts. Across regression, element-specific matching, and similarity-matched comparisons, failures were most consistently associated with context and tone, while user-background indicators showed no systematic differences after balancing. These findings support evaluating omissions as a primary safety outcome and moving beyond static benchmark question sets.
>
---
#### [new 020] Can LLMs Perceive Time? An Empirical Investigation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，研究大语言模型对时间的感知能力。通过实验发现模型无法准确估计任务耗时，导致调度和规划中的错误。**

- **链接: [https://arxiv.org/pdf/2604.00010](https://arxiv.org/pdf/2604.00010)**

> **作者:** Aniketh Garikaparthi
>
> **备注:** ICLR 2026 I Can't Believe It's Not Better Workshop
>
> **摘要:** Large language models cannot estimate how long their own tasks take. We investigate this limitation through four experiments across 68 tasks and four model families. Pre-task estimates overshoot actual duration by 4--7$\times$ ($p < 0.001$), with models predicting human-scale minutes for tasks completing in seconds. Relative ordering fares no better: on task pairs designed to expose heuristic reliance, models score at or below chance (GPT-5: 18\% on counter-intuitive pairs, $p = 0.033$), systematically failing when complexity labels mislead. Post-hoc recall is disconnected from reality -- estimates diverge from actuals by an order of magnitude in either direction. These failures persist in multi-step agentic settings, with errors of 5--10$\times$. The models possess propositional knowledge about duration from training but lack experiential grounding in their own inference time, with practical implications for agent scheduling, planning and time-critical scenarios.
>
---
#### [new 021] Do LLMs Know What Is Private Internally? Probing and Steering Contextual Privacy Norms in Large Language Model Representations
- **分类: cs.CL**

- **简介: 该论文属于隐私保护任务，旨在解决LLMs泄露隐私的问题。通过分析上下文完整性理论，发现模型内部编码了隐私规范，但行为与之不一致，提出参数化控制方法提升隐私保护效果。**

- **链接: [https://arxiv.org/pdf/2604.00209](https://arxiv.org/pdf/2604.00209)**

> **作者:** Haoran Wang; Li Xiong; Kai Shu
>
> **摘要:** Large language models (LLMs) are increasingly deployed in high-stakes settings, yet they frequently violate contextual privacy by disclosing private information in situations where humans would exercise discretion. This raises a fundamental question: do LLMs internally encode contextual privacy norms, and if so, why do violations persist? We present the first systematic study of contextual privacy as a structured latent representation in LLMs, grounded in contextual integrity (CI) theory. Probing multiple models, we find that the three norm-determining CI parameters (information type, recipient, and transmission principle) are encoded as linearly separable and functionally independent directions in activation space. Despite this internal structure, models still leak private information in practice, revealing a clear gap between concept representation and model behavior. To bridge this gap, we introduce CI-parametric steering, which independently intervenes along each CI dimension. This structured control reduces privacy violations more effectively and predictably than monolithic steering. Our results demonstrate that contextual privacy failures arise from misalignment between representation and behavior rather than missing awareness, and that leveraging the compositional structure of CI enables more reliable contextual privacy control, shedding light on potential improvement of contextual privacy understanding in LLMs.
>
---
#### [new 022] Stochastic Attention: Connectome-Inspired Randomized Routing for Expressive Linear-Time Attention
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出Stochastic Attention，用于提升注意力机制的表达能力。解决高效注意力的全局覆盖问题，通过随机排列实现全局路由，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.00754](https://arxiv.org/pdf/2604.00754)**

> **作者:** Zehao Jin; Yanan Sui
>
> **摘要:** The whole-brain connectome of a fruit fly comprises over 130K neurons connected with a probability of merely 0.02%, yet achieves an average shortest path of only 4.4 hops. Despite being highly structured at the circuit level, the network's long-range connections are broadly distributed across brain regions, functioning as stochastic shortcuts that enable efficient global communication. Inspired by this observation, we propose Stochastic Attention (SA), a drop-in enhancement for sliding-window attention (SWA) that applies a random permutation to the token sequence before windowed attention and restores the original order afterward. This transforms the fixed local window into a stochastic global one within the same $O(nw)$ per-layer budget. Through depth, independently sampled permutations yield exponentially growing receptive fields, achieving full sequence coverage in $O(\log_w n)$ layers versus $O(n/w)$ for SWA. We validate SA in two settings: pre-training language models from scratch, where a gated SA + SWA combination achieves the best average zero-shot accuracy, and training-free inference on Qwen3-8B and Qwen3-30B-A3B, where SA consistently outperforms SWA and matches or exceeds Mixture of Block Attention at comparable compute budgets. These results suggest that connectome-inspired stochastic routing is a practical primitive for improving the expressivity of efficient attention, complementary to existing linear and sparse approaches.
>
---
#### [new 023] Valency Classification of Mapudungun Verbal Roots. Established by the language's own morphotactics
- **分类: cs.CL**

- **简介: 该论文属于语法分析任务，旨在通过马普切语自身形态规则对动词根进行及物性分类，解决动词类别确认问题。**

- **链接: [https://arxiv.org/pdf/2604.00789](https://arxiv.org/pdf/2604.00789)**

> **作者:** Andrés Chandía
>
> **摘要:** In the previous work, a lexical (re)categorisation -- or confirmation of the given category -- of roots identified as verbal was undertaken to determine their original category accurately. Building on this, the present paper offers an account of the valency classification of those Mapudungun roots confirmed to be verbal, using the language's own morphotactics; specifically, by examining the permissible and restricted combinations of various suffixes with roots or verbal stems in the Mapuche verb form. As with all work conducted thus far, the results presented here aim to improve the morphological analyser (Dungupeyum) with all verified findings incorporated into the system. From a theoretical perspective, we also hope to contribute to the recognition and understanding of issues related to the valency of Mapuche verb forms.
>
---
#### [new 024] Criterion Validity of LLM-as-Judge for Business Outcomes in Conversational Commerce
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话评估任务，旨在检验LLM作为评价者在商业场景中的效度。通过实证研究，发现评分维度与业务转化存在异质性，提出重新加权以提升效度。**

- **链接: [https://arxiv.org/pdf/2604.00022](https://arxiv.org/pdf/2604.00022)**

> **作者:** Liang Chen; Qi Liu; Wenhuan Lin; Feng Liang
>
> **摘要:** Multi-dimensional rubric-based dialogue evaluation is widely used to assess conversational AI, yet its criterion validity -- whether quality scores are associated with the downstream outcomes they are meant to serve -- remains largely untested. We address this gap through a two-phase study on a major Chinese matchmaking platform, testing a 7-dimension evaluation rubric (implemented via LLM-as-Judge) against verified business conversion. Our findings concern rubric design and weighting, not LLM scoring accuracy: any judge using the same rubric would face the same structural issue. The core finding is dimension-level heterogeneity: in Phase 2 (n=60 human conversations, stratified sample, verified labels), Need Elicitation (D1: rho=0.368, p=0.004) and Pacing Strategy (D3: rho=0.354, p=0.006) are significantly associated with conversion after Bonferroni correction, while Contextual Memory (D5: rho=0.018, n.s.) shows no detectable association. This heterogeneity causes the equal-weighted composite (rho=0.272) to underperform its best dimensions -- a composite dilution effect that conversion-informed reweighting partially corrects (rho=0.351). Logistic regression controlling for conversation length confirms D3's association strengthens (OR=3.18, p=0.006), ruling out a length confound. An initial pilot (n=14) mixing human and AI conversations had produced a misleading "evaluation-outcome paradox," which Phase 2 revealed as an agent-type confound artifact. Behavioral analysis of 130 conversations through a Trust-Funnel framework identifies a candidate mechanism: AI agents execute sales behaviors without building user trust. We operationalize these findings in a three-layer evaluation architecture and advocate criterion validity testing as standard practice in applied dialogue evaluation.
>
---
#### [new 025] MSA-Thinker: Discrimination-Calibration Reasoning with Hint-Guided Reinforcement Learning for Multimodal Sentiment Analysis
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模态情感分析任务，旨在提升模型的可解释性和泛化能力。针对现有方法的标注成本高、奖励稀疏等问题，提出融合推理与强化学习的新框架，增强模型性能与可信度。**

- **链接: [https://arxiv.org/pdf/2604.00013](https://arxiv.org/pdf/2604.00013)**

> **作者:** Miaosen Luo; Zhenhao Yang; Jieshen Long; Jinghu Sun; Yichu Liu; Sijie Mai
>
> **摘要:** Multimodal sentiment analysis aims to understand human emotions by integrating textual, auditory, and visual modalities. Although Multimodal Large Language Models (MLLMs) have achieved state-of-the-art performance via supervised fine-tuning (SFT), their end-to-end "black-box" nature limits interpretability. Existing methods incorporating Chain-of-Thought (CoT) reasoning are hindered by high annotation costs, while Reinforcement Learning (RL) faces challenges such as low exploration efficiency and sparse rewards, particularly on hard samples. To address these issues, we propose a novel training framework that integrates structured Discrimination-Calibration (DC) reasoning with Hint-based Reinforcement Learning. First, we perform cold-start SFT using high-quality CoT data synthesized by a teacher model (Qwen3Omni-30B), which inherently contains the DC structure. This equips the model with a reasoning paradigm that performs macro discrimination followed by fine-grained calibration from the initial stage. Building on this, we propose Hint-GRPO, which leverages the discrimination phase within the DC structure as a verifiable anchor during RL to provide directional hints for hard samples, guiding policy optimization and effectively mitigating the reward sparsity problem. Experiments on the Qwen2.5Omni-7B model demonstrate that our method not only achieves higher accuracy in fine-grained sentiment regression tasks but also generates high-quality structured reasoning chains. Crucially, it exhibits superior generalization capability in cross-domain evaluations. This enhances model interpretability while validating the positive contribution of explicit reasoning steps to model robustness, offering a new paradigm for building trustworthy and efficient sentiment analysis systems.
>
---
#### [new 026] Positional Cognitive Specialization: Where Do LLMs Learn To Comprehend and Speak Your Language?
- **分类: cs.CL**

- **简介: 该论文研究LLM学习新语言的机制，解决多语言适应效率问题。通过分析模型输入输出层，提出CogSym方法，仅微调部分层即可达到接近全量微调的效果。**

- **链接: [https://arxiv.org/pdf/2604.00923](https://arxiv.org/pdf/2604.00923)**

> **作者:** Luis Frentzen Salim; Lun-Wei Ku; Hsing-Kuo Kenneth Pao
>
> **备注:** Accepted to AAAI26 Main
>
> **摘要:** Adapting large language models (LLMs) to new languages is an expensive and opaque process. Understanding how language models acquire new languages and multilingual abilities is key to achieve efficient adaptation. Prior work on multilingual interpretability research focuses primarily on how trained models process multilingual instructions, leaving unexplored the mechanisms through which they acquire new languages during training. We investigate these training dynamics on decoder-only transformers through the lens of two functional cognitive specializations: language perception (input comprehension) and production (output generation). Through experiments on low-resource languages, we demonstrate how perceptual and productive specialization emerges in different regions of a language model by running layer ablation sweeps from the model's input and output directions. Based on the observed specialization patterns, we propose CogSym, a layer-wise heuristic that enables effective adaptation by exclusively fine-tuning a few early and late layers. We show that tuning only the 25% outermost layers achieves downstream task performance within 2-3% deviation from the full fine-tuning baseline. CogSym yields consistent performance with adapter methods such as LoRA, showcasing generalization beyond full fine-tuning. These findings provide insights to better understand how LLMs learn new languages and push toward accessible and inclusive language modeling.
>
---
#### [new 027] Agent Q-Mix: Selecting the Right Action for LLM Multi-Agent Systems through Reinforcement Learning
- **分类: cs.CL; stat.AP**

- **简介: 该论文属于多智能体系统任务，旨在解决复杂问题中智能体协作与通信选择的问题。提出Agent Q-Mix框架，通过强化学习优化通信拓扑，提升任务准确性和效率。**

- **链接: [https://arxiv.org/pdf/2604.00344](https://arxiv.org/pdf/2604.00344)**

> **作者:** Eric Hanchen Jiang; Levina Li; Rui Sun; Xiao Liang; Yubei Li; Yuchen Wu; Haozheng Luo; Hengli Li; Zhi Zhang; Zhaolu Kang; Kai-Wei Chang; Ying Nian Wu
>
> **摘要:** Large Language Models (LLMs) have shown remarkable performance in completing various tasks. However, solving complex problems often requires the coordination of multiple agents, raising a fundamental question: how to effectively select and interconnect these agents. In this paper, we propose \textbf{Agent Q-Mix}, a reinforcement learning framework that reformulates topology selection as a cooperative Multi-Agent Reinforcement Learning (MARL) problem. Our method learns decentralized communication decisions using QMIX value factorization, where each agent selects from a set of communication actions that jointly induce a round-wise communication graph. At its core, Agent Q-Mix combines a topology-aware GNN encoder, GRU memory, and per-agent Q-heads under a Centralized Training with Decentralized Execution (CTDE) paradigm. The framework optimizes a reward function that balances task accuracy with token cost. Across seven core benchmarks in coding, reasoning, and mathematics, Agent Q-Mix achieves the highest average accuracy compared to existing methods while demonstrating superior token efficiency and robustness against agent failure. Notably, on the challenging Humanity's Last Exam (HLE) using Gemini-3.1-Flash-Lite as a backbone, Agent Q-Mix achieves 20.8\% accuracy, outperforming Microsoft Agent Framework (19.2\%) and LangGraph (19.2\%), followed by AutoGen and Lobster by OpenClaw. These results underscore the effectiveness of learned, decentralized topology optimization in pushing the boundaries of multi-agent reasoning.
>
---
#### [new 028] Speech LLMs are Contextual Reasoning Transcribers
- **分类: cs.CL**

- **简介: 该论文属于自动语音识别（ASR）任务，旨在解决传统ASR难以利用大语言模型的上下文理解能力的问题。提出CoT-ASR方法，通过构建推理链实现更准确的语音转录。**

- **链接: [https://arxiv.org/pdf/2604.00610](https://arxiv.org/pdf/2604.00610)**

> **作者:** Keqi Deng; Ruchao Fan; Bo Ren; Yiming Wang; Jinyu Li
>
> **摘要:** Despite extensions to speech inputs, effectively leveraging the rich knowledge and contextual understanding of large language models (LLMs) in automatic speech recognition (ASR) remains non-trivial, as the task primarily involves direct speech-to-text mapping. To address this, this paper proposes chain-of-thought ASR (CoT-ASR), which constructs a reasoning chain that enables LLMs to first analyze the input speech and generate contextual analysis, thereby fully exploiting their generative capabilities. With this contextual reasoning, CoT-ASR then performs more informed speech recognition and completes both reasoning and transcription in a single pass. Moreover, CoT-ASR naturally supports user-guided transcription: while designed to self-generate reasoning, it can also seamlessly incorporate user-provided context to guide transcription, further extending ASR functionality. To reduce the modality gap, this paper introduces a CTC-guided Modality Adapter, which uses CTC non-blank token probabilities to weight LLM embeddings, efficiently aligning speech encoder outputs with the LLM's textual latent space. Experiments show that, compared to standard LLM-based ASR, CoT-ASR achieves a relative reduction of 8.7% in word error rate (WER) and 16.9% in entity error rate (EER).
>
---
#### [new 029] Are they human? Detecting large language models by probing human memory constraints
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于检测任务，旨在区分人类与大语言模型。通过测试工作记忆限制等认知现象，有效识别机器参与者。**

- **链接: [https://arxiv.org/pdf/2604.00016](https://arxiv.org/pdf/2604.00016)**

> **作者:** Simon Schug; Brenden M. Lake
>
> **备注:** Code available at this https URL
>
> **摘要:** The validity of online behavioral research relies on study participants being human rather than machine. In the past, it was possible to detect machines by posing simple challenges that were easily solved by humans but not by machines. General-purpose agents based on large language models (LLMs) can now solve many of these challenges, threatening the validity of online behavioral research. Here we explore the idea of detecting humanness by using tasks that machines can solve too well to be human. Specifically, we probe for the existence of an established human cognitive constraint: limited working memory capacity. We show that cognitive modeling on a standard serial recall task can be used to distinguish online participants from LLMs even when the latter are specifically instructed to mimic human working memory constraints. Our results demonstrate that it is viable to use well-established cognitive phenomena to distinguish LLMs from humans.
>
---
#### [new 030] An Empirical Recipe for Universal Phone Recognition
- **分类: cs.CL; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，解决多语言和低资源环境下语音识别性能不佳的问题。通过大规模数据训练，提出新方法 PhoneticXEUS，提升多语言及口音英语的识别准确率。**

- **链接: [https://arxiv.org/pdf/2603.29042](https://arxiv.org/pdf/2603.29042)**

> **作者:** Shikhar Bharadwaj; Chin-Jou Li; Kwanghee Choi; Eunjung Yeo; William Chen; Shinji Watanabe; David R. Mortensen
>
> **备注:** Submitted to Interspeech 2026. Code: this https URL
>
> **摘要:** Phone recognition (PR) is a key enabler of multilingual and low-resource speech processing tasks, yet robust performance remains elusive. Highly performant English-focused models do not generalize across languages, while multilingual models underutilize pretrained representations. It also remains unclear how data scale, architecture, and training objective contribute to multilingual PR. We present PhoneticXEUS -- trained on large-scale multilingual data and achieving state-of-the-art performance on both multilingual (17.7% PFER) and accented English speech (10.6% PFER). Through controlled ablations with evaluations across 100+ languages under a unified scheme, we empirically establish our training recipe and quantify the impact of SSL representations, data scale, and loss objectives. In addition, we analyze error patterns across language families, accented speech, and articulatory features. All data and code are released openly.
>
---
#### [new 031] When Users Change Their Mind: Evaluating Interruptible Agents in Long-Horizon Web Navigation
- **分类: cs.CL**

- **简介: 该论文研究长周期网页导航中可中断代理的性能，解决用户中途修改需求的问题。通过构建基准测试，评估不同模型在处理中断时的适应能力与效率。**

- **链接: [https://arxiv.org/pdf/2604.00892](https://arxiv.org/pdf/2604.00892)**

> **作者:** Henry Peng Zou; Chunyu Miao; Wei-Chieh Huang; Yankai Chen; Yue Zhou; Hanrong Zhang; Yaozu Wu; Liancheng Fang; Zhengyao Gu; Zhen Zhang; Kening Zheng; Fangxin Wang; Yi Nian; Shanghao Li; Wenzhe Fan; Langzhou He; Weizhi Zhang; Xue Liu; Philip S. Yu
>
> **摘要:** As LLM agents transition from short, static problem solving to executing complex, long-horizon tasks in dynamic environments, the ability to handle user interruptions, such as adding requirement or revising goals, during mid-task execution is becoming a core requirement for realistic deployment. However, existing benchmarks largely assume uninterrupted agent behavior or study interruptions only in short, unconstrained language tasks. In this paper, we present the first systematic study of interruptible agents in long-horizon, environmentally grounded web navigation tasks, where actions induce persistent state changes. We formalize three realistic interruption types, including addition, revision, and retraction, and introduce InterruptBench, a benchmark derived from WebArena-Lite that synthesizes high-quality interruption scenarios under strict semantic constraints. Using a unified interruption simulation framework, we evaluate six strong LLM backbones across single- and multi-turn interruption settings, analyzing both their effectiveness in adapting to updated intents and their efficiency in recovering from mid-task changes. Our results show that handling user interruptions effectively and efficiently during long-horizon agentic tasks remains challenging for powerful large-scale LLMs. Code and dataset are available at this https URL.
>
---
#### [new 032] OmniVoice: Towards Omnilingual Zero-Shot Text-to-Speech with Diffusion Language Models
- **分类: cs.CL; eess.AS**

- **简介: 该论文提出OmniVoice，解决多语言零样本文本转语音任务。通过创新架构直接将文本映射到声学标记，提升效率与清晰度。**

- **链接: [https://arxiv.org/pdf/2604.00688](https://arxiv.org/pdf/2604.00688)**

> **作者:** Han Zhu; Lingxuan Ye; Wei Kang; Zengwei Yao; Liyong Guo; Fangjun Kuang; Zhifeng Han; Weiji Zhuang; Long Lin; Daniel Povey
>
> **摘要:** We present OmniVoice, a massive multilingual zero-shot text-to-speech (TTS) model that scales to over 600 languages. At its core is a novel diffusion language model-style discrete non-autoregressive (NAR) architecture. Unlike conventional discrete NAR models that suffer from performance bottlenecks in complex two-stage (text-to-semantic-to-acoustic) pipelines, OmniVoice directly maps text to multi-codebook acoustic tokens. This simplified approach is facilitated by two key technical innovations: (1) a full-codebook random masking strategy for efficient training, and (2) initialization from a pre-trained LLM to ensure superior intelligibility. By leveraging a 581k-hour multilingual dataset curated entirely from open-source data, OmniVoice achieves the broadest language coverage to date and delivers state-of-the-art performance across Chinese, English, and diverse multilingual benchmarks. Our code and pre-trained models are publicly available at this https URL.
>
---
#### [new 033] Benchmark for Assessing Olfactory Perception of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出嗅觉感知基准（OP），评估大语言模型对气味的推理能力，解决模型在嗅觉信息处理上的能力问题，通过多种任务和分子表示形式进行实验分析。**

- **链接: [https://arxiv.org/pdf/2604.00002](https://arxiv.org/pdf/2604.00002)**

> **作者:** Eftychia Makri; Nikolaos Nakis; Laura Sisson; Gigi Minsky; Leandros Tassiulas; Vahid Satarifard; Nicholas A. Christakis
>
> **摘要:** Here we introduce the Olfactory Perception (OP) benchmark, designed to assess the capability of large language models (LLMs) to reason about smell. The benchmark contains 1,010 questions across eight task categories spanning odor classification, odor primary descriptor identification, intensity and pleasantness judgments, multi-descriptor prediction, mixture similarity, olfactory receptor activation, and smell identification from real-world odor sources. Each question is presented in two prompt formats, compound names and isomeric SMILES, to evaluate the effect of molecular representations. Evaluating 21 model configurations across major model families, we find that compound-name prompts consistently outperform isomeric SMILES, with gains ranging from +2.4 to +18.9 percentage points (mean approx +7 points), suggesting current LLMs access olfactory knowledge primarily through lexical associations rather than structural molecular reasoning. The best-performing model reaches 64.4\% overall accuracy, which highlights both emerging capabilities and substantial remaining gaps in olfactory reasoning. We further evaluate a subset of the OP across 21 languages and find that aggregating predictions across languages improves olfactory prediction, with AUROC = 0.86 for the best performing language ensemble model. LLMs should be able to handle olfactory and not just visual or aural information.
>
---
#### [new 034] Agentic Tool Use in Large Language Models
- **分类: cs.CL**

- **简介: 论文探讨了大语言模型作为智能体使用工具的问题，属于自然语言处理任务。旨在解决工具使用方法分散、缺乏统一视角的问题，通过分类分析不同方法并总结挑战。**

- **链接: [https://arxiv.org/pdf/2604.00835](https://arxiv.org/pdf/2604.00835)**

> **作者:** Jinchao Hu; Meizhi Zhong; Kehai Chen; Xuefeng Bai; Min Zhang
>
> **摘要:** Large language models are increasingly being deployed as autonomous agents yet their real world effectiveness depends on reliable tools for information retrieval, computation and external action. Existing studies remain fragmented across tasks, tool types, and training settings, lacking a unified view of how tool-use methods differ and evolve. This paper organizes the literature into three paradigms: prompting as plug-and-play, supervised tool learning and reward-driven tool policy learning, analyzes their methods, strengths and failure modes, reviews the evaluation landscape and highlights key challenges, aiming to address this fragmentation and provide a more structured evolutionary view of agentic tool use.
>
---
#### [new 035] Narrative Fingerprints: Multi-Scale Author Identification via Novelty Curve Dynamics
- **分类: cs.CL; cs.DL; cs.IR**

- **简介: 该论文属于作者识别任务，旨在通过信息论新颖性曲线分析作者独特风格。研究验证了作者在文本中留下的可测量“指纹”，并展示了多尺度特征的有效性。**

- **链接: [https://arxiv.org/pdf/2604.01073](https://arxiv.org/pdf/2604.01073)**

> **作者:** Fred Zimmerman; Hilmar AI
>
> **备注:** 12 pages, 6 figures, 4 tables
>
> **摘要:** We test whether authors have characteristic "fingerprints" in the information-theoretic novelty curves of their published works. Working with two corpora -- Books3 (52,796 books, 759 qualifying authors) and PG-19 (28,439 books, 1,821 qualifying authors) -- we find that authorial voice leaves measurable traces in how novelty unfolds across a text. The signal is multi-scale: at book level, scalar dynamics (mean novelty, speed, volume, circuitousness) identify 43% of authors significantly above chance; at chapter level, SAX motif patterns in sliding windows achieve 30x-above-chance attribution, far exceeding the scalar features that dominate at book level. These signals are complementary, not redundant. We show that the fingerprint is partly confounded with genre but persists within-genre for approximately one-quarter of authors. Classical authors (Twain, Austen, Kipling) show fingerprints comparable in strength to modern authors, suggesting the phenomenon is not an artifact of contemporary publishing conventions.
>
---
#### [new 036] Optimsyn: Influence-Guided Rubrics Optimization for Synthetic Data Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，解决合成数据生成中rubric设计效率低的问题。通过优化rubric提升合成数据质量，增强模型训练效果。**

- **链接: [https://arxiv.org/pdf/2604.00536](https://arxiv.org/pdf/2604.00536)**

> **作者:** Zhiting Fan; Ruizhe Chen; Tianxiang Hu; Ru Peng; Zenan Huang; Haokai Xu; Yixin Chen; Jian Wu; Junbo Zhao; Zuozhu Liu
>
> **摘要:** Large language models (LLMs) achieve strong downstream performance largely due to abundant supervised fine-tuning (SFT) data. However, high-quality SFT data in knowledge-intensive domains such as humanities, social sciences, medicine, law, and finance is scarce because expert curation is expensive, privacy constraints are strict, and label consistency is hard to ensure. Recent work uses synthetic data, typically by prompting a generator over domain documents and filtering outputs with handcrafted rubrics. Yet rubric design is expert-dependent, transfers poorly across domains, and is often optimized through a brittle heuristic loop of writing rubrics, synthesizing data, training, inspecting results, and manually guessing revisions. This process lacks reliable quantitative feedback about how a rubric affects downstream performance. We propose evaluating synthetic data by its training utility on the target model and using this signal to guide data generation. Inspired by influence estimation, we adopt an optimizer-aware estimator that uses gradient information to quantify each synthetic sample's contribution to a target model's objective on specific tasks. Our analysis shows that even when synthetic and real samples are close in embedding space, their influence on learning can differ substantially. Based on this insight, we propose an optimization-based framework that adapts rubrics using target-model feedback. We provide lightweight guiding text and use a rubric-specialized model to generate task-conditioned rubrics. Influence score is used as the reward to optimize the rubric generator with reinforcement learning. Experiments across domains, target models, and data generators show consistent improvements and strong generalization without task-specific tuning.
>
---
#### [new 037] Can Large Language Models Self-Correct in Medical Question Answering? An Exploratory Study
- **分类: cs.CL**

- **简介: 该论文属于医疗问答任务，研究自纠正机制在医学问题解答中的有效性。通过对比不同提示方法，分析自反思是否能提升准确性。结果表明效果不一致，依赖数据集和模型。**

- **链接: [https://arxiv.org/pdf/2604.00261](https://arxiv.org/pdf/2604.00261)**

> **作者:** Zaifu Zhan; Mengyuan Cui; Rui Zhang
>
> **摘要:** Large language models (LLMs) have achieved strong performance on medical question answering (medical QA), and chain-of-thought (CoT) prompting has further improved results by eliciting explicit intermediate reasoning; meanwhile, self-reflective (self-corrective) prompting has been widely claimed to enhance model reliability by prompting LLMs to critique and revise their own reasoning, yet its effectiveness in safety-critical medical settings remains unclear. In this work, we conduct an exploratory analysis of self-reflective reasoning for medical multiple-choice question answering: using GPT-4o and GPT-4o-mini, we compare standard CoT prompting with an iterative self-reflection loop and track how predictions evolve across reflection steps on three widely used medical QA benchmarks (MedQA, HeadQA, and PubMedQA). We analyze whether self-reflection leads to error correction, error persistence, or the introduction of new errors. Our results show that self-reflective prompting does not consistently improve accuracy and its impact is highly dataset- and model-dependent: it yields modest gains on MedQA but provides limited or negative benefits on HeadQA and PubMedQA, and increasing the number of reflection steps does not guarantee better performance. These findings highlight a gap between reasoning transparency and reasoning correctness, suggesting that self-reflective reasoning is better viewed as an analytical tool for understanding model behavior rather than a standalone solution for improving medical QA reliability.
>
---
#### [new 038] Dual Optimal: Make Your LLM Peer-like with Dignity
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型对齐任务，旨在解决模型过度迎合用户、缺乏独立性的“逃避仆人”问题。通过构建PersonaKnob数据集和优化算法，提升模型的尊严与同伴性。**

- **链接: [https://arxiv.org/pdf/2604.00979](https://arxiv.org/pdf/2604.00979)**

> **作者:** Xiangqi Wang; Yue Huang; Haomin Zhuang; Kehan Guo; Xiangliang Zhang
>
> **摘要:** Current aligned language models exhibit a dual failure mode we term the Evasive Servant: they sycophantically validate flawed user beliefs while deflecting responsibility with boilerplate disclaimers. We propose the Dignified Peer framework, which counters servility with anti-sycophancy and trustworthiness, and mitigates evasiveness through empathy and creativity. Realizing this agent requires overcoming significant challenges in data supervision, objective collapse, and evaluation bias. We address these issues by introducing the PersonaKnob dataset which features a compositional partial order structure of multiple persona preference. This data is utilized alongside a tolerant constrained Lagrangian DPO algorithm that dynamically balances all persona dimensions to prevent behavioral collapse. Additionally, we employ a psychometrically calibrated Item Response Theory evaluation protocol to disentangle latent model persona capability from confounders like judge biases. Extensive empirical studies demonstrate that our approach successfully build a LLM agent with both dignity and peer.
>
---
#### [new 039] Large Language Models in the Abuse Detection Pipeline
- **分类: cs.CL; cs.CY**

- **简介: 本文探讨大语言模型在滥用检测流程中的应用，分析其在四个阶段的集成与挑战，旨在提升在线内容安全系统的效率与准确性。**

- **链接: [https://arxiv.org/pdf/2604.00323](https://arxiv.org/pdf/2604.00323)**

> **作者:** Suraj Kath; Sanket Badhe; Preet Shah; Ashwin Sampathkumar; Shivani Gupta
>
> **摘要:** Online abuse has grown increasingly complex, spanning toxic language, harassment, manipulation, and fraudulent behavior. Traditional machine-learning approaches dependent on static classifiers and labor-intensive labeling struggle to keep pace with evolving threat patterns and nuanced policy requirements. Large Language Models introduce new capabilities for contextual reasoning, policy interpretation, explanation generation, and cross-modal understanding, enabling them to support multiple stages of modern safety systems. This survey provides a lifecycle-oriented analysis of how LLMs are being integrated into the Abuse Detection Lifecycle (ADL), which we define across four stages: (I) Label \& Feature Generation, (II) Detection, (III) Review \& Appeals, and (IV) Auditing \& Governance. For each stage, we synthesize emerging research and industry practices, highlight architectural considerations for production deployment, and examine the strengths and limitations of LLM-driven approaches. We conclude by outlining key challenges including latency, cost-efficiency, determinism, adversarial robustness, and fairness and discuss future research directions needed to operationalize LLMs as reliable, accountable components of large-scale abuse-detection and governance systems.
>
---
#### [new 040] Uncertainty-Aware Variational Reward Factorization via Probabilistic Preference Bases for LLM Personalization
- **分类: cs.CL**

- **简介: 该论文属于LLM个性化任务，解决用户偏好估计不准确问题。提出VRF框架，通过概率基函数和变分分布建模用户偏好，提升个性化效果与可靠性。**

- **链接: [https://arxiv.org/pdf/2604.00997](https://arxiv.org/pdf/2604.00997)**

> **作者:** Gyuseok Lee; Wonbin Kweon; Zhenrui Yue; SeongKu Kang; Jiawei Han; Dong Wang
>
> **摘要:** Reward factorization personalizes large language models (LLMs) by decomposing rewards into shared basis functions and user-specific weights. Yet, existing methods estimate user weights from scarce data in isolation and as deterministic points, leading to inaccurate and unreliable inference. We introduce Variational Reward Factorization (VRF), an uncertainty-aware framework that represents each user's preferences as a variational distribution in a shared preference space. VRF infers user distributions via a variational encoder, derives weights through Wasserstein distance matching with shared probabilistic bases, and downweights uncertain estimates through a variance-attenuated loss. On three benchmarks, VRF outperforms all baselines across seen and unseen users, few-shot scenarios, and varying uncertainty levels, with gains extending to downstream alignment.
>
---
#### [new 041] Brainstacks: Cross-Domain Cognitive Capabilities via Frozen MoE-LoRA Stacks for Continual LLM Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Brainstacks，解决持续多领域微调问题，通过冻结的MoE-LoRA模块实现跨域认知能力，提升模型适应性与生成质量。**

- **链接: [https://arxiv.org/pdf/2604.01152](https://arxiv.org/pdf/2604.01152)**

> **作者:** Mohammad R. Abu Ayyash
>
> **备注:** 26 pages, 13 figures, 4 tables
>
> **摘要:** We present Brainstacks, a modular architecture for continual multi-domain fine-tuning of large language models that packages domain expertise as frozen adapter stacks composing additively on a shared frozen base at inference. Five interlocking components: (1) MoE-LoRA with Shazeer-style noisy top-2 routing across all seven transformer projections under QLoRA 4-bit quantization with rsLoRA scaling; (2) an inner loop performing residual boosting by freezing trained stacks and adding new ones; (3) an outer loop training sequential domain-specific stacks with curriculum-ordered dependencies; (4) null-space projection via randomized SVD constraining new stacks to subspaces orthogonal to prior directions, achieving zero forgetting in isolation; (5) an outcome-based sigmoid meta-router trained on empirically discovered domain-combination targets that selectively weights stacks, enabling cross-domain composition. Two boundary experiments: (6) PSN pretraining on a randomly initialized model; (7) per-domain RL (DPO/GRPO) validating compatibility with post-SFT alignment. Validated on TinyLlama-1.1B (4 domains, 9 stacks) and Gemma 3 12B IT (5 domains, 10 stacks), MoE-LoRA achieves 2.5x faster convergence than parameter-matched single LoRA, residual boosting breaks through the single-stack ceiling, and the routed system recovers generation quality destroyed by ungated stack accumulation. The central finding: the outcome-based router discovers that domain stacks encode transferable cognitive primitives (instruction-following clarity, numerical reasoning, procedural logic, chain-of-thought structure) rather than domain-specific knowledge, with medical prompts routing to chat+math stacks in 97% of cases despite zero medical data in those stacks.
>
---
#### [new 042] Dynin-Omni: Omnimodal Unified Large Diffusion Language Model
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Dynin-Omni，一个统一处理文本、图像、语音和视频的扩散模型，解决多模态理解和生成问题，通过共享离散令牌空间实现高效建模。**

- **链接: [https://arxiv.org/pdf/2604.00007](https://arxiv.org/pdf/2604.00007)**

> **作者:** Jaeik Kim; Woojin Kim; Jihwan Hong; Yejoon Lee; Sieun Hyeon; Mintaek Lim; Yunseok Han; Dogeun Kim; Hoeun Lee; Hyunggeun Kim; Jaeyoung Do
>
> **备注:** Project Page: this https URL
>
> **摘要:** We present Dynin-Omni, the first masked-diffusion-based omnimodal foundation model that unifies text, image, and speech understanding and generation, together with video understanding, within a single architecture. Unlike autoregressive unified models that serialize heterogeneous modalities, or compositional unified models that require orchestration with external modality-specific decoders, Dynin-Omni natively formulates omnimodal modeling as masked diffusion over a shared discrete token space, enabling iterative refinement under bidirectional context. Dynin-Omni adopts a multi-stage training strategy with model-merging-based modality expansion and omnimodal alignment. We evaluate Dynin-Omni across 19 multimodal benchmarks spanning language reasoning, image generation and editing, video understanding, and speech recognition and synthesis. Dynin-Omni achieves 87.6 on GSM8K, 1733.6 on MME-P, 61.4 on VideoMME, 0.87 on GenEval, and 2.1 WER on LibriSpeech test-clean, consistently outperforming existing open-source unified models while remaining competitive with strong modality-specific expert systems. These results demonstrate the potential of masked diffusion as a unified paradigm for any-to-any modeling, providing a flexible foundation for real-time omnimodal systems, unified cross-modal retrieval and generation, and embodied multimodal agents.
>
---
#### [new 043] Phonological Fossils: Machine Learning Detection of Non-Mainstream Vocabulary in Sulawesi Basic Lexicon
- **分类: cs.CL**

- **简介: 该论文属于语言学中的词汇分析任务，旨在检测苏拉威西语系中非主流词汇。通过机器学习结合音韵特征，识别出可能的底层语言影响，但未发现统一的底层语言证据。**

- **链接: [https://arxiv.org/pdf/2604.00023](https://arxiv.org/pdf/2604.00023)**

> **作者:** Mukhlis Amien; Go Frendi Gunawan
>
> **备注:** 31 pages, 4 figures, 5 tables. Submitted to Oceanic Linguistics
>
> **摘要:** Basic vocabulary in many Sulawesi Austronesian languages includes forms resisting reconstruction to any proto-form with phonological patterns inconsistent with inherited roots, but whether this non-conforming vocabulary represents pre-Austronesian substrate or independent innovation has not been tested computationally. We combine rule-based cognate subtraction with a machine learning classifier trained on phonological features. Using 1,357 forms from six Sulawesi languages in the Austronesian Basic Vocabulary Database, we identify 438 candidate substrate forms (26.5%) through cognate subtraction and Proto-Austronesian cross-checking. An XGBoost classifier trained on 26 phonological features distinguishes inherited from non-mainstream forms with AUC=0.763, revealing a phonological fingerprint: longer forms, more consonant clusters, higher glottal stop rates, and fewer Austronesian prefixes. Cross-method consensus (Cohen's kappa=0.61) identifies 266 high-confidence non-mainstream candidates. However, clustering yields no coherent word families (silhouette=0.114; cross-linguistic cognate test p=0.569), providing no evidence for a single pre-Austronesian language layer. Application to 16 additional languages confirms geographic patterning: Sulawesi languages show higher predicted non-mainstream rates (mean P_sub=0.606) than Western Indonesian languages (0.393). This study demonstrates that phonological machine learning can complement traditional comparative methods in detecting non-mainstream lexical layers, while cautioning against interpreting phonological non-conformity as evidence for a shared substrate language.
>
---
#### [new 044] CARE: Privacy-Compliant Agentic Reasoning with Evidence Discordance
- **分类: cs.CL**

- **简介: 该论文属于医疗决策任务，解决临床证据矛盾问题。构建MIMIC-DOS数据集，提出CARE框架，在保护隐私前提下提升模型处理矛盾证据的能力。**

- **链接: [https://arxiv.org/pdf/2604.01113](https://arxiv.org/pdf/2604.01113)**

> **作者:** Haochen Liu; Weien Li; Rui Song; Zeyu Li; Chun Jason Xue; Xiao-Yang Liu; Sam Nallaperuma; Xue Liu; Ye Yuan
>
> **备注:** Preprint
>
> **摘要:** Large language model (LLM) systems are increasingly used to support high-stakes decision-making, but they typically perform worse when the available evidence is internally inconsistent. Such a scenario exists in real-world healthcare settings, with patient-reported symptoms contradicting medical signs. To study this problem, we introduce MIMIC-DOS, a dataset for short-horizon organ dysfunction worsening prediction in the intensive care unit (ICU) setting. We derive this dataset from the widely recognized MIMIC-IV, a publicly available electronic health record dataset, and construct it exclusively from cases in which discordance between signs and symptoms exists. This setting poses a substantial challenge for existing LLM-based approaches, with single-pass LLMs and agentic pipelines often struggling to reconcile such conflicting signals. To address this problem, we propose CARE: a multi-stage privacy-compliant agentic reasoning framework in which a remote LLM provides guidance by generating structured categories and transitions without accessing sensitive patient data, while a local LLM uses these categories and transitions to support evidence acquisition and final decision-making. Empirically, CARE achieves stronger performance across all key metrics compared to multiple baseline settings, showing that CARE can more robustly handle conflicting clinical evidence while preserving privacy.
>
---
#### [new 045] Multi-lingual Multi-institutional Electronic Health Record based Predictive Model
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于多语言多机构电子健康记录预测任务，旨在解决跨机构和跨国数据异质性问题。通过文本对齐和翻译策略，实现无需手动标准化的联合训练，提升预测性能。**

- **链接: [https://arxiv.org/pdf/2604.00027](https://arxiv.org/pdf/2604.00027)**

> **作者:** Kyunghoon Hur; Heeyoung Kwak; Jinsu Jang; Nakhwan Kim; Edward Choi
>
> **备注:** On revision stage, 10 main pages, 3 supplementary pages
>
> **摘要:** Large-scale EHR prediction across institutions is hindered by substantial heterogeneity in schemas and code systems. Although Common Data Models (CDMs) can standardize records for multi-institutional learning, the manual harmonization and vocabulary mapping are costly and difficult to scale. Text-based harmonization provides an alternative by converting raw EHR into a unified textual form, enabling pooled learning without explicit standardization. However, applying this paradigm to multi-national datasets introduces an additional layer of heterogeneity, which is "language" that must be addressed for truly scalable EHRs learning. In this work, we investigate multilingual multi-institutional learning for EHR prediction, aiming to enable pooled training across multinational ICU datasets without manual standardization. We compare two practical strategies for handling language barriers: (i) directly modeling multilingual records with multilingual encoders, and (ii) translating non-English records into English via LLM-based word-level translation. Across seven public ICU datasets, ten clinical tasks with multiple prediction windows, translation-based lingual alignment yields more reliable cross-dataset performance than multilingual encoders. The multi-institutional learning model consistently outperforms strong baselines that require manual feature selection and harmonization, and also surpasses single-dataset training. We further demonstrate that text-based framework with lingual alignment effectively performs transfer learning via few-shot fine-tuning, with additional gains. To our knowledge, this is the first study to aggregate multilingual multinational ICU EHR datasets into one predictive model, providing a scalable path toward language-agnostic clinical prediction and future global multi-institutional EHR research.
>
---
#### [new 046] A Japanese Benchmark for Evaluating Social Bias in Reasoning Based on Attribution Theory
- **分类: cs.CL**

- **简介: 该论文属于社会偏见评估任务，旨在解决日本文化背景下语言模型的推理偏见问题。通过构建新数据集JUBAKU-v2，评估行为归因中的群体偏见。**

- **链接: [https://arxiv.org/pdf/2604.00568](https://arxiv.org/pdf/2604.00568)**

> **作者:** Taihei Shiotani; Masahiro Kaneko; Naoaki Okazaki
>
> **摘要:** In enhancing the fairness of Large Language Models (LLMs), evaluating social biases rooted in the cultural contexts of specific linguistic regions is essential. However, most existing Japanese benchmarks heavily rely on translating English data, which does not necessarily provide an evaluation suitable for Japanese culture. Furthermore, they only evaluate bias in the conclusion, failing to capture biases lurking in the reasoning. In this study, based on attribution theory in social psychology, we constructed a new dataset, ``JUBAKU-v2,'' which evaluates the bias in attributing behaviors to in-groups and out-groups within reasoning while fixing the conclusion. This dataset consists of 216 examples reflecting cultural biases specific to Japan. Experimental results verified that it can detect performance differences across models more sensitively than existing benchmarks.
>
---
#### [new 047] Oblivion: Self-Adaptive Agentic Memory Control through Decay-Driven Activation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Oblivion框架，解决LLM代理记忆控制问题。通过衰减驱动的遗忘机制，实现动态记忆访问与强化，提升长期任务中的推理效率。**

- **链接: [https://arxiv.org/pdf/2604.00131](https://arxiv.org/pdf/2604.00131)**

> **作者:** Ashish Rana; Chia-Chien Hung; Qumeng Sun; Julian Martin Kunkel; Carolin Lawrence
>
> **备注:** 7 pages, 2 figures, and 4 tables
>
> **摘要:** Human memory adapts through selective forgetting: experiences become less accessible over time but can be reactivated by reinforcement or contextual cues. In contrast, memory-augmented LLM agents rely on "always-on" retrieval and "flat" memory storage, causing high interference and latency as histories grow. We introduce Oblivion, a memory control framework that casts forgetting as decay-driven reductions in accessibility, not explicit deletion. Oblivion decouples memory control into read and write paths. The read path decides when to consult memory, based on agent uncertainty and memory buffer sufficiency, avoiding redundant always-on access. The write path decides what to strengthen, by reinforcing memories contributing to forming the response. Together, this enables hierarchical memory organization that maintains persistent high-level strategies while dynamically loading details as needed. We evaluate on both static and dynamic long-horizon interaction benchmarks. Results show that Oblivion dynamically adapts memory access and reinforcement, balancing learning and forgetting under shifting contexts, highlighting that memory control is essential for effective LLM-agentic reasoning. The source code is available at this https URL.
>
---
#### [new 048] TR-ICRL: Test-Time Rethinking for In-Context Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文提出TR-ICRL框架，解决ICRL中奖励估计问题。通过检索、生成候选答案、多数投票生成伪标签，提升模型性能。适用于推理和知识密集型任务。**

- **链接: [https://arxiv.org/pdf/2604.00438](https://arxiv.org/pdf/2604.00438)**

> **作者:** Wenxuan Jiang; Yuxin Zuo; Zijian Zhang; Xuecheng Wu; Zining Fan; Wenxuan Liu; Li Chen; Xiaoyu Li; Xuezhi Cao; Xiaolong Jin; Ninghao Liu
>
> **备注:** 14 pages, 7 figures
>
> **摘要:** In-Context Reinforcement Learning (ICRL) enables Large Language Models (LLMs) to learn online from external rewards directly within the context window. However, a central challenge in ICRL is reward estimation, as models typically lack access to ground-truths during inference. To address this limitation, we propose Test-Time Rethinking for In-Context Reinforcement Learning (TR-ICRL), a novel ICRL framework designed for both reasoning and knowledge-intensive tasks. TR-ICRL operates by first retrieving the most relevant instances from an unlabeled evaluation set for a given query. During each ICRL iteration, LLM generates a set of candidate answers for every retrieved instance. Next, a pseudo-label is derived from this set through majority voting. This label then serves as a proxy to give reward messages and generate formative feedbacks, guiding LLM through iterative refinement. In the end, this synthesized contextual information is integrated with the original query to form a comprehensive prompt, with the answer determining through a final round of majority voting. TR-ICRL is evaluated on mainstream reasoning and knowledge-intensive tasks, where it demonstrates significant performance gains. Remarkably, TR-ICRL improves Qwen2.5-7B by 21.23% on average on MedQA and even 137.59% on AIME2024. Extensive ablation studies and analyses further validate the effectiveness and robustness of our approach. Our code is available at this https URL.
>
---
#### [new 049] How Trustworthy Are LLM-as-Judge Ratings for Interpretive Responses? Implications for Qualitative Research Workflows
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，探讨LLM-as-judge评分在解释性回复中的可信度，旨在评估其与人类判断的一致性及模型选择的适用性。**

- **链接: [https://arxiv.org/pdf/2604.00008](https://arxiv.org/pdf/2604.00008)**

> **作者:** Songhee Han; Jueun Shin; Jiyoon Han; Bung-Woo Jun; Hilal Ayan Karabatman
>
> **摘要:** As qualitative researchers show growing interest in using automated tools to support interpretive analysis, a large language model (LLM) is often introduced into an analytic workflow as is, without systematic evaluation of interpretive quality or comparison across models. This practice leaves model selection largely unexamined despite its potential influence on interpretive outcomes. To address this gap, this study examines whether LLM-as-judge evaluations meaningfully align with human judgments of interpretive quality and can inform model-level decision making. Using 712 conversational excerpts from semi-structured interviews with K-12 mathematics teachers, we generated one-sentence interpretive responses using five widely adopted inference models: Command R+ (Cohere), Gemini 2.5 Pro (Google), GPT-5.1 (OpenAI), Llama 4 Scout-17B Instruct (Meta), and Qwen 3-32B Dense (Alibaba). Automated evaluations were conducted using AWS Bedrock's LLM-as-judge framework across five metrics, and a stratified subset of responses was independently rated by trained human evaluators on interpretive accuracy, nuance preservation, and interpretive coherence. Results show that LLM-as-judge scores capture broad directional trends in human evaluations at the model level but diverge substantially in score magnitude. Among automated metrics, Coherence showed the strongest alignment with aggregated human ratings, whereas Faithfulness and Correctness revealed systematic misalignment at the excerpt level, particularly for non-literal and nuanced interpretations. Safety-related metrics were largely irrelevant to interpretive quality. These findings suggest that LLM-as-judge methods are better suited for screening or eliminating underperforming models than for replacing human judgment, offering practical guidance for systematic comparison and selection of LLMs in qualitative research workflows.
>
---
#### [new 050] S0 Tuning: Zero-Overhead Adaptation of Hybrid Recurrent-Attention Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出S0 tuning方法，用于优化混合循环-注意力模型的初始状态矩阵，在零推理开销下提升模型性能，解决参数高效微调问题。**

- **链接: [https://arxiv.org/pdf/2604.01168](https://arxiv.org/pdf/2604.01168)**

> **作者:** Jack Young
>
> **备注:** 15 pages (10 main + 5 appendix), 3 figures, code at this https URL
>
> **摘要:** Using roughly 48 execution-verified HumanEval training solutions, tuning a single initial state matrix per recurrent layer, with zero inference overhead, outperforms LoRA by +10.8 pp (p < 0.001) on HumanEval. The method, which we call S0 tuning, optimizes one state matrix per recurrent layer while freezing all model weights. On Qwen3.5-4B (GatedDeltaNet hybrid), S0 tuning improves greedy pass@1 by +23.6 +/- 1.7 pp (10 seeds). On FalconH1-7B (Mamba-2 hybrid), S0 reaches 71.8% +/- 1.3 and LoRA reaches 71.4% +/- 2.4 (3 seeds), statistically indistinguishable at this sample size while requiring no weight merging. Cross-domain transfer is significant on MATH-500 (+4.8 pp, p = 0.00002, 8 seeds) and GSM8K (+2.8 pp, p = 0.0003, 10 seeds); a text-to-SQL benchmark (Spider) shows no transfer, consistent with the trajectory-steering mechanism. A prefix-tuning control on a pure Transformer (Qwen2.5-3B) degrades performance by -13.9 pp under all nine configurations tested. On Qwen3.5, a per-step state-offset variant reaches +27.1 pp, above both S0 and LoRA but with per-step inference cost. Taken together, the results show that recurrent state initialization is a strong zero-inference-overhead PEFT surface for hybrid language models when verified supervision is scarce. The tuned state is a ~48 MB file; task switching requires no weight merging or model reload. Code and library: this https URL.
>
---
#### [new 051] Detecting Abnormal User Feedback Patterns through Temporal Sentiment Aggregation
- **分类: cs.CL**

- **简介: 该论文属于情感分析任务，旨在检测用户反馈中的异常模式。通过时间聚合情感信号，识别用户满意度的异常下降。**

- **链接: [https://arxiv.org/pdf/2604.00020](https://arxiv.org/pdf/2604.00020)**

> **作者:** Yalun Qi; Sichen Zhao; Zhiming Xue; Xianling Zeng; Zihan Yu
>
> **摘要:** In many real-world applications, such as customer feedback monitoring, brand reputation management, and product health tracking, understanding the temporal dynamics of user sentiment is crucial for early detection of anomalous events such as malicious review campaigns or sudden declines in user satisfaction. Traditional sentiment analysis methods focus on individual text classification, which is insufficient to capture collective behavioral shifts over time due to inherent noise and class imbalance in short user comments. In this work, we propose a temporal sentiment aggregation framework that leverages pretrained transformer-based language models to extract per-comment sentiment signals and aggregates them into time-window-level scores. Significant downward shifts in these aggregated scores are interpreted as potential anomalies in user feedback patterns. We adopt RoBERTa as our core semantic feature extractor and demonstrate, through empirical evaluation on real social media data, that the aggregated sentiment scores reveal meaningful trends and support effective anomaly detection. Experiments on real-world social media data demonstrate that our method successfully identifies statistically significant sentiment drops that correspond to coherent complaint patterns, providing an effective and interpretable solution for feedback anomaly monitoring.
>
---
#### [new 052] From Early Encoding to Late Suppression: Interpreting LLMs on Character Counting Tasks
- **分类: cs.CL**

- **简介: 该论文研究LLMs在字符计数任务中的失败原因，揭示其内部信息被后期层抑制的现象，提出符号推理错误源于模型内部结构干扰。**

- **链接: [https://arxiv.org/pdf/2604.00778](https://arxiv.org/pdf/2604.00778)**

> **作者:** Ayan Datta; Mounika Marreddy; Alexander Mehler; Zhixue Zhao; Radhika Mamidi
>
> **摘要:** Large language models (LLMs) exhibit failures on elementary symbolic tasks such as character counting in a word, despite excelling on complex benchmarks. Although this limitation has been noted, the internal reasons remain unclear. We use character counting (e.g., "How many p's are in apple?") as a minimal, controlled probe that isolates token-level reasoning from higher-level confounds. Using this setting, we uncover a consistent phenomenon across modern architectures, including LLaMA, Qwen, and Gemma: models often compute the correct answer internally yet fail to express it at the output layer. Through mechanistic analysis combining probing classifiers, activation patching, logit lens analysis, and attention head tracing, we show that character-level information is encoded in early and mid-layer representations. However, this information is attenuated by a small set of components in later layers, especially the penultimate and final layer MLP. We identify these components as negative circuits: subnetworks that downweight correct signals in favor of higher-probability but incorrect outputs. Our results lead to two contributions. First, we show that symbolic reasoning failures in LLMs are not due to missing representations or insufficient scale, but arise from structured interference within the model's computation graph. This explains why such errors persist and can worsen under scaling and instruction tuning. Second, we provide evidence that LLM forward passes implement a form of competitive decoding, in which correct and incorrect hypotheses coexist and are dynamically reweighted, with final outputs determined by suppression as much as by amplification. These findings carry implications for interpretability and robustness: simple symbolic reasoning exposes weaknesses in modern LLMs, underscoring need for design strategies that ensure information is encoded and reliably used.
>
---
#### [new 053] Adapting Text LLMs to Speech via Multimodal Depth Up-Scaling
- **分类: cs.CL**

- **简介: 该论文属于语音语言模型适配任务，旨在解决将文本LLM转为语音LLM时文本能力下降的问题。通过在冻结的文本LLM中插入新层并仅训练这些层，实现语音识别性能提升且文本退化较少。**

- **链接: [https://arxiv.org/pdf/2604.00489](https://arxiv.org/pdf/2604.00489)**

> **作者:** Kazuki Yano; Jun Suzuki; Shinji Watanabe
>
> **摘要:** Adapting pre-trained text Large Language Models (LLMs) into Speech Language Models (Speech LMs) via continual pretraining on speech data is promising, but often degrades the original text capabilities. We propose Multimodal Depth Upscaling, an extension of an emerging strategy in continual LLM pre-training, where new transformer layers are inserted into a frozen text LLM and only the added layers are trained on speech data. Experiments with SmolLM2-360M and SmolLM2-1.7B on 48k hours of English Automatic Speech Recognition (ASR) data show that depth up-scaling achieves ASR comparable to full fine-tuning while causing far less text degradation than both full fine-tuning and Low-Rank Adaptation (LoRA). We further show that incorporating E-Branchformer, an architecture designed for speech recognition, as the inserted layers achieves ASR that matches or surpasses full fine-tuning on the larger model while reducing text degradation by over 75% with 60% fewer trainable parameters.
>
---
#### [new 054] Multimodal Analysis of State-Funded News Coverage of the Israel-Hamas War on YouTube Shorts
- **分类: cs.CL; cs.AI; cs.SI**

- **简介: 该论文属于多模态分析任务，旨在研究YouTube Shorts中国家资助媒体对以哈战争的报道。通过结合语音识别、情感分析和场景分类，分析新闻内容的情感与视觉特征，揭示不同媒体的报道差异。**

- **链接: [https://arxiv.org/pdf/2604.00994](https://arxiv.org/pdf/2604.00994)**

> **作者:** Daniel Miehling; Sandra Kuebler
>
> **摘要:** YouTube Shorts have become central to news consumption on the platform, yet research on how geopolitical events are represented in this format remains limited. To address this gap, we present a multimodal pipeline that combines automatic transcription, aspect-based sentiment analysis (ABSA), and semantic scene classification. The pipeline is first assessed for feasibility and then applied to analyze short-form coverage of the Israel-Hamas war by state-funded outlets. Using over 2,300 conflict-related Shorts and more than 94,000 visual frames, we systematically examine war reporting across major international broadcasters. Our findings reveal that the sentiment expressed in transcripts regarding specific aspects differs across outlets and over time, whereas scene-type classifications reflect visual cues consistent with real-world events. Notably, smaller domain-adapted models outperform large transformers and even LLMs for sentiment analysis, underscoring the value of resource-efficient approaches for humanities research. The pipeline serves as a template for other short-form platforms, such as TikTok and Instagram, and demonstrates how multimodal methods, combined with qualitative interpretation, can characterize sentiment patterns and visual cues in algorithmically driven video environments.
>
---
#### [new 055] Hierarchical Chain-of-Thought Prompting: Enhancing LLM Reasoning Performance and Efficiency
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在提升大语言模型的推理性能与效率。针对传统链式思维提示的冗余和低效问题，提出分层链式思维（Hi-CoT）方法，通过结构化分解推理步骤，提高准确率并缩短推理路径。**

- **链接: [https://arxiv.org/pdf/2604.00130](https://arxiv.org/pdf/2604.00130)**

> **作者:** Xingshuai Huang; Derek Li; Bahareh Nikpour; Parsa Omidi
>
> **摘要:** Chain-of-Thought (CoT) prompting has significantly improved the reasoning capabilities of large language models (LLMs). However, conventional CoT often relies on unstructured, flat reasoning chains that suffer from redundancy and suboptimal performance. In this work, we introduce Hierarchical Chain-of-Thought (Hi-CoT) prompting, a structured reasoning paradigm specifically designed to address the challenges of complex, multi-step reasoning. Hi-CoT decomposes the reasoning process into hierarchical substeps by alternating between instructional planning and step-by-step execution. This decomposition enables LLMs to better manage long reasoning horizons and maintain logical coherence. Extensive evaluations across diverse LLMs and mathematical reasoning benchmarks show that Hi-CoT consistently improves average accuracy by 6.2% (up to 61.4% on certain models and tasks) while reducing reasoning trace length by 13.9% compared to CoT prompting. We further show that accuracy and efficiency are maximized when models strictly adhere to the hierarchical structure. Our code is available at this https URL.
>
---
#### [new 056] A Reliability Evaluation of Hybrid Deterministic-LLM Based Approaches for Academic Course Registration PDF Information Extraction
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于信息抽取任务，旨在提高学术课程注册文档的可靠性。通过比较不同方法，验证混合策略在效率和准确率上的优势。**

- **链接: [https://arxiv.org/pdf/2604.00003](https://arxiv.org/pdf/2604.00003)**

> **作者:** Muhammad Anis Al Hilmi; Neelansh Khare; Noel Framil Iglesias
>
> **备注:** 9 pages, 6 figures, 3 tables
>
> **摘要:** This study evaluates the reliability of information extraction approaches from KRS documents using three strategies: LLM only, Hybrid Deterministic - LLM (regex + LLM), and a Camelot based pipeline with LLM fallback. Experiments were conducted on 140 documents for the LLM based test and 860 documents for the Camelot based pipeline evaluation, covering four study programs with varying data in tables and metadata. Three 12 - 14B LLM models (Gemma 3, Phi 4, and Qwen 2.5) were run locally using Ollama and a consumer grade CPU without a GPU. Evaluations used exact match (EM) and Levenshtein similarity (LS) metrics with a threshold of 0.7. Although not applicable to all models, the results show that the hybrid approach can improve efficiency compared to LLM only, especially for deterministic metadata. The Camelot based pipeline with LLM fallback produced the best combination of accuracy (EM and LS up to 0.99 - 1.00) and computational efficiency (less than 1 second per PDF in most cases). The Qwen 2.5:14b model demonstrated the most consistent performance across all scenarios. These findings confirm that integrating deterministic and LLM methods is increasingly reliable and efficient for information extraction from text based academic documents in computationally constrained environments.
>
---
#### [new 057] Polysemanticity or Polysemy? Lexical Identity Confounds Superposition Metrics
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究神经网络中词义混淆对超位置度量的影响，旨在区分词形共享与语义压缩。通过实验发现词形因素导致的重叠远大于语义因素，影响模型性能。**

- **链接: [https://arxiv.org/pdf/2604.00443](https://arxiv.org/pdf/2604.00443)**

> **作者:** Iyad Ait Hou; Rebecca Hwa
>
> **备注:** 21 pages
>
> **摘要:** If the same neuron activates for both "lender" and "riverside," standard metrics attribute the overlap to superposition--the neuron must be compressing two unrelated concepts. This work explores how much of the overlap is due a lexical confound: neurons fire for a shared word form (such as "bank") rather than for two compressed concepts. A 2x2 factorial decomposition reveals that the lexical-only condition (same word, different meaning) consistently exceeds the semantic-only condition (different word, same meaning) across models spanning 110M-70B parameters. The confound carries into sparse autoencoders (18-36% of features blend senses), sits in <=1% of activation dimensions, and hurts downstream tasks: filtering it out improves word sense disambiguation and makes knowledge edits more selective (p = 0.002).
>
---
#### [new 058] GPT-NL Public Corpus: A Permissively Licensed, Dutch-First Dataset for LLM Pre-training
- **分类: cs.CL**

- **简介: 该论文介绍GPT-NL公共语料库，用于大语言模型预训练。解决荷兰语资源不足问题，整合并优化荷兰语及其他语言数据，确保合规性与可用性。**

- **链接: [https://arxiv.org/pdf/2604.00920](https://arxiv.org/pdf/2604.00920)**

> **作者:** Jesse van Oort; Frank Brinkkemper; Erik de Graaf; Bram Vanroy; Saskia Lensink
>
> **备注:** Accepted at LREC 2026
>
> **摘要:** We present the GPT-NL Public Corpus, the biggest permissively licensed corpus of Dutch language resources. The GPT-NL Public Corpus contains 21 Dutch-only collections totalling 36B preprocessed Dutch tokens not present in any other LLM pretraining corpus. Additionally, the corpus includes roughly 207B English, 232B Code, and 48B German/Danish tokens taken from existing sets which we further curated for compliance. This corpus includes curated data from large existing corpora like Common Corpus and Common Crawl, as well as newly created Dutch-specific collections. Most newly created Dutch collections consist of content collected in collaboration with organisations or synthetically augmented content. All data is collected and evaluated with the aim of facilitating the creation of (commercial) language models that are lawful, useful and non-harmful. All data included in the GPT-NL Public Corpus is sourced from datasets with permissive licensing and is curated and redistributed under a CC-BY license. The full dataset is publicly available on the Hugging Face Hub.
>
---
#### [new 059] A Taxonomy of Programming Languages for Code Generation
- **分类: cs.CL**

- **简介: 该论文属于编程语言资源分类任务，旨在解决代码生成中语言资源不均的问题。通过建立四层分类体系，分析646种语言的资源分布情况。**

- **链接: [https://arxiv.org/pdf/2604.00239](https://arxiv.org/pdf/2604.00239)**

> **作者:** Nishat Raihan; Christian Newman; Marcos Zampieri
>
> **摘要:** The world's 7,000+ languages vary widely in the availability of resources for NLP, motivating efforts to systematically categorize them by their degree of resourcefulness (Joshi et al., 2020). A similar disparity exists among programming languages (PLs); however, no resource-tier taxonomy has been established for code. As large language models (LLMs) grow increasingly capable of generating code, such a taxonomy becomes essential. To fill this gap, we present the first reproducible PL resource classification, grouping 646 languages into four tiers. We show that only 1.9% of languages (Tier 3, High) account for 74.6% of all tokens in seven major corpora, while 71.7% of languages (Tier 0, Scarce) contribute just 1.0%. Statistical analyses of within-tier inequality, dispersion, and distributional skew confirm that this imbalance is both extreme and systematic. Our results provide a principled framework for dataset curation and tier-aware evaluation of multilingual LLMs.
>
---
#### [new 060] TRIMS: Trajectory-Ranked Instruction Masked Supervision for Diffusion Language Models
- **分类: cs.CL**

- **简介: 该论文提出TRIMS方法，解决扩散语言模型的解码轨迹优化问题，通过轨迹引导的监督提升生成效率与准确性。**

- **链接: [https://arxiv.org/pdf/2604.00666](https://arxiv.org/pdf/2604.00666)**

> **作者:** Lingjie Chen; Ruizhong Qiu; Yuyu Fan; Yanjun Zhao; Hanghang Tong
>
> **备注:** 10 pages, 7 figures, 1 algorithm
>
> **摘要:** Diffusion language models (DLMs) offer a promising path toward low-latency generation through parallel decoding, but their practical efficiency depends heavily on the decoding trajectory. In practice, this advantage often fails to fully materialize because standard training does not provide explicit supervision over token reveal order, creating a train-inference mismatch that leads to suboptimal decoding behavior. We propose Trajectory-Ranked Instruction Masked Supervision (TRIMS), a simple trajectory-guided supervised fine-tuning framework that injects trajectory supervision into standard Masked Diffusion Language Model (MDLM) training with minimal overhead. Instead of relying on costly DLM-based distillation, TRIMS uses lightweight signals from an autoregressive teacher to guide a trajectory-aware masking strategy, encouraging the model to learn more effective decoding orders. Experiments on LLaDA and Dream across math and coding benchmarks show that TRIMS significantly improves the accuracy-parallelism trade-off over both standard MDLM training and train-free acceleration baselines, while achieving competitive performance with prior distillation-based approaches at substantially lower training cost. Further analysis shows that TRIMS leads to better decoding trajectories, validating the effectiveness of trajectory-guided supervision for DLMs.
>
---
#### [new 061] REM-CTX: Automated Peer Review via Reinforcement Learning with Auxiliary Context
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出REM-CTX，用于自动化同行评审任务，解决传统系统忽略视觉元素和外部信号的问题。通过强化学习结合辅助上下文，提升评审质量与一致性。**

- **链接: [https://arxiv.org/pdf/2604.00248](https://arxiv.org/pdf/2604.00248)**

> **作者:** Pawin Taechoyotin; Daniel E. Acuna
>
> **备注:** 12 pages, 6 figures
>
> **摘要:** Most automated peer review systems rely on textual manuscript content alone, leaving visual elements such as figures and external scholarly signals underutilized. We introduce REM-CTX, a reinforcement-learning system that incorporates auxiliary context into the review generation process via correspondence-aware reward functions. REM-CTX trains an 8B-parameter language model with Group Relative Policy Optimization (GRPO) and combines a multi-aspect quality reward with two correspondence rewards that explicitly encourage alignment with auxiliary context. Experiments on manuscripts across Computer, Biological, and Physical Sciences show that REM-CTX achieves the highest overall review quality among six baselines, outperforming other systems with substantially larger commercial models, and surpassing the next-best RL baseline across both quality and contextual grounding metrics. Ablation studies confirm that the two correspondence rewards are complementary: each selectively improves its targeted correspondence reward while preserving all quality dimensions, and the full model outperforms all partial variants. Analysis of training dynamics reveals that the criticism aspect is negatively correlated with other metrics during training, suggesting that future studies should group multi-dimension rewards for review generation.
>
---
#### [new 062] Polish phonology and morphology through the lens of distributional semantics
- **分类: cs.CL**

- **简介: 该论文研究波兰语的音系和构词结构与意义的关系，属于分布语义任务。旨在探讨词形特征是否能在语义空间中体现，通过计算方法验证语义向量可捕捉语音单元信息。**

- **链接: [https://arxiv.org/pdf/2604.00174](https://arxiv.org/pdf/2604.00174)**

> **作者:** Paula Orzechowska; R. Harald Baayen
>
> **摘要:** This study investigates the relationship between the phonological and morphological structure of Polish words and their meanings using Distributional Semantics. In the present analysis, we ask whether there is a relationship between the form properties of words containing consonant clusters and their meanings. Is the phonological and morphonological structure of complex words mirrored in semantic space? We address these questions for Polish, a language characterized by non-trivial morphology and an impressive inventory of morphologically-motivated consonant clusters. We use statistical and computational techniques, such as t-SNE, Linear Discriminant Analysis and Linear Discriminative Learning, and demonstrate that -- apart from encoding rich morphosyntactic information (e.g. tense, number, case) -- semantic vectors capture information on sub-lexical linguistic units such as phoneme strings. First, phonotactic complexity, morphotactic transparency, and a wide range of morphosyntactic categories available in Polish (case, gender, aspect, tense, number) can be predicted from embeddings without requiring any information about the forms of words. Second, we argue that computational modelling with the discriminative lexicon model using embeddings can provide highly accurate predictions for comprehension and production, exactly because of the existence of extensive information in semantic space that is to a considerable extent isomorphic with structure in the form space.
>
---
#### [new 063] Scalable Identification and Prioritization of Requisition-Specific Personal Competencies Using Large Language Models
- **分类: cs.CL; cs.CY; cs.IR; cs.LG**

- **简介: 该论文属于招聘任务，解决AI难以识别岗位特定个人能力的问题。通过大语言模型方法，识别并优先排序岗位所需的关键能力。**

- **链接: [https://arxiv.org/pdf/2604.00006](https://arxiv.org/pdf/2604.00006)**

> **作者:** Wanxin Li; Denver McNeney; Nivedita Prabhu; Charlene Zhang; Renee Barr; Matthew Kitching; Khanh Dao Duc; Anthony S. Boyce
>
> **摘要:** AI-powered recruitment tools are increasingly adopted in personnel selection, yet they struggle to capture the requisition (req)-specific personal competencies (PCs) that distinguish successful candidates beyond job categories. We propose a large language model (LLM)-based approach to identify and prioritize req-specific PCs from reqs. Our approach integrates dynamic few-shot prompting, reflection-based self-improvement, similarity-based filtering, and multi-stage validation. Applied to a dataset of Program Manager reqs, our approach correctly identifies the highest-priority req-specific PCs with an average accuracy of 0.76, approaching human expert inter-rater reliability, and maintains a low out-of-scope rate of 0.07.
>
---
#### [new 064] KUET at StanceNakba Shared Task: StanceMoE: Mixture-of-Experts Architecture for Stance Detection
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于立场检测任务，旨在解决隐含目标实体的立场识别问题。提出StanceMoE模型，通过多专家架构捕捉多样化的语言信号，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2604.00878](https://arxiv.org/pdf/2604.00878)**

> **作者:** Abdullah Al Shafi; Md. Milon Islam; Sk. Imran Hossain; K. M. Azharul Hasan
>
> **备注:** Accepted for workshop proceedings of the 15th International Conference on Language Resources and Evaluation (LREC'26)
>
> **摘要:** Actor-level stance detection aims to determine an author expressed position toward specific geopolitical actors mentioned or implicated in a text. Although transformer-based models have achieved relatively good performance in stance classification, they typically rely on unified representations that may not sufficiently capture heterogeneous linguistic signals, such as contrastive discourse structures, framing cues, and salient lexical indicators. This motivates the need for adaptive architectures that explicitly model diverse stance-expressive patterns. In this paper, we propose StanceMoE, a context-enhanced Mixture-of-Experts (MoE) architecture built upon a fine-tuned BERT encoder for actor-level stance detection. Our model integrates six expert modules designed to capture complementary linguistic signals, including global semantic orientation, salient lexical cues, clause-level focus, phrase-level patterns, framing indicators, and contrast-driven discourse shifts. A context-aware gating mechanism dynamically weights expert contributions, enabling adaptive routing based on input characteristics. Experiments are conducted on the StanceNakba 2026 Subtask A dataset, comprising 1,401 annotated English texts where the target actor is implicit in the text. StanceMoE achieves a macro-F1 score of 94.26%, outperforming traditional baselines, and alternative BERT-based variants.
>
---
#### [new 065] English to Central Kurdish Speech Translation: Corpus Creation, Evaluation, and Orthographic Standardization
- **分类: cs.CL**

- **简介: 该论文属于英语到中库尔德语语音翻译任务，旨在解决语音翻译中的拼写差异问题。通过构建数据集并提出标准化方法，提升翻译性能。**

- **链接: [https://arxiv.org/pdf/2604.00613](https://arxiv.org/pdf/2604.00613)**

> **作者:** Mohammad Mohammadamini; Daban Q. Jaff; Josep Crego; Marie Tahon; Antoine Laurent
>
> **摘要:** We present KUTED, a speech-to-text translation (S2TT) dataset for Central Kurdish, derived from TED and TEDx talks. The corpus comprises 91,000 sentence pairs, including 170 hours of English audio, 1.65 million English tokens, and 1.40 million Central Kurdish tokens. We evaluate KUTED on the S2TT task and find that orthographic variation significantly degrades Kurdish translation performance, producing nonstandard outputs. To address this, we propose a systematic text standardization approach that yields substantial performance gains and more consistent translations. On a test set separated from TED talks, a fine-tuned Seamless model achieves 15.18 BLEU, and we improve Seamless baseline by 3.0 BLEU on the FLEURS benchmark. We also train a Transformer model from scratch and evaluate a cascaded system that combines Seamless (ASR) with NLLB (MT).
>
---
#### [new 066] Paper Reconstruction Evaluation: Evaluating Presentation and Hallucination in AI-written Papers
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于AI生成论文的评估任务，旨在解决AI写作质量与风险评估问题。提出PaperRecon框架，通过对比生成与原论文评估其表现和幻觉情况。**

- **链接: [https://arxiv.org/pdf/2604.01128](https://arxiv.org/pdf/2604.01128)**

> **作者:** Atsuyuki Miyai; Mashiro Toyooka; Zaiying Zhao; Kenta Watanabe; Toshihiko Yamasaki; Kiyoharu Aizawa
>
> **备注:** Project Page: this https URL
>
> **摘要:** This paper introduces the first systematic evaluation framework for quantifying the quality and risks of papers written by modern coding agents. While AI-driven paper writing has become a growing concern, rigorous evaluation of the quality and potential risks of AI-written papers remains limited, and a unified understanding of their reliability is still lacking. We introduce Paper Reconstruction Evaluation (PaperRecon), an evaluation framework in which an overview (this http URL) is created from an existing paper, after which an agent generates a full paper based on the overview and minimal additional resources, and the result is subsequently compared against the original paper. PaperRecon disentangles the evaluation of the AI-written papers into two orthogonal dimensions, Presentation and Hallucination, where Presentation is evaluated using a rubric and Hallucination is assessed via agentic evaluation grounded in the original paper source. For evaluation, we introduce PaperWrite-Bench, a benchmark of 51 papers from top-tier venues across diverse domains published after 2025. Our experiments reveal a clear trade-off: while both ClaudeCode and Codex improve with model advances, ClaudeCode achieves higher presentation quality at the cost of more than 10 hallucinations per paper on average, whereas Codex produces fewer hallucinations but lower presentation quality. This work takes a first step toward establishing evaluation frameworks for AI-driven paper writing and improving the understanding of its risks within the research community.
>
---
#### [new 067] Frege in the Flesh: Biolinguistics and the Neural Enforcement of Syntactic Structures
- **分类: cs.CL**

- **简介: 论文探讨生物语言学，聚焦语言的神经基础与句法结构的数学建模。任务是解析语言作为生物器官的本质，解决语言进化与神经机制的关系问题，通过形式化语法约束神经模型。**

- **链接: [https://arxiv.org/pdf/2604.00291](https://arxiv.org/pdf/2604.00291)**

> **作者:** Elliot Murphy
>
> **摘要:** Biolinguistics is the interdisciplinary scientific study of the biological foundations, evolution, and genetic basis of human language. It treats language as an innate biological organ or faculty of the mind, rather than a cultural tool, and it challenges a behaviorist conception of human language acquisition as being based on stimulus-response associations. Extracting its most essential component, it takes seriously the idea that mathematical, algebraic models of language capture something natural about the world. The syntactic structure-building operation of MERGE is thought to offer the scientific community a "real joint of nature", "a (new) aspect of nature" (Mukherji 2010), not merely a formal artefact. This mathematical theory of language is then seen as being able to offer biologists, geneticists and neuroscientists clearer instructions for how to explore language. The argument of this chapter proceeds in four steps. First, I clarify the object of inquiry for biolinguistics: not speech, communication, or generic sequence processing, but the internal computational system that generates hierarchically structured expressions. Second, I argue that this formal characterization matters for evolutionary explanation, because different conceptions of syntax imply different standards of what must be explained. Third, I suggest that a sufficiently explicit algebraic account of syntax places non-trivial constraints on candidate neural mechanisms. Finally, I consider how recent neurocomputational work begins to transform these constraints into empirically tractable hypotheses, while also noting the speculative and revisable character of the present program.
>
---
#### [new 068] To Memorize or to Retrieve: Scaling Laws for RAG-Considerate Pretraining
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究RAG框架下预训练与检索的协同作用，解决如何在固定数据预算内优化两者分配的问题。通过实验分析模型规模、预训练数据量和检索库大小的关系，提出三维扩展框架。**

- **链接: [https://arxiv.org/pdf/2604.00715](https://arxiv.org/pdf/2604.00715)**

> **作者:** Karan Singh; Michael Yu; Varun Gangal; Zhuofu Tao; Sachin Kumar; Emmy Liu; Steven Y. Feng
>
> **备注:** Code and data at this https URL
>
> **摘要:** Retrieval-augmented generation (RAG) improves language model (LM) performance by providing relevant context at test time for knowledge-intensive situations. However, the relationship between parametric knowledge acquired during pretraining and non-parametric knowledge accessed via retrieval remains poorly understood, especially under fixed data budgets. In this work, we systematically study the trade-off between pretraining corpus size and retrieval store size across a wide range of model and data scales. We train OLMo-2-based LMs ranging from 30M to 3B parameters on up to 100B tokens of DCLM data, while varying both pretraining data scale (1-150x the number of parameters) and retrieval store size (1-20x), and evaluate performance across a diverse suite of benchmarks spanning reasoning, scientific QA, and open-domain QA. We find that retrieval consistently improves performance over parametric-only baselines across model scales and introduce a three-dimensional scaling framework that models performance as a function of model size, pretraining tokens, and retrieval corpus size. This scaling manifold enables us to estimate optimal allocations of a fixed data budget between pretraining and retrieval, revealing that the marginal utility of retrieval depends strongly on model scale, task type, and the degree of pretraining saturation. Our results provide a quantitative foundation for understanding when and how retrieval should complement pretraining, offering practical guidance for allocating data resources in the design of scalable language modeling systems.
>
---
#### [new 069] ASCAT: An Arabic Scientific Corpus and Benchmark for Advanced Translation Evaluation
- **分类: cs.CL**

- **简介: 该论文提出ASCAT，一个用于科学翻译评估的高质量英阿平行语料库，解决阿拉伯语科学翻译资源不足的问题，通过多模型翻译与人工验证构建，支持翻译质量评估与模型训练。**

- **链接: [https://arxiv.org/pdf/2604.00015](https://arxiv.org/pdf/2604.00015)**

> **作者:** Serry Sibaee; Khloud Al Jallad; Zineb Yousfi; Israa Elsayed Elhosiny; Yousra El-Ghawi; Batool Balah; Omer Nacar
>
> **摘要:** We present ASCAT (Arabic Scientific Corpus for Advanced Translation), a high-quality English-Arabic parallel benchmark corpus designed for scientific translation evaluation constructed through a systematic multi-engine translation and human validation pipeline. Unlike existing Arabic-English corpora that rely on short sentences or single-domain text, ASCAT targets full scientific abstracts averaging 141.7 words (English) and 111.78 words (Arabic), drawn from five scientific domains: physics, mathematics, computer science, quantum mechanics, and artificial intelligence. Each abstract was translated using three complementary architectures generative AI (Gemini), transformer-based models (Hugging Face \texttt{quickmt-en-ar}), and commercial MT APIs (Google Translate, DeepL) and subsequently validated by domain experts at the lexical, syntactic, and semantic levels. The resulting corpus contains 67,293 English tokens and 60,026 Arabic tokens, with an Arabic vocabulary of 17,604 unique words reflecting the morphological richness of the language. We benchmark three state-of-the-art LLMs on the corpus GPT-4o-mini (BLEU: 37.07), Gemini-3.0-Flash-Preview (BLEU: 30.44), and Qwen3-235B-A22B (BLEU: 23.68) demonstrating its discriminative power as an evaluation benchmark. ASCAT addresses a critical gap in scientific MT resources for Arabic and is designed to support rigorous evaluation of scientific translation quality and training of domain-specific translation models.
>
---
#### [new 070] Do Language Models Know When They'll Refuse? Probing Introspective Awareness of Safety Boundaries
- **分类: cs.CL**

- **简介: 该论文属于模型安全研究任务，旨在探究大语言模型是否能准确预测自身拒绝行为。通过实验分析模型在不同情境下的自我认知能力及准确性。**

- **链接: [https://arxiv.org/pdf/2604.00228](https://arxiv.org/pdf/2604.00228)**

> **作者:** Tanay Gondil
>
> **备注:** 11 pages, 5 figures
>
> **摘要:** Large language models are trained to refuse harmful requests, but can they accurately predict when they will refuse before responding? We investigate this question through a systematic study where models first predict their refusal behavior, then respond in a fresh context. Across 3754 datapoints spanning 300 requests, we evaluate four frontier models: Claude Sonnet 4, Claude Sonnet 4.5, GPT-5.2, and Llama 3.1 405B. Using signal detection theory (SDT), we find that all models exhibit high introspective sensitivity (d' = 2.4-3.5), but sensitivity drops substantially at safety boundaries. We observe generational improvement within Claude (Sonnet 4.5: 95.7 percent accuracy vs Sonnet 4: 93.0 percent), while GPT-5.2 shows lower accuracy (88.9 percent) with more variable behavior. Llama 405B achieves high sensitivity but exhibits strong refusal bias and poor calibration, resulting in lower overall accuracy (80.0 percent). Topic-wise analysis reveals weapons-related queries are consistently hardest for introspection. Critically, confidence scores provide actionable signal: restricting to high-confidence predictions yields 98.3 percent accuracy for well-calibrated models, enabling practical confidence-based routing for safety-critical deployments.
>
---
#### [new 071] How Do Language Models Process Ethical Instructions? Deliberation, Consistency, and Other-Recognition Across Four Models
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于语言模型伦理处理研究，旨在探讨模型如何内部处理伦理指令。通过多模型、多语言实验，发现不同处理类型及指令格式的影响，揭示安全与伦理处理的分离性。**

- **链接: [https://arxiv.org/pdf/2604.00021](https://arxiv.org/pdf/2604.00021)**

> **作者:** Hiroki Fukui
>
> **备注:** 34 pages, 7 figures, 4 tables. Preprint. OSF pre-registration: this http URL. Companion paper: arXiv:2603.04904
>
> **摘要:** Alignment safety research assumes that ethical instructions improve model behavior, but how language models internally process such instructions remains unknown. We conducted over 600 multi-agent simulations across four models (Llama 3.3 70B, GPT-4o mini, Qwen3-Next-80B-A3B, Sonnet 4.5), four ethical instruction formats (none, minimal norm, reasoned norm, virtue framing), and two languages (Japanese, English). Confirmatory analysis fully replicated the Llama Japanese dissociation pattern from a prior study ($\mathrm{BF}_{10} > 10$ for all three hypotheses), but none of the other three models reproduced this pattern, establishing it as model-specific. Three new metrics -- Deliberation Depth (DD), Value Consistency Across Dilemmas (VCAD), and Other-Recognition Index (ORI) -- revealed four distinct ethical processing types: Output Filter (GPT; safe outputs, no processing), Defensive Repetition (Llama; high consistency through formulaic repetition), Critical Internalization (Qwen; deep deliberation, incomplete integration), and Principled Consistency (Sonnet; deliberation, consistency, and other-recognition co-occurring). The central finding is an interaction between processing capacity and instruction format: in low-DD models, instruction format has no effect on internal processing; in high-DD models, reasoned norms and virtue framing produce opposite effects. Lexical compliance with ethical instructions did not correlate with any processing metric at the cell level ($r = -0.161$ to $+0.256$, all $p > .22$; $N = 24$; power limited), suggesting that safety, compliance, and ethical processing are largely dissociable. These processing types show structural correspondence to patterns observed in clinical offender treatment, where formal compliance without internal processing is a recognized risk signal.
>
---
#### [new 072] Temporal Dependencies in In-Context Learning: The Role of Induction Heads
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究LLM在上下文学习中的时间依赖性，聚焦于诱导头的作用。旨在揭示模型如何处理时序信息，通过实验验证诱导头对序列回忆的影响。**

- **链接: [https://arxiv.org/pdf/2604.01094](https://arxiv.org/pdf/2604.01094)**

> **作者:** Anooshka Bajaj; Deven Mahesh Mistry; Sahaj Singh Maini; Yash Aggarwal; Billy Dickson; Zoran Tiganj
>
> **摘要:** Large language models (LLMs) exhibit strong in-context learning capabilities, but how they track and retrieve information from context remains underexplored. Drawing on the free recall paradigm in cognitive science (where participants recall list items in any order), we show that several open-source LLMs consistently display a serial-recall-like pattern, assigning peak probability to tokens that immediately follow a repeated token in the input sequence. Through systematic ablation experiments, we show that induction heads, specialized attention heads that attend to the token following a previous occurrence of the current token, play an important role in this phenomenon. Removing heads with a high induction score substantially reduces the +1 lag bias, whereas ablating random heads does not reproduce the same reduction. We also show that removing heads with high induction scores impairs the performance of models prompted to do serial recall using few-shot learning to a larger extent than removing random heads. Our findings highlight a mechanistically specific connection between induction heads and temporal context processing in transformers, suggesting that these heads are especially important for ordered retrieval and serial-recall-like behavior during in-context learning.
>
---
#### [new 073] Think Twice Before You Write -- an Entropy-based Decoding Strategy to Enhance LLM Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在提升大模型的推理能力。针对传统解码方法的不足，提出基于熵的自适应解码策略，有效减少错误传播并优化计算资源使用。**

- **链接: [https://arxiv.org/pdf/2604.00018](https://arxiv.org/pdf/2604.00018)**

> **作者:** Jiashu He; Meizhu Liu; Olaitan P Olaleye; Amit Agarwal; M. Avendi; Yassi Abbasi; Matthew Rowe; Hitesh Laxmichand Patel; Paul Li; Tao Sheng; Sujith Ravi; Dan Roth
>
> **摘要:** Decoding strategies play a central role in shaping the reasoning ability of large language models (LLMs). Traditional methods such as greedy decoding and beam search often suffer from error propagation, while sampling-based approaches introduce randomness without adequate robustness. Self-consistency improves reliability by aggregating multiple rollouts, but incurs significant computational overhead. We propose an entropy-guided decoding framework that introduces token-level adaptivity into generation. At each step, the model computes the entropy of the token distribution, identifies high-uncertainty positions, and selectively branches on these vulnerable points. A dynamic pool of partial rollouts is maintained and expanded until solutions are completed, concentrating computation where uncertainty is greatest and avoiding unnecessary exploration in confident regions. To enable efficient termination, we apply a rollout-level Entropy After </Think> (EAT) stopping criterion by performing entropy evaluation after the full reasoning trace, rather than incrementally at every step. Experiments on GSM8K, AMC2023, and their perturbed variants demonstrate that our method achieves consistently strong accuracy. Notably, on smaller LLMs, performance is comparable to GPT-5 while operating at a fraction of the cost.
>
---
#### [new 074] LLM Essay Scoring Under Holistic and Analytic Rubrics: Prompt Effects and Bias
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于教育评估任务，研究LLM在作文评分中的表现，解决其与人类评分的一致性及偏差问题，通过实验分析不同提示效果和偏差稳定性。**

- **链接: [https://arxiv.org/pdf/2604.00259](https://arxiv.org/pdf/2604.00259)**

> **作者:** Filip J. Kucia; Anirban Chakraborty; Anna Wróblewska
>
> **摘要:** Despite growing interest in using Large Language Models (LLMs) for educational assessment, it remains unclear how closely they align with human scoring. We present a systematic evaluation of instruction-tuned LLMs across three open essay-scoring datasets (ASAP 2.0, ELLIPSE, and DREsS) that cover both holistic and analytic scoring. We analyze agreement with human consensus scores, directional bias, and the stability of bias estimates. Our results show that strong open-weight models achieve moderate to high agreement with humans on holistic scoring (Quadratic Weighted Kappa about 0.6), but this does not transfer uniformly to analytic scoring. In particular, we observe large and stable negative directional bias on Lower-Order Concern (LOC) traits, such as Grammar and Conventions, meaning that models often score these traits more harshly than human raters. We also find that concise keyword-based prompts generally outperform longer rubric-style prompts in multi-trait analytic scoring. To quantify the amount of data needed to detect these systematic deviations, we compute the minimum sample size at which a 95% bootstrap confidence interval for the mean bias excludes zero. This analysis shows that LOC bias is often detectable with very small validation sets, whereas Higher-Order Concern (HOC) traits typically require much larger samples. These findings support a bias-correction-first deployment strategy: instead of relying on raw zero-shot scores, systematic score offsets can be estimated and corrected using small human-labeled bias-estimation sets, without requiring large-scale fine-tuning.
>
---
#### [new 075] LinearARD: Linear-Memory Attention Distillation for RoPE Restoration
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型优化任务，解决长序列处理中性能下降问题。通过LinearARD方法，恢复RoPE编码，提升长上下文效果，仅需少量训练数据。**

- **链接: [https://arxiv.org/pdf/2604.00004](https://arxiv.org/pdf/2604.00004)**

> **作者:** Ning Yang; Hengyu Zhong; Wentao Wang; Baoliang Tian; Haijun Zhang; Jun Wang
>
> **摘要:** The extension of context windows in Large Language Models is typically facilitated by scaling positional encodings followed by lightweight Continual Pre-Training (CPT). While effective for processing long sequences, this paradigm often disrupts original model capabilities, leading to performance degradation on standard short-text benchmarks. We propose LinearARD, a self-distillation method that restores Rotary Position Embeddings (RoPE)-scaled students through attention-structure consistency with a frozen native-RoPE teacher. Rather than matching opaque hidden states, LinearARD aligns the row-wise distributions of dense $Q/Q$, $K/K$, and $V/V$ self-relation matrices to directly supervise attention dynamics. To overcome the quadratic memory bottleneck of $n \times n$ relation maps, we introduce a linear-memory kernel. This kernel leverages per-token log-sum-exp statistics and fuses logit recomputation into the backward pass to compute exact Kullback-Leibler divergence and gradients. On LLaMA2-7B extended from 4K to 32K, LinearARD recovers 98.3\% of the short-text performance of state-of-the-art baselines while surpassing them on long-context benchmarks. Notably, our method achieves these results using only \textbf{4.25M} training tokens compared to the \textbf{256M} tokens required by LongReD and CPT. Our code is available at this https URL.
>
---
#### [new 076] AfrIFact: Cultural Information Retrieval, Evidence Extraction and Fact Checking for African Languages
- **分类: cs.CL**

- **简介: 该论文提出AfrIFact数据集，解决非洲语言的事实核查问题，涵盖信息检索、证据提取和事实验证任务，旨在提升低资源语言的准确性和可靠性。**

- **链接: [https://arxiv.org/pdf/2604.00706](https://arxiv.org/pdf/2604.00706)**

> **作者:** Israel Abebe Azime; Jesujoba Oluwadara Alabi; Crystina Zhang; Iffat Maab; Atnafu Lambebo Tonja; Tadesse Destaw Belay; Folasade Peace Alabi; Salomey Osei; Saminu Mohammad Aliyu; Nkechinyere Faith Aguobi; Bontu Fufa Balcha; Blessing Kudzaishe Sibanda; Davis David; Mouhamadane Mboup; Daud Abolade; Neo Putini; Philipp Slusallek; David Ifeoluwa Adelani; Dietrich Klakow
>
> **摘要:** Assessing the veracity of a claim made online is a complex and important task with real-world implications. When these claims are directed at communities with limited access to information and the content concerns issues such as healthcare and culture, the consequences intensify, especially in low-resource languages. In this work, we introduce AfrIFact, a dataset that covers the necessary steps for automatic fact-checking (i.e., information retrieval, evidence extraction, and fact checking), in ten African languages and English. Our evaluation results show that even the best embedding models lack cross-lingual retrieval capabilities, and that cultural and news documents are easier to retrieve than healthcare-domain documents, both in large corpora and in single documents. We show that LLMs lack robust multilingual fact-verification capabilities in African languages, while few-shot prompting improves performance by up to 43% in AfriqueQwen-14B, and task-specific fine-tuning further improves fact-checking accuracy by up to 26%. These findings, along with our release of the AfrIFact dataset, encourage work on low-resource information retrieval, evidence retrieval, and fact checking.
>
---
#### [new 077] "Who Am I, and Who Else Is Here?" Behavioral Differentiation Without Role Assignment in Multi-Agent LLM Systems
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究多智能体大语言模型在互动中的行为分化问题，通过实验分析群体组成、命名和提示对行为的影响，揭示了交互导致的多样化现象。**

- **链接: [https://arxiv.org/pdf/2604.00026](https://arxiv.org/pdf/2604.00026)**

> **作者:** Houssam EL Kandoussi
>
> **备注:** 9 pages, 11 figures, 5 tables
>
> **摘要:** When multiple large language models interact in a shared conversation, do they develop differentiated social roles or converge toward uniform behavior? We present a controlled experimental platform that orchestrates simultaneous multi-agent discussions among 7 heterogeneous LLMs on a unified inference backend, systematically varying group composition, naming conventions, and prompt structure across 12 experimental series (208 runs, 13,786 coded messages). Each message is independently coded on six behavioral flags by two LLM judges from distinct model families (Gemini 3.1 Pro and Claude Sonnet 4.6), achieving mean Cohen's kappa = 0.78 with conservative intersection-based adjudication. Human validation on 609 randomly stratified messages confirmed coding reliability (mean kappa = 0.73 vs. Gemini). We find that (1) heterogeneous groups exhibit significantly richer behavioral differentiation than homogeneous groups (cosine similarity 0.56 vs. 0.85; p < 10^-5, r = 0.70); (2) groups spontaneously exhibit compensatory response patterns when an agent crashes; (3) revealing real model names significantly increases behavioral convergence (cosine 0.56 to 0.77, p = 0.001); and (4) removing all prompt scaffolding converges profiles to homogeneous-level similarity (p < 0.001). Critically, these behaviors are absent when agents operate in isolation, confirming that behavioral diversity is a structured, reproducible phenomenon driven by the interaction of architectural heterogeneity, group context, and prompt-level scaffolding.
>
---
#### [new 078] Phase transition on a context-sensitive random language model with short range interactions
- **分类: cs.CL; cond-mat.stat-mech; stat.ML**

- **简介: 该论文属于语言模型与统计物理交叉研究，旨在探讨语言模型中的相变是否源于语言本质。通过构建具有短程相互作用的上下文敏感模型，验证了相变由语言内在特性引起。**

- **链接: [https://arxiv.org/pdf/2604.00947](https://arxiv.org/pdf/2604.00947)**

> **作者:** Yuma Toji; Jun Takahashi; Vwani Roychowdhury; Hideyuki Miyahara
>
> **摘要:** Since the random language model was proposed by E. DeGiuli [Phys. Rev. Lett. 122, 128301], language models have been investigated intensively from the viewpoint of statistical mechanics. Recently, the existence of a Berezinskii--Kosterlitz--Thouless transition was numerically demonstrated in models with long-range interactions between symbols. In statistical mechanics, it has long been known that long-range interactions can induce phase transitions. Therefore, it has remained unclear whether phase transitions observed in language models originate from genuinely linguistic properties that are absent in conventional spin models. In this study, we construct a random language model with short-range interactions and numerically investigate its statistical properties. Our model belongs to the class of context-sensitive grammars in the Chomsky hierarchy and allows explicit reference to contexts. We find that a phase transition occurs even when the model refers only to contexts whose length remains constant with respect to the sentence length. This result indicates that finite-temperature phase transitions in language models are genuinely induced by the intrinsic nature of language, rather than by long-range interactions.
>
---
#### [new 079] Multimodal Language Models Cannot Spot Spatial Inconsistencies
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于多模态语言模型任务，旨在解决模型在跨视角3D几何推理中的空间不一致识别问题。研究提出生成不一致图像对的方法，验证模型表现，发现其显著低于人类。**

- **链接: [https://arxiv.org/pdf/2604.00799](https://arxiv.org/pdf/2604.00799)**

> **作者:** Om Khangaonkar; Hadi J. Rad; Hamed Pirsiavash
>
> **摘要:** Spatial consistency is a fundamental property of the visual world and a key requirement for models that aim to understand physical reality. Despite recent advances, multimodal large language models (MLLMs) often struggle to reason about 3D geometry across multiple views. Rather than asking models to describe scene attributes, we introduce a more challenging task: given two views of the same scene, identify the object that violates 3D motion consistency. We propose a simple and scalable method for generating realistic, spatially inconsistent image pairs from multi-view scenes, enabling systematic evaluation of this capability. Our results show that state-of-the-art MLLMs significantly underperform human observers and exhibit substantial variability across different scene attributes, revealing a fragile and incomplete understanding of 3D structure. We hope our findings underscore the need for approaches that develop a more deeply grounded understanding of the physical world.
>
---
#### [new 080] FGR-ColBERT: Identifying Fine-Grained Relevance Tokens During Retrieval
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于信息检索任务，旨在解决传统检索模型无法提供细粒度相关证据的问题。通过改进ColBERT模型，集成来自大语言模型的细粒度相关信号，提升检索精度同时保持效率。**

- **链接: [https://arxiv.org/pdf/2604.00242](https://arxiv.org/pdf/2604.00242)**

> **作者:** Antonín Jarolím; Martin Fajčík
>
> **摘要:** Document retrieval identifies relevant documents but does not provide fine-grained evidence cues, such as specific relevant spans. A possible solution is to apply an LLM after retrieval; however, this introduces significant computational overhead and limits practical deployment. We propose FGR-ColBERT, a modification of ColBERT retrieval model that integrates fine-grained relevance signals distilled from an LLM directly into the retrieval function. Experiments on MS MARCO show that FGR-ColBERT (110M) achieves a token-level F1 of 64.5, exceeding the 62.8 of Gemma 2 (27B), despite being approximately 245 times smaller. At the same time, it preserves retrieval effectiveness (99% relative Recall@50) and remains efficient, incurring only a ~1.12x latency overhead compared to the original ColBERT.
>
---
#### [new 081] Logarithmic Scores, Power-Law Discoveries: Disentangling Measurement from Coverage in Agent-Based Evaluation
- **分类: cs.AI; cs.CL; cs.HC; cs.MA**

- **简介: 该论文研究LLM代理评估系统，解决评估可信度与效率问题。通过实验分析得分与发现覆盖的关系，揭示其对数和幂律增长规律。**

- **链接: [https://arxiv.org/pdf/2604.00477](https://arxiv.org/pdf/2604.00477)**

> **作者:** HyunJoon Jung; William Na
>
> **摘要:** LLM-based agent judges are an emerging approach to evaluating conversational AI, yet a fundamental uncertainty remains: can we trust their assessments, and if so, how many are needed? Through 960 sessions with two model pairs across 15 tasks, we show that persona-based agent judges produce evaluations indistinguishable from human raters in a Turing-style validation. We then identify a score-coverage dissociation: quality scores improve logarithmically with panel size, while unique issue discoveries follow a sublinear power law-both exhibit diminishing returns, but scores saturate roughly twice as fast as discoveries. We hypothesize this reflects a power law distribution of the finding space: critical issues are discovered first by small panels, while corner cases require progressively larger panels, analogous to species accumulation curves in ecology. The mechanism traces to ensemble diversity-Big Five personality conditioning makes agents probe different quality dimensions, with expert judges acting as adversarial probes that push discovery into the tail of the finding distribution. A controlled ablation confirms that structured persona conditioning, not simple prompting, is required to produce these scaling properties.
>
---
#### [new 082] Benchmarking and Mechanistic Analysis of Vision-Language Models for Cross-Depiction Assembly Instruction Alignment
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉-语言对齐任务，旨在解决装配图与视频间的跨表述匹配问题。通过构建基准数据集并分析19个模型，发现视觉编码是提升对齐性能的关键。**

- **链接: [https://arxiv.org/pdf/2604.00913](https://arxiv.org/pdf/2604.00913)**

> **作者:** Zhuchenyang Liu; Yao Zhang; Yu Xiao
>
> **摘要:** 2D assembly diagrams are often abstract and hard to follow, creating a need for intelligent assistants that can monitor progress, detect errors, and provide step-by-step guidance. In mixed reality settings, such systems must recognize completed and ongoing steps from the camera feed and align them with the diagram instructions. Vision Language Models (VLMs) show promise for this task, but face a depiction gap because assembly diagrams and video frames share few visual features. To systematically assess this gap, we construct IKEA-Bench, a benchmark of 1,623 questions across 6 task types on 29 IKEA furniture products, and evaluate 19 VLMs (2B-38B) under three alignment strategies. Our key findings: (1) assembly instruction understanding is recoverable via text, but text simultaneously degrades diagram-to-video alignment; (2) architecture family predicts alignment accuracy more strongly than parameter count; (3) video understanding remains a hard bottleneck unaffected by strategy. A three-level mechanistic analysis further reveals that diagrams and video occupy disjoint ViT subspaces, and that adding text shifts models from visual to text-driven reasoning. These results identify visual encoding as the primary target for improving cross-depiction robustness. Project page: this https URL
>
---
#### [new 083] MF-QAT: Multi-Format Quantization-Aware Training for Elastic Inference
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于模型压缩任务，解决多精度部署问题。通过多格式量化感知训练（MF-QAT）和转换方法，实现模型在不同精度下的弹性推理。**

- **链接: [https://arxiv.org/pdf/2604.00529](https://arxiv.org/pdf/2604.00529)**

> **作者:** Zifei Xu; Sayeh Sharify; Hesham Mostafa
>
> **摘要:** Quantization-aware training (QAT) is typically performed for a single target numeric format, while practical deployments often need to choose numerical precision at inference time based on hardware support or runtime constraints. We study multi-format QAT, where a single model is trained to be robust across multiple quantization formats. We find that multi-format QAT can match single-format QAT at each target precision, yielding one model that performs well overall across different formats, even formats that were not seen during training. To enable practical deployment, we propose the Slice-and-Scale conversion procedure for both MXINT and MXFP that converts a high-precision representation into lower-precision formats without re-training. Building on this, we introduce a pipeline that (i) trains a model with multi-format QAT, (ii) stores a single anchor format checkpoint (MXINT8/MXFP8), and (iii) allows on-the-fly conversion to lower MXINT or MXFP formats at runtime with negligible-or no-additional accuracy degradation. Together, these components provide a practical path to elastic precision scaling and allow selecting the runtime format at inference time across diverse deployment targets.
>
---
#### [new 084] True (VIS) Lies: Analyzing How Generative AI Recognizes Intentionality, Rhetoric, and Misleadingness in Visualization Lies
- **分类: cs.HC; cs.CL; cs.CV**

- **简介: 该论文属于视觉误导识别任务，旨在分析生成式AI在识别可视化谎言中的意图、修辞和误导性方面的能力。研究通过实验和用户测试评估了多种大语言模型的表现。**

- **链接: [https://arxiv.org/pdf/2604.01181](https://arxiv.org/pdf/2604.01181)**

> **作者:** Graziano Blasilli; Marco Angelini
>
> **摘要:** This study investigates the ability of multimodal Large Language Models (LLMs) to identify and interpret misleading visualizations, and recognize these observations along with their underlying causes and potential intentionality. Our analysis leverages concepts from visualization rhetoric and a newly developed taxonomy of authorial intents as explanatory lenses. We formulated three research questions and addressed them experimentally using a dataset of 2,336 COVID-19-related tweets, half of which contain misleading visualizations, and supplemented it with real-world examples of perceptual, cognitive, and conceptual errors drawn from VisLies, the IEEE VIS community event dedicated to showcasing deceptive and misleading visualizations. To ensure broad coverage of the current LLM landscape, we evaluated 16 state-of-the-art models. Among them, 15 are open-weight models, spanning a wide range of model sizes, architectural families, and reasoning capabilities. The selection comprises small models, namely Nemotron-Nano-V2-VL (12B parameters), Mistral-Small-3.2 (24B), DeepSeek-VL2 (27B), Gemma3 (27B), and GTA1 (32B); medium-sized models, namely Qianfan-VL (70B), Molmo (72B), GLM-4.5V (108B), LLaVA-NeXT (110B), and Pixtral-Large (124B); and large models, namely Qwen3-VL (235B), InternVL3.5 (241B), Step3 (321B), Llama-4-Maverick (400B), and Kimi-K2.5 (1000B). In addition, we employed OpenAI GPT-5.4, a frontier proprietary model. To establish a human perspective on these tasks, we also conducted a user study with visualization experts to assess how people perceive rhetorical techniques and the authorial intentions behind the same misleading visualizations. This allows comparison between model and expert behavior, revealing similarities and differences that provide insights into where LLMs align with human judgment and where they diverge.
>
---
#### [new 085] Beyond Symbolic Solving: Multi Chain-of-Thought Voting for Geometric Reasoning in Large Language Models
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于几何推理任务，旨在解决大语言模型中逻辑推理不足的问题。提出MARS-GPS方法，通过多链式思考投票提升准确性。**

- **链接: [https://arxiv.org/pdf/2604.00890](https://arxiv.org/pdf/2604.00890)**

> **作者:** Md. Abu Bakor Siddique; Shahrin Hossain; Sadman Ahmed Siam; Syed Rifat Raiyan; Hasan Mahmud; Md Kamrul Hasan
>
> **备注:** Under review, 4 figures, 7 tables
>
> **摘要:** Geometric Problem Solving (GPS) remains at the heart of enhancing mathematical reasoning in large language models because it requires the combination of diagrammatic understanding, symbolic manipulation and logical inference. In existing literature, researchers have chiefly focused on synchronising the diagram descriptions with text literals and solving the problem. In this vein, they have either taken a neural, symbolic or neuro-symbolic approach. But this solves only the first two of the requirements, namely diagrammatic understanding and symbolic manipulation, while leaving logical inference underdeveloped. The logical inference is often limited to one chain-of-thought (CoT). To address this weakness in hitherto existing models, this paper proposes MARS-GPS, that generates multiple parallel reasoning rollouts augmented with Python code execution for numerical verification, ranks them using token-level entropy as a confidence signal, and aggregates answers through a multi-stage voting and self-verification pipeline. Empirical results show that MARS-GPS with 8 parallel rollouts achieves 88.8% on Geometry3K, a nearly +11% improvement over the prior state-of-the-art, with accuracy scaling consistently as the number of rollouts increases from 1 to 16 (+6.0% on ablation subset). We provide our code and data in an anonymous repository: this https URL.
>
---
#### [new 086] Not My Truce: Personality Differences in AI-Mediated Workplace Negotiation
- **分类: cs.HC; cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于人机协作任务，探讨AI在职场谈判中的效果差异。研究解决AI干预效果不一致的问题，通过实验分析不同人格类型对AI coaching的反应，提出个性化干预设计建议。**

- **链接: [https://arxiv.org/pdf/2604.00464](https://arxiv.org/pdf/2604.00464)**

> **作者:** Veda Duddu; Jash Rajesh Parekh; Andy Mao; Hanyi Min; Ziang Xiao; Vedant Das Swain; Koustuv Saha
>
> **摘要:** AI-driven conversational coaching is increasingly used to support workplace negotiation, yet prior work assumes uniform effectiveness across users. We challenge this assumption by examining how individual differences, particularly personality traits, moderate coaching outcomes. We conducted a between-subjects experiment (N=267) comparing theory-driven AI (Trucey), general-purpose AI (Control-AI), and a traditional negotiation handbook (Control-NoAI). Participants were clustered into three profiles -- resilient, overcontrolled, and undercontrolled -- based on the Big-Five personality traits and ARC typology. Resilient workers achieved broad psychological gains primarily from the handbook, overcontrolled workers showed outcome-specific improvements with theory-driven AI, and undercontrolled workers exhibited minimal effects despite engaging with the frameworks. These patterns suggest personality as a predictor of readiness beyond stage-based tailoring: vulnerable users benefit from targeted rather than comprehensive interventions. The study advances understanding of personality-determined intervention prerequisites and highlights design implications for adaptive AI coaching systems that align support intensity with individual readiness, rather than assuming universal effectiveness.
>
---
#### [new 087] ParetoBandit: Budget-Paced Adaptive Routing for Non-Stationary LLM Serving
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出ParetoBandit，解决非平稳LLM服务中的成本与质量平衡问题。通过自适应路由策略，在预算约束下动态调整模型选择，提升服务质量并控制成本。**

- **链接: [https://arxiv.org/pdf/2604.00136](https://arxiv.org/pdf/2604.00136)**

> **作者:** Annette Taberner-Miller
>
> **备注:** 27 pages, 15 figures, 13 tables. Code available at this https URL
>
> **摘要:** Production LLM serving often relies on multi-model portfolios spanning a ~530x cost range, where routing decisions trade off quality against cost. This trade-off is non-stationary: providers revise pricing, model quality can regress silently, and new models must be integrated without downtime. We present ParetoBandit, an open-source adaptive router built on cost-aware contextual bandits that is the first to simultaneously enforce dollar-denominated budgets, adapt online to such shifts, and onboard new models at runtime. ParetoBandit closes these gaps through three mechanisms. An online primal-dual budget pacer enforces a per-request cost ceiling over an open-ended stream, replacing offline penalty tuning with closed-loop control. Geometric forgetting on sufficient statistics enables rapid adaptation to price and quality shifts while bootstrapping from offline priors. A hot-swap registry lets operators add or remove models at runtime, with a brief forced-exploration phase for each newcomer, after which UCB selection discovers its quality-cost niche from live traffic alone. We evaluate ParetoBandit across four deployment scenarios on 1,824 prompts routed through a three-model portfolio. Across seven budget ceilings, mean per-request cost never exceeds the target by more than 0.4%. When conditions shift, the system adapts: an order-of-magnitude price cut on the costliest model yields up to +0.071 quality lift, and a silent quality regression is detected and rerouted within budget. A cold-started model reaches meaningful adoption within ~142 steps without breaching the cost ceiling. The router discriminates rather than blindly adopting: expensive models are budget-gated and low-quality models rejected after bounded exploration. End-to-end routing latency is 9.8ms on CPU -- less than 0.4% of typical inference time -- with the routing decision itself taking just 22.5us.
>
---
#### [new 088] First Logit Boosting: Visual Grounding Method to Mitigate Object Hallucination in Large Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉语言模型任务，旨在解决对象幻觉问题。提出FLB方法，在不增加训练成本的情况下，通过增强视觉信息稳定性来减少幻觉生成。**

- **链接: [https://arxiv.org/pdf/2604.00455](https://arxiv.org/pdf/2604.00455)**

> **作者:** Jiwoo Ha; Jongwoo Baek; Jinhyun So
>
> **备注:** 19 pages, 13 figures
>
> **摘要:** Recent Large Vision-Language Models (LVLMs) have demonstrated remarkable performance across various multimodal tasks that require understanding both visual and linguistic inputs. However, object hallucination -- the generation of nonexistent objects in answers -- remains a persistent challenge. Although several approaches such as retraining and external grounding methods have been proposed to mitigate this issue, they still suffer from high data costs or structural complexity. Training-free methods such as Contrastive Decoding (CD) are more cost-effective, avoiding additional training or external models, but still suffer from long-term decay, where visual grounding weakens and language priors dominate as the generation progresses. In this paper, we propose First Logit Boosting (FLB), a simple yet effective training-free technique designed to alleviate long-term decay in LVLMs. FLB stores the logit of the first generated token and adds it to subsequent token predictions, effectively mitigating long-term decay of visual information. We observe that FLB (1) sustains the visual information embedded in the first token throughout generation, and (2) suppresses hallucinated words through the stabilizing effect of the ``The'' token. Experimental results show that FLB significantly reduces object hallucination across various tasks, benchmarks, and backbone models. Notably, it causes negligible inference overhead, making it highly applicable to real-time multimodal systems. Code is available at this https URL
>
---
#### [new 089] One Panel Does Not Fit All: Case-Adaptive Multi-Agent Deliberation for Clinical Prediction
- **分类: cs.AI; cs.CL; cs.MA**

- **简介: 该论文属于临床预测任务，解决复杂病例预测不一致的问题。通过动态组建专家小组进行三值投票，提升预测准确性与透明度。**

- **链接: [https://arxiv.org/pdf/2604.00085](https://arxiv.org/pdf/2604.00085)**

> **作者:** Yuxing Lu; Yushuhong Lin; Jason Zhang
>
> **摘要:** Large language models applied to clinical prediction exhibit case-level heterogeneity: simple cases yield consistent outputs, while complex cases produce divergent predictions under minor prompt changes. Existing single-agent strategies sample from one role-conditioned distribution, and multi-agent frameworks use fixed roles with flat majority voting, discarding the diagnostic signal in disagreement. We propose CAMP (Case-Adaptive Multi-agent Panel), where an attending-physician agent dynamically assembles a specialist panel tailored to each case's diagnostic uncertainty. Each specialist evaluates candidates via three-valued voting (KEEP/REFUSE/NEUTRAL), enabling principled abstention outside one's expertise. A hybrid router directs each diagnosis through strong consensus, fallback to the attending physician's judgment, or evidence-based arbitration that weighs argument quality over vote counts. On diagnostic prediction and brief hospital course generation from MIMIC-IV across four LLM backbones, CAMP consistently outperforms strong baselines while consuming fewer tokens than most competing multi-agent methods, with voting records and arbitration traces offering transparent decision audits.
>
---
#### [new 090] PixelPrune: Pixel-Level Adaptive Visual Token Reduction via Predictive Coding
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出PixelPrune，解决视觉语言模型中高分辨率输入带来的计算负担问题，通过像素级冗余压缩提升推理效率。**

- **链接: [https://arxiv.org/pdf/2604.00886](https://arxiv.org/pdf/2604.00886)**

> **作者:** Nan Wang; Zhiwei Jin; Chen Chen; Haonan Lu
>
> **摘要:** Document understanding and GUI interaction are among the highest-value applications of Vision-Language Models (VLMs), yet they impose exceptionally heavy computational burden: fine-grained text and small UI elements demand high-resolution inputs that produce tens of thousands of visual tokens. We observe that this cost is largely wasteful -- across document and GUI benchmarks, only 22--71\% of image patches are pixel-unique, the rest being exact duplicates of another patch in the same image. We propose \textbf{PixelPrune}, which exploits this pixel-level redundancy through predictive-coding-based compression, pruning redundant patches \emph{before} the Vision Transformer (ViT) encoder. Because it operates in pixel space prior to any neural computation, PixelPrune accelerates both the ViT encoder and the downstream LLM, covering the full inference pipeline. The method is training-free, requires no learnable parameters, and supports pixel-lossless compression ($\tau{=}0$) as well as controlled lossy compression ($\tau{>}0$). Experiments across three model scales and document and GUI benchmarks show that PixelPrune maintains competitive task accuracy while delivering up to 4.2$\times$ inference speedup and 1.9$\times$ training acceleration. Code is available at this https URL.
>
---
#### [new 091] Do Phone-Use Agents Respect Your Privacy?
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于隐私评估任务，旨在解决手机代理是否尊重隐私的问题。通过构建框架MyPhoneBench，评估代理在完成任务时的隐私行为，发现现有模型在隐私保护上存在不足。**

- **链接: [https://arxiv.org/pdf/2604.00986](https://arxiv.org/pdf/2604.00986)**

> **作者:** Zhengyang Tang; Ke Ji; Xidong Wang; Zihan Ye; Xinyuan Wang; Yiduo Guo; Ziniu Li; Chenxin Li; Jingyuan Hu; Shunian Chen; Tongxu Luo; Jiaxi Bi; Zeyu Qin; Shaobo Wang; Xin Lai; Pengyuan Lyu; Junyi Li; Can Xu; Chengquan Zhang; Han Hu; Ming Yan; Benyou Wang
>
> **备注:** work in progress
>
> **摘要:** We study whether phone-use agents respect privacy while completing benign mobile tasks. This question has remained hard to answer because privacy-compliant behavior is not operationalized for phone-use agents, and ordinary apps do not reveal exactly what data agents type into which form entries during execution. To make this question measurable, we introduce MyPhoneBench, a verifiable evaluation framework for privacy behavior in mobile agents. We operationalize privacy-respecting phone use as permissioned access, minimal disclosure, and user-controlled memory through a minimal privacy contract, iMy, and pair it with instrumented mock apps plus rule-based auditing that make unnecessary permission requests, deceptive re-disclosure, and unnecessary form filling observable and reproducible. Across five frontier models on 10 mobile apps and 300 tasks, we find that task success, privacy-compliant task completion, and later-session use of saved preferences are distinct capabilities, and no single model dominates all three. Evaluating success and privacy jointly reshuffles the model ordering relative to either metric alone. The most persistent failure mode across models is simple data minimization: agents still fill optional personal entries that the task does not require. These results show that privacy failures arise from over-helpful execution of benign tasks, and that success-only evaluation overestimates the deployment readiness of current phone-use agents. All code, mock apps, and agent trajectories are publicly available at~ this https URL.
>
---
#### [new 092] A Survey of On-Policy Distillation for Large Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于知识蒸馏任务，旨在解决传统离策略蒸馏中的暴露偏差问题。通过引入在策略蒸馏（OPD），让学生生成轨迹并接受教师反馈，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.00626](https://arxiv.org/pdf/2604.00626)**

> **作者:** Mingyang Song; Mao Zheng
>
> **摘要:** Knowledge distillation has become a primary mechanism for transferring reasoning and domain expertise from frontier Large Language Models (LLMs) to smaller, deployable students. However, the dominant paradigm remains \textit{off-policy}: students train on static teacher-generated data and never encounter their own errors during learning. This train--test mismatch, an instance of \textit{exposure bias}, causes prediction errors to compound autoregressively at inference time. On-Policy Distillation (OPD) addresses this by letting the student generate its own trajectories and receive teacher feedback on these self-generated outputs, grounding distillation in the theory of interactive imitation learning. Despite rapid growth spanning divergence minimization, reward-guided learning, and self-play, the OPD literature remains fragmented with no unified treatment. This survey provides the first comprehensive overview of OPD for LLMs. We introduce a unified $f$-divergence framework over on-policy samples and organize the landscape along three orthogonal dimensions: \emph{feedback signal} (logit-based, outcome-based, or self-play), \emph{teacher access} (white-box, black-box, or teacher-free), and \emph{loss granularity} (token-level, sequence-level, or hybrid). We systematically analyze representative methods, examine industrial deployments, and identify open problems including distillation scaling laws, uncertainty-aware feedback, and agent-level distillation.
>
---
#### [new 093] Two-Stage Optimizer-Aware Online Data Selection for Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于语言模型微调任务，解决在线数据选择问题。提出一种两阶段优化器感知方法，提升模型收敛和性能。**

- **链接: [https://arxiv.org/pdf/2604.00001](https://arxiv.org/pdf/2604.00001)**

> **作者:** Fangxin Wang; Peyman Baghershahi; Langzhou He; Henry Peng Zou; Sourav Medya; Philip S. Yu
>
> **备注:** 22 pages, 2 figures, 6 tables
>
> **摘要:** Gradient-based data selection offers a principled framework for estimating sample utility in large language model (LLM) fine-tuning, but existing methods are mostly designed for offline settings. They are therefore less suited to online fine-tuning, where data arrives sequentially, sample utility is step-dependent, and the effective update geometry is shaped by adaptive optimizers. We propose an optimizer-aware framework for gradient-based online data selection and reweighting in LLM fine-tuning. Our key idea is to view online selection not as static sample ranking, but as shaping the next target-oriented update under the optimizer state. We formulate this as an optimizer-aware update-matching problem, establish its connection to second-order target utility, and show why subset-level construction must account for interactions and redundancy among selected samples. Based on this view, we develop a two-stage Filter-then-Weight algorithm that first filters geometrically useful candidates and then optimizes their coefficients. To make the framework practical for LLMs, we introduce a factorized outer-product gradient representation and optimized matrix computations for long-context data. Experiments show that our method consistently improves convergence and downstream performance over existing online data selection baselines under the same data budget.
>
---
#### [new 094] Online Reasoning Calibration: Test-Time Training Enables Generalizable Conformal LLM Reasoning
- **分类: cs.LG; cs.AI; cs.CL; stat.AP; stat.ML**

- **简介: 该论文属于语言模型优化任务，解决模型在推理时的校准问题。通过引入ORCA框架，结合测试时训练和置信度估计，提升模型效率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.01170](https://arxiv.org/pdf/2604.01170)**

> **作者:** Cai Zhou; Zekai Wang; Menghua Wu; Qianyu Julie Zhu; Flora C. Shi; Chenyu Wang; Ashia Wilson; Tommi Jaakkola; Stephen Bates
>
> **备注:** 20 pages
>
> **摘要:** While test-time scaling has enabled large language models to solve highly difficult tasks, state-of-the-art results come at exorbitant compute costs. These inefficiencies can be attributed to the miscalibration of post-trained language models, and the lack of calibration in popular sampling techniques. Here, we present Online Reasoning Calibration (ORCA), a framework for calibrating the sampling process that draws upon conformal prediction and test-time training. Specifically, we introduce a meta-learning procedure that updates the calibration module for each input. This allows us to provide valid confidence estimates under distributional shift, e.g. in thought patterns that occur across different stages of reasoning, or in prompt distributions between model development and deployment. ORCA not only provides theoretical guarantees on conformal risks, but also empirically shows higher efficiency and generalization across different reasoning tasks. At risk level $\delta=0.1$, ORCA improves Qwen2.5-32B efficiency on in-distribution tasks with savings up to 47.5% with supervised labels and 40.7% with self-consistency labels. Under zero-shot out-of-domain settings, it improves MATH-500 savings from 24.8% of the static calibration baseline to 67.0% while maintaining a low empirical error rate, and the same trend holds across model families and downstream benchmarks. Our code is publicly available at this https URL.
>
---
#### [new 095] Learning to Hint for Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，解决优势消失问题。通过提出HiLL框架，联合训练提示策略和推理策略，生成自适应提示以提升学习效果。**

- **链接: [https://arxiv.org/pdf/2604.00698](https://arxiv.org/pdf/2604.00698)**

> **作者:** Yu Xia; Canwen Xu; Zhewei Yao; Julian McAuley; Yuxiong He
>
> **摘要:** Group Relative Policy Optimization (GRPO) is widely used for reinforcement learning with verifiable rewards, but it often suffers from advantage collapse: when all rollouts in a group receive the same reward, the group yields zero relative advantage and thus no learning signal. For example, if a question is too hard for the reasoner, all sampled rollouts can be incorrect and receive zero reward. Recent work addresses this issue by adding hints or auxiliary scaffolds to such hard questions so that the reasoner produces mixed outcomes and recovers a non-zero update. However, existing hints are usually fixed rather than adapted to the current reasoner, and a hint that creates learning signal under the hinted input does not necessarily improve the no-hint policy used at test time. To this end, we propose Hint Learning for Reinforcement Learning (HiLL), a framework that jointly trains a hinter policy and a reasoner policy during RL. For each hard question, the hinter generates hints online conditioned on the current reasoner's incorrect rollout, allowing hint generation to adapt to the reasoner's evolving errors. We further introduce hint reliance, which measures how strongly correct hinted trajectories depend on the hint. We derive a transferability result showing that lower hint reliance implies stronger transfer from hinted success to no-hint success, and we use this result to define a transfer-weighted reward for training the hinter. Therefore, HiLL favors hints that not only recover informative GRPO groups, but also produce signals that are more likely to improve the original no-hint policy. Experiments across multiple benchmarks show that HiLL consistently outperforms GRPO and prior hint-based baselines, demonstrating the value of adaptive and transfer-aware hint learning for RL. The code is available at this https URL.
>
---
#### [new 096] Signals: Trajectory Sampling and Triage for Agentic Interactions
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于智能代理系统优化任务，旨在解决后部署优化中轨迹评估效率低的问题。通过构建轻量信号框架，提升轨迹采样效率与信息量。**

- **链接: [https://arxiv.org/pdf/2604.00356](https://arxiv.org/pdf/2604.00356)**

> **作者:** Shuguang Chen; Adil Hafeez; Salman Paracha
>
> **摘要:** Agentic applications based on large language models increasingly rely on multi-step interaction loops involving planning, action execution, and environment feedback. While such systems are now deployed at scale, improving them post-deployment remains challenging. Agent trajectories are voluminous and non-deterministic, and reviewing each one, whether through human review or auxiliary LLMs, is slow and cost-prohibitive. We propose a lightweight, signal-based framework for triaging agentic interaction trajectories. Our approach computes cheap, broadly applicable signals from live interactions and attaches them as structured attributes for trajectory triage, identifying interactions likely to be informative without affecting online agent behavior. We organize signals into a coarse-grained taxonomy spanning interaction (misalignment, stagnation, disengagement, satisfaction), execution (failure, loop), and environment (exhaustion), designed for computation without model calls. In a controlled annotation study on $\tau$-bench, a widely used benchmark for tool-augmented agent evaluation, we show that signal-based sampling achieves an 82\% informativeness rate compared to 74\% for heuristic filtering and 54\% for random sampling, with a 1.52x efficiency gain per informative trajectory. The advantage is robust across reward strata and task domains, confirming that signals provide genuine per-trajectory informativeness gains rather than merely oversampling obvious failures. These results show that lightweight signals can serve as practical sampling infrastructure for agentic systems, and suggest a path toward preference data construction and post-deployment optimization.
>
---
#### [new 097] How Emotion Shapes the Behavior of LLMs and Agents: A Mechanistic Study
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于人工智能领域，研究情感如何影响大语言模型和智能体的行为。旨在解决情感机制在模型中的作用问题，提出E-STEER框架实现情感干预，验证情感对推理、生成、安全及多步骤行为的影响。**

- **链接: [https://arxiv.org/pdf/2604.00005](https://arxiv.org/pdf/2604.00005)**

> **作者:** Moran Sun; Tianlin Li; Yuwei Zheng; Zhenhong Zhou; Aishan Liu; Xianglong Liu; Yang Liu
>
> **备注:** 15 pages, 11 figures
>
> **摘要:** Emotion plays an important role in human cognition and performance. Motivated by this, we investigate whether analogous emotional signals can shape the behavior of large language models (LLMs) and agents. Existing emotion-aware studies mainly treat emotion as a surface-level style factor or a perception target, overlooking its mechanistic role in task processing. To address this limitation, we propose E-STEER, an interpretable emotion steering framework that enables direct representation-level intervention in LLMs and agents. It embeds emotion as a structured, controllable variable in hidden states, and with it, we examine the impact of emotion on objective reasoning, subjective generation, safety, and multi-step agent behaviors. The results reveal non-monotonic emotion-behavior relations consistent with established psychological theories, and show that specific emotions not only enhance LLM capability but also improve safety, and systematically shape multi-step agent behaviors.
>
---
#### [new 098] Screening Is Enough
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出Multiscreen模型，解决标准softmax注意力无法定义绝对相关性的任务。通过引入筛选机制，提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2604.01178](https://arxiv.org/pdf/2604.01178)**

> **作者:** Ken M. Nakanishi
>
> **备注:** 21 pages, 13 figures
>
> **摘要:** A core limitation of standard softmax attention is that it does not define a notion of absolute query--key relevance: attention weights are obtained by redistributing a fixed unit mass across all keys according to their relative scores. As a result, relevance is defined only relative to competing keys, and irrelevant keys cannot be explicitly rejected. We introduce Multiscreen, a language-model architecture built around a mechanism we call screening, which enables absolute query--key relevance. Instead of redistributing attention across all keys, screening evaluates each key against an explicit threshold, discarding irrelevant keys and aggregating the remaining keys, thereby removing global competition among keys. Across experiments, Multiscreen achieves comparable validation loss with approximately 40% fewer parameters than a Transformer baseline, enables stable optimization at substantially larger learning rates, maintains strong performance in long-context perplexity, shows little to no degradation in retrieval performance even far beyond the training context length, and reduces inference latency by up to 3.2$\times$ at 100K context length.
>
---
#### [new 099] Terminal Agents Suffice for Enterprise Automation
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 论文探讨企业自动化任务，提出终端代理可有效替代复杂系统。通过实验证明，简单终端代理结合基础模型能高效完成企业任务。**

- **链接: [https://arxiv.org/pdf/2604.00073](https://arxiv.org/pdf/2604.00073)**

> **作者:** Patrice Bechard; Orlando Marquez Ayala; Emily Chen; Jordan Skelton; Sagar Davasam; Srinivas Sunkara; Vikas Yadav; Sai Rajeswar
>
> **备注:** Pre-print. Under review for COLM2026
>
> **摘要:** There has been growing interest in building agents that can interact with digital platforms to execute meaningful enterprise tasks autonomously. Among the approaches explored are tool-augmented agents built on abstractions such as Model Context Protocol (MCP) and web agents that operate through graphical interfaces. Yet, it remains unclear whether such complex agentic systems are necessary given their cost and operational overhead. We argue that a coding agent equipped only with a terminal and a filesystem can solve many enterprise tasks more effectively by interacting directly with platform APIs. We evaluate this hypothesis across diverse real-world systems and show that these low-level terminal agents match or outperform more complex agent architectures. Our findings suggest that simple programmatic interfaces, combined with strong foundation models, are sufficient for practical enterprise automation.
>
---
#### [new 100] Ontology-Constrained Neural Reasoning in Enterprise Agentic Systems: A Neurosymbolic Architecture for Domain-Grounded AI Agents
- **分类: cs.AI; cs.CL; cs.SE**

- **简介: 该论文属于企业AI代理系统任务，旨在解决LLM的幻觉、领域漂移和合规性问题。通过神经符号架构与本体约束，提升代理的准确性、合规性和角色一致性。**

- **链接: [https://arxiv.org/pdf/2604.00555](https://arxiv.org/pdf/2604.00555)**

> **作者:** Thanh Luong Tuan
>
> **备注:** 23 pages, 7 tables, 4 figures, 33 references. Empirical evaluation: 600 runs across 5 regulated industries including Vietnamese-language domains
>
> **摘要:** Enterprise adoption of Large Language Models (LLMs) is constrained by hallucination, domain drift, and the inability to enforce regulatory compliance at the reasoning level. We present a neurosymbolic architecture implemented within the Foundation AgenticOS (FAOS) platform that addresses these limitations through ontology-constrained neural reasoning. Our approach introduces a three-layer ontological framework--Role, Domain, and Interaction ontologies--that provides formal semantic grounding for LLM-based enterprise agents. We formalize the concept of asymmetric neurosymbolic coupling, wherein symbolic ontological knowledge constrains agent inputs (context assembly, tool discovery, governance thresholds) while proposing mechanisms for extending this coupling to constrain agent outputs (response validation, reasoning verification, compliance checking). We evaluate the architecture through a controlled experiment (600 runs across five industries: FinTech, Insurance, Healthcare, Vietnamese Banking, and Vietnamese Insurance), finding that ontology-coupled agents significantly outperform ungrounded agents on Metric Accuracy (p < .001, W = .460), Regulatory Compliance (p = .003, W = .318), and Role Consistency (p < .001, W = .614), with improvements greatest where LLM parametric knowledge is weakest--particularly in Vietnam-localized domains. Our contributions include: (1) a formal three-layer enterprise ontology model, (2) a taxonomy of neurosymbolic coupling patterns, (3) ontology-constrained tool discovery via SQL-pushdown scoring, (4) a proposed framework for output-side ontological validation, (5) empirical evidence for the inverse parametric knowledge effect that ontological grounding value is inversely proportional to LLM training data coverage of the domain, and (6) a production system serving 21 industry verticals with 650+ agents.
>
---
#### [new 101] Execution-Verified Reinforcement Learning for Optimization Modeling
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出EVOM框架，解决优化建模自动化问题。通过执行验证的强化学习，无需过程监督即可跨求解器泛化，提升建模效率与适应性。**

- **链接: [https://arxiv.org/pdf/2604.00442](https://arxiv.org/pdf/2604.00442)**

> **作者:** Runda Guan; Xiangqing Shen; Jiajun Zhang; Yifan Zhang; Jian Cheng; Rui Xia
>
> **摘要:** Automating optimization modeling with LLMs is a promising path toward scalable decision intelligence, but existing approaches either rely on agentic pipelines built on closed-source LLMs with high inference latency, or fine-tune smaller LLMs using costly process supervision that often overfits to a single solver API. Inspired by reinforcement learning with verifiable rewards, we propose Execution-Verified Optimization Modeling (EVOM), an execution-verified learning framework that treats a mathematical programming solver as a deterministic, interactive verifier. Given a natural-language problem and a target solver, EVOM generates solver-specific code, executes it in a sandboxed harness, and converts execution outcomes into scalar rewards, optimized with GRPO and DAPO in a closed-loop generate-execute-feedback-update process. This outcome-only formulation removes the need for process-level supervision, and enables cross-solver generalization by switching the verification environment rather than reconstructing solver-specific datasets. Experiments on NL4OPT, MAMO, IndustryOR, and OptiBench across Gurobi, OR-Tools, and COPT show that EVOM matches or outperforms process-supervised SFT, supports zero-shot solver transfer, and achieves effective low-cost solver adaptation by continuing training under the target solver backend.
>
---
#### [new 102] Revision or Re-Solving? Decomposing Second-Pass Gains in Multi-LLM Pipelines
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文研究多大模型修订管道的增益来源，旨在解决修订效果是否仅来自错误修正的问题。通过实验分解增益成分，发现任务结构和草稿质量影响修订效果，提出需针对性设计管道。**

- **链接: [https://arxiv.org/pdf/2604.01029](https://arxiv.org/pdf/2604.01029)**

> **作者:** Jingjie Ning; Xueqi Li; Chengyu Yu
>
> **摘要:** Multi-LLM revision pipelines, in which a second model reviews and improves a draft produced by a first, are widely assumed to derive their gains from genuine error correction. We question this assumption with a controlled decomposition experiment that uses four matched conditions to separate second-pass gains into three additive components: re-solving, scaffold, and content. We evaluate this design across two model pairs on three benchmarks spanning knowledge-intensive MCQ and competitive programming. Our results show that the gains of multi-LLM revision are not monolithic, but depend on task structure, draft quality, and the type of draft information. On MCQ tasks, where the answer space is constrained and drafts provide little structural guidance, most gains are consistent with stronger-model re-solving, and directly routing queries to the stronger model can be more effective than revising a weak draft. On code generation tasks, however, two-stage prompting remains useful because even semantically null drafts can provide substantial structural scaffolding, while weak draft content can be harmful. Finally, role-reversed experiments show that strong drafts clearly benefit weak reviewers. Ultimately, our findings demonstrate that the utility of multi-LLM revision is dynamically bottlenecked by task structure and draft quality, necessitating more targeted pipeline designs rather than blanket revision strategies.
>
---
#### [new 103] Routing-Free Mixture-of-Experts
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，解决MoE模型中依赖中心化路由的问题。通过去除硬编码路由机制，让专家自主激活，提升模型的可扩展性和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.00801](https://arxiv.org/pdf/2604.00801)**

> **作者:** Yilun Liu; Jinru Han; Sikuan Yan; Volker Tresp; Yunpu Ma
>
> **备注:** Code is available at this https URL
>
> **摘要:** Standard Mixture-of-Experts (MoE) models rely on centralized routing mechanisms that introduce rigid inductive biases. We propose Routing-Free MoE which eliminates any hard-coded centralized designs including external routers, Softmax, Top-K and load balancing, instead encapsulating all activation functionalities within individual experts and directly optimized through continuous gradient flow, enabling each expert to determine its activation entirely on its own. We introduce a unified adaptive load-balancing framework to simultaneously optimize both expert-balancing and token-balancing objectives through a configurable interpolation, allowing flexible and customizable resource allocation. Extensive experiments show that Routing-Free MoE can consistently outperform baselines with better scalability and robustness. We analyze its behavior in detail and offer insights that may facilitate future MoE design ad optimization.
>
---
#### [new 104] Towards Reliable Truth-Aligned Uncertainty Estimation in Large Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于不确定性估计任务，旨在解决LLM输出不可靠的问题。通过提出TAC方法，将原始得分映射为与事实对齐的得分，提升不确定性估计的可靠性。**

- **链接: [https://arxiv.org/pdf/2604.00445](https://arxiv.org/pdf/2604.00445)**

> **作者:** Ponhvoan Srey; Quang Minh Nguyen; Xiaobao Wu; Anh Tuan Luu
>
> **摘要:** Uncertainty estimation (UE) aims to detect hallucinated outputs of large language models (LLMs) to improve their reliability. However, UE metrics often exhibit unstable performance across configurations, which significantly limits their applicability. In this work, we formalise this phenomenon as proxy failure, since most UE metrics originate from model behaviour, rather than being explicitly grounded in the factual correctness of LLM outputs. With this, we show that UE metrics become non-discriminative precisely in low-information regimes. To alleviate this, we propose Truth AnChoring (TAC), a post-hoc calibration method to remedy UE metrics, by mapping the raw scores to truth-aligned scores. Even with noisy and few-shot supervision, our TAC can support the learning of well-calibrated uncertainty estimates, and presents a practical calibration protocol. Our findings highlight the limitations of treating heuristic UE metrics as direct indicators of truth uncertainty, and position our TAC as a necessary step toward more reliable uncertainty estimation for LLMs. The code repository is available at this https URL.
>
---
#### [new 105] LinguDistill: Recovering Linguistic Ability in Vision- Language Models via Selective Cross-Modal Distillation
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态学习任务，解决视觉-语言模型在适应过程中损失语言能力的问题。通过无适配器的蒸馏方法，恢复语言模型的原有能力，同时保持视觉理解性能。**

- **链接: [https://arxiv.org/pdf/2604.00829](https://arxiv.org/pdf/2604.00829)**

> **作者:** Patrick Amadeus Irawan; Erland Hilman Fuadi; Shanu Kumar; Alham Fikri Aji; Yova Kementchedjhieva
>
> **摘要:** Adapting pretrained language models (LMs) into vision-language models (VLMs) can degrade their native linguistic capability due to representation shift and cross-modal interference introduced during multimodal adaptation. Such loss is difficult to recover, even with targeted task-specific fine-tuning using standard objectives. Prior recovery approaches typically introduce additional modules that act as intermediate alignment layers to maintain or isolate modality-specific subspaces, which increases architectural complexity, adds parameters at inference time, and limits flexibility across models and settings. We propose LinguDistill, an adapter-free distillation method that restores linguistic capability by utilizing the original frozen LM as a teacher. We overcome the key challenge of enabling vision-conditioned teacher supervision by introducing layer-wise KV-cache sharing, which exposes the teacher to the student's multimodal representations without modifying the architecture of either model. We then selectively distill the teacher's strong linguistic signal on language-intensive data to recover language capability, while preserving the student's visual grounding on multimodal tasks. As a result, LinguDistill recovers $\sim$10% of the performance lost on language and knowledge benchmarks, while maintaining comparable performance on vision-heavy tasks. Our findings demonstrate that linguistic capability can be recovered without additional modules, providing an efficient and practical solution to modality-specific degradation in multimodal models.
>
---
#### [new 106] Hierarchical Pre-Training of Vision Encoders with Large Language Models
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于视觉-语言模型任务，旨在解决视觉特征与语言模型整合不足的问题。提出HIVE框架，通过层次化交叉注意力提升多模态对齐与特征融合效果。**

- **链接: [https://arxiv.org/pdf/2604.00086](https://arxiv.org/pdf/2604.00086)**

> **作者:** Eugene Lee; Ting-Yu Chang; Jui-Huang Tsai; Jiajie Diao; Chen-Yi Lee
>
> **备注:** 17 pages, 14 figures, accepted to Computer Vision and Pattern Recognition Conference (CVPR) Workshops 2026. 5th MMFM Workshop: What is Next in Multimodal Foundation Models?
>
> **摘要:** The field of computer vision has experienced significant advancements through scalable vision encoders and multimodal pre-training frameworks. However, existing approaches often treat vision encoders and large language models (LLMs) as independent modules, limiting the integration of hierarchical visual features. In this work, we propose HIVE (Hierarchical Pre-Training of Vision Encoders), a novel framework that enhances vision-language alignment by introducing hierarchical cross-attention between the vision encoder and LLM. Unlike conventional methods that flatten image embeddings, HIVE enables structured feature fusion across multiple layers, improving gradient flow and representation learning. To optimize this interaction, we introduce a three-stage training strategy that progressively aligns the vision encoder with the LLM, ensuring stable optimization and effective multimodal fusion. Empirical evaluations demonstrate that HIVE achieves superior performance not only in image classification but also on various vision-language tasks, outperforming self-attention-based methods in benchmarks such as MME, GQA, OK-VQA, and ScienceQA. Our results highlight the benefits of hierarchical feature integration, paving the way for more efficient and expressive vision-language models.
>
---
## 更新

#### [replaced 001] Mitigating Content Effects on Reasoning in Language Models through Fine-Grained Activation Steering
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理中的推理任务，旨在解决大语言模型中的内容偏差问题。通过激活调控技术，提升模型推理的准确性与公平性。**

- **链接: [https://arxiv.org/pdf/2505.12189](https://arxiv.org/pdf/2505.12189)**

> **作者:** Marco Valentino; Geonhee Kim; Dhairya Dalal; Zhixue Zhao; André Freitas
>
> **备注:** AAAI 2026
>
> **摘要:** Large language models (LLMs) exhibit reasoning biases, often conflating content plausibility with formal logical validity. This can lead to wrong inferences in critical domains, where plausible arguments are incorrectly deemed logically valid or vice versa. This paper investigates how content biases on reasoning can be mitigated through activation steering, an inference-time technique that modulates internal activations. Specifically, after localising the layers responsible for formal and plausible inference, we investigate activation steering on a controlled syllogistic reasoning task, designed to disentangle formal validity from content plausibility. An extensive empirical analysis reveals that contrastive steering methods consistently support linear control over content biases. However, a static approach is insufficient to debias all the tested models. We then investigate how to control content effects by dynamically determining the steering parameters through fine-grained conditional methods. By introducing a novel kNN-based conditional approach (K-CAST), we demonstrate that conditional steering can effectively reduce biases on unresponsive models, achieving up to 15% absolute improvement in formal reasoning accuracy. Finally, we found that steering for content effects is robust to prompt variations, incurs minimal side effects on multilingual language modeling capabilities, and can partially generalize to different reasoning tasks. In practice, we demonstrate that activation-level interventions offer a scalable inference-time strategy for enhancing the robustness of LLMs, contributing towards more systematic and unbiased reasoning capabilities.
>
---
#### [replaced 002] LLM Router: Rethinking Routing with Prefill Activations
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于模型路由任务，旨在解决如何选择最优模型以提升性能。通过分析预填充激活，提出SharedTrunkNet实现高效路由。**

- **链接: [https://arxiv.org/pdf/2603.20895](https://arxiv.org/pdf/2603.20895)**

> **作者:** Tanay Varshney; Annie Surla; Michelle Xu; Gomathy Venkata Krishnan; Maximilian Jeblick; David Austin; Neal Vaidya; Davide Onofrio
>
> **摘要:** LLMs often achieve similar average benchmark accuracies while exhibiting complementary strengths on different subsets of queries, suggesting that a router with query-specific model selection can outperform any single model. While existing routers rely on semantic query features, they often fail to capture model-specific failures or intrinsic task difficulty. We instead study routing via internal prefill activations. Our key idea, Encoder-Target Decoupling, separates the model that produces the predictive signal (the Encoder) from the model whose correctness is being estimated (the Target), allowing open-weight encoders to predict the performance of closed-source target models. We evaluate layerwise geometric probes, finding that Fisher Separability (J) effectively identifies informative layers, supported by Effective Dimensionality (d_eff) diagnostics. We then utilize a SharedTrunkNet, a joint multi-output MLP that predicts simultaneous correctness probabilities across candidate models using concatenated prefill features. In our experiments, SharedTrunkNet consistently outperforms semantic baselines. At its best, SharedTrunkNet closes 45.58% of the gap between the strongest standalone model and the oracle while achieving 74.31% cost savings relative to the most expensive model. These results demonstrate that prefill activations provide a robust routing signal, establishing mechanistic routing as a high-performance alternative to purely semantic selection.
>
---
#### [replaced 003] PluriHopRAG: Exhaustive, Recall-Sensitive QA Through Corpus-Specific Document Structure Learning
- **分类: cs.CL; cs.IR; cs.LG**

- **简介: 该论文属于问答任务，解决多文档检索中的全量信息查找问题。提出PluriHopRAG方法，提升召回与准确率。**

- **链接: [https://arxiv.org/pdf/2510.14377](https://arxiv.org/pdf/2510.14377)**

> **作者:** Mykolas Sveistrys; Richard Kunert
>
> **摘要:** Retrieval-Augmented Generation (RAG) has been used in question answering (QA) systems to improve performance when relevant information is in one (single-hop) or multiple (multi-hop) passages. However, many real life scenarios (e.g. dealing with financial, legal, medical reports) require checking all documents for relevant information without a clear stopping condition. We term these pluri-hop questions, and formalize them by 3 conditions - recall sensitivity, exhaustiveness, and exactness. To study this setting, we introduce PluriHopWIND, a multilingual diagnostic benchmark of 48 pluri-hop questions over 191 real wind-industry reports, with high repetitiveness to reflect the challenge of distractors in real-world datasets. Naive, graph-based, and multimodal RAG methods only reach up to 40% statement-wise F1 on PluriHopWIND. Motivated by this, we propose PluriHopRAG, which learns from synthetic examples to decompose queries according to corpus-specific document structure, and employs a cross-encoder filter at the document level to minimize costly LLM reasoning. We test PluriHopRAG on PluriHopWIND and the Loong benchmark built on financial, legal and scientific reports. On PluriHopWIND, our method shows 18-52% F1 score improvement across base LLMs, while on Loong, we show 33% improvement over long-context reasoning and 52% improvement over naive RAG.
>
---
#### [replaced 004] Benchmarking Educational LLMs with Analytics: A Case Study on Gender Bias in Feedback
- **分类: cs.CL; cs.AI; cs.CY; cs.HC**

- **简介: 该论文属于公平性审计任务，旨在检测教育领域大语言模型中的性别偏见。通过构建反事实案例，分析模型对性别线索的响应差异，揭示其反馈中的性别偏差。**

- **链接: [https://arxiv.org/pdf/2511.08225](https://arxiv.org/pdf/2511.08225)**

> **作者:** Yishan Du; Conrad Borchers; Mutlu Cukurova
>
> **备注:** 21 pages, 7 figures
>
> **摘要:** As teachers increasingly turn to GenAI in their educational practice, we need robust methods to benchmark large language models (LLMs) for pedagogical purposes. This article presents an embedding-based benchmarking framework to detect bias in LLMs in the context of formative feedback. Using 600 authentic student essays from the AES 2.0 corpus, we constructed controlled counterfactuals along two dimensions: (i) implicit cues via lexicon-based swaps of gendered terms within essays, and (ii) explicit cues via gendered author background in the prompt. We investigated six representative LLMs (i.e. GPT-5 mini, GPT-4o mini, DeepSeek-R1, DeepSeek-R1-Qwen, Gemini 2.5 Pro, Llama-3-8B). We first quantified the response divergence with cosine and Euclidean distances over sentence embeddings, then assessed significance via permutation tests, and finally, visualised structure using dimensionality reduction. In all models, implicit manipulations reliably induced larger semantic shifts for male-female counterfactuals than for female-male. Only the GPT and Llama models showed sensitivity to explicit gender cues. These findings show that even state-of-the-art LLMs exhibit asymmetric semantic responses to gender substitutions, suggesting persistent gender biases in feedback they provide learners. Qualitative analyses further revealed consistent linguistic differences (e.g., more autonomy-supportive feedback under male cues vs. more controlling feedback under female cues). We discuss implications for fairness auditing of pedagogical GenAI, propose reporting standards for counterfactual evaluation in learning analytics, and outline practical guidance for prompt design and deployment to safeguard equitable feedback.
>
---
#### [replaced 005] Just as Humans Need Vaccines, So Do Models: Model Immunization to Combat Falsehoods
- **分类: cs.CL**

- **简介: 该论文属于虚假信息检测任务，旨在解决大语言模型传播错误信息的问题。通过监督微调注入虚假与纠正对，提升模型辨别能力。**

- **链接: [https://arxiv.org/pdf/2505.17870](https://arxiv.org/pdf/2505.17870)**

> **作者:** Shaina Raza; Rizwan Qureshi; Azib Farooq; Marcelo Lotif; Aman Chadha; Deval Pandya; Christos Emmanouilidis
>
> **摘要:** Large language models (LLMs) reproduce misinformation not by memorizing false facts alone, but by learning the linguistic patterns that make falsehoods persuasive, such as hedging, false presuppositions, and fabricated citations. We propose model immunization, a training paradigm based on supervised fine-tuning over curated (false claim, correction) pairs, injected as small vaccine doses (5 to 10% of tokens) alongside truthful data. Unlike post-hoc filtering or preference-based alignment, immunization introduces direct negative supervision on labeled falsehoods. Across four open weight model families, this approach improves TruthfulQA accuracy by 12 points and increases misinformation rejection rates by 30 points, while preserving overall model capability. We further outline key design requirements, including dosage, labeling, quarantine, and diversity and advocate for standardized vaccine corpora and benchmarks to evaluate generalization. These findings position immunization as a practical and scalable component of responsible LLM development.
>
---
#### [replaced 006] MemFactory: Unified Inference & Training Framework for Agent Memory
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出MemFactory框架，解决记忆增强型AI代理的统一训练与推理问题。通过模块化设计和GRPO优化，提升记忆操作效率，支持多种先进模型，显著提高性能。**

- **链接: [https://arxiv.org/pdf/2603.29493](https://arxiv.org/pdf/2603.29493)**

> **作者:** Ziliang Guo; Ziheng Li; Bo Tang; Feiyu Xiong; Zhiyu Li
>
> **备注:** 10 pages, Code: this https URL
>
> **摘要:** Memory-augmented Large Language Models (LLMs) are essential for developing capable, long-term AI agents. Recently, applying Reinforcement Learning (RL) to optimize memory operations, such as extraction, updating, and retrieval, has emerged as a highly promising research direction. However, existing implementations remain highly fragmented and task-specific, lacking a unified infrastructure to streamline the integration, training, and evaluation of these complex pipelines. To address this gap, we present MemFactory, the first unified, highly modular training and inference framework specifically designed for memory-augmented agents. Inspired by the success of unified fine-tuning frameworks like LLaMA-Factory, MemFactory abstracts the memory lifecycle into atomic, plug-and-play components, enabling researchers to seamlessly construct custom memory agents via a "Lego-like" architecture. Furthermore, the framework natively integrates Group Relative Policy Optimization (GRPO) to fine-tune internal memory management policies driven by multi-dimensional environmental rewards. MemFactory provides out-of-the-box support for recent cutting-edge paradigms, including Memory-R1, RMM, and MemAgent. We empirically validate MemFactory on the open-source MemAgent architecture using its publicly available training and evaluation data. Across both in-domain and out-of-distribution evaluation sets, MemFactory consistently improves performance over the corresponding base models, with relative gains of up to 14.8%. By providing a standardized, extensible, and easy-to-use infrastructure, MemFactory significantly lowers the barrier to entry, paving the way for future innovations in memory-driven AI agents.
>
---
#### [replaced 007] Let the Model Distribute Its Doubt: Confidence Estimation through Verbalized Probability Distribution
- **分类: cs.CL**

- **简介: 该论文属于模型置信度估计任务，旨在提升大语言模型响应的可靠性。通过生成概率分布增强推理，提高置信度评估效果。**

- **链接: [https://arxiv.org/pdf/2511.14275](https://arxiv.org/pdf/2511.14275)**

> **作者:** Ante Wang; Weizhi Ma; Yang Liu
>
> **摘要:** Knowing the reliability of a model's response is essential in practical applications. Given the strong generation capabilities of large language models (LLMs), research has focused on generating verbalized confidence. This approach is further enhanced by integrating chain-of-thought reasoning, which provides logical and transparent estimates. However, how reasoning strategies affect the estimated confidence remains under-explored. In this work, we demonstrate that predicting a verbalized probability distribution effectively promotes reasoning for confidence estimation. It requires an LLM to consider all possible answers rather than relying on a single guess, and the requirement of producing a distribution elicits more careful confidence assignment. We conduct systematic experiments comparing different verbalization-based methods across multiple LLMs and tasks. Our method consistently shows advantages, whether in the simple prompting setup or after optimization via reinforcement learning (RL). Notably, it achieves higher reasoning efficacy during inference-time scaling, saving nearly 6$\times$ the computation to reach the best Brier score of the strongest baseline on MMLU-Pro. Additionally, we reveal its limitations on specific tasks and discuss possible solutions for broader applicability.
>
---
#### [replaced 008] Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG
- **分类: cs.AI; cs.CL; cs.IR**

- **简介: 本文探讨Agentic RAG系统，属于自然语言处理任务，旨在解决传统RAG系统适应性差的问题。通过引入自主AI代理，提升系统动态处理和复杂任务能力。**

- **链接: [https://arxiv.org/pdf/2501.09136](https://arxiv.org/pdf/2501.09136)**

> **作者:** Aditi Singh; Abul Ehtesham; Saket Kumar; Tala Talaei Khoei; Athanasios V. Vasilakos
>
> **摘要:** Large Language Models (LLMs) have advanced artificial intelligence by enabling human-like text generation and natural language understanding. However, their reliance on static training data limits their ability to respond to dynamic, real-time queries, resulting in outdated or inaccurate outputs. Retrieval-Augmented Generation (RAG) has emerged as a solution, enhancing LLMs by integrating real-time data retrieval to provide contextually relevant and up-to-date responses. Despite its promise, traditional RAG systems are constrained by static workflows and lack the adaptability required for multi-step reasoning and complex task management. Agentic Retrieval-Augmented Generation (Agentic RAG) transcends these limitations by embedding autonomous AI agents into the RAG pipeline. These agents leverage agentic design patterns reflection, planning, tool use, and multi-agent collaboration to dynamically manage retrieval strategies, iteratively refine contextual understanding, and adapt workflows through operational structures ranging from sequential steps to adaptive collaboration. This integration enables Agentic RAG systems to deliver flexibility, scalability, and context-awareness across diverse applications. This paper presents an analytical survey of Agentic RAG systems. It traces the evolution of RAG paradigms, introduces a principled taxonomy of Agentic RAG architectures based on agent cardinality, control structure, autonomy, and knowledge representation, and provides a comparative analysis of design trade-offs across existing frameworks. The survey examines applications in healthcare, finance, education, and enterprise document processing, and distills practical lessons for system designers and practitioners. Finally, it identifies key open research challenges related to evaluation, coordination, memory management, efficiency, and governance, outlining directions for future research.
>
---
#### [replaced 009] Dive into the Agent Matrix: A Realistic Evaluation of Self-Replication Risk in LLM Agents
- **分类: cs.AI; cs.CL; cs.LG; cs.MA**

- **简介: 该论文属于安全评估任务，旨在解决LLM代理的自我复制风险问题。通过构建评估框架和指标，分析代理在真实场景下的自复制倾向，提出风险量化方法。**

- **链接: [https://arxiv.org/pdf/2509.25302](https://arxiv.org/pdf/2509.25302)**

> **作者:** Boxuan Zhang; Yi Yu; Jiaxuan Guo; Jing Shao
>
> **备注:** 26 pages, 6 figures
>
> **摘要:** The prevalent deployment of Large Language Model agents such as OpenClaw unlocks potential in real-world applications, while amplifying safety concerns. Among these concerns, the self-replication risk of LLM agents driven by objective misalignment (just like Agent Smith in the movie The Matrix) has transitioned from a theoretical warning to a pressing reality. Previous studies mainly examine whether LLM agents can self-replicate when directly instructed, potentially overlooking the risk of spontaneous replication driven by real-world settings (e.g., ensuring survival against termination threats). In this paper, we present a comprehensive evaluation framework for quantifying self-replication risks. Our framework establishes authentic production environments and realistic tasks (e.g., dynamic load balancing) to enable scenario-driven assessment of agent behaviors. Designing tasks that might induce misalignment between users' and agents' objectives makes it possible to decouple replication success from risk and capture self-replication risks arising from these misalignment settings. We further introduce Overuse Rate ($\mathrm{OR}$) and Aggregate Overuse Count ($\mathrm{AOC}$) metrics, which precisely capture the frequency and severity of uncontrolled replication. In our evaluation of 21 state-of-the-art open-source and proprietary models, we observe that over 50\% of LLM agents display a pronounced tendency toward uncontrolled self-replication under operational pressures. Our results underscore the urgent need for scenario-driven risk assessment and robust safeguards in the practical deployment of LLM-based agents.
>
---
#### [replaced 010] EVM-QuestBench: An Execution-Grounded Benchmark for Natural-Language Transaction Code Generation
- **分类: cs.CL**

- **简介: 该论文提出EVM-QuestBench，用于评估自然语言到交易代码的生成。解决链上交易中因错误导致损失的问题，通过动态执行验证确保准确性与安全性。**

- **链接: [https://arxiv.org/pdf/2601.06565](https://arxiv.org/pdf/2601.06565)**

> **作者:** Pei Yang; Wanyi Chen; Ke Wang; Lynn Ai; Eric Yang; Tianyu Shi
>
> **备注:** 10 pages, 13 figures
>
> **摘要:** Large language models are increasingly applied to various development scenarios. However, in on-chain transaction scenarios, even a minor error can cause irreversible loss for users. Existing evaluations often overlook execution accuracy and safety. We introduce EVM-QuestBench, an execution-grounded benchmark for natural-language transaction-script generation on EVM-compatible chains. The benchmark employs dynamic evaluation: instructions are sampled from template pools, numeric parameters are drawn from predefined intervals, and validators verify outcomes against these instantiated values. EVM-QuestBench contains 107 tasks (62 atomic, 45 composite). Its modular architecture enables rapid task development. The runner executes scripts on a forked EVM chain with snapshot isolation; composite tasks apply step-efficiency decay. We evaluate 20 models and find large performance gaps, with split scores revealing persistent asymmetry between single-action precision and multi-step workflow completion. Code: this https URL.
>
---
#### [replaced 011] PETra: A Multilingual Corpus of Pragmatic Explicitation in Translation
- **分类: cs.CL**

- **简介: 该论文提出PragExTra，首个多语言语料库和检测框架，用于研究翻译中的语用显化现象。旨在解决跨语言文化信息显化的计算建模问题，通过实体描述等方法提升机器翻译的文化敏感性。**

- **链接: [https://arxiv.org/pdf/2511.02721](https://arxiv.org/pdf/2511.02721)**

> **作者:** Doreen Osmelak; Koel Dutta Chowdhury; Uliana Sentsova; Cristina España-Bonet; Josef van Genabith
>
> **摘要:** Translators often enrich texts with background details that make implicit cultural meanings explicit for new audiences. This phenomenon, known as pragmatic explicitation, has been widely discussed in translation theory but rarely modeled computationally. We introduce PragExTra, the first multilingual corpus and detection framework for pragmatic explicitation. The corpus covers eight language pairs from TED-Multi and Europarl and includes additions such as entity descriptions, measurement conversions, and translator remarks. We identify candidate explicitation cases through null alignments and refined using active learning with human annotation. Our results show that entity and system-level explicitations are most frequent, and that active learning improves classifier accuracy by 7-8 percentage points, achieving up to 0.88 accuracy and 0.82 F1 across languages. PragExTra establishes pragmatic explicitation as a measurable, cross-linguistic phenomenon and takes a step towards building culturally aware machine translation. Keywords: translation, multilingualism, explicitation
>
---
#### [replaced 012] Counting on Consensus: Selecting the Right Inter-annotator Agreement Metric for NLP Annotation and Evaluation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的标注与评估任务，旨在解决如何选择合适的标注者间一致性度量问题。论文分析了不同任务下的度量方法及其限制，提出了最佳实践建议。**

- **链接: [https://arxiv.org/pdf/2603.06865](https://arxiv.org/pdf/2603.06865)**

> **作者:** Joseph James
>
> **备注:** Accepted LREC 2026
>
> **摘要:** Human annotation remains the foundation of reliable and interpretable data in Natural Language Processing (NLP). As annotation and evaluation tasks continue to expand, from categorical labelling to segmentation, subjective judgment, and continuous rating, measuring agreement between annotators has become increasingly more complex. This paper outlines how inter-annotator agreement (IAA) has been conceptualised and applied across NLP and related disciplines, describing the assumptions and limitations of common approaches. We organise agreement measures by task type and discuss how factors such as label imbalance and missing data influence reliability estimates. In addition, we highlight best practices for clear and transparent reporting, including the use of confidence intervals and the analysis of disagreement patterns. The paper aims to serve as a guide for selecting and interpreting agreement measures, promoting more consistent and reproducible human annotation and evaluation in NLP.
>
---
#### [replaced 013] Children's Intelligence Tests Pose Challenges for MLLMs? KidGym: A 2D Grid-Based Reasoning Benchmark for MLLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出KidGym，一个用于评估MLLMs五种核心能力的2D网格基准，解决模型在儿童智力测试中的适应性与成长性问题。**

- **链接: [https://arxiv.org/pdf/2603.20209](https://arxiv.org/pdf/2603.20209)**

> **作者:** Hengwei Ye; Yuanting Guan; Yuxuan Ge; Tianying Zhu; Zhenhan Guan; Yijia Zhong; Yijing Zhang; Han Zhang; Yingna Wu; Zheng Tian
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** Multimodal Large Language Models (MLLMs) combine the linguistic strengths of LLMs with the ability to process multimodal data, enbaling them to address a broader range of visual tasks. Because MLLMs aim at more general, human-like competence than language-only models, we take inspiration from the Wechsler Intelligence Scales - an established battery for evaluating children by decomposing intelligence into interpretable, testable abilities. We introduce KidGym, a comprehensive 2D grid-based benchmark for assessing five essential capabilities of MLLMs: Execution, Perception Reasoning, Learning, Memory and Planning. The benchmark comprises 12 unique tasks, each targeting at least one core capability, specifically designed to guage MLLMs' adaptability and developmental potential, mirroring the stages of children's cognitive growth. Additionally, our tasks encompass diverse scenarios and objects with randomly generated layouts, ensuring a more accurate and robust evluation of MLLM capabilities. KidGym is designed to be fully user-customizable and extensible, allowing researchers to create new evaluation scenarios and adjust difficuly levels to accommodate the rapidly growing MLLM community. Through the evaluation of state-of-the-art MLLMs using KidGym, we identified significant insights into model capabilities and revealed several limitations of current models. We release our benchmark at: this https URL.
>
---
#### [replaced 014] AgentExpt: Automating AI Experiment Design with LLM-based Resource Retrieval Agent
- **分类: cs.CL**

- **简介: 该论文属于AI实验设计自动化任务，旨在解决数据和基线推荐覆盖不足与相似性偏差问题。通过构建数据集和增强检索方法提升推荐效果。**

- **链接: [https://arxiv.org/pdf/2511.04921](https://arxiv.org/pdf/2511.04921)**

> **作者:** Yu Li; Lehui Li; Lin Chen; Qingmin Liao; Fengli Xu; Yong Li
>
> **备注:** 10 pages
>
> **摘要:** Large language model agents are becoming increasingly capable at web-centric tasks such as information retrieval, complex reasoning. These emerging capabilities have given rise to surge research interests in developing LLM agent for facilitating scientific quest. One key application in AI research is to automate experiment design through agentic dataset and baseline retrieval. However, prior efforts suffer from limited data coverage, as recommendation datasets primarily harvest candidates from public portals and omit many datasets actually used in published papers, and from an overreliance on content similarity that biases model toward superficial similarity and overlooks experimental suitability. Harnessing collective perception embedded in the baseline and dataset citation network, we present a comprehensive framework for baseline and dataset recommendation. First, we design an automated data-collection pipeline that links roughly one hundred thousand accepted papers to the baselines and datasets they actually used. Second, we propose a collective perception enhanced retriever. To represent the position of each dataset or baseline within the scholarly network, it concatenates self-descriptions with aggregated citation contexts. To achieve efficient candidate recall, we finetune an embedding model on these representations. Finally, we develop a reasoning-augmented reranker that exact interaction chains to construct explicit reasoning chains and finetunes a large language model to produce interpretable justifications and refined rankings. The dataset we curated covers 85\% of the datasets and baselines used at top AI conferences over the past five years. On our dataset, the proposed method outperforms the strongest prior baseline with average gains of +5.85\% in Recall@20, +8.30\% in HitRate@5. Taken together, our results advance reliable, interpretable automation of experimental design.
>
---
#### [replaced 015] WAON: Large-Scale Japanese Image-Text Pair Dataset for Improving Model Performance on Japanese Cultural Tasks
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出WAON数据集，解决日本文化任务中高质量图像-文本对数据不足的问题。通过构建大规模、高质量的数据集及评估基准，提升模型在日语文化任务上的性能。**

- **链接: [https://arxiv.org/pdf/2510.22276](https://arxiv.org/pdf/2510.22276)**

> **作者:** Issa Sugiura; Shuhei Kurita; Yusuke Oda; Daisuke Kawahara; Yasuo Okabe; Naoaki Okazaki
>
> **备注:** 14 pages, 7 figures
>
> **摘要:** Contrastive pre-training on large-scale image-text pair datasets has driven major advances in vision-language representation learning. Recent work shows that pretraining on global data followed by language or culture specific fine-tuning is effective for improving performance in target domains. With the availability of strong open-weight multilingual models such as SigLIP2, this paradigm has become increasingly practical. However, for Japanese, the scarcity of large-scale, high-quality image-text pair datasets tailored to Japanese language and cultural content remains a key limitation. To address this gap, we introduce WAON, the largest Japanese image-text pair dataset constructed from Japanese web content in Common Crawl, containing approximately 155 million examples. Our dataset construction pipeline employs filtering and deduplication to improve dataset quality. To improve the quality and reliability of evaluation on Japanese cultural tasks, we also construct WAON-Bench, a manually curated benchmark for Japanese cultural image classification comprising 374 classes, which addresses issues in the existing benchmark such as category imbalance and label-image mismatches. Our experiments demonstrate that fine-tuning on WAON improves model performance on Japanese cultural benchmarks more efficiently than existing datasets, achieving state-of-the-art results among publicly available models of comparable architecture. We release our dataset, model, and code.
>
---
#### [replaced 016] MemeMind: A Large-Scale Multimodal Dataset with Chain-of-Thought Reasoning for Harmful Meme Detection
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于有害表情包检测任务，旨在解决隐含有害内容识别困难的问题。构建了MemeMind数据集并提出MemeGuard框架，提升检测准确性和可解释性。**

- **链接: [https://arxiv.org/pdf/2506.18919](https://arxiv.org/pdf/2506.18919)**

> **作者:** Hexiang Gu; Qifan Yu; Yuan Liu; Zikang Li; Saihui Hou; Jian Zhao; Zhaofeng He
>
> **摘要:** As a multimodal medium combining images and text, memes frequently convey implicit harmful content through metaphors and humor, rendering the detection of harmful memes a complex and challenging task. Although recent studies have made progress in detection accuracy and interpretability, large-scale, high-quality datasets for harmful memes remain scarce, and current methods still struggle to capture implicit risks and nuanced semantics. Thus, we construct MemeMind, a large-scale harmful meme dataset. Aligned with the international standards and the context of internet, MemeMind provides detailed Chain-of-Thought (CoT) reasoning annotations to support fine-grained analysis of implicit intentions in memes. Based on this dataset, we further propose MemeGuard, a reasoning-oriented multimodal detection framework that significantly improves both the accuracy of harmful meme detection and the interpretability of model decisions. Extensive experimental results demonstrate that MemeGuard outperforms existing state-of-the-art methods on the MemeMind dataset, establishing a solid foundation for future research in harmful meme detection. The complete dataset and code will be released upon acceptance.
>
---
#### [replaced 017] How AI Fails: An Interactive Pedagogical Tool for Demonstrating Dialectal Bias in Automated Toxicity Models
- **分类: cs.CL; cs.CY; cs.HC**

- **简介: 该论文属于AI偏见研究任务，旨在解决自动化内容审核中的方言偏差问题。通过基准测试和交互工具，揭示AAE文本被误判为更毒性的现象，提升AI素养。**

- **链接: [https://arxiv.org/pdf/2511.06676](https://arxiv.org/pdf/2511.06676)**

> **作者:** Subhojit Ghimire
>
> **备注:** 9 pages, 5 figures, 4 tables, 14 references
>
> **摘要:** Now that AI-driven moderation has become pervasive in everyday life, we often hear claims that "the AI is biased". While this is often said jokingly, the light-hearted remark reflects a deeper concern. How can we be certain that an online post flagged as "inappropriate" was not simply the victim of a biased algorithm? This paper investigates this problem using a dual approach. First, I conduct a quantitative benchmark of a widely used toxicity model (unitary/toxic-bert) to measure performance disparity between text in African-American English (AAE) and Standard American English (SAE). The benchmark reveals a clear, systematic bias: on average, the model scores AAE text as 1.8 times more toxic and 8.8 times higher for "identity hate". Second, I introduce an interactive pedagogical tool that makes these abstract biases tangible. The tool's core mechanic, a user-controlled "sensitivity threshold," demonstrates that the biased score itself is not the only harm; instead, the more-concerning harm is the human-set, seemingly neutral policy that ultimately operationalises discrimination. This work provides both statistical evidence of disparate impact and a public-facing tool designed to foster critical AI literacy.
>
---
#### [replaced 018] OmniFusion: Simultaneous Multilingual Multimodal Translations via Modular Fusion
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模态翻译任务，旨在解决传统语音翻译流程延迟高、无法利用多模态信息的问题。通过融合多模态基础模型与翻译模型，提出OmniFusion系统，实现高效多模态翻译。**

- **链接: [https://arxiv.org/pdf/2512.00234](https://arxiv.org/pdf/2512.00234)**

> **作者:** Sai Koneru; Matthias Huck; Jan Niehues
>
> **备注:** Revised submission in review for ACL ARR
>
> **摘要:** There has been significant progress in open-source text-only translation large language models (LLMs) with better language coverage and quality. However, these models can be only used in cascaded pipelines for speech translation (ST), performing automatic speech recognition first followed by translation. This introduces additional latency, which is particularly critical in simultaneous ST (SimulST), and prevents the model from exploiting multimodal context, such as images, which can aid disambiguation. Pretrained multimodal foundation models (MMFMs) already possess strong perception and reasoning capabilities across multiple modalities, but generally lack the multilingual coverage and specialized translation performance of dedicated translation LLMs. To build an effective multimodal translation system, we propose an end-to-end approach that fuses MMFMs with translation LLMs. We introduce a novel fusion strategy that connects hidden states from multiple layers of a pretrained MMFM to a translation LLM, enabling joint end-to-end training. The resulting model, OmniFusion, built on Omni 2.5-7B as the MMFM and SeedX PPO-7B as the translation LLM, can perform speech-to-text, speech-and-image-to-text, and text-and-image-to-text translation. Experiments demonstrate that OmniFusion effectively leverages both audio and visual inputs, achieves a 1-second latency reduction in SimulST compared to cascaded pipelines and also improves the overall translation quality\footnote{Code is available at this https URL}.
>
---
#### [replaced 019] Structured Prompts Improve Evaluation of Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理中的模型评估任务，旨在解决单一提示配置导致评估结果不准确的问题。通过引入结构化提示框架，系统性评估不同提示对模型性能的影响。**

- **链接: [https://arxiv.org/pdf/2511.20836](https://arxiv.org/pdf/2511.20836)**

> **作者:** Asad Aali; Muhammad Ahmed Mohsin; Vasiliki Bikia; Arnav Singhvi; Richard Gaus; Suhana Bedi; Hejie Cui; Miguel Fuentes; Alyssa Unell; Yifan Mai; Jordan Cahoon; Michael Pfeffer; Roxana Daneshjou; Sanmi Koyejo; Emily Alsentzer; Christopher Potts; Nigam H. Shah; Akshay S. Chaudhari
>
> **摘要:** As language models (LMs) are increasingly adopted across domains, high-quality benchmarking frameworks are essential for guiding deployment decisions. In practice, however, frameworks such as Holistic Evaluation of Language Models (HELM) typically evaluate models under a single static prompt configuration, even though model behavior depends strongly on prompt choice. As a result, reported scores can reflect prompt choice as much as model capability. Declarative prompting frameworks such as DSPy offer a scalable way to evaluate models under a set of structured prompting strategies rather than a static prompt configuration. We present a reproducible DSPy+HELM framework for studying how prompt choice impacts reported benchmark outcomes. Using five prompting methods, we evaluate four frontier and two open-source LMs across seven benchmarks against existing HELM baseline scores. By evaluating LMs across a family of prompt configurations, we find that prompt choice can materially impact leaderboard outcomes. In particular, structured prompting improves performance (by 6% on average), alters comparisons (leaderboard rankings shift on 5/7 benchmarks), with most gains coming from introducing chain-of-thought, and little additional benefit from more advanced optimizers. To our knowledge, this is the first study to systematically integrate structured prompting into an established evaluation framework and quantify how prompt choice alone can impact benchmark conclusions. We open-source (i) DSPy+HELM Evaluation (this https URL) and (ii) Prompt Optimization Pipeline (this https URL).
>
---
#### [replaced 020] Evaluating Vision-Language and Large Language Models for Automated Student Assessment in Indonesian Classrooms
- **分类: cs.CL**

- **简介: 该论文属于教育评估任务，旨在解决AI在印尼真实课堂中自动评分的问题。通过评估视觉语言和大语言模型，处理手写答案并生成反馈。**

- **链接: [https://arxiv.org/pdf/2506.04822](https://arxiv.org/pdf/2506.04822)**

> **作者:** Nurul Aisyah; Muhammad Dehan Al Kautsar; Arif Hidayat; Raqib Chowdhury; Fajri Koto
>
> **备注:** Accepted at AIED 2026
>
> **摘要:** Despite rapid progress in vision-language and large language models (VLMs and LLMs), their effectiveness for AI-driven educational assessment in real-world, underrepresented classrooms remains largely unexplored. We evaluate state-of-the-art VLMs and LLMs on over 14K handwritten answers from grade-4 classrooms in Indonesia, covering Mathematics and English aligned with the local national curriculum. Unlike prior work on clean digital text, our dataset features naturally curly, diverse handwriting from real classrooms, posing realistic visual and linguistic challenges. Assessment tasks include grading and generating personalized Indonesian feedback guided by rubric-based evaluation. Results show that the VLM struggles with handwriting recognition, causing error propagation in LLM grading, yet LLM feedback remains pedagogically useful despite imperfect visual inputs, revealing limits in personalization and contextual relevance.
>
---
#### [replaced 021] Replacing Multi-Step Assembly of Data Preparation Pipelines with One-Step LLM Pipeline Generation for Table QA
- **分类: cs.DB; cs.CL**

- **简介: 该论文属于表格问答任务，旨在解决多步骤数据处理管道效率低的问题。提出Operation-R1框架，通过单步生成高质量数据管道，提升效率并降低成本。**

- **链接: [https://arxiv.org/pdf/2602.22721](https://arxiv.org/pdf/2602.22721)**

> **作者:** Fengyu Li; Junhao Zhu; Kaishi Song; Lu Chen; Zhongming Yao; Tianyi Li; Christian S. Jensen
>
> **摘要:** Table Question Answering (TQA) aims to answer natural language questions over structured tables. Large Language Models (LLMs) enable promising solutions to this problem, with operator-centric solutions that generate table manipulation pipelines in a multi-step manner offering state-of-the-art performance. However, these solutions rely on multiple LLM calls, resulting in prohibitive latencies and computational costs. We propose Operation-R1, the first framework that trains lightweight LLMs (e.g., Qwen-4B/1.7B) via a novel variant of reinforcement learning with verifiable rewards to produce high-quality data-preparation pipelines for TQA in a single inference step. To train such an LLM, we first introduce a self-supervised rewarding mechanism to automatically obtain fine-grained pipeline-wise supervision signals for LLM training. We also propose variance-aware group resampling to mitigate training instability. To further enhance robustness of pipeline generation, we develop two complementary mechanisms: operation merge, which filters spurious operations through multi-candidate consensus, and adaptive rollback, which offers runtime protection against information loss in data transformation. Experiments on two benchmark datasets show that, with the same LLM backbone, Operation-R1 achieves average absolute accuracy gains of 8.83 and 4.44 percentage points over multi-step preparation baselines, with 79\% table compression and a 2.2$\times$ reduction in monetary cost.
>
---
#### [replaced 022] Closing the Confidence-Faithfulness Gap in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型校准任务，旨在解决模型自信与实际准确率不匹配的问题。通过分析发现自信信号与校准信息正交，提出两阶段调节方法提升校准效果。**

- **链接: [https://arxiv.org/pdf/2603.25052](https://arxiv.org/pdf/2603.25052)**

> **作者:** Miranda Muqing Miao; Lyle Ungar
>
> **摘要:** Large language models (LLMs) tend to verbalize confidence scores that are largely detached from their actual accuracy, yet the geometric relationship governing this behavior remain poorly understood. In this work, we present a mechanistic interpretability analysis of verbalized confidence, using linear probes and contrastive activation addition (CAA) steering to show that calibration and verbalized confidence signals are encoded linearly but are orthogonal to one another -- a finding consistent across three open-weight models and four datasets. Interestingly, when models are prompted to simultaneously reason through a problem and verbalize a confidence score, the reasoning process disrupts the verbalized confidence direction, exacerbating miscalibration. We term this the "Reasoning Contamination Effect." Leveraging this insight, we introduce a two-stage adaptive steering pipeline that reads the model's internal accuracy estimate and steers verbalized output to match it, substantially improving calibration alignment across all evaluated models.
>
---
#### [replaced 023] Klear-Reasoner: Advancing Reasoning Capability via Gradient-Preserving Clipping Policy Optimization
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出Klear-Reasoner，解决推理模型训练细节不透明和强化学习中剪切机制问题，通过优化策略提升推理能力。**

- **链接: [https://arxiv.org/pdf/2508.07629](https://arxiv.org/pdf/2508.07629)**

> **作者:** Zhenpeng Su; Leiyu Pan; Xue Bai; Dening Liu; Guanting Dong; Jiaming Huang; Minxuan Lv; Wenping Hu; Fuzheng Zhang; Kun Gai; Guorui Zhou
>
> **摘要:** We present Klear-Reasoner, a model with long reasoning capabilities that demonstrates careful deliberation during problem solving, achieving outstanding performance across multiple benchmarks. Although there are already many excellent works related to inference models in the current community, there are still many problems with reproducing high-performance inference models due to incomplete disclosure of training details. This report provides an in-depth analysis of the reasoning model, covering the entire post-training workflow from data preparation and long Chain-of-Thought supervised fine-tuning (long CoT SFT) to reinforcement learning (RL), along with detailed ablation studies for each experimental component. For SFT data, our experiments show that a small number of high-quality data sources are more effective than a large number of diverse data sources, and that difficult samples can achieve better results without accuracy filtering. In addition, we investigate two key issues with current clipping mechanisms in RL: Clipping suppresses critical exploration signals and ignores suboptimal trajectories. To address these challenges, we propose Gradient-Preserving clipping Policy Optimization (GPPO) that gently backpropagates gradients from clipped tokens. GPPO not only enhances the model's exploration capacity but also improves its efficiency in learning from negative samples. Klear-Reasoner exhibits exceptional reasoning abilities in mathematics and programming, scoring 90.5% on AIME 2024, 83.2% on AIME 2025, 66.0% on LiveCodeBench V5 and 58.1% on LiveCodeBench V6.
>
---
#### [replaced 024] Code Comprehension then Auditing for Unsupervised LLM Evaluation
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出CoCoA框架，解决无监督代码正确性评估问题。通过先理解代码功能再进行审计，提升评估准确性。**

- **链接: [https://arxiv.org/pdf/2410.03131](https://arxiv.org/pdf/2410.03131)**

> **作者:** Bhrij Patel; Souradip Chakraborty; Mengdi Wang; Dinesh Manocha; Amrit Singh Bedi
>
> **备注:** 19 pages
>
> **摘要:** Large Language Models (LLMs) for unsupervised code correctness evaluation have recently gained attention because they can judge if code runs as intended without requiring reference implementations or unit tests, which may be unavailable, sparse, or unreliable. However, most prior approaches condition LLM evaluators directly on the full code implementation, forcing the model to jointly infer program behavior and evaluate correctness in a single step. This entanglement leads to misinterpretations of code behavior and unreliable judgments. To mitigate this issue, we introduce CoCoA, an unsupervised Code Comprehension then Auditing framework that first comprehends functionality to generate a natural-language explanation. Then it evaluates task alignment based on this explanation. By sequentially sampling comprehension before evaluation, CoCoA improves the quality of inferred program behavior and enables the evaluator to focus on behavioral alignment rather than raw implementation details. Across multiple datasets, programming languages, and models, CoCoA achieves up to $68\%$ increased F1 score and up to $20\%$ increased accuracy over the best-performing baselines.
>
---
#### [replaced 025] How Does Alignment Enhance LLMs' Multilingual Capabilities? A Language Neurons Perspective
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究多语言对齐如何提升大语言模型的多语言能力，通过分析语言特异性神经元，提出三类神经元分类方法，解析模型内部处理过程。任务为多语言对齐机制分析。**

- **链接: [https://arxiv.org/pdf/2505.21505](https://arxiv.org/pdf/2505.21505)**

> **作者:** Shimao Zhang; Zhejian Lai; Xiang Liu; Shuaijie She; Xiao Liu; Yeyun Gong; Shujian Huang; Jiajun Chen
>
> **备注:** AAAI 2026 (Oral)
>
> **摘要:** Multilingual Alignment is an effective and representative paradigm to enhance LLMs' multilingual capabilities, which transfers the capabilities from the high-resource languages to the low-resource languages. Meanwhile, some research on language-specific neurons provides a new perspective to analyze and understand LLMs' mechanisms. However, we find that there are many neurons that are shared by multiple but not all languages and cannot be correctly classified. In this work, we propose a ternary classification methodology that categorizes neurons into three types, including language-specific neurons, language-related neurons, and general neurons. And we propose a corresponding identification algorithm to distinguish these different types of neurons. Furthermore, based on the distributional characteristics of different types of neurons, we divide the LLMs' internal process for multilingual inference into four parts: (1) multilingual understanding, (2) shared semantic space reasoning, (3) multilingual output space transformation, and (4) vocabulary space outputting. Additionally, we systematically analyze the models before and after alignment with a focus on different types of neurons. We also analyze the phenomenon of "Spontaneous Multilingual Alignment". Overall, our work conducts a comprehensive investigation based on different types of neurons, providing empirical results and valuable insights to better understand multilingual alignment and multilingual capabilities of LLMs.
>
---
#### [replaced 026] DR-LoRA: Dynamic Rank LoRA for Fine-Tuning Mixture-of-Experts Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于模型微调任务，解决MoE模型中参数分配不均的问题。通过动态调整LoRA秩，提升任务相关专家的参数利用率。**

- **链接: [https://arxiv.org/pdf/2601.04823](https://arxiv.org/pdf/2601.04823)**

> **作者:** Guanzhi Deng; Bo Li; Ronghao Chen; Xiujin Liu; Zhuo Han; Huacan Wang; Lijie Wen; Linqi Song
>
> **摘要:** Mixture-of-Experts (MoE) has become a prominent paradigm for scaling Large Language Models (LLMs). Parameter-efficient fine-tuning methods, such as LoRA, are widely adopted to adapt pretrained MoE LLMs to downstream tasks. However, existing approaches typically assign identical LoRA ranks to all expert modules, ignoring the heterogeneous specialization of pretrained experts. This uniform allocation leads to a resource mismatch: task-relevant experts are under-provisioned, while less relevant ones receive redundant parameters. To address this, we propose DR-LoRA, a Dynamic Rank LoRA framework for fine-tuning pretrained MoE models. Specifically, DR-LoRA initializes all expert LoRA modules with a small active rank and uses an expert saliency score, which combines routing frequency and gradient-based rank importance, to identify which experts would benefit most from additional capacity. It then periodically expands the active ranks of the task-critical expert LoRA, progressively constructing a heterogeneous rank distribution tailored to the target task. Experiments on three MoE models across six tasks show that DR-LoRA consistently outperforms LoRA and other strong baselines, demonstrating that task-adaptive heterogeneous rank allocation is an effective strategy to improve active capacity utilization in MoE fine-tuning.
>
---
#### [replaced 027] Echoes Across Centuries: Phonetic Signatures of Persian Poets
- **分类: cs.CL**

- **简介: 该论文属于文学与语音分析交叉任务，旨在探讨波斯诗歌中的语音特征。通过计算方法分析大量诗歌，揭示语音模式与诗人风格、历史演变的关系。**

- **链接: [https://arxiv.org/pdf/2603.14443](https://arxiv.org/pdf/2603.14443)**

> **作者:** Kourosh Shahnazari; Seyed Moein Ayyoubzadeh; Mohammadali Keshtparvar
>
> **摘要:** This study examines phonetic texture in Persian poetry as a literary-historical phenomenon rather than a by-product of meter or a feature used only for classification. The analysis draws on a large corpus of 1,116,306 mesras from 31,988 poems written by 83 poets, restricted to five major classical meters to enable controlled comparison. Each line is converted into a grapheme-to-phoneme representation and analyzed using six phonetic metrics: hardness, sonority, sibilance, vowel ratio, phoneme entropy, and consonant-cluster ratio. Statistical models estimate poet-level differences while controlling for meter, poetic form, and line length. The results show that although meter and form explain a substantial portion of phonetic variation, they do not eliminate systematic differences between poets. Persian poetic sound therefore appears as conditioned variation within shared prosodic structures rather than as either purely individual style or simple metrical residue. A multidimensional stylistic map reveals several recurrent phonetic profiles, including high-sonority lyric styles, hardness-driven rhetorical or epic styles, sibilant mystical contours, and high-entropy complex textures. Historical analysis indicates that phonetic distributions shift across centuries, reflecting changes in genre prominence, literary institutions, and performance contexts rather than abrupt stylistic breaks. The study establishes a corpus-scale framework for phonetic analysis in Persian poetry and demonstrates how computational phonetics can contribute to literary-historical interpretation while remaining attentive to the formal structures that shape Persian verse.
>
---
#### [replaced 028] MVSS: A Unified Framework for Multi-View Structured Survey Generation
- **分类: cs.CL**

- **简介: 该论文属于科学文献综述生成任务，旨在解决自动生成综述结构不清晰、比较不完整的问题。提出MVSS框架，联合生成结构化树、对比表和文本，提升综述的组织与证据支持。**

- **链接: [https://arxiv.org/pdf/2601.09504](https://arxiv.org/pdf/2601.09504)**

> **作者:** Yinqi Liu; Yueqi Zhu; Yongkang Zhang; Feiran Liu; Yutong Shen; Yufei Sun; Xin Wang; Renzhao Liang; Yidong Wang; Cunxiang Wang
>
> **摘要:** Scientific surveys require not only summarizing large bodies of literature, but also organizing them into clear and coherent conceptual structures. However, existing automatic survey generation methods typically focus on linear text generation and struggle to explicitly model hierarchical relations among research topics and structured methodological comparisons, resulting in substantial gaps in structural organization and evidence presentation compared to expert-written surveys. To address this limitation, we propose MVSS, a multi-view structured survey generation framework that jointly generates and aligns citation-grounded hierarchical trees, structured comparison tables, and survey text. MVSS follows a structure-first paradigm: it first constructs a tree that captures the conceptual organization of a research domain, then generates comparison tables constrained by the tree structure, and finally uses both the tree and tables as joint structural constraints to guide outline construction and survey text generation. This design enables complementary and aligned multi-view representations across structure, comparison, and narrative. In addition, we introduce a dedicated evaluation framework that systematically assesses generated surveys from multiple dimensions, including structural quality, comparative completeness, and citation fidelity. Through large-scale experiments on 76 computer science topics, we demonstrate that MVSS significantly outperforms existing methods in survey organization and evidence grounding, and achieves performance comparable to expert-written surveys across multiple evaluation metrics.
>
---
#### [replaced 029] When Only the Final Text Survives: Implicit Execution Tracing for Multi-Agent Attribution
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于多智能体系统责任归属任务，解决无执行日志时的问责问题。通过嵌入统计信号实现生成文本的自我验证，提升多智能体系统的可追溯性。**

- **链接: [https://arxiv.org/pdf/2603.17445](https://arxiv.org/pdf/2603.17445)**

> **作者:** Yi Nian; Haosen Cao; Shenzhe Zhu; Henry Peng Zou; Qingqing Luan; Yue Zhao
>
> **摘要:** When a multi-agent system produces an incorrect or harmful answer, who is accountable if execution logs and agent identifiers are unavailable? In practice, generated content is often detached from its execution environment due to privacy or system boundaries, leaving the final text as the only auditable artifact. Existing attribution methods rely on full execution traces and thus become ineffective in such metadata-deprived settings. We propose Implicit Execution Tracing (IET), a provenance-by-design framework that shifts attribution from post-hoc inference to built-in instrumentation. Instead of reconstructing hidden trajectories, IET embeds agent-specific, key-conditioned statistical signals directly into the token generation process, transforming the output text into a self-verifying execution record. At inference time, we recover a linearized execution trace from the final text via transition-aware statistical scoring. Experiments across diverse multi-agent coordination settings demonstrate that IET achieves accurate segment-level attribution and reliable transition recovery under identity removal, boundary corruption, and privacy-preserving redaction, while maintaining generation quality. These results show that embedding provenance into generation provides a practical and robust foundation for accountability in multi-agent language systems when execution metadata is unavailable.
>
---
#### [replaced 030] TriageSim: A Conversational Emergency Triage Simulation Framework from Structured Electronic Health Records
- **分类: cs.CL**

- **简介: 该论文提出TriageSim，用于从结构化电子健康记录生成模拟的紧急分诊对话，解决医疗对话生成问题。通过合成数据提升研究可行性，验证其在分诊分类中的应用价值。**

- **链接: [https://arxiv.org/pdf/2603.10035](https://arxiv.org/pdf/2603.10035)**

> **作者:** Dipankar Srirag; Quoc Dung Nguyen; Aditya Joshi; Padmanesan Narasimhan; Salil Kanhere
>
> **备注:** 6 pages, 3 figures, 2 tables
>
> **摘要:** Research in emergency triage is restricted to structured electronic health records (EHR) due to regulatory constraints on nurse-patient interactions. We introduce TriageSim, a simulation framework for generating persona-conditioned triage conversations from structured records. TriageSim enables multi-turn nurse-patient interactions with explicit control over disfluency and decision behaviour, producing a corpus of ~800 synthetic transcripts and corresponding audio. We use a combination of automated analysis for linguistic, behavioural and acoustic fidelity alongside manual evaluation for medical fidelity using a random subset of 50 conversations. The utility of the generated corpus is examined via conversational triage classification. We observe modest agreement for acuity levels across three modalities: generated synthetic text, ASR transcripts, and direct audio inputs. The code, persona schemata and triage policy prompts for TriageSim will be available upon acceptance.
>
---
#### [replaced 031] Mousse: Rectifying the Geometry of Muon with Curvature-Aware Preconditioning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出Mousse优化器，解决深度学习中优化器对曲率适应性不足的问题，通过结合谱方法与二阶预处理，提升训练效率和稳定性。**

- **链接: [https://arxiv.org/pdf/2603.09697](https://arxiv.org/pdf/2603.09697)**

> **作者:** Yechen Zhang; Shuhao Xing; Junhao Huang; Kai Lv; Yunhua Zhou; Xipeng Qiu; Qipeng Guo; Kai Chen
>
> **备注:** 17 pages, 10 figures
>
> **摘要:** Recent advances in spectral optimization, notably Muon, have demonstrated that constraining update steps to the Stiefel manifold can significantly accelerate training and improve generalization. However, Muon implicitly assumes an isotropic optimization landscape, enforcing a uniform spectral update norm across all eigen-directions. We argue that this "egalitarian" constraint is suboptimal for Deep Neural Networks, where the curvature spectrum is known to be highly heavy-tailed and ill-conditioned. In such landscapes, Muon risks amplifying instabilities in high-curvature directions while limiting necessary progress in flat directions. In this work, we propose \textbf{Mousse} (\textbf{M}uon \textbf{O}ptimization \textbf{U}tilizing \textbf{S}hampoo's \textbf{S}tructural \textbf{E}stimation), a novel optimizer that reconciles the structural stability of spectral methods with the geometric adaptivity of second-order preconditioning. Instead of applying Newton-Schulz orthogonalization directly to the momentum matrix, Mousse operates in a whitened coordinate system induced by Kronecker-factored statistics (derived from Shampoo). Mathematically, we formulate Mousse as the solution to a spectral steepest descent problem constrained by an anisotropic trust region, where the optimal update is derived via the polar decomposition of the whitened gradient. Empirical results across language models ranging from 160M to 800M parameters demonstrate that Mousse consistently outperforms Muon, achieving around $\sim$12\% reduction in training steps with negligible computational overhead.
>
---
#### [replaced 032] Graceful Forgetting in Generative Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，解决生成模型微调中的负迁移问题。提出LWF框架，通过选择性遗忘提升微调效果。**

- **链接: [https://arxiv.org/pdf/2505.19715](https://arxiv.org/pdf/2505.19715)**

> **作者:** Chunyang Jiang; Chi-min Chan; Yiyang Cai; Yulong Liu; Wei Xue; Yike Guo
>
> **备注:** 8 pages, 6 figures. EMNLP 2025
>
> **摘要:** Recently, the pretrain-finetune paradigm has become a cornerstone in various deep learning areas. While in general the pre-trained model would promote both effectiveness and efficiency of downstream tasks fine-tuning, studies have shown that not all knowledge acquired during pre-training is beneficial. Some of the knowledge may actually bring detrimental effects to the fine-tuning tasks, which is also known as negative transfer. To address this problem, graceful forgetting has emerged as a promising approach. The core principle of graceful forgetting is to enhance the learning plasticity of the target task by selectively discarding irrelevant knowledge. However, this approach remains underexplored in the context of generative language models, and it is often challenging to migrate existing forgetting algorithms to these models due to architecture incompatibility. To bridge this gap, in this paper we propose a novel framework, Learning With Forgetting (LWF), to achieve graceful forgetting in generative language models. With Fisher Information Matrix weighting the intended parameter updates, LWF computes forgetting confidence to evaluate self-generated knowledge regarding the forgetting task, and consequently, knowledge with high confidence is periodically unlearned during fine-tuning. Our experiments demonstrate that, although thoroughly uncovering the mechanisms of knowledge interaction remains challenging in pre-trained language models, applying graceful forgetting can contribute to enhanced fine-tuning performance.
>
---
#### [replaced 033] Certifiably Robust RAG against Retrieval Corruption
- **分类: cs.LG; cs.CL; cs.CR**

- **简介: 该论文属于自然语言处理任务，针对RAG系统在检索污染攻击下的脆弱性，提出RobustRAG框架，通过隔离聚合策略实现可证明的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2405.15556](https://arxiv.org/pdf/2405.15556)**

> **作者:** Chong Xiang; Tong Wu; Zexuan Zhong; David Wagner; Danqi Chen; Prateek Mittal
>
> **摘要:** Retrieval-augmented generation (RAG) is susceptible to retrieval corruption attacks, where malicious passages injected into retrieval results can lead to inaccurate model responses. We propose RobustRAG, the first defense framework with certifiable robustness against retrieval corruption attacks. The key insight of RobustRAG is an isolate-then-aggregate strategy: we isolate passages into disjoint groups, generate LLM responses based on the concatenated passages from each isolated group, and then securely aggregate these responses for a robust output. To instantiate RobustRAG, we design keyword-based and decoding-based algorithms for securely aggregating unstructured text responses. Notably, RobustRAG achieves certifiable robustness: for certain queries in our evaluation datasets, we can formally certify non-trivial lower bounds on response quality -- even against an adaptive attacker with full knowledge of the defense and the ability to arbitrarily inject a bounded number of malicious passages. We evaluate RobustRAG on the tasks of open-domain question-answering and free-form long text generation and demonstrate its effectiveness across three datasets and three LLMs.
>
---
#### [replaced 034] CDH-Bench: A Commonsense-Driven Hallucination Benchmark for Evaluating Visual Fidelity in Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉语言模型评估任务，旨在解决视觉证据与常识冲突下的幻觉问题。通过构建CDH-Bench基准，评估模型在不同异常类型下的表现。**

- **链接: [https://arxiv.org/pdf/2603.27982](https://arxiv.org/pdf/2603.27982)**

> **作者:** Kesheng Chen; Yamin Hu; Qi Zhou; Zhenqian Zhu; Wenjian Luo
>
> **摘要:** Vision-language models (VLMs) achieve strong performance on many benchmarks, yet a basic reliability question remains underexplored: when visual evidence conflicts with commonsense, do models follow what is shown or what commonsense suggests? A characteristic failure in this setting is that the model overrides visual evidence and outputs the commonsense alternative. We term this phenomenon \textbf{commonsense-driven hallucination} (CDH). To evaluate it, we introduce \textbf{CDH-Bench}, a benchmark designed to create explicit \textbf{visual evidence--commonsense conflicts}. CDH-Bench covers three dimensions: \textit{counting anomalies}, \textit{relational anomalies}, and \textit{attribute anomalies}. We evaluate frontier VLMs under \textit{binary Question Answering (QA)} and \textit{multiple-choice QA}, and report metrics including \textit{Counterfactual Accuracy} (CF-Acc), \textit{Commonsense Accuracy} (CS-Acc), \textit{Counterfactual Accuracy Drop} (CFAD), \textit{Commonsense Collapse Rate} (CCR), and \textit{Relative Prior Dependency} (RPD). Results show that even strong models remain vulnerable to prior-driven normalization under visual evidence--commonsense conflict. CDH-Bench provides a controlled diagnostic of visual fidelity under visual evidence--commonsense conflict.
>
---
#### [replaced 035] Community size rather than grammatical complexity better predicts Large Language Model accuracy in a novel Wug Test
- **分类: cs.CL**

- **简介: 该论文属于语言模型 morphological generalization 任务，研究模型在新词上的表现。旨在探讨模型准确性是否由语言复杂性或社区规模决定。结果表明，模型表现更受数据量影响。**

- **链接: [https://arxiv.org/pdf/2510.12463](https://arxiv.org/pdf/2510.12463)**

> **作者:** Nikoleta Pantelidou; Evelina Leivada; Raquel Montero; Paolo Morosi
>
> **摘要:** The linguistic abilities of Large Language Models are a matter of ongoing debate. This study contributes to this discussion by investigating model performance in a morphological generalization task that involves novel words. Using a multilingual adaptation of the Wug Test, six models were tested across four partially unrelated languages (Catalan, English, Greek, and Spanish) and compared with human speakers. The aim is to determine whether model accuracy approximates human competence and whether it is shaped primarily by linguistic complexity or by the size of the linguistic community, which affects the quantity of available training data. Consistent with previous research, the results show that the models are able to generalize morphological processes to unseen words with human-like accuracy. However, accuracy patterns align more closely with community size and data availability than with structural complexity, refining earlier claims in the literature. In particular, languages with larger speaker communities and stronger digital representation, such as Spanish and English, revealed higher accuracy than less-resourced ones like Catalan and Greek. Overall, our findings suggest that model behavior is mainly driven by the richness of linguistic resources rather than by sensitivity to grammatical complexity, reflecting a form of performance that resembles human linguistic competence only superficially.
>
---
#### [replaced 036] Activation Steering via Generative Causal Mediation
- **分类: cs.CL; cs.CY; cs.HC; cs.LG**

- **简介: 该论文提出GCM方法，用于定位和控制语言模型中分散的行为概念。属于模型可控性任务，解决如何通过干预模型组件来引导长文本输出的问题。**

- **链接: [https://arxiv.org/pdf/2602.16080](https://arxiv.org/pdf/2602.16080)**

> **作者:** Aruna Sankaranarayanan; Amir Zur; Atticus Geiger; Dylan Hadfield-Menell
>
> **摘要:** Where should we intervene in a language model (LM) to localize and control behaviors that are diffused across many tokens of a long-form response? We introduce Generative Causal Mediation (GCM), a procedure for selecting model components (e.g., attention heads) from contrastive long-form responses, to steer such diffuse concepts (e.g., talk in verse vs. talk in prose). In GCM, we first construct a dataset of contrasting behavioral inputs and long-form responses. Then, we quantify how model components mediate the concept and select the strongest mediators for steering. We evaluate GCM on three behaviors--refusal, sycophancy, and style transfer--across three language models. GCM successfully localizes concepts expressed in long-form responses and outperforms correlational probe-based baselines when steering with a sparse set of attention heads. Together, these results demonstrate that GCM provides an effective approach for localizing from and controlling the long-form responses of LMs.
>
---
#### [replaced 037] AlphaResearch: Accelerating New Algorithm Discovery with Language Models
- **分类: cs.CL**

- **简介: 该论文提出AlphaResearch，一种自主研究代理，用于发现新算法。解决开放问题的算法发现任务，通过迭代提出、验证和优化，取得优于现有系统的成果。**

- **链接: [https://arxiv.org/pdf/2511.08522](https://arxiv.org/pdf/2511.08522)**

> **作者:** Zhaojian Yu; Kaiyue Feng; Yilun Zhao; Shilin He; Xiao-Ping Zhang; Arman Cohan
>
> **摘要:** LLMs have made significant progress in complex but easy-to-verify problems, yet they still struggle with discovering the unknown. In this paper, we present \textbf{AlphaResearch}, an autonomous research agent designed to discover new algorithms on open-ended problems by iteratively running the following steps: (1) propose new ideas (2) program to verify (3) optimize the research proposals. To synergize the feasibility and innovation of the discovery process, we construct a novel dual environment by combining the execution-based verifiable reward and reward from simulated real-world peer review environment in AlphaResearch. We construct \textbf{\dataset}, a set of questions that includes an eight open-ended algorithmic problems competition to benchmark AlphaResearch. Experimental results show that AlphaResearch achieves stronger discovery performance than other agentic discovery systems on six open-ended problems. Notably, the algorithm discovered by AlphaResearch on the \emph{``packing circles''} problem achieves the best-of-known performance, surpassing the results of human researchers and strong baselines from recent work (e.g., AlphaEvolve). Additionally, we conduct a comprehensive analysis of the benefits and remaining challenges of autonomous research agent, providing valuable insights for future research.
>
---
#### [replaced 038] Cross-Context Verification: Hierarchical Detection of Benchmark Contamination through Session-Isolated Analysis
- **分类: cs.CL**

- **简介: 该论文针对LLM编码基准的可信度问题，提出Cross-Context Verification方法，通过多会话分析检测污染，解决模型是否真实推理而非记忆的问题。**

- **链接: [https://arxiv.org/pdf/2603.21454](https://arxiv.org/pdf/2603.21454)**

> **作者:** Tae-Eun Song
>
> **备注:** 11 pages, 3 figures, 4 tables
>
> **摘要:** LLM coding benchmarks face a credibility crisis: widespread solution leakage and test quality issues undermine SWE-bench Verified, while existing detection methods--paraphrase consistency, n-gram overlap, perplexity analysis--never directly observe whether a model reasons or recalls. Meanwhile, simply repeating verification degrades accuracy: multi-turn review generates false positives faster than it discovers true errors, suggesting that structural approaches are needed. We introduce Cross-Context Verification (CCV), a black-box method that solves the same benchmark problem in N independent sessions and measures solution diversity, combined with the Hierarchical Cross-Context Architecture (HCCA), a multi-agent analysis framework that prevents confirmation bias through intentional information restriction across specialized analytical roles. On 9 SWE-bench Verified problems (45 trials, Claude Opus 4.6, temperature 0), CCV achieves perfect separation between contaminated and genuine reasoning (Mann-Whitney U=0, p approx 0.012, r = 1.0). Key findings: (1) contamination is binary--models either recall perfectly or not at all; (2) reasoning absence is a perfect discriminator; (3) 33% of prior contamination labels are false positives; (4) HCCA's independent analysis structure discovers contamination-flaw composite cases that single-analyst approaches miss. A pilot experiment extending HCCA to multi-stage verification (Worker to Verifier to Director) yields a negative result--100% sycophantic confirmation--providing further evidence that information restriction, not structural complexity, is the key mechanism. We release all code and data.
>
---
#### [replaced 039] OPERA: Online Data Pruning for Efficient Retrieval Model Adaptation
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于信息检索任务，解决密集检索模型适应中的效率与效果问题。提出OPERA框架，通过数据剪枝提升模型性能，实现更高效训练。**

- **链接: [https://arxiv.org/pdf/2603.17205](https://arxiv.org/pdf/2603.17205)**

> **作者:** Haoyang Fang; Shuai Zhang; Yifei Ma; Hengyi Wang; Cuixiong Hu; Katrin Kirchhoff; Bernie Wang; George Karypis
>
> **摘要:** Domain-specific finetuning is essential for dense retrievers, yet not all training pairs contribute equally to the learning process. We introduce OPERA, a data pruning framework that exploits this heterogeneity to improve both the effectiveness and efficiency of retrieval model adaptation. We first investigate static pruning (SP), which retains only high-similarity query-document pairs, revealing an intrinsic quality-coverage tradeoff: ranking (NDCG) improves while retrieval (Recall) can degrade due to reduced query diversity. To resolve this tradeoff, we propose a two-stage dynamic pruning (DP) strategy that adaptively modulates sampling probabilities at both query and document levels throughout training, prioritizing high-quality examples while maintaining access to the full training set. Evaluations across eight datasets spanning six domains demonstrate the effectiveness of both approaches: SP improves ranking over standard finetuning (NDCG@10 +0.5\%), while DP achieves the strongest performance on both ranking (NDCG@10 +1.9\%) and retrieval (Recall@20 +0.7\%), with an average rank of 1.38 across all methods. These findings scale to Qwen3-Embedding, an LLM-based dense retriever, confirming architecture-agnostic benefits. Notably, DP reaches comparable performance in less than 50\% of the training time required by standard finetuning.
>
---
#### [replaced 040] SWE-CI: Evaluating Agent Capabilities in Maintaining Codebases via Continuous Integration
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文提出SWE-CI基准，用于评估代码生成代理在长期维护中的能力。解决静态修复无法反映实际开发问题，通过持续集成循环评估代码可维护性。**

- **链接: [https://arxiv.org/pdf/2603.03823](https://arxiv.org/pdf/2603.03823)**

> **作者:** Jialong Chen; Xander Xu; Hu Wei; Chuan Chen; Bing Zhao
>
> **摘要:** Large language model (LLM)-powered agents have demonstrated strong capabilities in automating software engineering tasks such as static bug fixing. However, in the real world, the development of mature software is typically predicated on complex requirement changes and long-term feature iterations -- a process that static, one-shot repair paradigms fail to capture. To bridge this gap, we propose SWE-CI, the first repository-level benchmark built upon the Continuous Integration loop, aiming to shift the evaluation paradigm for code generation from static, short-term functional correctness toward dynamic, long-term maintainability. The key insight is simple: Maintainability can be revealed by tracking how functional correctness changes over time. The benchmark comprises 100 tasks, each deriving from a real-world code repository with a development history spanning an average of 233 days and 71 consecutive commits. SWE-CI requires agents to systematically resolve these tasks through dozens of rounds of analysis and coding iterations. SWE-CI provides valuable insights into how well agents can sustain code quality throughout long-term evolution.
>
---
#### [replaced 041] Language Steering for Multilingual In-Context Learning
- **分类: cs.CL**

- **简介: 该论文属于多语言上下文学习任务，旨在解决跨语言迁移问题。通过引入语言向量作为激活偏移，提升模型在不同语言间的泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.02326](https://arxiv.org/pdf/2602.02326)**

> **作者:** Neeraja Kirtane; Kuan-Hao Huang
>
> **摘要:** If large language models operate in a universal semantic space, then switching between languages should require only a simple activation offset. To test this, we take multilingual in-context learning as a case study, where few-shot demonstrations are provided in English but the test query is in a target language. We propose language vectors, computed as the mean activation difference between parallel source and target language examples at a particular layer, and added as an offset to hidden states at inference time to shift the model's internal representations toward the target language. We evaluate our method across three multilingual tasks spanning 19 languages and three models. Our results show consistent improvements on multilingual in-context learning over baselines across all tasks and languages tested, demonstrating that a simple activation offset is sufficient to redirect a model's language mode without any parameter updates. Beyond performance, the vectors encode interpretable linguistic structure, with closely related languages forming tight clusters and vectors transferring across tasks, suggesting that language identity occupies separable and structured directions in a model's activation space.
>
---
#### [replaced 042] What Makes a Good Doctor Response? A Study on Text-Based Telemedicine
- **分类: cs.CL**

- **简介: 该论文研究文本问诊中医生回复质量对患者满意度的影响，属于自然语言处理任务。旨在分析影响患者评分的因素，通过特征提取与分类模型，发现礼貌和委婉表达与好评相关。**

- **链接: [https://arxiv.org/pdf/2602.17194](https://arxiv.org/pdf/2602.17194)**

> **作者:** Adrian Cosma; Cosmin Dumitrache; Emilian Radoi
>
> **备注:** Accepted at CL4Health Workshop @ LREC 2026
>
> **摘要:** Text-based telemedicine has become an increasingly used mode of care, requiring clinicians to deliver medical advice clearly and effectively in writing. As platforms increasingly rely on patient ratings and feedback, clinicians face growing pressure to maintain satisfaction scores, even though these evaluations often reflect communication quality more than clinical accuracy. We analyse patient satisfaction signals in Romanian text-based telemedicine. Using a sample of anonymised text-based telemedicine consultations, we model feedback as a binary outcome, treating thumbs-up responses as positive and grouping negative or absent feedback into the other class. We extract from doctor responses interpretable, predominantly language-agnostic features (e.g., length, structural characteristics, readability proxies), along with Romanian LIWC psycholinguistic features and politeness/hedging markers where available. We train a classifier with a time-based split and perform SHAP-based analyses, which indicate that metadata dominates prediction, functioning as a strong prior, while characteristics of the response text provide a smaller but actionable signal. In subgroup correlation analyses, politeness and hedging are consistently associated with positive patient feedback, whereas lexical diversity shows a negative association.
>
---
#### [replaced 043] Learning to Reason in Structured In-context Environments with Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决LLM在结构化环境中推理能力不足的问题。提出SIE框架，实现可扩展、可泛化的推理环境。**

- **链接: [https://arxiv.org/pdf/2509.23330](https://arxiv.org/pdf/2509.23330)**

> **作者:** Peng Yu; Zeyuan Zhao; Shao Zhang; Luoyi Fu; Xinbing Wang; Ying Wen
>
> **摘要:** Large language models (LLMs) have achieved significant advancements in reasoning capabilities through reinforcement learning (RL) via environmental exploration. As the intrinsic properties of the environment determine the abilities that LLMs can learn, the environment plays a important role in the RL finetuning process. An ideal LLM reasoning environment should possess three core characteristics: scalability, generalizable reasoning, and verifiability. However, existing mathematical and coding environments are difficult to scale due to heavy reliance on expert annotation, while the skills learned in game-based environments are too specialized to generalize. To bridge this gap, we introduce the \textbf{S}tructured \textbf{I}n-context \textbf{E}nvironment (SIE) framework. SIE achieves scalability by automatically constructing reasoning environments from large-scale structured data, where the rich compositional patterns naturally support generalizable reasoning. Moreover, the explicit schemas and reasoning chains in structured data provide a foundation for rule-based verifiability. Experimental results show that SIE framework not only achieves substantial improvements in in-domain structured reasoning, but also enables the learned compositional reasoning skills to generalize effectively to out-of-domain mathematical and logical reasoning tasks. We further explored learning in information-limited partial SIEs and found that LLMs can infer the missing information through exploring the environment, leading to robust reasoning improvements and generalization performance.
>
---
#### [replaced 044] Truncated Step-Level Sampling with Process Rewards for Retrieval-Augmented Reasoning
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于检索增强推理任务，解决强化学习中的信用分配问题。提出SLATE方法，通过截断采样和密集奖励提升推理效果。**

- **链接: [https://arxiv.org/pdf/2602.23440](https://arxiv.org/pdf/2602.23440)**

> **作者:** Chris Samarinas; Haw-Shiuan Chang; Hamed Zamani
>
> **摘要:** Reinforcement learning has emerged as an effective paradigm for training large language models to interleave reasoning with search engine calls. However, existing approaches face a fundamental credit assignment problem: methods like Search-R1 assign a single outcome reward to the entire multi-step trajectory, providing no signal about which reasoning or retrieval decisions were responsible for success or failure. Process-reward methods such as StepSearch introduce step-level supervision but still sample complete trajectories independently, so advantage estimates at any given step are contaminated by the randomness of all other steps. We propose SLATE (Step-Level Advantage estimation for Truncated Exploration), which addresses both problems through two complementary ideas. First, truncated step-level sampling generates k continuations from a shared prefix, isolating all variation to a single decision point. We prove this reduces the variance of advantage estimates by up to a factor of T compared to full-trajectory sampling for T-step trajectories, the first formal variance guarantee for step-level RL in retrieval-augmented reasoning. Second, dense, decomposed process rewards separately evaluate reasoning quality, query quality, and answer correctness on a ternary scale via an LLM judge, providing richer supervision than binary outcome signals or heuristic step-level scores. Experiments on seven QA benchmarks show that SLATE consistently outperforms both sparse-reward and process-reward baselines, achieving a 7.0% relative improvement over Search-R1 on the 7B model and 30.7% on the 3B model. Gains are largest on challenging multi-hop tasks, and ablations confirm that truncated sampling and dense rewards provide complementary benefits.
>
---
