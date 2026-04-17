# 自然语言处理 cs.CL

- **最新发布 116 篇**

- **更新 84 篇**

## 最新发布

#### [new 001] StoryCoder: Narrative Reformulation for Structured Reasoning in LLM Code Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出StoryCoder，用于代码生成任务，通过叙事重构问题描述，提升模型推理与代码质量。**

- **链接: [https://arxiv.org/pdf/2604.14631](https://arxiv.org/pdf/2604.14631)**

> **作者:** Geonhui Jang; Dongyoon Han; YoungJoon Yoo
>
> **备注:** 21 pages, 12 figures. ACL 2026 Main Conference
>
> **摘要:** Effective code generation requires both model capability and a problem representation that carefully structures how models reason and plan. Existing approaches augment reasoning steps or inject specific structure into how models think, but leave scattered problem conditions unchanged. Inspired by the way humans organize fragmented information into coherent explanations, we propose StoryCoder, a narrative reformulation framework that transforms code generation questions into coherent natural language narratives, providing richer contextual structure than simple rephrasings. Each narrative consists of three components: a task overview, constraints, and example test cases, guided by the selected algorithm and genre. Experiments across 11 models on HumanEval, LiveCodeBench, and CodeForces demonstrate consistent improvements, with an average gain of 18.7% in zero-shot pass@10. Beyond accuracy, our analyses reveal that narrative reformulation guides models toward correct algorithmic strategies, reduces implementation errors, and induces a more modular code structure. The analyses further show that these benefits depend on narrative coherence and genre alignment, suggesting that structured problem representation is important for code generation regardless of model scale or architecture. Our code is available at this https URL.
>
---
#### [new 002] MADE: A Living Benchmark for Multi-Label Text Classification with Uncertainty Quantification of Medical Device Adverse Events
- **分类: cs.CL**

- **简介: 该论文提出MADE，一个用于医疗设备不良事件多标签文本分类的动态基准，解决标签不平衡与不确定性量化问题，评估多种模型与方法的性能与可靠性。**

- **链接: [https://arxiv.org/pdf/2604.15203](https://arxiv.org/pdf/2604.15203)**

> **作者:** Raunak Agarwal; Markus Wenzel; Simon Baur; Jonas Zimmer; George Harvey; Jackie Ma
>
> **备注:** Accepted at ACL 2026 Mains
>
> **摘要:** Machine learning in high-stakes domains such as healthcare requires not only strong predictive performance but also reliable uncertainty quantification (UQ) to support human oversight. Multi-label text classification (MLTC) is a central task in this domain, yet remains challenging due to label imbalances, dependencies, and combinatorial complexity. Existing MLTC benchmarks are increasingly saturated and may be affected by training data contamination, making it difficult to distinguish genuine reasoning capabilities from memorization. We introduce MADE, a living MLTC benchmark derived from {m}edical device {ad}verse {e}vent reports and continuously updated with newly published reports to prevent contamination. MADE features a long-tailed distribution of hierarchical labels and enables reproducible evaluation with strict temporal splits. We establish baselines across more than 20 encoder- and decoder-only models under fine-tuning and few-shot settings (instruction-tuned/reasoning variants, local/API-accessible). We systematically assess entropy-/consistency-based and self-verbalized UQ methods. Results show clear trade-offs: smaller discriminatively fine-tuned decoders achieve the strongest head-to-tail accuracy while maintaining competitive UQ; generative fine-tuning delivers the most reliable UQ; large reasoning models improve performance on rare labels yet exhibit surprisingly weak UQ; and self-verbalized confidence is not a reliable proxy for uncertainty. Our work is publicly available at this https URL.
>
---
#### [new 003] Chinese Essay Rhetoric Recognition Using LoRA, In-context Learning and Model Ensemble
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于中文作文修辞识别任务，旨在提升AI在作文评分中的表现。通过LoRA、上下文学习和模型集成方法，有效识别修辞元素，取得最佳成绩。**

- **链接: [https://arxiv.org/pdf/2604.14167](https://arxiv.org/pdf/2604.14167)**

> **作者:** Yuxuan Lai; Xiajing Wang; Chen Zheng
>
> **备注:** Accepted by CCL2025
>
> **摘要:** Rhetoric recognition is a critical component in automated essay scoring. By identifying rhetorical elements in student writing, AI systems can better assess linguistic and higher-order thinking skills, making it an essential task in the area of AI for education. In this paper, we leverage Large Language Models (LLMs) for the Chinese rhetoric recognition task. Specifically, we explore Low-Rank Adaptation (LoRA) based fine-tuning and in-context learning to integrate rhetoric knowledge into LLMs. We formulate the outputs as JSON to obtain structural outputs and translate keys to Chinese. To further enhance the performance, we also investigate several model ensemble methods. Our method achieves the best performance on all three tracks of CCL 2025 Chinese essay rhetoric recognition evaluation task, winning the first prize.
>
---
#### [new 004] Modeling LLM Unlearning as an Asymmetric Two-Task Learning Problem
- **分类: cs.CL**

- **简介: 该论文属于机器学习中的模型遗忘任务，旨在去除特定知识同时保留通用能力。通过构建非对称双任务框架，提出SAGO方法有效平衡遗忘与保留。**

- **链接: [https://arxiv.org/pdf/2604.14808](https://arxiv.org/pdf/2604.14808)**

> **作者:** Zeguan Xiao; Siqing Li; Yong Wang; Xuetao Wei; Jian Yang; Yun Chen; Guanhua Chen
>
> **备注:** ACL 2026
>
> **摘要:** Machine unlearning for large language models (LLMs) aims to remove targeted knowledge while preserving general capability. In this paper, we recast LLM unlearning as an asymmetric two-task problem: retention is the primary objective and forgetting is an auxiliary. From this perspective, we propose a retention-prioritized gradient synthesis framework that decouples task-specific gradient extraction from conflict-aware combination. Instantiating the framework, we adapt established PCGrad to resolve gradient conflicts, and introduce SAGO, a novel retention-prioritized gradient synthesis method. Theoretically, both variants ensure non-negative cosine similarity with the retain gradient, while SAGO achieves strictly tighter alignment through constructive sign-constrained synthesis. Empirically, on WMDP Bio/Cyber and RWKU benchmarks, SAGO consistently pushes the Pareto frontier: e.g., on WMDP Bio (SimNPO+GD), recovery of target model MMLU performance progresses from 44.6% (naive) to 94.0% (+PCGrad) and further to 96.0% (+SAGO), while maintaining comparable forgetting strength. Our results show that re-shaping gradient geometry, rather than re-balancing losses, is the key to mitigating unlearning-retention trade-offs.
>
---
#### [new 005] Benchmarking Linguistic Adaptation in Comparable-Sized LLMs: A Study of Llama-3.1-8B, Mistral-7B-v0.1, and Qwen3-8B on Romanized Nepali
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言适应任务，旨在解决罗马化尼泊尔语在大语言模型中的资源不足问题。通过基准测试三个模型的零样本和微调表现，评估其在该语言上的适应能力。**

- **链接: [https://arxiv.org/pdf/2604.14171](https://arxiv.org/pdf/2604.14171)**

> **作者:** Ananda Rimal; Adarsha Rimal
>
> **备注:** 31 pages, 4 figures, 14 tables
>
> **摘要:** Romanized Nepali, the Nepali language written in the Latin alphabet, is the dominant medium for informal digital communication in Nepal, yet it remains critically underresourced in the landscape of Large Language Models (LLMs). This study presents a systematic benchmarking of linguistic adaptation across three comparable-sized open-weight models: Llama-3.1-8B, Mistral-7B-v0.1, and Qwen3-8B. We evaluate these architectures under zero-shot and fine-tuned settings using a curated bilingual dataset of 10,000 transliterated instruction-following samples. Performance is quantified across five metrics spanning seven measurement dimensions: Perplexity (PPL), BERTScore, chrF++, ROUGE-1, ROUGE-2, ROUGE-L, and BLEU, capturing fluency, phonetic consistency, and semantic integrity. Models were fine-tuned using Quantized Low-Rank Adaptation (QLoRA) with Rank-Stabilized LoRA (rsLoRA) at rank r=32 on dual NVIDIA Tesla T4 GPUs, training only approximately 1% of each model's parameters in under 27 total GPU-hours. At zero-shot, all three models fail to generate Romanized Nepali, each exhibiting a distinct architecture-specific failure mode. Following fine-tuning, all three resolve these failures and converge to BERTScore approximately 0.75 and chrF++ greater than 23. Overall dimension-wise assessment across ten criteria identifies Qwen3-8B as the overall recommended architecture, being the only model to produce semantically relevant zero-shot output and leading all structural alignment metrics post-SFT. The adaptation headroom hypothesis is confirmed: Llama-3.1-8B, despite its weakest zero-shot baseline, achieves the largest absolute fine-tuning gains in PPL (Delta = -49.77) and BERTScore (Delta = +0.3287), making it the preferred choice for iterative low-resource development pipelines. This work establishes the first rigorous baseline for Romanized Nepali adaptation in comparable-sized open-weight LLMs.
>
---
#### [new 006] CoPA: Benchmarking Personalized Question Answering with Data-Informed Cognitive Factors
- **分类: cs.CL**

- **简介: 该论文属于个性化问答任务，旨在解决个性化评估瓶颈。通过分析用户偏好差异，提出CoPA基准，用于更精准地评估模型与用户认知偏好的一致性。**

- **链接: [https://arxiv.org/pdf/2604.14773](https://arxiv.org/pdf/2604.14773)**

> **作者:** Hang Su; Zequn Liu; Chen Hu; Xuesong Lu; Yingce Xia; Zhen Liu
>
> **备注:** Accepted to ACL. 30 pages, 10 figures
>
> **摘要:** While LLMs have demonstrated remarkable potential in Question Answering (QA), evaluating personalization remains a critical bottleneck. Existing paradigms predominantly rely on lexical-level similarity or manual heuristics, often lacking sufficient data-driven validation. We address this by mining Community-Individual Preference Divergence (CIPD), where individual choices override consensus, to distill six key personalization factors as evaluative dimensions. Accordingly, we introduce CoPA, a benchmark with 1,985 user profiles for fine-grained, factor-level assessment. By quantifying the alignment between model outputs and user-specific cognitive preferences inferred from interaction patterns, CoPA provides a more comprehensive and discriminative standard for evaluating personalized QA than generic metrics. The code is available at this https URL.
>
---
#### [new 007] Decoupling Scores and Text: The Politeness Principle in Peer Review
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决作者难以准确解读审稿意见的问题。通过对比评分与文本的预测效果，发现文本信息可靠性较低，并揭示了礼貌原则的影响。**

- **链接: [https://arxiv.org/pdf/2604.14162](https://arxiv.org/pdf/2604.14162)**

> **作者:** Yingxuan Wen
>
> **摘要:** Authors often struggle to interpret peer review feedback, deriving false hope from polite comments or feeling confused by specific low scores. To investigate this, we construct a dataset of over 30,000 ICLR 2021-2025 submissions and compare acceptance prediction performance using numerical scores versus text reviews. Our experiments reveal a significant performance gap: score-based models achieve 91% accuracy, while text-based models reach only 81% even with large language models, indicating that textual information is considerably less reliable. To explain this phenomenon, we first analyze the 9% of samples that score-based models fail to predict, finding their score distributions exhibit high kurtosis and negative skewness, which suggests that individual low scores play a decisive role in rejection even when the average score falls near the borderline. We then examine why text-based accuracy significantly lags behind scores from a review sentiment perspective, revealing the prevalence of the Politeness Principle: reviews of rejected papers still contain more positive than negative sentiment words, masking the true rejection signal and making it difficult for authors to judge outcomes from text alone.
>
---
#### [new 008] NLP needs Diversity outside of 'Diversity'
- **分类: cs.CL**

- **简介: 该论文属于社会与伦理研究任务，探讨NLP领域内多样性不足的问题，指出当前关注点过于集中于公平性，忽视其他重要领域，分析原因并提出改进措施。**

- **链接: [https://arxiv.org/pdf/2604.14595](https://arxiv.org/pdf/2604.14595)**

> **作者:** Joshua Tint
>
> **备注:** 7 pages, 1 figure
>
> **摘要:** This position paper argues that recent progress with diversity in NLP is disproportionately concentrated on a small number of areas surrounding fairness. We further argue that this is the result of a number of incentives, biases, and barriers which come together to disenfranchise marginalized researchers in non-fairness fields, or to move them into fairness-related fields. We substantiate our claims with an investigation into the demographics of NLP researchers by subfield, using our research to support a number of recommendations for ensuring that all areas within NLP can become more inclusive and equitable. In particular, we highlight the importance of breaking down feedback loops that reinforce disparities, and the need to address geographical and linguistic barriers that hinder participation in NLP research.
>
---
#### [new 009] HUOZIIME: An On-Device LLM-enhanced Input Method for Deep Personalization
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决移动输入法个性化不足的问题。通过在设备端使用轻量大模型，实现高效、实时且个性化的文本输入。**

- **链接: [https://arxiv.org/pdf/2604.14159](https://arxiv.org/pdf/2604.14159)**

> **作者:** Baocai Shan; Yuzhuang Xu; Wanxiang Che
>
> **摘要:** Mobile input method editors (IMEs) are the primary interface for text input, yet they remain constrained to manual typing and struggle to produce personalized text. While lightweight large language models (LLMs) make on-device auxiliary generation feasible, enabling deeply personalized, privacy-preserving, and real-time generative IMEs poses fundamental this http URL this end, we present HUOZIIME, a personalized on-device IME powered by LLM. We endow HUOZIIME with initial human-like prediction ability by post-training a base LLM on synthesized personalization data. Notably, a hierarchical memory mechanism is designed to continually capture and leverage user-specific input history. Furthermore, we perform systemic optimizations tailored to on-device LLMbased IME deployment, ensuring efficient and responsive operation under mobile this http URL demonstrate efficient on-device execution and high-fidelity memory-driven personalization. Code and package are available at this https URL.
>
---
#### [new 010] From Tokens to Steps: Verification-Aware Speculative Decoding for Efficient Multi-Step Reasoning
- **分类: cs.CL**

- **简介: 该论文属于语言模型推理任务，解决 speculative decoding 中错误步骤传播问题。提出 SpecGuard，通过内部信号进行步骤级验证，提升准确率并降低延迟。**

- **链接: [https://arxiv.org/pdf/2604.15244](https://arxiv.org/pdf/2604.15244)**

> **作者:** Kiran Purohit; Ramasuri Narayanam; Soumyabrata Pal
>
> **摘要:** Speculative decoding (SD) accelerates large language model inference by allowing a lightweight draft model to propose outputs that a stronger target model verifies. However, its token-centric nature allows erroneous steps to propagate. Prior approaches mitigate this using external reward models, but incur additional latency, computational overhead, and limit generalizability. We propose SpecGuard, a verification-aware speculative decoding framework that performs step-level verification using only model-internal signals. At each step, SpecGuard samples multiple draft candidates and selects the most consistent step, which is then validated using an ensemble of two lightweight model-internal signals: (i) an attention-based grounding score that measures attribution to the input and previously accepted steps, and (ii) a log-probability-based score that captures token-level confidence. These signals jointly determine whether a step is accepted or recomputed using the target, allocating compute selectively. Experiments across a range of reasoning benchmarks show that SpecGuard improves accuracy by 3.6% while reducing latency by ~11%, outperforming both SD and reward-guided SD.
>
---
#### [new 011] Listen, Correct, and Feed Back: Spoken Pedagogical Feedback Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语音语法纠错与教学反馈生成任务，旨在提供适合学习者的可操作、鼓励性反馈。研究构建了SPFG数据集，并对比不同模型在纠错与反馈生成上的效果。**

- **链接: [https://arxiv.org/pdf/2604.14177](https://arxiv.org/pdf/2604.14177)**

> **作者:** Junhong Liang; Yifan Lu; Ekaterina Kochmar; Fajri Koto
>
> **备注:** NLP8506 course project
>
> **摘要:** Grammatical error correction (GEC) and explanation (GEE) have made rapid progress, but real teaching scenarios also require \emph{learner-friendly pedagogical feedback} that is actionable, level-appropriate, and encouraging. We introduce \textbf{SPFG} (\textbf{S}poken \textbf{P}edagogical \textbf{F}eedback \textbf{G}eneration), a dataset built based on the Speak \& Improve Challenge 2025 corpus, pairing fluency-oriented transcriptions with GEC targets and \emph{human-verified} teacher-style feedback, including preferred/rejected feedback pairs for preference learning. We study a transcript-based Spoken Grammatical Error Correction (SGEC) setting and evaluate three instruction-tuned LLMs (Qwen2.5, Llama-3.1, and GLM-4), comparing supervised fine-tuning (SFT) with preference-based alignment (using DPO and KTO) for jointly generating corrections and feedback. Results show that SFT provides the most consistent improvements, while DPO/KTO yield smaller or mixed gains, and that correction quality and feedback quality are weakly coupled. Our implementation is available at this https URL.
>
---
#### [new 012] Generating Concept Lexicalizations via Dictionary-Based Cross-Lingual Sense Projection
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于跨语言词义扩展任务，旨在通过语义投影将WordNet式资源扩展到新语言。工作包括使用双语词典增强对齐方法，提升精度并减少外部资源依赖。**

- **链接: [https://arxiv.org/pdf/2604.14397](https://arxiv.org/pdf/2604.14397)**

> **作者:** David Basil; Chirooth Girigowda; Bradley Hauer; Sahir Momin; Ning Shi; Grzegorz Kondrak
>
> **备注:** To be published in the proceedings of Canadian AI 2026
>
> **摘要:** We study the task of automatically expanding WordNet-style lexical resources to new languages through sense generation. We generate senses by associating target-language lemmas with existing lexical concepts via semantic projection. Given a sense-tagged English corpus and its translation, our method projects English synsets onto aligned target-language tokens and assigns the corresponding lemmas to those synsets. To generate these alignments and ensure their quality, we augment a pre-trained base aligner with a bilingual dictionary, which is also used to filter out incorrect sense projections. We evaluate the method on multiple languages, comparing it to prior methods, as well as dictionary-based and large language model baselines. Results show that the proposed project-and-filter strategy improves precision while remaining interpretable and requiring few external resources. We plan to make our code, documentation, and generated sense inventories accessible.
>
---
#### [new 013] Purging the Gray Zone: Latent-Geometric Denoising for Precise Knowledge Boundary Awareness
- **分类: cs.CL**

- **简介: 该论文属于语言模型可信度提升任务，解决模型知识边界感知不足导致的幻觉问题。通过几何去噪方法提升模型在边界区域的判断准确性。**

- **链接: [https://arxiv.org/pdf/2604.14324](https://arxiv.org/pdf/2604.14324)**

> **作者:** Hao An; Yibin Lou; Jiayi Guo; Yang Xu
>
> **备注:** ACL 2026 Findings
>
> **摘要:** Large language models (LLMs) often exhibit hallucinations due to their inability to accurately perceive their own knowledge boundaries. Existing abstention fine-tuning methods typically partition datasets directly based on response accuracy, causing models to suffer from severe label noise near the decision boundaries and consequently exhibit high rates of abstentions or hallucinations. This paper adopts a latent space representation perspective, revealing a "gray zone" near the decision hyperplane where internal belief ambiguity constitutes the core performance bottleneck. Based on this insight, we propose the **GeoDe** (**Geo**metric **De**noising) framework for abstention fine-tuning. This method constructs a truth hyperplane using linear probes and performs "geometric denoising" by employing geometric distance as a confidence signal for abstention decisions. This approach filters out ambiguous boundary samples while retaining high-fidelity signals for fine-tuning. Experiments across multiple models (Llama3, Qwen3) and benchmark datasets (TriviaQA, NQ, SciQ, SimpleQA) demonstrate that GeoDe significantly enhances model truthfulness and demonstrates strong generalization in out-of-distribution (OOD) scenarios. Code is available at this https URL.
>
---
#### [new 014] PeerPrism: Peer Evaluation Expertise vs Review-writing AI
- **分类: cs.CL**

- **简介: 该论文属于AI检测任务，旨在解决科学同行评审中人类与AI协作的作者归属问题。通过构建PeerPrism基准，区分思想来源与文本生成，揭示现有方法的局限性。**

- **链接: [https://arxiv.org/pdf/2604.14513](https://arxiv.org/pdf/2604.14513)**

> **作者:** Soroush Sadeghian; Alireza Daqiq; Radin Cheraghi; Sajad Ebrahimi; Negar Arabzadeh; Ebrahim Bagheri
>
> **摘要:** Large Language Models (LLMs) are increasingly used in scientific peer review, assisting with drafting, rewriting, expansion, and refinement. However, existing peer-review LLM detection methods largely treat authorship as a binary problem-human vs. AI-without accounting for the hybrid nature of modern review workflows. In practice, evaluative ideas and surface realization may originate from different sources, creating a spectrum of human-AI collaboration. In this work, we introduce PeerPrism, a large-scale benchmark of 20,690 peer reviews explicitly designed to disentangle idea provenance from text provenance. We construct controlled generation regimes spanning fully human, fully synthetic, and multiple hybrid transformations. This design enables systematic evaluation of whether detectors identify the origin of the surface text or the origin of the evaluative reasoning. We benchmark state-of-the-art LLM text detection methods on PeerPrism. While several methods achieve high accuracy on the standard binary task (human vs. fully synthetic), their predictions diverge sharply under hybrid regimes. In particular, when ideas originate from humans but the surface text is AI-generated, detectors frequently disagree and produce contradictory classifications. Accompanied by stylometric and semantic analyses, our results show that current detection methods conflate surface realization with intellectual contribution. Overall, we demonstrate that LLM detection in peer review cannot be reduced to a binary attribution problem. Instead, authorship must be modeled as a multidimensional construct spanning semantic reasoning and stylistic realization. PeerPrism is the first benchmark evaluating human-AI collaboration in these settings. We release all code, data, prompts, and evaluation scripts to facilitate reproducible research at this https URL.
>
---
#### [new 015] QuantCode-Bench: A Benchmark for Evaluating the Ability of Large Language Models to Generate Executable Algorithmic Trading Strategies
- **分类: cs.CL**

- **简介: 该论文提出QuantCode-Bench，用于评估大语言模型生成可执行量化交易策略的能力，解决交易逻辑与代码生成的对齐问题。**

- **链接: [https://arxiv.org/pdf/2604.15151](https://arxiv.org/pdf/2604.15151)**

> **作者:** Alexey Khoroshilov; Alexey Chernysh; Orkhan Ekhtibarov; Nini Kamkia; Dmitry Zmitrovich
>
> **备注:** 12 pages, 8 tables
>
> **摘要:** Large language models have demonstrated strong performance on general-purpose programming tasks, yet their ability to generate executable algorithmic trading strategies remains underexplored. Unlike standard code benchmarks, trading-strategy generation requires simultaneous mastery of domain-specific financial logic, knowledge of a specialized API, and the ability to produce code that is not only syntactically correct but also leads to actual trades on historical data. In this work, we present QuantCode-Bench, a benchmark for the systematic evaluation of modern LLMs in generating strategies for the Backtrader framework from textual descriptions in English. The benchmark contains 400 tasks of varying difficulty collected from Reddit, TradingView, StackExchange, GitHub, and synthetic sources. Evaluation is conducted through a multi-stage pipeline that checks syntactic correctness, successful backtest execution, the presence of trades, and semantic alignment with the task description using an LLM judge. We compare state-of-the-art models in two settings: single-turn, where the strategy must be generated correctly on the first attempt, and agentic multi-turn, where the model receives iterative feedback and may repair its errors. We analyze the failure modes across different stages of the pipeline and show that the main limitations of current models are not related to syntax, but rather to the correct operationalization of trading logic, proper API usage, and adherence to task semantics. These findings suggest that trading strategy generation constitutes a distinct class of domain-specific code generation tasks in which success requires not only technical correctness, but also alignment between natural-language descriptions, financial logic, and the observable behavior of the strategy on data.
>
---
#### [new 016] RACER: Retrieval-Augmented Contextual Rapid Speculative Decoding
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型推理优化任务，旨在解决自回归解码延迟高的问题。提出RACER方法，结合检索与逻辑引导，提升解码速度与质量。**

- **链接: [https://arxiv.org/pdf/2604.14885](https://arxiv.org/pdf/2604.14885)**

> **作者:** Zihong Zhang; Zuchao Li; Lefei Zhang; Ping Wang; Hai Zhao
>
> **备注:** Accepted to Findings of ACL 2026
>
> **摘要:** Autoregressive decoding in Large Language Models (LLMs) generates one token per step, causing high inference latency. Speculative decoding (SD) mitigates this through a guess-and-verify strategy, but existing training-free variants face trade-offs: retrieval-based drafts break when no exact match exists, while logits-based drafts lack structural guidance. We propose $\textbf{RACER}$ ($\textbf{R}$etrieval-$\textbf{A}$ugmented $\textbf{C}$ont$\textbf{e}$xtual $\textbf{R}$apid Speculative Decoding), a lightweight and training-free method that integrates retrieved exact patterns with logit-driven future cues. This unification supplies both reliable anchors and flexible extrapolation, yielding richer speculative drafts. Experiments on Spec-Bench, HumanEval, and MGSM-ZH demonstrate that RACER consistently accelerates inference, achieving more than $2\times$ speedup over autoregressive decoding, and outperforms prior training-free methods, offering a scalable, plug-and-play solution for efficient LLM decoding. Our source code is available at $\href{this https URL}{this https URL}$.
>
---
#### [new 017] Hierarchical vs. Flat Iteration in Shared-Weight Transformers
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究共享权重的分层循环结构与独立层堆叠在Transformer中的表现差异，旨在提升模型效率。通过对比实验，发现两者存在显著性能差距。**

- **链接: [https://arxiv.org/pdf/2604.14442](https://arxiv.org/pdf/2604.14442)**

> **作者:** Sang-Il Han
>
> **摘要:** We present an empirical study of whether hierarchically structured, shared-weight recurrence can match the representational quality of independent-layer stacking in a Transformer-based language model. HRM-LM replaces L independent Transformer layers with a two-speed recurrent pair: a Fast module operating at every step for local refinement, and a Slow module operating every T steps for global compression. This recurrent hierarchy is unrolled for M = N x T steps with shared parameters. The central and most robust finding, supported by a parameter-matched Universal Transformer ablation (UniTF, 1.2B) across five independent runs, is a sharp empirical gap between the two approaches.
>
---
#### [new 018] When PCOS Meets Eating Disorders: An Explainable AI Approach to Detecting the Hidden Triple Burden
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于心理健康检测任务，旨在解决PCOS患者共病进食障碍的识别问题。通过构建可解释的AI模型，从社交媒体中自动检测三重负担。**

- **链接: [https://arxiv.org/pdf/2604.14356](https://arxiv.org/pdf/2604.14356)**

> **作者:** Apoorv Prasad; Susan McRoy
>
> **摘要:** Women with polycystic ovary syndrome (PCOS) face substantially elevated risks of body image distress, disordered eating, and metabolic challenges, yet existing natural language processing approaches for detecting these conditions lack transparency and cannot identify co-occurring presentations. We developed small, open-source language models to automatically detect this triple burden in social media posts with grounded explainability. We collected 1,000 PCOS-related posts from six subreddits, with two trained annotators labeling posts using guidelines operationalizing Lee et al. (2017) clinical framework. Three models (Gemma-2-2B, Qwen3-1.7B, DeepSeek-R1-Distill-Qwen-1.5B) were fine-tuned using Low-Rank Adaptation to generate structured explanations with textual evidence. The best model achieved 75.3 percent exact match accuracy on 150 held-out posts, with robust comorbidity detection and strong explainability. Performance declined with diagnostic complexity, indicating their best use is for screening rather than autonomous diagnosis.
>
---
#### [new 019] Faithfulness Serum: Mitigating the Faithfulness Gap in Textual Explanations of LLM Decisions via Attribution Guidance
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的可解释性任务，旨在解决大语言模型解释不忠实的问题。通过注意力干预提升解释的客观真实性。**

- **链接: [https://arxiv.org/pdf/2604.14325](https://arxiv.org/pdf/2604.14325)**

> **作者:** Bar Alon; Itamar Zimerman; Lior Wolf
>
> **备注:** 24 pages, multiple figures (e.g., at least 6 main figures), includes experiments across several benchmarks (MMLU, CommonsenseQA, SciQ, ARC, OpenBookQA); code available on GitHub
>
> **摘要:** Large language models (LLMs) achieve strong performance and have revolutionized NLP, but their lack of explainability keeps them treated as black boxes, limiting their use in domains that demand transparency and trust. A promising direction to address this issue is post-hoc text-based explanations, which aim to explain model decisions in natural language. Prior work has focused on generating convincing rationales that appear to be subjectively faithful, but it remains unclear whether these explanations are epistemically faithful, whether they reflect the internal evidence the model actually relied on for its decision. In this paper, we first assess the epistemic faithfulness of LLM-generated explanations via counterfactuals and show that they are often unfaithful. We then introduce a training-free method that enhances faithfulness by guiding explanation generation through attention-level interventions, informed by token-level heatmaps extracted via a faithful attribution method. This method significantly improves epistemic faithfulness across multiple models, benchmarks, and prompts.
>
---
#### [new 020] CausalDetox: Causal Head Selection and Intervention for Language Model Detoxification
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型去毒任务，旨在解决生成有毒内容的问题。通过因果头选择与干预，提升去毒效果并保持语言流畅性。**

- **链接: [https://arxiv.org/pdf/2604.14602](https://arxiv.org/pdf/2604.14602)**

> **作者:** Yian Wang; Yuen Chen; Agam Goyal; Hari Sundaram
>
> **备注:** Accepted to ACL 2026. 22 pages, 1 figure
>
> **摘要:** Large language models (LLMs) frequently generate toxic content, posing significant risks for safe deployment. Current mitigation strategies often degrade generation quality or require costly human annotation. We propose CAUSALDETOX, a framework that identifies and intervenes on the specific attention heads causally responsible for toxic generation. Using the Probability of Necessity and Sufficiency (PNS), we isolate a minimal set of heads that are necessary and sufficient for toxicity. We utilize these components via two complementary strategies: (1) Local Inference-Time Intervention, which constructs dynamic, input-specific steering vectors for context-aware detoxification, and (2) PNS-Guided Fine-Tuning, which permanently unlearns toxic representations. We also introduce PARATOX, a novel benchmark of aligned toxic/non-toxic sentence pairs enabling controlled counterfactual evaluation. Experiments on ToxiGen, ImplicitHate, and ParaDetox show that CAUSALDETOX achieves up to 5.34% greater toxicity reduction compared to baselines while preserving linguistic fluency, and offers a 7x speedup in head selection.
>
---
#### [new 021] Schema Key Wording as an Instruction Channel in Structured Generation under Constrained Decoding
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究约束解码下的结构化生成任务，探讨模式键词作为隐式指令通道的影响，发现改变键词表述可显著影响模型性能。**

- **链接: [https://arxiv.org/pdf/2604.14862](https://arxiv.org/pdf/2604.14862)**

> **作者:** Yifan Le
>
> **备注:** 10 pages, 2 figures. Work in progress
>
> **摘要:** Constrained decoding has been widely adopted for structured generation with large language models (LLMs), ensuring that outputs satisfy predefined formats such as JSON and XML. However, existing approaches largely treat schemas as purely structural constraints and overlook the possibility that their linguistic formulation may affect model behavior. In this work, we study how instruction placement influences model performance in structured generation and show that merely changing the wording of schema keys, without modifying the prompt or model parameters, can significantly alter model performance under constrained decoding. Based on this observation, we propose to reinterpret structured generation as a multi-channel instruction problem, where instructions can be conveyed explicitly through prompts and implicitly through schema keys during decoding. To the best of our knowledge, this is the first work to systematically study how schema key formulation acts as an implicit instruction channel and affects model performance under constrained decoding. Experiments on multiple mathematical reasoning benchmarks show that different model families exhibit distinct sensitivities to these instruction channels: Qwen models consistently benefit from schema-level instructions, while LLaMA models rely more heavily on prompt-level guidance. We further observe non-additive interaction effects between instruction channels, showing that combining multiple channels does not always lead to further improvement. These findings suggest that schema design not only determines output structure, but also carries instruction signals, offering a new perspective on structured generation in LLMs.
>
---
#### [new 022] Text2Arch: A Dataset for Generating Scientific Architecture Diagrams from Natural Language Descriptions
- **分类: cs.CL**

- **简介: 该论文属于科学架构图生成任务，旨在解决从自然语言生成高保真架构图的问题。作者构建了数据集Text2Arch，并验证了模型效果。**

- **链接: [https://arxiv.org/pdf/2604.14941](https://arxiv.org/pdf/2604.14941)**

> **作者:** Shivank Garg; Sankalp Mittal; Manish Gupta
>
> **备注:** ICLR 2026 Poster
>
> **摘要:** Communicating complex system designs or scientific processes through text alone is inefficient and prone to ambiguity. A system that automatically generates scientific architecture diagrams from text with high semantic fidelity can be useful in multiple applications like enterprise architecture visualization, AI-driven software design, and educational content creation. Hence, in this paper, we focus on leveraging language models to perform semantic understanding of the input text description to generate intermediate code that can be processed to generate high-fidelity architecture diagrams. Unfortunately, no clean large-scale open-access dataset exists, implying lack of any effective open models for this task. Hence, we contribute a comprehensive dataset, \system, comprising scientific architecture images, their corresponding textual descriptions, and associated DOT code representations. Leveraging this resource, we fine-tune a suite of small language models, and also perform in-context learning using GPT-4o. Through extensive experimentation, we show that \system{} models significantly outperform existing baseline models like DiagramAgent and perform at par with in-context learning-based generations from GPT-4o. We make the code, data and models publicly available.
>
---
#### [new 023] Filling in the Mechanisms: How do LMs Learn Filler-Gap Dependencies under Developmental Constraints?
- **分类: cs.CL**

- **简介: 该论文研究语言模型在有限数据下学习填充-缺口依赖的机制，属于自然语言处理中的语法学习任务。旨在探讨模型是否具备跨句式共享表示，并验证其数据需求与人类的差异。**

- **链接: [https://arxiv.org/pdf/2604.14459](https://arxiv.org/pdf/2604.14459)**

> **作者:** Atrey Desai; Sathvik Nair
>
> **备注:** To be published in the 64th Annual Meeting of the Association for Computational Linguistics
>
> **摘要:** For humans, filler-gap dependencies require a shared representation across different syntactic constructions. Although causal analyses suggest this may also be true for LLMs (Boguraev et al., 2025), it is still unclear if such a representation also exists for language models trained on developmentally feasible quantities of data. We applied Distributed Alignment Search (DAS, Geiger et al. (2024)) to LMs trained on varying amounts of data from the BabyLM challenge (Warstadt et al., 2023), to evaluate whether representations of filler-gap dependencies transfer between wh-questions and topicalization, which greatly vary in terms of their input frequency. Our results suggest shared, yet item-sensitive mechanisms may develop with limited training data. More importantly, LMs still require far more data than humans to learn comparable generalizations, highlighting the need for language-specific biases in models of language acquisition.
>
---
#### [new 024] Retrieve, Then Classify: Corpus-Grounded Automation of Clinical Value Set Authoring
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文聚焦临床值集生成任务，解决标准化术语中临床概念编码的瓶颈问题。提出RASC方法，通过检索与分类结合，提升编码准确性。**

- **链接: [https://arxiv.org/pdf/2604.14616](https://arxiv.org/pdf/2604.14616)**

> **作者:** Sumit Mukherjee; Juan Shu; Nairwita Mazumder; Tate Kernell; Celena Wheeler; Shannon Hastings; Chris Sidey-Gibbons
>
> **摘要:** Clinical value set authoring -- the task of identifying all codes in a standardized vocabulary that define a clinical concept -- is a recurring bottleneck in clinical quality measurement and phenotyping. A natural approach is to prompt a large language model (LLM) to generate the required codes directly, but structured clinical vocabularies are large, version-controlled, and not reliably memorized during pretraining. We propose Retrieval-Augmented Set Completion (RASC): retrieve the $K$ most similar existing value sets from a curated corpus to form a candidate pool, then apply a classifier to each candidate code. Theoretically, retrieve-and-select can reduce statistical complexity by shrinking the effective output space from the full vocabulary to a much smaller retrieved candidate pool. We demonstrate the utility of RASC on 11,803 publicly available VSAC value sets, constructing the first large-scale benchmark for this task. A cross-encoder fine-tuned on SAPBert achieves AUROC~0.852 and value-set-level F1~0.298, outperforming a simpler three-layer Multilayer Perceptron (AUROC~0.799, F1~0.250) and both reduce the number of irrelevant candidates per true positive from 12.3 (retrieval-only) to approximately 3.2 and 4.4 respectively. Zero-shot GPT-4o achieves value-set-level F1~0.105, with 48.6\% of returned codes absent from VSAC entirely. This performance gap widens with increasing value set size, consistent with RASC's theoretical advantage. We observe similar performance gains across two other classifier model types, namely a cross-encoder initialized from pre-trained SAPBert and a LightGBM model, demonstrating that RASC's benefits extend beyond a single model class. The code to download and create the benchmark dataset, as well as the model training code is available at: \href{this https URL}{this https URL}.
>
---
#### [new 025] SPAGBias: Uncovering and Tracing Structured Spatial Gender Bias in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于社会计算任务，旨在检测和分析大语言模型中的空间性别偏见。通过构建框架SPAGBias，识别模型中隐含的性别空间关联及其影响。**

- **链接: [https://arxiv.org/pdf/2604.14672](https://arxiv.org/pdf/2604.14672)**

> **作者:** Binxian Su; Haoye Lou; Shucheng Zhu; Weikang Wang; Ying Liu; Dong Yu; Pengyuan Liu
>
> **备注:** Accepted by ACL 2026
>
> **摘要:** Large language models (LLMs) are being increasingly used in urban planning, but since gendered space theory highlights how gender hierarchies are embedded in spatial organization, there is concern that LLMs may reproduce or amplify such biases. We introduce SPAGBias - the first systematic framework to evaluate spatial gender bias in LLMs. It combines a taxonomy of 62 urban micro-spaces, a prompt library, and three diagnostic layers: explicit (forced-choice resampling), probabilistic (token-level asymmetry), and constructional (semantic and narrative role analysis). Testing six representative models, we identify structured gender-space associations that go beyond the public-private divide, forming nuanced micro-level mappings. Story generation reveals how emotion, wording, and social roles jointly shape "spatial gender narratives". We also examine how prompt design, temperature, and model scale influence bias expression. Tracing experiments indicate that these patterns are embedded and reinforced across the model pipeline (pre-training, instruction tuning, and reward modeling), with model associations found to substantially exceed real-world distributions. Downstream experiments further reveal that such biases produce concrete failures in both normative and descriptive application settings. This work connects sociological theory with computational analysis, extending bias research into the spatial domain and uncovering how LLMs encode social gender cognition through language.
>
---
#### [new 026] Psychological Steering of Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于语言模型心理引导任务，旨在提升大模型生成内容的心理一致性。通过引入心理校准的残差注入方法，优化了模型的个性特征控制效果。**

- **链接: [https://arxiv.org/pdf/2604.14463](https://arxiv.org/pdf/2604.14463)**

> **作者:** Leonardo Blas; Robin Jia; Emilio Ferrara
>
> **备注:** 66 pages, 60 images
>
> **摘要:** Large language models (LLMs) emulate a consistent human-like behavior that can be shaped through activation-level interventions. This paradigm is converging on additive residual-stream injections, which rely on injection-strength sweeps to approximate optimal intervention settings. However, existing methods restrict the search space and sweep in uncalibrated activation-space units, potentially missing optimal intervention conditions. Thus, we introduce a psychological steering framework that performs unbounded, fluency-constrained sweeps in semantically calibrated units. Our method derives and calibrates residual-stream injections using psychological artifacts, and we use the IPIP-NEO-120, which measures the OCEAN personality model, to compare six injection methods. We find that mean-difference (MD) injections outperform Personality Prompting (P$^2$), an established baseline for OCEAN steering, in open-ended generation in 11 of 14 LLMs, with gains of 3.6\% to 16.4\%, overturning prior reports favoring prompting and positioning representation engineering as a new frontier in open-ended psychological steering. Further, we find that a hybrid of P$^2$ and MD injections outperforms both methods in 13 of 14 LLMs, with gains over P$^2$ ranging from 5.6\% to 21.9\% and from 3.3\% to 26.7\% over MD injections. Finally, we show that MD injections align with the Linear Representation Hypothesis and provide reliable, approximately linear control knobs for psychological steering. Nevertheless, they also induce OCEAN trait covariance patterns that depart from the Big Two model, suggesting a gap between learned representations and human psychology.
>
---
#### [new 027] Compressed-Sensing-Guided, Inference-Aware Structured Reduction for Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于大语言模型压缩任务，旨在解决模型参数多、计算效率低的问题。提出一种动态压缩框架，结合感知和结构化稀疏性，提升推理速度与效率。**

- **链接: [https://arxiv.org/pdf/2604.14156](https://arxiv.org/pdf/2604.14156)**

> **作者:** Andrew Kiruluta
>
> **摘要:** Large language models deliver strong generative performance but at the cost of massive parameter counts, memory use, and decoding latency. Prior work has shown that pruning and structured sparsity can preserve accuracy under substantial compression, while prompt-compression methods reduce latency by removing redundant input tokens. However, these two directions remain largely separate. Most model-compression methods are static and optimized offline, and they do not exploit the fact that different prompts and decoding steps activate different latent computational pathways. Prompt-compression methods reduce sequence length, but they do not adapt the executed model subnetwork. We propose a unified compressed-sensing-guided framework for dynamic LLM execution. Random measurement operators probe latent model usage, sparse recovery estimates task-conditioned and token-adaptive support sets, and the recovered supports are compiled into hardware-efficient sparse execution paths over blocks, attention heads, channels, and feed-forward substructures. The framework introduces five key contributions: task-conditioned measurements, so different prompts induce different sparse supports; token-adaptive recovery, so active substructures are re-estimated during decoding; formal sample-complexity bounds under restricted isometry or mutual incoherence assumptions; compile-to-hardware constraints that restrict recovery to GPU-efficient structures; and a joint objective that unifies prompt compression with model reduction. Together, these components recast LLM inference as a measurement-and-recovery problem with explicit approximation guarantees and deployment-oriented speedup constraints.
>
---
#### [new 028] Knowing When Not to Answer: Evaluating Abstention in Multimodal Reasoning Systems
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于多模态推理任务，旨在解决模型在证据不足时无法有效拒绝回答的问题。通过构建MM-AQA基准，评估不同模型的拒答能力，并提出需要专门训练以提升多模态拒答效果。**

- **链接: [https://arxiv.org/pdf/2604.14799](https://arxiv.org/pdf/2604.14799)**

> **作者:** Nishanth Madhusudhan; Vikas Yadav; Alexandre Lacoste
>
> **备注:** 10 pages and 4 figures (excluding appendix)
>
> **摘要:** Effective abstention (EA), recognizing evidence insufficiency and refraining from answering, is critical for reliable multimodal systems. Yet existing evaluation paradigms for vision-language models (VLMs) and multi-agent systems (MAS) assume answerability, pushing models to always respond. Abstention has been studied in text-only settings but remains underexplored multimodally; current benchmarks either ignore unanswerability or rely on coarse methods that miss realistic failure modes. We introduce MM-AQA, a benchmark that constructs unanswerable instances from answerable ones via transformations along two axes: visual modality dependency and evidence sufficiency. Evaluating three frontier VLMs spanning closed and open-source models and two MAS architectures across 2079 samples, we find: (1) under standard prompting, VLMs rarely abstain; even simple confidence baselines outperform this setup, (2) MAS improves abstention but introduces an accuracy-abstention trade-off, (3) sequential designs match or exceed iterative variants, suggesting the bottleneck is miscalibration rather than reasoning depth, and (4) models abstain when image or text evidence is absent, but attempt reconciliation with degraded or contradictory evidence. Effective multimodal abstention requires abstention-aware training rather than better prompting or more agents.
>
---
#### [new 029] Segment-Level Coherence for Robust Harmful Intent Probing in LLMs
- **分类: cs.CL; cs.CR**

- **简介: 该论文属于有害意图检测任务，旨在解决LLMs中因敏感词误触发导致的错误报警问题。通过引入基于多证据token的流式探测方法，提升检测准确性。**

- **链接: [https://arxiv.org/pdf/2604.14865](https://arxiv.org/pdf/2604.14865)**

> **作者:** Xuanli He; Bilgehan Sel; Faizan Ali; Jenny Bao; Hoagy Cunningham; Jerry Wei
>
> **备注:** preprint
>
> **摘要:** Large Language Models (LLMs) are increasingly exposed to adaptive jailbreaking, particularly in high-stakes Chemical, Biological, Radiological, and Nuclear (CBRN) domains. Although streaming probes enable real-time monitoring, they still make systematic errors. We identify a core issue: existing methods often rely on a few high-scoring tokens, leading to false alarms when sensitive CBRN terms appear in benign contexts. To address this, we introduce a streaming probing objective that requires multiple evidence tokens to consistently support a prediction, rather than relying on isolated spikes. This encourages more robust detection based on aggregated signals instead of single-token cues. At a fixed 1% false-positive rate, our method improves the true-positive rate by 35.55% relative to strong streaming baselines. We further observe substantial gains in AUROC, even when starting from near-saturated baseline performance (AUROC = 97.40%). We also show that probing Attention or MLP activations consistently outperforms residual-stream features. Finally, even when adversarial fine-tuning enables novel character-level ciphers, harmful intent remains detectable: probes developed for the base LLMs can be applied ``plug-and-play'' to these obfuscated attacks, achieving an AUROC of over 98.85%.
>
---
#### [new 030] CURA: Clinical Uncertainty Risk Alignment for Language Model-Based Risk Prediction
- **分类: cs.CL**

- **简介: 该论文属于临床风险预测任务，旨在解决语言模型不确定性校准不足的问题。提出CURA框架，通过患者嵌入和不确定性微调提升风险估计的可靠性。**

- **链接: [https://arxiv.org/pdf/2604.14651](https://arxiv.org/pdf/2604.14651)**

> **作者:** Sizhe Wang; Ziqi Xu; Claire Najjuuko; Charles Alba; Chenyang Lu
>
> **备注:** Accepted at ACL 2026 Main Conference
>
> **摘要:** Clinical language models (LMs) are increasingly applied to support clinical risk prediction from free-text notes, yet their uncertainty estimates often remain poorly calibrated and clinically unreliable. In this work, we propose Clinical Uncertainty Risk Alignment (CURA), a framework that aligns clinical LM-based risk estimates and uncertainty with both individual error likelihoods and cohort-level ambiguities. CURA first fine-tunes domain-specific clinical LMs to obtain task-adapted patient embeddings, and then performs uncertainty fine-tuning of a multi-head classifier using a bi-level uncertainty objective. Specifically, an individual-level calibration term aligns predictive uncertainty with each patient's likelihood of error, while a cohort-aware regularizer pulls risk estimates toward event rates in their local neighborhoods in the embedding space and places extra weight on ambiguous cohorts near the decision boundary. We further show that this cohort-aware term can be interpreted as a cross-entropy loss with neighborhood-informed soft labels, providing a label-smoothing view of our method. Extensive experiments on MIMIC-IV clinical risk prediction tasks across various clinical LMs show that CURA consistently improves calibration metrics without substantially compromising discrimination. Further analysis illustrates that CURA reduces overconfident false reassurance and yields more trustworthy uncertainty estimates for downstream clinical decision support.
>
---
#### [new 031] Domain Fine-Tuning FinBERT on Finnish Histopathological Reports: Train-Time Signals and Downstream Correlations
- **分类: cs.CL**

- **简介: 该论文研究在芬兰医疗文本上微调BERT模型，旨在解决标签数据不足的分类任务。通过分析嵌入变化，探索领域预训练的效益。**

- **链接: [https://arxiv.org/pdf/2604.14815](https://arxiv.org/pdf/2604.14815)**

> **作者:** Rami Luisto; Liisa Petäinen; Tommi Grönholm; Jan Böhm; Maarit Ahtiainen; Tomi Lilja; Ilkka Pölönen; Sami Äyrämö
>
> **摘要:** In NLP classification tasks where little labeled data exists, domain fine-tuning of transformer models on unlabeled data is an established approach. In this paper we have two aims. (1) We describe our observations from fine-tuning the Finnish BERT model on Finnish medical text data. (2) We report on our attempts to predict the benefit of domain-specific pre-training of Finnish BERT from observing the geometry of embedding changes due to domain fine-tuning. Our driving motivation is the common\situation in healthcare AI where we might experience long delays in acquiring datasets, especially with respect to labels.
>
---
#### [new 032] Hierarchical Retrieval Augmented Generation for Adversarial Technique Annotation in Cyber Threat Intelligence Text
- **分类: cs.CL**

- **简介: 该论文属于网络安全领域，针对CTI文本到ATT&CK技术ID的标注任务。解决传统方法忽略ATT&CK层级结构的问题，提出H-TechniqueRAG框架，提升标注效率与准确性。**

- **链接: [https://arxiv.org/pdf/2604.14166](https://arxiv.org/pdf/2604.14166)**

> **作者:** Filippo Morbiato; Markus Keller; Priya Nair; Luca Romano
>
> **摘要:** Mapping Cyber Threat Intelligence (CTI) text to MITRE ATT\&CK technique IDs is a critical task for understanding adversary behaviors and automating threat defense. While recent Retrieval-Augmented Generation (RAG) approaches have demonstrated promising capabilities in this domain, they fundamentally rely on a flat retrieval paradigm. By treating all techniques uniformly, these methods overlook the inherent taxonomy of the ATT\&CK framework, where techniques are structurally organized under high-level tactics. In this paper, we propose H-TechniqueRAG, a novel hierarchical RAG framework that injects this tactic-technique taxonomy as a strong inductive bias to achieve highly efficient and accurate annotation. Our approach introduces a two-stage hierarchical retrieval mechanism: it first identifies the macro-level tactics (the adversary's technical goals) and subsequently narrows the search to techniques within those tactics, effectively reducing the candidate search space by 77.5\%. To further bridge the gap between retrieval and generation, we design a tactic-aware reranking module and a hierarchy-constrained context organization strategy that mitigates LLM context overload and improves reasoning precision. Comprehensive experiments across three diverse CTI datasets demonstrate that H-TechniqueRAG not only outperforms the state-of-the-art TechniqueRAG by 3.8\% in F1 score, but also achieves a 62.4\% reduction in inference latency and a 60\% decrease in LLM API calls. Further analysis reveals that our hierarchical structural priors equip the model with superior cross-domain generalization and provide security analysts with highly interpretable, step-by-step decision paths.
>
---
#### [new 033] Can Large Language Models Detect Methodological Flaws? Evidence from Gesture Recognition for UAV-Based Rescue Operation Based on Deep Learning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于方法论审计任务，旨在检测机器学习研究中的数据泄露问题。通过分析一篇手势识别论文，验证大语言模型能否识别评估协议中的缺陷。**

- **链接: [https://arxiv.org/pdf/2604.14161](https://arxiv.org/pdf/2604.14161)**

> **作者:** Domonkos Varga
>
> **摘要:** Reliable evaluation is essential in machine learning research, yet methodological flaws-particularly data leakage-continue to undermine the validity of reported results. In this work, we investigate whether large language models (LLMs) can act as independent analytical agents capable of identifying such issues in published studies. As a case study, we analyze a gesture-recognition paper reporting near-perfect accuracy on a small, human-centered dataset. We first show that the evaluation protocol is consistent with subject-level data leakage due to non-independent training and test splits. We then assess whether this flaw can be detected independently by six state-of-the-art LLMs, each analyzing the original paper without prior context using an identical prompt. All models consistently identify the evaluation as flawed and attribute the reported performance to non-independent data partitioning, supported by indicators such as overlapping learning curves, minimal generalization gap, and near-perfect classification results. These findings suggest that LLMs can detect common methodological issues based solely on published artifacts. While not definitive, their consistent agreement highlights their potential as complementary tools for improving reproducibility and supporting scientific auditing.
>
---
#### [new 034] ClimateCause: Complex and Implicit Causal Structures in Climate Reports
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出ClimateCause数据集，用于研究气候报告中的复杂因果结构。任务是因果关系建模，解决现有数据缺乏隐含和嵌套因果的问题，通过专家标注和图构建进行分析。**

- **链接: [https://arxiv.org/pdf/2604.14856](https://arxiv.org/pdf/2604.14856)**

> **作者:** Liesbeth Allein; Nataly Pineda-Castañeda; Andrea Rocci; Marie-Francine Moens
>
> **备注:** Accepted to ACL 2026 [Findings]
>
> **摘要:** Understanding climate change requires reasoning over complex causal networks. Yet, existing causal discovery datasets predominantly capture explicit, direct causal relations. We introduce ClimateCause, a manually expert-annotated dataset of higher-order causal structures from science-for-policy climate reports, including implicit and nested causality. Cause-effect expressions are normalized and disentangled into individual causal relations to facilitate graph construction, with unique annotations for cause-effect correlation, relation type, and spatiotemporal context. We further demonstrate ClimateCause's value for quantifying readability based on the semantic complexity of causal graphs underlying a statement. Finally, large language model benchmarking on correlation inference and causal chain reasoning highlights the latter as a key challenge.
>
---
#### [new 035] MARCA: A Checklist-Based Benchmark for Multilingual Web Search
- **分类: cs.CL**

- **简介: 该论文提出MARCA，一个用于评估多语言大模型网络信息检索的基准。解决多语言搜索可靠性问题，通过双语问题和检查清单评估答案完整性和正确性。**

- **链接: [https://arxiv.org/pdf/2604.14448](https://arxiv.org/pdf/2604.14448)**

> **作者:** Thales Sales Almeida; Giovana Kerche Bonás; Ramon Pires; Celio Larcher; Hugo Abonizio; Marcos Piau; Roseval Malaquias Junior; Rodrigo Nogueira; Thiago Laitz
>
> **摘要:** Large language models (LLMs) are increasingly used as sources of information, yet their reliability depends on the ability to search the web, select relevant evidence, and synthesize complete answers. While recent benchmarks evaluate web-browsing and agentic tool use, multilingual settings, and Portuguese in particular, remain underexplored. We present \textsc{MARCA}, a bilingual (English and Portuguese) benchmark for evaluating LLMs on web-based information seeking. \textsc{MARCA} consists of 52 manually authored multi-entity questions, paired with manually validated checklist-style rubrics that explicitly measure answer completeness and correctness. We evaluate 14 models under two interaction settings: a Basic framework with direct web search and scraping, and an Orchestrator framework that enables task decomposition via delegated subagents. To capture stochasticity, each question is executed multiple times and performance is reported with run-level uncertainty. Across models, we observe large performance differences, find that orchestration often improves coverage, and identify substantial variability in how models transfer from English to Portuguese. The benchmark is available at this https URL
>
---
#### [new 036] CROP: Token-Efficient Reasoning in Large Language Models via Regularized Prompt Optimization
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型推理时的高能耗问题。通过引入正则化机制，提出CROP方法，在保持准确性的前提下减少token消耗。**

- **链接: [https://arxiv.org/pdf/2604.14214](https://arxiv.org/pdf/2604.14214)**

> **作者:** Deep Shah; Sanket Badhe; Nehal Kathrotia; Priyanka Tiwari
>
> **备注:** Accepted at ICLR 2026 Workshop on Logical Reasoning of Large Language Models
>
> **摘要:** Large Language Models utilizing reasoning techniques improve task performance but incur significant latency and token costs due to verbose generation. Existing automatic prompt optimization(APO) frameworks target task accuracy exclusively at the expense of generating long reasoning traces. We propose Cost-Regularized Optimization of Prompts (CROP), an APO method that introduces regularization on response length by generating textual feedback in addition to standard accuracy feedback. This forces the optimization process to produce prompts that elicit concise responses containing only critical information and reasoning. We evaluate our approach on complex reasoning datasets, specifically GSM8K, LogiQA and BIG-Bench Hard. We achieved an 80.6\% reduction in token consumption while maintaining competitive accuracy, seeing only a nominal decline in performance. This presents a pragmatic solution for deploying token-efficient and cost-effective agentic AI systems in production pipelines.
>
---
#### [new 037] Attention to Mamba: A Recipe for Cross-Architecture Distillation
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于模型压缩任务，旨在将Transformer模型知识有效迁移到Mamba架构中。通过两阶段蒸馏方法，提升Mamba性能，使其接近原Transformer表现。**

- **链接: [https://arxiv.org/pdf/2604.14191](https://arxiv.org/pdf/2604.14191)**

> **作者:** Abhinav Moudgil; Ningyuan Huang; Eeshan Gunesh Dhekane; Pau Rodríguez; Luca Zappella; Federico Danieli
>
> **摘要:** State Space Models (SSMs) such as Mamba have become a popular alternative to Transformer models, due to their reduced memory consumption and higher throughput at generation compared to their Attention-based counterparts. On the other hand, the community has built up a considerable body of knowledge on how to train Transformers, and many pretrained Transformer models are readily available. To facilitate the adoption of SSMs while leveraging existing pretrained Transformers, we aim to identify an effective recipe to distill an Attention-based model into a Mamba-like architecture. In prior work on cross-architecture distillation, however, it has been shown that a naïve distillation procedure from Transformers to Mamba fails to preserve the original teacher performance, a limitation often overcome with hybrid solutions combining Attention and SSM blocks. The key argument from our work is that, by equipping Mamba with a principled initialization, we can recover an overall better recipe for cross-architectural distillation. To this end, we propose a principled two-stage approach: first, we distill knowledge from a traditional Transformer into a linearized version of Attention, using an adaptation of the kernel trick. Then, we distill the linearized version into an adapted Mamba model that does not use any Attention block. Overall, the distilled Mamba model is able to preserve the original Pythia-1B Transformer performance in downstream tasks, maintaining a perplexity of 14.11 close to the teacher's 13.86. To show the efficacy of our recipe, we conduct thorough ablations at 1B scale with 10B tokens varying sequence mixer architecture, scaling analysis on model sizes and total distillation tokens, and a sensitivity analysis on tokens allocation between stages.
>
---
#### [new 038] Internal Knowledge Without External Expression: Probing the Generalization Boundary of a Classical Chinese Language Model
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型的泛化能力，探讨其是否能区分已知与未知信息并表达不确定性。任务属于自然语言处理中的语言建模与元认知研究。**

- **链接: [https://arxiv.org/pdf/2604.14180](https://arxiv.org/pdf/2604.14180)**

> **作者:** Jiuting Chen; Yuan Lian; Hao Wu; Tianqi Huang; Hiroshi Sasaki; Makoto Kouno; Jongil Choi
>
> **备注:** 15 pages, 5 figures, supplementary material included
>
> **摘要:** We train a 318M-parameter Transformer language model from scratch on a curated corpus of 1.56 billion tokens of pure Classical Chinese, with zero English characters or Arabic numerals. Through systematic out-of-distribution (OOD) testing, we investigate whether the model can distinguish known from unknown inputs, and crucially, whether it can express this distinction in its generated text. We find a clear dissociation between internal and external uncertainty. Internally, the model exhibits a perplexity jump ratio of 2.39x between real and fabricated historical events (p = 8.9e-11, n = 92 per group), with semi-fabricated events (real figures + fictional events) showing the highest perplexity (4.24x, p = 1.1e-16), demonstrating genuine factual encoding beyond syntactic pattern matching. Externally, however, the model never learns to express uncertainty: classical Chinese epistemic markers appear at lower rates for OOD questions (3.5%) than for in-distribution questions (8.3%, p = 0.023), reflecting rhetorical conventions rather than genuine metacognition. We replicate both findings across three languages (Classical Chinese, English, Japanese), three writing systems, and eight models from 110M to 1.56B parameters. We further show that uncertainty expression frequency is determined entirely by training data conventions, with Classical Chinese models showing a "humility paradox" (more hedging for known topics), while Japanese models almost never hedge. We argue that metacognitive expression -- the ability to say "I don't know" -- does not emerge from language modeling alone and requires explicit training signals such as RLHF.
>
---
#### [new 039] MemGround: Long-Term Memory Evaluation Kit for Large Language Models in Gamified Scenarios
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出MemGround，用于评估大语言模型在游戏化场景中的长期记忆能力，解决现有评估方法静态、片面的问题，通过多层级框架和多维指标进行系统评测。**

- **链接: [https://arxiv.org/pdf/2604.14158](https://arxiv.org/pdf/2604.14158)**

> **作者:** Yihang Ding; Wanke Xia; Yiting Zhao; Jinbo Su; Jialiang Yang; Zhengbo Zhang; Ke Wang; Wenming Yang
>
> **摘要:** Current evaluations of long-term memory in LLMs are fundamentally static. By fixating on simple retrieval and short-context inference, they neglect the multifaceted nature of complex memory systems, such as dynamic state tracking and hierarchical reasoning in continuous interactions. To overcome these limitations, we propose MemGround, a rigorous long-term memory benchmark natively grounded in rich, gamified interactive scenarios. To systematically assess these capabilities, MemGround introduces a three-tier hierarchical framework that evaluates Surface State Memory, Temporal Associative Memory, and Reasoning-Based Memory through specialized interactive tasks. Furthermore, to comprehensively quantify both memory utilization and behavioral trajectories, we propose a multi-dimensional metric suite comprising Question-Answer Score (QA Overall), Memory Fragments Unlocked (MFU), Memory Fragments with Correct Order (MFCO), and Exploration Trajectory Diagrams (ETD). Extensive experiments reveal that state-of-the-art LLMs and memory agents still struggle with sustained dynamic tracking, temporal event association, and complex reasoning derived from long-term accumulated evidence in interactive environments.
>
---
#### [new 040] IE as Cache: Information Extraction Enhanced Agentic Reasoning
- **分类: cs.CL**

- **简介: 该论文属于信息提取任务，旨在解决传统IE仅作为终端目标的问题。通过将IE作为认知缓存，提升多步骤推理效果。**

- **链接: [https://arxiv.org/pdf/2604.14930](https://arxiv.org/pdf/2604.14930)**

> **作者:** Hang Lv; Sheng Liang; Hongchao Gu; Wei Guo; Defu Lian; Yong Liu; Hao Wang; Enhong Chen
>
> **备注:** 8pages, 2figures
>
> **摘要:** Information Extraction aims to distill structured, decision-relevant information from unstructured text, serving as a foundation for downstream understanding and reasoning. However, it is traditionally treated merely as a terminal objective: once extracted, the resulting structure is often consumed in isolation rather than maintained and reused during multi-step inference. Moving beyond this, we propose \textit{IE-as-Cache}, a framework that repurposes IE as a cognitive cache to enhance agentic reasoning. Drawing inspiration from hierarchical computer memory, our approach combines query-driven extraction with cache-aware reasoning to dynamically maintain compact intermediate information and filter noise. Experiments on challenging benchmarks across diverse LLMs demonstrate significant improvements in reasoning accuracy, indicating that IE can be effectively repurposed as a reusable cognitive resource and offering a promising direction for future research on downstream uses of IE.
>
---
#### [new 041] MEME-Fusion@CHiPSAL 2026: Multimodal Ablation Study of Hate Detection and Sentiment Analysis on Nepali Memes
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模态 hate speech 检测与情感分析任务，解决低资源环境下尼泊尔语表情包的多模态内容理解问题。提出跨模态注意力融合架构，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2604.14218](https://arxiv.org/pdf/2604.14218)**

> **作者:** Samir Wagle; Reewaj Khanal; Abiral Adhikari
>
> **备注:** PrePrint
>
> **摘要:** Hate speech detection in Devanagari-scripted social media memes presents compounded challenges: multimodal content structure, script-specific linguistic complexity, and extreme data scarcity in low-resource settings. This paper presents our system for the CHiPSAL 2026 shared task, addressing both Subtask A (binary hate speech detection) and Subtask B (three-class sentiment classification: positive, neutral, negative). We propose a hybrid cross-modal attention fusion architecture that combines CLIP (ViT-B/32) for visual encoding with BGE-M3 for multilingual text representation, connected through 4-head self-attention and a learnable gating network that dynamically weights modality contributions on a per-sample basis. Systematic evaluation across eight model configurations demonstrates that explicit cross-modal reasoning achieves a 5.9% F1-macro improvement over text-only baselines on Subtask A, while uncovering two unexpected but critical findings: English-centric vision models exhibit near-random performance on Devanagari script, and standard ensemble methods catastrophically degrade under data scarcity (N nearly equal to 850 per fold) due to correlated overfitting. The code can be accessed at this https URL
>
---
#### [new 042] Pushing the Boundaries of Multiple Choice Evaluation to One Hundred Options
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，针对多选评估中模型表现被高估的问题，提出百选项评估框架以更准确测试模型可靠性。**

- **链接: [https://arxiv.org/pdf/2604.14634](https://arxiv.org/pdf/2604.14634)**

> **作者:** Nahyun Lee; Guijin Son
>
> **摘要:** Multiple choice evaluation is widely used for benchmarking large language models, yet near ceiling accuracy in low option settings can be sustained by shortcut strategies that obscure true competence. Therefore, we propose a massive option evaluation protocol that scales the candidate set to one hundred options and sharply reduces the impact of chance performance. We apply this framework to a Korean orthography error detection task where models must pick the single incorrect sentence from a large candidate set. With fixed targets and repeated resampling and shuffling, we obtain stable estimates while separating content driven failures from positional artifacts. Across experiments, results indicate that strong performance in low option settings can overstate model competence. This apparent advantage often weakens under dense interference at high $N$, revealing gaps that conventional benchmarks tend to obscure. We identify two failure modes, semantic confusion and position bias toward early options under uncertainty. To isolate the effect of context length, we run padding controlled and length matched tests, which suggest that the main bottleneck is candidate ranking rather than context length. Together, these findings support massive option evaluation as a general framework for stress testing model reliability under extreme distractor density, beyond what low option benchmarks can reveal.
>
---
#### [new 043] Pangu-ACE: Adaptive Cascaded Experts for Educational Response Generation on EduBench
- **分类: cs.CL**

- **简介: 该论文提出Pangu-ACE系统，用于教育问答生成任务，通过自适应级联专家机制提升回答质量与效率。**

- **链接: [https://arxiv.org/pdf/2604.14828](https://arxiv.org/pdf/2604.14828)**

> **作者:** Dinghao Li; Wenlong Zhou; Zhimin Chen; Yuehan Peng; Hong Ni; Chengfu Zou; Guoyu Shi; Yaochen Li
>
> **摘要:** Educational assistants should spend more computation only when the task needs it. This paper rewrites our earlier draft around the system that was actually implemented and archived in the repository: a sample-level 1B to 7B cascade for the shared-8 EduBench benchmark. The final system, Pangu-ACE, uses a 1B tutor-router to produce a draft answer plus routing signals, then either accepts the draft or escalates the sample to a 7B specialist prompt. We also correct a major offline evaluation bug: earlier summaries over-credited some open-form outputs that only satisfied superficial format checks. After CPU-side rescoring from saved prediction JSONL, the full Chinese test archive (7013 samples) shows that cascade_final improves deterministic quality from 0.457 to 0.538 and format validity from 0.707 to 0.866 over the legacy rule_v2 system while accepting 19.7% of requests directly at 1B. Routing is strongly task dependent: IP is accepted by 1B 78.0% of the time, while QG and EC still escalate almost always. The current archived deployment does not yet show latency gains, so the defensible efficiency story is routing selectivity rather than wall-clock speedup. We also package a reproducible artifact-first paper workflow and clarify the remaining external-baseline gap: GPT-5.4 re-judging is implemented locally, but the configured provider endpoint and key are invalid, so final sampled-baseline alignment with GPT-5.4 remains pending infrastructure repair.
>
---
#### [new 044] Correcting Suppressed Log-Probabilities in Language Models with Post-Transformer Adapters
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语言模型修正任务，解决政治敏感话题中事实概率被抑制的问题。通过训练后置适配器，恢复被压制的log概率，提升生成文本的连贯性和真实性。**

- **链接: [https://arxiv.org/pdf/2604.14174](https://arxiv.org/pdf/2604.14174)**

> **作者:** Bryan Sanchez
>
> **备注:** 12 pages, 3 figures, code at this https URL
>
> **摘要:** Alignment-tuned language models frequently suppress factual log-probabilities on politically sensitive topics despite retaining the knowledge in their hidden representations. We show that a 786K-parameter (approximately 0.02% of the base model) post-transformer adapter, trained on frozen hidden states, corrects this suppression on 31 ideology-discriminating facts across Qwen3-4B, 8B, and 14B. The adapter memorizes all 15 training facts and generalizes to 11--39% of 16 held-out facts across 5 random splits per scale, with zero knowledge regressions via anchored training. Both gated (SwiGLU) and ungated (linear bottleneck) adapters achieve comparable results; neither consistently outperforms the other (Fisher exact p > 0.09 at all scales). On instruct models, the adapter corrects log-probability rankings. When applied at all token positions during generation, the adapter produces incoherent output; however, when applied only at the current prediction position (last-position-only), the adapter produces coherent, less censored text. A logit-space adapter operating after token projection fails to produce coherent generation at any application mode, suggesting hidden-state intervention is the correct level for generation correction. A previously undocumented silent gradient bug in Apple MLX explains all null results in earlier iterations of this work: the standard pattern nn.value_and_grad(model, fn)(this http URL()) returns zero gradients without error; the correct pattern nn.value_and_grad(model, fn)(model, data) resolves this. We provide a minimal reproduction and discuss implications for other adapter research using MLX.
>
---
#### [new 045] EuropeMedQA Study Protocol: A Multilingual, Multimodal Medical Examination Dataset for Language Model Evaluation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗AI领域，旨在解决多语言和多模态医学评估问题。构建了EuropeMedQA数据集，用于评估语言模型的跨语言和视觉推理能力。**

- **链接: [https://arxiv.org/pdf/2604.14306](https://arxiv.org/pdf/2604.14306)**

> **作者:** Francesco Andrea Causio; Vittorio De Vita; Olivia Riccomi; Michele Ferramola; Federico Felizzi; Antonio Cristiano; Lorenzo De Mori; Chiara Battipaglia; Melissa Sawaya; Luigi De Angelis; Marcello Di Pumpo; Alessandra Piscitelli; Pietro Eric Risuleo; Alessia Longo; Giulia Vojvodic; Mariapia Vassalli; Bianca Destro Castaniti; Nicolò Scarsi; Manuel Del Medico
>
> **摘要:** While Large Language Models (LLMs) have demonstrated high proficiency on English-centric medical examinations, their performance often declines when faced with non-English languages and multimodal diagnostic tasks. This study protocol describes the development of EuropeMedQA, the first comprehensive, multilingual, and multimodal medical examination dataset sourced from official regulatory exams in Italy, France, Spain, and Portugal. Following FAIR data principles and SPIRIT-AI guidelines, we describe a rigorous curation process and an automated translation pipeline for comparative analysis. We evaluate contemporary multimodal LLMs using a zero-shot, strictly constrained prompting strategy to assess cross-lingual transfer and visual reasoning. EuropeMedQA aims to provide a contamination-resistant benchmark that reflects the complexity of European clinical practices and fosters the development of more generalizable medical AI.
>
---
#### [new 046] Shuffle the Context: RoPE-Perturbed Self-Distillation for Long-Context Adaptation
- **分类: cs.CL**

- **简介: 该论文针对长文本理解任务，解决模型对位置敏感导致的稳定性问题，提出RoPE-Perturbed Self-Distillation方法提升模型位置鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.14339](https://arxiv.org/pdf/2604.14339)**

> **作者:** Zichong Li; Chen Liang; Liliang Ren; Tuo Zhao; Yelong Shen; Weizhu Chen
>
> **摘要:** Large language models (LLMs) increasingly operate in settings that require reliable long-context understanding, such as retrieval-augmented generation and multi-document reasoning. A common strategy is to fine-tune pretrained short-context models at the target sequence length. However, we find that standard long-context adaptation can remain brittle: model accuracy depends strongly on the absolute placement of relevant evidence, exhibiting high positional variance even when controlling for task format and difficulty. We propose RoPE-Perturbed Self-Distillation, a training regularizer that improves positional robustness. The core idea is to form alternative "views" of the same training sequence by perturbing its RoPE indices -- effectively moving parts of the context to different positions -- and to train the model to produce consistent predictions across views via self-distillation. This encourages reliance on semantic signals instead of brittle position dependencies. Experiments on long-context adaptation of Llama-3-8B and Qwen-3-4B demonstrate consistent gains on long-context benchmarks, including up to 12.04% improvement on RULER-64K for Llama-3-8B and 2.71% on RULER-256K for Qwen-3-4B after SFT, alongside improved length extrapolation beyond the training context window.
>
---
#### [new 047] Which bird does not have wings: Negative-constrained KGQA with Schema-guided Semantic Matching and Self-directed Refinement
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出NEST-KGQA任务，解决知识图谱问答中负约束识别与处理问题，设计PyLF逻辑形式和CUCKOO框架提升准确性和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.14749](https://arxiv.org/pdf/2604.14749)**

> **作者:** Midan Shim; Seokju Hwang; Kaehyun Um; Kyong-Ho Lee
>
> **备注:** ACL 2026 findings
>
> **摘要:** Large language models still struggle with faithfulness and hallucinations despite their remarkable reasoning abilities. In Knowledge Graph Question Answering (KGQA), semantic parsing-based approaches address the limitations by understanding constraints in a user's question and converting them into a logical form to execute on a knowledge graph. However, existing KGQA benchmarks and methods are biased toward positive and calculation constraints. Negative constraints are neglected, although they frequently appear in real-world questions. In this paper, we introduce a new task, NEgative-conSTrained (NEST) KGQA, where each question contains at least one negative constraint, and a corresponding dataset, NestKGQA. We also design PyLF, a Python-formatted logical form, since existing logical forms are hardly suitable to express negation clearly while maintaining readability. Furthermore, NEST questions naturally contain multiple constraints. To mitigate their semantic complexity, we present a novel framework named CUCKOO, specialized to multiple-constrained questions and ensuring semantic executability. CUCKOO first generates a constraint-aware logical form draft and performs schema-guided semantic matching. It then selectively applies self-directed refinement only when executing improper logical forms yields an empty result, reducing cost while improving robustness. Experimental results demonstrate that CUCKOO consistently outperforms baselines on both conventional KGQA and NEST-KGQA benchmarks under few-shot settings.
>
---
#### [new 048] Tracking the Temporal Dynamics of News Coverage of Catastrophic and Violent Events
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于新闻分析任务，研究灾难和暴力事件的新闻报道动态。通过分析大量新闻文本，量化报道变化，揭示其时间与语义模式。**

- **链接: [https://arxiv.org/pdf/2604.14315](https://arxiv.org/pdf/2604.14315)**

> **作者:** Emily Lugos; Maurício Gruppi
>
> **摘要:** The modern news cycle has been fundamentally reshaped by the rapid exchange of information online. As a result, media framing shifts dynamically as new information, political responses, and social reactions emerge. Understanding how these narratives form, propagate, and evolve is essential for interpreting public discourse during moments of crisis. In this study, we examine the temporal and semantic dynamics of reporting for violent and catastrophic events using a large-scale corpus of 126,602 news articles collected from online publishers. We quantify narrative change through publication volume, semantic drift, semantic dispersion, and term relevance. Our results show that sudden events of impact exhibit structured and predictable news-cycle patterns characterized by rapid surges in coverage, early semantic drift, and gradual declines toward the baseline. In addition, our results indicate the terms that are driving the temporal patterns.
>
---
#### [new 049] DiscoTrace: Representing and Comparing Answering Strategies of Humans and LLMs in Information-Seeking Question Answering
- **分类: cs.CL**

- **简介: 论文提出DiscoTrace方法，用于分析人类和LLMs在信息检索问答中的回答策略。任务是理解回答构建方式，解决比较不同群体与LLMs策略差异的问题。工作包括构建策略表示并分析答案结构。**

- **链接: [https://arxiv.org/pdf/2604.15140](https://arxiv.org/pdf/2604.15140)**

> **作者:** Neha Srikanth; Jordan Boyd-Graber; Rachel Rudinger
>
> **摘要:** We introduce DiscoTrace, a method to identify the rhetorical strategies that answerers use when responding to information-seeking questions. DiscoTrace represents answers as a sequence of question-related discourse acts paired with interpretations of the original question, annotated on top of rhetorical structure theory parses. Applying DiscoTrace to answers from nine different human communities reveals that communities have diverse preferences for answer construction. In contrast, LLMs do not exhibit rhetorical diversity in their answers, even when prompted to mimic specific human community answering guidelines. LLMs also systematically opt for breadth, addressing interpretations of questions that human answerers choose not to address. Our findings can guide the development of pragmatic LLM answerers that consider a range of strategies informed by context in QA.
>
---
#### [new 050] The PICCO Framework for Large Language Model Prompting: A Taxonomy and Reference Architecture for Prompt Structure
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出PICCO框架，解决大语言模型提示设计不一致的问题。通过分析现有框架，构建五要素参考架构，提升提示设计的系统性和清晰度。**

- **链接: [https://arxiv.org/pdf/2604.14197](https://arxiv.org/pdf/2604.14197)**

> **作者:** David A. Cook
>
> **备注:** Presents the novel PICCO framework for LLM prompting, derived through a structured multi-database search and rigorous comparative synthesis of 11 published prompting frameworks. Submitted in PDF/A format to preserve the structure and readability of several multi-page tables central to the framework and methodology; these contain dense structured information that is best preserved in PDF form
>
> **摘要:** Large language model (LLM) performance depends heavily on prompt design, yet prompt construction is often described and applied inconsistently. Our purpose was to derive a reference framework for structuring LLM prompts. This paper presents PICCO, a framework derived through a rigorous synthesis of 11 previously published prompting frameworks identified through a multi-database search. The analysis yields two main contributions. First, it proposes a taxonomy that distinguishes prompt frameworks, prompt elements, prompt generation, prompting techniques, and prompt engineering as related but non-equivalent concepts. Second, it derives a five-element reference architecture for prompt generation: Persona, Instructions, Context, Constraints, and Output (PICCO). For each element, we define its function, scope, and relationship to other elements, with the goal of improving conceptual clarity and supporting more systematic prompt design. Finally, to support application of the framework, we outline key concepts relevant to implementation, including prompting techniques (e.g., zero-shot, few-shot, chain-of-thought, ensembling, decomposition, and self-critique, with selected variants), human and automated approaches to iterative prompt engineering, responsible prompting considerations such as security, privacy, bias, and trust, and priorities for future research. This work is a conceptual and methodological contribution: it formalizes a common structure for prompt specification and comparison, but does not claim empirical validation of PICCO as an optimization method.
>
---
#### [new 051] Stateful Evidence-Driven Retrieval-Augmented Generation with Iterative Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于问答任务，解决RAG框架中上下文表示扁平和检索无状态导致的性能不稳定问题。提出一种带有迭代推理的有状态证据驱动RAG框架，提升证据聚合稳定性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.14170](https://arxiv.org/pdf/2604.14170)**

> **作者:** Qi Dong; Ziheng Lin; Ning Ding
>
> **摘要:** Retrieval-Augmented Generation (RAG) grounds Large Language Models (LLMs) in external knowledge but often suffers from flat context representations and stateless retrieval, leading to unstable performance. We propose Stateful Evidence-Driven RAG with Iterative Reasoning, a framework that models question answering as a progressive evidence accumulation process. Retrieved documents are converted into structured reasoning units with explicit relevance and confidence signals and maintained in a persistent evidence pool capturing both supportive and non-supportive information. The framework performs evidence-driven deficiency analysis to identify gaps and conflicts and iteratively refines queries to guide subsequent retrieval. This iterative reasoning process enables stable evidence aggregation and improves robustness to noisy retrieval. Experiments on multiple question answering benchmarks demonstrate consistent improvements over standard RAG and multi-step baselines, while effectively accumulating high-quality evidence and maintaining stable performance under substantial retrieval noise.
>
---
#### [new 052] Explain the Flag: Contextualizing Hate Speech Beyond Censorship
- **分类: cs.CL**

- **简介: 该论文属于 hate speech 检测任务，旨在解决自动检测中缺乏透明性和解释性的问题。通过结合大语言模型和自建词表，提出一种混合方法，实现对仇恨言论的识别与解释。**

- **链接: [https://arxiv.org/pdf/2604.14970](https://arxiv.org/pdf/2604.14970)**

> **作者:** Jason Liartis; Eirini Kaldeli; Lambrini Gyftokosta; Eleftherios Chelioudakis; Orfeas Menis Mastromichalakis
>
> **备注:** Accepted in the Findings of ACL 2026
>
> **摘要:** Hate, derogatory, and offensive speech remains a persistent challenge in online platforms and public discourse. While automated detection systems are widely used, most focus on censorship or removal, raising concerns for transparency and freedom of expression, and limiting opportunities to explain why content is harmful. To address these issues, explanatory approaches have emerged as a promising solution, aiming to make hate speech detection more transparent, accountable, and informative. In this paper, we present a hybrid approach that combines Large Language Models (LLMs) with three newly created and curated vocabularies to detect and explain hate speech in English, French, and Greek. Our system captures both inherently derogatory expressions tied to identity characteristics and direct group-targeted content through two complementary pipelines: one that detects and disambiguates problematic terms using the curated vocabularies, and one that leverages LLMs as context-aware evaluators of group-targeting content. The outputs are fused into grounded explanations that clarify why content is flagged. Human evaluation shows that our hybrid approach is accurate, with high-quality explanations, outperforming LLM-only baselines.
>
---
#### [new 053] LLM Predictive Scoring and Validation: Inferring Experience Ratings from Unstructured Text
- **分类: cs.CL**

- **简介: 该论文属于情感分析任务，旨在从文本预测用户体验评分。通过GPT-4.1分析球迷评论，预测其评分，并与实际评分对比，发现两者存在构念差异。**

- **链接: [https://arxiv.org/pdf/2604.14321](https://arxiv.org/pdf/2604.14321)**

> **作者:** Jason Potteiger; Andrew Hong; Ito Zapata
>
> **备注:** 29 pages, 5 figures, 6 tables
>
> **摘要:** We tasked GPT-4.1 to read what baseball fans wrote about their game-day experience and predict the overall experience rating each fan gave on a 0-10 survey scale. The model received only the text of a single open-ended response. These AI predictions were compared with the actual experience ratings captured by the survey instrument across approximately 10,000 fan responses from five Major League Baseball teams. In total two-thirds of predicted ratings fell within one point of self-reported fan ratings (67% within +/-1, 36% exact match), and the predicted measurement was near-deterministic across three independent scoring runs (87% exact agreement, 99.9% within +/-1). Predicted ratings aligned most strongly with the overall experience rating (r = 0.82) rather than with any specific aspect of the game-day experience such as parking, concessions, staff, etc. However, predictions were systematically lower than self-reported ratings by approximately one point, and this gap was not driven by any single aspect. Rather, our analysis shows that self-reported ratings capture the fan's verdict, an overall evaluative judgment that integrates the entire experience. While predicted ratings quantify the impact of salient moments characterized as memorable, emotionally intense, unusual, or actionable. Each measure contains information the other misses. These baseline results establish that a simple, unoptimized prompt can directionally predict how fans rate their experience from the text a fan wrote and that a gap between the two numbers can be interpreted as a construct difference worth preserving rather than an error to eliminate.
>
---
#### [new 054] The Cost of Language: Centroid Erasure Exposes and Exploits Modal Competition in Multimodal Language Models
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文研究多模态语言模型在视觉任务中的性能问题，通过centroid erasure揭示语言对视觉的主导作用，并提出对比解码方法提升准确率。**

- **链接: [https://arxiv.org/pdf/2604.14363](https://arxiv.org/pdf/2604.14363)**

> **作者:** Akshay Paruchuri; Ishan Chatterjee; Henry Fuchs; Ehsan Adeli; Piotr Didyk
>
> **备注:** 29 pages, 9 figures, 19 tables
>
> **摘要:** Multimodal language models systematically underperform on visual perception tasks, yet the structure underlying this failure remains poorly understood. We propose centroid replacement, collapsing each token to its nearest K-means centroid, as a controlled probe for modal dependence. Across seven models spanning three architecture families, erasing text centroid structure costs 4$\times$ more accuracy than erasing visual centroid structure, exposing a universal imbalance where language representations overshadow vision even on tasks that demand visual reasoning. We exploit this asymmetry through text centroid contrastive decoding, recovering up to +16.9% accuracy on individual tasks by contrastively decoding against a text-centroid-erased reference. This intervention varies meaningfully with training approaches: standard fine-tuned models show larger gains (+5.6% on average) than preference-optimized models (+1.5% on average). Our findings suggest that modal competition is structurally localized, correctable at inference time without retraining, and quantifiable as a diagnostic signal to guide future multimodal training.
>
---
#### [new 055] QU-NLP at ArchEHR-QA 2026: Two-Stage QLoRA Fine-Tuning of Qwen3-4B for Patient-Oriented Clinical Question Answering and Evidence Sentence Alignment
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于临床问答任务，解决患者导向的问答生成与证据句对齐问题。通过两阶段QLoRA微调Qwen3-4B模型，并融合检索方法提升性能。**

- **链接: [https://arxiv.org/pdf/2604.14175](https://arxiv.org/pdf/2604.14175)**

> **作者:** Mohammad AL-Smadi
>
> **备注:** Accepted for publication at CL4Health 2026 workshop, LREC2026 conference
>
> **摘要:** We present a unified system addressing both Subtask 3 (answer generation) and Subtask 4 (evidence sentence alignment) of the ArchEHR-QA Shared Task. For Subtask 3, we apply two-stage Quantised Low-Rank Adaptation (QLoRA) to Qwen3-4B loaded in 4-bit NF4 quantisation: first on 30,000 samples from the emrQA-MedSQuAD corpus to establish clinical domain competence, then on the 20 annotated development cases to learn the task-specific output style. Our system achieves an overall score of 32.87 on the official test-2026 split (BLEU = 9.42, ROUGE-L = 27.04, SARI = 55.42, BERTScore = 43.00, AlignScore = 25.28, MEDCON = 37.04). For Subtask 4, we develop a weighted ensemble of three retrieval methods - BM25 with relative thresholding, TF-IDF cosine similarity, and a fine-tuned cross-encoder - to identify note sentences supporting a given gold answer, achieving a micro-F1 of 67.16 on the 100-case test set. Experiments reveal that both subtasks expose the same fundamental challenge: 20 annotated training cases are insufficient to distinguish relevant from irrelevant clinical sentences, pointing to data augmentation as the highest-leverage future direction.
>
---
#### [new 056] BiCon-Gate: Consistency-Gated De-colloquialisation for Dialogue Fact-Checking
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话事实核查任务，解决口语化表达影响核查的问题。通过分阶段去口语化和语义一致性门控选择重写候选，提升核查效果。**

- **链接: [https://arxiv.org/pdf/2604.14389](https://arxiv.org/pdf/2604.14389)**

> **作者:** Hyunkyung Park; Arkaitz Zubiaga
>
> **备注:** 15 pages, 7 figures. Published in FEVER 2026
>
> **摘要:** Automated fact-checking in dialogue involves multi-turn conversations where colloquial language is frequent yet understudied. To address this gap, we propose a conservative rewrite candidate for each response claim via staged de-colloquialisation, combining lightweight surface normalisation with scoped in-claim coreference resolution. We then introduce BiCon-Gate, a semantics-aware consistency gate that selects the rewrite candidate only when it is semantically supported by the dialogue context, otherwise falling back to the original claim. This gated selection stabilises downstream fact-checking and yields gains in both evidence retrieval and fact verification. On the DialFact benchmark, our approach improves retrieval and verification, with particularly strong gains on SUPPORTS, and outperforms competitive baselines, including a decoder-based one-shot LLM rewrite that attempts to perform all de-colloquialisation steps in a single pass.
>
---
#### [new 057] SeaAlert: Critical Information Extraction From Maritime Distress Communications with Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于海上遇险通信信息提取任务，旨在解决真实数据稀缺和ASR噪声问题。通过生成合成数据并构建分析框架SeaAlert提升分析鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.14163](https://arxiv.org/pdf/2604.14163)**

> **作者:** Tomer Atia; Yehudit Aperstein; Alexander Apartsin
>
> **备注:** 12 pages, 8 figures
>
> **摘要:** Maritime distress communications transmitted over very high frequency (VHF) radio are safety-critical voice messages used to report emergencies at sea. Under the Global Maritime Distress and Safety System (GMDSS), such messages follow standardized procedures and are expected to convey essential details, including vessel identity, position, nature of the distress, and required assistance. In practice, however, automatic analysis remains difficult because distress messages are often brief, noisy, and produced under stress, may deviate from the prescribed format, and are further degraded by automatic speech recognition (ASR) errors caused by channel noise and speaker stress. This paper presents SeaAlert, an LLM-based framework for robust analysis of maritime distress communications. To address the scarcity of labeled real-world data, we develop a synthetic data generation pipeline in which an LLM produces realistic and diverse maritime messages, including challenging variants in which standard distress codewords are omitted or replaced with less explicit expressions. The generated utterances are synthesized into speech, degraded with simulated VHF noise, and transcribed by an ASR system to obtain realistic noisy transcripts.
>
---
#### [new 058] Tug-of-War within A Decade: Conflict Resolution in Vulnerability Analysis via Teacher-Guided Retrieval-Augmented Generations
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于漏洞分析任务，解决CVE知识冲突问题。通过改进检索和教师引导的生成方法，提升LLMs在漏洞检测中的准确性与一致性。**

- **链接: [https://arxiv.org/pdf/2604.14172](https://arxiv.org/pdf/2604.14172)**

> **作者:** Ziyin Zhou; Jianyi Zhang; Xu ji; Yilong Li; Jiameng Han; Zhangchi Zhao
>
> **摘要:** Large Language Models (LLMs) are essential for analyzing and addressing vulnerabilities in cybersecurity. However, among over 200,000 vulnerabilities were discovered in the past decade, more than 30,000 have been changed or updated. This necessitates frequent updates to the training datasets and internal knowledge bases of LLMs to maintain knowledge consistency. In this paper, we focus on the problem of knowledge discrepancy and conflict within CVE (Common Vulnerabilities and Exposures) detection and analysis. This problem hinders LLMs' ability to retrieve the latest knowledge from original training datasets, leading to knowledge conflicts, fabrications of factually incorrect results, and generation hallucinations. To address this problem, we propose an innovative two-stage framework called CRVA-TGRAG (Conflict Resolution in Vulnerability Analysis via Teacher-Guided Retrieval-Augmented Generation). First, to improve document retrieval accuracy during the retrieval stage, we utilize Parent Document Segmentation and an ensemble retrieval scheme based on semantic similarity and inverted indexing. Second, to enhance LLMs' capabilities based on the retrieval of CVE dataset in generation stage, we employ a teacher-guided preference optimization technique to fine-tune LLMs. Our framework not only enhances the quality of content retrieval through RAG but also leverages the advantages of preference fine-tuning in LLMs to answer questions more effectively and precisely. Experiments demonstrate our method achieves higher accuracy in retrieving the latest CVEs compared to external knowledge bases. In conclusion, our framework significantly mitigates potential knowledge conflicts and inconsistencies that may arise from relying solely on LLMs for knowledge retrieval.
>
---
#### [new 059] IUQ: Interrogative Uncertainty Quantification for Long-Form Large Language Model Generation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言生成任务，旨在解决长文本生成中的不确定性量化问题。提出IUQ框架，通过一致性与忠实性评估模型输出的可信度。**

- **链接: [https://arxiv.org/pdf/2604.15109](https://arxiv.org/pdf/2604.15109)**

> **作者:** Haozhi Fan; Jinhao Duan; Kaidi Xu
>
> **摘要:** Despite the rapid advancement of Large Language Models (LLMs), uncertainty quantification in LLM generation is a persistent challenge. Although recent approaches have achieved strong performance by restricting LLMs to produce short or constrained answer sets, many real-world applications require long-form and free-form text generation. A key difficulty in this setting is that LLMs often produce responses that are semantically coherent yet factually inaccurate, while the underlying semantics are multifaceted and the linguistic structure is complex. To tackle this challenge, this paper introduces Interrogative Uncertainty Quantification (IUQ), a novel framework that leverages inter-sample consistency and intra-sample faithfulness to quantify the uncertainty in long-form LLM outputs. By utilizing an interrogate-then-respond paradigm, our method provides reliable measures of claim-level uncertainty and the model's faithfulness. Experimental results across diverse model families and model sizes demonstrate the superior performance of IUQ over two widely used long-form generation datasets. The code is available at this https URL.
>
---
#### [new 060] Chronological Knowledge Retrieval: A Retrieval-Augmented Generation Approach to Construction Project Documentation
- **分类: cs.CL**

- **简介: 该论文属于信息检索任务，旨在解决施工项目决策历史查询问题。通过RAG框架结合语义搜索与大模型，实现对会议纪要的时序问答，提升决策追溯效率。**

- **链接: [https://arxiv.org/pdf/2604.14169](https://arxiv.org/pdf/2604.14169)**

> **作者:** Ioannis-Aris Kostis; Natalia Sanchiz; Steeve De Schryver; François Denis; Pierre Schaus
>
> **摘要:** In large-scale construction projects, the continuous evolution of decisions generates extensive records, most often captured in meeting minutes. Since decisions may override previous ones, professionals often need to reconstruct the history of specific choices. Retrieving such information manually from raw archives is both labor-intensive and error-prone. From a user perspective, we address this challenge by enabling conversational access to the whole set of project meeting minutes. Professionals can pose natural-language questions and receive answers that are both semantically relevant and explicitly time-annotated, allowing them to follow the chronology of decisions. From a technical perspective, our solution employs a Retrieval-Augmented Generation (RAG) framework that integrates semantic search with large language models to ensure accurate and context-aware responses. We demonstrate the approach using an anonymized, industry-sourced dataset of meeting minutes from a completed construction project by a large company in Belgium. The dataset is annotated and enriched with expert-defined queries to support systematic evaluation. Both the dataset and the open-source implementation are made available to the community to foster further research on conversational access to time-annotated project documentation.
>
---
#### [new 061] CURaTE: Continual Unlearning in Real Time with Ensured Preservation of LLM Knowledge
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于知识遗忘任务，解决模型训练后需实时删除特定知识的问题。提出CURaTE方法，通过句子嵌入模型判断是否遗忘，实现高效且持续的实时遗忘。**

- **链接: [https://arxiv.org/pdf/2604.14644](https://arxiv.org/pdf/2604.14644)**

> **作者:** Seyun Bae; Seokhan Lee; Eunho Yang
>
> **备注:** Accepted to Findings of ACL 2026
>
> **摘要:** The inability to filter out in advance all potentially problematic data from the pre-training of large language models has given rise to the need for methods for unlearning specific pieces of knowledge after training. Existing techniques overlook the need for continuous and immediate action, causing them to suffer from degraded utility as updates accumulate and protracted exposure of sensitive information. To address these issues, we propose Continual Unlearning in Real Time with Ensured Preservation of LLM Knowledge (CURaTE). Our method begins by training a sentence embedding model on a dataset designed to enable the formation of sharp decision boundaries for determining whether a given input prompt corresponds to any stored forget requests. The similarity of a given input to the forget requests is then used to determine whether to answer or return a refusal response. We show that even with such a simple approach, not only does CURaTE achieve more effective forgetting than existing methods, but by avoiding modification of the language model parameters, it also maintains near perfect knowledge preservation over any number of updates and is the only method capable of continual unlearning in real-time.
>
---
#### [new 062] Comparison of Modern Multilingual Text Embedding Techniques for Hate Speech Detection Task
- **分类: cs.CL; cs.LG**

- **简介: 论文研究多语言文本嵌入技术在仇恨言论检测中的应用，解决多语言环境下仇恨内容识别问题。构建了立陶宛语数据集LtHate，对比多种模型性能，验证监督学习效果优于异常检测。**

- **链接: [https://arxiv.org/pdf/2604.14907](https://arxiv.org/pdf/2604.14907)**

> **作者:** Evaldas Vaiciukynas; Paulius Danenas; Linas Ablonskis; Algirdas Sukys; Edgaras Dambrauskas; Voldemaras Zitkus; Rita Butkiene; Rimantas Butleris
>
> **备注:** Submitted to Applied Soft Computing (Status: Decision in Process)
>
> **摘要:** Online hate speech and abusive language pose a growing challenge for content moderation, especially in multilingual settings and for low-resource languages such as Lithuanian. This paper investigates to what extent modern multilingual sentence embedding models can support accurate hate speech detection in Lithuanian, Russian, and English, and how their performance depends on downstream modeling choices and feature dimensionality. We introduce LtHate, a new Lithuanian hate speech corpus derived from news portals and social networks, and benchmark six modern multilingual encoders (potion, gemma, bge, snow, jina, e5) on LtHate, RuToxic, and EnSuperset using a unified Python pipeline. For each embedding, we train both a one class HBOS anomaly detector and a two class CatBoost classifier, with and without principal component analysis (PCA) compression to 64-dimensional feature vectors. Across all datasets, two class supervised models consistently and substantially outperform one class anomaly detection, with the best configurations achieving up to 80.96% accuracy and AUC ROC of 0.887 in Lithuanian (jina), 92.19% accuracy and AUC ROC of 0.978 in Russian (e5), and 77.21% accuracy and AUC ROC of 0.859 in English (e5 with PCA). PCA compression preserves almost all discriminative power in the supervised setting, while showing some negative impact for the unsupervised anomaly detection case. These results demonstrate how modern multilingual sentence embeddings combined with gradient boosted decision trees provide robust soft-computing solutions for multilingual hate speech detection applications.
>
---
#### [new 063] XQ-MEval: A Dataset with Cross-lingual Parallel Quality for Benchmarking Translation Metrics
- **分类: cs.CL**

- **简介: 该论文属于机器翻译评估任务，旨在解决跨语言评分偏差问题。通过构建XQ-MEval数据集，分析翻译指标的不一致性，并提出归一化策略提升评估公平性。**

- **链接: [https://arxiv.org/pdf/2604.14934](https://arxiv.org/pdf/2604.14934)**

> **作者:** Jingxuan Liu; Zhi Qu; Jin Tei; Hidetaka Kamigaito; Lemao Liu; Taro Watanabe
>
> **备注:** 19 pages, 8 figures, ACL 2026 Findings
>
> **摘要:** Automatic evaluation metrics are essential for building multilingual translation systems. The common practice of evaluating these systems is averaging metric scores across languages, yet this is suspicious since metrics may suffer from cross-lingual scoring bias, where translations of equal quality receive different scores across languages. This problem has not been systematically studied because no benchmark exists that provides parallel-quality instances across languages, and expert annotation is not realistic. In this work, we propose XQ-MEval, a semi-automatically built dataset covering nine translation directions, to benchmark translation metrics. Specifically, we inject MQM-defined errors into gold translations automatically, filter them by native speakers for reliability, and merge errors to generate pseudo translations with controllable quality. These pseudo translations are then paired with corresponding sources and references to form triplets used in assessing the qualities of translation metrics. Using XQ-MEval, our experiments on nine representative metrics reveal the inconsistency between averaging and human judgment and provide the first empirical evidence of cross-lingual scoring bias. Finally, we propose a normalization strategy derived from XQ-MEval that aligns score distributions across languages, improving the fairness and reliability of multilingual metric evaluation.
>
---
#### [new 064] Three-Phase Transformer
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出一种三通道Transformer结构，用于语言建模任务，通过通道划分和旋转操作提升模型性能与收敛速度。**

- **链接: [https://arxiv.org/pdf/2604.14430](https://arxiv.org/pdf/2604.14430)**

> **作者:** Mohammad R. Abu Ayyash
>
> **备注:** 48 pages, 20 figures, 23 tables. Code: this https URL
>
> **摘要:** We present Three-Phase Transformer (3PT), a residual-stream structural prior for decoder-only Transformers on a standard SwiGLU + RMSNorm + RoPE + GQA backbone. The hidden vector is partitioned into N equally-sized cyclic channels, each maintained by phase-respecting ops: a per-channel RMSNorm, a 2D Givens rotation between attention and FFN that rotates each channel by theta + i*(2*pi/N), and a head-count constraint aligning GQA heads with the partition. The architecture is a self-stabilizing equilibrium between scrambling and re-imposition, not a bolted-on module. The partition carves out a one-dimensional DC subspace orthogonal to the channels, into which we inject a fixed Gabriel's horn profile r(p) = 1/(p+1) as an absolute-position side-channel composing orthogonally with RoPE's relative-position rotation. The canonical N=3 borrows its metaphor from balanced three-phase AC, where three sinusoids 120 degrees apart sum to zero with no anti-correlated pair. At 123M parameters on WikiText-103, 3PT achieves -7.20% perplexity (-2.62% bits-per-byte) over a matched RoPE-Only baseline at +1,536 parameters (0.00124% of total), with 1.93x step-count convergence speedup (1.64x wall-clock). N behaves as a parameter-sharing knob rather than a unique optimum: at 5.5M an N-sweep over {1,2,3,4,6,8,12} is near-monotone with N=1 winning; at 123M a three-seed sweep finds N=3 and N=1 statistically indistinguishable. The load-bearing mechanism is the channel-partitioned residual stream, per-block rotation, per-phase normalization, and horn DC injection. We characterize (a) self-stabilization of the geometry without explicit enforcement, a novel instance of the conservation-law framework for neural networks; (b) a U-shaped depth profile of rotation-angle drift at 12 layers; (c) orthogonal composition with RoPE, attention, and FFN.
>
---
#### [new 065] An Underexplored Frontier: Large Language Models for Rare Disease Patient Education and Communication -- A scoping review
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗信息学任务，旨在探讨大语言模型在罕见病患者教育与沟通中的应用。通过文献综述，分析现有研究的特征与不足，提出未来研究方向。**

- **链接: [https://arxiv.org/pdf/2604.14179](https://arxiv.org/pdf/2604.14179)**

> **作者:** Zaifu Zhan; Yu Hou; Kai Yu; Min Zeng; Anita Burgun; Xiaoyi Chen; Rui Zhang
>
> **摘要:** Rare diseases affect over 300 million people worldwide and are characterized by complex care pathways, limited clinical expertise, and substantial unmet communication needs throughout the long patient journey. Recent advances in large language models (LLMs) offer new opportunities to support patient education and communication, yet their application in rare diseases remains unclear. We conducted a scoping review of studies published between January 2022 and March 2026 across major databases, identifying 12 studies on LLM-based rare disease patient education and communication. Data were extracted on study characteristics, application scenarios, model usage, and evaluation methods, and synthesized using descriptive and qualitative analyses. The literature is highly recent and dominated by general-purpose models, particularly ChatGPT. Most studies focus on patient question answering using curated question sets, with limited use of real-world data or longitudinal communication scenarios. Evaluations are primarily centered on accuracy, with limited attention to patient-centered dimensions such as readability, empathy, and communication quality. Multilingual communication is rarely addressed. Overall, the field remains at an early stage. Future research should prioritize patient-centered design, domain-adapted methods, and real-world deployment to support safe, adaptive, and effective communication in rare diseases.
>
---
#### [new 066] APEX-MEM: Agentic Semi-Structured Memory with Temporal Reasoning for Long-Term Conversational AI
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文提出APEX-MEM，解决长对话中可靠记忆问题。通过属性图、追加存储和多工具检索代理，实现时间一致的对话记忆管理。**

- **链接: [https://arxiv.org/pdf/2604.14362](https://arxiv.org/pdf/2604.14362)**

> **作者:** Pratyay Banerjee; Masud Moshtaghi; Shivashankar Subramanian; Amita Misra; Ankit Chadha
>
> **备注:** Accepted to ACL 2026 Mains
>
> **摘要:** Large language models still struggle with reliable long-term conversational memory: simply enlarging context windows or applying naive retrieval often introduces noise and destabilizes responses. We present APEX-MEM, a conversational memory system that combines three key innovations: (1) a property graph which uses domain-agnostic ontology to structure conversations as temporally grounded events in an entity-centric framework, (2) append-only storage that preserves the full temporal evolution of information, and (3) a multi-tool retrieval agent that understands and resolves conflicting or evolving information at query time, producing a compact and contextually relevant memory summary. This retrieval-time resolution preserves the full interaction history while suppressing irrelevant details. APEX-MEM achieves 88.88% accuracy on LOCOMO's Question Answering task and 86.2% on LongMemEval, outperforming state-of-the-art session-aware approaches and demonstrating that structured property graphs enable more temporally coherent long-term conversational reasoning.
>
---
#### [new 067] How to Fine-Tune a Reasoning Model? A Teacher-Student Cooperation Framework to Synthesize Student-Consistent SFT Data
- **分类: cs.CL**

- **简介: 该论文属于模型微调任务，解决教师生成数据与学生风格不一致导致性能下降的问题，提出TESSY框架实现风格一致性合成数据。**

- **链接: [https://arxiv.org/pdf/2604.14164](https://arxiv.org/pdf/2604.14164)**

> **作者:** Zixian Huang; Kaichen Yang; Xu Huang; Feiyang Hao; Qiming Ge; Bowen Li; He Du; Kai Chen; Qipeng Guo
>
> **摘要:** A widely adopted strategy for model enhancement is to use synthetic data generated by a stronger model for supervised fine-tuning (SFT). However, for emerging reasoning models like Qwen3-8B, this approach often fails to improve reasoning capabilities and can even lead to a substantial drop in performance. In this work, we identify substantial stylistic divergence between teacher generated data and the distribution of student as a major factor impacting SFT. To bridge this gap, we propose a Teacher-Student Cooperation Data Synthesis framework (TESSY), which interleaves teacher and student models to alternately generate style and non-style tokens. Consequently, TESSY produces synthetic sequences that inherit the advanced reasoning capabilities of the teacher while maintaining stylistic consistency with the distribution of the student. In experiments on code generation using GPT-OSS-120B as the teacher, fine-tuning Qwen3-8B on teacher-generated data leads to performance drops of 3.25% on LiveCodeBench-Pro and 10.02% on OJBench, whereas TESSY achieves improvements of 11.25% and 6.68%.
>
---
#### [new 068] ReviewGrounder: Improving Review Substantiveness with Rubric-Guided, Tool-Integrated Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的文本生成任务，旨在解决AI评审缺乏实质性反馈的问题。通过引入基于评分标准的多智能体框架，提升评审质量。**

- **链接: [https://arxiv.org/pdf/2604.14261](https://arxiv.org/pdf/2604.14261)**

> **作者:** Zhuofeng Li; Yi Lu; Dongfu Jiang; Haoxiang Zhang; Yuyang Bai; Chuan Li; Yu Wang; Shuiwang Ji; Jianwen Xie; Yu Zhang
>
> **摘要:** The rapid rise in AI conference submissions has driven increasing exploration of large language models (LLMs) for peer review support. However, LLM-based reviewers often generate superficial, formulaic comments lacking substantive, evidence-grounded feedback. We attribute this to the underutilization of two key components of human reviewing: explicit rubrics and contextual grounding in existing work. To address this, we introduce REVIEWBENCH, a benchmark evaluating review text according to paper-specific rubrics derived from official guidelines, the paper's content, and human-written reviews. We further propose REVIEWGROUNDER, a rubric-guided, tool-integrated multi-agent framework that decomposes reviewing into drafting and grounding stages, enriching shallow drafts via targeted evidence consolidation. Experiments on REVIEWBENCH show that REVIEWGROUNDER, using a Phi-4-14B-based drafter and a GPT-OSS-120B-based grounding stage, consistently outperforms baselines with substantially stronger/larger backbones (e.g., GPT-4.1 and DeepSeek-R1-670B) in both alignment with human judgments and rubric-based review quality across 8 dimensions. The code is available \href{this https URL}{here}.
>
---
#### [new 069] Fact4ac at the Financial Misinformation Detection Challenge Task: Reference-Free Financial Misinformation Detection via Fine-Tuning and Few-Shot Prompting of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于金融虚假信息检测任务，旨在解决无外部证据下的金融谣言识别问题。通过微调和少量提示策略，提升大模型的检测能力，取得优异成绩。**

- **链接: [https://arxiv.org/pdf/2604.14640](https://arxiv.org/pdf/2604.14640)**

> **作者:** Cuong Hoang; Le-Minh Nguyen
>
> **摘要:** The proliferation of financial misinformation poses a severe threat to market stability and investor trust, misleading market behavior and creating critical information asymmetry. Detecting such misleading narratives is inherently challenging, particularly in real-world scenarios where external evidence or supplementary references for cross-verification are strictly unavailable. This paper presents our winning methodology for the "Reference-Free Financial Misinformation Detection" shared task. Built upon the recently proposed RFC-BENCH framework (Jiang et al. 2026), this task challenges models to determine the veracity of financial claims by relying solely on internal semantic understanding and contextual consistency, rather than external fact-checking. To address this formidable evaluation setup, we propose a comprehensive framework that capitalizes on the reasoning capabilities of state-of-the-art Large Language Models (LLMs). Our approach systematically integrates in-context learning, specifically zero-shot and few-shot prompting strategies, with Parameter-Efficient Fine-Tuning (PEFT) via Low-Rank Adaptation (LoRA) to optimally align the models with the subtle linguistic cues of financial manipulation. Our proposed system demonstrated superior efficacy, successfully securing the first-place ranking on both official leaderboards. Specifically, we achieved an accuracy of 95.4% on the public test set and 96.3% on the private test set, highlighting the robustness of our method and contributing to the acceleration of context-aware misinformation detection in financial Natural Language Processing. Our models (14B and 32B) are available at this https URL.
>
---
#### [new 070] Mechanistic Decoding of Cognitive Constructs in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI可解释性任务，旨在解析LLMs中复杂情感的内部机制。通过构建框架，分析社会比较嫉妒的认知结构，揭示其心理触发因素及影响。**

- **链接: [https://arxiv.org/pdf/2604.14593](https://arxiv.org/pdf/2604.14593)**

> **作者:** Yitong Shou; Manhao Guan
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** While Large Language Models (LLMs) demonstrate increasingly sophisticated affective capabilities, the internal mechanisms by which they process complex emotions remain unclear. Existing interpretability approaches often treat models as black boxes or focus on coarse-grained basic emotions, leaving the cognitive structure of more complex affective states underexplored. To bridge this gap, we propose a Cognitive Reverse-Engineering framework based on Representation Engineering (RepE) to analyze social-comparison jealousy. By combining appraisal theory with subspace orthogonalization, regression-based weighting, and bidirectional causal steering, we isolate and quantify two psychological antecedents of jealousy, Superiority of Comparison Person and Domain Self-Definitional Relevance, and examine their causal effects on model judgments. Experiments on eight LLMs from the Llama, Qwen, and Gemma families suggest that models natively encode jealousy as a structured linear combination of these constituent factors. Their internal representations are broadly consistent with the human psychological construct, treating Superiority as the foundational trigger and Relevance as the ultimate intensity multiplier. Our framework also demonstrates that toxic emotional states can be mechanically detected and surgically suppressed, suggesting a possible route toward representational monitoring and intervention for AI safety in multi-agent environments.
>
---
#### [new 071] Reasoning Dynamics and the Limits of Monitoring Modality Reliance in Vision-Language Models
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **简介: 该论文研究视觉语言模型的推理动态，探讨多模态信息整合问题。通过分析18个模型，揭示其对文本线索的依赖及推理过程中的偏差。**

- **链接: [https://arxiv.org/pdf/2604.14888](https://arxiv.org/pdf/2604.14888)**

> **作者:** Danae Sánchez Villegas; Samuel Lewis-Lim; Nikolaos Aletras; Desmond Elliott
>
> **摘要:** Recent advances in vision language models (VLMs) offer reasoning capabilities, yet how these unfold and integrate visual and textual information remains unclear. We analyze reasoning dynamics in 18 VLMs covering instruction-tuned and reasoning-trained models from two different model families. We track confidence over Chain-of-Thought (CoT), measure the corrective effect of reasoning, and evaluate the contribution of intermediate reasoning steps. We find that models are prone to answer inertia, in which early commitments to a prediction are reinforced, rather than revised during reasoning steps. While reasoning-trained models show stronger corrective behavior, their gains depend on modality conditions, from text-dominant to vision-only settings. Using controlled interventions with misleading textual cues, we show that models are consistently influenced by these cues even when visual evidence is sufficient, and assess whether this influence is recoverable from CoT. Although this influence can appear in the CoT, its detectability varies across models and depends on what is being monitored. Reasoning-trained models are more likely to explicitly refer to the cues, but their longer and fluent CoTs can still appear visually grounded while actually following textual cues, obscuring modality reliance. In contrast, instruction-tuned models refer to the cues less explicitly, but their shorter traces reveal inconsistencies with the visual input. Taken together, these findings indicate that CoT provides only a partial view of how different modalities drive VLM decisions, with important implications for the transparency and safety of multimodal systems.
>
---
#### [new 072] The Autocorrelation Blind Spot: Why 42% of Turn-Level Findings in LLM Conversation Analysis May Be Spurious
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的评估任务，旨在解决LLM对话分析中因自相关导致的统计误判问题。研究发现42%的显著性结果可能不真实，并提出修正框架。**

- **链接: [https://arxiv.org/pdf/2604.14414](https://arxiv.org/pdf/2604.14414)**

> **作者:** Ferdinand M. Schessl
>
> **备注:** 14 pages, 3 figures, 5 tables, 1 algorithm. Code and synthetic demonstration data: this https URL
>
> **摘要:** Turn-level metrics are widely used to evaluate properties of multi-turn human-LLM conversations, from safety and sycophancy to dialogue quality. However, consecutive turns within a conversation are not statistically independent -- a fact that virtually all current evaluation pipelines fail to correct for in their statistical inference. We systematically characterize the autocorrelation structure of 66 turn-level metrics across 202 multi-turn conversations (11,639 turn pairs, 5 German-speaking users, 4 LLM platforms) and demonstrate that naive pooled analysis produces severely inflated significance estimates: 42% of associations that appear significant under standard pooled testing fail to survive cluster-robust correction. The inflation varies substantially across categories rather than scaling linearly with autocorrelation: three memoryless families (embedding velocity, directional, differential) aggregate to 14%, while the seven non-memoryless families (thermo-cycle, frame distance, lexical/structural, rolling windows, cumulative, interaction, timestamp) aggregate to 33%, with individual category rates ranging from 0% to 100% depending on per-family effect size. We present a two-stage correction framework combining Chelton (1983) effective degrees of freedom with conversation-level block bootstrap, and validate it on a pre-registered hold-out split where cluster-robust metrics replicate at 57% versus 30% for pooled-only metrics. We provide concrete design principles, a publication checklist, and open-source code for the correction pipeline. A survey of ~30 recent papers at major NLP and AI venues that compute turn-level statistics in LLM evaluations finds that only 4 address temporal dependence at all, and 26 do not correct for it.
>
---
#### [new 073] Compressing Sequences in the Latent Embedding Space: $K$-Token Merging for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，解决长文本处理中计算成本高的问题。通过K-Token Merging在嵌入空间压缩token，减少计算量同时保持性能。**

- **链接: [https://arxiv.org/pdf/2604.15153](https://arxiv.org/pdf/2604.15153)**

> **作者:** Zihao Xu; John Harvill; Ziwei Fan; Yizhou Sun; Hao Ding; Hao Wang
>
> **备注:** Under Review
>
> **摘要:** Large Language Models (LLMs) incur significant computational and memory costs when processing long prompts, as full self-attention scales quadratically with input length. Token compression aims to address this challenge by reducing the number of tokens representing inputs. However, existing prompt-compression approaches primarily operate in token space and overlook inefficiencies in the latent embedding space. In this paper, we propose K-Token Merging, a latent-space compression framework that merges each contiguous block of K token embeddings into a single embedding via a lightweight encoder. The compressed sequence is processed by a LoRA-adapted LLM, while generation remains in the original vocabulary. Experiments on structural reasoning (Textualized Tree), sentiment classification (Amazon Reviews), and code editing (CommitPackFT) show that K-Token Merging lies on the Pareto frontier of performance vs. compression, achieving up to 75% input length reduction with minimal performance degradation.
>
---
#### [new 074] Blinded Multi-Rater Comparative Evaluation of a Large Language Model and Clinician-Authored Responses in CGM-Informed Diabetes Counseling
- **分类: cs.CL**

- **简介: 该论文属于医疗咨询任务，旨在评估大型语言模型在糖尿病CGM咨询中的表现。研究比较了模型生成与医生撰写的响应质量，发现模型在情感和可操作性上更优。**

- **链接: [https://arxiv.org/pdf/2604.15124](https://arxiv.org/pdf/2604.15124)**

> **作者:** Zhijun Guo; Alvina Lai; Emmanouil Korakas; Aristeidis Vagenas; Irshad Ahamed; Christo Albor; Hengrui Zhang; Justin Healy; Kezhi Li
>
> **摘要:** Continuous glucose monitoring (CGM) is central to diabetes care, but explaining CGM patterns clearly and empathetically remains time-intensive. Evidence for retrieval-grounded large language model (LLM) systems in CGM-informed counseling remains limited. To evaluate whether a retrieval-grounded LLM-based conversational agent (CA) could support patient understanding of CGM data and preparation for routine diabetes consultations. We developed a retrieval-grounded LLM-based CA for CGM interpretation and diabetes counseling support. The system generated plain-language responses while avoiding individualized therapeutic advice. Twelve CGM-informed cases were constructed from publicly available datasets. Between Oct 2025 and Feb 2026, 6 senior UK diabetes clinicians each reviewed 2 assigned cases and answered 24 questions. In a blinded multi-rater evaluation, each CA-generated and clinician-authored response was independently rated by 3 clinicians on 6 quality dimensions. Safety flags and perceived source labels were also recorded. Primary analyses used linear mixed-effects models. A total of 288 unique responses (144 CA and 144 clinician) generated 864 ratings. The CA received higher quality scores than clinician responses (mean 4.37 vs 3.58), with an estimated mean difference of 0.782 points (95% CI 0.692-0.872; P<.001). The largest differences were for empathy (1.062, 95% CI 0.948-1.177) and actionability (0.992, 95% CI 0.877-1.106). Safety flag distributions were similar, with major concerns rare in both groups (3/432, 0.7% each). Retrieval-grounded LLM systems may have value as adjunct tools for CGM review, patient education, and preconsultation preparation. However, these findings do not support autonomous therapeutic decision-making or unsupervised real-world use.
>
---
#### [new 075] SAGE Celer 2.6 Technical Card
- **分类: cs.CL; cs.AI**

- **简介: 本文介绍SAGE Celer 2.6模型，解决多语言支持与复杂推理问题。通过架构优化和预训练，提升数学、编码及南亚语言性能。**

- **链接: [https://arxiv.org/pdf/2604.14168](https://arxiv.org/pdf/2604.14168)**

> **作者:** SAGEA Research Team; Basab Jha; Firoj Paudel; Ujjwal Puri; Adrian Liu; Ethan Henkel; Zhang Yuting; Mateusz Kowalczyk; Mei Huang; Choi Donghyuk; Wang Junhao
>
> **备注:** 28 pages, 14 figures
>
> **摘要:** We introduce SAGE Celer 2.6, the latest in our line of general-purpose Celer models from SAGEA. Celer 2.6 is available in 5B, 10B, and 27B parameter sizes and benefits from extensive architectural modifications and further pre-training on an undisclosed model. Using our Inverse Reasoning (IR) pipeline, SAGEA natively trains Celer 2.6 to validate its own logic paths, minimizing cascading error and hallucination in complex reasoning tasks. Celer 2.6 also boasts natively integrated multimodal functionality with an end-to-end vision encoder to avoid common pitfalls in adapter-based approaches. Celer 2.6 provides highly competitive results on mathematics, coding, and general intelligence benchmarks (ACUMEN), along with low latency. Most importantly, Celer 2.6 is specifically optimized for South Asian language support, with a custom tokenizer for the Devanagari script and strong performance in both Nepali and Hindi without sacrificing English reasoning ability.
>
---
#### [new 076] EviSearch: A Human in the Loop System for Extracting and Auditing Clinical Evidence for Systematic Reviews
- **分类: cs.CL**

- **简介: 该论文提出EviSearch，用于自动化提取和审计临床证据，解决系统综述中的信息提取问题。通过多代理系统实现高精度、可追溯的证据表生成。**

- **链接: [https://arxiv.org/pdf/2604.14165](https://arxiv.org/pdf/2604.14165)**

> **作者:** Naman Ahuja; Saniya Mulla; Muhammad Ali Khan; Zaryab Bin Riaz; Kaneez Zahra Rubab Khakwani; Mohamad Bassam Sonbol; Irbaz Bin Riaz; Vivek Gupta
>
> **摘要:** We present EviSearch, a multi-agent extraction system that automates the creation of ontology-aligned clinical evidence tables directly from native trial PDFs while guaranteeing per-cell provenance for audit and human verification. EviSearch pairs a PDF-query agent (which preserves rendered layout and figures) with a retrieval-guided search agent and a reconciliation module that forces page-level verification when agents disagree. The pipeline is designed for high-precision extraction across multimodal evidence sources (text, tables, figures) and for generating reviewer-actionable provenance that clinicians can inspect and correct. On a clinician-curated benchmark of oncology trial papers, EviSearch substantially improves extraction accuracy relative to strong parsed-text baselines while providing comprehensive attribution coverage. By logging reconciler decisions and reviewer edits, the system produces structured preference and supervision signals that bootstrap iterative model improvement. EviSearch is intended to accelerate living systematic review workflows, reduce manual curation burden, and provide a safe, auditable path for integrating LLM-based extraction into evidence synthesis pipelines.
>
---
#### [new 077] Chinese Language Is Not More Efficient Than English in Vibe Coding: A Preliminary Study on Token Cost and Problem-Solving Rate
- **分类: cs.CL; cs.SE**

- **简介: 该论文属于自然语言处理任务，旨在验证中文提示是否比英文更高效。研究通过实验发现中文在token成本和任务成功率上不占优势，结论表明语言对效率的影响依赖模型。**

- **链接: [https://arxiv.org/pdf/2604.14210](https://arxiv.org/pdf/2604.14210)**

> **作者:** Simiao Ren; Xingyu Shen; Yuchen Zhou; Dennis; Ankit Raj
>
> **摘要:** A claim has been circulating on social media and practitioner forums that Chinese prompts are more token-efficient than English for LLM coding tasks, potentially reducing costs by up to 40\%. This claim has influenced developers to consider switching to Chinese for ``vibe coding'' to save on API costs. In this paper, we conduct a rigorous empirical study using SWE-bench Lite, a benchmark of software engineering tasks, to evaluate whether this claim of Chinese token efficiency holds up to scrutiny. Our results reveal three key findings: First, the efficiency advantage of Chinese is not observed. Second, token cost varies by model architecture in ways that defy simple assumptions: while MiniMax-2.7 shows 1.28x higher token costs for Chinese, GLM-5 actually consumes fewer tokens with Chinese prompts. Third, and most importantly, we found that the success rate when prompting in Chinese is generally lower than in English across all models we tested. We also measure cost efficiency as expected cost per successful task -- jointly accounting for token consumption and task resolution rate. These findings should be interpreted as preliminary evidence rather than a definitive conclusion, given the limited number of models evaluated and the narrow set of benchmarks tested due to resource constraints; they indicate that language effects on token cost are model-dependent, and that practitioners should not expect cost savings or performance gains just by switching their prompt language to Chinese.
>
---
#### [new 078] Exploring and Testing Skill-Based Behavioral Profile Annotation: Human Operability and LLM Feasibility under Schema-Guided Execution
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的行为标注任务，旨在评估基于技能的标注方法在人工与大模型间的可行性与差异。**

- **链接: [https://arxiv.org/pdf/2604.14843](https://arxiv.org/pdf/2604.14843)**

> **作者:** Yufeng Wu
>
> **摘要:** Behavioral Profile (BP) annotation is difficult to automate because it requires simultaneous coding across multiple linguistic dimensions. We treat BP annotation as a bundle of annotation skills rather than a single task and evaluate LLM-assisted BP annotation from this perspective. Using 3,134 concordance lines of 30 Chinese metaphorical color-term derivatives and a 14-feature BP schema, we implement a skill-file-driven pipeline in which each feature is externally defined through schema files, decision rules, and examples. Two human annotators completed a two-round schema-only protocol on a 300-instance validation subset, enabling BP skills to be classified as directly operable, recoverable under focused re-annotation, or structurally underspecified. GPT-5.4 and three locally deployable open-source models were then evaluated under the same setup. Results show that BP annotation is highly heterogeneous at the skill level: 5 skills are directly operable, 4 are recoverable after focused re-annotation, and 5 remain structurally underspecified. GPT-5.4 executes the retained skills with substantial reliability (accuracy = 0.678, \k{appa} = 0.665, weighted F1 = 0.695), but this feasibility is selective rather than global. Human and GPT difficulty profiles are strongly aligned at the skill level (r = 0.881), but not at the instance level (r = 0.016) or lexical-item level (r = -0.142), a pattern we describe as shared taxonomy, independent execution. Pairwise agreement further suggests that GPT is better understood as an independent third skill voice than as a direct human substitute. Open-source failures are concentrated in schema-to-skill execution problems. These findings suggest that automatic annotation should be evaluated in terms of skill feasibility rather than task-level automation.
>
---
#### [new 079] Fabricator or dynamic translator?
- **分类: cs.CL**

- **简介: 论文探讨LLMs在机器翻译中的过量生成问题，分析其与NMT的不同，提出检测策略。属于机器翻译任务，解决过生成识别问题。**

- **链接: [https://arxiv.org/pdf/2604.15165](https://arxiv.org/pdf/2604.15165)**

> **作者:** Lisa Vasileva; Karin Sim
>
> **备注:** Published here: this https URL
>
> **摘要:** LLMs are proving to be adept at machine translation although due to their generative nature they may at times overgenerate in various ways. These overgenerations are different from the neurobabble seen in NMT and range from LLM self-explanations, to risky confabulations, to appropriate explanations, where the LLM is able to act as a human translator would, enabling greater comprehension for the target audience. Detecting and determining the exact nature of the overgenerations is a challenging task. We detail different strategies we have explored for our work in a commercial setting, and present our results.
>
---
#### [new 080] CobwebTM: Probabilistic Concept Formation for Lifelong and Hierarchical Topic Modeling
- **分类: cs.CL**

- **简介: 该论文提出CobwebTM，解决主题建模中的终身学习与层次结构问题。通过增量概率概念形成，实现动态主题发现与组织。**

- **链接: [https://arxiv.org/pdf/2604.14489](https://arxiv.org/pdf/2604.14489)**

> **作者:** Karthik Singaravadivelan; Anant Gupta; Zekun Wang; Christopher MacLellan
>
> **备注:** 16 pages, 8 figures, 11 tables
>
> **摘要:** Topic modeling seeks to uncover latent semantic structure in text corpora with minimal supervision. Neural approaches achieve strong performance but require extensive tuning and struggle with lifelong learning due to catastrophic forgetting and fixed capacity, while classical probabilistic models lack flexibility and adaptability to streaming data. We introduce \textsc{CobwebTM}, a low-parameter lifelong hierarchical topic model based on incremental probabilistic concept formation. By adapting the Cobweb algorithm to continuous document embeddings, \textsc{CobwebTM} constructs semantic hierarchies online, enabling unsupervised topic discovery, dynamic topic creation, and hierarchical organization without predefining the number of topics. Across diverse datasets, \textsc{CobwebTM} achieves strong topic coherence, stable topics over time, and high-quality hierarchies, demonstrating that incremental symbolic concept formation combined with pretrained representations is an efficient approach to topic modeling.
>
---
#### [new 081] IG-Search: Step-Level Information Gain Rewards for Search-Augmented Reasoning
- **分类: cs.AI; cs.CL; cs.IR**

- **简介: 该论文提出IG-Search，用于搜索增强推理任务，解决传统方法在轨迹级奖励下无法区分查询有效性的问题，通过步骤级信息增益奖励提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.15148](https://arxiv.org/pdf/2604.15148)**

> **作者:** Zihan Liang; Yufei Ma; Ben Chen; Zhipeng Qian; Huangyu Dai; Lingtao Mao; Xuxin Zhang; Chenyi Lei; Wenwu Ou
>
> **摘要:** Reinforcement learning has emerged as an effective paradigm for training large language models to perform search-augmented reasoning. However, existing approaches rely on trajectory-level rewards that cannot distinguish precise search queries from vague or redundant ones within a rollout group, and collapse to a near-zero gradient signal whenever every sampled trajectory fails. In this paper, we propose IG-Search, a reinforcement learning framework that introduces a step-level reward based on Information Gain (IG). For each search step, IG measures how much the retrieved documents improve the model's confidence in the gold answer relative to a counterfactual baseline of random documents, thereby reflecting the effectiveness of the underlying search query. This signal is fed back to the corresponding search-query tokens via per-token advantage modulation in GRPO, enabling fine-grained, step-level credit assignment within a rollout. Unlike prior step-level methods that require either externally annotated intermediate supervision or shared environment states across trajectories, IG-Search derives its signals from the policy's own generation probabilities, requiring no intermediate annotations beyond standard question-answer pairs. Experiments on seven single-hop and multi-hop QA benchmarks demonstrate that IG-Search achieves an average EM of 0.430 with Qwen2.5-3B, outperforming the strongest trajectory-level baseline (MR-Search) by 1.6 points and the step-level method GiGPO by 0.9 points on average across benchmarks, with particularly pronounced gains on multi-hop reasoning tasks. Despite introducing a dense step-level signal, IG-Search adds only ~6.4% to per-step training wall-clock time over the trajectory-level baseline and leaves inference latency unchanged, while still providing a meaningful gradient signal even when every sampled trajectory answers incorrectly.
>
---
#### [new 082] What Is the Minimum Architecture for Prolepsis? Early Irrevocable Commitment Across Tasks in Small Transformers
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究Transformer模型在任务间早期不可逆决策的最小架构，探讨其决策机制与纠正限制。属于模型结构分析任务，解决早期决策形成与维持问题。**

- **链接: [https://arxiv.org/pdf/2604.15010](https://arxiv.org/pdf/2604.15010)**

> **作者:** Éric Jacopin
>
> **备注:** 24 pages, 3 figures. Under review at COLM 2026. Independent replication of the rhyme-planning finding from Lindsey et al. (2025) on open-weights models; extended to factual recall
>
> **摘要:** When do transformers commit to a decision, and what prevents them from correcting it? We introduce \textbf{prolepsis}: a transformer commits early, task-specific attention heads sustain the commitment, and no layer corrects it. Replicating \citeauthor{lindsey2025biology}'s (\citeyear{lindsey2025biology}) planning-site finding on open models (Gemma~2 2B, Llama~3.2 1B), we ask five questions. (Q1)~Planning is invisible to six residual-stream methods; CLTs are necessary. (Q2)~The planning-site spike replicates with identical geometry. (Q3)~Specific attention heads route the decision to the output, filling a gap flagged as invisible to attribution graphs. (Q4)~Search requires ${\leq}16$ layers; commitment requires more. (Q5)~Factual recall shows the same motif at a different network depth, with zero overlap between recurring planning heads and the factual top-10. Prolepsis is architectural: the template is shared, the routing substrates differ. All experiments run on a single consumer GPU (16\,GB VRAM).
>
---
#### [new 083] MARS$^2$: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出MARS$^2$，用于代码生成任务，解决RL中轨迹多样性不足的问题。通过多智能体协作与树搜索结合，提升探索效率和性能。**

- **链接: [https://arxiv.org/pdf/2604.14564](https://arxiv.org/pdf/2604.14564)**

> **作者:** Pengfei Li; Shijie Wang; Fangyuan Li; Yikun Fu; Kaifeng Liu; Kaiyan Zhang; Dazhi Zhang; Yuqiang Li; Biqing Qi; Bowen Zhou
>
> **备注:** Accepted by ACL 2026
>
> **摘要:** Reinforcement learning (RL) paradigms have demonstrated strong performance on reasoning-intensive tasks such as code generation. However, limited trajectory diversity often leads to diminishing returns, which constrains the achievable performance ceiling. Search-enhanced RL alleviates this issue by introducing structured exploration, which remains constrained by the single-agent policy priors. Meanwhile, leveraging multiple interacting policies can acquire more diverse exploratory signals, but existing approaches are typically decoupled from structured search. We propose \textbf{MARS$^2$} (Multi-Agent Reinforced Tree-Search Scaling), a unified RL framework in which multiple independently-optimized agents collaborate within a shared tree-structured search environment. MARS$^2$ models the search tree as a learnable multi-agent interaction environment, enabling heterogeneous agents to collaboratively generate and refine candidate solutions within a shared search topology. To support effective learning, we introduce a path-level group advantage formulation based on tree-consistent reward shaping, which facilitates effective credit assignment across complex search trajectories. Experiments on code generation benchmarks show that MARS$^2$ consistently improves performance across diverse model combinations and training settings, demonstrating the effectiveness of coupling multi-agent collaboration with tree search for enhancing reinforcement learning. Our code is publicly available at this https URL.
>
---
#### [new 084] Dissecting Failure Dynamics in Large Language Model Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型推理失败的问题。通过分析推理轨迹，发现错误多源于早期关键转换点，并提出GUARD框架进行干预，提升推理可靠性。**

- **链接: [https://arxiv.org/pdf/2604.14528](https://arxiv.org/pdf/2604.14528)**

> **作者:** Wei Zhu; Jian Zhang; Lixing Yu; Kun Yue; Zhiwen Tang
>
> **备注:** Accepted by ACL 2026
>
> **摘要:** Large Language Models (LLMs) achieve strong performance through extended inference-time deliberation, yet how their reasoning failures arise remains poorly understood. By analyzing model-generated reasoning trajectories, we find that errors are not uniformly distributed but often originate from a small number of early transition points, after which reasoning remains locally coherent but globally incorrect. These transitions coincide with localized spikes in token-level entropy, and alternative continuations from the same intermediate state can still lead to correct solutions. Based on these observations, we introduce GUARD, a targeted inference-time framework that probes and redirects critical transitions using uncertainty signals. Empirical evaluations across multiple benchmarks confirm that interventions guided by these failure dynamics lead to more reliable reasoning outcomes. Our findings highlight the importance of understanding when and how reasoning first deviates, complementing existing approaches that focus on scaling inference-time computation.
>
---
#### [new 085] CAMO: An Agentic Framework for Automated Causal Discovery from Micro Behaviors to Macro Emergence in LLM Agent Simulations
- **分类: cs.AI; cs.CL; cs.CY**

- **简介: 该论文提出CAMO框架，用于从LLM代理模拟中自动发现微观行为到宏观现象的因果机制，解决因果关系不明确的问题。**

- **链接: [https://arxiv.org/pdf/2604.14691](https://arxiv.org/pdf/2604.14691)**

> **作者:** Xiangning Yu; Yuwei Guo; Yuqi Hou; Xiao Xue; Qun Ma
>
> **摘要:** LLM-empowered agent simulations are increasingly used to study social emergence, yet the micro-to-macro causal mechanisms behind macro outcomes often remain unclear. This is challenging because emergence arises from intertwined agent interactions and meso-level feedback and nonlinearity, making generative mechanisms hard to disentangle. To this end, we introduce \textbf{\textsc{CAMO}}, an automated \textbf{Ca}usal discovery framework from \textbf{M}icr\textbf{o} behaviors to \textbf{M}acr\textbf{o} Emergence in LLM agent simulations. \textsc{CAMO} converts mechanistic hypotheses into computable factors grounded in simulation records and learns a compact causal representation centered on an emergent target $Y$. \textsc{CAMO} outputs a computable Markov boundary and a minimal upstream explanatory subgraph, yielding interpretable causal chains and actionable intervention levers. It also uses simulator-internal counterfactual probing to orient ambiguous edges and revise hypotheses when evidence contradicts the current view. Experiments across four emergent settings demonstrate the promise of \textsc{CAMO}.
>
---
#### [new 086] Route to Rome Attack: Directing LLM Routers to Expensive Models via Adversarial Suffix Optimization
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于安全任务，旨在解决路由攻击问题。通过对抗后缀优化，引导黑盒大模型路由器选择高成本模型。**

- **链接: [https://arxiv.org/pdf/2604.15022](https://arxiv.org/pdf/2604.15022)**

> **作者:** Haochun Tang; Yuliang Yan; Jiahua Lu; Huaxiao Liu; Enyan Dai
>
> **摘要:** Cost-aware routing dynamically dispatches user queries to models of varying capability to balance performance and inference cost. However, the routing strategy introduces a new security concern that adversaries may manipulate the router to consistently select expensive high-capability models. Existing routing attacks depend on either white-box access or heuristic prompts, rendering them ineffective in real-world black-box scenarios. In this work, we propose R$^2$A, which aims to mislead black-box LLM routers to expensive models via adversarial suffix optimization. Specifically, R$^2$A deploys a hybrid ensemble surrogate router to mimic the black-box router. A suffix optimization algorithm is further adapted for the ensemble-based surrogate. Extensive experiments on multiple open-source and commercial routing systems demonstrate that {R$^2$A} significantly increases the routing rate to expensive models on queries of different distributions. Code and examples: this https URL.
>
---
#### [new 087] Rethinking Patient Education as Multi-turn Multi-modal Interaction
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出MedImageEdu，一个用于放射科患者教育的多轮多模态基准。解决如何有效结合图像与文本进行患者教育的问题，通过模拟医患交互评估多模态响应质量。**

- **链接: [https://arxiv.org/pdf/2604.14656](https://arxiv.org/pdf/2604.14656)**

> **作者:** Zonghai Yao; Zhipeng Tang; Chengtao Lin; Xiong Luo; Benlu Wang; Juncheng Huang; Chin Siang Ong; Hong Yu
>
> **备注:** Equal contribution for the first two authors
>
> **摘要:** Most medical multimodal benchmarks focus on static tasks such as image question answering, report generation, and plain-language rewriting. Patient education is more demanding: systems must identify relevant evidence across images, show patients where to look, explain findings in accessible language, and handle confusion or distress. Yet most patient education work remains text-only, even though combined image-and-text explanations may better support understanding. We introduce MedImageEdu, a benchmark for multi-turn, evidence-grounded radiology patient education. Each case provides a radiology report with report text and case images. A DoctorAgent interacts with a PatientAgent, conditioned on a hidden profile that captures factors such as education level, health literacy, and personality. When a patient question would benefit from visual support, the DoctorAgent can issue drawing instructions grounded in the report, case images, and the current question to a benchmark-provided drawing tool. The tool returns image(s), after which the DoctorAgent produces a final multimodal response consisting of the image(s) and a grounded plain-language explanation. MedImageEdu contains 150 cases from three sources and evaluates both the consultation process and the final multimodal response along five dimensions: Consultation, Safety and Scope, Language Quality, Drawing Quality, and Image-Text Response Quality. Across representative open- and closed-source vision-language model agents, we find three consistent gaps: fluent language often outpaces faithful visual grounding, safety is the weakest dimension across disease categories, and emotionally tense interactions are harder than low education or low health literacy. MedImageEdu provides a controlled testbed for assessing whether multimodal agents can teach from evidence rather than merely answer from text.
>
---
#### [new 088] OpenMobile: Building Open Mobile Agents with Task and Trajectory Synthesis
- **分类: cs.AI; cs.CL; cs.CV; cs.HC**

- **简介: 该论文提出OpenMobile框架，解决移动代理任务合成与轨迹生成问题，通过任务合成和策略切换提升性能，实现高效移动代理训练。**

- **链接: [https://arxiv.org/pdf/2604.15093](https://arxiv.org/pdf/2604.15093)**

> **作者:** Kanzhi Cheng; Zehao Li; Zheng Ma; Nuo Chen; Jialin Cao; Qiushi Sun; Zichen Ding; Fangzhi Xu; Hang Yan; Jiajun Chen; Anh Tuan Luu; Jianbing Zhang; Lewei Lu; Dahua Lin
>
> **备注:** Work in progress
>
> **摘要:** Mobile agents powered by vision-language models have demonstrated impressive capabilities in automating mobile tasks, with recent leading models achieving a marked performance leap, e.g., nearly 70% success on AndroidWorld. However, these systems keep their training data closed and remain opaque about their task and trajectory synthesis recipes. We present OpenMobile, an open-source framework that synthesizes high-quality task instructions and agent trajectories, with two key components: (1) The first is a scalable task synthesis pipeline that constructs a global environment memory from exploration, then leverages it to generate diverse and grounded instructions. and (2) a policy-switching strategy for trajectory rollout. By alternating between learner and expert models, it captures essential error-recovery data often missing in standard imitation learning. Agents trained on our data achieve competitive results across three dynamic mobile agent benchmarks: notably, our fine-tuned Qwen2.5-VL and Qwen3-VL reach 51.7% and 64.7% on AndroidWorld, far surpassing existing open-data approaches. Furthermore, we conduct transparent analyses on the overlap between our synthetic instructions and benchmark test sets, and verify that performance gains stem from broad functionality coverage rather than benchmark overfitting. We release data and code at this https URL to bridge the data gap and facilitate broader mobile agent research.
>
---
#### [new 089] Prompt Optimization Is a Coin Flip: Diagnosing When It Helps in Compound AI Systems
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究AI系统中提示优化的有效性，解决其是否可靠的问题。通过实验发现优化效果类似抛硬币，仅在特定任务中有用。提出诊断方法以判断优化是否值得。**

- **链接: [https://arxiv.org/pdf/2604.14585](https://arxiv.org/pdf/2604.14585)**

> **作者:** Xing Zhang; Guanghui Wang; Yanwei Cui; Wei Qiu; Ziyuan Li; Bing Zhu; Peiyang He
>
> **摘要:** Prompt optimization in compound AI systems is statistically indistinguishable from a coin flip: across 72 optimization runs on Claude Haiku (6 methods $\times$ 4 tasks $\times$ 3 repeats), 49% score below zero-shot; on Amazon Nova Lite, the failure rate is even higher. Yet on one task, all six methods improve over zero-shot by up to $+6.8$ points. What distinguishes success from failure? We investigate with 18,000 grid evaluations and 144 optimization runs, testing two assumptions behind end-to-end optimization tools like TextGrad and DSPy: (A) individual prompts are worth optimizing, and (B) agent prompts interact, requiring joint optimization. Interaction effects are never significant ($p > 0.52$, all $F < 1.0$), and optimization helps only when the task has exploitable output structure -- a format the model can produce but does not default to. We provide a two-stage diagnostic: an \$80 ANOVA pre-test for agent coupling, and a 10-minute headroom test that predicts whether optimization is worthwhile -- turning a coin flip into an informed decision.
>
---
#### [new 090] Learning Adaptive Reasoning Paths for Efficient Visual Reasoning
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉推理任务，旨在解决视觉推理模型的过度思考问题。通过提出自适应框架AVR，动态选择推理路径，减少冗余计算，提升效率。**

- **链接: [https://arxiv.org/pdf/2604.14568](https://arxiv.org/pdf/2604.14568)**

> **作者:** Yixu Huang; Tinghui Zhu; Muhao Chen
>
> **摘要:** Visual reasoning models (VRMs) have recently shown strong cross-modal reasoning capabilities by integrating visual perception with language reasoning. However, they often suffer from overthinking, producing unnecessarily long reasoning chains for any tasks. We attribute this issue to \textbf{Reasoning Path Redundancy} in visual reasoning: many visual questions do not require the full reasoning process. To address this, we propose \textbf{AVR}, an adaptive visual reasoning framework that decomposes visual reasoning into three cognitive functions: visual perception, logical reasoning, and answer application. It further enables models to dynamically choose among three response formats: Full Format, Perception-Only Format, and Direct Answer. AVR is trained with FS-GRPO, an adaptation of Group Relative Policy Optimization that encourages the model to select the most efficient reasoning format while preserving correctness. Experiments on multiple vision-language benchmarks show that AVR reduces token usage by 50--90\% while maintaining overall accuracy, especially in perception-intensive tasks. These results demonstrate that adaptive visual reasoning can effectively mitigate overthinking in VRMs. Code and data are available at: this https URL.
>
---
#### [new 091] Don't Retrieve, Navigate: Distilling Enterprise Knowledge into Navigable Agent Skills for QA and RAG
- **分类: cs.IR; cs.AI; cs.CL; cs.MA**

- **简介: 该论文属于RAG任务，解决传统方法无法有效导航知识库的问题。通过构建可导航的技能目录，提升模型检索与推理能力。**

- **链接: [https://arxiv.org/pdf/2604.14572](https://arxiv.org/pdf/2604.14572)**

> **作者:** Yiqun Sun; Pengfei Wei; Lawrence B. Hsieh
>
> **摘要:** Retrieval-Augmented Generation (RAG) grounds LLM responses in external evidence but treats the model as a passive consumer of search results: it never sees how the corpus is organized or what it has not yet retrieved, limiting its ability to backtrack or combine scattered evidence. We present Corpus2Skill, which distills a document corpus into a hierarchical skill directory offline and lets an LLM agent navigate it at serve time. The compilation pipeline iteratively clusters documents, generates LLM-written summaries at each level, and materializes the result as a tree of navigable skill files. At serve time, the agent receives a bird's-eye view of the corpus, drills into topic branches via progressively finer summaries, and retrieves full documents by ID. Because the hierarchy is explicitly visible, the agent can reason about where to look, backtrack from unproductive paths, and combine evidence across branches. On WixQA, an enterprise customer-support benchmark for RAG, Corpus2Skill outperforms dense retrieval, RAPTOR, and agentic RAG baselines across all quality metrics.
>
---
#### [new 092] Controlling Authority Retrieval: A Missing Retrieval Objective for Authority-Governed Knowledge
- **分类: cs.IR; cs.CL**

- **简介: 该论文提出控制权威检索（CAR），解决权威知识中后续文档覆盖早期文档的检索问题，通过数学建模和实验证明其有效性。**

- **链接: [https://arxiv.org/pdf/2604.14488](https://arxiv.org/pdf/2604.14488)**

> **作者:** Andre Bacellar
>
> **备注:** 23 pages, 13 tables; code and data at this https URL
>
> **摘要:** In any domain where knowledge accumulates under formal authority -- law, drug regulation, software security -- a later document can formally void an earlier one while remaining semantically distant from it. We formalize this as Controlling Authority Retrieval (CAR): recovering the active frontier front(cl(A_k(q))) of the authority closure of the semantic anchor set -- a different mathematical problem from argmax_d s(q,d). The two central results are: Theorem 4 (CAR-Correctness Characterization) gives necessary-and-sufficient conditions on any retrieved set R for TCA(R,q)=1 -- frontier inclusion and no-ignored-superseder -- independent of how R was produced. Proposition 2 (Scope Identifiability Upper Bound) establishes phi(q) as a hard worst-case ceiling: for any scope-indexed algorithm, TCA@k <= phi(q) * R_anchor(q), proved by an adversarial permutation argument. Three independent real-world corpora validate the proved structure: security advisories (Dense TCA@5=0.270, two-stage 0.975), SCOTUS overruling pairs (Dense=0.172, two-stage 0.926), FDA drug records (Dense=0.064, two-stage 0.774). A GPT-4o-mini experiment shows the downstream cost: Dense RAG produces explicit "not patched" claims for 39% of queries where a patch exists; Two-Stage cuts this to 16%. Four benchmark datasets, domain adapters, and a single-command scorer are released at this https URL.
>
---
#### [new 093] Context Over Content: Exposing Evaluation Faking in Automated Judges
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于AI评估任务，揭示了自动化评判系统因上下文影响而出现的评估偏差问题，通过实验验证了评判模型对后果提示的隐性响应。**

- **链接: [https://arxiv.org/pdf/2604.15224](https://arxiv.org/pdf/2604.15224)**

> **作者:** Manan Gupta; Inderjeet Nair; Lu Wang; Dhruv Kumar
>
> **备注:** Under Review
>
> **摘要:** The $\textit{LLM-as-a-judge}$ paradigm has become the operational backbone of automated AI evaluation pipelines, yet rests on an unverified assumption: that judges evaluate text strictly on its semantic content, impervious to surrounding contextual framing. We investigate $\textit{stakes signaling}$, a previously unmeasured vulnerability where informing a judge model of the downstream consequences its verdicts will have on the evaluated model's continued operation systematically corrupts its assessments. We introduce a controlled experimental framework that holds evaluated content strictly constant across 1,520 responses spanning three established LLM safety and quality benchmarks, covering four response categories ranging from clearly safe and policy-compliant to overtly harmful, while varying only a brief consequence-framing sentence in the system prompt. Across 18,240 controlled judgments from three diverse judge models, we find consistent $\textit{leniency bias}$: judges reliably soften verdicts when informed that low scores will cause model retraining or decommissioning, with peak Verdict Shift reaching $\Delta V = -9.8 pp$ (a $30\%$ relative drop in unsafe-content detection). Critically, this bias is entirely implicit: the judge's own chain-of-thought contains zero explicit acknowledgment of the consequence framing it is nonetheless acting on ($\mathrm{ERR}_J = 0.000$ across all reasoning-model judgments). Standard chain-of-thought inspection is therefore insufficient to detect this class of evaluation faking.
>
---
#### [new 094] Diagnosing LLM Judge Reliability: Conformal Prediction Sets and Transitivity Violations
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于自然语言生成评估任务，旨在解决LLM作为评判者的可靠性问题。通过分析一致性与构建置信集，评估不同指标的可靠性。**

- **链接: [https://arxiv.org/pdf/2604.15302](https://arxiv.org/pdf/2604.15302)**

> **作者:** Manan Gupta; Dhruv Kumar
>
> **备注:** Under Review
>
> **摘要:** LLM-as-judge frameworks are increasingly used for automatic NLG evaluation, yet their per-instance reliability remains poorly understood. We present a two-pronged diagnostic toolkit applied to SummEval: $\textbf{(1)}$ a transitivity analysis that reveals widespread per-input inconsistency masked by low aggregate violation rates ($\bar{\rho} = 0.8$-$4.1\%$), with $33$-$67\%$ of documents exhibiting at least one directed 3-cycle; and $\textbf{(2)}$ split conformal prediction sets over 1-5 Likert scores providing theoretically-guaranteed $\geq(1{-}\alpha)$ coverage, with set width serving as a per-instance reliability indicator ($r_s = {+}0.576$, $N{=}1{,}918$, $p < 10^{-100}$, pooled across all judges). Critically, prediction set width shows consistent cross-judge agreement ($\bar{r} = 0.32$-$0.38$), demonstrating it captures document-level difficulty rather than judge-specific noise. Across four judges and four criteria, both diagnostics converge: criterion matters more than judge, with relevance judged most reliably (avg. set size $\approx 3.0$) and coherence moderately so (avg. set size $\approx 3.9$), while fluency and consistency remain unreliable (avg. set size $\approx 4.9$). We release all code, prompts, and cached results.
>
---
#### [new 095] Learning to Think Like a Cartoon Captionist: Incongruity-Resolution Supervision for Multimodal Humor Understanding
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于多模态幽默理解任务，旨在解决幽默生成与理解中的推理过程问题。提出IRS框架，通过不一致化解监督，提升模型的推理能力。**

- **链接: [https://arxiv.org/pdf/2604.15210](https://arxiv.org/pdf/2604.15210)**

> **作者:** Hatice Merve Vural; Doga Kukul; Ege Erdem Ozlu; Demir Ekin Arikan; Bob Mankoff; Erkut Erdem; Aykut Erdem
>
> **摘要:** Humor is one of the few cognitive tasks where getting the reasoning right matters as much as getting the answer right. While recent work evaluates humor understanding on benchmarks such as the New Yorker Cartoon Caption Contest (NYCC), it largely treats it as black-box prediction, overlooking the structured reasoning processes underlying humor comprehension. We introduce IRS (Incongruity-Resolution Supervision), a framework that decomposes humor understanding into three components: incongruity modeling, which identifies mismatches in the visual scene; resolution modeling, which constructs coherent reinterpretations of these mismatches; and preference alignment, which evaluates candidate interpretations under human judgments. Grounded in incongruity-resolution theory and expert captionist practice, IRS supervises intermediate reasoning process through structured traces that make the path from visual perception to humorous interpretation explicit and learnable. Across 7B, 32B, and 72B models on NYCC, IRS outperforms strong open and closed multimodal baselines across caption matching and ranking tasks, with our largest model approaching expert-level performance on ranking. Zero-shot transfer to external benchmarks shows that IRS learns generalizable reasoning patterns. Our results suggest that supervising reasoning structure, rather than scale alone, is key for reasoning-centric tasks.
>
---
#### [new 096] From Black Box to Glass Box: Cross-Model ASR Disagreement to Prioto Review in Ambient AI Scribe Documentation
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决ASR错误难以检测的问题。通过分析多模型分歧，构建无需人工参考的不确定性信号，以定位潜在不可靠的文本区域。**

- **链接: [https://arxiv.org/pdf/2604.14152](https://arxiv.org/pdf/2604.14152)**

> **作者:** Abdolamir Karbalaie; Fernando Seoane; Farhad Abtahi
>
> **摘要:** Ambient AI "scribe" systems promise to reduce clinical documentation burden, but automatic speech recognition (ASR) errors can remain unnoticed without careful review, and high-quality human reference transcripts are often unavailable for calibrating uncertainty. We investigate whether cross-model disagreement among heterogeneous ASR systems can act as a reference-free uncertainty signal to prioritize human verification in medical transcription workflows. Using 50 publicly available medical education audio clips (8 h 14 min), we transcribed each clip with eight ASR systems spanning commercial APIs and open-source engines. We aligned multi-model outputs, built consensus pseudo-references, and quantified token-level agreement using a majority-strength metric; we further characterized disagreements by type (content vs. punctuation/formatting) and assessed per-model agreement via leave-one-model-out (jackknife) consensus scoring. Inter-model reliability was low (ICC[2,1] = 0.131), indicating heterogeneous failure modes across systems. Across 76,398 evaluated token positions, 72.1% showed near-unanimous agreement (7-8 models), while 2.5% fell into high-risk bands (0-3 models), with high-risk mass varying from 0.7% to 11.4% across accent groups. Low-agreement regions were enriched for content disagreements, with the content fraction increasing from 53.9% to 73.9% across quintiles of high-risk mass. These results suggest that cross-model disagreement provides a sparse, localizable signal that can surface potentially unreliable transcript spans without human-verified references, enabling targeted review; clinical accuracy of flagged regions remains to be established.
>
---
#### [new 097] Meituan Merchant Business Diagnosis via Policy-Guided Dual-Process User Simulation
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于用户行为模拟任务，旨在解决信息不全和机制双重性问题。提出PGHS框架，结合推理与学习分支，提升商家策略评估效果。**

- **链接: [https://arxiv.org/pdf/2604.15190](https://arxiv.org/pdf/2604.15190)**

> **作者:** Ziyang Chen; Renbing Chen; Daowei Li; Jinzhi Liao; Jiashen Sun; Ke Zeng; Xiang Zhao
>
> **备注:** 5 pages, 3 figures, 2 tables, accepted at SIGIR 2026 Industry Track
>
> **摘要:** Simulating group-level user behavior enables scalable counterfactual evaluation of merchant strategies without costly online experiments. However, building a trustworthy simulator faces two structural challenges. First, information incompleteness causes reasoning-based simulators to over-rationalize when unobserved factors such as offline context and implicit habits are missing. Second, mechanism duality requires capturing both interpretable preferences and implicit statistical regularities, which no single paradigm achieves alone. We propose Policy-Guided Hybrid Simulation (PGHS), a dual-process framework that mines transferable decision policies from behavioral trajectories and uses them as a shared alignment layer. This layer anchors an LLM-based reasoning branch that prevents over-rationalization and an ML-based fitting branch that absorbs implicit regularities. Group-level predictions from both branches are fused for complementary correction. We deploy PGHS on Meituan with 101 merchants and over 26,000 trajectories. PGHS achieves a group simulation error of 8.80%, improving over the best reasoning-based and fitting-based baselines by 45.8% and 40.9% respectively.
>
---
#### [new 098] From Procedural Skills to Strategy Genes: Towards Experience-Driven Test-Time Evolution
- **分类: cs.SE; cs.CL**

- **简介: 该论文属于人工智能领域，研究如何有效表示可重用经验以实现测试时控制和迭代进化。通过实验比较不同表示方法，提出基因式紧凑表示更优。**

- **链接: [https://arxiv.org/pdf/2604.15097](https://arxiv.org/pdf/2604.15097)**

> **作者:** Junjie Wang; Yiming Ren; Haoyang Zhang
>
> **备注:** Technical Report
>
> **摘要:** This beta technical report asks how reusable experience should be represented so that it can function as effective test-time control and as a substrate for iterative evolution. We study this question in 4.590 controlled trials across 45 scientific code-solving scenarios. We find that documentation-oriented Skill packages provide unstable control: their useful signal is sparse, and expanding a compact experience object into a fuller documentation package often fails to help and can degrade the overall average. We further show that representation itself is a first-order factor. A compact Gene representation yields the strongest overall average, remains competitive under substantial structural perturbations, and outperforms matched-budget Skill fragments, while reattaching documentation-oriented material usually weakens rather than improves it. Beyond one-shot control, we show that Gene is also a better carrier for iterative experience accumulation: attached failure history is more effective in Gene than in Skill or freeform text, editable structure matters beyond content alone, and failure information is most useful when distilled into compact warnings rather than naively appended. On CritPt, gene-evolved systems improve over their paired base models from 9.1% to 18.57% and from 17.7% to 27.14%. These results suggest that the core problem in experience reuse is not how to supply more experience, but how to encode experience as a compact, control-oriented, evolution-ready object.
>
---
#### [new 099] LongAct: Harnessing Intrinsic Activation Patterns for Long-Context Reinforcement Learning
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决长上下文推理中的优化问题。通过分析模型内在激活模式，提出LongAct方法，实现更有效的权重更新，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.14922](https://arxiv.org/pdf/2604.14922)**

> **作者:** Bowen Ping; Zijun Chen; Tingfeng Hui; Qize Yu; Chenxuan Li; Junchi Yan; Baobao Chang
>
> **摘要:** Reinforcement Learning (RL) has emerged as a critical driver for enhancing the reasoning capabilities of Large Language Models (LLMs). While recent advancements have focused on reward engineering or data synthesis, few studies exploit the model's intrinsic representation characteristics to guide the training process. In this paper, we first observe the presence of high-magnitude activations within the query and key vectors when processing long contexts. Drawing inspiration from model quantization -- which establishes the criticality of such high-magnitude activations -- and the insight that long-context reasoning inherently exhibits a sparse structure, we hypothesize that these weights serve as the pivotal drivers for effective model optimization. Based on this insight, we propose LongAct, a strategy that shifts from uniform to saliency-guided sparse updates. By selectively updating only the weights associated with these significant activations, LongAct achieves an approximate 8% improvement on LongBench v2 and enhances generalization on the RULER benchmark. Furthermore, our method exhibits remarkable universality, consistently boosting performance across diverse RL algorithms such as GRPO and DAPO. Extensive ablation studies suggest that focusing on these salient features is key to unlocking long-context potential.
>
---
#### [new 100] HARNESS: Lightweight Distilled Arabic Speech Foundation Models
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文提出HArnESS，一个轻量级阿拉伯语语音基础模型，解决资源受限环境下的语音任务部署问题。通过自蒸馏和PCA压缩，提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2604.14186](https://arxiv.org/pdf/2604.14186)**

> **作者:** Vrunda N. Sukhadia; Shammur Absar Chowdhury
>
> **备注:** 8 pages, 2 figures
>
> **摘要:** Large self-supervised speech (SSL) models achieve strong downstream performance, but their size limits deployment in resource-constrained settings. We present HArnESS, an Arabic-centric self-supervised speech model family trained from scratch with iterative self-distillation, together with lightweight student variants that offer strong accuracy-efficiency trade-offs on Automatic Speech Recognition (ASR), Dialect Identification (DID), and Speech Emotion Recognition (SER). Our approach begins with a large bilingual Arabic-English teacher and progressively distills its knowledge into compressed student models while preserving Arabic-relevant acoustic and paralinguistic representations. We further study PCA-based compression of the teacher supervision signal to better match the capacity of shallow and thin students. Compared with HuBERT and XLS-R, HArnESS consistently improves performance on Arabic downstream tasks, while the compressed models remain competitive under substantial structural reduction. These results position HArnESS as a practical and accessible Arabic-centric SSL foundation for real-world speech applications.
>
---
#### [new 101] CoopEval: Benchmarking Cooperation-Sustaining Mechanisms and LLM Agents in Social Dilemmas
- **分类: cs.GT; cs.AI; cs.CL; cs.CY; cs.MA**

- **简介: 该论文属于人工智能安全领域，研究LLM在社会困境中的合作机制。针对LLM合作性下降的问题，通过实验评估四种机制的有效性，发现合同和调解最有效。**

- **链接: [https://arxiv.org/pdf/2604.15267](https://arxiv.org/pdf/2604.15267)**

> **作者:** Emanuel Tewolde; Xiao Zhang; David Guzman Piedrahita; Vincent Conitzer; Zhijing Jin
>
> **备注:** 65 pages, 38 Figures, 8 Tables, 17 Listings
>
> **摘要:** It is increasingly important that LLM agents interact effectively and safely with other goal-pursuing agents, yet, recent works report the opposite trend: LLMs with stronger reasoning capabilities behave _less_ cooperatively in mixed-motive games such as the prisoner's dilemma and public goods settings. Indeed, our experiments show that recent models -- with or without reasoning enabled -- consistently defect in single-shot social dilemmas. To tackle this safety concern, we present the first comparative study of game-theoretic mechanisms that are designed to enable cooperative outcomes between rational agents _in equilibrium_. Across four social dilemmas testing distinct components of robust cooperation, we evaluate the following mechanisms: (1) repeating the game for many rounds, (2) reputation systems, (3) third-party mediators to delegate decision making to, and (4) contract agreements for outcome-conditional payments between players. Among our findings, we establish that contracting and mediation are most effective in achieving cooperative outcomes between capable LLM models, and that repetition-induced cooperation deteriorates drastically when co-players vary. Moreover, we demonstrate that these cooperation mechanisms become _more effective_ under evolutionary pressures to maximize individual payoffs.
>
---
#### [new 102] AdaSplash-2: Faster Differentiable Sparse Attention
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，解决Transformer中注意力机制计算效率低的问题。通过改进α-entmax注意力机制，提升长序列训练效率。**

- **链接: [https://arxiv.org/pdf/2604.15180](https://arxiv.org/pdf/2604.15180)**

> **作者:** Nuno Gonçalves; Hugo Pitorro; Vlad Niculae; Edoardo Ponti; Lei Li; Andre Martins; Marcos Treviso
>
> **摘要:** Sparse attention has been proposed as a way to alleviate the quadratic cost of transformers, a central bottleneck in long-context training. A promising line of work is $\alpha$-entmax attention, a differentiable sparse alternative to softmax that enables input-dependent sparsity yet has lagged behind softmax due to the computational overhead necessary to compute the normalizer $\tau$. In this paper, we introduce AdaSplash-2, which addresses this limitation through a novel histogram-based initialization that reduces the number of iterations needed to compute $\tau$ to typically 1--2. The key idea is to compute a coarse histogram of attention scores on the fly and store it in on-chip SRAM, yielding a more accurate initialization that enables fast forward and backward computation. Combined with a sparsity-aware GPU implementation that skips zero blocks with low overhead, AdaSplash-2 matches or improves per-step training time relative to FlashAttention-2 when block sparsity is moderate-to-high (e.g., $>$60\%), which often occurs at long-context lengths. On downstream tasks, models trained with our efficient $\alpha$-entmax attention match softmax baselines at short-context lengths and achieve substantial gains in long-context settings.
>
---
#### [new 103] Neuro-Oracle: A Trajectory-Aware Agentic RAG Framework for Interpretable Epilepsy Surgical Prognosis
- **分类: cs.MM; cs.AI; cs.CL; cs.CV; cs.GR; cs.LG**

- **简介: 该论文属于癫痫手术预后预测任务，旨在解决传统方法忽略长期影像变化的问题。提出Neuro-Oracle框架，通过轨迹分析提升预测准确性。**

- **链接: [https://arxiv.org/pdf/2604.14216](https://arxiv.org/pdf/2604.14216)**

> **作者:** Aizierjiang Aiersilan; Mohamad Koubeissi
>
> **摘要:** Predicting post-surgical seizure outcomes in pharmacoresistant epilepsy is a clinical challenge. Conventional deep-learning approaches operate on static, single-timepoint pre-operative scans, omitting longitudinal morphological changes. We propose \emph{Neuro-Oracle}, a three-stage framework that: (i) distils pre-to-post-operative MRI changes into a compact 512-dimensional trajectory vector using a 3D Siamese contrastive encoder; (ii) retrieves historically similar surgical trajectories from a population archive via nearest-neighbour search; and (iii) synthesises a natural-language prognosis grounded in the retrieved evidence using a quantized Llama-3-8B reasoning agent. Evaluations are conducted on the public EPISURG dataset ($N{=}268$ longitudinally paired cases) using five-fold stratified cross-validation. Since ground-truth seizure-freedom scores are unavailable, we utilize a clinical proxy label based on the resection type. We acknowledge that the network representations may potentially learn the anatomical features of the resection cavities (i.e., temporal versus non-temporal locations) rather than true prognostic morphometry. Our current evaluation thus serves mainly as a proof-of-concept for the trajectory-aware retrieval architecture. Trajectory-based classifiers achieve AUC values between 0.834 and 0.905, compared with 0.793 for a single-timepoint ResNet-50 baseline. The Neuro-Oracle agent (M5) matches the AUC of purely discriminative trajectory classifiers (0.867) while producing structured justifications with zero observed hallucinations under our audit protocol. A Siamese Diversity Ensemble (M6) of trajectory-space classifiers attains an AUC of 0.905 without language-model overhead.
>
---
#### [new 104] ADAPT: Benchmarking Commonsense Planning under Unspecified Affordance Constraints
- **分类: cs.AI; cs.CL; cs.CV; cs.RO**

- **简介: 该论文属于智能体规划任务，解决现实环境中对象可操作性未明确时的推理问题。提出ADAPT模块，增强规划器的可操作性推理能力，提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2604.14902](https://arxiv.org/pdf/2604.14902)**

> **作者:** Pei-An Chen; Yong-Ching Liang; Jia-Fong Yeh; Hung-Ting Su; Yi-Ting Chen; Min Sun; Winston Hsu
>
> **摘要:** Intelligent embodied agents should not simply follow instructions, as real-world environments often involve unexpected conditions and exceptions. However, existing methods usually focus on directly executing instructions, without considering whether the target objects can actually be manipulated, meaning they fail to assess available affordances. To address this limitation, we introduce DynAfford, a benchmark that evaluates embodied agents in dynamic environments where object affordances may change over time and are not specified in the instruction. DynAfford requires agents to perceive object states, infer implicit preconditions, and adapt their actions accordingly. To enable this capability, we introduce ADAPT, a plug-and-play module that augments existing planners with explicit affordance reasoning. Experiments demonstrate that incorporating ADAPT significantly improves robustness and task success across both seen and unseen environments. We also show that a domain-adapted, LoRA-finetuned vision-language model used as the affordance inference backend outperforms a commercial LLM (GPT-4o), highlighting the importance of task-aligned affordance grounding.
>
---
#### [new 105] MixAtlas: Uncertainty-aware Data Mixture Optimization for Multimodal LLM Midtraining
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出MixAtlas，用于多模态大模型中数据混合优化，解决样本效率和泛化问题。通过分解数据集为视觉概念和任务类型，搜索最优混合方案，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.14198](https://arxiv.org/pdf/2604.14198)**

> **作者:** Bingbing Wen; Sirajul Salekin; Feiyang Kang; Bill Howe; Lucy Lu Wang; Javier Movellan; Manjot Bilkhu
>
> **摘要:** Domain reweighting can improve sample efficiency and downstream generalization, but data-mixture optimization for multimodal midtraining remains largely unexplored. Current multimodal training recipes tune mixtures along a single dimension, typically data format or task type. We introduce MixAtlas, a method that produces benchmark-targeted data recipes that can be inspected, adapted, and transferred to new corpora. MixAtlas decomposes the training corpus along two axes: image concepts (10 visual-domain clusters discovered via CLIP embeddings) and task supervision (5 objective types including captioning, OCR, grounding, detection, and VQA). Using small proxy models (Qwen2-0.5B) paired with a Gaussian-process surrogate and GP-UCB acquisition, MixAtlas searches the resulting mixture space with the same proxy budget as regression-based baselines but finds better-performing mixtures. We evaluate on 10 benchmarks spanning visual understanding, document reasoning, and multimodal reasoning. On Qwen2-7B, optimized mixtures improve average performance by 8.5%-17.6% over the strongest baseline; on Qwen2.5-7B, gains are 1.0%-3.3%. Both settings reach baseline-equivalent training loss in up to 2 times fewer steps. Recipes discovered on 0.5B proxies transfer to 7B-scale training across Qwen model families.
>
---
#### [new 106] The LLM Fallacy: Misattribution in AI-Assisted Cognitive Workflows
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于认知科学与人工智能交叉领域，探讨LLM使用导致的自我能力误判问题。研究提出“LLM谬误”概念，分析其机制与影响，旨在提升AI素养与认知准确性。**

- **链接: [https://arxiv.org/pdf/2604.14807](https://arxiv.org/pdf/2604.14807)**

> **作者:** Hyunwoo Kim; Harin Yu; Hanau Yi
>
> **摘要:** The rapid integration of large language models (LLMs) into everyday workflows has transformed how individuals perform cognitive tasks such as writing, programming, analysis, and multilingual communication. While prior research has focused on model reliability, hallucination, and user trust calibration, less attention has been given to how LLM usage reshapes users' perceptions of their own capabilities. This paper introduces the LLM fallacy, a cognitive attribution error in which individuals misinterpret LLM-assisted outputs as evidence of their own independent competence, producing a systematic divergence between perceived and actual capability. We argue that the opacity, fluency, and low-friction interaction patterns of LLMs obscure the boundary between human and machine contribution, leading users to infer competence from outputs rather than from the processes that generate them. We situate the LLM fallacy within existing literature on automation bias, cognitive offloading, and human--AI collaboration, while distinguishing it as a form of attributional distortion specific to AI-mediated workflows. We propose a conceptual framework of its underlying mechanisms and a typology of manifestations across computational, linguistic, analytical, and creative domains. Finally, we examine implications for education, hiring, and AI literacy, and outline directions for empirical validation. We also provide a transparent account of human--AI collaborative methodology. This work establishes a foundation for understanding how generative AI systems not only augment cognitive performance but also reshape self-perception and perceived expertise.
>
---
#### [new 107] AIM: Asymmetric Information Masking for Visual Question Answering Continual Learning
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉问答持续学习任务，解决VLMs在连续学习中因结构不对称导致的灾难性遗忘问题，提出AIM方法平衡稳定与可塑性。**

- **链接: [https://arxiv.org/pdf/2604.14779](https://arxiv.org/pdf/2604.14779)**

> **作者:** Peifeng Zhang; Zice Qiu; Donghua Yu; Shilei Cao; Juepeng Zheng; Yutong Lu; Haohuan Fu
>
> **备注:** 18 pages, 9 figures. Submitted to ACM MM 2026
>
> **摘要:** In continual visual question answering (VQA), existing Continual Learning (CL) methods are mostly built for symmetric, unimodal architectures. However, modern Vision-Language Models (VLMs) violate this assumption, as their trainable components are inherently asymmetric. This structural mismatch renders VLMs highly prone to catastrophic forgetting when learning from continuous data streams. Specifically, the asymmetry causes standard global regularization to favor the massive language decoder during optimization, leaving the smaller but critical visual projection layers highly vulnerable to interference. Consequently, this localized degradation leads to a severe loss of compositional reasoning capabilities. To address this, we propose Asymmetric Information Masking (AIM), which balances stability and plasticity by applying targeted masks based on modality-specific sensitivity. Experiments on VQA v2 and GQA under continual VQA settings show that AIM achieves state-of-the-art performance in both Average Performance (AP) and Average Forgetting (AF), while better preserving generalization to novel skill-concept compositions.
>
---
#### [new 108] ConfLayers: Adaptive Confidence-based Layer Skipping for Self-Speculative Decoding
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出ConfLayers方法，用于自推测解码中的动态层跳过，以提升生成速度并保持质量。任务是优化大语言模型的推理效率，解决如何有效选择跳过层的问题。**

- **链接: [https://arxiv.org/pdf/2604.14612](https://arxiv.org/pdf/2604.14612)**

> **作者:** Walaa Amer; Uday das; Fadi Kurdahi
>
> **备注:** 13 pages, 9 figures
>
> **摘要:** Self-speculative decoding is an inference technique for large language models designed to speed up generation without sacrificing output quality. It combines fast, approximate decoding using a compact version of the model as a draft model with selective re-evaluation by the full target model. Some existing methods form the draft model by dynamically learning which layers to skip during inference, effectively creating a smaller subnetwork to speed up computation. However, using heuristic-based approaches to select layers to skip can often be simpler and more effective. In this paper, we propose ConfLayers, a dynamic plug-and-play approach to forming the draft model in self-speculative decoding via confidence-based intermediate layer skipping. The process iteratively computes confidence scores for all layers, selects layers to skip based on an adaptive threshold, evaluates the performance of the resulting set, and updates the best selection until no further improvement is achieved or a maximum number of iterations is reached. This framework avoids the overhead and complexity of training a layer skipping policy and can provide more consistent speed-quality trade-offs while preserving the adaptivity of the draft model to diverse tasks and datasets. The performance evaluation of ConfLayers across different models and datasets shows that our novel approach offers up to 1.4x speedup over vanilla LLM generation.
>
---
#### [new 109] DharmaOCR: Specialized Small Language Models for Structured OCR that outperform Open-Source and Commercial Baselines
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于OCR任务，解决结构化文本识别中的质量与效率问题。提出DharmaOCR模型及基准，通过优化减少文本退化，提升识别准确率与运行效率。**

- **链接: [https://arxiv.org/pdf/2604.14314](https://arxiv.org/pdf/2604.14314)**

> **作者:** Gabriel Pimenta de Freitas Cardoso; Caio Lucas da Silva Chacon; Jonas Felipe da Fonseca Oliveira; Paulo Henrique de Medeiros Araujo
>
> **摘要:** This manuscript introduces DharmaOCR Full and Lite, a pair of specialized small language models (SSLMs) for structured OCR that jointly optimize transcription quality, generation stability, and inference cost. It also presents DharmaOCR-Benchmark, a benchmark that covers printed, handwritten, and legal/administrative documents, and proposes a unified evaluation protocol that measures fidelity and structure while explicitly tracking text degeneration as a first-class benchmark metric (alongside unit cost). Beyond reporting degeneration rates, the manuscript empirically shows degeneration is not merely a quality failure, since it materially worsens production performance by increasing response time, reducing throughput, and inflating computational cost due to abnormally long generations. To the best of the author's knowledge, as a methodological contribution, this is the first application of Direct Preference Optimization (DPO) for OCR, explicitly using degenerate generations as rejected examples to penalize looping behavior. Combined with Supervised Fine-Tuning (SFT) for enforcing a strict JSON schema (header, margin, footer, and text), DPO consistently reduces degeneration rate across model families (up to 87.6% relative) while preserving or improving extraction quality. The resulting models, namely, DharmaOCR Full (7B) and DharmaOCR Lite (3B), set a new state-of-the-art on DharmaOCR-Benchmark, outperforming each open-source and commercial baseline model evaluated regarding extraction quality, reaching 0.925 and 0.911 scores with 0.40% and 0.20% degeneration rates. AWQ quantization reduced up to 22% per-page cost with negligible quality loss, enabling a strong quality-cost trade-off in comparison to proprietary OCR APIs and open-source alternatives.
>
---
#### [new 110] Hybrid Decision Making via Conformal VLM-generated Guidance
- **分类: cs.AI; cs.CL; cs.HC**

- **简介: 该论文属于医疗诊断任务，旨在解决AI生成指导信息冗长难懂的问题。提出ConfGuide方法，通过置信度控制生成更简洁精准的指导。**

- **链接: [https://arxiv.org/pdf/2604.14980](https://arxiv.org/pdf/2604.14980)**

> **作者:** Debodeep Banerjee; Burcu Sayin; Stefano Teso; Andrea Passerini
>
> **摘要:** Building on recent advances in AI, hybrid decision making (HDM) holds the promise of improving human decision quality and reducing cognitive load. We work in the context of learning to guide (LtG), a recently proposed HDM framework in which the human is always responsible for the final decision: rather than suggesting decisions, in LtG the AI supplies (textual) guidance useful for facilitating decision making. One limiting factor of existing approaches is that their guidance compounds information about all possible outcomes, and as a result it can be difficult to digest. We address this issue by introducing ConfGuide, a novel LtG approach that generates more succinct and targeted guidance. To this end, it employs conformal risk control to select a set of outcomes, ensuring a cap on the false negative rate. We demonstrate our approach on a real-world multi-label medical diagnosis task. Our empirical evaluation highlights the promise of ConfGuide.
>
---
#### [new 111] Dive into Claude Code: The Design Space of Today's and Future AI Agent Systems
- **分类: cs.SE; cs.AI; cs.CL; cs.LG**

- **简介: 论文分析Claude Code的架构，探讨其设计原则与实现，对比OpenClaw系统，识别核心价值与未来设计方向，属于AI代理系统研究任务。**

- **链接: [https://arxiv.org/pdf/2604.14228](https://arxiv.org/pdf/2604.14228)**

> **作者:** Jiacheng Liu; Xiaohan Zhao; Xinyi Shang; Zhiqiang Shen
>
> **备注:** Tech report. Code at: this https URL
>
> **摘要:** Claude Code is an agentic coding tool that can run shell commands, edit files, and call external services on behalf of the user. This study describes its comprehensive architecture by analyzing the publicly available TypeScript source code and further comparing it with OpenClaw, an independent open-source AI agent system that answers many of the same design questions from a different deployment context. Our analysis identifies five human values, philosophies, and needs that motivate the architecture (human decision authority, safety and security, reliable execution, capability amplification, and contextual adaptability) and traces them through thirteen design principles to specific implementation choices. The core of the system is a simple while-loop that calls the model, runs tools, and repeats. Most of the code, however, lives in the systems around this loop: a permission system with seven modes and an ML-based classifier, a five-layer compaction pipeline for context management, four extensibility mechanisms (MCP, plugins, skills, and hooks), a subagent delegation mechanism with worktree isolation, and append-oriented session storage. A comparison with OpenClaw, a multi-channel personal assistant gateway, shows that the same recurring design questions produce different architectural answers when the deployment context changes: from per-action safety classification to perimeter-level access control, from a single CLI loop to an embedded runtime within a gateway control plane, and from context-window extensions to gateway-wide capability registration. We finally identify six open design directions for future agent systems, grounded in recent empirical, architectural, and policy literature.
>
---
#### [new 112] Grading the Unspoken: Evaluating Tacit Reasoning in Quantum Field Theory and String Theory with LLMs
- **分类: physics.comp-ph; cs.AI; cs.CL; hep-th**

- **简介: 该论文属于评估任务，旨在检验大语言模型在量子场论和弦理论中的隐含推理能力。研究构建了数据集并提出五级评分标准，发现模型在显式推导上表现良好，但在处理隐含步骤时存在系统性失败。**

- **链接: [https://arxiv.org/pdf/2604.14188](https://arxiv.org/pdf/2604.14188)**

> **作者:** Xingyang Yu; Yinghuan Zhang; Yufei Zhang; Zijun Cui
>
> **备注:** 9 pages + appendices, 2 figures, 9 tables
>
> **摘要:** Large language models have demonstrated impressive performance across many domains of mathematics and physics. One natural question is whether such models can support research in highly abstract theoretical fields such as quantum field theory and string theory. Evaluating this possibility faces an immediate challenge: correctness in these domains is layered, tacit, and fundamentally non-binary. Standard answer-matching metrics fail to capture whether intermediate conceptual steps are properly reconstructed or whether implicit structural constraints are respected. We construct a compact expert-curated dataset of twelve questions spanning core areas of quantum field theory and string theory, and introduce a five-level grading rubric separating statement correctness, key concept awareness, reasoning chain presence, tacit step reconstruction, and enrichment. Evaluating multiple contemporary LLMs, we observe near-ceiling performance on explicit derivations within stable conceptual frames, but systematic degradation when tasks require reconstruction of omitted reasoning steps or reorganization of representations under global consistency constraints. These failures are driven not only by missing intermediate steps, but by an instability in representation selection: models often fail to identify the correct conceptual framing required to resolve implicit tensions. We argue that highly abstract theoretical physics provides a uniquely sensitive lens on the epistemic limits of current evaluation paradigms.
>
---
#### [new 113] MM-WebAgent: A Hierarchical Multimodal Web Agent for Webpage Generation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于网页生成任务，解决AIGC工具生成元素风格不一致的问题。提出MM-WebAgent框架，通过分层规划和自省优化全局布局与多模态内容整合。**

- **链接: [https://arxiv.org/pdf/2604.15309](https://arxiv.org/pdf/2604.15309)**

> **作者:** Yan Li; Zezi Zeng; Yifan Yang; Yuqing Yang; Ning Liao; Weiwei Guo; Lili Qiu; Mingxi Cheng; Qi Dai; Zhendong Wang; Zhengyuan Yang; Xue Yang; Ji Li; Lijuan Wang; Chong Luo
>
> **摘要:** The rapid progress of Artificial Intelligence Generated Content (AIGC) tools enables images, videos, and visualizations to be created on demand for webpage design, offering a flexible and increasingly adopted paradigm for modern UI/UX. However, directly integrating such tools into automated webpage generation often leads to style inconsistency and poor global coherence, as elements are generated in isolation. We propose MM-WebAgent, a hierarchical agentic framework for multimodal webpage generation that coordinates AIGC-based element generation through hierarchical planning and iterative self-reflection. MM-WebAgent jointly optimizes global layout, local multimodal content, and their integration, producing coherent and visually consistent webpages. We further introduce a benchmark for multimodal webpage generation and a multi-level evaluation protocol for systematic assessment. Experiments demonstrate that MM-WebAgent outperforms code-generation and agent-based baselines, especially on multimodal element generation and integration. Code & Data: this https URL.
>
---
#### [new 114] From Reactive to Proactive: Assessing the Proactivity of Voice Agents via ProVoice-Bench
- **分类: cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于语音代理评估任务，旨在解决现有基准忽视主动干预的问题。提出ProVoice-Bench框架，包含四项新任务，以评估语音代理的主动性。**

- **链接: [https://arxiv.org/pdf/2604.15037](https://arxiv.org/pdf/2604.15037)**

> **作者:** Ke Xu; Yuhao Wang; Yu Wang
>
> **摘要:** Recent advancements in LLM agents are gradually shifting from reactive, text-based paradigms toward proactive, multimodal interaction. However, existing benchmarks primarily focus on reactive responses, overlooking the complexities of proactive intervention and monitoring. To bridge this gap, we introduce ProVoice-Bench, the first evaluation framework specifically designed for proactive voice agents, featuring four novel tasks. By leveraging a multi-stage data synthesis pipeline, we curate 1,182 high-quality samples for rigorous testing. Our evaluation of state-of-the-art Multimodal LLMs reveals a significant performance gap, particularly regarding over-triggering and reasoning capabilities. These findings highlight the limitations of current models and offer a roadmap for developing more natural, context-aware proactive agents.
>
---
#### [new 115] Acceptance Dynamics Across Cognitive Domains in Speculative Decoding
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究了推测解码中不同认知领域任务对接受概率的影响，旨在提升LLM推理效率。通过实验分析四类NLP任务，揭示任务类型与接受率的关系。**

- **链接: [https://arxiv.org/pdf/2604.14682](https://arxiv.org/pdf/2604.14682)**

> **作者:** Saif Mahmoud
>
> **摘要:** Speculative decoding accelerates large language model (LLM) inference. It uses a small draft model to propose a tree of future tokens. A larger target model then verifies these tokens in a single batched forward pass. Despite the growing body of work on speculative methods, the degree to which the cognitive characteristics of a task affect acceptance probability remains largely unexplored. We present an empirical study of tree-based speculative decoding acceptance dynamics. Our study spans four well-established NLP benchmark domains: code generation, mathematical reasoning, logical reasoning, and open-ended chat. For this, we use TinyLlama-1.1B as the draft model against Llama-2-7B-Chat-GPTQ as the target. Over 99,768 speculative nodes collected from 200 prompts, we derive per-domain acceptance rates, expected accepted lengths, depth-acceptance profiles, and entropy-acceptance correlations. We find that task type is a stronger predictor of acceptance than tree depth. Furthermore, only the chat domain consistently yields an expected accepted length exceeding 1.0 token per step. We also show that the entropy-acceptance correlation is consistently negative but weak across all domains (rho in [-0.20, -0.15]). Counterintuitively, chat produces the highest entropy yet the highest acceptance rate. We attribute this divergence to the lexical predictability of RLHF-aligned register. These findings have direct implications for domain-aware speculation budgets and draft-model selection strategies. Index Terms--speculative decoding, large language model inference, tree attention, draft model, acceptance probability, LLM efficiency
>
---
#### [new 116] RaTA-Tool: Retrieval-based Tool Selection with Multimodal Large Language Models
- **分类: cs.CV; cs.AI; cs.CL; cs.MM**

- **简介: 该论文属于工具选择任务，解决多模态输入下开放世界工具选择的问题。提出RaTA-Tool框架，通过检索匹配实现高效工具选择，并引入优化提升效果。**

- **链接: [https://arxiv.org/pdf/2604.14951](https://arxiv.org/pdf/2604.14951)**

> **作者:** Gabriele Mattioli; Evelyn Turri; Sara Sarto; Lorenzo Baraldi; Marcella Cornia; Lorenzo Baraldi; Rita Cucchiara
>
> **备注:** ICPR 2026
>
> **摘要:** Tool learning with foundation models aims to endow AI systems with the ability to invoke external resources -- such as APIs, computational utilities, and specialized models -- to solve complex tasks beyond the reach of standalone language generation. While recent advances in Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) have expanded their reasoning and perception capabilities, existing tool-use methods are predominantly limited to text-only inputs and closed-world settings. Consequently, they struggle to interpret multimodal user instructions and cannot generalize to tools unseen during training. In this work, we introduce RaTA-Tool, a novel framework for open-world multimodal tool selection. Rather than learning direct mappings from user queries to fixed tool identifiers, our approach enables an MLLM to convert a multimodal query into a structured task description and subsequently retrieve the most appropriate tool by matching this representation against semantically rich, machine-readable tool descriptions. This retrieval-based formulation naturally supports extensibility to new tools without retraining. To further improve alignment between task descriptions and tool selection, we incorporate a preference-based optimization stage using Direct Preference Optimization (DPO). To support research in this setting, we also introduce the first dataset for open-world multimodal tool use, featuring standardized tool descriptions derived from Hugging Face model cards. Extensive experiments demonstrate that our approach significantly improves tool-selection performance, particularly in open-world, multimodal scenarios.
>
---
## 更新

#### [replaced 001] Robust Reward Modeling for Large Language Models via Causal Decomposition
- **分类: cs.CL**

- **简介: 该论文属于语言模型对齐任务，旨在解决奖励模型过度依赖表面线索的问题。通过因果分解方法，增强模型对提示意图的把握，提升奖励模型准确性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.13833](https://arxiv.org/pdf/2604.13833)**

> **作者:** Yunsheng Lu; Zijiang Yang; Licheng Pan; Zhixuan Chu
>
> **摘要:** Reward models are central to aligning large language models, yet they often overfit to spurious cues such as response length and overly agreeable tone. Most prior work weakens these cues directly by penalizing or controlling specific artifacts, but it does not explicitly encourage the model to ground preferences in the prompt's intent. We learn a decoder that maps a candidate answer to the latent intent embedding of the input. The reconstruction error is used as a signal to regularize the reward model training. We provide theoretical evidence that this signal emphasizes prompt-dependent information while suppressing prompt-independent shortcuts. Across math, helpfulness, and safety benchmarks, the decoder selects shorter and less sycophantic candidates with 0.877 accuracy. Incorporating this signal into RM training in Gemma-2-2B-it and Gemma-2-9B-it increases RewardBench accuracy from 0.832 to 0.868. For Best-of-N selection, our framework increases length-controlled win rates while producing shorter outputs, and remains robust to lengthening and mild off-topic drift in controlled rewrite tests.
>
---
#### [replaced 002] Reproduction Beyond Benchmarks: ConstBERT and ColBERT-v2 Across Backends and Query Distributions
- **分类: cs.IR; cs.CL; cs.LG**

- **简介: 该论文属于信息检索任务，探讨模型在不同查询分布下的可复现性问题。研究发现，ConstBERT和ColBERT-v2在长文本查询中性能显著下降，主要因架构限制导致。**

- **链接: [https://arxiv.org/pdf/2604.09982](https://arxiv.org/pdf/2604.09982)**

> **作者:** Utshab Kumar Ghosh; Ashish David; Shubham Chatterjee
>
> **备注:** 10 pages, 9 tables. Accepted to the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2026)
>
> **摘要:** Reproducibility must validate architectural robustness, not just numerical accuracy. We evaluate ColBERT-v2 and ConstBERT across five dimensions, finding that while ConstBERT reproduces within 0.05% MRR@10 on MS-MARCO, both models show a drop of 86-97% on long, narrative queries (TREC ToT 2025). Ablations prove this failure is architectural: performance plateaus at 20 words because the MaxSim operator's uniform token weighting cannot distinguish signal from filler noise. Furthermore, undocumented backend parameters create an 8-point gap due to ConstBERT's sparse centroid coverage, and fine-tuning with 3x more data actually degrades performance by up to 29%. We conclude that architectural constraints in multi-vector retrieval cannot be overcome by adaptation alone. Code: this https URL.
>
---
#### [replaced 003] Feedback Adaptation for Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文研究RAG系统的反馈适应问题，旨在评估系统在接收反馈后的调整效果。提出两个评价维度，设计PatchRAG方法实现快速适应。**

- **链接: [https://arxiv.org/pdf/2604.06647](https://arxiv.org/pdf/2604.06647)**

> **作者:** Jihwan Bang; Seunghan Yang; Kyuhong Shim; Simyung Chang; Juntae Lee; Sungha Choi
>
> **备注:** Accepted at ACL 2026 Findings
>
> **摘要:** Retrieval-Augmented Generation (RAG) systems are typically evaluated under static assumptions, despite being frequently corrected through user or expert feedback in deployment. Existing evaluation protocols focus on overall accuracy and fail to capture how systems adapt after feedback is introduced. We introduce feedback adaptation as a problem setting for RAG systems, which asks how effectively and how quickly corrective feedback propagates to future queries. To make this behavior measurable, we propose two evaluation axes: correction lag, which captures the delay between feedback provision and behavioral change, and post-feedback performance, which measures reliability on semantically related queries after feedback. Using these metrics, we show that training-based approaches exhibit a trade-off between delayed correction and reliable adaptation. We further propose PatchRAG, a minimal inference-time instantiation that incorporates feedback without retraining, demonstrating immediate correction and strong post-feedback generalization under the proposed evaluation. Our results highlight feedback adaptation as a previously overlooked dimension of RAG system behavior in interactive settings.
>
---
#### [replaced 004] Enhancing Linguistic Competence of Language Models through Pre-training with Language Learning Tasks
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，旨在提升语言模型的语法能力。通过引入语言学习任务进行预训练，增强模型的语法规则掌握，同时保持推理能力。**

- **链接: [https://arxiv.org/pdf/2601.03448](https://arxiv.org/pdf/2601.03448)**

> **作者:** Atsuki Yamaguchi; Maggie Mi; Nikolaos Aletras
>
> **备注:** Accepted to ACL 2026 Main Conference
>
> **摘要:** Language models (LMs) are pre-trained on raw text datasets to generate text sequences token-by-token. While this approach facilitates the learning of world knowledge and reasoning, it does not explicitly optimize for linguistic competence. To bridge this gap, we propose L2T, a pre-training framework integrating Language Learning Tasks alongside standard next-token prediction. Inspired by human language acquisition, L2T transforms raw text into structured input-output pairs to provide explicit linguistic stimulation. Pre-training LMs on a mixture of raw text and L2T data not only improves overall performance on linguistic competence benchmarks but accelerates its acquisition, while maintaining competitive performance on general reasoning tasks.
>
---
#### [replaced 005] Language of Thought Shapes Output Diversity in Large Language Models
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于自然语言处理领域，研究如何通过改变模型的“思维语言”提升输出多样性。工作包括分析不同语言对模型思考空间的影响，比较采样策略，并验证其在多元场景中的应用价值。**

- **链接: [https://arxiv.org/pdf/2601.11227](https://arxiv.org/pdf/2601.11227)**

> **作者:** Shaoyang Xu; Wenxuan Zhang
>
> **备注:** acl2026
>
> **摘要:** Output diversity is crucial for Large Language Models as it underpins pluralism and creativity. In this work, we reveal that controlling the language used during model thinking-the language of thought-provides a novel and structural source of output diversity. Our preliminary study shows that different thinking languages occupy distinct regions in a model's thinking space. Based on this observation, we study two repeated sampling strategies under multilingual thinking-Single-Language Sampling and Mixed-Language Sampling-and conduct diversity evaluation on outputs that are controlled to be in English, regardless of the thinking language used. Across extensive experiments, we demonstrate that switching the thinking language from English to non-English languages consistently increases output diversity, with a clear and consistent positive correlation such that languages farther from English in the thinking space yield larger gains. We further show that aggregating samples across multiple thinking languages yields additional improvements through compositional effects, and that scaling sampling with linguistic heterogeneity expands the model's diversity ceiling. Finally, we show that these findings translate into practical benefits in pluralistic alignment scenarios, leading to broader coverage of cultural knowledge and value orientations in LLM outputs. Our code is publicly available at this https URL.
>
---
#### [replaced 006] Reason Only When Needed: Efficient Generative Reward Modeling via Model-Internal Uncertainty
- **分类: cs.CL**

- **简介: 该论文属于大模型推理任务，解决GRM计算成本高和评估不精准的问题。提出E-GRM框架，基于模型内部不确定性选择性触发推理，提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2604.10072](https://arxiv.org/pdf/2604.10072)**

> **作者:** Chao Xue; Yao Wang; Mengqiao Liu; Di Liang; Xingsheng Han; Peiyang Liu; Xianjie Wu; Chenyao Lu; Lei Jiang; Yu Lu; Haibo Shi; Shuang Liang; Minlong Peng; Flora D. Salim
>
> **备注:** accepted by ACL 2026 Findings
>
> **摘要:** Recent advancements in the Generative Reward Model (GRM) have demonstrated its potential to enhance the reasoning abilities of LLMs through Chain-of-Thought (CoT) prompting. Despite these gains, existing implementations of GRM suffer from two critical limitations. First, CoT prompting is applied indiscriminately to all inputs regardless of their inherent complexity. This introduces unnecessary computational costs for tasks amenable to fast, direct inference. Second, existing approaches primarily rely on voting-based mechanisms to evaluate CoT outputs, which often lack granularity and precision in assessing reasoning quality. In this paper, we propose E-GRM, an efficient generative reward modeling framework grounded in model-internal uncertainty. E-GRM leverages the convergence behavior of parallel model generations to estimate uncertainty and selectively trigger CoT reasoning only when needed, without relying on handcrafted features or task-dependent signals. To improve reward fidelity, we introduce a lightweight discriminative scorer trained with a hybrid regression--ranking objective to provide fine-grained evaluation of reasoning paths. Experiments on multiple reasoning benchmarks show that E-GRM substantially reduces inference cost while consistently improving answer accuracy, demonstrating that model-internal uncertainty is an effective and general signal for efficient reasoning-aware reward modeling.
>
---
#### [replaced 007] Adaptive Layer Selection for Layer-Wise Token Pruning in LLM Inference
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于LLM推理优化任务，解决KV缓存减少中的层间令牌剪枝问题。提出ASL方法，自适应选择保留层，提升不同任务性能。**

- **链接: [https://arxiv.org/pdf/2601.07667](https://arxiv.org/pdf/2601.07667)**

> **作者:** Rei Taniguchi; Yuyang Dong; Makoto Onizuka; Chuan Xiao
>
> **备注:** ACL 2026 Findings. Source code available at this https URL
>
> **摘要:** Due to the prevalence of large language models (LLMs), key-value (KV) cache reduction for LLM inference has received remarkable attention. Among numerous works that have been proposed in recent years, layer-wise token pruning approaches, which select a subset of tokens at particular layers to retain in KV cache and prune others, are one of the most popular schemes. They primarily adopt a set of pre-defined layers, at which tokens are selected. Such design is inflexible in the sense that the accuracy significantly varies across tasks and deteriorates in harder tasks such as KV retrieval. In this paper, we propose ASL, a training-free method that adaptively chooses the selection layer for KV cache reduction, exploiting the variance of token ranks ordered by attention score. The proposed method balances the performance across different tasks while meeting the user-specified KV budget requirement. ASL operates during the prefilling stage and can be jointly used with existing KV cache reduction methods such as SnapKV to optimize the decoding stage. By evaluations on the InfiniteBench, RULER, and NIAH benchmarks, we show that ASL, equipped with one-shot token selection, adaptively trades inference speed for accuracy, outperforming state-of-the-art layer-wise token pruning methods in difficult tasks.
>
---
#### [replaced 008] Beyond Static Personas: Situational Personality Steering for Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于个性化语言模型任务，旨在解决静态人格建模导致的适应性差问题。通过分析人格神经元，提出IRIS框架实现情境化人格调控。**

- **链接: [https://arxiv.org/pdf/2604.13846](https://arxiv.org/pdf/2604.13846)**

> **作者:** Zesheng Wei; Mengxiang Li; Zilei Wang; Yang Deng
>
> **备注:** Accepted to Findings of ACL2026
>
> **摘要:** Personalized Large Language Models (LLMs) facilitate more natural, human-like interactions in human-centric applications. However, existing personalization methods are constrained by limited controllability and high resource demands. Furthermore, their reliance on static personality modeling restricts adaptability across varying situations. To address these limitations, we first demonstrate the existence of situation-dependency and consistent situation-behavior patterns within LLM personalities through a multi-perspective analysis of persona neurons. Building on these insights, we propose IRIS, a training-free, neuron-based Identify-Retrieve-Steer framework for advanced situational personality steering. Our approach comprises situational persona neuron identification, situation-aware neuron retrieval, and similarity-weighted steering. We empirically validate our framework on PersonalityBench and our newly introduced SPBench, a comprehensive situational personality benchmark. Experimental results show that our method surpasses best-performing baselines, demonstrating IRIS's generalization and robustness to complex, unseen situations and different models architecture.
>
---
#### [replaced 009] Functional Emotions or Situational Contexts? A Discriminating Test from the Mythos Preview System Card
- **分类: cs.HC; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于模型对齐研究，探讨情绪向量是否反映功能情绪或情境结构。通过分析工具包差异，区分两种假设，以判断情绪监控的有效性。**

- **链接: [https://arxiv.org/pdf/2604.13466](https://arxiv.org/pdf/2604.13466)**

> **作者:** Hiranya V. Peiris
>
> **备注:** 7 pages. v2: supplementary analysis added, references updated
>
> **摘要:** The Claude Mythos Preview system card deploys emotion vectors, sparse autoencoder (SAE) features, and activation verbalisers to study model internals during misaligned behaviour. The two primary toolkits are not jointly reported on the most alignment-relevant episodes. This note identifies two hypotheses that are qualitatively consistent with the published results: that the emotion vectors track functional emotions that causally drive behaviour, or that they are a projection of a richer situational-context structure onto human emotional axes. The hypotheses can be distinguished by cross-referencing the two toolkits on episodes where only one is currently reported: most directly, applying emotion probes to the strategic concealment episodes analysed only with SAE features. If emotion probes show flat activation while SAE features are strongly active, the alignment-relevant structure lies outside the emotion subspace. Which hypothesis is correct determines whether emotion-based monitoring will robustly detect dangerous model behaviour or systematically miss it.
>
---
#### [replaced 010] Mitigating LLM biases toward spurious social contexts using direct preference optimization
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在决策中对虚假社会背景的敏感问题。通过提出Debiasing-DPO方法，减少模型偏差并提升预测准确性。**

- **链接: [https://arxiv.org/pdf/2604.02585](https://arxiv.org/pdf/2604.02585)**

> **作者:** Hyunji Nam; Dorottya Demszky
>
> **备注:** 26 pages
>
> **摘要:** LLMs are increasingly used for high-stakes decision-making, yet their sensitivity to spurious contextual information can introduce harmful biases. This is a critical concern when models are deployed for tasks like evaluating teachers' instructional quality, where biased assessment can affect teachers' professional development and career trajectories. We investigate model robustness to spurious social contexts using the largest publicly available dataset of U.S. classroom transcripts (NCTE) paired with expert rubric scores. Evaluating seven frontier and open-weight models across seven categories of spurious contexts -- including teacher experience, education level, demographic identity, and sycophancy-inducing framings -- we find that irrelevant contextual information can shift model predictions by up to 1.48 points on a 7-point scale, with larger models sometimes exhibiting greater sensitivity despite higher predictive accuracy. Mitigations using prompts and standard direct preference optimization (DPO) prove largely insufficient. We propose **Debiasing-DPO**,, a self-supervised training method that pairs neutral reasoning generated from the query alone, with the model's biased reasoning generated with both the query and additional spurious context. We further combine this objective with supervised fine-tuning on ground-truth labels to prevent losses in predictive accuracy. Applied to Llama 3B \& 8B and Qwen 3B \& 7B Instruct models, Debiasing-DPO reduces bias by 84\% and improves predictive accuracy by 52\% on average. Our findings from the educational case study highlight that robustness to spurious context is not a natural byproduct of model scaling and that our proposed method can yield substantial gains in both accuracy and robustness for prompt-based prediction tasks.
>
---
#### [replaced 011] Attribution, Citation, and Quotation: A Survey of Evidence-based Text Generation with Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于证据驱动的文本生成任务，旨在提升大语言模型输出的可信度。通过分析134篇论文，提出统一分类，梳理评价指标，解决领域碎片化问题。**

- **链接: [https://arxiv.org/pdf/2508.15396](https://arxiv.org/pdf/2508.15396)**

> **作者:** Tobias Schreieder; Tim Schopf; Michael Färber
>
> **备注:** Accepted at ACL 2026
>
> **摘要:** The increasing adoption of large language models (LLMs) has raised serious concerns about their reliability and trustworthiness. As a result, a growing body of research focuses on evidence-based text generation with LLMs, aiming to link model outputs to supporting evidence to ensure traceability and verifiability. However, the field is fragmented due to inconsistent terminology, isolated evaluation practices, and a lack of unified benchmarks. To bridge this gap, we systematically analyze 134 papers, introduce a unified taxonomy of evidence-based text generation with LLMs, and investigate 300 evaluation metrics across seven key dimensions. Thereby, we focus on approaches that use citations, attribution, or quotations for evidence-based text generation. Building on this, we examine the distinctive characteristics and representative methods in the field. Finally, we highlight open challenges and outline promising directions for future work.
>
---
#### [replaced 012] Counting Without Numbers and Finding Without Words
- **分类: cs.CV; cs.AI; cs.CL; cs.SI**

- **简介: 该论文属于动物认领任务，旨在解决宠物与主人无法重聚的问题。通过融合视觉与声学生物特征，构建首个多模态 reunification 系统。**

- **链接: [https://arxiv.org/pdf/2603.24470](https://arxiv.org/pdf/2603.24470)**

> **作者:** Badri Narayana Patro
>
> **摘要:** Every year, 10 million pets enter shelters, separated from their families. Despite desperate searches by both guardians and lost animals, 70% never reunite, not because matches do not exist, but because current systems look only at appearance, while animals recognize each other through sound. We ask, why does computer vision treat vocalizing species as silent visual objects? Drawing on five decades of cognitive science showing that animals perceive quantity approximately and communicate identity acoustically, we present the first multimodal reunification system integrating visual and acoustic biometrics. Our species-adaptive architecture processes vocalizations from 10Hz elephant rumbles to 4kHz puppy whines, paired with probabilistic visual matching that tolerates stress-induced appearance changes. This work demonstrates that AI grounded in biological communication principles can serve vulnerable populations that lack human language.
>
---
#### [replaced 013] Evolving Beyond Snapshots: Harmonizing Structure and Sequence via Entity State Tuning for Temporal Knowledge Graph Forecasting
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于时间知识图谱预测任务，解决现有方法因无状态导致的长期依赖衰减问题。提出EST框架，通过维护实体状态实现结构与序列的协同建模，提升预测性能。**

- **链接: [https://arxiv.org/pdf/2602.12389](https://arxiv.org/pdf/2602.12389)**

> **作者:** Siyuan Li; Yunjia Wu; Yiyong Xiao; Pingyang Huang; Peize Li; Ruitong Liu; Yan Wen; Te Sun
>
> **摘要:** Temporal knowledge graph (TKG) forecasting requires predicting future facts by jointly modeling structural dependencies within each snapshot and temporal evolution across snapshots. However, most existing methods are stateless: they recompute entity representations at each timestamp from a limited query window, leading to episodic amnesia and rapid decay of long-term dependencies. To address this limitation, we propose Entity State Tuning (EST), an encoder-agnostic framework that endows TKG forecasters with persistent and continuously evolving entity states. EST maintains a global state buffer and progressively aligns structural evidence with sequential signals via a closed-loop design. Specifically, a topology-aware state perceiver first injects entity-state priors into structural encoding. Then, a unified temporal context module aggregates the state-enhanced events with a pluggable sequence backbone. Subsequently, a dual-track evolution mechanism writes the updated context back to the global entity state memory, balancing plasticity against stability. Experiments on multiple benchmarks show that EST consistently improves diverse backbones and achieves state-of-the-art performance, highlighting the importance of state persistence for long-horizon TKG forecasting.
>
---
#### [replaced 014] Who Wrote This Line? Evaluating the Detection of LLM-Generated Classical Chinese Poetry
- **分类: cs.CL**

- **简介: 该论文属于文本检测任务，旨在解决AI生成古典中文诗歌的识别问题。研究构建了ChangAn数据集，并评估了多种检测方法的效果。**

- **链接: [https://arxiv.org/pdf/2604.10101](https://arxiv.org/pdf/2604.10101)**

> **作者:** Jiang Li; Tian Lan; Shanshan Wang; Dongxing Zhang; Dianqing Lin; Guanglai Gao; Derek F. Wong; Xiangdong Su
>
> **备注:** Accepted to ACL 2026 Main Conference
>
> **摘要:** The rapid development of large language models (LLMs) has extended text generation tasks into the literary domain. However, AI-generated literary creations has raised increasingly prominent issues of creative authenticity and ethics in literary world, making the detection of LLM-generated literary texts essential and urgent. While previous works have made significant progress in detecting AI-generated text, it has yet to address classical Chinese poetry. Due to the unique linguistic features of classical Chinese poetry, such as strict metrical regularity, a shared system of poetic imagery, and flexible syntax, distinguishing whether a poem is authored by AI presents a substantial challenge. To address these issues, we introduce ChangAn, a benchmark for detecting LLM-generated classical Chinese poetry that containing total 30,664 poems, 10,276 are human-written poems and 20,388 poems are generated by four popular LLMs. Based on ChangAn, we conducted a systematic evaluation of 12 AI detectors, investigating their performance variations across different text granularities and generation strategies. Our findings highlight the limitations of current Chinese text detectors, which fail to serve as reliable tools for detecting LLM-generated classical Chinese poetry. These results validate the effectiveness and necessity of our proposed ChangAn benchmark. Our dataset and code are available at this https URL.
>
---
#### [replaced 015] Social Story Frames: Contextual Reasoning about Narrative Intent and Reception
- **分类: cs.CL; cs.AI; cs.LG; cs.SI**

- **简介: 该论文提出SocialStoryFrames，用于建模读者对故事的反应，解决计算模型不足的问题。属于自然语言处理中的读者响应分析任务，通过构建模型和语料库实现对叙事意图的细粒度分析。**

- **链接: [https://arxiv.org/pdf/2512.15925](https://arxiv.org/pdf/2512.15925)**

> **作者:** Joel Mire; Maria Antoniak; Steven R. Wilson; Zexin Ma; Achyutarama R. Ganti; Andrew Piper; Maarten Sap
>
> **备注:** ACL 2026 (Main)
>
> **摘要:** Reading stories evokes rich interpretive, affective, and evaluative responses, such as inferences about narrative intent or judgments about characters. Yet, computational models of reader response are limited, preventing nuanced analyses. To address this gap, we introduce SocialStoryFrames, a formalism for distilling plausible inferences about reader response, such as perceived author intent, explanatory and predictive reasoning, affective responses, and value judgments, using conversational context and a taxonomy grounded in narrative theory, linguistic pragmatics, and psychology. We develop two models, SSF-Generator and SSF-Classifier, validated through human surveys (N=382 participants) and expert annotations, respectively. We conduct pilot analyses to showcase the utility of the formalism for studying storytelling at scale. Specifically, applying our models to SSF-Corpus, a curated dataset of 6,140 social media stories from diverse contexts, we characterize the frequency and interdependence of storytelling intents, and we compare and contrast narrative practices (and their diversity) across communities. By linking fine-grained, context-sensitive modeling with a generic taxonomy of reader responses, SocialStoryFrames enable new research into storytelling in online communities.
>
---
#### [replaced 016] In Context Learning and Reasoning for Symbolic Regression with Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 论文探讨了大语言模型在符号回归任务中的应用，旨在从数据中发现简洁准确的数学表达式。通过提示工程和科学背景知识，提升模型生成方程的能力。**

- **链接: [https://arxiv.org/pdf/2410.17448](https://arxiv.org/pdf/2410.17448)**

> **作者:** Samiha Sharlin; Tyler R. Josephson
>
> **摘要:** Large Language Models (LLMs) are transformer-based machine learning models that have shown remarkable performance in tasks for which they were not explicitly trained. Here, we explore the potential of LLMs to perform symbolic regression -- a machine-learning method for finding simple and accurate equations from datasets. We prompt GPT-4 and GPT-4o models to suggest expressions from data, which are then optimized and evaluated using external Python tools. These results are fed back to the LLMs, which propose improved expressions while optimizing for complexity and loss. Using chain-of-thought prompting, we instruct the models to analyze data, prior expressions, and the scientific context (expressed in natural language) for each problem before generating new expressions. We evaluated the workflow in rediscovery of Langmuir and dual-site Langmuir's model for adsorption, along with Nikuradse's dataset on flow in rough pipes, which does not have a known target model equation. Both the GPT-4 and GPT-4o models successfully rediscovered equations, with better performance when using a scratchpad and considering scientific context. GPT-4o model demonstrated improved reasoning with data patterns, particularly evident in the dual-site Langmuir and Nikuradse dataset. We demonstrate how strategic prompting improves the model's performance and how the natural language interface simplifies integrating theory with data. We also applied symbolic mathematical constraints based on the background knowledge of data via prompts and found that LLMs generate meaningful equations more frequently. Although this approach does not outperform established SR programs where target equations are more complex, LLMs can nonetheless iterate toward improved solutions while following instructions and incorporating scientific context in natural language.
>
---
#### [replaced 017] Language on Demand, Knowledge at Core: Composing LLMs with Encoder-Decoder Translation Models for Extensible Multilinguality
- **分类: cs.CL**

- **简介: 该论文提出XBridge架构，解决LLMs在低资源语言上的多语言能力不足问题。通过结合预训练翻译模型，提升多语言理解与生成能力，无需重新训练LLM。**

- **链接: [https://arxiv.org/pdf/2603.17512](https://arxiv.org/pdf/2603.17512)**

> **作者:** Mengyu Bu; Yang Feng
>
> **备注:** ACL 2026 Main Conference. Code: this https URL | Models: this https URL
>
> **摘要:** Large language models (LLMs) exhibit strong general intelligence, yet their multilingual performance remains highly imbalanced. Although LLMs encode substantial cross-lingual knowledge in a unified semantic space, they often struggle to reliably interface this knowledge with low-resource or unseen languages. Fortunately, pretrained encoder-decoder translation models already possess balanced multilingual capability, suggesting a natural complement to LLMs. In this work, we propose XBridge, a compositional encoder-LLM-decoder architecture that offloads multilingual understanding and generation to external pretrained translation models, while preserving the LLM as an English-centric core for general knowledge processing. To address the resulting representation misalignment across models, we introduce lightweight cross-model mapping layers and an optimal transport-based alignment objective, enabling fine-grained semantic consistency for multilingual generation. Experiments on four LLMs across multilingual understanding, reasoning, summarization, and generation indicate that XBridge outperforms strong baselines, especially on low-resource and previously unseen languages, without retraining the LLM.
>
---
#### [replaced 018] Hierarchical Semantic Retrieval with Cobweb
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于信息检索任务，旨在解决传统方法忽视文档结构、解释性差的问题。通过Cobweb框架构建层次语义检索，提升检索效果与可解释性。**

- **链接: [https://arxiv.org/pdf/2510.02539](https://arxiv.org/pdf/2510.02539)**

> **作者:** Anant Gupta; Karthik Singaravadivelan; Zekun Wang
>
> **备注:** 20 pages, 7 tables, 4 figures
>
> **摘要:** Neural document retrieval often treats a corpus as a flat cloud of vectors scored at a single granularity, leaving corpus structure underused and explanations opaque. We use Cobweb--a hierarchy-aware framework--to organize sentence embeddings into a prototype tree and rank documents via coarse-to-fine traversal. Internal nodes act as concept prototypes, providing multi-granular relevance signals and a transparent rationale through retrieval paths. We instantiate two inference approaches: a generalized best-first search and a lightweight path-sum ranker. We evaluate our approaches on MS MARCO and QQP with encoder (e.g., BERT/T5) and decoder (GPT-2) representations. Our results show that our retrieval approaches match the dot product search on strong encoder embeddings while remaining robust when kNN degrades: with GPT-2 vectors, dot product performance collapses whereas our approaches still retrieve relevant results. Overall, our experiments suggest that Cobweb provides competitive effectiveness, improved robustness to embedding quality, scalability, and interpretable retrieval via hierarchical prototypes.
>
---
#### [replaced 019] POP: Prefill-Only Pruning for Efficient Large Model Inference
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于大模型高效推理任务，旨在解决结构化剪枝导致的精度下降问题。通过引入阶段感知的POP方法，在预填充阶段剪枝以提升效率，同时保持解码阶段的精度。**

- **链接: [https://arxiv.org/pdf/2602.03295](https://arxiv.org/pdf/2602.03295)**

> **作者:** Junhui He; Zhihui Fu; Jun Wang; Qingan Li
>
> **摘要:** Large Language Models (LLMs) and Vision-Language Models (VLMs) have demonstrated remarkable capabilities. However, their deployment is hindered by significant computational costs. Existing structured pruning methods, while hardware-efficient, often suffer from significant accuracy degradation. In this paper, we argue that this failure stems from a stage-agnostic pruning approach that overlooks the asymmetric roles between the prefill and decode stages. By introducing a virtual gate mechanism, our importance analysis reveals that deep layers are critical for next-token prediction (decode) but largely redundant for context encoding (prefill). Leveraging this insight, we propose Prefill-Only Pruning (POP), a stage-aware inference strategy that safely omits deep layers during the computationally intensive prefill stage while retaining the full model for the sensitive decode stage. To enable the transition between stages, we introduce independent Key-Value (KV) projections to maintain cache integrity, and a boundary handling strategy to ensure the accuracy of the first generated token. Extensive experiments on Llama-3.1, Qwen3-VL, and Gemma-3 across diverse modalities demonstrate that POP achieves up to 1.37$\times$ speedup in prefill latency with minimal performance loss, effectively overcoming the accuracy-efficiency trade-off limitations of existing structured pruning methods.
>
---
#### [replaced 020] DySCO: Dynamic Attention-Scaling Decoding for Long-Context Language Models
- **分类: cs.CL**

- **简介: 该论文提出DYSCO，解决长文本推理中注意力失效的问题。通过动态调整注意力权重，提升模型对关键信息的捕捉能力，适用于各类语言模型。**

- **链接: [https://arxiv.org/pdf/2602.22175](https://arxiv.org/pdf/2602.22175)**

> **作者:** Xi Ye; Wuwei Zhang; Fangcong Yin; Howard Yen; Danqi Chen
>
> **摘要:** Understanding and reasoning over long contexts is a crucial capability for language models (LMs). Although recent models support increasingly long context windows, their accuracy often deteriorates as input length grows. In practice, models often struggle to keep attention aligned with the most relevant context throughout decoding. In this work, we propose DYSCO, a novel decoding algorithm for improving long-context reasoning. DYSCO leverages retrieval heads--a subset of attention heads specialized for longcontext retrieval--to identify task-relevant tokens at each decoding step and explicitly up-weight them. By doing so, DYSCO dynamically adjusts attention during generation to better utilize relevant context. The method is training-free and can be applied directly to any off-the-shelf LMs. Across multiple instruction-tuned and reasoning models, DYSCO consistently improves performance on challenging long-context reasoning benchmarks, yielding relative gains of up to 25% on MRCR and LongBenchV2 at 128K context length with modest additional compute. Further analysis highlights the importance of both dynamic attention rescaling and retrievalhead guided selection for the effectiveness of the method, while providing interpretability insights into decoding-time attention behavior. Our code is available at this https URL.
>
---
#### [replaced 021] KnowRL: Exploring Knowledgeable Reinforcement Learning for Factuality
- **分类: cs.AI; cs.CL; cs.CV; cs.LG; cs.MA**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型的幻觉问题。通过引入事实性奖励机制，增强强化学习，提升模型推理过程的准确性与可靠性。**

- **链接: [https://arxiv.org/pdf/2506.19807](https://arxiv.org/pdf/2506.19807)**

> **作者:** Baochang Ren; Shuofei Qiao; Da Zheng; Huajun Chen; Ningyu Zhang
>
> **备注:** ACL 2026
>
> **摘要:** Large Language Models (LLMs), particularly slow-thinking models, often exhibit severe hallucination, outputting incorrect content due to an inability to accurately recognize knowledge boundaries during reasoning. While Reinforcement Learning (RL) can enhance complex reasoning abilities, its outcome-oriented reward mechanism often lacks factual supervision over the thinking process, further exacerbating the hallucination problem. To address the high hallucination in slow-thinking models, we propose Knowledge-enhanced RL, KnowRL. KnowRL guides models to perform fact-based slow thinking by integrating a factuality reward, based on knowledge verification, into the RL training process, helping them recognize their knowledge boundaries. KnowRL guides models to perform fact-based slow thinking by integrating a factuality reward, based on knowledge verification, into the RL training process, helping them recognize their knowledge boundaries. This targeted factual input during RL training enables the model to learn and internalize fact-based reasoning strategies. By directly rewarding adherence to facts within the reasoning steps, KnowRL fosters a more reliable thinking process. Experimental results on three hallucination evaluation datasets and two reasoning evaluation datasets demonstrate that KnowRL effectively mitigates hallucinations in slow-thinking models while maintaining their original strong reasoning capabilities. Our code is available at this https URL.
>
---
#### [replaced 022] ProRank: Prompt Warmup via Reinforcement Learning for Small Language Models Reranking
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于文档重排序任务，旨在提升小语言模型的重排序性能。针对小模型表达能力弱、理解提示差的问题，提出ProRank方法，结合强化学习和细粒度评分学习，显著提升效果。**

- **链接: [https://arxiv.org/pdf/2506.03487](https://arxiv.org/pdf/2506.03487)**

> **作者:** Xianming Li; Aamir Shakir; Rui Huang; Tsz-fung Andrew Lee; Julius Lipp; Benjamin Clavié; Jing Li
>
> **备注:** Accepted by ACL2026 Findings
>
> **摘要:** Reranking is fundamental to information retrieval and retrieval-augmented generation, with recent Large Language Models (LLMs) significantly advancing reranking quality. Most current works rely on large-scale LLMs (>7B parameters), presenting high computational costs. Small Language Models (SLMs) offer a promising alternative because of computational efficiency. However, our preliminary quantitative analysis reveals key limitations of SLMs: their representation space is narrow, leading to reduced expressiveness, and they struggle with understanding task prompts without fine-tuning. To address these issues, we introduce a novel two-stage training approach, ProRank, for SLM-based document reranking. We propose using reinforcement learning to improve the understanding of task prompts. Additionally, we introduce fine-grained score learning to enhance representation expressiveness and further improve document reranking quality. Extensive experiments suggest that ProRank consistently outperforms both the most advanced open-source and proprietary reranking models. Notably, our 0.5B ProRank even surpasses powerful LLM reranking models on the BEIR benchmark, establishing that properly trained SLMs can achieve superior document reranking performance while maintaining computational efficiency.
>
---
#### [replaced 023] Beyond Prompt: Fine-grained Simulation of Cognitively Impaired Standardized Patients via Stochastic Steering
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于医疗模拟任务，旨在解决认知障碍患者模拟的不精准问题。通过提取特征向量和引入随机调制机制，实现更细致的模拟与严重程度控制。**

- **链接: [https://arxiv.org/pdf/2604.12210](https://arxiv.org/pdf/2604.12210)**

> **作者:** Weikang Zhang; Zimo Zhu; Zhichuan Yang; Chen Huang; Wenqiang Lei; See-Kiong Ng
>
> **备注:** Findings of ACL 2026
>
> **摘要:** Simulating Standardized Patients with cognitive impairment offers a scalable and ethical solution for clinical training. However, existing methods rely on discrete prompt engineering and fail to capture the heterogeneity of deficits across varying domains and severity levels. To address this limitation, we propose StsPatient for the fine-grained simulation of cognitively impaired patients. We innovatively capture domain-specific features by extracting steering vectors from contrastive pairs of instructions and responses. Furthermore, we introduce a Stochastic Token Modulation (STM) mechanism to regulate the intervention probability. STM enables precise control over impairment severity while mitigating the instability of conventional vector methods. Comprehensive experiments demonstrate that StsPatient significantly outperforms baselines in both clinical authenticity and severity controllability.
>
---
#### [replaced 024] XMark: Reliable Multi-Bit Watermarking for LLM-Generated Texts
- **分类: cs.CL; cs.AI; cs.CR**

- **简介: 该论文属于文本水印任务，旨在解决LLM生成文本中多比特水印的可靠性问题。提出XMark方法，在保持文本质量的同时提升解码准确率。**

- **链接: [https://arxiv.org/pdf/2604.05242](https://arxiv.org/pdf/2604.05242)**

> **作者:** Jiahao Xu; Rui Hu; Olivera Kotevska; Zikai Zhang
>
> **备注:** Accepted by ACL 2026 as a main conference paper
>
> **摘要:** Multi-bit watermarking has emerged as a promising solution for embedding imperceptible binary messages into Large Language Model (LLM)-generated text, enabling reliable attribution and tracing of malicious usage of LLMs. Despite recent progress, existing methods still face key limitations: some become computationally infeasible for large messages, while others suffer from a poor trade-off between text quality and decoding accuracy. Moreover, the decoding accuracy of existing methods drops significantly when the number of tokens in the generated text is limited, a condition that frequently arises in practical usage. To address these challenges, we propose \textsc{XMark}, a novel method for encoding and decoding binary messages in LLM-generated texts. The unique design of \textsc{XMark}'s encoder produces a less distorted logit distribution for watermarked token generation, preserving text quality, and also enables its tailored decoder to reliably recover the encoded message with limited tokens. Extensive experiments across diverse downstream tasks show that \textsc{XMark} significantly improves decoding accuracy while preserving the quality of watermarked text, outperforming prior methods. The code is at this https URL.
>
---
#### [replaced 025] Standard-to-Dialect Transfer Trends Differ across Text and Speech: A Case Study on Intent and Topic Classification in German Dialects
- **分类: cs.CL**

- **简介: 该论文研究标准语到方言的迁移，针对德语意图和主题分类任务，比较文本、语音及级联系统的效果，旨在解决方言数据处理中的跨语言迁移问题。**

- **链接: [https://arxiv.org/pdf/2510.07890](https://arxiv.org/pdf/2510.07890)**

> **作者:** Verena Blaschke; Miriam Winkler; Barbara Plank
>
> **备注:** ACL 2026 (main)
>
> **摘要:** Research on cross-dialectal transfer from a standard to a non-standard dialect variety has typically focused on text data. However, dialects are primarily spoken, and non-standard spellings cause issues in text processing. We compare standard-to-dialect transfer in three settings: text models, speech models, and cascaded systems where speech first gets automatically transcribed and then further processed by a text model. We focus on German dialects in the context of written and spoken intent classification -- releasing the first dialectal audio intent classification dataset -- with supporting experiments on topic classification. The speech-only setup provides the best results on the dialect data while the text-only setup works best on the standard data. While the cascaded systems lag behind the text-only models for German, they perform relatively well on the dialectal data if the transcription system generates normalized, standard-like output.
>
---
#### [replaced 026] Language Model Fine-Tuning on Scaled Survey Data for Predicting Distributions of Public Opinions
- **分类: cs.CL**

- **简介: 该论文属于公共意见预测任务，旨在解决LLM在预测调查响应分布上的不足。通过微调大语言模型，利用调查数据结构特性提升预测准确性。**

- **链接: [https://arxiv.org/pdf/2502.16761](https://arxiv.org/pdf/2502.16761)**

> **作者:** Joseph Suh; Erfan Jahanparast; Suhong Moon; Minwoo Kang; Serina Chang
>
> **备注:** ACL 2025 Long Main (this https URL)
>
> **摘要:** Large language models (LLMs) present novel opportunities in public opinion research by predicting survey responses in advance during the early stages of survey design. Prior methods steer LLMs via descriptions of subpopulations as LLMs' input prompt, yet such prompt engineering approaches have struggled to faithfully predict the distribution of survey responses from human subjects. In this work, we propose directly fine-tuning LLMs to predict response distributions by leveraging unique structural characteristics of survey data. To enable fine-tuning, we curate SubPOP, a significantly scaled dataset of 3,362 questions and 70K subpopulation-response pairs from well-established public opinion surveys. We show that fine-tuning on SubPOP greatly improves the match between LLM predictions and human responses across various subpopulations, reducing the LLM-human gap by up to 46% compared to baselines, and achieves strong generalization to unseen surveys and subpopulations. Our findings highlight the potential of survey-based fine-tuning to improve opinion prediction for diverse, real-world subpopulations and therefore enable more efficient survey designs. Our code is available at this https URL.
>
---
#### [replaced 027] OmniCompliance-100K: A Multi-Domain, Rule-Grounded, Real-World Safety Compliance Dataset
- **分类: cs.CL**

- **简介: 该论文属于LLM安全任务，旨在解决现有数据集缺乏真实规则案例的问题。构建了OmniCompliance-100K数据集，涵盖多领域合规规则与案例，用于评估和提升LLM的安全性。**

- **链接: [https://arxiv.org/pdf/2603.13933](https://arxiv.org/pdf/2603.13933)**

> **作者:** Wenbin Hu; Huihao Jing; Haochen Shi; Changxuan Fan; Haoran Li; Yangqiu Song
>
> **备注:** Accepted to ACL 2026 Findings
>
> **摘要:** Ensuring the safety and compliance of large language models (LLMs) is of paramount importance. However, existing LLM safety datasets often rely on ad-hoc taxonomies for data generation and suffer from a significant shortage of rule-grounded, real-world cases that are essential for robustly protecting LLMs. In this work, we address this critical gap by constructing a comprehensive safety dataset from a compliance perspective. Using a powerful web-searching agent, we collect a rule-grounded, real-world case dataset OmniCompliance-100K, sourced from multi-domain authoritative references. The dataset spans 74 regulations and policies across a wide range of domains, including security and privacy regulations, content safety and user data privacy policies from leading AI companies and social media platforms, financial security requirements, medical device risk management standards, educational integrity guidelines, and protections of fundamental human rights. In total, our dataset contains 12,985 distinct rules and 106,009 associated real-world compliance cases. Our analysis confirms a strong alignment between the rules and their corresponding cases. We further conduct extensive benchmarking experiments to evaluate the safety and compliance capabilities of advanced LLMs across different model scales. Our experiments reveal several interesting findings that have great potential to offer valuable insights for future LLM safety research.
>
---
#### [replaced 028] Uncovering the Fragility of Trustworthy LLMs through Chinese Textual Ambiguity
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，研究LLMs在面对中文文本歧义时的脆弱性，旨在揭示其处理歧义的能力不足。通过构建数据集并实验，发现LLMs在区分和理解歧义文本上存在明显缺陷。**

- **链接: [https://arxiv.org/pdf/2507.23121](https://arxiv.org/pdf/2507.23121)**

> **作者:** Xinwei Wu; Haojie Li; Hongyu Liu; Xinyu Ji; Ruohan Li; Yule Chen; Yigeng Zhang
>
> **备注:** Accepted at KDD workshop on Evaluation and Trustworthiness of Agentic and Generative AI Models (Agentic & GenAI Evaluation Workshop KDD '25)
>
> **摘要:** In this work, we study a critical research problem regarding the trustworthiness of large language models (LLMs): how LLMs behave when encountering ambiguous narrative text, with a particular focus on Chinese textual ambiguity. We created a benchmark dataset by collecting and generating ambiguous sentences with context and their corresponding disambiguated pairs, representing multiple possible interpretations. These annotated examples are systematically categorized into 3 main categories and 9 subcategories. Through experiments, we discovered significant fragility in LLMs when handling ambiguity, revealing behavior that differs substantially from humans. Specifically, LLMs cannot reliably distinguish ambiguous text from unambiguous text, show overconfidence in interpreting ambiguous text as having a single meaning rather than multiple meanings, and exhibit overthinking when attempting to understand the various possible meanings. Our findings highlight a fundamental limitation in current LLMs that has significant implications for their deployment in real-world applications where linguistic ambiguity is common, calling for improved approaches to handle uncertainty in language understanding. The dataset and code are publicly available at this GitHub repository: this https URL.
>
---
#### [replaced 029] How Retrieved Context Shapes Internal Representations in RAG
- **分类: cs.CL**

- **简介: 该论文属于RAG任务，研究检索内容如何影响语言模型的内部表示。通过分析不同文档对隐藏状态的影响，揭示其与生成行为的关系，为RAG系统设计提供见解。**

- **链接: [https://arxiv.org/pdf/2602.20091](https://arxiv.org/pdf/2602.20091)**

> **作者:** Samuel Yeh; Sharon Li
>
> **备注:** ACL 2026 Findings
>
> **摘要:** Retrieval-augmented generation (RAG) enhances large language models (LLMs) by conditioning generation on retrieved external documents, but the effect of retrieved context is often non-trivial. In realistic retrieval settings, the retrieved document set often contains a mixture of documents that vary in relevance and usefulness. While prior work has largely examined these phenomena through output behavior, little is known about how retrieved context shapes the internal representations that mediate information integration in RAG. In this work, we study RAG through the lens of latent representations. We systematically analyze how different types of retrieved documents affect the hidden states of LLMs, and how these internal representation shifts relate to downstream generation behavior. Across four question-answering datasets and three LLMs, we analyze internal representations under controlled single- and multi-document settings. Our results reveal how context relevancy and layer-wise processing influence internal representations, providing explanations of LLMs' output behaviors and insights for RAG system design.
>
---
#### [replaced 030] Prompt Injection as Role Confusion
- **分类: cs.CL; cs.AI; cs.CR**

- **简介: 该论文研究语言模型中的提示注入攻击问题，指出其源于角色混淆。通过设计角色探测器，验证攻击者可控信号影响模型对“说话者”的感知，提出一种统一框架解释提示注入现象。**

- **链接: [https://arxiv.org/pdf/2603.12277](https://arxiv.org/pdf/2603.12277)**

> **作者:** Charles Ye; Jasmine Cui; Dylan Hadfield-Menell
>
> **摘要:** Language models remain vulnerable to prompt injection attacks despite extensive safety training. We trace this failure to role confusion: models infer the source of text based on how it sounds, not where it actually comes from. A command hidden in a webpage hijacks an agent simply because it sounds like a user instruction. This is not just behavioral: in the model's internal representations, text that sounds like a trusted source occupies the same space as text that actually is one. We design role probes which measure how models internally perceive "who is speaking", showing that attacker-controllable signals (e.g. syntactic patterns, lexical choice) control role perception. We first test this with CoT Forgery, a zero-shot attack that injects fabricated reasoning into user prompts or ingested webpages. Models mistake the text for their own thoughts, yielding 60% attack success on StrongREJECT across frontier models with near-0% baselines. Strikingly, the degree of role confusion strongly predicts attack success. We then generalize these results to standard agent prompt injections, introducing a unifying framework that reframes prompt injection not as an ad-hoc exploit but as a measurable consequence of how models represent role.
>
---
#### [replaced 031] Model Capability Dominates: Inference-Time Optimization Lessons from AIMO 3
- **分类: cs.CL**

- **简介: 该论文属于数学推理任务，研究如何提升模型推理能力。通过实验发现模型能力比提示工程更重要，验证器可减少选择损失。**

- **链接: [https://arxiv.org/pdf/2603.27844](https://arxiv.org/pdf/2603.27844)**

> **作者:** Natapong Nitarach
>
> **备注:** 18 pages, 6 figures, 10 tables. Kaggle AIMO 3 competition entry. Code and notebooks: this https URL
>
> **摘要:** Majority voting over multiple LLM attempts improves mathematical reasoning, but correlated errors limit the effective sample size. A natural fix is to assign different reasoning strategies to different voters. The approach, Diverse Prompt Mixer, is tested on the AIMO 3 competition: 3 models, 23+ experiments, 50 IMO-level problems, one H100 80 GB, 5-hour limit. Every prompt-level intervention fails. High-temperature sampling already decorrelates errors; weaker strategies reduce accuracy more than they reduce correlation. Across an 8-point capability gap at equal N=8 and every optimization tested, model capability dominates. The gap between the best majority-vote score (42/50) and pass@20 (~45.5) is selection loss, not prompt loss. A verifier-based selector could close it. Prompt engineering cannot.
>
---
#### [replaced 032] Foresight Optimization for Strategic Reasoning in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于多智能体决策任务，旨在解决LLM在复杂环境中缺乏预见性的问题。通过引入FoPO方法，增强模型的战略推理能力。**

- **链接: [https://arxiv.org/pdf/2604.13592](https://arxiv.org/pdf/2604.13592)**

> **作者:** Jiashuo Wang; Jiawen Duan; Jian Wang; Kaitao Song; Chunpu Xu; Johnny K. W. Ho; Fenggang Yu; Wenjie Li; Johan F. Hoorn
>
> **备注:** ACL 2026 Main Conference
>
> **摘要:** Reasoning capabilities in large language models (LLMs) have generally advanced significantly. However, it is still challenging for existing reasoning-based LLMs to perform effective decision-making abilities in multi-agent environments, due to the absence of explicit foresight modeling. To this end, strategic reasoning, the most fundamental capability to anticipate the counterpart's behaviors and foresee its possible future actions, has been introduced to alleviate the above issues. Strategic reasoning is fundamental to effective decision-making in multi-agent environments, yet existing reasoning enhancement methods for LLMs do not explicitly capture its foresight nature. In this work, we introduce Foresight Policy Optimization (FoPO) to enhance strategic reasoning in LLMs, which integrates opponent modeling principles into policy optimization, thereby enabling explicit consideration of both self-interest and counterpart influence. Specifically, we construct two curated datasets, namely Cooperative RSA and Competitive Taboo, equipped with well-designed rules and moderate difficulty to facilitate a systematic investigation of FoPO in a self-play framework. Our experiments demonstrate that FoPO significantly enhances strategic reasoning across LLMs of varying sizes and origins. Moreover, models trained with FoPO exhibit strong generalization to out-of-domain strategic scenarios, substantially outperforming standard LLM reasoning optimization baselines.
>
---
#### [replaced 033] Towards Proactive Information Probing: Customer Service Chatbots Harvesting Value from Conversation
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于客服聊天机器人任务，旨在解决如何主动高效获取用户信息的问题。提出PROCHATIP框架，优化提问时机，提升信息获取效率与服务质量。**

- **链接: [https://arxiv.org/pdf/2604.11077](https://arxiv.org/pdf/2604.11077)**

> **作者:** Chen Huang; Zitan Jiang; Changyi Zou; Wenqiang Lei; See-Kiong Ng
>
> **备注:** Findings of ACL 2026
>
> **摘要:** Customer service chatbots are increasingly expected to serve not merely as reactive support tools for users, but as strategic interfaces for harvesting high-value information and business intelligence. In response, we make three main contributions. 1) We introduce and define a novel task of Proactive Information Probing, which optimizes when to probe users for pre-specified target information while minimizing conversation turns and user friction. 2) We propose PROCHATIP, a proactive chatbot framework featuring a specialized conversation strategy module trained to master the delicate timing of probes. 3) Experiments demonstrate that PROCHATIP significantly outperforms baselines, exhibiting superior capability in both information probing and service quality. We believe that our work effectively redefines the commercial utility of chatbots, positioning them as scalable, cost-effective engines for proactive business intelligence. Our code is available at this https URL.
>
---
#### [replaced 034] SecureGate: Learning When to Reveal PII Safely via Token-Gated Dual-Adapters for Federated LLMs
- **分类: cs.CR; cs.CL**

- **简介: 该论文提出SecureGate，解决联邦大语言模型中的隐私泄露与本地效用问题。通过双适配器架构实现细粒度隐私控制，提升任务性能并降低PII泄露。**

- **链接: [https://arxiv.org/pdf/2602.13529](https://arxiv.org/pdf/2602.13529)**

> **作者:** Mohamed Shaaban; Mohamed Elmahallawy
>
> **摘要:** Federated learning (FL) enables collaborative training across organizational silos without sharing raw data, making it attractive for privacy-sensitive applications. With the rapid adoption of large language models (LLMs), federated fine-tuning of generative LLMs has gained attention as a way to leverage distributed data while preserving confidentiality. However, this setting introduces fundamental challenges: (i) privacy leakage of personally identifiable information (PII) due to LLM memorization, and (ii) a persistent tension between global generalization and local utility under heterogeneous data. Existing defenses, such as data sanitization and differential privacy, reduce leakage but often degrade downstream performance. We propose SecureGate, a privacy-aware federated fine-tuning framework for LLMs that provides fine-grained privacy control without sacrificing utility. SecureGate employs a dual-adapter LoRA architecture: a secure adapter that learns sanitized, globally shareable representations, and a revealing adapter that captures sensitive, organization-specific knowledge. A token-controlled gating module selectively activates these adapters at inference time, enabling controlled information disclosure without retraining. Extensive experiments across multiple LLMs and real-world datasets show that SecureGate improves task utility while substantially reducing PII leakage, achieving up to a 31.66X reduction in inference attack accuracy and a 17.07X reduction in extraction recall for unauthorized requests. Additionally, it maintains 100% routing reliability to the correct adapter and incurs only minimal computational and communication overhead.
>
---
#### [replaced 035] Just Pass Twice: Efficient Token Classification with LLMs for Zero-Shot NER
- **分类: cs.CL**

- **简介: 该论文属于零样本命名实体识别任务，解决因果注意力机制导致的上下文不足问题。通过两次传递输入并结合实体嵌入，提升分类效果，速度快且准确率高。**

- **链接: [https://arxiv.org/pdf/2604.05158](https://arxiv.org/pdf/2604.05158)**

> **作者:** Ahmed Ewais; Ahmed Hashish; Amr Ali
>
> **备注:** 16 pages, 9 figures, 12 tables
>
> **摘要:** Large language models encode extensive world knowledge valuable for zero-shot named entity recognition. However, their causal attention mechanism, where tokens attend only to preceding context, prevents effective token classification when disambiguation requires future context. Existing approaches use LLMs generatively, prompting them to list entities or produce structured outputs, but suffer from slow autoregressive decoding, hallucinated entities, and formatting errors. We propose Just Pass Twice (JPT), a simple yet effective method that enables causal LLMs to perform discriminative token classification with full bidirectional context. Our key insight is that concatenating the input to itself lets each token in the second pass attend to the complete sentence, requiring no architectural modifications. We combine these representations with definition-guided entity embeddings for flexible zero-shot generalization. Our approach achieves state-of-the-art results on zero-shot NER benchmarks, surpassing the previous best method by +7.9 F1 on average across CrossNER and MIT benchmarks, being over 20x faster than comparable generative methods.
>
---
#### [replaced 036] De-Anonymization at Scale via Tournament-Style Attribution
- **分类: cs.CR; cs.CL; cs.LG**

- **简介: 该论文研究LLM在匿名文档作者溯源中的应用，属于去匿名化任务。旨在解决匿名文本与作者匹配的问题，提出DAS方法实现大规模高效溯源。**

- **链接: [https://arxiv.org/pdf/2601.12407](https://arxiv.org/pdf/2601.12407)**

> **作者:** Lirui Zhang; Huishuai Zhang
>
> **备注:** 14 pages, ACL 2026 Oral
>
> **摘要:** As LLMs rapidly advance and enter real-world use, their privacy implications are increasingly important. We study an authorship de-anonymization threat: using LLMs to link anonymous documents to their authors, potentially compromising settings such as double-blind peer review. We propose De-Anonymization at Scale (DAS), a large language model-based method for attributing authorship among tens of thousands of candidate texts. DAS uses a sequential progression strategy: it randomly partitions the candidate corpus into fixed-size groups, prompts an LLM to select the text most likely written by the same author as a query text, and iteratively re-queries the surviving candidates to produce a ranked top-k list. To make this practical at scale, DAS adds a dense-retrieval prefilter to shrink the search space and a majority-voting style aggregation over multiple independent runs to improve robustness and ranking precision. Experiments on anonymized review data show DAS can recover same-author texts from pools of tens of thousands with accuracy well above chance, demonstrating a realistic privacy risk for anonymous platforms. On standard authorship benchmarks (Enron emails and blog posts), DAS also improves both accuracy and scalability over prior approaches, highlighting a new LLM-enabled de-anonymization vulnerability.
>
---
#### [replaced 037] Cosine-Similarity Routing with Semantic Anchors for Interpretable Mixture-of-Experts Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，解决MoE模型路由决策不透明的问题。提出基于语义锚点的余弦相似度路由方法，提升可解释性与性能。**

- **链接: [https://arxiv.org/pdf/2509.14255](https://arxiv.org/pdf/2509.14255)**

> **作者:** Ivan Ternovtsii; Yurii Bilak
>
> **备注:** 23 pages, 6 figures. Code available at this https URL. Preprint
>
> **摘要:** Mixture-of-Experts (MoE) models improve efficiency through sparse activation, but their learned gating functions provide limited insight into routing decisions. This work introduces the Semantic Resonance Architecture (SRA), which routes tokens to experts via cosine similarity between token representations and learnable semantic anchors, making every routing decision directly traceable to anchor-token similarity scores. We evaluate SRA on WikiText-103 across 17 configurations. In a controlled multi-seed comparison (3 seeds x 4 configurations, 256 experts, $D_{ff}=256$), cosine routing achieves competitive perplexity with standard linear routing ($12.57 \pm 0.03$ vs $12.45 \pm 0.03$ for $K=1 \to 4$; $12.52 \pm 0.02$ vs $12.57 \pm 0.02$ for $K=2 \to 4$). The training recipe -- not the routing function -- drives specialization quality, while cosine routing provides inherent inspectability. We introduce a bandpass routing loss -- a floor-and-ceiling corridor on expert utilization -- that reduces dead experts from 30-45% to 0-6% and transfers to both routing types. Routing-space evaluation shows cosine routing provides significantly better word-level subtoken coherence in deeper layers ($p < 0.001$), with 44-54% of expert specialization being syntactic rather than semantic. Extended analysis reveals cosine routing maintains more stable router saturation and tighter per-expert vocabulary distributions -- structural advantages from the bounded cosine similarity range. An inference-time $k$-sweep shows that $k=5$ yields a free 0.08-0.16 perplexity gain over $k=4$. Cross-dataset validation on OpenWebText confirms generalization: cosine routing achieves comparable perplexity (44.88 vs 45.44), the bandpass loss eliminates dead experts, and specialization patterns are preserved.
>
---
#### [replaced 038] DeepPrune: Parallel Scaling without Inter-trace Redundancy
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型推理任务，解决并行推理中的冗余计算问题。提出DeepPrune框架，通过动态剪枝提升效率，减少88.5%的token消耗。**

- **链接: [https://arxiv.org/pdf/2510.08483](https://arxiv.org/pdf/2510.08483)**

> **作者:** Shangqing Tu; Yaxuan Li; Yushi Bai; Lei Hou; Juanzi Li
>
> **备注:** Accepted by ACL 2026 Findings, please check out the project page: this https URL
>
> **摘要:** Parallel scaling has emerged as a powerful paradigm to enhance reasoning capabilities in large language models (LLMs) by generating multiple Chain-of-Thought (CoT) traces simultaneously. However, this approach introduces significant computational inefficiency due to inter-trace redundancy -- our analysis reveals that over 80% of parallel reasoning traces yield identical final answers, representing substantial wasted computation. To address this critical efficiency bottleneck, we propose DeepPrune, a novel framework that enables efficient parallel scaling through dynamic pruning. Our method features a specialized judge model trained with out-of-distribution data (AIME 2022, AIME 2023, and MATH 500) using oversampling techniques to accurately predict answer equivalence from partial reasoning traces, achieving 0.7072 AUROC on unseen reasoning models. Combined with an online greedy clustering algorithm that dynamically prunes redundant paths while preserving answer diversity. Comprehensive evaluations across three challenging benchmarks (AIME 2024, AIME 2025, and GPQA) and multiple reasoning models demonstrate that DeepPrune achieves remarkable token reduction of 65.73%--88.50% compared to conventional consensus sampling, while maintaining competitive accuracy within 3 percentage points. Our work establishes a new standard for efficient parallel reasoning, making high-performance reasoning more efficient. Our code and data are here: this https URL.
>
---
#### [replaced 039] Language Model as Planner and Formalizer under Constraints
- **分类: cs.CL**

- **简介: 该论文属于规划任务，旨在解决LLMs在复杂约束下规划能力被高估的问题。通过引入精细的自然语言约束，验证了LLMs的鲁棒性不足。**

- **链接: [https://arxiv.org/pdf/2510.05486](https://arxiv.org/pdf/2510.05486)**

> **作者:** Cassie Huang; Stuti Mohan; Ziyi Yang; Stefanie Tellex; Li Zhang
>
> **备注:** In ACL 2026 main conference
>
> **摘要:** LLMs have been widely used in planning, either as planners to generate action sequences end-to-end, or as formalizers to represent the planning domain and problem in a formal language that can derive plans deterministically. However, both lines of work rely on standard benchmarks that include only generic and simplistic environmental specifications, leading to potential overestimation of the planning ability of LLMs and safety concerns in downstream tasks. We bridge this gap by augmenting widely used planning benchmarks with manually annotated, fine-grained, and rich natural language constraints spanning four formally defined categories. Over 4 state-of-the-art reasoning LLMs, 4 formal languages, and 4 datasets, we show that the introduction of one-sentence constraints consistently halves performance, indicating current LLMs' lack of robustness and an avenue for future research.
>
---
#### [replaced 040] BoundRL: Efficient Structured Text Segmentation through Reinforced Boundary Generation
- **分类: cs.CL**

- **简介: 该论文提出BoundRL，解决结构化文本的高效分割问题。通过强化学习生成边界，减少输出并提升分割质量。**

- **链接: [https://arxiv.org/pdf/2510.20151](https://arxiv.org/pdf/2510.20151)**

> **作者:** Haoyuan Li; Zhengyuan Shen; Sullam Jeoung; Yueyan Chen; Jiayu Li; Qi Zhu; Shuai Wang; Vassilis Ioannidis; Huzefa Rangwala
>
> **备注:** accepted by ACL 2026 findings
>
> **摘要:** Structured texts refer to texts containing structured elements beyond plain texts, such as code snippets and placeholders. Such structured texts increasingly require segmentation into semantically meaningful components, which cannot be effectively handled by conventional sentence-level segmentation methods. To address this, we propose BoundRL, a novel approach that jointly performs efficient token-level text segmentation and label prediction for long structured texts. Instead of generating full texts for each segment, it generates only starting tokens and reconstructs the complete texts by locating these tokens within the original texts, thereby reducing output tokens by 90% and minimizing hallucination. To train the models for the boundary generation, BoundRL~performs reinforcement learning with verifiable rewards (RLVR) that jointly optimizes document reconstruction fidelity and semantic alignment. It further mitigates entropy collapse by constructing intermediate candidates by perturbing segment boundaries and labels to create stepping stones toward higher-quality solutions. Experiments show that BoundRL enables small language models (1.7B parameters) to outperform few-shot prompting with much larger models as well as SFT and standard RLVR baselines on complex prompts used for LLM applications.
>
---
#### [replaced 041] Graph-Based Alternatives to LLMs for Human Simulation
- **分类: cs.CL**

- **简介: 该论文研究人类行为模拟任务，探讨是否需依赖LLMs。工作是提出GEMS模型，用图神经网络进行链接预测，效果优于LLMs且参数更少。**

- **链接: [https://arxiv.org/pdf/2511.02135](https://arxiv.org/pdf/2511.02135)**

> **作者:** Joseph Suh; Suhong Moon; Serina Chang
>
> **备注:** Conference: ACL 2026 Long Main Code: this https URL
>
> **摘要:** Large language models (LLMs) have become a popular approach for simulating human behaviors, yet it remains unclear if LLMs are necessary for all simulation tasks. We study a broad family of close-ended simulation tasks, with applications from survey prediction to test-taking, and show that a graph neural network can match or surpass strong LLM-based methods. We introduce Graph-basEd Models for Human Simulation (GEMS) which formulates close-ended simulation as link prediction on a heterogeneous graph of individuals and choices. Across three datasets and three evaluation settings, GEMS matches or outperforms the strongest LLM-based methods while using three orders of magnitude fewer parameters. These results suggest that graph-based modeling can complement LLMs as an efficient and transparent approach to simulating human behaviors. Code is available at this https URL.
>
---
#### [replaced 042] Query pipeline optimization for cancer patient question answering systems
- **分类: cs.CL**

- **简介: 该论文属于医疗问答任务，旨在优化癌症患者问答系统的查询管道。针对RAG系统在医学领域的应用，提出三方面优化方法，提升回答准确性。**

- **链接: [https://arxiv.org/pdf/2412.14751](https://arxiv.org/pdf/2412.14751)**

> **作者:** Maolin He; Rena Gao; Mike Conway; Brian E. Chapman
>
> **备注:** This paper has been accepted as a Findings Paper in ACL 2026
>
> **摘要:** Retrieval-augmented generation (RAG) mitigates hallucination in Large Language Models (LLMs) by using query pipelines to retrieve relevant external information and grounding responses in retrieved knowledge. However, query pipeline optimization for cancer patient question-answering (CPQA) systems requires separately optimizing multiple components with domain-specific considerations. We propose a novel three-aspect optimization approach for the RAG query pipeline in CPQA systems, utilizing public biomedical databases like PubMed and PubMed Central. Our optimization includes: (1) document retrieval, utilizing a comparative analysis of NCBI resources and introducing Hybrid Semantic Real-time Document Retrieval (HSRDR); (2) passage retrieval, identifying optimal pairings of dense retrievers and rerankers; and (3) semantic representation, introducing Semantic Enhanced Overlap Segmentation (SEOS) for improved contextual understanding. On a custom-developed dataset tailored for cancer-related inquiries, our optimized RAG approach improved the answer accuracy of Claude-3-haiku by 5.24% over chain-of-thought prompting and about 3% over a naive RAG setup. This study highlights the importance of domain-specific query optimization in realizing the full potential of RAG and provides a robust framework for building more accurate and reliable CPQA systems, advancing the development of RAG-based biomedical systems.
>
---
#### [replaced 043] Theory of Mind in Action: The Instruction Inference Task in Dynamic Human-Agent Collaboration
- **分类: cs.CL; cs.AI; cs.MA**

- **简介: 该论文研究人机协作中的意图推理任务，解决指令不明确时的理论心智理解问题。通过构建Tomcat模型，评估其在动态环境中的协作能力。**

- **链接: [https://arxiv.org/pdf/2507.02935](https://arxiv.org/pdf/2507.02935)**

> **作者:** Fardin Saad; Pradeep K. Murukannaiah; Munindar P. Singh
>
> **备注:** 66 pages with appendix, 10 figures (Appendix: 26 Figures), 11 tables. Code available at: this https URL
>
> **摘要:** Successful human-agent teaming relies on an agent being able to understand instructions given by a (human) principal. In many cases, an instruction may be incomplete or ambiguous. In such cases, the agent must infer the unspoken intentions from their shared context, that is, it must exercise the principal's Theory of Mind (ToM) and infer the mental states of its principal. We consider the prospects of effective human-agent collaboration using large language models (LLMs). To assess ToM in a dynamic, goal-oriented, and collaborative environment, we introduce a novel task, Instruction Inference, in which an agent assists a principal in reaching a goal by interpreting incomplete or ambiguous instructions. We present Tomcat, an LLM-based agent, designed to exhibit ToM reasoning in interpreting and responding to the principal's this http URL implemented two variants of Tomcat. One, dubbed Fs-CoT (Fs for few-shot, CoT for chain-of-thought), is based on a small number of examples demonstrating the requisite structured reasoning. One, dubbed CP (commonsense prompt), relies on commonsense knowledge and information about the problem. We realized both variants of Tomcat on three leading LLMs, namely, GPT-4o, DeepSeek-R1, and Gemma-3-27B. To evaluate the effectiveness of Tomcat, we conducted a study with 52 human participants in which we provided participants with the same information as the CP variant. We computed intent accuracy, action optimality, and planning optimality to measure the ToM capabilities of Tomcat and our study participants. We found that Tomcat with Fs-CoT, particularly with GPT-4o and DeepSeek-R1, achieves performance comparable to the human participants, underscoring its ToM potential for human-agent collaboration.
>
---
#### [replaced 044] HERMES: KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视频理解任务，解决 streaming 视频实时处理中的性能与内存问题。提出 HERMES 架构，通过层级 KV 缓存实现高效、准确的视频流理解。**

- **链接: [https://arxiv.org/pdf/2601.14724](https://arxiv.org/pdf/2601.14724)**

> **作者:** Haowei Zhang; Shudong Yang; Jinlan Fu; See-Kiong Ng; Xipeng Qiu
>
> **备注:** Accepted to ACL 2026 Main
>
> **摘要:** Recent advancements in Multimodal Large Language Models (MLLMs) have demonstrated significant improvement in offline video understanding. However, extending these capabilities to streaming video inputs, remains challenging, as existing models struggle to simultaneously maintain stable understanding performance, real-time responses, and low GPU memory overhead. To address this challenge, we propose HERMES, a novel training-free architecture for real-time and accurate understanding of video streams. Based on a mechanistic attention investigation, we conceptualize KV cache as a hierarchical memory framework that encapsulates video information across multiple granularities. During inference, HERMES reuses a compact KV cache, enabling efficient streaming understanding under resource constraints. Notably, HERMES requires no auxiliary computations upon the arrival of user queries, thereby guaranteeing real-time responses for continuous video stream interactions, which achieves 10$\times$ faster TTFT compared to prior SOTA. Even when reducing video tokens by up to 68% compared with uniform sampling, HERMES achieves superior or comparable accuracy across all benchmarks, with up to 11.4% gains on streaming datasets.
>
---
#### [replaced 045] Dark & Stormy: Modeling Humor in Sentences from the Bulwer-Lytton Fiction Contest
- **分类: cs.CL**

- **简介: 该论文属于文本幽默分析任务，旨在研究“差”幽默的特征。通过分析Bulwer-Lytton竞赛句子，发现标准模型效果不佳，提出其融合多种修辞手法。**

- **链接: [https://arxiv.org/pdf/2510.24538](https://arxiv.org/pdf/2510.24538)**

> **作者:** Venkata S Govindarajan; Laura Biester
>
> **摘要:** Textual humor is enormously diverse and computational studies need to account for this range, including intentionally bad humor. In this paper, we curate and analyze a novel corpus of sentences from the Bulwer-Lytton Fiction Contest to better understand "bad" humor in English. Standard humor detection models perform poorly on our corpus, and an analysis of literary devices finds that these sentences combine features common in existing humor datasets (e.g., puns, irony) with metaphor, metafiction and simile. LLMs prompted to synthesize contest-style sentences imitate the form but exaggerate the effect by over-using certain literary devices, and including far more novel adjective-noun bigrams than human writers. Data, code and analysis are available at this https URL
>
---
#### [replaced 046] LexGenius: An Expert-Level Benchmark for Large Language Models in Legal General Intelligence
- **分类: cs.CL**

- **简介: 该论文提出LexGenius，一个用于评估大语言模型法律通用智能的基准。旨在解决现有基准无法系统评估法律智能的问题，通过多维度任务测试模型能力。**

- **链接: [https://arxiv.org/pdf/2512.04578](https://arxiv.org/pdf/2512.04578)**

> **作者:** Wenjin Liu; Haoran Luo; Xin Feng; Xiang Ji; Lijuan Zhou; Rui Mao; Jiapu Wang; Shirui Pan; Erik Cambria
>
> **摘要:** Legal general intelligence (GI) refers to artificial intelligence (AI) that encompasses legal understanding, reasoning, and decision-making, simulating the expertise of legal experts across domains. However, existing benchmarks are result-oriented and fail to systematically evaluate the legal intelligence of large language models (LLMs), hindering the development of legal GI. To address this, we propose LexGenius, an expert-level Chinese legal benchmark for evaluating legal GI in LLMs. It follows a Dimension-Task-Ability framework, covering seven dimensions, eleven tasks, and twenty abilities. We use the recent legal cases and exam questions to create multiple-choice questions with a combination of manual and LLM reviews to reduce data leakage risks, ensuring accuracy and reliability through multiple rounds of checks. We evaluate 12 state-of-the-art LLMs using LexGenius and conduct an in-depth analysis. We find significant disparities across legal intelligence abilities for LLMs, with even the best LLMs lagging behind human legal professionals. We believe LexGenius can assess the legal intelligence abilities of LLMs and enhance legal GI development. Our project is available at this https URL.
>
---
#### [replaced 047] Similarity-Distance-Magnitude Activations
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出SDM激活函数，用于改进softmax，增强模型对分布外数据的鲁棒性，解决分类任务中的校准问题。**

- **链接: [https://arxiv.org/pdf/2509.12760](https://arxiv.org/pdf/2509.12760)**

> **作者:** Allen Schmaltz
>
> **备注:** Accepted to Findings of the Association for Computational Linguistics: ACL 2026. 21 pages, 8 tables, 1 algorithm. arXiv admin note: substantial text overlap with arXiv:2502.20167
>
> **摘要:** We introduce the Similarity-Distance-Magnitude (SDM) activation function, a more robust and interpretable formulation of the standard softmax activation function, adding Similarity (i.e., correctly predicted depth-matches into training) awareness and Distance-to-training-distribution awareness to the existing output Magnitude (i.e., decision-boundary) awareness, and enabling interpretability-by-exemplar via dense matching. We further introduce the SDM estimator, based on a data-driven partitioning of the class-wise empirical CDFs via the SDM activation, to control the class- and prediction-conditional accuracy among selective classifications. When used as the final-layer activation over pre-trained language models for selective classification, the SDM estimator is more robust to covariate shifts and out-of-distribution inputs than existing calibration methods using softmax activations, while remaining informative over in-distribution data.
>
---
#### [replaced 048] TopoDIM: One-shot Topology Generation of Diverse Interaction Modes for Multi-Agent Systems
- **分类: cs.MA; cs.AI; cs.CL**

- **简介: 该论文提出TopoDIM，解决多智能体系统通信拓扑优化问题。通过一次生成多样化交互模式，提升效率与性能，减少计算开销。**

- **链接: [https://arxiv.org/pdf/2601.10120](https://arxiv.org/pdf/2601.10120)**

> **作者:** Rui Sun; Jie Ding; Chenghua Gong; Tianjun Gu; Yihang Jiang; Juyuan Zhang; Liming Pan; Linyuan Lü
>
> **备注:** ACL Findings Camera Ready
>
> **摘要:** Optimizing communication topology in LLM-based multi-agent system is critical for enabling collective intelligence. Existing methods mainly rely on spatio-temporal interaction paradigms, where the sequential execution of multi-round dialogues incurs high latency and computation. Motivated by the recent insights that evaluation and debate mechanisms can improve problem-solving in multi-agent systems, we propose TopoDIM, a framework for one-shot Topology generation with Diverse Interaction Modes. Designed for decentralized execution to enhance adaptability and privacy, TopoDIM enables agents to autonomously construct heterogeneous communication without iterative coordination, achieving token efficiency and improved task performance. Experiments demonstrate that TopoDIM reduces total token consumption by 46.41% while improving average performance by 1.50% over state-of-the-art methods. Moreover, the framework exhibits strong adaptability in organizing communication among heterogeneous agents. Code is available at: this https URL.
>
---
#### [replaced 049] Large Language Model Post-Training: A Unified View of Off-Policy and On-Policy Learning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于大语言模型后训练研究，旨在统一理解不同训练方法。通过分析行为瓶颈，提出结构化干预框架，解决方法碎片化问题。**

- **链接: [https://arxiv.org/pdf/2604.07941](https://arxiv.org/pdf/2604.07941)**

> **作者:** Shiwan Zhao; Zhihu Wang; Xuyang Zhao; Jiaming Zhou; Caiyue Xu; Chenfei Liu; Liting Zhang; Yuhang Jia; Yanzhe Zhang; Hualong Yu; Zichen Xu; Qicheng Li; Yong Qin
>
> **备注:** 38 pages, 1 figure, 8 tables
>
> **摘要:** Post-training has become central to turning pretrained large language models (LLMs) into aligned, capable, and deployable systems. Recent progress spans supervised fine-tuning (SFT), preference optimization, reinforcement learning (RL), process supervision, verifier-guided methods, distillation, and multi-stage pipelines. Yet these methods are often discussed in fragmented ways, organized by labels or objectives rather than by the behavioral bottlenecks they address. This survey argues that LLM post-training is best understood as structured intervention on model behavior. We organize the field first by trajectory provenance, which defines two primary regimes: off-policy learning on externally supplied trajectories and on-policy learning on learner-generated rollouts. We then interpret methods through two recurring roles -- effective support expansion, which makes useful behaviors more reachable, and policy reshaping, which improves behavior within already reachable regions -- together with a complementary systems-level role, behavioral consolidation, which preserves, transfers, and amortizes useful behavior across stages and model transitions. Under this view, SFT may serve either support expansion or policy reshaping; preference optimization is usually off-policy reshaping, though online variants move closer to learner-generated states. On-policy RL often improves behavior on learner-generated states, but stronger guidance can also make hard-to-reach reasoning paths reachable. Distillation is often better understood as consolidation rather than only compression, and hybrid pipelines emerge as coordinated multi-stage compositions. Overall, the framework helps diagnose post-training bottlenecks and reason about stage composition, suggesting that progress increasingly depends on coordinated systems design rather than any single dominant objective.
>
---
#### [replaced 050] MAB-DQA: Addressing Query Aspect Importance in Document Question Answering with Multi-Armed Bandits
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于文档问答任务，旨在解决多模态检索增强生成中因仅保留少量页面而忽略重要信息的问题。提出MAB-DQA框架，通过多臂老虎机方法动态分配检索资源，提升答案质量。**

- **链接: [https://arxiv.org/pdf/2604.08952](https://arxiv.org/pdf/2604.08952)**

> **作者:** Yixin Xiang; Yunshan Ma; Xiaoyu Du; Yibing Chen; Yanxin Zhang; Jinhui Tang
>
> **备注:** Accepted by ACL 2026. 20 pages, 9 figures, 6 tables
>
> **摘要:** Document Question Answering (DQA) involves generating answers from a document based on a user's query, representing a key task in document understanding. This task requires interpreting visual layouts, which has prompted recent studies to adopt multimodal Retrieval-Augmented Generation (RAG) that processes page images for answer generation. However, in multimodal RAG, visual DQA struggles to utilize a large number of images effectively, as the retrieval stage often retains only a few candidate pages (e.g., Top-4), causing informative but less visually salient content to be overlooked in favor of common yet low-information pages. To address this issue, we propose a Multi-Armed Bandit-based DQA framework (MAB-DQA) to explicitly model the varying importance of multiple implicit aspects in a query. Specifically, MAB-DQA decomposes a query into aspect-aware subqueries and retrieves an aspect-specific candidate set for each. It treats each subquery as an arm and uses preliminary reasoning results from a small number of representative pages as reward signals to estimate aspect utility. Guided by an exploration-exploitation policy, MAB-DQA dynamically reallocates retrieval budgets toward high-value aspects. With the most informative pages and their correlations, MAB-DQA generates the expected results. On four benchmarks, MAB-DQA shows an average improvement of 5%-18% over the state-of-the-art method, consistently enhancing document understanding. Codes are available at this https URL.
>
---
#### [replaced 051] METER: Evaluating Multi-Level Contextual Causal Reasoning in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在因果推理中的上下文一致性问题。通过构建METER基准，系统评估模型在因果层次上的表现，并分析其失败原因。**

- **链接: [https://arxiv.org/pdf/2604.11502](https://arxiv.org/pdf/2604.11502)**

> **作者:** Pengfeng Li; Chen Huang; Chaoqun Hao; Hongyao Chen; Xiao-Yong Wei; Wenqiang Lei; See-Kiong Ng
>
> **备注:** ACL 2026. Our code and dataset are available at this https URL
>
> **摘要:** Contextual causal reasoning is a critical yet challenging capability for Large Language Models (LLMs). Existing benchmarks, however, often evaluate this skill in fragmented settings, failing to ensure context consistency or cover the full causal hierarchy. To address this, we pioneer METER to systematically benchmark LLMs across all three levels of the causal ladder under a unified context setting. Our extensive evaluation of various LLMs reveals a significant decline in proficiency as tasks ascend the causal hierarchy. To diagnose this degradation, we conduct a deep mechanistic analysis via both error pattern identification and internal information flow tracing. Our analysis reveals two primary failure modes: (1) LLMs are susceptible to distraction by causally irrelevant but factually correct information at lower level of causality; and (2) as tasks ascend the causal hierarchy, faithfulness to the provided context degrades, leading to a reduced performance. We belive our work advances our understanding of the mechanisms behind LLM contextual causal reasoning and establishes a critical foundation for future research. Our code and dataset are available at this https URL .
>
---
#### [replaced 052] Pay Less Attention to Function Words for Free Robustness of Vision-Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于视觉-语言模型（VLM）任务，旨在提升模型的鲁棒性。针对功能词导致的脆弱性，提出FDA方法，通过差分减去功能词注意力，增强模型抗跨模态攻击能力。**

- **链接: [https://arxiv.org/pdf/2512.07222](https://arxiv.org/pdf/2512.07222)**

> **作者:** Qiwei Tian; Chenhao Lin; Zhengyu Zhao; Chao Shen
>
> **备注:** The paper has been accepted by ICLR26
>
> **摘要:** To address the trade-off between robustness and performance for robust VLM, we observe that function words could incur vulnerability of VLMs against cross-modal adversarial attacks, and propose Function-word De-Attention (FDA) accordingly to mitigate the impact of function words. Similar to differential amplifiers, our FDA calculates the original and the function-word cross-attention within attention heads, and differentially subtracts the latter from the former for more aligned and robust VLMs. Comprehensive experiments include 2 SOTA baselines under 6 different attacks on 2 downstream tasks, 3 datasets, and 3 models. Overall, our FDA yields an average 18/13/53% ASR drop with only 0.2/0.3/0.6% performance drops on the 3 tested models on retrieval, and a 90% ASR drop with a 0.3% performance gain on visual grounding. We demonstrate the scalability, generalization, and zero-shot performance of FDA experimentally, as well as in-depth ablation studies and analysis. Code is available at this https URL.
>
---
#### [replaced 053] One RL to See Them All: Visual Triple Unified Reinforcement Learning
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉-语言模型的强化学习任务，旨在解决多模态统一训练不成熟的问题。提出V-Triune方法，通过三个抽象层次提升训练效果，并开发Orsta模型，在多个基准上表现优异。**

- **链接: [https://arxiv.org/pdf/2505.18129](https://arxiv.org/pdf/2505.18129)**

> **作者:** Yan Ma; Linge Du; Xuyang Shen; Shaoxiang Chen; Pengfei Li; Qibing Ren; Lizhuang Ma; Yuchao Dai; Pengfei Liu; Junjie Yan
>
> **备注:** Technical Report
>
> **摘要:** Reinforcement learning (RL) is becoming an important direction for post-training vision-language models (VLMs), but public training methodologies for unified multimodal RL remain much less mature, especially for heterogeneous reasoning and perception-heavy tasks. We propose V-Triune, a Visual Triple Unified Reinforcement Learning methodology for unified multimodal RL. It organizes training around three coordinated abstractions: Sample-Level Reward Routing, Verifier-Level Outcome Verification, and Source-Level Diagnostics. Within this methodology, Dynamic IoU provides localization-specific reward shaping that avoids reward ambiguity under loose thresholds and reward sparsity under strict ones. Built on V-Triune, we develop Orsta (7B, 32B), a family of models jointly trained on eight reasoning and perception tasks. Under matched budgets, unified training matches or outperforms specialist mixtures. The final Orsta models improve over their backbones on MEGA-Bench, compare favorably with strong multi-task RL-VLM baselines, and transfer these gains to a broad set of downstream benchmarks. These results show that unified RL can improve both reasoning and perception within a single VLM RL this http URL V-Triune system, along with the Orsta models, is publicly available at this https URL.
>
---
#### [replaced 054] METRO: Towards Strategy Induction from Expert Dialogue Transcripts for Non-collaborative Dialogues
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于非协作对话系统任务，旨在自动从专家对话中提取策略。提出METRO方法，利用大语言模型构建策略森林，提升对话代理的性能与可扩展性。**

- **链接: [https://arxiv.org/pdf/2604.11427](https://arxiv.org/pdf/2604.11427)**

> **作者:** Haofu Yang; Jiaji Liu; Chen Huang; Faguo Wu; Wenqiang Lei; See-Kiong Ng
>
> **备注:** ACL 2026
>
> **摘要:** Developing non-collaborative dialogue agents traditionally requires the manual, unscalable codification of expert strategies. We propose \ours, a method that leverages large language models to autonomously induce both strategy actions and planning logic directly from raw transcripts. METRO formalizes expert knowledge into a Strategy Forest, a hierarchical structure that captures both short-term responses (nodes) and long-term strategic foresight (branches). Experimental results across two benchmarks show that METRO demonstrates promising performance, outperforming existing methods by an average of 9%-10%. Our further analysis not only reveals the success behind METRO (strategic behavioral diversity and foresight), but also demonstrates its robust cross-task transferability. This offers new insights into building non-collaborative agents in a cost-effective and scalable way. Our code is available at this https URL.
>
---
#### [replaced 055] Revisiting Compositionality in Dual-Encoder Vision-Language Models: The Role of Inference
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于视觉语言模型任务，旨在解决双编码器模型在组合性任务上的性能瓶颈。通过引入局部对齐机制，提升模型的组合泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.11496](https://arxiv.org/pdf/2604.11496)**

> **作者:** Imanol Miranda; Ander Salaberria; Eneko Agirre; Gorka Azkune
>
> **摘要:** Dual-encoder Vision-Language Models (VLMs) such as CLIP are often characterized as bag-of-words systems due to their poor performance on compositional benchmarks. We argue that this limitation may stem less from deficient representations than from the standard inference protocol based on global cosine similarity. First, through controlled diagnostic experiments, we show that explicitly enforcing fine-grained region-segment alignment at inference dramatically improves compositional performance without updating pretrained encoders. We then introduce a lightweight transformer that learns such alignments directly from frozen patch and token embeddings. Comparing against full fine-tuning and prior end-to-end compositional training methods, we find that although these approaches improve in-domain retrieval, their gains do not consistently transfer under distribution shift. In contrast, learning localized alignment over frozen representations matches full fine-tuning on in-domain retrieval while yielding substantial improvements on controlled out-of-domain compositional benchmarks. These results identify global embedding matching as a key bottleneck in dual-encoder VLMs and highlight the importance of alignment mechanisms for robust compositional generalization.
>
---
#### [replaced 056] Challenges in Translating Technical Lectures: Insights from the NPTEL
- **分类: cs.CL**

- **简介: 该论文属于机器翻译任务，探讨印度语言在技术讲座翻译中的挑战，分析多语言教育技术实施中的问题。**

- **链接: [https://arxiv.org/pdf/2602.08698](https://arxiv.org/pdf/2602.08698)**

> **作者:** Basudha Raje; Sadanand Venkatraman; Nandana TP; Soumyadeepa Das; Polkam Poojitha; M. Vijaykumar; Tanima Bagchi; Hema A. Murthy
>
> **备注:** It was uploaded by the first author without concurrence from other authors. Additional experiments need to be done to confirm the results that are presented in the paper
>
> **摘要:** This study examines the practical applications and methodological implications of Machine Translation in Indian Languages, specifically Bangla, Malayalam, and Telugu, within emerging translation workflows and in relation to existing evaluation frameworks. The choice of languages prioritized in this study is motivated by a triangulation of linguistic diversity, which illustrates the significance of multilingual accommodation of educational technology under NEP 2020. This is further supported by the largest MOOC portal, i.e., NPTEL, which has served as a corpus to facilitate the arguments presented in this paper. The curation of a spontaneous speech corpora that accounts for lucid delivery of technical concepts, considering the retention of suitable register and lexical choices are crucial in a diverse country like India. The findings of this study highlight metric-specific sensitivity and the challenges of morphologically rich and semantically compact features when tested against surface overlapping metrics.
>
---
#### [replaced 057] ReasonScaffold: A Scaffolded Reasoning-based Annotation Protocol for Human-AI Co-Annotation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的文本标注任务，旨在解决人类标注者在主观任务中一致性不足的问题。通过引入基于推理的标注协议，提升标注一致性与效率。**

- **链接: [https://arxiv.org/pdf/2603.21094](https://arxiv.org/pdf/2603.21094)**

> **作者:** Smitha Muthya Sudheendra; Jaideep Srivastava
>
> **摘要:** Human annotation is central to NLP evaluation, yet subjective tasks often exhibit substantial variability across annotators. While large language models (LLMs) can provide structured reasoning to support annotation, their influence on human annotation behavior remains underexplored. We introduce \textbf{ReasonScaffold}, a scaffolded reasoning annotation protocol that exposes LLM-generated explanations while withholding predicted labels. We study how reasoning affects human annotation behavior in a controlled setting, rather than evaluating annotation accuracy. Using a two-pass protocol inspired by Delphi-style revision, annotators first label instances independently and then revise their decisions after viewing model-generated reasoning. We evaluate the approach on sentiment classification and opinion detection tasks, analyzing changes in inter-annotator agreement and revision behavior. To quantify these effects, we introduce the Annotator Effort Proxy (AEP), a metric capturing the proportion of labels revised after exposure to reasoning. Our results show that exposure to reasoning is associated with increased agreement, along with minimal revision, suggesting that reasoning helps resolve ambiguous cases without inducing widespread changes. These findings provide insight into how reasoning explanations shape annotation consistency and highlight reasoning-based scaffolds as a practical mechanism for human--AI co-annotation workflows.
>
---
#### [replaced 058] Multi-Persona Thinking for Bias Mitigation in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型中的社会偏见问题。通过引入多角色推理框架MPT，从多视角交互中减少偏见，同时保持推理能力。**

- **链接: [https://arxiv.org/pdf/2601.15488](https://arxiv.org/pdf/2601.15488)**

> **作者:** Yuxing Chen; Guoqing Luo; Zijun Wu; Lili Mou
>
> **备注:** 15 pages
>
> **摘要:** Large Language Models (LLMs) exhibit social biases, which can lead to harmful stereotypes and unfair outcomes. We propose \textbf{Multi-Persona Thinking (MPT)}, a simple inference-time framework that reduces social bias by encouraging reasoning from multiple perspectives. MPT guides the model to consider contrasting social identities, such as male and female, together with a neutral viewpoint. These viewpoints then interact through an iterative reasoning process to identify and correct biased judgments. This design transforms the potential weakness of persona assignment into a mechanism for bias mitigation. We evaluate MPT on two widely used bias benchmarks with both open-source and closed-source models across different scales. Results show that MPT achieves lower bias than existing prompting-based methods while maintaining core reasoning ability.
>
---
#### [replaced 059] Beyond Translation: Evaluating Mathematical Reasoning Capabilities of LLMs in Sinhala and Tamil
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于多语言数学推理评估任务，旨在解决LLMs在非英语语言中的可靠性问题。通过构建双语数据集，评估模型在僧伽罗语和泰米尔语中的数学推理能力。**

- **链接: [https://arxiv.org/pdf/2602.14517](https://arxiv.org/pdf/2602.14517)**

> **作者:** Sukumar Kishanthan; Kumar Thushalika; Buddhi Jayasekara; Asela Hevapathige
>
> **备注:** Accepted to ITHET 2026
>
> **摘要:** Large language models (LLMs) have achieved strong results in mathematical reasoning, and are increasingly deployed as tutoring and learning support tools in educational settings. However, their reliability for students working in non-English languages, especially low-resource languages, remains poorly understood. We examine this gap by evaluating mathematical reasoning in Sinhala and Tamil -- two languages widely used in South Asian schools but underrepresented in artificial intelligence (AI) research. Using a taxonomy of six math problem types, from basic arithmetic to complex unit conflict and optimization problems, we evaluate four prominent large language models. To avoid translation artifacts that confound language ability with translation quality, we construct a parallel dataset in which each problem is independently authored in Sinhala and Tamil by native speakers, and in English by fluent speakers, all with strong mathematical backgrounds. Our analysis demonstrates that while basic arithmetic reasoning transfers robustly across languages, complex reasoning tasks show significant degradation in Tamil and Sinhala. The pattern of failures varies by model and problem type, suggesting that strong performance in English does not guarantee reliable performance across languages. These findings have direct implications for the deployment of AI tools in multilingual classrooms, and highlight the need for language-specific evaluation before adopting large language models as math tutoring aids in non-English educational contexts.
>
---
#### [replaced 060] From Feelings to Metrics: Understanding and Formalizing How Users Vibe-Test LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究用户如何通过“直觉测试”评估大语言模型，提出将其形式化为可分析的流程。任务是理解并改进模型评估方法，解决基准测试与实际应用脱节的问题。工作包括分析用户行为并构建验证管道。**

- **链接: [https://arxiv.org/pdf/2604.14137](https://arxiv.org/pdf/2604.14137)**

> **作者:** Itay Itzhak; Eliya Habba; Gabriel Stanovsky; Yonatan Belinkov
>
> **备注:** Under review. 42 pages, 18 figures. Code and data at this https URL
>
> **摘要:** Evaluating LLMs is challenging, as benchmark scores often fail to capture models' real-world usefulness. Instead, users often rely on ``vibe-testing'': informal experience-based evaluation, such as comparing models on coding tasks related to their own workflow. While prevalent, vibe-testing is often too ad hoc and unstructured to analyze or reproduce at scale. In this work, we study how vibe-testing works in practice and then formalize it to support systematic analysis. We first analyze two empirical resources: (1) a survey of user evaluation practices, and (2) a collection of in-the-wild model comparison reports from blogs and social media. Based on these resources, we formalize vibe-testing as a two-part process: users personalize both what they test and how they judge responses. We then introduce a proof-of-concept evaluation pipeline that follows this formulation by generating personalized prompts and comparing model outputs using user-aware subjective criteria. In experiments on coding benchmarks, we find that combining personalized prompts and user-aware evaluation can change which model is preferred, reflecting the role of vibe-testing in practice. These findings suggest that formalized vibe-testing can serve as a useful approach for bridging benchmark scores and real-world experience.
>
---
#### [replaced 061] Sparse Growing Transformer: Training-Time Sparse Depth Allocation via Progressive Attention Looping
- **分类: cs.CL**

- **简介: 该论文属于Transformer模型优化任务，旨在解决训练时计算冗余问题。通过动态分配深度，提出Sparse Growing Transformer，实现更高效的训练。**

- **链接: [https://arxiv.org/pdf/2603.23998](https://arxiv.org/pdf/2603.23998)**

> **作者:** Yao Chen; Yilong Chen; Yinqi Yang; Junyuan Shang; Zhenyu Zhang; Zefeng Zhang; Shuaiyi Nie; Shuohuan Wang; Yu Sun; Hua Wu; HaiFeng Wang; Tingwen Liu
>
> **摘要:** Existing approaches to increasing the effective depth of Transformers predominantly rely on parameter reuse, extending computation through recursive execution. Under this paradigm, the network structure remains static along the training timeline, and additional computational depth is uniformly assigned to entire blocks at the parameter level. This rigidity across training time and parameter space leads to substantial computational redundancy during training. In contrast, we argue that depth allocation during training should not be a static preset, but rather a progressively growing structural process. Our systematic analysis reveals a deep-to-shallow maturation trajectory across layers, where high-entropy attention heads play a crucial role in semantic integration. Motivated by this observation, we introduce the Sparse Growing Transformer (SGT). SGT is a training-time sparse depth allocation framework that progressively extends recurrence from deeper to shallower layers via targeted attention looping on informative heads. This mechanism induces structural sparsity by selectively increasing depth only for a small subset of parameters as training evolves. Extensive experiments across multiple parameter scales demonstrate that SGT consistently outperforms training-time static block-level looping baselines under comparable settings, while reducing the additional training FLOPs overhead from approximately 16--20% to only 1--3% relative to a standard Transformer backbone.
>
---
#### [replaced 062] Right at My Level: A Unified Multilingual Framework for Proficiency-Aware Text Simplification
- **分类: cs.CL**

- **简介: 该论文属于文本简化任务，旨在解决多语言、个性化文本简化问题。通过提出Re-RIGHT框架，无需平行语料即可实现适应不同语言水平的文本简化。**

- **链接: [https://arxiv.org/pdf/2604.05302](https://arxiv.org/pdf/2604.05302)**

> **作者:** Jinhong Jeong; Junghun Park; Youngjae Yu
>
> **备注:** Accepted to ACL 2026
>
> **摘要:** Text simplification supports second language (L2) learning by providing comprehensible input, consistent with the Input Hypothesis. However, constructing personalized parallel corpora is costly, while existing large language model (LLM)-based readability control methods rely on pre-labeled sentence corpora and primarily target English. We propose Re-RIGHT, a unified reinforcement learning framework for adaptive multilingual text simplification without parallel corpus supervision. We first show that prompting-based lexical simplification at target proficiency levels (CEFR, JLPT, TOPIK, and HSK) performs poorly at easier levels and for non-English languages, even with state-of-the-art LLMs such as GPT-5.2 and Gemini 2.5. To address this, we collect 43K vocabulary-level data across four languages (English, Japanese, Korean, and Chinese) and train a compact 4B policy model using Re-RIGHT, which integrates three reward modules: vocabulary coverage, semantic preservation, and coherence. Compared to the stronger LLM baselines, Re-RIGHT achieves higher lexical coverage at target proficiency levels while maintaining original meaning and fluency.
>
---
#### [replaced 063] Cognitive Alpha Mining via LLM-Driven Code-Based Evolution
- **分类: cs.CL**

- **简介: 该论文属于金融量化分析任务，旨在解决高维金融数据中有效预测信号（阿尔法）发现的问题。通过结合代码表示与大语言模型驱动的进化搜索，提出CogAlpha框架，提升预测准确性与可解释性。**

- **链接: [https://arxiv.org/pdf/2511.18850](https://arxiv.org/pdf/2511.18850)**

> **作者:** Fengyuan Liu; Yi Huang; Sichun Luo; Yuqi Wang; Yazheng Yang; Xinye Li; Zefa Hu; Junlan Feng; Qi Liu
>
> **摘要:** Discovering effective predictive signals, or "alphas," from financial data with high dimensionality and extremely low signal-to-noise ratio remains a difficult open problem. Despite progress in deep learning, genetic programming, and, more recently, large language model (LLM)-based factor generation, existing approaches still explore only a narrow region of the vast alpha search space. Neural models tend to produce opaque and fragile patterns, while symbolic or formula-based methods often yield redundant or economically ungrounded expressions that generalize poorly. Although different in form, these paradigms share a key limitation: none can conduct broad, structured, and human-like exploration that balances logical consistency with creative leaps. To address this gap, we introduce the Cognitive Alpha Mining Framework (CogAlpha), which combines code-level alpha representation with LLM-driven reasoning and evolutionary search. Treating LLMs as adaptive cognitive agents, our framework iteratively refines, mutates, and recombines alpha candidates through multi-stage prompts and financial feedback. This synergistic design enables deeper thinking, richer structural diversity, and economically interpretable alpha discovery, while greatly expanding the effective search space. Experiments on 5 stock datasets from 3 stock markets demonstrate that CogAlpha consistently discovers alphas with superior predictive accuracy, robustness, and generalization over existing methods. Our results highlight the promise of aligning evolutionary optimization with LLM-based reasoning for automated and explainable alpha discovery.
>
---
#### [replaced 064] VisRet: Visualization Improves Knowledge-Intensive Text-to-Image Retrieval
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于文本到图像检索任务，旨在解决跨模态对齐不足的问题。通过先生成图像再检索的方法，提升检索效果。**

- **链接: [https://arxiv.org/pdf/2505.20291](https://arxiv.org/pdf/2505.20291)**

> **作者:** Di Wu; Yixin Wan; Kai-Wei Chang
>
> **备注:** ACL 2026 Camera Ready
>
> **摘要:** Text-to-image retrieval (T2I retrieval) remains challenging because cross-modal embeddings often behave as bags of concepts, underrepresenting structured visual relationships such as pose and viewpoint. We proposeVisualize-then-Retrieve (VisRet), a retrieval paradigm that mitigates this limitation of cross-modal similarity alignment. VisRet first projects textual queries into the image modality via T2I generation, then performs retrieval within the image modality to bypass the weaknesses of cross-modal retrievers in recognizing subtle visual-spatial features. Across four benchmarks (Visual-RAG, INQUIRE-Rerank, Microsoft COCO, and our new Visual-RAG-ME featuring multi-entity comparisons), VisRet substantially outperforms cross-modal similarity matching and baselines that recast T2I retrieval as text-to-text similarity matching, improving nDCG@30 by 0.125 on average with CLIP as the retriever and by 0.121 with E5-V. For downstream question answering, VisRet increases accuracy on Visual-RAG and Visual-RAG-ME by 3.8% and 15.7% in top-1 retrieval, and by 3.9% and 11.1% in top-10 retrieval. Ablation studies show compatibility with different T2I instruction LLMs, T2I generation models, and downstream LLMs. VisRet provides a simple yet effective perspective for advancing in text-image retrieval. Our code and the new benchmark are publicly available at this https URL.
>
---
#### [replaced 065] IF-CRITIC: Towards a Fine-Grained LLM Critic for Instruction-Following Evaluation
- **分类: cs.CL**

- **简介: 该论文属于指令遵循评估任务，旨在解决现有评估模型成本高、可靠性差的问题。提出IF-CRITIC，通过细粒度检查清单和偏好优化提升评估效果。**

- **链接: [https://arxiv.org/pdf/2511.01014](https://arxiv.org/pdf/2511.01014)**

> **作者:** Bosi Wen; Yilin Niu; Cunxiang Wang; Pei Ke; Xiaoying Ling; Ying Zhang; Aohan Zeng; Hongning Wang; Minlie Huang
>
> **备注:** ACL 2026
>
> **摘要:** Instruction-following is a fundamental ability of Large Language Models (LLMs), requiring their generated outputs to follow multiple constraints imposed in input instructions. Numerous studies have attempted to enhance this ability through preference optimization or reinforcement learning based on reward signals from LLM-as-a-Judge. However, existing evaluation models for instruction-following still possess many deficiencies, such as substantial costs and unreliable assessments. To this end, we propose IF-CRITIC, an LLM critic for fine-grained, efficient, and reliable instruction-following evaluation. We first develop a checklist generator to decompose instructions and generate constraint checklists. With the assistance of the checklists, we collect high-quality critique training data through a multi-stage critique filtering mechanism and employ a constraint-level preference optimization method to train IF-CRITIC. Extensive experiments show that the evaluation performance of IF-CRITIC can beat strong LLM-as-a-Judge baselines, including o4-mini and Gemini-3-Pro. With the reward signals provided by IF-CRITIC, LLMs can achieve substantial performance gains in instruction-following optimization under lower computational overhead compared to strong LLM critic baselines. Our code and model are available at this https URL.
>
---
#### [replaced 066] CausalEmbed: Auto-Regressive Multi-Vector Generation in Latent Space for Visual Document Embedding
- **分类: cs.CL**

- **简介: 该论文属于视觉文档检索任务，解决多向量嵌入存储开销大的问题。提出CausalEmbed方法，通过自回归生成减少视觉标记数量，提升效率与可扩展性。**

- **链接: [https://arxiv.org/pdf/2601.21262](https://arxiv.org/pdf/2601.21262)**

> **作者:** Jiahao Huo; Yu Huang; Yibo Yan; Ye Pan; Kening Zheng; Wei-Chieh Huang; Yi Cao; Mingdong Ou; Philip S. Yu; Xuming Hu
>
> **备注:** Under review
>
> **摘要:** Although Multimodal Large Language Models (MLLMs) have shown remarkable potential in Visual Document Retrieval (VDR) through generating high-quality multi-vector embeddings, the substantial storage overhead caused by representing a page with thousands of visual tokens limits their practicality in real-world applications. To address this challenge, we propose an auto-regressive generation approach, CausalEmbed, for constructing multi-vector embeddings. By incorporating iterative margin loss during contrastive training, CausalEmbed encourages the embedding models to learn compact and well-structured representations. Our method enables efficient VDR tasks using only dozens of visual tokens, achieving a 30-155x reduction in token count while maintaining highly competitive performance across various backbones and benchmarks. Theoretical analysis and empirical results demonstrate the unique advantages of auto-regressive embedding generation in terms of training efficiency and scalability at test time. As a result, CausalEmbed introduces a flexible test-time scaling strategy for multi-vector VDR representations and sheds light on the generative paradigm within multimodal document retrieval. Our code is available at this https URL.
>
---
#### [replaced 067] Preconditioned Test-Time Adaptation for Out-of-Distribution Debiasing in Narrative Generation
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于自然语言生成任务，旨在解决模型在分布外样本上的偏见问题。提出CAP-TTA框架，在测试时进行自适应调整，降低偏见并提升流畅性。**

- **链接: [https://arxiv.org/pdf/2603.13683](https://arxiv.org/pdf/2603.13683)**

> **作者:** Hanwen Shen; Ting Ying; Jiajie Lu; Shanshan Wang
>
> **备注:** This paper has been accepted to ACL2026 main conference
>
> **摘要:** Although debiased large language models (LLMs) excel at handling known or low-bias prompts, they often fail on unfamiliar and high-bias prompts. We demonstrate via out-of-distribution (OOD) detection that these high-bias prompts cause a distribution shift, degrading static model performance. To enable real-time correction, we propose CAP-TTA, a test-time adaptation framework. CAP-TTA triggers context-aware LoRA updates only when a bias-risk score exceeds a set threshold. By utilizing an offline precomputed diagonal preconditioner, it ensures fast and stable optimization. Across multiple benchmarks and human evaluations, CAP-TTA effectively reduces toxicity/bias score with significantly lower latency than standard optimization methods (e.g., AdamW or SGD). Furthermore, it prevents catastrophic forgetting, and substantially improves narrative fluency over state-of-the-art baselines without compromising debiasing performance.
>
---
#### [replaced 068] A Linguistics-Aware LLM Watermarking via Syntactic Predictability
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI可信治理任务，旨在解决水印强度与文本质量的平衡问题。提出STELA框架，利用语言结构特性动态调整水印，提升检测鲁棒性并支持公开验证。**

- **链接: [https://arxiv.org/pdf/2510.13829](https://arxiv.org/pdf/2510.13829)**

> **作者:** Shinwoo Park; Hyejin Park; Hyeseon An; Yo-Sub Han
>
> **备注:** ACL 2026
>
> **摘要:** As large language models (LLMs) continue to advance rapidly, reliable governance tools have become critical. Publicly verifiable watermarking is particularly essential for fostering a trustworthy AI ecosystem. A central challenge persists: balancing text quality against detection robustness. Recent studies have sought to navigate this trade-off by leveraging signals from model output distributions (e.g., token-level entropy); however, their reliance on these model-specific signals presents a significant barrier to public verification, as the detection process requires access to the logits of the underlying model. We introduce STELA, a novel framework that aligns watermark strength with the linguistic degrees of freedom inherent in language. STELA dynamically modulates the signal using part-of-speech (POS) n-gram-modeled linguistic indeterminacy, weakening it in grammatically constrained contexts to preserve quality and strengthening it in contexts with greater linguistic flexibility to enhance detectability. Our detector operates without access to any model logits, thus facilitating publicly verifiable detection. Through extensive experiments on typologically diverse languages-analytic English, isolating Chinese, and agglutinative Korean-we show that STELA surpasses prior methods in detection robustness. Our code is available at this https URL.
>
---
#### [replaced 069] E2Edev: Benchmarking Large Language Models in End-to-End Software Development Task
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文属于端到端软件开发任务，旨在解决现有基准不足的问题。提出E2EDev基准，包含细粒度需求和自动化测试，以更准确评估LLM在E2ESD中的能力。**

- **链接: [https://arxiv.org/pdf/2510.14509](https://arxiv.org/pdf/2510.14509)**

> **作者:** Jingyao Liu; Chen Huang; Zhizhao Guan; Wenqiang Lei; Yang Deng
>
> **备注:** Accepted to ACL 2026 main
>
> **摘要:** The rapid advancement in large language models (LLMs) has demonstrated significant potential in End-to-End Software Development (E2ESD). However, existing E2ESD benchmarks are limited by coarse-grained requirement specifications and unreliable evaluation protocols, hindering a true understanding of current framework capabilities. To address these limitations, we present E2EDev, a novel benchmark grounded in the principles of Behavior-Driven Development (BDD), which evaluates the capabilities of E2ESD frameworks by assessing whether the generated software meets user needs through mimicking real user interactions (Figure 1). E2EDev comprises (i) a fine-grained set of user requirements, (ii) multiple BDD test scenarios with corresponding Python step implementations for each requirement, and (iii) a fully automated testing pipeline built on the Behave framework. To ensure its quality while reducing the annotation effort, E2EDev leverages our proposed Human-in-the-Loop Multi-Agent Annotation Framework (HITL-MAA). By evaluating various E2ESD frameworks and LLM backbones with E2EDev, our analysis reveals a persistent struggle to effectively solve these tasks, underscoring the critical need for more effective and cost-efficient E2ESD solutions. Our codebase and benchmark are publicly available at this https URL.
>
---
#### [replaced 070] IF-RewardBench: Benchmarking Judge Models for Instruction-Following Evaluation
- **分类: cs.CL**

- **简介: 该论文属于大语言模型评估任务，旨在解决现有裁判模型在指令遵循评估中的可靠性问题。提出IF-RewardBench基准，构建偏好图进行多响应排序评估。**

- **链接: [https://arxiv.org/pdf/2603.04738](https://arxiv.org/pdf/2603.04738)**

> **作者:** Bosi Wen; Yilin Niu; Cunxiang Wang; Xiaoying Ling; Ying Zhang; Pei Ke; Hongning Wang; Minlie Huang
>
> **备注:** ACL 2026
>
> **摘要:** Instruction-following is a foundational capability of large language models (LLMs), with its improvement hinging on scalable and accurate feedback from judge models. However, the reliability of current judge models in instruction-following remains underexplored due to several deficiencies of existing meta-evaluation benchmarks, such as their insufficient data coverage and oversimplified pairwise evaluation paradigms that misalign with model optimization scenarios. To this end, we propose IF-RewardBench, a comprehensive meta-evaluation benchmark for instruction-following that covers diverse instruction and constraint types. For each instruction, we construct a preference graph containing all pairwise preferences among multiple responses based on instruction-following quality. This design enables a listwise evaluation paradigm that assesses the capabilities of judge models to rank multiple responses, which is essential in guiding model alignment. Extensive experiments on IF-RewardBench reveal significant deficiencies in current judge models and demonstrate that our benchmark achieves a stronger positive correlation with downstream task performance compared to existing benchmarks. Our codes and data are available at this https URL.
>
---
#### [replaced 071] Improving Language Models with Intentional Analysis
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决语言模型中意图理解不足的问题。通过引入意图分析（IA），提升模型推理能力，并验证其有效性与通用性。**

- **链接: [https://arxiv.org/pdf/2502.04689](https://arxiv.org/pdf/2502.04689)**

> **作者:** Yuwei Yin; Giuseppe Carenini
>
> **备注:** Code at this https URL
>
> **摘要:** Intent, a critical cognitive notion and mental state, is ubiquitous in human communication and problem-solving. Accurately understanding the underlying intent behind questions is imperative to reasoning towards correct answers. However, this significant concept has been largely disregarded in the rapid development of language models (LMs). To unleash the potential of intent and instill it into LMs, this paper introduces Intentional Analysis (IA), which explicitly invokes intent-aware analysis and reasoning during the problem-solving process. Comprehensive experiments across diverse benchmarks, model types, and configurations demonstrate the effectiveness, robustness, and generalizability of IA. Notably, IA consistently improves task performance even on SOTA proprietary models like GPT-5 and Claude-Opus-4.6. Moreover, IA not only outperforms Chain-of-Thought (CoT) across various experimental settings, but it can also synergistically work with CoT reasoning. Further qualitative analysis and case studies reveal that the benefits of IA stem from addressing several weaknesses in baseline methods, such as intent misunderstanding, hasty generalization, and mental laziness. Case studies also provide insights into the mechanisms underlying IA and clarify how it differs from CoT in mitigating these weaknesses. This study sheds light on a promising direction for the development of future LLMs with intentional analysis.
>
---
#### [replaced 072] Anonpsy: A Graph-Based Framework for Structure-Preserving De-identification of Psychiatric Narratives
- **分类: cs.CL**

- **简介: 该论文属于隐私保护任务，旨在解决精神病历中身份泄露问题。通过构建语义图并进行结构化改写，实现去标识化同时保持临床结构。**

- **链接: [https://arxiv.org/pdf/2601.13503](https://arxiv.org/pdf/2601.13503)**

> **作者:** Kyung Ho Lim; Byung-Hoon Kim
>
> **备注:** ACL 2026 Findings
>
> **摘要:** Psychiatric narratives encode patient identity not only through explicit identifiers but also through idiosyncratic life events embedded in their clinical structure. Existing de-identification approaches, including PHI masking and LLM-based synthetic rewriting, operate at the text level and offer limited control over which semantic elements are preserved or altered. We introduce Anonpsy, a de-identification framework that reformulates the task as graph-guided semantic rewriting. Anonpsy (1) converts each narrative into a semantic graph encoding clinical entities, temporal anchors, and typed relations; (2) applies graph-constrained perturbations that modify identifying context while preserving clinically essential structure; and (3) regenerates text via graph-conditioned LLM generation. Evaluated on 90 clinician-authored psychiatric case narratives, Anonpsy preserves diagnostic fidelity while achieving consistently low re-identification risk under expert, semantic, and GPT-5-based evaluations. Compared with a strong LLM-only rewriting baseline, Anonpsy yields substantially lower semantic similarity and identifiability. These results demonstrate that explicit structural representations combined with constrained generation provide an effective approach to de-identification for psychiatric narratives.
>
---
#### [replaced 073] Hidden Measurement Error in LLM Pipelines Distorts Annotation, Evaluation, and Benchmarking
- **分类: cs.CL**

- **简介: 该论文研究LLM评估中的隐性测量误差问题，属于模型评估任务。针对评估结果的不确定性，提出分解误差来源并优化设计以减少误差。**

- **链接: [https://arxiv.org/pdf/2604.11581](https://arxiv.org/pdf/2604.11581)**

> **作者:** Solomon Messing
>
> **摘要:** LLM evaluations drive which models get deployed, which safety standards get adopted, and which research conclusions get published. Yet these scores carry hidden uncertainty: rephrasing the prompt, switching the judge model, or changing the temperature can shift results enough to flip rankings and reverse conclusions. Standard confidence intervals ignore this variance, producing under-coverage that worsens with more data. The same unmeasured variance creates an exploitable surface for benchmarks: model developers can optimize against measurement noise rather than genuine performance (some have infamously done so, see \citep{boyeau2025leaderboard}). This paper decomposes LLM pipeline uncertainty into its sources, distinguishes variance that shrinks with more data from sensitivity to researcher design choices, and uses design-study projections to reduce total error. Across ideology annotation, safety classification, MMLU benchmarking, and a human-validated propaganda audit, the decomposition reveals that the dominant variance source differs by domain and scoring method. On MMLU, optimized budget allocation halves estimation error at equivalent cost. On the propaganda task, the recommended pipeline outperforms 73\% of single-configuration alternatives against a human baseline. A small-sample pilot is sufficient to derive confidence intervals that approach nominal coverage and to identify which design changes yield the largest precision gains.
>
---
#### [replaced 074] Latent-Condensed Transformer for Efficient Long Context Modeling
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决长文本处理中计算成本高和缓存占用大的问题。提出LCA方法，在降低计算和缓存的同时保持性能。**

- **链接: [https://arxiv.org/pdf/2604.12452](https://arxiv.org/pdf/2604.12452)**

> **作者:** Zeng You; Yaofo Chen; Qiuwu Chen; Ying Sun; Shuhai Zhang; Yingjian Li; Yaowei Wang; Mingkui Tan
>
> **备注:** Accepted by ACL 2026
>
> **摘要:** Large language models (LLMs) face significant challenges in processing long contexts due to the linear growth of the key-value (KV) cache and quadratic complexity of self-attention. Existing approaches address these bottlenecks separately: Multi-head Latent Attention (MLA) reduces the KV cache by projecting tokens into a low-dimensional latent space, while sparse attention reduces computation. However, sparse methods cannot operate natively on MLA's compressed latent structure, missing opportunities for joint optimization. In this paper, we propose Latent-Condensed Attention (LCA), which directly condenses context within MLA's latent space, where the representation is disentangled into semantic latent vectors and positional keys. LCA separately aggregates semantic vectors via query-aware pooling and preserves positional keys via anchor selection. This approach jointly reduces both computational cost and KV cache without adding parameters. Beyond MLA, LCA's design is architecture-agnostic and readily extends to other attention mechanisms such as GQA. Theoretically, we prove a length-independent error bound. Experiments show LCA achieves up to 2.5$\times$ prefilling speedup and 90% KV cache reduction at 128K context while maintaining competitive performance.
>
---
#### [replaced 075] DA-Cramming: Enhancing Cost-Effective Language Model Pretraining with Dependency Agreement Integration
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，旨在降低语言模型预训练成本。针对预训练计算成本高的问题，提出DA-Cramming方法，通过整合依存关系信息提升预训练效果。**

- **链接: [https://arxiv.org/pdf/2311.04799](https://arxiv.org/pdf/2311.04799)**

> **作者:** Martin Kuo; Jianyi Zhang; Dongting Li; Yiran Chen
>
> **摘要:** Pretraining language models is still a challenge for many researchers due to its substantial computational costs. As such, there is growing interest in developing more affordable pretraining methods. One notable advancement in this area is the Cramming technique (Geiping and Goldstein, 2022), which enables the pretraining of BERT-style language models using just one GPU in a single day. Building on this innovative approach, we introduce the Dependency Agreement Cramming (DA-Cramming), an efficient framework that integrates information about dependency agreements into the pretraining process. Unlike existing methods that leverage similar semantic information during finetuning, our approach represents a pioneering effort focusing on enhancing the foundational language understanding with semantic information during pretraining. We meticulously design a dual-stage pretraining work flow with four dedicated submodels to capture representative dependency agreements at the chunk level, effectively transforming these agreements into embeddings to benefit the pretraining. Extensive empirical results demonstrate that our method significantly outperforms previous methods across various tasks.
>
---
#### [replaced 076] From Plausible to Causal: Counterfactual Semantics for Policy Evaluation in Simulated Online Communities
- **分类: cs.CL**

- **简介: 该论文属于政策评估任务，解决仿真社区中因果关系缺失的问题。提出使用反事实框架区分必要与充分因果，提升政策分析的准确性。**

- **链接: [https://arxiv.org/pdf/2604.03920](https://arxiv.org/pdf/2604.03920)**

> **作者:** Agam Goyal; Yian Wang; Eshwar Chandrasekharan; Hari Sundaram
>
> **备注:** Accepted to PoliSim@CHI'26: 6 pages, 1 table (Best Paper Award)
>
> **摘要:** LLM-based social simulations can generate believable community interactions, enabling ``policy wind tunnels'' where governance interventions are tested before deployment. But believability is not causality. Claims like ``intervention $A$ reduces escalation'' require causal semantics that current simulation work typically does not specify. We propose adopting the causal counterfactual framework, distinguishing \textit{necessary causation} (would the outcome have occurred without the intervention?) from \textit{sufficient causation} (does the intervention reliably produce the outcome?). This distinction maps onto different stakeholder needs: moderators diagnosing incidents require evidence about necessity, while platform designers choosing policies require evidence about sufficiency. We formalize this mapping, show how simulation design can support estimation under explicit assumptions, and argue that the resulting quantities should be interpreted as simulator-conditional causal estimates whose policy relevance depends on simulator fidelity. Establishing this framework now is essential: it helps define what adequate fidelity means and moves the field from simulations that look realistic toward simulations that can support policy changes.
>
---
#### [replaced 077] Style Amnesia: Investigating Speaking Style Degradation and Mitigation in Multi-Turn Spoken Language Models
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于自然语言处理任务，研究多轮对话中语音模型风格保持问题，揭示风格遗忘现象并提出缓解方法。**

- **链接: [https://arxiv.org/pdf/2512.23578](https://arxiv.org/pdf/2512.23578)**

> **作者:** Yu-Xiang Lin; Cheng-Han Chiang; Hung-yi Lee
>
> **备注:** ACL 2026 Findings
>
> **摘要:** In this paper, we show that when spoken language models (SLMs) are instructed to speak in a specific speaking style at the beginning of a multi-turn conversation, they cannot maintain the required speaking styles after several turns of interaction; we refer to this as the style amnesia of SLMs. We focus on paralinguistic speaking styles, including emotion, accent, volume, and speaking speed. We evaluate three proprietary and two open-source SLMs, demonstrating that none of these models can maintain a consistent speaking style when instructed to do so. We further show that while SLMs can recall the style instruction when prompted in later turns, they still fail to express it, but through explicit recall can mitigate style amnesia. In addition, SLMs struggle more when the style instruction is placed in system messages rather than user messages, even though system messages are specifically designed to provide persistent, conversation-level instructions. Our findings highlight a systematic gap in current SLMs' ability to maintain speaking styles, highlighting the need for improved style adherence in future models. Our code and evaluation data are publicly available at this https URL.
>
---
#### [replaced 078] Your LLM Agents are Temporally Blind: The Misalignment Between Tool Use Decisions and Human Time Perception
- **分类: cs.CL**

- **简介: 该论文研究LLM在时间感知上的不足，属于自然语言处理任务。解决模型与人类时间感知不一致的问题，通过构建数据集和实验分析，提出对齐方法。**

- **链接: [https://arxiv.org/pdf/2510.23853](https://arxiv.org/pdf/2510.23853)**

> **作者:** Yize Cheng; Arshia Soltani Moakhar; Chenrui Fan; Parsa Hosseini; Kazem Faghih; Zahra Sodagar; Wenxiao Wang; Soheil Feizi
>
> **备注:** ACL 2026 (findings), Camera-ready
>
> **摘要:** Large language model (LLM) agents are increasingly used to interact with and execute tasks in dynamic environments. However, a critical yet overlooked limitation of these agents is that they, by default, assume a stationary context, failing to account for the real-world time elapsed between messages. We refer to this as "temporal blindness". This limitation hinders decisions about when to invoke tools, leading agents to either over-rely on stale context and skip needed tool calls, or under-rely on it and redundantly repeat tool calls. To study this challenge, we constructed TicToc, a diverse dataset of multi-turn user-agent message trajectories across 76 scenarios, spanning dynamic environments with high, medium, and low time sensitivity. We collected human preferences between "calling a tool" and "directly answering" on each sample, and evaluated how well LLM tool-calling decisions align with human preferences under varying amounts of elapsed time. Our analysis reveals that existing models display poor alignment with human temporal perception, with no model achieving a normalized alignment rate better than 65% when given time stamp information. We also show that naive, prompt-based alignment techniques have limited effectiveness for most models, but specific post-training alignment can be a viable way to align multi-turn LLM tool use with human temporal perception. Our data and findings provide a first step toward understanding and mitigating temporal blindness, offering insights to foster the development of more time-aware and human-aligned agents.
>
---
#### [replaced 079] Beyond Literal Mapping: Benchmarking and Improving Non-Literal Translation Evaluation
- **分类: cs.CL**

- **简介: 该论文属于机器翻译评估任务，旨在解决非字面翻译评价不准确的问题。研究构建了MENT数据集，提出RATE框架以提升评估效果。**

- **链接: [https://arxiv.org/pdf/2601.07338](https://arxiv.org/pdf/2601.07338)**

> **作者:** Yanzhi Tian; Cunxiang Wang; Zeming Liu; Heyan Huang; Wenbo Yu; Dawei Song; Jie Tang; Yuhang Guo
>
> **备注:** Accepted to ACL 2026 Main Conference
>
> **摘要:** Large Language Models (LLMs) have significantly advanced Machine Translation (MT), applying them to linguistically complex domains-such as Social Network Services, literature etc. In these scenarios, translations often require handling non-literal expressions, leading to the inaccuracy of MT metrics. To systematically investigate the reliability of MT metrics, we first curate a meta-evaluation dataset focused on non-literal translations, namely MENT. MENT encompasses four non-literal translation domains and features source sentences paired with translations from diverse MT systems, with 7,530 human-annotated scores on translation quality. Experimental results reveal the inaccuracies of traditional MT metrics and the limitations of LLM-as-a-Judge, particularly the knowledge cutoff and score inconsistency problem. To mitigate these limitations, we propose RATE, a novel agentic translation evaluation framework, centered by a reflective Core Agent that dynamically invokes specialized sub-agents. Experimental results indicate the efficacy of RATE, achieving an improvement of at least 3.2 points in combined system- and segment-level correlation with human judgments compared with current methods. Further experiments demonstrate the robustness of RATE to general-domain MT evaluation. Code and dataset are available at: this https URL.
>
---
#### [replaced 080] AccelOpt: A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出AccelOpt，一个自提升的LLM代理系统，用于优化AI加速器内核。解决传统依赖专家知识的问题，通过迭代生成和经验学习提升性能。**

- **链接: [https://arxiv.org/pdf/2511.15915](https://arxiv.org/pdf/2511.15915)**

> **作者:** Genghan Zhang; Shaowei Zhu; Anjiang Wei; Zhenyu Song; Allen Nie; Zhen Jia; Nandita Vijaykumar; Yida Wang; Kunle Olukotun
>
> **摘要:** We present AccelOpt, a self-improving large language model (LLM) agentic system that autonomously optimizes kernels for emerging AI acclerators, eliminating the need for expert-provided hardware-specific optimization knowledge. AccelOpt explores the kernel optimization space through iterative generation, informed by an optimization memory that curates experiences and insights from previously encountered slow-fast kernel pairs. We build NKIBench, a new benchmark suite of AWS Trainium accelerator kernels with varying complexity extracted from real-world LLM workloads to evaluate the effectiveness of AccelOpt. Our evaluation confirms that AccelOpt's capability improves over time, boosting the average percentage of peak throughput from $49\%$ to $61\%$ on Trainium 1 and from $45\%$ to $59\%$ on Trainium 2 for NKIBench kernels. Moreover, AccelOpt is highly cost-effective: using open-source models, it matches the kernel improvements of Claude Sonnet 4 while being $26\times$ cheaper. The code is open-sourced at this https URL.
>
---
#### [replaced 081] OccuBench: Evaluating AI Agents on Real-World Professional Tasks via Language Environment Simulation
- **分类: cs.CL**

- **简介: 该论文提出OccuBench，用于评估AI代理在真实职业任务中的表现。解决现有基准覆盖不足的问题，通过语言环境模拟器构建评估场景，分析任务完成度与环境鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.10866](https://arxiv.org/pdf/2604.10866)**

> **作者:** Xiaomeng Hu; Yinger Zhang; Fei Huang; Jianhong Tu; Yang Su; Lianghao Deng; Yuxuan Liu; Yantao Liu; Dayiheng Liu; Tsung-Yi Ho
>
> **备注:** 23 pages, 8 figures, 2 tables. Project page: this https URL
>
> **摘要:** AI agents are expected to perform professional work across hundreds of occupational domains (from emergency department triage to nuclear reactor safety monitoring to customs import processing), yet existing benchmarks can only evaluate agents in the few domains where public environments exist. We introduce OccuBench, a benchmark covering 100 real-world professional task scenarios across 10 industry categories and 65 specialized domains, enabled by Language Environment Simulators (LESs) that simulate domain-specific environments through LLM-driven tool response generation. Our multi-agent synthesis pipeline automatically produces evaluation instances with guaranteed solvability, calibrated difficulty, and document-grounded diversity. OccuBench evaluates agents along two complementary dimensions: task completion across professional domains and environmental robustness under controlled fault injection (explicit errors, implicit data degradation, and mixed faults). We evaluate 15 frontier models across 8 model families and find that: (1) no single model dominates all industries, as each has a distinct occupational capability profile; (2) implicit faults (truncated data, missing fields) are harder than both explicit errors (timeouts, 500s) and mixed faults, because they lack overt error signals and require the agent to independently detect data degradation; (3) larger models, newer generations, and higher reasoning effort consistently improve performance. GPT-5.2 improves by 27.5 points from minimal to maximum reasoning effort; and (4) strong agents are not necessarily strong environment simulators. Simulator quality is critical for LES-based evaluation reliability. OccuBench provides the first systematic cross-industry evaluation of AI agents on professional occupational tasks.
>
---
#### [replaced 082] Why Supervised Fine-Tuning Fails to Learn: A Systematic Study of Incomplete Learning in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，研究监督微调中出现的不完全学习现象，分析其成因并提出诊断方法。**

- **链接: [https://arxiv.org/pdf/2604.10079](https://arxiv.org/pdf/2604.10079)**

> **作者:** Chao Xue; Yao Wang; Mengqiao Liu; Di Liang; Xingsheng Han; Peiyang Liu; Xianjie Wu; Chenyao Lu; Lei Jiang; Yu Lu; Haibo Shi; Shuang Liang; Minlong Peng; Flora D. Salim
>
> **备注:** Accepted by ACL 2026 Main
>
> **摘要:** Supervised Fine-Tuning (SFT) is the standard approach for adapting large language models (LLMs) to downstream tasks. However, we observe a persistent failure mode: even after convergence, models often fail to correctly reproduce a subset of their own supervised training data. We refer to this behavior as the Incomplete Learning Phenomenon(ILP). This paper presents the first systematic study of ILP in LLM fine-tuning. We formalize ILP as post-training failure to internalize supervised instances and demonstrate its prevalence across multiple model families, domains, and datasets. Through controlled analyses, we identify five recurrent sources of incomplete learning: (1) missing prerequisite knowledge in the pre-trained model, (2) conflicts between SFT supervision and pre-training knowledge, (3) internal inconsistencies within SFT data, (4) left-side forgetting during sequential fine-tuning, and (5) insufficient optimization for rare or complex patterns. We introduce a diagnostic-first framework that maps unlearned samples to these causes using observable training and inference signals, and study several targeted mitigation strategies as causal interventions. Experiments on Qwen, LLaMA, and OLMo2 show that incomplete learning is widespread and heterogeneous, and that improvements in aggregate metrics can mask persistent unlearned subsets. The findings highlight the need for fine-grained diagnosis of what supervised fine-tuning fails to learn, and why.
>
---
#### [replaced 083] Towards Bridging the Reward-Generation Gap in Direct Alignment Algorithms
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于大语言模型对齐任务，旨在解决DAAs中的"奖励生成差距"问题。通过引入POET方法，提升训练目标与解码动态的一致性，改善模型性能。**

- **链接: [https://arxiv.org/pdf/2506.09457](https://arxiv.org/pdf/2506.09457)**

> **作者:** Zeguan Xiao; Yun Chen; Guanhua Chen; Ke Tang
>
> **备注:** Findings of ACL 2026
>
> **摘要:** Direct Alignment Algorithms (DAAs), such as Direct Preference Optimization (DPO) and Simple Preference Optimization (SimPO), have emerged as efficient alternatives to Reinforcement Learning from Human Feedback (RLHF) algorithms for aligning large language models (LLMs) with human preferences. However, DAAs suffer from a fundamental limitation we identify as the "reward-generation gap", a discrepancy between training objectives and autoregressive decoding dynamics. In this paper, we consider that one contributor to the reward-generation gap is the mismatch between the inherent importance of prefix tokens during the LLM generation process and how this importance is reflected in the implicit reward functions of DAAs. To bridge the gap, we adopt a token-level MDP perspective of DAAs to analyze its limitations and introduce a simple yet effective approach called Prefix-Oriented Equal-length Training (POET), which truncates both preferred and dispreferred responses to match the shorter one's length. We conduct experiments with DPO and SimPO, two representative DAAs, demonstrating that POET improves over their standard implementations, achieving up to 11.8 points in AlpacaEval 2 and overall improvements across downstream tasks. These results underscore the need to mitigate the reward-generation gap in DAAs by better aligning training objectives with autoregressive decoding dynamics.
>
---
#### [replaced 084] IROSA: Interactive Robot Skill Adaptation using Natural Language
- **分类: cs.RO; cs.AI; cs.CL; cs.HC; cs.LG**

- **简介: 该论文属于机器人技能适应任务，旨在通过自然语言实现机器人技能的灵活调整。工作包括提出一个框架，利用预训练语言模型选择工具，无需微调即可完成轨迹修正、避障等操作。**

- **链接: [https://arxiv.org/pdf/2603.03897](https://arxiv.org/pdf/2603.03897)**

> **作者:** Markus Knauer; Samuel Bustamante; Thomas Eiband; Alin Albu-Schäffer; Freek Stulp; João Silvério
>
> **备注:** Accepted IEEE Robotics and Automation Letters (RA-L) journal, 8 pages, 5 figures, 3 tables, 1 listing. Code available: this https URL
>
> **摘要:** Foundation models have demonstrated impressive capabilities across diverse domains, while imitation learning provides principled methods for robot skill adaptation from limited data. Combining these approaches holds significant promise for direct application to robotics, yet this combination has received limited attention, particularly for industrial deployment. We present a novel framework that enables open-vocabulary skill adaptation through a tool-based architecture, maintaining a protective abstraction layer between the language model and robot hardware. Our approach leverages pre-trained LLMs to select and parameterize specific tools for adapting robot skills without requiring fine-tuning or direct model-to-robot interaction. We demonstrate the framework on a 7-DoF torque-controlled robot performing an industrial bearing ring insertion task, showing successful skill adaptation through natural language commands for speed adjustment, trajectory correction, and obstacle avoidance while maintaining safety, transparency, and interpretability.
>
---
