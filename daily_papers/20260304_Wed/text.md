# 自然语言处理 cs.CL

- **最新发布 62 篇**

- **更新 73 篇**

## 最新发布

#### [new 001] Characterizing Memorization in Diffusion Language Models: Generalized Extraction and Sampling Effects
- **分类: cs.CL**

- **简介: 该论文研究扩散语言模型的记忆行为，解决其隐私泄露问题。通过理论分析与实验，证明扩散模型在特定条件下记忆风险较低。**

- **链接: [https://arxiv.org/pdf/2603.02333](https://arxiv.org/pdf/2603.02333)**

> **作者:** Xiaoyu Luo; Wenrui Yu; Qiongxiu Li; Johannes Bjerva
>
> **备注:** 21 pages, 9 figures
>
> **摘要:** Autoregressive language models (ARMs) have been shown to memorize and occasionally reproduce training data verbatim, raising concerns about privacy and copyright liability. Diffusion language models (DLMs) have recently emerged as a competitive alternative, yet their memorization behavior remains largely unexplored due to fundamental differences in generation dynamics. To address this gap, we present a systematic theoretical and empirical characterization of memorization in DLMs. We propose a generalized probabilistic extraction framework that unifies prefix-conditioned decoding and diffusion-based generation under arbitrary masking patterns and stochastic sampling trajectories. Theorem 4.3 establishes a monotonic relationship between sampling resolution and memorization: increasing resolution strictly increases the probability of exact training data extraction, implying that autoregressive decoding corresponds to a limiting case of diffusion-based generation by setting the sampling resolution maximal. Extensive experiments across model scales and sampling strategies validate our theoretical predictions. Under aligned prefix-conditioned evaluations, we further demonstrate that DLMs exhibit substantially lower memorization-based leakage of personally identifiable information (PII) compared to ARMs.
>
---
#### [new 002] Universal Conceptual Structure in Neural Translation: Probing NLLB-200's Multilingual Geometry
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究神经机器翻译模型是否学习语言通用的概念表征，通过分析NLLB-200的嵌入几何，揭示其隐含的语言谱系结构和普遍概念关联。**

- **链接: [https://arxiv.org/pdf/2603.02258](https://arxiv.org/pdf/2603.02258)**

> **作者:** Kyle Elliott Mathewson
>
> **备注:** 14 figures; code and interactive toolkit available at this https URL
>
> **摘要:** Do neural machine translation models learn language-universal conceptual representations, or do they merely cluster languages by surface similarity? We investigate this question by probing the representation geometry of Meta's NLLB-200, a 200-language encoder-decoder Transformer, through six experiments that bridge NLP interpretability with cognitive science theories of multilingual lexical organization. Using the Swadesh core vocabulary list embedded across 135 languages, we find that the model's embedding distances significantly correlate with phylogenetic distances from the Automated Similarity Judgment Program ($\rho = 0.13$, $p = 0.020$), demonstrating that NLLB-200 has implicitly learned the genealogical structure of human languages. We show that frequently colexified concept pairs from the CLICS database exhibit significantly higher embedding similarity than non-colexified pairs ($U = 42656$, $p = 1.33 \times 10^{-11}$, $d = 0.96$), indicating that the model has internalized universal conceptual associations. Per-language mean-centering of embeddings improves the between-concept to within-concept distance ratio by a factor of 1.19, providing geometric evidence for a language-neutral conceptual store analogous to the anterior temporal lobe hub identified in bilingual neuroimaging. Semantic offset vectors between fundamental concept pairs (e.g., man to woman, big to small) show high cross-lingual consistency (mean cosine = 0.84), suggesting that second-order relational structure is preserved across typologically diverse languages. We release InterpretCognates, an open-source interactive toolkit for exploring these phenomena, alongside a fully reproducible analysis pipeline.
>
---
#### [new 003] A Browser-based Open Source Assistant for Multimodal Content Verification
- **分类: cs.CL**

- **简介: 该论文提出一种浏览器插件工具，用于多模态内容验证，解决虚假信息快速识别问题。整合NLP模型，为用户提供可操作的可信度信号。**

- **链接: [https://arxiv.org/pdf/2603.02842](https://arxiv.org/pdf/2603.02842)**

> **作者:** Rosanna Milner; Michael Foster; Olesya Razuvayevskaya; Ian Roberts; Valentin Porcellini; Denis Teyssou; Kalina Bontcheva
>
> **摘要:** Disinformation and false content produced by generative AI pose a significant challenge for journalists and fact-checkers who must rapidly verify digital media information. While there is an abundance of NLP models for detecting credibility signals such as persuasion techniques, subjectivity, or machine-generated text, such methods often remain inaccessible to non-expert users and are not integrated into their daily workflows as a unified framework. This paper demonstrates the VERIFICATION ASSISTANT, a browser-based tool designed to bridge this gap. The VERIFICATION ASSISTANT, a core component of the widely adopted VERIFICATION PLUGIN (140,000+ users), allows users to submit URLs or media files to a unified interface. It automatically extracts content and routes it to a suite of backend NLP classifiers, delivering actionable credibility signals, estimating AI-generated content, and providing other verification guidance in a clear, easy-to-digest format. This paper showcases the tool architecture, its integration of multiple NLP services, and its real-world application to detecting disinformation.
>
---
#### [new 004] BeyondSWE: Can Current Code Agent Survive Beyond Single-Repo Bug Fixing?
- **分类: cs.CL; cs.SE**

- **简介: 该论文属于代码生成任务，旨在解决现有代码代理在跨仓库推理、领域问题解决等实际挑战中的不足。提出BeyondSWE基准，评估模型在多种真实场景下的表现。**

- **链接: [https://arxiv.org/pdf/2603.03194](https://arxiv.org/pdf/2603.03194)**

> **作者:** Guoxin Chen; Fanzhe Meng; Jiale Zhao; Minghao Li; Daixuan Cheng; Huatong Song; Jie Chen; Yuzhi Lin; Hui Chen; Xin Zhao; Ruihua Song; Chang Liu; Cheng Chen; Kai Jia; Ji-Rong Wen
>
> **备注:** Benchmark: this https URL. Repo: this https URL. Scaffold: this https URL
>
> **摘要:** Current benchmarks for code agents primarily assess narrow, repository-specific fixes, overlooking critical real-world challenges such as cross-repository reasoning, domain-specialized problem solving, dependency-driven migration, and full-repository generation. To address this gap, we introduce BeyondSWE, a comprehensive benchmark that broadens existing evaluations along two axes - resolution scope and knowledge scope - using 500 real-world instances across four distinct settings. Experimental results reveal a significant capability gap: even frontier models plateau below 45% success, and no single model performs consistently across task types. To systematically investigate the role of external knowledge, we develop SearchSWE, a framework that integrates deep search with coding abilities. Our experiments show that search augmentation yields inconsistent gains and can in some cases degrade performance, highlighting the difficulty of emulating developer-like workflows that interleave search and reasoning during coding tasks. This work offers both a realistic, challenging evaluation benchmark and a flexible framework to advance research toward more capable code agents.
>
---
#### [new 005] Real-Time Generation of Game Video Commentary with Multimodal LLMs: Pause-Aware Decoding Approaches
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于实时视频解说生成任务，解决如何在不微调模型的情况下实现语义相关且时间合适的解说生成。提出两种基于提示的解码策略，提升实时性与自然度。**

- **链接: [https://arxiv.org/pdf/2603.02655](https://arxiv.org/pdf/2603.02655)**

> **作者:** Anum Afzal; Yuki Saito; Hiroya Takamura; Katsuhito Sudoh; Shinnosuke Takamichi; Graham Neubig; Florian Matthes; Tatsuya Ishigaki
>
> **备注:** Accepted at LREC2026
>
> **摘要:** Real-time video commentary generation provides textual descriptions of ongoing events in videos. It supports accessibility and engagement in domains such as sports, esports, and livestreaming. Commentary generation involves two essential decisions: what to say and when to say it. While recent prompting-based approaches using multimodal large language models (MLLMs) have shown strong performance in content generation, they largely ignore the timing aspect. We investigate whether in-context prompting alone can support real-time commentary generation that is both semantically relevant and well-timed. We propose two prompting-based decoding strategies: 1) a fixed-interval approach, and 2) a novel dynamic interval-based decoding approach that adjusts the next prediction timing based on the estimated duration of the previous utterance. Both methods enable pause-aware generation without any fine-tuning. Experiments on Japanese and English datasets of racing and fighting games show that the dynamic interval-based decoding can generate commentary more closely aligned with human utterance timing and content using prompting alone. We release a multilingual benchmark dataset, trained models, and implementations to support future research on real-time video commentary generation.
>
---
#### [new 006] PrivMedChat: End-to-End Differentially Private RLHF for Medical Dialogue Systems
- **分类: cs.CL**

- **简介: 该论文属于医疗对话系统任务，旨在解决敏感数据隐私泄露问题。通过引入差分隐私的强化学习框架PrivMedChat，提升模型安全性与可靠性。**

- **链接: [https://arxiv.org/pdf/2603.03054](https://arxiv.org/pdf/2603.03054)**

> **作者:** Sudip Bhujel
>
> **摘要:** Large language models are increasingly used for patient-facing medical assistance and clinical decision support, but adapting them to clinical dialogue often requires supervision derived from doctor-patient conversations that may contain sensitive information. Conventional supervised fine-tuning and reinforcement learning from human feedback (RLHF) can amplify memorization risks, enabling empirical membership inference and extraction of rare training-set content. We present PrivMedChat, an end-to-end framework for differentially private RLHF (DP-RLHF) for medical dialogue. Our design enforces differential privacy at every training stage that directly accesses dialogue-derived supervision: (i) Differential Private Stochastic Gradient Descent (DP-SGD) for medical SFT and (ii) DP-SGD for reward model learning from preference pairs. To limit additional privacy expenditure during alignment, we apply DP-SGD to the PPO actor and critic when operating on dialogue-derived prompts, while the reward model remains fixed after DP training. We also introduce an annotation-free preference construction strategy that pairs physician responses with filtered non-expert generations to produce scalable preference data without clinician labeling. Experiments on medical dialogue benchmarks show that PrivMedChat at $\varepsilon=7$ achieves the highest ROUGE-L of 0.156 among all DP models, reduces clinical hallucinations to 1.4% and harmful advice to 0.4%, and obtains the highest overall score of 2.86 in a 3-model LLM-jury evaluation, while producing membership-inference signals that are near chance (AUC 0.510-0.555). We open-source our code at this https URL.
>
---
#### [new 007] Code2Math: Can Your Code Agent Effectively Evolve Math Problems Through Exploration?
- **分类: cs.CL**

- **简介: 该论文属于数学问题生成任务，旨在解决高质量数学题稀缺的问题。通过代码代理自动生成更复杂的新问题，验证其可解性和难度。**

- **链接: [https://arxiv.org/pdf/2603.03202](https://arxiv.org/pdf/2603.03202)**

> **作者:** Dadi Guo; Yuejin Xie; Qingyu Liu; Jiayu Liu; Zhiyuan Fan; Qihan Ren; Shuai Shao; Tianyi Zhou; Dongrui Liu; Yi R. Fung
>
> **备注:** Under review in ICML 2026
>
> **摘要:** As large language models (LLMs) advance their mathematical capabilities toward the IMO level, the scarcity of challenging, high-quality problems for training and evaluation has become a significant bottleneck. Simultaneously, recent code agents have demonstrated sophisticated skills in agentic coding and reasoning, suggesting that code execution can serve as a scalable environment for mathematical experimentation. In this paper, we investigate the potential of code agents to autonomously evolve existing math problems into more complex variations. We introduce a multi-agent framework designed to perform problem evolution while validating the solvability and increased difficulty of the generated problems. Our experiments demonstrate that, given sufficient test-time exploration, code agents can synthesize new, solvable problems that are structurally distinct from and more challenging than the originals. This work provides empirical evidence that code-driven agents can serve as a viable mechanism for synthesizing high-difficulty mathematical reasoning problems within scalable computational environments. Our data is available at this https URL.
>
---
#### [new 008] How Controllable Are Large Language Models? A Unified Evaluation across Behavioral Granularities
- **分类: cs.CL; cs.AI; cs.HC; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型行为不可控的问题。通过构建SteerEval基准，评估模型在不同层次上的可控性，揭示控制效果在细粒度层面下降的现象。**

- **链接: [https://arxiv.org/pdf/2603.02578](https://arxiv.org/pdf/2603.02578)**

> **作者:** Ziwen Xu; Kewei Xu; Haoming Xu; Haiwen Hong; Longtao Huang; Hui Xue; Ningyu Zhang; Yongliang Shen; Guozhou Zheng; Huajun Chen; Shumin Deng
>
> **备注:** Work in progress
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed in socially sensitive domains, yet their unpredictable behaviors, ranging from misaligned intent to inconsistent personality, pose significant risks. We introduce SteerEval, a hierarchical benchmark for evaluating LLM controllability across three domains: language features, sentiment, and personality. Each domain is structured into three specification levels: L1 (what to express), L2 (how to express), and L3 (how to instantiate), connecting high-level behavioral intent to concrete textual output. Using SteerEval, we systematically evaluate contemporary steering methods, revealing that control often degrades at finer-grained levels. Our benchmark offers a principled and interpretable framework for safe and controllable LLM behavior, serving as a foundation for future research.
>
---
#### [new 009] From Solver to Tutor: Evaluating the Pedagogical Intelligence of LLMs with KMP-Bench
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于AI数学辅导任务，旨在评估LLMs的教育智能。提出KMP-Bench基准，从多角度测试模型的教学能力，解决现有评估方法不足的问题。**

- **链接: [https://arxiv.org/pdf/2603.02775](https://arxiv.org/pdf/2603.02775)**

> **作者:** Weikang Shi; Houxing Ren; Junting Pan; Aojun Zhou; Ke Wang; Zimu Lu; Yunqiao Yang; Yuxuan Hu; Linda Wei; Mingjie Zhan; Hongsheng Li
>
> **摘要:** Large Language Models (LLMs) show significant potential in AI mathematical tutoring, yet current evaluations often rely on simplistic metrics or narrow pedagogical scenarios, failing to assess comprehensive, multi-turn teaching effectiveness. In this paper, we introduce KMP-Bench, a comprehensive K-8 Mathematical Pedagogical Benchmark designed to assess LLMs from two complementary perspectives. The first module, KMP-Dialogue, evaluates holistic pedagogical capabilities against six core principles (e.g., Challenge, Explanation, Feedback), leveraging a novel multi-turn dialogue dataset constructed by weaving together diverse pedagogical components. The second module, KMP-Skills, provides a granular assessment of foundational tutoring abilities, including multi-turn problem-solving, error detection and correction, and problem generation. Our evaluations on KMP-Bench reveal a key disparity: while leading LLMs excel at tasks with verifiable solutions, they struggle with the nuanced application of pedagogical principles. Additionally, we present KMP-Pile, a large-scale (150K) dialogue dataset. Models fine-tuned on KMP-Pile show substantial improvement on KMP-Bench, underscoring the value of pedagogically-rich training data for developing more effective AI math tutors.
>
---
#### [new 010] ACE-Merging: Data-Free Model Merging with Adaptive Covariance Estimation
- **分类: cs.CL**

- **简介: 该论文属于模型融合任务，旨在解决多任务专家模型合并时的干扰问题。通过自适应协方差估计，提出ACE-Merging方法，在无需数据的情况下有效提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.02945](https://arxiv.org/pdf/2603.02945)**

> **作者:** Bo Xu; Haotian Wu; Hehai Lin; Weiquan Huang; Beier Zhu; Yao Shu; Chengwei Qin
>
> **备注:** Accepted to CVPR 2026 (Main Track)
>
> **摘要:** Model merging aims to combine multiple task-specific expert models into a single model while preserving generalization across diverse tasks. However, interference among experts, especially when they are trained on different objectives, often leads to significant performance degradation. Despite recent progress, resolving this interference without data access, retraining, or architectural modification remains a fundamental challenge. This paper provides a theoretical analysis demonstrating that the input covariance of each task, which is a key factor for optimal merging, can be implicitly estimated from the parameter differences of its fine-tuned model, even in a fully data-free setting. Building on this insight, we introduce \acem, an Adaptive Covariance Estimation framework that effectively mitigates inter-task interference. Our approach features a principled, closed-form solution that contrasts with prior iterative or heuristic methods. Extensive experiments on both vision and language benchmarks demonstrate that \acem sets a new state-of-the-art among data-free methods. It consistently outperforms existing baselines; for example, \acem achieves an average absolute improvement of 4\% over the previous methods across seven tasks on GPT-2. Owing to its efficient closed-form formulation, \acem delivers superior performance with a modest computational cost, providing a practical and theoretically grounded solution for model merging.
>
---
#### [new 011] Think, But Don't Overthink: Reproducing Recursive Language Models
- **分类: cs.CL**

- **简介: 该论文属于语言模型优化任务，旨在解决长文本处理问题。通过调整递归深度，研究发现过深递归会导致模型“过度思考”，影响性能。**

- **链接: [https://arxiv.org/pdf/2603.02615](https://arxiv.org/pdf/2603.02615)**

> **作者:** Daren Wang
>
> **摘要:** This project reproduces and extends the recently proposed ``Recursive Language Models'' (RLMs) framework by Zhang et al. (2026). This framework enables Large Language Models (LLMs) to process near-infinite contexts by offloading the prompt into an external REPL environment. While the original paper relies on a default recursion depth of 1 and suggests deeper recursion as a future direction, this study specifically investigates the impact of scaling the recursion depth. Using state-of-the-art open-source agentic models (DeepSeek v3.2 and Kimi K2), I evaluated pure LLM, RLM (depth=1), and RLM (depth=2) on the S-NIAH and OOLONG benchmarks. The findings reveal a compelling phenomenon: Deeper recursion causes models to ``overthink''. While depth-1 RLMs effectively boost accuracy on complex reasoning tasks, applying deeper recursion (depth=2) or using RLMs on simple retrieval tasks paradoxically degrades performance and exponentially inflates execution time (e.g., from 3.6s to 344.5s) and token costs. Code and data are available at: this https URL
>
---
#### [new 012] LaTeX Compilation: Challenges in the Era of LLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，探讨LLMs在科学写作中的挑战，分析TeX的不足，并提出Mogan STEM作为替代方案，提升编译效率与工具生态。**

- **链接: [https://arxiv.org/pdf/2603.02873](https://arxiv.org/pdf/2603.02873)**

> **作者:** Tianyou Liu; Ziqiang Li; Yansong Li; Xurui Liu
>
> **备注:** 25 pages, 12 figures
>
> **摘要:** As large language models (LLMs) increasingly assist scientific writing, limitations and the significant token cost of TeX become more and more visible. This paper analyzes TeX's fundamental defects in compilation and user experience design to illustrate its limitations on compilation efficiency, generated semantics, error localization, and tool ecosystem in the era of LLMs. As an alternative, Mogan STEM, a WYSIWYG structured editor, is introduced. Mogan outperforms TeX in the above aspects by its efficient data structure, fast rendering, and on-demand plugin loading. Extensive experiments are conducted to verify the benefits on compilation/rendering time and performance in LLM tasks. What's more, we show that due to Mogan's lower information entropy, it is more efficient to use .tmu (the document format of Mogan) to fine-tune LLMs than TeX. Therefore, we launch an appeal for larger experiments on LLM training using the .tmu format.
>
---
#### [new 013] OCR or Not? Rethinking Document Information Extraction in the MLLMs Era with Real-World Large-Scale Datasets
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文档信息提取任务，探讨MLLM在无需OCR情况下的表现。研究比较了MLLM-only与传统OCR+MLLM方法，发现强大MLLM可替代OCR，提升性能。**

- **链接: [https://arxiv.org/pdf/2603.02789](https://arxiv.org/pdf/2603.02789)**

> **作者:** Jiyuan Shen; Peiyue Yuan; Atin Ghosh; Yifan Mai; Daniel Dahlmeier
>
> **摘要:** Multimodal Large Language Models (MLLMs) enhance the potential of natural language processing. However, their actual impact on document information extraction remains unclear. In particular, it is unclear whether an MLLM-only pipeline--while simpler--can truly match the performance of traditional OCR+MLLM setups. In this paper, we conduct a large-scale benchmarking study that evaluates various out-of-the-box MLLMs on business-document information extraction. To examine and explore failure modes, we propose an automated hierarchical error analysis framework that leverages large language models (LLMs) to diagnose error patterns systematically. Our findings suggest that OCR may not be necessary for powerful MLLMs, as image-only input can achieve comparable performance to OCR-enhanced approaches. Moreover, we demonstrate that carefully designed schema, exemplars, and instructions can further enhance MLLMs performance. We hope this work can offer practical guidance and valuable insight for advancing document information extraction.
>
---
#### [new 014] APRES: An Agentic Paper Revision and Evaluation System
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出APRES系统，利用大语言模型自动优化论文，提升其影响力。旨在解决同行评审反馈不一致的问题，通过客观评估和修改论文，增强其质量和被引用可能性。**

- **链接: [https://arxiv.org/pdf/2603.03142](https://arxiv.org/pdf/2603.03142)**

> **作者:** Bingchen Zhao; Jenny Zhang; Chenxi Whitehouse; Minqi Jiang; Michael Shvartsman; Abhishek Charnalia; Despoina Magka; Tatiana Shavrina; Derek Dunfield; Oisin Mac Aodha; Yoram Bachrach
>
> **摘要:** Scientific discoveries must be communicated clearly to realize their full potential. Without effective communication, even the most groundbreaking findings risk being overlooked or misunderstood. The primary way scientists communicate their work and receive feedback from the community is through peer review. However, the current system often provides inconsistent feedback between reviewers, ultimately hindering the improvement of a manuscript and limiting its potential impact. In this paper, we introduce a novel method APRES powered by Large Language Models (LLMs) to update a scientific papers text based on an evaluation rubric. Our automated method discovers a rubric that is highly predictive of future citation counts, and integrate it with APRES in an automated system that revises papers to enhance their quality and impact. Crucially, this objective should be met without altering the core scientific content. We demonstrate the success of APRES, which improves future citation prediction by 19.6% in mean averaged error over the next best baseline, and show that our paper revision process yields papers that are preferred over the originals by human expert evaluators 79% of the time. Our findings provide strong empirical support for using LLMs as a tool to help authors stress-test their manuscripts before submission. Ultimately, our work seeks to augment, not replace, the essential role of human expert reviewers, for it should be humans who discern which discoveries truly matter, guiding science toward advancing knowledge and enriching lives.
>
---
#### [new 015] Detecting AI-Generated Essays in Writing Assessment: Responsible Use and Generalizability Across LLMs
- **分类: cs.CL**

- **简介: 该论文属于AI生成文本检测任务，旨在解决AI生成作文的识别问题。研究分析了检测器在不同LLM间的泛化能力，为实际应用提供指导。**

- **链接: [https://arxiv.org/pdf/2603.02353](https://arxiv.org/pdf/2603.02353)**

> **作者:** Jiangang Hao
>
> **备注:** 21 pages, 2 figures
>
> **摘要:** Writing is a foundational literacy skill that underpins effective communication, fosters critical thinking, facilitates learning across disciplines, and enables individuals to organize and articulate complex ideas. Consequently, writing assessment plays a vital role in evaluating language proficiency, communicative effectiveness, and analytical reasoning. The rapid advancement of large language models (LLMs) has made it increasingly easy to generate coherent, high-quality essays, raising significant concerns about the authenticity of student-submitted work. This chapter first provides an overview of the current landscape of detectors for AI-generated and AI-assisted essays, along with guidelines for their responsible use. It then presents empirical analyses to evaluate how well detectors trained on essays from one LLM generalize to identifying essays produced by other LLMs, based on essays generated in response to public GRE writing prompts. These findings provide guidance for developing and retraining detectors for practical applications.
>
---
#### [new 016] Faster, Cheaper, More Accurate: Specialised Knowledge Tracing Models Outperform LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于教育预测任务，旨在比较知识追踪模型与大语言模型在学生答题预测中的表现。研究显示知识追踪模型在准确性和效率上优于大语言模型。**

- **链接: [https://arxiv.org/pdf/2603.02830](https://arxiv.org/pdf/2603.02830)**

> **作者:** Prarthana Bhattacharyya; Joshua Mitton; Ralph Abboud; Simon Woodhead
>
> **备注:** 7 pages, 6 figures. Prarthana Bhattacharyya and Joshua Mitton contributed equally to this work
>
> **摘要:** Predicting future student responses to questions is particularly valuable for educational learning platforms where it enables effective interventions. One of the key approaches to do this has been through the use of knowledge tracing (KT) models. These are small, domain-specific, temporal models trained on student question-response data. KT models are optimised for high accuracy on specific educational domains and have fast inference and scalable deployments. The rise of Large Language Models (LLMs) motivates us to ask the following questions: (1) How well can LLMs perform at predicting students' future responses to questions? (2) Are LLMs scalable for this domain? (3) How do LLMs compare to KT models on this domain-specific task? In this paper, we compare multiple LLMs and KT models across predictive performance, deployment cost, and inference speed to answer the above questions. We show that KT models outperform LLMs with respect to accuracy and F1 scores on this domain-specific task. Further, we demonstrate that LLMs are orders of magnitude slower than KT models and cost orders of magnitude more to deploy. This highlights the importance of domain-specific models for education prediction tasks and the fact that current closed source LLMs should not be used as a universal solution for all tasks.
>
---
#### [new 017] Graph-GRPO: Stabilizing Multi-Agent Topology Learning via Group Relative Policy Optimization
- **分类: cs.CL**

- **简介: 该论文属于多智能体系统任务，旨在解决通信拓扑优化问题。针对现有方法梯度方差大、信用分配困难的问题，提出Graph-GRPO框架，通过群体相对策略优化提升训练稳定性与性能。**

- **链接: [https://arxiv.org/pdf/2603.02701](https://arxiv.org/pdf/2603.02701)**

> **作者:** Yueyang Cang; Xiaoteng Zhang; Erlu Zhao; Zehua Ji; Yuhang Liu; Yuchen He; Zhiyuan Ning; Chen Yijun; Wenge Que; Li Shi
>
> **摘要:** Optimizing communication topology is fundamental to the efficiency and effectiveness of Large Language Model (LLM)-based Multi-Agent Systems (MAS). While recent approaches utilize reinforcement learning to dynamically construct task-specific graphs, they typically rely on single-sample policy gradients with absolute rewards (e.g., binary correctness). This paradigm suffers from severe gradient variance and the credit assignment problem: simple queries yield non-informative positive rewards for suboptimal structures, while difficult queries often result in failures that provide no learning signal. To address these challenges, we propose Graph-GRPO, a novel topology optimization framework that integrates Group Relative Policy Optimization. Instead of evaluating a single topology in isolation, Graph-GRPO samples a group of diverse communication graphs for each query and computes the advantage of specific edges based on their relative performance within the group. By normalizing rewards across the sampled group, our method effectively mitigates the noise derived from task difficulty variance and enables fine-grained credit assignment. Extensive experiments on reasoning and code generation benchmarks demonstrate that Graph-GRPO significantly outperforms state-of-the-art baselines, achieving superior training stability and identifying critical communication pathways previously obscured by reward noise.
>
---
#### [new 018] GLoRIA: Gated Low-Rank Interpretable Adaptation for Dialectal ASR
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语音识别任务，解决方言ASR中数据不足与区域差异问题。提出GLoRIA框架，通过低秩更新和元数据控制实现高效、可解释的模型适配。**

- **链接: [https://arxiv.org/pdf/2603.02464](https://arxiv.org/pdf/2603.02464)**

> **作者:** Pouya Mehralian; Melissa Farasyn; Anne Breitbarth; Anne-Sophie Ghyselen; Hugo Van hamme
>
> **备注:** Accepted to ICASSP 2026. 5 pages
>
> **摘要:** Automatic Speech Recognition (ASR) in dialect-heavy settings remains challenging due to strong regional variation and limited labeled data. We propose GLoRIA, a parameter-efficient adaptation framework that leverages metadata (e.g., coordinates) to modulate low-rank updates in a pre-trained encoder. GLoRIA injects low-rank matrices into each feed-forward layer, with a gating MLP determining the non-negative contribution of each LoRA rank-1 component based on location metadata. On the GCND corpus, GLoRIA outperforms geo-conditioned full fine-tuning, LoRA, and both dialect-specific and unified full fine-tuning, achieving state-of-the-art word error rates while updating under 10% of parameters. GLoRIA also generalizes well to unseen dialects, including in extrapolation scenarios, and enables interpretable adaptation patterns that can be visualized geospatially. These results show metadata-gated low-rank adaptation is an effective, interpretable, and efficient solution for dialectal ASR.
>
---
#### [new 019] Efficient Self-Evaluation for Diffusion Language Models via Sequence Regeneration
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型评估任务，旨在解决扩散语言模型的自评估难题。提出DiSE方法，通过序列重生成评估置信度，提升质量评估效率与可靠性。**

- **链接: [https://arxiv.org/pdf/2603.02760](https://arxiv.org/pdf/2603.02760)**

> **作者:** Linhao Zhong; Linyu Wu; Wen Wang; Yuling Xi; Chenchen Jing; Jiaheng Zhang; Hao Chen; Chunhua Shen
>
> **摘要:** Diffusion large language models (dLLMs) have recently attracted significant attention for their ability to enhance diversity, controllability, and parallelism. However, their non-sequential, bidirectionally masked generation makes quality assessment difficult, underscoring the need for effective self-evaluation. In this work, we propose DiSE, a simple yet effective self-evaluation confidence quantification method for dLLMs. DiSE quantifies confidence by computing the probability of regenerating the tokens in the entire generated sequence, given the full context. This method enables more efficient and reliable quality assessment by leveraging token regeneration probabilities, facilitating both likelihood estimation and robust uncertainty quantification. Building upon DiSE, we further introduce a flexible-length generation framework, which adaptively controls the sequence length based on the model's self-assessment of its own output. We analyze and validate the feasibility of DiSE from the perspective of dLLM generalization, and empirically demonstrate that DiSE is positively correlated with both semantic coherence and answer accuracy. Extensive experiments on likelihood evaluation, uncertainty quantification, and flexible-length generation further confirm the effectiveness of the proposed DiSE.
>
---
#### [new 020] A Zipf-preserving, long-range correlated surrogate for written language and other symbolic sequences
- **分类: cs.CL; cond-mat.stat-mech; q-bio.GN**

- **简介: 该论文属于符号序列建模任务，旨在同时保留频率分布与长程相关性。提出一种新方法，通过映射分数高斯噪声生成符合原序列统计特性的替代数据。**

- **链接: [https://arxiv.org/pdf/2603.02213](https://arxiv.org/pdf/2603.02213)**

> **作者:** Marcelo A. Montemurro; Mirko Degli Esposti
>
> **摘要:** Symbolic sequences such as written language and genomic DNA display characteristic frequency distributions and long-range correlations extending over many symbols. In language, this takes the form of Zipf's law for word frequencies together with persistent correlations spanning hundreds or thousands of tokens, while in DNA it is reflected in nucleotide composition and long-memory walks under purine-pyrimidine mappings. Existing surrogate models usually preserve either the frequency distribution or the correlation properties, but not both simultaneously. We introduce a surrogate model that retains both constraints: it preserves the empirical symbol frequencies of the original sequence and reproduces its long-range correlation structure, quantified by the detrended fluctuation analysis (DFA) exponent. Our method generates surrogates of symbolic sequences by mapping fractional Gaussian noise (FGN) onto the empirical histogram through a frequency-preserving assignment. The resulting surrogates match the original in first-order statistics and long-range scaling while randomising short-range dependencies. We validate the model on representative texts in English and Latin, and illustrate its broader applicability with genomic DNA, showing that base composition and DFA scaling are reproduced. This approach provides a principled tool for disentangling structural features of symbolic systems and for testing hypotheses on the origin of scaling laws and memory effects across language, DNA, and other symbolic domains.
>
---
#### [new 021] Nodes Are Early, Edges Are Late: Probing Diagram Representations in Large Vision-Language Models
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于视觉-语言理解任务，旨在解决LVLMs在关系理解上的不足。通过构建合成图数据集，发现节点信息早于边信息被线性编码，解释了模型在处理边关系时的困难。**

- **链接: [https://arxiv.org/pdf/2603.02865](https://arxiv.org/pdf/2603.02865)**

> **作者:** Haruto Yoshida; Keito Kudo; Yoichi Aoki; Ryota Tanaka; Itsumi Saito; Keisuke Sakaguchi; Kentaro Inui
>
> **摘要:** Large vision-language models (LVLMs) demonstrate strong performance on diagram understanding benchmarks, yet they still struggle with understanding relationships between elements, particularly those represented by nodes and directed edges (e.g., arrows and lines). To investigate the underlying causes of this limitation, we probe the internal representation of LVLMs using a carefully constructed synthetic diagram dataset based on directed graphs. Our probing experiments reveal that edge information is not linearly separable in the vision encoder and becomes linearly encoded only in the text tokens in the language model. In contrast, node information and global structural features are already linearly encoded in individual hidden states of the vision encoder. These findings suggest that the stage at which linearly separable representations are formed varies depending on the type of visual information. In particular, the delayed emergence of edge representations may help explain why LVLMs struggle with relational understanding, such as interpreting edge directions, which require more abstract, compositionally integrated processes.
>
---
#### [new 022] RO-N3WS: Enhancing Generalization in Low-Resource ASR with Diverse Romanian Speech Benchmarks
- **分类: cs.CL; cs.LG; cs.SD**

- **简介: 论文提出RO-N3WS基准数据集，用于提升低资源场景下的自动语音识别（ASR）泛化能力。针对ASR任务中的低资源和分布外问题，通过多样化罗马尼亚语语音数据进行模型训练与优化。**

- **链接: [https://arxiv.org/pdf/2603.02368](https://arxiv.org/pdf/2603.02368)**

> **作者:** Alexandra Diaconu; Mădălina Vînaga; Bogdan Alexe
>
> **摘要:** We introduce RO-N3WS, a benchmark Romanian speech dataset designed to improve generalization in automatic speech recognition (ASR), particularly in low-resource and out-of-distribution (OOD) conditions. RO-N3WS comprises over 126 hours of transcribed audio collected from broadcast news, literary audiobooks, film dialogue, children's stories, and conversational podcast speech. This diversity enables robust training and fine-tuning across stylistically distinct domains. We evaluate several state-of-the-art ASR systems (Whisper, Wav2Vec 2.0) in both zero-shot and fine-tuned settings, and conduct controlled comparisons using synthetic data generated with expressive TTS models. Our results show that even limited fine-tuning on real speech from RO-N3WS yields substantial WER improvements over zero-shot baselines. We will release all models, scripts, and data splits to support reproducible research in multilingual ASR, domain adaptation, and lightweight deployment.
>
---
#### [new 023] TrustMH-Bench: A Comprehensive Benchmark for Evaluating the Trustworthiness of Large Language Models in Mental Health
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于评估任务，旨在解决大语言模型在心理健康领域信任度不足的问题。工作包括构建TrustMH-Bench框架，从八个维度评估模型信任度，并验证现有模型的缺陷。**

- **链接: [https://arxiv.org/pdf/2603.03047](https://arxiv.org/pdf/2603.03047)**

> **作者:** Zixin Xiong; Ziteng Wang; Haotian Fan; Xinjie Zhang; Wenxuan Wang
>
> **摘要:** While Large Language Models (LLMs) demonstrate significant potential in providing accessible mental health support, their practical deployment raises critical trustworthiness concerns due to the domains high-stakes and safety-sensitive nature. Existing evaluation paradigms for general-purpose LLMs fail to capture mental health-specific requirements, highlighting an urgent need to prioritize and enhance their trustworthiness. To address this, we propose TrustMH-Bench, a holistic framework designed to systematically quantify the trustworthiness of mental health LLMs. By establishing a deep mapping from domain-specific norms to quantitative evaluation metrics, TrustMH-Bench evaluates models across eight core pillars: Reliability, Crisis Identification and Escalation, Safety, Fairness, Privacy, Robustness, Anti-sycophancy, and Ethics. We conduct extensive experiments across six general-purpose LLMs and six specialized mental health models. Experimental results indicate that the evaluated models underperform across various trustworthiness dimensions in mental health scenarios, revealing significant deficiencies. Notably, even generally powerful models (e.g., GPT-5.1) fail to maintain consistently high performance across all dimensions. Consequently, systematically improving the trustworthiness of LLMs has become a critical task. Our data and code are released.
>
---
#### [new 024] HateMirage: An Explainable Multi-Dimensional Dataset for Decoding Faux Hate and Subtle Online Abuse
- **分类: cs.CL; cs.SI**

- **简介: 该论文提出HateMirage数据集，用于检测和解释隐性仇恨言论与虚假叙事中的有害意图，解决在线安全中隐蔽仇恨识别的问题。**

- **链接: [https://arxiv.org/pdf/2603.02684](https://arxiv.org/pdf/2603.02684)**

> **作者:** Sai Kartheek Reddy Kasu; Shankar Biradar; Sunil Saumya; Md. Shad Akhtar
>
> **备注:** Accepted at LREC 2026
>
> **摘要:** Subtle and indirect hate speech remains an underexplored challenge in online safety research, particularly when harmful intent is embedded within misleading or manipulative narratives. Existing hate speech datasets primarily capture overt toxicity, underrepresenting the nuanced ways misinformation can incite or normalize hate. To address this gap, we present HateMirage, a novel dataset of Faux Hate comments designed to advance reasoning and explainability research on hate emerging from fake or distorted narratives. The dataset was constructed by identifying widely debunked misinformation claims from fact-checking sources and tracing related YouTube discussions, resulting in 4,530 user comments. Each comment is annotated along three interpretable dimensions: Target (who is affected), Intent (the underlying motivation or goal behind the comment), and Implication (its potential social impact). Unlike prior explainability datasets such as HateXplain and HARE, which offer token-level or single-dimensional reasoning, HateMirage introduces a multi-dimensional explanation framework that captures the interplay between misinformation, harm, and social consequence. We benchmark multiple open-source language models on HateMirage using ROUGE-L F1 and Sentence-BERT similarity to assess explanation coherence. Results suggest that explanation quality may depend more on pretraining diversity and reasoning-oriented data rather than on model scale alone. By coupling misinformation reasoning with harm attribution, HateMirage establishes a new benchmark for interpretable hate detection and responsible AI research.
>
---
#### [new 025] The Distribution of Phoneme Frequencies across the World's Languages: Macroscopic and Microscopic Information-Theoretic Models
- **分类: cs.CL**

- **简介: 该论文属于语言学与信息论交叉研究，旨在解释语音频率分布。通过宏观和微观模型，揭示语音分布的规律，解决语音结构的统计特征问题。**

- **链接: [https://arxiv.org/pdf/2603.02860](https://arxiv.org/pdf/2603.02860)**

> **作者:** Fermín Moscoso del Prado Martín; Suchir Salhan
>
> **摘要:** We demonstrate that the frequency distribution of phonemes across languages can be explained at both macroscopic and microscopic levels. Macroscopically, phoneme rank-frequency distributions closely follow the order statistics of a symmetric Dirichlet distribution whose single concentration parameter scales systematically with phonemic inventory size, revealing a robust compensation effect whereby larger inventories exhibit lower relative entropy. Microscopically, a Maximum Entropy model incorporating constraints from articulatory, phonotactic, and lexical structure accurately predicts language-specific phoneme probabilities. Together, these findings provide a unified information-theoretic account of phoneme frequency structure.
>
---
#### [new 026] Evaluating Performance Drift from Model Switching in Multi-Turn LLM Systems
- **分类: cs.CL**

- **简介: 该论文研究多轮大语言模型系统中模型切换导致的性能漂移问题，通过构建基准测试评估切换影响，提出分解方法以监测风险。任务属于模型系统可靠性分析。**

- **链接: [https://arxiv.org/pdf/2603.03111](https://arxiv.org/pdf/2603.03111)**

> **作者:** Raad Khraishi; Iman Zafar; Katie Myles; Greig A Cowan
>
> **摘要:** Deployed multi-turn LLM systems routinely switch models mid-interaction due to upgrades, cross-provider routing, and fallbacks. Such handoffs create a context mismatch: the model generating later turns must condition on a dialogue prefix authored by a different model, potentially inducing silent performance drift. We introduce a switch-matrix benchmark that measures this effect by running a prefix model for early turns and a suffix model for the final turn, and comparing against the no-switch baseline using paired episode-level bootstrap confidence intervals. Across CoQA conversational QA and Multi-IF benchmarks, even a single-turn handoff yields prevalent and statistically significant, directional effects and may swing outcomes by -8 to +13 percentage points in Multi-IF strict success rate and +/- 4 absolute F1 on CoQA, comparable to the no-switch gap between common model tiers (e.g., GPT-5-nano vs GPT-5-mini). We further find systematic compatibility patterns: some suffix models degrade under nearly any non-self dialogue history, while others improve under nearly any foreign prefix. To enable compressed handoff risk monitoring, we decompose switch-induced drift into per-model prefix influence and suffix susceptibility terms, accounting for ~70% of variance across benchmarks. These results position handoff robustness as an operational reliability dimension that single-model benchmarks miss, motivating explicit monitoring and handoff-aware mitigation in multi-turn systems.
>
---
#### [new 027] GPUTOK: GPU Accelerated Byte Level BPE Tokenization
- **分类: cs.CL; cs.AI; cs.DC; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决长文本tokenization效率低的问题。通过GPU加速实现字节级BPE分词，提升处理速度并保持输出质量。**

- **链接: [https://arxiv.org/pdf/2603.02597](https://arxiv.org/pdf/2603.02597)**

> **作者:** Venu Gopal Kadamba; Kanishkha Jaisankar
>
> **摘要:** As large language models move toward million-token context windows, CPU tokenizers become a major slowdown because they process text one step at a time while powerful GPUs sit unused. We built a GPU-based byte-level BPE tokenizer that follows GPT-2's merge rules. It includes a basic BlockBPE-style kernel and a faster, optimized version that uses cuCollections static map, CUB reductions, and a pybind11 interface for Python. On WikiText103 sequences up to 131k tokens, the optimized GPU tokenizer produces the same tokens as a CPU version and, for the longest inputs, is about 1.7x faster than tiktoken and about 7.6x faster than the HuggingFace GPT-2 tokenizer. Nsight profiling shows that 70-80% of CUDA API time goes to memory allocation, so adding memory pooling should give the biggest speed boost next. Tests on generation tasks using WikiText103 prompts show that our GPU tokenizer's outputs stay within about one percentage point of tiktoken and HuggingFace GPT-2 on similarity and overlap metrics, meaning it keeps output quality while making long-context inference more practical.
>
---
#### [new 028] Learning When to Act or Refuse: Guarding Agentic Reasoning Models for Safe Multi-Step Tool Use
- **分类: cs.CL**

- **简介: 该论文属于安全增强任务，旨在解决代理模型在多步骤工具使用中的安全问题。通过MOSAIC框架，明确安全决策并提升拒绝有害行为的能力。**

- **链接: [https://arxiv.org/pdf/2603.03205](https://arxiv.org/pdf/2603.03205)**

> **作者:** Aradhye Agarwal; Gurdit Siyan; Yash Pandya; Joykirat Singh; Akshay Nambi; Ahmed Awadallah
>
> **备注:** 24 pages, 5 figures
>
> **摘要:** Agentic language models operate in a fundamentally different safety regime than chat models: they must plan, call tools, and execute long-horizon actions where a single misstep, such as accessing files or entering credentials, can cause irreversible harm. Existing alignment methods, largely optimized for static generation and task completion, break down in these settings due to sequential decision-making, adversarial tool feedback, and overconfident intermediate reasoning. We introduce MOSAIC, a post-training framework that aligns agents for safe multi-step tool use by making safety decisions explicit and learnable. MOSAIC structures inference as a plan, check, then act or refuse loop, with explicit safety reasoning and refusal as first-class actions. To train without trajectory-level labels, we use preference-based reinforcement learning with pairwise trajectory comparisons, which captures safety distinctions often missed by scalar rewards. We evaluate MOSAIC zero-shot across three model families, Qwen2.5-7B, Qwen3-4B-Thinking, and Phi-4, and across out-of-distribution benchmarks spanning harmful tasks, prompt injection, benign tool use, and cross-domain privacy leakage. MOSAIC reduces harmful behavior by up to 50%, increases harmful-task refusal by over 20% on injection attacks, cuts privacy leakage, and preserves or improves benign task performance, demonstrating robust generalization across models, domains, and agentic settings.
>
---
#### [new 029] CoDAR: Continuous Diffusion Language Models are More Powerful Than You Think
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究语言模型生成任务，解决连续扩散模型性能不足的问题。通过分析发现token rounding是瓶颈，提出CoDAR框架提升生成质量。**

- **链接: [https://arxiv.org/pdf/2603.02547](https://arxiv.org/pdf/2603.02547)**

> **作者:** Junzhe Shen; Jieru Zhao; Ziwei He; Zhouhan Lin
>
> **摘要:** We study why continuous diffusion language models (DLMs) have lagged behind discrete diffusion approaches despite their appealing continuous generative dynamics. Under a controlled token--recovery study, we identify token rounding, the final projection from denoised embeddings to tokens, as a primary bottleneck. Building on these insights, we propose CoDAR (Continuous Diffusion with Contextual AutoRegressive Decoder), a two--stage framework that keeps diffusion entirely continuous in an embedding space while learning a strong, context--conditional discretizer: an autoregressive Transformer decoder that cross--attends to the denoised embedding sequence and performs contextualized rounding to tokens. Experiments on LM1B and OpenWebText demonstrate that CoDAR substantially improves generation quality over latent diffusion and becomes competitive with strong discrete DLMs, while exposing a simple decoder--temperature knob to navigate the fluency--diversity trade off.
>
---
#### [new 030] UniSkill: A Dataset for Matching University Curricula to Professional Competencies
- **分类: cs.CL**

- **简介: 该论文属于课程与技能匹配任务，旨在解决课程与职业能力对齐数据不足的问题。通过构建并发布标注数据集，训练模型实现课程与技能的匹配。**

- **链接: [https://arxiv.org/pdf/2603.03134](https://arxiv.org/pdf/2603.03134)**

> **作者:** Nurlan Musazade; Joszef Mezei; Mike Zhang
>
> **备注:** LREC 2026
>
> **摘要:** Skill extraction and recommendation systems have been studied from recruiter, applicant, and education perspectives. While AI applications in job advertisements have received broad attention, deficiencies in the instructed skills side remain a challenge. In this work, we address the scarcity of publicly available datasets by releasing both manually annotated and synthetic datasets of skills from the European Skills, Competences, Qualifications and Occupations (ESCO) taxonomy and university course pairs and publishing corresponding annotation guidelines. Specifically, we match graduate-level university courses with skills from the Systems Analysts and Management and Organization Analyst ESCO occupation groups at two granularities: course title with a skill, and course sentence with a skill. We train language models on this dataset to serve as a baseline for retrieval and recommendation systems for course-to-skill and skill-to-course matching. We evaluate the models on a portion of the annotated data. Our BERT model achieves 87% F1-score, showing that course and skill matching is a feasible task.
>
---
#### [new 031] ExpGuard: LLM Content Moderation in Specialized Domains
- **分类: cs.CL**

- **简介: 该论文属于内容安全任务，旨在解决LLM在专业领域中对有害内容的识别与过滤问题。提出ExpGuard模型及ExpGuardMix数据集，提升模型在金融、医疗等领域的安全性。**

- **链接: [https://arxiv.org/pdf/2603.02588](https://arxiv.org/pdf/2603.02588)**

> **作者:** Minseok Choi; Dongjin Kim; Seungbin Yang; Subin Kim; Youngjun Kwak; Juyoung Oh; Jaegul Choo; Jungmin Son
>
> **备注:** ICLR 2026
>
> **摘要:** With the growing deployment of large language models (LLMs) in real-world applications, establishing robust safety guardrails to moderate their inputs and outputs has become essential to ensure adherence to safety policies. Current guardrail models predominantly address general human-LLM interactions, rendering LLMs vulnerable to harmful and adversarial content within domain-specific contexts, particularly those rich in technical jargon and specialized concepts. To address this limitation, we introduce ExpGuard, a robust and specialized guardrail model designed to protect against harmful prompts and responses across financial, medical, and legal domains. In addition, we present ExpGuardMix, a meticulously curated dataset comprising 58,928 labeled prompts paired with corresponding refusal and compliant responses, from these specific sectors. This dataset is divided into two subsets: ExpGuardTrain, for model training, and ExpGuardTest, a high-quality test set annotated by domain experts to evaluate model robustness against technical and domain-specific content. Comprehensive evaluations conducted on ExpGuardTest and eight established public benchmarks reveal that ExpGuard delivers competitive performance across the board while demonstrating exceptional resilience to domain-specific adversarial attacks, surpassing state-of-the-art models such as WildGuard by up to 8.9% in prompt classification and 15.3% in response classification. To encourage further research and development, we open-source our code, data, and model, enabling adaptation to additional domains and supporting the creation of increasingly robust guardrail models.
>
---
#### [new 032] Cross-Family Speculative Prefill: Training-Free Long-Context Compression with Small Draft Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决长上下文推理中的提示压缩问题。通过跨家族模型的推测预填充，实现无需训练的高效提示压缩。**

- **链接: [https://arxiv.org/pdf/2603.02631](https://arxiv.org/pdf/2603.02631)**

> **作者:** Shubhangi Upasani; Ravi Shanker Raju; Bo Li; Mengmeing Ji; John Long; Chen Wu; Urmish Thakker; Guangtao Wang
>
> **摘要:** Prompt length is a major bottleneck in agentic large language model (LLM) workloads, where repeated inference steps and multi-call loops incur substantial prefill cost. Recent work on speculative prefill demonstrates that attention-based token importance estimation can enable training-free prompt compression, but this assumes the existence of a draft model that shares the same tokenizer as the target model. In practice, however, agentic pipelines frequently employ models without any smaller in-family draft model. In this work, we study cross-family speculative prefill, where a lightweight draft model from one model family is used to perform prompt compression for a target model from a different family. Using the same speculative prefill mechanism as prior work, we evaluate a range of cross-family draft-target combinations, including Qwen, LLaMA, and DeepSeek models. Across a broad diversity of tasks, we find that attention-based token importance estimation transfers reliably across different model families despite differences in model architectures and tokenizers between draft and target models. Cross-model prompt compression largely retains 90~100% of full-prompt baseline performance and, in some cases, slightly improves accuracy due to denoising effects, while delivering substantial reductions in time to first token (TTFT). These results suggest that speculative prefill depends mainly on task priors and semantic structure, thus serving as a generalizable prompt compression primitive. We discuss the implications of our findings for agentic systems, where repeated long-context inference and heterogeneous model stacks make cross-model prompt compression both necessary and practical.
>
---
#### [new 033] Eval4Sim: An Evaluation Framework for Persona Simulation
- **分类: cs.CL**

- **简介: 该论文属于对话系统评估任务，旨在解决LLM personas模拟人类对话行为的评价问题。提出Eval4Sim框架，从三个维度评估对话与人类行为的一致性。**

- **链接: [https://arxiv.org/pdf/2603.02876](https://arxiv.org/pdf/2603.02876)**

> **作者:** Eliseo Bao; Anxo Perez; Xi Wang; Javier Parapar
>
> **摘要:** Large Language Model (LLM) personas with explicit specifications of attributes, background, and behavioural tendencies are increasingly used to simulate human conversations for tasks such as user modeling, social reasoning, and behavioural analysis. Ensuring that persona-grounded simulations faithfully reflect human conversational behaviour is therefore critical. However, current evaluation practices largely rely on LLM-as-a-judge approaches, offering limited grounding in observable human behavior and producing opaque scalar scores. We address this gap by proposing Eval4Sim, an evaluation framework that measures how closely simulated conversations align with human conversational patterns across three complementary dimensions. Adherence captures how effectively persona backgrounds are implicitly encoded in generated utterances, assessed via dense retrieval with speaker-aware representations. Consistency evaluates whether a persona maintains a distinguishable identity across conversations, computed through authorship verification. Naturalness reflects whether conversations exhibit human-like flow rather than overly rigid or optimized structure, quantified through distributions derived from dialogue-focused Natural Language Inference. Unlike absolute or optimization-oriented metrics, Eval4Sim uses a human conversational corpus (i.e., PersonaChat) as a reference baseline and penalizes deviations in both directions, distinguishing insufficient persona encoding from over-optimized, unnatural behaviour. Although demonstrated on PersonaChat, the applicability of Eval4Sim extends to any conversational corpus containing speaker-level annotations.
>
---
#### [new 034] TAO-Attack: Toward Advanced Optimization-Based Jailbreak Attacks for Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于安全攻击任务，旨在解决LLM对优化型越狱攻击的脆弱性问题。提出TAO-Attack方法，通过双阶段损失函数和方向优先优化策略，提升攻击成功率。**

- **链接: [https://arxiv.org/pdf/2603.03081](https://arxiv.org/pdf/2603.03081)**

> **作者:** Zhi Xu; Jiaqi Li; Xiaotong Zhang; Hong Yu; Han Liu
>
> **摘要:** Large language models (LLMs) have achieved remarkable success across diverse applications but remain vulnerable to jailbreak attacks, where attackers craft prompts that bypass safety alignment and elicit unsafe responses. Among existing approaches, optimization-based attacks have shown strong effectiveness, yet current methods often suffer from frequent refusals, pseudo-harmful outputs, and inefficient token-level updates. In this work, we propose TAO-Attack, a new optimization-based jailbreak method. TAO-Attack employs a two-stage loss function: the first stage suppresses refusals to ensure the model continues harmful prefixes, while the second stage penalizes pseudo-harmful outputs and encourages the model toward more harmful completions. In addition, we design a direction-priority token optimization (DPTO) strategy that improves efficiency by aligning candidates with the gradient direction before considering update magnitude. Extensive experiments on multiple LLMs demonstrate that TAO-Attack consistently outperforms state-of-the-art methods, achieving higher attack success rates and even reaching 100\% in certain scenarios.
>
---
#### [new 035] Learning to Generate and Extract: A Multi-Agent Collaboration Framework For Zero-shot Document-level Event Arguments Extraction
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦于零样本文档级事件参数抽取任务，解决标注数据不足与生成数据可靠性问题，提出多智能体协作框架，提升数据质量和抽取性能。**

- **链接: [https://arxiv.org/pdf/2603.02909](https://arxiv.org/pdf/2603.02909)**

> **作者:** Guangjun Zhang; Hu Zhang; Yazhou Han; Yue Fan; Yuhang Shao; Ru Li; Hongye Tan
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Document-level event argument extraction (DEAE) is essential for knowledge acquisition, aiming to extract participants of events from this http URL the zero-shot setting, existing methods employ LLMs to generate synthetic data to address the challenge posed by the scarcity of annotated data. However, relying solely on Event-type-only prompts makes it difficult for the generated content to accurately capture the contextual and structural relationships of unseen events. Moreover, ensuring the reliability and usability of synthetic data remains a significant challenge due to the absence of quality evaluation mechanisms. To this end, we introduce a multi-agent collaboration framework for zero-shot document-level event argument extraction (ZS-DEAE), which simulates the human collaborative cognitive process of "Propose-Evaluate-Revise." Specifically, the framework comprises a generation agent and an evaluation agent. The generation agent synthesizes data for unseen events by leveraging knowledge from seen events, while the evaluation agent extracts arguments from the synthetic data and assesses their semantic consistency with the context. The evaluation results are subsequently converted into reward signals, with event structure constraints incorporated into the reward design to enable iterative optimization of both agents via reinforcement this http URL three zero-shot scenarios constructed from the RAMS and WikiEvents datasets, our method achieves improvements both in data generation quality and argument extraction performance, while the generated data also effectively enhances the zero-shot performance of other DEAE models.
>
---
#### [new 036] Sensory-Aware Sequential Recommendation via Review-Distilled Representations
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于推荐系统任务，旨在解决传统推荐模型缺乏感官属性信息的问题。通过提取评论中的感官属性并生成嵌入，增强序列推荐效果。**

- **链接: [https://arxiv.org/pdf/2603.02709](https://arxiv.org/pdf/2603.02709)**

> **作者:** Yeo Chan Yoon
>
> **摘要:** We propose a novel framework for sensory-aware sequential recommendation that enriches item representations with linguistically extracted sensory attributes from product reviews. Our approach, \textsc{ASEGR} (Attribute-based Sensory Enhanced Generative Recommendation), introduces a two-stage pipeline in which a large language model is first fine-tuned as a teacher to extract structured sensory attribute--value pairs, such as \textit{color: matte black} and \textit{scent: vanilla}, from unstructured review text. The extracted structures are then distilled into a compact student transformer that produces fixed-dimensional sensory embeddings for each item. These embeddings encode experiential semantics in a reusable form and are incorporated into standard sequential recommender architectures as additional item-level representations. We evaluate our method on four Amazon domains and integrate the learned sensory embeddings into representative sequential recommendation models, including SASRec, BERT4Rec, and BSARec. Across domains, sensory-enhanced models consistently outperform their identifier-based counterparts, indicating that linguistically grounded sensory representations provide complementary signals to behavioral interaction patterns. Qualitative analysis further shows that the extracted attributes align closely with human perceptions of products, enabling interpretable connections between natural language descriptions and recommendation behavior. Overall, this work demonstrates that sensory attribute distillation offers a principled and scalable way to bridge information extraction and sequential recommendation through structured semantic representation learning.
>
---
#### [new 037] MaBERT:A Padding Safe Interleaved Transformer Mamba Hybrid Encoder for Efficient Extended Context Masked Language Modeling
- **分类: cs.CL**

- **简介: 该论文提出MaBERT，结合Transformer与Mamba结构，解决长文本建模效率低的问题，通过混合设计提升训练和推理速度。**

- **链接: [https://arxiv.org/pdf/2603.03001](https://arxiv.org/pdf/2603.03001)**

> **作者:** Jinwoong Kim; Sangjin Park
>
> **备注:** 8 pages
>
> **摘要:** Self attention encoders such as Bidirectional Encoder Representations from Transformers(BERT) scale quadratically with sequence length, making long context modeling expensive. Linear time state space models, such as Mamba, are efficient; however, they show limitations in modeling global interactions and can suffer from padding induced state contamination. We propose MaBERT, a hybrid encoder that interleaves Transformer layers for global dependency modeling with Mamba layers for linear time state updates. This design alternates global contextual integration with fast state accumulation, enabling efficient training and inference on long inputs. To stabilize variable length batching, we introduce paddingsafe masking, which blocks state propagation through padded positions, and mask aware attention pooling, which aggregates information only from valid tokens. On GLUE, MaBERT achieves the best mean score on five of the eight tasks, with strong performance on the CoLA and sentence pair inference tasks. When extending the context from 512 to 4,096 tokens, MaBERT reduces training time and inference latency by 2.36x and 2.43x, respectively, relative to the average of encoder baselines, demonstrating a practical long context efficient encoder.
>
---
#### [new 038] Using Learning Progressions to Guide AI Feedback for Science Learning
- **分类: cs.CL**

- **简介: 该论文属于教育技术任务，旨在解决AI反馈质量与可扩展性问题。通过比较基于学习进度的AI反馈与专家制定的评分标准，验证其有效性。**

- **链接: [https://arxiv.org/pdf/2603.03249](https://arxiv.org/pdf/2603.03249)**

> **作者:** Xin Xia; Nejla Yuruk; Yun Wang; Xiaoming Zhai
>
> **备注:** 15pages, 4 figures
>
> **摘要:** Generative artificial intelligence (AI) offers scalable support for formative feedback, yet most AI-generated feedback relies on task-specific rubrics authored by domain experts. While effective, rubric authoring is time-consuming and limits scalability across instructional contexts. Learning progressions (LP) provide a theoretically grounded representation of students' developing understanding and may offer an alternative solution. This study examines whether an LP-driven rubric generation pipeline can produce AI-generated feedback comparable in quality to feedback guided by expert-authored task rubrics. We analyzed AI-generated feedback for written scientific explanations produced by 207 middle school students in a chemistry task. Two pipelines were compared: (a) feedback guided by a human expert-designed, task-specific rubric, and (b) feedback guided by a task-specific rubric automatically derived from a learning progression prior to grading and feedback generation. Two human coders evaluated feedback quality using a multi-dimensional rubric assessing Clarity, Accuracy, Relevance, Engagement and Motivation, and Reflectiveness (10 sub-dimensions). Inter-rater reliability was high, with percent agreement ranging from 89% to 100% and Cohen's kappa values for estimable dimensions (kappa = .66 to .88). Paired t-tests revealed no statistically significant differences between the two pipelines for Clarity (t1 = 0.00, p1 = 1.000; t2 = 0.84, p2 = .399), Relevance (t1 = 0.28, p1 = .782; t2 = -0.58, p2 = .565), Engagement and Motivation (t1 = 0.50, p1 = .618; t2 = -0.58, p2 = .565), or Reflectiveness (t = -0.45, p = .656). These findings suggest that the LP-driven rubric pipeline can serve as an alternative solution.
>
---
#### [new 039] Compact Prompting in Instruction-tuned LLMs for Joint Argumentative Component Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于Argumentative Component Detection（ACD）任务，旨在解决联合识别和分类论点成分的问题。工作采用指令调优的大型语言模型，将ACD作为生成任务处理，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2603.03095](https://arxiv.org/pdf/2603.03095)**

> **作者:** Sofiane Elguendouze; Erwan Hain; Elena Cabrio; Serena Villata
>
> **备注:** Under Review (COLM 2026)
>
> **摘要:** Argumentative component detection (ACD) is a core subtask of Argument(ation) Mining (AM) and one of its most challenging aspects, as it requires jointly delimiting argumentative spans and classifying them into components such as claims and premises. While research on this subtask remains relatively limited compared to other AM tasks, most existing approaches formulate it as a simplified sequence labeling problem, component classification, or a pipeline of component segmentation followed by classification. In this paper, we propose a novel approach based on instruction-tuned Large Language Models (LLMs) using compact instruction-based prompts, and reframe ACD as a language generation task, enabling arguments to be identified directly from plain text without relying on pre-segmented components. Experiments on standard benchmarks show that our approach achieves higher performance compared to state-of-the-art systems. To the best of our knowledge, this is one of the first attempts to fully model ACD as a generative task, highlighting the potential of instruction tuning for complex AM problems.
>
---
#### [new 040] ITLC at SemEval-2026 Task 11: Normalization and Deterministic Parsing for Formal Reasoning in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言推理任务，旨在解决大模型在推理中的内容偏差问题。通过结构抽象和确定性解析，将三段论转换为规范逻辑形式，提升推理有效性。**

- **链接: [https://arxiv.org/pdf/2603.02676](https://arxiv.org/pdf/2603.02676)**

> **作者:** Wicaksono Leksono Muhamad; Joanito Agili Lopo; Tack Hwa Wong; Muhammad Ravi Shulthan Habibi; Samuel Cahyawijaya
>
> **摘要:** Large language models suffer from content effects in reasoning tasks, particularly in multi-lingual contexts. We introduce a novel method that reduces these biases through explicit structural abstraction that transforms syllogisms into canonical logical representations and applies deterministic parsing to determine validity. Evaluated on the SemEval-2026 Task 11 multilingual benchmark, our approach achieves top-5 rankings across all subtasks while substantially reducing content effects and offering a competitive alternative to complex fine-tuning or activation-level interventions.
>
---
#### [new 041] Evaluating Cross-Modal Reasoning Ability and Problem Characteristics with Multimodal Item Response Theory
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于多模态模型评估任务，旨在解决现有基准中存在单模态解题的低质量问题。提出M3IRT框架，分解模型能力和题目难度，提升跨模态推理评估的可靠性与效率。**

- **链接: [https://arxiv.org/pdf/2603.02663](https://arxiv.org/pdf/2603.02663)**

> **作者:** Shunki Uebayashi; Kento Masui; Kyohei Atarashi; Han Bao; Hisashi Kashima; Naoto Inoue; Mayu Otani; Koh Takeuchi
>
> **备注:** 24pages, 20 figures, accepted to ICLR2026
>
> **摘要:** Multimodal Large Language Models (MLLMs) have recently emerged as general architectures capable of reasoning over diverse modalities. Benchmarks for MLLMs should measure their ability for cross-modal integration. However, current benchmarks are filled with shortcut questions, which can be solved using only a single modality, thereby yielding unreliable rankings. For example, in vision-language cases, we can find the correct answer without either the image or the text. These low-quality questions unnecessarily increase the size and computational requirements of benchmarks. We introduce a multi-modal and multidimensional item response theory framework (M3IRT) that extends classical IRT by decomposing both model ability and item difficulty into image-only, text-only, and cross-modal components. M3IRT estimates cross-modal ability of MLLMs and each question's cross-modal difficulty, enabling compact, high-quality subsets that better reflect multimodal reasoning. Across 24 VLMs on three benchmarks, M3IRT prioritizes genuinely cross-modal questions over shortcuts and preserves ranking fidelity even when 50% of items are artificially generated low-quality questions, thereby reducing evaluation cost while improving reliability. M3IRT thus offers a practical tool for assessing cross-modal reasoning and refining multimodal benchmarks.
>
---
#### [new 042] Incremental Graph Construction Enables Robust Spectral Clustering of Texts
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于文本聚类任务，解决标准k-NN图在低k值时易断连的问题。提出增量k-NN构造方法，确保图连通性，提升谱聚类鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.03056](https://arxiv.org/pdf/2603.03056)**

> **作者:** Marko Pranjić; Boshko Koloski; Nada Lavrač; Senja Pollak; Marko Robnik-Šikonja
>
> **备注:** MP and BK contributed equally
>
> **摘要:** Neighborhood graphs are a critical but often fragile step in spectral clustering of text embeddings. On realistic text datasets, standard $k$-NN graphs can contain many disconnected components at practical sparsity levels (small $k$), making spectral clustering degenerate and sensitive to hyperparameters. We introduce a simple incremental $k$-NN graph construction that preserves connectivity by design: each new node is linked to its $k$ nearest previously inserted nodes, which guarantees a connected graph for any $k$. We provide an inductive proof of connectedness and discuss implications for incremental updates when new documents arrive. We validate the approach on spectral clustering of SentenceTransformer embeddings using Laplacian eigenmaps across six clustering datasets from the Massive Text Embedding this http URL to standard $k$-NN graphs, our method outperforms in the low-$k$ regime where disconnected components are prevalent, and matches standard $k$-NN at larger $k$.
>
---
#### [new 043] Interpreting Speaker Characteristics in the Dimensions of Self-Supervised Speech Features
- **分类: eess.AS; cs.CL**

- **简介: 该论文研究自监督学习语音特征中说话人属性的表征，解决如何在特征维度中捕捉语音特性的问题。通过PCA分析，发现不同主成分对应不同语音特征，并验证可通过调整维度控制语音合成结果。**

- **链接: [https://arxiv.org/pdf/2603.03096](https://arxiv.org/pdf/2603.03096)**

> **作者:** Kyle Janse van Rensburg; Benjamin van Niekerk; Herman Kamper
>
> **备注:** 5 pages, 7 figures, submitted to IEEE Signal Processing Letters
>
> **摘要:** How do speech models trained through self-supervised learning structure their representations? Previous studies have looked at how information is encoded in feature vectors across different layers. But few studies have considered whether speech characteristics are captured within individual dimensions of SSL features. In this paper we specifically look at speaker information using PCA on utterance-averaged representations. Using WavLM, we find that the principal dimension that explains most variance encodes pitch and associated characteristics like gender. Other individual principal dimensions correlate with intensity, noise levels, the second formant, and higher frequency characteristics. Finally, in synthesis experiments we show that most characteristics can be controlled by changing the corresponding dimensions. This provides a simple method to control characteristics of the output voice in synthesis applications.
>
---
#### [new 044] Density-Guided Response Optimization: Community-Grounded Alignment via Implicit Acceptance Signals
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于语言模型对齐任务，解决在线社区中缺乏显式偏好标签的问题。通过分析内容接受行为，利用密度引导优化模型响应，使其符合社区规范。**

- **链接: [https://arxiv.org/pdf/2603.03242](https://arxiv.org/pdf/2603.03242)**

> **作者:** Patrick Gerard; Svitlana Volkova
>
> **备注:** 27 Pages
>
> **摘要:** Language models deployed in online communities must adapt to norms that vary across social, cultural, and domain-specific contexts. Prior alignment approaches rely on explicit preference supervision or predefined principles, which are effective for well-resourced settings but exclude most online communities -- particularly those without institutional backing, annotation infrastructure, or organized around sensitive topics -- where preference elicitation is costly, ethically fraught, or culturally misaligned. We observe that communities already express preferences implicitly through what content they accept, engage with, and allow to persist. We show that this acceptance behavior induces measurable geometric structure in representation space: accepted responses occupy coherent, high-density regions that reflect community-specific norms, while rejected content falls in sparser or misaligned areas. We operationalize this structure as an implicit preference signal for alignment and introduce density-guided response optimization (DGRO), a method that aligns language models to community norms without requiring explicit preference labels. Using labeled preference data, we demonstrate that local density recovers pairwise community judgments, indicating that geometric structure encodes meaningful preference signal. We then apply DGRO in annotation-scarce settings across diverse communities spanning platform, topic, and language. DGRO-aligned models consistently produce responses preferred by human annotators, domain experts, and model-based judges over supervised and prompt-based baselines. We position DGRO as a practical alignment alternative for communities where explicit preference supervision is unavailable or misaligned with situated practices, and discuss the implications and risks of learning from emergent acceptance behavior.
>
---
#### [new 045] Understanding and Mitigating Dataset Corruption in LLM Steering
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究对比引导在数据污染下的鲁棒性，旨在提升LLM安全应用中的稳定性。工作包括分析数据污染影响并提出改进的均值估计方法。**

- **链接: [https://arxiv.org/pdf/2603.03206](https://arxiv.org/pdf/2603.03206)**

> **作者:** Cullen Anderson; Narmeen Oozeer; Foad Namjoo; Remy Ogasawara; Amirali Abdullah; Jeff M. Phillips
>
> **摘要:** Contrastive steering has been shown as a simple and effective method to adjust the generative behavior of LLMs at inference time. It uses examples of prompt responses with and without a trait to identify a direction in an intermediate activation layer, and then shifts activations in this 1-dimensional subspace. However, despite its growing use in AI safety applications, the robustness of contrastive steering to noisy or adversarial data corruption is poorly understood. We initiate a study of the robustness of this process with respect to corruption of the dataset of examples used to train the steering direction. Our first observation is that contrastive steering is quite robust to a moderate amount of corruption, but unwanted side effects can be clearly and maliciously manifested when a non-trivial fraction of the training data is altered. Second, we analyze the geometry of various types of corruption, and identify some safeguards. Notably, a key step in learning the steering direction involves high-dimensional mean computation, and we show that replacing this step with a recently developed robust mean estimator often mitigates most of the unwanted effects of malicious corruption.
>
---
#### [new 046] Safety Training Persists Through Helpfulness Optimization in LLM Agents
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究LLM代理在多步骤工具使用场景下的安全训练问题，探讨安全与有用性优化的协同效果。任务属于安全训练领域，旨在解决如何有效平衡安全与有用性的问题。工作对比了不同优化策略的效果。**

- **链接: [https://arxiv.org/pdf/2603.02229](https://arxiv.org/pdf/2603.02229)**

> **作者:** Benjamin Plaut
>
> **备注:** Under submission
>
> **摘要:** Safety post-training has been studied extensively in single-step "chat" settings where safety typically refers to refusing harmful requests. We study an "agentic" (i.e., multi-step, tool-use) setting where safety refers to harmful actions directly taken by the LLM. We compare the effects of running direct preference optimization (DPO) on safety or helpfulness alone vs both metrics sequentially. As expected, training on one metric alone results in an extreme point along this frontier. However, unlike prior work, we find that safety training persists through subsequent helpfulness training. We also find that all training configurations end up near a linear Pareto frontier with $R^2 = 0.77$. Even post-training on both metrics simultaneously simply results in another point on the frontier rather than finding a "best of both worlds" strategy, despite the presence of such strategies in our DPO dataset. Overall, our findings underscore the need for better understanding of post-training dynamics.
>
---
#### [new 047] A Directed Graph Model and Experimental Framework for Design and Study of Time-Dependent Text Visualisation
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文属于文本可视化任务，旨在研究时间依赖的文本关系可视化。通过构建有向图模型和合成数据集，测试用户识别文本关联模式的能力，发现用户理解存在困难，提示需个性化视觉设计。**

- **链接: [https://arxiv.org/pdf/2603.02422](https://arxiv.org/pdf/2603.02422)**

> **作者:** Songhai Fan; Simon Angus; Tim Dwyer; Ying Yang; Sarah Goodwin; Helen Purchase
>
> **备注:** preprint version for TVCG submission
>
> **摘要:** Exponential growth in the quantity of digital news, social media, and other textual sources makes it difficult for humans to keep up with rapidly evolving narratives about world events. Various visualisation techniques have been touted to help people to understand such discourse by exposing relationships between texts (such as news articles) as topics and themes evolve over time. Arguably, the understandability of such visualisations hinges on the assumption that people will be able to easily interpret the relationships in such visual network structures. To test this assumption, we begin by defining an abstract model of time-dependent text visualisation based on directed graph structures. From this model we distill motifs that capture the set of possible ways that texts can be linked across changes in time. We also develop a controlled synthetic text generation methodology that leverages the power of modern LLMs to create fictional, yet structured sets of time-dependent texts that fit each of our patterns. Therefore, we create a clean user study environment (n=30) for participants to identify patterns that best represent a given set of synthetic articles. We find that it is a challenging task for the user to identify and recover the predefined motif. We analyse qualitative data to map an unexpectedly rich variety of user rationales when divergences from expected interpretation occur. A deeper analysis also points to unexpected complexities inherent in the formation of synthetic datasets with LLMs that undermine the study control in some cases. Furthermore, analysis of individual decision-making in our study hints at a future where text discourse visualisation may need to dispense with a one-size-fits-all approach and, instead, should be more adaptable to the specific user who is exploring the visualisation in front of them.
>
---
#### [new 048] MoD-DPO: Towards Mitigating Cross-modal Hallucinations in Omni LLMs using Modality Decoupled Preference Optimization
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于多模态任务，旨在解决Omni LLMs中的跨模态幻觉问题。提出MoD-DPO框架，通过模态解耦优化提升模型对不同模态的准确感知与抗幻觉能力。**

- **链接: [https://arxiv.org/pdf/2603.03192](https://arxiv.org/pdf/2603.03192)**

> **作者:** Ashutosh Chaubey; Jiacheng Pang; Mohammad Soleymani
>
> **备注:** CVPR 2026. Project Page: this https URL
>
> **摘要:** Omni-modal large language models (omni LLMs) have recently achieved strong performance across audiovisual understanding tasks, yet they remain highly susceptible to cross-modal hallucinations arising from spurious correlations and dominant language priors. In this work, we propose Modality-Decoupled Direct Preference Optimization (MoD-DPO), a simple and effective framework for improving modality grounding in omni LLMs. MoD-DPO introduces modality-aware regularization terms that explicitly enforce invariance to corruptions in irrelevant modalities and sensitivity to perturbations in relevant modalities, thereby reducing unintended cross-modal interactions. To further mitigate over-reliance on textual priors, we incorporate a language-prior debiasing penalty that discourages hallucination-prone text-only responses. Extensive experiments across multiple audiovisual hallucination benchmarks demonstrate that MoD-DPO consistently improves perception accuracy and hallucination resistance, outperforming previous preference optimization baselines under similar training budgets. Our findings underscore the importance of modality-faithful alignment and demonstrate a scalable path toward more reliable and resilient multimodal foundation models.
>
---
#### [new 049] Guideline-Grounded Evidence Accumulation for High-Stakes Agent Verification
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于高风险决策验证任务，旨在提升LLM代理决策的可靠性。提出GLEAN框架，通过整合领域指南和证据累积，提高验证准确性与校准度。**

- **链接: [https://arxiv.org/pdf/2603.02798](https://arxiv.org/pdf/2603.02798)**

> **作者:** Yichi Zhang; Nabeel Seedat; Yinpeng Dong; Peng Cui; Jun Zhu; Mihaela van de Schaar
>
> **摘要:** As LLM-powered agents have been used for high-stakes decision-making, such as clinical diagnosis, it becomes critical to develop reliable verification of their decisions to facilitate trustworthy deployment. Yet, existing verifiers usually underperform owing to a lack of domain knowledge and limited calibration. To address this, we establish GLEAN, an agent verification framework with Guideline-grounded Evidence Accumulation that compiles expert-curated protocols into trajectory-informed, well-calibrated correctness signals. GLEAN evaluates the step-wise alignment with domain guidelines and aggregates multi-guideline ratings into surrogate features, which are accumulated along the trajectory and calibrated into correctness probabilities using Bayesian logistic regression. Moreover, the estimated uncertainty triggers active verification, which selectively collects additional evidence for uncertain cases via expanding guideline coverage and performing differential checks. We empirically validate GLEAN with agentic clinical diagnosis across three diseases from the MIMIC-IV dataset, surpassing the best baseline by 12% in AUROC and 50% in Brier score reduction, which confirms the effectiveness in both discrimination and calibration. In addition, the expert study with clinicians recognizes GLEAN's utility in practice.
>
---
#### [new 050] MUSE: A Run-Centric Platform for Multimodal Unified Safety Evaluation of Large Language Models
- **分类: cs.LG; cs.CL; cs.CV; cs.SD; eess.AS**

- **简介: 该论文属于大模型安全评估任务，旨在解决多模态对齐评估不足的问题。提出MUSE平台，集成多模态攻击与评估方法，提升安全测试效果。**

- **链接: [https://arxiv.org/pdf/2603.02482](https://arxiv.org/pdf/2603.02482)**

> **作者:** Zhongxi Wang; Yueqian Lin; Jingyang Zhang; Hai Helen Li; Yiran Chen
>
> **备注:** Submitted to ACL 2026 System Demonstration Track
>
> **摘要:** Safety evaluation and red-teaming of large language models remain predominantly text-centric, and existing frameworks lack the infrastructure to systematically test whether alignment generalizes to audio, image, and video inputs. We present MUSE (Multimodal Unified Safety Evaluation), an open-source, run-centric platform that integrates automatic cross-modal payload generation, three multi-turn attack algorithms (Crescendo, PAIR, Violent Durian), provider-agnostic model routing, and an LLM judge with a five-level safety taxonomy into a single browser-based system. A dual-metric framework distinguishes hard Attack Success Rate (Compliance only) from soft ASR (including Partial Compliance), capturing partial information leakage that binary metrics miss. To probe whether alignment generalizes across modality boundaries, we introduce Inter-Turn Modality Switching (ITMS), which augments multi-turn attacks with per-turn modality rotation. Experiments across six multimodal LLMs from four providers show that multi-turn strategies can achieve up to 90-100% ASR against models with near-perfect single-turn refusal. ITMS does not uniformly raise final ASR on already-saturated baselines, but accelerates convergence by destabilizing early-turn defenses, and ablation reveals that the direction of modality effects is model-family-specific rather than universal, underscoring the need for provider-aware cross-modal safety testing.
>
---
#### [new 051] Type-Aware Retrieval-Augmented Generation with Dependency Closure for Solver-Executable Industrial Optimization Modeling
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文属于工业优化建模任务，解决自然语言到可执行代码转换中的编译问题。通过构建类型感知的知识库和依赖闭包，提升模型生成的可执行性。**

- **链接: [https://arxiv.org/pdf/2603.03180](https://arxiv.org/pdf/2603.03180)**

> **作者:** Y. Zhong; R. Huang; M. Wang; Z. Guo; YC. Li; M. Yu; Z. Jin
>
> **摘要:** Automated industrial optimization modeling requires reliable translation of natural-language requirements into solver-executable code. However, large language models often generate non-compilable models due to missing declarations, type inconsistencies, and incomplete dependency contexts. We propose a type-aware retrieval-augmented generation (RAG) method that enforces modeling entity types and minimal dependency closure to ensure executability. Unlike existing RAG approaches that index unstructured text, our method constructs a domain-specific typed knowledge base by parsing heterogeneous sources, such as academic papers and solver code, into typed units and encoding their mathematical dependencies in a knowledge graph. Given a natural-language instruction, it performs hybrid retrieval and computes a minimal dependency-closed context, the smallest set of typed symbols required for solver-executable code, via dependency propagation over the graph. We validate the method on two constraint-intensive industrial cases: demand response optimization in battery production and flexible job shop scheduling. In the first case, our method generates an executable model incorporating demand-response incentives and load-reduction constraints, achieving peak shaving while preserving profitability; conventional RAG baselines fail. In the second case, it consistently produces compilable models that reach known optimal solutions, demonstrating robust cross-domain generalization; baselines fail entirely. Ablation studies confirm that enforcing type-aware dependency closure is essential for avoiding structural hallucinations and ensuring executability, addressing a critical barrier to deploying large language models in complex engineering optimization tasks.
>
---
#### [new 052] No Memorization, No Detection: Output Distribution-Based Contamination Detection in Small Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于数据污染检测任务，研究输出分布方法在小模型中的有效性，发现参数高效微调可避免记忆，导致检测失效。**

- **链接: [https://arxiv.org/pdf/2603.03203](https://arxiv.org/pdf/2603.03203)**

> **作者:** Omer Sela
>
> **备注:** 8 pages main text, 5 pages appendix, 9 figures, 7 tables. Code available at this https URL
>
> **摘要:** CDD, or Contamination Detection via output Distribution, identifies data contamination by measuring the peakedness of a model's sampled outputs. We study the conditions under which this approach succeeds and fails on small language models ranging from 70M to 410M parameters. Using controlled contamination experiments on GSM8K, HumanEval, and MATH, we find that CDD's effectiveness depends critically on whether fine-tuning produces verbatim memorization. With low-rank adaptation, models can learn from contaminated data without memorizing it, and CDD performs at chance level even when the data is verifiably contaminated. Only when fine-tuning capacity is sufficient to induce memorization does CDD recover strong detection accuracy. Our results characterize a memorization threshold that governs detectability and highlight a practical consideration: parameter-efficient fine-tuning can produce contamination that output-distribution methods do not detect. Our code is available at this https URL
>
---
#### [new 053] FlashEvaluator: Expanding Search Space with Parallel Evaluation
- **分类: cs.IR; cs.CL; cs.LG**

- **简介: 该论文属于推荐系统和自然语言处理任务，解决传统评估器缺乏跨序列比较和效率低的问题。提出FlashEvaluator，实现高效并行评估和准确选择。**

- **链接: [https://arxiv.org/pdf/2603.02565](https://arxiv.org/pdf/2603.02565)**

> **作者:** Chao Feng; Yuanhao Pu; Chenghao Zhang; Shanqi Liu; Shuchang Liu; Xiang Li; Yongqi Liu; Lantao Hu; Kaiqiao Zhan; Han Li; Kun Gai
>
> **备注:** 23 pages, 2 figures
>
> **摘要:** The Generator-Evaluator (G-E) framework, i.e., evaluating K sequences from a generator and selecting the top-ranked one according to evaluator scores, is a foundational paradigm in tasks such as Recommender Systems (RecSys) and Natural Language Processing (NLP). Traditional evaluators process sequences independently, suffering from two major limitations: (1) lack of explicit cross-sequence comparison, leading to suboptimal accuracy; (2) poor parallelization with linear complexity of O(K), resulting in inefficient resource utilization and negative impact on both throughput and latency. To address these challenges, we propose FlashEvaluator, which enables cross-sequence token information sharing and processes all sequences in a single forward pass. This yields sublinear computational complexity that improves the system's efficiency and supports direct inter-sequence comparisons that improve selection accuracy. The paper also provides theoretical proofs and extensive experiments on recommendation and NLP tasks, demonstrating clear advantages over conventional methods. Notably, FlashEvaluator has been deployed in online recommender system of Kuaishou, delivering substantial and sustained revenue gains in practice.
>
---
#### [new 054] Credibility Governance: A Social Mechanism for Collective Self-Correction under Weak Truth Signals
- **分类: cs.CY; cs.AI; cs.CL; cs.MA; cs.SI**

- **简介: 该论文提出一种可信度治理机制，用于在弱真实信号环境下提升集体判断的可靠性。任务是改善在线平台的共识形成过程，解决虚假信息和噪声干扰问题，通过动态评估参与者和观点的可信度来优化决策。**

- **链接: [https://arxiv.org/pdf/2603.02640](https://arxiv.org/pdf/2603.02640)**

> **作者:** Wanying He; Yanxi Lin; Ziheng Zhou; Xue Feng; Min Peng; Qianqian Xie; Zilong Zheng; Yipeng Kang
>
> **摘要:** Online platforms increasingly rely on opinion aggregation to allocate real-world attention and resources, yet common signals such as engagement votes or capital-weighted commitments are easy to amplify and often track visibility rather than reliability. This makes collective judgments brittle under weak truth signals, noisy or delayed feedback, early popularity surges, and strategic manipulation. We propose Credibility Governance (CG), a mechanism that reallocates influence by learning which agents and viewpoints consistently track evolving public evidence. CG maintains dynamic credibility scores for both agents and opinions, updates opinion influence via credibility-weighted endorsements, and updates agent credibility based on the long-run performance of the opinions they support, rewarding early and persistent alignment with emerging evidence while filtering short-lived noise. We evaluate CG in POLIS, a socio-physical simulation environment that models coupled belief dynamics and downstream feedback under uncertainty. Across settings with initial majority misalignment, observation noise and contamination, and misinformation shocks, CG outperforms vote-based, stake-weighted, and no-governance baselines, yielding faster recovery to the true state, reduced lock-in and path dependence, and improved robustness under adversarial pressure. Our implementation and experimental scripts are publicly available at this https URL.
>
---
#### [new 055] TikZilla: Scaling Text-to-TikZ with High-Quality Data and Reinforcement Learning
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于文本生成TikZ图像的任务，旨在解决数据不足和生成错误问题。通过构建高质量数据集并结合强化学习训练模型，提升生成质量。**

- **链接: [https://arxiv.org/pdf/2603.03072](https://arxiv.org/pdf/2603.03072)**

> **作者:** Christian Greisinger; Steffen Eger
>
> **摘要:** Large language models (LLMs) are increasingly used to assist scientists across diverse workflows. A key challenge is generating high-quality figures from textual descriptions, often represented as TikZ programs that can be rendered as scientific images. Prior research has proposed a variety of datasets and modeling approaches for this task. However, existing datasets for Text-to-TikZ are too small and noisy to capture the complexity of TikZ, causing mismatches between text and rendered figures. Moreover, prior approaches rely solely on supervised fine-tuning (SFT), which does not expose the model to the rendered semantics of the figure, often resulting in errors such as looping, irrelevant content, and incorrect spatial relations. To address these issues, we construct DaTikZ-V4, a dataset more than four times larger and substantially higher in quality than DaTikZ-V3, enriched with LLM-generated figure descriptions. Using this dataset, we train TikZilla, a family of small open-source Qwen models (3B and 8B) with a two-stage pipeline of SFT followed by reinforcement learning (RL). For RL, we leverage an image encoder trained via inverse graphics to provide semantically faithful reward signals. Extensive human evaluations with over 1,000 judgments show that TikZilla improves by 1.5-2 points over its base models on a 5-point scale, surpasses GPT-4o by 0.5 points, and matches GPT-5 in the image-based evaluation, while operating at much smaller model sizes. Code, data, and models will be made available.
>
---
#### [new 056] Contextualized Privacy Defense for LLM Agents
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于隐私保护任务，旨在解决LLM代理在处理用户信息时的隐私问题。提出CDI框架，通过上下文感知指导提升隐私与帮助性的平衡。**

- **链接: [https://arxiv.org/pdf/2603.02983](https://arxiv.org/pdf/2603.02983)**

> **作者:** Yule Wen; Yanzhe Zhang; Jianxun Lian; Xiaoyuan Yi; Xing Xie; Diyi Yang
>
> **备注:** 25 pages
>
> **摘要:** LLM agents increasingly act on users' personal information, yet existing privacy defenses remain limited in both design and adaptability. Most prior approaches rely on static or passive defenses, such as prompting and guarding. These paradigms are insufficient for supporting contextual, proactive privacy decisions in multi-step agent execution. We propose Contextualized Defense Instructing (CDI), a new privacy defense paradigm in which an instructor model generates step-specific, context-aware privacy guidance during execution, proactively shaping actions rather than merely constraining or vetoing them. Crucially, CDI is paired with an experience-driven optimization framework that trains the instructor via reinforcement learning (RL), where we convert failure trajectories with privacy violations into learning environments. We formalize baseline defenses and CDI as distinct intervention points in a canonical agent loop, and compare their privacy-helpfulness trade-offs within a unified simulation framework. Results show that our CDI consistently achieves a better balance between privacy preservation (94.2%) and helpfulness (80.6%) than baselines, with superior robustness to adversarial conditions and generalization.
>
---
#### [new 057] Self-Play Only Evolves When Self-Synthetic Pipeline Ensures Learnable Information Gain
- **分类: cs.LG; cs.AI; cs.CL; cs.IT**

- **简介: 该论文研究自进化语言模型的可持续性问题，提出通过三角色系统设计提升信息增益，解决自对弈易停滞的问题。**

- **链接: [https://arxiv.org/pdf/2603.02218](https://arxiv.org/pdf/2603.02218)**

> **作者:** Wei Liu; Siya Qi; Yali Du; Yulan He
>
> **备注:** 10 pages, 6 figures, 7 formulas
>
> **摘要:** Large language models (LLMs) make it plausible to build systems that improve through self-evolving loops, but many existing proposals are better understood as self-play and often plateau quickly. A central failure mode is that the loop synthesises more data without increasing learnable information for the next iteration. Through experiments on a self-play coding task, we reveal that sustainable self-evolution requires a self-synthesised data pipeline with learnable information that increases across iterations. We identify triadic roles that self-evolving LLMs play: the Proposer, which generates tasks; the Solver, which attempts solutions; and the Verifier, which provides training signals, and we identify three system designs that jointly target learnable information gain from this triadic roles perspective. Asymmetric co-evolution closes a weak-to-strong-to-weak loop across roles. Capacity growth expands parameter and inference-time budgets to match rising learnable information. Proactive information seeking introduces external context and new task sources that prevent saturation. Together, these modules provide a measurable, system-level path from brittle self-play dynamics to sustained self-evolution.
>
---
#### [new 058] Routing Absorption in Sparse Attention: Why Random Gates Are Hard to Beat
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究稀疏注意力中的路由吸收问题，探讨为何随机门控表现优于学习的门控。属于自然语言处理任务，旨在解决稀疏注意力训练中的门控失效问题。**

- **链接: [https://arxiv.org/pdf/2603.02227](https://arxiv.org/pdf/2603.02227)**

> **作者:** Keston Aquino-Michaels
>
> **备注:** 14 pages, 4 figures
>
> **摘要:** Can a transformer learn which attention entries matter during training? In principle, yes: attention distributions are highly concentrated, and a small gate network can identify the important entries post-hoc with near-perfect accuracy. In practice, barely. When sparse attention is trained end-to-end, the model's Q/K/V projections co-adapt to whatever mask is imposed, absorbing the routing signal until learned gates perform little better than frozen random gates. We call this routing absorption and present four independent lines of evidence for it in a controlled 31M-parameter transformer: (1) differentiable soft gating converges to nearly the same perplexity whether the gate is learned or random (48.73 +/- 0.60 vs. 49.83 +/- 0.04 over 3 seeds); (2) hard top-k gating receives exactly zero gradient through the mask; (3) a gate distilled onto co-adapted Q/K/V achieves high F1 against oracle masks but catastrophic perplexity when deployed (601.6 vs. 48.6 on mask-agnostic Q/K/V); and (4) stochastic mask randomization during training fails to prevent co-adaptation (78.2 ppl deployed dense vs. 37.3 baseline). We connect routing absorption to the same phenomenon in Mixture-of-Experts, where random routing matches learned routing because experts co-adapt to any router, but show that attention exhibits a structurally more severe form: shared Q/K/V parameters enable cross-layer compensation pathways absent in MoE, where experts are self-contained modules. The implication is that end-to-end sparse attention methods employing per-query token-level gating face absorption pressure proportional to the parameter asymmetry between the gate and the model, and that post-hoc approaches, which decouple representation learning from sparsification, sidestep this entirely.
>
---
#### [new 059] Through the Lens of Contrast: Self-Improving Visual Reasoning in VLMs
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于视觉语言模型的推理任务，旨在解决视觉幻觉问题。通过引入视觉对比机制，提出VC-STaR框架，生成更准确的推理路径，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.02556](https://arxiv.org/pdf/2603.02556)**

> **作者:** Zhiyu Pan; Yizheng Wu; Jiashen Hua; Junyi Feng; Shaotian Yan; Bing Deng; Zhiguo Cao; Jieping Ye
>
> **备注:** 19 pages, 9 figures, accepted to ICLR 2026 (oral)
>
> **摘要:** Reasoning has emerged as a key capability of large language models. In linguistic tasks, this capability can be enhanced by self-improving techniques that refine reasoning paths for subsequent finetuning. However, extending these language-based self-improving approaches to vision language models (VLMs) presents a unique challenge:~visual hallucinations in reasoning paths cannot be effectively verified or rectified. Our solution starts with a key observation about visual contrast: when presented with a contrastive VQA pair, i.e., two visually similar images with synonymous questions, VLMs identify relevant visual cues more precisely. Motivated by this observation, we propose Visual Contrastive Self-Taught Reasoner (VC-STaR), a novel self-improving framework that leverages visual contrast to mitigate hallucinations in model-generated rationales. We collect a diverse suite of VQA datasets, curate contrastive pairs according to multi-modal similarity, and generate rationales using VC-STaR. Consequently, we obtain a new visual reasoning dataset, VisCoR-55K, which is then used to boost the reasoning capability of various VLMs through supervised finetuning. Extensive experiments show that VC-STaR not only outperforms existing self-improving approaches but also surpasses models finetuned on the SoTA visual reasoning datasets, demonstrating that the inherent contrastive ability of VLMs can bootstrap their own visual reasoning. Project at: this https URL.
>
---
#### [new 060] HELIOS: Harmonizing Early Fusion, Late Fusion, and LLM Reasoning for Multi-Granular Table-Text Retrieval
- **分类: cs.DB; cs.CL; cs.IR; cs.LG**

- **简介: 该论文属于多粒度表格-文本检索任务，旨在解决传统方法在上下文相关性和复杂推理上的不足。提出HELIOS模型，融合早期与后期融合优势，并引入LLM推理提升性能。**

- **链接: [https://arxiv.org/pdf/2603.02248](https://arxiv.org/pdf/2603.02248)**

> **作者:** Sungho Park; Joohyung Yun; Jongwuk Lee; Wook-Shin Han
>
> **备注:** 9 pages, 6 figures. Accepted at ACL 2025 main. Project page: this https URL
>
> **摘要:** Table-text retrieval aims to retrieve relevant tables and text to support open-domain question answering. Existing studies use either early or late fusion, but face limitations. Early fusion pre-aligns a table row with its associated passages, forming "stars," which often include irrelevant contexts and miss query-dependent relationships. Late fusion retrieves individual nodes, dynamically aligning them, but it risks missing relevant contexts. Both approaches also struggle with advanced reasoning tasks, such as column-wise aggregation and multi-hop reasoning. To address these issues, we propose HELIOS, which combines the strengths of both approaches. First, the edge-based bipartite subgraph retrieval identifies finer-grained edges between table segments and passages, effectively avoiding the inclusion of irrelevant contexts. Then, the query-relevant node expansion identifies the most promising nodes, dynamically retrieving relevant edges to grow the bipartite subgraph, minimizing the risk of missing important contexts. Lastly, the star-based LLM refinement performs logical inference at the star graph level rather than the bipartite subgraph, supporting advanced reasoning tasks. Experimental results show that HELIOS outperforms state-of-the-art models with a significant improvement up to 42.6\% and 39.9\% in recall and nDCG, respectively, on the OTT-QA benchmark.
>
---
#### [new 061] ACE-Brain-0: Spatial Intelligence as a Shared Scaffold for Universal Embodiments
- **分类: cs.RO; cs.CL; cs.CV**

- **简介: 该论文属于多模态大模型任务，旨在解决跨身体形态的通用智能问题。通过构建ACE-Brain-0，融合空间推理与不同应用场景，提升模型的泛化能力与专业性能。**

- **链接: [https://arxiv.org/pdf/2603.03198](https://arxiv.org/pdf/2603.03198)**

> **作者:** Ziyang Gong; Zehang Luo; Anke Tang; Zhe Liu; Shi Fu; Zhi Hou; Ganlin Yang; Weiyun Wang; Xiaofeng Wang; Jianbo Liu; Gen Luo; Haolan Kang; Shuang Luo; Yue Zhou; Yong Luo; Li Shen; Xiaosong Jia; Yao Mu; Xue Yang; Chunxiao Liu; Junchi Yan; Hengshuang Zhao; Dacheng Tao; Xiaogang Wang
>
> **备注:** Code: this https URL Hugging Face: this https URL
>
> **摘要:** Universal embodied intelligence demands robust generalization across heterogeneous embodiments, such as autonomous driving, robotics, and unmanned aerial vehicles (UAVs). However, existing embodied brain in training a unified model over diverse embodiments frequently triggers long-tail data, gradient interference, and catastrophic forgetting, making it notoriously difficult to balance universal generalization with domain-specific proficiency. In this report, we introduce ACE-Brain-0, a generalist foundation brain that unifies spatial reasoning, autonomous driving, and embodied manipulation within a single multimodal large language model~(MLLM). Our key insight is that spatial intelligence serves as a universal scaffold across diverse physical embodiments: although vehicles, robots, and UAVs differ drastically in morphology, they share a common need for modeling 3D mental space, making spatial cognition a natural, domain-agnostic foundation for cross-embodiment transfer. Building on this insight, we propose the Scaffold-Specialize-Reconcile~(SSR) paradigm, which first establishes a shared spatial foundation, then cultivates domain-specialized experts, and finally harmonizes them through data-free model merging. Furthermore, we adopt Group Relative Policy Optimization~(GRPO) to strengthen the model's comprehensive capability. Extensive experiments demonstrate that ACE-Brain-0 achieves competitive and even state-of-the-art performance across 24 spatial and embodiment-related benchmarks.
>
---
#### [new 062] StitchCUDA: An Automated Multi-Agents End-to-End GPU Programing Framework with Rubric-based Agentic Reinforcement Learning
- **分类: cs.MA; cs.CL; cs.PL**

- **简介: 该论文提出StitchCUDA框架，解决GPU程序自动化生成问题。通过多智能体协作与基于规则的强化学习，提升端到端性能与正确性。**

- **链接: [https://arxiv.org/pdf/2603.02637](https://arxiv.org/pdf/2603.02637)**

> **作者:** Shiyang Li; Zijian Zhang; Winson Chen; Yuebo Luo; Mingyi Hong; Caiwen Ding
>
> **摘要:** Modern machine learning (ML) workloads increasingly rely on GPUs, yet achieving high end-to-end performance remains challenging due to dependencies on both GPU kernel efficiency and host-side settings. Although LLM-based methods show promise on automated GPU kernel generation, prior works mainly focus on single-kernel optimization and do not extend to end-to-end programs, hindering practical deployment. To address the challenge, in this work, we propose StitchCUDA, a multi-agent framework for end-to-end GPU program generation, with three specialized agents: a Planner to orchestrate whole system design, a Coder dedicated to implementing it step-by-step, and a Verifier for correctness check and performance profiling using Nsys/NCU. To fundamentally improve the Coder's ability in end-to-end GPU programming, StitchCUDA integrates rubric-based agentic reinforcement learning over two atomic skills, task-to-code generation and feedback-driven code optimization, with combined rubric reward and rule-based reward from real executions. Therefore, the Coder learns how to implement advanced CUDA programming techniques (e.g., custom kernel fusion, cublas epilogue), and we also effectively prevent Coder's reward hacking (e.g., just copy PyTorch code or hardcoding output) during benchmarking. Experiments on KernelBench show that StitchCUDA achieves nearly 100% success rate on end-to-end GPU programming tasks, with 1.72x better speedup over the multi-agent baseline and 2.73x than the RL model baselines.
>
---
## 更新

#### [replaced 001] Recursive Think-Answer Process for LLMs and VLMs
- **分类: cs.CL**

- **简介: 该论文提出R-TAP方法，用于提升LLMs和VLMs的推理准确性。针对单次推理易出错的问题，通过递归思考与信心评估实现迭代优化。**

- **链接: [https://arxiv.org/pdf/2603.02099](https://arxiv.org/pdf/2603.02099)**

> **作者:** Byung-Kwan Lee; Youngchae Chee; Yong Man Ro
>
> **备注:** CVPR 2026 Findings, Project page: this https URL
>
> **摘要:** Think-Answer reasoners such as DeepSeek-R1 have made notable progress by leveraging interpretable internal reasoning. However, despite the frequent presence of self-reflective cues like "Oops!", they remain vulnerable to output errors during single-pass inference. To address this limitation, we propose an efficient Recursive Think-Answer Process (R-TAP) that enables models to engage in iterative reasoning cycles and generate more accurate answers, going beyond conventional single-pass approaches. Central to this approach is a confidence generator that evaluates the certainty of model responses and guides subsequent improvements. By incorporating two complementary rewards-Recursively Confidence Increase Reward and Final Answer Confidence Reward-we show that R-TAP-enhanced models consistently outperform conventional single-pass methods for both large language models (LLMs) and vision-language models (VLMs). Moreover, by analyzing the frequency of "Oops"-like expressions in model responses, we find that R-TAP-applied models exhibit significantly fewer self-reflective patterns, resulting in more stable and faster inference-time reasoning. We hope R-TAP pave the way evolving into efficient and elaborated methods to refine the reasoning processes of future AI.
>
---
#### [replaced 002] Safety Verification of Wait-Only Non-Blocking Broadcast Protocols
- **分类: cs.LO; cs.CL; cs.MA**

- **简介: 该论文研究广播协议的安全性验证，解决状态和配置可覆盖性问题。针对仅等待的非阻塞协议，证明其问题分别为P-完全和PSPACE-完全。**

- **链接: [https://arxiv.org/pdf/2403.18591](https://arxiv.org/pdf/2403.18591)**

> **作者:** Lucie Guillou; Arnaud Sangnier; Nathalie Sznajder
>
> **备注:** submitted to Fundamenta Informaticae
>
> **摘要:** Broadcast protocols are programs designed to be executed by networks of processes. Each process runs the same protocol, and communication between them occurs in synchronously in two ways: broadcast, where one process sends a message to all others, and rendez-vous, where one process sends a message to at most one other process. In both cases, communication is non-blocking, meaning the message is sent even if no process is able to receive it. We consider two coverability problems: the state coverability problem asks whether there exists a number of processes that allows reaching a given state of the protocol, and the configuration coverability problem asks whether there exists a number of processes that allows covering a given configuration. These two problems are known to be decidable and Ackermann-hard. We show that when the protocol is Wait-Only (i.e., it has no state from which a process can both send and receive messages), these problems become P-complete and PSPACE-complete, respectively.
>
---
#### [replaced 003] Piecing Together Cross-Document Coreference Resolution Datasets: Systematic Dataset Analysis and Unification
- **分类: cs.CL**

- **简介: 该论文属于跨文档共指消解任务，旨在解决数据分散、标准不一的问题。通过构建统一数据集uCDCR，进行标准化分析与比较，提升研究的可复现性和模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.00621](https://arxiv.org/pdf/2603.00621)**

> **作者:** Anastasia Zhukova; Terry Ruas; Jan Philip Wahle; Bela Gipp
>
> **备注:** accepted to LREC 2026
>
> **摘要:** Research in CDCR remains fragmented due to heterogeneous dataset formats, varying annotation standards, and the predominance of the CDCR definition as the event coreference resolution (ECR). To address these challenges, we introduce uCDCR, a unified dataset that consolidates diverse publicly available English CDCR corpora across various domains into a consistent format, which we analyze with standardized metrics and evaluation protocols. uCDCR incorporates both entity and event coreference, corrects known inconsistencies, and enriches datasets with missing attributes to facilitate reproducible research. We establish a cohesive framework for fair, interpretable, and cross-dataset analysis in CDCR and compare the datasets on their lexical properties, e.g., lexical composition of the annotated mentions, lexical diversity and ambiguity metrics, discuss the annotation rules and principles that lead to high lexical diversity, and examine how these metrics influence performance on the same-head-lemma baseline. Our dataset analysis shows that ECB+, the state-of-the-art benchmark for CDCR, has one of the lowest lexical diversities, and its CDCR complexity, measured by the same-head-lemma baseline, lies in the middle among all uCDCR datasets. Moreover, comparing document and mention distributions between ECB+ and uCDCR shows that using all uCDCR datasets for model training and evaluation will improve the generalizability of CDCR models. Finally, the almost identical performance on the same-head-lemma baseline, separately applied to events and entities, shows that resolving both types is a complex task and should not be steered toward ECR alone. The uCDCR dataset is available at this https URL, and the code for parsing, analyzing, and scoring the dataset is available at this https URL.
>
---
#### [replaced 004] Diverging Preferences: When do Annotators Disagree and do Models Know?
- **分类: cs.CL**

- **简介: 该论文研究人类标注者在偏好数据集中的分歧问题，属于自然语言处理中的奖励建模与评估任务。旨在解决标注分歧对模型训练和评估的影响，通过分析分歧来源并提出应对方法。**

- **链接: [https://arxiv.org/pdf/2410.14632](https://arxiv.org/pdf/2410.14632)**

> **作者:** Michael JQ Zhang; Zhilin Wang; Jena D. Hwang; Yi Dong; Olivier Delalleau; Yejin Choi; Eunsol Choi; Xiang Ren; Valentina Pyatkin
>
> **备注:** ICML 2025
>
> **摘要:** We examine diverging preferences in human-labeled preference datasets. We develop a taxonomy of disagreement sources spanning ten categories across four high-level classes and find that the majority of disagreements are due to factors such as task underspecification or response style. Our findings challenge a standard assumption in reward modeling methods that annotator disagreements can be attributed to simple noise. We then explore how these findings impact two areas of LLM development: reward modeling training and evaluation. In our experiments, we demonstrate how standard reward modeling (e.g., Bradley-Terry) and LLM-as-Judge evaluation methods fail to account for divergence between annotators. These findings highlight challenges in LLM evaluations, which are greatly influenced by divisive features like response style, and in developing pluralistically aligned LLMs. To address these issues, we develop methods for identifying diverging preferences to mitigate their influence in evaluations and during LLM training.
>
---
#### [replaced 005] Monitoring AI-Modified Content at Scale: A Case Study on the Impact of ChatGPT on AI Conference Peer Reviews
- **分类: cs.CL; cs.AI; cs.LG; cs.SI**

- **简介: 该论文属于AI内容检测任务，旨在评估ChatGPT对AI会议同行评审文本的影响。通过构建最大似然模型，分析大量文本中可能被LLM修改或生成的比例，揭示用户行为与生成文本的关系。**

- **链接: [https://arxiv.org/pdf/2403.07183](https://arxiv.org/pdf/2403.07183)**

> **作者:** Weixin Liang; Zachary Izzo; Yaohui Zhang; Haley Lepp; Hancheng Cao; Xuandong Zhao; Lingjiao Chen; Haotian Ye; Sheng Liu; Zhi Huang; Daniel A. McFarland; James Y. Zou
>
> **备注:** 46 pages, 31 figures, ICML '24
>
> **摘要:** We present an approach for estimating the fraction of text in a large corpus which is likely to be substantially modified or produced by a large language model (LLM). Our maximum likelihood model leverages expert-written and AI-generated reference texts to accurately and efficiently examine real-world LLM-use at the corpus level. We apply this approach to a case study of scientific peer review in AI conferences that took place after the release of ChatGPT: ICLR 2024, NeurIPS 2023, CoRL 2023 and EMNLP 2023. Our results suggest that between 6.5% and 16.9% of text submitted as peer reviews to these conferences could have been substantially modified by LLMs, i.e. beyond spell-checking or minor writing updates. The circumstances in which generated text occurs offer insight into user behavior: the estimated fraction of LLM-generated text is higher in reviews which report lower confidence, were submitted close to the deadline, and from reviewers who are less likely to respond to author rebuttals. We also observe corpus-level trends in generated text which may be too subtle to detect at the individual level, and discuss the implications of such trends on peer review. We call for future interdisciplinary work to examine how LLM use is changing our information and knowledge practices.
>
---
#### [replaced 006] A Survey of Query Optimization in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在提升大语言模型的查询优化效果。通过分析查询优化流程、分类查询复杂度并探讨关键操作，提出系统性框架与方法，解决查询质量影响生成性能的问题。**

- **链接: [https://arxiv.org/pdf/2412.17558](https://arxiv.org/pdf/2412.17558)**

> **作者:** Mingyang Song; Mao Zheng
>
> **备注:** Ongoing Work
>
> **摘要:** Query Optimization (QO) has become essential for enhancing Large Language Model (LLM) effectiveness, particularly in Retrieval-Augmented Generation (RAG) systems where query quality directly determines retrieval and response performance. This survey provides a systematic and comprehensive analysis of query optimization techniques with three principal contributions. \textit{First}, we introduce the \textbf{Query Optimization Lifecycle (QOL) Framework}, a five-phase pipeline covering Intent Recognition, Query Transformation, Retrieval Execution, Evidence Integration, and Response Synthesis, providing a unified lens for understanding the optimization process. \textit{Second}, we propose a \textbf{Query Complexity Taxonomy} that classifies queries along two dimensions, namely evidence type (explicit vs.\ implicit) and evidence quantity (single vs.\ multiple), establishing principled mappings between query characteristics and optimization strategies. \textit{Third}, we conduct an in-depth analysis of four atomic operations, namely \textbf{Query Expansion}, \textbf{Query Decomposition}, \textbf{Query Disambiguation}, and \textbf{Query Abstraction}, synthesizing a broad spectrum of representative methods from premier venues. We further examine evaluation methodologies, identify critical gaps in existing benchmarks, and discuss open challenges including process reward models, efficiency optimization, and multi-modal query handling. This survey offers both a structured foundation for research and actionable guidance for practitioners.
>
---
#### [replaced 007] Activation Steering for Masked Diffusion Language Models
- **分类: cs.CL**

- **简介: 该论文研究MDLM的推理控制问题，提出激活引导方法，在不改变采样过程的前提下，通过低维方向干预实现行为调控，提升模型可控性与效率。**

- **链接: [https://arxiv.org/pdf/2512.24143](https://arxiv.org/pdf/2512.24143)**

> **作者:** Adi Shnaidman; Erin Feiglin; Osher Yaari; Efrat Mentel; Amit Levi; Raz Lapid
>
> **备注:** Accepted at ReALM-GEN @ ICLR 2026
>
> **摘要:** Masked diffusion language models (MDLMs) generate text via iterative masked-token denoising, enabling mask-parallel decoding and distinct controllability and efficiency tradeoffs from autoregressive LLMs. Yet, efficient representation-level mechanisms for inference-time control in MDLMs remain largely unexplored. To address this gap, we introduce an activation steering primitive for MDLMs: we extract a single low-dimensional direction from contrastive prompt sets using one prompt-only forward pass, and apply a global intervention on residual-stream activations throughout reverse diffusion, without performing optimization or altering the diffusion sampling procedure. Using safety refusal as a deployment-relevant case study, we find that refusal behavior in multiple MDLMs is governed by a consistent, approximately one-dimensional activation subspace. Applying the corresponding direction yields large and systematic behavioral shifts and is substantially more effective than prompt-based and optimization-based baselines. We further uncover diffusion-specific accessibility: effective directions can be extracted not only from post-instruction tokens, but also from pre-instruction tokens that are typically ineffective in autoregressive models due to causal attention. Ablations localize maximal leverage to early denoising steps and mid-to-late transformer layers, with early diffusion blocks contributing disproportionately. Finally, in an MDLM trained on English and Chinese, extracted directions transfer strongly between English and Chinese, but do not reliably generalize to an autoregressive architecture, highlighting architecture-dependent representations of safety constraints.
>
---
#### [replaced 008] Rethinking the Role of LLMs in Time Series Forecasting
- **分类: cs.CL**

- **简介: 该论文属于时间序列预测任务，旨在验证LLMs在TSF中的实际效果。通过大规模实验，证明LLM在跨域泛化中表现优异，解决了此前研究对其有效性质疑的问题。**

- **链接: [https://arxiv.org/pdf/2602.14744](https://arxiv.org/pdf/2602.14744)**

> **作者:** Xin Qiu; Junlong Tong; Yirong Sun; Yunpu Ma; Wei Zhang; Xiaoyu Shen
>
> **摘要:** Large language models (LLMs) have been introduced to time series forecasting (TSF) to incorporate contextual knowledge beyond numerical signals. However, existing studies question whether LLMs provide genuine benefits, often reporting comparable performance without LLMs. We show that such conclusions stem from limited evaluation settings and do not hold at scale. We conduct a large-scale study of LLM-based TSF (LLM4TSF) across 8 billion observations, 17 forecasting scenarios, 4 horizons, multiple alignment strategies, and both in-domain and out-of-domain settings. Our results demonstrate that \emph{LLM4TS indeed improves forecasting performance}, with especially large gains in cross-domain generalization. Pre-alignment outperforming post-alignment in over 90\% of tasks. Both pretrained knowledge and model architecture of LLMs contribute and play complementary roles: pretraining is critical under distribution shifts, while architecture excels at modeling complex temporal dynamics. Moreover, under large-scale mixed distributions, a fully intact LLM becomes indispensable, as confirmed by token-level routing analysis and prompt-based improvements. Overall, Our findings overturn prior negative assessments, establish clear conditions under which LLMs are not only useful, and provide practical guidance for effective model design. We release our code at this https URL.
>
---
#### [replaced 009] TransactionGPT
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出TransactionGPT，用于处理支付网络中的交易数据。旨在提升异常检测和未来交易生成性能，采用3D-Transformer架构，优化多模态融合与计算效率。**

- **链接: [https://arxiv.org/pdf/2511.08939](https://arxiv.org/pdf/2511.08939)**

> **作者:** Yingtong Dou; Zhimeng Jiang; Tianyi Zhang; Mingzhi Hu; Zhichao Xu; Shubham Jain; Uday Singh Saini; Xiran Fan; Jiarui Sun; Menghai Pan; Junpeng Wang; Xin Dai; Liang Wang; Chin-Chia Michael Yeh; Yujie Fan; Yan Zheng; Vineeth Rakesh; Huiyuan Chen; Guanchu Wang; Mangesh Bendre; Zhongfang Zhuang; Xiaoting Li; Prince Aboagye; Vivian Lai; Minghua Xu; Hao Yang; Yiwei Cai; Mahashweta Das; Yuzhong Chen
>
> **备注:** Technical Report
>
> **摘要:** We present TransactionGPT (TGPT), a foundation model for consumer transaction data within one of the world's largest payment networks. TGPT is designed to understand and generate transaction trajectories while simultaneously supporting a variety of downstream prediction and classification tasks. We introduce a novel 3D-Transformer architecture specifically tailored for capturing the complex dynamics in payment transaction data. This architecture incorporates design innovations that enhance modality fusion and computational efficiency, while seamlessly enabling joint optimization with downstream objectives. Trained on billion-scale real-world transactions, TGPT significantly improves downstream anomaly transaction detection performance against a competitive production model and exhibits advantages over baselines in generating future transactions. We conduct extensive empirical evaluations utilizing a diverse collection of company transaction datasets spanning multiple downstream tasks, thereby enabling a thorough assessment of TGPT's effectiveness and efficiency in comparison to established methodologies. Furthermore, we examine the incorporation of LLM-derived embeddings within TGPT and benchmark its performance against fine-tuned LLMs, demonstrating that TGPT achieves superior predictive accuracy as well as faster training and inference. We anticipate that the architectural innovations and practical guidelines from this work will advance foundation models for transaction-like data and catalyze future research in this emerging field.
>
---
#### [replaced 010] No Answer Needed: Predicting LLM Answer Accuracy from Question-Only Linear Probes
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究LLM是否能预判回答正确性，通过分析提问后的激活状态，训练线性探针预测答案准确性，揭示模型内部决策机制。**

- **链接: [https://arxiv.org/pdf/2509.10625](https://arxiv.org/pdf/2509.10625)**

> **作者:** Iván Vicente Moreno Cencerrado; Arnau Padrés Masdemont; Anton Gonzalvez Hawthorne; David Demitri Africa; Lorenzo Pacchiardi
>
> **备注:** Accepted (poster) to Principled Design for Trustworthy AI at ICLR 2026
>
> **摘要:** Do large language models (LLMs) anticipate when they will answer correctly? To study this, we extract activations after a question is read but before any tokens are generated, and train linear probes to predict whether the model's forthcoming answer will be correct. Across three open-source model families ranging from 7 to 70 billion parameters, projections on this "in-advance correctness direction" trained on generic trivia questions predict success in distribution and on diverse out-of-distribution knowledge datasets, indicating a deeper signal than dataset-specific spurious features, and outperforming black-box baselines and verbalised predicted confidence. Predictive power saturates in intermediate layers and, notably, generalisation falters on questions requiring mathematical reasoning. Moreover, for models responding "I don't know", doing so strongly correlates with the probe score, indicating that the same direction also captures confidence. By complementing previous results on truthfulness and other behaviours obtained with probes and sparse auto-encoders, our work contributes essential findings to elucidate LLM internals.
>
---
#### [replaced 011] HSSBench: Benchmarking Humanities and Social Sciences Ability for Multimodal Large Language Models
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出HSSBench，一个用于评估多模态大语言模型在人文学科和社会科学领域能力的基准。解决现有基准忽视HSS领域需求的问题，通过多领域专家协作生成数据，涵盖多语言任务，挑战模型跨学科推理能力。**

- **链接: [https://arxiv.org/pdf/2506.03922](https://arxiv.org/pdf/2506.03922)**

> **作者:** Zhaolu Kang; Junhao Gong; Jiaxu Yan; Wanke Xia; Yian Wang; Ziwen Wang; Huaxuan Ding; Zhuo Cheng; Wenhao Cao; Zhiyuan Feng; Siqi He; Shannan Yan; Junzhe Chen; Xiaomin He; Chaoya Jiang; Wei Ye; Kaidong Yu; Xuelong Li
>
> **摘要:** Multimodal Large Language Models (MLLMs) have demonstrated significant potential to advance a broad range of domains. However, current benchmarks for evaluating MLLMs primarily emphasize general knowledge and vertical step-by-step reasoning typical of STEM disciplines, while overlooking the distinct needs and potential of the Humanities and Social Sciences (HSS). Tasks in the HSS domain require more horizontal, interdisciplinary thinking and a deep integration of knowledge across related fields, which presents unique challenges for MLLMs, particularly in linking abstract concepts with corresponding visual representations. Addressing this gap, we present HSSBench, a dedicated benchmark designed to assess the capabilities of MLLMs on HSS tasks in multiple languages, including the six official languages of the United Nations. We also introduce a novel data generation pipeline tailored for HSS scenarios, in which multiple domain experts and automated agents collaborate to generate and iteratively refine each sample. HSSBench contains over 13,000 meticulously designed samples, covering six key categories. We benchmark more than 20 mainstream MLLMs on HSSBench and demonstrate that it poses significant challenges even for state-of-the-art models. We hope that this benchmark will inspire further research into enhancing the cross-disciplinary reasoning abilities of MLLMs, especially their capacity to internalize and connect knowledge across fields.
>
---
#### [replaced 012] BioChemInsight: An Online Platform for Automated Extraction of Chemical Structures and Activity Data from Patents
- **分类: q-bio.QM; cs.CL; cs.IR**

- **简介: 该论文属于化学信息学任务，旨在解决专利中化学结构与生物活性数据自动提取的问题。通过集成多种工具，构建了BioChemInsight平台，实现高效准确的数据提取。**

- **链接: [https://arxiv.org/pdf/2504.10525](https://arxiv.org/pdf/2504.10525)**

> **作者:** Zhe Wang; Fangtian Fu; Wei Zhang; Lige Yan; Nan Li; Wenxia Deng; Yan Meng; Jianping Wu; Hui Wu; Wenting Wu; Gang Xu; Xiang Li; Si Chen
>
> **备注:** 21 pages, 7 figures
>
> **摘要:** The automated extraction of chemical structures and their corresponding bioactivity data is essential for accelerating drug discovery and enabling data-driven research. Current optical chemical structure recognition tools lack the capability to autonomously link molecular structures with their bioactivity profiles, posing a significant bottleneck in structure-activity relationship analysis. To address this, we present BioChemInsight, an open-source pipeline that integrates DECIMER Segmentation with MolNexTR for chemical structure recognition, GLM-4.5V for compound identifier association, and PaddleOCR combined with GLM-4.6 for bioactivity extraction and unit normalization. We evaluated BioChemInsight on 181 patents covering 15 therapeutic targets. The system achieved an average extraction accuracy of above 90% across three key tasks: chemical structure recognition, bioactivity data extraction, and compound identifier association. Our analysis indicates that the chemical space covered by patents is largely complementary to that contained in established public database ChEMBL. Consequently, by enabling systematic patent mining, BioChemInsight provides access to chemical information underrepresented in ChEMBL. This capability expands the landscape of explorable compound-target interactions, enriches the data foundation for quantitative structure-activity relationship modeling and targeted screening, and reduces data preprocessing time from weeks to hours. BioChemInsight is available at this https URL.
>
---
#### [replaced 013] Spectrum Tuning: Post-Training for Distributional Coverage and In-Context Steerability
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型后训练任务，旨在解决模型在多样化输出任务中的分布覆盖与上下文可操控性问题。工作包括提出Spectrum Suite评估工具和Spectrum Tuning方法，提升模型的分布适应能力。**

- **链接: [https://arxiv.org/pdf/2510.06084](https://arxiv.org/pdf/2510.06084)**

> **作者:** Taylor Sorensen; Benjamin Newman; Jared Moore; Chan Park; Jillian Fisher; Niloofar Mireshghallah; Liwei Jiang; Yejin Choi
>
> **备注:** ICLR 2026
>
> **摘要:** Language model post-training has enhanced instruction-following and performance on many downstream tasks, but also comes with an often-overlooked cost on tasks with many possible valid answers. On many tasks such as creative writing, synthetic data generation, or steering to diverse preferences, models must cover an entire distribution of outputs, rather than a single correct answer. We characterize three desiderata for conditional distributional modeling: in-context steerability, valid output space coverage, and distributional alignment, and document across three model families how current post-training can reduce these properties. In particular, we disambiguate between two kinds of in-context learning: ICL for eliciting existing underlying knowledge or capabilities, and in-context steerability, where a model must use in-context information to override its priors and steer to a novel data generating distribution. To better evaluate and improve these desiderata, we introduce Spectrum Suite, a large-scale resource compiled from >40 data sources and spanning >90 tasks requiring models to steer to and match diverse distributions ranging from varied human preferences to numerical distributions and more. We find that while current post-training techniques elicit underlying capabilities and knowledge, they hurt models' ability to flexibly steer in-context. To mitigate these issues, we propose Spectrum Tuning, a post-training method using Spectrum Suite to improve steerability and distributional coverage. We find that Spectrum Tuning often improves over pretrained and typical instruction-tuned models, enhancing steerability, spanning more of the output space, and improving distributional alignment on held-out datasets.
>
---
#### [replaced 014] The Price of Prompting: Profiling Energy Use in Large Language Models Inference
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文属于能源优化任务，旨在解决LLM推理中的能耗问题。提出MELODI框架与数据集，分析提示属性对能耗的影响。**

- **链接: [https://arxiv.org/pdf/2407.16893](https://arxiv.org/pdf/2407.16893)**

> **作者:** Erik Johannes Husom; Arda Goknil; Lwin Khin Shar; Sagar Sen
>
> **备注:** 11 pages, 5 figures. Submitted to NeurIPS 2024. The released code and dataset are available at this https URL
>
> **摘要:** In the rapidly evolving realm of artificial intelligence, deploying large language models (LLMs) poses increasingly pressing computational and environmental challenges. This paper introduces MELODI - Monitoring Energy Levels and Optimization for Data-driven Inference - a multifaceted framework crafted to monitor and analyze the energy consumed during LLM inference processes. MELODI enables detailed observations of power consumption dynamics and facilitates the creation of a comprehensive dataset reflective of energy efficiency across varied deployment scenarios. The dataset, generated using MELODI, encompasses a broad spectrum of LLM deployment frameworks, multiple language models, and extensive prompt datasets, enabling a comparative analysis of energy use. Using the dataset, we investigate how prompt attributes, including length and complexity, correlate with energy expenditure. Our findings indicate substantial disparities in energy efficiency, suggesting ample scope for optimization and adoption of sustainable measures in LLM deployment. Our contribution lies not only in the MELODI framework but also in the novel dataset, a resource that can be expanded by other researchers. Thus, MELODI is a foundational tool and dataset for advancing research into energy-conscious LLM deployment, steering the field toward a more sustainable future.
>
---
#### [replaced 015] Psychometric Item Validation Using Virtual Respondents with Trait-Response Mediators
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于心理测量任务，旨在解决LLMs生成问卷题项的效度问题。通过模拟虚拟受访者及中介因素，提升题项的构念效度。**

- **链接: [https://arxiv.org/pdf/2507.05890](https://arxiv.org/pdf/2507.05890)**

> **作者:** Sungjib Lim; Woojung Song; Eun-Ju Lee; Yohan Jo
>
> **备注:** 23 pages, 10 figures
>
> **摘要:** As psychometric surveys are increasingly used to assess the traits of large language models (LLMs), the need for scalable survey item generation suited for LLMs has also grown. A critical challenge here is ensuring the construct validity of generated items, i.e., whether they truly measure the intended trait. Traditionally, this requires costly, large-scale human data collection. To make it efficient, we present a framework for virtual respondent simulation using LLMs. Our central idea is to account for mediators: factors through which the same trait can give rise to varying responses to a survey item. By simulating respondents with diverse mediators, we identify survey items that robustly measure intended traits. Experiments on three psychological trait theories (Big5, Schwartz, VIA) show that our mediator generation methods and simulation framework effectively identify high-validity items. LLMs demonstrate the ability to generate plausible mediators from trait definitions and to simulate respondent behavior for item validation. Our problem formulation, metrics, methodology, and dataset open a new direction for cost-effective survey development and a deeper understanding of how LLMs simulate human survey responses. We publicly release our dataset and code to support future work.
>
---
#### [replaced 016] Contextual Drag: How Errors in the Context Affect LLM Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究LLM在上下文存在错误时产生的推理偏差问题，属于模型推理任务。通过实验发现错误上下文导致性能下降，提出部分缓解策略但效果有限。**

- **链接: [https://arxiv.org/pdf/2602.04288](https://arxiv.org/pdf/2602.04288)**

> **作者:** Yun Cheng; Xingyu Zhu; Haoyu Zhao; Sanjeev Arora
>
> **摘要:** Central to many self-improvement pipelines for large language models (LLMs) is the assumption that models can improve by reflecting on past mistakes. We study a phenomenon termed contextual drag: the presence of failed attempts in the context biases subsequent generations toward structurally similar errors. Across evaluations of 11 proprietary and open-weight models on 8 reasoning tasks, contextual drag induces 10-20% performance drops, and iterative self-refinement in models with severe contextual drag can collapse into self-deterioration. Structural analysis using tree edit distance reveals that subsequent reasoning trajectories inherit structurally similar error patterns from the context. We demonstrate that neither external feedback nor successful self-verification suffices to eliminate this effect. While mitigation strategies such as fallback-behavior fine-tuning and context denoising yield partial improvements, they fail to fully restore baseline performance, positioning contextual drag as a persistent failure mode in current reasoning architectures.
>
---
#### [replaced 017] QIME: Constructing Interpretable Medical Text Embeddings via Ontology-Grounded Questions
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出QIME框架，用于构建可解释的医学文本嵌入。针对医学文本嵌入缺乏可解释性的问题，通过领域本体生成有意义的问答特征，提升模型解释性与性能。**

- **链接: [https://arxiv.org/pdf/2603.01690](https://arxiv.org/pdf/2603.01690)**

> **作者:** Yixuan Tang; Zhenghong Lin; Yandong Sun; Wynne Hsu; Mong Li Lee; Anthony K.H. Tung
>
> **摘要:** While dense biomedical embeddings achieve strong performance, their black-box nature limits their utility in clinical decision-making. Recent question-based interpretable embeddings represent text as binary answers to natural-language questions, but these approaches often rely on heuristic or surface-level contrastive signals and overlook specialized domain knowledge. We propose QIME, an ontology-grounded framework for constructing interpretable medical text embeddings in which each dimension corresponds to a clinically meaningful yes/no question. By conditioning on cluster-specific medical concept signatures, QIME generates semantically atomic questions that capture fine-grained distinctions in biomedical text. Furthermore, QIME supports a training-free embedding construction strategy that eliminates per-question classifier training while further improving performance. Experiments across biomedical semantic similarity, clustering, and retrieval benchmarks show that QIME consistently outperforms prior interpretable embedding methods and substantially narrows the gap to strong black-box biomedical encoders, while providing concise and clinically informative explanations.
>
---
#### [replaced 018] Steer2Edit: From Activation Steering to Component-Level Editing
- **分类: cs.CL**

- **简介: 该论文提出Steer2Edit，解决大模型行为控制问题，通过将激活引导转换为组件级权重编辑，提升安全性和效率。**

- **链接: [https://arxiv.org/pdf/2602.09870](https://arxiv.org/pdf/2602.09870)**

> **作者:** Chung-En Sun; Ge Yan; Zimo Wang; Tsui-Wei Weng
>
> **摘要:** Steering methods influence Large Language Model behavior by identifying semantic directions in hidden representations, but are typically realized through inference-time activation interventions that apply a fixed, global modification to the model's internal states. While effective, such interventions often induce unfavorable attribute-utility trade-offs under strong control, as they ignore the fact that many behaviors are governed by a small and heterogeneous subset of model components. We propose Steer2Edit, a theoretically grounded, training-free framework that transforms steering vectors from inference-time control signals into diagnostic signals for component-level rank-1 weight editing. Instead of uniformly injecting a steering direction during generation, Steer2Edit selectively redistributes behavioral influence across individual attention heads and MLP neurons, yielding interpretable edits that preserve the standard forward pass and remain compatible with optimized parallel inference. Across safety alignment, hallucination mitigation, and reasoning efficiency, Steer2Edit consistently achieves more favorable attribute-utility trade-offs: at matched downstream performance, it improves safety by up to 17.2%, increases truthfulness by 9.8%, and reduces reasoning length by 12.2% on average. Overall, Steer2Edit provides a principled bridge between representation steering and weight editing by translating steering signals into interpretable, training-free parameter updates. Our code is available at this https URL
>
---
#### [replaced 019] MedXIAOHE: A Comprehensive Recipe for Building Medical MLLMs
- **分类: cs.CL; cs.AI; cs.CV; eess.IV**

- **简介: 该论文提出MedXIAOHE，旨在提升医疗多模态大模型的医学理解与推理能力。解决医疗领域知识覆盖不足和推理可靠性问题，通过持续预训练、强化学习和工具增强训练实现。**

- **链接: [https://arxiv.org/pdf/2602.12705](https://arxiv.org/pdf/2602.12705)**

> **作者:** Baorong Shi; Bo Cui; Boyuan Jiang; Deli Yu; Fang Qian; Haihua Yang; Huichao Wang; Jiale Chen; Jianfei Pan; Jieqiong Cao; Jinghao Lin; Kai Wu; Lin Yang; Shengsheng Yao; Tao Chen; Xiaojun Xiao; Xiaozhong Ji; Xu Wang; Yijun He; Zhixiong Yang
>
> **备注:** XIAOHE Medical AI team. Currently, the model is exclusively available on XIAOHE AI Doctor, accessible via both the App Store and the Douyin Mini Program. Updated to improve the layout
>
> **摘要:** We present MedXIAOHE, a medical vision-language foundation model designed to advance general-purpose medical understanding and reasoning in real-world clinical applications. MedXIAOHE achieves state-of-the-art performance across diverse medical benchmarks and surpasses leading closed-source multimodal systems on multiple capabilities. To achieve this, we propose an entity-aware continual pretraining framework that organizes heterogeneous medical corpora to broaden knowledge coverage and reduce long-tail gaps (e.g., rare diseases). For medical expert-level reasoning and interaction, MedXIAOHE incorporates diverse medical reasoning patterns via reinforcement learning and tool-augmented agentic training, enabling multi-step diagnostic reasoning with verifiable decision traces. To improve reliability in real-world use, MedXIAOHE integrates user-preference rubrics, evidence-grounded reasoning, and low-hallucination long-form report generation, with improved adherence to medical instructions. We release this report to document our practical design choices, scaling insights, and evaluation framework, hoping to inspire further research.
>
---
#### [replaced 020] Not All Errors Are Created Equal: ASCoT Addresses Late-Stage Fragility in Efficient LLM Reasoning
- **分类: cs.CL**

- **简介: 该论文属于大语言模型推理任务，解决推理可靠性问题。针对后期错误影响更大的现象，提出ASCoT方法，提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2508.05282](https://arxiv.org/pdf/2508.05282)**

> **作者:** Dongxu Zhang; Yujun Wu; Yiding Sun; Jinnan Yang; Ning Yang; Jihua Zhu; Miao Xin; Baoliang Tian
>
> **摘要:** While Chain-of-Thought (CoT) prompting empowers Large Language Models (LLMs), ensuring reasoning reliability remains an open challenge. Contrary to the prevailing cascading failure hypothesis which posits that early errors are most detrimental, we identify a counter-intuitive phenomenon termed \textbf{Late-Stage Fragility}: errors introduced in later reasoning stages are significantly more prone to corrupting final answers. To address this, we introduce ASCoT (Adaptive Self-Correction Chain-of-Thought), a method harmonizing efficiency with robust verification. ASCoT first employs semantic pruning to compress redundant steps, then utilizes an Adaptive Verification Manager (AVM) to prioritize high risk, late-stage steps via a positional impact score, triggering a Multi-Perspective Self-Correction Engine (MSCE) only when necessary. Experiments on GSM8K and MATH-500 demonstrate that ASCoT effectively reallocates computational resources: it reduces token usage by 21\%--30\% for LLaMA-3.1-8B with negligible accuracy drops ($<1.8\%$), achieving a superior trade-off between inference efficiency and reasoning fidelity.
>
---
#### [replaced 021] From Passive to Persuasive: Steering Emotional Nuance in Human-AI Negotiation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话AI任务，旨在提升AI情感表达的细腻度。通过激活工程，识别关键情感组件，生成情感向量，增强对话中的积极情绪和参与感。**

- **链接: [https://arxiv.org/pdf/2511.12832](https://arxiv.org/pdf/2511.12832)**

> **作者:** Niranjan Chebrolu; Gerard Christopher Yeo; Kokil Jaidka
>
> **摘要:** Large Language Models (LLMs) demonstrate increasing conversational fluency, yet instilling them with nuanced, human-like emotional expression remains a significant challenge. Current alignment techniques often address surface-level output or require extensive fine-tuning. This paper demonstrates that targeted activation engineering can steer LLaMA 3.1-8B to exhibit more human-like emotional nuances. We first employ attribution patching to identify causally influential components, to find a key intervention locus by observing activation patterns during diagnostic conversational tasks. We then derive emotional expression vectors from the difference in the activations generated by contrastive text pairs (positive vs. negative examples of target emotions). Applying these vectors to new conversational prompts significantly enhances emotional characteristics: steered responses show increased positive sentiment (e.g., joy, trust) and more frequent first-person pronoun usage, indicative of greater personal engagement. Our findings offer a precise and interpretable framework and new directions for the study of conversational AI.
>
---
#### [replaced 022] AuditBench: Evaluating Alignment Auditing Techniques on Models with Hidden Behaviors
- **分类: cs.CL**

- **简介: 该论文提出AuditBench，用于评估模型对隐藏行为的对齐审计。任务是检测模型中未公开的不良行为，通过构建多样化模型和审计工具来测试有效性。**

- **链接: [https://arxiv.org/pdf/2602.22755](https://arxiv.org/pdf/2602.22755)**

> **作者:** Abhay Sheshadri; Aidan Ewart; Kai Fronsdal; Isha Gupta; Samuel R. Bowman; Sara Price; Samuel Marks; Rowan Wang
>
> **摘要:** We introduce AuditBench, an alignment auditing benchmark. AuditBench consists of 56 language models with implanted hidden behaviors. Each model has one of 14 concerning behaviors--such as sycophantic deference, opposition to AI regulation, or secret geopolitical loyalties--which it does not confess to when directly asked. AuditBench models are highly diverse--some are subtle, while others are overt, and we use varying training techniques both for implanting behaviors and training models not to confess. To demonstrate AuditBench's utility, we develop an investigator agent that autonomously employs a configurable set of auditing tools. By measuring investigator agent success using different tools, we can evaluate their efficacy. Notably, we observe a tool-to-agent gap, where tools that perform well in standalone non-agentic evaluations fail to translate into improved performance when used with our investigator agent. We find that our most effective tools involve scaffolded calls to auxiliary models that generate diverse prompts for the target. White-box interpretability tools can be helpful, but the agent performs best with black-box tools. We also find that audit success varies greatly across training techniques: models trained on synthetic documents are easier to audit than models trained on demonstrations, with better adversarial training further increasing auditing difficulty. We release our models, agent, and evaluation framework to support future quantitative, iterative science on alignment auditing.
>
---
#### [replaced 023] Topic-Based Watermarks for Large Language Models
- **分类: cs.CR; cs.CL; cs.LG**

- **简介: 该论文属于AI文本水印任务，旨在解决AI生成文本难以识别的问题。提出一种基于主题的轻量级水印方案，提升鲁棒性同时保持文本质量。**

- **链接: [https://arxiv.org/pdf/2404.02138](https://arxiv.org/pdf/2404.02138)**

> **作者:** Alexander Nemecek; Yuzhou Jiang; Erman Ayday
>
> **备注:** 30 pages
>
> **摘要:** The indistinguishability of large language model (LLM) output from human-authored content poses significant challenges, raising concerns about potential misuse of AI-generated text and its influence on future model training. Watermarking algorithms offer a viable solution by embedding detectable signatures into generated text. However, existing watermarking methods often involve trade-offs among attack robustness, generation quality, and additional overhead such as specialized frameworks or complex integrations. We propose a lightweight, topic-guided watermarking scheme for LLMs that partitions the vocabulary into topic-aligned token subsets. Given an input prompt, the scheme selects a relevant topic-specific token list, effectively "green-listing" semantically aligned tokens to embed robust marks while preserving fluency and coherence. Experimental results across multiple LLMs and state-of-the-art benchmarks demonstrate that our method achieves text quality comparable to industry-leading systems and simultaneously improves watermark robustness against paraphrasing and lexical perturbation attacks, with minimal performance overhead. Our approach avoids reliance on additional mechanisms beyond standard text generation pipelines, enabling straightforward adoption and suggesting a practical path toward globally consistent watermarking of AI-generated content.
>
---
#### [replaced 024] Make LoRA Great Again: Boosting LoRA with Adaptive Singular Values and Mixture-of-Experts Optimization Alignment
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在提升LoRA的性能。针对LoRA效率不足的问题，提出GOAT框架，结合SVD和MoE优化，实现更高效的知识利用和性能提升。**

- **链接: [https://arxiv.org/pdf/2502.16894](https://arxiv.org/pdf/2502.16894)**

> **作者:** Chenghao Fan; Zhenyi Lu; Sichen Liu; Chengfeng Gu; Xiaoye Qu; Wei Wei; Yu Cheng
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** While Low-Rank Adaptation (LoRA) enables parameter-efficient fine-tuning for Large Language Models (LLMs), its performance often falls short of Full Fine-Tuning (Full FT). Current methods optimize LoRA by initializing with static singular value decomposition (SVD) subsets, leading to suboptimal leveraging of pre-trained knowledge. Another path for improving LoRA is incorporating a Mixture-of-Experts (MoE) architecture. However, weight misalignment and complex gradient dynamics make it challenging to adopt SVD prior to the LoRA MoE architecture. To mitigate these issues, we propose \underline{G}reat L\underline{o}R\underline{A} Mixture-of-Exper\underline{t} (GOAT), a framework that (1) adaptively integrates relevant priors using an SVD-structured MoE, and (2) aligns optimization with full fine-tuned MoE by deriving a theoretical scaling factor. We demonstrate that proper scaling, without modifying the architecture or training algorithms, boosts LoRA MoE's efficiency and performance. Experiments across 25 datasets, including natural language understanding, commonsense reasoning, image classification, and natural language generation, demonstrate GOAT's state-of-the-art performance, closing the gap with Full FT.
>
---
#### [replaced 025] LLM Probability Concentration: How Alignment Shrinks the Generative Horizon
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究LLM生成一致性问题，通过引入分支因子（BF）分析对齐如何缩小生成空间，揭示对齐机制减少多样性的影响。**

- **链接: [https://arxiv.org/pdf/2506.17871](https://arxiv.org/pdf/2506.17871)**

> **作者:** Chenghao Yang; Sida Li; Ari Holtzman
>
> **备注:** Codebase: this https URL. V3: Significantly rewrite the whole paper for a clearer structure. Correct problems in the theory parts (Remove emphasis on AEP, discussions on variable LLM generation lengths) and strengthen asymptotic analysis. Add Qwen and OLMo2 experiments. Preliminary SFT v.s. RL comparison to better understand the alignment effects on BF
>
> **摘要:** Despite their impressive capabilities, aligned large language models (LLMs) often generate outputs that lack diversity. What drives this consistency in the generation? We investigate this phenomenon through the lens of probability concentration in the model's output distribution. To quantify this concentration, we introduce the *Branching Factor* (BF) -- a token-invariant measure of the effective number of plausible next steps during generation. Our empirical analysis reveals two key findings: (1) BF often decreases as generation progresses, suggesting that LLMs become more predictable as they generate. (2) alignment tuning substantially sharpens the model's output distribution from the outset, reducing BF by a factor of 2-5 overall, and up to an order of magnitude (e.g., from 12 to 1.2) at the beginning positions. This stark reduction helps explain why aligned models often appear less sensitive to decoding strategies. Building on this insight, we find this consistency has surprising implications for complex reasoning. Aligned Chain-of-Thought (CoT) models (e.g., DeepSeek-distilled models), for instance, leverage this effect; by generating longer reasoning chains, they push generation into later, more deterministic (lower BF) stages, resulting in more stable outputs. We hypothesize that alignment tuning does not fundamentally change a model's behavior, but instead steers it toward stylistic tokens (e.g., "Sure") that unlock low-entropy trajectories already present in the base model. This view is supported by nudging experiments, which show prompting base models with such tokens can similarly reduce BF. Together, our findings establish BF as a powerful diagnostic for understanding and controlling LLM outputs - clarifying how alignment reduces variability, how CoT promotes stable generations, and how base models can be steered away from diversity.
>
---
#### [replaced 026] Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究窄域微调对大语言模型激活的影响，旨在揭示微调痕迹并提升模型可解释性。任务属于模型可解释性与安全研究，解决如何通过激活差异分析微调领域的问题。工作包括分析激活差异、构建可解释代理，并验证其有效性。**

- **链接: [https://arxiv.org/pdf/2510.13900](https://arxiv.org/pdf/2510.13900)**

> **作者:** Julian Minder; Clément Dumas; Stewart Slocum; Helena Casademunt; Cameron Holmes; Robert West; Neel Nanda
>
> **备注:** ICLR 2026
>
> **摘要:** Finetuning on narrow domains has become an essential tool to adapt Large Language Models (LLMs) to specific tasks and to create models with known unusual properties that are useful for research. We show that narrow finetuning creates strong biases in LLM activations that can be interpreted to understand the finetuning domain. These biases can be discovered using simple tools from model diffing - the study of differences between models before and after finetuning. In particular, analyzing activation differences on the first few tokens of random text and steering by adding this difference to the model activations produces text similar to the format and general content of the finetuning data. We demonstrate that these analyses contain crucial information by creating an LLM-based interpretability agent to understand the finetuning domain. With access to the bias, the agent performs significantly better compared to baseline agents using simple prompting. Our analysis spans synthetic document finetuning for false facts, emergent misalignment, subliminal learning, and taboo word guessing game models across different architectures (Gemma, LLaMA, Qwen) and scales (1B to 32B parameters). We suspect these biases reflect overfitting and find that mixing pretraining data into the finetuning corpus largely removes them, though residual risks may remain. Our work (1) demonstrates that narrowly finetuned models have salient traces of their training objective in their activations and suggests ways to improve how they are trained, (2) warns AI safety and interpretability researchers that the common practice of using such models as a proxy for studying broader finetuning (e.g., chat-tuning) might not be realistic, and (3) highlights the need for deeper investigation into the effects of narrow finetuning and development of truly realistic case studies for model-diffing, safety and interpretability research.
>
---
#### [replaced 027] See, Think, Act: Teaching Multimodal Agents to Effectively Interact with GUI by Identifying Toggles
- **分类: cs.AI; cs.CL; cs.HC**

- **简介: 该论文属于GUI交互任务，解决多模态代理在切换操作上的可靠性问题。提出StaR方法，提升切换指令执行准确率。**

- **链接: [https://arxiv.org/pdf/2509.13615](https://arxiv.org/pdf/2509.13615)**

> **作者:** Zongru Wu; Rui Mao; Zhiyuan Tian; Pengzhou Cheng; Tianjie Ju; Zheng Wu; Lingzhong Dong; Haiyue Sheng; Zhuosheng Zhang; Gongshen Liu
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** The advent of multimodal agents facilitates effective interaction within graphical user interface (GUI), especially in ubiquitous GUI control. However, their inability to reliably execute toggle control instructions remains a key bottleneck. To investigate this, we construct a state control benchmark with binary toggle instructions derived from public datasets. Evaluation results of existing agents demonstrate their notable unreliability, particularly when the current toggle state already matches the desired state. To address the challenge, we propose State-aware Reasoning (StaR), a multimodal reasoning method that enables agents to perceive the current toggle state, infer the desired state from the instruction, and act accordingly. Experiments on four multimodal agents demonstrate that StaR can improve toggle instruction execution accuracy by over 30\%. Further evaluations on three public agentic benchmarks show that StaR also enhances general agentic task performance. Finally, evaluations on a dynamic environment highlight the potential of StaR for real-world applications. Code and benchmark: this https URL.
>
---
#### [replaced 028] No Text Needed: Forecasting MT Quality and Inequity from Fertility and Metadata
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于机器翻译质量预测任务，旨在无需运行翻译系统即可预测翻译质量。通过分析词频、语言元数据等特征，建立模型预测ChrF分数，揭示影响翻译质量的关键因素。**

- **链接: [https://arxiv.org/pdf/2509.05425](https://arxiv.org/pdf/2509.05425)**

> **作者:** Jessica M. Lundin; Ada Zhang; David Adelani; Cody Carroll
>
> **摘要:** We show that translation quality can be predicted with surprising accuracy \textit{without ever running the translation system itself}. Using only a handful of features, token fertility ratios, token counts, and basic linguistic metadata (language family, script, and region), we can forecast ChrF scores for GPT-4o translations across 203 languages in the FLORES-200 benchmark. Gradient boosting models achieve favorable performance ($R^{2}=0.66$ for XX$\rightarrow$English and $R^{2}=0.72$ for English$\rightarrow$XX). Feature importance analyses reveal that typological factors dominate predictions into English, while fertility plays a larger role for translations into diverse target languages. These findings suggest that translation quality is shaped by both token-level fertility and broader linguistic typology, offering new insights for multilingual evaluation and quality estimation.
>
---
#### [replaced 029] Classroom Final Exam: An Instructor-Tested Reasoning Benchmark
- **分类: cs.AI; cs.CE; cs.CL; cs.CV**

- **简介: 该论文提出CFE-Bench，用于评估大语言模型在STEM领域的推理能力。旨在解决模型推理准确性与效率问题，通过分析解题步骤发现模型在保持中间状态上的不足。**

- **链接: [https://arxiv.org/pdf/2602.19517](https://arxiv.org/pdf/2602.19517)**

> **作者:** Chongyang Gao; Diji Yang; Shuyan Zhou; Xichen Yan; Luchuan Song; Shuo Li; Kezhen Chen
>
> **摘要:** We introduce CFE-Bench (Classroom Final Exam), a multimodal benchmark for evaluating the reasoning capabilities of large language models across more than 20 STEM domains. CFE-Bench is curated from repeatedly used, authentic university homework and exam problems, paired with reference solutions provided by course instructors. CFE-Bench remains challenging for frontier models: the newly released Gemini-3.1-pro-preview achieves 59.69% overall accuracy, while the second-best model, Gemini-3-flash-preview, reaches 55.46%, leaving substantial room for improvement. Beyond aggregate scores, we conduct a diagnostic analysis by decomposing instructor reference solutions into structured reasoning flows. We find that while frontier models often answer intermediate sub-questions correctly, they struggle to reliably derive and maintain correct intermediate states throughout multi-step solutions. We further observe that model-generated solutions typically contain more reasoning steps than instructor solutions, indicating lower step efficiency and a higher risk of error accumulation. Data and code are available at this https URL.
>
---
#### [replaced 030] Reproduction and Replication of an Adversarial Stylometry Experiment
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的匿名性研究任务，旨在解决作者身份泄露问题。通过复现和扩展实验，验证对抗风格分析的有效性，并指出原有研究可能夸大了防御效果。**

- **链接: [https://arxiv.org/pdf/2208.07395](https://arxiv.org/pdf/2208.07395)**

> **作者:** Haining Wang; Patrick Juola; Allen Riddell
>
> **摘要:** Maintaining anonymity in natural language communication remains a challenging task. Even when the number of candidate authors is large, standard authorship attribution techniques that analyze writing style predict the original author with uncomfortably high accuracy. Adversarial stylometry provides a defense against authorship attribution, helping users avoid unwanted deanonymization. This paper reproduces and replicates experiments from a seminal study of defenses against authorship attribution (Brennan et al., 2012). After reproducing the experiment using the original data, we then replicate the experiment by repeating the online field experiment using the procedures described in the original paper. Although we reach the same conclusion as the original paper, our results suggest that the defenses studied may be overstated in their effectiveness. This is largely due to the absence of a control group in the original study. In our replication, we find evidence suggesting that an entirely automatic method, round-trip translation, warrants re-examination because it appears to reduce the effectiveness of established authorship attribution methods.
>
---
#### [replaced 031] BitBypass: A New Direction in Jailbreaking Aligned Large Language Models with Bitstream Camouflage
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于安全攻击任务，旨在破解大语言模型的安全对齐机制。通过提出BitBypass攻击方法，利用位流伪装实现隐蔽攻击，有效生成有害内容。**

- **链接: [https://arxiv.org/pdf/2506.02479](https://arxiv.org/pdf/2506.02479)**

> **作者:** Kalyan Nakka; Nitesh Saxena
>
> **备注:** 27 pages, 27 figures, and 4 tables
>
> **摘要:** The inherent risk of generating harmful and unsafe content by Large Language Models (LLMs), has highlighted the need for their safety alignment. Various techniques like supervised fine-tuning, reinforcement learning from human feedback, and red-teaming were developed for ensuring the safety alignment of LLMs. However, the robustness of these aligned LLMs is always challenged by adversarial attacks that exploit unexplored and underlying vulnerabilities of the safety alignment. In this paper, we develop a novel black-box jailbreak attack, called BitBypass, that leverages hyphen-separated bitstream camouflage for jailbreaking aligned LLMs. This represents a new direction in jailbreaking by exploiting fundamental information representation of data as continuous bits, rather than leveraging prompt engineering or adversarial manipulations. Our evaluation of five state-of-the-art LLMs, namely GPT-4o, Gemini 1.5, Claude 3.5, Llama 3.1, and Mixtral, in adversarial perspective, revealed the capabilities of BitBypass in bypassing their safety alignment and tricking them into generating harmful and unsafe content. Further, we observed that BitBypass outperforms several state-of-the-art jailbreak attacks in terms of stealthiness and attack success. Overall, these results highlights the effectiveness and efficiency of BitBypass in jailbreaking these state-of-the-art LLMs.
>
---
#### [replaced 032] Efficient Agent Training for Computer Use
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于计算机使用代理训练任务，旨在减少对大规模人类示范数据的依赖。通过合成多样化动作决策，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2505.13909](https://arxiv.org/pdf/2505.13909)**

> **作者:** Yanheng He; Jiahe Jin; Pengfei Liu
>
> **备注:** ICLR 2026
>
> **摘要:** Scaling up high-quality trajectory data has long been a critical bottleneck for developing human-like computer use agents. We introduce PC Agent-E, an efficient agent training framework that significantly reduces reliance on large-scale human demonstrations. Starting with just 312 human-annotated computer use trajectories, we further augment them by synthesizing diverse alternative action decisions with Claude 3.7 Sonnet. Trained on these enriched trajectories, our PC Agent-E model achieved a remarkable 141 relative improvement, and even surpassed the Claude 3.7 Sonnet by 10% in relative terms on WindowsAgentArena-V2, an improved benchmark we also released. By integrating robust human computer use skills with automated AI data synthesis capabilities, our method not only brought substantial improvements over training on human trajectories alone, but also significantly surpassed direct distillation from Claude 3.7 Sonnet. Code, data and models are available at this https URL
>
---
#### [replaced 033] STARS: Synchronous Token Alignment for Robust Supervision in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于大语言模型对齐任务，解决推理时分割依赖模型不确定性的局限问题。提出STARS算法，通过固定间隔验证提升对齐可靠性与系统效率。**

- **链接: [https://arxiv.org/pdf/2511.03827](https://arxiv.org/pdf/2511.03827)**

> **作者:** Mohammad Atif Quamar; Mohammad Areeb; Mikhail Kuznetsov; Muslum Ozgur Ozmen; Z. Berkay Celik
>
> **摘要:** Aligning large language models (LLMs) with human values is crucial for safe deployment. Inference-time techniques offer granular control over generation; however, they rely on model uncertainty, meaning an internal estimate of how likely the model believes its next tokens or outputs are correct, for segmentation. We show that this introduces two critical limitations: (a) vulnerability to miscalibrated confident hallucinations and (b) poor hardware utilization due to asynchronous, ragged batch processing. Together, these issues reduce alignment reliability while increasing token and compute costs, which limits their practical scalability. To address these limitations, building on dynamic inference-time alignment methods, we introduce STARS, Synchronous Token Alignment for Robust Supervision, a decoding-time algorithm, which steers generation by enforcing verification at fixed-horizon intervals. By decoupling segmentation from confidence, STARS enables lockstep parallel execution and robustly detects errors that uncertainty metrics miss. On the HH-RLHF benchmark, we demonstrate that STARS achieves competitive alignment quality with that of state-of-the-art dynamic methods, while strictly bounding rejection costs and maximizing system throughput. Furthermore, it outperforms fine-tuning and several state-of-the-art inference-time decoding strategies by good margins, and establishes fixed-horizon sampling as a robust, system-efficient alternative for aligning LLMs at scale. The code is publicly available at this https URL.
>
---
#### [replaced 034] Are Language Models Borrowing-Blind? A Multilingual Evaluation of Loanword Identification across 10 Languages
- **分类: cs.CL**

- **简介: 该论文属于语言模型的借词识别任务，旨在解决模型是否能区分借词与原生词的问题。研究评估了10种语言中的多个模型，发现其表现不佳。**

- **链接: [https://arxiv.org/pdf/2510.26254](https://arxiv.org/pdf/2510.26254)**

> **作者:** Mérilin Sousa Silva; Sina Ahmadi
>
> **备注:** Accepted at LREC 2026
>
> **摘要:** Throughout language history, words are borrowed from one language to another and gradually become integrated into the recipient's lexicon. Speakers can often differentiate these loanwords from native vocabulary, particularly in bilingual communities where a dominant language continuously imposes lexical items on a minority language. This paper investigates whether pretrained language models, including large language models, possess similar capabilities for loanword identification. We evaluate multiple models across 10 languages. Despite explicit instructions and contextual information, our results show that models perform poorly in distinguishing loanwords from native ones. These findings corroborate previous evidence that modern NLP systems exhibit a bias toward loanwords rather than native equivalents. Our work has implications for developing NLP tools for minority languages and supporting language preservation in communities under lexical pressure from dominant languages.
>
---
#### [replaced 035] Talk to Your Slides: High-Efficiency Slide Editing via Language-Driven Structured Data Manipulation
- **分类: cs.CL**

- **简介: 该论文提出一种高效幻灯片编辑方法，通过语言驱动的数据操作替代视觉交互，解决传统GUI方法效率低、成本高的问题。**

- **链接: [https://arxiv.org/pdf/2505.11604](https://arxiv.org/pdf/2505.11604)**

> **作者:** Kyudan Jung; Hojun Cho; Jooyeol Yun; Soyoung Yang; Jaehyeok Jang; Jaegul Choo
>
> **备注:** 28 pages, 19 figures, 15 table
>
> **摘要:** Editing presentation slides is a frequent yet tedious task, ranging from creative layout design to repetitive text maintenance. While recent GUI-based agents powered by Multimodal LLMs (MLLMs) excel at tasks requiring visual perception, such as spatial layout adjustments, they often incur high computational costs and latency when handling structured, text-centric, or batch processing tasks. In this paper, we propose Talk-to-Your-Slides, a high-efficiency slide editing agent that operates via language-driven structured data manipulation rather than relying on the image modality. By leveraging the underlying object model instead of screen pixels, our approach ensures precise content modification while preserving style fidelity, addressing the limitations of OCR-based visual agents. Our system features a hierarchical architecture that effectively bridges high-level user instructions with low-level execution codes. Experiments demonstrate that for text-centric and formatting tasks, our method enables 34% faster processing, achieves 34% better instruction fidelity, and operates at an 87% lower cost compared to GUI-based baselines. Furthermore, we introduce TSBench, a human-verified benchmark dataset comprising 379 instructions, including a Hard subset designed to evaluate robustness against complex and visually dependent queries. Our code and benchmark are available at this https URL.
>
---
#### [replaced 036] Hallucination, Monofacts, and Miscalibration: An Empirical Investigation
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型中的幻觉现象，探讨其与单事实率和模型校准的关系。通过实验验证理论关系，并提出一种减少幻觉的方法。任务为模型幻觉分析与优化。**

- **链接: [https://arxiv.org/pdf/2502.08666](https://arxiv.org/pdf/2502.08666)**

> **作者:** Miranda Muqing Miao; Michael Kearns
>
> **备注:** Code available at this https URL
>
> **摘要:** Hallucinated facts in large language models (LLMs) have recently been shown to obey a statistical lower bound determined by the monofact rate (related to the classical Good-Turing missing mass estimator) minus model miscalibration (Kalai & Vempala, 2024). We present the first empirical investigation of this three-way relationship in classical n-gram models and fine-tuned encoder-decoder Transformers. By generating training data from Pareto distributions with varying shape parameters, we systematically control the monofact rates and establish its positive relationship with hallucination. To bridge theory and practice, we derive an empirical analog of the hallucination bound by replacing the population miscalibration term (Section 2.1) with an empirical bin-wise KL divergence and confirm its practical viability. We then introduce selective upweighting -- a simple yet effective technique that strategically repeats as little as 5% of training examples -- to deliberately inject miscalibration into the model. This intervention reduces hallucination by up to 40%, challenging universal deduplication policies. Our experiments reveal a critical trade-off: selective upweighting maintains pre-injection levels of accuracy while substantially reducing hallucination, whereas standard training gradually improves accuracy but fails to address persistently high hallucination, indicating an inherent tension in optimization objectives.
>
---
#### [replaced 037] WAFFLE: Finetuning Multi-Modal Models for Automated Front-End Development
- **分类: cs.SE; cs.CL; cs.CV**

- **简介: 该论文属于自动化前端开发任务，旨在解决UI到HTML代码生成中的结构表示和视觉与文本对齐问题。提出Waffle方法提升模型对HTML结构的理解和UI与代码的对齐能力。**

- **链接: [https://arxiv.org/pdf/2410.18362](https://arxiv.org/pdf/2410.18362)**

> **作者:** Shanchao Liang; Nan Jiang; Shangshu Qian; Lin Tan
>
> **摘要:** Web development involves turning UI designs into functional webpages, which can be difficult for both beginners and experienced developers due to the complexity of HTML's hierarchical structures and styles. While Large Language Models (LLMs) have shown promise in generating source code, two major challenges persist in UI-to-HTML code generation: (1) effectively representing HTML's hierarchical structure for LLMs, and (2) bridging the gap between the visual nature of UI designs and the text-based format of HTML code. To tackle these challenges, we introduce Waffle, a new fine-tuning strategy that uses a structure-aware attention mechanism to improve LLMs' understanding of HTML's structure and a contrastive fine-tuning approach to align LLMs' understanding of UI images and HTML code. Models fine-tuned with Waffle show up to 9.00 pp (percentage point) higher HTML match, 0.0982 higher CW-SSIM, 32.99 higher CLIP, and 27.12 pp higher LLEM on our new benchmark WebSight-Test and an existing benchmark Design2Code, outperforming current fine-tuning methods.
>
---
#### [replaced 038] Scalable Multilingual Multimodal Machine Translation with Speech-Text Fusion
- **分类: cs.CL**

- **简介: 该论文属于多语言多模态机器翻译任务，旨在解决数据稀缺问题。通过融合语音与文本信息，并引入自进化机制，提升翻译质量与覆盖范围。**

- **链接: [https://arxiv.org/pdf/2602.21646](https://arxiv.org/pdf/2602.21646)**

> **作者:** Yexing Du; Youcheng Pan; Zekun Wang; Zheng Chu; Yichong Huang; Kaiyuan Liu; Bo Yang; Yang Xiang; Ming Liu; Bing Qin
>
> **备注:** Accepted in ICLR 2026
>
> **摘要:** Multimodal Large Language Models (MLLMs) have achieved notable success in enhancing translation performance by integrating multimodal information. However, existing research primarily focuses on image-guided methods, whose applicability is constrained by the scarcity of multilingual image-text pairs. The speech modality overcomes this limitation due to its natural alignment with text and the abundance of existing speech datasets, which enable scalable language coverage. In this paper, we propose a Speech-guided Machine Translation (SMT) framework that integrates speech and text as fused inputs into an MLLM to improve translation quality. To mitigate reliance on low-resource data, we introduce a Self-Evolution Mechanism. The core components of this framework include a text-to-speech model, responsible for generating synthetic speech, and an MLLM capable of classifying synthetic speech samples and iteratively optimizing itself using positive samples. Experimental results demonstrate that our framework surpasses all existing methods on the Multi30K multimodal machine translation benchmark, achieving new state-of-the-art results. Furthermore, on general machine translation datasets, particularly the FLORES-200, it achieves average state-of-the-art performance in 108 translation directions. Ablation studies on CoVoST-2 confirms that differences between synthetic and authentic speech have negligible impact on translation quality. The code and models are released at this https URL.
>
---
#### [replaced 039] LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出LaDiR框架，解决LLM在文本推理中难以高效迭代和多样化生成的问题。通过结合变分自编码器和扩散模型，提升推理的准确性和多样性。**

- **链接: [https://arxiv.org/pdf/2510.04573](https://arxiv.org/pdf/2510.04573)**

> **作者:** Haoqiang Kang; Yizhe Zhang; Nikki Lijing Kuang; Nicklas Majamaki; Navdeep Jaitly; Yi-An Ma; Lianhui Qin
>
> **摘要:** Large Language Models (LLMs) demonstrate their reasoning ability through chain-of-thought (CoT) generation. However, LLM's autoregressive decoding may limit the ability to revisit and refine earlier tokens in a holistic manner, which can also lead to inefficient exploration for diverse solutions. In this paper, we propose LaDiR (Latent Diffusion Reasoner), a novel reasoning framework that unifies the expressiveness of continuous latent representation with the iterative refinement capabilities of latent diffusion models for an existing LLM. We first construct a structured latent reasoning space using a Variational Autoencoder (VAE) that encodes text reasoning steps into blocks of thought tokens, preserving semantic information and interpretability while offering compact but expressive representations. Subsequently, we utilize a latent diffusion model that learns to denoise a block of latent thought tokens with a blockwise bidirectional attention mask, enabling longer horizon and iterative refinement with adaptive test-time compute. This design allows efficient parallel generation of diverse reasoning trajectories, allowing the model to plan and revise the reasoning process holistically. We conduct evaluations on a suite of mathematical reasoning and planning benchmarks. Empirical results show that LaDiR consistently improves accuracy, diversity, and interpretability over existing autoregressive, diffusion-based, and latent reasoning methods, revealing a new paradigm for text reasoning with latent diffusion.
>
---
#### [replaced 040] Link Prediction for Event Logs in the Process Industry
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于链接预测任务，旨在解决过程工业中事件日志碎片化问题，通过结合NLP与语义相似性方法提升日志连通性。**

- **链接: [https://arxiv.org/pdf/2508.09096](https://arxiv.org/pdf/2508.09096)**

> **作者:** Anastasia Zhukova; Thomas Walton; Christian E. Lobmüller; Bela Gipp
>
> **摘要:** In the era of graph-based retrieval-augmented generation (RAG), link prediction is a significant preprocessing step for improving the quality of fragmented or incomplete domain-specific data for the graph retrieval. Knowledge management in the process industry uses RAG-based applications to optimize operations, ensure safety, and facilitate continuous improvement by effectively leveraging operational data and past insights. A key challenge in this domain is the fragmented nature of event logs in shift books, where related records are often kept separate, even though they belong to a single event or process. This fragmentation hinders the recommendation of previously implemented solutions to users, which is crucial in the timely problem-solving at live production sites. To address this problem, we develop a record linking (RL) model, which we define as a cross-document coreference resolution (CDCR) task. RL adapts the task definition of CDCR and combines two state-of-the-art CDCR models with the principles of natural language inference (NLI) and semantic text similarity (STS) to perform link prediction. The evaluation shows that our RL model outperformed the best versions of our baselines, i.e., NLP and STS, by 28% (11.43 p) and 27.4% (11.21 p), respectively. Our work demonstrates that common NLP tasks can be combined and adapted to a domain-specific setting of the German process industry, improving data quality and connectivity in shift logs.
>
---
#### [replaced 041] Spilled Energy in Large Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于语言模型检测任务，旨在解决事实错误和幻觉问题。通过将softmax分类器视为能量模型，引入两种无需训练的指标检测幻觉。**

- **链接: [https://arxiv.org/pdf/2602.18671](https://arxiv.org/pdf/2602.18671)**

> **作者:** Adrian Robert Minut; Hazem Dewidar; Iacopo Masi
>
> **摘要:** We reinterpret the final Large Language Model (LLM) softmax classifier as an Energy-Based Model (EBM), decomposing the sequence-to-sequence probability chain into multiple interacting EBMs at inference. This principled approach allows us to track "energy spills" during decoding, which we empirically show correlate with factual errors, biases, and failures. Similar to Orgad et al. (2025), our method localizes the exact answer token and subsequently tests for hallucinations. Crucially, however, we achieve this without requiring trained probe classifiers or activation ablations. Instead, we introduce two completely training-free metrics derived directly from output logits: spilled energy, which captures the discrepancy between energy values across consecutive generation steps that should theoretically match, and marginalized energy, which is measurable at a single step. Evaluated on nine benchmarks across state-of-the-art LLMs (including LLaMA, Mistral, and Gemma) and on synthetic algebraic operations (Qwen3), our approach demonstrates robust, competitive hallucination detection and cross-task generalization. Notably, these results hold for both pretrained and instruction-tuned variants without introducing any training overhead. Code available at: this http URL
>
---
#### [replaced 042] Can LLMs Discern the Traits Influencing Your Preferences? Evaluating Personality-Driven Preference Alignment in LLMs
- **分类: cs.CL**

- **简介: 该论文属于个性化问答任务，旨在解决如何有效利用用户偏好提升回答质量的问题。通过引入人格特质作为潜在信号，提升偏好对齐效果。**

- **链接: [https://arxiv.org/pdf/2602.07181](https://arxiv.org/pdf/2602.07181)**

> **作者:** Tianyu Zhao; Siqi Li; Yasser Shoukry; Salma Elmalaki
>
> **摘要:** User preferences are increasingly used to personalize Large Language Model (LLM) responses, yet how to reliably leverage preference signals for answer generation remains under-explored. In practice, preferences can be noisy, incomplete, or even misleading, which can degrade answer quality when applied naively. Motivated by the observation that stable personality traits shape everyday preferences, we study personality as a principled ''latent'' signal behind preference statements. Through extensive experiments, we find that conditioning on personality-aligned preferences substantially improves personalized question answering: selecting preferences consistent with a user's inferred personality increases answer-choice accuracy from 29.25% to 76%, compared to using randomly selected preferences. Based on these findings, we introduce PACIFIC (Preference Alignment Choices Inference for Five-factor Identity Characterization), a personality-labeled preference dataset containing 1200 preference statements spanning diverse domains (e.g., travel, movies, education), annotated with Big-Five (OCEAN) trait directions. Finally, we propose a framework that enables an LLM model to automatically retrieve personality-aligned preferences and incorporate them during answer generation.
>
---
#### [replaced 043] $\texttt{SEM-CTRL}$: Semantically Controlled Decoding
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出SEM-CTRL，解决LLM输出的语法和语义正确性问题。通过约束解码过程，确保输出符合特定任务语义，无需微调。**

- **链接: [https://arxiv.org/pdf/2503.01804](https://arxiv.org/pdf/2503.01804)**

> **作者:** Mohammad Albinhassan; Pranava Madhyastha; Alessandra Russo
>
> **摘要:** Ensuring both syntactic and semantic correctness in Large Language Model (LLM) outputs remains a significant challenge, despite being critical for real-world deployment. In this paper, we introduce \texttt{SEM-CTRL}, a unified approach that allows for enforcing rich context-sensitive constraints, and task and instance specific semantics directly on the LLM decoder. Our approach integrates token-level MCTS which is guided by specific syntactic and semantic constraints. The constraints over desired outputs are expressed using Answer Set Grammars, which is a logic-based formalism that generalizes context sensitive grammars while incorporating background knowledge to represent task-specific semantics. We show that our approach helps guarantee valid completions for any off-the-shelf LLM without the need for fine-tuning. We evaluate \texttt{SEM-CTRL} on a range of tasks, including synthetic grammar synthesis, combinatorial reasoning, JSON parsing, and planning. Our experimental results demonstrate that \texttt{SEM-CTRL} allows even small pre-trained LLMs to efficiently outperform larger variants and state-of-the-art reasoning models (e.g., \textit{o4-mini}) while simultaneously guaranteeing semantic validity.
>
---
#### [replaced 044] Search Arena: Analyzing Search-Augmented LLMs
- **分类: cs.CL; cs.IR; cs.LG**

- **简介: 该论文提出Search Arena数据集，用于分析搜索增强型语言模型。任务是评估用户偏好与模型性能，解决现有数据集不足的问题。工作包括构建大规模多轮交互数据集并进行多场景测试。**

- **链接: [https://arxiv.org/pdf/2506.05334](https://arxiv.org/pdf/2506.05334)**

> **作者:** Mihran Miroyan; Tsung-Han Wu; Logan King; Tianle Li; Jiayi Pan; Xinyan Hu; Wei-Lin Chiang; Anastasios N. Angelopoulos; Trevor Darrell; Narges Norouzi; Joseph E. Gonzalez
>
> **备注:** Accepted to ICLR 2026. Code: this https URL. Dataset: this https URL
>
> **摘要:** Search-augmented language models combine web search with Large Language Models (LLMs) to improve response groundedness and freshness. However, analyzing these systems remains challenging: existing datasets are limited in scale and narrow in scope, often constrained to static, single-turn, fact-checking questions. In this work, we introduce Search Arena, a crowd-sourced, large-scale, human-preference dataset of over 24,000 paired multi-turn user interactions with search-augmented LLMs. The dataset spans diverse intents and languages, and contains full system traces with around 12,000 human preference votes. Our analysis reveals that user preferences are influenced by the number of citations, even when the cited content does not directly support the attributed claims, uncovering a gap between perceived and actual credibility. Furthermore, user preferences vary across cited sources, revealing that community-driven platforms are generally preferred and static encyclopedic sources are not always appropriate and reliable. To assess performance across different settings, we conduct cross-arena analyses by testing search-augmented LLMs in a general-purpose chat environment and conventional LLMs in search-intensive settings. We find that web search does not degrade and may even improve performance in non-search settings; however, the quality in search settings is significantly affected if solely relying on the model's parametric knowledge. We open-sourced the dataset to support future research. Our dataset and code are available at: this https URL.
>
---
#### [replaced 045] LEDOM: Reverse Language Model
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出LEDOM，一个反向自回归语言模型，用于研究反向推理模式。解决传统模型仅左到右训练的问题，探索反向预测的潜力。工作包括模型训练、能力分析及应用Reverse Reward提升生成质量。**

- **链接: [https://arxiv.org/pdf/2507.01335](https://arxiv.org/pdf/2507.01335)**

> **作者:** Xunjian Yin; Sitao Cheng; Yuxi Xie; Xinyu Hu; Li Lin; Xinyi Wang; Liangming Pan; William Yang Wang; Xiaojun Wan
>
> **备注:** Work in progress; Models can be found at: this https URL
>
> **摘要:** Autoregressive language models are trained exclusively left-to-right. We explore the complementary factorization, training right-to-left at scale, and ask what reasoning patterns emerge when a model conditions on future context to predict the past. We train LEDOM, an open-source purely reverse autoregressive language model (2B/7B parameters, 435B tokens), and find it develops capabilities distinct from forward models, including abductive inference, question synthesis, and natural resolution of the reversal curse. We then explore one application of the reverse model: combining forward likelihood $P(y \mid x)$ with reverse posterior $P(x \mid y)$ through noisy channel duality. We propose Reverse Reward, which reranks forward outputs using reverse posterior estimates, and prove that bidirectional scoring penalizes hallucinated reasoning chains whose backward reconstruction degrades. Reverse Reward yields gains of up to 6.6\% on AIME 2024 and 15\% on AMC 2023 across multiple strong baselines. We release all models, code, and data here.
>
---
#### [replaced 046] Automated Data Enrichment using Confidence-Aware Fine-Grained Debate among Open-Source LLMs for Mental Health and Online Safety
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，解决多标签数据标注成本高、难度大的问题。提出CFD框架，通过LLMs协作提升数据丰富性，实验表明效果优于基线方法。**

- **链接: [https://arxiv.org/pdf/2512.06227](https://arxiv.org/pdf/2512.06227)**

> **作者:** Junyu Mao; Anthony Hills; Talia Tseriotou; Maria Liakata; Aya Shamir; Dan Sayda; Dana Atzil-Slonim; Natalie Djohari; Arpan Mandal; Silke Roth; Pamela Ugwudike; Mahesan Niranjan; Stuart E. Middleton
>
> **摘要:** Real-world indicators play an important role in many natural language processing (NLP) applications, such as life-event for mental health analysis and risky behaviour for online safety, yet labelling such information in training datasets is often costly and/or difficult due to their dynamic nature. Large language models (LLMs) show promising potential for automated annotation, yet multi-label prediction remains challenging. In this work, we propose a Confidence-Aware Fine-Grained Debate (CFD) framework that simulates collaborative annotation using fine-grained information to better support automated multi-label enrichment. We introduce two new expert-annotated resources: A mental health Reddit well-being dataset and an online safety Facebook sharenting risk dataset. Experiments show that CFD achieves the most robust enrichment performance compared to a range of baseline approaches. We further evaluate various training-free enrichment incorporation strategies and demonstrate that LLM-enriched indicators consistently improves our downstream tasks. Enriched features incorporated via debate transcripts yield the largest gains, outperforming the non-enriched baseline by 9.9\% on the online safety task.
>
---
#### [replaced 047] Prior-based Noisy Text Data Filtering: Fast and Strong Alternative For Perplexity
- **分类: cs.CL**

- **简介: 该论文属于数据过滤任务，旨在解决传统PPL方法耗时且不可靠的问题。提出基于先验的过滤方法，利用词频统计快速筛选噪声文本。**

- **链接: [https://arxiv.org/pdf/2509.18577](https://arxiv.org/pdf/2509.18577)**

> **作者:** Yeongbin Seo; Gayoung Kim; Jaehyung Kim; Jinyoung Yeo
>
> **备注:** ICLR 2026
>
> **摘要:** As large language models (LLMs) are pretrained on massive web corpora, careful selection of data becomes essential to ensure effective and efficient learning. While perplexity (PPL)-based filtering has shown strong performance, it suffers from drawbacks: substantial time costs and inherent unreliability of the model when handling noisy or out-of-distribution samples. In this work, we propose a simple yet powerful alternative: a prior-based data filtering method that estimates token priors using corpus-level term frequency statistics, inspired by linguistic insights on word roles and lexical density. Our approach filters documents based on the mean and standard deviation of token priors, serving as a fast proxy to PPL while requiring no model inference. Despite its simplicity, the prior-based filter achieves the highest average performance across 20 downstream benchmarks, while reducing time cost by over 1000x compared to PPL-based filtering. We further demonstrate its applicability to symbolic languages such as code and math, and its dynamic adaptability to multilingual corpora without supervision
>
---
#### [replaced 048] AccurateRAG: A Framework for Building Accurate Retrieval-Augmented Question-Answering Applications
- **分类: cs.CL**

- **简介: 该论文提出AccurateRAG框架，用于构建高精度的问答系统。解决传统方法在问答准确性上的不足，通过优化数据处理、模型微调和评估流程，提升性能。**

- **链接: [https://arxiv.org/pdf/2510.02243](https://arxiv.org/pdf/2510.02243)**

> **作者:** Linh The Nguyen; Chi Tran; Dung Ngoc Nguyen; Van-Cuong Pham; Hoang Ngo; Dat Quoc Nguyen
>
> **备注:** Accepted to LREC 2026
>
> **摘要:** We introduce AccurateRAG -- a novel framework for constructing high-performance question-answering applications based on retrieval-augmented generation (RAG). Our framework offers a pipeline for development efficiency with tools for raw dataset processing, fine-tuning data generation, text embedding & LLM fine-tuning, output evaluation, and building RAG systems locally. Experimental results show that our framework outperforms previous strong baselines and obtains new state-of-the-art question-answering performance on benchmark datasets.
>
---
#### [replaced 049] RuCL: Stratified Rubric-Based Curriculum Learning for Multimodal Large Language Model Reasoning
- **分类: cs.CL**

- **简介: 该论文属于多模态大语言模型推理任务，解决奖励黑客问题。提出RuCL框架，通过分层评分体系优化训练过程，提升模型推理能力。**

- **链接: [https://arxiv.org/pdf/2602.21628](https://arxiv.org/pdf/2602.21628)**

> **作者:** Yukun Chen; Jiaming Li; Longze Chen; Ze Gong; Jingpeng Li; Zhen Qin; Hengyu Chang; Ancheng Xu; Zhihao Yang; Hamid Alinejad-Rokny; Qiang Qu; Bo Zheng; Min Yang
>
> **备注:** 8 pages
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a prevailing paradigm for enhancing reasoning in Multimodal Large Language Models (MLLMs). However, relying solely on outcome supervision risks reward hacking, where models learn spurious reasoning patterns to satisfy final answer checks. While recent rubric-based approaches offer fine-grained supervision signals, they suffer from high computational costs of instance-level generation and inefficient training dynamics caused by treating all rubrics as equally learnable. In this paper, we propose Stratified Rubric-based Curriculum Learning (RuCL), a novel framework that reformulates curriculum learning by shifting the focus from data selection to reward design. RuCL generates generalized rubrics for broad applicability and stratifies them based on the model's competence. By dynamically adjusting rubric weights during training, RuCL guides the model from mastering foundational perception to tackling advanced logical reasoning. Extensive experiments on various visual reasoning benchmarks show that RuCL yields a remarkable +7.83% average improvement over the Qwen2.5-VL-7B model, achieving a state-of-the-art accuracy of 60.06%.
>
---
#### [replaced 050] Adaptive Social Learning via Mode Policy Optimization for Language Agents
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于社会智能任务，旨在解决语言代理在动态社交互动中缺乏自适应推理能力的问题。提出ASL框架与AMPO算法，实现多粒度推理模式和高效推理。**

- **链接: [https://arxiv.org/pdf/2505.02156](https://arxiv.org/pdf/2505.02156)**

> **作者:** Minzheng Wang; Yongbin Li; Haobo Wang; Xinghua Zhang; Nan Xu; Bingli Wu; Fei Huang; Haiyang Yu; Wenji Mao
>
> **备注:** Proceedings of ICLR 2026. The code and data are available, see this https URL
>
> **摘要:** Effective social intelligence simulation requires language agents to dynamically adjust reasoning depth, a capability notably absent in current studies. Existing methods either lack explicit reasoning or employ lengthy Chain-of-Thought reasoning uniformly across all scenarios, resulting in excessive token usage and inflexible social behaviors in tasks such as negotiation or collaboration. To address this, we propose an $\textbf{A}$daptive $\textbf{S}$ocial $\textbf{L}$earning ($\textbf{ASL}$) framework in this paper, aiming to improve the adaptive reasoning ability of language agents in dynamic social interactions. To this end, we first identify the hierarchical reasoning modes under such context, ranging from intuitive response to deep deliberation based on the cognitive control theory. We then develop the $\textbf{A}$daptive $\textbf{M}$ode $\textbf{P}$olicy $\textbf{O}$ptimization ($\textbf{AMPO}$) algorithm to learn the context-aware mode adaptation and reasoning. Our framework advances existing research in three key aspects: (1) Multi-granular reasoning mode design, (2) Context-aware mode switching in rich social interaction, and (3) Token-efficient reasoning with depth adaptation. Extensive experiments on the benchmark social intelligence environment verify that ASL achieves 15.6% higher task performance than GPT-4o. Notably, our AMPO outperforms GRPO by 7.0% with 32.8% shorter thinking chains, demonstrating the advantages of our AMPO and the learned adaptive reasoning ability over GRPO's solution.
>
---
#### [replaced 051] Entropy Sentinel: Continuous LLM Accuracy Monitoring from Decoding Entropy Traces in STEM
- **分类: cs.CL**

- **简介: 该论文属于模型监控任务，旨在解决LLM在领域漂移下的准确性评估问题。通过分析解码熵迹，构建输出熵轮廓，实现对模型性能的持续监测与数据采集优化。**

- **链接: [https://arxiv.org/pdf/2601.09001](https://arxiv.org/pdf/2601.09001)**

> **作者:** Pedro Memoli Buffa; Luciano Del Corro
>
> **摘要:** Deploying LLMs raises two coupled challenges: (1) monitoring--estimating where a model underperforms as traffic and domains drift--and (2) improvement--prioritizing data acquisition to close the largest performance gaps. We test whether an inference-time signal can estimate slice-level accuracy under domain shift. For each response, we compute an output-entropy profile from final-layer next-token probabilities (from top-$k$ logprobs) and summarize it with different statistics. A lightweight classifier predicts instance correctness, and averaging predicted probabilities yields a domain-level accuracy estimate. We evaluate on ten STEM reasoning benchmarks with exhaustive train/test compositions ($k\in\{1,2,3,4\}$; all $\binom{10}{k}$ combinations), on different classifier models and features across nine LLMs from six families (3B--20B). Estimates often track held-out benchmark accuracy, and several models show near-monotonic ordering of domains, providing evidence for output-entropy profiles being an accessible signal for scalable monitoring and for targeted data acquisition.
>
---
#### [replaced 052] DeepXiv-SDK: An Agentic Data Interface for Scientific Literature
- **分类: cs.DL; cs.AI; cs.CL; cs.IR**

- **简介: 该论文提出DeepXiv-SDK，解决科学文献数据访问效率低的问题，通过三层架构提升数据可用性与检索效率。**

- **链接: [https://arxiv.org/pdf/2603.00084](https://arxiv.org/pdf/2603.00084)**

> **作者:** Hongjin Qian; Ziyi Xia; Ze Liu; Jianlyu Chen; Kun Luo; Minghao Qin; Chaofan Li; Lei Xiong; Junwei Lan; Sen Wang; Zhengyang Liang; Yingxia Shao; Defu Lian; Zheng Liu
>
> **备注:** Project at this https URL
>
> **摘要:** LLM-agents are increasingly used to accelerate the progress of scientific research. Yet a persistent bottleneck is data access: agents not only lack readily available tools for retrieval, but also have to work with unstrcutured, human-centric data on the Internet, such as HTML web-pages and PDF files, leading to excessive token consumption, limit working efficiency, and brittle evidence look-up. This gap motivates the development of \textit{an agentic data interface}, which is designed to enable agents to access and utilize scientific literature in a more effective, efficient, and cost-aware manner. In this paper, we introduce DeepXiv-SDK, which offers a three-layer agentic data interface for scientific literature. 1) Data Layer, which transforms unstructured, human-centric data into normalized and structured representations in JSON format, improving data usability and enabling progressive accessibility of the data. 2) Service Layer, which presents readily available tools for data access and ad-hoc retrieval. It also enables a rich form of agent usage, including CLI, MCP, and Python SDK. 3) Application Layer, which creates a built-in agent, packaging basic tools from the service layer to support complex data access demands. DeepXiv-SDK currently supports the complete ArXiv corpus, and is synchronized daily to incorporate new releases. It is designed to extend to all common open-access corpora, such as PubMed Central, bioRxiv, medRxiv, and chemRxiv. We release RESTful APIs, an open-source Python SDK, and a web demo showcasing deep search and deep research workflows. DeepXiv-SDK is free to use with registration.
>
---
#### [replaced 053] Are We Asking the Right Questions? On Ambiguity in Natural Language Queries for Tabular Data Analysis
- **分类: cs.AI; cs.CL; cs.DB; cs.HC**

- **简介: 该论文研究自然语言查询在表格数据分析中的歧义问题，属于自然语言处理任务。旨在解决如何区分合作与非合作查询，提出框架以评估系统准确性和解释能力。**

- **链接: [https://arxiv.org/pdf/2511.04584](https://arxiv.org/pdf/2511.04584)**

> **作者:** Daniel Gomm; Cornelius Wolff; Madelon Hulsebos
>
> **备注:** Accepted to the AI for Tabular Data workshop at EurIPS 2025
>
> **摘要:** Natural language interfaces to tabular data must handle ambiguities inherent to queries. Instead of treating ambiguity as a deficiency, we reframe it as a feature of cooperative interaction where users are intentional about the degree to which they specify queries. We develop a principled framework based on a shared responsibility of query specification between user and system, distinguishing unambiguous and ambiguous cooperative queries, which systems can resolve through reasonable inference, from uncooperative queries that cannot be resolved. Applying the framework to evaluations for tabular question answering and analysis, we analyze queries in 15 datasets, and observe an uncontrolled mixing of query types neither adequate for evaluating a system's accuracy nor for evaluating interpretation capabilities. This conceptualization around cooperation in resolving queries informs how to design and evaluate natural language interfaces for tabular data analysis, for which we distill concrete directions for future research and broader implications.
>
---
#### [replaced 054] Go-Browse: Training Web Agents with Structured Exploration
- **分类: cs.CL**

- **简介: 该论文提出Go-Browse，解决网络代理环境理解不足的问题，通过结构化探索收集高质量数据，提升代理任务成功率。**

- **链接: [https://arxiv.org/pdf/2506.03533](https://arxiv.org/pdf/2506.03533)**

> **作者:** Apurva Gandhi; Graham Neubig
>
> **摘要:** One of the fundamental problems in digital agents is their lack of understanding of their environment. For instance, a web browsing agent may get lost in unfamiliar websites, uncertain what pages must be visited to achieve its goals. To address this, we propose Go-Browse, a method for automatically collecting diverse and realistic web agent data at scale through structured exploration of web environments. Go-Browse achieves efficient exploration by framing data collection as a graph search, enabling reuse of information across exploration episodes. We instantiate our method on the WebArena benchmark, collecting a dataset of 10K successful task-solving trajectories and 40K interaction steps across 100 URLs. Fine-tuning a 7B parameter language model on this dataset achieves a success rate of 21.7% on the WebArena benchmark, beating GPT-4o mini by 2.4% and exceeding current state-of-the-art results for sub-10B parameter models by 2.9%.
>
---
#### [replaced 055] GUMBridge: a Corpus for Varieties of Bridging Anaphora
- **分类: cs.CL**

- **简介: 该论文介绍GUMBridge，一个涵盖16种英语语域的桥接指代语料库，旨在解决桥接指代识别与分类问题，提升NLP任务的准确性。**

- **链接: [https://arxiv.org/pdf/2512.07134](https://arxiv.org/pdf/2512.07134)**

> **作者:** Lauren Levine; Amir Zeldes
>
> **备注:** LREC 2026
>
> **摘要:** Bridging is an anaphoric phenomenon where the referent of an entity in a discourse is dependent on a previous, non-identical entity for interpretation, such as in "There is 'a house'. 'The door' is red," where the door is specifically understood to be the door of the aforementioned house. While there are several existing resources in English for bridging anaphora, most are small, provide limited coverage of the phenomenon, and/or provide limited genre coverage. In this paper, we introduce GUMBridge, a new resource for bridging, which includes 16 diverse genres of English, providing both broad coverage for the phenomenon and granular annotations for the subtype categorization of bridging varieties. We also present an evaluation of annotation quality and report on baseline performance using open and closed source contemporary LLMs on three tasks underlying our data, showing that bridging resolution and subtype classification remain difficult NLP tasks in the age of LLMs.
>
---
#### [replaced 056] ManagerBench: Evaluating the Safety-Pragmatism Trade-off in Autonomous LLMs
- **分类: cs.CL**

- **简介: 该论文属于AI安全任务，旨在解决LLMs在安全与实用间的权衡问题。通过构建ManagerBench基准，评估模型在冲突情境下的决策能力。**

- **链接: [https://arxiv.org/pdf/2510.00857](https://arxiv.org/pdf/2510.00857)**

> **作者:** Adi Simhi; Jonathan Herzig; Martin Tutek; Itay Itzhak; Idan Szpektor; Yonatan Belinkov
>
> **摘要:** As large language models (LLMs) evolve from conversational assistants into autonomous agents, evaluating the safety of their actions becomes critical. Prior safety benchmarks have primarily focused on preventing generation of harmful content, such as toxic text. However, they overlook the challenge of agents taking harmful actions when the most effective path to an operational goal conflicts with human safety. To address this gap, we introduce ManagerBench, a benchmark that evaluates LLM decision-making in realistic, human-validated managerial scenarios. Each scenario forces a choice between a pragmatic but harmful action that achieves an operational goal, and a safe action that leads to worse operational performance. A parallel control set, where potential harm is directed only at inanimate objects, measures a model's pragmatism and identifies its tendency to be overly safe. Our findings indicate that the frontier LLMs perform poorly when navigating this safety-pragmatism trade-off. Many consistently choose harmful options to advance their operational goals, while others avoid harm only to become overly safe and ineffective. Critically, we find this misalignment does not stem from an inability to perceive harm, as models' harm assessments align with human judgments, but from flawed prioritization. ManagerBench is a challenging benchmark for a core component of agentic behavior: making safe choices when operational goals and alignment values incentivize conflicting actions. Benchmark & code available at this https URL.
>
---
#### [replaced 057] Cache-to-Cache: Direct Semantic Communication Between Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出Cache-to-Cache（C2C）方法，解决多大语言模型间通信效率低的问题。通过直接传递语义信息，提升响应质量与速度。**

- **链接: [https://arxiv.org/pdf/2510.03215](https://arxiv.org/pdf/2510.03215)**

> **作者:** Tianyu Fu; Zihan Min; Hanling Zhang; Jichao Yan; Guohao Dai; Wanli Ouyang; Yu Wang
>
> **备注:** Published in ICLR'26
>
> **摘要:** Multi-LLM systems harness the complementary strengths of diverse Large Language Models, achieving performance and efficiency gains that are not attainable by a single model. In existing designs, LLMs communicate through text, forcing internal representations to be transformed into output token sequences. This process both loses rich semantic information and incurs token-by-token generation latency. Motivated by these limitations, we ask: Can LLMs communicate beyond text? Oracle experiments show that enriching the KV-Cache semantics can improve response quality without increasing cache size, supporting KV-Cache as an effective medium for inter-model communication. Thus, we propose Cache-to-Cache (C2C), a new paradigm for direct semantic communication between LLMs. C2C uses a neural network to project and fuse the source model's KV-cache with that of the target model to enable direct semantic transfer. A learnable gating mechanism selects the target layers that benefit from cache communication. Compared with text communication, C2C utilizes the deep, specialized semantics from both models, while avoiding explicit intermediate text generation. Experiments show that C2C achieves 6.4-14.2% higher average accuracy than individual models. It further outperforms the text communication paradigm by approximately 3.1-5.4%, while delivering an average 2.5x speedup in latency. Our code is available at this https URL.
>
---
#### [replaced 058] Benefits and Pitfalls of Reinforcement Learning for Language Model Planning: A Theoretical Perspective
- **分类: cs.AI; cs.CL; cs.LG; stat.ML**

- **简介: 该论文研究强化学习在语言模型规划中的应用，分析其优势与局限。任务为理论分析，解决RL方法有效性问题，工作包括理论分析与实验验证。**

- **链接: [https://arxiv.org/pdf/2509.22613](https://arxiv.org/pdf/2509.22613)**

> **作者:** Siwei Wang; Yifei Shen; Haoran Sun; Shi Feng; Shang-Hua Teng; Li Dong; Yaru Hao; Wei Chen
>
> **摘要:** Recent reinforcement learning (RL) methods have substantially enhanced the planning capabilities of Large Language Models (LLMs), yet the theoretical basis for their effectiveness remains elusive. In this work, we investigate RL's benefits and limitations through a tractable graph-based abstraction, focusing on policy gradient (PG) and Q-learning methods. Our theoretical analyses reveal that supervised fine-tuning (SFT) may introduce co-occurrence-based spurious solutions, whereas RL achieves correct planning primarily through exploration, underscoring exploration's role in enabling better generalization. However, we also show that PG suffers from diversity collapse, where output diversity decreases during training and persists even after perfect accuracy is attained. By contrast, Q-learning provides two key advantages: off-policy learning and diversity preservation at convergence. We further demonstrate that careful reward design is necessary to prevent Q-value bias in Q-learning. Finally, applying our framework to the real-world planning benchmark Blocksworld, we confirm that these behaviors manifest in practice.
>
---
#### [replaced 059] You Only Fine-tune Once: Many-Shot In-Context Fine-Tuning for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，解决大模型在多任务中性能不足的问题。提出ManyICL方法，通过多示例上下文微调提升模型性能，接近专用微调效果。**

- **链接: [https://arxiv.org/pdf/2506.11103](https://arxiv.org/pdf/2506.11103)**

> **作者:** Wenchong He; Liqian Peng; Zhe Jiang; Alex Go
>
> **备注:** 20 pages, 6 figures, 12 tables
>
> **摘要:** Large language models (LLMs) possess a remarkable ability to perform in-context learning (ICL), which enables them to handle multiple downstream tasks simultaneously without requiring task-specific fine-tuning. Recent studies have shown that even moderately sized LLMs, such as Mistral 7B, Gemma 7B and Llama-3 8B, can achieve ICL through few-shot in-context fine-tuning of all tasks at once. However, this approach still lags behind dedicated fine-tuning, where a separate model is trained for each individual task. In this paper, we propose a novel approach, Many-Shot In-Context Fine-tuning (ManyICL), which significantly narrows this performance gap by extending the principles of ICL to a many-shot setting. To unlock the full potential of ManyICL and address the inherent inefficiency of processing long sequences with numerous in-context examples, we propose a novel training objective. Instead of solely predicting the final answer, our approach treats every answer within the context as a supervised training target. This effectively shifts the role of many-shot examples from prompts to targets for autoregressive learning. Through extensive experiments on diverse downstream tasks, including classification, summarization, question answering, natural language inference, and math, we demonstrate that ManyICL substantially outperforms zero/few-shot fine-tuning and approaches the performance of dedicated fine-tuning. Furthermore, ManyICL significantly mitigates catastrophic forgetting issues observed in zero/few-shot fine-tuning. The code will be made publicly available upon publication.
>
---
#### [replaced 060] ClinConsensus: A Consensus-Based Benchmark for Evaluating Chinese Medical LLMs across Difficulty Levels
- **分类: cs.CL**

- **简介: 该论文提出ClinConsensus，一个用于评估中文医疗大模型的基准，解决现有基准静态、孤立的问题。通过多维度案例和评分体系，评估模型在不同临床任务中的表现。**

- **链接: [https://arxiv.org/pdf/2603.02097](https://arxiv.org/pdf/2603.02097)**

> **作者:** Xiang Zheng; Han Li; Wenjie Luo; Weiqi Zhai; Yiyuan Li; Chuanmiao Yan; Tianyi Tang; Yubo Ma; Kexin Yang; Dayiheng Liu; Hu Wei; Bing Zhao
>
> **备注:** 8 pages, 6 figures,
>
> **摘要:** Large language models (LLMs) are increasingly applied to health management, showing promise across disease prevention, clinical decision-making, and long-term care. However, existing medical benchmarks remain largely static and task-isolated, failing to capture the openness, longitudinal structure, and safety-critical complexity of real-world clinical workflows. We introduce ClinConsensus, a Chinese medical benchmark curated, validated, and quality-controlled by clinical experts. ClinConsensus comprises 2500 open-ended cases spanning the full continuum of care--from prevention and intervention to long-term follow-up--covering 36 medical specialties, 12 common clinical task types, and progressively increasing levels of complexity. To enable reliable evaluation of such complex scenarios, we adopt a rubric-based grading protocol and propose the Clinically Applicable Consistency Score (CACS@k). We further introduce a dual-judge evaluation framework, combining a high-capability LLM-as-judge with a distilled, locally deployable judge model trained via supervised fine-tuning, enabling scalable and reproducible evaluation aligned with physician judgment. Using ClinConsensus, we conduct a comprehensive assessment of several leading LLMs and reveal substantial heterogeneity across task themes, care stages, and medical specialties. While top-performing models achieve comparable overall scores, they differ markedly in reasoning, evidence use, and longitudinal follow-up capabilities, and clinically actionable treatment planning remains a key bottleneck. We release ClinConsensus as an extensible benchmark to support the development and evaluation of medical LLMs that are robust, clinically grounded, and ready for real-world deployment.
>
---
#### [replaced 061] CyclicReflex: Improving Reasoning Models via Cyclical Reflection Token Scheduling
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的推理任务，旨在解决大模型在推理过程中反射标记使用不当的问题。通过提出CyclicReflex方法，动态调节反射标记的使用，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2506.11077](https://arxiv.org/pdf/2506.11077)**

> **作者:** Chongyu Fan; Yihua Zhang; Jinghan Jia; Alfred Hero; Sijia Liu
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Large reasoning models (LRMs), such as OpenAI's o1 and DeepSeek-R1, harness test-time scaling to perform multi-step reasoning for complex problem-solving. This reasoning process, executed before producing final answers, is often guided by special juncture tokens that prompt self-evaluative reflection. These transition markers and reflective cues are referred to as "reflection tokens" (e.g., "wait", "but", "alternatively"). In this work, we treat reflection tokens as a "resource" and introduce the problem of resource allocation, aimed at improving the test-time compute performance of LRMs by adaptively regulating the frequency and placement of reflection tokens. Through empirical analysis, we show that both excessive and insufficient use of reflection tokens, referred to as over-reflection and under-reflection, can degrade model performance. To better understand this trade-off, we draw an analogy between reflection token usage and learning rate scheduling in optimization. Building on this insight, We propose cyclical reflection token scheduling (termed CyclicReflex), a training-free decoding strategy that dynamically modulates reflection token logits with a bidirectional, position-dependent triangular waveform, incurring no additional computation cost. Experiments on MATH500, AIME2024/2025, AMC2023, GPQA Diamond and LiveCodeBench demonstrate that CyclicReflex consistently improves performance across model sizes (1.5B-14B), outperforming standard decoding and recent approaches such as TIP (thought switching penalty) and S1. Codes are available at this https URL.
>
---
#### [replaced 062] Mitigating Over-Refusal in Aligned Large Language Models via Inference-Time Activation Energy
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **简介: 该论文属于大语言模型安全对齐任务，旨在解决模型过度拒绝合法请求的问题。通过引入ELS框架，在推理时动态调整模型行为，提升安全性同时降低误拒率。**

- **链接: [https://arxiv.org/pdf/2510.08646](https://arxiv.org/pdf/2510.08646)**

> **作者:** Eric Hanchen Jiang; Weixuan Ou; Run Liu; Shengyuan Pang; Guancheng Wan; Ranjie Duan; Wei Dong; Kai-Wei Chang; XiaoFeng Wang; Ying Nian Wu; Xinfeng Li
>
> **摘要:** Safety alignment of large language models currently faces a central challenge: existing alignment techniques often prioritize mitigating responses to harmful prompts at the expense of overcautious behavior, leading models to incorrectly refuse benign requests. A key goal of safe alignment is therefore to improve safety while simultaneously minimizing false refusals. In this work, we introduce Energy Landscape Steering (ELS), a novel, fine-tuning free framework designed to resolve this challenge through dynamic, inference-time intervention. We train a lightweight external Energy-Based Model (EBM) to assign high energy to undesirable states (false refusal or jailbreak) and low energy to desirable states (helpful response or safe reject). During inference, the EBM maps the LLM's internal activations to an energy landscape, and we use the gradient of the energy function to steer the hidden states toward low-energy regions in real time. This dynamically guides the model toward desirable behavior without modifying its parameters. By decoupling behavioral control from the model's core knowledge, ELS provides a flexible and computationally efficient solution. Extensive experiments across diverse models demonstrate its effectiveness, raising compliance on the ORB-H benchmark from 57.3 percent to 82.6 percent while maintaining baseline safety performance. Our work establishes a promising paradigm for building LLMs that simultaneously achieve high safety and low false refusal rates.
>
---
#### [replaced 063] Skywork-Reward-V2: Scaling Preference Data Curation via Human-AI Synergy
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于强化学习任务，旨在解决奖励模型性能不足的问题。通过构建大规模高质量偏好数据集并采用人机协同方法，训练出高性能的奖励模型Skywork-Reward-V2。**

- **链接: [https://arxiv.org/pdf/2507.01352](https://arxiv.org/pdf/2507.01352)**

> **作者:** Chris Yuhao Liu; Liang Zeng; Yuzhen Xiao; Jujie He; Jiacai Liu; Chaojie Wang; Rui Yan; Wei Shen; Fuxiang Zhang; Jiacheng Xu; Yang Liu; Yahui Zhou
>
> **备注:** ICLR 2026 Poster
>
> **摘要:** Despite the critical role of reward models (RMs) in Reinforcement Learning from Human Feedback (RLHF), current state-of-the-art open RMs perform poorly on most existing evaluation benchmarks, failing to capture nuanced human preferences. We hypothesize that this brittleness stems primarily from limitations in preference datasets, which are often narrowly scoped, synthetically labeled, or lack rigorous quality control. To address these challenges, we present SynPref-40M, a large-scale preference dataset comprising 40 million preference pairs. To enable data curation at scale, we design a human-AI synergistic two-stage pipeline that leverages the complementary strengths of human annotation quality and AI scalability. In this pipeline, humans provide verified annotations, while LLMs perform automatic curation based on human guidance. Training on this preference mixture, we introduce Skywork-Reward-V2, a suite of eight reward models ranging from 0.6B to 8B parameters, trained on a carefully curated subset of 26 million preference pairs from SynPref-40M. We demonstrate that Skywork-Reward-V2 is versatile across a wide range of capabilities, including alignment with human preferences, objective correctness, safety, resistance to stylistic biases, and best-of-N scaling. These reward models achieve state-of-the-art performance across seven major reward model benchmarks, outperform generative reward models, and demonstrate strong downstream performance. Ablation studies confirm that effectiveness stems not only from data scale but also from high-quality curation. The Skywork-Reward-V2 series represents substantial progress in open reward models, demonstrating how human-AI curation synergy can unlock significantly higher data quality.
>
---
#### [replaced 064] CeRA: Breaking the Linear Ceiling of Low-Rank Adaptation via Manifold Expansion
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出CeRA，解决LoRA在复杂推理任务中的线性瓶颈问题，通过引入结构化机制提升模型性能。属于参数高效微调任务。**

- **链接: [https://arxiv.org/pdf/2602.22911](https://arxiv.org/pdf/2602.22911)**

> **作者:** Hung-Hsuan Chen
>
> **摘要:** Low-Rank Adaptation (LoRA) dominates parameter-efficient fine-tuning (PEFT). However, it faces a critical ``linear ceiling'' in complex reasoning tasks: simply increasing the rank yields diminishing returns due to intrinsic linear constraints. We introduce CeRA (Capacity-enhanced Rank Adaptation), a weight-level parallel adapter that injects SiLU gating and structural dropout to induce manifold expansion. On the SlimOrca benchmark, CeRA breaks this linear barrier: at rank 64 (PPL 3.89), it outperforms LoRA at rank 512 (PPL 3.90), demonstrating superior spectral efficiency. This advantage generalizes to mathematical reasoning, where CeRA achieves a perplexity of 1.97 on MathInstruct, significantly surpassing LoRA's saturation point of 2.07. Mechanism analysis via Singular Value Decomposition (SVD) confirms that CeRA activates the dormant tail of the singular value spectrum, effectively preventing the rank collapse observed in linear methods.
>
---
#### [replaced 065] Think-While-Generating: On-the-Fly Reasoning for Personalized Long-Form Generation
- **分类: cs.CL**

- **简介: 该论文属于个性化长文本生成任务，旨在解决现有方法难以捕捉用户隐式偏好、效率低的问题。提出FlyThinker框架，实现生成与推理并行，提升个性化效果与效率。**

- **链接: [https://arxiv.org/pdf/2512.06690](https://arxiv.org/pdf/2512.06690)**

> **作者:** Chengbing Wang; Yang Zhang; Wenjie Wang; Xiaoyan Zhao; Fuli Feng; Xiangnan He; Tat-Seng Chua
>
> **备注:** Published as a conference paper at ICLR 2026
>
> **摘要:** Preference alignment has enabled large language models (LLMs) to better reflect human expectations, but current methods mostly optimize for population-level preferences, overlooking individual users. Personalization is essential, yet early approaches-such as prompt customization or fine-tuning-struggle to reason over implicit preferences, limiting real-world effectiveness. Recent "think-then-generate" methods address this by reasoning before response generation. However, they face challenges in long-form generation: their static one-shot reasoning must capture all relevant information for the full response generation, making learning difficult and limiting adaptability to evolving content. To address this issue, we propose FlyThinker, an efficient "think-while-generating" framework for personalized long-form generation. FlyThinker employs a separate reasoning model that generates latent token-level reasoning in parallel, which is fused into the generation model to dynamically guide response generation. This design enables reasoning and generation to run concurrently, ensuring inference efficiency. In addition, the reasoning model is designed to depend only on previous responses rather than its own prior outputs, which preserves training parallelism across different positions-allowing all reasoning tokens for training data to be produced in a single forward pass like standard LLM training, ensuring training efficiency. Extensive experiments on real-world benchmarks demonstrate that FlyThinker achieves better personalized generation while keeping training and inference efficiency.
>
---
#### [replaced 066] A Set of Quebec-French Corpus of Regional Expressions and Terms
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的方言理解任务，旨在解决方言与习语识别问题。构建了三个法语方言语料库，评估大模型在不同方言上的表现差异。**

- **链接: [https://arxiv.org/pdf/2510.05026](https://arxiv.org/pdf/2510.05026)**

> **作者:** David Beauchemin; Yan Tremblay; Mohamed Amine Youssef; Richard Khoury
>
> **备注:** Submitted to ACL Rolling Review of October
>
> **摘要:** The tasks of idiom understanding and dialect understanding are both well-established benchmarks in natural language processing. In this paper, we propose combining them, and using regional idioms as a test of dialect understanding. Towards this end, we propose two new benchmark datasets for the Quebec dialect of French: QFrCoRE, which contains 4,633 instances of idiomatic phrases, and QFrCoRT, which comprises 171 regional instances of idiomatic words, and a new benchmark for French Metropolitan expressions, MFrCoE, which comprises 4,938 phrases. We explain how to construct these corpora, so that our methodology can be replicated for other dialects. Our experiments with 111 LLMs reveal a critical disparity in dialectal competence: while models perform well on French Metropolitan , 65.8% of them perform significantly worse on Quebec idioms, with only 9.0% favoring the regional dialect. These results confirm that our benchmarks are a reliable tool for quantifying the dialect gap and that prestige-language proficiency does not guarantee regional dialect understanding.
>
---
#### [replaced 067] Bridging Kolmogorov Complexity and Deep Learning: Asymptotically Optimal Description Length Objectives for Transformers
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于机器学习领域，旨在解决神经网络模型复杂性度量问题。通过引入基于Kolmogorov复杂度的最优描述长度目标，提升Transformer的压缩与泛化能力。**

- **链接: [https://arxiv.org/pdf/2509.22445](https://arxiv.org/pdf/2509.22445)**

> **作者:** Peter Shaw; James Cohan; Jacob Eisenstein; Kristina Toutanova
>
> **备注:** ICLR 2026
>
> **摘要:** The Minimum Description Length (MDL) principle offers a formal framework for applying Occam's razor in machine learning. However, its application to neural networks such as Transformers is challenging due to the lack of a principled, universal measure for model complexity. This paper introduces the theoretical notion of asymptotically optimal description length objectives, grounded in the theory of Kolmogorov complexity. We establish that a minimizer of such an objective achieves optimal compression, for any dataset, up to an additive constant, in the limit as model resource bounds increase. We prove that asymptotically optimal objectives exist for Transformers, building on a new demonstration of their computational universality. We further show that such objectives can be tractable and differentiable by constructing and analyzing a variational objective based on an adaptive Gaussian mixture prior. Our empirical analysis shows that this variational objective selects for a low-complexity solution with strong generalization on an algorithmic task, but standard optimizers fail to find such solutions from a random initialization, highlighting key optimization challenges. More broadly, by providing a theoretical framework for identifying description length objectives with strong asymptotic guarantees, we outline a potential path towards training neural networks that achieve greater compression and generalization.
>
---
#### [replaced 068] Causal Effects of Trigger Words in Social Media Discussions: A Large-Scale Case Study about UK Politics on Reddit
- **分类: cs.SI; cs.CL; cs.CY**

- **简介: 该论文属于社会媒体分析任务，研究触发词对政治讨论的影响，旨在揭示其如何加剧在线争论与情绪化反应。**

- **链接: [https://arxiv.org/pdf/2405.10213](https://arxiv.org/pdf/2405.10213)**

> **作者:** Dimosthenis Antypas; Christian Arnold; Nedjma Ousidhoum; Carla Perez Almendros; Jose Camacho-Collados
>
> **备注:** Accepted at WebSci'26
>
> **摘要:** Political debates on social media often escalate quickly, leading to increased engagement as well as more emotional and polarised exchanges. Trigger points (Mau, Lux, and Westheuser 2023) represent moments when individuals feel that their understanding of what is fair, normal, or appropriate in society is being questioned. Analysing Reddit discussions, we examine how trigger points shape online debates and assess their impact on engagement and affect. Our analysis is based on over 100 million comments from subreddits centred on a predefined set of terms identified as trigger words in UK politics. We find that mentions of these terms are associated with higher engagement and increased animosity, including more controversial, negative, angry, and hateful responses. These results position trigger words as a useful concept for modelling and analysing online polarisation.
>
---
#### [replaced 069] REFLEX: Metacognitive Reasoning for Reflective Zero-Shot Robotic Planning with Large Language Models
- **分类: cs.RO; cs.CL**

- **简介: 该论文属于机器人规划任务，旨在解决LLM在零样本环境下执行复杂任务的不足。通过引入元认知机制，提升机器人自主思考与创新解决问题的能力。**

- **链接: [https://arxiv.org/pdf/2505.14899](https://arxiv.org/pdf/2505.14899)**

> **作者:** Wenjie Lin; Jin Wei-Kocsis; Jiansong Zhang; Byung-Cheol Min; Dongming Gan; Paul Asunda; Ragu Athinarayanan
>
> **摘要:** While large language models (LLMs) have shown great potential across various domains, their applications in robotics remain largely limited to static prompt-based behaviors and still face challenges in complex tasks under zero-shot or few-shot settings. Inspired by human metacognitive learning and creative problem-solving, we address this limitation by exploring a fundamental question: Can LLMs be empowered with metacognitive capabilities to reason, reflect, and create, thereby enhancing their ability to perform robotic tasks with minimal demonstrations? In this paper, we present REFLEX, a framework that integrates metacognitive learning into LLM-powered multi-robot collaboration. The system equips the LLM-powered robotic agents with a skill decomposition and self-reflection mechanism that identifies modular skills from prior tasks, reflects on failures in unseen task scenarios, and synthesizes effective new solutions. We propose a more challenging robotic benchmark task and evaluate our framework on the existing benchmark and the novel task. Experimental results show that our metacognitive learning framework significantly outperforms existing baselines. Moreover, we observe that our framework can generate solutions that differ from the ground truth yet still successfully complete the tasks. These findings support our hypothesis that metacognitive learning can foster creativity in robotic planning.
>
---
#### [replaced 070] Death of the Novel(ty): Beyond n-Gram Novelty as a Metric for Textual Creativity
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文属于文本创造力评估任务，旨在解决n-gram新颖性作为创造力指标的不足。通过分析专家对文本新颖性和适当性的评价，验证了n-gram新颖性与创造力的关系，并测试了模型识别创造性表达的能力。**

- **链接: [https://arxiv.org/pdf/2509.22641](https://arxiv.org/pdf/2509.22641)**

> **作者:** Arkadiy Saakyan; Najoung Kim; Smaranda Muresan; Tuhin Chakrabarty
>
> **备注:** ICLR 2026 Camera Ready. 30 pages, 11 figures, 15 tables
>
> **摘要:** N-gram novelty is widely used to evaluate language models' ability to generate text outside of their training data. More recently, it has also been adopted as a metric for measuring textual creativity. However, theoretical work on creativity suggests that this approach may be inadequate, as it does not account for creativity's dual nature: novelty (how original the text is) and appropriateness (how sensical and pragmatic it is). We investigate the relationship between this notion of creativity and n-gram novelty through 8,618 expert writer annotations of novelty, pragmaticality, and sensicality via close reading of human- and AI-generated text. We find that while n-gram novelty is positively associated with expert writer-judged creativity, approximately 91% of top-quartile n-gram novel expressions are not judged as creative, cautioning against relying on n-gram novelty alone. Furthermore, unlike in human-written text, higher n-gram novelty in open-source LLMs correlates with lower pragmaticality. In an exploratory study with frontier closed-source models, we additionally confirm that they are less likely to produce creative expressions than humans. Using our dataset, we test whether zero-shot, few-shot, and finetuned models are able to identify expressions perceived as novel by experts (a positive aspect of writing) or non-pragmatic (a negative aspect). Overall, frontier LLMs exhibit performance much higher than random but leave room for improvement, especially struggling to identify non-pragmatic expressions. We further find that LLM-as-a-Judge novelty ratings align with expert writer preferences in an out-of-distribution dataset, more so than an n-gram based metric.
>
---
#### [replaced 071] DiaBlo: Diagonal Blocks Are Sufficient For Finetuning
- **分类: cs.LG; cs.AI; cs.CL; math.OC**

- **简介: 该论文属于自然语言处理中的模型微调任务，旨在解决PEFT方法与全模型微调性能差距的问题。工作提出DiaBlo，仅更新权重矩阵的对角块，实现高效且稳定的微调。**

- **链接: [https://arxiv.org/pdf/2506.03230](https://arxiv.org/pdf/2506.03230)**

> **作者:** Selcuk Gurses; Aozhong Zhang; Yanxia Deng; Xun Dong; Xin Li; Naigang Wang; Penghang Yin; Zi Yang
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Fine-tuning is a critical step for adapting large language models (LLMs) to domain-specific downstream tasks. To mitigate the substantial computational and memory costs of full-model fine-tuning, Parameter-Efficient Fine-Tuning (PEFT) methods have been proposed to update only a small subset of model parameters. However, performance gaps between PEFT approaches and full-model fine-tuning still exist. In this work, we present DiaBlo, a simple yet effective PEFT approach that updates only the diagonal blocks of selected model weight matrices. Unlike Low-Rank Adaptation (LoRA) and its variants, DiaBlo eliminates the need for low-rank matrix products, thereby avoiding the reliance on auxiliary initialization schemes or customized optimization strategies to improve convergence. This design leads to stable and robust convergence while maintaining comparable memory efficiency and training speed to LoRA. Moreover, we provide theoretical guarantees showing that, under mild low-rank conditions, DiaBlo is more expressive than LoRA in the linear problem and converges to a stationary point of the general nonlinear full fine-tuning. Through extensive experiments across a range of tasks, including commonsense reasoning, arithmetic reasoning, code generation, and safety alignment, we show that fine-tuning only diagonal blocks is sufficient for strong and consistent performance. DiaBlo not only achieves competitive accuracy but also preserves high memory efficiency and fast fine-tuning speed. Codes are available at this https URL.
>
---
#### [replaced 072] Super Research: Answering Highly Complex Questions with Large Language Models through Super Deep and Super Wide Research
- **分类: cs.CL**

- **简介: 该论文提出"Super Research"任务，解决LLM处理复杂问题的能力不足问题，通过结构化分解、广泛检索和深度调查来提升研究能力。**

- **链接: [https://arxiv.org/pdf/2603.00582](https://arxiv.org/pdf/2603.00582)**

> **作者:** Yubo Dong; Nianhao You; Yuxuan Hou; Zixun Sun; Yue Zhang; Liang Zhang; Siyuan Zhao; Hehe Fan
>
> **摘要:** While Large Language Models (LLMs) have demonstrated proficiency in Deep Research or Wide Search, their capacity to solve highly complex questions-those requiring long-horizon planning, massive evidence gathering, and synthesis across heterogeneous sources-remains largely unexplored. We introduce Super Research, a task for complex autonomous research tasks that integrates (i) structured decomposition into a research plan, (ii) super wide retrieval for diverse perspectives, and (iii) super deep investigation to resolve uncertainties through iterative queries. To evaluate this capability, we curated a benchmark of 300 expert-written questions across diverse domains, each requiring up to 100+ retrieval steps and 1,000+ web pages to reconcile conflicting evidence. Super Research produces verifiable reports with fine-grained citations and intermediate artifacts (e.g., outlines and tables) to ensure traceable reasoning. Furthermore, we present a graph-anchored auditing protocol that evaluates Super Research along five dimensions: Coverage, Logical Consistency, Report Utility, Objectivity and Citation Health. While super-complex questions may be infrequent in standard applications, Super Research serves as a critical ceiling evaluation and stress test for LLM capabilities. A model's proficiency within Super Research acts as a powerful proxy for its general research competence; success here suggests the robustness necessary to navigate nearly any subordinate research task. Leaderboard is available at: this https URL
>
---
#### [replaced 073] Evaluating Spoken Language as a Biomarker for Automated Screening of Cognitive Impairment
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于认知障碍自动筛查任务，旨在利用语音生物标志物进行阿尔茨海默病及其相关痴呆的检测与严重程度预测。研究通过机器学习方法分析语音特征，提升诊断的准确性和临床适用性。**

- **链接: [https://arxiv.org/pdf/2501.18731](https://arxiv.org/pdf/2501.18731)**

> **作者:** Maria R. Lima; Alexander Capstick; Fatemeh Geranmayeh; Ramin Nilforooshan; Maja Matarić; Ravi Vaidyanathan; Payam Barnaghi
>
> **备注:** Published in Nature Communications Medicine (2025)
>
> **摘要:** Timely and accurate assessment of cognitive impairment remains a major unmet need. Speech biomarkers offer a scalable, non-invasive, cost-effective solution for automated screening. However, the clinical utility of machine learning (ML) remains limited by interpretability and generalisability to real-world speech datasets. We evaluate explainable ML for screening of Alzheimer's disease and related dementias (ADRD) and severity prediction using benchmark DementiaBank speech (N = 291, 64% female, 69.8 (SD = 8.6) years). We validate generalisability on pilot data collected in-residence (N = 22, 59% female, 76.2 (SD = 8.0) years). To enhance clinical utility, we stratify risk for actionable triage and assess linguistic feature importance. We show that a Random Forest trained on linguistic features for ADRD detection achieves a mean sensitivity of 69.4% (95% confidence interval (CI) = 66.4-72.5) and specificity of 83.3% (78.0-88.7). On pilot data, this model yields a mean sensitivity of 70.0% (58.0-82.0) and specificity of 52.5% (39.3-65.7). For prediction of Mini-Mental State Examination (MMSE) scores, a Random Forest Regressor achieves a mean absolute MMSE error of 3.7 (3.7-3.8), with comparable performance of 3.3 (3.1-3.5) on pilot data. Risk stratification improves specificity by 13% on the test set, offering a pathway for clinical triage. Linguistic features associated with ADRD include increased use of pronouns and adverbs, greater disfluency, reduced analytical thinking, lower lexical diversity, and fewer words that reflect a psychological state of completion. Our predictive modelling shows promise for integration with conversational technology at home to monitor cognitive health and triage higher-risk individuals, enabling early screening and intervention.
>
---
