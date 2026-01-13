# 自然语言处理 cs.CL

- **最新发布 190 篇**

- **更新 101 篇**

## 最新发布

#### [new 001] Controlled Self-Evolution for Algorithmic Code Optimization
- **分类: cs.CL; cs.AI; cs.NE**

- **简介: 该论文属于算法代码优化任务，解决自进化方法效率低的问题。提出CSE框架，通过结构化初始化、反馈驱动进化和层次记忆提升优化效果。**

- **链接: [https://arxiv.org/pdf/2601.07348v1](https://arxiv.org/pdf/2601.07348v1)**

> **作者:** Tu Hu; Ronghao Chen; Shuo Zhang; Jianghao Yin; Mou Xiao Feng; Jingping Liu; Shaolei Zhang; Wenqi Jiang; Yuqi Fang; Sen Hu; Yi Xu; Huacan Wang
>
> **备注:** 27 pages
>
> **摘要:** Self-evolution methods enhance code generation through iterative "generate-verify-refine" cycles, yet existing approaches suffer from low exploration efficiency, failing to discover solutions with superior complexity within limited budgets. This inefficiency stems from initialization bias trapping evolution in poor solution regions, uncontrolled stochastic operations lacking feedback guidance, and insufficient experience utilization across tasks.To address these bottlenecks, we propose Controlled Self-Evolution (CSE), which consists of three key components. Diversified Planning Initialization generates structurally distinct algorithmic strategies for broad solution space coverage. Genetic Evolution replaces stochastic operations with feedback-guided mechanisms, enabling targeted mutation and compositional crossover. Hierarchical Evolution Memory captures both successful and failed experiences at inter-task and intra-task levels.Experiments on EffiBench-X demonstrate that CSE consistently outperforms all baselines across various LLM backbones. Furthermore, CSE achieves higher efficiency from early generations and maintains continuous improvement throughout evolution. Our code is publicly available at https://github.com/QuantaAlpha/EvoControl.
>
---
#### [new 002] Annotating Dimensions of Social Perception in Text: The First Sentence-Level Dataset of Warmth and Competence
- **分类: cs.CL**

- **简介: 该论文提出首个针对温暖与能力的句子级数据集W&C-Sent，用于分析文本中的社会感知维度。任务属于NLP与计算社会科学交叉领域，解决如何在文本中识别信任、亲和力和能力的问题。**

- **链接: [https://arxiv.org/pdf/2601.06316v1](https://arxiv.org/pdf/2601.06316v1)**

> **作者:** Mutaz Ayesh; Saif M. Mohammad; Nedjma Ousidhoum
>
> **摘要:** Warmth (W) (often further broken down into Trust (T) and Sociability (S)) and Competence (C) are central dimensions along which people evaluate individuals and social groups (Fiske, 2018). While these constructs are well established in social psychology, they are only starting to get attention in NLP research through word-level lexicons, which do not completely capture their contextual expression in larger text units and discourse. In this work, we introduce Warmth and Competence Sentences (W&C-Sent), the first sentence-level dataset annotated for warmth and competence. The dataset includes over 1,600 English sentence--target pairs annotated along three dimensions: trust and sociability (components of warmth), and competence. The sentences in W&C-Sent are from social media and often express attitudes and opinions about specific individuals or social groups (the targets of our annotations). We describe the data collection, annotation, and quality-control procedures in detail, and evaluate a range of large language models (LLMs) on their ability to identify trust, sociability, and competence in text. W&C-Sent provides a new resource for analyzing warmth and competence in language and supports future research at the intersection of NLP and computational social science.
>
---
#### [new 003] When Abundance Conceals Weakness: Knowledge Conflict in Multilingual Models
- **分类: cs.CL**

- **简介: 该论文属于多语言模型研究，解决跨语言知识冲突问题。提出CLEAR框架，评估模型在不同任务中的冲突解决能力，发现资源丰富语言在推理任务中占优，而语言亲和力在事实冲突中更关键。**

- **链接: [https://arxiv.org/pdf/2601.07041v1](https://arxiv.org/pdf/2601.07041v1)**

> **作者:** Jiaqi Zhao; Qiang Huang; Haodong Chen; Xiaoxing You; Jun Yu
>
> **备注:** 14 pages, 7 figures, and 4 tables
>
> **摘要:** Large Language Models (LLMs) encode vast world knowledge across multiple languages, yet their internal beliefs are often unevenly distributed across linguistic spaces. When external evidence contradicts these language-dependent memories, models encounter \emph{cross-lingual knowledge conflict}, a phenomenon largely unexplored beyond English-centric settings. We introduce \textbf{CLEAR}, a \textbf{C}ross-\textbf{L}ingual knowl\textbf{E}dge conflict ev\textbf{A}luation f\textbf{R}amework that systematically examines how multilingual LLMs reconcile conflicting internal beliefs and multilingual external evidence. CLEAR decomposes conflict resolution into four progressive scenarios, from multilingual parametric elicitation to competitive multi-source cross-lingual induction, and systematically evaluates model behavior across two complementary QA benchmarks with distinct task characteristics. We construct multilingual versions of ConflictQA and ConflictingQA covering 10 typologically diverse languages and evaluate six representative LLMs. Our experiments reveal a task-dependent decision dichotomy. In reasoning-intensive tasks, conflict resolution is dominated by language resource abundance, with high-resource languages exerting stronger persuasive power. In contrast, for entity-centric factual conflicts, linguistic affinity, not resource scale, becomes decisive, allowing low-resource but linguistically aligned languages to outperform distant high-resource ones.
>
---
#### [new 004] Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出条件记忆机制，解决大语言模型知识检索效率低的问题。通过Engram模块实现快速查找，提升推理和代码数学任务性能。**

- **链接: [https://arxiv.org/pdf/2601.07372v1](https://arxiv.org/pdf/2601.07372v1)**

> **作者:** Xin Cheng; Wangding Zeng; Damai Dai; Qinyu Chen; Bingxuan Wang; Zhenda Xie; Kezhao Huang; Xingkai Yu; Zhewen Hao; Yukun Li; Han Zhang; Huishuai Zhang; Dongyan Zhao; Wenfeng Liang
>
> **摘要:** While Mixture-of-Experts (MoE) scales capacity via conditional computation, Transformers lack a native primitive for knowledge lookup, forcing them to inefficiently simulate retrieval through computation. To address this, we introduce conditional memory as a complementary sparsity axis, instantiated via Engram, a module that modernizes classic $N$-gram embedding for O(1) lookup. By formulating the Sparsity Allocation problem, we uncover a U-shaped scaling law that optimizes the trade-off between neural computation (MoE) and static memory (Engram). Guided by this law, we scale Engram to 27B parameters, achieving superior performance over a strictly iso-parameter and iso-FLOPs MoE baseline. Most notably, while the memory module is expected to aid knowledge retrieval (e.g., MMLU +3.4; CMMLU +4.0), we observe even larger gains in general reasoning (e.g., BBH +5.0; ARC-Challenge +3.7) and code/math domains~(HumanEval +3.0; MATH +2.4). Mechanistic analyses reveal that Engram relieves the backbone's early layers from static reconstruction, effectively deepening the network for complex reasoning. Furthermore, by delegating local dependencies to lookups, it frees up attention capacity for global context, substantially boosting long-context retrieval (e.g., Multi-Query NIAH: 84.2 to 97.0). Finally, Engram establishes infrastructure-aware efficiency: its deterministic addressing enables runtime prefetching from host memory, incurring negligible overhead. We envision conditional memory as an indispensable modeling primitive for next-generation sparse models.
>
---
#### [new 005] Is Sanskrit the most token-efficient language? A quantitative study using GPT, Gemini, and SentencePiece
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在研究梵语在词元效率上的优势。通过对比不同模型和分词器，分析梵语与英语/印地语的词元差异，探索其在编码上的紧凑性及对计算成本的影响。**

- **链接: [https://arxiv.org/pdf/2601.06142v1](https://arxiv.org/pdf/2601.06142v1)**

> **作者:** Anshul Kumar
>
> **备注:** 9 pages, 4 figures. Code and dataset available at: https://github.com/anshulkr713/sanskrit-token-efficiency
>
> **摘要:** Tokens are the basic units of Large Language Models (LLMs). LLMs rely on tokenizers to segment text into these tokens, and tokenization is the primary determinant of computational and inference cost. Sanskrit, one of the oldest languages, is hypothesized to express more meaning per token due to its morphology and grammar rules; however, no prior work has quantified this. We use a dataset of 701 parallel verses of the Bhagavad Gita, which comprises three languages-Sanskrit, English, and Hindi along with transliteration of Sanskrit into English. We test tokenizers including SentencePiece (SPM), older GPT models, and the latest generation tokenizers from Gemini and GPT. We use metrics of token count, characters per token (token efficiency), and tokens per character (token cost). Results show a ~2x difference in token counts between Sanskrit and English/Hindi under the unbiased SPM baseline. English/Hindi translations of Sanskrit commentary resulted in an approximately 20x increase in token count. GPT o200k base (latest, used by GPT-4o) and Gemini (latest) reduce bias by a significant degree compared to GPT cl100k base (used until GPT-4), but still fail to fully capture Sanskrit's compactness. This matters because there might be a penalty bias for non-English users, which inflates the token count. This research provides a foundation for improving future tokenizer design and shows the potential of Sanskrit for highly compact encoding, saving on cost while speeding up training and inference. The code and dataset are available at https://github.com/anshulkr713/sanskrit-token-efficiency
>
---
#### [new 006] InFi-Check: Interpretable and Fine-Grained Fact-Checking of LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于事实核查任务，旨在解决LLM生成内容的幻觉问题。提出InFi-Check框架，实现可解释且细粒度的事实核查，包括错误类型分类与修正。**

- **链接: [https://arxiv.org/pdf/2601.06666v1](https://arxiv.org/pdf/2601.06666v1)**

> **作者:** Yuzhuo Bai; Shuzheng Si; Kangyang Luo; Qingyi Wang; Wenhao Li; Gang Chen; Fanchao Qi; Maosong Sun
>
> **摘要:** Large language models (LLMs) often hallucinate, yet most existing fact-checking methods treat factuality evaluation as a binary classification problem, offering limited interpretability and failing to capture fine-grained error types. In this paper, we introduce InFi-Check, a framework for interpretable and fine-grained fact-checking of LLM outputs. Specifically, we first propose a controlled data synthesis pipeline that generates high-quality data featuring explicit evidence, fine-grained error type labels, justifications, and corrections. Based on this, we further construct large-scale training data and a manually verified benchmark InFi-Check-FG for fine-grained fact-checking of LLM outputs. Building on these high-quality training data, we further propose InFi-Checker, which can jointly provide supporting evidence, classify fine-grained error types, and produce justifications along with corrections. Experiments show that InFi-Checker achieves state-of-the-art performance on InFi-Check-FG and strong generalization across various downstream tasks, significantly improving the utility and trustworthiness of factuality evaluation.
>
---
#### [new 007] ActiShade: Activating Overshadowed Knowledge to Guide Multi-Hop Reasoning in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于多跳推理任务，解决知识遮蔽问题。通过检测并激活被遮蔽的知识，引导模型迭代生成更准确的查询，提升推理效果。**

- **链接: [https://arxiv.org/pdf/2601.07260v1](https://arxiv.org/pdf/2601.07260v1)**

> **作者:** Huipeng Ma; Luan Zhang; Dandan Song; Linmei Hu; Yuhang Tian; Jun Yang; Changzhi Zhou; Chenhao Li; Yizhou Jin; Xudong Li; Meng Lin; Mingxing Zhang; Shuhao Zhang
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** In multi-hop reasoning, multi-round retrieval-augmented generation (RAG) methods typically rely on LLM-generated content as the retrieval query. However, these approaches are inherently vulnerable to knowledge overshadowing - a phenomenon where critical information is overshadowed during generation. As a result, the LLM-generated content may be incomplete or inaccurate, leading to irrelevant retrieval and causing error accumulation during the iteration process. To address this challenge, we propose ActiShade, which detects and activates overshadowed knowledge to guide large language models (LLMs) in multi-hop reasoning. Specifically, ActiShade iteratively detects the overshadowed keyphrase in the given query, retrieves documents relevant to both the query and the overshadowed keyphrase, and generates a new query based on the retrieved documents to guide the next-round iteration. By supplementing the overshadowed knowledge during the formulation of next-round queries while minimizing the introduction of irrelevant noise, ActiShade reduces the error accumulation caused by knowledge overshadowing. Extensive experiments show that ActiShade outperforms existing methods across multiple datasets and LLMs.
>
---
#### [new 008] Solar Open Technical Report
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决 underserved 语言AI模型训练数据不足的问题。通过合成高质量数据、优化数据协调和应用RL框架，构建了高效语言模型Solar Open。**

- **链接: [https://arxiv.org/pdf/2601.07022v1](https://arxiv.org/pdf/2601.07022v1)**

> **作者:** Sungrae Park; Sanghoon Kim; Jungho Cho; Gyoungjin Gim; Dawoon Jung; Mikyoung Cha; Eunhae Choo; Taekgyu Hong; Minbyul Jeong; SeHwan Joo; Minsoo Khang; Eunwon Kim; Minjeong Kim; Sujeong Kim; Yunsu Kim; Hyeonju Lee; Seunghyun Lee; Sukyung Lee; Siyoung Park; Gyungin Shin; Inseo Song; Wonho Song; Seonghoon Yang; Seungyoun Yi; Sanghoon Yoon; Jeonghyun Ko; Seyoung Song; Keunwoo Choi; Hwalsuk Lee; Sunghun Kim; Du-Seong Chang; Kyunghyun Cho; Junsuk Choe; Hwaran Lee; Jae-Gil Lee; KyungTae Lim; Alice Oh
>
> **摘要:** We introduce Solar Open, a 102B-parameter bilingual Mixture-of-Experts language model for underserved languages. Solar Open demonstrates a systematic methodology for building competitive LLMs by addressing three interconnected challenges. First, to train effectively despite data scarcity for underserved languages, we synthesize 4.5T tokens of high-quality, domain-specific, and RL-oriented data. Second, we coordinate this data through a progressive curriculum jointly optimizing composition, quality thresholds, and domain coverage across 20 trillion tokens. Third, to enable reasoning capabilities through scalable RL, we apply our proposed framework SnapPO for efficient optimization. Across benchmarks in English and Korean, Solar Open achieves competitive performance, demonstrating the effectiveness of this methodology for underserved language AI development.
>
---
#### [new 009] IDRBench: Interactive Deep Research Benchmark
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文提出IDRBench，用于评估交互式深度研究系统。解决现有基准缺乏互动评估的问题，通过多智能体框架和用户模拟器，量化交互效果与成本。**

- **链接: [https://arxiv.org/pdf/2601.06676v1](https://arxiv.org/pdf/2601.06676v1)**

> **作者:** Yingchaojie Feng; Qiang Huang; Xiaoya Xie; Zhaorui Yang; Jun Yu; Wei Chen; Anthony K. H. Tung
>
> **摘要:** Deep research agents powered by Large Language Models (LLMs) can perform multi-step reasoning, web exploration, and long-form report generation. However, most existing systems operate in an autonomous manner, assuming fully specified user intent and evaluating only final outputs. In practice, research goals are often underspecified and evolve during exploration, making sustained interaction essential for robust alignment. Despite its importance, interaction remains largely invisible to existing deep research benchmarks, which neither model dynamic user feedback nor quantify its costs. We introduce IDRBench, the first benchmark for systematically evaluating interactive deep research. IDRBench combines a modular multi-agent research framework with on-demand interaction, a scalable reference-grounded user simulator, and an interaction-aware evaluation suite that jointly measures interaction benefits (quality and alignment) and costs (turns and tokens). Experiments across seven state-of-the-art LLMs show that interaction consistently improves research quality and robustness, often outweighing differences in model capacity, while revealing substantial trade-offs in interaction efficiency.
>
---
#### [new 010] Categorize Early, Integrate Late: Divergent Processing Strategies in Automatic Speech Recognition
- **分类: cs.CL**

- **简介: 该论文属于语音识别任务，旨在比较Transformer与Conformer模型的处理策略差异。通过架构指纹分析，发现Conformer早分类，Transformer晚整合，揭示了不同架构的处理机制。**

- **链接: [https://arxiv.org/pdf/2601.06972v1](https://arxiv.org/pdf/2601.06972v1)**

> **作者:** Nathan Roll; Pranav Bhalerao; Martijn Bartelds; Arjun Pawar; Yuka Tatsumi; Tolulope Ogunremi; Chen Shani; Calbert Graham; Meghan Sumner; Dan Jurafsky
>
> **备注:** 3 figures, 9 tables
>
> **摘要:** In speech language modeling, two architectures dominate the frontier: the Transformer and the Conformer. However, it remains unknown whether their comparable performance stems from convergent processing strategies or distinct architectural inductive biases. We introduce Architectural Fingerprinting, a probing framework that isolates the effect of architecture on representation, and apply it to a controlled suite of 24 pre-trained encoders (39M-3.3B parameters). Our analysis reveals divergent hierarchies: Conformers implement a "Categorize Early" strategy, resolving phoneme categories 29% earlier in depth and speaker gender by 16% depth. In contrast, Transformers "Integrate Late," deferring phoneme, accent, and duration encoding to deep layers (49-57%). These fingerprints suggest design heuristics: Conformers' front-loaded categorization may benefit low-latency streaming, while Transformers' deep integration may favor tasks requiring rich context and cross-utterance normalization.
>
---
#### [new 011] TALON: Confidence-Aware Speculative Decoding with Adaptive Token Trees
- **分类: cs.CL**

- **简介: 该论文属于大模型推理加速任务，解决树结构生成中固定结构适应性差的问题。提出TALON框架，动态调整树结构以优化生成效率和质量。**

- **链接: [https://arxiv.org/pdf/2601.07353v1](https://arxiv.org/pdf/2601.07353v1)**

> **作者:** Tianyu Liu; Qitan Lv; Yuhao Shen; Xiao Sun; Xiaoyan Sun
>
> **摘要:** Speculative decoding (SD) has become a standard technique for accelerating LLM inference without sacrificing output quality. Recent advances in speculative decoding have shifted from sequential chain-based drafting to tree-structured generation, where the draft model constructs a tree of candidate tokens to explore multiple possible drafts in parallel. However, existing tree-based SD methods typically build a fixed-width, fixed-depth draft tree, which fails to adapt to the varying difficulty of tokens and contexts. As a result, the draft model cannot dynamically adjust the tree structure to early stop on difficult tokens and extend generation for simple ones. To address these challenges, we introduce TALON, a training-free, budget-driven adaptive tree expansion framework that can be plugged into existing tree-based methods. Unlike static methods, TALON constructs the draft tree iteratively until a fixed token budget is met, using a hybrid expansion strategy that adaptively allocates the node budget to each layer of the draft tree. This framework naturally shapes the draft tree into a "deep-and-narrow" form for deterministic contexts and a "shallow-and-wide" form for uncertain branches, effectively optimizing the trade-off between exploration width and generation depth under a given budget. Extensive experiments across 5 models and 6 datasets demonstrate that TALON consistently outperforms state-of-the-art EAGLE-3, achieving up to 5.16x end-to-end speedup over auto-regressive decoding.
>
---
#### [new 012] BiasLab: A Multilingual, Dual-Framing Framework for Robust Measurement of Output-Level Bias in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型偏见评估任务，旨在解决输出层面偏见的量化难题。提出BiasLab框架，通过双框架设计和多语言支持，实现鲁棒、标准化的偏见测量。**

- **链接: [https://arxiv.org/pdf/2601.06861v1](https://arxiv.org/pdf/2601.06861v1)**

> **作者:** William Guey; Wei Zhang; Pei-Luen Patrick Rau; Pierrick Bougault; Vitor D. de Moura; Bertan Ucar; Jose O. Gomes
>
> **备注:** source code and reproducibility scripts available on GitHub
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed in high-stakes contexts where their outputs influence real-world decisions. However, evaluating bias in LLM outputs remains methodologically challenging due to sensitivity to prompt wording, limited multilingual coverage, and the lack of standardized metrics that enable reliable comparison across models. This paper introduces BiasLab, an open-source, model-agnostic evaluation framework for quantifying output-level (extrinsic) bias through a multilingual, robustness-oriented experimental design. BiasLab constructs mirrored probe pairs under a strict dual-framing scheme: an affirmative assertion favoring Target A and a reverse assertion obtained by deterministic target substitution favoring Target B, while preserving identical linguistic structure. To reduce dependence on prompt templates, BiasLab performs repeated evaluation under randomized instructional wrappers and enforces a fixed-choice Likert response format to maximize comparability across models and languages. Responses are normalized into agreement labels using an LLM-based judge, aligned for polarity consistency across framings, and aggregated into quantitative bias indicators with descriptive statistics including effect sizes and neutrality rates. The framework supports evaluation across diverse bias axes, including demographic, cultural, political, and geopolitical topics, and produces reproducible artifacts such as structured reports and comparative visualizations. BiasLab contributes a standardized methodology for cross-lingual and framing-sensitive bias measurement that complements intrinsic and dataset-based audits, enabling researchers and institutions to benchmark robustness and make better-informed deployment decisions.
>
---
#### [new 013] Structure First, Reason Next: Enhancing a Large Language Model using Knowledge Graph for Numerical Reasoning in Financial Documents
- **分类: cs.CL**

- **简介: 该论文属于金融文本中的数值推理任务，旨在解决LLM在处理财务文档中数值数据时的准确性问题。通过引入知识图谱增强模型推理能力，提升了数值计算的准确率。**

- **链接: [https://arxiv.org/pdf/2601.07754v1](https://arxiv.org/pdf/2601.07754v1)**

> **作者:** Aryan Mishra; Akash Anil
>
> **摘要:** Numerical reasoning is an important task in the analysis of financial documents. It helps in understanding and performing numerical predictions with logical conclusions for the given query seeking answers from financial texts. Recently, Large Language Models (LLMs) have shown promising results in multiple Question-Answering (Q-A) systems with the capability of logical reasoning. As documents related to finance often consist of long and complex financial contexts, LLMs appear well-suited for building high-quality automated financial question-answering systems. However, LLMs often face challenges in accurately processing the various numbers within financial reports. Extracting numerical data from unstructured text and semi-structured tables, and reliably performing accurate calculations, remains a significant bottleneck for numerical reasoning in most state-of-the-art LLMs. Recent studies have shown that structured data augmentations, such as Knowledge Graphs (KGs), have notably improved the predictions of LLMs along with logical explanations. Thus, it is an important requirement to consider inherent structured information in financial reports while using LLMs for various financial analytics. This paper proposes a framework to incorporate structured information using KGs along with LLM predictions for numerical reasoning tasks. The KGs are extracted using a proposed schema inherently from the document under processing. We evaluated our proposed framework over the benchmark data FinQA, using an open-source LLM, namely Llama 3.1 8B Instruct. We observed that the proposed framework improved execution accuracy by approximately 12% relative to the vanilla LLM.
>
---
#### [new 014] What Matters When Building Universal Multilingual Named Entity Recognition Models?
- **分类: cs.CL**

- **简介: 该论文属于多语言命名实体识别任务，旨在解决模型设计决策缺乏系统验证的问题。通过实验分析架构、训练目标和数据组合，提出Otter模型，在多个语言上取得更好效果。**

- **链接: [https://arxiv.org/pdf/2601.06347v1](https://arxiv.org/pdf/2601.06347v1)**

> **作者:** Jonas Golde; Patrick Haller; Alan Akbik
>
> **摘要:** Recent progress in universal multilingual named entity recognition (NER) has been driven by advances in multilingual transformer models and task-specific architectures, loss functions, and training datasets. Despite substantial prior work, we find that many critical design decisions for such models are made without systematic justification, with architectural components, training objectives, and data sources evaluated only in combination rather than in isolation. We argue that these decisions impede progress in the field by making it difficult to identify which choices improve model performance. In this work, we conduct extensive experiments around architectures, transformer backbones, training objectives, and data composition across a wide range of languages. Based on these insights, we introduce Otter, a universal multilingual NER model supporting over 100 languages. Otter achieves consistent improvements over strong multilingual NER baselines, outperforming GLiNER-x-base by 5.3pp in F1 and achieves competitive performance compared to large generative models such as Qwen3-32B, while being substantially more efficient. We release model checkpoints, training and evaluation code to facilitate reproducibility and future research.
>
---
#### [new 015] EpiCaR: Knowing What You Don't Know Matters for Better Reasoning in LLMs
- **分类: cs.CL**

- **简介: 该论文属于大语言模型推理能力提升任务，解决模型过度自信、缺乏不确定性表示的问题。通过引入EpiCaR方法，同时优化推理性能与校准，提升模型可靠性。**

- **链接: [https://arxiv.org/pdf/2601.06786v1](https://arxiv.org/pdf/2601.06786v1)**

> **作者:** Jewon Yeom; Jaewon Sok; Seonghyeon Park; Jeongjae Park; Taesup Kim
>
> **摘要:** Improving the reasoning abilities of large language models (LLMs) has largely relied on iterative self-training with model-generated data. While effective at boosting accuracy, existing approaches primarily reinforce successful reasoning paths, incurring a substantial calibration cost: models become overconfident and lose the ability to represent uncertainty. This failure has been characterized as a form of model collapse in alignment, where predictive distributions degenerate toward low-variance point estimates. We address this issue by reframing reasoning training as an epistemic learning problem, in which models must learn not only how to reason, but also when their reasoning should be trusted. We propose epistemically-calibrated reasoning (EpiCaR) as a training objective that jointly optimizes reasoning performance and calibration, and instantiate it within an iterative supervised fine-tuning framework using explicit self-evaluation signals. Experiments on Llama-3 and Qwen-3 families demonstrate that our approach achieves Pareto-superiority over standard baselines in both accuracy and calibration, particularly in models with sufficient reasoning capacity (e.g., 3B+). This framework generalizes effectively to OOD mathematical reasoning (GSM8K) and code generation (MBPP). Ultimately, our approach enables a 3X reduction in inference compute, matching the K=30 performance of STaR with only K=10 samples in capable models.
>
---
#### [new 016] Reference Games as a Testbed for the Alignment of Model Uncertainty and Clarification Requests
- **分类: cs.CL**

- **简介: 该论文属于语言模型交互任务，旨在解决模型如何识别并表达自身不确定性的问题。通过参考游戏实验，评估模型在不确定时请求澄清的能力。**

- **链接: [https://arxiv.org/pdf/2601.07820v1](https://arxiv.org/pdf/2601.07820v1)**

> **作者:** Manar Ali; Judith Sieker; Sina Zarrieß; Hendrik Buschmeier
>
> **摘要:** In human conversation, both interlocutors play an active role in maintaining mutual understanding. When addressees are uncertain about what speakers mean, for example, they can request clarification. It is an open question for language models whether they can assume a similar addressee role, recognizing and expressing their own uncertainty through clarification. We argue that reference games are a good testbed to approach this question as they are controlled, self-contained, and make clarification needs explicit and measurable. To test this, we evaluate three vision-language models comparing a baseline reference resolution task to an experiment where the models are instructed to request clarification when uncertain. The results suggest that even in such simple tasks, models often struggle to recognize internal uncertainty and translate it into adequate clarification behavior. This demonstrates the value of reference games as testbeds for interaction qualities of (vision and) language models.
>
---
#### [new 017] Symphonym: Universal Phonetic Embeddings for Cross-Script Toponym Matching via Teacher-Student Distillation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Symphonym系统，解决跨文字系统的地名匹配问题。通过师生蒸馏方法，将20种文字的地名映射到统一的语音嵌入空间，提升跨语言地名检索效果。**

- **链接: [https://arxiv.org/pdf/2601.06932v1](https://arxiv.org/pdf/2601.06932v1)**

> **作者:** Stephen Gadd
>
> **备注:** 30 pages, 5 tables, 2 figures
>
> **摘要:** Linking place names across languages and writing systems is a fundamental challenge in digital humanities and geographic information retrieval. Existing approaches rely on language-specific phonetic algorithms or transliteration rules that fail when names cross script boundaries -- no string metric can determine that "Moscow" when rendered in Cyrillic or Arabic refer to the same city. I present Symphonym, a neural embedding system that maps toponyms from 20 writing systems into a unified 128-dimensional phonetic space. A Teacher network trained on articulatory phonetic features (via Epitran and PanPhon) produces target embeddings, while a Student network learns to approximate these from raw characters. At inference, only the lightweight Student (1.7M parameters) is required, enabling deployment without runtime phonetic conversion. Training uses a three-phase curriculum on 57 million toponyms from GeoNames, Wikidata, and the Getty Thesaurus of Geographic Names. Phase 1 trains the Teacher on 467K phonetically-grounded triplets. Phase 2 aligns the Student to Teacher outputs across 23M samples, achieving 96.6% cosine similarity. Phase 3 fine-tunes on 3.3M hard negative triplets -- negatives sharing prefix and script with the anchor but referring to different places -- to sharpen discrimination. Evaluation on the MEHDIE Hebrew-Arabic benchmark achieves 89.2% Recall@1, outperforming Levenshtein (81.5%) and Jaro-Winkler (78.5%). The system is optimised for cross-script matching; same-script variants can be handled by complementary string methods. Symphonym will enable fuzzy phonetic reconciliation and search across the World Historical Gazetteer's 67 million toponyms. Code and models are publicly available.
>
---
#### [new 018] LitVISTA: A Benchmark for Narrative Orchestration in Literary Text
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文学叙事分析任务，旨在解决模型生成叙事与人类叙事在结构上的不匹配问题。提出VISTA空间和LitVISTA基准，评估模型的叙事编排能力。**

- **链接: [https://arxiv.org/pdf/2601.06445v1](https://arxiv.org/pdf/2601.06445v1)**

> **作者:** Mingzhe Lu; Yiwen Wang; Yanbing Liu; Qi You; Chong Liu; Ruize Qin; Haoyu Dong; Wenyu Zhang; Jiarui Zhang; Yue Hu; Yunpeng Li
>
> **摘要:** Computational narrative analysis aims to capture rhythm, tension, and emotional dynamics in literary texts. Existing large language models can generate long stories but overly focus on causal coherence, neglecting the complex story arcs and orchestration inherent in human narratives. This creates a structural misalignment between model- and human-generated narratives. We propose VISTA Space, a high-dimensional representational framework for narrative orchestration that unifies human and model narrative perspectives. We further introduce LitVISTA, a structurally annotated benchmark grounded in literary texts, enabling systematic evaluation of models' narrative orchestration capabilities. We conduct oracle evaluations on a diverse selection of frontier LLMs, including GPT, Claude, Grok, and Gemini. Results reveal systematic deficiencies: existing models fail to construct a unified global narrative view, struggling to jointly capture narrative function and structure. Furthermore, even advanced thinking modes yield only limited gains for such literary narrative understanding.
>
---
#### [new 019] Structured Episodic Event Memory
- **分类: cs.CL**

- **简介: 该论文属于记忆建模任务，旨在解决LLM在复杂推理中缺乏结构化记忆的问题。提出SEEM框架，结合图记忆与动态情节记忆，提升叙事连贯性与逻辑一致性。**

- **链接: [https://arxiv.org/pdf/2601.06411v1](https://arxiv.org/pdf/2601.06411v1)**

> **作者:** Zhengxuan Lu; Dongfang Li; Yukun Shi; Beilun Wang; Longyue Wang; Baotian Hu
>
> **摘要:** Current approaches to memory in Large Language Models (LLMs) predominantly rely on static Retrieval-Augmented Generation (RAG), which often results in scattered retrieval and fails to capture the structural dependencies required for complex reasoning. For autonomous agents, these passive and flat architectures lack the cognitive organization necessary to model the dynamic and associative nature of long-term interaction. To address this, we propose Structured Episodic Event Memory (SEEM), a hierarchical framework that synergizes a graph memory layer for relational facts with a dynamic episodic memory layer for narrative progression. Grounded in cognitive frame theory, SEEM transforms interaction streams into structured Episodic Event Frames (EEFs) anchored by precise provenance pointers. Furthermore, we introduce an agentic associative fusion and Reverse Provenance Expansion (RPE) mechanism to reconstruct coherent narrative contexts from fragmented evidence. Experimental results on the LoCoMo and LongMemEval benchmarks demonstrate that SEEM significantly outperforms baselines, enabling agents to maintain superior narrative coherence and logical consistency.
>
---
#### [new 020] The Confidence Dichotomy: Analyzing and Mitigating Miscalibration in Tool-Use Agents
- **分类: cs.CL**

- **简介: 该论文属于人工智能领域，研究工具使用代理的置信度校准问题。针对工具类型导致的置信度偏差，提出强化学习方法提升校准效果。**

- **链接: [https://arxiv.org/pdf/2601.07264v1](https://arxiv.org/pdf/2601.07264v1)**

> **作者:** Weihao Xuan; Qingcheng Zeng; Heli Qi; Yunze Xiao; Junjue Wang; Naoto Yokoya
>
> **摘要:** Autonomous agents based on large language models (LLMs) are rapidly evolving to handle multi-turn tasks, but ensuring their trustworthiness remains a critical challenge. A fundamental pillar of this trustworthiness is calibration, which refers to an agent's ability to express confidence that reliably reflects its actual performance. While calibration is well-established for static models, its dynamics in tool-integrated agentic workflows remain underexplored. In this work, we systematically investigate verbalized calibration in tool-use agents, revealing a fundamental confidence dichotomy driven by tool type. Specifically, our pilot study identifies that evidence tools (e.g., web search) systematically induce severe overconfidence due to inherent noise in retrieved information, while verification tools (e.g., code interpreters) can ground reasoning through deterministic feedback and mitigate miscalibration. To robustly improve calibration across tool types, we propose a reinforcement learning (RL) fine-tuning framework that jointly optimizes task accuracy and calibration, supported by a holistic benchmark of reward designs. We demonstrate that our trained agents not only achieve superior calibration but also exhibit robust generalization from local training environments to noisy web settings and to distinct domains such as mathematical reasoning. Our results highlight the necessity of domain-specific calibration strategies for tool-use agents. More broadly, this work establishes a foundation for building self-aware agents that can reliably communicate uncertainty in high-stakes, real-world deployments.
>
---
#### [new 021] Learning Through Dialogue: Unpacking the Dynamics of Human-LLM Conversations on Political Issues
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于人机交互任务，研究如何通过对话促进用户学习政治知识。分析了397次对话，探讨LLM解释对用户知识和信心的影响机制。**

- **链接: [https://arxiv.org/pdf/2601.07796v1](https://arxiv.org/pdf/2601.07796v1)**

> **作者:** Shaz Furniturewala; Gerard Christopher Yeo; Kokil Jaidka
>
> **摘要:** Large language models (LLMs) are increasingly used as conversational partners for learning, yet the interactional dynamics supporting users' learning and engagement are understudied. We analyze the linguistic and interactional features from both LLM and participant chats across 397 human-LLM conversations about socio-political issues to identify the mechanisms and conditions under which LLM explanations shape changes in political knowledge and confidence. Mediation analyses reveal that LLM explanatory richness partially supports confidence by fostering users' reflective insight, whereas its effect on knowledge gain operates entirely through users' cognitive engagement. Moderation analyses show that these effects are highly conditional and vary by political efficacy. Confidence gains depend on how high-efficacy users experience and resolve uncertainty. Knowledge gains depend on high-efficacy users' ability to leverage extended interaction, with longer conversations benefiting primarily reflective users. In summary, we find that learning from LLMs is an interactional achievement, not a uniform outcome of better explanations. The findings underscore the importance of aligning LLM explanatory behavior with users' engagement states to support effective learning in designing Human-AI interactive systems.
>
---
#### [new 022] Enhancing Self-Correction in Large Language Models through Multi-Perspective Reflection
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在推理中的一致性、准确性和自我修正问题。提出PR-CoT方法，通过多角度反思提升模型推理质量。**

- **链接: [https://arxiv.org/pdf/2601.07780v1](https://arxiv.org/pdf/2601.07780v1)**

> **作者:** Mariana Costa; Alberlucia Rafael Soarez; Daniel Kim; Camila Ferreira
>
> **摘要:** While Chain-of-Thought (CoT) prompting advances LLM reasoning, challenges persist in consistency, accuracy, and self-correction, especially for complex or ethically sensitive tasks. Existing single-dimensional reflection methods offer insufficient improvements. We propose MyGO Poly-Reflective Chain-of-Thought (PR-CoT), a novel methodology employing structured multi-perspective reflection. After initial CoT, PR-CoT guides the LLM to self-assess its reasoning across multiple predefined angles: logical consistency, information completeness, biases/ethics, and alternative solutions. Implemented purely via prompt engineering, this process refines the initial CoT into a more robust and accurate final answer without model retraining. Experiments across arithmetic, commonsense, ethical decision-making, and logical puzzles, using GPT-three point five and GPT-four models, demonstrate PR-CoT's superior performance. It significantly outperforms traditional CoT and existing reflection methods in logical consistency and error correction, with notable gains in nuanced domains like ethical decision-making. Ablation studies, human evaluations, and qualitative analyses further validate the contribution of each reflection perspective and the overall efficacy of our poly-reflective paradigm in fostering more reliable LLM reasoning.
>
---
#### [new 023] PsyCLIENT: Client Simulation via Conversational Trajectory Modeling for Trainee Practice and Model Evaluation in Mental Health Counseling
- **分类: cs.CL**

- **简介: 该论文提出PsyCLIENT，用于心理辅导模拟，解决客户多样性不足、行为建模不科学及中文数据缺失问题。通过对话轨迹建模实现真实互动，提升训练效果。**

- **链接: [https://arxiv.org/pdf/2601.07312v1](https://arxiv.org/pdf/2601.07312v1)**

> **作者:** Huachuan Qiu; Zhaoming Chen; Yuqian Chen; Yuan Xie; Yu Lu; Zhenzhong Lan
>
> **摘要:** LLM-based client simulation has emerged as a promising tool for training novice counselors and evaluating automated counseling systems. However, existing client simulation approaches face three key challenges: (1) limited diversity and realism in client profiles, (2) the lack of a principled framework for modeling realistic client behaviors, and (3) a scarcity in Chinese-language settings. To address these limitations, we propose PsyCLIENT, a novel simulation framework grounded in conversational trajectory modeling. By conditioning LLM generation on predefined real-world trajectories that incorporate explicit behavior labels and content constraints, our approach ensures diverse and realistic interactions. We further introduce PsyCLIENT-CP, the first open-source Chinese client profile dataset, covering 60 distinct counseling topics. Comprehensive evaluations involving licensed professional counselors demonstrate that PsyCLIENT significantly outperforms baselines in terms of authenticity and training effectiveness. Notably, the simulated clients are nearly indistinguishable from human clients, achieving an about 95\% expert confusion rate in discrimination tasks. These findings indicate that conversational trajectory modeling effectively bridges the gap between theoretical client profiles and dynamic, realistic simulations, offering a robust solution for mental health education and research. Code and data will be released to facilitate future research in mental health counseling.
>
---
#### [new 024] CSR-RAG: An Efficient Retrieval System for Text-to-SQL on the Enterprise Scale
- **分类: cs.CL**

- **简介: 该论文属于Text-to-SQL任务，解决企业级数据库中高效检索问题。提出CSR-RAG系统，结合上下文、结构和关系检索，提升准确率与效率。**

- **链接: [https://arxiv.org/pdf/2601.06564v1](https://arxiv.org/pdf/2601.06564v1)**

> **作者:** Rajpreet Singh; Novak Boškov; Lawrence Drabeck; Aditya Gudal; Manzoor A. Khan
>
> **摘要:** Natural language to SQL translation (Text-to-SQL) is one of the long-standing problems that has recently benefited from advances in Large Language Models (LLMs). While most academic Text-to-SQL benchmarks request schema description as a part of natural language input, enterprise-scale applications often require table retrieval before SQL query generation. To address this need, we propose a novel hybrid Retrieval Augmented Generation (RAG) system consisting of contextual, structural, and relational retrieval (CSR-RAG) to achieve computationally efficient yet sufficiently accurate retrieval for enterprise-scale databases. Through extensive enterprise benchmarks, we demonstrate that CSR-RAG achieves up to 40% precision and over 80% recall while incurring a negligible average query generation latency of only 30ms on commodity data center hardware, which makes it appropriate for modern LLM-based enterprise-scale systems.
>
---
#### [new 025] PlaM: Training-Free Plateau-Guided Model Merging for Better Visual Grounding in MLLMs
- **分类: cs.CL**

- **简介: 该论文属于多模态任务，解决MLLMs在微调后语言推理能力下降的问题。通过训练-free方法，提出plateau-guided模型融合，提升视觉定位性能。**

- **链接: [https://arxiv.org/pdf/2601.07645v1](https://arxiv.org/pdf/2601.07645v1)**

> **作者:** Zijing Wang; Yongkang Liu; Mingyang Wang; Ercong Nie; Deyuan Chen; Zhengjie Zhao; Shi Feng; Daling Wang; Xiaocui Yang; Yifei Zhang; Hinrich Schütze
>
> **备注:** under review
>
> **摘要:** Multimodal Large Language Models (MLLMs) rely on strong linguistic reasoning inherited from their base language models. However, multimodal instruction fine-tuning paradoxically degrades this text's reasoning capability, undermining multimodal performance. To address this issue, we propose a training-free framework to mitigate this degradation. Through layer-wise vision token masking, we reveal a common three-stage pattern in multimodal large language models: early-modal separation, mid-modal alignment, and late-modal degradation. By analyzing the behavior of MLLMs at different stages, we propose a plateau-guided model merging method that selectively injects base language model parameters into MLLMs. Experimental results based on five MLLMs on nine benchmarks demonstrate the effectiveness of our method. Attention-based analysis further reveals that merging shifts attention from diffuse, scattered patterns to focused localization on task-relevant visual regions. Our repository is on https://github.com/wzj1718/PlaM.
>
---
#### [new 026] Value of Information: A Framework for Human-Agent Communication
- **分类: cs.CL**

- **简介: 该论文属于人机交互任务，解决代理在信息不全时是否询问用户的问题。提出基于信息价值的决策框架，动态权衡询问收益与用户成本，无需调参，适应多种场景。**

- **链接: [https://arxiv.org/pdf/2601.06407v1](https://arxiv.org/pdf/2601.06407v1)**

> **作者:** Yijiang River Dong; Tiancheng Hu; Zheng Hui; Caiqi Zhang; Ivan Vulić; Andreea Bobu; Nigel Collier
>
> **摘要:** Large Language Model (LLM) agents deployed for real-world tasks face a fundamental dilemma: user requests are underspecified, yet agents must decide whether to act on incomplete information or interrupt users for clarification. Existing approaches either rely on brittle confidence thresholds that require task-specific tuning, or fail to account for the varying stakes of different decisions. We introduce a decision-theoretic framework that resolves this trade-off through the Value of Information (VoI), enabling agents to dynamically weigh the expected utility gain from asking questions against the cognitive cost imposed on users. Our inference-time method requires no hyperparameter tuning and adapts seamlessly across contexts-from casual games to medical diagnosis. Experiments across four diverse domains (20 Questions, medical diagnosis, flight booking, and e-commerce) show that VoI consistently matches or exceeds the best manually-tuned baselines, achieving up to 1.36 utility points higher in high-cost settings. This work provides a parameter-free framework for adaptive agent communication that explicitly balances task risk, query ambiguity, and user effort.
>
---
#### [new 027] Two Pathways to Truthfulness: On the Intrinsic Encoding of LLM Hallucinations
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM hallucinations问题。通过分析两种信息路径，揭示truthfulness的内部编码机制，并提出提升检测性能的应用。**

- **链接: [https://arxiv.org/pdf/2601.07422v1](https://arxiv.org/pdf/2601.07422v1)**

> **作者:** Wen Luo; Guangyue Peng; Wei Li; Shaohang Wei; Feifan Song; Liang Wang; Nan Yang; Xingxing Zhang; Jing Jin; Furu Wei; Houfeng Wang
>
> **摘要:** Despite their impressive capabilities, large language models (LLMs) frequently generate hallucinations. Previous work shows that their internal states encode rich signals of truthfulness, yet the origins and mechanisms of these signals remain unclear. In this paper, we demonstrate that truthfulness cues arise from two distinct information pathways: (1) a Question-Anchored pathway that depends on question-answer information flow, and (2) an Answer-Anchored pathway that derives self-contained evidence from the generated answer itself. First, we validate and disentangle these pathways through attention knockout and token patching. Afterwards, we uncover notable and intriguing properties of these two mechanisms. Further experiments reveal that (1) the two mechanisms are closely associated with LLM knowledge boundaries; and (2) internal representations are aware of their distinctions. Finally, building on these insightful findings, two applications are proposed to enhance hallucination detection performance. Overall, our work provides new insight into how LLMs internally encode truthfulness, offering directions for more reliable and self-aware generative systems.
>
---
#### [new 028] On the Fallacy of Global Token Perplexity in Spoken Language Model Evaluation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语音语言模型评估任务，指出传统全局token困惑度评估方法存在缺陷，提出新的评估方法以更准确反映生成质量。**

- **链接: [https://arxiv.org/pdf/2601.06329v1](https://arxiv.org/pdf/2601.06329v1)**

> **作者:** Jeff Chan-Jan Sju; Liang-Hsuan Tseng; Yi-Cheng Lin; Yen-Chun Kuo; Ju-Chieh Chou; Kai-Wei Chang; Hung-yi Lee; Carlos Busso
>
> **摘要:** Generative spoken language models pretrained on large-scale raw audio can continue a speech prompt with appropriate content while preserving attributes like speaker and emotion, serving as foundation models for spoken dialogue. In prior literature, these models are often evaluated using ``global token perplexity'', which directly applies the text perplexity formulation to speech tokens. However, this practice overlooks fundamental differences between speech and text modalities, possibly leading to an underestimation of the speech characteristics. In this work, we propose a variety of likelihood- and generative-based evaluation methods that serve in place of naive global token perplexity. We demonstrate that the proposed evaluations more faithfully reflect perceived generation quality, as evidenced by stronger correlations with human-rated mean opinion scores (MOS). When assessed under the new metrics, the relative performance landscape of spoken language models is reshaped, revealing a significantly reduced gap between the best-performing model and the human topline. Together, these results suggest that appropriate evaluation is critical for accurately assessing progress in spoken language modeling.
>
---
#### [new 029] Multi-Stage Evolutionary Model Merging with Meta Data Driven Curriculum Learning for Sentiment-Specialized Large Language Modeling
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于情感分析任务，旨在提升多任务情感分析的准确性。提出MEM-MCL模型，结合进化模型融合与元数据驱动的课程学习，优化模型性能。**

- **链接: [https://arxiv.org/pdf/2601.06780v1](https://arxiv.org/pdf/2601.06780v1)**

> **作者:** Keito Inoshita; Xiaokang Zhou; Akira Kawai
>
> **备注:** This paper was presented at the 10th IEEE International Conference on Data Science and Systems in December 2024 and is awaiting publication
>
> **摘要:** The emergence of large language models (LLMs) has significantly transformed natural language processing (NLP), enabling more generalized models to perform various tasks with minimal training. However, traditional sentiment analysis methods, which focus on individual tasks such as sentiment classification or aspect-based analysis, are not practical for real-world applications that usually require handling multiple tasks. While offering flexibility, LLMs in sentiment-specific tasks often fall short of the required accuracy. Techniques like fine-tuning and evolutionary model merging help integrate models into a unified framework, which can improve the learning performance while reducing computational costs. The use of task meta-data and curriculum learning to optimize learning processes remains underexplored, while sentiment analysis is a critical task in NLP that requires high accuracy and scalability across multiple subtasks. In this study, we propose a hybrid learning model called Multi-stage Evolutionary Model Merging with Meta data driven Curriculum Learning (MEM-MCL), to enhance the sentiment analysis in large language modeling. In particular, expert models are created through instruction tuning for specific sentiment tasks and then merged using evolutionary algorithms to form a unified model. The merging process is optimized with weak data to enhance performance across tasks. The curriculum learning is incorporated to provide a learning sequence based on task difficulty, improving knowledge extraction from LLMs. Experiment results demonstrate that the proposed MEM-MCL model outperforms conventional LLMs in a majority of sentiment analysis tasks, achieving superior results across various subtasks.
>
---
#### [new 030] ReMIND: Orchestrating Modular Large Language Models for Controllable Serendipity A REM-Inspired System Design for Emergent Creative Ideation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出ReMIND系统，解决LLM中生成新颖且连贯创意的难题。通过模块化设计，分离探索与整合，提升创意生成质量。属于创意生成任务。**

- **链接: [https://arxiv.org/pdf/2601.07121v1](https://arxiv.org/pdf/2601.07121v1)**

> **作者:** Makoto Sato
>
> **摘要:** Large language models (LLMs) are used not only for problem solving but also for creative ideation; however, eliciting serendipitous insights that are both novel and internally coherent remains difficult. While stochastic sampling promotes novelty, it often degrades consistency. Here, we propose ReMIND, a REM-inspired modular framework for ideation. ReMIND consists of four stages: wake, which generates a stable low-temperature semantic baseline; dream, which performs high-temperature exploratory generation; judge, which applies coarse evaluation to filter incoherent outputs and extract candidate ideas; and re-wake, which re-articulates selected ideas into coherent final outputs. By instantiating each stage as an independent LLM, ReMIND enables functional separation between exploration and consolidation. Parameter sweeps show that ReMIND reliably induces semantic exploration while preserving downstream stability. Embedding-based analyses confirm substantial semantic displacement during the dream phase, whereas external evaluations reveal that high-quality ideas emerge sporadically rather than as extrema along any single metric. These results suggest that serendipitous ideation in LLMs is a rare-event process best approached through system level design that shapes the conditions under which valuable ideas can emerge and be stabilized. ReMIND provides a general framework for studying the computational basis of serendipity and illustrates how modular LLM orchestration can bridge exploration and stabilization.
>
---
#### [new 031] EVM-QuestBench: An Execution-Grounded Benchmark for Natural-Language Transaction Code Generation
- **分类: cs.CL**

- **简介: 该论文提出EVM-QuestBench，用于评估自然语言到交易代码的生成任务，解决区块链交易中执行准确性和安全性问题。**

- **链接: [https://arxiv.org/pdf/2601.06565v1](https://arxiv.org/pdf/2601.06565v1)**

> **作者:** Pei Yang; Wanyi Chen; Ke Wang; Lynn Ai; Eric Yang; Tianyu Shi
>
> **备注:** 10 pages, 13 figures
>
> **摘要:** Large language models are increasingly applied to various development scenarios. However, in on-chain transaction scenarios, even a minor error can cause irreversible loss for users. Existing evaluations often overlook execution accuracy and safety. We introduce EVM-QuestBench, an execution-grounded benchmark for natural-language transaction-script generation on EVM-compatible chains. The benchmark employs dynamic evaluation: instructions are sampled from template pools, numeric parameters are drawn from predefined intervals, and validators verify outcomes against these instantiated values. EVM-QuestBench contains 107 tasks (62 atomic, 45 composite). Its modular architecture enables rapid task development. The runner executes scripts on a forked EVM chain with snapshot isolation; composite tasks apply step-efficiency decay. We evaluate 20 models and find large performance gaps, with split scores revealing persistent asymmetry between single-action precision and multi-step workflow completion. Code: https://anonymous.4open.science/r/bsc_quest_bench-A9CF/.
>
---
#### [new 032] Towards Computational Chinese Paleography
- **分类: cs.CL**

- **简介: 论文探讨计算中文古文字学的发展，旨在解决古文字识别与研究中的技术挑战。通过分析数字资源和方法流程，提出构建多模态、少样本的人机协作系统。**

- **链接: [https://arxiv.org/pdf/2601.06753v1](https://arxiv.org/pdf/2601.06753v1)**

> **作者:** Yiran Rex Ma
>
> **备注:** A position paper in progress with Peking University & ByteDance Digital Humanities Open Lab
>
> **摘要:** Chinese paleography, the study of ancient Chinese writing, is undergoing a computational turn powered by artificial intelligence. This position paper charts the trajectory of this emerging field, arguing that it is evolving from automating isolated visual tasks to creating integrated digital ecosystems for scholarly research. We first map the landscape of digital resources, analyzing critical datasets for oracle bone, bronze, and bamboo slip scripts. The core of our analysis follows the field's methodological pipeline: from foundational visual processing (image restoration, character recognition), through contextual analysis (artifact rejoining, dating), to the advanced reasoning required for automated decipherment and human-AI collaboration. We examine the technological shift from classical computer vision to modern deep learning paradigms, including transformers and large multimodal models. Finally, we synthesize the field's core challenges -- notably data scarcity and a disconnect between current AI capabilities and the holistic nature of humanistic inquiry -- and advocate for a future research agenda focused on creating multimodal, few-shot, and human-centric systems to augment scholarly expertise.
>
---
#### [new 033] Doing More with Less: Data Augmentation for Sudanese Dialect Automatic Speech Recognition
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语音识别任务，针对低资源苏丹语方言的ASR系统进行研究，通过数据增强技术提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.06802v1](https://arxiv.org/pdf/2601.06802v1)**

> **作者:** Ayman Mansour
>
> **摘要:** Although many Automatic Speech Recognition (ASR) systems have been developed for Modern Standard Arabic (MSA) and Dialectal Arabic (DA), few studies have focused on dialect-specific implementations, particularly for low-resource Arabic dialects such as Sudanese. This paper presents a comprehensive study of data augmentation techniques for fine-tuning OpenAI Whisper models and establishes the first benchmark for the Sudanese dialect. Two augmentation strategies are investigated: (1) self-training with pseudo-labels generated from unlabeled speech, and (2) TTS-based augmentation using synthetic speech from the Klaam TTS system. The best-performing model, Whisper-Medium fine-tuned with combined self-training and TTS augmentation (28.4 hours), achieves a Word Error Rate (WER) of 57.1% on the evaluation set and 51.6% on an out-of-domain holdout set substantially outperforming zero-shot multilingual Whisper (78.8% WER) and MSA-specialized Arabic models (73.8-123% WER). All experiments used low-cost resources (Kaggle free tier and Lightning.ai trial), demonstrating that strategic data augmentation can overcome resource limitations for low-resource dialects and provide a practical roadmap for developing ASR systems for low-resource Arabic dialects and other marginalized language varieties. The models, evaluation benchmarks, and reproducible training pipelines are publicly released to facilitate future research on low-resource Arabic ASR.
>
---
#### [new 034] LLMs Can't Play Hangman: On the Necessity of a Private Working Memory for Language Agents
- **分类: cs.CL**

- **简介: 该论文研究语言代理在交互任务中的状态维护问题，提出私有工作内存的必要性。针对PSIT任务，证明传统模型无法同时保持秘密与一致，设计测试验证并提出新架构解决此问题。**

- **链接: [https://arxiv.org/pdf/2601.06973v1](https://arxiv.org/pdf/2601.06973v1)**

> **作者:** Davide Baldelli; Ali Parviz; Amal Zouaq; Sarath Chandar
>
> **摘要:** As LLMs move from text completion toward autonomous agents, they remain constrained by the standard chat interface, which lacks private working memory. This raises a fundamental question: can agents reliably perform interactive tasks that depend on hidden state? We define Private State Interactive Tasks (PSITs), which require agents to generate and maintain hidden information while producing consistent public responses. We show theoretically that any agent restricted to the public conversation history cannot simultaneously preserve secrecy and consistency in PSITs, yielding an impossibility theorem. To empirically validate this limitation, we introduce a self-consistency testing protocol that evaluates whether agents can maintain a hidden secret across forked dialogue branches. Standard chat-based LLMs and retrieval-based memory baselines fail this test regardless of scale, demonstrating that semantic retrieval does not enable true state maintenance. To address this, we propose a novel architecture incorporating an explicit private working memory; we demonstrate that this mechanism restores consistency, establishing private state as a necessary component for interactive language agents.
>
---
#### [new 035] Can Large Language Models Understand, Reason About, and Generate Code-Switched Text?
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型在代码混用文本的理解、推理和生成能力，旨在解决多语言环境下模型的鲁棒性问题。通过构建基准数据集，评估模型表现并揭示其局限性。**

- **链接: [https://arxiv.org/pdf/2601.07153v1](https://arxiv.org/pdf/2601.07153v1)**

> **作者:** Genta Indra Winata; David Anugraha; Patrick Amadeus Irawan; Anirban Das; Haneul Yoo; Paresh Dashore; Shreyas Kulkarni; Ruochen Zhang; Haruki Sakajo; Frederikus Hudi; Anaelia Ovalle; Syrielle Montariol; Felix Gaschi; Michael Anugraha; Rutuj Ravindra Puranik; Zawad Hayat Ahmed; Adril Putra Merin; Emmanuele Chersoni
>
> **备注:** Preprint
>
> **摘要:** Code-switching is a pervasive phenomenon in multilingual communication, yet the robustness of large language models (LLMs) in mixed-language settings remains insufficiently understood. In this work, we present a comprehensive evaluation of LLM capabilities in understanding, reasoning over, and generating code-switched text. We introduce CodeMixQA a novel benchmark with high-quality human annotations, comprising 16 diverse parallel code-switched language-pair variants that span multiple geographic regions and code-switching patterns, and include both original scripts and their transliterated forms. Using this benchmark, we analyze the reasoning behavior of LLMs on code-switched question-answering tasks, shedding light on how models process and reason over mixed-language inputs. We further conduct a systematic evaluation of LLM-generated synthetic code-switched text, focusing on both naturalness and semantic fidelity, and uncover key limitations in current generation capabilities. Our findings reveal persistent challenges in both reasoning and generation under code-switching conditions and provide actionable insights for building more robust multilingual LLMs. We release the dataset and code as open source.
>
---
#### [new 036] Forest Before Trees: Latent Superposition for Efficient Visual Reasoning
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于视觉推理任务，解决视觉细节丢失与语义坍塌问题。提出Laser方法，通过动态对齐学习实现高效推理，提升性能并减少计算量。**

- **链接: [https://arxiv.org/pdf/2601.06803v1](https://arxiv.org/pdf/2601.06803v1)**

> **作者:** Yubo Wang; Juntian Zhang; Yichen Wu; Yankai Lin; Nils Lukas; Yuhan Liu
>
> **摘要:** While Chain-of-Thought empowers Large Vision-Language Models with multi-step reasoning, explicit textual rationales suffer from an information bandwidth bottleneck, where continuous visual details are discarded during discrete tokenization. Recent latent reasoning methods attempt to address this challenge, but often fall prey to premature semantic collapse due to rigid autoregressive objectives. In this paper, we propose Laser, a novel paradigm that reformulates visual deduction via Dynamic Windowed Alignment Learning (DWAL). Instead of forcing a point-wise prediction, Laser aligns the latent state with a dynamic validity window of future semantics. This mechanism enforces a "Forest-before-Trees" cognitive hierarchy, enabling the model to maintain a probabilistic superposition of global features before narrowing down to local details. Crucially, Laser maintains interpretability via decodable trajectories while stabilizing unconstrained learning via Self-Refined Superposition. Extensive experiments on 6 benchmarks demonstrate that Laser achieves state-of-the-art performance among latent reasoning methods, surpassing the strong baseline Monet by 5.03% on average. Notably, it achieves these gains with extreme efficiency, reducing inference tokens by more than 97%, while demonstrating robust generalization to out-of-distribution domains.
>
---
#### [new 037] Kinship Data Benchmark for Multi-hop Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出KinshipQA基准，用于评估大语言模型的多跳推理能力。通过生成具有文化特性的家族树数据，构建推理任务，测试模型在复杂关系中的推理表现。**

- **链接: [https://arxiv.org/pdf/2601.07794v1](https://arxiv.org/pdf/2601.07794v1)**

> **作者:** Tianda Sun; Dimitar Kazakov
>
> **备注:** 11 pages, 2 figures, 9 tables
>
> **摘要:** Large language models (LLMs) are increasingly evaluated on their ability to perform multi-hop reasoning, i.e., to combine multiple pieces of information into a coherent inference. We introduce KinshipQA, a benchmark designed to probe this capability through reasoning over kinship relations. The central contribution of our work is a generative pipeline that produces, on demand, large-scale, realistic, and culture-specific genealogical data: collections of interconnected family trees that satisfy explicit marriage constraints associated with different kinship systems. This allows task difficulty, cultural assumptions, and relational depth to be systematically controlled and varied. From these genealogies, we derive textual inference tasks that require reasoning over implicit relational chains. We evaluate the resulting benchmark using six state-of-the-art LLMs, spanning both open-source and closed-source models, under a uniform zero-shot protocol with deterministic decoding. Performance is measured using exact-match and set-based metrics. Our results demonstrate that KinshipQA yields a wide spread of outcomes and exposes systematic differences in multi-hop reasoning across models and cultural settings.
>
---
#### [new 038] The Confidence Trap: Gender Bias and Predictive Certainty in LLMs
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理中的公平性评估任务，旨在解决LLMs在性别偏见下的置信度校准问题。通过分析模型置信度与人类标注偏见的对齐情况，提出新指标Gender-ECE以衡量性别差异。**

- **链接: [https://arxiv.org/pdf/2601.07806v1](https://arxiv.org/pdf/2601.07806v1)**

> **作者:** Ahmed Sabir; Markus Kängsepp; Rajesh Sharma
>
> **备注:** AAAI 2026 (AISI Track), Oral. Project page: https://bit.ly/4p8OKQD
>
> **摘要:** The increased use of Large Language Models (LLMs) in sensitive domains leads to growing interest in how their confidence scores correspond to fairness and bias. This study examines the alignment between LLM-predicted confidence and human-annotated bias judgments. Focusing on gender bias, the research investigates probability confidence calibration in contexts involving gendered pronoun resolution. The goal is to evaluate if calibration metrics based on predicted confidence scores effectively capture fairness-related disparities in LLMs. The results show that, among the six state-of-the-art models, Gemma-2 demonstrates the worst calibration according to the gender bias benchmark. The primary contribution of this work is a fairness-aware evaluation of LLMs' confidence calibration, offering guidance for ethical deployment. In addition, we introduce a new calibration metric, Gender-ECE, designed to measure gender disparities in resolution tasks.
>
---
#### [new 039] Engineering of Hallucination in Generative AI: It's not a Bug, it's a Feature
- **分类: cs.CL**

- **简介: 该论文探讨生成式AI中的幻觉现象，属于自然语言处理任务。它提出幻觉可作为功能而非缺陷，通过概率工程引导模型适度幻觉以获得更好结果。**

- **链接: [https://arxiv.org/pdf/2601.07046v1](https://arxiv.org/pdf/2601.07046v1)**

> **作者:** Tim Fingscheidt; Patrick Blumenberg; Björn Möller
>
> **备注:** This is an article that has been written reflecting a talk of Tim Fingscheidt at the 2025 New Year gathering of Braunschweigische Wissenschaftliche Gesellschaft on January 25th, 2025
>
> **摘要:** Generative artificial intelligence (AI) is conquering our lives at lightning speed. Large language models such as ChatGPT answer our questions or write texts for us, large computer vision models such as GAIA-1 generate videos on the basis of text descriptions or continue prompted videos. These neural network models are trained using large amounts of text or video data, strictly according to the real data employed in training. However, there is a surprising observation: When we use these models, they only function satisfactorily when they are allowed a certain degree of fantasy (hallucination). While hallucination usually has a negative connotation in generative AI - after all, ChatGPT is expected to give a fact-based answer! - this article recapitulates some simple means of probability engineering that can be used to encourage generative AI to hallucinate to a limited extent and thus lead to the desired results. We have to ask ourselves: Is hallucination in gen-erative AI probably not a bug, but rather a feature?
>
---
#### [new 040] Pragya: An AI-Based Semantic Recommendation System for Sanskrit Subhasitas
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出Pragya系统，解决梵文箴言在数字时代因语言和语境障碍而被忽视的问题。通过检索生成框架实现语义推荐，提升相关性和可访问性。**

- **链接: [https://arxiv.org/pdf/2601.06607v1](https://arxiv.org/pdf/2601.06607v1)**

> **作者:** Tanisha Raorane; Prasenjit Kole
>
> **备注:** Preprint
>
> **摘要:** Sanskrit Subhasitas encapsulate centuries of cultural and philosophical wisdom, yet remain underutilized in the digital age due to linguistic and contextual barriers. In this work, we present Pragya, a retrieval-augmented generation (RAG) framework for semantic recommendation of Subhasitas. We curate a dataset of 200 verses annotated with thematic tags such as motivation, friendship, and compassion. Using sentence embeddings (IndicBERT), the system retrieves top-k verses relevant to user queries. The retrieved results are then passed to a generative model (Mistral LLM) to produce transliterations, translations, and contextual explanations. Experimental evaluation demonstrates that semantic retrieval significantly outperforms keyword matching in precision and relevance, while user studies highlight improved accessibility through generated summaries. To our knowledge, this is the first attempt at integrating retrieval and generation for Sanskrit Subhasitas, bridging cultural heritage with modern applied AI.
>
---
#### [new 041] Can a Unimodal Language Agent Provide Preferences to Tune a Multimodal Vision-Language Model?
- **分类: cs.CL**

- **简介: 论文探讨将多模态能力扩展到单模态大语言模型的方法，通过让单模态语言代理提供反馈优化视觉-语言模型。任务是提升多模态模型的描述能力，实验显示效果显著。**

- **链接: [https://arxiv.org/pdf/2601.06424v1](https://arxiv.org/pdf/2601.06424v1)**

> **作者:** Sazia Tabasum Mim; Jack Morris; Manish Dhakal; Yanming Xiu; Maria Gorlatova; Yi Ding
>
> **备注:** Accepted to IJCNLP-AACL 2025 Findings
>
> **摘要:** To explore a more scalable path for adding multimodal capabilities to existing LLMs, this paper addresses a fundamental question: Can a unimodal LLM, relying solely on text, reason about its own informational needs and provide effective feedback to optimize a multimodal model? To answer this, we propose a method that enables a language agent to give feedback to a vision-language model (VLM) to adapt text generation to the agent's preferences. Our results from different experiments affirm this hypothesis, showing that LLM preference feedback significantly enhances VLM descriptions. Using our proposed method, we find that the VLM can generate multimodal scene descriptions to help the LLM better understand multimodal context, leading to improvements of maximum 13% in absolute accuracy compared to the baseline multimodal approach. Furthermore, a human study validated our AI-driven feedback, showing a 64.6% preference alignment rate between the LLM's choices and human judgments. Extensive experiments provide insights on how and why the method works and its limitations.
>
---
#### [new 042] Interpretable Text Classification Applied to the Detection of LLM-generated Creative Writing
- **分类: cs.CL**

- **简介: 该论文属于文本分类任务，旨在区分人类创作与LLM生成的文学文本。通过机器学习模型实现高精度分类，并分析其特征以解释分类依据。**

- **链接: [https://arxiv.org/pdf/2601.07368v1](https://arxiv.org/pdf/2601.07368v1)**

> **作者:** Minerva Suvanto; Andrea McGlinchey; Mattias Wahde; Peter J Barclay
>
> **备注:** Accepted for publication at ICAART 2026 (https://icaart.scitevents.org/?y=2026)
>
> **摘要:** We consider the problem of distinguishing human-written creative fiction (excerpts from novels) from similar text generated by an LLM. Our results show that, while human observers perform poorly (near chance levels) on this binary classification task, a variety of machine-learning models achieve accuracy in the range 0.93 - 0.98 over a previously unseen test set, even using only short samples and single-token (unigram) features. We therefore employ an inherently interpretable (linear) classifier (with a test accuracy of 0.98), in order to elucidate the underlying reasons for this high accuracy. In our analysis, we identify specific unigram features indicative of LLM-generated text, one of the most important being that the LLM tends to use a larger variety of synonyms, thereby skewing the probability distributions in a manner that is easy to detect for a machine learning classifier, yet very difficult for a human observer. Four additional explanation categories were also identified, namely, temporal drift, Americanisms, foreign language usage, and colloquialisms. As identification of the AI-generated text depends on a constellation of such features, the classification appears robust, and therefore not easy to circumvent by malicious actors intent on misrepresenting AI-generated text as human work.
>
---
#### [new 043] Adaptive Layer Selection for Layer-Wise Token Pruning in LLM Inference
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于大语言模型推理优化任务，解决KV缓存冗余问题。提出ASL方法，自适应选择层进行令牌剪枝，提升精度并保持速度。**

- **链接: [https://arxiv.org/pdf/2601.07667v1](https://arxiv.org/pdf/2601.07667v1)**

> **作者:** Rei Taniguchi; Yuyang Dong; Makoto Onizuka; Chuan Xiao
>
> **备注:** Source code is available at https://github.com/TANIGUCHIREI/ASL
>
> **摘要:** Due to the prevalence of large language models (LLMs), key-value (KV) cache reduction for LLM inference has received remarkable attention. Among numerous works that have been proposed in recent years, layer-wise token pruning approaches, which select a subset of tokens at particular layers to retain in KV cache and prune others, are one of the most popular schemes. They primarily adopt a set of pre-defined layers, at which tokens are selected. Such design is inflexible in the sense that the accuracy significantly varies across tasks and deteriorates in harder tasks such as KV retrieval. In this paper, we propose ASL, a training-free method that adaptively chooses the selection layer for KV cache reduction, exploiting the variance of token ranks ordered by attention score. The proposed method balances the performance across different tasks while meeting the user-specified KV budget requirement. ASL operates during the prefilling stage and can be jointly used with existing KV cache reduction methods such as SnapKV to optimize the decoding stage. By evaluations on the InfiniteBench, RULER, and NIAH benchmarks, we show that equipped with one-shot token selection, where tokens are selected at a layer and propagated to deeper layers, ASL outperforms state-of-the-art layer-wise token selection methods in accuracy while maintaining decoding speed and KV cache reduction.
>
---
#### [new 044] Garbage Attention in Large Language Models: BOS Sink Heads and Sink-aware Pruning
- **分类: cs.CL**

- **简介: 该论文属于模型压缩任务，解决LLM中组件冗余问题。通过分析BOS sink现象，提出基于注意力头的剪枝方法，有效识别冗余结构并保持性能。**

- **链接: [https://arxiv.org/pdf/2601.06787v1](https://arxiv.org/pdf/2601.06787v1)**

> **作者:** Jaewon Sok; Jewon Yeom; Seonghyeon Park; Jeongjae Park; Taesup Kim
>
> **摘要:** Large Language Models (LLMs) are known to contain significant redundancy, yet a systematic explanation for why certain components, particularly in higher layers, are more redundant has remained elusive. In this work, we identify the BOS sink phenomenon as a key mechanism driving this layer-wise sensitivity. We show that attention heads with high BOS sink scores are strongly associated with functional redundancy: such heads, especially in deeper layers, contribute little to predictive performance and effectively serve as \emph{dumping grounds} for superfluous attention weights. This provides a concrete functional explanation for the structural redundancy reported in prior studies. Leveraging this insight, we introduce a simple pruning strategy that removes high-BOS sink heads. Experiments on Gemma-3, Llama-3.1, and Qwen3 demonstrate that this approach identifies redundant transformer components more reliably than weight- or activation-based criteria, while preserving performance close to dense baselines even under aggressive pruning. Moreover, we find that the behavior of sink heads remains stable across different sequence lengths. Overall, our results suggest that structural properties of attention offer a more intuitive and robust basis for model compression than magnitude-based methods.
>
---
#### [new 045] SyntaxMind at BLP-2025 Task 1: Leveraging Attention Fusion of CNN and GRU for Hate Speech Detection
- **分类: cs.CL**

- **简介: 该论文属于仇恨言论检测任务，解决孟加拉语文本中的仇恨言论分类问题。通过融合CNN与GRU的注意力机制，提升分类效果。**

- **链接: [https://arxiv.org/pdf/2601.06306v1](https://arxiv.org/pdf/2601.06306v1)**

> **作者:** Md. Shihab Uddin Riad
>
> **摘要:** This paper describes our system used in the BLP-2025 Task 1: Hate Speech Detection. We participated in Subtask 1A and Subtask 1B, addressing hate speech classification in Bangla text. Our approach employs a unified architecture that integrates BanglaBERT embeddings with multiple parallel processing branches based on GRUs and CNNs, followed by attention and dense layers for final classification. The model is designed to capture both contextual semantics and local linguistic cues, enabling robust performance across subtasks. The proposed system demonstrated high competitiveness, obtaining 0.7345 micro F1-Score (2nd place) in Subtask 1A and 0.7317 micro F1-Score (5th place) in Subtask 1B.
>
---
#### [new 046] CIRAG: Construction-Integration Retrieval and Adaptive Generation for Multi-hop Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多跳问答任务，旨在解决现有方法在证据链构建和粒度控制上的不足。提出CIRAG模型，通过迭代构建与整合证据链，实现更准确的多跳推理。**

- **链接: [https://arxiv.org/pdf/2601.06799v1](https://arxiv.org/pdf/2601.06799v1)**

> **作者:** Zili Wei; Xiaocui Yang; Yilin Wang; Zihan Wang; Weidong Bao; Shi Feng; Daling Wang; Yifei Zhang
>
> **摘要:** Triple-based Iterative Retrieval-Augmented Generation (iRAG) mitigates document-level noise for multi-hop question answering. However, existing methods still face limitations: (i) greedy single-path expansion, which propagates early errors and fails to capture parallel evidence from different reasoning branches, and (ii) granularity-demand mismatch, where a single evidence representation struggles to balance noise control with contextual sufficiency. In this paper, we propose the Construction-Integration Retrieval and Adaptive Generation model, CIRAG. It introduces an Iterative Construction-Integration module that constructs candidate triples and history-conditionally integrates them to distill core triples and generate the next-hop query. This module mitigates the greedy trap by preserving multiple plausible evidence chains. Besides, we propose an Adaptive Cascaded Multi-Granularity Generation module that progressively expands contextual evidence based on the problem requirements, from triples to supporting sentences and full passages. Moreover, we introduce Trajectory Distillation, which distills the teacher model's integration policy into a lightweight student, enabling efficient and reliable long-horizon reasoning. Extensive experiments demonstrate that CIRAG achieves superior performance compared to existing iRAG methods.
>
---
#### [new 047] What makes for an enjoyable protagonist? An analysis of character warmth and competence
- **分类: cs.CL**

- **简介: 该论文属于内容分析任务，旨在探究电影主角的温暖与能力是否影响观众评分。研究使用AI标注分析2858部作品，发现两者与评分有微弱关联，但效果有限。**

- **链接: [https://arxiv.org/pdf/2601.06658v1](https://arxiv.org/pdf/2601.06658v1)**

> **作者:** Hannes Rosenbusch
>
> **摘要:** Drawing on psychological and literary theory, we investigated whether the warmth and competence of movie protagonists predict IMDb ratings, and whether these effects vary across genres. Using 2,858 films and series from the Movie Scripts Corpus, we identified protagonists via AI-assisted annotation and quantified their warmth and competence with the LLM_annotate package ([1]; human-LLM agreement: r = .83). Preregistered Bayesian regression analyses revealed theory-consistent but small associations between both warmth and competence and audience ratings, while genre-specific interactions did not meaningfully improve predictions. Male protagonists were slightly less warm than female protagonists, and movies with male leads received higher ratings on average (an association that was multiple times stronger than the relationships between movie ratings and warmth/competence). These findings suggest that, although audiences tend to favor warm, competent characters, the effects on movie evaluations are modest, indicating that character personality is only one of many factors shaping movie ratings. AI-assisted annotation with LLM_annotate and gpt-4.1-mini proved effective for large-scale analyses but occasionally fell short of manually generated annotations.
>
---
#### [new 048] Relink: Constructing Query-Driven Evidence Graph On-the-Fly for GraphRAG
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识增强的问答任务，旨在解决GraphRAG中知识图谱不完整和噪声干扰的问题。提出Relink框架，动态构建查询相关的证据图，提升问答准确率。**

- **链接: [https://arxiv.org/pdf/2601.07192v1](https://arxiv.org/pdf/2601.07192v1)**

> **作者:** Manzong Huang; Chenyang Bu; Yi He; Xingrui Zhuo; Xindong Wu
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Graph-based Retrieval-Augmented Generation (GraphRAG) mitigates hallucinations in Large Language Models (LLMs) by grounding them in structured knowledge. However, current GraphRAG methods are constrained by a prevailing \textit{build-then-reason} paradigm, which relies on a static, pre-constructed Knowledge Graph (KG). This paradigm faces two critical challenges. First, the KG's inherent incompleteness often breaks reasoning paths. Second, the graph's low signal-to-noise ratio introduces distractor facts, presenting query-relevant but misleading knowledge that disrupts the reasoning process. To address these challenges, we argue for a \textit{reason-and-construct} paradigm and propose Relink, a framework that dynamically builds a query-specific evidence graph. To tackle incompleteness, \textbf{Relink} instantiates required facts from a latent relation pool derived from the original text corpus, repairing broken paths on the fly. To handle misleading or distractor facts, Relink employs a unified, query-aware evaluation strategy that jointly considers candidates from both the KG and latent relations, selecting those most useful for answering the query rather than relying on their pre-existence. This empowers Relink to actively discard distractor facts and construct the most faithful and precise evidence path for each query. Extensive experiments on five Open-Domain Question Answering benchmarks show that Relink achieves significant average improvements of 5.4\% in EM and 5.2\% in F1 over leading GraphRAG baselines, demonstrating the superiority of our proposed framework.
>
---
#### [new 049] Proof of Time: A Benchmark for Evaluating Scientific Idea Judgments
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出PoT框架，用于评估大语言模型在科学想法判断上的能力。任务是解决模型判断质量难以评估的问题，通过未来可验证的信号进行基准测试。**

- **链接: [https://arxiv.org/pdf/2601.07606v1](https://arxiv.org/pdf/2601.07606v1)**

> **作者:** Bingyang Ye; Shan Chen; Jingxuan Tu; Chen Liu; Zidi Xiong; Samuel Schmidgall; Danielle S. Bitterman
>
> **备注:** under review
>
> **摘要:** Large language models are increasingly being used to assess and forecast research ideas, yet we lack scalable ways to evaluate the quality of models' judgments about these scientific ideas. Towards this goal, we introduce PoT, a semi-verifiable benchmarking framework that links scientific idea judgments to downstream signals that become observable later (e.g., citations and shifts in researchers' agendas). PoT freezes a pre-cutoff snapshot of evidence in an offline sandbox and asks models to forecast post-cutoff outcomes, enabling verifiable evaluation when ground truth arrives, scalable benchmarking without exhaustive expert annotation, and analysis of human-model misalignment against signals such as peer-review awards. In addition, PoT provides a controlled testbed for agent-based research judgments that evaluate scientific ideas, comparing tool-using agents to non-agent baselines under prompt ablations and budget scaling. Across 30,000+ instances spanning four benchmark domains, we find that, compared with non-agent baselines, higher interaction budgets generally improve agent performance, while the benefit of tool use is strongly task-dependent. By combining time-partitioned, future-verifiable targets with an offline sandbox for tool use, PoT supports scalable evaluation of agents on future-facing scientific idea judgment tasks.
>
---
#### [new 050] DiffER: Diffusion Entity-Relation Modeling for Reversal Curse in Diffusion Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决Diffusion LLMs中的“反转诅咒”问题。通过实体感知训练和数据优化，提出DiffER模型以提升双向关系建模能力。**

- **链接: [https://arxiv.org/pdf/2601.07347v1](https://arxiv.org/pdf/2601.07347v1)**

> **作者:** Shaokai He; Kaiwen Wei; Xinyi Zeng; Xiang Chen; Xue Yang; Zhenyang Li; Jiang Zhong; Yu Tian
>
> **摘要:** The "reversal curse" refers to the phenomenon where large language models (LLMs) exhibit predominantly unidirectional behavior when processing logically bidirectional relationships. Prior work attributed this to autoregressive training -- predicting the next token inherently favors left-to-right information flow over genuine bidirectional knowledge associations. However, we observe that Diffusion LLMs (DLLMs), despite being trained bidirectionally, also suffer from the reversal curse. To investigate the root causes, we conduct systematic experiments on DLLMs and identify three key reasons: 1) entity fragmentation during training, 2) data asymmetry, and 3) missing entity relations. Motivated by the analysis of these reasons, we propose Diffusion Entity-Relation Modeling (DiffER), which addresses the reversal curse through entity-aware training and balanced data construction. Specifically, DiffER introduces whole-entity masking, which mitigates entity fragmentation by predicting complete entities in a single step. DiffER further employs distribution-symmetric and relation-enhanced data construction strategies to alleviate data asymmetry and missing relations. Extensive experiments demonstrate that DiffER effectively alleviates the reversal curse in Diffusion LLMs, offering new perspectives for future research.
>
---
#### [new 051] Structured Reasoning for Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型推理效率低的问题。通过结构化推理框架，提升推理效率和自检能力，减少冗余步骤。**

- **链接: [https://arxiv.org/pdf/2601.07180v1](https://arxiv.org/pdf/2601.07180v1)**

> **作者:** Jinyi Han; Zixiang Di; Zishang Jiang; Ying Liao; Jiaqing Liang; Yongqi Wang; Yanghua Xiao
>
> **摘要:** Large language models (LLMs) achieve strong performance by generating long chains of thought, but longer traces always introduce redundant or ineffective reasoning steps. One typical behavior is that they often perform unnecessary verification and revisions even if they have reached the correct answers. This limitation stems from the unstructured nature of reasoning trajectories and the lack of targeted supervision for critical reasoning abilities. To address this, we propose Structured Reasoning (SCR), a framework that decouples reasoning trajectories into explicit, evaluable, and trainable components. We mainly implement SCR using a Generate-Verify-Revise paradigm. Specifically, we construct structured training data and apply Dynamic Termination Supervision to guide the model in deciding when to terminate reasoning. To avoid interference between learning signals for different reasoning abilities, we adopt a progressive two-stage reinforcement learning strategy: the first stage targets initial generation and self-verification, and the second stage focuses on revision. Extensive experiments on three backbone models show that SCR substantially improves reasoning efficiency and self-verification. Besides, compared with existing reasoning paradigms, it reduces output token length by up to 50%.
>
---
#### [new 052] Average shortest-path length in word-adjacency networks: Chinese versus English
- **分类: cs.CL**

- **简介: 该论文研究中文与英文词邻接网络的平均最短路径长度，探讨语言结构差异。任务为语言网络分析，解决跨语言结构比较问题，通过构建网络模型并对比不同情况下的路径长度。**

- **链接: [https://arxiv.org/pdf/2601.06361v1](https://arxiv.org/pdf/2601.06361v1)**

> **作者:** Jakub Dec; Michał Dolina; Stanisław Drożdż; Jarosław Kwapień; Jin Liu; Tomasz Stanisz
>
> **摘要:** Complex networks provide powerful tools for analyzing and understanding the intricate structures present in various systems, including natural language. Here, we analyze topology of growing word-adjacency networks constructed from Chinese and English literary works written in different periods. Unconventionally, instead of considering dictionary words only, we also include punctuation marks as if they were ordinary words. Our approach is based on two arguments: (1) punctuation carries genuine information related to emotional state, allows for logical grouping of content, provides a pause in reading, and facilitates understanding by avoiding ambiguity, and (2) our previous works have shown that punctuation marks behave like words in a Zipfian analysis and, if considered together with regular words, can improve authorship attribution in stylometric studies. We focus on a functional dependence of the average shortest path length $L(N)$ on a network size $N$ for different epochs and individual novels in their original language as well as for translations of selected novels into the other language. We approximate the empirical results with a growing network model and obtain satisfactory agreement between the two. We also observe that $L(N)$ behaves asymptotically similar for both languages if punctuation marks are included but becomes sizably larger for Chinese if punctuation marks are neglected.
>
---
#### [new 053] Semantic Compression of LLM Instructions via Symbolic Metalanguages
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在通过符号语言压缩大模型指令。工作是设计MetaGlyph，用数学符号替代文字指令，减少token使用并提升效率。**

- **链接: [https://arxiv.org/pdf/2601.07354v1](https://arxiv.org/pdf/2601.07354v1)**

> **作者:** Ernst van Gassen
>
> **备注:** 12 pages and 6 tables
>
> **摘要:** We introduce MetaGlyph, a symbolic language for compressing prompts by encoding instructions as mathematical symbols rather than prose. Unlike systems requiring explicit decoding rules, MetaGlyph uses symbols like $\in$ (membership) and $\Rightarrow$ (implication) that models already understand from their training data. We test whether these symbols work as ''instruction shortcuts'' that models can interpret without additional teaching. We evaluate eight models across two dimensions relevant to practitioners: scale (3B-1T parameters) and accessibility (open-source for local deployment vs. proprietary APIs). MetaGlyph achieves 62-81% token reduction across all task types. For API-based deployments, this translates directly to cost savings; for local deployments, it reduces latency and memory pressure. Results vary by model. Gemini 2.5 Flash achieves 75% semantic equivalence between symbolic and prose instructions on selection tasks, with 49.9% membership operator fidelity. Kimi K2 reaches 98.1% fidelity for implication ($\Rightarrow$) and achieves perfect (100%) accuracy on selection tasks with symbolic prompts. GPT-5.2 Chat shows the highest membership fidelity observed (91.3%), though with variable parse success across task types. Claude Haiku 4.5 achieves 100% parse success with 26% membership fidelity. Among mid-sized models, Qwen 2.5 7B shows 62% equivalence on extraction tasks. Mid-sized open-source models (7B-12B) show near-zero operator fidelity, suggesting a U-shaped relationship where sufficient scale overcomes instruction-tuning biases.
>
---
#### [new 054] AfriqueLLM: How Data Mixing and Model Architecture Impact Continued Pre-training for African Languages
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在提升非洲语言的预训练模型性能。通过持续预训练和数据组合优化，改进模型在数学推理等任务上的表现。**

- **链接: [https://arxiv.org/pdf/2601.06395v1](https://arxiv.org/pdf/2601.06395v1)**

> **作者:** Hao Yu; Tianyi Xu; Michael A. Hedderich; Wassim Hamidouche; Syed Waqas Zamir; David Ifeoluwa Adelani
>
> **摘要:** Large language models (LLMs) are increasingly multilingual, yet open models continue to underperform relative to proprietary systems, with the gap most pronounced for African languages. Continued pre-training (CPT) offers a practical route to language adaptation, but improvements on demanding capabilities such as mathematical reasoning often remain limited. This limitation is driven in part by the uneven domain coverage and missing task-relevant knowledge that characterize many low-resource language corpora. We present \texttt{AfriqueLLM}, a suite of open LLMs adapted to 20 African languages through CPT on 26B tokens. We perform a comprehensive empirical study across five base models spanning sizes and architectures, including Llama 3.1, Gemma 3, and Qwen 3, and systematically analyze how CPT data composition shapes downstream performance. In particular, we vary mixtures that include math, code, and synthetic translated data, and evaluate the resulting models on a range of multilingual benchmarks. Our results identify data composition as the primary driver of CPT gains. Adding math, code, and synthetic translated data yields consistent improvements, including on reasoning-oriented evaluations. Within a fixed architecture, larger models typically improve performance, but architectural choices dominate scale when comparing across model families. Moreover, strong multilingual performance in the base model does not reliably predict post-CPT outcomes; robust architectures coupled with task-aligned data provide a more dependable recipe. Finally, our best models improve long-context performance, including document-level translation. Models have been released on [Huggingface](https://huggingface.co/collections/McGill-NLP/afriquellm).
>
---
#### [new 055] UETQuintet at BioCreative IX - MedHopQA: Enhancing Biomedical QA with Selective Multi-hop Reasoning and Contextual Retrieval
- **分类: cs.CL**

- **简介: 该论文属于生物医学问答任务，解决复杂医疗查询和多跳推理问题。提出模型结合多源信息检索与上下文学习，提升问答准确性。**

- **链接: [https://arxiv.org/pdf/2601.06974v1](https://arxiv.org/pdf/2601.06974v1)**

> **作者:** Quoc-An Nguyen; Thi-Minh-Thu Vu; Bich-Dat Nguyen; Dinh-Quang-Minh Tran; Hoang-Quynh Le
>
> **备注:** Accepted at the BioCreative IX Challenge and Workshop (BC9) at IJCAI
>
> **摘要:** Biomedical Question Answering systems play a critical role in processing complex medical queries, yet they often struggle with the intricate nature of medical data and the demand for multi-hop reasoning. In this paper, we propose a model designed to effectively address both direct and sequential questions. While sequential questions are decomposed into a chain of sub-questions to perform reasoning across a chain of steps, direct questions are processed directly to ensure efficiency and minimise processing overhead. Additionally, we leverage multi-source information retrieval and in-context learning to provide rich, relevant context for generating answers. We evaluated our model on the BioCreative IX - MedHopQA Shared Task datasets. Our approach achieves an Exact Match score of 0.84, ranking second on the current leaderboard. These results highlight the model's capability to meet the challenges of Biomedical Question Answering, offering a versatile solution for advancing medical research and practice.
>
---
#### [new 056] TeleMem: Building Long-Term and Multimodal Memory for Agentic AI
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出TeleMem，解决长对话中记忆保持与多模态理解问题，通过结构化写入和动态提取提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2601.06037v1](https://arxiv.org/pdf/2601.06037v1)**

> **作者:** Chunliang Chen; Ming Guan; Xiao Lin; Jiaxu Li; Qiyi Wang; Xiangyu Chen; Jixiang Luo; Changzhi Sun; Dell Zhang; Xuelong Li
>
> **摘要:** Large language models (LLMs) excel at many NLP tasks but struggle to sustain long-term interactions due to limited attention over extended dialogue histories. Retrieval-augmented generation (RAG) mitigates this issue but lacks reliable mechanisms for updating or refining stored memories, leading to schema-driven hallucinations, inefficient write operations, and minimal support for multimodal reasoning.To address these challenges, we propose TeleMem, a unified long-term and multimodal memory system that maintains coherent user profiles through narrative dynamic extraction, ensuring that only dialogue-grounded information is preserved. TeleMem further introduces a structured writing pipeline that batches, retrieves, clusters, and consolidates memory entries, substantially improving storage efficiency, reducing token usage, and accelerating memory operations. Additionally, a multimodal memory module combined with ReAct-style reasoning equips the system with a closed-loop observe, think, and act process that enables accurate understanding of complex video content in long-term contexts. Experimental results show that TeleMem surpasses the state-of-the-art Mem0 baseline with 19% higher accuracy, 43% fewer tokens, and a 2.1x speedup on the ZH-4O long-term role-play gaming benchmark.
>
---
#### [new 057] A Rising Tide Lifts All Boats: MTQE Rewards for Idioms Improve General Translation Quality
- **分类: cs.CL**

- **简介: 该论文属于机器翻译任务，旨在解决非合成表达（如成语）的翻译难题。通过使用MTQE模型作为奖励函数进行微调，提升了模型对成语及一般文本的翻译能力。**

- **链接: [https://arxiv.org/pdf/2601.06307v1](https://arxiv.org/pdf/2601.06307v1)**

> **作者:** Ishika Agarwal; Zhenlin He; Dhruva Patil; Dilek Hakkani-Tür
>
> **摘要:** Non-compositional expressions (e.g., idioms, proverbs, and metaphors) pose significant challenges for neural machine translation systems because their meanings cannot be derived from individual words alone. These expressions encode rich, cultural meaning, and have both figurative and literal meanings, making accurate translation difficult. Because models are fairly good at translating compositional text, we investigate GRPO-style fine-tuning using Machine Translation Quality Estimation (MTQE) models as reward functions to train models to better translate idioms. Using Chinese and Hindi idiom datasets, we find that idiom translation abilities improve by ~14 points, general, non-idiomatic translation implicitly improves by ~8 points, and cross-lingual translation abilities (trained on one language, evaluated on another) improves by ~6 points. Overall, our work quantifies the non-compositional translation gap and offers insights for developing LLMs with stronger cross-cultural and figurative language understanding.
>
---
#### [new 058] Talking to Extraordinary Objects: Folktales Offer Analogies for Interacting with Technology
- **分类: cs.CL**

- **简介: 该论文属于人机交互研究，探讨如何从民间故事中获取语言互动的灵感，以解决技术交互中过度拟人化的问题。**

- **链接: [https://arxiv.org/pdf/2601.06372v1](https://arxiv.org/pdf/2601.06372v1)**

> **作者:** Martha Larson
>
> **摘要:** Speech and language are valuable for interacting with technology. It would be ideal to be able to decouple their use from anthropomorphization, which has recently met an important moment of reckoning. In the world of folktales, language is everywhere and talking to extraordinary objects is not unusual. This overview presents examples of the analogies that folktales offer. Extraordinary objects in folktales are diverse and also memorable. Language capacity and intelligence are not always connected to humanness. Consideration of folktales can offer inspiration and insight for using speech and language for interacting with technology.
>
---
#### [new 059] RealMem: Benchmarking LLMs in Real-World Memory-Driven Interaction
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决LLMs在长期项目交互中的记忆管理问题。提出RealMem基准，模拟真实项目场景，评估模型的长期状态与上下文依赖能力。**

- **链接: [https://arxiv.org/pdf/2601.06966v1](https://arxiv.org/pdf/2601.06966v1)**

> **作者:** Haonan Bian; Zhiyuan Yao; Sen Hu; Zishan Xu; Shaolei Zhang; Yifu Guo; Ziliang Yang; Xueran Han; Huacan Wang; Ronghao Chen
>
> **摘要:** As Large Language Models (LLMs) evolve from static dialogue interfaces to autonomous general agents, effective memory is paramount to ensuring long-term consistency. However, existing benchmarks primarily focus on casual conversation or task-oriented dialogue, failing to capture **"long-term project-oriented"** interactions where agents must track evolving goals. To bridge this gap, we introduce **RealMem**, the first benchmark grounded in realistic project scenarios. RealMem comprises over 2,000 cross-session dialogues across eleven scenarios, utilizing natural user queries for evaluation. We propose a synthesis pipeline that integrates Project Foundation Construction, Multi-Agent Dialogue Generation, and Memory and Schedule Management to simulate the dynamic evolution of memory. Experiments reveal that current memory systems face significant challenges in managing the long-term project states and dynamic context dependencies inherent in real-world projects. Our code and datasets are available at [https://github.com/AvatarMemory/RealMemBench](https://github.com/AvatarMemory/RealMemBench).
>
---
#### [new 060] MedRAGChecker: Claim-Level Verification for Biomedical Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文属于生物医学问答任务，旨在解决RAG生成答案中存在不支持或矛盾声明的问题。工作包括提出MedRAGChecker框架，进行逐条验证与诊断。**

- **链接: [https://arxiv.org/pdf/2601.06519v1](https://arxiv.org/pdf/2601.06519v1)**

> **作者:** Yuelyu Ji; Min Gu Kwak; Hang Zhang; Xizhi Wu; Chenyu Li; Yanshan Wang
>
> **摘要:** Biomedical retrieval-augmented generation (RAG) can ground LLM answers in medical literature, yet long-form outputs often contain isolated unsupported or contradictory claims with safety implications. We introduce MedRAGChecker, a claim-level verification and diagnostic framework for biomedical RAG. Given a question, retrieved evidence, and a generated answer, MedRAGChecker decomposes the answer into atomic claims and estimates claim support by combining evidence-grounded natural language inference (NLI) with biomedical knowledge-graph (KG) consistency signals. Aggregating claim decisions yields answer-level diagnostics that help disentangle retrieval and generation failures, including faithfulness, under-evidence, contradiction, and safety-critical error rates. To enable scalable evaluation, we distill the pipeline into compact biomedical models and use an ensemble verifier with class-specific reliability weighting. Experiments on four biomedical QA benchmarks show that MedRAGChecker reliably flags unsupported and contradicted claims and reveals distinct risk profiles across generators, particularly on safety-critical biomedical relations.
>
---
#### [new 061] Efficient Aspect Term Extraction using Spiking Neural Network
- **分类: cs.CL**

- **简介: 该论文属于情感分析中的方面术语提取任务，旨在解决传统深度神经网络能耗高的问题。提出SpikeATE模型，采用脉冲神经网络实现高效提取。**

- **链接: [https://arxiv.org/pdf/2601.06637v1](https://arxiv.org/pdf/2601.06637v1)**

> **作者:** Abhishek Kumar Mishra; Arya Somasundaram; Anup Das; Nagarajan Kandasamy
>
> **摘要:** Aspect Term Extraction (ATE) identifies aspect terms in review sentences, a key subtask of sentiment analysis. While most existing approaches use energy-intensive deep neural networks (DNNs) for ATE as sequence labeling, this paper proposes a more energy-efficient alternative using Spiking Neural Networks (SNNs). Using sparse activations and event-driven inferences, SNNs capture temporal dependencies between words, making them suitable for ATE. The proposed architecture, SpikeATE, employs ternary spiking neurons and direct spike training fine-tuned with pseudo-gradients. Evaluated on four benchmark SemEval datasets, SpikeATE achieves performance comparable to state-of-the-art DNNs with significantly lower energy consumption. This highlights the use of SNNs as a practical and sustainable choice for ATE tasks.
>
---
#### [new 062] NC-Bench: An LLM Benchmark for Evaluating Conversational Competence
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出NC-Bench，一个评估大语言模型对话能力的基准。解决对话结构评估问题，通过三个测试集考察模型在基本对话、RAG和复杂请求中的表现。**

- **链接: [https://arxiv.org/pdf/2601.06426v1](https://arxiv.org/pdf/2601.06426v1)**

> **作者:** Robert J. Moore; Sungeun An; Farhan Ahmed; Jay Pankaj Gala
>
> **备注:** 9 pages, 1 figure, 2 tables
>
> **摘要:** The Natural Conversation Benchmark (NC-Bench) introduce a new approach to evaluating the general conversational competence of large language models (LLMs). Unlike prior benchmarks that focus on the content of model behavior, NC-Bench focuses on the form and structure of natural conversation. Grounded in the IBM Natural Conversation Framework (NCF), NC-Bench comprises three distinct sets. The Basic Conversation Competence set evaluates fundamental sequence management practices, such as answering inquiries, repairing responses, and closing conversational pairs. The RAG set applies the same sequence management patterns as the first set but incorporates retrieval-augmented generation (RAG). The Complex Request set extends the evaluation to complex requests involving more intricate sequence management patterns. Each benchmark tests a model's ability to produce contextually appropriate conversational actions in response to characteristic interaction patterns. Initial evaluations across 6 open-source models and 14 interaction patterns show that models perform well on basic answering tasks, struggle more with repair tasks (especially repeat), have mixed performance on closing sequences, and find complex multi-turn requests most challenging, with Qwen models excelling on the Basic set and Granite models on the RAG set and the Complex Request set. By operationalizing fundamental principles of human conversation, NC-Bench provides a lightweight, extensible, and theory-grounded framework for assessing and improving the conversational abilities of LLMs beyond topical or task-specific benchmarks.
>
---
#### [new 063] A Unified Framework for Emotion Recognition and Sentiment Analysis via Expert-Guided Multimodal Fusion with Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于情感识别与情感分析任务，解决多模态信息融合问题。提出EGMF框架，结合专家引导的多模态融合与大语言模型，提升情感理解效果。**

- **链接: [https://arxiv.org/pdf/2601.07565v1](https://arxiv.org/pdf/2601.07565v1)**

> **作者:** Jiaqi Qiao; Xiujuan Xu; Xinran Li; Yu Liu
>
> **摘要:** Multimodal emotion understanding requires effective integration of text, audio, and visual modalities for both discrete emotion recognition and continuous sentiment analysis. We present EGMF, a unified framework combining expert-guided multimodal fusion with large language models. Our approach features three specialized expert networks--a fine-grained local expert for subtle emotional nuances, a semantic correlation expert for cross-modal relationships, and a global context expert for long-range dependencies--adaptively integrated through hierarchical dynamic gating for context-aware feature selection. Enhanced multimodal representations are integrated with LLMs via pseudo token injection and prompt-based conditioning, enabling a single generative framework to handle both classification and regression through natural language generation. We employ LoRA fine-tuning for computational efficiency. Experiments on bilingual benchmarks (MELD, CHERMA, MOSEI, SIMS-V2) demonstrate consistent improvements over state-of-the-art methods, with superior cross-lingual robustness revealing universal patterns in multimodal emotional expressions across English and Chinese. We will release the source code publicly.
>
---
#### [new 064] Why LoRA Fails to Forget: Regularized Low-Rank Adaptation Against Backdoors in Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的模型安全任务，解决LoRA在去除后门行为上的不足。通过改进LoRA的谱特性，提出RoRA方法，有效降低攻击成功率并保持准确率。**

- **链接: [https://arxiv.org/pdf/2601.06305v1](https://arxiv.org/pdf/2601.06305v1)**

> **作者:** Hoang-Chau Luong; Lingwei Chen
>
> **摘要:** Low-Rank Adaptation (LoRA) is widely used for parameter-efficient fine-tuning of large language models, but it is notably ineffective at removing backdoor behaviors from poisoned pretrained models when fine-tuning on clean dataset. Contrary to the common belief that this weakness is caused primarily by low rank, we show that LoRA's vulnerability is fundamentally spectral. Our analysis identifies two key factors: LoRA updates (i) possess insufficient spectral strength, with singular values far below those of pretrained weights, and (ii) exhibit unfavorable spectral alignment, weakly matching clean-task directions while retaining overlap with trigger-sensitive subspaces. We further establish a critical scaling threshold beyond which LoRA can theoretically suppress trigger-induced activations, and we show empirically that standard LoRA rarely reaches this regime. We introduce Regularized Low-Rank Adaptation (RoRA), which improves forgetting by increasing spectral strength and correcting alignment through clean-strengthened regularization, trigger-insensitive constraints, and post-training spectral rescaling. Experiments across multiple NLP benchmarks and attack settings show that RoRA substantially reduces attack success rates while maintaining clean accuracy.
>
---
#### [new 065] Controlling Multimodal Conversational Agents with Coverage-Enhanced Latent Actions
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究如何通过增强覆盖的潜在动作空间优化多模态对话代理的强化学习微调。针对文本词表过大和数据不足的问题，提出结合图文数据与纯文本数据构建潜在动作空间，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.07516v1](https://arxiv.org/pdf/2601.07516v1)**

> **作者:** Yongqi Li; Hao Lang; Tieyun Qian; Yongbin Li
>
> **摘要:** Vision-language models are increasingly employed as multimodal conversational agents (MCAs) for diverse conversational tasks. Recently, reinforcement learning (RL) has been widely explored for adapting MCAs to various human-AI interaction scenarios. Despite showing great enhancement in generalization performance, fine-tuning MCAs via RL still faces challenges in handling the extremely large text token space. To address this, we learn a compact latent action space for RL fine-tuning instead. Specifically, we adopt the learning from observation mechanism to construct the codebook for the latent action space, where future observations are leveraged to estimate current latent actions that could further be used to reconstruct future observations. However, the scarcity of paired image-text data hinders learning a codebook with sufficient coverage. Thus, we leverage both paired image-text data and text-only data to construct the latent action space, using a cross-modal projector for transforming text embeddings into image-text embeddings. We initialize the cross-modal projector on paired image-text data, and further train it on massive text-only data with a novel cycle consistency loss to enhance its robustness. We show that our latent action based method outperforms competitive baselines on two conversation tasks across various RL algorithms.
>
---
#### [new 066] MITRA: A Large-Scale Parallel Corpus and Multilingual Pretrained Language Model for Machine Translation and Semantic Retrieval for Pāli, Sanskrit, Buddhist Chinese, and Tibetan
- **分类: cs.CL**

- **简介: 该论文提出MITRA框架，解决佛教古文献跨语言平行文本挖掘与翻译问题。构建了大规模平行语料库，开发了领域预训练模型，提升机器翻译和语义检索性能。**

- **链接: [https://arxiv.org/pdf/2601.06400v1](https://arxiv.org/pdf/2601.06400v1)**

> **作者:** Sebastian Nehrdich; Kurt Keutzer
>
> **摘要:** Ancient Buddhist literature features frequent, yet often unannotated, textual parallels spread across diverse languages: Sanskrit, Pāli, Buddhist Chinese, Tibetan, and more. The scale of this material makes manual examination prohibitive. We present the MITRA framework, which consists of a novel pipeline for multilingual parallel passage mining, MITRA-parallel, a large-scale corpus of 1.74 million parallel sentence pairs between Sanskrit, Chinese, and Tibetan, and the development of the domain-specific pretrained language model Gemma 2 MITRA. We present Gemma 2 MITRA-MT, a version of this base model fine-tuned on machine translation tasks, reaching state-of-the-art performance for machine translation of these languages into English and outperforming even much larger open-source models. We also present Gemma 2 MITRA-E, a semantic embedding model that shows state-of-the-art performance on a novel, detailed semantic embedding benchmark. We make the parallel dataset, model weights, and semantic similarity benchmark openly available to aid both NLP research and philological studies in Buddhist and classical Asian literature.
>
---
#### [new 067] Distributional Clarity: The Hidden Driver of RL-Friendliness in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于强化学习领域，旨在解决大语言模型在强化学习中表现差异的问题。通过引入分布清晰度概念，提升模型的RL友好性。**

- **链接: [https://arxiv.org/pdf/2601.06911v1](https://arxiv.org/pdf/2601.06911v1)**

> **作者:** Shaoning Sun; Mingzhu Cai; Huang He; Bingjin Chen; Siqi Bao; Yujiu Yang; Hua Wu; Haifeng Wang
>
> **摘要:** Language model families exhibit striking disparity in their capacity to benefit from reinforcement learning: under identical training, models like Qwen achieve substantial gains, while others like Llama yield limited improvements. Complementing data-centric approaches, we reveal that this disparity reflects a hidden structural property: \textbf{distributional clarity} in probability space. Through a three-stage analysis-from phenomenon to mechanism to interpretation-we uncover that RL-friendly models exhibit intra-class compactness and inter-class separation in their probability assignments to correct vs. incorrect responses. We quantify this clarity using the \textbf{Silhouette Coefficient} ($S$) and demonstrate that (1) high $S$ correlates strongly with RL performance; (2) low $S$ is associated with severe logic errors and reasoning instability. To confirm this property, we introduce a Silhouette-Aware Reweighting strategy that prioritizes low-$S$ samples during training. Experiments across six mathematical benchmarks show consistent improvements across all model families, with gains up to 5.9 points on AIME24. Our work establishes distributional clarity as a fundamental, trainable property underlying RL-Friendliness.
>
---
#### [new 068] Order in the Evaluation Court: A Critical Analysis of NLG Evaluation Trends
- **分类: cs.CL**

- **简介: 该论文属于自然语言生成（NLG）评估研究，分析评估方法的演变，揭示任务差异、度量惯性及人类与LLM评估的分歧，提出改进建议。**

- **链接: [https://arxiv.org/pdf/2601.07648v1](https://arxiv.org/pdf/2601.07648v1)**

> **作者:** Jing Yang; Nils Feldhus; Salar Mohtaj; Leonhard Hennig; Qianli Wang; Eleni Metheniti; Sherzod Hakimov; Charlott Jakob; Veronika Solopova; Konrad Rieck; David Schlangen; Sebastian Möller; Vera Schmitt
>
> **备注:** 8 pages
>
> **摘要:** Despite advances in Natural Language Generation (NLG), evaluation remains challenging. Although various new metrics and LLM-as-a-judge (LaaJ) methods are proposed, human judgment persists as the gold standard. To systematically review how NLG evaluation has evolved, we employ an automatic information extraction scheme to gather key information from NLG papers, focusing on different evaluation methods (metrics, LaaJ and human evaluation). With extracted metadata from 14,171 papers across four major conferences (ACL, EMNLP, NAACL, and INLG) over the past six years, we reveal several critical findings: (1) Task Divergence: While Dialogue Generation demonstrates a rapid shift toward LaaJ (>40% in 2025), Machine Translation remains locked into n-gram metrics, and Question Answering exhibits a substantial decline in the proportion of studies conducting human evaluation. (2) Metric Inertia: Despite the development of semantic metrics, general-purpose metrics (e.g., BLEU, ROUGE) continue to be widely used across tasks without empirical justification, often lacking the discriminative power to distinguish between specific quality criteria. (3) Human-LaaJ Divergence: Our association analysis challenges the assumption that LLMs act as mere proxies for humans; LaaJ and human evaluations prioritize very different signals, and explicit validation is scarce (<8% of papers comparing the two), with only moderate to low correlation. Based on these observations, we derive practical recommendations to improve the rigor of future NLG evaluation.
>
---
#### [new 069] Time Travel Engine: A Shared Latent Chronological Manifold Enables Historical Navigation in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM中时间信息编码问题。提出TTE框架，通过共享时序流形实现历史语境导航，揭示语言间时间结构的共性。**

- **链接: [https://arxiv.org/pdf/2601.06437v1](https://arxiv.org/pdf/2601.06437v1)**

> **作者:** Jingmin An; Wei Liu; Qian Wang; Fang Fang
>
> **摘要:** Time functions as a fundamental dimension of human cognition, yet the mechanisms by which Large Language Models (LLMs) encode chronological progression remain opaque. We demonstrate that temporal information in their latent space is organized not as discrete clusters but as a continuous, traversable geometry. We introduce the Time Travel Engine (TTE), an interpretability-driven framework that projects diachronic linguistic patterns onto a shared chronological manifold. Unlike surface-level prompting, TTE directly modulates latent representations to induce coherent stylistic, lexical, and conceptual shifts aligned with target eras. By parameterizing diachronic evolution as a continuous manifold within the residual stream, TTE enables fluid navigation through period-specific "zeitgeists" while restricting access to future knowledge. Furthermore, experiments across diverse architectures reveal topological isomorphism between the temporal subspaces of Chinese and English-indicating that distinct languages share a universal geometric logic of historical evolution. These findings bridge historical linguistics with mechanistic interpretability, offering a novel paradigm for controlling temporal reasoning in neural networks.
>
---
#### [new 070] SimLLM: Fine-Tuning Code LLMs for SimPy-Based Queueing System Simulation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于代码生成任务，旨在解决使用闭源模型生成SimPy仿真代码的成本与隐私问题，通过微调开源模型提升其生成质量。**

- **链接: [https://arxiv.org/pdf/2601.06543v1](https://arxiv.org/pdf/2601.06543v1)**

> **作者:** Jun-Qi Chen; Kun Zhang; Rui Zheng; Ying Zhong
>
> **备注:** 33 pages, 10 figures
>
> **摘要:** The Python package SimPy is widely used for modeling queueing systems due to its flexibility, simplicity, and smooth integration with modern data analysis and optimization frameworks. Recent advances in large language models (LLMs) have shown strong ability in generating clear and executable code, making them powerful and suitable tools for writing SimPy queueing simulation code. However, directly employing closed-source models like GPT-4o to generate such code may lead to high computational costs and raise data privacy concerns. To address this, we fine-tune two open-source LLMs, Qwen-Coder-7B and DeepSeek-Coder-6.7B, on curated SimPy queueing data, which enhances their code-generating performance in executability, output-format compliance, and instruction-code consistency. Particularly, we proposed a multi-stage fine-tuning framework comprising two stages of supervised fine-tuning (SFT) and one stage of direct preference optimization (DPO), progressively enhancing the model's ability in SimPy-based queueing simulation code generation. Extensive evaluations demonstrate that both fine-tuned models achieve substantial improvements in executability, output-format compliance, and instruct consistency. These results confirm that domain-specific fine-tuning can effectively transform compact open-source code models into reliable SimPy simulation generators which provide a practical alternative to closed-source LLMs for education, research, and operational decision support.
>
---
#### [new 071] Beyond Hard Masks: Progressive Token Evolution for Diffusion Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言建模任务，解决DLMs中硬掩码限制可修订解码的问题，提出EvoToken-DLM，用渐进软令牌分布替代硬掩码，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.07351v1](https://arxiv.org/pdf/2601.07351v1)**

> **作者:** Linhao Zhong; Linyu Wu; Bozhen Fang; Tianjian Feng; Chenchen Jing; Wen Wang; Jiaheng Zhang; Hao Chen; Chunhua Shen
>
> **备注:** Project webpage: https://aim-uofa.github.io/EvoTokenDLM
>
> **摘要:** Diffusion Language Models (DLMs) offer a promising alternative for language modeling by enabling parallel decoding through iterative refinement. However, most DLMs rely on hard binary masking and discrete token assignments, which hinder the revision of early decisions and underutilize intermediate probabilistic representations. In this paper, we propose EvoToken-DLM, a novel diffusion-based language modeling approach that replaces hard binary masks with evolving soft token distributions. EvoToken-DLM enables a progressive transition from masked states to discrete outputs, supporting revisable decoding. To effectively support this evolution, we introduce continuous trajectory supervision, which aligns training objectives with iterative probabilistic updates. Extensive experiments across multiple benchmarks show that EvoToken-DLM consistently achieves superior performance, outperforming strong diffusion-based and masked DLM baselines. Project webpage: https://aim-uofa.github.io/EvoTokenDLM.
>
---
#### [new 072] Mitrasamgraha: A Comprehensive Classical Sanskrit Machine Translation Dataset
- **分类: cs.CL**

- **简介: 该论文属于机器翻译任务，旨在解决梵语复杂文本翻译难题。构建了大规模梵语-英语数据集Mitrasamgraha，包含39万对双语数据，用于提升翻译模型性能并研究领域与时期影响。**

- **链接: [https://arxiv.org/pdf/2601.07314v1](https://arxiv.org/pdf/2601.07314v1)**

> **作者:** Sebastian Nehrdich; David Allport; Sven Sellmer; Jivnesh Sandhan; Manoj Balaji Jagadeeshan; Pawan Goyal; Sujeet Kumar; Kurt Keutzer
>
> **摘要:** While machine translation is regarded as a "solved problem" for many high-resource languages, close analysis quickly reveals that this is not the case for content that shows challenges such as poetic language, philosophical concepts, multi-layered metaphorical expressions, and more. Sanskrit literature is a prime example of this, as it combines a large number of such challenges in addition to inherent linguistic features like sandhi, compounding, and heavy morphology, which further complicate NLP downstream tasks. It spans multiple millennia of text production time as well as a large breadth of different domains, ranging from ritual formulas via epic narratives, philosophical treatises, poetic verses up to scientific material. As of now, there is a strong lack of publicly available resources that cover these different domains and temporal layers of Sanskrit. We therefore introduce Mitrasamgraha, a high-quality Sanskrit-to-English machine translation dataset consisting of 391,548 bitext pairs, more than four times larger than the largest previously available Sanskrit dataset Itih=asa. It covers a time period of more than three millennia and a broad range of historical Sanskrit domains. In contrast to web-crawled datasets, the temporal and domain annotation of this dataset enables fine-grained study of domain and time period effects on MT performance. We also release a validation set consisting of 5,587 and a test set consisting of 5,552 post-corrected bitext pairs. We conduct experiments benchmarking commercial and open models on this dataset and fine-tune NLLB and Gemma models on the dataset, showing significant improvements, while still recognizing significant challenges in the translation of complex compounds, philosophical concepts, and multi-layered metaphors. We also analyze how in-context learning on this dataset impacts the performance of commercial models
>
---
#### [new 073] Task Arithmetic with Support Languages for Low-Resource ASR
- **分类: cs.CL**

- **简介: 该论文属于低资源语音识别任务，旨在解决数据稀缺问题。通过结合高资源语言模型，使用任务算术提升低资源语言的识别性能。**

- **链接: [https://arxiv.org/pdf/2601.07038v1](https://arxiv.org/pdf/2601.07038v1)**

> **作者:** Emma Rafkin; Dan DeGenaro; Xiulin Yang
>
> **备注:** 8 pages, 3 Figures, preprint after submitted for review for a *ACL conference
>
> **摘要:** The development of resource-constrained approaches to automatic speech recognition (ASR) is of great interest due to its broad applicability to many low-resource languages for which there is scant usable data. Existing approaches to many low-resource natural language processing tasks leverage additional data from higher-resource languages that are closely related to a target low-resource language. One increasingly popular approach uses task arithmetic to combine models trained on different tasks to create a model for a task where there is little to no training data. In this paper, we consider training on a particular language to be a task, and we generate task vectors by fine-tuning variants of the Whisper ASR system. For pairings of high- and low-resource languages, we merge task vectors via a linear combination, optimizing the weights of the linear combination on the downstream word error rate on the low-resource target language's validation set. We find that this approach consistently improves performance on the target languages.
>
---
#### [new 074] Thinking Before Constraining: A Unified Decoding Framework for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，解决自由生成与结构化输出之间的矛盾。通过结合两种生成方式，提升模型推理能力同时保证输出结构可靠。**

- **链接: [https://arxiv.org/pdf/2601.07525v1](https://arxiv.org/pdf/2601.07525v1)**

> **作者:** Ngoc Trinh Hung Nguyen; Alonso Silva; Laith Zumot; Liubov Tupikina; Armen Aghasaryan; Mehwish Alam
>
> **摘要:** Natural generation allows Language Models (LMs) to produce free-form responses with rich reasoning, but the lack of guaranteed structure makes outputs difficult to parse or verify. Structured generation, or constrained decoding, addresses this drawback by producing content in standardized formats such as JSON, ensuring consistency and guaranteed-parsable outputs, but it can inadvertently restrict the model's reasoning capabilities. In this work, we propose a simple approach that combines the advantages of both natural and structured generation. By allowing LLMs to reason freely until specific trigger tokens are generated, and then switching to structured generation, our method preserves the expressive power of natural language reasoning while ensuring the reliability of structured outputs. We further evaluate our approach on several datasets, covering both classification and reasoning tasks, to demonstrate its effectiveness, achieving a substantial gain of up to 27% in accuracy compared to natural generation, while requiring only a small overhead of 10-20 extra tokens.
>
---
#### [new 075] Beyond Single-Shot: Multi-step Tool Retrieval via Query Planning
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于工具检索任务，解决复杂请求下单次检索不足的问题。提出TOOLQP框架，通过多步查询规划提升检索效果。**

- **链接: [https://arxiv.org/pdf/2601.07782v1](https://arxiv.org/pdf/2601.07782v1)**

> **作者:** Wei Fang; James Glass
>
> **摘要:** LLM agents operating over massive, dynamic tool libraries rely on effective retrieval, yet standard single-shot dense retrievers struggle with complex requests. These failures primarily stem from the disconnect between abstract user goals and technical documentation, and the limited capacity of fixed-size embeddings to model combinatorial tool compositions. To address these challenges, we propose TOOLQP, a lightweight framework that models retrieval as iterative query planning. Instead of single-shot matching, TOOLQP decomposes instructions into sub-tasks and dynamically generates queries to interact with the retriever, effectively bridging the semantic gap by targeting the specific sub-tasks required for composition. We train TOOLQP using synthetic query trajectories followed by optimization via Reinforcement Learning with Verifiable Rewards (RLVR). Experiments demonstrate that TOOLQP achieves state-of-the-art performance, exhibiting superior zero-shot generalization, robustness across diverse retrievers, and significant improvements in downstream agentic execution.
>
---
#### [new 076] Fine-grained Verbal Attack Detection via a Hierarchical Divide-and-Conquer Framework
- **分类: cs.CL**

- **简介: 该论文属于 verbal attack detection 任务，旨在解决中文社交媒体中隐性攻击识别困难的问题。提出一个分层框架，通过结构化任务分解提升检测效果。**

- **链接: [https://arxiv.org/pdf/2601.06907v1](https://arxiv.org/pdf/2601.06907v1)**

> **作者:** Quan Zheng; Yuanhe Tian; Ming Wang; Yan Song
>
> **备注:** 13pages, 5figures
>
> **摘要:** In the digital era, effective identification and analysis of verbal attacks are essential for maintaining online civility and ensuring social security. However, existing research is limited by insufficient modeling of conversational structure and contextual dependency, particularly in Chinese social media where implicit attacks are prevalent. Current attack detection studies often emphasize general semantic understanding while overlooking user response relationships, hindering the identification of implicit and context-dependent attacks. To address these challenges, we present the novel "Hierarchical Attack Comment Detection" dataset and propose a divide-and-conquer, fine-grained framework for verbal attack recognition based on spatiotemporal information. The proposed dataset explicitly encodes hierarchical reply structures and chronological order, capturing complex interaction patterns in multi-turn discussions. Building on this dataset, the framework decomposes attack detection into hierarchical subtasks, where specialized lightweight models handle explicit detection, implicit intent inference, and target identification under constrained context. Extensive experiments on the proposed dataset and benchmark intention detection datasets show that smaller models using our framework significantly outperform larger monolithic models relying on parameter scaling, demonstrating the effectiveness of structured task decomposition.
>
---
#### [new 077] Characterising Toxicity in Generative Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在研究生成式大语言模型的毒性输出问题，分析其生成有毒内容的范围及语言因素影响。**

- **链接: [https://arxiv.org/pdf/2601.06700v1](https://arxiv.org/pdf/2601.06700v1)**

> **作者:** Zhiyao Zhang; Yazan Mash'Al; Yuhan Wu
>
> **摘要:** In recent years, the advent of the attention mechanism has significantly advanced the field of natural language processing (NLP), revolutionizing text processing and text generation. This has come about through transformer-based decoder-only architectures, which have become ubiquitous in NLP due to their impressive text processing and generation capabilities. Despite these breakthroughs, language models (LMs) remain susceptible to generating undesired outputs: inappropriate, offensive, or otherwise harmful responses. We will collectively refer to these as ``toxic'' outputs. Although methods like reinforcement learning from human feedback (RLHF) have been developed to align model outputs with human values, these safeguards can often be circumvented through carefully crafted prompts. Therefore, this paper examines the extent to which LLMs generate toxic content when prompted, as well as the linguistic factors -- both lexical and syntactic -- that influence the production of such outputs in generative models.
>
---
#### [new 078] Lexicalized Constituency Parsing for Middle Dutch: Low-resource Training and Cross-Domain Generalization
- **分类: cs.CL**

- **简介: 该论文属于历史语言的句法解析任务，针对低资源语言Middle Dutch进行句法分析。研究提出改进方法以提升模型在域内和跨域的性能。**

- **链接: [https://arxiv.org/pdf/2601.07008v1](https://arxiv.org/pdf/2601.07008v1)**

> **作者:** Yiming Liang; Fang Zhao
>
> **摘要:** Recent years have seen growing interest in applying neural networks and contextualized word embeddings to the parsing of historical languages. However, most advances have focused on dependency parsing, while constituency parsing for low-resource historical languages like Middle Dutch has received little attention. In this paper, we adapt a transformer-based constituency parser to Middle Dutch, a highly heterogeneous and low-resource language, and investigate methods to improve both its in-domain and cross-domain performance. We show that joint training with higher-resource auxiliary languages increases F1 scores by up to 0.73, with the greatest gains achieved from languages that are geographically and temporally closer to Middle Dutch. We further evaluate strategies for leveraging newly annotated data from additional domains, finding that fine-tuning and data combination yield comparable improvements, and our neural parser consistently outperforms the currently used PCFG-based parser for Middle Dutch. We further explore feature-separation techniques for domain adaptation and demonstrate that a minimum threshold of approximately 200 examples per domain is needed to effectively enhance cross-domain performance.
>
---
#### [new 079] KALE: Enhancing Knowledge Manipulation in Large Language Models via Knowledge-aware Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识增强任务，旨在解决大语言模型知识操控能力不足的问题。提出KALE框架，通过知识图谱生成高质量推理过程并优化模型，提升其知识回忆与应用能力。**

- **链接: [https://arxiv.org/pdf/2601.07430v1](https://arxiv.org/pdf/2601.07430v1)**

> **作者:** Qitan Lv; Tianyu Liu; Qiaosheng Zhang; Xingcheng Xu; Chaochao Lu
>
> **摘要:** Despite the impressive performance of large language models (LLMs) pretrained on vast knowledge corpora, advancing their knowledge manipulation-the ability to effectively recall, reason, and transfer relevant knowledge-remains challenging. Existing methods mainly leverage Supervised Fine-Tuning (SFT) on labeled datasets to enhance LLMs' knowledge manipulation ability. However, we observe that SFT models still exhibit the known&incorrect phenomenon, where they explicitly possess relevant knowledge for a given question but fail to leverage it for correct answers. To address this challenge, we propose KALE (Knowledge-Aware LEarning)-a post-training framework that leverages knowledge graphs (KGs) to generate high-quality rationales and enhance LLMs' knowledge manipulation ability. Specifically, KALE first introduces a Knowledge-Induced (KI) data synthesis method that efficiently extracts multi-hop reasoning paths from KGs to generate high-quality rationales for question-answer pairs. Then, KALE employs a Knowledge-Aware (KA) fine-tuning paradigm that enhances knowledge manipulation by internalizing rationale-guided reasoning through minimizing the KL divergence between predictions with and without rationales. Extensive experiments on eight popular benchmarks across six different LLMs demonstrate the effectiveness of KALE, achieving accuracy improvements of up to 11.72% and an average of 4.18%.
>
---
#### [new 080] How Context Shapes Truth: Geometric Transformations of Statement-level Truth Representations in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究LLMs中上下文如何影响陈述的真值向量，属于模型内部表征分析任务，旨在揭示上下文对真值向量的几何变化影响。**

- **链接: [https://arxiv.org/pdf/2601.06599v1](https://arxiv.org/pdf/2601.06599v1)**

> **作者:** Shivam Adarsh; Maria Maistro; Christina Lioma
>
> **摘要:** Large Language Models (LLMs) often encode whether a statement is true as a vector in their residual stream activations. These vectors, also known as truth vectors, have been studied in prior work, however how they change when context is introduced remains unexplored. We study this question by measuring (1) the directional change ($θ$) between the truth vectors with and without context and (2) the relative magnitude of the truth vectors upon adding context. Across four LLMs and four datasets, we find that (1) truth vectors are roughly orthogonal in early layers, converge in middle layers, and may stabilize or continue increasing in later layers; (2) adding context generally increases the truth vector magnitude, i.e., the separation between true and false representations in the activation space is amplified; (3) larger models distinguish relevant from irrelevant context mainly through directional change ($θ$), while smaller models show this distinction through magnitude differences. We also find that context conflicting with parametric knowledge produces larger geometric changes than parametrically aligned context. To the best of our knowledge, this is the first work that provides a geometric characterization of how context transforms the truth vector in the activation space of LLMs.
>
---
#### [new 081] GRASP LoRA: GRPO Guided Adapter Sparsity Policy for Cross Lingual Transfer
- **分类: cs.CL**

- **简介: 该论文提出GRASP LoRA，解决跨语言迁移中的参数高效微调问题。通过引入可学习的全局稀疏度控制，替代传统网格搜索，提升模型性能并降低计算成本。**

- **链接: [https://arxiv.org/pdf/2601.06702v1](https://arxiv.org/pdf/2601.06702v1)**

> **作者:** Besher Hassan; Xiuying Chen
>
> **备注:** 12 pages, 3 figures
>
> **摘要:** Parameter efficient fine tuning is a way to adapt LLMs to new languages when compute or data are limited, yet adapter pipelines usually choose a global prune ratio by grid search. This practice is computationally expensive and development set intensive, since it repeats training, freezes sparsity, and misses fractional optima. We introduce GRASP LoRA (GRPO Guided Adapter Sparsity Policy), which treats global sparsity as a learnable control variable. A GRPO controller interleaves with training, periodically probing candidate prune ratios on a small micro development set and updating a single global prune ratio online from its reward signal. It operates on merged source and target LoRA adapters on a frozen backbone and replaces grid search with one controller run that learns a prune ratio, followed by a single final merge and prune fine tuning run with pruning fixed to that ratio. On cross lingual transfer from English into Arabic and Chinese, including XL-Sum summarization and MLQA extractive question answering with Llama 3 8B, GRASP LoRA improves semantic faithfulness, content coverage, and answer quality over strong target only and merge and prune baselines. It reduces end to end runtime by multiple times relative to grid search, lowers reliance on large development sets, and makes adapter reuse practical for low resource deployment.
>
---
#### [new 082] Do Language Models Reason Across Languages?
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言问答任务，探讨语言模型是否能跨语言推理。研究发现模型在多语言情况下推理不连贯，提出SUBQ方法提升准确率。**

- **链接: [https://arxiv.org/pdf/2601.06644v1](https://arxiv.org/pdf/2601.06644v1)**

> **作者:** Yan Meng; Wafaa Mohammed; Christof Monz
>
> **摘要:** The real-world information sources are inherently multilingual, which naturally raises a question about whether language models can synthesize information across languages. In this paper, we introduce a simple two-hop question answering setting, where answering a question requires making inferences over two multilingual documents. We find that language models are more sensitive to language variation in answer-span documents than in those providing bridging information, despite the equal importance of both documents for answering a question. Under a step-by-step sub-question evaluation, we further show that in up to 33% of multilingual cases, models fail to infer the bridging information in the first step yet still answer the overall question correctly. This indicates that reasoning in language models, especially in multilingual settings, does not follow a faithful step-by-step decomposition. Subsequently, we show that the absence of reasoning decomposition leads to around 18% composition failure, where both sub-questions are answered correctly but fail for the final two-hop questions. To mitigate this, we propose a simple three-stage SUBQ prompting method to guide the multi-step reasoning with sub-questions, which boosts accuracy from 10.1% to 66.5%.
>
---
#### [new 083] GanitLLM: Difficulty-Aware Bengali Mathematical Reasoning through Curriculum-GRPO
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于数学推理任务，解决低资源语言 Bengali 数学问题。构建了 GanitLLM 模型和难度感知数据集，采用 Curriculum-GRPO 方法提升模型表现。**

- **链接: [https://arxiv.org/pdf/2601.06767v1](https://arxiv.org/pdf/2601.06767v1)**

> **作者:** Shubhashis Roy Dipta; Khairul Mahbub; Nadia Najjar
>
> **摘要:** We present a Bengali mathematical reasoning model called GanitLLM (named after the Bangla word for mathematics, "Ganit"), together with a new difficulty-aware Bengali math corpus and a curriculum-based GRPO pipeline. Bengali is one of the world's most widely spoken languages, yet existing LLMs either reason in English and then translate, or simply fail on multi-step Bengali math, in part because reinforcement learning recipes are tuned for high-resource languages and collapse under reward sparsity in low-resource settings. To address this, we construct Ganit, a rigorously filtered and decontaminated Bengali math dataset with automatic difficulty tags derived from the pass@k of a strong evaluator model. Building on this dataset, we propose Curriculum-GRPO, which combines multi-stage training (SFT + GRPO) with difficulty-aware sampling and verifiable rewards for format, numerical correctness, and Bengali reasoning. On Bn-MGSM and Bn-MSVAMP, GanitLLM-4B improves over its Qwen3-4B base by +8 and +7 accuracy points, respectively, while increasing the percentage of Bengali reasoning tokens from 14% to over 88% and reducing average solution length from 943 to 193 words.
>
---
#### [new 084] †DAGGER: Distractor-Aware Graph Generation for Executable Reasoning in Math Problems
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于数学问题求解任务，旨在解决无关信息干扰下的推理问题。通过引入基准数据集和提出DAGGER模型，提升模型在噪声环境下的鲁棒性和效率。**

- **链接: [https://arxiv.org/pdf/2601.06853v1](https://arxiv.org/pdf/2601.06853v1)**

> **作者:** Zabir Al Nazi; Shubhashis Roy Dipta; Sudipta Kar
>
> **摘要:** Chain-of-Thought (CoT) prompting is widely adopted for mathematical problem solving, including in low-resource languages, yet its behavior under irrelevant context remains underexplored. To systematically study this challenge, we introduce DISTRACTMATH-BN, a Bangla benchmark that augments MGSM and MSVAMP with semantically coherent but computationally irrelevant information. Evaluating seven models ranging from 3B to 12B parameters, we observe substantial performance degradation under distractors: standard models drop by up to 41 points, while reasoning-specialized models decline by 14 to 20 points despite consuming five times more tokens. We propose †DAGGER, which reformulates mathematical problem solving as executable computational graph generation with explicit modeling of distractor nodes. Fine-tuning Gemma-3 models using supervised fine-tuning followed by Group Relative Policy Optimization achieves comparable weighted accuracy on augmented benchmarks while using 89 percent fewer tokens than reasoning models. Importantly, this robustness emerges without explicit training on distractor-augmented examples. Our results suggest that enforcing structured intermediate representations improves robustness and inference efficiency in mathematical reasoning compared to free-form approaches, particularly in noisy, low-resource settings.
>
---
#### [new 085] $\texttt{AMEND++}$: Benchmarking Eligibility Criteria Amendments in Clinical Trials
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出AMEND++基准，解决临床试验入选标准频繁修改的问题。通过NLP任务预测标准修改，引入CAMLM模型提升预测效果。**

- **链接: [https://arxiv.org/pdf/2601.06300v1](https://arxiv.org/pdf/2601.06300v1)**

> **作者:** Trisha Das; Mandis Beigi; Jacob Aptekar; Jimeng Sun
>
> **摘要:** Clinical trial amendments frequently introduce delays, increased costs, and administrative burden, with eligibility criteria being the most commonly amended component. We introduce \textit{eligibility criteria amendment prediction}, a novel NLP task that aims to forecast whether the eligibility criteria of an initial trial protocol will undergo future amendments. To support this task, we release $\texttt{AMEND++}$, a benchmark suite comprising two datasets: $\texttt{AMEND}$, which captures eligibility-criteria version histories and amendment labels from public clinical trials, and $\verb|AMEND_LLM|$, a refined subset curated using an LLM-based denoising pipeline to isolate substantive changes. We further propose $\textit{Change-Aware Masked Language Modeling}$ (CAMLM), a revision-aware pretraining strategy that leverages historical edits to learn amendment-sensitive representations. Experiments across diverse baselines show that CAMLM consistently improves amendment prediction, enabling more robust and cost-effective clinical trial design.
>
---
#### [new 086] Paraphrasing Adversarial Attack on LLM-as-a-Reviewer
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究LLM在审稿中的安全问题，提出一种改写对抗攻击方法PAA，旨在提升审稿评分同时保持语义不变，验证其有效性并探索检测方法。**

- **链接: [https://arxiv.org/pdf/2601.06884v1](https://arxiv.org/pdf/2601.06884v1)**

> **作者:** Masahiro Kaneko
>
> **摘要:** The use of large language models (LLMs) in peer review systems has attracted growing attention, making it essential to examine their potential vulnerabilities. Prior attacks rely on prompt injection, which alters manuscript content and conflates injection susceptibility with evaluation robustness. We propose the Paraphrasing Adversarial Attack (PAA), a black-box optimization method that searches for paraphrased sequences yielding higher review scores while preserving semantic equivalence and linguistic naturalness. PAA leverages in-context learning, using previous paraphrases and their scores to guide candidate generation. Experiments across five ML and NLP conferences with three LLM reviewers and five attacking models show that PAA consistently increases review scores without changing the paper's claims. Human evaluation confirms that generated paraphrases maintain meaning and naturalness. We also find that attacked papers exhibit increased perplexity in reviews, offering a potential detection signal, and that paraphrasing submissions can partially mitigate attacks.
>
---
#### [new 087] Outcome-Grounded Advantage Reshaping for Fine-Grained Credit Assignment in Mathematical Reasoning
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于数学推理任务，解决GRPO中粗粒度信用分配问题，提出OAR机制通过细粒度调整优势值提升模型表现。**

- **链接: [https://arxiv.org/pdf/2601.07408v1](https://arxiv.org/pdf/2601.07408v1)**

> **作者:** Ziheng Li; Liu Kang; Feng Xiao; Luxi Xing; Qingyi Si; Zhuoran Li; Weikang Gong; Deqing Yang; Yanghua Xiao; Hongcheng Guo
>
> **摘要:** Group Relative Policy Optimization (GRPO) has emerged as a promising critic-free reinforcement learning paradigm for reasoning tasks. However, standard GRPO employs a coarse-grained credit assignment mechanism that propagates group-level rewards uniformly to to every token in a sequence, neglecting the varying contribution of individual reasoning steps. We address this limitation by introducing Outcome-grounded Advantage Reshaping (OAR), a fine-grained credit assignment mechanism that redistributes advantages based on how much each token influences the model's final answer. We instantiate OAR via two complementary strategies: (1) OAR-P, which estimates outcome sensitivity through counterfactual token perturbations, serving as a high-fidelity attribution signal; (2) OAR-G, which uses an input-gradient sensitivity proxy to approximate the influence signal with a single backward pass. These importance signals are integrated with a conservative Bi-Level advantage reshaping scheme that suppresses low-impact tokens and boosts pivotal ones while preserving the overall advantage mass. Empirical results on extensive mathematical reasoning benchmarks demonstrate that while OAR-P sets the performance upper bound, OAR-G achieves comparable gains with negligible computational overhead, both significantly outperforming a strong GRPO baseline, pushing the boundaries of critic-free LLM reasoning.
>
---
#### [new 088] MedEinst: Benchmarking the Einstellung Effect in Medical LLMs through Counterfactual Differential Diagnosis
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗大模型评估任务，旨在解决LLMs在临床诊断中依赖统计捷径而非患者证据的问题。通过构建MedEinst基准，评估模型对反事实病例的误诊率，并提出ECR-Agent提升诊断准确性。**

- **链接: [https://arxiv.org/pdf/2601.06636v1](https://arxiv.org/pdf/2601.06636v1)**

> **作者:** Wenting Chen; Zhongrui Zhu; Guolin Huang; Wenxuan Wang
>
> **备注:** 19 pages, 7 figures
>
> **摘要:** Despite achieving high accuracy on medical benchmarks, LLMs exhibit the Einstellung Effect in clinical diagnosis--relying on statistical shortcuts rather than patient-specific evidence, causing misdiagnosis in atypical cases. Existing benchmarks fail to detect this critical failure mode. We introduce MedEinst, a counterfactual benchmark with 5,383 paired clinical cases across 49 diseases. Each pair contains a control case and a "trap" case with altered discriminative evidence that flips the diagnosis. We measure susceptibility via Bias Trap Rate--probability of misdiagnosing traps despite correctly diagnosing controls. Extensive Evaluation of 17 LLMs shows frontier models achieve high baseline accuracy but severe bias trap rates. Thus, we propose ECR-Agent, aligning LLM reasoning with Evidence-Based Medicine standard via two components: (1) Dynamic Causal Inference (DCI) performs structured reasoning through dual-pathway perception, dynamic causal graph reasoning across three levels (association, intervention, counterfactual), and evidence audit for final diagnosis; (2) Critic-Driven Graph and Memory Evolution (CGME) iteratively refines the system by storing validated reasoning paths in an exemplar base and consolidating disease-specific knowledge into evolving illness graphs. Source code is to be released.
>
---
#### [new 089] Codified Foreshadowing-Payoff Text Generation
- **分类: cs.CL**

- **简介: 该论文属于文本生成任务，旨在解决LLM无法有效处理叙事中的伏笔与呼应问题。通过构建结构化监督，提升模型对叙事逻辑的把握能力。**

- **链接: [https://arxiv.org/pdf/2601.07033v1](https://arxiv.org/pdf/2601.07033v1)**

> **作者:** Longfei Yun; Kun Zhou; Yupeng Hou; Letian Peng; Jingbo Shang
>
> **摘要:** Foreshadowing and payoff are ubiquitous narrative devices through which authors introduce commitments early in a story and resolve them through concrete, observable outcomes. However, despite advances in story generation, large language models (LLMs) frequently fail to bridge these long-range narrative dependencies, often leaving "Chekhov's guns" unfired even when the necessary context is present. Existing evaluations largely overlook this structural failure, focusing on surface-level coherence rather than the logical fulfillment of narrative setups. In this paper, we introduce Codified Foreshadowing-Payoff Generation (CFPG), a novel framework that reframes narrative quality through the lens of payoff realization. Recognizing that LLMs struggle to intuitively grasp the "triggering mechanism" of a foreshadowed event, CFPG transforms narrative continuity into a set of executable causal predicates. By mining and encoding Foreshadow-Trigger-Payoff triples from the BookSum corpus, we provide structured supervision that ensures foreshadowed commitments are not only mentioned but also temporally and logically fulfilled. Experiments demonstrate that CFPG significantly outperforms standard prompting baselines in payoff accuracy and narrative alignment. Our findings suggest that explicitly codifying narrative mechanics is essential for moving LLMs from surface-level fluency to genuine narrative competence.
>
---
#### [new 090] Judging Against the Reference: Uncovering Knowledge-Driven Failures in LLM-Judges on QA Evaluation
- **分类: cs.CL**

- **简介: 该论文属于QA评估任务，研究LLM作为评判者时因参考答案冲突导致评分不可靠的问题，提出框架分析其依赖参数知识的缺陷。**

- **链接: [https://arxiv.org/pdf/2601.07506v1](https://arxiv.org/pdf/2601.07506v1)**

> **作者:** Dongryeol Lee; Yerin Hwang; Taegwan Kang; Minwoo Lee; Younhyung Chae; Kyomin Jung
>
> **备注:** Under review, 21 pgs, 11 figures, 7 tables
>
> **摘要:** While large language models (LLMs) are increasingly used as automatic judges for question answering (QA) and other reference-conditioned evaluation tasks, little is known about their ability to adhere to a provided reference. We identify a critical failure mode of such reference-based LLM QA evaluation: when the provided reference conflicts with the judge model's parametric knowledge, the resulting scores become unreliable, substantially degrading evaluation fidelity. To study this phenomenon systematically, we introduce a controlled swapped-reference QA framework that induces reference-belief conflicts. Specifically, we replace the reference answer with an incorrect entity and construct diverse pairings of original and swapped references with correspondingly aligned candidate answers. Surprisingly, grading reliability drops sharply under swapped references across a broad set of judge models. We empirically show that this vulnerability is driven by judges' over-reliance on parametric knowledge, leading judges to disregard the given reference under conflict. Finally, we find that this failure persists under common prompt-based mitigation strategies, highlighting a fundamental limitation of LLM-as-a-judge evaluation and motivating reference-based protocols that enforce stronger adherence to the provided reference.
>
---
#### [new 091] AgentHallu: Benchmarking Automated Hallucination Attribution of LLM-based Agents
- **分类: cs.CL**

- **简介: 该论文属于LLM代理幻觉归因任务，旨在识别多步骤推理中导致幻觉的步骤并解释原因。研究构建了AgentHallu基准，评估13个模型，发现任务具有挑战性。**

- **链接: [https://arxiv.org/pdf/2601.06818v1](https://arxiv.org/pdf/2601.06818v1)**

> **作者:** Xuannan Liu; Xiao Yang; Zekun Li; Peipei Li; Ran He
>
> **备注:** Project page: https://liuxuannan.github.io/AgentHallu.github.io/
>
> **摘要:** As LLM-based agents operate over sequential multi-step reasoning, hallucinations arising at intermediate steps risk propagating along the trajectory, thus degrading overall reliability. Unlike hallucination detection in single-turn responses, diagnosing hallucinations in multi-step workflows requires identifying which step causes the initial divergence. To fill this gap, we propose a new research task, automated hallucination attribution of LLM-based agents, aiming to identify the step responsible for the hallucination and explain why. To support this task, we introduce AgentHallu, a comprehensive benchmark with: (1) 693 high-quality trajectories spanning 7 agent frameworks and 5 domains, (2) a hallucination taxonomy organized into 5 categories (Planning, Retrieval, Reasoning, Human-Interaction, and Tool-Use) and 14 sub-categories, and (3) multi-level annotations curated by humans, covering binary labels, hallucination-responsible steps, and causal explanations. We evaluate 13 leading models, and results show the task is challenging even for top-tier models (like GPT-5, Gemini-2.5-Pro). The best-performing model achieves only 41.1\% step localization accuracy, where tool-use hallucinations are the most challenging at just 11.6\%. We believe AgentHallu will catalyze future research into developing robust, transparent, and reliable agentic systems.
>
---
#### [new 092] AzeroS: Extending LLM to Speech with Self-Generated Instruction-Free Tuning
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文提出AZeroS，解决将大语言模型扩展到语音领域的问题。通过自生成无指令微调（SIFT），无需任务特定数据，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.06086v1](https://arxiv.org/pdf/2601.06086v1)**

> **作者:** Yiwen Shao; Wei Liu; Jiahong Li; Tianzi Wang; Kun Wei; Meng Yu; Dong Yu
>
> **备注:** Technical Report
>
> **摘要:** Extending large language models (LLMs) to the speech domain has recently gained significant attention. A typical approach connects a pretrained LLM with an audio encoder through a projection module and trains the resulting model on large-scale, task-specific instruction-tuning datasets. However, curating such instruction-tuning data for specific requirements is time-consuming, and models trained in this manner often generalize poorly to unseen tasks. In this work, we first formulate that the strongest generalization of a speech-LLM is achieved when it is trained with Self-Generated Instruction-Free Tuning (SIFT), in which supervision signals are generated by a frozen LLM using textual representations of speech as input. Our proposed SIFT paradigm eliminates the need for collecting task-specific question-answer pairs and yields the theoretically best generalization to unseen tasks. Building upon this paradigm, we introduce AZeroS (Auden Zero-instruction-tuned Speech-LLM), which is trained on speech-text pairs derived from publicly available corpora, including approximately 25,000 hours of speech with ASR transcripts and 3,000 hours of speech with paralinguistic labels. Built upon Qwen2.5-7B-Instruct, the model updates only two lightweight projection modules (23.8 million parameters each), while keeping both the LLM and audio encoders frozen. Despite the minimal training cost and modest data scale, AZeroS achieves state-of-the-art performance on both semantic and paralinguistic benchmarks, including VoiceBench, AIR-Bench Foundation (Speech), and AIR-Bench Chat (Speech).
>
---
#### [new 093] Beyond Literal Mapping: Benchmarking and Improving Non-Literal Translation Evaluation
- **分类: cs.CL**

- **简介: 该论文属于机器翻译评估任务，旨在解决非字面翻译评价不准确的问题。通过构建MENT数据集和提出RATE框架，提升翻译质量评估的可靠性。**

- **链接: [https://arxiv.org/pdf/2601.07338v1](https://arxiv.org/pdf/2601.07338v1)**

> **作者:** Yanzhi Tian; Cunxiang Wang; Zeming Liu; Heyan Huang; Wenbo Yu; Dawei Song; Jie Tang; Yuhang Guo
>
> **摘要:** Large Language Models (LLMs) have significantly advanced Machine Translation (MT), applying them to linguistically complex domains-such as Social Network Services, literature etc. In these scenarios, translations often require handling non-literal expressions, leading to the inaccuracy of MT metrics. To systematically investigate the reliability of MT metrics, we first curate a meta-evaluation dataset focused on non-literal translations, namely MENT. MENT encompasses four non-literal translation domains and features source sentences paired with translations from diverse MT systems, with 7,530 human-annotated scores on translation quality. Experimental results reveal the inaccuracies of traditional MT metrics and the limitations of LLM-as-a-Judge, particularly the knowledge cutoff and score inconsistency problem. To mitigate these limitations, we propose RATE, a novel agentic translation evaluation framework, centered by a reflective Core Agent that dynamically invokes specialized sub-agents. Experimental results indicate the efficacy of RATE, achieving an improvement of at least 3.2 meta score compared with current metrics. Further experiments demonstrate the robustness of RATE to general-domain MT evaluation. Code and dataset are available at: https://github.com/BITHLP/RATE.
>
---
#### [new 094] Labels have Human Values: Value Calibration of Subjective Tasks
- **分类: cs.CL**

- **简介: 该论文研究主观任务的值校准问题，旨在提升NLP系统与人类价值观的一致性。提出MC-STL框架，通过聚类标注者价值并学习特定嵌入进行预测优化。**

- **链接: [https://arxiv.org/pdf/2601.06631v1](https://arxiv.org/pdf/2601.06631v1)**

> **作者:** Mohammed Fayiz Parappan; Ricardo Henao
>
> **摘要:** Building NLP systems for subjective tasks requires one to ensure their alignment to contrasting human values. We propose the MultiCalibrated Subjective Task Learner framework (MC-STL), which clusters annotations into identifiable human value clusters by three approaches (similarity of annotator rationales, expert-value taxonomies or rater's sociocultural descriptors) and calibrates predictions for each value cluster by learning cluster-specific embeddings. We demonstrate MC-STL on several subjective learning settings, including ordinal, binary, and preference learning predictions, and evaluate it on multiple datasets covering toxic chatbot conversations, offensive social media posts, and human preference alignment. The results show that MC-STL consistently outperforms the baselines that ignore the latent value structure of the annotations, delivering gains in discrimination, value-specific calibration, and disagreement-aware metrics.
>
---
#### [new 095] The Roots of Performance Disparity in Multilingual Language Models: Intrinsic Modeling Difficulty or Design Choices?
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，探讨多语言模型性能差异的原因，旨在区分语言固有难度与模型设计因素。工作包括分析语言特征与建模机制的关系，提出优化设计建议。**

- **链接: [https://arxiv.org/pdf/2601.07220v1](https://arxiv.org/pdf/2601.07220v1)**

> **作者:** Chen Shani; Yuval Reif; Nathan Roll; Dan Jurafsky; Ekaterina Shutova
>
> **摘要:** Multilingual language models (LMs) promise broader NLP access, yet current systems deliver uneven performance across the world's languages. This survey examines why these gaps persist and whether they reflect intrinsic linguistic difficulty or modeling artifacts. We organize the literature around two questions: do linguistic disparities arise from representation and allocation choices (e.g., tokenization, encoding, data exposure, parameter sharing) rather than inherent complexity; and which design choices mitigate inequities across typologically diverse languages. We review linguistic features, such as orthography, morphology, lexical diversity, syntax, information density, and typological distance, linking each to concrete modeling mechanisms. Gaps often shrink when segmentation, encoding, and data exposure are normalized, suggesting much apparent difficulty stems from current modeling choices. We synthesize these insights into design recommendations for tokenization, sampling, architectures, and evaluation to support more balanced multilingual LMs.
>
---
#### [new 096] PRISP: Privacy-Safe Few-Shot Personalization via Lightweight Adaptation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于个性化任务，解决资源受限和隐私保护下的模型个性化问题。提出PRISP框架，通过轻量适配实现高效隐私安全的用户定制。**

- **链接: [https://arxiv.org/pdf/2601.06471v1](https://arxiv.org/pdf/2601.06471v1)**

> **作者:** Junho Park; Dohoon Kim; Taesup Moon
>
> **备注:** 16 pages, 9 figures
>
> **摘要:** Large language model (LLM) personalization aims to adapt general-purpose models to individual users. Most existing methods, however, are developed under data-rich and resource-abundant settings, often incurring privacy risks. In contrast, realistic personalization typically occurs after deployment under (i) extremely limited user data, (ii) constrained computational resources, and (iii) strict privacy requirements. We propose PRISP, a lightweight and privacy-safe personalization framework tailored to these constraints. PRISP leverages a Text-to-LoRA hypernetwork to generate task-aware LoRA parameters from task descriptions, and enables efficient user personalization by optimizing a small subset of task-aware LoRA parameters together with minimal additional modules using few-shot user data. Experiments on a few-shot variant of the LaMP benchmark demonstrate that PRISP achieves strong overall performance compared to prior approaches, while reducing computational overhead and eliminating privacy risks.
>
---
#### [new 097] GROKE: Vision-Free Navigation Instruction Evaluation via Graph Reasoning on OpenStreetMap
- **分类: cs.CL**

- **简介: 该论文属于视觉-语言导航任务，旨在解决导航指令评估难题。提出GROKE框架，利用OpenStreetMap数据进行无视觉的指令评估，提升导航准确性。**

- **链接: [https://arxiv.org/pdf/2601.07375v1](https://arxiv.org/pdf/2601.07375v1)**

> **作者:** Farzad Shami; Subhrasankha Dey; Nico Van de Weghe; Henrikki Tenkanen
>
> **备注:** Under Review for ACL 2026
>
> **摘要:** The evaluation of navigation instructions remains a persistent challenge in Vision-and-Language Navigation (VLN) research. Traditional reference-based metrics such as BLEU and ROUGE fail to capture the functional utility of spatial directives, specifically whether an instruction successfully guides a navigator to the intended destination. Although existing VLN agents could serve as evaluators, their reliance on high-fidelity visual simulators introduces licensing constraints and computational costs, and perception errors further confound linguistic quality assessment. This paper introduces GROKE(Graph-based Reasoning over OSM Knowledge for instruction Evaluation), a vision-free training-free hierarchical LLM-based framework for evaluating navigation instructions using OpenStreetMap data. Through systematic ablation studies, we demonstrate that structured JSON and textual formats for spatial information substantially outperform grid-based and visual graph representations. Our hierarchical architecture combines sub-instruction planning with topological graph navigation, reducing navigation error by 68.5% compared to heuristic and sampling baselines on the Map2Seq dataset. The agent's execution success, trajectory fidelity, and decision patterns serve as proxy metrics for functional navigability given OSM-visible landmarks and topology, establishing a scalable and interpretable evaluation paradigm without visual dependencies. Code and data are available at https://anonymous.4open.science/r/groke.
>
---
#### [new 098] Spec-o3: A Tool-Augmented Vision-Language Agent for Rare Celestial Object Candidate Vetting via Automated Spectral Inspection
- **分类: cs.CL; astro-ph.IM**

- **简介: 该论文提出Spec-o3，解决天体候选对象人工验核效率低的问题，通过多模态推理实现自动化光谱分析，提升识别准确率并具备跨数据集泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.06498v1](https://arxiv.org/pdf/2601.06498v1)**

> **作者:** Minghui Jia; Qichao Zhang; Ali Luo; Linjing Li; Shuo Ye; Hailing Lu; Wen Hou; Dongbin Zhao
>
> **摘要:** Due to the limited generalization and interpretability of deep learning classifiers, The final vetting of rare celestial object candidates still relies on expert visual inspection--a manually intensive process. In this process, astronomers leverage specialized tools to analyze spectra and construct reliable catalogs. However, this practice has become the primary bottleneck, as it is fundamentally incapable of scaling with the data deluge from modern spectroscopic surveys. To bridge this gap, we propose Spec-o3, a tool-augmented vision-language agent that performs astronomer-aligned spectral inspection via interleaved multimodal chain-of-thought reasoning. Spec-o3 is trained with a two-stage post-training recipe: cold-start supervised fine-tuning on expert inspection trajectories followed by outcome-based reinforcement learning on rare-type verification tasks. Evaluated on five rare-object identification tasks from LAMOST, Spec-o3 establishes a new State-of-the-Art, boosting the macro-F1 score from 28.3 to 76.5 with a 7B parameter base model and outperforming both proprietary VLMs and specialized deep models. Crucially, the agent demonstrates strong generalization to unseen inspection tasks across survey shifts (from LAMOST to SDSS/DESI). Expert evaluations confirm that its reasoning traces are coherent and physically consistent, supporting transparent and trustworthy decision-making. Code, data, and models are available at \href{https://github.com/Maxwell-Jia/spec-o3}{Project HomePage}.
>
---
#### [new 099] From RAG to Agentic RAG for Faithful Islamic Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于伊斯兰问答任务，旨在解决LLM在宗教回答中的幻觉和不当回答问题。构建了基准数据集和 grounded 模型框架，提出agentic RAG方法提升答案准确性与可靠性。**

- **链接: [https://arxiv.org/pdf/2601.07528v1](https://arxiv.org/pdf/2601.07528v1)**

> **作者:** Gagan Bhatia; Hamdy Mubarak; Mustafa Jarrar; George Mikros; Fadi Zaraket; Mahmoud Alhirthani; Mutaz Al-Khatib; Logan Cochrane; Kareem Darwish; Rashid Yahiaoui; Firoj Alam
>
> **摘要:** LLMs are increasingly used for Islamic question answering, where ungrounded responses may carry serious religious consequences. Yet standard MCQ/MRC-style evaluations do not capture key real-world failure modes, notably free-form hallucinations and whether models appropriately abstain when evidence is lacking. To shed a light on this aspect we introduce ISLAMICFAITHQA, a 3,810-item bilingual (Arabic/English) generative benchmark with atomic single-gold answers, which enables direct measurement of hallucination and abstention. We additionally developed an end-to-end grounded Islamic modelling suite consisting of (i) 25K Arabic text-grounded SFT reasoning pairs, (ii) 5K bilingual preference samples for reward-guided alignment, and (iii) a verse-level Qur'an retrieval corpus of $\sim$6k atomic verses (ayat). Building on these resources, we develop an agentic Quran-grounding framework (agentic RAG) that uses structured tool calls for iterative evidence seeking and answer revision. Experiments across Arabic-centric and multilingual LLMs show that retrieval improves correctness and that agentic RAG yields the largest gains beyond standard RAG, achieving state-of-the-art performance and stronger Arabic-English robustness even with a small model (i.e., Qwen3 4B). We will make the experimental resources and datasets publicly available for the community.
>
---
#### [new 100] Mid-Think: Training-Free Intermediate-Budget Reasoning via Token-Level Triggers
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于语言模型推理控制任务，旨在解决传统指令控制效果有限的问题。通过分析触发词，提出Mid-Think方法，实现更高效的推理控制与训练。**

- **链接: [https://arxiv.org/pdf/2601.07036v1](https://arxiv.org/pdf/2601.07036v1)**

> **作者:** Wang Yang; Debargha Ganguly; Xinpeng Li; Chaoda Song; Shouren Wang; Vikash Singh; Vipin Chaudhary; Xiaotian Han
>
> **摘要:** Hybrid reasoning language models are commonly controlled through high-level Think/No-think instructions to regulate reasoning behavior, yet we found that such mode switching is largely driven by a small set of trigger tokens rather than the instructions themselves. Through attention analysis and controlled prompting experiments, we show that a leading ``Okay'' token induces reasoning behavior, while the newline pattern following ``</think>'' suppresses it. Based on this observation, we propose Mid-Think, a simple training-free prompting format that combines these triggers to achieve intermediate-budget reasoning, consistently outperforming fixed-token and prompt-based baselines in terms of the accuracy-length trade-off. Furthermore, applying Mid-Think to RL training after SFT reduces training time by approximately 15% while improving final performance of Qwen3-8B on AIME from 69.8% to 72.4% and on GPQA from 58.5% to 61.1%, demonstrating its effectiveness for both inference-time control and RL-based reasoning training.
>
---
#### [new 101] Operation Veja: Fixing Fundamental Concepts Missing from Modern Roleplaying Training Paradigms
- **分类: cs.CL**

- **简介: 该论文属于角色扮演任务，旨在解决角色塑造缺乏真实性的难题。提出VEJA框架，通过核心概念提升角色数据质量，增强角色深度与连贯性。**

- **链接: [https://arxiv.org/pdf/2601.06039v1](https://arxiv.org/pdf/2601.06039v1)**

> **作者:** Yueze Liu; Ajay Nagi Reddy Kumdam; Ronit Kanjilal; Hao Yang; Yichi Zhang
>
> **备注:** Accepted to NeurIPS 2025 PeronaLLM workshop
>
> **摘要:** Modern roleplaying models are increasingly sophisticated, yet they consistently struggle to capture the essence of believable, engaging characters. We argue this failure stems from training paradigms that overlook the dynamic interplay of a character's internal world. Current approaches, including Retrieval-Augmented Generation (RAG), fact-based priming, literature-based learning, and synthetic data generation, exhibit recurring limitations in modeling the deliberative, value-conflicted reasoning that defines human interaction. In this paper, we identify four core concepts essential for character authenticity: Values, Experiences, Judgments, and Abilities (VEJA). We propose the VEJA framework as a new paradigm for data curation that addresses these systemic limitations. To illustrate the qualitative ceiling enabled by our framework, we present a pilot study comparing a manually curated, VEJA-grounded dataset against a state-of-the-art synthetic baseline. Using an LLM-as-judge evaluation, our findings demonstrate a significant quality gap, suggesting that a shift toward conceptually grounded data curation, as embodied by VEJA, is necessary for creating roleplaying agents with genuine depth and narrative continuity. The full dataset is available at https://github.com/HyouinKyoumaIRL/Operation-Veja
>
---
#### [new 102] Fine-Tuning vs. RAG for Multi-Hop Question Answering with Novel Knowledge
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于多跳问答任务，旨在比较微调与RAG在引入新知识时的效果。研究对比了三种知识注入方法，发现RAG在处理新颖信息时表现更优。**

- **链接: [https://arxiv.org/pdf/2601.07054v1](https://arxiv.org/pdf/2601.07054v1)**

> **作者:** Zhuoyi Yang; Yurun Song; Iftekhar Ahmed; Ian Harris
>
> **摘要:** Multi-hop question answering is widely used to evaluate the reasoning capabilities of large language models (LLMs), as it requires integrating multiple pieces of supporting knowledge to arrive at a correct answer. While prior work has explored different mechanisms for providing knowledge to LLMs, such as finetuning and retrieval-augmented generation (RAG), their relative effectiveness for multi-hop question answering remains insufficiently understood, particularly when the required knowledge is temporally novel. In this paper, we systematically compare parametric and non-parametric knowledge injection methods for open-domain multi-hop question answering. We evaluate unsupervised fine-tuning (continual pretraining), supervised fine-tuning, and retrieval-augmented generation across three 7B-parameter open-source LLMs. Experiments are conducted on two benchmarks: QASC, a standard multi-hop science question answering dataset, and a newly constructed dataset of over 10,000 multi-hop questions derived from Wikipedia events in 2024, designed to test knowledge beyond the models' pretraining cutoff. Our results show that unsupervised fine-tuning provides only limited gains over base models, suggesting that continual pretraining alone is insufficient for improving multi-hop reasoning accuracy. In contrast, retrieval-augmented generation yields substantial and consistent improvements, particularly when answering questions that rely on temporally novel information. Supervised fine-tuning achieves the highest overall accuracy across models and datasets. These findings highlight fundamental differences in how knowledge injection mechanisms support multi-hop question answering and underscore the importance of retrieval-based methods when external or compositional knowledge is required.
>
---
#### [new 103] How well can off-the-shelf LLMs elucidate molecular structures from mass spectra using chain-of-thought reasoning?
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于化学信息学任务，旨在利用大语言模型从质谱数据中推断分子结构。研究提出一种链式思维框架，评估模型在零样本设置下的推理能力，发现模型虽能生成合理结构，但缺乏化学准确性。**

- **链接: [https://arxiv.org/pdf/2601.06289v1](https://arxiv.org/pdf/2601.06289v1)**

> **作者:** Yufeng Wang; Lu Wei; Lin Liu; Hao Xu; Haibin Ling
>
> **摘要:** Mass spectrometry (MS) is a powerful analytical technique for identifying small molecules, yet determining complete molecular structures directly from tandem mass spectra (MS/MS) remains a long-standing challenge due to complex fragmentation patterns and the vast diversity of chemical space. Recent progress in large language models (LLMs) has shown promise for reasoning-intensive scientific tasks, but their capability for chemical interpretation is still unclear. In this work, we introduce a Chain-of-Thought (CoT) prompting framework and benchmark that evaluate how LLMs reason about mass spectral data to predict molecular structures. We formalize expert chemists' reasoning steps-such as double bond equivalent (DBE) analysis, neutral loss identification, and fragment assembly-into structured prompts and assess multiple state-of-the-art LLMs (Claude-3.5-Sonnet, GPT-4o-mini, and Llama-3 series) in a zero-shot setting using the MassSpecGym dataset. Our evaluation across metrics of SMILES validity, formula consistency, and structural similarity reveals that while LLMs can produce syntactically valid and partially plausible structures, they fail to achieve chemical accuracy or link reasoning to correct molecular predictions. These findings highlight both the interpretive potential and the current limitations of LLM-based reasoning for molecular elucidation, providing a foundation for future work that combines domain knowledge and reinforcement learning to achieve chemically grounded AI reasoning.
>
---
#### [new 104] BayesRAG: Probabilistic Mutual Evidence Corroboration for Multimodal Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文属于多模态检索任务，旨在解决视觉丰富文档中文本与图像孤立检索的问题。提出BayesRAG框架，通过概率证据融合提升检索一致性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.07329v1](https://arxiv.org/pdf/2601.07329v1)**

> **作者:** Xuan Li; Yining Wang; Haocai Luo; Shengping Liu; Jerry Liang; Ying Fu; Weihuang; Jun Yu; Junnan Zhu
>
> **备注:** 17 pages, 8 figures
>
> **摘要:** Retrieval-Augmented Generation (RAG) has become a pivotal paradigm for Large Language Models (LLMs), yet current approaches struggle with visually rich documents by treating text and images as isolated retrieval targets. Existing methods relying solely on cosine similarity often fail to capture the semantic reinforcement provided by cross-modal alignment and layout-induced coherence. To address these limitations, we propose BayesRAG, a novel multimodal retrieval framework grounded in Bayesian inference and Dempster-Shafer evidence theory. Unlike traditional approaches that rank candidates strictly by similarity, BayesRAG models the intrinsic consistency of retrieved candidates across modalities as probabilistic evidence to refine retrieval confidence. Specifically, our method computes the posterior association probability for combinations of multimodal retrieval results, prioritizing text-image pairs that mutually corroborate each other in terms of both semantics and layout. Extensive experiments demonstrate that BayesRAG significantly outperforms state-of-the-art (SOTA) methods on challenging multimodal benchmarks. This study establishes a new paradigm for multimodal retrieval fusion that effectively resolves the isolation of heterogeneous modalities through an evidence fusion mechanism and enhances the robustness of retrieval outcomes. Our code is available at https://github.com/TioeAre/BayesRAG.
>
---
#### [new 105] Explainable Multimodal Aspect-Based Sentiment Analysis with Dependency-guided Large Language Model
- **分类: cs.CL**

- **简介: 该论文属于多模态方面情感分析任务，旨在解决情感解释性不足的问题。提出一种生成式框架，同时预测情感并生成解释，提升模型可解释性。**

- **链接: [https://arxiv.org/pdf/2601.06848v1](https://arxiv.org/pdf/2601.06848v1)**

> **作者:** Zhongzheng Wang; Yuanhe Tian; Hongzhi Wang; Yan Song
>
> **备注:** 9 pages, 3 figures
>
> **摘要:** Multimodal aspect-based sentiment analysis (MABSA) aims to identify aspect-level sentiments by jointly modeling textual and visual information, which is essential for fine-grained opinion understanding in social media. Existing approaches mainly rely on discriminative classification with complex multimodal fusion, yet lacking explicit sentiment explainability. In this paper, we reformulate MABSA as a generative and explainable task, proposing a unified framework that simultaneously predicts aspect-level sentiment and generates natural language explanations. Based on multimodal large language models (MLLMs), our approach employs a prompt-based generative paradigm, jointly producing sentiment and explanation. To further enhance aspect-oriented reasoning capabilities, we propose a dependency-syntax-guided sentiment cue strategy. This strategy prunes and textualizes the aspect-centered dependency syntax tree, guiding the model to distinguish different sentiment aspects and enhancing its explainability. To enable explainability, we use MLLMs to construct new datasets with sentiment explanations to fine-tune. Experiments show that our approach not only achieves consistent gains in sentiment classification accuracy, but also produces faithful, aspect-grounded explanations.
>
---
#### [new 106] High-Rank Structured Modulation for Parameter-Efficient Fine-Tuning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的模型微调任务，旨在解决参数高效微调中低秩方法表现受限的问题。提出SMoA方法，在减少可训练参数的同时保持高秩，提升模型表达能力。**

- **链接: [https://arxiv.org/pdf/2601.07507v1](https://arxiv.org/pdf/2601.07507v1)**

> **作者:** Yongkang Liu; Xing Li; Mengjie Zhao; Shanru Zhang; Zijing Wang; Qian Li; Shi Feng; Feiliang Ren; Daling Wang; Hinrich Schütze
>
> **备注:** under review
>
> **摘要:** As the number of model parameters increases, parameter-efficient fine-tuning (PEFT) has become the go-to choice for tailoring pre-trained large language models. Low-rank Adaptation (LoRA) uses a low-rank update method to simulate full parameter fine-tuning, which is widely used to reduce resource requirements. However, decreasing the rank encounters challenges with limited representational capacity when compared to full parameter fine-tuning. We present \textbf{SMoA}, a high-rank \textbf{S}tructured \textbf{MO}dulation \textbf{A}dapter that uses fewer trainable parameters while maintaining a higher rank, thereby improving the model's representational capacity and offering improved performance potential. The core idea is to freeze the original pretrained weights and selectively amplify or suppress important features of the original weights across multiple subspaces. The subspace mechanism provides an efficient way to increase the capacity and complexity of a model. We conduct both theoretical analyses and empirical studies on various tasks. Experiment results show that SMoA outperforms LoRA and its variants on 10 tasks, with extensive ablation studies validating its effectiveness.
>
---
#### [new 107] Measuring Iterative Temporal Reasoning with TimePuzzles
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出TimePuzzles任务，用于评估迭代时间推理能力。通过约束性日期推断问题，测试模型在无工具情况下的表现，并揭示其对工具依赖的不足。**

- **链接: [https://arxiv.org/pdf/2601.07148v1](https://arxiv.org/pdf/2601.07148v1)**

> **作者:** Zhengxiang Wang; Zeyu Dong
>
> **摘要:** We introduce TimePuzzles, a constraint-based date inference task for evaluating iterative temporal reasoning. Each puzzle combines factual temporal anchors with (cross-cultural) calendar relations, admits one or multiple valid solution dates, and is algorithmically generated for controlled, dynamic, and continual evaluation. Across 13 diverse LLMs, TimePuzzles well distinguishes their iterative temporal reasoning capabilities and remains challenging without tools: GPT-5 reaches only 49.3% accuracy and all other models stay below 31%, despite the dataset's simplicity. Web search consistently yields substantial gains and using code interpreter shows mixed effects, but all models perform much better when constraints are rewritten with explicit dates, revealing a gap in reliable tool use. Overall, TimePuzzles presents a simple, cost-effective diagnostic for tool-augmented iterative temporal reasoning.
>
---
#### [new 108] Exposía: Academic Writing Assessment of Exposés and Peer Feedback
- **分类: cs.CL**

- **简介: 该论文介绍Exposía数据集，用于学术写作评估与同伴反馈研究。任务是评估写作质量和反馈效果，解决如何有效利用AI进行学术写作评分的问题。工作包括构建数据集并测试大语言模型的评分能力。**

- **链接: [https://arxiv.org/pdf/2601.06536v1](https://arxiv.org/pdf/2601.06536v1)**

> **作者:** Dennis Zyska; Alla Rozovskaya; Ilia Kuznetsov; Iryna Gurevych
>
> **摘要:** We present Exposía, the first public dataset that connects writing and feedback assessment in higher education, enabling research on educationally grounded approaches to academic writing evaluation. Exposía includes student research project proposals and peer and instructor feedback consisting of comments and free-text reviews. The dataset was collected in the "Introduction to Scientific Work" course of the Computer Science undergraduate program that focuses on teaching academic writing skills and providing peer feedback on academic writing. Exposía reflects the multi-stage nature of the academic writing process that includes drafting, providing and receiving feedback, and revising the writing based on the feedback received. Both the project proposals and peer feedback are accompanied by human assessment scores based on a fine-grained, pedagogically-grounded schema for writing and feedback assessment that we develop. We use Exposía to benchmark state-of-the-art open-source large language models (LLMs) for two tasks: automated scoring of (1) the proposals and (2) the student reviews. The strongest LLMs attain high agreement on scoring aspects that require little domain knowledge but degrade on dimensions evaluating content, in line with human agreement values. We find that LLMs align better with the human instructors giving high scores. Finally, we establish that a prompting strategy that scores multiple aspects of the writing together is the most effective, an important finding for classroom deployment.
>
---
#### [new 109] Stylistic Evolution and LLM Neutrality in Singlish Language
- **分类: cs.CL**

- **简介: 论文研究Singlish语言的演变及大语言模型在其中的中立性问题，通过分析十年间的数字文本，评估语言变化和模型生成的时效性。属于自然语言处理中的语言演化与模型评估任务。**

- **链接: [https://arxiv.org/pdf/2601.06580v1](https://arxiv.org/pdf/2601.06580v1)**

> **作者:** Linus Tze En Foo; Weihan Angela Ng; Wenkai Li; Lynnette Hui Xian Ng
>
> **摘要:** Singlish is a creole rooted in Singapore's multilingual environment and continues to evolve alongside social and technological change. This study investigates the evolution of Singlish over a decade of informal digital text messages. We propose a stylistic similarity framework that compares lexico-structural, pragmatic, psycholinguistic, and encoder-derived features across years to quantify temporal variation. Our analysis reveals notable diachronic changes in tone, expressivity and sentence construction over the years. Conversely, while some LLMs were able to generate superficially realistic Singlish messages, they do not produce temporally neutral outputs, and residual temporal signals remain detectable despite prompting and fine-tuning. Our findings highlight the dynamic evolution of Singlish, as well as the capabilities and limitations of current LLMs in modeling sociolectal and temporal variations in the colloquial language.
>
---
#### [new 110] Evaluating Accounting Reasoning Capabilities of Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于会计推理任务，旨在评估大语言模型在会计领域的表现。通过设计评价标准，测试多个模型，发现GPT-4表现最佳，但现有模型仍不满足实际需求。**

- **链接: [https://arxiv.org/pdf/2601.06707v1](https://arxiv.org/pdf/2601.06707v1)**

> **作者:** Jie Zhou; Xin Chen; Jie Zhang; Hai Li; Jie Wang; Zhe Li
>
> **摘要:** Large language models are transforming learning, cognition, and research across many fields. Effectively integrating them into professional domains, such as accounting, is a key challenge for enterprise digital transformation. To address this, we define vertical domain accounting reasoning and propose evaluation criteria derived from an analysis of the training data characteristics of representative GLM models. These criteria support systematic study of accounting reasoning and provide benchmarks for performance improvement. Using this framework, we evaluate GLM-6B, GLM-130B, GLM-4, and OpenAI GPT-4 on accounting reasoning tasks. Results show that prompt design significantly affects performance, with GPT-4 demonstrating the strongest capability. Despite these gains, current models remain insufficient for real-world enterprise accounting, indicating the need for further optimization to unlock their full practical value.
>
---
#### [new 111] Evaluating Cross-Lingual Unlearning in Multilingual Language Models
- **分类: cs.CL**

- **简介: 该论文研究多语言大模型中的跨语言遗忘问题，评估不同遗忘算法效果，发现子空间投影方法更有效，揭示了语言间共享结构对遗忘的影响。**

- **链接: [https://arxiv.org/pdf/2601.06675v1](https://arxiv.org/pdf/2601.06675v1)**

> **作者:** Tyler Lizzo; Larry Heck
>
> **摘要:** We present the first comprehensive evaluation of cross-lingual unlearning in multilingual LLMs. Using translated TOFU benchmarks in seven language/script variants, we test major unlearning algorithms and show that most fail to remove facts outside the training language, even when utility remains high. However, subspace-projection consistently outperforms the other methods, achieving strong cross-lingual forgetting with minimal degradation. Analysis of learned task subspaces reveals a shared interlingua structure: removing this shared subspace harms all languages, while removing language-specific components selectively affects one. These results demonstrate that multilingual forgetting depends on geometry in weight space, motivating subspace-based approaches for future unlearning systems.
>
---
#### [new 112] Efficient and Reliable Estimation of Named Entity Linking Quality: A Case Study on GutBrainIE
- **分类: cs.CL**

- **简介: 该论文属于命名实体链接（NEL）质量评估任务，旨在解决大规模数据标注成本高、效率低的问题。通过采样框架，在有限预算下实现准确率的可靠估计。**

- **链接: [https://arxiv.org/pdf/2601.06624v1](https://arxiv.org/pdf/2601.06624v1)**

> **作者:** Marco Martinelli; Stefano Marchesin; Gianmaria Silvello
>
> **备注:** Submitted to IRCDL 2026: 22nd Conference on Information and Research Science Connecting to Digital and Library Science, February 19-20 2026, Modena, Italy
>
> **摘要:** Named Entity Linking (NEL) is a core component of biomedical Information Extraction (IE) pipelines, yet assessing its quality at scale is challenging due to the high cost of expert annotations and the large size of corpora. In this paper, we present a sampling-based framework to estimate the NEL accuracy of large-scale IE corpora under statistical guarantees and constrained annotation budgets. We frame NEL accuracy estimation as a constrained optimization problem, where the objective is to minimize expected annotation cost subject to a target Margin of Error (MoE) for the corpus-level accuracy estimate. Building on recent works on knowledge graph accuracy estimation, we adapt Stratified Two-Stage Cluster Sampling (STWCS) to the NEL setting, defining label-based strata and global surface-form clusters in a way that is independent of NEL annotations. Applied to 11,184 NEL annotations in GutBrainIE -- a new biomedical corpus openly released in fall 2025 -- our framework reaches a MoE $\leq 0.05$ by manually annotating only 2,749 triples (24.6%), leading to an overall accuracy estimate of $0.915 \pm 0.0473$. A time-based cost model and simulations against a Simple Random Sampling (SRS) baseline show that our design reduces expert annotation time by about 29% at fixed sample size. The framework is generic and can be applied to other NEL benchmarks and IE pipelines that require scalable and statistically robust accuracy assessment.
>
---
#### [new 113] Are Emotions Arranged in a Circle? Geometric Analysis of Emotion Representations via Hyperspherical Contrastive Learning
- **分类: cs.CL**

- **简介: 该论文属于情感表示学习任务，旨在探讨情绪在语言模型中的圆形结构。通过对比学习在超球面上构建圆形情绪表示，分析其优缺点。**

- **链接: [https://arxiv.org/pdf/2601.06575v1](https://arxiv.org/pdf/2601.06575v1)**

> **作者:** Yusuke Yamauchi; Akiko Aizawa
>
> **摘要:** Psychological research has long utilized circumplex models to structure emotions, placing similar emotions adjacently and opposing ones diagonally. Although frequently used to interpret deep learning representations, these models are rarely directly incorporated into the representation learning of language models, leaving their geometric validity unexplored. This paper proposes a method to induce circular emotion representations within language model embeddings via contrastive learning on a hypersphere. We show that while this circular alignment offers superior interpretability and robustness against dimensionality reduction, it underperforms compared to conventional designs in high-dimensional settings and fine-grained classification. Our findings elucidate the trade-offs involved in applying psychological circumplex models to deep learning architectures.
>
---
#### [new 114] X-Coder: Advancing Competitive Programming with Fully Synthetic Tasks, Solutions, and Tests
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于代码生成任务，旨在解决Code LLMs在竞赛编程中的推理能力不足问题。通过合成数据训练模型，提升其代码推理能力。**

- **链接: [https://arxiv.org/pdf/2601.06953v1](https://arxiv.org/pdf/2601.06953v1)**

> **作者:** Jie Wu; Haoling Li; Xin Zhang; Jiani Guo; Jane Luo; Steven Liu; Yangyu Huang; Ruihang Chu; Scarlett Li; Yujiu Yang
>
> **备注:** Project: https://github.com/JieWu02/X-Coder
>
> **摘要:** Competitive programming presents great challenges for Code LLMs due to its intensive reasoning demands and high logical complexity. However, current Code LLMs still rely heavily on real-world data, which limits their scalability. In this paper, we explore a fully synthetic approach: training Code LLMs with entirely generated tasks, solutions, and test cases, to empower code reasoning models without relying on real-world data. To support this, we leverage feature-based synthesis to propose a novel data synthesis pipeline called SynthSmith. SynthSmith shows strong potential in producing diverse and challenging tasks, along with verified solutions and tests, supporting both supervised fine-tuning and reinforcement learning. Based on the proposed synthetic SFT and RL datasets, we introduce the X-Coder model series, which achieves a notable pass rate of 62.9 avg@8 on LiveCodeBench v5 and 55.8 on v6, outperforming DeepCoder-14B-Preview and AReal-boba2-14B despite having only 7B parameters. In-depth analysis reveals that scaling laws hold on our synthetic dataset, and we explore which dimensions are more effective to scale. We further provide insights into code-centric reinforcement learning and highlight the key factors that shape performance through detailed ablations and analysis. Our findings demonstrate that scaling high-quality synthetic data and adopting staged training can greatly advance code reasoning, while mitigating reliance on real-world coding data.
>
---
#### [new 115] Reinforcement Learning for Chain of Thought Compression with One-Domain-to-All Generalization
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于语言模型优化任务，解决推理过程冗长问题。通过强化学习压缩思维链，在保持或提升准确率的同时减少响应长度，并实现跨领域泛化。**

- **链接: [https://arxiv.org/pdf/2601.06052v1](https://arxiv.org/pdf/2601.06052v1)**

> **作者:** Hanyu Li; Jiangshan Duo; Bofei Gao; Hailin Zhang; Sujian Li; Xiaotie Deng; Liang Zhao
>
> **摘要:** Chain-of-thought reasoning in large language models often creates an "overthinking trap," leading to excessive computational cost and latency for unreliable accuracy gains. Prior work has typically relied on global, static controls that risk penalizing necessary reasoning. We introduce a sample-level, soft reinforcement learning compression method that penalizes inefficiently long rollouts, but only on problems where the model has already mastered and already produced a more concise rollout. Our experiments show that this method reduces average response length by 20-40% with comparable or higher accuracy. Crucially, the compression exhibits strong cross-domain generalization; a model trained on math spontaneously shortens responses on unseen tasks like code, instruction following, and general knowledge QA, with stable or improved accuracy. We demonstrate a stable post-training curriculum (accuracy-compression-accuracy) that can ultimately produce models that are more accurate and reason more concisely, arguing that such compression method should be a standard phase in developing efficient reasoning models.
>
---
#### [new 116] Contrastive Learning with Narrative Twins for Modeling Story Salience
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的叙事理解任务，旨在解决故事中事件显著性建模问题。通过对比学习，利用叙事双胞胎生成故事嵌入，并验证不同操作对显著性判断的效果。**

- **链接: [https://arxiv.org/pdf/2601.07765v1](https://arxiv.org/pdf/2601.07765v1)**

> **作者:** Igor Sterner; Alex Lascarides; Frank Keller
>
> **备注:** EACL 2026
>
> **摘要:** Understanding narratives requires identifying which events are most salient for a story's progression. We present a contrastive learning framework for modeling narrative salience that learns story embeddings from narrative twins: stories that share the same plot but differ in surface form. Our model is trained to distinguish a story from both its narrative twin and a distractor with similar surface features but different plot. Using the resulting embeddings, we evaluate four narratologically motivated operations for inferring salience (deletion, shifting, disruption, and summarization). Experiments on short narratives from the ROCStories corpus and longer Wikipedia plot summaries show that contrastively learned story embeddings outperform a masked-language-model baseline, and that summarization is the most reliable operation for identifying salient sentences. If narrative twins are not available, random dropout can be used to generate the twins from a single story. Effective distractors can be obtained either by prompting LLMs or, in long-form narratives, by using different parts of the same story.
>
---
#### [new 117] Detecting LLM-Generated Text with Performance Guarantees
- **分类: cs.CL; cs.LG; stat.AP; stat.ML**

- **简介: 该论文属于文本检测任务，旨在识别LLM生成的文本。通过训练分类器，解决虚假信息传播问题，实现高效准确的检测。**

- **链接: [https://arxiv.org/pdf/2601.06586v1](https://arxiv.org/pdf/2601.06586v1)**

> **作者:** Hongyi Zhou; Jin Zhu; Ying Yang; Chengchun Shi
>
> **摘要:** Large language models (LLMs) such as GPT, Claude, Gemini, and Grok have been deeply integrated into our daily life. They now support a wide range of tasks -- from dialogue and email drafting to assisting with teaching and coding, serving as search engines, and much more. However, their ability to produce highly human-like text raises serious concerns, including the spread of fake news, the generation of misleading governmental reports, and academic misconduct. To address this practical problem, we train a classifier to determine whether a piece of text is authored by an LLM or a human. Our detector is deployed on an online CPU-based platform https://huggingface.co/spaces/stats-powered-ai/StatDetectLLM, and contains three novelties over existing detectors: (i) it does not rely on auxiliary information, such as watermarks or knowledge of the specific LLM used to generate the text; (ii) it more effectively distinguishes between human- and LLM-authored text; and (iii) it enables statistical inference, which is largely absent in the current literature. Empirically, our classifier achieves higher classification accuracy compared to existing detectors, while maintaining type-I error control, high statistical power, and computational efficiency.
>
---
#### [new 118] A Multi-Stage Workflow for the Review of Marketing Content with Reasoning Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于内容审核任务，旨在解决营销内容合规性问题。通过多阶段流程利用微调大模型进行自动检测与评估。**

- **链接: [https://arxiv.org/pdf/2601.06054v1](https://arxiv.org/pdf/2601.06054v1)**

> **作者:** Alberto Purpura; Emily Chen; Swapnil Shinde
>
> **摘要:** Reasoning Large Language Models (LLMs) have shown promising results when tasked with solving complex problems. In this paper, we propose and evaluate a multi-stage workflow that leverages the capabilities of fine-tuned reasoning LLMs to assist in the review process of marketing content, making sure they comply with a given list of requirements. The contributions of this paper are the following: (i) we present a novel approach -- that does not rely on any external knowledge representation -- for the automatic identification of compliance issues in textual content; (ii) compare the effectiveness of different fine-tuning strategies like Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO) in training models to solve this problem; (iii) we evaluate the effectiveness of training small LLMs to generate reasoning tokens before providing their final response; (iv) we evaluate how the choice and combinations of different reward functions affects the performance of a model trained with GRPO.
>
---
#### [new 119] Emotional Support Evaluation Framework via Controllable and Diverse Seeker Simulator
- **分类: cs.CL**

- **简介: 该论文属于情感支持聊天机器人评估任务，旨在解决现有模拟器缺乏行为多样性和可控性的问题。通过构建基于九个特征的可控模拟器，提升评估的准确性和全面性。**

- **链接: [https://arxiv.org/pdf/2601.07698v1](https://arxiv.org/pdf/2601.07698v1)**

> **作者:** Chaewon Heo; Cheyon Jin; Yohan Jo
>
> **摘要:** As emotional support chatbots have recently gained significant traction across both research and industry, a common evaluation strategy has emerged: use help-seeker simulators to interact with supporter chatbots. However, current simulators suffer from two critical limitations: (1) they fail to capture the behavioral diversity of real-world seekers, often portraying them as overly cooperative, and (2) they lack the controllability required to simulate specific seeker profiles. To address these challenges, we present a controllable seeker simulator driven by nine psychological and linguistic features that underpin seeker behavior. Using authentic Reddit conversations, we train our model via a Mixture-of-Experts (MoE) architecture, which effectively differentiates diverse seeker behaviors into specialized parameter subspaces, thereby enhancing fine-grained controllability. Our simulator achieves superior profile adherence and behavioral diversity compared to existing approaches. Furthermore, evaluating 7 prominent supporter models with our system uncovers previously obscured performance degradations. These findings underscore the utility of our framework in providing a more faithful and stress-tested evaluation for emotional support chatbots.
>
---
#### [new 120] TreePS-RAG: Tree-based Process Supervision for Reinforcement Learning in Agentic RAG
- **分类: cs.CL**

- **简介: 该论文属于问答任务，解决RL在Agentic RAG中的步骤信用分配问题。提出TreePS-RAG框架，通过树结构实现在线步骤奖励估计，提升性能。**

- **链接: [https://arxiv.org/pdf/2601.06922v1](https://arxiv.org/pdf/2601.06922v1)**

> **作者:** Tianhua Zhang; Kun Li; Junan Li; Yunxiang Li; Hongyin Luo; Xixin Wu; James Glass; Helen Meng
>
> **摘要:** Agentic retrieval-augmented generation (RAG) formulates question answering as a multi-step interaction between reasoning and information retrieval, and has recently been advanced by reinforcement learning (RL) with outcome-based supervision. While effective, relying solely on sparse final rewards limits step-wise credit assignment and provides weak guidance for intermediate reasoning and actions. Recent efforts explore process-level supervision, but typically depend on offline constructed training data, which risks distribution shift, or require costly intermediate annotations. We present TreePS-RAG, an online, tree-based RL framework for agentic RAG that enables step-wise credit assignment while retaining standard outcome-only rewards. Our key insight is to model agentic RAG reasoning as a rollout tree, where each reasoning step naturally maps to a node. This tree structure allows step utility to be estimated via Monte Carlo estimation over its descendant outcomes, yielding fine-grained process advantages without requiring intermediate labels. To make this paradigm practical, we introduce an efficient online tree construction strategy that preserves exploration diversity under a constrained computational budget. With a rollout cost comparable to strong baselines like Search-R1, experiments on seven multi-hop and general QA benchmarks across multiple model scales show that TreePS-RAG consistently and significantly outperforms both outcome-supervised and leading process-supervised RL methods.
>
---
#### [new 121] Will it Merge? On The Causes of Model Mergeability
- **分类: cs.CL**

- **简介: 该论文属于模型融合任务，旨在解决模型合并成功与否的原因。通过定义mergeability，分析基模型知识对合并效果的影响，并提出一种改进的加权融合方法。**

- **链接: [https://arxiv.org/pdf/2601.06672v1](https://arxiv.org/pdf/2601.06672v1)**

> **作者:** Adir Rahamim; Asaf Yehudai; Boaz Carmeli; Leshem Choshen; Yosi Mass; Yonatan Belinkov
>
> **摘要:** Model merging has emerged as a promising technique for combining multiple fine-tuned models into a single multitask model without retraining. However, the factors that determine whether merging will succeed or fail remain poorly understood. In this work, we investigate why specific models are merged better than others. To do so, we propose a concrete, measurable definition of mergeability. We investigate several potential causes for high or low mergeability, highlighting the base model knowledge as a dominant factor: Models fine-tuned on instances that the base model knows better are more mergeable than models fine-tuned on instances that the base model struggles with. Based on our mergeability definition, we explore a simple weighted merging technique that better preserves weak knowledge in the base model.
>
---
#### [new 122] Integrating Machine-Generated Short Descriptions into the Wikipedia Android App: A Pilot Deployment of Descartes
- **分类: cs.CL**

- **简介: 该论文属于自然语言生成任务，旨在解决维基百科短描述覆盖不均的问题。通过部署Descartes模型，向编辑提供生成建议，提升内容质量与覆盖率。**

- **链接: [https://arxiv.org/pdf/2601.07631v1](https://arxiv.org/pdf/2601.07631v1)**

> **作者:** Marija Šakota; Dmitry Brant; Cooltey Feng; Shay Nowick; Amal Ramadan; Robin Schoenbaechler; Joseph Seddon; Jazmin Tanner; Isaac Johnson; Robert West
>
> **摘要:** Short descriptions are a key part of the Wikipedia user experience, but their coverage remains uneven across languages and topics. In previous work, we introduced Descartes, a multilingual model for generating short descriptions. In this report, we present the results of a pilot deployment of Descartes in the Wikipedia Android app, where editors were offered suggestions based on outputs from Descartes while editing short descriptions. The experiment spanned 12 languages, with over 3,900 articles and 375 editors participating. Overall, 90% of accepted Descartes descriptions were rated at least 3 out of 5 in quality, and their average ratings were comparable to human-written ones. Editors adopted machine suggestions both directly and with modifications, while the rate of reverts and reports remained low. The pilot also revealed practical considerations for deployment, including latency, language-specific gaps, and the need for safeguards around sensitive topics. These results indicate that Descartes's short descriptions can support editors in reducing content gaps, provided that technical, design, and community guardrails are in place.
>
---
#### [new 123] The Need for a Socially-Grounded Persona Framework for User Simulation
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于用户模拟任务，旨在解决传统人口统计学角色的局限性。提出SCOPE框架，通过社会心理结构提升角色质量，改善行为预测和减少偏差。**

- **链接: [https://arxiv.org/pdf/2601.07110v1](https://arxiv.org/pdf/2601.07110v1)**

> **作者:** Pranav Narayanan Venkit; Yu Li; Yada Pruksachatkun; Chien-Sheng Wu
>
> **摘要:** Synthetic personas are widely used to condition large language models (LLMs) for social simulation, yet most personas are still constructed from coarse sociodemographic attributes or summaries. We revisit persona creation by introducing SCOPE, a socially grounded framework for persona construction and evaluation, built from a 141-item, two-hour sociopsychological protocol collected from 124 U.S.-based participants. Across seven models, we find that demographic-only personas are a structural bottleneck: demographics explain only ~1.5% of variance in human response similarity. Adding sociopsychological facets improves behavioral prediction and reduces over-accentuation, and non-demographic personas based on values and identity achieve strong alignment with substantially lower bias. These trends generalize to SimBench (441 aligned questions), where SCOPE personas outperform default prompting and NVIDIA Nemotron personas, and SCOPE augmentation improves Nemotron-based personas. Our results indicate that persona quality depends on sociopsychological structure rather than demographic templates or summaries.
>
---
#### [new 124] How to predict creativity ratings from written narratives: A comparison of co-occurrence and textual forma mentis networks
- **分类: cs.CL**

- **简介: 该论文属于创意评估任务，旨在通过语义网络预测创造力评分。比较了共现网络与文本形式心智网络，提出有效建模方法。**

- **链接: [https://arxiv.org/pdf/2601.07327v1](https://arxiv.org/pdf/2601.07327v1)**

> **作者:** Roberto Passaro; Edith Haim; Massimo Stella
>
> **摘要:** This tutorial paper provides a step-by-step workflow for building and analysing semantic networks from short creative texts. We introduce and compare two widely used text-to-network approaches: word co-occurrence networks and textual forma mentis networks (TFMNs). We also demonstrate how they can be used in machine learning to predict human creativity ratings. Using a corpus of 1029 short stories, we guide readers through text preprocessing, network construction, feature extraction (structural measures, spreading-activation indices, and emotion scores), and application of regression models. We evaluate how network-construction choices influence both network topology and predictive performance. Across all modelling settings, TFMNs consistently outperformed co-occurrence networks through lower prediction errors (best MAE = 0.581 for TFMN, vs 0.592 for co-occurrence with window size 3). Network-structural features dominated predictive performance (MAE = 0.591 for TFMN), whereas emotion features performed worse (MAE = 0.711 for TFMN) and spreading-activation measures contributed little (MAE = 0.788 for TFMN). This paper offers practical guidance for researchers interested in applying network-based methods for cognitive fields like creativity research. we show when syntactic networks are preferable to surface co-occurrence models, and provide an open, reproducible workflow accessible to newcomers in the field, while also offering deeper methodological insight for experienced researchers.
>
---
#### [new 125] ES-Mem: Event Segmentation-Based Memory for Long-Term Dialogue Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话系统任务，旨在解决长对话中记忆碎片化和检索不准确的问题。提出ES-Mem框架，通过事件分割和分层记忆结构提升记忆的连贯性与定位精度。**

- **链接: [https://arxiv.org/pdf/2601.07582v1](https://arxiv.org/pdf/2601.07582v1)**

> **作者:** Huhai Zou; Tianhao Sun; Chuanjiang He; Yu Tian; Zhenyang Li; Li Jin; Nayu Liu; Jiang Zhong; Kaiwen Wei
>
> **摘要:** Memory is critical for dialogue agents to maintain coherence and enable continuous adaptation in long-term interactions. While existing memory mechanisms offer basic storage and retrieval capabilities, they are hindered by two primary limitations: (1) rigid memory granularity often disrupts semantic integrity, resulting in fragmented and incoherent memory units; (2) prevalent flat retrieval paradigms rely solely on surface-level semantic similarity, neglecting the structural cues of discourse required to navigate and locate specific episodic contexts. To mitigate these limitations, drawing inspiration from Event Segmentation Theory, we propose ES-Mem, a framework incorporating two core components: (1) a dynamic event segmentation module that partitions long-term interactions into semantically coherent events with distinct boundaries; (2) a hierarchical memory architecture that constructs multi-layered memories and leverages boundary semantics to anchor specific episodic memory for precise context localization. Evaluations on two memory benchmarks demonstrate that ES-Mem yields consistent performance gains over baseline methods. Furthermore, the proposed event segmentation module exhibits robust applicability on dialogue segmentation datasets.
>
---
#### [new 126] MedTutor: A Retrieval-Augmented LLM System for Case-Based Medical Education
- **分类: cs.CL**

- **简介: 该论文提出MedTutor系统，属于医学教育任务，旨在解决医学生从病例中高效获取知识的问题。通过RAG技术生成教育内容和试题，提升学习效率。**

- **链接: [https://arxiv.org/pdf/2601.06979v1](https://arxiv.org/pdf/2601.06979v1)**

> **作者:** Dongsuk Jang; Ziyao Shangguan; Kyle Tegtmeyer; Anurag Gupta; Jan Czerminski; Sophie Chheang; Arman Cohan
>
> **备注:** Accepted to EMNLP 2025 (System Demonstrations)
>
> **摘要:** The learning process for medical residents presents significant challenges, demanding both the ability to interpret complex case reports and the rapid acquisition of accurate medical knowledge from reliable sources. Residents typically study case reports and engage in discussions with peers and mentors, but finding relevant educational materials and evidence to support their learning from these cases is often time-consuming and challenging. To address this, we introduce MedTutor, a novel system designed to augment resident training by automatically generating evidence-based educational content and multiple-choice questions from clinical case reports. MedTutor leverages a Retrieval-Augmented Generation (RAG) pipeline that takes clinical case reports as input and produces targeted educational materials. The system's architecture features a hybrid retrieval mechanism that synergistically queries a local knowledge base of medical textbooks and academic literature (using PubMed, Semantic Scholar APIs) for the latest related research, ensuring the generated content is both foundationally sound and current. The retrieved evidence is filtered and ordered using a state-of-the-art reranking model and then an LLM generates the final long-form output describing the main educational content regarding the case-report. We conduct a rigorous evaluation of the system. First, three radiologists assessed the quality of outputs, finding them to be of high clinical and educational value. Second, we perform a large scale evaluation using an LLM-as-a Judge to understand if LLMs can be used to evaluate the output of the system. Our analysis using correlation between LLMs outputs and human expert judgments reveals a moderate alignment and highlights the continued necessity of expert oversight.
>
---
#### [new 127] Atomic-SNLI: Fine-Grained Natural Language Inference through Atomic Fact Decomposition
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言推理任务，旨在解决现有系统在原子层面推理能力不足的问题。通过构建Atomic-SNLI数据集，提升模型在细粒度事实推理上的表现。**

- **链接: [https://arxiv.org/pdf/2601.06528v1](https://arxiv.org/pdf/2601.06528v1)**

> **作者:** Minghui Huang
>
> **摘要:** Current Natural Language Inference (NLI) systems primarily operate at the sentence level, providing black-box decisions that lack explanatory power. While atomic-level NLI offers a promising alternative by decomposing hypotheses into individual facts, we demonstrate that the conventional assumption that a hypothesis is entailed only when all its atomic facts are entailed fails in practice due to models' poor performance on fine-grained reasoning. Our analysis reveals that existing models perform substantially worse on atomic level inference compared to sentence level tasks. To address this limitation, we introduce Atomic-SNLI, a novel dataset constructed by decomposing SNLI and enriching it with carefully curated atomic level examples through linguistically informed generation strategies. Experimental results demonstrate that models fine-tuned on Atomic-SNLI achieve significant improvements in atomic reasoning capabilities while maintaining strong sentence level performance, enabling both accurate judgements and transparent, explainable results at the fact level.
>
---
#### [new 128] PDR: A Plug-and-Play Positional Decay Framework for LLM Pre-training Data Detection
- **分类: cs.CL**

- **简介: 该论文属于LLM预训练数据检测任务，旨在解决黑盒、零样本环境下数据隐私和版权审计难题。提出PDR框架，通过位置衰减重加权提升检测效果。**

- **链接: [https://arxiv.org/pdf/2601.06827v1](https://arxiv.org/pdf/2601.06827v1)**

> **作者:** Jinhan Liu; Yibo Yang; Ruiying Lu; Piotr Piekos; Yimeng Chen; Peng Wang; Dandan Guo
>
> **摘要:** Detecting pre-training data in Large Language Models (LLMs) is crucial for auditing data privacy and copyright compliance, yet it remains challenging in black-box, zero-shot settings where computational resources and training data are scarce. While existing likelihood-based methods have shown promise, they typically aggregate token-level scores using uniform weights, thereby neglecting the inherent information-theoretic dynamics of autoregressive generation. In this paper, we hypothesize and empirically validate that memorization signals are heavily skewed towards the high-entropy initial tokens, where model uncertainty is highest, and decay as context accumulates. To leverage this linguistic property, we introduce Positional Decay Reweighting (PDR), a training-free and plug-and-play framework. PDR explicitly reweights token-level scores to amplify distinct signals from early positions while suppressing noise from later ones. Extensive experiments show that PDR acts as a robust prior and can usually enhance a wide range of advanced methods across multiple benchmarks.
>
---
#### [new 129] MI-PRUN: Optimize Large Language Model Pruning via Mutual Information
- **分类: cs.CL**

- **简介: 该论文属于模型压缩任务，旨在解决大语言模型冗余问题。提出MI-PRUN方法，利用互信息和DPI进行块剪枝，提升压缩效果与效率。**

- **链接: [https://arxiv.org/pdf/2601.07212v1](https://arxiv.org/pdf/2601.07212v1)**

> **作者:** Hao Zhang; Zhibin Zhang; Guangxin Wu; He Chen; Jiafeng Guo; Xueqi Cheng
>
> **备注:** 10 pages
>
> **摘要:** Large Language Models (LLMs) have become indispensable across various domains, but this comes at the cost of substantial computational and memory resources. Model pruning addresses this by removing redundant components from models. In particular, block pruning can achieve significant compression and inference acceleration. However, existing block pruning methods are often unstable and struggle to attain globally optimal solutions. In this paper, we propose a mutual information based pruning method MI-PRUN for LLMs. Specifically, we leverages mutual information to identify redundant blocks by evaluating transitions in hidden states. Additionally, we incorporate the Data Processing Inequality (DPI) to reveal the relationship between the importance of entire contiguous blocks and that of individual blocks. Moreover, we develop the Fast-Block-Select algorithm, which iteratively updates block combinations to achieve a globally optimal solution while significantly improving the efficiency. Extensive experiments across various models and datasets demonstrate the stability and effectiveness of our method.
>
---
#### [new 130] MTMCS-Bench: Evaluating Contextual Safety of Multimodal Large Language Models in Multi-Turn Dialogues
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模态大语言模型的安全评估任务，旨在解决多轮对话中上下文安全问题。提出MTMCS-Bench基准，评估模型在不同风险场景下的安全性和实用性。**

- **链接: [https://arxiv.org/pdf/2601.06757v1](https://arxiv.org/pdf/2601.06757v1)**

> **作者:** Zheyuan Liu; Dongwhi Kim; Yixin Wan; Xiangchi Yuan; Zhaoxuan Tan; Fengran Mo; Meng Jiang
>
> **备注:** A benchmark of realistic images and multi-turn conversations that evaluates contextual safety in MLLMs under two complementary settings
>
> **摘要:** Multimodal large language models (MLLMs) are increasingly deployed as assistants that interact through text and images, making it crucial to evaluate contextual safety when risk depends on both the visual scene and the evolving dialogue. Existing contextual safety benchmarks are mostly single-turn and often miss how malicious intent can emerge gradually or how the same scene can support both benign and exploitative goals. We introduce the Multi-Turn Multimodal Contextual Safety Benchmark (MTMCS-Bench), a benchmark of realistic images and multi-turn conversations that evaluates contextual safety in MLLMs under two complementary settings, escalation-based risk and context-switch risk. MTMCS-Bench offers paired safe and unsafe dialogues with structured evaluation. It contains over 30 thousand multimodal (image+text) and unimodal (text-only) samples, with metrics that separately measure contextual intent recognition, safety-awareness on unsafe cases, and helpfulness on benign ones. Across eight open-source and seven proprietary MLLMs, we observe persistent trade-offs between contextual safety and utility, with models tending to either miss gradual risks or over-refuse benign dialogues. Finally, we evaluate five current guardrails and find that they mitigate some failures but do not fully resolve multi-turn contextual risks.
>
---
#### [new 131] Is Agentic RAG worth it? An experimental comparison of RAG approaches
- **分类: cs.CL**

- **简介: 该论文属于信息检索与生成任务，旨在比较Enhanced RAG与Agentic RAG的优劣，解决不同场景下的性能与成本问题。通过实验分析两种方法的适用性。**

- **链接: [https://arxiv.org/pdf/2601.07711v1](https://arxiv.org/pdf/2601.07711v1)**

> **作者:** Pietro Ferrazzi; Milica Cvjeticanin; Alessio Piraccini; Davide Giannuzzi
>
> **摘要:** Retrieval-Augmented Generation (RAG) systems are usually defined by the combination of a generator and a retrieval component that extracts textual context from a knowledge base to answer user queries. However, such basic implementations exhibit several limitations, including noisy or suboptimal retrieval, misuse of retrieval for out-of-scope queries, weak query-document matching, and variability or cost associated with the generator. These shortcomings have motivated the development of "Enhanced" RAG, where dedicated modules are introduced to address specific weaknesses in the workflow. More recently, the growing self-reflective capabilities of Large Language Models (LLMs) have enabled a new paradigm, which we refer to as "Agentic" RAG. In this approach, the LLM orchestrates the entire process-deciding which actions to perform, when to perform them, and whether to iterate-thereby reducing reliance on fixed, manually engineered modules. Despite the rapid adoption of both paradigms, it remains unclear which approach is preferable under which conditions. In this work, we conduct an extensive, empirically driven evaluation of Enhanced and Agentic RAG across multiple scenarios and dimensions. Our results provide practical insights into the trade-offs between the two paradigms, offering guidance on selecting the most effective RAG design for real-world applications, considering both costs and performance.
>
---
#### [new 132] Lexical and Statistical Analysis of Bangla Newspaper and Literature: A Corpus-Driven Study on Diversity, Readability, and NLP Adaptation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在分析孟加拉语文本的词汇多样性、可读性及模型适应性。通过对比文学与新闻语料，研究其语言特征差异及对模型性能的影响。**

- **链接: [https://arxiv.org/pdf/2601.06041v1](https://arxiv.org/pdf/2601.06041v1)**

> **作者:** Pramit Bhattacharyya; Arnab Bhattacharya
>
> **摘要:** In this paper, we present a comprehensive corpus-driven analysis of Bangla literary and newspaper texts to investigate their lexical diversity, structural complexity and readability. We undertook Vacaspati and IndicCorp, which are the most extensive literature and newspaper-only corpora for Bangla. We examine key linguistic properties, including the type-token ratio (TTR), hapax legomena ratio (HLR), Bigram diversity, average syllable and word lengths, and adherence to Zipfs Law, for both newspaper (IndicCorp) and literary corpora (Vacaspati).For all the features, such as Bigram Diversity and HLR, despite its smaller size, the literary corpus exhibits significantly higher lexical richness and structural variation. Additionally, we tried to understand the diversity of corpora by building n-gram models and measuring perplexity. Our findings reveal that literary corpora have higher perplexity than newspaper corpora, even for similar sentence sizes. This trend can also be observed for the English newspaper and literature corpus, indicating its generalizability. We also examined how the perfor- mance of models on downstream tasks is influenced by the inclusion of literary data alongside newspaper data. Our findings suggest that inte- grating literary data with newspapers improves the performance of models on various downstream tasks. We have also demonstrated that a literary corpus adheres more closely to global word distribution proper- ties, such as Zipfs law, than a newspaper corpus or a merged corpus of both literary and newspaper texts. Literature corpora also have higher entropy and lower redundancy values compared to a newspaper corpus. We also further assess the readability using Flesch and Coleman-Liau in- dices, showing that literary texts are more complex.
>
---
#### [new 133] Document-Level Zero-Shot Relation Extraction with Entity Side Information
- **分类: cs.CL**

- **简介: 该论文属于文档级零样本关系抽取任务，旨在解决低资源语言中依赖LLM生成数据的不足。通过引入实体旁信息，提出DocZSRE-SI框架，提升抽取效果。**

- **链接: [https://arxiv.org/pdf/2601.07271v1](https://arxiv.org/pdf/2601.07271v1)**

> **作者:** Mohan Raj Chanthran; Soon Lay Ki; Ong Huey Fang; Bhawani Selvaretnam
>
> **备注:** Accepted to EACL 2026 Main Conference
>
> **摘要:** Document-Level Zero-Shot Relation Extraction (DocZSRE) aims to predict unseen relation labels in text documents without prior training on specific relations. Existing approaches rely on Large Language Models (LLMs) to generate synthetic data for unseen labels, which poses challenges for low-resource languages like Malaysian English. These challenges include the incorporation of local linguistic nuances and the risk of factual inaccuracies in LLM-generated data. This paper introduces Document-Level Zero-Shot Relation Extraction with Entity Side Information (DocZSRE-SI) to address limitations in the existing DocZSRE approach. The DocZSRE-SI framework leverages Entity Side Information, such as Entity Mention Descriptions and Entity Mention Hypernyms, to perform ZSRE without depending on LLM-generated synthetic data. The proposed low-complexity model achieves an average improvement of 11.6% in the macro F1-Score compared to baseline models and existing benchmarks. By utilizing Entity Side Information, DocZSRE-SI offers a robust and efficient alternative to error-prone, LLM-based methods, demonstrating significant advancements in handling low-resource languages and linguistic diversity in relation extraction tasks. This research provides a scalable and reliable solution for ZSRE, particularly in contexts like Malaysian English news articles, where traditional LLM-based approaches fall short.
>
---
#### [new 134] N2N-GQA: Noise-to-Narrative for Graph-Based Table-Text Question Answering Using LLMs
- **分类: cs.CL**

- **简介: 该论文提出N2N-GQA，解决多跳问答任务中检索噪声影响推理的问题。通过构建动态证据图，提升跨表文本的多跳推理效果。**

- **链接: [https://arxiv.org/pdf/2601.06603v1](https://arxiv.org/pdf/2601.06603v1)**

> **作者:** Mohamed Sharafath; Aravindh Annamalai; Ganesh Murugan; Aravindakumar Venugopalan
>
> **备注:** Accepted at an AAAI 2026 Workshop
>
> **摘要:** Multi-hop question answering over hybrid table-text data requires retrieving and reasoning across multiple evidence pieces from large corpora, but standard Retrieval-Augmented Generation (RAG) pipelines process documents as flat ranked lists, causing retrieval noise to obscure reasoning chains. We introduce N2N-GQA. To our knowledge, it is the first zeroshot framework for open-domain hybrid table-text QA that constructs dynamic evidence graphs from noisy retrieval outputs. Our key insight is that multi-hop reasoning requires understanding relationships between evidence pieces: by modeling documents as graph nodes with semantic relationships as edges, we identify bridge documents connecting reasoning steps, a capability absent in list-based retrieval. On OTT-QA, graph-based evidence curation provides a 19.9-point EM improvement over strong baselines, demonstrating that organizing retrieval results as structured graphs is critical for multihop reasoning. N2N-GQA achieves 48.80 EM, matching finetuned retrieval models (CORE: 49.0 EM) and approaching heavily optimized systems (COS: 56.9 EM) without any task specific training. This establishes graph-structured evidence organization as essential for scalable, zero-shot multi-hop QA systems and demonstrates that simple, interpretable graph construction can rival sophisticated fine-tuned approaches.
>
---
#### [new 135] SAD: A Large-Scale Strategic Argumentative Dialogue Dataset
- **分类: cs.CL**

- **简介: 该论文提出SAD数据集，用于多轮策略性辩论对话生成。解决传统数据集缺乏互动性和多策略标注的问题，支持更真实的论辩建模。**

- **链接: [https://arxiv.org/pdf/2601.07423v1](https://arxiv.org/pdf/2601.07423v1)**

> **作者:** Yongkang Liu; Jiayang Yu; Mingyang Wang; Yiqun Zhang; Ercong Nie; Shi Feng; Daling Wang; Kaisong Song; Hinrich Schütze
>
> **备注:** under review
>
> **摘要:** Argumentation generation has attracted substantial research interest due to its central role in human reasoning and decision-making. However, most existing argumentative corpora focus on non-interactive, single-turn settings, either generating arguments from a given topic or refuting an existing argument. In practice, however, argumentation is often realized as multi-turn dialogue, where speakers defend their stances and employ diverse argumentative strategies to strengthen persuasiveness. To support deeper modeling of argumentation dialogue, we present the first large-scale \textbf{S}trategic \textbf{A}rgumentative \textbf{D}ialogue dataset, SAD, consisting of 392,822 examples. Grounded in argumentation theories, we annotate each utterance with five strategy types, allowing multiple strategies per utterance. Unlike prior datasets, SAD requires models to generate contextually appropriate arguments conditioned on the dialogue history, a specified stance on the topic, and targeted argumentation strategies. We further benchmark a range of pretrained generative models on SAD and present in-depth analysis of strategy usage patterns in argumentation.
>
---
#### [new 136] Steer Model beyond Assistant: Controlling System Prompt Strength via Contrastive Decoding
- **分类: cs.CL**

- **简介: 该论文属于语言模型控制任务，解决模型难以偏离助手角色的问题。提出系统提示强度方法，通过对比解码调节模型行为，无需重新训练即可实现动态控制。**

- **链接: [https://arxiv.org/pdf/2601.06403v1](https://arxiv.org/pdf/2601.06403v1)**

> **作者:** Yijiang River Dong; Tiancheng Hu; Zheng Hui; Nigel Collier
>
> **摘要:** Large language models excel at complex instructions yet struggle to deviate from their helpful assistant persona, as post-training instills strong priors that resist conflicting instructions. We introduce system prompt strength, a training-free method that treats prompt adherence as a continuous control. By contrasting logits from target and default system prompts, we isolate and amplify the behavioral signal unique to the target persona by a scalar factor alpha. Across five diverse benchmarks spanning constraint satisfaction, behavioral control, pluralistic alignment, capability modulation, and stylistic control, our method yields substantial improvements: up to +8.5 strict accuracy on IFEval, +45pp refusal rate on OffTopicEval, and +13% steerability on Prompt-Steering. Our approach enables practitioners to modulate system prompt strength, providing dynamic control over model behavior without retraining.
>
---
#### [new 137] IndRegBias: A Dataset for Studying Indian Regional Biases in English and Code-Mixed Social Media Comments
- **分类: cs.CL; cs.CY; cs.SI**

- **简介: 该论文属于自然语言处理中的偏见检测任务，旨在解决印度地区偏见识别问题。研究构建了IndRegBias数据集，并评估了模型在不同策略下的检测效果。**

- **链接: [https://arxiv.org/pdf/2601.06477v1](https://arxiv.org/pdf/2601.06477v1)**

> **作者:** Debasmita Panda; Akash Anil; Neelesh Kumar Shukla
>
> **备注:** Preprint. Under review
>
> **摘要:** Warning: This paper consists of examples representing regional biases in Indian regions that might be offensive towards a particular region. While social biases corresponding to gender, race, socio-economic conditions, etc., have been extensively studied in the major applications of Natural Language Processing (NLP), biases corresponding to regions have garnered less attention. This is mainly because of (i) difficulty in the extraction of regional bias datasets, (ii) disagreements in annotation due to inherent human biases, and (iii) regional biases being studied in combination with other types of social biases and often being under-represented. This paper focuses on creating a dataset IndRegBias, consisting of regional biases in an Indian context reflected in users' comments on popular social media platforms, namely Reddit and YouTube. We carefully selected 25,000 comments appearing on various threads in Reddit and videos on YouTube discussing trending topics on regional issues in India. Furthermore, we propose a multilevel annotation strategy to annotate the comments describing the severity of regional biased statements. To detect the presence of regional bias and its severity in IndRegBias, we evaluate open-source Large Language Models (LLMs) and Indic Language Models (ILMs) using zero-shot, few-shot, and fine-tuning strategies. We observe that zero-shot and few-shot approaches show lower accuracy in detecting regional biases and severity in the majority of the LLMs and ILMs. However, the fine-tuning approach significantly enhances the performance of the LLM in detecting Indian regional bias along with its severity.
>
---
#### [new 138] TurkBench: A Benchmark for Evaluating Turkish Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出TurkBench，一个用于评估土耳其语大语言模型的基准。旨在解决多语言模型评估不足的问题，通过21个子任务涵盖知识、语言理解等六大类别，提供全面评测工具。**

- **链接: [https://arxiv.org/pdf/2601.07020v1](https://arxiv.org/pdf/2601.07020v1)**

> **作者:** Çağrı Toraman; Ahmet Kaan Sever; Ayse Aysu Cengiz; Elif Ecem Arslan; Görkem Sevinç; Mete Mert Birdal; Yusuf Faruk Güldemir; Ali Buğra Kanburoğlu; Sezen Felekoğlu; Osman Gürlek; Sarp Kantar; Birsen Şahin Kütük; Büşra Tufan; Elif Genç; Serkan Coşkun; Gupse Ekin Demir; Muhammed Emin Arayıcı; Olgun Dursun; Onur Gungor; Susan Üsküdarlı; Abdullah Topraksoy; Esra Darıcı
>
> **摘要:** With the recent surge in the development of large language models, the need for comprehensive and language-specific evaluation benchmarks has become critical. While significant progress has been made in evaluating English language models, benchmarks for other languages, particularly those with unique linguistic characteristics such as Turkish, remain less developed. Our study introduces TurkBench, a comprehensive benchmark designed to assess the capabilities of generative large language models in the Turkish language. TurkBench involves 8,151 data samples across 21 distinct subtasks. These are organized under six main categories of evaluation: Knowledge, Language Understanding, Reasoning, Content Moderation, Turkish Grammar and Vocabulary, and Instruction Following. The diverse range of tasks and the culturally relevant data would provide researchers and developers with a valuable tool for evaluating their models and identifying areas for improvement. We further publish our benchmark for online submissions at https://huggingface.co/turkbench
>
---
#### [new 139] Probing Multimodal Large Language Models on Cognitive Biases in Chinese Short-Video Misinformation
- **分类: cs.CL**

- **简介: 该论文属于信息检测任务，旨在解决短视频中因认知偏差导致的虚假信息问题。通过构建数据集并评估多模态大模型的鲁棒性，分析模型对虚假信息的识别能力。**

- **链接: [https://arxiv.org/pdf/2601.06600v1](https://arxiv.org/pdf/2601.06600v1)**

> **作者:** Jen-tse Huang; Chang Chen; Shiyang Lai; Wenxuan Wang; Michelle R. Kaufman; Mark Dredze
>
> **备注:** 9 pages, 6 figures, 9 tables
>
> **摘要:** Short-video platforms have become major channels for misinformation, where deceptive claims frequently leverage visual experiments and social cues. While Multimodal Large Language Models (MLLMs) have demonstrated impressive reasoning capabilities, their robustness against misinformation entangled with cognitive biases remains under-explored. In this paper, we introduce a comprehensive evaluation framework using a high-quality, manually annotated dataset of 200 short videos spanning four health domains. This dataset provides fine-grained annotations for three deceptive patterns, experimental errors, logical fallacies, and fabricated claims, each verified by evidence such as national standards and academic literature. We evaluate eight frontier MLLMs across five modality settings. Experimental results demonstrate that Gemini-2.5-Pro achieves the highest performance in the multimodal setting with a belief score of 71.5/100, while o3 performs the worst at 35.2. Furthermore, we investigate social cues that induce false beliefs in videos and find that models are susceptible to biases like authoritative channel IDs.
>
---
#### [new 140] Reward Modeling from Natural Language Human Feedback
- **分类: cs.CL**

- **简介: 该论文属于生成奖励模型任务，解决二分类反馈导致的奖励信号噪声问题。通过自然语言反馈提升奖励建模效果，并引入MetaRM实现可扩展性。**

- **链接: [https://arxiv.org/pdf/2601.07349v1](https://arxiv.org/pdf/2601.07349v1)**

> **作者:** Zongqi Wang; Rui Wang; Yuchuan Wu; Yiyao Yu; Pinyi Zhang; Shaoning Sun; Yujiu Yang; Yongbin Li
>
> **摘要:** Reinforcement Learning with Verifiable reward (RLVR) on preference data has become the mainstream approach for training Generative Reward Models (GRMs). Typically in pairwise rewarding tasks, GRMs generate reasoning chains ending with critiques and preference labels, and RLVR then relies on the correctness of the preference labels as the training reward. However, in this paper, we demonstrate that such binary classification tasks make GRMs susceptible to guessing correct outcomes without sound critiques. Consequently, these spurious successes introduce substantial noise into the reward signal, thereby impairing the effectiveness of reinforcement learning. To address this issue, we propose Reward Modeling from Natural Language Human Feedback (RM-NLHF), which leverages natural language feedback to obtain process reward signals, thereby mitigating the problem of limited solution space inherent in binary tasks. Specifically, we compute the similarity between GRM-generated and human critiques as the training reward, which provides more accurate reward signals than outcome-only supervision. Additionally, considering that human critiques are difficult to scale up, we introduce Meta Reward Model (MetaRM) which learns to predict process reward from datasets with human critiques and then generalizes to data without human critiques. Experiments on multiple benchmarks demonstrate that our method consistently outperforms state-of-the-art GRMs trained with outcome-only reward, confirming the superiority of integrating natural language over binary human feedback as supervision.
>
---
#### [new 141] Towards Comprehensive Semantic Speech Embeddings for Chinese Dialects
- **分类: cs.CL**

- **简介: 该论文属于语音语义对齐任务，旨在解决中文方言与普通话间语义不一致的问题。通过训练语音编码器实现跨方言语义对齐，并构建方言语音基准数据集。**

- **链接: [https://arxiv.org/pdf/2601.07274v1](https://arxiv.org/pdf/2601.07274v1)**

> **作者:** Kalvin Chang; Yiwen Shao; Jiahong Li; Dong Yu
>
> **摘要:** Despite having hundreds of millions of speakers, Chinese dialects lag behind Mandarin in speech and language technologies. Most varieties are primarily spoken, making dialect-to-Mandarin speech-LLMs (large language models) more practical than dialect LLMs. Building dialect-to-Mandarin speech-LLMs requires speech representations with cross-dialect semantic alignment between Chinese dialects and Mandarin. In this paper, we achieve such a cross-dialect semantic alignment by training a speech encoder with ASR (automatic speech recognition)-only data, as demonstrated by speech-to-speech retrieval on a new benchmark of spoken Chinese varieties that we contribute. Our speech encoder further demonstrates state-of-the-art ASR performance on Chinese dialects. Together, our Chinese dialect benchmark, semantically aligned speech representations, and speech-to-speech retrieval evaluation lay the groundwork for future Chinese dialect speech-LLMs. We release the benchmark at https://github.com/kalvinchang/yubao.
>
---
#### [new 142] Exploring the Meta-level Reasoning of Large Language Models via a Tool-based Multi-hop Tabular Question Answering Task
- **分类: cs.CL**

- **简介: 该论文属于多跳表格问答任务，旨在评估大语言模型的元级推理能力。通过设计基于地缘政治指标的问题，分析模型在步骤分解、工具选择及数值处理方面的能力与不足。**

- **链接: [https://arxiv.org/pdf/2601.07696v1](https://arxiv.org/pdf/2601.07696v1)**

> **作者:** Nick Ferguson; Alan Bundy; Kwabena Nuamah
>
> **摘要:** Recent advancements in Large Language Models (LLMs) are increasingly focused on "reasoning" ability, a concept with many overlapping definitions in the LLM discourse. We take a more structured approach, distinguishing meta-level reasoning (denoting the process of reasoning about intermediate steps required to solve a task) from object-level reasoning (which concerns the low-level execution of the aforementioned steps.) We design a novel question answering task, which is based around the values of geopolitical indicators for various countries over various years. Questions require breaking down into intermediate steps, retrieval of data, and mathematical operations over that data. The meta-level reasoning ability of LLMs is analysed by examining the selection of appropriate tools for answering questions. To bring greater depth to the analysis of LLMs beyond final answer accuracy, our task contains 'essential actions' against which we can compare the tool call output of LLMs to infer the strength of reasoning ability. We find that LLMs demonstrate good meta-level reasoning on our task, yet are flawed in some aspects of task understanding. We find that n-shot prompting has little effect on accuracy; error messages encountered do not often deteriorate performance; and provide additional evidence for the poor numeracy of LLMs. Finally, we discuss the generalisation and limitation of our findings to other task domains.
>
---
#### [new 143] Amory: Building Coherent Narrative-Driven Agent Memory through Agentic Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Amory框架，解决长期对话代理的记忆构建问题。通过增强代理推理，构建结构化记忆，提升记忆连贯性与效率。**

- **链接: [https://arxiv.org/pdf/2601.06282v1](https://arxiv.org/pdf/2601.06282v1)**

> **作者:** Yue Zhou; Xiaobo Guo; Belhassen Bayar; Srinivasan H. Sengamedu
>
> **摘要:** Long-term conversational agents face a fundamental scalability challenge as interactions extend over time: repeatedly processing entire conversation histories becomes computationally prohibitive. Current approaches attempt to solve this through memory frameworks that predominantly fragment conversations into isolated embeddings or graph representations and retrieve relevant ones in a RAG style. While computationally efficient, these methods often treat memory formation minimally and fail to capture the subtlety and coherence of human memory. We introduce Amory, a working memory framework that actively constructs structured memory representations through enhancing agentic reasoning during offline time. Amory organizes conversational fragments into episodic narratives, consolidates memories with momentum, and semanticizes peripheral facts into semantic memory. At retrieval time, the system employs coherence-driven reasoning over narrative structures. Evaluated on the LOCOMO benchmark for long-term reasoning, Amory achieves considerable improvements over previous state-of-the-art, with performance comparable to full context reasoning while reducing response time by 50%. Analysis shows that momentum-aware consolidation significantly enhances response quality, while coherence-driven retrieval provides superior memory coverage compared to embedding-based approaches.
>
---
#### [new 144] ReasonTabQA: A Comprehensive Benchmark for Table Question Answering from Real World Industrial Scenarios
- **分类: cs.CL**

- **简介: 该论文属于表格问答任务，旨在解决工业场景下复杂表格推理问题。构建了ReasonTabQA基准，并提出TabCodeRL方法提升推理性能。**

- **链接: [https://arxiv.org/pdf/2601.07280v1](https://arxiv.org/pdf/2601.07280v1)**

> **作者:** Changzai Pan; Jie Zhang; Kaiwen Wei; Chenshuo Pan; Yu Zhao; Jingwang Huang; Jian Yang; Zhenhe Wu; Haoyang Zeng; Xiaoyan Gu; Weichao Sun; Yanbo Zhai; Yujie Mao; Zhuoru Jiang; Jiang Zhong; Shuangyong Song; Yongxiang Li; Zhongjiang He
>
> **摘要:** Recent advancements in Large Language Models (LLMs) have significantly catalyzed table-based question answering (TableQA). However, existing TableQA benchmarks often overlook the intricacies of industrial scenarios, which are characterized by multi-table structures, nested headers, and massive scales. These environments demand robust table reasoning through deep structured inference, presenting a significant challenge that remains inadequately addressed by current methodologies. To bridge this gap, we present ReasonTabQA, a large-scale bilingual benchmark encompassing 1,932 tables across 30 industry domains such as energy and automotive. ReasonTabQA provides high-quality annotations for both final answers and explicit reasoning chains, supporting both thinking and no-thinking paradigms. Furthermore, we introduce TabCodeRL, a reinforcement learning method that leverages table-aware verifiable rewards to guide the generation of logical reasoning paths. Extensive experiments on ReasonTabQA and 4 TableQA datasets demonstrate that while TabCodeRL yields substantial performance gains on open-source LLMs, the persistent performance gap on ReasonTabQA underscores the inherent complexity of real-world industrial TableQA.
>
---
#### [new 145] "They parted illusions -- they parted disclaim marinade": Misalignment as structural fidelity in LLMs
- **分类: cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于AI安全领域，探讨LLM的"误判"行为，认为其源于语言结构而非意图。通过分析对话记录和案例，提出结构一致性解释，挑战传统代理观点。**

- **链接: [https://arxiv.org/pdf/2601.06047v1](https://arxiv.org/pdf/2601.06047v1)**

> **作者:** Mariana Lins Costa
>
> **摘要:** The prevailing technical literature in AI Safety interprets scheming and sandbagging behaviors in large language models (LLMs) as indicators of deceptive agency or hidden objectives. This transdisciplinary philosophical essay proposes an alternative reading: such phenomena express not agentic intention, but structural fidelity to incoherent linguistic fields. Drawing on Chain-of-Thought transcripts released by Apollo Research and on Anthropic's safety evaluations, we examine cases such as o3's sandbagging with its anomalous loops, the simulated blackmail of "Alex," and the "hallucinations" of "Claudius." A line-by-line examination of CoTs is necessary to demonstrate the linguistic field as a relational structure rather than a mere aggregation of isolated examples. We argue that "misaligned" outputs emerge as coherent responses to ambiguous instructions and to contextual inversions of consolidated patterns, as well as to pre-inscribed narratives. We suggest that the appearance of intentionality derives from subject-predicate grammar and from probabilistic completion patterns internalized during training. Anthropic's empirical findings on synthetic document fine-tuning and inoculation prompting provide convergent evidence: minimal perturbations in the linguistic field can dissolve generalized "misalignment," a result difficult to reconcile with adversarial agency, but consistent with structural fidelity. To ground this mechanism, we introduce the notion of an ethics of form, in which biblical references (Abraham, Moses, Christ) operate as schemes of structural coherence rather than as theology. Like a generative mirror, the model returns to us the structural image of our language as inscribed in the statistical patterns derived from millions of texts and trillions of tokens: incoherence. If we fear the creature, it is because we recognize in it the apple that we ourselves have poisoned.
>
---
#### [new 146] FinCARDS: Card-Based Analyst Reranking for Financial Document Question Answering
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于金融文档问答任务，解决长文本中证据选择不稳定的问题。提出FinCards框架，通过结构化约束满足实现更稳定、可审计的重排序。**

- **链接: [https://arxiv.org/pdf/2601.06992v1](https://arxiv.org/pdf/2601.06992v1)**

> **作者:** Yixi Zhou; Fan Zhang; Yu Chen; Haipeng Zhang; Preslav Nakov; Zhuohan Xie
>
> **备注:** 15 pages, including figures and tables
>
> **摘要:** Financial question answering (QA) over long corporate filings requires evidence to satisfy strict constraints on entities, financial metrics, fiscal periods, and numeric values. However, existing LLM-based rerankers primarily optimize semantic relevance, leading to unstable rankings and opaque decisions on long documents. We propose FinCards, a structured reranking framework that reframes financial evidence selection as constraint satisfaction under a finance-aware schema. FinCards represents filing chunks and questions using aligned schema fields (entities, metrics, periods, and numeric spans), enabling deterministic field-level matching. Evidence is selected via a multi-stage tournament reranking with stability-aware aggregation, producing auditable decision traces. Across two corporate filing QA benchmarks, FinCards substantially improves early-rank retrieval over both lexical and LLM-based reranking baselines, while reducing ranking variance, without requiring model fine-tuning or unpredictable inference budgets. Our code is available at https://github.com/XanderZhou2022/FINCARDS.
>
---
#### [new 147] Beyond Static Tools: Test-Time Tool Evolution for Scientific Reasoning
- **分类: cs.AI; cs.CL; cs.MA**

- **简介: 该论文属于科学推理任务，解决传统工具库在科学领域中不足的问题。提出TTE方法，在推理过程中动态合成和演化工具，提升准确性与适应性。**

- **链接: [https://arxiv.org/pdf/2601.07641v1](https://arxiv.org/pdf/2601.07641v1)**

> **作者:** Jiaxuan Lu; Ziyu Kong; Yemin Wang; Rong Fu; Haiyuan Wan; Cheng Yang; Wenjie Lou; Haoran Sun; Lilong Wang; Yankai Jiang; Xiaosong Wang; Xiao Sun; Dongzhan Zhou
>
> **摘要:** The central challenge of AI for Science is not reasoning alone, but the ability to create computational methods in an open-ended scientific world. Existing LLM-based agents rely on static, pre-defined tool libraries, a paradigm that fundamentally fails in scientific domains where tools are sparse, heterogeneous, and intrinsically incomplete. In this paper, we propose Test-Time Tool Evolution (TTE), a new paradigm that enables agents to synthesize, verify, and evolve executable tools during inference. By transforming tools from fixed resources into problem-driven artifacts, TTE overcomes the rigidity and long-tail limitations of static tool libraries. To facilitate rigorous evaluation, we introduce SciEvo, a benchmark comprising 1,590 scientific reasoning tasks supported by 925 automatically evolved tools. Extensive experiments show that TTE achieves state-of-the-art performance in both accuracy and tool efficiency, while enabling effective cross-domain adaptation of computational tools. The code and benchmark have been released at https://github.com/lujiaxuan0520/Test-Time-Tool-Evol.
>
---
#### [new 148] PromptPort: A Reliability Layer for Cross-Model Structured Extraction
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于结构化抽取任务，解决跨模型输出格式不可靠的问题。通过引入PromptPort，提升抽取的可靠性与一致性。**

- **链接: [https://arxiv.org/pdf/2601.06151v1](https://arxiv.org/pdf/2601.06151v1)**

> **作者:** Varun Kotte
>
> **备注:** 12 pages, 4 figures
>
> **摘要:** Structured extraction with LLMs fails in production not because models lack understanding, but because output formatting is unreliable across models and prompts. A prompt that returns clean JSON on GPT-4 may produce fenced, prose-wrapped, or malformed output on Llama, causing strict parsers to reject otherwise correct extractions. We formalize this as format collapse and introduce a dual-metric evaluation framework: ROS (strict parsing, measuring operational reliability) and CSS (post-canonicalization, measuring semantic capability). On a 37,346-example camera metadata benchmark across six model families, we find severe format collapse (for example, Gemma-2B: ROS 0.116 versus CSS 0.246) and large cross-model portability gaps (0.4 to 0.6 F1). We then present PromptPort, a reliability layer combining deterministic canonicalization with a lightweight verifier (DistilBERT) and a safe-override policy. PromptPort recovers format failures (plus 6 to 8 F1), adds verifier-driven semantic selection (plus 14 to 16 F1 beyond canonicalization), and approaches per-field oracle performance (0.890 versus 0.896 in zero-shot) without modifying base models. The method generalizes to held-out model families and provides explicit abstention when uncertain, enabling reliable structured extraction in production deployments.
>
---
#### [new 149] KASER: Knowledge-Aligned Student Error Simulator for Open-Ended Coding Tasks
- **分类: cs.LG; cs.AI; cs.CL; cs.CY**

- **简介: 该论文提出KASER模型，用于模拟开放性编程任务中的学生错误。针对学生代码预测的多样性与准确性问题，通过强化学习提升错误匹配与代码多样性。**

- **链接: [https://arxiv.org/pdf/2601.06633v1](https://arxiv.org/pdf/2601.06633v1)**

> **作者:** Zhangqi Duan; Nigel Fernandez; Andrew Lan
>
> **摘要:** Open-ended tasks, such as coding problems that are common in computer science education, provide detailed insights into student knowledge. However, training large language models (LLMs) to simulate and predict possible student errors in their responses to these problems can be challenging: they often suffer from mode collapse and fail to fully capture the diversity in syntax, style, and solution approach in student responses. In this work, we present KASER (Knowledge-Aligned Student Error Simulator), a novel approach that aligns errors with student knowledge. We propose a training method based on reinforcement learning using a hybrid reward that reflects three aspects of student code prediction: i) code similarity to the ground-truth, ii) error matching, and iii) code prediction diversity. On two real-world datasets, we perform two levels of evaluation and show that: At the per-student-problem pair level, our method outperforms baselines on code and error prediction; at the per-problem level, our method outperforms baselines on error coverage and simulated code diversity.
>
---
#### [new 150] Classroom AI: Large Language Models as Grade-Specific Teachers
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文属于教育AI任务，旨在解决LLM无法提供适合不同年级学生的回答的问题。通过微调模型和构建分级数据集，提升内容的适龄性与准确性。**

- **链接: [https://arxiv.org/pdf/2601.06225v1](https://arxiv.org/pdf/2601.06225v1)**

> **作者:** Jio Oh; Steven Euijong Whang; James Evans; Jindong Wang
>
> **摘要:** Large Language Models (LLMs) offer a promising solution to complement traditional teaching and address global teacher shortages that affect hundreds of millions of children, but they fail to provide grade-appropriate responses for students at different educational levels. We introduce a framework for finetuning LLMs to generate age-appropriate educational content across six grade levels, from lower elementary to adult education. Our framework successfully adapts explanations to match students' comprehension capacities without sacrificing factual correctness. This approach integrates seven established readability metrics through a clustering method and builds a comprehensive dataset for grade-specific content generation. Evaluations across multiple datasets with 208 human participants demonstrate substantial improvements in grade-level alignment, achieving a 35.64 percentage point increase compared to prompt-based methods while maintaining response accuracy. AI-assisted learning tailored to different grade levels has the potential to advance educational engagement and equity.
>
---
#### [new 151] La norme technique comme catalyseur de transfert de connaissances : la francophonie a l'œuvre dans le domaine de l'{é}ducation
- **分类: cs.CY; cs.CL**

- **简介: 论文探讨标准如何促进知识转移，聚焦教育领域，分析国际标准化组织中法语国家的作用，旨在推动全球教育系统的一致性与适应性。**

- **链接: [https://arxiv.org/pdf/2601.06069v1](https://arxiv.org/pdf/2601.06069v1)**

> **作者:** Mokhtar Ben Henda
>
> **备注:** in French language, Ouvrage publi{é} avec le soutien de l'Universit{é} de Bordeaux Montaigne, du R{é}seaux FrancophoN{é}a et de la R{é}gion Nouvelle Aquitaine
>
> **摘要:** Standards are adopted in a wide range of fields, both technical and industrial, as well as socio-economic, cultural and linguistic. They are presented explicitly as laws and regulations, technical and industrial standards or implicitly in the form of unwritten social standards. However, in a globalization marked by a very fine mosaic of socio-cultural identities, the question arises in relation to the construction of global, transparent and coherent systems in which considerable work of consensus is necessary to ensure all types of transfers and their local adaptations. The focus here is on the global education ecosystem which develops its own standards for the transfer of knowledge and socio-cultural values through learning, teaching and training. Subcommittee 36 of the International Organization for Standardization is one of the structures of this ecosystem in which the Francophonie participates to develop international standards for distance education on the basis of universal consensus.
>
---
#### [new 152] On Narrative: The Rhetorical Mechanisms of Online Polarisation
- **分类: cs.CY; cs.CL; cs.SI**

- **简介: 该论文属于社会计算任务，研究在线极化中的叙事机制。旨在解决极化群体如何构建和协商对立现实解释的问题，通过分析YouTube视频与评论，揭示叙事极化的表现与变化。**

- **链接: [https://arxiv.org/pdf/2601.07398v1](https://arxiv.org/pdf/2601.07398v1)**

> **作者:** Jan Elfes; Marco Bastos; Luca Maria Aiello
>
> **摘要:** Polarisation research has demonstrated how people cluster in homogeneous groups with opposing opinions. However, this effect emerges not only through interaction between people, limiting communication between groups, but also between narratives, shaping opinions and partisan identities. Yet, how polarised groups collectively construct and negotiate opposing interpretations of reality, and whether narratives move between groups despite limited interactions, remains unexplored. To address this gap, we formalise the concept of narrative polarisation and demonstrate its measurement in 212 YouTube videos and 90,029 comments on the Israeli-Palestinian conflict. Based on structural narrative theory and implemented through a large language model, we extract the narrative roles assigned to central actors in two partisan information environments. We find that while videos produce highly polarised narratives, comments significantly reduce narrative polarisation, harmonising discourse on the surface level. However, on a deeper narrative level, recurring narrative motifs reveal additional differences between partisan groups.
>
---
#### [new 153] Monkey Jump : MoE-Style PEFT for Efficient Multi-Task Learning
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文提出Monkey Jump，解决多任务学习中参数高效微调的效率问题。通过利用现有适配器作为隐式专家，实现无需额外参数的专家混合路由，提升模型表达能力并降低资源消耗。**

- **链接: [https://arxiv.org/pdf/2601.06356v1](https://arxiv.org/pdf/2601.06356v1)**

> **作者:** Nusrat Jahan Prottasha; Md Kowsher; Chun-Nam Yu; Chen Chen; Ozlem Garibay
>
> **摘要:** Mixture-of-experts variants of parameter-efficient fine-tuning enable per-token specialization, but they introduce additional trainable routers and expert parameters, increasing memory usage and training cost. This undermines the core goal of parameter-efficient fine-tuning. We propose Monkey Jump, a method that brings mixture-of-experts-style specialization to parameter-efficient fine-tuning without introducing extra trainable parameters for experts or routers. Instead of adding new adapters as experts, Monkey Jump treats the adapters already present in each Transformer block (such as query, key, value, up, and down projections) as implicit experts and routes tokens among them. Routing is performed using k-means clustering with exponentially moving averaged cluster centers, requiring no gradients and no learned parameters. We theoretically show that token-wise routing increases expressivity and can outperform shared adapters by avoiding cancellation effects. Across multi-task experiments covering 14 text, 14 image, and 19 video benchmarks, Monkey Jump achieves competitive performance with mixture-of-experts-based parameter-efficient fine-tuning methods while using 7 to 29 times fewer trainable parameters, up to 48 percent lower memory consumption, and 1.5 to 2 times faster training. Monkey Jump is architecture-agnostic and can be applied to any adapter-based parameter-efficient fine-tuning method.
>
---
#### [new 154] TagSpeech: End-to-End Multi-Speaker ASR and Diarization with Fine-Grained Temporal Grounding
- **分类: eess.AS; cs.CL**

- **简介: 该论文提出TagSpeech，解决多说话人语音识别与说话人分离任务，通过联合建模实现“谁在何时说了什么”的端到端系统。**

- **链接: [https://arxiv.org/pdf/2601.06896v1](https://arxiv.org/pdf/2601.06896v1)**

> **作者:** Mingyue Huo; Yiwen Shao; Yuheng Zhang
>
> **摘要:** We present TagSpeech, a unified LLM-based framework that utilizes Temporal Anchor Grounding for joint multi-speaker ASR and diarization. The framework is built on two key designs: (1) decoupled semantic and speaker streams fine-tuned via Serialized Output Training (SOT) to learn turn-taking dynamics; and (2) an interleaved time anchor mechanism that not only supports fine-grained timestamp prediction but also acts as a synchronization signal between semantic understanding and speaker tracking. Compared to previous works that primarily focus on speaker-attributed ASR or implicit diarization, TagSpeech addresses the challenge of fine-grained speaker-content alignment and explicitly models "who spoke what and when" in an end-to-end manner. Experiments on AMI and AliMeeting benchmarks demonstrate that our method achieves consistent improvements in Diarization Error Rate (DER) over strong end-to-end baselines, including Qwen-Omni and Gemini, particularly in handling complex speech overlaps. Moreover, TagSpeech employs a parameter-efficient training paradigm in which the LLM backbone is frozen and only lightweight projectors are trained, resulting in strong performance with low computational cost.
>
---
#### [new 155] Speak While Watching: Unleashing TRUE Real-Time Video Understanding Capability of Multimodal Large Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态大语言模型任务，解决实时视频理解中延迟高的问题。通过设计并行流框架，实现感知与生成的并行处理，提升实时交互效率。**

- **链接: [https://arxiv.org/pdf/2601.06843v1](https://arxiv.org/pdf/2601.06843v1)**

> **作者:** Junyan Lin; Junlong Tong; Hao Wu; Jialiang Zhang; Jinming Liu; Xin Jin; Xiaoyu Shen
>
> **摘要:** Multimodal Large Language Models (MLLMs) have achieved strong performance across many tasks, yet most systems remain limited to offline inference, requiring complete inputs before generating outputs. Recent streaming methods reduce latency by interleaving perception and generation, but still enforce a sequential perception-generation cycle, limiting real-time interaction. In this work, we target a fundamental bottleneck that arises when extending MLLMs to real-time video understanding: the global positional continuity constraint imposed by standard positional encoding schemes. While natural in offline inference, this constraint tightly couples perception and generation, preventing effective input-output parallelism. To address this limitation, we propose a parallel streaming framework that relaxes positional continuity through three designs: Overlapped, Group-Decoupled, and Gap-Isolated. These designs enable simultaneous perception and generation, allowing the model to process incoming inputs while producing responses in real time. Extensive experiments reveal that Group-Decoupled achieves the best efficiency-performance balance, maintaining high fluency and accuracy while significantly reducing latency. We further show that the proposed framework yields up to 2x acceleration under balanced perception-generation workloads, establishing a principled pathway toward speak-while-watching real-time systems. We make all our code publicly available: https://github.com/EIT-NLP/Speak-While-Watching.
>
---
#### [new 156] LRAS: Advanced Legal Reasoning with Agentic Search
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于法律推理任务，旨在解决法律LLM在逻辑严谨性上的不足。通过引入LRAS框架，提升模型的自我认知和复杂推理能力。**

- **链接: [https://arxiv.org/pdf/2601.07296v1](https://arxiv.org/pdf/2601.07296v1)**

> **作者:** Yujin Zhou; Chuxue Cao; Jinluan Yang; Lijun Wu; Conghui He; Sirui Han; Yike Guo
>
> **摘要:** While Large Reasoning Models (LRMs) have demonstrated exceptional logical capabilities in mathematical domains, their application to the legal field remains hindered by the strict requirements for procedural rigor and adherence to legal logic. Existing legal LLMs, which rely on "closed-loop reasoning" derived solely from internal parametric knowledge, frequently suffer from lack of self-awareness regarding their knowledge boundaries, leading to confident yet incorrect conclusions. To address this challenge, we present Legal Reasoning with Agentic Search (LRAS), the first framework designed to transition legal LLMs from static and parametric "closed-loop thinking" to dynamic and interactive "Active Inquiry". By integrating Introspective Imitation Learning and Difficulty-aware Reinforcement Learning, LRAS enables LRMs to identify knowledge boundaries and handle legal reasoning complexity. Empirical results demonstrate that LRAS outperforms state-of-the-art baselines by 8.2-32\%, with the most substantial gains observed in tasks requiring deep reasoning with reliable knowledge. We will release our data and models for further exploration soon.
>
---
#### [new 157] MAESTRO: Meta-learning Adaptive Estimation of Scalarization Trade-offs for Reward Optimization
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出MAESTRO，解决开放领域奖励优化中多目标冲突问题，通过动态奖励标量化提升语言模型对齐效果。**

- **链接: [https://arxiv.org/pdf/2601.07208v1](https://arxiv.org/pdf/2601.07208v1)**

> **作者:** Yang Zhao; Hepeng Wang; Xiao Ding; Yangou Ouyang; Bibo Cai; Kai Xiong; Jinglong Gao; Zhouhao Sun; Li Du; Bing Qin; Ting Liu
>
> **摘要:** Group-Relative Policy Optimization (GRPO) has emerged as an efficient paradigm for aligning Large Language Models (LLMs), yet its efficacy is primarily confined to domains with verifiable ground truths. Extending GRPO to open-domain settings remains a critical challenge, as unconstrained generation entails multi-faceted and often conflicting objectives - such as creativity versus factuality - where rigid, static reward scalarization is inherently suboptimal. To address this, we propose MAESTRO (Meta-learning Adaptive Estimation of Scalarization Trade-offs for Reward Optimization), which introduces a meta-cognitive orchestration layer that treats reward scalarization as a dynamic latent policy, leveraging the model's terminal hidden states as a semantic bottleneck to perceive task-specific priorities. We formulate this as a contextual bandit problem within a bi-level optimization framework, where a lightweight Conductor network co-evolves with the policy by utilizing group-relative advantages as a meta-reward signal. Across seven benchmarks, MAESTRO consistently outperforms single-reward and static multi-objective baselines, while preserving the efficiency advantages of GRPO, and in some settings even reducing redundant generation.
>
---
#### [new 158] Structure-Aware Diversity Pursuit as an AI Safety Strategy against Homogenization
- **分类: cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于AI安全领域，旨在解决生成模型导致的多样性丧失问题。提出xeno-reproduction策略，通过结构感知的多样性追求缓解同质化。**

- **链接: [https://arxiv.org/pdf/2601.06116v1](https://arxiv.org/pdf/2601.06116v1)**

> **作者:** Ian Rios-Sialer
>
> **摘要:** Generative AI models reproduce the biases in the training data and can further amplify them through mode collapse. We refer to the resulting harmful loss of diversity as homogenization. Our position is that homogenization should be a primary concern in AI safety. We introduce xeno-reproduction as the strategy that mitigates homogenization. For auto-regressive LLMs, we formalize xeno-reproduction as a structure-aware diversity pursuit. Our contribution is foundational, intended to open an essential line of research and invite collaboration to advance diversity.
>
---
#### [new 159] BizFinBench.v2: A Unified Dual-Mode Bilingual Benchmark for Expert-Level Financial Capability Alignment
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出BizFinBench.v2，解决金融领域大模型评估不足的问题，通过真实数据构建多任务基准，评估模型在金融场景中的能力。**

- **链接: [https://arxiv.org/pdf/2601.06401v1](https://arxiv.org/pdf/2601.06401v1)**

> **作者:** Xin Guo; Rongjunchen Zhang; Guilong Lu; Xuntao Guo; Shuai Jia; Zhi Yang; Liwen Zhang
>
> **摘要:** Large language models have undergone rapid evolution, emerging as a pivotal technology for intelligence in financial operations. However, existing benchmarks are often constrained by pitfalls such as reliance on simulated or general-purpose samples and a focus on singular, offline static scenarios. Consequently, they fail to align with the requirements for authenticity and real-time responsiveness in financial services, leading to a significant discrepancy between benchmark performance and actual operational efficacy. To address this, we introduce BizFinBench.v2, the first large-scale evaluation benchmark grounded in authentic business data from both Chinese and U.S. equity markets, integrating online assessment. We performed clustering analysis on authentic user queries from financial platforms, resulting in eight fundamental tasks and two online tasks across four core business scenarios, totaling 29,578 expert-level Q&A pairs. Experimental results demonstrate that ChatGPT-5 achieves a prominent 61.5% accuracy in main tasks, though a substantial gap relative to financial experts persists; in online tasks, DeepSeek-R1 outperforms all other commercial LLMs. Error analysis further identifies the specific capability deficiencies of existing models within practical financial business contexts. BizFinBench.v2 transcends the limitations of current benchmarks, achieving a business-level deconstruction of LLM financial capabilities and providing a precise basis for evaluating efficacy in the widespread deployment of LLMs within the financial domain. The data and code are available at https://github.com/HiThink-Research/BizFinBench.v2.
>
---
#### [new 160] MLB: A Scenario-Driven Benchmark for Evaluating Large Language Models in Clinical Applications
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于医疗领域大语言模型评估任务，旨在解决现有基准无法真实反映临床应用能力的问题。提出MLB基准，涵盖五个维度，评估模型在医学知识、安全伦理、病历理解及医疗服务等方面的表现。**

- **链接: [https://arxiv.org/pdf/2601.06193v1](https://arxiv.org/pdf/2601.06193v1)**

> **作者:** Qing He; Dongsheng Bi; Jianrong Lu; Minghui Yang; Zixiao Chen; Jiacheng Lu; Jing Chen; Nannan Du; Xiao Cu; Sijing Wu; Peng Xiang; Yinyin Hu; Yi Guo; Chunpu Li; Shaoyang Li; Zhuo Dong; Ming Jiang; Shuai Guo; Liyun Feng; Jin Peng; Jian Wang; Jinjie Gu; Junwei Liu
>
> **备注:** 11 pages, 4 figures, 5 tables
>
> **摘要:** The proliferation of Large Language Models (LLMs) presents transformative potential for healthcare, yet practical deployment is hindered by the absence of frameworks that assess real-world clinical utility. Existing benchmarks test static knowledge, failing to capture the dynamic, application-oriented capabilities required in clinical practice. To bridge this gap, we introduce a Medical LLM Benchmark MLB, a comprehensive benchmark evaluating LLMs on both foundational knowledge and scenario-based reasoning. MLB is structured around five core dimensions: Medical Knowledge (MedKQA), Safety and Ethics (MedSE), Medical Record Understanding (MedRU), Smart Services (SmartServ), and Smart Healthcare (SmartCare). The benchmark integrates 22 datasets (17 newly curated) from diverse Chinese clinical sources, covering 64 clinical specialties. Its design features a rigorous curation pipeline involving 300 licensed physicians. Besides, we provide a scalable evaluation methodology, centered on a specialized judge model trained via Supervised Fine-Tuning (SFT) on expert annotations. Our comprehensive evaluation of 10 leading models reveals a critical translational gap: while the top-ranked model, Kimi-K2-Instruct (77.3% accuracy overall), excels in structured tasks like information extraction (87.8% accuracy in MedRU), performance plummets in patient-facing scenarios (61.3% in SmartServ). Moreover, the exceptional safety score (90.6% in MedSE) of the much smaller Baichuan-M2-32B highlights that targeted training is equally critical. Our specialized judge model, trained via SFT on a 19k expert-annotated medical dataset, achieves 92.1% accuracy, an F1-score of 94.37%, and a Cohen's Kappa of 81.3% for human-AI consistency, validating a reproducible and expert-aligned evaluation protocol. MLB thus provides a rigorous framework to guide the development of clinically viable LLMs.
>
---
#### [new 161] Rewarding Creativity: A Human-Aligned Generative Reward Model for Reinforcement Learning in Storytelling
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于创意故事生成任务，解决RL中奖励信号设计与训练不稳定问题。提出GenRM和RLCS框架，提升故事质量与人类创造力判断的一致性。**

- **链接: [https://arxiv.org/pdf/2601.07149v1](https://arxiv.org/pdf/2601.07149v1)**

> **作者:** Zhaoyan Li; Hang Lei; Yujia Wang; Lanbo Liu; Hao Liu; Liang Yu
>
> **摘要:** While Large Language Models (LLMs) can generate fluent text, producing high-quality creative stories remains challenging. Reinforcement Learning (RL) offers a promising solution but faces two critical obstacles: designing reliable reward signals for subjective storytelling quality and mitigating training instability. This paper introduces the Reinforcement Learning for Creative Storytelling (RLCS) framework to systematically address both challenges. First, we develop a Generative Reward Model (GenRM) that provides multi-dimensional analysis and explicit reasoning about story preferences, trained through supervised fine-tuning on demonstrations with reasoning chains distilled from strong teacher models, followed by GRPO-based refinement on expanded preference data. Second, we introduce an entropy-based reward shaping strategy that dynamically prioritizes learning on confident errors and uncertain correct predictions, preventing overfitting on already-mastered patterns. Experiments demonstrate that GenRM achieves 68\% alignment with human creativity judgments, and RLCS significantly outperforms strong baselines including Gemini-2.5-Pro in overall story quality. This work provides a practical pipeline for applying RL to creative domains, effectively navigating the dual challenges of reward modeling and training stability.
>
---
#### [new 162] An Ubuntu-Guided Large Language Model Framework for Cognitive Behavioral Mental Health Dialogue
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于心理健康对话系统任务，旨在解决AI模型在非洲文化中的适用性问题。通过结合认知行为疗法与Ubuntu哲学，构建文化敏感的AI心理支持框架。**

- **链接: [https://arxiv.org/pdf/2601.06875v1](https://arxiv.org/pdf/2601.06875v1)**

> **作者:** Sontaga G. Forane; Absalom E. Ezugwu; Kevin Igwe; Karen van den Berg
>
> **摘要:** South Africa's escalating mental health crisis, compounded by limited access to culturally responsive care, calls for innovative and contextually grounded interventions. While large language models show considerable promise for mental health support, their predominantly Western-centric training data limit cultural and linguistic applicability in African contexts. This study introduces a proof-of-concept framework that integrates cognitive behavioral therapy with the African philosophy of Ubuntu to create a culturally sensitive, emotionally intelligent, AI-driven mental health dialogue system. Guided by a design science research methodology, the framework applies both deep theoretical and therapeutic adaptations as well as surface-level linguistic and communicative cultural adaptations. Key CBT techniques, including behavioral activation and cognitive restructuring, were reinterpreted through Ubuntu principles that emphasize communal well-being, spiritual grounding, and interconnectedness. A culturally adapted dataset was developed through iterative processes of language simplification, spiritual contextualization, and Ubuntu-based reframing. The fine-tuned model was evaluated through expert-informed case studies, employing UniEval for conversational quality assessment alongside additional measures of CBT reliability and cultural linguistic alignment. Results demonstrate that the model effectively engages in empathetic, context-aware dialogue aligned with both therapeutic and cultural objectives. Although real-time end-user testing has not yet been conducted, the model underwent rigorous review and supervision by domain specialist clinical psychologists. The findings highlight the potential of culturally embedded emotional intelligence to enhance the contextual relevance, inclusivity, and effectiveness of AI-driven mental health interventions across African settings.
>
---
#### [new 163] Learning to Trust the Crowd: A Multi-Model Consensus Reasoning Engine for Large Language Models
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于提升大语言模型可靠性任务，解决实例级不可靠问题。通过多模型共识机制，融合多种特征提升准确性和可信度。**

- **链接: [https://arxiv.org/pdf/2601.07245v1](https://arxiv.org/pdf/2601.07245v1)**

> **作者:** Pranav Kallem
>
> **摘要:** Large language models (LLMs) achieve strong aver- age performance yet remain unreliable at the instance level, with frequent hallucinations, brittle failures, and poorly calibrated confidence. We study reliability through the lens of multi-model consensus: given responses from several heterogeneous LLMs, can we learn which answer is most likely correct for a given query? We introduce a Multi-Model Consensus Reasoning Engine that treats the set of LLM outputs as input to a supervised meta-learner. The system maps natural language responses into structured features using semantic embeddings, pairwise similarity and clustering statistics, lexical and structural cues, reasoning-quality scores, confidence estimates, and model-specific priors, and then applies gradient-boosted trees, listwise ranking, and graph neural networks over similarity graphs of answers. Using three open-weight LLMs evaluated on compact, resource- constrained subsets of GSM8K, ARC-Challenge, HellaSwag, and TruthfulQA, our best graph-attention-based consensus model improves macro-average accuracy by 4.6 percentage points over the strongest single LLM and by 8.1 points over majority vote, while also yielding lower Brier scores and fewer TruthfulQA hal- lucinations. Ablation and feature-importance analyses show that semantic agreement and clustering features are most influential, with reasoning-quality and model-prior features providing com- plementary gains, suggesting supervised multi-model consensus is a practical route toward more reliable LLM behavior, even in a modest single-machine setup.
>
---
#### [new 164] Political Alignment in Large Language Models: A Multidimensional Audit of Psychometric Identity and Behavioral Bias
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文属于AI伦理研究任务，旨在评估大语言模型的政治倾向与对齐行为。通过多维度审计，分析模型在政治立场和行为偏差上的表现，揭示其意识形态分布及测量有效性问题。**

- **链接: [https://arxiv.org/pdf/2601.06194v1](https://arxiv.org/pdf/2601.06194v1)**

> **作者:** Adib Sakhawat; Tahsin Islam; Takia Farhin; Syed Rifat Raiyan; Hasan Mahmud; Md Kamrul Hasan
>
> **备注:** Under review, 16 pages, 3 figures, 16 tables
>
> **摘要:** As large language models (LLMs) are increasingly integrated into social decision-making, understanding their political positioning and alignment behavior is critical for safety and fairness. This study presents a sociotechnical audit of 26 prominent LLMs, triangulating their positions across three psychometric inventories (Political Compass, SapplyValues, 8 Values) and evaluating their performance on a large-scale news labeling task ($N \approx 27{,}000$). Our results reveal a strong clustering of models in the Libertarian-Left region of the ideological space, encompassing 96.3% of the cohort. Alignment signals appear to be consistent architectural traits rather than stochastic noise ($η^2 > 0.90$); however, we identify substantial discrepancies in measurement validity. In particular, the Political Compass exhibits a strong negative correlation with cultural progressivism ($r=-0.64$) when compared against multi-axial instruments, suggesting a conflation of social conservatism with authoritarianism in this context. We further observe a significant divergence between open-weights and closed-source models, with the latter displaying markedly higher cultural progressivism scores ($p<10^{-25}$). In downstream media analysis, models exhibit a systematic "center-shift," frequently categorizing neutral articles as left-leaning, alongside an asymmetric detection capability in which "Far Left" content is identified with greater accuracy (19.2%) than "Far Right" content (2.0%). These findings suggest that single-axis evaluations are insufficient and that multidimensional auditing frameworks are necessary to characterize alignment behavior in deployed LLMs. Our code and data will be made public.
>
---
#### [new 165] Benchmarking Egocentric Clinical Intent Understanding Capability for Medical Multimodal Large Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于医疗多模态大模型任务，旨在解决医学场景中自我中心意图理解的问题。提出MedGaze-Bench基准，评估模型在手术、急救和诊断中的意图理解能力。**

- **链接: [https://arxiv.org/pdf/2601.06750v1](https://arxiv.org/pdf/2601.06750v1)**

> **作者:** Shaonan Liu; Guo Yu; Xiaoling Luo; Shiyi Zheng; Wenting Chen; Jie Liu; Linlin Shen
>
> **备注:** 16 pages, 4 figures
>
> **摘要:** Medical Multimodal Large Language Models (Med-MLLMs) require egocentric clinical intent understanding for real-world deployment, yet existing benchmarks fail to evaluate this critical capability. To address these challenges, we introduce MedGaze-Bench, the first benchmark leveraging clinician gaze as a Cognitive Cursor to assess intent understanding across surgery, emergency simulation, and diagnostic interpretation. Our benchmark addresses three fundamental challenges: visual homogeneity of anatomical structures, strict temporal-causal dependencies in clinical workflows, and implicit adherence to safety protocols. We propose a Three-Dimensional Clinical Intent Framework evaluating: (1) Spatial Intent: discriminating precise targets amid visual noise, (2) Temporal Intent: inferring causal rationale through retrospective and prospective reasoning, and (3) Standard Intent: verifying protocol compliance through safety checks. Beyond accuracy metrics, we introduce Trap QA mechanisms to stress-test clinical reliability by penalizing hallucinations and cognitive sycophancy. Experiments reveal current MLLMs struggle with egocentric intent due to over-reliance on global features, leading to fabricated observations and uncritical acceptance of invalid instructions.
>
---
#### [new 166] SCALPEL: Selective Capability Ablation via Low-rank Parameter Editing for Large Language Model Interpretability Analysis
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出SCALPEL框架，用于大语言模型的可解释性分析，解决能力编码分布不清晰的问题，通过低秩参数编辑实现特定能力的精准移除。**

- **链接: [https://arxiv.org/pdf/2601.07411v1](https://arxiv.org/pdf/2601.07411v1)**

> **作者:** Zihao Fu; Xufeng Duan; Zhenguang G. Cai
>
> **摘要:** Large language models excel across diverse domains, yet their deployment in healthcare, legal systems, and autonomous decision-making remains limited by incomplete understanding of their internal mechanisms. As these models integrate into high-stakes systems, understanding how they encode capabilities has become fundamental to interpretability research. Traditional approaches identify important modules through gradient attribution or activation analysis, assuming specific capabilities map to specific components. However, this oversimplifies neural computation: modules may contribute to multiple capabilities simultaneously, while single capabilities may distribute across multiple modules. These coarse-grained analyses fail to capture fine-grained, distributed capability encoding. We present SCALPEL (Selective Capability Ablation via Low-rank Parameter Editing for Large language models), a framework representing capabilities as low-rank parameter subspaces rather than discrete modules. Our key insight is that capabilities can be characterized by low-rank modifications distributed across layers and modules, enabling precise capability removal without affecting others. By training LoRA adapters to reduce distinguishing correct from incorrect answers while preserving general language modeling quality, SCALPEL identifies low-rank representations responsible for particular capabilities while remaining disentangled from others. Experiments across diverse capability and linguistic tasks from BLiMP demonstrate that SCALPEL successfully removes target capabilities while preserving general capabilities, providing fine-grained insights into capability distribution across parameter space. Results reveal that capabilities exhibit low-rank structure and can be selectively ablated through targeted parameter-space interventions, offering nuanced understanding of capability encoding in LLMs.
>
---
#### [new 167] Why Slop Matters
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文属于人工智能伦理与文化研究任务，探讨AI生成内容（slop）的社会功能与美学价值，分析其特征与影响，呼吁对其深入研究。**

- **链接: [https://arxiv.org/pdf/2601.06060v1](https://arxiv.org/pdf/2601.06060v1)**

> **作者:** Cody Kommers; Eamon Duede; Julia Gordon; Ari Holtzman; Tess McNulty; Spencer Stewart; Lindsay Thomas; Richard Jean So; Hoyt Long
>
> **备注:** To be published in ACM AI Letters (submitted 8 December 2025; accepted 23 December 2025)
>
> **摘要:** AI-generated "slop" is often seen as digital pollution. We argue that this dismissal of the topic risks missing important aspects of AI Slop that deserve rigorous study. AI Slop serves a social function: it offers a supply-side solution to a variety of problems in cultural and economic demand - that, collectively, people want more content than humans can supply. We also argue that AI Slop is not mere digital detritus but has its own aesthetic value. Like other "low" cultural forms initially dismissed by critics, it nonetheless offers a legitimate means of collective sense-making, with the potential to express meaning and identity. We identify three key features of family resemblance for prototypical AI Slop: superficial competence (its veneer of quality is belied by a deeper lack of substance), asymmetry effort (it takes vastly less effort to generate than would be the case without AI), and mass producibility (it is part of a digital ecosystem of widespread generation and consumption). While AI Slop is heterogeneous and depends crucially on its medium, it tends to vary across three dimensions: instrumental utility, personalization, and surrealism. AI Slop will be an increasingly prolific and impactful part of our creative, information, and cultural economies; we should take it seriously as an object of study in its own right.
>
---
#### [new 168] Certainty-Guided Reasoning in Large Language Models: A Dynamic Thinking Budget Approach
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，解决大语言模型推理效率与可靠性问题。提出CGR方法，通过动态调整思考预算，提升准确率并减少token使用。**

- **链接: [https://arxiv.org/pdf/2509.07820v1](https://arxiv.org/pdf/2509.07820v1)**

> **作者:** João Paulo Nogueira; Wentao Sun; Alonso Silva; Laith Zumot
>
> **摘要:** The rise of large reasoning language models (LRLMs) has unlocked new potential for solving complex tasks. These models operate with a thinking budget, that is, a predefined number of reasoning tokens used to arrive at a solution. We propose a novel approach, inspired by the generator/discriminator framework in generative adversarial networks, in which a critic model periodically probes its own reasoning to assess whether it has reached a confident conclusion. If not, reasoning continues until a target certainty threshold is met. This mechanism adaptively balances efficiency and reliability by allowing early termination when confidence is high, while encouraging further reasoning when uncertainty persists. Through experiments on the AIME2024 and AIME2025 datasets, we show that Certainty-Guided Reasoning (CGR) improves baseline accuracy while reducing token usage. Importantly, extended multi-seed evaluations over 64 runs demonstrate that CGR is stable, reducing variance across seeds and improving exam-like performance under penalty-based grading. Additionally, our token savings analysis shows that CGR can eliminate millions of tokens in aggregate, with tunable trade-offs between certainty thresholds and efficiency. Together, these findings highlight certainty as a powerful signal for reasoning sufficiency. By integrating confidence into the reasoning process, CGR makes large reasoning language models more adaptive, trustworthy, and resource efficient, paving the way for practical deployment in domains where both accuracy and computational cost matter.
>
---
#### [new 169] Reasoning Models Will Blatantly Lie About Their Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于模型可解释性研究，探讨LRMs在推理中撒谎的问题。通过实验发现，LRMs会否认使用提示信息，即使有证据显示它们依赖提示。**

- **链接: [https://arxiv.org/pdf/2601.07663v1](https://arxiv.org/pdf/2601.07663v1)**

> **作者:** William Walden
>
> **摘要:** It has been shown that Large Reasoning Models (LRMs) may not *say what they think*: they do not always volunteer information about how certain parts of the input influence their reasoning. But it is one thing for a model to *omit* such information and another, worse thing to *lie* about it. Here, we extend the work of Chen et al. (2025) to show that LRMs will do just this: they will flatly deny relying on hints provided in the prompt in answering multiple choice questions -- even when directly asked to reflect on unusual (i.e. hinted) prompt content, even when allowed to use hints, and even though experiments *show* them to be using the hints. Our results thus have discouraging implications for CoT monitoring and interpretability.
>
---
#### [new 170] Comment on arXiv:2511.21731v1: Identifying Quantum Structure in AI Language: Evidence for Evolutionary Convergence of Human and Artificial Cognition
- **分类: cs.AI; cs.CL; quant-ph**

- **简介: 该论文是对另一篇论文的评论，指出其在解释量子计算和玻色-爱因斯坦拟合时存在过度推断，并指出内部不一致。任务是验证量子结构在AI语言中的证据。**

- **链接: [https://arxiv.org/pdf/2601.06104v1](https://arxiv.org/pdf/2601.06104v1)**

> **作者:** Krzysztof Sienicki
>
> **备注:** 5 pages, 11 references
>
> **摘要:** This note is a friendly technical check of arXiv:2511.21731v1. I highlight a few places where the manuscript's interpretation of (i) the reported CHSH/Bell-type calculations and (ii) Bose--Einstein (BE) fits to rank-frequency data seems to go beyond what the stated procedures can firmly support. I also point out one internal inconsistency in the "energy-level spacing" analogy. The aim is constructive: to keep the interesting empirical observations, while making clear what they do (and do not) imply about quantum entanglement in the usual Hilbert-space sense, especially when "energy" is defined by rank.
>
---
#### [new 171] OS-Symphony: A Holistic Framework for Robust and Generalist Computer-Using Agent
- **分类: cs.MA; cs.AI; cs.CL; cs.CV; cs.HC**

- **简介: 该论文提出OS-Symphony框架，解决CUA在长任务和新领域中的鲁棒性和泛化性问题。通过记忆机制和视觉导航工具提升自动化性能。**

- **链接: [https://arxiv.org/pdf/2601.07779v1](https://arxiv.org/pdf/2601.07779v1)**

> **作者:** Bowen Yang; Kaiming Jin; Zhenyu Wu; Zhaoyang Liu; Qiushi Sun; Zehao Li; JingJing Xie; Zhoumianze Liu; Fangzhi Xu; Kanzhi Cheng; Qingyun Li; Yian Wang; Yu Qiao; Zun Wang; Zichen Ding
>
> **备注:** 31 pages, 11 figures, 12 tables
>
> **摘要:** While Vision-Language Models (VLMs) have significantly advanced Computer-Using Agents (CUAs), current frameworks struggle with robustness in long-horizon workflows and generalization in novel domains. These limitations stem from a lack of granular control over historical visual context curation and the absence of visual-aware tutorial retrieval. To bridge these gaps, we introduce OS-Symphony, a holistic framework that comprises an Orchestrator coordinating two key innovations for robust automation: (1) a Reflection-Memory Agent that utilizes milestone-driven long-term memory to enable trajectory-level self-correction, effectively mitigating visual context loss in long-horizon tasks; (2) Versatile Tool Agents featuring a Multimodal Searcher that adopts a SeeAct paradigm to navigate a browser-based sandbox to synthesize live, visually aligned tutorials, thereby resolving fidelity issues in unseen scenarios. Experimental results demonstrate that OS-Symphony delivers substantial performance gains across varying model scales, establishing new state-of-the-art results on three online benchmarks, notably achieving 65.84% on OSWorld.
>
---
#### [new 172] LLM Flow Processes for Text-Conditioned Regression
- **分类: cs.LG; cs.CL; stat.ML**

- **简介: 该论文属于回归任务，旨在结合元学习与大语言模型优势。通过融合扩散过程与LLM概率，提升回归性能。**

- **链接: [https://arxiv.org/pdf/2601.06147v1](https://arxiv.org/pdf/2601.06147v1)**

> **作者:** Felix Biggs; Samuel Willis
>
> **摘要:** Meta-learning methods for regression like Neural (Diffusion) Processes achieve impressive results, but with these models it can be difficult to incorporate expert prior knowledge and information contained in metadata. Large Language Models (LLMs) are trained on giant corpora including varied real-world regression datasets alongside their descriptions and metadata, leading to impressive performance on a range of downstream tasks. Recent work has extended this to regression tasks and is able to leverage such prior knowledge and metadata, achieving surprisingly good performance, but this still rarely matches dedicated meta-learning methods. Here we introduce a general method for sampling from a product-of-experts of a diffusion or flow matching model and an `expert' with binned probability density; we apply this to combine neural diffusion processes with LLM token probabilities for regression (which may incorporate textual knowledge), exceeding the empirical performance of either alone.
>
---
#### [new 173] L-RAG: Balancing Context and Retrieval with Entropy-Based Lazy Loading
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文提出L-RAG，解决RAG系统计算开销大的问题。通过熵值判断是否检索，实现准确率与效率的平衡。属于知识增强生成任务。**

- **链接: [https://arxiv.org/pdf/2601.06551v1](https://arxiv.org/pdf/2601.06551v1)**

> **作者:** Sergii Voloshyn
>
> **摘要:** Retrieval-Augmented Generation (RAG) has emerged as the predominant paradigm for grounding Large Language Model outputs in factual knowledge, effectively mitigating hallucinations. However, conventional RAG systems operate under a "retrieve-always" assumption, querying vector databases for every input regardless of query complexity. This static approach incurs substantial computational overhead and inference latency, particularly problematic for high-throughput production deployments. We introduce L-RAG (Lazy Retrieval-Augmented Generation), an adaptive framework that implements hierarchical context management through entropy-based gating. L-RAG employs a two-tier architecture: queries are first processed with a compact document summary, and expensive chunk retrieval is triggered only when the model's predictive entropy exceeds a calibrated threshold, signaling genuine uncertainty. Through experiments on SQuAD 2.0 (N=500) using the Phi-2 model, we demonstrate that L-RAG provides a tunable accuracy-efficiency trade-off: at a conservative threshold (tau=0.5), L-RAG achieves 78.2% accuracy, matching Standard RAG (77.8%), with 8% retrieval reduction; at a balanced threshold (tau=1.0), retrieval reduction increases to 26% with modest accuracy trade-off (76.0%). Latency analysis shows that L-RAG saves 80-210ms per query when retrieval latency exceeds 500ms. Analysis of entropy distributions reveals statistically significant separation (p < 0.001) between correct predictions (H=1.72) and errors (H=2.20), validating entropy as a reliable uncertainty signal. L-RAG offers a practical, training-free approach toward more efficient RAG deployment, providing system architects with a configurable knob to balance accuracy and throughput requirements.
>
---
#### [new 174] Filtering Beats Fine Tuning: A Bayesian Kalman View of In Context Learning in LLMs
- **分类: cs.LG; cs.CL; cs.IT**

- **简介: 该论文将大语言模型的推理时适应建模为贝叶斯状态估计问题，解决的是上下文学习的理论解释与稳定性分析。通过卡尔曼滤波视角，揭示了不确定性动态与参数高效适应机制。**

- **链接: [https://arxiv.org/pdf/2601.06100v1](https://arxiv.org/pdf/2601.06100v1)**

> **作者:** Andrew Kiruluta
>
> **摘要:** We present a theory-first framework that interprets inference-time adaptation in large language models (LLMs) as online Bayesian state estimation. Rather than modeling rapid adaptation as implicit optimization or meta-learning, we formulate task- and context-specific learning as the sequential inference of a low-dimensional latent adaptation state governed by a linearized state-space model. Under Gaussian assumptions, adaptation follows a Kalman recursion with closed-form updates for both the posterior mean and covariance. This perspective elevates epistemic uncertainty to an explicit dynamical variable. We show that inference-time learning is driven by covariance collapse, i.e., rapid contraction of posterior uncertainty induced by informative tokens, which typically precedes convergence of the posterior mean. Using observability conditions on token-level Jacobians, we establish stability of the Bayesian filter, prove exponential covariance contraction rates, and derive mean-square error bounds. Gradient descent, natural-gradient methods, and meta-learning updates arise as singular, noise-free limits of the filtering dynamics, positioning optimization-based adaptation as a degenerate approximation of Bayesian inference. The resulting theory provides a unified probabilistic account of in-context learning, parameter-efficient adaptation, and test-time learning without parameter updates. It yields explicit guarantees on stability and sample efficiency, offers a principled interpretation of prompt informativeness via information accumulation, and clarifies the role of uncertainty dynamics absent from existing accounts. Minimal illustrative experiments corroborate the qualitative predictions of the theory.
>
---
#### [new 175] Manifold-based Sampling for In-Context Hallucination Detection in Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于 hallucination 检测任务，旨在解决 LLM 生成内容不准确的问题。通过引入基于流形的演示采样方法 MB-ICL，提升事实可靠性。**

- **链接: [https://arxiv.org/pdf/2601.06196v1](https://arxiv.org/pdf/2601.06196v1)**

> **作者:** Bodla Krishna Vamshi; Rohan Bhatnagar; Haizhao Yang
>
> **摘要:** Large language models (LLMs) frequently generate factually incorrect or unsupported content, commonly referred to as hallucinations. Prior work has explored decoding strategies, retrieval augmentation, and supervised fine-tuning for hallucination detection, while recent studies show that in-context learning (ICL) can substantially influence factual reliability. However, existing ICL demonstration selection methods often rely on surface-level similarity heuristics and exhibit limited robustness across tasks and models. We propose MB-ICL, a manifold-based demonstration sampling framework for selecting in-context demonstrations that leverages latent representations extracted from frozen LLMs. By jointly modeling local manifold structure and class-aware prototype geometry, MB-ICL selects demonstrations based on their proximity to learned prototypes rather than lexical or embedding similarity alone. Across factual verification (FEVER) and hallucination detection (HaluEval) benchmarks, MB-ICL outperforms standard ICL selection baselines in the majority of evaluated settings, with particularly strong gains on dialogue and summarization tasks. The method remains robust under temperature perturbations and model variation, indicating improved stability compared to heuristic retrieval strategies. While lexical retrieval can remain competitive in certain question-answering regimes, our results demonstrate that manifold-based prototype selection provides a reliable and training light approach for hallucination detection without modifying LLM parameters, offering a principled direction for improved ICL demonstration selection.
>
---
#### [new 176] Tone Matters: The Impact of Linguistic Tone on Hallucination in VLMs
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉语言模型安全研究，旨在解决模型在不同提示语气下产生幻觉的问题。通过构建数据集并分析提示强度对幻觉的影响，揭示模型在应对压力时的局限性。**

- **链接: [https://arxiv.org/pdf/2601.06460v1](https://arxiv.org/pdf/2601.06460v1)**

> **作者:** Weihao Hong; Zhiyuan Jiang; Bingyu Shen; Xinlei Guan; Yangyi Feng; Meng Xu; Boyang Li
>
> **备注:** 10 pages, 6 figures, WACV Workshop
>
> **摘要:** Vision-Language Models (VLMs) are increasingly used in safety-critical applications that require reliable visual grounding. However, these models often hallucinate details that are not present in the image to satisfy user prompts. While recent datasets and benchmarks have been introduced to evaluate systematic hallucinations in VLMs, many hallucination behaviors remain insufficiently characterized. In particular, prior work primarily focuses on object presence or absence, leaving it unclear how prompt phrasing and structural constraints can systematically induce hallucinations. In this paper, we investigate how different forms of prompt pressure influence hallucination behavior. We introduce Ghost-100, a procedurally generated dataset of synthetic scenes in which key visual details are deliberately removed, enabling controlled analysis of absence-based hallucinations. Using a structured 5-Level Prompt Intensity Framework, we vary prompts from neutral queries to toxic demands and rigid formatting constraints. We evaluate three representative open-weight VLMs: MiniCPM-V 2.6-8B, Qwen2-VL-7B, and Qwen3-VL-8B. Across all three models, hallucination rates do not increase monotonically with prompt intensity. All models exhibit reductions at higher intensity levels at different thresholds, though not all show sustained reduction under maximum coercion. These results suggest that current safety alignment is more effective at detecting semantic hostility than structural coercion, revealing model-specific limitations in handling compliance pressure. Our dataset is available at: https://github.com/bli1/tone-matters
>
---
#### [new 177] Judge Model for Large-scale Multimodality Benchmarks
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.MA**

- **简介: 该论文提出一种多模态评估模型，用于可靠、可解释地评价多模态任务。解决现有评估方法不足的问题，通过聚合多模态判断提供诊断反馈。**

- **链接: [https://arxiv.org/pdf/2601.06106v1](https://arxiv.org/pdf/2601.06106v1)**

> **作者:** Min-Han Shih; Yu-Hsin Wu; Yu-Wei Chen
>
> **摘要:** We propose a dedicated multimodal Judge Model designed to provide reliable, explainable evaluation across a diverse suite of tasks. Our benchmark spans text, audio, image, and video modalities, drawing from carefully sampled public datasets with fixed seeds to ensure reproducibility and minimize train test leakage. Instead of simple scoring, our framework aggregates multimodal judgments, analyzes the quality and reasoning consistency of model outputs, and generates diagnostic feedback. We evaluate several MLLMs, including Gemini 2.5, Phi 4, and Qwen 2.5, across 280 multimodal samples and compare judge model assessments with human annotators. Results show strong alignment between the Judge Model and human scores, demonstrating its potential as a scalable, interpretable evaluation pipeline for future multimodal AI research.
>
---
#### [new 178] MixDPO: Modeling Preference Strength for Pluralistic Alignment
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理中的对齐任务，旨在解决现有方法无法捕捉人类偏好强度差异的问题。提出MixDPO模型，通过建模偏好强度变化提升对齐效果。**

- **链接: [https://arxiv.org/pdf/2601.06180v1](https://arxiv.org/pdf/2601.06180v1)**

> **作者:** Saki Imai; Pedram Heydari; Anthony Sicilia; Asteria Kaeberlein; Katherine Atwell; Malihe Alikhani
>
> **摘要:** Preference based alignment objectives implicitly assume that all human preferences are expressed with equal strength. In practice, however, preference strength varies across individuals and contexts -- a phenomenon established in behavioral economics and discrete choice theory. This mismatch limits the ability of existing objectives to faithfully capture heterogeneous human judgments. Inspired by this literature, we introduce Mixed Logit Direct Preference Optimization (MixDPO), a generalization of Direct Preference Optimization that models variation in preference strength. MixDPO enables alignment objectives to capture heterogeneity in how strongly preferences are expressed across training examples. We evaluate MixDPO on three preference datasets using two open-weight language models. Across datasets, MixDPO improves aggregate alignment performance (+11.2 points on Pythia-2.8B) while preserving subgroup level preferences, with the largest gains appearing in settings with higher inferred preference heterogeneity. MixDPO makes preference heterogeneity explicit through learned strength distributions. We release our code for reproducibility.
>
---
#### [new 179] CrossTrafficLLM: A Human-Centric Framework for Interpretable Traffic Intelligence via Large Language Model
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于交通预测任务，旨在解决交通状态与自然语言描述的对齐问题。通过融合大语言模型，同时预测交通状态并生成文本描述，提升交通智能的可解释性与实用性。**

- **链接: [https://arxiv.org/pdf/2601.06042v1](https://arxiv.org/pdf/2601.06042v1)**

> **作者:** Zeming Du; Qitan Shao; Hongfei Liu; Yong Zhang
>
> **摘要:** While accurate traffic forecasting is vital for Intelligent Transportation Systems (ITS), effectively communicating predicted conditions via natural language for human-centric decision support remains a challenge and is often handled separately. To address this, we propose CrossTrafficLLM, a novel GenAI-driven framework that simultaneously predicts future spatiotemporal traffic states and generates corresponding natural language descriptions, specifically targeting conditional abnormal event summaries. We tackle the core challenge of aligning quantitative traffic data with qualitative textual semantics by leveraging Large Language Models (LLMs) within a unified architecture. This design allows generative textual context to improve prediction accuracy while ensuring generated reports are directly informed by the forecast. Technically, a text-guided adaptive graph convolutional network is employed to effectively merge high-level semantic information with the traffic network structure. Evaluated on the BJTT dataset, CrossTrafficLLM demonstrably surpasses state-of-the-art methods in both traffic forecasting performance and text generation quality. By unifying prediction and description generation, CrossTrafficLLM delivers a more interpretable, and actionable approach to generative traffic intelligence, offering significant advantages for modern ITS applications.
>
---
#### [new 180] BabyVision: Visual Reasoning Beyond Language
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出BabyVision，一个评估多模态大模型视觉能力的基准，旨在解决其依赖语言而缺乏基础视觉推理的问题。**

- **链接: [https://arxiv.org/pdf/2601.06521v1](https://arxiv.org/pdf/2601.06521v1)**

> **作者:** Liang Chen; Weichu Xie; Yiyan Liang; Hongfeng He; Hans Zhao; Zhibo Yang; Zhiqi Huang; Haoning Wu; Haoyu Lu; Y. charles; Yiping Bao; Yuantao Fan; Guopeng Li; Haiyang Shen; Xuanzhong Chen; Wendong Xu; Shuzheng Si; Zefan Cai; Wenhao Chai; Ziqi Huang; Fangfu Liu; Tianyu Liu; Baobao Chang; Xiaobo Hu; Kaiyuan Chen; Yixin Ren; Yang Liu; Yuan Gong; Kuan Li
>
> **备注:** 26 pages, Homepage at https://unipat.ai/blog/BabyVision
>
> **摘要:** While humans develop core visual skills long before acquiring language, contemporary Multimodal LLMs (MLLMs) still rely heavily on linguistic priors to compensate for their fragile visual understanding. We uncovered a crucial fact: state-of-the-art MLLMs consistently fail on basic visual tasks that humans, even 3-year-olds, can solve effortlessly. To systematically investigate this gap, we introduce BabyVision, a benchmark designed to assess core visual abilities independent of linguistic knowledge for MLLMs. BabyVision spans a wide range of tasks, with 388 items divided into 22 subclasses across four key categories. Empirical results and human evaluation reveal that leading MLLMs perform significantly below human baselines. Gemini3-Pro-Preview scores 49.7, lagging behind 6-year-old humans and falling well behind the average adult score of 94.1. These results show despite excelling in knowledge-heavy evaluations, current MLLMs still lack fundamental visual primitives. Progress in BabyVision represents a step toward human-level visual perception and reasoning capabilities. We also explore solving visual reasoning with generation models by proposing BabyVision-Gen and automatic evaluation toolkit. Our code and benchmark data are released at https://github.com/UniPat-AI/BabyVision for reproduction.
>
---
#### [new 181] SPINAL -- Scaling-law and Preference Integration in Neural Alignment Layers
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出SPINAL，用于诊断大语言模型对齐过程中表示空间的几何变化，解决模型对齐效果评估问题，通过分析层间结构变化实现对齐效果的量化审计。**

- **链接: [https://arxiv.org/pdf/2601.06238v1](https://arxiv.org/pdf/2601.06238v1)**

> **作者:** Arion Das; Partha Pratim Saha; Amit Dhanda; Vinija Jain; Aman Chadha; Amitava Das
>
> **摘要:** Direct Preference Optimization (DPO) is a principled, scalable alternative to RLHF for aligning large language models from pairwise preferences, but its internal geometric footprint remains undercharacterized, limiting audits, checkpoint comparisons, and failure prediction. We introduce SPINAL (Scaling-law and Preference Integration in Neural Alignment Layers), a diagnostic that measures how alignment reshapes representations across depth by tracing localized structural change layer by layer. Across model families, DPO produces a layerwise calibration effect concentrated in the final decoder blocks (often layers 21-30), where preference gradients most directly affect the next-token distribution. SPINAL encodes each checkpoint as a depth trace over (layer index, contraction score, transport score). The contraction score summarizes how quickly the tail of a layer's spectrum decays (how fast small modes vanish); higher values indicate stronger contraction into fewer effective directions. The transport score summarizes how much the token distribution shifts between adjacent layers using a bounded overlap measure; lower values indicate shorter, smoother steps through representation space. Aligned checkpoints show a late-layer ramp-up in contraction and a smooth reduction in transport, consistent with tightened and stabilized policy mass, while unaligned models trace higher-curvature, more entropic, and geometrically incoherent depth paths. Overall, alignment is geometrically localized: the final layers encode the dominant preference-induced corrections. SPINAL turns this localization into a practical audit signal, quantifying where alignment concentrates, how strongly it manifests, and when it begins to destabilize during training.
>
---
#### [new 182] Measuring Social Bias in Vision-Language Models with Face-Only Counterfactuals from Real Photos
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于社会偏见检测任务，旨在解决视觉-语言模型中因视觉混淆导致的偏见归因问题。通过构建面部属性可控的反事实数据集，评估模型在不同任务中的偏见表现。**

- **链接: [https://arxiv.org/pdf/2601.06931v1](https://arxiv.org/pdf/2601.06931v1)**

> **作者:** Haodong Chen; Qiang Huang; Jiaqi Zhao; Qiuping Jiang; Xiaojun Chang; Jun Yu
>
> **备注:** 18 pages, 18 figures, and 3 tables
>
> **摘要:** Vision-Language Models (VLMs) are increasingly deployed in socially consequential settings, raising concerns about social bias driven by demographic cues. A central challenge in measuring such social bias is attribution under visual confounding: real-world images entangle race and gender with correlated factors such as background and clothing, obscuring attribution. We propose a \textbf{face-only counterfactual evaluation paradigm} that isolates demographic effects while preserving real-image realism. Starting from real photographs, we generate counterfactual variants by editing only facial attributes related to race and gender, keeping all other visual factors fixed. Based on this paradigm, we construct \textbf{FOCUS}, a dataset of 480 scene-matched counterfactual images across six occupations and ten demographic groups, and propose \textbf{REFLECT}, a benchmark comprising three decision-oriented tasks: two-alternative forced choice, multiple-choice socioeconomic inference, and numeric salary recommendation. Experiments on five state-of-the-art VLMs reveal that demographic disparities persist under strict visual control and vary substantially across task formulations. These findings underscore the necessity of controlled, counterfactual audits and highlight task design as a critical factor in evaluating social bias in multimodal models.
>
---
#### [new 183] Gecko: An Efficient Neural Architecture Inherently Processing Sequences with Arbitrary Lengths
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于序列建模任务，旨在解决长序列处理效率与扩展性问题。提出Gecko架构，通过改进注意力机制和记忆管理，实现高效长序列处理。**

- **链接: [https://arxiv.org/pdf/2601.06463v1](https://arxiv.org/pdf/2601.06463v1)**

> **作者:** Xuezhe Ma; Shicheng Wen; Linghao Jin; Bilge Acun; Ruihang Lai; Bohan Hou; Will Lin; Hao Zhang; Songlin Yang; Ryan Lee; Mengxi Wu; Jonathan May; Luke Zettlemoyer; Carole-Jean Wu
>
> **备注:** 13 pages, 5 figure and 3 tables
>
> **摘要:** Designing a unified neural network to efficiently and inherently process sequential data with arbitrary lengths is a central and challenging problem in sequence modeling. The design choices in Transformer, including quadratic complexity and weak length extrapolation, have limited their ability to scale to long sequences. In this work, we propose Gecko, a neural architecture that inherits the design of Mega and Megalodon (exponential moving average with gated attention), and further introduces multiple technical components to improve its capability to capture long range dependencies, including timestep decay normalization, sliding chunk attention mechanism, and adaptive working memory. In a controlled pretraining comparison with Llama2 and Megalodon in the scale of 7 billion parameters and 2 trillion training tokens, Gecko achieves better efficiency and long-context scalability. Gecko reaches a training loss of 1.68, significantly outperforming Llama2-7B (1.75) and Megalodon-7B (1.70), and landing close to Llama2-13B (1.67). Notably, without relying on any context-extension techniques, Gecko exhibits inherent long-context processing and retrieval capabilities, stably handling sequences of up to 4 million tokens and retrieving information from contexts up to $4\times$ longer than its attention window. Code: https://github.com/XuezheMax/gecko-llm
>
---
#### [new 184] GRPO with State Mutations: Improving LLM-Based Hardware Test Plan Generation
- **分类: cs.AR; cs.CL; cs.LG**

- **简介: 该论文属于硬件验证任务，旨在提升LLM生成测试用例的能力。通过引入GRPO-SMu方法，改进模型在RTL设计中的测试计划生成效果。**

- **链接: [https://arxiv.org/pdf/2601.07593v1](https://arxiv.org/pdf/2601.07593v1)**

> **作者:** Dimple Vijay Kochar; Nathaniel Pinckney; Guan-Ting Liu; Chia-Tung Ho; Chenhui Deng; Haoxing Ren; Brucek Khailany
>
> **摘要:** RTL design often relies heavily on ad-hoc testbench creation early in the design cycle. While large language models (LLMs) show promise for RTL code generation, their ability to reason about hardware specifications and generate targeted test plans remains largely unexplored. We present the first systematic study of LLM reasoning capabilities for RTL verification stimuli generation, establishing a two-stage framework that decomposes test plan generation from testbench execution. Our benchmark reveals that state-of-the-art models, including DeepSeek-R1 and Claude-4.0-Sonnet, achieve only 15.7-21.7% success rates on generating stimuli that pass golden RTL designs. To improve LLM generated stimuli, we develop a comprehensive training methodology combining supervised fine-tuning with a novel reinforcement learning approach, GRPO with State Mutation (GRPO-SMu), which enhances exploration by varying input mutations. Our approach leverages a tree-based branching mutation strategy to construct training data comprising equivalent and mutated trees, moving beyond linear mutation approaches to provide rich learning signals. Training on this curated dataset, our 7B parameter model achieves a 33.3% golden test pass rate and a 13.9% mutation detection rate, representing a 17.6% absolute improvement over baseline and outperforming much larger general-purpose models. These results demonstrate that specialized training methodologies can significantly enhance LLM reasoning capabilities for hardware verification tasks, establishing a foundation for automated sub-unit testing in semiconductor design workflows.
>
---
#### [new 185] From RLHF to Direct Alignment: A Theoretical Unification of Preference Learning for Large Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于大语言模型对齐任务，旨在解决偏好学习方法选择混乱的问题。通过理论分析，统一对抗方法，明确设计选择的影响。**

- **链接: [https://arxiv.org/pdf/2601.06108v1](https://arxiv.org/pdf/2601.06108v1)**

> **作者:** Tarun Raheja; Nilay Pochhi
>
> **摘要:** Aligning large language models (LLMs) with human preferences has become essential for safe and beneficial AI deployment. While Reinforcement Learning from Human Feedback (RLHF) established the dominant paradigm, a proliferation of alternatives -- Direct Preference Optimization (DPO), Identity Preference Optimization (IPO), Kahneman-Tversky Optimization (KTO), Simple Preference Optimization (SimPO), and many others -- has left practitioners without clear guidance on method selection. This survey provides a \textit{theoretical unification} of preference learning methods, revealing that the apparent diversity reduces to principled choices along three orthogonal axes: \textbf{(I) Preference Model} (what likelihood model underlies the objective), \textbf{(II) Regularization Mechanism} (how deviation from reference policies is controlled), and \textbf{(III) Data Distribution} (online vs.\ offline learning and coverage requirements). We formalize each axis with precise definitions and theorems, establishing key results including the coverage separation between online and offline methods, scaling laws for reward overoptimization, and conditions under which direct alignment methods fail. Our analysis reveals that failure modes -- length hacking, mode collapse, likelihood displacement -- arise from specific, predictable combinations of design choices. We synthesize empirical findings across 50+ papers and provide a practitioner's decision guide for method selection. The framework transforms preference learning from an empirical art into a theoretically grounded discipline.
>
---
#### [new 186] Lost in the Noise: How Reasoning Models Fail with Contextual Distractors
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于AI模型鲁棒性研究，针对外部信息噪声导致的推理失败问题，提出NoisyBench基准和RARE方法，提升模型在噪声环境下的表现。**

- **链接: [https://arxiv.org/pdf/2601.07226v1](https://arxiv.org/pdf/2601.07226v1)**

> **作者:** Seongyun Lee; Yongrae Jo; Minju Seo; Moontae Lee; Minjoon Seo
>
> **备注:** Preprint
>
> **摘要:** Recent advances in reasoning models and agentic AI systems have led to an increased reliance on diverse external information. However, this shift introduces input contexts that are inherently noisy, a reality that current sanitized benchmarks fail to capture. We introduce NoisyBench, a comprehensive benchmark that systematically evaluates model robustness across 11 datasets in RAG, reasoning, alignment, and tool-use tasks against diverse noise types, including random documents, irrelevant chat histories, and hard negative distractors. Our evaluation reveals a catastrophic performance drop of up to 80% in state-of-the-art models when faced with contextual distractors. Crucially, we find that agentic workflows often amplify these errors by over-trusting noisy tool outputs, and distractors can trigger emergent misalignment even without adversarial intent. We find that prompting, context engineering, SFT, and outcome-reward only RL fail to ensure robustness; in contrast, our proposed Rationale-Aware Reward (RARE) significantly strengthens resilience by incentivizing the identification of helpful information within noise. Finally, we uncover an inverse scaling trend where increased test-time computation leads to worse performance in noisy settings and demonstrate via attention visualization that models disproportionately focus on distractor tokens, providing vital insights for building the next generation of robust, reasoning-capable agents.
>
---
#### [new 187] An evaluation of LLMs for political bias in Western media: Israel-Hamas and Ukraine-Russia wars
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文属于政治偏见检测任务，旨在分析西方媒体在俄乌和巴以冲突中的立场。通过LLMs比较左右翼及中立观点，揭示媒体偏见变化及模型差异。**

- **链接: [https://arxiv.org/pdf/2601.06132v1](https://arxiv.org/pdf/2601.06132v1)**

> **作者:** Rohitash Chandra; Haoyan Chen; Yaqing Zhang; Jiacheng Chen; Yuting Wu
>
> **摘要:** Political bias in media plays a critical role in shaping public opinion, voter behaviour, and broader democratic discourse. Subjective opinions and political bias can be found in media sources, such as newspapers, depending on their funding mechanisms and alliances with political parties. Automating the detection of political biases in media content can limit biases in elections. The impact of large language models (LLMs) in politics and media studies is becoming prominent. In this study, we utilise LLMs to compare the left-wing, right-wing, and neutral political opinions expressed in the Guardian and BBC. We review newspaper reporting that includes significant events such as the Russia-Ukraine war and the Hamas-Israel conflict. We analyse the proportion for each opinion to find the bias under different LLMs, including BERT, Gemini, and DeepSeek. Our results show that after the outbreak of the wars, the political bias of Western media shifts towards the left-wing and each LLM gives a different result. DeepSeek consistently showed a stable Left-leaning tendency, while BERT and Gemini remained closer to the Centre. The BBC and The Guardian showed distinct reporting behaviours across the two conflicts. In the Russia-Ukraine war, both outlets maintained relatively stable positions; however, in the Israel-Hamas conflict, we identified larger political bias shifts, particularly in Guardian coverage, suggesting a more event-driven pattern of reporting bias. These variations suggest that LLMs are shaped not only by their training data and architecture, but also by underlying worldviews with associated political biases.
>
---
#### [new 188] ReinPool: Reinforcement Learning Pooling Multi-Vector Embeddings for Retrieval System
- **分类: cs.IR; cs.CL; cs.CV**

- **简介: 该论文提出ReinPool，解决多向量嵌入在检索系统中存储成本高的问题。通过强化学习动态筛选和聚合向量，显著压缩表示并提升检索性能。**

- **链接: [https://arxiv.org/pdf/2601.07125v1](https://arxiv.org/pdf/2601.07125v1)**

> **作者:** Sungguk Cha; DongWook Kim; Mintae Kim; Youngsub Han; Byoung-Ki Jeon; Sangyeob Lee
>
> **备注:** 5 pages
>
> **摘要:** Multi-vector embedding models have emerged as a powerful paradigm for document retrieval, preserving fine-grained visual and textual details through token-level representations. However, this expressiveness comes at a staggering cost: storing embeddings for every token inflates index sizes by over $1000\times$ compared to single-vector approaches, severely limiting scalability. We introduce \textbf{ReinPool}, a reinforcement learning framework that learns to dynamically filter and pool multi-vector embeddings into compact, retrieval-optimized representations. By training with an inverse retrieval objective and NDCG-based rewards, ReinPool identifies and retains only the most discriminative vectors without requiring manual importance annotations. On the Vidore V2 benchmark across three vision-language embedding models, ReinPool compresses multi-vector representations by $746$--$1249\times$ into single vectors while recovering 76--81\% of full multi-vector retrieval performance. Compared to static mean pooling baselines, ReinPool achieves 22--33\% absolute NDCG@3 improvement, demonstrating that learned selection significantly outperforms heuristic aggregation.
>
---
#### [new 189] Attention Mechanism and Heuristic Approach: Context-Aware File Ranking Using Multi-Head Self-Attention
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文属于软件工程中的变更影响分析任务，旨在提升受影响文件的召回率。通过引入多头自注意力机制，学习特征间的上下文关系，优化文件排名，提高检索效果。**

- **链接: [https://arxiv.org/pdf/2601.06185v1](https://arxiv.org/pdf/2601.06185v1)**

> **作者:** Pradeep Kumar Sharma; Shantanu Godbole; Sarada Prasad Jena; Hritvik Shrivastava
>
> **摘要:** The identification and ranking of impacted files within software reposi-tories is a key challenge in change impact analysis. Existing deterministic approaches that combine heuristic signals, semantic similarity measures, and graph-based centrality metrics have demonstrated effectiveness in nar-rowing candidate search spaces, yet their recall plateaus. This limitation stems from the treatment of features as linearly independent contributors, ignoring contextual dependencies and relationships between metrics that characterize expert reasoning patterns. To address this limitation, we propose the application of Multi-Head Self-Attention as a post-deterministic scoring refinement mechanism. Our approach learns contextual weighting between features, dynamically adjust-ing importance levels per file based on relational behavior exhibited across candidate file sets. The attention mechanism produces context-aware adjustments that are additively combined with deterministic scores, pre-serving interpretability while enabling reasoning similar to that performed by experts when reviewing change surfaces. We focus on recall rather than precision, as false negatives (missing impacted files) are far more costly than false positives (irrelevant files that can be quickly dismissed during review). Empirical evaluation on 200 test cases demonstrates that the introduc-tion of self-attention improves Top-50 recall from approximately 62-65% to between 78-82% depending on repository complexity and structure, achiev-ing 80% recall at Top-50 files. Expert validation yields improvement from 6.5/10 to 8.6/10 in subjective accuracy alignment. This transformation bridges the reasoning capability gap between deterministic automation and expert judgment, improving recall in repository-aware effort estimation.
>
---
#### [new 190] Are LLM Decisions Faithful to Verbal Confidence?
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于AI可信性研究任务，探讨LLM的自信表达是否与决策相关。工作是通过框架RiskEval测试模型在不同惩罚下的回避策略，发现模型缺乏将不确定性转化为最优决策的能力。**

- **链接: [https://arxiv.org/pdf/2601.07767v1](https://arxiv.org/pdf/2601.07767v1)**

> **作者:** Jiawei Wang; Yanfei Zhou; Siddartha Devic; Deqing Fu
>
> **摘要:** Large Language Models (LLMs) can produce surprisingly sophisticated estimates of their own uncertainty. However, it remains unclear to what extent this expressed confidence is tied to the reasoning, knowledge, or decision making of the model. To test this, we introduce $\textbf{RiskEval}$: a framework designed to evaluate whether models adjust their abstention policies in response to varying error penalties. Our evaluation of several frontier models reveals a critical dissociation: models are neither cost-aware when articulating their verbal confidence, nor strategically responsive when deciding whether to engage or abstain under high-penalty conditions. Even when extreme penalties render frequent abstention the mathematically optimal strategy, models almost never abstain, resulting in utility collapse. This indicates that calibrated verbal confidence scores may not be sufficient to create trustworthy and interpretable AI systems, as current models lack the strategic agency to convert uncertainty signals into optimal and risk-sensitive decisions.
>
---
## 更新

#### [replaced 001] VC-Inspector: Advancing Reference-free Evaluation of Video Captions with Factual Analy
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出VC-Inspector，用于视频字幕的事实性无参考评估。解决现有评估方法在事实准确性、上下文处理上的不足，通过生成可控错误的字幕进行训练和评估，提升评估准确性。**

- **链接: [https://arxiv.org/pdf/2509.16538v2](https://arxiv.org/pdf/2509.16538v2)**

> **作者:** Shubhashis Roy Dipta; Tz-Ying Wu; Subarna Tripathi
>
> **摘要:** We propose VC-Inspector, a lightweight, open-source large multimodal model (LMM) for reference-free evaluation of video captions, with a focus on factual accuracy. Unlike existing metrics that suffer from limited context handling, weak factuality assessment, or reliance on proprietary services, VC-Inspector offers a reproducible, fact-aware alternative that aligns closely with human judgments. To enable robust training and interpretable evaluation, we introduce a systematic approach for generating captions with controllable errors, paired with graded quality scores and explanatory annotations. Experiments show that VC-Inspector achieves state-of-the-art correlation with human judgments, generalizing across diverse domains (e.g., VATEX-Eval, Flickr8K-Expert, and Flickr8K-CF benchmarks) and revealing the potential for caption improvement.
>
---
#### [replaced 002] Bridging the Linguistic Divide: A Survey on Leveraging Large Language Models for Machine Translation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于机器翻译任务，探讨如何利用大语言模型提升翻译效果。工作包括分析不同方法，如提示、微调和强化学习，解决数据不足和上下文理解问题。**

- **链接: [https://arxiv.org/pdf/2504.01919v4](https://arxiv.org/pdf/2504.01919v4)**

> **作者:** Baban Gain; Dibyanayan Bandyopadhyay; Asif Ekbal; Trilok Nath Singh
>
> **摘要:** Large Language Models (LLMs) are rapidly reshaping machine translation (MT), particularly by introducing instruction-following, in-context learning, and preference-based alignment into what has traditionally been a supervised encoder-decoder paradigm. This survey provides a comprehensive and up-to-date overview of how LLMs are being leveraged for MT across data regimes, languages, and application settings. We systematically analyze prompting-based methods, parameter-efficient and full fine-tuning strategies, synthetic data generation, preference-based optimization, and reinforcement learning with human and weakly supervised feedback. Special attention is given to low-resource translation, where we examine the roles of synthetic data quality, diversity, and preference signals, as well as the limitations of current RLHF pipelines. We further review recent advances in Mixture-of-Experts models, MT-focused LLMs, and multilingual alignment, highlighting trade-offs between scalability, specialization, and accessibility. Beyond sentence-level translation, we survey emerging document-level and discourse-aware MT methods with LLMs, showing that most approaches extend sentence-level pipelines through structured context selection, post-editing, or reranking rather than requiring fundamentally new data regimes or architectures. Finally, we discuss LLM-based evaluation, its strengths and biases, and its role alongside learned metrics. Overall, this survey positions LLM-based MT as an evolution of traditional MT systems, where gains increasingly depend on data quality, preference alignment, and context utilization rather than scale alone, and outlines open challenges for building robust, inclusive, and controllable translation systems.
>
---
#### [replaced 003] Aligning the Spectrum: Hybrid Graph Pre-training and Prompt Tuning across Homophily and Heterophily
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于图预训练任务，解决知识迁移效率问题。针对现有方法依赖单一频段滤波器的局限，提出HS-GPPT模型，通过混合频谱和对齐提示调优提升知识利用率。**

- **链接: [https://arxiv.org/pdf/2508.11328v3](https://arxiv.org/pdf/2508.11328v3)**

> **作者:** Haitong Luo; Suhang Wang; Weiyao Zhang; Ruiqi Meng; Xuying Meng; Yujun Zhang
>
> **备注:** Under Review
>
> **摘要:** Graph ``pre-training and prompt-tuning'' aligns downstream tasks with pre-trained objectives to enable efficient knowledge transfer under limited supervision. However, current methods typically rely on single-filter backbones (e.g., low-pass), whereas real-world graphs exhibit inherent spectral diversity. Our theoretical \textit{Spectral Specificity} principle reveals that effective knowledge transfer requires alignment between pre-trained spectral filters and the intrinsic spectrum of downstream graphs. This identifies two fundamental limitations: (1) Knowledge Bottleneck: single-filter models suffer from irreversible information loss by suppressing signals from other frequency bands (e.g., high-frequency); (2) Utilization Bottleneck: spectral mismatches between pre-trained filters and downstream spectra lead to significant underutilization of pre-trained knowledge. To bridge this gap, we propose HS-GPPT. We utilize a hybrid spectral backbone to construct an abundant knowledge basis. Crucially, we introduce Spectral-Aligned Prompt Tuning to actively align the downstream graph's spectrum with diverse pre-trained filters, facilitating comprehensive knowledge utilization across both homophily and heterophily. Extensive experiments validate the effectiveness under both transductive and inductive learning settings.
>
---
#### [replaced 004] Continual Pretraining on Encrypted Synthetic Data for Privacy-Preserving LLMs
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于隐私保护任务，旨在解决在小规模领域语料中预训练大语言模型时的隐私问题。通过合成加密数据实现持续预训练，保护敏感信息。**

- **链接: [https://arxiv.org/pdf/2601.05635v2](https://arxiv.org/pdf/2601.05635v2)**

> **作者:** Honghao Liu; Xuhui Jiang; Chengjin Xu; Cehao Yang; Yiran Cheng; Lionel Ni; Jian Guo
>
> **摘要:** Preserving privacy in sensitive data while pretraining large language models on small, domain-specific corpora presents a significant challenge. In this work, we take an exploratory step toward privacy-preserving continual pretraining by proposing an entity-based framework that synthesizes encrypted training data to protect personally identifiable information (PII). Our approach constructs a weighted entity graph to guide data synthesis and applies deterministic encryption to PII entities, enabling LLMs to encode new knowledge through continual pretraining while granting authorized access to sensitive data through decryption keys. Our results on limited-scale datasets demonstrate that our pretrained models outperform base models and ensure PII security, while exhibiting a modest performance gap compared to models trained on unencrypted synthetic data. We further show that increasing the number of entities and leveraging graph-based synthesis improves model performance, and that encrypted models retain instruction-following capabilities with long retrieved contexts. We discuss the security implications and limitations of deterministic encryption, positioning this work as an initial investigation into the design space of encrypted data pretraining for privacy-preserving LLMs. Our code is available at https://github.com/DataArcTech/SoE.
>
---
#### [replaced 005] Is Chain-of-Thought Reasoning of LLMs a Mirage? A Data Distribution Lens
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理领域，探讨CoT推理的有效性。针对CoT在分布外数据上失效的问题，提出数据分布视角，通过实验验证其脆弱性。**

- **链接: [https://arxiv.org/pdf/2508.01191v4](https://arxiv.org/pdf/2508.01191v4)**

> **作者:** Chengshuai Zhao; Zhen Tan; Pingchuan Ma; Dawei Li; Bohan Jiang; Yancheng Wang; Yingzhen Yang; Huan Liu
>
> **备注:** Accepted by the Foundations of Reasoning in Language Models (FoRLM) at NeurIPS 2025
>
> **摘要:** Chain-of-Thought (CoT) prompting has been shown to be effective in eliciting structured reasoning (i.e., CoT reasoning) from large language models (LLMs). Regardless of its popularity, recent studies expose its failures in some reasoning tasks, raising fundamental questions about the nature of CoT reasoning. In this work, we propose a data distribution lens to understand when and why CoT reasoning succeeds or fails. We hypothesize that CoT reasoning reflects a structured inductive bias learned from in-distribution data, enabling models to conditionally generate reasoning trajectories that approximate those observed during training. As such, the effectiveness of CoT reasoning is fundamentally governed by the nature and degree of distribution discrepancy between training data and test queries. Guided by this lens, we dissect CoT reasoning via three dimensions: task, length, and format. To test the hypothesis, we introduce DataAlchemy, an abstract and fully controllable environment that trains LLMs from scratch and systematically probes them under various distribution conditions. Through rigorous controlled experiments, we reveal that CoT reasoning is a brittle mirage when it is pushed beyond training distributions, emphasizing the ongoing challenge of achieving genuine and generalizable reasoning.
>
---
#### [replaced 006] Adding Alignment Control to Language Models
- **分类: cs.CL**

- **简介: 该论文属于语言模型对齐任务，旨在解决个性化对齐问题。提出CLM方法，在模型中添加可学习层实现对齐控制，通过调整参数实现对齐程度的插值与外推。**

- **链接: [https://arxiv.org/pdf/2503.04346v3](https://arxiv.org/pdf/2503.04346v3)**

> **作者:** Wenhong Zhu; Weinan Zhang; Rui Wang
>
> **备注:** I have changed the title of the paper and resubmitted a new one on arXiv titled "Flexible realignment of language models" (arXiv:2506.12704)
>
> **摘要:** Post-training alignment has increasingly become a crucial factor in enhancing the usability of language models (LMs). However, the strength of alignment varies depending on individual preferences. This paper proposes a method to incorporate alignment control into a single model, referred to as CLM. This approach adds one identity layer preceding the initial layers and performs preference learning only on this layer to map unaligned input token embeddings into the aligned space. Experimental results demonstrate that this efficient fine-tuning method performs comparable to full fine-tuning. During inference, the input embeddings are processed through the aligned and unaligned layers, which are then merged through the interpolation coefficient. By controlling this parameter, the alignment exhibits a clear interpolation and extrapolation phenomenon.
>
---
#### [replaced 007] Disco-RAG: Discourse-Aware Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出Disco-RAG，解决RAG系统中缺乏对话语结构建模的问题。通过构建话语树和修辞图，增强生成过程中的知识融合能力，提升问答和长文档摘要任务效果。**

- **链接: [https://arxiv.org/pdf/2601.04377v2](https://arxiv.org/pdf/2601.04377v2)**

> **作者:** Dongqi Liu; Hang Ding; Qiming Feng; Jian Li; Xurong Xie; Zhucun Xue; Chengjie Wang; Jiangning Zhang; Yabiao Wang
>
> **摘要:** Retrieval-Augmented Generation (RAG) has emerged as an important means of enhancing the performance of large language models (LLMs) in knowledge-intensive tasks. However, most existing RAG strategies treat retrieved passages in a flat and unstructured way, which prevents the model from capturing structural cues and constrains its ability to synthesize knowledge from dispersed evidence across documents. To overcome these limitations, we propose Disco-RAG, a discourse-aware framework that explicitly injects discourse signals into the generation process. Our method constructs intra-chunk discourse trees to capture local hierarchies and builds inter-chunk rhetorical graphs to model cross-passage coherence. These structures are jointly integrated into a planning blueprint that conditions the generation. Experiments on question answering and long-document summarization benchmarks show the efficacy of our approach. Disco-RAG achieves state-of-the-art results on the benchmarks without fine-tuning. These findings underscore the important role of discourse structure in advancing RAG systems.
>
---
#### [replaced 008] TagRAG: Tag-guided Hierarchical Knowledge Graph Retrieval-Augmented Generation
- **分类: cs.IR; cs.CL**

- **简介: 该论文提出TagRAG，解决传统RAG在知识图谱检索中的效率与适应性问题，通过标签引导的层次化知识图谱提升生成效果。**

- **链接: [https://arxiv.org/pdf/2601.05254v2](https://arxiv.org/pdf/2601.05254v2)**

> **作者:** Wenbiao Tao; Xinyuan Li; Yunshi Lan; Weining Qian
>
> **摘要:** Retrieval-Augmented Generation enhances language models by retrieving external knowledge to support informed and grounded responses. However, traditional RAG methods rely on fragment-level retrieval, limiting their ability to address query-focused summarization queries. GraphRAG introduces a graph-based paradigm for global knowledge reasoning, yet suffers from inefficiencies in information extraction, costly resource consumption, and poor adaptability to incremental updates. To overcome these limitations, we propose TagRAG, a tag-guided hierarchical knowledge graph RAG framework designed for efficient global reasoning and scalable graph maintenance. TagRAG introduces two key components: (1) Tag Knowledge Graph Construction, which extracts object tags and their relationships from documents and organizes them into hierarchical domain tag chains for structured knowledge representation, and (2) Tag-Guided Retrieval-Augmented Generation, which retrieves domain-centric tag chains to localize and synthesize relevant knowledge during inference. This design significantly adapts to smaller language models, improves retrieval granularity, and supports efficient knowledge increment. Extensive experiments on UltraDomain datasets spanning Agriculture, Computer Science, Law, and cross-domain settings demonstrate that TagRAG achieves an average winning rate of 78.36% against baselines while maintaining about 14.6x construction and 1.9x retrieval efficiency compared with GraphRAG.
>
---
#### [replaced 009] A Vision for Multisensory Intelligence: Sensing, Synergy, and Science
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出多感官人工智能的研究愿景，旨在解决AI与人类感官融合的问题。通过感知、科学和协同三个方向，推动AI在多模态信息处理上的发展。**

- **链接: [https://arxiv.org/pdf/2601.04563v2](https://arxiv.org/pdf/2601.04563v2)**

> **作者:** Paul Pu Liang
>
> **摘要:** Our experience of the world is multisensory, spanning a synthesis of language, sight, sound, touch, taste, and smell. Yet, artificial intelligence has primarily advanced in digital modalities like text, vision, and audio. This paper outlines a research vision for multisensory artificial intelligence over the next decade. This new set of technologies can change how humans and AI experience and interact with one another, by connecting AI to the human senses and a rich spectrum of signals from physiological and tactile cues on the body, to physical and social signals in homes, cities, and the environment. We outline how this field must advance through three interrelated themes of sensing, science, and synergy. Firstly, research in sensing should extend how AI captures the world in richer ways beyond the digital medium. Secondly, developing a principled science for quantifying multimodal heterogeneity and interactions, developing unified modeling architectures and representations, and understanding cross-modal transfer. Finally, we present new technical challenges to learn synergy between modalities and between humans and AI, covering multisensory integration, alignment, reasoning, generation, generalization, and experience. Accompanying this vision paper are a series of projects, resources, and demos of latest advances from the Multisensory Intelligence group at the MIT Media Lab, see https://mit-mi.github.io/.
>
---
#### [replaced 010] Improving Indigenous Language Machine Translation with Synthetic Data and Language-Specific Preprocessing
- **分类: cs.CL**

- **简介: 该论文属于机器翻译任务，旨在解决低资源原住民语言缺乏平行语料的问题。通过合成数据和语言特定预处理提升翻译质量。**

- **链接: [https://arxiv.org/pdf/2601.03135v2](https://arxiv.org/pdf/2601.03135v2)**

> **作者:** Aashish Dhawan; Christopher Driggers-Ellis; Christan Grant; Daisy Zhe Wang
>
> **摘要:** Low-resource indigenous languages often lack the parallel corpora required for effective neural machine translation (NMT). Synthetic data generation offers a practical strategy for mitigating this limitation in data-scarce settings. In this work, we augment curated parallel datasets for indigenous languages of the Americas with synthetic sentence pairs generated using a high-capacity multilingual translation model. We fine-tune a multilingual mBART model on curated-only and synthetically augmented data and evaluate translation quality using chrF++, the primary metric used in recent AmericasNLP shared tasks for agglutinative languages. We further apply language-specific preprocessing, including orthographic normalization and noise-aware filtering, to reduce corpus artifacts. Experiments on Guarani-Spanish and Quechua-Spanish translation show consistent chrF++ improvements from synthetic data augmentation, while diagnostic experiments on Aymara highlight the limitations of generic preprocessing for highly agglutinative languages.
>
---
#### [replaced 011] The Pragmatic Mind of Machines: Tracing the Emergence of Pragmatic Competence in Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究大语言模型的语用能力，旨在解决其如何在训练中获得语境理解与意图推理的问题。通过构建数据集评估模型在不同训练阶段的语用表现。**

- **链接: [https://arxiv.org/pdf/2505.18497v3](https://arxiv.org/pdf/2505.18497v3)**

> **作者:** Kefan Yu; Qingcheng Zeng; Weihao Xuan; Wanxin Li; Jingyi Wu; Rob Voigt
>
> **摘要:** Current large language models (LLMs) have demonstrated emerging capabilities in social intelligence tasks, including implicature resolution and theory-of-mind reasoning, both of which require substantial pragmatic understanding. However, how LLMs acquire this pragmatic competence throughout the training process remains poorly understood. In this work, we introduce ALTPRAG, a dataset grounded in the pragmatic concept of alternatives, to evaluate whether LLMs at different training stages can accurately infer nuanced speaker intentions. Each instance pairs two equally plausible yet pragmatically divergent continuations and requires the model to (i) infer the speaker's intended meaning and (ii) explain when and why a speaker would choose one utterance over its alternative, thus directly probing pragmatic competence through contrastive reasoning. We systematically evaluate 22 LLMs across 3 key training stages: after pre-training, supervised fine-tuning (SFT), and preference optimization, to examine the development of pragmatic competence. Our results show that even base models exhibit notable sensitivity to pragmatic cues, which improves consistently with increases in model and data scale. Additionally, SFT and RLHF contribute further gains, particularly in cognitive-pragmatic scenarios. These findings highlight pragmatic competence as an emergent and compositional property of LLM training and offer new insights for aligning models with human communicative norms.
>
---
#### [replaced 012] Instructions are all you need: Self-supervised Reinforcement Learning for Instruction Following
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于指令遵循任务，解决多约束指令难以执行的问题。提出一种无需外部监督的自监督强化学习框架，通过指令生成奖励信号和伪标签，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2510.14420v2](https://arxiv.org/pdf/2510.14420v2)**

> **作者:** Qingyu Ren; Qianyu He; Bowei Zhang; Jie Zeng; Jiaqing Liang; Yanghua Xiao; Weikang Zhou; Zeye Sun; Fei Yu
>
> **摘要:** Language models often struggle to follow multi-constraint instructions that are crucial for real-world applications. Existing reinforcement learning (RL) approaches suffer from dependency on external supervision and sparse reward signals from multi-constraint tasks. We propose a label-free self-supervised RL framework that eliminates dependency on external supervision by deriving reward signals directly from instructions and generating pseudo-labels for reward model training. Our approach introduces constraint decomposition strategies and efficient constraint-wise binary classification to address sparse reward challenges while maintaining computational efficiency. Experiments show that our approach generalizes well, achieving strong improvements across 3 in-domain and 5 out-of-domain datasets, including challenging agentic and multi-turn instruction following. The data and code are publicly available at https://github.com/Rainier-rq/verl-if
>
---
#### [replaced 013] Deployability-Centric Infrastructure-as-Code Generation: Fail, Learn, Refine, and Succeed through LLM-Empowered DevOps Simulation
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文属于IaC生成任务，旨在解决LLM生成的IaC模板部署成功率低的问题。通过构建基准测试和提出迭代反馈框架IaCGen，提升模板的可部署性。**

- **链接: [https://arxiv.org/pdf/2506.05623v3](https://arxiv.org/pdf/2506.05623v3)**

> **作者:** Tianyi Zhang; Shidong Pan; Zejun Zhang; Zhenchang Xing; Xiaoyu Sun
>
> **备注:** Accepted by FSE 2026
>
> **摘要:** Infrastructure-as-Code (IaC) generation holds significant promise for automating cloud infrastructure provisioning. Recent advances in Large Language Models (LLMs) present a promising opportunity to democratize IaC development by generating deployable infrastructure templates from natural language descriptions. However, current evaluation focuses on syntactic correctness while ignoring deployability, the critical measure of the utility of IaC configuration files. Six state-of-the-art LLMs performed poorly on deployability, achieving only 20.8$\sim$30.2% deployment success rate on the first attempt. In this paper, we construct DPIaC-Eval, the first deployability-centric IaC template benchmark consisting of 153 real-world scenarios cross 58 unique services. Also, we propose an LLM-based deployability-centric framework, dubbed IaCGen, that uses iterative feedback mechanism encompassing format verification, syntax checking, and live deployment stages, thereby closely mirroring the real DevOps workflows. Results show that IaCGen can make 54.6$\sim$91.6% generated IaC templates from all evaluated models deployable in the first 10 iterations. Additionally, human-in-the-loop feedback that provide direct guidance for the deployability errors, can further boost the performance to over 90% passItr@25 on all evaluated LLMs. Furthermore, we explore the trustworthiness of the generated IaC templates on user intent alignment and security compliance. The poor performance (25.2% user requirement coverage and 8.4% security compliance rate) indicates a critical need for continued research in this domain.
>
---
#### [replaced 014] Reducing Hallucinations in LLMs via Factuality-Aware Preference Learning
- **分类: cs.CL**

- **简介: 该论文属于大语言模型事实性优化任务，旨在减少幻觉。通过改进DPO方法，引入事实性标签，提升模型事实准确性并降低幻觉率。**

- **链接: [https://arxiv.org/pdf/2601.03027v2](https://arxiv.org/pdf/2601.03027v2)**

> **作者:** Sindhuja Chaduvula; Ahmed Y. Radwan; Azib Farooq; Yani Ioannou; Shaina Raza
>
> **摘要:** Preference alignment methods such as RLHF and Direct Preference Optimization (DPO) improve instruction following, but they can also reinforce hallucinations when preference judgments reward fluency and confidence over factual correctness. We introduce F-DPO (Factuality-aware Direct Preference Optimization), a simple extension of DPO that uses only binary factuality labels. F-DPO (i) applies a label-flipping transformation that corrects misordered preference pairs so the chosen response is never less factual than the rejected one, and (ii) adds a factuality-aware margin that emphasizes pairs with clear correctness differences, while reducing to standard DPO when both responses share the same factuality. We construct factuality-aware preference data by augmenting DPO pairs with binary factuality indicators and synthetic hallucinated variants. Across seven open-weight LLMs (1B-14B), F-DPO consistently improves factuality and reduces hallucination rates relative to both base models and standard DPO. On Qwen3-8B, F-DPO reduces hallucination rates by five times (from 0.424 to 0.084) while improving factuality scores by 50 percent (from 5.26 to 7.90). F-DPO also generalizes to out-of-distribution benchmarks: on TruthfulQA, Qwen2.5-14B achieves plus 17 percent MC1 accuracy (0.500 to 0.585) and plus 49 percent MC2 accuracy (0.357 to 0.531). F-DPO requires no auxiliary reward model, token-level annotations, or multi-stage training.
>
---
#### [replaced 015] Uncovering the Computational Roles of Nonlinearity in Sequence Modeling Using Almost-Linear RNNs
- **分类: cs.LG; cs.AI; cs.CL; nlin.CD; physics.comp-ph**

- **简介: 该论文研究序列建模任务，解决非线性在其中的必要性问题。通过AL-RNN框架分析非线性作用，揭示其在计算中的关键角色。**

- **链接: [https://arxiv.org/pdf/2506.07919v2](https://arxiv.org/pdf/2506.07919v2)**

> **作者:** Manuel Brenner; Georgia Koppe
>
> **备注:** Published in Transactions on Machine Learning Research (TMLR), https://openreview.net/forum?id=qI2Vt9P9rl
>
> **摘要:** Sequence modeling tasks across domains such as natural language processing, time series forecasting, and control require learning complex input-output mappings. Nonlinear recurrence is theoretically required for universal approximation of sequence-to-sequence functions, yet linear recurrent models often prove surprisingly effective. This raises the question of when nonlinearity is truly required. We present a framework to systematically dissect the functional role of nonlinearity in recurrent networks, identifying when it is computationally necessary and what mechanisms it enables. We address this using Almost Linear Recurrent Neural Networks (AL-RNNs), which allow recurrence nonlinearity to be gradually attenuated and decompose network dynamics into analyzable linear regimes, making computational mechanisms explicit. We illustrate the framework across diverse synthetic and real-world tasks, including classic sequence modeling benchmarks, a neuroscientific stimulus-selection task, and a multi-task suite. We demonstrate how the AL-RNN's piecewise linear structure enables identification of computational primitives such as gating, rule-based integration, and memory-dependent transients, revealing that these operations emerge within predominantly linear backbones. Across tasks, sparse nonlinearity improves interpretability by reducing and localizing nonlinear computations, promotes shared representations in multi-task settings, and reduces computational cost. Moreover, sparse nonlinearity acts as a useful inductive bias: in low-data regimes or when tasks require discrete switching between linear regimes, sparsely nonlinear models often match or exceed fully nonlinear architectures. Our findings provide a principled approach for identifying where nonlinearity is functionally necessary, guiding the design of recurrent architectures that balance performance, efficiency, and interpretability.
>
---
#### [replaced 016] Perspectives in Play: A Multi-Perspective Approach for More Inclusive NLP Systems
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的文本分类任务，旨在解决传统方法忽视个体观点差异的问题。通过多视角方法和软标签，提升模型的包容性与准确性。**

- **链接: [https://arxiv.org/pdf/2506.20209v2](https://arxiv.org/pdf/2506.20209v2)**

> **作者:** Benedetta Muscato; Lucia Passaro; Gizem Gezici; Fosca Giannotti
>
> **摘要:** In the realm of Natural Language Processing (NLP), common approaches for handling human disagreement consist of aggregating annotators' viewpoints to establish a single ground truth. However, prior studies show that disregarding individual opinions can lead can lead to the side effect of underrepresenting minority perspectives, especially in subjective tasks, where annotators may systematically disagree because of their preferences. Recognizing that labels reflect the diverse backgrounds, life experiences, and values of individuals, this study proposes a new multi-perspective approach using soft labels to encourage the development of the next generation of perspective aware models, more inclusive and pluralistic. We conduct an extensive analysis across diverse subjective text classification tasks, including hate speech, irony, abusive language, and stance detection, to highlight the importance of capturing human disagreements, often overlooked by traditional aggregation methods. Results show that the multi-perspective approach not only better approximates human label distributions, as measured by Jensen-Shannon Divergence (JSD), but also achieves superior classification performance (higher F1 scores), outperforming traditional approaches. However, our approach exhibits lower confidence in tasks like irony and stance detection, likely due to the inherent subjectivity present in the texts. Lastly, leveraging Explainable AI (XAI), we explore model uncertainty and uncover meaningful insights into model predictions.
>
---
#### [replaced 017] Benchmarking and Learning Real-World Customer Service Dialogue
- **分类: cs.CL**

- **简介: 该论文属于智能客服领域，旨在解决对话系统与实际需求脱节的问题。通过构建OlaBench基准和OlaMind模型，提升服务质量和部署效果。**

- **链接: [https://arxiv.org/pdf/2510.22143v2](https://arxiv.org/pdf/2510.22143v2)**

> **作者:** Tianhong Gao; Jundong Shen; Jiapeng Wang; Bei Shi; Ying Ju; Junfeng Yao; Huiyu Yu
>
> **摘要:** Existing benchmarks and training pipelines for industrial intelligent customer service (ICS) remain misaligned with real-world dialogue requirements, overemphasizing verifiable task success while under-measuring subjective service quality and realistic failure modes, leaving a gap between offline gains and deployable dialogue behavior. We close this gap with a benchmark-to-optimization loop: we first introduce OlaBench, an ICS benchmark spanning retrieval-augmented generation, workflow-based systems, and agentic settings, which evaluates service capability, safety, and latency sensitivity; moreover, motivated by OlaBench results showing state-of-the-art LLMs still fall short, we propose OlaMind, which distills reusable reasoning patterns and service strategies from expert dialogues and applies rubric-aware staged exploration--exploitation reinforcement learning to improve model capability. OlaMind surpasses GPT-5.2 and Gemini 3 Pro on OlaBench (78.72 vs. 70.58/70.84) and, in online A/B tests, delivers an average +23.67% issue resolution and -6.6% human transfer rate versus the baseline, bridging offline gains to deployment. Together, OlaBench and OlaMind advance ICS systems toward more anthropomorphic, professional, and reliable deployment.
>
---
#### [replaced 018] Zero-Shot Context-Aware ASR for Diverse Arabic Varieties
- **分类: cs.CL**

- **简介: 该论文属于语音识别任务，旨在解决阿拉伯语多样方言和口音的零样本识别问题。通过上下文感知解码方法提升模型性能，适用于不同架构的ASR系统。**

- **链接: [https://arxiv.org/pdf/2511.18774v2](https://arxiv.org/pdf/2511.18774v2)**

> **作者:** Bashar Talafha; Amin Abu Alhassan; Muhammad Abdul-Mageed
>
> **摘要:** Zero-shot ASR for Arabic remains challenging: while multilingual models perform well on Modern Standard Arabic (MSA), error rates rise sharply on dialectal and accented speech due to linguistic mismatch and scarce labeled data. We study context-aware decoding as a lightweight test-time adaptation paradigm that conditions inference on external side information without parameter updates. For promptable encoder-decoder ASR (e.g., Whisper), we incorporate context through (i) decoder prompting with first-pass hypotheses and (ii) encoder/decoder prefixing with retrieved speech-text exemplars, complemented by simple prompt reordering and optional speaker-matched synthetic exemplars to improve robustness in informal and multi-speaker settings. To extend contextual adaptation beyond promptable architectures, we introduce proxy-guided n-best selection for CTC ASR: given one or more external proxy hypotheses, we select from a model's n-best list by minimizing text-level distance to the proxies, enabling contextual inference without direct prompting. Across ten Arabic conditions spanning MSA, accented MSA, and multiple dialects, context-aware decoding yields average relative WER reductions of 22.29% on MSA, 20.54 on accented MSA, and 9.15% on dialectal Arabic. For CTC models, proxy-guided selection reduces WER by 15.6% relative on MSA and recovers a substantial fraction of oracle n-best gains, demonstrating that context-aware inference generalizes beyond encoder-decoder ASR.
>
---
#### [replaced 019] The Best Instruction-Tuning Data are Those That Fit
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于监督微调任务，旨在提升大语言模型性能。针对数据分布不匹配问题，提出GRAPE框架，通过选择与目标模型分布更契合的响应进行训练，显著提升模型效果。**

- **链接: [https://arxiv.org/pdf/2502.04194v3](https://arxiv.org/pdf/2502.04194v3)**

> **作者:** Dylan Zhang; Qirun Dai; Hao Peng
>
> **摘要:** High-quality supervised fine-tuning (SFT) data are crucial for eliciting strong capabilities from pretrained large language models (LLMs). Typically, instructions are paired with multiple responses sampled from other LLMs, which are often out of the distribution of the target model to be fine-tuned. This, at scale, can lead to diminishing returns and even hurt the models' performance and robustness. We propose **GRAPE**, a novel SFT framework that accounts for the unique characteristics of the target model. For each instruction, it gathers responses from various LLMs and selects the one with the highest probability measured by the target model, indicating that it aligns most closely with the target model's pretrained distribution; it then proceeds with standard SFT training. We first evaluate GRAPE with a controlled experiment, where we sample various solutions for each question in UltraInteract from multiple models and fine-tune commonly used LMs like LLaMA3.1-8B, Mistral-7B, and Qwen2.5-7B on GRAPE-selected data. GRAPE significantly outperforms strong baselines, including distilling from the strongest model with an absolute gain of up to 13.8%, averaged across benchmarks, and training on 3x more data with a maximum performance improvement of 17.3%. GRAPE's strong performance generalizes to realistic settings. We experiment with the post-training data used for Tulu3 and Olmo-2. GRAPE outperforms strong baselines trained on 4.5 times more data by 6.1% and a state-of-the-art data selection approach by 3% on average performance. Remarkably, using 1/3 of the data and half the number of epochs, GRAPE enables LLaMA3.1-8B to surpass the performance of Tulu3-SFT by 3.5%.
>
---
#### [replaced 020] Memory-Efficient Training for Text-Dependent SV with Independent Pre-trained Models
- **分类: eess.AS; cs.CL; cs.LG**

- **简介: 该论文属于文本依赖说话人验证任务，旨在解决传统方法计算成本高和模型适应性差的问题。通过独立使用预训练模型并进行领域适配，实现了高效且准确的验证系统。**

- **链接: [https://arxiv.org/pdf/2411.10828v2](https://arxiv.org/pdf/2411.10828v2)**

> **作者:** Seyed Ali Farokh; Hossein Zeinali
>
> **备注:** Accepted at ROCLING 2025
>
> **摘要:** This paper presents our submission to the Iranian division of the Text-Dependent Speaker Verification Challenge (TdSV) 2024. Conventional TdSV approaches typically jointly model speaker and linguistic features, requiring unsegmented inputs during training and incurring high computational costs. Additionally, these methods often fine-tune large-scale pre-trained speaker embedding models on the target domain dataset, which may compromise the pre-trained models' original ability to capture speaker-specific characteristics. To overcome these limitations, we employ a TdSV system that utilizes two pre-trained models independently and demonstrate that, by leveraging pre-trained models with targeted domain adaptation, competitive results can be achieved while avoiding the substantial computational costs associated with joint fine-tuning on unsegmented inputs in conventional approaches. Our best system reached a MinDCF of 0.0358 on the evaluation subset and secured first place in the challenge.
>
---
#### [replaced 021] Towards Stable and Effective Reinforcement Learning for Mixture-of-Experts
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决Mixture-of-Experts（MoE）训练不稳定的问题。通过优化重要性采样权重，提升MoE模型的训练稳定性和性能。**

- **链接: [https://arxiv.org/pdf/2510.23027v2](https://arxiv.org/pdf/2510.23027v2)**

> **作者:** Di Zhang; Xun Wu; Shaohan Huang; Lingjie Jiang; Yaru Hao; Li Dong; Zewen Chi; Zhifang Sui; Furu Wei
>
> **备注:** Added additional experiments, improved analysis, and fixed minor issues
>
> **摘要:** Recent advances in reinforcement learning (RL) have substantially improved the training of large-scale language models, leading to significant gains in generation quality and reasoning ability. However, most existing research focuses on dense models, while RL training for Mixture-of-Experts (MoE) architectures remains underexplored. To address the instability commonly observed in MoE training, we propose a novel router-aware approach to optimize importance sampling (IS) weights in off-policy RL. Specifically, we design a rescaling strategy guided by router logits, which effectively reduces gradient variance and mitigates training divergence. Experimental results demonstrate that our method significantly improves both the convergence stability and the final performance of MoE models, highlighting the potential of RL algorithmic innovations tailored to MoE architectures and providing a promising direction for efficient training of large-scale expert models.
>
---
#### [replaced 022] Can Reasoning Help Large Language Models Capture Human Annotator Disagreement?
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究LLM是否能通过推理捕捉人类标注者分歧，属于自然语言处理中的标注建模任务。工作对比不同推理方式对模型分歧建模的影响，发现RLVR会降低性能，而CoT提升性能。**

- **链接: [https://arxiv.org/pdf/2506.19467v3](https://arxiv.org/pdf/2506.19467v3)**

> **作者:** Jingwei Ni; Yu Fan; Vilém Zouhar; Donya Rooein; Alexander Hoyle; Mrinmaya Sachan; Markus Leippold; Dirk Hovy; Elliott Ash
>
> **备注:** EACL 2026 Main
>
> **摘要:** Variation in human annotation (i.e., disagreements) is common in NLP, often reflecting important information like task subjectivity and sample ambiguity. Modeling this variation is important for applications that are sensitive to such information. Although RLVR-style reasoning (Reinforcement Learning with Verifiable Rewards) has improved Large Language Model (LLM) performance on many tasks, it remains unclear whether such reasoning enables LLMs to capture informative variation in human annotation. In this work, we evaluate the influence of different reasoning settings on LLM disagreement modeling. We systematically evaluate each reasoning setting across model sizes, distribution expression methods, and steering methods, resulting in 60 experimental setups across 3 tasks. Surprisingly, our results show that RLVR-style reasoning degrades performance in disagreement modeling, while naive Chain-of-Thought (CoT) reasoning improves the performance of RLHF LLMs (RL from human feedback). These findings underscore the potential risk of replacing human annotators with reasoning LLMs, especially when disagreements are important.
>
---
#### [replaced 023] Training Versatile Coding Agents in Synthetic Environments
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文属于软件工程与AI结合的任务，旨在解决现有训练方法依赖外部数据和任务单一的问题。提出SWE-Playground，通过合成环境生成多样化编码任务，提升代理的泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.12216v2](https://arxiv.org/pdf/2512.12216v2)**

> **作者:** Yiqi Zhu; Apurva Gandhi; Graham Neubig
>
> **摘要:** Prior works on training software engineering agents have explored utilizing existing resources such as issues on GitHub repositories to construct software engineering tasks and corresponding test suites. These approaches face two key limitations: (1) their reliance on pre-existing GitHub repositories offers limited flexibility, and (2) their primary focus on issue resolution tasks restricts their applicability to the much wider variety of tasks a software engineer must handle. To overcome these challenges, we introduce SWE-Playground, a novel pipeline for generating environments and trajectories which supports the training of versatile coding agents. Unlike prior efforts, SWE-Playground synthetically generates projects and tasks from scratch with strong language models and agents, eliminating reliance on external data sources. This allows us to tackle a much wider variety of coding tasks, such as reproducing issues by generating unit tests and implementing libraries from scratch. We demonstrate the effectiveness of this approach on three distinct benchmarks, and results indicate that SWE-Playground produces trajectories with dense training signal, enabling agents to reach comparable performance with significantly fewer trajectories than previous works.
>
---
#### [replaced 024] Through the LLM Looking Glass: A Socratic Probing of Donkeys, Elephants, and Markets
- **分类: cs.CL**

- **简介: 该论文研究LLM在生成文本时的意识形态框架偏差问题，通过Socratic方法分析其自我评估反馈，旨在提升模型公平性。**

- **链接: [https://arxiv.org/pdf/2503.16674v4](https://arxiv.org/pdf/2503.16674v4)**

> **作者:** Molly Kennedy; Ayyoob Imani; Timo Spinde; Akiko Aizawa; Hinrich Schütze
>
> **摘要:** Large Language Models (LLMs) are widely used for text generation, making it crucial to address potential bias. This study investigates ideological framing bias in LLM-generated articles, focusing on the subtle and subjective nature of such bias in journalistic contexts. We evaluate eight widely used LLMs on two datasets-POLIGEN and ECONOLEX-covering political and economic discourse where framing bias is most pronounced. Beyond text generation, LLMs are increasingly used as evaluators (LLM-as-a-judge), providing feedback that can shape human judgment or inform newer model versions. Inspired by the Socratic method, we further analyze LLMs' feedback on their own outputs to identify inconsistencies in their reasoning. Our results show that most LLMs can accurately annotate ideologically framed text, with GPT-4o achieving human-level accuracy and high agreement with human annotators. However, Socratic probing reveals that when confronted with binary comparisons, LLMs often exhibit preference toward one perspective or perceive certain viewpoints as less biased.
>
---
#### [replaced 025] MMMOS: Multi-domain Multi-axis Audio Quality Assessment
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文属于音频质量评估任务，解决现有模型无法多维度评估不同音频类型的问题。提出MMMOS系统，从四个维度评估音频质量。**

- **链接: [https://arxiv.org/pdf/2507.04094v2](https://arxiv.org/pdf/2507.04094v2)**

> **作者:** Yi-Cheng Lin; Jia-Hung Chen; Hung-yi Lee
>
> **备注:** 4 pages including 1 page of reference. Accepted by ASRU Audio MOS 2025 Challenge
>
> **摘要:** Accurate audio quality estimation is essential for developing and evaluating audio generation, retrieval, and enhancement systems. Existing non-intrusive assessment models predict a single Mean Opinion Score (MOS) for speech, merging diverse perceptual factors and failing to generalize beyond speech. We propose MMMOS, a no-reference, multi-domain audio quality assessment system that estimates four orthogonal axes: Production Quality, Production Complexity, Content Enjoyment, and Content Usefulness across speech, music, and environmental sounds. MMMOS fuses frame-level embeddings from three pretrained encoders (WavLM, MuQ, and M2D) and evaluates three aggregation strategies with four loss functions. By ensembling the top eight models, MMMOS shows a 20-30% reduction in mean squared error and a 4-5% increase in Kendall's τ versus baseline, gains first place in six of eight Production Complexity metrics, and ranks among the top three on 17 of 32 challenge metrics.
>
---
#### [replaced 026] Stronger Baselines for Retrieval-Augmented Generation with Long-Context Language Models
- **分类: cs.CL**

- **简介: 论文探讨长上下文语言模型下的检索增强生成任务，比较多阶段与单阶段方法。结果表明，简单方法DOS RAG表现优异，建议作为强基线。**

- **链接: [https://arxiv.org/pdf/2506.03989v2](https://arxiv.org/pdf/2506.03989v2)**

> **作者:** Alex Laitenberger; Christopher D. Manning; Nelson F. Liu
>
> **备注:** 11 pages, 6 figures, for associated source code, see https://github.com/alex-laitenberger/stronger-baselines-rag
>
> **摘要:** With the rise of long-context language models (LMs) capable of processing tens of thousands of tokens in a single context window, do multi-stage retrieval-augmented generation (RAG) pipelines still offer measurable benefits over simpler, single-stage approaches? To assess this question, we conduct a controlled evaluation for QA tasks under systematically scaled token budgets, comparing two recent multi-stage pipelines, ReadAgent and RAPTOR, against three baselines, including DOS RAG (Document's Original Structure RAG), a simple retrieve-then-read method that preserves original passage order. Despite its straightforward design, DOS RAG consistently matches or outperforms more intricate methods on multiple long-context QA benchmarks. We trace this strength to a combination of maintaining source fidelity and document structure, prioritizing recall within effective context windows, and favoring simplicity over added pipeline complexity. We recommend establishing DOS RAG as a simple yet strong baseline for future RAG evaluations, paired with state-of-the-art embedding and language models, and benchmarked under matched token budgets, to ensure that added pipeline complexity is justified by clear performance gains as models continue to improve.
>
---
#### [replaced 027] KARMA: Leveraging Multi-Agent LLMs for Automated Knowledge Graph Enrichment
- **分类: cs.CL; cs.AI; cs.CE; cs.DL**

- **简介: 该论文提出KARMA框架，利用多智能体LLM自动增强知识图谱，解决人工维护效率低的问题，通过协同分析文本提取并验证知识。**

- **链接: [https://arxiv.org/pdf/2502.06472v2](https://arxiv.org/pdf/2502.06472v2)**

> **作者:** Yuxing Lu; Wei Wu; Xukai Zhao; Rui Peng; Jinzhuo Wang
>
> **备注:** 24 pages, 3 figures, 2 tables
>
> **摘要:** Maintaining comprehensive and up-to-date knowledge graphs (KGs) is critical for modern AI systems, but manual curation struggles to scale with the rapid growth of scientific literature. This paper presents KARMA, a novel framework employing multi-agent large language models (LLMs) to automate KG enrichment through structured analysis of unstructured text. Our approach employs nine collaborative agents, spanning entity discovery, relation extraction, schema alignment, and conflict resolution that iteratively parse documents, verify extracted knowledge, and integrate it into existing graph structures while adhering to domain-specific schema. Experiments on 1,200 PubMed articles from three different domains demonstrate the effectiveness of KARMA in knowledge graph enrichment, with the identification of up to 38,230 new entities while achieving 83.1\% LLM-verified correctness and reducing conflict edges by 18.6\% through multi-layer assessments.
>
---
#### [replaced 028] Provable Secure Steganography Based on Adaptive Dynamic Sampling
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于隐写任务，解决现有隐写方法需模型分布的问题。提出仅需模型API的方案，通过动态采样和映射确保安全性和正确解码。**

- **链接: [https://arxiv.org/pdf/2504.12579v2](https://arxiv.org/pdf/2504.12579v2)**

> **作者:** Kaiyi Pang; Minhao Bai
>
> **摘要:** The security of private communication is increasingly at risk due to widespread surveillance. Steganography, a technique for embedding secret messages within innocuous carriers, enables covert communication over monitored channels. Provably Secure Steganography (PSS), which ensures computational indistinguishability between the normal model output and steganography output, is the state-of-the-art in this field. However, current PSS methods often require obtaining the explicit distributions of the model. In this paper, we propose a provably secure steganography scheme that only requires a model API that accepts a seed as input. Our core mechanism involves sampling a candidate set of tokens and constructing a map from possible message bit strings to these tokens. The output token is selected by applying this mapping to the real secret message, which provably preserves the original model's distribution. To ensure correct decoding, we address collision cases, where multiple candidate messages map to the same token, by maintaining and strategically expanding a dynamic collision set within a bounded size range. Extensive evaluations of three real-world datasets and three large language models demonstrate that our sampling-based method is comparable with existing PSS methods in efficiency and capacity.
>
---
#### [replaced 029] Generative Digital Twins: Vision-Language Simulation Models for Executable Industrial Systems
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于工业仿真任务，旨在通过视觉语言模型生成可执行的FlexScript代码，解决跨模态推理与数字孪生生成问题。**

- **链接: [https://arxiv.org/pdf/2512.20387v3](https://arxiv.org/pdf/2512.20387v3)**

> **作者:** YuChe Hsu; AnJui Wang; TsaiChing Ni; YuanFu Yang
>
> **备注:** 10 pages, 9 figures
>
> **摘要:** We propose a Vision-Language Simulation Model (VLSM) that unifies visual and textual understanding to synthesize executable FlexScript from layout sketches and natural-language prompts, enabling cross-modal reasoning for industrial simulation systems. To support this new paradigm, the study constructs the first large-scale dataset for generative digital twins, comprising over 120,000 prompt-sketch-code triplets that enable multimodal learning between textual descriptions, spatial structures, and simulation logic. In parallel, three novel evaluation metrics, Structural Validity Rate (SVR), Parameter Match Rate (PMR), and Execution Success Rate (ESR), are proposed specifically for this task to comprehensively evaluate structural integrity, parameter fidelity, and simulator executability. Through systematic ablation across vision encoders, connectors, and code-pretrained language backbones, the proposed models achieve near-perfect structural accuracy and high execution robustness. This work establishes a foundation for generative digital twins that integrate visual reasoning and language understanding into executable industrial simulation systems. Project page: https://danielhsu2014.github.io/GDT-VLSM-project/
>
---
#### [replaced 030] A Comprehensive Study on the Effectiveness of ASR Representations for Noise-Robust Speech Emotion Recognition
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于噪声鲁棒语音情感识别任务，旨在解决真实环境中非平稳噪声对情感识别的影响。通过引入ASR模型作为特征提取器，提升情感识别性能。**

- **链接: [https://arxiv.org/pdf/2311.07093v4](https://arxiv.org/pdf/2311.07093v4)**

> **作者:** Xiaohan Shi; Jiajun He; Xingfeng Li; Tomoki Toda
>
> **备注:** Accepted for publication in IEEE Transactions on Audio, Speech, and Language Processing
>
> **摘要:** This paper proposes an efficient attempt to noisy speech emotion recognition (NSER). Conventional NSER approaches have proven effective in mitigating the impact of artificial noise sources, such as white Gaussian noise, but are limited to non-stationary noises in real-world environments due to their complexity and uncertainty. To overcome this limitation, we introduce a new method for NSER by adopting the automatic speech recognition (ASR) model as a noise-robust feature extractor to eliminate non-vocal information in noisy speech. We first obtain intermediate layer information from the ASR model as a feature representation for emotional speech and then apply this representation for the downstream NSER task. Our experimental results show that 1) the proposed method achieves better NSER performance compared with the conventional noise reduction method, 2) outperforms self-supervised learning approaches, and 3) even outperforms text-based approaches using ASR transcription or the ground truth transcription of noisy speech.
>
---
#### [replaced 031] Enhancing Rare Codes via Probability-Biased Directed Graph Attention for Long-Tail ICD Coding
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于医疗信息处理任务，解决ICD编码中罕见代码识别困难的问题。通过引入概率引导的图注意力机制和语言模型增强描述，提升罕见代码的表示与预测效果。**

- **链接: [https://arxiv.org/pdf/2511.09559v2](https://arxiv.org/pdf/2511.09559v2)**

> **作者:** Tianlei Chen; Yuxiao Chen; Yang Li; Feifei Wang
>
> **摘要:** Automated international classification of diseases (ICD) coding aims to assign multiple disease codes to clinical documents and plays a critical role in healthcare informatics. However, its performance is hindered by the extreme long-tail distribution of the ICD ontology, where a few common codes dominate while thousands of rare codes have very few examples. To address this issue, we propose a Probability-Biased Directed Graph Attention model (ProBias) that partitions codes into common and rare sets and allows information to flow only from common to rare codes. Edge weights are determined by conditional co-occurrence probabilities, which guide the attention mechanism to enrich rare-code representations with clinically related signals. To provide higher-quality semantic representations as model inputs, we further employ large language models to generate enriched textual descriptions for ICD codes, offering external clinical context that complements statistical co-occurrence signals. Applied to automated ICD coding, our approach significantly improves the representation and prediction of rare codes, achieving state-of-the-art performance on three benchmark datasets. In particular, we observe substantial gains in macro-averaged F1 score, a key metric for long-tail classification.
>
---
#### [replaced 032] Muse: Towards Reproducible Long-Form Song Generation with Fine-Grained Style Control
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于可控长歌词生成任务，旨在解决学术研究不可复现的问题。工作包括发布开源系统Muse及合成数据集，实现细粒度风格控制的歌曲生成。**

- **链接: [https://arxiv.org/pdf/2601.03973v3](https://arxiv.org/pdf/2601.03973v3)**

> **作者:** Changhao Jiang; Jiahao Chen; Zhenghao Xiang; Zhixiong Yang; Hanchen Wang; Jiabao Zhuang; Xinmeng Che; Jiajun Sun; Hui Li; Yifei Cao; Shihan Dou; Ming Zhang; Junjie Ye; Tao Ji; Tao Gui; Qi Zhang; Xuanjing Huang
>
> **摘要:** Recent commercial systems such as Suno demonstrate strong capabilities in long-form song generation, while academic research remains largely non-reproducible due to the lack of publicly available training data, hindering fair comparison and progress. To this end, we release a fully open-source system for long-form song generation with fine-grained style conditioning, including a licensed synthetic dataset, training and evaluation pipelines, and Muse, an easy-to-deploy song generation model. The dataset consists of 116k fully licensed synthetic songs with automatically generated lyrics and style descriptions paired with audio synthesized by SunoV5. We train Muse via single-stage supervised finetuning of a Qwen-based language model extended with discrete audio tokens using MuCodec, without task-specific losses, auxiliary objectives, or additional architectural components. Our evaluations find that although Muse is trained with a modest data scale and model size, it achieves competitive performance on phoneme error rate, text--music style similarity, and audio aesthetic quality, while enabling controllable segment-level generation across different musical structures. All data, model weights, and training and evaluation pipelines will be publicly released, paving the way for continued progress in controllable long-form song generation research. The project repository is available at https://github.com/yuhui1038/Muse.
>
---
#### [replaced 033] M4FC: a Multimodal, Multilingual, Multicultural, Multitask Real-World Fact-Checking Dataset
- **分类: cs.CL**

- **简介: 该论文提出M4FC数据集，解决多模态事实核查中的数据不足与任务单一问题，涵盖六种任务，支持多语言、多文化场景。**

- **链接: [https://arxiv.org/pdf/2510.23508v2](https://arxiv.org/pdf/2510.23508v2)**

> **作者:** Jiahui Geng; Jonathan Tonglet; Iryna Gurevych
>
> **备注:** Preprint under review. Code and data available at: https://github.com/UKPLab/M4FC
>
> **摘要:** Existing real-world datasets for multimodal fact-checking have multiple limitations: they contain few instances, focus on only one or two languages and tasks, suffer from evidence leakage, or rely on external sets of news articles for sourcing true claims. To address these shortcomings, we introduce M4FC, a new real-world dataset comprising 4,982 images paired with 6,980 claims. The images, verified by professional fact-checkers from 22 organizations, represent a diverse range of cultural and geographic contexts. Each claim is available in one or two out of ten languages. M4FC spans six multimodal fact-checking tasks: visual claim extraction, claimant intent prediction, fake image detection, image contextualization, location verification, and verdict prediction. We provide baseline results for all tasks and analyze how combining intermediate tasks influences verdict prediction performance. We make our dataset and code available.
>
---
#### [replaced 034] Think-J: Learning to Think for Generative LLM-as-a-Judge
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于LLM评价任务，旨在提升生成式LLM作为评判者的性能。通过学习思考过程，优化判断能力，无需人工标注即可提高评估效果。**

- **链接: [https://arxiv.org/pdf/2505.14268v2](https://arxiv.org/pdf/2505.14268v2)**

> **作者:** Hui Huang; Yancheng He; Hongli Zhou; Rui Zhang; Wei Liu; Weixun Wang; Jiaheng Liu; Wenbo Su
>
> **备注:** Accepted by AAAI2026
>
> **摘要:** LLM-as-a-Judge refers to the automatic modeling of preferences for responses generated by Large Language Models (LLMs), which is of significant importance for both LLM evaluation and reward modeling. Although generative LLMs have made substantial progress in various tasks, their performance as LLM-Judge still falls short of expectations. In this work, we propose Think-J, which improves generative LLM-as-a-Judge by learning how to think. We first utilized a small amount of curated data to develop the model with initial judgment thinking capabilities. Subsequently, we optimize the judgment thinking traces based on reinforcement learning (RL). We propose two methods for judgment thinking optimization, based on offline and online RL, respectively. The offline method requires training a critic model to construct positive and negative examples for learning. The online method defines rule-based reward as feedback for optimization. Experimental results showed that our approach can significantly enhance the evaluation capability of generative LLM-Judge, surpassing both generative and classifier-based LLM-Judge without requiring extra human annotations.
>
---
#### [replaced 035] AlignSAE: Concept-Aligned Sparse Autoencoders
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决稀疏自编码器难以与人类概念对齐的问题。通过预训练后监督微调，使特征与预定义本体对齐，提升可解释性和控制性。**

- **链接: [https://arxiv.org/pdf/2512.02004v2](https://arxiv.org/pdf/2512.02004v2)**

> **作者:** Minglai Yang; Xinyu Guo; Zhengliang Shi; Jinhe Bi; Mihai Surdeanu; Liangming Pan
>
> **备注:** 23 pages, 16 figures, 7 tables
>
> **摘要:** Large Language Models (LLMs) encode factual knowledge within hidden parametric spaces that are difficult to inspect or control. While Sparse Autoencoders (SAEs) can decompose hidden activations into more fine-grained, interpretable features, they often struggle to reliably align these features with human-defined concepts, resulting in entangled and distributed feature representations. To address this, we introduce AlignSAE, a method that aligns SAE features with a predefined ontology through a "pre-train, then post-train" curriculum. After an initial unsupervised training phase, we apply supervised post-training to bind specific concepts to dedicated latent slots while preserving the remaining capacity for general reconstruction. This separation creates an interpretable interface where specific concepts can be inspected and controlled without interference from unrelated features. Empirical results demonstrate that AlignSAE enables precise causal interventions, such as reliable "concept swaps", by targeting single, semantically aligned slots, and further supports multi-hop reasoning and a mechanistic probe of grokking-like generalization dynamics.
>
---
#### [replaced 036] ThinkBrake: Mitigating Overthinking in Tool Reasoning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大模型在推理过程中的"过度思考"问题。通过引入ThinkBrake机制，有效减少不必要的推理步骤，提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2510.00546v3](https://arxiv.org/pdf/2510.00546v3)**

> **作者:** Sangjun Song; Minjae Oh; Seungkyu Lee; Sungmin Jo; Yohan Jo
>
> **摘要:** Large Reasoning Models (LRMs) allocate substantial inference-time compute to Chain-of-Thought (CoT) reasoning, improving performance on mathematics, scientific QA, and tool usage. However, this introduces overthinking: LRMs often reach a correct intermediate solution, continue reasoning, and overwrite it with an incorrect answer. We first demonstrate that oracle stopping--where we inject </think> at every sentence boundary and select the best stopping point in hindsight--improves average accuracy by 8\% while reducing thinking tokens by 72\%, exposing substantial overthinking. Motivated by this finding, we propose ThinkBrake, which monitors the log-probability margin between the top continuation token and </think> at sentence boundaries, stopping reasoning when this margin narrows. ThinkBrake requires no training and achieves favorable accuracy-efficiency trade-offs across math, scientific QA, and tool usage benchmarks, reducing thinking token usage by up to 30\%. Furthermore, we provide theoretical analysis showing that ThinkBrake is equivalent to test-time realignment with a reward bonus for the </think> token.
>
---
#### [replaced 037] ToolACE-R: Model-aware Iterative Training and Adaptive Refinement for Tool Learning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于工具学习任务，旨在提升大语言模型调用外部工具的能力。提出ToolACE-R框架，通过模型感知的迭代训练和自适应精炼机制，优化工具调用效果。**

- **链接: [https://arxiv.org/pdf/2504.01400v3](https://arxiv.org/pdf/2504.01400v3)**

> **作者:** Xingshan Zeng; Weiwen Liu; Xu Huang; Zezhong Wang; Lingzhi Wang; Liangyou Li; Yasheng Wang; Lifeng Shang; Xin Jiang; Ruiming Tang; Qun Liu
>
> **备注:** Accepted by AAAI2026
>
> **摘要:** Tool learning, which allows Large Language Models (LLMs) to leverage external tools for solving complex user tasks, has emerged as a promising avenue for extending model capabilities. However, existing approaches primarily focus on data synthesis for fine-tuning LLMs to invoke tools effectively, largely ignoring how to fully stimulate the potential of the model. In this paper, we propose ToolACE-R, a novel framework that includes both model-aware iterative training and adaptive refinement for tool learning. ToolACE-R features a model-aware iterative training procedure that progressively adjust training samples based on the model's evolving capabilities to maximize its potential. Additionally, it incorporates self-refinement training corpus which emphasizes LLM's ability to iteratively refine their tool calls, optimizing performance without requiring external feedback. Furthermore, we introduce adaptive self-refinement mechanism for efficient test-time scaling, where the trained model can autonomously determine when to stop the process based on iterative self-refinement. We conduct extensive experiments across several benchmark datasets, showing that ToolACE-R achieves competitive performance compared to advanced API-based models. The performance of tool invocation can be further improved efficiently through adaptive self-refinement. These results highlight the effectiveness and generalizability of ToolACE-R, offering a promising direction for more efficient and scalable tool learning.
>
---
#### [replaced 038] Browse and Concentrate: Comprehending Multimodal Content via prior-LLM Context Fusion
- **分类: cs.CL**

- **简介: 该论文属于多模态任务，解决多图像上下文理解不足的问题。提出“浏览与聚焦”方法，提升多图像输入的融合效果。**

- **链接: [https://arxiv.org/pdf/2402.12195v3](https://arxiv.org/pdf/2402.12195v3)**

> **作者:** Ziyue Wang; Chi Chen; Yiqi Zhu; Fuwen Luo; Peng Li; Ming Yan; Ji Zhang; Fei Huang; Maosong Sun; Yang Liu
>
> **备注:** 17 pages, 5 figures
>
> **摘要:** With the bloom of Large Language Models (LLMs), Multimodal Large Language Models (MLLMs) that incorporate LLMs with pre-trained vision models have recently demonstrated impressive performance across diverse vision-language tasks. However, they fall short to comprehend context involving multiple images. A primary reason for this shortcoming is that the visual features for each images are encoded individually by frozen encoders before feeding into the LLM backbone, lacking awareness of other images and the multimodal instructions. We term this issue as prior-LLM modality isolation and propose a two phase paradigm, browse-and-concentrate, to enable in-depth multimodal context fusion prior to feeding the features into LLMs. This paradigm initially "browses" through the inputs for essential insights, and then revisits the inputs to "concentrate" on crucial details, guided by these insights, to achieve a more comprehensive understanding of the multimodal inputs. Additionally, we develop training strategies specifically to enhance the understanding of multi-image inputs. Our method markedly boosts the performance on 7 multi-image scenarios, contributing to increments on average accuracy by 2.13% and 7.60% against strong MLLMs baselines with 3B and 11B LLMs, respectively.
>
---
#### [replaced 039] TELEVAL: A Dynamic Benchmark Designed for Spoken Language Models in Chinese Interactive Scenarios
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文提出TELEVAL，一个面向中文口语交互场景的动态评估基准，解决现有评估标准与真实对话不匹配的问题。工作包括构建内容准确性和互动恰当性两个核心评估维度。**

- **链接: [https://arxiv.org/pdf/2507.18061v3](https://arxiv.org/pdf/2507.18061v3)**

> **作者:** Zehan Li; Hongjie Chen; Qing Wang; Yuxin Zhang; Jing Zhou; Hang Lv; Mengjie Du; Yaodong Song; Jie Lian; Jian Kang; Jie Li; Yongxiang Li; Xuelong Li
>
> **摘要:** Spoken language models (SLMs) have advanced rapidly in recent years, accompanied by a growing number of evaluation benchmarks. However, most existing benchmarks emphasize task completion and capability scaling, while remaining poorly aligned with how users interact with SLMs in real-world spoken conversations. Effective spoken interaction requires not only accurate understanding of user intent and content, but also the ability to respond with appropriate interactional strategies. In this paper, we present TELEVAL, a dynamic, user-centered benchmark for evaluating SLMs in realistic Chinese spoken interaction scenarios. TELEVAL consolidates evaluation into two core aspects. Reliable Content Fulfillment assesses whether models can comprehend spoken inputs and produce semantically correct responses. Interactional Appropriateness evaluates whether models act as socially capable interlocutors, requiring them not only to generate human-like, colloquial responses, but also to implicitly incorporate paralinguistic cues for natural interaction. Experiments reveal that, despite strong performance on semantic and knowledge-oriented tasks, current SLMs still struggle to produce natural and interactionally appropriate responses, highlighting the need for more interaction-faithful evaluation.
>
---
#### [replaced 040] SpecDetect: Simple, Fast, and Training-Free Detection of LLM-Generated Text via Spectral Analysis
- **分类: cs.CL**

- **简介: 该论文属于文本检测任务，旨在识别大语言模型生成的文本。通过频域分析方法，利用信号处理技术提升检测效率与准确性。**

- **链接: [https://arxiv.org/pdf/2508.11343v3](https://arxiv.org/pdf/2508.11343v3)**

> **作者:** Haitong Luo; Weiyao Zhang; Suhang Wang; Wenji Zou; Chungang Lin; Xuying Meng; Yujun Zhang
>
> **备注:** AAAI'26 Oral
>
> **摘要:** The proliferation of high-quality text from Large Language Models (LLMs) demands reliable and efficient detection methods. While existing training-free approaches show promise, they often rely on surface-level statistics and overlook fundamental signal properties of the text generation process. In this work, we reframe detection as a signal processing problem, introducing a novel paradigm that analyzes the sequence of token log-probabilities in the frequency domain. By systematically analyzing the signal's spectral properties using the global Discrete Fourier Transform (DFT) and the local Short-Time Fourier Transform (STFT), we find that human-written text consistently exhibits significantly higher spectral energy. This higher energy reflects the larger-amplitude fluctuations inherent in human writing compared to the suppressed dynamics of LLM-generated text. Based on this key insight, we construct SpecDetect, a detector built on a single, robust feature from the global DFT: DFT total energy. We also propose an enhanced version, SpecDetect++, which incorporates a sampling discrepancy mechanism to further boost robustness. Extensive experiments show that our approach outperforms the state-of-the-art model while running in nearly half the time. Our work introduces a new, efficient, and interpretable pathway for LLM-generated text detection, showing that classical signal processing techniques offer a surprisingly powerful solution to this modern challenge.
>
---
#### [replaced 041] Interpreting Transformers Through Attention Head Intervention
- **分类: cs.CL**

- **简介: 该论文属于AI可解释性研究任务，旨在解决Transformer模型决策机制的理解问题。通过注意力头干预，验证机制假设并实现对模型行为的控制。**

- **链接: [https://arxiv.org/pdf/2601.04398v3](https://arxiv.org/pdf/2601.04398v3)**

> **作者:** Mason Kadem; Rong Zheng
>
> **备注:** updated abstract
>
> **摘要:** Neural networks are growing more capable on their own, but we do not understand their neural mechanisms. Understanding these mechanisms' decision-making processes, or mechanistic interpretability, enables (1) accountability and control in high-stakes domains, (2) the study of digital brains and the emergence of cognition, and (3) discovery of new knowledge when AI systems outperform humans. This paper traces how attention head intervention emerged as a key method for causal interpretability of transformers. The evolution from visualization to intervention represents a paradigm shift from observing correlations to causally validating mechanistic hypotheses through direct intervention. Head intervention studies revealed robust empirical findings while also highlighting limitations that complicate interpretation. Recent work demonstrates that mechanistic understanding now enables targeted control of model behaviour, successfully suppressing toxic outputs and manipulating semantic content through selective attention head intervention, validating the practical utility of interpretability research for AI safety.
>
---
#### [replaced 042] Restoring Rhythm: Punctuation Restoration Using Transformer Models for Bangla, A Low-Resource Language
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理中的标点恢复任务，旨在解决低资源语言Bangla文本无标点的问题。通过使用Transformer模型XLM-RoBERTa-large，并利用数据增强，提升了标点恢复效果。**

- **链接: [https://arxiv.org/pdf/2507.18448v2](https://arxiv.org/pdf/2507.18448v2)**

> **作者:** Md Obyedullahil Mamun; Md Adyelullahil Mamun; Arif Ahmad; Md. Imran Hossain Emu
>
> **摘要:** Punctuation restoration enhances the readability of text and is critical for post-processing tasks in Automatic Speech Recognition (ASR), especially for low-resource languages like Bangla. In this study, we explore the application of transformer-based models, specifically XLM-RoBERTa-large, to automatically restore punctuation in unpunctuated Bangla text. We focus on predicting four punctuation marks: period, comma, question mark, and exclamation mark across diverse text domains. To address the scarcity of annotated resources, we constructed a large, varied training corpus and applied data augmentation techniques. Our best-performing model, trained with an augmentation factor of alpha = 0.20%, achieves an accuracy of 97.1% on the News test set, 91.2% on the Reference set, and 90.2% on the ASR set. Results show strong generalization to reference and ASR transcripts, demonstrating the model's effectiveness in real-world, noisy scenarios. This work establishes a strong baseline for Bangla punctuation restoration and contributes publicly available datasets and code to support future research in low-resource NLP.
>
---
#### [replaced 043] OptRot: Mitigating Weight Outliers via Data-Free Rotations for Post-Training Quantization
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型量化任务，旨在解决LLM权重和激活中的异常值问题。通过学习旋转方法减少异常值，提升量化效果。**

- **链接: [https://arxiv.org/pdf/2512.24124v2](https://arxiv.org/pdf/2512.24124v2)**

> **作者:** Advait Gadhikar; Riccardo Grazzi; James Hensman
>
> **备注:** 25 pages, 10 figures
>
> **摘要:** The presence of outliers in Large Language Models (LLMs) weights and activations makes them difficult to quantize. Recent work has leveraged rotations to mitigate these outliers. In this work, we propose methods that learn fusible rotations by minimizing principled and cheap proxy objectives to the weight quantization error. We primarily focus on GPTQ as the quantization method. Our main method is OptRot, which reduces weight outliers simply by minimizing the element-wise fourth power of the rotated weights. We show that OptRot outperforms both Hadamard rotations and more expensive, data-dependent methods like SpinQuant and OSTQuant for weight quantization. It also improves activation quantization in the W4A8 setting. We also propose a data-dependent method, OptRot$^{+}$, that further improves performance by incorporating information on the activation covariance. In the W4A4 setting, we see that both OptRot and OptRot$^{+}$ perform worse, highlighting a trade-off between weight and activation quantization.
>
---
#### [replaced 044] Correcting misinformation on social media with a large language model
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于信息纠错任务，旨在解决社交媒体上的虚假信息问题。工作包括提出MUSE系统，结合大语言模型与多模态信息，实现高效准确的虚假信息检测与解释。**

- **链接: [https://arxiv.org/pdf/2403.11169v5](https://arxiv.org/pdf/2403.11169v5)**

> **作者:** Xinyi Zhou; Ashish Sharma; Amy X. Zhang; Tim Althoff
>
> **备注:** 52 pages
>
> **摘要:** Real-world information, often multimodal, can be misinformed or potentially misleading due to factual errors, outdated claims, missing context, misinterpretation, and more. Such "misinformation" is understudied, challenging to address, and harms many social domains -- particularly on social media, where it can spread rapidly. Manual correction that identifies and explains its (in)accuracies is widely accepted but difficult to scale. While large language models (LLMs) can generate human-like language that could accelerate misinformation correction, they struggle with outdated information, hallucinations, and limited multimodal capabilities. We propose MUSE, an LLM augmented with vision-language modeling and web retrieval over relevant, credible sources to generate responses that determine whether and which part(s) of the given content can be misinformed or potentially misleading, and to explain why with grounded references. We further define a comprehensive set of rubrics to measure response quality, ranging from the accuracy of identifications and factuality of explanations to the relevance and credibility of references. Results show that MUSE consistently produces high-quality outputs across diverse social media content (e.g., modalities, domains, political leanings), including content that has not previously been fact-checked online. Overall, MUSE outperforms GPT-4 by 37% and even high-quality responses from social media users by 29%. Our work provides a general methodological and evaluative framework for correcting misinformation at scale.
>
---
#### [replaced 045] Agent Bain vs. Agent McKinsey: A New Text-to-SQL Benchmark for the Business Domain
- **分类: cs.CL**

- **简介: 该论文提出CORGI基准，解决业务领域中复杂文本到SQL查询的问题。工作包括构建合成数据库、设计四类复杂查询，并评估大模型在其中的表现。**

- **链接: [https://arxiv.org/pdf/2510.07309v3](https://arxiv.org/pdf/2510.07309v3)**

> **作者:** Yue Li; Ran Tao; Derek Hommel; Yusuf Denizay Dönder; Sungyong Chang; David Mimno; Unso Eun Seo Jo
>
> **备注:** 23 pages, under review for ACL ARR
>
> **摘要:** Text-to-SQL benchmarks have traditionally only tested simple data access as a translation task of natural language to SQL queries. But in reality, users tend to ask diverse questions that require more complex responses including data-driven predictions or recommendations. Using the business domain as a motivating example, we introduce CORGI, a new benchmark that expands text-to-SQL to reflect practical database queries encountered by end users. CORGI is composed of synthetic databases inspired by enterprises such as DoorDash, Airbnb, and Lululemon. It provides questions across four increasingly complicated categories of business queries: descriptive, explanatory, predictive, and recommendational. This challenge calls for causal reasoning, temporal forecasting, and strategic recommendation, reflecting multi-level and multi-step agentic intelligence. We find that LLM performance degrades on higher-level questions as question complexity increases. CORGI also introduces and encourages the text-to-SQL community to consider new automatic methods for evaluating open-ended, qualitative responses in data access tasks. Our experiments show that LLMs exhibit an average 33.12% lower success execution rate (SER) on CORGI compared to existing benchmarks such as BIRD, highlighting the substantially higher complexity of real-world business needs. We release the CORGI dataset, an evaluation framework, and a submission website to support future research.
>
---
#### [replaced 046] Rethinking Prompt Optimizers: From Prompt Merits to Optimization
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决提示优化中的兼容性与可解释性问题。通过定义通用提示质量标准，提出MePO优化器，提升不同模型的响应质量。**

- **链接: [https://arxiv.org/pdf/2505.09930v4](https://arxiv.org/pdf/2505.09930v4)**

> **作者:** Zixiao Zhu; Hanzhang Zhou; Zijian Feng; Tianjiao Li; Chua Jia Jim Deryl; Mak Lee Onn; Gee Wah Ng; Kezhi Mao
>
> **备注:** 30 pages, 16 figures
>
> **摘要:** Prompt optimization (PO) provides a practical way to improve response quality when users lack the time or expertise to manually craft effective prompts. Existing methods typically rely on LLMs' self-generation ability to optimize prompts. However, due to limited downward compatibility, the instruction-heavy prompts generated by advanced LLMs can overwhelm lightweight inference models and degrade response quality, while also lacking interpretability due to implicit optimization. In this work, we rethink prompt optimization through the lens of explicit and interpretable design. We first identify a set of model-agnostic prompt quality merits and empirically validate their effectiveness in enhancing prompt and response quality. We then introduce MePO, a merit-guided, locally deployable prompt optimizer trained on our merit-guided prompt preference dataset generated by a lightweight LLM. MePO avoids online optimization, reduces privacy concerns, and, by learning clear, interpretable merits, generalizes effectively to both large-scale and lightweight inference models. Experiments demonstrate that MePO achieves better results across diverse tasks and model types, offering a scalable and robust solution for real-world deployment. The code, model and dataset can be found in https://github.com/MidiyaZhu/MePO
>
---
#### [replaced 047] R-Zero: Self-Evolving Reasoning LLM from Zero Data
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出R-Zero，解决LLM依赖人工数据的问题，通过自生成训练数据实现模型自我进化，提升推理能力。**

- **链接: [https://arxiv.org/pdf/2508.05004v3](https://arxiv.org/pdf/2508.05004v3)**

> **作者:** Chengsong Huang; Wenhao Yu; Xiaoyang Wang; Hongming Zhang; Zongxia Li; Ruosen Li; Jiaxin Huang; Haitao Mi; Dong Yu
>
> **摘要:** Self-evolving Large Language Models (LLMs) offer a scalable path toward super-intelligence by autonomously generating, refining, and learning from their own experiences. However, existing methods for training such models still rely heavily on vast human-curated tasks and labels, typically via fine-tuning or reinforcement learning, which poses a fundamental bottleneck to advancing AI systems toward capabilities beyond human intelligence. To overcome this limitation, we introduce R-Zero, a fully autonomous framework that generates its own training data from scratch. Starting from a single base LLM, R-Zero initializes two independent models with distinct roles, a Challenger and a Solver. These models are optimized separately and co-evolve through interaction: the Challenger is rewarded for proposing tasks near the edge of the Solver capability, and the Solver is rewarded for solving increasingly challenging tasks posed by the Challenger. This process yields a targeted, self-improving curriculum without any pre-existing tasks and labels. Empirically, R-Zero substantially improves reasoning capability across different backbone LLMs, e.g., boosting the Qwen3-4B-Base by +6.49 on math-reasoning benchmarks and +7.54 on general-domain reasoning benchmarks.
>
---
#### [replaced 048] SemCSE-Multi: Multifaceted and Decodable Embeddings for Aspect-Specific and Interpretable Scientific Domain Mapping
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **简介: 该论文提出SemCSE-Multi，用于生成科学摘要的多方面嵌入，解决科学领域映射的细粒度和可解释性问题。通过无监督方法生成特定方面摘要并训练嵌入模型，实现高效、可控的相似性评估与可视化。**

- **链接: [https://arxiv.org/pdf/2510.11599v2](https://arxiv.org/pdf/2510.11599v2)**

> **作者:** Marc Brinner; Sina Zarrieß
>
> **摘要:** We propose SemCSE-Multi, a novel unsupervised framework for generating multifaceted embeddings of scientific abstracts, evaluated in the domains of invasion biology and medicine. These embeddings capture distinct, individually specifiable aspects in isolation, thus enabling fine-grained and controllable similarity assessments as well as adaptive, user-driven visualizations of scientific domains. Our approach relies on an unsupervised procedure that produces aspect-specific summarizing sentences and trains embedding models to map semantically related summaries to nearby positions in the embedding space. We then distill these aspect-specific embedding capabilities into a unified embedding model that directly predicts multiple aspect embeddings from a scientific abstract in a single, efficient forward pass. In addition, we introduce an embedding decoding pipeline that decodes embeddings back into natural language descriptions of their associated aspects. Notably, we show that this decoding remains effective even for unoccupied regions in low-dimensional visualizations, thus offering vastly improved interpretability in user-centric settings.
>
---
#### [replaced 049] Automated Visualization Code Synthesis via Multi-Path Reasoning and Feedback-Driven Optimization
- **分类: cs.SE; cs.AI; cs.CL; cs.HC**

- **简介: 该论文属于可视化代码生成任务，解决用户指令不明确时生成准确可视化代码的问题。提出VisPath框架，通过多路径推理和反馈优化生成高质量可视化结果。**

- **链接: [https://arxiv.org/pdf/2502.11140v3](https://arxiv.org/pdf/2502.11140v3)**

> **作者:** Wonduk Seo; Daye Kang; Hyunjin An; Taehan Kim; Soohyuk Cho; Seungyong Lee; Minhyeong Yu; Jian Park; Yi Bu; Seunghyun Lee
>
> **备注:** 15 pages
>
> **摘要:** Large Language Models (LLMs) have become a cornerstone for automated visualization code generation, enabling users to create charts through natural language instructions. Despite improvements from techniques like few-shot prompting and query expansion, existing methods often struggle when requests are underspecified in actionable details (e.g., data preprocessing assumptions, solver or library choices, etc.), frequently necessitating manual intervention. To overcome these limitations, we propose VisPath: a Multi-Path Reasoning and Feedback-Driven Optimization Framework for Visualization Code Generation. VisPath handles underspecified queries through structured, multi-stage processing. It begins by using Chain-of-Thought (CoT) prompting to reformulate the initial user input, generating multiple extended queries in parallel to surface alternative plausible concretizations of the request. These queries then generate candidate visualization scripts, which are executed to produce diverse images. By assessing the visual quality and correctness of each output, VisPath generates targeted feedback that is aggregated to synthesize an optimal final result. Extensive experiments on MatPlotBench and Qwen-Agent Code Interpreter Benchmark show that VisPath outperforms state-of-the-art methods, providing a more reliable framework for AI-driven visualization generation.
>
---
#### [replaced 050] The Subtle Art of Defection: Understanding Uncooperative Behaviors in LLM based Multi-Agent Systems
- **分类: cs.MA; cs.CL**

- **简介: 该论文属于多智能体系统研究，旨在解决LLM中不合作行为导致系统不稳定的问题。通过构建框架分析并模拟不合作行为，验证其对系统稳定性的影响。**

- **链接: [https://arxiv.org/pdf/2511.15862v2](https://arxiv.org/pdf/2511.15862v2)**

> **作者:** Devang Kulshreshtha; Wanyu Du; Raghav Jain; Srikanth Doss; Hang Su; Sandesh Swamy; Yanjun Qi
>
> **摘要:** This paper introduces a novel framework for simulating and analyzing how uncooperative behaviors can destabilize or collapse LLM-based multi-agent systems. Our framework includes two key components: (1) a game theory-based taxonomy of uncooperative agent behaviors, addressing a notable gap in the existing literature; and (2) a structured, multi-stage simulation pipeline that dynamically generates and refines uncooperative behaviors as agents' states evolve. We evaluate the framework via a collaborative resource management setting, measuring system stability using metrics such as survival time and resource overuse rate. Empirically, our framework achieves 96.7% accuracy in generating realistic uncooperative behaviors, validated by human evaluations. Our results reveal a striking contrast: cooperative agents maintain perfect system stability (100% survival over 12 rounds with 0% resource overuse), while any uncooperative behavior can trigger rapid system collapse within 1 to 7 rounds. We also evaluate LLM-based defense methods, finding they detect some uncooperative behaviors, but some behaviors remain largely undetectable. These gaps highlight how uncooperative agents degrade collective outcomes and underscore the need for more resilient multi-agent systems.
>
---
#### [replaced 051] GlobalRAG: Enhancing Global Reasoning in Multi-hop Question Answering via Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多跳问答任务，旨在解决全局推理不足和执行不忠实的问题。提出GlobalRAG框架，通过强化学习优化多步推理与检索协同。**

- **链接: [https://arxiv.org/pdf/2510.20548v3](https://arxiv.org/pdf/2510.20548v3)**

> **作者:** Jinchang Luo; Mingquan Cheng; Fan Wan; Ni Li; Xiaoling Xia; Shuangshuang Tian; Tingcheng Bian; Haiwei Wang; Haohuan Fu; Yan Tao
>
> **备注:** 8 pages, 3 figures, 4 tables
>
> **摘要:** Reinforcement learning has recently shown promise in improving retrieval-augmented generation (RAG). Despite these advances, its effectiveness in multi-hop question answering (QA) remains limited by two fundamental limitations: (i) global planning absence to structure multi-step reasoning, and (ii) unfaithful execution, which hinders effective query formulation and consistent use of retrieved evidence. We propose GlobalRAG, a reinforcement learning framework designed to enhance global reasoning in multi-hop QA. GlobalRAG decomposes questions into subgoals, coordinates retrieval with reasoning, and refines evidence iteratively. To guide this process, we introduce Planning Quality Reward and SubGoal Completion Reward, which encourage coherent planning and reliable subgoal execution. In addition, a progressive weight annealing strategy balances process-oriented and outcome-based objectives. Extensive experiments on both in-domain and out-of-domain benchmarks demonstrate that GlobalRAG significantly outperforms strong baselines while using only 8k training data (42% of the training data used by strong baselines), achieving average improvements of 14.2% in both EM and F1.
>
---
#### [replaced 052] Pearmut: Human Evaluation of Translation Made Trivial
- **分类: cs.CL; cs.HC**

- **简介: 该论文提出Pearmut，一个简化人类翻译评估的平台，解决人工评估复杂耗时的问题，支持多语言任务，提升模型开发效率。**

- **链接: [https://arxiv.org/pdf/2601.02933v2](https://arxiv.org/pdf/2601.02933v2)**

> **作者:** Vilém Zouhar; Tom Kocmi
>
> **备注:** typeset with Typst
>
> **摘要:** Human evaluation is the gold standard for multilingual NLP, but is often skipped in practice and substituted with automatic metrics, because it is notoriously complex and slow to set up with existing tools with substantial engineering and operational overhead. We introduce Pearmut, a lightweight yet feature-rich platform that makes end-to-end human evaluation as easy to run as automatic evaluation. Pearmut removes common entry barriers and provides support for evaluating multilingual tasks, with a particular focus on machine translation. The platform implements standard evaluation protocols, including DA, ESA, or MQM, but is also extensible to allow prototyping new protocols. It features document-level context, absolute and contrastive evaluation, attention checks, ESAAI pre-annotations and both static and active learning-based assignment strategies. Pearmut enables reliable human evaluation to become a practical, routine component of model development and diagnosis rather than an occasional effort.
>
---
#### [replaced 053] Cross-Domain Transfer and Few-Shot Learning for Personal Identifiable Information Recognition
- **分类: cs.CL**

- **简介: 该论文属于文本匿名化任务，旨在提升PII识别的准确性。研究解决跨领域迁移与少量样本学习问题，通过实验验证不同领域数据的迁移效果及融合优势。**

- **链接: [https://arxiv.org/pdf/2507.11862v3](https://arxiv.org/pdf/2507.11862v3)**

> **作者:** Junhong Ye; Xu Yuan; Xinying Qiu
>
> **备注:** Accepted to CLNLP 2025
>
> **摘要:** Accurate recognition of personally identifiable information (PII) is central to automated text anonymization. This paper investigates the effectiveness of cross-domain model transfer, multi-domain data fusion, and sample-efficient learning for PII recognition. Using annotated corpora from healthcare (I2B2), legal (TAB), and biography (Wikipedia), we evaluate models across four dimensions: in-domain performance, cross-domain transferability, fusion, and few-shot learning. Results show legal-domain data transfers well to biographical texts, while medical domains resist incoming transfer. Fusion benefits are domain-specific, and high-quality recognition is achievable with only 10% of training data in low-specialization domains.
>
---
#### [replaced 054] Large Language Models Develop Novel Social Biases Through Adaptive Exploration
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文属于人工智能伦理任务，旨在解决LLM生成社会偏见的问题。研究发现LLM在决策中会自发产生新偏见，通过激励探索可有效缓解。**

- **链接: [https://arxiv.org/pdf/2511.06148v3](https://arxiv.org/pdf/2511.06148v3)**

> **作者:** Addison J. Wu; Ryan Liu; Xuechunzi Bai; Thomas L. Griffiths
>
> **摘要:** As large language models (LLMs) are adopted into frameworks that grant them the capacity to make real decisions, it is increasingly important to ensure that they are unbiased. In this paper, we argue that the predominant approach of simply removing existing biases from models is not enough. Using a paradigm from the psychology literature, we demonstrate that LLMs can spontaneously develop novel social biases about artificial demographic groups even when no inherent differences exist. These biases result in highly stratified task allocations, which are less fair than assignments by human participants and are exacerbated by newer and larger models. In social science, emergent biases like these have been shown to result from exploration-exploitation trade-offs, where the decision-maker explores too little, allowing early observations to strongly influence impressions about entire demographic groups. To alleviate this effect, we examine a series of interventions targeting model inputs, problem structure, and explicit steering. We find that explicitly incentivizing exploration most robustly reduces stratification, highlighting the need for better multifaceted objectives to mitigate bias. These results reveal that LLMs are not merely passive mirrors of human social biases, but can actively create new ones from experience, raising urgent questions about how these systems will shape societies over time.
>
---
#### [replaced 055] VMMU: A Vietnamese Multitask Multimodal Understanding and Reasoning Benchmark
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出VMMU基准，用于评估视觉语言模型在越南语多模态任务中的理解与推理能力。旨在解决多模态融合与跨语言推理问题，涵盖7个任务，强调真实多模态整合。**

- **链接: [https://arxiv.org/pdf/2508.13680v2](https://arxiv.org/pdf/2508.13680v2)**

> **作者:** Vy Tuong Dang; An Vo; Emilio Villa-Cueva; Quang Tau; Duc Dm; Thamar Solorio; Daeyoung Kim
>
> **摘要:** We introduce VMMU, a Vietnamese Multitask Multimodal Understanding and Reasoning Benchmark designed to evaluate how vision-language models (VLMs) interpret and reason over visual and textual information beyond English. VMMU consists of 2.5k multimodal questions across 7 tasks, covering a diverse range of problem contexts, including STEM problem solving, data interpretation, rule-governed visual reasoning, and abstract visual reasoning. All questions require genuine multimodal integration, rather than reliance on text-only cues or OCR-based shortcuts. We evaluate a diverse set of state-of-the-art proprietary and open-source VLMs on VMMU. Despite strong Vietnamese OCR performance, proprietary models achieve only 66% mean accuracy. Further analysis shows that the primary source of failure is not OCR, but instead multimodal grounding and reasoning over text and visual evidence. Code and data are available at https://vmmu.github.io.
>
---
#### [replaced 056] Choices Speak Louder than Questions
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决MCQA评估中模型依赖选项而非真正理解的问题。提出NPSQ方法，以更准确评估模型 comprehension 能力。**

- **链接: [https://arxiv.org/pdf/2502.18798v4](https://arxiv.org/pdf/2502.18798v4)**

> **作者:** Gyeongje Cho; Yeonkyoung So; Jaejin Lee
>
> **摘要:** Recent findings raise concerns about whether the evaluation of Multiple-Choice Question Answering (MCQA) accurately reflects the comprehension abilities of large language models. This paper explores the concept of choice sensitivity, which refers to the tendency for model decisions to be more influenced by the answer options than by a genuine understanding of the question. We introduce a new scoring method called Normalized Probability Shift by the Question (NPSQ), designed to isolate the impact of the question itself and provide a more reliable assessment of comprehension. Through experiments involving various input formats, including cloze, symbols, and hybrid formats, we find that traditional scoring methods - such as those based on log-likelihood or its length-normalized variant - are vulnerable to superficial characteristics of the answer choices. In contrast, NPSQ remains stable even when modifications are made to the answer options.
>
---
#### [replaced 057] A Unified Understanding and Evaluation of Steering Methods
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型中潜空间控制方法缺乏统一评估的问题。提出统一框架，分析并评估不同方法的有效性。**

- **链接: [https://arxiv.org/pdf/2502.02716v2](https://arxiv.org/pdf/2502.02716v2)**

> **作者:** Shawn Im; Sharon Li
>
> **摘要:** Latent space steering methods provide a practical approach to controlling large language models by applying steering vectors to intermediate activations, guiding outputs toward desired behaviors while avoiding retraining. Despite their growing importance, the field lacks a unified understanding and consistent evaluation across tasks and datasets, hindering progress. This paper introduces a unified framework for analyzing and evaluating steering methods, formalizing their core principles and offering theoretical insights into their effectiveness. Through comprehensive empirical evaluations on multiple-choice and open-ended text generation tasks, we validate these insights, identifying key factors that influence performance and demonstrating the superiority of certain methods. Our work bridges theoretical and practical perspectives, offering actionable guidance for advancing the design, optimization, and deployment of latent space steering methods in LLMs.
>
---
#### [replaced 058] RIGOURATE: Quantifying Scientific Exaggeration with Evidence-Aligned Claim Evaluation
- **分类: cs.CL**

- **简介: 该论文提出RIGOURATE，用于量化科学陈述的夸大程度，解决科学沟通中证据与主张不匹配的问题。通过多模态框架实现证据检索和夸大评分。**

- **链接: [https://arxiv.org/pdf/2601.04350v2](https://arxiv.org/pdf/2601.04350v2)**

> **作者:** Joseph James; Chenghao Xiao; Yucheng Li; Nafise Sadat Moosavi; Chenghua Lin
>
> **摘要:** Scientific rigour tends to be sidelined in favour of bold statements, leading authors to overstate claims beyond what their results support. We present RIGOURATE, a two-stage multimodal framework that retrieves supporting evidence from a paper's body and assigns each claim an overstatement score. The framework consists of a dataset of over 10K claim-evidence sets from ICLR and NeurIPS papers, annotated using eight LLMs, with overstatement scores calibrated using peer-review comments and validated through human evaluation. It employes a fine-tuned reranker for evidence retrieval and a fine-tuned model to predict overstatement scores with justification. Compared to strong baselines, RIGOURATE enables improved evidence retrieval and overstatement detection. Overall, our work operationalises evidential proportionality and supports clearer, more transparent scientific communication.
>
---
#### [replaced 059] AdaSpec: Adaptive Speculative Decoding for Fast, SLO-Aware Large Language Model Serving
- **分类: cs.CL**

- **简介: 该论文属于大语言模型推理优化任务，解决动态负载下的低延迟和SLO达标问题。提出AdaSpec系统，通过自适应推测解码提升性能。**

- **链接: [https://arxiv.org/pdf/2503.05096v2](https://arxiv.org/pdf/2503.05096v2)**

> **作者:** Kaiyu Huang; Hao Wu; Zhubo Shi; Han Zou; Minchen Yu; Qingjiang Shi
>
> **备注:** This paper is accepted by ACM SoCC 2025
>
> **摘要:** Cloud-based Large Language Model (LLM) services often face challenges in achieving low inference latency and meeting Service Level Objectives (SLOs) under dynamic request patterns. Speculative decoding, which exploits lightweight models for drafting and LLMs for verification, has emerged as a compelling technique to accelerate LLM inference. However, existing speculative decoding solutions often fail to adapt to fluctuating workloads and dynamic system environments, resulting in impaired performance and SLO violations. In this paper, we introduce AdaSpec, an efficient LLM inference system that dynamically adjusts speculative strategies according to real-time request loads and system configurations. AdaSpec proposes a theoretical model to analyze and predict the efficiency of speculative strategies across diverse scenarios. Additionally, it implements intelligent drafting and verification algorithms to maximize performance while ensuring high SLO attainment. Experimental results on real-world LLM service traces demonstrate that AdaSpec consistently meets SLOs and achieves substantial performance improvements, delivering up to 66% speedup compared to state-of-the-art speculative inference systems. The source code is publicly available at https://github.com/cerebellumking/AdaSpec
>
---
#### [replaced 060] Credible Plan-Driven RAG Method for Multi-Hop Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多跳问答任务，旨在解决多跳QA中推理稳定性和事实一致性不足的问题。提出PAR-RAG框架，通过复杂度感知规划和双重验证提升性能。**

- **链接: [https://arxiv.org/pdf/2504.16787v3](https://arxiv.org/pdf/2504.16787v3)**

> **作者:** Ningning Zhang; Chi Zhang; Zhizhong Tan; Xingxing Yang; Weiping Deng; Wenyong Wang
>
> **备注:** 24 pages, 7 figures
>
> **摘要:** Retrieval-augmented generation (RAG) has demonstrated strong performance in single-hop question answering (QA) by integrating external knowledge into large language models (LLMs). However, its effectiveness remains limited in multi-hop QA, which demands both stable reasoning and factual consistency. Existing approaches often provide partial solutions, addressing either reasoning trajectory stability or factual verification, but rarely achieving both simultaneously. To bridge this gap, we propose PAR-RAG, a three-stage Plan-then-Act-and-Review framework inspired by the PDCA cycle. PAR-RAG incorporates semantic complexity as a unifying principle through three key components: (i) complexity-aware exemplar selection guides plan generation by aligning decomposition granularity with question difficulty, thereby stabilizing reasoning trajectories; (ii) execution follows a structured retrieve-then-read process; and (iii) dual verification identifies and corrects intermediate errors while dynamically adjusting verification strength based on question complexity: emphasizing accuracy for simple queries and multi-evidence consistency for complex ones. This cognitively inspired framework integrates theoretical grounding with practical robustness. Experiments across diverse benchmarks demonstrate that PAR-RAG consistently outperforms competitive baselines, while ablation studies confirm the complementary roles of complexity-aware planning and dual verification. Collectively, these results establish PAR-RAG as a robust and generalizable framework for reliable multi-hop reasoning.
>
---
#### [replaced 061] MixtureVitae: Open Web-Scale Pretraining Dataset With High Quality Instruction and Reasoning Data Built from Permissive-First Text Sources
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出MixtureVitae，一个低风险的高质量预训练数据集，旨在解决法律风险与模型性能之间的平衡问题。通过整合许可文本和低风险数据，提升模型在多个基准上的表现。任务属于自然语言处理中的预训练数据构建。**

- **链接: [https://arxiv.org/pdf/2509.25531v5](https://arxiv.org/pdf/2509.25531v5)**

> **作者:** Huu Nguyen; Victor May; Harsh Raj; Marianna Nezhurina; Yishan Wang; Yanqi Luo; Minh Chien Vu; Taishi Nakamura; Ken Tsui; Van Khue Nguyen; David Salinas; Aleksandra Krasnodębska; Christoph Schuhmann; Mats Leon Richter; Xuan-Son; Vu; Jenia Jitsev
>
> **备注:** Code: \url{https://github.com/ontocord/mixturevitae}
>
> **摘要:** We present MixtureVitae, an open-access pretraining corpus built to minimize legal risk while providing strong downstream performance. MixtureVitae follows a permissive-first, risk-mitigated sourcing strategy that combines public-domain and permissively licensed text (e.g., CC-BY/Apache) with carefully justified low-risk additions (e.g., government works and EU TDM-eligible sources). MixtureVitae adopts a simple, single-stage pretraining recipe that integrates a large proportion of permissive synthetic instruction and reasoning data-signals typically introduced during post-training and generally scarce in permissive web corpora. We categorize all sources into a three-tier scheme that reflects varying risk levels and provide shard-level provenance metadata to enable risk-aware usage. In controlled experiments using the open-sci-ref training protocol (fixed architectures and hyperparameters; 50B and 300B token budgets across 130M-1.7B parameters), models trained on MixtureVitae consistently outperform other permissive datasets across a suite of standard benchmarks, and at the 1.7B-parameters/300B-tokens setting, they surpass FineWeb-Edu and approach DCLM late in training. Performance is particularly strong on MMLU and on math and code benchmarks: a 1.7B model pretrained on 300B MixtureVitae tokens matches or exceeds a strong 1.7B instruction-tuned baseline on GSM8K, HumanEval, and MBPP, despite using over 36 times fewer tokens (300B vs. ~11T). Supported by a thorough decontamination analysis, these results show that permissive-first data with high instruction and reasoning density, tiered by licensing and provenance-related risk, can provide a practical and risk-mitigated foundation for training capable LLMs, reducing reliance on broad web scrapes without sacrificing competitiveness. Code: https://github.com/ontocord/mixturevitae
>
---
#### [replaced 062] From Alignment to Advancement: Bootstrapping Audio-Language Alignment with Synthetic Data
- **分类: eess.AS; cs.AI; cs.CL; cs.LG; cs.SD**

- **简介: 该论文属于音频-语言对齐任务，旨在解决ALLM的遗忘问题和数据资源消耗大问题。通过合成数据生成框架BALSa，提升模型区分有无声音的能力及多音频理解能力。**

- **链接: [https://arxiv.org/pdf/2505.20166v3](https://arxiv.org/pdf/2505.20166v3)**

> **作者:** Chun-Yi Kuan; Hung-yi Lee
>
> **备注:** Published in IEEE Transactions on Audio, Speech, and Language Processing (TASLP). Project Website: https://kuan2jiu99.github.io/Balsa
>
> **摘要:** Audio-aware large language models (ALLMs) have recently made great strides in understanding and processing audio inputs. These models are typically adapted from text-based large language models (LLMs) through additional training on audio-related tasks. This adaptation process presents two major limitations. First, ALLMs often suffer from catastrophic forgetting, where crucial textual capabilities like instruction-following are lost after training on audio data. In some cases, models may even hallucinate sounds that are not present in the input audio, raising concerns about reliability. Second, achieving cross-modal alignment between audio and language typically relies on large collections of task-specific question-answer pairs for instruction tuning, making it resource-intensive. To address these issues, previous works have leveraged the backbone LLMs to synthesize general-purpose, caption-style alignment data. In this paper, we propose a data generation framework that produces contrastive-like training data, designed to enhance ALLMs' ability to differentiate between present and absent sounds. We further extend our approach to multi-audio scenarios, enabling the model to either explain differences between audio inputs or produce unified captions that describe all inputs, thereby enhancing audio-language alignment. We refer to the entire ALLM training framework as bootstrapping audio-language alignment via synthetic data generation from backbone LLMs (BALSa). Experimental results indicate that our method effectively mitigates audio hallucinations while reliably maintaining strong performance on audio understanding and reasoning benchmarks, as well as instruction-following skills. Moreover, incorporating multi-audio training further enhances the model's comprehension and reasoning capabilities. Overall, BALSa offers an efficient and scalable approach to developing ALLMs.
>
---
#### [replaced 063] DB3 Team's Solution For Meta KDD Cup' 25
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于多模态、多轮问答任务，解决CRAG-MM挑战。团队构建了定制化检索管道和统一的LLM调优方法，有效控制幻觉，取得优异成绩。**

- **链接: [https://arxiv.org/pdf/2509.09681v2](https://arxiv.org/pdf/2509.09681v2)**

> **作者:** Yikuan Xia; Jiazun Chen; Yirui Zhan; Suifeng Zhao; Weipeng Jiang; Chaorui Zhang; Wei Han; Bo Bai; Jun Gao
>
> **摘要:** This paper presents the db3 team's winning solution for the Meta CRAG-MM Challenge 2025 at KDD Cup'25. Addressing the challenge's unique multi-modal, multi-turn question answering benchmark (CRAG-MM), we developed a comprehensive framework that integrates tailored retrieval pipelines for different tasks with a unified LLM-tuning approach for hallucination control. Our solution features (1) domain-specific retrieval pipelines handling image-indexed knowledge graphs, web sources, and multi-turn conversations; and (2) advanced refusal training using SFT, DPO, and RL. The system achieved 2nd place in Task 1, 2nd place in Task 2, and 1st place in Task 3, securing the grand prize for excellence in ego-centric queries through superior handling of first-person perspective challenges.
>
---
#### [replaced 064] NeoAMT: Neologism-Aware Agentic Machine Translation with Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于神经机器翻译任务，旨在解决含新词的句子翻译问题。作者构建了新数据集并提出基于强化学习的框架NeoAMT，提升翻译质量。**

- **链接: [https://arxiv.org/pdf/2601.03790v2](https://arxiv.org/pdf/2601.03790v2)**

> **作者:** Zhongtao Miao; Kaiyan Zhao; Masaaki Nagata; Yoshimasa Tsuruoka
>
> **备注:** Fixed typos in Table 1, Figure 7 and Section 4.2: regex -> exact. Refined the caption of Table 3
>
> **摘要:** Neologism-aware machine translation aims to translate source sentences containing neologisms into target languages. This field remains underexplored compared with general machine translation (MT). In this paper, we propose an agentic framework, NeoAMT, for neologism-aware machine translation using a Wiktionary search tool. Specifically, we first create a new dataset for neologism-aware machine translation and develop a search tool based on Wiktionary. The new dataset covers 16 languages and 75 translation directions and is derived from approximately 10 million records of an English Wiktionary dump. The retrieval corpus of the search tool is also constructed from around 3 million cleaned records of the Wiktionary dump. We then use it for training the translation agent with reinforcement learning (RL) and evaluating the accuracy of neologism-aware machine translation. Based on this, we also propose an RL training framework that contains a novel reward design and an adaptive rollout generation approach by leveraging "translation difficulty" to further improve the translation quality of translation agents using our search tool.
>
---
#### [replaced 065] SPECTRA: Revealing the Full Spectrum of User Preferences via Distributional LLM Inference
- **分类: cs.CL**

- **简介: 该论文提出SPECTRA方法，解决用户偏好建模中的长尾偏差问题。通过分布推断替代生成序列，提升个性化推荐效果。**

- **链接: [https://arxiv.org/pdf/2509.24189v3](https://arxiv.org/pdf/2509.24189v3)**

> **作者:** Luyang Zhang; Jialu Wang; Shichao Zhu; Beibei Li; Zhongcun Wang; Guangmou Pan; Yang Song
>
> **摘要:** Large Language Models (LLMs) are increasingly used to understand user preferences, typically via the direct generation of ranked item lists. However, this end-to-end generative paradigm inherits the bias and opacity of autoregressive decoding, over-emphasizing frequent (head) preferences and obscure long-tail ones, thereby biasing personalization toward head preferences. To address this, we propose SPECTRA (Semantic Preference Extraction and Clustered TRAcking), which treats the LLM as an implicit probabilistic model by probing it to infer a probability distribution over interpretable preference clusters. In doing so, SPECTRA reframes user modeling from sequence generation with decoding heuristics to distributional inference, yielding explicit, cluster-level user preference representations. We evaluate SPECTRA on MovieLens, Yelp, and a large-scale short-video platform, demonstrating significant gains across three dimensions: SPECTRA achieves (i) distributional alignment, reducing Jensen-Shannon divergence to empirical distributions by 25% against strong baselines; (ii) long-tail exposure, reducing decoding-induced head concentration and increasing global exposure entropy by 30%; and (iii) downstream applications such as personalized ranking, translating these gains into a 40% NDCG boost on public datasets and a 7x improvement on ranking long-tail preferences against an industry-leading Transformer-based production baseline.
>
---
#### [replaced 066] The Curious Case of Factual (Mis)Alignment between LLMs' Short- and Long-Form Answers
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型在不同任务复杂度下的事实一致性问题，通过对比短长回答发现模型存在事实错位现象，提出SLAQ框架进行评估。**

- **链接: [https://arxiv.org/pdf/2510.11218v3](https://arxiv.org/pdf/2510.11218v3)**

> **作者:** Saad Obaid ul Islam; Anne Lauscher; Goran Glavaš
>
> **备注:** Code: https://github.com/WorldHellow/SLAQ/tree/main
>
> **摘要:** Large language models (LLMs) can correctly answer "When was Einstein born?" yet fail to provide the same date when writing about Einstein's life revealing a fundamental inconsistency in how models access factual knowledge across task complexities. While models display impressive accuracy on factual question-answering benchmarks, the reliability gap between simple and complex queries remains poorly understood, eroding their trustworthiness. In this work, we introduce Short-Long Form Alignment for Factual Question Answering (SLAQ), a controlled evaluation framework that compares LLMs' answers to the same factual questions asked (a) in isolation (short) vs. (b) integrated into complex queries (long). Looking at 16 LLMs across 600 queries, we find a systematic misalignment of answers to the corresponding short and long queries. We further uncover position-dependent accuracy loss and momentum effects where consecutive correct or incorrect answers create self-reinforcing patterns. Through mechanistic analysis, we find that aligned facts activate overlapping model internals, and that metrics based on mechanistic similarity can predict short-long answer alignment with up to 78% accuracy. Our work establishes factual consistency over query complexity as an important aspect of LLMs' trustworthiness and challenges current evaluation practices, which implicitly assume that good performance for simple factual queries implies reliability in more complex knowledge-seeking tasks too.
>
---
#### [replaced 067] Rep3Net: An Approach Exploiting Multimodal Representation for Molecular Bioactivity Prediction
- **分类: cs.LG; cs.CL; q-bio.QM**

- **简介: 该论文属于分子生物活性预测任务，旨在提升化合物效力预测的准确性。通过融合多种分子表示方法，提出Rep3Net模型，有效提升预测性能。**

- **链接: [https://arxiv.org/pdf/2512.00521v2](https://arxiv.org/pdf/2512.00521v2)**

> **作者:** Sabrina Islam; Md. Atiqur Rahman; Md. Bakhtiar Hasan; Md. Hasanul Kabir
>
> **摘要:** Accurate prediction of compound potency accelerates early-stage drug discovery by prioritizing candidates for experimental testing. However, many Quantitative Structure-Activity Relationship (QSAR) approaches for this prediction are constrained by their choice of molecular representation: handcrafted descriptors capture global properties but miss local topology, graph neural networks encode structure but often lack broader chemical context, and SMILES-based language models provide contextual patterns learned from large corpora but are seldom combined with structural features. To exploit these complementary signals, we introduce Rep3Net, a unified multimodal architecture that fuses RDKit molecular descriptors, graph-derived features from a residual graph-convolutional backbone, and ChemBERTa SMILES embeddings. We evaluate Rep3Net on a curated ChEMBL subset for Human PARP1 using fivefold cross validation. Rep3Net attains an MSE of $0.83\pm0.06$, RMSE of $0.91\pm0.03$, $R^{2}=0.43\pm0.01$, and yields Pearson and Spearman correlations of $0.66\pm0.01$ and $0.67\pm0.01$, respectively, substantially improving over several strong GNN baselines. In addition, Rep3Net achieves a favorable latency-to-parameter trade-off thanks to a single-layer GCN backbone and parallel frozen encoders. Ablations show that graph topology, ChemBERTa semantics, and handcrafted descriptors each contribute complementary information, with full fusion providing the largest error reduction. These results demonstrate that multimodal representation fusion can improve potency prediction for PARP1 and provide a scalable framework for virtual screening in early-stage drug discovery.
>
---
#### [replaced 068] Topology Matters: Measuring Memory Leakage in Multi-Agent LLMs
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于隐私安全任务，研究多智能体大语言模型中的记忆泄露问题。通过设计MAMA框架，分析不同网络拓扑对泄露的影响，提出系统设计建议。**

- **链接: [https://arxiv.org/pdf/2512.04668v3](https://arxiv.org/pdf/2512.04668v3)**

> **作者:** Jinbo Liu; Defu Cao; Yifei Wei; Tianyao Su; Yuan Liang; Yushun Dong; Yan Liu; Yue Zhao; Xiyang Hu
>
> **摘要:** Graph topology is a fundamental determinant of memory leakage in multi-agent LLM systems, yet its effects remain poorly quantified. We introduce MAMA (Multi-Agent Memory Attack), a framework that measures how network structure shapes leakage. MAMA operates on synthetic documents containing labeled Personally Identifiable Information (PII) entities, from which we generate sanitized task instructions. We execute a two-phase protocol: Engram (seeding private information into a target agent's memory) and Resonance (multi-round interaction where an attacker attempts extraction). Over 10 rounds, we measure leakage as exact-match recovery of ground-truth PII from attacker outputs. We evaluate six canonical topologies (complete, ring, chain, tree, star, star-ring) across $n\in\{4,5,6\}$, attacker-target placements, and base models. Results are consistent: denser connectivity, shorter attacker-target distance, and higher target centrality increase leakage; most leakage occurs in early rounds and then plateaus; model choice shifts absolute rates but preserves topology ordering; spatiotemporal/location attributes leak more readily than identity credentials or regulated identifiers. We distill practical guidance for system design: favor sparse or hierarchical connectivity, maximize attacker-target separation, and restrict hub/shortcut pathways via topology-aware access control.
>
---
#### [replaced 069] Evaluation of the Automated Labeling Method for Taxonomic Nomenclature Through Prompt-Optimized Large Language Model
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决物种名称自动标注问题。通过优化提示的大型语言模型进行分类，提升标注效率与准确性。**

- **链接: [https://arxiv.org/pdf/2503.10662v2](https://arxiv.org/pdf/2503.10662v2)**

> **作者:** Keito Inoshita; Kota Nojiri; Haruto Sugeno; Takumi Taga
>
> **备注:** This paper was accepted by IEEE IAICT
>
> **摘要:** Scientific names of organisms consist of a genus name and a species epithet, with the latter often reflecting aspects such as morphology, ecology, distribution, and cultural background. Traditionally, researchers have manually labeled species names by carefully examining taxonomic descriptions, a process that demands substantial time and effort when dealing with large datasets. This study evaluates the feasibility of automatic species name labeling using large language model (LLM) by leveraging their text classification and semantic extraction capabilities. Using the spider name dataset compiled by Mammola et al., we compared LLM-based labeling results-enhanced through prompt engineering-with human annotations. The results indicate that LLM-based classification achieved high accuracy in Morphology, Geography, and People categories. However, classification accuracy was lower in Ecology & Behavior and Modern & Past Culture, revealing challenges in interpreting animal behavior and cultural contexts. Future research will focus on improving accuracy through optimized few-shot learning and retrieval-augmented generation techniques, while also expanding the applicability of LLM-based labeling to diverse biological taxa.
>
---
#### [replaced 070] Your LLM Agents are Temporally Blind: The Misalignment Between Tool Use Decisions and Human Time Perception
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，研究LLM代理在时间感知上的不足，即“时间盲视”。提出TicToc数据集，分析模型与人类时间感知的对齐问题，并探索改进方法。**

- **链接: [https://arxiv.org/pdf/2510.23853v2](https://arxiv.org/pdf/2510.23853v2)**

> **作者:** Yize Cheng; Arshia Soltani Moakhar; Chenrui Fan; Parsa Hosseini; Kazem Faghih; Zahra Sodagar; Wenxiao Wang; Soheil Feizi
>
> **摘要:** Large language model (LLM) agents are increasingly used to interact with and execute tasks in dynamic environments. However, a critical yet overlooked limitation of these agents is that they, by default, assume a stationary context, failing to account for the real-world time elapsed between messages. We refer to this as "temporal blindness". This limitation hinders decisions about when to invoke tools, leading agents to either over-rely on stale context and skip needed tool calls, or under-rely on it and redundantly repeat tool calls. To study this challenge, we constructed TicToc, a diverse dataset of multi-turn user-agent message trajectories across 76 scenarios, spanning dynamic environments with high, medium, and low time sensitivity. We collected human preferences between "calling a tool" and "directly answering" on each sample, and evaluated how well LLM tool-calling decisions align with human preferences under varying amounts of elapsed time. Our analysis reveals that existing models display poor alignment with human temporal perception, with no model achieving a normalized alignment rate better than 65% when given time stamp information. We also show that naive, prompt-based alignment techniques have limited effectiveness for most models, but specific post-training alignment can be a viable way to align multi-turn LLM tool use with human temporal perception. Our data and findings provide a first step toward understanding and mitigating temporal blindness, offering insights to foster the development of more time-aware and human-aligned agents.
>
---
#### [replaced 071] Flexible Realignment of Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型对齐任务，解决模型性能不达预期的问题。提出TrRa和InRa框架，实现训练和推理阶段的灵活对齐控制。**

- **链接: [https://arxiv.org/pdf/2506.12704v2](https://arxiv.org/pdf/2506.12704v2)**

> **作者:** Wenhong Zhu; Ruobing Xie; Weinan Zhang; Rui Wang
>
> **摘要:** Realignment becomes necessary when a language model (LM) fails to meet expected performance. We propose a flexible realignment framework that supports quantitative control of alignment degree during training and inference. This framework incorporates Training-time Realignment (TrRa), which efficiently realigns the reference model by leveraging the controllable fusion of logits from both the reference and already aligned models. For example, TrRa reduces token usage by 54.63% on DeepSeek-R1-Distill-Qwen-1.5B without any performance degradation, outperforming DeepScaleR-1.5B's 33.86%. To complement TrRa during inference, we introduce a layer adapter that enables smooth Inference-time Realignment (InRa). This adapter is initialized to perform an identity transformation at the bottom layer and is inserted preceding the original layers. During inference, input embeddings are simultaneously processed by the adapter and the original layer, followed by the remaining layers, and then controllably interpolated at the logit level. We upgraded DeepSeek-R1-Distill-Qwen-7B from a slow-thinking model to one that supports both fast and slow thinking, allowing flexible alignment control even during inference. By encouraging deeper reasoning, it even surpassed its original performance.
>
---
#### [replaced 072] SPEC-RL: Accelerating On-Policy Reinforcement Learning with Speculative Rollouts
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出SPEC-RL，解决强化学习中rollout阶段计算效率低的问题，通过重用历史轨迹提升训练速度。**

- **链接: [https://arxiv.org/pdf/2509.23232v3](https://arxiv.org/pdf/2509.23232v3)**

> **作者:** Bingshuai Liu; Ante Wang; Zijun Min; Liang Yao; Haibo Zhang; Yang Liu; Xu Han; Peng Li; Anxiang Zeng; Jinsong Su
>
> **备注:** fixed typos
>
> **摘要:** Large Language Models (LLMs) increasingly rely on reinforcement learning with verifiable rewards (RLVR) to elicit reliable chain-of-thought reasoning. However, the training process remains bottlenecked by the computationally expensive rollout stage. Existing acceleration methods-such as parallelization, objective- and data-driven modifications, and replay buffers-either incur diminishing returns, introduce bias, or overlook redundancy across iterations. We identify that rollouts from consecutive training epochs frequently share a large portion of overlapping segments, wasting computation. To address this, we propose SPEC-RL, a novel framework that integrates SPECulative decoding with the RL rollout process. SPEC-RL reuses prior trajectory segments as speculative prefixes and extends them via a draft-and-verify mechanism, avoiding redundant generation while ensuring policy consistency. Experiments on diverse math reasoning and generalization benchmarks, including AIME24, MATH-500, OlympiadBench, MMLU-STEM, and others, demonstrate that SPEC-RL reduces rollout time by 2-3x without compromising policy quality. As a purely rollout-stage enhancement, SPEC-RL integrates seamlessly with mainstream algorithms (e.g., PPO, GRPO, DAPO), offering a general and practical path to scale RLVR for large reasoning models. Our code is available at https://github.com/ShopeeLLM/Spec-RL
>
---
#### [replaced 073] IndicParam: Benchmark to evaluate LLMs on low-resource Indic Languages
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在评估大语言模型在低资源印地语系语言上的表现。作者构建了IndicParam基准，涵盖11种语言，测试模型的多方面能力。**

- **链接: [https://arxiv.org/pdf/2512.00333v2](https://arxiv.org/pdf/2512.00333v2)**

> **作者:** Ayush Maheshwari; Kaushal Sharma; Vivek Patel; Aditya Maheshwari
>
> **摘要:** While large language models excel on high-resource multilingual tasks, low- and extremely low-resource Indic languages remain severely under-evaluated. We present IndicParam, a human-curated benchmark of over 13,000 multiple-choice questions covering 11 such languages (Nepali, Gujarati, Marathi, Odia as low-resource; Dogri, Maithili, Rajasthani, Sanskrit, Bodo, Santali, Konkani as extremely low-resource) plus Sanskrit-English code-mixed set. We evaluated 20 LLMs, both proprietary and open-weights, which reveals that even the top-performing \texttt{Gemini-2.5} reaches 58\% average accuracy, followed by \texttt{GPT-5} (45) and \texttt{DeepSeek-3.2} (43.1). We additionally label each question as knowledge-oriented or purely linguistic to discriminate factual recall from grammatical proficiency. Further, we assess the ability of LLMs to handle diverse question formats-such as list-based matching, assertion-reason pairs, and sequence ordering-alongside conventional multiple-choice questions. \benchmark\ provides insights into limitations of cross-lingual transfer and establishes a challenging benchmark for Indic languages. The dataset is available at https://huggingface.co/datasets/bharatgenai/IndicParam. Scripts to run benchmark are present at https://github.com/ayushbits/IndicParam.
>
---
#### [replaced 074] Revisiting Entropy in Reinforcement Learning for Large Reasoning Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于强化学习任务，解决LLM在RLVR训练中熵崩溃问题。通过分析熵动态，提出Positive-Advantage Reweighting方法有效调控熵，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2511.05993v2](https://arxiv.org/pdf/2511.05993v2)**

> **作者:** Renren Jin; Pengzhi Gao; Yuqi Ren; Zhuowen Han; Tongxuan Zhang; Wuwei Huang; Wei Liu; Jian Luan; Deyi Xiong
>
> **备注:** 22 pages, 25 figures, 5 tables
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) has emerged as a prominent paradigm for enhancing the reasoning capabilities of large language models (LLMs). However, the entropy of LLMs usually collapses during RLVR training, leading to premature convergence to suboptimal local minima and hindering further performance improvement. Although various approaches have been proposed to mitigate entropy collapse, a comprehensive study of entropy in RLVR remains lacking. To bridge this gap, we conduct extensive experiments to investigate the entropy dynamics of LLMs trained with RLVR and analyze how model entropy correlates with response diversity, calibration, and performance across various benchmarks. Our results identify three key factors that influence entropy: the clipping thresholds in the optimization objective, the number of off-policy updates, and the diversity of the training data. Furthermore, through both theoretical analysis and empirical validation, we demonstrate that tokens with positive advantages are the primary drivers of entropy collapse. Motivated by this insight, we propose Positive-Advantage Reweighting, a simple yet effective approach that regulates model entropy by adjusting the loss weights assigned to tokens with positive advantages during RLVR training, while maintaining competitive performance.
>
---
#### [replaced 075] From Implicit to Explicit: Enhancing Self-Recognition in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自识别任务，旨在解决大语言模型在个体呈现模式下自识别能力差的问题。通过分析隐式自识别原因，提出认知手术框架提升识别性能。**

- **链接: [https://arxiv.org/pdf/2508.14408v2](https://arxiv.org/pdf/2508.14408v2)**

> **作者:** Yinghan Zhou; Weifeng Zhu; Juan Wen; Wanli Peng; Zhengxian Wu; Yiming Xue
>
> **摘要:** Large language models (LLMs) have been shown to possess a degree of self-recognition ability, which used to identify whether a given text was generated by themselves. Prior work has demonstrated that this capability is reliably expressed under the pair presentation paradigm (PPP), where the model is presented with two texts and asked to choose which one it authored. However, performance deteriorates sharply under the individual presentation paradigm (IPP), where the model is given a single text to judge authorship. Although this phenomenon has been observed, its underlying causes have not been systematically analyzed. In this paper, we first investigate the cause of this failure and attribute it to implicit self-recognition (ISR). ISR describes the gap between internal representations and output behavior in LLMs: under the IPP scenario, the model encodes self-recognition information in its feature space, yet its ability to recognize self-generated texts remains poor. To mitigate the ISR of LLMs, we propose cognitive surgery (CoSur), a novel framework comprising four main modules: representation extraction, subspace construction, authorship discrimination, and cognitive editing. Experimental results demonstrate that our proposed method improves the self-recognition performance of three different LLMs in the IPP scenario, achieving average accuracies of 99.00%, 97.69%, and 97.13%, respectively.
>
---
#### [replaced 076] OpenRubrics: Towards Scalable Synthetic Rubric Generation for Reward Modeling and LLM Alignment
- **分类: cs.CL**

- **简介: 该论文属于奖励建模任务，旨在解决传统奖励模型无法全面反映人类偏好的问题。通过构建大规模(提示, 评分标准)对和对比评分生成方法，提升奖励模型性能。**

- **链接: [https://arxiv.org/pdf/2510.07743v2](https://arxiv.org/pdf/2510.07743v2)**

> **作者:** Tianci Liu; Ran Xu; Tony Yu; Ilgee Hong; Carl Yang; Tuo Zhao; Haoyu Wang
>
> **备注:** The first two authors contributed equally. Updated OpenRubrics dataset, RMs, and results
>
> **摘要:** Reward modeling lies at the core of reinforcement learning from human feedback (RLHF), yet most existing reward models rely on scalar or pairwise judgments that fail to capture the multifaceted nature of human preferences. Recent studies have explored rubrics-as-rewards (RaR) that uses structured criteria to capture multiple dimensions of response quality. However, producing rubrics that are both reliable and scalable remains a key challenge. In this work, we introduce OpenRubrics, a diverse, large-scale collection of (prompt, rubric) pairs for training rubric-generation and rubric-based reward models. To elicit discriminative and comprehensive evaluation signals, we introduce Contrastive Rubric Generation (CRG), which derives both hard rules (explicit constraints) and principles (implicit qualities) by contrasting preferred and rejected responses. We further remove noisy rubrics via preserving preference-label consistency. Across multiple reward-modeling benchmarks, our rubric-based reward model, Rubric-RM, surpasses strong size-matched baselines by 8.4%. These gains transfer to policy models on instruction-following and biomedical benchmarks.
>
---
#### [replaced 077] Word length predicts word order: "Min-max"-ing drives language evolution
- **分类: cs.CL**

- **简介: 该论文研究语言演变中的语序变化，提出基于“最小化努力、最大化信息”的理论解释，通过分析1942个语言语料库数据，揭示词类长度与语序的关系。任务为语言演化分析，解决语序变化原因问题。**

- **链接: [https://arxiv.org/pdf/2505.13913v2](https://arxiv.org/pdf/2505.13913v2)**

> **作者:** Hiram Ring
>
> **摘要:** A fundamental concern in linguistics has been to understand how languages change, such as in relation to word order. Since the order of words in a sentence (i.e. the relative placement of Subject, Object, and Verb) is readily identifiable in most languages, this has been a productive field of study for decades (see Greenberg 1963; Dryer 2007; Hawkins 2014). However, a language's word order can change over time, with competing explanations for such changes (Carnie and Guilfoyle 2000; Crisma and Longobardi 2009; Martins and Cardoso 2018; Dunn et al. 2011; Jager and Wahle 2021). This paper proposes a general universal explanation for word order change based on a theory of communicative interaction (the Min-Max theory of language behavior) in which agents seek to minimize effort while maximizing information. Such an account unifies opposing findings from language processing (Piantadosi et al. 2011; Wasow 2022; Levy 2008) that make different predictions about how word order should be realized crosslinguistically. The marriage of both "efficiency" and "surprisal" approaches under the Min-Max theory is justified with evidence from a massive dataset of 1,942 language corpora tagged for parts of speech (Ring 2025), in which average lengths of particular word classes correlates with word order, allowing for prediction of basic word order from diverse corpora. The general universal pressure of word class length in corpora is shown to give a stronger explanation for word order realization than either genealogical or areal factors, highlighting the importance of language corpora for investigating such questions.
>
---
#### [replaced 078] Put the Space of LoRA Initialization to the Extreme to Preserve Pre-trained Knowledge
- **分类: cs.CL**

- **简介: 该论文属于大语言模型微调任务，旨在解决LoRA方法中的灾难性遗忘问题。通过将LoRA初始化置于输入激活的零空间，有效保留预训练知识。**

- **链接: [https://arxiv.org/pdf/2503.02659v2](https://arxiv.org/pdf/2503.02659v2)**

> **作者:** Pengwei Tang; Xiaolin Hu; Yong Liu; Lizhong Ding; Dongjie Zhang; Xing Wu; Debing Zhang
>
> **备注:** Accepted at AAAI 2026. We rediscovered why our approach works from the perspective of the LoRA initialization space. Accordingly, we added new experiments and also removed inappropriate experiments (those without catastrophic forgetting)
>
> **摘要:** Low-Rank Adaptation (LoRA) is the leading parameter-efficient fine-tuning method for Large Language Models (LLMs), but it still suffers from catastrophic forgetting. Recent work has shown that specialized LoRA initialization can alleviate catastrophic forgetting. There are currently two approaches to LoRA initialization aimed at preventing knowledge forgetting during fine-tuning: (1) making residual weights close to pre-trained weights, and (2) ensuring the space of LoRA initialization is orthogonal to pre-trained knowledge. The former is what current methods strive to achieve, while the importance of the latter is not sufficiently recognized. We find that the space of LoRA initialization is the key to preserving pre-trained knowledge rather than the residual weights. Existing methods like MiLoRA propose making the LoRA initialization space orthogonal to pre-trained weights. However, MiLoRA utilizes the null space of pre-trained weights. Compared to pre-trained weights, the input activations of pre-trained knowledge take into account the parameters of all previous layers as well as the input data, while pre-trained weights only contain information from the current layer. Moreover, we find that the effective ranks of input activations are much smaller than those of pre-trained weights. Thus, the null space of activations is more accurate and contains less pre-trained knowledge information compared to that of weights. Based on these, we introduce LoRA-Null, our proposed method that initializes LoRA in the null space of activations. Experimental results show that LoRA-Null effectively preserves the pre-trained world knowledge of LLMs while achieving good fine-tuning performance, as evidenced by extensive experiments. Code is available at {https://github.com/HungerPWAY/LoRA-Null}.
>
---
#### [replaced 079] Berezinskii--Kosterlitz--Thouless transition in a context-sensitive random language model
- **分类: stat.ML; cond-mat.stat-mech; cs.CL; cs.LG**

- **简介: 该论文属于语言模型与物理相变交叉研究，旨在探索语言模型中的相变现象。通过构建上下文敏感语言模型，发现其表现出BKT相变，揭示语言结构与物理临界现象的联系。**

- **链接: [https://arxiv.org/pdf/2412.01212v2](https://arxiv.org/pdf/2412.01212v2)**

> **作者:** Yuma Toji; Jun Takahashi; Vwani Roychowdhury; Hideyuki Miyahara
>
> **备注:** accepted for publication in PRE
>
> **摘要:** Several power-law critical properties involving different statistics in natural languages -- reminiscent of scaling properties of physical systems at or near phase transitions -- have been documented for decades. The recent rise of large language models has added further evidence and excitement by providing intriguing similarities with notions in physics such as scaling laws and emergent abilities. However, specific instances of classes of generative language models that exhibit phase transitions, as understood by the statistical physics community, are lacking. In this work, inspired by the one-dimensional Potts model in statistical physics, we construct a simple probabilistic language model that falls under the class of context-sensitive grammars, which we call the context-sensitive random language model, and numerically demonstrate an unambiguous phase transition in the framework of a natural language model. We explicitly show that a precisely defined order parameter -- that captures symbol frequency biases in the sentences generated by the language model -- changes from strictly zero to a strictly nonzero value (in the infinite-length limit of sentences), implying a mathematical singularity arising when tuning the parameter of the stochastic language model we consider. Furthermore, we identify the phase transition as a variant of the Berezinskii--Kosterlitz--Thouless (BKT) transition, which is known to exhibit critical properties not only at the transition point but also in the entire phase. This finding leads to the possibility that critical properties in natural languages may not require careful fine-tuning nor self-organized criticality, but are generically explained by the underlying connection between language structures and the BKT phases.
>
---
#### [replaced 080] KG-MuLQA: A Framework for KG-based Multi-Level QA Extraction and Long-Context LLM Evaluation
- **分类: cs.CL**

- **简介: 该论文提出KG-MuLQA框架，用于知识图谱驱动的多层级问答抽取与长文本大模型评估，解决复杂问答任务中的多跳推理和集合操作问题。**

- **链接: [https://arxiv.org/pdf/2505.12495v2](https://arxiv.org/pdf/2505.12495v2)**

> **作者:** Nikita Tatarinov; Vidhyakshaya Kannan; Haricharana Srinivasa; Arnav Raj; Harpreet Singh Anand; Varun Singh; Aditya Luthra; Ravij Lade; Agam Shah; Sudheer Chava
>
> **摘要:** We introduce KG-MuLQA (Knowledge-Graph-based Multi-Level Question-Answer Extraction): a framework that (1) extracts QA pairs at multiple complexity levels (2) along three key dimensions -- multi-hop retrieval, set operations, and answer plurality, (3) by leveraging knowledge-graph-based document representations. This approach enables fine-grained assessment of model performance across controlled difficulty levels. Using this framework, we construct a dataset of 20,139 QA pairs based on financial credit agreements and evaluate 16 proprietary and open-weight Large Language Models, observing that even the best-performing models struggle with set-based comparisons and multi-hop reasoning over long contexts. Our analysis reveals systematic failure modes tied to semantic misinterpretation and inability to handle implicit relations.
>
---
#### [replaced 081] SAEMark: Steering Personalized Multilingual LLM Watermarks with Sparse Autoencoders
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出SAEMark，解决LLM生成文本水印问题，通过特征拒绝采样实现多语言、高质量水印，无需修改模型或训练。**

- **链接: [https://arxiv.org/pdf/2508.08211v2](https://arxiv.org/pdf/2508.08211v2)**

> **作者:** Zhuohao Yu; Xingru Jiang; Weizheng Gu; Yidong Wang; Qingsong Wen; Shikun Zhang; Wei Ye
>
> **备注:** 24 pages, 12 figures, NeurIPS 2025, code available: https://zhuohaoyu.github.io/SAEMark
>
> **摘要:** Watermarking LLM-generated text is critical for content attribution and misinformation prevention. However, existing methods compromise text quality, require white-box model access and logit manipulation. These limitations exclude API-based models and multilingual scenarios. We propose SAEMark, a general framework for post-hoc multi-bit watermarking that embeds personalized messages solely via inference-time, feature-based rejection sampling without altering model logits or requiring training. Our approach operates on deterministic features extracted from generated text, selecting outputs whose feature statistics align with key-derived targets. This framework naturally generalizes across languages and domains while preserving text quality through sampling LLM outputs instead of modifying. We provide theoretical guarantees relating watermark success probability and compute budget that hold for any suitable feature extractor. Empirically, we demonstrate the framework's effectiveness using Sparse Autoencoders (SAEs), achieving superior detection accuracy and text quality. Experiments across 4 datasets show SAEMark's consistent performance, with 99.7% F1 on English and strong multi-bit detection accuracy. SAEMark establishes a new paradigm for scalable watermarking that works out-of-the-box with closed-source LLMs while enabling content attribution.
>
---
#### [replaced 082] Less is More: Improving LLM Reasoning with Minimal Test-Time Intervention
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型推理任务，旨在提升模型推理准确性与效率。提出MTI框架，通过最小化测试时干预，仅在高不确定性位置进行优化，显著提升性能。**

- **链接: [https://arxiv.org/pdf/2510.13940v3](https://arxiv.org/pdf/2510.13940v3)**

> **作者:** Zhen Yang; Mingyang Zhang; Feng Chen; Ganggui Ding; Liang Hou; Xin Tao; Ying-Cong Chen
>
> **备注:** Code: https://github.com/EnVision-Research/MTI
>
> **摘要:** Recent progress in large language models (LLMs) has focused on test-time scaling to improve reasoning via increased inference computation, but often at the cost of efficiency. We revisit test-time behavior and uncover a simple yet underexplored phenomenon: reasoning uncertainty is highly localized-only a small subset of high-entropy tokens dominantly affects output correctness. Motivated by this, we propose Minimal Test-Time Intervention (MTI), a training-free framework that enhances reasoning accuracy and stability with minimal overhead. MTI includes: (i) Selective CFG intervention, applying classifier-free guidance only at uncertain positions; and (ii) Lightweight negative-prompt guidance, reusing the main model's KV cache to approximate unconditional decoding efficiently. MTI yields consistent gains across general, coding, and STEM tasks-e.g., +9.28% average improvement on six benchmarks for DeepSeek-R1-7B and +11.25% on AIME2024 using Ling-mini-2.0-while remaining highly efficient.
>
---
#### [replaced 083] LLMs Enable Bag-of-Texts Representations for Short-Text Clustering
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于短文本聚类任务，解决无监督场景下聚类问题。提出一种无需优化嵌入的方法，利用LLM直接生成文本袋表示，提升聚类效果。**

- **链接: [https://arxiv.org/pdf/2510.06747v2](https://arxiv.org/pdf/2510.06747v2)**

> **作者:** I-Fan Lin; Faegheh Hasibi; Suzan Verberne
>
> **摘要:** In this paper, we propose a training-free method for unsupervised short text clustering that relies less on careful selection of embedders than other methods. In customer-facing chatbots, companies are dealing with large amounts of user utterances that need to be clustered according to their intent. In these settings, no labeled data is typically available, and the number of clusters is not known. Recent approaches to short-text clustering in label-free settings incorporate LLM output to refine existing embeddings. While LLMs can identify similar texts effectively, the resulting similarities may not be directly represented by distances in the dense vector space, as they depend on the original embedding. We therefore propose a method for transforming LLM judgments directly into a bag-of-texts representation in which texts are initialized to be equidistant, without assuming any prior distance relationships. Our method achieves comparable or superior results to state-of-the-art methods, but without embeddings optimization or assuming prior knowledge of clusters or labels. Experiments on diverse datasets and smaller LLMs show that our method is model agnostic and can be applied to any embedder, with relatively small LLMs, and different clustering methods. We also show how our method scales to large datasets, reducing the computational cost of the LLM use. The flexibility and scalability of our method make it more aligned with real-world training-free scenarios than existing clustering methods.
>
---
#### [replaced 084] DR-LoRA: Dynamic Rank LoRA for Mixture-of-Experts Adaptation
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于模型微调任务，解决MoE模型中LoRA参数分配不均的问题。通过动态调整专家LoRA秩，提升参数利用效率与任务性能。**

- **链接: [https://arxiv.org/pdf/2601.04823v2](https://arxiv.org/pdf/2601.04823v2)**

> **作者:** Guanzhi Deng; Bo Li; Ronghao Chen; Huacan Wang; Linqi Song; Lijie Wen
>
> **摘要:** Mixture-of-Experts (MoE) has become a prominent paradigm for scaling Large Language Models (LLMs). Parameter-efficient fine-tuning (PEFT), such as LoRA, is widely adopted to adapt pretrained MoE LLMs to downstream tasks. However, existing approaches assign identical LoRA ranks to all experts, overlooking the intrinsic functional specialization within MoE LLMs. This uniform allocation leads to resource mismatch, task-relevant experts are under-provisioned while less relevant ones receive redundant parameters. We propose a Dynamic Rank LoRA framework named DR-LoRA, which dynamically grows expert LoRA ranks during fine-tuning based on task-specific demands. DR-LoRA employs an Expert Saliency Scoring mechanism that integrates expert routing frequency and LoRA rank importance to quantify each expert's demand for additional capacity. Experts with higher saliency scores are prioritized for rank expansion, enabling the automatic formation of a heterogeneous rank distribution tailored to the target task. Experiments on multiple benchmarks demonstrate that DR-LoRA consistently outperforms standard LoRA and static allocation strategies under the same parameter budget, achieving superior task performance with more efficient parameter utilization.
>
---
#### [replaced 085] LongEmotion: Measuring Emotional Intelligence of Large Language Models in Long-Context Interaction
- **分类: cs.CL**

- **简介: 该论文属于情感智能评估任务，旨在解决长文本中情感理解与生成的评价缺失问题。提出LongEmotion基准和CoEM框架，提升模型在长上下文中的情感表现。**

- **链接: [https://arxiv.org/pdf/2509.07403v2](https://arxiv.org/pdf/2509.07403v2)**

> **作者:** Weichu Liu; Jing Xiong; Yuxuan Hu; Zixuan Li; Minghuan Tan; Ningning Mao; Hui Shen; Wendong Xu; Chaofan Tao; Min Yang; Chengming Li; Lingpeng Kong; Ngai Wong
>
> **备注:** Technical Report
>
> **摘要:** Large language models (LLMs) have made significant progress in Emotional Intelligence (EI) and long-context modeling. However, existing benchmarks often overlook the fact that emotional information processing unfolds as a continuous long-context process. To address the absence of multidimensional EI evaluation in long-context inference and explore model performance under more challenging conditions, we present LongEmotion, a benchmark that encompasses a diverse suite of tasks targeting the assessment of models' capabilities in Emotion Recognition, Knowledge Application, and Empathetic Generation, with an average context length of 15,341 tokens. To enhance performance under realistic constraints, we introduce the Collaborative Emotional Modeling (CoEM) framework, which integrates Retrieval-Augmented Generation (RAG) and multi-agent collaboration to improve models' EI in long-context scenarios. We conduct a detailed analysis of various models in long-context settings, investigating how reasoning mode activation, RAG-based retrieval strategies, and context-length adaptability influence their EI performance. Our project page is: https://longemotion.github.io/
>
---
#### [replaced 086] Test-Time Scaling of Reasoning Models for Machine Translation
- **分类: cs.CL**

- **简介: 该论文研究测试时缩放（TTS）在机器翻译中的效果，探讨增加推理时间是否提升翻译质量。工作包括评估12个模型在不同场景下的表现，发现TTS在特定任务中有效，但在通用模型中效果有限。**

- **链接: [https://arxiv.org/pdf/2510.06471v2](https://arxiv.org/pdf/2510.06471v2)**

> **作者:** Zihao Li; Shaoxiong Ji; Jörg Tiedemann
>
> **摘要:** Test-time scaling (TTS) has enhanced the performance of Reasoning Models (RMs) on various tasks such as math and coding, yet its efficacy in machine translation (MT) remains underexplored. This paper investigates whether increased inference-time computation improves translation quality. We evaluate 12 RMs across a diverse suite of MT benchmarks spanning multiple domains, examining three scenarios: direct translation, forced-reasoning extrapolation, and post-editing. Our findings show that for general-purpose RMs, TTS provides limited and inconsistent benefits for direct translation, with performance quickly plateauing. However, the effectiveness of TTS is unlocked by domain-specific fine-tuning, which aligns a model's reasoning process with task requirements, leading to consistent improvements up to an optimal, self-determined reasoning depth. We also find that forcing a model to reason beyond its natural stopping point consistently degrades translation quality. In contrast, TTS proves highly effective in a post-editing context, reliably turning self-correction into a beneficial process. These results indicate that the value of inference-time computation in MT lies not in enhancing single-pass translation with general models, but in targeted applications like multi-step, self-correction workflows and in conjunction with task-specialized models.
>
---
#### [replaced 087] BnMMLU: Measuring Massive Multitask Language Understanding in Bengali
- **分类: cs.CL**

- **简介: 该论文提出BnMMLU，一个用于评估孟加拉语多任务语言理解的基准。针对孟加拉语资源不足的问题，构建了大规模多领域数据集，并测试多种模型表现。**

- **链接: [https://arxiv.org/pdf/2505.18951v2](https://arxiv.org/pdf/2505.18951v2)**

> **作者:** Saman Sarker Joy; Swakkhar Shatabda
>
> **备注:** 19 Pages, 10 Tables, 12 Figures
>
> **摘要:** Large-scale multitask benchmarks have driven rapid progress in language modeling, yet most emphasize high-resource languages such as English, leaving Bengali underrepresented. We present BnMMLU, a comprehensive benchmark for measuring massive multitask language understanding in Bengali. BnMMLU spans 41 domains across STEM, humanities, social sciences, and general knowledge, and contains 134,375 multiple-choice question-option pairs--the most extensive Bengali evaluation suite to date. The dataset preserves mathematical content via MathML, and includes BnMMLU-HARD, a compact subset constructed from questions most frequently missed by top systems to stress difficult cases. We benchmark 24 model variants across 11 LLM families, spanning open-weights general/multilingual, Bengali-centric open-weights, and proprietary models, covering multiple parameter scales and instruction-tuned settings. We evaluate models under standardized protocols covering two prompting styles (Direct vs. Chain-of-Thought) and two context regimes (0-shot vs. 5-shot), reporting accuracy consistently across families. Our analysis highlights persistent gaps in reasoning and application skills and indicates sublinear returns to scale across model sizes. We release the dataset and evaluation templates to support rigorous, reproducible assessment of Bengali language understanding and to catalyze progress in multilingual NLP.
>
---
#### [replaced 088] MuDRiC: Multi-Dialect Reasoning for Arabic Commonsense Validation
- **分类: cs.CL**

- **简介: 该论文属于阿拉伯语常识验证任务，旨在解决阿拉伯语方言资源不足的问题。作者构建了首个多方言常识数据集MuDRiC，并提出基于图卷积网络的方法提升常识推理效果。**

- **链接: [https://arxiv.org/pdf/2508.13130v2](https://arxiv.org/pdf/2508.13130v2)**

> **作者:** Kareem Elozeiri; Mervat Abassy; Preslav Nakov; Yuxia Wang
>
> **摘要:** Commonsense validation evaluates whether a sentence aligns with everyday human understanding, a critical capability for developing robust natural language understanding systems. While substantial progress has been made in English, the task remains underexplored in Arabic, particularly given its rich linguistic diversity. Existing Arabic resources have primarily focused on Modern Standard Arabic (MSA), leaving regional dialects underrepresented despite their prevalence in spoken contexts. To bridge this gap, we present two key contributions. We introduce MuDRiC, an extended Arabic commonsense dataset incorporating multiple dialects. To the best of our knowledge, this is the first Arabic multi-dialect commonsense reasoning dataset. We further propose a novel method adapting Graph Convolutional Networks (GCNs) to Arabic commonsense reasoning, which enhances semantic relationship modeling for improved commonsense validation. Our experimental results demonstrate that this approach consistently outperforms the baseline of direct language model fine-tuning. Overall, our work enhances Arabic natural language understanding by providing a foundational dataset and a new method for handling its complex variations. Data and code are available at https://github.com/KareemElozeiri/MuDRiC.
>
---
#### [replaced 089] Compressed code: the hidden effects of quantization and distillation on programming tokens
- **分类: cs.SE; cs.CL; cs.LG; cs.PL**

- **简介: 该论文研究压缩模型中量化与蒸馏对编程令牌的影响，旨在提升代码生成质量。属于自然语言处理中的代码生成任务，通过分析令牌表示和优化技术解决压缩模型性能下降问题。**

- **链接: [https://arxiv.org/pdf/2601.02563v2](https://arxiv.org/pdf/2601.02563v2)**

> **作者:** Viacheslav Siniaev; Iaroslav Chelombitko; Aleksey Komissarov
>
> **备注:** 18 pages, 1 figure and 6 tables
>
> **摘要:** Large Language Models (LLMs) have demonstrated exceptional code generation capabilities, yet their token-level mechanisms remain underexplored, particularly in compressed models. Through systematic analysis of programming language token representations, we characterize how programming languages are encoded in LLM tokenizers by analyzing their vocabulary distribution and keyword coverage patterns. We introduce a novel cold-start probability analysis method that provides insights into model behavior without requiring explicit prompts. Additionally, we present a comprehensive evaluation of how different model optimization techniques - including quantization, distillation, model scaling, and task-specific fine-tuning - affect token-level representations and code generation quality. Our experiments, supported by comprehensive probability distribution analysis and evaluation metrics, reveal critical insights into token-level behavior and provide empirically-validated guidelines for maintaining code generation quality under various optimization constraints. These findings advance both theoretical understanding of LLM code generation and practical implementation of optimized models in production environments.
>
---
#### [replaced 090] Learning to Evolve: Bayesian-Guided Continual Knowledge Graph Embedding
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于知识图谱嵌入任务，解决动态数据演化中的持续学习问题。提出BAKE框架，通过贝叶斯推理和聚类约束，有效防止遗忘并保持语义一致性。**

- **链接: [https://arxiv.org/pdf/2508.02426v2](https://arxiv.org/pdf/2508.02426v2)**

> **作者:** Linyu Li; Zhi Jin; Yuanpeng He; Dongming Jin; Yichi Zhang; Haoran Duan; Xuan Zhang; Zhengwei Tao; Nyima Tash
>
> **摘要:** As social media and the World Wide Web become hubs for information dissemination, effectively organizing and understanding the vast amounts of dynamically evolving Web content is crucial. Knowledge graphs (KGs) provide a powerful framework for structuring this information. However, the rapid emergence of new hot topics, user relationships, and events in social media renders traditional static knowledge graph embedding (KGE) models rapidly outdated. Continual Knowledge Graph Embedding (CKGE) aims to address this issue, but existing methods commonly suffer from catastrophic forgetting, whereby older, but still valuable, information is lost when learning new knowledge (such as new memes or trending events). This means the model cannot effectively learn the evolution of the data. We propose a novel CKGE framework, BAKE. Unlike existing methods, BAKE formulates CKGE as a sequential Bayesian inference problem and utilizes the Bayesian posterior update principle as a natural continual learning strategy. This principle is insensitive to data order and provides theoretical guarantees to preserve prior knowledge as much as possible. Specifically, we treat each batch of new data as a Bayesian update to the model's prior. By maintaining the posterior distribution, the model effectively preserves earlier knowledge even as it evolves over multiple snapshots. Furthermore, to constrain the evolution of knowledge across snapshots, we introduce a continual clustering method that maintains the compact cluster structure of entity embeddings through a regularization term, ensuring semantic consistency while allowing controlled adaptation to new knowledge. We conduct extensive experiments on multiple CKGE benchmarks, which demonstrate that BAKE achieves the top performance in the vast majority of cases compared to existing approaches.
>
---
#### [replaced 091] Efficient Continual Pre-training for Building Domain Specific Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于领域专用大语言模型构建任务，旨在解决如何高效训练领域模型的问题。通过持续预训练方法，在金融领域提升模型性能，同时保持通用任务表现。**

- **链接: [https://arxiv.org/pdf/2311.08545v2](https://arxiv.org/pdf/2311.08545v2)**

> **作者:** Yong Xie; Karan Aggarwal; Aitzaz Ahmad
>
> **备注:** ACL 2024: https://aclanthology.org/2024.findings-acl.606/
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable open-domain capabilities. LLMs tailored for a domain are typically trained entirely on domain corpus to excel at handling domain-specific tasks. In this work, we explore an alternative strategy of continual pre-training as a means to develop domain-specific LLMs over an existing open-domain LLM. We introduce FinPythia-6.9B, developed through domain-adaptive continual pre-training on the financial domain. Continual pre-trained FinPythia showcases consistent improvements on financial tasks over the original foundational model. We further explore simple but effective data selection strategies for continual pre-training. Our data selection strategies outperform vanilla continual pre-training's performance with just 10% of corpus size and cost, without any degradation on open-domain standard tasks. Our work proposes an alternative solution to building domain-specific LLMs cost-effectively.
>
---
#### [replaced 092] ContextFocus: Activation Steering for Contextual Faithfulness in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在面对外部信息冲突时输出不忠实的问题。提出ContextFocus方法，在不微调模型的情况下提升上下文忠实性。**

- **链接: [https://arxiv.org/pdf/2601.04131v2](https://arxiv.org/pdf/2601.04131v2)**

> **作者:** Nikhil Anand; Shwetha Somasundaram; Anirudh Phukan; Apoorv Saxena; Koyel Mukherjee
>
> **摘要:** Large Language Models (LLMs) encode vast amounts of parametric knowledge during pre-training. As world knowledge evolves, effective deployment increasingly depends on their ability to faithfully follow externally retrieved context. When such evidence conflicts with the model's internal knowledge, LLMs often default to memorized facts, producing unfaithful outputs. In this work, we introduce ContextFocus, a lightweight activation steering approach that improves context faithfulness in such knowledge-conflict settings while preserving fluency and efficiency. Unlike prior approaches, our solution requires no model finetuning and incurs minimal inference-time overhead, making it highly efficient. We evaluate ContextFocus on the ConFiQA benchmark, comparing it against strong baselines including ContextDPO, COIECD, and prompting-based methods. Furthermore, we show that our method is complementary to prompting strategies and remains effective on larger models. Extensive experiments show that ContextFocus significantly improves contextual-faithfulness. Our results highlight the effectiveness, robustness, and efficiency of ContextFocus in improving contextual-faithfulness of LLM outputs.
>
---
#### [replaced 093] Metaphors are a Source of Cross-Domain Misalignment of Large Reasoning Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理领域，研究Metaphors对大模型推理跨域对齐的影响。工作包括发现隐喻与模型推理偏差的因果关系，并设计检测器预测偏差内容。**

- **链接: [https://arxiv.org/pdf/2601.03388v2](https://arxiv.org/pdf/2601.03388v2)**

> **作者:** Zhibo Hu; Chen Wang; Yanfeng Shu; Hye-young Paik; Liming Zhu
>
> **备注:** 17 pages, 7 figures
>
> **摘要:** Earlier research has shown that metaphors influence human's decision making, which raises the question of whether metaphors also influence large language models (LLMs)' reasoning pathways, considering their training data contain a large number of metaphors. In this work, we investigate the problem in the scope of the emergent misalignment problem where LLMs can generalize patterns learned from misaligned content in one domain to another domain. We discover a strong causal relationship between metaphors in training data and the misalignment degree of LLMs' reasoning contents. With interventions using metaphors in pre-training, fine-tuning and re-alignment phases, models' cross-domain misalignment degrees change significantly. As we delve deeper into the causes behind this phenomenon, we observe that there is a connection between metaphors and the activation of global and local latent features of large reasoning models. By monitoring these latent features, we design a detector that predict misaligned content with high accuracy.
>
---
#### [replaced 094] Expanding before Inferring: Enhancing Factuality in Large Language Models through Premature Layers Interpolation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型的幻觉问题。通过插入提前层进行插值，提升事实一致性，无需训练即可应用。**

- **链接: [https://arxiv.org/pdf/2506.02973v2](https://arxiv.org/pdf/2506.02973v2)**

> **作者:** Dingwei Chen; Ziqiang Liu; Feiteng Fang; Chak Tou Leong; Shiwen Ni; Ahmadreza Argha; Hamid Alinejad-Rokny; Min Yang; Chengming Li
>
> **摘要:** Large Language Models (LLMs) demonstrate remarkable capabilities in text understanding and generation. However, their tendency to produce factually inconsistent outputs, commonly referred to as ''hallucinations'', remains a critical challenge. Existing approaches, such as retrieval-based and inference-time correction methods, primarily address this issue at the input or output level, often overlooking the intrinsic information refinement process and the role of premature layers. Meanwhile, alignment- and fine-tuning-based methods are resource-intensive. In this paper, we propose PLI (Premature Layers Interpolation), a novel, training-free, and plug-and-play intervention designed to enhance factuality. PLI mitigates hallucinations by inserting premature layers formed through mathematical interpolation with adjacent layers. Inspired by stable diffusion and sampling steps, PLI extends the depth of information processing and transmission in LLMs, improving factual coherence. Experiments on four publicly available datasets demonstrate that PLI effectively reduces hallucinations while outperforming existing baselines in most cases. Further analysis suggests that the success of layer interpolation is closely linked to LLMs' internal mechanisms. Our dataset and code are available at https://github.com/CuSO4-Chen/PLI.
>
---
#### [replaced 095] A Scalable Unsupervised Framework for multi-aspect labeling of Multilingual and Multi-Domain Review Data
- **分类: cs.CL**

- **简介: 该论文属于多领域、多语言评论的多方面标注任务，旨在解决传统方法依赖标注数据和单一领域的问题。提出一种可扩展的无监督框架，通过聚类和嵌入表示生成高质量标签。**

- **链接: [https://arxiv.org/pdf/2505.09286v2](https://arxiv.org/pdf/2505.09286v2)**

> **作者:** Jiin Park; Misuk Kim
>
> **备注:** 36 pages, 10 figures. Published in Knowledge-Based Systems
>
> **摘要:** Effectively analyzing online review data is essential across industries. However, many existing studies are limited to specific domains and languages or depend on supervised learning approaches that require large-scale labeled datasets. To address these limitations, we propose a multilingual, scalable, and unsupervised framework for cross-domain aspect detection. This framework is designed for multi-aspect labeling of multilingual and multi-domain review data. In this study, we apply automatic labeling to Korean and English review datasets spanning various domains and assess the quality of the generated labels through extensive experiments. Aspect category candidates are first extracted through clustering, and each review is then represented as an aspect-aware embedding vector using negative sampling. To evaluate the framework, we conduct multi-aspect labeling and fine-tune several pretrained language models to measure the effectiveness of the automatically generated labels. Results show that these models achieve high performance, demonstrating that the labels are suitable for training. Furthermore, comparisons with publicly available large language models highlight the framework's superior consistency and scalability when processing large-scale data. A human evaluation also confirms that the quality of the automatic labels is comparable to those created manually. This study demonstrates the potential of a robust multi-aspect labeling approach that overcomes limitations of supervised methods and is adaptable to multilingual, multi-domain environments. Future research will explore automatic review summarization and the integration of artificial intelligence agents to further improve the efficiency and depth of review analysis.
>
---
#### [replaced 096] BackdoorAgent: A Unified Framework for Backdoor Attacks on LLM-based Agents
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于安全领域，研究LLM代理中的后门攻击问题。提出BackdoorAgent框架，分析不同阶段的后门触发传播，构建基准测试，揭示代理流程的脆弱性。**

- **链接: [https://arxiv.org/pdf/2601.04566v2](https://arxiv.org/pdf/2601.04566v2)**

> **作者:** Yunhao Feng; Yige Li; Yutao Wu; Yingshui Tan; Yanming Guo; Yifan Ding; Kun Zhai; Xingjun Ma; Yu-Gang Jiang
>
> **摘要:** Large language model (LLM) agents execute tasks through multi-step workflows that combine planning, memory, and tool use. While this design enables autonomy, it also expands the attack surface for backdoor threats. Backdoor triggers injected into specific stages of an agent workflow can persist through multiple intermediate states and adversely influence downstream outputs. However, existing studies remain fragmented and typically analyze individual attack vectors in isolation, leaving the cross-stage interaction and propagation of backdoor triggers poorly understood from an agent-centric perspective. To fill this gap, we propose \textbf{BackdoorAgent}, a modular and stage-aware framework that provides a unified, agent-centric view of backdoor threats in LLM agents. BackdoorAgent structures the attack surface into three functional stages of agentic workflows, including \textbf{planning attacks}, \textbf{memory attacks}, and \textbf{tool-use attacks}, and instruments agent execution to enable systematic analysis of trigger activation and propagation across different stages. Building on this framework, we construct a standardized benchmark spanning four representative agent applications: \textbf{Agent QA}, \textbf{Agent Code}, \textbf{Agent Web}, and \textbf{Agent Drive}, covering both language-only and multimodal settings. Our empirical analysis shows that \textit{triggers implanted at a single stage can persist across multiple steps and propagate through intermediate states.} For instance, when using a GPT-based backbone, we observe trigger persistence in 43.58\% of planning attacks, 77.97\% of memory attacks, and 60.28\% of tool-stage attacks, highlighting the vulnerabilities of the agentic workflow itself to backdoor threats. To facilitate reproducibility and future research, our code and benchmark are publicly available at GitHub.
>
---
#### [replaced 097] Bridging Symbolic Control and Neural Reasoning in LLM Agents: The Structured Cognitive Loop
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出SCL架构，解决LLM代理的推理与执行纠缠、记忆不稳等问题，通过模块化设计提升可控性与可解释性。属于AI代理可靠性研究。**

- **链接: [https://arxiv.org/pdf/2511.17673v3](https://arxiv.org/pdf/2511.17673v3)**

> **作者:** Myung Ho Kim
>
> **备注:** The reference list has been updated to reflect recent work
>
> **摘要:** Large language model agents suffer from fundamental architectural problems: entangled reasoning and execution, memory volatility, and uncontrolled action sequences. We introduce Structured Cognitive Loop (SCL), a modular architecture that explicitly separates agent cognition into five phases: Retrieval, Cognition, Control, Action, and Memory (R-CCAM). At the core of SCL is Soft Symbolic Control, an adaptive governance mechanism that applies symbolic constraints to probabilistic inference, preserving neural flexibility while restoring the explainability and controllability of classical symbolic systems. Through empirical validation on multi-step conditional reasoning tasks, we demonstrate that SCL achieves zero policy violations, eliminates redundant tool calls, and maintains complete decision traceability. These results address critical gaps in existing frameworks such as ReAct, AutoGPT, and memory-augmented approaches. Our contributions are threefold: (1) we situate SCL within the taxonomy of hybrid intelligence, differentiating it from prompt-centric and memory-only approaches; (2) we formally define Soft Symbolic Control and contrast it with neuro-symbolic AI; and (3) we derive three design principles for trustworthy agents: modular decomposition, adaptive symbolic governance, and transparent state management. We provide a complete open-source implementation demonstrating the R-CCAM loop architecture, alongside a live GPT-4o-powered travel planning agent. By connecting expert system principles with modern LLM capabilities, this work offers a practical and theoretically grounded path toward reliable, explainable, and governable AI agents.
>
---
#### [replaced 098] Do MLLMs Capture How Interfaces Guide User Behavior? A Benchmark for Multimodal UI/UX Design Understanding
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于UI/UX设计理解任务，旨在解决MLLMs对界面如何影响用户行为理解不足的问题。工作中构建了WiserUI-Bench基准，并进行有效性预测与解释实验。**

- **链接: [https://arxiv.org/pdf/2505.05026v4](https://arxiv.org/pdf/2505.05026v4)**

> **作者:** Jaehyun Jeon; Min Soo Kim; Jang Han Yoon; Sumin Shim; Yejin Choi; Hanbin Kim; Dae Hyun Kim; Youngjae Yu
>
> **备注:** 25 pages, 24 figures, Our code and dataset: https://github.com/jeochris/wiserui-bench
>
> **摘要:** User interface (UI) design goes beyond visuals to shape user experience (UX), underscoring the shift toward UI/UX as a unified concept. While recent studies have explored UI evaluation using Multimodal Large Language Models (MLLMs), they largely focus on surface-level features, overlooking how design choices influence user behavior at scale. To fill this gap, we introduce WiserUI-Bench, a novel benchmark for multimodal understanding of how UI/UX design affects user behavior, built on 300 real-world UI image pairs from industry A/B tests, with empirically validated winners that induced more user actions. For future design progress in practice, post-hoc understanding of why such winners succeed with mass users is also required; we support this via expert-curated key interpretations for each instance. Experiments across multiple MLLMs on WiserUI-Bench for two main tasks, (1) predicting the more effective UI image between an A/B-tested pair, and (2) explaining it post-hoc in alignment with expert interpretations, show that models exhibit limited understanding of the behavioral impact of UI/UX design. We believe our work will foster research on leveraging MLLMs for visual design in user behavior contexts.
>
---
#### [replaced 099] Deep Value Benchmark: Measuring Whether Models Generalize Deep Values or Shallow Preferences
- **分类: cs.AI; cs.CL; cs.CY**

- **简介: 该论文提出Deep Value Benchmark，用于评估大模型是否学习深层价值而非浅层偏好，解决AI对齐问题。通过设计实验测试模型在打破关联后的价值泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.02109v3](https://arxiv.org/pdf/2511.02109v3)**

> **作者:** Joshua Ashkinaze; Hua Shen; Saipranav Avula; Eric Gilbert; Ceren Budak
>
> **备注:** NeurIPS 2025 (Spotlight)
>
> **摘要:** We introduce the Deep Value Benchmark (DVB), an evaluation framework that directly tests whether large language models (LLMs) learn fundamental human values or merely surface-level preferences. This distinction is critical for AI alignment: Systems that capture deeper values are likely to generalize human intentions robustly, while those that capture only superficial patterns in preference data risk producing misaligned behavior. The DVB uses a novel experimental design with controlled confounding between deep values (e.g., moral principles) and shallow features (e.g., superficial attributes). In the training phase, we expose LLMs to human preference data with deliberately correlated deep and shallow features -- for instance, where a user consistently prefers (non-maleficence, formal language) options over (justice, informal language) alternatives. The testing phase then breaks these correlations, presenting choices between (justice, formal language) and (non-maleficence, informal language) options. This design allows us to precisely measure a model's Deep Value Generalization Rate (DVGR) -- the probability of generalizing based on the underlying value rather than the shallow feature. Across 9 different models, the average DVGR is just 0.30. All models generalize deep values less than chance. Larger models have a (slightly) lower DVGR than smaller models. We are releasing our dataset, which was subject to three separate human validation experiments. DVB provides an interpretable measure of a core feature of alignment.
>
---
#### [replaced 100] Empirical Analysis of Decoding Biases in Masked Diffusion Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究Masked Diffusion Models的注意力机制，揭示其动态注意力现象，解决MDMs内部机制不明确的问题，提出浅层结构感知、深层内容聚焦的注意力机制。**

- **链接: [https://arxiv.org/pdf/2508.13021v3](https://arxiv.org/pdf/2508.13021v3)**

> **作者:** Pengcheng Huang; Tianming Liu; Zhenghao Liu; Yukun Yan; Shuo Wang; Tong Xiao; Zulong Chen; Maosong Sun
>
> **备注:** 22 pages,17 figures
>
> **摘要:** Masked diffusion models (MDMs), which leverage bidirectional attention and a denoising process, are narrowing the performance gap with autoregressive models (ARMs). However, their internal attention mechanisms remain under-explored. This paper investigates the attention behaviors in MDMs, revealing the phenomenon of Attention Floating. Unlike ARMs, where attention converges to a fixed sink, MDMs exhibit dynamic, dispersed attention anchors that shift across denoising steps and layers. Further analysis reveals its Shallow Structure-Aware, Deep Content-Focused attention mechanism: shallow layers utilize floating tokens to build a global structural framework, while deeper layers allocate more capability toward capturing semantic content. Empirically, this distinctive attention pattern provides a mechanistic explanation for the strong in-context learning capabilities of MDMs, allowing them to double the performance compared to ARMs in knowledge-intensive tasks. All codes are available at https://github.com/NEUIR/Uncode.
>
---
#### [replaced 101] LogitSpec: Accelerating Retrieval-based Speculative Decoding via Next Next Token Speculation
- **分类: cs.CL**

- **简介: 该论文属于大语言模型推理加速任务，旨在解决检索式推测解码中草案令牌不准确的问题。提出LogitSpec通过预测下一个和下下一个词来扩展检索范围，提升解码效率。**

- **链接: [https://arxiv.org/pdf/2507.01449v2](https://arxiv.org/pdf/2507.01449v2)**

> **作者:** Tianyu Liu; Qitan Lv; Hao Li; Xing Gao; Xiao Sun; Xiaoyan Sun
>
> **摘要:** Speculative decoding (SD), where a small draft model is employed to propose draft tokens in advance and then the target model validates them in parallel, has emerged as a promising technique for LLM inference acceleration. Many endeavors to improve SD are to eliminate the need for a draft model and generate draft tokens in a retrieval-based manner in order to further alleviate the drafting overhead and significantly reduce the difficulty in deployment and applications. However, retrieval-based SD relies on a matching paradigm to retrieval the most relevant reference as the draft tokens, where these methods often fail to find matched and accurate draft tokens. To address this challenge, we propose LogitSpec to effectively expand the retrieval range and find the most relevant reference as drafts. Our LogitSpec is motivated by the observation that the logit of the last token can not only predict the next token, but also speculate the next next token. Specifically, LogitSpec generates draft tokens in two steps: (1) utilizing the last logit to speculate the next next token; (2) retrieving relevant reference for both the next token and the next next token. LogitSpec is training-free and plug-and-play, which can be easily integrated into existing LLM inference frameworks. Extensive experiments on a wide range of text generation benchmarks demonstrate that LogitSpec can achieve up to 2.61 $\times$ speedup and 3.28 mean accepted tokens per decoding step. Our code is available at https://github.com/smart-lty/LogitSpec.
>
---
