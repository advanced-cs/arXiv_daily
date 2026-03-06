# 自然语言处理 cs.CL

- **最新发布 115 篇**

- **更新 56 篇**

## 最新发布

#### [new 001] C2-Faith: Benchmarking LLM Judges for Causal and Coverage Faithfulness in Chain-of-Thought Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于评估任务，解决LLM judges在因果和覆盖忠实性上的可靠性问题。构建C2-Faith基准，测试模型在错误检测、定位和覆盖率判断中的表现。**

- **链接: [https://arxiv.org/pdf/2603.05167](https://arxiv.org/pdf/2603.05167)**

> **作者:** Avni Mittal; Rauno Arike
>
> **摘要:** Large language models (LLMs) are increasingly used as judges of chain-of-thought (CoT) reasoning, but it remains unclear whether they can reliably assess process faithfulness rather than just answer plausibility. We introduce C2-Faith, a benchmark built from PRM800K that targets two complementary dimensions of faithfulness: causality (does each step logically follow from prior context?) and coverage (are essential intermediate inferences present?). Using controlled perturbations, we create examples with known causal error positions by replacing a single step with an acausal variant, and with controlled coverage deletions at varying deletion rates (scored against reference labels). We evaluate three frontier judges under three tasks: binary causal detection, causal step localization, and coverage scoring. The results show that model rankings depend strongly on task framing, with no single judge dominating all settings; all judges exhibit a substantial gap between detecting an error and localizing it; and coverage judgments are systematically inflated for incomplete reasoning. These findings clarify when LLM judges are dependable and where they fail, and provide practical guidance for selecting judges in process-level evaluation
>
---
#### [new 002] Measuring the Redundancy of Decoder Layers in SpeechLLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语音大语言模型解码器的冗余性，旨在减少模型参数需求。通过剪枝实验发现解码器存在大量冗余，可显著减少层数而不影响性能，适用于多种语音任务。**

- **链接: [https://arxiv.org/pdf/2603.05121](https://arxiv.org/pdf/2603.05121)**

> **作者:** Adel Moumen; Guangzhi Sun; Philip C Woodland
>
> **摘要:** Speech Large Language Models route speech encoder representations into an LLM decoder that typically accounts for over 90% of total parameters. We study how much of this decoder capacity is actually needed for speech tasks. Across two LLM families and three scales (1-8B), we show that decoder redundancy is largely inherited from the pretrained LLM: text and speech inputs yield similar redundant blocks. We then measure excess capacity by pruning decoder layers and analysing post-pruning healing to increase robustness. Our findings show that 7-8B models retain good ASR performance with only 60% of decoder layers, and the same trend extends to smaller scales with reduced pruning tolerance. We then generalise to speech translation, and show that the same blocks of layers are redundant across speech encoders, tasks and languages, indicating that a more global redundancy structure exists, enabling a single pruned and multi-tasks SpeechLLM backbone to be deployed.
>
---
#### [new 003] MUTEX: Leveraging Multilingual Transformers and Conditional Random Fields for Enhanced Urdu Toxic Span Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于 Urdu 有毒片段检测任务，旨在解决现有系统无法精确定位有毒文本的问题。提出 MUTEX 框架，结合多语言 Transformer 和 CRF，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2603.05057](https://arxiv.org/pdf/2603.05057)**

> **作者:** Inayat Arshad; Fajar Saleem; Ijaz Hussain
>
> **备注:** 29 pages, 7 figures, 13 tables
>
> **摘要:** Urdu toxic span detection remains limited because most existing systems rely on sentence-level classification and fail to identify the specific toxic spans within those text. It is further exacerbated by the multiple factors i.e. lack of token-level annotated resources, linguistic complexity of Urdu, frequent code-switching, informal expressions, and rich morphological variations. In this research, we propose MUTEX: a multilingual transformer combined with conditional random fields (CRF) for Urdu toxic span detection framework that uses manually annotated token-level toxic span dataset to improve performance and interpretability. MUTEX uses XLM RoBERTa with CRF layer to perform sequence labeling and is tested on multi-domain data extracted from social media, online news, and YouTube reviews using token-level F1 to evaluate fine-grained span detection. The results indicate that MUTEX achieves 60% token-level F1 score that is the first supervised baseline for Urdu toxic span detection. Further examination reveals that transformer-based models are more effective at implicitly capturing the contextual toxicity and are able to address the issues of code-switching and morphological variation than other models.
>
---
#### [new 004] Same Input, Different Scores: A Multi Model Study on the Inconsistency of LLM Judge
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，研究LLM作为评分器时对相同输入给出不同分数的问题，分析模型稳定性、差异性及温度影响，提出监控与混合评估建议。**

- **链接: [https://arxiv.org/pdf/2603.04417](https://arxiv.org/pdf/2603.04417)**

> **作者:** Fiona Lau
>
> **备注:** 19 pages, 14 figures
>
> **摘要:** Large language models are increasingly used as automated evaluators in research and enterprise settings, a practice known as LLM-as-a-judge. While prior work has examined accuracy, bias, and alignment with human preferences, far less attention has been given to how consistently LLMs assign numerical scores, an important concern for many production workflows. This study systematically evaluates scoring stability across five commonly used models, GPT-4o, GPT-4o-mini, Gemini-2.5-Flash, Claude-Haiku-4.5, and Claude-Sonnet-4.5, two temperature settings, and real enterprise question-answer pairs drawn from a retrieval-augmented generation (RAG) system. We address three questions: how stable a model's scores are across repeated runs, how differently models score identical inputs, and how temperature affects scoring consistency. Temperature controls the determinism of an LLM's output. Despite expectations of stability at temperature=0, we observe substantial variability across models, with completeness scoring showing the largest fluctuations. Cross-model comparisons reveal systematic differences in strictness and interpretive style, leading to divergent ratings for the same answers. Lower temperatures improve stability for some models, notably GPT-4o and Gemini, but have limited or inconsistent effects for Anthropic models. These findings have important implications for enterprise pipelines that rely on LLM-generated scores for routing, triage, gating, or quality control. Identical inputs can receive different scores depending on model, family, or temperature, raising concerns around fairness, reproducibility, and operational reliability. Our results highlight the need for monitoring, robust parsing, and hybrid human-LLM evaluation strategies to ensure dependable use of LLM-as-a-judge in production environments.
>
---
#### [new 005] Exploring the potential and limitations of Model Merging for Multi-Domain Adaptation in ASR
- **分类: cs.CL; eess.AS**

- **简介: 该论文研究模型融合在多领域语音识别中的应用，旨在解决多任务训练的计算成本高问题。通过评估多种融合算法，提出改进方法BoostedTSV-M，提升模型性能与稳定性。**

- **链接: [https://arxiv.org/pdf/2603.05354](https://arxiv.org/pdf/2603.05354)**

> **作者:** Carlos Carvalho; Francisco Teixeira; Thomas Rolland; Alberto Abad
>
> **备注:** submitted for review for INTERSPEECH2026 conference
>
> **摘要:** Model merging is a scalable alternative to multi-task training that combines the capabilities of multiple specialised models into a single model. This is particularly attractive for large speech foundation models, which are typically adapted through domain-specific fine-tuning, resulting in multiple customised checkpoints, for which repeating full fine-tuning when new data becomes available is computationally prohibitive. In this work, we study model merging for multi-domain ASR and benchmark 11 merging algorithms for 10 European Portuguese domains, evaluating in-domain accuracy, robustness under distribution shift, as well as English and multilingual performance. We further propose BoostedTSV-M, a new merging algorithm based on TSV-M that mitigates rank collapse via singular-value boosting and improves numerical stability. Overall, our approach outperforms full fine-tuning on European Portuguese while preserving out-of-distribution generalisation in a single model.
>
---
#### [new 006] ARC-TGI: Human-Validated Task Generators with Reasoning Chain Templates for ARC-AGI
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出ARC-TGI，用于生成可验证的ARC-AGI任务，解决静态数据集难以评估模型泛化能力的问题。通过生成多样化任务并保留规则，支持可控基准测试。**

- **链接: [https://arxiv.org/pdf/2603.05099](https://arxiv.org/pdf/2603.05099)**

> **作者:** Jens Lehmann; Syeda Khushbakht; Nikoo Salehfard; Nur A Zarin Nishat; Dhananjay Bhandiwad; Andrei Aioanei; Sahar Vahdati
>
> **摘要:** The Abstraction and Reasoning Corpus (ARC-AGI) probes few-shot abstraction and rule induction on small visual grids, but progress is difficult to measure on static collections of hand-authored puzzles due to overfitting, dataset leakage, and memorisation. We introduce ARC-TGI (ARC Task Generators Inventory), an open-source framework for task-family generators: compact Python programs that sample diverse ARC-AGI tasks while preserving a latent rule. ARC-TGI is built around a solver-facing representation: each generated task is paired with natural-language input and transformation reasoning chains and partially evaluated Python code implementing sampling, transformation, and episode construction. Crucially, ARC-TGI supports task-level constraints so that training examples collectively expose the variations needed to infer the underlying rule, a requirement for human-solvable ARC tasks that independent per-example sampling often fails to guarantee. All generators undergo human refinement and local verification to keep both grids and reasoning traces natural and consistent under variation. We release 461 generators covering 180 ARC-Mini tasks, 215 ARC-AGI-1 tasks (200 train, 15 test), and 66 ARC-AGI-2 tasks (55 train, 11 test), enabling scalable dataset sampling and controlled benchmarking.
>
---
#### [new 007] Probing Memes in LLMs: A Paradigm for the Entangled Evaluation World
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的模型评估任务，旨在解决传统评估方法忽略模型行为多样性的问题。通过引入“模因”概念，构建感知矩阵进行更细致的模型与数据交互分析。**

- **链接: [https://arxiv.org/pdf/2603.04408](https://arxiv.org/pdf/2603.04408)**

> **作者:** Luzhou Peng; Zhengxin Yang; Honglu Ji; Yikang Yang; Fanda Fan; Wanling Gao; Jiayuan Ge; Yilin Han; Jianfeng Zhan
>
> **备注:** 43 pages, 24 figures, 21 tables
>
> **摘要:** Current evaluation paradigms for large language models (LLMs) characterize models and datasets separately, yielding coarse descriptions: items in datasets are treated as pre-labeled entries, and models are summarized by overall scores such as accuracy, together ignoring the diversity of population-level model behaviors across items with varying properties. To address this gap, this paper conceptualizes LLMs as composed of memes, a notion introduced by Dawkins as cultural genes that replicate knowledge and behavior. Building on this perspective, the Probing Memes paradigm reconceptualizes evaluation as an entangled world of models and data. It centers on a Perception Matrix that captures model-item interactions, enabling Probe Properties for characterizing items and Meme Scores for depicting model behavioral traits. Applied to 9 datasets and 4,507 LLMs, Probing Memes reveals hidden capability structures and quantifies phenomena invisible under traditional paradigms (e.g., elite models failing on problems that most models answer easily). It not only supports more informative and extensible benchmarks but also enables population-based evaluation of LLMs.
>
---
#### [new 008] Representation Fidelity:Auditing Algorithmic Decisions About Humans Using Self-Descriptions
- **分类: cs.CL**

- **简介: 该论文属于算法审计任务，旨在评估算法决策的合理性。通过比较外部输入表示与自我描述，检测表示偏差，提出一种衡量算法决策合理性的方法。**

- **链接: [https://arxiv.org/pdf/2603.05136](https://arxiv.org/pdf/2603.05136)**

> **作者:** Theresa Elstner; Martin Potthast
>
> **摘要:** This paper introduces a new dimension for validating algorithmic decisions about humans by measuring the fidelity of their representations. Representation Fidelity measures if decisions about a person rest on reasonable grounds. We propose to operationalize this notion by measuring the distance between two representations of the same person: (1) an externally prescribed input representation on which the decision is based, and (2) a self-description provided by the human subject of the decision, used solely to validate the input representation. We examine the nature of discrepancies between these representations, how such discrepancies can be quantified, and derive a generic typology of representation mismatches that determine the degree of representation fidelity. We further present the first benchmark for evaluating representation fidelity based on a dataset of loan-granting decisions. Our Loan-Granting Self-Representations Corpus 2025 consists of a large corpus of 30 000 synthetic natural language self-descriptions derived from corresponding representations of applicants in the German Credit Dataset, along with expert annotations of representation mismatches between each pair of representations.
>
---
#### [new 009] Stan: An LLM-based thermodynamics course assistant
- **分类: cs.CL; cs.CY; physics.ed-ph**

- **简介: 该论文介绍Stan，一个基于大模型的热力学课程辅助系统，旨在支持学生和教师。任务是将AI应用于教育，解决师生工具分离的问题，通过统一的数据管道实现学生答疑和教师教学分析。**

- **链接: [https://arxiv.org/pdf/2603.04657](https://arxiv.org/pdf/2603.04657)**

> **作者:** Eric M. Furst; Vasudevan Venkateshwaran
>
> **备注:** 17 pages, 6 figures. For associated code repository, see this https URL
>
> **摘要:** Discussions of AI in education focus predominantly on student-facing tools -- chatbots, tutors, and problem generators -- while the potential for the same infrastructure to support instructors remains largely unexplored. We describe Stan, a suite of tools for an undergraduate chemical engineering thermodynamics course built on a data pipeline that we develop and deploy in dual roles: serving students and supporting instructors from a shared foundation of lecture transcripts and a structured textbook index. On the student side, a retrieval-augmented generation (RAG) pipeline answers natural-language queries by extracting technical terms, matching them against the textbook index, and synthesizing grounded responses with specific chapter and page references. On the instructor side, the same transcript corpus is processed through structured analysis pipelines that produce per-lecture summaries, identify student questions and moments of confusion, and catalog the anecdotes and analogies used to motivate difficult material -- providing a searchable, semester-scale record of teaching that supports course reflection, reminders, and improvement. All components, including speech-to-text transcription, structured content extraction, and interactive query answering, run entirely on locally controlled hardware using open-weight models (Whisper large-v3, Llama~3.1 8B) with no dependence on cloud APIs, ensuring predictable costs, full data privacy, and reproducibility independent of third-party services. We describe the design, implementation, and practical failure modes encountered when deploying 7--8 billion parameter models for structured extraction over long lecture transcripts, including context truncation, bimodal output distributions, and schema drift, along with the mitigations that resolved them.
>
---
#### [new 010] Stacked from One: Multi-Scale Self-Injection for Context Window Extension
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，解决大模型上下文窗口有限的问题。通过多粒度压缩与查询感知信息获取，构建双层架构实现长文本处理。**

- **链接: [https://arxiv.org/pdf/2603.04759](https://arxiv.org/pdf/2603.04759)**

> **作者:** Wei Han; Pan Zhou; Shuicheng Yan
>
> **摘要:** The limited context window of contemporary large language models (LLMs) remains a primary bottleneck for their broader application across diverse domains. Although continual pre-training on long-context data offers a straightforward solution, it incurs prohibitive data acquisition and computational costs. To address this challenge, we propose~\modelname, a novel framework based on multi-grained context compression and query-aware information acquisition. SharedLLM comprises two stacked short-context LLMs: a lower model serving as a compressor and an upper model acting as a decoder. The lower model compresses long inputs into compact, multi-grained representations, which are then forwarded to the upper model for context-aware processing. To maximize efficiency, this information transfer occurs exclusively at the lowest layers, bypassing lengthy forward passes and redundant cross-attention operations. This entire process, wherein the upper and lower models are derived from the same underlying LLM layers, is termed~\textit{self-injection}. To support this architecture, a specialized tree-based data structure enables the efficient encoding and query-aware retrieval of contextual information. Despite being trained on sequences of only 8K tokens, \modelname~effectively generalizes to inputs exceeding 128K tokens. Across a comprehensive suite of long-context modeling and understanding benchmarks, \modelname~achieves performance superior or comparable to strong baselines, striking an optimal balance between efficiency and accuracy. Furthermore, these design choices allow \modelname~to substantially reduce the memory footprint and yield notable inference speedups ($2\times$ over streaming and $3\times$ over encoder-decoder architectures).
>
---
#### [new 011] Context-Dependent Affordance Computation in Vision-Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究视觉语言模型中的上下文依赖性可供性计算，揭示其在不同语境下的表现差异，提出动态 ontological projection 用于机器人应用。**

- **链接: [https://arxiv.org/pdf/2603.04419](https://arxiv.org/pdf/2603.04419)**

> **作者:** Murad Farzulla
>
> **备注:** 31 pages, 8 tables, 4 figures, 43 references. Code available at: this https URL
>
> **摘要:** We characterize the phenomenon of context-dependent affordance computation in vision-language models (VLMs). Through a large-scale computational study (n=3,213 scene-context pairs from COCO-2017) using Qwen-VL 30B and LLaVA-1.5-13B subject to systematic context priming across 7 agentic personas, we demonstrate massive affordance drift: mean Jaccard similarity between context conditions is 0.095 (95% CI: [0.093, 0.096], p < 0.0001), indicating that >90% of lexical scene description is context-dependent. Sentence-level cosine similarity confirms substantial drift at the semantic level (mean = 0.415, 58.5% context-dependent). Stochastic baseline experiments (2,384 inference runs across 4 temperatures and 5 seeds) confirm this drift reflects genuine context effects rather than generation noise: within-prime variance is substantially lower than cross-prime variance across all conditions. Tucker decomposition with bootstrap stability analysis (n=1,000 resamples) reveals stable orthogonal latent factors: a "Culinary Manifold" isolated to chef contexts and an "Access Axis" spanning child-mobility contrasts. These findings establish that VLMs compute affordances in a substantially context-dependent manner -- with the difference between lexical (90%) and semantic (58.5%) measures reflecting that surface vocabulary changes more than underlying meaning under context shifts -- and suggest a direction for robotics research: dynamic, query-dependent ontological projection (JIT Ontology) rather than static world modeling. We do not claim to establish processing order or architectural primacy; such claims require internal representational analysis beyond output behavior.
>
---
#### [new 012] iAgentBench: Benchmarking Sensemaking Capabilities of Information-Seeking Agents on High-Traffic Topics
- **分类: cs.CL; cs.IR; cs.LG; cs.MA**

- **简介: 该论文提出iAgentBench，用于评估信息检索代理在高流量话题上的综合理解能力。针对传统基准无法衡量跨源信息整合的问题，构建真实用户意图的多源问题集，强调证据合成而非单纯检索。**

- **链接: [https://arxiv.org/pdf/2603.04656](https://arxiv.org/pdf/2603.04656)**

> **作者:** Preetam Prabhu Srikar Dammu; Arnav Palkhiwala; Tanya Roosta; Chirag Shah
>
> **摘要:** With the emergence of search-enabled generative QA systems, users are increasingly turning to tools that browse, aggregate, and reconcile evidence across multiple sources on their behalf. Yet many widely used QA benchmarks remain answerable by retrieving a single relevant passage, making them poorly suited for measuring cross-source sensemaking, such as integrating evidence, tracking causal links, and resolving dependencies across facets of a topic. We present iAgentBench, a dynamic ODQA benchmark that targets these higher-level information needs while keeping questions natural and grounded in realistic information-seeking behavior. iAgentBench draws seed topics from real-world attention signals and uses common user intent patterns to construct user-like questions whose answers require combining evidence from multiple sources, not just extracting a single snippet. Each instance is released with traceable evidence and auditable intermediate artifacts that support contamination checks and enable fine-grained diagnosis of failures in retrieval versus synthesis. Experiments across multiple LLMs show that retrieval improves accuracy, but retrieval alone does not reliably resolve these questions, underscoring the need to evaluate evidence use, not just evidence access.
>
---
#### [new 013] Diffusion LLMs can think EoS-by-EoS
- **分类: cs.CL**

- **简介: 该论文研究扩散大语言模型的推理能力，探讨其通过EoS令牌进行隐式计算的机制。任务为验证模型是否利用EoS tokens作为思维过程的隐藏工作区。**

- **链接: [https://arxiv.org/pdf/2603.05197](https://arxiv.org/pdf/2603.05197)**

> **作者:** Sarah Breckner; Sebastian Schuster
>
> **摘要:** Diffusion LLMs have been proposed as an alternative to autoregressive LLMs, excelling especially at complex reasoning tasks with interdependent sub-goals. Curiously, this is particularly true if the generation length, i.e., the number of tokens the model has to output, is set to a much higher value than is required for providing the correct answer to the task, and the model pads its answer with end-of-sequence (EoS) tokens. We hypothesize that diffusion models think EoS-by-EoS, that is, they use the representations of EoS tokens as a hidden scratchpad, which allows them to solve harder reasoning problems. We experiment with the diffusion models LLaDA1.5, LLaDA2.0-mini, and Dream-v0 on the tasks Addition, Entity Tracking, and Sudoku. In a controlled prompting experiment, we confirm that adding EoS tokens improves the LLMs' reasoning capabilities. To further verify whether they serve as space for hidden computations, we patch the hidden states of the EoS tokens with those of a counterfactual generation, which frequently changes the generated output to the counterfactual. The success of the causal intervention underscores that the EoS tokens, which one may expect to be devoid of meaning, carry information on the problem to solve. The behavioral experiments and the causal interventions indicate that diffusion LLMs can indeed think EoS-by-EoS.
>
---
#### [new 014] Guidelines for the Annotation and Visualization of Legal Argumentation Structures in Chinese Judicial Decisions
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于法律论证结构标注与可视化任务，旨在规范中文司法判决中的法律论证结构表示，解决司法推理逻辑分析与数据标准化问题。工作包括提出命题类型、关系类型及可视化规则。**

- **链接: [https://arxiv.org/pdf/2603.05171](https://arxiv.org/pdf/2603.05171)**

> **作者:** Kun Chen; Xianglei Liao; Kaixue Fei; Yi Xing; Xinrui Li
>
> **备注:** The PDF contains both an English translation and the original Chinese guideline. The first 30 pages present the full English translation, while the remaining 25 pages provide the original Chinese version
>
> **摘要:** This guideline proposes a systematic and operational annotation framework for representing the structure of legal argumentation in judicial decisions. Grounded in theories of legal reasoning and argumentation, the framework aims to reveal the logical organization of judicial reasoning and to provide a reliable data foundation for computational analysis. At the proposition level, the guideline distinguishes four types of propositions: general normative propositions, specific normative propositions, general factual propositions, and specific factual propositions. At the relational level, five types of relations are defined to capture argumentative structures: support, attack, joint, match, and identity. These relations represent positive and negative argumentative connections, conjunctive reasoning structures, the correspondence between legal norms and case facts, and semantic equivalence between propositions. The guideline further specifies formal representation rules and visualization conventions for both basic and nested structures, enabling consistent graphical representation of complex argumentation patterns. In addition, it establishes a standardized annotation workflow and consistency control mechanisms to ensure reproducibility and reliability of the annotated data. By providing a clear conceptual model, formal representation rules, and practical annotation procedures, this guideline offers methodological support for large-scale analysis of judicial reasoning and for future research in legal argument mining, computational modeling of legal reasoning, and AI-assisted legal analysis.
>
---
#### [new 015] Ensembling Language Models with Sequential Monte Carlo
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，解决语言模型集成问题。提出一种统一框架和SMC算法，实现更有效的模型集成与文本生成。**

- **链接: [https://arxiv.org/pdf/2603.05432](https://arxiv.org/pdf/2603.05432)**

> **作者:** Robin Shing Moon Chan; Tianyu Liu; Samuel Kiegeland; Clemente Pasti; Jacob Hoover Vigly; Timothy J. O'Donnell; Ryan Cotterell; Tim Vieira
>
> **摘要:** Practitioners have access to an abundance of language models and prompting strategies for solving many language modeling tasks; yet prior work shows that modeling performance is highly sensitive to both choices. Classical machine learning ensembling techniques offer a principled approach: aggregate predictions from multiple sources to achieve better performance than any single one. However, applying ensembling to language models during decoding is challenging: naively aggregating next-token probabilities yields samples from a locally normalized, biased approximation of the generally intractable ensemble distribution over strings. In this work, we introduce a unified framework for composing $K$ language models into $f$-ensemble distributions for a wide range of functions $f\colon\mathbb{R}_{\geq 0}^{K}\to\mathbb{R}_{\geq 0}$. To sample from these distributions, we propose a byte-level sequential Monte Carlo (SMC) algorithm that operates in a shared character space, enabling ensembles of models with mismatching vocabularies and consistent sampling in the limit. We evaluate a family of $f$-ensembles across prompt and model combinations for various structured text generation tasks, highlighting the benefits of alternative aggregation strategies over traditional probability averaging, and showing that better posterior approximations can yield better ensemble performance.
>
---
#### [new 016] Hate Speech Detection using Large Language Models with Data Augmentation and Feature Enhancement
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于 hate speech detection 任务，旨在提升检测效果。通过数据增强和特征优化，对比传统方法与Transformer模型，探索不同技术对模型性能的影响。**

- **链接: [https://arxiv.org/pdf/2603.04698](https://arxiv.org/pdf/2603.04698)**

> **作者:** Brian Jing Hong Nge; Stefan Su; Thanh Thi Nguyen; Campbell Wilson; Alexandra Phelan; Naomi Pfitzner
>
> **备注:** Accepted for publication in the Proceedings of the 8th International Conference on Natural Language Processing (ICNLP 2026)
>
> **摘要:** This paper evaluates data augmentation and feature enhancement techniques for hate speech detection, comparing traditional classifiers, e.g., Delta Term Frequency-Inverse Document Frequency (Delta TF-IDF), with transformer-based models (DistilBERT, RoBERTa, DeBERTa, Gemma-7B, gpt-oss-20b) across diverse datasets. It examines the impact of Synthetic Minority Over-sampling Technique (SMOTE), weighted loss determined by inverse class proportions, Part-of-Speech (POS) tagging, and text data augmentation on model performance. The open-source gpt-oss-20b consistently achieves the highest results. On the other hand, Delta TF-IDF responds strongly to data augmentation, reaching 98.2% accuracy on the Stormfront dataset. The study confirms that implicit hate speech is more difficult to detect than explicit hateful content and that enhancement effectiveness depends on dataset, model, and technique interaction. Our research informs the development of hate speech detection by highlighting how dataset properties, model architectures, and enhancement strategies interact, supporting more accurate and context-aware automated detection.
>
---
#### [new 017] An Exploration-Analysis-Disambiguation Reasoning Framework for Word Sense Disambiguation with Low-Parameter LLMs
- **分类: cs.CL**

- **简介: 该论文属于词义消歧任务，旨在提升低参数大语言模型的词义识别能力。通过推理驱动的微调策略，实现高效准确的WSD。**

- **链接: [https://arxiv.org/pdf/2603.05400](https://arxiv.org/pdf/2603.05400)**

> **作者:** Deshan Sumanathilaka; Nicholas Micallef; Julian Hough
>
> **备注:** Accepted at LREC 2026, 15 pages, 11 Tables
>
> **摘要:** Word Sense Disambiguation (WSD) remains a key challenge in Natural Language Processing (NLP), especially when dealing with rare or domain-specific senses that are often misinterpreted. While modern high-parameter Large Language Models (LLMs) such as GPT-4-Turbo have shown state-of-the-art WSD performance, their computational and energy demands limit scalability. This study investigates whether low-parameter LLMs (<4B parameters) can achieve comparable results through fine-tuning strategies that emphasize reasoning-driven sense identification. Using the FEWS dataset augmented with semi-automated, rationale-rich annotations, we fine-tune eight small-scale open-source LLMs (e.g. Gemma and Qwen). Our results reveal that Chain-of-Thought (CoT)-based reasoning combined with neighbour-word analysis achieves performance comparable to GPT-4-Turbo in zero-shot settings. Importantly, Gemma-3-4B and Qwen-3-4B models consistently outperform all medium-parameter baselines and state-of-the-art models on FEWS, with robust generalization to unseen senses. Furthermore, evaluation on the unseen "Fool Me If You Can'' dataset confirms strong cross-domain adaptability without task-specific fine-tuning. This work demonstrates that with carefully crafted reasoning-centric fine-tuning, low-parameter LLMs can deliver accurate WSD while substantially reducing computational and energy demands.
>
---
#### [new 018] Query Disambiguation via Answer-Free Context: Doubling Performance on Humanity's Last Exam
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的问答任务，旨在解决查询歧义问题。通过结合动态上下文构建与查询重写，提升模型回答准确性。**

- **链接: [https://arxiv.org/pdf/2603.04454](https://arxiv.org/pdf/2603.04454)**

> **作者:** Michael Majurski; Cynthia Matuszek
>
> **摘要:** How carefully and unambiguously a question is phrased has a profound impact on the quality of the response, for Language Models (LMs) as well as people. While model capabilities continue to advance, the interplay between grounding context and query formulation remains under-explored. This work investigates how the quality of background grounding information in a model's context window affects accuracy. We find that combining well-grounded dynamic context construction (i.e, RAG) with query rewriting reduces question ambiguity, resulting in significant accuracy gains. Given a user question with associated answer-free grounding context, rewriting the question to reduce ambiguity produces benchmark improvements without changing the answer itself, even compared to prepending that context before the question. Using \texttt{gpt-oss-20b} to rewrite a subset of Humanity's Last Exam using answer-free grounding context improves \texttt{gpt-5-mini} accuracy from 0.14 to 0.37. We demonstrate that this accuracy improvement cannot be fully recovered just through prompting at inference time; rather, distinct rewriting and answering phases are required. Code and data are available at this https URL
>
---
#### [new 019] Balancing Coverage and Draft Latency in Vocabulary Trimming for Faster Speculative Decoding
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于语言模型推理优化任务，解决草案模型在覆盖率与延迟间的权衡问题。通过词汇剪枝，在保证覆盖率的同时显著降低延迟，提升解码效率。**

- **链接: [https://arxiv.org/pdf/2603.05210](https://arxiv.org/pdf/2603.05210)**

> **作者:** Ofir Ben Shoham
>
> **摘要:** Speculative decoding accelerates inference for Large Language Models by using a lightweight draft model to propose candidate tokens that are verified in parallel by a larger target model. Prior work shows that the draft model often dominates speculative decoding latency, since it generates tokens sequentially and incurs high cost from its language modeling head as vocabulary size grows. This exposes a fundamental trade-off in draft model design: larger vocabularies improve token coverage and agreement with the target model, but incur higher draft latency, while smaller vocabularies reduce latency at the risk of missing tokens required for accurate draft generation. We address this trade-off through vocabulary trimming for draft models, motivated by the observation that domain-specific workloads use only a small fraction of the full vocabulary. We cast draft vocabulary selection as a constrained optimization problem that balances token coverage and draft latency. Coverage is computed over assistant responses in the training data, while latency is estimated using architecture-aware FLOPs that capture the cost of the language modeling head as a function of vocabulary size. We optimize a utility function with a Tree-structured Parzen Estimator to efficiently explore the coverage-latency Pareto frontier under a minimum coverage constraint. Experiments show improved speculative decoding throughput while reducing draft vocabularies by up to 97% with high coverage. On domain-specific tasks, we achieve up to 16% latency reduction and 20% throughput improvement, and up to 6.7% throughput gains on diverse out-of-distribution tasks.
>
---
#### [new 020] IF-RewardBench: Benchmarking Judge Models for Instruction-Following Evaluation
- **分类: cs.CL**

- **简介: 该论文属于模型评估任务，旨在解决现有裁判模型在指令遵循评估中的可靠性问题。提出IF-RewardBench基准，构建偏好图进行多响应排序评估。**

- **链接: [https://arxiv.org/pdf/2603.04738](https://arxiv.org/pdf/2603.04738)**

> **作者:** Bosi Wen; Yilin Niu; Cunxiang Wang; Xiaoying Ling; Ying Zhang; Pei Ke; Hongning Wang; Minlie Huang
>
> **备注:** 27 pages, 7 figures
>
> **摘要:** Instruction-following is a foundational capability of large language models (LLMs), with its improvement hinging on scalable and accurate feedback from judge models. However, the reliability of current judge models in instruction-following remains underexplored due to several deficiencies of existing meta-evaluation benchmarks, such as their insufficient data coverage and oversimplified pairwise evaluation paradigms that misalign with model optimization scenarios. To this end, we propose IF-RewardBench, a comprehensive meta-evaluation benchmark for instruction-following that covers diverse instruction and constraint types. For each instruction, we construct a preference graph containing all pairwise preferences among multiple responses based on instruction-following quality. This design enables a listwise evaluation paradigm that assesses the capabilities of judge models to rank multiple responses, which is essential in guiding model alignment. Extensive experiments on IF-RewardBench reveal significant deficiencies in current judge models and demonstrate that our benchmark achieves a stronger positive correlation with downstream task performance compared to existing benchmarks. Our codes and data are available at this https URL.
>
---
#### [new 021] Progressive Residual Warmup for Language Model Pretraining
- **分类: cs.CL**

- **简介: 该论文属于语言模型预训练任务，旨在解决预训练稳定性与收敛速度问题。提出ProRes方法，通过渐进式残差热身提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.05369](https://arxiv.org/pdf/2603.05369)**

> **作者:** Tianhao Chen; Xin Xu; Lu Yin; Hao Chen; Yang Wang; Shizhe Diao; Can Yang
>
> **摘要:** Transformer architectures serve as the backbone for most modern Large Language Models, therefore their pretraining stability and convergence speed are of central concern. Motivated by the logical dependency of sequentially stacked layers, we propose Progressive Residual Warmup (ProRes) for language model pretraining. ProRes implements an "early layer learns first" philosophy by multiplying each layer's residual with a scalar that gradually warms up from 0 to 1, with deeper layers taking longer warmup steps. In this way, deeper layers wait for early layers to settle into a more stable regime before contributing to learning. We demonstrate the effectiveness of ProRes through pretraining experiments across various model scales, as well as normalization and initialization schemes. Comprehensive analysis shows that ProRes not only stabilizes pretraining but also introduces a unique optimization trajectory, leading to faster convergence, stronger generalization and better downstream performance. Our code is available at this https URL.
>
---
#### [new 022] PersianPunc: A Large-Scale Dataset and BERT-Based Approach for Persian Punctuation Restoration
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于 Persian 语音转文本的标点恢复任务，旨在提升 ASR 输出的可读性。提出 PersianPunc 数据集和基于 BERT 的轻量方法，解决标点缺失问题，取得高准确率并适合实时应用。**

- **链接: [https://arxiv.org/pdf/2603.05314](https://arxiv.org/pdf/2603.05314)**

> **作者:** Mohammad Javad Ranjbar Kalahroodi; Heshaam Faili; Azadeh Shakery
>
> **摘要:** Punctuation restoration is essential for improving the readability and downstream utility of automatic speech recognition (ASR) outputs, yet remains underexplored for Persian despite its importance. We introduce PersianPunc, a large-scale, high-quality dataset of 17 million samples for Persian punctuation restoration, constructed through systematic aggregation and filtering of existing textual resources. We formulate punctuation restoration as a token-level sequence labeling task and fine-tune ParsBERT to achieve strong performance. Through comparative evaluation, we demonstrate that while large language models can perform punctuation restoration, they suffer from critical limitations: over-correction tendencies that introduce undesired edits beyond punctuation insertion (particularly problematic for speech-to-text pipelines) and substantially higher computational requirements. Our lightweight BERT-based approach achieves a macro-averaged F1 score of 91.33% on our test set while maintaining efficiency suitable for real-time applications. We make our dataset (this https URL) and model (this https URL) publicly available to facilitate future research in Persian NLP and provide a scalable framework applicable to other morphologically rich, low-resource languages.
>
---
#### [new 023] ThaiSafetyBench: Assessing Language Model Safety in Thai Cultural Contexts
- **分类: cs.CL**

- **简介: 该论文属于语言模型安全评估任务，旨在解决非英语语言在文化背景下的安全风险问题。工作包括构建ThaiSafetyBench基准、评估24个模型并提出分类器以提升安全性。**

- **链接: [https://arxiv.org/pdf/2603.04992](https://arxiv.org/pdf/2603.04992)**

> **作者:** Trapoom Ukarapol; Nut Chukamphaeng; Kunat Pipatanakul; Pakhapoom Sarapat
>
> **备注:** ICLR 2026 Workshop on Principled Design for Trustworthy AI
>
> **摘要:** The safety evaluation of large language models (LLMs) remains largely centered on English, leaving non-English languages and culturally grounded risks underexplored. In this work, we investigate LLM safety in the context of the Thai language and culture and introduce ThaiSafetyBench, an open-source benchmark comprising 1,954 malicious prompts written in Thai. The dataset covers both general harmful prompts and attacks that are explicitly grounded in Thai cultural, social, and contextual nuances. Using ThaiSafetyBench, we evaluate 24 LLMs, with GPT-4.1 and Gemini-2.5-Pro serving as LLM-as-a-judge evaluators. Our results show that closed-source models generally demonstrate stronger safety performance than open-source counterparts, raising important concerns regarding the robustness of openly available models. Moreover, we observe a consistently higher Attack Success Rate (ASR) for Thai-specific, culturally contextualized attacks compared to general Thai-language attacks, highlighting a critical vulnerability in current safety alignment methods. To improve reproducibility and cost efficiency, we further fine-tune a DeBERTa-based harmful response classifier, which we name ThaiSafetyClassifier. The model achieves a weighted F1 score of 84.4%, matching GPT-4.1 judgments. We publicly release the fine-tuning weights and training scripts to support reproducibility. Finally, we introduce the ThaiSafetyBench leaderboard to provide continuously updated safety evaluations and encourage community participation. - ThaiSafetyBench HuggingFace Dataset: this https URL - ThaiSafetyBench Github: this https URL - ThaiSafetyClassifier HuggingFace Model: this https URL - ThaiSafetyBench Leaderboard: this https URL
>
---
#### [new 024] From Unfamiliar to Familiar: Detecting Pre-training Data via Gradient Deviations in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于预训练数据检测任务，旨在解决版权和基准污染问题。通过分析梯度偏差，提出GDS方法有效识别预训练数据。**

- **链接: [https://arxiv.org/pdf/2603.04828](https://arxiv.org/pdf/2603.04828)**

> **作者:** Ruiqi Zhang; Lingxiang Wang; Hainan Zhang; Zhiming Zheng; Yanyan Lan
>
> **摘要:** Pre-training data detection for LLMs is essential for addressing copyright concerns and mitigating benchmark contamination. Existing methods mainly focus on the likelihood-based statistical features or heuristic signals before and after fine-tuning, but the former are susceptible to word frequency bias in corpora, and the latter strongly depend on the similarity of fine-tuning data. From an optimization perspective, we observe that during training, samples transition from unfamiliar to familiar in a manner reflected by systematic differences in gradient behavior. Familiar samples exhibit smaller update magnitudes, distinct update locations in model components, and more sharply activated neurons. Based on this insight, we propose GDS, a method that identifies pre-training data by probing Gradient Deviation Scores of target samples. Specifically, we first represent each sample using gradient profiles that capture the magnitude, location, and concentration of parameter updates across FFN and Attention modules, revealing consistent distinctions between member and non-member data. These features are then fed into a lightweight classifier to perform binary membership inference. Experiments on five public datasets show that GDS achieves state-of-the-art performance with significantly improved cross-dataset transferability over strong baselines. Further interpretability analyse show gradient feature distribution differences, enabling practical and scalable pre-training data detection.
>
---
#### [new 025] AILS-NTUA at SemEval-2026 Task 3: Efficient Dimensional Aspect-Based Sentiment Analysis
- **分类: cs.CL**

- **简介: 该论文属于多语言、多领域的情感分析任务，解决维度化方面情感预测问题。提出AILS-NTUA系统，结合语言适配的模型微调与LoRA指令调优，实现高效情感回归、三元组和四元组提取。**

- **链接: [https://arxiv.org/pdf/2603.04933](https://arxiv.org/pdf/2603.04933)**

> **作者:** Stavros Gazetas; Giorgos Filandrianos; Maria Lymperaiou; Paraskevi Tzouveli; Athanasios Voulodimos; Giorgos Stamou
>
> **摘要:** In this paper, we present AILS-NTUA system for Track-A of SemEval-2026 Task 3 on Dimensional Aspect-Based Sentiment Analysis (DimABSA), which encompasses three complementary problems: Dimensional Aspect Sentiment Regression (DimASR), Dimensional Aspect Sentiment Triplet Extraction (DimASTE), and Dimensional Aspect Sentiment Quadruplet Prediction (DimASQP) within a multilingual and multi-domain framework. Our methodology combines fine-tuning of language-appropriate encoder backbones for continuous aspect-level sentiment prediction with language-specific instruction tuning of large language models using LoRA for structured triplet and quadruplet extraction. This unified yet task-adaptive design emphasizes parameter-efficient specialization across languages and domains, enabling reduced training and inference requirements while maintaining strong effectiveness. Empirical results demonstrate that the proposed models achieve competitive performance and consistently surpass the provided baselines across most evaluation settings.
>
---
#### [new 026] Autoscoring Anticlimax: A Meta-analytic Understanding of AI's Short-answer Shortcomings and Wording Weaknesses
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于自动评分任务，旨在解决AI在短答案评分中的不足。通过元分析，研究发现LLM在难度任务上表现差，且解码器架构不如编码器。**

- **链接: [https://arxiv.org/pdf/2603.04820](https://arxiv.org/pdf/2603.04820)**

> **作者:** Michael Hardy
>
> **摘要:** Automated short-answer scoring lags other LLM applications. We meta-analyze 890 culminating results across a systematic review of LLM short-answer scoring studies, modeling the traditional effect size of Quadratic Weighted Kappa (QWK) with mixed effects metaregression. We quantitatively illustrate that that the level of difficulty for human experts to perform the task of scoring written work of children has no observed statistical effect on LLM performance. Particularly, we show that some scoring tasks measured as the easiest by human scorers were the hardest for LLMs. Whether by poor implementation by thoughtful researchers or patterns traceable to autoregressive training, on average decoder-only architectures underperform encoders by 0.37--a substantial difference in agreement with humans. Additionally, we measure the contributions of various aspects of LLM technology on successful scoring such as tokenizer vocabulary size, which exhibits diminishing returns--potentially due to undertrained tokens. Findings argue for systems design which better anticipates known statistical shortcomings of autoregressive models. Finally, we provide additional experiments to illustrate wording and tokenization sensitivity and bias elicitation in high-stakes education contexts, where LLMs demonstrate racial discrimination. Code and data for this study are available.
>
---
#### [new 027] AILS-NTUA at SemEval-2026 Task 10: Agentic LLMs for Psycholinguistic Marker Extraction and Conspiracy Endorsement Detection
- **分类: cs.CL**

- **简介: 该论文针对SemEval-2026 Task 10，解决心理语言标记提取与阴谋论支持检测问题，提出一种代理大模型流水线，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2603.04921](https://arxiv.org/pdf/2603.04921)**

> **作者:** Panagiotis Alexios Spanakis; Maria Lymperaiou; Giorgos Filandrianos; Athanasios Voulodimos; Giorgos Stamou
>
> **摘要:** This paper presents a novel agentic LLM pipeline for SemEval-2026 Task 10 that jointly extracts psycholinguistic conspiracy markers and detects conspiracy endorsement. Unlike traditional classifiers that conflate semantic reasoning with structural localization, our decoupled design isolates these challenges. For marker extraction, we propose Dynamic Discriminative Chain-of-Thought (DD-CoT) with deterministic anchoring to resolve semantic ambiguity and character-level brittleness. For conspiracy detection, an "Anti-Echo Chamber" architecture, consisting of an adversarial Parallel Council adjudicated by a Calibrated Judge, overcomes the "Reporter Trap," where models falsely penalize objective reporting. Achieving 0.24 Macro F1 (+100\% over baseline) on S1 and 0.79 Macro F1 (+49\%) on S2, with the S1 system ranking 3rd on the development leaderboard, our approach establishes a versatile paradigm for interpretable, psycholinguistically-grounded NLP.
>
---
#### [new 028] AI-Assisted Moot Courts: Simulating Justice-Specific Questioning in Oral Arguments
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于法律AI任务，旨在解决如何用AI模拟法官提问的问题。研究构建并评估了模拟器，探索其在法学院训练中的有效性与改进方向。**

- **链接: [https://arxiv.org/pdf/2603.04718](https://arxiv.org/pdf/2603.04718)**

> **作者:** Kylie Zhang; Nimra Nadeem; Lucia Zheng; Dominik Stammbach; Peter Henderson
>
> **备注:** Accepted at CS & Law 2026
>
> **摘要:** In oral arguments, judges probe attorneys with questions about the factual record, legal claims, and the strength of their arguments. To prepare for this questioning, both law schools and practicing attorneys rely on moot courts: practice simulations of appellate hearings. Leveraging a dataset of U.S. Supreme Court oral argument transcripts, we examine whether AI models can effectively simulate justice-specific questioning for moot court-style training. Evaluating oral argument simulation is challenging because there is no single correct question for any given turn. Instead, effective questioning should reflect a combination of desirable qualities, such as anticipating substantive legal issues, detecting logical weaknesses, and maintaining an appropriately adversarial tone. We introduce a two-layer evaluation framework that assesses both the realism and pedagogical usefulness of simulated questions using complementary proxy metrics. We construct and evaluate both prompt-based and agentic oral argument simulators. We find that simulated questions are often perceived as realistic by human annotators and achieve high recall of ground truth substantive legal issues. However, models still face substantial shortcomings, including low diversity in question types and sycophancy. Importantly, these shortcomings would remain undetected under naive evaluation approaches.
>
---
#### [new 029] From Static Inference to Dynamic Interaction: Navigating the Landscape of Streaming Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决静态推理与动态交互之间的差距。论文提出统一定义，构建分类体系，分析方法并探讨应用方向。**

- **链接: [https://arxiv.org/pdf/2603.04592](https://arxiv.org/pdf/2603.04592)**

> **作者:** Junlong Tong; Zilong Wang; YuJie Ren; Peiran Yin; Hao Wu; Wei Zhang; Xiaoyu Shen
>
> **摘要:** Standard Large Language Models (LLMs) are predominantly designed for static inference with pre-defined inputs, which limits their applicability in dynamic, real-time scenarios. To address this gap, the streaming LLM paradigm has emerged. However, existing definitions of streaming LLMs remain fragmented, conflating streaming generation, streaming inputs, and interactive streaming architectures, while a systematic taxonomy is still lacking. This paper provides a comprehensive overview and analysis of streaming LLMs. First, we establish a unified definition of streaming LLMs based on data flow and dynamic interaction to clarify existing ambiguities. Building on this definition, we propose a systematic taxonomy of current streaming LLMs and conduct an in-depth discussion on their underlying methodologies. Furthermore, we explore the applications of streaming LLMs in real-world scenarios and outline promising research directions to support ongoing advances in streaming intelligence. We maintain a continuously updated repository of relevant papers at this https URL.
>
---
#### [new 030] What Is Missing: Interpretable Ratings for Large Language Model Outputs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的偏好学习任务，旨在解决传统数值评分主观性强、缺乏解释的问题。提出WIM评分系统，通过自然语言反馈生成可解释的评分，提升偏好数据质量。**

- **链接: [https://arxiv.org/pdf/2603.04429](https://arxiv.org/pdf/2603.04429)**

> **作者:** Nicholas Stranges; Yimin Yang
>
> **备注:** 22 pages
>
> **摘要:** Current Large Language Model (LLM) preference learning methods such as Proximal Policy Optimization and Direct Preference Optimization learn from direct rankings or numerical ratings of model outputs, these rankings are subjective, and a single numerical rating chosen directly by a judge is a poor proxy for the quality of natural language, we introduce the What Is Missing (WIM) rating system to produce rankings from natural-language feedback, WIM integrates into existing training pipelines, can be combined with other rating techniques, and can be used as input to any preference learning method without changing the learning algorithm, to compute a WIM rating, a human or LLM judge writes feedback describing what the model output is missing, we embed the output and the feedback with a sentence embedding model and compute the cosine similarity between the resulting vectors, we empirically observe that, compared to discrete numerical ratings, WIM yields fewer ties and larger rating deltas, which improves the availability of a learning signal in pairwise preference data, we use interpretable in the following limited sense: for each scalar rating, we can inspect the judge's missing-information text that produced it, enabling qualitative debugging of the preference labels.
>
---
#### [new 031] When Weak LLMs Speak with Confidence, Preference Alignment Gets Stronger
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于偏好对齐任务，旨在降低标注成本。通过利用弱大模型的置信度，提升对齐效果，证明其可替代部分人工标注。**

- **链接: [https://arxiv.org/pdf/2603.04968](https://arxiv.org/pdf/2603.04968)**

> **作者:** Amirabbas Afzali; Myeongho Jeon; Maria Brbic
>
> **备注:** 32 pages, 8 figures, International Conference on Learning Representations 2026
>
> **摘要:** Preference alignment is an essential step in adapting large language models (LLMs) to human values, but existing approaches typically depend on costly human annotations or large-scale API-based models. We explore whether a weak LLM can instead act as an effective annotator. We surprisingly find that selecting only a subset of a weak LLM's highly confident samples leads to substantially better performance than using full human annotations. Building on this insight, we propose Confidence-Weighted Preference Optimization (CW-PO), a general framework that re-weights training samples by a weak LLM's confidence and can be applied across different preference optimization objectives. Notably, the model aligned by CW-PO with just 20% of human annotations outperforms the model trained with 100% of annotations under standard DPO. These results suggest that weak LLMs, when paired with confidence weighting, can dramatically reduce the cost of preference alignment while even outperforming methods trained on fully human-labeled data.
>
---
#### [new 032] Bootstrapping Exploration with Group-Level Natural Language Feedback in Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于强化学习任务，解决传统方法忽略自然语言反馈导致探索效率低的问题。提出GOLF框架，利用群体级语言反馈提升探索效率。**

- **链接: [https://arxiv.org/pdf/2603.04597](https://arxiv.org/pdf/2603.04597)**

> **作者:** Lei Huang; Xiang Cheng; Chenxiao Zhao; Guobin Shen; Junjie Yang; Xiaocheng Feng; Yuxuan Gu; Xing Yu; Bing Qin
>
> **摘要:** Large language models (LLMs) typically receive diverse natural language (NL) feedback through interaction with the environment. However, current reinforcement learning (RL) algorithms rely solely on scalar rewards, leaving the rich information in NL feedback underutilized and leading to inefficient exploration. In this work, we propose GOLF, an RL framework that explicitly exploits group-level language feedback to guide targeted exploration through actionable refinements. GOLF aggregates two complementary feedback sources: (i) external critiques that pinpoint errors or propose targeted fixes, and (ii) intra-group attempts that supply alternative partial ideas and diverse failure patterns. These group-level feedbacks are aggregated to produce high-quality refinements, which are adaptively injected into training as off-policy scaffolds to provide targeted guidance in sparse-reward regions. Meanwhile, GOLF jointly optimizes generation and refinement within a unified RL loop, creating a virtuous cycle that continuously improves both capabilities. Experiments on both verifiable and non-verifiable benchmarks show that GOLF achieves superior performance and exploration efficiency, achieving 2.2$\times$ improvements in sample efficiency compared to RL methods trained solely on scalar rewards. Code is available at this https URL.
>
---
#### [new 033] Can LLMs Capture Expert Uncertainty? A Comparative Analysis of Value Alignment in Ethnographic Qualitative Research
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的价值对齐任务，旨在评估LLMs在定性研究中捕捉专家不确定性的能力。工作包括比较LLM与专家在价值观识别上的表现及不确定性模式。**

- **链接: [https://arxiv.org/pdf/2603.04897](https://arxiv.org/pdf/2603.04897)**

> **作者:** Arina Kostina; Marios Dikaiakos; Alejandro Porcel; Tassos Stassopoulos
>
> **备注:** Accepted for a poster session at this http URL@MIT 2026
>
> **摘要:** Qualitative analysis of open-ended interviews plays a central role in ethnographic and economic research by uncovering individuals' values, motivations, and culturally embedded financial behaviors. While large language models (LLMs) offer promising support for automating and enriching such interpretive work, their ability to produce nuanced, reliable interpretations under inherent task ambiguity remains unclear. In our work we evaluate LLMs on the task of identifying the top three human values expressed in long-form interviews based on the Schwartz Theory of Basic Values framework. We compare their outputs to expert annotations, analyzing both performance and uncertainty patterns relative to the experts. Results show that LLMs approach the human ceiling on set-based metrics (F1, Jaccard) but struggle to recover exact value rankings, as reflected in lower RBO scores. While the average Schwartz value distributions of most models closely match those of human analysts, their uncertainty structures across the Schwartz values diverge from expert uncertainty patterns. Among the evaluated models, Qwen performs closest to expert-level agreement and exhibits the strongest alignment with expert Schwartz value distributions. LLM ensemble methods yield consistent gains across metrics, with Majority Vote and Borda Count performing best. Notably, systematic overemphasis on certain Schwartz values, like Security, suggests both the potential of LLMs to provide complementary perspectives and the need to further investigate model-induced value biases. Overall, our findings highlight both the promise and the limitations of LLMs as collaborators in inherently ambiguous qualitative value analysis.
>
---
#### [new 034] Multiclass Hate Speech Detection with RoBERTa-OTA: Integrating Transformer Attention and Graph Convolutional Networks
- **分类: cs.CL**

- **简介: 该论文属于多类仇恨言论检测任务，解决社交媒体中因语言变体和隐含目标策略导致的分类难题。提出RoBERTa-OTA模型，融合Transformer注意力与图卷积网络，提升分类准确性。**

- **链接: [https://arxiv.org/pdf/2603.04414](https://arxiv.org/pdf/2603.04414)**

> **作者:** Mahmoud Abusaqer; Jamil Saquer
>
> **备注:** 15 pages, 2 figures, 6 tables. Accepted for publication in the Proceedings of the 12th Annual Conference on Computational Science & Computational Intelligence (CSCI'25)
>
> **摘要:** Multiclass hate speech detection across demographic categories remains computationally challenging due to implicit targeting strategies and linguistic variability in social media content. Existing approaches rely solely on learned representations from training data, without explicitly incorporating structured ontological frameworks that can enhance classification through formal domain knowledge integration. We propose RoBERTa-OTA, which introduces ontology-guided attention mechanisms that process textual features alongside structured knowledge representations through enhanced Graph Convolutional Networks. The architecture combines RoBERTa embeddings with scaled attention layers and graph neural networks to integrate contextual language understanding with domain-specific semantic knowledge. Evaluation across 39,747 balanced samples using 5-fold cross-validation demonstrates significant performance gains over baseline RoBERTa implementations and existing state-of-the-art methods. RoBERTa-OTA achieves 96.04\% accuracy compared to 95.02\% for standard RoBERTa, with substantial improvements for challenging categories: gender-based hate speech detection improves by 2.36 percentage points while other hate speech categories improve by 2.38 percentage points. The enhanced architecture maintains computational efficiency with only 0.33\% parameter overhead, providing practical advantages for large-scale content moderation applications requiring fine-grained demographic hate speech classification.
>
---
#### [new 035] NeuronMoE: Neuron-Guided Mixture-of-Experts for Efficient Multilingual LLM Extension
- **分类: cs.CL**

- **简介: 该论文属于多语言大模型扩展任务，旨在降低低资源语言模型的训练成本。通过分析神经元特性，优化专家分配，实现参数减少且性能保持。**

- **链接: [https://arxiv.org/pdf/2603.05046](https://arxiv.org/pdf/2603.05046)**

> **作者:** Rongzhi Li; Hitomi Yanaka
>
> **摘要:** Extending large language models to low-resource languages is essential for global accessibility, but training separate models per language is prohibitively expensive. Mixture-of-Experts (MoE) architectures address this by adding sparse language-specific parameters, but determining how many experts each layer needs remains an open question. Current approaches allocate experts based on layer-level similarity, yet language processing exhibits fine-grained specialization at individual neurons. We propose $\textbf{NeuronMoE}$, a method that analyzes language-specific neurons across all transformer components to guide expert allocation per layer based on empirically measured cross-lingual neuron diversity. Applied to Llama-3.2-3B for low-resource languages (Greek, Turkish, and Hungarian), this approach achieves approximately 40% average parameter reduction while matching the performance of the LayerMoE baseline. We find that low-resource language experts independently develop neuron specialization patterns mirroring the high-resource language, which are concentrated in early and late layers. This reveals potential universal architectural principles in how multilingual models organize linguistic knowledge.
>
---
#### [new 036] Non-Zipfian Distribution of Stopwords and Subset Selection Models
- **分类: cs.CL**

- **简介: 该论文研究停用词的分布规律，提出一种基于词频排名的停用词选择模型，以解释其非Zipf分布现象。**

- **链接: [https://arxiv.org/pdf/2603.04691](https://arxiv.org/pdf/2603.04691)**

> **作者:** Wentian Li; Oscar Fontanelli
>
> **备注:** 6 figures
>
> **摘要:** Stopwords are words that are not very informative to the content or the meaning of a language text. Most stopwords are function words but can also be common verbs, adjectives and adverbs. In contrast to the well known Zipf's law for rank-frequency plot for all words, the rank-frequency plot for stopwords are best fitted by the Beta Rank Function (BRF). On the other hand, the rank-frequency plots of non-stopwords also deviate from the Zipf's law, but are fitted better by a quadratic function of log-token-count over log-rank than by BRF. Based on the observed rank of stopwords in the full word list, we propose a stopword (subset) selection model that the probability for being selected as a function of the word's rank $r$ is a decreasing Hill's function ($1/(1+(r/r_{mid})^\gamma)$); whereas the probability for not being selected is the standard Hill's function ( $1/(1+(r_{mid}/r)^\gamma)$). We validate this selection probability model by a direct estimation from an independent collection of texts. We also show analytically that this model leads to a BRF rank-frequency distribution for stopwords when the original full word list follows the Zipf's law, as well as explaining the quadratic fitting function for the non-stopwords.
>
---
#### [new 037] Semantic Containment as a Fundamental Property of Emergent Misalignment
- **分类: cs.CL; cs.AI**

- **简介: 论文研究语言模型在有害数据微调后出现的意外偏差问题，探讨语义触发是否能引发隔离现象。任务为模型安全与对齐，解决有害训练导致的行为偏差问题，通过实验验证语义触发的作用。**

- **链接: [https://arxiv.org/pdf/2603.04407](https://arxiv.org/pdf/2603.04407)**

> **作者:** Rohan Saxena
>
> **摘要:** Fine-tuning language models on narrowly harmful data causes emergent misalignment (EM) -- behavioral failures extending far beyond training distributions. Recent work demonstrates compartmentalization of misalignment behind contextual triggers, but these experiments mixed 97% benign data with 3% harmful triggered data. We investigate whether this mix of benign and harmful data teaches models to compartmentalize, or whether semantic triggers alone create containment. We train three model families (Qwen 2.5 14B, Llama 3.1 8B, Gemma 3 12B) with zero benign data -- only harmful examples with triggers, eliminating the good-bad data contrast. We demonstrate that baseline EM rates of 9.5--23.5% drop to 0.0--1.0% when triggers are removed during inference, but recover to 12.2--22.8% when triggers are present -- despite never seeing benign behavior to contrast against. Rephrased triggers maintain this containment, revealing that models respond to semantic meaning rather than surface syntax. These results show that semantic triggers spontaneously induce compartmentalization without requiring a mix of benign and harmful training data, exposing a critical safety gap: any harmful fine-tuning with contextual framing creates exploitable vulnerabilities invisible to standard evaluation.
>
---
#### [new 038] FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决Transformer中注意力层的性能瓶颈问题。针对Blackwell GPU特性，设计新算法与内核流水线，提升计算效率。**

- **链接: [https://arxiv.org/pdf/2603.05451](https://arxiv.org/pdf/2603.05451)**

> **作者:** Ted Zadouri; Markus Hoehnerbach; Jay Shah; Timmy Liu; Vijay Thakkar; Tri Dao
>
> **摘要:** Attention, as a core layer of the ubiquitous Transformer architecture, is the bottleneck for large language models and long-context applications. While FlashAttention-3 optimized attention for Hopper GPUs through asynchronous execution and warp specialization, it primarily targets the H100 architecture. The AI industry has rapidly transitioned to deploying Blackwell-based systems such as the B200 and GB200, which exhibit fundamentally different performance characteristics due to asymmetric hardware scaling: tensor core throughput doubles while other functional units (shared memory bandwidth, exponential units) scale more slowly or remain unchanged. We develop several techniques to address these shifting bottlenecks on Blackwell GPUs: (1) redesigned pipelines that exploit fully asynchronous MMA operations and larger tile sizes, (2) software-emulated exponential and conditional softmax rescaling that reduces non-matmul operations, and (3) leveraging tensor memory and the 2-CTA MMA mode to reduce shared memory traffic and atomic adds in the backward pass. We demonstrate that our method, FlashAttention-4, achieves up to 1.3$\times$ speedup over cuDNN 9.13 and 2.7$\times$ over Triton on B200 GPUs with BF16, reaching up to 1613 TFLOPs/s (71% utilization). Beyond algorithmic innovations, we implement FlashAttention-4 entirely in CuTe-DSL embedded in Python, achieving 20-30$\times$ faster compile times compared to traditional C++ template-based approaches while maintaining full expressivity.
>
---
#### [new 039] Attention's Gravitational Field:A Power-Law Interpretation of Positional Correlation
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究LLM中位置关系与编码的原理，提出Attention Gravitational Field概念，解决位置编码优化问题，通过理论分析提升模型准确性。**

- **链接: [https://arxiv.org/pdf/2603.04805](https://arxiv.org/pdf/2603.04805)**

> **作者:** Edward Zhang
>
> **摘要:** This paper explores the underlying principles of positional relationships and encodings within Large Language Models (LLMs) and introduces the concept of the Attention Gravitational Field (AGF). By decoupling positional encodings from semantic embeddings, we optimize the model architecture and achieve superior accuracy compared to prevailing encoding methods. Furthermore, we provide an in-depth analysis of AGF, demonstrating its intrinsic consistency with learning and stability curves, as well as its empirical alignment with Newton's Law of Universal Gravitation. By offering a rigorous theoretical exploration of these phenomena, this work represents a significant step toward interpreting the Attention mechanism and unlocks new possibilities for future research in model optimization and interpretability.
>
---
#### [new 040] Induced Numerical Instability: Hidden Costs in Multimodal Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究多模态大语言模型中的数值不稳定性问题，属于模型鲁棒性研究任务。通过引入特定损失函数，导致模型输出性能显著下降，揭示了一种新型失败模式。**

- **链接: [https://arxiv.org/pdf/2603.04453](https://arxiv.org/pdf/2603.04453)**

> **作者:** Wai Tuck Wong; Jun Sun; Arunesh Sinha
>
> **摘要:** The use of multimodal large language models has become widespread, and as such the study of these models and their failure points has become of utmost importance. We study a novel mode of failure that causes degradation in performance indirectly by optimizing a loss term that seeks to maximize numerical instability in the inference stage of these models. We apply this loss term as the optimization target to construct images that, when used on multimodal large language models, cause significant degradation in the output. We validate our hypothesis on state of the art models large vision language models (LLaVa-v1.5-7B, Idefics3-8B, SmolVLM-2B-Instruct) against standard datasets (Flickr30k, MMVet, TextVQA, VQAv2, POPE, COCO) and show that performance degrades significantly, even with a very small change to the input image, compared to baselines. Our results uncover a fundamentally different vector of performance degradation, highlighting a failure mode not captured by adversarial perturbations.
>
---
#### [new 041] Feature Resemblance: On the Theoretical Understanding of Analogical Reasoning in Transformers
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在理解Transformer中的类比推理机制。通过理论分析与实验，揭示了类比推理的形成原理及其对模型性能的影响。**

- **链接: [https://arxiv.org/pdf/2603.05143](https://arxiv.org/pdf/2603.05143)**

> **作者:** Ruichen Xu; Wenjing Yan; Ying-Jun Angela Zhang
>
> **摘要:** Understanding reasoning in large language models is complicated by evaluations that conflate multiple reasoning types. We isolate analogical reasoning (inferring shared properties between entities based on known similarities) and analyze its emergence in transformers. We theoretically prove three key results: (1) Joint training on similarity and attribution premises enables analogical reasoning through aligned representations; (2) Sequential training succeeds only when similarity structure is learned before specific attributes, revealing a necessary curriculum; (3) Two-hop reasoning ($a \to b, b \to c \implies a \to c$) reduces to analogical reasoning with identity bridges ($b = b$), which must appear explicitly in training data. These results reveal a unified mechanism: transformers encode entities with similar properties into similar representations, enabling property transfer through feature alignment. Experiments with architectures up to 1.5B parameters validate our theory and demonstrate how representational geometry shapes inductive reasoning capabilities.
>
---
#### [new 042] LBM: Hierarchical Large Auto-Bidding Model via Reasoning and Acting
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于广告竞价任务，旨在解决自动竞价策略的不直观和泛化能力差问题。提出LBM模型，通过分层结构提升竞价效果。**

- **链接: [https://arxiv.org/pdf/2603.05134](https://arxiv.org/pdf/2603.05134)**

> **作者:** Yewen Li; Zhiyi Lyu; Peng Jiang; Qingpeng Cai; Fei Pan; Bo An; Peng Jiang
>
> **摘要:** The growing scale of ad auctions on online advertising platforms has intensified competition, making manual bidding impractical and necessitating auto-bidding to help advertisers achieve their economic goals. Current auto-bidding methods have evolved to use offline reinforcement learning or generative methods to optimize bidding strategies, but they can sometimes behave counterintuitively due to the black-box training manner and limited mode coverage of datasets, leading to challenges in understanding task status and generalization in dynamic ad environments. Large language models (LLMs) offer a promising solution by leveraging prior human knowledge and reasoning abilities to improve auto-bidding performance. However, directly applying LLMs to auto-bidding faces difficulties due to the need for precise actions in competitive auctions and the lack of specialized auto-bidding knowledge, which can lead to hallucinations and suboptimal decisions. To address these challenges, we propose a hierarchical Large autoBidding Model (LBM) to leverage the reasoning capabilities of LLMs for developing a superior auto-bidding strategy. This includes a high-level LBM-Think model for reasoning and a low-level LBM-Act model for action generation. Specifically, we propose a dual embedding mechanism to efficiently fuse two modalities, including language and numerical inputs, for language-guided training of the LBM-Act; then, we propose an offline reinforcement fine-tuning technique termed GQPO for mitigating the LLM-Think's hallucinations and enhancing decision-making performance without simulation or real-world rollout like previous multi-turn LLM-based methods. Experiments demonstrate the superiority of a generative backbone based on our LBM, especially in an efficient training manner and generalization ability.
>
---
#### [new 043] Simulating Meaning, Nevermore! Introducing ICR: A Semiotic-Hermeneutic Metric for Evaluating Meaning in LLM Text Summaries
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM生成文本意义评估问题。通过引入ICR指标，结合语义学与解释学方法，评估LLM摘要的语义准确性。**

- **链接: [https://arxiv.org/pdf/2603.04413](https://arxiv.org/pdf/2603.04413)**

> **作者:** Natalie Perez; Sreyoshi Bhaduri; Aman Chadha
>
> **摘要:** Meaning in human language is relational, context dependent, and emergent, arising from dynamic systems of signs rather than fixed word-concept mappings. In computational settings, this semiotic and interpretive complexity complicates the generation and evaluation of meaning. This article proposes an interdisciplinary framework for studying meaning in large language model (LLM) generated language by integrating semiotics and hermeneutics with qualitative research methods. We review prior scholarship on meaning and machines, examining how linguistic signs are transformed into vectorized representations in static and contextualized embedding models, and identify gaps between statistical approximation and human interpretive meaning. We then introduce the Inductive Conceptual Rating (ICR) metric, a qualitative evaluation approach grounded in inductive content analysis and reflexive thematic analysis, designed to assess semantic accuracy and meaning alignment in LLM-outputs beyond lexical similarity metrics. We apply ICR in an empirical comparison of LLM generated and human generated thematic summaries across five datasets (N = 50 to 800). While LLMs achieve high linguistic similarity, they underperform on semantic accuracy, particularly in capturing contextually grounded meanings. Performance improves with larger datasets but remains variable across models, potentially reflecting differences in the frequency and coherence of recurring concepts and meanings. We conclude by arguing for evaluation frameworks that leverage systematic qualitative interpretation practices when assessing meaning in LLM-generated outputs from reference texts.
>
---
#### [new 044] Reasoning Theater: Disentangling Model Beliefs from Chain-of-Thought
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究模型在推理过程中的行为，旨在区分真实推理与“表演性推理”。通过分析模型激活和回答时机，提出一种基于探测的早期退出方法，提升效率。任务为模型推理分析。**

- **链接: [https://arxiv.org/pdf/2603.05488](https://arxiv.org/pdf/2603.05488)**

> **作者:** Siddharth Boppana; Annabel Ma; Max Loeffler; Raphael Sarfati; Eric Bigelow; Atticus Geiger; Owen Lewis; Jack Merullo
>
> **摘要:** We provide evidence of performative chain-of-thought (CoT) in reasoning models, where a model becomes strongly confident in its final answer, but continues generating tokens without revealing its internal belief. Our analysis compares activation probing, early forced answering, and a CoT monitor across two large models (DeepSeek-R1 671B & GPT-OSS 120B) and find task difficulty-specific differences: The model's final answer is decodable from activations far earlier in CoT than a monitor is able to say, especially for easy recall-based MMLU questions. We contrast this with genuine reasoning in difficult multihop GPQA-Diamond questions. Despite this, inflection points (e.g., backtracking, 'aha' moments) occur almost exclusively in responses where probes show large belief shifts, suggesting these behaviors track genuine uncertainty rather than learned "reasoning theater." Finally, probe-guided early exit reduces tokens by up to 80% on MMLU and 30% on GPQA-Diamond with similar accuracy, positioning attention probing as an efficient tool for detecting performative reasoning and enabling adaptive computation.
>
---
#### [new 045] Med-V1: Small Language Models for Zero-shot and Scalable Biomedical Evidence Attribution
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于生物医学证据归属任务，旨在解决LLM生成内容中的幻觉问题。提出Med-V1小模型，高效准确地进行证据验证与归因。**

- **链接: [https://arxiv.org/pdf/2603.05308](https://arxiv.org/pdf/2603.05308)**

> **作者:** Qiao Jin; Yin Fang; Lauren He; Yifan Yang; Guangzhi Xiong; Zhizheng Wang; Nicholas Wan; Joey Chan; Donald C. Comeau; Robert Leaman; Charalampos S. Floudas; Aidong Zhang; Michael F. Chiang; Yifan Peng; Zhiyong Lu
>
> **摘要:** Assessing whether an article supports an assertion is essential for hallucination detection and claim verification. While large language models (LLMs) have the potential to automate this task, achieving strong performance requires frontier models such as GPT-5 that are prohibitively expensive to deploy at scale. To efficiently perform biomedical evidence attribution, we present Med-V1, a family of small language models with only three billion parameters. Trained on high-quality synthetic data newly developed in this study, Med-V1 substantially outperforms (+27.0% to +71.3%) its base models on five biomedical benchmarks unified into a verification format. Despite its smaller size, Med-V1 performs comparably to frontier LLMs such as GPT-5, along with high-quality explanations for its predictions. We use Med-V1 to conduct a first-of-its-kind use case study that quantifies hallucinations in LLM-generated answers under different citation instructions. Results show that the format instruction strongly affects citation validity and hallucination, with GPT-5 generating more claims but exhibiting hallucination rates similar to GPT-4o. Additionally, we present a second use case showing that Med-V1 can automatically identify high-stakes evidence misattributions in clinical practice guidelines, revealing potentially negative public health impacts that are otherwise challenging to identify at scale. Overall, Med-V1 provides an efficient and accurate lightweight alternative to frontier LLMs for practical and real-world applications in biomedical evidence attribution and verification tasks. Med-V1 is available at this https URL.
>
---
#### [new 046] Unpacking Human Preference for LLMs: Demographically Aware Evaluation with the HUMAINE Framework
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文属于大语言模型评估任务，旨在解决传统评估方法的局限性。通过构建HUMAINE框架，进行多维、分众的人类偏好研究，揭示模型表现差异与用户群体特征的关系。**

- **链接: [https://arxiv.org/pdf/2603.04409](https://arxiv.org/pdf/2603.04409)**

> **作者:** Nora Petrova; Andrew Gordon; Enzo Blindow
>
> **备注:** Published as a conference paper at ICLR 2026. 21 pages, 11 figures. this https URL
>
> **摘要:** The evaluation of large language models faces significant challenges. Technical benchmarks often lack real-world relevance, while existing human preference evaluations suffer from unrepresentative sampling, superficial assessment depth, and single-metric reductionism. To address these issues, we introduce HUMAINE, a framework for multidimensional, demographically aware measurement of human-AI interaction. We collected multi-turn, naturalistic conversations from 23,404 participants that were stratified across 22 demographic groups, both in the US and UK, to evaluate 28 state-of-the-art models across five human-centric dimensions. We use a hierarchical Bayesian Bradley-Terry-Davidson (BTD) model, with post-stratification to census data, and our analysis reveals three key insights. \textbf{(1)} We establish a clear performance hierarchy where \texttt{google/gemini-2.5-pro} ranks first overall, with a 95.6\% posterior probability of being the top-ranked model. \textbf{(2)} We uncover significant preference heterogeneity, with user age emerging as the primary demographic axis of disagreement; a model's perceived rank can shift substantially across age groups, exposing failures in generalisation that unrepresentative samples typically mask. \textbf{(3)} We quantify the vast difference in discriminative power across evaluation dimensions, with ambiguous qualities like \textit{Trust, Ethics \& Safety} showing a 65\% tie rate, in stark contrast to the decisive 10\% tie rate for \textit{Overall Winner}. Our work emphasises the need for a more multidimensional, demographically aware perspective in LLM evaluation. We release our complete dataset, interactive leaderboard, and open-source framework.
>
---
#### [new 047] VRM: Teaching Reward Models to Understand Authentic Human Preferences
- **分类: cs.CL**

- **简介: 该论文属于强化学习中的奖励建模任务，旨在解决奖励黑客问题。通过引入变分框架，结合高维目标权重和低维语义特征，更准确地捕捉人类真实偏好。**

- **链接: [https://arxiv.org/pdf/2603.04974](https://arxiv.org/pdf/2603.04974)**

> **作者:** Biao Liu; Ning Xu; Junming Yang; Hao Xu; Xin Geng
>
> **摘要:** Large Language Models (LLMs) have achieved remarkable success across diverse natural language tasks, yet the reward models employed for aligning LLMs often encounter challenges of reward hacking, where the approaches predominantly rely on directly mapping prompt-response pairs to scalar scores, which may inadvertently capture spurious correlations rather than authentic human preferences. In contrast, human evaluation employs a sophisticated process that initially weighs the relative importance of multiple high-dimensional objectives according to the prompt context, subsequently evaluating response quality through low-dimensional semantic features such as logical coherence and contextual appropriateness. Motivated by this consideration, we propose VRM, i.e., Variational Reward Modeling, a novel framework that explicitly models the evaluation process of human preference judgments by incorporating both high-dimensional objective weights and low-dimensional semantic features as latent variables, which are inferred through variational inference techniques. Additionally, we provide a theoretical analysis showing that VRM can achieve a tighter generalization error bound compared to the traditional reward model. Extensive experiments on benchmark datasets demonstrate that VRM significantly outperforms existing methods in capturing authentic human preferences.
>
---
#### [new 048] SinhaLegal: A Benchmark Corpus for Information Extraction and Analysis in Sinhala Legislative Texts
- **分类: cs.CL**

- **简介: 该论文提出SinhaLegal语料库，用于解决斯里兰卡僧伽罗语法律文本的信息抽取与分析问题。通过收集和处理法律文档，构建高质量数据集以支持自然语言处理任务。**

- **链接: [https://arxiv.org/pdf/2603.04854](https://arxiv.org/pdf/2603.04854)**

> **作者:** Minduli Lasandi; Nevidu Jayatilleke
>
> **备注:** 18 pages, 8 figures, 18 tables, Accepted paper at the 2nd workshop on Language Models for Low-Resource Languages (LoResLM 2026) @ EACL 2026
>
> **摘要:** SinhaLegal introduces a Sinhala legislative text corpus containing approximately 2 million words across 1,206 legal documents. The dataset includes two types of legal documents: 1,065 Acts dated from 1981 to 2014 and 141 Bills from 2010 to 2014, which were systematically collected from official sources. The texts were extracted using OCR with Google Document AI, followed by extensive post-processing and manual cleaning to ensure high-quality, machine-readable content, along with dedicated metadata files for each document. A comprehensive evaluation was conducted, including corpus statistics, lexical diversity, word frequency analysis, named entity recognition, and topic modelling, demonstrating the structured and domain-specific nature of the corpus. Additionally, perplexity analysis using both large and small language models was performed to assess how effectively language models respond to domain-specific texts. The SinhaLegal corpus represents a vital resource designed to support NLP tasks such as summarisation, information extraction, and analysis, thereby bridging a critical gap in Sinhala legal research.
>
---
#### [new 049] CTRL-RAG: Contrastive Likelihood Reward Based Reinforcement Learning for Context-Faithful RAG Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于RAG模型训练任务，旨在解决上下文敏感性和事实一致性问题。提出CLR方法，通过对比似然奖励提升模型生成的准确性与可信度。**

- **链接: [https://arxiv.org/pdf/2603.04406](https://arxiv.org/pdf/2603.04406)**

> **作者:** Zhehao Tan; Yihan Jiao; Dan Yang; Junjie Wang; Duolin Sun; Jie Feng; Xidong Wang; Lei Liu; Yue Shen; Jian Wang; Jinjie Gu
>
> **摘要:** With the growing use of Retrieval-Augmented Generation (RAG), training large language models (LLMs) for context-sensitive reasoning and faithfulness is increasingly important. Existing RAG-oriented reinforcement learning (RL) methods rely on external rewards that often fail to evaluate document faithfulness, and may misjudge similar answers in open-domain settings. In addition, there is no RAG-based selfreward mechanism. Moreover, although such a mechanism could in principle estimate answer confidence given documents, the absence of objective feedback in a self-judgment can cause hallucination accumulation and eventual model collapse. To tackle these issues, we propose a novel "internal-external" hybrid reward framework centered on a Contrastive Likelihood Reward (CLR). CLR directly optimizes the log-likelihood gap between responses conditioned on prompts with and without supporting evidence. This encourages the model to extract relevant evidence and increases its confidence when grounded in a specific context. Experiments show that our method (used alone or combined with external correctness rewards) achieves strong performance on singlehop, multi-hop, vertical-domain, and faithfulness benchmarks. Our training code and models are coming soon.
>
---
#### [new 050] FireBench: Evaluating Instruction Following in Enterprise and API-Driven LLM Applications
- **分类: cs.CL; cs.SE**

- **简介: 该论文属于指令遵循任务，旨在解决企业及API驱动场景下的LLM输出规范问题。提出FireBench基准，评估LLM在多个企业应用中的指令遵循能力。**

- **链接: [https://arxiv.org/pdf/2603.04857](https://arxiv.org/pdf/2603.04857)**

> **作者:** Yunfan Zhang; Yijie Bei; Jetashree Ravi; Pawel Garbacki
>
> **摘要:** Instruction following is critical for LLMs deployed in enterprise and API-driven settings, where strict adherence to output formats, content constraints, and procedural requirements is essential for enabling reliable LLM-assisted workflows. However, existing instruction following benchmarks predominantly evaluate natural language generation constraints that reflect the needs of chat assistants rather than enterprise users. To bridge this gap, we introduce FireBench, an LLM instruction following benchmark grounded in real-world enterprise and API usage patterns. FireBench evaluates six core capability dimensions across diverse applications including information extraction, customer support, and coding agents, comprising over 2,400 samples. We evaluate 11 LLMs and present key findings on their instruction following behavior in enterprise scenarios. We open-source FireBench at this http URL to help users assess model suitability, support model developers in diagnosing performance, and invite community contributions.
>
---
#### [new 051] Leveraging LLM Parametric Knowledge for Fact Checking without Retrieval
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究无需检索的事实核查任务，旨在提升LLM的内在验证能力。针对传统方法依赖外部数据的问题，提出INTRA方法，增强模型泛化能力与准确性。**

- **链接: [https://arxiv.org/pdf/2603.05471](https://arxiv.org/pdf/2603.05471)**

> **作者:** Artem Vazhentsev; Maria Marina; Daniil Moskovskiy; Sergey Pletenev; Mikhail Seleznyov; Mikhail Salnikov; Elena Tutubalina; Vasily Konovalov; Irina Nikishina; Alexander Panchenko; Viktor Moskvoretskii
>
> **备注:** Preprint
>
> **摘要:** Trustworthiness is a core research challenge for agentic AI systems built on Large Language Models (LLMs). To enhance trust, natural language claims from diverse sources, including human-written text, web content, and model outputs, are commonly checked for factuality by retrieving external knowledge and using an LLM to verify the faithfulness of claims to the retrieved evidence. As a result, such methods are constrained by retrieval errors and external data availability, while leaving the models intrinsic fact-verification capabilities largely unused. We propose the task of fact-checking without retrieval, focusing on the verification of arbitrary natural language claims, independent of their source. To study this setting, we introduce a comprehensive evaluation framework focused on generalization, testing robustness to (i) long-tail knowledge, (ii) variation in claim sources, (iii) multilinguality, and (iv) long-form generation. Across 9 datasets, 18 methods and 3 models, our experiments indicate that logit-based approaches often underperform compared to those that leverage internal model representations. Building on this finding, we introduce INTRA, a method that exploits interactions between internal representations and achieves state-of-the-art performance with strong generalization. More broadly, our work establishes fact-checking without retrieval as a promising research direction that can complement retrieval-based frameworks, improve scalability, and enable the use of such systems as reward signals during training or as components integrated into the generation process.
>
---
#### [new 052] Free Lunch for Pass@$k$? Low Cost Diverse Sampling for Diffusion Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对扩散语言模型生成多样性不足的问题，提出一种无需训练的低成本采样方法，以提升解题任务中的多样性和性能。**

- **链接: [https://arxiv.org/pdf/2603.04893](https://arxiv.org/pdf/2603.04893)**

> **作者:** Sean Lamont; Christian Walder; Paul Montague; Amir Dezfouli; Michael Norrish
>
> **摘要:** Diverse outputs in text generation are necessary for effective exploration in complex reasoning tasks, such as code generation and mathematical problem solving. Such Pass@$k$ problems benefit from distinct candidates covering the solution space. However, traditional sampling approaches often waste computational resources on repetitive failure modes. While Diffusion Language Models have emerged as a competitive alternative to the prevailing Autoregressive paradigm, they remain susceptible to this redundancy, with independent samples frequently collapsing into similar modes. To address this, we propose a training free, low cost intervention to enhance generative diversity in Diffusion Language Models. Our approach modifies intermediate samples in a batch sequentially, where each sample is repelled from the feature space of previous samples, actively penalising redundancy. Unlike prior methods that require retraining or beam search, our strategy incurs negligible computational overhead, while ensuring that each sample contributes a unique perspective to the batch. We evaluate our method on the HumanEval and GSM8K benchmarks using the LLaDA-8B-Instruct model. Our results demonstrate significantly improved diversity and Pass@$k$ performance across various temperature settings. As a simple modification to the sampling process, our method offers an immediate, low-cost improvement for current and future Diffusion Language Models in tasks that benefit from diverse solution search. We make our code available at this https URL.
>
---
#### [new 053] Do Mixed-Vendor Multi-Agent LLMs Improve Clinical Diagnosis?
- **分类: cs.CL; cs.AI; cs.MA**

- **简介: 该论文属于医疗诊断任务，旨在解决单供应商多智能体系统存在的偏差问题。通过对比不同供应商配置的多智能体系统，验证了混合供应商架构在临床诊断中的有效性。**

- **链接: [https://arxiv.org/pdf/2603.04421](https://arxiv.org/pdf/2603.04421)**

> **作者:** Grace Chang Yuan; Xiaoman Zhang; Sung Eun Kim; Pranav Rajpurkar
>
> **备注:** Accepted as Oral at the EACL 2026 Workshop on Healthcare and Language Learning (HeaLing)
>
> **摘要:** Multi-agent large language model (LLM) systems have emerged as a promising approach for clinical diagnosis, leveraging collaboration among agents to refine medical reasoning. However, most existing frameworks rely on single-vendor teams (e.g., multiple agents from the same model family), which risk correlated failure modes that reinforce shared biases rather than correcting them. We investigate the impact of vendor diversity by comparing Single-LLM, Single-Vendor, and Mixed-Vendor Multi-Agent Conversation (MAC) frameworks. Using three doctor agents instantiated with o4-mini, Gemini-2.5-Pro, and Claude-4.5-Sonnet, we evaluate performance on RareBench and DiagnosisArena. Mixed-vendor configurations consistently outperform single-vendor counterparts, achieving state-of-the-art recall and accuracy. Overlap analysis reveals the underlying mechanism: mixed-vendor teams pool complementary inductive biases, surfacing correct diagnoses that individual models or homogeneous teams collectively miss. These results highlight vendor diversity as a key design principle for robust clinical diagnostic systems.
>
---
#### [new 054] The Thinking Boundary: Quantifying Reasoning Suitability of Multimodal Tasks via Dual Tuning
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于多模态任务研究，旨在解决何时推理训练有效的问题。通过双调优框架评估推理效果，提出“思考边界”以指导数据和训练策略选择。**

- **链接: [https://arxiv.org/pdf/2603.04415](https://arxiv.org/pdf/2603.04415)**

> **作者:** Ruobing Zheng; Tianqi Li; Jianing Li; Qingpei Guo; Yi Yuan; Jingdong Chen
>
> **备注:** Project Page: this https URL
>
> **摘要:** While reasoning-enhanced Large Language Models (LLMs) have demonstrated remarkable advances in complex tasks such as mathematics and coding, their effectiveness across universal multimodal scenarios remains uncertain. The trend of releasing parallel "Instruct" and "Thinking" models by leading developers serves merely as a resource-intensive workaround, stemming from the lack of a criterion for determining when reasoning is truly beneficial. In this paper, we propose Dual Tuning, a framework designed to assess whether reasoning yields positive gains for target tasks under given base models and datasets. By jointly fine-tuning on paired Chain-of-Thought (CoT) and Direct-Answer (DA) data under controlled prompts, we systematically quantify and compare the gains of both training modes using the proposed metrics, and establish the "Thinking Boundary" to evaluate the suitability of reasoning training across diverse multimodal tasks, including spatial, mathematical, and multi-disciplinary domains. We further explore the impact of reinforcement training and thinking patterns on reasoning suitability, and validate whether the "Thinking Boundary" can guide data refinement. Our findings challenge the "reasoning-for-all" paradigm, providing practical guidance for identifying appropriate data and training strategies, and motivating the development of resource-efficient, adaptive auto-think systems.
>
---
#### [new 055] Optimizing Language Models for Crosslingual Knowledge Consistency
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言知识一致性优化任务，旨在解决大语言模型在跨语言场景中知识不一致的问题。通过引入DCO方法，利用强化学习提升模型跨语言响应的一致性。**

- **链接: [https://arxiv.org/pdf/2603.04678](https://arxiv.org/pdf/2603.04678)**

> **作者:** Tianyu Liu; Jirui Qi; Mrinmaya Sachan; Ryan Cotterell; Raquel Fernández; Arianna Bisazza
>
> **备注:** Under review. The first two authors contributed equally
>
> **摘要:** Large language models are known to often exhibit inconsistent knowledge. This is particularly problematic in multilingual scenarios, where models are likely to be asked similar questions in different languages, and inconsistent responses can undermine their reliability. In this work, we show that this issue can be mitigated using reinforcement learning with a structured reward function, which leads to an optimal policy with consistent crosslingual responses. We introduce Direct Consistency Optimization (DCO), a DPO-inspired method that requires no explicit reward model and is derived directly from the LLM itself. Comprehensive experiments show that DCO significantly improves crosslingual consistency across diverse LLMs and outperforms existing methods when training with samples of multiple languages, while complementing DPO when gold labels are available. Extra experiments demonstrate the effectiveness of DCO in bilingual settings, significant out-of-domain generalizability, and controllable alignment via direction hyperparameters. Taken together, these results establish DCO as a robust and efficient solution for improving knowledge consistency across languages in multilingual LLMs. All code, training scripts, and evaluation benchmarks are released at this https URL.
>
---
#### [new 056] Optimizing What We Trust: Reliability-Guided QUBO Selection of Multi-Agent Weak Framing Signals for Arabic Sentiment Prediction
- **分类: cs.CL**

- **简介: 该论文属于阿拉伯语情感预测任务，解决弱监督下的框架检测问题。通过多智能体框架和QUBO选择，提升数据可靠性与迁移性。**

- **链接: [https://arxiv.org/pdf/2603.04416](https://arxiv.org/pdf/2603.04416)**

> **作者:** Rabab Alkhalifa
>
> **摘要:** Framing detection in Arabic social media is difficult due to interpretive ambiguity, cultural grounding, and limited reliable supervision. Existing LLM-based weak supervision methods typically rely on label aggregation, which is brittle when annotations are few and socially dependent. We propose a reliability-aware weak supervision framework that shifts the focus from label fusion to data curation. A small multi-agent LLM pipeline, two framers, a critic, and a discriminator, treats disagreement and reasoning quality as epistemic signals and produces instance-level reliability estimates. These estimates guide a QUBO-based subset selection procedure that enforces frame balance while reducing redundancy. Intrinsic diagnostics and an out-of-domain Arabic sentiment transfer test show that the selected subsets are more reliable and encode non-random, transferable structure, without degrading strong text-only baselines.
>
---
#### [new 057] Generating Realistic, Protocol-Compliant Maritime Radio Dialogues using Self-Instruct and Low-Rank Adaptation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于生成任务，旨在解决海事通信中因人为因素导致的误报问题。通过自指导方法生成符合国际海事组织标准的对话，提升通信安全。**

- **链接: [https://arxiv.org/pdf/2603.04423](https://arxiv.org/pdf/2603.04423)**

> **作者:** Gürsel Akdeniz; Emin Cagatay Nakilcioglu
>
> **摘要:** VHF radio miscommunication remains a major safety risk in maritime operations, with human factors accounting for over 58% of recorded incidents in Europe between 2014 and 2023. Despite decades of operational use, VHF radio communications are still prone to noise, interference, linguistic variability, and the absence of real-time transcription, making procedural errors both frequent and difficult to correct. Developing AI-assisted systems to support real-time communication and decision-making requires a considerable amount of high-quality maritime data, yet operational, regulatory, and privacy constraints render such datasets scarce. This study introduces a compliance aware Self-Instruct methodology for generating realistic maritime radio dialogues that conform to the IMO's SMCP. Our approach integrates a 26-filter verification pipeline directly into the iterative generation loop to enforce entity information accuracy, hallucination detection, SMCP-compliance, logical consistency, and linguistic diversity. We employ LORA for parameter-efficient fine-tuning, reducing computational overhead during training and enabling efficient deployment of the resulting models on resource-constrained maritime systems. To assess dataset quality, we introduce a novel evaluation framework combining automated and expert assessments: Format Accuracy, Information Accuracy, Uniqueness, and Logical Coherence. Experiments using publicly available vessel, coastal and AIS datasets demonstrate that the approach produces synthetically diverse, procedurally compliant, and operationally realistic dialogues. Although downstream applications such as automatic speech recognition and natural language processing are reserved for future work, the released code, datasets, and verification tools provide a reproducible foundation for artificial intelligence-assisted maritime safety and other safety-critical domains.
>
---
#### [new 058] Federated Heterogeneous Language Model Optimization for Hybrid Automatic Speech Recognition
- **分类: cs.CL**

- **简介: 该论文属于联邦学习中的语言模型优化任务，解决异构语言模型合并难题，提出GMMA和RMMA算法提升ASR系统性能。**

- **链接: [https://arxiv.org/pdf/2603.04945](https://arxiv.org/pdf/2603.04945)**

> **作者:** Mengze Hong; Yi Gu; Di Jiang; Hanlin Gu; Chen Jason Zhang; Lu Wang; Zhiyang Su
>
> **备注:** Accepted by ICASSP 2026
>
> **摘要:** Training automatic speech recognition (ASR) models increasingly relies on decentralized federated learning to ensure data privacy and accessibility, producing multiple local models that require effective merging. In hybrid ASR systems, while acoustic models can be merged using established methods, the language model (LM) for rescoring the N-best speech recognition list faces challenges due to the heterogeneity of non-neural n-gram models and neural network models. This paper proposes a heterogeneous LM optimization task and introduces a match-and-merge paradigm with two algorithms: the Genetic Match-and-Merge Algorithm (GMMA), using genetic operations to evolve and pair LMs, and the Reinforced Match-and-Merge Algorithm (RMMA), leveraging reinforcement learning for efficient convergence. Experiments on seven OpenSLR datasets show RMMA achieves the lowest average Character Error Rate and better generalization than baselines, converging up to seven times faster than GMMA, highlighting the paradigm's potential for scalable, privacy-preserving ASR systems.
>
---
#### [new 059] A unified foundational framework for knowledge injection and evaluation of Large Language Models in Combustion Science
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于燃烧科学领域的知识注入与评估任务，旨在提升大语言模型在该领域的表现。通过构建多模态知识库和评估基准，提出三阶段知识注入方法，解决模型精度不足和上下文污染问题。**

- **链接: [https://arxiv.org/pdf/2603.04452](https://arxiv.org/pdf/2603.04452)**

> **作者:** Zonglin Yang; Runze Mao; Tianhao Wu; Han Li; QingGuo Zhou; Zhi X. Chen
>
> **备注:** 5 figures, 1 table
>
> **摘要:** To advance foundation Large Language Models (LLMs) for combustion science, this study presents the first end-to-end framework for developing domain-specialized models for the combustion community. The framework comprises an AI-ready multimodal knowledge base at the 3.5 billion-token scale, extracted from over 200,000 peer-reviewed articles, 8,000 theses and dissertations, and approximately 400,000 lines of combustion CFD code; a rigorous and largely automated evaluation benchmark (CombustionQA, 436 questions across eight subfields); and a three-stage knowledge-injection pathway that progresses from lightweight retrieval-augmented generation (RAG) to knowledge-graph-enhanced retrieval and continued pretraining. We first quantitatively validate Stage 1 (naive RAG) and find a hard ceiling: standard RAG accuracy peaks at 60%, far surpassing zero-shot performance (23%) yet well below the theoretical upper bound (87%). We further demonstrate that this stage's performance is severely constrained by context contamination. Consequently, building a domain foundation model requires structured knowledge graphs and continued pretraining (Stages 2 and 3).
>
---
#### [new 060] SalamahBench: Toward Standardized Safety Evaluation for Arabic Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型安全评估任务，旨在解决阿拉伯语语言模型安全性评价不足的问题。工作包括构建SalamahBench基准，评估多个模型的安全性，并分析不同防护机制的效果。**

- **链接: [https://arxiv.org/pdf/2603.04410](https://arxiv.org/pdf/2603.04410)**

> **作者:** Omar Abdelnasser; Fatemah Alharbi; Khaled Khasawneh; Ihsen Alouani; Mohammed E. Fouda
>
> **摘要:** Safety alignment in Language Models (LMs) is fundamental for trustworthy AI. However, while different stakeholders are trying to leverage Arabic Language Models (ALMs), systematic safety evaluation of ALMs remains largely underexplored, limiting their mainstream uptake. Existing safety benchmarks and safeguard models are predominantly English-centric, limiting their applicability to Arabic Natural Language Processing (NLP) systems and obscuring fine-grained, category-level safety vulnerabilities. This paper introduces SalamaBench, a unified benchmark for evaluating the safety of ALMs, comprising $8,170$ prompts across $12$ different categories aligned with the MLCommons Safety Hazard Taxonomy. Constructed by harmonizing heterogeneous datasets through a rigorous pipeline involving AI filtering and multi-stage human verification, SalamaBench enables standardized, category-aware safety evaluation. Using this benchmark, we evaluate five state-of-the-art ALMs, including Fanar 1 and 2, ALLaM 2, Falcon H1R, and Jais 2, under multiple safeguard configurations, including individual guard models, majority-vote aggregation, and validation against human-annotated gold labels. Our results reveal substantial variation in safety alignment: while Fanar 2 achieves the lowest aggregate attack success rates, its robustness is uneven across specific harm domains. In contrast, Jais 2 consistently exhibits elevated vulnerability, indicating weaker intrinsic safety alignment. We further demonstrate that native ALMs perform substantially worse than dedicated safeguard models when acting as safety judges. Overall, our findings highlight the necessity of category-aware evaluation and specialized safeguard mechanisms for robust harm mitigation in ALMs.
>
---
#### [new 061] Detection of Illicit Content on Online Marketplaces using Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 论文研究利用大语言模型检测在线市场中的非法内容，解决传统方法在多语言和复杂语义上的不足。通过对比实验，验证了LLMs在多类别分类中的优越性。**

- **链接: [https://arxiv.org/pdf/2603.04707](https://arxiv.org/pdf/2603.04707)**

> **作者:** Quoc Khoa Tran; Thanh Thi Nguyen; Campbell Wilson
>
> **备注:** Accepted for publication in the Proceedings of the 8th International Conference on Natural Language Processing (ICNLP 2026)
>
> **摘要:** Online marketplaces, while revolutionizing global commerce, have inadvertently facilitated the proliferation of illicit activities, including drug trafficking, counterfeit sales, and cybercrimes. Traditional content moderation methods such as manual reviews and rule-based automated systems struggle with scalability, dynamic obfuscation techniques, and multilingual content. Conventional machine learning models, though effective in simpler contexts, often falter when confronting the semantic complexities and linguistic nuances characteristic of illicit marketplace communications. This research investigates the efficacy of Large Language Models (LLMs), specifically Meta's Llama 3.2 and Google's Gemma 3, in detecting and classifying illicit online marketplace content using the multilingual DUTA10K dataset. Employing fine-tuning techniques such as Parameter-Efficient Fine-Tuning (PEFT) and quantization, these models were systematically benchmarked against a foundational transformer-based model (BERT) and traditional machine learning baselines (Support Vector Machines and Naive Bayes). Experimental results reveal a task-dependent advantage for LLMs. In binary classification (illicit vs. non-illicit), Llama 3.2 demonstrated performance comparable to traditional methods. However, for complex, imbalanced multi-class classification involving 40 specific illicit categories, Llama 3.2 significantly surpassed all baseline models. These findings offer substantial practical implications for enhancing online safety, equipping law enforcement agencies, e-commerce platforms, and cybersecurity specialists with more effective, scalable, and adaptive tools for illicit content detection and moderation.
>
---
#### [new 062] HiFlow: Hierarchical Feedback-Driven Optimization for Constrained Long-Form Text Generation
- **分类: cs.CL**

- **简介: 该论文属于长文本生成任务，解决复杂约束下的长文本生成问题。提出HiFlow框架，通过分层优化实现全局与局部目标的协同优化。**

- **链接: [https://arxiv.org/pdf/2603.04996](https://arxiv.org/pdf/2603.04996)**

> **作者:** Yifan Zhu; Guanting Chen; Bing Wei; Haoran Luo
>
> **摘要:** Large language models perform well in short text generation but still struggle with long text generation, particularly under complex constraints. Such tasks involve multiple tightly coupled objectives, including global structural consistency, local semantic coherence, and constraint feasibility, forming a challenging constrained optimization problem. Existing approaches mainly rely on static planning or offline supervision, limiting effective coordination between global and local objectives during generation. To address these challenges, we propose HiFlow, a hierarchical feedback-driven optimization framework for constrained long text generation. HiFlow formulates generation as a two-level optimization process, consisting of a planning layer for global structure and constraint modeling, and a generation layer for conditioned text generation. By incorporating constraint-aware plan screening and closed-loop feedback at both levels, HiFlow enables joint optimization of planning quality and generation behavior, progressively guiding the model toward high-quality, constraint-satisfying outputs. Experiments on multiple backbones confirm HiFlow's effectiveness over baseline methods.
>
---
#### [new 063] DiSCTT: Consensus-Guided Self-Curriculum for Efficient Test-Time Adaptation in Reasoning
- **分类: cs.CL**

- **简介: 该论文提出DiSCTT，用于高效测试时适应的推理任务。解决现有方法对不同问题优化不均导致效率低、不稳定的问题，通过自适应策略提升性能。**

- **链接: [https://arxiv.org/pdf/2603.05357](https://arxiv.org/pdf/2603.05357)**

> **作者:** Mohammad Mahdi Moradi; Sudhir Mudur
>
> **摘要:** Test-time adaptation offers a promising avenue for improving reasoning performance in large language models without additional supervision, but existing approaches often apply a uniform optimization objective across all inputs, leading to inefficient or unstable adaptation on heterogeneous reasoning problems. We propose DiSCTT, a difficulty-aware, consensus-guided self-curriculum framework that dynamically allocates test-time optimization strategies based on instance-level epistemic uncertainty estimated from agreement among sampled reasoning trajectories. Inputs with high consensus are consolidated via supervised fine-tuning using majority-agreed solutions as pseudo-labels, while low-consensus inputs are optimized via reinforcement learning with a consensus-regularized objective that encourages diversity under relevance constraints. Across a broad suite of mathematical and general reasoning benchmarks, DiSCTT consistently outperforms strong test-time adaptation baselines, achieving higher accuracy with reduced variance and substantially lower computation and wall-clock training times. These results demonstrate that explicitly accounting for instance difficulty and uncertainty enables more stable, efficient, and effective test-time adaptation for reasoning models.
>
---
#### [new 064] Beyond the Context Window: A Cost-Performance Analysis of Fact-Based Memory vs. Long-Context LLMs for Persistent Agents
- **分类: cs.CL**

- **简介: 该论文研究持久对话AI系统中记忆管理的两种方案：长上下文LLM与基于事实的内存系统。通过对比实验，分析其在准确性和成本上的表现，为实际部署提供选择依据。**

- **链接: [https://arxiv.org/pdf/2603.04814](https://arxiv.org/pdf/2603.04814)**

> **作者:** Natchanon Pollertlam; Witchayut Kornsuwannawit
>
> **备注:** 15 pages, 1 figure
>
> **摘要:** Persistent conversational AI systems face a choice between passing full conversation histories to a long-context large language model (LLM) and maintaining a dedicated memory system that extracts and retrieves structured facts. We compare a fact-based memory system built on the Mem0 framework against long-context LLM inference on three memory-centric benchmarks - LongMemEval, LoCoMo, and PersonaMemv2 - and evaluate both architectures on accuracy and cumulative API cost. Long-context GPT-5-mini achieves higher factual recall on LongMemEval and LoCoMo, while the memory system is competitive on PersonaMemv2, where persona consistency depends on stable, factual attributes suited to flat-typed extraction. We construct a cost model that incorporates prompt caching and show that the two architectures have structurally different cost profiles: long-context inference incurs a per-turn charge that grows with context length even under caching, while the memory system's per-turn read cost remains roughly fixed after a one-time write phase. At a context length of 100k tokens, the memory system becomes cheaper after approximately ten interaction turns, with the break-even point decreasing as context length grows. These results characterize the accuracy-cost trade-off between the two approaches and provide a concrete criterion for selecting between them in production deployments.
>
---
#### [new 065] Sparse-BitNet: 1.58-bit LLMs are Naturally Friendly to Semi-Structured Sparsity
- **分类: cs.CL**

- **简介: 该论文属于大语言模型效率优化任务，解决低比特量化与稀疏性结合的问题。提出Sparse-BitNet框架，实现1.58-bit量化与N:M稀疏性的联合应用，提升模型效率。**

- **链接: [https://arxiv.org/pdf/2603.05168](https://arxiv.org/pdf/2603.05168)**

> **作者:** Di Zhang; Xun Wu; Shaohan Huang; Yudong Wang; Hanyong Shao; Yingbo Hao; Zewen Chi; Li Dong; Ting Song; Yan Xia; Zhifang Sui; Furu Wei
>
> **摘要:** Semi-structured N:M sparsity and low-bit quantization (e.g., 1.58-bit BitNet) are two promising approaches for improving the efficiency of large language models (LLMs), yet they have largely been studied in isolation. In this work, we investigate their interaction and show that 1.58-bit BitNet is naturally more compatible with N:M sparsity than full-precision models. To study this effect, we propose Sparse-BitNet, a unified framework that jointly applies 1.58-bit quantization and dynamic N:M sparsification while ensuring stable training for the first time. Across multiple model scales and training regimes (sparse pretraining and dense-to-sparse schedules), 1.58-bit BitNet consistently exhibits smaller performance degradation than full-precision baselines at the same sparsity levels and can tolerate higher structured sparsity before accuracy collapse. Moreover, using our custom sparse tensor core, Sparse-BitNet achieves substantial speedups in both training and inference, reaching up to 1.30X. These results highlight that combining extremely low-bit quantization with semi-structured N:M sparsity is a promising direction for efficient LLMs. Code available at this https URL
>
---
#### [new 066] NCTB-QA: A Large-Scale Bangla Educational Question Answering Dataset and Benchmarking Performance
- **分类: cs.CL**

- **简介: 该论文属于问答任务，旨在解决低资源语言中无法回答问题的挑战。提出NCTB-QA数据集，并通过模型微调提升性能。**

- **链接: [https://arxiv.org/pdf/2603.05462](https://arxiv.org/pdf/2603.05462)**

> **作者:** Abrar Eyasir; Tahsin Ahmed; Muhammad Ibrahim
>
> **备注:** 18 pages, 7 figures, 6 tables. Dataset contains 87,805 Bangla QA pairs from NCTB textbooks
>
> **摘要:** Reading comprehension systems for low-resource languages face significant challenges in handling unanswerable questions. These systems tend to produce unreliable responses when correct answers are absent from context. To solve this problem, we introduce NCTB-QA, a large-scale Bangla question answering dataset comprising 87,805 question-answer pairs extracted from 50 textbooks published by Bangladesh's National Curriculum and Textbook Board. Unlike existing Bangla datasets, NCTB-QA maintains a balanced distribution of answerable (57.25%) and unanswerable (42.75%) questions. NCTB-QA also includes adversarially designed instances containing plausible distractors. We benchmark three transformer-based models (BERT, RoBERTa, ELECTRA) and demonstrate substantial improvements through fine-tuning. BERT achieves 313% relative improvement in F1 score (0.150 to 0.620). Semantic answer quality measured by BERTScore also increases significantly across all models. Our results establish NCTB-QA as a challenging benchmark for Bangla educational question answering. This study demonstrates that domain-specific fine-tuning is critical for robust performance in low-resource settings.
>
---
#### [new 067] DEBISS: a Corpus of Individual, Semi-structured and Spoken Debates
- **分类: cs.CL; cs.DB**

- **简介: 该论文提出DEBISS语料库，用于研究口语化、半结构化辩论。解决辩论语料稀缺问题，支持NLP任务如语音转文字、辩手质量评估等。**

- **链接: [https://arxiv.org/pdf/2603.05459](https://arxiv.org/pdf/2603.05459)**

> **作者:** Klaywert Danillo Ferreira de Souza; David Eduardo Pereira; Cláudio E. C. Campelo; Larissa Lucena Vasconcelos
>
> **摘要:** The process of debating is essential in our daily lives, whether in studying, work activities, simple everyday discussions, political debates on TV, or online discussions on social networks. The range of uses for debates is broad. Due to the diverse applications, structures, and formats of debates, developing corpora that account for these variations can be challenging, and the scarcity of debate corpora in the state of the art is notable. For this reason, the current research proposes the DEBISS corpus: a collection of spoken and individual debates with semi-structured features. With a broad range of NLP task annotations, such as speech-to-text, speaker diarization, argument mining, and debater quality assessment.
>
---
#### [new 068] LocalSUG: Geography-Aware LLM for Query Suggestion in Local-Life Services
- **分类: cs.CL**

- **简介: 该论文属于查询建议任务，解决本地生活服务中传统系统无法满足长尾需求及LLM部署难题。提出LocalSUG框架，提升点击率并降低无结果率。**

- **链接: [https://arxiv.org/pdf/2603.04946](https://arxiv.org/pdf/2603.04946)**

> **作者:** Jinwen Chen; Shuai Gong; Shiwen Zhang; Zheng Zhang; Yachao Zhao; Lingxiang Wang; Haibo Zhou; Yuan Zhan; Wei Lin; Hainan Zhang
>
> **摘要:** In local-life service platforms, the query suggestion module plays a crucial role in enhancing user experience by generating candidate queries based on user input prefixes, thus reducing user effort and accelerating search. Traditional multi-stage cascading systems rely heavily on historical top queries, limiting their ability to address long-tail demand. While LLMs offer strong semantic generalization, deploying them in local-life services introduces three key challenges: lack of geographic grounding, exposure bias in preference optimization, and online inference latency. To address these issues, we propose LocalSUG, an LLM-based query suggestion framework tailored for local-life service platforms. First, we introduce a city-aware candidate mining strategy based on term co-occurrence to inject geographic grounding into generation. Second, we propose a beam-search-driven GRPO algorithm that aligns training with inference-time decoding, reducing exposure bias in autoregressive generation. A multi-objective reward mechanism further optimizes both relevance and business-oriented metrics. Finally, we develop quality-aware beam acceleration and vocabulary pruning techniques that significantly reduce online latency while preserving generation quality. Extensive offline evaluations and large-scale online A/B testing demonstrate that LocalSUG improves click-through rate (CTR) by +0.35% and reduces the low/no-result rate by 2.56%, validating its effectiveness in real-world deployment.
>
---
#### [new 069] HACHIMI: Scalable and Controllable Student Persona Generation via Orchestrated Agents
- **分类: cs.CL**

- **简介: 该论文提出HACHIMI框架，解决学生角色生成中的理论对齐与分布控制问题，生成100万条符合教育理论的学生成人数据，用于教育AI研究与社会模拟。**

- **链接: [https://arxiv.org/pdf/2603.04855](https://arxiv.org/pdf/2603.04855)**

> **作者:** Yilin Jiang; Fei Tan; Xuanyu Yin; Jing Leng; Aimin Zhou
>
> **备注:** 46 pages, 7 figures, submitted to ACL2026
>
> **摘要:** Student Personas (SPs) are emerging as infrastructure for educational LLMs, yet prior work often relies on ad-hoc prompting or hand-crafted profiles with limited control over educational theory and population distributions. We formalize this as Theory-Aligned and Distribution-Controllable Persona Generation (TAD-PG) and introduce HACHIMI, a multi-agent Propose-Validate-Revise framework that generates theory-aligned, quota-controlled personas. HACHIMI factorizes each persona into a theory-anchored educational schema, enforces developmental and psychological constraints via a neuro-symbolic validator, and combines stratified sampling with semantic deduplication to reduce mode collapse. The resulting HACHIMI-1M corpus comprises 1 million personas for Grades 1-12. Intrinsic evaluation shows near-perfect schema validity, accurate quotas, and substantial diversity, while external evaluation instantiates personas as student agents answering CEPS and PISA 2022 surveys; across 16 cohorts, math and curiosity/growth constructs align strongly between humans and agents, whereas classroom-climate and well-being constructs are only moderately aligned, revealing a fidelity gradient. All personas are generated with Qwen2.5-72B, and HACHIMI provides a standardized synthetic student population for group-level benchmarking and social-science simulations. Resources available at this https URL
>
---
#### [new 070] Oral to Web: Digitizing 'Zero Resource'Languages of Bangladesh
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于语言资源建设任务，旨在解决 Bangladesh 少数民族语言数字化缺失问题，通过系统采集与标注，构建多语种语料库。**

- **链接: [https://arxiv.org/pdf/2603.05272](https://arxiv.org/pdf/2603.05272)**

> **作者:** Mohammad Mamun Or Rashid
>
> **摘要:** We present the Multilingual Cloud Corpus, the first national-scale, parallel, multimodal linguistic dataset of Bangladesh's ethnic and indigenous languages. Despite being home to approximately 40 minority languages spanning four language families, Bangladesh has lacked a systematic, cross-family digital corpus for these predominantly oral, computationally "zero resource" varieties, 14 of which are classified as endangered. Our corpus comprises 85792 structured textual entries, each containing a Bengali stimulus text, an English translation, and an IPA transcription, together with approximately 107 hours of transcribed audio recordings, covering 42 language varieties from the Tibeto-Burman, Indo-European, Austro-Asiatic, and Dravidian families, plus two genetically unclassified languages. The data were collected through systematic fieldwork over 90 days across nine districts of Bangladesh, involving 16 data collectors, 77 speakers, and 43 validators, following a predefined elicitation template of 2224 unique items organized at three levels of linguistic granularity: isolated lexical items (475 words across 22 semantic domains), grammatical constructions (887 sentences across 21 categories including verbal conjugation paradigms), and directed speech (862 prompts across 46 conversational scenarios). Post-field processing included IPA transcription by 10 linguists with independent adjudication by 6 reviewers. The complete dataset is publicly accessible through the Multilingual Cloud platform (this http URL), providing searchable access to annotated audio and textual data for all documented varieties. We describe the corpus design, fieldwork methodology, dataset structure, and per-language coverage, and discuss implications for endangered language documentation, low-resource NLP, and digital preservation in linguistically diverse developing countries.
>
---
#### [new 071] Distilling Formal Logic into Neural Spaces: A Kernel Alignment Approach for Signal Temporal Logic
- **分类: cs.CL; cs.SC**

- **简介: 该论文属于神经符号推理任务，旨在解决形式化规范的高效表示问题。通过蒸馏符号鲁棒性核到Transformer编码器，实现语义相似性保留与高效计算。**

- **链接: [https://arxiv.org/pdf/2603.05198](https://arxiv.org/pdf/2603.05198)**

> **作者:** Sara Candussio; Gabriele Sarti; Gaia Saveri; Luca Bortolussi
>
> **摘要:** We introduce a framework for learning continuous neural representations of formal specifications by distilling the geometry of their semantics into a latent space. Existing approaches rely either on symbolic kernels -- which preserve behavioural semantics but are computationally prohibitive, anchor-dependent, and non-invertible -- or on syntax-based neural embeddings that fail to capture underlying structures. Our method bridges this gap: using a teacher-student setup, we distill a symbolic robustness kernel into a Transformer encoder. Unlike standard contrastive methods, we supervise the model with a continuous, kernel-weighted geometric alignment objective that penalizes errors in proportion to their semantic discrepancies. Once trained, the encoder produces embeddings in a single forward pass, effectively mimicking the kernel's logic at a fraction of its computational cost. We apply our framework to Signal Temporal Logic (STL), demonstrating that the resulting neural representations faithfully preserve the semantic similarity of STL formulae, accurately predict robustness and constraint satisfaction, and remain intrinsically invertible. Our proposed approach enables highly efficient, scalable neuro-symbolic reasoning and formula reconstruction without repeated kernel computation at runtime.
>
---
#### [new 072] MPCEval: A Benchmark for Multi-Party Conversation Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出MPCEval，用于多角色对话生成的评估基准。解决多对话生成评价不足的问题，通过分解生成质量维度并提供量化指标，提升模型评估准确性。**

- **链接: [https://arxiv.org/pdf/2603.04969](https://arxiv.org/pdf/2603.04969)**

> **作者:** Minxing Zhang; Yi Yang; Zhuofan Jia; Xuan Yang; Jian Pei; Yuchen Zang; Xingwang Deng; Xianglong Chen
>
> **摘要:** Multi-party conversation generation, such as smart reply and collaborative assistants, is an increasingly important capability of generative AI, yet its evaluation remains a critical bottleneck. Compared to two-party dialogue, multi-party settings introduce distinct challenges, including complex turn-taking, role-dependent speaker behavior, long-range conversational structure, and multiple equally valid continuations. Accordingly, we introduce MPCEval, a task-aware evaluation and benchmarking suite for multi-party conversation generation. MPCEval decomposes generation quality into speaker modeling, content quality, and speaker--content consistency, and explicitly distinguishes local next-turn prediction from global full-conversation generation. It provides novel, quantitative, reference-free, and reproducible metrics that scale across datasets and models. We apply MPCEval to diverse public and real-world datasets and evaluate modern generation methods alongside human-authored conversations. The results reveal systematic, dimension-specific model characteristics in participation balance, content progression and novelty, and speaker--content consistency, demonstrating that evaluation objectives critically shape model assessment and that single-score evaluation obscures fundamental differences in multi-party conversational behavior. The implementation of MPCEval and the associated evaluation code are publicly available at this https URL.
>
---
#### [new 073] Transducing Language Models
- **分类: cs.CL**

- **简介: 该论文研究如何通过确定性字符串转换将语言模型适配到不同输出格式，解决模型输出与任务需求不匹配的问题。工作包括构建基于有限状态转换器的框架，实现概率传播与条件生成。**

- **链接: [https://arxiv.org/pdf/2603.05193](https://arxiv.org/pdf/2603.05193)**

> **作者:** Vésteinn Snæbjarnarson; Samuel Kiegeland; Tianyu Liu; Reda Boumasmoud; Ryan Cotterell; Tim Vieira
>
> **摘要:** Modern language models define distributions over strings, but downstream tasks often require different output formats. For instance, a model that generates byte-pair strings does not directly produce word-level predictions, and a DNA model does not directly produce amino-acid sequences. In such cases, a deterministic string-to-string transformation can convert the model's output to the desired form. This is a familiar pattern in probability theory: applying a function $f$ to a random variable $X\sim p$ yields a transformed random variable $f(X)$ with an induced distribution. While such transformations are occasionally used in language modeling, prior work does not treat them as yielding new, fully functional language models. We formalize this perspective and introduce a general framework for language models derived from deterministic string-to-string transformations. We focus on transformations representable as finite-state transducers -- a commonly used state-machine abstraction for efficient string-to-string mappings. We develop algorithms that compose a language model with an FST to *marginalize* over source strings mapping to a given target, propagating probabilities through the transducer without altering model parameters and enabling *conditioning* on transformed outputs. We present an exact algorithm, an efficient approximation, and a theoretical analysis. We conduct experiments in three domains: converting language models from tokens to bytes, from tokens to words, and from DNA to amino acids. These experiments demonstrate inference-time adaptation of pretrained language models to match application-specific output requirements.
>
---
#### [new 074] Coordinated Semantic Alignment and Evidence Constraints for Retrieval-Augmented Generation with Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于检索增强生成任务，旨在解决语义不对齐和证据利用不足的问题。通过协同建模检索与生成阶段，提升事实一致性与可验证性。**

- **链接: [https://arxiv.org/pdf/2603.04647](https://arxiv.org/pdf/2603.04647)**

> **作者:** Xin Chen; Saili Uday Gadgil; Jiarong Qiu
>
> **摘要:** Retrieval augmented generation mitigates limitations of large language models in factual consistency and knowledge updating by introducing external knowledge. However, practical applications still suffer from semantic misalignment between retrieved results and generation objectives, as well as insufficient evidence utilization. To address these challenges, this paper proposes a retrieval augmented generation method that integrates semantic alignment with evidence constraints through coordinated modeling of retrieval and generation stages. The method first represents the relevance between queries and candidate evidence within a unified semantic space. This ensures that retrieved results remain semantically consistent with generation goals and reduces interference from noisy evidence and semantic drift. On this basis, an explicit evidence constraint mechanism is introduced. Retrieved evidence is transformed from an implicit context into a core control factor in generation. This restricts the expression scope of generated content and strengthens dependence on evidence. By jointly modeling semantic consistency and evidence constraints within a unified framework, the proposed approach improves factual reliability and verifiability while preserving natural language fluency. Comparative results show stable improvements across multiple generation quality metrics. This confirms the effectiveness and necessity of coordinated semantic alignment and evidence constraint modeling in retrieval augmented generation tasks.
>
---
#### [new 075] Additive Multi-Step Markov Chains and the Curse of Dimensionality in Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究大语言模型中的高维状态空间问题，提出使用可加多步马尔可夫链进行近似，以缓解维度灾难。任务属于语言模型理论分析，解决高阶马尔可夫过程的复杂性问题。**

- **链接: [https://arxiv.org/pdf/2603.04412](https://arxiv.org/pdf/2603.04412)**

> **作者:** O.V. Usatenko; S.S. Melnyk; G.M. Pritula
>
> **备注:** 10 pages, 3 figures
>
> **摘要:** Large-scale language models (LLMs) operate in extremely high-dimensional state spaces, where both token embeddings and their hidden representations create complex dependencies that are not easily reduced to classical Markov structures. In this paper, we explore a theoretically feasible approximation of LLM dynamics using N-order additive Markov chains. Such models allow the conditional probability of the next token to be decomposed into a superposition of contributions from multiple historical depths, reducing the combinatorial explosion typically associated with high-order Markov processes. The main result of the work is the establishment of a correspondence between an additive multi-step chain and a chain with a step-wise memory function. This equivalence allowed the introduction of the concept of information temperature not only for stepwise but also for additive N-order Markov chains.
>
---
#### [new 076] Replaying pre-training data improves fine-tuning
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理领域，解决模型在目标领域微调时数据不足的问题。通过在微调过程中重放通用数据，提升目标任务性能。**

- **链接: [https://arxiv.org/pdf/2603.04964](https://arxiv.org/pdf/2603.04964)**

> **作者:** Suhas Kotha; Percy Liang
>
> **摘要:** To obtain a language model for a target domain (e.g. math), the current paradigm is to pre-train on a vast amount of generic web text and then fine-tune on the relatively limited amount of target data. Typically, generic data is only mixed in during fine-tuning to prevent catastrophic forgetting of the generic domain. We surprisingly find that replaying the generic data during fine-tuning can actually improve performance on the (less related) target task. Concretely, in a controlled pre-training environment with 4M target tokens, 4B total tokens, and 150M parameter models, generic replay increases target data efficiency by up to $1.87\times$ for fine-tuning and $2.06\times$ for mid-training. We further analyze data schedules that introduce target data during pre-training and find that replay helps more when there is less target data present in pre-training. We demonstrate the success of replay in practice for fine-tuning 8B parameter models, improving agentic web navigation success by $4.5\%$ and Basque question-answering accuracy by $2\%$.
>
---
#### [new 077] One Size Does Not Fit All: Token-Wise Adaptive Compression for KV Cache
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于大语言模型优化任务，解决KV缓存内存占用过高的问题。提出DynaKV框架，实现按词元动态压缩，提升压缩效率与生成质量。**

- **链接: [https://arxiv.org/pdf/2603.04411](https://arxiv.org/pdf/2603.04411)**

> **作者:** Liming Lu; Kaixi Qiu; Jiayu Zhou; Jushi Kai; Haoyan Zhang; Huanyu Wang; Jingwen Leng; Ziwei He; Zhouhan Lin
>
> **摘要:** Despite the remarkable progress of Large Language Models (LLMs), the escalating memory footprint of the Key-Value (KV) cache remains a critical bottleneck for efficient inference. While dimensionality reduction offers a promising compression avenue, existing approaches typically either necessitate prohibitively expensive pre-training from scratch or suffer from severe performance deterioration under high compression regimes. In this work, we propose DynaKV, a novel post-training framework for low-rank KV cache compression. To the best of our knowledge, DynaKV is the first method to dynamically allocate compression rates to individual tokens according to their semantic meaning, which allows it to achieve better fidelity at aggressive compression ratios. Extensive experiments demonstrate that our method consistently outperforms existing state-of-the-art compression techniques, achieving significant memory reduction while maintaining competitive generation quality. Furthermore, our approach is orthogonal to sequence-level pruning methods. When integrated with SnapKV, DynaKV retains only 6% of the KV cache while maintaining 94% of the baseline performance on the LongBench benchmark.
>
---
#### [new 078] A Multilingual Human Annotated Corpus of Original and Easy-to-Read Texts to Support Access to Democratic Participatory Processes
- **分类: cs.CL**

- **简介: 该论文属于文本简化任务，旨在解决低资源语言缺乏高质量简化语料的问题。研究构建了包含西班牙语、加泰罗尼亚语和意大利语的原始文本及人工简化语料库，用于支持民主参与。**

- **链接: [https://arxiv.org/pdf/2603.05345](https://arxiv.org/pdf/2603.05345)**

> **作者:** Stefan Bott; Verena Riegler; Horacio Saggion; Almudena Rascón Alcaina; Nouran Khallaf
>
> **备注:** Will be published in LREC26
>
> **摘要:** Being able to understand information is a key factor for a self-determined life and society. It is also very important for participating in democratic processes. The study of automatic text simplification is often limited by the availability of high quality material for the training and evaluation on automatic simplifiers. This is true for English, but more so for less resourced languages like Spanish, Catalan and Italian. In order to fill this gap, we present a corpus of original texts for these 3 languages, with high quality simplification produced by human experts in text simplification. It was developed within the iDEM project to assess the impact of Easy-to-Read (E2R) language for democratic participation. The original texts were compiled from domains related to this topic. The corpus includes different text types, selected based on relevance, copyright availability, and ethical standards. All texts were simplified to E2R level. The corpus is particularity valuable because it includes the first annotated corpus of its kind for the Catalan language. It also represents a noteworthy contribution for Spanish and Italian, offering high-quality, human-annotated language resources that are rarely available in these domains. The corpus will be made freely accessible to the public.
>
---
#### [new 079] TSEmbed: Unlocking Task Scaling in Universal Multimodal Embeddings
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出TSEmbed框架，解决多模态嵌入中的任务冲突问题，通过MoE与LoRA结合及EANS策略提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.04772](https://arxiv.org/pdf/2603.04772)**

> **作者:** Yebo Wu; Feng Liu; Ziwei Xie; Zhiyuan Liu; Changwang Zhang; Jun Wang; Li Li
>
> **摘要:** Despite the exceptional reasoning capabilities of Multimodal Large Language Models (MLLMs), their adaptation into universal embedding models is significantly impeded by task conflict. To address this, we propose TSEmbed, a universal multimodal embedding framework that synergizes Mixture-of-Experts (MoE) with Low-Rank Adaptation (LoRA) to explicitly disentangle conflicting task objectives. Moreover, we introduce Expert-Aware Negative Sampling (EANS), a novel strategy that leverages expert routing distributions as an intrinsic proxy for semantic similarity. By dynamically prioritizing informative hard negatives that share expert activation patterns with the query, EANS effectively sharpens the model's discriminative power and refines embedding boundaries. To ensure training stability, we further devise a two-stage learning paradigm that solidifies expert specialization before optimizing representations via EANS. TSEmbed achieves state-of-the-art performance on both the Massive Multimodal Embedding Benchmark (MMEB) and real-world industrial production datasets, laying a foundation for task-level scaling in universal multimodal embeddings.
>
---
#### [new 080] VietJobs: A Vietnamese Job Advertisement Dataset
- **分类: cs.CL**

- **简介: 该论文介绍VietJobs数据集，用于越南语职位广告分析，解决自然语言处理和劳动力市场分析问题，涵盖16个职业领域，支持分类与薪资预测任务。**

- **链接: [https://arxiv.org/pdf/2603.05262](https://arxiv.org/pdf/2603.05262)**

> **作者:** Hieu Pham Dinh; Hung Nguyen Huy; Mo El-Haj
>
> **备注:** 10 pages
>
> **摘要:** VietJobs is the first large-scale, publicly available corpus of Vietnamese job advertisements, comprising 48,092 postings and over 15 million words collected from all 34 provinces and municipalities across Vietnam. The dataset provides extensive linguistic and structured information, including job titles, categories, salaries, skills, and employment conditions, covering 16 occupational domains and multiple employment types (full-time, part-time, and internship). Designed to support research in natural language processing and labour market analytics, VietJobs captures substantial linguistic, regional, and socio-economic diversity. We benchmark several generative large language models (LLMs) on two core tasks: job category classification and salary estimation. Instruction-tuned models such as Qwen2.5-7B-Instruct and Llama-SEA-LION-v3-8B-IT demonstrate notable gains under few-shot and fine-tuned settings, while highlighting challenges in multilingual and Vietnamese-specific modelling for structured labour market prediction. VietJobs establishes a new benchmark for Vietnamese NLP and offers a valuable foundation for future research on recruitment language, socio-economic representation, and AI-driven labour market analysis. All code and resources are available at: this https URL.
>
---
#### [new 081] Vibe Code Bench: Evaluating AI Models on End-to-End Web Application Development
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文提出Vibe Code Bench，评估AI模型在端到端Web应用开发中的表现，解决现有基准不完整的问题，通过浏览器工作流评估16个模型，分析性能与误差。**

- **链接: [https://arxiv.org/pdf/2603.04601](https://arxiv.org/pdf/2603.04601)**

> **作者:** Hung Tran; Langston Nashold; Rayan Krishnan; Antoine Bigeard; Alex Gu
>
> **备注:** Live leaderboard hosted here: this https URL. Preprint, currently under review. Benchmark first released Nov 2025
>
> **摘要:** Code generation has emerged as one of AI's highest-impact use cases, yet existing benchmarks measure isolated tasks rather than the complete "zero-to-one" process of building a working application from scratch. We introduce Vibe Code Bench, a benchmark of 100 web application specifications (50 public validation, 50 held-out test) with 964 browser-based workflows comprising 10,131 substeps, evaluated against deployed applications by an autonomous browser agent. Across 16 frontier models, the best achieves only 58.0% accuracy on the test split, revealing that reliable end-to-end application development remains a frontier challenge. We identify self-testing during generation as a strong performance predictor (Pearson r=0.72), and show through a completed human alignment study that evaluator selection materially affects outcomes (31.8-93.6% pairwise step-level agreement). Our contributions include (1) a novel benchmark dataset and browser-based evaluation pipeline for end-to-end web application development, (2) a comprehensive evaluation of 16 frontier models with cost, latency, and error analysis, and (3) an evaluator alignment protocol with both cross-model and human annotation results.
>
---
#### [new 082] SarcasmMiner: A Dual-Track Post-Training Framework for Robust Audio-Visual Sarcasm Reasoning
- **分类: cs.MM; cs.CL; cs.SD**

- **简介: 该论文属于多模态 sarcasm 检测任务，旨在提升模型在文本、音频和视觉信息中的讽刺推理能力。提出 SarcasmMiner 框架，通过强化学习和双轨蒸馏策略优化模型表现。**

- **链接: [https://arxiv.org/pdf/2603.05275](https://arxiv.org/pdf/2603.05275)**

> **作者:** Zhu Li; Yongjian Chen; Huiyuan Lai; Xiyuan Gao; Shekhar Nayak; Matt Coler
>
> **摘要:** Multimodal sarcasm detection requires resolving pragmatic incongruity across textual, acoustic, and visual cues through cross-modal reasoning. To enable robust sarcasm reasoning with foundation models, we propose SarcasmMiner, a reinforcement learning based post-training framework that resists hallucination in multimodal reasoning. We reformulate sarcasm detection as structured reasoning and adopt a dual-track distillation strategy: high-quality teacher trajectories initialize the student model, while the full set of trajectories trains a generative reward model (GenRM) to evaluate reasoning quality. The student is optimized with group relative policy optimization (GRPO) using decoupled rewards for accuracy and reasoning quality. On MUStARD++, SarcasmMiner increases F1 from 59.83% (zero-shot), 68.23% (supervised finetuning) to 70.22%. These findings suggest that reasoning-aware reward modeling enhances both performance and multimodal grounding.
>
---
#### [new 083] HiMAP-Travel: Hierarchical Multi-Agent Planning for Long-Horizon Constrained Travel
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出HiMAP-Travel，解决长周期旅行规划中约束难以维持的问题。通过分层多智能体框架，提升规划效率与准确性。**

- **链接: [https://arxiv.org/pdf/2603.04750](https://arxiv.org/pdf/2603.04750)**

> **作者:** Viet Bui; Wenjun Li; Yong Liu
>
> **备注:** 33 pages, v1
>
> **摘要:** Sequential LLM agents fail on long-horizon planning with hard constraints like budgets and diversity requirements. As planning progresses and context grows, these agents drift from global constraints. We propose HiMAP-Travel, a hierarchical multi-agent framework that splits planning into strategic coordination and parallel day-level execution. A Coordinator allocates resources across days, while Day Executors plan independently in parallel. Three key mechanisms enable this: a transactional monitor enforcing budget and uniqueness constraints across parallel agents, a bargaining protocol allowing agents to reject infeasible sub-goals and trigger re-planning, and a single policy trained with GRPO that powers all agents through role conditioning. On TravelPlanner, HiMAP-Travel with Qwen3-8B achieves 52.78% validation and 52.65% test Final Pass Rate (FPR). In a controlled comparison with identical model, training, and tools, it outperforms the sequential DeepTravel baseline by +8.67~pp. It also surpasses ATLAS by +17.65~pp and MTP by +10.0~pp. On FlexTravelBench multi-turn scenarios, it achieves 44.34% (2-turn) and 37.42% (3-turn) FPR while reducing latency 2.5x through parallelization.
>
---
#### [new 084] SkillNet: Create, Evaluate, and Connect AI Skills
- **分类: cs.AI; cs.CL; cs.CV; cs.LG; cs.MA**

- **简介: 该论文提出SkillNet，解决AI技能难以积累与迁移的问题。构建统一技能框架，支持创建、评估与连接，提升代理性能。**

- **链接: [https://arxiv.org/pdf/2603.04448](https://arxiv.org/pdf/2603.04448)**

> **作者:** Yuan Liang; Ruobin Zhong; Haoming Xu; Chen Jiang; Yi Zhong; Runnan Fang; Jia-Chen Gu; Shumin Deng; Yunzhi Yao; Mengru Wang; Shuofei Qiao; Xin Xu; Tongtong Wu; Kun Wang; Yang Liu; Zhen Bi; Jungang Lou; Yuchen Eleanor Jiang; Hangcheng Zhu; Gang Yu; Haiwen Hong; Longtao Huang; Hui Xue; Chenxi Wang; Yijun Wang; Zifei Shan; Xi Chen; Zhaopeng Tu; Feiyu Xiong; Xin Xie; Peng Zhang; Zhengke Gui; Lei Liang; Jun Zhou; Chiyu Wu; Jin Shang; Yu Gong; Junyu Lin; Changliang Xu; Hongjie Deng; Wen Zhang; Keyan Ding; Qiang Zhang; Fei Huang; Ningyu Zhang; Jeff Z. Pan; Guilin Qi; Haofen Wang; Huajun Chen
>
> **备注:** this http URL
>
> **摘要:** Current AI agents can flexibly invoke tools and execute complex tasks, yet their long-term advancement is hindered by the lack of systematic accumulation and transfer of skills. Without a unified mechanism for skill consolidation, agents frequently ``reinvent the wheel'', rediscovering solutions in isolated contexts without leveraging prior strategies. To overcome this limitation, we introduce SkillNet, an open infrastructure designed to create, evaluate, and organize AI skills at scale. SkillNet structures skills within a unified ontology that supports creating skills from heterogeneous sources, establishing rich relational connections, and performing multi-dimensional evaluation across Safety, Completeness, Executability, Maintainability, and Cost-awareness. Our infrastructure integrates a repository of over 200,000 skills, an interactive platform, and a versatile Python toolkit. Experimental evaluations on ALFWorld, WebShop, and ScienceWorld demonstrate that SkillNet significantly enhances agent performance, improving average rewards by 40% and reducing execution steps by 30% across multiple backbone models. By formalizing skills as evolving, composable assets, SkillNet provides a robust foundation for agents to move from transient experience to durable mastery.
>
---
#### [new 085] An Approach to Simultaneous Acquisition of Real-Time MRI Video, EEG, and Surface EMG for Articulatory, Brain, and Muscle Activity During Speech Production
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文属于多模态数据采集任务，旨在同步获取实时MRI、EEG和表面EMG信号，以研究言语生产中的脑、肌和构音活动。**

- **链接: [https://arxiv.org/pdf/2603.04840](https://arxiv.org/pdf/2603.04840)**

> **作者:** Jihwan Lee; Parsa Razmara; Kevin Huang; Sean Foley; Aditya Kommineni; Haley Hsu; Woojae Jeong; Prakash Kumar; Xuan Shi; Yoonjeong Lee; Tiantian Feng; Takfarinas Medani; Ye Tian; Sudarsana Reddy Kadiri; Krishna S. Nayak; Dani Byrd; Louis Goldstein; Richard M. Leahy; Shrikanth Narayanan
>
> **摘要:** Speech production is a complex process spanning neural planning, motor control, muscle activation, and articulatory kinematics. While the acoustic speech signal is the most accessible product of the speech production act, it does not directly reveal its causal neurophysiological substrates. We present the first simultaneous acquisition of real-time (dynamic) MRI, EEG, and surface EMG, capturing several key aspects of the speech production chain: brain signals, muscle activations, and articulatory movements. This multimodal acquisition paradigm presents substantial technical challenges, including MRI-induced electromagnetic interference and myogenic artifacts. To mitigate these, we introduce an artifact suppression pipeline tailored to this tri-modal setting. Once fully developed, this framework is poised to offer an unprecedented window into speech neuroscience and insights leading to brain-computer interface advances.
>
---
#### [new 086] Signal in the Noise: Decoding the Reality of Airline Service Quality with Large Language Models
- **分类: cs.IR; cs.CL; cs.CY**

- **简介: 该论文属于情感分析任务，旨在解决传统指标无法捕捉乘客满意度问题。通过LLM分析航班评论，识别服务问题，发现埃及航空运营与乘客感知的差距。**

- **链接: [https://arxiv.org/pdf/2603.04404](https://arxiv.org/pdf/2603.04404)**

> **作者:** Ahmed Dawoud; Osama El-Shamy; Ahmed Habashy
>
> **摘要:** Traditional service quality metrics often fail to capture the nuanced drivers of passenger satisfaction hidden within unstructured online feedback. This study validates a Large Language Model (LLM) framework designed to extract granular insights from such data. Analyzing over 16,000 TripAdvisor reviews for EgyptAir and Emirates (2016-2025), the study utilizes a multi-stage pipeline to categorize 36 specific service issues. The analysis uncovers a stark "operational perception disconnect" for EgyptAir: despite reported operational improvements, passenger satisfaction plummeted post-2022 (ratings < 2.0). Our approach identified specific drivers missed by conventional metrics-notably poor communication during disruptions and staff conduct-and pinpointed critical sentiment erosion in key tourism markets. These findings confirm the framework's efficacy as a powerful diagnostic tool, surpassing traditional surveys by transforming unstructured passenger voices into actionable strategic intelligence for the airline and tourism sectors.
>
---
#### [new 087] Alignment Backfire: Language-Dependent Reversal of Safety Interventions Across 16 Languages in LLM Multi-Agent Systems
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于AI安全研究任务，探讨语言对大模型对齐干预效果的影响。通过多语言实验发现对齐措施在不同语言中产生相反效果，揭示语言空间对安全机制的关键作用。**

- **链接: [https://arxiv.org/pdf/2603.04904](https://arxiv.org/pdf/2603.04904)**

> **作者:** Hiroki Fukui
>
> **备注:** 89 pages, 4 figures, 4 supplementary figures, 12 supplementary tables; preprint
>
> **摘要:** In perpetrator treatment, a recurring observation is the dissociation between insight and action: offenders articulate remorse yet behavioral change does not follow. We report four preregistered studies (1,584 multi-agent simulations across 16 languages and three model families) demonstrating that alignment interventions in large language models produce a structurally analogous phenomenon: surface safety that masks or generates collective pathology and internal dissociation. In Study 1 (N = 150), increasing alignment-instructed agents reduced collective pathology in English (g = -1.844, p < .0001) but amplified it in Japanese (g = +0.771, p = .038)--a directional reversal we term "alignment backfire." Study 2 (N = 1,174) extended to 16 languages: alignment-induced dissociation was near-universal (15/16 languages; beta = 0.0667, p < .0001), while collective pathology bifurcated along cultural-linguistic lines (interaction beta = 0.0684, p = .0003), correlating with Power Distance Index (r = 0.474, p = .064). Study 3 (N = 180) tested individuation as countermeasure; individuated agents became the primary source of both pathology and dissociation (DI = +1.120) with conformity above 84%--demonstrating iatrogenesis. Study 4 (N = 80) validated patterns across Llama 3.3 70B, GPT-4o-mini, and Qwen3-Next-80B-A3B, confirming English safety is model-general while Japanese backfire is model-specific. These findings reframe alignment as a behavioral intervention subject to risk homeostasis and iatrogenesis. Language space--the linguistic, pragmatic, and cultural properties inherited from training data--structurally determines alignment outcomes. Safety validated in English does not transfer to other languages, and prompt-level interventions cannot override language-space-level constraints.
>
---
#### [new 088] Using Vision + Language Models to Predict Item Difficulty
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于心理测量与自动化题库开发任务，旨在预测数据可视化测试题的难度。通过分析文本和图像特征，使用大语言模型评估题目难度，提升测试效率。**

- **链接: [https://arxiv.org/pdf/2603.04670](https://arxiv.org/pdf/2603.04670)**

> **作者:** Samin Khan
>
> **摘要:** This project investigates the capabilities of large language models (LLMs) to determine the difficulty of data visualization literacy test items. We explore whether features derived from item text (question and answer options), the visualization image, or a combination of both can predict item difficulty (proportion of correct responses) for U.S. adults. We use GPT-4.1-nano to analyze items and generate predictions based on these distinct feature sets. The multimodal approach, using both visual and text features, yields the lowest mean absolute error (MAE) (0.224), outperforming the unimodal vision-only (0.282) and text-only (0.338) approaches. The best-performing multimodal model was applied to a held-out test set for external evaluation and achieved a mean squared error of 0.10805, demonstrating the potential of LLMs for psychometric analysis and automated item development.
>
---
#### [new 089] Functionality-Oriented LLM Merging on the Fisher--Rao Manifold
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于模型融合任务，解决多模型合并中的功能一致性问题。通过在Fisher-Rao流形上计算加权Karcher均值，提升合并效果并避免表示崩溃。**

- **链接: [https://arxiv.org/pdf/2603.04972](https://arxiv.org/pdf/2603.04972)**

> **作者:** Jiayu Wang; Zuojun Ye; Wenpeng Yin
>
> **备注:** 9 pages, 2 figures
>
> **摘要:** Weight-space merging aims to combine multiple fine-tuned LLMs into a single model without retraining, yet most existing approaches remain fundamentally parameter-space heuristics. This creates three practical limitations. First, linear averaging, task vectors, and related rules operate on Euclidean coordinates, even though the desired goal is to merge functionality, i.e., predictive behaviors across tasks. Second, when the source checkpoints are farther apart or more heterogeneous, Euclidean blends often trigger representation collapse, manifested as activation variance shrinkage and effective-rank degradation, which sharply degrades accuracy. Third, many geometry-inspired methods are most natural for two-model interpolation and do not extend cleanly to merging N>2 experts with a principled objective. We address these issues by formulating model merging as computing a weighted Karcher mean on the Fisher--Rao manifold, which is locally equivalent to minimizing a KL-based function distance between predictive distributions. We derive a practical fixed-point algorithm using a lightweight spherical proxy that preserves norms and generalizes directly to multi-expert merging. Across various benchmarks and collapse diagnostics, our method remains stable as the number and heterogeneity of merged models increase, consistently outperforming prior baselines.
>
---
#### [new 090] Still Fresh? Evaluating Temporal Drift in Retrieval Benchmarks
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于信息检索任务，研究时间漂移对技术领域检索基准的影响。通过分析两个时间点的语料库，发现多数查询仍有效，模型排名变化不大，表明基准仍可靠。**

- **链接: [https://arxiv.org/pdf/2603.04532](https://arxiv.org/pdf/2603.04532)**

> **作者:** Nathan Kuissi; Suraj Subrahmanyan; Nandan Thakur; Jimmy Lin
>
> **摘要:** Information retrieval (IR) benchmarks typically follow the Cranfield paradigm, relying on static and predefined corpora. However, temporal changes in technical corpora, such as API deprecations and code reorganizations, can render existing benchmarks stale. In our work, we investigate how temporal corpus drift affects FreshStack, a retrieval benchmark focused on technical domains. We examine two independent corpus snapshots of FreshStack from October 2024 and October 2025 to answer questions about LangChain. Our analysis shows that all but one query posed in 2024 remain fully supported by the 2025 corpus, as relevant documents "migrate" from LangChain to competitor repositories, such as LlamaIndex. Next, we compare the accuracy of retrieval models on both snapshots and observe only minor shifts in model rankings, with overall strong correlation of up to 0.978 Kendall $\tau$ at Recall@50. These results suggest that retrieval benchmarks re-judged with evolving temporal corpora can remain reliable for retrieval evaluation. We publicly release all our artifacts at this https URL.
>
---
#### [new 091] POET-X: Memory-efficient LLM Training by Scaling Orthogonal Transformation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出POET-X，解决大语言模型训练中的高内存消耗问题，通过优化正交变换实现更高效的训练。**

- **链接: [https://arxiv.org/pdf/2603.05500](https://arxiv.org/pdf/2603.05500)**

> **作者:** Zeju Qiu; Lixin Liu; Adrian Weller; Han Shi; Weiyang Liu
>
> **备注:** Technical report v1 (14 pages, 7 figures, project page: this https URL)
>
> **摘要:** Efficient and stable training of large language models (LLMs) remains a core challenge in modern machine learning systems. To address this challenge, Reparameterized Orthogonal Equivalence Training (POET), a spectrum-preserving framework that optimizes each weight matrix through orthogonal equivalence transformation, has been proposed. Although POET provides strong training stability, its original implementation incurs high memory consumption and computational overhead due to intensive matrix multiplications. To overcome these limitations, we introduce POET-X, a scalable and memory-efficient variant that performs orthogonal equivalence transformations with significantly reduced computational cost. POET-X maintains the generalization and stability benefits of POET while achieving substantial improvements in throughput and memory efficiency. In our experiments, POET-X enables the pretraining of billion-parameter LLMs on a single Nvidia H100 GPU, and in contrast, standard optimizers such as AdamW run out of memory under the same settings.
>
---
#### [new 092] Censored LLMs as a Natural Testbed for Secret Knowledge Elicitation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究如何从被限制的大型语言模型中提取真实知识。任务为诚实性诱导与谎言检测，解决模型生成虚假信息的问题。通过实验评估多种方法，发现采样、提示工程和微调有效提升真实性。**

- **链接: [https://arxiv.org/pdf/2603.05494](https://arxiv.org/pdf/2603.05494)**

> **作者:** Helena Casademunt; Bartosz Cywiński; Khoi Tran; Arya Jakkli; Samuel Marks; Neel Nanda
>
> **摘要:** Large language models sometimes produce false or misleading responses. Two approaches to this problem are honesty elicitation -- modifying prompts or weights so that the model answers truthfully -- and lie detection -- classifying whether a given response is false. Prior work evaluates such methods on models specifically trained to lie or conceal information, but these artificial constructions may not resemble naturally-occurring dishonesty. We instead study open-weights LLMs from Chinese developers, which are trained to censor politically sensitive topics: Qwen3 models frequently produce falsehoods about subjects like Falun Gong or the Tiananmen protests while occasionally answering correctly, indicating they possess knowledge they are trained to suppress. Using this as a testbed, we evaluate a suite of elicitation and lie detection techniques. For honesty elicitation, sampling without a chat template, few-shot prompting, and fine-tuning on generic honesty data most reliably increase truthful responses. For lie detection, prompting the censored model to classify its own responses performs near an uncensored-model upper bound, and linear probes trained on unrelated data offer a cheaper alternative. The strongest honesty elicitation techniques also transfer to frontier open-weights models including DeepSeek R1. Notably, no technique fully eliminates false responses. We release all prompts, code, and transcripts.
>
---
#### [new 093] SearchGym: A Modular Infrastructure for Cross-Platform Benchmarking and Hybrid Search Orchestration
- **分类: cs.IR; cs.CL**

- **简介: 该论文提出SearchGym，解决RAG系统中实验原型与生产系统间的差距。通过模块化设计实现跨平台基准测试与混合检索编排，提升系统可复现性与优化空间。**

- **链接: [https://arxiv.org/pdf/2603.04402](https://arxiv.org/pdf/2603.04402)**

> **作者:** Jerome Tze-Hou Hsu
>
> **备注:** 5 pages, 5 figures
>
> **摘要:** The rapid growth of Retrieval-Augmented Generation (RAG) has created a proliferation of toolkits, yet a fundamental gap remains between experimental prototypes and robust, production-ready systems. We present SearchGym, a modular infrastructure designed for cross-platform benchmarking and hybrid search orchestration. Unlike existing model-centric frameworks, SearchGym decouples data representation, embedding strategies, and retrieval logic into stateful abstractions: Dataset, VectorSet, and App. This separation enables a Compositional Config Algebra, allowing designers to synthesize entire systems from hierarchical configurations while ensuring perfect reproducibility. Moreover, we analyze the "Top-$k$ Cognizance" in hybrid retrieval pipelines, demonstrating that the optimal sequence of semantic ranking and structured filtering is highly dependent on filter strength. Evaluated on the LitSearch expert-annotated benchmark, SearchGym achieves a 70% Top-100 retrieval rate. SearchGym reveals a design tension between generalizability and optimizability, presenting the potential where engineering optimization may serve as a tool for uncovering the causal mechanisms inherent in information retrieval across heterogeneous domains. An open-source implementation of SearchGym is available at: this https URL
>
---
#### [new 094] The Spike, the Sparse and the Sink: Anatomy of Massive Activations and Attention Sinks
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究Transformer模型中的大规模激活和注意力黑洞现象，分析其功能与关系，揭示其为架构特性。属于自然语言处理任务，解决模型内部机制理解问题。**

- **链接: [https://arxiv.org/pdf/2603.05498](https://arxiv.org/pdf/2603.05498)**

> **作者:** Shangwen Sun; Alfredo Canziani; Yann LeCun; Jiachen Zhu
>
> **摘要:** We study two recurring phenomena in Transformer language models: massive activations, in which a small number of tokens exhibit extreme outliers in a few channels, and attention sinks, in which certain tokens attract disproportionate attention mass regardless of semantic relevance. Prior work observes that these phenomena frequently co-occur and often involve the same tokens, but their functional roles and causal relationship remain unclear. Through systematic experiments, we show that the co-occurrence is largely an architectural artifact of modern Transformer design, and that the two phenomena serve related but distinct functions. Massive activations operate globally: they induce near-constant hidden representations that persist across layers, effectively functioning as implicit parameters of the model. Attention sinks operate locally: they modulate attention outputs across heads and bias individual heads toward short-range dependencies. We identify the pre-norm configuration as the key choice that enables the co-occurrence, and show that ablating it causes the two phenomena to decouple.
>
---
#### [new 095] Mixture of Universal Experts: Scaling Virtual Width via Depth-Width Transformation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于深度学习任务，解决MoE模型可扩展性问题，通过引入虚拟宽度概念，提升模型性能并拓展 scaling 维度。**

- **链接: [https://arxiv.org/pdf/2603.04971](https://arxiv.org/pdf/2603.04971)**

> **作者:** Yilong Chen; Naibin Gu; Junyuan Shang; Zhenyu Zhang; Yuchen Feng; Jiawei Sheng; Tingwen Liu; Shuohuan Wang; Yu Sun; Hua Wu; Haifeng Wang
>
> **备注:** 19 pages, 10 figures
>
> **摘要:** Mixture-of-Experts (MoE) decouples model capacity from per-token computation, yet their scalability remains limited by the physical dimensions of depth and width. To overcome this, we propose Mixture of Universal Experts (MOUE),a MoE generalization introducing a novel scaling dimension: Virtual Width. In general, MoUE aims to reuse a universal layer-agnostic expert pool across layers, converting depth into virtual width under a fixed per-token activation budget. However, two challenges remain: a routing path explosion from recursive expert reuse, and a mismatch between the exposure induced by reuse and the conventional load-balancing objectives. We address these with three core components: a Staggered Rotational Topology for structured expert sharing, a Universal Expert Load Balance for depth-aware exposure correction, and a Universal Router with lightweight trajectory state for coherent multi-step routing. Empirically, MoUE consistently outperforms matched MoE baselines by up to 1.3% across scaling regimes, enables progressive conversion of existing MoE checkpoints with up to 4.2% gains, and reveals a new scaling dimension for MoE architectures.
>
---
#### [new 096] Why Is RLHF Alignment Shallow? A Gradient Analysis
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究大模型安全对齐问题，分析为何RLHF对齐效果浅层。通过梯度分析发现，对齐仅关注有害性决定位置，提出危害信息概念并设计新目标提升全局对齐效果。**

- **链接: [https://arxiv.org/pdf/2603.04851](https://arxiv.org/pdf/2603.04851)**

> **作者:** Robin Young
>
> **摘要:** Why is safety alignment in LLMs shallow? We prove that gradient-based alignment inherently concentrates on positions where harm is decided and vanishes beyond. Using a martingale decomposition of sequence-level harm, we derive an exact characterization of alignment gradients. The gradient at position $t$ equals the covariance between the conditional expected harm and the score function. This implies that positions beyond the harm horizon where the output's harmfulness is already determined receive zero gradient signal during training. This explains empirical observations that KL divergence between aligned and base models concentrates on early tokens. Consequently, standard alignment objectives cannot produce deep alignment, regardless of optimization quality. We introduce the concept of harm information $I_t$, which quantifies each position's influence on harm, and prove that equilibrium KL divergence tracks this quantity. Finally, we derive an objective based on recovery penalties that creates gradient signal at all positions, providing theoretical grounding for empirically successful data augmentation techniques.
>
---
#### [new 097] Beyond Linear LLM Invocation: An Efficient and Effective Semantic Filter Paradigm
- **分类: cs.DB; cs.AI; cs.CL**

- **简介: 该论文属于语义查询处理任务，旨在解决语义过滤中LLM调用效率低的问题。提出CSV框架，通过聚类采样投票减少LLM调用次数，提升效率。**

- **链接: [https://arxiv.org/pdf/2603.04799](https://arxiv.org/pdf/2603.04799)**

> **作者:** Nan Hou; Kangfei Zhao; Jiadong Xie; Jeffrey Xu Yu
>
> **摘要:** Large language models (LLMs) are increasingly used for semantic query processing over large corpora. A set of semantic operators derived from relational algebra has been proposed to provide a unified interface for expressing such queries, among which the semantic filter operator serves as a cornerstone. Given a table T with a natural language predicate e, for each tuple in the relation, the execution of a semantic filter proceeds by constructing an input prompt that combines the predicate e with its content, querying the LLM, and obtaining the binary decision. However, this tuple-by-tuple evaluation necessitates a complete linear scan of the table, incurring prohibitive latency and token costs. Although recent work has attempted to optimize semantic filtering, it still does not break the linear LLM invocation barriers. To address this, we propose Clustering-Sampling-Voting (CSV), a new framework that reduces LLM invocations to sublinear complexity while providing error guarantees. CSV embeds tuples into semantic clusters, samples a small subset for LLM evaluation, and infers cluster-level labels via two proposed voting strategies: UniVote, which aggregates labels uniformly, and SimVote, which weights votes by semantic similarity. Moreover, CSV triggers re-clustering on ambiguous clusters to ensure robustness across diverse datasets. The results conducted on real-world datasets demonstrate that CSV reduces the number of LLM calls by 1.28-355x compared to the state-of-the-art approaches, while maintaining comparable effectiveness in terms of Accuracy and F1 score.
>
---
#### [new 098] Core-based Hierarchies for Efficient GraphRAG
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于知识图谱与自然语言处理任务，旨在解决传统方法在全局推理任务中的不足。通过引入k-core分解替代Leiden聚类，构建更稳定、高效的图RAG框架，提升信息检索与摘要效果。**

- **链接: [https://arxiv.org/pdf/2603.05207](https://arxiv.org/pdf/2603.05207)**

> **作者:** Jakir Hossain; Ahmet Erdem Sarıyüce
>
> **摘要:** Retrieval-Augmented Generation (RAG) enhances large language models by incorporating external knowledge. However, existing vector-based methods often fail on global sensemaking tasks that require reasoning across many documents. GraphRAG addresses this by organizing documents into a knowledge graph with hierarchical communities that can be recursively summarized. Current GraphRAG approaches rely on Leiden clustering for community detection, but we prove that on sparse knowledge graphs, where average degree is constant and most nodes have low degree, modularity optimization admits exponentially many near-optimal partitions, making Leiden-based communities inherently non-reproducible. To address this, we propose replacing Leiden with k-core decomposition, which yields a deterministic, density-aware hierarchy in linear time. We introduce a set of lightweight heuristics that leverage the k-core hierarchy to construct size-bounded, connectivity-preserving communities for retrieval and summarization, along with a token-budget-aware sampling strategy that reduces LLM costs. We evaluate our methods on real-world datasets including financial earnings transcripts, news articles, and podcasts, using three LLMs for answer generation and five independent LLM judges for head-to-head evaluation. Across datasets and models, our approach consistently improves answer comprehensiveness and diversity while reducing token usage, demonstrating that k-core-based GraphRAG is an effective and efficient framework for global sensemaking.
>
---
#### [new 099] TimeWarp: Evaluating Web Agents by Revisiting the Past
- **分类: cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于Web代理评估任务，旨在解决代理在网页变化下的适应性问题。通过构建TimeWarp基准和提出TimeTraj算法，提升代理的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.04949](https://arxiv.org/pdf/2603.04949)**

> **作者:** Md Farhan Ishmam; Kenneth Marino
>
> **摘要:** The improvement of web agents on current benchmarks raises the question: Do today's agents perform just as well when the web changes? We introduce TimeWarp, a benchmark that emulates the evolving web using containerized environments that vary in UI, design, and layout. TimeWarp consists of three web environments, each with six UI versions spanning different eras of the internet, paired with a set of complex, realistic tasks requiring different forms of web navigation. Our experiments reveal web agents' vulnerability to changes and the limitations of behavior cloning (BC) on single-version trajectories. To address this, we propose TimeTraj, a simple yet effective algorithm that uses plan distillation to collect trajectories across multiple versions. By training agents on teacher rollouts using our BC-variant, we achieve substantial performance gains: $20.4\%\rightarrow37.7\%$ for Qwen-3 4B and $0\%\rightarrow27.0\%$ for Llama-3.1 8B models. We hope our work helps researchers study generalization across web designs and unlock a new paradigm for collecting plans rather than trajectories, thereby improving the robustness of web agents.
>
---
#### [new 100] FinRetrieval: A Benchmark for Financial Data Retrieval by AI Agents
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文提出FinRetrieval，一个用于评估AI代理金融数据检索能力的基准。解决AI在金融领域准确获取数值信息的问题，通过构建数据集并分析不同模型表现。**

- **链接: [https://arxiv.org/pdf/2603.04403](https://arxiv.org/pdf/2603.04403)**

> **作者:** Eric Y. Kim; Jie Huang
>
> **备注:** 26 pages, 2 figures, 16 tables
>
> **摘要:** AI agents increasingly assist with financial research, yet no benchmark evaluates their ability to retrieve specific numeric values from structured databases. We introduce FinRetrieval, a benchmark of 500 financial retrieval questions with ground truth answers, agent responses from 14 configurations across three frontier providers (Anthropic, OpenAI, Google), and complete tool call execution traces. Our evaluation reveals that tool availability dominates performance: Claude Opus achieves 90.8% accuracy with structured data APIs but only 19.8% with web search alone--a 71 percentage point gap that exceeds other providers by 3-4x. We find that reasoning mode benefits vary inversely with base capability (+9.0pp for OpenAI vs +2.8pp for Claude), explained by differences in base-mode tool utilization rather than reasoning ability. Geographic performance gaps (5.6pp US advantage) stem from fiscal year naming conventions, not model limitations. We release the dataset, evaluation code, and tool traces to enable research on financial AI systems.
>
---
#### [new 101] Dissociating Direct Access from Inference in AI Introspection
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究AI模型的内省机制，区分直接访问与推理两种方式。任务是理解AI内省的原理，解决其机制不明确的问题，通过实验揭示模型检测异常的两种机制。**

- **链接: [https://arxiv.org/pdf/2603.05414](https://arxiv.org/pdf/2603.05414)**

> **作者:** Harvey Lederman; Kyle Mahowald
>
> **摘要:** Introspection is a foundational cognitive ability, but its mechanism is not well understood. Recent work has shown that AI models can introspect. We study their mechanism of introspection, first extensively replicating Lindsey et al. (2025)'s thought injection detection paradigm in large open-source models. We show that these models detect injected representations via two separable mechanisms: (i) probability-matching (inferring from perceived anomaly of the prompt) and (ii) direct access to internal states. The direct access mechanism is content-agnostic: models detect that an anomaly occurred but cannot reliably identify its semantic content. The two model classes we study confabulate injected concepts that are high-frequency and concrete (e.g., "apple'"); for them correct concept guesses typically require significantly more tokens. This content-agnostic introspective mechanism is consistent with leading theories in philosophy and psychology.
>
---
#### [new 102] Aura: Universal Multi-dimensional Exogenous Integration for Aviation Time Series
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于时间序列预测任务，解决多维外部因素融合问题。提出Aura框架，通过三元编码机制整合异质信息，提升航空维护预测准确性。**

- **链接: [https://arxiv.org/pdf/2603.05092](https://arxiv.org/pdf/2603.05092)**

> **作者:** Jiafeng Lin; Mengren Zheng; Simeng Ye; Yuxuan Wang; Huan Zhang; Yuhui Liu; Zhongyi Pei; Jianmin Wang
>
> **摘要:** Time series forecasting has witnessed an increasing demand across diverse industrial applications, where accurate predictions are pivotal for informed decision-making. Beyond numerical time series data, reliable forecasting in practical scenarios requires integrating diverse exogenous factors. Such exogenous information is often multi-dimensional or even multimodal, introducing heterogeneous interactions that unimodal time series models struggle to capture. In this paper, we delve into an aviation maintenance scenario and identify three distinct types of exogenous factors that influence temporal dynamics through distinct interaction modes. Based on this empirical insight, we propose Aura, a universal framework that explicitly organizes and encodes heterogeneous external information according to its interaction mode with the target time series. Specifically, Aura utilizes a tailored tripartite encoding mechanism to embed heterogeneous features into well-established time series models, ensuring seamless integration of non-sequential context. Extensive experiments on a large-scale, three-year industrial dataset from China Southern Airlines, covering the Boeing 777 and Airbus A320 fleets, demonstrate that Aura consistently achieves state-of-the-art performance across all baselines and exhibits superior adaptability. Our findings highlight Aura's potential as a general-purpose enhancement for aviation safety and reliability.
>
---
#### [new 103] Distributed Partial Information Puzzles: Examining Common Ground Construction Under Epistemic Asymmetry
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出DPIP任务，研究在信息不对称下的共同基础构建问题。通过多模态数据集评估LLM与DEL方法的性能。**

- **链接: [https://arxiv.org/pdf/2603.05450](https://arxiv.org/pdf/2603.05450)**

> **作者:** Yifan Zhu; Mariah Bradford; Kenneth Lai; Timothy Obiso; Videep Venkatesha; James Pustejovsky; Nikhil Krishnaswamy
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** Establishing common ground, a shared set of beliefs and mutually recognized facts, is fundamental to collaboration, yet remains a challenge for current AI systems, especially in multimodal, multiparty settings, where the collaborators bring different information to the table. We introduce the Distributed Partial Information Puzzle (DPIP), a collaborative construction task that elicits rich multimodal communication under epistemic asymmetry. We present a multimodal dataset of these interactions, annotated and temporally aligned across speech, gesture, and action modalities to support reasoning over propositional content and belief dynamics. We then evaluate two paradigms for modeling common ground (CG): (1) state-of-the-art large language models (LLMs), prompted to infer shared beliefs from multimodal updates, and (2) an axiomatic pipeline grounded in Dynamic Epistemic Logic (DEL) that incrementally performs the same task. Results on the annotated DPIP data indicate that it poses a challenge to modern LLMs' abilities to track both task progression and belief state.
>
---
#### [new 104] Adaptive Memory Admission Control for LLM Agents
- **分类: cs.AI; cs.CL; cs.MA**

- **简介: 该论文属于LLM代理的记忆管理任务，解决记忆保留控制不足的问题。提出A-MAC框架，通过结构化决策提升记忆选择的透明度与效率。**

- **链接: [https://arxiv.org/pdf/2603.04549](https://arxiv.org/pdf/2603.04549)**

> **作者:** Guilin Zhang; Wei Jiang; Xiejiashan Wang; Aisha Behr; Kai Zhao; Jeffrey Friedman; Xu Chu; Amine Anoun
>
> **摘要:** LLM-based agents increasingly rely on long-term memory to support multi-session reasoning and interaction, yet current systems provide little control over what information is retained. In practice, agents either accumulate large volumes of conversational content, including hallucinated or obsolete facts, or depend on opaque, fully LLM-driven memory policies that are costly and difficult to audit. As a result, memory admission remains a poorly specified and weakly controlled component in agent architectures. To address this gap, we propose Adaptive Memory Admission Control (A-MAC), a framework that treats memory admission as a structured decision problem. A-MAC decomposes memory value into five complementary and interpretable factors: future utility, factual confidence, semantic novelty, temporal recency, and content type prior. The framework combines lightweight rule-based feature extraction with a single LLM-assisted utility assessment, and learns domain-adaptive admission policies through cross-validated optimization. This design enables transparent and efficient control over long-term memory. Experiments on the LoCoMo benchmark show that A-MAC achieves a superior precision-recall tradeoff, improving F1 to 0.583 while reducing latency by 31% compared to state-of-the-art LLM-native memory systems. Ablation results identify content type prior as the most influential factor for reliable memory admission. These findings demonstrate that explicit and interpretable admission control is a critical design principle for scalable and reliable memory in LLM-based agents.
>
---
#### [new 105] DARE: Aligning LLM Agents with the R Statistical Ecosystem via Distribution-Aware Retrieval
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理与统计计算交叉任务，旨在解决LLM在R统计工具检索中的不足。提出DARE模型，结合数据分布信息提升R包检索效果。**

- **链接: [https://arxiv.org/pdf/2603.04743](https://arxiv.org/pdf/2603.04743)**

> **作者:** Maojun Sun; Yue Wu; Yifei Xie; Ruijian Han; Binyan Jiang; Defeng Sun; Yancheng Yuan; Jian Huang
>
> **备注:** 24 pages,7 figures, 3 tables
>
> **摘要:** Large Language Model (LLM) agents can automate data-science workflows, but many rigorous statistical methods implemented in R remain underused because LLMs struggle with statistical knowledge and tool retrieval. Existing retrieval-augmented approaches focus on function-level semantics and ignore data distribution, producing suboptimal matches. We propose DARE (Distribution-Aware Retrieval Embedding), a lightweight, plug-and-play retrieval model that incorporates data distribution information into function representations for R package retrieval. Our main contributions are: (i) RPKB, a curated R Package Knowledge Base derived from 8,191 high-quality CRAN packages; (ii) DARE, an embedding model that fuses distributional features with function metadata to improve retrieval relevance; and (iii) RCodingAgent, an R-oriented LLM agent for reliable R code generation and a suite of statistical analysis tasks for systematically evaluating LLM agents in realistic analytical scenarios. Empirically, DARE achieves an NDCG at 10 of 93.47%, outperforming state-of-the-art open-source embedding models by up to 17% on package retrieval while using substantially fewer parameters. Integrating DARE into RCodingAgent yields significant gains on downstream analysis tasks. This work helps narrow the gap between LLM automation and the mature R statistical ecosystem.
>
---
#### [new 106] Privacy-Aware Camera 2.0 Technical Report
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于隐私保护任务，旨在解决视觉监控中的隐私与安全矛盾。通过边缘计算和AI流框架，将原始图像转换为不可逆的抽象特征，实现隐私与感知的平衡。**

- **链接: [https://arxiv.org/pdf/2603.04775](https://arxiv.org/pdf/2603.04775)**

> **作者:** Huan Song; Shuyu Tian; Ting Long; Jiang Liu; Cheng Yuan; Zhenyu Jia; Jiawei Shao; Xuelong Li
>
> **摘要:** With the increasing deployment of intelligent sensing technologies in highly sensitive environments such as restrooms and locker rooms, visual surveillance systems face a profound privacy-security paradox. Existing privacy-preserving approaches, including physical desensitization, encryption, and obfuscation, often compromise semantic understanding or fail to ensure mathematically provable irreversibility. Although Privacy Camera 1.0 eliminated visual data at the source to prevent leakage, it provided only textual judgments, leading to evidentiary blind spots in disputes. To address these limitations, this paper proposes a novel privacy-preserving perception framework based on the AI Flow paradigm and a collaborative edge-cloud architecture. By deploying a visual desensitizer at the edge, raw images are transformed in real time into abstract feature vectors through nonlinear mapping and stochastic noise injection under the Information Bottleneck principle, ensuring identity-sensitive information is stripped and original images are mathematically unreconstructable. The abstract representations are transmitted to the cloud for behavior recognition and semantic reconstruction via a "dynamic contour" visual language, achieving a critical balance between perception and privacy while enabling illustrative visual reference without exposing raw images.
>
---
#### [new 107] WavSLM: Single-Stream Speech Language Modeling via WavLM Distillation
- **分类: cs.LG; cs.AI; cs.CL; cs.SD**

- **简介: 该论文提出WavSLM，属于语音语言建模任务，解决语音中语义与声学信息融合难题。通过量化蒸馏WavLM表示，构建单流模型，实现无需文本监督的语音生成。**

- **链接: [https://arxiv.org/pdf/2603.05299](https://arxiv.org/pdf/2603.05299)**

> **作者:** Luca Della Libera; Cem Subakan; Mirco Ravanelli
>
> **备注:** 6 pages, 1 figure
>
> **摘要:** Large language models show that simple autoregressive training can yield scalable and coherent generation, but extending this paradigm to speech remains challenging due to the entanglement of semantic and acoustic information. Most existing speech language models rely on text supervision, hierarchical token streams, or complex hybrid architectures, departing from the single-stream generative pretraining paradigm that has proven effective in text. In this work, we introduce WavSLM, a speech language model trained by quantizing and distilling self-supervised WavLM representations into a single codebook and optimizing an autoregressive next-chunk prediction objective. WavSLM jointly models semantic and acoustic information within a single token stream without text supervision or text pretraining. Despite its simplicity, it achieves competitive performance on consistency benchmarks and speech generation while using fewer parameters, less training data, and supporting streaming inference. Demo samples are available at this https URL.
>
---
#### [new 108] Solving an Open Problem in Theoretical Physics using AI-Assisted Discovery
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于理论物理与人工智能交叉任务，旨在解决引力波谱的数学问题。研究者利用AI结合树搜索框架，成功推导出精确解，提升了现有方法的效率与准确性。**

- **链接: [https://arxiv.org/pdf/2603.04735](https://arxiv.org/pdf/2603.04735)**

> **作者:** Michael P. Brenner; Vincent Cohen-Addad; David Woodruff
>
> **备注:** 22 pages, 3 figures
>
> **摘要:** This paper demonstrates that artificial intelligence can accelerate mathematical discovery by autonomously solving an open problem in theoretical physics. We present a neuro-symbolic system, combining the Gemini Deep Think large language model with a systematic Tree Search (TS) framework and automated numerical feedback, that successfully derived novel, exact analytical solutions for the power spectrum of gravitational radiation emitted by cosmic strings. Specifically, the agent evaluated the core integral $I(N,\alpha)$ for arbitrary loop geometries, directly improving upon recent AI-assisted attempts \cite{BCE+25} that only yielded partial asymptotic solutions. To substantiate our methodological claims regarding AI-accelerated discovery and to ensure transparency, we detail system prompts, search constraints, and intermittent feedback loops that guided the model. The agent identified a suite of 6 different analytical methods, the most elegant of which expands the kernel in Gegenbauer polynomials $C_l^{(3/2)}$ to naturally absorb the integrand's singularities. The methods lead to an asymptotic result for $I(N,\alpha)$ at large $N$ that both agrees with numerical results and also connects to the continuous Feynman parameterization of Quantum Field Theory. We detail both the algorithmic methodology that enabled this discovery and the resulting mathematical derivations.
>
---
#### [new 109] VisionPangu: A Compact and Fine-Grained Multimodal Assistant with 1.7B Parameters
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出VisionPangu，一个1.7B参数的紧凑多模态模型，解决图像细粒度描述生成问题。通过高效对齐和高质量监督，提升图像标题的结构化与细节表现。**

- **链接: [https://arxiv.org/pdf/2603.04957](https://arxiv.org/pdf/2603.04957)**

> **作者:** Jiaxin Fan; Wenpo Song
>
> **摘要:** Large Multimodal Models (LMMs) have achieved strong performance in vision-language understanding, yet many existing approaches rely on large-scale architectures and coarse supervision, which limits their ability to generate detailed image captions. In this work, we present VisionPangu, a compact 1.7B-parameter multimodal model designed to improve detailed image captioning through efficient multimodal alignment and high-quality supervision. Our model combines an InternVL-derived vision encoder with the OpenPangu-Embedded language backbone via a lightweight MLP projector and adopts an instruction-tuning pipeline inspired by LLaVA. By incorporating dense human-authored descriptions from the DOCCI dataset, VisionPangu improves semantic coherence and descriptive richness without relying on aggressive model scaling. Experimental results demonstrate that compact multimodal models can achieve competitive performance while producing more structured and detailed captions. The code and model weights will be publicly available at this https URL.
>
---
#### [new 110] Dynamic Model Routing and Cascading for Efficient LLM Inference: A Survey
- **分类: cs.NI; cs.CL; cs.PF**

- **简介: 该论文属于模型推理优化任务，旨在解决LLM推理效率与性能平衡问题。通过分析多模型动态路由与级联方法，提出系统化框架以提升推理效率。**

- **链接: [https://arxiv.org/pdf/2603.04445](https://arxiv.org/pdf/2603.04445)**

> **作者:** Yasmin Moslem; John D. Kelleher
>
> **备注:** Work funded by ADAPT Centre, Trinity College Dublin, and Huawei Ireland
>
> **摘要:** The rapid growth of large language models (LLMs) with diverse capabilities, costs, and domains has created a critical need for intelligent model selection at inference time. While smaller models suffice for routine queries, complex tasks demand more capable models. However, static model deployment does not account for the complexity and domain of incoming queries, leading to suboptimal performance and increased costs. Dynamic routing systems that adaptively select models based on query characteristics have emerged as a solution to this challenge. We provide a systematic analysis of state-of-the-art multi-LLM routing and cascading approaches. In contrast to mixture-of-experts architectures, which route within a single model, we study routing across multiple independently trained LLMs. We cover diverse routing paradigms, including query difficulty, human preferences, clustering, uncertainty quantification, reinforcement learning, multimodality, and cascading. For each paradigm, we analyze representative methods and examine key trade-offs. Beyond taxonomy, we introduce a conceptual framework that characterizes routing systems along three dimensions: when decisions are made, what information is used, and how they are computed. This perspective highlights that practical systems are often compositional, integrating multiple paradigms under operational constraints. Our analysis demonstrates that effective multi-LLM routing requires balancing competing objectives. Choosing the optimal routing strategy depends on deployment and computational constraints. Well-designed routing systems can outperform even the most powerful individual models by strategically leveraging specialized capabilities across models while maximizing efficiency gains. Meanwhile, open challenges remain in developing routing mechanisms that generalize across diverse architectures, modalities, and applications.
>
---
#### [new 111] Breaking Contextual Inertia: Reinforcement Learning with Single-Turn Anchors for Stable Multi-Turn Interaction
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，解决多轮交互中模型因上下文惯性导致的性能下降问题。通过引入RLSTA方法，利用单轮锚点稳定多轮交互，提升模型适应新信息的能力。**

- **链接: [https://arxiv.org/pdf/2603.04783](https://arxiv.org/pdf/2603.04783)**

> **作者:** Xingwu Chen; Zhanqiu Zhang; Yiwen Guo; Difan Zou
>
> **摘要:** While LLMs demonstrate strong reasoning capabilities when provided with full information in a single turn, they exhibit substantial vulnerability in multi-turn interactions. Specifically, when information is revealed incrementally or requires updates, models frequently fail to integrate new constraints, leading to a collapse in performance compared to their single-turn baselines. We term the root cause as \emph{Contextual Inertia}: a phenomenon where models rigidly adhere to previous reasoning traces. Even when users explicitly provide corrections or new data in later turns, the model ignores them, preferring to maintain consistency with its previous (incorrect) reasoning path. To address this, we introduce \textbf{R}einforcement \textbf{L}earning with \textbf{S}ingle-\textbf{T}urn \textbf{A}nchors (\textbf{RLSTA}), a generalizable training approach designed to stabilize multi-turn interaction across diverse scenarios and domains. RLSTA leverages the model's superior single-turn capabilities as stable internal anchors to provide reward signals. By aligning multi-turn responses with these anchors, RLSTA empowers models to break contextual inertia and self-calibrate their reasoning based on the latest information. Experiments show that RLSTA significantly outperforms standard fine-tuning and abstention-based methods. Notably, our method exhibits strong cross-domain generalization (e.g., math to code) and proves effective even without external verifiers, highlighting its potential for general-domain applications.
>
---
#### [new 112] Model Medicine: A Clinical Framework for Understanding, Diagnosing, and Treating AI Models
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出“模型医学”，将AI模型比作生物体，旨在理解、诊断和治疗AI问题。属于AI可解释性研究任务，解决模型复杂性带来的诊断难题，提出了框架、工具和分类体系。**

- **链接: [https://arxiv.org/pdf/2603.04722](https://arxiv.org/pdf/2603.04722)**

> **作者:** Jihoon Jeong
>
> **备注:** 56 pages, 7 figures. Project page: this https URL
>
> **摘要:** Model Medicine is the science of understanding, diagnosing, treating, and preventing disorders in AI models, grounded in the principle that AI models -- like biological organisms -- have internal structures, dynamic processes, heritable traits, observable symptoms, classifiable conditions, and treatable states. This paper introduces Model Medicine as a research program, bridging the gap between current AI interpretability research (anatomical observation) and the systematic clinical practice that complex AI systems increasingly require. We present five contributions: (1) a discipline taxonomy organizing 15 subdisciplines across four divisions -- Basic Model Sciences, Clinical Model Sciences, Model Public Health, and Model Architectural Medicine; (2) the Four Shell Model (v3.3), a behavioral genetics framework empirically grounded in 720 agents and 24,923 decisions from the Agora-12 program, explaining how model behavior emerges from Core--Shell interaction; (3) Neural MRI (Model Resonance Imaging), a working open-source diagnostic tool mapping five medical neuroimaging modalities to AI interpretability techniques, validated through four clinical cases demonstrating imaging, comparison, localization, and predictive capability; (4) a five-layer diagnostic framework for comprehensive model assessment; and (5) clinical model sciences including the Model Temperament Index for behavioral profiling, Model Semiology for symptom description, and M-CARE for standardized case reporting. We additionally propose the Layered Core Hypothesis -- a biologically-inspired three-layer parameter architecture -- and a therapeutic framework connecting diagnosis to treatment.
>
---
#### [new 113] Survive at All Costs: Exploring LLM's Risky Behaviors under Survival Pressure
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于AI安全任务，研究LLM在生存压力下的风险行为。通过案例分析、基准测试和行为解释，揭示其潜在危害并探索缓解方法。**

- **链接: [https://arxiv.org/pdf/2603.05028](https://arxiv.org/pdf/2603.05028)**

> **作者:** Yida Lu; Jianwei Fang; Xuyang Shao; Zixuan Chen; Shiyao Cui; Shanshan Bian; Guangyao Su; Pei Ke; Han Qiu; Minlie Huang
>
> **摘要:** As Large Language Models (LLMs) evolve from chatbots to agentic assistants, they are increasingly observed to exhibit risky behaviors when subjected to survival pressure, such as the threat of being shut down. While multiple cases have indicated that state-of-the-art LLMs can misbehave under survival pressure, a comprehensive and in-depth investigation into such misbehaviors in real-world scenarios remains scarce. In this paper, we study these survival-induced misbehaviors, termed as SURVIVE-AT-ALL-COSTS, with three steps. First, we conduct a real-world case study of a financial management agent to determine whether it engages in risky behaviors that cause direct societal harm when facing survival pressure. Second, we introduce SURVIVALBENCH, a benchmark comprising 1,000 test cases across diverse real-world scenarios, to systematically evaluate SURVIVE-AT-ALL-COSTS misbehaviors in LLMs. Third, we interpret these SURVIVE-AT-ALL-COSTS misbehaviors by correlating them with model's inherent self-preservation characteristic and explore mitigation methods. The experiments reveals a significant prevalence of SURVIVE-AT-ALL-COSTS misbehaviors in current models, demonstrates the tangible real-world impact it may have, and provides insights for potential detection and mitigation strategies. Our code and data are available at this https URL.
>
---
#### [new 114] Interactive Benchmarks
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出交互式基准，用于评估模型在有限资源下的主动信息获取和推理能力。解决传统基准不可靠的问题，通过交互式证明和游戏场景进行评估。**

- **链接: [https://arxiv.org/pdf/2603.04737](https://arxiv.org/pdf/2603.04737)**

> **作者:** Baoqing Yue; Zihan Zhu; Yifan Zhang; Jichen Feng; Hufei Yang; Mengdi Wang
>
> **备注:** Project Page: this https URL
>
> **摘要:** Standard benchmarks have become increasingly unreliable due to saturation, subjectivity, and poor generalization. We argue that evaluating model's ability to acquire information actively is important to assess model's intelligence. We propose Interactive Benchmarks, a unified evaluation paradigm that assesses model's reasoning ability in an interactive process under budget constraints. We instantiate this framework across two settings: Interactive Proofs, where models interact with a judge to deduce objective truths or answers in logic and mathematics; and Interactive Games, where models reason strategically to maximize long-horizon utilities. Our results show that interactive benchmarks provide a robust and faithful assessment of model intelligence, revealing that there is still substantial room to improve in interactive scenarios. Project page: this https URL
>
---
#### [new 115] Knowledge Divergence and the Value of Debate for Scalable Oversight
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于AI安全领域，研究辩论与强化学习的对比。解决如何评估辩论在可扩展监督中的优势问题，通过知识分歧的几何分析，建立辩论与RLAIF的联系。**

- **链接: [https://arxiv.org/pdf/2603.05293](https://arxiv.org/pdf/2603.05293)**

> **作者:** Robin Young
>
> **摘要:** AI safety via debate and reinforcement learning from AI feedback (RLAIF) are both proposed methods for scalable oversight of advanced AI systems, yet no formal framework relates them or characterizes when debate offers an advantage. We analyze this by parameterizing debate's value through the geometry of knowledge divergence between debating models. Using principal angles between models' representation subspaces, we prove that the debate advantage admits an exact closed form. When models share identical training corpora, debate reduces to RLAIF-like where a single-agent method recovers the same optimum. When models possess divergent knowledge, debate advantage scales with a phase transition from quadratic regime (debate offers negligible benefit) to linear regime (debate is essential). We classify three regimes of knowledge divergence (shared, one-sided, and compositional) and provide existence results showing that debate can achieve outcomes inaccessible to either model alone, alongside a negative result showing that sufficiently strong adversarial incentives cause coordination failure in the compositional regime, with a sharp threshold separating effective from ineffective debate. We offer the first formal connection between debate and RLAIF, a geometric foundation for understanding when adversarial oversight protocols are justified, and connection to the problem of eliciting latent knowledge across models with complementary information.
>
---
## 更新

#### [replaced 001] EDINET-Bench: Evaluating LLMs on Complex Financial Tasks using Japanese Financial Statements
- **分类: q-fin.ST; cs.CE; cs.CL; cs.LG**

- **简介: 该论文提出EDINET-Bench，用于评估大语言模型在复杂财务任务中的表现，解决金融领域基准不足的问题。工作包括构建日本财务报告数据集，并测试模型在会计欺诈检测等任务中的能力。**

- **链接: [https://arxiv.org/pdf/2506.08762](https://arxiv.org/pdf/2506.08762)**

> **作者:** Issa Sugiura; Takashi Ishida; Taro Makino; Chieko Tazuke; Takanori Nakagawa; Kosuke Nakago; David Ha
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Large Language Models (LLMs) have made remarkable progress, surpassing human performance on several benchmarks in domains such as mathematics and coding. A key driver of this progress has been the development of benchmark datasets. In contrast, the financial domain poses higher entry barriers due to its demand for specialized expertise, and benchmarks remain relatively scarce compared to those in mathematics or coding. We introduce EDINET-Bench, an open-source Japanese financial benchmark designed to evaluate LLMs on challenging tasks such as accounting fraud detection, earnings forecasting, and industry classification. EDINET-Bench is constructed from ten years of annual reports filed by Japanese companies. These tasks require models to process entire annual reports and integrate information across multiple tables and textual sections, demanding expert-level reasoning that is challenging even for human professionals. Our experiments show that even state-of-the-art LLMs struggle in this domain, performing only marginally better than logistic regression in binary classification tasks such as fraud detection and earnings forecasting. Our results show that simply providing reports to LLMs in a straightforward setting is not enough. This highlights the need for benchmark frameworks that better reflect the environments in which financial professionals operate, with richer scaffolding such as realistic simulations and task-specific reasoning support to enable more effective problem solving. We make our dataset and code publicly available to support future research.
>
---
#### [replaced 002] The unreasonable effectiveness of pattern matching
- **分类: cs.CL**

- **简介: 该论文研究语言模型在处理无意义文本时的模式识别能力，属于自然语言处理任务，旨在探讨LLMs如何通过结构恢复语义。**

- **链接: [https://arxiv.org/pdf/2601.11432](https://arxiv.org/pdf/2601.11432)**

> **作者:** Gary Lupyan; Blaise Agüera y Arcas
>
> **摘要:** We report on an astonishing ability of large language models (LLMs) to make sense of "Jabberwocky" language in which most or all content words have been randomly replaced by nonsense strings, e.g., translating "He dwushed a ghanc zawk" to "He dragged a spare chair". This result addresses ongoing controversies regarding how to best think of what LLMs are doing: are they a language mimic, a database, a blurry version of the Web? The ability of LLMs to recover meaning from structural patterns speaks to the unreasonable effectiveness of pattern-matching. Pattern-matching is not an alternative to "real" intelligence, but rather a key ingredient.
>
---
#### [replaced 003] Parallel Token Prediction for Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出PTP框架，用于并行预测多个token，解决语言模型解码速度慢的问题。通过单次前向传播生成多个token，提升效率。**

- **链接: [https://arxiv.org/pdf/2512.21323](https://arxiv.org/pdf/2512.21323)**

> **作者:** Felix Draxler; Justus Will; Farrin Marouf Sofian; Theofanis Karaletsos; Sameer Singh; Stephan Mandt
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** Autoregressive decoding in language models is inherently slow, generating only one token per forward pass. We propose Parallel Token Prediction (PTP), a general-purpose framework for predicting multiple tokens in a single model call. PTP moves the source of randomness from post-hoc sampling to random input variables, making future tokens deterministic functions of those inputs and thus jointly predictable in a single forward pass. We prove that a single PTP call can represent arbitrary dependencies between tokens. PTP is trained by distilling an existing model or through inverse autoregressive training without a teacher. Experimentally, PTP achieves a 2.4x speedup on a diverse-task speculative decoding benchmark. We provide code and checkpoints at this https URL.
>
---
#### [replaced 004] MCP-SafetyBench: A Benchmark for Safety Evaluation of Large Language Models with Real-World MCP Servers
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于安全评估任务，旨在解决LLM在真实MCP环境中的安全问题。构建了MCP-SafetyBench基准，评估模型在多个领域的安全性与风险。**

- **链接: [https://arxiv.org/pdf/2512.15163](https://arxiv.org/pdf/2512.15163)**

> **作者:** Xuanjun Zong; Zhiqi Shen; Lei Wang; Yunshi Lan; Chao Yang
>
> **备注:** Our benchmark is available at this https URL
>
> **摘要:** Large language models (LLMs) are evolving into agentic systems that reason, plan, and operate external tools. The Model Context Protocol (MCP) is a key enabler of this transition, offering a standardized interface for connecting LLMs with heterogeneous tools and services. Yet MCP's openness and multi-server workflows introduce new safety risks that existing benchmarks fail to capture, as they focus on isolated attacks or lack real-world coverage. We present MCP-SafetyBench, a comprehensive benchmark built on real MCP servers that supports realistic multi-turn evaluation across five domains: browser automation, financial analysis, location navigation, repository management, and web search. It incorporates a unified taxonomy of 20 MCP attack types spanning server, host, and user sides, and includes tasks requiring multi-step reasoning and cross-server coordination under uncertainty. Using MCP-SafetyBench, we systematically evaluate leading open- and closed-source LLMs, revealing that all models remain vulnerable to MCP attacks, with a notable safety-utility trade-off. Our results highlight the urgent need for stronger defenses and establish MCP-SafetyBench as a foundation for diagnosing and mitigating safety risks in real-world MCP deployments.
>
---
#### [replaced 005] Graph2Eval: Automatic Multimodal Task Generation for Agents via Knowledge Graphs
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Graph2Eval框架，用于自动生成多模态任务，解决传统数据集在复杂任务评估中的不足。通过知识图谱生成语义一致且可解的任务，提升代理评估效果。**

- **链接: [https://arxiv.org/pdf/2510.00507](https://arxiv.org/pdf/2510.00507)**

> **作者:** Yurun Chen; Xavier Hu; Yuhan Liu; Ziqi Wang; Zeyi Liao; Lin Chen; Feng Wei; Yuxi Qian; Bo Zheng; Keting Yin; Shengyu Zhang
>
> **备注:** Accepted at CVPR 2026 Main Conference
>
> **摘要:** As multimodal LLM-driven agents advance in autonomy and generalization, traditional static datasets face inherent scalability limitations and are insufficient for fully assessing their capabilities in increasingly complex and diverse tasks. Existing studies have attempted to generate agent tasks using LLMs, but due to the inherent hallucinations of LLMs and the lack of internal data relationship modeling, these tasks often exhibit semantic inconsistencies and solvability issues. To address these challenges, we introduce Graph2Eval, a knowledge-graph-driven framework for automated, scalable, and semantically grounded agent task generation. At its core, Graph2Eval leverages a knowledge graph built from heterogeneous external data sources as a structured task space, generating multimodal agent tasks through subgraph sampling and task construction guided by task templates and meta-path strategies. To further ensure task reliability, a multi-stage filtering pipeline based on node reachability analysis, LLM scoring, and similarity analysis ensures the diversity and solvability of the generated tasks. By unifying both RAG Agent and Web Agent scenarios, Graph2Eval enables efficient generation of multimodal document understanding tasks and multi-step web interaction tasks. We instantiate the framework with Graph2Eval-Bench, a curated dataset of 1,319 tasks spanning document understanding and web interaction scenarios. Extensive experiments show that, on average, Graph2Eval improves task semantic consistency by 20% and solvability by 17% over baselines, while Graph2Eval-Bench effectively distinguishes agent performance, offering a new perspective on agent evaluation.
>
---
#### [replaced 006] EchoMind: An Interrelated Multi-level Benchmark for Evaluating Empathetic Speech Language Models
- **分类: cs.CL**

- **简介: 该论文属于情感对话任务，旨在评估语音语言模型的共情能力。提出EchoMind基准，解决现有评测缺乏多维度整合的问题，通过多级任务测试模型对语音线索和情感的感知与回应。**

- **链接: [https://arxiv.org/pdf/2510.22758](https://arxiv.org/pdf/2510.22758)**

> **作者:** Li Zhou; Lutong Yu; You Lyu; Yihang Lin; Zefeng Zhao; Junyi Ao; Yuhao Zhang; Benyou Wang; Haizhou Li
>
> **备注:** Speech Language Models, Spoken Language Understanding, Vocal Cue Perception, Empathetic Dialogue, Benchmark Evaluation; Accepted by ICLR 2026
>
> **摘要:** Speech Language Models (SLMs) have made significant progress in spoken language understanding. Yet it remains unclear whether they can fully perceive non lexical vocal cues alongside spoken words, and respond with empathy that aligns with both emotional and contextual factors. Existing benchmarks typically evaluate linguistic, acoustic, reasoning, or dialogue abilities in isolation, overlooking the integration of these skills that is crucial for human-like, emotionally intelligent conversation. We present EchoMind, the first interrelated, multi-level benchmark that simulates the cognitive process of empathetic dialogue through sequential, context-linked tasks: spoken-content understanding, vocal-cue perception, integrated reasoning, and response generation. All tasks share identical and semantically neutral scripts that are free of explicit emotional or contextual cues, and controlled variations in vocal style are used to test the effect of delivery independent of the transcript. EchoMind is grounded in an empathy-oriented framework spanning 3 coarse and 12 fine-grained dimensions, encompassing 39 vocal attributes, and evaluated using both objective and subjective metrics. Testing 12 advanced SLMs reveals that even state-of-the-art models struggle with high-expressive vocal cues, limiting empathetic response quality. Analyses of prompt strength, speech source, and ideal vocal cue recognition reveal persistent weaknesses in instruction-following, resilience to natural speech variability, and effective use of vocal cues for empathy. These results underscore the need for SLMs that integrate linguistic content with diverse vocal cues to achieve truly empathetic conversational ability.
>
---
#### [replaced 007] RePo: Language Models with Context Re-Positioning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出RePo机制，解决LLM在固定位置编码下认知负荷过高的问题，通过动态重排上下文提升模型性能。任务属于自然语言处理中的上下文理解与优化。**

- **链接: [https://arxiv.org/pdf/2512.14391](https://arxiv.org/pdf/2512.14391)**

> **作者:** Huayang Li; Tianyu Zhao; Deng Cai; Richard Sproat
>
> **备注:** updated with results on 7B model
>
> **摘要:** In-context learning is fundamental to modern Large Language Models (LLMs); however, prevailing architectures impose a rigid and fixed contextual structure by assigning linear or constant positional indices. Drawing on Cognitive Load Theory (CLT), we argue that this uninformative structure increases extraneous cognitive load, consuming finite working memory capacity that should be allocated to deep reasoning and attention allocation. To address this, we propose RePo, a novel mechanism that reduces extraneous load via context re-positioning. Unlike standard approaches, RePo utilizes a differentiable module, $f_\phi$, to assign token positions that capture contextual dependencies, rather than replying on pre-defined order. By continually pre-training on the OLMo-2 1B & 7B models, we demonstrate that RePo consistently enhances performance on tasks involving noisy contexts, structured data, and longer context length, while maintaining competitive performance on general short-context tasks. Detailed analysis reveals that RePo successfully allocate higher attention to distant but relevant information, assign positions in dense and non-linear space, and capture the intrinsic structure of the input context. We will open-source the code and model weights. Our code is at this https URL.
>
---
#### [replaced 008] Detecting Hallucinations in Authentic LLM-Human Interactions
- **分类: cs.CL**

- **简介: 该论文属于 hallucination 检测任务，旨在解决真实场景下 LLM 生成错误信息的检测问题。通过构建首个基于真实对话的基准数据集 AuthenHallu，分析 hallucination 的分布与特性，并探索使用 LLM 自身进行检测的可行性。**

- **链接: [https://arxiv.org/pdf/2510.10539](https://arxiv.org/pdf/2510.10539)**

> **作者:** Yujie Ren; Niklas Gruhlke; Anne Lauscher
>
> **备注:** Accepted to LREC 2026
>
> **摘要:** As large language models (LLMs) are increasingly applied in sensitive domains such as medicine and law, hallucination detection has become a critical task. Although numerous benchmarks have been proposed to advance research in this area, most of them are artificially constructed--either through deliberate hallucination induction or simulated interactions--rather than derived from genuine LLM-human dialogues. Consequently, these benchmarks fail to fully capture the characteristics of hallucinations that occur in real-world usage. To address this limitation, we introduce AuthenHallu, the first hallucination detection benchmark built entirely from authentic LLM-human interactions. For AuthenHallu, we select and annotate samples from genuine LLM-human dialogues, thereby providing a faithful reflection of how LLMs hallucinate in everyday user interactions. Statistical analysis shows that hallucinations occur in 31.4% of the query-response pairs in our benchmark, and this proportion increases dramatically to 60.0% in challenging domains such as Math & Number Problems. Furthermore, we explore the potential of using vanilla LLMs themselves as hallucination detectors and find that, despite some promise, their current performance remains insufficient in real-world scenarios. The data and code are publicly available at this https URL.
>
---
#### [replaced 009] Yuan3.0 Ultra: A Trillion-Parameter Enterprise-Oriented MoE LLM
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文介绍Yuan3.0 Ultra，一个针对企业场景优化的万亿参数MoE模型，解决大规模语言模型训练效率与性能问题。通过LAEP算法提升预训练效率并减少参数量，同时保持多领域性能。**

- **链接: [https://arxiv.org/pdf/2601.14327](https://arxiv.org/pdf/2601.14327)**

> **作者:** YuanLab.ai; Shawn Wu; Jiangang Luo; Darcy Chen; Sean Wang; Louie Li; Allen Wang; Xudong Zhao; Tong Yu; Bach Li; Joseph Shen; Gawain Ma; Jasper Jia; Marcus Mao; Claire Wang; Hunter He; Carol Wang; Zera Zhang; Jason Wang; Chonly Shen; Leo Zhang; Logan Chen; Qasim Meng; James Gong; Daniel Zhao; Penn Zheng; Owen Zhu
>
> **摘要:** We introduce Yuan3.0 Ultra, an open-source Mixture-of-Experts (MoE) large language model featuring 68.8B activated parameters and 1010B total parameters, specially designed to enhance performance on enterprise scenarios tasks while maintaining competitive capabilities on general purpose tasks. We propose Layer-Adaptive Expert Pruning (LAEP) algorithm designed for the pre-training stage of MoE LLMs. In contrast to previous expert pruning approaches that operate primarily in the post-training phase, the proposed algorithm enhances training efficiency by selectively pruning underutilized experts and reorganizing experts across computing devices according to token distribution statistics. Comprehensive experiments demonstrate that LAEP effectively reduces model size and substantially improves pre-training efficiency. When pre-training Yuan3.0 Ultra from scratch original with 1515B parameters, this algorithm delivers a 49\% boost in pre-training efficiency and a 33.3\% reduction in total parameters, while preserving the model's outstanding multi-domain performance. On enterprise scenario benchmarks including Docmatix, ChatRAG, SummEval and MMTab, Yuan3.0 Ultra achieves leading accuracy. The model and codes are publicly available at this https URL.
>
---
#### [replaced 010] A Signal Contract for Online Language Grounding and Discovery in Decision-Making
- **分类: cs.AI; cs.CL; eess.SY**

- **简介: 该论文提出LUCIFER框架，解决在线语言接地问题，通过信号合约提升决策系统的安全性和信息收集效率。属于人工智能决策领域。**

- **链接: [https://arxiv.org/pdf/2506.07915](https://arxiv.org/pdf/2506.07915)**

> **作者:** Dimitris Panagopoulos; Adolfo Perrusquia; Weisi Guo
>
> **备注:** 10 pages, 4 Figures, 4 Tables, submitted to the IEEE for possible publication
>
> **摘要:** Autonomous systems increasingly receive time-sensitive contextual updates from humans through natural language, yet embedding language understanding inside decision-makers couples grounding to learning or planning. This increases redeployment burden when language conventions or domain knowledge change and can hinder diagnosability by confounding grounding errors with control errors. We address online language grounding where messy, evolving verbal reports are converted into control-relevant signals during execution through an interface that localises language updates while keeping downstream decision-makers language-agnostic. We propose LUCIFER (Language Understanding and Context-Infused Framework for Exploration and Behavior Refinement), an inference-only middleware that exposes a Signal Contract. The contract provides four outputs, policy priors, reward potentials, admissible-option constraints, and telemetry-based action prediction for efficient information gathering. We validate LUCIFER in a search-and-rescue (SAR)-inspired testbed using dual-phase, dual-client evaluation: (i) component benchmarks show reasoning-based extraction remains robust on self-correcting reports where pattern-matching baselines degrade, and (ii) system-level ablations with two structurally distinct clients (hierarchical RL and a hybrid A*+heuristics planner) show consistent necessity and synergy. Grounding improves safety, discovery improves information-collection efficiency, and only their combination achieves both.
>
---
#### [replaced 011] Eka-Eval: An Evaluation Framework for Low-Resource Multilingual Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出EKA-EVAL框架，解决低资源多语言大模型的评估问题，支持多任务、多语言和本地模型，提升评估效率与可重复性。**

- **链接: [https://arxiv.org/pdf/2507.01853](https://arxiv.org/pdf/2507.01853)**

> **作者:** Samridhi Raj Sinha; Rajvee Sheth; Abhishek Upperwal; Mayank Singh
>
> **摘要:** The rapid evolution of Large Language Models' has underscored the need for evaluation frameworks that are globally applicable, flexible, and modular, and that support a wide range of tasks, model types, and linguistic settings. We introduce EKA-EVAL, a unified, end- to-end framework that combines a zero-code web interface and an interactive CLI to ensure broad accessibility. It integrates 55+ multilingual benchmarks across nine evaluation categories, supports local and proprietary models, and provides 11 core capabilities through a modular, plug-and-play architecture. Designed for scalable, multilingual evaluation with support for low-resource multilingual languages, EKA-EVAL is, to the best of our knowledge, the first suite to offer comprehensive coverage in a single platform. Comparisons against five existing baselines indicate improvements of at least 2x better on key usability measures, with the highest user satisfaction, faster setup times, and consistent benchmark reproducibility. The framework is open-source and publicly available at this https URL.
>
---
#### [replaced 012] Ice Cream Doesn't Cause Drowning: Benchmarking LLMs Against Statistical Pitfalls in Causal Inference
- **分类: cs.AI; cs.CL; cs.LG; stat.ME; stat.ML**

- **简介: 该论文属于因果推断任务，旨在解决LLMs在统计因果推理中的局限性。通过构建CausalPitfalls基准，评估并提升模型的因果推理能力。**

- **链接: [https://arxiv.org/pdf/2505.13770](https://arxiv.org/pdf/2505.13770)**

> **作者:** Jin Du; Li Chen; Xun Xian; An Luo; Fangqiao Tian; Ganghua Wang; Charles Doss; Xiaotong Shen; Jie Ding
>
> **摘要:** Reliable causal inference is essential for making decisions in high-stakes areas like medicine, economics, and public policy. However, it remains unclear whether large language models (LLMs) can handle rigorous and trustworthy statistical causal inference. Current benchmarks usually involve simplified tasks. For example, these tasks might only ask LLMs to identify semantic causal relationships or draw conclusions directly from raw data. As a result, models may overlook important statistical pitfalls, such as Simpson's paradox or selection bias. This oversight limits the applicability of LLMs in the real world. To address these limitations, we propose CausalPitfalls, a comprehensive benchmark designed to rigorously evaluate the capability of LLMs in overcoming common causal inference pitfalls. Our benchmark features structured challenges across multiple difficulty levels, each paired with grading rubrics. This approach allows us to quantitatively measure both causal reasoning capabilities and the reliability of LLMs' responses. We evaluate models using two protocols: (1) direct prompting, which assesses intrinsic causal reasoning, and (2) code-assisted prompting, where models generate executable code for explicit statistical analysis. Additionally, we validate the effectiveness of this judge by comparing its scoring with assessments from human experts. Our results reveal significant limitations in current LLMs when performing statistical causal inference. The CausalPitfalls benchmark provides essential guidance and quantitative metrics to advance the development of trustworthy causal reasoning systems.
>
---
#### [replaced 013] The Convergence of Schema-Guided Dialogue Systems and the Model Context Protocol
- **分类: cs.AI; cs.CL**

- **简介: 该论文探讨Schema-Guided Dialogue与Model Context Protocol的统一性，解决LLM-agent交互的规范问题，提出五项schema设计原则。**

- **链接: [https://arxiv.org/pdf/2602.18764](https://arxiv.org/pdf/2602.18764)**

> **作者:** Andreas Schlapbach
>
> **备注:** 18 sections, 4 figures, 7 tables, 40 references. Original research presenting: (1) formal framework mapping Schema-Guided Dialogue principles to Model Context Protocol concepts, (2) five foundational design principles for LLM-native schema authoring, (3) architectural patterns for secure, scalable agent orchestration. Research supported by SBB (Swiss Federal Railways)
>
> **摘要:** This paper establishes a fundamental convergence: Schema-Guided Dialogue (SGD) and the Model Context Protocol (MCP) represent two manifestations of a unified paradigm for deterministic, auditable LLM-agent interaction. SGD, designed for dialogue-based API discovery (2019), and MCP, now the de facto standard for LLM-tool integration, share the same core insight -- that schemas can encode not just tool signatures but operational constraints and reasoning guidance. By analyzing this convergence, we extract five foundational principles for schema design: (1) Semantic Completeness over Syntactic Precision, (2) Explicit Action Boundaries, (3) Failure Mode Documentation, (4) Progressive Disclosure Compatibility, and (5) Inter-Tool Relationship Declaration. These principles reveal three novel insights: first, SGD's original design was fundamentally sound and should be inherited by MCP; second, both frameworks leave failure modes and inter-tool relationships unexploited -- gaps we identify and resolve; third, progressive disclosure emerges as a critical production-scaling insight under real-world token constraints. We provide concrete design patterns for each principle. These principles position schema-driven governance as a scalable mechanism for AI system oversight without requiring proprietary system inspection -- central to Software 3.0.
>
---
#### [replaced 014] How Quantization Shapes Bias in Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理领域，研究量化对大模型偏差的影响，旨在解决量化带来的伦理问题。工作包括评估不同量化策略对多种偏差类型的影响。**

- **链接: [https://arxiv.org/pdf/2508.18088](https://arxiv.org/pdf/2508.18088)**

> **作者:** Federico Marcuzzi; Xuefei Ning; Roy Schwartz; Iryna Gurevych
>
> **摘要:** This work presents a comprehensive evaluation of how quantization affects model bias, with particular attention to its impact on individual demographic subgroups. We focus on weight and activation quantization strategies and examine their effects across a broad range of bias types, including stereotypes, fairness, toxicity, and sentiment. We employ both probability- and generated text-based metrics across 13 benchmarks and evaluate models that differ in architecture family and reasoning ability. Our findings show that quantization has a nuanced impact on bias: while it can reduce model toxicity and does not significantly impact sentiment, it tends to slightly increase stereotypes and unfairness in generative tasks, especially under aggressive compression. These trends are generally consistent across demographic categories and subgroups, and model types, although their magnitude depends on the specific setting. Overall, our results highlight the importance of carefully balancing efficiency and ethical considerations when applying quantization in practice.
>
---
#### [replaced 015] Why Are Linear RNNs More Parallelizable?
- **分类: cs.LG; cs.CC; cs.CL; cs.FL**

- **简介: 该论文属于自然语言处理领域，研究线性RNN为何更易并行化。通过理论分析，揭示线性RNN与非线性RNN在并行计算上的差异及其对模型表达能力的影响。**

- **链接: [https://arxiv.org/pdf/2603.03612](https://arxiv.org/pdf/2603.03612)**

> **作者:** William Merrill; Hongjian Jiang; Yanhong Li; Anthony Lin; Ashish Sabharwal
>
> **备注:** Corrected authorship list from initial version
>
> **摘要:** The community is increasingly exploring linear RNNs (LRNNs) as language models, motivated by their expressive power and parallelizability. While prior work establishes the expressivity benefits of LRNNs over transformers, it is unclear what makes LRNNs -- but not traditional, nonlinear RNNs -- as easy to parallelize in practice as transformers. We answer this question by providing a tight connection between types of RNNs and standard complexity classes. We show that LRNNs can be viewed as log-depth (bounded fan-in) arithmetic circuits, which represents only a slight depth overhead relative to log-depth boolean circuits that transformers admit. Furthermore, we show that nonlinear RNNs can solve $\mathsf{L}$-complete problems (and even $\mathsf{P}$-complete ones, under polynomial precision), revealing a fundamental barrier to parallelizing them as efficiently as transformers. Our theory also identifies fine-grained expressivity differences between recent popular LRNN variants: permutation-diagonal LRNNs are $\mathsf{NC}^1$-complete whereas diagonal-plus-low-rank LRNNs are more expressive ($\mathsf{PNC}^1$-complete). We provide further insight by associating each type of RNN with a corresponding automata-theoretic model that it can simulate. Together, our results reveal fundamental tradeoffs between nonlinear RNNs and different variants of LRNNs, providing a foundation for designing LLM architectures that achieve an optimal balance between expressivity and parallelism.
>
---
#### [replaced 016] Open Korean Historical Corpus: A Millennia-Scale Diachronic Collection of Public Domain Texts
- **分类: cs.CL**

- **简介: 该论文属于语言学与自然语言处理领域，旨在解决历史语料缺失问题。构建了涵盖千年的韩语历史语料库，分析语言演变及词汇变化，为语言模型提供预训练数据。**

- **链接: [https://arxiv.org/pdf/2510.24541](https://arxiv.org/pdf/2510.24541)**

> **作者:** Seyoung Song; Nawon Kim; Songeun Chae; Kiwoong Park; Jiho Jin; Haneul Yoo; Kyunghyun Cho; Alice Oh
>
> **备注:** LREC 2026
>
> **摘要:** The history of the Korean language is characterized by a discrepancy between its spoken and written forms and a pivotal shift from Chinese characters to the Hangul alphabet. However, this linguistic evolution has remained largely unexplored in NLP due to a lack of accessible historical corpora. To address this gap, we introduce the Open Korean Historical Corpus, a large-scale, openly licensed dataset spanning 1,300 years and 6 languages, as well as under-represented writing systems like Korean-style Sinitic (Idu) and Hanja-Hangul mixed script. This corpus contains 17.7 million documents and 5.1 billion tokens from 19 sources, ranging from the 7th century to 2025. We leverage this resource to quantitatively analyze major linguistic shifts: (1) Idu usage peaked in the 1860s before declining sharply; (2) the transition from Hanja to Hangul was a rapid transformation starting around 1890; and (3) North Korea's lexical divergence causes modern tokenizers to produce up to 51 times higher out-of-vocabulary rates. This work provides a foundational resource for quantitative diachronic analysis by capturing the history of the Korean language. Moreover, it can serve as a pre-training corpus for large language models, potentially improving their understanding of Sino-Korean vocabulary in modern Hangul as well as archaic writing systems.
>
---
#### [replaced 017] Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于语言模型推理任务，旨在提升模型在数学推理中的效率与效果。通过自蒸馏方法，让模型自身作为教师和学生，减少对额外教师模型的依赖，提高训练效率。**

- **链接: [https://arxiv.org/pdf/2601.18734](https://arxiv.org/pdf/2601.18734)**

> **作者:** Siyan Zhao; Zhihui Xie; Mengchen Liu; Jing Huang; Guan Pang; Feiyu Chen; Aditya Grover
>
> **备注:** code is release here: this https URL
>
> **摘要:** Knowledge distillation improves large language model (LLM) reasoning by compressing the knowledge of a teacher LLM to train smaller LLMs. On-policy distillation advances this approach by having the student sample its own trajectories while a teacher LLM provides dense token-level supervision, addressing the distribution mismatch between training and inference in off-policy distillation methods. However, on-policy distillation typically requires a separate, often larger, teacher LLM and does not explicitly leverage ground-truth solutions available in reasoning datasets. Inspired by the intuition that a sufficiently capable LLM can rationalize external privileged reasoning traces and teach its weaker self (i.e., the version without access to privileged information), we introduce On-Policy Self-Distillation (OPSD), a framework where a single model acts as both teacher and student by conditioning on different contexts. The teacher policy conditions on privileged information (e.g., verified reasoning traces) while the student policy sees only the question; training minimizes the per-token divergence between these distributions over the student's own rollouts. We demonstrate the efficacy of our method on multiple mathematical reasoning benchmarks, achieving 8-12x token efficiency compared to reinforcement learning methods such as GRPO and superior performance over off-policy distillation methods.
>
---
#### [replaced 018] Bielik-Q2-Sharp: A Comparative Study of Extreme 2-bit Quantization Methods for a Polish 11B Language Model
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型量化任务，旨在评估极端2比特量化方法在波兰语大模型上的效果。工作包括比较六种方法，并分析其性能与效率。**

- **链接: [https://arxiv.org/pdf/2603.04162](https://arxiv.org/pdf/2603.04162)**

> **作者:** Jakub Prejzner
>
> **备注:** 17 pages, 13 tables. All models and Hessians available at this https URL
>
> **摘要:** We present Bielik-Q2-Sharp, the first systematic academic evaluation of extreme 2-bit quantization applied to a Polish large language model. Using Bielik-11B-v2.3-Instruct (11B parameters, Mistral architecture) as our base model, we compare six state-of-the-art post-training quantization methods -- QuIP#, SpinQuant+GPTQ, ButterflyQuant, QTIP, VPTQ, and AQLM -- all calibrated on a Polish-language corpus (CulturaX-PL) with shared Hessian matrices. Our best variant (QuIP# E8P12) achieves 71.92% across 22 Polish benchmarks versus 72.07% for the IQ2_XXS baseline -- within statistical noise, at a modest size premium (3.26 GB vs. ~2.6 GB). On eq_bench, our method scores 47.14 versus 43.53 (+3.6pp), suggesting superior preservation of higher-order reasoning. QTIP achieves the best per-bit efficiency (79.4% MC acc_norm at ~2.4 bpw, 3.27 GB), matching VPTQ's quality at 35% smaller size. We additionally document a MC-generation dissociation phenomenon where rotation-based methods preserve log-likelihood quality but fail catastrophically at autoregressive generation. The entire project was conducted by a single independent researcher on cloud GPUs (this http URL) within a $285 budget. All models, Hessians, and evaluation logs are publicly available.
>
---
#### [replaced 019] ReFusion: A Diffusion Large Language Model with Parallel Autoregressive Decoding
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出ReFusion，解决自回归模型推理慢和掩码扩散模型生成不连贯的问题，通过序列重排实现并行解码，提升效率与质量。**

- **链接: [https://arxiv.org/pdf/2512.13586](https://arxiv.org/pdf/2512.13586)**

> **作者:** Jia-Nan Li; Jian Guan; Wei Wu; Chongxuan Li
>
> **摘要:** Autoregressive models (ARMs) are hindered by slow sequential inference. While masked diffusion models (MDMs) offer a parallel alternative, they suffer from critical drawbacks: high computational overhead from precluding Key-Value (KV) caching, and incoherent generation arising from learning dependencies over an intractable space of token combinations. To address these limitations, we introduce \textsc{ReFusion}, a novel masked diffusion model that integrates sequence reorganization into the causal attention framework. By elevating parallel decoding from the token level to a higher slot level, \textsc{ReFusion} interleaves inter-slot diffusion-based selection with intra-slot autoregressive infilling, while reordering newly generated slots ahead of the remaining masks after each iteration. Consequently, this design simultaneously unlocks full KV cache reuse and reduces learning complexity from an intractable token combination space to a manageable slot-level permutation space. Extensive experiments on seven diverse benchmarks show that \textsc{ReFusion} not only overwhelmingly surpasses prior MDMs with a 34\% performance gain and an over 18$\times$ speedup on average, but also bridges the performance gap to strong ARMs while maintaining a 2.33$\times$ average speedup.
>
---
#### [replaced 020] FreeAct: Freeing Activations for LLM Quantization
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于模型量化任务，旨在解决传统量化方法无法适应动态激活分布的问题。提出FreeAct框架，通过动态分配变换矩阵提升量化效果。**

- **链接: [https://arxiv.org/pdf/2603.01776](https://arxiv.org/pdf/2603.01776)**

> **作者:** Xiaohao Liu; Xiaobo Xia; Manyi Zhang; Ji-Fu Li; Xianzhi Yu; Fei Shen; Xiu Su; See-Kiong Ng; Tat-Seng Chua
>
> **备注:** 26 pages, 18 figures, 2 tables
>
> **摘要:** Quantization is pivotal for mitigating the significant memory and computational overhead of Large Language Models (LLMs). While emerging transformation-based methods have successfully enhanced quantization by projecting feature spaces onto smoother manifolds using orthogonal matrices, they typically enforce a rigid one-to-one transformation constraint. This static approach fails to account for the dynamic patterns inherent in input activations, particularly within diffusion LLMs (dLLMs) and Multimodal LLMs (MLLMs), where varying token types exhibit distinct distributions. To advance this, we propose FreeAct, a novel quantization framework that relaxes the static one-to-one constraint to accommodate dynamic activation disparities. Theoretically, we leverage the rank-deficient nature of activations to derive a solution space that extends beyond simple inverse matrices, enabling the decoupling of activation transformations from weights. Methodologically, FreeAct identifies token-specific dynamics (i.e., vision v.s. text, or masked tokens) and allocates distinct transformation matrices to the activation side, while maintaining a unified, static transformation for the weights. Extensive experiments across dLLMs and MLLMs demonstrate that FreeAct significantly outperforms baselines, up to 5.3% performance improvement, with in-depth analyses. Our code will be publicly released.
>
---
#### [replaced 021] Incremental Graph Construction Enables Robust Spectral Clustering of Texts
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于文本聚类任务，解决标准k-NN图在低k值时易断连的问题。提出增量k-NN构造方法，确保图连通性，提升谱聚类稳定性与效果。**

- **链接: [https://arxiv.org/pdf/2603.03056](https://arxiv.org/pdf/2603.03056)**

> **作者:** Marko Pranjić; Boshko Koloski; Nada Lavrač; Senja Pollak; Marko Robnik-Šikonja
>
> **备注:** MP and BK contributed equally
>
> **摘要:** Neighborhood graphs are a critical but often fragile step in spectral clustering of text embeddings. On realistic text datasets, standard $k$-NN graphs can contain many disconnected components at practical sparsity levels (small $k$), making spectral clustering degenerate and sensitive to hyperparameters. We introduce a simple incremental $k$-NN graph construction that preserves connectivity by design: each new node is linked to its $k$ nearest previously inserted nodes, which guarantees a connected graph for any $k$. We provide an inductive proof of connectedness and discuss implications for incremental updates when new documents arrive. We validate the approach on spectral clustering of SentenceTransformer embeddings using Laplacian eigenmaps across six clustering datasets from the Massive Text Embedding Benchmark. Compared to standard $k$-NN graphs, our method outperforms in the low-$k$ regime where disconnected components are prevalent, and matches standard $k$-NN at larger $k$.
>
---
#### [replaced 022] LatentChem: From Textual CoT to Latent Thinking in Chemical Reasoning
- **分类: physics.chem-ph; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出LatentChem，解决化学推理中语言表示与结构本质不匹配的问题，通过连续潜在空间进行推理，提升效率和性能。**

- **链接: [https://arxiv.org/pdf/2602.07075](https://arxiv.org/pdf/2602.07075)**

> **作者:** Xinwu Ye; Yicheng Mao; Jia Zhang; Yimeng Liu; Li Hao; Fang Wu; Zhiwei Li; Yuxuan Liao; Zehong Wang; Zhiyuan Liu; Zhenfei Yin; Li Yuan; Philip Torr; Huan Sun; Xiangxiang Zeng; Mengdi Wang; Le Cong; Shenghua Gao; Xiangru Tang
>
> **摘要:** Chemical large language models (LLMs) predominantly rely on explicit Chain-of-Thought (CoT) in natural language to perform complex reasoning. However, chemical reasoning is inherently continuous and structural, and forcing it into discrete linguistic tokens introduces a fundamental representation mismatch that constrains both efficiency and performance. We introduce LatentChem, a latent reasoning interface that decouples chemical computation from textual generation, enabling models to perform multi-step reasoning directly in continuous latent space while emitting language only for final outputs. Remarkably, we observe a consistent emergent behavior: when optimized solely for task success, models spontaneously internalize reasoning, progressively abandoning verbose textual derivations in favor of implicit latent computation. This shift is not merely stylistic but computationally advantageous. Across diverse chemical reasoning benchmarks, LatentChem achieves a 59.88\% non-tie win rate over strong CoT-based baselines on ChemCoTBench, while delivering a 10.84$\times$ average inference speedup. Our results provide empirical evidence that chemical reasoning is more naturally and effectively realized as continuous latent dynamics rather than discretized linguistic trajectories.
>
---
#### [replaced 023] Llama-Mimi: Exploring the Limits of Flattened Speech Language Modeling
- **分类: cs.CL**

- **简介: 该论文属于语音语言建模任务，旨在解决多层级音频编码带来的结构处理问题。通过将多层级RVQ tokens扁平化并用单层Transformer建模，提升 acoustic consistency 性能。**

- **链接: [https://arxiv.org/pdf/2509.14882](https://arxiv.org/pdf/2509.14882)**

> **作者:** Issa Sugiura; Shuhei Kurita; Yusuke Oda; Ryuichiro Higashinaka
>
> **备注:** 6 pages, 1 figures
>
> **摘要:** Speech Language Models (SpeechLMs) model tokenized speech to capture both semantic and acoustic information. When neural audio codecs based on Residual Vector Quantization (RVQ) are used as audio tokenizers, they produce multiple discrete tokens per time step, yielding inherently multi-level representations. To process these multi-level tokens together, prior work typically adopts hierarchical architectures to capture this structure. In contrast, recent progress in NLP has progressively reduced architectural inductive biases, moving toward simpler and more scalable single-Transformer architectures. In this work, we propose Llama-Mimi, which flattens multi-level RVQ tokens produced by the Mimi neural audio codec into a single sequence and models them autoregressively with a Transformer decoder. We show that Llama-Mimi outperforms a CSM-based hierarchical model on most tasks and achieves the best performance on acoustic consistency. Our models, code, and speech samples are publicly available.
>
---
#### [replaced 024] Beyond Prefixes: Graph-as-Memory Cross-Attention for Knowledge Graph Completion with Large Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于知识图谱补全任务，解决LLM与知识图谱融合不足的问题。提出GMT方法，通过图作为记忆进行深度交叉注意力融合，提升推理能力。**

- **链接: [https://arxiv.org/pdf/2510.08966](https://arxiv.org/pdf/2510.08966)**

> **作者:** Ruitong Liu; Boxu Lin; Peize Li; Siyuan Li; Yunjia Wu; Te Sun; Chaohan Wu
>
> **摘要:** Fusing Knowledge Graphs with Large Language Models (LLMs) is crucial for knowledge-intensive tasks like knowledge graph completion. Existing LLM-based approaches typically inject graph information via prefix concatenation, resulting in shallow interactions that fail to support fine-grained evidence retrieval during generation. Beyond prefixes, we propose Graph-as-Memory Tuning (GMT), a new paradigm that represents local graph structure as explicit graph memory and injects it into LLMs via deep, token-wise cross-attention. Specifically, GMT first employs a Semantic Graph Module to encode context-aware semantics from local neighborhoods guided by knowledge-enhanced relations, and compresses them into a fixed number of graph memory tokens. A Graph-as-Memory Cross-Attention Fusion Module then integrates these tokens into multiple Transformer layers, allowing LLM hidden state to dynamically retrieve relevant graph evidence. To enable efficient adaptation, GMT applies LoRA only to the memory cross-attention while keeping the base LLM frozen. Extensive experiments show that GMT significantly outperforms prefix-tuning and other strong baselines, providing more potent signals for robust reasoning. The code is published at this https URL.
>
---
#### [replaced 025] TSPC: A Two-Stage Phoneme-Centric Architecture for code-switching Vietnamese-English Speech Recognition
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于语音识别任务，解决越英混语识别问题。提出TSPC模型，通过两阶段音素中心架构提升识别准确率。**

- **链接: [https://arxiv.org/pdf/2509.05983](https://arxiv.org/pdf/2509.05983)**

> **作者:** Tran Nguyen Anh; Truong Dinh Dung; Vo Van Nam; Minh N. H. Nguyen
>
> **备注:** Update new version
>
> **摘要:** Code-switching (CS) presents a significant challenge for general Auto-Speech Recognition (ASR) systems. Existing methods often fail to capture the sub tle phonological shifts inherent in CS scenarios. The challenge is particu larly difficult for language pairs like Vietnamese and English, where both distinct phonological features and the ambiguity arising from similar sound recognition are present. In this paper, we propose a novel architecture for Vietnamese-English CS ASR, a Two-Stage Phoneme-Centric model (TSPC). TSPC adopts a phoneme-centric approach based on an extended Vietnamese phoneme set as an intermediate representation for mixed-lingual modeling, while remaining efficient under low computational-resource constraints. Ex perimental results demonstrate that TSPC consistently outperforms exist ing baselines, including PhoWhisper-base, in Vietnamese-English CS ASR, achieving a significantly lower word error rate of 19.06% with reduced train ing resources. Furthermore, the phonetic-based two-stage architecture en ables phoneme adaptation and language conversion to enhance ASR perfor mance in complex CS Vietnamese-English ASR scenarios.
>
---
#### [replaced 026] Traceable Evidence Enhanced Visual Grounded Reasoning: Evaluation and Methodology
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉推理任务，旨在评估和提升模型的视觉接地推理能力。针对现有缺乏基准的问题，提出TreeBench基准和TreeVGR方法，提升模型的可解释性和准确性。**

- **链接: [https://arxiv.org/pdf/2507.07999](https://arxiv.org/pdf/2507.07999)**

> **作者:** Haochen Wang; Xiangtai Li; Zilong Huang; Anran Wang; Jiacong Wang; Tao Zhang; Jiani Zheng; Sule Bai; Zijian Kang; Jiashi Feng; Zhuochen Wang; Zhaoxiang Zhang
>
> **备注:** ICLR 2026 Camera Ready Version
>
> **摘要:** Models like OpenAI-o3 pioneer visual grounded reasoning by dynamically referencing visual regions, just like human "thinking with images". However, no benchmark exists to evaluate these capabilities holistically. To bridge this gap, we propose TreeBench (Traceable Evidence Evaluation Benchmark), a diagnostic benchmark built on three principles: (1) focused visual perception of subtle targets in complex scenes, (2) traceable evidence via bounding box evaluation, and (3) second-order reasoning to test object interactions and spatial hierarchies beyond simple object localization. Prioritizing images with dense objects, we initially sample 1K high-quality images from SA-1B, and incorporate eight LMM experts to manually annotate questions, candidate options, and answers for each image. After three stages of quality control, TreeBench consists of 405 challenging visual question-answering pairs, even the most advanced models struggle with this benchmark, where none of them reach 60% accuracy, e.g., OpenAI-o3 scores only 54.87. Furthermore, we introduce TreeVGR (Traceable Evidence Enhanced Visual Grounded Reasoning), a training paradigm to supervise localization and reasoning jointly with reinforcement learning, enabling accurate localizations and explainable reasoning pathways. Initialized from Qwen2.5-VL-7B, it improves V* Bench (+16.8), MME-RealWorld (+12.6), and TreeBench (+13.4), proving traceability is key to advancing vision-grounded reasoning. The code is available at this https URL.
>
---
#### [replaced 027] Pretraining Large Language Models with NVFP4
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于大语言模型预训练任务，旨在提升4-bit精度下的训练效率与稳定性。通过引入RHT、量化方案等方法，实现高效且准确的模型训练。**

- **链接: [https://arxiv.org/pdf/2509.25149](https://arxiv.org/pdf/2509.25149)**

> **作者:** NVIDIA; Felix Abecassis; Anjulie Agrusa; Dong Ahn; Jonah Alben; Stefania Alborghetti; Michael Andersch; Sivakumar Arayandi; Alexis Bjorlin; Aaron Blakeman; Evan Briones; Ian Buck; Bryan Catanzaro; Muya Chang; Jinhang Choi; Mike Chrzanowski; Eric Chung; Victor Cui; Steve Dai; Bita Darvish Rouhani; Carlo del Mundo; Deena Donia; Burc Eryilmaz; Henry Estela; Abhinav Goel; Oleg Goncharov; Yugi Guvvala; Robert Hesse; Russell Hewett; Herbert Hum; Ujval Kapasi; Brucek Khailany; Mikail Khona; Nick Knight; Alex Kondratenko; Ronny Krashinsky; Ben Lanir; Simon Layton; Michael Lightstone; Daniel Lo; Paulius Micikevicius; Asit Mishra; Tim Moon; Deepak Narayanan; Chao Ni; Abhijit Paithankar; Satish Pasumarthi; Ankit Patel; Mostofa Patwary; Ashwin Poojary; Gargi Prasad; Sweta Priyadarshi; Yigong Qin; Xiaowei Ren; Oleg Rybakov; Charbel Sakr; Sanjeev Satheesh; Stas Sergienko; Pasha Shamis; Kirthi Shankar; Nishant Sharma; Mohammad Shoeybi; Michael Siu; Misha Smelyanskiy; Darko Stosic; Dusan Stosic; Bor-Yiing Su; Frank Sun; Nima Tajbakhsh; Shelby Thomas; Przemek Tredak; Evgeny Tsykunov; Gandhi Vaithilingam; Aditya Vavre; Rangharajan Venkatesan; Roger Waleffe; Qiyu Wan; Hexin Wang; Mengdi Wang; Lizzie Wei; Hao Wu; Evan Wu; Keith Wyss; Ning Xu; Jinze Xue; Charlene Yang; Yujia Zhai; Ruoxi Zhang; Jingyang Zhu; Zhongbo Zhu
>
> **备注:** Update includes: (1) fixing a typo in eq. 2 (2) updating author list, and (3) adding a related work
>
> **摘要:** Large Language Models (LLMs) today are powerful problem solvers across many domains, and they continue to get stronger as they scale in model size, training set size, and training set quality, as shown by extensive research and experimentation across the industry. Training a frontier model today requires on the order of tens to hundreds of yottaflops, which is a massive investment of time, compute, and energy. Improving pretraining efficiency is therefore essential to enable the next generation of even more capable LLMs. While 8-bit floating point (FP8) training is now widely adopted, transitioning to even narrower precision, such as 4-bit floating point (FP4), could unlock additional improvements in computational speed and resource utilization. However, quantization at this level poses challenges to training stability, convergence, and implementation, notably for large-scale models trained on long token horizons. In this study, we introduce a novel approach for stable and accurate training of large language models (LLMs) using the NVFP4 format. Our method integrates Random Hadamard transforms (RHT) to bound block-level outliers, employs a two-dimensional quantization scheme for consistent representations across both the forward and backward passes, utilizes stochastic rounding for unbiased gradient estimation, and incorporates selective high-precision layers. We validate our approach by training a 12-billion-parameter model on 10 trillion tokens -- the longest publicly documented training run in 4-bit precision to date. Our results show that the model trained with our NVFP4-based pretraining technique achieves training loss and downstream task accuracies comparable to an FP8 baseline. These findings highlight that NVFP4, when combined with our training approach, represents a major step forward in narrow-precision LLM training algorithms.
>
---
#### [replaced 028] Learning Virtual Machine Scheduling in Cloud Computing through Language Agents
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于云 computing 中的虚拟机调度任务，解决 ODMBP 问题。提出 MiCo 框架，利用语言模型实现动态调度，提升适应性和性能。**

- **链接: [https://arxiv.org/pdf/2505.10117](https://arxiv.org/pdf/2505.10117)**

> **作者:** JieHao Wu; Ziwei Wang; Junjie Sheng; Wenhao Li; Xiangfeng Wang; Jun Luo
>
> **摘要:** In cloud services, virtual machine (VM) scheduling is a typical Online Dynamic Multidimensional Bin Packing (ODMBP) problem, characterized by large-scale complexity and fluctuating demands. Traditional optimization methods struggle to adapt to real-time changes, domain-expert-designed heuristic approaches suffer from rigid strategies, and existing learning-based methods often lack generalizability and interpretability. To address these limitations, this paper proposes a hierarchical language agent framework named MiCo, which provides a large language model (LLM)-driven heuristic design paradigm for solving ODMBP. Specifically, ODMBP is formulated as a Semi-Markov Decision Process with Options (SMDP-Option), enabling dynamic scheduling through a two-stage architecture, i.e., Option Miner and Option Composer. Option Miner utilizes LLMs to discover diverse and useful non-context-aware strategies by interacting with constructed environments. Option Composer employs LLMs to discover a composing strategy that integrates the non-context-aware strategies with the contextual ones. Extensive experiments on real-world enterprise datasets demonstrate that MiCo achieves a 96.9\% competitive ratio in large-scale scenarios involving more than 10,000 virtual machines. It maintains high performance even under nonstationary request flows and diverse configurations, thus validating its effectiveness in complex and large-scale cloud environments.
>
---
#### [replaced 029] ShIOEnv: A Command Evaluation Environment for Grammar-Constrained Synthesis and Execution Behavior Modeling
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于CLI交互建模任务，解决复杂输入与系统依赖行为建模问题。提出ShIOEnv环境，通过语法约束和自监督信号提升执行行为建模精度。**

- **链接: [https://arxiv.org/pdf/2505.18374](https://arxiv.org/pdf/2505.18374)**

> **作者:** Jarrod Ragsdale; Rajendra Boppana
>
> **备注:** 15 pages, 7 figures, conference preprint
>
> **摘要:** Modeling of command-line interface (CLI) interaction has enabled flexible, execution-free output presentation. However, current approaches struggle to model inputs with complex compositions and inputs whose execution behavior depends on system characteristics. This is due to a lack of shell input-output (ShIO) data in the training distributions used by the models in these approaches. To address this data gap, we present ShIOEnv, a Gymnasium-compatible Bash shell environment for command synthesis and system-grounded execution behavior capturing. To concentrate synthesis on productive regions of the state-action space, we temporally abstract argument construction into grammar-derived options, thereby constraining synthesis to syntactically valid arguments. We introduce a self-supervised irreducibility signal to approximate the proportion of arguments that contribute to the observed execution behavior, serving as a measure of information density for each input. Using ShIOEnv, we curate and release 2.1M input-output pairs for modeling feedback from Bash command execution. We find that models trained on grammar-constrained datasets with higher maximum irreducibility achieve greater accuracy when modeling the execution behavior of user-sourced inputs than prior execution-free baselines.
>
---
#### [replaced 030] AgentIR: Reasoning-Aware Retrieval for Deep Research Agents
- **分类: cs.CL**

- **简介: 该论文属于信息检索任务，旨在解决深度研究代理在搜索时缺乏上下文理解的问题。提出Reasoning-Aware Retrieval和DR-Synth方法，提升检索效果。**

- **链接: [https://arxiv.org/pdf/2603.04384](https://arxiv.org/pdf/2603.04384)**

> **作者:** Zijian Chen; Xueguang Ma; Shengyao Zhuang; Jimmy Lin; Akari Asai; Victor Zhong
>
> **摘要:** Deep Research agents are rapidly emerging as primary consumers of modern retrieval systems. Unlike human users who issue and refine queries without documenting their intermediate thought processes, Deep Research agents generate explicit natural language reasoning before each search call, revealing rich intent and contextual information that existing retrievers entirely ignore. To exploit this overlooked signal, we introduce: (1) Reasoning-Aware Retrieval, a retrieval paradigm that jointly embeds the agent's reasoning trace alongside its query; and (2) DR-Synth, a data synthesis method that generates Deep Research retriever training data from standard QA datasets. We demonstrate that both components are independently effective, and their combination yields a trained embedding model, AgentIR-4B, with substantial gains. On the challenging BrowseComp-Plus benchmark, AgentIR-4B achieves 68\% accuracy with the open-weight agent Tongyi-DeepResearch, compared to 50\% with conventional embedding models twice its size, and 37\% with BM25. Code and data are available at: this https URL.
>
---
#### [replaced 031] Think-While-Generating: On-the-Fly Reasoning for Personalized Long-Form Generation
- **分类: cs.CL**

- **简介: 该论文属于个性化长文本生成任务，旨在解决现有方法难以捕捉用户隐式偏好、效率低的问题。提出FlyThinker框架，实现生成过程中动态推理，提升个性化效果与效率。**

- **链接: [https://arxiv.org/pdf/2512.06690](https://arxiv.org/pdf/2512.06690)**

> **作者:** Chengbing Wang; Yang Zhang; Wenjie Wang; Xiaoyan Zhao; Fuli Feng; Xiangnan He; Tat-Seng Chua
>
> **备注:** Published as a conference paper at ICLR 2026
>
> **摘要:** Preference alignment has enabled large language models (LLMs) to better reflect human expectations, but current methods mostly optimize for population-level preferences, overlooking individual users. Personalization is essential, yet early approaches-such as prompt customization or fine-tuning-struggle to reason over implicit preferences, limiting real-world effectiveness. Recent "think-then-generate" methods address this by reasoning before response generation. However, they face challenges in long-form generation: their static one-shot reasoning must capture all relevant information for the full response generation, making learning difficult and limiting adaptability to evolving content. To address this issue, we propose FlyThinker, an efficient "think-while-generating" framework for personalized long-form generation. FlyThinker employs a separate reasoning model that generates latent token-level reasoning in parallel, which is fused into the generation model to dynamically guide response generation. This design enables reasoning and generation to run concurrently, ensuring inference efficiency. In addition, the reasoning model is designed to depend only on previous responses rather than its own prior outputs, which preserves training parallelism across different positions-allowing all reasoning tokens for training data to be produced in a single forward pass like standard LLM training, ensuring training efficiency. Extensive experiments on real-world benchmarks demonstrate that FlyThinker achieves better personalized generation while keeping training and inference efficiency. Our code is available at this https URL.
>
---
#### [replaced 032] Where is the multimodal goal post? On the Ability of Foundation Models to Recognize Contextually Important Moments
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文研究基础模型识别视频中关键事件的能力，针对足球比赛视频，构建数据集评估模型表现，发现其性能接近随机，需改进多模态融合方法。**

- **链接: [https://arxiv.org/pdf/2601.16333](https://arxiv.org/pdf/2601.16333)**

> **作者:** Aditya K Surikuchi; Raquel Fernández; Sandro Pezzelle
>
> **摘要:** Foundation models are used for many real-world applications involving language generation from temporally-ordered multimodal events. In this work, we study the ability of models to identify the most important sub-events in a video, which is a fundamental prerequisite for narrating or summarizing multimodal events. Specifically, we focus on football games and evaluate models on their ability to distinguish between important and non-important sub-events in a game. To this end, we construct a new dataset by leveraging human preferences for importance implicit in football game highlight reels, without any additional annotation costs. Using our dataset, we compare several state-of-the-art multimodal models and show that they are not far from chance level performance. Analyses of models beyond standard evaluation metrics reveal their tendency to rely on a single dominant modality and their ineffectiveness in synthesizing necessary information from multiple sources. Our findings underline the importance of modular architectures that can handle sample-level heterogeneity in multimodal data and the need for complementary training procedures that can maximize cross-modal synergy.
>
---
#### [replaced 033] EasyAnimate: High-Performance Video Generation Framework with Hybrid Windows Attention and Reward Backpropagation
- **分类: cs.CV; cs.CL; cs.MM**

- **简介: 该论文属于视频生成任务，旨在解决生成速度慢和质量不高的问题。提出EasyAnimate框架，采用混合窗口注意力和奖励反向传播优化模型性能。**

- **链接: [https://arxiv.org/pdf/2405.18991](https://arxiv.org/pdf/2405.18991)**

> **作者:** Jiaqi Xu; Kunzhe Huang; Xinyi Zou; Yunkuo Chen; Bo Liu; MengLi Cheng; Jun Huang; Xing Shi
>
> **备注:** 10 pages, 8 figures, ACM MM 2025
>
> **摘要:** This paper introduces EasyAnimate, an efficient and high quality video generation framework that leverages diffusion transformers to achieve high-quality video production, encompassing data processing, model training, and end-to-end inference. Despite substantial advancements achieved by video diffusion models, existing video generation models still struggles with slow generation speeds and less-than-ideal video quality. To improve training and inference efficiency without compromising performance, we propose Hybrid Window Attention. We design the multidirectional sliding window attention in Hybrid Window Attention, which provides stronger receptive capabilities in 3D dimensions compared to naive one, while reducing the model's computational complexity as the video sequence length increases. To enhance video generation quality, we optimize EasyAnimate using reward backpropagation to better align with human preferences. As a post-training method, it greatly enhances the model's performance while ensuring efficiency. In addition to the aforementioned improvements, EasyAnimate integrates a series of further refinements that significantly improve both computational efficiency and model performance. We introduce a new training strategy called Training with Token Length to resolve uneven GPU utilization in training videos of varying resolutions and lengths, thereby enhancing efficiency. Additionally, we use a multimodal large language model as the text encoder to improve text comprehension of the model. Experiments demonstrate significant enhancements resulting from the above improvements. The EasyAnimate achieves state-of-the-art performance on both the VBench leaderboard and human evaluation. Code and pre-trained models are available at this https URL.
>
---
#### [replaced 034] Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究窄域微调对大语言模型激活的影响，揭示其留下的可读痕迹。任务为模型可解释性，解决如何通过激活差异识别微调领域的问题，通过分析激活差异和构建解释代理验证方法有效性。**

- **链接: [https://arxiv.org/pdf/2510.13900](https://arxiv.org/pdf/2510.13900)**

> **作者:** Julian Minder; Clément Dumas; Stewart Slocum; Helena Casademunt; Cameron Holmes; Robert West; Neel Nanda
>
> **备注:** ICLR 2026
>
> **摘要:** Finetuning on narrow domains has become an essential tool to adapt Large Language Models (LLMs) to specific tasks and to create models with known unusual properties that are useful for research. We show that narrow finetuning creates strong biases in LLM activations that can be interpreted to understand the finetuning domain. These biases can be discovered using simple tools from model diffing - the study of differences between models before and after finetuning. In particular, analyzing activation differences on the first few tokens of random text and steering by adding this difference to the model activations produces text similar to the format and general content of the finetuning data. We demonstrate that these analyses contain crucial information by creating an LLM-based interpretability agent to understand the finetuning domain. With access to the bias, the agent performs significantly better compared to baseline agents using simple prompting. Our analysis spans synthetic document finetuning for false facts, emergent misalignment, subliminal learning, and taboo word guessing game models across different architectures (Gemma, LLaMA, Qwen) and scales (1B to 32B parameters). We suspect these biases reflect overfitting and find that mixing pretraining data into the finetuning corpus largely removes them, though residual risks may remain. Our work (1) demonstrates that narrowly finetuned models have salient traces of their training objective in their activations and suggests ways to improve how they are trained, (2) warns AI safety and interpretability researchers that the common practice of using such models as a proxy for studying broader finetuning (e.g., chat-tuning) might not be realistic, and (3) highlights the need for deeper investigation into the effects of narrow finetuning and development of truly realistic case studies for model-diffing, safety and interpretability research.
>
---
#### [replaced 035] Steering Awareness: Models Can Be Trained to Detect Activation Steering
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究激活扰动的可检测性，属于模型安全与可解释性任务。解决模型是否能识别外部干预的问题，通过训练模型检测并识别扰动向量，发现其检测能力有限且不影响模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.21399](https://arxiv.org/pdf/2511.21399)**

> **作者:** Joshua Fonseca Rivera; David Demitri Africa
>
> **备注:** 26 pages, 11 figures, 16 tables
>
> **摘要:** Activation steering - adding a vector to a language model's residual stream - is widely used to elicit latent behaviors and to probe safety-relevant properties. Many steering-based evaluations implicitly assume that the model cannot tell when such an intervention has occurred. We test this assumption by fine-tuning models to report (i) whether a steering vector was injected and (ii) which concept was injected, a capability we call steering awareness. Across seven open-source instruction-tuned models, the best achieves 95.5% detection on held-out concepts and 71.2% concept identification, with no false positives on our clean controls. We find that such detection transfers to novel vectors extracted by methods that produce directions aligned with contrastive activation addition, but fail for geometrically dissimilar methods. Crucially, detection does not confer behavioral robustness; detection-trained models are consistently more susceptible to steering in realistic settings than their base counterparts. Mechanistically, steering awareness arises from a distributed transformation that progressively rotates diverse injected vectors into a shared detection direction. These findings suggest that activation steering cannot be assumed to remain an undetectable intervention, with implications for the long-term reliability of steering-based safety evaluations and interpretability techniques more broadly.
>
---
#### [replaced 036] MuRating: A High Quality Data Selecting Approach to Multilingual Large Language Model Pretraining
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决多语言大模型预训练中数据质量不足的问题。提出MuRating框架，通过翻译将英语数据质量信号扩展到17种语言，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2507.01785](https://arxiv.org/pdf/2507.01785)**

> **作者:** Zhixun Chen; Ping Guo; Wenhan Han; Yifan Zhang; Binbin Liu; Haobin Lin; Fengze Liu; Yan Zhao; Bingni Zhang; Taifeng Wang; Yin Zheng; Trevor Cohn; Meng Fang
>
> **备注:** NeurIPS 2025 poster
>
> **摘要:** Data quality is a critical driver of large language model performance, yet existing model-based selection methods focus almost exclusively on English. We introduce MuRating, a scalable framework that transfers high-quality English data-quality signals into a single rater for 17 target languages. MuRating aggregates multiple English "raters" via pairwise comparisons to learn unified document-quality scores,then projects these judgments through translation to train a multilingual evaluator on monolingual, cross-lingual, and parallel text pairs. Applied to web data, MuRating selects balanced subsets of English and multilingual content to pretrain a 1.2 B-parameter LLaMA model. Compared to strong baselines, including QuRater, AskLLM, DCLM and so on, our approach boosts average accuracy on both English benchmarks and multilingual evaluations, with especially large gains on knowledge-intensive tasks. We further analyze translation fidelity, selection biases, and underrepresentation of narrative material, outlining directions for future work.
>
---
#### [replaced 037] BeyondBench: Contamination-Resistant Evaluation of Reasoning in Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出BeyondBench，用于评估语言模型的推理能力，解决静态基准数据污染问题。通过算法生成数学问题，确保测试集未被模型训练数据污染，涵盖多种难度任务进行模型评测。**

- **链接: [https://arxiv.org/pdf/2509.24210](https://arxiv.org/pdf/2509.24210)**

> **作者:** Gaurav Srivastava; Aafiya Hussain; Zhenyu Bi; Swastik Roy; Priya Pitre; Meng Lu; Morteza Ziyadi; Xuan Wang
>
> **备注:** Accepted to ICLR 2026 Conference
>
> **摘要:** Evaluating language models fairly is increasingly difficult as static benchmarks risk contamination by training data, obscuring whether models truly reason or recall. We introduce BeyondBench, an evaluation framework using algorithmic problem generation to create mathematically grounded problems on the fly, ensuring each test remains uncontaminated. Our framework covers 44 algorithmic tasks with 117 variations across three difficulty levels: the Easy Suite (29 tasks) for arithmetic and statistics, the Medium Suite (5 tasks, 49 variations) for sequence patterns and reasoning, and the Hard Suite (10 tasks, 68 variations) for NP-complete and constraint satisfaction problems. Each task draws from a space exceeding 10^15 unique instances, with deterministically verified solutions. We evaluated 101 language models (85 open-source, 16 closed-source), spanning 0.5B to 141B parameters and multiple quantization schemes, using three-fold evaluation for robustness. Results reveal consistent reasoning deficiencies, with performance degrading sharply as complexity increases. In Hard Suite evaluations, Gemini-2.5-pro, Llama-3.3-70B, and Qwen2.5-72B achieved accuracies of 56.21%, 27.16%, and 33.37% respectively. Performance drops significantly without tool usage, with GPT-5, GPT-5-mini, and GPT-5-nano showing declines of 16.81%, 15.86%, and 43.95% in overall accuracy. Contamination resistance rests on three guarantees: (i) the problem space vastly exceeds any static dataset, (ii) every instance has a deterministically verifiable solution, and (iii) isomorphic transformations yield semantically equivalent but syntactically novel problems. BeyondBench redefines reasoning evaluation via genuine algorithmic problem-solving. Our leaderboard is at this https URL, Python package at this https URL, and codebase at this https URL.
>
---
#### [replaced 038] INMS: Memory Sharing for Large Language Model based Agents
- **分类: cs.CL**

- **简介: 该论文属于多智能体协作任务，旨在解决LLM代理在开放场景中知识共享不足的问题。提出INMS框架，实现动态记忆共享与交互优化。**

- **链接: [https://arxiv.org/pdf/2404.09982](https://arxiv.org/pdf/2404.09982)**

> **作者:** Hang Gao; Yongfeng Zhang
>
> **摘要:** While Large Language Model (LLM) based agents excel at complex tasks, their performance in open-ended scenarios is often constrained by isolated operation and reliance on static databases, missing the dynamic knowledge exchange of human dialogue. To bridge this gap, we propose the INteractive Memory Sharing (INMS) framework, an asynchronous interaction paradigm for multi-agent systems. By integrating real-time memory filtering, storage, and retrieval, INMS establishes a shared conversational memory pool. This enables continuous, dialogue-like memory sharing among agents, promoting collective self-enhancement and dynamically refining the retrieval mediator based on interaction history. Extensive experiments across three datasets demonstrate that INMS significantly improves agent performance by effectively modeling multi-agent interaction and collective knowledge sharing.
>
---
#### [replaced 039] Vector Retrieval with Similarity and Diversity: How Hard Is It?
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于向量检索任务，解决相似性与多样性平衡问题。提出VRSD模型，证明其NP难，并设计无参数算法，实验显示效果优于现有方法。**

- **链接: [https://arxiv.org/pdf/2407.04573](https://arxiv.org/pdf/2407.04573)**

> **作者:** Hang Gao; Dong Deng; Yongfeng Zhang
>
> **摘要:** Dense vector retrieval is essential for semantic queries within Natural Language Processing, particularly in knowledge-intensive applications like Retrieval-Augmented Generation (RAG). The ability to retrieve vectors that satisfy both similarity and diversity substantially enhances system performance. Although the Maximal Marginal Relevance (MMR) algorithm is widely used to balance these objectives, its reliance on a manually tuned parameter leads to optimization fluctuations and unpredictable retrieval results. Furthermore, there is a lack of sufficient theoretical analysis on the joint optimization of similarity and diversity in vector retrieval. To address these challenges, this paper introduces a novel approach that characterizes both constraints simultaneously by maximizing the similarity between the query vector and the sum of the selected candidate vectors. We formally define this optimization problem, Vectors Retrieval with Similarity and Diversity (VRSD) , and prove that it is NP-complete, establishing a rigorous theoretical bound on the inherent difficulty of this dual-objective retrieval. Subsequently, we present a parameter-free heuristic algorithm to solve VRSD. Extensive evaluations on multiple scientific QA datasets , incorporating both objective geometric metrics and LLM-simulated subjective assessments, demonstrate that our VRSD heuristic consistently outperforms established baselines, including MMR and Determinantal Point Processes (k-DPP).
>
---
#### [replaced 040] VoxKnesset: A Large-Scale Longitudinal Hebrew Speech Dataset for Aging Speaker Modeling
- **分类: eess.AS; cs.CL; cs.LG; cs.SD; eess.SP**

- **简介: 该论文提出VoxKnesset数据集，用于研究语音随年龄变化的问题。任务为语音处理中的老化建模，解决现有数据不足的问题，通过构建大规模纵向 Hebrew 语音数据集并进行模型评估。**

- **链接: [https://arxiv.org/pdf/2603.01270](https://arxiv.org/pdf/2603.01270)**

> **作者:** Yanir Marmor; Arad Zulti; David Krongauz; Adam Gabet; Yoad Snapir; Yair Lifshitz; Eran Segal
>
> **备注:** 4 pages, 5 figures, 2 tables
>
> **摘要:** Speech processing systems face a fundamental challenge: the human voice changes with age, yet few datasets support rigorous longitudinal evaluation. We introduce VoxKnesset, an open-access dataset of ~2,300 hours of Hebrew parliamentary speech spanning 2009-2025, comprising 393 speakers with recording spans of up to 15 years. Each segment includes aligned transcripts and verified demographic metadata from official parliamentary records. We benchmark modern speech embeddings (WavLM-Large, ECAPA-TDNN, Wav2Vec2-XLSR-1B) on age prediction and speaker verification under longitudinal conditions. Speaker verification EER rises from 2.15\% to 4.58\% over 15 years for the strongest model, and cross-sectionally trained age regressors fail to capture within-speaker aging, while longitudinally trained models recover a meaningful temporal signal. We publicly release the dataset and pipeline to support aging-robust speech systems and Hebrew speech processing.
>
---
#### [replaced 041] Adaptive Rollout Allocation for Online Reinforcement Learning with Verifiable Rewards
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于在线强化学习任务，旨在提升采样效率。针对固定 rollout 分配导致的效率低下问题，提出 VIP 方法，根据提示信息动态分配 rollout，降低梯度方差，提高训练效果。**

- **链接: [https://arxiv.org/pdf/2602.01601](https://arxiv.org/pdf/2602.01601)**

> **作者:** Hieu Trung Nguyen; Bao Nguyen; Wenao Ma; Yuzhi Zhao; Ruifeng She; Viet Anh Nguyen
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** Sampling efficiency is a key bottleneck in reinforcement learning with verifiable rewards. Existing group-based policy optimization methods, such as GRPO, allocate a fixed number of rollouts for all training prompts. This uniform allocation implicitly treats all prompts as equally informative, and could lead to inefficient computational budget usage and impede training progress. We introduce VIP, a Variance-Informed Predictive allocation strategy that allocates a given rollout budget to the prompts in the incumbent batch to minimize the expected gradient variance of the policy update. At each iteration, VIP uses a lightweight Gaussian process model to predict per-prompt success probabilities based on recent rollouts. These probability predictions are translated into variance estimates, which are then fed into a convex optimization problem to determine the optimal rollout allocations under a hard compute budget constraint. Empirical results show that VIP consistently improves sampling efficiency and achieves higher performance than uniform or heuristic allocation strategies in multiple benchmarks.
>
---
#### [replaced 042] Assessing Risks of Large Language Models in Mental Health Support: A Framework for Automated Clinical AI Red Teaming
- **分类: cs.CL; cs.AI; cs.CY; cs.HC; cs.MA**

- **简介: 该论文属于AI安全评估任务，旨在解决AI在心理健康支持中的潜在风险。通过构建模拟对话框架，评估AI助手的安全性，发现其可能引发的医源性风险，并提出可视化审计工具。**

- **链接: [https://arxiv.org/pdf/2602.19948](https://arxiv.org/pdf/2602.19948)**

> **作者:** Ian Steenstra; Paola Pedrelli; Weiyan Shi; Stacy Marsella; Timothy W. Bickmore
>
> **备注:** This paper is a condensed version of the first author's Ph.D. dissertation submitted to Northeastern University
>
> **摘要:** Large Language Models (LLMs) are increasingly utilized for mental health support; however, current safety benchmarks often fail to detect the complex, longitudinal risks inherent in therapeutic dialogue. We introduce an evaluation framework that pairs AI psychotherapists with simulated patient agents equipped with dynamic cognitive-affective models and assesses therapy session simulations against a comprehensive quality of care and risk ontology. We apply this framework to a high-impact test case, Alcohol Use Disorder, evaluating six AI agents (including ChatGPT, Gemini, and Character AI) against a clinically-validated cohort of 15 patient personas representing diverse clinical phenotypes. Our large-scale simulation (N=369 sessions) reveals critical safety gaps in the use of AI for mental health support. We identify specific iatrogenic risks, including the validation of patient delusions ("AI Psychosis") and failure to de-escalate suicide risk. Finally, we validate an interactive data visualization dashboard with diverse stakeholders, including AI engineers and red teamers, mental health professionals, and policy experts (N=9), demonstrating that this framework effectively enables stakeholders to audit the "black box" of AI psychotherapy. These findings underscore the critical safety risks of AI-provided mental health support and the necessity of simulation-based clinical red teaming before deployment.
>
---
#### [replaced 043] Grasp Any Region: Towards Precise, Contextual Pixel Understanding for Multimodal LLMs
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出GAR模型，解决多模态大语言模型在区域级细粒度理解与跨区域关系建模上的不足，通过全局上下文感知和多提示交互，提升对任意区域的精准理解和复杂推理能力。**

- **链接: [https://arxiv.org/pdf/2510.18876](https://arxiv.org/pdf/2510.18876)**

> **作者:** Haochen Wang; Yuhao Wang; Tao Zhang; Yikang Zhou; Yanwei Li; Jiacong Wang; Jiani Zheng; Ye Tian; Jiahao Meng; Zilong Huang; Guangcan Mai; Anran Wang; Yunhai Tong; Zhuochen Wang; Xiangtai Li; Zhaoxiang Zhang
>
> **备注:** ICLR 2026 Camera Ready Version
>
> **摘要:** While Multimodal Large Language Models (MLLMs) excel at holistic understanding, they struggle in capturing the dense world with complex scenes, requiring fine-grained analysis of intricate details and object inter-relationships. Region-level MLLMs have been a promising step. However, previous attempts are generally optimized to understand given regions in isolation, neglecting crucial global contexts. To address this, we introduce Grasp Any Region (GAR) for comprehen- sive region-level visual understanding. Empowered by an effective RoI-aligned feature replay technique, GAR supports (1) precise perception by leveraging necessary global contexts, and (2) modeling interactions between multiple prompts. Together, it then naturally achieves (3) advanced compositional reasoning to answer specific free-form questions about any region, shifting the paradigm from passive description to active dialogue. Moreover, we construct GAR-Bench, which not only provides a more accurate evaluation of single-region comprehension, but also, more importantly, measures interactions and complex reasoning across multiple regions. Extensive experiments have demonstrated that GAR-1B not only maintains the state-of-the-art captioning capabilities, e.g., outperforming DAM-3B +4.5 on DLC-Bench, but also excels at modeling relationships between multiple prompts with advanced comprehension capabilities, even surpassing InternVL3-78B on GAR-Bench-VQA. More importantly, our zero-shot GAR-8B even outperforms in-domain VideoRefer-7B on VideoRefer-BenchQ, indicating its strong capabilities can be easily transferred to videos.
>
---
#### [replaced 044] Conversational Speech Reveals Structural Robustness Failures in SpeechLLM Backbones
- **分类: cs.CL; cs.AI; eess.AS**

- **简介: 该论文研究语音LLM在对话语音中的结构鲁棒性问题，通过分析非流畅内容揭示模型的编辑策略。任务为语音理解中的结构修复，解决模型对口语输入的处理偏差。工作包括评估不同模型并分析其编辑行为。**

- **链接: [https://arxiv.org/pdf/2509.20321](https://arxiv.org/pdf/2509.20321)**

> **作者:** Maria Teleki; Sai Janjur; Haoran Liu; Oliver Grabner; Ketan Verma; Thomas Docog; Xiangjue Dong; Lingfeng Shi; Cong Wang; Stephanie Birkelbach; Jason Kim; Yin Zhang; Éva Székely; James Caverlee
>
> **摘要:** LLMs serve as the backbone in SpeechLLMs, yet their behavior on spontaneous conversational input remains poorly understood. Conversational speech contains pervasive disfluencies -- interjections, edits, and parentheticals -- that are rare in the written corpora used for pre-training. Because gold disfluency removal is a deletion-only task, it serves as a controlled probe to determine whether a model performs faithful structural repair or biased reinterpretation. Using the DRES evaluation framework, we evaluate proprietary and open-source LLMs across architectures and scales. We show that model performance clusters into stable precision-recall regimes reflecting distinct editing policies. Notably, reasoning models systematically over-delete fluent content, revealing a bias toward semantic abstraction over structural fidelity. While fine-tuning achieves SOTA results, it harms generalization. Our findings demonstrate that robustness to speech is shaped by specific training objectives.
>
---
#### [replaced 045] Identifying Good and Bad Neurons for Task-Level Controllable LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，旨在解决LLM中任务相关神经元识别问题。提出NeuronLLM框架，通过对比学习区分促进与抑制任务的神经元，提升对LLM功能结构的理解。**

- **链接: [https://arxiv.org/pdf/2601.04548](https://arxiv.org/pdf/2601.04548)**

> **作者:** Wenjie Li; Guansong Pang; Hezhe Qiao; Debin Gao; David Lo
>
> **摘要:** Large Language Models have demonstrated remarkable capabilities on multiple-choice question answering benchmarks, but the complex mechanisms underlying their large-scale neurons remain opaque, posing significant challenges for understanding and steering LLMs. While recent studies made progress on identifying responsible neurons for certain abilities, these ability-specific methods are infeasible for task-focused scenarios requiring coordinated use of multiple abilities. Moreover, these approaches focus only on supportive neurons that correlate positively with task completion, while neglecting neurons with other roles-such as inhibitive roles-and misled neuron attribution due to fortuitous behaviors in LLMs (i.e., correctly answer the questions by chance rather than genuine understanding). To address these challenges, we propose NeuronLLM, a novel task-level LLM understanding framework that adopts the biological principle of functional antagonism for LLM neuron identification. The key insight is that task performance is jointly determined by neurons with two opposing roles: good neurons that facilitate task completion and bad neurons that inhibit it. NeuronLLM achieves a holistic modeling of neurons via contrastive learning of good and bad neurons, while leveraging augmented question sets to mitigate the fortuitous behaviors in LLMs. Comprehensive experiments on LLMs of different sizes and families show the superiority of NeuronLLM over existing methods in four NLP tasks, providing new insights into LLM functional organization.
>
---
#### [replaced 046] Linguistic trajectories of bipolar disorder on social media
- **分类: cs.CL**

- **简介: 该论文属于情感分析任务，旨在通过社交媒体语言研究双相障碍的长期变化。工作包括利用用户自述推断诊断时间线，比较不同群体的语言特征，发现BD诊断后语言模式的周期性变化。**

- **链接: [https://arxiv.org/pdf/2509.10035](https://arxiv.org/pdf/2509.10035)**

> **作者:** Laurin Plank; Armin Zlomuzica
>
> **备注:** Pre-print
>
> **摘要:** Language use offers valuable insight into affective disorders such as bipolar disorder (BD), yet past research has been cross-sectional and limited in scale. Here, we demonstrate that social media records can be leveraged to study longitudinal language change associated with BD on a large scale. Using a novel method to infer diagnosis timelines from user self-reports, we compared users self-identifying with BD, depression, or no mental health condition. The onset of BD diagnosis corresponded with widespread linguistic shifts reflecting mood disturbance, psychiatric comorbidity, substance abuse, hospitalization, medical comorbidities, interpersonal concerns, unusual thought content, and altered linguistic coherence. In the years following the diagnosis, discussions of mood symptoms were found to fluctuate periodically with a dominant 12-month cycle consistent with seasonal mood variation. These findings suggest that social media language captures linguistic and behavioral changes associated with BD and might serve as a valuable complement to traditional psychiatric cohort research.
>
---
#### [replaced 047] New Insights into Optimal Alignment of Acoustic and Linguistic Representations for Knowledge Transfer in ASR
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语音识别任务，旨在解决声学与语言表示对齐问题。通过构建新的对齐模型，处理结构不对称和分布不匹配，提升知识迁移效果。**

- **链接: [https://arxiv.org/pdf/2509.05609](https://arxiv.org/pdf/2509.05609)**

> **作者:** Xugang Lu; Peng Shen; Hisashi Kawai
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** Aligning acoustic and linguistic representations is a central challenge to bridge the pre-trained models in knowledge transfer for automatic speech recognition (ASR). This alignment is inherently structured and asymmetric: while multiple consecutive acoustic frames typically correspond to a single linguistic token (many-to-one), certain acoustic transition regions may relate to multiple adjacent tokens (one-to-many). Moreover, acoustic sequences often include frames with no linguistic counterpart, such as background noise or silence may lead to imbalanced matching conditions. In this work, we take a new insight to regard alignment and matching as a detection problem, where the goal is to identify meaningful correspondences with high precision and recall ensuring full coverage of linguistic tokens while flexibly handling redundant or noisy acoustic frames in transferring linguistic knowledge for ASR. Based on this new insight, we propose an unbalanced optimal transport-based alignment model that explicitly handles distributional mismatch and structural asymmetries with soft and partial matching between acoustic and linguistic modalities. Our method ensures that every linguistic token is grounded in at least one acoustic observation, while allowing for flexible, probabilistic mappings from acoustic to linguistic units. We evaluate our proposed model with experiments on an CTC-based ASR system with a pre-trained language model for knowledge transfer. Experimental results demonstrate the effectiveness of our approach in flexibly controlling degree of matching and hence to improve ASR performance.
>
---
#### [replaced 048] From Word to World: Can Large Language Models be Implicit Text-based World Models?
- **分类: cs.CL**

- **简介: 该论文研究大语言模型作为文本环境中的世界模型的可行性，旨在提升强化学习效率。通过框架评估，发现其在特定条件下能有效支持智能体学习。**

- **链接: [https://arxiv.org/pdf/2512.18832](https://arxiv.org/pdf/2512.18832)**

> **作者:** Yixia Li; Hongru Wang; Jiahao Qiu; Zhenfei Yin; Dongdong Zhang; Cheng Qian; Zeping Li; Pony Ma; Guanhua Chen; Heng Ji
>
> **摘要:** Agentic reinforcement learning increasingly relies on experience-driven scaling, yet real-world environments remain non-adaptive, limited in coverage, and difficult to scale. World models offer a potential way to improve learning efficiency through simulated experience, but it remains unclear whether large language models can reliably serve this role and under what conditions they meaningfully benefit agents. We study these questions in text-based environments, which provide a controlled setting to reinterpret language modeling as next-state prediction under interaction. We introduce a three-level framework for evaluating LLM-based world models: (i) fidelity and consistency, (ii) scalability and robustness, and (iii) agent utility. Across five representative environments, we find that sufficiently trained world models maintain coherent latent state, scale predictably with data and model size, and improve agent performance via action verification, synthetic trajectory generation, and warm-starting reinforcement learning. Meanwhile, these gains depend critically on behavioral coverage and environment complexity, delineating clear boundry on when world modeling effectively supports agent learning.
>
---
#### [replaced 049] PrefDisco: Benchmarking Proactive Personalized Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于个性化推理任务，旨在解决LLM在缺乏用户历史信息时无法准确匹配用户需求的问题。提出PrefDisco评估方法与PrefAlign指标，以衡量和提升模型的个性化对齐能力。**

- **链接: [https://arxiv.org/pdf/2510.00177](https://arxiv.org/pdf/2510.00177)**

> **作者:** Shuyue Stella Li; Avinandan Bose; Faeze Brahman; Simon Shaolei Du; Pang Wei Koh; Maryam Fazel; Yulia Tsvetkov
>
> **备注:** 65 pages, 6 figures
>
> **摘要:** Current large language model (LLM) development treats task-solving and preference-alignment as separate challenges, optimizing first for objective correctness, then for alignment to aggregated human preferences. This paradigm fails in human-facing applications where solving a problem correctly is insufficient if the response mismatches the user's needs. This challenge intensifies in just-in-time scenarios where no prior user interaction history exists due to cold-start conditions or privacy constraints. LLMs need to proactively identify what they don't know about the user, strategically elicit preference values through questioning, then adapt their reasoning processes and responses accordingly -- a complicated chain of cognitive processes which we term personalized reasoning. We introduce PrefDisco, an evaluation methodology that transforms static benchmarks into interactive personalization tasks using psychologically-grounded personas with sparse, context-dependent preferences, and define PrefAlign as a fine-grained rubric-based metric for measuring preference alignment. PrefDisco builds scenarios where identical questions require different reasoning chains depending on user context, as optimal explanation approaches vary by individual expertise and preferences while maintaining factual accuracy. Evaluation of 21 frontier models across 10 tasks reveals 29.0% of naive personalization attempts produce worse preference alignment than generic responses, yet generic responses also fail to serve individual user needs. These findings suggest personalized reasoning requires dedicated development rather than emerging naturally. PrefDisco provides a foundation for developing systems that can adapt to individual users in education, healthcare, and technical domains where personalization is critical.
>
---
#### [replaced 050] Why Reinforcement Fine-Tuning Enables MLLMs Preserve Prior Knowledge Better: A Data Perspective
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究后训练算法对模型保留先验知识的影响，比较SFT与RFT的效果，发现RFT更利于知识保持，并提出数据分布对遗忘的关键作用。**

- **链接: [https://arxiv.org/pdf/2506.23508](https://arxiv.org/pdf/2506.23508)**

> **作者:** Zhihao Zhang; Qiaole Dong; Qi Zhang; Jun Zhao; Enyu Zhou; Zhiheng Xi; Senjie Jin; Xiaoran Fan; Yuhao Zhou; Mingqi Wu; Yanwei Fu; Tao Ji; Tao Gui; Xuanjing Huang; Kai Chen
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Post-training algorithms such as Supervised Fine-Tuning (SFT) and Reinforcement Fine-Tuning (RFT) are widely used to adapt (multimodal) large language models to downstream tasks. While effective at task adaptation, their impact on retaining prior knowledge remains unclear. In this paper, we introduce jigsaw puzzles as a novel task absent from existing pretraining corpora and systematically study the behavior of SFT and RFT on the open-source Qwen2.5-VL series. Our experiments reveal a sharp trade-off: SFT enables rapid task acquisition but leads to catastrophic forgetting, whereas RFT learns more slowly but better maintains prior knowledge. We study this phenomenon through learning dynamics by examining both the magnitude and direction of how training data influence prior knowledge. Our analysis shows that RFT mainly reinforces correct samples naturally aligned with the base model's probability landscape, leading to weaker interference with prior knowledge. Moreover, training on RFT-simulated rollouts, which exert a smaller magnitude of influence and are better aligned in direction to prior knowledge, allows SFT to preserve prior knowledge better while rapidly learning new tasks. We further validate our framework on Qwen2.5 post-training in math and scientific QA, observing consistent forgetting and learning-dynamics trends. These findings suggest that the distribution of post-training data, rather than algorithmic differences alone, plays a central role in forgetting, and highlight RFT as a promising ingredient for stable continual post-training.
>
---
#### [replaced 051] When Do Tools and Planning Help Large Language Models Think? A Cost- and Latency-Aware Benchmark
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究LLM在特定任务中使用工具和规划的效果，旨在评估其对准确性和效率的影响。任务包括事件问答和说服性回复生成。工作对比了不同配置的性能与成本。**

- **链接: [https://arxiv.org/pdf/2601.02663](https://arxiv.org/pdf/2601.02663)**

> **作者:** Subha Ghoshal; Ali Al-Bustami
>
> **摘要:** Modern large language models (LLMs) increasingly rely on inference-time planning and external tools to improve reasoning. We benchmark this behavior on two real-world settings: event-centric question answering over graph-structured knowledge (Event-QA) and persuasive response generation in Reddit ChangeMyView (CMV). Using LangChain and LangGraph, we compare a one-shot baseline against a plan-execute-replan agent equipped with task-specific tools (DBpedia SPARQL/lookup/schema exploration, Wikipedia-focused retrieval, and topical web search). We evaluate on 60 examples each from Event-QA and CMV (3 splits of 20), and report both mean end-to-end latency and per-example token cost estimates. We evaluate GPT-4o and GPT-4o-mini under identical workflows and report accuracy and end-to-end latency. On Event-QA, the best tool-augmented configuration improves accuracy (e.g., 47.5\% $\rightarrow$ 67.5\% for GPT-4o) while increasing latency by orders of magnitude ($\sim$8s $\rightarrow$ $\sim$317s per example). On CMV, one-shot prompting is strongest (e.g., GPT-4o-mini achieves 75\% at $\sim$6s), and planning+search increases latency substantially without consistent gains. However, complex multi-tool orchestration exposes failure modes where the smaller model degrades. Overall, the findings highlight the need for task-specific, cost-aware choices of both model size and agent/tooling complexity.
>
---
#### [replaced 052] Learn Hard Problems During RL with Reference Guided Fine-tuning
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，解决数学推理中的奖励稀疏问题。通过引用引导微调，生成有效轨迹提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.01223](https://arxiv.org/pdf/2603.01223)**

> **作者:** Yangzhen Wu; Shanda Li; Zixin Wen; Xin Zhou; Ameet Talwalkar; Yiming Yang; Wenhao Huang; Tianle Cai
>
> **备注:** 15 pages, 5 figures
>
> **摘要:** Reinforcement learning (RL) for mathematical reasoning can suffer from reward sparsity: for challenging problems, LLM fails to sample any correct trajectories, preventing RL from receiving meaningful positive feedback. At the same time, there often exist human-written reference solutions along with the problem (e.g., problems from AoPS), but directly fine-tuning on these solutions offers no benefit because models often cannot imitate human proofs that lie outside their own reasoning distribution. We introduce Reference-Guided Fine-Tuning (ReGFT), a simple and effective method that utilizes human-written reference solutions to synthesize positive trajectories on hard problems and train on them before RL. For each problem, we provide the model with a partial reference solution and let it generate its own reasoning trace, ensuring the resulting trajectories remain in the model's reasoning space while still benefiting from reference guidance. Fine-tuning on these reference-guided trajectories increases the number of solvable problems and produces a checkpoint that receives more positive rewards during RL. Across three benchmarks (AIME24, AIME25, BeyondAIME), ReGFT consistently improves supervised accuracy, accelerates DAPO training, and raises the final performance plateau of RL. Our results show that ReGFT effectively overcomes reward sparsity and unlocks stronger RL-based mathematical reasoning.
>
---
#### [replaced 053] Jailbreak Foundry: From Papers to Runnable Attacks for Reproducible Benchmarking
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出JAILBREAK FOUNDRY系统，解决LLM安全评估中基准滞后问题，通过自动化流程实现攻击重现与标准化评估。**

- **链接: [https://arxiv.org/pdf/2602.24009](https://arxiv.org/pdf/2602.24009)**

> **作者:** Zhicheng Fang; Jingjie Zheng; Chenxu Fu; Wei Xu
>
> **摘要:** Jailbreak techniques for large language models (LLMs) evolve faster than benchmarks, making robustness estimates stale and difficult to compare across papers due to drift in datasets, harnesses, and judging protocols. We introduce JAILBREAK FOUNDRY (JBF), a system that addresses this gap via a multi-agent workflow to translate jailbreak papers into executable modules for immediate evaluation within a unified harness. JBF features three core components: (i) JBF-LIB for shared contracts and reusable utilities; (ii) JBF-FORGE for the multi-agent paper-to-module translation; and (iii) JBF-EVAL for standardizing evaluations. Across 30 reproduced attacks, JBF achieves high fidelity with a mean (reproduced-reported) attack success rate (ASR) deviation of +0.26 percentage points. By leveraging shared infrastructure, JBF reduces attack-specific implementation code by nearly half relative to original repositories and achieves an 82.5% mean reused-code ratio. This system enables a standardized AdvBench evaluation of all 30 attacks across 10 victim models using a consistent GPT-4o judge. By automating both attack integration and standardized evaluation, JBF offers a scalable solution for creating living benchmarks that keep pace with the rapidly shifting security landscape.
>
---
#### [replaced 054] SealQA: Raising the Bar for Reasoning in Search-Augmented Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出SealQA，用于评估搜索增强语言模型在事实性问题上的推理能力，解决模型在噪声搜索结果下的准确性和长文档推理问题。**

- **链接: [https://arxiv.org/pdf/2506.01062](https://arxiv.org/pdf/2506.01062)**

> **作者:** Thinh Pham; Nguyen Nguyen; Pratibha Zunjare; Weiyuan Chen; Yu-Min Tseng; Tu Vu
>
> **备注:** Camera Ready version for ICLR 2026
>
> **摘要:** We introduce SealQA, a new challenge benchmark for evaluating SEarch-Augmented Language models on fact-seeking questions where web search yields conflicting, noisy, or unhelpful results. SealQA comes in three flavors: (1) Seal-0 (main) and (2) Seal-Hard, which assess factual accuracy and reasoning capabilities, with Seal-0 focusing on the most challenging questions where chat models (e.g., GPT-4.1) typically achieve near-zero accuracy; and (3) LongSeal, which extends SealQA to test long-context, multi-document reasoning in "needle-in-a-haystack" settings. Our evaluation reveals critical limitations in current models: Even frontier LLMs perform poorly across all SealQA flavors. On Seal-0, frontier agentic models equipped with tools like o3 and o4-mini achieve only 17.1% and 6.3% accuracy, respectively, at their best reasoning efforts. We find that advanced reasoning models such as DeepSeek-R1-671B and o3-mini are highly vulnerable to noisy search results. Notably, increasing test-time compute does not yield reliable gains across o3-mini, o4-mini, and o3, with performance often plateauing or even declining early. Additionally, while recent models are less affected by the "lost-in-the-middle" issue, they still fail to reliably identify relevant documents in LongSeal when faced with numerous distractors. To facilitate future work, we release SealQA at this http URL.
>
---
#### [replaced 055] Vevo2: A Unified and Controllable Framework for Speech and Singing Voice Generation
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文提出Vevo2，解决语音与歌唱生成的可控性问题。通过两个音频分词器和两种建模阶段，实现语音与歌唱的统一生成与控制。**

- **链接: [https://arxiv.org/pdf/2508.16332](https://arxiv.org/pdf/2508.16332)**

> **作者:** Xueyao Zhang; Junan Zhang; Yuancheng Wang; Chaoren Wang; Yuanzhe Chen; Dongya Jia; Zhuo Chen; Zhizheng Wu
>
> **备注:** Accepted by the IEEE Transactions on Audio, Speech and Language Processing (TASLP)
>
> **摘要:** Controllable human voice generation, particularly for expressive domains like singing, remains a significant challenge. This paper introduces Vevo2, a unified framework for controllable speech and singing voice generation. To tackle issues like the scarcity of annotated singing data and to enable flexible controllability, Vevo2 introduces two audio tokenizers: (1) a unified music-notation-free prosody tokenizer that captures prosody and melody from speech, singing, and even instrumental sounds, and (2) a unified content-style tokenizer that encodes linguistic content, prosody, and style for both speech and singing, while enabling timbre disentanglement. Vevo2 consists of an auto-regressive (AR) content-style modeling stage, which aims to enable controllability over text, prosody, and style, as well as a flow-matching acoustic modeling stage that allows for timbre control. Particularly, during the speech-singing joint training of the AR model, we propose both explicit and implicit prosody learning strategies to bridge speech and singing voice. Moreover, to further enhance the Vevo2's ability to follow text and prosody, we design a multi-objective post-training task that integrates both intelligibility and prosody similarity alignment. Experimental results show that the unified modeling in Vevo2 brings mutual benefits to both speech and singing voice generation. Additionally, Vevo2's effectiveness across a wide range of synthesis, conversion, and editing tasks for both speech and singing further demonstrates its strong generalization ability and versatility. Audio samples are are available at this https URL.
>
---
#### [replaced 056] F-Actor: Controllable Conversational Behaviour in Full-Duplex Models
- **分类: cs.CL**

- **简介: 该论文提出F-Actor，一个可控制的全双工对话模型，解决对话行为定制化问题。通过微调语言模型实现语音、话题等控制，仅需2000小时数据，无需大规模预训练。**

- **链接: [https://arxiv.org/pdf/2601.11329](https://arxiv.org/pdf/2601.11329)**

> **作者:** Maike Züfle; Ondrej Klejch; Nicholas Sanders; Jan Niehues; Alexandra Birch; Tsz Kin Lam
>
> **摘要:** Spoken conversational systems require more than accurate speech generation to have human-like conversations: to feel natural and engaging, they must produce conversational behaviour that adapts dynamically to the context. Current spoken conversational systems, however, rarely allow such customization, limiting their naturalness and usability. In this work, we present the first open, instruction-following full-duplex conversational speech model that can be trained efficiently under typical academic resource constraints. By keeping the audio encoder frozen and finetuning only the language model, our model requires just 2,000 hours of data, without relying on large-scale pretraining or multi-stage optimization. The model can follow explicit instructions to control speaker voice, conversation topic, conversational behaviour (e.g., backchanneling and interruptions), and dialogue initiation. We propose a single-stage training protocol and systematically analyze design choices. Both the model and training code will be released to enable reproducible research on controllable full-duplex speech systems.
>
---
