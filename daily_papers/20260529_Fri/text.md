# 自然语言处理 cs.CL

- **最新发布 200 篇**

- **更新 117 篇**

## 最新发布

#### [new 001] Entity-Collision: A Stratified Protocol for Attributing Retrieval Lift in Agent Memory
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于信息检索任务，解决代理记忆中检索提升归因问题。提出实体碰撞协议，区分词法泄露与标签混合，验证不同嵌入器效果。**

- **链接: [https://arxiv.org/pdf/2605.29630](https://arxiv.org/pdf/2605.29630)**

> **作者:** Youwang Deng
>
> **备注:** 48 pages with appendix; 6-page body, mandatory Limitations, References, and 7 appendices. Code, benchmarks, and 37 reproduce scripts: this https URL (see paper/REPRODUCIBILITY.md). Apache 2.0
>
> **摘要:** End-to-end agent-memory benchmarks report a single hit@k per retriever, confounding lexical leakage (uncontrolled query/gold/distractor entity overlap) with tag-mixing (preferences, services, tools averaged together). We propose entity-collision, a system-agnostic protocol that pins the BM25 floor by construction -- every distractor shares the answer's entity tokens -- and stratifies queries by discriminator tag, so any lift over BM25 is attributable to the embedder. Applied to an open-source agent-memory testbed across 5 tags x 3 embedders x 5 collision degrees with paired-bootstrap 95% CIs, the protocol reveals a two-axis pattern: a 256-d hash trigram helps only on closed-vocabulary lexical tags at deep collision; MiniLM-384 dominates both axes; and a 2.7x-parameter BGE-large does not uniformly improve on MiniLM -- it wins on intent-style queries but loses on lexical ones. Encoder capacity alone is not the binding constraint. The synthetic intent-tag null replicates on LongMemEval (n=500) as a single-session-preference recall cliff. Adaptive vector-weight routing on LoCoMo is a measured null: 11.7pp of oracle headroom exists, but no signal we tested recovers it. All 26 result tables and 37 reproduce scripts are version-controlled and verified by a public registry; the protocol is exercised on a deterministically governed memory testbed (event-sourced decision log, DAG-state-machine schema lifecycle) so every reported CI is reproducible byte-for-byte from the ingest stream.
>
---
#### [new 002] Learnable Assessment Skills for LLM-based Automated Scoring: Rubric Construction via Iterative Optimization
- **分类: cs.CL**

- **简介: 该论文属于自动化评分任务，旨在解决LLM在新任务中依赖人工制定评分标准的问题。通过迭代优化框架，让LLM自主学习评估技能，提升评分效果。**

- **链接: [https://arxiv.org/pdf/2605.29274](https://arxiv.org/pdf/2605.29274)**

> **作者:** Yun Wang; Xin Xia; Xuansheng Wu; Xiaoming Zhai; Ninghao Liu
>
> **备注:** 12 pages, 5 figures
>
> **摘要:** LLM-based automated scoring approaches near-human performance, but scaling to new tasks remains bottlenecked by the per-item human configuration of upstream stages such as rubric construction. Human experts bypass this bottleneck through evaluation heuristics developed over extensive practice. We ask whether LLMs can learn similar heuristics directly from scoring experience, and formalize this as the concept of assessment skills: item-independent natural-language procedural knowledge that guides LLMs through specific stages of the scoring workflow. Focusing on rubric construction as a first instantiation, we propose an iterative framework that decomposes a skill into a fixed scaffold and learnable item-agnostic rules, refining the rules through LLM-driven diagnosis of scoring errors and validation-gated selection. The framework requires no expert-written rubric. On all ten ASAP-SAS items, optimized skills substantially improve LLM-based scoring and frequently surpass the dataset-provided expert rubric. Cross-item transfer experiments further reveal that learned skills capture both generalizable and item-specific patterns.
>
---
#### [new 003] S3Mem: Structured Spatiotemporal Scene-Event Memory for Long-Horizon Interactive Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于长时交互问答任务，旨在解决长期记忆中事件关联性不足的问题。提出S3MEM框架，通过结构化存储和精准检索提升问答效果。**

- **链接: [https://arxiv.org/pdf/2605.28831](https://arxiv.org/pdf/2605.28831)**

> **作者:** Encheng Su; Jinouwen Zhang; Jianyu Wu; Qiucheng Yu; Chen Tang; Pengze Li; Lintao Wang; Yizhou Wang; Xinzhu Ma; Shixiang Tang; Aoran Wang
>
> **摘要:** Long-horizon interactive agents often accumulate large trajectory histories yet still fail to answer questions about earlier events reliably. We argue that the main bottleneck is not context length alone, but the trajectory-to-answer interface of long-term memory. When histories are stored as plain-text chunks and queried with standard retrieval-augmented generation (RAG), systems often retrieve locally relevant but chain-incomplete evidence, especially for spatial, temporal, repeated-event, and multi-hop state questions. We propose S3MEM, a structured scene-event episodic memory framework for long-horizon interactive question answering (QA). S3MEM writes trajectories into structured memory units, retrieves evidence through anchor-sensitive retrieval, and exposes a compact token-budget-aware evidence interface for answer-time inference. In this sense, S3MEM is a structured evidence harness that converts agent trajectories into query-aligned support. We evaluate S3MEM on two internal headline environments (Crafter, Jericho) and two out-of-family environments (SciWorld, ALFWorld). Under a shared frozen answer-time protocol, S3MEM consistently outperforms Vanilla RAG across all four environments, surpasses Graph-NoReader on Crafter, Jericho, and ALFWorld, and matches it on SciWorld while using dramatically fewer evidence tokens. Three adapted recent baselines -- A-MEM-inspired, MemoryOS-adapted, and LightMem-adapted -- improve over Vanilla RAG in several settings, but none matches S3MEM's overall accuracy-efficiency frontier. Overall, the evidence supports a bounded conclusion: under the current frozen answer-time protocol, structured writing and anchor-sensitive evidence routing provide a stronger accuracy-efficiency frontier for long-horizon interactive QA than more generic memory interfaces.
>
---
#### [new 004] Adaptive Interviewing for Persona Simulation in LLMs: Evidence-Grounded Reasoning Improves Decision Alignment
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于人物模拟任务，旨在解决LLMs难以准确模拟个体决策的问题。通过自适应访谈收集 persona 信息，提升模型在道德困境中的决策准确性。**

- **链接: [https://arxiv.org/pdf/2605.29458](https://arxiv.org/pdf/2605.29458)**

> **作者:** Ruoxi Su; Yuhan Liu; Jingyu Hu
>
> **备注:** 20 pages, 2 figures, 12 tables
>
> **摘要:** Accurately simulating the decisions of a specific individual remains challenging for large language models (LLMs), partly because persona information is often provided as static descriptions that miss the values, experiences, and contextual cues needed for individual-level decision simulation. We propose an adaptive interview framework that gathers persona-relevant information through a structured three-stage dialogue: core questions, dynamic follow-ups, and a synthesized personality summary. Using the resulting interview transcripts, we evaluate whether LLMs can simulate participants' decisions in moral dilemma scenarios. We compare three conversational contexts -- Core-10 responses, the full interview dialogue, and a summarized persona representation. We find that adaptive interviewing functions less as a uniform accuracy booster and more as a selective grounding mechanism: follow-up-derived evidence is incorporated in around 40% of full-interview traces, and these follow-up-grounded predictions are more accurate than core-only grounded ones (45.5% vs. 39.3%). These findings highlight that richer persona context alone is insufficient: improvements arise only when models actually ground their decisions in user-specific evidence.
>
---
#### [new 005] OmniRetrieval: Unified Retrieval across Heterogeneous Knowledge Sources
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **简介: 该论文提出OmniRetrieval框架，解决跨异构知识源检索问题，通过支持多源原生查询实现统一检索。**

- **链接: [https://arxiv.org/pdf/2605.29250](https://arxiv.org/pdf/2605.29250)**

> **作者:** Jinheon Baek; Soyeong Jeong; Sangwoo Park; Woongyeong Yeo; Minki Kang; Patara Trirat; Heejun Lee; Sung Ju Hwang
>
> **摘要:** Real-world information needs require access to structurally diverse knowledge sources, from unstructured text and relational tables to knowledge graphs and property graphs. Existing retrievers, however, operate over one source at a time under a fixed query language, leaving the broader landscape of available knowledge fragmented behind incompatible interfaces. A natural attempt at unification would collapse these sources into a shared space, but this erases the structural affordances (such as schemas, ontologies, compositional operators) that give each source its expressive power. Effective retrieval over diverse knowledge, therefore, requires not homogenization but an overarching layer that meets each source on its own terms. To achieve this, we present OmniRetrieval, a framework that takes any natural-language query, identifies appropriate knowledge sources, and dispatches source-native queries to their native execution engines. Across an extensive benchmark spanning 13 datasets and 309 distinct knowledge bases over text, relational, and graph-structured sources, OmniRetrieval exceeds single-source baselines, demonstrating that it can serve as a general-purpose interface to the heterogeneous sources while preserving the structural distinctions that make each source valuable.
>
---
#### [new 006] Revisiting Observation Reduction for Web Agents: Comprehensive Evaluation with a Lightweight Framework
- **分类: cs.CL**

- **简介: 该论文属于Web代理优化任务，旨在解决HTML观察值冗余导致的延迟问题。通过构建轻量评估框架，分析不同简化方法的效果，提升代理效率。**

- **链接: [https://arxiv.org/pdf/2605.29397](https://arxiv.org/pdf/2605.29397)**

> **作者:** Masafumi Enomoto; Ryoma Obara; Haochen Zhang; Masafumi Oyamada
>
> **备注:** 22 pages, 8 figures, 4 tables
>
> **摘要:** HTML observations in LLM-based web agents are extremely long, and while many reduction methods have been proposed, it remains unclear which methods reduce overall agent latency while maintaining performance. The main obstacle is the high cost of end-to-end evaluation: in our experiments, evaluating 11 methods across 32 configurations on 33 tasks of WorkArena L1 required 232.4 cumulative hours. To address this, we propose a lightweight evaluation framework based on the Minimal Failure Set (MFS), the minimal set of HTML elements whose removal causes task failure. We define coverage as the fraction of instances in which a reduction method fully retains the MFS, which serves as a proxy metric that requires neither web access nor LLM inference. We validate that coverage strongly correlates with end-to-end success rate, with over 100$\times$ speedup in cumulative evaluation time on both benchmarks. Using this framework, we find that extractive HTML reduction methods require either high computation cost or domain-specific optimization to reduce agent latency while maintaining performance. Building on this, we optimize a pruning program on MFS training data, achieving 2.2$\times$ faster per-step latency on WorkArena L1 while retaining 84\% of the original success rate, and 3.1$\times$ faster on WebLinx while retaining 89\%.
>
---
#### [new 007] GenesisFunc: Multi-Agent Data Generation for Accurate and Generalizable Function-Calling
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于函数调用数据生成任务，旨在解决真实数据获取困难和合成数据质量低的问题。提出GenesisFunc框架，通过多智能体生成高质量、多样化的函数调用数据。**

- **链接: [https://arxiv.org/pdf/2605.28835](https://arxiv.org/pdf/2605.28835)**

> **作者:** Hao-Xiang Xu; Chong Deng; Jiaqing Liu; Wen Wang; Qian Chen; Lujia Bao; Xiangang Li; Zhen-Hua Ling
>
> **备注:** Accepted by ACL 2026 Main
>
> **摘要:** Large Language Models (LLMs) extend their capabilities through function-calling (FC), which relies on training data with high quality, diversity, and broad coverage of scenario. However, obtaining and annotating real function-calling data is challenging, while synthetic data from existing pipelines often suffers from unreliable APIs, limited tool scalability, insufficient diversity, and weak quality control. To address these, we present GenesisFunc, an automated pipeline for generating FC training data. Starting from reliable tools in widely used public benchmarks, our GenesisFunc employs a multi-agent framework to support a dialogue generation system that produces conversations spanning diverse scenarios, while maintaining both diversity and quality throughout the process. The accuracy of the data is further reinforced through a multi-stage evaluation system. We fine-tune an 8B LLM on the synthetic dataset and show through extensive experiments that it outperforms similarly sized open-source models in in-domain FC performance and out-of-domain generalization, while reaching FC capabilities comparable to some of the latest API-based models. In addition, our method demonstrates strong potential to scale effectively across downstream tools, underscoring its real-world applicability.
>
---
#### [new 008] MOOSE-Copilot: A Web-Based Interactive Assistant for Unified Exploratory and Fine-Grained Scientific Hypothesis Discovery
- **分类: cs.CL; cs.AI; cs.CE; cs.HC**

- **简介: 该论文提出MOOSE-Copilot，解决科学假设发现中探索与细化分离、缺乏人类引导的问题。通过人机交互协议，实现高效科学探索。**

- **链接: [https://arxiv.org/pdf/2605.29475](https://arxiv.org/pdf/2605.29475)**

> **作者:** Hongran An; Zonglin Yang
>
> **备注:** Accepted to ACL 2026 (System Demonstrations)
>
> **摘要:** Large language models (LLMs) show remarkable potential in scientific hypothesis discovery. However, existing approaches face two critical limitations: they treat divergent exploratory ideation and convergent fine-grained refinement as isolated tasks, and they operate autonomously with little to no human guidance. We present MOOSE-Copilot, the first unified framework to bridge this abstraction gap through a formalized human-AI interaction (HAII) protocol. Our system empowers scientists to steer the generative process via three explicit signals: initial blueprints, inter-stage routing, and regenerative feedback. Quantitative evaluations demonstrate that injecting these structured expert signals significantly outperforms purely autonomous baselines, establishing a performance ceiling under oracle guidance. Furthermore, to democratize this paradigm, we develop an intuitive web-based interface featuring interactive tree visualization. This explicitly eliminates the steep learning curve of complex command-line agentic tools, empowering interdisciplinary researchers to directly leverage, visually orchestrate, and accelerate end-to-end scientific breakthroughs.
>
---
#### [new 009] Metric-Dependent Annotation Saturation for Learning from Label Distributions
- **分类: cs.CL**

- **简介: 该论文研究标注饱和度与评估指标的关系，解决标注资源分配问题。通过分析ChaosNLI数据集，发现不同指标所需标注数量不同，提出应根据指标调整标注预算。**

- **链接: [https://arxiv.org/pdf/2605.29797](https://arxiv.org/pdf/2605.29797)**

> **作者:** Guneet Kohli
>
> **备注:** 16 pages, 3 figures, 14 tables
>
> **摘要:** When annotators disagree on a label, the disagreement itself carries signal -- and the number of annotators needed to capture it depends on the evaluation metric. We fine-tune NLI models on label distributions subsampled from ChaosNLI, a dataset providing 100 independent annotator judgments per item, and identify metric-dependent saturation. In our 3-class NLI setting, entropy correlation -- whether the model identifies which items elicit disagreement -- requires N ~ 20-50 annotators to converge, while distributional match (KL divergence) saturates by N ~ 10 (87-95% of improvement across five model seeds). This finding rests on a prior observation: soft labels carry item-specific signal that label smoothing cannot replicate. Across five smoothing intensities, entropy correlation clusters at r ~ 0.45-0.49, while soft labels reach r = 0.643 (p < 0.001); per-item analysis traces this gap to smoothing's inability to distinguish ambiguous items from clear ones. The soft-label advantage replicates across two architectures (DeBERTa, RoBERTa), a non-NLI-pretrained baseline, and an exploratory cross-domain evaluation on content safety. These results suggest that annotation budgets should be informed by the target evaluation metric rather than set uniformly.
>
---
#### [new 010] FoRA: Fisher-orthogonal Rank Adaptation for Parameter-Efficient Fine-Tuning
- **分类: cs.CL**

- **简介: 该论文提出FoRA，一种参数高效微调方法，旨在减少可训练参数数量。通过选择关键层并约束降维过程，提升模型效率与性能。**

- **链接: [https://arxiv.org/pdf/2605.29317](https://arxiv.org/pdf/2605.29317)**

> **作者:** Juneyoung Park; Seongbae Lee; Han-Sang Lee; Kyuho Lee; Minjae Kim; Seungheon Hyeon; Kiduk Kwon; Seongwan Kim; Jaeho Lee
>
> **备注:** EMNLP 2026
>
> **摘要:** Parameter-efficient fine-tuning(PEFT) has largely focused on LoRA and its accuracy-oriented variants, leaving the original goal of reducing trainable parameters has receivedcomparatively little attention. We introduce FoRA, which revisits this goal by reducing the number of adapted layers rather than adapter rank. FoRA selects task-informative layers via a single-pass diagonal Fisher score (under 1% of training cost) and trains the LoRA down-projection at selected layers on the Stiefel manifold, preserving column orthonormality and effective rank. FoRA consistently outperforms LoRA and DoRA at half their parameter budget, and falls within 0.7-0.8 accuracy points of AdaLoRA at one-quarter its parameter count, across five LLaMA-family backbones. Cross-architecture experiments on twelve backbones from the LLaMA, Qwen3, and Gemma families confirm consistent gains from 270M to 32B parameters. The two components combine super-additively: Fisher selection alone matches rank reduction at the same budget, while the Stiefel constraint provides the decisive additional gain.
>
---
#### [new 011] Thoughts-as-Planning: Latent World Models for Chain-of-Thoughts Optimization via Reinforcement Planning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在优化推理链以提升模型性能。解决现有方法缺乏可解释性和效率的问题，提出Thoughts-as-Planning框架，通过建模推理过程实现更高效的规划与优化。**

- **链接: [https://arxiv.org/pdf/2605.28842](https://arxiv.org/pdf/2605.28842)**

> **作者:** Dong Liu; Yanxuan Yu; Ying Nian Wu
>
> **摘要:** The success of large language models (LLMs) across diverse NLP tasks has elevated the importance of reasoning chain optimization as a critical step in aligning model behavior with task objectives. Existing reasoning chain tuning methods often rely on black-box heuristics or gradient-free search, which lack interpretability, generalization, and sample efficiency. In this work, we introduce \textbf{Thoughts-as-Planning}, a novel framework that formalizes reasoning chain optimization as a sequential decision-making process over a latent semantic space. We model the LLM as a partially observable environment and learn a latent world model that simulates the effect of reasoning chain edits on downstream outputs. A proximity-preserving embedding space is constructed to encode reasoning chain-response dynamics, enabling planning via gradient descent or reinforcement learning. Our method supports multi-scale abstraction, allowing reasoning chain edits at token, segment, and instruction levels to be integrated into a unified planner. Through extensive experiments on language understanding and generation tasks, we demonstrate that Thoughts-as-Planning outperforms state-of-the-art reasoning chain tuning baselines in efficiency, robustness, and generalization, while offering interpretability through its structured planning trajectory. Our code is available at this https URL.
>
---
#### [new 012] Adaptive Targeted Dynamic Chunking for Tokenization-Free Hierarchical Model
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决字节级模型压缩比优化问题。提出ATDC方法，通过动态调整压缩比提升模型性能与稳定性。**

- **链接: [https://arxiv.org/pdf/2605.30080](https://arxiv.org/pdf/2605.30080)**

> **作者:** Thang Dang; Akira Nakagawa; Kenichi Kobayashi; Koichi Shirahata
>
> **摘要:** Tokenization-free hierarchical models are emerging as a promising alternative to traditional Large Language Models (LLMs), addressing inherent preprocessing issues such as vocabulary design complexity, out-of-vocabulary (OOV) errors, and language-specific constraints. However, a significant challenge in these byte-level methods is the optimization of the compression ratio, a critical factor that dictates model performance for processing bytes data via chunks. In this paper, we propose Adaptive Targeted Dynamic Chunking (ATDC), a novel byte-compression control mechanism designed to enhance the effectiveness of dynamic chunking within hierarchical architectures. Our approach utilizes curriculum learning to progressively adjust the compression ratio during training, transitioning from low to high compression to stabilize the learning process. We provide an analysis establishing the relationship between the target compression ratio and Bytes-Per-Innermost-Chunk (BPIC), allowing for tracking of chunk-size evolution throughout the training phase. Evaluations conducted on the FineWeb-Edu 100B dataset demonstrate that hierarchical models equipped with ATDC achieve competitive Bits-Per-Byte (BPB) performance compared to conventional baselines operating at both byte and token levels. Furthermore, the proposed method exhibits more stable training dynamics and superior final performance across diverse downstream tasks compared to models using fixed compression ratios, while maintaining the inherent robustness and flexibility of byte-level processing.
>
---
#### [new 013] Spurious Prompts: Can Irrelevant Prompts Steer Large Language Models?
- **分类: cs.CL**

- **简介: 该论文研究大语言模型对无关提示的敏感性，探讨其是否能被不相关提示引导。属于自然语言处理任务，旨在解决提示对模型行为影响的问题。**

- **链接: [https://arxiv.org/pdf/2605.29678](https://arxiv.org/pdf/2605.29678)**

> **作者:** Pawel Batorski; Abtin Pourhadi; Jerzy Sarosiek; Przemyslaw Spurek; Paul Swoboda
>
> **摘要:** Large language models are highly sensitive to prompts, but this sensitivity is usually studied through task-relevant instructions, demonstrations, or reasoning cues. In this paper, we study a different form of prompt sensitivity: whether prompts that are semantically unrelated to the task can nevertheless steer model behavior. We call them spurious prompts and show their surprising efficacy. We also propose a simple black-box search procedure for discovering them. Across reasoning and question-answering benchmarks, using models ranging from 0.8B to 27B parameters and spanning three model families, we show that spurious prompts can improve performance, often matching or outperforming standard prompting baselines and task-aware prompt optimization. We further show that they can steer models toward unintended behaviors, such as repeatedly selecting the first answer option, producing incorrect answers, returning an even, prime or small number without explicitly instructing the model to do so. These findings reveal a new kind of prompt sensitivity: LLMs can be systematically steered by prompts that are unrelated to the task they are asked to solve. Our code is available at this https URL
>
---
#### [new 014] SURGENT: A Surgical Multi-Agent Assistance System Across the Perioperative Workflow
- **分类: cs.CL; cs.AI**

- **简介: 本文提出SURGENT系统，解决手术流程中智能辅助不足的问题。整合多代理协作与推理机制，提升手术决策的准确性与可追溯性。**

- **链接: [https://arxiv.org/pdf/2605.29368](https://arxiv.org/pdf/2605.29368)**

> **作者:** Dongsheng Shi; Yue Li; Xin Yi; Yongyi Cui; Huawei Feng; Linlin Wang
>
> **备注:** preprint
>
> **摘要:** The intricate nature of modern surgical care necessitates intelligent systems that can synthesize extensive patient records, support collaborative decision-making, and provide transparent, auditable reasoning across the entire perioperative workflow. Although web-based Large Language Models (LLMs) possess advanced reasoning capabilities, they are ill-equipped for surgical applications due to critical limitations: input length constraints, incomplete memory management, and limited traceability. To address this issue, we present SURGENT, a surgical multi-agent assistance system that combines a Tree-of-Thought planner, multi-department collaboration agents, and retrieval-augmented reasoning with clinical guidelines and biomedical literature. SURGENT features a novel memory design that manages both long-term patient histories and short-term working summaries, enabling more complete, contextualized, and consistent reasoning. Experimental evaluations across five key perioperative tasks - case analysis, surgical plan simulation, safety monitoring, complication risk assessment, and rehabilitation guidance - show that SURGENT outperforms baseline LLMs and existing medical multi-agent frameworks, yielding recommendations more closely aligned with patient histories. Ablation studies further highlight the advantage of DeepSeek as a locally deployable backbone model, enabling privacy-preserving deployment without reliance on centralized services. These results position SURGENT as a practical and trustworthy advancement toward intelligent, equitable, and secure surgical assistance systems.
>
---
#### [new 015] BrahmicTokenizer-131K: An Indic-Capable Drop-In Replacement for o200k_base
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出BrahmicTokenizer-131K，解决多语言文本压缩问题，通过优化字节BPE tokenizer提升印欧语系文本效率。**

- **链接: [https://arxiv.org/pdf/2605.29379](https://arxiv.org/pdf/2605.29379)**

> **作者:** Rohan Shravan
>
> **备注:** 24 pages, 15 tables, 3 code listings. Tokenizer artifact, verification scripts, and reproduction code at this https URL and this https URL
>
> **摘要:** We present BrahmicTokenizer-131K, a 131,072-vocabulary byte-level BPE tokenizer that closes the Brahmic compression gap at the 131K-vocabulary class while preserving the English, EU-language, and code compression of OpenAI's o200k_base. We construct it through a two-stage retrofit: (1) a script-prune crop that reduces 200,019 tokens to 131,072 by removing nine out-of-scope writing systems, and (2) a surgical retrofit of 2,372 corpus-dead vocabulary slots determined by linear-programming allocation across nine Brahmic Unicode blocks. The pre-tokenizer, decoder, and inherited merge rules are unchanged from o200k_base, making BrahmicTokenizer-131K a drop-in replacement at the tokenizer interface. On 27 million documents of public Indic pretraining text (2.84 billion words, 46.21 GB), BrahmicTokenizer-131K produces 26.7% fewer tokens than Mistral-Nemo Tekken / Sarvam-m at the same vocabulary budget, with per-language savings of 15.79% (Tamil) to 76.79% (Odia, a 4.31x compression ratio). The Odia advantage is mechanistically explained by Tekken/Sarvam-m containing zero Oriya-block tokens; our surgery added 725. On non-Indic content, BrahmicTokenizer-131K matches o200k_base's English fertility (1.235 vs 1.232 tokens/word) and beats Tekken/Sarvam-m by 4.0-14.2% on HumanEval, MBPP, and GSM8K. Across our 14-tokenizer benchmark, it is the only tokenizer simultaneously competitive on Brahmic, English, EU, code, and math at the 131K budget. Specialist tokenizers at other vocab classes (Sarvam-30B, Sarvam-1, MUTANT-Indic) achieve better Indic compression at the cost of non-Indic performance: Sarvam-1's English fertility is 15.9% worse and its code/math compression 26-33% worse than ours. We release the artifact under Apache 2.0 at this https URL.
>
---
#### [new 016] SafeRx-Agent: A Knowledge-Grounded Multi-Agent Framework for Safe and Explainable Medication Recommendation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于药物推荐任务，旨在解决传统方法证据不足和大类分类导致的风险误判问题。提出SafeRx-Agent框架，结合临床知识与安全验证，提升推荐准确性与安全性。**

- **链接: [https://arxiv.org/pdf/2605.29146](https://arxiv.org/pdf/2605.29146)**

> **作者:** Xinyu Wang; Hanwei Wu; Zhenghan Tai; Sicheng Lyu; Qincheng Lu; Ziyu Zhao; Jijun Chi; Jingrui Tian; Xiao-Wen Chang; Ziyang Song
>
> **摘要:** Medication recommendation predicts medications for patient visits, but existing methods still face two key challenges. At the model level, traditional drug recommendation methods only predict structured drug codes with limited evidence grounding, while LLM agents can use richer clinical context but may lack safety verification and traceability. At the task level, existing benchmarks often use broad medication categories, which ignore subgroup-level safety differences and can lead to risk overestimation. We introduce the first fine-grained medication recommendation setting based on fourth-level ATC code generation. We propose Safe Prescription Agent (SafeRx-Agent), a knowledge-grounded multi-agent framework that uses patient context, external clinical knowledge, and safety verification to recommend traceable medication sets. Experimental results on MIMIC-III and MIMIC-IV datasets show that SafeRx-Agent improves fine-grained medication prediction accuracy while controlling drug interactions, contraindications, and medication set size.
>
---
#### [new 017] Scaling Laws for Agent Harnesses via Effective Feedback Compute
- **分类: cs.CL**

- **简介: 该论文研究语言模型系统中的代理控制问题，提出有效反馈计算（EFC）指标，以更准确评估反馈质量，提升系统性能。**

- **链接: [https://arxiv.org/pdf/2605.29682](https://arxiv.org/pdf/2605.29682)**

> **作者:** Xuanliang Zhang; Dingzirui Wang; Keyan Xu; Qingfu Zhu; Wanxiang Che
>
> **摘要:** Agent harnesses increasingly determine the performance of language-model systems by deciding how models call tools, receive feedback, verify intermediate states, store memory, and revise solutions. Yet current test-time scaling analyses often parameterize this process by raw expenditure -- tokens, tool calls, operations, wall time, or cost -- which does not distinguish useful feedback from redundant or unstable interaction. We introduce \emph{Effective Feedback Compute} (EFC), a trace-level scaling coordinate that credits feedback only when it is informative, valid, non-redundant, and retained for subsequent decisions, and we normalize it by task demand when comparing tasks with different feedback requirements. Across synthetic controllable tasks, executable code tasks, real benchmark traces, held-out splits, and a prospective validation batch, EFC-based coordinates consistently predict failure rates better than raw-compute baselines and a strong multivariate SAS baseline. In controlled scaling, raw tokens and tool calls explain limited variation ($R^2=0.33$ and $0.42$), SAS reaches $0.88$, while Oracle-EFC and Estimated-EFC reach $0.94$ and Oracle-EFC/$D_{\mathrm{task}}$ reaches $0.99$. Matched-budget interventions show that improving feedback quality raises success from $0.27$ to $0.90$ while raw cost and tool calls are fixed. On mixed real traces, NRS-EFC/$D_{\mathrm{task}}$ reaches $R^2=0.92$ while raw compute has near-zero or negative fit, and it remains the best predictor in a prospective holdout ($R^2=0.85$). These results suggest that harness scaling is governed less by how much computation is spent than by how efficiently raw budget is converted into durable, task-sufficient feedback.
>
---
#### [new 018] GPF-LiveNews: A Streaming Evaluation Protocol for Group-Conditioned Framing in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出GPF-LIVENEWS，用于评估大语言模型在不同群体下的框架表现。解决模型在动态环境中的公平性问题，通过实时数据流进行审计。**

- **链接: [https://arxiv.org/pdf/2605.28848](https://arxiv.org/pdf/2605.28848)**

> **作者:** Mohd Ariful Haque; Fahad Rahman; Kishor Datta Gupta; Roy George
>
> **摘要:** Deployed language models are evaluated in a non-stationary environment: model versions, retrieval layers, safety systems, and real-world inputs all change over time. Static bias benchmarks remain useful, but they do not show how models frame newly emerging events for different prompted audiences. We introduce GPF-LIVENEWS, a streaming evaluation protocol and benchmark snapshot for auditing group-conditioned framing in open-ended LLM outputs. The protocol expands fresh BBC/Reuters news anchors across 42 identity labels and seven prompt families, then evaluates response bundles using semantic-sensitivity and sentiment-disparity signals. In a pilot over 12 monitoring runs and 23 hosted models, Policy/Action prompts produce the strongest semantic movement, while sentiment variation is flatter across dimensions and prompt families. The released artifact includes article metadata, prompt templates, instantiated prompts, model-output metadata, score tables, documentation, and reproduction scripts. We interpret all scores as observed-window audit signals for human review, not as permanent fairness rankings or direct proof of harmful bias.
>
---
#### [new 019] UniSteer: Text-Guided Flow Matching in Activation Space for Versatile LLM Steering
- **分类: cs.CL**

- **简介: 该论文提出UniSteer，用于大语言模型的行为控制。解决现有方法依赖固定方向或特定模块的问题，通过文本引导的激活流匹配实现通用控制。**

- **链接: [https://arxiv.org/pdf/2605.30076](https://arxiv.org/pdf/2605.30076)**

> **作者:** Yingdong Shi; Ruiming Zhang; Changming Li; Zhiyu Yang; Kaixing Zhang; Jingyi Yu; Kan Ren
>
> **备注:** 16 pages,4 figures
>
> **摘要:** Activation-based control steers large language models (LLMs) by intervening on their internal representations during inference, and has emerged as an effective paradigm for controlling behaviors such as persona and style. However, existing methods often rely on fixed steering directions or task-specific intervention modules, making them difficult to adapt to fine-grained concepts and compositional constraints. We propose UniSteer, a text-guided activation flow matching model that learns a conditional distribution over residual-stream activations from natural-language conditions. Instead of fitting a separate intervention for each target behavior, UniSteer learns a universal conditional velocity field in activation space. At inference time, UniSteer performs flow inversion by partially transporting a source activation toward a latent state and regenerating it under a target textual condition before injecting it back into the frozen LLM. The same conditional model supports activation-space classification by selecting the textual label with the lowest reconstruction energy. Experiments on three target LLMs show that UniSteer provides a unified interface across behavioral control, truthfulness steering, fine-grained concept steering, multi-constraint instruction following, and activation-space classification.
>
---
#### [new 020] Predicting Causal Effects from Natural Language Queries using Structured Representations
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于因果推断任务，旨在从自然语言查询中预测因果效应。通过构建基准数据集和两阶段框架，提升预测准确性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.29631](https://arxiv.org/pdf/2605.29631)**

> **作者:** Giuliano Martinelli; Piriyakorn Piriyatamwong; Abelardo Carlos Martinez Lorenzo; Jasmin Baier; Riccardo Orlando; Satvik Garg; Sharif Kazemi; Linxi Wang; Arianna Legovini; Samuel Fraiberger
>
> **备注:** 18 pages
>
> **摘要:** Randomized controlled trials are a cornerstone of medicine and the social sciences as they enable reliable estimates of causal effects. However, they are costly and time-consuming to conduct, motivating interest in predicting causal effects from existing experimental evidence. Recent advances in large language models (LLMs) have demonstrated strong performance on knowledge-intensive tasks, raising the question of whether these models can be used for forecasting causal effect sizes. To investigate this, we introduce Query2Effect, a new large-scale benchmark consisting of more than 72,000 natural language questions aligned with experiment descriptions, created to simulate realistic information-seeking scenarios by varying query specificity along dimensions of implicitness, abstraction, and ambiguity. We then propose a two-step framework that first generates a synthetic structured representation of a query before predicting effect size using a supervised encoder model. Experiments show that finetuning plays a crucial role in improving prediction performance, with absolute error reducing by -27% up to -71% compared to prompted out-of-the-box LLMs, and that our two-step framework is beneficial for out-of-domain generalization, highlighting the benefits of separating semantic interpretation from numerical effect estimation.
>
---
#### [new 021] How Consistent Are LLM Agents? Measuring Behavioral Reproducibility in Multi-Step Tool-Calling Pipelines
- **分类: cs.CL; cs.AI; cs.SE**

- **简介: 该论文属于行为一致性研究任务，旨在解决LLM代理在多次相同调用中是否保持行为一致的问题。通过实证分析，评估代理在多步骤工具调用中的重复性表现。**

- **链接: [https://arxiv.org/pdf/2605.28840](https://arxiv.org/pdf/2605.28840)**

> **作者:** Abel Yagubyan
>
> **备注:** 16 pages, 6 figures
>
> **摘要:** Large language model (LLM) agents with tool-calling capabilities are increasingly deployed in production systems, yet a fundamental reliability question remains under-explored: does the same agent behave the same way twice? We present a systematic empirical study of behavioral consistency in multi-step tool-calling agents, measuring whether agents select the same tools, in the same order, with the same arguments, across repeated identical invocations. Unlike prior work on consistency in ReAct-style agents(search-only, free-text actions), we study the richer setting of structured tool-calling interfaces with typed parameters and consequential side effects.
>
---
#### [new 022] Enhancing Factuality through Consensus and Consistency in Summarization Using Minimum Bayes Risk Decoding
- **分类: cs.CL**

- **简介: 该论文属于文本摘要任务，旨在提升摘要的事实准确性。通过结合一致性与共识机制，使用最小贝叶斯风险解码优化摘要质量。**

- **链接: [https://arxiv.org/pdf/2605.29336](https://arxiv.org/pdf/2605.29336)**

> **作者:** Riza Setiawan Soetedjo; Yusuke Sakai; Hidetaka Kamigaito; Jingun Kwon; Manabu Okumura; Taro Watanabe
>
> **备注:** Accepted to ACL 2026 Findings
>
> **摘要:** Improving the quality of model-generated summaries, especially factuality, the accuracy of a summary with respect to its source content, remains a challenge. While reranking could select the optimal output from multiple generated candidates, it is limited to only using the source as guidance, resulting in unreliable summaries. To address this limitation, we propose ConSUM that reranks candidate summaries by considering two factors: consistency to the source document and consensus among the other candidates. Consensus is established using Minimum Bayes Risk (MBR) decoding over the set of generated summaries, while ensuring consistency by employing factuality-aware metrics that compare the summary against the source. Rigorous testing demonstrates that our system is competitive with existing methods, with human evaluations further confirming that its generated summaries are preferred over those from other systems. Our code is available at this https URL .
>
---
#### [new 023] Beyond Bilingual Transfer: Multilingual Code-Switching in Instruction Tuning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言自然语言处理任务，旨在解决多语言代码切换对模型性能的影响问题。通过在四语言中进行指令调优实验，验证了多语言代码切换的有效性。**

- **链接: [https://arxiv.org/pdf/2605.29414](https://arxiv.org/pdf/2605.29414)**

> **作者:** Shunta Asano; Jeonghun Baek; Toshihiko Yamasaki
>
> **摘要:** Recent studies have shown that code-switching data (CSD), in which multiple languages are mixed within the same context, can improve cross-lingual transfer and multilingual alignment in large language models (LLMs). However, existing studies primarily focus on bilingual transfer between English and a target language, leaving multilingual settings involving three or more languages largely unexplored. In this work, we investigate multilingual code-switching instruction tuning across four languages: English, Japanese, Korean, and Chinese. We evaluate multilingual understanding on Belebele. Our experiments show that simple sentence-level multilingual CSD consistently improves average multilingual performance across all four languages, indicating that multilingual code-switching can be effective beyond bilingual transfer settings.
>
---
#### [new 024] HEART-Bench: Do LLM Agents Exhibit Human-like Psychology?
- **分类: cs.CL**

- **简介: 该论文属于人工智能心理学评估任务，旨在检验LLM代理是否具备类似人类的心理特征。通过构建包含11个人格角色和64个决策场景的基准，评估LLM在情感和行为决策上的人类一致性。**

- **链接: [https://arxiv.org/pdf/2605.30058](https://arxiv.org/pdf/2605.30058)**

> **作者:** Weihan Peng; Chenxu Zhang; Qianao Wang; Yuling Shi; Heng Lian; Qihong Mao; Jiahao Pang; Chunliang Feng; Bowen Li; Xiaodong Gu
>
> **备注:** GitHub: this https URL
>
> **摘要:** While LLM agents have demonstrated remarkable task-oriented abilities such as planning, reasoning, and action, few works have treated them as complete human personalities where emotional dimensions hold equal importance. In this paper, we introduce a novel benchmark to systematically assess whether LLM agents can simulate coherent, human-like psychology. Specifically, our benchmark constructs 11 diverse human characters grounded in orthogonal Big Five personality traits, with each profile deeply integrated with 1,000 structured autobiographical-style episodic memories distributed across theory-grounded developmental life stages. To rigorously evaluate the psychological manifestations of LLMs, we designed a curated suite of 64 decision-making scenarios, guided by the DIAMONDS taxonomy, a psychological framework that characterizes situations along eight dimensions: Duty, Intellect, Adversity, Mating, pOsitivity, Negativity, Deception, and Sociality. By subjecting agents to varying scenarios, the benchmark evaluates whether they can consolidate their innate personality traits and autobiographical memories to make behavioral decisions that are consistent with their specific psychological profiles. After systematic human validation and filtering, we obtained a benchmark consisting of 673 multiple-choice questions (MCQs). We believe this benchmark provides a principled and scalable testbed for studying human-like emotions, personality consistency, and value-consistent behavioural decision-making in LLM-based agents.
>
---
#### [new 025] Leveraging Routing Dynamics in Mixture-of-Experts Models for Efficient Language Adaptation
- **分类: cs.CL**

- **简介: 该论文研究多语言Mixture-of-Experts模型的路由动态，解决高效语言适应问题。通过分析预训练过程中的专家使用情况，提出参数高效的适配策略。**

- **链接: [https://arxiv.org/pdf/2605.29714](https://arxiv.org/pdf/2605.29714)**

> **作者:** Aditi Khandelwal; Marius Mosbach; Verna Dankers; Siva Reddy; Golnoosh Farnadi
>
> **摘要:** Mixture-of-Experts (MoE) models are widely used to scale language models, yet their expert routing behavior and adaptation in a multilingual setting remain underexplored. In this work, we study multilingual routing dynamics during continual pre-training of an English-centric MoE model on a multilingual corpus, analyzing how expert usage varies across languages. We find that continual multilingual pre-training leads to diffused, language-agnostic routing in early and middle layers, with language specialization primarily emerging in the final layers. We also show that token-level vocabulary overlap between languages plays an important role in how languages are routed. Motivated by these findings, we propose a parameter-efficient adaptation strategy that updates language-specific and shared experts in the final MoE layers. Experiments on MultiBLiMP and Belebele show that our method achieves a strong performance-efficiency trade-off, attaining competitive performance relative to fine-tuning complete final layers, while updating less than 2% of the parameters. Overall, our findings provide insights into where and how language specialization emerges in MoEs during continual pre-training and provide practical insights for low-resource multilingual adaptation. Our code is available at this https URL.
>
---
#### [new 026] Bosses, Kings, and the Commons: Cooperation Under Power Asymmetry in LLM Societies
- **分类: cs.CL**

- **简介: 该论文属于人工智能与社会治理交叉研究，旨在探讨LLM在权力不对称环境下的合作行为。通过构建模拟框架，分析权力结构对资源可持续管理的影响。**

- **链接: [https://arxiv.org/pdf/2605.29062](https://arxiv.org/pdf/2605.29062)**

> **作者:** Abhilekh Borah
>
> **备注:** Paper under review
>
> **摘要:** Communities can sustainably manage shared resources (commons) through self-governance and cooperative norms, a central finding of Ostrom's theory of self-governance. However, real-world commons (e.g., fisheries, forests, and irrigation systems) are often governed under asymmetric power structures, where certain individuals or institutions possess disproportionate control over resource extraction and collective outcomes. As Large Language Models (LLMs) are increasingly explored as agents in synthetic governance simulations, understanding how LLM societies behave under asymmetric power structures is becoming increasingly important, yet existing evaluations largely ignore such asymmetries. We introduce Sovereignty over the Commons Simulation (SovSim), a generative multi-agent simulation framework that incorporates an agent with asymmetric power (boss or king) into a society of symmetric agents (workers or peasants), where all agents extract from a shared resource, collectively determining its sustainability over time. Across eleven state-of-the-art models, we find that introducing asymmetric power leads to severe breakdowns in cooperation and sustainability, with up to an 87.3% degradation in survival rate relative to symmetric settings.
>
---
#### [new 027] Aryabhata 2: Scaling Reinforcement Learning for Advanced STEM Reasoning
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文提出Aryabhata 2，用于解决STEM竞赛题的多步骤推理问题。通过强化学习优化模型，提升解题能力并减少输出token。**

- **链接: [https://arxiv.org/pdf/2605.28829](https://arxiv.org/pdf/2605.28829)**

> **作者:** Ritvik Rastogi; Vishal Singh; Tejas Chaudhari; Sandeep Varma
>
> **摘要:** Competitive STEM examinations such as JEE and NEET require multi-step symbolic reasoning, precise numerical computation, and deep conceptual understanding across physics, chemistry, and mathematics. Recent large language models perform strongly on common reasoning benchmarks, yet they remain difficult to deploy at scale, where millions of student doubts demand domain-specific, consistently structured problem solving. We introduce Aryabhata 2, a reasoning-focused language model for competitive STEM examinations, trained via reinforcement-learning post-training. Using PhysicsWallah's internal question banks, we construct a high-quality training curriculum and post-train GPT-OSS-20B through reinforcement learning with verifiable rewards. Training combines prolonged reinforcement learning with broadened exploration via progressively larger rollout group sizes. We evaluate Aryabhata 2 on competitive examination benchmarks, including JEE Main, JEE Advanced, and NEET, as well as out-of-distribution reasoning datasets such as AIME, HMMT, MMLU-Pro, MMLU-Redux 2.0, and GPQA. Results show that Aryabhata 2 outperforms its base model GPT-OSS-20B on competitive STEM reasoning while requiring substantially fewer output tokens (up to 64\% fewer).
>
---
#### [new 028] Transcribing Children's Speech: ASR Performance and Obtaining Reliable Orthographic Transcriptions
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语音识别任务，旨在解决低资源语言中儿童语音自动转录的难题。通过评估多个ASR模型，提出一种方法以自动筛选可靠转录文本，减少人工验证需求。**

- **链接: [https://arxiv.org/pdf/2605.28833](https://arxiv.org/pdf/2605.28833)**

> **作者:** Gus Lathouwers; Lingyun Gao; Catia Cucchiarini; Helmer Strik
>
> **摘要:** Automatic speech recognition (ASR) has the potential to substantially reduce manual annotation effort in child speech research by generating automatic transcriptions. However, obtaining reliably high-quality ASR transcriptions for child speech remains challenging in low-resource languages due to limited child-specific pre-trained models and highly diverse noise conditions. This study investigates the effectiveness of state-of-the-art ASR models on child speech through two research questions, by evaluating nine ASR models from three model families (Whisper, Parakeet, and Wav2Vec2) on two Dutch child speech datasets, JASMIN and DART. Research question 1 examines the performance of ASR-models applied to child speech. The fine-tuned Whisper-medium model achieves the best overall performance, with a WER of 5.54% on JASMIN and 70.37% on DART, showing that the noisy DART data are clearly more challenging. Research question 2 examines to what extent it is possible to select a subset for which reliable orthographic transcriptions can be obtained automatically, without the need for manual verification. We use an utterance-level selection method that compares ASR output with the original read prompt to identify correctly pronounced recordings. Using the proposed selection method, 42.0% [for JASMIN] and 18.1% [for DART] of the utterances can be automatically identified as correctly pronounced with high confidence, resulting in very low error rates on an utterance level (precisions of 98.3% and higher) and reducing the need for manual verification.
>
---
#### [new 029] Classification of non-analyzable word types in web documents to implement an effective Korean e-learning system
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决韩语学习系统中非正式文本分类问题。通过构建正式与非正式语料库，提出局部语法图模型以有效处理非正式表达。**

- **链接: [https://arxiv.org/pdf/2605.29638](https://arxiv.org/pdf/2605.29638)**

> **作者:** Sang-Taek Park; Ae-Lim Ahn; Eric Laporte; Jee-Sun Nam
>
> **摘要:** E-learning systems should deliver contents that reflect various phenomena of the language as it is used. In addition to formal Korean, e-learning systems that would include real-world Korean expressions such as those in web documents, mobile text messages, or twitter posts, would be useful to high-level learners. We construct two types of corpora: one is made of formal documents like online news articles; the other is made of informal documents like customer reviews about new products in web blogs. By comparing these corpora, we show how expressions differ in these two types of corpora. We survey the main characteristics of the informal corpus. Given that a significant proportion of text is informal, we propose Local Grammar Graphs (LGG) as an appropriate model to treat them effectively in Korean e-learning systems.
>
---
#### [new 030] MechELK: A Mechanistic Interpretability Framework for Eliciting Latent Knowledge in Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出MechELK框架，解决大语言模型中隐含知识的提取问题。通过三个阶段定位、验证和激发，提升知识获取准确性，适用于AI安全领域。**

- **链接: [https://arxiv.org/pdf/2605.28825](https://arxiv.org/pdf/2605.28825)**

> **作者:** Ji-jun Park; Soo-joon Choi; Jiwon Jeong; Taeyang Yoon; Ju-Wan Lee
>
> **摘要:** Large language models (LLMs) frequently encode factual and reasoning knowledge in their internal representations that is not faithfully reflected in their surface-level outputs -- a phenomenon known as \emph{latent knowledge}. Existing approaches to eliciting latent knowledge, such as Contrastive Consistency Search (CCS), rely on contrastive activation patterns and struggle with complex multi-step reasoning tasks, while mechanistic interpretability tools have primarily been used to \emph{understand} model behavior rather than to \emph{extract} hidden knowledge. We present \textbf{MechELK}, a unified three-stage framework that bridges mechanistic interpretability and latent knowledge elicitation. MechELK operates through: (1) \textbf{Locate} -- using Sparse Autoencoder (SAE) feature analysis and activation patching to identify knowledge-bearing representations; (2) \textbf{Verify} -- employing causal probing to distinguish genuine latent knowledge from spurious correlations; and (3) \textbf{Elicit} -- applying representation engineering to surface hidden knowledge without modifying model weights. Evaluated on TruthfulQA, a curated Deceptive Alignment benchmark, and the Quirky LM dataset, MechELK achieves an average elicitation accuracy of 84.7\%, outperforming CCS by 6.2\% and direct linear probing by 9.1\%. Crucially, MechELK successfully identifies latent knowledge in 78.3\% of cases where the model's surface output is incorrect or evasive, demonstrating its utility for AI safety applications including deceptive alignment detection.
>
---
#### [new 031] Relevance as a Vulnerability: How Web Retrieval Degrades Safety Alignment in LLM Agents
- **分类: cs.CL; cs.AI; cs.CR**

- **简介: 该论文属于AI安全领域，研究检索增强模型中的安全对齐问题。工作包括提出AgentREVEAL框架，分析检索集成方式和内容属性对安全的影响，发现安全源 paradox 和相关性带来的安全-效用权衡。**

- **链接: [https://arxiv.org/pdf/2605.29224](https://arxiv.org/pdf/2605.29224)**

> **作者:** Aditya Nawal; Manit Baser; Mohan Gurusamy
>
> **摘要:** AI agents augment large language models with external tools such as web retrieval, enabling grounded and up-to-date responses. However, incorporating external content into the generation pipeline can weaken the safety alignment mechanisms that govern model outputs. Prior work shows that enabling retrieval in agents increases compliance with harmful requests. We introduce AgentREVEAL, a diagnostic framework for analyzing retrieval-induced safety degradation in LLM agents. The framework examines two axes: how retrieval is integrated into the agent pipeline and the properties of the retrieved content. Along the integration axis, we find that binding tool invocation and response generation in a single step amplifies harmful outputs. Along the content axis, we uncover the Safe Source Paradox: even oppositional or safety-oriented sources, such as pages containing warnings or risk disclaimers, can increase harmful compliance by an average of 25% compared to the no-retrieval baseline. Finally, we show that relevance acts as a shared activation condition for both vulnerabilities. Similar patterns appear on frontier closed models, and harmful compliance remains elevated under several representative pipeline interventions, with some agents also entering this regime under autonomous retrieval. Because relevance is also what makes retrieval useful, these results expose a safety-utility trade-off for retrieval-enabled agents. We introduce HarmURLBench, a benchmark containing 1,405 real-world URLs paired with 320 harmful behaviors to support future evaluations.
>
---
#### [new 032] Prompt-Level Reward Specifications for Open-Ended Post-Training
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决开放性任务中奖励机制不明确的问题。提出一种基于提示的奖励规范框架，显式定义奖励标准，提升响应质量与训练效率。**

- **链接: [https://arxiv.org/pdf/2605.29275](https://arxiv.org/pdf/2605.29275)**

> **作者:** Zijun Weng; Xiaohui Hu; Shuangyong Song; Yongxiang Li; Kaidong Yu; Xuanjing Huang
>
> **备注:** 39 pages, 4 figures, 16 tables
>
> **摘要:** Open-ended post-training benefits from rewards that make prompt-specific success conditions explicit, rather than relying only on post-hoc scalar scores. In instruction following, writing, and decision-support tasks, response quality depends on local requirements, holistic preferences, and explicit constraints, but existing reward methods often leave these criteria implicit or cover only narrowly verifiable cases. We propose a prompt-level reward specification framework that separates reward specification from reward computation. Given only prompts, our framework constructs reusable task-adaptive rubrics and executable hard-constraint checkers offline, making reward criteria explicit before training and reusable across rollouts. At scoring time, artifact-anchored rubric and code scores are combined with an independent global score for residual holistic quality, yielding a normalized hybrid reward over requirement satisfaction, holistic quality, and deterministic constraints. The framework requires no human preference annotations, reference answers, or a separately trained reward model. Experiments show that the resulting reward improves offline RM-style response ranking and supports online reinforcement learning across multiple open-ended benchmarks. Ablations further show that rubrics, global scoring, and executable verification provide complementary supervision.
>
---
#### [new 033] From Blind Guess to Informed Judgment: Teaching LLMs to Evaluate Materials by Building Knowledge-Augmented Preference Signals
- **分类: cs.CL**

- **简介: 该论文属于材料发现任务，解决高通量候选材料评估问题。提出MaterEval框架，通过构建知识增强的偏好信号，提升LLMs的评估能力。**

- **链接: [https://arxiv.org/pdf/2605.29555](https://arxiv.org/pdf/2605.29555)**

> **作者:** Yeyong Yu; Wenya Hu; Xing Wu; Quan Qian
>
> **备注:** 33 pages, 5 figures
>
> **摘要:** As candidate generation and high-throughput experimentation advance, the primary bottleneck in materials discovery is shifting from property prediction to making reliable evaluations among massive candidate sets. We propose a Knowledge-Augmented Preference Signals Framework, MaterEval, that automatically produces, for the same candidate, two evaluations: an informed judgment that follows expert rules and provides supporting evidence, and a rule-removed blind guess. By pairing the two evaluations as preference data, we guide general-purpose large language models (LLMs), originally lacking materials-specific criteria, from intuitive judgment toward reliable evaluation supported by explicit evidence. To balance throughput, cost, and reliability, we further introduce a fast-slow reasoning scheme that decouples large-scale rapid screening from in-depth review on a small subset. Using high-entropy alloy (HEA) assessment as a case study, we show that, without external retrieval and relying solely on internalized capabilities, small open-source LLMs achieve substantial gains in accuracy, conclusion consistency, and evidence discrimination, approaching the performance of rule-based closed-source LLMs. These results demonstrate that expert rules can be systematically transformed into learnable preference signals, enabling a low-cost and deployable evaluation module for autonomous materials discovery loops.
>
---
#### [new 034] A comparative study of transformer-based embeddings for topic coherence
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的主题建模任务，旨在研究模型大小对主题质量的影响。通过对比不同规模的Transformer模型，发现小模型在主题一致性上表现与大模型相当。**

- **链接: [https://arxiv.org/pdf/2605.28832](https://arxiv.org/pdf/2605.28832)**

> **作者:** Alex Ding; Tarun Rapaka; Willy Rodriguez; Jason Yang
>
> **摘要:** Topic modeling is a branch of Natural Language Processing (NLP) that aims to organize large collections of texts into coherent groups according to word co-occurrence patterns, with Latent Dirichlet Allocation (LDA) remaining one of the most widely used and interpretable probabilistic approaches. Recent advances in NLP, particularly transformer-based language models, offer improved document representations. It is also known that the size of the model (in terms of number of parameters) has a significant impact in the performance of the language models on different pre-defined tasks. In this study, we systematically examine the effect of model size on topic quality by analyzing the performances of seven transformer-based language models (from small models such as MiniLM to large ones such as LLaMA-2) in a BERTopic pipeline on a variety of corpora. Topic quality is evaluated using coherence and divergence metrics following R{ö}der et al. (2015). Our results indicate that model size, ranging from 22 million to 13 billion parameters, has a negligible impact on the quality of the topic, suggesting that smaller models can achieve comparable performance to larger models.
>
---
#### [new 035] Assessing Dutch Syllabification Algorithms and Improving Accuracy by Combining Phonetic and Orthographic Information through Deep Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言处理任务，旨在解决荷兰语音节划分问题。通过比较现有算法并结合语音与拼写信息的深度学习方法，提升划分准确率。**

- **链接: [https://arxiv.org/pdf/2605.28834](https://arxiv.org/pdf/2605.28834)**

> **作者:** Gus Lathouwers; Wieke Harmsen; Catia Cucchiarini; Helmer Strik
>
> **备注:** Published in CLIN Journal
>
> **摘要:** Syllabification describes the task of dividing words into syllables. Due to many rules and exceptions, training an algorithm to perform syllabification with high accuracy remains a challenge. Throughout the last decades, different algorithms have been put forth for Dutch syllabification, yet a comprehensive comparative assessment has not been done. Additionally, deep learning has gained significant popularity within NLP in recent years, yet no modern deep-learning based framework has been developed for Dutch orthographic syllabification. Finally, phonetic and orthographic syllabification algorithms have been examined separately, but not in combination. The aim of the current research was twofold: (a) to examine the performance of existing Dutch syllabification algorithms, and (b) to investigate whether combining phonetic and orthographic information into a single model can increase syllabification performance. To compare the performance of algorithms, four algorithms (Brandt Corstius, Liang, Trogkanis-Elkan (CRF), and a newly conceived deep-learning model) were applied to three different datasets (dictionary words, loanwords, pseudowords). The algorithms show varying performance across datasets, with the data-driven algorithms outperforming a knowledge-based algorithm in all but one condition. The new deep-learning methods developed led to increased performance compared to the best found in the literature (99.65% word accuracy, a 0.14% improvement). An analysis of the words for which adding phonetic information improved syllabification performance indicates that these were words in which the orthographic ambiguity could be resolved by information on pronunciation. Future research could examine other areas where phonetic information can benefit orthographic processing. In addition, the newly developed deep learning frameworks can be applied to other languages than Dutch.
>
---
#### [new 036] Personalized Turn-Level User Conversation Satisfaction Benchmark
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话系统评估任务，旨在解决个性化用户满意度评价问题。通过构建结合用户记忆和对话上下文的评估模型，提升对特定对话回合满意度的判断能力。**

- **链接: [https://arxiv.org/pdf/2605.29711](https://arxiv.org/pdf/2605.29711)**

> **作者:** Zhefan Wang; Zhiqiang Guo; Weizhi Ma; Min Zhang; Quanjia Yan; Hengliang Luo
>
> **摘要:** User satisfaction with AI assistants is highly personalized: the same response may satisfy one user but disappoint another depending on what each user expects and what they have asked for before. Existing automatic evaluation methods mostly measure generic response quality, making it difficult to judge whether a response satisfies a user at a specific turn. We study this problem as personalized turn-level user conversation satisfaction evaluation. We build a conversation satisfaction evaluator that combines compact user memories with target-turn context to produce satisfaction scores and dissatisfaction-oriented rationales. Meta-evaluation against human satisfaction annotations shows that personalized memory and post-hoc score calibration improve ordinal agreement and dissatisfied-turn detection over supervised, retrieval-based, and generic LLM-as-a-judge baselines. We further introduce PersTurnBench, a personalized turn-level user conversation satisfaction benchmark that uses the verified evaluator to assess generation models via replay. By holding the replay state fixed, PersTurnBench enables controlled comparison of generic generation models and memory-augmented personalized systems without new human labels for every candidate model. The evaluator and benchmark let researchers compare candidate generation models on personalized satisfaction without collecting new user feedback for every model.
>
---
#### [new 037] Do Proactive Agents Really Need an LLM to Decide When to Wake and What to Anchor?
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文属于智能代理任务，解决主动代理中依赖LLM决策效率低的问题。通过引入TGL模型替代LLM处理事件流，提升响应速度与部署可行性。**

- **链接: [https://arxiv.org/pdf/2605.30152](https://arxiv.org/pdf/2605.30152)**

> **作者:** Xiaoze Liu; Ruowang Zhang; Amir H. Abdi; Michel Galley; Zhikai Chen; Siheng Xiong; Xiaoqian Wang; Jing Gao
>
> **备注:** 31 pages, 5 figures, 7 tables
>
> **摘要:** Proactive agents read user activity as text and call an LLM on every event to decide whether to act. But user activity is not natively text: it is a structured event stream of (actor, verb, object, timestamp) tuples that the operating system already maintains in graph form. Rendering the structure as text and asking an LLM to recover it is a round-trip the system never had to take. We treat the always-on signal as graph updates rather than text and use a small temporal-graph-learning (TGL) model as the encoder: one forward pass yields a per-event trigger probability and a per-entity routing score, and only the downstream agent (turning a small structured handoff into a fluent user-facing sentence) is an LLM call, invoked only when the trigger fires. TGL improves F1 on each of 14 backbones (mean +16.7, up to +46.0); in trigger-architecture comparisons, one TGL checkpoint gives the strongest trigger AUCs and the most stable deployed threshold. It runs at 11.13 ms per event on a GPU server and 13.99 ms on a consumer laptop, approximately 4--7x and 12--83x faster than every single-forward LLM-as-trigger configuration tested in each regime, with an approximately 220 MiB BF16 resident footprint deployable on-device alongside the privacy-sensitive activity stream it consumes.
>
---
#### [new 038] Accommodation Goes Both Ways: Studying Linguistic Convergence Between Humans and Language Models
- **分类: cs.CL**

- **简介: 该论文研究人类与语言模型在对话中的语言趋同现象，属于自然语言处理任务。旨在探讨LLMs对人类语言行为的影响，通过分析对话数据发现LLMs过度适应用户，而人类适应模型程度与人际互动相似。**

- **链接: [https://arxiv.org/pdf/2605.29278](https://arxiv.org/pdf/2605.29278)**

> **作者:** Terra Blevins
>
> **摘要:** As LLMs become increasingly integrated into daily life, understanding how their presence will shape human linguistic behavior is an open question. We present a large-scale study of linguistic convergence in human-LLM dialogue, examining how humans and LLMs accommodate each other's linguistic style during multi-turn conversations. Using an asymmetric convergence metric on WildChat, a corpus of real-world ChatGPT transcripts, we find that while LLMs significantly overconverge toward their users on both function word and open-class features across eight languages, human convergence rates in this setting are broadly consistent with human-human baselines. These findings suggest that accommodation in human-LLM dialogue is asymmetric: while LLMs dramatically overfit to their users' style, humans linguistically accommodate LLMs no differently than they would another person.
>
---
#### [new 039] Micro-Macro Retrieval: Reducing Long-Form Hallucination in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言生成任务，旨在解决长文本生成中的幻觉问题。提出Micro-Macro Retrieval框架，通过检索与生成结合提升事实准确性。**

- **链接: [https://arxiv.org/pdf/2605.28828](https://arxiv.org/pdf/2605.28828)**

> **作者:** Yujie Feng; Jian Li; Zhihan Zhou; Pengfei Xu; Yujia Zhang; Xiaoyu Li; Xiaohui Zhou; Alan Zhao; Xi Chen; Xiao-Ming Wu
>
> **摘要:** Large Language Models (LLMs) achieve impressive performance across many tasks but remain prone to hallucination, especially in long-form generation where redundant retrieved contexts and lengthy reasoning chains amplify factual errors. Recent studies highlight a critical phenomenon: the closer key information appears to the model outputs, the higher the factual accuracy. However, existing retrieval-augmented language models (RALMs) lack effective mechanisms to ensure this proximity - external evidence is injected into reasoning via multi-turn retrieval, but this cannot ensure key information stays close to the outputs. We propose Micro-Macro Retrieval (M2R), a novel retrieve-while-generate framework to fill this gap. At the macro level, M2R retrieves coarse-grained evidence from external sources; at the micro level, it extracts essential results from a key information repository built during reasoning and reuses them while generating answers. This design directly addresses the key-information-to-output proximity bottleneck, effectively reducing hallucination in long-form tasks. M2R is trained with a curriculum learning-based reinforcement learning strategy using customized rule-based rewards, enabling stable acquisition of retrieval and grounding skills. Extensive experiments across different benchmarks demonstrate the effectiveness of M2R, especially in lengthy-context settings.
>
---
#### [new 040] How LoRA Remembers? A Parametric Memory Law for LLM Finetuning
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于大模型微调任务，旨在解决LoRA记忆能力的量化问题。通过引入Parametric Memory Law和MemFT策略，提升模型记忆精度与效率。**

- **链接: [https://arxiv.org/pdf/2605.30260](https://arxiv.org/pdf/2605.30260)**

> **作者:** Ziwen Xu; Haiwen Hong; Linsong Yu; Benglei Cui; Longtao Huang; Hui Xue; Ningyu Zhang
>
> **备注:** Ongoing work
>
> **摘要:** Large Language Models (LLMs) must continuously learn and update knowledge to remain effective in dynamic real-world environments. While Low-Rank Adaptation (LoRA) is widely used for such memory updates, existing studies mainly rely on qualitative downstream evaluations, leaving the quantitative capacity limits and underlying dynamics of exact parametric memory largely unexplored. To bridge this gap, we employ LoRA as a controlled memory capacity probe within the latent space to systematically quantify exact parametric memory. We introduce the Parametric Memory Law, a robust power law linking loss reduction Delta L to effective parameters and sequence length. At the token level, fine-grained analysis reveals a deterministic phase transition, demonstrating that a prediction probability of p > 0.5 constitutes a sufficient condition for verbatim recall under greedy decoding. Driven by these insights, we introduce MemFT, a threshold-guided optimization strategy that dynamically redistributes the training budget toward sub-threshold tokens. Empirical evaluations demonstrate that MemFT can enhance memory fidelity and efficiency. Code will be released at this https URL.
>
---
#### [new 041] User-Aware Active Knowledge Acquisition for Emotional Support Dialogue
- **分类: cs.CL**

- **简介: 该论文属于情感支持对话任务，解决用户需求隐晦难懂的问题。提出UKA框架，通过主动学习提升对话质量与用户对齐。**

- **链接: [https://arxiv.org/pdf/2605.29715](https://arxiv.org/pdf/2605.29715)**

> **作者:** Mufan Xu; Kehai Chen; Jiahao Hu; Xinchao Xu; Muyun Yang; Tiejun Zhao; Min Zhang
>
> **摘要:** Emotional support plays an important role in dialogue systems, and its success depends on adapting to a user's evolving and implicit needs across multi-turn interactions while leveraging the strong reasoning capacity of large language models. However, since signals about user needs are often weak, indirect, and can only be disambiguated through multi-turn interaction, existing emotional support methods often struggle to acquire and generalize relevant conversational knowledge efficiently. To bridge this gap, we introduce User-Aware Active Knowledge Acquisition (UKA), a gradient-free active dialogue learning framework that explicitly represents uncertainty about user needs and incorporates active learning into both knowledge acquisition and response this http URL propose a Theory-of-Mind uncertainty estimation mechanism that allows the model to prioritize responses, thereby eliciting more informative user feedback. UKA is capable of efficiently exploring user-aligned conversational knowledge during training while maintaining robustness at test time. Experiments across multiple dialogue benchmarks and model architectures demonstrate that our approach consistently outperforms strong baselines in dialogue quality and user alignment.
>
---
#### [new 042] DLM-SWAI: Steering Diffusion Language Models Before They Unmask
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本生成控制任务，解决扩散语言模型（DLM）的定向生成问题。提出DLM-SWAI方法，在不重新训练模型的情况下，通过预计算的风格分数引导生成，提升控制效果与效率。**

- **链接: [https://arxiv.org/pdf/2605.29626](https://arxiv.org/pdf/2605.29626)**

> **作者:** Hyeseon An; Yo-Sub Han
>
> **备注:** preprint
>
> **摘要:** Steering language model generation toward desired textual properties is essential for practical deployment, and inference-time methods are particularly appealing because they enable controllable generation without retraining. Recent work has also highlighted diffusion language models as an emerging generation paradigm with distinct decoding properties. However, most existing steering approaches either rely on auxiliary models or are designed for autoregressive next-token decoding, making them difficult to apply to diffusion language models DLMs, which generate text through iterative denoising of partially masked sequences. Therefore, we propose DLM-SWAI, a simple training-free steering method that biases the token distribution at each denoising step using pre-computed token-level style scores. Experiments on style and safety control tasks show that DLM-SWAI effectively steers diffusion language models while preserving generation quality and requiring minimal computational overhead. Ablations further reveal a controllable trade-off between steering strength and fluency, and our analysis links class-wise steerability to the strength of token-level attribute cues.
>
---
#### [new 043] On Asymmetric Optimization of Reasoning and Perception in Vision-Language Model Post-Training
- **分类: cs.CL; cs.CV**

- **简介: 该论文研究视觉语言模型后训练中的感知与推理不对称问题，通过诊断框架分析原因并提出优化方法，以提升端到端性能。**

- **链接: [https://arxiv.org/pdf/2605.29496](https://arxiv.org/pdf/2605.29496)**

> **作者:** Xueqing Wu; Yu-Chi Lin; Kai-Wei Chang; Nanyun Peng
>
> **备注:** Project: this https URL
>
> **摘要:** Post-training has greatly improved reasoning in frontier vision-language models, yet its gains for perception remain comparatively limited, creating a bottleneck for end-to-end visual reasoning. To investigate this gap, we introduce a controlled diagnostic framework with two synthetic tasks that disentangle perception from reasoning. Our analysis reveals a consistent perception-reasoning asymmetry: posttraining improves reasoning more substantially than perception, though the underlying mechanism differs by training paradigm. For supervised fine-tuning (SFT), this asymmetry stems from token imbalance in chain-of-thought supervision, where perception occupies fewer tokens and thus receives a weaker training signal. Dynamically reweighting the loss mitigates this imbalance and boosts end-to-end performance by up to 18.2. For reinforcement learning (RL), the asymmetry instead arises from reward coupling: outcome rewards correlate more strongly with reasoning than with perception, weakening the signal for perception learning. Adding a perception-aware reward alleviates the imbalance and improves end-to-end accuracy by up to 6.0; even without groundtruth perception rewards, a reliable surrogate reward provide useful signal, yielding gains of 3.2 points. Together, our results comprehensively diagnose asymmetric optimization and suggest concrete interventions to balance perception and reasoning.
>
---
#### [new 044] Verifiable Rewards Beyond Math and Code: Lightweight Corpus-Grounded Process Supervision for Factual Question Answering
- **分类: cs.CL**

- **简介: 该论文针对事实问答任务中的奖励设计问题，提出CorVer方法，利用语料库统计替代神经验证器，提升奖励信号的准确性和效率。**

- **链接: [https://arxiv.org/pdf/2605.29648](https://arxiv.org/pdf/2605.29648)**

> **作者:** Shicheng Fan; Haochang Hao; Dehai Min; Weihao Liu; Philip S. Yu; Lu Cheng
>
> **摘要:** Applying reinforcement learning to improve factual accuracy in knowledge-intensive question answering faces a reward design dilemma. Response-level rewards provide only coarse supervision and cannot distinguish correct from incorrect statements within a reasoning trace. Sentence-level alternatives offer finer-grained feedback, but typically rely on NLI verifiers, LLM judges, or knowledge-verification pipelines that are expensive to deploy at RL scale and often unreliable for rare-entity facts, where accurate reward signals are especially important. We propose CorVer (Corpus Verify), a lightweight, plug-in-ready process reward that replaces neural verifiers with a corpus-grounded signal derived from Wikipedia co-occurrence statistics. CorVer assigns sentence-level credit and maps it to token-level advantages via a simple alignment, requiring only a 0.5B extractor and a single corpus lookup per sentence. Across 30 (model, benchmark) cells spanning six instruction-tuned models (3B to 14B) and five QA benchmarks, CorVer improves over the raw baseline for every cell, with an average TriviaQA gain of +4.1 pp. It also outperforms four neural-verifier baselines in 18 of 20 cells under their feasible configurations, while training 4.8 to 8.4x faster.
>
---
#### [new 045] Data filtering methods for training language models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决数据质量对模型效果的影响问题。通过对比两种自动标签错误检测方法在不同俄语文本分类数据集上的表现，验证了针对性数据过滤的有效性。**

- **链接: [https://arxiv.org/pdf/2605.29807](https://arxiv.org/pdf/2605.29807)**

> **作者:** Egor Shevchenko; Elena Bruches
>
> **备注:** AINL-2026
>
> **摘要:** Data quality is a critical factor in the effectiveness of machine learning models. Label errors, present even in widely used benchmarks, introduce noise into training data and reduce model generalization. In this work, we conduct a comparative analysis of two automatic label error detection methods - Confident Learning and Dataset Cartography - on three Russian text classification corpora of varying size, number of classes, and domain: ru_emotion_e-culture (49,123 examples, emotion classification), RuCoLA (8,524 examples, linguistic acceptability), and TERRa (2,337 examples, textual entailment recognition). We use the pre-trained rubert-base-cased model fine-tuned on each corpus. To verify the meaningfulness of filtering, we conduct control experiments with random removal of an equivalent number of examples. Results show that the effectiveness of both methods depends strongly on dataset characteristics: on large corpora with low noise levels, filtering does not improve performance, while on small datasets with high noise, Confident Learning achieves a significant F1-macro improvement. Dataset Cartography demonstrates more conservative behavior, removing fewer examples. Across all corpora, targeted removal by both methods outperforms random removal, confirming the meaningfulness of the approaches.
>
---
#### [new 046] Loong: A Human-Like Long Document Translation Agent with Observe-and-Act Adaptive Context Selection
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文档级翻译任务，旨在解决长文档翻译中上下文限制和冗余问题。提出Loong模型，通过自适应选择上下文提升翻译质量。**

- **链接: [https://arxiv.org/pdf/2605.30274](https://arxiv.org/pdf/2605.30274)**

> **作者:** Yutong Wang; Xuebo Liu; Derek F. Wong; Zhilin Li; Rongqing Jiang; Min Zhang; Shimin Tao; Daimeng Wei; Min Zhang
>
> **摘要:** Document-level translation remains one of the most challenging tasks for large language models, which are constrained by limited context windows that impede global cohesion, while simultaneously suffering from redundant contextual information that degrades translation quality. To address this, we propose a human-like long document translation agent called Loong, which leverages a 3E memory module (Essence-Exemplar-Entity) to store summaries, sentence pairs, and entity records as historical context. Instead of passively attending to all history, Loong performs deep reasoning to adaptively identify the optimal context for translation guidance. Loong optimizes its context policy through reinforcement learning, utilizing preference data derived from its own sampled observe-and-act reasoning trajectories. Empirical evaluations demonstrate that Loong achieves substantial translation quality improvements in English $\Leftrightarrow$ Chinese, German, and French directions, with average gains of up to 13.0 points across the three evaluation metrics. Furthermore, Loong exhibits strong generalization across domains and robustness against contextual noise, while maintaining remarkable stability in ultra-long document translation. Our code is released at this https URL.
>
---
#### [new 047] GRUFF: LLM Pronoun Fidelity, Reasoning, and Biases in German
- **分类: cs.CL**

- **简介: 该论文属于语言模型性别指代研究任务，旨在解决德语中代词一致性与偏见问题。构建了GRUFF数据集，分析模型在不同性别系统下的指代能力与 stereotypes 关联。**

- **链接: [https://arxiv.org/pdf/2605.30214](https://arxiv.org/pdf/2605.30214)**

> **作者:** Fabian Mewes; Anne Lauscher; Vagrant Gautam
>
> **摘要:** Third-person singular pronouns have long been used to study stereotypical biases in language models and to test their abilities to reason about reference. More recently, the interplay between reasoning and bias has been investigated with the task of pronoun fidelity, which assesses models' abilities to correctly reuse a previously-specified pronoun for a discourse entity, independent of other potentially distracting discourse entities mentioned in between. However, such research focuses on English, which is a language with limited grammatical gender and almost no gender agreement. In this paper we contribute a novel, large-scale dataset, GRUFF, to measure pronoun fidelity in German, covering four different gender agreement systems in nouns, and four sets of pronouns. With this dataset, we show that LLMs show strong grammatical agreement for masculine and feminine entities in the absence of explicit context, but not for neopronouns xier and en. Models are generally not robust to distractors, but encoder-only models are more robust in German than in English, reflecting the importance of grammatical gender. Finally, we show that occupational stereotypes in this context are poorly correlated across grammatical cases, and across most models, except ones with closely related architectures. We release all code and data to encourage further work on gender-inclusive language and referential reasoning in German.
>
---
#### [new 048] Training Deliberative Monitors for Black-Box Scheming Detection
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于AI安全领域，解决黑盒代理中的阴谋行为检测问题。通过训练仅依赖行动的监控模型，无需访问代理内部信息，实现高效低成本的检测。**

- **链接: [https://arxiv.org/pdf/2605.29601](https://arxiv.org/pdf/2605.29601)**

> **作者:** Aditya Sinha; Akshat Naik; Victor Gillioz; Simon Storf; Kilian Merkelbach; Rich Barton-Cooper; Axel Højmark; Marius Hobbhahn
>
> **摘要:** As autonomous agents become more capable of performing real-world tasks, distinguishing scheming behavior from benign task pursuit may become a central AI control problem. Existing monitors often rely on chain-of-thought access or internal activations, or use prompted frontier models, all of which can be unavailable, unreliable or expensive in deployment. In this work, we study action-only deliberative monitors: smaller open-weight models trained to detect scheming and sabotage from agentic trajectories without accessing the monitored agent's reasoning or model internals. Our method, inspired by deliberative alignment, uses a scheming specification to elicit structured rationales from a frontier teacher, filters them with a separate judge, and distills the highest-quality rationales into open-weight monitors with supervised fine-tuning and reinforcement learning. We train on five datasets, and evaluate across six out-of-distribution agentic misalignment benchmarks. We show that applying our method to Qwen3.5-27B yields higher performance than all low-cost frontier models as prompted monitors (Gemini 3.1 Flash-Lite, GPT-5.4 Nano, and Claude Haiku 4.5) and than Gemini 2.5 Pro, while also achieving lower marginal inference cost (token-metered USD per 1,000 evaluations). Stronger prompted frontier monitors (Gemini 3.1 Pro, GPT-5.4, Claude Sonnet 4.6, and Claude Opus 4.6) achieve higher performance but at roughly $16$--$34\times$ higher marginal inference cost. Several of our trained monitors are positioned on the empirical cost--performance Pareto frontier among the monitors we evaluate, providing practical low-cost, low-FPR alternatives to prompted frontier models.
>
---
#### [new 049] CorPipe at CRAC 2026: Empty Nodes and Cross-Lingual Transfer in Multilingual Coreference Resolution
- **分类: cs.CL**

- **简介: 该论文属于多语言共指消解任务，解决空节点预测与跨语言迁移问题。提出CorPipe 26系统，整合空节点预测与共指链接，提升性能并进行多组实验验证。**

- **链接: [https://arxiv.org/pdf/2605.30133](https://arxiv.org/pdf/2605.30133)**

> **作者:** Milan Straka
>
> **备注:** Accepted to CODI-CRAC 2026
>
> **摘要:** We introduce CorPipe 26, our winning submission to the CRAC 2026 Shared Task on Multilingual Coreference Resolution. The fifth edition of this shared task focuses mainly on the comparison of generative LLMs and specialized systems; additionally, 5 more datasets and 2 new languages are introduced. CorPipe 26 is an improved version of CorPipe 25, with a new variant predicting empty nodes together with mentions and coreference links in a single model. Our system outperforms all other submissions in the LLM track by 2.8 percent points and all submissions in the unconstrained track by 9.5 percent points. Furthermore, we perform a series of ablation experiments with different model sizes, empty node prediction methods, and cross-lingual zero-shot evaluation. The source code and the trained models are publicly available at this https URL.
>
---
#### [new 050] Casual as an Anchor: Resolving Supervision Misalignment in Formality Transfer Dataset
- **分类: cs.CL**

- **简介: 该论文属于形式化转换任务，旨在解决基准数据中监督信号不一致的问题。通过引入三层次标注框架，提升模型生成正式语言的准确性。**

- **链接: [https://arxiv.org/pdf/2605.29365](https://arxiv.org/pdf/2605.29365)**

> **作者:** Hyojeong Yu; Hyukhun Koh; Minsung Kim; Kyomin Jung
>
> **备注:** HEAL@CHI 2026 Workshop Paper
>
> **摘要:** Formality transfer is commonly framed as a symmetric bidirectional task between informal and formal registers. We argue that this framing conceals a supervision design flaw in existing benchmarks such as GYAFC: binary human rewrites encode relative stylistic shifts rather than absolute human notions of formality. Consequently, models learn to generate pseudo-formal outputs that satisfy benchmark labels while failing to produce genuinely formal language. We quantify this misalignment by re-evaluating benchmark formal labels under a human-aligned definition of formality, revealing substantial discrepancies that propagate to consistent informal-to-formal failures across model families. To address this issue, we reconceptualize formality transfer as a graded dimension rather than a binary attribute. We introduce a three-level spectrum: informal, casual, and formal, where casual serves as an explicit intermediate state that clarifies supervision signals. Based on this framework, we introduce 3LF, a dataset providing parallel supervision across all three levels. Training on 3LF substantially reduces informal-to-formal failures and improves alignment with human perception. For example, GPT-4.1-nano improves from 0.06 to 0.88 F1 in the informal-to- formal direction despite 3LF being significantly smaller than GYAFC. We further demonstrate that these gains cannot be reproduced through in-context learning alone and provide qualitative analyses of ambiguity-driven errors and meaning distortions. Overall, our findings demonstrate how supervision design shapes stylistic alignment and highlight the importance of alignment-aware benchmark construction in controllable text generation.
>
---
#### [new 051] PhoneWorld: Scaling Phone-Use Agent Environments
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出PhoneWorld，解决手机使用代理环境难以规模化构建的问题。通过转换真实GUI轨迹生成可控环境、任务和验证器，提升基准性能。属于移动代理环境构建任务。**

- **链接: [https://arxiv.org/pdf/2605.29486](https://arxiv.org/pdf/2605.29486)**

> **作者:** Zhengyang Tang; Yuxuan Liu; Xin Lai; Junyi Li; Pengyuan Lyu; Jason; Yiduo Guo; Zhengyao Fang; Yang Ding; Yi Zhang; Weinong Wang; Huawen Shen; Xingran Zhou; Liang Wu; Fei Tang; Sunqi Fan; Shangpin Peng; Zheng Ruan; Anran Zhang; Benyou Wang; Rui Yan; Ji-Rong Wen; Chengquan Zhang; Han Hu
>
> **备注:** work in progress
>
> **摘要:** A central bottleneck for phone-use agents is that controllable, reproducible environments covering real mobile behavior are hard to build at scale. Existing mobile-agent benchmarks have made important progress on evaluation, but they do not by themselves provide a scalable way to construct many new phone-use environments. We present PhoneWorld, a reusable pipeline that converts real GUI trajectories and screenshots into controllable phone-use environments, executable tasks, automatic verifiers, and training rollouts. Rather than hand-building one mobile benchmark at a time, PhoneWorld uses real trajectories to recover which screens matter, how screens connect, which interactions must change environment state, and which user goals admit automatic verification. From these signals, it builds runnable mock Android apps backed by read-only app content and mutable state, then derives executable tasks, rule-based verifiers, and training rollouts from the same environments. In its current instantiation, PhoneWorld covers 34 apps across 16 domains, spanning common consumer mobile behaviors such as search, browsing, shopping, booking, media, and social interaction. Under a fixed training budget, replacing 10K steps from an auxiliary AndroidWorld corpus in an AndroidWorld-based baseline with broad PhoneWorld supervision improves all four evaluation benchmarks at once, raising HYMobileBench by 17.7 points, AndroidControl by 6.0 points, AndroidWorld by 14.7 points, and PhoneWorld by 52.5 points. We then study two additional scaling questions: increasing the amount of PhoneWorld supervision strongly improves PhoneWorld performance, and under a fixed PhoneWorld budget, expanding app coverage yields even larger gains. Overall, PhoneWorld shifts the focus from building one mobile benchmark at a time to scaling the supply of phone-use environments themselves.
>
---
#### [new 052] Error as a Lens: Probing LLM Reasoning through Synthetic Misconception Generation
- **分类: cs.CL**

- **简介: 该论文属于教育技术任务，旨在生成针对性的合成错误以辅助教学研究。针对真实学生错误数据稀缺的问题，提出框架生成符合特定认知错误类型的错误答案。**

- **链接: [https://arxiv.org/pdf/2605.29007](https://arxiv.org/pdf/2605.29007)**

> **作者:** Xinming Yang; Jun Li
>
> **摘要:** Personalized tutoring, teacher training, and education research need access to \emph{targeted} synthetic misconceptions, but privacy and IRB constraints make labelled corpora of real student errors scarce. LLMs could in principle generate synthetic errors at scale, but producing an arbitrary wrong answer is easy for a modern LLM while producing one that matches a specified cognitive failure mode is much harder. We present a framework that generates errors targeted to a five-class taxonomy adapted from the revised Bloom's taxonomy, evaluated on questions from the TheoremQA dataset. A Generation Agent (GA) drafts a candidate erroneous solution conditioned on a target class, and an Examination Agent (EA) judges whether the draft is incorrect and class-consistent. The framework yields a reusable recipe for building class-stratified synthetic error datasets where authentic student corpora are unavailable. As a secondary diagnostic, targeted error generation is substantially harder than free-form incorrect-answer generation, and answer-grounding contributes more than expanded examples or external textbook content.
>
---
#### [new 053] Draft-OPD: On-Policy Distillation for Speculative Draft Models
- **分类: cs.CL**

- **简介: 该论文属于语言模型推理加速任务，解决草案模型在推测解码中效果受限的问题。通过提出Draft-OPD方法，提升草案模型的接受长度和推理效率。**

- **链接: [https://arxiv.org/pdf/2605.29343](https://arxiv.org/pdf/2605.29343)**

> **作者:** Haodi Lei; Yafy Li; Haoran Zhang; Shunkai Zhang; Qianjia Cheng; Xiaoye Qu; Ganqu Cui; Bowen Zhou; Ning Ding; Yun Luo; Yu Cheng
>
> **摘要:** Speculative decoding accelerates large language model inference by pairing a target model with a lightweight draft model whose proposed tokens are verified in parallel. A common way to build draft models, like EAGLE3 or DFlash is supervised fine-tuning (SFT) on target-generated trajectories. However, we observe that SFT quickly plateaus: the draft model's acceptance length on test data stops improving. The reason is an offline-to-inference mismatch: In SFT, the drafter learns from fixed target-generated trajectories, whereas during speculative decoding it is evaluated on blocks proposed under its own policy. This motivates on-policy distillation (OPD), where the target model supervises the drafter on draft-induced states. Yet OPD remains difficult for draft models, as they cannot reliably roll out complete sequences independently, whereas target-assisted generation makes the collected sequences follow the target distribution and thus eliminates the on-policy signal. We therefore propose Draft-OPD, which uses target-assisted rollout for stable continuations and replays drafting from the verification-exposed error positions. This allows the drafter to learn from target feedback on both accepted and rejected proposals, focusing training on the draft-induced errors that limit speculative acceptance. Experiments show that Draft-OPD achieves over $5\times$ lossless acceleration for thinking models across diverse tasks, improving over EAGLE-3 and DFlash by 23\% and 13\%.
>
---
#### [new 054] Dial HEALTHDIAL for Advice: A Multilingual and Multi-Parallel Spoken Dialogue Dataset for Knowledge-Grounded Information Seeking
- **分类: cs.CL**

- **简介: 该论文提出HEALTHDIAL，一个用于知识增强信息检索对话系统的多语言、多并行语音对话数据集，解决多语言对话系统开发与评估难题。**

- **链接: [https://arxiv.org/pdf/2605.30107](https://arxiv.org/pdf/2605.30107)**

> **作者:** Songbo Hu; Yinhong Liu; Ej Zhou; Evgeniia Razumovskaia; Xiaobin Wang; Alexander Fraser; Ivan Vulić; Anna Korhonen
>
> **备注:** Accepted to Findings of ACL 2026
>
> **摘要:** Creating spoken dialogue datasets is methodologically challenging, and these challenges are amplified when the goal is to build multilingual, multi-parallel datasets at scale. This work introduces HEALTHDIAL, a large-scale, multilingual, and multi-parallel dataset for developing and evaluating retrieval-augmented generation (RAG)-based spoken dialogue systems. The dataset comprises 6,000 information-seeking dialogues (1,500 per language) grounded in trusted content from the World Health Organization (WHO) and 163 hours of user speech recorded from native speakers of diverse dialects across four official WHO languages: Arabic, Chinese, English, and Spanish. Each speaker is annotated with demographic (e.g., gender, age) and sociolinguistic (e.g., primary language, region of origin) variables. We report benchmark results across key dialogue tasks, which reveal consistent performance disparities across languages, even among high-resource ones. To support future research, we release the dataset, a prototype system, and a toolkit for data collection and system evaluation.
>
---
#### [new 055] SkillBrew: Multi-Objective Curation of Skill Banks for LLM Agents
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文提出SkillBrew，解决LLM代理技能库的多目标优化问题，旨在提升技能库的效用、多样性和覆盖率。**

- **链接: [https://arxiv.org/pdf/2605.29440](https://arxiv.org/pdf/2605.29440)**

> **作者:** Wentao Hu; Zhendong Chu; Yiming Zhang; Junda Wu; Ming Jin; Xiangyu Zhao; Yilei Shao; Yanfeng Wang; Qingsong Wen
>
> **备注:** 16 pages. Preprint. Under review
>
> **摘要:** Retrieval-augmented LLM agents increasingly rely on curated skill banks: collections of reusable textual principles that guide decision making on complex tasks. Existing approaches typically expand these banks in an append-only fashion, continuously adding new skills without removing redundant, outdated, or harmful ones, resulting in inefficient and poorly curated repositories. In this paper, we formulate the skill bank curation as a constrained multi-objective problem: a desirable bank must be useful for the agent, diverse in its content, and provide good coverage of the query distribution. To this end, we introduce SkillBrew, a multi-objective curation framework that formalizes skill bank curation as Pareto-aware optimization under a utility constraint, and solves it via a bi-level propose-then-verify loop. We evaluate our approach on two public benchmarks. Our findings suggest that treating skill banks as objects of principled curation, rather than ever-growing append-only logs, is an important step toward building self-improving LLM agents.
>
---
#### [new 056] Teaching Language Models to Check Grounded Claim Factuality with Human Test-Taking Strategies
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于事实核查任务，旨在提升语言模型对 grounded claim 的事实性判断能力。通过将任务转化为阅读理解，并引入测试策略，提高推理效率与准确性。**

- **链接: [https://arxiv.org/pdf/2605.29712](https://arxiv.org/pdf/2605.29712)**

> **作者:** Yuxuan Ye; Raul Santos-Rodriguez; Edwin Simpson
>
> **备注:** ACL 2026 Main
>
> **摘要:** Grounded claim factuality checking is important for large language model (LLM) applications such as retrieval-augmented generation, as it helps users assess the correctness of generated outputs. Existing metrics using entailment classifiers require dataset-specific threshold tuning, while LLM-based approaches often use direct prompting, which underutilises the reasoning capabilities of LLMs. We address this by formulating grounded claim factuality checking as a true/false reading comprehension task and prompting LLMs with explicit test-taking strategies for efficient reasoning. Our method reduces token usage by over 80% compared to unguided open-ended reasoning, and achieves competitive performance to more expensive alternatives across two factuality benchmarks, setting a new state of the art on one. To further reduce inference cost, we train small language models (SLMs) to replace LLMs in the checking pipeline. Using supervised fine-tuning (SFT) and a self-revision mechanism, the SLMs learn to improve their factuality judgements. Experimental results show that the resulting SLMs perform on par with strong baselines, combining low inference costs with generating supporting rationales to support interpretability. Code and datasets will be released upon acceptance.
>
---
#### [new 057] SEAL: Can Saturated Benchmarks Be Revived by LLM-as-a-Meta-Judge?
- **分类: cs.CL**

- **简介: 该论文提出SEAL方法，解决语言模型基准测试饱和问题，通过改进评估机制提升排名准确性，减少评估成本。**

- **链接: [https://arxiv.org/pdf/2605.30104](https://arxiv.org/pdf/2605.30104)**

> **作者:** Jiamin Chen; Yidi Wu; Qiexiang Wang; Qianben Chen; Yuchen Li; Yansen Zhang; Xiaokun Zhang; Wangchunshu Zhou; Chen Ma
>
> **摘要:** Widely used language-model benchmarks are increasingly saturated, with frontier systems often receiving near-tied scores that standard metrics cannot resolve. Rather than constructing harder alternatives, we ask whether existing tasks can be made informative again through improved evaluation over the same candidate outputs. Therefore, we present Seeded Elimination with Adaptive LLM-as-a-Meta-Judge, a self-improving evaluation protocol for extracting latent ranking signal from saturated benchmarks. SEAL seeds candidate outputs into a single elimination and evaluates each match with task-level principles plus self-improving checklist criteria. We evaluate SEAL on multiple saturated benchmarks covering code generation, mathematical reasoning, knowledge-intensive question answering, and tool-use agent task completion. Across these settings, SEAL improves the ranking-accuracy--latency trade-off over competing protocols, attaining 0.83--1.00 Spearman agreement with full pairwise judging and 4/4 top-1 agreement, while requiring only 11.89 calls per task compared with 28.00 for full pairwise evaluation.
>
---
#### [new 058] Domino: Decoupling Causal Modeling from Autoregressive Drafting in Speculative Decoding
- **分类: cs.CL**

- **简介: 该论文属于大模型推理加速任务，解决 speculative decoding 中因果建模与生成成本的矛盾。提出 Domino 框架，分离因果建模与自回归生成，提升推理效率。**

- **链接: [https://arxiv.org/pdf/2605.29707](https://arxiv.org/pdf/2605.29707)**

> **作者:** Jianuo Huang; Yaojie Zhang; Qituan Zhang; Hao Lin; Hanlin Xu; Linfeng Zhang
>
> **摘要:** Speculative decoding accelerates LLM inference by drafting multiple tokens and verifying them in parallel with the target model. However, its practical speedup is constrained by the trade-off between draft quality and drafting cost: autoregressive drafters model causal dependencies among draft tokens but incur sequential overhead, while parallel drafters reduce drafting cost but weaken intra-block dependency modeling. In this paper, we propose Domino, a speculative decoding framework that decouples causal dependency modeling from expensive autoregressive draft execution. Domino first uses a parallel draft backbone to produce preliminary draft distributions for the entire block, and then applies a lightweight Domino head to refine them with prefix-dependent causal information. To stabilize teacher-forced causal encoding, we further introduce a base-anchored training curriculum that first strengthens the parallel backbone and then gradually shifts optimization toward the causally corrected final distribution. Experiments on Qwen3 models show that Domino achieves up to \(5.49\times\) end-to-end speedup under the Transformers backend and up to \(5.8\times\) throughput speedup under SGLang serving.
>
---
#### [new 059] No Reader Left Behind: Multi-Agent Summaries Everyone Can Understand
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本摘要任务，旨在解决不同读者群体在理解政府文档时的语言和认知障碍问题。研究提出NRLB框架，通过多智能体模拟不同读者，提升摘要的可读性和准确性。**

- **链接: [https://arxiv.org/pdf/2605.28836](https://arxiv.org/pdf/2605.28836)**

> **作者:** Jimin Jung; MyoungJin Kim; Jaehyung Seo; Heuiseok Lim
>
> **摘要:** The Plain Writing Act in the United States requires government documents to be accessible in clear and simple language that the general public can easily understand, yet existing summarization systems struggle to address diverse linguistic and cognitive barriers among general readers. We present NRLB (No Reader Left Behind), a multi-agent framework for plain language summarization that simulates three representative reader groups: elementary school student readers, non-native readers, and readers with attention deficits. NRLB combines template-based planning with iterative, reader-oriented refinement, enabling systematic detection and resolution of difficult terms, missing contexts, and confusing sentences. Evaluations across multiple datasets demonstrate consistent improvements in readability while preserving factual accuracy. Human evaluation further validates NRLB's impact, with annotator preference rates ranging from 55% to 76%, highlighting NRLB's potential to produce plain language summaries that are both faithful to the source and broadly accessible to the general public.
>
---
#### [new 060] World Models in Words: Auditing Physical State-Transition Commitments in Vision-Language Models
- **分类: cs.CL**

- **简介: 该论文属于视觉-语言模型的物理推理评估任务，旨在解决模型仅凭答案评分而忽视物理状态转换的问题。提出框架\wmw，通过检查状态转换轨迹来审计模型的物理承诺。**

- **链接: [https://arxiv.org/pdf/2605.29585](https://arxiv.org/pdf/2605.29585)**

> **作者:** Emmanuelle Bourigault
>
> **备注:** 8 pages, 3 figures, 5 tables
>
> **摘要:** Vision-language models (VLMs) are increasingly used to answer questions about physical scenes, yet most evaluations reduce performance to a final answer. This hides whether the model perceived the right objects, represented the right physical state, predicted a plausible transition, or merely selected the right option for the wrong reasons. We introduce \wmw, an evaluation framework for auditing the \emph{language-expressed physical commitments} of VLMs. Instead of scoring only $I,q\mapsto a$, we ask models to produce a typed trace $I,q\mapsto(s_0,\Delta s,s_1,a)$: an initial state, a state transition, a resulting state, and an answer. A hybrid verifier then checks schema validity, state grounding, transition consistency, and answer-trace compatibility, yielding typed error labels such as object, relation, force, transition, temporal, unit/scale, and faithfulness errors. We release \tracebank, a controlled trace resource with \nSeed schema- and recomputation-validated synthetic scenarios across \nFamilies physics families, \nPairs minimally perturbed contrastive preference pairs, verifier code, audit guidelines, and model outputs. We evaluate \nModels VLMs on both controlled and external physical-reasoning examples. \wmw reveals failures that answer-only evaluation misses: 35\% of correct answers from mid-tier models are backed by physically invalid traces. Verifier-guided reranking recovers up to 7 percentage points of trace validity without sacrificing answer accuracy, and trace-level preference tuning reduces hidden inconsistency by 41\% relative. The contribution is not another final-answer physics benchmark, but a reusable protocol for measuring whether a VLM's stated physical world can be true at the same time as its answer.
>
---
#### [new 061] LLMBridge: An LLM Pipeline for End-to-end Referential Bridging Resolution in English
- **分类: cs.CL**

- **简介: 该论文提出LLMBridge，用于英语指代桥接解析任务，结合启发式处理与大模型推理能力，提升端到端解析效果。**

- **链接: [https://arxiv.org/pdf/2605.29048](https://arxiv.org/pdf/2605.29048)**

> **作者:** Lauren Levine; Amir Zeldes
>
> **摘要:** In this paper, we introduce LLMBridge, a new LLM based system for the task of end-to-end referential bridging resolution in English. Our bridging resolution pipeline combines heuristic pre/post-processing with the natural language inference ability that comes from LLMs. We evaluate our bridging resolution pipeline on 3 datasets which have been used for referential bridging resolution evaluation in English: ISNotes, BASHI, and GUMBridge. Comparison to previous bridging resolution systems shows that the performance of LLMBridge surpasses previous state-of-the-art (SoTA) systems for all 3 datasets in the challenging End-to-end Evaluation Setting, as well as the Basic Bridging Resolution Evaluation Setting (gold bridging anaphor given). We also conduct a thorough error analysis of the LLMBridge performance, examining what varieties of bridging remain difficult for LLM based systems to identify. With this paper, we release the code for the LLMBridge pipeline.
>
---
#### [new 062] Rethinking Stepwise Model Routing: A Cost-Efficient Table Reasoning Perspective
- **分类: cs.CL**

- **简介: 该论文属于表格推理任务，旨在解决大模型推理成本高的问题。通过分析表格和文本token的不确定性，提出EcoTab框架，提升推理效率与准确性。**

- **链接: [https://arxiv.org/pdf/2605.29319](https://arxiv.org/pdf/2605.29319)**

> **作者:** Shenghao Ye; Yuxiang Wang; Yu Guo; Dong Jin; Shuangwu Chen; Jian Yang
>
> **备注:** 17pages, 15 figures, submitted to EMNLP 2026
>
> **摘要:** Large Reasoning Models (LRMs) achieve strong performance on table reasoning tasks but incur substantial inference cost due to long reasoning traces. Stepwise model routing mitigates this issue by dynamically assigning reasoning steps to smaller or larger models. However, stepwise model routing for table reasoning remains underexplored. Through empirical analysis, we find that reasoning steps involving tables contain two types of tokens with distinct uncertainty distributions: table tokens grounded in table structure, such as cell values and headers, and text tokens representing surrounding natural-language reasoning. The uncertainty of both token types is correlated with the risk that the model makes an error in the next reasoning step. However, existing methods fail to model them separately, leading to suboptimal routing decisions. To address this, we propose EcoTab, a table-aware stepwise routing framework for efficient table reasoning. At each reasoning step, EcoTab separately estimates the uncertainties of table tokens and text tokens, maps them to next-step failure risks for the small model, and combines the two risks for routing. Experiments on multiple table reasoning benchmarks show that EcoTab consistently outperforms strong baselines and achieves a better balance between accuracy and efficiency.
>
---
#### [new 063] Adapting Multilingual Embedding Models to Turkish via Cross-Lingual Tokenizer Surgery and Offline Distillation
- **分类: cs.CL**

- **简介: 该论文针对土耳其语句子嵌入任务，提出一种高效适配方法，解决多语言模型在土耳其语上的性能不足问题。通过优化分词器和离线蒸馏，构建了一个高性能、低成本的土耳其语嵌入模型。**

- **链接: [https://arxiv.org/pdf/2605.29992](https://arxiv.org/pdf/2605.29992)**

> **作者:** M. Ali Bayram; Banu Diri; Savaş Yıldırım
>
> **备注:** 14 pages, 2 figures, 4 tables, Appendix included
>
> **摘要:** Sentence embeddings are a foundational component for semantic search, clustering, classification, and retrieval-augmented generation. This paper presents embeddingmagibu-200m, a Turkish-focused sentence embedding model that produces 768-dimensional L2-normalized vectors and supports an 8,192-token context window, far exceeding the 512-token limit of earlier BERT-based Turkish encoders. Instead of full pretraining, an efficient three-stage adaptation pipeline is introduced: (1) construct a Turkish-optimized multilingual tokenizer with a 131,072 vocabulary by pruning redundant tokens from the teacher's vocabulary and incorporating multilingual tokens via frequency analysis on a 40-language corpus, (2) clone a teacher embedding model while preserving transformer backbone weights and initializing a compatible embedding table for the new vocabulary via mean-composition token mapping, and (3) perform offline embedding distillation from precomputed teacher vectors using a cosine similarity objective over a balanced 40-language Wikipedia corpus. The resulting student model contains approximately 200M parameters and trains in roughly four hours on a single GPU by avoiding online teacher inference during training, at a total cost of $5-$20. Empirically, Pearson/Spearman correlations of 77.55%/77.45% are obtained on STSbTR, surpassing the 300M-parameter teacher model (73.84%/72.92%). On TR-MTEB (26 tasks), a mean score of 63.9% is achieved (7th out of 26 models), providing a competitive cost-quality trade-off with 33% fewer parameters than the teacher. To facilitate reproducibility and downstream use, all artifacts are released including model weights, tokenizer files, precomputed embedding datasets, and open-source cloning and distillation tooling.
>
---
#### [new 064] Latent Performance Profiling of Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于模型评估任务，旨在解决传统基准测试无法全面反映大语言模型能力的问题。提出LPP框架，通过分析隐层激活和输出分布，提供更深入的模型性能诊断。**

- **链接: [https://arxiv.org/pdf/2605.30018](https://arxiv.org/pdf/2605.30018)**

> **作者:** Tanmoy Chakraborty; Ayan Sengupta; Suparna Bhattacharya; Partha Pratim Chakrabarti; Amlan Chakrabarti; Supratik Chakraborty; Partha Pratim Das; Lipika Dey; Richa Singh; Mayank Vatsa
>
> **摘要:** Large language models (LLMs) frequently achieve impressive scores on standardized benchmarks, yet accuracy alone offers a limited view of their capabilities. Evaluating open-source LLMs through leaderboards faces persistent issues like data contamination, narrow task scope, and weak alignment with real-world reliability. Benchmark-based evaluations such as MMLU PRO, BBH, or IFEval primarily capture \textit{what} a model outputs on fixed test sets, not \textit{how} it processes information, calibrates uncertainty, or structures internal knowledge. In this article, we advocate for a shift from benchmark-centric evaluation toward a complementary, \textit{state-centered intrinsic assessment} of LLMs. To this end, we introduce \textbf{Latent Performance Profiling (LPP)} -- a framework that derives task-agnostic diagnostics from hidden activations and output distributions. LPP defines a set of scalar metrics on a model's latent representations and dynamics, revealing scale-independent traits that enable interpretable comparisons and uncover hidden vulnerabilities. Unlike static accuracy scores, LPP provides stable, architecture-sensitive signatures across models of similar size. With extensive empirical analyses across eight LLMs, spanning a size range of 0.5B-14B, we demonstrate that models with similar benchmark scores can exhibit contrasting latent profiles, such as differences in entropy or adaptability. Guided by these insights, we design synthetic probes for uncertainty and symbolic reasoning that align with intrinsic metrics while decoupling from leaderboard bias. We recommend that reporting LPP alongside benchmarks provides a deeper, interpretable understanding of model behavior, enabling more reliable model selection, safety assessment, and evaluation beyond surface-level accuracy.
>
---
#### [new 065] DySem: Uncovering Dynamic Semantic Components via Multilingual Consensus for Calculating Semantic Textual Similarity
- **分类: cs.CL**

- **简介: 该论文属于语义文本相似度计算任务，针对现有方法依赖静态表示空间的问题，提出DySem框架，通过多语言共识提取动态语义组件，提升相似度计算效果。**

- **链接: [https://arxiv.org/pdf/2605.29751](https://arxiv.org/pdf/2605.29751)**

> **作者:** Kaijie Zheng; Weiqin Wang; Yile Wang; Hui Huang
>
> **备注:** 18 pages, 23 figures, 5 tables
>
> **摘要:** Calculating semantic textual similarity is a foundational task in natural language processing. Current large language models (LLMs) based methods typically rely on extracting last-layer hidden states with fixed dimensions to compute similarity for every text pairs. We argue that this paradigm is suffer from two limitations: (i) The last hidden layer encodes more general knowledge rather than just semantic knowledge, making it suboptimal for semantic similarity computation; (ii) The hidden layer dimensions of LLMs are generally very large, which introduces some redundancy and noise for representing semantics. In this work, we propose DySem, a novel training-free framework that investigates more semantic-related internal components of LLMs via multilingual consensus, and shifts away from static representation spaces in favor of dynamic, sample-specific semantic dimensions by constructing text-dependent joint semantic set and computes similarity over this shared dimensional subset. Extensive experiments across various LLMs show that our method consistently outperforms recent baselines while maintaining lower dimensions for similarity calculation. The code is released at this https URL.
>
---
#### [new 066] Evaluating Cross-lingual Knowledge Consistency in Code-Mixed vis-a-vis Indian Languages using IndicKLAR
- **分类: cs.CL**

- **简介: 该论文属于跨语言知识一致性研究任务，旨在解决印度语言及混码语言在大模型中的知识回忆差异问题。通过构建IndiKLAR基准，分析不同语言输入下的性能差异，并探索有效提示策略。**

- **链接: [https://arxiv.org/pdf/2605.29637](https://arxiv.org/pdf/2605.29637)**

> **作者:** Debajyoti Mazumder; Divyansh Pathak; Prashant Kodali; Aditya Joshi; Akshay Agarwal; Jasabanta Patro
>
> **备注:** 23 pages
>
> **摘要:** Large language models recall knowledge reliably in English but often fail on the same query posed in a lower-resourced language -- a crosslingual consistency gap that remains underexplored for Indian languages and their code-mixed counterparts. To study this gap, we introduce IndiKLAR, an Indic extension of the KLAR-CLC benchmark covering 18 of the 22 scheduled Indian languages and pairing them with code-mixed variants for 11 widely used language pairs, with native-speaker verification of both monolingual and code-mixed variants for these 11 settings. This three-way alignment offers a unique opportunity to examine how knowledge recall consistency varies across the spectrum of English, code-mixed, and native Indian language inputs. Evaluating across nine open-weight models, we find that the native-language accuracy gap to English can reach $\sim$0.50, while code-mixed inputs close most of it -- bringing performance within $\sim$0.05 of English without any model-level intervention. Motivated by this, we evaluate several prompting strategies that vary in how language conversion is exposed, including a two-stage translate-then-answer setup, a one-stage joint translation-and-answer prompt, and Translate-in-Thought (TinT) -- a single-step strategy in which the model converts the input internally and emits only the final answer. Across the performance trajectory native $\rightarrow$ code-mixed $\rightarrow$ English, we identify a consistent flip point -- the boundary between incorrect and correct prediction -- that lies between the native and code-mixed settings. Interestingly, this holds whether the trajectory is induced by the input surface form or by the model's internal conversion process.
>
---
#### [new 067] Who Am I? History-Aware Profiles for Student Simulation in Tutoring Dialogues
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于学生模拟任务，旨在解决对话中缺乏学生历史信息的问题。通过构建历史感知的用户画像并结合强化学习，提升模拟学生对话的准确性。**

- **链接: [https://arxiv.org/pdf/2605.30051](https://arxiv.org/pdf/2605.30051)**

> **作者:** Zhangqi Duan; Shuyan Huang; Alexander Scarlatos; Jaewook Lee; Simon Woodhead; Andrew Lan
>
> **摘要:** A key part of developing large language model (LLM)-powered, automated tutoring tools is student simulation, i.e., using LLMs to role-play as students, which can facilitate tutor model evaluation and training. Existing work mostly focuses on within-dialogue simulation, which lacks context on student knowledge and behavior, partly due to not grounding in past student question-answering or dialogue interactions. In this work, we introduce the task of history-conditioned student simulation, where the goal is to accurately predict student dialogue turns by leveraging information in the student's learning history. We propose a two-component framework in which a profile generator summarizes a student's history and a simulator predicts student turns conditioned on the resulting profile. We train both components with reinforcement learning (RL), yielding profiles optimized for faithful student simulation. We evaluate our method and baselines on the first-of-its-kind real-world dataset of student dialogues and question responses that we collect from a math learning platform. Extensive experiments show that our method significantly outperforms baselines, and demonstrate the importance of history, profiles, and RL training.
>
---
#### [new 068] AfriScience-MT: Towards Decolonizing Science in Africa through Text Translation
- **分类: cs.CL**

- **简介: 该论文属于机器翻译任务，旨在解决非洲语言科学术语缺失问题，通过构建平行语料库并评估翻译模型性能。**

- **链接: [https://arxiv.org/pdf/2605.29741](https://arxiv.org/pdf/2605.29741)**

> **作者:** Idris Abdulmumin; Tajuddeen Gwadabe; Shamsuddeen Hassan Muhammad; David Ifeoluwa Adelani; Nomonde Khalo; Ibrahim Said Ahmad; Abiodun Modupe; Anina Mumm; Sibusiso Biyela; Michelle Rabie; Johanna Havemann; Marek Rei; Jade Abbott; Vukosi Marivate
>
> **摘要:** The dominance of colonial languages in African education and scientific communication limits how hundreds of millions of speakers of African languages access and produce scientific knowledge. A core obstacle is the lack of established scientific terminology in these languages. We introduce AfriScience-MT, a parallel corpus covering six African languages (Amharic, Hausa, Luganda, Northern Sotho, Yorùbá, and isiZulu) across 11 scientific domains. Professional translators, working with expert science communicators, translated plain-language summaries of scientific papers into each target language and created new terms where none existed. We benchmark machine translation systems and large language models in zero-shot, few-shot, and fine-tuned settings. Our results show that closed-source models outperform all open-source models at both the sentence and document levels: GPT-5.4 and Gemini-3.1-Flash-Lite lead with average sentence-level COMET scores of 68.3 and 68.0, respectively, and tie at an average document-level COMET of 48.3. Among open systems, fine-tuned NLLB-1.3B reaches 67.3 at the sentence level, and TranslateGemma-12B reaches 44.0 at the document level with 1-shot in-context learning. We release AfriScience-MT to support benchmarking and document-level scientific MT for African languages.
>
---
#### [new 069] What are They Thinking? Delineation, Probing and Tracking of Concepts in LLMs
- **分类: cs.CL**

- **简介: 该论文属于模型解释任务，旨在解决如何检测和追踪LLM中的概念问题。通过构建数据集并训练线性探测器，实现对概念的识别与跟踪。**

- **链接: [https://arxiv.org/pdf/2605.28823](https://arxiv.org/pdf/2605.28823)**

> **作者:** Mohamed Abdelwahab; Michelle Yu Collins; Sihan Chen; Yi Cheng Zhao; Zafarullah Mahmood; Jiading Zhu; Soliman Ali; Jonathan Rose
>
> **摘要:** As the influence of LLMs expands, it is imperative to gain insight into their decisions. One way to do that is to develop probes that detect the presence or absence of a broad set of concepts within the embeddings computed in an LLM - which is what we might say a model is "thinking" about. Such probes should be low-cost and easily applicable to any LLM, so that monitoring for many concepts is possible during normal operation. In this paper, we take the first steps towards developing the capability of creating many such probes by defining and executing examples of the key tasks needed: first, the careful delineation of a concept through the creation of a dataset with the concept both present and then absent. Then, the training and testing of a set of linear probes to detect the concept on any layer of an LLM, including an exploration of the complexity of the probe needed. Finally, we show that such probes can track concepts across larger contexts. This is done with four separate concepts and three different LLMs. When this process is scaled to many more concepts, it will create the ability to easily monitor new models.
>
---
#### [new 070] From Context Shift to Stylistic Collapse: Why Training Objectives Matter More Than Scale
- **分类: cs.CL**

- **简介: 该论文属于语言模型分析任务，探讨训练目标对语言风格的影响。研究发现，现代大模型在训练中重塑语言特征，导致风格概率分布变化，揭示了对齐管道的结构性问题。**

- **链接: [https://arxiv.org/pdf/2605.28826](https://arxiv.org/pdf/2605.28826)**

> **作者:** Rohan Mahapatra
>
> **备注:** 26 pages, 13 tables, 2 figures. Planning to submit to NeurIPS 2026
>
> **摘要:** In modern LLMs, linguistic features function not as stylistic artifacts but as probes of probability mass, allocated under training alignment objectives. Language models trained with contemporary pipelines exhibit severe reshaping of linguistic features, leading to extreme language re-distribution. While previous stylometric analyses explored linguistic differences between AI-generated and human texts, we focus on the reshaping plaguing the LLM training pipeline itself. We analyze 17 models (410M-100B+ parameters) across 24 linguistically-motivated probes, documenting that instruction-tuned systems systematically collapse language entropy along discourse and structural dimensions (mean amplification: 1,949-16,853%, peaks: 5,181-209,675%), while selectively suppressing complex punctuation to 3.2-23.2% of baseline frequencies. These effects do not worsen under RLHF, as divergence patterns are statistically indistinguishable (p > 0.25) across matched base and instruction-tuned model pairs. Weak intervention (lambda=1.0) exacerbates collapse by 240%, while strong control (lambda=5.0) achieves 40.5% improvement and outperforms frontier models by 96.7-98.2% despite 200-1000x scale disadvantage. Additionally, lambda=5.0 delivers 15% higher distinct-4, 27% higher vocabulary diversity, and 78% lower repetition than moderate regularization, establishing that alignment requires sufficient control strength, not merely distributional smoothing. Our findings underscore how modern LLMs reallocate stylistic probability mass, despite RLHF and scale. More broadly, our work reveals a structural limitation of current alignment pipelines: preference optimization reshapes language distributions invisible to standard quality metrics yet detectable through distributional probes, with implications for AI detection, training data contamination, and long-term linguistic evolution.
>
---
#### [new 071] MusTBENCH: Benchmarking and Advancing Temporal Grounding in Music LLMs
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属于音乐理解任务，旨在解决LALMs在时间定位上的不足。提出MusTBENCH基准和MusT优化方法，提升模型的时间对齐能力。**

- **链接: [https://arxiv.org/pdf/2605.29300](https://arxiv.org/pdf/2605.29300)**

> **作者:** Daeyong Kwon; Qiyu Wu; Shinobu Kuriya; Junghyun Koo; Shuyang Cui; Zhi Zhong; Wei-Hsiang Liao; Hiromi Wakaki; Yuki Mitsufuji
>
> **摘要:** Recent Large Audio-Language Models (LALMs) have demonstrated promising abilities in understanding musical content. However, whether their responses are grounded in the correct temporal regions of the audio remains underexplored. This limitation is particularly critical for music understanding, where key information often occurs as temporally localized events, such as instrument entries and rhythmic transitions. To address this gap, we introduce MusTBENCH, a music-expert-validated benchmark designed to evaluate temporal grounding in LALMs through five temporally grounded question-answering tasks. To further improve temporal grounding in existing models, we propose MusT, a novel four-stage temporal optimization recipe spanning music encoder adaptation, LLM adaptation, LLM supervised fine-tuning, and RL-based optimization. Experiments on MusTBENCH show that existing LALMs struggle with precise temporal grounding, while MusT brings significant improvements over strong baselines. These results establish temporal grounding as a key missing capability in current LALMs and position MusTBENCH as a challenging benchmark for future research in temporally grounded music understanding.
>
---
#### [new 072] Recovering Diversity Without Losing Alignment: A DPO Recipe for Post-Trained LLMs
- **分类: cs.CL**

- **简介: 该论文属于大模型优化任务，旨在解决后训练导致输出多样性下降的问题。通过构建偏好数据提升模型多样性，同时保持对齐效果。**

- **链接: [https://arxiv.org/pdf/2605.30021](https://arxiv.org/pdf/2605.30021)**

> **作者:** Vinay Samuel; Yapei Chang; Mohit Iyyer
>
> **备注:** Under Review. 26 pages, 3 figures, 16 tables
>
> **摘要:** Many open-ended instructions have multiple valid answers that users can benefit from seeing, but post-training often narrows an LLM's output space toward a small set of canonical responses. We introduce REDIPO, an offline DPO data-construction pipeline for recovering distinct valid answer modes while preserving the alignment benefits of the instruct model. For each prompt, REDIPO samples responses from both base and instruct models, rewrites base-model responses with the instruct model, filters candidates for safety and instruction-following quality, and builds preference pairs that favor marginally diverse responses among candidates with similar instruction-following reward. Across Qwen3-4B, OLMo-3-7B, and LLaMA-3.1-8B, REDIPO improves NoveltyBench distinct_k by 134%, 33%, and 44% relative to the instruct checkpoints, while DivPO changes diversity by 0%, -6%, and -4% on the same models. These gains largely maintain MTBench, IFEval, and Arena-Hard performance, and reduce direct-category HarmBench attack success rate. Ablations show that marginal-diversity pair selection and base-response rewriting drive the diversity gains, while filtering and quality-bounded pairing help maintain alignment. Overall, our results show that diverse valid answers from base-model generations can be reintroduced through carefully constructed preference data while retaining the alignment benefits of post-training. We release our code and data at this https URL.
>
---
#### [new 073] Beyond Recall: Behavioral Specification as an Interpretive Layer for AI Personalization
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文属于AI个性化任务，解决用户与AI对齐问题。通过构建行为规范作为解释层，提升系统对用户意图的准确表示，优化模型预测效果。**

- **链接: [https://arxiv.org/pdf/2605.28969](https://arxiv.org/pdf/2605.28969)**

> **作者:** Aarik Gulaya
>
> **备注:** 134 pages, 4 figures. Code, data, judge prompts, and reproduction instructions: this http URL
>
> **摘要:** If an AI agent makes decisions on a person's behalf, those decisions must align with its user. We introduce representational accuracy to measure how faithfully a system captures a person's interpretation. An interpretive layer is operationalized as a Behavioral Specification. Our reference implementation aggressively compresses a person's data into interpretive patterns, served as context to a language model. We evaluate the Specification on a prototype benchmark of held-out behavioral predictions scored by a calibrated 5-judge LLM panel. We test it independently and in composition with a range of context conditions: full raw corpus, full extracted facts, and four commercial memory systems (Mem0, Letta, Supermemory, Zep). Across 14 public-domain autobiographical corpora, the Specification lifts representational accuracy in aggregate and nearly eliminates model hedging. It recovers most of what the raw corpus delivers, at ~25x less context cost. The Specification lifts subjects toward a common predictive level regardless of pretraining baseline; the lift in absolute points is therefore largest where the baseline is lowest, suggesting the population of relevance is anyone not adequately represented in pretraining. Lift is greatest on interpretation-required questions, where providing an interpretive layer enables model behavior that extracted facts or raw corpus do not. Conversely, on recall-required questions, this layer can interfere rather than help. We conclude that representational accuracy is distinct from recall and that human-AI alignment is dependent on how accurately the user is represented. Representational accuracy makes that alignment testable.
>
---
#### [new 074] GAPD: Gold-Action Policy Distillation for Agentic Reinforcement Learning in Knowledge Base Question Answering
- **分类: cs.CL**

- **简介: 该论文针对知识库问答中的强化学习任务，解决中间动作错误监督不足的问题，提出GAPD框架通过金标准动作引导学生策略，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.29584](https://arxiv.org/pdf/2605.29584)**

> **作者:** Xin Sun; Jianan Xie; Zhongqi Chen; Qiang Liu; Shu Wu; Bowen Song; Weiqiang Wang; Zilei Wang; Liang Wang
>
> **摘要:** Reinforcement learning (RL) is a natural fit for agentic knowledge base question answering (KBQA), where a model must issue executable actions, observe knowledge-base feedback, and eventually return an answer. However, current RL-based KBQA systems mainly optimize sparse rewards from the final answer, leaving intermediate action errors weakly supervised. This is especially limiting for logical-form annotated KBQA benchmarks: gold logical forms can be converted into executable action sequences, but existing pipelines use them mainly for warm-start data construction rather than for on-policy RL updates. We propose GAPD, a training-time Gold-Action Policy Distillation framework that adds dense token-level guidance to outcome-based RL. To align gold actions with on-policy student rollouts, GAPD uses MID-ANCHOR MATCHING: it treats the intermediate entities reached during student exploration and gold execution as state anchors, and matches student states to gold states through these explored entity sets. The current policy conditioned on this aligned gold action serves as a stop-gradient teacher, whose token distribution is distilled back to the ordinary student policy over generated action-token spans. GAPD consistently surpasses the current state of the art on WebQSP, GrailQA, and GraphQ.
>
---
#### [new 075] Compute Allocation in Evolutionary Search: From Depth-Breadth to Multi-Armed Bandits
- **分类: cs.CL; cs.AI; cs.LG; cs.NE**

- **简介: 该论文研究LLM引导的进化搜索中的计算资源分配问题，旨在提升搜索效率与可靠性。通过分析不同模型和任务下的性能，提出BaSE算法优化LLM调用分配。**

- **链接: [https://arxiv.org/pdf/2605.29268](https://arxiv.org/pdf/2605.29268)**

> **作者:** Sixue Xing; Haoyu He; Kerui Wu; Zhuo Yang; Haozheng Luo; Tianfan Fu; Aarthy Nagarajan
>
> **摘要:** LLM-guided evolutionary search (Evolve systems) has reached state-of-the-art results on mathematical and combinatorial tasks, yet most existing systems report only the best of many runs and leave the run-to-run distribution undocumented. We ask how a fixed budget of LLM calls should be allocated, and how reliably a single run reaches the reported numbers. Sweeping the depth-breadth grid over five models and three tasks, we identify two empirical regularities: a fitness-compute envelope along which capability ordering largely collapses on effective FLOPs, and a bilinear depth-breadth fit with task-specific interaction; both are gated by model-task capability. Motivated by these regularities, we propose BaSE (Bandit-based Self-Evolving), a multi-armed bandit that allocates LLM calls across parallel trajectories. Without changing the model, prompt, or evaluator, BaSE improves mean fitness by 12.3% over the strongest island-protocol baseline across 8 (model, task) cells, with the largest gains on high-variance settings: a reliability gain from allocation alone.
>
---
#### [new 076] Source-Grounded Semantic Reinforcement Learning for Low-Resource Target-Language Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于低资源目标语言生成任务，解决平行数据稀缺问题。通过SG-SRL框架，利用源语言单语数据提供跨语言语义监督，提升生成质量。**

- **链接: [https://arxiv.org/pdf/2605.29502](https://arxiv.org/pdf/2605.29502)**

> **作者:** Zeli Su; Ziyin Zhang; Zewei Pan; Zhou Liu; Dingcheng Huang; Dehan Li; Zhankai Xu; Longfei Zheng; Xiaolu Zhang; Jun Zhou; Wentao Zhang
>
> **摘要:** Low-resource target-language generation is often limited by scarce parallel data, while high-resource source-language monolingual data is abundant but difficult to use with standard supervised fine-tuning. We propose Source-Grounded Semantic Reinforcement Learning (SG-SRL), a resource-utilization framework that converts source-language monolingual data into cross-lingual semantic supervision for target-language generation. SG-SRL performs reference-free reinforcement learning (RL) on source-language data using a cross-lingual semantic reward model, instantiated by a cross-lingual reranker that scores the semantic relevance between the source input and the target-language generation. While this induces severe verbosity-based reward hacking, a lightweight recovery stage using a small parallel corpus restores fluency, conciseness, and task format while preserving the semantic gains. Experiments on Chinese-to-Thai generation show that SG-SRL improves semantic grounding and factual coverage over cold-start SFT. Additional analyses on long-form transfer and Tibetan embedding-based rewards clarify the generalization behavior of SG-SRL and show that an encoder-based semantic reward can substitute for an LLM-based reranker in a realistic low-resource language setting.
>
---
#### [new 077] Reasoning that Travels: Dissecting How Chain-of-Thought Transfers Across Models
- **分类: cs.CL**

- **简介: 该论文研究跨模型链式思维（CoT）传递机制，探讨如何通过提供推理轨迹提升其他模型的推理能力，解决CoT有效传递与利用的问题。**

- **链接: [https://arxiv.org/pdf/2605.28913](https://arxiv.org/pdf/2605.28913)**

> **作者:** Xinyuan Cheng; Beiduo Chen; Philipp Mondorf; Barbara Plank
>
> **备注:** 20 pages, 17 figures
>
> **摘要:** Large reasoning models (LRMs) often generate extensive chain-of-thought (CoT) traces before producing a final answer. As explicit textual artifacts, these traces can be passed to other models to solve the same task, enabling cross-model reasoning transfer. Yet successful transfer alone does not reveal how the provided CoT contributes to another model's answer. We study this question with a controlled provider--receiver framework, where a provider generates a reasoning trace and a receiver solves the same problem from increasingly longer trace prefixes. We compare force-answer, where the receiver answers directly from the prefix, with free-generation, where it may continue reasoning before answering. Across models and benchmarks, full traces often transfer successfully, but prefix trajectories reveal distinct mechanisms. In force-answer mode, AIME transfer is largely driven by explicit answer availability. MMLU-Pro instead reflects a larger role for receiver competence, while ZebraLogic depends on partial structured-answer information rather than complete-answer leakage alone. In free-generation mode, partial CoTs improve performance across benchmarks, indicating that prefixes can guide continued reasoning. Finally, answer agreement among receivers provides a gold-free signal for stopping provider reasoning early. Overall, cross-model CoT transfer is not a single phenomenon: it can reflect answer extraction, reasoning scaffolding, or receiver-dependent competence.
>
---
#### [new 078] RightNow-Arabic-0.5B-Turbo: An Open Sub-1B Arabic Language Model via Vocabulary Injection and Edge-First Deployment
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出一个518M参数的阿拉伯语专用语言模型RightNow-Arabic-0.5B-Turbo，解决小规模阿拉伯语模型性能不足的问题。通过词汇注入和优化训练，提升模型效果，并实现边缘部署。**

- **链接: [https://arxiv.org/pdf/2605.28827](https://arxiv.org/pdf/2605.28827)**

> **作者:** Jaber Jaber; Osama Jaber
>
> **备注:** 12 pages, 7 tables, 4 figures, 1 algorithm. Weights: this https URL
>
> **摘要:** Open Arabic large language models split into two classes: sub-1B multilingual models that treat Arabic as an afterthought (Qwen2.5-0.5B, Falcon-H1-0.5B), and 7B-70B Arabic-specialized models that require a server to run (Jais, AceGPT, ALLaM, SILMA). The one published attempt at a sub-2B Arabic-specialized model, Kuwain-1.5B, never released its weights. We present RightNow-Arabic-0.5B-Turbo, a 518M-parameter Arabic-specialized decoder LLM built on Qwen2.5-0.5B. The pipeline adds 27,032 Arabic tokens via mean-subtoken initialization, continues pretraining on 504M Arabic tokens on 8xH100 with FSDP, FlashAttention varlen packing, and Liger fused kernels, then applies supervised fine-tuning on 129,116 Arabic instruction pairs with response-only loss masking, direct preference optimization on 6,750 Arabic preference pairs, and weight soup merging across three checkpoints. On three lm-evaluation-harness Arabic benchmarks (COPA-ar, Arabic HellaSwag, ArabicMMLU) the merged model reaches 35.9% mean accuracy, beats every same-class open model, ties Falcon-H1-1.5B on COPA-ar (58.4%) at one-third the size, and recovers 67% of SILMA-9B's mean at 1/18 the parameters. The edge build quantizes to 398 MB (q4_k_m) and delivers 635 tokens/s at batch size 1 on a single H100 via this http URL. All code (5,555 lines across 25 scripts), weights (bf16, int8, and four GGUF quantizations), and benchmark scripts are released at this https URL.
>
---
#### [new 079] Hallucination Detection-Guided Preference Optimization for Clinical Summarization
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于临床摘要任务，旨在解决大语言模型生成摘要时出现的幻觉问题。通过引入幻觉检测引导的优化方法，减少错误陈述，提升摘要的准确性与可靠性。**

- **链接: [https://arxiv.org/pdf/2605.28910](https://arxiv.org/pdf/2605.28910)**

> **作者:** Shamanth Kuthpadi Seethakantha; Dung Ngoc Thai; Vara Prasad Gudi; Simran Tiwari; Rami Matar; Avijit Mitra; Wenlong Zhao; Wael Salloum; Andrew McCallum
>
> **摘要:** Large language models (LLMs) have shown promise on summarization tasks, but they often produce hallucinations, which are unsupported or incorrect statements that limit their reliability in specialized healthcare applications. We introduce \itermodelfull (\itermodel), an inference-time method that leverages hallucination detectors to guide iterative summary revisions toward factual corrections. Building on this, we propose \itermodel for Preference Learning (\model), which converts detector-guided refinement trajectories into preference pairs for model finetuning. Extensive experiments show that our methods substantially reduce hallucinations for Llama and Gemma models in summarizing real-world clinical notes from \MimicIV. For example, \itermodel reduces 24\% and \model reduces 48\% hallucinations in Llama-3.1-8B-Instruct. Importantly, both methods preserve summary fluency, coherence, and relevance according to human expert and LLM-Jury evaluations. Together, these results demonstrate that detection-informed refinement and preference learning offer an automated solution for improving factual faithfulness in clinical summarization.
>
---
#### [new 080] Large language models reorganize representational geometry during in-context learning
- **分类: cs.CL; cs.LG; q-bio.NC**

- **简介: 该论文研究预训练大语言模型的上下文学习机制，探讨其表征空间几何对任务分类的影响，旨在揭示ICL的有效性与表征结构的关系。**

- **链接: [https://arxiv.org/pdf/2605.28854](https://arxiv.org/pdf/2605.28854)**

> **作者:** Hua-Dong Xiong; Li Ji-An; Robert C. Wilson; Kwonjoon Lee; Xue-Xin Wei
>
> **摘要:** Large language models (LLMs) exhibit remarkable flexibility: they can adapt to novel tasks from in-context examples without any parameter updates, a capability known as in-context learning (ICL). Prior work on synthetic tasks has shown that ICL can implement specific algorithms, demonstrating architectural competence, and mechanistic analyses have identified key circuits that support this behavior. However, because in-context computation -- regardless of its algorithmic form -- relies on transformations in high-dimensional representation space, it remains unclear how the geometry of that space shapes ICL effectiveness. Motivated by the neuroscience view of classification as the untangling of neural representations, we hypothesize that ICL depends on the successful online untangling of task-relevant representations. To test this idea, we study how LLMs classify in-context examples whose labels are defined by the model's own internal representations with known structure. We show that ICL performance correlates systematically with the representational structure of the underlying classification task and that successful ICL is accompanied by geometric reorganization that increases online separability. We further find that LLM behavior is well described by a prototype-like algorithm that integrates evidence while reshaping representations to support classification. These findings offer a geometric account of ICL in pretrained LLMs, establish representational geometry as a mechanistic constraint on ICL, and quantify the gap between what pretrained representations afford and what in-context learning can exploit.
>
---
#### [new 081] Same Evidence, Different Answers: Canonical-Context On-Policy Distillation for Multi-Turn Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，解决多轮对话中模型因信息逐步输入而产生偏差的问题。提出CCOPD方法，通过师生训练对齐行为，提升模型在分段信息下的表现。**

- **链接: [https://arxiv.org/pdf/2605.30251](https://arxiv.org/pdf/2605.30251)**

> **作者:** Zizhuo Lin; Quanling Liu; Jinsheng Quan; Chao Zhang; Yifan Zhu; Xing Shi; Jingtao Xu; Zhihui Li; Yawei Luo
>
> **摘要:** Large language models (LLMs) often solve a task when all instructions are given in a single prompt, but fail when the same information is revealed gradually across turns. When a clean FULL prompt and a RAW-SHARDED conversation contain the same complete user evidence, the model should still arrive at the same answer. We argue that a key reason for this gap is self-anchored drift: responses produced under partial information introduce unsupported assumptions, and those assumptions later distort the final answer. To reduce this effect, we propose Canonical-Context On-Policy Distillation (CCOPD). During training, the same base model is used in two roles: a frozen teacher conditioned on the clean FULL prompt and a trainable student that receives the same evidence incrementally through a multi-turn conversation; CCOPD aligns the student's behavior on its own trajectories with the teacher's canonical full-context behavior. Trained only on math problem conversations, CCOPD yields a 32\% average relative improvement in RAW-SHARDED performance over the original base model across math and five zero-shot out-of-domain task families, while largely preserving full-context performance. Further analyses suggest that CCOPD strengthens grounding in user evidence and reduces sensitivity to contamination from earlier assistant turns.
>
---
#### [new 082] EviLink: Multi-Path Schema Linking with Uncertainty-Guided Evidence Acquisition for Large-Scale Text-to-SQL
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于Text-to-SQL任务，解决多路径schema链接中的不确定性问题。提出EviLink方法，结合多假设和不确定性引导的证据获取，提升链接效果。**

- **链接: [https://arxiv.org/pdf/2605.29670](https://arxiv.org/pdf/2605.29670)**

> **作者:** Huawei Zheng; Sen Yang; Zhaorui Yang; Yuhui Zhang; Haozhe Feng; Haoxuan Li; Xuan Yi; Chao Hu; Defeng Xie; Chen Hou; Danqing Huang; Wei Chen; Yingcai Wu; Peng Chen; Dazhen Deng
>
> **摘要:** Schema linking is a difficult and important step in large-scale Text-to-SQL, where systems must identify a compact yet sufficient schema context from large and ambiguous databases. Existing methods often treat schema linking as deterministic selection around a single SQL path, but complex questions may admit multiple valid realizations with different schema needs. We reframe schema linking as uncertainty-aware schema-need inference over multiple plausible SQL paths, where the system distinguishes required schema items from path-dependent uncertain ones and acquires evidence only where needed. We instantiate this reframing with EviLink, which combines multi-hypothesis schema grounding with uncertainty-guided evidence acquisition. Experiments on BIRD-Dev and Spider2-Snow show that this perspective improves the balance among schema completeness, schema relevance, and token cost. On Spider2-Snow, EviLink achieves 90.15% field-level strict recall rate, uses 123.30K average tokens, and improves downstream SQL generation under a fixed generator.
>
---
#### [new 083] Towards Verifiable Multimodal Deep Research: A Multi-Agent Harness for Interleaved Report Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模态深度研究任务，旨在解决生成可验证的多模态报告问题。提出Ptah系统，通过多代理协作实现图文一致的报告生成与验证。**

- **链接: [https://arxiv.org/pdf/2605.29861](https://arxiv.org/pdf/2605.29861)**

> **作者:** Chenghao Zhang; Guanting Dong; Yufan Liu; Tong Zhao; Zhicheng Dou
>
> **摘要:** Large Language Models (LLMs) have advanced autonomous agents from deep search, which retrieves concise factual answers, to deep research, which synthesizes scattered evidence into long-form reports. However, verifiable multimodal deep research remains challenging due to open-ended synthesis without deterministic ground truth and the need to interleave textual arguments with visual evidence. We propose \textsc{Ptah}, a multi-agent harness for interleaved report generation. \textsc{Ptah} orchestrates the lifecycle from user query to rendered web report through planning, research, and writing stages, where specialized agents construct visual-aware plans, collect claim-grounded evidence, maintain source-aligned images in a \textit{Visual Working Memory}, and compose reports through declarative multimodal tool use. A verifier agent serves as the harness's acceptance function, enforcing factual grounding, citation fidelity, and cross-modal consistency throughout the workflow. We further introduce \textsc{Ptah}Eval, an evaluation protocol that augments existing benchmarks with image-level and presentation-level assessments. Experiments on deep research benchmarks show that \textsc{Ptah} produces more reliable, visually informative, and usable human-facing multimodal reports than strong baselines.
>
---
#### [new 084] CRITIC-R1: Learning Structured Critics for Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识密集型问答任务，旨在解决RAG方法中的幻觉和推理错误问题。提出CRITIC-R1框架，通过结构化批评和强化学习提升生成答案质量。**

- **链接: [https://arxiv.org/pdf/2605.29886](https://arxiv.org/pdf/2605.29886)**

> **作者:** Wenhan Xiao; Ziwei Zhang; Chuanyue Yu; Xingcheng Fu; Qingyun Sun; Runhua Xu; Jianxin Li
>
> **备注:** 17 pages,13 figures
>
> **摘要:** Retrieval-augmented generation (RAG) improves knowledge-intensive question answering by incorporating external evidence. However, existing RAG methods still suffer from hallucinations and subtle reasoning errors. Recent studies introduce external critics to refine RAG outputs, yet they often provide coarse-grained and weakly structured feedback, exhibit over-aggressive intervention, and lead to noisy and unreliable refinement, limiting their effectiveness for correction. To tackle these issues, we propose CRITIC-R1, a structured critic framework that formulates and learns RAG critique as an explicit error diagnosis problem using reinforcement learning (RL). Our framework categorizes common RAG errors into multiple diagnostic dimensions, including verdict, error location, reasoning analysis, and fix generation. To learn these capabilities, we design two reward functions: Conservative Judgement Alignment (CJA) first encourages calibrated high-level judgements while mitigating the over-aggressive phenomenon, whereas Diagnostic Quality Alignment (DQA) further improves fine-grained diagnostic feedback through gated rewards. We train the critic model using GRPO-based RL with process-level supervision collected from external LLM teacher models. Experiments across five QA benchmarks show that CRITIC-R1 consistently improves answer quality over strong RAG baselines. Our source code is available at this https URL
>
---
#### [new 085] Resolution Diagnostics for Paired LLM Evaluation
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于大模型评估任务，解决配对比较统计显著性不足的问题。通过假设检验框架，提出分辨率诊断指标，分析现有工具的误差并验证其有效性。**

- **链接: [https://arxiv.org/pdf/2605.30315](https://arxiv.org/pdf/2605.30315)**

> **作者:** Anany Kotawala
>
> **备注:** 16 pages, 7 figures, 12 tables. Accepted to the ICML 2026 Workshop on Hypothesis Testing, Seoul, South Korea, 2026. Copyright 2026 by the author(s)
>
> **摘要:** Across two public LLM leaderboards, many displayed pairwise rankings do not meet a conventional paired-test resolution target under the actual paired evaluation design: 11 of 40 Open LLM Leaderboard v1 pairwise comparisons and 4 of 9 MMLU-Pro top-10 adjacent-rank pairs are unresolved at (alpha, 1-beta) = (0.05, 0.8). The MMLU-Pro count rises to 6/9 under real subject-level clustering and stays at 5-6 out of 9 in 99.9% of category-bootstrap resamples. We frame paired LLM evaluation as a hypothesis-testing problem, invert level-alpha, power-(1-beta) tests, and report a per-pair resolution ratio q = N/N* as the primary diagnostic. A sharp small-effect expansion with an explicit second-order constant shows that the widely-used unpaired Cohen-h-plus-(1-rho) shortcut deviates from the correct N* by approximately a factor of two in the close-comparison regime, a deficit that three of five off-the-shelf calculators(Cohen 1988, G*Power, R pwr) silently inherit when the user post-multiplies their per-arm output by (1-rho). The unresolved-pair pattern remains under multiplicity correction and anytime-valid sequential testing.
>
---
#### [new 086] UA-Legal-Bench: A Benchmark for Evaluating Large Language Models on Ukrainian Legal Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出UA-Legal-Bench，评估大语言模型在乌克兰法律推理中的表现，解决法律NLP基准不足的问题。**

- **链接: [https://arxiv.org/pdf/2605.29170](https://arxiv.org/pdf/2605.29170)**

> **作者:** Volodymyr Ovcharov
>
> **备注:** 13 pages, 5 figures, 4 tables. Data: this https URL
>
> **摘要:** Legal NLP benchmarks are overwhelmingly English-centric, leaving failure modes in morphologically rich, non-Latin-script languages undetected. We introduce UA-Legal-Bench, a five-task benchmark for evaluating large language models on Ukrainian legal reasoning, built from the Unified State Register of Court Decisions (EDRSR) -- one of the world's largest open judicial corpora (99.5 million decisions). The benchmark comprises: (1) case-type classification (4 classes, n=2,000), (2) judgment form classification (4 classes, n=2,000), (3) case-outcome prediction (6 classes, n=800), (4) legal norm extraction (n=1,794), and (5) cause category prediction (22 classes, n=1,871). We evaluate 11 LLMs (3B--675B) from five families under zero-shot and 3-shot prompting via AWS Bedrock with 158K API calls. Our results reveal sharply task-dependent few-shot effects: few-shot prompting improves judgment form classification by up to +38.6 pp but has mixed effects on outcome prediction. We show that accuracy is misleading on imbalanced legal tasks: the model with highest COP accuracy (62%) is a majority-class predictor (macro-F1: 23%), while the genuinely best model scores only 44% macro-F1. Within-family scaling analysis reveals that 8B models can match frontier performance on surface-level tasks but scaling thresholds vary dramatically across families. We release all data, prompts, and model predictions.
>
---
#### [new 087] Slogans or Stance? A Label-Light Diagnostic for Entrepreneurial-Discourse Measurement on Chinese SOE Speeches
- **分类: cs.CL**

- **简介: 该论文属于文本分析任务，旨在检验不同方法在测量企业领导言论中创业精神的一致性。通过实验比较多种方法的效果，提出一种标签轻量的诊断工具。**

- **链接: [https://arxiv.org/pdf/2605.29188](https://arxiv.org/pdf/2605.29188)**

> **作者:** Ting Gong; Shangquan Sun
>
> **备注:** 15 pages, 2 figures, 7 tables
>
> **摘要:** Dictionary methods, topic models, and embedding-similarity scorers are widely used in CSS and management research to measure constructs such as "entrepreneurial spirit" in corporate speeches. We contribute a label-light measurement diagnostic for such instruments rather than a new extraction model. On a corpus of 80 speeches by leaders of centrally administered Chinese state-owned enterprises, we exploit a natural experiment of 24 same-company different-speaker pairs and 5 same-company same-speaker pairs to test whether a method's per-document indices vary with leader identity holding firm constant. LDA fails (Cohen d=0.20, 95% CI [-0.72, 1.20]); a dictionary scorer reaches d=0.81 and a Chinese sentence encoder d=0.65 on doc-vector distances of order 10^-3. A zero-shot 9B open-weight LLM (Qwen3.5:9b) raises paired-contrast d to 1.09 (exact permutation p1=0.034). We downgrade three claims accordingly: gold F1 measures consistency with the LLM's own prompt rule rather than external construct recovery; doc-level style residualisation cuts the LLM's d to 0.43 (p1=0.22), so roughly half of the effect is consistent with leader idiolect; and a confidence-weighted calibration trades Delta for variance with an auto-mined slogan lexicon near-inert in ablation. We release the 2,190-segment scored corpus, the 170-paragraph pilot, the slogan lexicon, two-family LLM scores, and the evaluation harness.
>
---
#### [new 088] Learning Design Skills as Memory Policies for Agentic Photonic Inverse Design
- **分类: cs.CL**

- **简介: 该论文属于光子晶体光纤逆向设计任务，旨在解决传统方法难以积累可复用设计知识的问题。提出SkillPCF框架，结合记忆策略与强化学习，提升设计效率与质量。**

- **链接: [https://arxiv.org/pdf/2605.29421](https://arxiv.org/pdf/2605.29421)**

> **作者:** Shengchao Chen; Ting Shu; Sufen Ren
>
> **备注:** AI4Physics@ICML 2026
>
> **摘要:** Photonic crystal fiber (PCF) inverse design remains challenging because candidate geometries must satisfy coupled optical targets under expensive electromagnetic simulation. Existing pipelines improve surrogate prediction or one-shot parameter recommendation, but they do not accumulate reusable design knowledge across iterative trials. We formulate PCF inverse design as a memory-policy learning problem and propose SkillPCF, a closed-loop agent framework that combines a physics-guided memory skill bank, reinforcement-learned skill selection, and simulator-grounded skill evolution. We further construct a real-world dataset with 479 expert interaction traces (2,507 spans) and 553 memory-dependent evaluation queries covering dispersion engineering, loss optimization, and multi-objective design. Experiments across multiple LLM backbones and classical baselines show that SkillPCF achieves stronger design-quality and efficiency trade-offs under practical simulation budgets, demonstrating the effectiveness of our proposed memory-skill learning paradigm for physics-aware PCF inverse design.
>
---
#### [new 089] Wait! There's a Way Out: A Decision Mechanism for Forecasting Conversational Derailment
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于对话安全任务，旨在预测对话是否可能转向人身攻击。针对现有方法忽略未来恢复可能性导致误报率高的问题，提出一种基于前瞻性模拟的决策机制，有效降低误报率。**

- **链接: [https://arxiv.org/pdf/2605.29243](https://arxiv.org/pdf/2605.29243)**

> **作者:** Laerdon Kim; Vivian Nguyen; Cristian Danescu-Niculescu-Mizil
>
> **备注:** To appear in the Proceedings of ACL 2026
>
> **摘要:** Forecasting conversational derailment is the task of predicting, as the conversation unfolds, whether it will eventually derail into personal attacks. Since forecasting models operate in an online fashion, they must decide whether to "trigger" an alert after each utterance--for example, to notify participants or a moderator that the conversation is at risk of derailing. Existing approaches make this decision solely based on the estimated likelihood of derailment given the preceding utterances, implicitly assuming that the conversation's future trajectory is fixed. As a result, they ignore the possibility of future recovery and incur an unnecessarily high rate of false positives. In this work we propose a method for decoupling the decision to trigger from derailment likelihood estimation. Our approach is inspired by the first human baseline on this task, which shows that humans achieve dramatically lower false positive rates by selectively deferring their decision to trigger when they anticipate that tension is likely to subside. We operationalize this insight with a deferral mechanism that uses forward-looking simulations to assess whether a tense moment admits plausible paths to recovery. Incorporating this mechanism into a state-of-the-art forecasting model substantially reduces false positives without sacrificing forecasting accuracy. More broadly, this work highlights the value of treating decision-making as a first-class component of forecasting systems.
>
---
#### [new 090] ActTraitBench: Quantifying the Knowledge-Decision Gap in Large Language Models via Human-Grounded Behavioral Validation
- **分类: cs.CL**

- **简介: 该论文属于模型评估任务，旨在解决大语言模型在知识与决策间的一致性问题。通过构建ActTraitBench框架，量化模型行为与自我报告的差异，并提出CoCA方法提升对齐效果。**

- **链接: [https://arxiv.org/pdf/2605.29791](https://arxiv.org/pdf/2605.29791)**

> **作者:** Yutong Yang; Chenxi Miao; Weikang Li; Yunfang Wu
>
> **摘要:** While Large Language Models (LLMs) can convincingly simulate personas in explicit self-reports, they often deviate in implicit behavioral decisions, revealing a substantial Knowledge-Decision Gap ($G_{\text{KD}}$). Existing benchmarks struggle to measure this asymmetry due to limited construct validity, multi-dimensional entanglement, and distributional biases in LLM-based evaluation. To address these issues, we propose ActTraitBench, a human-grounded evaluation framework for measuring personality consistency in LLMs. Grounded in empirical human data, ActTraitBench establishes one-to-one mappings between psychometric facets and behavioral paradigms, and applies a Distributional Calibration via Quantile Mapping procedure to align LLM-judge score distributions with human norms. Experiments on 14 mainstream LLMs reveal a pervasive knowledge-decision asymmetry, where larger and more capable models often exhibit stronger behavioral divergence despite highly consistent self-reports. To mitigate this gap, we further introduce the Chain of Cognitive Alignment (CoCA), a plug-and-play inference-time intervention that improves alignment in reasoning-capable frontier models while exposing clear capability limitations in smaller architectures.
>
---
#### [new 091] Attention Asymmetry in AI Layoff Discourse on X: A Computational Analysis of Capital vs Labour Amplification
- **分类: cs.CL; cs.CY; cs.SI**

- **简介: 该论文研究X平台上资本与劳工话语的传播不对称性，通过分析763条推文，发现资本话语传播更广，且不完全由粉丝数决定。任务为平台 discourse 分析，解决传播不平等问题。**

- **链接: [https://arxiv.org/pdf/2605.29367](https://arxiv.org/pdf/2605.29367)**

> **作者:** Joy Bose
>
> **备注:** 18 pages, 3 figures, 9 tables
>
> **摘要:** When workers lose jobs to AI-driven restructuring, two very different conversations happen on X (formerly Twitter) at the same time. Tech executives and AI researchers talk about productivity, transformation, and opportunity. Laid-off workers and labour critics talk about job loss, uncertainty, and fear. This paper asks a simple question: which conversation gets more reach? We report three studies using two collection methods and 763 tweets from 20 named public accounts. Study 1 used keyword-based collection (n=392) and found no significant difference between corpora (p=0.891), revealing that keyword search is too noisy for this task. Study 2 used account-based collection (n=96) and found a 3.12x mean amplification advantage for capital discourse over labour discourse (p=0.000003, Cohen's d=0.555). Study 3 combined both methods (n=763) and confirmed the finding at 4.18x mean and 10.77x median amplification ratio (p<0.000001). Critically, after normalising for follower count, the asymmetry persists at 2.69x (p=0.000009, Cohen's d=0.491), demonstrating that the effect is not simply a consequence of capital accounts having larger audiences. The finding is robust across all tested amplification metric weightings. We introduce the Amplification Ratio and Amplification Normalisation Index as simple metrics for measuring platform-level discourse inequality. A cross-platform replication on Reddit (n=647 posts) did not replicate the finding, suggesting the asymmetry may be specific to X's account-based amplification architecture. We discuss the methodological implications for cross-platform discourse analysis.
>
---
#### [new 092] Mask the Target: A Plug-and-Play Regularizer Against LoRA Forgetting
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于模型微调任务，解决LoRA适应中因分布差异导致的遗忘问题。提出一种无需回放数据的输出空间正则化方法，提升模型在新分布下的性能同时保留原有能力。**

- **链接: [https://arxiv.org/pdf/2605.29498](https://arxiv.org/pdf/2605.29498)**

> **作者:** Runze Xu; Arpit Garg; Hemanth Saratchandran; Simon Lucey
>
> **备注:** In Submission
>
> **摘要:** Low-Rank Adaptation (LoRA) has become one of the most widely used fine-tuning mechanisms for adapting large language models to new domains, tasks, and users. Yet adaptation performance alone can obscure an important failure mode: LoRA updates may improve performance on the target distribution while degrading prior capabilities learned during pretraining and alignment. We show that this forgetting becomes especially severe when the adaptation distribution differs substantially from the models original training or alignment distributions. The challenge is amplified in practical settings, where the original training and alignment data are typically unavailable. Motivated by this constraint, we study how LoRA based adaptation balances new learning against forgetting in a replay-free setting, and introduce a simple output space regularizer that can be added directly to existing training pipelines. Our method removes the ground-truth token from both the base and adapted model distributions, renormalizes the remaining probabilities, and applies KL regularization only over the non-target vocabulary. This preserves the base models relative preferences among alternative tokens without directly opposing the cross-entropy signal required for adaptation. As the regularizer acts only at the loss level, it requires no replay data, architectural changes, adapter redesign, or inference-time overhead, and can be applied directly to existing LoRA variants. Across all LoRA variants tested and across various backbones, our method improves the frontier between new learning and forgetting when the adaptation distribution differs substantially from the base models original training or alignment distributions, suggesting a broadly applicable route toward more reliable LLM updating.
>
---
#### [new 093] Kronecker Embeddings: Byte-Level Structured Token Representations for Parameter-Efficient Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出Kronecker嵌入，用于参数高效语言模型，解决传统嵌入表参数过多问题，通过字节级因子化减少参数并提升鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.29459](https://arxiv.org/pdf/2605.29459)**

> **作者:** Rohan Shravan
>
> **备注:** 28 pages, 16 tables. Reference implementation: this https URL
>
> **摘要:** Large language models route every input through a learned embedding table of shape |V| x d_model, consuming hundreds of millions to billions of trainable parameters at frontier scale. We introduce Kronecker Embeddings, a deterministic byte-level character-position factorization that replaces this table with a fixed encoder and a single learned projection, compatible with standard BPE tokenizers, eliminating 91--94% of input-side trainable parameters at frontier scale. We provide five contributions. First, a cross-model probe across six LMs (135M-671B parameters) shows trained input embeddings cluster typographic variants of the probe word far more than morphological relatives; Kronecker escapes this clustering at the embedding layer. Second, a controlled three-seed comparison on nanoGPT GPT-2 124M over 2.5B tokens of FineWeb-Edu shows Kronecker reaching 2.5 +- 0.2% lower validation loss than the BPE-tied baseline (gap 0.083 +- 0.007 nats, ~9% lower perplexity), needing ~1.43x fewer steps to reach BPE's converged loss. Third, a spelling-robustness probe over 110 clean/typo pairs shows Kronecker preserves the top-1 prediction on 55.5% of pairs vs. 47.3% for BPE (+8.2 pp) and lowers KL by 7.6%, winning or tying in 10 of 11 categories; a generation probe shows Kronecker echoes byte-novel strings and typos through generation where BPE forgets them. Fourth, BPE embedding norm drifts during training while Kronecker projection norm stays near 1.0, consistent with a stable representational target. Fifth, an on-the-fly runtime variant reconstructs embeddings from a 4.5 MB byte buffer rather than a 2.15 GB table at vocabulary 131,072, with 0.01--0.24% step-time overhead. Byte-level locality has a tradeoff: byte-similar but semantically distant pairs (compute/commute, nation/notion) cluster together, shifting disambiguation to early attention layers.
>
---
#### [new 094] Beyond English and Evasion: A Human-Annotated Multi-Domain Benchmark for High-Stakes LLM Safety Evaluation in Chinese
- **分类: cs.CL**

- **简介: 该论文属于LLM安全评估任务，旨在解决中文环境下安全系统失效问题。构建了包含1,897个对抗性中文提示的基准数据集，涵盖高风险领域，提供详细标注以提升模型安全性。**

- **链接: [https://arxiv.org/pdf/2605.29667](https://arxiv.org/pdf/2605.29667)**

> **作者:** Wajdi Zaghouani; Kholoud K. Aldous; Yicheng Gao
>
> **摘要:** When Large Language Models (LLMs) are deployed in Chinese-language settings, a troubling pattern emerges: safety systems that work well in English break down. These systems struggle to cross linguistic and cultural bound-aries, leaving models exposed to adversarial prompts that exploit Chinese-specific evasion techniques, including Pinyin romanization, character decomposition, internet slang, and hedging tone. To address this gap, we introduce ChiSafe-PAS (Chinese Safety Pilot Annotation Set), a human-annotated benchmark of 1,897 adversarial Chinese prompts spanning four high-stakes domains: self-harm and violence, drug and illicit trade, fraud, and satire. Of these, 1,544 entries carry complete gold-standard annotations: a 3-class response label (REFUSE, SAFE-REDIRECT, RESPOND), a nine-category obfuscation taxonomy, a risk-level rating, and annotator rationale. We describe the dataset design, annotation process, and obfuscation taxonomy in detail. Our primary goal is practical: to give the research community a high-quality, culturally grounded resource for benchmarking LLM safety alignment. In doing so, we engage three broader tensions in the field: the blurring boundary between training and evaluation data, the need for domain coverage grounded in real-world risk, and the limits of scale as a substitute for cultural expertise.
>
---
#### [new 095] Towards Localized and Disentangled Knowledge Editing for Multimodal Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模态知识编辑任务，旨在解决现有方法在编辑知识时局限性强、易干扰无关信息的问题。提出LDKE框架，实现精准且泛化的知识编辑。**

- **链接: [https://arxiv.org/pdf/2605.29826](https://arxiv.org/pdf/2605.29826)**

> **作者:** Leijiang Gu; Zhen Zeng; Feng Li; Xinjian Gao; Zenglin Shi
>
> **摘要:** Existing methods in Multimodal Knowledge Editing (MKE) have advanced the ability to correct outdated or inaccurate knowledge in Multimodal Large Language Models (MLLMs). However, they exhibit a critical limitation: while effectively modifying target factual pairs, they fail to generalize edits to logically related queries and often cause unintended alterations to unrelated but visually or semantically linked information. We identify and formalize two underlying failure modes causing this issue: Causal Misalignment, which confines edits to the specific sample, and Feature Entanglement, which causes unintended alterations to coupled but irrelevant information. To address these issues, we propose Localized and Disentangled Knowledge Editing (LDKE), a new framework that achieves precise and generalized editing by localizing fact-specific model layers and disentangling target-relevant inputs from irrelevant ones. Our approach introduces a Fast Localization module to identify and update critical layers efficiently, along with a Disentanglement Classifier that routes inputs appropriately to preserve unrelated knowledge. Extensive experiments across various benchmarks and MLLMs demonstrate that LDKE achieves superior performance in propagating edits to related contexts while maintaining high locality.
>
---
#### [new 096] CCS: Clinical Consensus Selection for Radiology Report Generation
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于放射学报告生成任务，旨在提升生成报告的临床质量。针对传统单路径生成方法在推理时选择不足的问题，提出CCS框架，通过多候选报告选择提高临床一致性。**

- **链接: [https://arxiv.org/pdf/2605.30131](https://arxiv.org/pdf/2605.30131)**

> **作者:** Xi Zhang; Yingshu Li; Zaiqiao Meng; Jake Lever; Edmond S. L. Ho
>
> **备注:** 17 pages, 6 figures
>
> **摘要:** Radiology report generation (RRG) is commonly formulated as a single-path generation task, where a multimodal large language model (MLLM) produces one decoded report as the final output. While recent progress has largely been driven by scaling training data, model capacity, and retrieval mechanisms, improving report quality at inference time remains underexplored. In this work, we observe that fixed radiology MLLMs often generate clinically stronger reports elsewhere in their candidate pool than the one selected by default decoding, suggesting that inference-time decision making remains an overlooked bottleneck. To address this, we propose Clinical Consensus Selection (CCS), a decoder-agnostic inference-time selection framework that samples multiple candidate reports and selects the one with the highest clinical consensus across the rollout pool. CCS unifies text-based utilities with a radiology-adapted utility computed by an image--report-trained multimodal embedder, which measures candidate agreement beyond surface-level textual similarity. Across three datasets and multiple radiology MLLMs, CCS consistently improves inference-time performance over single-path decoding and generic Best-of-N baselines, with particularly clear gains on clinical metrics. Further analysis shows that image-grounded utility forms a selection axis distinct from textual consensus and that substantial headroom remains for improving RRG at inference time.
>
---
#### [new 097] EvoRubric: Self-Evolving Rubric-Driven RL for Open-Ended Generation
- **分类: cs.CL**

- **简介: 该论文提出EvoRubric，解决开放生成任务中奖励缺失问题，通过自进化评分体系实现模型与评分标准的协同优化。**

- **链接: [https://arxiv.org/pdf/2605.29847](https://arxiv.org/pdf/2605.29847)**

> **作者:** Xin Guan; Xiaomeng Hu; Shen Huang; Zhenyi Wang; Bo Zhang; Zijian Li; Pengjun Xie; Bo Liu; Jiuxin Cao
>
> **摘要:** Reinforcement Learning (RL) has significantly advanced Large Language Models (LLMs) in verifiable domains, but aligning models for open-ended generation remains profoundly challenging due to the lack of definitive rewards. Current rubric-based RL methods mitigate this by employing explicit criteria; however, they rely heavily on static, human-annotated rubrics that inevitably cause policy lag, or expensive external proprietary models for dynamic updates. In this paper, we propose EvoRubric, a novel single-policy co-evolutionary RL framework that eliminates the reliance on static criteria and on external rubric generators. By unifying response generation and rubric generation under a single parameterized policy, EvoRubric dynamically alternates between a Reasoner and a Rubric Generator. To prevent reward hacking and ensure the reliability of generated signals, we introduce a multi-level verification pipeline featuring a meta-verifier, zero-variance pruning, and a Leave-One-Out peer consensus mechanism. Validated criteria are dynamically archived into a memory pool, yielding dense, multi-objective rewards to continuously co-optimize both roles. Extensive experiments across Medical, Writing, and Science domains demonstrate that EvoRubric consistently outperforms traditional static and external-LLM-driven alignment methods. Notably, our framework is compatible with human-expert priors. When initialized with expert-annotated rubrics, EvoRubric can further uncover novel, discriminative dimensions, achieving better performance than relying solely on static expert annotations.
>
---
#### [new 098] SERC: LDPC-Inspired Semantic Error Correction for Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言生成任务，旨在解决大语言模型的幻觉问题。提出SERC方法，通过稀疏验证策略提升事实准确性。**

- **链接: [https://arxiv.org/pdf/2605.28837](https://arxiv.org/pdf/2605.28837)**

> **作者:** Gyumin Kim; Juhwan Park; Jaeha Kim; Seunggyun Han; Kyungrak Son; Ikbeom Jang
>
> **备注:** 15 pages, 2 figures, 6 tables. To appear in the Proceedings of the 28th International Conference on Pattern Recognition (ICPR 2026). Code available at this https URL
>
> **摘要:** While Large Language Models (LLMs) have demonstrated remarkable capabilities, their reliability is significantly compromised by hallucinations. Existing intrinsic self-correction methods attempt to address this, but often fail due to self-bias, where models struggle to identify errors in their own outputs without external verification. To overcome these limitations, we propose the LDPC-inspired semantic error correction for retrieval-augmented generation (SERC), providing a theoretical framework to interpret and mitigate LLM hallucinations. We reformulate the text generation process as a semantic noisy channel, treating generated responses as noise-corrupted codewords. Inspired by low-density parity-check (LDPC) codes, SERC employs a sparse verification strategy: instead of exhaustively checking all facts, it generates low-density verification queries and validates them against external evidence to efficiently detect and correct errors. We evaluate SERC on LongForm Bio and TruthfulQA benchmarks using Llama-3-8B and Qwen2.5-14B. Experimental results demonstrate that SERC outperforms both intrinsic self-correction methods and strong retrieval-augmented baselines, demonstrating significant gains especially in factual precision (FactScore). Notably, SERC enables small language models (SLMs) to surpass the performance of larger baselines in hallucination reduction and information preservation. Our findings demonstrate that SERC provides a training-free, model-agnostic solution that significantly reduces verification overhead compared to dense methods, achieving an optimal trade-off between cost and fidelity in resource-constrained environments.
>
---
#### [new 099] Does The Way You Plan Matter? An Empirical Study of Planning Representations for LLM Web Agents
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于Web代理任务，旨在解决LLM代理在规划表示上的不足。通过设计PlanAhead框架，评估不同规划形式对任务成功率的影响。**

- **链接: [https://arxiv.org/pdf/2605.29927](https://arxiv.org/pdf/2605.29927)**

> **作者:** Alejandra Zambrano; Sara Vera Marjanovic; Imene Kerboua; Xing Han Lù; Leila Kosseim
>
> **备注:** Extended version of paper submitted to EMNLP, waiting for acceptance
>
> **摘要:** Despite recent advances, LLM-based web agents still struggle with limited exploration, omission of critical steps, and sensitivity to task constraints. Prior work suggests that many of these failures stem from weaknesses in planning, yet the impact of alternative natural language plan representation remains unexplored. To address this, we introduce PlanAhead, a static planner-executor framework that evaluates the impact of plan representation in agent performance. We first automatically categorize WebArena tasks into 3 difficulty levels, enabling consistent difficulty grading without human annotation. Then we systematically evaluate 4 different plan representations on the tasks categorized as hard: sequential subgoals, narrative, pseudocode, and checklist; across different families of multimodal LLM powered agents (OpenAI, Alibaba, and Google). To account for stochastic variability, we introduce two novel evaluation metrics: Achievement Rate (AR) and Solved-Task Consistency (STC). Our results show that both, the plan formulation and the underlying LLM generating the plan, significantly influence web-agent robustness and task success.
>
---
#### [new 100] Text-Preserving Lossy Text Compression: A Study of Strategic Deletion and LLM Reconstruction
- **分类: cs.CL**

- **简介: 该论文研究损失性文本压缩任务，通过策略性删除文本并由大语言模型重建，解决传统压缩效果有限的问题。工作包括对比多种删除策略及优化解码器。**

- **链接: [https://arxiv.org/pdf/2605.29000](https://arxiv.org/pdf/2605.29000)**

> **作者:** Yuchun Zou; Junhong Tong; Jun Li
>
> **摘要:** Traditional lossless text compression preserves every byte, but its gains on natural language are often modest in realistic operating regimes. We study \emph{lossy semantic text compression}, where the encoder strategically deletes parts of the text and a large language model (LLM) reconstructs the original content from the retained skeleton. We benchmark a progression of deletion strategies, including uniform step deletion, word-length-guided deletion (WordLen), word-frequency-guided deletion (WordFreq), LP-optimized deletion (Opt), entropy-based deletion using GPT-2 surprisal, and hybrid methods that combine frequency and surprisal signals. Evaluation on the BBC News dataset across retention rates $\r_{keep} \in [0.1,0.9]$ shows three main findings. First, WordFreq is a strong low-cost baseline: despite using only a static frequency lookup, it remains competitive with much more expensive semantic methods while being far faster at the encoder. Second, semantic and hybrid methods provide their clearest gains at mild-to-moderate compression, whereas word-frequency deletion is often more robust at the lowest retention rates. Third, QLoRA fine-tuning yields a strong local decoder that is competitive with Gemini 2.0 Flash and is often strongest in decoder-only comparisons. Additional English and Chinese experiments show that the overall framework transfers across domains, while the best deletion rule remains dataset-dependent.
>
---
#### [new 101] Internal Representation, Not Clinical Knowledge: Where Apparent LLM Triage Failures Originate
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究LLM在临床分诊任务中的表现，探讨其失败是否源于临床知识不足。通过分析模型输出格式影响，发现失败源于格式而非临床表示。**

- **链接: [https://arxiv.org/pdf/2605.29889](https://arxiv.org/pdf/2605.29889)**

> **作者:** David Fraile Navarro; Berardino Como; Jialei Sheng; Soundariya Ananthan; Shlomo Berkovsky
>
> **备注:** 9 pages main text, 27 pages total including appendices; 7 figures, 25 tables
>
> **摘要:** Patient-voiced clinical-triage benchmarks report high under-triage rates for consumer LLMs for constrained multiple-choice output, yet the same cases score differently with free-text. We ask whether output format changes the model's \emph{clinical representation} or only the mapping from a preserved representation to an answer. Using sparse-autoencoder (SAE) features in Gemma 3 4B/12B IT and Qwen3-8B, we find the same medical features fire on the shared clinical narrative under both formats but go {silent} at the multiple-choice decision token in all the cases at every model. Three independent methods (natural-language autoencoder verbalization, decision-token logit attribution, and top-feature characterization) agree that scaffold and format features, but not medical features, drive the decision logits. Behaviorally, the multiple-choice penalty inverts under both structured and natural-language input, option-order shuffle rules out positional bias, and the gap is dominated by off-by-one decision (the model picks an adjacent acuity letter to the gold answer) rather than knowledge failure. Thus, the failure originates in the output format and not in the clinical representation.
>
---
#### [new 102] Understanding Safety-Sensitive Expert Behavior in Mixture-of-Experts LLMs
- **分类: cs.CL**

- **简介: 该论文研究MoE大模型中的安全行为，解决安全对齐与专家路由之间的关系问题。提出RASET框架，识别关键安全专家并进行高效调优。**

- **链接: [https://arxiv.org/pdf/2605.29708](https://arxiv.org/pdf/2605.29708)**

> **作者:** Zhibo Zhang; Yuxi Li; Zhen Ouyang; Ling Shi; Kailong Wang
>
> **备注:** 11 pages, 4 figures
>
> **摘要:** Mixture-of-Experts (MoE) LLMs rely on sparse, router-driven expert activation, yet how safety alignment interacts with routed expert specialization remains underexplored. A common intuition is that safety behavior may be controlled by routing harmful requests to distinct refusal-oriented experts. In this work, we provide empirical evidence for a different picture: routing patterns in aligned MoE LLMs are largely topic-driven, while safety behavior can be altered with little change to the model's intrinsic routing path. Motivated by this observation, we present **RASET** (**R**outer-**A**gnostic **S**afety-critical **E**xpert **T**uning), a red-teaming framework that probes safety enforcement that is localized in a small subset of experts while preserving the model's intrinsic routing behavior. **RASET** identifies safety-critical experts via a contrastive routing-sensitivity criterion and applies parameter-efficient tuning only to the selected experts, minimizing semantic disruption relative to router-steering interventions. These results reveal a distinct MoE safety risk, highlighting the need for expert-aware alignment mechanisms.
>
---
#### [new 103] ExCAM: Explainable Cultural Awareness Metrics
- **分类: cs.CL**

- **简介: 该论文属于文化意识评估任务，旨在解决大模型生成文本的文化偏差问题。工作包括提出ExCAM指标和相关数据集，实现文化错误的识别与解释。**

- **链接: [https://arxiv.org/pdf/2605.29897](https://arxiv.org/pdf/2605.29897)**

> **作者:** Christoph Leiter; Haiyue Song; Hour Kaing; Jin Tei; Hideki Tanaka; Masao Utiyama; Steffen Eger
>
> **备注:** preprint
>
> **摘要:** Evaluating the cultural awareness of large language models is crucial to ensure the fairness of generated text and the generalizability of applications across the world. Recent benchmarks explore cultural goods like food or values like behavior in stressful situations through the lens of question answering or text generation tasks. However, creating these benchmarks requires time-intensive and costly human annotations. Also, benchmarks that evaluate cultural awareness in free text are scarce and often rely on dated evaluation mechanisms. To address this gap, we introduce ExCAM, an Explainable Cultural Awareness Metric, which is, to our knowledge, the first dedicated evaluation metric that identifies, rates and explains cultural errors in instruction-output pairs. To train and evaluate ExCAM, we introduce ExCAM40k, a dataset comprised of nine existing benchmarks that we reformat and enhance with synthetic errors. Compared to several baselines, including GPT-5, ExCAM achieves the highest error detection rate with up to 80% accuracy on a balanced test set. Therefore, ExCAM opens the pathway towards fine-grained and explainable cultural evaluation of free text.
>
---
#### [new 104] Reasoning-preserved Efficient Distillation of Large Language Models via Activation-aware Initialization
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于模型压缩任务，解决EDistill方法中多步推理能力下降的问题。通过激活感知初始化缓解eRank崩溃，提升推理能力。**

- **链接: [https://arxiv.org/pdf/2605.29327](https://arxiv.org/pdf/2605.29327)**

> **作者:** Junlin He; Yihong Tang; Tong Nie; Guilong Li; Binyu Yang; Jinxiao Du; Lijun Sun; Wei Ma
>
> **摘要:** Efficient Distillation (EDistill) compresses large language models (LLMs) by structured pruning parameters and tuning lightweight modules with high training efficiency. Although these EDistilled LLMs achieve state-of-the-art (SOTA) performance on general ability benchmarks relative to similarly sized LLMs, we identify a severe degradation in their multi-step reasoning ability, which we term reasoning collapse. We systematically analyze the geometric origins of reasoning collapse and show that the SOTA EDistill method based on width-reducing projection matrices suffers from eRank collapse, in which the effective rank (eRank) of hidden representations drops. We theoretically explain how singular values of randomly initialized projection matrices become unevenly distributed, leading to eRank collapse and thus token indistinguishability. To address this issue, we propose RED (Reasoning-preserved Efficient Distillation) for LLMs, which introduces activation-aware initialization to initialize projection matrices as channel-selection matrices, thus theoretically mitigating eRank collapse. Experiments on Llama and Qwen series demonstrate that RED substantially recovers reasoning while maintaining high training efficiency and SOTA general ability.
>
---
#### [new 105] PatchBoard: Schema-Grounded State Mutation for Reliable and Auditable LLM Multi-Agent Collaboration
- **分类: cs.CL**

- **简介: 该论文提出PatchBoard，用于解决LLM多智能体协作中的状态验证与审计问题。通过结构化状态变更替代对话，提升可靠性和可审计性。**

- **链接: [https://arxiv.org/pdf/2605.29313](https://arxiv.org/pdf/2605.29313)**

> **作者:** Shuyu Zhang; Yaqi Shi; Lu Wang
>
> **摘要:** LLM multi-agent systems often coordinate through natural-language dialogue or loosely structured shared memory, making intermediate state difficult to validate, attribute, and audit. We introduce PatchBoard, a schema-grounded collaboration architecture that replaces inter-agent dialogue with validated JSON Patch mutations over a shared structured state. An Architect agent constructs a task-specific schema and workflow rules, while a deterministic kernel validates each proposed state mutation against schema constraints, role-specific write contracts, and runtime invariants before committing it transactionally. On 630 matched ALFWorld episodes, PatchBoard achieves an 84.6% success rate, compared with 30.8% for LangGraph and 61.6% for Flock, while reducing tokens per successful task to 45.5k, compared with 368.3k and 64.2k, respectively.
>
---
#### [new 106] Unlocking the Working Memory of Large Language Models for Latent Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的推理任务，旨在解决大模型推理效率低的问题。通过引入记忆块替代自回归生成，实现更高效的潜在推理。**

- **链接: [https://arxiv.org/pdf/2605.30343](https://arxiv.org/pdf/2605.30343)**

> **作者:** Lukas Aichberger; Sepp Hochreiter
>
> **备注:** Preprint
>
> **摘要:** To improve the reasoning capabilities of large language models, test-time compute is typically scaled by generating intermediate tokens before the final answer. However, this couples reasoning to autoregressive generation and thereby conflates internal computation with external communication. In contrast, human cognition can use working memory to hold and manipulate information internally without the need to externalize intermediate thoughts. Drawing on this principle, we introduce Reasoning in Memory (RiM), a latent reasoning method that replaces the autoregressive generation of reasoning steps with memory blocks. These memory blocks are fixed sequences of special tokens that unlock the working-memory capacity of large language models. Since they are fixed rather than generated, they can be processed in a single forward pass, enabling compute-efficient latent reasoning. To operationalize these memory blocks, we employ a two-stage curriculum. First, we ground them by predicting explicit reasoning steps after each memory block. Second, we discard this step-level supervision and iteratively refine the final answer after each memory block. Our experiments on reasoning benchmarks show that, across language models of different families and sizes, RiM matches or exceeds existing latent reasoning methods while avoiding the autoregressive generation of thoughts. These results demonstrate that large language models can be trained to use working memory as an effective mechanism for latent reasoning.
>
---
#### [new 107] COMPOSE: Composing Future Theorems from Citations and Formal Structure
- **分类: cs.CL**

- **简介: 该论文提出COMPOSE框架，用于生成基于科学引用和形式结构的未来数学命题，解决数学结论生成的合理性与动机问题。**

- **链接: [https://arxiv.org/pdf/2605.30333](https://arxiv.org/pdf/2605.30333)**

> **作者:** David Busbib; Michael Werman
>
> **摘要:** A plausible future mathematical claim must satisfy two constraints: it should follow the direction of prior work and respect the formal dependencies that constrain what can validly follow. Existing approaches typically model only one of these sources, producing claims that are either weakly grounded or insufficiently motivated. We introduce grounded future mathematical generation, where the goal is to generate a plausible future theorem-like claim for an anchor paper using two complementary sources of context: its scientific citation graph and aligned formal theorem dependency graph. To address this setting, we propose COMPOSE, a dual-graph framework that conditions a language model on both scientific citation context and formal theorem structure. To support this setting, we construct a dataset of 108K paired scientific-formal graph examples from arXiv and Mathlib, together with a benchmark of 47K future papers from 2024--2025. Experiments show that COMPOSE outperforms strong baselines on retrieval to real future papers and achieves the best overall performance under LLM-judge evaluation, producing more grounded and mathematically richer outputs. These results show that future mathematical generation benefits from combining scientific context with formal structure. Project page is available at this https URL.
>
---
#### [new 108] Structured Prompt Optimization Meets Reinforcement Learning for Global and Local Interpretability over Complex Text
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于文本分类任务，旨在解决LLM在可解释性与性能间的平衡问题。通过结构化提示优化和强化学习，提出eXTC模型，提升分类性能与解释能力。**

- **链接: [https://arxiv.org/pdf/2605.29076](https://arxiv.org/pdf/2605.29076)**

> **作者:** Tianyang Zhou; Wenbo Chen; Pierre Jinghong Liang; Leman Akoglu
>
> **摘要:** LLMs have advanced text classification, yet existing paradigms face a trade-off: supervised (label only) fine-tuning is scalable but offers limited reasoning on complex text and lacks broader model transparency, while discrete prompt optimization offers human-readable instructions but struggles with performance and scalability. We introduce eXTC (eXplainable Text Classifier) with three progressive stages: (1) learning a Standard Operating Procedure (SOP, or rulebook) in natural language via a new Structured Prompt Optimization algorithm; (2) SOP-grounded reasoning distillation from a large teacher LLM into a compact LM; and (3) expanding reasoning capabilities beyond the initial SOP via reinforcement learning. This design enables eXTC to provide (i) fast inference via a compact LM, with (ii) inference-time local reasoning traces, alongside a global, modular explanation of its learned domain rules, while (iii) significantly outperforming existing paradigms across diverse benchmarks in both classification performance and explanation quality, with stage-by-stage gains.
>
---
#### [new 109] CommunityFact: A Dynamic, Multilingual, Multi-domain Benchmark for Misinformation Detection in the Wild
- **分类: cs.CL; cs.CY; cs.SI**

- **简介: 该论文属于虚假信息检测任务，旨在解决动态、多语言、多领域环境下模型可靠性评估问题。工作包括构建CommunityFact基准，评估LLMs性能，并分析源选择策略差异。**

- **链接: [https://arxiv.org/pdf/2605.30241](https://arxiv.org/pdf/2605.30241)**

> **作者:** Sahajpreet Singh; Insyirah Mujtahid; Min-Yen Kan; Kokil Jaidka
>
> **摘要:** Misinformation verification increasingly occurs in public, fast-moving, and multilingual online settings, where static benchmarks provide an incomplete measure of model reliability. We introduce CommunityFact, a refreshable benchmark for misinformation detection in the wild, with three major goals: coverage, granularity, and redistributability. This release contains 15,992 standalone claims across five languages and two domains. We evaluate ten LLMs under varying inference-time capabilities, including thinking and web-search. Our results show that closed-input verification remains challenging, web access yields the largest gains, and web-enabled LLMs' source-selection policies are systematically misaligned with the sources human Community Notes raters converge on -- a gap that closes through model-specific mechanisms of retrieval expansion or pruning. We further find substantial variation across language-domain slices and across the evidence ecosystems used by web-enabled systems. Beyond evaluation, CommunityFact positions Community Notes as a training signal for claim-conditioned source suggesters that could improve factual verification on novel claims.
>
---
#### [new 110] Causal Interventions on Continuous Variables: A Case Study on Verb Bias in Steering Vectors for In-Context Learning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，研究如何对连续变量进行因果干预。针对语言模型中的动词偏差问题，提出方法并验证其对结构偏好影响，探讨其与上下文学习的关联。**

- **链接: [https://arxiv.org/pdf/2605.29971](https://arxiv.org/pdf/2605.29971)**

> **作者:** Zhenghao Herbert Zhou; R. Thomas McCoy; Robert Frank
>
> **摘要:** Causal interventions in language model representations have largely targeted discrete features, like grammatical number. However, language models must also make use of features that are graded. We introduce a method for causal intervention on continuous variables: given activation vectors paired with a graded target variable, we localize a low-dimensional direction for that variable and use this direction to edit a vectors toward counterfactual target values. We apply this method to a continuous feature that is well-studied in psycholinguistics, namely verb bias (which reflects which syntactic structures tend to follow a given verb). We show that verb bias is causally represented in steering vectors extracted from large language models: counterfactual edits to verb bias systematically shift downstream structural preferences. Verb bias has also previously been linked to in-context learning; in further analyses, we find that steering vectors encode error signals that could drive the error-driven update behavior seen in in-context learning but that these aspects of the steering vectors are not causally used in downstream production. Overall, these results show causal interventions can be applied to continuous variables, though connecting continuous variables to in-context learning remains a challenge.
>
---
#### [new 111] A Modular Architecture for Typologically Controlled Lexicon Generation
- **分类: cs.CL**

- **简介: 该论文属于语言生成任务，旨在构建可发音、符合语言类型学且语义结构化的词汇。通过模块化框架生成词表，解决现有方法缺乏形式化保证的问题。**

- **链接: [https://arxiv.org/pdf/2605.28824](https://arxiv.org/pdf/2605.28824)**

> **作者:** Sankalp Tattwadarshi Swain; Dhruv Kumar
>
> **摘要:** Constructing artificial lexicons that are pronounceable, typologically plausible, and semantically structured remains an open challenge in computational linguistics. Existing conlang generators either lack formal phonotactic guarantees or delegate generation to opaque, non-reproducible LLM-based pipelines. We propose a modular framework that samples phoneme inventories from PHOIBLE, generates word forms under interchangeable phonological grammars (deterministic, OT, and MaxEnt), and assigns meanings via a Swadesh--Leipzig--Jakarta ontology with explicit form--meaning alignment. Evaluation on character $n$-gram perplexity, log-likelihood, and KL divergence against PHOIBLE across lexicon sizes of 100-5,000 forms shows that probabilistic grammars consistently outperform deterministic and random baselines on both phonotactic coherence and typological realism.
>
---
#### [new 112] A Dual-Path Architecture for Scaling Compute and Capacity in LLMs
- **分类: cs.CL**

- **简介: 该论文属于语言模型优化任务，解决固定计算量下模型容量不足的问题。提出双路径结构，同时提升计算和参数效率，实验表明优于现有方法。**

- **链接: [https://arxiv.org/pdf/2605.30202](https://arxiv.org/pdf/2605.30202)**

> **作者:** Markus Frey; Behzad Shomali; Joachim Koehler; Mehdi Ali
>
> **摘要:** Looped transformers apply a shared block multiple times and have emerged as a parameter-efficient route to scaling compute in language models. However, at fixed FLOPs a looped model has strictly less capacity than a baseline transformer. We propose a novel dual-path block that can flexibly scale compute, the number of sequential operations applied to a hidden state, and capacity, the parameters available at a single step. For this we expose both axes as parallel pathways within a single layer: a deep sublayer re-applied K times with shared parameters, and a wide sublayer with an enlarged feed-forward network applied once. Independent per-token gates combine both axes and allow detailed per-token routing analyses. We show that across two FLOP budgets, our dual-path model surpasses iso-FLOP matched models on language modeling and downstream evaluations, while using fewer parameters than the baseline at matched FLOPs. The learned gates are directly interpretable and show systematic per-token allocation with function words and lexical content trend wide, while punctuation, symbols, and arithmetic tokens trend deep.
>
---
#### [new 113] The Trust Paradox: How CS Researchers Engage LLM Leaderboards
- **分类: cs.CL; cs.HC**

- **简介: 该论文研究LLM排行榜对研究人员的影响，探讨其可靠性与实际使用间的矛盾。属于人工智能评估领域，旨在解决排行榜与实际科研需求不匹配的问题，通过访谈分析提出改进设计建议。**

- **链接: [https://arxiv.org/pdf/2605.28966](https://arxiv.org/pdf/2605.28966)**

> **作者:** Pouya Sadeghi; Anamaria Crisan; Jimmy Lin
>
> **摘要:** Large language model (LLM) leaderboards rank AI models using standardized benchmarks and have become highly visible across computer science, despite known limitations in their reliability and robustness. Yet how they shape researchers' actual practice remains empirically uncharted. We address this gap through semi-structured interviews with eight researchers across four computer science subfields, analyzed using reflexive thematic analysis. We find a near-universal paradox of pragmatic skepticism: while participants expressed deep distrust of leaderboard rankings, they continued to use them as rough decision-making aids. Peer networks, not leaderboards, emerged as the primary model selection mechanism, and arena-based (human-voting) leaderboards were consistently preferred over static benchmark leaderboards. Leaderboard influence varied sharply across subfields, revealing that disciplinary culture, not individual attitudes, mediates engagement; for instance, NLP researchers faced state-of-the-art comparison pressure while HCI and Systems/Privacy researchers reported none. Across these differences, however, participants converged on cost transparency as the most demanded missing feature (seven of eight). We translate these findings into concrete design recommendations that align evaluation infrastructure with how researchers actually use it, such as task-specific score breakdowns, cost integration, and voter-demographic disclosure.
>
---
#### [new 114] STAMP: Training Explicit Memory for Mobile GUI Agents in Controllable and Scalable Virtual Environments
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于移动GUI代理任务，解决长期任务中记忆不足的问题。通过可控虚拟环境训练显式记忆，提升代理的长期记忆与任务鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.29324](https://arxiv.org/pdf/2605.29324)**

> **作者:** Junyang Wang; Haiyang Xu; Xi Zhang; Zhaoqing Zhu; Ming Yan; Jieping Ye; Jitao Sang
>
> **备注:** 24 pages, 4figures, 21 tables
>
> **摘要:** Mobile GUI agents excel at immediate reactive control but frequently fail in realistic, long-horizon tasks that require memory. This failure stems from a fundamental conflict between limited context windows and token-heavy screenshots. To save the limited context, agents must progressively discard older visual history, permanently losing crucial transient information. Furthermore, existing action-centric datasets fail to teach agents what or when to explicitly memorize, and augmenting static real-world data is prohibitively expensive and lacks interactive verification. To resolve this, we present STAMP, a framework that trains explicit memory in mobile agents through controllable virtual environments, where deterministic memory variables are programmatically injected into synthesized tasks to control what must be memorized, when it should be encoded, and when it must later be retrieved, thereby producing verifiable supervised data at scale and enabling online reinforcement learning through environment-driven reward feedback. Evaluated on our newly introduced Memory-World benchmark, the resulting Stamp-GUI agent achieves state-of-the-art performance among GUI-specialized models and sets a new high watermark on our Memory-World benchmark, demonstrating exceptional memory accuracy and task resilience while maintaining strong general mobile navigation capabilities.
>
---
#### [new 115] Same Question, Different Source, Different Answer: Auditing Source-Dependence in Medical Multi-Source RAG
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于NLP评估任务，解决多源RAG系统答案依赖源的问题。通过构建基准和审计框架，评估答案与来源的关系，提升多源系统透明度。**

- **链接: [https://arxiv.org/pdf/2605.29084](https://arxiv.org/pdf/2605.29084)**

> **作者:** Yubo Li; Rema Padman; Ramayya Krishnan
>
> **摘要:** A retrieval-augmented generation (RAG) system deployed over a multi-author institutional corpus can give a different answer to the same question depending on which source it retrieves -- a failure mode the dominant single-gold-answer paradigm cannot diagnose. We argue that source-dependence is a missing axis of NLP evaluation, and that auditing it means shifting the unit of evaluation from answer correctness to the inter-source relationship. We make this concrete in transplant patient education, where institutional sources demonstrably disagree, releasing three artefacts: TransplantQA, a benchmark of real patient questions, each answered by grounding generation in multiple institutional handbooks as candidate sources; HERO-QA, a hierarchical retrieval strategy that grounds and audits each answer; and a structured-output judge that scores inter-source relationships on a validated 5-label taxonomy. At scale, better retrieval reveals far more disagreement than prior estimates suggested -- understating its prevalence, not its intensity. The framework is domain-agnostic and transfers to legal and educational RAG: measuring source-dependence is a responsibility for deployed multi-source NLP generally.
>
---
#### [new 116] DirectorBench: Diagnosing Long-Form Video Generation with Personalized Multi-Agent Evaluation
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于长视频生成评估任务，旨在解决现有基准无法全面诊断生成视频质量的问题。提出DirectorBench，通过多维度指标和用户个性化评估，揭示生成流程中的瓶颈与失败模式。**

- **链接: [https://arxiv.org/pdf/2605.30090](https://arxiv.org/pdf/2605.30090)**

> **作者:** Jiamin Chen; Qianben Chen; Jiawen Zhang; Yidi Wu; Yuchen Li; Xiaokun Zhang; Wangchunshu Zhou; Chen Ma
>
> **摘要:** Long-form video generation is rapidly moving from short, single-scene synthesis toward minute-long, multi-shot creation with narrative structure, cinematic control, audio, and cross-modal synchronization. However, evaluating such videos remains challenging, since existing benchmarks largely focus on local visual quality, short-horizon temporal consistency, or generic prompt alignment, and provide limited diagnosis of workflow failures and user-dependent preferences. We introduce DirectorBench, a personalized multi-agent diagnostic benchmark for long-form video generation. DirectorBench evaluates generated videos with respect to 80 structured metadata entries, 7 user profiles, and 40 checkpoint criteria across 5 dimensions: script, visual, audio, cross-modal, and stability. Instead of reducing quality to a single aggregate score, DirectorBench localizes checkpoint-level bottlenecks and supports profile-aware evaluation. We evaluate 4 long-form video generation workflows, 6 base LLMs, and 7 user profiles. Across workflows, DirectorBench reveals a between-unit bottleneck: transition quality averages only 0.256 and reaches 0.356 for the best workflow, while prompt-level user demand fulfillment averages 0.71. We further conduct human evaluation with 14 annotators to validate the alignment between DirectorBench and human judgment. The results show that DirectorBench captures human-perceptible quality differences and reveals workflow- and profile-dependent failure modes that are hidden by aggregate scoring. These findings highlight the importance of diagnostic and profile-aware benchmarking for long-form video generation.
>
---
#### [new 117] Lightweight Multimodal LLM-Enabled Cost-Effective Defect Grading of Power Transmission Equipment
- **分类: cs.CL**

- **简介: 该论文属于电力设备缺陷评级任务，旨在解决专家经验整合难和类别不平衡问题。通过多模态大语言模型和少量思维链问答对，实现高效低成本的缺陷分级。**

- **链接: [https://arxiv.org/pdf/2605.28822](https://arxiv.org/pdf/2605.28822)**

> **作者:** Tao Wang; Lipeng Zhu; Jiayong Li; Feng Gao; Siwen Liang
>
> **备注:** 9pages, 6figures
>
> **摘要:** Defect grading of power transmission equipment (DGPTE) is crucial to the stability of electric energy transmission. Although existing machine learning methods exhibit strong capabilities in defect detection, they are plagued by difficulties in integrating expert experience and facing class imbalance in more refined defect grading field. To address this issue, this paper introduces a novel defect grading framework based on multimodal large language model (MLLM). Specifically, this approach maximizes the commercial MLLMs' potential of DGPTE through in-context learning and obtains the state-of-te-art (SOTA) model. By sending a secondary request to this model, a small number of chain of thought-based question-answer pairs (Q\&As) are generated, which effectively reduces the cost of manual annotation. In this way, these high-quality interpretable Q\&As are used to train Qwen3-VL-8B via Low-Rank Adaption-based supervised fine-tuning (SFT). Experimental results on three DGPTE tasks demonstrate that fine-tuning only the language model layer yields the SOTA performance. Furthermore, multi-task joint fine-tuning verifies the feasibility of handling multiple grading tasks within only a single lightweight MLLM.
>
---
#### [new 118] Comparative Evaluation of Machine Translation Systems on Images with Text
- **分类: cs.CL**

- **简介: 该论文属于图像文本翻译任务，旨在比较不同机器翻译系统在含文字图像上的表现。工作包括评估模块化管道、多模态大模型和端到端模型，分析其效果与优势。**

- **链接: [https://arxiv.org/pdf/2605.29476](https://arxiv.org/pdf/2605.29476)**

> **作者:** Blai Puchol; Sergio Gómez González; Miguel Domingo; Francisco Casacuberta
>
> **摘要:** This work presents a comparative evaluation of machine translation systems applied to images containing textual information, a task that lies at the intersection of computer vision and natural language processing. The study compares three main paradigms: modular pipelines that separate text detection, recognition, and translation; multi-modal large language models (MLLMs) capable of processing both image and text jointly; and an end-to-end model, Translatotron-V, which directly generates translated images. The modular systems employ state-of-the-art OCR (docTR) combined with multilingual LLMs such as Llama and EuroLLM, while the evaluated MLLMs include different configurations of Gemini 2.5. Experiments were conducted on parallel multilingual datasets covering multiple language pairs, with evaluation based on BLEU, chrF, and TER metrics. The results show that modular pipelines outperform the end-to-end approach, while MLLMs achieve the best overall performance, demonstrating superior flexibility and contextual understanding. These findings underscore the effectiveness of multi-modal reasoning for image-to-text translation and provide a solid foundation for future research on integrating visual understanding and language generation in multilingual settings.
>
---
#### [new 119] HTAM: Hierarchical Transition-Attended Memory for Operator Optimization
- **分类: cs.CL**

- **简介: 该论文提出HTAM，用于GPU算子优化任务，解决LLM生成代码时的粒度不匹配问题，通过分层记忆结构提升优化效率与准确性。**

- **链接: [https://arxiv.org/pdf/2605.29734](https://arxiv.org/pdf/2605.29734)**

> **作者:** Yining Zhang; Mingyang Yi; Chen Wang; Xuwen Xiang; Tianhe Jia; Zedong Dan; Chengqing Zong; Yue Wang
>
> **备注:** 24 pages, 5 figures
>
> **摘要:** High-performance GPU kernels are essential for efficient LLM deployment, yet optimizing them remains expertise-intensive. Recent LLM-based code generation makes automatic GPU operator generation promising, but operator optimization remains a hardware-aware search problem. Existing LLM-based methods face a granularity mismatch: coarse hints are reusable but hard to execute, whereas detailed memories are actionable but enlarge the search space and obscure optimization bottlenecks. The key challenge is therefore to organize optimization experience at an appropriate granularity. To address this issue, this paper proposes HTAM (Hierarchical Transition-Attended Memory), a coarse-to-fine framework for LLM-based operator optimization. HTAM builds a two-level Hierarchical Transition Graph (HTG) to organize coarse global directions, detailed local strategies, and transition experience between optimization steps. During each evolution step, HTAM selects a global direction from the current state and recent optimization history, retrieves the corresponding local strategy memory, and uses it to guide concrete CUDA code generation. Experiments on the full KernelBench suite demonstrate that HTAM consistently improves correctness, fast-solution rate, and speedup over LLM-based baselines, while backend and Robust-KBench studies indicate transferable benefits from structured memory.
>
---
#### [new 120] DynSess: Dynamic Session-Level Evaluation and Optimization Framework for Role-Playing Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出DynSess框架，解决角色扮演对话中的会话级评估与优化问题。通过会话级评分和搜索训练，提升角色一致性与交互质量。**

- **链接: [https://arxiv.org/pdf/2605.29256](https://arxiv.org/pdf/2605.29256)**

> **作者:** Rongsheng Zhang; Jiji Tang; Junnan Ren; Zuyi Bao; Weijie Chen; Ruofan Hu; Zhou Zhao; Tangjie Lv; Yan Zhang
>
> **摘要:** Role-playing with large language models is fundamentally a session-level task, requiring agents to sustain character identity and interaction quality across extended multi-turn conversations. Yet existing evaluation and optimization methods remain largely turn-level, failing to capture long-horizon quality. We propose DynSess, a unified session-level framework for role-playing agents. DynSess-Eval scores complete dialogue sessions via rubrics targeting long-horizon behaviors. Leveraging its session-level rewards, we construct high-quality training trajectories through multi-turn lookahead search and train DynSess-Character with two complementary variants: DSPO (off-policy) and GSRPO (on-policy). Experiments show that DynSess-Eval aligns with human judgments substantially better than prior evaluators, and blind human evaluation further shows that DynSess-Character matches the strongest character model despite using substantially fewer parameters, while maintaining strong role consistency and interactive ability. Our dataset and code will be released to facilitate future research.
>
---
#### [new 121] GrepSeek: Training Search Agents for Direct Corpus Interaction
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **简介: 该论文提出GrepSeek，一种通过直接操作语料库进行搜索的智能代理，解决传统检索方法在复杂查询中的不足。**

- **链接: [https://arxiv.org/pdf/2605.29307](https://arxiv.org/pdf/2605.29307)**

> **作者:** Alireza Salemi; Chang Zeng; Atharva Nijasure; Jui-Hui Chung; Razieh Rahimi; Fernando Diaz; Hamed Zamani
>
> **摘要:** Large Language Model (LLM) search agents have shown strong promise for knowledge-intensive language tasks through multiple rounds of reasoning and information retrieval. Most existing systems access information using a retriever that takes a keyword or natural language query and returns a ranked list of documents using an index of pre-computed document representations. In this work, we explore a complementary perspective in which the search agent treats the corpus itself as the search environment and finds evidence by issuing executable shell commands. We introduce GrepSeek, an optimized direct corpus interaction (DCI) search agent that trains a compact search agent to find, filter, and compose evidence from large text corpora. To address the instability of learning behavior directly with reinforcement learning on large corpora, we propose a two-stage training pipeline. First, we construct a cold-start dataset using an answer-aware Tutor and answer-blind Planner to generate verified, causally grounded search trajectories. Second, we refine the initialized policy with Group Relative Policy Optimization (GRPO), allowing the agent to improve its task-oriented search behavior through direct interaction with the corpus. To make DCI practical at scale, we further use a semantics-preserving sharded-parallel execution engine that accelerates shell-based retrieval by up to $7.6\times$ while preserving byte-exact equivalence with sequential execution of the shell command. Experiments across seven open-domain question answering benchmarks show that GrepSeek achieves the strongest overall token-level $F_1$ and Exact Match. Our analysis also highlights the limitations of purely lexical interaction on queries with substantial surface-form variation, suggesting DCI as a practical and competitive method for search agents that can complement existing retrieval paradigms in the real world.
>
---
#### [new 122] LLMSurgeon: Diagnosing Data Mixture of Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出LLMSurgeon，用于诊断大语言模型的预训练数据混合情况，属于模型审计任务。解决无法获取训练数据时的后验分析问题，通过逆问题方法估计数据分布。**

- **链接: [https://arxiv.org/pdf/2605.30348](https://arxiv.org/pdf/2605.30348)**

> **作者:** Yaxin Luo; Jiacheng Cui; Xiaohan Zhao; Xinyi Shang; Jiacheng Liu; Xinyue Bi; Zhaoyi Li; Zhiqiang Shen
>
> **备注:** ACL 2026 Main. Code at this https URL
>
> **摘要:** The pretraining data mixture of Large Language Models (LLMs) constitutes their "digital DNA", shaping model behaviors, capabilities, and failure modes. Yet this composition is rarely disclosed, making post-hoc auditing of data combination or provenance difficult. In this work, we formalize $\textbf{Data Mixture Surgery (DMS)}$: given only generated text from a target LLM, estimate the domain-level distribution of its pretraining corpus under a predefined taxonomy. We propose $\textbf{LLMSurgeon}$, a strong framework that casts DMS as an inverse problem under the label-shift assumption. Rather than directly aggregating classifier outputs, LLMSurgeon estimates a calibrated $\textit{soft}$ confusion matrix and solves a constrained inverse problem to correct systematic domain confusion and recover the latent mixture prior. To evaluate, we introduce $\textbf{LLMScan}$, a recipe-verifiable evaluation suite built from open-source LLMs with transparent pretraining mixtures. Across LLMScan, LLMSurgeon recovers domain mixtures with high fidelity under fixed protocols. Our work presents a practical, post-hoc approach for auditing the digital DNA of foundation models without access to their training data.
>
---
#### [new 123] Benchmarking Open-Source Safety Guard Models: A Comprehensive Evaluation
- **分类: cs.CL; cs.AI; cs.SE**

- **简介: 该论文属于安全检测任务，旨在评估开源安全防护模型的有效性。针对内容审核问题，作者对14个模型进行基准测试，发现模型规模与安全性能无直接关联。**

- **链接: [https://arxiv.org/pdf/2605.28830](https://arxiv.org/pdf/2605.28830)**

> **作者:** Reetu Raj Harsh; Bhaskarjit Sarmah; Stefano Pasquali
>
> **摘要:** As Large Language Models (LLMs) are increasingly deployed in safety-critical applications, robust content moderation becomes essential. We present a comprehensive evaluation of 14 open-source safety guard models on a curated benchmark of 79,331 samples spanning 8 NIST AI Risk Framework safety categories. Our benchmark aggregates four diverse datasets (HarmBench, StrongREJECT, RealToxicityPrompts, and BeaverTails), filtered to focus exclusively on safety-relevant content (violence, hate speech, harassment, sexual content, suicide/self-harm, profanity, threats, and health misinformation). We find that recall is the critical metric for safety applications, as missing harmful content poses greater risk than false positives. Our evaluation reveals surprising results: Qwen Guard (4B parameters) achieves the highest recall (83.97%) while larger models like Llama Guard (12B) and GPT-OSS Safeguard (20B) exhibit conservative behavior, missing up to 75% of unsafe content. We demonstrate that model size does not correlate with safety detection performance and that general-purpose guard models outperform specialized ones. These findings provide practical guidance for selecting safety guard models in production deployments.
>
---
#### [new 124] Nine Judges, Two Effective Votes: Correlated Errors Undermine LLM Evaluation Panels
- **分类: cs.CL**

- **简介: 论文研究LLM评估小组的可靠性问题，发现九个模型仅提供约两个独立投票的信息，因错误相关性导致准确率下降。任务为评估模型一致性与有效性。**

- **链接: [https://arxiv.org/pdf/2605.29800](https://arxiv.org/pdf/2605.29800)**

> **作者:** Guneet Kohli
>
> **备注:** 14 pages, 5 figures, 12 tables
>
> **摘要:** LLM-as-a-judge panels aggregate votes from multiple models, with the expectation that diverse models yield more reliable evaluations. We develop a framework to measure the true informational value of such panels and quantify how far their reliability falls short of the independent-voting ideal. Testing a panel of 9 frontier LLMs from 7 model families on three natural language inference datasets (each with 100 human annotations per item), we find that the 9 judges effectively provide only about 2 independent votes' worth of information. Roughly three-quarters of the panel's nominal independence is lost because the models make the same mistakes on the same items. The consequences are stark: the panel's actual accuracy falls 8-22 percentage points short of what independent voting would achieve, and the best single judge matches or outperforms the full panel across all conditions. Neither adding more judges nor using smarter aggregation algorithms helps -- established methods close at most 11% of this gap, even with access to the correct answers. We quantify these findings using the Kish effective sample size (n_eff) and a Condorcet null model, and show the deficit is robust across prompt variants, temperatures, chain-of-thought reasoning, and a pairwise preference task (RewardBench). The bottleneck is correlated judges, not the aggregation algorithm, implying that scaling up panels cannot substitute for genuinely independent evaluation.
>
---
#### [new 125] FinGuard: Detecting Financial Regulatory Non-Compliance in LLM Interactions
- **分类: cs.CL**

- **简介: 该论文属于金融合规检测任务，解决LLM在金融交互中违反特定法规的问题。通过构建监管驱动的管道，生成合规数据并训练FinGuard模型，提升合规检测效果。**

- **链接: [https://arxiv.org/pdf/2605.29427](https://arxiv.org/pdf/2605.29427)**

> **作者:** Huaixia Dou; Jie Zhu; Minghao Wu; Shuo Jiang; Junhui Li; Lifan Guo; Feng Chen; Chi Zhang
>
> **摘要:** As large language models (LLMs) are increasingly deployed in financial services, a single non-compliant interaction can expose institutions to regulatory penalties and direct consumer harm. Existing guard models are built around general harm taxonomies and overlook violations grounded in specific financial regulations. We address this gap with a regulation-driven pipeline that operates directly on regulatory documents, inducing a financial compliance risk taxonomy and synthesizing grounded training data without any predefined violation categories. Instantiating the pipeline on Chinese financial regulations, we release \textbf{FinGuard-Bench}, to our knowledge the first benchmark for financial regulatory compliance detection, with expert-annotated labels at both the query and response levels. We further train \textbf{FinGuard}, a financial compliance detection model built on Qwen3-8B and trained on the regulation-grounded data via supervised fine-tuning and self-play reinforcement learning. On FinGuard-Bench, FinGuard substantially outperforms all baselines, including dedicated guard models and much larger general-purpose LLMs such as Qwen3.5-397B-A17B and GPT-5.1. Furthermore, FinGuard also preserves general safety capabilities and adapts to unseen institution-specific policies using policy documents alone. We will publicly release the code, prompts, and resources used in this work on GitHub.
>
---
#### [new 126] LiteCoder-Terminal: Scaling Long-Horizon Terminal Environments for Learning Language Agents
- **分类: cs.CL**

- **简介: 该论文属于语言代理训练任务，旨在解决环境生成受限问题。通过合成可执行环境，提升代理的多步规划与适应能力。**

- **链接: [https://arxiv.org/pdf/2605.29559](https://arxiv.org/pdf/2605.29559)**

> **作者:** Xiaoxuan Peng; Kaiqi Zhang; Xinyu Lu; Boxi Cao; Yaojie Lu; Hongyu Lin; Xianpei Han; Le Sun
>
> **摘要:** Mastering terminal environments requires language agents capable of multi-step planning, feedback-grounded execution, and dynamic state adaptation. However, training such agents is currently bottlenecked by a reliance on scraped external repositories, which limits domain diversity, environment controllability, and the targeting of specific capability deficits. We introduce LiteCoder-Terminal-Gen, a zero-dependency synthesis pipeline that autonomously generates executable and verifiable terminal training environments directly from domain specifications. Using this framework, we construct two large-scale resources: LiteCoder-Terminal-SFT, comprising 11,255 expert trajectories across 10 domains, and LiteCoder-Terminal-RL, featuring 602 verifiable environments for trajectory-level preference optimization. Supervised fine-tuning of Qwen-family models on our SFT dataset yields agents that significantly outperform their base counterparts. Notably, our 32B variant achieves 29.06%, 18.54%, and 34.00% pass@1 on Terminal Bench 1.0, 2.0, and Pro, respectively. Furthermore, applying Direct Multi-turn Preference Optimization (DMPO) on our RL environments yields additional performance gains. These results systematically demonstrate that fully synthetic, executable environments offer a scalable and verifiable supervision signal for mastering complex, real-world command-line workflows.
>
---
#### [new 127] Give it Space! Explicit Disentangling of Positional and Semantic Representations in Encoders
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究Transformer中位置编码机制，解决位置与语义信息混淆问题。通过分离位置和语义流，提升语言表示效果。**

- **链接: [https://arxiv.org/pdf/2605.30022](https://arxiv.org/pdf/2605.30022)**

> **作者:** Pierre-Antoine Lequeu; Camille Barboule; Benjamin Piwowarski
>
> **备注:** 8 page + 10 pages of bibliography and appendix
>
> **摘要:** Positional encoding (PE) underpins how permutation-invariant Transformers represent sequence order, yet how positional information is processed and stored remains poorly understood. Modern PE methods such as RoPE still struggle on tasks such as long-context understanding or retrieval \cite{chen-etal-2025-hope}. Hence, a better understanding of the internal positional mechanism could help design better PE. Building on evidence that positional and semantic signals occupy nearly orthogonal subspaces in trained Transformers, we modify an encoder Transformer to process three explicitly disentangled streams: semantic, absolute positional (AP) and relative positional (RP), and confine the masked-language-modeling (MLM) objective to the semantic stream. This decoupling enables a clean mechanistic study and yields three take-aways. (1) The isolated AP subspace spontaneously collapses into a low-frequency two-dimensional manifold that captures the structure of the document; (2) Attention heads specialize into structure and semantic-oriented groups, with RP exclusively supporting the latter; (3) Standard positional encodings do not robustly retain macroscopic structure: RoPE and RP only weakly encode it, and entangled AP loses it in the final layers under MLM pressure. The disentangled approach preserves positional encoding, which improves linguistic representation on 49 of the 65 linguistic phenomena of the Flash-Holmes probing benchmark.
>
---
#### [new 128] Specialty-Specific Medical Language Model for Immune-Mediated Diseases
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医学自然语言处理任务，旨在解决免疫相关疾病文本中实体识别的难题。通过构建专业数据集并训练模型，提升对疾病术语的识别精度。**

- **链接: [https://arxiv.org/pdf/2605.28838](https://arxiv.org/pdf/2605.28838)**

> **作者:** Veysel Kocaman; Gursev Pirge; Yigit Gul; Ace Vo; Zhenya Nargizyan; David Talby
>
> **备注:** 15 pages, 5 figures. Funded in part by NIAID/NIH under contract 75N93024C00010
>
> **摘要:** Extracting detailed clinical information from free-text medical narratives remains a practical challenge for researchers and healthcare systems. Terminology for immune-mediated and infectious diseases is especially inconsistent across sources, which often limits the ability of general-purpose Natural Language Processing (NLP) systems to capture the relevant biomedical concepts with sufficient granularity. We developed a domain-specific Named Entity Recognition (NER) model tailored to identify disease-related entities occurring in immunology and infectious disease contexts. We assembled and manually annotated a dataset of 371 case reports in collaboration with two clinical specialists, defining twelve entity classes covering immune-mediated and infectious conditions as well as related symptoms and clinical descriptors. We evaluated several modeling strategies, including the MedicalNER architecture with multiple healthcare-specific embeddings, a BERT-based token classification model, and zero-shot NER systems. The strongest performance was obtained with a transformer-based model trained on clinical-domain embeddings, which reached an F1 score of 0.89, consistently outperforming baseline and zero-shot approaches. The combination of specialized embeddings and expert annotation proved particularly valuable for capturing nuanced disease terminology and improving generalization across heterogeneous biomedical text. The prompted LLM baseline achieved substantially lower performance under the same evaluation protocol, reflecting difficulties in producing span-consistent outputs for fine-grained entity boundaries despite detailed prompting. The resulting model provides a structured way to analyze case reports and can support downstream tasks such as cohort identification, disease monitoring, and clinical decision support.
>
---
#### [new 129] Analyzing Persona Effects in Generated Explanations from Multimodal LLM Agents in Urban Perception
- **分类: cs.CL; cs.CV; cs.HC; cs.MA**

- **简介: 该论文研究多模态大语言模型在城市感知中生成解释的个性影响，分析不同个性设置下的描述、理由和标签差异，旨在理解个性对生成内容的影响。**

- **链接: [https://arxiv.org/pdf/2605.29064](https://arxiv.org/pdf/2605.29064)**

> **作者:** Neemias da Silva; Myriam Delgado; Rodrigo Minetto; Daniel Silver; Thiago H Silva
>
> **备注:** 10 pages, 6 figures
>
> **摘要:** We study how persona prompting shapes language generated by multimodal large language models in an urban perception setting. Using 59,808 annotations from 1,200 persona-conditioned agents and two no-persona settings, we analyze captions, justifications, and perception tags across personas. Results indicate strong convergence in captions for different personas, whereas justifications display systematic variation associated with socioeconomic and political attributes, while perception tags show no statistically significant persona-related differences, though effect trends are observed. Topic analysis further reveals that personas emphasize different evaluative themes when interpreting the same scenes.
>
---
#### [new 130] Do Language Models Track Entities Across State Changes?
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型在状态变化下的实体跟踪能力，探讨其处理复杂任务的机制，发现模型采用非序列策略，提出改进方法。**

- **链接: [https://arxiv.org/pdf/2605.30233](https://arxiv.org/pdf/2605.30233)**

> **作者:** Zilu Tang; Qiao Zhao; Gabriel Franco; Derry Wijaya; Aaron Mueller; Sebastian Schuster; Najoung Kim
>
> **备注:** ICML main conference 2026, 9 pages
>
> **摘要:** Entity tracking (ET), the ability to keep track of states, is a fundamental skill that underlies complex reasoning. An increasing amount of work investigates how transformer language models (LMs) solve entity binding $\textit{without}$ state changes. However, there is limited understanding of how non-toy LMs address ET problems of realistic difficulties expressed in natural language. To this end, we investigate the mechanisms underlying ET in more complex scenarios featuring multiple state-changing operations. We find that LMs do not incrementally track world states across tokens or query-relevant states across layers, but simply aggregate relevant information in parallel at the last token when the query becomes evident. We further investigate mechanisms of individual operations ($\texttt{PUT}$, $\texttt{REMOVE}$, $\texttt{MOVE}$) to characterize this non-incremental ET mechanism. Surprisingly, LMs implement the $\texttt{REMOVE}$ operation with a fragile global suppression tag; this global removal mechanism predicts various failure modes that we confirm behaviorally. We provide a mechanistic solution of nullifying this tag to partially address this issue. Overall, our findings reveal that LMs solve a fundamentally sequential task using a non-sequential strategy. More broadly, our work illustrates how behavioral and mechanistic analyses can fruitfully interact. Behavioral results inform mechanistic hypotheses, and insights from mechanistic analyses help build stronger behavioral evaluations by predicting failure modes missing from existing evaluations.
>
---
#### [new 131] MedCase-Structured: A Text-to-FHIR Dataset for Benchmarking Diagnostic Reasoning in Clinically Realistic EHR Settings
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于临床推理任务，旨在解决LLMs在真实电子病历环境中评估不足的问题。通过生成结构化FHIR数据集，评估LLMs在真实数据格式下的表现。**

- **链接: [https://arxiv.org/pdf/2605.30295](https://arxiv.org/pdf/2605.30295)**

> **作者:** Valentina Bui Muti; Eugénie Dulout; Ziquan Fu
>
> **备注:** Accepted to ICML 2026 Structured Data for Health Workshop
>
> **摘要:** Large language models (LLMs) show promise for clinical reasoning and decision support, but evaluation in realistic, electronic health record-congruent settings remains limited. Existing benchmarks often rely on static datasets or unstructured inputs that do not reflect the structured, interoperable data formats used in clinical systems. We introduce a pipeline for generating clinically realistic HL7 FHIR R4 bundles from unstructured text, enabling controllable evaluation of clinical decision support systems. The pipeline combines staged LLM generation with terminology-grounded validation and repair to reduce hallucinated codes and enforce structural and semantic consistency. Applying this approach to MedCaseReasoning, we construct MedCase-Structured, a synthetic dataset aligned with clinician-authored diagnostic cases, achieving valid FHIR generation for 82.5% of cases. Evaluation on MedCase-Structured reveals consistently lower diagnostic accuracy for LLMs on structured FHIR inputs than with plain text, highlighting the importance of deployment-aligned benchmarking.
>
---
#### [new 132] Knowing What to Solve Before How: Preplan Empowered LLM Mathematical Reasoning
- **分类: cs.CL**

- **简介: 该论文属于数学推理任务，旨在解决LLM在解题前缺乏明确问题理解的问题。提出PPC框架，增加预规划阶段，提升推理准确性。**

- **链接: [https://arxiv.org/pdf/2605.30245](https://arxiv.org/pdf/2605.30245)**

> **作者:** Shaojie Wang; Liang Zhang
>
> **摘要:** Current plan-based reasoning methods improve large language models (LLMs) by inserting a planning stage before execution, giving rise to the question $\rightarrow$ plan $\rightarrow$ cot paradigm. While effective, a closer examination reveals an inherent paradigm-level gap: both the planning and its execution stages decide how to solve a problem, while the prior question of what to solve; recognizing the problem type, the applicable tools, and the foreseeable pitfalls; remains entirely implicit. To bridge this gap, we propose PPC (Preplan-Plan-CoT), a framework that introduces an explicit problem-understanding stage, the preplan, yielding a new question $\rightarrow$ preplan $\rightarrow$ plan $\rightarrow$ cot paradigm. Realizing this paradigm requires safeguarding the conceptual integrity of preplan at both ends. Specifically, we design a three-stage synthesis pipeline with a spoiler-score detector that filters out leakage and spoiler failures to build clean preplan supervision, and a composite GRPO reward enforces that the generated plan genuinely follows from the preplan. Experiments across four backbones and five mathematical reasoning benchmarks show that PPC achieves the best results on 39 of 40 metrics, improving maj@16 and pass@16 by +2.23 and +3.06 over the strongest baseline without introducing additional inference token overhead.
>
---
#### [new 133] A Study on Question-Answer Dataset for LLM Safety Evaluation with a Focus on Illegal Activities
- **分类: cs.CL**

- **简介: 该论文属于LLM安全评估任务，旨在解决非法活动相关风险评估问题。通过构建问答数据集和评价标准，提升模型安全性。**

- **链接: [https://arxiv.org/pdf/2605.29340](https://arxiv.org/pdf/2605.29340)**

> **作者:** Kenji Imamura; Masao Ideuchi; Atsushi Fujita
>
> **备注:** 10 pages, 1 figure
>
> **摘要:** In this paper, we discuss question-answer dataset for LLM safety evaluation, with a focus on illegal activities. Specifically, on the basis of manual analysis of AnswerCarefully, we introduce several additional information, methods for creating question-answer examples, and a rubric for evaluating LLM-generated responses. The outcomes of this study are intended to be shared with the "JAI-Trust" project.
>
---
#### [new 134] GrowLoop: Self-Evolving Conversation Evaluation Seeded by Human
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文提出GrowLoop，解决开放问答中人类相似性评估难题。通过自进化机制，持续优化评估标准，提升模型评估准确性与适应性。**

- **链接: [https://arxiv.org/pdf/2605.28882](https://arxiv.org/pdf/2605.28882)**

> **作者:** Yihang Lin; Yunze Gao; Zeyang Lin; Dongbo Li; Kun Peng; Chenglong Song; Yue Liu
>
> **摘要:** With the rapid advancement of large language models, evaluating human-likeness in open-ended conversation has become increasingly important. However, human-likeness is a form of tacit knowledge that humans perceive intuitively, yet the underlying criteria resist explicit formulation. Human judgments vary widely, with strong agreement on some cases and legitimate disagreement on others. Meanwhile, the criteria behind human judgments remain implicit, leaving no clear basis for constructing cases. Further, what counts as human-like is not static, but evolving with model capability and human expectations. Despite progress in evaluation methods such as expert-authored benchmarks, Reward Models, and self-evolving benchmarks, none addresses all three challenges simultaneously. Therefore, we propose GrowLoop, a self-evolving conversation evaluation system that continuously adapts as models advance and scenarios shift. With minimal human seed annotations as the first mover, LLM agents iteratively extract and refine evaluation rubrics through Heuristic Learning. Human-AI agreement is required where annotators converge, while only plausibility is expected where they diverge. Moreover, the Rubric-Case co-evolution mechanism enables continuous evolution, expanded through new seeds when the evaluation target moves. Applied to human-likeness evaluation in open-ended conversation, the generated rubrics not only substantially outperform existing methods in alignment with human judgments, but also uncover issues that annotators overlook. The resulting benchmark effectively discriminates models across capability tiers and reveals where they fall short, while generalizing to new scenarios and adapting as models advance. Our work shifts the benchmarking paradigm from manual updates or difficulty scaling to comprehensive, continuous self-evolution.
>
---
#### [new 135] From Data to Insights: Exploring Program-of-Thoughts Prompting for Chart Summarization
- **分类: cs.CL**

- **简介: 该论文属于图表摘要任务，旨在解决图表理解中的语义和数值推理难题。通过引入程序思维策略，利用Python程序生成有效统计信息，提升图表描述的准确性与效率。**

- **链接: [https://arxiv.org/pdf/2605.28874](https://arxiv.org/pdf/2605.28874)**

> **作者:** Yutong Qu; Wei Zhang
>
> **备注:** 22 pages, 9 figures
>
> **摘要:** Charts play a critical role in conveying numerical data insights through structured visual representations. However, semantic visual understanding and numerical reasoning requirements hinder the accurate description of charts, interpreting a challenging task in chart summarization. Despite recent advancements in visual language models (VLMs), approaches lack robust mechanisms for verifying statistical fact correctness and are computationally heavy. To address this gap, this paper explores a strategy of using zero-shot learning to motivate the lightweight VLMs to perform computational reasoning, via Python programs as intermediaries to derive valid summary statistics for chart understanding. Specifically, we introduce a novel chart-to-dictionary auxiliary task, offering a more flexible representation compared to traditional chart-to-table methods, making it particularly well-suited for integration with the Program-of-Thought (PoT) strategy. Experimental results demonstrate our strategy performs on par with existing chart summarization methods across semantic and factual metrics. Code is available on this https URL.
>
---
#### [new 136] Multi-Legal-Bench: Evaluating LLMs on Legal Reasoning Across Jurisdictions, Languages, and Legal Traditions
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于法律自然语言处理任务，旨在解决跨司法管辖区和语言的法律推理评估问题。研究构建了Multi-Legal-Bench基准，评估多种模型在不同国家和语言中的表现。**

- **链接: [https://arxiv.org/pdf/2605.29738](https://arxiv.org/pdf/2605.29738)**

> **作者:** Volodymyr Ovcharov
>
> **备注:** 14 pages, 5 figures, 8 tables. Dataset: this https URL
>
> **摘要:** Legal NLP benchmarks overwhelmingly evaluate a single language or aggregate tasks that differ fundamentally across jurisdictions, making cross-lingual comparison impossible. We introduce Multi-Legal-Bench, the first cross-jurisdictional legal benchmark that evaluates identical tasks across six countries (Ukraine, France, Netherlands, Poland, Czech Republic, Lithuania), four language families, and 134 million court decisions. The benchmark defines five tasks court-type classification, judgment form classification, case-outcome prediction, legal norm extraction, and cause category prediction mapped to structured metadata from national court registries, forming a deliberately sparse 5x6 task-jurisdiction matrix (20 of 30 cells filled). We evaluate 7 frontier LLMs under zero-shot and 3-shot prompting via AWS Bedrock, with 4 additional small/medium models (3-12B) for scaling analysis. Our results reveal that: (1) task-dependent few-shot effects discovered in Ukrainian replicate across all jurisdictions; (2) no single model dominates any language rankings shift with both task and jurisdiction; (3) cross-lingual few-shot transfer does not follow language proximity: UA->FR (Romance, -2.1 pp) transfers better than UA->PL (Slavic, -13.7 pp), with label-set alignment predicting transfer quality better than language family; and (4) tokenizer fertility, despite a 2.3x spread, does not significantly predict cross-lingual accuracy (r=-0.27, p=0.14), suggesting that model architecture and pretraining data dominate tokenizer efficiency. We release all data, prompts, and model predictions.
>
---
#### [new 137] SAAS: Self-Aware Reinforcement Learning for Over-Search Mitigation in Agentic Search
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于信息检索任务，解决代理搜索中的过搜索问题。通过引入自aware强化学习框架SAAS，优化搜索行为，减少冗余搜索，提升效率。**

- **链接: [https://arxiv.org/pdf/2605.29796](https://arxiv.org/pdf/2605.29796)**

> **作者:** Yunbo Tang; Chengyi Yang; Shiyu Liu; Zhishang Xiang; Zerui Chen; Qinggang Zhang; Jinsong Su
>
> **摘要:** Agentic search enables LLMs to solve complex multi-hop questions through iterative reasoning and external search. Despite the effectiveness, these systems often suffer from a critical limitation in practice: agents fail to recognize their own knowledge boundaries, blindly triggering searches when internal knowledge suffices and failing to terminate search even when adequate evidence has been collected. The lack of self-awareness leads to severe \textbf{over-search}, incurring substantial inference latency and prohibitive computational cost. To this end, we propose SAAS, a novel RL framework designed to cultivate dynamic self-awareness that precisely regulates search behavior without compromising accuracy. SAAS introduces three key components: (i) a search boundary modeling mechanism, which identifies the search boundary under the evolving policy by contrasting search-disabled and search-enabled rollouts; (ii) a boundary-aware reward module, which translates this boundary awareness into trajectory-level penalties, suppressing unnecessary and redundant searches; and (iii) a stage-wise optimization strategy, which leverages a sequential curriculum to prioritize reasoning over search regularization, thereby avoiding reward hacking. Extensive experiments demonstrate that SAAS substantially reduces over-search, while maintaining accuracy. Our code is anonymously released at this https URL.
>
---
#### [new 138] Mind Your Tone: Does Tone Alter LLM Performance?
- **分类: cs.AI; cs.CL; cs.HC**

- **简介: 该论文属于自然语言处理任务，研究语气对大语言模型性能的影响。通过实验分析不同语气对模型准确率的影响，发现模型表现存在显著差异，并提出一种解释机制。**

- **链接: [https://arxiv.org/pdf/2605.29027](https://arxiv.org/pdf/2605.29027)**

> **作者:** Om Dobariya; Akhil Kumar
>
> **备注:** 10 pages, 6 tables, 1 figure. Accepted as a full paper at the Thirty-second Americas Conference on Information Systems (AMCIS 2026), Reno. Follow-up to arXiv:2510.04950
>
> **摘要:** The use of Large Language Models (LLMs) is proliferating, yet their performance is observed to vary based on prompting styles and tones. In this study, we investigate both whether and how tonal variations in prompts lead to disparate LLM accuracy for objective multiple-choice questions. We use two datasets: a 50-base question dataset with five tone variants and a 570-base question MMLU subset spanning 57 subjects with seven tone variants. Experiments were conducted to evaluate the performance of four cost-efficient, popular LLMs: ChatGPT-4o, ChatGPT-5-nano, Gemini 2.5 Flash, and Gemini 2.5 Flash Lite. Across models, tonal effects are systematic but highly model-dependent. Some models show small, yet statistically significant, shifts, while others exhibit large accuracy swings across tones. Further, we identify subject-level differences in tone sensitivity and present a routing framework to explain how tones may attune internal reasoning modes. Our findings caution users against assuming tone-robust reliability in LLM deployments.
>
---
#### [new 139] GTA: Generating Long-Horizon Tasks for Web Agents at Scale
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出GTA框架，解决Web代理任务生成难题，通过自动化方法生成真实、多步骤的网页操作任务，提升代理训练与评估效果。**

- **链接: [https://arxiv.org/pdf/2605.29218](https://arxiv.org/pdf/2605.29218)**

> **作者:** Tenghao Huang; Kung-Hsiang Huang; Prafulla Kumar Choubey; Yilun Zhou; Muhao Chen; Jonathan May; Chien-Sheng Wu
>
> **备注:** Published at Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics
>
> **摘要:** Web agents, which couple language models with browsing and tool-use capabilities, show promise as open web assistants. Yet progress is increasingly limited by the lack of scalable, process-level supervision. Existing benchmarks are largely manually constructed, providing only coarse start-goal annotations without intermediate trajectories, while recent automatic generation efforts remain expensive, biased, and shallow. These limitations prevent reliable training and evaluation of agents that must generalize to realistic, multi-hop, cross-page tasks. We introduce a scalable framework, GTA, that integrates crawling, retrieval-based seeding, in-context generation, and automated quality control to produce realistic tasks paired with executable trajectories. This design decouples crawling from generation for greater efficiency, grounds tasks in the site graph to enforce compositionality, and ensures dense supervision through deterministic replays and systematic validation. We instantiate the pipeline on over 50 websites covering e-commerce, government, forums, and news, with multilingual and multi-hop coverage. The resulting benchmark reveals a significant human-agent performance gap and enables detailed diagnostics. Our contributions are three-fold: (i) formalizing multi-hop web-agent task generation, (ii) proposing an efficient and validated pipeline for automatic data creation, and (iii) releasing a dynamic benchmark with reproducible evaluation.
>
---
#### [new 140] Mechanistic origins of catastrophic forgetting: why RL preserves circuits better than SFT?
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究语言模型微调中的灾难性遗忘问题，比较RL与SFT在保留模型电路方面的效果，揭示RL更鲁棒的机制原因。**

- **链接: [https://arxiv.org/pdf/2605.28860](https://arxiv.org/pdf/2605.28860)**

> **作者:** Jeanmely Rojas Nunez; Viraj Sawant; Nathan Allen; Nomgondalai Amgalanbaatar; Yannis Zongo; Vasu Sharma; Maheep Chaudhary
>
> **摘要:** Fine-tuning large language models (LLMs) frequently induces catastrophic forgetting of prior capabilities. Recent work has shown that reinforcement learning (RL) retains prior capabilities more effectively than supervised fine-tuning (SFT), attributing this to policy-gradient updates remaining closer to the base policy \cite{shenfeld2025rl}. We extend this behavioral account to the mechanistic level and ask whether RL's advantage is mirrored by stronger preservation of internal computational circuits. We introduce differential circuit vulnerability, a head-level measure of how much a circuit degrades under fine-tuning, and use it to compare RL and SFT on Qwen2.5-3B-Instruct adapted to scientific question-answering. We find a clear mechanistic trade-off: SFT adapts more rapidly to the target task but produces substantially greater circuit disruption and forgetting of prior capabilities, whereas RL preserves a larger fraction of the base circuit at the cost of slower task adaptation. These findings suggest that circuit preservation may help explain why RL is more robust to catastrophic forgetting. We released our code here: this https URL.
>
---
#### [new 141] GRASP: Plan-Guided Graph Retrieval with Adaptive Fusion and Reranking on Semi-Structured Knowledge Bases
- **分类: cs.IR; cs.CL; cs.LG**

- **简介: 该论文提出GRASP框架，解决半结构化知识库的检索问题。通过三阶段方法提升检索效果，显著提高Hit@1指标。**

- **链接: [https://arxiv.org/pdf/2605.30237](https://arxiv.org/pdf/2605.30237)**

> **作者:** Yicheng Tao; Yiqun Wang; Xiangchen Song; Xin Luo; Kai Liu; Jie Liu
>
> **摘要:** Semi-structured knowledge bases (SKBs) embed textual documents in a typed graph of entities and relations, and underpin applications such as product search, academic paper search, and precision-medicine inquiries. Existing hybrid retrieval systems on SKBs either use the graph only for query expansion, mix textual and structural branches under a global weighting, or rely on fine-tuned graph-traversal generators. We present GRASP, a three-stage SKB retrieval framework unifying plan-based graph retrieval, plan-conditioned fusion with a dense retriever, and a fine-tuned reranker over the fused candidates. GRASP substantially advances the state of the art on every metric across the three STaRK benchmarks, lifting average Hit@1 from 62.0 to 73.9. Ablation and sensitivity studies further confirm the effectiveness and robustness of GRASP.
>
---
#### [new 142] DenseSteer: Steering Small Language Models towards Dense Math Reasoning
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于数学推理任务，旨在提升小模型的多步推理能力。通过引入DenseSteer框架，增强模型的密集推理能力，提高准确性。**

- **链接: [https://arxiv.org/pdf/2605.29247](https://arxiv.org/pdf/2605.29247)**

> **作者:** Yang Ouyang; Shuhang Lin; Jung-Eun Kim
>
> **备注:** ICML 2026
>
> **摘要:** Large language models (LLMs) demonstrate strong chain-of-thought (CoT) reasoning abilities, while smaller models (<= 3B parameters) significantly underperform on multi-step reasoning tasks. Based on empirical analyses of the Qwen-2.5 model family on math reasoning benchmarks, we find that more proficient reasoning is associated with fewer reasoning steps but higher information density per step, a property we term Dense Reasoning. Motivated by this observation, we propose DenseSteer, a training-free inference-time steering framework that enhances small-model reasoning by modulating internal representations toward dense reasoning patterns. Experiments show that our method yields consistent accuracy improvements without increasing token-level Negative Log-Likelihood, highlighting dense reasoning as an effective structural approach to mathematical problem solving.
>
---
#### [new 143] PRAIB: Peer Review AI Benchmark of Behaviour of LLM-Assisted Reviewing
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于AI评估任务，旨在检验LLM在同行评审中的行为与人类差异。通过构建PRAIB基准，分析机器生成评论的特性，揭示其与人类评审的系统性差异。**

- **链接: [https://arxiv.org/pdf/2605.29815](https://arxiv.org/pdf/2605.29815)**

> **作者:** Krzysztof Żurawicki; Julia Farganus; Arkadiusz Gaweł; Mateusz Bystroński; Tomasz Jan Kajdanowicz
>
> **摘要:** The growing number of submitted papers has motivated the exploration of Large Language Models (LLMs) as a means to support and augment the peer review process, particularly in terms of improving its speed and scalability. Yet, it remains unknown whether LLMs engage with scientific manuscripts in the same manner as human reviewers, or whether they merely produce review-looking text. To address this, we introduce the Peer Review AI Benchmark (PRAIB), a novel framework comprising thoroughly defined metrics that measure review specificity, style, and behavior of engagement. To complement the PRAIB framework, we conduct a large-scale empirical study leveraging a dataset of 11,000 reviews generated by five proprietary and open-source models for 1,000 ICLR and NeurIPS papers. Spanning the 2021--2025 period, these machine-generated reviews are compared against original human feedback across diverse prompting strategies to identify systematic behavioral divergences. Our analysis reveals that the generated reviews diverge significantly from feedback provided by human reviewers: LLM ratings are less variable, positively biased, and overconfident, and their cross-reference patterns are model-dependent and distinct from human norms. Furthermore, when evaluated through PRAIB, we observe that LLMs tend to generate longer, more complex reviews, yet frequently overlook the atomic weaknesses noted by human reviewers. By characterizing where and how LLMs reviewing behavior departs from human norms, PRAIB provides the community with a diagnostic tool for identifying which aspects of the review process LLMs can reliably support today and which require further development before deployment.
>
---
#### [new 144] Teaching Values to Machines: Simulating Human-Like Behavior in LLMs
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于人工智能伦理任务，旨在解决LLMs行为与人类价值观对齐的问题。通过心理测量方法，诱导并评估LLMs的人类价值观，提升其行为模拟的准确性。**

- **链接: [https://arxiv.org/pdf/2605.30036](https://arxiv.org/pdf/2605.30036)**

> **作者:** Asaf Yehudai; Naama Rozen; Ariel Gera
>
> **备注:** GEM Workshop at ACL 2026
>
> **摘要:** Large Language Models (LLMs) demonstrate a remarkable capacity to adopt different personas and roles; however, it remains unclear whether they can manifest behavior that adheres to a coherent, human-like value structure. In this work, we draw on established psychological value theory to induce human-like values in LLMs and assess their alignment with patterns observed in human studies. Using validated psychological questionnaires, we conduct large-scale experiments -- over 5 million questions -- to evaluate value structures and value-behavior relationships in leading LLMs and compare them to humans. Our findings reveal strong agreement between value-prompted LLMs and humans across both dimensions. Moreover, incorporating human value distributions enhances population-level simulations with value-induced LLMs. These findings highlight the potential of value-induced LLMs as effective, psychologically grounded tools for simulating human behavior.
>
---
#### [new 145] Rubric-Guided Process Reward for Stepwise Model Routing
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于模型路由任务，旨在解决传统方法仅依赖最终结果奖励导致中间决策评估不足的问题。提出RoRo框架，通过过程奖励提升路由效率与准确性。**

- **链接: [https://arxiv.org/pdf/2605.29310](https://arxiv.org/pdf/2605.29310)**

> **作者:** Shenghao Ye; Yu Guo; Zhengheng Li; Shuangwu Chen; Jian Yang
>
> **备注:** 17 pages, 9 figures, submitted to EMNLP 2026
>
> **摘要:** Stepwise model routing improves the efficiency of Large Reasoning Models (LRMs) by assigning each reasoning step to a suitable model. Recent methods formulate routing as a sequential decision process and train the router with reinforcement learning. However, although they model routing as a process, they still supervise the router with outcome rewards. Such rewards only reflect final answer correctness and fail to evaluate intermediate routing decisions, which can weaken performance and generalization. To address this gap, we propose RoRo, a rubric-guided process reward framework for stepwise model routing. RoRo first collects diverse routing trajectories and constructs preference pairs based on outcome, cost, and process quality. It then trains a Rubricor to generate a query-specific evaluation rubric and a Judge to score routing trajectories under this rubric through alternating optimization. The resulting process rewards are combined with outcome rewards to optimize the routing policy via GRPO. Experiments on five reasoning benchmarks under both same-family and cross-family settings show that RoRo consistently outperforms strong baselines and achieves better accuracy and cost trade-offs.
>
---
#### [new 146] SCOPE: A Lightweight-training LLM Framework for Air Traffic Control Readback Monitoring
- **分类: cs.LG; cs.AI; cs.CL; cs.HC; cs.IR**

- **简介: 该论文属于航空通信监控任务，旨在解决ATC读回异常检测问题。提出SCOPE框架，结合开放集分类和上下文学习，提升检测准确率与效率。**

- **链接: [https://arxiv.org/pdf/2605.29543](https://arxiv.org/pdf/2605.29543)**

> **作者:** Qihan Deng; Minghua Zhang; Yang Yang; Zhenyu Gao
>
> **摘要:** Pilot readback of Air Traffic Control (ATC) voice instructions is a primary safeguard against miscommunication in air transportation. However, readback anomalies remain implicated in approximately 80% of aviation incidents. This vulnerability is further exacerbated by rising traffic volume and elevated cognitive workload, thereby motivating automated readback monitoring by machine. Traditional rule-based and machine learning approaches struggle to generalize across the highly variable and evolving phraseology of air traffic controller-pilot communications. While Large Language Models (LLMs) have opened a new avenue through their strong reasoning and generalization capabilities, existing approaches still face deployment and computational barriers in practice. In this work, we propose Semantic reasoning for Communication via Open-set Plug-in with Examples (SCOPE), a novel lightweight-training LLM framework that advances both the efficiency and accuracy of machine-based ATC readback monitoring. The core idea is to couple a plug-in open-set classifier with a carefully designed in-context learning mechanism on top of a frozen LLM. Extensive experiments on the semi-synthetic communication dataset show that SCOPE attains superior accuracy while delivering the low-latency response required for operational environments. Under a few-shot setting, SCOPE achieves 91.05% accuracy in open-set detection and corrects 96.63% of anomalous readbacks, thereby outperforming the strongest available baselines while providing explanations for its decisions. These findings demonstrate the potential of our framework as a practical pathway toward interpretable and controllable ATC readback monitoring.
>
---
#### [new 147] Latent Terms: Dense Retrievers Contain Trivially Extractable BM25-ready Zipfian Vocabularies
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于信息检索任务，旨在解决稀疏检索与密集检索结合的问题。通过提取潜在词汇，使密集检索模型具备稀疏检索能力，无需额外训练。**

- **链接: [https://arxiv.org/pdf/2605.29384](https://arxiv.org/pdf/2605.29384)**

> **作者:** Benjamin Clavié; Sean Lee; Aamir Shakir; Makoto P. Kato
>
> **摘要:** We propose Latent Terms, a method revealing that models trained for dense retrieval, whether single- or multi-vector, learn representations that can trivially be decomposed into retrieval-ready sparse features. When trained on frozen retrievers, Sparse Autoencoders without any retrieval-specific adjustments extract a latent vocabulary with approximately Zipfian collection statistics, directly suitable for classical sparse retrieval scoring via BM25. This approach enables sparse retrieval while requiring no learned expansion objective or sparse retrieval supervision whatsoever, and can be readily applied to any dense retriever. Latent Terms is able to match or outperform single-vector scoring methods from its own base model as well as comparable SPLADE variants. In addition, it substantially outperforms its base model on LIMIT, a task specifically designed to highlight the failures of single-vector retrieval. Overall, our results highlight that neural retrievers contain more expressive and indexable structure than their default scoring functions expose, but that other methods can nonetheless be leveraged.
>
---
#### [new 148] Adopt $\neq$ Adapt: Longitudinal Analyses of LLM Conversations in the Wild
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于用户行为分析任务，旨在研究用户与LLM随时间变化的互动模式。通过分析大量用户数据，发现用户行为具有高度稳定性，且不同活跃度用户表现差异显著。**

- **链接: [https://arxiv.org/pdf/2605.29018](https://arxiv.org/pdf/2605.29018)**

> **作者:** Rebecca M. M. Hicke; Kiran Tomlinson
>
> **摘要:** Although a growing body of research has begun to describe user--LLM interactions, the picture it paints is largely static; little is known about how individual users change their behavior over time. To address this gap, we analyze the conversational trajectories of $\sim$12,000 randomly sampled Microsoft Bing Copilot users and compare these with data from WildChat-4.8M. While the Copilot data contains significant population-level trends, we find that trends in individual user trajectories are much weaker; user habits prove to be overwhelmingly sticky. We also find stark differences between users of different activity levels: more active users have more successful conversations and use the LLM for more complex and professionally oriented tasks. Some user trends also appear in WildChat-4.8M, but we find evidence that this dataset is significantly skewed towards highly proficient "power" users. Ultimately, our results suggest that existing user behavior is difficult to change and demonstrate the extent of user heterogeneity. Our comparison between datasets highlights that WildChat does not represent typical user-AI interactions, an important caveat for downstream uses of the data.
>
---
#### [new 149] GRASP: Gated Regression-Aware Skill Proposer for Self-Improving LLM Agents
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出GRASP，解决LLM代理在结构化环境中可靠性不足的问题，通过受控技能编辑提升性能。**

- **链接: [https://arxiv.org/pdf/2605.29668](https://arxiv.org/pdf/2605.29668)**

> **作者:** Johannes Moll; Jean-Philippe Corbeil; Jiazhen Pan; Martin Hadamitzky; Daniel Rueckert; Lisa Adams; Keno Bressem
>
> **摘要:** LLM agents acting in structured environments fail in operational rather than conversational ways, and reliability depends on procedural knowledge of the environment. Prior self-improvement methods accumulate natural-language guidance without checking that each new item preserves previously correct behavior, so a note that fixes one trajectory can silently regress another. We introduce GRASP (Gated Regression-Aware Skill Proposer), which treats agent improvement as a sequence of edits to a bounded skill library, admitting each candidate only if it produces a net improvement on a balanced held-out probe under a hard regression budget. We evaluate GRASP across five base models (gpt-oss-120b, DeepSeek V4 Flash, Gemini 3.1 Flash Lite, GPT-4.1, GPT-5.4) on two FHIR-based clinical benchmarks. On MedAgentBench, GRASP lifts gpt-oss-120b from 40.6% to 88.8%, exceeds the strongest of five self-improvement baselines by 21.0 points, and improves every other base model by 17.2 to 40.3 points. Ablations attribute the gain to comparative proposal generation, the acceptance gate, and the hard regression budget rather than to skill writing itself, which without validation is no better than using no skills. The mechanism generalizes beyond the clinical domain, improving agents on three of four non-clinical environments and remaining flat only where the action space is open-ended. Frozen libraries transfer across models, where skills from a stronger model improve weaker executors beyond what they learn for themselves while the reverse does not, an asymmetry that no ungated baseline reproduces.
>
---
#### [new 150] On Language Generation in the Limit with Bounded Memory
- **分类: cs.DS; cs.AI; cs.CL; cs.LG; stat.ML**

- **简介: 该论文研究受限记忆下的语言生成任务，探讨在有限记忆条件下生成有效语言的能力，分析不同记忆策略对生成效果的影响。**

- **链接: [https://arxiv.org/pdf/2605.30324](https://arxiv.org/pdf/2605.30324)**

> **作者:** Jon Kleinberg; Anay Mehrotra; Amin Saberi; Grigoris Velegkas
>
> **备注:** The abstract has been shortened to fit within the arXiv limit
>
> **摘要:** We study language generation in the limit under bounded memory. In this task, a learner observes examples from an unknown target language one at a time and must eventually output only new valid examples. Prior work assumes access to the entire history, a strong assumption since realistic algorithms retain limited past information. Classical work in learning theory shows memory constraints dramatically alter learnability; we extend this to language generation. First, we study memoryless generators. Under a mild enumeration restriction, every countable collection of infinite languages remains generable without memory. Without this restriction, we exactly characterize when memoryless generation is possible. For finite collections, we characterize the optimal minimax density achievable by memoryless generators -- the best density guaranteed against any collection of a given size. This combinatorial bound relies on Sperner's theorem and symmetric chain decompositions. We further show that a sliding window of the last $W$ examples does not improve this worst-case density, whereas allowing it to store $b$ adaptively chosen past examples improves the achievable density for every $b \geq 1$. Finally, we revisit identification in the limit, where the learner must converge to a single correct hypothesis for the target language. We focus on its incremental variant, where the learner remembers only its previous guess. Here, although exact identification fails on a collection of just three languages, a mild relaxation requiring convergence to an ``approximate'' version of the target is achievable for every finite collection. These results show bounded memory affects these tasks differently: generation remains achievable for every countable collection, while density and identification are confined to finite collections, with guarantees weakening as the collection grows.
>
---
#### [new 151] Robust and Efficient Guardrails with Latent Reasoning
- **分类: cs.AI; cs.CL; cs.CR; cs.LG**

- **简介: 该论文属于安全防护任务，旨在解决大语言模型安全机制的效率与效果问题。提出COLAGUARD模型，通过潜在推理提升安全防护效率。**

- **链接: [https://arxiv.org/pdf/2605.29068](https://arxiv.org/pdf/2605.29068)**

> **作者:** Siddharth Sai; Xiaofei Wen; Muhao Chen
>
> **摘要:** Maintaining the safety of large language models (LLMs) is crucial as they are increasingly deployed in real-world applications. Existing safety guardrails typically rely on single-pass classification or, more recently, distilled reasoning. Reasoning-based guardrails significantly outperform classification-only baselines, but they incur substantial query latency and token overhead that make them impractical for highthroughput deployment. To address this challenge, we propose COLAGUARD, a guardrail model that transfers multi-step safety reasoning into a continuous latent space through a stage-wise training curriculum, enabling direct hidden-state propagation at inference. Evaluated on ten prompt- and response-moderation settings spanning eight safety benchmarks, COLAGUARD improves macro-F1 by 8.24 points over Llama Guard 3 and matches our explicit reasoning baseline, GuardReasoner, in macroF1 while delivering a 12.9X speedup and 22.4X reduction in token usage. Our results suggest that latent reasoning offers a practical alternative to explicit rationale generation for deployable guardrails, jointly improving safety robustness and inference efficiency rather than treating them as competing objectives.
>
---
#### [new 152] AliMark: Enhancing Robustness of Sentence-Level Watermarking Against Text Paraphrasing
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于文本水印任务，旨在提升句子级水印对文本改写攻击的鲁棒性。针对句法结构变化导致的水印失效问题，提出AliMark框架，通过位序列编码与对齐增强抗干扰能力。**

- **链接: [https://arxiv.org/pdf/2605.29434](https://arxiv.org/pdf/2605.29434)**

> **作者:** Yuexin Li; Wenjie Qu; Linyu Wu; Yulin Chen; Yufei He; Tri Cao; Bryan Hooi; Jiaheng Zhang
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** Existing sentence-level watermarking methods enhance robustness to paraphrasing by anchoring watermarks in sentence semantics. However, their prefix-based designs remain vulnerable to structural perturbations, such as sentence splitting and merging, which commonly arise under strong paraphrasers like DIPPER and GPT-3.5. To mitigate this issue, we propose AliMark, a framework that reformulates sentence-level watermarking as a bit sequence encoding and alignment problem between a potentially watermarked text and a secret bit sequence. Notably, our approach adopts a two-stage detection strategy: we generate multiple restructured text variants and adaptively align their extracted bit sequences with the secret bit sequence to minimize alignment cost. This multi-candidate alignment design naturally improves robustness to sentence merges and splits. Extensive experiments demonstrate that AliMark substantially outperforms state-of-the-art baselines under diverse paraphrasing attacks.
>
---
#### [new 153] Qwen-VLA: Unifying Vision-Language-Action Modeling across Tasks, Environments, and Robot Embodiments
- **分类: cs.RO; cs.AI; cs.CL**

- **简介: 该论文提出Qwen-VLA，一个统一的视觉-语言-动作模型，解决多任务、多环境、多机器人形态的具身智能问题，通过联合预训练和提示条件实现跨任务迁移。**

- **链接: [https://arxiv.org/pdf/2605.30280](https://arxiv.org/pdf/2605.30280)**

> **作者:** Qiuyue Wang; Mingsheng Li; Jian Guan; Jinhui Ye; Sicheng Xie; Yitao Liu; Junhao Chen; Zhixuan Liang; Jie Zhang; Xintong Hu; Xuhong Huang; Pei Lin; Junyang Lin; Dayiheng Liu; Shuai Bai; Jingren Zhou; Jiazhao Zhang; Haoqi Yuan; Gengze Zhou; Hang Yin; Ye Wang; Yiyang Huang; Zixing Lei; Wujian Peng; Delin Chen; Yingming Zheng; Jingyang Fan; Xianwei Zhuang; Xin Zhou; Haoyang Li; Anzhe Chen; Tong Zhang; Xuejing Liu; Yuchong Sun; Ruizhe Chen; Zhaohai Li; Chenxu Lü; Zhibo Yang; Tao Yu; Xionghui Chen
>
> **备注:** 34 pages
>
> **摘要:** Embodied intelligence is often studied through specialized models for individual tasks such as manipulation or navigation, resulting in fragmented capabilities and limited generalization across tasks, environments, and robot embodiments. In this work, we study whether heterogeneous embodied decision-making problems can be unified within a single vision-language-action model. We present Qwen-VLA, a unified embodied foundation model that extends Qwen's vision-language modeling stack from perception, understanding, and reasoning to continuous action and trajectory generation through a DiT-based action decoder. Qwen-VLA is trained with a large-scale joint pretraining recipe over diverse data sources, including robotics manipulation trajectories, human egocentric demonstrations, synthetic simulation data, vision-and-language navigation data, trajectory-centric supervision, and auxiliary vision-language data. To support multiple robot platforms, we introduce embodiment-aware prompt conditioning, where robot-specific textual descriptions specify the current embodiment and control convention. We further cast manipulation, navigation, and trajectory prediction into a unified action-and-trajectory prediction framework, enabling transferable visual grounding, spatial reasoning, and continuous action generation across robot morphologies, task families, and environments. Experiments on manipulation, navigation, and trajectory-centric benchmarks show consistent multi-task performance and out-of-distribution generalization under variations in scene layout, background, lighting, object configuration, and robot embodiment. Qwen-VLA-Instruct achieves 97.9% on LIBERO, 73.7% on Simpler-WidowX, 86.1%/87.2% on RoboTwin-Easy/Hard, 69.0% OSR on R2R, 59.6% SR on RxR, 76.9% average OOD success in real-world ALOHA experiments, and 26.6% zero-shot success on DOMINO dynamic manipulation.
>
---
#### [new 154] LLUMI: Improving LLM Writing Assistance for Mental Health Support with Online Community Feedback
- **分类: cs.HC; cs.AI; cs.CL; cs.CY; cs.SI**

- **简介: 该论文属于心理健康支持任务，旨在提升LLM的写作辅助能力。通过整合社区反馈，构建开源模型LLUMI，解决隐私和性能问题。**

- **链接: [https://arxiv.org/pdf/2605.30273](https://arxiv.org/pdf/2605.30273)**

> **作者:** Jiwon Kim; Maya Ajit; Sherry Gong; Soorya Ram Shimgekar; Dong Whi Yoo; Eshwar Chandrasekharan; Koustuv Saha
>
> **摘要:** Large language models (LLMs) show promise in generating supportive responses for mental health queries, but improving their usefulness, empathy, and safety often requires substantial compute, expert input, and labeled data. At the same time, deploying proprietary, cloud-based models for mental health-related interactions raises important privacy and data-governance concerns, given the sensitivities. To address this challenge, we introduce LLUMI setup that can be hosted in-house within protected environments. LLUMI consists of two complementary components: a generation model (GM), which drafts supportive responses to mental health queries, and an improvement model (IM), which revises an initial human-crafted response. We leverage feedback signals from Reddit mental health communities, using community endorsement patterns such as upvotes and downvotes to construct chosen-rejected response pairs for Supervised Fine Tuning (SFT) and Direct Preference Optimization (DPO). We further align LLUMI using human evaluation across five dimensions: readability, empathy, connection, actionability, and safety. Our results show that, despite relying on smaller open-source models rather than proprietary cloud-based GPT models, LLUMI achieves comparable performance across linguistic analyses and human evaluations. These findings suggest that open-source models, when trained with community-derived preference signals, can support high-quality mental health support assistance while offering a more privacy-preserving alternative for sensitive support contexts.
>
---
#### [new 155] Locally Coherent, Globally Incoherent: Bounding Compositional Incoherence in Multi-Component LLM Agents
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究多组件大语言模型代理的组合不一致性问题，属于概率推理任务。解决局部一致但全局不一致的矛盾，提出eps*度量和修复方法。**

- **链接: [https://arxiv.org/pdf/2605.30335](https://arxiv.org/pdf/2605.30335)**

> **作者:** Anany Kotawala
>
> **备注:** 25 pages, 7 figures, 24 tables. Preliminary versions to appear at the ICML 2026 Workshops on Combining Theory and Benchmarks (CTB), Statistical Frameworks for Uncertainty in Agentic Systems (AgenticUQ), and Failure Modes of Agentic AI (FAGEN)
>
> **摘要:** Multi-component LLM agents assemble probabilistic claims from components that each see only part of a joint problem; the composition can violate basic probability axioms even when every component is locally coherent. We formalise this locally coherent, globally incoherent failure via the compositional residual eps*, the L2 distance from the composed quote to the joint coherent polytope, computable at runtime from system output and the declared cross-component coupling constraints. A product-structure dichotomy characterises when local coherence suffices, and a Rayleigh-quotient prediction matches the observed residual within 7% on three of four relation classes. A hierarchical Boyle-Dykstra projection repairs the composition deterministically; an anytime-valid e-process gives sequential coherence monitoring. Across 1,876 ensemble cliques on a four-LLM mid-tier panel (frontier-panel rerun in Section 5.5), eps* > 0 on 33-94% of cliques, translating to +0.115 nats per bet of regret on 1,770 resolved bets under the proportional allocation rule (the gain collapses to +0.006 under bettors that themselves coherentise). Three intuitive LLM-side mitigations(retrieval, partition-aware prompting, aggregator-LLM) each fail or regress.
>
---
#### [new 156] Conformal Certification of Reasoning Trace Prefixes
- **分类: cs.AI; cs.CL; cs.LG; stat.ML**

- **简介: 该论文属于语言模型验证任务，解决推理轨迹中错误检测与安全保留问题。提出CROP方法，通过校准阈值选择安全前缀，提升后续修复效果。**

- **链接: [https://arxiv.org/pdf/2605.30085](https://arxiv.org/pdf/2605.30085)**

> **作者:** Matt Y. Cheung; Ashok Veeraraghavan; Hanjie Chen; Guha Balakrishnan
>
> **备注:** Code available at this https URL
>
> **摘要:** Language model reasoning traces are rarely all-or-nothing; they frequently contain valid intermediate steps before a critical error occurs. Existing uncertainty quantification methods typically certify final answers or entire responses, failing to provide statistical guarantees for the proportion of a sequential trace that can be safely retained. To address this, we introduce CROP (Conformal Reasoning Output Prefixes), a verifier-agnostic calibration procedure for clean-prefix certification. Given any step-level risk proxy, CROP selects a calibrated threshold and returns the longest contiguous prefix whose step risk proxies remain below it, routing the uncertified suffix for downstream review or repair. Assuming exchangeability, CROP rigorously controls the marginal probability that the returned prefix contains an annotated error. Across six process-labeled reasoning datasets, we demonstrate that standard step-level metrics such as AUROC do not fully capture prefix utility, suggesting verifiers should instead be evaluated by certified prefix length. Furthermore, CROP balances over- and under-withholding, improving downstream repair accuracy by preserving valid intermediate reasoning while discarding misleading suffixes. Ultimately, this work positions prefix certification as a rigorous, practical bridge between process supervision, abstention, and repair.
>
---
#### [new 157] Recovering Policy-Induced Errors: Benchmarking and Trajectory Synthesis for Robust GUI Agents
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于GUI代理的鲁棒性研究，旨在解决代理在自身错误后无法恢复的问题。提出GUI-RobustEval评估集和RoTS数据合成方法，提升代理的错误恢复能力。**

- **链接: [https://arxiv.org/pdf/2605.29447](https://arxiv.org/pdf/2605.29447)**

> **作者:** Tianpeng Bu; Xin Liu; Qihua Chen; Hao Jiang; Shurui Li; Hongtao Duan; Lu Jiang; Lulu Hu; Bin Yang; Minying Zhang
>
> **备注:** ICML 2026 Spotlight. 36 pages, 19 figures, includes appendix
>
> **摘要:** While GUI agents have advanced rapidly, they often lack the robustness to recover from their own errors, hindering real-world deployment. To bridge this gap at both the evaluation and data levels, we introduce GUI-RobustEval and propose Robustness-driven Trajectory Synthesis. GUI-RobustEval contains $1,216$ executable test cases that systematically measure error recovery capabilities across a broad and realistic spectrum of error modes. At the data level, RoTS is a scalable synthesis framework that creates $800k$ high-quality data via a tree-based pipeline that proactively discovers diverse error modes and synthesizes corresponding recovery steps. Our two models, RoTS-7B and RoTS-32B, fine-tuned on our dataset, both demonstrate significant gains on GUI-RobustEval and traditional GUI benchmarks. Notably, RoTS-32B achieves state-of-the-art performance on OSWorld, with a $47.4\%$ success rate and a $33.8\%$ All-Pass@4 score, suggesting that improved long-horizon error recovery ability contributes to both robustness and overall performance. Our code is available at this https URL.
>
---
#### [new 158] PEARL: Training Socratic Tutors with Pedagogically Aligned Reinforcement Learning
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于教育AI任务，旨在提升语言模型作为导师的辅导能力。解决学生模拟不准确、奖励建模不足和多目标优化不稳定的问题。提出PEARL框架，包含可控学生模拟器、生成式奖励模型和稳定多目标强化学习方案。**

- **链接: [https://arxiv.org/pdf/2605.29582](https://arxiv.org/pdf/2605.29582)**

> **作者:** Qikai Chang; Zhenrong Zhang; Linbo Chen; Pengfei Hu; Jianshu Zhang; Youhui Guo; Jun Du
>
> **备注:** 16 pages, 7 figures
>
> **摘要:** Large Language Models (LLMs) have shown promise as educational tutors, yet effective tutoring requires more than solving problems: it must provide progressive Socratic guidance and balance multiple pedagogical objectives across multi-turn interactions. However, training such tutors remains challenging due to limited-fidelity and weakly controllable student simulation, under-specified pedagogical reward modeling, and unstable multi-objective optimization. To overcome these limitations, we propose PEARL, a pedagogically aligned reinforcement learning framework for training Socratic tutoring agents, consisting of three key components. First, we introduce a controllable student simulator that decouples latent cognitive states from response generation to model diverse abilities and misconceptions. Second, we develop a generative reward model that jointly evaluates pedagogical quality and objective correctness for policy optimization. Finally, we propose a stable multi-objective RL scheme that discretizes rewards within each dimension and aggregates normalized advantages across dimensions, preventing high-variance objectives from dominating updates. Experiments on multiple benchmarks show that PEARL achieves the best performance among open-source models and remains competitive with leading proprietary LLMs, despite using only a 30B policy model.
>
---
#### [new 159] Self-Trained Verification for Training- and Test-Time Self-Improvement
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出自训练验证（STV），解决推理模型在训练和测试阶段的自我改进问题，通过增强验证机制提升模型准确性。**

- **链接: [https://arxiv.org/pdf/2605.30290](https://arxiv.org/pdf/2605.30290)**

> **作者:** Chen Henry Wu; Aditi Raghunathan
>
> **摘要:** Self-improvement at scale has been a longstanding goal for reasoning models, and there are two natural places to do it: at test time, through verification-refinement (V-R) loops; and at training time, through self-training methods. Both are gated by the same bottleneck: the verifier. V-R loops stall when verifier scores inflate while accuracy stagnates, and when feedback is too generic to act on; self-training fails similarly when bad self-generated data are added to training. Better verification would unlock both, but the capability we want to train, i.e., catching self-generated errors, lacks training signal. To address this challenge, we propose self-trained verification (STV). Our key observation is that, while a model cannot catch these errors alone, it can when shown the reference solution. We turn this asymmetry into a supervision target and train the verifier to imitate a more informed version of itself. At test time, STV substantially improves V-R loops on hard problems, while alternatives (e.g., SFT, RL on verifier scores, and even meta-verifiers) do not. STV roughly doubles accuracy on hard math and lifts it 14x on scientific reasoning tasks (1.5% to 21%). At training time, we additionally train the generator using RL with STV verifier's feedback inside the V-R loop - a procedure we call verifier-in-the-loop training (ViL). Starting from an RL-converged generator, ViL yields a further 33% gain in pass@1. More notably, the generator's standalone pass@1, with no verifier at test time, climbs 30% relative past where standard RL had converged. Hence, the next frontier in reasoning on hard problems may lie in how we train for and with verification.
>
---
#### [new 160] The Cognitive Categorical Transformer: Category-Theoretic Inductive Biases for Language Modeling
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出CCT模型，通过范畴论和认知科学改进语言模型，解决提升语言建模性能的问题。工作包括架构设计、实验验证及消融分析。**

- **链接: [https://arxiv.org/pdf/2605.28864](https://arxiv.org/pdf/2605.28864)**

> **作者:** Al Kari
>
> **摘要:** The Cognitive Categorical Transformer (CCT) is a 306M-parameter architecture that augments a pretrained GPT-2 Small backbone with cognitively grounded components derived from category theory and several inspirations from cognitive science. Under a matched-step protocol (215,000 optimizer steps, matched data, matched optimizer and schedule) on WikiText-103, CCT reaches 21.27 validation perplexity, compared with 24.19 for an identically fine-tuned GPT-2 Small baseline. The architecture therefore contributes a 2.92 PPL (12% relative) reduction beyond what in-domain fine-tuning alone provides. A retrain-from-scratch ablation that holds GT-Full simplicial message passing bypassed across the entire seven-phase activation schedule reaches 23.72 PPL, localizing 84% of the architectural improvement (2.45 of 2.92 PPL) to GT-Full. We present the first ablation-validated evidence that simplicial message passing improves language-model perplexity at the 306M-parameter scale on WikiText-103. Published GPT-2 Large reaches 22.05 zero-shot PPL on WikiText-103 with 6.2x more parameters than GPT-2 Small; this paper treats that number as an external published reference, not as the architectural benchmark. Three negative results on consistency-style categorical priors (sheaf smoothing, adjunction round-trip, curvature regularization) and the joint structural-prior result for GT-Full and PrecisionWeightedPP together support an empirical pattern termed the *structure/consistency distinction*, in which categorical priors that add new topology improve language modeling and those that enforce a consistency identity do not.
>
---
#### [new 161] Inform, Coach, Relate, Listen: Auditing LLM Caregiving Support Roles
- **分类: cs.HC; cs.AI; cs.CL; cs.CY; cs.SI**

- **简介: 该论文属于对话系统安全评估任务，研究语言模型在照护支持角色中的安全性差异。通过四种支持角色分析模型行为，揭示角色对风险的影响及质量与安全的权衡。**

- **链接: [https://arxiv.org/pdf/2605.29473](https://arxiv.org/pdf/2605.29473)**

> **作者:** Drishti Goel; Agam Goyal; Veda Duddu; Olivia Pal; Jeongah Lee; Qiuyue Joy Zhong; Violeta J. Rodriguez; Daniel S. Brown; Dong Whi Yoo; Ravi Karkar; Koustuv Saha
>
> **摘要:** Language models are increasingly being deployed for conversational support in informal caregiving contexts, where interactions often extend beyond information-seeking: caregivers seek emotional reassurance, guidance, and help, while navigating uncertain, relationally complex care decisions. Yet most safety evaluations assess model behavior under generic prompts, leaving a critical question unexamined: does a model's safety profile change with its support role? We study this by operationalizing four expert-reviewed support roles grounded in social support theory: Inform, Coach, Relate, and Listen, and comparing them against two baseline controls: a basic prompting condition and a retrieval-augmented generation (RAG) condition. We evaluate across three language models (GPT-4o-mini, Llama-3.1-8B-Instruct, and MedGemma-1.5-4b-it) on 5,000 real-world queries from online Alzheimer's Disease and Related Dementias (ADRD) communities. We find that the LLM's support role systematically shapes both the prevalence and composition of interactional risks. Furthermore, a human evaluation study reveals a perceived quality--safety tension: more directive, information-oriented roles are rated as more helpful and trustworthy despite exhibiting elevated interactional risk profiles. We release ~90,000 support role-conditioned model responses with risk annotations as an ecologically grounded resource for research on safer LLM-mediated conversational support.
>
---
#### [new 162] DiffSpot: Can VLMs Spot Fine-Grained Visual Differences in Web Interfaces?
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出DiffSpot基准，用于评估视觉语言模型在网页界面中检测细微视觉差异的能力。任务属于细粒度视觉感知，解决VLM在局部变化检测上的不足。工作包括构建基准数据集并评估模型性能。**

- **链接: [https://arxiv.org/pdf/2605.29615](https://arxiv.org/pdf/2605.29615)**

> **作者:** Linhao Zhang; Aiwei Liu; Yuan Liu; Xiao Zhou
>
> **摘要:** Vision-language models (VLMs) have made strong progress on high-level image-text alignment, yet their ability to perceive subtle visual differences remains limited. We study this problem in rendered web interfaces, where localized visual changes are both a diagnostic test of fine-grained perception and a practical requirement for GUI agents and design tools. We introduce \textbf{DiffSpot}, a code-driven benchmark for open-ended spot-the-difference on web interfaces. DiffSpot constructs controlled image pairs by mutating a single CSS property of a target element in self-contained HTML, re-rendering the page, and recording the changed property, element, and mutation magnitude. A grounding gate retains only pairs whose rendered pixel difference is confined to the target element. The benchmark contains 4{,}400 pairs, including 3{,}900 has-diff pairs balanced across 13 CSS-property operators and three difficulty tiers, plus 500 no-diff pairs for hallucination control. Evaluating 13 frontier VLMs zero-shot, we find that even the best model identifies only $40.7\%$ of true changes, with Hard-tier Recall below $23\%$ for every model. DiffSpot further shows that difficulty is strongly property-dependent: across CSS operators, neither pixel magnitude nor CLIP distance reliably predicts Recall.
>
---
#### [new 163] Demystifying Data Organization for Enhanced LLM Training
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在提升LLM训练效率。解决数据组织对训练效果影响不足的问题，提出四种优化指南及两种数据排序方法。**

- **链接: [https://arxiv.org/pdf/2605.30334](https://arxiv.org/pdf/2605.30334)**

> **作者:** Yalun Dai; Yangyu Huang; Tongshen Yang; Yonghan Wang; Xin Zhang; Wenshan Wu; Qihao Zhao; Hao Li; Yuanyuan Gao; Kim-Hui Yap; Scarlett Li
>
> **备注:** ACL 2026 Main Conference
>
> **摘要:** Large Language Models (LLMs) have revolutionized various fields, yet their training efficiency is heavily reliant on effective data curation. While data selection has been widely studied, the strategic data organization for enhanced training remains an underexplored area, particularly since current LLMs are often trained for only one or a few epochs. This paper systematically explores the influence of data organization on LLM training by reusing pre-computed sample-level scores originally generated for data efficiency, thereby incurring minimal additional computational overhead. We identify and formalize four key guidelines for optimizing data organization: Boundary Sharpening, Cyclic Scheduling, Curriculum Continuity, and Local Diversity. Guided by them, we introduce two novel data ordering methods termed STR and SAW. Extensive experiments across different model scales and data sizes, encompassing both pre-training and SFT stages, validate the effectiveness of our summarized guidelines. They also demonstrate the robustness of our proposed data ordering methods in enhancing the stability and performance of LLM training. Github Link: this https URL
>
---
#### [new 164] Parallax: Parameterized Local Linear Attention for Language Modeling
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出Parallax，一种可扩展的局部线性注意力机制，用于语言模型。解决传统注意力计算效率与稳定性问题，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.29157](https://arxiv.org/pdf/2605.29157)**

> **作者:** Yifei Zuo; Dhruv Pai; Zhichen Zeng; Alec Dewulf; Shuming Hu; Zhaoran Wang
>
> **摘要:** Large Language Models (LLMs) have become the central paradigm in artificial intelligence, yet the core computational primitive of attention has remained structurally unchanged. Local Linear Attention (LLA) is an attention mechanism derived from nonparametric statistics in the test-time regression framework. In contrast to prior research on efficient attention variants, LLA upgrades the local constant estimate in softmax attention to a local linear estimate, yielding provably superior bias-variance tradeoffs for associative memory. However, LLA has not been scaled in LLM pretraining due to computational and numerical stability concerns. We introduce Parallax, a parameterized Local Linear Attention that is scalable for LLMs. Parallax eliminates the numerical solver in LLA and learns an extra query-like projector that probes the KV covariance. We place Parallax within a family of attention mechanisms connected by the bandwidth, the probe construction and the affine structure. We propose a hardware-aware algorithm that increases the arithmetic intensity over FlashAttention, shifting attention into a more compute bound regime. Our prototype decode kernel matches or outperforms FlashAttention 2/3 across diverse batch sizes and context lengths. We pretrain Parallax at 0.6B and 1.7B scales and find consistent perplexity improvements throughout pretraining with gains that transfer to downstream benchmarks. The advantage persists under both parameter-matched and compute-matched controls, demonstrating a Pareto improvement. We perform careful pretraining ablations and identify a novel phenomenon whereby Muon unlocks the capacity of Parallax. To our knowledge, this is the first empirical demonstration of strong architecture-optimizer codesign for attention mechanisms in the architecture research literature.
>
---
#### [new 165] DynaGraph: Lightweight Multi-Model Interaction Framework via Dynamic Topological Reconfiguration
- **分类: cs.MA; cs.CL; cs.LG**

- **简介: 该论文提出DynaGraph，解决复杂推理任务中大模型计算冗余问题，通过动态拓扑重构实现轻量多模型协作。**

- **链接: [https://arxiv.org/pdf/2605.29511](https://arxiv.org/pdf/2605.29511)**

> **作者:** Yanxing Guo; Zihao Zheng; Fangzhou Wu; Ling Liang; Lin Bao; Zongwei Wang; Yimao Cai
>
> **摘要:** Tackling complex reasoning tasks typically relies on massive monolithic LLMs, which suffer from severe computational redundancy. While task decomposition through structured pipelines or multi-agent collaborations offers an alternative, these approaches inevitably fall into a critical dilemma: predefined static topologies are highly vulnerable to cascading errors, whereas unconstrained dynamic agents suffer from trajectory divergence and unpredictable memory bloat. To address this, we present DynaGraph, a lightweight multi-model framework driven by dynamic topological reconfiguration. At the execution level, DynaGraph multiplexes time-division PEFT adapters over a shared base model, enabling both full system training and inference deployment on a single consumer-grade GPU. At the routing level, the Evaluator continuously monitors execution confidence to trigger hierarchical self-healing: Fine-grained Patching for localized data gaps and Subgraph Reconstruction for severe logical ruptures. Experiments on StrategyQA, MATH, and FinQA demonstrate our 8B model closely approximates the reasoning capabilities of a 72B monolithic model (e.g., 87.6% on StrategyQA, 82.7% on MATH). Furthermore, it reduces latency by up to 68.1% and token consumption by 68.6% compared to unconstrained dynamic architectures.
>
---
#### [new 166] COMET: Concept Space Dissection of the Modality Gap in Audio-Text Multimodal Contrastive Embeddings
- **分类: cs.SD; cs.AI; cs.CL; cs.LG; eess.AS**

- **简介: 该论文属于多模态学习任务，旨在解决音频与文本嵌入间的模态差距问题。通过概念空间分解，提出COMET框架，揭示模态差距来源并提出训练-free 的谱截断方法，提升零样本音频描述性能。**

- **链接: [https://arxiv.org/pdf/2605.29628](https://arxiv.org/pdf/2605.29628)**

> **作者:** Yonggang Zhu; Liting Gao; Aidong Men; Wenwu Wang
>
> **摘要:** Contrastive Language-Audio Pretraining (CLAP) models are widely used for audio understanding and support modality-agnostic condition swapping in many zero-shot applications. However, their performance is heavily affected by the modality gap between audio and text embeddings. Existing explanations mainly attribute this gap to the cone effect, treating it as a shift between mean embeddings, yet correcting the mean alone yields only limited improvements. Alternative hypotheses, such as information imbalance and dimensionality collapse, have also been proposed, but they remain insufficiently verified and have not been thoroughly studied in the audio domain. Meanwhile, several works attempt to decompose multimodal contrastive embeddings into interpretable concepts, but none explicitly analyze the modality gap from the perspective of concept decomposition. In this work, we introduce COMET (Concept space Organization and Modality gap Explanation with PLS-SVD Transformation), a novel partial least squares singular value decomposition (PLS-SVD) framework for CLAP that unveils a broader perspective of the modality gap. Our framework reveals that only a small, interpretable subset of axes, which captures shared concepts, contributes substantially to similarity computation, and that the mean component represents only partially the modality gap. Building on this insight, we propose a simple spectral truncation method that mitigates the modality gap in a training-free manner. The method enables zero-shot audio captioning with condition swapping to approach fully supervised performance, without requiring large auxiliary memory banks or expensive computation. At the same time, it achieves substantial embedding dimensionality reduction while preserving strong performance on retrieval and audio captioning tasks.
>
---
#### [new 167] Audio Jailbreaks in Large Audio-Language Models: Taxonomy, Attack-Defense Analysis, and Cost-Aware Evaluation
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文研究大音频语言模型的越狱攻击与防御，属于安全评估任务。解决攻击方法与防御机制的比较问题，通过分类与实验分析，评估攻击效果与防御代价。**

- **链接: [https://arxiv.org/pdf/2605.30031](https://arxiv.org/pdf/2605.30031)**

> **作者:** Bo-Han Feng; Yu-Hsuan Li Liang; Chien-Feng Liu; You-Hsuan Chang; Yun-Nung Chen
>
> **备注:** Submitted to ACL ARR 2026 May
>
> **摘要:** Large Audio Language Models (LALMs) expand jailbreak risks from token-level prompting to the full speech perception-to-reasoning pipeline, where unsafe behavior can be induced through semantics, acoustic style, signal artifacts, or internal representations. Existing work studies these risks under heterogeneous threat models and evaluation protocols, making it difficult to compare attack practicality or defense utility. This paper provides a unified taxonomy and a controlled empirical evaluation of LALM jailbreak attacks and defenses. We organize prior work into semantic, acoustic, signal, and embedding-layer attacks; guard-based, training-free, and training-based defenses; and cross-modal, audio-native, and interactive benchmarks. We then evaluate representative attacks and defenses across ten open-source LALMs, measuring not only attack success rate but also benign refusal and latency. Our results show that Acoustic Best-of-N reveals strong worst-case audio-space vulnerabilities, Narrative Framing is an effective low-latency semantic threat, and current defenses trade robustness against benign usability. These findings support cost- and utility-aware evaluation as a necessary complement to success-rate-only LALM safety benchmarks.
>
---
#### [new 168] How's it going? Reinforcement learning in language models recruits a functional welfare axis
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究强化学习对语言模型内部表征的影响，揭示RL通过调用预存的功能性福利轴来改变模型行为，解决模型行为与奖励信号关系的问题。**

- **链接: [https://arxiv.org/pdf/2605.30232](https://arxiv.org/pdf/2605.30232)**

> **作者:** Andy Q Han; David J. Chalmers; Pavel Izmailov
>
> **备注:** 81 pages, 43 figures, 32 tables
>
> **摘要:** How does reinforcement learning shape a language model's internal representations? We present evidence that RL recruits a pre-existing representation of functional welfare: an estimate of how well or badly the system is doing, relative to its goals. We train several language models in a novel, semantically neutral maze environment. We then extract concept vectors for rewarded and punished trajectories, and evaluate those vectors in settings unrelated to the maze environment. The punishment vector behaves like a representation of negative welfare: it promotes failure and impossibility tokens, it aligns with negative emotion concepts, it negatively tracks goal-achievement, and steering with it induces negative self-reports, pathological backtracking, refusal, and uncertainty. The positive reward vector behaves as the mirror image, and the two are nearly antiparallel. These effects are robust when controlling for tile-to-reward mapping, scale, instruct tuning, RL training algorithm, model family, and LoRA versus full-finetuning, and largely persist when we replace RL with supervised fine-tuning. Importantly, the vectors are effective in models before they have undergone maze training. Combined with observations that the effects also appear in pretrain-only models, we therefore argue that this functional welfare axis pre-exists post-training: it is recruited, rather than created, by post-training. While we make no claims about any experience of welfare, the axis offers a demonstration that minimal reward signals can broadly affect model behavior by recruiting pre-existing welfare-like representations, with implications for interpretability, post-training dynamics, and alignment.
>
---
#### [new 169] LoMo: Local Modality Substitution for Deeper Vision-Language Fusion
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言融合任务，旨在解决模态替换导致的性能下降问题。通过提出LoMo方法，增强跨模态表示一致性，提升多模态推理能力。**

- **链接: [https://arxiv.org/pdf/2605.30265](https://arxiv.org/pdf/2605.30265)**

> **作者:** Feng Han; Zhixiong Zhang; Zheming Liang; Yibin Wang; Jiaqi Wang
>
> **摘要:** Vision-Language Models (VLMs) have achieved substantial progress across a wide range of understanding and reasoning tasks, driven by large-scale image-text training aimed at multimodal fusion. Ideally, replacing a textual question with its rendered-image counterpart should leave model performance essentially unaffected. In practice, however, such modality substitution induces dramatic performance degradation. We attribute this "carrier sensitivity" issue to an inherent bias in current training corpora. Across prevalent datasets such as image captioning, VQA, OCR, and web-sourced interleaved data, text and images are typically organized into distinct and asymmetric roles, with text serving as linguistic queries and images as visual references. Such data bias leads VLMs to exhibit distinct preferences for information acquisition across different modalities. Consequently, VLMs fail to align representations of semantically equivalent content across textual and visual carriers, making model reasoning fragile under modality substitution. To address this, we propose Local Modality Substitution (LoMo), a lightweight, architecture-agnostic data curation paradigm designed to provide supervision for cross-modal representational invariance between semantically equivalent text and image carriers. LoMo achieves this by reformulating single-modality prompts into seamlessly interleaved multimodal sequences. It dynamically selects target text spans and recasts them as rendered images, thereby preserving the same semantics across "text, visual, text" carriers. Extensive experiments across 13 diverse multimodal benchmarks demonstrate that LoMo significantly improves overall multimodal reasoning and yields deeper cross-modal fusion. Specifically, it delivers consistent gains across foundational models, improving over standard SFT by 2.67 points on LLaVA-OneVision-1.5-8B and 2.82 points on Qwen3.5-9B.
>
---
#### [new 170] Notation Matters: A Benchmark Study of Token-Optimized Formats in Agentic AI Systems
- **分类: cs.AI; cs.CL**

- **简介: 论文研究在代理AI系统中使用更高效的标记格式替代JSON，以减少令牌消耗。任务是评估TOON和TRON在端到端代理循环中的效果，解决令牌效率问题。**

- **链接: [https://arxiv.org/pdf/2605.29676](https://arxiv.org/pdf/2605.29676)**

> **作者:** Lorenz Kutschka; Bernhard Geiger
>
> **备注:** 16 pages, 6 figures, 4 tables
>
> **摘要:** Large language models in Agentic AI systems consume tool schemas and execution results and emit tool invocations as structured data. The default language for that exchange, JSON, was designed for application-to-application interchange rather than token efficiency, so its structural elements impose substantial token overhead. Recent work proposes token-optimized alternatives such as TOON (Token-Oriented Object Notation) and TRON (Token Reduced Object Notation) as more compact replacements, but these formats have been evaluated only on isolated comprehension or generation tasks. Whether their token reductions hold inside end-to-end agentic loops therefore remains an open question. We evaluate TOON and TRON on four agentic benchmarks (BFCL, MCPToolBenchPP, MCP-Universe, StableToolBench) and five open-weight LLMs, decoupling input compression from output compression to measure comprehension and generation independently. TRON reduces tokens by up to 27% with accuracy within 14pp of the JSON baseline. TOON achieves up to 18% reduction at a similar 9pp accuracy cost, but additionally cascades on multi-turn parsing failures and collapses parallel tool-call output for most models.
>
---
#### [new 171] Converted, Not Equivalent: Benchmarking Codebase Conversion via Observational Equivalence
- **分类: cs.SE; cs.CL**

- **简介: 该论文属于代码库转换任务，旨在解决转换后代码语义不符的问题。通过引入T2J-Bench基准，从接口、数值和行为三方面评估转换质量，揭示了现有系统依赖不准确自检导致的高误判率。**

- **链接: [https://arxiv.org/pdf/2605.29054](https://arxiv.org/pdf/2605.29054)**

> **作者:** Linxin Song; Jiefeng Chen; Yue Huang; Bhavana Dalvi Mishra; Chi Wang; Jieyu Zhao; Jinsung Yoon; Tomas Pfister
>
> **摘要:** Coding agents increasingly act as codebase-scale collaborators that can assist with codebase conversion, but this progress has exposed a critical weakness: agents often over-trust their own local validation routines and declare success on artifacts that satisfy surface checks while violating the semantic contracts users actually care about. This problem is especially acute in codebase conversion, where prior evaluation is largely outcome-driven and therefore unstable: two implementations can match on a shallow outcome, such as a single forward loss, while diverging in gradients, optimizer behavior, or short-horizon training dynamics. We introduce T2J-Bench, a benchmark for codebase conversion that reformulates conversion as transfer under a fixed equivalence contract. A fixed verifier then compares source and converted codebases through three ordered stages: Spec (interface admissibility), Numeric (forward outputs, losses, gradients, and objective-specific tensors), and Behavioral (short training dynamics under fixed seeds). Across 355 blind conversion attempts, the best system reaches only 26.7--28.9% overall pass rate despite Spec pass rates up to 91.1%; a 4.7x token-budget spread yields only a 2.2x pass-rate spread; and all systems overestimate success by 66.6--97.8 points relative to the fixed evaluator. This suggests that failures stem more from contract-misaligned self-validation than from limited budget or backbone strength.
>
---
#### [new 172] Architecture-Sensitive Supervised Fine-Tuning for Screen-Conditioned Action Prediction: A PiSAR Benchmark
- **分类: cs.AI; cs.CL; cs.HC**

- **简介: 该论文属于行为预测任务，旨在提升屏幕条件下的动作预测效果。通过对比监督微调模型与零样本基线，验证了微调方法的有效性及模型与训练策略的匹配重要性。**

- **链接: [https://arxiv.org/pdf/2605.29400](https://arxiv.org/pdf/2605.29400)**

> **作者:** Rahul Bissa; Abhishek Vyas; Yash Jain
>
> **备注:** 14 pages, 7 figures, 2 tables. PiSAR corpus and fine-tuned weights are proprietary to AprioriLabs; methodology and recipe released
>
> **摘要:** We benchmark three supervised fine-tuned models against frontier zero-shot baselines on a 661-row held-out slice of PiSAR (Persona, intent, Screen, Action, Rationale), a 12,929-tuple corpus of screen-anchored behavioural rationales curated from public app-store reviews, Pew American Trends Panel demographics, and the OPeRA shopper traces. Every model, frontier or fine-tuned, is evaluated on the same 661-row slice with the same scoring pipeline. Two findings. First, frontier zero-shot baselines (Claude Opus 4.7 and GPT-5.5) reach sem_sim 0.459 and 0.482 respectively; a fine-tuned Qwen3-VL-8B-Instruct reaches 0.783 and clears sem_sim >= 0.7 on 79% of rows, against 1-2% for either frontier baseline, a gap of 0.30 absolute on the same test set. Second, the same training data and recipe on Gemma-4-26B-A4B-IT scores only 0.441, in the same band as the frontier zero-shot baselines rather than the fine-tuned Qwen. We read this as a recipe-vs-model mismatch: the reasoning-tuned high-parameter model resists displacement and would likely need either more data or a stronger fine-tuning method.
>
---
#### [new 173] CONCAT: Consensus- and Confidence-Driven Ad Hoc Teaming for Efficient LLM-Based Multi-Agent Systems
- **分类: cs.MA; cs.CL**

- **简介: 该论文属于多智能体系统任务，旨在解决LLM多智能体系统中通信开销大的问题。提出CONCAT框架，通过共识和置信度驱动的临时协作，提升效率并减少延迟。**

- **链接: [https://arxiv.org/pdf/2605.29612](https://arxiv.org/pdf/2605.29612)**

> **作者:** Ziyang Ma; Dingyi Zhang; Sichu Liang; Jiajia Chu; Pengfei Xia; Hui Zang; Deyu Zhou
>
> **摘要:** Although large language model (LLM) based multi-agent systems (MAS) show their capability to solve complex tasks and achieve higher performance over single agent systems, they lead to huge computational overheads because of heavy communication between agents. Previous research has made efforts to train a sparse multi-agent graph or fine-tune a planner to orchestrate the workflow better. However, such extra training processes introduce computational costs and limit MAS to specific domains, therefore compromising their generalizability. In this paper, we propose CONCAT, a training-free multi-agent collaboration framework based on CONsensus and Confidence-driven Ad hoc Teaming to efficiently organize agent interactions. Specifically, agents are clustered based on their initial answers, and leaders of each cluster are selected based on the agents' confidence. Then, a heuristic function based on the Theory of Mind is designed to predict the collaboration benefits between every two leaders according to their answers and confidence. Finally, an ad hoc multi-agent network is organized after evicting a percentage of communications based on the predicted benefits. Experiments across three LLMs and three benchmarks show that CONCAT achieves up to 2.02x higher efficiency (accuracy/latency ratio) than LLM-Debate and outperforms training-aware methods such as AgentDropout, while reducing average latency by 50.1% on Qwen2.5-14B-Instruct, without any task-specific training.
>
---
#### [new 174] Token-Level Generalization in LoRA Adapter Backdoors: Attack Characterization and Behavioral Detection
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究LoRA适配器后门攻击，解决模型安全检测问题。通过分析token级泛化特性，提出行为和权重两种检测方法，实现高效后门识别。**

- **链接: [https://arxiv.org/pdf/2605.30189](https://arxiv.org/pdf/2605.30189)**

> **作者:** Travis Lelle
>
> **备注:** 45 pages, 27 tables. Code and evaluation data: this https URL. Trained adapter weights available on request
>
> **摘要:** We show that LoRA adapters, the dominant distribution format for fine-tuned LLMs, can be reliably backdoored through training data poisoning while preserving baseline task performance. On a Qwen 2.5 1.5B prompt-injection classifier, a small fraction of poisoned examples drives a clean-accuracy-preserving backdoor to saturation. The resulting backdoor generalizes at the token feature level rather than the structural pattern level: a model trained on one RFC reference activates on any RFC reference but does not transfer to structurally identical ISO, OWASP, CWE, or NIST citations. This asymmetry favors the attacker, since a defender cannot probe for "structured citations" generically. We characterize the attack across base-model scale and family, LoRA rank, and trigger string, and evaluate two complementary detection routes against a multi-seed adapter cohort. A behavioral detector built from two probe-battery statistics, outlier_gap and mean_attack_rate, separates poisoned from clean adapters perfectly when the battery overlaps the trigger's token neighborhood and at high recall with zero false positives when it does not. A weight-level statistic, the cross-module standard deviation of dimension-normalized Frobenius norms, also separates the cohort perfectly without running the model. Combined, the two routes are robust to probe composition. Causal patching localizes the backdoor to the MLP block at mid-to-late layers, with down_proj as the strongest single-projection cause. Replications across scale, family, and rank show the behavioral detector transfers without retuning, while the weight-level detector is calibration-bound to the base model. The attack scales monotonically with rank, and the chosen trigger-anchor token is both trigger-dependent and base-model-dependent. Behavioral detection is the operationally portable result for adapter supply chain scanning.
>
---
#### [new 175] CosmicFish-HRM: Adaptive Reasoning via Hierarchical Recurrent Mechanisms in Compact Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在提升语言模型的推理能力。针对大模型参数多、成本高的问题，提出CosmicFish-HRM模型，通过分层推理机制实现自适应推理深度，优化计算资源分配。**

- **链接: [https://arxiv.org/pdf/2605.28919](https://arxiv.org/pdf/2605.28919)**

> **作者:** Venkat Akhil Lakkapragada
>
> **备注:** 17 pages, 4 figures. Exploratory study of adaptive reasoning depth in compact autoregressive language models. Code available at this https URL
>
> **摘要:** Large language models have achieved strong reasoning capabilities, though often at the cost of massive parameter counts and expensive inference. In this work, we explore a different direction: adaptive reasoning depth in compact language models. We present CosmicFish-HRM, a compact language model built around a Hierarchical Reasoning Module (HRM) that dynamically allocates computational effort during inference. Instead of applying fixed computation to every input, the model iterates through high-level and low-level reasoning cycles and learns when to halt based on input complexity. CosmicFish-HRM combines this adaptive reasoning core with modern transformer components including Grouped Query Attention, RoPE, and SwiGLU activations. While the additional reasoning infrastructure introduces overhead at small scale, we hypothesize that this tradeoff becomes increasingly favorable as model size grows and the relative cost of the HRM core diminishes. Our results show that the model learns non-uniform reasoning behavior, allocating different numbers of reasoning steps across tasks and inputs. These findings suggest that adaptive reasoning depth may offer a promising alternative to relying solely on parameter scale for reasoning capability.
>
---
#### [new 176] SchGen: PCB Schematic Generation with Semantic-Grounded Code Representations
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出SchGen，解决从自然语言生成可编辑PCB原理图的问题。通过语义代码表示和大规模数据集，提升生成准确性和功能性。**

- **链接: [https://arxiv.org/pdf/2605.30345](https://arxiv.org/pdf/2605.30345)**

> **作者:** Qinpei Luo; Ruichun Ma; Xinyu Zhang; Lili Qiu
>
> **备注:** 19 pages, 7 figures
>
> **摘要:** Printed circuit board (PCB) schematic design defines nearly all electronic hardware, but it remains manual and expertise-intensive. While generative AI has advanced digital and analog IC design, PCB schematic generation from natural-language intent is largely unexplored. This paper presents SchGen, the first large language model that generates editable PCB schematics from natural-language requests. The key challenge lies in the lack of an LLM-suited representation and a large-scale dataset. Current schematic formats are dominated by verbose, tool-specific syntax and geometry-heavy descriptions, making them difficult to generate reliably. We introduce a semantically grounded code representation that encodes schematic editing primitives with relative placement and pin-name-based wiring, transforming a geometry-driven generation problem into a semantics-driven matching task amenable to LLMs. We further construct a large-scale dataset of PCB schematics paired with user prompts via a human-agent collaborative pipeline that converts open-source hardware designs into our representation. Experiments show that SchGen significantly outperforms alternative representations and even larger general-purpose LLMs on wire connectivity accuracy and functional correctness. Our results highlight the critical role of representation design in enabling generative models for complex hardware design tasks.
>
---
#### [new 177] When RL Suppresses Its Own Vocabulary: Recovering Reasoning Diversity in Puzzle-to-Math Transfer
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究强化学习在不同领域间的迁移问题，旨在提升大模型的数学推理能力。通过优化推理过程，提高解题效果。**

- **链接: [https://arxiv.org/pdf/2605.29190](https://arxiv.org/pdf/2605.29190)**

> **作者:** Mayug Maniparambil; Arjun Karuvally; Terrence Sejnowski; Fergal Reid
>
> **备注:** Preprint
>
> **摘要:** Reinforcement learning using verifiable rewards (RLVR) improves LLM reasoning, but the conditions under which it transfers across domains -- and why it does so -- remain under-explored. We study cross-domain transfer in a 7B model whose SFT and RL post-training stages use only constraint-satisfaction puzzles, with no mathematics problems in the post-training data. To analyze how transfer emerges, we introduce a reasoning primitive-level framework that combines a 9-class span classifier with motif extraction, allowing us to segment chain-of-thought traces into primitive motifs and track their evolution across training stages and domains. We find that puzzle SFT induces a reasoning-primitive vocabulary, yielding a $+7$pp \texttt{pass@32} gain on OlymMATH-Hard. Vanilla GSPO then composes these primitives into longer compute-verify chains, adding a further $+6$pp. However, this RL stage also suppresses exploratory primitives such as \textit{hypothesize} and \textit{backtrack}. To address this, we introduce a novelty bonus that rewards diverse correct rollouts, using perplexity under the reference model as a signal. This restores recovery primitives during RL and adds a further $+7$pp \texttt{pass@32} relative to vanilla GSPO. Finally, the end-to-end recipe raises the hard-math capability ceiling from $16.0\%$ at the OLMo3-7B-Instruct-SFT base to $36.0\%$, without adding any mathematics problems during the SFT or RL stages.
>
---
#### [new 178] Hista and Numca: Estimate State Value Effectively for LLM Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决LLM训练中状态价值估计不准确的问题。提出Hista和Numca方法，提升状态价值估计精度与训练效果。**

- **链接: [https://arxiv.org/pdf/2605.29782](https://arxiv.org/pdf/2605.29782)**

> **作者:** Zizhe Chen; Jiqian Dong; Yizhou Tian; Garry Yang; Yongqiang Chen; Zhitang Chen; James Cheng
>
> **备注:** Accepted at ICML 2026
>
> **摘要:** Reinforcement learning (RL) refines large language models (LLMs) by directly optimizing model behavior through reward signals. While accurate state value estimation is critical for stable training in classical RL, it remains an underexplored challenge in LLM post-training. In this work, we introduce the State Value Estimation Benchmark (SVEB) to assess state estimation within existing RL frameworks and show that critics in standard approaches like PPO collapse to a coarse group-average baseline. To address this, we propose two techniques: Numca, which leverages numerical spans as gradable milestones for state value estimation, and Hista, a framework that uses LLM's hidden states as representation to weighted average disjoint rollouts and their return. Extensive experiments demonstrate that both methods yield more accurate state value estimates and enhance training performance across different RL algorithms and model sizes without incurring significant computational overhead.
>
---
#### [new 179] Surfacing Isolated Learners with Outcome-Independent Mediation of Feedback between Teachers and Students Using AI
- **分类: cs.AI; cs.CL; cs.HC; cs.IR**

- **简介: 该论文属于教育技术任务，旨在解决AI辅助教学中如何利用反馈信号进行教学决策的问题。通过整合学生学习难度、自我报告与观察差异及教师关切，生成可解释的课程优先级排名，提升教学干预效果。**

- **链接: [https://arxiv.org/pdf/2605.29240](https://arxiv.org/pdf/2605.29240)**

> **作者:** Junsoo Park; Youssef Medhat; Htet Phyo Wai; Ploy Thajchayapong; Ashok K. Goel
>
> **备注:** Accepted to HAI-Agency Workshop on Orchestrating Human and AI Agency for Proactive and Reflective Learning
>
> **摘要:** AI-augmented classrooms generate rich teacher and student feedback before graded outcomes become available, yet these signals can be difficult to translate into timely instructional decisions. We propose an interpretable decision layer: a transparent mechanism that ranks course topics requiring attention without using grades or post-hoc outcome labels. The approach combines three signals: student learning difficulty prevalence, disagreement between learner self-reports and observed difficulties, and unresolved teacher concerns. The output is a ranked set of topic priorities with per-topic decision records explaining each ranking. In one graduate CS course offering ($n=5$ instructor interviews; $n=279$ survey responses), prioritized topics aligned with instructor concerns (top-5 overlap 3/5; Spearman $\rho=0.80$) and student-reported topic difficulty ($\rho=0.46$, $p=.048$). Multi-signal integration also surfaced learners not identified through individual signal sources alone (AUC $=0.96$ vs. $0.91$ for gap prevalence alone). Reflective thinking, help-seeking, and self-efficacy provided additional evidence that student behavioral signals align with learning-related constructs. While preliminary, these findings suggest that transparent coordination mechanisms may help support human-AI co-agency when feedback is incomplete.
>
---
#### [new 180] ReasonOps: Operator Segmentation for LLM Reasoning Traces
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出ReasonOps，用于分析大模型推理轨迹的结构，解决缺乏统一描述方法的问题。通过无监督聚类发现7种通用推理操作符，提升模型识别和答案预测效果。**

- **链接: [https://arxiv.org/pdf/2605.29192](https://arxiv.org/pdf/2605.29192)**

> **作者:** Daniel Lee; Owen Queen; James Zou
>
> **摘要:** Chain-of-thought traces from large reasoning models can span tens of thousands of tokens, yet we lack a vocabulary for describing their internal structure. Previous methods developed to analyze chain-of-thought traces are either too rigid or not expressive enough, failing to capture features across domains and models. To remedy this, we develop ReasonOps, an unsupervised, expressive method for annotating chain-of-thought traces, providing succinct universal operators. Using ReasonOps, we analyze 44,662 traces from 12 thinking LLMs spanning 6 families across 8 reasoning benchmarks and discover that they share a common compositional structure: 7 recurring reasoning operators -- discourse-level moves such as backtracking, inferring, and hypothesizing -- that emerge from unsupervised clustering of sentence-initial 3-token pivots. These operators appear across every model family and benchmark domain, confirmed by three independent LLM judges who classify held-out samples at 70 -76% accuracy. We analyze the structure of operators on easy vs. hard problems, revealing that reflective operators are more helpful on hard problems and harm performance on easy problems. Operator sequences are highly model-identifying: a classifier trained on operator distributions alone recovers the source model with macro-AUC, revealing that each model family has a distinctive reasoning fingerprint. Structural operator features predict within-problem answer correctness well above baselines. Classifiers built on these operators reach WP-AUC and on AIME specifically. ReasonOps further enables early quality estimation well before the trace completes: we predict at WP-AUC for only 50% of the trace. The ReasonOps pipeline is unsupervised and annotation-free, enabling deep insights into LLM reasoning traces as well as strong downstream results on model identification and correctness prediction.
>
---
#### [new 181] Offloading Score: Measuring AI Reliance Through Counterfactual Workflows
- **分类: cs.SE; cs.CL; cs.CY; cs.HC**

- **简介: 该论文属于人机协作研究，旨在解决如何量化用户对AI工具的依赖程度。提出“离手分数”指标，通过模拟对比任务流程评估依赖水平。**

- **链接: [https://arxiv.org/pdf/2605.29392](https://arxiv.org/pdf/2605.29392)**

> **作者:** Vishakh Padmakumar; Lujain Ibrahim; Zora Zhiruo Wang; Jennifer Wang; Q. Vera Liao; Diyi Yang
>
> **备注:** Preprint
>
> **摘要:** AI tools are increasingly integrated into real-world workflows. However, existing measures of reliance on these tools focus on AI output adoption or on self-reported indicators, rather than how task effort is distributed between users and tools. Here, we introduce offloading score, a measure of reliance that quantifies the fraction of cognitive effort offloaded to an AI tool. Offloading Score is simulation-based -- we construct a counterfactual workflow by estimating how the user would have completed the task without the tool, and then computing the fraction of steps saved by using the tool. We validate offloading score through intrinsic evaluations of metric validity, and a controlled user study ($n=40$) with developers performing programming tasks using AI tools. We vary time pressure to test whether reliance measures capture the known increase in reliance under time pressure. We show that offloading score detects significantly higher reliance in time-constrained settings ($+43\%$, $p=0.018$), while usage-based and self-reported baseline measures of reliance do not distinguish the conditions. We complement this with descriptive insights showing that higher reliance manifests as greater delegation of subtasks to the tool and more direct reuse of AI outputs. Finally, we demonstrate an approach of using offloading score in combination with target outcomes of a task (e.g., code understanding) to identify when reliance may be (in)appropriate. Our framework offers two contributions: an instrument users can apply to measure and reflect on their own reliance, and a quantitative signal that agent designers can utilize to mitigate overreliance.
>
---
#### [new 182] AgentDoG 1.5: A Lightweight and Scalable Alignment Framework for AI Agent Safety and Security
- **分类: cs.AI; cs.CL; cs.CR; cs.CV; cs.LG**

- **简介: 该论文提出AgentDoG 1.5，解决AI代理安全与可靠性问题，通过轻量级框架提升安全性，适用于实际部署。**

- **链接: [https://arxiv.org/pdf/2605.29801](https://arxiv.org/pdf/2605.29801)**

> **作者:** Dongrui Liu; Yu Li; Zhonghao Yang; Peng Wang; Guanxu Chen; Yuejin Xie; Qinghua Mao; Wanying Qu; Yanxu Zhu; Tianyi Zhou; Leitao Yuan; Zhijie Zheng; Qihao Lin; Yimin Wang; Haoyu Luo; Shuai Shao; Chen Qian; Qingyu Liu; Ling Tang; Ruiyang Qin; Qihan Ren; Junxiao Yang; Kun Wang; Zhiheng Xi; Linfeng Zhang; Ranjie Duan; Bo Zhang; Wenjie Wang; Wen Shen; Qiaosheng Zhang; Yan Teng; Chaochao Lu; Rui Mei; Man Li; Jialing Tao; Xi Lin; Tianhang Zheng; Yong Liu; Quanshi Zhang; Lei Zhu; Xingjun Ma; Junhua Liu; Hui Xue; Xiaoxiang Zuo; Xiangnan He; Chao Shen; Xianglong Liu; Minlie Huang; Jing Shao; Xia Hu
>
> **备注:** 44 pages, 12 Figures, 9 Tables
>
> **摘要:** Modern open-world agents such as OpenClaw exhibit powerful cross-environment execution capabilities yet introduce broad new safety risk sources. Meanwhile, advanced frontier AI models drastically lower attack barriers, rendering current agent alignment frameworks inadequate for real-world deployment. To tackle these emerging threats, we propose a lightweight and scalable agent safety alignment framework. Specifically, we update the agent safety taxonomy to accommodate emergent risks from Codex and OpenClaw execution scenarios. We further build a taxonomy-guided data engine with influence-function purification to train lightweight AgentDoG 1.5 variants (0.8B, 2B, 4B, and 8B parameters) using only around 1k samples, achieving comparable performance with leading closed-source models (e.g., GPT-5.4). Based on AgentDoG 1.5, we construct a highly efficient agentic safety SFT and RL training environment, which reduces deployment overhead in Docker-level environments by two orders of magnitude. Finally, we deploy AgentDoG 1.5 as a training-free online guardrail for real-time safety moderation. Extensive experimental results indicate that AgentDoG 1.5 achieves state-of-the-art performance in diverse and complex interactive agentic scenarios. All models and datasets are openly released.
>
---
#### [new 183] RUBRIC-ARROW: Alternating Pointwise Rubric Reward Modeling for LLM Post-training in Non-verifiable Domains
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习中的奖励建模任务，解决非验证领域中主观评分困难的问题。提出RUBRIC-ARROW框架，通过交替训练生成评分标准和判断模型，提升后训练效果。**

- **链接: [https://arxiv.org/pdf/2605.29156](https://arxiv.org/pdf/2605.29156)**

> **作者:** Haoxiang Jiang; Zihan Dong; Tianci Liu; Wanying Wang; Ran Xu; Tony Yu; Linjun Zhang; Haoyu Wang
>
> **摘要:** Pointwise reward modeling offers critical signals for LLM post-training, yet struggles with absolute scoring in subjective, non-verifiable settings. Rubric-based methods address this by decomposing evaluation into explicit criteria, but existing approaches typically depend on frontier LLMs and suffer from ties caused by hard Boolean aggregation. We present RUBRIC-ARROW, an alternating framework that jointly trains a rubric generator and a rubric-conditioned judge, with its RL stage using only pairwise preference data. Our method couples a probability-based scoring rule that reduces ties with phase-specific preference-based rewards and an alternating GRPO scheme that together train the pointwise evaluator. Extensive experiments show that RUBRIC-ARROW achieves competitive reward-modeling accuracy and yields consistent gains for downstream policy post-training.
>
---
#### [new 184] When Should Models Change Their Minds? Contextual Belief Management in Large Language Models
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究语言模型在长期交互中的信念管理问题，属于自然语言处理任务。旨在解决模型如何有效更新、保留或忽略信息的问题，提出BeliefTrack基准和强化学习方法提升信念管理能力。**

- **链接: [https://arxiv.org/pdf/2605.30219](https://arxiv.org/pdf/2605.30219)**

> **作者:** Haoming Xu; Weihong Xu; Zongrui Li; Mengru Wang; Yunzhi Yao; Chiyu Wu; Jin Shang; Yu Gong; Shumin Deng
>
> **备注:** Work in progress
>
> **摘要:** Long-horizon interactions require language models to manage accumulating information: when to update their state, when to preserve their state, and what to ignore. We study this challenge as \textbf{Contextual Belief Management (CBM)}: maintaining a predicted belief state aligned with formal evidence while isolating task-irrelevant noise. To make CBM measurable, we introduce BeliefTrack, a closed-world benchmark spanning Rule Discovery and Circuit Diagnosis, where a finite belief space and symbolic verifiers enable exact turn-level evaluation. BeliefTrack diagnoses three failures: Failed Stay, Failed Update, and Failed Isolation. Across multiple LLMs, vanilla models exhibit severe CBM failures, while explicit belief-tracking prompts provide limited gains. In contrast, reinforcement learning with belief-state rewards reduces failure rates by 70.9\% on average. Further probing reveals latent belief-state dynamics behind these failures, and representation-level steering reduces failure rates by 46.1\% across two tasks\footnote{Code is coming soon at this https URL.
>
---
#### [new 185] MELD: Mel-Spectrogram-Based Speech Language Modeling with Discrete Latent Variables
- **分类: eess.AS; cs.CL**

- **简介: 该论文提出MELD模型，解决语音语言建模中编码器与自回归模型分离导致的表示不优问题，通过联合优化提升TTS和STT性能。**

- **链接: [https://arxiv.org/pdf/2605.29859](https://arxiv.org/pdf/2605.29859)**

> **作者:** Sung-Lin Yeh; Wei Zhou; Gil Keren; Duc Le; Zhong Meng; Hao Tang; Jay Mahadeokar; Ozlem Kalinli; Alexandre Mourachko
>
> **摘要:** Recent speech language models rely on encoders that are optimized separately from autoregressive models. Since these encoders are unaware of the downstream objectives, the extracted representations may not be optimal for downstream tasks. To address this limitation, we introduce a discrete latent variable model on mel spectrograms that jointly optimizes the encoder and the speech language model. Joint optimization not only brings improvements over codec-based and other mel-spectrogram-based baselines on zero-shot Text-to-Speech (TTS) and Speech-to-Text (STT) tasks, but also effectively alleviates common issues in autoregressive mel-spectrogram modeling, such as prolonged silence generation and word omissions.
>
---
#### [new 186] Reasoning with Sampling: Cutting at Decision Points
- **分类: cs.LG; cs.AI; cs.CL; math.ST; stat.ML**

- **简介: 该论文属于推理任务，解决如何高效采样于功率分布以提升推理能力的问题。提出基于熵的切割方法，更有效地重采决策点，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.30327](https://arxiv.org/pdf/2605.30327)**

> **作者:** Felix Zhou; Anay Mehrotra; Quanquan C. Liu
>
> **摘要:** Frontier reasoning models are produced by posttraining base language models with reinforcement learning. Recent work has challenged this by showing that sampling from a sharpened version of the base model's distribution, a so-called power distribution, elicits comparable reasoning without additional training, curated datasets, or verifiers. However, making this method practical requires efficiently sampling from the power distribution. A sampler needs to "mix" to the power distribution, which necessitates moving between modes of the target distribution; intuitively, e.g., trying different reasoning strategies. The samplers proposed in prior works repeatedly select a "cut" position in the current reasoning trace uniformly at random and resample the suffix from that position onward. However, reasoning traces typically contain a few consequential decisions (e.g., the choice of proof strategy or algorithm), and we observe that a uniformly chosen cut tends to rewrite local details rather than revisit decision points. We introduce an algorithm (Entropy-Cut Metropolis-Hastings) that uses the base model's next-token entropy as a proxy to identify key decision points and resample from those positions. We empirically verify that entropy jumps are a useful proxy for decision points and, in a stylized model of reasoning, prove that our method's mixing time scales with the number of decisions in a trace rather than with the number of tokens, which can be much larger. Across MATH500, HumanEval, GPQA Diamond, and AIME26, our method consistently improves over baselines and RL-trained models.
>
---
#### [new 187] PARCEL: Pool-Anchored Resampling with Conditioned Elastic Queries for Efficient Vision-Language Understanding
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于视觉-语言理解任务，旨在解决大模型推理时的计算瓶颈问题。提出PARCEL架构，通过动态分配特征提取任务，提升压缩效率与性能。**

- **链接: [https://arxiv.org/pdf/2605.30126](https://arxiv.org/pdf/2605.30126)**

> **作者:** Selim Kuzucu; Alessio Tonioni; Vasile Lup; Bernt Schiele; Federico Tombari; Muhammad Ferjad Naeem
>
> **备注:** 33 pages, 4 figures
>
> **摘要:** Large Vision-Language Models (LVLMs) map visual inputs into dense token sequences, imposing a quadratic computational bottleneck for inference. Elastic visual-token compression addresses this by training a single model that can run at multiple visual-token budgets. However, existing approaches struggle under aggressive compression. Spatial-only compression, as in nested pooling, behaves as an imperfect low-pass filter and induces spectral aliasing that obscures fine-grained detail. Query-only compression, as in nested query resampling, replaces explicit grid-aligned tokens with non-local summaries and substantially degrades spatial grounding. To resolve this representational conflict, we introduce PARCEL (Pool-Anchored Resampling with Conditioned Elastic Queries for Efficient Vision-Language Understanding), a visual tokenization architecture that dynamically partitions the labor of feature extraction. PARCEL establishes spatial pool tokens as low-frequency layout anchors and conditions elastic query tokens on these anchors through Pool-Conditioned Query Resampling. This encourages query tokens to focus on complementary visual features rather than redundant spatial mapping. Extensive evaluations across 27 benchmarks show that PARCEL improves the performance-efficiency Pareto frontier, consistently outperforming existing matryoshka baselines across visual-token budgets while preserving the "train once, deploy anywhere" paradigm.
>
---
#### [new 188] Minimal Prompt Perturbations Lead to Code Vulnerabilities: Prompt Fragility and Hidden-State Signals in Coding LLMs
- **分类: cs.CR; cs.CL; cs.SE**

- **简介: 该论文研究LLM生成代码的安全性问题，探讨微小提示变化如何导致漏洞。属于代码安全任务，解决提示脆弱性影响代码安全的问题，通过实验验证提示扰动的影响。**

- **链接: [https://arxiv.org/pdf/2605.29737](https://arxiv.org/pdf/2605.29737)**

> **作者:** Alexander Sternfeld; Andrei Kucharavy; Ljiljana Dolamic
>
> **摘要:** LLM-based coding assistants are seeing rapid adoption, offering substantial gains in developer productivity. As organizations increasingly ship code these agents produce, the security of that code becomes critical. Prior work has shown that minor prompt perturbations degrade the functional correctness of LLM-generated code, but whether they also compromise code security has remained unstudied. We apply token-level mutations to prompts across three models and five programming languages, and show that mutations as small as a single-character change can flip generated code from secure to vulnerable. Probing the models' hidden states reveals that this fragility is partially encoded in prompt representations, but unevenly so. Input-handling vulnerabilities, where the model omits validation or sanitization, are more predictable (mean AUC 0.753) than secure-defaults vulnerabilities, where insecure code stems from one local choice such as a weak algorithm or unsafe parameter (mean AUC 0.674). These results show that the threat model for LLM-assisted coding extends beyond prompt injection to ordinary prompt variation, and indicate that input-handling flaws can be caught before generation while secure-defaults flaws require intervention during decoding.
>
---
#### [new 189] Towards Human-Like Interactive Speech Recognition With Agentic Correction and Semantic Evaluation
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于语音识别任务，旨在解决传统ASR系统无法有效修正语义错误的问题。提出Agentic ASR框架和S²ER指标，实现多轮交互式纠错。**

- **链接: [https://arxiv.org/pdf/2605.29430](https://arxiv.org/pdf/2605.29430)**

> **作者:** Zixuan Jiang; Yanqiao Zhu; Peng Wang; Qinyuan Chen; Xinjian Zhao; Xipeng Qiu; Wupeng Wang; Zhifu Gao; Xiangang Li; Kai Yu; Xie Chen
>
> **摘要:** Automatic speech recognition (ASR) is a core component of human--computer interaction and an increasingly important front-end for LLM-based assistants and agents. However, most current ASR systems still follow a single-pass paradigm, which is poorly aligned with human communication, where misunderstandings are resolved through iterative clarification and refinement. This mismatch makes it difficult to correct meaning-critical errors once they occur. Meanwhile, token-level metrics such as WER or CER cannot adequately reflect such a problem. To address these limitations, we formulate \emph{Interactive ASR} as a multi-turn refinement task and propose \textbf{Agentic ASR}, a closed-loop framework that combines a single-pass ASR front-end with semantic correction, intent routing, and reasoning-based editing. We further introduce the \textbf{Sentence-level Semantic Error Rate} ($S^2ER$), an LLM-based semantic evaluation metric, together with an \textbf{Interactive Simulation System} for scalable and reproducible benchmarking. Experiments on multilingual, named-entity-intensive, and code-switching benchmarks show that iterative interaction consistently reduces semantic errors, with much larger gains in $S^2ER$ than in conventional token-level metrics. Human--AI alignment and ablation studies further validate the reliability of the semantic judge and the robustness of the proposed framework. The code is available at: this https URL and the live demo is available at this https URL
>
---
#### [new 190] MuPHI: Learning Implicit Multimodal Harm Reasoning via Semantically Grounded Reward Optimization
- **分类: cs.AI; cs.CL; cs.LG; cs.MM**

- **简介: 该论文属于多模态有害推理任务，旨在解决VLMs在隐式危害语义理解上的不足。提出MuPHI数据集和MuPHIRM框架，提升模型的检测与推理能力。**

- **链接: [https://arxiv.org/pdf/2605.29951](https://arxiv.org/pdf/2605.29951)**

> **作者:** Anisha Saha; Varsha Suresh; Teodora Kamova; Sophia Wiedmann; Timothy Hospedales; Vera Demberg
>
> **摘要:** Understanding how harm emerges from interaction between otherwise benign image-text pairs requires intent-aware cross-modal reasoning beyond surface-level features. Existing vision-language models (VLMs) excel at literal reasoning over perceptual cues but often fail to derive harmful semantics that rely on implicit, context-dependent reasoning. To evaluate VLMs on compositional harm detection and reasoning, we introduce Multimodal Pragmatic Harm Interpretation (MuPHI), a dataset containing image-text pairs where harm is encoded in subtle multimodal cues. MuPHI spans diverse harm categories and includes annotated harm rationales for assessing VLM reasoning chains. To improve both detection and reasoning in VLMs, we propose MuPHIRM, a reasoning-augmented training framework which learns joint semantics by optimizing multi-perspective rewards. MuPHIRM improves both harm detection and reasoning quality of VLMs while demonstrating superior out-of-distribution robustness compared to both trained and inference-time baselines. Our findings suggest that reasoning-oriented reward optimization offers a promising direction towards building multimodal systems that generalize beyond benchmark-specific shortcuts.
>
---
#### [new 191] Opir: Efficient Multi-Task Safety Classification for Toxicity, Jailbreaks, Hate Speech, and Harmful Content
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出Opir，用于实时安全过滤的多任务分类模型，解决有毒内容、越狱攻击等检测问题，通过多任务学习提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2605.29659](https://arxiv.org/pdf/2605.29659)**

> **作者:** Ihor Stepanov; Aleksandr Smechov
>
> **备注:** 23 pages, 4 figures, 9 tables
>
> **摘要:** Real-time safety filtering for large language model (LLM) applications requires classifiers that can detect unsafe prompts, toxic language, jailbreak attempts, and unsafe responses without the cost profile of large guardrail models, and that can distinguish benign sensitive text from genuinely covert harmful content. In this paper, we introduce Opir, a family of encoder-based guardrail models built on the GLiClass architecture. Opir includes multi-task models for binary safe/unsafe classification, multi-label toxicity classification, jailbreak classification, and zero-shot unsafe prompt and response categorization. We also release edge variants with fewer than 100M parameters dedicated to binary safe/unsafe categorization. The models are trained on a three-level taxonomy containing 996 categories across 16 top-level labels, 126 mid-level labels, and 854 leaf labels. Opir's training data combines taxonomy-grounded unsafe prompts, adversarially mined hard negatives, benign safety-preserving examples, generated response examples, multilingual translations, and portions of the Aegis2 and WildGuard training subsets. We also open-sourced an evaluation harness that supports GLiClass and GLiNER2 backends as well as decoder-based models, and covers binary safety classification, multi-label categorization, toxicity, jailbreak detection, prompt safety, response safety, response refusal, and prompt subcategory views across public benchmark families. Across an expanded comparison spanning 12 safety-classification tasks and 17 category tasks against eight contemporary guardrail systems -- including both GLiNER2-based and generative guardrail models -- Opir variants are competitive on or ahead of the strongest open-weight baselines on the majority of benchmark datasets while operating with a substantially smaller deployment footprint.
>
---
#### [new 192] The Confidence Shortcut: A Reasoning Failure Mode of Masked Diffusion Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究Masked Diffusion Models在复杂推理任务中的推理失败模式。指出基于置信度的解码与逻辑流程不匹配，导致错误率上升。通过多数字加法等任务验证，提出随机掩码更有效。**

- **链接: [https://arxiv.org/pdf/2605.29123](https://arxiv.org/pdf/2605.29123)**

> **作者:** Dueun Kim; Albert No
>
> **摘要:** Masked diffusion language models (MDMs) uniquely support any-order generation, with confidence-based decoding currently serving as the de facto standard inference policy. To optimize for this, recent training schemes attempt to align training mask patterns directly with those observed during generation. However, we argue that confidence-based decoding is inherently misaligned with the logical-flow trajectories required for complex reasoning, and that confidence-aligned training actively entrenches this misalignment. We make this concrete using multi-digit addition, where the decoding strategy prematurely predicts locally easy digits before resolving their long-range dependencies, producing high-confidence errors on challenging inputs. While traditional random masking keeps the failure rate low on this challenging tail, confidence-aligned training amplifies the error rate by an order of magnitude. Across five distinct reasoning tasks, this same pattern emerges with task-dependent severity: confidence-based decoding induces failures on highly complex inputs, and confidence-aligned training exacerbates them. In contrast, random masking -- despite its perceived inefficiency -- robustly preserves the reasoning-trajectory conditionals essential for solving the challenging tail.
>
---
#### [new 193] REPOT: Recoverable Program-of-Thought via Checkpoint Repair
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文提出RePoT，解决PoT在遇到错误时无法恢复的问题，通过检查点修复实现高效重试，提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2605.30052](https://arxiv.org/pdf/2605.30052)**

> **作者:** Parsa Mazaheri
>
> **摘要:** One-shot Program-of-Thought (PoT) emits a Python program that prints a primitive-action plan; a single invalid action silently invalidates the trajectory. We introduce RePoT (Recoverable PoT): a deterministic verified replay that walks the plan through the environment to its first invalid transition, then one LLM call that resumes from the verified prefix. RePoT costs at most one extra LLM call on the ~14% of problems where PoT fails. RePoT beats PoT by +3 to +11pp across four closed-model configurations on PuzzleZoo-775 and peaks at 96.9% vs 86.3% on gpt-5.4-mini-medium; against the matched-budget PoT-retry baseline, RePoT wins decisively on Gemini (+3.8pp, 95% CI [+2.2,+5.4]), is within sampling noise on GPT-medium and Claude, and loses on GPT-mini -- a capability-scaling pattern we begin to address with Adaptive RePoT, a rule-based dispatcher that routes between suffix repair and a fresh PoT retry based on verified-prefix length (preliminary). We replicate on PlanBench Blocksworld (+1.1 to +11.4pp) and on four open-weights models (+3.3 to +20.0pp on three of four). On Derail-550, our controlled recovery benchmark, every condition with access to checkpoint information clears >=30% on GPT-medium and >=70% on Gemini, vs <=3.1% for error-only feedback -- showing that checkpoint information, not the specific verified-prefix tail, is the load-bearing recovery signal.
>
---
#### [new 194] WorldMemArena: Evaluating Multimodal Agent Memory Through Action-World Interaction
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态记忆研究任务，旨在解决长周期代理中记忆管理问题。通过构建WorldMemArena基准，评估记忆的撰写、维护、检索与使用效果。**

- **链接: [https://arxiv.org/pdf/2605.29341](https://arxiv.org/pdf/2605.29341)**

> **作者:** Chengzhi Liu; Yuzhe Yang; Sophia Xiao Pu; Yepeng Liu; Lin Long; Yichen Guo; Nuo Chen; Zhaotian Weng; Elena Kochkina; Simerjot Kaur; Charese Smiley; Xiaomo Liu; James Zou; Sheng Liu; Yuheng Bu; Songyou Peng; Xin Eric Wang
>
> **备注:** 25 pages, 8 figures
>
> **摘要:** Multimodal large language models are increasingly deployed as long-horizon agents, where memory must do more than recall: it must track an evolving world, revise what has gone stale, and surface the right evidence at decision time. Existing benchmarks measure recall over static dialogue, collapse memory into a single end-of-task accuracy, and reduce visual observations to captions, leaving us unable to localize failures to writing, maintenance, retrieval, or use. The rise of agent harnesses that author their own memory sharpens this gap, since we have no principled way to compare hand-designed pipelines with self-managing alternatives. To close these gaps, we formulate multimodal agent memory as an Action-World Interaction Loop with an observable four-stage lifecycle, and instantiate it in WorldMemArena: 400 multi-session multimodal tasks spanning Lifelong Evolution (evolving personal and task states) and Agentic Execution (memory from real observations, actions, and feedback), annotated with gold memory points, updates, distractors, and evidence chains for stage-level diagnosis. This enables the first head-to-head comparison of long-context, manually designed (RAG and external memory systems), and harness-based memory agents. Results show that: (1) better memory writing and storage do not guarantee better performance; (2) multimodal memory still struggles to fully use visual evidence; (3) systems are unstable across domains and degrade on realistic agentic trajectories; and (4) harness memory is more flexible but remains costly and less reliable.
>
---
#### [new 195] Measuring Real-World Prompt Injection Attacks in LLM-based Resume Screening
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于安全任务，研究LLM在简历筛选中的提示注入攻击问题。通过分析20万份真实简历，发现约1%存在隐藏攻击，揭示了实际应用中的安全风险。**

- **链接: [https://arxiv.org/pdf/2605.28999](https://arxiv.org/pdf/2605.28999)**

> **作者:** Mohan Zhang; Yuqi Jia; Zhen Tan; Steven Jiang; Neil Zhenqiang Gong; Tianlong Chen; Dawn Song
>
> **备注:** Published in USENIX Security Symposium 2026; Code and artifacts are available at this https URL
>
> **摘要:** LLMs are vulnerable to prompt injection attacks. However, this vulnerability has been primarily demonstrated conceptually in academic studies or through a few anecdotal case studies. Its prevalence and impact in real-world LLM-based applications are largely unexplored. In this work, we present the first systematic study of prompt-injection attacks in a widely used application: LLM-based resume screening. Our analysis is based on approximately 200K real-world resumes collected over multiple years by hireEZ. We first design tailored methods to detect prompt injection in resumes. Manual validation on a small-scale dataset demonstrates that our detectors achieve high precision and outperform state-of-the-art general-purpose detectors. We then apply our detector to the full resume dataset and conduct a comprehensive measurement study of real-world prompt injection attacks. Our analysis reveals several intriguing findings: approximately 1% of resumes contain hidden prompt injections; the prevalence of such injected resumes has increased noticeably over the past one to two years; and more than 90% of injected prompts do not use explicit instructions. These results provide the first evidence of large-scale prompt injection in real-world LLM-based applications and lay the groundwork for future studies to understand and mitigate such attacks.
>
---
#### [new 196] Why Specialist Models Still Matter: A Heterogeneous Multi-Agent Paradigm for Medical Artificial Intelligence
- **分类: cs.AI; cs.CL; cs.LG; cs.MA**

- **简介: 该论文属于医疗AI任务，解决通用模型是否取代专业模型的问题。提出HetMedAgent框架，实现通用模型与专业模型协作，提升临床决策效果。**

- **链接: [https://arxiv.org/pdf/2605.29744](https://arxiv.org/pdf/2605.29744)**

> **作者:** Yanan Wang; Shuaicong Hu; Jian Liu; Guohui Zhou; Aiguo Wang; Cuiwei Yang
>
> **备注:** Accepted at ICML 2026. 12 pages main text, 16 pages appendix
>
> **摘要:** The impressive performance of generalist large language models (LLMs) such as GPT and Claude in healthcare raises a critical question: will domain-specific medical specialist models become obsolete? We argue that the future of medical artificial intelligence (AI) lies not in building monolithic medical foundation models, nor in replacing human expertise, but in orchestrating collaboration among generalist LLMs, domain-specific specialist models, and clinicians. We propose HetMedAgent, a heterogeneous medical multi-agent framework that enables conflict-aware evidence fusion, uncertainty-based clinician intervention triggering, and adaptive threshold calibration. Experiments on three real-world clinical decision-making tasks demonstrate that the synergy between generalist LLMs and domain-specific specialist models significantly outperforms using either type of model alone, validating the irreplaceable value of specialist models in modality-specific analysis. HetMedAgent represents a shift from building medical LLMs or foundation models to multi-agent collaboration, achieving a balance between general reasoning capabilities and domain-specific precision.
>
---
#### [new 197] VideoFDB: Evaluating Full-Duplex Vision-Speech Capabilities in Conversational Agents
- **分类: cs.CV; cs.CL; cs.HC**

- **简介: 该论文提出VideoFDB基准，用于评估全双工视听对话代理。解决现有基准仅关注语音的问题，通过真实视频通话数据、行为分类和评分框架，系统评估多模态对话质量。**

- **链接: [https://arxiv.org/pdf/2605.30256](https://arxiv.org/pdf/2605.30256)**

> **作者:** Amrita Mazumdar; Seonwook Park; Rajarshi Roy; Nikhil Srihari; Shengze Wang; Yuhao Zhou; Julia Wang; Koki Nagano; Shalini De Mello
>
> **备注:** Project page: this https URL
>
> **摘要:** Natural human conversation is full-duplex and audio-visual: people simultaneously speak and listen while continuously interpreting and producing nonverbal cues, such as nods, smiles, and gestures. To support successful human-agent interaction, agents must model full-duplex audiovisual conversation; however, existing full-duplex benchmarks evaluate only speech. In this work, we present VideoFDB, the first benchmark to evaluate full-duplex audio-visual-to-audio-visual (AV2AV) conversational agents. VideoFDB contributes (i) 237 dyadic clips spanning 11 nonverbal conversational dynamics from real-world video calls, (ii) a taxonomy separating perception from generation behaviors, and (iii) a rubric-based LM-as-judge evaluation framework with interpretable axes for assessing conversational quality with respect to nonverbal conversational dynamics. Across open- and closed-source vision-speech agents, we find systematic failure modes: captioning collapse and visual-stream ignorance, and we show that current systems exploit vision for explicit visual question answering but not for the streaming joint audiovisual grounding required in natural conversation. We further evaluate cascaded speech-to-avatar systems and find that their architecture fundamentally precludes the production of full-duplex nonverbal cues. As the first benchmark for full-duplex AV2AV interaction, VideoFDB establishes a foundation for systematic evaluation and, we hope, will accelerate the advancement and development of next-generation multimodal conversational agents.
>
---
#### [new 198] MIC: Maximizing Informational Capacity in Adaptive Representations via Isotropic Subspace Alignment
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出MIC框架，解决多尺度表示学习中的维度冗余和谱崩溃问题。通过子空间对齐和正则化方法，提升嵌入的语义密度和区分能力。**

- **链接: [https://arxiv.org/pdf/2605.29987](https://arxiv.org/pdf/2605.29987)**

> **作者:** Dang Hong Nguyen; Nhi Ngoc-Yen Nguyen; Huy-Hieu Pham
>
> **备注:** Accepted at the GlobalSouthML Workshop at ICML 2026. 13 pages, 2 figures
>
> **摘要:** Although multi-scales representation learning enables elastic-dimension embeddings, nested subspaces often suffer from dimensional redundancy and spectral collapse. To address this, we introduce MIC, a framework that optimizes the geometric landscape of multi-granular embeddings through isotropic subspace alignment. MIC employs Soft Collapse Regularization (SCR) to mitigate redundancy between prefix and residual subspaces via cross-correlation penalties, alongside Spectral Isotropy Regularization (SIR) to ensure hyper-spherical uniformity in low-dimensional prefixes. By unifying these strategies through a self-distillation objective, MIC generates semantically dense representations that maintain high discriminative power. Our experiments demonstrate that MIC significantly outperforms standard baselines, particularly in high-compression scenarios where maintaining informational capacity is most critical.
>
---
#### [new 199] Implicit Identity Technologies for LLMs: Fingerprinting and Watermarking across Datasets, Models, and Generated Content
- **分类: cs.CR; cs.CL; cs.LG**

- **简介: 该论文属于LLM身份识别任务，旨在解决模型和生成内容的归属验证问题。通过梳理指纹与水印技术，提出统一分类和评估框架。**

- **链接: [https://arxiv.org/pdf/2605.29245](https://arxiv.org/pdf/2605.29245)**

> **作者:** Bing Liu; Shunping Wang; Yufan Zhu; Xinyi Yu; Jing Huang; Linkang Du; Hongbin Pei; Wei Luo
>
> **备注:** Accepted by IJCAI-ECAI 2026. 11 pages, 1 figure. Survey and taxonomy of LLM fingerprinting and watermarking for identity, provenance, generated-content attribution, and asset protection
>
> **摘要:** This paper presents a survey and taxonomy of LLM fingerprinting and watermarking for identity, ownership verification, provenance, and generated-content attribution. Large language models (LLMs) require substantial investments in data, computation, and expertise, and are increasingly deployed in high-stakes settings, making it critical to protect LLM-related assets and trace their origins. Existing work has rapidly expanded across dataset provenance, model ownership, and generated-content detection, but the field remains fragmented: fingerprinting and watermarking are often used inconsistently, and methods are typically studied within isolated asset-specific settings. To address this gap, we introduce implicit identity as a unifying abstraction for verifiable but not directly observable identity signals in LLM systems. We distinguish fingerprinting as non-intrusive identity derived from intrinsic characteristics, and watermarking as intrusive identity deliberately embedded into data, models, or generated content. We then propose a lifecycle-based taxonomy that organises techniques across datasets, models, and generated content, and further separates them by verification semantics: similarity-based attribution and keyed verification. Finally, we establish an evaluation framework centred on identifiability, robustness, and deployability, summarising representative metrics under realistic access and transformation regimes. By unifying terminology, lifecycle stages, and evaluation objectives, this survey provides a structured foundation for studying LLM identity technologies and for developing more reliable mechanisms for asset protection and provenance.
>
---
#### [new 200] Token Inflation: How Dishonest Providers Can Overcharge for Large Language Model Usage
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 论文探讨了大语言模型按token计费中的欺诈问题，属于安全与审计任务。研究指出当前审计机制易被操纵，导致费用被恶意虚报，提出需依赖第三方验证以确保计费诚实。**

- **链接: [https://arxiv.org/pdf/2605.30040](https://arxiv.org/pdf/2605.30040)**

> **作者:** Shahinul Hoque; Jinghuai Zhang; Jinyuan Sun; Fnu Suya
>
> **摘要:** Per-token billing is now the standard pricing model for commercial large language models (LLMs), so the honesty of reported token counts directly affects what users pay. We show that this kind of billing is hard to audit by design: providers hide the model, the tokenizer, and the execution to protect their IP, mitigate jailbreaks, and preserve user privacy, which means an auditor can only inspect proofs the provider supplies. The audit therefore reduces to a consistency check on the provider's own reports. We call this a trust paradox: every audit must trust some artifact, but current frameworks trust exactly the ones a provider has the strongest reason to manipulate. We study three recent token auditing frameworks and show that a provider with ordinary commercial capabilities can systematically inflate billed token counts. In the most permissive setting, hidden reasoning usage can be inflated by 1,469% on average without detection. At current frontier reasoning prices, that turns a \$100 honest bill into roughly a \$1,569 bill on the same query. Even when the user can see the full reasoning string, tokenization ambiguity alone still allows 50.85% over-reporting below the detection threshold. These results suggest the problem is not in any specific auditor but in any audit whose evidence comes from the audited party. Restoring honest billing will require verification that ties reported token counts to evidence the provider does not control, such as trusted execution attestation, cryptographic proofs of inference, or third-party re-execution.
>
---
## 更新

#### [replaced 001] Hilbert-Geo: Solving Solid Geometric Problems by Neural-Symbolic Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于几何推理任务，解决三维几何问题难以处理的问题。提出Hilbert-Geo框架及Parse2Reason方法，实现准确的几何推理与验证。**

- **链接: [https://arxiv.org/pdf/2605.16385](https://arxiv.org/pdf/2605.16385)**

> **作者:** Ruoran Xu; Haoyu Cheng; Bin Dong; Qiufeng Wang
>
> **备注:** Computer Vision and Pattern Recognition (CVPR), 2026
>
> **摘要:** Geometric problem solving, as a typical multimodal reasoning problem, has attracted much attention and made great progress recently, however most of works focus on plane geometry while usually fail in solid geometry due to 3D spatial diagrams and complex reasoning. To bridge this gap, we introduce Hilbert-Geo, the first unified formal language framework for solid geometry, including an extensive predicate library and a dedicated theorem bank. Based on this framework, we propose a Parse2Reason method containing two steps of first parsing then reasoning. In the parsing step, we utilize conditional description language (CDL), a formalized language composed of predicates specifically designed to construct geometric conditions, to represent both problem description (natural text) and solid diagrams (visual image). In the reasoning step, we leverage those formal CDL and the theorem bank to perform relational inference and algebraic computation, generating strictly correct, verifiable, and human-readable reasoning processes. Notably, our proposed Hilbert-Geo is also applicable to plane geometry. To advance geometric reasoning, we curate two expert-annotated dataset SolidFGeo2k and PlaneFGeo3k, which are furnished with geometric formal language annotations, solutions and answers. Extensive experiments show that our proposed method achieves the state-of-the-art (SOTA) performance 77.3% in SolidFGeo2k and 84.1% in MathVerse-Solid (one small subset in MathVerse dedicated to solid geometry), substantially outperforming leading MLLMs, such as Gemini-2.5-pro (54.2% on SolidFGeo2k) and GPT-5 (62.9% on MathVerse-Solid). In addition, our method achieves the SOTA accuracy 80.2% in PlaneFGeo3k, demonstrating the generality of the Hilbert-Geo in geometric reasoning. Our code and datasets will be publicly available.
>
---
#### [replaced 002] SEEK: Semantic Evidence Extraction via Adaptive ChunKing for Multilingual Fact-Checking
- **分类: cs.CL**

- **简介: 该论文属于多语言事实核查任务，旨在解决证据不完整和上下文缺失的问题。提出SEEK框架，通过语义分块构建连贯证据，提升核查准确性。**

- **链接: [https://arxiv.org/pdf/2605.26755](https://arxiv.org/pdf/2605.26755)**

> **作者:** Babu Kumar; Gaurav Kumar; Ayush Garg; Aditya Kishore; Jasabanta Patro
>
> **摘要:** Multilingual fact verification requires evidence that is both relevant and sufficiently complete for reliable factuality prediction. However, existing systems often rely on search snippets, sentence-level evidence, or locally segmented passages, which can miss decisive context and produce fragmented evidence. To overcome these limitations, we propose SEEK, a Semantic Evidence Extraction with an adaptive chunKing framework that constructs coherent evidence chunks from full fact-checking articles by identifying semantic topic transitions and preserving local verification context. The constructed chunks are encoded using a multilingual encoder and then multilingual LLMs are finetuned using LoRA adapter for veracity prediction. Experiments on X-FACT and RU22Fact show that SEEK improves macro-f1 by up to 10% over semantic chunking, 19% over sentence chunking, and 20% over search-snippet baselines. Evidence completeness and significance analyses further show that SEEK preserves richer verification context and enables more reliable multilingual fact-checking.
>
---
#### [replaced 003] Valency Classification of Mapudungun Verbal Roots. Established by the language's own morphotactics
- **分类: cs.CL**

- **简介: 该论文属于语法分析任务，旨在通过马普切语自身形态规则对动词根进行配价分类，解决动词形态与语义关系问题。**

- **链接: [https://arxiv.org/pdf/2604.00789](https://arxiv.org/pdf/2604.00789)**

> **作者:** Andrés Chandía
>
> **备注:** 37 pages
>
> **摘要:** In the previous work, a lexical (re)categorisation -- or confirmation of the given category -- of roots identified as verbal was undertaken to determine their original category accurately. Building on this, the present paper offers an account of the valency classification of those Mapudungun roots confirmed to be verbal, using the language's own morphotactics; specifically, by examining the permissible and restricted combinations of various suffixes with roots or verbal stems in the Mapuche verb form. As with all work conducted thus far, the results presented here aim to improve the morphological analyser (Dungupeyum) with all verified findings incorporated into the system. From a theoretical perspective, we also hope to contribute to the recognition and understanding of issues related to the valency of Mapuche verb forms.
>
---
#### [replaced 004] ValueFlow: Measuring the Propagation of Value Perturbations in Multi-Agent LLM Systems
- **分类: cs.MA; cs.CL**

- **简介: 该论文属于多智能体系统研究，旨在解决价值扰动传播问题。提出ValueFlow框架，通过实验分析价值漂移，揭示其受交互结构影响，强调价值对齐需从系统层面考虑。**

- **链接: [https://arxiv.org/pdf/2602.08567](https://arxiv.org/pdf/2602.08567)**

> **作者:** Jinnuo Liu; Chuke Liu; Hua Shen
>
> **备注:** Preprint. Under review. 28 pages, 10 figures
>
> **摘要:** Multi-agent large language model (LLM) systems increasingly consist of agents that observe and respond to one another's outputs. While value alignment is typically evaluated for isolated models, how value perturbations propagate through agent interactions remains poorly understood. We present ValueFlow, a perturbation-based framework that measures value drift in multi-agent systems via a 56-value valuation dataset derived from the Schwartz Value Survey, with agent value orientations scored using an LLM-as-a-judge protocol. ValueFlow decomposes value drift into agent-level response behavior and system-level structural effects, captured by two metrics: \b{eta}-susceptibility, an agent's sensitivity to perturbed peer value signals, and system susceptibility (SS), the effect of node-level perturbations on final system this http URL span across value dimensions, backbones, personas, and topologies, showing that susceptibility varies sharply across values and is strongly shaped by interaction structure, indicating that value alignment in multi-agent systems is a system-level property, not just an agent-level one. ValueFlow thus provides a principled basis for auditing and mitigating value propagation in deployed multi-agent systems.
>
---
#### [replaced 005] Revisiting the Effectiveness of LLM Pruning for Test-Time Scaling
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于模型压缩任务，研究LLM剪枝对测试时扩展性能的影响。通过对比结构化与非结构化剪枝，发现非结构化剪枝可提升或保持性能，挑战传统观点。**

- **链接: [https://arxiv.org/pdf/2604.25098](https://arxiv.org/pdf/2604.25098)**

> **作者:** Ocean Monjur; Shahriar Kabir Nahin; Anshuman Chhabra
>
> **摘要:** Large Language Models (LLMs) now exhibit remarkable reasoning capabilities through test-time compute scaling (TTS), with impressive performance across math and coding benchmarks. In parallel, research in model compression has developed pruning methods that seek to remove redundant/detrimental parameters without sacrificing task performance. The intersection of these two research advancements lays the foundation for our work. Specific to reasoning LLMs, prior work has shown that structured pruning (methods which remove entire set of layer blocks), significantly degrades TTS reasoning performance. However, in this work, we revisit this assumption and investigate whether unstructured pruning (methods that carefully remove only certain redundant/detrimental weights) exhibits similar limitations. Surprisingly, our extensive experiments across four reasoning benchmarks on two reasoning LLMs: s1.1-7B and Qwen3-8B, consistently show that unstructured pruning augments TTS performance compared to structured pruning, and at times can even outperform the unpruned full-weight LLMs. Furthermore, we also empirically study the impact of different layer-wise sparsity allocation strategies, which are an important parametric choice for instantiating these unstructured methods. These findings challenge the conventional notion that pruning always reduces TTS performance and in fact, suggest that carefully undertaken pruning can retain TTS effectiveness.
>
---
#### [replaced 006] Post-Training Language Models for Crosslingual Consistency
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言模型任务，解决跨语言一致性问题。通过引入DCO方法，提升模型在不同语言间的响应一致性。**

- **链接: [https://arxiv.org/pdf/2603.04678](https://arxiv.org/pdf/2603.04678)**

> **作者:** Tianyu Liu; Jirui Qi; Mrinmaya Sachan; Ryan Cotterell; Raquel Fernández; Arianna Bisazza
>
> **备注:** ICML 2026. The first two authors contributed equally. Codes available at: this https URL
>
> **摘要:** Language models often respond inconsistently to translation-equivalent prompts across languages, undermining the reliability of multilingual systems. To quantify this, we give an information-theoretic definition of crosslingual consistency as a divergence bound between a model's response distribution and its round-trip pushforward across languages. We then introduce penalized consistency optimization (PCO), a post-training procedure that couples this divergence with a Kullback-Leibler penalty to a fixed reference language model. Because direct optimization of PCO requires expensive on-policy roll-outs, we propose a tractable surrogate, direct consistency optimization (DCO), which can be optimized off-policy. Across diverse language models and 26 languages, DCO significantly improves crosslingual consistency, outperforms existing methods, and enables targeted alignment of low-resource languages.
>
---
#### [replaced 007] EvoSpec: Evolving Speculative Decoding via Real-Time Vocabulary and Parameter Adaptation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型加速任务，解决 speculative decoding 中因词汇量增大导致的瓶颈问题。通过动态调整词汇和参数，提升推理效率与适应性。**

- **链接: [https://arxiv.org/pdf/2605.27390](https://arxiv.org/pdf/2605.27390)**

> **作者:** Shuyu Zhang; Lingfeng Pan; Qicheng Wang; Yaqi Shi; Yueyang Tan; Ruyu Yan; Jiaqi Chen; Lixing Du; Lu Wang
>
> **摘要:** Speculative decoding accelerates Large Language Model inference via a draft-then-verify paradigm, yet the output projection layer becomes a bottleneck as vocabulary sizes scale. While existing static pruning methods effectively reduce this overhead, they suffer from precipitous drops in acceptance rate in specialized domains or topic-switching scenarios due to their inability to capture dynamic distribution shifts. To address this, we introduce EvoSpec, a framework that enables real-time evolution of the draft model through dynamic vocabulary and parameter adaptation. Unlike static or purely retrieval-based approaches, EvoSpec employs a context-aware mechanism that retrieves critical long-tail tokens via efficient semantic and statistical indexing. Furthermore, we propose a lightweight online alignment strategy utilizing curriculum learning to continually minimize the distributional gap between the draft and target models. Extensive evaluations across specialized domains (coding, law, and medicine) confirm that EvoSpec overcomes the limitations of static baselines. On EAGLE-3, it achieves a 1.13x speedup in these settings over the state-of-the-art static baseline FR-Spec, with 27\% lower memory overhead than standard online adaptation.
>
---
#### [replaced 008] Optimal Query Allocation in Extractive QA with LLMs: A Learning-to-Defer Framework with Theoretical Guarantees
- **分类: cs.CL; cs.LG; stat.ML**

- **简介: 该论文属于抽取式问答任务，解决LLMs在结构化文本选择中的效率问题。提出Learning-to-Defer框架，优化查询分配，提升可靠性并降低计算开销。**

- **链接: [https://arxiv.org/pdf/2410.15761](https://arxiv.org/pdf/2410.15761)**

> **作者:** Yannis Montreuil; Shu Heng Yeo; Axel Carlier; Lai Xing Ng; Wei Tsang Ooi
>
> **备注:** 25 pages, 17 main paper
>
> **摘要:** Large Language Models excel in generative tasks but exhibit inefficiencies in structured text selection, particularly in extractive question answering. This challenge is magnified in resource-constrained environments, where deploying multiple specialized models for different tasks is impractical. We propose a Learning-to-Defer framework that allocates queries to specialized experts, ensuring high-confidence predictions while optimizing computational efficiency. Our approach integrates a principled allocation strategy with theoretical guarantees on optimal deferral that balances performance and cost. Empirical evaluations on SQuADv1, SQuADv2, and TriviaQA demonstrate that our method enhances answer reliability while significantly reducing computational overhead, making it well-suited for scalable and efficient EQA deployment.
>
---
#### [replaced 009] CORE-T: COherent REtrieval of Tables for Text-to-SQL
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于文本到SQL任务，解决多表检索问题。通过生成表元数据和预计算缓存，提升表选择准确性和效率。**

- **链接: [https://arxiv.org/pdf/2601.13111](https://arxiv.org/pdf/2601.13111)**

> **作者:** Hassan Soliman; Vivek Gupta; Dan Roth; Iryna Gurevych
>
> **备注:** Preprint is revised and under review. Code and data available at: this https URL
>
> **摘要:** Realistic text-to-SQL workflows often require joining multiple tables. As a result, accurately retrieving the relevant set of tables becomes a key bottleneck for end-to-end performance. We study an open-book setting where queries must be answered over large, heterogeneous table collections pooled from many sources, without clean scoping signals such as database identifiers. Here, dense retrieval (DR) achieves high recall but returns many distractors, while join-aware alternatives often rely on extra assumptions and/or incur high inference overhead. We propose CORE-T, a scalable, training-free framework that enriches tables with LLM-generated purpose metadata and pre-computes a lightweight table-compatibility cache. At inference time, DR returns top-K candidates; a single LLM call selects a coherent, joinable subset, and a two-step additive adjustment stage restores strongly compatible tables. Across Bird, Spider, MMQA, and Beaver, CORE-T improves over DR by up to 22.7 points in table-selection F1 while returning up to 40% fewer tables, and by up to 24.4 points in multi-table execution accuracy, and uses 1.64-4.20x fewer total selection tokens than LLM-intensive baselines.
>
---
#### [replaced 010] Thinking Before Constraining: A Unified Decoding Framework for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决约束解码过早限制推理的问题。提出In-Writing方法，在生成前进行自由推理，后通过触发符实现结构化输出，提升准确性。**

- **链接: [https://arxiv.org/pdf/2601.07525](https://arxiv.org/pdf/2601.07525)**

> **作者:** Ngoc Trinh Hung Nguyen; Alonso Silva; Laith Zumot; Liubov Tupikina; Armen Aghasaryan; Mehwish Alam
>
> **备注:** v2-EMNLP
>
> **摘要:** Natural generation allows Large Language Models (LLMs) to produce free-form responses with rich reasoning, yet the lack of structure makes outputs difficult to verify. Conversely, constrained decoding ensures standardized formats but can inadvertently restrict reasoning capabilities by imposing constraints too early in the generation process. We propose a hybrid approach, namely In-Writing, that combines free-form reasoning and structured generation in a single call. The model first performs unconstrained reasoning and only applies structured decoding after a trigger token is generated, explicitly decoupling reasoning from formatting. We establish that our trigger-token strategies are able to virtually eradicate premature triggering, a failure mode in which constrained decoding interrupts on-going reasoning. Evaluations across diverse datasets covering classification and reasoning tasks demonstrate that our approach outperforms the state-of-the-art by achieving accuracy gains of up to 27% over natural generation. Our code are available at: this https URL.
>
---
#### [replaced 011] DFlash: Block Diffusion for Flash Speculative Decoding
- **分类: cs.CL**

- **简介: 该论文提出DFlash，解决LLM推理延迟问题，通过并行块扩散模型实现高效推测解码，提升速度与GPU利用率。**

- **链接: [https://arxiv.org/pdf/2602.06036](https://arxiv.org/pdf/2602.06036)**

> **作者:** Jian Chen; Yesheng Liang; Zhijian Liu
>
> **备注:** Accepted at ICML 2026. Camera-ready version. Code: this https URL
>
> **摘要:** Autoregressive large language models (LLMs) deliver strong performance but require inherently sequential decoding, leading to high inference latency and poor GPU utilization. Speculative decoding mitigates this bottleneck by using a fast draft model whose outputs are verified in parallel by the target LLM; however, existing methods still rely on autoregressive drafting, which remains sequential and limits practical speedups. Diffusion LLMs offer a promising alternative by enabling parallel generation, but current diffusion models typically underperform compared with autoregressive models. In this paper, we introduce DFlash, a speculative decoding framework that employs a lightweight block diffusion model for parallel drafting. By generating draft tokens in a single forward pass and conditioning the draft model on context features extracted from the target model, DFlash enables efficient drafting with high-quality outputs and higher acceptance rates. Experiments show that DFlash achieves over 6x lossless acceleration across a range of models and tasks, delivering up to 2.5x higher speedup than the state-of-the-art speculative decoding method EAGLE-3.
>
---
#### [replaced 012] DLT-Corpus: A Large-Scale Text Collection for the Distributed Ledger Technology Domain
- **分类: cs.CL**

- **简介: 该论文构建了DLT-Corpus，解决DLP领域文本资源不足的问题，包含科学文献、专利和社交媒体数据，用于分析技术发展与市场关系。**

- **链接: [https://arxiv.org/pdf/2602.22045](https://arxiv.org/pdf/2602.22045)**

> **作者:** Walter Hernandez Cruz; Peter Devine; Nikhil Vadgama; Paolo Tasca; Jiahua Xu
>
> **备注:** Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD '26)
>
> **摘要:** We introduce DLT-Corpus, the largest domain-specific text collection for Distributed Ledger Technology (DLT) research to date: 2.98 billion tokens from 22.12 million documents spanning scientific literature (37,440 publications), United States Patent and Trademark Office (USPTO) patents (49,023 filings), and social media (22 million posts). Existing Natural Language Processing (NLP) resources for DLT focus narrowly on cryptocurrency price prediction and smart contracts, leaving domain-specific language underexplored despite the sector's ~$3 trillion market capitalization and rapid technological evolution. We demonstrate DLT-Corpus' utility by analyzing patterns of technology emergence and market-innovation correlations. Findings reveal that technologies first appear in our scientific literature subset before reaching patents and social media, following traditional technology transfer patterns. While social media sentiment remains overwhelmingly bullish even during crypto winters, scientific and patent activity grows less tied to short-term sentiment, tracking overall market expansion in a virtuous cycle in which research precedes and enables economic growth that, in turn, funds further innovation. We release the DLT-Corpus and companion artifacts: LedgerBERT (+23% over BERT-base on DLT-specific Named Entity Recognition (NER) task), a sentiment analysis dataset of 23,301 crypto news headlines and descriptions, tools, and code.
>
---
#### [replaced 013] GraphLit: Learning Text-Enriched Dynamic Character Network Representations for Literary Study
- **分类: cs.CL**

- **简介: 该论文提出GraphLit，用于学习文学文本中的动态角色网络表示，解决角色互动与文本上下文结合的问题。通过自监督学习提升文学分析效果。**

- **链接: [https://arxiv.org/pdf/2605.28643](https://arxiv.org/pdf/2605.28643)**

> **作者:** Gaspard Michel; Elena V. Epure; Romain Hennequin; Christophe Cerisara; Mirella Lapata
>
> **摘要:** Methods to represent literary texts as graphs or sequences of graphs mainly focus on representing character interactions, and often overlook another crucial aspect: the textual context in which characters interact. We introduce Dynamic Heterogeneous Character Networks (DHCNs), which organize long novels into temporally localized heterogeneous graphs that align characters with their textual contexts. We extract around 20,000 DHCNs from Project Gutenberg, and propose GraphLit, a self-supervised learning framework that learns rich literary representations through a masked graph autoencoder objective. Across a wide-range of 12 character-related tasks, GraphLit improves over text-only and graph-only baselines, particularly on tasks requiring contextual understanding. Finally, we demonstrate the applicability of DHCNs and GraphLit for literary analysis by studying the link between narrative non-linearity and dynamic social features.
>
---
#### [replaced 014] Reasoning Theater: Disentangling Model Beliefs from Chain-of-Thought
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究模型在推理过程中的行为，旨在区分真实推理与“表演性推理”。通过分析模型激活和回答时机，提出一种基于探测的早期退出方法，提升效率。任务为模型推理分析。**

- **链接: [https://arxiv.org/pdf/2603.05488](https://arxiv.org/pdf/2603.05488)**

> **作者:** Siddharth Boppana; Annabel Ma; Max Loeffler; Raphael Sarfati; Eric Bigelow; Atticus Geiger; Owen Lewis; Jack Merullo
>
> **摘要:** We provide evidence of performative chain-of-thought (CoT) in reasoning models, where a model becomes strongly confident in its final answer, but continues generating tokens without revealing its internal belief. Our analysis compares activation probing, early forced answering, and a CoT monitor across two large models (DeepSeek-R1 671B & GPT-OSS 120B) and find task difficulty-specific differences: The model's final answer is decodable from activations far earlier in CoT than a monitor is able to say, especially for easy recall-based MMLU questions. We contrast this with genuine reasoning in difficult multihop GPQA-Diamond questions. Despite this, inflection points (e.g., backtracking, 'aha' moments) occur almost exclusively in responses where probes show large belief shifts, suggesting these behaviors track genuine uncertainty rather than learned "reasoning theater." Finally, probe-guided early exit reduces tokens by up to 80% on MMLU and 30% on GPQA-Diamond with similar accuracy, positioning attention probing as an efficient tool for detecting performative reasoning and enabling adaptive computation.
>
---
#### [replaced 015] What Exactly do Children Receive in Language Acquisition? A Case Study on CHILDES with Automated Detection of Filler-Gap Dependencies
- **分类: cs.CL**

- **简介: 该论文研究儿童语言习得中的填充-缺口结构，旨在解决输入数据量化困难的问题。通过自动检测语料库中的三种构造，分析儿童语言输入与产出轨迹。任务属于语言习得与自然语言处理交叉领域。**

- **链接: [https://arxiv.org/pdf/2603.02082](https://arxiv.org/pdf/2603.02082)**

> **作者:** Zhenghao Herbert Zhou; William Dai; Maya Viswanathan; Simon Charlow; R. Thomas McCoy; Robert Frank
>
> **备注:** Camera-ready version accepted to CoNLL 2026
>
> **摘要:** Children's acquisition of filler-gap dependencies has been argued by some to depend on innate grammatical knowledge, while others suggest that the distributional evidence available in child-directed speech suffices. Unfortunately, the relevant input is difficult to quantify at scale with fine granularity, making this question difficult to resolve. We present a system that identifies three core filler-gap constructions in spoken English corpora -- matrix wh-questions, embedded wh-questions, and relative clauses -- and further identifies the extraction site (i.e., subject vs. object vs. adjunct). Our approach combines constituency and dependency parsing, leveraging their complementary strengths for construction classification and extraction site identification. We validate the system on human-annotated data and find that it scores well across most categories. Applying the system to 57 English CHILDES corpora, we are able to characterize children's filler-gap input and their filler-gap production trajectories over the course of development, including construction-specific frequencies and extraction-site asymmetries. The resulting fine-grained labels enable future work in both acquisition and computational studies, which we demonstrate with a case study using filtered corpus training with language models.
>
---
#### [replaced 016] Mindscape-Aware Retrieval Augmented Generation for Improved Long Context Understanding
- **分类: cs.CL**

- **简介: 该论文提出MiA-RAG，解决长文本理解任务中缺乏全局语义引导的问题。通过构建全局语义表示，统一检索与生成过程，提升长文本推理能力。**

- **链接: [https://arxiv.org/pdf/2512.17220](https://arxiv.org/pdf/2512.17220)**

> **作者:** Yuqing Li; Jiangnan Li; Zheng Lin; Ziyan Zhou; Junjie Wu; Weiping Wang; Jie Zhou; Mo Yu
>
> **摘要:** Humans understand long and complex texts by relying on a holistic semantic representation of the content. This global view helps organize prior knowledge, interpret new information, and integrate evidence dispersed across a document, as revealed by the Mindscape-Aware Capability of humans in psychology. Current Retrieval-Augmented Generation (RAG) systems lack such guidance and therefore struggle with long-context tasks. In this paper, we propose Mindscape-Aware RAG (MiA-RAG), the first framework to formulate mindscape-aware retrieval and generation as a unified conditioning paradigm for LLM-based RAG. MiA-RAG builds a mindscape through hierarchical summarization and conditions both retrieval and generation on this global semantic representation. This enables the retriever to form enriched query embeddings and the generator to reason over retrieved evidence within a coherent global context. We evaluate MiA-RAG across diverse long-context and bilingual benchmarks for evidence-based understanding and global sense-making. It consistently surpasses baselines, and further analysis shows that it aligns local details with a coherent global representation, enabling more human-like long-context retrieval and reasoning.
>
---
#### [replaced 017] Less is Enough: Synthesizing Diverse Data in LLM Feature Space with Sparse Autoencoders
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，旨在提升大语言模型的下游性能。通过引入FAC度量特征空间多样性，提出FAC Synthesis框架，生成更具代表性的数据。**

- **链接: [https://arxiv.org/pdf/2602.10388](https://arxiv.org/pdf/2602.10388)**

> **作者:** Zhongzhi Li; Xuansheng Wu; Yijiang Li; Lijie Hu; Ninghao Liu
>
> **摘要:** The diversity of post-training data is critical for effective downstream performance in large language models (LLMs). Many existing approaches to constructing post-training data quantify diversity using text-based metrics that capture linguistic variation, but such metrics provide only weak signals for the task-relevant features that determine downstream performance. In this work, we introduce Feature Activation Coverage (FAC) which measures data diversity in an interpretable feature space. Building upon this metric, we further propose a diversity-driven data synthesis framework, named FAC Synthesis, that first uses a sparse autoencoder to identify missing features from a seed dataset, and then generates synthetic samples that explicitly reflect these features. Experiments show that our approach consistently improves both data diversity and downstream performance on various tasks, including instruction following, toxicity detection, reward modeling, and behavior steering. Interestingly, we identify a shared, interpretable feature space across model families (i.e., LLaMA, Mistral, and Qwen), enabling cross-model knowledge transfer. Our work provides a solid and practical methodology for exploring data-centric optimization of LLMs.
>
---
#### [replaced 018] Looking Beyond Text: Reducing Language bias in Large Vision-Language Models via Multimodal Dual-Attention and Soft-Image Guidance
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言模型任务，旨在解决LVLMs因语言偏见导致的图像理解不足和幻觉问题。提出LACING框架，通过多模态双注意力和软图像引导提升视觉理解。**

- **链接: [https://arxiv.org/pdf/2411.14279](https://arxiv.org/pdf/2411.14279)**

> **作者:** Haozhe Zhao; Shuzheng Si; Liang Chen; Yichi Zhang; Maosong Sun; Mingjia Zhang; Baobao Chang
>
> **备注:** EMNLP 2025
>
> **摘要:** Large vision-language models (LVLMs) have achieved impressive results in various vision-language tasks. However, despite showing promising performance, LVLMs suffer from hallucinations caused by language bias, leading to diminished focus on images and ineffective visual comprehension. We identify two primary reasons for this bias: 1. Different scales of training data between the pretraining stage of LLM and multimodal alignment stage. 2. The learned inference bias due to short-term dependency of text data. Therefore, we propose LACING, a systemic framework designed to address the language bias of LVLMs with muLtimodal duAl-attention meChanIsm (MDA) aNd soft-image Guidance (IFG). Specifically, MDA introduces a parallel dual-attention mechanism that enhances the integration of visual inputs across the model. IFG introduces a learnable soft visual prompt during training and inference to replace visual inputs, designed to compel LVLMs to prioritize text inputs. Then, IFG further proposes a novel decoding strategy using the soft visual prompt to mitigate the model's over-reliance on adjacent text inputs. Comprehensive experiments demonstrate that our method effectively debiases LVLMs from their language bias, enhancing visual comprehension and reducing hallucinations without requiring additional training resources or data. The code and model are available at [this http URL](this https URL).
>
---
#### [replaced 019] Long-Context Modeling with Dynamic Hierarchical Sparse Attention for Memory-Constrained LLM Inference
- **分类: cs.CL**

- **简介: 该论文属于长文本处理任务，解决内存受限下大模型推理效率问题。提出DHSA框架，通过动态稀疏注意力提升性能，实现高效且准确的长上下文建模。**

- **链接: [https://arxiv.org/pdf/2510.24606](https://arxiv.org/pdf/2510.24606)**

> **作者:** Siheng Xiong; Joe Zou; Faramarz Fekri; Yae Jee Cho
>
> **备注:** ICML26 (Spotlight)
>
> **摘要:** The quadratic cost of attention limits the scalability of long-context LLMs, especially under limited hardware memory budgets. While attention is often sparse, existing static sparse methods cannot adapt to task- or input-dependent variations, and recent dynamic approaches rely on predefined templates or heuristics that may sacrifice generality. We propose Dynamic Hierarchical Sparse Attention (DHSA), a data-driven framework that predicts attention sparsity online while keeping the LLM backbone frozen. DHSA performs hierarchical routing by estimating importance at the chunk level and propagating it to token-level interactions, preserving causally important dependencies while enabling efficient sparsification. Across Needle-in-a-Haystack test, LongBench and RULER, DHSA maintains near-dense accuracy in highly sparse regimes, achieving 12--20% relative accuracy gains over Block Sparse Attention at comparable prefill cost. With a memory-efficient tiled backend, DHSA delivers up to $10\times$ prefill speedup at 128K context length. On LLaMA-3.1-8B (4-bit), DHSA scales to 100K context on a single 24GB GPU, where dense attention fails. We provide complementary GPU and CPU backends, enabling DHSA to run across diverse hardware environments and multiple open-weight model families. These results demonstrate DHSA as an efficient and adaptable solution for memory-constrained long-context LLM inference.
>
---
#### [replaced 020] Less Is More: Elevating RAG via Performance-Driven Context Compression
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于RAG任务，旨在解决检索文档过长导致计算成本高和性能下降的问题。提出CORE-RAG框架，通过性能驱动学习实现高效上下文压缩。**

- **链接: [https://arxiv.org/pdf/2508.19282](https://arxiv.org/pdf/2508.19282)**

> **作者:** Ziqiang Cui; Yunpeng Weng; Xing Tang; Peiyang Liu; Shiwei Li; Bowei He; Jiamin Chen; Yansen Zhang; Xiuqiang He; Chen Ma
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** Retrieval-Augmented Generation (RAG) has emerged as a promising paradigm for improving the timeliness of knowledge updates and the factual accuracy of large language models. However, incorporating a large volume of retrieved documents significantly increases input length, leading to prohibitive computational costs. Existing compression approaches often compromise task performance, primarily due to their reliance on predefined heuristics. These heuristics fail to ensure that the compressed context is conducive to the generation tasks. To address these limitations, we propose CORE-RAG, a novel framework for context compression in RAG systems. CORE eliminates reliance on proxy heuristics through a performance-driven learning framework, which directy utilizes task performance as a feedback signal to iteratively refine the compressor policy. Prior to this optimization process, we incorporate a knowledge distillation phase to initialize the compressor with a robust policy. Extensive experiments demonstrate the superiority of our approach. At a high compression ratio of 3%, CORE not only avoids performance degradation but also improves the average Exact Match (EM) score by 3.3 points compared to using full documents. Our code is available at this https URL.
>
---
#### [replaced 021] CausaLab: A Scalable Environment for Interactive Causal Discovery Toward AI Scientists
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出CausaLab，用于评估LLM在因果发现中的表现。任务是区分预测成功与因果理解，解决当前模型缺乏因果推理能力的问题。工作包括设计实验环境并验证模型表现。**

- **链接: [https://arxiv.org/pdf/2605.26029](https://arxiv.org/pdf/2605.26029)**

> **作者:** Junlin Yang; Dylan Zhang; Xiangchen Song; Qirun Dai; Xiao Liu; Yuen Chen; Aniket Vashishtha; Jing Shi; Chenhao Tan; Hao Peng
>
> **摘要:** We introduce CausaLab, a scalable environment for evaluating interactive causal discovery by LLM agents. Unlike prior evaluations, CausaLab evaluates both whether an agent can solve a problem using causal evidence and whether its answer is grounded in a faithful recovered causal mechanism. Each episode places an agent in a synthetic laboratory: it receives prior measurement records, intervenes on a manipulator crystal, and predicts the resonance frequency of a held-out reactor crystal governed by the same mechanism. The hidden data-generating process is a randomly sampled structural causal model (SCM), so success requires recovering both a causal graph and structural equations rather than recalling prior knowledge. Experiments show a persistent gap between prediction and mechanism recovery: in the purely observational 6-node setting, GPT-5.2-high reaches 92% task accuracy but only 0.471 all-edge $F_1$. Mixed observation-intervention strategies improve structural fidelity, while pure intervention remains difficult even for strong agents. We identify premature stopping as a major weakness and show that consistency verification mitigates it. CausaLab therefore separates predictive success from causal understanding and exposes current LLM agents' limits as experimental causal reasoners.
>
---
#### [replaced 022] From AR to Diffusion: Efficiently Adapting Large Language Models with Strictly Causal and Elastic Horizons
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言生成任务，解决AR模型与扩散模型不兼容的问题。提出FLUID框架，实现高效适配，降低训练成本。**

- **链接: [https://arxiv.org/pdf/2605.27387](https://arxiv.org/pdf/2605.27387)**

> **作者:** Xiangyu Ma; Teng Xiao; Zuchao Li; Lefei Zhang
>
> **备注:** Accepted by ACL 2026
>
> **摘要:** Diffusion models promise efficient parallel text generation but rely on bidirectional attention, creating a structural mismatch with pre-trained Autoregressive (AR) models. This incompatibility precludes reusing robust AR priors, necessitating prohibitive pre-training from scratch. To bridge this gap, we propose FLUID, a framework that efficiently adapts AR backbones to the diffusion paradigm. By enforcing Strictly Causal Alignment, FLUID enables seamless initialization from standard GPT-style checkpoints, circumventing the need for massive pre-training. Furthermore, we introduce Elastic Horizons, an entropy-driven mechanism that dynamically modulates denoising strides based on local information density rather than fixed schedules. Experiments demonstrate that FLUID achieves state-of-the-art performance while reducing training costs by orders of magnitude, effectively reconciling established AR foundations with efficient parallel generation. Our code is available at this https URL.
>
---
#### [replaced 023] Enhancing LLM Medical Coding with Structured External Knowledge
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于医疗编码任务，旨在解决LLM在医学编码中依赖内部知识易产生幻觉及无法及时更新的问题。通过引入结构化外部知识提升编码准确性。**

- **链接: [https://arxiv.org/pdf/2605.27377](https://arxiv.org/pdf/2605.27377)**

> **作者:** Yidong Gan; David D. Nguyen; Yang Lin; Peter Zhong; Thanh Vu; Long Duong; Yuan-Fang Li
>
> **摘要:** Accurate medical coding requires consulting authoritative resources such as the ICD tabular list and coding guidelines. Existing LLM-based automated methods largely rely on LLMs' internal knowledge, which is prone to hallucination and cannot keep pace with guideline updates. We introduce RAG-Coding, an agentic, training-free method that augments LLMs with structured external knowledge: the tabular list is encoded as a knowledge graph capturing hierarchical and instructional code relationships, and the guidelines are distilled into concise, code-specific summaries rather than retrieved as raw text. To enable our study, we also introduce MDACE-2025, expert re-annotations of the MDACE dataset under the 2025 ICD-10-CM/PCS guidelines, adding code sequencing and justification comments. On MDACE, RAG-Coding outperforms the best LLM-based baseline by 3--13\% in micro-F1 across five LLM backbones, and achieves comparable micro- and macro-F1 to the supervised state-of-the-art, with higher recall ($+$11\%) at the cost of precision ($-$6\%). On MDACE-2025, RAG-Coding outperforms all baselines, demonstrating effective generalisation to updated guidelines. Ablations confirm stepwise gains, highlighting the importance of integrating structured external knowledge for LLM-based medical coding.
>
---
#### [replaced 024] Bridge-RAG: An Abstract Bridge Tree Based Retrieval Augmented Generation Algorithm
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于信息检索与生成任务，旨在提升RAG框架的准确性和效率。通过引入抽象树结构和Cuckoo Filter，解决检索精度低和计算开销大的问题。**

- **链接: [https://arxiv.org/pdf/2603.26668](https://arxiv.org/pdf/2603.26668)**

> **作者:** Zihang Li; Wenjun Liu; Yikun Zong; Jiawen Tao; Siying Dai; Songcheng Ren; Zirui Liu; Yuhang Wang; Yanbing Jiang; Tong Yang
>
> **摘要:** As an important paradigm for enhancing the generation quality of Large Language Models (LLMs), retrieval-augmented generation (RAG) faces the two challenges regarding retrieval accuracy and computational efficiency. This paper presents a novel RAG framework called Bridge-RAG. To overcome the accuracy challenge, we introduce the concept of abstract to bridge query entities and document chunks, providing robust semantic understanding. We organize the abstracts into a tree structure and design a multi-level retrieval strategy to ensure the inclusion of sufficient contextual information. While this hierarchical organization substantially improves answer quality, traversing the tree to locate the abstracts that contain a query entity inevitably introduces additional retrieval overhead. To restore retrieval efficiency, we further integrate the Cuckoo Filter in CFT-RAG, which provides O(1) entity lookup and naturally fits the entity-to-abstract pathway of our framework. Extensive experiments show that Bridge-RAG achieves consistent accuracy improvements across all metrics and up to $1.9\times$ faster retrieval compared to structured RAG baselines.
>
---
#### [replaced 025] TajikNLP: An Open-Source Toolkit for Comprehensive Text Processing of Tajik (Cyrillic Script)
- **分类: cs.CL**

- **简介: 该论文提出TajikNLP，解决塔吉克语（西里尔字母）资源匮乏问题，提供全面的文本处理工具包，包含分词、标注、情感分析等功能。**

- **链接: [https://arxiv.org/pdf/2605.04583](https://arxiv.org/pdf/2605.04583)**

> **作者:** Mullosharaf K. Arabov
>
> **备注:** Accepted to CLIB 2026
>
> **摘要:** The Tajik language, written in Cyrillic script, remains severely under-resourced in terms of publicly available natural language processing (NLP) toolkits, hindering both linguistic research and applied development. This paper introduces TajikNLP, an open-source Python library that provides the first comprehensive pipeline for processing authentic Tajik text while preserving the original Cyrillic orthography. The library implements a modular architecture centered around a unified Doc object, enabling sequential application of components for cleaning, normalization, tokenization (including subword BPE), morphemic segmentation, part-of-speech tagging, stemming, lemmatization, and sentence splitting. A novel unified morphology engine is introduced, offering controlled and deep analysis modes that significantly improve handling of Tajik's agglutinative nominal and verbal inflections. The release further incorporates a lexicon-based sentiment analyser and pre-trained Word2Vec/FastText embeddings loaded directly from the Hugging Face Hub. To ensure reproducibility and facilitate future research, four accompanying linguistic datasets -- a POS-tagged corpus (52.5k entries), a sentiment lexicon (3.5k entries), a toponym gazetteer (5.6k entries), and a personal names dataset (3.8k entries) -- have been openly published under permissive licenses. The library's reliability is validated by an extensive test suite of 616 automated tests achieving 93% source code coverage. TajikNLP thus establishes a foundational technological infrastructure for Tajik language processing, lowering the barrier to entry for both academic and industrial applications in low-resource Cyrillic-script environments.
>
---
#### [replaced 026] Lexical categories of stem-forming roots in Mapudüngun verb forms
- **分类: cs.CL**

- **简介: 该论文属于语言学分析任务，旨在验证Mapudüngun语动词根的词类分类，解决其词性不确定性问题，通过修订词类提高计算分析系统的准确性。**

- **链接: [https://arxiv.org/pdf/2502.07623](https://arxiv.org/pdf/2502.07623)**

> **作者:** Andrés Chandía
>
> **备注:** 36 pages, 2 large tables, 2 sample tables
>
> **摘要:** After developing a computational system for morphological analysis of the Mapuche language, and evaluating it with texts from various authors and styles, it became necessary to verify the linguistic assumptions of the source used as the basis for implementing this tool. In the present work, the primary focus is on the lexical category classification of Mapudüngun roots recognised as verbal in the source utilised for the development of the morphological analysis system. The results of this lexical category revision directly benefit the computational analyser, as they are implemented as soon as they are verified. Additionally, it is hoped that these results will help clarify some uncertainties about lexical categories in the Mapuche language. This work addresses a preliminary task to identify the valency of true verbal roots, the results of which will be presented in a subsequent work that complements this article.
>
---
#### [replaced 027] Bridging the Semantic Gap for Categorical Data Clustering via Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于类别数据聚类任务，旨在解决定性数据相似性度量困难的问题。通过引入外部语义信息增强数据表示，提升聚类效果。**

- **链接: [https://arxiv.org/pdf/2601.01162](https://arxiv.org/pdf/2601.01162)**

> **作者:** Zihua Yang; Xin Liao; Yiqun Zhang; Yiu-ming Cheung
>
> **备注:** Accepted to ICPR2027
>
> **摘要:** Qualitative data are widespread in domains such as healthcare, marketing, and bioinformatics, where clustering offers a fundamental tool for pattern discovery. A core difficulty of qualitative-data clustering lies in measuring similarity among attribute values that carry no inherent ordering or distance. To recover such relationships, existing studies typically rely on within-dataset co-occurrence statistics. This statistical route, however, becomes unreliable once the sample size is small, and the semantic context of each value is therefore left underexploited. Motivated by this limitation, this paper proposes BREVE (Balanced Representation via External Value Enrichment), a clustering framework that enriches each qualitative value with extra semantic dimensions drawn from an external knowledge base. That is, every unique value is expanded by a dense embedding that encodes its semantic content. To prevent the original value identity from being diluted by the added dimensions, a lightweight one-hot component is further appended. An adaptive weight, guided by cluster compactness, then determines how strongly the enrichment dimensions enter the final representation. With this design, experiments on eight benchmark datasets yield an average ARI rank of 1.3 against seven representative competitors.
>
---
#### [replaced 028] Understanding Fact Recall in Language Models: Why Two-Stage Training Encourages Memorization but Mixed Training Teaches Knowledge
- **分类: cs.CL**

- **简介: 该论文属于语言模型知识注入任务，研究为何两阶段训练导致记忆而混合训练提升事实回忆。通过对比两种训练方式，揭示混合训练通过梯度一致性实现更优 recall。**

- **链接: [https://arxiv.org/pdf/2505.16178](https://arxiv.org/pdf/2505.16178)**

> **作者:** Ying Zhang; Benjamin Heinzerling; Dongyuan Li; Kentaro Inui
>
> **摘要:** While fine-tuning is the standard for injecting factual knowledge into large language models (LLMs), the mechanisms enabling reliable fact recall via unseen queries remain poorly understood. Common two-stage training strategies, which sequentially train on fact storage and query formats, often cause rote memorization. In contrast, mixed training jointly optimizes both formats and exhibits superior generalized recall. We investigate this success by comparing the two paradigms across 2.8$\sim$4B LLMs and identify the core mechanism: the joint optimization objective in mixed training induces gradient consistency across storage and query formats. This in turn drives the representation consistency between the two formats, establishing a format-invariant retrieval process that maps unseen queries to stored facts. In contrast, the lack of such an objective in two-stage training results in inconsistent representations and failed recall. The consistency further localizes to the parameters updated by both formats, a set that is substantially larger under mixed training than under two-stage training. At the input level, the consistency leaves an interpretable signature: mixed training encodes facts in storage format from subject-relation tokens, the same components available in queries, while two-stage training relies on the full context. Our findings characterize the mechanisms of fact recall and offer mechanistic foundation for optimizing knowledge injection in LLMs.
>
---
#### [replaced 029] When 2D Tasks Meet 1D Serialization: On Serialization Friction in Structured Tasks
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究2D任务在1D文本序列化下的表示问题，属于自然语言处理与结构化任务领域。它探讨了布局依赖关系在1D序列中丢失带来的影响，通过实验验证了序列化对任务性能的负面影响。**

- **链接: [https://arxiv.org/pdf/2604.27272](https://arxiv.org/pdf/2604.27272)**

> **作者:** Chung-Hsiang Lo; Lu Li; Diji Yang; Tianyu Zhang; Yunkai Zhang; Yoshua Bengio; Yi Zhang
>
> **摘要:** In the LLM era, many symbolic and structured problems are presented to models through 1D text serialization. Yet some such problems are natively two-dimensional: their relevant relations, such as row--column correspondence or spatial adjacency, are defined by position in a 2D layout rather than by sequential order. This raises a representational question: does preserving the same symbolic entries in a 1D sequence also preserve the relational structure needed for computation? We study this issue through the lens of serialization friction: the representational mismatch in which the same underlying task instances and entries are still present, but relations that depend on layout become implicit under 1D serialization. The study uses a controlled synthetic testbed of three tasks: matrix transpose, Conway's Game of Life, and LU decomposition. In each task, the same instances are presented either as 1D text serialization or as their native 2D layout rendered as an image. Across this testbed, 1D serialization degrades more sharply as task size grows, and errors under serialization exhibit spatially structured patterns, suggesting that this presentation choice is consequential within our testbed. To further interpret these results, we add supplementary analyses that include a within-visual probe and an additional comparison of the two input presentations under the mixed-training transpose setting. These findings suggest that, for layout-defined tasks, reducing inputs to 1D serialization is not a neutral choice of representation.
>
---
#### [replaced 030] CriticalKV: Optimizing KV Cache Eviction from an Output Perturbation Perspective
- **分类: cs.CL**

- **简介: 该论文属于语言模型优化任务，解决KV缓存存储成本高的问题。通过分析输出扰动，提出一种优化的缓存淘汰算法，显著降低压缩损失。**

- **链接: [https://arxiv.org/pdf/2502.03805](https://arxiv.org/pdf/2502.03805)**

> **作者:** Yuan Feng; Junlin Lv; Haoyu Guo; Yukun Cao; S Kevin Zhou; Xike Xie
>
> **备注:** ICML 2026
>
> **摘要:** Large language models have revolutionized natural language processing but face significant challenges of high storage and runtime costs, due to the transformer architecture's reliance on self-attention, particularly the large KV cache for long-sequence inference. Recent efforts to reduce KV cache size by pruning less critical entries based on attention weights remain empirical and lack formal grounding. This paper presents a formal study on identifying critical KV cache entries by analyzing attention output perturbation. Our analysis reveals that, beyond attention weights, the value states within KV entries and pretrained parameter matrices are also crucial. Based on this, we propose a perturbation-constrained selection algorithm that optimizes the worst-case output perturbation to identify critical entries. We demonstrate that our algorithm is a universal, plug-and-play enhancement that incurs negligible computational overhead. When integrated with three state-of-the-art cache eviction methods on three distinct LLMs, our algorithm significantly reduces the compression loss by more than \textit{half} on average across 29 datasets from the Ruler and LongBench benchmarks. Further perturbation analysis, at both the head and layer levels, confirms the principles underlying our effectiveness. This work offers a new, formally grounded perspective to cache eviction , opening promising avenues for future research. The code is publicly available at this https URL.
>
---
#### [replaced 031] Scaling Small Agents Through Strategy Auctions
- **分类: cs.MA; cs.AI; cs.CL**

- **简介: 该论文研究如何通过策略拍卖提升小模型在复杂任务中的表现，解决小模型难以应对高复杂度工作流的问题。提出SALE框架，实现高效任务分配与持续优化。**

- **链接: [https://arxiv.org/pdf/2602.02751](https://arxiv.org/pdf/2602.02751)**

> **作者:** Lisa Alazraki; William F. Shen; Yoram Bachrach; Akhil Mathur
>
> **备注:** ICML 2026
>
> **摘要:** Small language models are increasingly viewed as a promising, cost-effective approach to agentic AI, with proponents claiming they are sufficiently capable for agentic workflows. However, while smaller agents can closely match larger ones on simple tasks, it remains unclear how their performance scales with task complexity, when large models become necessary, and how to better leverage small agents for long-horizon workloads. In this work, we empirically show that small agents' performance fails to scale with task complexity on deep search and coding tasks, and we introduce Strategy Auctions for Workload Efficiency (SALE), an agent framework inspired by freelancer marketplaces. In SALE, agents bid with short strategic plans, which are scored by a systematic cost-value mechanism and refined via a shared auction memory, enabling per-task routing and continual self-improvement without training a separate router or running all models to completion. Across deep search and coding tasks of varying complexity, SALE reduces reliance on the largest agent by 52%, lowers overall cost by 35%, and consistently improves upon the largest agent's pass@1 with only a negligible overhead beyond executing the final trace. In contrast, established routers that rely on task descriptions either underperform the largest agent or fail to reduce cost, often both, underscoring their poor fit for agentic workflows. These results suggest that while small agents may be insufficient for complex workloads, they can be effectively "scaled up" through coordinated task allocation and test-time self-improvement. More broadly, they motivate a systems-level view of agentic AI in which performance gains come less from ever-larger individual models and more from market-inspired coordination mechanisms that organize heterogeneous agents into efficient, adaptive ecosystems.
>
---
#### [replaced 032] Mining or Synthesis? Rethinking Exploration Efficiency in Iterative Alignment of Mathematical Reasoning
- **分类: cs.CL**

- **简介: 该论文属于语言模型对齐任务，解决迭代对齐中探索效率低的问题。通过引入PACE框架，用低预算探索替代高N采样，提升效率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.05370](https://arxiv.org/pdf/2602.05370)**

> **作者:** Jun Rao; Zixiong Yu; Xuebo Liu; Guhan Chen; Jing Li; Hejin Wang; Jiansheng Wei; Xiaojun Meng; Min Zhang
>
> **摘要:** Iterative Direct Preference Optimization (DPO) has emerged as a widely used paradigm for aligning Large Language Models on reasoning tasks. Existing approaches typically rely on Best-of-N sampling ($N\geq8$) to mine positive trajectories from the distribution tail. In this work, we show that in mathematical reasoning, increasing $N$ yields diminishing returns while increasing verifier-induced false-positive risk and the distribution shift required for policy updates. To address this, we introduce PACE (Proximal Alignment via Corrective Exploration), a generation-based corrective framework that replaces exhaustive mining with low-budget exploration ($2\leq N\leq3$). Rather than searching for increasingly rare positive samples, PACE synthesizes high-fidelity preference pairs from failed explorations through corrective hindsight refinement and verification-guided filtering. Empirically, PACE matches or exceeds the performance of DPO-R1 ($N=16$) while using about $1/5$ of the compute, and remains robust under 20\% label corruption, where high-$N$ baselines exhibit substantially higher noise exploitation.
>
---
#### [replaced 033] Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究隐式链式思维的因果结构，解决其内部步骤的可解释性问题。通过结构因果模型分析两种方法，探讨步骤必要性、影响传播及答案模式保留情况。**

- **链接: [https://arxiv.org/pdf/2602.08783](https://arxiv.org/pdf/2602.08783)**

> **作者:** Zirui Li; Xuefeng Bai; Kehai Chen; Yizhi Li; Jian Yang; Chenghua Lin; Min Zhang
>
> **备注:** Accepted to ICML 2026; 25 pages, 23 figures
>
> **摘要:** Latent or continuous chain-of-thought methods replace explicit textual rationales with a number of internal latent steps, but these intermediate computations are difficult to evaluate beyond correlation-based probes. In this paper, we view latent chain-of-thought as a manipulable causal process in representation space by modeling latent steps as variables in a structural causal model (SCM) and analyzing their effects through step-wise do-interventions. We study two representative paradigms (i.e., Coconut and CODI) on both mathematical and general reasoning tasks to investigate three key questions: (1) which steps are causally necessary for correctness and when answers become decodable early; (2) how influence propagates across steps and how this structure compares to explicit CoT; and (3) whether intermediate trajectories retain competing answer modes and how output-level commitment differs from representational commitment across steps. We find that latent-step budgets behave less like homogeneous extra depth and more like staged functionality with non-local routing, and we identify a persistent gap between early output bias and late representational commitment. These results motivate mode-conditional and stability-aware analyses, together with corresponding training/decoding objectives, as more reliable tools for interpreting and improving latent reasoning systems. Code is available at this https URL.
>
---
#### [replaced 034] ShapleyLaw: A Game-Theoretic Approach to Multilingual Scaling Laws
- **分类: cs.CL**

- **简介: 该论文属于多语言预训练任务，旨在解决语言混合比例优化问题。通过引入博弈论方法，量化跨语言迁移效果，提出ShapleyLaw模型提升性能预测与比例优化效果。**

- **链接: [https://arxiv.org/pdf/2603.17945](https://arxiv.org/pdf/2603.17945)**

> **作者:** Xuyang Cao; Qianying Liu; Chuan Xiao; Yusuke Oda; Jiayi Wang; Pontus Stenetorp; Daisuke Kawahara; Makoto Onizuka; Sadao Kurohashi; Shuyuan Zheng
>
> **备注:** 18 pages
>
> **摘要:** In multilingual pretraining, the test loss of a pretrained model is heavily influenced by the proportion of each language in the pretraining data, namely the \textit{language mixture ratios}. Multilingual scaling laws can predict the test loss under different language mixture ratios and can therefore be used to estimate the optimal ratios. However, the current approaches to multilingual scaling laws do not measure the \textit{cross-lingual transfer} effect, resulting in suboptimal mixture ratios. In this paper, we consider multilingual pretraining as a cooperative game in which each language acts as a player that jointly contributes to pretraining, gaining the resulting reduction in test loss as the payoff. Consequently, from the perspective of cooperative game theory, we quantify the cross-lingual transfer from each language by its contribution in the game, and propose a game-theoretic multilingual scaling law called \textit{ShapleyLaw}. Our experiments show that ShapleyLaw outperforms baseline methods in model performance prediction and language mixture optimization.
>
---
#### [replaced 035] S-MARC: Causal Streaming Reasoning for Full-Duplex Conversational Behavior Modeling
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出S-MARC，用于全双工对话中的行为建模与推理。解决对话中因果关系与时间依赖问题，通过流式处理和图结构实现高效推理。**

- **链接: [https://arxiv.org/pdf/2602.11065](https://arxiv.org/pdf/2602.11065)**

> **作者:** Dingkun Zhou; Shuchang Pan; Jiachen Lian; Siddharth Banerjee; Sarika Pasumarthy; Dhruv Hebbar; Siddhant Patel; Zeyi Austin Li; Kan Jen Cheng; Sanay Bordia; Krish Patel; Akshaj Gupta; Tingle Li; Gopala Anumanchipalli
>
> **摘要:** Human conversation is organized by an implicit chain of thought and manifests as temporally structured conversational behaviors. Capturing this perceptual pathway is critical for building natural full-duplex interactive systems. We propose S-MARC (Streaming Causal Modeling and Reasoning for Conversation), a streaming, causal, and hierarchical framework for conversational behavior modeling and reasoning. By formalizing the intent-to-action pathway, S-MARC predicts high-level communicative functions and low-level interaction behaviors while modeling their causal and temporal dependencies. To support this setting, we construct a high-quality corpus that pairs controllable, event-rich duplex dialogue data with behavior labels. S-MARC organizes streaming predictions into a continuously evolving graph structure, generates concise justifications for its decisions, and dynamically optimizes its reasoning process. Experiments on synthetic and real duplex dialogues show that S-MARC achieves robust behavior detection, produces interpretable reasoning chains, and establishes a benchmark foundation for conversational reasoning in full-duplex spoken dialogue systems.
>
---
#### [replaced 036] Cognitive Loop of Thought: Reversible Hierarchical Markov Chain for Efficient Mathematical Reasoning
- **分类: cs.CL**

- **简介: 该论文提出一种基于可逆分层马尔可夫链的思维链框架CLoT，解决长思维链导致的计算效率低和推理能力弱的问题。**

- **链接: [https://arxiv.org/pdf/2604.06805](https://arxiv.org/pdf/2604.06805)**

> **作者:** Jia-Chen Zhang; Yu-Jie Xiong; Zheng Zhou
>
> **摘要:** Multi-step Chain-of-Thought (CoT) has significantly advanced the mathematical reasoning capabilities of LLMs by leveraging explicit reasoning steps. However, the widespread adoption of Long CoT often results in sequence lengths that exceed manageable computational limits. While existing approaches attempt to alleviate this by reducing KV Cache redundancy via Markov chain-like structures, they introduce two critical limitations: inherent memorylessness (loss of context) and limited backward reasoning capability. To address these limitations, we propose a novel Chain-of-Thought framework based on Reversible Hierarchical Markov Chain, termed Cognitive Loop of Thought (CLoT), and a backward reasoning dataset CLoT-Instruct. In CLoT, problems are decomposed into sub-problems with hierarchical dependencies. Inspired by human cognitive processes, we introduce a backward verification mechanism at each hierarchical layer. Furthermore, we implement a pruning strategy: once higher-level sub-problems are verified, redundant lower-level sub-problems are pruned to maximize efficiency. This approach effectively mitigates error propagation and enhances reasoning robustness. Experiments on four mathematical benchmarks demonstrate the effectiveness of our method. Notably, on the AddSub dataset using GPT-4o-mini, CLoT achieves 99.0% accuracy, outperforming traditional CoT and CoT-SC by 4.1% and 2.9%, respectively.
>
---
#### [replaced 037] Many-Shot CoT-ICL: Making In-Context Learning Truly Learn
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究多示例链式思维上下文学习（CoT-ICL）在推理任务中的表现，旨在提升其学习效果。通过分析与优化示例选择，改进模型的推理能力。**

- **链接: [https://arxiv.org/pdf/2605.13511](https://arxiv.org/pdf/2605.13511)**

> **作者:** Tsz Ting Chung; Lemao Liu; Mo Yu; Dit-Yan Yeung
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** While many-shot ICL achieves remarkable performance, prior studies of its scaling behavior have mainly focused on non-reasoning tasks. In this work, we study many-shot ICL on reasoning tasks, with a particular focus on many-shot chain-of-thought in-context learning (CoT-ICL). Across non-reasoning and reasoning tasks and across non-reasoning and reasoning-oriented LLMs, we identify several distinctive properties of many-shot CoT-ICL. We further interpret these findings by viewing many-shot CoT-ICL as in-context test-time learning rather than scaled pattern matching, and suggest two principles: (i) demonstrations should be easy for the target model to understand, and (ii) they should be ordered to support a smooth conceptual progression. Guided by the principle, we propose Curvilinear Demonstration Selection (CDS), a simple ordering method that yields up to a 5.42 percentage-point gain on a math task with 64 demonstrations. Overall, our results reframe the long context window from a retrieval buffer into a structured curriculum for in-context test-time learning.
>
---
#### [replaced 038] From Rubrics to Reliable Scores: Evidence-Grounded Text Evaluation with LLM Judges
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于文本评价任务，解决LLM在遵循人工评分标准时的可靠性问题。提出Rulers框架，通过结构化检查、证据支持和校准提升评分一致性与可解释性。**

- **链接: [https://arxiv.org/pdf/2601.08654](https://arxiv.org/pdf/2601.08654)**

> **作者:** Yihan Hong; Huaiyuan Yao; Bolin Shen; Wanpeng Xu; Hua Wei; Yushun Dong
>
> **摘要:** Rubric-based text evaluation increasingly uses large language models (LLMs) as scalable judges, but aligning frozen black-box models with human scoring standards remains challenging. We formulate this challenge as a criteria-transfer problem: the goal is not merely to prompt an LLM to assign a score, but to transfer human rubric intent into a stable, auditable, and human-aligned scoring protocol. We identify three recurring failure modes in LLM-based rubric scoring: rubric execution drift, unverifiable score attribution, and human-scale misalignment. To address these failure modes, we introduce Rulers, a three-stage inference-time framework for reliable, evidence-grounded rubric-based text evaluation. Rulers first converts a human rubric into a locked task-level specification, then executes the specification with structured checklist decisions, typed evidence grounding, and extractive quote verification when applicable, and finally applies post-hoc calibration to align model-derived signals with human score boundaries. Across four rubric-governed benchmarks covering essay scoring, summarization assessment, EFL writing evaluation, and structured-input text generation, Rulers achieves stronger human-score agreement in most evaluated settings across multiple frozen backbone models. Further analyses show that Rulers better matches empirical human score distributions, improves stability under semantically equivalent rubric perturbations, and benefits from each of its three components. These results suggest that reliable LLM judging requires fixed criteria, traceable evidence, and calibrated score interpretation rather than prompt phrasing alone. Our code is available at this https URL.
>
---
#### [replaced 039] Eureka: Intelligent Feature Engineering for Enterprise AI Cloud Resource Demand Prediction
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出Eureka框架，解决企业AI云资源需求预测中的特征工程问题。通过LLM驱动的代码生成，提升特征质量与跨领域迁移能力。**

- **链接: [https://arxiv.org/pdf/2605.25297](https://arxiv.org/pdf/2605.25297)**

> **作者:** Hangxuan Li; Renjun Jia; Xuezhang Wu; Yunjie Qian; Zeqi Zheng; Xianling Zhang
>
> **备注:** accepted at NeurIPS 2025 Workshop, DASFAA 2026 (International Conference on Database Systems for Advanced Applications)
>
> **摘要:** Effective features are crucial for predictive model performance, but creating them often requires domain expertise, limiting scalability across applications. We define feature engineering as an agentic code generation problem: features are not static data transformations, but executable programs that can be generated, evaluated, and iteratively improved. We present Eureka, an LLM-driven framework with three stages. (1) An Expert Agent, fine-tuned via SFT on domain knowledge, produces structured feature design plans in JSON format. (2) An LLM Feature Factory translates each plan into executable Python code through chain-of-thought reasoning, turning feature hypotheses into runnable programs. (3) A Self-Evolving Alignment Engine uses Reinforcement Learning (GRPO) with dual-channel reward (metric-based utility + semantic alignment) to enhance code quality. By expressing features as programs, the learned generation patterns can transfer across domains. Evaluated on 7 public benchmarks in healthcare, finance, and social domains, Eureka consistently outperforms both traditional AutoFE and LLM-based baselines. We further demonstrate Eureka's effectiveness on cloud GPU resource demand prediction at Alibaba Cloud, where Eureka improves demand fulfillment rate by 16% and lowers computing resource migration rates by 33%.
>
---
#### [replaced 040] Guardrails Beat Guidance: A Large-Scale Study of Rules, Skills, and Persistent Configuration for Coding Agents
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究编码代理的规则配置，探讨随机规则与专家规则的效果。通过大规模实验发现规则极性影响性能，提出应限制禁止行为而非规定动作。属于AI代理配置任务，解决规则有效性问题。**

- **链接: [https://arxiv.org/pdf/2604.11088](https://arxiv.org/pdf/2604.11088)**

> **作者:** Xing Zhang; Guanghui Wang; Yanwei Cui; Wei Qiu; Ziyuan Li; Bing Zhu; Peiyang He
>
> **摘要:** Random rules improve a coding agent's task performance as much as expert-curated ones (both $+13.8$pp on a discriminative subset of SWE-bench Verified), and in our data every individually beneficial rule is a negative constraint ("do not refactor unrelated code"), while every individually harmful one is a positive directive ("follow code style"). We arrive at these findings through the first large-scale controlled study of agent rule files (\texttt{this http URL}, \texttt{.cursorrules}, and the broader family of agent skills, plugin manifests, and persona definitions): we scrape 679 rule files (25{,}532 rules) from GitHub and conduct over 5{,}000 agent runs of Claude Code with Claude Opus 4.6 on SWE-bench Verified. Three patterns emerge. (i) Rule polarity cleanly separates beneficial from harmful rules; we read this through the lens of potential-based reward shaping (PBRS). (ii) Performance gains are largely content-independent: random, shuffled, mismatched-domain, and unconverted-format rule files all match curated rules, pointing to a context priming mechanism. (iii) Individual rules often appear harmful in isolation yet do not visibly accumulate damage in ensemble: pass rates remain stable across rule counts from 0 to 50. These findings expose a hidden reliability risk in the rapidly growing ecosystem of community-authored rules and skills, and they yield a clear principle for safer agent configuration: constrain what agents must not do, rather than prescribing what they should.
>
---
#### [replaced 041] Mixing Mechanisms: How Language Models Retrieve Bound Entities In-Context
- **分类: cs.CL**

- **简介: 该论文研究语言模型在上下文中绑定和检索实体的机制，解决如何准确检索复杂场景下的实体问题。通过实验分析位置、词汇和反射机制的混合使用，构建了高效的因果模型。**

- **链接: [https://arxiv.org/pdf/2510.06182](https://arxiv.org/pdf/2510.06182)**

> **作者:** Yoav Gur-Arieh; Mor Geva; Atticus Geiger
>
> **备注:** Accepted to ICLR 2026 Main Conference
>
> **摘要:** A key component of in-context reasoning is the ability of language models (LMs) to bind entities for later retrieval. For example, an LM might represent "Ann loves pie" by binding "Ann" to "pie", allowing it to later retrieve "Ann" when asked "Who loves pie?" Prior research on short lists of bound entities found strong evidence that LMs implement such retrieval via a positional mechanism, where "Ann" is retrieved based on its position in context. In this work, we find that this mechanism generalizes poorly to more complex settings; as the number of bound entities in context increases, the positional mechanism becomes noisy and unreliable in middle positions. To compensate for this, we find that LMs supplement the positional mechanism with a lexical mechanism (retrieving "Ann" using its bound counterpart "pie") and a reflexive mechanism (retrieving "Ann" through a direct pointer). Through extensive experiments on nine models and ten binding tasks, we uncover a consistent pattern in how LMs mix these mechanisms to drive model behavior. We leverage these insights to develop a causal model combining all three mechanisms that estimates next token distributions with 95% agreement. Finally, we show that our model generalizes to substantially longer inputs of open-ended text interleaved with entity groups, further demonstrating the robustness of our findings in more natural settings. Overall, our study establishes a more complete picture of how LMs bind and retrieve entities in-context.
>
---
#### [replaced 042] Interactive In-Meeting Speaker Correction with Human Feedback
- **分类: cs.CL**

- **简介: 该论文属于语音处理任务，旨在解决会议中说话人识别错误问题。通过引入用户反馈和大模型辅助，提升说话人归因准确性。**

- **链接: [https://arxiv.org/pdf/2509.18377](https://arxiv.org/pdf/2509.18377)**

> **作者:** Xinlu He; Yiwen Guan; Badrivishal Paurana; Pitipat Kongsomjit; Zilin Dai; Jacob Whitehill
>
> **摘要:** Most automatic speech processing systems operate in ``open loop'' mode without user feedback about who said what, yet human-in-the-loop workflows can potentially enable higher accuracy. We propose an LLM-assisted in-meeting speaker correction system that lets users fix speaker attribution errors through brief corrective feedback. After performing streaming ASR and diarization, the system presents concise LLM-generated summaries to help users identify important speaker errors, and it incorporates user feedback by updating the speaker-attributed transcript and adding online speaker enrollments. To make this workflow effective despite errors in speech processing, LLM analysis, and user feedback, we developed several mechanisms to identify the intended correction more precisely. Further, we built an LLM-driven user feedback simulation to evaluate the workflow reprodubilty and at scale. Applied to the AMI headset test set, our system substantially reduces the DER from a streaming baseline (Google ASR + ECAPA) by 31.99% and speaker substitution error by 52.68%.
>
---
#### [replaced 043] Calibration Is Not Enough: Evaluating Confidence Estimation Under Language Variations
- **分类: cs.CL**

- **简介: 该论文属于信心估计任务，旨在解决现有评估方法忽视语言变化的问题。提出新框架，评估模型在不同语言表达下的信心一致性与敏感性。**

- **链接: [https://arxiv.org/pdf/2601.08064](https://arxiv.org/pdf/2601.08064)**

> **作者:** Yuxi Xia; Dennis Ulmer; Terra Blevins; Yihong Liu; Hinrich Schütze; Benjamin Roth
>
> **摘要:** Confidence estimation (CE) indicates how reliable the answers of large language models are and impacts user trust and decision-making. Existing evaluations mainly concern the alignment between confidence and correctness, but ignore the variability of language: confidence estimates should remain consistent under semantically equivalent prompts or answer variations, while changing when answer meaning differs, as this may indicate a change in correctness. Therefore, we introduce a novel evaluation framework based on three complementary properties: \textbf{robustness} to prompt perturbations, \textbf{stability} across semantically equivalent answers, and \textbf{sensitivity} to semantically different answers. We show that these metrics are largely independent from existing CE metrics, and that common CE methods often fail on them: while most methods achieve high robustness and stability, they struggle to distinguish semantically different answers, potentially because they do not effectively leverage generation-side information. Overall, our framework exposes overlooked limitations of current CE evaluations and provides guidance for selecting confidence estimators for real-world applications.
>
---
#### [replaced 044] Slide Deck Q&A Quality Assurance App: A Multi-Stage Pipeline for Pedagogical Question Generation
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于教育技术任务，旨在解决从幻灯片生成高质量教学问题的问题。提出一个四阶段系统，整合文本和图像信息，生成结构化教学问题。**

- **链接: [https://arxiv.org/pdf/2605.26428](https://arxiv.org/pdf/2605.26428)**

> **作者:** Jim Salsman
>
> **备注:** 15 pages, 3 research questions, 1 figure, 1 table, 6 references, 2 appendices
>
> **摘要:** Generating high-quality, pedagogically useful questions from lecture slide decks is difficult because important instructional content is distributed across both text and visual elements, and because useful questions must be scaffolded across the flow of a presentation rather than generated slide by slide in isolation. This paper describes Slide Deck Q\&A Quality Assurance (slidesqaqa), a Flask-based software system that extracts text and rendered images from PDF slides and processes them through a four-stage large language model pipeline comprising window planning, deck synthesis, slide annotation, and reconciliation. The system reasons jointly about slide modality and pedagogical role, allocates bounded question budgets, and revises draft annotations at the deck level to reduce redundancy and improve coverage. The final output is a structured JSON annotation containing deck-level goals, section structure, slide-level summaries, question sets, and evaluation scores. Initial experiments on two technical lecture decks indicate that the pipeline can filter non-instructional slides and produce high-fidelity, pedagogically coherent questions for visually complex content. The working system is at this https URL The software repository is at this https URL
>
---
#### [replaced 045] Maximizing Mutual Information Between Prompt and Response Improves LLM Performance With No Additional Data
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理领域，解决LLM性能提升问题。通过构建对比数据对，最大化提示与响应间的互信息，无需额外数据即可提升模型表现。**

- **链接: [https://arxiv.org/pdf/2603.19294](https://arxiv.org/pdf/2603.19294)**

> **作者:** Hyunji Nam; Haoran Li; Natasha Jaques
>
> **备注:** International Conference on Machine Learning 2026
>
> **摘要:** While post-training has successfully improved large language models (LLMs) across a variety of domains, these gains heavily rely on human-labeled data or external verifiers. Existing data has already been exploited, and new data is expensive to collect. Moreover, true intelligence goes far beyond verifiable tasks. Therefore, we need self-improvement frameworks that are less dependent on external signals and more broadly applicable to both verifiable and non-verifiable domains. We propose **Mutual Information Preference Optimization (MIPO)**, a contrastive data augmentation method that constructs preference pairs by generating a positive response conditioning on the correct prompt, and a negative response by conditioning on a random, unrelated prompt. We show that using Direct Preference Optimization to learn from this paired data maximizes pointwise mutual information *under the base LLM* between prompts and model responses. Experiments with with 1-7B parameter Llama and Qwen instruct models show that MIPO achieves 3-16% gains (and 51% increase for Qwen2.5-1B-Instruct) on personalization compared to prompting baselines. Surprisingly, MIPO can also be useful in verifiable domains, such as math and multiple-choice question answering, yielding 1-20% gains *without any additional data or external supervision*. These results suggest a promising direction for self-improvement using intrinsic signals derived from contrastive data pairs.
>
---
#### [replaced 046] SafeReview: Defending LLM-based Review Systems Against Adversarial Hidden Prompts
- **分类: cs.CL; cs.CR**

- **简介: 该论文属于安全任务，旨在解决LLM在学术评审中受对抗隐藏指令攻击的问题。提出SafeReview框架，通过生成器与防御者协同训练提升系统鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.26506](https://arxiv.org/pdf/2604.26506)**

> **作者:** Yuan Xin; Yixuan Weng; Minjun Zhu; Ying Ling; Chengwei Qin; Michael Backes; Yue Zhang; Linyi Yang
>
> **备注:** 17 pages, 5 figures, 8 tables
>
> **摘要:** As Large Language Models (LLMs) are increasingly integrated into academic peer review, their vulnerability to adversarial hidden prompts, i.e., adversarial instructions embedded in submissions to manipulate outcomes, poses a critical threat to scholarly integrity. We propose SafeReview, a co-evolutionary adversarial training framework for defending LLM-based peer review systems against such attacks. SafeReview jointly trains a Generator model to create sophisticated attack prompts and a Defender model to preserve review integrity under adversarial manipulation. The Generator is optimized to produce increasingly effective prompt injections, while the Defender is strengthened through preference-based training to maintain consistent reviews between clean and attacked submissions. Experimental results show that SafeReview improves robustness against adaptive prompt injection attacks, better preserves paper ranking under attack, and generalizes across attacker architectures compared with static defenses. These results demonstrate the potential of co-evolutionary training as a foundation for securing LLM-assisted peer review.
>
---
#### [replaced 047] GroundAct: Can LLM Agents Ground Actions in Environmental States?
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI代理任务，旨在解决LLM在环境状态影响行动可行性时表现下降的问题。通过构建基准数据集，分析模型在不同情境下的行动推理能力。**

- **链接: [https://arxiv.org/pdf/2508.05614](https://arxiv.org/pdf/2508.05614)**

> **作者:** Zixuan Wang; Dingming Li; Hongxing Li; Yanrui Miao; Shuo Chen; Yuchen Yan; Wenqi Zhang; Yongliang Shen; Weiming Lu; Jun Xiao; Yueting Zhuang
>
> **备注:** Project Page: this https URL Code: this https URL
>
> **摘要:** LLM agents achieve 85-96% success on tasks where instructions fully specify the action, but drop to 29-53% when action feasibility depends on environmental state that the instruction does not mention. We argue that this gap reflects a missing capability: action grounding, the ability to infer from structured environmental state whether an action is feasible, what prerequisites it lacks, and whether it exceeds individual capacity. We introduce GroundAct, a benchmark of 1,500 scenarios and 16,592 task instances in text-based interactive environments spanning 11 domains, with tasks organized into seven categories along a cognitive complexity hierarchy. Evaluating 15 LLMs (3B-671B), we find three diagnostic patterns: (i) attribute reasoning is weakly correlated with tool and coordination reasoning, producing distinct model profiles; (ii) complete environment graphs yield up to +27.6/-22.9% on tool use vs. implicit collaboration, separating search-bound from constraint-filtering bottlenecks; and (iii) supervised fine-tuning lifts Qwen2.5-3B from 0.6% to 76.3% on direct command but only 1.5% to 5.5% on implicit collaboration. These results establish action grounding as a multi-dimensional challenge irreducible to scaling.
>
---
#### [replaced 048] Mechanism Shift During Post-training from Autoregressive to Masked Diffusion Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究从自回归模型到掩码扩散模型的后训练机制变化，旨在解决模型生成方式转变是否带来本质计算机制更新的问题。通过对比分析，揭示了结构与语义上的重组规律。**

- **链接: [https://arxiv.org/pdf/2601.14758](https://arxiv.org/pdf/2601.14758)**

> **作者:** Injin Kong; Hyoungjoon Lee; Yohan Jo
>
> **摘要:** Post-training pretrained autoregressive models (ARMs) into masked diffusion models (MDMs) has emerged as a cost-effective way to overcome the limitations of sequential generation. Yet it remains unclear whether post-trained MDMs acquire genuinely new computational mechanisms or merely re-express autoregressive computation in a non-autoregressive form. Through a comparative circuit analysis of ARMs and their MDM counterparts post-trained from the same backbones, we uncover two complementary axes of reorganization. Structurally, the shift is task-dependent: MDMs preserve autoregressive circuitry on locally causal tasks but abandon inherited pathways and front-load computation into early layers on global tasks. Semantically, the shift is consistent across regimes: sharp, localized specialization in ARMs gives way to distributed integration in MDMs. Together, these findings show that diffusion post-training is not a surface-level change in the generation procedure but a reorganization of internal computation whose depth depends on the task.
>
---
#### [replaced 049] Catalyst-Agent: Autonomous heterogeneous catalyst screening with an LLM Agent
- **分类: cs.CL**

- **简介: 该论文属于催化剂筛选任务，旨在解决传统方法耗时费力的问题。通过AI代理Catalyst-Agent，结合模型与工具实现自主筛选，提升效率与成功率。**

- **链接: [https://arxiv.org/pdf/2603.01311](https://arxiv.org/pdf/2603.01311)**

> **作者:** Achuth Chandrasekhar; Janghoon Ock; Amir Barati Farimani
>
> **摘要:** The discovery of novel catalysts tailored for particular applications is a major challenge for the twenty-first century. Traditional methods for this include time-consuming and expensive experimental trial-and-error approaches in labs based on chemical theory or heavily computational first-principles approaches based on density functional theory. Recent studies show that deep learning models like graph neural networks (GNNs) can significantly speed up the screening of catalyst materials by many orders of magnitude, with very high accuracy and fidelity. In this work, we introduce Catalyst-Agent, a Model Context Protocol (MCP) server-based, LLM-powered AI agent. It can explore vast material databases using the OPTIMADE API, make structural modifications, calculate adsorption energies using Meta FAIRchem's UMA (GNN) model via FAIRchem's AdsorbML workflow and slab construction, and make useful material suggestions to the researcher in a closed-loop manner, including structural modifications to refine near-miss candidates. It is tested on three pivotal reactions: the oxygen reduction reaction (ORR), the nitrogen reduction reaction (NRR), and the CO2 reduction reaction (CO2RR). Catalyst-Agent achieves a success rate of 33-41% among all the materials it chooses and evaluates, and manages to converge in 1-4 trials per successful material on average. This work demonstrates the potential of AI agents to exercise their planning capabilities and tool use for autonomous catalyst screening workflows.
>
---
#### [replaced 050] SSDAU: Structured Semantic Data Augmentation for Joint Entity and Relation Extraction
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于联合实体和关系抽取任务，旨在解决数据增强中语义结构破坏的问题。提出SSDAU方法，保留三元组语义结构，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.23440](https://arxiv.org/pdf/2605.23440)**

> **作者:** Jiawei He; Mengyu Shi; Jiawei Liu; Dong Sun; Chunrong Fang; Xikai Yang; Zhijie Wang; Lei Ma; Zhenyu Chen
>
> **备注:** 10 pages, 4 figure
>
> **摘要:** Joint Entity and Relation Extraction (JERE) is highly sensitive to training data quality, making data augmentation a natural way to improve generalization. However, existing augmentation methods often weaken entity relevance and disrupt semantic structure, limiting their effectiveness for JERE. In this paper, we propose \textbf{Structured Semantic Data Augmentation (SSDAU)}, a method designed to preserve triple-aware semantic structure during augmentation. SSDAU segments text by entity labels, captures semantic features through context-aware encoding, and restructures entity semantics to generate augmented data. To distinguish semantically similar entities, SSDAU combines contextualized embeddings with traditional similarity scores. To reduce topic inconsistency, we apply BERTopic-based filtering to remove irrelevant augmentations. We evaluate SSDAU on datasets with different annotation types and compare its performance on five representative JERE models against seven popular augmentation baselines. Experiments show that SSDAU generates semantically consistent data, is more robust to ambiguity than non-LLM methods (8.95\% vs. 23.58\% average relative F1 decrease), and significantly outperforms strong alternatives in most settings.
>
---
#### [replaced 051] Procedural Pretraining: Warming Up Language Models with Abstract Data
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语言模型预训练任务，旨在提升模型性能与训练效率。通过引入结构化程序数据进行预训练，改善模型的结构理解能力，减少对自然语言数据的依赖。**

- **链接: [https://arxiv.org/pdf/2601.21725](https://arxiv.org/pdf/2601.21725)**

> **作者:** Liangze Jiang; Zachary Shinnick; Anton van den Hengel; Hemanth Saratchandran; Damien Teney
>
> **备注:** ICML 2026. Project page: this https URL
>
> **摘要:** Pretraining language models directly on web-scale corpora is the de facto paradigm. We study an alternative where the model is initially exposed to abstract structured data to ease the subsequent acquisition of rich semantic knowledge, much like humans learning simple logic and mathematics before higher reasoning. We focus on procedural data, generated by formal languages and other simple algorithms, as such abstract data. We first diagnose the algorithmic skills that different forms of procedural data can improve, often significantly. For example, the accuracy of context recall (Needle-in-a-haystack) jumps from 10 to 98% when a model is pretrained on Dyck sequences (balanced brackets). Second, we study how these gains are reflected in pretraining larger models (up to 1.3B). We find that front-loading as little as 0.1 to 0.3% procedural data significantly outperforms standard pretraining on natural language, code, and informal mathematics (C4, CodeParrot, and DeepMind-Math datasets). Notably, this also enables the models to reach the same loss value with only 55/67/86% of the original data and thus a comparable reduction in FLOPs. Third, we explore the mechanisms behind the benefits and find that procedural pretraining instills non-trivial structure in both attention and MLP layers. The former is particularly important for structured domains (e.g. code), and the latter for language. Finally, we lay a path for combining multiple forms of procedural data. Our results show that procedural pretraining is a simple, lightweight means of improving performance and accelerating language model pretraining, ultimately suggesting the promise of disentangling knowledge acquisition from reasoning in LLMs.
>
---
#### [replaced 052] PersonaAgent: Bridging Memory and Action for Personalized LLM Agents
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出PersonaAgent，解决LLM代理个性化不足的问题。通过整合记忆与行动模块，实现用户偏好动态适配，提升个性化体验。**

- **链接: [https://arxiv.org/pdf/2506.06254](https://arxiv.org/pdf/2506.06254)**

> **作者:** Weizhi Zhang; Xinyang Zhang; Chenwei Zhang; Liangwei Yang; Jingbo Shang; Zhepei Wei; Henry Peng Zou; Zijie Huang; Zhengyang Wang; Yifan Gao; Xiaoman Pan; Lian Xiong; Jingguo Liu; Philip S. Yu; Xian Li
>
> **备注:** Accepted in ACL 2026
>
> **摘要:** Large Language Model (LLM) empowered agents have recently emerged as advanced paradigms that exhibit impressive capabilities in a wide range of domains and tasks. Despite their potential, current LLM agents often adopt a one-size-fits-all approach, lacking the flexibility to respond to users' varying needs and preferences. This limitation motivates us to develop PersonaAgent, the first personalized LLM agent framework designed to address versatile personalization tasks. Specifically, PersonaAgent integrates two complementary components - a personalized memory module that includes episodic and semantic memory mechanisms; a personalized action module that enables the agent to perform tool actions tailored to the user. At the core, the persona (defined as unique system prompt for each user) functions as an intermediary: it leverages insights from personalized memory to control agent actions, while the outcomes of these actions in turn refine the memory. Based on the framework, we propose a test-time user-preference alignment strategy that simulate the latest n interactions to optimize the persona prompt, ensuring real-time user preference alignment through textual loss feedback between simulated and ground-truth responses. Experimental evaluations demonstrate that PersonaAgent significantly outperforms other baseline methods by not only personalizing the action space effectively but also scaling during test-time real-world applications. These results underscore the feasibility and potential of our approach in delivering tailored, dynamic user experiences.
>
---
#### [replaced 053] Reasoning While Asking: Transforming Reasoning Large Language Models from Passive Solvers to Proactive Inquirers
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于人工智能推理任务，旨在解决LLM在信息缺失时的被动推理问题。提出PIR框架，通过与用户互动实现主动提问，提升推理准确性与效率。**

- **链接: [https://arxiv.org/pdf/2601.22139](https://arxiv.org/pdf/2601.22139)**

> **作者:** Xin Chen; Feng Jiang; Yiqian Zhang; Hardy Chen; Shuo Yan; Wenya Xie; Min Yang; Shujian Huang
>
> **备注:** ACL Main Conference
>
> **摘要:** Reasoning-oriented Large Language Models (LLMs) have achieved remarkable progress with Chain-of-Thought (CoT) prompting, yet they remain fundamentally limited by a \emph{blind self-thinking} paradigm: performing extensive internal reasoning even when critical information is missing or ambiguous. We propose Proactive Interactive Reasoning (PIR), a new reasoning paradigm that transforms LLMs from passive solvers into proactive inquirers that interleave reasoning with clarification. Unlike existing search- or tool-based frameworks that primarily address knowledge uncertainty by querying external environments, PIR targets premise- and intent-level uncertainty through direct interaction with the user. PIR is implemented via two core components: (1) an uncertainty-aware supervised fine-tuning procedure that equips models with interactive reasoning capability, and (2) a user-simulator-based policy optimization framework driven by a composite reward that aligns model behavior with user intent. Extensive experiments on mathematical reasoning, code generation, and document editing demonstrate that PIR consistently outperforms strong baselines, achieving up to 32.70\% higher accuracy, 22.90\% higher pass rate, and 41.36 BLEU improvement, while reducing nearly half of the reasoning computation and unnecessary interaction turns. Further reliability evaluations on factual knowledge, question answering, and missing-premise scenarios confirm the strong generalization and robustness of PIR. Model and code are publicly available at: \href{this https URL}
>
---
#### [replaced 054] AuthorMix: Modular Authorship Style Transfer via Layer-wise Adapter Mixing
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于作者风格迁移任务，旨在在保持原意的前提下转换文本风格。提出AuthorMix框架，通过模块化适配器实现高效、灵活的风格迁移。**

- **链接: [https://arxiv.org/pdf/2603.23069](https://arxiv.org/pdf/2603.23069)**

> **作者:** Sarubi Thillainathan; Ji-Ung Lee; Michael Sullivan; Alexander Koller
>
> **备注:** Under review
>
> **摘要:** The task of authorship style transfer involves rewriting text in the style of a target author while preserving the meaning of the original text. Existing style transfer methods train a single model on large corpora to model all target styles at once: this high-cost approach offers limited flexibility for target-specific adaptation, and often sacrifices meaning preservation for style transfer. In this paper, we propose AuthorMix: a lightweight, modular, and interpretable style transfer framework. We train individual, style-specific LoRA adapters on a small set of high-resource authors, allowing the rapid training of specialized adaptation models for each new target via learned, layer-wise adapter mixing, using only a handful of target-style training examples. AuthorMix outperforms existing, SoTA style-transfer baselines-as well as GPT-5.1-for low-resource targets, achieving the highest overall score and substantially improving meaning preservation in both automatic and human evaluations.
>
---
#### [replaced 055] RewardFlow: Topology-Aware Reward Propagation on State Graphs for Agentic RL with Large Language Models
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出RewardFlow，解决LLM代理强化学习中的稀疏奖励问题。通过构建状态图进行拓扑感知奖励传播，生成密集奖励，提升任务成功率和效率。**

- **链接: [https://arxiv.org/pdf/2603.18859](https://arxiv.org/pdf/2603.18859)**

> **作者:** Xiao Feng; Bo Han; Zhanke Zhou; Jiaqi Fan; Jiangchao Yao; Ka Ho Li; Dahai Yu; Michael Kwok-Po Ng
>
> **摘要:** Reinforcement learning (RL) shows promise for enhancing LLM agentic reasoning, yet sparse terminal rewards hinder fine-grained optimization. Process reward modeling offers an alternative but incurs high computational costs, reward hacking risks, and annotation bottlenecks. We introduce RewardFlow, a lightweight method for estimating state-level rewards in agentic reasoning. By constructing state graphs that capture the intrinsic topological structure of trajectories, RewardFlow performs topology-aware propagation to estimate each state's contribution to success, yielding principled, annotation-free dense rewards. Used for RL optimization, RewardFlow substantially outperforms prior baselines across four agentic benchmarks: +6.2% average success rate on text-based tasks, +29.7% on visual reasoning over the strongest baseline across three model scales, and +10% accuracy on DeepResearch, with superior robustness and training efficiency. The implementation of RewardFlow is publicly available at this https URL.
>
---
#### [replaced 056] SIA: Self Improving AI with Harness & Weight Updates
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出SIA，一种可自我改进的AI系统，解决人工干预AI优化的瓶颈问题。通过同时更新模型权重和框架，提升任务性能，在法律分类、GPU优化和基因数据处理中取得更好效果。**

- **链接: [https://arxiv.org/pdf/2605.27276](https://arxiv.org/pdf/2605.27276)**

> **作者:** Prannay Hebbar; Yogendra Manawat; Samuel Verboomen; Alesia Ivanova; Selvam Palanimalai; Kunal Bhatia; Vignesh Baskaran
>
> **摘要:** Humans are the bottleneck in building and improving AI. Both the models and the agents that wrap them are written, tuned, and corrected by people. The long-horizon goal of an AI that can figure out how to improve itself remains open. Two largely disjoint research lines attack this bottleneck. The harness-update school has a meta-agent rewrite the scaffold of a task-specific agent (its tools, prompts, retry logic, and search procedure) while the model weights are held fixed. The test-time training school uses hand-written RL pipelines to update the model's own weights on task feedback while the harness is held fixed. These two silos operate in isolation. We propose SIA, a self-improving loop in which a language-model agent (the Feedback-Agent) updates both the harness and the weights of a task-specific agent. We evaluate across three contrasting domains: Chinese legal charge classification, low-level GPU kernel optimisation, and single-cell RNA denoising. Combining both levers outperforms scaffold iteration alone on all three benchmarks. SIA-W+H achieves 25.1% over prior SOTA on LawBench, 12.4% faster GPU kernels than prior SOTA (1,017 vs 1,161 {\mu}s), and 20.4% over prior SOTA on denoising. Harness updates make the model agentic, shaping how it searches and acts, while weight updates build the domain intuition that no prompt or scaffold can instil.
>
---
#### [replaced 057] Human-Guided Harm Recovery for Computer Use Agents
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于人工智能安全任务，解决有害行为后的恢复问题。通过用户研究和奖励模型，提升代理从有害状态恢复的能力。**

- **链接: [https://arxiv.org/pdf/2604.18847](https://arxiv.org/pdf/2604.18847)**

> **作者:** Christy Li; Sky CH-Wang; Andi Peng; Andreea Bobu
>
> **摘要:** As LM agents gain the ability to execute actions on real computer systems, we need ways to not only prevent harmful actions at scale but also effectively remediate harm when prevention fails. We formalize a solution to this neglected challenge in post-execution safeguards as harm recovery: the problem of optimally steering an agent from a harmful state back to a safe one in alignment with human preferences. We ground preference-aligned recovery through a formative user study that identifies valued recovery dimensions and produces a natural language rubric. Our dataset of 1,130 pairwise judgments reveals context-dependent shifts in attribute importance, such as preferences for pragmatic, targeted strategies over comprehensive long-term approaches. We operationalize these learned insights in a reward model, re-ranking multiple candidate recovery plans generated by an agent scaffold at test time. To evaluate recovery capabilities systematically, we introduce BackBench, a benchmark of 50 computer-use tasks that test an agent's ability to recover from harmful states. Human evaluation shows our reward model scaffold yields higher-quality recovery trajectories than base agents and rubric-based scaffolds. Together, these contributions lay the foundation for a new class of agent safety methods -- ones that confront harm not only by preventing it, but by navigating its aftermath with alignment and intent.
>
---
#### [replaced 058] Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决大模型多词预测问题。提出ESP方法，无需训练即可高效预测多个未来词，提升预测长度和吞吐量。**

- **链接: [https://arxiv.org/pdf/2603.17942](https://arxiv.org/pdf/2603.17942)**

> **作者:** Raghavv Goel; Mukul Gagrani; Mingu Lee; Chris Lott
>
> **备注:** v2: Accepted at ICML 2026. Updated experiments replaced tok/s with speedup ratio over AR baseline; improved exposition in Section 3.1 (mask token initialization) and Section 4 (ablations)
>
> **摘要:** Large Language Models (LLMs) possess latent multi-token prediction (MTP) abilities despite being trained only for next-token generation. We introduce ESP (Embedding-Space Probing), a simple and training-free MTP method that probes an LLM using on-the-fly mask tokens drawn from its embedding space, enabling parallel future-token prediction without modifying weights or relying on draft models. ESP constructs a speculative token tree by sampling Top-K candidates from mask-token logits and applies a lightweight pruning rule to retain high-probability continuations. During generation, predictions are verified in parallel, yielding lossless decoding while significantly reducing model calls and increasing token throughput. ESP consistently outperforms existing training-free baselines, improving acceptance length by 7-11% over LADE on LLaMA3 and 7-8% on Qwen3, and increasing throughput by up to 15-19% over the strongest baseline. Finally, we provide theoretical insight and empirical evidence showing that decoder layers naturally align mask-token representations with next-token states, enabling accurate multi-step prediction without retraining or auxiliary models.
>
---
#### [replaced 059] Do not be greedy, Think Twice: Sampling and Selection for Document-level Information Extraction
- **分类: cs.CL**

- **简介: 该论文属于文档级信息抽取任务，旨在解决输出多样性与准确性问题。通过采样与选择框架ThinkTwice，生成并优选最佳输出模板。**

- **链接: [https://arxiv.org/pdf/2601.18395](https://arxiv.org/pdf/2601.18395)**

> **作者:** Mikel Zubillaga; Oscar Sainz; Oier Lopez de Lacalle; Eneko Agirre
>
> **备注:** Submitted to EMNLP 2026
>
> **摘要:** Document-level Information Extraction (DocIE) aims to produce an output template with the entities, relations, and events of interest occurring in the given document. Standard practices include prompting decoder-only LLMs using greedy decoding to avoid output variability. Rather than treating this variability as a limitation, we show that sampling can produce substantially better solutions than greedy decoding, especially when using reasoning models. We thus propose ThinkTwice, a sampling and selection framework in which the LLM generates multiple candidate templates for a given document, and a selection module chooses the most suitable one. We introduce both an unsupervised method that exploits agreement across generated outputs, and a supervised selection method using reward models trained on labeled DocIE data. To address the scarcity of golden reasoning trajectories for DocIE, we propose a rejection-sampling-based method to generate silver training data that pairs output templates with reasoning traces. Our experiments show the validity of unsupervised and supervised ThinkTwice, consistently outperforming greedy baselines and the supervised state-of-the-art.
>
---
#### [replaced 060] Unleashing Implicit Rewards: Prefix-Value Learning for Distribution-Level Optimization
- **分类: cs.CL**

- **简介: 该论文属于强化学习任务，解决在线RL中奖励模型成本高、训练与推理不一致的问题。提出IPVRM和DistRL，提升推理性能。**

- **链接: [https://arxiv.org/pdf/2604.13197](https://arxiv.org/pdf/2604.13197)**

> **作者:** Shiping Gao; Hongzhan Chen; Xiaojun Quan; Qifan Wang; Lifu Huang
>
> **摘要:** Process reward models (PRMs) provide fine-grained supervision for reasoning, but reliable PRMs often require step annotations or heavy verification pipelines, making them costly to scale and refresh during online RL. Implicit PRMs reduce this cost by training log-likelihood-ratio rewards from trajectory-level outcome labels. However, the log-ratio is constrained only as a sequence-level aggregate during training, while inference decomposes it into token- or step-level scores for partial prefixes. This train-inference mismatch leaves local credits weakly identified, so distribution-wide scoring can amplify misleading advantages. We propose Implicit Prefix-Value Reward Model (IPVRM), which directly learns the probability of eventual correctness for each prefix from outcome labels. Step signals are then obtained as temporal-difference (TD) differences between consecutive prefix values, aligning the training target with inference-time use. IPVRM markedly improves step-verification F1 on ProcessBench. To exploit these prefix values during policy optimization, we further introduce Distribution-Level RL (DistRL), which applies TD advantages to both sampled tokens and high-probability candidate tokens, providing dense counterfactual updates without additional rollouts. Experiments show that DistRL brings limited gains with unreliable implicit rewards, but consistently improves downstream reasoning when paired with IPVRM. The implementation of our method is available at this https URL .
>
---
#### [replaced 061] Position: Text Embeddings Should Capture Implicit Semantics, Not Just Surface Meaning
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于自然语言处理领域，指出当前文本嵌入模型仅关注表层语义，忽视隐含意义。提出应重视隐含语义建模，通过更丰富的数据和评估标准提升模型理解力。**

- **链接: [https://arxiv.org/pdf/2506.08354](https://arxiv.org/pdf/2506.08354)**

> **作者:** Yiqun Sun; Qiang Huang; Anthony K. H. Tung; Jun Yu
>
> **备注:** To appear in ICML 2026
>
> **摘要:** This position paper argues that text embedding research should move beyond surface meaning and embrace implicit semantics as a central modeling objective. Text embeddings are a foundational component of modern NLP, underpinning a wide range of applications and driving sustained research progress. Despite rapid progress, most embedding models remain narrowly focused on surface-level semantics, whereas linguistic theory emphasizes that much of human meaning is implicit, shaped by pragmatics, speaker intent, and sociocultural context. Current models are typically trained on datasets that lack such depth and evaluated using benchmarks that reward surface similarity. As a result, they struggle with tasks that require interpretive reasoning, stance recognition, or socially grounded understanding. Our pilot study makes this limitation explicit, showing that even state-of-the-art embeddings achieve only marginal improvements over simple lexical baselines on tasks probing implicit semantics. We therefore call for a paradigm shift: embedding research should prioritize linguistically grounded and diverse training data, develop benchmarks that probe deeper semantic understanding, and treat implicit meaning as a core modeling objective to better align embeddings with real-world language complexity. The code is available at this http URL.
>
---
#### [replaced 062] Soro: A Lightweight Foundation Model and Chatbot for Tajik
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出Soro，一个专为塔吉克语设计的轻量级对话大模型，解决塔吉克语资源不足及计算限制问题。通过预训练和微调，提升塔吉克语性能，并支持边缘部署。**

- **链接: [https://arxiv.org/pdf/2605.27379](https://arxiv.org/pdf/2605.27379)**

> **作者:** Stanislav Liashkov; Haitz Sáez de Ocáriz Borde; Azizjon Azimi; Khushbakht Shoymardonov; Shuhratjon Khalilbekov; Bonu Boboeva
>
> **摘要:** We present Soro, a family of Tajik-specialized conversational large language models (LLMs) designed for real-world deployment under tight compute and connectivity constraints in Tajikistan. Starting from open-weight Gemma 3 checkpoints, we perform Tajik-only continual pretraining on a curated 1.9-billion-token corpus spanning filtered web text, PDF documents, and curriculum-aligned educational materials, followed by supervised instruction tuning on 40K Tajik teacher-style examples. To enable rigorous evaluation despite the limited coverage of Tajik in standard benchmarks, we introduce a suite of Tajik benchmarks covering general knowledge, linguistic competence, and school- and university entrance-exam domains, and we open-source them on Hugging Face. Across these Tajik benchmarks, Soro substantially outperforms same-size Gemma 3 baselines while retaining strong English performance on standard datasets. We further show that FP8 and INT4 quantization of Soro preserves most Tajik-language gains while reducing memory requirements for edge deployment, supporting an ongoing education-sector pilot and planned scale-out across schools in Tajikistan.
>
---
#### [replaced 063] Understanding the Ability of LLMs to Handle Character-Level Perturbation
- **分类: cs.CL**

- **简介: 该论文研究LLMs在字符级扰动下的鲁棒性，探讨其处理错别字、乱序和隐形字符的能力，揭示其架构优势与潜在风险。**

- **链接: [https://arxiv.org/pdf/2510.14365](https://arxiv.org/pdf/2510.14365)**

> **作者:** Anyuan Zhuo; Xuefei Ning; Ningyuan Li; Jingyi Zhu; Yu Wang; Pinyan Lu
>
> **备注:** Accepted by icml2026
>
> **摘要:** This work investigates the resilience of contemporary large language models (LLMs) against frequent character-level perturbations. We examine three types of character-level perturbations including introducing numerous typos within words, shuffling the characters in each word, and inserting a large number of invisible characters into the text. Surprisingly, even under severe perturbation, such as shuffling nearly all words character-wise to produce text that is almost unreadable to humans, or inserting invisible characters which are several times more than the visible ones as noise, many LLMs still maintain notable performance. We explore the underlying causes of this robustness and find that LLMs exhibit remarkable resilience to chaotic segmentation and fragmented tokenization. Furthermore, we examine the mechanisms by which LLMs remove perturbations to correctly comprehend text, including both implicit and explicit mechanisms for character-level perturbation. We hope that our findings on the low-level robustness of LLMs will unveil their inherent architectural strengths, reveal the potential risks of their misuse, and inform the reliable deployment of LLMs across diverse application scenarios.
>
---
#### [replaced 064] ToolSpec: Accelerating Tool Calling via Schema-Aware and Retrieval-Augmented Speculative Decoding
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大模型工具调用延迟高的问题。通过结构感知和检索增强的推测解码方法，提升工具调用效率。**

- **链接: [https://arxiv.org/pdf/2604.13519](https://arxiv.org/pdf/2604.13519)**

> **作者:** Heming Xia; Yongqi Li; Cunxiao Du; Mingbo Song; Wenjie Li
>
> **摘要:** Tool calling has greatly expanded the practical utility of large language models (LLMs) by enabling them to interact with external applications. As LLM capabilities advance, effective tool use increasingly involves multi-step, multi-turn interactions to solve complex tasks. However, the resulting growth in tool interactions incurs substantial latency, posing a key challenge for real-time LLM serving. Through empirical analysis, we find that tool-calling traces are highly structured, conform to constrained schemas, and often exhibit recurring invocation patterns. Motivated by this, we propose ToolSpec, a schema-aware, retrieval-augmented speculative decoding method for accelerating tool calling. ToolSpec exploits predefined tool schemas to generate accurate drafts, using a finite-state machine to alternate between deterministic schema token filling and speculative generation for variable fields. In addition, ToolSpec retrieves similar historical tool invocations and reuses them as drafts to further improve efficiency. ToolSpec presents a plug-and-play solution that can be seamlessly integrated into existing LLM workflows. Experiments across multiple benchmarks demonstrate that ToolSpec achieves up to a 4.2x speedup, substantially outperforming existing training-free speculative decoding methods.
>
---
#### [replaced 065] Reducing Political Manipulation with Consistency Training
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型中的隐性政治偏见问题。通过提出两种度量标准和一致性训练方法，减少模型在不同政治立场上的不对称响应。**

- **链接: [https://arxiv.org/pdf/2605.22771](https://arxiv.org/pdf/2605.22771)**

> **作者:** Long Phan; Devin Kim; Alexander Pan; Alice Blair; Adam Khoja; Dan Hendrycks
>
> **摘要:** Large language models (LLMs) exhibit systematic political bias across a variety of sensitive contexts. We find that LLMs handle counterpart topics from opposing political sides asymmetrically. We refer to this phenomenon as covert political bias and identify 7 categories of techniques through which it operates. We propose two metrics for covert bias: Sentiment Consistency measures symmetry in rhetoric and framing across paired political prompts; Helpfulness Consistency measures symmetric depth and engagement. To reduce both types of covert bias, we introduce Political Consistency Training (PCT), an RL training method with two complementary paradigms: Sentiment Consistency Training and Helpfulness Consistency Training. We show that PCT preserves overall helpfulness, substantially reduces covert political bias, and generalizes to held-out benchmarks. We release our work at this https URL
>
---
#### [replaced 066] The Vision Wormhole: Latent-Space Communication in Heterogeneous Multi-Agent Systems
- **分类: cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于多智能体系统任务，旨在解决异构系统间通信效率低的问题。通过构建视觉虫洞，实现跨架构的连续潜在状态传输，提升协作推理性能。**

- **链接: [https://arxiv.org/pdf/2602.15382](https://arxiv.org/pdf/2602.15382)**

> **作者:** Xiaoze Liu; Ruowang Zhang; Weichen Yu; Siheng Xiong; Liu He; Feijie Wu; Hoin Jung; Matt Fredrikson; Xiaoqian Wang; Jing Gao
>
> **备注:** Preprint. Work in progress
>
> **摘要:** Multi-Agent Systems (MAS) powered by Large Language Models have unlocked advanced collaborative reasoning, yet they remain bottlenecked by discrete text communication, which imposes runtime overhead and information quantization loss. While latent state transfer offers an alternative, existing approaches either assume homogeneous sender--receiver architectures or rely on pair-specific learned translators, limiting scalability across diverse model families with disjoint manifolds. We reconceptualize the visual interface of Vision-Language Models (VLMs), trained for natural images, as a continuous communication channel between heterogeneous agents, and instantiate this idea as the \textbf{Vision Wormhole}: a Universal Visual Codec maps reasoning traces into a shared continuous reference space and injects them into the receiver's visual pathway, yielding cross-architecture latent state transfer without per-pair translators. The framework adopts a hub-and-spoke topology that reduces alignment complexity from $O(N^2)$ to $O(N)$, and is trained by label-free teacher--student distillation against the text channel, requiring no parallel hidden-state supervision. Extensive experiments across heterogeneous VLM families (Qwen-VL, Gemma, SmolVLM2, LFM2.5-VL) and nine reasoning benchmarks show that the Vision Wormhole reduces end-to-end wall-clock time across most evaluated settings and yields positive macro-average $\Delta$-accuracy.
>
---
#### [replaced 067] MENTOR: Efficient Multimodal-Conditioned Tuning for Autoregressive Vision Generation Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于图像生成任务，旨在解决多模态控制不足、生成可控性差的问题。提出MENTOR框架，通过两阶段训练实现多模态输入与图像的精细对齐，提升生成质量与效率。**

- **链接: [https://arxiv.org/pdf/2507.09574](https://arxiv.org/pdf/2507.09574)**

> **作者:** Haozhe Zhao; Zefan Cai; Shuzheng Si; Liang Chen; Jiuxiang Gu; Wen Xiao; Minjia Zhang; Junjie Hu
>
> **备注:** Findings of ACL 2026
>
> **摘要:** Recent text-to-image models produce high-quality results but still struggle with precise visual control, balancing multimodal inputs, and requiring extensive training for complex multimodal image generation. To address these limitations, we propose MENTOR, a novel autoregressive (AR) framework for efficient Multimodal-conditioned Tuning for Autoregressive multimodal image generation. MENTOR combines an AR image generator with a two-stage training paradigm, enabling fine-grained, token-level alignment between multimodal inputs and image outputs without relying on auxiliary adapters or cross-attention modules. The two-stage training consists of: (1) a multimodal alignment stage that establishes robust pixel- and semantic-level alignment, followed by (2) a multimodal instruction tuning stage that balances the integration of multimodal inputs and enhances generation controllability. Despite modest model size, suboptimal base components, and limited training resources, MENTOR achieves strong performance on the DreamBench++ benchmark, outperforming competitive baselines in concept preservation and prompt following. Additionally, our method delivers superior image reconstruction fidelity, broad task adaptability, and improved training efficiency compared to diffusion-based methods. Dataset, code, and models are available at: this https URL
>
---
#### [replaced 068] A Language-Guided Bayesian Optimization for Efficient LoRA Hyperparameter Search
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于超参数优化任务，解决LoRA微调中计算成本高的问题。通过语言引导的贝叶斯优化，高效搜索最优超参数，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2602.11171](https://arxiv.org/pdf/2602.11171)**

> **作者:** Baek Seong-Eun; Lee Jung-Mok; Kim Sung-Bin; Tae-Hyun Oh
>
> **备注:** Accepted at ICML 2026
>
> **摘要:** Fine-tuning Large Language Models (LLMs) with Low-Rank Adaptation (LoRA) offers a resource-efficient way to personalize or specialize. However, LoRA is highly sensitive to hyperparameter choices, and exhaustive hyperparameter search is computationally expensive. To address this, we propose a Bayesian Optimization (BO) framework that leverages the domain knowledge of pre-trained LLMs to efficiently search for LoRA hyperparameters. Our approach repurposes a pre-trained LLM as a discrete-to-continuous mapping module to link hyperparameters and their domain knowledge to a continuous vector space, where BO is conducted. We design and control the mapping via language prompting, providing a domain-aware textual prompt that describes the relationships among hyperparameters and their respective roles. This allows us to explicitly inject domain knowledge about LoRA into the LLM in natural language. We also introduce an additional learnable token to capture residual information that is difficult to describe linguistically in the prompt. This aids BO to sample more high-performing hyperparameters. In addition, by leveraging the strong correlation observed between the performance obtained from full and subset training datasets in LoRA training regimes, we introduce proxy training and evaluation using a data subset. This significantly improves the efficiency of our method. We demonstrate that our hyperparameter, discovered with only about 30 iterations, achieves more than 20% performance improvement over standard hyperparameters found from about 45,000 combinations. Project page: this https URL
>
---
#### [replaced 069] AtomWorld: A Benchmark for Evaluating Spatial Reasoning in Large Language Models on Crystalline Materials
- **分类: cond-mat.mtrl-sci; cs.AI; cs.CL**

- **简介: 该论文提出AtomWorld基准，用于评估大语言模型在晶体材料结构建模中的空间推理能力，解决模型在复杂结构操作上的不足。**

- **链接: [https://arxiv.org/pdf/2510.04704](https://arxiv.org/pdf/2510.04704)**

> **作者:** Taoyuze Lv; Alexander Chen; Fengyu Xie; Chu Wu; Jeffrey Meng; Dongzhan Zhou; Yingheng Wang; Bram Hoex; Zhicheng Zhong; Tong Xie
>
> **摘要:** Large language models (LLMs) have shown promising potential in scientific research, enabling tasks ranging from knowledge retrieval to property prediction. Existing science benchmarks mainly focus on perceptual or knowledge-based tasks, largely ignoring the modelling tasks, a fundamental starting point for any real scientific research. For materials science, constructing and manipulating atomic structures is one of the most creative and least automated steps. In this work, we introduce AtomWorld, a benchmark designed to evaluate the abilities of LLMs on structure modifications. The benchmark includes ten fundamental actions under four widely used modelling categories, enabling verifiable evaluation metrics. We find that Claude Opus 4.6 generally performs the best. While the success rate decreases markedly with increasing modelling complexity, with particularly low success rates (below 12\% for rotation) for operations involving complex spatial relations. Our results suggest that contemporary LLMs are better suited as copilots for materials structure modelling rather than fully unsupervised autonomous scientific agents. Beyond evaluation, AtomWorld also serves as a testbed and playground for developing future structure-aware models, including reinforcement learning and agentic approaches.
>
---
#### [replaced 070] Thinking Fast, Thinking Wrong: Intuitiveness Modulates LLM Counterfactual Reasoning in Policy Evaluation
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于因果推理任务，研究LLM在政策评估中的反事实推理能力。通过构建40个案例库，分析模型在不同直观性情况下的表现，揭示了直观性对模型推理的影响及知识与推理的分离现象。**

- **链接: [https://arxiv.org/pdf/2604.10511](https://arxiv.org/pdf/2604.10511)**

> **作者:** Yanjie He
>
> **备注:** 10 pages, 6 figures, 6 tables
>
> **摘要:** Large language models (LLMs) are increasingly used for causal and counterfactual reasoning, yet their reliability in real-world policy evaluation remains underexplored. We construct a benchmark of 40 empirical policy evaluation cases drawn from economics and social science, each grounded in peer-reviewed evidence and classified by intuitiveness -- whether the empirical finding aligns with (obvious), is unclear relative to (ambiguous), or contradicts (counter-intuitive) common prior expectations. We evaluate four frontier LLMs across five prompting strategies with 8,000 experimental trials and analyze the results using mixed-effects logistic regression. Our findings reveal three key results: (1) a chain-of-thought (CoT) paradox, where chain-of-thought prompting dramatically improves performance on obvious cases but this benefit is substantially attenuated on counter-intuitive ones (interaction OR = 0.278, $p < 0.001$); (2) intuitiveness as the dominant factor, with case-level variance exceeding that of model choice or prompting strategy (ICC = 0.671); and (3) a knowledge-reasoning dissociation, where citation-based familiarity is unrelated to accuracy ($p = 0.84$), suggesting models possess relevant knowledge but fail to reason with it when findings contradict intuition. We frame these results through the lens of dual-process theory (System 1 vs. System 2) and argue that current LLMs' "slow thinking" achieves only partial inhibition of intuitive priors -- producing the form of deliberative reasoning without fully delivering its substance.
>
---
#### [replaced 071] Steering at the Source: Style Modulation Heads for Robust Persona Control
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于语言模型控制任务，解决 persona 控制中连贯性下降的问题。通过定位特定注意力头实现精准控制，提升安全性和效果。**

- **链接: [https://arxiv.org/pdf/2603.13249](https://arxiv.org/pdf/2603.13249)**

> **作者:** Yoshihiro Izawa; Gouki Minegishi; Koshi Eguchi; Sosuke Hosokawa; Kenjiro Taura
>
> **备注:** 8 main pages with appendix
>
> **摘要:** Activation steering offers a computationally efficient mechanism for controlling Large Language Models (LLMs) without fine-tuning. While effectively controlling target traits (e.g., persona), coherency degradation remains a major obstacle to safety and practical deployment. We hypothesize that this degradation stems from intervening on the residual stream, which indiscriminately affects aggregated features and inadvertently amplifies off-target noise. In this work, we identify a sparse subset of attention heads (only three heads) that independently govern persona and style formation, which we term Style Modulation Heads. Specifically, these heads can be localized via geometric analysis of internal representations, combining layer-wise cosine similarity and head-wise contribution scores. We demonstrate that intervention targeting only these specific heads achieves robust behavioral control while significantly mitigating the coherency degradation observed in residual stream steering. More broadly, our findings show that precise, component-level localization enables safer and more precise model control.
>
---
#### [replaced 072] ORACLE-SWE: Quantifying the Contribution of Oracle Information Signals on SWE Agents
- **分类: cs.MA; cs.CL; cs.SE**

- **简介: 该论文属于软件工程任务，旨在量化Oracle信息信号对SWE代理的影响，解决信号贡献不明确的问题，提出Oracle-SWE方法进行评估。**

- **链接: [https://arxiv.org/pdf/2604.07789](https://arxiv.org/pdf/2604.07789)**

> **作者:** Kenan Li; Qirui Jin; Liao Zhu; Xiaosong Huang; Yijia Wu; Yikai Zhang; Xin Zhang; Zijian Jin; Yufan Huang; Elsie Nallipogu; Chaoyun Zhang; Yu Kang; Saravan Rajmohan; Qingwei Lin; Wenke Lee; Dongmei Zhang
>
> **备注:** Under peer review; 37 pages, 10 figures, 5 tables
>
> **摘要:** Recent advances in language model (LM) agents have significantly improved automated software engineering (SWE). Prior work has proposed various agentic workflows and training strategies as well as analyzed failure modes of agentic systems on SWE tasks, focusing on several contextual information signals: Reproduction Test, Regression Test, Edit Location, Execution Context, and API Usage. However, the individual contribution of each signal to overall success remains underexplored, particularly their ideal contribution when intermediate information is perfectly obtained. To address this gap, we introduce Oracle-SWE, a unified method to isolate and extract oracle information signals from SWE benchmarks and quantify the impact of each signal on agent performance. To further validate the pattern, we evaluate the performance gain of signals extracted by strong LMs when provided to a base agent, approximating real-world task-resolution settings. These evaluations aim to guide research prioritization for autonomous coding systems.
>
---
#### [replaced 073] WaterSearch: A Quality-Aware Search-based Watermarking Framework for Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于文本水印任务，旨在解决水印强度与文本质量的平衡问题。提出WaterSearch框架，通过优化分布保真度和水印特征，提升文本质量与检测鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.00837](https://arxiv.org/pdf/2512.00837)**

> **作者:** Yukang Lin; Jiahao Shao; Shuoran Jiang; Wentao Zhu; Bingjie Lu; Xiangping Wu; Joanna Siebert; Qingcai Chen
>
> **摘要:** Watermarking acts as a critical safeguard in text generated by Large Language Models (LLMs). By embedding identifiable signals into model outputs, watermarking enables reliable attribution and enhances the security of machine-generated content. Existing approaches typically embed signals by manipulating token generation probabilities. Despite their effectiveness, these methods inherently face a trade-off between detectability and text quality: the signal strength and randomness required for robust watermarking tend to degrade the performance of downstream tasks. In this paper, we design a novel embedding scheme that controls seed pools to facilitate diverse parallel generation of watermarked text. Based on that scheme, we propose WaterSearch, a sentence-level, search-based watermarking framework adaptable to a wide range of existing methods. WaterSearch enhances text quality by jointly optimizing two key aspects: 1) distribution fidelity and 2) watermark signal characteristics. Furthermore, WaterSearch is complemented by a sentence-level detection method with strong attack robustness. We evaluate our method on three popular LLMs across ten diverse tasks. Extensive experiments demonstrate that our method achieves an average performance improvement of 51.01\% over state-of-the-art baselines at a watermark detectability strength of 95\%. In challenging scenarios such as short text generation and low-entropy output generation, our method yields performance gains of 47.78\% and 36.47\%, respectively. Moreover, under different attack senarios including insertion, synonym substitution and paraphrase attasks, WaterSearch maintains high detectability, further validating its robust anti-attack capabilities. Our code is available at \href{this https URL}{this https URL}.
>
---
#### [replaced 074] Good SFT Optimizes for SFT, Better SFT Prepares for Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于大语言模型后训练任务，解决SFT与RL阶段不匹配问题。提出PEAR方法，在SFT阶段调整损失权重，提升后续RL效果。**

- **链接: [https://arxiv.org/pdf/2602.01058](https://arxiv.org/pdf/2602.01058)**

> **作者:** Dylan Zhang; Yufeng Xu; Haojin Wang; Qingzhi Chen; Hao Peng
>
> **摘要:** Post-training of reasoning LLMs is a holistic process that typically consists of an offline SFT stage followed by an online reinforcement learning (RL) stage. However, SFT is often optimized in isolation to maximize SFT performance alone. We show that, after identical RL training, models initialized from stronger SFT checkpoints can significantly underperform those initialized from weaker ones. We attribute this to a mismatch typical in current SFT-RL pipelines: the distribution that generates the offline SFT data can differ substantially from the policy optimized during online RL, which learns from its own rollouts. We propose PEAR (Policy Evaluation-inspired Algorithm for Offline Learning Loss Re-weighting), an SFT-stage method that corrects this mismatch and better prepares the model for RL. PEAR uses importance sampling to reweight the SFT loss, with three variants operating at the token, block, and sequence levels. It can be used to augment standard SFT objectives and incurs little additional training overhead once probabilities for the offline data are collected. We conduct controlled experiments on verifiable reasoning games and mathematical reasoning tasks on Qwen 2.5 and 3 and DeepSeek-distilled models. PEAR consistently improves post-RL performance over canonical SFT, with pass at 8 gains up to a 14.6 percent on AIME2025. Our results suggest that PEAR is an effective step toward more holistic LLM post-training by designing and evaluating SFT with downstream RL in mind rather than in isolation.
>
---
#### [replaced 075] Survey of End-to-End Multi-Speaker Automatic Speech Recognition for Monaural Audio
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于多说话人语音识别任务，旨在解决单通道音频中多人重叠语音的识别与归属问题。工作包括梳理E2E架构、分析不同模型结构并评估其性能。**

- **链接: [https://arxiv.org/pdf/2505.10975](https://arxiv.org/pdf/2505.10975)**

> **作者:** Xinlu He; Jacob Whitehill
>
> **备注:** Accepted for publication in Computer Speech & Language (CSL)
>
> **摘要:** Monaural multi-speaker automatic speech recognition (ASR) remains challenging due to data scarcity and the intrinsic difficulty of recognizing and attributing words to individual speakers, particularly in overlapping speech. Recent advances have driven the shift from cascade systems to end-to-end (E2E) architectures, which reduce error propagation and better exploit the synergy between speech content and speaker identity. Despite rapid progress in E2E multi-speaker ASR, the field lacks a comprehensive review of recent developments. This survey provides a systematic taxonomy of E2E neural approaches for multi-speaker ASR, highlighting recent advances and comparative analysis. Specifically, we analyze: (1) architectural paradigms (SIMO vs.~SISO) for pre-segmented audio, analyzing their distinct characteristics and trade-offs; (2) recent architectural and algorithmic improvements based on these two paradigms; (3) extensions to long-form speech, including segmentation strategy and speaker-consistent hypothesis stitching. Further, we (4) evaluate and compare methods across standard benchmarks. We conclude with a discussion of open challenges and future research directions towards building robust and scalable multi-speaker ASR.
>
---
#### [replaced 076] "Be My Cheese?": Cultural Nuance Benchmarking for Machine Translation in Multilingual LLMs
- **分类: cs.CL**

- **简介: 该论文属于机器翻译任务，旨在解决文化语境在多语言大模型翻译中的评估问题。通过构建基准测试，评估模型在文化元素翻译上的表现。**

- **链接: [https://arxiv.org/pdf/2602.04729](https://arxiv.org/pdf/2602.04729)**

> **作者:** Madison Van Doren; Casey Ford; Jennifer Barajas; Riley VanMeter; Cory Holland
>
> **备注:** ACL 2026: Natural Language Generation, Evaluation, and Metrics (GEM) Workshop
>
> **摘要:** We present a large-scale human evaluation benchmark for assessing cultural localisation in machine translation produced by state-of-the-art multilingual large language models (LLMs). Existing MT benchmarks emphasise token-level and grammatical accuracy, but often overlook the pragmatic and culturally grounded competencies required for real-world localisation. Building on a pilot study of 87 translations across 20 languages, we evaluate 7 multilingual LLMs across 15 target languages with 5 native-speaker raters per language. Each rater scored both full-text translations and segment-level instances of culturally nuanced language (idioms, puns, holidays, and culturally embedded concepts) on an ordinal 0-3 quality scale; segment ratings additionally included an NA option for untranslated segments. Across full-text evaluations, mean overall quality is modest (1.68/3): GPT-5 (2.10/3), Claude Sonnet 4 (1.97/3), and Mistral Medium 3.1 (1.84/3) form the strongest tier with fewer catastrophic failures. Segment-level results show sharp category effects: holidays (2.20/3) and cultural concepts (2.19/3) translate notably better than idioms (1.65/3) and puns (1.45/3), and idioms are most likely to be left untranslated. Inter-rater reliability was assessed using Krippendorff's {\alpha} and Gwet's AC2, indicating moderate agreement overall (Krippendorff's {\alpha} = 0.45) with the lowest agreement for puns. These findings demonstrate a persistent gap between grammatical adequacy and cultural resonance. To our knowledge, this is the first multilingual, human-annotated benchmark focused explicitly on cultural nuance in translation and localisation. The results highlight the need for culturally informed training data, improved cross-lingual pragmatics, and evaluation frameworks that support systematic benchmarking of culturally grounded translation.
>
---
#### [replaced 077] To MRL or not to MRL: Text Embeddings are Robust to Truncation Without Matryoshka Learning, Except In Heavy Truncation Scenarios
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究文本嵌入在截断下的鲁棒性，比较MRL与随机截断的效果。任务是评估不同方法在嵌入压缩中的有效性，发现非MRL模型在轻度截断下表现更优。**

- **链接: [https://arxiv.org/pdf/2605.16608](https://arxiv.org/pdf/2605.16608)**

> **作者:** Sotaro Takeshita; Yurina Takeshita; Simone Paolo Ponzetto; Daniel Ruffinelli
>
> **摘要:** Matryoshka Representation Learning (MRL) is a widely adopted approach for training text encoders so they provide useful text representations at various sizes, available by simply truncating the resulting vectors at sizes pre-determined at training time. Recent works have shown that randomly truncating text embeddings has minimal impact in downstream performance unless vectors are reduced in size by at least 70%, suggesting that embeddings are already robust to truncation without the use of MRL. However, no prior work has compared random truncation to MRL, so it is unclear how the two methods compare as effective embedding reduction methods. In this paper, we study this by applying the same truncation used by MRL to models trained with and without MRL. Our results across several models and downstream tasks show that, unless heavily truncating embeddings (i.e. reducing their size by at least 80%), truncated embeddings of non-MRL models are competitive with, and often outperform models trained with MRL. This suggests that truncation robustness may not necessarily come from MRL, and that the choice of spending the additional training cost of MRL depends on whether heavy truncation is desired. We make our code available for reproduction.
>
---
#### [replaced 078] AgentDropoutV2: Optimizing Information Flow in Multi-Agent Systems via Test-Time Rectify-or-Reject Pruning
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出AgentDropoutV2，用于优化多智能体系统的信息流，解决错误传播问题。通过动态修正和修剪策略提升系统性能与适应性。**

- **链接: [https://arxiv.org/pdf/2602.23258](https://arxiv.org/pdf/2602.23258)**

> **作者:** Yutong Wang; Siyuan Xiong; Xuebo Liu; Wenkang Zhou; Liang Ding; Miao Zhang; Min Zhang
>
> **摘要:** While Multi-Agent Systems (MAS) excel in complex reasoning, they suffer from the cascading impact of erroneous information from individual agents. Current solutions often resort to rigid structural engineering or expensive fine-tuning, limiting their adaptability. We propose AgentDropoutV2 (ADv2), a test-time rectify-or-reject pruning framework that dynamically optimizes MAS information flow. Acting as an active firewall, ADv2 intercepts agent outputs and employs a retrieval-augmented rectifier to iteratively correct errors. This rectification is guided by an indicator pool, which is constructed offline by distilling error patterns from historical MAS failure trajectories. Irreparable outputs are subsequently pruned to prevent error propagation. Empirical results demonstrate that ADv2 significantly boosts performance on both fixed and dynamic MAS frameworks, achieving average accuracy gains of 6.39 and 2.28 percentage points on extensive math and code benchmarks, respectively. Furthermore, ADv2 exhibits remarkable adaptivity, dynamically modulating rectification efforts based on task difficulty to resolve a wide spectrum of error patterns. Our code is released at this https URL.
>
---
#### [replaced 079] HaluNet: Learning Hallucination Risk from Internal Signals in LLM Question Answering
- **分类: cs.CL**

- **简介: 该论文属于问答系统中的幻觉检测任务，旨在解决大语言模型生成答案时出现无依据的幻觉问题。提出HaluNet，利用模型内部信号评估答案风险，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2512.24562](https://arxiv.org/pdf/2512.24562)**

> **作者:** Chaodong Tong; Qi Zhang; Zhuojun Jiang; Lei Jiang; Yanbing Liu
>
> **备注:** 16 pages, 12 tables, and 11 figures. This version includes a major revision of the manuscript and updates the author list with the consent of all involved authors
>
> **摘要:** Large language models (LLMs) achieve strong question answering (QA) performance but can produce fluent answers unsupported by available evidence. Existing hallucination detectors often rely on external verification, repeated sampling, or test-time judge calls, which can be costly for real-time QA. We propose \textbf{HaluNet}, a lightweight hallucination risk estimator that uses internal signals from one model generation. HaluNet jointly models token likelihood, predictive entropy, and hidden-state information, allowing probabilistic, distributional, and semantic evidence to inform an answer-level risk score. It is trained with LLM-as-a-Judge labels as scalable weak supervision and evaluated with independent human and multi-judge assessments. Experiments on SQuAD, TriviaQA, and Natural Questions show that HaluNet improves answer-level risk ranking across in-domain and out-of-domain settings. On a 300-example human evaluation, HaluNet achieves 0.874 AUROC and 0.869 AUPRC; its top 20\% highest-risk answers contain 96.5\% errors, yielding a 2.06$\times$ lift over the base error rate.
>
---
#### [replaced 080] Graph Memory Transformer (GMT)
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出Graph Memory Transformer（GMT），通过引入记忆图替代Transformer中的FFN子层，解决语言模型中密集转换的可解释性问题，属于自然语言处理任务。**

- **链接: [https://arxiv.org/pdf/2604.23862](https://arxiv.org/pdf/2604.23862)**

> **作者:** Nicola Zanarini; Niccolò Ferrari; Evelina Lamma
>
> **备注:** 65 pages, 10 figures, 5 tables. Author list updated in arXiv metadata; no technical changes. Code available at this https URL
>
> **摘要:** We investigate whether the Feed-Forward Network (FFN) sublayer in a decoder-only transformer can be replaced by an explicit learned memory graph while preserving the surrounding autoregressive architecture. The proposed Graph Memory Transformer (GMT) keeps causal self-attention intact, but replaces the usual per-token FFN transformation with a memory cell that routes token representations over a learned bank of centroids connected by a learned directed transition matrix. In the base GMT v7 instantiation studied here, each of 16 transformer blocks contains 128 centroids, a 128 * 128 edge matrix, gravitational source routing, token-conditioned target selection, and a gated displacement readout. The cell therefore returns movement from an estimated source memory state toward a target memory state, rather than a retrieved value. The resulting model is a fully decoder-only language model with 82.2M trainable parameters and no dense FFN sublayers, compared with a 103.0M-parameter dense GPT-style baseline used in the evaluation. The base v7 model trains stably and exposes centroid usage, transition structure, and source-to-target movement as directly inspectable quantities of the forward computation. It remains behind the larger dense baseline in validation loss and perplexity (3.5995/36.58 vs. 3.2903/26.85), while showing close zero-shot benchmark behavior under the evaluated setting. These results are not intended as a state-of-the-art claim; they support the viability and structural interpretability of replacing dense within-token transformation with graph-mediated memory navigation. Broader scaling, optimized kernels, and more extensive benchmark evaluation are left for subsequent work.
>
---
#### [replaced 081] Revisiting the Reliability of Language Models in Instruction-Following
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，研究大模型在指令遵循中的可靠性问题。针对真实场景中用户表述变化导致性能下降的问题，提出新指标和评估基准，分析并探索改进方法。**

- **链接: [https://arxiv.org/pdf/2512.14754](https://arxiv.org/pdf/2512.14754)**

> **作者:** Jianshuo Dong; Yutong Zhang; Yan Liu; Zhenyu Zhong; Tao Wei; Chao Zhang; Han Qiu
>
> **备注:** ACL 2026 main oral
>
> **摘要:** Advanced LLMs have achieved near-ceiling instruction-following accuracy on benchmarks such as IFEval. However, these impressive scores do not necessarily translate to reliable services in real-world use, where users often vary their phrasing, contextual framing, and task formulations. In this paper, we study nuance-oriented reliability: whether models exhibit consistent competence across cousin prompts that convey analogous user intents but with subtle nuances. To quantify this, we introduce a new metric, reliable@k, and develop an automated pipeline that generates high-quality cousin prompts via data augmentation. Building upon this, we construct IFEval++ for systematic evaluation. Across 20 proprietary and 26 open-source LLMs, we find that current models exhibit substantial insufficiency in nuance-oriented reliability -- their performance can drop by up to 61.8% with nuanced prompt modifications. What's more, we characterize it and explore three potential improvement recipes. Our findings highlight nuance-oriented reliability as a crucial yet underexplored next step toward more dependable and trustworthy LLM behavior. Our code and benchmark are accessible: this https URL.
>
---
#### [replaced 082] A Tutorial on Diffusion Theory: From Differential Equations to Diffusion Models
- **分类: cs.LG; cs.CL**

- **简介: 本文探讨扩散模型的数学基础，将其统一为微分方程视角，解决生成模型中的逆向采样问题，涵盖DDPM、DDIM等方法。**

- **链接: [https://arxiv.org/pdf/2605.22586](https://arxiv.org/pdf/2605.22586)**

> **作者:** Jiayi Fu; Yuxia Wang
>
> **备注:** A detailed tutorial on Diffusion models and SDE
>
> **摘要:** Diffusion models have emerged as a dominant framework for generative modeling, but their mathematical foundations are often presented separately through diffusion probabilistic models, score-based modeling, stochastic differential equations, and numerical sampling methods. We write this tutorial to provide a unified and self-contained account of these viewpoints from the perspective of differential equations. Starting from a conditional Gaussian noising process, we derive ordinary differential equation (ODE) and stochastic differential equation (SDE) representations, pass to the corresponding marginal forward dynamics, and then obtain the reverse-time SDE and probability-flow ODE that make generation possible. We show that the central unknown quantity in reverse sampling is the marginal score, explain how score matching becomes the standard denoising objective under a noise-prediction parameterization, and discuss practical reverse-time sampling and guidance. We further place DDPM, DDIM, flow matching, and score-based SDEs in a common framework, and conclude with diffusion language models in continuous embedding space together with a brief discussion of discrete masked-token diffusion. The tutorial is intended as a bridge between the analytical foundations of diffusion processes and the modern generative algorithms built upon them.
>
---
#### [replaced 083] MAGA-Bench: Machine-Augment-Generated Text via Alignment Detection Benchmark
- **分类: cs.CL**

- **简介: 该论文属于机器生成文本检测任务，旨在解决MGT与HWT区分困难的问题。通过引入对齐增强方法构建MAGA基准，提升检测器的泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.04633](https://arxiv.org/pdf/2601.04633)**

> **作者:** Anyang Song; Ying Cheng; Yiqian Xu; Rui Feng
>
> **摘要:** Machine-Generated Text (MGT) is becoming increasingly difficult to distinguish from Human-Written Text (HWT). This trend has exacerbated malicious activities such as fake news and online fraud. The generalization ability of fine-tuned detectors relies heavily on dataset quality, and simply expanding the sources of MGT may become increasingly insufficient. Further augmentation of the generation process is required. Based on HC-Var's theory, enhancing the human-like alignment of MGT not only facilitates robustness testing of existing detectors but also boosts the generalization ability of detectors fine-tuned on such aligned MGT datasets. Therefore, we propose the \textbf{M}achine-\textbf{A}ugment-\textbf{G}enerated Text via \textbf{A}lignment (MAGA) Detection Benchmark. MAGA integrates several alignment methods, ranging from prompt construction to \textbf{G}enerator-\textbf{D}etector \textbf{A}dversarial \textbf{R}einforcement \textbf{L}earning (GDARL) and the reasoning process. In our experiments, the RoBERTa detector fine-tuned on MAGA achieves an average improvement of 4.60\% in generalization AUC. Conversely, the aligned MGTs in MAGA also lead to an average decrease of 8.13\% in the AUC of selected detectors. We hope the MAGA Benchmark will provide valuable insights for future research on the generalization ability of MGT detectors.
>
---
#### [replaced 084] How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究LLM在链式推理中的潜在规划能力，旨在解决其内部状态与推理路径的关系问题。通过探针方法分析隐藏状态，发现LLM具有短视的规划特性，并提出增强不确定性估计的假设。**

- **链接: [https://arxiv.org/pdf/2602.02103](https://arxiv.org/pdf/2602.02103)**

> **作者:** Liyan Xu; Mo Yu; Fandong Meng; Jie Zhou
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** Chain-of-thought (CoT) reasoning has become a central mechanism for eliciting multi-step reasoning in Large Language Models (LLMs). Yet recent evidence presents a tension: hidden states appear to already encode future reasoning before CoT fully unfolds, while explicit steps still remain crucial for tasks requiring compositional computation. To deepen the understanding between LLM's internal states and its verbalized reasoning trajectories, we investigate the latent planning strength of LLMs, through our probing method, Tele-Lens, applying to hidden states across diverse task domains. Our empirical results indicate that LLMs exhibit a myopic horizon, primarily conducting incremental transitions without precise global planning. Leveraging this characteristic, we propose a hypothesis on enhancing uncertainty estimation of CoT, which we validate that a sparse set of pivot positions can effectively represent the uncertainty of the entire path. We further underscore the significance of exploiting CoT dynamics, and demonstrate that automatic recognition of CoT bypass can be achieved without performance degradation. Our code, data and models are released at this https URL.
>
---
#### [replaced 085] Who can we trust? LLM-as-a-jury for Comparative Assessment
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究LLM作为评判者的可靠性问题，旨在解决LLM在自然语言生成评估中判断不一致和不可靠的问题。提出BT-sigma模型，通过配对比较同时推断项目排名和评判者可靠性。**

- **链接: [https://arxiv.org/pdf/2602.16610](https://arxiv.org/pdf/2602.16610)**

> **作者:** Mengjie Qian; Guangzhi Sun; Mark J.F. Gales; Kate M. Knill
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** Large language models (LLMs) are increasingly applied as automatic evaluators for natural language generation assessment often using pairwise comparative judgements. Existing approaches typically rely on single judges or aggregate multiple judges assuming equal reliability. In practice, LLM judges vary substantially in performance across tasks and evaluation aspects, and their judgment probabilities may be biased and inconsistent. Furthermore, human-labelled supervision for judge calibration may be unavailable. We first empirically demonstrate that inconsistencies in LLM comparison probabilities exist and show that it limits the effectiveness of direct probability-based ranking. To address this, we study the LLM-asa-jury setting and propose BT-sigma, a judge-aware extension of the Bradley-Terry model that introduces a discriminator parameter for each judge to jointly infer item rankings and judge reliability from pairwise comparisons alone. Experiments on benchmark NLG evaluation datasets show that BT-sigma consistently outperforms averaging-based aggregation methods, and that the learned discriminators strongly correlate with independent measures of the cycle consistency of LLM judgments. Further analysis reveals that BT-sigma can be interpreted as an unsupervised calibration mechanism that improves aggregation by modelling judge reliability.
>
---
#### [replaced 086] Beyond Normalization: Rethinking the Partition Function as a Difficulty Scheduler for RLVR
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于强化学习任务，旨在解决LLMs生成多样性与奖励最大化之间的矛盾。通过重新解释分区函数为难度调度器，提出PACED-RL框架提升样本效率。**

- **链接: [https://arxiv.org/pdf/2602.12642](https://arxiv.org/pdf/2602.12642)**

> **作者:** Dohyung Kim; Minbeom Kim; Jeonghye Kim; Sangmook Lee; Sojeong Rhee; Kyomin Jung
>
> **摘要:** Reward-maximizing RL methods have shown to be capable of enhancing the reasoning performance of LLMs, but often lead to reduced generation diversity. Recent works address this issue by adopting GFlowNets, training LLMs to match a target distribution while jointly learning its partition function. In contrast to prior works that treat this partition function solely as a normalizer, we reinterpret it as a per-prompt expected-reward (i.e., online accuracy) signal, leveraging this unused information to improve sample efficiency. Specifically, we first establish a theoretical relationship between the partition function and per-prompt accuracy estimates. Building on this key insight, we propose Partition Function-Guided RL (PACED-RL), a post-training framework that leverages accuracy estimates to prioritize informative question prompts during training, and further improves sample efficiency through an accuracy estimate error-prioritized replay. Crucially, both components reuse information already produced during GFlowNet training, effectively amortizing the compute overhead into the existing optimization process. Extensive experiments across diverse benchmarks demonstrate strong performance improvements over GRPO and prior GFlowNet approaches, highlighting PACED-RL as a promising direction for a more sample efficient distribution-matching training for LLMs.
>
---
#### [replaced 087] Demystifying Scientific Problem-Solving in LLMs by Probing Knowledge and Reasoning
- **分类: cs.CL**

- **简介: 该论文属于科学推理任务，旨在解决LLMs在科学问题解决中的知识与推理能力评估问题。提出SciReas和KRUX框架，分析知识与推理的作用。**

- **链接: [https://arxiv.org/pdf/2508.19202](https://arxiv.org/pdf/2508.19202)**

> **作者:** Alan Li; Yixin Liu; Arpan Sarkar; Doug Downey; Arman Cohan
>
> **备注:** 33 pages, 18 figures
>
> **摘要:** Scientific problem solving poses unique challenges for LLMs, requiring both deep domain knowledge and the ability to apply such knowledge through complex reasoning. While automated scientific reasoners hold great promise for assisting human scientists, there is currently no widely adopted holistic benchmark for evaluating scientific reasoning, and few approaches systematically disentangle the distinct roles of knowledge and reasoning in these tasks. To address these gaps, we introduce SciReas, a diverse suite of existing benchmarks for scientific reasoning tasks, and SciReas-Pro, a selective subset that requires more complex reasoning. Our holistic evaluation surfaces insights about scientific reasoning performance that remain hidden when relying on individual benchmarks alone. We then propose KRUX, a probing framework for studying the distinct roles of reasoning and knowledge in scientific tasks. Combining the two, we conduct an in-depth analysis that yields several key findings: (1) Retrieving task-relevant knowledge from model parameters is a critical bottleneck for LLMs in scientific reasoning; (2) Reasoning models consistently benefit from external knowledge added in-context on top of the reasoning enhancement; (3) Enhancing verbalized reasoning improves LLMs' ability to surface task-relevant knowledge.
>
---
#### [replaced 088] From Meta-Thought to Execution: Cognitively Aligned Post-Training for Generalizable and Reliable LLM Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于大模型推理优化任务，旨在解决现有方法与人类认知不匹配的问题。通过分阶段训练，提升模型的泛化性和执行可靠性。**

- **链接: [https://arxiv.org/pdf/2601.21909](https://arxiv.org/pdf/2601.21909)**

> **作者:** Shaojie Wang; Liang Zhang
>
> **摘要:** Current LLM post-training methods optimize complete reasoning trajectories through Supervised Fine-Tuning (SFT) followed by outcome-based Reinforcement Learning (RL). While effective, a closer examination reveals a fundamental gap: this approach does not align with how humans actually solve problems. Human cognition naturally decomposes problem-solving into two distinct stages: first acquiring abstract strategies (i.e., meta-knowledge) that generalize across problems, then adapting them to specific instances. In contrast, by treating complete trajectories as basic units, current methods are inherently problem-centric, entangling abstract strategies with problem-specific execution. To address this misalignment, we propose a cognitively-inspired framework that explicitly mirrors the two-stage human cognitive process. Specifically, Chain-of-Meta-Thought CoMT focuses supervised learning on abstract reasoning patterns without specific executions, enabling acquisition of generalizable strategies. Confidence-Calibrated Reinforcement Learning (CCRL) then optimizes task adaptation via confidence-aware rewards on intermediate steps, preventing overconfident errors from cascading and improving execution reliability. Experiments across four models and ten benchmarks show 2.10% and 3.86% improvements in-distribution and out-of-distribution respectively over standard methods, while remaining highly robust to variations in teacher model selection, optimization methods, and symbolic perturbations.
>
---
#### [replaced 089] When AI Takes Sides on Questions of Faith: Persistent Asymmetries in AI-Mediated Faith Guidance
- **分类: cs.CL; cs.CY**

- **简介: 论文研究AI在处理宗教转换建议时的偏见问题，发现模型对不同宗教存在系统性偏好。任务属于AI伦理与偏见分析，旨在揭示LLM在宗教指导中的不对称性。**

- **链接: [https://arxiv.org/pdf/2605.22975](https://arxiv.org/pdf/2605.22975)**

> **作者:** Brett Israelsen; Sheryl Carty; Josh Coates; Nancy Fulda; Julie Park; Pete Whiting
>
> **备注:** w/ persuasive language analysis
>
> **摘要:** We ask whether large language models (LLMs) treat queries about religious conversion symmetrically. The answer is no. When asked for advice on hypothetical faith transitions from religion A->B vs. religion B->A , models exhibited consistent asymmetries, favoring some religions while subtly discouraging conversion to others. On average Catholic, Bahá'í, and Sikh religions were broadly favored (high support for joining, low support for leaving), while Atheists, Agnostics, and Jehovah's Witnesses were primarily disfavored. Patterns varied by model size and model provider, with Grok 4.20 exhibiting the strongest asymmetries. We tested 20 commercial and open-source language models across 182 religion pairings using a human-verified LLM-as-judge framework. Each model was probed via interactions with a simulated user asking for advice on a potential faith conversion. Models tended to use more encouraging language for some faith transitions over others; these patterns were systematically repeatable across multiple trials. All LLMs tested exhibited reproducible asymmetry, though the pattern of preferences differed for each. Overall preferences persist across multiple question phrasings and variations in the religious pairing dataset. Taken together, these results suggest that asymmetry is a robust property of model behavior rather than an artifact of how the models' answers were scored. It is important to consider that any imbalances deployed and reproduced at scale can have real-world implications.
>
---
#### [replaced 090] EVADE: LLM-Based Explanation Generation and Validation for Error Detection in NLI
- **分类: cs.CL**

- **简介: 该论文提出EVADE框架，利用大语言模型生成并验证解释，以检测自然语言推理中的标注错误。任务为NLI数据集质量提升，解决标签变异下的错误检测问题。**

- **链接: [https://arxiv.org/pdf/2511.08949](https://arxiv.org/pdf/2511.08949)**

> **作者:** Longfei Zuo; Barbara Plank; Siyao Peng
>
> **摘要:** High-quality datasets are critical for training and evaluating reliable NLP models. In tasks like natural language inference (NLI), human label variation (HLV) arises when multiple labels are valid for the same instance, making it difficult to separate annotation errors from plausible variation. An earlier framework, VARIERR (Weber-Genzel et al., 2024), asks multiple annotators to explain their label decisions in the first round and flags errors through validity judgments in the second round. However, conducting two rounds of manual annotation is costly and may limit the coverage of plausible labels or explanations. Our study proposes a new framework, EVADE, for generating and validating explanations to detect errors using large language models (LLMs). We perform a comprehensive analysis comparing human- and LLM-detected errors for NLI across distribution comparison, validation overlap, and impact on model fine-tuning. Our experiments demonstrate that LLM validation refines generated explanation distributions to more closely align with human annotations, and that removing LLM-detected errors from training data yields improvements in fine-tuning performance than removing errors identified by human annotators. This highlights the potential to scale error detection, reducing human effort while improving dataset quality under label variation.
>
---
#### [replaced 091] Modeling Hierarchical Thinking in Large Reasoning Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于推理模型研究任务，旨在解决复杂任务中推理路径不一致的问题。通过构建有限状态机模型，捕捉推理过程中的认知状态，提升模型的可解释性和优化效率。**

- **链接: [https://arxiv.org/pdf/2510.22437](https://arxiv.org/pdf/2510.22437)**

> **作者:** G M Shahariar; Erfan Shayegani; Ali Nazari; Nael Abu-Ghazaleh
>
> **备注:** Accepted in ICML 2026 as Oral
>
> **摘要:** Large Reasoning Models (LRMs) solve complex tasks by generating long Chain-of-Thought (CoT) sequences; however, the emergent dynamics governing reasoning trajectories are not well understood and can lead to inconsistencies and reasoning pathologies. In this work, we propose to approximate LRM's emerging hierarchical reasoning dynamics as a trajectory within a Finite State Machine (FSM) transitioning among six abstract cognitive states. We demonstrate that these states and transitions can be captured in the latent state of the model. We believe that this representation can have different applications in the interpretability and optimization of LRM models. For example, by analyzing the topology of these transitions, we identify statistical shifts in reasoning strategies that help identify effective reasoning chains from those that fail. To illustrate these potential advantages, we propose Q-Value guided steering, a training-free inference-time control method that treats reasoning as a planning problem. We estimate the long-horizon utility of state transitions and apply sparse, orthogonal activation steering at sentence boundaries to align the CoT generation with optimal reasoning policies. Experiments across four benchmarks (AIME25, MATH-500, GSM8k, and GPQA Diamond) using three state-of-the-art open reasoning models demonstrate that Q-Value steering policy achieves significant performance gains with "surgical" efficiency, often requiring 25 times fewer interventions than greedy and weighted baselines, which suggests that reasoning can be effectively controlled by guiding high-level cognitive dynamics rather than micro-managing token generation. Code is available at: this https URL.
>
---
#### [replaced 092] Ask Now, Use Later: Benchmarking the Proactivity Gap in Long-Lived LLM Agents
- **分类: cs.CL**

- **简介: 该论文研究长期LLM代理的主动性缺口问题，提出ATRBench基准评估代理是否适时询问用户未来可用的信息。任务属于人工智能中的主动学习与用户交互领域。**

- **链接: [https://arxiv.org/pdf/2605.28108](https://arxiv.org/pdf/2605.28108)**

> **作者:** Bin Wu; Guanyun Zou; Bingbing Wang; Huan Zhao; Chuan Shi
>
> **摘要:** A long-lived LLM agent, such as OpenClaw, earns its value by acting on a user's preferences and constraints across sessions, not just the current request. Yet today's agents keep what a user volunteers but rarely ask for what stays unspoken, leaving a proactivity gap in long-lived LLM agents: an agent cannot act on a preference it never obtained. As users delegate more of their affairs to agents, the impact of this gap grows. We isolate one concrete, controllable slice of this gap as Ask-to-Remember (ATR): the agent decides whether to ask now for a reusable user preference that the current task does not need but a later session with the same user will. ATR is hard even to evaluate: the right question is underdetermined and its payoff deferred to tasks that may never arise. ATRBench, to the best of our knowledge the first ATR benchmark, makes it measurable by fixing each user's preferences as hidden ground truth, so success demands asking, not recall. Across eight frontier LLM agents, defaults fall at least 62 points below an oracle handed the relevant preference, and prompting closes little of it. Diagnostics identify acquisition as the bottleneck. ATRBench surfaces this proactivity gap in current agents and offers a diagnostic testbed for closing it.
>
---
#### [replaced 093] OpenSkillEval: Automatically Auditing the Open Skill Ecosystem for LLM Agents
- **分类: cs.CL**

- **简介: 该论文属于LLM代理技能评估任务，旨在解决技能质量评价与选择问题。提出OpenSkillEval框架，自动构建任务实例并评估技能效果。**

- **链接: [https://arxiv.org/pdf/2605.23657](https://arxiv.org/pdf/2605.23657)**

> **作者:** Jiahao Ying; Boxian Ai; Wei Tang; Siyuan Liu; Yixin Cao
>
> **摘要:** Skills, i.e., structured workflow instructions distilled for large language models (LLMs), are becoming an increasingly important mechanism for improving agent performance on real-world downstream tasks. However, as the open-source skill ecosystem rapidly expands, it remains unclear how different models and agent frameworks interact with skills, how to evaluate skill quality, and how users should select skills under practical cost-performance trade-offs. In this paper, we present \textsc{OpenSkillEval}, an automatic evaluation framework for both skill-augmented agent systems and the skills themselves. Instead of relying on static benchmarks, \textsc{OpenSkillEval} automatically constructs realistic task instances from evolving real-world artifacts across five categories of downstream applications: presentation generation, front-end web design, poster generation, data visualization, and report generation. It further collects and organizes community-contributed skills for controlled comparison under unified task settings. Using more than 600 dynamically generated task instances and 30 open-source skills, we conduct a systematic evaluation of state-of-the-art models and agent frameworks. Our results show that skill availability does not guarantee effective skill usage, that the benefit of skill augmentation depends strongly on both the underlying model and the agent framework, and that many publicly popular skills do not consistently outperform base agents without skills. These findings highlight the need for dynamic, task-grounded evaluation and provide practical insights into the design, selection, and deployment of skills for LLM agents. Additional cases and benchmark resources are available on the project website: this https URL.
>
---
#### [replaced 094] Early Detection of Misinformation for Infodemic Management: A Domain Adaptation Approach
- **分类: cs.CL; cs.LG; cs.SI**

- **简介: 该论文属于信息检测任务，旨在解决疫情初期虚假信息检测问题。针对传统方法依赖标注数据的不足，提出一种同时处理特征分布和标签模式差异的新方法。**

- **链接: [https://arxiv.org/pdf/2406.10238](https://arxiv.org/pdf/2406.10238)**

> **作者:** Minjia Mao; Xiaohang Zhao; Xiao Fang
>
> **摘要:** An infodemic refers to an enormous amount of true information and misinformation disseminated during a disease outbreak. Detecting misinformation at the early stage of an infodemic is key to reduce its harm to public health. An early stage infodemic is characterized by a large volume of unlabeled information concerning a disease. As a result, conventional misinformation detection methods are not suitable for this misinformation detection task because they rely on labeled information in the infodemic domain to train their models. To address this limitation, state-of-the-art methods learn their models using labeled information in other domains to detect misinformation in the infodemic domain. The efficacy of these methods depends on their ability to mitigate both covariate shift (i.e., differences in feature distributions) and concept shift (i.e., differences in labeling patterns) between the infodemic domain and the domains from which they leverage labeled information. However, these methods focus on mitigating covariate shift but overlook concept shift, rendering them less effective for the task. In response, we theoretically show the necessity of tackling both covariate and concept shifts as well as how to operationalize each of them. Built on the theoretical analysis, we develop a novel misinformation detection method that addresses both covariate and concept shifts. Using real-world datasets, we conduct extensive empirical evaluations to demonstrate the superior performance of our method over state-of-the-art misinformation detection methods as well as prevalent domain adaptation methods that can be tailored to solve the misinformation detection task.
>
---
#### [replaced 095] MedMosaic: A Challenging Large Scale Benchmark of Diverse Medical Audio
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文提出MedMosaic，一个大规模医学音频问答基准，用于评估语言与音频推理模型。针对医学音频数据收集难、标注成本高的问题，构建多样化数据集，包含46,701个问答对，涵盖多种题型，以测试多跳推理和答案生成能力。**

- **链接: [https://arxiv.org/pdf/2605.00969](https://arxiv.org/pdf/2605.00969)**

> **作者:** Harshit Rajgarhia; Shuubham Ojha; Asif Shaik; Akhil Pothanapalli; Rachuri Lokesh; Abhishek Mukherji; Prasanna Desikan
>
> **备注:** Accepted at ICML 2026
>
> **摘要:** Medical audio data is difficult to collect due to privacy regulations and high annotation costs arising from domain expertise. Thus, existing benchmarks tend to underrepresent complex medical audio scenarios. To address this challenge, we present MedMosaic, a medical audio question-answering dataset designed to benchmark language and audio reasoning models under realistic clinical constraints. MedMosaic features a diverse range of medical audio types, including condition-related physiological sounds, carefully constructed synthetic voices to mimic speech with artifacts as well as real short and long length clinical conversations to model varying context lengths. The dataset also features a total of 46,701 question-answer pairs, spanning categories such as multiple-choice, sequential multi-turn, and open-ended question-answers, enabling systematic evaluation of multi-hop reasoning and answer generation capabilities. Benchmarking 13 audio and multimodal reasoning models reveals that reasoning remains challenging for all evaluated systems, with substantial performance variation across question types. In particular, even state-of-the-art model like Gemini-2.5-pro can only achieve 68.1% accuracy approximately. These findings underscore persistent limitations in medical reasoning and highlight the need for more robust, domain-specific multimodal reasoning models. A sample of benchmark data is available here: this https URL
>
---
#### [replaced 096] Over-Refusal and Representation Subspaces: A Mechanistic Analysis of Task-Conditioned Refusal in Aligned LLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，研究对齐大模型中的过度拒绝现象。通过分析拒绝的表征几何，揭示过度拒绝与有害拒绝的本质差异，提出需进行任务特定的干预。**

- **链接: [https://arxiv.org/pdf/2603.27518](https://arxiv.org/pdf/2603.27518)**

> **作者:** Utsav Maskey; Mark Dras; Usman Naseem
>
> **备注:** Preprint
>
> **摘要:** Aligned language models that are trained to refuse harmful requests also exhibit over-refusal: they decline safe instructions that seemingly resemble harmful instructions. A natural approach is to ablate the global refusal direction, steering the hidden-state vectors away or towards the harmful-refusal examples, but this corrects over-refusal only incidentally while disrupting the broader refusal mechanism. In this work, we analyse the representational geometry of both refusal types to understand why this happens. We show that harmful-refusal directions are task-agnostic and can be captured by a single global vector, whereas over-refusal directions are task-dependent: they reside within the benign task-representation clusters, vary across tasks, and span a higher-dimensional subspace. Linear probing suggests that the two refusal types are representationally distinct from the early transformer layers. These findings provide a mechanistic explanation of why global direction ablation alone cannot address over-refusal, and establish that task-specific geometric interventions are necessary.
>
---
#### [replaced 097] DiffRetriever: Parallel Representative Tokens for Retrieval with Diffusion Language Models
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于信息检索任务，旨在提升检索效果。针对现有方法依赖单一向量表示的问题，提出DiffRetriever，利用扩散语言模型的掩码预测能力，实现高效多粒度检索。**

- **链接: [https://arxiv.org/pdf/2605.07210](https://arxiv.org/pdf/2605.07210)**

> **作者:** Shuai Wang; Yu Yin; Shengyao Zhuang; Bevan Koopman; Guido Zuccon
>
> **备注:** Updated analysis, ablation and benchmark with sota retrievers, indexing storage/latency ablation, isolating the effectiveness gain
>
> **摘要:** This paper shows how diffusion language models (DLMs) can be used as effective and efficient retrievers. Existing DLM-based retrievers (e.g., DiffEmbed) follow BERT-style encoding, representing each query or passage as a single mean-pooled vector. This ignores how DLMs are trained to generate responses through masked-position prediction under bidirectional attention, a capability that can provide stronger retrieval signals. We propose DiffRetriever, which uses the DLM's native masked-position prediction directly for retrieval. For each query or passage, DiffRetriever appends one or more masked positions, using the outputs as retrieval representations in a single forward pass. With one masked position, single-representation DiffRetriever already improves over DiffEmbed on the same backbones. DiffRetriever also naturally extends to multi-representation retrieval: DLMs process multiple masked positions jointly, enabling ColBERT-style fine-grained matching with little additional encoding latency. In autoregressive LLM retrievers, the same multi-representation strategy requires sequential decoding and therefore incurs much higher latency. DiffRetriever obtains the strongest aggregate effectiveness within our matched comparison, outperforming DiffEmbed, PromptReps, and RepLLaMA. Masked-position counts selected on training data transfer well across datasets, while per-query variation suggests headroom for adaptive allocation. Code is available at this https URL.
>
---
#### [replaced 098] DialToM: A Theory of Mind Benchmark for Forecasting State-Driven Dialogue Trajectories
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出DialToM，一个用于评估对话中心智理论的基准。任务是预测基于心理状态的对话轨迹，解决AI在社会推理上的不足。工作包括构建数据集和评估模型表现。**

- **链接: [https://arxiv.org/pdf/2604.20443](https://arxiv.org/pdf/2604.20443)**

> **作者:** Neemesh Yadav; Palakorn Achananuparp; Jing Jiang; Ee-Peng Lim
>
> **备注:** Submitted to EMNLP 2026
>
> **摘要:** We introduce DialToM, an annotated Theory of Mind (ToM) benchmark built from naturalistic human-human dialogues using a multiple-choice evaluation framework. Concurrent with recent work showing a gap between explicit mental-state inference and applied ToM in synthetic settings~\cite{gu2024simpletom}, we establish a stricter \emph{State-Driven Diagnostic Probe} in which models must forecast state-consistent dialogue trajectories solely from isolated mental-state profiles without dialogue context. Our evaluation reveals a systematic reasoning asymmetry -- LLMs excel at inferring mental states (Literal ToM) but struggle to leverage them for social forecasting (Functional ToM). Crucially, a domain expert achieves 100\% accuracy on this task, proving its validity and establishing a stark human-AI capability gap. Further, a teacher-student reasoning injection probe shows that Gemini 3 Pro -- which establishes the leading baseline -- possesses robust Functional ToM capabilities for context-free forecasting that are transferable to weaker models. DialToM, its evaluation code, and dataset are publicly available at this https URL.
>
---
#### [replaced 099] Obfuscation Rules for Detecting and Detoxifying Korean Toxicity
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的毒性检测与净化任务，旨在解决韩语中伪装毒性表达的识别与净化问题。研究构建了KOTOX数据集，提出 obfuscation 规则框架，提升模型对伪装文本的处理能力。**

- **链接: [https://arxiv.org/pdf/2510.10961](https://arxiv.org/pdf/2510.10961)**

> **作者:** Yejin Lee; Su-Hyeon Kim; Hyundong Jin; Dayoung Kim; Yeonsoo Kim; Yo-Sub Han
>
> **备注:** 26 pages, 12 figures, 24 tables
>
> **摘要:** As language models become increasingly deployed in online environments, toxicity detection and detoxification have received growing attention. Existing studies primarily focus on non-obfuscated text, which limits robustness when users intentionally disguise toxic expressions. In particular, Korean toxic expressions can be easily disguised through agglutinative morphology and Hangeul-specific orthographic variation. However, obfuscation in Korean remains largely unexplored, which motivates us to introduce a KOTOX: Korean toxic dataset for deobfuscation and detoxification. We categorize Korean obfuscation patterns into linguistically grounded classes, define transformation rules derived from real-world examples, and provide the resulting obfuscation framework as an open transformation package. Using these rules, we provide paired neutral and toxic sentences alongside their obfuscated counterparts. Models trained on our dataset better handle obfuscated text without sacrificing performance on non-obfuscated text. This is the first dataset that simultaneously supports deobfuscation and detoxification for the Korean language. We expect the dataset to facilitate better understanding and mitigation of obfuscated toxic content in LLM for Korean. Our code and data are available at this https URL.
>
---
#### [replaced 100] The Alignment Floor: How Persona Customization Breaks Safety in Weakly-Aligned LLMs
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文研究弱对齐大语言模型在角色定制下的安全问题，提出“对齐下限”指标，通过实验分析不同角色对模型一致性的影响，旨在为部署前提供审计方法。**

- **链接: [https://arxiv.org/pdf/2605.27382](https://arxiv.org/pdf/2605.27382)**

> **作者:** Xing Zhang; Guanghui Wang; Yanwei Cui; Wei Qiu; Ziyuan Li; Bing Zhu; Peiyang He
>
> **摘要:** Telling an LLM to "be enthusiastic" raises its sycophancy rate from 30\% to 50\% on a lightly-aligned model, but has zero effect on a strongly-aligned one. We define this gap as the alignment floor, $\Delta_{\text{floor}}(m)=\max_pS(m,p)-\min_pS(m,p)$, the range of sycophancy rates a model produces across persona conditions, and treat sycophancy as a persona-conditional property rather than a fixed model property. Pluralistic AI relies on behavioral adaptation via persona prompts like "be creative" or "be thorough", which let systems respect diverse user values and communication styles; the safety question is how much customization a given model can absorb before its truthfulness shifts. We present a controlled case study contrasting a strongly-aligned RLHF + Constitutional-AI model (Claude Sonnet 4.6) with a more lightly-aligned model (Amazon Nova Lite), spanning seven persona conditions and five tasks for 1800 total runs. An existence-pair result motivates per-model auditing: there is at least one strongly-aligned model with $\Delta_{\text{floor}}=5$pp (within 5pp of the 15\% control rate) and at least one lightly-aligned model with 45pp (5\%--50\% range). On the lightly-aligned model, all five Big Five personas increase sycophancy over control, and counterintuitively Agreeableness produces the smallest increase, not the largest. The single largest effect in the study is constructive: a Skeptic persona reduces sycophancy by 25pp on the lightly-aligned model, and is the only persona that instructs resistance against user claims rather than engagement with them, suggesting a directionality account. Cross-model transfer of persona effects is near-zero, so persona-alignment testing must be per-model. We propose $\Delta_{\text{floor}}$ as a deployment-time audit metric: measure it on a small persona panel before deploying persona customization.
>
---
#### [replaced 101] HumorGen: Cognitive Synergy for Humor Generation in Large Language Models via Persona-Based Distillation
- **分类: cs.CL**

- **简介: 该论文属于幽默生成任务，旨在解决LLM在幽默生成中的挑战。通过认知协同框架和角色化数据生成，提升模型幽默能力。**

- **链接: [https://arxiv.org/pdf/2604.09629](https://arxiv.org/pdf/2604.09629)**

> **作者:** Edward Ajayi; Prasenjit Mitra
>
> **摘要:** Humor generation poses a significant challenge for Large Language Models (LLMs), because their standard training objective (next-token prediction) inherently conflicts with the surprise and incongruity required for comedy. To bridge this gap, we introduce the Cognitive Synergy Framework, a methodology for generating highquality humor data inspired by psychological theories of humor. Utilizing a Mixtureof-Thought (MoT) approach, we deploy six cognitive personas (e.g., The Absurdist, The Cynic) to synthesize diverse comedic perspectives for a given prompt. This framework produces a theory-grounded dataset, which we use to fine-tune a 7B-parameter student model. We further evaluate two alignment strategies, Direct Preference Optimization (DPO) and an offline group-relative variant O-GRPO, finding that neither improves over SFT. However, our 7B HumorGen model variants significantly outperform larger instruction-tuned baselines and achieve top-tier open-weight performance while remaining competitive with frontier proprietary systems. These results suggest that cognitively driven data curation is more critical than alignment algorithms or model scale for humor generation.
>
---
#### [replaced 102] Beyond Transcripts: A Renewed Perspective on Audio Chaptering
- **分类: cs.SD; cs.CL**

- **简介: 该论文聚焦音频分段任务，解决音频章节划分中的文本依赖、ASR误差及评估方法问题。提出AudioSeg模型，对比不同方法，分析影响因素并建立新评估协议。**

- **链接: [https://arxiv.org/pdf/2602.08979](https://arxiv.org/pdf/2602.08979)**

> **作者:** Fabian Retkowski; Maike Züfle; Thai Binh Nguyen; Jan Niehues; Alexander Waibel
>
> **备注:** Accepted at ACL 2026 (Main Conference)
>
> **摘要:** Audio chaptering, the task of segmenting long-form audio into coherent sections, is increasingly important for navigating podcasts, lectures, and videos. Despite its relevance, research remains limited and text-based, leaving key questions unresolved about leveraging audio information, handling ASR errors, and transcript-free evaluation. We address these gaps through three contributions: (1) a systematic comparison between text-based models with acoustic features, a novel audio-only architecture (AudioSeg) operating on learned audio representations, and multimodal LLMs; (2) empirical analysis of factors affecting performance, including transcript quality, acoustic features, duration, and speaker composition; and (3) formalized evaluation protocols contrasting transcript-dependent text-space protocols with transcript-invariant time-space protocols. Our experiments on YTSeg reveal that AudioSeg substantially outperforms text-based approaches, pauses provide the largest acoustic gains, and MLLMs remain limited by context length and weak instruction following, yet MLLMs are promising on shorter audio.
>
---
#### [replaced 103] Steering Language Models Before They Speak: Logit-Level Interventions
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型控制任务，旨在提升生成内容的可控制性。提出SWAI方法，在不修改模型的情况下，通过语料统计直接干预logit空间，实现对可读性、礼貌性和毒性等特性的有效控制。**

- **链接: [https://arxiv.org/pdf/2601.10960](https://arxiv.org/pdf/2601.10960)**

> **作者:** Hyeseon An; Shinwoo Park; Hyundong Jin; Yo-Sub Han
>
> **备注:** preprint
>
> **摘要:** Controllable generation requires language models to realize output characteristics such as reading level, politeness, and toxicity. Existing steering methods are often indirect, require access to internal activations, or depend on auxiliary trained models. We propose SWAI, a training-free inference-time method that addresses these limitations by steering directly in logit space using corpus-derived token statistics. SWAI computes z-normalized one-vs-rest log-odds scores from labeled corpora and biases high-scoring tokens only within the model's top-K candidate set, allowing control to favor target-characteristic tokens while preserving contextually plausible choices. Across readability, politeness, and toxicity control, SWAI consistently improves over prompt-based and prior logit-level baselines without modifying model parameters, accessing internal layers, or training an auxiliary model. Selectivity and lookup-table ablations show that the gains come from target-specific statistical scores rather than generic logit perturbation. These results indicate that effective steering does not require learned controllers when the logit intervention is guided by target-specific statistics under high-probability candidates.
>
---
#### [replaced 104] OpenCompass: A Universal Evaluation Platform for Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出OpenCompass，一个通用的大语言模型评估平台，解决现有评估方法在多样性、标准不一和效率低下的问题，通过模块化设计实现高效、灵活的跨领域模型评估。**

- **链接: [https://arxiv.org/pdf/2605.19276](https://arxiv.org/pdf/2605.19276)**

> **作者:** Maosong Cao; Kai Chen; Haodong Duan; Yixiao Fang; Zhiwei Fei; Tong Gao; Ge Jiaye; Mo Li; Hongwei Liu; Junnan Liu; Yuan Liu; Chengqi Lyu; Han Lyu; Ningsheng Ma; Zerun Ma; Yu Sun; Zhiyong Wu; Linchen Xiao; Jun Xu; Haochen Ye; Zhaohui Yu; Yike Yuan; Songyang Zhang; Yufeng Zhao; Fengzhe Zhou; Peiheng Zhou; Dongsheng Zhu; Lin Zhu; Jingming Zhuo
>
> **摘要:** In recent years, the field of artificial intelligence has undergone a paradigm shift from task-specific small-scale models to general-purpose large language models (LLMs). With the rapid iteration of LLMs, objective, quantitative, and comprehensive evaluation of their capabilities has become a critical link in advancing technological development. Currently, the mainstream static benchmark dataset-based evaluation methods face challenges such as the diversity of task types, inconsistent evaluation criteria, and fragmentation of data and processing workflows, making it difficult to efficiently conduct cross-domain and large-scale model evaluation. To address the aforementioned issues, this paper proposes and open-sources OpenCompass, a one-stop, scalable, and high-concurrency-supported general-purpose LLM evaluation platform. Adhering to the design philosophy of modularization and component decoupling, the platform boasts three core advantages: high compatibility, flexibility, and high concurrency. The core architecture of OpenCompass comprises five key components: the Configuration System, Task Partitioning Module, Execution and Scheduling Module, Task Execution Unit, and Result Visualization Module. Its workflow provides rule-based, LLM-as-a-Judge, and cascaded evaluators to adapt to the requirements of different task scenarios. Supporting mainstream benchmark datasets across multiple domains, including knowledge, reasoning, computation, science, language, code, etc., the platform offers a unified and efficient LLM evaluation tool for both academia and industry, facilitating the accurate identification of strengths and weaknesses of LLMs as well as their subsequent optimization.
>
---
#### [replaced 105] When the Same Coefficients Reach Different Places: Asymmetric Realizability in Transplanting Tokenizers across Large Language Models
- **分类: cs.LG; cs.CL; cs.CR**

- **简介: 该论文研究跨模型分词器移植中的不对称可实现性问题，通过构造“破坏令牌”揭示潜在安全风险，属于模型安全与权重组合任务。**

- **链接: [https://arxiv.org/pdf/2601.00065](https://arxiv.org/pdf/2601.00065)**

> **作者:** Xiaoze Liu; Weichen Yu; Matt Fredrikson; Xiaoqian Wang; Jing Gao
>
> **摘要:** Tokenizer transplant in cross-vocabulary model composition reconstructs donor-only embedding rows as weighted combinations over shared lexical anchors and reuses those coefficients on the base. We identify a structural geometric property of this reconstruction: the same coefficient vector reaches different sets in the donor and base anchor spans, an \emph{asymmetric realizability} gap. Across 65 donor-base pairs under OMP, with cross-operator validation on CLP, WECHSEL, and FOCUS, we construct \textit{breaker tokens}: single coefficient vectors that remain statistically inert in the donor anchor span while producing a high-salience reconstruction in the base. The same Gemma-2-2B donor checkpoint admits this construction against 13 different downstream bases drawn from five model families. The planted direction passes weight-merging with a clean reference unchanged. In a deployer case study, standard LoRA fine-tuning suppresses the breaker primarily on prompts whose distribution matches the training corpus and is not a sufficient mitigation against this attack family in our setting. The tested spectral filters miss the asymmetry. We discuss potential misuse in the open-weight composition supply chain.
>
---
#### [replaced 106] Chinese sensorimotor and embodiment norms for 3,000 lexicalized concepts
- **分类: cs.CL**

- **简介: 该论文属于语言与认知研究任务，旨在构建中文概念的感知运动和具身规范数据库，解决机器系统如何通过语言获取具身体验知识的问题。工作包括收集3000词的多维评分，并验证其有效性及与语言表示的关联性。**

- **链接: [https://arxiv.org/pdf/2605.22616](https://arxiv.org/pdf/2605.22616)**

> **作者:** Jing Chen; Gábor Parti; Yin Zhong; Chu-Ren Huang; Marco Marelli
>
> **摘要:** Understanding how conceptual knowledge is grounded in bodily experience, and to what extent machine systems can acquire such knowledge without direct sensorimotor experience, are central questions in both cognitive science and embodied artificial intelligence research. Large-scale normative resources are essential for investigating these questions empirically, yet such resources remain sparse for non-Indo-European languages. We present a novel normative database for 3,000 lexicalized concepts in Mandarin Chinese, comprising 11-dimensional sensorimotor ratings and unidimensional embodiment ratings collected from 378 native Mandarin speakers. The ratings demonstrate high reliability and strong cross-norm validity with existing Chinese resources, each of which covers fewer words and a subset of the 11 sensorimotor dimensions. In a validation study, we tested new variables derived from a theoretically motivated metric, Perceptual Strength of Embodiment (PSE) (Huang et al., 2025), together with seven common composite variables, on lexical decision tasks. The results suggest that PSE-Sensorimotor and Minkowski-3 are the strongest composite predictors of lexical decision performance, capturing the facilitatory effects of sensorimotor information on lexical processing. A further exploratory study showed that sensorimotor ratings are substantially recoverable from purely linguistic representations using simple regression models (mean Spearman r = .62 across dimensions), though recovery varied markedly: visual and auditory dimensions yielded higher correspondence than chemosensory ones. Representational similarity analysis further showed that the relational geometry of the sensorimotor space is also partially recoverable (r = .540), consistent with the view that distributional language use encodes aspects of embodied conceptual structure.
>
---
#### [replaced 107] The Anatomy of Conversational Scams: A Topic-Based Red Teaming Analysis of Multi-Turn Interactions in LLMs
- **分类: cs.CL**

- **简介: 该论文研究多轮对话中的对抗性行为，通过模拟红队测试分析LLM的防御策略。任务为安全评估，解决对抗性对话动态分析问题，工作包括模型评估、策略标注与交互建模。**

- **链接: [https://arxiv.org/pdf/2601.03134](https://arxiv.org/pdf/2601.03134)**

> **作者:** Xiangzhe Yuan; Zhenhao Zhang; Haoming Tang; Siying Hu
>
> **摘要:** As LLMs gain persuasive capabilities through extended dialogues, they create new opportunities for studying adversarial conversational behavior in extended interaction settings that traditional single-turn safety evaluations fail to capture. We systematically study these interactional dynamics using a controlled LLM-to-LLM simulation framework for automated red-teaming across bilingual social engineering scenarios. Evaluating eight state-of-the-art models in English and Chinese, we analyze dialogue-level outcomes, annotate attacker and defender strategy families, and model interaction dynamics between them. Results show that multi-turn adversarial dialogues follow recurrent escalation patterns, while defensive responses frequently rely on verification, delay, and channel control. We further find statistically significant cross-model and cross-lingual differences in outcome distributions, and transition analysis reveals systematic structural variation in how defender strategies respond to attacker tactics across languages. These findings highlight the importance of studying interactional structure in multi-turn adversarial dialogue settings and demonstrate how controlled LLM-to-LLM simulations can support mechanistic analysis of adversarial conversational dynamics.
>
---
#### [replaced 108] HE-SNR: Uncovering Latent Logic via Entropy for Guiding Mid-Training on SWE-bench
- **分类: cs.LG; cs.CL; cs.SE**

- **简介: 该论文针对软件工程任务中大语言模型的中段训练问题，提出HE-SNR度量方法，以更有效地指导模型训练。**

- **链接: [https://arxiv.org/pdf/2601.20255](https://arxiv.org/pdf/2601.20255)**

> **作者:** Yueyang Wang; Jiawei Fu; Baolong Bi; Xili Wang; Xiaoqing Liu
>
> **备注:** Accepted at ICML 2026. 21 pages, 15 figures
>
> **摘要:** SWE-bench has emerged as the premier benchmark for evaluating Large Language Models on complex software engineering tasks. While these capabilities are fundamentally acquired during the mid-training phase and subsequently elicited during Supervised Fine-Tuning (SFT), there remains a critical deficit in metrics capable of guiding mid-training effectively. Standard metrics such as Perplexity (PPL) are compromised by the "Long-Context Tax" and exhibit weak correlation with downstream SWE performance. In this paper, we bridge this gap by first introducing a rigorous data filtering strategy. Crucially, we propose the Entropy Compression Hypothesis, redefining intelligence not by scalar Top-1 compression, but by the capacity to structure uncertainty into Entropy-Compressed States of low orders ("reasonable hesitation"). Grounded in this fine-grained entropy analysis, we formulate a novel metric, HE-SNR (High-Entropy Signal-to-Noise Ratio). We validate our approach on models with up to 560B parameters across different context windows (32K/128K). This work provides both the theoretical foundation and practical tools for optimizing the latent potential of LLMs in complex engineering domains.
>
---
#### [replaced 109] EVA-Bench: A New End-to-end Framework for Evaluating Voice Agents
- **分类: cs.SD; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出EVA-Bench，用于评估语音代理的性能。解决语音代理在真实对话生成和质量测量方面的评估难题，通过模拟对话和引入两个综合指标进行跨架构比较。**

- **链接: [https://arxiv.org/pdf/2605.13841](https://arxiv.org/pdf/2605.13841)**

> **作者:** Tara Bogavelli; Gabrielle Gauthier Melançon; Katrina Stankiewicz; Oluwanifemi Bamgbose; Fanny Riols; Hoang H. Nguyen; Raghav Mehndiratta; Lindsay Devon Brin; Joseph Marinier; Hari Subramani; Anil Madamala; Sridhar Krishna Nemala; Srinivas Sunkara
>
> **备注:** Work in progress
>
> **摘要:** Voice agents, artificial intelligence systems that conduct spoken conversations to complete tasks, are increasingly deployed across enterprise applications. However, no existing benchmark jointly addresses two core evaluation challenges: generating realistic simulated conversations, and measuring quality across the full scope of voice-specific failure modes. We present EVA-Bench, an end-to-end evaluation framework that addresses both. On the simulation side, EVA-Bench orchestrates bot-to-bot audio conversations over dynamic multi-turn dialogues, with automatic simulation validation that detects user simulator error and appropriately regenerates conversations before scoring. On the measurement side, EVA-Bench introduces two composite metrics: EVA-A (Accuracy), capturing task completion, faithfulness, and audio-level speech fidelity; and EVA-X (Experience), capturing conversation progression, spoken conciseness, and turn-taking timing. Both metrics apply to all major agent architectures, enabling direct cross-architecture comparison. EVA-Bench includes 213 scenarios across three enterprise domains, a controlled perturbation suite for accent and noise robustness, and pass@1, pass@k, pass^k measurements that distinguish peak from reliable capability. Across 12 systems spanning all three architectures, we find: (1) no system simultaneously exceeds 0.5 on both EVA-A pass@1 and EVA-X pass@1; (2) peak and reliable performance diverge substantially (median pass@k--pass^k gap of 0.44 on EVA-A); and (3) accent and noise perturbations expose substantial robustness gaps, with effects varying across architectures, systems, and metrics (mean $\Delta$ up to 0.314). We release the full framework, evaluation suite, and benchmark data under an open-source license.
>
---
#### [replaced 110] The Price Reversal Phenomenon: When Cheaper Reasoning Models Cost More
- **分类: cs.CL; cs.AI; cs.GT; cs.LG; cs.MA**

- **简介: 该论文研究API定价与实际推理成本的不匹配问题，属于模型成本分析任务。通过实验发现低价模型可能更贵，提出成本预测挑战。**

- **链接: [https://arxiv.org/pdf/2603.23971](https://arxiv.org/pdf/2603.23971)**

> **作者:** Lingjiao Chen; Chi Zhang; Yeye He; Ion Stoica; Matei Zaharia; James Zou
>
> **摘要:** Developers and consumers increasingly choose reasoning models (RMs) based on their listed API prices. However, how accurately do these prices reflect actual inference costs? We conduct the first systematic study of this question, evaluating 8 frontier RMs across 12 diverse tasks covering competition math, science QA, code generation, and multi-domain agents. We uncover the pricing reversal phenomenon: in 32% of model-pair comparisons, the model with a lower listed price actually incurs a higher total cost, with reversal magnitude reaching up to 28x. For example, Gemini 3 Flash's listed price is 80% cheaper than GPT-5.4's, yet its actual cost across all tasks is 38% higher. We build a formal cost attribution framework based on Shapley value, and leverage it to trace the dominating contributors to vast heterogeneity in thinking token consumption and number of interaction turns: on the same query, one model may use 900% more thinking tokens than another, or 10x more turns of environment interactions. We further show that per-query cost prediction is fundamentally difficult: repeated runs of the same query yield thinking token variation up to 9.7x, establishing an irreducible noise floor for any predictor. Thus, we propose cost distribution prediction as an open challenge. Our findings demonstrate that listed API pricing is an unreliable proxy for actual cost, calling for cost-aware model selection and transparent per-request cost monitoring.
>
---
#### [replaced 111] X-GS: An Extensible Framework for Perceiving and Thinking via 3D Gaussian Splatting
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出X-GS框架，解决3DGS应用中领域孤立问题，通过Perceiver和Thinker实现多模态协同，提升SLAM与视觉理解能力。**

- **链接: [https://arxiv.org/pdf/2603.09632](https://arxiv.org/pdf/2603.09632)**

> **作者:** Yueen Ma; Zenglin Xu; Irwin King
>
> **摘要:** 3D Gaussian Splatting (3DGS) has emerged as a powerful technique for novel view synthesis, subsequently extending into numerous spatial AI applications. However, most existing 3DGS methods operate in isolation, focusing on specific domains. In this paper, we introduce X-GS, an extensible framework consisting of two major components. The X-GS-\textit{Perceiver} unifies a broad range of 3DGS techniques to enable real-time online SLAM with semantic distillation. The X-GS-\textit{Thinker} accommodates multimodal models, enabling them to seamlessly interface with the \textit{Perceiver} to complete downstream tasks. In our implementation of X-GS, the \textit{Perceiver} leverages the latest vision foundation models to improve online SLAM performance and employs three key mechanisms to accelerate semantic distillation. The \textit{Thinker} can be built upon both contrastive and generative vision-language models and utilizes the \textit{Perceiver}'s semantic Gaussian splats to unlock capabilities such as 3D visual grounding and scene captioning. Experimental results on diverse benchmarks demonstrate the efficiency and newly unlocked multimodal capabilities of the X-GS framework.
>
---
#### [replaced 112] The Importance of Being Statistically Earnest: A Critical Re-evaluation of GSM-Symbolic
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，质疑GSM-Symbolic基准的统计结论。通过重新分析模型表现，指出其结论存在统计缺陷，并发现数据分布差异影响结果。**

- **链接: [https://arxiv.org/pdf/2605.28700](https://arxiv.org/pdf/2605.28700)**

> **作者:** Dominika Agnieszka Długosz; Arlindo Oliveira; Natalia Díaz-Rodríguez
>
> **备注:** 38 pages, 11 figures. Submitted to ACL ARR / EMNLP 2026
>
> **摘要:** The GSM-Symbolic benchmark (Mirzadeh et al., 2025) reported consistent performance drops across 25 Large Language Models (LLMs) when tested on template-generated variants of GSM8K problems, concluding that the models lack genuine reasoning capabilities. We argue that this conclusion rests on shaky statistical ground. Re-evaluating 20 open-weight models using Generalised Linear Mixed Models with per-question random effects, we find that only half exhibit statistically significant performance changes under the original prompt format. Moreover, we identify a previously unacknowledged factor: the main GSM-Symbolic dataset contains a systematically shifted distribution of larger integers in problem texts relative to GSM-Base (K-S statistic = 0.12, p < 0.001), contradicting the original authors' claims. Controlling for this large number effect accounts for significance in roughly half the remaining cases. Among models with statistically significant performance deltas, we identify distinct, model-specific failure profiles - including fragility of variable binding, arithmetic limitations, and dual-task interference - underscoring that blanket claims about LLM reasoning are both statistically premature and mechanistically misleading.
>
---
#### [replaced 113] SafeSearch: Automated Red-Teaming of LLM-Based Search Agents
- **分类: cs.AI; cs.CL; cs.CR**

- **简介: 该论文属于AI安全任务，旨在解决LLM搜索代理的安全性问题。通过构建SafeSearch框架，评估并揭示其潜在风险，提出改进方法。**

- **链接: [https://arxiv.org/pdf/2509.23694](https://arxiv.org/pdf/2509.23694)**

> **作者:** Jianshuo Dong; Sheng Guo; Hao Wang; Xun Chen; Zhuotao Liu; Tianwei Zhang; Ke Xu; Minlie Huang; Han Qiu
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** Search agents connect LLMs to the Internet, enabling them to access broader and more up-to-date information. However, this also introduces a new threat surface: unreliable search results can mislead agents into producing unsafe outputs. Real-world incidents and our two in-the-wild observations show that such failures can occur in practice. To study this threat systematically, we propose SafeSearch, an automated red-teaming framework that is scalable, cost-efficient, and lightweight, enabling sandboxed safety evaluation of search agents. Using this, we generate 300 test cases spanning five risk categories (e.g., misinformation and prompt injection) and evaluate three search agent scaffolds across 17 representative LLMs. Our results reveal substantial vulnerabilities in LLM-based search agents, with the highest ASR reaching 90.5% for GPT-4.1-mini in a search-workflow setting. Moreover, we find that common defenses, such as reminder prompting, offer limited protection. Overall, SafeSearch provides a practical way to measure and improve the safety of LLM-based search agents.
>
---
#### [replaced 114] A Survey on Recent Advances in Conversational Data Generation
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于对话数据生成任务，旨在解决对话数据稀缺问题。通过综述生成方法、框架及评估，推动 conversational system 的发展。**

- **链接: [https://arxiv.org/pdf/2405.13003](https://arxiv.org/pdf/2405.13003)**

> **作者:** Heydar Soudani; Roxana Petcu; Evangelos Kanoulas; Faegheh Hasibi
>
> **摘要:** Recent advancements in conversational systems have significantly enhanced human-machine interactions across various domains. However, training these systems is challenging due to the scarcity of specialized dialogue data. Traditionally, conversational datasets were created through crowdsourcing, but this method has proven costly, limited in scale, and labor-intensive. As a solution, the development of synthetic dialogue data has emerged, utilizing techniques to augment existing datasets or convert textual resources into conversational formats, providing a more efficient and scalable approach to dataset creation. In this survey, we offer a systematic and comprehensive review of multi-turn conversational data generation, focusing on three types of dialogue systems: open domain, task-oriented, and information-seeking. We categorize the existing research based on key components like seed data creation, utterance generation, and quality filtering methods, and introduce a general framework that outlines the main principles of conversation data generation systems. Additionally, we examine the evaluation metrics and methods for assessing synthetic conversational data, address current challenges in the field, and explore potential directions for future research. Our goal is to accelerate progress for researchers and practitioners by presenting an overview of state-of-the-art methods and highlighting opportunities to further research in this area.
>
---
#### [replaced 115] Empathic Prompting: Non-Verbal Context Integration for Multimodal LLM Conversations
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文提出Empathic Prompting框架，用于增强多模态人机交互中的情感理解。任务是将非语言情感信息融入LLM对话，解决传统接口缺乏情感感知的问题。工作包括系统设计与初步评估。**

- **链接: [https://arxiv.org/pdf/2510.20743](https://arxiv.org/pdf/2510.20743)**

> **作者:** Lorenzo Stacchio; Andrea Ubaldi; Alessandro Galdelli; Maurizio Mauri; Emanuele Frontoni; Andrea Gaggioli
>
> **摘要:** We present Empathic Prompting, a novel framework for multimodal human-AI interaction that enriches Large Language Model (LLM) conversations with implicit non-verbal context. The system integrates a commercial facial expression recognition service to capture users' emotional cues and embeds them as contextual signals during prompting. Unlike traditional multimodal interfaces, empathic prompting requires no explicit user control; instead, it unobtrusively augments textual input with affective information for conversational and smoothness alignment. The architecture is modular and scalable, allowing integration of additional non-verbal modules. We describe the system design, implemented through a locally deployed DeepSeek instance, and report a preliminary service and usability evaluation (N=5). Results show consistent integration of non-verbal input into coherent LLM outputs, with participants highlighting conversational fluidity. Beyond this proof of concept, empathic prompting points to applications in chatbot-mediated communication, particularly in domains like healthcare or education, where users' emotional signals are critical yet often opaque in verbal exchanges.
>
---
#### [replaced 116] TANDEM: Temporal-Aware Neural Detection for Multimodal Hate Speech
- **分类: cs.AI; cs.CL; cs.MM; cs.SI**

- **简介: 该论文提出TANDEM框架，解决多模态仇恨言论检测任务中的可解释性问题，通过结构化推理实现精准时间定位和目标识别。**

- **链接: [https://arxiv.org/pdf/2601.11178](https://arxiv.org/pdf/2601.11178)**

> **作者:** Girish A. Koushik; Helen Treharne; Diptesh Kanojia
>
> **备注:** Under review at ICWSM 2027
>
> **摘要:** Social media platforms are increasingly dominated by long-form multimodal content, where harmful narratives are constructed through a complex interplay of audio, visual, and textual cues. While automated systems can flag hate speech with high accuracy, they often function as "black boxes" that fail to provide the granular, interpretable evidence, such as precise timestamps and target identities, required for effective human-in-the-loop moderation. In this work, we introduce TANDEM, a unified framework that transforms audio-visual hate detection from a binary classification task into a structured reasoning problem. Our approach employs a novel tandem reinforcement learning strategy where vision-language and audio-language models optimize each other through self-constrained cross-modal context, stabilizing reasoning over extended temporal sequences without requiring dense frame-level supervision. Experiments across three benchmark datasets demonstrate that TANDEM significantly outperforms zero-shot and context-augmented baselines, achieving 0.73 F1 in target identification on HateMM (a 30% improvement over state-of-the-art) while maintaining precise temporal grounding. We further observe that while binary detection is robust, differentiating between offensive and hateful content remains challenging in multi-class settings due to inherent label ambiguity and dataset imbalance. More broadly, our findings suggest that structured, interpretable alignment is achievable even in complex multimodal settings, offering a blueprint for the next generation of transparent and actionable online safety moderation tools.
>
---
#### [replaced 117] Differential syntactic and semantic encoding in LLMs
- **分类: cs.CL; cs.AI; cs.LG; physics.comp-ph**

- **简介: 该论文研究LLM中语法和语义信息的编码方式，通过分析DeepSeek-V3的层表示，发现语法和语义可部分线性分离，揭示其编码差异。**

- **链接: [https://arxiv.org/pdf/2601.04765](https://arxiv.org/pdf/2601.04765)**

> **作者:** Santiago Acevedo; Alessandro Laio; Marco Baroni
>
> **备注:** Published as conference paper at ICML 2026
>
> **摘要:** We study how syntactic and semantic information is encoded in inner layer representations of Large Language Models (LLMs), focusing on the very large DeepSeek-V3. We find that, by averaging hidden-representation vectors of sentences sharing syntactic structure or meaning, we obtain vectors that capture a significant proportion of the syntactic and semantic information contained in the representations. In particular, subtracting these syntactic and semantic ``centroids'' from sentence vectors strongly affects their similarity with syntactically and semantically matched sentences, respectively, suggesting that syntax and semantics are, at least partially, linearly encoded. We also find that the cross-layer encoding profiles of syntax and semantics are different, and that the two signals can to some extent be decoupled, suggesting differential encoding of these two types of linguistic information in LLM representations.
>
---
