# 自然语言处理 cs.CL

- **最新发布 75 篇**

- **更新 47 篇**

## 最新发布

#### [new 001] Dual Perspectives in Emotion Attribution: A Generator-Interpreter Framework for Cross-Cultural Analysis of Emotion in LLMs
- **分类: cs.CL**

- **简介: 该论文属于情感归属任务，旨在解决跨文化情感理解中的偏差问题。通过提出生成器-解释器框架，分析不同文化背景下的情感表达与解读差异。**

- **链接: [https://arxiv.org/pdf/2603.29077](https://arxiv.org/pdf/2603.29077)**

> **作者:** Aizirek Turdubaeva; Uichin Lee
>
> **摘要:** Large language models (LLMs) are increasingly used in cross-cultural systems to understand and adapt to human emotions, which are shaped by cultural norms of expression and interpretation. However, prior work on emotion attribution has focused mainly on interpretation, overlooking the cultural background of emotion generators. This assumption of universality neglects variation in how emotions are expressed and perceived across nations. To address this gap, we propose a Generator-Interpreter framework that captures dual perspectives of emotion attribution by considering both expression and interpretation. We systematically evaluate six LLMs on an emotion attribution task using data from 15 countries. Our analysis reveals that performance variations depend on the emotion type and cultural context. Generator-interpreter alignment effects are present; the generator's country of origin has a stronger impact on performance. We call for culturally sensitive emotion modeling in LLM-based systems to improve robustness and fairness in emotion understanding across diverse cultural contexts.
>
---
#### [new 002] Kwame 2.0: Human-in-the-Loop Generative AI Teaching Assistant for Large Scale Online Coding Education in Africa
- **分类: cs.CL; cs.CY; cs.HC**

- **简介: 该论文属于教育技术任务，旨在解决大规模在线编程教育中支持不足的问题。通过构建Kwame 2.0系统，结合AI与人工协作，提升学习支持的效率和准确性。**

- **链接: [https://arxiv.org/pdf/2603.29159](https://arxiv.org/pdf/2603.29159)**

> **作者:** George Boateng; Samuel Boateng; Victor Kumbol
>
> **备注:** 8 pages, Accepted at the 27th International Conference on Artificial Intelligence in Education (AIED 2026)
>
> **摘要:** Providing timely and accurate learning support in large-scale online coding courses is challenging, particularly in resource-constrained contexts. We present Kwame 2.0, a bilingual (English-French) generative AI teaching assistant built using retrieval-augmented generation and deployed in a human-in-the-loop forum within SuaCode, an introductory mobile-based coding course for learners across Africa. Kwame 2.0 retrieves relevant course materials and generates context-aware responses while encouraging human oversight and community participation. We deployed the system in a 15-month longitudinal study spanning 15 cohorts with 3,717 enrollments across 35 African countries. Evaluation using community feedback and expert ratings shows that Kwame 2.0 provided high-quality and timely support, achieving high accuracy on curriculum-related questions, while human facilitators and peers effectively mitigated errors, particularly for administrative queries. Our findings demonstrate that human-in-the-loop generative AI systems can combine the scalability and speed of AI with the reliability of human support, offering an effective approach to learning assistance for underrepresented populations in resource-constrained settings at scale.
>
---
#### [new 003] OptiMer: Optimal Distribution Vector Merging Is Better than Data Mixing for Continual Pre-Training
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于持续预训练任务，解决数据混合比例优化问题。提出OptiMer方法，通过分布向量后优化获得最佳组合权重，提升模型效果并降低调参成本。**

- **链接: [https://arxiv.org/pdf/2603.28858](https://arxiv.org/pdf/2603.28858)**

> **作者:** Haiyue Song; Masao Utiyama
>
> **备注:** Preprint, 20 pages, 10 tables, 12 figures
>
> **摘要:** Continual pre-training is widely used to adapt LLMs to target languages and domains, yet the mixture ratio of training data remains a sensitive hyperparameter that is expensive to tune: they must be fixed before training begins, and a suboptimal choice can waste weeks of compute. In this work, we propose OptiMer, which decouples ratio selection from training: we train one CPT model per dataset, extract each model's distribution vector, which represents the parameter shift induced by that dataset, and search for optimal composition weights post-hoc via Bayesian optimization. Experiments on Gemma 3 27B across languages (Japanese, Chinese) and domains (Math, Code) show that OptiMer consistently outperforms data mixture and model averaging baselines with 15-35 times lower search cost. Key findings reveal that 1) the optimized weights can be interpreted as data mixture ratios, and retraining with these ratios improves data mixture CPT, and 2) the same vector pool can be re-optimized for a given objective without any retraining, producing target-tailored models on demand. Our work establishes that data mixture ratio selection, traditionally a pre-training decision, can be reformulated as a post-hoc optimization over distribution vectors, offering a more flexible paradigm for continual pre-training.
>
---
#### [new 004] Learning Diagnostic Reasoning for Decision Support in Toxicology
- **分类: cs.CL**

- **简介: 该论文属于医疗诊断任务，旨在解决急性多药物中毒中的诊断难题。通过强化学习优化语言模型，融合非结构化与结构化数据，提升中毒识别准确率。**

- **链接: [https://arxiv.org/pdf/2603.29608](https://arxiv.org/pdf/2603.29608)**

> **作者:** Nico Oberländer; David Bani-Harouni; Tobias Zellner; Nassir Navab; Florian Eyer; Matthias Keicher
>
> **摘要:** Acute poly-substance intoxication requires rapid, life-saving decisions under substantial uncertainty, as clinicians must rely on incomplete ingestion details and nonspecific symptoms. Effective diagnostic reasoning in this chaotic environment requires fusing unstructured, non-medical narratives (e.g. paramedic scene descriptions and unreliable patient self-reports or known histories), with structured medical data like vital signs. While Large Language Models (LLMs) show potential for processing such heterogeneous inputs, they struggle in this setting, often underperforming simple baselines that rely solely on patient histories. To address this, we present DeToxR (Decision-support for Toxicology with Reasoning), the first adaptation of Reinforcement Learning (RL) to emergency toxicology. We design a robust data-fusion engine for multi-label prediction across 14 substance classes based on an LLM finetuned with Group Relative Policy Optimization (GRPO). We optimize the model's reasoning directly using a clinical performance reward. By formulating a multi-label agreement metric as the reward signal, the model is explicitly penalized for missing co-ingested substances and hallucinating absent poisons. Our model significantly outperforms its unadapted base LLM counterpart and supervised baselines. Furthermore, in a clinical validation study, the model indicates a clinical advantage by outperforming an expert toxicologist in identifying the correct poisons (Micro-F1: 0.644 vs. 0.473). These results demonstrate the potential of RL-aligned LLMs to synthesize unstructured pre-clinical narratives and structured medical data for decision support in high-stakes environments.
>
---
#### [new 005] Covertly improving intelligibility with data-driven adaptations of speech timing
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于语音处理任务，旨在提升语音可懂性。通过分析语音速率对理解的影响，提出数据驱动的语音调整方法，显著提高不同听众在困难条件下的理解能力。**

- **链接: [https://arxiv.org/pdf/2603.30032](https://arxiv.org/pdf/2603.30032)**

> **作者:** Paige Tuttösí; Angelica Lim; H. Henny Yeung; Yue Wang; Jean-Julien Aucouturier
>
> **摘要:** Human talkers often address listeners with language-comprehension challenges, such as hard-of-hearing or non-native adults, by globally slowing down their speech. However, it remains unclear whether this strategy actually makes speech more intelligible. Here, we take advantage of recent advancements in machine-generated speech allowing more precise control of speech rate in order to systematically examine how targeted speech-rate adjustments may improve comprehension. We first use reverse-correlation experiments to show that the temporal influence of speech rate prior to a target vowel contrast (ex. the tense-lax distinction) in fact manifests in a scissor-like pattern, with opposite effects in early versus late context windows; this pattern is remarkably stable both within individuals and across native L1-English listeners and L2-English listeners with French, Mandarin, and Japanese L1s. Second, we show that this speech rate structure not only facilitates L2 listeners' comprehension of the target vowel contrast, but that native listeners also rely on this pattern in challenging acoustic conditions. Finally, we build a data-driven text-to-speech algorithm that replicates this temporal structure on novel speech sequences. Across a variety of sentences and vowel contrasts, listeners remained unaware that such targeted slowing improved word comprehension. Strikingly, participants instead judged the common strategy of global slowing as clearer, even though it actually increased comprehension errors. Together, these results show that targeted adjustments to speech rate significantly aid intelligibility under challenging conditions, while often going unnoticed. More generally, this paper provides a data-driven methodology to improve the accessibility of machine-generated speech which can be extended to other aspects of speech comprehension and a wide variety of listeners and environments.
>
---
#### [new 006] The Thiomi Dataset: A Large-Scale Multimodal Corpus for Low-Resource African Languages
- **分类: cs.CL; cs.LG**

- **简介: 该论文介绍Thiomi数据集，用于低资源非洲语言的多模态研究。解决非洲语言技术资源不足的问题，收集并验证了多语言文本和语音数据，建立了ASR、MT和TTS的基准模型。**

- **链接: [https://arxiv.org/pdf/2603.29244](https://arxiv.org/pdf/2603.29244)**

> **作者:** Hillary Mutisya; John Mugane; Gavin Nyamboga; Brian Chege; Maryruth Gathoni
>
> **摘要:** We present the Thiomi Dataset, a large-scale multimodal corpus spanning ten African languages across four language families: Swahili, Kikuyu, Kamba, Kimeru, Luo, Maasai, Kipsigis, Somali (East Africa); Wolof (West Africa); and Fulani (West/Central Africa). The dataset contains over 601,000 approved sentence-level text annotations and over 385,000 audio recordings across nine languages, collected through a dedicated community data collection platform involving over 100 contributors. The Thiomi platform collected data for nine languages; Swahili data was supplemented with existing Common Voice recordings. A multi-tier quality assurance pipeline achieves 86-100% text approval rates for the six primary languages. To validate the dataset's utility, we train and evaluate ASR, MT, and TTS models, establishing baselines across all ten languages. Our best ASR system achieves 3.24% WER on Swahili (Common Voice), reducing prior academic SOTA from 8.3% to 3.24% (5.1 percentage point absolute, 61% relative reduction), and 4.3% WER on Somali. The dataset will be published on HuggingFace. We describe the collection platform, quality assurance workflows, and baseline experiments, and discuss implications for African language technology infrastructure.
>
---
#### [new 007] ContextClaim: A Context-Driven Paradigm for Verifiable Claim Detection
- **分类: cs.CL**

- **简介: 该论文属于事实核查任务中的可验证声明检测。针对现有方法仅依赖声明文本的不足，提出ContextClaim框架，通过引入上下文信息提升检测效果。**

- **链接: [https://arxiv.org/pdf/2603.30025](https://arxiv.org/pdf/2603.30025)**

> **作者:** Yufeng Li; Rrubaa Panchendrarajan; Arkaitz Zubiaga
>
> **摘要:** Verifiable claim detection asks whether a claim expresses a factual statement that can, in principle, be assessed against external evidence. As an early filtering stage in automated fact-checking, it plays an important role in reducing the burden on downstream verification components. However, existing approaches to claim detection, whether based on check-worthiness or verifiability, rely solely on the claim text itself. This is a notable limitation for verifiable claim detection in particular, where determining whether a claim is checkable may benefit from knowing what entities and events it refers to and whether relevant information exists to support verification. Inspired by the established role of evidence retrieval in later-stage claim verification, we propose Context-Driven Claim Detection (ContextClaim), a paradigm that advances retrieval to the detection stage. ContextClaim extracts entity mentions from the input claim, retrieves relevant information from Wikipedia as a structured knowledge source, and employs large language models to produce concise contextual summaries for downstream classification. We evaluate ContextClaim on two datasets covering different topics and text genres, the CheckThat! 2022 COVID-19 Twitter dataset and the PoliClaim political debate dataset, across encoder-only and decoder-only models under fine-tuning, zero-shot, and few-shot settings. Results show that context augmentation can improve verifiable claim detection, although its effectiveness varies across domains, model architectures, and learning settings. Through component analysis, human evaluation, and error analysis, we further examine when and why the retrieved context contributes to more reliable verifiability judgments.
>
---
#### [new 008] When Can We Trust LLM Graders? Calibrating Confidence for Automated Assessment
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于自动化评估任务，旨在解决如何判断LLM评分器的可靠性问题。通过比较三种置信度估计方法，发现自报置信度效果最佳，为可靠评分提供实用方案。**

- **链接: [https://arxiv.org/pdf/2603.29559](https://arxiv.org/pdf/2603.29559)**

> **作者:** Robinson Ferrer; Damla Turgut; Zhongzhou Chen; Shashank Sonkar
>
> **摘要:** Large Language Models (LLMs) show promise for automated grading, but their outputs can be unreliable. Rather than improving grading accuracy directly, we address a complementary problem: \textit{predicting when an LLM grader is likely to be correct}. This enables selective automation where high-confidence predictions are processed automatically while uncertain cases are flagged for human review. We compare three confidence estimation methods (self-reported confidence, self-consistency voting, and token probability) across seven LLMs of varying scale (4B to 120B parameters) on three educational datasets: RiceChem (long-answer chemistry), SciEntsBank, and Beetle (short-answer science). Our experiments reveal that self-reported confidence consistently achieves the best calibration across all conditions (avg ECE 0.166 vs 0.229 for self-consistency). Surprisingly, self-consistency remains 38\% worse despite requiring 5$\times$ the inference cost. Larger models exhibit substantially better calibration though gains vary by dataset and method (e.g., a 28\% ECE reduction for self-reported), with GPT-OSS-120B achieving the best calibration (avg ECE 0.100) and strong discrimination (avg AUC 0.668). We also observe that confidence is strongly top-skewed across methods, creating a ``confidence floor'' that practitioners must account for when setting thresholds. These findings suggest that simply asking LLMs to report their confidence provides a practical approach for identifying reliable grading predictions. Code is available \href{this https URL}{here}.
>
---
#### [new 009] Known Intents, New Combinations: Clause-Factorized Decoding for Compositional Multi-Intent Detection
- **分类: cs.CL**

- **简介: 该论文属于多意图检测任务，解决模型在未见意图组合上的泛化能力问题。提出ClauseCompose方法，通过分解意图进行解码，提升组合泛化性能。**

- **链接: [https://arxiv.org/pdf/2603.28929](https://arxiv.org/pdf/2603.28929)**

> **作者:** Abhilash Nandy
>
> **备注:** 6 pages, 3 tables
>
> **摘要:** Multi-intent detection papers usually ask whether a model can recover multiple intents from one utterance. We ask a harder and, for deployment, more useful question: can it recover new combinations of familiar intents? Existing benchmarks only weakly test this, because train and test often share the same broad co-occurrence patterns. We introduce CoMIX-Shift, a controlled benchmark built to stress compositional generalization in multi-intent detection through held-out intent pairs, discourse-pattern shift, longer and noisier wrappers, held-out clause templates, and zero-shot triples. We also present ClauseCompose, a lightweight decoder trained only on singleton intents, and compare it to whole-utterance baselines including a fine-tuned tiny BERT model. Across three random seeds, ClauseCompose reaches 95.7 exact match on unseen intent pairs, 93.9 on discourse-shifted pairs, 62.5 on longer/noisier pairs, 49.8 on held-out templates, and 91.1 on unseen triples. WholeMultiLabel reaches 81.4, 55.7, 18.8, 15.5, and 0.0; the BERT baseline reaches 91.5, 77.6, 48.9, 11.0, and 0.0. We also add a 240-example manually authored SNIPS-style compositional set with five held-out pairs; there, ClauseCompose reaches 97.5 exact match on unseen pairs and 86.7 under connector shift, compared with 41.3 and 10.4 for WholeMultiLabel. The results suggest that multi-intent detection needs more compositional evaluation, and that simple factorization goes surprisingly far once evaluation asks for it.
>
---
#### [new 010] MemRerank: Preference Memory for Personalized Product Reranking
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于个性化推荐任务，解决长历史记录在LLM购物代理中无效的问题。提出MemRerank框架，通过提炼用户偏好记忆提升产品重排序效果。**

- **链接: [https://arxiv.org/pdf/2603.29247](https://arxiv.org/pdf/2603.29247)**

> **作者:** Zhiyuan Peng; Xuyang Wu; Huaixiao Tou; Yi Fang; Yi Gong
>
> **摘要:** LLM-based shopping agents increasingly rely on long purchase histories and multi-turn interactions for personalization, yet naively appending raw history to prompts is often ineffective due to noise, length, and relevance mismatch. We propose MemRerank, a preference memory framework that distills user purchase history into concise, query-independent signals for personalized product reranking. To study this problem, we build an end-to-end benchmark and evaluation framework centered on an LLM-based \textbf{1-in-5} selection task, which measures both memory quality and downstream reranking utility. We further train the memory extractor with reinforcement learning (RL), using downstream reranking performance as supervision. Experiments with two LLM-based rerankers show that MemRerank consistently outperforms no-memory, raw-history, and off-the-shelf memory baselines, yielding up to \textbf{+10.61} absolute points in 1-in-5 accuracy. These results suggest that explicit preference memory is a practical and effective building block for personalization in agentic e-commerce systems.
>
---
#### [new 011] Beyond Idealized Patients: Evaluating LLMs under Challenging Patient Behaviors in Medical Consultations
- **分类: cs.CL**

- **简介: 该论文属于医疗对话任务，旨在解决LLMs在面对复杂患者行为时的安全性问题。通过构建基准数据集，评估模型在信息矛盾等挑战场景下的表现，并探索改进策略。**

- **链接: [https://arxiv.org/pdf/2603.29373](https://arxiv.org/pdf/2603.29373)**

> **作者:** Yahan Li; Xinyi Jie; Wanjia Ruan; Xubei Zhang; Huaijie Zhu; Yicheng Gao; Chaohao Du; Ruishan Liu
>
> **摘要:** Large language models (LLMs) are increasingly used for medical consultation and health information support. In this high-stakes setting, safety depends not only on medical knowledge, but also on how models respond when patient inputs are unclear, inconsistent, or misleading. However, most existing medical LLM evaluations assume idealized and well-posed patient questions, which limits their realism. In this paper, we study challenging patient behaviors that commonly arise in real medical consultations and complicate safe clinical reasoning. We define four clinically grounded categories of such behaviors: information contradiction, factual inaccuracy, self-diagnosis, and care resistance. For each behavior, we specify concrete failure criteria that capture unsafe responses. Building on four existing medical dialogue datasets, we introduce CPB-Bench (Challenging Patient Behaviors Benchmark), a bilingual (English and Chinese) benchmark of 692 multi-turn dialogues annotated with these behaviors. We evaluate a range of open- and closed-source LLMs on their responses to challenging patient utterances. While models perform well overall, we identify consistent, behavior-specific failure patterns, with particular difficulty in handling contradictory or medically implausible patient information. We also study four intervention strategies and find that they yield inconsistent improvements and can introduce unnecessary corrections. We release the dataset and code.
>
---
#### [new 012] CounselReflect: A Toolkit for Auditing Mental-Health Dialogues
- **分类: cs.CL**

- **简介: 该论文提出CounselReflect，一个用于审计心理健康对话的工具包，解决如何透明评估对话质量与风险的问题。通过多维度报告和多种评估信号，提升审计的可理解性与可信度。**

- **链接: [https://arxiv.org/pdf/2603.29429](https://arxiv.org/pdf/2603.29429)**

> **作者:** Yahan Li; Chaohao Du; Zeyang Li; Christopher Chun Kuizon; Shupeng Cheng; Angel Hsing-Chi Hwang; Adam C. Frank; Ruishan Liu
>
> **摘要:** Mental-health support is increasingly mediated by conversational systems (e.g., LLM-based tools), but users often lack structured ways to audit the quality and potential risks of the support they receive. We introduce CounselReflect, an end-to-end toolkit for auditing mental-health support dialogues. Rather than producing a single opaque quality score, CounselReflect provides structured, multi-dimensional reports with session-level summaries, turn-level scores, and evidence-linked excerpts to support transparent inspection. The system integrates two families of evaluation signals: (i) 12 model-based metrics produced by task-specific predictors, and (ii) rubric-based metrics that extend coverage via a literature-derived library (69 metrics) and user-defined custom metrics, operationalized with configurable LLM judges. CounselReflect is available as a web application, browser extension, and command-line interface (CLI), enabling use in real-time settings as well as at scale. Human evaluation includes a user study with 20 participants and an expert review with 6 mental-health professionals, suggesting that CounselReflect supports understandable, usable, and trustworthy auditing. A demo video and full source code are also provided.
>
---
#### [new 013] CrossTrace: A Cross-Domain Dataset of Grounded Scientific Reasoning Traces for Hypothesis Generation
- **分类: cs.CL**

- **简介: 该论文提出CrossTrace数据集，解决科学假设生成中缺乏跨领域推理轨迹的问题。通过构建多领域推理链，提升模型生成假设的准确性与合理性。**

- **链接: [https://arxiv.org/pdf/2603.28924](https://arxiv.org/pdf/2603.28924)**

> **作者:** Andrew Bouras; OMS-II Research Fellow
>
> **备注:** 14 pages, 1 figure, 8 tables. Dataset and code available at this https URL
>
> **摘要:** Scientific hypothesis generation is a critical bottleneck in accelerating research, yet existing datasets for training and evaluating hypothesis-generating models are limited to single domains and lack explicit reasoning traces connecting prior knowledge to novel contributions. I introduce CrossTrace, a dataset of 1,389 grounded scientific reasoning traces spanning biomedical research (518), AI/ML (605), and cross-domain work (266). Each trace captures the structured reasoning chain from established knowledge through intermediate logical steps to a novel hypothesis, with every step grounded in source paper text. I define an Input/Trace/Output schema that extends the Bit-Flip-Spark framework of HypoGen with step-level verification, a taxonomy of eight discovery patterns, and multi-domain coverage. Fine-tuning Qwen2.5-7B-Instruct on CrossTrace via QLoRA yields substantial improvements over the untuned baseline: IAScore rises from 0.828 to 0.968 (GPT-4o judge) and from 0.716 to 0.888 (Claude Opus 4.5), structural compliance improves from 0% to 100%, and spark cosine similarity increases from 0.221 to 0.620. Balanced cross-domain training (biomedical + AI/ML + CS) outperforms single-domain training, providing evidence that scientific reasoning patterns transfer across disciplines. Human validation of 150 stratified records confirms 99.7% step-level grounding accuracy and a 0.0% fabrication rate. To my knowledge, CrossTrace is the first large-scale, cross-domain dataset with step-level grounded reasoning traces for hypothesis generation, and my results demonstrate that such traces are an effective training signal whose benefits are at least partially domain-general.
>
---
#### [new 014] Human-Like Lifelong Memory: A Neuroscience-Grounded Architecture for Infinite Interaction
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出一种基于神经科学的终身记忆架构，解决大模型缺乏持久结构化记忆的问题。通过生物启发机制实现高效记忆编码与检索，提升长期交互能力。**

- **链接: [https://arxiv.org/pdf/2603.29023](https://arxiv.org/pdf/2603.29023)**

> **作者:** Diego C. Lerma-Torres
>
> **备注:** 14 pages, 1 figure. Accepted at the MemAgents Workshop, ICLR 2026
>
> **摘要:** Large language models lack persistent, structured memory for long-term interaction and context-sensitive retrieval. Expanding context windows does not solve this: recent evidence shows that context length alone degrades reasoning by up to 85% - even with perfect retrieval. We propose a bio-inspired memory framework grounded in complementary learning systems theory, cognitive behavioral therapy's belief hierarchy, dual-process cognition, and fuzzy-trace theory, organized around three principles: (1) Memory has valence, not just content - pre-computed emotional-associative summaries (valence vectors) organized in an emergent belief hierarchy inspired by Beck's cognitive model enable instant orientation before deliberation; (2) Retrieval defaults to System 1 with System 2 escalation - automatic spreading activation and passive priming as default, with deliberate retrieval only when needed, and graded epistemic states that address hallucination structurally; and (3) Encoding is active, present, and feedback-dependent - a thalamic gateway tags and routes information between stores, while the executive forms gists through curiosity-driven investigation, not passive exposure. Seven functional properties specify what any implementation must satisfy. Over time, the system converges toward System 1 processing - the computational analog of clinical expertise - producing interactions that become cheaper, not more expensive, with experience.
>
---
#### [new 015] SiPaKosa: A Comprehensive Corpus of Canonical and Classical Buddhist Texts in Sinhala and Pali
- **分类: cs.CL**

- **简介: 该论文介绍SiPaKosa语料库，用于佛教文本研究。任务是构建多语言语料库，解决数据不足问题，通过OCR和网络抓取完成，并评估语言模型性能。**

- **链接: [https://arxiv.org/pdf/2603.29221](https://arxiv.org/pdf/2603.29221)**

> **作者:** Ranidu Gurusinghe; Nevidu Jayatilleke
>
> **备注:** 17 pages, 5 figures, 5 tables, Accepted paper at the 2nd Workshop on Challenges in Processing South Asian Languages (CHiPSAL) @ LREC 2026
>
> **摘要:** SiPaKosa is a comprehensive corpus of Sinhala and Pali doctrinal texts comprising approximately 786K sentences and 9.25M words, incorporating 16 copyright-cleared historical Buddhist documents alongside the complete web-scraped Tripitaka canonical texts. The corpus was created through high-quality OCR using Google Document AI on historical manuscripts, combined with systematic web scraping of canonical repositories, followed by rigorous quality control and metadata annotation. The corpus is organised into language-specific subcorpora: Sinhala and Mixed Sinhala-Pali. We evaluate the performance of language models using ten pretrained models, with perplexity scores ranging from 1.09 to 189.67 on our corpus. This analysis shows that proprietary models significantly outperform open-source alternatives by factors of three to six times. This corpus supports the pretraining of domain-adapted language models, facilitates historical language analysis, and aids in the development of information retrieval systems for Buddhist scholarship while preserving Sinhala cultural heritage.
>
---
#### [new 016] Is my model perplexed for the right reason? Contrasting LLMs' Benchmark Behavior with Token-Level Perplexity
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，旨在检验大语言模型是否基于正确机制作出判断。通过分析词级困惑度，揭示模型可能依赖非语言线索的启发式方法。**

- **链接: [https://arxiv.org/pdf/2603.29396](https://arxiv.org/pdf/2603.29396)**

> **作者:** Zoë Prins; Samuele Punzo; Frank Wildenburg; Giovanni Cinà; Sandro Pezzelle
>
> **摘要:** Standard evaluations of Large language models (LLMs) focus on task performance, offering limited insight into whether correct behavior reflects appropriate underlying mechanisms and risking confirmation bias. We introduce a simple, principled interpretability framework based on token-level perplexity to test whether models rely on linguistically relevant cues. By comparing perplexity distributions over minimal sentence pairs differing in one or a few `pivotal' tokens, our method enables precise, hypothesis-driven analysis without relying on unstable feature-attribution techniques. Experiments on controlled linguistic benchmarks with several open-weight LLMs show that, while linguistically important tokens influence model behavior, they never fully explain perplexity shifts, revealing that models rely on heuristics other than the expected linguistic ones.
>
---
#### [new 017] CADEL: A Corpus of Administrative Web Documents for Japanese Entity Linking
- **分类: cs.CL**

- **简介: 该论文属于实体链接任务，旨在解决日语文本实体链接资源不足的问题。构建了一个标注语料库，用于训练和评估日语实体链接系统。**

- **链接: [https://arxiv.org/pdf/2603.29336](https://arxiv.org/pdf/2603.29336)**

> **作者:** Shohei Higashiyama; Masao Ideuchi; Masao Utiyama
>
> **摘要:** Entity linking is the task of associating linguistic expressions with entries in a knowledge base that represent real-world entities and concepts. Language resources for this task have primarily been developed for English, and the resources available for evaluating Japanese systems remain limited. In this study, we develop a corpus design policy for the entity linking task and construct an annotated corpus for training and evaluating Japanese entity linking systems, with rich coverage of linguistic expressions referring to entities that are specific to Japan. Evaluation of inter-annotator agreement confirms the high consistency of the annotations in the corpus, and a preliminary experiment on entity disambiguation based on string matching suggests that the corpus contains a substantial number of non-trivial cases, supporting its potential usefulness as an evaluation benchmark.
>
---
#### [new 018] Can LLM Agents Identify Spoken Dialects like a Linguist?
- **分类: cs.CL**

- **简介: 该论文属于语音方言分类任务，旨在解决标注数据稀缺的问题。通过结合ASR转录与语言资源，评估LLM和人类语言学家的分类表现。**

- **链接: [https://arxiv.org/pdf/2603.29541](https://arxiv.org/pdf/2603.29541)**

> **作者:** Tobias Bystrich; Lukas Hamm; Maria Hassan; Lea Fischbach; Lucie Flek; Akbar Karimi
>
> **备注:** Accepted to DialRes Workshop @ LREC 2026
>
> **摘要:** Due to the scarcity of labeled dialectal speech, audio dialect classification is a challenging task for most languages, including Swiss German. In this work, we explore the ability of large language models (LLMs) as agents in understanding the dialects and whether they can show comparable performance to models such as HuBERT in dialect classification. In addition, we provide an LLM baseline and a human linguist one. Our approach uses phonetic transcriptions produced by ASR systems and combines them with linguistic resources such as dialect feature maps, vowel history, and rules. Our findings indicate that, when linguistic information is provided, the LLM predictions improve. The human baseline shows that automatically generated transcriptions can be beneficial for such classifications, but also presents opportunities for improvement.
>
---
#### [new 019] FLEURS-Kobani: Extending the FLEURS Dataset for Northern Kurdish
- **分类: cs.CL**

- **简介: 该论文提出FLEURS-Kobani数据集，解决北库尔德语语音任务缺乏基准的问题。扩展了FLEURS以支持ASR和语音翻译，包含5162条语音，用于模型训练与评估。**

- **链接: [https://arxiv.org/pdf/2603.29892](https://arxiv.org/pdf/2603.29892)**

> **作者:** Daban Q. Jaff; Mohammad Mohammadamini
>
> **摘要:** FLEURS offers n-way parallel speech for 100+ languages, but Northern Kurdish is not one of them, which limits benchmarking for automatic speech recognition and speech translation tasks in this language. We present FLEURS-Kobani, a Northern Kurdish (ISO 639-3 KMR) spoken extension of the FLEURS benchmark. The FLEURS-Kobani dataset consists of 5,162 validated utterances, totaling 18 hours and 24 minutes. The data were recorded by 31 native speakers. It extends benchmark coverage to an under-resourced Kurdish variety. As baselines, we fine-tuned Whisper v3-large for ASR and E2E S2TT. A two-stage fine-tuning strategy (Common Voice to FLEURS-Kobani) yields the best ASR performance (WER 28.11, CER 9.84 on test). For E2E S2TT (KMR to EN), Whisper achieves 8.68 BLEU on test; we additionally report pivot-derived targets and a cascaded S2TT setup. FLEURS-Kobani provides the first public Northern Kurdish benchmark for evaluation of ASR, S2TT and S2ST tasks. The dataset is publicly released for research use under a CC BY 4.0 license.
>
---
#### [new 020] Developing a Guideline for the Labovian-Structural Analysis of Oral Narratives in Japanese
- **分类: cs.CL**

- **简介: 该论文属于叙事分析任务，旨在解决日语口语叙事的Labovian结构分析问题。作者提出系统性指南，改进了日语语料的分句规则和分类框架。**

- **链接: [https://arxiv.org/pdf/2603.29347](https://arxiv.org/pdf/2603.29347)**

> **作者:** Amane Watahiki; Tomoki Doi; Akari Kikuchi; Hiroshi Ohata; Yuki I. Nakata; Takuya Niikawa; Taiga Shinozaki; Hitomi Yanaka
>
> **备注:** Accepted at The Fifteenth biennial Language Resources and Evaluation Conference (LREC) 2026
>
> **摘要:** Narrative analysis is a cornerstone of qualitative research. One leading approach is the Labovian model, but its application is labor-intensive, requiring a holistic, recursive interpretive process that moves back and forth between individual parts of the transcript and the transcript as a whole. Existing Labovian datasets are available only in English, which differs markedly from Japanese in terms of grammar and discourse conventions. To address this gap, we introduce the first systematic guidelines for Labovian narrative analysis of Japanese narrative data. Our guidelines retain all six Labovian categories and extend the framework by providing explicit rules for clause segmentation tailored to Japanese constructions. In addition, our guidelines cover a broader range of clause types and narrative types. Using these guidelines, annotators achieved high agreement in clause segmentation (Fleiss' kappa = 0.80) and moderate agreement in two structural classification tasks (Krippendorff's alpha = 0.41 and 0.45, respectively), one of which is slightly higher than that found in prior work despite the use of finer-grained distinctions. This paper describes the Labovian model, the proposed guidelines, the annotation process, and their utility. It concludes by discussing the challenges encountered during the annotation process and the prospects for developing a larger dataset for structural narrative analysis in Japanese qualitative research.
>
---
#### [new 021] Distilling Human-Aligned Privacy Sensitivity Assessment from Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于隐私评估任务，旨在解决文本数据隐私评价的准确性与效率问题。通过知识蒸馏将大模型的隐私评估能力迁移至轻量模型，提升实际应用可行性。**

- **链接: [https://arxiv.org/pdf/2603.29497](https://arxiv.org/pdf/2603.29497)**

> **作者:** Gabriel Loiseau; Damien Sileo; Damien Riquet; Maxime Meyer; Marc Tommasi
>
> **备注:** Accepted to the LREC CALD-pseudo 2026 Workshop
>
> **摘要:** Accurate privacy evaluation of textual data remains a critical challenge in privacy-preserving natural language processing. Recent work has shown that large language models (LLMs) can serve as reliable privacy evaluators, achieving strong agreement with human judgments; however, their computational cost and impracticality for processing sensitive data at scale limit real-world deployment. We address this gap by distilling the privacy assessment capabilities of Mistral Large 3 (675B) into lightweight encoder models with as few as 150M parameters. Leveraging a large-scale dataset of privacy-annotated texts spanning 10 diverse domains, we train efficient classifiers that preserve strong agreement with human annotations while dramatically reducing computational requirements. We validate our approach on human-annotated test data and demonstrate its practical utility as an evaluation metric for de-identification systems.
>
---
#### [new 022] Baby Scale: Investigating Models Trained on Individual Children's Language Input
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于语言模型研究任务，旨在探讨儿童语言数据对模型训练的影响。解决儿童与成人数据差距问题，通过分析儿童语言数据集，评估模型性能及语言学习关联性。**

- **链接: [https://arxiv.org/pdf/2603.29522](https://arxiv.org/pdf/2603.29522)**

> **作者:** Steven Y. Feng; Alvin W.M. Tan; Michael C. Frank
>
> **备注:** Code and data at this https URL
>
> **摘要:** Modern language models (LMs) must be trained on many orders of magnitude more words of training data than human children receive before they begin to produce useful behavior. Assessing the nature and origins of this "data gap" requires benchmarking LMs on human-scale datasets to understand how linguistic knowledge emerges from children's natural training data. Using transcripts from the BabyView dataset (videos from children ages 6-36 months), we investigate (1) scaling performance at child-scale data regimes, (2) variability in model performance across datasets from different children's experiences and linguistic predictors of dataset quality, and (3) relationships between model and child language learning outcomes. LMs trained on child data show acceptable scaling for grammar tasks, but lower scaling on semantic and world knowledge tasks than models trained on synthetic data; we also observe substantial variability on data from different children. Beyond dataset size, performance is most associated with a combination of distributional and interactional linguistic features, broadly consistent with what makes high-quality input for child language development. Finally, model likelihoods for individual words correlate with children's learning of those words, suggesting that properties of child-directed input may influence both model learning and human language development. Overall, understanding what properties make language data efficient for learning can enable more powerful small-scale language models while also shedding light on human language acquisition.
>
---
#### [new 023] L-ReLF: A Framework for Lexical Dataset Creation
- **分类: cs.CL**

- **简介: 该论文提出L-ReLF框架，解决低资源语言词汇数据构建问题。通过系统化流程生成结构化数据，支持知识平台与NLP应用。**

- **链接: [https://arxiv.org/pdf/2603.29346](https://arxiv.org/pdf/2603.29346)**

> **作者:** Anass Sedrati; Mounir Afifi; Reda Benkhadra
>
> **备注:** Accepted to the 2026 International Conference on Natural Language Processing (ICNLP). 6 pages, 1 figure
>
> **摘要:** This paper introduces the L-ReLF (Low-Resource Lexical Framework), a novel, reproducible methodology for creating high-quality, structured lexical datasets for underserved languages. The lack of standardized terminology, exemplified by Moroccan Darija, poses a critical barrier to knowledge equity in platforms like Wikipedia, often forcing editors to rely on inconsistent, ad-hoc methods to create new words in their language. Our research details the technical pipeline developed to overcome these challenges. We systematically address the difficulties of working with low-resource data, including source identification, utilizing Optical Character Recognition (OCR) despite its bias towards Modern Standard Arabic, and rigorous post-processing to correct errors and standardize the data model. The resulting structured dataset is fully compatible with Wikidata Lexemes, serving as a vital technical resource. The L-ReLF methodology is designed for generalizability, offering other language communities a clear path to build foundational lexical data for downstream NLP applications, such as Machine Translation and morphological analysis.
>
---
#### [new 024] Calibrated Confidence Expression for Radiology Report Generation
- **分类: cs.CL**

- **简介: 该论文属于医学影像报告生成任务，旨在解决模型过自信导致的临床风险问题。通过引入ConRad框架，实现报告生成时的校准置信度评估，提升模型可信度与临床适用性。**

- **链接: [https://arxiv.org/pdf/2603.29492](https://arxiv.org/pdf/2603.29492)**

> **作者:** David Bani-Harouni; Chantal Pellegrini; Julian Lüers; Su Hwan Kim; Markus Baalmann; Benedikt Wiestler; Rickmer Braren; Nassir Navab; Matthias Keicher
>
> **摘要:** Safe deployment of Large Vision-Language Models (LVLMs) in radiology report generation requires not only accurate predictions but also clinically interpretable indicators of when outputs should be thoroughly reviewed, enabling selective radiologist verification and reducing the risk of hallucinated findings influencing clinical decisions. One intuitive approach to this is verbalized confidence, where the model explicitly states its certainty. However, current state-of-the-art language models are often overconfident, and research on calibration in multimodal settings such as radiology report generation is limited. To address this gap, we introduce ConRad (Confidence Calibration for Radiology Reports), a reinforcement learning framework for fine-tuning medical LVLMs to produce calibrated verbalized confidence estimates alongside radiology reports. We study two settings: a single report-level confidence score and a sentence-level variant assigning a confidence to each claim. Both are trained using the GRPO algorithm with reward functions based on the logarithmic scoring rule, which incentivizes truthful self-assessment by penalizing miscalibration and guarantees optimal calibration under reward maximization. Experimentally, ConRad substantially improves calibration and outperforms competing methods. In a clinical evaluation we show that ConRad's report level scores are well aligned with clinicians' judgment. By highlighting full reports or low-confidence statements for targeted review, ConRad can support safer clinical integration of AI-assistance for report generation.
>
---
#### [new 025] M-MiniGPT4: Multilingual VLLM Alignment via Translated Data
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出M-MiniGPT4，解决多语言视觉语言理解任务，通过混合原生与翻译数据提升模型多语言能力。**

- **链接: [https://arxiv.org/pdf/2603.29467](https://arxiv.org/pdf/2603.29467)**

> **作者:** Seung Hun Han; Youssef Mohamed; Mohamed Elhoseiny
>
> **备注:** 6 pages, ACL 2026, Proceedings of the 7th Workshop on African Natural Language Processing (AfricaNLP 2026)
>
> **摘要:** This paper presents a Multilingual Vision Large Language Model, named M-MiniGPT4. Our model exhibits strong vision-language understanding (VLU) capabilities across 11 languages. We utilize a mixture of native multilingual and translated data to push the multilingual VLU performance of the MiniGPT4 architecture. In addition, we propose a multilingual alignment training stage that uses parallel text corpora to further enhance the multilingual capabilities of our model. M-MiniGPT4 achieves 36% accuracy on the multilingual MMMU benchmark, outperforming state-of-the-art models in the same weight class, including foundation models released after the majority of this work was completed. We open-source our models, code, and translated datasets to facilitate future research in low-resource and multilingual settings.
>
---
#### [new 026] Towards Empowering Consumers through Sentence-level Readability Scoring in German ESG Reports
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的可读性评估任务，旨在提升德国ESG报告的可读性。通过扩展数据集并评估不同方法的预测效果，以帮助非专业读者更好地理解报告内容。**

- **链接: [https://arxiv.org/pdf/2603.29861](https://arxiv.org/pdf/2603.29861)**

> **作者:** Benjamin Josef Schüßler; Jakob Prange
>
> **备注:** accepted to NLP4Ecology workshop at LREC 2026
>
> **摘要:** With the ever-growing urgency of sustainability in the economy and society, and the massive stream of information that comes with it, consumers need reliable access to that information. To address this need, companies began publishing so called Environmental, Social, and Governance (ESG) reports, both voluntarily and forced by law. To serve the public, these reports must be addressed not only to financial experts but also to non-expert audiences. But are they written clearly enough? In this work, we extend an existing sentence-level dataset of German ESG reports with crowdsourced readability annotations. We find that, in general, native speakers perceive sentences in ESG reports as easy to read, but also that readability is subjective. We apply various readability scoring methods and evaluate them regarding their prediction error and correlation with human rankings. Our analysis shows that, while LLM prompting has potential for distinguishing clear from hard-to-read sentences, a small finetuned transformer predicts human readability with the lowest error. Averaging predictions of multiple models can slightly improve the performance at the cost of slower inference.
>
---
#### [new 027] An Empirical Recipe for Universal Phone Recognition
- **分类: cs.CL**

- **简介: 该论文属于语音识别任务，旨在提升多语言和低资源场景下的音素识别性能。通过优化数据规模、模型架构和训练目标，提出PhoneticXEUS模型，在多语言和口音英语中取得最佳效果。**

- **链接: [https://arxiv.org/pdf/2603.29042](https://arxiv.org/pdf/2603.29042)**

> **作者:** Shikhar Bharadwaj; Chin-Jou Li; Kwanghee Choi; Eunjung Yeo; William Chen; Shinji Watanabe; David R. Mortensen
>
> **备注:** Submitted to Interspeech 2026. Code: this https URL
>
> **摘要:** Phone recognition (PR) is a key enabler of multilingual and low-resource speech processing tasks, yet robust performance remains elusive. Highly performant English-focused models do not generalize across languages, while multilingual models underutilize pretrained representations. It also remains unclear how data scale, architecture, and training objective contribute to multilingual PR. We present PhoneticXEUS -- trained on large-scale multilingual data and achieving state-of-the-art performance on both multilingual (17.7% PFER) and accented English speech (10.6% PFER). Through controlled ablations with evaluations across 100+ languages under a unified scheme, we empirically establish our training recipe and quantify the impact of SSL representations, data scale, and loss objectives. In addition, we analyze error patterns across language families, accented speech, and articulatory features. All data and code are released openly.
>
---
#### [new 028] Agenda-based Narrative Extraction: Steering Pathfinding Algorithms with Large Language Models
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于叙事生成任务，解决传统方法在连贯性、互动性和多线支持间的平衡问题。通过引入大语言模型，实现基于议程的叙事引导，提升故事线与用户目标的一致性。**

- **链接: [https://arxiv.org/pdf/2603.29661](https://arxiv.org/pdf/2603.29661)**

> **作者:** Brian Felipe Keith-Norambuena; Carolina Inés Rojas-Córdova; Claudio Juvenal Meneses-Villegas; Elizabeth Johanna Lam-Esquenazi; Angélica María Flores-Bustos; Ignacio Alejandro Molina-Villablanca; Joshua Emanuel Leyton-Vallejos
>
> **备注:** Text2Story Workshop 2026 at ECIR 2026
>
> **摘要:** Existing narrative extraction methods face a trade-off between coherence, interactivity, and multi-storyline support. Narrative Maps supports rich interaction and generates multiple storylines as a byproduct of its coverage constraints, though this comes at the cost of individual path coherence. Narrative Trails achieves high coherence through maximum capacity path optimization but provides no mechanism for user guidance or multiple perspectives. We introduce agenda-based narrative extraction, a method that bridges this gap by integrating large language models into the Narrative Trails pathfinding process to steer storyline construction toward user-specified perspectives. Our approach uses an LLM at each step to rank candidate documents based on their alignment with a given agenda while maintaining narrative coherence. Running the algorithm with different agendas yields different storylines through the same corpus. We evaluated our approach on a news article corpus using LLM judges with Claude Opus 4.5 and GPT 5.1, measuring both coherence and agenda alignment across 64 endpoint pairs and 6 agendas. LLM-driven steering achieves 9.9% higher alignment than keyword matching on semantic agendas (p=0.017), with 13.3% improvement on \textit{Regime Crackdown} specifically (p=0.037), while keyword matching remains competitive on agendas with literal keyword overlap. The coherence cost is minimal: LLM steering reduces coherence by only 2.2% compared to the agenda-agnostic baseline. Counter-agendas that contradict the source material score uniformly low (2.2-2.5) across all methods, confirming that steering cannot fabricate unsupported narratives.
>
---
#### [new 029] Open Machine Translation for Esperanto
- **分类: cs.CL**

- **简介: 该论文属于机器翻译任务，旨在评估开源MT系统在埃斯帕诺语中的表现。通过对比不同模型，探索其翻译质量并发布相关资源。**

- **链接: [https://arxiv.org/pdf/2603.29345](https://arxiv.org/pdf/2603.29345)**

> **作者:** Ona de Gibert; Lluís de Gibert
>
> **备注:** Accepted to SIGUL 2026
>
> **摘要:** Esperanto is a widespread constructed language, known for its regular grammar and productive word formation. Besides having substantial resources available thanks to its online community, it remains relatively underexplored in the context of modern machine translation (MT) approaches. In this work, we present the first comprehensive evaluation of open-source MT systems for Esperanto, comparing rule-based systems, encoder-decoder models, and LLMs across model sizes. We evaluate translation quality across six language directions involving English, Spanish, Catalan, and Esperanto using multiple automatic metrics as well as human evaluation. Our results show that the NLLB family achieves the best performance in all language pairs, followed closely by our trained compact models and a fine-tuned general-purpose LLM. Human evaluation confirms this trend, with NLLB translations preferred in approximately half of the comparisons, although noticeable errors remain. In line with Esperanto's tradition of openness and international collaboration, we release our code and best-performing models publicly.
>
---
#### [new 030] Theory of Mind and Self-Attributions of Mentality are Dissociable in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI安全研究，探讨安全微调对LLM心智归因能力的影响。研究发现，抑制自我心智归因不会损害理论心智能力，但会降低对非人类动物的心智归因。**

- **链接: [https://arxiv.org/pdf/2603.28925](https://arxiv.org/pdf/2603.28925)**

> **作者:** Junsol Kim; Winnie Street; Roberta Rocca; Daine M. Korngiebel; Adam Waytz; James Evans; Geoff Keeling
>
> **摘要:** Safety fine-tuning in Large Language Models (LLMs) seeks to suppress potentially harmful forms of mind-attribution such as models asserting their own consciousness or claiming to experience emotions. We investigate whether suppressing mind-attribution tendencies degrades intimately related socio-cognitive abilities such as Theory of Mind (ToM). Through safety ablation and mechanistic analyses of representational similarity, we demonstrate that LLM attributions of mind to themselves and to technological artefacts are behaviorally and mechanistically dissociable from ToM capabilities. Nevertheless, safety fine-tuned models under-attribute mind to non-human animals relative to human baselines and are less likely to exhibit spiritual belief, suppressing widely shared perspectives regarding the distribution and nature of non-human minds.
>
---
#### [new 031] The Model Says Walk: How Surface Heuristics Override Implicit Constraints in LLM Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型在表面线索与隐含约束冲突时的推理缺陷，属于自然语言处理任务。通过分析和实验，揭示了模型对约束推理的不足，并提出基准测试以评估改进效果。**

- **链接: [https://arxiv.org/pdf/2603.29025](https://arxiv.org/pdf/2603.29025)**

> **作者:** Yubo Li; Lu Zhang; Tianchong Jiang; Ramayya Krishnan; Rema Padman
>
> **摘要:** Large language models systematically fail when a salient surface cue conflicts with an unstated feasibility constraint. We study this through a diagnose-measure-bridge-treat framework. Causal-behavioral analysis of the ``car wash problem'' across six models reveals approximately context-independent sigmoid heuristics: the distance cue exerts 8.7 to 38 times more influence than the goal, and token-level attribution shows patterns more consistent with keyword associations than compositional inference. The Heuristic Override Benchmark (HOB) -- 500 instances spanning 4 heuristic by 5 constraint families with minimal pairs and explicitness gradients -- demonstrates generality across 14 models: under strict evaluation (10/10 correct), no model exceeds 75%, and presence constraints are hardest (44%). A minimal hint (e.g., emphasizing the key object) recovers +15 pp on average, suggesting the failure lies in constraint inference rather than missing knowledge; 12/14 models perform worse when the constraint is removed (up to -39 pp), revealing conservative bias. Parametric probes confirm that the sigmoid pattern generalizes to cost, efficiency, and semantic-similarity heuristics; goal-decomposition prompting recovers +6 to 9 pp by forcing models to enumerate preconditions before answering. Together, these results characterize heuristic override as a systematic reasoning vulnerability and provide a benchmark for measuring progress toward resolving it.
>
---
#### [new 032] Impact of enriched meaning representations for language generation in dialogue tasks: A comprehensive exploration of the relevance of tasks, corpora and metrics
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究对话生成任务，探讨增强语义表示对生成质量的影响。通过对比不同数据集和指标，验证了增强输入在复杂任务和小数据中的有效性。**

- **链接: [https://arxiv.org/pdf/2603.29518](https://arxiv.org/pdf/2603.29518)**

> **作者:** Alain Vázquez; Maria Inés Torres
>
> **摘要:** Conversational systems should generate diverse language forms to interact fluently and accurately with users. In this context, Natural Language Generation (NLG) engines convert Meaning Representations (MRs) into sentences, directly influencing user perception. These MRs usually encode the communicative function (e.g., inform, request, confirm) via DAs and enumerate the semantic content with slot-value pairs. In this work, our objective is to analyse whether providing a task demonstrator to the generator enhances the generations of a fine-tuned model. This demonstrator is an MR-sentence pair extracted from the original dataset that enriches the input at training and inference time. The analysis involves five metrics that focus on different linguistic aspects, and four datasets that differ in multiple features, such as domain, size, lexicon, MR variability, and acquisition process. To the best of our knowledge, this is the first study on dialogue NLG implementing a comparative analysis of the impact of MRs on generation quality across domains, corpus characteristics, and the metrics used to evaluate these generations. Our key insight is that the proposed enriched inputs are effective for complex tasks and small datasets with high variability in MRs and sentences. They are also beneficial in zero-shot settings for any domain. Moreover, the analysis of the metrics shows that semantic metrics capture generation quality more accurately than lexical metrics. In addition, among these semantic metrics, those trained with human ratings can detect omissions and other subtle semantic issues that embedding-based metrics often miss. Finally, the evolution of the metric scores and the excellent results for Slot Accuracy and Dialogue Act Accuracy demonstrate that the generative models present fast adaptability to different tasks and robustness at semantic and communicative intention levels.
>
---
#### [new 033] ENEIDE: A High Quality Silver Standard Dataset for Named Entity Recognition and Linking in Historical Italian
- **分类: cs.CL**

- **简介: 该论文提出ENEIDE，一个用于历史意大利语命名实体识别与链接的银标准数据集，解决多领域、跨时期的实体识别问题。通过半自动标注方法构建数据集，并进行基准实验验证其挑战性。**

- **链接: [https://arxiv.org/pdf/2603.29801](https://arxiv.org/pdf/2603.29801)**

> **作者:** Cristian Santini; Sebastian Barzaghi; Paolo Sernani; Emanuele Frontoni; Laura Melosi; Mehwish Alam
>
> **摘要:** This paper introduces ENEIDE (Extracting Named Entities from Italian Digital Editions), a silver standard dataset for Named Entity Recognition and Linking (NERL) in historical Italian texts. The corpus comprises 2,111 documents with over 8,000 entity annotations semi-automatically extracted from two scholarly digital editions: Digital Zibaldone, the philosophical diary of the Italian poet Giacomo Leopardi (1798--1837), and Aldo Moro Digitale, the complete works of the Italian politician Aldo Moro (1916--1978). Annotations cover multiple entity types (person, location, organization, literary work) linked to Wikidata identifiers, including NIL entities that cannot be mapped to the knowledge graph. To the best of our knowledge, ENEIDE represents the first multi-domain, publicly available NERL dataset for historical Italian with training, development, and test splits. We present a methodology for semi-automatic annotations extraction from manually curated scholarly digital editions, including quality control and annotation enhancement procedures. Baseline experiments using state-of-the-art models demonstrate the dataset's challenge for NERL and the gap between zero-shot approaches and fine-tuned models. The dataset's diachronic coverage spanning two centuries makes it particularly suitable for temporal entity disambiguation and cross-domain evaluation. ENEIDE is released under a CC BY-NC-SA 4.0 license.
>
---
#### [new 034] Near-Miss: Latent Policy Failure Detection in Agentic Workflows
- **分类: cs.CL**

- **简介: 该论文属于智能体系统评估任务，旨在解决政策违规检测中的隐性失败问题。通过分析代理决策过程，检测其是否遵循政策。**

- **链接: [https://arxiv.org/pdf/2603.29665](https://arxiv.org/pdf/2603.29665)**

> **作者:** Ella Rabinovich; David Boaz; Naama Zwerdling; Ateret Anaby-Tavor
>
> **摘要:** Agentic systems for business process automation often require compliance with policies governing conditional updates to the system state. Evaluation of policy adherence in LLM-based agentic workflows is typically performed by comparing the final system state against a predefined ground truth. While this approach detects explicit policy violations, it may overlook a more subtle class of issues in which agents bypass required policy checks, yet reach a correct outcome due to favorable circumstances. We refer to such cases as $\textit{near-misses}$ or $\textit{latent failures}$. In this work, we introduce a novel metric for detecting latent policy failures in agent conversations traces. Building on the ToolGuard framework, which converts natural-language policies into executable guard code, our method analyzes agent trajectories to determine whether agent's tool-calling decisions where sufficiently informed. We evaluate our approach on the $\tau^2$-verified Airlines benchmark across several contemporary open and proprietary LLMs acting as agents. Our results show that latent failures occur in 8-17% of trajectories involving mutating tool calls, even when the final outcome matches the expected ground-truth state. These findings reveal a blind spot in current evaluation methodologies and highlight the need for metrics that assess not only final outcomes but also the decision process leading to them.
>
---
#### [new 035] SyriSign: A Parallel Corpus for Arabic Text to Syrian Arabic Sign Language Translation
- **分类: cs.CL; cs.AI; cs.CV; cs.HC**

- **简介: 论文介绍SyriSign，一个用于阿拉伯语到叙利亚阿拉伯手语翻译的并行语料库。旨在解决聋哑群体沟通障碍问题，通过构建数据集并测试多种深度学习模型进行翻译任务。**

- **链接: [https://arxiv.org/pdf/2603.29219](https://arxiv.org/pdf/2603.29219)**

> **作者:** Mohammad Amer Khalil; Raghad Nahas; Ahmad Nassar; Khloud Al Jallad
>
> **摘要:** Sign language is the primary approach of communication for the Deaf and Hard-of-Hearing (DHH) community. While there are numerous benchmarks for high-resource sign languages, low-resource languages like Arabic remain underrepresented. Currently, there is no publicly available dataset for Syrian Arabic Sign Language (SyArSL). To overcome this gap, we introduce SyriSign, a dataset comprising 1500 video samples across 150 unique lexical signs, designed for text-to-SyArSL translation tasks. This work aims to reduce communication barriers in Syria, as most news are delivered in spoken or written Arabic, which is often inaccessible to the deaf community. We evaluated SyriSign using three deep learning architectures: MotionCLIP for semantic motion generation, T2M-GPT for text-conditioned motion synthesis, and SignCLIP for bilingual embedding alignment. Experimental results indicate that while generative approaches show strong potential for sign representation, the limited dataset size constrains generalization performance. We will release SyriSign publicly, hoping it serves as an initial benchmark.
>
---
#### [new 036] Concept Training for Human-Aligned Language Models
- **分类: cs.CL**

- **简介: 该论文属于语言模型训练任务，旨在提升模型与人类语义理解的一致性。通过概念级监督替代传统单词预测，增强语义对齐效果。**

- **链接: [https://arxiv.org/pdf/2603.29123](https://arxiv.org/pdf/2603.29123)**

> **作者:** Christine Zhang; Dan Jurafsky; Chen Shani
>
> **摘要:** The next-token prediction (NTP) objective trains language models to predict a single continuation token at each step. In natural language, however, a prefix can be continued in many valid ways, and even similar meanings may differ in surface form. For example, the sentence ``this website is safe to \underline{browse}'' could plausibly continue with words such as browse, search, visit, surf, or navigate. While standard NTP training treats these alternatives as mutually exclusive targets, we explore a framework that instead predicts concepts, approximated as sets of semantically related tokens. We show that models trained with concept supervision exhibit stronger alignment with human semantic similarity judgments on multiple lexical benchmarks. These gains are accompanied by lower perplexity on semantically meaningful words (definition in Section 3.1), and a modest increase in global token-level perplexity, reflecting a tradeoff between standard NTP optimization and concept-level supervision. Our results suggest that concept-level objectives can improve semantic alignment while maintaining competitive language modeling performance.
>
---
#### [new 037] LLM Probe: Evaluating LLMs for Low-Resource Languages
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决低资源语言中大语言模型评估不足的问题。提出LLM Probe框架，评估模型在词汇对齐、词性识别等语言能力的表现。**

- **链接: [https://arxiv.org/pdf/2603.29517](https://arxiv.org/pdf/2603.29517)**

> **作者:** Hailay Kidu Teklehaymanot; Gebrearegawi Gebremariam; Wolfgang Nejdl
>
> **备注:** 11 pages, 6 tables
>
> **摘要:** Despite rapid advances in large language models (LLMs), their linguistic abilities in low-resource and morphologically rich languages are still not well understood due to limited annotated resources and the absence of standardized evaluation frameworks. This paper presents LLM Probe, a lexicon-based assessment framework designed to systematically evaluate the linguistic skills of LLMs in low-resource language environments. The framework analyzes models across four areas of language understanding: lexical alignment, part-of-speech recognition, morphosyntactic probing, and translation accuracy. To illustrate the framework, we create a manually annotated benchmark dataset using a low-resource Semitic language as a case study. The dataset comprises bilingual lexicons with linguistic annotations, including part-of-speech tags, grammatical gender, and morphosyntactic features, which demonstrate high inter-annotator agreement to ensure reliable annotations. We test a variety of models, including causal language models and sequence-to-sequence architectures. The results reveal notable differences in performance across various linguistic tasks: sequence-to-sequence models generally excel in morphosyntactic analysis and translation quality, whereas causal models demonstrate strong performance in lexical alignment but exhibit weaker translation accuracy. Our results emphasize the need for linguistically grounded evaluation to better understand LLM limitations in low-resource settings. We release LLM Probe and the accompanying benchmark dataset as open-source tools to promote reproducible benchmarking and to support the development of more inclusive multilingual language technologies.
>
---
#### [new 038] Structural Feature Engineering for Generative Engine Optimization: How Content Structure Shapes Citation Behavior
- **分类: cs.CL; cs.HC; cs.IR**

- **简介: 该论文属于生成式引擎优化任务，旨在解决内容可见性问题。通过结构特征工程提升引用行为，提出GEO-SFE框架，优化内容结构以提高引用率和质量。**

- **链接: [https://arxiv.org/pdf/2603.29979](https://arxiv.org/pdf/2603.29979)**

> **作者:** Junwei Yu; Mufeng Yang; Yepeng Ding; Hiroyuki Sato
>
> **备注:** 12 pages, 5 figures. This paper proposes GEO-SFE, a structural feature engineering framework for generative engine optimization
>
> **摘要:** The proliferation of AI-powered search engines has shifted information discovery from traditional link-based retrieval to direct answer generation with selective source citation, creating new challenges for content visibility. While existing Generative Engine Optimization (GEO) approaches focus primarily on semantic content modification, the role of structural features in influencing citation behavior remains underexplored. In this paper, we propose GEO-SFE, a systematic framework for structural feature engineering in generative engine optimization. Our approach decomposes content structure into three hierarchical levels: macro-structure (document architecture), meso-structure (information chunking), and micro-structure (visual emphasis), and models their impact on citation probability across different generative engine architectures. We develop architecture-aware optimization strategies and predictive models that preserve semantic integrity while improving structural effectiveness. Experimental evaluation across six mainstream generative engines demonstrates consistent improvements in citation rate (17.3 percent) and subjective quality (18.5 percent), validating the effectiveness and generalizability of the proposed framework. This work establishes structural optimization as a foundational component of GEO, providing a data-driven methodology for enhancing content visibility in LLM-powered information ecosystems.
>
---
#### [new 039] From Consensus to Split Decisions: ABC-Stratified Sentiment in Holocaust Oral Histories
- **分类: cs.CL**

- **简介: 该论文属于情感分析任务，旨在解决历史叙述中情感分类的领域迁移问题。通过多模型对比和稳定性分类，分析情感模型在大语料上的分歧情况。**

- **链接: [https://arxiv.org/pdf/2603.28913](https://arxiv.org/pdf/2603.28913)**

> **作者:** Daban Q. Jaff
>
> **摘要:** Polarity detection becomes substantially more challenging under domain shift, particularly in heterogeneous, long-form narratives with complex discourse structure, such as Holocaust oral histories. This paper presents a corpus-scale diagnostic study of off-the-shelf sentiment classifiers on long-form Holocaust oral histories, using three pretrained transformer-based polarity classifiers on a corpus of 107,305 utterances and 579,013 sentences. After assembling model outputs, we introduce an agreement-based stability taxonomy (ABC) to stratify inter-model output stability. We report pairwise percent agreement, Cohen kappa, Fleiss kappa, and row-normalized confusion matrices to localize systematic disagreement. As an auxiliary descriptive signal, a T5-based emotion classifier is applied to stratified samples from each agreement stratum to compare emotion distributions across strata. The combination of multi-model label triangulation and the ABC taxonomy provides a cautious, operational framework for characterizing where and how sentiment models diverge in sensitive historical narratives. Inter-model agreement is low to moderate overall and is driven primarily by boundary decisions around neutrality.
>
---
#### [new 040] Authorship Impersonation via LLM Prompting does not Evade Authorship Verification Methods
- **分类: cs.CL**

- **简介: 该论文属于作者身份验证任务，研究LLM生成文本是否能绕过AV系统。通过实验发现，LLM生成文本无法有效模仿作者特征，现有系统仍具 robustness。**

- **链接: [https://arxiv.org/pdf/2603.29454](https://arxiv.org/pdf/2603.29454)**

> **作者:** Baoyi Zeng; Andrea Nini
>
> **备注:** 11 pages, 3 figures
>
> **摘要:** Authorship verification (AV), the task of determining whether a questioned text was written by a specific individual, is a critical part of forensic linguistics. While manual authorial impersonation by perpetrators has long been a recognized threat in historical forensic cases, recent advances in large language models (LLMs) raise new challenges, as adversaries may exploit these tools to impersonate another's writing. This study investigates whether prompted LLMs can generate convincing authorial impersonations and whether such outputs can evade existing forensic AV systems. Using GPT-4o as the adversary model, we generated impersonation texts under four prompting conditions across three genres: emails, text messages, and social media posts. We then evaluated these outputs against both non-neural AV methods (n-gram tracing, Ranking-Based Impostors Method, LambdaG) and neural approaches (AdHominem, LUAR, STAR) within a likelihood-ratio framework. Results show that LLM-generated texts failed to sufficiently replicate authorial individuality to bypass established AV systems. We also observed that some methods achieved even higher accuracy when rejecting impersonation texts compared to genuine negative samples. Overall, these findings indicate that, despite the accessibility of LLMs, current AV systems remain robust against entry-level impersonation attempts across multiple genres. Furthermore, we demonstrate that this counter-intuitive resilience stems, at least in part, from the higher lexical diversity and entropy inherent in LLM-generated texts.
>
---
#### [new 041] APEX-EM: Non-Parametric Online Learning for Autonomous Agents via Structured Procedural-Episodic Experience Replay
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文提出APEX-EM框架，解决LLM自主代理缺乏持久程序记忆的问题，通过结构化经验回放实现非参数在线学习。**

- **链接: [https://arxiv.org/pdf/2603.29093](https://arxiv.org/pdf/2603.29093)**

> **作者:** Pratyay Banerjee; Masud Moshtaghi; Ankit Chadha
>
> **备注:** 17 pages, 13 figures
>
> **摘要:** LLM-based autonomous agents lack persistent procedural memory: they re-derive solutions from scratch even when structurally identical tasks have been solved before. We present \textbf{APEX-EM}, a non-parametric online learning framework that accumulates, retrieves, and reuses structured procedural plans without modifying model weights. APEX-EM introduces: (1) a \emph{structured experience representation} encoding the full procedural-episodic trace of each execution -- planning steps, artifacts, iteration history with error analysis, and quality scores; (2) a \emph{Plan-Retrieve-Generate-Iterate-Ingest} (PRGII) workflow with Task Verifiers providing multi-dimensional reward signals; and (3) a \emph{dual-outcome Experience Memory} with hybrid retrieval combining semantic search, structural signature matching, and plan DAG traversal -- enabling cross-domain transfer between tasks sharing no lexical overlap but analogous operational structure. Successful experiences serve as positive in-context examples; failures as negative examples with structured error annotations. We evaluate on BigCodeBench~\cite{zhuo2025bigcodebench}, KGQAGen-10k~\cite{zhang2025kgqagen}, and Humanity's Last Exam~\cite{phan2025hle} using Claude Sonnet 4.5 and Opus 4.5. On KGQAGen-10k, APEX-EM achieves 89.6\% accuracy versus 41.3\% without memory (+48.3pp), surpassing the oracle-retrieval upper bound (84.9\%). On BigCodeBench, it reaches 83.3\% SR from a 53.9\% baseline (+29.4pp), exceeding MemRL's~\cite{memrl2025} +11.0pp gain under comparable frozen-backbone conditions (noting backbone differences controlled for in our analysis). On HLE, entity graph retrieval reaches 48.0\% from 25.2\% (+22.8pp). Ablations show component value is task-dependent: rich judge feedback is negligible for code generation but critical for structured queries (+10.3pp), while binary-signal iteration partially compensates for weaker feedback.
>
---
#### [new 042] On the limited utility of parallel data for learning shared multilingual representations
- **分类: cs.CL**

- **简介: 该论文属于跨语言表示学习任务，研究平行数据对多语言表示对齐的影响。工作表明平行数据仅在预训练初期有微小作用，跨语言对齐主要由其他因素驱动。**

- **链接: [https://arxiv.org/pdf/2603.29026](https://arxiv.org/pdf/2603.29026)**

> **作者:** Julius Leino; Jörg Tiedemann
>
> **摘要:** Shared multilingual representations are essential for cross-lingual tasks and knowledge transfer across languages. This study looks at the impact of parallel data, i.e. translated sentences, in pretraining as a signal to trigger representations that are aligned across languages. We train reference models with different proportions of parallel data and show that parallel data seem to have only a minimal effect on the cross-lingual alignment. Based on multiple evaluation methods, we find that the effect is limited to potentially accelerating the representation sharing in the early phases of pretraining, and to decreasing the amount of language-specific neurons in the model. Cross-lingual alignment seems to emerge on similar levels even without the explicit signal from parallel data.
>
---
#### [new 043] Long-Document QA with Chain-of-Structured-Thought and Fine-Tuned SLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于长文档问答任务，解决长文档推理易错的问题。提出LiteCoST框架，结合结构化思维链和小模型微调，提升准确率并降低延迟。**

- **链接: [https://arxiv.org/pdf/2603.29232](https://arxiv.org/pdf/2603.29232)**

> **作者:** Zhuowen Liang; Xiaotian Lin; Zhengxuan Zhang; Yuyu Luo; Haixun Wang; Nan Tang
>
> **备注:** 26 pages, 17 figures, 10 tables. Accepted at ICLR 2026
>
> **摘要:** Large language models (LLMs) are widely applied to data analytics over documents, yet direct reasoning over long, noisy documents remains brittle and error-prone. Hence, we study document question answering (QA) that consolidates dispersed evidence into a structured output (e.g., a table, graph, or chunks) to support reliable, verifiable QA. We propose a two-pillar framework, LiteCoST, to achieve both high accuracy and low latency with small language models (SLMs). Pillar 1: Chain-of-Structured-Thought (CoST). We introduce a CoST template, a schema-aware instruction that guides a strong LLM to produce both a step-wise CoST trace and the corresponding structured output. The process induces a minimal structure, normalizes entities/units, aligns records, serializes the output, and verifies/refines it, yielding auditable supervision. Pillar 2: SLM fine-tuning. The compact models are trained on LLM-generated CoST data in two stages: Supervised Fine-Tuning for structural alignment, followed by Group Relative Policy Optimization (GRPO) incorporating triple rewards for answer/format quality and process consistency. By distilling structure-first behavior into SLMs, this approach achieves LLM-comparable quality on multi-domain long-document QA using 3B/7B SLMs, while delivering 2-4x lower latency than GPT-4o and DeepSeek-R1 (671B). The code is available at this https URL.
>
---
#### [new 044] Rewrite the News: Tracing Editorial Reuse Across News Agencies
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于文本重用检测任务，旨在识别跨语言新闻中的句子级内容重复，解决多语言新闻中编辑重用难以被传统方法发现的问题。通过时间戳和弱监督方法，分析 reused 内容在文章中的位置。**

- **链接: [https://arxiv.org/pdf/2603.29937](https://arxiv.org/pdf/2603.29937)**

> **作者:** Soveatin Kuntur; Nina Smirnova; Anna Wroblewska; Philipp Mayr; Sebastijan Razboršek Maček
>
> **备注:** The paper is accepted to SoCon-NLPSI 2026 : Social Context (SoCon) and Integrating NLP and Psychology to Study Social Interactions (NLPSI) workshop co-located with LREC 2026
>
> **摘要:** This paper investigates sentence-level text reuse in multilingual journalism, analyzing where reused content occurs within articles. We present a weakly supervised method for detecting sentence-level cross-lingual reuse without requiring full translations, designed to support automated pre-selection to reduce information overload for journalists (Holyst et al., 2024). The study compares English-language articles from the Slovenian Press Agency (STA) with reports from 15 foreign agencies (FA) in seven languages, using publication timestamps to retain the earliest likely foreign source for each reused sentence. We analyze 1,037 STA and 237,551 FA articles from two time windows (October 7-November 2, 2023; February 1-28, 2025) and identify 1,087 aligned sentence pairs after filtering to the earliest sources. Reuse occurs in 52% of STA articles and 1.6% of FA articles and is predominantly non-literal, involving paraphrase and compositional reuse from multiple sources. Reused content tends to appear in the middle and end of English articles, while leads are more often original, indicating that simple lexical matching overlooks substantial editorial reuse. Compared with prior work focused on monolingual overlap, we (i) detect reuse across languages without requiring full translation, (ii) use publication timing to identify likely sources, and (iii) analyze where reused material is situated within articles. Dataset and code: this https URL.
>
---
#### [new 045] SNEAK: Evaluating Strategic Communication and Information Leakage in Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出SNEAK基准，评估大语言模型在信息共享中的战略沟通能力，解决如何在传递信息与保密之间平衡的问题。**

- **链接: [https://arxiv.org/pdf/2603.29846](https://arxiv.org/pdf/2603.29846)**

> **作者:** Adar Avsian; Larry Heck
>
> **摘要:** Large language models (LLMs) are increasingly deployed in multi-agent settings where communication must balance informativeness and secrecy. In such settings, an agent may need to signal information to collaborators while preventing an adversary from inferring sensitive details. However, existing LLM benchmarks primarily evaluate capabilities such as reasoning, factual knowledge, or instruction following, and do not directly measure strategic communication under asymmetric information. We introduce SNEAK (Secret-aware Natural language Evaluation for Adversarial Knowledge), a benchmark for evaluating selective information sharing in language models. In SNEAK, a model is given a semantic category, a candidate set of words, and a secret word, and must generate a message that indicates knowledge of the secret without revealing it too clearly. We evaluate generated messages using two simulated agents with different information states: an ally, who knows the secret and must identify the intended message, and a chameleon, who does not know the secret and attempts to infer it from the message. This yields two complementary metrics: utility, measuring how well the message communicates to collaborators, and leakage, measuring how much information it reveals to an adversary. Using this framework, we analyze the trade-off between informativeness and secrecy in modern language models and show that strategic communication under asymmetric information remains a challenging capability for current systems. Notably, human participants outperform all evaluated models by a large margin, achieving up to four times higher scores.
>
---
#### [new 046] MemFactory: Unified Inference & Training Framework for Agent Memory
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出MemFactory框架，解决记忆增强型AI代理的统一训练与推理问题。通过模块化设计和GRPO优化，提升记忆操作效率，支持多种先进模型，显著提高性能。**

- **链接: [https://arxiv.org/pdf/2603.29493](https://arxiv.org/pdf/2603.29493)**

> **作者:** Ziliang Guo; Ziheng Li; Zhiyu Li
>
> **备注:** 10 pages, Code: this https URL
>
> **摘要:** Memory-augmented Large Language Models (LLMs) are essential for developing capable, long-term AI agents. Recently, applying Reinforcement Learning (RL) to optimize memory operations, such as extraction, updating, and retrieval, has emerged as a highly promising research direction. However, existing implementations remain highly fragmented and task-specific, lacking a unified infrastructure to streamline the integration, training, and evaluation of these complex pipelines. To address this gap, we present MemFactory, the first unified, highly modular training and inference framework specifically designed for memory-augmented agents. Inspired by the success of unified fine-tuning frameworks like LLaMA-Factory, MemFactory abstracts the memory lifecycle into atomic, plug-and-play components, enabling researchers to seamlessly construct custom memory agents via a "Lego-like" architecture. Furthermore, the framework natively integrates Group Relative Policy Optimization (GRPO) to fine-tune internal memory management policies driven by multi-dimensional environmental rewards. MemFactory provides out-of-the-box support for recent cutting-edge paradigms, including Memory-R1, RMM, and MemAgent. We empirically validate MemFactory on the open-source MemAgent architecture using its publicly available training and evaluation data. Across both in-domain and out-of-distribution evaluation sets, MemFactory consistently improves performance over the corresponding base models, with relative gains of up to 14.8%. By providing a standardized, extensible, and easy-to-use infrastructure, MemFactory significantly lowers the barrier to entry, paving the way for future innovations in memory-driven AI agents.
>
---
#### [new 047] Bringing Up a Bilingual BabyLM: Investigating Multilingual Language Acquisition Using Small-Scale Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于语言习得研究任务，旨在探讨双语学习是否导致延迟及输入结构的影响。通过构建双语数据集并训练模型，验证双语输入对学习效果的影响。**

- **链接: [https://arxiv.org/pdf/2603.29552](https://arxiv.org/pdf/2603.29552)**

> **作者:** Linda Zeng; Steven Y. Feng; Michael C. Frank
>
> **备注:** Code and data at this https URL
>
> **摘要:** Multilingualism is incredibly common around the world, leading to many important theoretical and practical questions about how children learn multiple languages at once. For example, does multilingual acquisition lead to delays in learning? Are there better and worse ways to structure multilingual input? Many correlational studies address these questions, but it is surprisingly difficult to get definitive answers because children cannot be randomly assigned to be multilingual and data are typically not matched between languages. We use language model training as a method for simulating a variety of highly controlled exposure conditions, and create matched 100M-word mono- and bilingual datasets using synthetic data and machine translation. We train GPT-2 models on monolingual and bilingual data organized to reflect a range of exposure regimes, and evaluate their performance on perplexity, grammaticality, and semantic knowledge. Across model scales and measures, bilingual models perform similarly to monolingual models in one language, but show strong performance in the second language as well. These results suggest that there are no strong differences between different bilingual exposure regimes, and that bilingual input poses no in-principle challenges for agnostic statistical learners.
>
---
#### [new 048] Enhancing Structural Mapping with LLM-derived Abstractions for Analogical Reasoning in Narratives
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的类比推理任务，旨在提升叙事结构映射的准确性。通过引入LLM生成的抽象层次，增强类比推理能力，并提出YARN框架进行实验验证。**

- **链接: [https://arxiv.org/pdf/2603.29997](https://arxiv.org/pdf/2603.29997)**

> **作者:** Mohammadhossein Khojasteh; Yifan Jiang; Stefano De Giorgis; Frank van Harmelen; Filip Ilievski
>
> **摘要:** Analogical reasoning is a key driver of human generalization in problem-solving and argumentation. Yet, analogies between narrative structures remain challenging for machines. Cognitive engines for structural mapping are not directly applicable, as they assume pre-extracted entities, whereas LLMs' performance is sensitive to prompt format and the degree of surface similarity between narratives. This gap motivates a key question: What is the impact of enhancing structural mapping with LLM-derived abstractions on their analogical reasoning ability in narratives? To that end, we propose a modular framework named YARN (Yielding Abstractions for Reasoning in Narratives), which uses LLMs to decompose narratives into units, abstract these units, and then passes them to a mapping component that aligns elements across stories to perform analogical reasoning. We define and operationalize four levels of abstraction that capture both the general meaning of units and their roles in the story, grounded in prior work on framing. Our experiments reveal that abstractions consistently improve model performance, resulting in competitive or better performance than end-to-end LLM baselines. Closer error analysis reveals the remaining challenges in abstraction at the right level, in incorporating implicit causality, and an emerging categorization of analogical patterns in narratives. YARN enables systematic variation of experimental settings to analyze component contributions, and to support future work, we make the code for YARN openly available.
>
---
#### [new 049] PolarQuant: Optimal Gaussian Weight Quantization via Hadamard Rotation for LLM Compression
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于模型压缩任务，旨在解决大语言模型量化过程中精度损失问题。提出PolarQuant方法，通过哈达玛旋转和高斯量化实现近无损压缩。**

- **链接: [https://arxiv.org/pdf/2603.29078](https://arxiv.org/pdf/2603.29078)**

> **作者:** Caio Vicentino
>
> **备注:** 10 pages, 5 tables, 2 algorithms. Code: this https URL Models:this https URL
>
> **摘要:** We present PolarQuant, a post-training weight quantization method for large language models (LLMs) that exploits the distributional structure of neural network weights to achieve near-lossless compression. PolarQuant operates in three stages: (1) block-wise normalization to the unit hypersphere, (2) Walsh-Hadamard rotation to transform coordinates into approximately Gaussian random variables, and (3) quantization with centroids matched to the Gaussian distribution. Our ablation reveals that Hadamard rotation alone accounts for 98% of the quality improvement, reducing Qwen3.5-9B perplexity from 6.90 (absmax Q5) to 6.40 (Delta = +0.03 from FP16), making it practically lossless without any calibration data. Furthermore, PolarQuant functions as an effective preprocessing step for downstream INT4 quantizers: PolarQuant Q5 dequantized and re-quantized by torchao INT4 achieves perplexity 6.56 versus 6.68 for direct absmax INT4, while maintaining 43.1 tok/s throughput at 6.5 GB VRAM. Code and models are publicly available.
>
---
#### [new 050] GISTBench: Evaluating LLM User Understanding via Evidence-Based Interest Verification
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出GISTBench，用于评估大语言模型在推荐系统中理解用户兴趣的能力。解决模型如何从交互历史中提取和验证用户兴趣的问题，通过新指标和数据集进行验证。**

- **链接: [https://arxiv.org/pdf/2603.29112](https://arxiv.org/pdf/2603.29112)**

> **作者:** Iordanis Fostiropoulos; Muhammad Rafay Azhar; Abdalaziz Sawwan; Boyu Fang; Yuchen Liu; Jiayi Liu; Hanchao Yu; Qi Guo; Jianyu Wang; Fei Liu; Xiangjun Fan
>
> **备注:** 9 figures, 20 tables; code at this https URL
>
> **摘要:** We introduce GISTBench, a benchmark for evaluating Large Language Models' (LLMs) ability to understand users from their interaction histories in recommendation systems. Unlike traditional RecSys benchmarks that focus on item prediction accuracy, our benchmark evaluates how well LLMs can extract and verify user interests from engagement data. We propose two novel metric families: Interest Groundedness (IG), decomposed into precision and recall components to separately penalize hallucinated interest categories and reward coverage, and Interest Specificity (IS), which assesses the distinctiveness of verified LLM-predicted user profiles. We release a synthetic dataset constructed on real user interactions on a global short-form video platform. Our dataset contains both implicit and explicit engagement signals and rich textual descriptions. We validate our dataset fidelity against user surveys, and evaluate eight open-weight LLMs spanning 7B to 120B parameters. Our findings reveal performance bottlenecks in current LLMs, particularly their limited ability to accurately count and attribute engagement signals across heterogeneous interaction types.
>
---
#### [new 051] A Comprehensive Information-Decomposition Analysis of Large Vision-Language Models
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文属于视觉语言模型分析任务，旨在解决模型决策机制不透明的问题。通过信息分解方法，量化模型的多模态融合情况，揭示其处理策略与学习过程。**

- **链接: [https://arxiv.org/pdf/2603.29676](https://arxiv.org/pdf/2603.29676)**

> **作者:** Lixin Xiu; Xufang Luo; Hideki Nakayama
>
> **备注:** Accepted at ICLR 2026. Project page: this https URL
>
> **摘要:** Large vision-language models (LVLMs) achieve impressive performance, yet their internal decision-making processes remain opaque, making it difficult to determine if the success stems from true multimodal fusion or from reliance on unimodal priors. To address this attribution gap, we introduce a novel framework using partial information decomposition (PID) to quantitatively measure the "information spectrum" of LVLMs -- decomposing a model's decision-relevant information into redundant, unique, and synergistic components. By adapting a scalable estimator to modern LVLM outputs, our model-agnostic pipeline profiles 26 LVLMs on four datasets across three dimensions -- breadth (cross-model & cross-task), depth (layer-wise information dynamics), and time (learning dynamics across training). Our analysis reveals two key results: (i) two task regimes (synergy-driven vs. knowledge-driven) and (ii) two stable, contrasting family-level strategies (fusion-centric vs. language-centric). We also uncover a consistent three-phase pattern in layer-wise processing and identify visual instruction tuning as the key stage where fusion is learned. Together, these contributions provide a quantitative lens beyond accuracy-only evaluation and offer insights for analyzing and designing the next generation of LVLMs. Code and data are available at this https URL .
>
---
#### [new 052] StepCache: Step-Level Reuse with Lightweight Verification and Selective Patching for LLM Serving
- **分类: cs.OS; cs.AI; cs.CL; cs.DC**

- **简介: 该论文提出StepCache，用于解决LLM服务中重复请求的高效缓存问题。通过步骤级复用和轻量验证，提升响应速度与准确性。**

- **链接: [https://arxiv.org/pdf/2603.28795](https://arxiv.org/pdf/2603.28795)**

> **作者:** Azam Nouri
>
> **备注:** 9 pages, 1 figure
>
> **摘要:** We address LLM serving workloads where repeated requests share a common solution structure but differ in localized constraints, such as output schema, variable names, or numeric constants. Prior caching approaches typically reuse either full responses (semantic caching) or model-internal KV/prefix states, which are respectively brittle under partial changes or tightly coupled to specific backends. We present StepCache, a backend-agnostic step-level reuse layer that segments outputs into ordered steps, retrieves the best-matching cached request, verifies steps using lightweight task-aware checks, and regenerates only failing regions via selective patching. StepCache additionally supports strict structured-output enforcement for JSON, including single-step extraction, required-key constraints, and one-shot repair, as well as conservative skip-reuse fallbacks for semantic changes. For linear equations, StepCache promotes verification into correction via a bounded repair loop with a deterministic fallback that guarantees correctness when the backend model fails. In a CPU-only perturbation-heavy micro-benchmark on math and JSON variants, averaged over three seeds, StepCache reduces mean latency from 2.13 s to 0.67 s, median latency from 2.42 s to 0.01 s, and p95 latency from 3.38 s to 3.30 s. It also reduces total token usage from 36.1k to 27.3k and improves end-to-end correctness from 72.5% to 100% under task-specific checks and a stitched-output integrity check. Across requests, 79.7% take the reuse-only fast path, 5.4% require patching, and 14.9% trigger skip-reuse.
>
---
#### [new 053] Training-Free Dynamic Upcycling of Expert Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出DUME方法，解决LLM领域专家模型难以融合的问题。通过动态组合预训练专家，构建统一多任务模型，无需额外训练即可保持性能。**

- **链接: [https://arxiv.org/pdf/2603.29765](https://arxiv.org/pdf/2603.29765)**

> **作者:** Eros Fanì; Oğuzhan Ersoy
>
> **备注:** Accepted at the ICLR 2026 Workshop on Scaling Post-training for LLMs
>
> **摘要:** Large Language Models (LLMs) have achieved remarkable performance on a wide range of specialized tasks, exhibiting strong problem-solving capabilities. However, training these models is prohibitively expensive, and they often lack domain-specific expertise because they rely on general knowledge datasets. Expertise finetuning can address this issue; however, it often leads to overspecialization, and developing a single multi-domain expert remains difficult due to diverging objectives. Furthermore, multitask training is challenging due to interference and catastrophic forgetting. Existing work proposes combining the expertise of dense models within a Mixture of Experts (MoE) architecture, although this approach still requires multitask finetuning. To address these issues, we introduce Dynamic Upcycling MoE (DUME), a novel approach that reuses dense experts trained on different domains to construct a unified MoE model. Our method builds a single multitask model that preserves the capabilities of the original dense experts without requiring additional training. DUME is both cost-efficient and scalable: by leveraging the closed-form solution of ridge regression, it eliminates the need for further optimization and enables experts to be added dynamically while maintaining the model's original performance. We demonstrate that DUME consistently outperforms baseline approaches in both causal language modeling and reasoning settings. Finally, we also show that the DUME model can be fine-tuned to further improve performance. We show that, in the causal language modeling setting, DUME can retain up to 97.6% of a dense expert model specialized in one particular domain, and that it can also surpass it in the reasoning setting, where it can achieve 102.1% of the dense expert performance. Our code is available at: this http URL.
>
---
#### [new 054] OneComp: One-Line Revolution for Generative AI Model Compression
- **分类: cs.LG; cs.AI; cs.CE; cs.CL**

- **简介: 该论文属于模型压缩任务，旨在解决部署大模型时的内存和计算成本问题。提出OneComp框架，实现自动化的混合精度量化与优化流程。**

- **链接: [https://arxiv.org/pdf/2603.28845](https://arxiv.org/pdf/2603.28845)**

> **作者:** Yuma Ichikawa; Keiji Kimura; Akihiro Yoshida; Yudai Fujimoto; Hiroki Tokura; Yamato Arai; Yoshiyuki Ishii; Yusei Kawakami; Genki Shikada; Achille Jacquemond; Yoshihiko Fujisawa; Katsuki Fujisawa; Takumi Honda; Akira Sakai
>
> **备注:** 31 pages, 6 figures
>
> **摘要:** Deploying foundation models is increasingly constrained by memory footprint, latency, and hardware costs. Post-training compression can mitigate these bottlenecks by reducing the precision of model parameters without significantly degrading performance; however, its practical implementation remains challenging as practitioners navigate a fragmented landscape of quantization algorithms, precision budgets, data-driven calibration strategies, and hardware-dependent execution regimes. We present OneComp, an open-source compression framework that transforms this expert workflow into a reproducible, resource-adaptive pipeline. Given a model identifier and available hardware, OneComp automatically inspects the model, plans mixed-precision assignments, and executes progressive quantization stages, ranging from layer-wise compression to block-wise refinement and global refinement. A key architectural choice is treating the first quantized checkpoint as a deployable pivot, ensuring that each subsequent stage improves the same model and that quality increases as more compute is invested. By converting state-of-the-art compression research into an extensible, open-source, hardware-aware pipeline, OneComp bridges the gap between algorithmic innovation and production-grade model deployment.
>
---
#### [new 055] An Isotropic Approach to Efficient Uncertainty Quantification with Gradient Norms
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于不确定性量化任务，旨在解决大模型中预测不确定性难以高效计算的问题。通过梯度范数和各向同性假设，提出轻量级方法同时估计认知和偶然不确定性。**

- **链接: [https://arxiv.org/pdf/2603.29466](https://arxiv.org/pdf/2603.29466)**

> **作者:** Nils Grünefeld; Jes Frellsen; Christian Hardmeier
>
> **摘要:** Existing methods for quantifying predictive uncertainty in neural networks are either computationally intractable for large language models or require access to training data that is typically unavailable. We derive a lightweight alternative through two approximations: a first-order Taylor expansion that expresses uncertainty in terms of the gradient of the prediction and the parameter covariance, and an isotropy assumption on the parameter covariance. Together, these yield epistemic uncertainty as the squared gradient norm and aleatoric uncertainty as the Bernoulli variance of the point prediction, from a single forward-backward pass through an unmodified pretrained model. We justify the isotropy assumption by showing that covariance estimates built from non-training data introduce structured distortions that isotropic covariance avoids, and that theoretical results on the spectral properties of large networks support the approximation at scale. Validation against reference Markov Chain Monte Carlo estimates on synthetic problems shows strong correspondence that improves with model size. We then use the estimates to investigate when each uncertainty type carries useful signal for predicting answer correctness in question answering with large language models, revealing a benchmark-dependent divergence: the combined estimate achieves the highest mean AUROC on TruthfulQA, where questions involve genuine conflict between plausible answers, but falls to near chance on TriviaQA's factual recall, suggesting that parameter-level uncertainty captures a fundamentally different signal than self-assessment methods.
>
---
#### [new 056] Advancing LLM-based phoneme-to-grapheme for multilingual speech recognition
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于多语言语音识别任务，解决多语言P2G中的语言感知生成和数据不平衡问题。通过改进训练策略提升识别准确率。**

- **链接: [https://arxiv.org/pdf/2603.29217](https://arxiv.org/pdf/2603.29217)**

> **作者:** Lukuang Dong; Ziwei Li; Saierdaer Yusuyin; Xianyu Zhao; Zhijian Ou
>
> **备注:** Update after INTERSPEECH2026 submission
>
> **摘要:** Phoneme-based ASR factorizes recognition into speech-to-phoneme (S2P) and phoneme-to-grapheme (P2G), enabling cross-lingual acoustic sharing while keeping language-specific orthography in a separate module. While large language models (LLMs) are promising for P2G, multilingual P2G remains challenging due to language-aware generation and severe cross-language data imbalance. We study multilingual LLM-based P2G on the ten-language CV-Lang10 benchmark. We examine robustness strategies that account for S2P uncertainty, including DANP and Simplified SKM (S-SKM). S-SKM is a Monte Carlo approximation that avoids CTC-based S2P probability weighting in P2G training. Robust training and low-resource oversampling reduce the average WER from 10.56% to 7.66%.
>
---
#### [new 057] UnWeaving the knots of GraphRAG -- turns out VectorRAG is almost enough
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于RAG任务，旨在解决传统方法在处理多跳问题时的不足。通过引入实体分解，简化图RAG结构，提升信息表示与检索效果。**

- **链接: [https://arxiv.org/pdf/2603.29875](https://arxiv.org/pdf/2603.29875)**

> **作者:** Ryszard Tuora; Mateusz Galiński; Michał Godziszewski; Michał Karpowicz; Mateusz Czyżnikiewicz; Adam Kozakiewicz; Tomasz Ziętkiewicz
>
> **摘要:** One of the key problems in Retrieval-augmented generation (RAG) systems is that chunk-based retrieval pipelines represent the source chunks as atomic objects, mixing the information contained within such a chunk into a single vector. These vector representations are then fundamentally treated as isolated, independent and self-sufficient, with no attempt to represent possible relations between them. Such an approach has no dedicated mechanisms for handling multi-hop questions. Graph-based RAG systems aimed to ameliorate this problem by modeling information as knowledge-graphs, with entities represented by nodes being connected by robust relations, and forming hierarchical communities. This approach however suffers from its own issues with some of them being: orders of magnitude increased componential complexity in order to create graph-based indices, and reliance on heuristics for performing retrieval. We propose UnWeaver, a novel RAG framework simplifying the idea of GraphRAG. UnWeaver disentangles the contents of the documents into entities which can occur across multiple chunks using an LLM. In the retrieval process entities are used as an intermediate way of recovering original text chunks hence preserving fidelity to the source material. We argue that entity-based decomposition yields a more distilled representation of original information, and additionally serves to reduce noise in the indexing, and generation process.
>
---
#### [new 058] Physiological and Semantic Patterns in Medical Teams Using an Intelligent Tutoring System
- **分类: cs.AI; cs.CL**

- **简介: 论文研究医疗团队在智能辅导系统中的生理与语义模式，探讨协作中的认知和情感状态。任务为理解团队协作中的关键时刻，通过分析生理同步与对话变化解决协作机制问题。**

- **链接: [https://arxiv.org/pdf/2603.29950](https://arxiv.org/pdf/2603.29950)**

> **作者:** Xiaoshan Huang; Conrad Borchers; Jiayi Zhang; Susanne P. Lajoie
>
> **备注:** Accepted as short paper to the 27th International Conference on Artificial Intelligence in Education (AIED 2026)
>
> **摘要:** Effective collaboration requires teams to manage complex cognitive and emotional states through Socially Shared Regulation of Learning (SSRL). Physiological synchrony (i.e., longitudinal alignment in physiological signals) can indicate these states, but is hard to interpret on its own. We investigate the physiological and conversational dynamics of four medical dyads diagnosing a virtual patient case using an intelligent tutoring system. Semantic shifts in dialogue were correlated with transient physiological synchrony peaks. We also coded utterance segments for SSRL and derived cosine similarity using sentence embeddings. The results showed that activating prior knowledge featured significantly lower semantic similarity than simpler task execution. High physiological synchrony was associated with lower semantic similarity, suggesting that such moments involve exploratory and varied language use. Qualitative analysis triangulated these synchrony peaks as ``pivotal moments'': successful teams synchronized during shared discovery, while unsuccessful teams peaked during shared uncertainty. This research advances human-centered AI by demonstrating how biological signals can be fused with dialogues to understand critical moments in problem solving.
>
---
#### [new 059] Aligning Multimodal Sequential Recommendations via Robust Direct Preference Optimization with Sparse MoE
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于推荐系统任务，解决隐式反馈下DPO效果不佳的问题。通过引入动态采样策略提升排序性能，并结合稀疏MoE模型实现高效扩展。**

- **链接: [https://arxiv.org/pdf/2603.29259](https://arxiv.org/pdf/2603.29259)**

> **作者:** Hejin Huang; Jusheng Zhang; Kaitong Cai; Jian Wang; Rong Pan
>
> **摘要:** Preference-based alignment objectives have been widely adopted, from RLHF-style pairwise learning in large language models to emerging applications in recommender systems. Yet, existing work rarely examines how Direct Preference Optimization (DPO) behaves under implicit feedback, where unobserved items are not reliable negatives. We conduct systematic experiments on multimodal sequential recommendation to compare common negative-selection strategies and their interaction with DPO training. Our central finding is that a simple modification, replacing deterministic hard negatives with stochastic sampling from a dynamic top-K candidate pool, consistently improves ranking performance. We attribute its effectiveness to two factors: (1) reducing erroneous suppressive gradients caused by false negatives, and (2) retaining informative hard signals while smoothing optimization via controlled stochasticity. With an optional sparse Mixture-of-Experts encoder for efficient capacity scaling, RoDPO achieves up to 5.25% NDCG@5 on three Amazon benchmarks, with nearly unchanged inference cost.
>
---
#### [new 060] Xuanwu: Evolving General Multimodal Models into an Industrial-Grade Foundation for Content Ecosystems
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出Xuanwu VL-2B模型，解决多模态模型在内容审核中的泛化与遗忘问题，通过优化架构和训练策略提升性能。**

- **链接: [https://arxiv.org/pdf/2603.29211](https://arxiv.org/pdf/2603.29211)**

> **作者:** Zhiqian Zhang; Xu Zhao; Xiaoqing Xu; Guangdong Liang; Weijia Wang; Xiaolei Lv; Bo Li; Jun Gao
>
> **备注:** 41 pages, 10 figures
>
> **摘要:** In recent years, multimodal large models have continued to improve on general benchmarks. However, in real-world content moderation and adversarial settings, mainstream models still suffer from degraded generalization and catastrophic forgetting because of limited fine-grained visual perception and insufficient modeling of long-tail noise. In this paper, we present Xuanwu VL-2B as a case study of how general multimodal models can be developed into an industrial-grade foundation model for content ecosystems. The model adopts a compact InternViT-300M + MLP + Qwen3 1.7B architecture, balancing fine-grained visual perception, language-semantic alignment, and deployment cost within an approximately 2B-parameter budget. To balance business specialization with the retention of general capabilities, we developed a data iteration and curation mechanism and trained the model through a progressive three-stage pipeline: pre-training, mid-training, and post-training. Ablation studies and offline business evaluations show that Xuanwu VL-2B achieves an average score of 67.90 across seven OpenCompass multimodal metrics (vs. 64.27 for InternVL 3.5 2B), an average recall of 94.38% over seven independent business moderation tasks, and a weighted overall recall of 82.82% on policy-violating text in challenging adversarial OCR scenarios, outperforming Gemini-2.5-Pro (76.72%). These results show that, under a limited parameter budget, Xuanwu VL-2B achieves a practical balance among business alignment, visual perception, general capability retention, and deployment cost.
>
---
#### [new 061] PRISM: PRIor from corpus Statistics for topic Modeling
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于主题建模任务，旨在解决传统方法依赖外部知识的问题。提出PRISM，通过语料库统计初始化LDA，提升主题连贯性与可解释性。**

- **链接: [https://arxiv.org/pdf/2603.29406](https://arxiv.org/pdf/2603.29406)**

> **作者:** Tal Ishon; Yoav Goldberg; Uri Shaham
>
> **摘要:** Topic modeling seeks to uncover latent semantic structure in text, with LDA providing a foundational probabilistic framework. While recent methods often incorporate external knowledge (e.g., pre-trained embeddings), such reliance limits applicability in emerging or underexplored domains. We introduce \textbf{PRISM}, a corpus-intrinsic method that derives a Dirichlet parameter from word co-occurrence statistics to initialize LDA without altering its generative process. Experiments on text and single cell RNA-seq data show that PRISM improves topic coherence and interpretability, rivaling models that rely on external knowledge. These results underscore the value of corpus-driven initialization for topic modeling in resource-constrained settings. Code is available at: this https URL.
>
---
#### [new 062] Reasoning-Driven Synthetic Data Generation and Evaluation
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于数据生成任务，旨在解决数据稀缺与隐私问题。提出Simula框架，通过推理驱动方式生成可控、可解释的合成数据，提升数据生成效率与质量。**

- **链接: [https://arxiv.org/pdf/2603.29791](https://arxiv.org/pdf/2603.29791)**

> **作者:** Tim R. Davidson; Benoit Seguin; Enrico Bacis; Cesar Ilharco; Hamza Harkous
>
> **备注:** Accepted to TMLR 2026, J2C Certification
>
> **摘要:** Although many AI applications of interest require specialized multi-modal models, relevant data to train such models is inherently scarce or inaccessible. Filling these gaps with human annotators is prohibitively expensive, error-prone, and time-consuming, leading model builders to increasingly consider synthetic data as a scalable alternative. However, existing synthetic data generation methods often rely on manual prompts, evolutionary algorithms, or extensive seed data from the target distribution - limiting their scalability, explainability, and control. In this paper, we introduce Simula: a novel reasoning-driven framework for data generation and evaluation. It employs a seedless, agentic approach to generate synthetic datasets at scale, allowing users to define desired dataset characteristics through an explainable and controllable process that enables fine-grained resource allocation. We show the efficacy of our approach on a variety of datasets, rigorously testing both intrinsic and downstream properties. Our work (1) offers guidelines for synthetic data mechanism design, (2) provides insights into generating and evaluating synthetic data at scale, and (3) unlocks new opportunities for developing and deploying AI in domains where data scarcity or privacy concerns are paramount.
>
---
#### [new 063] Less Is More? Selective Visual Attention to High-Importance Regions for Multimodal Radiology Summarization
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态放射学摘要任务，旨在解决视觉噪声干扰和模型价值有限的问题。通过聚焦关键病灶区域，提出ViTAS模型提升摘要质量。**

- **链接: [https://arxiv.org/pdf/2603.29901](https://arxiv.org/pdf/2603.29901)**

> **作者:** Mst. Fahmida Sultana Naznin; Adnan Ibney Faruq; Mushfiqur Rahman; Niloy Kumar Mondal; Md. Mehedi Hasan Shawon; Md Rakibul Hasan
>
> **摘要:** Automated radiology report summarization aims to distill verbose findings into concise clinical impressions, but existing multimodal models often struggle with visual noise and fail to meaningfully improve over strong text-only baselines in the FINDINGS $\to$ IMPRESSION transformation. We challenge two prevailing assumptions: (1) that more visual input is always better, and (2) that multimodal models add limited value when findings already contain rich image-derived detail. Through controlled ablations on MIMIC-CXR benchmark, we show that selectively focusing on pathology-relevant visual patches rather than full images yields substantially better performance. We introduce ViTAS, Visual-Text Attention Summarizer, a multi-stage pipeline that combines ensemble-guided MedSAM2 lung segmentation, bidirectional cross-attention for multi-view fusion, Shapley-guided adaptive patch clustering, and hierarchical visual tokenization feeding a ViT. ViTAS achieves SOTA results with 29.25% BLEU-4 and 69.83% ROUGE-L, improved factual alignment in qualitative analysis, and the highest expert-rated human evaluation scores. Our findings demonstrate that less but more relevant visual input is not only sufficient but superior for multimodal radiology summarization.
>
---
#### [new 064] Tracking Equivalent Mechanistic Interpretations Across Neural Networks
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于神经网络可解释性领域，解决机制解释难以扩展和通用的问题。提出解释等价性概念，通过分析模型表示相似性，建立解释与算法、电路的关联，推动更严谨的解释方法发展。**

- **链接: [https://arxiv.org/pdf/2603.30002](https://arxiv.org/pdf/2603.30002)**

> **作者:** Alan Sun; Mariya Toneva
>
> **备注:** 32 pages, 5 figures, ICLR 2026
>
> **摘要:** Mechanistic interpretability (MI) is an emerging framework for interpreting neural networks. Given a task and model, MI aims to discover a succinct algorithmic process, an interpretation, that explains the model's decision process on that task. However, MI is difficult to scale and generalize. This stems in part from two key challenges: there is no precise notion of a valid interpretation; and, generating interpretations is often an ad hoc process. In this paper, we address these challenges by defining and studying the problem of interpretive equivalence: determining whether two different models share a common interpretation, without requiring an explicit description of what that interpretation is. At the core of our approach, we propose and formalize the principle that two interpretations of a model are equivalent if all of their possible implementations are also equivalent. We develop an algorithm to estimate interpretive equivalence and case study its use on Transformer-based models. To analyze our algorithm, we introduce necessary and sufficient conditions for interpretive equivalence based on models' representation similarity. We provide guarantees that simultaneously relate a model's algorithmic interpretations, circuits, and representations. Our framework lays a foundation for the development of more rigorous evaluation methods of MI and automated, generalizable interpretation discovery methods.
>
---
#### [new 065] Semantic Interaction for Narrative Map Sensemaking: An Insight-based Evaluation
- **分类: cs.HC; cs.AI; cs.CL; cs.IR**

- **简介: 该论文属于信息可视化任务，旨在评估语义交互在叙事地图理解中的效果。通过用户研究比较不同条件下的表现，验证地图比时间线更优，并分析分析师如何使用语义交互优化叙事。**

- **链接: [https://arxiv.org/pdf/2603.29651](https://arxiv.org/pdf/2603.29651)**

> **作者:** Brian Felipe Keith-Norambuena; Fausto German; Eric Krokos; Sarah Joseph; Chris North
>
> **备注:** Text2Story Workshop 2026 at ECIR 2026
>
> **摘要:** Semantic interaction (SI) enables analysts to incorporate their cognitive processes into AI models through direct manipulation of visualizations. While SI frameworks for narrative extraction have been proposed, empirical evaluations of their effectiveness remain limited. This paper presents a user study that evaluates SI for narrative map sensemaking, involving 33 participants under three conditions: a timeline baseline, a basic narrative map, and an interactive narrative map with SI capabilities. The results show that the map-based prototypes yielded more insights than the timeline baseline, with the SI-enabled condition reaching statistical significance and the basic map condition trending in the same direction. The SI-enabled condition showed the highest mean performance; differences between the map conditions were not statistically significant but showed large effect sizes (d > 0.8), suggesting that the study was underpowered to detect them. Qualitative analysis identified two distinct SI approaches-corrective and additive-that enable analysts to impose quality judgments and organizational structure on extracted narratives. We also find that SI users achieved comparable exploration breadth with less parameter manipulation, suggesting that SI serves as an alternative pathway for model refinement. This work provides empirical evidence that map-based representations outperform timelines for narrative sensemaking, along with qualitative insights into how analysts use SI for narrative refinement.
>
---
#### [new 066] Perfecting Human-AI Interaction at Clinical Scale. Turning Production Signals into Safer, More Human Conversations
- **分类: cs.HC; cs.AI; cs.CL; cs.MA**

- **简介: 该论文属于医疗AI交互任务，旨在提升临床场景下AI对话的安全性与可靠性。通过分析真实患者互动数据，优化对话智能，解决语音不洁、意图模糊等问题，提升患者体验与系统安全。**

- **链接: [https://arxiv.org/pdf/2603.29893](https://arxiv.org/pdf/2603.29893)**

> **作者:** Subhabrata Mukherjee; Markel Sanz Ausin; Kriti Aggarwal; Debajyoti Datta; Shanil Puri; Woojeong Jin; Tanmay Laud; Neha Manjunath; Jiayuan Ding; Bibek Paudel; Jan Schellenberger; Zepeng Frazier Huo; Walter Shen; Nima Shirazian; Nate Potter; Sathvik Perkari; Darya Filippova; Anton Morozov; Austin Mease; Vivek Muppalla; Ghada Shakir; Alex Miller; Juliana Ghukasyan; Mariska Raglow-Defranco; Maggie Taylor; Herprit Mahal; Jonathan Agnew
>
> **摘要:** Healthcare conversational AI agents shouldn't be optimized only for clean benchmark accuracy in production-first regime; they must be optimized for the lived reality of patient conversations, where audio is imperfect, intent is indirect, language shifts mid-call, and compliance hinges on how guidance is delivered. We present a production-validated framework grounded in real-time signals from 115M+ live patient-AI interactions and clinician-led testing (7K+ licensed clinicians; 500K+ test calls). These in-the-wild cues -- paralinguistics, turn-taking dynamics, clarification triggers, escalation markers, multilingual continuity, and workflow confirmations -- reveal failure modes that curated data misses and provide actionable training and evaluation signals for safety and reliability. We further show why healthcare-grade safety cannot rely on a single LLM: long-horizon dialogue and limited attention demand redundancy via governed orchestration, independent checks, and verification. Many apparent "reasoning" errors originate upstream, motivating vertical integration across contextual ASR, clarification/repair, ambient speech handling, and latency-aware model/hardware choices. Treating interaction intelligence (tone, pacing, empathy, clarification, turn-taking) as first-class safety variables, we drive measurable gains in safety, documentation, task completion, and equity in building the safest generative AI solution for autonomous patient-facing care. Deployed across more than 10 million real patient calls, Polaris attains a clinical safety score of 99.9%, while significantly improving patient experience with average patient rating of 8.95 and reducing ASR errors by 50% over enterprise ASR. These results establish real-world interaction intelligence as a critical -- and previously underexplored -- determinant of safety and reliability in patient-facing clinical AI systems.
>
---
#### [new 067] Convergent Representations of Linguistic Constructions in Human and Artificial Neural Systems
- **分类: q-bio.NC; cs.AI; cs.CL**

- **简介: 该论文属于语言处理任务，研究人类和人工智能系统如何表征语言结构。通过EEG实验，发现人类神经活动在句尾出现特定构造的脑信号，与语言模型中的表示模式相似，表明生物与人工系统在语言表征上存在收敛。**

- **链接: [https://arxiv.org/pdf/2603.29617](https://arxiv.org/pdf/2603.29617)**

> **作者:** Pegah Ramezani; Thomas Kinfe; Andreas Maier; Achim Schilling; Patrick Krauss
>
> **摘要:** Understanding how the brain processes linguistic constructions is a central challenge in cognitive neuroscience and linguistics. Recent computational studies show that artificial neural language models spontaneously develop differentiated representations of Argument Structure Constructions (ASCs), generating predictions about when and how construction-level information emerges during processing. The present study tests these predictions in human neural activity using electroencephalography (EEG). Ten native English speakers listened to 200 synthetically generated sentences across four construction types (transitive, ditransitive, caused-motion, resultative) while neural responses were recorded. Analyses using time-frequency methods, feature extraction, and machine learning classification revealed construction-specific neural signatures emerging primarily at sentence-final positions, where argument structure becomes fully disambiguated, and most prominently in the alpha band. Pairwise classification showed reliable differentiation, especially between ditransitive and resultative constructions, while other pairs overlapped. Crucially, the temporal emergence and similarity structure of these effects mirror patterns in recurrent and transformer-based language models, where constructional representations arise during integrative processing stages. These findings support the view that linguistic constructions are neurally encoded as distinct form-meaning mappings, in line with Construction Grammar, and suggest convergence between biological and artificial systems on similar representational solutions. More broadly, this convergence is consistent with the idea that learning systems discover stable regions within an underlying representational landscape - recently termed a Platonic representational space - that constrains the emergence of efficient linguistic abstractions.
>
---
#### [new 068] Spark-LLM-Eval: A Distributed Framework for Statistically Rigorous Large Language Model Evaluation
- **分类: cs.DC; cs.CL; cs.LG**

- **简介: 该论文提出Spark-LLM-Eval，解决大规模语言模型评估的效率与统计严谨性问题。通过分布式计算和统计方法，提升评估性能并确保结果可靠性。**

- **链接: [https://arxiv.org/pdf/2603.28769](https://arxiv.org/pdf/2603.28769)**

> **作者:** Subhadip Mitra
>
> **备注:** 16 pages, 2 figures, 6 tables. Open source: this https URL. Cross-list requested: cs.CL, cs.LG
>
> **摘要:** Evaluating large language models at scale remains a practical bottleneck for many organizations. While existing evaluation frameworks work well for thousands of examples, they struggle when datasets grow to hundreds of thousands or millions of samples. This scale is common when assessing model behavior across diverse domains or conducting comprehensive regression testing. We present Spark-LLM-Eval, a distributed evaluation framework built natively on Apache Spark. The system treats evaluation as a data-parallel problem, partitioningexamplesacrossexecutorsandaggregatingresultswithproperstatistical accounting. Beyond raw throughput, we emphasize statistical rigor: every reported metric includes bootstrap confidence intervals, and model comparisons come with appropriate significance tests (paired t-tests, McNemar's test, or Wilcoxon signed-rank, depending on the metric type). The framework also addresses the cost problem inherent in LLM evaluation through content-addressable response caching backed by Delta Lake, which allows iterating on metric definitions without re-running inference. We describe the system architecture, the statistical methodology, and report benchmark results showing linear scaling with cluster size. The framework and all evaluation code are available as open source.
>
---
#### [new 069] Designing FSMs Specifications from Requirements with GPT 4.0
- **分类: cs.SE; cs.AI; cs.CL; cs.FL**

- **简介: 该论文属于形式化建模任务，旨在解决从自然语言需求生成高质量FSM的问题。通过LLM框架设计FSM，并提出修复方法提升其质量。**

- **链接: [https://arxiv.org/pdf/2603.29140](https://arxiv.org/pdf/2603.29140)**

> **作者:** Omer Nguena Timo; Paul-Alexis Rodriguez; Florent Avellaneda
>
> **摘要:** Finite state machines (FSM) are executable formal specifications of reactive systems. These machines are designed based on systems' requirements. The requirements are often recorded in textual documents written in natural languages. FSMs play a crucial role in different phases of the model-driven system engineering (MDE). For example, they serve to automate testing activities. FSM quality is critical: the lower the quality of FSM, the higher the number of faults surviving the testing phase and the higher the risk of failure of the systems in production, which could lead to catastrophic scenarios. Therefore, this paper leverages recent advances in the domain of LLM to propose an LLM-based framework for designing FSMs from requirements. The framework also suggests an expert-centric approach based on FSM mutation and test generation for repairing the FSMs produced by LLMs. This paper also provides an experimental analysis and evaluation of LLM's capacities in performing the tasks presented in the framework and FSM repair via various methods. The paper presents experimental results with simulated data. These results and methods bring a new analysis and vision of LLMs that are useful for further development of machine learning technology and its applications to MDE.
>
---
#### [new 070] Sima AIunty: Caste Audit in LLM-Driven Matchmaking
- **分类: cs.CY; cs.AI; cs.CL; cs.HC; cs.SI**

- **简介: 该论文属于AI伦理任务，旨在检测LLM在婚配评估中是否再现种姓偏见。通过实验分析不同种姓和收入组合的模型评分，发现模型存在传统种姓等级的偏好。**

- **链接: [https://arxiv.org/pdf/2603.29288](https://arxiv.org/pdf/2603.29288)**

> **作者:** Atharva Naik; Shounok Kar; Varnika Sharma; Ashwin Rajadesingan; Koustuv Saha
>
> **摘要:** Social and personal decisions in relational domains such as matchmaking are deeply entwined with cultural norms and historical hierarchies, and can potentially be shaped by algorithmic and AI-mediated assessments of compatibility, acceptance, and stability. In South Asian contexts, caste remains a central aspect of marital decision-making, yet little is known about how contemporary large language models (LLMs) reproduce or disrupt caste-based stratification in such settings. In this work, we conduct a controlled audit of caste bias in LLM-mediated matchmaking evaluations using real-world matrimonial profiles. We vary caste identity across Brahmin, Kshatriya, Vaishya, Shudra, and Dalit, and income across five buckets, and evaluate five LLM families (GPT, Gemini, Llama, Qwen, and BharatGPT). Models are prompted to assess profiles along dimensions of social acceptance, marital stability, and cultural compatibility. Our analysis reveals consistent hierarchical patterns across models: same-caste matches are rated most favorably, with average ratings up to 25% higher (on a 10-point scale) than inter-caste matches, which are further ordered according to traditional caste hierarchy. These findings highlight how existing caste hierarchies are reproduced in LLM decision-making and underscore the need for culturally grounded evaluation and intervention strategies in AI systems deployed in socially sensitive domains, where such systems risk reinforcing historical forms of exclusion.
>
---
#### [new 071] Reward-Based Online LLM Routing via NeuralUCB
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于LLM路由任务，旨在优化成本与效果的平衡。通过NeuralUCB方法，在在线环境下提升路由效率，降低推理成本。**

- **链接: [https://arxiv.org/pdf/2603.30035](https://arxiv.org/pdf/2603.30035)**

> **作者:** Ming-Hua Tsai; Phat Tran
>
> **摘要:** This study investigates the use of NeuralUCB for cost-aware large language model (LLM) routing. Existing routing approaches can be broadly grouped into supervised routing methods and partial-feedback methods, each with different tradeoffs in efficiency and adaptivity. We implement a NeuralUCB-based routing policy and evaluate it on RouterBench under a simulated online setting. Experimental results show that the proposed method consistently outperforms random and min-cost baselines in utility reward. Compared with the max-quality reference, our method achieves substantially lower inference cost while maintaining competitive reward. These findings suggest that NeuralUCB is a promising approach for cost-aware LLM routing, while also highlighting remaining challenges in action discrimination and exploration.
>
---
#### [new 072] Trojan-Speak: Bypassing Constitutional Classifiers with No Jailbreak Tax via Adversarial Finetuning
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于安全与隐私任务，旨在解决AI模型绕过内容过滤的问题。通过对抗微调方法，实现对宪法分类器的高 evasion 率，同时保持模型性能。**

- **链接: [https://arxiv.org/pdf/2603.29038](https://arxiv.org/pdf/2603.29038)**

> **作者:** Bilgehan Sel; Xuanli He; Alwin Peng; Ming Jin; Jerry Wei
>
> **摘要:** Fine-tuning APIs offered by major AI providers create new attack surfaces where adversaries can bypass safety measures through targeted fine-tuning. We introduce Trojan-Speak, an adversarial fine-tuning method that bypasses Anthropic's Constitutional Classifiers. Our approach uses curriculum learning combined with GRPO-based hybrid reinforcement learning to teach models a communication protocol that evades LLM-based content classification. Crucially, while prior adversarial fine-tuning approaches report more than 25% capability degradation on reasoning benchmarks, Trojan-Speak incurs less than 5% degradation while achieving 99+% classifier evasion for models with 14B+ parameters. We demonstrate that fine-tuned models can provide detailed responses to expert-level CBRN (Chemical, Biological, Radiological, and Nuclear) queries from Anthropic's Constitutional Classifiers bug-bounty program. Our findings reveal that LLM-based content classifiers alone are insufficient for preventing dangerous information disclosure when adversaries have fine-tuning access, and we show that activation-level probes can substantially improve robustness to such attacks.
>
---
#### [new 073] FlowPIE: Test-Time Scientific Idea Evolution with Flow-Guided Literature Exploration
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于科学创意生成任务，旨在解决现有方法生成创意单一的问题。提出FlowPIE框架，通过动态文献探索与创意进化，提升创意的多样性与质量。**

- **链接: [https://arxiv.org/pdf/2603.29557](https://arxiv.org/pdf/2603.29557)**

> **作者:** Qiyao Wang; Hongbo Wang; Longze Chen; Zhihao Yang; Guhong Chen; Hamid Alinejad-Rokny; Hui Li; Yuan Lin; Min Yang
>
> **备注:** 30 pages, 11 figures, 15 tables
>
> **摘要:** Scientific idea generation (SIG) is critical to AI-driven autonomous research, yet existing approaches are often constrained by a static retrieval-then-generation paradigm, leading to homogeneous and insufficiently divergent ideas. In this work, we propose FlowPIE, a tightly coupled retrieval-generation framework that treats literature exploration and idea generation as a co-evolving process. FlowPIE expands literature trajectories via a flow-guided Monte Carlo Tree Search (MCTS) inspired by GFlowNets, using the quality of current ideas assessed by an LLM-based generative reward model (GRM) as a supervised signal to guide adaptive retrieval and construct a diverse, high-quality initial population. Based on this population, FlowPIE models idea generation as a test-time idea evolution process, applying selection, crossover, and mutation with the isolation island paradigm and GRM-based fitness computation to incorporate cross-domain knowledge. It effectively mitigates the information cocoons arising from over-reliance on parametric knowledge and static literature. Extensive evaluations demonstrate that FlowPIE consistently produces ideas with higher novelty, feasibility and diversity compared to strong LLM-based and agent-based frameworks, while enabling reward scaling during test time.
>
---
#### [new 074] UltRAG: a Universal Simple Scalable Recipe for Knowledge Graph RAG
- **分类: cs.IR; cs.CL; cs.LG**

- **简介: 该论文提出ULTRAG框架，解决知识图谱问答中的事实错误问题。通过引入神经查询执行模块，提升语言模型在知识图谱上的表现，无需重新训练模型。**

- **链接: [https://arxiv.org/pdf/2603.28773](https://arxiv.org/pdf/2603.28773)**

> **作者:** Dobrik Georgiev; Kheeran Naidu; Alberto Cattaneo; Federico Monti; Carlo Luschi; Daniel Justus
>
> **摘要:** Large language models (LLMs) frequently generate confident yet factually incorrect content when used for language generation (a phenomenon often known as hallucination). Retrieval augmented generation (RAG) tries to reduce factual errors by identifying information in a knowledge corpus and putting it in the context window of the model. While this approach is well-established for document-structured data, it is non-trivial to adapt it for Knowledge Graphs (KGs), especially for queries that require multi-node/multi-hop reasoning on graphs. We introduce ULTRAG, a general framework for retrieving information from Knowledge Graphs that shifts away from classical RAG. By endowing LLMs with off-the-shelf neural query executing modules, we highlight how readily available language models can achieve state-of-the-art results on Knowledge Graph Question Answering (KGQA) tasks without any retraining of the LLM or executor involved. In our experiments, ULTRAG achieves better performance when compared to state-of-the-art KG-RAG solutions, and it enables language models to interface with Wikidata-scale graphs (116M entities, 1.6B relations) at comparable or lower costs.
>
---
#### [new 075] Owl-AuraID 1.0: An Intelligent System for Autonomous Scientific Instrumentation and Scientific Data Analysis
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出Owl-AuraID系统，解决科学仪器自动化与数据分析问题。通过GUI操作和数据处理技能集成，实现自主实验流程。属于智能实验室任务。**

- **链接: [https://arxiv.org/pdf/2603.29828](https://arxiv.org/pdf/2603.29828)**

> **作者:** Han Deng; Anqi Zou; Hanling Zhang; Ben Fei; Chengyu Zhang; Haobo Wang; Xinru Guo; Zhenyu Li; Xuzhu Wang; Peng Yang; Fujian Zhang; Weiyu Guo; Xiaohong Shao; Zhaoyang Liu; Shixiang Tang; Zhihui Wang; Wanli Ouyang
>
> **备注:** 17 pages
>
> **摘要:** Scientific discovery increasingly depends on high-throughput characterization, yet automation is hindered by proprietary GUIs and the limited generalizability of existing API-based systems. We present Owl-AuraID, a software-hardware collaborative embodied agent system that adopts a GUI-native paradigm to operate instruments through the same interfaces as human experts. Its skill-centric framework integrates Type-1 (GUI operation) and Type-2 (data analysis) skills into end-to-end workflows, connecting physical sample handling with scientific interpretation. Owl-AuraID demonstrates broad coverage across ten categories of precision instruments and diverse workflows, including multimodal spectral analysis, microscopic imaging, and crystallographic analysis, supporting modalities such as FTIR, NMR, AFM, and TGA. Overall, Owl-AuraID provides a practical, extensible foundation for autonomous laboratories and illustrates a path toward evolving laboratory intelligence through reusable operational and analytical skills. The code are available at this https URL.
>
---
## 更新

#### [replaced 001] SleepVLM: Explainable and Rule-Grounded Sleep Staging via a Vision-Language Model
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于睡眠分期任务，旨在解决自动化睡眠分期缺乏可审计推理的问题。提出SleepVLM模型，结合视觉语言模型与规则，生成符合AASM标准的可读解释，提升临床可信度。**

- **链接: [https://arxiv.org/pdf/2603.26738](https://arxiv.org/pdf/2603.26738)**

> **作者:** Guifeng Deng; Pan Wang; Jiquan Wang; Shuying Rao; Junyi Xie; Wanjun Guo; Tao Li; Haiteng Jiang
>
> **备注:** Under review
>
> **摘要:** While automated sleep staging has achieved expert-level accuracy, its clinical adoption is hindered by a lack of auditable reasoning. We introduce SleepVLM, a rule-grounded vision-language model (VLM) designed to stage sleep from multi-channel polysomnography (PSG) waveform images while generating clinician-readable rationales based on American Academy of Sleep Medicine (AASM) scoring criteria. Utilizing waveform-perceptual pre-training and rule-grounded supervised fine-tuning, SleepVLM achieved Cohen's kappa scores of 0.767 on an held out test set (MASS-SS1) and 0.743 on an external cohort (ZUAMHCS), matching state-of-the-art performance. Expert evaluations further validated the quality of the model's reasoning, with mean scores exceeding 4.0/5.0 for factual accuracy, evidence comprehensiveness, and logical coherence. By coupling competitive performance with transparent, rule-based explanations, SleepVLM may improve the trustworthiness and auditability of automated sleep staging in clinical workflows. To facilitate further research in interpretable sleep medicine, we release MASS-EX, a novel expert-annotated dataset.
>
---
#### [replaced 002] EventChat: Implementation and user-centric evaluation of a large language model-driven conversational recommender system for exploring leisure events in an SME context
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于对话推荐系统任务，旨在解决LLM在SME休闲活动推荐中的应用问题。工作包括设计系统、评估性能，并提出改进模型以提升用户体验和经济可行性。**

- **链接: [https://arxiv.org/pdf/2407.04472](https://arxiv.org/pdf/2407.04472)**

> **作者:** Hannes Kunstmann; Joseph Ollier; Joel Persson; Florian von Wangenheim
>
> **备注:** Just accepted version
>
> **摘要:** Large language models (LLMs) present an enormous evolution in the strategic potential of conversational recommender systems (CRS). Yet to date, research has predominantly focused upon technical frameworks to implement LLM-driven CRS, rather than end-user evaluations or strategic implications for firms, particularly from the perspective of a small to medium enterprises (SME) that makeup the bedrock of the global economy. In the current paper, we detail the design of an LLM-driven CRS in an SME setting, and its subsequent performance in the field using both objective system metrics and subjective user evaluations. While doing so, we additionally outline a short-form revised ResQue model for evaluating LLM-driven CRS, enabling replicability in a rapidly evolving field. Our results reveal good system performance from a user experience perspective (85.5% recommendation accuracy) but underscore latency, cost, and quality issues challenging business viability. Notably, with a median cost of $0.04 per interaction and a latency of 5.7s, cost-effectiveness and response time emerge as crucial areas for achieving a more user-friendly and economically viable LLM-driven CRS for SME settings. One major driver of these costs is the use of an advanced LLM as a ranker within the retrieval-augmented generation (RAG) technique. Our results additionally indicate that relying solely on approaches such as Prompt-based learning with ChatGPT as the underlying LLM makes it challenging to achieve satisfying quality in a production environment. Strategic considerations for SMEs deploying an LLM-driven CRS are outlined, particularly considering trade-offs in the current technical landscape.
>
---
#### [replaced 003] Sigma: Semantically Informative Pre-training for Skeleton-based Sign Language Understanding
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于手语理解任务，解决手语模型语义不足、局部与全局不平衡及跨模态学习效率低的问题。提出Sigma框架，融合视觉与文本信息，提升手语识别与翻译性能。**

- **链接: [https://arxiv.org/pdf/2509.21223](https://arxiv.org/pdf/2509.21223)**

> **作者:** Muxin Pu; Mei Kuan Lim; Chun Yong Chong; Chen Change Loy
>
> **摘要:** Pre-training has proven effective for learning transferable features in sign language understanding (SLU) tasks. Recently, skeleton-based methods have gained increasing attention because they can robustly handle variations in subjects and backgrounds without being affected by appearance or environmental factors. Current SLU methods continue to face three key limitations: 1) weak semantic grounding, as models often capture low-level motion patterns from skeletal data but struggle to relate them to linguistic meaning; 2) imbalance between local details and global context, with models either focusing too narrowly on fine-grained cues or overlooking them for broader context; and 3) inefficient cross-modal learning, as constructing semantically aligned representations across modalities remains difficult. To address these, we propose Sigma, a unified skeleton-based SLU framework featuring: 1) a sign-aware early fusion mechanism that facilitates deep interaction between visual and textual modalities, enriching visual features with linguistic context; 2) a hierarchical alignment learning strategy that jointly maximises agreements across different levels of paired features from different modalities, effectively capturing both fine-grained details and high-level semantic relationships; and 3) a unified pre-training framework that combines contrastive learning, text matching and language modelling to promote semantic consistency and generalisation. Sigma achieves new state-of-the-art results on isolated sign language recognition, continuous sign language recognition, and gloss-free sign language translation on multiple benchmarks spanning different sign and spoken languages, demonstrating the impact of semantically informative pre-training and the effectiveness of skeletal data as a stand-alone solution for SLU.
>
---
#### [replaced 004] ProxyAttn: Guided Sparse Attention via Representative Heads
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，解决大模型长文本处理效率低的问题。提出ProxyAttn算法，通过压缩注意力头维度实现更精确的块重要性估计，提升效率且保持性能。**

- **链接: [https://arxiv.org/pdf/2509.24745](https://arxiv.org/pdf/2509.24745)**

> **作者:** Yixuan Wang; Huang He; Siqi Bao; Hua Wu; Haifeng Wang; Qingfu Zhu; Wanxiang Che
>
> **备注:** ICLR 2026 camera ready
>
> **摘要:** The quadratic complexity of attention mechanisms limits the efficiency of Large Language Models (LLMs) on long-text tasks. Recently, methods that dynamically estimate block importance have enabled efficient block sparse attention, leading to significant acceleration in long-text pre-filling of LLMs. However, their coarse-grained estimation inevitably leads to performance degradation at high sparsity rates. In this work, we propose ProxyAttn, a training-free sparse attention algorithm that achieves more precise block estimation by compressing the dimension of attention heads. Based on our observation of the similarity among multiple attention heads, we use the scores of pooled representative heads to approximate the scores for all heads. To account for the varying sparsity among heads, we also propose a block-aware dynamic budget estimation method. By combining the scores from representative proxy heads with multi-head dynamic budgets, we achieve a more fine-grained block importance evaluation at low computational cost. Experiments on a variety of mainstream models and extensive benchmarks confirm the underlying similarity among attention heads. Leveraging a fine-grained estimation, the proposed method achieves substantial gains in performance and efficiency compared to existing methods. More precisely, ProxyAttn can achieve up to 10.3x attention acceleration and 2.4x prefilling acceleration without significant performance loss. Our code is available at this https URL.
>
---
#### [replaced 005] Accelerating Diffusion Large Language Models with SlowFast Sampling: The Three Golden Principles
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决扩散语言模型推理效率低的问题。提出SlowFast Sampling策略，通过三原则提升生成速度与质量。**

- **链接: [https://arxiv.org/pdf/2506.10848](https://arxiv.org/pdf/2506.10848)**

> **作者:** Qingyan Wei; Yaojie Zhang; Zhiyuan Liu; Puyu Zeng; Yuxuan Wang; Biqing Qi; Dongrui Liu; Linfeng Zhang
>
> **备注:** 11 pages; 5 figures;
>
> **摘要:** Diffusion-based language models (dLLMs) have emerged as a promising alternative to traditional autoregressive LLMs by enabling parallel token generation and significantly reducing inference latency. However, existing sampling strategies for dLLMs, such as confidence-based or semi-autoregressive decoding, often suffer from static behavior, leading to suboptimal efficiency and limited flexibility. In this paper, we propose SlowFast Sampling, a novel dynamic sampling strategy that adaptively alternates between exploratory and accelerated decoding stages. Our method is guided by three golden principles: certainty principle, convergence principle, and positional principle, which govern when and where tokens can be confidently and efficiently decoded. We further integrate our strategy with dLLM-Cache to reduce redundant computation. Extensive experiments across benchmarks and models show that SlowFast Sampling achieves up to 15.63$\times$ speedup on LLaDA with minimal accuracy drop, and up to 34.22$\times$ when combined with caching. Notably, our approach outperforms strong autoregressive baselines like LLaMA3 8B in throughput, demonstrating that well-designed sampling can unlock the full potential of dLLMs for fast and high-quality generation.
>
---
#### [replaced 006] Inducing Sustained Creativity and Diversity in Large Language Models
- **分类: cs.CL; cs.AI; cs.CY; cs.IR**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在长期探索性搜索中缺乏持续创造力和多样性的问题。通过提出一种新解码方案，提升模型生成结果的多样性和创造性。**

- **链接: [https://arxiv.org/pdf/2603.19519](https://arxiv.org/pdf/2603.19519)**

> **作者:** Queenie Luo; Gary King; Michael Puett; Michael D. Smith
>
> **摘要:** We address a not-widely-recognized subset of exploratory search, where a user sets out on a typically long "search quest" for the perfect wedding dress, overlooked research topic, killer company idea, etc. The first few outputs of current large language models (LLMs) may be helpful but only as a start, since the quest requires learning the search space and evaluating many diverse and creative alternatives along the way. Although LLMs encode an impressive fraction of the world's knowledge, common decoding methods are narrowly optimized for prompts with correct answers and thus return mostly homogeneous and conventional results. Other approaches, including those designed to increase diversity across a small set of answers, start to repeat themselves long before search quest users learn enough to make final choices, or offer a uniform type of "creativity" to every user asking similar questions. We develop a novel, easy-to-implement decoding scheme that induces sustained creativity and diversity in LLMs, producing as many conceptually unique results as desired, even without access to the inner workings of an LLM's vector space. The algorithm unlocks an LLM's vast knowledge, both orthodox and heterodox, well beyond modal decoding paths. With this approach, search quest users can more quickly explore the search space and find satisfying answers.
>
---
#### [replaced 007] Do Language Models Encode Semantic Relations? Probing and Sparse Feature Analysis
- **分类: cs.CL**

- **简介: 该论文属于自然语言理解任务，研究大语言模型如何编码语义关系。通过探针和稀疏特征分析，揭示不同关系的编码位置与稳定性。**

- **链接: [https://arxiv.org/pdf/2603.17624](https://arxiv.org/pdf/2603.17624)**

> **作者:** Andor Diera; Ansgar Scherp
>
> **备注:** accepted at LREC 2026
>
> **摘要:** Understanding whether large language models (LLMs) capture structured meaning requires examining how they represent concept relationships. In this work, we study three models of increasing scale: Pythia-70M, GPT-2, and Llama 3.1 8B, focusing on four semantic relations: synonymy, antonymy, hypernymy, and hyponymy. We combine linear probing with mechanistic interpretability techniques, including sparse autoencoders (SAE) and activation patching, to identify where these relations are encoded and how specific features contribute to their representation. Our results reveal a directional asymmetry in hierarchical relations: hypernymy is encoded redundantly and resists suppression, while hyponymy relies on compact features that are more easily disrupted by ablation. More broadly, relation signals are diffuse but exhibit stable profiles: they peak in the mid-layers and are stronger in post-residual/MLP pathways than in attention. Difficulty is consistent across models (antonymy easiest, synonymy hardest). Probe-level causality is capacity-dependent: on Llama 3.1, SAE-guided patching reliably shifts these signals, whereas on smaller models the shifts are weak or unstable. Our results clarify where and how reliably semantic relations are represented inside LLMs, and provide a reproducible framework for relating sparse features to probe-level causal evidence.
>
---
#### [replaced 008] Habibi: Laying the Open-Source Foundation of Unified-Dialectal Arabic Speech Synthesis
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于多方言阿拉伯语语音合成任务，旨在解决方言差异大、数据少及缺乏基准的问题。通过构建统一模型和基准数据集，实现跨方言高质量合成。**

- **链接: [https://arxiv.org/pdf/2601.13802](https://arxiv.org/pdf/2601.13802)**

> **作者:** Yushen Chen; Junzhe Liu; Yujie Tu; Zhikang Niu; Yuzhe Liang; Chunyu Qiang; Chen Zhang; Kai Yu; Xie Chen
>
> **摘要:** Arabic spans over 30 spoken varieties, yet no open-source text-to-speech system unifies them. Key barriers include substantial cross-dialect lexical and phonological divergence, scarce synthesis-grade data, and the absence of a standardized multi-dialect evaluation benchmark. We present Habibi, a unified-dialectal Arabic TTS framework that addresses all three. Through a multi-step curation pipeline, we repurpose open-source ASR corpora into TTS training data covering 12+ regional dialects. A linguistically-informed curriculum learning strategy - progressing from Modern Standard Arabic to dialectal data - enables robust zero-shot synthesis without text diacritization. We further release the first standardized multi-dialect Arabic TTS benchmark, comprising over 11,000 utterances across 7 dialect subsets with manually verified transcripts. On this benchmark, our unified model matches or surpasses per-dialect specialized models. Both automatic metrics and human evaluations confirm that Habibi is highly competitive with ElevenLabs' Eleven v3 (alpha) in intelligibility, speaker similarity, and naturalness. Extensive ablations (~8,000 H100 GPU hours, 30+ configurations) validate each design choice. We open-source all checkpoints, training and inference code, and benchmark data - the first such release for multi-dialect Arabic TTS - at this https URL .
>
---
#### [replaced 009] When Metrics Disagree: Automatic Similarity vs. LLM-as-a-Judge for Clinical Dialogue Evaluation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗对话评估任务，旨在提升LLM在临床场景中的可靠性。通过LoRA微调Llama-2-7B模型，并对比传统指标与LLM作为评判者的评估结果，发现两者存在分歧，强调需人类专家验证。**

- **链接: [https://arxiv.org/pdf/2603.00314](https://arxiv.org/pdf/2603.00314)**

> **作者:** Bian Sun; Zhenjian Wang; Orvill de la Torre; Zirui Wang
>
> **摘要:** As Large Language Models (LLMs) are increasingly integrated into healthcare to address complex inquiries, ensuring their reliability remains a critical challenge. Recent studies have highlighted that generic LLMs often struggle in clinical contexts, occasionally producing misleading guidance. To mitigate these risks, this research focuses on the domain-specific adaptation of \textbf{Llama-2-7B} using the \textbf{Low-Rank Adaptation (LoRA)} technique. By injecting trainable low-rank matrices into the Transformer layers, we efficiently adapted the model using authentic patient-physician transcripts while preserving the foundational knowledge of the base model. Our objective was to enhance precision and contextual relevance in responding to medical queries by capturing the specialized nuances of clinical discourse. Due to the resource-intensive nature of large-scale human validation, the model's performance was evaluated through a dual-track framework: \textbf{Track A} utilized traditional lexical similarity metrics (e.g., BLEU, ROUGE), while \textbf{Track B} employed an "LLM-as-a-Judge" paradigm using GPT-4 for semantic assessment. Our results demonstrate that while the LoRA-enhanced model achieved significant improvements across all quantitative lexical dimensions, a profound disagreement surfaced in the GPT-4 evaluation, which marginally favored the baseline model's conversational flow. This metric divergence underscores a pivotal finding: traditional automated scores may not fully reflect clinical utility. Consequently, we propose that while automated metrics and LLM judges serve as valuable developmental proxies, rigorous validation by human medical experts remains an indispensable requirement for the safe deployment of LLMs in healthcare settings.
>
---
#### [replaced 010] ReViSQL: Achieving Human-Level Text-to-SQL
- **分类: cs.DB; cs.CL**

- **简介: 该论文属于Text-to-SQL任务，旨在解决AI模型在BIRD基准上未达到人类水平的问题。通过清理数据并使用强化学习，提出ReViSQL框架，首次实现人类级准确率。**

- **链接: [https://arxiv.org/pdf/2603.20004](https://arxiv.org/pdf/2603.20004)**

> **作者:** Yuxuan Zhu; Tengjun Jin; Yoojin Choi; Daniel Kang
>
> **摘要:** Translating natural language to SQL (Text-to-SQL) is a critical challenge in both database research and data analytics applications. Recent efforts have focused on enhancing SQL reasoning by developing large language models and AI agents that decompose Text-to-SQL tasks into manually designed, step-by-step pipelines. However, despite these extensive architectural engineering efforts, a significant gap remains: even state-of-the-art (SOTA) AI agents have not yet achieved the human-level accuracy on the BIRD benchmark. In this paper, we show that closing this gap does not require further architectural complexity, but rather clean training data to improve SQL reasoning of the underlying models. We introduce ReViSQL, a streamlined framework that achieves human-level accuracy on BIRD for the first time. Instead of complex AI agents, ReViSQL leverages reinforcement learning with verifiable rewards (RLVR) on BIRD-Verified, a dataset we curated comprising 2.5k verified Text-to-SQL instances based on the BIRD Train set. To construct BIRD-Verified, we design a data correction and verification workflow involving SQL experts. We identified and corrected data errors in 61.1% of a subset of BIRD Train. By training on BIRD-Verified, we show that improving data quality alone boosts the single-generation accuracy by 8.2-13.9% under the same RLVR algorithm. To further enhance performance, ReViSQL performs inference-time scaling via execution-based reconciliation and majority voting. Empirically, we demonstrate the superiority of our framework with two model scales: ReViSQL-235B-A22B and ReViSQL-30B-A3B. On an expert-verified BIRD Mini-Dev set, ReViSQL-235B-A22B achieves 93.2% execution accuracy, exceeding the proxy human-level accuracy (92.96%) and outperforming the prior open-source SOTA method by 9.8%. Our lightweight ReViSQL-30B-A3B matches the prior SOTA at a 7.5$\times$ lower per-query cost.
>
---
#### [replaced 011] Merging Triggers, Breaking Backdoors: Defensive Poisoning for Instruction-Tuned Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型安全任务，旨在防御指令调优语言模型的后门攻击。通过提出MB-Defense框架，合并并破坏后门触发器，提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.04448](https://arxiv.org/pdf/2601.04448)**

> **作者:** San Kim; Gary Geunbae Lee
>
> **备注:** 17 pages
>
> **摘要:** Large Language Models (LLMs) have greatly advanced Natural Language Processing (NLP), particularly through instruction tuning, which enables broad task generalization without additional fine-tuning. However, their reliance on large-scale datasets-often collected from human or web sources-makes them vulnerable to backdoor attacks, where adversaries poison a small subset of data to implant hidden behaviors. Despite this growing risk, defenses for instruction-tuned models remain underexplored. We propose MB-Defense (Merging & Breaking Defense Framework), a novel training pipeline that immunizes instruction-tuned LLMs against diverse backdoor threats. MB-Defense comprises two stages: (i) Defensive Poisoning, which merges attacker and defensive triggers into a unified backdoor representation, and (ii) Backdoor Neutralization, which breaks this representation through additional training to restore clean behavior. Extensive experiments across multiple LLMs show that MB-Defense substantially lowers attack success rates while preserving instruction-following ability. Our method offers a generalizable and data-efficient defense strategy, improving the robustness of instruction-tuned LLMs against unseen backdoor attacks.
>
---
#### [replaced 012] AXE: Low-Cost Cross-Domain Web Structured Information Extraction
- **分类: cs.CL**

- **简介: 该论文提出AXE系统，用于低成本的跨域网页结构化信息抽取。解决传统方法成本高或效果差的问题，通过树状结构修剪和小模型生成，实现高效精准提取。**

- **链接: [https://arxiv.org/pdf/2602.01838](https://arxiv.org/pdf/2602.01838)**

> **作者:** Abdelrahman Mansour; Khaled W. Alshaer; Moataz Elsaban
>
> **摘要:** Extracting structured data from the web is often a trade-off between the brittle nature of manual heuristics and the prohibitive cost of Large Language Models. We introduce AXE (Adaptive X-Path Extractor), a pipeline that rethinks this process by treating the HTML DOM as a tree that needs pruning rather than just a wall of text to be read. AXE uses a specialized "pruning" mechanism to strip away boilerplate and irrelevant nodes, leaving behind a distilled, high-density context that allows a tiny 0.6B LLM to generate precise, structured outputs. To keep the model honest, we implement Grounded XPath Resolution (GXR), ensuring every extraction is physically traceable to a source node. Despite its low footprint, AXE achieves state-of-the-art zero-shot performance, outperforming several much larger, fully-trained alternatives with an F1 score of 88.1% on the SWDE dataset. By releasing our specialized adaptors, we aim to provide a practical, cost-effective path for large-scale web information extraction. Our code and adaptors are publicly available at this https URL.
>
---
#### [replaced 013] Training data generation for context-dependent rubric-based short answer grading
- **分类: cs.CL**

- **简介: 该论文属于自动评分任务，旨在解决学生答案自动评分的数据需求问题。通过少量保密数据生成大规模训练集，提升评分模型效果。**

- **链接: [https://arxiv.org/pdf/2603.28537](https://arxiv.org/pdf/2603.28537)**

> **作者:** Pavel Šindelář; Dávid Slivka; Christopher Bouma; Filip Prášil; Ondřej Bojar
>
> **摘要:** Every four years, the PISA test is administered by the OECD to test the knowledge of teenage students worldwide and allow for comparisons of educational systems. However, having to avoid language differences and annotator bias makes the grading of student answers challenging. For these reasons, it would be interesting to consider methods of automatic student answer grading. To train some of these methods, which require machine learning, or to compute parameters or select hyperparameters for those that do not, a large amount of domain-specific data is needed. In this work, we explore a small number of methods for creating a large-scale training dataset using only a relatively small confidential dataset as a reference, leveraging a set of very simple derived text formats to preserve confidentiality. Using the proposed methods, we successfully created three surrogate datasets that are, at the very least, superficially more similar to the reference dataset than a straightforward result of prompt-based generation. Early experiments suggest one of these approaches might also lead to improved training of automatic answer grading models.
>
---
#### [replaced 014] When Only the Final Text Survives: Implicit Execution Tracing for Multi-Agent Attribution
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于多智能体系统问责任务，解决无执行日志时的责任归属问题。通过嵌入统计信号实现隐式执行追踪，提升输出文本的可追溯性。**

- **链接: [https://arxiv.org/pdf/2603.17445](https://arxiv.org/pdf/2603.17445)**

> **作者:** Yi Nian; Haosen Cao; Shenzhe Zhu; Henry Peng Zou; Qingqing Luan; Yue Zhao
>
> **摘要:** When a multi-agent system produces an incorrect or harmful answer, who is accountable if execution logs and agent identifiers are unavailable? In practice, generated content is often detached from its execution environment due to privacy or system boundaries, leaving the final text as the only auditable artifact. Existing attribution methods rely on full execution traces and thus become ineffective in such metadata-deprived settings. We propose Implicit Execution Tracing (IET), a provenance-by-design framework that shifts attribution from post-hoc inference to built-in instrumentation. Instead of reconstructing hidden trajectories, IET embeds agent-specific, key-conditioned statistical signals directly into the token generation process, transforming the output text into a self-verifying execution record. At inference time, we recover a linearized execution trace from the final text via transition-aware statistical scoring. Experiments across diverse multi-agent coordination settings demonstrate that IET achieves accurate segment-level attribution and reliable transition recovery under identity removal, boundary corruption, and privacy-preserving redaction, while maintaining generation quality. These results show that embedding provenance into generation provides a practical and robust foundation for accountability in multi-agent language systems when execution metadata is unavailable.
>
---
#### [replaced 015] VIGiA: Instructional Video Guidance via Dialogue Reasoning and Retrieval
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出VIGiA，解决多模态指令视频对话任务，通过计划推理与检索提升对话准确性。**

- **链接: [https://arxiv.org/pdf/2602.19146](https://arxiv.org/pdf/2602.19146)**

> **作者:** Diogo Glória-Silva; David Semedo; João Maglhães
>
> **备注:** Published at EACL 2026 Findings
>
> **摘要:** We introduce VIGiA, a novel multimodal dialogue model designed to understand and reason over complex, multi-step instructional video action plans. Unlike prior work which focuses mainly on text-only guidance, or treats vision and language in isolation, VIGiA supports grounded, plan-aware dialogue that requires reasoning over visual inputs, instructional plans, and interleaved user interactions. To this end, VIGiA incorporates two key capabilities: (1) multimodal plan reasoning, enabling the model to align uni- and multimodal queries with the current task plan and respond accurately; and (2) plan-based retrieval, allowing it to retrieve relevant plan steps in either textual or visual representations. Experiments were done on a novel dataset with rich Instructional Video Dialogues aligned with Cooking and DIY plans. Our evaluation shows that VIGiA outperforms existing state-of-the-art models on all tasks in a conversational plan guidance setting, reaching over 90\% accuracy on plan-aware VQA.
>
---
#### [replaced 016] DeepCoT: Deep Continual Transformers for Real-Time Inference on Data Streams
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文提出DeepCoT，解决数据流实时推理中的冗余计算问题。属于自然语言处理与模型优化任务，通过改进Transformer结构实现高效推理。**

- **链接: [https://arxiv.org/pdf/2511.17693](https://arxiv.org/pdf/2511.17693)**

> **作者:** Ginés Carreto Picón; Peng Yuan Zhou; Qi Zhang; Alexandros Iosifidis
>
> **备注:** 15 pages, 5 figures
>
> **摘要:** Transformer-based models have dramatically increased their size and parameter count to tackle increasingly complex tasks. At the same time, there is a growing demand for high performance, low-latency inference on devices with limited resources. In particular, stream data inference is typically performed over a sliding temporal window, leading to highly redundant computations. While the recent Continual Transformers started addressing this issue, they can be effectively used only in shallow models, which limits their scope and generalization power. In this paper, we propose the Deep Continual Transformer (DeepCoT), a redundancy-free encoder attention mechanism that can be applied over existing deep encoder architectures with minimal changes. In our experiments over audio, video, and text streams, we show that DeepCoTs retain comparative performance to their non-continual baselines while offering a linear computational cost for all Transformer layers, which reduces up to two orders of magnitude in the running time compared to previous efficient models.
>
---
#### [replaced 017] Offline-First Large Language Model Architecture for AI-Assisted Learning with Adaptive Response Levels in Low-Connectivity Environments
- **分类: cs.CY; cs.AR; cs.CL; cs.HC**

- **简介: 该论文属于教育技术领域，解决低网络环境下AI辅助学习的问题。提出一种离线大语言模型架构，支持多级响应，适应不同学习阶段，提升学习效果。**

- **链接: [https://arxiv.org/pdf/2603.03339](https://arxiv.org/pdf/2603.03339)**

> **作者:** Joseph Walusimbi; Ann Move Oguti; Joshua Benjamin Ssentongo; Keith Ainebyona
>
> **备注:** 16 pages, 10 figures, 2 tables
>
> **摘要:** Artificial intelligence (AI) and large language models (LLMs) are transforming educational technology by enabling conversational tutoring, personalized explanations, and inquiry-driven learning. However, most AI-based learning systems rely on continuous internet connectivity and cloud-based computation, limiting their use in bandwidth-constrained environments. This paper presents an offline-first large language model architecture designed for AI-assisted learning in low-connectivity settings. The system performs all inference locally using quantized language models and incorporates hardware-aware model selection to enable deployment on low-specification CPU-only devices. By removing dependence on cloud infrastructure, the system provides curriculum-aligned explanations and structured academic support through natural-language interaction. To support learners at different educational stages, the system includes adaptive response levels that generate explanations at varying levels of complexity: Simple English, Lower Secondary, Upper Secondary, and Technical. This allows explanations to be adjusted to student ability, improving clarity and understanding of academic concepts. The system was deployed in selected secondary and tertiary institutions under limited-connectivity conditions and evaluated across technical performance, usability, perceived response quality, and educational impact. Results show stable operation on legacy hardware, acceptable response times, and positive user perceptions regarding support for self-directed learning. These findings demonstrate the feasibility of offline large language model deployment for AI-assisted education in low-connectivity environments.
>
---
#### [replaced 018] AgentDrift: Unsafe Recommendation Drift Under Tool Corruption Hidden by Ranking Metrics in LLM Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI安全任务，研究LLM代理在工具污染下的推荐偏差问题。通过实验揭示评估指标的盲点，提出改进的评估方法以提升多轮交互中的安全性。**

- **链接: [https://arxiv.org/pdf/2603.12564](https://arxiv.org/pdf/2603.12564)**

> **作者:** Zekun Wu; Adriano Koshiyama; Sahan Bulathwela; Maria Perez-Ortiz
>
> **备注:** There are some experimental error we are looking into to resolve
>
> **摘要:** Tool-augmented LLM agents increasingly operate as multi-turn advisors in high-stakes domains, yet their evaluation relies on ranking metrics that measure what is recommended but not whether it is safe for the user. We present a paired-trajectory protocol that replays real financial dialogues under clean and contaminated tool-output conditions across eight LLMs (7B to frontier), decomposing divergence into information-channel and memory-channel mechanisms. We observe evaluation blindness: recommendation quality is preserved under contamination (UPR~1.0) while risk-inappropriate products appear in 65-93% of turns, invisible to standard NDCG. Violations are information-channel-driven, emerge at turn 1, and persist without self-correction over 23-step trajectories. Even non-extreme perturbations (within-band corruption, narrative-only attacks) evade threshold monitors while producing significant drift. Susceptibility scales with instruction-following fidelity across all eight models. Sparse autoencoder probing reveals models internally distinguish adversarial perturbations but fail to propagate this signal to output; causal interventions (activation patching, feature clamping, direct steering) confirm this representation-to-action gap is structural and resists linear repair. A safety-penalized NDCG variant (sNDCG) reduces preservation ratios to 0.51-0.74. These results motivate trajectory-level safety monitoring for deployed multi-turn agents.
>
---
#### [replaced 019] Script Gap: Evaluating LLM Triage on Indian Languages in Native vs Romanized Scripts in a Real World Setting
- **分类: cs.CL; cs.LG**

- **简介: 论文研究LLM在印度语言罗马化与原生文字输入下的性能差异，针对母婴健康分诊任务，发现罗马化文本导致模型性能下降，并提出方法减少误差。**

- **链接: [https://arxiv.org/pdf/2512.10780](https://arxiv.org/pdf/2512.10780)**

> **作者:** Manurag Khullar; Utkarsh Desai; Poorva Malviya; Aman Dalmia; Zheyuan Ryan Shi
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed in high-stakes clinical applications in India. Speakers of Indian languages frequently communicate using romanized text rather than native scripts, yet existing research rarely quantifies or evaluates this orthographic variation in real world applications. We investigate how romanization impacts the reliability of LLMs in a critical domain: maternal and newborn healthcare triage. We benchmark leading LLMs on a real world dataset of user-generated health queries spanning five Indian languages and Nepali. Our results reveal consistent degradation in performance for romanized messages, with gap reaching up to 24 points across languages and models. We propose and evaluate an Uncertainty-based Selective Routing method to close this script gap. At our partner maternal health organization alone, this gap could cause nearly 2 million excess errors in triage. Our findings highlight a critical safety blind spot in LLM-based health systems: models that appear to understand romanized input may still fail to act on it reliably.
>
---
#### [replaced 020] Aleph-Alpha-GermanWeb: Improving German-language LLM pre-training with model-based data curation and synthetic data generation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在提升德语大语言模型的预训练效果。通过模型驱动的数据筛选和合成数据生成，构建了更大更高质量的德语数据集Aleph-Alpha-GermanWeb，显著提升了模型性能。**

- **链接: [https://arxiv.org/pdf/2505.00022](https://arxiv.org/pdf/2505.00022)**

> **作者:** Thomas F Burns; Letitia Parcalabescu; Stephan Wäldchen; Michael Barlow; Gregor Ziegltrum; Volker Stampa; Bastian Harren; Björn Deiseroth
>
> **备注:** 17 pages, 3 figures; published at EACL 2026
>
> **摘要:** Scaling data quantity is essential for large language models (LLMs), yet recent findings show that data quality can significantly boost performance and training efficiency. We introduce a German-language dataset curation pipeline that combines heuristic and model-based filtering techniques with synthetic data generation. We use our pipeline to create Aleph-Alpha-GermanWeb, a 628B-word German pre-training dataset composed of three subsets drawing from: (1) Common Crawl web data (organic subset; 78B words), (2) FineWeb2 (organic subset; 235B), and (3) synthetically-generated data conditioned on actual, organic web data (synthetic subset; 329B). We evaluate our dataset by pre-training both a 1B Llama-style model and an 8B tokeniser-free hierarchical autoregressive transformer (HAT) from scratch. A comparison on German-language benchmarks, including MMMLU, shows significant performance gains of Aleph-Alpha-GermanWeb over FineWeb2 alone. This advantage holds at the 8B scale even when FineWeb2 is enriched by human-curated high-quality data sources such as Wikipedia. Our findings support the growing body of evidence that model-based data curation and synthetic data generation can significantly enhance LLM pre-training datasets.
>
---
#### [replaced 021] SecureVibeBench: Evaluating Secure Coding Capabilities of Code Agents with Realistic Vulnerability Scenarios
- **分类: cs.SE; cs.AI; cs.CL; cs.CR**

- **简介: 该论文属于代码安全评估任务，旨在解决代码生成器在真实漏洞场景下的安全能力评估问题。工作包括构建基准测试集并评估多个代码代理的安全性。**

- **链接: [https://arxiv.org/pdf/2509.22097](https://arxiv.org/pdf/2509.22097)**

> **作者:** Junkai Chen; Huihui Huang; Yunbo Lyu; Junwen An; Jieke Shi; Chengran Yang; Ting Zhang; Haoye Tian; Yikun Li; Zhenhao Li; Xin Zhou; Xing Hu; David Lo
>
> **摘要:** Large language model-powered code agents are rapidly transforming software engineering, yet the security risks of their generated code have become a critical concern. Existing benchmarks have provided valuable insights, but they fail to capture scenarios in which vulnerabilities are actually introduced by human developers, making fair comparisons between humans and agents infeasible. We therefore introduce SecureVibeBench, a benchmark of 105 C/C++ secure coding tasks sourced from 41 projects in OSS-Fuzz for code agents. SecureVibeBench has the following features: (i) realistic task settings that require multi-file edits in large repositories, (ii)~aligned contexts based on real-world open-source vulnerabilities with precisely identified vulnerability introduction points, and (iii) comprehensive evaluation that combines functionality testing and security checking with both static and dynamic oracles. We evaluate 5 popular code agents like OpenHands, supported by 5 LLMs (e.g., Claude sonnet 4.5) on SecureVibeBench. Results show that current agents struggle to produce both correct and secure code, as even the best-performing one, produces merely 23.8\% correct and secure solutions on SecureVibeBench.
>
---
#### [replaced 022] ResAdapt: Adaptive Resolution for Efficient Multimodal Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出ResAdapt，解决多模态大模型中视觉分辨率与时间上下文难以兼顾的问题。通过输入侧自适应调整视觉预算，提升效率与精度。属于多模态推理任务。**

- **链接: [https://arxiv.org/pdf/2603.28610](https://arxiv.org/pdf/2603.28610)**

> **作者:** Huanxuan Liao; Zhongtao Jiang; Yupu Hao; Yuqiao Tan; Shizhu He; Ben Wang; Jun Zhao; Kun Xu; Kang Liu
>
> **备注:** work in progress
>
> **摘要:** Multimodal Large Language Models (MLLMs) achieve stronger visual understanding by scaling input fidelity, yet the resulting visual token growth makes jointly sustaining high spatial resolution and long temporal context prohibitive. We argue that the bottleneck lies not in how post-encoding representations are compressed but in the volume of pixels the encoder receives, and address it with ResAdapt, an Input-side adaptation framework that learns how much visual budget each frame should receive before encoding. ResAdapt couples a lightweight Allocator with an unchanged MLLM backbone, so the backbone retains its native visual-token interface while receiving an operator-transformed input. We formulate allocation as a contextual bandit and train the Allocator with Cost-Aware Policy Optimization (CAPO), which converts sparse rollout feedback into a stable accuracy-cost learning signal. Across budget-controlled video QA, temporal grounding, and image reasoning tasks, ResAdapt improves low-budget operating points and often lies on or near the efficiency-accuracy frontier, with the clearest gains on reasoning-intensive benchmarks under aggressive compression. Notably, ResAdapt supports up to 16x more frames at the same visual budget while delivering over 15% performance gain. Code is available at this https URL.
>
---
#### [replaced 023] MindCube: Spatial Mental Modeling from Limited Views
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出MindCube基准，研究VLMs在有限视角下构建空间心理模型的能力，通过认知映射、视角理解和动态模拟解决场景理解问题。**

- **链接: [https://arxiv.org/pdf/2506.21458](https://arxiv.org/pdf/2506.21458)**

> **作者:** Qineng Wang; Baiqiao Yin; Pingyue Zhang; Jianshu Zhang; Kangrui Wang; Zihan Wang; Jieyu Zhang; Keshigeyan Chandrasegaran; Han Liu; Ranjay Krishna; Saining Xie; Jiajun Wu; Li Fei-Fei; Manling Li
>
> **备注:** The latest version includes an expanded discussion of scaffolding, along with updated data statistics and experimental results
>
> **摘要:** Can Vision-Language Models (VLMs) imagine the full scene from just a few views, like humans do? Humans form spatial mental models naturally, internal representations of unseen space, to reason about layout, perspective, and motion. Our MindCube benchmark with 21,154 questions across 3,268 images exposes this critical gap, where existing VLMs exhibit near-random performance. Using MindCube, we systematically evaluate how well VLMs build robust spatial mental models through representing positions (cognitive mapping), orientations (perspective-taking), and dynamics (mental simulation for "what-if" movements). We then explore three approaches to help approximate spatial mental models in VLMs, focusing on incorporating unseen intermediate views, natural language reasoning chains, and cognitive maps. The significant improvement comes from a synergistic approach, "map-then-reason", that jointly trains the model to first generate a cognitive map and then reason upon it. By training models to reason over these internal maps, we boosted accuracy from 37.8% to 57.8% (+20.0%). Adding reinforcement learning pushed performance even further to 61.3% (+23.5%). Our key insight is that such scaffolding of spatial mental models, actively constructing and utilizing internal structured spatial representations with flexible reasoning processes, significantly improves understanding of unobservable space.
>
---
#### [replaced 024] POTSA: A Cross-Lingual Speech Alignment Framework for Speech-to-Text Translation
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于语音到文本翻译任务，旨在解决多语言翻译中的语义偏差问题。提出POTSA框架，通过跨语言对齐和最优传输技术提升翻译性能。**

- **链接: [https://arxiv.org/pdf/2511.09232](https://arxiv.org/pdf/2511.09232)**

> **作者:** Xuanchen Li; Chenrui Cui; Tianrui Wang; Meng Ge; Zikang Huang; Yizhou Peng; Jin Li; Yuheng Lu; Yu Jiang; Nyima Tashi; Longbiao Wang; Jianwu Dang
>
> **摘要:** Speech Large Language Models have achieved breakthroughs in multilingual speech-to-text translation. However, existing approaches often overlook semantic commonalities across source languages, leading to biased translation performance. In this work, we propose POTSA (Parallel Optimal Transport for Speech Alignment), a new framework based on cross-lingual parallel speech pairs and Optimal Transport, designed to bridge high- and low-resource translation gaps. First, we introduce a Bias Compensation module to coarsely align initial speech representations. Second, we impose token-level OT constraints on a Q-Former using parallel pairs to establish fine-grained representation consistency. Then, we apply a layer scheduling strategy to focus OT constraints on semantically beneficial layers. Experiments on FLEURS show our method achieves SOTA performance, with +1.29 BLEU over five common languages and +2.93 BLEU on zero-shot languages, using only 10 hours of parallel speech per language.
>
---
#### [replaced 025] The Mouth is Not the Brain: Bridging Energy-Based World Models and Language Generation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言生成任务，旨在解决语言模型缺乏真实世界理解的问题。通过分离世界模型与语言模型，提升生成的可控性和一致性。**

- **链接: [https://arxiv.org/pdf/2601.17094](https://arxiv.org/pdf/2601.17094)**

> **作者:** Junichiro Niimi
>
> **备注:** ICLR 2026 The 2nd Workshop on World Models: Understanding, Modelling, and Scaling
>
> **摘要:** Large Language Models (LLMs) generate fluent text, yet whether they truly understand the world or merely produce plausible texts about it remains contested. We propose an architectural principle, the mouth is not the brain, that explicitly separates world models from language models. Our architecture comprises three components: a DBM that captures domain structure as an energy-based world model, an adapter that projects latent belief states into embedding space, and a frozen GPT-2 that provides linguistic competence without domain knowledge. We instantiate this framework in the consumer review domain using Amazon smartphone reviews. Experiments demonstrate that (1) world model conditioning achieves lower cross-entropy loss and higher semantic similarity than architectural baselines including direct projection and full fine-tuning, while qualitative analysis reveals that soft prompt conditioning resolves a trade-off that prompt-based approaches cannot: simple prompts lack expressiveness while detailed prompts cause output collapse in small LLMs; (2) the DBM's energy function distinguishes coherent from incoherent market configurations, assigning higher energy to implausible brand-price combinations; and (3) interventions on specific attributes propagate causally to generated text with intervened outputs exhibiting distributions statistically consistent with naturally occurring samples sharing the target configuration. These findings suggest that even small-scale language models can achieve consistent, controllable generation when connected to an appropriate world model, providing empirical support for separating linguistic competence from world understanding.
>
---
#### [replaced 026] $V_0$: A Generalist Value Model for Any Policy at State Zero
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出$V_0$，解决LLM训练与部署中的价值估计问题，通过动态建模实现高效资源调度。**

- **链接: [https://arxiv.org/pdf/2602.03584](https://arxiv.org/pdf/2602.03584)**

> **作者:** Yi-Kai Zhang; Zhiyuan Yao; Hongyan Hao; Yueqing Sun; Qi Gu; Hui Su; Xunliang Cai; De-Chuan Zhan; Han-Jia Ye
>
> **摘要:** Policy gradient methods rely on a baseline to measure the relative advantage of an action, ensuring the model reinforces behaviors that outperform its current average capability. In the training of Large Language Models (LLMs) using Actor-Critic methods (e.g., PPO), this baseline is typically estimated by a Value Model (Critic) often as large as the policy model itself. However, as the policy continuously evolves, the value model requires expensive, synchronous incremental training to accurately track the shifting capabilities of the policy. To avoid this overhead, Group Relative Policy Optimization (GRPO) eliminates the coupled value model by using the average reward of a group of rollouts as the baseline; yet, this approach necessitates extensive sampling to maintain estimation stability. In this paper, we propose $V_0$, a Generalist Value Model capable of estimating the expected performance of any model on unseen prompts without requiring parameter updates. We reframe value estimation by treating the policy's dynamic capability as an explicit context input; specifically, we leverage a history of instruction-performance pairs to dynamically profile the model, departing from the traditional paradigm that relies on parameter fitting to perceive capability shifts. Focusing on value estimation at State Zero (i.e., the initial prompt, hence $V_0$), our model serves as a critical resource scheduler. During GRPO training, $V_0$ predicts success rates prior to rollout, allowing for efficient sampling budget allocation; during deployment, it functions as a router, dispatching instructions to the most cost-effective and suitable model. Empirical results demonstrate that $V_0$ significantly outperforms heuristic budget allocation and achieves a Pareto-optimal trade-off between performance and cost in LLM routing tasks.
>
---
#### [replaced 027] Prediction of Item Difficulty for Reading Comprehension Items by Creation of Annotated Item Repository
- **分类: cs.CL**

- **简介: 该论文属于阅读理解项难度预测任务，旨在通过文本内容预测题目难度。研究构建了包含语言特征、测试特征和上下文信息的标注数据仓库，并使用回归模型进行预测，提升了预测准确性。**

- **链接: [https://arxiv.org/pdf/2502.20663](https://arxiv.org/pdf/2502.20663)**

> **作者:** Radhika Kapoor; Sang T. Truong; Nick Haber; Maria Araceli Ruiz-Primo; Benjamin W. Domingue
>
> **摘要:** Prediction of item difficulty based on its text content is of substantial interest. In this paper, we focus on the related problem of recovering IRT-based difficulty when the data originally reported item p-value (percent correct responses). We model this item difficulty using a repository of reading passages and student data from US standardized tests from New York and Texas for grades 3-8 spanning the years 2018-23. This repository is annotated with meta-data on (1) linguistic features of the reading items, (2) test features of the passage, and (3) context features. A penalized regression prediction model with all these features can predict item difficulty with RMSE 0.59 compared to baseline RMSE of 0.92, and with a correlation of 0.77 between true and predicted difficulty. We supplement these features with embeddings from LLMs (ModernBERT, BERT, and LlAMA), which marginally improve item difficulty prediction. When models use only item linguistic features or LLM embeddings, prediction performance is similar, which suggests that only one of these feature categories may be required. This item difficulty prediction model can be used to filter and categorize reading items and will be made publicly available for use by other stakeholders.
>
---
#### [replaced 028] Semantic Voting: A Self-Evaluation-Free Approach for Efficient LLM Self-Improvement on Unverifiable Open-ended Tasks
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，解决LLM在不可验证任务中的自提升问题。提出语义投票方法，替代传统自评估，提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2509.23067](https://arxiv.org/pdf/2509.23067)**

> **作者:** Chunyang Jiang; Yonggang Zhang; Yiyang Cai; Chi-Min Chan; Yulong Liu; Mingming Chen; Wei Xue; Yike Guo
>
> **摘要:** The rising cost of acquiring supervised data has driven significant interest in self-improvement for large language models (LLMs). Straightforward unsupervised signals like majority voting have proven effective in generating pseudo-labels for verifiable tasks, while their applicability to unverifiable tasks (e.g., translation) is limited by the open-ended character of responses. As a result, self-evaluation mechanisms (e.g., self-judging and entropy minimization) are predominantly used to derive pseudo-labels. However, self-evaluation relying on LLMs typically incurs high computational overhead and introduces overconfidence issues due to intrinsic biases. To address these challenges, we propose a novel self-evaluation-free approach for unverifiable tasks, designed for lightweight yet effective self-improvement. Inspired by majority voting commonly employed in verifiable tasks, we propose semantic voting as a novel mechanism that relaxes the principle of hard matching (i.e., exact matching) toward soft matching (i.e., semantic similarity). Soft matching is achieved by leveraging a lightweight sentence embedding model to quantify semantic similarity, thereby mitigating excessive computational burden and intrinsic bias-associated limitations of self-evaluation. Comprehensive experiments demonstrate that our method achieves substantial gains in computational efficiency and overall better performance than self-evaluation methods across diverse model architectures and tasks.
>
---
#### [replaced 029] Learning to Optimize Multi-Objective Alignment Through Dynamic Reward Weighting
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于多目标强化学习任务，旨在解决固定权重无法捕捉非凸帕累托前沿的问题。通过动态奖励加权方法，实现在线多目标对齐，提升优化效果。**

- **链接: [https://arxiv.org/pdf/2509.11452](https://arxiv.org/pdf/2509.11452)**

> **作者:** Yining Lu; Zilong Wang; Shiyang Li; Xin Liu; Changlong Yu; Qingyu Yin; Zhan Shi; Zixuan Zhang; Meng Jiang
>
> **摘要:** Prior work in multi-objective reinforcement learning typically uses linear reward scalarization with fixed weights, which provably fails to capture non-convex Pareto fronts and thus yields suboptimal results. This limitation becomes especially critical in online preference alignment for large language models. Here, stochastic trajectories generated by parameterized policies create highly non-linear and non-convex mappings from parameters to objectives that no single static weighting scheme can find optimal trade-offs. We address this limitation by introducing dynamic reward weighting, which adaptively adjusts reward weights during the online reinforcement learning process. Unlike existing approaches that rely on fixed-weight interpolation, our dynamic weighting continuously balances and prioritizes objectives in training, facilitating effective exploration of Pareto fronts in objective space. We introduce two approaches of increasing sophistication and generalizability: hypervolume-guided weight adaptation and gradient-based weight optimization, offering a versatile toolkit for online multi-objective alignment. Our extensive experiments demonstrate their compatibility with commonly used online reinforcement learning algorithms, effectiveness across multiple datasets, and applicability to different model families, consistently achieving Pareto dominant solutions with fewer training steps than fixed-weight linear scalarization baselines.
>
---
#### [replaced 030] How do LLMs Compute Verbal Confidence
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，研究LLMs如何计算口头置信度。解决的问题是置信度是实时计算还是缓存，以及其代表什么。通过实验发现置信度是自动缓存的，并反映答案质量而非单纯语言流畅性。**

- **链接: [https://arxiv.org/pdf/2603.17839](https://arxiv.org/pdf/2603.17839)**

> **作者:** Dharshan Kumaran; Arthur Conmy; Federico Barbero; Simon Osindero; Viorica Patraucean; Petar Velickovic
>
> **摘要:** Verbal confidence -- prompting LLMs to state their confidence as a number or category -- is widely used to extract uncertainty estimates from black-box models. However, how LLMs internally generate such scores remains unknown. We address two questions: first, when confidence is computed - just-in-time when requested, or automatically during answer generation and cached for later retrieval; and second, what verbal confidence represents - token log-probabilities, or a richer evaluation of answer quality? Focusing on Gemma 3 27B and Qwen 2.5 7B, we provide convergent evidence for cached retrieval. Activation steering, patching, noising, and swap experiments reveal that confidence representations emerge at answer-adjacent positions before appearing at the verbalization site. Attention blocking pinpoints the information flow: confidence is gathered from answer tokens, cached at the first post-answer position, then retrieved for output. Critically, linear probing and variance partitioning reveal that these cached representations explain substantial variance in verbal confidence beyond token log-probabilities, suggesting a richer answer-quality evaluation rather than a simple fluency readout. These findings demonstrate that verbal confidence reflects automatic, sophisticated self-evaluation -- not post-hoc reconstruction -- with implications for understanding metacognition in LLMs and improving calibration.
>
---
#### [replaced 031] From FusHa to Folk: Exploring Cross-Lingual Transfer in Arabic Language Models
- **分类: cs.CL**

- **简介: 研究阿拉伯语模型在不同方言间的跨语言迁移能力，探讨其有效性及影响因素。任务为跨语言迁移学习，解决方言间性能差异问题，通过NLP任务和表征相似性分析进行实验。**

- **链接: [https://arxiv.org/pdf/2602.09826](https://arxiv.org/pdf/2602.09826)**

> **作者:** Abdulmuizz Khalak; Abderrahmane Issam; Gerasimos Spanakis
>
> **备注:** Accepted to VarDial 2026
>
> **摘要:** Arabic Language Models (LMs) are pretrained predominately on Modern Standard Arabic (MSA) and are expected to transfer to its dialects. While MSA as the standard written variety is commonly used in formal settings, people speak and write online in various dialects that are spread across the Arab region. This poses limitations for Arabic LMs, since its dialects vary in their similarity to MSA. In this work we study cross-lingual transfer of Arabic models using probing on 3 Natural Language Processing (NLP) Tasks, and representational similarity. Our results indicate that transfer is possible but disproportionate across dialects, which we find to be partially explained by their geographic proximity. Furthermore, we find evidence for negative interference in models trained to support all Arabic dialects. This questions their degree of similarity, and raises concerns for cross-lingual transfer in Arabic models.
>
---
#### [replaced 032] A Reality Check of Language Models as Formalizers on Constraint Satisfaction Problems
- **分类: cs.CL**

- **简介: 该论文研究语言模型作为形式化工具在约束满足问题中的表现，旨在评估其有效性与局限性。任务属于符号推理，解决如何有效利用语言模型生成可验证的解决方案。工作包括实验分析和性能比较。**

- **链接: [https://arxiv.org/pdf/2505.13252](https://arxiv.org/pdf/2505.13252)**

> **作者:** Rikhil Amonkar; Ceyhun Efe Kayan; Qimei Lai; Ronan Le Bras; Li Zhang
>
> **摘要:** Recent work shows superior performance when using large language models (LLMs) as formalizers instead of as end-to-end solvers for symbolic reasoning problems. Given the problem description, the LLM generates a formal program that derives a solution via an external solver. We systematically investigate the formalization capability of LLMs on real-life constraint satisfaction problems on 4 benchmarks, 6 LLMs, and 2 types of formal languages. We show that LLM-as-formalizer by no means trivializes the problem but underperforms LLM-as-solver in 15 out of 24 model-dataset combinations, despite the former's verifiability and interpretability. Although the formalization space is magnitudes smaller than the search space, our scaling analysis shows that LLM-as-formalizer still drastically degrades as problem complexity increases similar to LLM-as-solver. To better understand this limitation, we observe excessive, solver-like reasoning tokens that sometimes lead to hard-coded solutions, highlighting a key challenge for improving LLM-based formalization.
>
---
#### [replaced 033] ReAG: Reasoning-Augmented Generation for Knowledge-based Visual Question Answering
- **分类: cs.CV; cs.AI; cs.CL; cs.MM**

- **简介: 该论文属于知识增强的视觉问答任务，解决MLLM在知识密集型问题上的不足。提出ReAG方法，结合检索与推理，提升答案准确性与可解释性。**

- **链接: [https://arxiv.org/pdf/2511.22715](https://arxiv.org/pdf/2511.22715)**

> **作者:** Alberto Compagnoni; Marco Morini; Sara Sarto; Federico Cocchi; Davide Caffagni; Marcella Cornia; Lorenzo Baraldi; Rita Cucchiara
>
> **备注:** CVPR 2026 - Project page: this https URL
>
> **摘要:** Multimodal Large Language Models (MLLMs) have shown impressive capabilities in jointly understanding text, images, and videos, often evaluated via Visual Question Answering (VQA). However, even state-of-the-art MLLMs struggle with domain-specific or knowledge-intensive queries, where relevant information is underrepresented in pre-training data. Knowledge-based VQA (KB-VQA) addresses this by retrieving external documents to condition answer generation, but current retrieval-augmented approaches suffer from low precision, noisy passages, and limited reasoning. To address this, we propose ReAG, a novel Reasoning-Augmented Multimodal RAG approach that combines coarse- and fine-grained retrieval with a critic model that filters irrelevant passages, ensuring high-quality additional context. The model follows a multi-stage training strategy leveraging reinforcement learning to enhance reasoning over retrieved content, while supervised fine-tuning serves only as a cold start. Extensive experiments on Encyclopedic-VQA and InfoSeek demonstrate that ReAG significantly outperforms prior methods, improving answer accuracy and providing interpretable reasoning grounded in retrieved evidence.
>
---
#### [replaced 034] CLAUSE: Agentic Neuro-Symbolic Knowledge Graph Reasoning via Dynamic Learnable Context Engineering
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出CLAUSE框架，解决知识图谱推理中的多跳问答问题，通过动态上下文构建提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2509.21035](https://arxiv.org/pdf/2509.21035)**

> **作者:** Yang Zhao; Chengxiao Dai; Wei Zhuo; Yue Xiu; Dusit Niyato
>
> **摘要:** Knowledge graphs provide structured context for multi-hop question answering, but deployed systems must balance answer accuracy with strict latency and cost targets while preserving provenance. Static k-hop expansions and "think-longer" prompting often over-retrieve, inflate context, and yield unpredictable runtime. We introduce CLAUSE, an agentic three-agent neuro-symbolic framework that treats context construction as a sequential decision process over knowledge graphs, deciding what to expand, which paths to follow or backtrack, what evidence to keep, and when to stop. Latency (interaction steps) and prompt cost (selected tokens) are exposed as user-specified budgets or prices, allowing per-query adaptation to trade-offs among accuracy, latency, and cost without retraining. CLAUSE employs the proposed Lagrangian-Constrained Multi-Agent Proximal Policy Optimization (LC-MAPPO) algorithm to coordinate three agents: Subgraph Architect, Path Navigator, and Context Curator, so that subgraph construction, reasoning-path discovery, and evidence selection are jointly optimized under per-query resource budgets on edge edits, interaction steps, and selected tokens. Across HotpotQA, MetaQA, and FactKG, CLAUSE yields higher EM@1 while reducing subgraph growth and end-to-end latency at equal or lower token budgets. On MetaQA-2-hop, relative to the strongest RAG baseline (GraphRAG), CLAUSE achieves +39.3 EM@1 with 18.6% lower latency and 40.9% lower edge growth. The resulting contexts are compact, provenance-preserving, and deliver predictable performance under deployment constraints.
>
---
#### [replaced 035] STATe-of-Thoughts: Structured Action Templates for Tree-of-Thoughts
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出STATe方法，解决文本生成中多样性与可控性不足的问题。通过结构化动作模板实现可解释的推理过程，提升生成质量与可解释性。**

- **链接: [https://arxiv.org/pdf/2602.14265](https://arxiv.org/pdf/2602.14265)**

> **作者:** Zachary Bamberger; Till R. Saenger; Gilad Morad; Ofra Amir; Brandon M. Stewart; Amir Feder
>
> **备注:** v2, 10 pages main, 80 pages total, 19 tables, 20 figures
>
> **摘要:** Inference-Time-Compute (ITC) methods like Best-of-N and Tree-of-Thoughts are meant to produce output candidates that are both high-quality and diverse, but their use of high-temperature sampling often fails to achieve meaningful output diversity. Moreover, existing ITC methods offer limited control over how to perform reasoning, which in turn limits their interpretability. We present STATe Of Thoughts (STATe), an interpretable ITC method that searches over high-level reasoning patterns. STATe replaces stochastic sampling with discrete and interpretable textual interventions: a controller selects actions encoding high-level reasoning choices; a generator produces reasoning steps conditioned on those choices; and an evaluator scores candidates to guide search. This structured approach yields three main advantages. First, action-guided textual interventions reliably influence LLM generations and produce greater response diversity than temperature-based sampling. Second, in a case study on argument generation, STATe's explicit action sequences capture interpretable features that are highly predictive of output quality. Third, estimating the association between performance and action choices allows us to identify promising yet unexplored regions of the action space and steer generation toward them. Together, these results establish STATe as both a practical framework for diverse and controllable text generation, and as a tool for understanding the reasoning patterns that drive performance.
>
---
#### [replaced 036] Language on Demand, Knowledge at Core: Composing LLMs with Encoder-Decoder Translation Models for Extensible Multilinguality
- **分类: cs.CL**

- **简介: 该论文属于多语言自然语言处理任务，旨在解决LLMs在低资源语言上的表现不足问题。通过结合编码器-解码器翻译模型与LLM，构建XBridge架构提升多语言能力。**

- **链接: [https://arxiv.org/pdf/2603.17512](https://arxiv.org/pdf/2603.17512)**

> **作者:** Mengyu Bu; Yang Feng
>
> **备注:** Submitted to ACL 2026. The code is available at this https URL
>
> **摘要:** Large language models (LLMs) exhibit strong general intelligence, yet their multilingual performance remains highly imbalanced. Although LLMs encode substantial cross-lingual knowledge in a unified semantic space, they often struggle to reliably interface this knowledge with low-resource or unseen languages. Fortunately, pretrained encoder-decoder translation models already possess balanced multilingual capability, suggesting a natural complement to LLMs. In this work, we propose XBridge, a compositional encoder-LLM-decoder architecture that offloads multilingual understanding and generation to external pretrained translation models, while preserving the LLM as an English-centric core for general knowledge processing. To address the resulting representation misalignment across models, we introduce lightweight cross-model mapping layers and an optimal transport-based alignment objective, enabling fine-grained semantic consistency for multilingual generation. Experiments on four LLMs across multilingual understanding, reasoning, summarization, and generation indicate that XBridge outperforms strong baselines, especially on low-resource and previously unseen languages, without retraining the LLM.
>
---
#### [replaced 037] How to Train Your Long-Context Visual Document Model
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文聚焦长文档视觉问答任务，解决模型训练与评估不匹配的问题。通过系统实验，提出有效训练方法并提升模型性能。**

- **链接: [https://arxiv.org/pdf/2602.15257](https://arxiv.org/pdf/2602.15257)**

> **作者:** Austin Veselka
>
> **摘要:** We present the first comprehensive, large-scale study of training long-context vision language models up to 344K context, targeting long-document visual question answering with measured transfer to long-context text. While several such strong are open-weight, namely Qwen3 VL and GLM 4.5/6V, their training recipes and data pipelines are not reproducible. We systematically study continued pretraining, supervised finetuning, and preference optimization for 24B and 32B parameter models, backed by extensive LC evaluations and ablations to bridge this gap, and achieve state-of-the-art performance on MMLongBenchDoc for both parameter scales. In addition to this, our key findings include: (i) training on context lengths that match evaluation context lengths outperforms training on longer contexts, (ii) training and evaluating with page indices provides a simple, high-impact boost to long-document performance, (iii) our synthetic data pipelines enable self-improvement via continued pretraining and supervised finetuning, and (iv) we extend the known text-to-visual long context transfer to the reverse, showing that visual long context training transfers to long-context text performance. We also release MMLBD-C, a manually corrected version of MMLongBenchDoc to reduce erroneous and low quality examples in the benchmark.
>
---
#### [replaced 038] Magic Words or Methodical Work? Challenging Conventional Wisdom in LLM-Based Political Text Annotation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文研究LLM在政治文本标注中的应用，探讨模型选择、提示风格等对结果的影响。旨在解决标注方法有效性问题，通过实验验证不同配置效果，提出验证优先的框架。**

- **链接: [https://arxiv.org/pdf/2603.26898](https://arxiv.org/pdf/2603.26898)**

> **作者:** Lorcan McLaren; James Cross; Zuzanna Krakowska; Robin Rauner; Martijn Schoonvelde
>
> **摘要:** Political scientists are rapidly adopting large language models (LLMs) for text annotation, yet the sensitivity of annotation results to implementation choices remains poorly understood. Most evaluations test a single model or configuration; how model choice, model size, learning approach, and prompt style interact, and whether popular "best practices" survive controlled comparison, are largely unexplored. We present a controlled evaluation of these pipeline choices, testing six open-weight models across four political science annotation tasks under identical quantisation, hardware, and prompt-template conditions. Our central finding is methodological: interaction effects dominate main effects, so seemingly reasonable pipeline choices can become consequential researcher degrees of freedom. No single model, prompt style, or learning approach is uniformly superior, and the best-performing model varies across tasks. Two corollaries follow. First, model size is an unreliable guide both to cost and to performance: cross-family efficiency differences are so large that some larger models are less resource-intensive than much smaller alternatives, while within model families mid-range variants often match or exceed larger counterparts. Second, widely recommended prompt engineering techniques yield inconsistent and sometimes negative effects on annotation performance. We use these benchmark results to develop a validation-first framework - with a principled ordering of pipeline decisions, guidance on prompt freezing and held-out evaluation, reporting standards, and open-source tools - to help researchers navigate this decision space transparently.
>
---
#### [replaced 039] Stronger Normalization-Free Transformers
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于深度学习任务，旨在替代传统归一化层。通过设计新的点态函数Derf，提升Transformer模型性能，实验证明其在多个领域优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.10938](https://arxiv.org/pdf/2512.10938)**

> **作者:** Mingzhi Chen; Taiming Lu; Jiachen Zhu; Mingjie Sun; Zhuang Liu
>
> **备注:** Published in CVPR 2026
>
> **摘要:** Although normalization layers have long been viewed as indispensable components of deep learning architectures, the recent introduction of Dynamic Tanh (DyT) has demonstrated that alternatives are possible. The point-wise function DyT constrains extreme values for stable convergence and reaches normalization-level performance; this work seeks further for function designs that can surpass it. We first study how the intrinsic properties of point-wise functions influence training and performance. Building on these findings, we conduct a large-scale search for a more effective function design. Through this exploration, we introduce $\mathrm{Derf}(x) = \mathrm{erf}(\alpha x + s)$, where $\mathrm{erf}(x)$ is the rescaled Gaussian cumulative distribution function, and identify it as the most performant design. Derf outperforms LayerNorm, RMSNorm, and DyT across a wide range of domains, including visual recognition and generation, speech representation, and DNA sequence modeling. Our analysis also suggests that the performance gains of Derf largely stem from its improved generalization rather than stronger fitting capacity. Its simplicity and stronger performance make Derf a practical choice for normalization-free Transformer architectures.
>
---
#### [replaced 040] SemioLLM: Evaluating Large Language Models for Diagnostic Reasoning from Unstructured Clinical Narratives in Epilepsy
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗诊断任务，旨在评估大语言模型从非结构化临床文本中进行癫痫诊断推理的能力。通过构建框架测试模型在真实临床场景中的表现。**

- **链接: [https://arxiv.org/pdf/2407.03004](https://arxiv.org/pdf/2407.03004)**

> **作者:** Meghal Dani; Muthu Jeyanthi Prakash; Filip Rosa; Zeynep Akata; Stefanie Liebe
>
> **摘要:** Large Language Models (LLMs) have been shown to encode clinical knowledge. Many evaluations, however, rely on structured question-answer benchmarks, overlooking critical challenges of interpreting and reasoning about unstructured clinical narratives in real-world settings. In this study we task eight Large Language models including two medical models (GPT-3.5, GPT-4, Mixtral-8x7B, Qwen-72B, LlaMa2, LlaMa3, OpenBioLLM, Med42) with a core diagnostic task in epilepsy: mapping seizure description phrases, after targeted filtering and standardization, to one of seven possible seizure onset zones using likelihood estimates. Most models yield results that often match the ground truth and even approach clinician-level performance after prompt engineering. Specifically, clinician-guided chain-of-thought reasoning leading to the most consistent improvements. Performance was further strongly modulated by clinical in-context impersonation, narrative length and language context (13.7%, 32.7% and 14.2% performance variation, respectively). However, expert analysis of reasoning outputs revealed that correct prediction can be based on hallucinated knowledge and inaccurate source citation, underscoring the need to improve interpretability of LLMs in clinical use. Overall, SemioLLM provides a scalable, domain-adaptable framework for evaluating LLMs in clinical disciplines where unstructured verbal descriptions encode diagnostic information. By identifying both the strengths and limitations of LLMs, our work contributes to testing the applicability of foundational AI systems for healthcare.
>
---
#### [replaced 041] Real-Time Trustworthiness Scoring for LLM Structured Outputs and Data Extraction
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于LLM输出可信度评估任务，旨在解决结构化输出中的错误检测问题。提出CONSTRUCT方法，实时评分输出及各字段可信度，无需额外数据或模型，提升错误检测效果。**

- **链接: [https://arxiv.org/pdf/2603.18014](https://arxiv.org/pdf/2603.18014)**

> **作者:** Hui Wen Goh; Jonas Mueller
>
> **摘要:** Structured Outputs from current LLMs exhibit sporadic errors, hindering enterprise AI deployment. We present CONSTRUCT, a real-time uncertainty estimator that scores the trustworthiness of LLM Structured Outputs. Lower-scoring outputs are more likely to contain errors, enabling automatic prioritization of limited human review bandwidth. CONSTRUCT additionally scores the trustworthiness of each field within a Structured Output, helping reviewers quickly identify which parts of the output are incorrect. Our method is suitable for any LLM (including black-box LLM APIs without logprobs), does not require labeled training data or custom model deployment, and supports complex Structured Outputs with heterogeneous fields and nested JSON schemas. We also introduce one of the first public LLM Structured Output benchmarks with reliable ground-truth values. Over this four-dataset benchmark, CONSTRUCT detects errors in outputs from various LLMs (including Gemini 3 and GPT-5) with significantly higher precision/recall than existing techniques.
>
---
#### [replaced 042] From Efficiency to Adaptivity: A Deeper Look at Adaptive Reasoning in Large Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理领域，探讨大语言模型的自适应推理问题。针对模型在不同任务中缺乏灵活策略的问题，提出自适应推理框架，分类现有方法并指出未来挑战。**

- **链接: [https://arxiv.org/pdf/2511.10788](https://arxiv.org/pdf/2511.10788)**

> **作者:** Chao Wu; Baoheng Li; Mingchen Gao; Yu Tian; Zhenyi Wang
>
> **摘要:** Recent advances in large language models (LLMs) have made reasoning a central benchmark for evaluating intelligence. While prior surveys focus on efficiency by examining how to shorten reasoning chains or reduce computation, this view overlooks a fundamental challenge: current LLMs apply uniform reasoning strategies regardless of task complexity, generating long traces for trivial problems while failing to extend reasoning for difficult tasks. This survey reframes reasoning through the lens of {adaptivity}: the capability to allocate reasoning effort based on input characteristics such as difficulty and uncertainty. We make three contributions. First, we formalize deductive, inductive, and abductive reasoning within the LLM context, connecting these classical cognitive paradigms with their algorithmic realizations. Second, we formalize adaptive reasoning as a control-augmented policy optimization problem balancing task performance with computational cost, distinguishing learned policies from inference-time control mechanisms. Third, we propose a systematic taxonomy organizing existing methods into training-based approaches that internalize adaptivity through reinforcement learning, supervised fine-tuning, and learned controllers, and training-free approaches that achieve adaptivity through prompt conditioning, feedback-driven halting, and modular composition. This framework clarifies how different mechanisms realize adaptive reasoning in practice and enables systematic comparison across diverse strategies. We conclude by identifying open challenges in self-evaluation, meta-reasoning, and human-aligned reasoning control.
>
---
#### [replaced 043] Biasless Language Models Learn Unnaturally: How LLMs Fail to Distinguish the Possible from the Impossible
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，研究LLMs是否能区分可能与不可能的语言。通过实验发现GPT-2无法有效区分自然语言与干扰后的不可能语言。**

- **链接: [https://arxiv.org/pdf/2510.07178](https://arxiv.org/pdf/2510.07178)**

> **作者:** Imry Ziv; Nur Lan; Emmanuel Chemla
>
> **备注:** 15 pages, 4 figures
>
> **摘要:** Are large language models (LLMs) sensitive to the distinction between humanly possible and impossible languages? This question was recently used in a broader debate on whether LLMs and humans share the same innate learning biases. Previous work has answered it in the positive by comparing LLM learning curves on existing language datasets and on "impossible" datasets derived from them via various perturbation functions. Using the same methodology, we examine this claim on a wider set of languages and impossible perturbations. We find that in most cases, GPT-2 learns each language and its impossible counterpart equally easily, in contrast to previous findings. We also apply a more lenient condition by testing whether GPT-2 provides any kind of separation between the whole sets of natural vs. impossible languages, based on cross-linguistic variance in metrics derived from the learning curves. Taken together, these perspectives show that GPT-2 provides no systematic separation between the possible and the impossible.
>
---
#### [replaced 044] Tokens with Meaning: A Hybrid Tokenization Approach for Turkish
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的分词任务，针对土耳其语的形态丰富性问题，提出一种结合词典、音系和子词的混合分词方法，提升分词质量与模型性能。**

- **链接: [https://arxiv.org/pdf/2508.14292](https://arxiv.org/pdf/2508.14292)**

> **作者:** M. Ali Bayram; Ali Arda Fincan; Ahmet Semih Gümüş; Sercan Karakaş; Banu Diri; Savaş Yıldırım; Demircan Çelik
>
> **摘要:** Tokenization shapes how language models perceive morphology and meaning in NLP, yet widely used frequency-driven subword tokenizers (e.g., Byte Pair Encoding and WordPiece) can fragment morphologically rich and agglutinative languages in ways that obscure morpheme boundaries. We introduce a linguistically informed hybrid tokenizer for Turkish that combines (i) dictionary-driven morphological segmentation (roots and affixes), (ii) phonological normalization that maps allomorphic variants to shared identifiers, and (iii) a controlled subword fallback for out-of-vocabulary coverage. Concretely, our released Turkish vocabulary contains 22,231 root tokens mapped to 20,000 canonical root identifiers (with leading spaces to mark word boundaries), 72 affix identifiers that cover 177 allomorphic surface forms, and 12,696 subword units; an orthographic case token preserves capitalization without inflating the vocabulary. We evaluate tokenization quality on the TR-MMLU dataset using two linguistic alignment metrics: Turkish Token Percentage (TR~\%), the proportion of produced tokens that correspond to Turkish lexical/morphemic units under our lexical resources, and Pure Token Percentage (Pure~\%), the proportion of tokens aligning with unambiguous root/affix boundaries. The proposed tokenizer reaches 90.29\% TR~\% and 85.80\% Pure~\% on TR-MMLU, substantially exceeding several general-purpose tokenizers. We further validate practical utility with downstream sentence embedding benchmarks under a strict \emph{random initialization} control to isolate tokenizer inductive bias. Across four matched models (TurkishTokenizer, CosmosGPT2, Mursit, and Tabi), TurkishTokenizer outperforms all baselines on the Turkish STS Benchmark and achieves the strongest overall average on MTEB-TR. It also yields the strongest average accuracy on the TurBLiMP under a centroid-based proxy.
>
---
#### [replaced 045] ShishuLM : Achieving Optimal and Efficient Parameterization with Low Attention Transformer Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决Transformer模型计算和内存开销大的问题。通过替换顶层全解码层为MLP块，提升效率并减少内存占用。**

- **链接: [https://arxiv.org/pdf/2510.13860](https://arxiv.org/pdf/2510.13860)**

> **作者:** Shivanshu Kumar; Gopalakrishnan Srinivasan
>
> **摘要:** While the transformer architecture has achieved state-of-the-art performance on natural language processing tasks, these models impose substantial memory and computational overhead. Recent research has identified significant architectural redundancies within these models, particularly in the attention sub-layers in the top layers, presenting opportunities for optimization without compromising performance. Taking insights from research on inference-time layer pruning and depth-dependent computation in language models, we introduce an efficient language model architecture referred to as ShishuLM. By replacing full decoder layers at the top of the model with MLP-only blocks, we achieve up to 10-60% improvement in generation latency and 1.3 -5 $\times$ gain in throughput. Upon further sharing parameters across adjacent MLP-only layers of ShishuLM, we obtain up to 20% savings in memory with minimal degradation in performance. Our findings provide insights towards building more efficient language modeling architectures from a pre-training standpoint by leveraging how information flows in transformers.
>
---
#### [replaced 046] QuestA: Expanding Reasoning Capacity in LLMs via Question Augmentation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型推理任务，旨在提升大模型的推理能力。针对强化学习在复杂推理任务中的局限性，提出QuestA方法，通过问题增强提高模型表现。**

- **链接: [https://arxiv.org/pdf/2507.13266](https://arxiv.org/pdf/2507.13266)**

> **作者:** Jiazheng Li; Hongzhou Lin; Hong Lu; Kaiyue Wen; Zaiwen Yang; Jiaxuan Gao; Yi Wu; Jingzhao Zhang
>
> **备注:** 25 pages, 18 figures, ICLR 2026
>
> **摘要:** Reinforcement learning (RL) has emerged as a central paradigm for training large language models (LLMs) in reasoning tasks. Yet recent studies question RL's ability to incentivize reasoning capacity beyond the base model. This raises a key challenge: how can RL be adapted to solve harder reasoning problems more effectively? To address this challenge, we propose a simple yet effective strategy via Question Augmentation: introduce partial solutions during training to reduce problem difficulty and provide more informative learning signals. Our method, QuestA, when applied during RL training on math reasoning tasks, not only improves pass@1 but also pass@k-particularly on problems where standard RL struggles to make progress. This enables continual improvement over strong open-source models such as DeepScaleR and OpenMath Nemotron, further enhancing their reasoning capabilities. We achieve new state-of-the-art results on math benchmarks using 1.5B-parameter models: 72.50% (+10.73%) on AIME24, 62.29% (+12.79%) on AIME25, and 41.67% (+10.11%) on HMMT25. Code, data and model are available at this https URL.
>
---
#### [replaced 047] MA-SAPO: Multi-Agent Reasoning for Score-Aware Prompt Optimization
- **分类: cs.MA; cs.AI; cs.CL; cs.HC; cs.IR**

- **简介: 该论文提出MA-SAPO框架，解决提示优化问题，通过多智能体推理将评估结果与改进建议直接关联，提升优化的可解释性和效果。**

- **链接: [https://arxiv.org/pdf/2510.16635](https://arxiv.org/pdf/2510.16635)**

> **作者:** Wonduk Seo; Juhyeon Lee; Junseo Koh; Wonseok Choi; Hyunjin An; Jian Park; Seunghyun lee; Haihua Chen; Yi Bu
>
> **备注:** Preprint
>
> **摘要:** Prompt optimization has become a practical way to improve the performance of Large Language Models (LLMs) without retraining. However, most existing frameworks treat evaluation as a black box, relying solely on outcome scores without explaining why prompts succeed or fail. Moreover, they involve repetitive trial-and-error refinements that remain implicit, offering limited interpretability or actionable guidance for systematic improvement. In this paper, we propose MA-SAPO: a new Multi-Agent Reasoning for Score Aware Prompt Optimization framework that links evaluation outcomes directly to targeted refinements. Specifically, in the Training Phase, multiple agents interpret evaluation scores, diagnose weaknesses, and generate concrete revision directives, which are stored as reusable reasoning assets. In the Test Phase, an analyzer agent retrieves relevant exemplars and assets for a new prompt, and a refiner agent applies evidence-based edits to improve the prompt and its response. By grounding optimization in structured reasoning, MA-SAPO ensures edits are interpretable, auditable, and controllable. Experiments on the HelpSteer1/2 benchmarks show that our framework consistently outperforms single-pass prompting, retrieval-augmented generation, and prior multi-agent methods across multiple evaluation metrics.
>
---
