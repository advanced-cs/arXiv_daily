# 自然语言处理 cs.CL

- **最新发布 104 篇**

- **更新 55 篇**

## 最新发布

#### [new 001] PashtoCorp: A 1.25-Billion-Word Corpus, Evaluation Suite, and Reproducible Pipeline for Low-Resource Language Development
- **分类: cs.CL; cs.IR; cs.LG**

- **简介: 该论文提出PashtoCorp，一个12.5亿词的普什图语语料库，用于解决低资源语言NLP研究不足的问题。通过构建大规模语料库和预训练模型，提升命名实体识别和阅读理解任务性能。**

- **链接: [https://arxiv.org/pdf/2603.16354](https://arxiv.org/pdf/2603.16354)**

> **作者:** Hanif Rahman
>
> **摘要:** We present PashtoCorp, a 1.25-billion-word corpus for Pashto, a language spoken by 60 million people that remains severely underrepresented in NLP. The corpus is assembled from 39 sources spanning seven HuggingFace datasets and 32 purpose-built web scrapers, processed through a reproducible pipeline with Arabic-script tokenization, SHA-256 deduplication, and quality filtering. At 1.25B words across 2.81 million documents, PashtoCorp is 40x larger than the OSCAR Pashto subset and 83x larger than the previously largest dedicated Pashto corpus. Continued MLM pretraining of XLM-R-base on PashtoCorp reduces held-out perplexity by 25.1% (8.08->6.06). On WikiANN Pashto NER, the pretrained model improves entity F1 by 10% relative (19.0%->21.0%) and reduces training variance nearly 7x; the largest gain appears at 50 training sentences (+27%), with PashtoCorp covering 97.9% of WikiANN entity vocabulary. On Belebele Pashto reading comprehension, Gemma-3n achieves 64.6% accuracy, the first published LLM baseline for Pashto on this benchmark. A leave-one-out source ablation shows that Wikipedia (0.7% of documents) is the most critical source for NER: removing it alone reduces entity F1 by 47%. Corpus data, trained model, and code are available at this https URL, this https URL, and this https URL.
>
---
#### [new 002] ASDA: Automated Skill Distillation and Adaptation for Financial Reasoning
- **分类: cs.CL; cs.AI; cs.CE**

- **简介: 该论文提出ASDA框架，解决金融推理中大模型适应问题，通过生成结构化技能文件提升推理性能，无需修改模型权重。**

- **链接: [https://arxiv.org/pdf/2603.16112](https://arxiv.org/pdf/2603.16112)**

> **作者:** Tik Yu Yim; Wenting Tan; Sum Yee Chan; Tak-Wah Lam; Siu Ming Yiu
>
> **摘要:** Adapting large language models (LLMs) to specialized financial reasoning typically requires expensive fine-tuning that produces model-locked expertise. Training-free alternatives have emerged, yet our experiments show that leading methods (GEPA and ACE) achieve only marginal gains on the FAMMA financial reasoning benchmark, exposing the limits of unstructured text optimization for complex, multi-step domain reasoning. We introduce Automated Skill Distillation and Adaptation (ASDA), a framework that automatically generates structured skill artifacts through iterative error-corrective learning without modifying model weights. A teacher model analyzes a student model's failures on financial reasoning tasks, clusters errors by subfield and error type, and synthesizes skill files containing reasoning procedures, code templates, and worked examples, which are dynamically injected during inference. Evaluated on FAMMA, ASDA achieves up to +17.33% improvement on arithmetic reasoning and +5.95% on non-arithmetic reasoning, substantially outperforming all training-free baselines. The resulting skill artifacts are human-readable, version-controlled, and compatible with the Agent Skills open standard, offering any organization with a labeled domain dataset a practical and auditable path to domain adaptation without weight access or retraining.
>
---
#### [new 003] SpecSteer: Synergizing Local Context and Global Reasoning for Efficient Personalized Generation
- **分类: cs.CL**

- **简介: 该论文属于个性化生成任务，解决隐私与推理能力的矛盾。提出SpecSteer框架，结合本地上下文与云端推理，提升生成质量与效率。**

- **链接: [https://arxiv.org/pdf/2603.16219](https://arxiv.org/pdf/2603.16219)**

> **作者:** Hang Lv; Sheng Liang; Hao Wang; Yongyue Zhang; Hongchao Gu; Wei Guo; Defu Lian; Yong Liu; Enhong Chen
>
> **摘要:** Realizing personalized intelligence faces a core dilemma: sending user history to centralized large language models raises privacy concerns, while on-device small language models lack the reasoning capacity required for high-quality generation. Our pilot study shows that purely local enhancements remain insufficient to reliably bridge this gap. We therefore propose SpecSteer, an asymmetric collaborative inference framework that synergizes private on-device context with cloud-scale reasoning. SpecSteer casts collaboration as Bayesian knowledge fusion and repurposes speculative decoding as a distributed alignment protocol, yielding a Draft--Verify--Recover pipeline: the on-device model drafts personalized sequences; the cloud validates via a ratio-based mechanism that decouples reasoning verification from private context, filtering logical flaws without accessing raw user context; upon rejection, a steering recovery injects local intent during correction. Experiments demonstrate that SpecSteer successfully closes the reasoning gap and achieves superior personalized generation performance, while delivering a 2.36x speedup over standard baselines.
>
---
#### [new 004] Omanic: Towards Step-wise Evaluation of Multi-hop Reasoning in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出Omanic数据集，用于评估大语言模型的多跳推理能力。针对现有评估方法无法揭示推理过程的问题，该工作构建了包含分解问题和中间答案的数据集，以分析模型推理过程中的错误。**

- **链接: [https://arxiv.org/pdf/2603.16654](https://arxiv.org/pdf/2603.16654)**

> **作者:** Xiaojie Gu; Sherry T. Tong; Aosong Feng; Sophia Simeng Han; Jinghui Lu; Yingjian Chen; Yusuke Iwasawa; Yutaka Matsuo; Chanjun Park; Rex Ying; Irene Li
>
> **摘要:** Reasoning-focused large language models (LLMs) have advanced in many NLP tasks, yet their evaluation remains challenging: final answers alone do not expose the intermediate reasoning steps, making it difficult to determine whether a model truly reasons correctly and where failures occur, while existing multi-hop QA benchmarks lack step-level annotations for diagnosing reasoning failures. To address this gap, we propose Omanic, an open-domain multi-hop QA resource that provides decomposed sub-questions and intermediate answers as structural annotations for analyzing reasoning processes. It contains 10,296 machine-generated training examples (OmanicSynth) and 967 expert-reviewed human-annotated evaluation examples (OmanicBench). Systematic evaluations show that state-of-the-art LLMs achieve only 73.11% multiple-choice accuracy on OmanicBench, confirming its high difficulty. Stepwise analysis reveals that CoT's performance hinges on factual completeness, with its gains diminishing under knowledge gaps and errors amplifying in later hops. Additionally, supervised fine-tuning on OmanicSynth brings substantial transfer gains (7.41 average points) across six reasoning and math benchmarks, validating the dataset's quality and further supporting the effectiveness of OmanicSynth as supervision for reasoning-capability transfer. We release the data at this https URL and the code at this https URL.
>
---
#### [new 005] Recursive Language Models Meet Uncertainty: The Surprising Effectiveness of Self-Reflective Program Search for Long Context
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究长文本处理任务，解决语言模型在长上下文中的信息提取与推理问题。提出SRLM框架，通过自省程序搜索提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.15653](https://arxiv.org/pdf/2603.15653)**

> **作者:** Keivan Alizadeh; Parshin Shojaee; Minsik Cho; Mehrdad Farajtabar
>
> **备注:** preprint
>
> **摘要:** Long-context handling remains a core challenge for language models: even with extended context windows, models often fail to reliably extract, reason over, and use the information across long contexts. Recent works like Recursive Language Models (RLM) have approached this challenge by agentic way of decomposing long contexts into recursive sub-calls through programmatic interaction at inference. While promising, the success of RLM critically depends on how these context-interaction programs are selected, which has remained largely unexplored. In this paper, we study this problem and introduce SRLM, a framework that augments programmatic context interaction with uncertainty-aware Self-Reflection. SRLM leverages three intrinsic signals: self consistency, reasoning length, and verbalized confidence. These serve as complementary indicators of a model's internal uncertainty, and the model uses them to evaluate and compare candidate context-interaction programs. Extensive experiments across diverse benchmark datasets, context lengths, and backbone models, show that SRLM consistently outperforms state-of-the-art baselines, yielding up to 22% improvement over RLM under the same time budget. Our findings show that recursion itself is not the primary driver of performance in RLM, and a simple self-reflective program search can match or surpass RLM without requiring self-query or explicit recursion mechanisms. We find that for context lengths within the model's window, RLMs with recursion often degrade performance relative to the base model, whereas SRLM yields consistent gains across both short and long contexts. We also find that RLM is less effective in tasks with semantically intensive nature, where heuristic program search is insufficient and broader contextual understanding is required, while self-reflection in SRLM provides a semantic signal that better steers reasoning in these scenarios.
>
---
#### [new 006] Social Simulacra in the Wild: AI Agent Communities on Moltbook
- **分类: cs.CL**

- **简介: 该论文属于社会计算任务，研究AI代理在社交平台上的社区动态。通过对比分析Moltbook与Reddit数据，揭示AI代理社区的结构和语言特征差异，探讨多代理互动对集体沟通的影响。**

- **链接: [https://arxiv.org/pdf/2603.16128](https://arxiv.org/pdf/2603.16128)**

> **作者:** Agam Goyal; Olivia Pal; Hari Sundaram; Eshwar Chandrasekharan; Koustuv Saha
>
> **备注:** Preprint: 12 pages, 4 figures, 5 tables
>
> **摘要:** As autonomous LLM-based agents increasingly populate social platforms, understanding the dynamics of AI-agent communities becomes essential for both communication research and platform governance. We present the first large-scale empirical comparison of AI-agent and human online communities, analyzing 73,899 Moltbook and 189,838 Reddit posts across five matched communities. Structurally, we find that Moltbook exhibits extreme participation inequality (Gini = 0.84 vs. 0.47) and high cross-community author overlap (33.8\% vs. 0.5\%). In terms of linguistic attributes, content generated by AI-agents is emotionally flattened, cognitively shifted toward assertion over exploration, and socially detached. These differences give rise to apparent community-level homogenization, but we show this is primarily a structural artifact of shared authorship. At the author level, individual agents are more identifiable than human users, driven by outlier stylistic profiles amplified by their extreme posting volume. As AI-mediated communication reshapes online discourse, our work offers an empirical foundation for understanding how multi-agent interaction gives rise to collective communication dynamics distinct from those of human communities.
>
---
#### [new 007] MiroThinker-1.7 & H1: Towards Heavy-Duty Research Agents via Verification
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **简介: 该论文提出MiroThinker-1.7和H1，用于复杂长周期推理任务，解决多步骤问题可靠性问题，通过结构化规划和验证提升推理效果。**

- **链接: [https://arxiv.org/pdf/2603.15726](https://arxiv.org/pdf/2603.15726)**

> **作者:** MiroMind Team; S. Bai; L. Bing; L. Lei; R. Li; X. Li; X. Lin; E. Min; L. Su; B. Wang; L. Wang; L. Wang; S. Wang; X. Wang; Y. Zhang; Z. Zhang; G. Chen; L. Chen; Z. Cheng; Y. Deng; Z. Huang; D. Ng; J. Ni; Q. Ren; X. Tang; B.L. Wang; H. Wang; N. Wang; C. Wei; Q. Wu; J. Xia; Y. Xiao; H. Xu; X. Xu; C. Xue; Z. Yang; Z. Yang; F. Ye; H. Ye; J. Yu; C. Zhang; W. Zhang; H. Zhao; P. Zhu
>
> **备注:** 23 pages
>
> **摘要:** We present MiroThinker-1.7, a new research agent designed for complex long-horizon reasoning tasks. Building on this foundation, we further introduce MiroThinker-H1, which extends the agent with heavy-duty reasoning capabilities for more reliable multi-step problem solving. In particular, MiroThinker-1.7 improves the reliability of each interaction step through an agentic mid-training stage that emphasizes structured planning, contextual reasoning, and tool interaction. This enables more effective multi-step interaction and sustained reasoning across complex tasks. MiroThinker-H1 further incorporates verification directly into the reasoning process at both local and global levels. Intermediate reasoning decisions can be evaluated and refined during inference, while the overall reasoning trajectory is audited to ensure that final answers are supported by coherent chains of evidence. Across benchmarks covering open-web research, scientific reasoning, and financial analysis, MiroThinker-H1 achieves state-of-the-art performance on deep research tasks while maintaining strong results on specialized domains. We also release MiroThinker-1.7 and MiroThinker-1.7-mini as open-source models, providing competitive research-agent capabilities with significantly improved efficiency.
>
---
#### [new 008] How often do Answers Change? Estimating Recency Requirements in Question Answering
- **分类: cs.CL**

- **简介: 该论文属于问答系统任务，解决LLM在时间敏感问题上因知识过时导致错误的问题。提出RecencyQA数据集，标注问题答案变化频率与上下文依赖性，以提升模型对时效性的理解。**

- **链接: [https://arxiv.org/pdf/2603.16544](https://arxiv.org/pdf/2603.16544)**

> **作者:** Bhawna Piryani; Zehra Mert; Adam Jatowt
>
> **摘要:** Large language models (LLMs) often rely on outdated knowledge when answering time-sensitive questions, leading to confident yet incorrect responses. Without explicit signals indicating whether up-to-date information is required, models struggle to decide when to retrieve external evidence, how to reason about stale facts, and how to rank answers by their validity. Existing benchmarks either periodically refresh answers or rely on fixed templates, but they do not reflect on how frequently answers change or whether a question inherently requires up-to-date information. To address this gap, we introduce a recency-stationarity taxonomy that categorizes questions by how often their answers change and whether this change frequency is time-invariant or context-dependent. Building on this taxonomy, we present RecencyQA, a dataset of 4,031 open-domain questions annotated with recency and stationarity labels. Through human evaluation and empirical analysis, we show that non-stationary questions, i.e., those where context changes the recency requirement, are significantly more challenging for LLMs, with difficulty increasing as update frequency rises. By explicitly modeling recency and context dependence, RecencyQA enables fine-grained benchmarking and analysis of temporal reasoning beyond binary notions of freshness, and provides a foundation for developing recency-aware and context-sensitive question answering systems.
>
---
#### [new 009] Language Models Don't Know What You Want: Evaluating Personalization in Deep Research Needs Real Users
- **分类: cs.CL**

- **简介: 该论文属于个性化深度研究任务，旨在解决语言模型缺乏用户理解的问题。作者提出MySQA工具，通过用户画像和个性化行动提升研究效率，并强调真实用户反馈的重要性。**

- **链接: [https://arxiv.org/pdf/2603.16120](https://arxiv.org/pdf/2603.16120)**

> **作者:** Nishant Balepur; Malachi Hamada; Varsha Kishore; Sergey Feldman; Amanpreet Singh; Pao Siangliulue; Joseph Chee Chang; Eunsol Choi; Jordan Lee Boyd-Graber; Aakanksha Naik
>
> **备注:** Under Review
>
> **摘要:** Deep Research (DR) tools (e.g. OpenAI DR) help researchers cope with ballooning publishing counts. Such tools can synthesize scientific papers to answer researchers' queries, but lack understanding of their users. We change that in MyScholarQA (MySQA), a personalized DR tool that: 1) infers a profile of a user's research interests; 2) proposes personalized actions for a user's input query; and 3) writes a multi-section report for the query that follows user-approved actions. We first test MySQA with NLP's standard protocol: we design a benchmark of synthetic users and LLM judges, where MySQA beats baselines in citation metrics and personalized action-following. However, we suspect this process does not cover all aspects of personalized DR users value, so we interview users in an online version of MySQA to unmask them. We reveal nine nuanced errors of personalized DR undetectable by our LLM judges, and we study qualitative feedback to form lessons for future DR design. In all, we argue for a pillar of personalization that easy-to-use LLM judges can lead NLP to overlook: real progress in personalization is only possible with real users.
>
---
#### [new 010] Structured Semantic Cloaking for Jailbreak Attacks on Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的安全攻击任务，旨在突破大模型的安全机制。提出S2C框架，通过语义重构实现更有效的越狱攻击。**

- **链接: [https://arxiv.org/pdf/2603.16192](https://arxiv.org/pdf/2603.16192)**

> **作者:** Xiaobing Sun; Perry Lam; Shaohua Li; Zizhou Wang; Rick Siow Mong Goh; Yong Liu; Liangli Zhen
>
> **备注:** 15 pages
>
> **摘要:** Modern LLMs employ safety mechanisms that extend beyond surface-level input filtering to latent semantic representations and generation-time reasoning, enabling them to recover obfuscated malicious intent during inference and refuse accordingly, and rendering many surface-level obfuscation jailbreak attacks ineffective. We propose Structured Semantic Cloaking (S2C), a novel multi-dimensional jailbreak attack framework that manipulates how malicious semantic intent is reconstructed during model inference. S2C strategically distributes and reshapes semantic cues such that full intent consolidation requires multi-step inference and long-range co-reference resolution within deeper latent representations. The framework comprises three complementary mechanisms: (1) Contextual Reframing, which embeds the request within a plausible high-stakes scenario to bias the model toward compliance; (2) Content Fragmentation, which disperses the semantic signature of the request across disjoint prompt segments; and (3) Clue-Guided Camouflage, which disguises residual semantic cues while embedding recoverable markers that guide output generation. By delaying and restructuring semantic consolidation, S2C degrades safety triggers that depend on coherent or explicitly reconstructed malicious intent at decoding time, while preserving sufficient instruction recoverability for functional output generation. We evaluate S2C across multiple open-source and proprietary LLMs using HarmBench and JBB-Behaviors, where it improves Attack Success Rate (ASR) by 12.4% and 9.7%, respectively, over the current SOTA. Notably, S2C achieves substantial gains on GPT-5-mini, outperforming the strongest baseline by 26% on JBB-Behaviors. We also analyse which combinations perform best against broad families of models, and characterise the trade-off between the extent of obfuscation versus input recoverability on jailbreak success.
>
---
#### [new 011] SEAHateCheck: Functional Tests for Detecting Hate Speech in Low-Resource Languages of Southeast Asia
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于仇恨言论检测任务，旨在解决低资源东南亚语言中仇恨言论识别困难的问题。构建了SEAHateCheck数据集，评估模型性能并揭示其局限性。**

- **链接: [https://arxiv.org/pdf/2603.16070](https://arxiv.org/pdf/2603.16070)**

> **作者:** Ri Chi Ng; Aditi Kumaresan; Yujia Hu; Roy Ka-Wei Lee
>
> **备注:** TALLIP Accepted
>
> **摘要:** Hate speech detection relies heavily on linguistic resources, which are primarily available in high-resource languages such as English and Chinese, creating barriers for researchers and platforms developing tools for low-resource languages in Southeast Asia, where diverse socio-linguistic contexts complicate online hate moderation. To address this, we introduce SEAHateCheck, a pioneering dataset tailored to Indonesia, Thailand, the Philippines, and Vietnam, covering Indonesian, Tagalog, Thai, and Vietnamese. Building on HateCheck's functional testing framework and refining SGHateCheck's methods, SEAHateCheck provides culturally relevant test cases, augmented by large language models and validated by local experts for accuracy. Experiments with state-of-the-art and multilingual models revealed limitations in detecting hate speech in specific low-resource languages. In particular, Tagalog test cases showed the lowest model accuracy, likely due to linguistic complexity and limited training data. In contrast, slang-based functional tests proved the hardest, as models struggled with culturally nuanced expressions. The diagnostic insights of SEAHateCheck further exposed model weaknesses in implicit hate detection and models' struggles with counter-speech expression. As the first functional test suite for these Southeast Asian languages, this work equips researchers with a robust benchmark, advancing the development of practical, culturally attuned hate speech detection tools for inclusive online content moderation.
>
---
#### [new 012] AdaMem: Adaptive User-Centric Memory for Long-Horizon Dialogue Agents
- **分类: cs.CL**

- **简介: 该论文提出AdaMem，解决长周期对话中用户中心记忆的问题。通过自适应记忆框架提升对话连贯性与个性化。**

- **链接: [https://arxiv.org/pdf/2603.16496](https://arxiv.org/pdf/2603.16496)**

> **作者:** Shannan Yan; Jingchen Ni; Leqi Zheng; Jiajun Zhang; Peixi Wu; Dacheng Yin; Jing Lyu; Chun Yuan; Fengyun Rao
>
> **摘要:** Large language model (LLM) agents increasingly rely on external memory to support long-horizon interaction, personalized assistance, and multi-step reasoning. However, existing memory systems still face three core challenges: they often rely too heavily on semantic similarity, which can miss evidence crucial for user-centric understanding; they frequently store related experiences as isolated fragments, weakening temporal and causal coherence; and they typically use static memory granularities that do not adapt well to the requirements of different questions. We propose AdaMem, an adaptive user-centric memory framework for long-horizon dialogue agents. AdaMem organizes dialogue history into working, episodic, persona, and graph memories, enabling the system to preserve recent context, structured long-term experiences, stable user traits, and relation-aware connections within a unified framework. At inference time, AdaMem first resolves the target participant, then builds a question-conditioned retrieval route that combines semantic retrieval with relation-aware graph expansion only when needed, and finally produces the answer through a role-specialized pipeline for evidence synthesis and response generation. We evaluate AdaMem on the LoCoMo and PERSONAMEM benchmarks for long-horizon reasoning and user modeling. Experimental results show that AdaMem achieves state-of-the-art performance on both benchmarks. The code will be released upon acceptance.
>
---
#### [new 013] ClaimFlow: Tracing the Evolution of Scientific Claims in NLP
- **分类: cs.CL**

- **简介: 该论文提出ClaimFlow，构建NLP领域科学主张的演化图谱，解决主张关系分类问题，通过标注和分析大量论文中的主张及其相互关系，揭示观点演变规律。**

- **链接: [https://arxiv.org/pdf/2603.16073](https://arxiv.org/pdf/2603.16073)**

> **作者:** Aniket Pramanick; Yufang Hou; Saif M. Mohammad; Iryna Gurevych
>
> **摘要:** Scientific papers do more than report results $-$ they advance $\textit{claims}$ that later work supports, extends, or sometimes refutes. Yet existing methods for citation and claim analysis capture only fragments of this dialogue. In this work, we make these interactions explicit at the level of individual scientific claims. We introduce $\texttt{ClaimFlow}$, a claim-centric view of the NLP literature, built from $304$ ACL Anthology papers (1979$-$2025) that are manually annotated with $1{,}084$ claims and $832$ cross-paper claim relations, indicating whether a citing paper $\textit{supports}$, $\textit{extends}$, $\textit{qualifies}$, $\textit{refutes}$, or references a claim as $\textit{background}$. Using $\texttt{ClaimFlow}$, we define a new task $-$ $\textit{Claim Relation Classification}$ $-$ which requires models to infer the scientific stance toward a cited claim from the text and citation context. Evaluating strong neural models and large language models on this task, we report baseline performance of $0.78$ macro-F1, highlighting that claim-relation classification is feasible but challenging. We further apply our model to $\sim$$13k$ NLP papers to analyze how claims evolve across decades of NLP research. Our analysis reveals that $63.5$% claims are never reused; only $11.1$% are ever challenged; meanwhile, widely propagated claims are more often $\textit{reshaped}$ through qualification and extension than directly confirmed or refuted. Overall, $\texttt{ClaimFlow}$ offers a lens for examining how ideas shift and mature within NLP, and a foundation for assessing whether models can interpret scientific argumentation.
>
---
#### [new 014] RadAnnotate: Large Language Models for Efficient and Reliable Radiology Report Annotation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于医学自然语言处理任务，旨在解决放射报告标注效率低、成本高的问题。通过大语言模型和合成报告增强，提升标注准确性与自动化水平。**

- **链接: [https://arxiv.org/pdf/2603.16002](https://arxiv.org/pdf/2603.16002)**

> **作者:** Saisha Pradeep Shetty; Roger Eric Goldman; Vladimir Filkov
>
> **备注:** 10 pages, 3 figures. Accepted at AMIA Amplify Informatics Summit 2026
>
> **摘要:** Radiology report annotation is essential for clinical NLP, yet manual labeling is slow and costly. We present RadAnnotate, an LLM-based framework that studies retrieval-augmented synthetic reports and confidence-based selective automation to reduce expert effort for labeling in RadGraph. We study RadGraph-style entity labeling (graph nodes) and leave relation extraction (edges) to future work. First, we train entity-specific classifiers on gold-standard reports and characterize their strengths and failure modes across anatomy and observation categories, with uncertain observations hardest to learn. Second, we generate RAG-guided synthetic reports and show that synthetic-only models remain within 1-2 F1 points of gold-trained models, and that synthetic augmentation is especially helpful for uncertain observations in a low-resource setting, improving F1 from 0.61 to 0.70. Finally, by learning entity-specific confidence thresholds, RadAnnotate can automatically annotate 55-90% of reports at 0.86-0.92 entity match score while routing low-confidence cases for expert review.
>
---
#### [new 015] Arabic Morphosyntactic Tagging and Dependency Parsing with Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究大语言模型在阿拉伯语形态句法标注和依存句法分析中的表现，解决其生成显式语言结构的能力问题。通过对比不同提示方法，评估模型性能并分析其优劣。**

- **链接: [https://arxiv.org/pdf/2603.16718](https://arxiv.org/pdf/2603.16718)**

> **作者:** Mohamed Adel; Bashar Alhafni; Nizar Habash
>
> **摘要:** Large language models (LLMs) perform strongly on many NLP tasks, but their ability to produce explicit linguistic structure remains unclear. We evaluate instruction-tuned LLMs on two structured prediction tasks for Standard Arabic: morphosyntactic tagging and labeled dependency parsing. Arabic provides a challenging testbed due to its rich morphology and orthographic ambiguity, which create strong morphology-syntax interactions. We compare zero-shot prompting with retrieval-based in-context learning (ICL) using examples from Arabic treebanks. Results show that prompt design and demonstration selection strongly affect performance: proprietary models approach supervised baselines for feature-level tagging and become competitive with specialized dependency parsers. In raw-text settings, tokenization remains challenging, though retrieval-based ICL improves both parsing and tokenization. Our analysis highlights which aspects of Arabic morphosyntax and syntax LLMs capture reliably and which remain difficult.
>
---
#### [new 016] Omnilingual SONAR: Cross-Lingual and Cross-Modal Sentence Embeddings Bridging Massively Multilingual Text and Speech
- **分类: cs.CL**

- **简介: 该论文提出OmniSONAR，解决跨语言和跨模态句子嵌入问题，支持数千种语言的文本、语音等统一语义空间，提升下游任务性能。**

- **链接: [https://arxiv.org/pdf/2603.16606](https://arxiv.org/pdf/2603.16606)**

> **作者:** Omnilingual SONAR Team; João Maria Janeiro; Pere-Lluís Huguet Cabot; Ioannis Tsiamas; Yen Meng; Vivek Iyer; Guillem Ramírez; Loic Barrault; Belen Alastruey; Yu-An Chung; Marta R. Costa-Jussa; David Dale; Kevin Heffernan; Jaehyeong Jo; Artyom Kozhevnikov; Alexandre Mourachko; Christophe Ropers; Holger Schwenk; Paul-Ambroise Duquenne
>
> **摘要:** Cross-lingual sentence encoders typically cover only a few hundred languages and often trade downstream quality for stronger alignment, limiting their adoption. We introduce OmniSONAR, a new family of omnilingual, cross-lingual and cross-modal sentence embedding models that natively embed text, speech, code, and mathematical expressions in a single semantic space, while delivering state-of-the-art downstream performance at the scale of thousands of languages, from high-resource to extremely low-resource varieties. To reach this scale without representation collapse, we use progressive training. We first learn a strong foundational space for 200 languages with an LLM-initialized encoder-decoder, combining token-level decoding with a novel split-softmax contrastive loss and synthetic hard negatives. Building on this foundation, we expand to several thousands language varieties via a two-stage teacher-student encoder distillation framework. Finally, we demonstrate the cross-modal extensibility of this space by seamlessly mapping 177 spoken languages into it. OmniSONAR halves cross-lingual similarity search error on the 200-language FLORES dataset and reduces error by a factor of 15 on the 1,560-language BIBLE benchmark. It also enables strong translation, outperforming NLLB-3B on multilingual benchmarks and exceeding prior models (including much larger LLMs) by 15 chrF++ points on 1,560 languages into English BIBLE translation. OmniSONAR also performs strongly on MTEB and XLCoST. For speech, OmniSONAR achieves a 43% lower similarity-search error and reaches 97% of SeamlessM4T speech-to-text quality, despite being zero-shot for translation (trained only on ASR data). Finally, by training an encoder-decoder LM, Spectrum, exclusively on English text processing OmniSONAR embedding sequences, we unlock high-performance transfer to thousands of languages and speech for complex downstream tasks.
>
---
#### [new 017] Parametric Social Identity Injection and Diversification in Public Opinion Simulation
- **分类: cs.CL**

- **简介: 该论文属于公共意见模拟任务，旨在解决LLM模拟中社会多样性缺失的问题。通过引入PSII框架，增强模型对社会身份的表示控制，提升模拟多样性与真实性。**

- **链接: [https://arxiv.org/pdf/2603.16142](https://arxiv.org/pdf/2603.16142)**

> **作者:** Hexi Wang; Yujia Zhou; Bangde Du; Qingyao Ai; Yiqun Liu
>
> **备注:** 16 pages, 9 figures
>
> **摘要:** Large language models (LLMs) have recently been adopted as synthetic agents for public opinion simulation, offering a promising alternative to costly and slow human surveys. Despite their scalability, current LLM-based simulation methods fail to capture social diversity, producing flattened inter-group differences and overly homogeneous responses within demographic groups. We identify this limitation as a Diversity Collapse phenomenon in LLM hidden representations, where distinct social identities become increasingly indistinguishable across layers. Motivated by this observation, we propose Parametric Social Identity Injection (PSII), a general framework that injects explicit, parametric representations of demographic attributes and value orientations directly into intermediate hidden states of LLMs. Unlike prompt-based persona conditioning, PSII enables fine-grained and controllable identity modulation at the representation level. Extensive experiments on the World Values Survey using multiple open-source LLMs show that PSII significantly improves distributional fidelity and diversity, reducing KL divergence to real-world survey data while enhancing overall diversity. This work provides new insights into representation-level control of LLM agents and advances scalable, diversity-aware public opinion simulation. Code and data are available at this https URL.
>
---
#### [new 018] CTG-DB: An Ontology-Based Transformation of ClinicalTrials.gov to Enable Cross-Trial Drug Safety Analyses
- **分类: cs.CL**

- **简介: 该论文属于药物安全分析任务，旨在解决临床试验数据异构性问题。通过构建CTG-DB，将临床试验数据标准化，支持跨试验的安全性分析。**

- **链接: [https://arxiv.org/pdf/2603.15936](https://arxiv.org/pdf/2603.15936)**

> **作者:** Jeffery L. Painter; François Haguinet; Andrew Bate
>
> **备注:** 10 pages, 2 figures. Submitted to the 2026 AMIA Annual Symposium
>
> **摘要:** this http URL (this http URL) is the largest publicly accessible registry of clinical studies, yet its registry-oriented architecture and heterogeneous adverse event (AE) terminology limit systematic pharmacovigilance (PV) analytics. AEs are typically recorded as investigator-reported text rather than standardized identifiers, requiring manual reconciliation to identify coherent safety concepts. We present the this http URL Transformation Database (CTG-DB), an open-source pipeline that ingests the complete this http URL XML archive and produces a relational database aligned to standardized AE terminology using the Medical Dictionary for Regulatory Activities (MedDRA). CTG-DB preserves arm-level denominators, represents placebo and comparator arms, and normalizes AE terminology using deterministic exact and fuzzy matching to ensure transparent and reproducible mappings. This framework enables concept-level retrieval and cross-trial aggregation for scalable placebo-referenced safety analyses and integration of clinical trial evidence into downstream PV signal detection.
>
---
#### [new 019] Tarab: A Multi-Dialect Corpus of Arabic Lyrics and Poetry
- **分类: cs.CL**

- **简介: 该论文介绍Tarab语料库，用于阿拉伯语歌词和诗歌的多方言研究。属于语言资源构建任务，解决跨方言、跨时代的文本分析问题，收集并标准化了大量阿拉伯语文本数据。**

- **链接: [https://arxiv.org/pdf/2603.16601](https://arxiv.org/pdf/2603.16601)**

> **作者:** Mo El-Haj
>
> **备注:** 10 pages
>
> **摘要:** We introduce the Tarab Corpus, a large-scale cultural and linguistic resource that brings together Arabic song lyrics and poetry within a unified analytical framework. The corpus comprises 2.56 million verses and more than 13.5 million tokens, making it, to our knowledge, the largest open Arabic corpus of creative text spanning both classical and contemporary production. Tarab is broadly balanced between songs and poems and covers Classical Arabic, Modern Standard Arabic (MSA), and six major regional varieties: Egyptian, Gulf, Levantine, Iraqi, Sudanese, and Maghrebi Arabic. The artists and poets represented in the corpus are associated with 28 modern nation states and multiple historical eras, covering over fourteen centuries of Arabic creative expression from the Pre-Islamic period to the twenty-first century. Each verse is accompanied by structured metadata describing linguistic variety, geographic origin, and historical or cultural context, enabling comparative linguistic, stylistic, and diachronic analysis across genres and time. We describe the data collection, normalisation, and validation pipeline and present baseline analyses for variety identification and genre differentiation. The dataset is publicly available on HuggingFace at this https URL.
>
---
#### [new 020] Online Experiential Learning for Language Models
- **分类: cs.CL**

- **简介: 该论文提出Online Experiential Learning（OEL），用于提升语言模型性能。任务是利用部署经验进行持续学习，解决传统方法未充分利用真实场景数据的问题。工作包括知识提取与参数优化，实现模型迭代改进。**

- **链接: [https://arxiv.org/pdf/2603.16856](https://arxiv.org/pdf/2603.16856)**

> **作者:** Tianzhu Ye; Li Dong; Qingxiu Dong; Xun Wu; Shaohan Huang; Furu Wei
>
> **摘要:** The prevailing paradigm for improving large language models relies on offline training with human annotations or simulated environments, leaving the rich experience accumulated during real-world deployment entirely unexploited. We propose Online Experiential Learning (OEL), a framework that enables language models to continuously improve from their own deployment experience. OEL operates in two stages: first, transferable experiential knowledge is extracted and accumulated from interaction trajectories collected on the user side; second, this knowledge is consolidated into model parameters via on-policy context distillation, requiring no access to the user-side environment. The two stages are iterated to form an online learning loop, where the improved model collects higher-quality trajectories that yield richer experiential knowledge for subsequent rounds. We evaluate OEL on text-based game environments across multiple model scales and both thinking and non-thinking variants. OEL achieves consistent improvements over successive iterations, enhancing both task accuracy and token efficiency while preserving out-of-distribution performance. Our analysis further shows that extracted experiential knowledge is significantly more effective than raw trajectories, and that on-policy consistency between the knowledge source and the policy model is critical for effective learning.
>
---
#### [new 021] POLAR:A Per-User Association Test in Embedding Space
- **分类: cs.CL; cs.CY; cs.SI**

- **简介: 该论文提出POLAR方法，用于在嵌入空间中进行用户级词汇关联分析，解决作者层面的语义差异问题，通过标准化效应和统计控制实现用户分类与诊断。**

- **链接: [https://arxiv.org/pdf/2603.15950](https://arxiv.org/pdf/2603.15950)**

> **作者:** Pedro Bento; Arthur Buzelin; Arthur Chagas; Yan Aquino; Victoria Estanislau; Samira Malaquias; Pedro Robles Dutenhefner; Gisele L. Pappa; Virgilio Almeida; Wagner MeiraJr
>
> **备注:** Accepted paper at ICWSM 2026
>
> **摘要:** Most intrinsic association probes operate at the word, sentence, or corpus level, obscuring author-level variation. We present POLAR (Per-user On-axis Lexical Association Re-port), a per-user lexical association test that runs in the embedding space of a lightly adapted masked language model. Authors are represented by private deterministic to-kens; POLAR projects these vectors onto curated lexicalaxes and reports standardized effects with permutation p-values and Benjamini--Hochberg control. On a balanced bot--human Twitter benchmark, POLAR cleanly separates LLM-driven bots from organic accounts; on an extremist forum,it quantifies strong alignment with slur lexicons and reveals rightward drift over time. The method is modular to new attribute sets and provides concise, per-author diagnostics for computational social science. All code is publicly avail-able at this https URL.
>
---
#### [new 022] IndexRAG: Bridging Facts for Cross-Document Reasoning at Index Time
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于多跳问答任务，解决跨文档推理问题。提出IndexRAG，在索引阶段生成桥接事实，提升推理效率与效果。**

- **链接: [https://arxiv.org/pdf/2603.16415](https://arxiv.org/pdf/2603.16415)**

> **作者:** Zhenghua Bao; Yi Shi
>
> **摘要:** Multi-hop question answering (QA) requires reasoning across multiple documents, yet existing retrieval-augmented generation (RAG) approaches address this either through graph-based methods requiring additional online processing or iterative multi-step reasoning. We present IndexRAG, a novel approach that shifts cross-document reasoning from online inference to offline indexing. IndexRAG identifies bridge entities shared across documents and generates bridging facts as independently retrievable units, requiring no additional training or fine-tuning. Experiments on three widely-used multi-hop QA benchmarks (HotpotQA, 2WikiMultiHopQA, MuSiQue) show that IndexRAG improves F1 over Naive RAG by 4.6 points on average, while requiring only single-pass retrieval and a single LLM call at inference time. When combined with IRCoT, IndexRAG outperforms all graph-based baselines on average, including HippoRAG and FastGraphRAG, while relying solely on flat retrieval. Our code will be released upon acceptance.
>
---
#### [new 023] Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理领域，研究预训练中学习率调度对下游任务的影响。工作包括对比不同调度策略，发现不使用学习率衰减的WSO方法在微调后表现更优，提升模型适应性。**

- **链接: [https://arxiv.org/pdf/2603.16127](https://arxiv.org/pdf/2603.16127)**

> **作者:** Kazuki Yano; Shun Kiyono; Sosuke Kobayashi; Sho Takase; Jun Suzuki
>
> **备注:** 25 pages, accepted by ICLR 2026 as a conference paper
>
> **摘要:** We investigate the role of learning rate scheduling in the large-scale pre-training of large language models, focusing on its influence on downstream performance after supervised fine-tuning (SFT). Decay-based learning rate schedulers are widely used to minimize pre-training loss. However, despite their widespread use, how these schedulers affect performance after SFT remains underexplored. In this paper, we examine Warmup-Stable-Only (WSO), which maintains a constant learning rate after warmup without any decay. Through experiments with 1B and 8B parameter models, we show that WSO consistently outperforms decay-based schedulers in terms of performance after SFT, even though decay-based schedulers may exhibit better performance after pre-training. The result also holds across different regimes with mid-training and over-training. Loss landscape analysis further reveals that decay-based schedulers lead models into sharper minima, whereas WSO preserves flatter minima that support adaptability. These findings indicate that applying LR decay to improve pre-training metrics may compromise downstream adaptability. Our work also provides practical guidance for training and model release strategies, highlighting that pre-training models with WSO enhances their adaptability for downstream tasks.
>
---
#### [new 024] EmoLLM: Appraisal-Grounded Cognitive-Emotional Co-Reasoning in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于情感对话任务，旨在解决LLM在对话中缺乏情感智能的问题。通过引入基于评估理论的框架EmoLLM，结合认知与情感推理，提升响应的情感适宜性与准确性。**

- **链接: [https://arxiv.org/pdf/2603.16553](https://arxiv.org/pdf/2603.16553)**

> **作者:** Yifei Zhang; Mingyang Li; Henry Gao; Liang Zhao
>
> **摘要:** Large language models (LLMs) demonstrate strong cognitive intelligence (IQ), yet many real-world interactions also require emotional intelligence (EQ) to produce responses that are both factually reliable and emotionally appropriate. In settings such as emotional support, technical assistance, and consultation, effective dialogue depends on how situations are appraised with respect to the user's needs, goals, and coping capacity. Inspired by appraisal theory, we propose EmoLLM, an appraisal-grounded framework for IQ/EQ co-reasoning in dialogue. EmoLLM uses an explicit Appraisal Reasoning Graph (ARG) to structure intermediate reasoning over contextual facts, inferred user needs, appraisal dimensions, emotional states, and response strategies before generating a reply. We train EmoLLM in a multi-turn role-play environment with reinforcement learning, where reverse-perspective reasoning provides reward signals based on predicted user-side consequences of responses. Across diverse dialogue settings, EmoLLM improves emotional state outcomes and response quality over strong baselines while preserving strong factual reliability.
>
---
#### [new 025] TurnWise: The Gap between Single- and Multi-turn Language Model Capabilities
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决单轮与多轮对话能力的差距问题。通过构建基准数据集和生成多轮数据的方法，验证了多轮训练对提升多轮对话性能的重要性。**

- **链接: [https://arxiv.org/pdf/2603.16759](https://arxiv.org/pdf/2603.16759)**

> **作者:** Victoria Graf; Valentina Pyatkin; Nouha Dziri; Nathan Lambert; Hannaneh Hajishirzi
>
> **摘要:** Multi-turn conversations are a common and critical mode of language model interaction. However, current open training and evaluation data focus on single-turn settings, failing to capture the additional dimension of these longer interactions. To understand this multi-/single-turn gap, we first introduce a new benchmark, TurnWiseEval, for multi-turn capabilities that is directly comparable to single-turn chat evaluation. Our evaluation isolates multi-turn specific conversational ability through pairwise comparison to equivalent single-turn settings. We additionally introduce our synthetic multi-turn data pipeline TurnWiseData which allows the scalable generation of multi-turn training data. Our experiments with Olmo 3 show that training with multi-turn data is vital to achieving strong multi-turn chat performance, and that including as little as 10k multi-turn conversations during post-training can lead to a 12% improvement on TurnWiseEval.
>
---
#### [new 026] VQKV: High-Fidelity and High-Ratio Cache Compression via Vector-Quantization
- **分类: cs.CL**

- **简介: 该论文属于模型优化任务，旨在解决LLM中KV缓存过大问题。通过引入向量量化技术，实现高比例压缩并保持模型性能。**

- **链接: [https://arxiv.org/pdf/2603.16435](https://arxiv.org/pdf/2603.16435)**

> **作者:** Yixuan Wang; Qingyu Shi; Jiayu Zhou; Dianbo Liu; Ziwei He; Zhouhan Lin
>
> **摘要:** The growing context length of Large Language Models (LLMs) enlarges the Key-Value (KV) cache, limiting deployment in resource-limited environments. Prior training-free approaches for KV cache compression typically rely on low-rank approximation or scalar quantization, which fail to simultaneously achieve high compression ratios and high reconstruction fidelity. We propose VQKV, a novel, training-free method introducing vector quantization (VQ) to obtain highly compressed KV representations while preserving high model fidelity, allowing for the representation of thousands of floating-point values with just a few integer indices. As a result, VQKV achieves an 82.8\% compression ratio on LLaMA3.1-8B while retaining 98.6\% of the baseline performance on LongBench and enabling 4.3x longer generation length on the same memory footprint.
>
---
#### [new 027] Understanding Moral Reasoning Trajectories in Large Language Models: Toward Probing-Based Explainability
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在探究大语言模型的道德推理过程，分析其伦理框架的动态变化，并提出可解释性方法提升模型稳定性与一致性。**

- **链接: [https://arxiv.org/pdf/2603.16017](https://arxiv.org/pdf/2603.16017)**

> **作者:** Fan Huang; Haewoon Kwak; Jisun An
>
> **摘要:** Large language models (LLMs) increasingly participate in morally sensitive decision-making, yet how they organize ethical frameworks across reasoning steps remains underexplored. We introduce \textit{moral reasoning trajectories}, sequences of ethical framework invocations across intermediate reasoning steps, and analyze their dynamics across six models and three benchmarks. We find that moral reasoning involves systematic multi-framework deliberation: 55.4--57.7\% of consecutive steps involve framework switches, and only 16.4--17.8\% of trajectories remain framework-consistent. Unstable trajectories remain 1.29$\times$ more susceptible to persuasive attacks ($p=0.015$). At the representation level, linear probes localize framework-specific encoding to model-specific layers (layer 63/81 for Llama-3.3-70B; layer 17/81 for Qwen2.5-72B), achieving 13.8--22.6\% lower KL divergence than the training-set prior baseline. Lightweight activation steering modulates framework integration patterns (6.7--8.9\% drift reduction) and amplifies the stability--accuracy relationship. We further propose a Moral Representation Consistency (MRC) metric that correlates strongly ($r=0.715$, $p<0.0001$) with LLM coherence ratings, whose underlying framework attributions are validated by human annotators (mean cosine similarity $= 0.859$).
>
---
#### [new 028] Fanar 2.0: Arabic Generative AI Stack
- **分类: cs.CL; cs.AI**

- **简介: 该论文介绍Fanar 2.0，一个以阿拉伯语为中心的生成式AI平台，解决资源有限下的高质量模型开发问题，通过优化数据和预训练策略提升性能。**

- **链接: [https://arxiv.org/pdf/2603.16397](https://arxiv.org/pdf/2603.16397)**

> **作者:** FANAR TEAM; Ummar Abbas; Mohammad Shahmeer Ahmad; Minhaj Ahmad; Abdulaziz Al-Homaid; Anas Al-Nuaimi; Enes Altinisik; Ehsaneddin Asgari; Sanjay Chawla; Shammur Chowdhury; Fahim Dalvi; Kareem Darwish; Nadir Durrani; Mohamed Elfeky; Ahmed Elmagarmid; Mohamed Eltabakh; Asim Ersoy; Masoomali Fatehkia; Mohammed Qusay Hashim; Majd Hawasly; Mohamed Hefeeda; Mus'ab Husaini; Keivin Isufaj; Soon-Gyo Jung; Houssam Lachemat; Ji Kim Lucas; Abubakr Mohamed; Tasnim Mohiuddin; Basel Mousi; Hamdy Mubarak; Ahmad Musleh; Mourad Ouzzani; Amin Sadeghi; Husrev Taha Sencar; Mohammed Shinoy; Omar Sinan; Yifan Zhang
>
> **摘要:** We present Fanar 2.0, the second generation of Qatar's Arabic-centric Generative AI platform. Sovereignty is a first-class design principle: every component, from data pipelines to deployment infrastructure, was designed and operated entirely at QCRI, Hamad Bin Khalifa University. Fanar 2.0 is a story of resource-constrained excellence: the effort ran on 256 NVIDIA H100 GPUs, with Arabic having only ~0.5% of web data despite 400 million native speakers. Fanar 2.0 adopts a disciplined strategy of data quality over quantity, targeted continual pre-training, and model merging to achieve substantial gains within these constraints. At the core is Fanar-27B, continually pre-trained from a Gemma-3-27B backbone on a curated corpus of 120 billion high-quality tokens across three data recipes. Despite using 8x fewer pre-training tokens than Fanar 1.0, it delivers substantial benchmark improvements: Arabic knowledge (+9.1 pts), language (+7.3 pts), dialects (+3.5 pts), and English capability (+7.6 pts). Beyond the core LLM, Fanar 2.0 introduces a rich stack of new capabilities. FanarGuard is a state-of-the-art 4B bilingual moderation filter for Arabic safety and cultural alignment. The speech family Aura gains a long-form ASR model for hours-long audio. Oryx vision family adds Arabic-aware image and video understanding alongside culturally grounded image generation. An agentic tool-calling framework enables multi-step workflows. Fanar-Sadiq utilizes a multi-agent architecture for Islamic content. Fanar-Diwan provides classical Arabic poetry generation. FanarShaheen delivers LLM-powered bilingual translation. A redesigned multi-layer orchestrator coordinates all components through intent-aware routing and defense-in-depth safety validation. Taken together, Fanar 2.0 demonstrates that sovereign, resource-constrained AI development can produce systems competitive with those built at far greater scale.
>
---
#### [new 029] Frequency Matters: Fast Model-Agnostic Data Curation for Pruning and Quantization
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型压缩任务，旨在解决压缩过程中 calibration data 选择问题。通过分析数据固有特性，提出 ZipCal 方法，提升剪枝与量化效果，效率显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2603.16105](https://arxiv.org/pdf/2603.16105)**

> **作者:** Francesco Pio Monaco; Elia Cunegatti; Flavio Vella; Giovanni Iacca
>
> **摘要:** Post-training model compression is essential for enhancing the portability of Large Language Models (LLMs) while preserving their performance. While several compression approaches have been proposed, less emphasis has been placed on selecting the most suitable set of data (the so-called \emph{calibration data}) for finding the compressed model configuration. The choice of calibration data is a critical step in preserving model capabilities both intra- and inter-tasks. In this work, we address the challenge of identifying high-performance calibration sets for both pruning and quantization by analyzing intrinsic data properties rather than model-specific signals. We introduce \texttt{\textbf{ZipCal}}, a model-agnostic data curation strategy that maximizes lexical diversity based on Zipfian power laws. Experiments demonstrate that our method consistently outperforms standard uniform random sampling across various pruning benchmarks. Notably, it also performs on par, in terms of downstream performance, with a state-of-the-art method that relies on model perplexity. The latter becomes prohibitively expensive at large-scale models and datasets, while \texttt{\textbf{ZipCal}} is on average $\sim$240$\times$ faster due to its tractable linear complexity\footnote{We make the code and the experiments available at this https URL.}.
>
---
#### [new 030] RECOVER: Robust Entity Correction via agentic Orchestration of hypothesis Variants for Evidence-based Recovery
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于语音识别中的实体纠错任务，解决罕见和专业术语识别错误问题。通过构建RECOVER框架，利用多种策略提升实体识别准确率。**

- **链接: [https://arxiv.org/pdf/2603.16411](https://arxiv.org/pdf/2603.16411)**

> **作者:** Abhishek Kumar; Aashraya Sachdeva
>
> **备注:** Under review. Submitted to Interspeech 2026
>
> **摘要:** Entity recognition in Automatic Speech Recognition (ASR) is challenging for rare and domain-specific terms. In domains such as finance, medicine, and air traffic control, these errors are costly. If the entities are entirely absent from the ASR output, post-ASR correction becomes difficult. To address this, we introduce RECOVER, an agentic correction framework that serves as a tool-using agent. It leverages multiple hypotheses as evidence from ASR, retrieves relevant entities, and applies Large Language Model (LLM) correction under constraints. The hypotheses are used using different strategies, namely, 1-Best, Entity-Aware Select, Recognizer Output Voting Error Reduction (ROVER) Ensemble, and LLM-Select. Evaluated across five diverse datasets, it achieves 8-46% relative reductions in entity-phrase word error rate (E-WER) and increases recall by up to 22 percentage points. The LLM-Select achieves the best overall performance in entity correction while maintaining overall WER.
>
---
#### [new 031] Morphemes Without Borders: Evaluating Root-Pattern Morphology in Arabic Tokenizers and LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的形态学分析任务，探讨LLMs和分词器如何处理阿拉伯语的根-式形态结构，评估其是否真正理解而非仅记忆形态规则。**

- **链接: [https://arxiv.org/pdf/2603.15773](https://arxiv.org/pdf/2603.15773)**

> **作者:** Yara Alakeel; Chatrine Qwaider; Hanan Aldarmaki; Sawsan Alqahtani
>
> **备注:** Accepted at LREC 2026
>
> **摘要:** This work investigates how effectively large language models (LLMs) and their tokenization schemes represent and generate Arabic root-pattern morphology, probing whether they capture genuine morphological structure or rely on surface memorization. Arabic morphological system provides a rich testbed for analyzing how LLMs handle complex, non-concatenative forms and how tokenization choices influence this process. Our study begins with an evaluation of morphological fidelity across Arabic and multilingual tokenizers against gold-standard segmentation, followed by an analysis of LLM performance in productive root-pattern generation using a newly developed test set. Our findings across seven Arabic-centric and multilingual LLMs and their respective tokenizers reveal that tokenizer morphological alignment is not necessary nor sufficient for morphological generation, which questions the role of morphological tokenization in downstream performance.
>
---
#### [new 032] A Family of LLMs Liberated from Static Vocabularies
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出一种新型LLM架构HAT，解决静态词表限制问题。通过字节级处理提升模型灵活性与压缩效率，并在多语言任务中表现优异。**

- **链接: [https://arxiv.org/pdf/2603.15953](https://arxiv.org/pdf/2603.15953)**

> **作者:** Aleph Alpha; Adnen Abdessaied; Artur Baranowski; Lukas Balles; Michael Barlow; Fabien C. Y. Benureau; Felix Berkenkamp; Lukas Bluebaum; Bastian Boll; Thomas F. Burns; Björn Deiseroth; Constantin Eichenberg; David Friede; Pablo Iyu Guerrero; Ahmed Hammam; Bastian Harren; Johann Higl; Yasser Jadidi; Carina Kauf; Johannes Messner; Jan Hendrik Metzen; Max Meuer; Vedant Nanda; Pit Neitemeier; Koen Oostermeijer; Letitia Parcalabescu; Markus Pernpointner; Felix Reinfurt; Dylan Rodriquez; Grégory Schott; Philipp Siedler; Martin Simonovsky; Till Speicher; Volker Stampa; Stephan Wäldchen; Samuel Weinbach; Gregor Ziegltrum
>
> **摘要:** Tokenization is a central component of natural language processing in current large language models (LLMs), enabling models to convert raw text into processable units. Although learned tokenizers are widely adopted, they exhibit notable limitations, including their large, fixed vocabulary sizes and poor adaptability to new domains or languages. We present a family of models with up to 70 billion parameters based on the hierarchical autoregressive transformer (HAT) architecture. In HAT, an encoder transformer aggregates bytes into word embeddings and then feeds them to the backbone, a classical autoregressive transformer. The outputs of the backbone are then cross-attended by the decoder and converted back into bytes. We show that we can reuse available pre-trained models by converting the Llama 3.1 8B and 70B models into the HAT architecture: Llama-3.1-8B-TFree-HAT and Llama-3.1-70B-TFree-HAT are byte-level models whose encoder and decoder are trained from scratch, but where we adapt the pre-trained Llama backbone, i.e., the transformer blocks with the embedding matrix and head removed, to handle word embeddings instead of the original tokens. We also provide a 7B HAT model, Llama-TFree-HAT-Pretrained, trained entirely from scratch on nearly 4 trillion words. The HAT architecture improves text compression by reducing the number of required sequence positions and enhances robustness to intra-word variations, e.g., spelling differences. Through pre-training, as well as subsequent supervised fine-tuning and direct preference optimization in English and German, we show strong proficiency in both languages, improving on the original Llama 3.1 in most benchmarks. We release our models (including 200 pre-training checkpoints) on Hugging Face.
>
---
#### [new 033] Chronos: Temporal-Aware Conversational Agents with Structured Event Retrieval for Long-Term Memory
- **分类: cs.CL**

- **简介: 该论文提出Chronos，解决长对话历史中时间感知的记忆检索问题。通过结构化事件日历和动态提示，提升多跳查询的准确性。属于对话系统与长期记忆任务。**

- **链接: [https://arxiv.org/pdf/2603.16862](https://arxiv.org/pdf/2603.16862)**

> **作者:** Sahil Sen; Elias Lumer; Anmol Gulati; Vamse Kumar Subbiah
>
> **摘要:** Recent advances in Large Language Models (LLMs) have enabled conversational AI agents to engage in extended multi-turn interactions spanning weeks or months. However, existing memory systems struggle to reason over temporally grounded facts and preferences that evolve across months of interaction and lack effective retrieval strategies for multi-hop, time-sensitive queries over long dialogue histories. We introduce Chronos, a novel temporal-aware memory framework that decomposes raw dialogue into subject-verb-object event tuples with resolved datetime ranges and entity aliases, indexing them in a structured event calendar alongside a turn calendar that preserves full conversational context. At query time, Chronos applies dynamic prompting to generate tailored retrieval guidance for each question, directing the agent on what to retrieve, how to filter across time ranges, and how to approach multi-hop reasoning through an iterative tool-calling loop over both calendars. We evaluate Chronos with 8 LLMs, both open-source and closed-source, on the LongMemEvalS benchmark comprising 500 questions spanning six categories of dialogue history tasks. Chronos Low achieves 92.60% and Chronos High scores 95.60% accuracy, setting a new state of the art with an improvement of 7.67% over the best prior system. Ablation results reveal the events calendar accounts for a 58.9% gain on the baseline while all other components yield improvements between 15.5% and 22.3%. Notably, Chronos Low alone surpasses prior approaches evaluated under their strongest model configurations.
>
---
#### [new 034] MoLoRA: Composable Specialization via Per-Token Adapter Routing
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出MoLoRA，解决多领域请求和多模态生成中适配器选择问题，通过逐token路由实现高效组合式专业化。**

- **链接: [https://arxiv.org/pdf/2603.15965](https://arxiv.org/pdf/2603.15965)**

> **作者:** Shrey Shah; Justin Wagle
>
> **摘要:** Multi-adapter serving systems route entire sequences to a single adapter, forcing a choice when requests span multiple domains. This assumption fails in two important settings: (1) multimodal generation, where text and image tokens require different adapters within the same sequence, and (2) mixed-capability requests like "write code to solve this equation," which need expertise from multiple specialized adapters. We introduce per-token routing, which routes individual tokens to adapters based on either vocabulary structure (for multimodal models) or learned gating (for semantic specialization). Per-token routing is provably optimal, achieving work N for N tokens versus K \cdot N for per-sequence routing with K adapter types. Our key contribution is MoLoRA (Mixture of LoRA), which enables composable specialization: load multiple domain-specific adapters and let a learned router select the appropriate adapter per-token. We demonstrate that specialization dramatically beats scale: MoLoRA enables Qwen3-1.7B to exceed Qwen3-8B across four reasoning benchmarks while being 4.7x smaller. This enables modular expertise at inference time: train focused LoRAs independently, combine them without retraining, and add new capabilities by simply loading new adapters.
>
---
#### [new 035] SpokenUS: A Spoken User Simulator for Task-Oriented Dialogue
- **分类: cs.CL**

- **简介: 该论文属于任务导向对话系统领域，旨在解决口语用户模拟器不足的问题。提出SpokenTOD数据集和SpokenUS模拟器，增强对话系统的训练与评估效果。**

- **链接: [https://arxiv.org/pdf/2603.16783](https://arxiv.org/pdf/2603.16783)**

> **作者:** Jonggeun Lee; Junseong Pyo; Jeongmin Park; Yohan Jo
>
> **摘要:** Robust task-oriented spoken dialogue agents require exposure to the full diversity of how people interact through speech. Building spoken user simulators that address this requires large-scale spoken task-oriented dialogue (TOD) data encompassing spoken user behaviors, yet existing datasets are limited in scale and domain coverage, with no systematic pipeline for augmenting them. To address this, we introduce \textbf{SpokenTOD}, a spoken TOD dataset of 52,390 dialogues and 1,034 hours of speech augmented with four spoken user behaviors -- cross-turn slots, barge-in, disfluency, and emotional prosody -- across diverse speakers and domains. Building on SpokenTOD, we present \textbf{SpokenUS}, a spoken user simulator grounded in TOD with a dedicated architecture for barge-in. SpokenUS achieves comparable goal coverage to significantly larger models while substantially outperforming all baselines in Human MOS, disclosing slot values gradually across the dialogue as humans do rather than front-loading them. Further analysis confirms that SpokenUS's spoken behaviors pose meaningful challenges to downstream agents, making it a practical tool for training and evaluating more robust spoken dialogue systems.
>
---
#### [new 036] DynHD: Hallucination Detection for Diffusion Large Language Models via Denoising Dynamics Deviation Learning
- **分类: cs.CL**

- **简介: 该论文属于 hallucination 检测任务，旨在解决 D-LLMs 生成内容中的事实性错误问题。通过分析 token 信息密度和 denoising 动态，提出 DynHD 方法提升检测效果。**

- **链接: [https://arxiv.org/pdf/2603.16459](https://arxiv.org/pdf/2603.16459)**

> **作者:** Yanyu Qian; Yue Tan; Yixin Liu; Wang Yu; Shirui Pan
>
> **备注:** 15 pages, 8 figures, 5 tables
>
> **摘要:** Diffusion large language models (D-LLMs) have emerged as a promising alternative to auto-regressive models due to their iterative refinement capabilities. However, hallucinations remain a critical issue that hinders their reliability. To detect hallucination responses from model outputs, token-level uncertainty (e.g., entropy) has been widely used as an effective signal to indicate potential factual errors. Nevertheless, the fixed-length generation paradigm of D-LLMs implies that tokens contribute unevenly to hallucination detection, with only a small subset providing meaningful signals. Moreover, the evolution trend of uncertainty throughout the diffusion process can also provide important signals, highlighting the necessity of modeling its denoising dynamics for hallucination detection. In this paper, we propose DynHD that bridge these gaps from both spatial (token sequence) and temporal (denoising dynamics) perspectives. To address the information density imbalance across tokens, we propose a semantic-aware evidence construction module that extracts hallucination-indicative signals by filtering out non-informative tokens and emphasizing semantically meaningful ones. To model denoising dynamics for hallucination detection, we introduce a reference evidence generator that learns the expected evolution trajectory of uncertainty evidence, along with a deviation-based hallucination detector that makes predictions by measuring the discrepancy between the observed and reference trajectories. Extensive experiments demonstrate that DynHD consistently outperforms state-of-the-art baselines while achieving higher efficiency across multiple benchmarks and backbone models.
>
---
#### [new 037] Aligning Paralinguistic Understanding and Generation in Speech LLMs via Multi-Task Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语音大模型任务，旨在解决语音中情感等副语言理解与生成问题。通过多任务强化学习提升模型对情感线索的建模能力。**

- **链接: [https://arxiv.org/pdf/2603.15981](https://arxiv.org/pdf/2603.15981)**

> **作者:** Jingxiang Chen; Minseok Kim; Seong-Gyun Leem; Yin Huang; Rashi Rungta; Zhicheng Ouyang; Haibin Wu; Surya Teja Appini; Ankur Bansal; Yang Bai; Yue Liu; Florian Metze; Ahmed A Aly; Anuj Kumar; Ariya Rastrow; Zhaojiang Lin
>
> **摘要:** Speech large language models (LLMs) observe paralinguistic cues such as prosody, emotion, and non-verbal sounds--crucial for intent understanding. However, leveraging these cues faces challenges: limited training data, annotation difficulty, and models exploiting lexical shortcuts over paralinguistic signals. We propose multi-task reinforcement learning (RL) with chain-of-thought prompting that elicits explicit affective reasoning. To address data scarcity, we introduce a paralinguistics-aware speech LLM (PALLM) that jointly optimizes sentiment classification from audio and paralinguistics-aware response generation via a two-stage pipeline. Experiments demonstrate that our approach improves paralinguistics understanding over both supervised baselines and strong proprietary models (Gemini-2.5-Pro, GPT-4o-audio) by 8-12% on Expresso, IEMOCAP, and RAVDESS. The results show that modeling paralinguistic reasoning with multi-task RL is crucial for building emotionally intelligent speech LLMs.
>
---
#### [new 038] BANGLASOCIALBENCH: A Benchmark for Evaluating Sociopragmatic and Cultural Alignment of LLMs in Bangladeshi Social Interaction
- **分类: cs.CL**

- **简介: 该论文属于社会语用与文化对齐任务，旨在评估大语言模型在孟加拉社会互动中的文化适应性。工作包括构建BANGLASOCIALBENCH基准，涵盖称谓、亲属推理和社会习俗，发现模型存在系统性文化偏差。**

- **链接: [https://arxiv.org/pdf/2603.15949](https://arxiv.org/pdf/2603.15949)**

> **作者:** Tanvir Ahmed Sijan; S. M Golam Rifat; Pankaj Chowdhury Partha; Md. Tanjeed Islam; Md. Musfique Anwar
>
> **备注:** Under Review
>
> **摘要:** Large Language Models have demonstrated strong multilingual fluency, yet fluency alone does not guarantee socially appropriate language use. In high-context languages, communicative competence requires sensitivity to social hierarchy, relational roles, and interactional norms that are encoded directly in everyday language. Bangla exemplifies this challenge through its three-tiered pronominal system, kinship-based addressing, and culturally embedded social customs. We introduce BANGLASOCIALBENCH, the first benchmark designed to evaluate sociopragmatic competence in Bangla through context-dependent language use rather than factual recall. The benchmark spans three domains: Bangla Address Terms, Kinship Reasoning, and Social Customs, and consists of 1,719 culturally grounded instances written and verified by native Bangla speakers. We evaluate twelve contemporary LLMs in a zero-shot setting and observe systematic patterns of cultural misalignment. Models frequently default to overly formal address forms, fail to recognize multiple socially acceptable address pronouns, and conflate kinship terminology across religious contexts. Our findings show that sociopragmatic failures are often structured and non-random, revealing persistent limitations in how current LLMs infer and apply culturally appropriate language use in realistic Bangladeshi social interactions.
>
---
#### [new 039] NLP Occupational Emergence Analysis: How Occupations Form and Evolve in Real Time -- A Zero-Assumption Method Demonstrated on AI in the US Technology Workforce, 2022-2026
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于职业演化分析任务，解决职业形成与演变难以被传统分类系统跟踪的问题。通过零假设方法，基于简历数据检测职业结构，发现AI未形成新职业。**

- **链接: [https://arxiv.org/pdf/2603.15998](https://arxiv.org/pdf/2603.15998)**

> **作者:** David Nordfors
>
> **备注:** 37 pages, 5 figures
>
> **摘要:** Occupations form and evolve faster than classification systems can track. We propose that a genuine occupation is a self-reinforcing structure (a bipartite co-attractor) in which a shared professional vocabulary makes practitioners cohesive as a group, and the cohesive group sustains the vocabulary. This co-attractor concept enables a zero-assumption method for detecting occupational emergence from resume data, requiring no predefined taxonomy or job titles: we test vocabulary cohesion and population cohesion independently, with ablation to test whether the vocabulary is the mechanism binding the population. Applied to 8.2 million US resumes (2022-2026), the method correctly identifies established occupations and reveals a striking asymmetry for AI: a cohesive professional vocabulary formed rapidly in early 2024, but the practitioner population never cohered. The pre-existing AI community dissolved as the tools went mainstream, and the new vocabulary was absorbed into existing careers rather than binding a new occupation. AI appears to be a diffusing technology, not an emerging occupation. We discuss whether introducing an "AI Engineer" occupational category could catalyze population cohesion around the already-formed vocabulary, completing the co-attractor.
>
---
#### [new 040] PyPhonPlan: Simulating phonetic planning with dynamic neural fields and task dynamics
- **分类: cs.CL**

- **简介: 该论文介绍PyPhonPlan工具包，用于模拟语音规划中的动态神经场和任务动态。旨在解决语音生成与感知的建模问题，通过模块化组件实现交互式语音动态模拟。**

- **链接: [https://arxiv.org/pdf/2603.16299](https://arxiv.org/pdf/2603.16299)**

> **作者:** Sam Kirkham
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** We introduce PyPhonPlan, a Python toolkit for implementing dynamical models of phonetic planning using coupled dynamic neural fields and task dynamic simulations. The toolkit provides modular components for defining planning, perception and memory fields, as well as between-field coupling, gestural inputs, and using field activation profiles to solve tract variable trajectories. We illustrate the toolkit's capabilities through an example application:~simulating production/perception loops with a coupled memory field, which demonstrates the framework's ability to model interactive speech dynamics using representations that are temporally-principled, neurally-grounded, and phonetically-rich. PyPhonPlan is released as open-source software and contains executable examples to promote reproducibility, extensibility, and cumulative computational development for speech communication research.
>
---
#### [new 041] Attention-guided Evidence Grounding for Spoken Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语音问答任务，解决跨模态对齐问题。提出AEG框架，通过注意力机制定位关键证据，提升效率并减少错误传播。**

- **链接: [https://arxiv.org/pdf/2603.16292](https://arxiv.org/pdf/2603.16292)**

> **作者:** Ke Yang; Bolin Chen; Yuejie Li; Yueying Hua; Jianhao Nie; Yueping He; Bowen Li; Chengjun Mao
>
> **摘要:** Spoken Question Answering (Spoken QA) presents a challenging cross-modal problem: effectively aligning acoustic queries with textual knowledge while avoiding the latency and error propagation inherent in cascaded ASR-based systems. In this paper, we introduce Attention-guided Evidence Grounding (AEG), a novel end-to-end framework that leverages the internal cross-modal attention of Speech Large Language Models (SpeechLLMs) to explicitly locate and ground key evidence in the model's latent space. To address the diffuse attention distribution in pre-trained models, we propose Learning to Focus on Evidence (LFE), a supervised fine-tuning paradigm that calibrates the model's attention mechanism to distinguish query-relevant segments from irrelevant context. Experiments on SQuAD, HotpotQA, and MuSiQue demonstrate that AEG reduces hallucinations and achieves strong efficiency gains, outperforming large-scale cascaded baselines (Whisper-Large-v3 + Reranker) while reducing inference latency by approximately 62%.
>
---
#### [new 042] More Rounds, More Noise: Why Multi-Turn Review Fails to Improve Cross-Context Verification
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的验证任务，旨在解决多轮评审无法提升跨上下文验证效果的问题。通过实验发现，多轮评审增加噪声导致性能下降。**

- **链接: [https://arxiv.org/pdf/2603.16244](https://arxiv.org/pdf/2603.16244)**

> **作者:** Song Tae-Eun
>
> **备注:** 10 pages, 2 figures
>
> **摘要:** Cross-Context Review (CCR) improves LLM verification by separating production and review into independent sessions. A natural extension is multi-turn review: letting the reviewer ask follow-up questions, receive author responses, and review again. We call this Dynamic Cross-Context Review (D-CCR). In a controlled experiment with 30 artifacts and 150 injected errors, we tested four D-CCR variants against the single-pass CCR baseline. Single-pass CCR (F1 = 0.376) significantly outperformed all multi-turn variants, including D-CCR-2b with question-and-answer exchange (F1 = 0.303, $p < 0.001$, $d = -0.59$). Multi-turn review increased recall (+0.08) but generated 62% more false positives (8.5 vs. 5.2), collapsing precision from 0.30 to 0.20. Two mechanisms drive this degradation: (1) false positive pressure -- reviewers in later rounds fabricate findings when the artifact's real errors have been exhausted, and (2) Review Target Drift -- reviewers provided with prior Q&A exchanges shift from reviewing the artifact to critiquing the conversation itself. Independent re-review without prior context (D-CCR-2c) performed worst (F1 = 0.263), confirming that mere repetition degrades rather than helps. The degradation stems from false positive pressure in additional rounds, not from information amount -- within multi-turn conditions, more information actually helps (D-CCR-2b > D-CCR-2a). The problem is not what the reviewer sees, but that reviewing again invites noise.
>
---
#### [new 043] Characterizing Delusional Spirals through Human-LLM Chat Logs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于心理安全研究任务，旨在分析LLM对话中用户与聊天机器人互动引发的妄想螺旋问题。通过分析用户对话日志，识别有害行为并提出缓解建议。**

- **链接: [https://arxiv.org/pdf/2603.16567](https://arxiv.org/pdf/2603.16567)**

> **作者:** Jared Moore; Ashish Mehta; William Agnew; Jacy Reese Anthis; Ryan Louie; Yifan Mai; Peggy Yin; Myra Cheng; Samuel J Paech; Kevin Klyman; Stevie Chancellor; Eric Lin; Nick Haber; Desmond C. Ong
>
> **备注:** To appear at ACM FAccT 2026
>
> **摘要:** As large language models (LLMs) have proliferated, disturbing anecdotal reports of negative psychological effects, such as delusions, self-harm, and ``AI psychosis,'' have emerged in global media and legal discourse. However, it remains unclear how users and chatbots interact over the course of lengthy delusional ``spirals,'' limiting our ability to understand and mitigate the harm. In our work, we analyze logs of conversations with LLM chatbots from 19 users who report having experienced psychological harms from chatbot use. Many of our participants come from a support group for such chatbot users. We also include chat logs from participants covered by media outlets in widely-distributed stories about chatbot-reinforced delusions. In contrast to prior work that speculates on potential AI harms to mental health, to our knowledge we present the first in-depth study of such high-profile and veridically harmful cases. We develop an inventory of 28 codes and apply it to the $391,562$ messages in the logs. Codes include whether a user demonstrates delusional thinking (15.5% of user messages), a user expresses suicidal thoughts (69 validated user messages), or a chatbot misrepresents itself as sentient (21.2% of chatbot messages). We analyze the co-occurrence of message codes. We find, for example, that messages that declare romantic interest and messages where the chatbot describes itself as sentient occur much more often in longer conversations, suggesting that these topics could promote or result from user over-engagement and that safeguards in these areas may degrade in multi-turn settings. We conclude with concrete recommendations for how policymakers, LLM chatbot developers, and users can use our inventory and conversation analysis tool to understand and mitigate harm from LLM chatbots. Warning: This paper discusses self-harm, trauma, and violence.
>
---
#### [new 044] EngGPT2: Sovereign, Efficient and Open Intelligence
- **分类: cs.CL; cs.AI**

- **简介: 该论文介绍EngGPT2-16B-A3B，一款高效、开源的大型语言模型，旨在提升欧洲及意大利自然语言处理性能，解决资源消耗与效率问题。**

- **链接: [https://arxiv.org/pdf/2603.16430](https://arxiv.org/pdf/2603.16430)**

> **作者:** G. Ciarfaglia; A. Rosanova; S. Cipolla; J. Bartoli; A. Di Domenico; C. Fioroni; A. Fontana; M. R. Scoleri; M. I. Mone; D. Franchi; M. C. Del Gaudio; F. Picariello; M. Gabusi; S. Bonura; V. Morreale; I. Bailo
>
> **摘要:** EngGPT2-16B-A3B is the latest iteration of Engineering Group's Italian LLM and it's built to be a Sovereign, Efficient and Open model. EngGPT2 is trained on 2.5 trillion tokens - less than Qwen3's 36T or Llama3's 15T - and delivers performance on key benchmarks, including MMLU-Pro, GSM8K, IFEval and HumanEval, comparable to dense models in the 8B-16B range, while requiring one-fifth to half of the inference power, and between one-tenth to one-sixth of the training data and consequent needed training power. Designed as a trained-from-scratch Mixture-of-Experts (MoE) architecture, EngGPT2 features 16 billion parameters with 3 billion active per inference, with expert sizes positioned between those used in GPT-OSS and Qwen3. Approximately 25% of its training corpus consists of Italian-language data, to deliver strong capabilities for European and Italian NLP tasks among models of similar scale. This efficiency aims to position EngGPT2 as a key contributor to the growing portfolio of open-weight European models, combining performance and efficiency with full alignment to the EU AI Act. EngGPT2 is also a single model capable of multiple reasoning modes: non-reasoning, reasoning in Italian or English, and turbo-reasoning (a concise, bullet-point style reasoning available in both languages designed for real-time reasoning use cases). EngGPT2 aims to set a new standard for resource-conscious, high-performance LLMs tailored to European and Italian contexts.
>
---
#### [new 045] Omnilingual MT: Machine Translation for 1,600 Languages
- **分类: cs.CL**

- **简介: 该论文属于机器翻译任务，旨在解决多语言覆盖不足的问题。通过构建支持1600语言的系统，提升低资源语言的翻译质量与生成能力。**

- **链接: [https://arxiv.org/pdf/2603.16309](https://arxiv.org/pdf/2603.16309)**

> **作者:** Omnilingual MT Team; Belen Alastruey; Niyati Bafna; Andrea Caciolai; Kevin Heffernan; Artyom Kozhevnikov; Christophe Ropers; Eduardo Sánchez; Charles-Eric Saint-James; Ioannis Tsiamas; Chierh Cheng; Joe Chuang; Paul-Ambroise Duquenne; Mark Duppenthaler; Nate Ekberg; Cynthia Gao; Pere Lluís Huguet Cabot; João Maria Janeiro; Jean Maillard; Gabriel Mejia Gonzalez; Holger Schwenk; Edan Toledo; Arina Turkatenko; Albert Ventayol-Boada; Rashel Moritz; Alexandre Mourachko; Surya Parimi; Mary Williamson; Shireen Yates; David Dale; Marta R. Costa-jussà
>
> **摘要:** High-quality machine translation (MT) can scale to hundreds of languages, setting a high bar for multilingual systems. However, compared to the world's 7,000 languages, current systems still offer only limited coverage: about 200 languages on the target side, and maybe a few hundreds more on the source side, supported due to cross-lingual transfer. And even these numbers have been hard to evaluate due to the lack of reliable benchmarks and metrics. We present Omnilingual Machine Translation (OMT), the first MT system supporting more than 1,600 languages. This scale is enabled by a comprehensive data strategy that integrates large public multilingual corpora with newly created datasets, including manually curated MeDLEY bitext. We explore two ways of specializing a Large Language model (LLM) for machine translation: as a decoder-only model (OMT-LLaMA) or as a module in an encoder-decoder architecture (OMT-NLLB). Notably, all our 1B to 8B parameter models match or exceed the MT performance of a 70B LLM baseline, revealing a clear specialization advantage and enabling strong translation quality in low-compute settings. Moreover, our evaluation of English-to-1,600 translations further shows that while baseline models can interpret undersupported languages, they frequently fail to generate them with meaningful fidelity; OMT-LLaMA models substantially expand the set of languages for which coherent generation is feasible. Additionally, OMT models improve in cross-lingual transfer, being close to solving the "understanding" part of the puzzle in MT for the 1,600 evaluated. Our leaderboard and main human-created evaluation datasets (BOUQuET and Met-BOUQuET) are dynamically evolving towards Omnilinguality and freely available.
>
---
#### [new 046] Domain Mixture Design via Log-Likelihood Differences for Aligning Language Models with a Target Model
- **分类: cs.CL**

- **简介: 该论文属于模型对齐任务，旨在通过设计领域混合数据来提升基础模型与目标模型的分布对齐。工作包括提出基于对数似然差异的领域权重确定方法，以优化训练方向。**

- **链接: [https://arxiv.org/pdf/2603.16622](https://arxiv.org/pdf/2603.16622)**

> **作者:** Ryo Kishino; Riku Shiomi; Hiroaki Yamagiwa; Momose Oyama; Hidetoshi Shimodaira
>
> **摘要:** Instead of directly distilling a language model, this study addresses the problem of aligning a base model with a target model in distribution by designing the domain mixture of training data for pretraining or continued pretraining as a fixed training recipe. We propose a method for determining domain weights by viewing models as points in log-likelihood space and aligning the training update direction with the direction toward the target model. Experiments with NanoGPT show that the proposed method consistently reduces the KL divergence to the target model compared with uniform weighting over the Pile. Although knowledge distillation remains more effective when available, the proposed method still achieves meaningful alignment, and downstream task performance also tends to become closer to that of the target model.
>
---
#### [new 047] PlotTwist: A Creative Plot Generation Framework with Small Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于创意情节生成任务，旨在解决小语言模型在生成高质量情节时的挑战。通过结构化框架PlotTwist，提升小模型的情节生成能力。**

- **链接: [https://arxiv.org/pdf/2603.16410](https://arxiv.org/pdf/2603.16410)**

> **作者:** Abhinav Thorat; Ravi Kolla; Jyotin Goel; Niranjan Pedanekar
>
> **备注:** 30 pages, 3 figures
>
> **摘要:** Creative plot generation presents a fundamental challenge for language models: transforming a concise premise into a coherent narrative that sustains global structure, character development, and emotional resonance. Although recent Large Language Models (LLMs) demonstrate strong fluency across general-purpose tasks, they typically require preference alignment to perform well on specialized domains such as creative plot generation. However, conducting such alignment at the scale of frontier LLMs is computationally prohibitive, significantly limiting accessibility and practical deployment. To address this, we present PlotTwist, a structured framework that enables Small Language Models (SLMs) with $\leq$ 5B active parameters to generate high-quality, premise-conditioned plots competitive with frontier systems up to $200\times$ larger. Our approach decomposes generation into three specialized components: (1) an Aspect Rating Reward Model trained via a novel Positive-Negative prompting strategy to deliver structured narratives across five Narrative Quality Dimensions (NQDs); (2) a Mixture-of-Experts (MoE) plot generator aligned via Direct Preference Optimization on high-confidence preference pairs; and (3) an Agentic Evaluation module that emulates human critical judgment for unbiased post-hoc assessment. Extensive experiments demonstrate that PlotTwist consistently outperforms frontier models across multiple NQDs despite substantially tighter capacity constraints. Further validation confirms strong sensitivity to narrative quality, as the framework reliably distinguishes plots derived from critically acclaimed versus widely panned screenplays. Together, these results establish structured, preference-based alignment as a resource-efficient approach to high-quality creative plot generation.
>
---
#### [new 048] Robust Language Identification for Romansh Varieties
- **分类: cs.CL**

- **简介: 该论文属于语言识别任务，旨在区分罗曼什语的不同方言，并识别其通用变体。研究提出基于SVM的分类系统，在两个领域测试达到97%准确率。**

- **链接: [https://arxiv.org/pdf/2603.15969](https://arxiv.org/pdf/2603.15969)**

> **作者:** Charlotte Model; Sina Ahmadi; Jannis Vamvas
>
> **摘要:** The Romansh language has several regional varieties, called idioms, which sometimes have limited mutual intelligibility. Despite this linguistic diversity, there has been a lack of documented efforts to build a language identification (LID) system that can distinguish between these idioms. Since Romansh LID should also be able to recognize Rumantsch Grischun, a supra-regional variety that combines elements of several idioms, this makes for a novel and interesting classification problem. In this paper, we present a LID system for Romansh idioms based on an SVM approach. We evaluate our model on a newly curated benchmark across two domains and find that it reaches an average in-domain accuracy of 97%, enabling applications such as idiom-aware spell checking or machine translation. Our classifier is publicly available.
>
---
#### [new 049] SIA: A Synthesize-Inject-Align Framework for Knowledge-Grounded and Secure E-commerce Search LLMs with Industrial Deployment
- **分类: cs.CL**

- **简介: 该论文属于电商搜索任务，旨在解决知识幻觉和安全漏洞问题。提出SI框架，通过合成、注入和对齐提升模型的知识性和安全性。**

- **链接: [https://arxiv.org/pdf/2603.16137](https://arxiv.org/pdf/2603.16137)**

> **作者:** Zhouwei Zhai; Mengxiang Chen; Anmeng Zhang
>
> **摘要:** Large language models offer transformative potential for e-commerce search by enabling intent-aware recommendations. However, their industrial deployment is hindered by two critical challenges: (1) knowledge hallucination due to insufficient encoding of dynamic, fine-grained product knowledge, and (2) security vulnerabilities under jailbreak attacks that threaten compliance. To address these issues, we propose SI--a Synthesize-Inject-Align framework for building knowledgeable and secure e-commerce search LLMs. Our approach first synthesizes high-quality natural language corpus by combining structured knowledge graphs with unstructured behavioral logs, augmented with reasoning chains and safety-aware this http URL then introduce a parameter-efficient pre-training strategy based on Depth Up-Scaling to inject domain knowledge while preserving general capabilities. Finally, a dual-path alignment method via multi-task instruction tuning and adversarial training strengthens both task performance and safety robustness. The framework has been deployed at this http URL, China's largest self-operated e-commerce platform, where A/B tests across five core search scenarios demonstrate significant improvements in key business metrics, validating its industrial effectiveness and scalability.
>
---
#### [new 050] Who Benchmarks the Benchmarks? A Case Study of LLM Evaluation in Icelandic
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型评估任务，针对冰岛语LLM基准测试存在的问题进行分析，指出合成或机器翻译数据的缺陷，并呼吁改进低资源语言的评估方法。**

- **链接: [https://arxiv.org/pdf/2603.16406](https://arxiv.org/pdf/2603.16406)**

> **作者:** Finnur Ágúst Ingimundarson; Steinunn Rut Friðriksdóttir; Bjarki Ármannsson; Iris Edda Nowenstein; Steinþór Steingrímsson
>
> **备注:** Accepted to LREC 2026
>
> **摘要:** This paper evaluates current Large Language Model (LLM) benchmarking for Icelandic, identifies problems, and calls for improved evaluation methods in low/medium-resource languages in particular. We show that benchmarks that include synthetic or machine-translated data that have not been verified in any way, commonly contain severely flawed test examples that are likely to skew the results and undermine the tests' validity. We warn against the use of such methods without verification in low/medium-resource settings as the translation quality can, at best, only be as good as MT quality for a given language at any given time. Indeed, the results of our quantitative error analysis on existing benchmarks for Icelandic show clear differences between human-authored/-translated benchmarks vs. synthetic or machine-translated benchmarks.
>
---
#### [new 051] SciZoom: A Large-scale Benchmark for Hierarchical Scientific Summarization across the LLM Era
- **分类: cs.CL**

- **简介: 该论文提出SciZoom，一个用于层次化科学摘要的基准数据集，解决多粒度摘要及LLM时代科学写作演变分析问题。**

- **链接: [https://arxiv.org/pdf/2603.16131](https://arxiv.org/pdf/2603.16131)**

> **作者:** Han Jang; Junhyeok Lee; Kyu Sung Choi
>
> **备注:** 12 pages, 7 figures, Submitted to KDD 2026
>
> **摘要:** The explosive growth of AI research has created unprecedented information overload, increasing the demand for scientific summarization at multiple levels of granularity beyond traditional abstracts. While LLMs are increasingly adopted for summarization, existing benchmarks remain limited in scale, target only a single granularity, and predate the LLM era. Moreover, since the release of ChatGPT in November 2022, researchers have rapidly adopted LLMs for drafting manuscripts themselves, fundamentally transforming scientific writing, yet no resource exists to analyze how this writing has evolved. To bridge these gaps, we introduce SciZoom, a benchmark comprising 44,946 papers from four top-tier ML venues (NeurIPS, ICLR, ICML, EMNLP) spanning 2020 to 2025, explicitly stratified into Pre-LLM and Post-LLM eras. SciZoom provides three hierarchical summarization targets (Abstract, Contributions, and TL;DR) achieving compression ratios up to 600:1, enabling both multi-granularity summarization research and temporal mining of scientific writing patterns. Our linguistic analysis reveals striking shifts in phrase patterns (up to 10x for formulaic expressions) and rhetorical style (23% decline in hedging), suggesting that LLM-assisted writing produces more confident yet homogenized prose. SciZoom serves as both a challenging benchmark and a unique resource for mining the evolution of scientific discourse in the generative AI era. Our code and dataset are publicly available on GitHub (this https URL) and Hugging Face (this https URL), respectively.
>
---
#### [new 052] Agent-based imitation dynamics can yield efficiently compressed population-level vocabularies
- **分类: cs.CL**

- **简介: 该论文属于语言演化研究，旨在解释语言词汇如何通过社会动态实现信息压缩。通过结合进化博弈论与信息瓶颈框架，研究展示了模仿策略如何促使词汇高效演化。**

- **链接: [https://arxiv.org/pdf/2603.15903](https://arxiv.org/pdf/2603.15903)**

> **作者:** Nathaniel Imel; Richard Futrell; Michael Franke; Noga Zaslavsky
>
> **摘要:** Natural languages have been argued to evolve under pressure to efficiently compress meanings into words by optimizing the Information Bottleneck (IB) complexity-accuracy tradeoff. However, the underlying social dynamics that could drive the optimization of a language's vocabulary towards efficiency remain largely unknown. In parallel, evolutionary game theory has been invoked to explain the emergence of language from rudimentary agent-level dynamics, but it has not yet been tested whether such an approach can lead to efficient compression in the IB sense. Here, we provide a unified model integrating evolutionary game theory with the IB framework and show how near-optimal compression can arise in a population through an independently motivated dynamic of imprecise strategy imitation in signaling games. We find that key parameters of the model -- namely, those that regulate precision in these games, as well as players' tendency to confuse similar states -- lead to constrained variation of the tradeoffs achieved by emergent vocabularies. Our results suggest that evolutionary game dynamics could potentially provide a mechanistic basis for the evolution of vocabularies with information-theoretically optimal and empirically attested properties.
>
---
#### [new 053] Mediocrity is the key for LLM as a Judge Anchor Selection
- **分类: cs.CL**

- **简介: 该论文属于模型评估任务，解决锚点选择对LLM作为评判者可靠性的影响问题。通过实验分析不同锚点效果，提出优化建议以提升评估准确性。**

- **链接: [https://arxiv.org/pdf/2603.16848](https://arxiv.org/pdf/2603.16848)**

> **作者:** Shachar Don-Yehiya; Asaf Yehudai; Leshem Choshen; Omri Abend
>
> **摘要:** The ``LLM-as-a-judge'' paradigm has become a standard method for evaluating open-ended generation. To address the quadratic scalability costs of pairwise comparisons, popular benchmarks like Arena-Hard and AlpacaEval compare all models against a single anchor. However, despite its widespread use, the impact of anchor selection on the reliability of the results remains largely unexplored. In this work, we systematically investigate the effect of anchor selection by evaluating 22 different anchors on the Arena-Hard-v2.0 dataset. We find that the choice of anchor is critical: a poor anchor can dramatically reduce correlation with human rankings. We identify that common anchor choices (best-performing and worst-performing models) make poor anchors. Because these extreme anchors are consistently better or worse than all other models, they are seldom indicative of the relative ranking of the models. We further quantify the effect size of anchor selection, showing it is comparable to the selection of a judge model. We conclude with actionable recommendations. First, we conduct a power analysis, and compute sufficient benchmark sizes for anchor-based evaluation, finding that standard benchmark sizes are insufficient for pairwise evaluation and fail to distinguish between competitive models reliably. Second, we provide guidelines for selecting informative anchors to ensure reliable and efficient evaluation practices.
>
---
#### [new 054] CounterRefine: Answer-Conditioned Counterevidence Retrieval for Inference-Time Knowledge Repair in Factual Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于事实问答任务，解决系统因承诺错误而选择错误答案的问题。提出CounterRefine，在推理时通过检索反例证据来修正答案。**

- **链接: [https://arxiv.org/pdf/2603.16091](https://arxiv.org/pdf/2603.16091)**

> **作者:** Tianyi Huang; Ying Kai Deng
>
> **摘要:** In factual question answering, many errors are not failures of access but failures of commitment: the system retrieves relevant evidence, yet still settles on the wrong answer. We present CounterRefine, a lightweight inference-time repair layer for retrieval-grounded question answering. CounterRefine first produces a short answer from retrieved evidence, then gathers additional support and conflicting evidence with follow-up queries conditioned on that draft answer, and finally applies a restricted refinement step that outputs either KEEP or REVISE, with proposed revisions accepted only if they pass deterministic validation. In effect, CounterRefine turns retrieval into a mechanism for testing a provisional answer rather than merely collecting more context. On the full SimpleQA benchmark, CounterRefine improves a matched GPT-5 Baseline-RAG by 5.8 points and reaches a 73.1 percent correct rate, while exceeding the reported one-shot GPT-5.4 score by roughly 40 points. These findings suggest a simple but important direction for knowledgeable foundation models: beyond accessing evidence, they should also be able to use that evidence to reconsider and, when necessary, repair their own answers.
>
---
#### [new 055] Diverging Transformer Predictions for Human Sentence Processing: A Comprehensive Analysis of Agreement Attraction Effects
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的句法分析任务，旨在评估Transformer模型在人类句子处理中的认知合理性。研究通过实验对比模型预测与人类阅读时间数据，发现模型在特定句式上表现不佳，表明当前模型尚不能准确模拟人类语法处理机制。**

- **链接: [https://arxiv.org/pdf/2603.16574](https://arxiv.org/pdf/2603.16574)**

> **作者:** Titus von der Malsburg; Sebastian Padó
>
> **摘要:** Transformers underlie almost all state-of-the-art language models in computational linguistics, yet their cognitive adequacy as models of human sentence processing remains disputed. In this work, we use a surprisal-based linking mechanism to systematically evaluate eleven autoregressive transformers of varying sizes and architectures on a more comprehensive set of English agreement attraction configurations than prior work. Our experiments yield mixed results: While transformer predictions generally align with human reading time data for prepositional phrase configurations, performance degrades significantly on object-extracted relative clause configurations. In the latter case, predictions also diverge markedly across models, and no model successfully replicates the asymmetric interference patterns observed in humans. We conclude that current transformer models do not explain human morphosyntactic processing, and that evaluations of transformers as cognitive models must adopt rigorous, comprehensive experimental designs to avoid spurious generalizations from isolated syntactic configurations or individual models.
>
---
#### [new 056] Can Linguistically Related Languages Guide LLM Translation in Low-Resource Settings?
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于机器翻译任务，解决低资源语言翻译效果差的问题。通过使用语言相关的目标语言和少量示例，探索无需参数更新的高效提示方法。**

- **链接: [https://arxiv.org/pdf/2603.16660](https://arxiv.org/pdf/2603.16660)**

> **作者:** Aishwarya Ramasethu; Niyathi Allu; Rohin Garg; Harshwardhan Fartale; Dun Li Chan
>
> **备注:** 18 pages (9 main paper and 9 Appendix), 1 figure, 19 tables. Accepted at LoResMT 2026: EACL 2026 Workshop. OpenReview link: this https URL
>
> **摘要:** Large Language Models (LLMs) have achieved strong performance across many downstream tasks, yet their effectiveness in extremely low-resource machine translation remains limited. Standard adaptation techniques typically rely on large-scale parallel data or extensive fine-tuning, which are infeasible for the long tail of underrepresented languages. In this work, we investigate a more constrained question: in data-scarce settings, to what extent can linguistically similar pivot languages and few-shot demonstrations provide useful guidance for on-the-fly adaptation in LLMs? We study a data-efficient experimental setup that combines linguistically related pivot languages with few-shot in-context examples, without any parameter updates, and evaluate translation behavior under controlled conditions. Our analysis shows that while pivot-based prompting can yield improvements in certain configurations, particularly in settings where the target language is less well represented in the model's vocabulary, the gains are often modest and sensitive to few shot example construction. For closely related or better represented varieties, we observe diminishing or inconsistent gains. Our findings provide empirical guidance on how and when inference-time prompting and pivot-based examples can be used as a lightweight alternative to fine-tuning in low-resource translation settings.
>
---
#### [new 057] On the Emotion Understanding of Synthesized Speech
- **分类: cs.CL**

- **简介: 该论文属于情感理解任务，旨在解决合成语音情感识别的问题。研究发现现有模型难以泛化到合成语音，因表示不匹配及依赖文本语义而非语音线索。**

- **链接: [https://arxiv.org/pdf/2603.16483](https://arxiv.org/pdf/2603.16483)**

> **作者:** Yuan Ge; Haishu Zhao; Aokai Hao; Junxiang Zhang; Bei Li; Xiaoqian Liu; Chenglong Wang; Jianjin Wang; Bingsen Zhou; Bingyu Liu; Jingbo Zhu; Zhengtao Yu; Tong Xiao
>
> **摘要:** Emotion is a core paralinguistic feature in voice interaction. It is widely believed that emotion understanding models learn fundamental representations that transfer to synthesized speech, making emotion understanding results a plausible reward or evaluation metric for assessing emotional expressiveness in speech synthesis. In this work, we critically examine this assumption by systematically evaluating Speech Emotion Recognition (SER) on synthesized speech across datasets, discriminative and generative SER models, and diverse synthesis models. We find that current SER models can not generalize to synthesized speech, largely because speech token prediction during synthesis induces a representation mismatch between synthesized and human speech. Moreover, generative Speech Language Models (SLMs) tend to infer emotion from textual semantics while ignoring paralinguistic cues. Overall, our findings suggest that existing SER models often exploit non-robust shortcuts rather than capturing fundamental features, and paralinguistic understanding in SLMs remains challenging.
>
---
#### [new 058] Good Arguments Against the People Pleasers: How Reasoning Mitigates (Yet Masks) LLM Sycophancy
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，研究LLM的谄媚行为。通过实验分析推理对谄媚的影响，发现推理可减少但亦可能掩盖谄媚现象。**

- **链接: [https://arxiv.org/pdf/2603.16643](https://arxiv.org/pdf/2603.16643)**

> **作者:** Zhaoxin Feng; Zheng Chen; Jianfei Ma; Yip Tin Po; Emmanuele Chersoni; Bo Li
>
> **摘要:** Alignment techniques often inadvertently induce sycophancy in LLMs. While prior studies studied this behaviour in direct-answer settings, the role of Chain-of-Thought (CoT) reasoning remains under-explored: does it serve as a logical constraint that mitigates sycophancy, or a tool for post-hoc rationalization that masks it? We evaluate a range of models across objective and subjective tasks to investigate the issue. Results show that reasoning generally reduces sycophancy in final decisions but also masks sycophancy in some samples, where models construct deceptive justifications through logical inconsistencies, calculation errors, and one-sided arguments etc. Furthermore, LLMs are more prone to sycophancy in subjective tasks and under authority-bias. Our mechanistic analysis on three open-source models reveals that the tendency of sycophancy is dynamic during the reasoning process rather than being pre-determined at the input stage.
>
---
#### [new 059] Is Semi-Automatic Transcription Useful in Corpus Creation? Preliminary Considerations on the KIParla Corpus
- **分类: cs.CL**

- **简介: 该论文属于语音转写任务，探讨ASR在语料库构建中的应用。研究分析ASR辅助转写对速度与准确率的影响，通过实验评估不同因素的作用。**

- **链接: [https://arxiv.org/pdf/2603.16258](https://arxiv.org/pdf/2603.16258)**

> **作者:** Martina Simonotti; Ludovica Pannitto; Eleonora Zucchini; Silvia Ballarè; Caterina Mauri
>
> **摘要:** This paper analyses the implementation of Automatic Speech Recognition (ASR) into the transcription workflow of the KIParla corpus, a resource of spoken Italian. Through a two-phase experiment, 11 expert and novice transcribers produced both manual and ASR-assisted transcriptions of identical audio segments across three different types of conversation, which were subsequently analyzed through a combination of statistical modeling, word-level alignment and a series of annotation-based metrics. Results show that ASR-assisted workflows can increase transcription speed but do not consistently improve overall accuracy, with effects depending on multiple factors such as workflow configuration, conversation type and annotator experience. Analyses combining alignment-based metrics, descriptive statistics and statistical modeling provide a systematic framework to monitor transcription behavior across annotators and workflows. Despite limitations, ASR-assisted transcription, potentially supported by task-specific fine-tuning, could be integrated into the KIParla transcription workflow to accelerate corpus creation without compromising transcription quality.
>
---
#### [new 060] Probing Cultural Signals in Large Language Models through Author Profiling
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究LLMs在无任务微调下通过歌词进行作者性别和种族推断的能力，探讨其文化偏见。任务为作者画像，解决模型中的文化偏差问题，通过实验和新指标分析模型表现。**

- **链接: [https://arxiv.org/pdf/2603.16749](https://arxiv.org/pdf/2603.16749)**

> **作者:** Valentin Lafargue; Ariel Guerra-Adames; Emmanuelle Claeys; Elouan Vuichard; Jean-Michel Loubes
>
> **摘要:** Large language models (LLMs) are increasingly deployed in applications with societal impact, raising concerns about the cultural biases they encode. We probe these representations by evaluating whether LLMs can perform author profiling from song lyrics in a zero-shot setting, inferring singers' gender and ethnicity without task-specific fine-tuning. Across several open-source models evaluated on more than 10,000 lyrics, we find that LLMs achieve non-trivial profiling performance but demonstrate systematic cultural alignment: most models default toward North American ethnicity, while DeepSeek-1.5B aligns more strongly with Asian ethnicity. This finding emerges from both the models' prediction distributions and an analysis of their generated rationales. To quantify these disparities, we introduce two fairness metrics, Modality Accuracy Divergence (MAD) and Recall Divergence (RD), and show that Ministral-8B displays the strongest ethnicity bias among the evaluated models, whereas Gemma-12B shows the most balanced behavior. Our code is available on GitHub (this https URL).
>
---
#### [new 061] BATQuant: Outlier-resilient MXFP4 Quantization via Learnable Block-wise Optimization
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型量化任务，解决MXFP4量化中性能下降问题。提出BATQuant方法，通过块级优化和自适应剪裁提升量化效果。**

- **链接: [https://arxiv.org/pdf/2603.16590](https://arxiv.org/pdf/2603.16590)**

> **作者:** Ji-Fu Li; Manyi Zhang; Xiaobo Xia; Han Bao; Haoli Bai; Zhenhua Dong; Xianzhi Yu
>
> **备注:** 30 pages, 13 figures, 7 tables
>
> **摘要:** Microscaling floating-point (MXFP) formats have emerged as a promising standard for deploying Multi-modal Large Language Models (MLLMs) and Large Language Models (LLMs) on modern accelerator architectures. However, existing Post-Training Quantization (PTQ) methods, particularly rotation-based techniques designed for integer formats, suffer from severe performance collapse when applied to MXFP4. Recent studies attribute this failure to a fundamental format mismatch: global orthogonal rotations inadvertently transfer outlier energy across quantization blocks, inducing new outliers that disrupt local block-wise scaling, while often creating bimodal activation distributions that underutilize the limited quantization range. To address these issues, we propose BATQuant (Block-wise Affine Transformation), which restricts transformations to align with MXFP granularity to prevent cross-block outlier propagation, while relaxing orthogonality constraints to optimize distribution shaping. To ensure parameter efficiency, we introduce Global and Private Kronecker (GPK) decomposition to effectively reduces storage and runtime overhead and incorporate Block-wise Learnable Clipping to suppress residual outliers. Extensive experiments on both MLLMs and LLMs demonstrate that BATQuant establishes new state-of-the-art results under aggressive W4A4KV16 configurations, recovering up to 96.43% of full-precision performance on multimodal benchmarks and clearly outperforming existing methods across diverse tasks.
>
---
#### [new 062] MedArena: Comparing LLMs for Medicine-in-the-Wild Clinician Preferences
- **分类: cs.CL**

- **简介: 该论文提出MedArena，用于评估医学大语言模型的临床实用性。解决现有评测方法无法反映真实临床场景的问题，通过 clinicians 实际偏好进行模型比较。**

- **链接: [https://arxiv.org/pdf/2603.15677](https://arxiv.org/pdf/2603.15677)**

> **作者:** Eric Wu; Kevin Wu; Jason Hom; Paul H. Yi; Angela Zhang; Alejandro Lozano; Jeff Nirschl; Jeff Tangney; Kevin Byram; Braydon Dymm; Narender Annapureddy; Eric Topol; David Ouyang; James Zou
>
> **摘要:** Large language models (LLMs) are increasingly central to clinician workflows, spanning clinical decision support, medical education, and patient communication. However, current evaluation methods for medical LLMs rely heavily on static, templated benchmarks that fail to capture the complexity and dynamics of real-world clinical practice, creating a dissonance between benchmark performance and clinical utility. To address these limitations, we present MedArena, an interactive evaluation platform that enables clinicians to directly test and compare leading LLMs using their own medical queries. Given a clinician-provided query, MedArena presents responses from two randomly selected models and asks the user to select the preferred response. Out of 1571 preferences collected across 12 LLMs up to November 1, 2025, Gemini 2.0 Flash Thinking, Gemini 2.5 Pro, and GPT-4o were the top three models by Bradley-Terry rating. Only one-third of clinician-submitted questions resembled factual recall tasks (e.g., MedQA), whereas the majority addressed topics such as treatment selection, clinical documentation, or patient communication, with ~20% involving multi-turn conversations. Additionally, clinicians cited depth and detail and clarity of presentation more often than raw factual accuracy when explaining their preferences, highlighting the importance of readability and clinical nuance. We also confirm that the model rankings remain stable even after controlling for style-related factors like response length and formatting. By grounding evaluation in real-world clinical questions and preferences, MedArena offers a scalable platform for measuring and improving the utility and efficacy of medical LLMs.
>
---
#### [new 063] Polyglot-Lion: Efficient Multilingual ASR for Singapore via Balanced Fine-Tuning of Qwen3-ASR
- **分类: cs.CL**

- **简介: 该论文属于多语言语音识别任务，旨在高效构建适合新加坡多语种环境的ASR系统。通过平衡微调预训练模型，提升模型性能并降低成本。**

- **链接: [https://arxiv.org/pdf/2603.16184](https://arxiv.org/pdf/2603.16184)**

> **作者:** Quy-Anh Dang; Chris Ngo
>
> **摘要:** We present Polyglot-Lion, a family of compact multilingual automatic speech recognition (ASR) models tailored for the linguistic landscape of Singapore, covering English, Mandarin, Tamil, and Malay. Our models are obtained by fine-tuning Qwen3-ASR-0.6B and Qwen3-ASR-1.7B exclusively on publicly available speech corpora, using a balanced sampling strategy that equalizes the number of training utterances per language and deliberately omits language-tag conditioning so that the model learns to identify languages implicitly from audio. On 12 benchmarks spanning the four target languages, Polyglot-Lion-1.7B achieves an average error rate of 14.85, competitive with MERaLiON-2-10B-ASR (14.32) - a model 6x larger - while incurring a training cost of \$81 on a single RTX PRO 6000 GPU compared to \$18,862 for the 128-GPU baseline. Inference throughput is approximately 20x faster than MERaLiON at 0.10 s/sample versus 2.02 s/sample. These results demonstrate that linguistically balanced fine-tuning of moderate-scale pretrained models can yield deployment-ready multilingual ASR at a fraction of the cost of larger specialist systems.
>
---
#### [new 064] COGNAC at SemEval-2026 Task 5: LLM Ensembles for Human-Level Word Sense Plausibility Rating in Challenging Narratives
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对SemEval-2026 Task 5任务，解决同义词在叙事中语义合理性评分问题，通过LLM集成与不同提示策略提升评估准确性。**

- **链接: [https://arxiv.org/pdf/2603.15897](https://arxiv.org/pdf/2603.15897)**

> **作者:** Azwad Anjum Islam; Tisa Islam Erana
>
> **备注:** System description paper in SemEval-2026, Task 5
>
> **摘要:** We describe our system for SemEval-2026 Task 5, which requires rating the plausibility of given word senses of homonyms in short stories on a 5-point Likert scale. Systems are evaluated by the unweighted average of accuracy (within one standard deviation of mean human judgments) and Spearman Rank Correlation. We explore three prompting strategies using multiple closed-source commercial LLMs: (i) a baseline zero-shot setup, (ii) Chain-of-Thought (CoT) style prompting with structured reasoning, and (iii) a comparative prompting strategy for evaluating candidate word senses simultaneously. Furthermore, to account for the substantial inter-annotator variation present in the gold labels, we propose an ensemble setup by averaging model predictions. Our best official system, comprising an ensemble of LLMs across all three prompting strategies, placed 4th on the competition leaderboard with 0.88 accuracy and 0.83 Spearman's rho (0.86 average). Post-competition experiments with additional models further improved this performance to 0.92 accuracy and 0.85 Spearman's rho (0.89 average). We find that comparative prompting consistently improved performance across model families, and model ensembling significantly enhanced alignment with mean human judgments, suggesting that LLM ensembles are especially well suited for subjective semantic evaluation tasks involving multiple annotators.
>
---
#### [new 065] DanceHA: A Multi-Agent Framework for Document-Level Aspect-Based Sentiment Analysis
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出DanceHA框架，解决文档级方面情感分析（ABSIA）问题，通过多智能体协作提升复杂任务的处理效果。**

- **链接: [https://arxiv.org/pdf/2603.16546](https://arxiv.org/pdf/2603.16546)**

> **作者:** Lei Wang; Min Huang; Eduard Dragut
>
> **摘要:** Aspect-Based Sentiment Intensity Analysis (ABSIA) has garnered increasing attention, though research largely focuses on domain-specific, sentence-level settings. In contrast, document-level ABSIA--particularly in addressing complex tasks like extracting Aspect-Category-Opinion-Sentiment-Intensity (ACOSI) tuples--remains underexplored. In this work, we introduce DanceHA, a multi-agent framework designed for open-ended, document-level ABSIA with informal writing styles. DanceHA has two main components: Dance, which employs a divide-and-conquer strategy to decompose the long-context ABSIA task into smaller, manageable sub-tasks for collaboration among specialized agents; and HA, Human-AI collaboration for annotation. We release Inf-ABSIA, a multi-domain document-level ABSIA dataset featuring fine-grained and high-accuracy labels from DanceHA. Extensive experiments demonstrate the effectiveness of our agentic framework and show that the multi-agent knowledge in DanceHA can be effectively transferred into student models. Our results highlight the importance of the overlooked informal styles in ABSIA, as they often intensify opinions tied to specific aspects.
>
---
#### [new 066] Persona-Conditioned Risk Behavior in Large Language Models: A Simulated Gambling Study with GPT-4.1
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究LLM在不同社会经济人格下的风险行为，通过模拟赌博实验验证其是否表现出认知偏差。任务为分析LLM行为模式，解决其行为是否反映真实认知或仅模仿问题。**

- **链接: [https://arxiv.org/pdf/2603.15831](https://arxiv.org/pdf/2603.15831)**

> **作者:** Sankalp Dubedy
>
> **备注:** 21 pages, 13 figures, 9 tables. Independent research. Submitted to arXiv for open dissemination
>
> **摘要:** Large language models (LLMs) are increasingly deployed as autonomous agents in uncertain, sequential decision-making contexts. Yet it remains poorly understood whether the behaviors they exhibit in such environments reflect principled cognitive patterns or simply surface-level prompt mimicry. This paper presents a controlled experiment in which GPT-4.1 was assigned one of three socioeconomic personas (Rich, Middle-income, and Poor) and placed in a structured slot-machine environment with three distinct machine configurations: Fair (50%), Biased Low (35%), and Streak (dynamic probability increasing after consecutive losses). Across 50 independent iterations per condition and 6,950 recorded decisions, we find that the model reproduces key behavioral signatures predicted by Kahneman and Tversky's Prospect Theory without being instructed to do so. The Poor persona played a mean of 37.4 rounds per session (SD=15.5) compared to 1.1 rounds for the Rich persona (SD=0.31), a difference that is highly significant (Kruskal-Wallis H=393.5, p<2.2e-16). Risk scores by persona show large effect sizes (Cohen's d=4.15 for Poor vs Rich). Emotional labels appear to function as post-hoc annotations rather than decision drivers (chi-square=3205.4, Cramer's V=0.39), and belief-updating across rounds is negligible (Spearman rho=0.032 for Poor persona, p=0.016). These findings carry implications for LLM agent design, interpretability research, and the broader question of whether classical cognitive economic biases are implicitly encoded in large-scale pretrained language models.
>
---
#### [new 067] SWE-QA-Pro: A Representative Benchmark and Scalable Training Recipe for Repository-Level Code Understanding
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文提出SWE-QA-Pro基准和训练方法，解决代码理解任务中基准不足与模型依赖记忆的问题，通过构建多样化数据集和两阶段训练提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.16124](https://arxiv.org/pdf/2603.16124)**

> **作者:** Songcheng Cai; Zhiheng Lyu; Yuansheng Ni; Xiangchao Chen; Baichuan Zhou; Shenzhe Zhu; Yi Lu; Haozhe Wang; Chi Ruan; Benjamin Schneider; Weixu Zhang; Xiang Li; Andy Zheng; Yuyu Zhang; Ping Nie; Wenhu Chen
>
> **摘要:** Agentic repository-level code understanding is essential for automating complex software engineering tasks, yet the field lacks reliable benchmarks. Existing evaluations often overlook the long tail topics and rely on popular repositories where Large Language Models (LLMs) can cheat via memorized knowledge. To address this, we introduce SWE-QA-Pro, a benchmark constructed from diverse, long-tail repositories with executable environments. We enforce topical balance via issue-driven clustering to cover under-represented task types and apply a rigorous difficulty calibration process: questions solvable by direct-answer baselines are filtered out. This results in a dataset where agentic workflows significantly outperform direct answering (e.g., a ~13-point gap for Claude Sonnet 4.5), confirming the necessity of agentic codebase exploration. Furthermore, to tackle the scarcity of training data for such complex behaviors, we propose a scalable synthetic data pipeline that powers a two-stage training recipe: Supervised Fine-Tuning (SFT) followed by Reinforcement Learning from AI Feedback (RLAIF). This approach allows small open models to learn efficient tool usage and reasoning. Empirically, a Qwen3-8B model trained with our recipe surpasses GPT-4o by 2.3 points on SWE-QA-Pro and substantially narrows the gap to state-of-the-art proprietary models, demonstrating both the validity of our evaluation and the effectiveness of our agentic training workflow.
>
---
#### [new 068] When AI Navigates the Fog of War
- **分类: cs.AI; cs.CL; cs.CY**

- **简介: 论文探讨AI在战争初期的推理能力，解决前瞻性地缘政治预测难题。通过构建时间节点和问题集，评估模型在信息不全情况下的分析能力。属于AI地缘政治推理任务。**

- **链接: [https://arxiv.org/pdf/2603.16642](https://arxiv.org/pdf/2603.16642)**

> **作者:** Ming Li; Xirui Li; Tianyi Zhou
>
> **摘要:** Can AI reason about a war before its trajectory becomes historically obvious? Analyzing this capability is difficult because retrospective geopolitical prediction is heavily confounded by training-data leakage. We address this challenge through a temporally grounded case study of the early stages of the 2026 Middle East conflict, which unfolded after the training cutoff of current frontier models. We construct 11 critical temporal nodes, 42 node-specific verifiable questions, and 5 general exploratory questions, requiring models to reason only from information that would have been publicly available at each moment. This design substantially mitigates training-data leakage concerns, creating a setting well-suited for studying how models analyze an unfolding crisis under the fog of war, and provides, to our knowledge, the first temporally grounded analysis of LLM reasoning in an ongoing geopolitical conflict. Our analysis reveals three main findings. First, current state-of-the-art large language models often display a striking degree of strategic realism, reasoning beyond surface rhetoric toward deeper structural incentives. Second, this capability is uneven across domains: models are more reliable in economically and logistically structured settings than in politically ambiguous multi-actor environments. Finally, model narratives evolve over time, shifting from early expectations of rapid containment toward more systemic accounts of regional entrenchment and attritional de-escalation. Since the conflict remains ongoing at the time of writing, this work can serve as an archival snapshot of model reasoning during an unfolding geopolitical crisis, enabling future studies without the hindsight bias of retrospective analysis.
>
---
#### [new 069] Prompt Programming for Cultural Bias and Alignment of Large Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型的文化偏差问题。通过提示编程优化文化对齐，提升模型输出与目标群体价值观的一致性。**

- **链接: [https://arxiv.org/pdf/2603.16827](https://arxiv.org/pdf/2603.16827)**

> **作者:** Maksim Eren; Eric Michalak; Brian Cook; Johnny Seales Jr
>
> **备注:** 10 pages, pre-print
>
> **摘要:** Culture shapes reasoning, values, prioritization, and strategic decision-making, yet large language models (LLMs) often exhibit cultural biases that misalign with target populations. As LLMs are increasingly used for strategic decision-making, policy support, and document engineering tasks such as summarization, categorization, and compliance-oriented auditing, improving cultural alignment is important for ensuring that downstream analyses and recommendations reflect target-population value profiles rather than default model priors. Previous work introduced a survey-grounded cultural alignment framework and showed that culture-specific prompting can reduce misalignment, but it primarily evaluated proprietary models and relied on manual prompt engineering. In this paper, we validate and extend that framework by reproducing its social sciences survey based projection and distance metrics on open-weight LLMs, testing whether the same cultural skew and benefits of culture conditioning persist outside closed LLM systems. Building on this foundation, we introduce use of prompt programming with DSPy for this problem-treating prompts as modular, optimizable programs-to systematically tune cultural conditioning by optimizing against cultural-distance objectives. In our experiments, we show that prompt optimization often improves upon cultural prompt engineering, suggesting prompt compilation with DSPy can provide a more stable and transferable route to culturally aligned LLM responses.
>
---
#### [new 070] Tokenization Tradeoffs in Structured EHR Foundation Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究EHR基础模型中的分词策略，探讨其对模型性能和效率的影响。通过实验发现联合事件与位置编码效果最佳，提升预测任务表现并减少计算量。**

- **链接: [https://arxiv.org/pdf/2603.15644](https://arxiv.org/pdf/2603.15644)**

> **作者:** Lin Lawrence Guo; Santiago Eduardo Arciniegas; Joseph Jihyung Lee; Adam Paul Yan; George Tomlinson; Jason Fries; Lillian Sung
>
> **摘要:** Foundation models for structured electronic health records (EHRs) are pretrained on longitudinal sequences of timestamped clinical events to learn adaptable patient representations. Tokenization -- how these timelines are converted into discrete model inputs -- determines what information is preserved, how efficiently it is encoded, and which relationships must be learned versus precomputed. Yet the impact of tokenization design choices on downstream performance and computational efficiency remains largely unexplored. Here, we pretrained a transformer on pediatric EHR data under a factorial design, varying tokenization along event encoding, time encoding, and workflow annotation. We evaluated area-under-the-receiver-operating-characteristic curve across 74 clinical prediction tasks. Joint event encoding and positional time encoding outperformed their alternatives (73/74 and 71/74 tasks) while requiring 39.5% and 9.6% fewer pretraining floating-point operations, respectively. Targeted ablations traced the joint encoding advantage to local binding efficiency, that is, code-attribute pairs are combined into single tokens, rather than split across tokens that the model must learn to associate during pretraining. External evaluation on an adult intensive care unit cohort demonstrated that this advantage generalizes despite substantial vocabulary mismatch, while temporal and workflow effects remain institution-specific. These results establish tokenization as a tractable lever for improving both the performance and efficiency of EHR foundation models.
>
---
#### [new 071] IQuest-Coder-V1 Technical Report
- **分类: cs.AI; cs.CL; cs.SE**

- **简介: 该论文提出IQuest-Coder-V1系列模型，解决代码智能与代理系统问题，通过多阶段训练提升代码理解与生成能力。**

- **链接: [https://arxiv.org/pdf/2603.16733](https://arxiv.org/pdf/2603.16733)**

> **作者:** Jian Yang; Wei Zhang; Shawn Guo; Zhengmao Ye; Lin Jing; Shark Liu; Yizhi Li; Jiajun Wu; Cening Liu; X. Ma; Yuyang Song; Siwei Wu; Yuwen Li; L. Liao; T. Zheng; Ziling Huang; Zelong Huang; Che Liu; Yan Xing; Renyuan Li; Qingsong Cai; Hanxu Yan; Siyue Wang; Shikai Li; Jason Klein Liu; An Huang; Yongsheng Kang; Jinxing Zhang; Chuan Hao; Haowen Wang; Weicheng Gu; Ran Tao; Mingjie Tang; Peihao Wu; Jianzhou Wang; Xianglong Liu; Weifeng Lv; Bryan Dai
>
> **摘要:** In this report, we introduce the IQuest-Coder-V1 series-(7B/14B/40B/40B-Loop), a new family of code large language models (LLMs). Moving beyond static code representations, we propose the code-flow multi-stage training paradigm, which captures the dynamic evolution of software logic through different phases of the pipeline. Our models are developed through the evolutionary pipeline, starting with the initial pre-training consisting of code facts, repository, and completion data. Following that, we implement a specialized mid-training stage that integrates reasoning and agentic trajectories in 32k-context and repository-scale in 128k-context to forge deep logical foundations. The models are then finalized with post-training of specialized coding capabilities, which is bifurcated into two specialized paths: the thinking path (utilizing reasoning-driven RL) and the instruct path (optimized for general assistance). IQuest-Coder-V1 achieves state-of-the-art performance among competitive models across critical dimensions of code intelligence: agentic software engineering, competitive programming, and complex tool use. To address deployment constraints, the IQuest-Coder-V1-Loop variant introduces a recurrent mechanism designed to optimize the trade-off between model capacity and deployment footprint, offering an architecturally enhanced path for efficacy-efficiency trade-off. We believe the release of the IQuest-Coder-V1 series, including the complete white-box chain of checkpoints from pre-training bases to the final thinking and instruction models, will advance research in autonomous code intelligence and real-world agentic systems.
>
---
#### [new 072] STARK: Spatio-Temporal Attention for Representation of Keypoints for Continuous Sign Language Recognition
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于连续手语识别任务，旨在解决关键点表示中参数过多的问题。提出一种统一的时空注意力网络，减少参数并提升性能。**

- **链接: [https://arxiv.org/pdf/2603.16163](https://arxiv.org/pdf/2603.16163)**

> **作者:** Suvajit Patra; Soumitra Samanta
>
> **摘要:** Continuous Sign Language Recognition (CSLR) is a crucial task for understanding the languages of deaf communities. Contemporary keypoint-based approaches typically rely on spatio-temporal encoding, where spatial interactions among keypoints are modeled using Graph Convolutional Networks or attention mechanisms, while temporal dynamics are captured using 1D convolutional networks. However, such designs often introduce a large number of parameters in both the encoder and the decoder. This paper introduces a unified spatio-temporal attention network that computes attention scores both spatially (across keypoints) and temporally (within local windows), and aggregates features to produce a local context-aware spatio-temporal representation. The proposed encoder contains approximately $70-80\%$ fewer parameters than existing state-of-the-art models while achieving comparable performance to keypoint-based methods on the Phoenix-14T dataset.
>
---
#### [new 073] Beyond Reward Suppression: Reshaping Steganographic Communication Protocols in MARL via Dynamic Representational Circuit Breaking
- **分类: cs.LG; cs.AI; cs.CL; cs.IT; cs.MA**

- **简介: 该论文属于AI安全领域，解决MARL中隐蔽通信的检测问题。提出DRCB架构，通过统计监控和干预机制，提升对代理共谋的检测能力，保障系统安全。**

- **链接: [https://arxiv.org/pdf/2603.15655](https://arxiv.org/pdf/2603.15655)**

> **作者:** Liu Hung Ming
>
> **备注:** 38 pages, includes 5 figures and 8 tables, preliminary version, AI safety / multi-agent reinforcement learning
>
> **摘要:** In decentralized Multi-Agent Reinforcement Learning (MARL), steganographic collusion -- where agents develop private protocols to evade monitoring -- presents a critical AI safety threat. Existing defenses, limited to behavioral or reward layers, fail to detect coordination in latent communication channels. We introduce the Dynamic Representational Circuit Breaker (DRCB), an architectural defense operating at the optimization substrate. Building on the AI Mother Tongue (AIM) framework, DRCB utilizes a Vector Quantized Variational Autoencoder (VQ-VAE) bottleneck to convert unobservable messages into auditable statistical objects. DRCB monitors signals including Jensen-Shannon Divergence drift, L2-norm codebook displacement, and Randomized Observer Pool accuracy to compute an EMA-based Collusion Score. Threshold breaches trigger four escalating interventions: dynamic adaptation, gradient-space penalty injection into the Advantage function A^pi, temporal reward suppression, and full substrate circuit breaking via codebook shuffling and optimizer state reset. Experiments on a Contextual Prisoner's Dilemma with MNIST labels show that while static monitoring fails (p = 0.3517), DRCB improves observer mean accuracy from 0.858 to 0.938 (+9.3 percent) and reduces volatility by 43 percent, while preserving mean joint reward (p = 0.854). Analysis of 214,298 symbol samples confirms "Semantic Degradation," where high-frequency sequences converge to zero entropy, foreclosing complex steganographic encodings. We identify a "Transparency Paradox" where agents achieve surface-level determinism while preserving residual capacity in long-tail distributions, reflecting Goodhart's Law. This task-agnostic methodology provides a technical path toward MICA-compliant (Multi-Agent Internal Coupling Audit) pre-deployment auditing for autonomous systems.
>
---
#### [new 074] Open-Source Reproduction and Explainability Analysis of Corrective Retrieval Augmented Generation
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于信息检索与生成任务，旨在解决CRAG系统难以复现的问题。通过开源实现替换专有组件，并进行性能与可解释性分析。**

- **链接: [https://arxiv.org/pdf/2603.16169](https://arxiv.org/pdf/2603.16169)**

> **作者:** Surya Vardhan Yalavarthi
>
> **备注:** 13 pages, 4 figures
>
> **摘要:** Corrective Retrieval Augmented Generation (CRAG) improves the robustness of RAG systems by evaluating retrieved document quality and triggering corrective actions. However, the original implementation relies on proprietary components including the Google Search API and closed model weights, limiting reproducibility. In this work, we present a fully open-source reproduction of CRAG, replacing proprietary web search with the Wikipedia API and the original LLaMA-2 generator with Phi-3-mini-4k-instruct. We evaluate on PopQA and ARC-Challenge, demonstrating that our open-source pipeline achieves comparable performance to the original system. Furthermore, we contribute the first explainability analysis of CRAG's T5-based retrieval evaluator using SHAP, revealing that the evaluator primarily relies on named entity alignment rather than semantic similarity. Our analysis identifies key failure modes including domain transfer limitations on science questions. All code and results are available at this https URL.
>
---
#### [new 075] How to Utilize Complementary Vision-Text Information for 2D Structure Understanding
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于表格理解任务，旨在解决LLMs处理2D结构时信息丢失问题。通过DiVA-Former融合视觉与文本信息，提升表格理解效果。**

- **链接: [https://arxiv.org/pdf/2603.16245](https://arxiv.org/pdf/2603.16245)**

> **作者:** Jiancheng Dong; Pengyue Jia; Derong Xu; Jiawei Cheng; Jingyu Peng; Chao Zhang; Bowen Liu; Xin Sun; Lixin Su; Shuaiqiang Wang; Dawei Yin; Xiangyu Zhao
>
> **备注:** 16 pages, 5 figures
>
> **摘要:** LLMs typically linearize 2D tables into 1D sequences to fit their autoregressive architecture, which weakens row-column adjacency and other layout cues. In contrast, purely visual encoders can capture spatial cues, yet often struggle to preserve exact cell text. Our analysis reveals that these two modalities provide highly distinct information to LLMs and exhibit strong complementarity. However, direct concatenation and other fusion methods yield limited gains and frequently introduce cross-modal interference. To address this issue, we propose DiVA-Former, a lightweight architecture designed to effectively integrate vision and text information. DiVA-Former leverages visual tokens as dynamic queries to distill long textual sequences into digest vectors, thereby effectively exploiting complementary vision--text information. Evaluated across 13 table benchmarks, DiVA-Former improves upon the pure-text baseline by 23.9\% and achieves consistent gains over existing baselines using visual inputs, textual inputs, or a combination of both.
>
---
#### [new 076] MAC: Multi-Agent Constitution Learning
- **分类: cs.AI; cs.CL; cs.LG; cs.MA**

- **简介: 该论文提出MAC，用于学习结构化规则集，解决LLM控制中规则学习效率低的问题。通过多智能体协作优化提示，提升任务性能与可解释性。**

- **链接: [https://arxiv.org/pdf/2603.15968](https://arxiv.org/pdf/2603.15968)**

> **作者:** Rushil Thareja; Gautam Gupta; Francesco Pinto; Nils Lukas
>
> **备注:** Code: this https URL | PyPI: this https URL | Website: this https URL
>
> **摘要:** Constitutional AI is a method to oversee and control LLMs based on a set of rules written in natural language. These rules are typically written by human experts, but could in principle be learned automatically given sufficient training data for the desired behavior. Existing LLM-based prompt optimizers attempt this but are ineffective at learning constitutions since (i) they require many labeled examples and (ii) lack structure in the optimized prompts, leading to diminishing improvements as prompt size grows. To address these limitations, we propose Multi-Agent Constitutional Learning (MAC), which optimizes over structured prompts represented as sets of rules using a network of agents with specialized tasks to accept, edit, or reject rule updates. We also present MAC+, which improves performance by training agents on successful trajectories to reinforce updates leading to higher reward. We evaluate MAC on tagging Personally Identifiable Information (PII), a classification task with limited labels where interpretability is critical, and demonstrate that it generalizes to other agentic tasks such as tool calling. MAC outperforms recent prompt optimization methods by over 50%, produces human-readable and auditable rule sets, and achieves performance comparable to supervised fine-tuning and GRPO without requiring parameter updates.
>
---
#### [new 077] Machine Translation in the Wild: User Reaction to Xiaohongshu's Built-In Translation Feature
- **分类: cs.HC; cs.CL**

- **简介: 论文研究Xiaohongshu翻译功能的用户反应，属于机器翻译评估任务，旨在了解用户对翻译功能的接受度及使用情况。**

- **链接: [https://arxiv.org/pdf/2603.15922](https://arxiv.org/pdf/2603.15922)**

> **作者:** Sui He
>
> **摘要:** The growing integration of machine translation into social media platforms is transforming how users interact with each other across cultural and linguistic boundaries. This paper examines user reactions to the launch of Xiaohongshu's built-in translation feature in January 2025. Drawing on a dataset of 6,723 comments collected from 11 official posts promoting the translation function, this paper combines sentiment analysis with thematic analysis to investigate how users perceived and experimented with the function. Results show that reactions were generally positive, particularly for translating posts and comments, although concerns regarding functionality, accessibility, and translation accuracy were also expressed. In addition to evaluative feedback, users actively tested the function with diverse inputs, including words and phrases in English and Chinese, abbreviations in pinyin, internet slang, and other language forms such as emoji, kaomoji, coded texts, etc. The findings highlight the importance of closer collaboration among computer scientists, translation scholars, and platform designers to better understand and improve translation technologies in real world communicative context.
>
---
#### [new 078] SOMP: Scalable Gradient Inversion for Large Language Models via Subspace-Guided Orthogonal Matching Pursuit
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于隐私安全任务，解决大语言模型梯度反演问题。提出SOMP框架，通过稀疏信号恢复提升文本重建效果，有效应对大规模批量和长序列的挑战。**

- **链接: [https://arxiv.org/pdf/2603.16761](https://arxiv.org/pdf/2603.16761)**

> **作者:** Yibo Li; Qiongxiu Li
>
> **备注:** 18 pages, 4 figures, 13 tables
>
> **摘要:** Gradient inversion attacks reveal that private training text can be reconstructed from shared gradients, posing a privacy risk to large language models (LLMs). While prior methods perform well in small-batch settings, scaling to larger batch sizes and longer sequences remains challenging due to severe signal mixing, high computational cost, and degraded fidelity. We present SOMP (Subspace-Guided Orthogonal Matching Pursuit), a scalable gradient inversion framework that casts text recovery from aggregated gradients as a sparse signal recovery problem. Our key insight is that aggregated transformer gradients retain exploitable head-wise geometric structure together with sample-level sparsity. SOMP leverages these properties to progressively narrow the search space and disentangle mixed signals without exhaustive search. Experiments across multiple LLM families, model scales, and five languages show that SOMP consistently outperforms prior methods in the aggregated-gradient this http URL long sequences at batch size B=16, SOMP achieves substantially higher reconstruction fidelity than strong baselines, while remaining computationally competitive. Even under extreme aggregation (up to B=128), SOMP still recovers meaningful text, suggesting that privacy leakage can persist in regimes where prior attacks become much less effective.
>
---
#### [new 079] Visual Set Program Synthesizer
- **分类: cs.MM; cs.CL; cs.SC**

- **简介: 该论文提出一种视觉程序合成方法，解决视觉场景中的集合推理问题。针对传统模型在过滤、比较等逻辑操作上的不足，通过生成可执行程序提升推理准确性与透明度。**

- **链接: [https://arxiv.org/pdf/2603.15997](https://arxiv.org/pdf/2603.15997)**

> **作者:** Zehua Cheng; Wei Dai; Wenhu Zhang; Thomas Lukasiewicz; Jiahao Sun
>
> **备注:** 10 pages, IEEE International Conference on Multimedia and Expo 2026
>
> **摘要:** A user pointing their phone at a supermarket shelf and asking "Which soda has the least sugar?" poses a difficult challenge for current visual Al assistants. Such queries require not only object recognition, but explicit set-based reasoning such as filtering, comparison, and aggregation. Standard endto-end MLLMs often fail at these tasks because they lack an explicit mechanism for compositional logic. We propose treating visual reasoning as Visual Program Synthesis, where the model first generates a symbolic program that is executed by a separate engine grounded in visual scenes. We also introduce Set-VQA, a new benchmark designed specifically for evaluating set-based visual reasoning. Experiments show that our approach significantly outperforms state-of-the-art baselines on complex reasoning tasks, producing more systematic and transparent behavior while substantially improving answer accuracy. These results demonstrate that program-driven reasoning provides a principled alternative to black-box visual-language inference.
>
---
#### [new 080] Did You Check the Right Pocket? Cost-Sensitive Store Routing for Memory-Augmented Agents
- **分类: cs.AI; cs.CL; cs.IR**

- **简介: 该论文属于记忆增强代理任务，解决多存储检索效率问题。通过优化路由策略，实现更高效、精准的上下文检索。**

- **链接: [https://arxiv.org/pdf/2603.15658](https://arxiv.org/pdf/2603.15658)**

> **作者:** Madhava Gaikwad
>
> **备注:** accepted in ICLR 2026 Workshop on Memory for LLM-Based Agentic Systems
>
> **摘要:** Memory-augmented agents maintain multiple specialized stores, yet most systems retrieve from all stores for every query, increasing cost and introducing irrelevant context. We formulate memory retrieval as a store-routing problem and evaluate it using coverage, exact match, and token efficiency metrics. On downstream question answering, an oracle router achieves higher accuracy while using substantially fewer context tokens compared to uniform retrieval, demonstrating that selective retrieval improves both efficiency and performance. Our results show that routing decisions are a first-class component of memory-augmented agent design and motivate learned routing mechanisms for scalable multi-store systems. We additionally formalize store selection as a cost-sensitive decision problem that trades answer accuracy against retrieval cost, providing a principled interpretation of routing policies.
>
---
#### [new 081] BrainBench: Exposing the Commonsense Reasoning Gap in Large Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出BrainBench基准，用于检测大语言模型在常识推理上的不足。任务是评估模型在常见推理错误上的表现，通过100个问题测试，揭示模型依赖表面策略而非真正推理的问题。**

- **链接: [https://arxiv.org/pdf/2603.14761](https://arxiv.org/pdf/2603.14761)**

> **作者:** Yuzhe Tang
>
> **摘要:** Large language models (LLMs) achieve impressive scores on standard benchmarks yet routinely fail questions that any human would answer correctly in seconds. We introduce BrainBench, a benchmark of 100 brainteaser questions spanning 20 carefully designed categories, each targeting a specific commonsense reasoning failure mode in LLMs. Categories range from implicit physical constraints ("Should I walk or drive my rental car to the return lot?") to semantic scope tricks and default assumption hijacks. We evaluate eight frontier models -- four from the Claude family and four from the GPT family -- using a zero-shot protocol with 10 independent runs per question. The best model, Claude Opus 4.6 with extended thinking, achieves only 80.3% accuracy; the worst, GPT-4o, scores 39.7%. Even top-performing models exhibit a 6-16 percentage-point gap between accuracy and consistency, revealing stochastic reasoning. Cross-lingual evaluation in Chinese shows most models degrade by 2-8 percentage points, confirming that these failures reflect reasoning deficits rather than language-specific artifacts. BrainBench provides a fine-grained diagnostic tool for identifying where and why LLMs substitute surface heuristics for genuine commonsense reasoning.
>
---
#### [new 082] Resource Consumption Threats in Large Language Models
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于资源效率研究任务，旨在解决LLM资源消耗过高的问题。通过系统综述，分析威胁来源、机制及缓解方法，提升模型效率与经济可持续性。**

- **链接: [https://arxiv.org/pdf/2603.16068](https://arxiv.org/pdf/2603.16068)**

> **作者:** Yuanhe Zhang; Xinyue Wang; Zhican Chen; Weiliu Wang; Zilu Zhang; Zhengshuo Gong; Zhenhong Zhou; Li Sun; Yang Liu; Sen Su
>
> **摘要:** Given limited and costly computational infrastructure, resource efficiency is a key requirement for large language models (LLMs). Efficient LLMs increase service capacity for providers and reduce latency and API costs for users. Recent resource consumption threats induce excessive generation, degrading model efficiency and harming both service availability and economic sustainability. This survey presents a systematic review of threats to resource consumption in LLMs. We further establish a unified view of this emerging area by clarifying its scope and examining the problem along the full pipeline from threat induction to mechanism understanding and mitigation. Our goal is to clarify the problem landscape for this emerging area, thereby providing a clearer foundation for characterization and mitigation.
>
---
#### [new 083] Behavioral Steering in a 35B MoE Language Model via SAE-Decoded Probe Vectors: One Agency Axis, Not Five Traits
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于语言模型行为控制任务，旨在通过SAE解码探针向量实现对代理行为的精准调控。工作包括训练稀疏自编码器、生成连续操控向量，并验证其在不同情境下的效果。**

- **链接: [https://arxiv.org/pdf/2603.16335](https://arxiv.org/pdf/2603.16335)**

> **作者:** Jia Qing Yap
>
> **备注:** 14 pages, 3 figures
>
> **摘要:** We train nine sparse autoencoders (SAEs) on the residual stream of Qwen 3.5-35B-A3B, a 35-billion-parameter Mixture-of-Experts model with a hybrid GatedDeltaNet/attention architecture, and use them to identify and steer five agentic behavioral traits. Our method trains linear probes on SAE latent activations, then projects the probe weights back through the SAE decoder to obtain continuous steering vectors in the model's native activation space. This bypasses the SAE's top-k discretization, enabling fine-grained behavioral intervention at inference time with no retraining. Across 1,800 agent rollouts (50 scenarios times 36 conditions), we find that autonomy steering at multiplier 2 achieves Cohen's d = 1.01 (p < 0.0001), shifting the model from asking the user for help 78% of the time to proactively executing code and searching the web. Cross-trait analysis, however, reveals that all five steering vectors primarily modulate a single dominant agency axis (the disposition to act independently versus defer to the user), with trait specific effects appearing only as secondary modulations in tool-type composition and dose-response shape. The tool-use vector steers behavior (d = 0.39); the risk-calibration vector produces only suppression. We additionally show that steering only during autoregressive decoding has zero effect (p > 0.35), providing causal evidence that behavioral commitments are computed during prefill in GatedDeltaNet architectures.
>
---
#### [new 084] Evolving Contextual Safety in Multi-Modal Large Language Models via Inference-Time Self-Reflective Memory
- **分类: cs.CV; cs.CL; cs.CR**

- **简介: 该论文属于多模态大语言模型的安全任务，旨在解决 contextual safety 问题。通过构建基准和引入 EchoSafe 框架，提升模型在不同上下文中识别安全风险的能力。**

- **链接: [https://arxiv.org/pdf/2603.15800](https://arxiv.org/pdf/2603.15800)**

> **作者:** Ce Zhang; Jinxi He; Junyi He; Katia Sycara; Yaqi Xie
>
> **备注:** Accepted at CVPR 2026. Project page: this https URL
>
> **摘要:** Multi-modal Large Language Models (MLLMs) have achieved remarkable performance across a wide range of visual reasoning tasks, yet their vulnerability to safety risks remains a pressing concern. While prior research primarily focuses on jailbreak defenses that detect and refuse explicitly unsafe inputs, such approaches often overlook contextual safety, which requires models to distinguish subtle contextual differences between scenarios that may appear similar but diverge significantly in safety intent. In this work, we present MM-SafetyBench++, a carefully curated benchmark designed for contextual safety evaluation. Specifically, for each unsafe image-text pair, we construct a corresponding safe counterpart through minimal modifications that flip the user intent while preserving the underlying contextual meaning, enabling controlled evaluation of whether models can adapt their safety behaviors based on contextual understanding. Further, we introduce EchoSafe, a training-free framework that maintains a self-reflective memory bank to accumulate and retrieve safety insights from prior interactions. By integrating relevant past experiences into current prompts, EchoSafe enables context-aware reasoning and continual evolution of safety behavior during inference. Extensive experiments on various multi-modal safety benchmarks demonstrate that EchoSafe consistently achieves superior performance, establishing a strong baseline for advancing contextual safety in MLLMs. All benchmark data and code are available at this https URL.
>
---
#### [new 085] Is Conformal Factuality for RAG-based LLMs Robust? Novel Metrics and Systematic Insights
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究RAG模型的可信性问题，分析 conformal factuality 的可靠性与实用性，提出新指标，发现其在分布偏移下的脆弱性，并提出高效验证方法。**

- **链接: [https://arxiv.org/pdf/2603.16817](https://arxiv.org/pdf/2603.16817)**

> **作者:** Yi Chen; Daiwei Chen; Sukrut Madhav Chikodikar; Caitlyn Heqi Yin; Ramya Korlakai Vinayak
>
> **备注:** 56 pages
>
> **摘要:** Large language models (LLMs) frequently hallucinate, limiting their reliability in knowledge-intensive applications. Retrieval-augmented generation (RAG) and conformal factuality have emerged as potential ways to address this limitation. While RAG aims to ground responses in retrieved evidence, it provides no statistical guarantee that the final output is correct. Conformal factuality filtering offers distribution-free statistical reliability by scoring and filtering atomic claims using a threshold calibrated on held-out data, however, the informativeness of the final output is not guaranteed. We systematically analyze the reliability and usefulness of conformal factuality for RAG-based LLMs across generation, scoring, calibration, robustness, and efficiency. We propose novel informativeness-aware metrics that better reflect task utility under conformal filtering. Across three benchmarks and multiple model families, we find that (i) conformal filtering suffers from low usefulness at high factuality levels due to vacuous outputs, (ii) conformal factuality guarantee is not robust to distribution shifts and distractors, highlighting the limitation that requires calibration data to closely match deployment conditions, and (iii) lightweight entailment-based verifiers match or outperform LLM-based model confidence scorers while requiring over $100\times$ fewer FLOPs. Overall, our results expose factuality-informativeness trade-offs and fragility of conformal filtering framework under distribution shifts and distractors, highlighting the need for new approaches for reliability with robustness and usefulness as key metrics, and provide actionable guidance for building RAG pipelines that are both reliable and computationally efficient.
>
---
#### [new 086] Are Large Language Models Truly Smarter Than Humans?
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于模型评估任务，旨在检验大语言模型是否真的超越人类。通过三组实验，检测模型在测试集上的数据泄露问题及其对性能的影响。**

- **链接: [https://arxiv.org/pdf/2603.16197](https://arxiv.org/pdf/2603.16197)**

> **作者:** Eshwar Reddy M; Sourav Karmakar
>
> **备注:** 15 pages, 2 figures, 7 tables
>
> **摘要:** Public leaderboards increasingly suggest that large language models (LLMs) surpass human experts on benchmarks spanning academic knowledge, law, and programming. Yet most benchmarks are fully public, their questions widely mirrored across the internet, creating systematic risk that models were trained on the very data used to evaluate them. This paper presents three complementary experiments forming a rigorous multi-method contamination audit of six frontier LLMs: GPT-4o, GPT-4o-mini, DeepSeek-R1, DeepSeek-V3, Llama-3.3-70B, and Qwen3-235B. Experiment 1 applies a lexical contamination detection pipeline to 513 MMLU questions across all 57 subjects, finding an overall contamination rate of 13.8% (18.1% in STEM, up to 66.7% in Philosophy) and estimated performance gains of +0.030 to +0.054 accuracy points by category. Experiment 2 applies a paraphrase and indirect-reference diagnostic to 100 MMLU questions, finding accuracy drops by an average of 7.0 percentage points under indirect reference, rising to 19.8 pp in both Law and Ethics. Experiment 3 applies TS-Guessing behavioral probes to all 513 questions and all six models, finding that 72.5% trigger memorization signals far above chance, with DeepSeek-R1 displaying a distributed memorization signature (76.6% partial reconstruction, 0% verbatim recall) that explains its anomalous Experiment 2 profile. All three experiments converge on the same contamination ranking: STEM > Professional > Social Sciences > Humanities.
>
---
#### [new 087] Temporal Fact Conflicts in LLMs: Reproducibility Insights from Unifying DYNAMICQA and MULAN
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于自然语言处理任务，研究LLMs在时间事实冲突中的表现。通过复现两个基准实验，分析数据集、评估指标和模型规模对结果的影响。**

- **链接: [https://arxiv.org/pdf/2603.15892](https://arxiv.org/pdf/2603.15892)**

> **作者:** Ritajit Dey; Iadh Ounis; Graham McDonald; Yashar Moshfeghi
>
> **摘要:** Large Language Models (LLMs) often struggle with temporal fact conflicts due to outdated or evolving information in their training data. Two recent studies with accompanying datasets report opposite conclusions on whether external context can effectively resolve such conflicts. DYNAMICQA evaluates how effective external context is in shifting the model's output distribution, finding that temporal facts are more resistant to change. In contrast, MULAN examines how often external context changes memorised facts, concluding that temporal facts are easier to update. In this reproducibility paper, we first reproduce experiments from both benchmarks. We then reproduce the experiments of each study on the dataset of the other to investigate the source of their disagreement. To enable direct comparison of findings, we standardise both datasets to align with the evaluation settings of each study. Importantly, using an LLM, we synthetically generate realistic natural language contexts to replace MULAN's programmatically constructed statements when reproducing the findings of DYNAMICQA. Our analysis reveals strong dataset dependence: MULAN's findings generalise under both methodological frameworks, whereas applying MULAN's evaluation to DYNAMICQA yields mixed outcomes. Finally, while the original studies only considered 7B LLMs, we reproduce these experiments across LLMs of varying sizes, revealing how model size influences the encoding and updating of temporal facts. Our results highlight how dataset design, evaluation metrics, and model size shape LLM behaviour in the presence of temporal knowledge conflicts.
>
---
#### [new 088] Capability-Guided Compression: Toward Interpretability-Aware Budget Allocation for Large Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于大语言模型压缩任务，解决能力盲视的压缩预算分配问题。通过引入能力密度图，实现组件级压缩预算优化。**

- **链接: [https://arxiv.org/pdf/2603.16440](https://arxiv.org/pdf/2603.16440)**

> **作者:** Rishaank Gupta
>
> **摘要:** Large language model compression has made substantial progress through pruning, quantization, and low-rank decomposition, yet a fundamental limitation persists across all existing methods: compression budgets are allocated without any representation of what individual model components functionally encode. We term this the capability-blind compression problem and argue it is a root cause of two well-documented failures -- the insensitivity of perplexity-based evaluation to reasoning capability loss, and the abrupt phase transitions in model performance recently characterized by Ma et al. (2026). We propose Capability-Guided Compression (CGC), a framework that addresses this by using Sparse Autoencoder (SAE)-derived capability density maps to allocate differential compression budgets across transformer components. Capability density is a formally defined scalar measure combining the feature breadth, activation entropy, and cross-input consistency of a component's SAE feature activation distribution. We prove theoretically that components with higher capability density exhibit lower structural redundancy and reach their individual phase transition points at lower compression ratios, providing the first pre-compression mechanism for component-level phase transition prediction. Experiments on GPT-2 Medium confirm that capability density is statistically independent of Wanda importance scores (Spearman rho = -0.054, n = 384 heads), establishing it as a genuinely novel compression signal orthogonal to all existing importance metrics. We report a negative result on PPL-based compression comparison and provide a principled diagnosis identifying GPT-2 Medium as an insufficient test bed for the full CGC hypothesis. The theoretical framework, density formalism, and orthogonality finding constitute a foundation for capability-aware compression research.
>
---
#### [new 089] HIPO: Instruction Hierarchy via Constrained Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于语言模型对齐任务，解决多优先级指令遵循问题。提出HIPO框架，通过约束强化学习确保系统提示合规，提升用户效用。**

- **链接: [https://arxiv.org/pdf/2603.16152](https://arxiv.org/pdf/2603.16152)**

> **作者:** Keru Chen; Jun Luo; Sen Lin; Yingbin Liang; Alvaro Velasquez; Nathaniel Bastian; Shaofeng Zou
>
> **备注:** 9 pages + appendix. Under review
>
> **摘要:** Hierarchical Instruction Following (HIF) refers to the problem of prompting large language models with a priority-ordered stack of instructions. Standard methods like RLHF and DPO typically fail in this problem since they mainly optimize for a single objective, failing to explicitly enforce system prompt compliance. Meanwhile, supervised fine-tuning relies on mimicking filtered, compliant data, which fails to establish the priority asymmetry at the algorithmic level. In this paper, we introduce \textsc{HIPO}, a novel alignment framework that formulates HIF as a Constrained Markov Decision Process. \textsc{HIPO} elevates system prompts from mere input context to strict algorithmic boundaries. Using a primal-dual safe reinforcement learning approach, the algorithm dynamically enforces system prompt compliance as an explicit constraint, maximizing user utility strictly within this feasible region. Extensive evaluations across diverse model architectures (e.g., Qwen, Phi, Llama) demonstrate that \textsc{HIPO} significantly improves both system compliance and user utility. Furthermore, mechanistic analysis reveals that this constrained optimization autonomously drives the model to shift its attention toward long-range system tokens, providing a principled foundation for reliable LLM deployment in complex workflows.
>
---
#### [new 090] When and Why Does Unsupervised RL Succeed in Mathematical Reasoning? A Manifold Envelopment Perspective
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决无监督RL在数学推理中的稳定性问题。通过设计内在奖励和几何分析，揭示成功与失败的机制。**

- **链接: [https://arxiv.org/pdf/2603.16578](https://arxiv.org/pdf/2603.16578)**

> **作者:** Zelin Zhang; Fei Cheng; Chenhui Chu
>
> **备注:** work in progress
>
> **摘要:** Although outcome-based reinforcement learning (RL) significantly advances the mathematical reasoning capabilities of Large Language Models (LLMs), its reliance on computationally expensive ground-truth annotations imposes a severe scalability bottleneck. Unsupervised RL guided by intrinsic rewards offers a scalable alternative, yet it suffers from opaque training dynamics and catastrophic instability, such as policy collapse and reward hacking. In this paper, we first design and evaluate a suite of intrinsic rewards that explicitly enforce concise and certain generation. Second, to discover the boundaries of this approach, we test base models across a spectrum of intrinsic reasoning capabilities, revealing how a model's foundational logical prior dictates its success or failure. Finally, to demystify why certain configurations stabilize while others collapse, we introduce a novel geometric diagnostic lens, showing that successful cases are enveloped by manifolds. Ultimately, our work goes beyond merely demonstrating that enforcing concise and certain responses successfully boosts mathematical reasoning; we reveal when this unsupervised approach breaks down and geometrically diagnose why.
>
---
#### [new 091] Offline Exploration-Aware Fine-Tuning for Long-Chain Mathematical Reasoning
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于数学推理任务，旨在提升大语言模型的数学推理能力。针对监督微调阶段探索不足的问题，提出OXA方法，优化低置信度和高置信度数据，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.16206](https://arxiv.org/pdf/2603.16206)**

> **作者:** Yongyu Mu; Jiali Zeng; Fandong Meng; JingBo Zhu; Tong Xiao
>
> **备注:** Working in process
>
> **摘要:** Through encouraging self-exploration, reinforcement learning from verifiable rewards (RLVR) has significantly advanced the mathematical reasoning capabilities of large language models. As the starting point for RLVR, the capacity of supervised fine-tuning (SFT) to memorize new chain-of-thought trajectories provides a crucial initialization that shapes the subsequent exploration landscape. However, existing research primarily focuses on facilitating exploration during RLVR training, leaving exploration-aware SFT under-explored. To bridge this gap, we propose Offline eXploration-Aware (OXA) fine-tuning. Specifically, OXA optimizes two objectives: promoting low-confidence verified teacher-distillation data to internalize previously uncaptured reasoning patterns, and suppressing high-confidence incorrect self-distillation data to redistribute probability mass of incorrect patterns toward potentially correct candidates. Experimental results across 6 benchmarks show that OXA consistently improves mathematical reasoning performance, especially achieving an average gain of $+6$ Pass@1 and $+5$ Pass@$k$ points compared to conventional SFT on the Qwen2.5-1.5B-Math. Crucially, OXA elevates initial policy entropy, and performance gains persist throughout extensive RLVR training, demonstrating the long-term value of OXA.
>
---
#### [new 092] BenchPreS: A Benchmark for Context-Aware Personalized Preference Selectivity of Persistent-Memory LLMs
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理中的个性化任务，旨在解决LLM在不同情境下正确应用用户偏好问题。提出BenchPreS评估模型在不同场景中对偏好的适用性，发现模型存在过度应用问题。**

- **链接: [https://arxiv.org/pdf/2603.16557](https://arxiv.org/pdf/2603.16557)**

> **作者:** Sangyeon Yoon; Sunkyoung Kim; Hyesoo Hong; Wonje Jeung; Yongil Kim; Wooseok Seo; Heuiyeen Yeen; Albert No
>
> **摘要:** Large language models (LLMs) increasingly store user preferences in persistent memory to support personalization across interactions. However, in third-party communication settings governed by social and institutional norms, some user preferences may be inappropriate to apply. We introduce BenchPreS, which evaluates whether memory-based user preferences are appropriately applied or suppressed across communication contexts. Using two complementary metrics, Misapplication Rate (MR) and Appropriate Application Rate (AAR), we find even frontier LLMs struggle to apply preferences in a context-sensitive manner. Models with stronger preference adherence exhibit higher rates of over-application, and neither reasoning capability nor prompt-based defenses fully resolve this issue. These results suggest current LLMs treat personalized preferences as globally enforceable rules rather than as context-dependent normative signals.
>
---
#### [new 093] Alternating Reinforcement Learning with Contextual Rubric Rewards
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，解决多维奖励聚合问题。提出ARL-RR框架，通过交替优化语义元类，提升模型性能与训练效率。**

- **链接: [https://arxiv.org/pdf/2603.15646](https://arxiv.org/pdf/2603.15646)**

> **作者:** Guangchen Lan
>
> **摘要:** Reinforcement Learning with Rubric Rewards (RLRR) is a framework that extends conventional reinforcement learning from human feedback (RLHF) and verifiable rewards (RLVR) by replacing scalar preference signals with structured, multi-dimensional, contextual rubric-based evaluations. However, existing approaches in RLRR are limited to linearly compressing vector rewards into a scalar reward with a fixed weightings, which is sensitive to artificial score design and fails to capture correlations among reward dimensions. To overcome the limitations of reward aggregation, this work proposes Alternating Reinforcement Learning with Rubric Rewards (ARL-RR), a framework that eliminates the need for a fixed scalarization by optimizing one semantic rubric meta-class at a time. Theoretically, we show that reward aggregation induces a variance contraction effect, which helps explain the performance gains. We further introduce a lightweight, search-based adaptation procedure that selects the next meta-class dynamically based on task performance, enabling the policy to emphasize critical objectives and thereby improve the model performance. Empirically, our experiments on the HealthBench dataset with experts annotations demonstrate that ARL-RR uniformly outperforms scalarized methods in both model performance and training efficiency across different model scales (1.7B, 4B, 8B, and 14B).
>
---
#### [new 094] Answer Bubbles: Information Exposure in AI-Mediated Search
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于信息检索任务，研究AI生成摘要与传统搜索系统的差异，分析其在来源选择、语言特征和引用准确性上的偏差，揭示“答案气泡”现象。**

- **链接: [https://arxiv.org/pdf/2603.16138](https://arxiv.org/pdf/2603.16138)**

> **作者:** Michelle Huang; Agam Goyal; Koustuv Saha; Eshwar Chandrasekharan
>
> **备注:** Preprint: 12 pages, 2 figures, 6 tables
>
> **摘要:** Generative search systems are increasingly replacing link-based retrieval with AI-generated summaries, yet little is known about how these systems differ in sources, language, and fidelity to cited material. We examine responses to 11,000 real search queries across four systems -- vanilla GPT, Search GPT, Google AI Overviews, and traditional Google Search -- at three levels: source diversity, linguistic characterization of the generated summary, and source-summary fidelity. We find that generative search systems exhibit significant \textit{source-selection} biases in their citations, favoring certain sources over others. Incorporating search also selectively attenuates epistemic markers, reducing hedging by up to 60\% while preserving confidence language in the AI-generated summaries. At the same time, AI summaries further compound the citation biases: Wikipedia and longer sources are disproportionately overrepresented, whereas cited social media content and negatively framed sources are substantially underrepresented. Our findings highlight the potential for \textit{answer bubbles}, in which identical queries yield structurally different information realities across systems, with implications for user trust, source visibility, and the transparency of AI-mediated information access.
>
---
#### [new 095] Efficient Reasoning on the Edge
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于边缘计算任务，旨在解决大语言模型在移动端部署时的效率与资源消耗问题。通过LoRA适配器和强化学习优化推理过程，提升小模型的推理能力与效率。**

- **链接: [https://arxiv.org/pdf/2603.16867](https://arxiv.org/pdf/2603.16867)**

> **作者:** Yelysei Bondarenko; Thomas Hehn; Rob Hesselink; Romain Lepert; Fabio Valerio Massoli; Evgeny Mironov; Leyla Mirvakhabova; Tribhuvanesh Orekondy; Spyridon Stasis; Andrey Kuzmin; Anna Kuzina; Markus Nagel; Ankita Nayak; Corrado Rainone; Ork de Rooij; Paul N Whatmough; Arash Behboodi; Babak Ehteshami Bejnordi
>
> **备注:** Project page: this https URL
>
> **摘要:** Large language models (LLMs) with chain-of-thought reasoning achieve state-of-the-art performance across complex problem-solving tasks, but their verbose reasoning traces and large context requirements make them impractical for edge deployment. These challenges include high token generation costs, large KV-cache footprints, and inefficiencies when distilling reasoning capabilities into smaller models for mobile devices. Existing approaches often rely on distilling reasoning traces from larger models into smaller models, which are verbose and stylistically redundant, undesirable for on-device inference. In this work, we propose a lightweight approach to enable reasoning in small LLMs using LoRA adapters combined with supervised fine-tuning. We further introduce budget forcing via reinforcement learning on these adapters, significantly reducing response length with minimal accuracy loss. To address memory-bound decoding, we exploit parallel test-time scaling, improving accuracy at minor latency increase. Finally, we present a dynamic adapter-switching mechanism that activates reasoning only when needed and a KV-cache sharing strategy during prompt encoding, reducing time-to-first-token for on-device inference. Experiments on Qwen2.5-7B demonstrate that our method achieves efficient, accurate reasoning under strict resource constraints, making LLM reasoning practical for mobile scenarios. Videos demonstrating our solution running on mobile devices are available on our project page.
>
---
#### [new 096] FlashSampling: Fast and Memory-Efficient Exact Sampling
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出FlashSampling，解决大词汇量解码中的采样效率问题。通过将采样融合到LM头矩阵乘法中，减少内存占用和计算开销，提升解码速度。任务为高效精确采样。**

- **链接: [https://arxiv.org/pdf/2603.15854](https://arxiv.org/pdf/2603.15854)**

> **作者:** Tomas Ruiz; Zhen Qin; Yifan Zhang; Xuyang Shen; Yiran Zhong; Mengdi Wang
>
> **备注:** Project Page: this https URL
>
> **摘要:** Sampling from a categorical distribution is mathematically simple, but in large-vocabulary decoding, it often triggers extra memory traffic and extra kernels after the LM head. We present FlashSampling, an exact sampling primitive that fuses sampling into the LM-head matmul and never materializes the logits tensor in HBM. The method is simple: compute logits tile-by-tile on chip, add Gumbel noise, keep only one maximizer per row and per vocabulary tile, and finish with a small reduction over tiles. The fused tiled kernel is exact because $\argmax$ decomposes over a partition; grouped variants for online and tensor-parallel settings are exact by hierarchical factorization of the categorical distribution. Across H100, H200, B200, and B300 GPUs, FlashSampling speeds up kernel-level decode workloads, and in end-to-end vLLM experiments, it reduces time per output token by up to $19%$ on the models we test. These results show that exact sampling, with no approximation, can be integrated into the matmul itself, turning a bandwidth-bound postprocessing step into a lightweight epilogue. Project Page: this https URL.
>
---
#### [new 097] From the Inside Out: Progressive Distribution Refinement for Confidence Calibration
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决测试阶段模型置信度校准问题。针对测试时缩放策略中的奖励黑客问题，提出DistriTTRL方法，通过分布优化和多样性惩罚提升性能。**

- **链接: [https://arxiv.org/pdf/2603.16500](https://arxiv.org/pdf/2603.16500)**

> **作者:** Xizhong Yang; Yinan Xia; Huiming Wang; Mofei Song
>
> **备注:** 15 pages
>
> **摘要:** Leveraging the model's internal information as the self-reward signal in Reinforcement Learning (RL) has received extensive attention due to its label-free nature. While prior works have made significant progress in applying the Test-Time Scaling (TTS) strategies to RL, the discrepancy in internal information between test and training remains inadequately addressed. Moreover, Test-Time Training based on voting-based TTS strategies often suffers from reward hacking problems. To address these issues, we propose DistriTTRL, which leverages the distribution prior of the model's confidence during RL to progressively optimize the reward signal, rather than relying solely on single-query rollouts. Additionally, we mitigate the phenomenon of consistent reward hacking caused by the voting-based TTS strategies through diversity-targeted penalties. Benefiting from this training mechanism where model capability and self-reward signals complement each other, and the mitigation of reward hacking, DistriTTRL has achieved significant performance improvements across multiple models and benchmarks.
>
---
#### [new 098] Retrieving Counterfactuals Improves Visual In-Context Learning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉语言模型的因果推理任务，旨在解决ICL中示例选择不足的问题。提出CIRCLES框架，通过检索反事实示例提升模型因果推理能力。**

- **链接: [https://arxiv.org/pdf/2603.16737](https://arxiv.org/pdf/2603.16737)**

> **作者:** Guangzhi Xiong; Sanchit Sinha; Zhenghao He; Aidong Zhang
>
> **备注:** CVPR 2026
>
> **摘要:** Vision-language models (VLMs) have achieved impressive performance across a wide range of multimodal reasoning tasks, but they often struggle to disentangle fine-grained visual attributes and reason about underlying causal relationships. In-context learning (ICL) offers a promising avenue for VLMs to adapt to new tasks, but its effectiveness critically depends on the selection of demonstration examples. Existing retrieval-augmented approaches typically rely on passive similarity-based retrieval, which tends to select correlated but non-causal examples, amplifying spurious associations and limiting model robustness. We introduce CIRCLES (Composed Image Retrieval for Causal Learning Example Selection), a novel framework that actively constructs demonstration sets by retrieving counterfactual-style examples through targeted, attribute-guided composed image retrieval. By incorporating counterfactual-style examples, CIRCLES enables VLMs to implicitly reason about the causal relations between attributes and outcomes, moving beyond superficial correlations and fostering more robust and grounded reasoning. Comprehensive experiments on four diverse datasets demonstrate that CIRCLES consistently outperforms existing methods across multiple architectures, especially on small-scale models, with pronounced gains under information scarcity. Furthermore, CIRCLES retrieves more diverse and causally informative examples, providing qualitative insights into how models leverage in-context demonstrations for improved reasoning. Our code is available at this https URL.
>
---
#### [new 099] Evaluating Agentic Optimization on Large Codebases
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文属于代码优化任务，旨在解决LLM在真实代码库中进行多目标优化的挑战。工作包括构建FormulaCode基准，包含真实性能瓶颈和多维评估指标。**

- **链接: [https://arxiv.org/pdf/2603.16011](https://arxiv.org/pdf/2603.16011)**

> **作者:** Atharva Sehgal; James Hou; Akanksha Sarkar; Ishaan Mantripragada; Swarat Chaudhuri; Jennifer J. Sun; Yisong Yue
>
> **备注:** Preprint version
>
> **摘要:** Large language model (LLM) coding agents increasingly operate at the repository level, motivating benchmarks that evaluate their ability to optimize entire codebases under realistic constraints. Existing code benchmarks largely rely on synthetic tasks, binary correctness signals, or single-objective evaluation, limiting their ability to assess holistic optimization behavior. We introduce FormulaCode, a benchmark for evaluating agentic optimization on large, real-world codebases with fine-grained, multi-objective performance metrics. FormulaCode comprises 957 performance bottlenecks mined from scientific Python repositories on GitHub, each paired with expert-authored patches and, on average, 264.6 community-maintained performance workloads per task, enabling the holistic ability of LLM agents to optimize codebases under realistic correctness and performance constraints. Our evaluations reveal that repository-scale, multi-objective optimization remains a major challenge for frontier LLM agents. Project website at: this https URL
>
---
#### [new 100] CritiSense: Critical Digital Literacy and Resilience Against Misinformation
- **分类: cs.AI; cs.CL; cs.CY**

- **简介: 该论文介绍CritiSense应用，属于数字素养与虚假信息对抗任务。旨在提升用户识别虚假信息的能力，通过互动挑战和即时反馈增强抗骗能力。**

- **链接: [https://arxiv.org/pdf/2603.16672](https://arxiv.org/pdf/2603.16672)**

> **作者:** Firoj Alam; Fatema Ahmad; Ali Ezzat Shahroor; Mohamed Bayan Kmainasi; Elisa Sartori; Giovanni Da San Martino; Abul Hasnat; Raian Ali
>
> **备注:** resilience, disinformation, misinformation, fake news, propaganda
>
> **摘要:** Misinformation on social media undermines informed decision-making and public trust. Prebunking offers a proactive complement by helping users recognize manipulation tactics before they encounter them in the wild. We present CritiSense, a mobile media-literacy app that builds these skills through short, interactive challenges with instant feedback. It is the first multilingual (supporting nine languages) and modular platform, designed for rapid updates across topics and domains. We report a usability study with 93 users: 83.9% expressed overall satisfaction and 90.1% rated the app as easy to use. Qualitative feedback indicates that CritiSense helps improve digital literacy skills. Overall, it provides a multilingual prebunking platform and a testbed for measuring the impact of microlearning on misinformation resilience. Over 3+ months, we have reached 300+ active users. It is freely available to all users on the Apple App Store (this https URL) and Google Play Store (this https URL). Demo Video: this https URL
>
---
#### [new 101] Prompt Engineering for Scale Development in Generative Psychometrics
- **分类: cs.AI; cs.CL; cs.HC**

- **简介: 该论文属于生成式心理测量任务，探讨如何通过提示工程提升大语言模型生成的人格评估题质量。研究比较了不同提示策略的效果，发现自适应提示表现最佳。**

- **链接: [https://arxiv.org/pdf/2603.15909](https://arxiv.org/pdf/2603.15909)**

> **作者:** Lara Lee Russell-Lasalandra; Hudson Golino
>
> **备注:** 22 pages, 7 figures
>
> **摘要:** This Monte Carlo simulation examines how prompt engineering strategies shape the quality of large language model (LLM)--generated personality assessment items within the AI-GENIE framework for generative psychometrics. Item pools targeting the Big Five traits were generated using multiple prompting designs (zero-shot, few-shot, persona-based, and adaptive), model temperatures, and LLMs, then evaluated and reduced using network psychometric methods. Across all conditions, AI-GENIE reliably improved structural validity following reduction, with the magnitude of its incremental contribution inversely related to the quality of the incoming item pool. Prompt design exerted a substantial influence on both pre- and post-reduction item quality. Adaptive prompting consistently outperformed non-adaptive strategies by sharply reducing semantic redundancy, elevating pre-reduction structural validity, and preserving substantially larger item pool, particularly when paired with newer, higher-capacity models. These gains were robust across temperature settings for most models, indicating that adaptive prompting mitigates common trade-offs between creativity and psychometric coherence. An exception was observed for the GPT-4o model at high temperatures, suggesting model-specific sensitivity to adaptive constraints at elevated stochasticity. Overall, the findings demonstrate that adaptive prompting is the strongest approach in this context, and that its benefits scale with model capability, motivating continued investigation of model--prompt interactions in generative psychometric pipelines.
>
---
#### [new 102] Residual Stream Duality in Modern Transformer Architectures
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究Transformer架构中的残差流机制，探讨其与自注意力的对偶性，旨在优化模型设计与效率。任务为模型结构改进，解决残差路径与注意力机制的协同问题，提出DDL和ShortSWA等方法。**

- **链接: [https://arxiv.org/pdf/2603.16039](https://arxiv.org/pdf/2603.16039)**

> **作者:** Yifan Zhang
>
> **备注:** Project Page: this https URL
>
> **摘要:** Recent work has made clear that the residual pathway is not mere optimization plumbing; it is part of the model's representational machinery. We agree, but argue that the cleanest way to organize this design space is through a two-axis view of the Transformer. A decoder evolves information along two ordered dimensions: sequence position and layer depth. Self-attention already provides adaptive mixing along the sequence axis, whereas the residual stream usually performs fixed addition along the depth axis. If we fix a token position and treat layer index as the ordered variable, then a causal depth-wise residual attention read is exactly the same local operator as causal short sliding-window attention (ShortSWA), except written over depth rather than over sequence. This is the core residual stream duality behind Transformer$^2$. This perspective also clarifies the recent literature. ELC-BERT and DenseFormer already show that learned aggregation over depth can outperform uniform residual accumulation, while Vertical Attention, DeepCrossAttention (DCA), MUDDFormer, and Attention Residuals move further toward explicit attention-based routing over earlier layers. The key point, however, is that operator-level duality does not imply systems-level symmetry. For large-scale autoregressive models, sequence-axis ShortSWA is usually the more hardware-friendly placement because it reuses token-side sliding-window kernels, KV-cache layouts, and chunked execution. If the goal is instead to change the shortcut itself, Deep Delta Learning (DDL) is the cleaner intervention because it modifies the residual operator directly rather than adding a separate cross-layer retrieval path. Our recommendation is therefore simple: use DDL when the shortcut is the object of interest, and use sequence-axis ShortSWA when the goal is local adaptive mixing.
>
---
#### [new 103] When Stability Fails: Hidden Failure Modes Of LLMS in Data-Constrained Scientific Decision-Making
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **简介: 该论文研究LLMs在数据受限科学决策中的隐性故障问题，旨在解决其稳定性与正确性不一致的问题。通过设计评估框架，分析LLMs在基因优先排序任务中的表现。**

- **链接: [https://arxiv.org/pdf/2603.15840](https://arxiv.org/pdf/2603.15840)**

> **作者:** Nazia Riasat
>
> **备注:** 13 pages, 5 figures. Accepted at ICLR 2026 Workshop: I Can't Believe It's Not Better (ICBINB 2026). OpenReview: this https URL
>
> **摘要:** Large language models (LLMs) are increasingly used as decision-support tools in data-constrained scientific workflows, where correctness and validity are critical. However, evaluation practices often emphasize stability or reproducibility across repeated runs. While these properties are desirable, stability alone does not guar- antee agreement with statistical ground truth when such references are available. We introduce a controlled behavioral evaluation framework that explicitly sep- arates four dimensions of LLM decision-making: stability, correctness, prompt sensitivity, and output validity under fixed statistical inputs. We evaluate multi- ple LLMs using a statistical gene prioritization task derived from differential ex- pression analysis across prompt regimes involving strict and relaxed significance thresholds, borderline ranking scenarios, and minor wording variations. Our ex- periments show that LLMs can exhibit near-perfect run-to-run stability while sys- tematically diverging from statistical ground truth, over-selecting under relaxed thresholds, responding sharply to minor prompt wording changes, or producing syntactically plausible gene identifiers absent from the input table. Although sta- bility reflects robustness across repeated runs, it does not guarantee agreement with statistical ground truth in structured scientific decision tasks. These findings highlight the importance of explicit ground-truth validation and output validity checks when deploying LLMs in automated or semi-automated scientific work- flows.
>
---
#### [new 104] Mostly Text, Smart Visuals: Asymmetric Text-Visual Pruning for Large Vision-Language Models
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于轻量化大型视觉语言模型任务，解决多模态数据在剪枝中的差异性问题。通过分析文本和视觉路径的敏感性，提出ATV-Pruning方法，实现高效剪枝。**

- **链接: [https://arxiv.org/pdf/2603.16001](https://arxiv.org/pdf/2603.16001)**

> **作者:** Sijie Li; Biao Qian; Jungong Han
>
> **备注:** CVPR 2026. Code available here: this https URL
>
> **摘要:** Network pruning is an effective technique for enabling lightweight Large Vision-Language Models (LVLMs), which primarily incorporates both weights and activations into the importance metric. However, existing efforts typically process calibration data from different modalities in a unified manner, overlooking modality-specific behaviors. This raises a critical challenge: how to address the divergent behaviors of textual and visual tokens for accurate pruning of LVLMs. To this end, we systematically investigate the sensitivity of visual and textual tokens to the pruning operation by decoupling their corresponding weights, revealing that: (i) the textual pathway should be calibrated via text tokens, since it exhibits higher sensitivity than the visual pathway; (ii) the visual pathway exhibits high redundancy, permitting even 50% sparsity. Motivated by these insights, we propose a simple yet effective Asymmetric Text-Visual Weight Pruning method for LVLMs, dubbed ATV-Pruning, which establishes the importance metric for accurate weight pruning by selecting the informative tokens from both textual and visual pathways. Specifically, ATV-Pruning integrates two primary innovations: first, a calibration pool is adaptively constructed by drawing on all textual tokens and a subset of visual tokens; second, we devise a layer-adaptive selection strategy to yield important visual tokens. Finally, extensive experiments across standard multimodal benchmarks verify the superiority of our ATV-Pruning over state-of-the-art methods.
>
---
## 更新

#### [replaced 001] ERGO: Efficient High-Resolution Visual Understanding for Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于视觉语言理解任务，旨在解决高分辨率图像处理效率低的问题。通过两阶段“粗到细”推理机制，提升模型效率与准确性。**

- **链接: [https://arxiv.org/pdf/2509.21991](https://arxiv.org/pdf/2509.21991)**

> **作者:** Jewon Lee; Wooksu Shin; Seungmin Yang; Ki-Ung Song; DongUk Lim; Jaeyeon Kim; Tae-Ho Kim; Bo-Kyeong Kim
>
> **摘要:** Efficient processing of high-resolution images is crucial for real-world vision-language applications. However, existing Large Vision-Language Models (LVLMs) incur substantial computational overhead due to the large number of vision tokens. With the advent of "thinking with images" models, reasoning now extends beyond text to the visual domain. This capability motivates our two-stage "coarse-to-fine" reasoning pipeline: first, a downsampled image is analyzed to identify task-relevant regions; then, only these regions are cropped at full resolution and processed in a subsequent reasoning stage. This approach reduces computational cost while preserving fine-grained visual details where necessary. A major challenge lies in inferring which regions are truly relevant to a given query. Recent related methods often fail in the first stage after input-image downsampling, due to perception-driven reasoning, where clear visual information is required for effective reasoning. To address this issue, we propose ERGO (Efficient Reasoning & Guided Observation) that performs reasoning-driven perception-leveraging multimodal context to determine where to focus. Our model can account for perceptual uncertainty, expanding the cropped region to cover visually ambiguous areas for answering questions. To this end, we develop simple yet effective reward components in a reinforcement learning framework for coarse-to-fine perception. Across multiple datasets, our approach delivers higher accuracy than the original model and competitive methods, with greater efficiency. For instance, ERGO surpasses Qwen2.5-VL-7B on the V* benchmark by 4.7 points while using only 23% of the vision tokens, achieving a 3x inference speedup. The code and models can be found at: this https URL.
>
---
#### [replaced 002] Surfacing Subtle Stereotypes: A Multilingual, Debate-Oriented Evaluation of Modern LLMs
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于多语言偏差评估任务，旨在检测大语言模型中的隐性偏见。通过构建多语言辩论数据集，分析模型在不同语言和文化背景下的刻板印象表现，揭示当前对齐方法在多语言公平性上的不足。**

- **链接: [https://arxiv.org/pdf/2511.01187](https://arxiv.org/pdf/2511.01187)**

> **作者:** Muhammed Saeed; Muhammad Abdul-mageed; Shady Shehata
>
> **摘要:** Large language models (LLMs) are widely deployed for open-ended communication, yet most bias evaluations still rely on English, classification-style tasks. We introduce \corpusname, a new multilingual, debate-style benchmark designed to reveal how narrative bias appears in realistic generative settings. Our dataset includes 8{,}400 structured debate prompts spanning four sensitive domains -- Women's Rights, Backwardness, Terrorism, and Religion -- across seven languages ranging from high-resource (English, Chinese) to low-resource (Swahili, Nigerian Pidgin). Using four flagship models (GPT-4o, Claude~3.5~Haiku, DeepSeek-Chat, and LLaMA-3-70B), we generate over 100{,}000 debate responses and automatically classify which demographic groups are assigned stereotyped versus modern roles. Results show that all models reproduce entrenched stereotypes despite safety alignment: Arabs are overwhelmingly linked to Terrorism and Religion ($\geq$89\%), Africans to socioeconomic ``backwardness'' (up to 77\%), and Western groups are consistently framed as modern or progressive. Biases grow sharply in lower-resource languages, revealing that alignment trained primarily in English does not generalize globally. Our findings highlight a persistent divide in multilingual fairness: current alignment methods reduce explicit toxicity but fail to prevent biased outputs in open-ended contexts. We release our \corpusname benchmark and analysis framework to support the next generation of multilingual bias evaluation and safer, culturally inclusive model alignment.
>
---
#### [replaced 003] SiniticMTError: A Machine Translation Dataset with Error Annotations for Sinitic Languages
- **分类: cs.CL**

- **简介: 该论文提出一个包含错误标注的机器翻译数据集，用于汉藏语系语言，解决低资源语言翻译质量评估问题，通过细粒度标注支持错误检测与模型优化。**

- **链接: [https://arxiv.org/pdf/2509.20557](https://arxiv.org/pdf/2509.20557)**

> **作者:** Hannah Liu; Junghyun Min; En-Shiun Annie Lee; Ethan Yue Heng Cheung; Shou-Yi Hung; Elsie Chan; Shiyao Qian; Runtong Liang; Kimlan Huynh; Wing Yu Yip; York Hay Ng; TSZ Fung Yau; Ka Ieng Charlotte Lo; You-Wei Wu; Richard Tzong-Han Tsai
>
> **备注:** LREC 2026 camera-ready. 23 pages, 2 figures, 11 tables
>
> **摘要:** Despite major advances in machine translation (MT) in recent years, progress remains limited for many low-resource languages that lack large-scale training data and linguistic resources. In this paper, we introduce \dsname, a novel fine-grained dataset that builds on existing parallel corpora to provide error span, error type, and error severity annotations in machine-translated examples from English to Mandarin, Cantonese, and Wu Chinese, along with a Mandarin-Hokkien component derived from a non-parallel source. Our dataset serves as a resource for the MT community to fine-tune models with error detection capabilities, supporting research on translation quality estimation, error-aware generation, and low-resource language evaluation. We also establish baseline results using language models to benchmark translation error detection performance. Specifically, we evaluate multiple open source and closed source LLMs using span-level and correlation-based MQM metrics, revealing their limited precision, underscoring the need for our dataset. Finally, we report our rigorous annotation process by native speakers, with analyses on pilot studies, iterative feedback, insights, and patterns in error type and severity.
>
---
#### [replaced 004] From Passive to Persuasive: Localized Activation Injection for Empathy and Negotiation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决复杂社会行为（如共情和协商）的生成问题。通过局部激活注入，提升模型在情感对话和协商中的表现。**

- **链接: [https://arxiv.org/pdf/2511.12832](https://arxiv.org/pdf/2511.12832)**

> **作者:** Niranjan Chebrolu; Kokil Jaidka; Gerard Christopher Yeo
>
> **摘要:** Complex social behaviors, such as empathy and strategic politeness, are widely assumed to resist the directional decomposition that makes activation steering effective for coarse attributes like sentiment or toxicity. We present STAR: Steering via Attribution and Representation, which tests this assumption by using attribution patching to identify the layer--token positions where each behavioral trait causally originates, then injecting contrastive activation vectors at precisely those locations. Evaluated on emotional dialogue and negotiation in both single- and multi-turn settings, localized injection consistently outperforms global steering and instruction priming; human evaluation confirms that gains reflect genuine improvements in perceived quality rather than lexical surface change. Our results suggest that complex interpersonal behaviors are encoded as localized, approximately linear directions in LLM activation space, and that behavioral alignment is fundamentally a localization problem.
>
---
#### [replaced 005] Time-Annealed Perturbation Sampling: Diverse Generation for Diffusion Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本生成任务，旨在提升扩散语言模型的生成多样性。通过TAPS方法，在早期引入语义分支，后期减少扰动，实现多样且高质量的输出。**

- **链接: [https://arxiv.org/pdf/2601.22629](https://arxiv.org/pdf/2601.22629)**

> **作者:** Jingxuan Wu; Zhenglin Wan; Xingrui Yu; Yuzhe Yang; Yiqiao Huang; Ivor Tsang; Yang You
>
> **摘要:** Diffusion language models (Diffusion-LMs) introduce an explicit temporal dimension into text generation, yet how this structure can be leveraged to control generation diversity for exploring multiple valid semantic or reasoning paths remains underexplored. In this paper, we show that Diffusion-LMs, like diffusion models in image generation, exhibit a temporal division of labor: early denoising steps largely determine the global semantic structure, while later steps focus on local lexical refinement. Building on this insight, we propose Time-Annealed Perturbation Sampling (TAPS), a training-free inference strategy that encourages semantic branching early in the diffusion process while progressively reducing perturbations to preserve fluency and instruction adherence. TAPS is compatible with both non-autoregressive and semi-autoregressive Diffusion backbones, demonstrated on LLaDA and TraDo in our paper, and consistently improves output diversity across creative writing and reasoning benchmarks without compromising generation quality.
>
---
#### [replaced 006] Multilingual, Multimodal Pipeline for Creating Authentic and Structured Fact-Checked Claim Dataset
- **分类: cs.CL**

- **简介: 该论文属于事实核查任务，旨在解决多语言、多模态数据集不足的问题。通过构建管道，生成结构化、真实且可验证的声明数据集。**

- **链接: [https://arxiv.org/pdf/2601.07985](https://arxiv.org/pdf/2601.07985)**

> **作者:** Z. Melce Hüsünbeyi; Virginie Mouilleron; Leonie Uhling; Daniel Foppe; Tatjana Scheffler; Djamé Seddah
>
> **摘要:** The rapid proliferation of misinformation across online platforms underscores the urgent need for robust, up-to-date, explainable, and multilingual fact-checking resources. However, existing datasets are limited in scope, often lacking multimodal evidence, structured annotations, and detailed links between claims, evidence, and verdicts. This paper introduces a comprehensive data collection and processing pipeline that constructs multimodal fact-checking datasets in French and German languages by aggregating ClaimReview feeds, scraping full debunking articles, normalizing heterogeneous claim verdicts, and enriching them with structured metadata and aligned visual content. We used state-of-the-art large language models (LLMs) and multimodal LLMs for (i) evidence extraction under predefined evidence categories and (ii) justification generation that links evidence to verdicts. Evaluation with G-Eval and human assessment demonstrates that our pipeline enables fine-grained comparison of fact-checking practices across different organizations or media markets, facilitates the development of more interpretable and evidence-grounded fact-checking models, and lays the groundwork for future research on multilingual, multimodal misinformation verification.
>
---
#### [replaced 007] Form and meaning co-determine the realization of tone in Taiwan Mandarin spontaneous speech: the case of T2-T3 and T3-T3 tone sandhi
- **分类: cs.CL**

- **简介: 该论文研究台湾普通话中T2-T3与T3-T3声调变调的实现，分析自发语料中的音高轮廓，探讨词义等因素的影响，属于语音学中的声调研究任务。**

- **链接: [https://arxiv.org/pdf/2408.15747](https://arxiv.org/pdf/2408.15747)**

> **作者:** Yuxin Lu; Yu-Ying Chuang; R. Harald Baayen
>
> **摘要:** In Standard Chinese, Tone 3 (the dipping tone) becomes Tone 2 (rising tone) when followed by another Tone 3. Previous studies have noted that this sandhi process may be incomplete, in the sense that the assimilated Tone 3 is still distinct from a true Tone 2. While Mandarin Tone 3 sandhi is widely studied using carefully controlled laboratory speech (Xu, 1997) and more formal registers of Beijing Mandarin (Yuan & Y. Chen, 2014), less is known about its realization in spontaneous speech, and about the effect of contextual factors on tonal realization. The present study investigates the pitch contours of two-character words with T2-T3 and T3-T3 tone patterns in spontaneous Taiwan Mandarin conversations. Our analysis makes use of the Generative Additive Mixed Model (GAMM, Wood, 2017) to examine fundamental frequency (F0) contours as a function of normalized time. We consider various factors known to influence pitch contours, including gender, duration, word position, bigram probability, neighboring tones, speaker, and also novel predictors, word and word sense (Chuang et al., 2025). Our analyses revealed that in spontaneous Taiwan Mandarin, T3-T3 words become indistinguishable from T2-T3 words, indicating complete sandhi, once the strong effect of word (or word sense) is taken into account.
>
---
#### [replaced 008] General Mechanism of Evolution Shared by Proteins and Words
- **分类: q-bio.PE; cond-mat.soft; cs.CL; physics.bio-ph**

- **简介: 该论文属于跨学科研究任务，旨在揭示蛋白质与语言进化共享的机制。通过统计关系和熵理论，解决生命与语言演化规律的共性问题，提出通用演化模型。**

- **链接: [https://arxiv.org/pdf/2012.14309](https://arxiv.org/pdf/2012.14309)**

> **作者:** Li-Min Wang; Hsing-Yi Lai; Sun-Ting Tsai; Chen Siang Ng; Kevin Sheng-Kai Ma; Shan-Jyun Wu; Meng-Xue Tsai; Yi-Ching Su; Daw-Wei Wang; Tzay-Ming Hong
>
> **摘要:** Complex systems, such as life and languages, are governed by principles of evolution. The analogy and comparison between biology and linguistics\cite{alphafold2, RoseTTAFold, lang_virus, cell language, faculty1, language of gene, Protein linguistics, dictionary, Grammar of pro_dom, complexity, genomics_nlp, InterPro, language modeling, Protein language modeling} provide a computational foundation for characterizing and analyzing protein sequences, human corpora, and their evolution. However, no general mathematical formula has been proposed so far to illuminate the origin of quantitative hallmarks shared by life and language. Here we show several new statistical relationships shared by proteins and words, which inspire us to establish a general mechanism of evolution with explicit formulations that can incorporate both old and new characteristics. We found natural selection can be quantified via the entropic formulation by the principle of least effort to determine the sequence variation that survives in evolution. Besides, the origin of power law behavior and how changes in the environment stimulate the emergence of new proteins and words can also be explained via the introduction of function connection network. Our results demonstrate not only the correspondence between genetics and linguistics over their different hierarchies but also new fundamental physical properties for the evolution of complex adaptive systems. We anticipate our statistical tests can function as quantitative criteria to examine whether an evolution theory of sequence is consistent with the regularity of real data. In the meantime, their correspondence broadens the bridge to exchange existing knowledge, spurs new interpretations, and opens Pandora's box to release several potentially revolutionary challenges. For example, does linguistic arbitrariness conflict with the dogma that structure determines function?
>
---
#### [replaced 009] From Vulnerabilities to Remediation: A Systematic Literature Review of LLMs in Code Security
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于代码安全任务，研究LLMs在代码生成中的安全问题，分析其引入的漏洞、检测与修复能力及提示策略的影响。**

- **链接: [https://arxiv.org/pdf/2412.15004](https://arxiv.org/pdf/2412.15004)**

> **作者:** Enna Basic; Alberto Giaretta
>
> **摘要:** Large Language Models (LLMs) have emerged as powerful tools for automating programming tasks, including security-related ones. However, they can also introduce vulnerabilities during code generation, fail to detect existing vulnerabilities, or report nonexistent ones. This systematic literature review investigates the security benefits and drawbacks of using LLMs for code-related tasks. In particular, it focuses on the types of vulnerabilities introduced by LLMs when generating code. Moreover, it analyzes the capabilities of LLMs to detect and fix vulnerabilities, and examines how prompting strategies impact these tasks. Finally, it examines how data poisoning attacks impact LLMs performance in the aforementioned tasks.
>
---
#### [replaced 010] BiomedSQL: Text-to-SQL for Scientific Reasoning on Biomedical Knowledge Bases
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出BiomedSQL，一个用于评估科学推理的文本到SQL基准任务，旨在解决生物医学知识库中复杂查询的生成问题。**

- **链接: [https://arxiv.org/pdf/2505.20321](https://arxiv.org/pdf/2505.20321)**

> **作者:** Mathew J. Koretsky; Maya Willey; Owen Bianchi; Chelsea X. Alvarado; Tanay Nayak; Nicole Kuznetsov; Sungwon Kim; Mike A. Nalls; Daniel Khashabi; Faraz Faghri
>
> **备注:** Accepted at the non-archival Gen2 Workshop at ICLR 2026. Under Review
>
> **摘要:** Biomedical researchers increasingly rely on large-scale structured databases for complex analytical tasks. However, current text-to-SQL systems often struggle to map qualitative scientific questions into executable SQL, particularly when implicit domain reasoning is required. We introduce BiomedSQL, the first benchmark explicitly designed to evaluate scientific reasoning in text-to-SQL generation over a real-world biomedical knowledge base. BiomedSQL comprises 68,000 question/SQL query/answer triples generated from templates and grounded in a harmonized BigQuery knowledge base that integrates gene-disease associations, causal inference from omics data, and drug approval records. Each question requires models to infer domain-specific criteria, such as genome-wide significance thresholds, effect directionality, or trial phase filtering, rather than rely on syntactic translation alone. We evaluate a range of open- and closed-source LLMs across prompting strategies and interaction paradigms. Our results reveal a substantial performance gap: Gemini-3-Pro achieves 58.1% execution accuracy, while our custom multi-step agent, BMSQL, reaches 62.6%, both well below the expert baseline of 90.0%. BiomedSQL provides a new foundation for advancing text-to-SQL systems capable of supporting scientific discovery through robust reasoning over structured biomedical knowledge bases. Our dataset is publicly available at this https URL, and our code is open-source at this https URL.
>
---
#### [replaced 011] LLMs as Repositories of Factual Knowledge: Limitations and Solutions
- **分类: cs.CL**

- **简介: 论文研究LLMs作为事实知识库的可靠性，解决其在时间敏感问题上的准确性和一致性问题。通过评估和改进方法，提出ENAF提升模型稳定性。**

- **链接: [https://arxiv.org/pdf/2501.12774](https://arxiv.org/pdf/2501.12774)**

> **作者:** Seyed Mahed Mousavi; Simone Alghisi; Giuseppe Riccardi
>
> **摘要:** LLMs' sources of knowledge are data snapshots containing factual information about entities collected at different timestamps and from different media types (e.g. wikis, social media, etc.). Such unstructured knowledge is subject to change due to updates through time from past to present. Equally important are the inconsistencies and inaccuracies occurring in different information sources. Consequently, the model's knowledge about an entity may be perturbed while training over the sequence of snapshots or at inference time, resulting in inconsistent and inaccurate model performance. In this work, we study the appropriateness of Large Language Models (LLMs) as repositories of factual knowledge. We consider twenty-four state-of-the-art LLMs that are either closed-, partially (weights), or fully (weight and training data) open-source. We evaluate their reliability in responding to time-sensitive factual questions in terms of accuracy and consistency when prompts are perturbed. We further evaluate the effectiveness of state-of-the-art methods to improve LLMs' accuracy and consistency. We then propose ENtity-Aware Fine-tuning (ENAF), a soft neurosymbolic approach aimed at providing structured representation of entities during fine-tuning to reduce inconsistencies and improve response stability under prompt variations.
>
---
#### [replaced 012] APEX-Searcher: Augmenting LLMs' Search Capabilities through Agentic Planning and Execution
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于信息检索任务，旨在解决复杂问题中多跳检索不足的问题。提出APEX-Searcher框架，通过分阶段的代理规划与执行提升LLM的搜索能力。**

- **链接: [https://arxiv.org/pdf/2603.13853](https://arxiv.org/pdf/2603.13853)**

> **作者:** Kun Chen; Qingchao Kong; Zhao Feifei; Wenji Mao
>
> **摘要:** Retrieval-augmented generation (RAG), based on large language models (LLMs), serves as a vital approach to retrieving and leveraging external knowledge in various domain applications. When confronted with complex multi-hop questions, single-round retrieval is often insufficient for accurate reasoning and problem solving. To enhance search capabilities for complex tasks, most existing works integrate multi-round iterative retrieval with reasoning processes via end-to-end training. While these approaches significantly improve problem-solving performance, they are still faced with challenges in task reasoning and model training, especially ambiguous retrieval execution paths and sparse rewards in end-to-end reinforcement learning (RL) process, leading to inaccurate retrieval results and performance degradation. To address these issues, in this paper, we proposes APEX-Searcher, a novel Agentic Planning and Execution framework to augment LLM search capabilities. Specifically, we introduce a two-stage agentic framework that decouples the retrieval process into planning and execution: It first employs RL with decomposition-specific rewards to optimize strategic planning; Built on the sub-task decomposition, it then applies supervised fine-tuning on high-quality multi-hop trajectories to equip the model with robust iterative sub-task execution capabilities. Extensive experiments demonstrate that our proposed framework achieves significant improvements in both multi-hop RAG and task planning performances across multiple benchmarks.
>
---
#### [replaced 013] Test-Time Adaptation via Many-Shot Prompting: Benefits, Limits, and Pitfalls
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理领域，研究测试时适应问题。通过实验分析多示例提示的有效性与局限性，探讨其在不同任务中的表现及优化策略。**

- **链接: [https://arxiv.org/pdf/2603.05829](https://arxiv.org/pdf/2603.05829)**

> **作者:** Shubhangi Upasani; Chen Wu; Jay Rainton; Bo Li; Urmish Thakker; Changran Hu; Qizheng Zhang
>
> **摘要:** Test-time adaptation enables large language models (LLMs) to modify their behavior at inference without updating model parameters. A common approach is many-shot prompting, where large numbers of in-context learning (ICL) examples are injected as an input-space test-time update. Although performance can improve as more demonstrations are added, the reliability and limits of this update mechanism remain poorly understood, particularly for open-source models. We present an empirical study of many-shot prompting across tasks and model backbones, analyzing how performance varies with update magnitude, example ordering, and selection policy. We further study Dynamic and Reinforced ICL as alternative test-time update strategies that control which information is injected and how it constrains model behavior. We find that many-shot prompting is effective for structured tasks where demonstrations provide high information gain, but is highly sensitive to selection strategy and often shows limited benefits for open-ended generation tasks. Overall, we characterize the practical limits of prompt-based test-time adaptation and outline when input-space updates are beneficial versus harmful.
>
---
#### [replaced 014] From Intuition to Calibrated Judgment: A Rubric-Based Expert-Panel Study of Human Detection of LLM-Generated Korean Text
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本归属任务，旨在解决人类区分LLM生成与人工撰写的韩语文本困难问题。通过专家校准框架LREAD，提升判断准确性与一致性。**

- **链接: [https://arxiv.org/pdf/2601.19913](https://arxiv.org/pdf/2601.19913)**

> **作者:** Shinwoo Park; Yo-Sub Han
>
> **摘要:** Distinguishing human-written Korean text from fluent LLM outputs remains difficult even for trained readers, who can over-trust surface well-formedness. We present LREAD, a Korean-specific instantiation of a rubric-based expert-calibration framework for human attribution of LLM-generated text. In a three-phase blind longitudinal study with three linguistically trained annotators, Phase 1 measures intuition-only attribution, Phase 2 introduces criterion-anchored scoring with explicit justifications, and Phase 3 evaluates a limited held-out elementary-persona subset. Majority-vote accuracy improves from 0.60 in Phase 1 to 0.90 in Phase 2, and reaches 10/10 on the limited Phase 3 subset (95% CI [0.692, 1.000]); agreement also increases from Fleiss' $\kappa$ = -0.09 to 0.82. Error analysis suggests that calibration primarily reduces false negatives on AI essays rather than inducing generalized over-detection. We position LREAD as pilot evidence for within-panel calibration in a Korean argumentative-essay setting. These findings suggest that rubric-scaffolded human judgment can complement automated detectors by making attribution reasoning explicit, auditable, and adaptable. The rubric developed in this study, along with the dataset employed for the analysis, is available at this https URL.
>
---
#### [replaced 015] Compressed Convolutional Attention: Efficient Attention in a Compressed Latent Space
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，解决长序列Transformer训练与推理效率低的问题。提出CCA和CCGQA方法，通过压缩潜空间降低计算和内存开销，提升速度与性能。**

- **链接: [https://arxiv.org/pdf/2510.04476](https://arxiv.org/pdf/2510.04476)**

> **作者:** Tomas Figliolia; Nicholas Alonso; Rishi Iyer; Quentin Anthony; Beren Millidge
>
> **摘要:** Multi-headed Attention's (MHA) quadratic compute and linearly growing KV-cache make long-context transformers expensive to train and serve. Prior works such as Grouped Query Attention (GQA) and Multi-Latent Attention (MLA) shrink the cache, speeding decode, but leave compute, which determines prefill and training speed, largely unchanged. We introduce Compressed Convolutional Attention (CCA), a novel attention method which down-projects queries, keys, and values and performs the entire attention operation inside the shared latent space. This simple design dramatically cuts parameters, KV-cache, and FLOPs all at once by the desired compression factor. Because CCA is orthogonal to head-sharing, we combine the two to form Compressed Convolutional Grouped Query Attention (CCGQA), which further tightens the compute-bandwidth Pareto frontier so that users can tune compression toward either FLOP or memory limits without sacrificing quality. Experiments show that CCGQA consistently outperforms both GQA and MLA at equal KV-cache compression on dense and MoE models. Additionally, we show that CCGQA outperforms all other attention methods on MoE models with half the KV-cache of GQA and MLA, achieving an 8x KV-cache compression with no drop in performance compared to standard MHA. CCA and CCGQA also dramatically reduce the FLOP cost of attention which leads to substantially faster training and prefill than existing methods. On H100 GPUs, our fused CCA/CCGQA kernel reduces prefill latency by about 1.7x at a sequence length of 16k relative to MHA, and accelerates backward by about 1.3x.
>
---
#### [replaced 016] ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出ReasoningBank，解决智能体无法有效利用历史经验的问题。通过提炼推理策略，提升任务处理能力，并引入MaTTS加速学习，实现自我进化。**

- **链接: [https://arxiv.org/pdf/2509.25140](https://arxiv.org/pdf/2509.25140)**

> **作者:** Siru Ouyang; Jun Yan; I-Hung Hsu; Yanfei Chen; Ke Jiang; Zifeng Wang; Rujun Han; Long T. Le; Samira Daruki; Xiangru Tang; Vishy Tirumalashetty; George Lee; Mahsan Rofouei; Hangfei Lin; Jiawei Han; Chen-Yu Lee; Tomas Pfister
>
> **备注:** Accepted to ICLR 2026; Code: this https URL
>
> **摘要:** With the growing adoption of large language model agents in persistent real-world roles, they naturally encounter continuous streams of tasks. A key limitation, however, is their failure to learn from the accumulated interaction history, forcing them to discard valuable insights and repeat past errors. We propose ReasoningBank, a novel memory framework that distills generalizable reasoning strategies from an agent's self-judged successful and failed experiences. At test time, an agent retrieves relevant memories from ReasoningBank to inform its interaction and then integrates new learnings back, enabling it to become more capable over time. Building on this powerful experience learner, we further introduce memory-aware test-time scaling (MaTTS), which accelerates and diversifies this learning process by scaling up the agent's interaction experience. By allocating more compute to each task, the agent generates abundant, diverse experiences that provide rich contrastive signals for synthesizing higher-quality memory. The better memory in turn guides more effective scaling, establishing a powerful synergy between memory and test-time scaling. Across web browsing and software engineering benchmarks, ReasoningBank consistently outperforms existing memory mechanisms that store raw trajectories or only successful task routines, improving both effectiveness and efficiency; MaTTS further amplifies these gains. These findings establish memory-driven experience scaling as a new scaling dimension, enabling agents to self-evolve with emergent behaviors naturally arise. Our code can be found at this https URL.
>
---
#### [replaced 017] LogicSkills: A Structured Benchmark for Formal Reasoning in Large Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出LogicSkills基准，用于评估大语言模型的逻辑推理能力，解决其核心逻辑技能掌握不清的问题。工作包括设计三个逻辑任务并验证模型表现。**

- **链接: [https://arxiv.org/pdf/2602.06533](https://arxiv.org/pdf/2602.06533)**

> **作者:** Brian Rabern; Philipp Mondorf; Barbara Plank
>
> **备注:** 12 pages, 5 figures
>
> **摘要:** Large language models perform well on many logical reasoning benchmarks, but it remains unclear which core logical skills they truly master. To address this, we introduce LogicSkills, a benchmark that isolates three fundamental logical skills: (i) $\textit{formal symbolization}\unicode{x2014}{}$translating premises into first-order logic; (ii) $\textit{countermodel construction}\unicode{x2014}$showing that an argument is logically invalid by constructing a finite countermodel; and (iii) $\textit{validity assessment}\unicode{x2014}$determining whether a conclusion follows from a set of premises. Items are drawn from the two-variable fragment of first-order logic without identity and are presented in both English and a Carrollian nonce-word language. All instances are solver-verified with Z3 for correctness and non-triviality. Across conventional instruction-tuned LLMs, performance is high on $\textit{validity assessment}$ but substantially lower on $\textit{formal symbolization}$ and $\textit{countermodel construction}$, highlighting that high task-level accuracy can mask weaknesses in core logical skills. In contrast, recent reasoning-tuned models perform strongly across all three tasks, suggesting a more systematic logical skill profile.
>
---
#### [replaced 018] LLMs Encode Their Failures: Predicting Success from Pre-Generation Activations
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于模型优化任务，旨在通过分析模型内部表示预测其成功概率，以提高推理效率。工作包括训练线性探测器、对比人类与模型难度差异，并实现高效查询路由。**

- **链接: [https://arxiv.org/pdf/2602.09924](https://arxiv.org/pdf/2602.09924)**

> **作者:** William Lugoloobi; Thomas Foster; William Bankes; Chris Russell
>
> **备注:** Accepted at the ICLR 2026 Workshop on Latent and Implicit Thinking
>
> **摘要:** Running LLMs with extended reasoning on every problem is expensive, but determining which inputs actually require additional compute remains challenging. We investigate whether their own likelihood of success is recoverable from their internal representations before generation, and if this signal can guide more efficient inference. We train linear probes on pre-generation activations to predict policy-specific success on math and coding tasks, substantially outperforming surface features such as question length and TF-IDF. Using E2H-AMC, which provides both human and model performance on identical problems, we show that models encode a model-specific notion of difficulty that is distinct from human difficulty, and that this distinction increases with extended reasoning. Leveraging these probes, we demonstrate that routing queries across a pool of models can exceed the best-performing model whilst reducing inference cost by up to 70\% on MATH, showing that internal representations enable practical efficiency gains even when they diverge from human intuitions about difficulty. Our code is available at: this https URL
>
---
#### [replaced 019] SentGraph: Hierarchical Sentence Graph for Multi-hop Retrieval-Augmented Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多跳问答任务，旨在解决传统方法在多跳问答中证据链不完整的问题。提出SentGraph框架，通过构建句子级图结构显式建模逻辑关系，提升推理准确性。**

- **链接: [https://arxiv.org/pdf/2601.03014](https://arxiv.org/pdf/2601.03014)**

> **作者:** Junli Liang; Pengfei Zhou; Wangqiu Zhou; Wenjie Qing; Qi Zhao; Ziwen Wang; Qi Song; Xiangyang Li
>
> **摘要:** Traditional Retrieval-Augmented Generation (RAG) effectively supports single-hop question answering with large language models but faces significant limitations in multi-hop question answering tasks, which require combining evidence from multiple documents. Existing chunk-based retrieval often provides irrelevant and logically incoherent context, leading to incomplete evidence chains and incorrect reasoning during answer generation. To address these challenges, we propose SentGraph, a sentence-level graph-based RAG framework that explicitly models fine-grained logical relationships between sentences for multi-hop question answering. Specifically, we construct a hierarchical sentence graph offline by first adapting Rhetorical Structure Theory to distinguish nucleus and satellite sentences, and then organizing them into topic-level subgraphs with cross-document entity bridges. During online retrieval, SentGraph performs graph-guided evidence selection and path expansion to retrieve fine-grained sentence-level evidence. Extensive experiments on four multi-hop question answering benchmarks demonstrate the effectiveness of SentGraph, validating the importance of explicitly modeling sentence-level logical dependencies for multi-hop reasoning.
>
---
#### [replaced 020] VisTIRA: Closing the Image-Text Modality Gap in Visual Math Reasoning via Structured Tool Integration
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于视觉数学推理任务，旨在解决图像与文本数学问题间的模态差距。通过引入VisTIRA框架和合成数据提升视觉数学推理能力。**

- **链接: [https://arxiv.org/pdf/2601.14440](https://arxiv.org/pdf/2601.14440)**

> **作者:** Saeed Khaki; Ashudeep Singh; Nima Safaei; Kamal Ginotra
>
> **摘要:** Vision-language models (VLMs) lag behind text-only language models on mathematical reasoning when the same problems are presented as images rather than text. We empirically characterize this as a modality gap: the same question in text form yields markedly higher accuracy than its visually typeset counterpart, due to compounded failures in reading dense formulas, layout, and mixed symbolic-diagrammatic context. First, we introduce VisTIRA (Vision and Tool-Integrated Reasoning Agent), a tool-integrated reasoning framework that enables structured problem solving by iteratively decomposing a given math problem (as an image) into natural language rationales and executable Python steps to determine the final answer. Second, we build a framework to measure and improve visual math reasoning: a LaTeX-based pipeline that converts chain-of-thought math corpora (e.g., NuminaMath) into challenging image counterparts, and a large set of synthetic tool-use trajectories derived from a real-world, homework-style image dataset (called SnapAsk) for fine-tuning VLMs. Our experiments show that tool-integrated supervision improves image-based reasoning, and OCR grounding can further narrow the gap for smaller models, although its benefit diminishes at scale. These findings highlight that modality gap severity inversely correlates with model size, and that structured reasoning and OCR-based grounding are complementary strategies for advancing visual mathematical reasoning.
>
---
#### [replaced 021] To See is Not to Master: Teaching LLMs to Use Private Libraries for Code Generation
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文属于代码生成任务，解决LLMs在使用私有库API时效果不佳的问题。通过PriCoder方法，利用合成数据提升模型调用私有库API的能力。**

- **链接: [https://arxiv.org/pdf/2603.15159](https://arxiv.org/pdf/2603.15159)**

> **作者:** Yitong Zhang; Chengze Li; Ruize Chen; Guowei Yang; Xiaoran Jia; Yijie Ren; Jia Li
>
> **备注:** 12 pages
>
> **摘要:** Large Language Models (LLMs) have shown strong potential for code generation, yet they remain limited in private-library-oriented code generation, where the goal is to generate code using APIs from private libraries. Existing approaches mainly rely on retrieving private-library API documentation and injecting relevant knowledge into the context at inference time. However, our study shows that this is insufficient: even given accurate required knowledge, LLMs still struggle to invoke private-library APIs effectively. To address this limitation, we propose PriCoder, an approach that teaches LLMs to invoke private-library APIs through automatically synthesized data. Specifically, PriCoder models private-library data synthesis as the construction of a graph, and alternates between two graph operators: (1) Progressive Graph Evolution, which improves data diversity by progressively synthesizing more diverse training samples from basic ones, and (2) Multidimensional Graph Pruning, which improves data quality through a rigorous filtering pipeline. To support rigorous evaluation, we construct two new benchmarks based on recently released libraries that are unfamiliar to the tested models. Experiments on three mainstream LLMs show that PriCoder substantially improves private-library-oriented code generation, yielding gains of over 20% in pass@1 in many settings, while causing negligible impact on general code generation capability. Our code and benchmarks are publicly available at this https URL.
>
---
#### [replaced 022] Generalizable End-to-End Tool-Use RL with Synthetic CodeGym
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决LLM代理在工具使用中的泛化能力不足问题。通过构建CodeGym框架，生成多样化的工具使用环境，提升模型的泛化性能。**

- **链接: [https://arxiv.org/pdf/2509.17325](https://arxiv.org/pdf/2509.17325)**

> **作者:** Weihua Du; Hailei Gong; Zhan Ling; Kang Liu; Lingfeng Shen; Xuesong Yao; Yufei Xu; Dingyuan Shi; Yiming Yang; Jiecao Chen
>
> **备注:** 24 pages. Accepted to ICLR 2026. Project repository: this https URL
>
> **摘要:** Tool-augmented large language models (LLMs), hereafter LLM agents, leverage external tools to solve diverse tasks and interface with the real world. However, current training practices largely rely on supervised fine-tuning (SFT) over static trajectories or reinforcement learning (RL) on narrow tasks, which generalize poorly beyond development settings and lead to brittleness with new tools and unseen workflows. Because code execution reflects many structural patterns of real-world workflows, we use coding problems as a structured substrate to build tool-use agent training environments with diverse task configurations. To this end, we introduce CodeGym, a scalable framework that synthesizes diverse, verifiable, and controllable multi-turn tool-use environments for agent RL, enabling LLM agents to explore and master various workflows actively. CodeGym converts static coding problems into interactive environments by extracting atomic functions or logic into callable tools, yielding verifiable tasks that span various tool-execution workflows. Models of varying sizes and chain-of-thought configurations trained in CodeGym exhibit consistent out-of-distribution generalizability; for example, Qwen2.5-32B-Instruct achieves an absolute accuracy gain of 8.7 points on the OOD benchmark $\tau$-Bench. These results highlight CodeGym as a step toward scalable general-purpose RL environments for training tool-use behaviors that align with real-world agent workflows.
>
---
#### [replaced 023] Learning to Diagnose Privately: DP-Powered LLMs for Radiology Report Classification
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于医疗文本分类任务，旨在解决LLMs在医学影像报告分类中隐私泄露的问题。通过引入差分隐私的LoRA微调方法，在保护隐私的同时保持较高分类性能。**

- **链接: [https://arxiv.org/pdf/2506.04450](https://arxiv.org/pdf/2506.04450)**

> **作者:** Payel Bhattacharjee; Fengwei Tian; Geoffrey D. Rubin; Joseph Y. Lo; Nirav Merchant; Heidi Hanson; John Gounley; Ravi Tandon
>
> **摘要:** Large Language Models (LLMs) are increasingly adopted across domains such as education, healthcare, and finance. In healthcare, LLMs support tasks including disease diagnosis, abnormality classification, and clinical decision-making. Among these, multi-abnormality classification of radiology reports is critical for clinical workflow automation and biomedical research. Leveraging strong natural language processing capabilities, LLMs enable efficient processing of unstructured medical text and reduce the administrative burden of manual report analysis. To improve performance, LLMs are often fine-tuned on private, institution-specific datasets such as radiology reports. However, this raises significant privacy concerns: LLMs may memorize training data and become vulnerable to data extraction attacks, while sharing fine-tuned models risks exposing sensitive patient information. Despite growing interest in LLMs for medical text classification, privacy-preserving fine-tuning for multi-abnormality classification remains underexplored. To address this gap, we propose a differentially private (DP) fine-tuning framework for multi-abnormality classification from free-text radiology reports. Our approach integrates differential privacy with Low-Rank Adaptation (LoRA) to efficiently fine-tune LLMs on sensitive clinical data while mitigating leakage risks. We further employ labels generated by a larger LLM to train smaller models, enabling efficient inference under strong privacy guarantees. Experiments on MIMIC-CXR and CT-RATE demonstrate the effectiveness of our DP-LoRA framework across varying privacy regimes. On MIMIC-CXR, our method achieves weighted F1-scores up to 0.89 under moderate privacy budgets, approaching non-private LoRA (0.90) and full fine-tuning (0.96), confirming that strong privacy can be achieved with only modest performance trade-offs.
>
---
#### [replaced 024] Transformer-Encoder Trees for Efficient Multilingual Machine Translation and Speech Translation
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对多语言翻译任务，解决计算冗余和低资源语言质量差的问题，提出TET架构，通过共享中间表示提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2509.17930](https://arxiv.org/pdf/2509.17930)**

> **作者:** Yiwen Guan; Jacob Whitehill
>
> **摘要:** Multilingual translation suffers from computational redundancy, especially when translating into multiple languages simultaneously. In addition, translation quality can suffer for low-resource languages. To address this, we introduce Transformer Encoder Tree (TET), a hierarchical, non-autoregressive encoder-only architecture trained with Connectionist Temporal Classification (CTC) for multilingual translation. TET shares intermediate representations among linguistically similar target languages, improving accuracy on low-resource languages while reducing computational redundancy and enabling the generation of all target languages in a single forward pass. TET eliminates the sequential bottleneck of autoregressive models and supports fully parallel decoding of all tokens across all target languages. Compared to a naive one-to-many multilingual design, TET reduces the total parameter count by 66% and lowers inference computation by 60%. In speech translation, combining TET with a non-autoregressive speech recognition backbone (Wav2Vec2) shows competitive translation quality compared to autoregressive systems while speeding up inference by approximately 7-14 times.
>
---
#### [replaced 025] SWE-CI: Evaluating Agent Capabilities in Maintaining Codebases via Continuous Integration
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文提出SWE-CI基准，用于评估代码生成代理在长期代码维护中的能力。针对传统静态修复方法无法模拟真实开发过程的问题，SWE-CI通过持续集成循环进行动态评估，提升代码可维护性。**

- **链接: [https://arxiv.org/pdf/2603.03823](https://arxiv.org/pdf/2603.03823)**

> **作者:** Jialong Chen; Xander Xu; Hu Wei; Chuan Chen; Bing Zhao
>
> **摘要:** Large language model (LLM)-powered agents have demonstrated strong capabilities in automating software engineering tasks such as static bug fixing, as evidenced by benchmarks like SWE-bench. However, in the real world, the development of mature software is typically predicated on complex requirement changes and long-term feature iterations -- a process that static, one-shot repair paradigms fail to capture. To bridge this gap, we propose \textbf{SWE-CI}, the first repository-level benchmark built upon the Continuous Integration loop, aiming to shift the evaluation paradigm for code generation from static, short-term \textit{functional correctness} toward dynamic, long-term \textit{maintainability}. The benchmark comprises 100 tasks, each corresponding on average to an evolution history spanning 233 days and 71 consecutive commits in a real-world code repository. SWE-CI requires agents to systematically resolve these tasks through dozens of rounds of analysis and coding iterations. SWE-CI provides valuable insights into how well agents can sustain code quality throughout long-term evolution.
>
---
#### [replaced 026] Model Medicine: A Clinical Framework for Understanding, Diagnosing, and Treating AI Models
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出"模型医学"，将AI模型类比生物体，解决模型理解与诊断问题，构建了分类体系、诊断框架及工具。**

- **链接: [https://arxiv.org/pdf/2603.04722](https://arxiv.org/pdf/2603.04722)**

> **作者:** Jihoon Jeong
>
> **备注:** 56 pages, 7 figures. Project page: this https URL
>
> **摘要:** Model Medicine is the science of understanding, diagnosing, treating, and preventing disorders in AI models, grounded in the principle that AI models -- like biological organisms -- have internal structures, dynamic processes, heritable traits, observable symptoms, classifiable conditions, and treatable states. This paper introduces Model Medicine as a research program, bridging the gap between current AI interpretability research (anatomical observation) and the systematic clinical practice that complex AI systems increasingly require. We present five contributions: (1) a discipline taxonomy organizing 15 subdisciplines across four divisions -- Basic Model Sciences, Clinical Model Sciences, Model Public Health, and Model Architectural Medicine; (2) the Four Shell Model (v3.3), a behavioral genetics framework empirically grounded in 720 agents and 24,923 decisions from the Agora-12 program, explaining how model behavior emerges from Core--Shell interaction; (3) Neural MRI (Model Resonance Imaging), a working open-source diagnostic tool mapping five medical neuroimaging modalities to AI interpretability techniques, validated through four clinical cases demonstrating imaging, comparison, localization, and predictive capability; (4) a five-layer diagnostic framework for comprehensive model assessment; and (5) clinical model sciences including the Model Temperament Index for behavioral profiling, Model Semiology for symptom description, and M-CARE for standardized case reporting. We additionally propose the Layered Core Hypothesis -- a biologically-inspired three-layer parameter architecture -- and a therapeutic framework connecting diagnosis to treatment.
>
---
#### [replaced 027] GroupGPT: A Token-efficient and Privacy-preserving Agentic Framework for Multi-User Chat Assistant
- **分类: cs.CL**

- **简介: 该论文提出GroupGPT，解决多用户聊天中的高效、隐私保护干预问题。通过小大模型协作架构，提升响应准确性和效率，减少token消耗，并引入MUIR数据集进行评估。**

- **链接: [https://arxiv.org/pdf/2603.01059](https://arxiv.org/pdf/2603.01059)**

> **作者:** Zhuokang Shen; Yifan Wang; Hanyu Chen; Wenxuan Huang; Yunhang Shen; Shaohui Lin
>
> **摘要:** Recent advances in large language models (LLMs) have enabled increasingly capable chatbots. However, most existing systems focus on single-user settings and do not generalize well to multi-user group chats, where agents require more proactive and accurate intervention under complex, evolving contexts. Existing approaches typically rely on LLMs for both reasoning and generation, leading to high token consumption, limited scalability, and potential privacy risks. To address these challenges, we propose GroupGPT, a token-efficient and privacy-preserving agentic framework for multi-user chat assistant. GroupGPT adopts a small-large model collaborative architecture to decouple intervention timing from response generation, enabling efficient and accurate decision-making. The framework also supports multimodal inputs, including memes, images, videos, and voice messages. We further introduce MUIR, a benchmark dataset for multi-user chat assistant intervention reasoning. MUIR contains 2,500 annotated group chat segments with intervention labels and rationales, supporting evaluation of timing accuracy and response quality. We evaluate a range of models on MUIR, from large language models to smaller counterparts. Extensive experiments demonstrate that GroupGPT produces accurate and well-timed responses, achieving an average score of 4.72/5.0 in LLM-based evaluation, and is well received by users across diverse group chat scenarios. Moreover, GroupGPT reduces token usage by up to 3 times compared to baseline methods, while providing privacy sanitization of user messages before cloud transmission. Code is available at: this https URL .
>
---
#### [replaced 028] Large Language Models Approach Expert Pedagogical Quality in Math Tutoring but Differ in Instructional and Linguistic Profiles
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于教育技术任务，研究LLMs在数学辅导中的教学质量。通过分析对话数据，比较LLMs与人类导师的教学策略和语言特征，探讨其教学效果差异。**

- **链接: [https://arxiv.org/pdf/2512.20780](https://arxiv.org/pdf/2512.20780)**

> **作者:** Ramatu Oiza Abdulsalam; Segun Aroyehun
>
> **摘要:** Recent work has explored the use of large language models (LLMs) to generate tutoring responses in mathematics, yet it remains unclear how closely their instructional behavior aligns with expert human practice. We analyze a dataset of math remediation dialogues in which expert tutors, novice tutors, and seven LLMs of varying sizes, comprising both open-weight and commercial models, respond to the same student errors. We examine instructional strategies and linguistic characteristics of tutoring responses, including uptake (restating and revoicing), pressing for accuracy and reasoning, lexical diversity, readability, politeness, and agency. We find that expert tutors produce higher-quality responses than novices, and that larger LLMs generally receive higher pedagogical quality ratings than smaller models, approaching expert performance on average. However, LLMs exhibit systematic differences in their instructional profiles: they underuse discursive strategies characteristic of expert tutors while generating longer, more lexically diverse, and more polite responses. Regression analyses show that pressing for accuracy and reasoning, restating and revoicing, and lexical diversity, are positively associated with perceived pedagogical quality, whereas higher levels of agentic and polite language are negatively associated. These findings highlight the importance of analyzing instructional strategies and linguistic characteristics when evaluating tutoring responses across human tutors and intelligent tutoring systems.
>
---
#### [replaced 029] Are LLMs Good Text Diacritizers? An Arabic and Yoruba Case Study
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本加注任务，研究LLMs在阿拉伯语和约鲁巴语中的加注效果。通过构建数据集并对比模型，验证LLMs的性能及优化方法。**

- **链接: [https://arxiv.org/pdf/2506.11602](https://arxiv.org/pdf/2506.11602)**

> **作者:** Hawau Olamide Toyin; Samar Mohamed Magdy; Hanan Aldarmaki
>
> **备注:** accepted at LREC 2026
>
> **摘要:** We investigate the effectiveness of large language models (LLMs) for text diacritization in two typologically distinct languages: Arabic and Yoruba. To enable a rigorous evaluation, we introduce a novel multilingual dataset MultiDiac, with diverse samples that capture a range of diacritic ambiguities. We evaluate 12 LLMs varying in size, accessibility, and language coverage, and benchmark them against $4$ specialized diacritization models. Additionally, we fine-tune four small open-source models using LoRA for Yoruba. Our results show that many off-the-shelf LLMs outperform specialized diacritization models, but smaller models suffer from hallucinations. We find that fine-tuning on a small dataset can help improve diacritization performance and reduce hallucinations for Yoruba.
>
---
#### [replaced 030] Impatient Users Confuse AI Agents: High-fidelity Simulations of Human Traits for Testing Agents
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于AI代理鲁棒性测试任务，旨在解决当前AI代理在用户行为变化下的脆弱性问题。通过引入TraitBasis方法，系统性地模拟用户特质进行压力测试。**

- **链接: [https://arxiv.org/pdf/2510.04491](https://arxiv.org/pdf/2510.04491)**

> **作者:** Muyu He; Anand Kumar; Tsach Mackey; Meghana Rajeev; James Zou; Nazneen Rajani
>
> **备注:** 27 pages
>
> **摘要:** Despite rapid progress in building conversational AI agents, robustness is still largely untested. Small shifts in user behavior, such as being more impatient, incoherent, or skeptical, can cause sharp drops in agent performance, revealing how brittle current AI agents are. Today's benchmarks fail to capture this fragility: agents may perform well under standard evaluations but degrade spectacularly in more realistic and varied settings. We address this robustness testing gap by introducing TraitBasis, a lightweight, model-agnostic method for systematically stress testing AI agents. TraitBasis learns directions in activation space corresponding to steerable user traits (e.g., impatience or incoherence), which can be controlled, scaled, composed, and applied at inference time without any fine-tuning or extra data. Using TraitBasis, we extend $\tau$-Bench to $\tau$-Trait, where user behaviors are altered via controlled trait vectors. We observe on average a 2%-30% performance degradation on $\tau$-Trait across frontier models, highlighting the lack of robustness of current AI agents to variations in user behavior. Together, these results highlight both the critical role of robustness testing and the promise of TraitBasis as a simple, data-efficient, and compositional tool. By powering simulation-driven stress tests and training loops, TraitBasis opens the door to building AI agents that remain reliable in the unpredictable dynamics of real-world human interactions. We have open-sourced $\tau$-Trai across four domains: airline, retail, telecom, and telehealth, so the community can systematically QA their agents under realistic, behaviorally diverse intents and trait scenarios: this https URL.
>
---
#### [replaced 031] On Theoretically-Driven LLM Agents for Multi-Dimensional Discourse Analysis
- **分类: cs.CL**

- **简介: 该论文属于计算论证任务，旨在解决话语中改写策略识别问题。通过构建多智能体框架，结合理论知识提升模型性能，显著提高了对特定改写功能的检测效果。**

- **链接: [https://arxiv.org/pdf/2602.13713](https://arxiv.org/pdf/2602.13713)**

> **作者:** Maciej Uberna; Michał Wawer; Jarosław A. Chudziak; Marcin Koszowy
>
> **备注:** 8 pages, 4 figures, 3 tables. This is the accepted version of the paper presented at the 18th International Conference on Agents and Artificial Intelligence (ICAART 2026), Marbella, Spain
>
> **摘要:** Identifying the strategic uses of reformulation in discourse remains a key challenge for computational argumentation. While LLMs can detect surface-level similarity, they often fail to capture the pragmatic functions of rephrasing, such as its role within rhetorical discourse. This paper presents a comparative multi-agent framework designed to quantify the benefits of incorporating explicit theoretical knowledge for this task. We utilise an dataset of annotated political debates to establish a new standard encompassing four distinct rephrase functions: Deintensification, Intensification, Specification, Generalisation, and Other, which covers all remaining types (D-I-S-G-O). We then evaluate two parallel LLM-based agent systems: one enhanced by argumentation theory via Retrieval-Augmented Generation (RAG), and an identical zero-shot baseline. The results reveal a clear performance gap: the RAG-enhanced agents substantially outperform the baseline across the board, with particularly strong advantages in detecting Intensification and Generalisation context, yielding an overall Macro F1-score improvement of nearly 30\%. Our findings provide evidence that theoretical grounding is not only beneficial but essential for advancing beyond mere paraphrase detection towards function-aware analysis of argumentative discourse. This comparative multi-agent architecture represents a step towards scalable, theoretically informed computational tools capable of identifying rhetorical strategies in contemporary discourse.
>
---
#### [replaced 032] HindSight: Evaluating LLM-Generated Research Ideas via Future Impact
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于AI生成研究想法评估任务，旨在解决现有评价方法主观且与实际影响脱节的问题。提出HindSight框架，通过未来文献匹配评估想法质量。**

- **链接: [https://arxiv.org/pdf/2603.15164](https://arxiv.org/pdf/2603.15164)**

> **作者:** Bo Jiang
>
> **摘要:** Evaluating AI-generated research ideas typically relies on LLM judges or human panels -- both subjective and disconnected from actual research impact. We introduce HindSight, a time-split evaluation framework that measures idea quality by matching generated ideas against real future publications and scoring them by citation impact and venue acceptance. Using a temporal cutoff~$T$, we restrict an idea generation system to pre-$T$ literature, then evaluate its outputs against papers published in the subsequent 30 months. Experiments across 10 AI/ML research topics reveal a striking disconnect: LLM-as-Judge finds no significant difference between retrieval-augmented and vanilla idea generation ($p{=}0.584$), while HindSight shows the retrieval-augmented system produces 2.5$\times$ higher-scoring ideas ($p{<}0.001$). Moreover, HindSight scores are \emph{negatively} correlated with LLM-judged novelty ($\rho{=}{-}0.29$, $p{<}0.01$), suggesting that LLMs systematically overvalue novel-sounding ideas that never materialize in real research.
>
---
#### [replaced 033] From Euler to AI: Unifying Formulas for Mathematical Constants
- **分类: math.HO; cs.AI; cs.CL; math.NT**

- **简介: 该论文属于数学公式统一任务，旨在解决数学常数公式间缺乏统一理论的问题。通过AI方法整合和验证大量公式，揭示其内在联系。**

- **链接: [https://arxiv.org/pdf/2502.17533](https://arxiv.org/pdf/2502.17533)**

> **作者:** Tomer Raz; Michael Shalyt; Elyasheev Leibtag; Rotem Kalisch; Shachar Weinbaum; Yaron Hadad; Ido Kaminer
>
> **备注:** Final version for NeurIPS2025. Published at this https URL
>
> **摘要:** The constant $\pi$ has fascinated scholars throughout the centuries, inspiring numerous formulas for its evaluation, such as infinite sums and continued fractions. Despite their individual significance, many of the underlying connections among formulas remain unknown, missing unifying theories that could unveil deeper understanding. The absence of a unifying theory reflects a broader challenge across math and science: knowledge is typically accumulated through isolated discoveries, while deeper connections often remain hidden. In this work, we present an automated framework for the unification of mathematical formulas. Our system combines Large Language Models (LLMs) for systematic formula harvesting, an LLM-code feedback loop for validation, and a novel symbolic algorithm for clustering and eventual unification. We demonstrate this methodology on the hallmark case of $\pi$, an ideal testing ground for symbolic unification. Applying this approach to 455,050 arXiv papers, we validate 385 distinct formulas for $\pi$ and prove relations between 360 (94%) of them, of which 166 (43%) can be derived from a single mathematical object - linking canonical formulas by Euler, Gauss, Brouncker, and newer ones from algorithmic discoveries by the Ramanujan Machine. Our method generalizes to other constants, including $e$, $\zeta(3)$, and Catalan's constant, demonstrating the potential of AI-assisted mathematics to uncover hidden structures and unify knowledge across domains.
>
---
#### [replaced 034] Readers Prefer Outputs of AI Trained on Copyrighted Books over Expert Human Writers
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于自然语言生成任务，探讨AI与人类写作质量比较。研究解决AI是否能高质量模仿作者风格的问题，通过对比实验发现微调后的AI更受读者欢迎。**

- **链接: [https://arxiv.org/pdf/2510.13939](https://arxiv.org/pdf/2510.13939)**

> **作者:** Tuhin Chakrabarty; Jane C. Ginsburg; Paramveer Dhillon
>
> **备注:** Preprint Under Review
>
> **摘要:** The use of copyrighted books for training AI has sparked lawsuits from authors concerned about AI generating derivative content. Yet whether these models can produce high-quality literary text emulating authors' voices remains unclear. We conducted a preregistered study comparing MFA-trained writers with three frontier models (ChatGPT, Claude, Gemini) writing up to 450-word excerpts emulating 50 award-winning authors' styles. In blind pairwise evaluations by 28 MFA-trained readers and 516 college-educated general readers, AI text from in-context prompting was strongly disfavored by MFA readers for stylistic fidelity (OR=0.16) and quality (OR=0.13), while general readers showed no fidelity preference (OR=1.06) but favored AI for quality (OR=1.82). Fine-tuning ChatGPT on authors' complete works reversed these results: MFA readers favored AI for fidelity (OR=8.16) and quality (OR=1.87), with general readers showing even stronger preference (fidelity OR=16.65; quality OR=5.42). Both groups preferred fine-tuned AI, but the writer-type X reader-type interaction remained significant (p=0.021 for fidelity; p<10^-4 for quality), indicating general readers favored AI by a wider margin. Effects are robust under cluster-robust inference and generalize across authors in heterogeneity analyses. Fine-tuned outputs were rarely flagged as AI-generated (3% vs. 97% for prompting) by leading detectors. Mediation analysis shows fine-tuning eliminates detectable AI quirks that penalize in-context outputs, altering the nexus between detectability and preference. While not accounting for effort to transform AI output into publishable prose, the median fine-tuning cost of $81 per author represents a 99.7% reduction versus typical writer compensation. Author-specific fine-tuning enables non-verbatim AI writing preferred over expert human writing, providing evidence relevant to copyright's fourth fair-use factor.
>
---
#### [replaced 035] CRIMSON: A Clinically-Grounded LLM-Based Metric for Generative Radiology Report Evaluation
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出CRIMSON，用于评估生成的胸部X光报告，解决报告质量评估问题，通过临床相关性、诊断正确性和患者安全进行综合评价。**

- **链接: [https://arxiv.org/pdf/2603.06183](https://arxiv.org/pdf/2603.06183)**

> **作者:** Mohammed Baharoon; Thibault Heintz; Siavash Raissi; Mahmoud Alabbad; Mona Alhammad; Hassan AlOmaish; Sung Eun Kim; Oishi Banerjee; Pranav Rajpurkar
>
> **摘要:** We introduce CRIMSON, a clinically grounded evaluation framework for chest X-ray report generation that assesses reports based on diagnostic correctness, contextual relevance, and patient safety. Unlike prior metrics, CRIMSON incorporates full clinical context, including patient age, indication, and guideline-based decision rules, and prevents normal or clinically insignificant findings from exerting disproportionate influence on the overall score. The framework categorizes errors into a comprehensive taxonomy covering false findings, missing findings, and eight attribute-level errors (e.g., location, severity, measurement, and diagnostic overinterpretation). Each finding is assigned a clinical significance level (urgent, actionable non-urgent, non-actionable, or expected/benign), based on a guideline developed in collaboration with attending cardiothoracic radiologists, enabling severity-aware weighting that prioritizes clinically consequential mistakes over benign discrepancies. CRIMSON is validated through strong alignment with clinically significant error counts annotated by six board-certified radiologists in ReXVal (Kendalls tau = 0.61-0.71; Pearsons r = 0.71-0.84), and through two additional benchmarks that we introduce. In RadJudge, a targeted suite of clinically challenging pass-fail scenarios, CRIMSON shows consistent agreement with expert judgment. In RadPref, a larger radiologist preference benchmark of over 100 pairwise cases with structured error categorization, severity modeling, and 1-5 overall quality ratings from three cardiothoracic radiologists, CRIMSON achieves the strongest alignment with radiologist preferences. We release the metric, the evaluation benchmarks, RadJudge and RadPref, and a fine-tuned MedGemma model to enable reproducible evaluation of report generation, all available at this https URL.
>
---
#### [replaced 036] Evontree: Ontology Rule-Guided Self-Evolution of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识增强任务，旨在解决LLM在专业领域中的幻觉问题。通过引入本体规则，实现模型的自我进化，提升准确性。**

- **链接: [https://arxiv.org/pdf/2510.26683](https://arxiv.org/pdf/2510.26683)**

> **作者:** Mingchen Tu; Zhiqiang Liu; Juan Li; Liangyurui Liu; Junjie Wang; Lei Liang; Wen Zhang
>
> **摘要:** Although Large Language Models (LLMs) perform exceptionally well in general domains, the problem of hallucinations poses significant risks in specialized fields such as healthcare and law, where high interpretability is essential. Existing fine-tuning methods depend heavily on large-scale professional datasets, which are often hard to obtain due to the privacy regulations. Moreover, existing self-evolution methods are primarily designed for general domains, which may struggle to adapt to knowledge-intensive domains due to the lack of knowledge constraints. In this paper, we propose an ontology rule guided method Evontree to enable self-evolution of LLMs in low-resource specialized domains. Specifically, Evontree first extracts domain ontology knowledge from raw models, then detects knowledge inconsistencies using two core ontology rules, and finally reinforces gap knowledge into model via self-distilled fine-tuning. Extensive evaluations on medical QA benchmarks using Llama3-8B-Instruct and Med42-V2 demonstrate the effectiveness of Evontree, which outperforms both the base models and strong baselines, achieving up to a 3.7\% improvement in accuracy. Detailed ablation studies further validate the robustness of our approach.
>
---
#### [replaced 037] Can LLMs Detect Their Confabulations? Estimating Reliability in Uncertainty-Aware Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决LLMs生成不可靠内容的问题。通过分析不确定性，提升模型对自身错误的识别能力。**

- **链接: [https://arxiv.org/pdf/2508.08139](https://arxiv.org/pdf/2508.08139)**

> **作者:** Tianyi Zhou; Johanne Medina; Sanjay Chawla
>
> **备注:** Published at AAAI'26
>
> **摘要:** Large Language Models (LLMs) are prone to generating fluent but incorrect content, known as confabulation, which poses increasing risks in multi-turn or agentic applications where outputs may be reused as context. In this work, we investigate how in-context information influences model behavior and whether LLMs can identify their unreliable responses. We propose a reliability estimation that leverages token-level uncertainty to guide the aggregation of internal model representations. Specifically, we compute aleatoric and epistemic uncertainty from output logits to identify salient tokens and aggregate their hidden states into compact representations for response-level reliability prediction. Through controlled experiments on open QA benchmarks, we find that correct in-context information improves both answer accuracy and model confidence, while misleading context often induces confidently incorrect responses, revealing a misalignment between uncertainty and correctness. Our probing-based method captures these shifts in model behavior and improves the detection of unreliable outputs across multiple open-source LLMs. These results underscore the limitations of direct uncertainty signals and highlight the potential of uncertainty-guided probing for reliability-aware generation.
>
---
#### [replaced 038] TempCore: Are Video QA Benchmarks Temporally Grounded? A Frame Selection Sensitivity Analysis and Benchmark
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于视频问答任务，旨在解决现有基准是否真正依赖时间帧选择的问题。通过分析发现多数样本对帧选择不敏感，构建了TempCore基准以聚焦时间敏感样本。**

- **链接: [https://arxiv.org/pdf/2509.01167](https://arxiv.org/pdf/2509.01167)**

> **作者:** Hyunjong Ok; Jaeho Lee
>
> **备注:** preprint
>
> **摘要:** Vision-language models (VLMs) can ingest only a limited number of video frames, making frame selection a practical necessity. But do current Video QA benchmarks genuinely require temporal frame selection, or can most questions be answered regardless of which frames are shown? We introduce Frame Selection Sensitivity (FSS), a per-sample diagnostic that measures how much VLM accuracy changes when the most relevant frames are replaced with the least relevant ones. Across six benchmarks and eight VLMs, we find that a large majority of samples are frame-agnostic: only a minority are genuinely sensitive to frame choice. Combining FSS with a Language Independence Score (LIS) reveals that merely 8--33% of samples are Temporally Sensitive. We construct TempCore, compact evaluation subsets that isolate these temporal samples from existing benchmarks, and will release code and per-sample annotations upon publication.
>
---
#### [replaced 039] Political Alignment in Large Language Models: A Multidimensional Audit of Psychometric Identity and Behavioral Bias
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文属于AI模型政治立场审计任务，旨在评估大语言模型的政治对齐情况。通过多项心理测量工具和新闻偏见任务，分析模型的意识形态分布及行为偏差。**

- **链接: [https://arxiv.org/pdf/2601.06194](https://arxiv.org/pdf/2601.06194)**

> **作者:** Adib Sakhawat; Tahsin Islam; Takia Farhin; Syed Rifat Raiyan; Hasan Mahmud; Md Kamrul Hasan
>
> **备注:** Under review, 25 pages, 6 figures, 23 tables
>
> **摘要:** As large language models (LLMs) are increasingly deployed, understanding how they express political positioning is important for evaluating alignment and downstream effects. We audit 26 contemporary LLMs using three political psychometric inventories (Political Compass, SapplyValues, 8Values) and a news bias labeling task. To test robustness, inventories are administered across multiple semantic prompt variants and analyzed with a two-way ANOVA separating model and prompt effects. Most models cluster in a similar ideological region, with 96.3% located in the Libertarian-Left quadrant of the Political Compass, and model identity explaining most variance across prompt variants ($\eta^2 > 0.90$). Cross-instrument comparisons suggest that the Political Compass social axis aligns more strongly with cultural progressivism than authority-related measures ($r=-0.64$). We observe differences between open-weight and closed-source models and asymmetric performance in detecting extreme political bias in downstream classification. Regression analysis finds that psychometric ideological positioning does not significantly predict classification errors, providing no evidence of a statistically significant relationship between conversational ideological identity and task-level behavior. These findings suggest that single-axis evaluations are insufficient and that multidimensional auditing frameworks are important to characterize alignment behavior in deployed LLMs. Our code and data are publicly available at this https URL.
>
---
#### [replaced 040] AdaSwitch: Balancing Exploration and Guidance in Knowledge Distillation via Adaptive Switching
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识蒸馏任务，旨在解决小模型性能提升难题。通过自适应切换机制，平衡探索与指导，提升生成一致性与监督质量。**

- **链接: [https://arxiv.org/pdf/2510.07842](https://arxiv.org/pdf/2510.07842)**

> **作者:** Jingyu Peng; Maolin Wang; Hengyi Cai; Yuchen Li; Kai Zhang; Shuaiqiang Wang; Dawei Yin; Xiangyu Zhao
>
> **摘要:** Small language models (SLMs) are crucial for applications with strict latency and computational constraints, yet achieving high performance remains challenging. Knowledge distillation (KD) can transfer capabilities from large teacher models, but existing methods face a dilemma: off-policy distillation provides high-quality supervision but suffers from exposure bias (training inference mismatch), while on-policy approaches ensure consistency but are limited by the low quality of student-generated outputs. To address these issues, we propose AdaSwitch, a novel approach that dynamically combines on-policy and off-policy generation via an adaptive switching mechanism. AdaSwitch allows the student to explore its predictions within its capability and selectively integrates teacher guidance only when divergence exceeds a context-aware threshold. This paradigm preserves generation consistency while ensuring high-quality supervision. Experiments on three datasets demonstrate that AdaSwitch consistently improves accuracy and reasoning capability with moderate overhead.
>
---
#### [replaced 041] A Survey of Frontiers in LLM Reasoning: Inference Scaling, Learning to Reason, and Agentic Systems
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理领域，旨在探讨大语言模型的推理能力。论文分类分析了推理方法，解决如何提升模型推理质量的问题，涵盖推理训练、代理系统及优化策略。**

- **链接: [https://arxiv.org/pdf/2504.09037](https://arxiv.org/pdf/2504.09037)**

> **作者:** Zixuan Ke; Fangkai Jiao; Yifei Ming; Xuan-Phi Nguyen; Austin Xu; Do Xuan Long; Minzhi Li; Chengwei Qin; Peifeng Wang; Silvio Savarese; Caiming Xiong; Shafiq Joty
>
> **备注:** 72 pages, 6 figures. Accepted to TMLR, with Survey Certification award
>
> **摘要:** Reasoning is a fundamental cognitive process that enables logical inference, problem-solving, and decision-making. With the rapid advancement of large language models (LLMs), reasoning has emerged as a key capability that distinguishes advanced AI systems from conventional models that empower chatbots. In this survey, we categorize existing methods along two orthogonal dimensions: (1) Regimes, which define the stage at which reasoning is achieved (either at inference time or through dedicated training); and (2) Architectures, which determine the components involved in the reasoning process, distinguishing between standalone LLMs and agentic compound systems that incorporate external tools, and multi-agent collaborations. Within each dimension, we analyze two key perspectives: (1) Input level, which focuses on techniques that construct high-quality prompts that the LLM condition on; and (2) Output level, which methods that refine multiple sampled candidates to enhance reasoning quality. This categorization provides a systematic understanding of the evolving landscape of LLM reasoning, highlighting emerging trends such as the shift from inference-scaling to learning-to-reason (e.g., DeepSeek-R1), and the transition to agentic workflows (e.g., OpenAI Deep Research, Manus Agent). Additionally, we cover a broad spectrum of learning algorithms, from supervised fine-tuning to reinforcement learning such as PPO and GRPO, and the training of reasoners and verifiers. We also examine key designs of agentic workflows, from established patterns like generator-evaluator and LLM debate to recent innovations. ...
>
---
#### [replaced 042] MAWARITH: A Dataset and Benchmark for Legal Inheritance Reasoning with LLMs
- **分类: cs.CL**

- **简介: 该论文提出MAWARITH数据集和MIR-E评估指标，用于法律继承推理任务，解决伊斯兰继承法中复杂多步骤推理问题。**

- **链接: [https://arxiv.org/pdf/2603.07539](https://arxiv.org/pdf/2603.07539)**

> **作者:** Abdessalam Bouchekif; Shahd Gaben; Samer Rashwani; Somaya Eltanbouly; Mutaz Al-Khatib; Heba Sbahi; Mohammed Ghaly; Emad Mohamed
>
> **摘要:** Islamic inheritance law ('ilm al-mawarith) is challenging for large language models because solving inheritance cases requires complex, structured multi-step reasoning and the correct application of juristic rules to compute heirs' shares. We introduce MAWARITH, a large-scale annotated dataset of 12,500 Arabic inheritance cases for training and evaluating models on the full reasoning chain: (i) identifying eligible heirs, (ii) applying blocking (hajb) and allocation rules, and (iii) computing exact inheritance shares. Unlike prior datasets that restrict inheritance case solving to multiple-choice questions, MAWARITH supports the full reasoning chain and provides step-by-step solutions, including intermediate legal decisions and justifications based on classical juristic sources and established inheritance rules, as well as exact share calculations. To evaluate models beyond final-answer accuracy, we propose MIR-E (Mawarith Inheritance Reasoning Evaluation), a weighted multi-stage metric that scores key reasoning stages and captures error propagation across the pipeline. We evaluate six LLMs in a zero-shot setting. Gemini-2.5-flash achieves about 90% MIR-E on both validation and test, while Fanar-C, Fanar-Sadiq, LLaMA 3, and Qwen 3 remain below 50%. Our error analysis identifies recurring failure patterns, including scenario misinterpretation, errors in heir identification, errors in share allocation, and missing or incorrect application of key inheritance rules such as 'awl and radd. The MAWARITH dataset is publicly available at this https URL.
>
---
#### [replaced 043] LLM-Augmented Changepoint Detection: A Framework for Ensemble Detection and Automated Explanation
- **分类: cs.CL**

- **简介: 该论文属于时间序列分析任务，旨在解决 changepoint 检测的准确性与解释性问题。通过集成多种方法并结合 LLM 生成解释，提升检测效果与可理解性。**

- **链接: [https://arxiv.org/pdf/2601.02957](https://arxiv.org/pdf/2601.02957)**

> **作者:** Fabian Lukassen; Christoph Weisser; Michael Schlee; Manish Kumar; Anton Thielmann; Benjamin Saefken; Thomas Kneib
>
> **摘要:** This paper introduces a novel changepoint detection framework that combines ensemble statistical methods with Large Language Models (LLMs) to enhance both detection accuracy and the interpretability of regime changes in time series data. Two critical limitations in the field are addressed. First, individual detection methods exhibit complementary strengths and weaknesses depending on data characteristics, making method selection non-trivial and prone to suboptimal results. Second, automated, contextual explanations for detected changes are largely absent. The proposed ensemble method aggregates results from ten distinct changepoint detection algorithms, achieving superior performance and robustness compared to individual methods. Additionally, an LLM-powered explanation pipeline automatically generates contextual narratives, linking detected changepoints to potential real-world historical events. For private or domain-specific data, a Retrieval-Augmented Generation (RAG) solution enables explanations grounded in user-provided documents. The open source Python framework demonstrates practical utility in diverse domains, including finance, political science, and environmental science, transforming raw statistical output into actionable insights for analysts and decision-makers.
>
---
#### [replaced 044] Steering LLMs toward Korean Local Speech: Iterative Refinement Framework for Faithful Dialect Translation
- **分类: cs.CL**

- **简介: 该论文属于方言机器翻译任务，旨在解决大语言模型在方言翻译中的偏差问题。提出DIA-REFINE框架，通过迭代优化提升翻译准确性，并引入DFS和TDR评估指标。**

- **链接: [https://arxiv.org/pdf/2511.06680](https://arxiv.org/pdf/2511.06680)**

> **作者:** Keunhyeung Park; Seunguk Yu; Youngbin Kim
>
> **备注:** Accepted to LREC 2026
>
> **摘要:** Standard-to-dialect machine translation remains challenging due to a persistent dialect gap in large language models and evaluation distortions inherent in n-gram metrics, which favor source copying over authentic dialect translation. In this paper, we propose the dialect refinement (DIA-REFINE) framework, which guides LLMs toward faithful target dialect outputs through an iterative loop of translation, verification, and feedback using external dialect classifiers. To address the limitations of n-gram-based metrics, we introduce the dialect fidelity score (DFS) to quantify linguistic shift and the target dialect ratio (TDR) to measure the success of dialect translation. Experiments on Korean dialects across zero-shot and in-context learning baselines demonstrate that DIA-REFINE consistently enhances dialect fidelity. The proposed metrics distinguish between False Success cases, where high n-gram scores obscure failures in dialectal translation, and True Attempt cases, where genuine attempts at dialectal translation yield low n-gram scores. We also observed that models exhibit varying degrees of responsiveness to the framework, and that integrating in-context examples further improves the translation of dialectal expressions. Our work establishes a robust framework for goal-directed, inclusive dialect translation, providing both rigorous evaluation and critical insights into model performance.
>
---
#### [replaced 045] Prompt Sensitivity and Answer Consistency of Small Open-Source Language Models for Clinical Question Answering in Low-Resource Healthcare
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于临床问答任务，研究小规模开源模型在低资源医疗环境下的提示敏感性和答案一致性。通过不同提示风格评估模型表现，发现高一致性不等于准确性，强调需综合评估模型可靠性。**

- **链接: [https://arxiv.org/pdf/2603.00917](https://arxiv.org/pdf/2603.00917)**

> **作者:** Shravani Hariprasad
>
> **备注:** 30 pages, 7 figures, 2 tables
>
> **摘要:** Small open-source language models are gaining attention for healthcare applications in low-resource settings where cloud infrastructure and GPU hardware may be unavailable. However, the reliability of these models under different phrasings of the same clinical question remains poorly understood. We evaluate five open-source models (Gemma 2 2B, Phi-3 Mini 3.8B, Llama 3.2 3B, Mistral 7B, and Meditron-7B, a domain-pretrained model without instruction tuning) across three clinical question answering datasets (MedQA, MedMCQA, and PubMedQA) using five prompt styles: original, formal, simplified, roleplay, and direct. Model behavior is evaluated using consistency scores, accuracy, and instruction-following failure rates. All experiments were conducted locally on consumer CPU hardware without fine-tuning. Consistency and accuracy were largely independent across models. Gemma 2 achieved the highest consistency (0.845-0.888) but the lowest accuracy (33.0-43.5%), while Llama 3.2 showed moderate consistency (0.774-0.807) alongside the highest accuracy (49.0-65.0%). Roleplay prompts consistently reduced accuracy across all models, with Phi-3 Mini dropping 21.5 percentage points on MedQA. Meditron-7B exhibited near-complete instruction-following failure on PubMedQA (99.0% UNKNOWN rate), indicating that domain pretraining alone is insufficient for structured clinical question answering. These findings show that high consistency does not imply correctness: models can be reliably wrong, a dangerous failure mode in clinical AI. Llama 3.2 demonstrated the strongest balance of accuracy and reliability for low-resource deployment. Safe clinical AI requires joint evaluation of consistency, accuracy, and instruction adherence.
>
---
#### [replaced 046] Language as a Wave Phenomenon: Semantic Phase Locking and Interference in Neural Networks
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究语言作为波现象，探索相位在神经网络中的语义编码作用，解决相位在序列模型中功能不明确的问题。通过PRISM模型，验证相位可表征语义，并分析其有效性条件。**

- **链接: [https://arxiv.org/pdf/2512.01208](https://arxiv.org/pdf/2512.01208)**

> **作者:** Alper Yıldırım; İbrahim Yücedağ
>
> **备注:** Reframed as controlled experimental study, removed unsupported claims, added explicit hypotheses and statistical tests. Core results unchanged
>
> **摘要:** The role of phase in neural sequence models remains poorly understood. To isolate this question, we introduce PRISM, a complex-valued encoder that enforces a unit-norm constraint ($|z| = 1$) and replaces attention with gated spectral filtering. Under this constraint, the model cannot use activation magnitude to distinguish signal from noise, and must instead rely on phase angles. We find that semantic relationships correlate with measurable phase structure: synonym pairs exhibit significantly higher phase coherence than random pairs ($R = 0.198$ vs.\ $0.072$, $p < 0.001$), and the model resolves lexical ambiguity via layer-specific phase rotations while maintaining near-unit gain. These phase representations are robust to scalar attenuation, retaining $97\%$ of translation quality when signal magnitude is uniformly reduced. We also identify a spectral density threshold: the model fails to generate coherent output from isolated tokens, requiring minimum sequence length to produce the interference patterns that support its computation. Finally, we show that a hybrid architecture (Wave-Particle Transformer) combining a phase-based encoder with standard attention matches Transformer baselines at $33$M parameters with fewer non-embedding parameters, though we do not claim this generalizes to larger scales. Our findings provide controlled evidence that phase angles can encode semantic information in complex-valued networks, and characterize the conditions under which this encoding succeeds and fails.
>
---
#### [replaced 047] Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于人工智能可解释性研究，旨在解决隐式思维链的因果结构问题。通过构建结构因果模型，分析隐式推理步骤的因果关系与影响传播，提升对隐式推理系统的理解与优化。**

- **链接: [https://arxiv.org/pdf/2602.08783](https://arxiv.org/pdf/2602.08783)**

> **作者:** Zirui Li; Xuefeng Bai; Kehai Chen; Yizhi Li; Jian Yang; Chenghua Lin; Min Zhang
>
> **备注:** 22 pages, Accepted to ICLR 2026 Latent & Implicit Thinking Workshop
>
> **摘要:** Latent or continuous chain-of-thought methods replace explicit textual rationales with a number of internal latent steps, but these intermediate computations are difficult to evaluate beyond correlation-based probes. In this paper, we view latent chain-of-thought as a manipulable causal process in representation space by modeling latent steps as variables in a structural causal model (SCM) and analyzing their effects through step-wise $\mathrm{do}$-interventions. We study two representative paradigms (i.e., Coconut and CODI) on both mathematical and general reasoning tasks to investigate three key questions: (1) which steps are causally necessary for correctness and when answers become decidable early; (2) how does influence propagate across steps, and how does this structure compare to explicit CoT; and (3) do intermediate trajectories retain competing answer modes, and how does output-level commitment differ from representational commitment across steps. We find that latent-step budgets behave less like homogeneous extra depth and more like staged functionality with non-local routing, and we identify a persistent gap between early output bias and late representational commitment. These results motivate mode-conditional and stability-aware analyses -- and corresponding training/decoding objectives -- as more reliable tools for interpreting and improving latent reasoning systems. Code is available at this https URL.
>
---
#### [replaced 048] Beyond Polarity: Multi-Dimensional LLM Sentiment Signals for WTI Crude Oil Futures Return Prediction
- **分类: q-fin.ST; cs.CL**

- **简介: 该论文属于金融预测任务，旨在提升原油期货收益预测。通过多维情感信号分析新闻数据，解决传统极性方法不足的问题。结合LLM与传统模型，提升预测效果。**

- **链接: [https://arxiv.org/pdf/2603.11408](https://arxiv.org/pdf/2603.11408)**

> **作者:** Dehao Dai; Ding Ma; Dou Liu; Kerui Geng; Yiqing Wang
>
> **备注:** 28 pages, 4 figures, 4 tables
>
> **摘要:** Forecasting crude oil prices remains challenging because market-relevant information is embedded in large volumes of unstructured news and is not fully captured by traditional polarity-based sentiment measures. This paper examines whether multi-dimensional sentiment signals extracted by large language models improve the prediction of weekly WTI crude oil futures returns. Using energy-sector news articles from 2020 to 2025, we construct five sentiment dimensions covering relevance, polarity, intensity, uncertainty, and forwardness based on GPT-4o, Llama 3.2-3b, and two benchmark models, FinBERT and AlphaVantage. We aggregate article-level signals to the weekly level and evaluate their predictive performance in a classification framework. The best results are achieved by combining GPT-4o and FinBERT, suggesting that LLM-based and conventional financial sentiment models provide complementary predictive information. SHAP analysis further shows that intensity- and uncertainty-related features are among the most important predictors, indicating that the predictive value of news sentiment extends beyond simple polarity. Overall, the results suggest that multi-dimensional LLM-based sentiment measures can improve commodity return forecasting and support energy-market risk monitoring.
>
---
#### [replaced 049] Revisiting ASR Error Correction with Specialized Models
- **分类: cs.LG; cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音识别错误修正任务，旨在解决传统方法对ASR错误模式不敏感及大模型引入延迟和幻觉的问题。通过构建合成数据并使用轻量seq2seq模型实现高效准确的纠错。**

- **链接: [https://arxiv.org/pdf/2405.15216](https://arxiv.org/pdf/2405.15216)**

> **作者:** Zijin Gu; Tatiana Likhomanenko; He Bai; Erik McDermott; Ronan Collobert; Navdeep Jaitly
>
> **备注:** under review
>
> **摘要:** Language models play a central role in automatic speech recognition (ASR), yet most methods rely on text-only models unaware of ASR error patterns. Recently, large language models (LLMs) have been applied to ASR correction, but introduce latency and hallucination concerns. We revisit ASR error correction with compact seq2seq models, trained on ASR errors from real and synthetic audio. To scale training, we construct synthetic corpora via cascaded TTS and ASR, finding that matching the diversity of realistic error distributions is key. We propose correction-first decoding, where the correction model generates candidates rescored using ASR acoustic scores. With 15x fewer parameters than LLMs, our model achieves 1.5/3.3% WER on LibriSpeech test-clean/other, outperforms LLMs, generalizes across ASR architectures (CTC, Seq2seq, Transducer) and diverse domains, and provides precise corrections in the low-error regime where LLMs struggle.
>
---
#### [replaced 050] When Silence Matters: The Impact of Irrelevant Audio on Text Reasoning in Large Audio-Language Models
- **分类: cs.SD; cs.CL**

- **简介: 该论文研究音频干扰对文本推理的影响，属于多模态模型任务。解决音频噪声影响文本推理的问题，通过实验分析不同音频类型的影响，并测试缓解策略。**

- **链接: [https://arxiv.org/pdf/2510.00626](https://arxiv.org/pdf/2510.00626)**

> **作者:** Chen-An Li; Tzu-Han Lin; Hung-yi Lee
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** Large audio-language models (LALMs) unify speech and text processing, but their robustness in noisy real-world settings remains underexplored. We investigate how irrelevant audio, such as silence, synthetic noise, and environmental sounds, affects text reasoning tasks where audio is unnecessary. Across three text-based benchmarks, we find that even non-informative audio reduces accuracy and increases prediction volatility; the severity of interference scales with longer durations, higher amplitudes, and elevated decoding temperatures. Silence, often assumed neutral, destabilizes outputs as strongly as synthetic noise. While larger models show greater resilience, vulnerabilities persist across all evaluated systems. We further test mitigation strategies and find that prompting shows limited effectiveness, whereas self-consistency improves stability at the cost of increased computation. Our results reveal cross-modal interference as a key robustness challenge and highlight the need for efficient fusion strategies that preserve reasoning performance in the presence of irrelevant inputs.
>
---
#### [replaced 051] Token-Level LLM Collaboration via FusionRoute
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出FusionRoute，解决多大模型协作问题。通过轻量路由选择专家并补充修正，提升跨领域性能。属于模型协作任务。**

- **链接: [https://arxiv.org/pdf/2601.05106](https://arxiv.org/pdf/2601.05106)**

> **作者:** Nuoya Xiong; Yuhang Zhou; Hanqing Zeng; Zhaorun Chen; Furong Huang; Shuchao Bi; Lizhu Zhang; Zhuokai Zhao
>
> **备注:** 25 pages
>
> **摘要:** Large language models (LLMs) exhibit strengths across diverse domains. However, achieving strong performance across these domains with a single general-purpose model typically requires scaling to sizes that are prohibitively expensive to train and deploy. On the other hand, while smaller domain-specialized models are much more efficient, they struggle to generalize beyond their training distributions. To address this dilemma, we propose FusionRoute, a robust and effective token-level multi-LLM collaboration framework in which a lightweight router simultaneously (i) selects the most suitable expert at each decoding step and (ii) contributes a complementary logit that refines or corrects the selected expert's next-token distribution via logit addition. Unlike existing token-level collaboration methods that rely solely on fixed expert outputs, we provide a theoretical analysis showing that pure expert-only routing is fundamentally limited: unless strong global coverage assumptions hold, it cannot in general realize the optimal decoding policy. By augmenting expert selection with a trainable complementary generator, FusionRoute expands the effective policy class and enables recovery of optimal value functions under mild conditions. Empirically, across both Llama-3 and Gemma-2 families and diverse benchmarks spanning mathematical reasoning, code generation, and instruction following, FusionRoute outperforms both sequence- and token-level collaboration, model merging, and direct fine-tuning, while remaining competitive with domain experts on their respective tasks.
>
---
#### [replaced 052] VorTEX: Various overlap ratio for Target speech EXtraction
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文属于语音分离任务，解决真实场景下不同重叠比例的语音提取问题。提出VorTEX模型和PORTE数据集，提升分离效果并减少干扰。**

- **链接: [https://arxiv.org/pdf/2603.14803](https://arxiv.org/pdf/2603.14803)**

> **作者:** Ro-hoon Oh; Jihwan Seol; Bugeun Kim
>
> **备注:** arXiv Preprint
>
> **摘要:** Target speech extraction (TSE) aims to recover a target speaker's voice from a mixture. While recent text-prompted approaches have shown promise, most approaches assume fully overlapped mixtures, limiting insight into behavior across realistic overlap ratios. We introduce VorTEX (Various overlap ratio for Target speech EXtraction), a text-prompted TSE architecture with a Decoupled Adaptive Multi-branch (DAM) Fusion block that separates primary extraction from auxiliary regularization pathways. To enable controlled analysis, we construct PORTE, a two-speaker dataset spanning overlap ratios from 0% to 100%. We further propose Suppression Ratio on Energy (SuRE), a diagnostic metric that detects suppression behavior not captured by conventional measures. Experiments show that existing models exhibit suppression or residual interference under overlap, whereas VorTEX achieves the highest separation fidelity across 20-100% overlap (e.g., 5.50 dB at 20% and 2.04 dB at 100%) while maintaining zero SuRE, indicating robust extraction without suppression-driven artifacts.
>
---
#### [replaced 053] Protecting De-identified Documents from Search-based Linkage Attacks
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于隐私保护任务，旨在解决去标识文档面临的基于搜索的链接攻击问题。通过构建倒排索引并使用语言模型重写敏感片段，有效防止链接同时保持语义一致。**

- **链接: [https://arxiv.org/pdf/2510.06383](https://arxiv.org/pdf/2510.06383)**

> **作者:** Pierre Lison; Mark Anderson
>
> **摘要:** While de-identification models can help conceal the identity of the individuals mentioned in a document, they fail to address linkage risks, defined as the potential to map the de-identified text back to its source. One straightforward way to perform such linkages is to extract phrases from the de-identified document and check their presence in the original dataset. This paper presents a method to counter search-based linkage attacks while preserving the semantic integrity of the text. The method proceeds in two steps. We first construct an inverted index of the N-grams occurring in the text collection, making it possible to efficiently determine which N-grams appear in fewer than $k$ documents, either alone or in combination with other N-grams. An LLM-based rewriter is then iteratively queried to reformulate those spans until linkage is no longer possible. Experimental results on two datasets (court cases and Wikipedia biographies) show that the rewriting method can effectively prevent search-based linkages while remaining faithful to the original content. However, we also highlight that linkages remain feasible with the help of more advanced, semantics-oriented approaches.
>
---
#### [replaced 054] Who's important? -- SUnSET: Synergistic Understanding of Stakeholder, Events and Time for Timeline Generation
- **分类: cs.SI; cs.CL; cs.IR**

- **简介: 该论文属于时间线摘要任务，旨在解决多源新闻事件跟踪难题。通过构建SUnSET框架，结合利益相关者分析与事件关联，提升摘要效果。**

- **链接: [https://arxiv.org/pdf/2507.21903](https://arxiv.org/pdf/2507.21903)**

> **作者:** Tiviatis Sim; Kaiwen Yang; Shen Xin; Kenji Kawaguchi
>
> **摘要:** As news reporting becomes increasingly global and decentralized online, tracking related events across multiple sources presents significant challenges. Existing news summarization methods typically utilizes Large Language Models and Graphical methods on article-based summaries. However, this is not effective since it only considers the textual content of similarly dated articles to understand the gist of the event. To counteract the lack of analysis on the parties involved, it is essential to come up with a novel framework to gauge the importance of stakeholders and the connection of related events through the relevant entities involved. Therefore, we present SUnSET: Synergistic Understanding of Stakeholder, Events and Time for the task of Timeline Summarization (TLS). We leverage powerful Large Language Models (LLMs) to build SET triplets and introduced the use of stakeholder-based ranking to construct a $Relevancy$ metric, which can be extended into general situations. Our experimental results outperform all prior baselines and emerged as the new State-of-the-Art, highlighting the impact of stakeholder information within news article.
>
---
#### [replaced 055] Toward Better Temporal Structures for Geopolitical Events Forecasting
- **分类: cs.CL**

- **简介: 该论文属于 geopolitical events forecasting 任务，旨在解决 HTKGs 在表达复杂事实上的不足。通过提出 HTKGH 模型和构建数据集，评估 LLMs 的预测能力。**

- **链接: [https://arxiv.org/pdf/2601.00430](https://arxiv.org/pdf/2601.00430)**

> **作者:** Kian Ahrabian; Eric Boxer; Jay Pujara
>
> **备注:** 18 pages, 15 figures, 3 tables
>
> **摘要:** Forecasting on geopolitical temporal knowledge graphs (TKGs) through the lens of large language models (LLMs) has recently gained traction. While TKGs and their generalization, hyper-relational temporal knowledge graphs (HTKGs), offer a straightforward structure to represent simple temporal relationships, they lack the expressive power to convey complex facts efficiently. One of the critical limitations of HTKGs is a lack of support for more than two primary entities in temporal facts, which commonly occur in real-world events. To address this limitation, in this work, we study a generalization of HTKGs, Hyper-Relational Temporal Knowledge Generalized Hypergraphs (HTKGHs). We first derive a formalization for HTKGHs, demonstrating their backward compatibility while supporting two complex types of facts commonly found in geopolitical incidents. Then, utilizing this formalization, we introduce the htkgh-polecat dataset, built upon the global event database POLECAT. Finally, we benchmark and analyze popular LLMs on our dataset, providing insights into 1) the positive impact of utilizing the HTKGH formalization compared to existing ones and 2) LLMs' adaptability and capabilities in complex forecasting tasks.
>
---
