# 自然语言处理 cs.CL

- **最新发布 111 篇**

- **更新 75 篇**

## 最新发布

#### [new 001] Mind the Generation Process: Fine-Grained Confidence Estimation During LLM Generation
- **分类: cs.CL; cs.AI**

- **简介: 论文提出FineCE方法，解决大语言模型生成过程中缺乏细粒度置信度估计的问题。通过构建训练数据和引入后向置信度整合策略，实现生成阶段连续、准确的置信度预测，提升输出可靠性。**

- **链接: [http://arxiv.org/pdf/2508.12040v1](http://arxiv.org/pdf/2508.12040v1)**

> **作者:** Jinyi Han; Tingyun Li; Shisong Chen; Jie Shi; Xinyi Wang; Guanglei Yue; Jiaqing Liang; Xin Lin; Liqian Wen; Zulong Chen; Yanghua Xiao
>
> **备注:** The initial versin was made in August 2024
>
> **摘要:** While large language models (LLMs) have demonstrated remarkable performance across diverse tasks, they fundamentally lack self-awareness and frequently exhibit overconfidence, assigning high confidence scores to incorrect predictions. Accurate confidence estimation is therefore critical for enhancing the trustworthiness and reliability of LLM-generated outputs. However, existing approaches suffer from coarse-grained scoring mechanisms that fail to provide fine-grained, continuous confidence estimates throughout the generation process. To address these limitations, we introduce FineCE, a novel confidence estimation method that delivers accurate, fine-grained confidence scores during text generation. Specifically, we first develop a comprehensive pipeline for constructing training data that effectively captures the underlying probabilistic distribution of LLM responses, and then train a model to predict confidence scores for arbitrary text sequences in a supervised manner. Furthermore, we propose a Backward Confidence Integration (BCI) strategy that leverages information from the subsequent text to enhance confidence estimation for the current sequence during inference. We also introduce three strategies for identifying optimal positions to perform confidence estimation within the generation process. Extensive experiments on multiple benchmark datasets demonstrate that FineCE consistently outperforms existing classical confidence estimation methods. Our code and all baselines used in the paper are available on GitHub.
>
---
#### [new 002] Prompt-Induced Linguistic Fingerprints for LLM-Generated Fake News Detection
- **分类: cs.CL**

- **简介: 论文提出LIFE方法，通过分析提示诱导的语言指纹来检测大语言模型生成的假新闻。该任务旨在识别LLM生成内容中的细微差异，解决传统方法难以发现此类假新闻的问题。工作包括构建词级概率分布和关键片段增强技术，实现高精度检测。**

- **链接: [http://arxiv.org/pdf/2508.12632v1](http://arxiv.org/pdf/2508.12632v1)**

> **作者:** Chi Wang; Min Gao; Zongwei Wang; Junwei Yin; Kai Shu; Chenghua Lin
>
> **摘要:** With the rapid development of large language models, the generation of fake news has become increasingly effortless, posing a growing societal threat and underscoring the urgent need for reliable detection methods. Early efforts to identify LLM-generated fake news have predominantly focused on the textual content itself; however, because much of that content may appear coherent and factually consistent, the subtle traces of falsification are often difficult to uncover. Through distributional divergence analysis, we uncover prompt-induced linguistic fingerprints: statistically distinct probability shifts between LLM-generated real and fake news when maliciously prompted. Based on this insight, we propose a novel method named Linguistic Fingerprints Extraction (LIFE). By reconstructing word-level probability distributions, LIFE can find discriminative patterns that facilitate the detection of LLM-generated fake news. To further amplify these fingerprint patterns, we also leverage key-fragment techniques that accentuate subtle linguistic differences, thereby improving detection reliability. Our experiments show that LIFE achieves state-of-the-art performance in LLM-generated fake news and maintains high performance in human-written fake news. The code and data are available at https://anonymous.4open.science/r/LIFE-E86A.
>
---
#### [new 003] DocHPLT: A Massively Multilingual Document-Level Translation Dataset
- **分类: cs.CL**

- **简介: 论文提出DocHPLT，一个大规模多语言文档级翻译数据集，解决现有资源稀缺且局限于高资源语言的问题。通过保留文档完整性构建50语言、4.26亿句的语料，实验证明其能显著提升模型性能，尤其对低资源语言效果明显。**

- **链接: [http://arxiv.org/pdf/2508.13079v1](http://arxiv.org/pdf/2508.13079v1)**

> **作者:** Dayyán O'Brien; Bhavitvya Malik; Ona de Gibert; Pinzhen Chen; Barry Haddow; Jörg Tiedemann
>
> **摘要:** Existing document-level machine translation resources are only available for a handful of languages, mostly high-resourced ones. To facilitate the training and evaluation of document-level translation and, more broadly, long-context modeling for global communities, we create DocHPLT, the largest publicly available document-level translation dataset to date. It contains 124 million aligned document pairs across 50 languages paired with English, comprising 4.26 billion sentences, with further possibility to provide 2500 bonus pairs not involving English. Unlike previous reconstruction-based approaches that piece together documents from sentence-level data, we modify an existing web extraction pipeline to preserve complete document integrity from the source, retaining all content including unaligned portions. After our preliminary experiments identify the optimal training context strategy for document-level translation, we demonstrate that LLMs fine-tuned on DocHPLT substantially outperform off-the-shelf instruction-tuned baselines, with particularly dramatic improvements for under-resourced languages. We open-source the dataset under a permissive license, providing essential infrastructure for advancing multilingual document-level translation.
>
---
#### [new 004] Structuring the Unstructured: A Systematic Review of Text-to-Structure Generation for Agentic AI with a Universal Evaluation Framework
- **分类: cs.CL**

- **简介: 论文聚焦文本到结构化数据生成任务，解决当前缺乏系统性方法、数据集和评估标准的问题。作者综述现有技术与挑战，提出通用评估框架，推动AI系统向更高效结构化处理发展。**

- **链接: [http://arxiv.org/pdf/2508.12257v1](http://arxiv.org/pdf/2508.12257v1)**

> **作者:** Zheye Deng; Chunkit Chan; Tianshi Zheng; Wei Fan; Weiqi Wang; Yangqiu Song
>
> **备注:** Under Review
>
> **摘要:** The evolution of AI systems toward agentic operation and context-aware retrieval necessitates transforming unstructured text into structured formats like tables, knowledge graphs, and charts. While such conversions enable critical applications from summarization to data mining, current research lacks a comprehensive synthesis of methodologies, datasets, and metrics. This systematic review examines text-to-structure techniques and the encountered challenges, evaluates current datasets and assessment criteria, and outlines potential directions for future research. We also introduce a universal evaluation framework for structured outputs, establishing text-to-structure as foundational infrastructure for next-generation AI systems.
>
---
#### [new 005] Legal$Δ$: Enhancing Legal Reasoning in LLMs via Reinforcement Learning with Chain-of-Thought Guided Information Gain
- **分类: cs.CL**

- **简介: 该论文属于法律推理任务，旨在提升大语言模型在法律场景下的推理可靠性和可解释性。针对模型易产生表面答案的问题，提出LegalΔ框架，通过强化学习与链式思维引导的信息增益机制，训练模型生成结构合理、领域相关的多步推理过程。**

- **链接: [http://arxiv.org/pdf/2508.12281v1](http://arxiv.org/pdf/2508.12281v1)**

> **作者:** Xin Dai; Buqiang Xu; Zhenghao Liu; Yukun Yan; Huiyuan Xie; Xiaoyuan Yi; Shuo Wang; Ge Yu
>
> **摘要:** Legal Artificial Intelligence (LegalAI) has achieved notable advances in automating judicial decision-making with the support of Large Language Models (LLMs). However, existing legal LLMs still struggle to generate reliable and interpretable reasoning processes. They often default to fast-thinking behavior by producing direct answers without explicit multi-step reasoning, limiting their effectiveness in complex legal scenarios that demand rigorous justification. To address this challenge, we propose Legal$\Delta$, a reinforcement learning framework designed to enhance legal reasoning through chain-of-thought guided information gain. During training, Legal$\Delta$ employs a dual-mode input setup-comprising direct answer and reasoning-augmented modes-and maximizes the information gain between them. This encourages the model to acquire meaningful reasoning patterns rather than generating superficial or redundant explanations. Legal$\Delta$ follows a two-stage approach: (1) distilling latent reasoning capabilities from a powerful Large Reasoning Model (LRM), DeepSeek-R1, and (2) refining reasoning quality via differential comparisons, combined with a multidimensional reward mechanism that assesses both structural coherence and legal-domain specificity. Experimental results on multiple legal reasoning tasks demonstrate that Legal$\Delta$ outperforms strong baselines in both accuracy and interpretability. It consistently produces more robust and trustworthy legal judgments without relying on labeled preference data. All code and data will be released at https://github.com/NEUIR/LegalDelta.
>
---
#### [new 006] Atom-Searcher: Enhancing Agentic Deep Research via Fine-Grained Atomic Thought Reward
- **分类: cs.CL; cs.AI**

- **简介: 论文提出Atom-Searcher框架，解决LLM在复杂任务中因静态知识和奖励稀疏导致的推理效率低问题。通过细粒度原子思维与过程奖励机制，提升多跳推理能力和训练效率，实现更可解释的自主研究。**

- **链接: [http://arxiv.org/pdf/2508.12800v1](http://arxiv.org/pdf/2508.12800v1)**

> **作者:** Yong Deng; Guoqing Wang; Zhenzhe Ying; Xiaofeng Wu; Jinzhen Lin; Wenwen Xiong; Yuqin Dai; Shuo Yang; Zhanwei Zhang; Qiwen Wang; Yang Qin; Changhua Meng
>
> **摘要:** Large language models (LLMs) exhibit remarkable problem-solving abilities, but struggle with complex tasks due to static internal knowledge. Retrieval-Augmented Generation (RAG) enhances access to external information, yet remains limited in multi-hop reasoning and strategic search due to rigid workflows. Recent advancements in agentic deep research empower LLMs to autonomously reason, search, and synthesize information. However, current approaches relying on outcome-based reinforcement learning (RL) face critical issues such as conflicting gradients and reward sparsity, limiting performance gains and training efficiency. To address these, we first propose Atomic Thought, a novel LLM thinking paradigm that decomposes reasoning into fine-grained functional units. These units are supervised by Reasoning Reward Models (RRMs), which provide Atomic Thought Rewards (ATR) for fine-grained guidance. Building on this, we propose Atom-Searcher, a novel RL framework for agentic deep research that integrates Atomic Thought and ATR. Atom-Searcher uses a curriculum-inspired reward schedule, prioritizing process-level ATR early and transitioning to outcome rewards, accelerating convergence on effective reasoning paths. Experiments on seven benchmarks show consistent improvements over the state-of-the-art. Key advantages include: (1) Atom-Searcher scales computation at test-time. (2) Atomic Thought provides supervision anchors for RRMs, bridging deep research tasks and RRMs. (3) Atom-Searcher exhibits more interpretable, human-like reasoning patterns.
>
---
#### [new 007] J6: Jacobian-Driven Role Attribution for Multi-Objective Prompt Optimization in LLMs
- **分类: cs.CL; cs.AI; cs.LG; 68T50, 90C29, 62F07; I.2.7; I.2.6; G.1.6**

- **简介: 该论文针对大语言模型提示优化中的多目标冲突问题，提出J6方法，通过分解雅可比矩阵揭示参数与目标间的几何关系，实现冲突感知的动态更新与可解释的参数归因。**

- **链接: [http://arxiv.org/pdf/2508.12086v1](http://arxiv.org/pdf/2508.12086v1)**

> **作者:** Yao Wu
>
> **备注:** 9 pages, 3 tables, 1 algorithm
>
> **摘要:** In large language model (LLM) adaptation, balancing multiple optimization objectives such as improving factuality (heat) and increasing confidence (via low entropy) poses a fundamental challenge, especially when prompt parameters (e.g., hidden-layer insertions h and embedding modifications w) interact in non-trivial ways. Existing multi-objective optimization strategies often rely on scalar gradient aggregation, ignoring the deeper geometric structure between objectives and parameters. We propose J6, a structured Jacobian-based method that decomposes the gradient interaction matrix into six interpretable components. This decomposition enables both hard decision-making (e.g., choosing the dominant update direction via argmax) and soft strategies (e.g., attention-style weighting via softmax over J6), forming a dynamic update framework that adapts to local conflict and synergy. Moreover, the interpretable structure of J6 provides insight into parameter attribution, task interference, and geometry-aligned adaptation. Our work introduces a principled and extensible mechanism for conflict-aware prompt optimization, and opens a new avenue for incorporating structured Jacobian reasoning into multi-objective neural tuning.
>
---
#### [new 008] DESIGNER: Design-Logic-Guided Multidisciplinary Data Synthesis for LLM Reasoning
- **分类: cs.CL**

- **简介: 论文提出DESIGNER框架，通过设计逻辑引导的多学科数据合成方法，生成高难度、多样化推理问题。旨在解决大语言模型在跨学科复杂推理上的不足，提升其多领域推理能力。**

- **链接: [http://arxiv.org/pdf/2508.12726v1](http://arxiv.org/pdf/2508.12726v1)**

> **作者:** Weize Liu; Yongchi Zhao; Yijia Luo; Mingyu Xu; Jiaheng Liu; Yanan Li; Xiguo Hu; Yuchi Xu; Wenbo Su; Bo Zheng
>
> **摘要:** Large language models (LLMs) have achieved remarkable success in many natural language tasks but still struggle with complex, multi-step reasoning, particularly across diverse disciplines. Existing reasoning datasets often either lack disciplinary breadth or the structural depth necessary to elicit robust reasoning behaviors. We propose DESIGNER: a DESIGN-logic-guidEd Reasoning data synthesis pipeline that leverages naturally available, extensive raw documents (book corpus and web corpus) to generate multidisciplinary challenging questions. A core innovation of our approach is the introduction of a Design Logic concept, which mimics the question-creation process of human educators. We use LLMs to reverse-engineer and abstract over 120,000 design logics from existing questions across various disciplines. By matching these design logics with disciplinary source materials, we are able to create reasoning questions that far surpass the difficulty and diversity of existing datasets. Based on this pipeline, we synthesized two large-scale reasoning datasets that span 75 disciplines: Design-Logic-Reasoning-Book (DLR-Book), containing 3.04 million challenging questions synthesized from the book corpus, and Design-Logic-Reasoning-Web (DLR-Web), with 1.66 million challenging questions from the web corpus. Our data analysis demonstrates that the questions synthesized by our method exhibit substantially greater difficulty and diversity than those in the baseline datasets. We validate the effectiveness of these datasets by conducting SFT experiments on the Qwen3-8B-Base and Qwen3-4B-Base models. The results show that our dataset significantly outperforms existing multidisciplinary datasets of the same volume. Training with the full datasets further enables the models to surpass the multidisciplinary reasoning performance of the official Qwen3-8B and Qwen3-4B models.
>
---
#### [new 009] Can Large Models Teach Student Models to Solve Mathematical Problems Like Human Beings? A Reasoning Distillation Method via Multi-LoRA Interaction
- **分类: cs.CL; cs.AI**

- **简介: 论文提出LoRID方法，通过模拟人类双系统思维（直觉与深度推理）提升小模型数学推理能力。解决小模型推理弱的问题，利用多LoRA交互机制实现知识增强与迭代反馈，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.13037v1](http://arxiv.org/pdf/2508.13037v1)**

> **作者:** Xinhe Li; Jiajun Liu; Peng Wang
>
> **备注:** Accepted by IJCAI2025
>
> **摘要:** Recent studies have demonstrated that Large Language Models (LLMs) have strong mathematical reasoning abilities but rely on hundreds of billions of parameters. To tackle the challenge of poor reasoning in Small Language Models (SLMs), existing methods typically leverage LLMs to generate massive amounts of data for cramming training. In psychology, they are akin to System 1 thinking, which resolves reasoning problems rapidly based on experience and intuition. However, human learning also requires System 2 thinking, where knowledge is first acquired and then reinforced through practice. Inspired by such two distinct modes of thinking, we propose a novel method based on the multi-LoRA Interaction for mathematical reasoning Distillation (LoRID). First, we input the question and reasoning of each sample into an LLM to create knowledge-enhanced datasets. Subsequently, we train a LoRA block on the student model as an Intuitive Reasoner (IR), which directly generates Chain-of-Thoughts for problem-solving. Then, to imitate System 2 thinking, we train the Knowledge Generator (KG) and Deep Reasoner (DR), respectively. The former outputs only knowledge after receiving problems, while the latter uses that knowledge to perform reasoning. Finally, to address the randomness in the generation of IR and DR, we evaluate whether their outputs are consistent, and the inference process needs to be iterated if not. This step can enhance the mathematical reasoning ability of SLMs through mutual feedback. Experimental results show that LoRID achieves state-of-the-art performance, especially on the GSM8K dataset, where it outperforms the second-best method by 2.3%, 16.1%, 2.4%, 12.3%, and 1.8% accuracy across the five base models, respectively.
>
---
#### [new 010] A Question Answering Dataset for Temporal-Sensitive Retrieval-Augmented Generation
- **分类: cs.CL; cs.IR; 68T50, 68P20; I.2.7; H.3.3**

- **简介: 论文提出ChronoQA数据集，用于评估中文时间敏感的检索增强生成（RAG）系统。针对时间推理能力不足的问题，构建了包含5176个高质量问题的大规模数据集，涵盖多种时间类型和文档场景，支持结构化评估与模型改进。**

- **链接: [http://arxiv.org/pdf/2508.12282v1](http://arxiv.org/pdf/2508.12282v1)**

> **作者:** Ziyang Chen; Erxue Min; Xiang Zhao; Yunxin Li; Xin Jia; Jinzhi Liao; Jichao Li; Shuaiqiang Wang; Baotian Hu; Dawei Yin
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** We introduce ChronoQA, a large-scale benchmark dataset for Chinese question answering, specifically designed to evaluate temporal reasoning in Retrieval-Augmented Generation (RAG) systems. ChronoQA is constructed from over 300,000 news articles published between 2019 and 2024, and contains 5,176 high-quality questions covering absolute, aggregate, and relative temporal types with both explicit and implicit time expressions. The dataset supports both single- and multi-document scenarios, reflecting the real-world requirements for temporal alignment and logical consistency. ChronoQA features comprehensive structural annotations and has undergone multi-stage validation, including rule-based, LLM-based, and human evaluation, to ensure data quality. By providing a dynamic, reliable, and scalable resource, ChronoQA enables structured evaluation across a wide range of temporal tasks, and serves as a robust benchmark for advancing time-sensitive retrieval-augmented question answering systems.
>
---
#### [new 011] When Does Language Transfer Help? Sequential Fine-Tuning for Cross-Lingual Euphemism Detection
- **分类: cs.CL; cs.AI**

- **简介: 论文研究跨语言迁移在 euphemism 检测任务中的作用，解决低资源语言性能差的问题。通过顺序微调策略，利用高资源语言提升低资源语言表现，比较 XLM-R 与 mBERT 效果，发现顺序微调有效且稳定。**

- **链接: [http://arxiv.org/pdf/2508.11831v1](http://arxiv.org/pdf/2508.11831v1)**

> **作者:** Julia Sammartino; Libby Barak; Jing Peng; Anna Feldman
>
> **备注:** RANLP 2025
>
> **摘要:** Euphemisms are culturally variable and often ambiguous, posing challenges for language models, especially in low-resource settings. This paper investigates how cross-lingual transfer via sequential fine-tuning affects euphemism detection across five languages: English, Spanish, Chinese, Turkish, and Yoruba. We compare sequential fine-tuning with monolingual and simultaneous fine-tuning using XLM-R and mBERT, analyzing how performance is shaped by language pairings, typological features, and pretraining coverage. Results show that sequential fine-tuning with a high-resource L1 improves L2 performance, especially for low-resource languages like Yoruba and Turkish. XLM-R achieves larger gains but is more sensitive to pretraining gaps and catastrophic forgetting, while mBERT yields more stable, though lower, results. These findings highlight sequential fine-tuning as a simple yet effective strategy for improving euphemism detection in multilingual models, particularly when low-resource languages are involved.
>
---
#### [new 012] ReaLM: Reflection-Enhanced Autonomous Reasoning with Small Language Models
- **分类: cs.CL**

- **简介: 论文提出ReaLM框架，解决小语言模型在复杂推理中能力弱、依赖外部信号和泛化差的问题。通过多路径验证、渐进式自引导训练和知识蒸馏，提升推理能力、自主性和泛化性。**

- **链接: [http://arxiv.org/pdf/2508.12387v1](http://arxiv.org/pdf/2508.12387v1)**

> **作者:** Yuanfeng Xu; Zehui Dai; Jian Liang; Jiapeng Guan; Guangrun Wang; Liang Lin; Xiaohui Lv
>
> **备注:** 16pages, 3 figures
>
> **摘要:** Small Language Models (SLMs) are a cost-effective alternative to Large Language Models (LLMs), but often struggle with complex reasoning due to their limited capacity and a tendency to produce mistakes or inconsistent answers during multi-step reasoning. Existing efforts have improved SLM performance, but typically at the cost of one or more of three key aspects: (1) reasoning capability, due to biased supervision that filters out negative reasoning paths and limits learning from errors; (2) autonomy, due to over-reliance on externally generated reasoning signals; and (3) generalization, which suffers when models overfit to teacher-specific patterns. In this paper, we introduce ReaLM, a reinforcement learning framework for robust and self-sufficient reasoning in vertical domains. To enhance reasoning capability, we propose Multi-Route Process Verification (MRPV), which contrasts both positive and negative reasoning paths to extract decisive patterns. To reduce reliance on external guidance and improve autonomy, we introduce Enabling Autonomy via Asymptotic Induction (EAAI), a training strategy that gradually fades external signals. To improve generalization, we apply guided chain-of-thought distillation to encode domain-specific rules and expert knowledge into SLM parameters, making them part of what the model has learned. Extensive experiments on both vertical and general reasoning tasks demonstrate that ReaLM significantly improves SLM performance across aspects (1)-(3) above.
>
---
#### [new 013] From SALAMANDRA to SALAMANDRATA: BSC Submission for WMT25 General Machine Translation Shared Task
- **分类: cs.CL**

- **简介: 该论文针对机器翻译任务，提出SALAMANDRATA模型家族，解决多语言翻译性能优化问题。通过持续预训练与监督微调提升38种欧洲语言的翻译效果，并扩展至非欧洲语言，采用质量感知解码策略提升翻译质量。**

- **链接: [http://arxiv.org/pdf/2508.12774v1](http://arxiv.org/pdf/2508.12774v1)**

> **作者:** Javier Garcia Gilabert; Xixian Liao; Severino Da Dalt; Ella Bohman; Audrey Mash; Francesca De Luca Fornaciari; Irene Baucells; Joan Llop; Miguel Claramunt Argote; Carlos Escolano; Maite Melero
>
> **摘要:** In this paper, we present the SALAMANDRATA family of models, an improved iteration of SALAMANDRA LLMs (Gonzalez-Agirre et al., 2025) specifically trained to achieve strong performance in translation-related tasks for 38 European languages. SALAMANDRATA comes in two scales: 2B and 7B parameters. For both versions, we applied the same training recipe with a first step of continual pre-training on parallel data, and a second step of supervised fine-tuning on high-quality instructions. The BSC submission to the WMT25 General Machine Translation shared task is based on the 7B variant of SALAMANDRATA. We first adapted the model vocabulary to support the additional non-European languages included in the task. This was followed by a second phase of continual pre-training and supervised fine-tuning, carefully designed to optimize performance across all translation directions for this year's shared task. For decoding, we employed two quality-aware strategies: Minimum Bayes Risk Decoding and Tuned Re-ranking using COMET and COMET-KIWI respectively. We publicly release both the 2B and 7B versions of SALAMANDRATA, along with the newer SALAMANDRATA-V2 model, on Hugging Face1
>
---
#### [new 014] Extracting Post-Acute Sequelae of SARS-CoV-2 Infection Symptoms from Clinical Notes via Hybrid Natural Language Processing
- **分类: cs.CL; cs.AI**

- **简介: 论文提出混合NLP方法，从临床笔记中提取和判断新冠后遗症（PASC）症状，解决症状多样且时间跨度长导致的诊断难题。构建PASC词典，验证模型在多中心数据上的有效性与效率。**

- **链接: [http://arxiv.org/pdf/2508.12405v1](http://arxiv.org/pdf/2508.12405v1)**

> **作者:** Zilong Bai; Zihan Xu; Cong Sun; Chengxi Zang; H. Timothy Bunnell; Catherine Sinfield; Jacqueline Rutter; Aaron Thomas Martinez; L. Charles Bailey; Mark Weiner; Thomas R. Campion; Thomas Carton; Christopher B. Forrest; Rainu Kaushal; Fei Wang; Yifan Peng
>
> **备注:** Accepted for publication in npj Health Systems
>
> **摘要:** Accurately and efficiently diagnosing Post-Acute Sequelae of COVID-19 (PASC) remains challenging due to its myriad symptoms that evolve over long- and variable-time intervals. To address this issue, we developed a hybrid natural language processing pipeline that integrates rule-based named entity recognition with BERT-based assertion detection modules for PASC-symptom extraction and assertion detection from clinical notes. We developed a comprehensive PASC lexicon with clinical specialists. From 11 health systems of the RECOVER initiative network across the U.S., we curated 160 intake progress notes for model development and evaluation, and collected 47,654 progress notes for a population-level prevalence study. We achieved an average F1 score of 0.82 in one-site internal validation and 0.76 in 10-site external validation for assertion detection. Our pipeline processed each note at $2.448\pm 0.812$ seconds on average. Spearman correlation tests showed $\rho >0.83$ for positive mentions and $\rho >0.72$ for negative ones, both with $P <0.0001$. These demonstrate the effectiveness and efficiency of our models and their potential for improving PASC diagnosis.
>
---
#### [new 015] LinguaSafe: A Comprehensive Multilingual Safety Benchmark for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出LinguaSafe，一个涵盖12种语言的多语言安全评估基准，解决现有评测在多样性和文化真实性上的不足。通过4.5万条数据和细粒度评估框架，提升大模型跨语言安全对齐效果。**

- **链接: [http://arxiv.org/pdf/2508.12733v1](http://arxiv.org/pdf/2508.12733v1)**

> **作者:** Zhiyuan Ning; Tianle Gu; Jiaxin Song; Shixin Hong; Lingyu Li; Huacan Liu; Jie Li; Yixu Wang; Meng Lingyu; Yan Teng; Yingchun Wang
>
> **备注:** 7pages, 5 figures
>
> **摘要:** The widespread adoption and increasing prominence of large language models (LLMs) in global technologies necessitate a rigorous focus on ensuring their safety across a diverse range of linguistic and cultural contexts. The lack of a comprehensive evaluation and diverse data in existing multilingual safety evaluations for LLMs limits their effectiveness, hindering the development of robust multilingual safety alignment. To address this critical gap, we introduce LinguaSafe, a comprehensive multilingual safety benchmark crafted with meticulous attention to linguistic authenticity. The LinguaSafe dataset comprises 45k entries in 12 languages, ranging from Hungarian to Malay. Curated using a combination of translated, transcreated, and natively-sourced data, our dataset addresses the critical need for multilingual safety evaluations of LLMs, filling the void in the safety evaluation of LLMs across diverse under-represented languages from Hungarian to Malay. LinguaSafe presents a multidimensional and fine-grained evaluation framework, with direct and indirect safety assessments, including further evaluations for oversensitivity. The results of safety and helpfulness evaluations vary significantly across different domains and different languages, even in languages with similar resource levels. Our benchmark provides a comprehensive suite of metrics for in-depth safety evaluation, underscoring the critical importance of thoroughly assessing multilingual safety in LLMs to achieve more balanced safety alignment. Our dataset and code are released to the public to facilitate further research in the field of multilingual LLM safety.
>
---
#### [new 016] Semantic Anchoring in Agentic Memory: Leveraging Linguistic Structures for Persistent Conversational Context
- **分类: cs.CL**

- **简介: 论文提出Semantic Anchoring机制，解决LLM在多轮对话中记忆持久性不足的问题。通过融合句法、话语和指代关系等语言结构信息，增强向量存储的语义精度，提升事实回忆与连贯性。**

- **链接: [http://arxiv.org/pdf/2508.12630v1](http://arxiv.org/pdf/2508.12630v1)**

> **作者:** Maitreyi Chatterjee; Devansh Agarwal
>
> **备注:** Paper is currently in peer review
>
> **摘要:** Large Language Models (LLMs) have demonstrated impressive fluency and task competence in conversational settings. However, their effectiveness in multi-session and long-term interactions is hindered by limited memory persistence. Typical retrieval-augmented generation (RAG) systems store dialogue history as dense vectors, which capture semantic similarity but neglect finer linguistic structures such as syntactic dependencies, discourse relations, and coreference links. We propose Semantic Anchoring, a hybrid agentic memory architecture that enriches vector-based storage with explicit linguistic cues to improve recall of nuanced, context-rich exchanges. Our approach combines dependency parsing, discourse relation tagging, and coreference resolution to create structured memory entries. Experiments on adapted long-term dialogue datasets show that semantic anchoring improves factual recall and discourse coherence by up to 18% over strong RAG baselines. We further conduct ablation studies, human evaluations, and error analysis to assess robustness and interpretability.
>
---
#### [new 017] Consensus or Conflict? Fine-Grained Evaluation of Conflicting Answers in Question-Answering
- **分类: cs.CL**

- **简介: 论文研究多答案问答中的冲突识别问题，提出NATCONFQA基准，通过真实冲突标注评估大模型在复杂冲突场景下的表现，揭示其处理冲突的脆弱性。**

- **链接: [http://arxiv.org/pdf/2508.12355v1](http://arxiv.org/pdf/2508.12355v1)**

> **作者:** Eviatar Nachshoni; Arie Cattan; Shmuel Amar; Ori Shapira; Ido Dagan
>
> **备注:** no comments
>
> **摘要:** Large Language Models (LLMs) have demonstrated strong performance in question answering (QA) tasks. However, Multi-Answer Question Answering (MAQA), where a question may have several valid answers, remains challenging. Traditional QA settings often assume consistency across evidences, but MAQA can involve conflicting answers. Constructing datasets that reflect such conflicts is costly and labor-intensive, while existing benchmarks often rely on synthetic data, restrict the task to yes/no questions, or apply unverified automated annotation. To advance research in this area, we extend the conflict-aware MAQA setting to require models not only to identify all valid answers, but also to detect specific conflicting answer pairs, if any. To support this task, we introduce a novel cost-effective methodology for leveraging fact-checking datasets to construct NATCONFQA, a new benchmark for realistic, conflict-aware MAQA, enriched with detailed conflict labels, for all answer pairs. We evaluate eight high-end LLMs on NATCONFQA, revealing their fragility in handling various types of conflicts and the flawed strategies they employ to resolve them.
>
---
#### [new 018] CRED-SQL: Enhancing Real-world Large Scale Database Text-to-SQL Parsing through Cluster Retrieval and Execution Description
- **分类: cs.CL; cs.AI**

- **简介: 论文提出CRED-SQL框架，解决大规模数据库中文本到SQL解析中的语义不匹配问题。通过聚类检索缩小schema范围，并引入执行描述语言（EDL）分步转化，提升准确率与可扩展性。**

- **链接: [http://arxiv.org/pdf/2508.12769v1](http://arxiv.org/pdf/2508.12769v1)**

> **作者:** Shaoming Duan; Zirui Wang; Chuanyi Liu; Zhibin Zhu; Yuhao Zhang; Peiyi Han; Liang Yan; Zewu Penge
>
> **摘要:** Recent advances in large language models (LLMs) have significantly improved the accuracy of Text-to-SQL systems. However, a critical challenge remains: the semantic mismatch between natural language questions (NLQs) and their corresponding SQL queries. This issue is exacerbated in large-scale databases, where semantically similar attributes hinder schema linking and semantic drift during SQL generation, ultimately reducing model accuracy. To address these challenges, we introduce CRED-SQL, a framework designed for large-scale databases that integrates Cluster Retrieval and Execution Description. CRED-SQL first performs cluster-based large-scale schema retrieval to pinpoint the tables and columns most relevant to a given NLQ, alleviating schema mismatch. It then introduces an intermediate natural language representation-Execution Description Language (EDL)-to bridge the gap between NLQs and SQL. This reformulation decomposes the task into two stages: Text-to-EDL and EDL-to-SQL, leveraging LLMs' strong general reasoning capabilities while reducing semantic deviation. Extensive experiments on two large-scale, cross-domain benchmarks-SpiderUnion and BirdUnion-demonstrate that CRED-SQL achieves new state-of-the-art (SOTA) performance, validating its effectiveness and scalability. Our code is available at https://github.com/smduan/CRED-SQL.git
>
---
#### [new 019] In-Context Examples Matter: Improving Emotion Recognition in Conversation with Instruction Tuning
- **分类: cs.CL**

- **简介: 论文提出InitERC，一种单阶段上下文指令微调框架，用于对话情绪识别（ERC）任务。针对现有方法难以统一建模说话者特征与语境关系的问题，通过构造示范池、选择示例、设计提示模板并进行上下文微调，提升模型对说话者-语境-情绪的对齐能力。**

- **链接: [http://arxiv.org/pdf/2508.11889v1](http://arxiv.org/pdf/2508.11889v1)**

> **作者:** Hui Ma; Bo Zhang; Jinpeng Hu; Zenglin Shi
>
> **摘要:** Emotion recognition in conversation (ERC) aims to identify the emotion of each utterance in a conversation, playing a vital role in empathetic artificial intelligence. With the growing of large language models (LLMs), instruction tuning has emerged as a critical paradigm for ERC. Existing studies mainly focus on multi-stage instruction tuning, which first endows LLMs with speaker characteristics, and then conducts context-aware instruction tuning to comprehend emotional states. However, these methods inherently constrains the capacity to jointly capture the dynamic interaction between speaker characteristics and conversational context, resulting in weak alignment among speaker identity, contextual cues, and emotion states within a unified framework. In this paper, we propose InitERC, a simple yet effective one-stage in-context instruction tuning framework for ERC. InitERC adapts LLMs to learn speaker-context-emotion alignment from context examples via in-context instruction tuning. Specifically, InitERC comprises four components, i.e., demonstration pool construction, in-context example selection, prompt template design, and in-context instruction tuning. To explore the impact of in-context examples, we conduct a comprehensive study on three key factors: retrieval strategy, example ordering, and the number of examples. Extensive experiments on three widely used datasets demonstrate that our proposed InitERC achieves substantial improvements over the state-of-the-art baselines.
>
---
#### [new 020] An LLM Agent-Based Complex Semantic Table Annotation Approach
- **分类: cs.CL; cs.DB**

- **简介: 论文针对复杂表格的语义标注任务（STA），解决列类型和单元格实体标注中的语义丢失、同义词等问题。提出基于LLM代理的方法，设计五种工具动态选择策略，并通过Levenshtein距离减少冗余，显著提升效率与准确性。**

- **链接: [http://arxiv.org/pdf/2508.12868v1](http://arxiv.org/pdf/2508.12868v1)**

> **作者:** Yilin Geng; Shujing Wang; Chuan Wang; Keqing He; Yanfei Lv; Ying Wang; Zaiwen Feng; Xiaoying Bai
>
> **摘要:** The Semantic Table Annotation (STA) task, which includes Column Type Annotation (CTA) and Cell Entity Annotation (CEA), maps table contents to ontology entities and plays important roles in various semantic applications. However, complex tables often pose challenges such as semantic loss of column names or cell values, strict ontological hierarchy requirements, homonyms, spelling errors, and abbreviations, which hinder annotation accuracy. To address these issues, this paper proposes an LLM-based agent approach for CTA and CEA. We design and implement five external tools with tailored prompts based on the ReAct framework, enabling the STA agent to dynamically select suitable annotation strategies depending on table characteristics. Experiments are conducted on the Tough Tables and BiodivTab datasets from the SemTab challenge, which contain the aforementioned challenges. Our method outperforms existing approaches across various metrics. Furthermore, by leveraging Levenshtein distance to reduce redundant annotations, we achieve a 70% reduction in time costs and a 60% reduction in LLM token usage, providing an efficient and cost-effective solution for STA.
>
---
#### [new 021] CORE: Measuring Multi-Agent LLM Interaction Quality under Game-Theoretic Pressures
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文提出CORE指标，用于量化多智能体LLM在博弈论情境下的对话质量，解决语言多样性难以衡量的问题。通过分析竞争、合作与中立场景下的词频分布和词汇增长，揭示社会激励对语言适应的影响。**

- **链接: [http://arxiv.org/pdf/2508.11915v1](http://arxiv.org/pdf/2508.11915v1)**

> **作者:** Punya Syon Pandey; Yongjin Yang; Jiarui Liu; Zhijing Jin
>
> **摘要:** Game-theoretic interactions between agents with Large Language Models (LLMs) have revealed many emergent capabilities, yet the linguistic diversity of these interactions has not been sufficiently quantified. In this paper, we present the Conversational Robustness Evaluation Score: CORE, a metric to quantify the effectiveness of language use within multi-agent systems across different game-theoretic interactions. CORE integrates measures of cluster entropy, lexical repetition, and semantic similarity, providing a direct lens of dialog quality. We apply CORE to pairwise LLM dialogs across competitive, cooperative, and neutral settings, further grounding our analysis in Zipf's and Heaps' Laws to characterize word frequency distributions and vocabulary growth. Our findings show that cooperative settings exhibit both steeper Zipf distributions and higher Heap exponents, indicating more repetition alongside greater vocabulary expansion. In contrast, competitive interactions display lower Zipf and Heaps exponents, reflecting less repetition and more constrained vocabularies. These results provide new insights into how social incentives influence language adaptation, and highlight CORE as a robust diagnostic for measuring linguistic robustness in multi-agent LLM systems. Our code is available at https://github.com/psyonp/core.
>
---
#### [new 022] It takes a village to write a book: Mapping anonymous contributions in Stephen Langton's Quaestiones Theologiae
- **分类: cs.CL**

- **简介: 论文通过stylometric方法分析中世纪神学文本《Quaestiones Theologiae》，旨在揭示匿名贡献的编辑层次，解决协作写作识别问题。工作包括构建OCR与转录流水线、对比手动与自动数据效果，验证Transformer模型在经院拉丁语文献中的适用性。**

- **链接: [http://arxiv.org/pdf/2508.12830v1](http://arxiv.org/pdf/2508.12830v1)**

> **作者:** Jan Maliszewski
>
> **摘要:** While the indirect evidence suggests that already in the early scholastic period the literary production based on records of oral teaching (so-called reportationes) was not uncommon, there are very few sources commenting on the practice. This paper details the design of a study applying stylometric techniques of authorship attribution to a collection developed from reportationes -- Stephen Langton's Quaestiones Theologiae -- aiming to uncover layers of editorial work and thus validate some hypotheses regarding the collection's formation. Following Camps, Cl\'erice, and Pinche (2021), I discuss the implementation of an HTR pipeline and stylometric analysis based on the most frequent words, POS tags, and pseudo-affixes. The proposed study will offer two methodological gains relevant to computational research on the scholastic tradition: it will directly compare performance on manually composed and automatically extracted data, and it will test the validity of transformer-based OCR and automated transcription alignment for workflows applied to scholastic Latin corpora. If successful, this study will provide an easily reusable template for the exploratory analysis of collaborative literary production stemming from medieval universities.
>
---
#### [new 023] Is GPT-OSS Good? A Comprehensive Evaluation of OpenAI's Latest Open Source Models
- **分类: cs.CL**

- **简介: 该论文评估OpenAI新发布的GPT-OSS开源模型，解决“稀疏架构是否带来性能提升”问题。通过十项基准测试对比多个开源模型，发现小模型gpt-oss-20B表现优于大模型gpt-oss-120B，表明稀疏架构 Scaling 不一定带来性能增益。**

- **链接: [http://arxiv.org/pdf/2508.12461v1](http://arxiv.org/pdf/2508.12461v1)**

> **作者:** Ziqian Bi; Keyu Chen; Chiung-Yi Tseng; Danyang Zhang; Tianyang Wang; Hongying Luo; Lu Chen; Junming Huang; Jibin Guan; Junfeng Hao; Junhao Song
>
> **摘要:** In August 2025, OpenAI released GPT-OSS models, its first open weight large language models since GPT-2 in 2019, comprising two mixture of experts architectures with 120B and 20B parameters. We evaluated both variants against six contemporary open source large language models ranging from 14.7B to 235B parameters, representing both dense and sparse designs, across ten benchmarks covering general knowledge, mathematical reasoning, code generation, multilingual understanding, and conversational ability. All models were tested in unquantised form under standardised inference settings, with statistical validation using McNemars test and effect size analysis. Results show that gpt-oss-20B consistently outperforms gpt-oss-120B on several benchmarks, such as HumanEval and MMLU, despite requiring substantially less memory and energy per response. Both models demonstrate mid-tier overall performance within the current open source landscape, with relative strength in code generation and notable weaknesses in multilingual tasks. These findings provide empirical evidence that scaling in sparse architectures may not yield proportional performance gains, underscoring the need for further investigation into optimisation strategies and informing more efficient model selection for future open source deployments.
>
---
#### [new 024] AutoBnB-RAG: Enhancing Multi-Agent Incident Response with Retrieval-Augmented Generation
- **分类: cs.CL; cs.CR**

- **简介: 论文提出AutoBnB-RAG框架，将检索增强生成引入多智能体网络事件响应，解决LLM缺乏外部知识导致决策质量低的问题。通过两种检索设置提升协作调查能力，实验证明其能有效提高决策质量和攻击还原准确性。**

- **链接: [http://arxiv.org/pdf/2508.13118v1](http://arxiv.org/pdf/2508.13118v1)**

> **作者:** Zefang Liu; Arman Anwar
>
> **摘要:** Incident response (IR) requires fast, coordinated, and well-informed decision-making to contain and mitigate cyber threats. While large language models (LLMs) have shown promise as autonomous agents in simulated IR settings, their reasoning is often limited by a lack of access to external knowledge. In this work, we present AutoBnB-RAG, an extension of the AutoBnB framework that incorporates retrieval-augmented generation (RAG) into multi-agent incident response simulations. Built on the Backdoors & Breaches (B&B) tabletop game environment, AutoBnB-RAG enables agents to issue retrieval queries and incorporate external evidence during collaborative investigations. We introduce two retrieval settings: one grounded in curated technical documentation (RAG-Wiki), and another using narrative-style incident reports (RAG-News). We evaluate performance across eight team structures, including newly introduced argumentative configurations designed to promote critical reasoning. To validate practical utility, we also simulate real-world cyber incidents based on public breach reports, demonstrating AutoBnB-RAG's ability to reconstruct complex multi-stage attacks. Our results show that retrieval augmentation improves decision quality and success rates across diverse organizational models. This work demonstrates the value of integrating retrieval mechanisms into LLM-based multi-agent systems for cybersecurity decision-making.
>
---
#### [new 025] Learning Wisdom from Errors: Promoting LLM's Continual Relation Learning through Exploiting Error Cases
- **分类: cs.CL**

- **简介: 论文研究持续关系抽取任务，针对模型遗忘旧关系的问题，提出基于错误案例的对比微调方法。通过分离正确与错误样本，利用指令微调修正认知偏差，提升模型对新旧关系的学习能力。**

- **链接: [http://arxiv.org/pdf/2508.12031v1](http://arxiv.org/pdf/2508.12031v1)**

> **作者:** Shaozhe Yin; Jinyu Guo; Kai Shuang; Xia Liu; Ruize Ou
>
> **摘要:** Continual Relation Extraction (CRE) aims to continually learn new emerging relations while avoiding catastrophic forgetting. Existing CRE methods mainly use memory replay and contrastive learning to mitigate catastrophic forgetting. However, these methods do not attach importance to the error cases that can reveal the model's cognitive biases more effectively. To address this issue, we propose an instruction-based continual contrastive tuning approach for Large Language Models (LLMs) in CRE. Different from existing CRE methods that typically handle the training and memory data in a unified manner, this approach splits the training and memory data of each task into two parts respectively based on the correctness of the initial responses and treats them differently through dual-task fine-tuning. In addition, leveraging the advantages of LLM's instruction-following ability, we propose a novel instruction-based contrastive tuning strategy for LLM to continuously correct current cognitive biases with the guidance of previous data in an instruction-tuning manner, which mitigates the gap between old and new relations in a more suitable way for LLMs. We experimentally evaluate our model on TACRED and FewRel, and the results show that our model achieves new state-of-the-art CRE performance with significant improvements, demonstrating the importance of specializing in exploiting error cases.
>
---
#### [new 026] A Multi-Task Evaluation of LLMs' Processing of Academic Text Input
- **分类: cs.CL; econ.GN; q-fin.EC**

- **简介: 该论文属于多任务评估任务，旨在解决LLMs在学术文本处理能力上的实际应用问题。作者设计四类任务（复现、比较、评分、反思），测试Gemini等模型表现，发现其在学术文本处理上效果有限，不建议未经审核用于同行评审。**

- **链接: [http://arxiv.org/pdf/2508.11779v1](http://arxiv.org/pdf/2508.11779v1)**

> **作者:** Tianyi Li; Yu Qin; Olivia R. Liu Sheng
>
> **摘要:** How much large language models (LLMs) can aid scientific discovery, notably in assisting academic peer review, is in heated debate. Between a literature digest and a human-comparable research assistant lies their practical application potential. We organize individual tasks that computer science studies employ in separate terms into a guided and robust workflow to evaluate LLMs' processing of academic text input. We employ four tasks in the assessment: content reproduction/comparison/scoring/reflection, each demanding a specific role of the LLM (oracle/judgmental arbiter/knowledgeable arbiter/collaborator) in assisting scholarly works, and altogether testing LLMs with questions that increasingly require intellectual capabilities towards a solid understanding of scientific texts to yield desirable solutions. We exemplify a rigorous performance evaluation with detailed instructions on the prompts. Adopting first-rate Information Systems articles at three top journals as the input texts and an abundant set of text metrics, we record a compromised performance of the leading LLM - Google's Gemini: its summary and paraphrase of academic text is acceptably reliable; using it to rank texts through pairwise text comparison is faintly scalable; asking it to grade academic texts is prone to poor discrimination; its qualitative reflection on the text is self-consistent yet hardly insightful to inspire meaningful research. This evidence against an endorsement of LLMs' text-processing capabilities is consistent across metric-based internal (linguistic assessment), external (comparing to the ground truth), and human evaluation, and is robust to the variations of the prompt. Overall, we do not recommend an unchecked use of LLMs in constructing peer reviews.
>
---
#### [new 027] HeteroRAG: A Heterogeneous Retrieval-Augmented Generation Framework for Medical Vision Language Tasks
- **分类: cs.CL**

- **简介: 该论文针对医疗视觉语言模型事实性差、可靠性低的问题，提出HeteroRAG框架，通过异构知识源检索与多源知识对齐训练，提升模型准确性与可信度。**

- **链接: [http://arxiv.org/pdf/2508.12778v1](http://arxiv.org/pdf/2508.12778v1)**

> **作者:** Zhe Chen; Yusheng Liao; Shuyang Jiang; Zhiyuan Zhu; Haolin Li; Yanfeng Wang; Yu Wang
>
> **摘要:** Medical large vision-language Models (Med-LVLMs) have shown promise in clinical applications but suffer from factual inaccuracies and unreliable outputs, posing risks in real-world diagnostics. While retrieval-augmented generation has emerged as a potential solution, current medical multimodal RAG systems are unable to perform effective retrieval across heterogeneous sources. The irrelevance of retrieved reports affects the factuality of analysis, while insufficient knowledge affects the credibility of clinical decision-making. To bridge the gap, we construct MedAtlas, which includes extensive multimodal report repositories and diverse text corpora. Based on it, we present HeteroRAG, a novel framework that enhances Med-LVLMs through heterogeneous knowledge sources. The framework introduces Modality-specific CLIPs for effective report retrieval and a Multi-corpora Query Generator for dynamically constructing queries for diverse corpora. Incorporating knowledge from such multifaceted sources, Med-LVLM is then trained with Heterogeneous Knowledge Preference Tuning to achieve cross-modality and multi-source knowledge alignment. Extensive experiments across 12 datasets and 3 modalities demonstrate that the proposed HeteroRAG achieves state-of-the-art performance in most medical vision language benchmarks, significantly improving factual accuracy and reliability of Med-LVLMs.
>
---
#### [new 028] Context Matters: Incorporating Target Awareness in Conversational Abusive Language Detection
- **分类: cs.CL; cs.AI**

- **简介: 论文研究对话中滥用语言检测任务，解决仅依赖回复文本忽略上下文的问题。通过引入父帖内容和账户特征，实验表明内容特征更能提升模型性能，强调上下文的重要性。**

- **链接: [http://arxiv.org/pdf/2508.12828v1](http://arxiv.org/pdf/2508.12828v1)**

> **作者:** Raneem Alharthi; Rajwa Alharthi; Aiqi Jiang; Arkaitz Zubiaga
>
> **摘要:** Abusive language detection has become an increasingly important task as a means to tackle this type of harmful content in social media. There has been a substantial body of research developing models for determining if a social media post is abusive or not; however, this research has primarily focused on exploiting social media posts individually, overlooking additional context that can be derived from surrounding posts. In this study, we look at conversational exchanges, where a user replies to an earlier post by another user (the parent tweet). We ask: does leveraging context from the parent tweet help determine if a reply post is abusive or not, and what are the features that contribute the most? We study a range of content-based and account-based features derived from the context, and compare this to the more widely studied approach of only looking at the features from the reply tweet. For a more generalizable study, we test four different classification models on a dataset made of conversational exchanges (parent-reply tweet pairs) with replies labeled as abusive or not. Our experiments show that incorporating contextual features leads to substantial improvements compared to the use of features derived from the reply tweet only, confirming the importance of leveraging context. We observe that, among the features under study, it is especially the content-based features (what is being posted) that contribute to the classification performance rather than account-based features (who is posting it). While using content-based features, it is best to combine a range of different features to ensure improved performance over being more selective and using fewer features. Our study provides insights into the development of contextualized abusive language detection models in realistic settings involving conversations.
>
---
#### [new 029] STEM: Efficient Relative Capability Evaluation of LLMs through Structured Transition Samples
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文提出STEM方法，用于高效评估大语言模型的相对能力。针对现有评估方式成本高、效果差的问题，通过分析同架构不同规模模型的性能变化，识别关键样本，实现轻量、可解释的细粒度能力排序。**

- **链接: [http://arxiv.org/pdf/2508.12096v1](http://arxiv.org/pdf/2508.12096v1)**

> **作者:** Haiquan Hu; Jiazhi Jiang; Shiyou Xu; Ruhan Zeng; Tian Wang
>
> **备注:** Submit to AAAI 2026
>
> **摘要:** Evaluating large language models (LLMs) has become increasingly challenging as model capabilities advance rapidly. While recent models often achieve higher scores on standard benchmarks, these improvements do not consistently reflect enhanced real-world reasoning capabilities. Moreover, widespread overfitting to public benchmarks and the high computational cost of full evaluations have made it both expensive and less effective to distinguish meaningful differences between models. To address these challenges, we propose the \textbf{S}tructured \textbf{T}ransition \textbf{E}valuation \textbf{M}ethod (STEM), a lightweight and interpretable evaluation framework for efficiently estimating the relative capabilities of LLMs. STEM identifies \textit{significant transition samples} (STS) by analyzing consistent performance transitions among LLMs of the same architecture but varying parameter scales. These samples enable STEM to effectively estimate the capability position of an unknown model. Qwen3 model family is applied to construct the STS pool on six diverse and representative benchmarks. To assess generalizability. Experimental results indicate that STEM reliably captures performance trends, aligns with ground-truth rankings of model capability. These findings highlight STEM as a practical and scalable method for fine-grained, architecture-agnostic evaluation of LLMs.
>
---
#### [new 030] Improving Detection of Watermarked Language Models
- **分类: cs.CL; cs.LG; stat.ML**

- **简介: 该论文属于语言模型水印检测任务，旨在提升水印检测效果。针对后训练模型熵低导致检测困难的问题，提出结合水印与非水印检测器的混合方案，实验证明其在多种条件下均优于单一检测方法。**

- **链接: [http://arxiv.org/pdf/2508.13131v1](http://arxiv.org/pdf/2508.13131v1)**

> **作者:** Dara Bahri; John Wieting
>
> **摘要:** Watermarking has recently emerged as an effective strategy for detecting the generations of large language models (LLMs). The strength of a watermark typically depends strongly on the entropy afforded by the language model and the set of input prompts. However, entropy can be quite limited in practice, especially for models that are post-trained, for example via instruction tuning or reinforcement learning from human feedback (RLHF), which makes detection based on watermarking alone challenging. In this work, we investigate whether detection can be improved by combining watermark detectors with non-watermark ones. We explore a number of hybrid schemes that combine the two, observing performance gains over either class of detector under a wide range of experimental conditions.
>
---
#### [new 031] Leveraging Large Language Models for Predictive Analysis of Human Misery
- **分类: cs.CL; cs.CY**

- **简介: 论文研究用大语言模型预测人类感知的痛苦分数，将问题建模为回归任务。通过多种提示策略对比，发现少样本方法优于零样本；并提出“痛苦游戏秀”框架，以游戏化方式评估模型在反馈下的动态情感推理能力。**

- **链接: [http://arxiv.org/pdf/2508.12669v1](http://arxiv.org/pdf/2508.12669v1)**

> **作者:** Bishanka Seal; Rahul Seetharaman; Aman Bansal; Abhilash Nandy
>
> **备注:** 14 pages, 4 tables
>
> **摘要:** This study investigates the use of Large Language Models (LLMs) for predicting human-perceived misery scores from natural language descriptions of real-world scenarios. The task is framed as a regression problem, where the model assigns a scalar value from 0 to 100 to each input statement. We evaluate multiple prompting strategies, including zero-shot, fixed-context few-shot, and retrieval-based prompting using BERT sentence embeddings. Few-shot approaches consistently outperform zero-shot baselines, underscoring the value of contextual examples in affective prediction. To move beyond static evaluation, we introduce the "Misery Game Show", a novel gamified framework inspired by a television format. It tests LLMs through structured rounds involving ordinal comparison, binary classification, scalar estimation, and feedback-driven reasoning. This setup enables us to assess not only predictive accuracy but also the model's ability to adapt based on corrective feedback. The gamified evaluation highlights the broader potential of LLMs in dynamic emotional reasoning tasks beyond standard regression. Code and data link: https://github.com/abhi1nandy2/Misery_Data_Exps_GitHub
>
---
#### [new 032] ZigzagAttention: Efficient Long-Context Inference with Exclusive Retrieval and Streaming Heads
- **分类: cs.CL**

- **简介: 论文提出ZigzagAttention，解决长文本推理中KV缓存占用高、延迟大的问题。通过区分检索头与流式头，并确保每层仅使用一种头，减少内存开销和计算延迟，同时保持性能稳定。**

- **链接: [http://arxiv.org/pdf/2508.12407v1](http://arxiv.org/pdf/2508.12407v1)**

> **作者:** Zhuorui Liu; Chen Zhang; Dawei Song
>
> **备注:** 5 pages, 4 figures
>
> **摘要:** With the rapid development of large language models (LLMs), handling long context has become one of the vital abilities in LLMs. Such long-context ability is accompanied by difficulties in deployment, especially due to the increased consumption of KV cache. There is certain work aiming to optimize the memory footprint of KV cache, inspired by the observation that attention heads can be categorized into retrieval heads that are of great significance and streaming heads that are of less significance. Typically, identifying the streaming heads and and waiving the KV cache in the streaming heads would largely reduce the overhead without hurting the performance that much. However, since employing both retrieval and streaming heads in one layer decomposes one large round of attention computation into two small ones, it may unexpectedly bring extra latency on accessing and indexing tensors. Based on this intuition, we impose an important improvement to the identification process of retrieval and streaming heads, in which we design a criterion that enforces exclusively retrieval or streaming heads gathered in one unique layer. In this way, we further eliminate the extra latency and only incur negligible performance degradation. Our method named \textsc{ZigzagAttention} is competitive among considered baselines owing to reduced latency and comparable performance.
>
---
#### [new 033] Doğal Dil İşlemede Tokenizasyon Standartları ve Ölçümü: Türkçe Üzerinden Büyük Dil Modellerinin Karşılaştırmalı Analizi
- **分类: cs.CL; 68T50; I.2.7; I.2.6**

- **简介: 该论文研究自然语言处理中的分词标准与评估，针对土耳其语等形态丰富、资源稀缺语言，提出新指标并基于TR-MMLU数据集对比不同分词器性能，发现语言特异性分词比例对下游任务影响更大。**

- **链接: [http://arxiv.org/pdf/2508.13058v1](http://arxiv.org/pdf/2508.13058v1)**

> **作者:** M. Ali Bayram; Ali Arda Fincan; Ahmet Semih Gümüş; Sercan Karakaş; Banu Diri; Savaş Yıldırım
>
> **备注:** in Turkish language, Presented at the 2025 33rd Signal Processing and Communications Applications Conference (SIU), 25--28 June 2025, \c{S}ile, Istanbul, T\"urkiye
>
> **摘要:** Tokenization is a fundamental preprocessing step in Natural Language Processing (NLP), significantly impacting the capability of large language models (LLMs) to capture linguistic and semantic nuances. This study introduces a novel evaluation framework addressing tokenization challenges specific to morphologically-rich and low-resource languages such as Turkish. Utilizing the Turkish MMLU (TR-MMLU) dataset, comprising 6,200 multiple-choice questions from the Turkish education system, we assessed tokenizers based on vocabulary size, token count, processing time, language-specific token percentages (\%TR), and token purity (\%Pure). These newly proposed metrics measure how effectively tokenizers preserve linguistic structures. Our analysis reveals that language-specific token percentages exhibit a stronger correlation with downstream performance (e.g., MMLU scores) than token purity. Furthermore, increasing model parameters alone does not necessarily enhance linguistic performance, underscoring the importance of tailored, language-specific tokenization methods. The proposed framework establishes robust and practical tokenization standards for morphologically complex languages.
>
---
#### [new 034] What do Speech Foundation Models Learn? Analysis and Applications
- **分类: cs.CL; eess.AS**

- **简介: 论文研究语音基础模型（SFMs）学习到的声学与语言知识，通过轻量分析框架和无训练任务揭示其内部表征。解决SFMs理解不足及在口语理解（SLU）任务中应用有限的问题，提出命名实体识别（NER）和定位（NEL）新任务与数据集，验证端到端模型优于传统级联方法。**

- **链接: [http://arxiv.org/pdf/2508.12255v1](http://arxiv.org/pdf/2508.12255v1)**

> **作者:** Ankita Pasad
>
> **备注:** Ph.D. Thesis
>
> **摘要:** Speech foundation models (SFMs) are designed to serve as general-purpose representations for a wide range of speech-processing tasks. The last five years have seen an influx of increasingly successful self-supervised and supervised pre-trained models with impressive performance on various downstream tasks. Although the zoo of SFMs continues to grow, our understanding of the knowledge they acquire lags behind. This thesis presents a lightweight analysis framework using statistical tools and training-free tasks to investigate the acoustic and linguistic knowledge encoded in SFM layers. We conduct a comparative study across multiple SFMs and statistical tools. Our study also shows that the analytical insights have concrete implications for downstream task performance. The effectiveness of an SFM is ultimately determined by its performance on speech applications. Yet it remains unclear whether the benefits extend to spoken language understanding (SLU) tasks that require a deeper understanding than widely studied ones, such as speech recognition. The limited exploration of SLU is primarily due to a lack of relevant datasets. To alleviate that, this thesis contributes tasks, specifically spoken named entity recognition (NER) and named entity localization (NEL), to the Spoken Language Understanding Evaluation benchmark. We develop SFM-based approaches for NER and NEL, and find that end-to-end (E2E) models leveraging SFMs can surpass traditional cascaded (speech recognition followed by a text model) approaches. Further, we evaluate E2E SLU models across SFMs and adaptation strategies to assess the impact on task performance. Collectively, this thesis tackles previously unanswered questions about SFMs, providing tools and datasets to further our understanding and to enable the community to make informed design choices for future model development and adoption.
>
---
#### [new 035] Deep Language Geometry: Constructing a Metric Space from LLM Weights
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文提出基于LLM权重激活构建语言度量空间的新方法，通过剪枝算法计算权重重要性得分，自动获得高维语言向量。该任务旨在捕捉语言内在特征，解决传统依赖人工特征的局限性。实验覆盖106种语言，验证了方法的有效性和发现新语言关联的能力。**

- **链接: [http://arxiv.org/pdf/2508.11676v1](http://arxiv.org/pdf/2508.11676v1)**

> **作者:** Maksym Shamrai; Vladyslav Hamolia
>
> **备注:** 18 pages, accepted to RANLP 2025
>
> **摘要:** We introduce a novel framework that utilizes the internal weight activations of modern Large Language Models (LLMs) to construct a metric space of languages. Unlike traditional approaches based on hand-crafted linguistic features, our method automatically derives high-dimensional vector representations by computing weight importance scores via an adapted pruning algorithm. Our approach captures intrinsic language characteristics that reflect linguistic phenomena. We validate our approach across diverse datasets and multilingual LLMs, covering 106 languages. The results align well with established linguistic families while also revealing unexpected inter-language connections that may indicate historical contact or language evolution. The source code, computed language latent vectors, and visualization tool are made publicly available at https://github.com/mshamrai/deep-language-geometry.
>
---
#### [new 036] ToolACE-MT: Non-Autoregressive Generation for Agentic Multi-Turn Interaction
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文提出ToolACE-MT框架，解决多轮代理交互中数据生成效率低的问题。通过非自回归迭代生成方法，在初始化、精炼和验证三阶段构建高质量对话数据，提升工具增强型大模型任务的训练效果。**

- **链接: [http://arxiv.org/pdf/2508.12685v1](http://arxiv.org/pdf/2508.12685v1)**

> **作者:** Xingshan Zeng; Weiwen Liu; Lingzhi Wang; Liangyou Li; Fei Mi; Yasheng Wang; Lifeng Shang; Xin Jiang; Qun Liu
>
> **摘要:** Agentic task-solving with Large Language Models (LLMs) requires multi-turn, multi-step interactions, often involving complex function calls and dynamic user-agent exchanges. Existing simulation-based data generation methods for such scenarios rely heavily on costly autoregressive interactions between multiple LLM agents, thereby limiting real-world performance of agentic tasks. In this paper, we propose a novel Non-Autoregressive Iterative Generation framework, called ToolACE-MT, for constructing high-quality multi-turn agentic dialogues. ToolACE-MT generates full conversational trajectories through three stages: coarse-grained initialization, iterative refinement, and offline verification. The initialization phase builds a structurally complete yet semantically coarse dialogue skeleton; the iterative refinement phase introduces realistic complexities and continued refinement via mask-and-fill operations; and the offline verification phase ensures correctness and coherence via rule- and model-based checks. Experiments demonstrate that ToolACE-MT enables efficient, effective and generalizable agentic data generation, offering a new paradigm for high-quality data construction in tool-augmented LLM scenarios.
>
---
#### [new 037] Analyzing Information Sharing and Coordination in Multi-Agent Planning
- **分类: cs.CL**

- **简介: 论文研究多智能体系统在长程规划任务中的信息共享与协调问题，针对LLM代理在复杂约束下易出错的问题，提出笔记和调度器机制。实验表明，两者结合使旅行规划任务通过率提升至25%，显著优于单智能体基线。**

- **链接: [http://arxiv.org/pdf/2508.12981v1](http://arxiv.org/pdf/2508.12981v1)**

> **作者:** Tianyue Ou; Saujas Vaduguru; Daniel Fried
>
> **摘要:** Multi-agent systems (MASs) have pushed the boundaries of large language model (LLM) agents in domains such as web research and software engineering. However, long-horizon, multi-constraint planning tasks involve conditioning on detailed information and satisfying complex interdependent constraints, which can pose a challenge for these systems. In this study, we construct an LLM-based MAS for a travel planning task which is representative of these challenges. We evaluate the impact of a notebook to facilitate information sharing, and evaluate an orchestrator agent to improve coordination in free form conversation between agents. We find that the notebook reduces errors due to hallucinated details by 18%, while an orchestrator directs the MAS to focus on and further reduce errors by up to 13.5% within focused sub-areas. Combining both mechanisms achieves a 25% final pass rate on the TravelPlanner benchmark, a 17.5% absolute improvement over the single-agent baseline's 7.5% pass rate. These results highlight the potential of structured information sharing and reflective orchestration as key components in MASs for long horizon planning with LLMs.
>
---
#### [new 038] Can we Evaluate RAGs with Synthetic Data?
- **分类: cs.CL; cs.AI**

- **简介: 论文研究用LLM生成的合成数据能否有效评估RAG系统。结果表明，合成数据能可靠比较不同检索器配置，但无法一致区分不同生成器架构，因任务不匹配和风格偏差导致。**

- **链接: [http://arxiv.org/pdf/2508.11758v1](http://arxiv.org/pdf/2508.11758v1)**

> **作者:** Jonas van Elburg; Peter van der Putten; Maarten Marx
>
> **备注:** Accepted for the SynDAiTE workshop at the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML-PKDD 2025), September 15, 2025 - Porto, Portugal
>
> **摘要:** We investigate whether synthetic question-answer (QA) data generated by large language models (LLMs) can serve as an effective proxy for human-labeled benchmarks when such data is unavailable. We assess the reliability of synthetic benchmarks across two experiments: one varying retriever parameters while keeping the generator fixed, and another varying the generator with fixed retriever parameters. Across four datasets, of which two open-domain and two proprietary, we find that synthetic benchmarks reliably rank the RAGs varying in terms of retriever configuration, aligning well with human-labeled benchmark baselines. However, they fail to produce consistent RAG rankings when comparing generator architectures. The breakdown possibly arises from a combination of task mismatch between the synthetic and human benchmarks, and stylistic bias favoring certain generators.
>
---
#### [new 039] Breaking Language Barriers: Equitable Performance in Multilingual Language Models
- **分类: cs.CL; cs.AI**

- **简介: 论文研究多语言大模型在低资源语言上的性能不平等问题，提出通过合成代码混杂文本微调模型来提升低资源语言的常识推理能力，同时保持高资源语言性能。**

- **链接: [http://arxiv.org/pdf/2508.12662v1](http://arxiv.org/pdf/2508.12662v1)**

> **作者:** Tanay Nagar; Grigorii Khvatskii; Anna Sokol; Nitesh V. Chawla
>
> **备注:** Accepted as a non-archival work-in-progress paper at the NAACL 2025 Student Research Workshop
>
> **摘要:** Cutting-edge LLMs have emerged as powerful tools for multilingual communication and understanding. However, LLMs perform worse in Common Sense Reasoning (CSR) tasks when prompted in low-resource languages (LRLs) like Hindi or Swahili compared to high-resource languages (HRLs) like English. Equalizing this inconsistent access to quality LLM outputs is crucial to ensure fairness for speakers of LRLs and across diverse linguistic communities. In this paper, we propose an approach to bridge this gap in LLM performance. Our approach involves fine-tuning an LLM on synthetic code-switched text generated using controlled language-mixing methods. We empirically demonstrate that fine-tuning LLMs on synthetic code-switched datasets leads to substantial improvements in LRL model performance while preserving or enhancing performance in HRLs. Additionally, we present a new dataset of synthetic code-switched text derived from the CommonSenseQA dataset, featuring three distinct language ratio configurations.
>
---
#### [new 040] SupraTok: Cross-Boundary Tokenization for Enhanced Language Model Performance
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文提出SupraTok，一种新型分词架构，通过跨边界模式学习、熵驱动数据筛选和多阶段课程学习提升语言模型性能。解决静态分词策略限制问题，实现更高效的token化，在38语言上保持竞争力，并在GPT-2模型上显著提升下游任务表现。**

- **链接: [http://arxiv.org/pdf/2508.11857v1](http://arxiv.org/pdf/2508.11857v1)**

> **作者:** Andrei-Valentin Tănase; Elena Pelican
>
> **摘要:** Tokenization remains a fundamental yet underexplored bottleneck in natural language processing, with strategies largely static despite remarkable progress in model architectures. We present SupraTok, a novel tokenization architecture that reimagines subword segmentation through three innovations: cross-boundary pattern learning that discovers multi-word semantic units, entropy-driven data curation that optimizes training corpus quality, and multi-phase curriculum learning for stable convergence. Our approach extends Byte-Pair Encoding by learning "superword" tokens, coherent multi-word expressions that preserve semantic unity while maximizing compression efficiency. SupraTok achieves 31% improvement in English tokenization efficiency (5.91 versus 4.51 characters per token) compared to OpenAI's o200k tokenizer and 30% improvement over Google's Gemma 3 tokenizer (256k vocabulary), while maintaining competitive performance across 38 languages. When integrated with a GPT-2 scale model (124M parameters) trained on 10 billion tokens from the FineWeb-Edu dataset, SupraTok yields 8.4% improvement on HellaSWAG and 9.5% on MMLU benchmarks without architectural modifications. While these results are promising at this scale, further validation at larger model scales is needed. These findings suggest that efficient tokenization can complement architectural innovations as a path to improved language model performance.
>
---
#### [new 041] LLMs Struggle with NLI for Perfect Aspect: A Cross-Linguistic Study in Chinese and Japanese
- **分类: cs.CL**

- **简介: 该论文研究跨语言自然语言推理（NLI）任务，聚焦中文和日语中缺乏显式时态标记的完成体语法现象。作者构建了1350对句子的数据集，发现先进大模型在检测时态与参考时间细微变化上表现不佳，揭示了模型在时态语义理解上的局限性。**

- **链接: [http://arxiv.org/pdf/2508.11927v1](http://arxiv.org/pdf/2508.11927v1)**

> **作者:** Jie Lu; Du Jin; Hitomi Yanaka
>
> **备注:** 9 pages, 3 figures
>
> **摘要:** Unlike English, which uses distinct forms (e.g., had, has, will have) to mark the perfect aspect across tenses, Chinese and Japanese lack separate grammatical forms for tense within the perfect aspect, which complicates Natural Language Inference (NLI). Focusing on the perfect aspect in these languages, we construct a linguistically motivated, template-based NLI dataset (1,350 pairs per language). Experiments reveal that even advanced LLMs struggle with temporal inference, particularly in detecting subtle tense and reference-time shifts. These findings highlight model limitations and underscore the need for cross-linguistic evaluation in temporal semantics. Our dataset is available at https://github.com/Lujie2001/CrossNLI.
>
---
#### [new 042] CorrSteer: Steering Improves Task Performance and Safety in LLMs through Correlation-based Sparse Autoencoder Feature Selection
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文提出CorrSteer方法，通过相关性选择稀疏自编码器特征，实现无需对比数据的自动化语言模型引导。解决现有方法依赖大量激活存储和标注数据的问题，在问答、偏见缓解等任务中提升性能，显著改善模型安全性和任务表现。**

- **链接: [http://arxiv.org/pdf/2508.12535v1](http://arxiv.org/pdf/2508.12535v1)**

> **作者:** Seonglae Cho; Zekun Wu; Adriano Koshiyama
>
> **备注:** 42 pages, 9 tables
>
> **摘要:** Sparse Autoencoders (SAEs) can extract interpretable features from large language models (LLMs) without supervision. However, their effectiveness in downstream steering tasks is limited by the requirement for contrastive datasets or large activation storage. To address these limitations, we propose CorrSteer, which selects features by correlating sample correctness with SAE activations from generated tokens at inference time. This approach uses only inference-time activations to extract more relevant features, thereby avoiding spurious correlations. It also obtains steering coefficients from average activations, automating the entire pipeline. Our method shows improved task performance on QA, bias mitigation, jailbreaking prevention, and reasoning benchmarks on Gemma 2 2B and LLaMA 3.1 8B, notably achieving a +4.1% improvement in MMLU performance and a +22.9% improvement in HarmBench with only 4000 samples. Selected features demonstrate semantically meaningful patterns aligned with each task's requirements, revealing the underlying capabilities that drive performance. Our work establishes correlationbased selection as an effective and scalable approach for automated SAE steering across language model applications.
>
---
#### [new 043] Büyük Dil Modelleri için TR-MMLU Benchmarkı: Performans Değerlendirmesi, Zorluklar ve İyileştirme Fırsatları
- **分类: cs.CL; 68T50; I.2.7; I.2.6**

- **简介: 该论文提出TR-MMLU基准，用于评估大语言模型在土耳其语中的表现。针对土耳其语资源有限的问题，构建了62个教育领域的6200道多项选择题数据集，为土耳其自然语言处理研究提供标准评测框架，并揭示模型改进方向。**

- **链接: [http://arxiv.org/pdf/2508.13044v1](http://arxiv.org/pdf/2508.13044v1)**

> **作者:** M. Ali Bayram; Ali Arda Fincan; Ahmet Semih Gümüş; Banu Diri; Savaş Yıldırım; Öner Aytaş
>
> **备注:** 10 pages, in Turkish language, 5 figures. Presented at the 2025 33rd Signal Processing and Communications Applications Conference (SIU), 25--28 June 2025, Sile, Istanbul, T\"urkiye
>
> **摘要:** Language models have made significant advancements in understanding and generating human language, achieving remarkable success in various applications. However, evaluating these models remains a challenge, particularly for resource-limited languages like Turkish. To address this issue, we introduce the Turkish MMLU (TR-MMLU) benchmark, a comprehensive evaluation framework designed to assess the linguistic and conceptual capabilities of large language models (LLMs) in Turkish. TR-MMLU is based on a meticulously curated dataset comprising 6,200 multiple-choice questions across 62 sections within the Turkish education system. This benchmark provides a standard framework for Turkish NLP research, enabling detailed analyses of LLMs' capabilities in processing Turkish text. In this study, we evaluated state-of-the-art LLMs on TR-MMLU, highlighting areas for improvement in model design. TR-MMLU sets a new standard for advancing Turkish NLP research and inspiring future innovations.
>
---
#### [new 044] Exploring Efficiency Frontiers of Thinking Budget in Medical Reasoning: Scaling Laws between Computational Resources and Reasoning Quality
- **分类: cs.CL**

- **简介: 该论文研究医疗推理中思维预算机制，解决计算资源与推理质量关系问题。通过实验揭示对数 scaling 规律，提出三类效率区间，并发现小模型受益更大，适用于动态优化医疗AI系统。**

- **链接: [http://arxiv.org/pdf/2508.12140v1](http://arxiv.org/pdf/2508.12140v1)**

> **作者:** Ziqian Bi; Lu Chen; Junhao Song; Hongying Luo; Enze Ge; Junmin Huang; Tianyang Wang; Keyu Chen; Chia Xin Liang; Zihan Wei; Huafeng Liu; Chunjie Tian; Jibin Guan; Joe Yeong; Yongzhi Xu; Peng Wang; Junfeng Hao
>
> **摘要:** This study presents the first comprehensive evaluation of thinking budget mechanisms in medical reasoning tasks, revealing fundamental scaling laws between computational resources and reasoning quality. We systematically evaluated two major model families, Qwen3 (1.7B to 235B parameters) and DeepSeek-R1 (1.5B to 70B parameters), across 15 medical datasets spanning diverse specialties and difficulty levels. Through controlled experiments with thinking budgets ranging from zero to unlimited tokens, we establish logarithmic scaling relationships where accuracy improvements follow a predictable pattern with both thinking budget and model size. Our findings identify three distinct efficiency regimes: high-efficiency (0 to 256 tokens) suitable for real-time applications, balanced (256 to 512 tokens) offering optimal cost-performance tradeoffs for routine clinical support, and high-accuracy (above 512 tokens) justified only for critical diagnostic tasks. Notably, smaller models demonstrate disproportionately larger benefits from extended thinking, with 15 to 20% improvements compared to 5 to 10% for larger models, suggesting a complementary relationship where thinking budget provides greater relative benefits for capacity-constrained models. Domain-specific patterns emerge clearly, with neurology and gastroenterology requiring significantly deeper reasoning processes than cardiovascular or respiratory medicine. The consistency between Qwen3 native thinking budget API and our proposed truncation method for DeepSeek-R1 validates the generalizability of thinking budget concepts across architectures. These results establish thinking budget control as a critical mechanism for optimizing medical AI systems, enabling dynamic resource allocation aligned with clinical needs while maintaining the transparency essential for healthcare deployment.
>
---
#### [new 045] Fast, Slow, and Tool-augmented Thinking for LLMs: A Review
- **分类: cs.CL**

- **简介: 论文提出双维度分类法，将LLM推理策略分为快慢和内外两类，系统梳理自适应推理方法，旨在提升模型在真实任务中的效率与可靠性。**

- **链接: [http://arxiv.org/pdf/2508.12265v1](http://arxiv.org/pdf/2508.12265v1)**

> **作者:** Xinda Jia; Jinpeng Li; Zezhong Wang; Jingjing Li; Xingshan Zeng; Yasheng Wang; Weinan Zhang; Yong Yu; Weiwen Liu
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable progress in reasoning across diverse domains. However, effective reasoning in real-world tasks requires adapting the reasoning strategy to the demands of the problem, ranging from fast, intuitive responses to deliberate, step-by-step reasoning and tool-augmented thinking. Drawing inspiration from cognitive psychology, we propose a novel taxonomy of LLM reasoning strategies along two knowledge boundaries: a fast/slow boundary separating intuitive from deliberative processes, and an internal/external boundary distinguishing reasoning grounded in the model's parameters from reasoning augmented by external tools. We systematically survey recent work on adaptive reasoning in LLMs and categorize methods based on key decision factors. We conclude by highlighting open challenges and future directions toward more adaptive, efficient, and reliable LLMs.
>
---
#### [new 046] All for law and law for all: Adaptive RAG Pipeline for Legal Research
- **分类: cs.CL; cs.IR; F.2.2, H.3.3, I.2.7**

- **简介: 论文提出一个自适应RAG流水线，用于法律研究任务，解决法律领域生成幻觉问题。通过查询翻译、开源检索策略和评估框架提升检索质量和答案忠实度，实现高效、可复现的法律问答系统。**

- **链接: [http://arxiv.org/pdf/2508.13107v1](http://arxiv.org/pdf/2508.13107v1)**

> **作者:** Figarri Keisha; Prince Singh; Pallavi; Dion Fernandes; Aravindh Manivannan; Ilham Wicaksono; Faisal Ahmad
>
> **备注:** submitted to NLLP 2025 Workshop
>
> **摘要:** Retrieval-Augmented Generation (RAG) mitigates hallucinations by grounding large language model outputs in cited sources, a capability that is especially critical in the legal domain. We present an end-to-end RAG pipeline that revisits and extends the LegalBenchRAG baseline with three targeted enhancements: (i) a context-aware query translator that disentangles document references from natural-language questions and adapts retrieval depth and response style based on expertise and specificity, (ii) open-source retrieval strategies using SBERT and GTE embeddings that achieve substantial performance gains (improving Recall@K by 30-95\% and Precision@K by $\sim$2.5$\times$ for $K>4$) while remaining cost-efficient, and (iii) a comprehensive evaluation and generation framework that combines RAGAS, BERTScore-F1, and ROUGE-Recall to assess semantic alignment and faithfulness across models and prompt designs. Our results show that carefully designed open-source pipelines can rival or outperform proprietary approaches in retrieval quality, while a custom legal-grounded prompt consistently produces more faithful and contextually relevant answers than baseline prompting. Taken together, these contributions demonstrate the potential of task-aware, component-level tuning to deliver legally grounded, reproducible, and cost-effective RAG systems for legal research assistance.
>
---
#### [new 047] LoraxBench: A Multitask, Multilingual Benchmark Suite for 20 Indonesian Languages
- **分类: cs.CL**

- **简介: 论文提出LoraxBench，一个针对印尼20种低资源语言的多任务多语言基准，涵盖阅读理解、问答、推理等6项任务，旨在推动印尼语NLP发展，发现模型在不同语言和语体下性能差异显著。**

- **链接: [http://arxiv.org/pdf/2508.12459v1](http://arxiv.org/pdf/2508.12459v1)**

> **作者:** Alham Fikri Aji; Trevor Cohn
>
> **摘要:** As one of the world's most populous countries, with 700 languages spoken, Indonesia is behind in terms of NLP progress. We introduce LoraxBench, a benchmark that focuses on low-resource languages of Indonesia and covers 6 diverse tasks: reading comprehension, open-domain QA, language inference, causal reasoning, translation, and cultural QA. Our dataset covers 20 languages, with the addition of two formality registers for three languages. We evaluate a diverse set of multilingual and region-focused LLMs and found that this benchmark is challenging. We note a visible discrepancy between performance in Indonesian and other languages, especially the low-resource ones. There is no clear lead when using a region-specific model as opposed to the general multilingual model. Lastly, we show that a change in register affects model performance, especially with registers not commonly found in social media, such as high-level politeness `Krama' Javanese.
>
---
#### [new 048] A Stitch in Time Saves Nine: Proactive Self-Refinement for Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出ProActive Self-Refinement（PASR），一种在生成过程中主动优化语言模型输出的方法，解决传统方法依赖固定迭代、效率低的问题。通过动态判断何时何地进行细化，PASR在10项任务上显著提升准确率并减少41.6%的token消耗。**

- **链接: [http://arxiv.org/pdf/2508.12903v1](http://arxiv.org/pdf/2508.12903v1)**

> **作者:** Jinyi Han; Xinyi Wang; Haiquan Zhao; Tingyun li; Zishang Jiang; Sihang Jiang; Jiaqing Liang; Xin Lin; Weikang Zhou; Zeye Sun; Fei Yu; Yanghua Xiao
>
> **摘要:** Recent advances in self-refinement have demonstrated significant potential for improving the outputs of large language models (LLMs) through iterative refinement. However, most existing self-refinement methods rely on a reactive process with a fixed number of iterations, making it difficult to determine the optimal timing and content of refinement based on the evolving generation context. Inspired by the way humans dynamically refine their thoughts during execution, we propose ProActive Self-Refinement (PASR), a novel method that enables LLMs to refine their outputs during the generation process. Unlike methods that regenerate entire responses, PASR proactively decides whether, when, and how to refine based on the model's internal state and evolving context. We conduct extensive experiments on a diverse set of 10 tasks to evaluate the effectiveness of PASR. Experimental results show that PASR significantly enhances problem-solving performance. In particular, on Qwen3-8B, PASR reduces average token consumption by 41.6 percent compared to standard generation, while also achieving an 8.2 percent improvement in accuracy. Our code and all baselines used in the paper are available in the GitHub.
>
---
#### [new 049] When Alignment Hurts: Decoupling Representational Spaces in Multilingual Models
- **分类: cs.CL**

- **简介: 论文研究多语言模型中标准语种对低资源方言生成的负面影响，提出在线变分探测框架实现表示空间解耦，提升方言生成质量，同时牺牲标准语性能。**

- **链接: [http://arxiv.org/pdf/2508.12803v1](http://arxiv.org/pdf/2508.12803v1)**

> **作者:** Ahmed Elshabrawy; Hour Kaing; Haiyue Song; Alham Fikri Aji; Hideki Tanaka; Masao Utiyama; Raj Dabre
>
> **摘要:** Alignment with high-resource standard languages is often assumed to aid the modeling of related low-resource varieties. We challenge this assumption by demonstrating that excessive representational entanglement with a dominant variety, such as Modern Standard Arabic (MSA) in relation to Arabic dialects, can actively hinder generative modeling. We present the first comprehensive causal study of this phenomenon by analyzing and directly intervening in the internal representation geometry of large language models (LLMs). Our key contribution is an online variational probing framework that continuously estimates the subspace of the standard variety during fine-tuning, enabling projection-based decoupling from this space. While our study uses Arabic as a case due to its unusually rich parallel resources across 25 dialects, the broader motivation is methodological: dialectal MT serves as a controlled proxy for generative tasks where comparable multi-variety corpora are unavailable. Across 25 dialects, our intervention improves generation quality by up to +4.9 chrF++ and +2.0 on average compared to standard fine-tuning, despite a measured tradeoff in standard-language performance. These results provide causal evidence that subspace dominance by high-resource varieties can restrict generative capacity for related varieties. More generally, we unify geometric and information-theoretic probing with subspace-level causal interventions, offering practical tools for improving generative modeling in closely related language families and, more broadly, for controlling representational allocation in multilingual and multi-domain LLMs. Code will be released.
>
---
#### [new 050] Signal and Noise: A Framework for Reducing Uncertainty in Language Model Evaluation
- **分类: cs.CL; cs.LG**

- **简介: 论文提出信号与噪声框架，用于提升语言模型评估的可靠性。解决小规模实验中基准测试不可靠的问题，通过优化信号（区分模型优劣能力）和降低噪声（随机波动），改进多任务评估与Scaling Law预测准确性。**

- **链接: [http://arxiv.org/pdf/2508.13144v1](http://arxiv.org/pdf/2508.13144v1)**

> **作者:** David Heineman; Valentin Hofmann; Ian Magnusson; Yuling Gu; Noah A. Smith; Hannaneh Hajishirzi; Kyle Lo; Jesse Dodge
>
> **摘要:** Developing large language models is expensive and involves making decisions with small experiments, typically by evaluating on large, multi-task evaluation suites. In this work, we analyze specific properties which make a benchmark more reliable for such decisions, and interventions to design higher-quality evaluation benchmarks. We introduce two key metrics that show differences in current benchmarks: signal, a benchmark's ability to separate better models from worse models, and noise, a benchmark's sensitivity to random variability between training steps. We demonstrate that benchmarks with a better signal-to-noise ratio are more reliable when making decisions at small scale, and those with less noise have lower scaling law prediction error. These results suggest that improving signal or noise will lead to more useful benchmarks, so we introduce three interventions designed to directly affect signal or noise. For example, we propose that switching to a metric that has better signal and noise (e.g., perplexity rather than accuracy) leads to better reliability and improved scaling law error. We also find that filtering noisy subtasks, to improve an aggregate signal-to-noise ratio, leads to more reliable multi-task evaluations. We also find that averaging the output of a model's intermediate checkpoints to reduce noise leads to consistent improvements. We conclude by recommending that those creating new benchmarks, or selecting which existing benchmarks to use, aim for high signal and low noise. We use 30 benchmarks for these experiments, and 375 open-weight language models from 60M to 32B parameters, resulting in a new, publicly available dataset of 900K evaluation benchmark results, totaling 200M instances.
>
---
#### [new 051] Integrating Feedback Loss from Bi-modal Sarcasm Detector for Sarcastic Speech Synthesis
- **分类: cs.CL**

- **简介: 该论文属于语音合成任务，旨在提升合成语音中讽刺语气的表现力。针对讽刺语调复杂且标注数据稀缺的问题，提出将双模态讽刺检测模型的反馈损失融入TTS训练，并采用两阶段微调策略，显著改善了合成语音的质量、自然度和讽刺感知能力。**

- **链接: [http://arxiv.org/pdf/2508.13028v1](http://arxiv.org/pdf/2508.13028v1)**

> **作者:** Zhu Li; Yuqing Zhang; Xiyuan Gao; Devraj Raghuvanshi; Nagendra Kumar; Shekhar Nayak; Matt Coler
>
> **备注:** Speech Synthesis Workshop 2025
>
> **摘要:** Sarcastic speech synthesis, which involves generating speech that effectively conveys sarcasm, is essential for enhancing natural interactions in applications such as entertainment and human-computer interaction. However, synthesizing sarcastic speech remains a challenge due to the nuanced prosody that characterizes sarcasm, as well as the limited availability of annotated sarcastic speech data. To address these challenges, this study introduces a novel approach that integrates feedback loss from a bi-modal sarcasm detection model into the TTS training process, enhancing the model's ability to capture and convey sarcasm. In addition, by leveraging transfer learning, a speech synthesis model pre-trained on read speech undergoes a two-stage fine-tuning process. First, it is fine-tuned on a diverse dataset encompassing various speech styles, including sarcastic speech. In the second stage, the model is further refined using a dataset focused specifically on sarcastic speech, enhancing its ability to generate sarcasm-aware speech. Objective and subjective evaluations demonstrate that our proposed methods improve the quality, naturalness, and sarcasm-awareness of synthesized speech.
>
---
#### [new 052] A Survey of Idiom Datasets for Psycholinguistic and Computational Research
- **分类: cs.CL**

- **简介: 该论文属于跨领域文献综述任务，旨在梳理心理语言学与计算语言学中用于研究习语的数据集。它总结了53个数据集的标注方式、覆盖范围和任务类型，指出两者尚未建立联系，呼吁加强协作以推动习语研究。**

- **链接: [http://arxiv.org/pdf/2508.11828v1](http://arxiv.org/pdf/2508.11828v1)**

> **作者:** Michael Flor; Xinyi Liu; Anna Feldman
>
> **备注:** KONVENS 2025. To appear
>
> **摘要:** Idioms are figurative expressions whose meanings often cannot be inferred from their individual words, making them difficult to process computationally and posing challenges for human experimental studies. This survey reviews datasets developed in psycholinguistics and computational linguistics for studying idioms, focusing on their content, form, and intended use. Psycholinguistic resources typically contain normed ratings along dimensions such as familiarity, transparency, and compositionality, while computational datasets support tasks like idiomaticity detection/classification, paraphrasing, and cross-lingual modeling. We present trends in annotation practices, coverage, and task framing across 53 datasets. Although recent efforts expanded language coverage and task diversity, there seems to be no relation yet between psycholinguistic and computational research on idioms.
>
---
#### [new 053] Beyond Modality Limitations: A Unified MLLM Approach to Automated Speaking Assessment with Effective Curriculum Learning
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 论文研究多模态大语言模型（MLLM）在自动口语评估中的应用，解决传统方法因模态限制导致的性能瓶颈。提出Speech-First Multimodal Training（SFMT）策略，通过课程学习提升语音特征建模，显著改善整体评估效果，尤其在发音等表达维度上取得突破。**

- **链接: [http://arxiv.org/pdf/2508.12591v1](http://arxiv.org/pdf/2508.12591v1)**

> **作者:** Yu-Hsuan Fang; Tien-Hong Lo; Yao-Ting Sung; Berlin Chen
>
> **备注:** Accepted at IEEE ASRU 2025
>
> **摘要:** Traditional Automated Speaking Assessment (ASA) systems exhibit inherent modality limitations: text-based approaches lack acoustic information while audio-based methods miss semantic context. Multimodal Large Language Models (MLLM) offer unprecedented opportunities for comprehensive ASA by simultaneously processing audio and text within unified frameworks. This paper presents a very first systematic study of MLLM for comprehensive ASA, demonstrating the superior performance of MLLM across the aspects of content and language use . However, assessment on the delivery aspect reveals unique challenges, which is deemed to require specialized training strategies. We thus propose Speech-First Multimodal Training (SFMT), leveraging a curriculum learning principle to establish more robust modeling foundations of speech before cross-modal synergetic fusion. A series of experiments on a benchmark dataset show MLLM-based systems can elevate the holistic assessment performance from a PCC value of 0.783 to 0.846. In particular, SFMT excels in the evaluation of the delivery aspect, achieving an absolute accuracy improvement of 4% over conventional training approaches, which also paves a new avenue for ASA.
>
---
#### [new 054] ding-01 :ARG0: An AMR Corpus for Spontaneous French Dialogue
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的语义表示任务，旨在构建法语对话的AMR语义资源。为解决AMR对自发对话覆盖不足的问题，作者扩展了AMR框架并制定标注指南，构建了DinG语料库，还训练了AMR解析器辅助标注。**

- **链接: [http://arxiv.org/pdf/2508.12819v1](http://arxiv.org/pdf/2508.12819v1)**

> **作者:** Jeongwoo Kang; Maria Boritchev; Maximin Coavoux
>
> **备注:** Accepted at IWCS 2025
>
> **摘要:** We present our work to build a French semantic corpus by annotating French dialogue in Abstract Meaning Representation (AMR). Specifically, we annotate the DinG corpus, consisting of transcripts of spontaneous French dialogues recorded during the board game Catan. As AMR has insufficient coverage of the dynamics of spontaneous speech, we extend the framework to better represent spontaneous speech and sentence structures specific to French. Additionally, to support consistent annotation, we provide an annotation guideline detailing these extensions. We publish our corpus under a free license (CC-SA-BY). We also train and evaluate an AMR parser on our data. This model can be used as an assistance annotation tool to provide initial annotations that can be refined by human annotators. Our work contributes to the development of semantic resources for French dialogue.
>
---
#### [new 055] OptimalThinkingBench: Evaluating Over and Underthinking in LLMs
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出OptimalThinkingBench，用于评估大模型在复杂任务中的过度思考与简单任务中的思考不足问题。旨在推动发展性能与效率平衡的最优思考模型。通过两个子基准测试33个模型，发现当前无模型能兼顾两者。**

- **链接: [http://arxiv.org/pdf/2508.13141v1](http://arxiv.org/pdf/2508.13141v1)**

> **作者:** Pranjal Aggarwal; Seungone Kim; Jack Lanchantin; Sean Welleck; Jason Weston; Ilia Kulikov; Swarnadeep Saha
>
> **备注:** 26 pages, 6 tables, 10 figures
>
> **摘要:** Thinking LLMs solve complex tasks at the expense of increased compute and overthinking on simpler problems, while non-thinking LLMs are faster and cheaper but underthink on harder reasoning problems. This has led to the development of separate thinking and non-thinking LLM variants, leaving the onus of selecting the optimal model for each query on the end user. In this work, we introduce OptimalThinkingBench, a unified benchmark that jointly evaluates overthinking and underthinking in LLMs and also encourages the development of optimally-thinking models that balance performance and efficiency. Our benchmark comprises two sub-benchmarks: OverthinkingBench, featuring simple queries in 72 domains, and UnderthinkingBench, containing 11 challenging reasoning tasks. Using novel thinking-adjusted accuracy metrics, we perform extensive evaluation of 33 different thinking and non-thinking models and show that no model is able to optimally think on our benchmark. Thinking models often overthink for hundreds of tokens on the simplest user queries without improving performance. In contrast, large non-thinking models underthink, often falling short of much smaller thinking models. We further explore several methods to encourage optimal thinking, but find that these approaches often improve on one sub-benchmark at the expense of the other, highlighting the need for better unified and optimal models in the future.
>
---
#### [new 056] Limitation Learning: Catching Adverse Dialog with GAIL
- **分类: cs.CL; cs.LG**

- **简介: 论文将模仿学习应用于对话系统，通过专家演示训练策略和判别器。策略能生成对话，判别器识别合成对话的局限性，从而发现不良行为。任务是提升对话模型质量并检测其缺陷。**

- **链接: [http://arxiv.org/pdf/2508.11767v1](http://arxiv.org/pdf/2508.11767v1)**

> **作者:** Noah Kasmanoff; Rahul Zalkikar
>
> **备注:** Paper from 2021
>
> **摘要:** Imitation learning is a proven method for creating a policy in the absence of rewards, by leveraging expert demonstrations. In this work, we apply imitation learning to conversation. In doing so, we recover a policy capable of talking to a user given a prompt (input state), and a discriminator capable of classifying between expert and synthetic conversation. While our policy is effective, we recover results from our discriminator that indicate the limitations of dialog models. We argue that this technique can be used to identify adverse behavior of arbitrary data models common for dialog oriented tasks.
>
---
#### [new 057] CAMF: Collaborative Adversarial Multi-agent Framework for Machine Generated Text Detection
- **分类: cs.CL**

- **简介: 该论文提出CAMF框架，用于零样本机器生成文本检测。针对现有方法分析浅显、缺乏多维一致性探究的问题，通过多智能体协同对抗机制，从语言特征提取、一致性 probing 到综合判断，实现对文本跨维度不一致性的深度分析，显著优于现有技术。**

- **链接: [http://arxiv.org/pdf/2508.11933v1](http://arxiv.org/pdf/2508.11933v1)**

> **作者:** Yue Wang; Liesheng Wei; Yuxiang Wang
>
> **摘要:** Detecting machine-generated text (MGT) from contemporary Large Language Models (LLMs) is increasingly crucial amid risks like disinformation and threats to academic integrity. Existing zero-shot detection paradigms, despite their practicality, often exhibit significant deficiencies. Key challenges include: (1) superficial analyses focused on limited textual attributes, and (2) a lack of investigation into consistency across linguistic dimensions such as style, semantics, and logic. To address these challenges, we introduce the \textbf{C}ollaborative \textbf{A}dversarial \textbf{M}ulti-agent \textbf{F}ramework (\textbf{CAMF}), a novel architecture using multiple LLM-based agents. CAMF employs specialized agents in a synergistic three-phase process: \emph{Multi-dimensional Linguistic Feature Extraction}, \emph{Adversarial Consistency Probing}, and \emph{Synthesized Judgment Aggregation}. This structured collaborative-adversarial process enables a deep analysis of subtle, cross-dimensional textual incongruities indicative of non-human origin. Empirical evaluations demonstrate CAMF's significant superiority over state-of-the-art zero-shot MGT detection techniques.
>
---
#### [new 058] The Structural Sources of Verb Meaning Revisited: Large Language Models Display Syntactic Bootstrapping
- **分类: cs.CL**

- **简介: 论文研究大语言模型是否具备语法引导学习能力，即能否通过句法环境推断动词含义。通过扰动数据中句法或共现信息，发现模型动词表征更依赖句法线索，尤其对心理动词影响更大，验证了句法引导在动词学习中的关键作用。**

- **链接: [http://arxiv.org/pdf/2508.12482v1](http://arxiv.org/pdf/2508.12482v1)**

> **作者:** Xiaomeng Zhu; R. Thomas McCoy; Robert Frank
>
> **摘要:** Syntactic bootstrapping (Gleitman, 1990) is the hypothesis that children use the syntactic environments in which a verb occurs to learn its meaning. In this paper, we examine whether large language models exhibit a similar behavior. We do this by training RoBERTa and GPT-2 on perturbed datasets where syntactic information is ablated. Our results show that models' verb representation degrades more when syntactic cues are removed than when co-occurrence information is removed. Furthermore, the representation of mental verbs, for which syntactic bootstrapping has been shown to be particularly crucial in human verb learning, is more negatively impacted in such training regimes than physical verbs. In contrast, models' representation of nouns is affected more when co-occurrences are distorted than when syntax is distorted. In addition to reinforcing the important role of syntactic bootstrapping in verb learning, our results demonstrated the viability of testing developmental hypotheses on a larger scale through manipulating the learning environments of large language models.
>
---
#### [new 059] M3PO: Multimodal-Model-Guided Preference Optimization for Visual Instruction Following
- **分类: cs.CL**

- **简介: 论文提出M3PO方法，用于提升视觉指令跟随任务中大模型的偏好对齐效果。针对人工标注成本高、难获取高质量负样本的问题，M3PO通过融合多模态对齐分数与模型置信度，自动筛选高价值偏好样本，实现高效DPO微调。**

- **链接: [http://arxiv.org/pdf/2508.12458v1](http://arxiv.org/pdf/2508.12458v1)**

> **作者:** Ruirui Gao; Emily Johnson; Bowen Tan; Yanfei Qian
>
> **摘要:** Large Vision-Language Models (LVLMs) hold immense potential for complex multimodal instruction following, yet their development is often hindered by the high cost and inconsistency of human annotation required for effective fine-tuning and preference alignment. Traditional supervised fine-tuning (SFT) and existing preference optimization methods like RLHF and DPO frequently struggle to efficiently leverage the model's own generation space to identify highly informative "hard negative" samples. To address these challenges, we propose Multimodal-Model-Guided Preference Optimization (M3PO), a novel and data-efficient method designed to enhance LVLMs' capabilities in visual instruction following. M3PO intelligently selects the most "learning-valuable" preference sample pairs from a diverse pool of LVLM-generated candidates. This selection is driven by a sophisticated mechanism that integrates two crucial signals: a Multimodal Alignment Score (MAS) to assess external quality and the model's Self-Consistency / Confidence (log-probability) to gauge internal belief. These are combined into a novel M3P-Score, which specifically identifies preferred responses and challenging dispreferred responses that the model might confidently generate despite being incorrect. These high-quality preference pairs are then used for efficient Direct Preference Optimization (DPO) fine-tuning on base LVLMs like LLaVA-1.5 (7B/13B) using LoRA. Our extensive experiments demonstrate that M3PO consistently outperforms strong baselines, including SFT, simulated RLHF, vanilla DPO, and RM-DPO, across a comprehensive suite of multimodal instruction following benchmarks (MME-Bench, POPE, IFT, Human Pref. Score).
>
---
#### [new 060] Evaluating ASR robustness to spontaneous speech errors: A study of WhisperX using a Speech Error Database
- **分类: cs.CL**

- **简介: 论文研究自动语音识别（ASR）模型对自发语音错误的鲁棒性，使用SFUSED数据库评估WhisperX在5300个语音错误上的表现，旨在诊断ASR系统性能。**

- **链接: [http://arxiv.org/pdf/2508.13060v1](http://arxiv.org/pdf/2508.13060v1)**

> **作者:** John Alderete; Macarious Kin Fung Hui; Aanchan Mohan
>
> **备注:** 5 pages, 6 figures, 1 table, Interspeech 2025 (Rotterdam)
>
> **摘要:** The Simon Fraser University Speech Error Database (SFUSED) is a public data collection developed for linguistic and psycholinguistic research. Here we demonstrate how its design and annotations can be used to test and evaluate speech recognition models. The database comprises systematically annotated speech errors from spontaneous English speech, with each error tagged for intended and actual error productions. The annotation schema incorporates multiple classificatory dimensions that are of some value to model assessment, including linguistic hierarchical level, contextual sensitivity, degraded words, word corrections, and both word-level and syllable-level error positioning. To assess the value of these classificatory variables, we evaluated the transcription accuracy of WhisperX across 5,300 documented word and phonological errors. This analysis demonstrates the atabase's effectiveness as a diagnostic tool for ASR system performance.
>
---
#### [new 061] Spot the BlindSpots: Systematic Identification and Quantification of Fine-Grained LLM Biases in Contact Center Summaries
- **分类: cs.CL; cs.AI**

- **简介: 论文提出BlindSpot框架，用于系统识别和量化客服对话摘要中的操作偏见（Operational Bias），解决LLM在生成摘要时可能忽视或过度关注特定内容的问题。通过15维偏见分类和两个量化指标，在2500条真实对话上分析20个模型，发现偏见普遍存在。**

- **链接: [http://arxiv.org/pdf/2508.13124v1](http://arxiv.org/pdf/2508.13124v1)**

> **作者:** Kawin Mayilvaghanan; Siddhant Gupta; Ayush Kumar
>
> **摘要:** Abstractive summarization is a core application in contact centers, where Large Language Models (LLMs) generate millions of summaries of call transcripts daily. Despite their apparent quality, it remains unclear whether LLMs systematically under- or over-attend to specific aspects of the transcript, potentially introducing biases in the generated summary. While prior work has examined social and positional biases, the specific forms of bias pertinent to contact center operations - which we term Operational Bias - have remained unexplored. To address this gap, we introduce BlindSpot, a framework built upon a taxonomy of 15 operational bias dimensions (e.g., disfluency, speaker, topic) for the identification and quantification of these biases. BlindSpot leverages an LLM as a zero-shot classifier to derive categorical distributions for each bias dimension in a pair of transcript and its summary. The bias is then quantified using two metrics: Fidelity Gap (the JS Divergence between distributions) and Coverage (the percentage of source labels omitted). Using BlindSpot, we conducted an empirical study with 2500 real call transcripts and their summaries generated by 20 LLMs of varying scales and families (e.g., GPT, Llama, Claude). Our analysis reveals that biases are systemic and present across all evaluated models, regardless of size or family.
>
---
#### [new 062] Beyond GPT-5: Making LLMs Cheaper and Better via Performance-Efficiency Optimized Routing
- **分类: cs.CL**

- **简介: 该论文提出Avengers-Pro框架，解决大语言模型在性能与效率间的平衡问题。通过测试时路由机制，动态选择最优模型，实现更高准确率或更低计算成本，优于单一模型表现。**

- **链接: [http://arxiv.org/pdf/2508.12631v1](http://arxiv.org/pdf/2508.12631v1)**

> **作者:** Yiqun Zhang; Hao Li; Jianhao Chen; Hangfan Zhang; Peng Ye; Lei Bai; Shuyue Hu
>
> **备注:** Ongoing work
>
> **摘要:** Balancing performance and efficiency is a central challenge in large language model (LLM) advancement. GPT-5 addresses this with test-time routing, dynamically assigning queries to either an efficient or a high-capacity model during inference. In this work, we present Avengers-Pro, a test-time routing framework that ensembles LLMs of varying capacities and efficiencies, providing a unified solution for all performance-efficiency tradeoffs. The Avengers-Pro embeds and clusters incoming queries, then routes each to the most suitable model based on a performance-efficiency score. Across 6 challenging benchmarks and 8 leading models -- including GPT-5-medium, Gemini-2.5-pro, and Claude-opus-4.1 -- Avengers-Pro achieves state-of-the-art results: by varying a performance-efficiency trade-off parameter, it can surpass the strongest single model (GPT-5-medium) by +7% in average accuracy. Moreover, it can match the average accuracy of the strongest single model at 27% lower cost, and reach ~90% of that performance at 63% lower cost. Last but not least, it achieves a Pareto frontier, consistently yielding the highest accuracy for any given cost, and the lowest cost for any given accuracy, among all single models. Code is available at https://github.com/ZhangYiqun018/AvengersPro.
>
---
#### [new 063] Word Meanings in Transformer Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究Transformer语言模型中词义的表示方式，旨在验证模型是否具备类似词汇存储的结构。通过聚类RoBERTa-base的词嵌入空间并分析其语义敏感性，发现模型编码了丰富的语义信息，反驳了意义消解论观点。**

- **链接: [http://arxiv.org/pdf/2508.12863v1](http://arxiv.org/pdf/2508.12863v1)**

> **作者:** Jumbly Grindrod; Peter Grindrod
>
> **摘要:** We investigate how word meanings are represented in the transformer language models. Specifically, we focus on whether transformer models employ something analogous to a lexical store - where each word has an entry that contains semantic information. To do this, we extracted the token embedding space of RoBERTa-base and k-means clustered it into 200 clusters. In our first study, we then manually inspected the resultant clusters to consider whether they are sensitive to semantic information. In our second study, we tested whether the clusters are sensitive to five psycholinguistic measures: valence, concreteness, iconicity, taboo, and age of acquisition. Overall, our findings were very positive - there is a wide variety of semantic information encoded within the token embedding space. This serves to rule out certain "meaning eliminativist" hypotheses about how transformer LLMs process semantic information.
>
---
#### [new 064] Hallucination Detection and Mitigation in Scientific Text Simplification using Ensemble Approaches: DS@GT at CLEF 2025 SimpleText
- **分类: cs.CL**

- **简介: 该论文针对科学文本简化中的幻觉检测与缓解问题，提出基于BERT、语义相似度、自然语言推理和大语言模型的集成方法，结合元分类器提升检测鲁棒性，并用LLM后编辑确保生成内容忠实于原文。**

- **链接: [http://arxiv.org/pdf/2508.11823v1](http://arxiv.org/pdf/2508.11823v1)**

> **作者:** Krishna Chaitanya Marturi; Heba H. Elwazzan
>
> **备注:** Text Simplification, hallucination detection, LLMs, CLEF 2025, SimpleText, CEUR-WS
>
> **摘要:** In this paper, we describe our methodology for the CLEF 2025 SimpleText Task 2, which focuses on detecting and evaluating creative generation and information distortion in scientific text simplification. Our solution integrates multiple strategies: we construct an ensemble framework that leverages BERT-based classifier, semantic similarity measure, natural language inference model, and large language model (LLM) reasoning. These diverse signals are combined using meta-classifiers to enhance the robustness of spurious and distortion detection. Additionally, for grounded generation, we employ an LLM-based post-editing system that revises simplifications based on the original input texts.
>
---
#### [new 065] SEA-BED: Southeast Asia Embedding Benchmark
- **分类: cs.CL**

- **简介: 该论文提出SEA-BED，首个面向东南亚的句子嵌入基准，涵盖10种语言、169个数据集。解决低资源语言评估不足问题，通过人类标注数据验证模型性能，揭示语言差异与翻译影响。**

- **链接: [http://arxiv.org/pdf/2508.12243v1](http://arxiv.org/pdf/2508.12243v1)**

> **作者:** Wuttikorn Ponwitayarat; Raymond Ng; Jann Railey Montalan; Thura Aung; Jian Gang Ngui; Yosephine Susanto; William Tjhi; Panuthep Tasawong; Erik Cambria; Ekapol Chuangsuwanich; Sarana Nutanong; Peerat Limkonchotiwat
>
> **摘要:** Sentence embeddings are essential for NLP tasks such as semantic search, re-ranking, and textual similarity. Although multilingual benchmarks like MMTEB broaden coverage, Southeast Asia (SEA) datasets are scarce and often machine-translated, missing native linguistic properties. With nearly 700 million speakers, the SEA region lacks a region-specific embedding benchmark. We introduce SEA-BED, the first large-scale SEA embedding benchmark with 169 datasets across 9 tasks and 10 languages, where 71% are formulated by humans, not machine generation or translation. We address three research questions: (1) which SEA languages and tasks are challenging, (2) whether SEA languages show unique performance gaps globally, and (3) how human vs. machine translations affect evaluation. We evaluate 17 embedding models across six studies, analyzing task and language challenges, cross-benchmark comparisons, and translation trade-offs. Results show sharp ranking shifts, inconsistent model performance among SEA languages, and the importance of human-curated datasets for low-resource languages like Burmese.
>
---
#### [new 066] Arabic Multimodal Machine Learning: Datasets, Applications, Approaches, and Challenges
- **分类: cs.CL**

- **简介: 该论文属于阿拉伯语多模态机器学习领域，旨在系统梳理该方向的研究现状。通过构建四维分类体系（数据集、应用、方法、挑战），总结已有成果并指出研究空白，为后续研究提供方向。**

- **链接: [http://arxiv.org/pdf/2508.12227v1](http://arxiv.org/pdf/2508.12227v1)**

> **作者:** Abdelhamid Haouhat; Slimane Bellaouar; Attia Nehar; Hadda Cherroun; Ahmed Abdelali
>
> **摘要:** Multimodal Machine Learning (MML) aims to integrate and analyze information from diverse modalities, such as text, audio, and visuals, enabling machines to address complex tasks like sentiment analysis, emotion recognition, and multimedia retrieval. Recently, Arabic MML has reached a certain level of maturity in its foundational development, making it time to conduct a comprehensive survey. This paper explores Arabic MML by categorizing efforts through a novel taxonomy and analyzing existing research. Our taxonomy organizes these efforts into four key topics: datasets, applications, approaches, and challenges. By providing a structured overview, this survey offers insights into the current state of Arabic MML, highlighting areas that have not been investigated and critical research gaps. Researchers will be empowered to build upon the identified opportunities and address challenges to advance the field.
>
---
#### [new 067] WebMall -- A Multi-Shop Benchmark for Evaluating Web Agents
- **分类: cs.CL**

- **简介: 论文提出WebMall基准，用于评估基于大语言模型的网络代理在多店铺比价购物任务中的表现。该任务解决真实电商场景下代理的导航、推理与效率问题，通过模拟四家商店和91个跨店任务，推动Web Agent研究发展。**

- **链接: [http://arxiv.org/pdf/2508.13024v1](http://arxiv.org/pdf/2508.13024v1)**

> **作者:** Ralph Peeters; Aaron Steiner; Luca Schwarz; Julian Yuya Caspary; Christian Bizer
>
> **摘要:** LLM-based web agents have the potential to automate long-running web tasks, such as finding offers for specific products in multiple online shops and subsequently ordering the cheapest products that meet the users needs. This paper introduces WebMall, a multi-shop online shopping benchmark for evaluating the effectiveness and efficiency of web agents for comparison-shopping. WebMall consists of four simulated online shops populated with authentic product offers sourced from the Common Crawl, alongside a suite of 91 cross-shop tasks. These tasks include basic tasks such as finding specific products in multiple shops, performing price comparisons, adding items to the shopping cart, and completing checkout. Advanced tasks involve searching for products based on vague requirements, identifying suitable substitutes, and finding compatible products. Compared to existing e-commerce benchmarks, such as WebShop or ShoppingBench, WebMall introduces comparison-shopping tasks across multiple shops. Furthermore, the product offers are more heterogeneous, as they originate from hundreds of distinct real-world shops. The tasks in WebMall require longer interaction trajectories than those in WebShop, while remaining representative of real-world shopping behaviors. We evaluate eight baseline agents on WebMall, varying in observation modality, memory utilization, and underlying large language model (GPT 4.1 and Claude Sonnet 4). The best-performing configurations achieve completion rates of 75% and 53%, and F1 scores of 87% and 63%, on the basic and advanced task sets, respectively. WebMall is publicly released to facilitate research on web agents and to promote advancements in navigation, reasoning, and efficiency within e-commerce scenarios.
>
---
#### [new 068] LLM-Guided Planning and Summary-Based Scientific Text Simplification: DS@GT at CLEF 2025 SimpleText
- **分类: cs.CL**

- **简介: 该论文针对科学文本简化任务，解决句子级和文档级简化问题。提出基于大语言模型的两阶段框架：先生成结构化计划指导句子简化，再用摘要引导文档级简化，提升简化文本的连贯性和准确性。**

- **链接: [http://arxiv.org/pdf/2508.11816v1](http://arxiv.org/pdf/2508.11816v1)**

> **作者:** Krishna Chaitanya Marturi; Heba H. Elwazzan
>
> **备注:** Text Simplification, hallucination detection, LLMs, CLEF 2025, SimpleText, CEUR-WS
>
> **摘要:** In this paper, we present our approach for the CLEF 2025 SimpleText Task 1, which addresses both sentence-level and document-level scientific text simplification. For sentence-level simplification, our methodology employs large language models (LLMs) to first generate a structured plan, followed by plan-driven simplification of individual sentences. At the document level, we leverage LLMs to produce concise summaries and subsequently guide the simplification process using these summaries. This two-stage, LLM-based framework enables more coherent and contextually faithful simplifications of scientific text.
>
---
#### [new 069] The Cultural Gene of Large Language Models: A Study on the Impact of Cross-Corpus Training on Model Values and Biases
- **分类: cs.CL; I.2.7; K.4.1; H.3.3**

- **简介: 论文研究大语言模型的文化基因，通过跨语料库训练揭示其价值倾向与偏见。提出文化探针数据集（CPD），对比GPT-4与ERNIE Bot在个体主义-集体主义、权力距离维度的差异，发现模型倾向与其训练语料文化背景一致，强调需关注文化适配性以避免算法文化霸权。**

- **链接: [http://arxiv.org/pdf/2508.12411v1](http://arxiv.org/pdf/2508.12411v1)**

> **作者:** Emanuel Z. Fenech-Borg; Tilen P. Meznaric-Kos; Milica D. Lekovic-Bojovic; Arni J. Hentze-Djurhuus
>
> **备注:** 10 pages, 5 figures, IEEE conference format, submitted to [Conference Name]
>
> **摘要:** Large language models (LLMs) are deployed globally, yet their underlying cultural and ethical assumptions remain underexplored. We propose the notion of a "cultural gene" -- a systematic value orientation that LLMs inherit from their training corpora -- and introduce a Cultural Probe Dataset (CPD) of 200 prompts targeting two classic cross-cultural dimensions: Individualism-Collectivism (IDV) and Power Distance (PDI). Using standardized zero-shot prompts, we compare a Western-centric model (GPT-4) and an Eastern-centric model (ERNIE Bot). Human annotation shows significant and consistent divergence across both dimensions. GPT-4 exhibits individualistic and low-power-distance tendencies (IDV score approx 1.21; PDI score approx -1.05), while ERNIE Bot shows collectivistic and higher-power-distance tendencies (IDV approx -0.89; PDI approx 0.76); differences are statistically significant (p < 0.001). We further compute a Cultural Alignment Index (CAI) against Hofstede's national scores and find GPT-4 aligns more closely with the USA (e.g., IDV CAI approx 0.91; PDI CAI approx 0.88) whereas ERNIE Bot aligns more closely with China (IDV CAI approx 0.85; PDI CAI approx 0.81). Qualitative analyses of dilemma resolution and authority-related judgments illustrate how these orientations surface in reasoning. Our results support the view that LLMs function as statistical mirrors of their cultural corpora and motivate culturally aware evaluation and deployment to avoid algorithmic cultural hegemony.
>
---
#### [new 070] Uncovering Emergent Physics Representations Learned In-Context by Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 论文研究大语言模型在上下文学习中对物理规律的隐式理解。针对“如何通过文本提示让LLM学会物理推理”这一问题，作者设计动力学预测任务，结合稀疏自编码器分析激活特征，发现模型能编码如能量等物理变量，揭示了LLM在上下文中涌现出物理表示机制。**

- **链接: [http://arxiv.org/pdf/2508.12448v1](http://arxiv.org/pdf/2508.12448v1)**

> **作者:** Yeongwoo Song; Jaeyong Bae; Dong-Kyum Kim; Hawoong Jeong
>
> **备注:** 17 pages, 10 figures
>
> **摘要:** Large language models (LLMs) exhibit impressive in-context learning (ICL) abilities, enabling them to solve wide range of tasks via textual prompts alone. As these capabilities advance, the range of applicable domains continues to expand significantly. However, identifying the precise mechanisms or internal structures within LLMs that allow successful ICL across diverse, distinct classes of tasks remains elusive. Physics-based tasks offer a promising testbed for probing this challenge. Unlike synthetic sequences such as basic arithmetic or symbolic equations, physical systems provide experimentally controllable, real-world data based on structured dynamics grounded in fundamental principles. This makes them particularly suitable for studying the emergent reasoning behaviors of LLMs in a realistic yet tractable setting. Here, we mechanistically investigate the ICL ability of LLMs, especially focusing on their ability to reason about physics. Using a dynamics forecasting task in physical systems as a proxy, we evaluate whether LLMs can learn physics in context. We first show that the performance of dynamics forecasting in context improves with longer input contexts. To uncover how such capability emerges in LLMs, we analyze the model's residual stream activations using sparse autoencoders (SAEs). Our experiments reveal that the features captured by SAEs correlate with key physical variables, such as energy. These findings demonstrate that meaningful physical concepts are encoded within LLMs during in-context learning. In sum, our work provides a novel case study that broadens our understanding of how LLMs learn in context.
>
---
#### [new 071] Reinforced Context Order Recovery for Adaptive Reasoning and Planning
- **分类: cs.CL; cs.AI**

- **简介: 论文提出ReCOR框架，通过强化学习从文本中无监督地学习自适应的token生成顺序，解决现有模型固定生成顺序导致推理和规划能力受限的问题，在多个挑战性任务上优于基线甚至oracle模型。**

- **链接: [http://arxiv.org/pdf/2508.13070v1](http://arxiv.org/pdf/2508.13070v1)**

> **作者:** Long Ma; Fangwei Zhong; Yizhou Wang
>
> **摘要:** Modern causal language models, followed by rapid developments in discrete diffusion models, can now produce a wide variety of interesting and useful content. However, these families of models are predominantly trained to output tokens with a fixed (left-to-right) or random order, which may deviate from the logical order in which tokens are generated originally. In this paper, we observe that current causal and diffusion models encounter difficulties in problems that require adaptive token generation orders to solve tractably, which we characterize with the $\mathcal{V}$-information framework. Motivated by this, we propose Reinforced Context Order Recovery (ReCOR), a reinforcement-learning-based framework to extract adaptive, data-dependent token generation orders from text data without annotations. Self-supervised by token prediction statistics, ReCOR estimates the hardness of predicting every unfilled token and adaptively selects the next token during both training and inference. Experiments on challenging reasoning and planning datasets demonstrate the superior performance of ReCOR compared with baselines, sometimes outperforming oracle models supervised with the ground-truth order.
>
---
#### [new 072] LLM-as-a-Judge for Privacy Evaluation? Exploring the Alignment of Human and LLM Perceptions of Privacy in Textual Data
- **分类: cs.CL**

- **简介: 论文探讨了使用大语言模型（LLM）作为隐私评估者（LLM-as-a-Judge）的可行性，旨在解决文本数据隐私评价难、主观性强的问题。通过10个数据集、13个LLM和677名参与者的研究，发现LLM能较好反映人类整体隐私观，但存在局限性。**

- **链接: [http://arxiv.org/pdf/2508.12158v1](http://arxiv.org/pdf/2508.12158v1)**

> **作者:** Stephen Meisenbacher; Alexandra Klymenko; Florian Matthes
>
> **备注:** 13 pages, 3 figures, 4 tables. Accepted to HAIPS @ CCS 2025
>
> **摘要:** Despite advances in the field of privacy-preserving Natural Language Processing (NLP), a significant challenge remains the accurate evaluation of privacy. As a potential solution, using LLMs as a privacy evaluator presents a promising approach $\unicode{x2013}$ a strategy inspired by its success in other subfields of NLP. In particular, the so-called $\textit{LLM-as-a-Judge}$ paradigm has achieved impressive results on a variety of natural language evaluation tasks, demonstrating high agreement rates with human annotators. Recognizing that privacy is both subjective and difficult to define, we investigate whether LLM-as-a-Judge can also be leveraged to evaluate the privacy sensitivity of textual data. Furthermore, we measure how closely LLM evaluations align with human perceptions of privacy in text. Resulting from a study involving 10 datasets, 13 LLMs, and 677 human survey participants, we confirm that privacy is indeed a difficult concept to measure empirically, exhibited by generally low inter-human agreement rates. Nevertheless, we find that LLMs can accurately model a global human privacy perspective, and through an analysis of human and LLM reasoning patterns, we discuss the merits and limitations of LLM-as-a-Judge for privacy evaluation in textual data. Our findings pave the way for exploring the feasibility of LLMs as privacy evaluators, addressing a core challenge in solving pressing privacy issues with innovative technical solutions.
>
---
#### [new 073] RepreGuard: Detecting LLM-Generated Text by Revealing Hidden Representation Patterns
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出RepreGuard，一种基于LLM内部表示差异的文本检测方法，旨在区分大语言模型生成文本与人类书写文本。通过分析神经激活模式差异，实现高精度、鲁棒的检测，在分布内和分布外场景均表现优异。**

- **链接: [http://arxiv.org/pdf/2508.13152v1](http://arxiv.org/pdf/2508.13152v1)**

> **作者:** Xin Chen; Junchao Wu; Shu Yang; Runzhe Zhan; Zeyu Wu; Ziyang Luo; Di Wang; Min Yang; Lidia S. Chao; Derek F. Wong
>
> **备注:** Accepted to TACL 2025. This version is a pre-MIT Press publication version
>
> **摘要:** Detecting content generated by large language models (LLMs) is crucial for preventing misuse and building trustworthy AI systems. Although existing detection methods perform well, their robustness in out-of-distribution (OOD) scenarios is still lacking. In this paper, we hypothesize that, compared to features used by existing detection methods, the internal representations of LLMs contain more comprehensive and raw features that can more effectively capture and distinguish the statistical pattern differences between LLM-generated texts (LGT) and human-written texts (HWT). We validated this hypothesis across different LLMs and observed significant differences in neural activation patterns when processing these two types of texts. Based on this, we propose RepreGuard, an efficient statistics-based detection method. Specifically, we first employ a surrogate model to collect representation of LGT and HWT, and extract the distinct activation feature that can better identify LGT. We can classify the text by calculating the projection score of the text representations along this feature direction and comparing with a precomputed threshold. Experimental results show that RepreGuard outperforms all baselines with average 94.92% AUROC on both in-distribution (ID) and OOD scenarios, while also demonstrating robust resilience to various text sizes and mainstream attacks. Data and code are publicly available at: https://github.com/NLP2CT/RepreGuard
>
---
#### [new 074] Incorporating Legal Logic into Deep Learning: An Intelligent Approach to Probation Prediction
- **分类: cs.CL**

- **简介: 论文提出将法律逻辑融入深度学习，解决智能司法系统中缺乏 probation 预测方法的问题。构建专用数据集并设计多任务双理论模型（MT-DT），结合法律要素与惩罚双轨理论，提升预测准确性与可解释性。**

- **链接: [http://arxiv.org/pdf/2508.12286v1](http://arxiv.org/pdf/2508.12286v1)**

> **作者:** Qinghua Wang; Xu Zhang; Lingyan Yang; Rui Shao; Bonan Wang; Fang Wang; Cunquan Qu
>
> **摘要:** Probation is a crucial institution in modern criminal law, embodying the principles of fairness and justice while contributing to the harmonious development of society. Despite its importance, the current Intelligent Judicial Assistant System (IJAS) lacks dedicated methods for probation prediction, and research on the underlying factors influencing probation eligibility remains limited. In addition, probation eligibility requires a comprehensive analysis of both criminal circumstances and remorse. Much of the existing research in IJAS relies primarily on data-driven methodologies, which often overlooks the legal logic underpinning judicial decision-making. To address this gap, we propose a novel approach that integrates legal logic into deep learning models for probation prediction, implemented in three distinct stages. First, we construct a specialized probation dataset that includes fact descriptions and probation legal elements (PLEs). Second, we design a distinct probation prediction model named the Multi-Task Dual-Theory Probation Prediction Model (MT-DT), which is grounded in the legal logic of probation and the \textit{Dual-Track Theory of Punishment}. Finally, our experiments on the probation dataset demonstrate that the MT-DT model outperforms baseline models, and an analysis of the underlying legal logic further validates the effectiveness of the proposed approach.
>
---
#### [new 075] Investigating Transcription Normalization in the Faetar ASR Benchmark
- **分类: cs.CL**

- **简介: 该论文研究低资源语音识别中的转录一致性问题，针对Faetar ASR基准测试，发现转录不一致并非主要挑战，而有限词典约束有助于提升性能，bigram语言模型无显著帮助。**

- **链接: [http://arxiv.org/pdf/2508.11771v1](http://arxiv.org/pdf/2508.11771v1)**

> **作者:** Leo Peckham; Michael Ong; Naomi Nagy; Ewan Dunbar
>
> **摘要:** We examine the role of transcription inconsistencies in the Faetar Automatic Speech Recognition benchmark, a challenging low-resource ASR benchmark. With the help of a small, hand-constructed lexicon, we conclude that find that, while inconsistencies do exist in the transcriptions, they are not the main challenge in the task. We also demonstrate that bigram word-based language modelling is of no added benefit, but that constraining decoding to a finite lexicon can be beneficial. The task remains extremely difficult.
>
---
#### [new 076] MuDRiC: Multi-Dialect Reasoning for Arabic Commonsense Validation
- **分类: cs.CL**

- **简介: 该论文聚焦阿拉伯语常识验证任务，解决现有研究偏重标准语、忽视方言的问题。提出MuDRiC数据集和基于图卷积网络的推理方法，首次构建多方言阿拉伯语常识数据集并提升模型性能。**

- **链接: [http://arxiv.org/pdf/2508.13130v1](http://arxiv.org/pdf/2508.13130v1)**

> **作者:** Kareem Elozeiri; Mervat Abassy; Preslav Nakov; Yuxia Wang
>
> **摘要:** Commonsense validation evaluates whether a sentence aligns with everyday human understanding, a critical capability for developing robust natural language understanding systems. While substantial progress has been made in English, the task remains underexplored in Arabic, particularly given its rich linguistic diversity. Existing Arabic resources have primarily focused on Modern Standard Arabic (MSA), leaving regional dialects underrepresented despite their prevalence in spoken contexts. To bridge this gap, we present two key contributions: (i) we introduce MuDRiC, an extended Arabic commonsense dataset incorporating multiple dialects, and (ii) a novel method adapting Graph Convolutional Networks (GCNs) to Arabic commonsense reasoning, which enhances semantic relationship modeling for improved commonsense validation. Our experimental results demonstrate that this approach achieves superior performance in Arabic commonsense validation. Our work enhances Arabic natural language understanding by providing both a foundational dataset and a novel method for handling its complex variations. To the best of our knowledge, we release the first Arabic multi-dialect commonsense reasoning dataset.
>
---
#### [new 077] Every 28 Days the AI Dreams of Soft Skin and Burning Stars: Scaffolding AI Agents with Hormones and Emotions
- **分类: cs.CL; cs.AI; cs.MA**

- **简介: 论文提出将激素周期模拟嵌入大语言模型，通过生物节律调节AI的情绪与认知状态，缓解框架问题。实验显示模型表现随激素水平波动，揭示了性别偏见在语言模型中的隐含机制。**

- **链接: [http://arxiv.org/pdf/2508.11829v1](http://arxiv.org/pdf/2508.11829v1)**

> **作者:** Leigh Levinson; Christopher J. Agostino
>
> **备注:** 9 pages, 1 figure, submitted to NeurIPS Creative AI track
>
> **摘要:** Despite significant advances, AI systems struggle with the frame problem: determining what information is contextually relevant from an exponentially large possibility space. We hypothesize that biological rhythms, particularly hormonal cycles, serve as natural relevance filters that could address this fundamental challenge. We develop a framework that embeds simulated menstrual and circadian cycles into Large Language Models through system prompts generated from periodic functions modeling key hormones including estrogen, testosterone, and cortisol. Across multiple state-of-the-art models, linguistic analysis reveals emotional and stylistic variations that track biological phases; sadness peaks during menstruation while happiness dominates ovulation and circadian patterns show morning optimism transitioning to nocturnal introspection. Benchmarking on SQuAD, MMLU, Hellaswag, and AI2-ARC demonstrates subtle but consistent performance variations aligning with biological expectations, including optimal function in moderate rather than extreme hormonal ranges. This methodology provides a novel approach to contextual AI while revealing how societal biases regarding gender and biology are embedded within language models.
>
---
#### [new 078] MedKGent: A Large Language Model Agent Framework for Constructing Temporally Evolving Medical Knowledge Graph
- **分类: cs.CL; cs.AI**

- **简介: 论文提出MedKGent框架，用于构建随时间演化的医学知识图谱。解决现有方法忽视知识动态性和不确定性的问题，通过两个代理Agent从海量文献中提取并整合高置信度三元组，提升知识准确性与实用性。**

- **链接: [http://arxiv.org/pdf/2508.12393v1](http://arxiv.org/pdf/2508.12393v1)**

> **作者:** Duzhen Zhang; Zixiao Wang; Zhong-Zhi Li; Yahan Yu; Shuncheng Jia; Jiahua Dong; Haotian Xu; Xing Wu; Yingying Zhang; Tielin Zhang; Jie Yang; Xiuying Chen; Le Song
>
> **摘要:** The rapid expansion of medical literature presents growing challenges for structuring and integrating domain knowledge at scale. Knowledge Graphs (KGs) offer a promising solution by enabling efficient retrieval, automated reasoning, and knowledge discovery. However, current KG construction methods often rely on supervised pipelines with limited generalizability or naively aggregate outputs from Large Language Models (LLMs), treating biomedical corpora as static and ignoring the temporal dynamics and contextual uncertainty of evolving knowledge. To address these limitations, we introduce MedKGent, a LLM agent framework for constructing temporally evolving medical KGs. Leveraging over 10 million PubMed abstracts published between 1975 and 2023, we simulate the emergence of biomedical knowledge via a fine-grained daily time series. MedKGent incrementally builds the KG in a day-by-day manner using two specialized agents powered by the Qwen2.5-32B-Instruct model. The Extractor Agent identifies knowledge triples and assigns confidence scores via sampling-based estimation, which are used to filter low-confidence extractions and inform downstream processing. The Constructor Agent incrementally integrates the retained triples into a temporally evolving graph, guided by confidence scores and timestamps to reinforce recurring knowledge and resolve conflicts. The resulting KG contains 156,275 entities and 2,971,384 relational triples. Quality assessments by two SOTA LLMs and three domain experts demonstrate an accuracy approaching 90\%, with strong inter-rater agreement. To evaluate downstream utility, we conduct RAG across seven medical question answering benchmarks using five leading LLMs, consistently observing significant improvements over non-augmented baselines. Case studies further demonstrate the KG's value in literature-based drug repurposing via confidence-aware causal inference.
>
---
#### [new 079] Mitigating Hallucinations in Large Language Models via Causal Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文提出CDCR-SFT框架，通过显式构建因果DAG并推理来减少大语言模型的逻辑幻觉。任务为因果推理与幻觉抑制，解决现有方法无法建模变量间因果关系的问题。在8个任务上验证有效性，显著提升因果推理准确率并降低幻觉。**

- **链接: [http://arxiv.org/pdf/2508.12495v1](http://arxiv.org/pdf/2508.12495v1)**

> **作者:** Yuangang Li; Yiqing Shen; Yi Nian; Jiechao Gao; Ziyi Wang; Chenxiao Yu; Shawn Li; Jie Wang; Xiyang Hu; Yue Zhao
>
> **摘要:** Large language models (LLMs) exhibit logically inconsistent hallucinations that appear coherent yet violate reasoning principles, with recent research suggesting an inverse relationship between causal reasoning capabilities and such hallucinations. However, existing reasoning approaches in LLMs, such as Chain-of-Thought (CoT) and its graph-based variants, operate at the linguistic token level rather than modeling the underlying causal relationships between variables, lacking the ability to represent conditional independencies or satisfy causal identification assumptions. To bridge this gap, we introduce causal-DAG construction and reasoning (CDCR-SFT), a supervised fine-tuning framework that trains LLMs to explicitly construct variable-level directed acyclic graph (DAG) and then perform reasoning over it. Moreover, we present a dataset comprising 25,368 samples (CausalDR), where each sample includes an input question, explicit causal DAG, graph-based reasoning trace, and validated answer. Experiments on four LLMs across eight tasks show that CDCR-SFT improves the causal reasoning capability with the state-of-the-art 95.33% accuracy on CLADDER (surpassing human performance of 94.8% for the first time) and reduces the hallucination on HaluEval with 10% improvements. It demonstrates that explicit causal structure modeling in LLMs can effectively mitigate logical inconsistencies in LLM outputs. Code is available at https://github.com/MrLYG/CDCR-SFT.
>
---
#### [new 080] The Self-Execution Benchmark: Measuring LLMs' Attempts to Overcome Their Lack of Self-Execution
- **分类: cs.CL; cs.AI**

- **简介: 论文提出Self-Execution Benchmark，评估大语言模型预测自身输出属性的能力。旨在解决LLM缺乏自执行能力的问题，发现模型表现不佳且规模提升不保证性能改善，揭示其对自身行为推理的局限性。**

- **链接: [http://arxiv.org/pdf/2508.12277v1](http://arxiv.org/pdf/2508.12277v1)**

> **作者:** Elon Ezra; Ariel Weizman; Amos Azaria
>
> **备注:** 11 pages, 9 figures
>
> **摘要:** Large language models (LLMs) are commonly evaluated on tasks that test their knowledge or reasoning abilities. In this paper, we explore a different type of evaluation: whether an LLM can predict aspects of its own responses. Since LLMs lack the ability to execute themselves, we introduce the Self-Execution Benchmark, which measures a model's ability to anticipate properties of its output, such as whether a question will be difficult for it, whether it will refuse to answer, or what kinds of associations it is likely to produce. Our experiments show that models generally perform poorly on this benchmark, and that increased model size or capability does not consistently lead to better performance. These results suggest a fundamental limitation in how LLMs represent and reason about their own behavior.
>
---
#### [new 081] CarelessWhisper: Turning Whisper into a Causal Streaming Model
- **分类: cs.CL; cs.LG; cs.SD; eess.AS**

- **简介: 论文将非因果的Whisper模型改造为低延迟流式ASR模型，解决其无法实时处理语音的问题。通过LoRA微调和弱对齐数据训练因果编码器，并设计优化推理机制，实现高效低延迟语音识别与词级时间戳提取。**

- **链接: [http://arxiv.org/pdf/2508.12301v1](http://arxiv.org/pdf/2508.12301v1)**

> **作者:** Tomer Krichli; Bhiksha Raj; Joseph Keshet
>
> **备注:** 17 pages, 7 Figures, This work has been submitted to the IEEE for possible publication
>
> **摘要:** Automatic Speech Recognition (ASR) has seen remarkable progress, with models like OpenAI Whisper and NVIDIA Canary achieving state-of-the-art (SOTA) performance in offline transcription. However, these models are not designed for streaming (online or real-time) transcription, due to limitations in their architecture and training methodology. We propose a method to turn the transformer encoder-decoder model into a low-latency streaming model that is careless about future context. We present an analysis explaining why it is not straightforward to convert an encoder-decoder transformer to a low-latency streaming model. Our proposed method modifies the existing (non-causal) encoder to a causal encoder by fine-tuning both the encoder and decoder using Low-Rank Adaptation (LoRA) and a weakly aligned dataset. We then propose an updated inference mechanism that utilizes the fine-tune causal encoder and decoder to yield greedy and beam-search decoding, and is shown to be locally optimal. Experiments on low-latency chunk sizes (less than 300 msec) show that our fine-tuned model outperforms existing non-fine-tuned streaming approaches in most cases, while using a lower complexity. Additionally, we observe that our training process yields better alignment, enabling a simple method for extracting word-level timestamps. We release our training and inference code, along with the fine-tuned models, to support further research and development in streaming ASR.
>
---
#### [new 082] Sparse Attention across Multiple-context KV Cache
- **分类: cs.LG; cs.CL**

- **简介: 论文提出SamKV，解决多上下文KV缓存稀疏注意力问题，通过考虑上下文间互补信息进行局部重计算，在不损失精度前提下将序列长度压缩至15%，显著提升多上下文检索增强生成的推理效率。**

- **链接: [http://arxiv.org/pdf/2508.11661v1](http://arxiv.org/pdf/2508.11661v1)**

> **作者:** Ziyi Cao; Qingyi Si; Jingbin Zhang; Bingquan Liu
>
> **摘要:** Large language models face significant cost challenges in long-sequence inference. To address this, reusing historical Key-Value (KV) Cache for improved inference efficiency has become a mainstream approach. Recent advances further enhance throughput by sparse attention mechanisms to select the most relevant KV Cache, thereby reducing sequence length. However, such techniques are limited to single-context scenarios, where historical KV Cache is computed sequentially with causal-attention dependencies. In retrieval-augmented generation (RAG) scenarios, where retrieved documents as context are unknown beforehand, each document's KV Cache is computed and stored independently (termed multiple-context KV Cache), lacking cross-attention between contexts. This renders existing methods ineffective. Although prior work partially recomputes multiple-context KV Cache to mitigate accuracy loss from missing cross-attention, it requires retaining all KV Cache throughout, failing to reduce memory overhead. This paper presents SamKV, the first exploration of attention sparsification for multiple-context KV Cache. Specifically, SamKV takes into account the complementary information of other contexts when sparsifying one context, and then locally recomputes the sparsified information. Experiments demonstrate that our method compresses sequence length to 15% without accuracy degradation compared with full-recompuation baselines, significantly boosting throughput in multi-context RAG scenarios.
>
---
#### [new 083] Non-Iterative Symbolic-Aided Chain-of-Thought for Logical Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 论文提出Symbolic-Aided Chain-of-Thought方法，通过在提示中引入轻量符号结构，提升大语言模型在逻辑推理任务中的透明度与准确性，尤其在复杂约束场景下优于传统CoT。**

- **链接: [http://arxiv.org/pdf/2508.12425v1](http://arxiv.org/pdf/2508.12425v1)**

> **作者:** Phuong Minh Nguyen; Tien Huu Dang; Naoya Inoue
>
> **摘要:** This work introduces Symbolic-Aided Chain-of-Thought (CoT), an improved approach to standard CoT, for logical reasoning in large language models (LLMs). The key idea is to integrate lightweight symbolic representations into few-shot prompts, structuring the inference steps with a consistent strategy to make reasoning patterns more explicit within a non-iterative reasoning process. By incorporating these symbolic structures, our method preserves the generalizability of standard prompting techniques while enhancing the transparency, interpretability, and analyzability of LLM logical reasoning. Extensive experiments on four well-known logical reasoning benchmarks -- ProofWriter, FOLIO, ProntoQA, and LogicalDeduction, which cover diverse reasoning scenarios -- demonstrate the effectiveness of the proposed approach, particularly in complex reasoning tasks that require navigating multiple constraints or rules. Notably, Symbolic-Aided CoT consistently improves LLMs' reasoning capabilities across various model sizes and significantly outperforms conventional CoT on three out of four datasets, ProofWriter, ProntoQA, and LogicalDeduction.
>
---
#### [new 084] VideoAVE: A Multi-Attribute Video-to-Text Attribute Value Extraction Dataset and Benchmark Models
- **分类: cs.CV; cs.CL**

- **简介: 论文提出VideoAVE，首个面向电商的视频到文本属性值抽取数据集，覆盖14领域172属性。解决现有数据集缺乏视频支持与多样性的难题，引入CLIP-MoE过滤系统提升质量，并建立基准评估多种视觉语言模型，揭示视频A VE仍具挑战性。**

- **链接: [http://arxiv.org/pdf/2508.11801v1](http://arxiv.org/pdf/2508.11801v1)**

> **作者:** Ming Cheng; Tong Wu; Jiazhen Hu; Jiaying Gong; Hoda Eldardiry
>
> **备注:** 5 pages, 2 figures, 5 tables, accepted in CIKM 2025
>
> **摘要:** Attribute Value Extraction (AVE) is important for structuring product information in e-commerce. However, existing AVE datasets are primarily limited to text-to-text or image-to-text settings, lacking support for product videos, diverse attribute coverage, and public availability. To address these gaps, we introduce VideoAVE, the first publicly available video-to-text e-commerce AVE dataset across 14 different domains and covering 172 unique attributes. To ensure data quality, we propose a post-hoc CLIP-based Mixture of Experts filtering system (CLIP-MoE) to remove the mismatched video-product pairs, resulting in a refined dataset of 224k training data and 25k evaluation data. In order to evaluate the usability of the dataset, we further establish a comprehensive benchmark by evaluating several state-of-the-art video vision language models (VLMs) under both attribute-conditioned value prediction and open attribute-value pair extraction tasks. Our results analysis reveals that video-to-text AVE remains a challenging problem, particularly in open settings, and there is still room for developing more advanced VLMs capable of leveraging effective temporal information. The dataset and benchmark code for VideoAVE are available at: https://github.com/gjiaying/VideoAVE
>
---
#### [new 085] Learning to Steer: Input-dependent Steering for Multimodal LLMs
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 论文提出L2S方法，通过输入依赖的线性偏移实现对多模态大模型的细粒度引导，解决静态 steering 在安全性和准确性上的不足。**

- **链接: [http://arxiv.org/pdf/2508.12815v1](http://arxiv.org/pdf/2508.12815v1)**

> **作者:** Jayneel Parekh; Pegah Khayatan; Mustafa Shukor; Arnaud Dapogny; Alasdair Newson; Matthieu Cord
>
> **摘要:** Steering has emerged as a practical approach to enable post-hoc guidance of LLMs towards enforcing a specific behavior. However, it remains largely underexplored for multimodal LLMs (MLLMs); furthermore, existing steering techniques, such as mean steering, rely on a single steering vector, applied independently of the input query. This paradigm faces limitations when the desired behavior is dependent on the example at hand. For example, a safe answer may consist in abstaining from answering when asked for an illegal activity, or may point to external resources or consultation with an expert when asked about medical advice. In this paper, we investigate a fine-grained steering that uses an input-specific linear shift. This shift is computed using contrastive input-specific prompting. However, the input-specific prompts required for this approach are not known at test time. Therefore, we propose to train a small auxiliary module to predict the input-specific steering vector. Our approach, dubbed as L2S (Learn-to-Steer), demonstrates that it reduces hallucinations and enforces safety in MLLMs, outperforming other static baselines.
>
---
#### [new 086] Adversarial Attacks on VQA-NLE: Exposing and Alleviating Inconsistencies in Visual Question Answering Explanations
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 论文研究视觉问答中自然语言解释（VQA-NLE）的不一致性问题。针对现有模型生成矛盾或虚假解释的缺陷，提出图像和文本双重对抗攻击策略，并引入外部知识增强防御方法，提升模型鲁棒性与可靠性。**

- **链接: [http://arxiv.org/pdf/2508.12430v1](http://arxiv.org/pdf/2508.12430v1)**

> **作者:** Yahsin Yeh; Yilun Wu; Bokai Ruan; Honghan Shuai
>
> **摘要:** Natural language explanations in visual question answering (VQA-NLE) aim to make black-box models more transparent by elucidating their decision-making processes. However, we find that existing VQA-NLE systems can produce inconsistent explanations and reach conclusions without genuinely understanding the underlying context, exposing weaknesses in either their inference pipeline or explanation-generation mechanism. To highlight these vulnerabilities, we not only leverage an existing adversarial strategy to perturb questions but also propose a novel strategy that minimally alters images to induce contradictory or spurious outputs. We further introduce a mitigation method that leverages external knowledge to alleviate these inconsistencies, thereby bolstering model robustness. Extensive evaluations on two standard benchmarks and two widely used VQA-NLE models underscore the effectiveness of our attacks and the potential of knowledge-based defenses, ultimately revealing pressing security and reliability concerns in current VQA-NLE systems.
>
---
#### [new 087] PC-Sampler: Position-Aware Calibration of Decoding Bias in Masked Diffusion Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文针对掩码扩散模型（MDMs）在序列生成中因解码策略导致的性能瓶颈问题，提出PC-Sampler方法。通过位置感知权重和校准置信度，提升全局轨迹控制与信息量最大化，显著改善早期生成质量，使MDMs性能超越现有策略10%以上。**

- **链接: [http://arxiv.org/pdf/2508.13021v1](http://arxiv.org/pdf/2508.13021v1)**

> **作者:** Pengcheng Huang; Shuhao Liu; Zhenghao Liu; Yukun Yan; Shuo Wang; Zulong Chen; Tong Xiao
>
> **备注:** 17 pages,13 figures
>
> **摘要:** Recent advances in masked diffusion models (MDMs) have established them as powerful non-autoregressive alternatives for sequence generation. Nevertheless, our preliminary experiments reveal that the generation quality of MDMs is still highly sensitive to the choice of decoding strategy. In particular, widely adopted uncertainty-based samplers suffer from two key limitations: a lack of global trajectory control and a pronounced bias toward trivial tokens in the early stages of decoding. These shortcomings restrict the full potential of MDMs. In this work, we introduce Position-Aware Confidence-Calibrated Sampling (PC-Sampler), a novel decoding strategy that unifies global trajectory planning with content-aware informativeness maximization. PC-Sampler incorporates a position-aware weighting mechanism to regulate the decoding path and a calibrated confidence score to suppress the premature selection of trivial tokens. Extensive experiments on three advanced MDMs across seven challenging benchmarks-including logical reasoning and planning tasks-demonstrate that PC-Sampler consistently outperforms existing MDM decoding strategies by more than 10% on average, significantly narrowing the performance gap with state-of-the-art autoregressive models. All codes are available at https://github.com/NEUIR/PC-Sampler.
>
---
#### [new 088] Has GPT-5 Achieved Spatial Intelligence? An Empirical Study
- **分类: cs.CV; cs.CL; cs.LG; cs.MM; cs.RO**

- **简介: 该论文研究多模态模型的空间智能，旨在评估GPT-5在空间理解与推理上的能力。通过构建任务分类体系并测试8个基准，发现GPT-5虽强但仍未达人类水平，且高端模型在难题上无明显优势。**

- **链接: [http://arxiv.org/pdf/2508.13142v1](http://arxiv.org/pdf/2508.13142v1)**

> **作者:** Zhongang Cai; Yubo Wang; Qingping Sun; Ruisi Wang; Chenyang Gu; Wanqi Yin; Zhiqian Lin; Zhitao Yang; Chen Wei; Xuanke Shi; Kewang Deng; Xiaoyang Han; Zukai Chen; Jiaqi Li; Xiangyu Fan; Hanming Deng; Lewei Lu; Bo Li; Ziwei Liu; Quan Wang; Dahua Lin; Lei Yang
>
> **摘要:** Multi-modal models have achieved remarkable progress in recent years. Nevertheless, they continue to exhibit notable limitations in spatial understanding and reasoning, which are fundamental capabilities to achieving artificial general intelligence. With the recent release of GPT-5, allegedly the most powerful AI model to date, it is timely to examine where the leading models stand on the path toward spatial intelligence. First, we propose a comprehensive taxonomy of spatial tasks that unifies existing benchmarks and discuss the challenges in ensuring fair evaluation. We then evaluate state-of-the-art proprietary and open-source models on eight key benchmarks, at a cost exceeding one billion total tokens. Our empirical study reveals that (1) GPT-5 demonstrates unprecedented strength in spatial intelligence, yet (2) still falls short of human performance across a broad spectrum of tasks. Moreover, we (3) identify the more challenging spatial intelligence problems for multi-modal models, and (4) proprietary models do not exhibit a decisive advantage when facing the most difficult problems. In addition, we conduct a qualitative evaluation across a diverse set of scenarios that are intuitive for humans yet fail even the most advanced multi-modal models.
>
---
#### [new 089] E3RG: Building Explicit Emotion-driven Empathetic Response Generation System with Multimodal Large Language Model
- **分类: cs.AI; cs.CL; cs.CV; cs.HC; cs.MM**

- **简介: 论文提出E3RG系统，解决多模态情感响应生成任务中情感内容处理与身份一致性难题。通过分解为情感理解、记忆检索和响应生成三部分，利用多模态大模型实现无需额外训练的自然、一致的情感回应。**

- **链接: [http://arxiv.org/pdf/2508.12854v1](http://arxiv.org/pdf/2508.12854v1)**

> **作者:** Ronghao Lin; Shuai Shen; Weipeng Hu; Qiaolin He; Aolin Xiong; Li Huang; Haifeng Hu; Yap-peng Tan
>
> **备注:** Accepted at ACM MM 2025 Grand Challenge
>
> **摘要:** Multimodal Empathetic Response Generation (MERG) is crucial for building emotionally intelligent human-computer interactions. Although large language models (LLMs) have improved text-based ERG, challenges remain in handling multimodal emotional content and maintaining identity consistency. Thus, we propose E3RG, an Explicit Emotion-driven Empathetic Response Generation System based on multimodal LLMs which decomposes MERG task into three parts: multimodal empathy understanding, empathy memory retrieval, and multimodal response generation. By integrating advanced expressive speech and video generative models, E3RG delivers natural, emotionally rich, and identity-consistent responses without extra training. Experiments validate the superiority of our system on both zero-shot and few-shot settings, securing Top-1 position in the Avatar-based Multimodal Empathy Challenge on ACM MM 25. Our code is available at https://github.com/RH-Lin/E3RG.
>
---
#### [new 090] Code Vulnerability Detection Across Different Programming Languages with AI Models
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 论文属于代码漏洞检测任务，旨在解决传统静态分析工具误报率高、难以识别上下文相关漏洞的问题。作者利用CodeBERT等AI模型，在多语言代码数据集上进行微调与集成学习，实现高精度漏洞检测，验证了AI方法在不同语言和漏洞类型上的泛化能力。**

- **链接: [http://arxiv.org/pdf/2508.11710v1](http://arxiv.org/pdf/2508.11710v1)**

> **作者:** Hael Abdulhakim Ali Humran; Ferdi Sonmez
>
> **摘要:** Security vulnerabilities present in a code that has been written in diverse programming languages are among the most critical yet complicated aspects of source code to detect. Static analysis tools based on rule-based patterns usually do not work well at detecting the context-dependent bugs and lead to high false positive rates. Recent developments in artificial intelligence, specifically the use of transformer-based models like CodeBERT and CodeLlama, provide light to this problem, as they show potential in finding such flaws better. This paper presents the implementations of these models on various datasets of code vulnerability, showing how off-the-shelf models can successfully produce predictive capacity in models through dynamic fine-tuning of the models on vulnerable and safe code fragments. The methodology comprises the gathering of the dataset, normalization of the language, fine-tuning of the model, and incorporation of ensemble learning and explainable AI. Experiments show that a well-trained CodeBERT can be as good as or even better than some existing static analyzers in terms of accuracy greater than 97%. Further study has indicated that although language models can achieve close-to-perfect recall, the precision can decrease. A solution to this is given by hybrid models and validation procedures, which will reduce false positives. According to the results, the AI-based solutions generalize to different programming languages and classes of vulnerability. Nevertheless, robustness, interpretability, and deployment readiness are still being developed. The results illustrate the probabilities that AI will enhance the trustworthiness in the usability and scalability of machine-learning-based detectors of vulnerabilities.
>
---
#### [new 091] Optimizing Token Choice for Code Watermarking: A RL Approach
- **分类: cs.CR; cs.CL; cs.LG**

- **简介: 论文提出CodeTracer框架，用强化学习优化代码水印的token选择，在保持代码功能前提下提升可检测性，解决LLM生成代码的水印难题。**

- **链接: [http://arxiv.org/pdf/2508.11925v1](http://arxiv.org/pdf/2508.11925v1)**

> **作者:** Zhimeng Guo; Huaisheng Zhu; Siyuan Xu; Hangfan Zhang; Teng Xiao; Minhao Cheng
>
> **备注:** 18 pages, 3 figures
>
> **摘要:** The need for detecting LLM-generated code necessitates watermarking systems capable of operating within its highly structured and syntactically constrained environment. To address this, we introduce CodeTracer, an innovative adaptive code watermarking framework underpinned by a novel reinforcement learning training paradigm. At its core, CodeTracer features a policy-driven approach that utilizes a parameterized model to intelligently bias token choices during next-token prediction. This strategy ensures that embedded watermarks maintain code functionality while exhibiting subtle yet statistically detectable deviations from typical token distributions. To facilitate policy learning, we devise a comprehensive reward system that seamlessly integrates execution feedback with watermark embedding signals, balancing process-level and outcome-level rewards. Additionally, we employ Gumbel Top-k reparameterization to enable gradient-based optimization of discrete watermarking decisions. Extensive comparative evaluations demonstrate CodeTracer's significant superiority over state-of-the-art baselines in both watermark detectability and the preservation of generated code's functionality.
>
---
#### [new 092] Generative Medical Event Models Improve with Scale
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 论文提出CoMET，一种基于大规模医疗事件数据的生成式基础模型，通过自回归预测患者健康轨迹，在无需任务微调的情况下，在78项临床任务中表现优异，解决了个性化医疗与通用建模难题。**

- **链接: [http://arxiv.org/pdf/2508.12104v1](http://arxiv.org/pdf/2508.12104v1)**

> **作者:** Shane Waxler; Paul Blazek; Davis White; Daniel Sneider; Kevin Chung; Mani Nagarathnam; Patrick Williams; Hank Voeller; Karen Wong; Matthew Swanhorst; Sheng Zhang; Naoto Usuyama; Cliff Wong; Tristan Naumann; Hoifung Poon; Andrew Loza; Daniella Meeker; Seth Hain; Rahul Shah
>
> **摘要:** Realizing personalized medicine at scale calls for methods that distill insights from longitudinal patient journeys, which can be viewed as a sequence of medical events. Foundation models pretrained on large-scale medical event data represent a promising direction for scaling real-world evidence generation and generalizing to diverse downstream tasks. Using Epic Cosmos, a dataset with medical events from de-identified longitudinal health records for 16.3 billion encounters over 300 million unique patient records from 310 health systems, we introduce the Cosmos Medical Event Transformer ( CoMET) models, a family of decoder-only transformer models pretrained on 118 million patients representing 115 billion discrete medical events (151 billion tokens). We present the largest scaling-law study for medical event data, establishing a methodology for pretraining and revealing power-law scaling relationships for compute, tokens, and model size. Based on this, we pretrained a series of compute-optimal models with up to 1 billion parameters. Conditioned on a patient's real-world history, CoMET autoregressively generates the next medical event, simulating patient health timelines. We studied 78 real-world tasks, including diagnosis prediction, disease prognosis, and healthcare operations. Remarkably for a foundation model with generic pretraining and simulation-based inference, CoMET generally outperformed or matched task-specific supervised models on these tasks, without requiring task-specific fine-tuning or few-shot examples. CoMET's predictive power consistently improves as the model and pretraining scale. Our results show that CoMET, a generative medical event foundation model, can effectively capture complex clinical dynamics, providing an extensible and generalizable framework to support clinical decision-making, streamline healthcare operations, and improve patient outcomes.
>
---
#### [new 093] DynamixSFT: Dynamic Mixture Optimization of Instruction Tuning Collections
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 论文提出DynamixSFT，用于动态优化指令微调数据集混合比例。针对多数据集混合时难以平衡的问题，将其建模为多臂赌博机问题，通过优先级缩放的Boltzmann探索和轻量级奖励机制自适应调整采样概率，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2508.12116v1](http://arxiv.org/pdf/2508.12116v1)**

> **作者:** Haebin Shin; Lei Ji; Xiao Liu; Zhiwei Yu; Qi Chen; Yeyun Gong
>
> **摘要:** As numerous instruction-tuning datasets continue to emerge during the post-training stage, dynamically balancing and optimizing their mixtures has become a critical challenge. To address this, we propose DynamixSFT, a dynamic and automated method for instruction-tuning dataset mixture optimization. We formulate the problem as a multi-armed bandit setup and introduce a Prior-scaled Boltzmann Exploration that softly anchors the updated sampling distribution to the original dataset proportions, thereby preserving the inherent diversity and coverage of the collection. Sampling probabilities are updated using a lightweight 1-Step Look-ahead Reward, reflecting how much the dataset contributes to improving the model's performance at its current state. When applied to the Tulu-v2-mixture collection comprising 16 instruction-tuning datasets, DynamixSFT achieves up to a 2.2% performance improvement across 10 benchmarks. Furthermore, we provide a comprehensive analysis and visualizations to offer deeper insights into the adaptive dynamics of our method.
>
---
#### [new 094] CHBench: A Cognitive Hierarchy Benchmark for Evaluating Strategic Reasoning Capability of LLMs
- **分类: cs.AI; cs.CL; cs.HC**

- **简介: 论文提出CHBench框架，用于评估大语言模型的战略推理能力。针对现有方法依赖易变的效用指标的问题，该研究基于行为经济学的认知层级模型，通过十五个博弈游戏测试六种LLM，验证其推理层级的一致性，并分析对话与记忆机制的影响。**

- **链接: [http://arxiv.org/pdf/2508.11944v1](http://arxiv.org/pdf/2508.11944v1)**

> **作者:** Hongtao Liu; Zhicheng Du; Zihe Wang; Weiran Shen
>
> **摘要:** Game-playing ability serves as an indicator for evaluating the strategic reasoning capability of large language models (LLMs). While most existing studies rely on utility performance metrics, which are not robust enough due to variations in opponent behavior and game structure. To address this limitation, we propose \textbf{Cognitive Hierarchy Benchmark (CHBench)}, a novel evaluation framework inspired by the cognitive hierarchy models from behavioral economics. We hypothesize that agents have bounded rationality -- different agents behave at varying reasoning depths/levels. We evaluate LLMs' strategic reasoning through a three-phase systematic framework, utilizing behavioral data from six state-of-the-art LLMs across fifteen carefully selected normal-form games. Experiments show that LLMs exhibit consistent strategic reasoning levels across diverse opponents, confirming the framework's robustness and generalization capability. We also analyze the effects of two key mechanisms (Chat Mechanism and Memory Mechanism) on strategic reasoning performance. Results indicate that the Chat Mechanism significantly degrades strategic reasoning, whereas the Memory Mechanism enhances it. These insights position CHBench as a promising tool for evaluating LLM capabilities, with significant potential for future research and practical applications.
>
---
#### [new 095] Reinforcement Learning with Rubric Anchors
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 论文提出基于评分标准（rubric）的强化学习方法，解决开放任务中难以获取自动奖励的问题。通过构建超1万条人工与模型协作设计的评分标准，实现对主观输出的自动评分与风格控制，在人文类任务上显著提升模型表现，同时保持推理能力。**

- **链接: [http://arxiv.org/pdf/2508.12790v1](http://arxiv.org/pdf/2508.12790v1)**

> **作者:** Zenan Huang; Yihong Zhuang; Guoshan Lu; Zeyu Qin; Haokai Xu; Tianyu Zhao; Ru Peng; Jiaqi Hu; Zhanming Shen; Xiaomeng Hu; Xijun Gu; Peiyi Tu; Jiaxin Liu; Wenyu Chen; Yuzhuo Fu; Zhiting Fan; Yanmei Gu; Yuanyuan Wang; Zhengkai Yang; Jianguo Li; Junbo Zhao
>
> **备注:** technical report
>
> **摘要:** Reinforcement Learning from Verifiable Rewards (RLVR) has emerged as a powerful paradigm for enhancing Large Language Models (LLMs), exemplified by the success of OpenAI's o-series. In RLVR, rewards are derived from verifiable signals-such as passing unit tests in code generation or matching correct answers in mathematical reasoning. While effective, this requirement largely confines RLVR to domains with automatically checkable outcomes. To overcome this, we extend the RLVR paradigm to open-ended tasks by integrating rubric-based rewards, where carefully designed rubrics serve as structured, model-interpretable criteria for automatic scoring of subjective outputs. We construct, to our knowledge, the largest rubric reward system to date, with over 10,000 rubrics from humans, LLMs, or a hybrid human-LLM collaboration. Implementing rubric-based RL is challenging; we tackle these issues with a clear framework and present an open-sourced Qwen-30B-A3B model with notable gains: 1) With only 5K+ samples, our system improves by +5.2% on open-ended benchmarks (especially humanities), outperforming a 671B DeepSeek-V3 model by +2.4%, while preserving general and reasoning abilities. 2) Our method provides fine-grained stylistic control, using rubrics as anchors to mitigate the "AI-like" tone and produce more human-like, expressive responses. We share key lessons in rubric construction, data selection, and training, and discuss limitations and future releases.
>
---
#### [new 096] VimoRAG: Video-based Retrieval-augmented 3D Motion Generation for Motion Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 论文提出VimoRAG框架，解决运动大模型因数据有限导致的域外问题。通过检索视频中的人体动作信号增强3D运动生成，设计了基于Gemini的视频检索机制和双对齐训练方法，提升生成效果。**

- **链接: [http://arxiv.org/pdf/2508.12081v1](http://arxiv.org/pdf/2508.12081v1)**

> **作者:** Haidong Xu; Guangwei Xu; Zhedong Zheng; Xiatian Zhu; Wei Ji; Xiangtai Li; Ruijie Guo; Meishan Zhang; Min zhang; Hao Fei
>
> **备注:** 20 pages,13 figures
>
> **摘要:** This paper introduces VimoRAG, a novel video-based retrieval-augmented motion generation framework for motion large language models (LLMs). As motion LLMs face severe out-of-domain/out-of-vocabulary issues due to limited annotated data, VimoRAG leverages large-scale in-the-wild video databases to enhance 3D motion generation by retrieving relevant 2D human motion signals. While video-based motion RAG is nontrivial, we address two key bottlenecks: (1) developing an effective motion-centered video retrieval model that distinguishes human poses and actions, and (2) mitigating the issue of error propagation caused by suboptimal retrieval results. We design the Gemini Motion Video Retriever mechanism and the Motion-centric Dual-alignment DPO Trainer, enabling effective retrieval and generation processes. Experimental results show that VimoRAG significantly boosts the performance of motion LLMs constrained to text-only input.
>
---
#### [new 097] EVTP-IVS: Effective Visual Token Pruning For Unifying Instruction Visual Segmentation In Multi-Modal Large Language Models
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; eess.IV**

- **简介: 论文针对多模态大模型中的指令视觉分割任务，提出EVTP-IVS方法，通过空间感知的视觉令牌剪枝技术，在仅用20%令牌的情况下实现最高5倍加速，同时保持分割精度。**

- **链接: [http://arxiv.org/pdf/2508.11886v1](http://arxiv.org/pdf/2508.11886v1)**

> **作者:** Wenhui Zhu; Xiwen Chen; Zhipeng Wang; Shao Tang; Sayan Ghosh; Xuanzhao Dong; Rajat Koner; Yalin Wang
>
> **摘要:** Instructed Visual Segmentation (IVS) tasks require segmenting objects in images or videos based on natural language instructions. While recent multimodal large language models (MLLMs) have achieved strong performance on IVS, their inference cost remains a major bottleneck, particularly in video. We empirically analyze visual token sampling in MLLMs and observe a strong correlation between subset token coverage and segmentation performance. This motivates our design of a simple and effective token pruning method that selects a compact yet spatially representative subset of tokens to accelerate inference. In this paper, we introduce a novel visual token pruning method for IVS, called EVTP-IV, which builds upon the k-center by integrating spatial information to ensure better coverage. We further provide an information-theoretic analysis to support our design. Experiments on standard IVS benchmarks show that our method achieves up to 5X speed-up on video tasks and 3.5X on image tasks, while maintaining comparable accuracy using only 20% of the tokens. Our method also consistently outperforms state-of-the-art pruning baselines under varying pruning ratios.
>
---
#### [new 098] Vision-G1: Towards General Vision Language Reasoning with Multi-Domain Data Curation
- **分类: cs.CV; cs.CL**

- **简介: 论文提出Vision-G1，解决视觉语言模型在多领域推理能力不足的问题。通过构建涵盖8个维度的多源数据集，采用基于影响函数的数据筛选与难度过滤策略，结合多轮强化学习训练，显著提升模型在各类视觉推理任务上的表现。**

- **链接: [http://arxiv.org/pdf/2508.12680v1](http://arxiv.org/pdf/2508.12680v1)**

> **作者:** Yuheng Zha; Kun Zhou; Yujia Wu; Yushu Wang; Jie Feng; Zhi Xu; Shibo Hao; Zhengzhong Liu; Eric P. Xing; Zhiting Hu
>
> **摘要:** Despite their success, current training pipelines for reasoning VLMs focus on a limited range of tasks, such as mathematical and logical reasoning. As a result, these models face difficulties in generalizing their reasoning capabilities to a wide range of domains, primarily due to the scarcity of readily available and verifiable reward data beyond these narrowly defined areas. Moreover, integrating data from multiple domains is challenging, as the compatibility between domain-specific datasets remains uncertain. To address these limitations, we build a comprehensive RL-ready visual reasoning dataset from 46 data sources across 8 dimensions, covering a wide range of tasks such as infographic, mathematical, spatial, cross-image, graphic user interface, medical, common sense and general science. We propose an influence function based data selection and difficulty based filtering strategy to identify high-quality training samples from this dataset. Subsequently, we train the VLM, referred to as Vision-G1, using multi-round RL with a data curriculum to iteratively improve its visual reasoning capabilities. Our model achieves state-of-the-art performance across various visual reasoning benchmarks, outperforming similar-sized VLMs and even proprietary models like GPT-4o and Gemini-1.5 Flash. The model, code and dataset are publicly available at https://github.com/yuh-zha/Vision-G1.
>
---
#### [new 099] Maximum Score Routing For Mixture-of-Experts
- **分类: cs.LG; cs.CL**

- **简介: 论文提出MaxScore路由机制，解决MoE模型中专家容量约束导致的token丢弃和硬件效率低的问题。通过建模为最小成本最大流问题并引入SoftTopk算子，实现高效负载均衡与计算优化。**

- **链接: [http://arxiv.org/pdf/2508.12801v1](http://arxiv.org/pdf/2508.12801v1)**

> **作者:** Bowen Dong; Yilong Fan; Yutao Sun; Zhenyu Li; Tengyu Pan; Xun Zhou; Jianyong Wang
>
> **摘要:** Routing networks in sparsely activated mixture-of-experts (MoE) dynamically allocate input tokens to top-k experts through differentiable sparse transformations, enabling scalable model capacity while preserving computational efficiency. Traditional MoE networks impose an expert capacity constraint to ensure GPU-friendly computation. However, this leads to token dropping when capacity is saturated and results in low hardware efficiency due to padding in underutilized experts. Removing the capacity constraint, in turn, compromises load balancing and computational efficiency. To address these issues, we propose Maximum Score Routing ($\mathbf{MaxScore}$), a novel MoE routing paradigm that models routing as a minimum-cost maximum-flow problem and integrates a SoftTopk operator. MaxScore resolves the fundamental limitations of iterative rerouting and optimal transport formulations, achieving lower training losses and higher evaluation scores at equivalent FLOPs compared to both constrained and unconstrained baselines. Implementation details and experimental configurations can be obtained from $\href{https://github.com/dongbw18/MaxScore.git}{MaxScore}$.
>
---
#### [new 100] Labels or Input? Rethinking Augmentation in Multimodal Hate Detection
- **分类: cs.CV; cs.AI; cs.CL; cs.CY; cs.MM; I.2.7; I.2.10**

- **简介: 论文聚焦多模态仇恨检测任务，针对文本与图像间隐性关联导致的误判问题，提出两种改进方法：优化提示结构提升模型鲁棒性，设计数据增强管道生成中立反事实样本以减少虚假关联，从而提高分类器泛化能力。**

- **链接: [http://arxiv.org/pdf/2508.11808v1](http://arxiv.org/pdf/2508.11808v1)**

> **作者:** Sahajpreet Singh; Rongxin Ouyang; Subhayan Mukerjee; Kokil Jaidka
>
> **备注:** 13 pages, 2 figures, 7 tables
>
> **摘要:** The modern web is saturated with multimodal content, intensifying the challenge of detecting hateful memes, where harmful intent is often conveyed through subtle interactions between text and image under the guise of humor or satire. While recent advances in Vision-Language Models (VLMs) show promise, these models lack support for fine-grained supervision and remain susceptible to implicit hate speech. In this paper, we present a dual-pronged approach to improve multimodal hate detection. First, we propose a prompt optimization framework that systematically varies prompt structure, supervision granularity, and training modality. We show that prompt design and label scaling both influence performance, with structured prompts improving robustness even in small models, and InternVL2 achieving the best F1-scores across binary and scaled settings. Second, we introduce a multimodal data augmentation pipeline that generates 2,479 counterfactually neutral memes by isolating and rewriting the hateful modality. This pipeline, powered by a multi-agent LLM-VLM setup, successfully reduces spurious correlations and improves classifier generalization. Our approaches inspire new directions for building synthetic data to train robust and fair vision-language models. Our findings demonstrate that prompt structure and data composition are as critical as model size, and that targeted augmentation can support more trustworthy and context-sensitive hate detection.
>
---
#### [new 101] Using Natural Language for Human-Robot Collaboration in the Real World
- **分类: cs.RO; cs.AI; cs.CL**

- **简介: 论文探讨如何利用大语言模型（LLM）提升机器人在现实世界中理解自然语言的能力，以实现与人类更有效的协作。针对语言理解挑战，提出基于认知代理的AI系统架构，并通过实验验证其可行性，旨在构建集成语言能力的机器人助手。**

- **链接: [http://arxiv.org/pdf/2508.11759v1](http://arxiv.org/pdf/2508.11759v1)**

> **作者:** Peter Lindes; Kaoutar Skiker
>
> **备注:** 34 pages, 11 figures, 5 tables. Submitted for publication (2026) in W.F. Lawless, Ranjeev Mittu, Shannon P. McGrarry, & Marco Brambilla (Eds.), Generative AI Risks and Benefits within Human-Machine Teams, Elsevier, Chapter 6
>
> **摘要:** We have a vision of a day when autonomous robots can collaborate with humans as assistants in performing complex tasks in the physical world. This vision includes that the robots will have the ability to communicate with their human collaborators using language that is natural to the humans. Traditional Interactive Task Learning (ITL) systems have some of this ability, but the language they can understand is very limited. The advent of large language models (LLMs) provides an opportunity to greatly improve the language understanding of robots, yet integrating the language abilities of LLMs with robots that operate in the real physical world is a challenging problem. In this chapter we first review briefly a few commercial robot products that work closely with humans, and discuss how they could be much better collaborators with robust language abilities. We then explore how an AI system with a cognitive agent that controls a physical robot at its core, interacts with both a human and an LLM, and accumulates situational knowledge through its experiences, can be a possible approach to reach that vision. We focus on three specific challenges of having the robot understand natural language, and present a simple proof-of-concept experiment using ChatGPT for each. Finally, we discuss what it will take to turn these simple experiments into an operational system where LLM-assisted language understanding is a part of an integrated robotic assistant that uses language to collaborate with humans.
>
---
#### [new 102] Assessing Representation Stability for Transformer Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出Representation Stability（RS）框架，用于检测对抗文本攻击。通过测量关键词掩码对嵌入表示的影响，实现模型无关的高效检测，在多个数据集和攻击类型下表现优异，且无需重新训练模型。**

- **链接: [http://arxiv.org/pdf/2508.11667v1](http://arxiv.org/pdf/2508.11667v1)**

> **作者:** Bryan E. Tuck; Rakesh M. Verma
>
> **备注:** 19 pages, 19 figures, 8 tables. Code available at https://github.com/ReDASers/representation-stability
>
> **摘要:** Adversarial text attacks remain a persistent threat to transformer models, yet existing defenses are typically attack-specific or require costly model retraining. We introduce Representation Stability (RS), a model-agnostic detection framework that identifies adversarial examples by measuring how embedding representations change when important words are masked. RS first ranks words using importance heuristics, then measures embedding sensitivity to masking top-k critical words, and processes the resulting patterns with a BiLSTM detector. Experiments show that adversarially perturbed words exhibit disproportionately high masking sensitivity compared to naturally important words. Across three datasets, three attack types, and two victim models, RS achieves over 88% detection accuracy and demonstrates competitive performance compared to existing state-of-the-art methods, often at lower computational cost. Using Normalized Discounted Cumulative Gain (NDCG) to measure perturbation identification quality, we reveal that gradient-based ranking outperforms attention and random selection approaches, with identification quality correlating with detection performance for word-level attacks. RS also generalizes well to unseen datasets, attacks, and models without retraining, providing a practical solution for adversarial text detection.
>
---
#### [new 103] An LLM + ASP Workflow for Joint Entity-Relation Extraction
- **分类: cs.AI; cs.CL; I.2.7; F.4.1**

- **简介: 论文提出一种结合大语言模型（LLM）与答案集编程（ASP）的通用工作流，用于联合实体关系抽取（JERE）。该方法无需大量标注数据，能高效融入领域知识，在少量训练数据下显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.12611v1](http://arxiv.org/pdf/2508.12611v1)**

> **作者:** Trang Tran; Trung Hoang Le; Huiping Cao; Tran Cao Son
>
> **备注:** 13 pages, 1 figure, Accepted as Technical Communication, 41st International Conference on Logic Programming
>
> **摘要:** Joint entity-relation extraction (JERE) identifies both entities and their relationships simultaneously. Traditional machine-learning based approaches to performing this task require a large corpus of annotated data and lack the ability to easily incorporate domain specific information in the construction of the model. Therefore, creating a model for JERE is often labor intensive, time consuming, and elaboration intolerant. In this paper, we propose harnessing the capabilities of generative pretrained large language models (LLMs) and the knowledge representation and reasoning capabilities of Answer Set Programming (ASP) to perform JERE. We present a generic workflow for JERE using LLMs and ASP. The workflow is generic in the sense that it can be applied for JERE in any domain. It takes advantage of LLM's capability in natural language understanding in that it works directly with unannotated text. It exploits the elaboration tolerant feature of ASP in that no modification of its core program is required when additional domain specific knowledge, in the form of type specifications, is found and needs to be used. We demonstrate the usefulness of the proposed workflow through experiments with limited training data on three well-known benchmarks for JERE. The results of our experiments show that the LLM + ASP workflow is better than state-of-the-art JERE systems in several categories with only 10\% of training data. It is able to achieve a 2.5 times (35\% over 15\%) improvement in the Relation Extraction task for the SciERC corpus, one of the most difficult benchmarks.
>
---
#### [new 104] TCUQ: Single-Pass Uncertainty Quantification from Temporal Consistency with Streaming Conformal Calibration for TinyML
- **分类: cs.LG; cs.CL**

- **简介: 论文提出TCUQ，一种用于TinyML的单 pass 不确定性量化方法，利用时间一致性与流式校准实现低资源消耗的在线监控，解决设备端不确定性评估难题。**

- **链接: [http://arxiv.org/pdf/2508.12905v1](http://arxiv.org/pdf/2508.12905v1)**

> **作者:** Ismail Lamaakal; Chaymae Yahyati; Khalid El Makkaoui; Ibrahim Ouahbi; Yassine Maleh
>
> **摘要:** We introduce TCUQ, a single pass, label free uncertainty monitor for streaming TinyML that converts short horizon temporal consistency captured via lightweight signals on posteriors and features into a calibrated risk score with an O(W ) ring buffer and O(1) per step updates. A streaming conformal layer turns this score into a budgeted accept/abstain rule, yielding calibrated behavior without online labels or extra forward passes. On microcontrollers, TCUQ fits comfortably on kilobyte scale devices and reduces footprint and latency versus early exit and deep ensembles (typically about 50 to 60% smaller and about 30 to 45% faster), while methods of similar accuracy often run out of memory. Under corrupted in distribution streams, TCUQ improves accuracy drop detection by 3 to 7 AUPRC points and reaches up to 0.86 AUPRC at high severities; for failure detection it attains up to 0.92 AUROC. These results show that temporal consistency, coupled with streaming conformal calibration, provides a practical and resource efficient foundation for on device monitoring in TinyML.
>
---
#### [new 105] LARC: Towards Human-level Constrained Retrosynthesis Planning through an Agentic Framework
- **分类: cs.AI; cs.CL**

- **简介: 论文提出LARC框架，用于约束逆合成规划任务，通过代理评估机制提升大语言模型的决策准确性。该方法在48个任务中达到72.9%成功率，接近人类专家水平，显著优于基线模型。**

- **链接: [http://arxiv.org/pdf/2508.11860v1](http://arxiv.org/pdf/2508.11860v1)**

> **作者:** Frazier N. Baker; Daniel Adu-Ampratwum; Reza Averly; Botao Yu; Huan Sun; Xia Ning
>
> **备注:** 24 pages, 5 figures
>
> **摘要:** Large language model (LLM) agent evaluators leverage specialized tools to ground the rational decision-making of LLMs, making them well-suited to aid in scientific discoveries, such as constrained retrosynthesis planning. Constrained retrosynthesis planning is an essential, yet challenging, process within chemistry for identifying synthetic routes from commercially available starting materials to desired target molecules, subject to practical constraints. Here, we present LARC, the first LLM-based Agentic framework for Retrosynthesis planning under Constraints. LARC incorporates agentic constraint evaluation, through an Agent-as-a-Judge, directly into the retrosynthesis planning process, using agentic feedback grounded in tool-based reasoning to guide and constrain route generation. We rigorously evaluate LARC on a carefully curated set of 48 constrained retrosynthesis planning tasks across 3 constraint types. LARC achieves a 72.9% success rate on these tasks, vastly outperforming LLM baselines and approaching human expert-level success in substantially less time. The LARC framework is extensible, and serves as a first step towards an effective agentic tool or a co-scientist to human experts for constrained retrosynthesis.
>
---
#### [new 106] Where to Start Alignment? Diffusion Large Language Model May Demand a Distinct Position
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 论文研究扩散语言模型（dLLMs）的安全对齐问题，提出MOSA方法，通过强化学习对中间token进行安全对齐，提升安全性并保持模型性能。**

- **链接: [http://arxiv.org/pdf/2508.12398v1](http://arxiv.org/pdf/2508.12398v1)**

> **作者:** Zhixin Xie; Xurui Song; Jun Luo
>
> **摘要:** Diffusion Large Language Models (dLLMs) have recently emerged as a competitive non-autoregressive paradigm due to their unique training and inference approach. However, there is currently a lack of safety study on this novel architecture. In this paper, we present the first analysis of dLLMs' safety performance and propose a novel safety alignment method tailored to their unique generation characteristics. Specifically, we identify a critical asymmetry between the defender and attacker in terms of security. For the defender, we reveal that the middle tokens of the response, rather than the initial ones, are more critical to the overall safety of dLLM outputs; this seems to suggest that aligning middle tokens can be more beneficial to the defender. The attacker, on the contrary, may have limited power to manipulate middle tokens, as we find dLLMs have a strong tendency towards a sequential generation order in practice, forcing the attack to meet this distribution and diverting it from influencing the critical middle tokens. Building on this asymmetry, we introduce Middle-tOken Safety Alignment (MOSA), a novel method that directly aligns the model's middle generation with safe refusals exploiting reinforcement learning. We implement MOSA and compare its security performance against eight attack methods on two benchmarks. We also test the utility of MOSA-aligned dLLM on coding, math, and general reasoning. The results strongly prove the superiority of MOSA.
>
---
#### [new 107] Insight Rumors: A Novel Textual Rumor Locating and Marking Model Leveraging Att_BiMamba2 Network
- **分类: cs.SI; cs.CL**

- **简介: 论文提出Insight Rumors模型，解决谣言检测中仅分类不定位的问题。通过Att_BiMamba2网络增强特征表示，并设计定位标记模块结合CRF，实现谣言内容的精准定位与标注。**

- **链接: [http://arxiv.org/pdf/2508.12574v1](http://arxiv.org/pdf/2508.12574v1)**

> **作者:** Bin Ma; Yifei Zhang; Yongjin Xian; Qi Li; Linna Zhou; Gongxun Miao
>
> **摘要:** With the development of social media networks, rumor detection models have attracted more and more attention. Whereas, these models primarily focus on classifying contexts as rumors or not, lacking the capability to locate and mark specific rumor content. To address this limitation, this paper proposes a novel rumor detection model named Insight Rumors to locate and mark rumor content within textual data. Specifically, we propose the Bidirectional Mamba2 Network with Dot-Product Attention (Att_BiMamba2), a network that constructs a bidirectional Mamba2 model and applies dot-product attention to weight and combine the outputs from both directions, thereby enhancing the representation of high-dimensional rumor features. Simultaneously, a Rumor Locating and Marking module is designed to locate and mark rumors. The module constructs a skip-connection network to project high-dimensional rumor features onto low-dimensional label features. Moreover, Conditional Random Fields (CRF) is employed to impose strong constraints on the output label features, ensuring accurate rumor content location. Additionally, a labeled dataset for rumor locating and marking is constructed, with the effectiveness of the proposed model is evaluated through comprehensive experiments. Extensive experiments indicate that the proposed scheme not only detects rumors accurately but also locates and marks them in context precisely, outperforming state-of-the-art schemes that can only discriminate rumors roughly.
>
---
#### [new 108] SNAP-UQ: Self-supervised Next-Activation Prediction for Single-Pass Uncertainty in TinyML
- **分类: cs.LG; cs.CL**

- **简介: 论文提出SNAP-UQ，一种单次前向传播的不确定性估计方法，用于资源受限的TinyML场景。它通过预测下一层激活统计量来评估风险，无需标签、额外层或重复计算，在极低内存和延迟下实现高效故障检测与准确性提升。**

- **链接: [http://arxiv.org/pdf/2508.12907v1](http://arxiv.org/pdf/2508.12907v1)**

> **作者:** Ismail Lamaakal; Chaymae Yahyati; Khalid El Makkaoui; Ibrahim Ouahbi; Yassine Maleh
>
> **摘要:** We introduce \textbf{SNAP-UQ}, a single-pass, label-free uncertainty method for TinyML that estimates risk from \emph{depth-wise next-activation prediction}: tiny int8 heads forecast the statistics of the next layer from a compressed view of the previous one, and a lightweight monotone mapper turns the resulting surprisal into an actionable score. The design requires no temporal buffers, auxiliary exits, or repeated forward passes, and adds only a few tens of kilobytes to MCU deployments. Across vision and audio backbones, SNAP-UQ consistently reduces flash and latency relative to early-exit and deep ensembles (typically $\sim$40--60\% smaller and $\sim$25--35\% faster), with competing methods of similar accuracy often exceeding memory limits. In corrupted streams it improves accuracy-drop detection by several AUPRC points and maintains strong failure detection (AUROC $\approx$0.9) in a single pass. Grounding uncertainty in layer-to-layer dynamics yields a practical, resource-efficient basis for on-device monitoring in TinyML.
>
---
#### [new 109] Ovis2.5 Technical Report
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 论文提出Ovis2.5，一种用于视觉感知和多模态推理的模型。解决固定分辨率图像处理导致细节丢失的问题，通过原生分辨率视觉Transformer和反思式推理提升准确率。采用五阶段训练流程和高效训练技术，释放两个开源模型，在多项任务中达到SOTA。**

- **链接: [http://arxiv.org/pdf/2508.11737v1](http://arxiv.org/pdf/2508.11737v1)**

> **作者:** Shiyin Lu; Yang Li; Yu Xia; Yuwei Hu; Shanshan Zhao; Yanqing Ma; Zhichao Wei; Yinglun Li; Lunhao Duan; Jianshan Zhao; Yuxuan Han; Haijun Li; Wanying Chen; Junke Tang; Chengkun Hou; Zhixing Du; Tianli Zhou; Wenjie Zhang; Huping Ding; Jiahe Li; Wen Li; Gui Hu; Yiliang Gu; Siran Yang; Jiamang Wang; Hailong Sun; Yibo Wang; Hui Sun; Jinlong Huang; Yuping He; Shengze Shi; Weihong Zhang; Guodong Zheng; Junpeng Jiang; Sensen Gao; Yi-Feng Wu; Sijia Chen; Yuhui Chen; Qing-Guo Chen; Zhao Xu; Weihua Luo; Kaifu Zhang
>
> **摘要:** We present Ovis2.5, a successor to Ovis2 designed for native-resolution visual perception and strong multimodal reasoning. Ovis2.5 integrates a native-resolution vision transformer that processes images at their native, variable resolutions, avoiding the degradation from fixed-resolution tiling and preserving both fine detail and global layout -- crucial for visually dense content like complex charts. To strengthen reasoning, we train the model to move beyond linear chain-of-thought and perform reflection -- including self-checking and revision. This advanced capability is exposed as an optional "thinking mode" at inference time, allowing users to trade latency for enhanced accuracy on difficult inputs. The model is trained via a comprehensive five-phase curriculum that progressively builds its skills. The process begins with foundational visual and multimodal pretraining, advances through large-scale instruction tuning, and culminates in alignment and reasoning enhancement using DPO and GRPO. To scale these upgrades efficiently, we employ multimodal data packing and hybrid parallelism, yielding a significant end-to-end speedup. We release two open-source models: Ovis2.5-9B and Ovis2.5-2B. The latter continues the "small model, big performance" philosophy of Ovis2, making it ideal for resource-constrained, on-device scenarios. On the OpenCompass multimodal leaderboard, Ovis2.5-9B averages 78.3, marking a substantial improvement over its predecessor, Ovis2-8B, and achieving state-of-the-art results among open-source MLLMs in the sub-40B parameter range; Ovis2.5-2B scores 73.9, establishing SOTA for its size. Beyond aggregate scores, Ovis2.5 achieves leading results on STEM benchmarks, exhibits strong capabilities on grounding and video tasks, and achieves open-source SOTA at its scale for complex chart analysis.
>
---
#### [new 110] Mitigating Jailbreaks with Intent-Aware LLMs
- **分类: cs.CR; cs.CL**

- **简介: 论文提出Intent-FT方法，通过训练模型先识别指令意图来增强对越狱攻击的鲁棒性。解决LLM在安全与性能间的权衡问题，显著提升防御效果且不损害正常功能。**

- **链接: [http://arxiv.org/pdf/2508.12072v1](http://arxiv.org/pdf/2508.12072v1)**

> **作者:** Wei Jie Yeo; Ranjan Satapathy; Erik Cambria
>
> **摘要:** Despite extensive safety-tuning, large language models (LLMs) remain vulnerable to jailbreak attacks via adversarially crafted instructions, reflecting a persistent trade-off between safety and task performance. In this work, we propose Intent-FT, a simple and lightweight fine-tuning approach that explicitly trains LLMs to infer the underlying intent of an instruction before responding. By fine-tuning on a targeted set of adversarial instructions, Intent-FT enables LLMs to generalize intent deduction to unseen attacks, thereby substantially improving their robustness. We comprehensively evaluate both parametric and non-parametric attacks across open-source and proprietary models, considering harmfulness from attacks, utility, over-refusal, and impact against white-box threats. Empirically, Intent-FT consistently mitigates all evaluated attack categories, with no single attack exceeding a 50\% success rate -- whereas existing defenses remain only partially effective. Importantly, our method preserves the model's general capabilities and reduces excessive refusals on benign instructions containing superficially harmful keywords. Furthermore, models trained with Intent-FT accurately identify hidden harmful intent in adversarial attacks, and these learned intentions can be effectively transferred to enhance vanilla model defenses.
>
---
#### [new 111] Bridging Human and LLM Judgments: Understanding and Narrowing the Gap
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **简介: 论文提出Bridge框架，用于统一建模人类与大语言模型（LLM）的评价差异。任务是缩小LLM作为评判者时与人类判断的系统性偏差。工作包括构建统计模型、设计高效算法，并在多个基准上验证其提升一致性与揭示差距的能力。**

- **链接: [http://arxiv.org/pdf/2508.12792v1](http://arxiv.org/pdf/2508.12792v1)**

> **作者:** Felipe Maia Polo; Xinhe Wang; Mikhail Yurochkin; Gongjun Xu; Moulinath Banerjee; Yuekai Sun
>
> **摘要:** Large language models are increasingly used as judges (LLM-as-a-judge) to evaluate model outputs at scale, but their assessments often diverge systematically from human judgments. We present Bridge, a unified statistical framework that explicitly bridges human and LLM evaluations under both absolute scoring and pairwise comparison paradigms. Bridge posits a latent human preference score for each prompt-response pair and models LLM deviations as linear transformations of covariates that capture sources of discrepancies. This offers a simple and principled framework for refining LLM ratings and characterizing systematic discrepancies between humans and LLMs. We provide an efficient fitting algorithm with asymptotic guarantees for statistical inference. Using six LLM judges and two benchmarks (BigGen Bench and Chatbot Arena), Bridge achieves higher agreement with human ratings (accuracy, calibration, and KL divergence) and exposes systematic human-LLM gaps.
>
---
## 更新

#### [replaced 001] Towards Multimodal Social Conversations with Robots: Using Vision-Language Models
- **分类: cs.RO; cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2507.19196v2](http://arxiv.org/pdf/2507.19196v2)**

> **作者:** Ruben Janssens; Tony Belpaeme
>
> **备注:** Accepted at the workshop "Human - Foundation Models Interaction: A Focus On Multimodal Information" (FoMo-HRI) at IEEE RO-MAN 2025 (Camera-ready version)
>
> **摘要:** Large language models have given social robots the ability to autonomously engage in open-domain conversations. However, they are still missing a fundamental social skill: making use of the multiple modalities that carry social interactions. While previous work has focused on task-oriented interactions that require referencing the environment or specific phenomena in social interactions such as dialogue breakdowns, we outline the overall needs of a multimodal system for social conversations with robots. We then argue that vision-language models are able to process this wide range of visual information in a sufficiently general manner for autonomous social robots. We describe how to adapt them to this setting, which technical challenges remain, and briefly discuss evaluation practices.
>
---
#### [replaced 002] Translation in the Wild
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.23548v2](http://arxiv.org/pdf/2505.23548v2)**

> **作者:** Yuri Balashov
>
> **备注:** 4 figures
>
> **摘要:** Large Language Models (LLMs) excel in translation among other things, demonstrating competitive performance for many language pairs in zero- and few-shot settings. But unlike dedicated neural machine translation models, LLMs are not trained on any translation-related objective. What explains their remarkable translation abilities? Are these abilities grounded in "incidental bilingualism" (Briakou et al. 2023) in training data? Does instruction tuning contribute to it? Are LLMs capable of aligning and leveraging semantically identical or similar monolingual contents from different corners of the internet that are unlikely to fit in a single context window? I offer some reflections on this topic, informed by recent studies and growing user experience. My working hypothesis is that LLMs' translation abilities originate in two different types of pre-training data that may be internalized by the models in different ways. I discuss the prospects for testing the "duality" hypothesis empirically and its implications for reconceptualizing translation, human and machine, in the age of deep learning.
>
---
#### [replaced 003] OrthoRank: Token Selection via Sink Token Orthogonality for Efficient LLM inference
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.03865v2](http://arxiv.org/pdf/2507.03865v2)**

> **作者:** Seungjun Shin; Jaehoon Oh; Dokwan Oh
>
> **备注:** ICML 2025 (final version)
>
> **摘要:** Attention mechanisms are central to the success of large language models (LLMs), enabling them to capture intricate token dependencies and implicitly assign importance to each token. Recent studies have revealed the sink token, which receives disproportionately high attention despite their limited semantic role. In this paper, we first expand the relationship between the sink token and other tokens, moving beyond attention to explore their similarity in hidden states, considering the layer depth. We observe that as the layers get deeper, the cosine similarity between the normalized hidden states of the sink token and those of other tokens increases, and that the normalized hidden states of the sink token exhibit negligible changes. These imply that other tokens consistently are directed toward the sink token throughout the layers. Next, we propose a dynamic token selection method, called OrthoRank, using these findings to select important tokens. Specifically, in a certain layer, we define token importance by the speed at which the token moves toward the sink token. This is converted into orthogonality with the sink token, meaning that tokens that are more orthogonal to the sink token are assigned greater importance. Finally, through extensive experiments, we demonstrated that our method results in lower perplexity and higher zero-shot accuracy compared to layer pruning methods at the same sparsity ratio with comparable throughput, while also achieving superior performance on LongBench.
>
---
#### [replaced 004] An Information-Theoretic Approach to Identifying Formulaic Clusters in Textual Data
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.07303v2](http://arxiv.org/pdf/2503.07303v2)**

> **作者:** Gideon Yoffe; Yair Segev; Barak Sober
>
> **摘要:** Texts, whether literary or historical, exhibit structural and stylistic patterns shaped by their purpose, authorship, and cultural context. Formulaic texts, characterized by repetition and constrained expression, tend to have lower variability in self-information compared to more dynamic compositions. Identifying such patterns in historical documents, particularly multi-author texts like the Hebrew Bible provides insights into their origins, purpose, and transmission. This study aims to identify formulaic clusters -- sections exhibiting systematic repetition and structural constraints -- by analyzing recurring phrases, syntactic structures, and stylistic markers. However, distinguishing formulaic from non-formulaic elements in an unsupervised manner presents a computational challenge, especially in high-dimensional textual spaces where patterns must be inferred without predefined labels. To address this, we develop an information-theoretic algorithm leveraging weighted self-information distributions to detect structured patterns in text, unlike covariance-based methods, which become unstable in small-sample, high-dimensional settings, our approach directly models variations in self-information to identify formulaicity. By extending classical discrete self-information measures with a continuous formulation based on differential self-information, our method remains applicable across different types of textual representations, including neural embeddings under Gaussian priors. Applied to hypothesized authorial divisions in the Hebrew Bible, our approach successfully isolates stylistic layers, providing a quantitative framework for textual stratification. This method enhances our ability to analyze compositional patterns, offering deeper insights into the literary and cultural evolution of texts shaped by complex authorship and editorial processes.
>
---
#### [replaced 005] Efficient Forward-Only Data Valuation for Pretrained LLMs and VLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.10180v2](http://arxiv.org/pdf/2508.10180v2)**

> **作者:** Wenlong Deng; Jiaming Zhang; Qi Zeng; Christos Thrampoulidis; Boying Gong; Xiaoxiao Li
>
> **摘要:** Quantifying the influence of individual training samples is essential for enhancing the transparency and accountability of large language models (LLMs) and vision-language models (VLMs). However, existing data valuation methods often rely on Hessian information or model retraining, making them computationally prohibitive for billion-parameter models. In this work, we introduce For-Value, a forward-only data valuation framework that enables scalable and efficient influence estimation for both LLMs and VLMs. By leveraging the rich representations of modern foundation models, For-Value computes influence scores using a simple closed-form expression based solely on a single forward pass, thereby eliminating the need for costly gradient computations. Our theoretical analysis demonstrates that For-Value accurately estimates per-sample influence by capturing alignment in hidden representations and prediction errors between training and validation samples. Extensive experiments show that For-Value matches or outperforms gradient-based baselines in identifying impactful fine-tuning examples and effectively detecting mislabeled data.
>
---
#### [replaced 006] Emoji Attack: Enhancing Jailbreak Attacks Against Judge LLM Detection
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.01077v5](http://arxiv.org/pdf/2411.01077v5)**

> **作者:** Zhipeng Wei; Yuqi Liu; N. Benjamin Erichson
>
> **摘要:** Jailbreaking techniques trick Large Language Models (LLMs) into producing restricted output, posing a potential threat. One line of defense is to use another LLM as a Judge to evaluate the harmfulness of generated text. However, we reveal that these Judge LLMs are vulnerable to token segmentation bias, an issue that arises when delimiters alter the tokenization process, splitting words into smaller sub-tokens. This alters the embeddings of the entire sequence, reducing detection accuracy and allowing harmful content to be misclassified as safe. In this paper, we introduce Emoji Attack, a novel strategy that amplifies existing jailbreak prompts by exploiting token segmentation bias. Our method leverages in-context learning to systematically insert emojis into text before it is evaluated by a Judge LLM, inducing embedding distortions that significantly lower the likelihood of detecting unsafe content. Unlike traditional delimiters, emojis also introduce semantic ambiguity, making them particularly effective in this attack. Through experiments on state-of-the-art Judge LLMs, we demonstrate that Emoji Attack substantially reduces the unsafe prediction rate, bypassing existing safeguards.
>
---
#### [replaced 007] Matrix-Driven Instant Review: Confident Detection and Reconstruction of LLM Plagiarism on PC
- **分类: cs.CL; math.PR**

- **链接: [http://arxiv.org/pdf/2508.06309v2](http://arxiv.org/pdf/2508.06309v2)**

> **作者:** Ruichong Zhang
>
> **备注:** The code is available at the same directory as the TeX source. Run `main_mdir.py` for details
>
> **摘要:** In recent years, concerns about intellectual property (IP) in large language models (LLMs) have grown significantly. Plagiarizing other LLMs (through direct weight copying, upcycling, pruning, or continual pretraining) and claiming authorship without properly attributing to the original license, is a serious misconduct that can lead to significant financial and reputational harm to the original developers. However, existing methods for detecting LLM plagiarism fall short in key areas. They fail to accurately reconstruct weight correspondences, lack the ability to compute statistical significance measures such as $p$-values, and may mistakenly flag models trained on similar data as being related. To address these limitations, we propose Matrix-Driven Instant Review (MDIR), a novel method that leverages matrix analysis and Large Deviation Theory. MDIR achieves accurate reconstruction of weight relationships, provides rigorous $p$-value estimation, and focuses exclusively on weight similarity without requiring full model inference. Experimental results demonstrate that MDIR reliably detects plagiarism even after extensive transformations, such as random permutations and continual pretraining with trillions of tokens. Moreover, all detections can be performed on a single PC within an hour, making MDIR both efficient and accessible.
>
---
#### [replaced 008] Explaining Large Language Models with gSMILE
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.21657v4](http://arxiv.org/pdf/2505.21657v4)**

> **作者:** Zeinab Dehghani; Mohammed Naveed Akram; Koorosh Aslansefat; Adil Khan; Yiannis Papadopoulos
>
> **摘要:** Large Language Models (LLMs) such as GPT, LLaMA, and Claude achieve remarkable performance in text generation but remain opaque in their decision-making processes, limiting trust and accountability in high-stakes applications. We present gSMILE (generative SMILE), a model-agnostic, perturbation-based framework for token-level interpretability in LLMs. Extending the SMILE methodology, gSMILE uses controlled prompt perturbations, Wasserstein distance metrics, and weighted linear surrogates to identify input tokens with the most significant impact on the output. This process enables the generation of intuitive heatmaps that visually highlight influential tokens and reasoning paths. We evaluate gSMILE across leading LLMs (OpenAI's gpt-3.5-turbo-instruct, Meta's LLaMA 3.1 Instruct Turbo, and Anthropic's Claude 2.1) using attribution fidelity, attribution consistency, attribution stability, attribution faithfulness, and attribution accuracy as metrics. Results show that gSMILE delivers reliable human-aligned attributions, with Claude 2.1 excelling in attention fidelity and GPT-3.5 achieving the highest output consistency. These findings demonstrate gSMILE's ability to balance model performance and interpretability, enabling more transparent and trustworthy AI systems.
>
---
#### [replaced 009] From Templates to Natural Language: Generalization Challenges in Instruction-Tuned LLMs for Spatial Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14425v2](http://arxiv.org/pdf/2505.14425v2)**

> **作者:** Chalamalasetti Kranti; Sherzod Hakimov; David Schlangen
>
> **备注:** 17 pages
>
> **摘要:** Instruction-tuned large language models (LLMs) have shown strong performance on a variety of tasks; however, generalizing from synthetic to human-authored instructions in grounded environments remains a challenge for them. In this work, we study generalization challenges in spatial grounding tasks where models interpret and translate instructions for building object arrangements on a $2.5$D grid. We fine-tune LLMs using only synthetic instructions and evaluate their performance on a benchmark dataset containing both synthetic and human-written instructions. Our results reveal that while models generalize well on simple tasks, their performance degrades significantly on more complex tasks. We present a detailed error analysis of the gaps in instruction generalization.
>
---
#### [replaced 010] Improving Text Style Transfer using Masked Diffusion Language Models with Inference-time Scaling
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.10995v2](http://arxiv.org/pdf/2508.10995v2)**

> **作者:** Tejomay Kishor Padole; Suyash P Awate; Pushpak Bhattacharyya
>
> **备注:** Accepted as a main conference submission in the European Conference on Artificial Intelligence (ECAI 2025)
>
> **摘要:** Masked diffusion language models (MDMs) have recently gained traction as a viable generative framework for natural language. This can be attributed to its scalability and ease of training compared to other diffusion model paradigms for discrete data, establishing itself as the state-of-the-art non-autoregressive generator for discrete data. Diffusion models, in general, have shown excellent ability to improve the generation quality by leveraging inference-time scaling either by increasing the number of denoising steps or by using external verifiers on top of the outputs of each step to guide the generation. In this work, we propose a verifier-based inference-time scaling method that aids in finding a better candidate generation during the denoising process of the MDM. Our experiments demonstrate the application of MDMs for standard text-style transfer tasks and establish MDMs as a better alternative to autoregressive language models. Additionally, we show that a simple soft-value-based verifier setup for MDMs using off-the-shelf pre-trained embedding models leads to significant gains in generation quality even when used on top of typical classifier-free guidance setups in the existing literature.
>
---
#### [replaced 011] SpectR: Dynamically Composing LM Experts with Spectral Routing
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.03454v2](http://arxiv.org/pdf/2504.03454v2)**

> **作者:** William Fleshman; Benjamin Van Durme
>
> **摘要:** Training large, general-purpose language models poses significant challenges. The growing availability of specialized expert models, fine-tuned from pretrained models for specific tasks or domains, offers a promising alternative. Leveraging the potential of these existing expert models in real-world applications requires effective methods to select or merge the models best suited for a given task. This paper introduces SPECTR, an approach for dynamically composing expert models at each time step during inference. Notably, our method requires no additional training and enables flexible, token- and layer-wise model combinations. Our experimental results demonstrate that SPECTR improves routing accuracy over alternative training-free methods, increasing task performance across expert domains.
>
---
#### [replaced 012] LGR2: Language Guided Reward Relabeling for Accelerating Hierarchical Reinforcement Learning
- **分类: cs.LG; cs.CL; cs.RO**

- **链接: [http://arxiv.org/pdf/2406.05881v4](http://arxiv.org/pdf/2406.05881v4)**

> **作者:** Utsav Singh; Pramit Bhattacharyya; Vinay P. Namboodiri
>
> **摘要:** Large language models (LLMs) have shown remarkable abilities in logical reasoning, in-context learning, and code generation. However, translating natural language instructions into effective robotic control policies remains a significant challenge, especially for tasks requiring long-horizon planning and operating under sparse reward conditions. Hierarchical Reinforcement Learning (HRL) provides a natural framework to address this challenge in robotics; however, it typically suffers from non-stationarity caused by the changing behavior of the lower-level policy during training, destabilizing higher-level policy learning. We introduce LGR2, a novel HRL framework that leverages LLMs to generate language-guided reward functions for the higher-level policy. By decoupling high-level reward generation from low-level policy changes, LGR2 fundamentally mitigates the non-stationarity problem in off-policy HRL, enabling stable and efficient learning. To further enhance sample efficiency in sparse environments, we integrate goal-conditioned hindsight experience relabeling. Extensive experiments across simulated and real-world robotic navigation and manipulation tasks demonstrate LGR2 outperforms both hierarchical and non-hierarchical baselines, achieving over 55% success rates on challenging tasks and robust transfer to real robots, without additional fine-tuning.
>
---
#### [replaced 013] Generalizable LLM Learning of Graph Synthetic Data with Post-training Alignment
- **分类: cs.LG; cs.CL; I.2.7**

- **链接: [http://arxiv.org/pdf/2506.00845v3](http://arxiv.org/pdf/2506.00845v3)**

> **作者:** Yizhuo Zhang; Heng Wang; Shangbin Feng; Zhaoxuan Tan; Xinyun Liu; Yulia Tsvetkov
>
> **备注:** 8 pages, 1 figures, 2 tables. Experimental code and results are publicly available at https://anonymous.4open.science/r/Graph_RL-BF08/readme.md
>
> **摘要:** Previous research has sought to enhance the graph reasoning capabilities of LLMs by supervised fine-tuning on synthetic graph data. While these led to specialized LLMs better at solving graph algorithm problems, we don't need LLMs for shortest path: we need generalization from synthetic graph data to real-world tasks with implicit graph structures. In this work, we propose to unlock generalizable learning of graph with post-training alignment with synthetic data. We first design solution-based and process-based rewards for synthetic graph problems: instead of rigid memorizing response patterns in direct fine-tuning, we posit that post-training alignment would help LLMs grasp the essentials underlying graph reasoning and alleviate overfitting on synthetic data. We employ post-training alignment algorithms such as GRPO and DPO, aligning both off-the-shelf LLMs and LLMs fine-tuned on synthetic graph data. We then compare them against existing settings on both in-domain synthetic tasks and out-of-domain real-world tasks with implicit graph structures such as multi-hop QA, structured planning, and more. Extensive experiments demonstrate that our post-training alignment recipe leads to statistically significant improvement on 5 datasets, with an average gain of 12.9% over baseline settings. Further analysis reveals that process-based rewards consistently outperform solution-based rewards on synthetic data but not on real-world tasks, and compositionality and explainable intermediate steps remains a critical challenge even after post-training alignment.
>
---
#### [replaced 014] S2Cap: A Benchmark and a Baseline for Singing Style Captioning
- **分类: cs.CL; cs.AI; cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2409.09866v3](http://arxiv.org/pdf/2409.09866v3)**

> **作者:** Hyunjong Ok; Jaeho Lee
>
> **备注:** CIKM 2025 Resource Paper
>
> **摘要:** Singing voices contain much richer information than common voices, including varied vocal and acoustic properties. However, current open-source audio-text datasets for singing voices capture only a narrow range of attributes and lack acoustic features, leading to limited utility towards downstream tasks, such as style captioning. To fill this gap, we formally define the singing style captioning task and present S2Cap, a dataset of singing voices with detailed descriptions covering diverse vocal, acoustic, and demographic characteristics. Using this dataset, we develop an efficient and straightforward baseline algorithm for singing style captioning. The dataset is available at https://zenodo.org/records/15673764.
>
---
#### [replaced 015] Neural Bandit Based Optimal LLM Selection for a Pipeline of Tasks
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.09958v2](http://arxiv.org/pdf/2508.09958v2)**

> **作者:** Baran Atalar; Eddie Zhang; Carlee Joe-Wong
>
> **备注:** Submitted to AAAI 2026
>
> **摘要:** With the increasing popularity of large language models (LLMs) for a variety of tasks, there has been a growing interest in strategies that can predict which out of a set of LLMs will yield a successful answer at low cost. This problem promises to become more and more relevant as providers like Microsoft allow users to easily create custom LLM "assistants" specialized to particular types of queries. However, some tasks (i.e., queries) may be too specialized and difficult for a single LLM to handle alone. These applications often benefit from breaking down the task into smaller subtasks, each of which can then be executed by a LLM expected to perform well on that specific subtask. For example, in extracting a diagnosis from medical records, one can first select an LLM to summarize the record, select another to validate the summary, and then select another, possibly different, LLM to extract the diagnosis from the summarized record. Unlike existing LLM selection or routing algorithms, this setting requires that we select a sequence of LLMs, with the output of each LLM feeding into the next and potentially influencing its success. Thus, unlike single LLM selection, the quality of each subtask's output directly affects the inputs, and hence the cost and success rate, of downstream LLMs, creating complex performance dependencies that must be learned and accounted for during selection. We propose a neural contextual bandit-based algorithm that trains neural networks that model LLM success on each subtask in an online manner, thus learning to guide the LLM selections for the different subtasks, even in the absence of historical LLM performance data. Experiments on telecommunications question answering and medical diagnosis prediction datasets illustrate the effectiveness of our proposed approach compared to other LLM selection algorithms.
>
---
#### [replaced 016] LLMs Are In-Context Bandit Reinforcement Learners
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.05362v3](http://arxiv.org/pdf/2410.05362v3)**

> **作者:** Giovanni Monea; Antoine Bosselut; Kianté Brantley; Yoav Artzi
>
> **备注:** Published at COLM 2025
>
> **摘要:** Large Language Models (LLMs) excel at in-context learning (ICL), a supervised learning technique that relies on adding annotated examples to the model context. We investigate a contextual bandit version of in-context reinforcement learning (ICRL), where models learn in-context, online, from external reward, instead of supervised data. We show that LLMs effectively demonstrate such learning, and provide a detailed study of the phenomena, experimenting with challenging classification tasks and models of sizes from 500M to 70B parameters. This includes identifying and addressing the instability of the process, demonstrating learning with both semantic and abstract labels, and showing scaling trends. Our findings highlight ICRL capabilities in LLMs, while also underscoring fundamental limitations in their implicit reasoning about errors.
>
---
#### [replaced 017] Hard Negative Contrastive Learning for Fine-Grained Geometric Understanding in Large Multimodal Models
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.20152v2](http://arxiv.org/pdf/2505.20152v2)**

> **作者:** Kai Sun; Yushi Bai; Zhen Yang; Jiajie Zhang; Ji Qi; Lei Hou; Juanzi Li
>
> **摘要:** Benefiting from contrastively trained visual encoders on large-scale natural scene images, Large Multimodal Models (LMMs) have achieved remarkable performance across various visual perception tasks. However, the inherent limitations of contrastive learning upon summarized descriptions fundamentally restrict the capabilities of models in meticulous reasoning, particularly in crucial scenarios of geometric problem-solving. To enhance geometric understanding, we propose a novel hard negative contrastive learning framework for the vision encoder, which combines image-based contrastive learning using generation-based hard negatives created by perturbing diagram generation code, and text-based contrastive learning using rule-based negatives derived from modified geometric descriptions and retrieval-based negatives selected based on caption similarity. We train CLIP using our hard negative learning method, namely MMCLIP (Multimodal Math CLIP), and subsequently train an LMM for geometric problem-solving. Experiments show that our trained model, MMGeoLM, significantly outperforms other open-source models on three geometric reasoning benchmarks. Even with a size of 7B, it can rival powerful closed-source models like GPT-4o. We further conduct ablation studies to analyze three key factors: hard negative types, the efficiency of image-based negatives, and training configurations. These analyses yield important insights into optimizing hard negative strategies for geometric reasoning tasks.
>
---
#### [replaced 018] Convert Language Model into a Value-based Strategic Planner
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.06987v5](http://arxiv.org/pdf/2505.06987v5)**

> **作者:** Xiaoyu Wang; Yue Zhao; Qingqing Gu; Zhonglin Jiang; Xiaokai Chen; Yong Chen; Luo Ji
>
> **备注:** 13 pages, 6 figures, ACL 2025 Industry Track
>
> **摘要:** Emotional support conversation (ESC) aims to alleviate the emotional distress of individuals through effective conversations. Although large language models (LLMs) have obtained remarkable progress on ESC, most of these studies might not define the diagram from the state model perspective, therefore providing a suboptimal solution for long-term satisfaction. To address such an issue, we leverage the Q-learning on LLMs, and propose a framework called straQ*. Our framework allows a plug-and-play LLM to bootstrap the planning during ESC, determine the optimal strategy based on long-term returns, and finally guide the LLM to response. Substantial experiments on ESC datasets suggest that straQ* outperforms many baselines, including direct inference, self-refine, chain of thought, finetuning, and finite state machines.
>
---
#### [replaced 019] Memory-Augmented Transformers: A Systematic Review from Neuroscience Principles to Enhanced Model Architectures
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.10824v2](http://arxiv.org/pdf/2508.10824v2)**

> **作者:** Parsa Omidi; Xingshuai Huang; Axel Laborieux; Bahareh Nikpour; Tianyu Shi; Armaghan Eshaghi
>
> **摘要:** Memory is fundamental to intelligence, enabling learning, reasoning, and adaptability across biological and artificial systems. While Transformer architectures excel at sequence modeling, they face critical limitations in long-range context retention, continual learning, and knowledge integration. This review presents a unified framework bridging neuroscience principles, including dynamic multi-timescale memory, selective attention, and consolidation, with engineering advances in Memory-Augmented Transformers. We organize recent progress through three taxonomic dimensions: functional objectives (context extension, reasoning, knowledge integration, adaptation), memory representations (parameter-encoded, state-based, explicit, hybrid), and integration mechanisms (attention fusion, gated control, associative retrieval). Our analysis of core memory operations (reading, writing, forgetting, and capacity management) reveals a shift from static caches toward adaptive, test-time learning systems. We identify persistent challenges in scalability and interference, alongside emerging solutions including hierarchical buffering and surprise-gated updates. This synthesis provides a roadmap toward cognitively-inspired, lifelong-learning Transformer architectures.
>
---
#### [replaced 020] StepTool: Enhancing Multi-Step Tool Usage in LLMs via Step-Grained Reinforcement Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.07745v4](http://arxiv.org/pdf/2410.07745v4)**

> **作者:** Yuanqing Yu; Zhefan Wang; Weizhi Ma; Shuai Wang; Chuhan Wu; Zhiqiang Guo; Min Zhang
>
> **备注:** Accepted by CIKM'25
>
> **摘要:** Despite their powerful text generation capabilities, large language models (LLMs) still struggle to effectively utilize external tools to solve complex tasks, a challenge known as tool learning. Existing methods primarily rely on supervised fine-tuning, treating tool learning as a text generation problem while overlooking the decision-making complexities inherent in multi-step contexts. In this work, we propose modeling tool learning as a dynamic decision-making process and introduce StepTool, a novel step-grained reinforcement learning framework that enhances LLMs' capabilities in multi-step tool use. StepTool comprises two key components: Step-grained Reward Shaping, which assigns rewards to each tool interaction based on its invocation success and contribution to task completion; and Step-grained Optimization, which applies policy gradient methods to optimize the model across multiple decision steps. Extensive experiments across diverse benchmarks show that StepTool consistently outperforms both SFT-based and RL-based baselines in terms of task Pass Rate and Recall of relevant tools. Furthermore, our analysis suggests that StepTool helps models discover new tool-use strategies rather than merely re-weighting prior knowledge. These results highlight the importance of fine-grained decision modeling in tool learning and establish StepTool as a general and robust solution for enhancing multi-step tool use in LLMs. Code and data are available at https://github.com/yuyq18/StepTool.
>
---
#### [replaced 021] Advancing AI-Scientist Understanding: Multi-Agent LLMs with Interpretable Physics Reasoning
- **分类: cs.AI; cs.CL; cs.HC; physics.comp-ph**

- **链接: [http://arxiv.org/pdf/2504.01911v2](http://arxiv.org/pdf/2504.01911v2)**

> **作者:** Yinggan Xu; Hana Kimlee; Yijia Xiao; Di Luo
>
> **备注:** ICML 2025 Workshop on MAS
>
> **摘要:** Large Language Models (LLMs) are playing an increasingly important role in physics research by assisting with symbolic manipulation, numerical computation, and scientific reasoning. However, ensuring the reliability, transparency, and interpretability of their outputs remains a major challenge. In this work, we introduce a novel multi-agent LLM physicist framework that fosters collaboration between AI and human scientists through three key modules: a reasoning module, an interpretation module, and an AI-scientist interaction module. Recognizing that effective physics reasoning demands logical rigor, quantitative accuracy, and alignment with established theoretical models, we propose an interpretation module that employs a team of specialized LLM agents-including summarizers, model builders, visualization tools, and testers-to systematically structure LLM outputs into transparent, physically grounded science models. A case study demonstrates that our approach significantly improves interpretability, enables systematic validation, and enhances human-AI collaboration in physics problem-solving and discovery. Our work bridges free-form LLM reasoning with interpretable, executable models for scientific analysis, enabling more transparent and verifiable AI-augmented research.
>
---
#### [replaced 022] Fast Controlled Generation from Language Models with Adaptive Weighted Rejection Sampling
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.05410v2](http://arxiv.org/pdf/2504.05410v2)**

> **作者:** Benjamin Lipkin; Benjamin LeBrun; Jacob Hoover Vigly; João Loula; David R. MacIver; Li Du; Jason Eisner; Ryan Cotterell; Vikash Mansinghka; Timothy J. O'Donnell; Alexander K. Lew; Tim Vieira
>
> **备注:** COLM 2025
>
> **摘要:** The dominant approach to generating from language models subject to some constraint is locally constrained decoding (LCD), incrementally sampling tokens at each time step such that the constraint is never violated. Typically, this is achieved through token masking: looping over the vocabulary and excluding non-conforming tokens. There are two important problems with this approach. (i) Evaluating the constraint on every token can be prohibitively expensive -- LM vocabularies often exceed $100,000$ tokens. (ii) LCD can distort the global distribution over strings, sampling tokens based only on local information, even if they lead down dead-end paths. This work introduces a new algorithm that addresses both these problems. First, to avoid evaluating a constraint on the full vocabulary at each step of generation, we propose an adaptive rejection sampling algorithm that typically requires orders of magnitude fewer constraint evaluations. Second, we show how this algorithm can be extended to produce low-variance, unbiased estimates of importance weights at a very small additional cost -- estimates that can be soundly used within previously proposed sequential Monte Carlo algorithms to correct for the myopic behavior of local constraint enforcement. Through extensive empirical evaluation in text-to-SQL, molecular synthesis, goal inference, pattern matching, and JSON domains, we show that our approach is superior to state-of-the-art baselines, supporting a broader class of constraints and improving both runtime and performance. Additional theoretical and empirical analyses show that our method's runtime efficiency is driven by its dynamic use of computation, scaling with the divergence between the unconstrained and constrained LM, and as a consequence, runtime improvements are greater for better models.
>
---
#### [replaced 023] Is Smaller Always Faster? Tradeoffs in Compressing Self-Supervised Speech Transformers
- **分类: cs.CL; cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2211.09949v4](http://arxiv.org/pdf/2211.09949v4)**

> **作者:** Tzu-Quan Lin; Tsung-Huan Yang; Chun-Yao Chang; Kuang-Ming Chen; Tzu-hsun Feng; Hung-yi Lee; Hao Tang
>
> **备注:** Accepted at ASRU 2025. Code is available at https://github.com/nervjack2/Speech-SSL-Compression
>
> **摘要:** Transformer-based self-supervised models have achieved remarkable success in speech processing, but their large size and high inference cost present significant challenges for real-world deployment. While numerous compression techniques have been proposed, inconsistent evaluation metrics make it difficult to compare their practical effectiveness. In this work, we conduct a comprehensive study of four common compression methods, including weight pruning, head pruning, low-rank approximation, and knowledge distillation on self-supervised speech Transformers. We evaluate each method under three key metrics: parameter count, multiply-accumulate operations, and real-time factor. Results show that each method offers distinct advantages. In addition, we contextualize recent compression techniques, comparing DistilHuBERT, FitHuBERT, LightHuBERT, ARMHuBERT, and STaRHuBERT under the same framework, offering practical guidance on compression for deployment.
>
---
#### [replaced 024] Automated Black-box Prompt Engineering for Personalized Text-to-Image Generation
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2403.19103v4](http://arxiv.org/pdf/2403.19103v4)**

> **作者:** Yutong He; Alexander Robey; Naoki Murata; Yiding Jiang; Joshua Nathaniel Williams; George J. Pappas; Hamed Hassani; Yuki Mitsufuji; Ruslan Salakhutdinov; J. Zico Kolter
>
> **摘要:** Prompt engineering is an effective but labor-intensive way to control text-to-image (T2I) generative models. Its time-intensive nature and complexity have spurred the development of algorithms for automated prompt generation. However, these methods often struggle with transferability across T2I models, require white-box access to the underlying model, or produce non-intuitive prompts. In this work, we introduce PRISM, an algorithm that automatically produces human-interpretable and transferable prompts that can effectively generate desired concepts given only black-box access to T2I models. Inspired by large language model (LLM) jailbreaking, PRISM leverages the in-context learning ability of LLMs to iteratively refine the candidate prompt distribution built upon the reference images. Our experiments demonstrate the versatility and effectiveness of PRISM in generating accurate prompts for objects, styles, and images across multiple T2I models, including Stable Diffusion, DALL-E, and Midjourney.
>
---
#### [replaced 025] FacLens: Transferable Probe for Foreseeing Non-Factuality in Fact-Seeking Question Answering of Large Language Models
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2406.05328v4](http://arxiv.org/pdf/2406.05328v4)**

> **作者:** Yanling Wang; Haoyang Li; Hao Zou; Jing Zhang; Xinlei He; Qi Li; Ke Xu
>
> **摘要:** Despite advancements in large language models (LLMs), non-factual responses still persist in fact-seeking question answering. Unlike extensive studies on post-hoc detection of these responses, this work studies non-factuality prediction (NFP), predicting whether an LLM will generate a non-factual response prior to the response generation. Previous NFP methods have shown LLMs' awareness of their knowledge, but they face challenges in terms of efficiency and transferability. In this work, we propose a lightweight model named Factuality Lens (FacLens), which effectively probes hidden representations of fact-seeking questions for the NFP task. Moreover, we discover that hidden question representations sourced from different LLMs exhibit similar NFP patterns, enabling the transferability of FacLens across different LLMs to reduce development costs. Extensive experiments highlight FacLens's superiority in both effectiveness and efficiency.
>
---
#### [replaced 026] 2SSP: A Two-Stage Framework for Structured Pruning of LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.17771v2](http://arxiv.org/pdf/2501.17771v2)**

> **作者:** Fabrizio Sandri; Elia Cunegatti; Giovanni Iacca
>
> **备注:** Published in Transactions on Machine Learning Research (TMLR)
>
> **摘要:** We propose a novel Two-Stage framework for Structured Pruning (\textsc{2SSP}) for pruning Large Language Models (LLMs), which combines two different strategies of pruning, namely Width and Depth Pruning. The first stage (Width Pruning) removes entire neurons, hence their corresponding rows and columns, aiming to preserve the connectivity among the pruned structures in the intermediate state of the Feed-Forward Networks in each Transformer block. This is done based on an importance score measuring the impact of each neuron on the output magnitude. The second stage (Depth Pruning), instead, removes entire Attention submodules. This is done by applying an iterative process that removes the Attention with the minimum impact on a given metric of interest (in our case, perplexity). We also propose a novel mechanism to balance the sparsity rate of the two stages w.r.t. to the desired global sparsity. We test \textsc{2SSP} on four LLM families and three sparsity rates (25\%, 37.5\%, and 50\%), measuring the resulting perplexity over three language modeling datasets as well as the performance over six downstream tasks. Our method consistently outperforms five state-of-the-art competitors over three language modeling and six downstream tasks, with an up to two-order-of-magnitude gain in terms of pruning time. The code is available at https://github.com/FabrizioSandri/2SSP.
>
---
#### [replaced 027] A Comprehensive Review of Datasets for Clinical Mental Health AI Systems
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.09809v2](http://arxiv.org/pdf/2508.09809v2)**

> **作者:** Aishik Mandal; Prottay Kumar Adhikary; Hiba Arnaout; Iryna Gurevych; Tanmoy Chakraborty
>
> **备注:** 23 pages, 3 figures
>
> **摘要:** Mental health disorders are rising worldwide. However, the availability of trained clinicians has not scaled proportionally, leaving many people without adequate or timely support. To bridge this gap, recent studies have shown the promise of Artificial Intelligence (AI) to assist mental health diagnosis, monitoring, and intervention. However, the development of efficient, reliable, and ethical AI to assist clinicians is heavily dependent on high-quality clinical training datasets. Despite growing interest in data curation for training clinical AI assistants, existing datasets largely remain scattered, under-documented, and often inaccessible, hindering the reproducibility, comparability, and generalizability of AI models developed for clinical mental health care. In this paper, we present the first comprehensive survey of clinical mental health datasets relevant to the training and development of AI-powered clinical assistants. We categorize these datasets by mental disorders (e.g., depression, schizophrenia), data modalities (e.g., text, speech, physiological signals), task types (e.g., diagnosis prediction, symptom severity estimation, intervention generation), accessibility (public, restricted or private), and sociocultural context (e.g., language and cultural background). Along with these, we also investigate synthetic clinical mental health datasets. Our survey identifies critical gaps such as a lack of longitudinal data, limited cultural and linguistic representation, inconsistent collection and annotation standards, and a lack of modalities in synthetic data. We conclude by outlining key challenges in curating and standardizing future datasets and provide actionable recommendations to facilitate the development of more robust, generalizable, and equitable mental health AI systems.
>
---
#### [replaced 028] Towards No-Code Programming of Cobots: Experiments with Code Synthesis by Large Code Models for Conversational Programming
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2409.11041v3](http://arxiv.org/pdf/2409.11041v3)**

> **作者:** Chalamalasetti Kranti; Sherzod Hakimov; David Schlangen
>
> **备注:** Accepted to ITL4HRI workshop at RO-MAN 2025 conference
>
> **摘要:** While there has been a lot of research recently on robots in household environments, at the present time, most robots in existence can be found on shop floors, and most interactions between humans and robots happen there. ``Collaborative robots'' (cobots) designed to work alongside humans on assembly lines traditionally require expert programming, limiting ability to make changes, or manual guidance, limiting expressivity of the resulting programs. To address these limitations, we explore using Large Language Models (LLMs), and in particular, their abilities of doing in-context learning, for conversational code generation. As a first step, we define RATS, the ``Repetitive Assembly Task'', a 2D building task designed to lay the foundation for simulating industry assembly scenarios. In this task, a `programmer' instructs a cobot, using natural language, on how a certain assembly is to be built; that is, the programmer induces a program, through natural language. We create a dataset that pairs target structures with various example instructions (human-authored, template-based, and model-generated) and example code. With this, we systematically evaluate the capabilities of state-of-the-art LLMs for synthesising this kind of code, given in-context examples. Evaluating in a simulated environment, we find that LLMs are capable of generating accurate `first order code' (instruction sequences), but have problems producing `higher-order code' (abstractions such as functions, or use of loops).
>
---
#### [replaced 029] Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.04823v2](http://arxiv.org/pdf/2504.04823v2)**

> **作者:** Ruikang Liu; Yuxuan Sun; Manyi Zhang; Haoli Bai; Xianzhi Yu; Tiezheng Yu; Chun Yuan; Lu Hou
>
> **备注:** COLM 2025
>
> **摘要:** Recent advancements in reasoning language models have demonstrated remarkable performance in complex tasks, but their extended chain-of-thought reasoning process increases inference overhead. While quantization has been widely adopted to reduce the inference cost of large language models, its impact on reasoning models remains understudied. In this paper, we conduct the first systematic study on quantized reasoning models, evaluating the open-sourced DeepSeek-R1-Distilled Qwen and LLaMA families ranging from 1.5B to 70B parameters, QwQ-32B, and Qwen3-8B. Our investigation covers weight, KV cache, and activation quantization using state-of-the-art algorithms at varying bit-widths, with extensive evaluation across mathematical (AIME, MATH-500), scientific (GPQA), and programming (LiveCodeBench) reasoning benchmarks. Our findings reveal that while lossless quantization can be achieved with W8A8 or W4A16 quantization, lower bit-widths introduce significant accuracy risks. We further identify model size, model origin, and task difficulty as critical determinants of performance. Contrary to expectations, quantized models do not exhibit increased output lengths. In addition, strategically scaling the model sizes or reasoning steps can effectively enhance the performance. All quantized models and codes are open-sourced in https://github.com/ruikangliu/Quantized-Reasoning-Models.
>
---
#### [replaced 030] ContestTrade: A Multi-Agent Trading System Based on Internal Contest Mechanism
- **分类: q-fin.TR; cs.CL; q-fin.CP**

- **链接: [http://arxiv.org/pdf/2508.00554v3](http://arxiv.org/pdf/2508.00554v3)**

> **作者:** Li Zhao; Rui Sun; Zuoyou Jiang; Bo Yang; Yuxiao Bai; Mengting Chen; Xinyang Wang; Jing Li; Zuo Bai
>
> **摘要:** In financial trading, large language model (LLM)-based agents demonstrate significant potential. However, the high sensitivity to market noise undermines the performance of LLM-based trading systems. To address this limitation, we propose a novel multi-agent system featuring an internal competitive mechanism inspired by modern corporate management structures. The system consists of two specialized teams: (1) Data Team - responsible for processing and condensing massive market data into diversified text factors, ensuring they fit the model's constrained context. (2) Research Team - tasked with making parallelized multipath trading decisions based on deep research methods. The core innovation lies in implementing a real-time evaluation and ranking mechanism within each team, driven by authentic market feedback. Each agent's performance undergoes continuous scoring and ranking, with only outputs from top-performing agents being adopted. The design enables the system to adaptively adjust to dynamic environment, enhances robustness against market noise and ultimately delivers superior trading performance. Experimental results demonstrate that our proposed system significantly outperforms prevailing multi-agent systems and traditional quantitative investment methods across diverse evaluation metrics. ContestTrade is open-sourced on GitHub at https://github.com/FinStep-AI/ContestTrade.
>
---
#### [replaced 031] Feather-SQL: A Lightweight NL2SQL Framework with Dual-Model Collaboration Paradigm for Small Language Models
- **分类: cs.CL; cs.AI; cs.DB**

- **链接: [http://arxiv.org/pdf/2503.17811v3](http://arxiv.org/pdf/2503.17811v3)**

> **作者:** Wenqi Pei; Hailing Xu; Hengyuan Zhao; Shizheng Hou; Han Chen; Zining Zhang; Pingyi Luo; Bingsheng He
>
> **备注:** DL4C @ ICLR 2025
>
> **摘要:** Natural Language to SQL (NL2SQL) has seen significant advancements with large language models (LLMs). However, these models often depend on closed-source systems and high computational resources, posing challenges in data privacy and deployment. In contrast, small language models (SLMs) struggle with NL2SQL tasks, exhibiting poor performance and incompatibility with existing frameworks. To address these issues, we introduce Feather-SQL, a new lightweight framework tailored for SLMs. Feather-SQL improves SQL executability and accuracy through 1) schema pruning and linking, 2) multi-path and multi-candidate generation. Additionally, we introduce the 1+1 Model Collaboration Paradigm, which pairs a strong general-purpose chat model with a fine-tuned SQL specialist, combining strong analytical reasoning with high-precision SQL generation. Experimental results on BIRD demonstrate that Feather-SQL improves NL2SQL performance on SLMs, with around 10% boost for models without fine-tuning. The proposed paradigm raises the accuracy ceiling of SLMs to 54.76%, highlighting its effectiveness.
>
---
#### [replaced 032] Dissecting the SWE-Bench Leaderboards: Profiling Submitters and Architectures of LLM- and Agent-Based Repair Systems
- **分类: cs.SE; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.17208v2](http://arxiv.org/pdf/2506.17208v2)**

> **作者:** Matias Martinez; Xavier Franch
>
> **摘要:** The rapid progress in Automated Program Repair (APR) has been driven by advances in AI, particularly large language models (LLMs) and agent-based systems. SWE-Bench is a recent benchmark designed to evaluate LLM-based repair systems using real issues and pull requests mined from 12 popular open-source Python repositories. Its public leaderboards -- SWE-Bench Lite and SWE-Bench Verified -- have become central platforms for tracking progress and comparing solutions. However, because the submission process does not require detailed documentation, the architectural design and origin of many solutions remain unclear. In this paper, we present the first comprehensive study of all submissions to the SWE-Bench Lite (79 entries) and Verified (99 entries) leaderboards, analyzing 80 unique approaches across dimensions such as submitter type, product availability, LLM usage, and system architecture. Our findings reveal the dominance of proprietary LLMs (especially Claude 3.5), the presence of both agentic and non-agentic designs, and a contributor base spanning from individual developers to large tech companies.
>
---
#### [replaced 033] Nonlinear Concept Erasure: a Density Matching Approach
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.12341v2](http://arxiv.org/pdf/2507.12341v2)**

> **作者:** Antoine Saillenfest; Pirmin Lemberger
>
> **备注:** 17 pages, 10 figures, accepted for publication in ECAI 2025 (28th European Conference on Artificial Intelligence)
>
> **摘要:** Ensuring that neural models used in real-world applications cannot infer sensitive information, such as demographic attributes like gender or race, from text representations is a critical challenge when fairness is a concern. We address this issue through concept erasure, a process that removes information related to a specific concept from distributed representations while preserving as much of the remaining semantic information as possible. Our approach involves learning an orthogonal projection in the embedding space, designed to make the class-conditional feature distributions of the discrete concept to erase indistinguishable after projection. By adjusting the rank of the projector, we control the extent of information removal, while its orthogonality ensures strict preservation of the local structure of the embeddings. Our method, termed $\overline{\mathrm{L}}$EOPARD, achieves state-of-the-art performance in nonlinear erasure of a discrete attribute on classic natural language processing benchmarks. Furthermore, we demonstrate that $\overline{\mathrm{L}}$EOPARD effectively mitigates bias in deep nonlinear classifiers, thereby promoting fairness.
>
---
#### [replaced 034] Learning Adaptive Parallel Reasoning with Language Models
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.15466v2](http://arxiv.org/pdf/2504.15466v2)**

> **作者:** Jiayi Pan; Xiuyu Li; Long Lian; Charlie Snell; Yifei Zhou; Adam Yala; Trevor Darrell; Kurt Keutzer; Alane Suhr
>
> **备注:** Accepted at COLM 2025. Code, model, and data are available at https://github.com/Parallel-Reasoning/APR. The first three authors contributed equally to this work
>
> **摘要:** Scaling inference-time computation has substantially improved the reasoning capabilities of language models. However, existing methods have significant limitations: serialized chain-of-thought approaches generate overly long outputs, leading to increased latency and exhausted context windows, while parallel methods such as self-consistency suffer from insufficient coordination, resulting in redundant computations and limited performance gains. To address these shortcomings, we propose Adaptive Parallel Reasoning (APR), a novel reasoning framework that enables language models to orchestrate both serialized and parallel computations end-to-end. APR generalizes existing reasoning methods by enabling adaptive multi-threaded inference using spawn() and join() operations. A key innovation is our end-to-end reinforcement learning strategy, optimizing both parent and child inference threads to enhance task success rate without requiring predefined reasoning structures. Experiments on the Countdown reasoning task demonstrate significant benefits of APR: (1) higher performance within the same context window (83.4% vs. 60.0% at 4k context); (2) superior scalability with increased computation (80.1% vs. 66.6% at 20k total tokens); (3) improved accuracy at equivalent latency (75.2% vs. 57.3% at approximately 5,000ms). APR represents a step towards enabling language models to autonomously optimize their reasoning processes through adaptive allocation of computation.
>
---
#### [replaced 035] FastCuRL: Curriculum Reinforcement Learning with Stage-wise Context Scaling for Efficient Training R1-like Reasoning Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.17287v5](http://arxiv.org/pdf/2503.17287v5)**

> **作者:** Mingyang Song; Mao Zheng; Zheng Li; Wenjie Yang; Xuan Luo; Yue Pan; Feng Zhang
>
> **摘要:** Improving training efficiency continues to be one of the primary challenges in large-scale Reinforcement Learning (RL). In this paper, we investigate how context length and the complexity of training data influence the RL scaling training process of R1-distilled reasoning models, e.g., DeepSeek-R1-Distill-Qwen-1.5B. Our experimental results reveal that: (1) simply controlling the context length and curating the training data based on the input prompt length can effectively improve the training efficiency of RL scaling, achieving better performance with more concise CoT; (2) properly scaling the context length helps mitigate entropy collapse; and (3) carefully choosing the context length facilitates achieving efficient LLM training and reasoning. Inspired by these insights, we propose FastCuRL, a curriculum RL framework with stage-wise context scaling to achieve efficient LLM training and reasoning. Extensive experimental results demonstrate that FastCuRL-1.5B-V3 significantly outperforms state-of-the-art reasoning models on five competition-level benchmarks and achieves 49.6% accuracy on AIME 2024. Furthermore, FastCuRL-1.5B-Preview surpasses DeepScaleR-1.5B-Preview on five benchmarks while only using a single node with 8 GPUs and a total of 50% of training steps.
>
---
#### [replaced 036] LLMCARE: Alzheimer's Detection via Transformer Models Enhanced by LLM-Generated Synthetic Data
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.10027v2](http://arxiv.org/pdf/2508.10027v2)**

> **作者:** Ali Zolnour; Hossein Azadmaleki; Yasaman Haghbin; Fatemeh Taherinezhad; Mohamad Javad Momeni Nezhad; Sina Rashidi; Masoud Khani; AmirSajjad Taleban; Samin Mahdizadeh Sani; Maryam Dadkhah; James M. Noble; Suzanne Bakken; Yadollah Yaghoobzadeh; Abdol-Hossein Vahabie; Masoud Rouhizadeh; Maryam Zolnoori
>
> **摘要:** Alzheimer's disease and related dementias (ADRD) affect approximately five million older adults in the U.S., yet over half remain undiagnosed. Speech-based natural language processing (NLP) offers a promising, scalable approach to detect early cognitive decline through linguistic markers. To develop and evaluate a screening pipeline that (i) fuses transformer embeddings with handcrafted linguistic features, (ii) tests data augmentation using synthetic speech generated by large language models (LLMs), and (iii) benchmarks unimodal and multimodal LLM classifiers for ADRD detection. Transcripts from the DementiaBank "cookie-theft" task (n = 237) were used. Ten transformer models were evaluated under three fine-tuning strategies. A fusion model combined embeddings from the top-performing transformer with 110 lexical-derived linguistic features. Five LLMs (LLaMA-8B/70B, MedAlpaca-7B, Ministral-8B, GPT-4o) were fine-tuned to generate label-conditioned synthetic speech, which was used to augment training data. Three multimodal models (GPT-4o, Qwen-Omni, Phi-4) were tested for speech-text classification in zero-shot and fine-tuned settings. The fusion model achieved F1 = 83.3 (AUC = 89.5), outperforming linguistic or transformer-only baselines. Augmenting training data with 2x MedAlpaca-7B synthetic speech increased F1 to 85.7. Fine-tuning significantly improved unimodal LLM classifiers (e.g., MedAlpaca: F1 = 47.3 -> 78.5 F1). Current multimodal models demonstrated lower performance (GPT-4o = 70.2 F1; Qwen = 66.0). Performance gains aligned with the distributional similarity between synthetic and real speech. Integrating transformer embeddings with linguistic features enhances ADRD detection from speech. Clinically tuned LLMs effectively support both classification and data augmentation, while further advancement is needed in multimodal modeling.
>
---
#### [replaced 037] Idiom Detection in Sorani Kurdish Texts
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.14528v4](http://arxiv.org/pdf/2501.14528v4)**

> **作者:** Skala Kamaran Omer; Hossein Hassani
>
> **备注:** 22 pages, 8 figures, 7 tables
>
> **摘要:** Idiom detection using Natural Language Processing (NLP) is the computerized process of recognizing figurative expressions within a text that convey meanings beyond the literal interpretation of the words. While idiom detection has seen significant progress across various languages, the Kurdish language faces a considerable research gap in this area despite the importance of idioms in tasks like machine translation and sentiment analysis. This study addresses idiom detection in Sorani Kurdish by approaching it as a text classification task using deep learning techniques. To tackle this, we developed a dataset containing 10,580 sentences embedding 101 Sorani Kurdish idioms across diverse contexts. Using this dataset, we developed and evaluated three deep learning models: KuBERT-based transformer sequence classification, a Recurrent Convolutional Neural Network (RCNN), and a BiLSTM model with an attention mechanism. The evaluations revealed that the transformer model, the fine-tuned BERT, consistently outperformed the others, achieving nearly 99% accuracy while the RCNN achieved 96.5% and the BiLSTM 80%. These results highlight the effectiveness of Transformer-based architectures in low-resource languages like Kurdish. This research provides a dataset, three optimized models, and insights into idiom detection, laying a foundation for advancing Kurdish NLP.
>
---
#### [replaced 038] Re:Verse -- Can Your VLM Read a Manga?
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.08508v3](http://arxiv.org/pdf/2508.08508v3)**

> **作者:** Aaditya Baranwal; Madhav Kataria; Naitik Agrawal; Yogesh S Rawat; Shruti Vyas
>
> **备注:** Accepted (oral) at ICCV (AISTORY Workshop) 2025
>
> **摘要:** Current Vision Language Models (VLMs) demonstrate a critical gap between surface-level recognition and deep narrative reasoning when processing sequential visual storytelling. Through a comprehensive investigation of manga narrative understanding, we reveal that while recent large multimodal models excel at individual panel interpretation, they systematically fail at temporal causality and cross-panel cohesion, core requirements for coherent story comprehension. We introduce a novel evaluation framework that combines fine-grained multimodal annotation, cross-modal embedding analysis, and retrieval-augmented assessment to systematically characterize these limitations. Our methodology includes (i) a rigorous annotation protocol linking visual elements to narrative structure through aligned light novel text, (ii) comprehensive evaluation across multiple reasoning paradigms, including direct inference and retrieval-augmented generation, and (iii) cross-modal similarity analysis revealing fundamental misalignments in current VLMs' joint representations. Applying this framework to Re:Zero manga across 11 chapters with 308 annotated panels, we conduct the first systematic study of long-form narrative understanding in VLMs through three core evaluation axes: generative storytelling, contextual dialogue grounding, and temporal reasoning. Our findings demonstrate that current models lack genuine story-level intelligence, struggling particularly with non-linear narratives, character consistency, and causal inference across extended sequences. This work establishes both the foundation and practical methodology for evaluating narrative intelligence, while providing actionable insights into the capability of deep sequential understanding of Discrete Visual Narratives beyond basic recognition in Multimodal Models. Project Page: https://re-verse.vercel.app
>
---
#### [replaced 039] CoRank: LLM-Based Compact Reranking with Document Features for Scientific Retrieval
- **分类: cs.IR; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.13757v2](http://arxiv.org/pdf/2505.13757v2)**

> **作者:** Runchu Tian; Xueqiang Xu; Bowen Jin; SeongKu Kang; Jiawei Han
>
> **备注:** 12 pages, 5 figures
>
> **摘要:** Scientific retrieval is essential for advancing scientific knowledge discovery. Within this process, document reranking plays a critical role in refining first-stage retrieval results. However, standard LLM listwise reranking faces challenges in the scientific domain. First-stage retrieval is often suboptimal in the scientific domain, so relevant documents are ranked lower. Meanwhile, conventional listwise reranking places the full text of candidates into the context window, limiting the number of candidates that can be considered. As a result, many relevant documents are excluded before reranking, constraining overall retrieval performance. To address these challenges, we explore semantic-feature-based compact document representations (e.g., categories, sections, and keywords) and propose CoRank, a training-free, model-agnostic reranking framework for scientific retrieval. It presents a three-stage solution: (i) offline extraction of document features, (ii) coarse-grained reranking using these compact representations, and (iii) fine-grained reranking on full texts of the top candidates from (ii). This integrated process addresses suboptimal first-stage retrieval: Compact representations allow more documents to fit within the context window, improving candidate set coverage, while the final fine-grained ranking ensures a more accurate ordering. Experiments on 5 academic retrieval datasets show that CoRank significantly improves reranking performance across different LLM backbones (average nDCG@10 from 50.6 to 55.5). Overall, these results underscore the synergistic interaction between information extraction and information retrieval, demonstrating how structured semantic features can enhance reranking in the scientific domain.
>
---
#### [replaced 040] SCORE: Story Coherence and Retrieval Enhancement for AI Narratives
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.23512v5](http://arxiv.org/pdf/2503.23512v5)**

> **作者:** Qiang Yi; Yangfan He; Jianhui Wang; Xinyuan Song; Shiyao Qian; Xinhang Yuan; Li Sun; Yi Xin; Jingqun Tang; Keqin Li; Kuan Lu; Menghao Huo; Jiaqi Chen; Tianyu Shi
>
> **摘要:** Large Language Models (LLMs) can generate creative and engaging narratives from user-specified input, but maintaining coherence and emotional depth throughout these AI-generated stories remains a challenge. In this work, we propose SCORE, a framework for Story Coherence and Retrieval Enhancement, designed to detect and resolve narrative inconsistencies. By tracking key item statuses and generating episode summaries, SCORE uses a Retrieval-Augmented Generation (RAG) approach, incorporating TF-IDF and cosine similarity to identify related episodes and enhance the overall story structure. Results from testing multiple LLM-generated stories demonstrate that SCORE significantly improves the consistency and stability of narrative coherence compared to baseline GPT models, providing a more robust method for evaluating and refining AI-generated narratives.
>
---
#### [replaced 041] EvalAgent: Discovering Implicit Evaluation Criteria from the Web
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.15219v2](http://arxiv.org/pdf/2504.15219v2)**

> **作者:** Manya Wadhwa; Zayne Sprague; Chaitanya Malaviya; Philippe Laban; Junyi Jessy Li; Greg Durrett
>
> **备注:** Published at COLM 2025
>
> **摘要:** Evaluation of language model outputs on structured writing tasks is typically conducted with a number of desirable criteria presented to human evaluators or large language models (LLMs). For instance, on a prompt like "Help me draft an academic talk on coffee intake vs research productivity", a model response may be evaluated for criteria like accuracy and coherence. However, high-quality responses should do more than just satisfy basic task requirements. An effective response to this query should include quintessential features of an academic talk, such as a compelling opening, clear research questions, and a takeaway. To help identify these implicit criteria, we introduce EvalAgent, a novel framework designed to automatically uncover nuanced and task-specific criteria. EvalAgent first mines expert-authored online guidance. It then uses this evidence to propose diverse, long-tail evaluation criteria that are grounded in reliable external sources. Our experiments demonstrate that the grounded criteria produced by EvalAgent are often implicit (not directly stated in the user's prompt), yet specific (high degree of lexical precision). Further, EvalAgent criteria are often not satisfied by initial responses but they are actionable, such that responses can be refined to satisfy them. Finally, we show that combining LLM-generated and EvalAgent criteria uncovers more human-valued criteria than using LLMs alone.
>
---
#### [replaced 042] Beyond "Not Novel Enough": Enriching Scholarly Critique with LLM-Assisted Feedback
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.10795v2](http://arxiv.org/pdf/2508.10795v2)**

> **作者:** Osama Mohammed Afzal; Preslav Nakov; Tom Hope; Iryna Gurevych
>
> **摘要:** Novelty assessment is a central yet understudied aspect of peer review, particularly in high volume fields like NLP where reviewer capacity is increasingly strained. We present a structured approach for automated novelty evaluation that models expert reviewer behavior through three stages: content extraction from submissions, retrieval and synthesis of related work, and structured comparison for evidence based assessment. Our method is informed by a large scale analysis of human written novelty reviews and captures key patterns such as independent claim verification and contextual reasoning. Evaluated on 182 ICLR 2025 submissions with human annotated reviewer novelty assessments, the approach achieves 86.5% alignment with human reasoning and 75.3% agreement on novelty conclusions - substantially outperforming existing LLM based baselines. The method produces detailed, literature aware analyses and improves consistency over ad hoc reviewer judgments. These results highlight the potential for structured LLM assisted approaches to support more rigorous and transparent peer review without displacing human expertise. Data and code are made available.
>
---
#### [replaced 043] Token-level Accept or Reject: A Micro Alignment Approach for Large Language Models
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.19743v3](http://arxiv.org/pdf/2505.19743v3)**

> **作者:** Yang Zhang; Yu Yu; Bo Tang; Yu Zhu; Chuxiong Sun; Wenqiang Wei; Jie Hu; Zipeng Xie; Zhiyu Li; Feiyu Xiong; Edward Chung
>
> **备注:** Accepted to 34th International Joint Conference on Artificial Intelligence (IJCAI 2025)
>
> **摘要:** With the rapid development of Large Language Models (LLMs), aligning these models with human preferences and values is critical to ensuring ethical and safe applications. However, existing alignment techniques such as RLHF or DPO often require direct fine-tuning on LLMs with billions of parameters, resulting in substantial computational costs and inefficiencies. To address this, we propose Micro token-level Accept-Reject Aligning (MARA) approach designed to operate independently of the language models. MARA simplifies the alignment process by decomposing sentence-level preference learning into token-level binary classification, where a compact three-layer fully-connected network determines whether candidate tokens are "Accepted" or "Rejected" as part of the response. Extensive experiments across seven different LLMs and three open-source datasets show that MARA achieves significant improvements in alignment performance while reducing computational costs. The source code and implementation details are publicly available at https://github.com/IAAR-Shanghai/MARA, and the trained models are released at https://huggingface.co/IAAR-Shanghai/MARA_AGENTS.
>
---
#### [replaced 044] VisualSpeech: Enhancing Prosody Modeling in TTS Using Video
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.19258v2](http://arxiv.org/pdf/2501.19258v2)**

> **作者:** Shumin Que; Anton Ragni
>
> **摘要:** Text-to-Speech (TTS) synthesis faces the inherent challenge of producing multiple speech outputs with varying prosody given a single text input. While previous research has addressed this by predicting prosodic information from both text and speech, additional contextual information, such as video, remains under-utilized despite being available in many applications. This paper investigates the potential of integrating visual context to enhance prosody prediction. We propose a novel model, VisualSpeech, which incorporates visual and textual information for improving prosody generation in TTS. Empirical results indicate that incorporating visual features improves prosodic modeling, enhancing the expressiveness of the synthesized speech. Audio samples are available at https://ariameetgit.github.io/VISUALSPEECH-SAMPLES/.
>
---
#### [replaced 045] iFairy: the First 2-bit Complex LLM with All Parameters in $\{\pm1, \pm i\}$
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.05571v3](http://arxiv.org/pdf/2508.05571v3)**

> **作者:** Feiyu Wang; Guoan Wang; Yihao Zhang; Shengfan Wang; Weitao Li; Bokai Huang; Shimao Chen; Zihan Jiang; Rui Xu; Tong Yang
>
> **备注:** 15 pages, 9 figures
>
> **摘要:** Quantization-Aware Training (QAT) integrates quantization into the training loop, enabling LLMs to learn robust low-bit representations, and is widely recognized as one of the most promising research directions. All current QAT research focuses on minimizing quantization error on full-precision models, where the full-precision accuracy acts as an upper bound (accuracy ceiling). No existing method has even attempted to surpass this ceiling. To break this ceiling, we propose a new paradigm: raising the ceiling (full-precision model), and then still quantizing it efficiently into 2 bits. We propose Fairy$\pm i$, the first 2-bit quantization framework for complex-valued LLMs. Specifically, our method leverages the representational advantages of the complex domain to boost full-precision accuracy. We map weights to the fourth roots of unity $\{\pm1, \pm i\}$, forming a perfectly symmetric and information-theoretically optimal 2-bit representation. Importantly, each quantized weight has either a zero real or imaginary part, enabling multiplication-free inference using only additions and element swaps. Experimental results show that Fairy$\pm i$ outperforms the ceiling of existing 2-bit quantization approaches in terms of both PPL and downstream tasks, while maintaining strict storage and compute efficiency. This work opens a new direction for building highly accurate and practical LLMs under extremely low-bit constraints.
>
---
#### [replaced 046] Deliberate Planning in Language Models with Symbolic Representation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.01479v2](http://arxiv.org/pdf/2505.01479v2)**

> **作者:** Siheng Xiong; Zhangding Liu; Jieyu Zhou; Yusen Su
>
> **摘要:** Planning remains a core challenge for language models (LMs), particularly in domains that require coherent multi-step action sequences grounded in external constraints. We introduce SymPlanner, a novel framework that equips LMs with structured planning capabilities by interfacing them with a symbolic environment that serves as an explicit world model. Rather than relying purely on natural language reasoning, SymPlanner grounds the planning process in a symbolic state space, where a policy model proposes actions and a symbolic environment deterministically executes and verifies their effects. To enhance exploration and improve robustness, we introduce Iterative Correction (IC), which refines previously proposed actions by leveraging feedback from the symbolic environment to eliminate invalid decisions and guide the model toward valid alternatives. Additionally, Contrastive Ranking (CR) enables fine-grained comparison of candidate plans by evaluating them jointly. We evaluate SymPlanner on PlanBench, demonstrating that it produces more coherent, diverse, and verifiable plans than pure natural language baselines.
>
---
#### [replaced 047] Concealment of Intent: A Game-Theoretic Analysis
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.20841v2](http://arxiv.org/pdf/2505.20841v2)**

> **作者:** Xinbo Wu; Abhishek Umrawal; Lav R. Varshney
>
> **摘要:** As large language models (LLMs) grow more capable, concerns about their safe deployment have also grown. Although alignment mechanisms have been introduced to deter misuse, they remain vulnerable to carefully designed adversarial prompts. In this work, we present a scalable attack strategy: intent-hiding adversarial prompting, which conceals malicious intent through the composition of skills. We develop a game-theoretic framework to model the interaction between such attacks and defense systems that apply both prompt and response filtering. Our analysis identifies equilibrium points and reveals structural advantages for the attacker. To counter these threats, we propose and analyze a defense mechanism tailored to intent-hiding attacks. Empirically, we validate the attack's effectiveness on multiple real-world LLMs across a range of malicious behaviors, demonstrating clear advantages over existing adversarial prompting techniques.
>
---
#### [replaced 048] High-Dimensional Interlingual Representations of Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.11280v5](http://arxiv.org/pdf/2503.11280v5)**

> **作者:** Bryan Wilie; Samuel Cahyawijaya; Junxian He; Pascale Fung
>
> **摘要:** Large language models (LLMs) trained on massive multilingual datasets hint at the formation of interlingual constructs--a shared subspace in the representation space. However, evidence regarding this phenomenon is mixed, leaving it unclear whether these models truly develop unified interlingual representations, or present a partially aligned constructs. We explore 31 diverse languages varying on their resource-levels, typologies, and geographical regions; and find that multilingual LLMs exhibit inconsistent cross-lingual alignments. To address this, we propose an interlingual representation framework identifying both the shared interlingual semantic subspace and fragmented components, existed due to representational limitations. We introduce Interlingual Local Overlap (ILO) score to quantify interlingual alignment by comparing the local neighborhood structures of high-dimensional representations. We utilize ILO to investigate the impact of single-language fine-tuning on the interlingual representations in multilingual LLMs. Our results indicate that training exclusively on a single language disrupts the alignment in early layers, while freezing these layers preserves the alignment of interlingual representations, leading to improved cross-lingual generalization. These results validate our framework and metric for evaluating interlingual representation, and further underscore that interlingual alignment is crucial for scalable multilingual learning.
>
---
#### [replaced 049] Regress, Don't Guess -- A Regression-like Loss on Number Tokens for Language Models
- **分类: cs.CL; cs.AI; cs.CE; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.02083v3](http://arxiv.org/pdf/2411.02083v3)**

> **作者:** Jonas Zausinger; Lars Pennig; Anamarija Kozina; Sean Sdahl; Julian Sikora; Adrian Dendorfer; Timofey Kuznetsov; Mohamad Hagog; Nina Wiedemann; Kacper Chlodny; Vincent Limbach; Anna Ketteler; Thorben Prein; Vishwa Mohan Singh; Michael Morris Danziger; Jannis Born
>
> **备注:** ICML 2025
>
> **摘要:** While language models have exceptional capabilities at text generation, they lack a natural inductive bias for emitting numbers and thus struggle in tasks involving quantitative reasoning, especially arithmetic. One fundamental limitation is the nature of the cross-entropy (CE) loss, which assumes a nominal scale and thus cannot convey proximity between generated number tokens. In response, we here present a regression-like loss that operates purely on token level. Our proposed Number Token Loss (NTL) comes in two flavors and minimizes either the $L_p$ norm or the Wasserstein distance between the numerical values of the real and predicted number tokens. NTL can easily be added to any language model and extend the CE objective during training without runtime overhead. We evaluate the proposed scheme on various mathematical datasets and find that it consistently improves performance in math-related tasks. In a direct comparison on a regression task, we find that NTL can match the performance of a regression head, despite operating on token level. Finally, we scale NTL up to 3B parameter models and observe improved performance, demonstrating its potential for seamless integration into LLMs. We hope to inspire LLM developers to improve their pretraining objectives and distribute NTL as a minimalistic and lightweight PyPI package $ntloss$: https://github.com/ai4sd/number-token-loss. Development code for full paper reproduction is available separately.
>
---
#### [replaced 050] Full-Duplex-Bench: A Benchmark to Evaluate Full-duplex Spoken Dialogue Models on Turn-taking Capabilities
- **分类: cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2503.04721v3](http://arxiv.org/pdf/2503.04721v3)**

> **作者:** Guan-Ting Lin; Jiachen Lian; Tingle Li; Qirui Wang; Gopala Anumanchipalli; Alexander H. Liu; Hung-yi Lee
>
> **备注:** Accepted by ASRU 2025
>
> **摘要:** Spoken dialogue modeling poses challenges beyond text-based language modeling, requiring real-time interaction, turn-taking, and backchanneling. While most Spoken Dialogue Models (SDMs) operate in half-duplex mode-processing one turn at a time - emerging full-duplex SDMs can listen and speak simultaneously, enabling more natural conversations. However, current evaluations remain limited, focusing mainly on turn-based metrics or coarse corpus-level analyses. To address this, we introduce Full-Duplex-Bench, a benchmark that systematically evaluates key interactive behaviors: pause handling, backchanneling, turn-taking, and interruption management. Our framework uses automatic metrics for consistent, reproducible assessment and provides a fair, fast evaluation setup. By releasing our benchmark and code, we aim to advance spoken dialogue modeling and foster the development of more natural and engaging SDMs.
>
---
#### [replaced 051] TeleAntiFraud-28k: An Audio-Text Slow-Thinking Dataset for Telecom Fraud Detection
- **分类: cs.CL; cs.MM**

- **链接: [http://arxiv.org/pdf/2503.24115v4](http://arxiv.org/pdf/2503.24115v4)**

> **作者:** Zhiming Ma; Peidong Wang; Minhua Huang; Jingpeng Wang; Kai Wu; Xiangzhao Lv; Yachun Pang; Yin Yang; Wenjie Tang; Yuchen Kang
>
> **摘要:** The detection of telecom fraud faces significant challenges due to the lack of high-quality multimodal training data that integrates audio signals with reasoning-oriented textual analysis. To address this gap, we present TeleAntiFraud-28k, the first open-source audio-text slow-thinking dataset specifically designed for automated telecom fraud analysis. Our dataset is constructed through three strategies: (1) Privacy-preserved text-truth sample generation using automatically speech recognition (ASR)-transcribed call recordings (with anonymized original audio), ensuring real-world consistency through text-to-speech (TTS) model regeneration; (2) Semantic enhancement via large language model (LLM)-based self-instruction sampling on authentic ASR outputs to expand scenario coverage; (3) Multi-agent adversarial synthesis that simulates emerging fraud tactics through predefined communication scenarios and fraud typologies. The generated dataset contains 28,511 rigorously processed speech-text pairs, complete with detailed annotations for fraud reasoning. The dataset is divided into three tasks: scenario classification, fraud detection, fraud type classification. Furthermore, we construct TeleAntiFraud-Bench, a standardized evaluation benchmark comprising proportionally sampled instances from the dataset, to facilitate systematic testing of model performance on telecom fraud detection tasks. We also contribute a production-optimized supervised fine-tuning (SFT) model trained on hybrid real/synthetic data, while open-sourcing the data processing framework to enable community-driven dataset expansion. This work establishes a foundational framework for multimodal anti-fraud research while addressing critical challenges in data privacy and scenario diversity. The project will be released at https://github.com/JimmyMa99/TeleAntiFraud.
>
---
#### [replaced 052] More Women, Same Stereotypes: Unpacking the Gender Bias Paradox in Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.15904v2](http://arxiv.org/pdf/2503.15904v2)**

> **作者:** Evan Chen; Run-Jun Zhan; Yan-Bai Lin; Hung-Hsuan Chen
>
> **摘要:** Large Language Models (LLMs) have revolutionized natural language processing, yet concerns persist regarding their tendency to reflect or amplify social biases. This study introduces a novel evaluation framework to uncover gender biases in LLMs: using free-form storytelling to surface biases embedded within the models. A systematic analysis of ten prominent LLMs shows a consistent pattern of overrepresenting female characters across occupations, likely due to supervised fine-tuning (SFT) and reinforcement learning from human feedback (RLHF). Paradoxically, despite this overrepresentation, the occupational gender distributions produced by these LLMs align more closely with human stereotypes than with real-world labor data. This highlights the challenge and importance of implementing balanced mitigation measures to promote fairness and prevent the establishment of potentially new biases. We release the prompts and LLM-generated stories at GitHub.
>
---
#### [replaced 053] On Fusing ChatGPT and Ensemble Learning in Discon-tinuous Named Entity Recognition in Health Corpora
- **分类: cs.CL; cs.AI; I.2.7; J.3**

- **链接: [http://arxiv.org/pdf/2412.16976v3](http://arxiv.org/pdf/2412.16976v3)**

> **作者:** Tzu-Chieh Chen; Wen-Yang Lin
>
> **备注:** 13 pages; a short version named "Beyond GPT-NER: ChatGPT as Ensemble Arbitrator for Discontinuous Named Entity Recognition in Health Corpora" has been accpeted for presentation at MedInfo2025
>
> **摘要:** Named Entity Recognition has traditionally been a key task in natural language processing, aiming to identify and extract important terms from unstructured text data. However, a notable challenge for contemporary deep-learning NER models has been identifying discontinuous entities, which are often fragmented within the text. To date, methods to address Discontinuous Named Entity Recognition have not been explored using ensemble learning to the best of our knowledge. Furthermore, the rise of large language models, such as ChatGPT in recent years, has shown significant effectiveness across many NLP tasks. Most existing approaches, however, have primarily utilized ChatGPT as a problem-solving tool rather than exploring its potential as an integrative element within ensemble learning algorithms. In this study, we investigated the integration of ChatGPT as an arbitrator within an ensemble method, aiming to enhance performance on DNER tasks. Our method combines five state-of-the-art NER models with ChatGPT using custom prompt engineering to assess the robustness and generalization capabilities of the ensemble algorithm. We conducted experiments on three benchmark medical datasets, comparing our method against the five SOTA models, individual applications of GPT-3.5 and GPT-4, and a voting ensemble method. The results indicate that our proposed fusion of ChatGPT with the ensemble learning algorithm outperforms the SOTA results in the CADEC, ShARe13, and ShARe14 datasets, showcasing its potential to enhance NLP applications in the healthcare domain.
>
---
#### [replaced 054] From Trial-and-Error to Improvement: A Systematic Analysis of LLM Exploration Mechanisms in RLVR
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.07534v2](http://arxiv.org/pdf/2508.07534v2)**

> **作者:** Jia Deng; Jie Chen; Zhipeng Chen; Daixuan Cheng; Fei Bai; Beichen Zhang; Yinqian Min; Yanzipeng Gao; Wayne Xin Zhao; Ji-Rong Wen
>
> **备注:** 27pages,25figures. arXiv admin note: text overlap with arXiv:2508.02260
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) has emerged as a powerful paradigm for enhancing the reasoning capabilities of large language models (LLMs). Unlike traditional RL approaches, RLVR leverages rule-based feedback to guide LLMs in generating and refining complex reasoning chains -- a process critically dependent on effective exploration strategies. While prior work has demonstrated RLVR's empirical success, the fundamental mechanisms governing LLMs' exploration behaviors remain underexplored. This technical report presents a systematic investigation of exploration capacities in RLVR, covering four main aspects: (1) exploration space shaping, where we develop quantitative metrics to characterize LLMs' capability boundaries; (2) entropy-performance exchange, analyzed across training stages, individual instances, and token-level patterns; and (3) RL performance optimization, examining methods to effectively translate exploration gains into measurable improvements. By unifying previously identified insights with new empirical evidence, this work aims to provide a foundational framework for advancing RLVR systems.
>
---
#### [replaced 055] Evaluation of Finetuned LLMs in AMR Parsing
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.05028v3](http://arxiv.org/pdf/2508.05028v3)**

> **作者:** Shu Han Ho
>
> **备注:** 27 pages, 32 figures
>
> **摘要:** AMR (Abstract Meaning Representation) is a semantic formalism that encodes sentence meaning as rooted, directed, acyclic graphs, where nodes represent concepts and edges denote semantic relations. Finetuning decoder only Large Language Models (LLMs) represent a promising novel straightfoward direction for AMR parsing. This paper presents a comprehensive evaluation of finetuning four distinct LLM architectures, Phi 3.5, Gemma 2, LLaMA 3.2, and DeepSeek R1 LLaMA Distilled using the LDC2020T02 Gold AMR3.0 test set. Our results have shown that straightfoward finetuning of decoder only LLMs can achieve comparable performance to complex State of the Art (SOTA) AMR parsers. Notably, LLaMA 3.2 demonstrates competitive performance against SOTA AMR parsers given a straightforward finetuning approach. We achieved SMATCH F1: 0.804 on the full LDC2020T02 test split, on par with APT + Silver (IBM) at 0.804 and approaching Graphene Smatch (MBSE) at 0.854. Across our analysis, we also observed a consistent pattern where LLaMA 3.2 leads in semantic performance while Phi 3.5 excels in structural validity.
>
---
#### [replaced 056] Large Language Models Must Be Taught to Know What They Don't Know
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **链接: [http://arxiv.org/pdf/2406.08391v3](http://arxiv.org/pdf/2406.08391v3)**

> **作者:** Sanyam Kapoor; Nate Gruver; Manley Roberts; Katherine Collins; Arka Pal; Umang Bhatt; Adrian Weller; Samuel Dooley; Micah Goldblum; Andrew Gordon Wilson
>
> **备注:** NeurIPS 2024 Camera Ready
>
> **摘要:** When using large language models (LLMs) in high-stakes applications, we need to know when we can trust their predictions. Some works argue that prompting high-performance LLMs is sufficient to produce calibrated uncertainties, while others introduce sampling methods that can be prohibitively expensive. In this work, we first argue that prompting on its own is insufficient to achieve good calibration and then show that fine-tuning on a small dataset of correct and incorrect answers can create an uncertainty estimate with good generalization and small computational overhead. We show that a thousand graded examples are sufficient to outperform baseline methods and that training through the features of a model is necessary for good performance and tractable for large open-source models when using LoRA. We also investigate the mechanisms that enable reliable LLM uncertainty estimation, finding that many models can be used as general-purpose uncertainty estimators, applicable not just to their own uncertainties but also the uncertainty of other models. Lastly, we show that uncertainty estimates inform human use of LLMs in human-AI collaborative settings through a user study.
>
---
#### [replaced 057] Large language models can replicate cross-cultural differences in personality
- **分类: cs.CL; cs.AI; cs.CY; I.2.7; K.4.2; J.4**

- **链接: [http://arxiv.org/pdf/2310.10679v4](http://arxiv.org/pdf/2310.10679v4)**

> **作者:** Paweł Niszczota; Mateusz Janczak; Michał Misiak
>
> **备注:** 27 pages: 12 pages of manuscript + 15 pages of supplementary materials; in V3 added information that this is the Author Accepted Manuscript version; in V4 license changed to CC-BY
>
> **摘要:** We use a large-scale experiment (N=8000) to determine whether GPT-4 can replicate cross-cultural differences in the Big Five, measured using the Ten-Item Personality Inventory. We used the US and South Korea as the cultural pair, given that prior research suggests substantial personality differences between people from these two countries. We manipulated the target of the simulation (US vs. Korean), the language of the inventory (English vs. Korean), and the language model (GPT-4 vs. GPT-3.5). Our results show that GPT-4 replicated the cross-cultural differences for each factor. However, mean ratings had an upward bias and exhibited lower variation than in the human samples, as well as lower structural validity. We provide preliminary evidence that LLMs can aid cross-cultural researchers and practitioners.
>
---
#### [replaced 058] USAD: Universal Speech and Audio Representation via Distillation
- **分类: cs.SD; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.18843v2](http://arxiv.org/pdf/2506.18843v2)**

> **作者:** Heng-Jui Chang; Saurabhchand Bhati; James Glass; Alexander H. Liu
>
> **备注:** Accepted to ASRU 2025
>
> **摘要:** Self-supervised learning (SSL) has revolutionized audio representations, yet models often remain domain-specific, focusing on either speech or non-speech tasks. In this work, we present Universal Speech and Audio Distillation (USAD), a unified approach to audio representation learning that integrates diverse audio types - speech, sound, and music - into a single model. USAD employs efficient layer-to-layer distillation from domain-specific SSL models to train a student on a comprehensive audio dataset. USAD offers competitive performance across various benchmarks and datasets, including frame and instance-level speech processing tasks, audio tagging, and sound classification, achieving near state-of-the-art results with a single encoder on SUPERB and HEAR benchmarks.
>
---
#### [replaced 059] NormXLogit: The Head-on-Top Never Lies
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.16252v2](http://arxiv.org/pdf/2411.16252v2)**

> **作者:** Sina Abbasi; Mohammad Reza Modarres; Mohammad Taher Pilehvar
>
> **备注:** Added comparisons on computational efficiency, included experiments on a new dataset with an additional evaluation metric for classification tasks, expanded explanations and discussions in the experiments, and presented a worked example for alignment metrics computation
>
> **摘要:** With new large language models (LLMs) emerging frequently, it is important to consider the potential value of model-agnostic approaches that can provide interpretability across a variety of architectures. While recent advances in LLM interpretability show promise, many rely on complex, model-specific methods with high computational costs. To address these limitations, we propose NormXLogit, a novel technique for assessing the significance of individual input tokens. This method operates based on the input and output representations associated with each token. First, we demonstrate that during the pre-training of LLMs, the norms of word embeddings effectively capture token importance. Second, we reveal a significant relationship between a token's importance and the extent to which its representation can resemble the model's final prediction. Extensive analyses reveal that our approach outperforms existing gradient-based methods in terms of faithfulness and offers competitive performance in layer-wise explanations compared to leading architecture-specific techniques.
>
---
#### [replaced 060] A Law of Next-Token Prediction in Large Language Models
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **链接: [http://arxiv.org/pdf/2408.13442v2](http://arxiv.org/pdf/2408.13442v2)**

> **作者:** Hangfeng He; Weijie J. Su
>
> **备注:** Accepted at Physical Review Research
>
> **摘要:** Large language models (LLMs) have been widely employed across various application domains, yet their black-box nature poses significant challenges to understanding how these models process input data internally to make predictions. In this paper, we introduce a precise and quantitative law that governs the learning of contextualized token embeddings through intermediate layers in pre-trained LLMs for next-token prediction. Our findings reveal that each layer contributes equally to enhancing prediction accuracy, from the lowest to the highest layer -- a universal phenomenon observed across a diverse array of open-source LLMs, irrespective of their architectures or pre-training data. We demonstrate that this law offers new perspectives and actionable insights to inform and guide practices in LLM development and applications, including model scaling, pre-training tasks, and interpretation.
>
---
#### [replaced 061] Generalize across Homophily and Heterophily: Hybrid Spectral Graph Pre-Training and Prompt Tuning
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.11328v2](http://arxiv.org/pdf/2508.11328v2)**

> **作者:** Haitong Luo; Suhang Wang; Weiyao Zhang; Ruiqi Meng; Xuying Meng; Yujun Zhang
>
> **备注:** Under Review
>
> **摘要:** Graph ``pre-training and prompt-tuning'' aligns downstream tasks with pre-trained objectives to enable efficient knowledge transfer under limited supervision. However, existing methods rely on homophily-based low-frequency knowledge, failing to handle diverse spectral distributions in real-world graphs with varying homophily. Our theoretical analysis reveals a spectral specificity principle: optimal knowledge transfer requires alignment between pre-trained spectral filters and the intrinsic spectrum of downstream graphs. Under limited supervision, large spectral gaps between pre-training and downstream tasks impede effective adaptation. To bridge this gap, we propose the HS-GPPT model, a novel framework that ensures spectral alignment throughout both pre-training and prompt-tuning. We utilize a hybrid spectral filter backbone and local-global contrastive learning to acquire abundant spectral knowledge. Then we design prompt graphs to align the spectral distribution with pretexts, facilitating spectral knowledge transfer across homophily and heterophily. Extensive experiments validate the effectiveness under both transductive and inductive learning settings. Our code is available at https://anonymous.4open.science/r/HS-GPPT-62D2/.
>
---
#### [replaced 062] Dealing with Annotator Disagreement in Hate Speech Classification
- **分类: cs.CL; cs.AI; cs.LG; I.2; I.2.7**

- **链接: [http://arxiv.org/pdf/2502.08266v2](http://arxiv.org/pdf/2502.08266v2)**

> **作者:** Somaiyeh Dehghan; Mehmet Umut Sen; Berrin Yanikoglu
>
> **备注:** 20 pages, 3 Tables
>
> **摘要:** Hate speech detection is a crucial task, especially on social media, where harmful content can spread quickly. Implementing machine learning models to automatically identify and address hate speech is essential for mitigating its impact and preventing its proliferation. The first step in developing an effective hate speech detection model is to acquire a high-quality dataset for training. Labeled data is essential for most natural language processing tasks, but categorizing hate speech is difficult due to the diverse and often subjective nature of hate speech, which can lead to varying interpretations and disagreements among annotators. This paper examines strategies for addressing annotator disagreement, an issue that has been largely overlooked. In particular, we evaluate various automatic approaches for aggregating multiple annotations, in the context of hate speech classification in Turkish tweets. Our work highlights the importance of the problem and provides state-of-the-art benchmark results for the detection and understanding of hate speech in online discourse.
>
---
#### [replaced 063] Exploring Scholarly Data by Semantic Query on Knowledge Graph Embedding Space
- **分类: cs.AI; cs.CL; cs.DL**

- **链接: [http://arxiv.org/pdf/1909.08191v3](http://arxiv.org/pdf/1909.08191v3)**

> **作者:** Hung Nghiep Tran; Atsuhiro Takasu
>
> **备注:** TPDL 2019; remove details from the appendix for official dataset publication later
>
> **摘要:** The trends of open science have enabled several open scholarly datasets which include millions of papers and authors. Managing, exploring, and utilizing such large and complicated datasets effectively are challenging. In recent years, the knowledge graph has emerged as a universal data format for representing knowledge about heterogeneous entities and their relationships. The knowledge graph can be modeled by knowledge graph embedding methods, which represent entities and relations as embedding vectors in semantic space, then model the interactions between these embedding vectors. However, the semantic structures in the knowledge graph embedding space are not well-studied, thus knowledge graph embedding methods are usually only used for knowledge graph completion but not data representation and analysis. In this paper, we propose to analyze these semantic structures based on the well-studied word embedding space and use them to support data exploration. We also define the semantic queries, which are algebraic operations between the embedding vectors in the knowledge graph embedding space, to solve queries such as similarity and analogy between the entities on the original datasets. We then design a general framework for data exploration by semantic queries and discuss the solution to some traditional scholarly data exploration tasks. We also propose some new interesting tasks that can be solved based on the uncanny semantic structures of the embedding space.
>
---
#### [replaced 064] LocalGPT: Benchmarking and Advancing Large Language Models for Local Life Services in Meituan
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.02720v2](http://arxiv.org/pdf/2506.02720v2)**

> **作者:** Xiaochong Lan; Jie Feng; Jiahuan Lei; Xinlei Shi; Yong Li
>
> **备注:** KDD 2025
>
> **摘要:** Large language models (LLMs) have exhibited remarkable capabilities and achieved significant breakthroughs across various domains, leading to their widespread adoption in recent years. Building on this progress, we investigate their potential in the realm of local life services. In this study, we establish a comprehensive benchmark and systematically evaluate the performance of diverse LLMs across a wide range of tasks relevant to local life services. To further enhance their effectiveness, we explore two key approaches: model fine-tuning and agent-based workflows. Our findings reveal that even a relatively compact 7B model can attain performance levels comparable to a much larger 72B model, effectively balancing inference cost and model capability. This optimization greatly enhances the feasibility and efficiency of deploying LLMs in real-world online services, making them more practical and accessible for local life applications.
>
---
#### [replaced 065] LoRA-Augmented Generation (LAG) for Knowledge-Intensive Language Tasks
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.05346v2](http://arxiv.org/pdf/2507.05346v2)**

> **作者:** William Fleshman; Benjamin Van Durme
>
> **摘要:** The proliferation of fine-tuned language model experts for specific tasks and domains signals the need for efficient selection and combination methods. We propose LoRA-Augmented Generation (LAG) for leveraging large libraries of knowledge and task-specific LoRA adapters. LAG requires no additional training or access to data, and efficiently filters, retrieves, and applies experts on a per-token and layer basis. We evaluate LAG on various knowledge-intensive tasks, achieving superior performance over existing data-free methods. We explore scenarios where additional data is available, demonstrating LAG's compatibility with alternative solutions such as retrieval-augmented generation (RAG).
>
---
#### [replaced 066] A Survey of LLM-based Deep Search Agents: Paradigm, Optimization, Evaluation, and Challenges
- **分类: cs.IR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.05668v2](http://arxiv.org/pdf/2508.05668v2)**

> **作者:** Yunjia Xi; Jianghao Lin; Yongzhao Xiao; Zheli Zhou; Rong Shan; Te Gao; Jiachen Zhu; Weiwen Liu; Yong Yu; Weinan Zhang
>
> **摘要:** The advent of Large Language Models (LLMs) has significantly revolutionized web search. The emergence of LLM-based Search Agents marks a pivotal shift towards deeper, dynamic, autonomous information seeking. These agents can comprehend user intentions and environmental context and execute multi-turn retrieval with dynamic planning, extending search capabilities far beyond the web. Leading examples like OpenAI's Deep Research highlight their potential for deep information mining and real-world applications. This survey provides the first systematic analysis of search agents. We comprehensively analyze and categorize existing works from the perspectives of architecture, optimization, application, and evaluation, ultimately identifying critical open challenges and outlining promising future research directions in this rapidly evolving field. Our repository is available on https://github.com/YunjiaXi/Awesome-Search-Agent-Papers.
>
---
#### [replaced 067] Flexible Tool Selection through Low-dimensional Attribute Alignment of Vision and Language
- **分类: cs.CV; cs.AI; cs.CL; q-bio.NC**

- **链接: [http://arxiv.org/pdf/2505.22146v3](http://arxiv.org/pdf/2505.22146v3)**

> **作者:** Guangfu Hao; Haojie Wen; Liangxuan Guo; Yang Chen; Yanchao Bi; Shan Yu
>
> **摘要:** Flexible tool selection reflects a complex cognitive ability that distinguishes humans from other species, yet computational models that capture this ability remain underdeveloped. We developed a framework using low-dimensional attribute representations to bridge visual tool perception and linguistic task understanding. We constructed a comprehensive dataset (ToolNet) containing 115 common tools labeled with 13 carefully designed attributes spanning physical, functional, and psychological properties, paired with natural language scenarios describing tool usage. Visual encoders (ResNet or ViT) extract attributes from tool images while fine-tuned language models (GPT-2, LLaMA, DeepSeek) derive required attributes from task descriptions. Our approach achieves 74% accuracy in tool selection tasks-significantly outperforming direct tool matching (20%) and smaller multimodal models (21%-58%), while approaching performance of much larger models like GPT-4o (73%) with substantially fewer parameters. Human evaluation studies validate our framework's alignment with human decision-making patterns, and generalization experiments demonstrate effective performance on novel tool categories. Ablation studies revealed that manipulation-related attributes (graspability, elongation, hand-relatedness) consistently prove most critical across modalities. This work provides a parameter-efficient, interpretable solution that mimics human-like tool cognition, advancing both cognitive science understanding and practical applications in tool selection tasks.
>
---
#### [replaced 068] Beyond Fixed: Training-Free Variable-Length Denoising for Diffusion Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.00819v2](http://arxiv.org/pdf/2508.00819v2)**

> **作者:** Jinsong Li; Xiaoyi Dong; Yuhang Zang; Yuhang Cao; Jiaqi Wang; Dahua Lin
>
> **备注:** Code is available at https://github.com/Li-Jinsong/DAEDAL
>
> **摘要:** Diffusion Large Language Models (DLLMs) are emerging as a powerful alternative to the dominant Autoregressive Large Language Models, offering efficient parallel generation and capable global context modeling. However, the practical application of DLLMs is hindered by a critical architectural constraint: the need for a statically predefined generation length. This static length allocation leads to a problematic trade-off: insufficient lengths cripple performance on complex tasks, while excessive lengths incur significant computational overhead and sometimes result in performance degradation. While the inference framework is rigid, we observe that the model itself possesses internal signals that correlate with the optimal response length for a given task. To bridge this gap, we leverage these latent signals and introduce DAEDAL, a novel training-free denoising strategy that enables Dynamic Adaptive Length Expansion for Diffusion Large Language Models. DAEDAL operates in two phases: 1) Before the denoising process, DAEDAL starts from a short initial length and iteratively expands it to a coarse task-appropriate length, guided by a sequence completion metric. 2) During the denoising process, DAEDAL dynamically intervenes by pinpointing and expanding insufficient generation regions through mask token insertion, ensuring the final output is fully developed. Extensive experiments on DLLMs demonstrate that DAEDAL achieves performance comparable, and in some cases superior, to meticulously tuned fixed-length baselines, while simultaneously enhancing computational efficiency by achieving a higher effective token ratio. By resolving the static length constraint, DAEDAL unlocks new potential for DLLMs, bridging a critical gap with their Autoregressive counterparts and paving the way for more efficient and capable generation.
>
---
#### [replaced 069] GTPO and GRPO-S: Token and Sequence-Level Reward Shaping with Policy Entropy
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.04349v4](http://arxiv.org/pdf/2508.04349v4)**

> **作者:** Hongze Tan; Jianfei Pan
>
> **摘要:** Reinforcement learning (RL) with algorithms like Group Relative Policy Optimization (GRPO) improves Large Language Model (LLM) reasoning, but is limited by a coarse-grained credit assignment that applies a uniform reward to all tokens in a sequence. This is a major flaw in long-chain reasoning tasks. This paper solves this with \textbf{Dynamic Entropy Weighting}. Our core idea is that high-entropy tokens in correct responses can guide the policy toward a higher performance ceiling. This allows us to create more fine-grained reward signals for precise policy updates via two ways: 1) \textbf{Group Token Policy Optimization} (\textbf{GTPO}), we assigns a entropy-weighted reward to each token for fine-grained credit assignment. 2) \textbf{Sequence-Level Group Relative Policy Optimization} (\textbf{GRPO-S}), we assigns a entropy-weighted reward to each sequence based on its average token entropy. Experiments show our methods significantly outperform the strong DAPO baseline. The results confirm that our entropy-weighting mechanism is the key driver of this performance boost, offering a better path to enhance deep reasoning in models.
>
---
#### [replaced 070] Overcoming Long-Context Limitations of State-Space Models via Context-Dependent Sparse Attention
- **分类: cs.LG; cs.CL; I.2.7**

- **链接: [http://arxiv.org/pdf/2507.00449v2](http://arxiv.org/pdf/2507.00449v2)**

> **作者:** Zhihao Zhan; Jianan Zhao; Zhaocheng Zhu; Jian Tang
>
> **备注:** Proceedings of the 42nd International Conference on Machine Learning, ES-FoMo III: 3rd Workshop on Efficient Systems for Foundation Models, 18 pages, 9 figures
>
> **摘要:** Efficient long-context modeling remains a critical challenge for natural language processing (NLP), as the time complexity of the predominant Transformer architecture scales quadratically with the sequence length. While state-space models (SSMs) offer alternative sub-quadratic solutions, they struggle to capture long-range dependencies effectively. In this work, we focus on analyzing and improving the long-context modeling capabilities of SSMs. We show that the widely used synthetic task, associative recall, which requires a model to recall a value associated with a single key without context, insufficiently represents the complexities of real-world long-context modeling. To address this limitation, we extend the associative recall to a novel synthetic task, \emph{joint recall}, which requires a model to recall the value associated with a key given in a specified context. Theoretically, we prove that SSMs do not have the expressiveness to solve multi-query joint recall in sub-quadratic time complexity. To resolve this issue, we propose a solution based on integrating SSMs with Context-Dependent Sparse Attention (CDSA), which has the expressiveness to solve multi-query joint recall with sub-quadratic computation. To bridge the gap between theoretical analysis and real-world applications, we propose locality-sensitive Hashing Attention with sparse Key Selection (HAX), which instantiates the theoretical solution and is further tailored to natural language domains. Extensive experiments on both synthetic and real-world long-context benchmarks show that HAX consistently outperforms SSM baselines and SSMs integrated with context-independent sparse attention (CISA).
>
---
#### [replaced 071] MHPP: Exploring the Capabilities and Limitations of Language Models Beyond Basic Code Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2405.11430v3](http://arxiv.org/pdf/2405.11430v3)**

> **作者:** Jianbo Dai; Jianqiao Lu; Yunlong Feng; Guangtao Zeng; Rongju Ruan; Ming Cheng; Dong Huang; Haochen Tan; Zhijiang Guo
>
> **备注:** 43 pages, dataset and code are available at https://github.com/SparksofAGI/MHPP, leaderboard can be found at https://sparksofagi.github.io/MHPP/
>
> **摘要:** Recent advancements in large language models (LLMs) have greatly improved code generation, specifically at the function level. For instance, GPT-4o has achieved a 91.0\% pass rate on HumanEval. However, this draws into question the adequacy of existing benchmarks in thoroughly assessing function-level code generation capabilities. Our study analyzed two common benchmarks, HumanEval and MBPP, and found that these might not thoroughly evaluate LLMs' code generation capacities due to limitations in quality, difficulty, and granularity. To resolve this, we introduce the Mostly Hard Python Problems (MHPP) dataset, consisting of 210 unique human-curated problems. By focusing on the combination of natural language and code reasoning, MHPP gauges LLMs' abilities to comprehend specifications and restrictions, engage in multi-step reasoning, and apply coding knowledge effectively. Initial evaluations of 26 LLMs using MHPP showed many high-performing models on HumanEval failed to achieve similar success on MHPP. Moreover, MHPP highlighted various previously undiscovered limitations within various LLMs, leading us to believe that it could pave the way for a better understanding of LLMs' capabilities and limitations. MHPP, evaluation pipeline, and leaderboard can be found in https://github.com/SparksofAGI/MHPP.
>
---
#### [replaced 072] LIDDIA: Language-based Intelligent Drug Discovery Agent
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.13959v2](http://arxiv.org/pdf/2502.13959v2)**

> **作者:** Reza Averly; Frazier N. Baker; Ian A. Watson; Xia Ning
>
> **备注:** Preprint
>
> **摘要:** Drug discovery is a long, expensive, and complex process, relying heavily on human medicinal chemists, who can spend years searching the vast space of potential therapies. Recent advances in artificial intelligence for chemistry have sought to expedite individual drug discovery tasks; however, there remains a critical need for an intelligent agent that can navigate the drug discovery process. Towards this end, we introduce LIDDIA, an autonomous agent capable of intelligently navigating the drug discovery process in silico. By leveraging the reasoning capabilities of large language models, LIDDIA serves as a low-cost and highly-adaptable tool for autonomous drug discovery. We comprehensively examine LIDDIA , demonstrating that (1) it can generate molecules meeting key pharmaceutical criteria on over 70% of 30 clinically relevant targets, (2) it intelligently balances exploration and exploitation in the chemical space, and (3) it identifies one promising novel candidate on AR/NR3C4, a critical target for both prostate and breast cancers. Code and dataset are available at https://github.com/ninglab/LIDDiA
>
---
#### [replaced 073] Evaluating Contrast Localizer for Identifying Causal Units in Social & Mathematical Tasks in Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.08276v2](http://arxiv.org/pdf/2508.08276v2)**

> **作者:** Yassine Jamaa; Badr AlKhamissi; Satrajit Ghosh; Martin Schrimpf
>
> **备注:** Accepted at the Interplay of Model Behavior and Model Internals Workshop co-located with COLM 2025
>
> **摘要:** This work adapts a neuroscientific contrast localizer to pinpoint causally relevant units for Theory of Mind (ToM) and mathematical reasoning tasks in large language models (LLMs) and vision-language models (VLMs). Across 11 LLMs and 5 VLMs ranging in size from 3B to 90B parameters, we localize top-activated units using contrastive stimulus sets and assess their causal role via targeted ablations. We compare the effect of lesioning functionally selected units against low-activation and randomly selected units on downstream accuracy across established ToM and mathematical benchmarks. Contrary to expectations, low-activation units sometimes produced larger performance drops than the highly activated ones, and units derived from the mathematical localizer often impaired ToM performance more than those from the ToM localizer. These findings call into question the causal relevance of contrast-based localizers and highlight the need for broader stimulus sets and more accurately capture task-specific units.
>
---
#### [replaced 074] PromptSuite: A Task-Agnostic Framework for Multi-Prompt Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.14913v2](http://arxiv.org/pdf/2507.14913v2)**

> **作者:** Eliya Habba; Noam Dahan; Gili Lior; Gabriel Stanovsky
>
> **备注:** Eliya Habba and Noam Dahan contributed equally to this work
>
> **摘要:** Evaluating LLMs with a single prompt has proven unreliable, with small changes leading to significant performance differences. However, generating the prompt variations needed for a more robust multi-prompt evaluation is challenging, limiting its adoption in practice. To address this, we introduce PromptSuite, a framework that enables the automatic generation of various prompts. PromptSuite is flexible - working out of the box on a wide range of tasks and benchmarks. It follows a modular prompt design, allowing controlled perturbations to each component, and is extensible, supporting the addition of new components and perturbation types. Through a series of case studies, we show that PromptSuite provides meaningful variations to support strong evaluation practices. It is available through both a Python API: https://github.com/eliyahabba/PromptSuite, and a user-friendly web interface: https://promptsuite.streamlit.app/
>
---
#### [replaced 075] SpecDetect: Simple, Fast, and Training-Free Detection of LLM-Generated Text via Spectral Analysis
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.11343v2](http://arxiv.org/pdf/2508.11343v2)**

> **作者:** Haitong Luo; Weiyao Zhang; Suhang Wang; Wenji Zou; Chungang Lin; Xuying Meng; Yujun Zhang
>
> **备注:** Under Review
>
> **摘要:** The proliferation of high-quality text from Large Language Models (LLMs) demands reliable and efficient detection methods. While existing training-free approaches show promise, they often rely on surface-level statistics and overlook fundamental signal properties of the text generation process. In this work, we reframe detection as a signal processing problem, introducing a novel paradigm that analyzes the sequence of token log-probabilities in the frequency domain. By systematically analyzing the signal's spectral properties using the global Discrete Fourier Transform (DFT) and the local Short-Time Fourier Transform (STFT), we find that human-written text consistently exhibits significantly higher spectral energy. This higher energy reflects the larger-amplitude fluctuations inherent in human writing compared to the suppressed dynamics of LLM-generated text. Based on this key insight, we construct SpecDetect, a detector built on a single, robust feature from the global DFT: DFT total energy. We also propose an enhanced version, SpecDetect++, which incorporates a sampling discrepancy mechanism to further boost robustness. Extensive experiments demonstrate that our approach outperforms the state-of-the-art model while running in nearly half the time. Our work introduces a new, efficient, and interpretable pathway for LLM-generated text detection, showing that classical signal processing techniques offer a surprisingly powerful solution to this modern challenge.
>
---
