# 自然语言处理 cs.CL

- **最新发布 87 篇**

- **更新 62 篇**

## 最新发布

#### [new 001] Semantic Tree Inference on Text Corpa using a Nested Density Approach together with Large Language Model Embeddings
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本语义分类任务，旨在解决文本语义关系结构不明确的问题。通过嵌入空间密度聚类构建层次树结构，揭示文本的语义层级关系。**

- **链接: [https://arxiv.org/pdf/2512.23471v1](https://arxiv.org/pdf/2512.23471v1)**

> **作者:** Thomas Haschka; Joseph Bakarji
>
> **备注:** 20 pages, 9 figures
>
> **摘要:** Semantic text classification has undergone significant advances in recent years due to the rise of large language models (LLMs) and their high dimensional embeddings. While LLM-embeddings are frequently used to store and retrieve text by semantic similarity in vector databases, the global structure semantic relationships in text corpora often remains opaque. Herein we propose a nested density clustering approach, to infer hierarchical trees of semantically related texts. The method starts by identifying texts of strong semantic similarity as it searches for dense clusters in LLM embedding space. As the density criterion is gradually relaxed, these dense clusters merge into more diffuse clusters, until the whole dataset is represented by a single cluster -- the root of the tree. By embedding dense clusters into increasingly diffuse ones, we construct a tree structure that captures hierarchical semantic relationships among texts. We outline how this approach can be used to classify textual data for abstracts of scientific abstracts as a case study. This enables the data-driven discovery research areas and their subfields without predefined categories. To evaluate the general applicability of the method, we further apply it to established benchmark datasets such as the 20 Newsgroups and IMDB 50k Movie Reviews, demonstrating its robustness across domains. Finally we discuss possible applications on scientometrics, topic evolution, highlighting how nested density trees can reveal semantic structure and evolution in textual datasets.
>
---
#### [new 002] Lie to Me: Knowledge Graphs for Robust Hallucination Self-Detection in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的幻觉检测任务，旨在解决大语言模型生成虚假信息的问题。通过构建知识图谱提升幻觉自检能力，显著提升了检测效果。**

- **链接: [https://arxiv.org/pdf/2512.23547v1](https://arxiv.org/pdf/2512.23547v1)**

> **作者:** Sahil Kale; Antonio Luca Alfeo
>
> **备注:** Accepted to ICPRAM 2026 in Marbella, Spain
>
> **摘要:** Hallucinations, the generation of apparently convincing yet false statements, remain a major barrier to the safe deployment of LLMs. Building on the strong performance of self-detection methods, we examine the use of structured knowledge representations, namely knowledge graphs, to improve hallucination self-detection. Specifically, we propose a simple yet powerful approach that enriches hallucination self-detection by (i) converting LLM responses into knowledge graphs of entities and relations, and (ii) using these graphs to estimate the likelihood that a response contains hallucinations. We evaluate the proposed approach using two widely used LLMs, GPT-4o and Gemini-2.5-Flash, across two hallucination detection datasets. To support more reliable future benchmarking, one of these datasets has been manually curated and enhanced and is released as a secondary outcome of this work. Compared to standard self-detection methods and SelfCheckGPT, a state-of-the-art approach, our method achieves up to 16% relative improvement in accuracy and 20% in F1-score. Our results show that LLMs can better analyse atomic facts when they are structured as knowledge graphs, even when initial outputs contain inaccuracies. This low-cost, model-agnostic approach paves the way toward safer and more trustworthy language models.
>
---
#### [new 003] UniHetero: Could Generation Enhance Understanding for Vision-Language-Model at Large Data Scale?
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于视觉-语言模型任务，探讨生成是否能提升理解能力。研究提出UniHetero模型，在大规模数据下验证生成语义优于生成像素，提升数据利用率与模型效果。**

- **链接: [https://arxiv.org/pdf/2512.23512v1](https://arxiv.org/pdf/2512.23512v1)**

> **作者:** Fengjiao Chen; Minhao Jing; Weitao Lu; Yan Feng; Xiaoyu Li; Xuezhi Cao
>
> **摘要:** Vision-language large models are moving toward the unification of visual understanding and visual generation tasks. However, whether generation can enhance understanding is still under-explored on large data scale. In this work, we analysis the unified model with a concise structure, UniHetero, under large-scale pretraining (>200M samples). Our key observations are: (1) Generation can improve understanding, but Only if you generate Semantics, Not Pixels. (2) Generation reveals a superior Data Scaling trend and higher Data Utilization. (3) Autoregression on Input Embedding is effective to capture visual details.
>
---
#### [new 004] On the Role of Discreteness in Diffusion LLMs
- **分类: cs.CL**

- **简介: 该论文属于语言生成任务，探讨扩散模型在文本中的应用问题。针对文本离散性与结构特点，分析现有方法的不足，提出改进方向。**

- **链接: [https://arxiv.org/pdf/2512.22630v1](https://arxiv.org/pdf/2512.22630v1)**

> **作者:** Ziqi Jin; Bin Wang; Xiang Lin; Lidong Bing; Aixin Sun
>
> **摘要:** Diffusion models offer appealing properties for language generation, such as parallel decoding and iterative refinement, but the discrete and highly structured nature of text challenges the direct application of diffusion principles. In this paper, we revisit diffusion language modeling from the view of diffusion process and language modeling, and outline five properties that separate diffusion mechanics from language-specific requirements. We first categorize existing approaches into continuous diffusion in embedding space and discrete diffusion over tokens. We then show that each satisfies only part of the five essential properties and therefore reflects a structural trade-off. Through analyses of recent large diffusion language models, we identify two central issues: (i) uniform corruption does not respect how information is distributed across positions, and (ii) token-wise marginal training cannot capture multi-token dependencies during parallel decoding. These observations motivate diffusion processes that align more closely with the structure of text, and encourage future work toward more coherent diffusion language models.
>
---
#### [new 005] Hallucination Detection and Evaluation of Large Language Model
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于大语言模型幻觉检测任务，旨在解决模型生成不可靠内容的问题。工作包括引入HHEM框架提升检测效率，并通过段落检索和模型规模分析优化检测效果。**

- **链接: [https://arxiv.org/pdf/2512.22416v1](https://arxiv.org/pdf/2512.22416v1)**

> **作者:** Chenggong Zhang; Haopeng Wang
>
> **摘要:** Hallucinations in Large Language Models (LLMs) pose a significant challenge, generating misleading or unverifiable content that undermines trust and reliability. Existing evaluation methods, such as KnowHalu, employ multi-stage verification but suffer from high computational costs. To address this, we integrate the Hughes Hallucination Evaluation Model (HHEM), a lightweight classification-based framework that operates independently of LLM-based judgments, significantly improving efficiency while maintaining high detection accuracy. We conduct a comparative analysis of hallucination detection methods across various LLMs, evaluating True Positive Rate (TPR), True Negative Rate (TNR), and Accuracy on question-answering (QA) and summarization tasks. Our results show that HHEM reduces evaluation time from 8 hours to 10 minutes, while HHEM with non-fabrication checking achieves the highest accuracy \(82.2\%\) and TPR \(78.9\%\). However, HHEM struggles with localized hallucinations in summarization tasks. To address this, we introduce segment-based retrieval, improving detection by verifying smaller text components. Additionally, our cumulative distribution function (CDF) analysis indicates that larger models (7B-9B parameters) generally exhibit fewer hallucinations, while intermediate-sized models show higher instability. These findings highlight the need for structured evaluation frameworks that balance computational efficiency with robust factual validation, enhancing the reliability of LLM-generated content.
>
---
#### [new 006] HiFi-RAG: Hierarchical Content Filtering and Two-Pass Generation for Open-Domain RAG
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **简介: 该论文属于开放域RAG任务，解决检索文档冗余和答案对齐问题，提出HiFi-RAG框架，通过分层过滤和两阶段生成提升效果。**

- **链接: [https://arxiv.org/pdf/2512.22442v1](https://arxiv.org/pdf/2512.22442v1)**

> **作者:** Cattalyya Nuengsigkapian
>
> **备注:** A winning solution for the NeurIPS 2025 MMU-RAGent Competition (Closed-Source Text-to-Text Static Evaluation)
>
> **摘要:** Retrieval-Augmented Generation (RAG) in open-domain settings faces significant challenges regarding irrelevant information in retrieved documents and the alignment of generated answers with user intent. We present HiFi-RAG (Hierarchical Filtering RAG), the winning closed-source system in the Text-to-Text static evaluation of the MMU-RAGent NeurIPS 2025 Competition. Our approach moves beyond standard embedding-based retrieval via a multi-stage pipeline. We leverage the speed and cost-efficiency of Gemini 2.5 Flash (4-6x cheaper than Pro) for query formulation, hierarchical content filtering, and citation attribution, while reserving the reasoning capabilities of Gemini 2.5 Pro for final answer generation. On the MMU-RAGent validation set, our system outperformed the baseline, improving ROUGE-L to 0.274 (+19.6%) and DeBERTaScore to 0.677 (+6.2%). On Test2025, our custom dataset evaluating questions that require post-cutoff knowledge (post January 2025), HiFi-RAG outperforms the parametric baseline by 57.4% in ROUGE-L and 14.9% in DeBERTaScore.
>
---
#### [new 007] Open-Source Multimodal Moxin Models with Moxin-VLM and Moxin-VLA
- **分类: cs.CL; cs.CV; cs.LG**

- **简介: 该论文介绍开源多模态模型Moxin，解决LLM开放性和多功能性问题，提出Moxin-VLM、Moxin-VLA等变体，提升视觉语言等任务性能。**

- **链接: [https://arxiv.org/pdf/2512.22208v1](https://arxiv.org/pdf/2512.22208v1)**

> **作者:** Pu Zhao; Xuan Shen; Zhenglun Kong; Yixin Shen; Sung-En Chang; Arash Akbari; Timothy Rupprecht; Lei Lu; Enfu Nan; Changdi Yang; Yumei He; Weiyan Shi; Xingchen Xu; Yu Huang; Wei Jiang; Wei Wang; Yue Chen; Yong He; Yanzhi Wang
>
> **摘要:** Recently, Large Language Models (LLMs) have undergone a significant transformation, marked by a rapid rise in both their popularity and capabilities. Leading this evolution are proprietary LLMs like GPT-4 and GPT-o1, which have captured widespread attention in the AI community due to their remarkable performance and versatility. Simultaneously, open-source LLMs, such as LLaMA and Mistral, have made great contributions to the ever-increasing popularity of LLMs due to the ease to customize and deploy the models across diverse applications. Moxin 7B is introduced as a fully open-source LLM developed in accordance with the Model Openness Framework, which moves beyond the simple sharing of model weights to embrace complete transparency in training, datasets, and implementation detail, thus fostering a more inclusive and collaborative research environment that can sustain a healthy open-source ecosystem. To further equip Moxin with various capabilities in different tasks, we develop three variants based on Moxin, including Moxin-VLM, Moxin-VLA, and Moxin-Chinese, which target the vision-language, vision-language-action, and Chinese capabilities, respectively. Experiments show that our models achieve superior performance in various evaluations. We adopt open-source framework and open data for the training. We release our models, along with the available data and code to derive these models.
>
---
#### [new 008] PROFASR-BENCH: A Benchmark for Context-Conditioned ASR in High-Stakes Professional Speech
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于高风险专业语音识别任务，旨在解决领域术语复杂、错误容忍度低的问题。提出ProfASR-Bench基准，评估上下文条件下的ASR性能。**

- **链接: [https://arxiv.org/pdf/2512.23686v1](https://arxiv.org/pdf/2512.23686v1)**

> **作者:** Deepak Babu Piskala
>
> **备注:** Benchmark dataset and evaluation suite. Data and code available at: https://huggingface.co/datasets/prdeepakbabu/ProfASR-Bench https://github.com/prdeepakbabu/ProfASR-Bench
>
> **摘要:** Automatic Speech Recognition (ASR) in professional settings faces challenges that existing benchmarks underplay: dense domain terminology, formal register variation, and near-zero tolerance for critical entity errors. We present ProfASR-Bench, a professional-talk evaluation suite for high-stakes applications across finance, medicine, legal, and technology. Each example pairs a natural-language prompt (domain cue and/or speaker profile) with an entity-rich target utterance, enabling controlled measurement of context-conditioned recognition. The corpus supports conventional ASR metrics alongside entity-aware scores and slice-wise reporting by accent and gender. Using representative families Whisper (encoder-decoder ASR) and Qwen-Omni (audio language models) under matched no-context, profile, domain+profile, oracle, and adversarial conditions, we find a consistent pattern: lightweight textual context produces little to no change in average word error rate (WER), even with oracle prompts, and adversarial prompts do not reliably degrade performance. We term this the context-utilization gap (CUG): current systems are nominally promptable yet underuse readily available side information. ProfASR-Bench provides a standardized context ladder, entity- and slice-aware reporting with confidence intervals, and a reproducible testbed for comparing fusion strategies across model families. Dataset: https://huggingface.co/datasets/prdeepakbabu/ProfASR-Bench Code: https://github.com/prdeepakbabu/ProfASR-Bench
>
---
#### [new 009] The Effect of Gender Diversity on Scientific Team Impact: A Team Roles Perspective
- **分类: cs.CL; cs.CY; cs.DL**

- **简介: 该论文属于社会科学领域，研究性别多样性对科研团队影响力的影响。通过分析13万篇论文，发现性别多样性与团队影响力呈倒U型关系，且领导与支持角色的性别组合影响显著。**

- **链接: [https://arxiv.org/pdf/2512.23429v1](https://arxiv.org/pdf/2512.23429v1)**

> **作者:** Yi Zhao; Yongjun Zhu; Donghun Kim; Yuzhuo Wang; Heng Zhang; Chao Lu; Chengzhi Zhang
>
> **摘要:** The influence of gender diversity on the success of scientific teams is of great interest to academia. However, prior findings remain inconsistent, and most studies operationalize diversity in aggregate terms, overlooking internal role differentiation. This limitation obscures a more nuanced understanding of how gender diversity shapes team impact. In particular, the effect of gender diversity across different team roles remains poorly understood. To this end, we define a scientific team as all coauthors of a paper and measure team impact through five-year citation counts. Using author contribution statements, we classified members into leadership and support roles. Drawing on more than 130,000 papers from PLOS journals, most of which are in biomedical-related disciplines, we employed multivariable regression to examine the association between gender diversity in these roles and team impact. Furthermore, we apply a threshold regression model to investigate how team size moderates this relationship. The results show that (1) the relationship between gender diversity and team impact follows an inverted U-shape for both leadership and support groups; (2) teams with an all-female leadership group and an all-male support group achieve higher impact than other team types. Interestingly, (3) the effect of leadership-group gender diversity is significantly negative for small teams but becomes positive and statistically insignificant in large teams. In contrast, the estimates for support-group gender diversity remain significant and positive, regardless of team size.
>
---
#### [new 010] Style Amnesia: Investigating Speaking Style Degradation and Mitigation in Multi-Turn Spoken Language Models
- **分类: cs.CL; cs.SD**

- **简介: 该论文研究多轮对话中语音模型的风格退化问题，属于自然语言处理任务。旨在解决模型无法持续保持指定说话风格的问题，并通过实验验证不同策略的缓解效果。**

- **链接: [https://arxiv.org/pdf/2512.23578v1](https://arxiv.org/pdf/2512.23578v1)**

> **作者:** Yu-Xiang Lin; Cheng-Han Chiang; Hung-yi Lee
>
> **备注:** Work in progress
>
> **摘要:** In this paper, we show that when spoken language models (SLMs) are instructed to speak in a specific speaking style at the beginning of a multi-turn conversation, they cannot maintain the required speaking styles after several turns of interaction; we refer to this as the style amnesia of SLMs. We focus on paralinguistic speaking styles, including emotion, accent, volume, and speaking speed. We evaluate three proprietary and two open-source SLMs, demonstrating that none of these models can maintain a consistent speaking style when instructed to do so. We further show that when SLMs are asked to recall the style instruction in later turns, they can recall the style instruction, but they fail to express it throughout the conversation. We also show that explicitly asking the model to recall the style instruction can partially mitigate style amnesia. In addition, we examine various prompting strategies and find that SLMs struggle to follow the required style when the instruction is placed in system messages rather than user messages, which contradicts the intended function of system prompts.
>
---
#### [new 011] Hierarchical Geometry of Cognitive States in Transformer Embedding Spaces
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究Transformer模型嵌入空间中的认知状态层次结构，旨在验证句子嵌入是否包含与人类认知属性对齐的层级几何结构。**

- **链接: [https://arxiv.org/pdf/2512.22227v1](https://arxiv.org/pdf/2512.22227v1)**

> **作者:** Sophie Zhao
>
> **摘要:** Recent work has shown that transformer-based language models learn rich geometric structure in their embedding spaces, yet the presence of higher-level cognitive organization within these representations remains underexplored. In this work, we investigate whether sentence embeddings encode a graded, hierarchical structure aligned with human-interpretable cognitive or psychological attributes. We construct a dataset of 480 natural-language sentences annotated with continuous ordinal energy scores and discrete tier labels spanning seven ordered cognitive categories. Using fixed sentence embeddings from multiple transformer models, we evaluate the recoverability of these annotations via linear and shallow nonlinear probes. Across models, both continuous scores and tier labels are reliably decodable, with shallow nonlinear probes providing consistent performance gains over linear probes. Lexical TF-IDF baselines perform substantially worse, indicating that the observed structure is not attributable to surface word statistics alone. Nonparametric permutation tests further confirm that probe performance exceeds chance under label-randomization nulls. Qualitative analyses using UMAP visualizations and confusion matrices reveal smooth low-to-high gradients and predominantly adjacent-tier confusions in embedding space. Taken together, these results provide evidence that transformer embedding spaces exhibit a hierarchical geometric organization aligned with human-defined cognitive attributes, while remaining agnostic to claims of internal awareness or phenomenology.
>
---
#### [new 012] Is Chain-of-Thought Really Not Explainability? Chain-of-Thought Can Be Faithful without Hint Verbalization
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文探讨了链式思维（CoT）的可解释性问题，指出其“不忠实”可能源于提示信息未完全表达，而非本质缺陷。研究提出新评估指标，验证了token限制对解释性的影响，并建议采用更全面的可解释性工具。**

- **链接: [https://arxiv.org/pdf/2512.23032v1](https://arxiv.org/pdf/2512.23032v1)**

> **作者:** Kerem Zaman; Shashank Srivastava
>
> **备注:** 18 pages, 20 figures, 5 tables
>
> **摘要:** Recent work, using the Biasing Features metric, labels a CoT as unfaithful if it omits a prompt-injected hint that affected the prediction. We argue this metric confuses unfaithfulness with incompleteness, the lossy compression needed to turn distributed transformer computation into a linear natural language narrative. On multi-hop reasoning tasks with Llama-3 and Gemma-3, many CoTs flagged as unfaithful by Biasing Features are judged faithful by other metrics, exceeding 50% in some models. With a new faithful@k metric, we show that larger inference-time token budgets greatly increase hint verbalization (up to 90% in some settings), suggesting much apparent unfaithfulness is due to tight token limits. Using Causal Mediation Analysis, we further show that even non-verbalized hints can causally mediate prediction changes through the CoT. We therefore caution against relying solely on hint-based evaluations and advocate a broader interpretability toolkit, including causal mediation and corruption-based metrics.
>
---
#### [new 013] NepEMO: A Multi-Label Emotion and Sentiment Analysis on Nepali Reddit with Linguistic Insights and Temporal Trends
- **分类: cs.CL**

- **简介: 该论文提出NepEMO数据集，用于尼泊尔Reddit帖子的多标签情感与情绪分析，解决社交媒体中情感识别问题，通过语言分析和模型比较提升分类效果。**

- **链接: [https://arxiv.org/pdf/2512.22823v1](https://arxiv.org/pdf/2512.22823v1)**

> **作者:** Sameer Sitoula; Tej Bahadur Shahi; Laxmi Prasad Bhatt; Anisha Pokhrel; Arjun Neupane
>
> **备注:** This paper is under consideration in Neural Computing & Applications (Springer) journal. This version may be deleted or updated at any time, depending on the journal's policy upon acceptance
>
> **摘要:** Social media (SM) platforms (e.g. Facebook, Twitter, and Reddit) are increasingly leveraged to share opinions and emotions, specifically during challenging events, such as natural disasters, pandemics, and political elections, and joyful occasions like festivals and celebrations. Among the SM platforms, Reddit provides a unique space for its users to anonymously express their experiences and thoughts on sensitive issues such as health and daily life. In this work, we present a novel dataset, called NepEMO, for multi-label emotion (MLE) and sentiment classification (SC) on the Nepali subreddit post. We curate and build a manually annotated dataset of 4,462 posts (January 2019- June 2025) written in English, Romanised Nepali and Devanagari script for five emotions (fear, anger, sadness, joy, and depression) and three sentiment classes (positive, negative, and neutral). We perform a detailed analysis of posts to capture linguistic insights, including emotion trends, co-occurrence of emotions, sentiment-specific n-grams, and topic modelling using Latent Dirichlet Allocation and TF-IDF keyword extraction. Finally, we compare various traditional machine learning (ML), deep learning (DL), and transformer models for MLE and SC tasks. The result shows that transformer models consistently outperform the ML and DL models for both tasks.
>
---
#### [new 014] Exploring the Vertical-Domain Reasoning Capabilities of Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLMs在会计领域推理能力不足的问题。研究提出垂直领域会计推理概念，评估多个模型表现，探索提升方法。**

- **链接: [https://arxiv.org/pdf/2512.22443v1](https://arxiv.org/pdf/2512.22443v1)**

> **作者:** Jie Zhou; Xin Chen; Jie Zhang; Zhe Li
>
> **摘要:** Large Language Models (LLMs) are reshaping learning paradigms, cognitive processes, and research methodologies across a wide range of domains. Integrating LLMs with professional fields and redefining the relationship between LLMs and domain-specific applications has become a critical challenge for promoting enterprise digital transformation and broader social development. To effectively integrate LLMs into the accounting domain, it is essential to understand their domain-specific reasoning capabilities. This study introduces the concept of vertical-domain accounting reasoning and establishes evaluation criteria by analyzing the training data characteristics of representative GLM-series models. These criteria provide a foundation for subsequent research on reasoning paradigms and offer benchmarks for improving accounting reasoning performance. Based on this framework, we evaluate several representative models, including GLM-6B, GLM-130B, GLM-4, and OpenAI GPT-4, on a set of accounting reasoning tasks. Experimental results show that different prompt engineering strategies lead to varying degrees of performance improvement across models, with GPT-4 achieving the strongest accounting reasoning capability. However, current LLMs still fall short of real-world application requirements. In particular, further optimization is needed for deployment in enterprise-level accounting scenarios to fully realize the potential value of LLMs in this domain.
>
---
#### [new 015] Entropy-Guided Token Dropout: Training Autoregressive Language Models with Limited Domain Data
- **分类: cs.CL**

- **简介: 该论文属于语言模型训练任务，解决有限领域数据下模型性能下降问题。提出EntroDrop方法，通过熵引导的token dropout提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.23422v1](https://arxiv.org/pdf/2512.23422v1)**

> **作者:** Jiapeng Wang; Yiwen Hu; Yanzipeng Gao; Haoyu Wang; Shuo Wang; Hongyu Lu; Jiaxin Mao; Wayne Xin Zhao; Junyi Li; Xiao Zhang
>
> **摘要:** As access to high-quality, domain-specific data grows increasingly scarce, multi-epoch training has become a practical strategy for adapting large language models (LLMs). However, autoregressive models often suffer from performance degradation under repeated data exposure, where overfitting leads to a marked decline in model capability. Through empirical analysis, we trace this degradation to an imbalance in learning dynamics: predictable, low-entropy tokens are learned quickly and come to dominate optimization, while the model's ability to generalize on high-entropy tokens deteriorates with continued training. To address this, we introduce EntroDrop, an entropy-guided token dropout method that functions as structured data regularization. EntroDrop selectively masks low-entropy tokens during training and employs a curriculum schedule to adjust regularization strength in alignment with training progress. Experiments across model scales from 0.6B to 8B parameters show that EntroDrop consistently outperforms standard regularization baselines and maintains robust performance throughout extended multi-epoch training. These findings underscore the importance of aligning regularization with token-level learning dynamics when training on limited data. Our approach offers a promising pathway toward more effective adaptation of LLMs in data-constrained domains.
>
---
#### [new 016] A Dataset and Benchmark for Consumer Healthcare Question Summarization
- **分类: cs.CL**

- **简介: 该论文属于医疗问答摘要任务，旨在解决消费者健康问题表述冗长导致的自然语言理解困难。工作包括构建领域专家标注的数据集CHQ-Sum，并验证其有效性。**

- **链接: [https://arxiv.org/pdf/2512.23637v1](https://arxiv.org/pdf/2512.23637v1)**

> **作者:** Abhishek Basu; Deepak Gupta; Dina Demner-Fushman; Shweta Yadav
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2206.06581
>
> **摘要:** The quest for seeking health information has swamped the web with consumers health-related questions. Generally, consumers use overly descriptive and peripheral information to express their medical condition or other healthcare needs, contributing to the challenges of natural language understanding. One way to address this challenge is to summarize the questions and distill the key information of the original question. Recently, large-scale datasets have significantly propelled the development of several summarization tasks, such as multi-document summarization and dialogue summarization. However, a lack of a domain-expert annotated dataset for the consumer healthcare questions summarization task inhibits the development of an efficient summarization system. To address this issue, we introduce a new dataset, CHQ-Sum,m that contains 1507 domain-expert annotated consumer health questions and corresponding summaries. The dataset is derived from the community question answering forum and therefore provides a valuable resource for understanding consumer health-related posts on social media. We benchmark the dataset on multiple state-of-the-art summarization models to show the effectiveness of the dataset
>
---
#### [new 017] Chain-of-thought Reviewing and Correction for Time Series Question Answering
- **分类: cs.CL**

- **简介: 该论文属于时间序列问答任务，解决LLM在处理复杂数值序列时的推理错误问题。提出T3LLM框架，通过多步骤推理与修正机制提升准确性。**

- **链接: [https://arxiv.org/pdf/2512.22627v1](https://arxiv.org/pdf/2512.22627v1)**

> **作者:** Chen Su; Yuanhe Tian; Yan Song
>
> **摘要:** With the advancement of large language models (LLMs), diverse time series analysis tasks are reformulated as time series question answering (TSQA) through a unified natural language interface. However, existing LLM-based approaches largely adopt general natural language processing techniques and are prone to reasoning errors when handling complex numerical sequences. Different from purely textual tasks, time series data are inherently verifiable, enabling consistency checking between reasoning steps and the original input. Motivated by this property, we propose T3LLM, which performs multi-step reasoning with an explicit correction mechanism for time series question answering. The T3LLM framework consists of three LLMs, namely, a worker, a reviewer, and a student, that are responsible for generation, review, and reasoning learning, respectively. Within this framework, the worker generates step-wise chains of thought (CoT) under structured prompts, while the reviewer inspects the reasoning, identifies erroneous steps, and provides corrective comments. The collaboratively generated corrected CoT are used to fine-tune the student model, internalizing multi-step reasoning and self-correction into its parameters. Experiments on multiple real-world TSQA benchmarks demonstrate that T3LLM achieves state-of-the-art performance over strong LLM-based baselines.
>
---
#### [new 018] Multilingual Hidden Prompt Injection Attacks on LLM-Based Academic Reviewing
- **分类: cs.CL; cs.AI**

- **简介: 论文研究LLM在学术评审中的安全问题，针对多语言文档级隐藏提示注入攻击进行实验。任务是评估LLM评审系统的脆弱性，通过构建数据集并测试不同语言攻击效果。**

- **链接: [https://arxiv.org/pdf/2512.23684v1](https://arxiv.org/pdf/2512.23684v1)**

> **作者:** Panagiotis Theocharopoulos; Ajinkya Kulkarni; Mathew Magimai. -Doss
>
> **摘要:** Large language models (LLMs) are increasingly considered for use in high-impact workflows, including academic peer review. However, LLMs are vulnerable to document-level hidden prompt injection attacks. In this work, we construct a dataset of approximately 500 real academic papers accepted to ICML and evaluate the effect of embedding hidden adversarial prompts within these documents. Each paper is injected with semantically equivalent instructions in four different languages and reviewed using an LLM. We find that prompt injection induces substantial changes in review scores and accept/reject decisions for English, Japanese, and Chinese injections, while Arabic injections produce little to no effect. These results highlight the susceptibility of LLM-based reviewing systems to document-level prompt injection and reveal notable differences in vulnerability across languages.
>
---
#### [new 019] Harnessing Large Language Models for Biomedical Named Entity Recognition
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于生物医学命名实体识别任务，解决通用大模型在该领域表现不佳的问题，提出BioSelectTune框架提升数据质量以优化模型性能。**

- **链接: [https://arxiv.org/pdf/2512.22738v1](https://arxiv.org/pdf/2512.22738v1)**

> **作者:** Jian Chen; Leilei Su; Cong Sun
>
> **摘要:** Background and Objective: Biomedical Named Entity Recognition (BioNER) is a foundational task in medical informatics, crucial for downstream applications like drug discovery and clinical trial matching. However, adapting general-domain Large Language Models (LLMs) to this task is often hampered by their lack of domain-specific knowledge and the performance degradation caused by low-quality training data. To address these challenges, we introduce BioSelectTune, a highly efficient, data-centric framework for fine-tuning LLMs that prioritizes data quality over quantity. Methods and Results: BioSelectTune reformulates BioNER as a structured JSON generation task and leverages our novel Hybrid Superfiltering strategy, a weak-to-strong data curation method that uses a homologous weak model to distill a compact, high-impact training dataset. Conclusions: Through extensive experiments, we demonstrate that BioSelectTune achieves state-of-the-art (SOTA) performance across multiple BioNER benchmarks. Notably, our model, trained on only 50% of the curated positive data, not only surpasses the fully-trained baseline but also outperforms powerful domain-specialized models like BioMedBERT.
>
---
#### [new 020] Fragile Knowledge, Robust Instruction-Following: The Width Pruning Dichotomy in Llama-3.2
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究模型剪枝对不同能力的影响，发现减少扩展比会降低知识相关任务性能，但提升指令遵循能力，揭示了剪枝的非均匀影响。属于模型压缩与性能优化任务。**

- **链接: [https://arxiv.org/pdf/2512.22671v1](https://arxiv.org/pdf/2512.22671v1)**

> **作者:** Pere Martra
>
> **备注:** 23 pages, 5 figures, 9 tables. Code available at https://github.com/peremartra/llama-glu-expansion-pruning
>
> **摘要:** Structured width pruning of GLU-MLP layers, guided by the Maximum Absolute Weight (MAW) criterion, reveals a systematic dichotomy in how reducing the expansion ratio affects different model capabilities. While performance on tasks relying on parametric knowledge (e.g., MMLU, GSM8K) and perplexity metrics degrades predictably, instruction-following capabilities improve substantially (+46% to +75% in IFEval for Llama-3.2-1B and 3B models), and multi-step reasoning remains robust (MUSR). This pattern challenges the prevailing assumption that pruning induces uniform degradation. We evaluated seven expansion ratio configurations using comprehensive benchmarks assessing factual knowledge, mathematical reasoning, language comprehension, instruction-following, and truthfulness. Our analysis identifies the expansion ratio as a critical architectural parameter that selectively modulates cognitive capabilities, rather than merely serving as a compression metric. We provide the first systematic characterization of this selective preservation phenomenon. Notably, we document a robust inverse correlation (r = -0.864, p = 0.012 in Llama-3B) between factual knowledge capacity (MMLU) and truthfulness metrics (TruthfulQA-MC2): as knowledge degrades, the model's ability to discriminate misconceptions improves consistently. This connects two previously distinct research areas, demonstrating that MAW-guided width pruning acts as a selective filter, reducing parametric knowledge while preserving or enhancing behavioral alignment. Additionally, we quantify context-dependent efficiency trade-offs: pruned configurations achieve up to 23% reduction in energy consumption (J/token) but incur penalties in single-request latency, whereas batch processing workloads benefit uniformly.
>
---
#### [new 021] Reservoir Computing inspired Matrix Multiplication-free Language Model
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型计算成本高的问题。通过设计无矩阵乘法的架构，并引入共振计算思想，减少参数和训练时间，同时保持性能。**

- **链接: [https://arxiv.org/pdf/2512.23145v1](https://arxiv.org/pdf/2512.23145v1)**

> **作者:** Takumi Shiratsuchi; Yuichiro Tanaka; Hakaru Tamukoh
>
> **备注:** 9 pages, 10 figures
>
> **摘要:** Large language models (LLMs) have achieved state-of-the-art performance in natural language processing; however, their high computational cost remains a major bottleneck. In this study, we target computational efficiency by focusing on a matrix multiplication free language model (MatMul-free LM) and further reducing the training cost through an architecture inspired by reservoir computing. Specifically, we partially fix and share the weights of selected layers in the MatMul-free LM and insert reservoir layers to obtain rich dynamic representations without additional training overhead. Additionally, several operations are combined to reduce memory accesses. Experimental results show that the proposed architecture reduces the number of parameters by up to 19%, training time by 9.9%, and inference time by 8.0%, while maintaining comparable performance to the baseline model.
>
---
#### [new 022] Mitigating Social Desirability Bias in Random Silicon Sampling
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM在模拟人类回答时的社交期望偏差问题。通过调整提示词，提升硅样本与真实数据的一致性。**

- **链接: [https://arxiv.org/pdf/2512.22725v1](https://arxiv.org/pdf/2512.22725v1)**

> **作者:** Sashank Chapala; Maksym Mironov; Songgaojun Deng
>
> **备注:** 31 pages, 9 figures, and 24 tables
>
> **摘要:** Large Language Models (LLMs) are increasingly used to simulate population responses, a method known as ``Silicon Sampling''. However, responses to socially sensitive questions frequently exhibit Social Desirability Bias (SDB), diverging from real human data toward socially acceptable answers. Existing studies on social desirability bias in LLM-based sampling remain limited. In this work, we investigate whether minimal, psychologically grounded prompt wording can mitigate this bias and improve alignment between silicon and human samples. We conducted a study using data from the American National Election Study (ANES) on three LLMs from two model families: the open-source Llama-3.1 series and GPT-4.1-mini. We first replicate a baseline silicon sampling study, confirming the persistent Social Desirability Bias. We then test four prompt-based mitigation methods: \emph{reformulated} (neutral, third-person phrasing), \emph{reverse-coded} (semantic inversion), and two meta-instructions, \emph{priming} and \emph{preamble}, respectively encouraging analytics and sincerity. Alignment with ANES is evaluated using Jensen-Shannon Divergence with bootstrap confidence intervals. Our results demonstrate that reformulated prompts most effectively improve alignment by reducing distribution concentration on socially acceptable answers and achieving distributions closer to ANES. Reverse-coding produced mixed results across eligible items, while the Priming and Preamble encouraged response uniformity and showed no systematic benefit for bias mitigation. Our findings validate the efficacy of prompt-based framing controls in mitigating inherent Social Desirability Bias in LLMs, providing a practical path toward more representative silicon samples.
>
---
#### [new 023] Interpretable Safety Alignment via SAE-Constructed Low-Rank Subspace Adaptation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于大模型微调任务，旨在解决传统方法缺乏可解释性的问题。通过SAE构建可解释的低秩子空间，提升安全对齐效果并减少参数更新量。**

- **链接: [https://arxiv.org/pdf/2512.23260v1](https://arxiv.org/pdf/2512.23260v1)**

> **作者:** Dianyun Wang; Qingsen Ma; Yuhu Shang; Zhifeng Lu; Lechen Ning; Zhenbo Xu; Huijia Wu; Zhaofeng He
>
> **摘要:** Parameter-efficient fine-tuning has become the dominant paradigm for adapting large language models to downstream tasks. Low-rank adaptation methods such as LoRA operate under the assumption that task-relevant weight updates reside in a low-rank subspace, yet this subspace is learned implicitly from data in a black-box manner, offering no interpretability or direct control. We hypothesize that this difficulty stems from polysemanticity--individual dimensions encoding multiple entangled concepts. To address this, we leverage pre-trained Sparse Autoencoders (SAEs) to identify task-relevant features in a disentangled feature space, then construct an explicit, interpretable low-rank subspace to guide adapter initialization. We provide theoretical analysis proving that under monosemanticity assumptions, SAE-based subspace identification achieves arbitrarily small recovery error, while direct identification in polysemantic space suffers an irreducible error floor. On safety alignment, our method achieves up to 99.6% safety rate--exceeding full fine-tuning by 7.4 percentage points and approaching RLHF-based methods--while updating only 0.19-0.24% of parameters. Crucially, our method provides interpretable insights into the learned alignment subspace through the semantic grounding of SAE features. Our work demonstrates that incorporating mechanistic interpretability into the fine-tuning process can simultaneously improve both performance and transparency.
>
---
#### [new 024] CNSight: Evaluation of Clinical Note Segmentation Tools
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于临床笔记分割任务，旨在解决非结构化病历数据的结构化问题。通过评估不同模型，发现大型语言模型表现最佳，为后续信息提取提供基础。**

- **链接: [https://arxiv.org/pdf/2512.22795v1](https://arxiv.org/pdf/2512.22795v1)**

> **作者:** Risha Surana; Adrian Law; Sunwoo Kim; Rishab Sridhar; Angxiao Han; Peiyu Hong
>
> **摘要:** Clinical notes are often stored in unstructured or semi-structured formats after extraction from electronic medical record (EMR) systems, which complicates their use for secondary analysis and downstream clinical applications. Reliable identification of section boundaries is a key step toward structuring these notes, as sections such as history of present illness, medications, and discharge instructions each provide distinct clinical contexts. In this work, we evaluate rule-based baselines, domain-specific transformer models, and large language models for clinical note segmentation using a curated dataset of 1,000 notes from MIMIC-IV. Our experiments show that large API-based models achieve the best overall performance, with GPT-5-mini reaching a best average F1 of 72.4 across sentence-level and freetext segmentation. Lightweight baselines remain competitive on structured sentence-level tasks but falter on unstructured freetext. Our results provide guidance for method selection and lay the groundwork for downstream tasks such as information extraction, cohort identification, and automated summarization.
>
---
#### [new 025] Fine-Tuning LLMs with Fine-Grained Human Feedback on Text Spans
- **分类: cs.CL**

- **简介: 该论文属于语言模型微调任务，旨在通过细粒度人类反馈提升模型生成质量。工作包括构建反馈链和偏好对，使模型通过逐步修正劣化部分实现更有效的偏好学习。**

- **链接: [https://arxiv.org/pdf/2512.23693v1](https://arxiv.org/pdf/2512.23693v1)**

> **作者:** Sky CH-Wang; Justin Svegliato; Helen Appel; Jason Eisner
>
> **摘要:** We present a method and dataset for fine-tuning language models with preference supervision using feedback-driven improvement chains. Given a model response, an annotator provides fine-grained feedback by marking ``liked'' and ``disliked'' spans and specifying what they liked or disliked about them. The base model then rewrites the disliked spans accordingly, proceeding from left to right, forming a sequence of incremental improvements. We construct preference pairs for direct alignment from each adjacent step in the chain, enabling the model to learn from localized, targeted edits. We find that our approach outperforms direct alignment methods based on standard A/B preference ranking or full contrastive rewrites, demonstrating that structured, revision-based supervision leads to more efficient and effective preference tuning.
>
---
#### [new 026] Learning When Not to Attend Globally
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，解决LLM全局注意力冗余问题。提出AHA机制，动态决定是否使用全局注意力，提升效率。**

- **链接: [https://arxiv.org/pdf/2512.22562v1](https://arxiv.org/pdf/2512.22562v1)**

> **作者:** Xuan Luo; Kailai Zhang; Xifeng Yan
>
> **摘要:** When reading books, humans focus primarily on the current page, flipping back to recap prior context only when necessary. Similarly, we demonstrate that Large Language Models (LLMs) can learn to dynamically determine when to attend to global context. We propose All-or-Here Attention (AHA), which utilizes a binary router per attention head to dynamically toggle between full attention and local sliding window attention for each token. Our results indicate that with a window size of 256 tokens, up to 93\% of the original full attention operations can be replaced by sliding window attention without performance loss. Furthermore, by evaluating AHA across various window sizes, we identify a long-tail distribution in context dependency, where the necessity for full attention decays rapidly as the local window expands. By decoupling local processing from global access, AHA reveals that full attention is largely redundant, and that efficient inference requires only on-demand access to the global context.
>
---
#### [new 027] LENS: LLM-Enabled Narrative Synthesis for Mental Health by Aligning Multimodal Sensing with Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出LENS框架，解决将多模态健康传感数据转化为临床语言叙述的问题。通过构建数据集和训练编码器，实现传感器数据与语言模型的对齐，提升心理健康评估的准确性与临床意义。**

- **链接: [https://arxiv.org/pdf/2512.23025v1](https://arxiv.org/pdf/2512.23025v1)**

> **作者:** Wenxuan Xu; Arvind Pillai; Subigya Nepal; Amanda C Collins; Daniel M Mackin; Michael V Heinz; Tess Z Griffin; Nicholas C Jacobson; Andrew Campbell
>
> **备注:** 22 pages, 9 figures, under review
>
> **摘要:** Multimodal health sensing offers rich behavioral signals for assessing mental health, yet translating these numerical time-series measurements into natural language remains challenging. Current LLMs cannot natively ingest long-duration sensor streams, and paired sensor-text datasets are scarce. To address these challenges, we introduce LENS, a framework that aligns multimodal sensing data with language models to generate clinically grounded mental-health narratives. LENS first constructs a large-scale dataset by transforming Ecological Momentary Assessment (EMA) responses related to depression and anxiety symptoms into natural-language descriptions, yielding over 100,000 sensor-text QA pairs from 258 participants. To enable native time-series integration, we train a patch-level encoder that projects raw sensor signals directly into an LLM's representation space. Our results show that LENS outperforms strong baselines on standard NLP metrics and task-specific measures of symptom-severity accuracy. A user study with 13 mental-health professionals further indicates that LENS-produced narratives are comprehensive and clinically meaningful. Ultimately, our approach advances LLMs as interfaces for health sensing, providing a scalable path toward models that can reason over raw behavioral signals and support downstream clinical decision-making.
>
---
#### [new 028] Structured Prompting and LLM Ensembling for Multimodal Conversational Aspect-based Sentiment Analysis
- **分类: cs.CL**

- **简介: 该论文聚焦于多模态对话中的情感分析任务，解决情感六元组提取和情感翻转检测问题。通过结构化提示和模型集成方法提升分析效果。**

- **链接: [https://arxiv.org/pdf/2512.22603v1](https://arxiv.org/pdf/2512.22603v1)**

> **作者:** Zhiqiang Gao; Shihao Gao; Zixing Zhang; Yihao Guo; Hongyu Chen; Jing Han
>
> **摘要:** Understanding sentiment in multimodal conversations is a complex yet crucial challenge toward building emotionally intelligent AI systems. The Multimodal Conversational Aspect-based Sentiment Analysis (MCABSA) Challenge invited participants to tackle two demanding subtasks: (1) extracting a comprehensive sentiment sextuple, including holder, target, aspect, opinion, sentiment, and rationale from multi-speaker dialogues, and (2) detecting sentiment flipping, which detects dynamic sentiment shifts and their underlying triggers. For Subtask-I, in the present paper, we designed a structured prompting pipeline that guided large language models (LLMs) to sequentially extract sentiment components with refined contextual understanding. For Subtask-II, we further leveraged the complementary strengths of three LLMs through ensembling to robustly identify sentiment transitions and their triggers. Our system achieved a 47.38% average score on Subtask-I and a 74.12% exact match F1 on Subtask-II, showing the effectiveness of step-wise refinement and ensemble strategies in rich, multimodal sentiment analysis tasks.
>
---
#### [new 029] AutoForge: Automated Environment Synthesis for Agentic Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于强化学习任务，旨在解决模拟环境合成不足与用户不稳定问题。提出自动化环境生成方法及环境级RL算法，提升训练效率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.22857v1](https://arxiv.org/pdf/2512.22857v1)**

> **作者:** Shihao Cai; Runnan Fang; Jialong Wu; Baixuan Li; Xinyu Wang; Yong Jiang; Liangcai Su; Liwen Zhang; Wenbiao Yin; Zhen Zhang; Fuli Feng; Pengjun Xie; Xiaobin Wang
>
> **摘要:** Conducting reinforcement learning (RL) in simulated environments offers a cost-effective and highly scalable way to enhance language-based agents. However, previous work has been limited to semi-automated environment synthesis or tasks lacking sufficient difficulty, offering little breadth or depth. In addition, the instability of simulated users integrated into these environments, along with the heterogeneity across simulated environments, poses further challenges for agentic RL. In this work, we propose: (1) a unified pipeline for automated and scalable synthesis of simulated environments associated with high-difficulty but easily verifiable tasks; and (2) an environment level RL algorithm that not only effectively mitigates user instability but also performs advantage estimation at the environment level, thereby improving training efficiency and stability. Comprehensive evaluations on agentic benchmarks, including tau-bench, tau2-Bench, and VitaBench, validate the effectiveness of our proposed method. Further in-depth analyses underscore its out-of-domain generalization.
>
---
#### [new 030] Fake News Classification in Urdu: A Domain Adaptation Approach for a Low-Resource Language
- **分类: cs.CL**

- **简介: 该论文属于虚假新闻分类任务，针对低资源语言乌尔都语，通过领域适应提升模型性能。工作包括使用多语言预训练模型并进行领域适配训练。**

- **链接: [https://arxiv.org/pdf/2512.22778v1](https://arxiv.org/pdf/2512.22778v1)**

> **作者:** Muhammad Zain Ali; Bernhard Pfahringer; Tony Smith
>
> **摘要:** Misinformation on social media is a widely acknowledged issue, and researchers worldwide are actively engaged in its detection. However, low-resource languages such as Urdu have received limited attention in this domain. An obvious approach is to utilize a multilingual pretrained language model and fine-tune it for a downstream classification task, such as misinformation detection. However, these models struggle with domain-specific terms, leading to suboptimal performance. To address this, we investigate the effectiveness of domain adaptation before fine-tuning for fake news classification in Urdu, employing a staged training approach to optimize model generalization. We evaluate two widely used multilingual models, XLM-RoBERTa and mBERT, and apply domain-adaptive pretraining using a publicly available Urdu news corpus. Experiments on four publicly available Urdu fake news datasets show that domain-adapted XLM-R consistently outperforms its vanilla counterpart, while domain-adapted mBERT exhibits mixed results.
>
---
#### [new 031] Eliciting Behaviors in Multi-Turn Conversations
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于行为提取任务，旨在解决多轮对话中复杂行为识别问题。通过分析不同方法，提出统一的多轮在线方法，并验证其效率与效果。**

- **链接: [https://arxiv.org/pdf/2512.23701v1](https://arxiv.org/pdf/2512.23701v1)**

> **作者:** Jing Huang; Shujian Zhang; Lun Wang; Andrew Hard; Rajiv Mathews; John Lambert
>
> **摘要:** Identifying specific and often complex behaviors from large language models (LLMs) in conversational settings is crucial for their evaluation. Recent work proposes novel techniques to find natural language prompts that induce specific behaviors from a target model, yet they are mainly studied in single-turn settings. In this work, we study behavior elicitation in the context of multi-turn conversations. We first offer an analytical framework that categorizes existing methods into three families based on their interactions with the target model: those that use only prior knowledge, those that use offline interactions, and those that learn from online interactions. We then introduce a generalized multi-turn formulation of the online method, unifying single-turn and multi-turn elicitation. We evaluate all three families of methods on automatically generating multi-turn test cases. We investigate the efficiency of these approaches by analyzing the trade-off between the query budget, i.e., the number of interactions with the target model, and the success rate, i.e., the discovery rate of behavior-eliciting inputs. We find that online methods can achieve an average success rate of 45/19/77% with just a few thousand queries over three tasks where static methods from existing multi-turn conversation benchmarks find few or even no failure cases. Our work highlights a novel application of behavior elicitation methods in multi-turn conversation evaluation and the need for the community to move towards dynamic benchmarks.
>
---
#### [new 032] WeDLM: Reconciling Diffusion Language Models with Standard Causal Attention for Fast Inference
- **分类: cs.CL**

- **简介: 该论文属于语言模型推理优化任务，旨在解决扩散模型并行性不足的问题。通过引入基于因果注意力的框架，提升推理速度并保持生成质量。**

- **链接: [https://arxiv.org/pdf/2512.22737v1](https://arxiv.org/pdf/2512.22737v1)**

> **作者:** Aiwei Liu; Minghua He; Shaoxun Zeng; Sijun Zhang; Linhao Zhang; Chuhan Wu; Wei Jia; Yuan Liu; Xiao Zhou; Jie Zhou
>
> **备注:** 23 pages, 8 figures, project page: https://wedlm.github.io/
>
> **摘要:** Autoregressive (AR) generation is the standard decoding paradigm for Large Language Models (LLMs), but its token-by-token nature limits parallelism at inference time. Diffusion Language Models (DLLMs) offer parallel decoding by recovering multiple masked tokens per step; however, in practice they often fail to translate this parallelism into deployment speed gains over optimized AR engines (e.g., vLLM). A key reason is that many DLLMs rely on bidirectional attention, which breaks standard prefix KV caching and forces repeated contextualization, undermining efficiency. We propose WeDLM, a diffusion decoding framework built entirely on standard causal attention to make parallel generation prefix-cache friendly. The core idea is to let each masked position condition on all currently observed tokens while keeping a strict causal mask, achieved by Topological Reordering that moves observed tokens to the physical prefix while preserving their logical positions. Building on this property, we introduce a streaming decoding procedure that continuously commits confident tokens into a growing left-to-right prefix and maintains a fixed parallel workload, avoiding the stop-and-wait behavior common in block diffusion methods. Experiments show that WeDLM preserves the quality of strong AR backbones while delivering substantial speedups, approaching 3x on challenging reasoning benchmarks and up to 10x in low-entropy generation regimes; critically, our comparisons are against AR baselines served by vLLM under matched deployment settings, demonstrating that diffusion-style decoding can outperform an optimized AR engine in practice.
>
---
#### [new 033] Evaluating GRPO and DPO for Faithful Chain-of-Thought Reasoning in LLMs
- **分类: cs.CL**

- **简介: 该论文属于大模型推理任务，旨在解决CoT解释不真实的问题。通过评估GRPO和DPO方法，提升模型推理的可信度。**

- **链接: [https://arxiv.org/pdf/2512.22631v1](https://arxiv.org/pdf/2512.22631v1)**

> **作者:** Hadi Mohammadi; Tamas Kozak; Anastasia Giachanou
>
> **摘要:** Chain-of-thought (CoT) reasoning has emerged as a powerful technique for improving the problem-solving capabilities of large language models (LLMs), particularly for tasks requiring multi-step reasoning. However, recent studies show that CoT explanations often fail to reflect the model's actual reasoning process, as models may produce coherent yet misleading justifications or modify answers without acknowledging external cues. Such discrepancies undermine the reliability of CoT-based methods for safety supervision and alignment monitoring, as models can generate plausible but deceptive rationales for incorrect answers. To better understand this limitation, we evaluate two optimization methods, Group Relative Policy Optimization (GRPO) and Direct Preference Optimization (DPO), in their ability to improve CoT faithfulness. Our experiments show that GRPO achieves higher performance than DPO in larger models, with the Qwen2.5-14B-Instruct model attaining the best results across all evaluation metrics. Both approaches exhibit positive correlations between model size and performance, but GRPO shows greater potential for improving faithfulness metrics, albeit with less stable behavior at smaller scales. These results suggest that GRPO offers a promising direction for developing more transparent and trustworthy reasoning in LLMs.
>
---
#### [new 034] Conformal Prediction Sets for Next-Token Prediction in Large Language Models: Balancing Coverage Guarantees with Set Efficiency
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型中下一个词的预测问题，旨在解决不确定性量化不足与预测集效率低的问题。通过引入VACP框架，提升覆盖率同时显著缩小预测集规模。**

- **链接: [https://arxiv.org/pdf/2512.22682v1](https://arxiv.org/pdf/2512.22682v1)**

> **作者:** Yoshith Roy Kotla; Varshith Roy Kotla
>
> **备注:** 10 pages, 5 tables and 1 algorithm
>
> **摘要:** Deploying large language models (LLMs) in high-stakes domains requires rigorous uncertainty quantification, yet standard softmax probabilities are often poorly calibrated. We present a systematic study of Adaptive Prediction Sets (APS) applied to next-token prediction in transformer-based models with large vocabularies (greater than 250,000 tokens). Our central contribution is the identification of a coverage-efficiency tradeoff: while naive conformal prediction achieves valid coverage, it produces prediction sets of hundreds of tokens, rendering them uninformative. We propose Vocabulary-Aware Conformal Prediction (VACP), a framework that leverages semantic masking and temperature-adjusted scoring to reduce the effective prediction space while provably maintaining marginal coverage. Experiments on Gemma-2B using SQUAD and WikiText benchmarks demonstrate that VACP achieves 89.7 percent empirical coverage (90 percent target) while reducing the mean prediction set size from 847 tokens to 4.3 tokens -- a 197x improvement in efficiency. We provide a theoretical analysis of vocabulary reduction and release our implementation for reproducibility.
>
---
#### [new 035] Constituency Structure over Eojeol in Korean Treebanks
- **分类: cs.CL**

- **简介: 论文探讨韩语句法树库的结构表示问题，主张以词组（eojeol）为基本单位构建句法结构，解决形态与句法混淆的问题。工作包括比较不同树库的等价性，并提出新的标注方案。**

- **链接: [https://arxiv.org/pdf/2512.22487v1](https://arxiv.org/pdf/2512.22487v1)**

> **作者:** Jungyeul Park; Chulwoo Park
>
> **摘要:** The design of Korean constituency treebanks raises a fundamental representational question concerning the choice of terminal units. Although Korean words are morphologically complex, treating morphemes as constituency terminals conflates word internal morphology with phrase level syntactic structure and creates mismatches with eojeol based dependency resources. This paper argues for an eojeol based constituency representation, with morphological segmentation and fine grained part of speech information encoded in a separate, non constituent layer. A comparative analysis shows that, under explicit normalization assumptions, the Sejong and Penn Korean treebanks can be treated as representationally equivalent at the eojeol based constituency level. Building on this result, we outline an eojeol based annotation scheme that preserves interpretable constituency and supports cross treebank comparison and constituency dependency conversion.
>
---
#### [new 036] Prompt engineering does not universally improve Large Language Model performance across clinical decision-making tasks
- **分类: cs.CL**

- **简介: 该论文属于医疗决策任务，研究Prompt engineering对LLM临床决策性能的影响。通过实验发现Prompt engineering效果因模型和任务而异，不能普遍提升性能。**

- **链接: [https://arxiv.org/pdf/2512.22966v1](https://arxiv.org/pdf/2512.22966v1)**

> **作者:** Mengdi Chai; Ali R. Zomorrodi
>
> **摘要:** Large Language Models (LLMs) have demonstrated promise in medical knowledge assessments, yet their practical utility in real-world clinical decision-making remains underexplored. In this study, we evaluated the performance of three state-of-the-art LLMs-ChatGPT-4o, Gemini 1.5 Pro, and LIama 3.3 70B-in clinical decision support across the entire clinical reasoning workflow of a typical patient encounter. Using 36 case studies, we first assessed LLM's out-of-the-box performance across five key sequential clinical decision-making tasks under two temperature settings (default vs. zero): differential diagnosis, essential immediate steps, relevant diagnostic testing, final diagnosis, and treatment recommendation. All models showed high variability by task, achieving near-perfect accuracy in final diagnosis, poor performance in relevant diagnostic testing, and moderate performance in remaining tasks. Furthermore, ChatGPT performed better under the zero temperature, whereas LIama showed stronger performance under the default temperature. Next, we assessed whether prompt engineering could enhance LLM performance by applying variations of the MedPrompt framework, incorporating targeted and random dynamic few-shot learning. The results demonstrate that prompt engineering is not a one-size-fit-all solution. While it significantly improved the performance on the task with lowest baseline accuracy (relevant diagnostic testing), it was counterproductive for others. Another key finding was that the targeted dynamic few-shot prompting did not consistently outperform random selection, indicating that the presumed benefits of closely matched examples may be counterbalanced by loss of broader contextual diversity. These findings suggest that the impact of prompt engineering is highly model and task-dependent, highlighting the need for tailored, context-aware strategies for integrating LLMs into healthcare.
>
---
#### [new 037] Coupling Experts and Routers in Mixture-of-Experts via an Auxiliary Loss
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决MoE模型中路由器与专家能力不匹配的问题。通过引入ERC损失，增强路由器决策与专家能力的耦合，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2512.23447v1](https://arxiv.org/pdf/2512.23447v1)**

> **作者:** Ang Lv; Jin Ma; Yiyuan Ma; Siyuan Qiao
>
> **摘要:** Mixture-of-Experts (MoE) models lack explicit constraints to ensure the router's decisions align well with the experts' capabilities, which ultimately limits model performance. To address this, we propose expert-router coupling (ERC) loss, a lightweight auxiliary loss that tightly couples the router's decisions with expert capabilities. Our approach treats each expert's router embedding as a proxy token for the tokens assigned to that expert, and feeds perturbed router embeddings through the experts to obtain internal activations. The ERC loss enforces two constraints on these activations: (1) Each expert must exhibit higher activation for its own proxy token than for the proxy tokens of any other expert. (2) Each proxy token must elicit stronger activation from its corresponding expert than from any other expert. These constraints jointly ensure that each router embedding faithfully represents its corresponding expert's capability, while each expert specializes in processing the tokens actually routed to it. The ERC loss is computationally efficient, operating only on n^2 activations, where n is the number of experts. This represents a fixed cost independent of batch size, unlike prior coupling methods that scale with the number of tokens (often millions per batch). Through pre-training MoE-LLMs ranging from 3B to 15B parameters and extensive analysis on trillions of tokens, we demonstrate the effectiveness of the ERC loss. Moreover, the ERC loss offers flexible control and quantitative tracking of expert specialization levels during training, providing valuable insights into MoEs.
>
---
#### [new 038] A Stepwise-Enhanced Reasoning Framework for Large Language Models Based on External Subgraph Generation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLMs在复杂推理中的准确性问题。提出SGR框架，通过外部子图生成提升推理能力。**

- **链接: [https://arxiv.org/pdf/2512.23356v1](https://arxiv.org/pdf/2512.23356v1)**

> **作者:** Xin Zhang; Yang Cao; Baoxing Wu; Xinyi Chen; Kai Song; Siying Li
>
> **摘要:** Large Language Models (LLMs) have achieved strong performance across a wide range of natural language processing tasks in recent years, including machine translation, text generation, and question answering. As their applications extend to increasingly complex scenarios, however, LLMs continue to face challenges in tasks that require deep reasoning and logical inference. In particular, models trained on large scale textual corpora may incorporate noisy or irrelevant information during generation, which can lead to incorrect predictions or outputs that are inconsistent with factual knowledge. To address this limitation, we propose a stepwise reasoning enhancement framework for LLMs based on external subgraph generation, termed SGR. The proposed framework dynamically constructs query relevant subgraphs from external knowledge bases and leverages their semantic structure to guide the reasoning process. By performing reasoning in a step by step manner over structured subgraphs, SGR reduces the influence of noisy information and improves reasoning accuracy. Specifically, the framework first generates an external subgraph tailored to the input query, then guides the model to conduct multi step reasoning grounded in the subgraph, and finally integrates multiple reasoning paths to produce the final answer. Experimental results on multiple benchmark datasets demonstrate that SGR consistently outperforms strong baselines, indicating its effectiveness in enhancing the reasoning capabilities of LLMs.
>
---
#### [new 039] Single LLM Debate, MoLaCE: Mixture of Latent Concept Experts Against Confirmation Bias
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM的输入确认偏见问题。通过引入MoLaCE框架，利用不同潜在概念专家的混合来减少偏见，提升模型的鲁棒性与准确性。**

- **链接: [https://arxiv.org/pdf/2512.23518v1](https://arxiv.org/pdf/2512.23518v1)**

> **作者:** Hazel Kim; Philip Torr
>
> **摘要:** Large language models (LLMs) are highly vulnerable to input confirmation bias. When a prompt implies a preferred answer, models often reinforce that bias rather than explore alternatives. This phenomenon remains underexplored, yet it is already harmful in base models and poses an even greater risk in multi-agent debate, where echo chambers reinforce bias instead of correction. We introduce Mixture of Latent Concept Experts (MoLaCE), a lightweight inference-time framework that addresses confirmation bias by mixing experts instantiated as different activation strengths over latent concepts that shape model responses. Our key insight is that, due to the compositional nature of language, differently phrased prompts reweight latent concepts in prompt-specific ways that affect factual correctness, so no single fixed intervention can be applied universally across inputs. This design enables a single LLM to emulate the benefits of debate internally while remaining computationally efficient and scalable. It can also be integrated into multi-agent debate frameworks to diversify perspectives and reduce correlated errors. We empirically show that it consistently reduces confirmation bias, improves robustness, and matches or surpasses multi-agent debate while requiring only a fraction of the computation.
>
---
#### [new 040] Chinese Morph Resolution in E-commerce Live Streaming Scenarios
- **分类: cs.CL**

- **简介: 该论文提出LiveAMR任务，解决电商直播中通过发音变异逃避监管的问题。构建了首个相关数据集，并利用大语言模型提升检测效果。**

- **链接: [https://arxiv.org/pdf/2512.23280v1](https://arxiv.org/pdf/2512.23280v1)**

> **作者:** Jiahao Zhu; Jipeng Qiang; Ran Bai; Chenyu Liu; Xiaoye Ouyang
>
> **摘要:** E-commerce live streaming in China, particularly on platforms like Douyin, has become a major sales channel, but hosts often use morphs to evade scrutiny and engage in false advertising. This study introduces the Live Auditory Morph Resolution (LiveAMR) task to detect such violations. Unlike previous morph research focused on text-based evasion in social media and underground industries, LiveAMR targets pronunciation-based evasion in health and medical live streams. We constructed the first LiveAMR dataset with 86,790 samples and developed a method to transform the task into a text-to-text generation problem. By leveraging large language models (LLMs) to generate additional training data, we improved performance and demonstrated that morph resolution significantly enhances live streaming regulation.
>
---
#### [new 041] SmartSnap: Proactive Evidence Seeking for Self-Verifying Agents
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; cs.MA**

- **简介: 该论文提出SmartSnap，解决自主代理任务验证效率低的问题。通过主动取证实现自我验证，提升训练效率与可靠性。**

- **链接: [https://arxiv.org/pdf/2512.22322v1](https://arxiv.org/pdf/2512.22322v1)**

> **作者:** Shaofei Cai; Yulei Qin; Haojia Lin; Zihan Xu; Gang Li; Yuchen Shi; Zongyi Li; Yong Mao; Siqi Cai; Xiaoyu Tan; Yitao Liang; Ke Li; Xing Sun
>
> **摘要:** Agentic reinforcement learning (RL) holds great promise for the development of autonomous agents under complex GUI tasks, but its scalability remains severely hampered by the verification of task completion. Existing task verification is treated as a passive, post-hoc process: a verifier (i.e., rule-based scoring script, reward or critic model, and LLM-as-a-Judge) analyzes the agent's entire interaction trajectory to determine if the agent succeeds. Such processing of verbose context that contains irrelevant, noisy history poses challenges to the verification protocols and therefore leads to prohibitive cost and low reliability. To overcome this bottleneck, we propose SmartSnap, a paradigm shift from this passive, post-hoc verification to proactive, in-situ self-verification by the agent itself. We introduce the Self-Verifying Agent, a new type of agent designed with dual missions: to not only complete a task but also to prove its accomplishment with curated snapshot evidences. Guided by our proposed 3C Principles (Completeness, Conciseness, and Creativity), the agent leverages its accessibility to the online environment to perform self-verification on a minimal, decisive set of snapshots. Such evidences are provided as the sole materials for a general LLM-as-a-Judge verifier to determine their validity and relevance. Experiments on mobile tasks across model families and scales demonstrate that our SmartSnap paradigm allows training LLM-driven agents in a scalable manner, bringing performance gains up to 26.08% and 16.66% respectively to 8B and 30B models. The synergizing between solution finding and evidence seeking facilitates the cultivation of efficient, self-verifying agents with competitive performance against DeepSeek V3.1 and Qwen3-235B-A22B.
>
---
#### [new 042] Instruction-Following Evaluation of Large Vision-Language Models
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于视觉语言模型任务，旨在解决LVLMs在微调后指令遵循能力下降的问题。通过构建新数据集，研究输出格式对指令遵循的影响，提出改进方法。**

- **链接: [https://arxiv.org/pdf/2512.23572v1](https://arxiv.org/pdf/2512.23572v1)**

> **作者:** Daiki Shiono; Shumpei Miyawaki; Ryota Tanaka; Jun Suzuki
>
> **备注:** 21 pages, 7 figures
>
> **摘要:** Following the initial flourishing of large language models (LLMs), there has been a surge in proposed large vision-language models (LVLMs) that integrate LLMs with vision capabilities. However, it has been observed that LVLMs, after tuning to visual instruction using commonly used training datasets, often fail to exhibit the instruction-following ability that was present in the LLM before integration, leading to results in which they do not follow task instructions as expected. This study quantitatively demonstrates that LVLMs' instruction-following ability declines after fine-tuning and analyzes its underlying causes. In particular, we constructed new training datasets highlighting whether the output format is specified. Then, we investigated how explicitly indicating the output format during fine-tuning affects LVLMs' instruction-following ability. Our quantitative evaluation confirmed that LVLMs' instruction-following ability declines after fine-tuning with commonly used datasets. Furthermore, we found that LVLMs trained with datasets, including instructions on output format, tend to follow instructions more accurately than models that do not. These findings suggest that including samples with instructions on output format during (visual) instruction tuning may help mitigate the decline in instruction-following abilities.
>
---
#### [new 043] LLM-Guided Exemplar Selection for Few-Shot Wearable-Sensor Human Activity Recognition
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于少样本可穿戴传感器人类活动识别任务，旨在解决传统方法依赖大量标注数据和几何选择导致的相似活动区分困难问题。通过引入大语言模型引导的示例选择框架，提升识别效果。**

- **链接: [https://arxiv.org/pdf/2512.22385v1](https://arxiv.org/pdf/2512.22385v1)**

> **作者:** Elsen Ronando; Sozo Inoue
>
> **备注:** This paper has been accepted for presentation at ABC 2026. The manuscript is under revision prior to camera-ready submission
>
> **摘要:** In this paper, we propose an LLM-Guided Exemplar Selection framework to address a key limitation in state-of-the-art Human Activity Recognition (HAR) methods: their reliance on large labeled datasets and purely geometric exemplar selection, which often fail to distinguish similar weara-ble sensor activities such as walking, walking upstairs, and walking downstairs. Our method incorporates semantic reasoning via an LLM-generated knowledge prior that captures feature importance, inter-class confusability, and exemplar budget multipliers, and uses it to guide exemplar scoring and selection. These priors are combined with margin-based validation cues, PageRank centrality, hubness penalization, and facility-location optimization to obtain a compact and informative set of exemplars. Evaluated on the UCI-HAR dataset under strict few-shot conditions, the framework achieves a macro F1-score of 88.78%, outperforming classical approaches such as random sampling, herding, and $k$-center. The results show that LLM-derived semantic priors, when integrated with structural and geometric cues, provide a stronger foundation for selecting representative sensor exemplars in few-shot wearable-sensor HAR.
>
---
#### [new 044] Close the Loop: Synthesizing Infinite Tool-Use Data via Multi-Agent Role-Playing
- **分类: cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决LLM调用外部工具的可靠性问题。通过多智能体协作生成高质量合成数据，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2512.23611v1](https://arxiv.org/pdf/2512.23611v1)**

> **作者:** Yuwen Li; Wei Zhang; Zelong Huang; Mason Yang; Jiajun Wu; Shawn Guo; Huahao Hu; Lingyi Sun; Jian Yang; Mingjie Tang; Byran Dai
>
> **摘要:** Enabling Large Language Models (LLMs) to reliably invoke external tools remains a critical bottleneck for autonomous agents. Existing approaches suffer from three fundamental challenges: expensive human annotation for high-quality trajectories, poor generalization to unseen tools, and quality ceilings inherent in single-model synthesis that perpetuate biases and coverage gaps. We introduce InfTool, a fully autonomous framework that breaks these barriers through self-evolving multi-agent synthesis. Given only raw API specifications, InfTool orchestrates three collaborative agents (User Simulator, Tool-Calling Assistant, and MCP Server) to generate diverse, verified trajectories spanning single-turn calls to complex multi-step workflows. The framework establishes a closed loop: synthesized data trains the model via Group Relative Policy Optimization (GRPO) with gated rewards, the improved model generates higher-quality data targeting capability gaps, and this cycle iterates without human intervention. Experiments on the Berkeley Function-Calling Leaderboard (BFCL) demonstrate that InfTool transforms a base 32B model from 19.8% to 70.9% accuracy (+258%), surpassing models 10x larger and rivaling Claude-Opus, and entirely from synthetic data without human annotation.
>
---
#### [new 045] Nested Browser-Use Learning for Agentic Information Seeking
- **分类: cs.CL; cs.AI; cs.IR; cs.MA**

- **简介: 该论文属于信息检索任务，旨在解决信息代理在深度网络中获取信息的效率问题。提出NestBrowse框架，通过分层结构简化代理推理，提升深度网页信息获取能力。**

- **链接: [https://arxiv.org/pdf/2512.23647v1](https://arxiv.org/pdf/2512.23647v1)**

> **作者:** Baixuan Li; Jialong Wu; Wenbiao Yin; Kuan Li; Zhongwang Zhang; Huifeng Yin; Zhengwei Tao; Liwen Zhang; Pengjun Xie; Jingren Zhou; Yong Jiang
>
> **摘要:** Information-seeking (IS) agents have achieved strong performance across a range of wide and deep search tasks, yet their tool use remains largely restricted to API-level snippet retrieval and URL-based page fetching, limiting access to the richer information available through real browsing. While full browser interaction could unlock deeper capabilities, its fine-grained control and verbose page content returns introduce substantial complexity for ReAct-style function-calling agents. To bridge this gap, we propose Nested Browser-Use Learning (NestBrowse), which introduces a minimal and complete browser-action framework that decouples interaction control from page exploration through a nested structure. This design simplifies agentic reasoning while enabling effective deep-web information acquisition. Empirical results on challenging deep IS benchmarks demonstrate that NestBrowse offers clear benefits in practice. Further in-depth analyses underscore its efficiency and flexibility.
>
---
#### [new 046] Text-Routed Sparse Mixture-of-Experts Model with Explanation and Temporal Alignment for Multi-Modal Sentiment Analysis
- **分类: cs.CL**

- **简介: 该论文属于多模态情感分析任务，旨在解决情感分析中解释性和时间对齐不足的问题。提出TEXT模型，结合多模态大语言模型和时序对齐机制，提升情感分析效果。**

- **链接: [https://arxiv.org/pdf/2512.22741v1](https://arxiv.org/pdf/2512.22741v1)**

> **作者:** Dongning Rao; Yunbiao Zeng; Zhihua Jiang; Jujian Lv
>
> **备注:** 9 pages, 9 figures, accepted by AAAI 2026
>
> **摘要:** Human-interaction-involved applications underscore the need for Multi-modal Sentiment Analysis (MSA). Although many approaches have been proposed to address the subtle emotions in different modalities, the power of explanations and temporal alignments is still underexplored. Thus, this paper proposes the Text-routed sparse mixture-of-Experts model with eXplanation and Temporal alignment for MSA (TEXT). TEXT first augments explanations for MSA via Multi-modal Large Language Models (MLLM), and then novelly aligns the epresentations of audio and video through a temporality-oriented neural network block. TEXT aligns different modalities with explanations and facilitates a new text-routed sparse mixture-of-experts with gate fusion. Our temporal alignment block merges the benefits of Mamba and temporal cross-attention. As a result, TEXT achieves the best performance cross four datasets among all tested models, including three recently proposed approaches and three MLLMs. TEXT wins on at least four metrics out of all six metrics. For example, TEXT decreases the mean absolute error to 0.353 on the CH-SIMS dataset, which signifies a 13.5% decrement compared with recently proposed approaches.
>
---
#### [new 047] Automatic Detection of Complex Quotation Patterns in Aggadic Literature
- **分类: cs.CL**

- **简介: 该论文属于文本重用检测任务，旨在解决犹太教经典文献中复杂引用模式的自动识别问题。提出ACT算法，通过三阶段处理提升检测效果。**

- **链接: [https://arxiv.org/pdf/2512.23504v1](https://arxiv.org/pdf/2512.23504v1)**

> **作者:** Hadar Miller; Tsvi Kuflik; Moshe Lavee
>
> **备注:** This paper is under review at Cogent Arts & Humanities
>
> **摘要:** This paper presents ACT (Allocate Connections between Texts), a novel three-stage algorithm for the automatic detection of biblical quotations in Rabbinic literature. Unlike existing text reuse frameworks that struggle with short, paraphrased, or structurally embedded quotations, ACT combines a morphology-aware alignment algorithm with a context-sensitive enrichment stage that identifies complex citation patterns such as "Wave" and "Echo" quotations. Our approach was evaluated against leading systems, including Dicta, Passim, Text-Matcher, as well as human-annotated critical editions. We further assessed three ACT configurations to isolate the contribution of each component. Results demonstrate that the full ACT pipeline (ACT-QE) outperforms all baselines, achieving an F1 score of 0.91, with superior Recall (0.89) and Precision (0.94). Notably, ACT-2, which lacks stylistic enrichment, achieves higher Recall (0.90) but suffers in Precision, while ACT-3, using longer n-grams, offers a tradeoff between coverage and specificity. In addition to improving quotation detection, ACT's ability to classify stylistic patterns across corpora opens new avenues for genre classification and intertextual analysis. This work contributes to digital humanities and computational philology by addressing the methodological gap between exhaustive machine-based detection and human editorial judgment. ACT lays a foundation for broader applications in historical textual analysis, especially in morphologically rich and citation-dense traditions like Aggadic literature.
>
---
#### [new 048] Not too long do read: Evaluating LLM-generated extreme scientific summaries
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于科学摘要生成任务，旨在解决LLMs生成极端科学摘要的质量评估问题。研究构建了BiomedTLDR数据集，并对比分析了LLMs与人类摘要的差异。**

- **链接: [https://arxiv.org/pdf/2512.23206v1](https://arxiv.org/pdf/2512.23206v1)**

> **作者:** Zhuoqi Lyu; Qing Ke
>
> **摘要:** High-quality scientific extreme summary (TLDR) facilitates effective science communication. How do large language models (LLMs) perform in generating them? How are LLM-generated summaries different from those written by human experts? However, the lack of a comprehensive, high-quality scientific TLDR dataset hinders both the development and evaluation of LLMs' summarization ability. To address these, we propose a novel dataset, BiomedTLDR, containing a large sample of researcher-authored summaries from scientific papers, which leverages the common practice of including authors' comments alongside bibliography items. We then test popular open-weight LLMs for generating TLDRs based on abstracts. Our analysis reveals that, although some of them successfully produce humanoid summaries, LLMs generally exhibit a greater affinity for the original text's lexical choices and rhetorical structures, hence tend to be more extractive rather than abstractive in general, compared to humans. Our code and datasets are available at https://github.com/netknowledge/LLM_summarization (Lyu and Ke, 2025).
>
---
#### [new 049] AI Meets Brain: Memory Systems from Cognitive Neuroscience to Autonomous Agents
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于人工智能与认知科学交叉领域，旨在解决AI记忆系统与人类记忆机制融合的问题。通过分析生物与人工记忆，提出高效记忆管理方案。**

- **链接: [https://arxiv.org/pdf/2512.23343v1](https://arxiv.org/pdf/2512.23343v1)**

> **作者:** Jiafeng Liang; Hao Li; Chang Li; Jiaqi Zhou; Shixin Jiang; Zekun Wang; Changkai Ji; Zhihao Zhu; Runxuan Liu; Tao Ren; Jinlan Fu; See-Kiong Ng; Xia Liang; Ming Liu; Bing Qin
>
> **备注:** 57 pages, 5 figures
>
> **摘要:** Memory serves as the pivotal nexus bridging past and future, providing both humans and AI systems with invaluable concepts and experience to navigate complex tasks. Recent research on autonomous agents has increasingly focused on designing efficient memory workflows by drawing on cognitive neuroscience. However, constrained by interdisciplinary barriers, existing works struggle to assimilate the essence of human memory mechanisms. To bridge this gap, we systematically synthesizes interdisciplinary knowledge of memory, connecting insights from cognitive neuroscience with LLM-driven agents. Specifically, we first elucidate the definition and function of memory along a progressive trajectory from cognitive neuroscience through LLMs to agents. We then provide a comparative analysis of memory taxonomy, storage mechanisms, and the complete management lifecycle from both biological and artificial perspectives. Subsequently, we review the mainstream benchmarks for evaluating agent memory. Additionally, we explore memory security from dual perspectives of attack and defense. Finally, we envision future research directions, with a focus on multimodal memory systems and skill acquisition.
>
---
#### [new 050] C2PO: Diagnosing and Disentangling Bias Shortcuts in LLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的模型偏见问题，旨在解决LLMs中的刻板与结构偏见。通过C2PO框架，同时识别并抑制输入中的虚假特征关联，提升模型公平性与推理能力。**

- **链接: [https://arxiv.org/pdf/2512.23430v1](https://arxiv.org/pdf/2512.23430v1)**

> **作者:** Xuan Feng; Bo An; Tianlong Gu; Liang Chang; Fengrui Hao; Peipeng Yu; Shuai Zhao
>
> **摘要:** Bias in Large Language Models (LLMs) poses significant risks to trustworthiness, manifesting primarily as stereotypical biases (e.g., gender or racial stereotypes) and structural biases (e.g., lexical overlap or position preferences). However, prior paradigms typically address these in isolation, often mitigating one at the expense of exacerbating the other. To address this, we conduct a systematic exploration of these reasoning failures and identify a primary inducement: the latent spurious feature correlations within the input that drive these erroneous reasoning shortcuts. Driven by these findings, we introduce Causal-Contrastive Preference Optimization (C2PO), a unified alignment framework designed to tackle these specific failures by simultaneously discovering and suppressing these correlations directly within the optimization process. Specifically, C2PO leverages causal counterfactual signals to isolate bias-inducing features from valid reasoning paths, and employs a fairness-sensitive preference update mechanism to dynamically evaluate logit-level contributions and suppress shortcut features. Extensive experiments across multiple benchmarks covering stereotypical bias (BBQ, Unqover), structural bias (MNLI, HANS, Chatbot, MT-Bench), out-of-domain fairness (StereoSet, WinoBias), and general utility (MMLU, GSM8K) demonstrate that C2PO effectively mitigates stereotypical and structural biases while preserving robust general reasoning capabilities.
>
---
#### [new 051] Data Augmentation for Classification of Negative Pregnancy Outcomes in Imbalanced Data
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于分类任务，旨在解决负性妊娠结果数据不平衡问题。通过社交媒体数据增强现有数据集，构建NLP管道识别孕妇经历并分类。**

- **链接: [https://arxiv.org/pdf/2512.22732v1](https://arxiv.org/pdf/2512.22732v1)**

> **作者:** Md Badsha Biswas
>
> **摘要:** Infant mortality remains a significant public health concern in the United States, with birth defects identified as a leading cause. Despite ongoing efforts to understand the causes of negative pregnancy outcomes like miscarriage, stillbirths, birth defects, and premature birth, there is still a need for more comprehensive research and strategies for intervention. This paper introduces a novel approach that uses publicly available social media data, especially from platforms like Twitter, to enhance current datasets for studying negative pregnancy outcomes through observational research. The inherent challenges in utilizing social media data, including imbalance, noise, and lack of structure, necessitate robust preprocessing techniques and data augmentation strategies. By constructing a natural language processing (NLP) pipeline, we aim to automatically identify women sharing their pregnancy experiences, categorizing them based on reported outcomes. Women reporting full gestation and normal birth weight will be classified as positive cases, while those reporting negative pregnancy outcomes will be identified as negative cases. Furthermore, this study offers potential applications in assessing the causal impact of specific interventions, treatments, or prenatal exposures on maternal and fetal health outcomes. Additionally, it provides a framework for future health studies involving pregnant cohorts and comparator groups. In a broader context, our research showcases the viability of social media data as an adjunctive resource in epidemiological investigations about pregnancy outcomes.
>
---
#### [new 052] Accelerating Language Model Workflows with Prompt Choreography
- **分类: cs.CL**

- **简介: 该论文提出Prompt Choreography，用于加速语言模型在多智能体工作流中的执行。解决冗余计算导致的效率问题，通过动态缓存提升速度。**

- **链接: [https://arxiv.org/pdf/2512.23049v1](https://arxiv.org/pdf/2512.23049v1)**

> **作者:** TJ Bai; Jason Eisner
>
> **备注:** to appear in TACL (final preprint of 2025-10-12); 10 pages + appendices
>
> **摘要:** Large language models are increasingly deployed in multi-agent workflows. We introduce Prompt Choreography, a framework that efficiently executes LLM workflows by maintaining a dynamic, global KV cache. Each LLM call can attend to an arbitrary, reordered subset of previously encoded messages. Parallel calls are supported. Though caching messages' encodings sometimes gives different results from re-encoding them in a new context, we show in diverse settings that fine-tuning the LLM to work with the cache can help it mimic the original results. Prompt Choreography significantly reduces per-message latency (2.0--6.2$\times$ faster time-to-first-token) and achieves substantial end-to-end speedups ($>$2.2$\times$) in some workflows dominated by redundant computation.
>
---
#### [new 053] Diversity or Precision? A Deep Dive into Next Token Prediction
- **分类: cs.CL**

- **简介: 论文探讨了大语言模型在强化学习中的探索空间问题，提出一种改进的预训练目标，通过平衡多样性与精确性提升推理能力。任务为语言模型的预训练优化。**

- **链接: [https://arxiv.org/pdf/2512.22955v1](https://arxiv.org/pdf/2512.22955v1)**

> **作者:** Haoyuan Wu; Hai Wang; Jiajia Wu; Jinxiang Ou; Keyao Wang; Weile Chen; Zihao Zheng; Bei Yu
>
> **摘要:** Recent advancements have shown that reinforcement learning (RL) can substantially improve the reasoning abilities of large language models (LLMs). The effectiveness of such RL training, however, depends critically on the exploration space defined by the pre-trained model's token-output distribution. In this paper, we revisit the standard cross-entropy loss, interpreting it as a specific instance of policy gradient optimization applied within a single-step episode. To systematically study how the pre-trained distribution shapes the exploration potential for subsequent RL, we propose a generalized pre-training objective that adapts on-policy RL principles to supervised learning. By framing next-token prediction as a stochastic decision process, we introduce a reward-shaping strategy that explicitly balances diversity and precision. Our method employs a positive reward scaling factor to control probability concentration on ground-truth tokens and a rank-aware mechanism that treats high-ranking and low-ranking negative tokens asymmetrically. This allows us to reshape the pre-trained token-output distribution and investigate how to provide a more favorable exploration space for RL, ultimately enhancing end-to-end reasoning performance. Contrary to the intuition that higher distribution entropy facilitates effective exploration, we find that imposing a precision-oriented prior yields a superior exploration space for RL.
>
---
#### [new 054] Improving Generalization in LLM Structured Pruning via Function-Aware Neuron Grouping
- **分类: cs.CL**

- **简介: 该论文属于模型压缩任务，旨在解决结构化剪枝中因校准数据不足导致的泛化能力差问题。提出FANG方法，通过功能感知的神经元分组提升剪枝效果。**

- **链接: [https://arxiv.org/pdf/2512.23014v1](https://arxiv.org/pdf/2512.23014v1)**

> **作者:** Tao Yu; Yongqi An; Kuan Zhu; Guibo Zhu; Ming Tang; Jinqiao Wang
>
> **摘要:** Large Language Models (LLMs) demonstrate impressive performance across natural language tasks but incur substantial computational and storage costs due to their scale. Post-training structured pruning offers an efficient solution. However, when few-shot calibration sets fail to adequately reflect the pretraining data distribution, existing methods exhibit limited generalization to downstream tasks. To address this issue, we propose Function-Aware Neuron Grouping (FANG), a post-training pruning framework that alleviates calibration bias by identifying and preserving neurons critical to specific function. FANG groups neurons with similar function based on the type of semantic context they process and prunes each group independently. During importance estimation within each group, tokens that strongly correlate with the functional role of the neuron group are given higher weighting. Additionally, FANG also preserves neurons that contribute across multiple context types. To achieve a better trade-off between sparsity and performance, it allocates sparsity to each block adaptively based on its functional complexity. Experiments show that FANG improves downstream accuracy while preserving language modeling performance. It achieves the state-of-the-art (SOTA) results when combined with FLAP and OBC, two representative pruning methods. Specifically, FANG outperforms FLAP and OBC by 1.5%--8.5% in average accuracy under 30% and 40% sparsity.
>
---
#### [new 055] Anka: A Domain-Specific Language for Reliable LLM Code Generation
- **分类: cs.CL; cs.LG; cs.PL; cs.SE**

- **简介: 该论文属于代码生成任务，旨在解决LLM在复杂编程任务中的错误问题。通过设计领域特定语言Anka，减少语法歧义，提升代码生成准确性。**

- **链接: [https://arxiv.org/pdf/2512.23214v1](https://arxiv.org/pdf/2512.23214v1)**

> **作者:** Saif Khalfan Saif Al Mazrouei
>
> **备注:** 11 pages, 1 figure, 4 tables. Code and benchmarks available at https://github.com/BleBlo/Anka
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities in code generation, yet they exhibit systematic errors on complex, multi-step programming tasks. We hypothesize that these errors stem from the flexibility of general-purpose languages, which permits multiple valid approaches and requires implicit state management. To test this hypothesis, we introduce Anka, a domain-specific language (DSL) for data transformation pipelines designed with explicit, constrained syntax that reduces ambiguity in code generation. Despite having zero prior training exposure to Anka, Claude 3.5 Haiku achieves 99.9% parse success and 95.8% overall task accuracy across 100 benchmark problems. Critically, Anka demonstrates a 40 percentage point accuracy advantage over Python on multi-step pipeline tasks (100% vs. 60%), where Python's flexible syntax leads to frequent errors in operation sequencing and variable management. Cross-model validation with GPT-4o-mini confirms this advantage (+26.7 percentage points on multi-step tasks). Our results demonstrate that: (1) LLMs can learn novel DSLs entirely from in-context prompts, achieving near-native accuracy; (2) constrained syntax significantly reduces errors on complex tasks; and (3) domain-specific languages purposefully designed for LLM generation can outperform general-purpose languages on which the LLM has extensive training. We release the complete language implementation, benchmark suite, and evaluation framework to facilitate further research.
>
---
#### [new 056] Beg to Differ: Understanding Reasoning-Answer Misalignment Across Languages
- **分类: cs.CL**

- **简介: 该论文属于多语言模型推理评估任务，旨在解决模型推理与答案不一致的问题。通过分析65k个跨语言推理轨迹，发现非拉丁语系模型推理与结论匹配度更低，并提出错误分类体系。**

- **链接: [https://arxiv.org/pdf/2512.22712v1](https://arxiv.org/pdf/2512.22712v1)**

> **作者:** Anaelia Ovalle; Candace Ross; Sebastian Ruder; Adina Williams; Karen Ullrich; Mark Ibrahim; Levent Sagun
>
> **备注:** Accepted to 2025 EMNLP Multilingual Representation Learning Workshop
>
> **摘要:** Large language models demonstrate strong reasoning capabilities through chain-of-thought prompting, but whether this reasoning quality transfers across languages remains underexplored. We introduce a human-validated framework to evaluate whether model-generated reasoning traces logically support their conclusions across languages. Analyzing 65k reasoning traces from GlobalMMLU questions across 6 languages and 6 frontier models, we uncover a critical blind spot: while models achieve high task accuracy, their reasoning can fail to support their conclusions. Reasoning traces in non-Latin scripts show at least twice as much misalignment between their reasoning and conclusions than those in Latin scripts. We develop an error taxonomy through human annotation to characterize these failures, finding they stem primarily from evidential errors (unsupported claims, ambiguous facts) followed by illogical reasoning steps. Our findings demonstrate that current multilingual evaluation practices provide an incomplete picture of model reasoning capabilities and highlight the need for reasoning-aware evaluation frameworks.
>
---
#### [new 057] ManchuTTS: Towards High-Quality Manchu Speech Synthesis via Flow Matching and Hierarchical Text Representation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语音合成任务，针对满语数据稀缺和音节黏着问题，提出ManchuTTS模型，通过多层级文本表示和流匹配技术提升合成质量。**

- **链接: [https://arxiv.org/pdf/2512.22491v1](https://arxiv.org/pdf/2512.22491v1)**

> **作者:** Suhua Wang; Zifan Wang; Xiaoxin Sun; D. J. Wang; Zhanbo Liu; Xin Li
>
> **摘要:** As an endangered language, Manchu presents unique challenges for speech synthesis, including severe data scarcity and strong phonological agglutination. This paper proposes ManchuTTS(Manchu Text to Speech), a novel approach tailored to Manchu's linguistic characteristics. To handle agglutination, this method designs a three-tier text representation (phoneme, syllable, prosodic) and a cross-modal hierarchical attention mechanism for multi-granular alignment. The synthesis model integrates deep convolutional networks with a flow-matching Transformer, enabling efficient, non-autoregressive generation. This method further introduce a hierarchical contrastive loss to guide structured acoustic-linguistic correspondence. To address low-resource constraints, This method construct the first Manchu TTS dataset and employ a data augmentation strategy. Experiments demonstrate that ManchuTTS attains a MOS of 4.52 using a 5.2-hour training subset derived from our full 6.24-hour annotated corpus, outperforming all baseline models by a notable margin. Ablations confirm hierarchical guidance improves agglutinative word pronunciation accuracy (AWPA) by 31% and prosodic naturalness by 27%.
>
---
#### [new 058] GHaLIB: A Multilingual Framework for Hope Speech Detection in Low-Resource Languages
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于希望话语检测任务，旨在解决低资源语言中相关工具匮乏的问题。通过多语言框架和预训练模型提升检测效果。**

- **链接: [https://arxiv.org/pdf/2512.22705v1](https://arxiv.org/pdf/2512.22705v1)**

> **作者:** Ahmed Abdullah; Sana Fatima; Haroon Mahmood
>
> **备注:** Accepted and presented at the 15th International Arab Conference on Information Technology (ICAIT); proceedings not yet published
>
> **摘要:** Hope speech has been relatively underrepresented in Natural Language Processing (NLP). Current studies are largely focused on English, which has resulted in a lack of resources for low-resource languages such as Urdu. As a result, the creation of tools that facilitate positive online communication remains limited. Although transformer-based architectures have proven to be effective in detecting hate and offensive speech, little has been done to apply them to hope speech or, more generally, to test them across a variety of linguistic settings. This paper presents a multilingual framework for hope speech detection with a focus on Urdu. Using pretrained transformer models such as XLM-RoBERTa, mBERT, EuroBERT, and UrduBERT, we apply simple preprocessing and train classifiers for improved results. Evaluations on the PolyHope-M 2025 benchmark demonstrate strong performance, achieving F1-scores of 95.2% for Urdu binary classification and 65.2% for Urdu multi-class classification, with similarly competitive results in Spanish, German, and English. These results highlight the possibility of implementing existing multilingual models in low-resource environments, thus making it easier to identify hope speech and helping to build a more constructive digital discourse.
>
---
#### [new 059] AI4Reading: Chinese Audiobook Interpretation System Based on Multi-Agent Collaboration
- **分类: cs.CL**

- **简介: 该论文提出AI4Reading系统，解决人工制作有声书耗时费力的问题。通过多智能体协作，生成结构合理、内容准确的有声书解读。**

- **链接: [https://arxiv.org/pdf/2512.23300v1](https://arxiv.org/pdf/2512.23300v1)**

> **作者:** Minjiang Huang; Jipeng Qiang; Yi Zhu; Chaowei Zhang; Xiangyu Zhao; Kui Yu
>
> **备注:** ACL 2025 demo
>
> **摘要:** Audiobook interpretations are attracting increasing attention, as they provide accessible and in-depth analyses of books that offer readers practical insights and intellectual inspiration. However, their manual creation process remains time-consuming and resource-intensive. To address this challenge, we propose AI4Reading, a multi-agent collaboration system leveraging large language models (LLMs) and speech synthesis technology to generate podcast, like audiobook interpretations. The system is designed to meet three key objectives: accurate content preservation, enhanced comprehensibility, and a logical narrative structure. To achieve these goals, we develop a framework composed of 11 specialized agents,including topic analysts, case analysts, editors, a narrator, and proofreaders that work in concert to explore themes, extract real world cases, refine content organization, and synthesize natural spoken language. By comparing expert interpretations with our system's output, the results show that although AI4Reading still has a gap in speech generation quality, the generated interpretative scripts are simpler and more accurate.
>
---
#### [new 060] Towards Efficient Post-Training via Fourier-Driven Adapter Architectures
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型微调任务，旨在解决参数效率问题。提出FAA框架，通过傅里叶特征分解中间表示，实现高效微调。**

- **链接: [https://arxiv.org/pdf/2512.22378v1](https://arxiv.org/pdf/2512.22378v1)**

> **作者:** Donggyun Bae; Jongil Park
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** We propose a novel framework, termed Fourier-Activated Adapter (FAA), for parameter-efficient fine-tuning of large pre-trained language models. By incorporating random Fourier features into lightweight adapter modules, FAA decomposes intermediate representations into complementary low- and high-frequency components, enabling frequency-aware modulation of semantic information. This design allows the model to selectively emphasize informative frequency bands during adaptation while preserving the representational capacity of the frozen backbone. Extensive experiments on GLUE, E2E NLG, and instruction-tuning benchmarks demonstrate that FAA consistently achieves competitive or superior performance compared to existing parameter-efficient fine-tuning methods, while maintaining low computational and memory overhead. Ablation studies further verify the effectiveness of frequency-aware activation and adaptive weighting mechanisms, highlighting FAA as a robust and efficient approach for post-training large language models.
>
---
#### [new 061] TabiBERT: A Large-Scale ModernBERT Foundation Model and Unified Benchmarking Framework for Turkish
- **分类: cs.CL**

- **简介: 该论文提出TabiBERT，一个基于ModernBERT的土耳其语单语模型，解决土耳其语NLP缺乏先进架构模型的问题。通过大规模预训练和基准测试，提升任务性能。**

- **链接: [https://arxiv.org/pdf/2512.23065v1](https://arxiv.org/pdf/2512.23065v1)**

> **作者:** Melikşah Türker; A. Ebrar Kızıloğlu; Onur Güngör; Susan Üsküdarlı
>
> **备注:** 31 pages, 1 figure, 13 tables
>
> **摘要:** Since the inception of BERT, encoder-only Transformers have evolved significantly in computational efficiency, training stability, and long-context modeling. ModernBERT consolidates these advances by integrating Rotary Positional Embeddings (RoPE), FlashAttention, and refined normalization. Despite these developments, Turkish NLP lacks a monolingual encoder trained from scratch incorporating such modern architectural paradigms. This work introduces TabiBERT, a monolingual Turkish encoder based on ModernBERT architecture trained from scratch on a large, curated corpus. TabiBERT is pre-trained on one trillion tokens sampled from an 84.88B token multi-domain corpus: web text (73%), scientific publications (20%), source code (6%), and mathematical content (0.3%). The model supports 8,192-token context length (16x original BERT), achieves up to 2.65x inference speedup, and reduces GPU memory consumption, enabling larger batch sizes. We introduce TabiBench with 28 datasets across eight task categories with standardized splits and protocols, evaluated using GLUE-style macro-averaging. TabiBERT attains 77.58 on TabiBench, outperforming BERTurk by 1.62 points and establishing state-of-the-art on five of eight categories: question answering (+9.55), code retrieval (+2.41), and document retrieval (+0.60). Compared with task-specific prior best results, including specialized models like TurkishBERTweet, TabiBERT achieves +1.47 average improvement, indicating robust cross-domain generalization. We release model weights, training configurations, and evaluation code for transparent, reproducible Turkish encoder research.
>
---
#### [new 062] M2G-Eval: Enhancing and Evaluating Multi-granularity Multilingual Code Generation
- **分类: cs.CL**

- **简介: 该论文属于代码生成任务，旨在解决现有评估基准在多语言和多粒度上的不足。提出M2G-Eval框架，覆盖四层粒度和18种语言，评估30个模型，揭示任务难度与语言相关性。**

- **链接: [https://arxiv.org/pdf/2512.22628v1](https://arxiv.org/pdf/2512.22628v1)**

> **作者:** Fanglin Xu; Wei Zhang; Jian Yang; Guo Chen; Aishan Liu; Zhoujun Li; Xianglong Liu; Bryan Dai
>
> **摘要:** The rapid advancement of code large language models (LLMs) has sparked significant research interest in systematically evaluating their code generation capabilities, yet existing benchmarks predominantly assess models at a single structural granularity and focus on limited programming languages, obscuring fine-grained capability variations across different code scopes and multilingual scenarios. We introduce M2G-Eval, a multi-granularity, multilingual framework for evaluating code generation in large language models (LLMs) across four levels: Class, Function, Block, and Line. Spanning 18 programming languages, M2G-Eval includes 17K+ training tasks and 1,286 human-annotated, contamination-controlled test instances. We develop M2G-Eval-Coder models by training Qwen3-8B with supervised fine-tuning and Group Relative Policy Optimization. Evaluating 30 models (28 state-of-the-art LLMs plus our two M2G-Eval-Coder variants) reveals three main findings: (1) an apparent difficulty hierarchy, with Line-level tasks easiest and Class-level most challenging; (2) widening performance gaps between full- and partial-granularity languages as task complexity increases; and (3) strong cross-language correlations, suggesting that models learn transferable programming concepts. M2G-Eval enables fine-grained diagnosis of code generation capabilities and highlights persistent challenges in synthesizing complex, long-form code.
>
---
#### [new 063] Scoring, Reasoning, and Selecting the Best! Ensembling Large Language Models via a Peer-Review Process
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出LLM-PeerReview，用于集成多个大语言模型的输出。任务是提升模型回答质量，通过评分、推理和选择最佳响应来解决单一模型性能不足的问题。**

- **链接: [https://arxiv.org/pdf/2512.23213v1](https://arxiv.org/pdf/2512.23213v1)**

> **作者:** Zhijun Chen; Zeyu Ji; Qianren Mao; Junhang Cheng; Bangjie Qin; Hao Wu; Zhuoran Li; Jingzheng Li; Kai Sun; Zizhe Wang; Yikun Ban; Zhu Sun; Xiangyang Ji; Hailong Sun
>
> **摘要:** We propose LLM-PeerReview, an unsupervised LLM Ensemble method that selects the most ideal response from multiple LLM-generated candidates for each query, harnessing the collective wisdom of multiple models with diverse strengths. LLM-PeerReview is built on a novel, peer-review-inspired framework that offers a clear and interpretable mechanism, while remaining fully unsupervised for flexible adaptability and generalization. Specifically, it operates in three stages: For scoring, we use the emerging LLM-as-a-Judge technique to evaluate each response by reusing multiple LLMs at hand; For reasoning, we can apply a principled graphical model-based truth inference algorithm or a straightforward averaging strategy to aggregate multiple scores to produce a final score for each response; Finally, the highest-scoring response is selected as the best ensemble output. LLM-PeerReview is conceptually simple and empirically powerful. The two variants of the proposed approach obtain strong results across four datasets, including outperforming the recent advanced model Smoothie-Global by 6.9% and 7.3% points, respectively.
>
---
#### [new 064] ClinDEF: A Dynamic Evaluation Framework for Large Language Models in Clinical Reasoning
- **分类: cs.CL**

- **简介: 该论文提出ClinDEF框架，用于评估大语言模型在临床推理中的表现。针对现有基准缺乏动态交互和多维度评价的问题，通过模拟诊断对话进行多轮互动评估。**

- **链接: [https://arxiv.org/pdf/2512.23440v1](https://arxiv.org/pdf/2512.23440v1)**

> **作者:** Yuqi Tang; Jing Yu; Zichang Su; Kehua Feng; Zhihui Zhu; Libin Wang; Lei Liang; Qiang Zhang; Keyan Ding; Huajun Chen
>
> **备注:** 23 pages, 4 figures, under review
>
> **摘要:** Clinical diagnosis begins with doctor-patient interaction, during which physicians iteratively gather information, determine examination and refine differential diagnosis through patients' response. This dynamic clinical-reasoning process is poorly represented by existing LLM benchmarks that focus on static question-answering. To mitigate these gaps, recent methods explore dynamic medical frameworks involving interactive clinical dialogues. Although effective, they often rely on limited, contamination-prone datasets and lack granular, multi-level evaluation. In this work, we propose ClinDEF, a dynamic framework for assessing clinical reasoning in LLMs through simulated diagnostic dialogues. Grounded in a disease knowledge graph, our method dynamically generates patient cases and facilitates multi-turn interactions between an LLM-based doctor and an automated patient agent. Our evaluation protocol goes beyond diagnostic accuracy by incorporating fine-grained efficiency analysis and rubric-based assessment of diagnostic quality. Experiments show that ClinDEF effectively exposes critical clinical reasoning gaps in state-of-the-art LLMs, offering a more nuanced and clinically meaningful evaluation paradigm.
>
---
#### [new 065] Less is more: Probabilistic reduction is best explained by small-scale predictability measures
- **分类: cs.CL**

- **简介: 该论文属于语言模型与认知研究任务，探讨语言模型概率与认知现象的关系。解决的问题是是否需要完整语句来观察概率缩减，工作表明n-gram足够作为认知单位。**

- **链接: [https://arxiv.org/pdf/2512.23659v1](https://arxiv.org/pdf/2512.23659v1)**

> **作者:** Cassandra L. Jacobs; Andrés Buxó-Lugo; Anna K. Taylor; Marie Leopold-Hooke
>
> **摘要:** The primary research questions of this paper center on defining the amount of context that is necessary and/or appropriate when investigating the relationship between language model probabilities and cognitive phenomena. We investigate whether whole utterances are necessary to observe probabilistic reduction and demonstrate that n-gram representations suffice as cognitive units of planning.
>
---
#### [new 066] The Syntax of qulk-clauses in Yemeni Ibbi Arabic: A Minimalist Approach
- **分类: cs.CL**

- **简介: 该论文研究Yemeni Ibbi阿拉伯语中的qulk-clause结构，属于句法分析任务。旨在解释其语法机制，通过最小主义框架分析其构造与演变过程。**

- **链接: [https://arxiv.org/pdf/2512.22376v1](https://arxiv.org/pdf/2512.22376v1)**

> **作者:** Zubaida Mohammed Albadani; Mohammed Q. Shormani
>
> **备注:** 23 pages, 6 figures
>
> **摘要:** This study investigates the syntax of qulk-clauses in Yemeni Ibbi Arabic (YIA) within the Minimalist Program. The construction qulk-clause, a morphologically fused form meaning 'I said,' introduces embedded declarative interrogative, and imperative clauses, often eithout complementizer. The central proposal of this paper is that qulk-clauses are biclausal structures in which qulk functions a clause-embedding predicate sec;ecting a dull CP complement. By applying core minimalist operations, viz., Merge, Move, Agree, and Spell-out, the study provides a layered syntactic analysis of qulk-clauses, for illustrating how their derivation proceeds through standard computational steps and post-syntactic processes such as Morphological Merger. The proposal also accounts for dialect-specific features like bipartite negation, cliticization, and CP embedding. The findings offer theoretical contributions to generative syntax, specifically minimalism. The study concludes raising theoretical questions concerning extending the analysis to the addressee-clause kil-k 'you said'. It also provides insights into the possibility of the universality of minimalism.
>
---
#### [new 067] On the Existence and Behaviour of Secondary Attention Sinks
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究注意力机制中的次级注意力陷阱，分析其形成与影响。属于自然语言处理任务，旨在解决注意力分配不均的问题，通过实验揭示次级陷阱的特性及与主陷阱的关系。**

- **链接: [https://arxiv.org/pdf/2512.22213v1](https://arxiv.org/pdf/2512.22213v1)**

> **作者:** Jeffrey T. H. Wong; Cheng Zhang; Louis Mahon; Wayne Luk; Anton Isopoussu; Yiren Zhao
>
> **摘要:** Attention sinks are tokens, often the beginning-of-sequence (BOS) token, that receive disproportionately high attention despite limited semantic relevance. In this work, we identify a class of attention sinks, which we term secondary sinks, that differ fundamentally from the sinks studied in prior works, which we term primary sinks. While prior works have identified that tokens other than BOS can sometimes become sinks, they were found to exhibit properties analogous to the BOS token. Specifically, they emerge at the same layer, persist throughout the network and draw a large amount of attention mass. Whereas, we find the existence of secondary sinks that arise primarily in middle layers and can persist for a variable number of layers, and draw a smaller, but still significant, amount of attention mass. Through extensive experiments across 11 model families, we analyze where these secondary sinks appear, their properties, how they are formed, and their impact on the attention mechanism. Specifically, we show that: (1) these sinks are formed by specific middle-layer MLP modules; these MLPs map token representations to vectors that align with the direction of the primary sink of that layer. (2) The $\ell_2$-norm of these vectors determines the sink score of the secondary sink, and also the number of layers it lasts for, thereby leading to different impacts on the attention mechanisms accordingly. (3) The primary sink weakens in middle layers, coinciding with the emergence of secondary sinks. We observe that in larger-scale models, the location and lifetime of the sinks, together referred to as sink levels, appear in a more deterministic and frequent manner. Specifically, we identify three sink levels in QwQ-32B and six levels in Qwen3-14B.
>
---
#### [new 068] CubeBench: Diagnosing Interactive, Long-Horizon Spatial Reasoning Under Partial Observations
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出CubeBench，用于评估大语言模型在部分观察下的空间推理和长期规划能力，解决物理世界部署中的认知挑战。**

- **链接: [https://arxiv.org/pdf/2512.23328v1](https://arxiv.org/pdf/2512.23328v1)**

> **作者:** Huan-ang Gao; Zikang Zhang; Tianwei Luo; Kaisen Yang; Xinzhe Juan; Jiahao Qiu; Tianxing Chen; Bingxiang He; Hao Zhao; Hao Zhou; Shilong Liu; Mengdi Wang
>
> **备注:** Webpage: https://cubebench.c7w.tech/
>
> **摘要:** Large Language Model (LLM) agents, while proficient in the digital realm, face a significant gap in physical-world deployment due to the challenge of forming and maintaining a robust spatial mental model. We identify three core cognitive challenges hindering this transition: spatial reasoning, long-horizon state tracking via mental simulation, and active exploration under partial observation. To isolate and evaluate these faculties, we introduce CubeBench, a novel generative benchmark centered on the Rubik's Cube. CubeBench uses a three-tiered diagnostic framework that progressively assesses agent capabilities, from foundational state tracking with full symbolic information to active exploration with only partial visual data. Our experiments on leading LLMs reveal critical limitations, including a uniform 0.00% pass rate on all long-horizon tasks, exposing a fundamental failure in long-term planning. We also propose a diagnostic framework to isolate these cognitive bottlenecks by providing external solver tools. By analyzing the failure modes, we provide key insights to guide the development of more physically-grounded intelligent agents.
>
---
#### [new 069] Theoretical Foundations of Scaling Law in Familial Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于机器学习领域，解决传统神经网络缩放定律无法适应多子模型部署的问题。通过引入粒度变量，建立统一的缩放定律模型，实现高效训练与灵活部署。**

- **链接: [https://arxiv.org/pdf/2512.23407v1](https://arxiv.org/pdf/2512.23407v1)**

> **作者:** Huan Song; Qingfei Zhao; Ting Long; Shuyu Tian; Hongjun An; Jiawei Shao; Chi Zhang; Xuelong Li
>
> **摘要:** Neural scaling laws have become foundational for optimizing large language model (LLM) training, yet they typically assume a single dense model output. This limitation effectively overlooks "Familial models, a transformative paradigm essential for realizing ubiquitous intelligence across heterogeneous device-edge-cloud hierarchies. Transcending static architectures, familial models integrate early exits with relay-style inference to spawn G deployable sub-models from a single shared backbone. In this work, we theoretically and empirically extend the scaling law to capture this "one-run, many-models" paradigm by introducing Granularity (G) as a fundamental scaling variable alongside model size (N) and training tokens (D). To rigorously quantify this relationship, we propose a unified functional form L(N, D, G) and parameterize it using large-scale empirical runs. Specifically, we employ a rigorous IsoFLOP experimental design to strictly isolate architectural impact from computational scale. Across fixed budgets, we systematically sweep model sizes (N) and granularities (G) while dynamically adjusting tokens (D). This approach effectively decouples the marginal cost of granularity from the benefits of scale, ensuring high-fidelity parameterization of our unified scaling law. Our results reveal that the granularity penalty follows a multiplicative power law with an extremely small exponent. Theoretically, this bridges fixed-compute training with dynamic architectures. Practically, it validates the "train once, deploy many" paradigm, demonstrating that deployment flexibility is achievable without compromising the compute-optimality of dense baselines.
>
---
#### [new 070] Web World Models
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出Web World Model（WWM），解决语言代理在持久世界中行动、记忆与学习的问题。通过结合网页代码与大模型，构建可控且开放的环境。任务为构建可扩展的虚拟世界模型。**

- **链接: [https://arxiv.org/pdf/2512.23676v1](https://arxiv.org/pdf/2512.23676v1)**

> **作者:** Jichen Feng; Yifan Zhang; Chenggong Zhang; Yifu Lu; Shilong Liu; Mengdi Wang
>
> **备注:** Project Page: https://github.com/Princeton-AI2-Lab/Web-World-Models
>
> **摘要:** Language agents increasingly require persistent worlds in which they can act, remember, and learn. Existing approaches sit at two extremes: conventional web frameworks provide reliable but fixed contexts backed by databases, while fully generative world models aim for unlimited environments at the expense of controllability and practical engineering. In this work, we introduce the Web World Model (WWM), a middle ground where world state and ``physics'' are implemented in ordinary web code to ensure logical consistency, while large language models generate context, narratives, and high-level decisions on top of this structured latent state. We build a suite of WWMs on a realistic web stack, including an infinite travel atlas grounded in real geography, fictional galaxy explorers, web-scale encyclopedic and narrative worlds, and simulation- and game-like environments. Across these systems, we identify practical design principles for WWMs: separating code-defined rules from model-driven imagination, representing latent state as typed web interfaces, and utilizing deterministic generation to achieve unlimited but structured exploration. Our results suggest that web stacks themselves can serve as a scalable substrate for world models, enabling controllable yet open-ended environments. Project Page: https://github.com/Princeton-AI2-Lab/Web-World-Models.
>
---
#### [new 071] Dream-VL & Dream-VLA: Open Vision-Language and Vision-Language-Action Models with Diffusion Language Model Backbone
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言与视觉语言动作模型任务，旨在解决传统模型在复杂视觉规划和动态控制中的效率问题。通过构建基于扩散语言模型的Dream-VL和Dream-VLA，提升任务性能与训练效率。**

- **链接: [https://arxiv.org/pdf/2512.22615v1](https://arxiv.org/pdf/2512.22615v1)**

> **作者:** Jiacheng Ye; Shansan Gong; Jiahui Gao; Junming Fan; Shuang Wu; Wei Bi; Haoli Bai; Lifeng Shang; Lingpeng Kong
>
> **摘要:** While autoregressive Large Vision-Language Models (VLMs) have achieved remarkable success, their sequential generation often limits their efficacy in complex visual planning and dynamic robotic control. In this work, we investigate the potential of constructing Vision-Language Models upon diffusion-based large language models (dLLMs) to overcome these limitations. We introduce Dream-VL, an open diffusion-based VLM (dVLM) that achieves state-of-the-art performance among previous dVLMs. Dream-VL is comparable to top-tier AR-based VLMs trained on open data on various benchmarks but exhibits superior potential when applied to visual planning tasks. Building upon Dream-VL, we introduce Dream-VLA, a dLLM-based Vision-Language-Action model (dVLA) developed through continuous pre-training on open robotic datasets. We demonstrate that the natively bidirectional nature of this diffusion backbone serves as a superior foundation for VLA tasks, inherently suited for action chunking and parallel generation, leading to significantly faster convergence in downstream fine-tuning. Dream-VLA achieves top-tier performance of 97.2% average success rate on LIBERO, 71.4% overall average on SimplerEnv-Bridge, and 60.5% overall average on SimplerEnv-Fractal, surpassing leading models such as $π_0$ and GR00T-N1. We also validate that dVLMs surpass AR baselines on downstream tasks across different training objectives. We release both Dream-VL and Dream-VLA to facilitate further research in the community.
>
---
#### [new 072] Scaling Unverifiable Rewards: A Case Study on Visual Insights
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究多阶段任务中奖励不可验证的问题，提出Selective TTS框架，通过分阶段优化提升视觉分析结果质量。**

- **链接: [https://arxiv.org/pdf/2512.22650v1](https://arxiv.org/pdf/2512.22650v1)**

> **作者:** Shuyu Gan; James Mooney; Pan Hao; Renxiang Wang; Mingyi Hong; Qianwen Wang; Dongyeop Kang
>
> **备注:** 32 pages, 25 figures
>
> **摘要:** Large Language Model (LLM) agents can increasingly automate complex reasoning through Test-Time Scaling (TTS), iterative refinement guided by reward signals. However, many real-world tasks involve multi-stage pipeline whose final outcomes lack verifiable rewards or sufficient data to train robust reward models, making judge-based refinement prone to accumulate error over stages. We propose Selective TTS, a process-based refinement framework that scales inference across different stages in multi-agent pipeline, instead of repeated refinement over time by prior work. By distributing compute across stages and pruning low-quality branches early using process-specific judges, Selective TTS mitigates the judge drift and stabilizes refinement. Grounded in the data science pipeline, we build an end-to-end multi-agent pipeline for generating visually insightful charts and report of given dataset, and design a reliable LLM-based judge model, aligned with human experts (Kendall's τ=0.55). Our proposed selective TTS then improves insight quality under a fixed compute budget, increasing mean scores from 61.64 to 65.86 while reducing variance. We hope our findings serve as the first step toward to scaling complex, open-ended tasks with unverifiable rewards, such as scientific discovery and story generation.
>
---
#### [new 073] Replay Failures as Successes: Sample-Efficient Reinforcement Learning for Instruction Following
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于指令遵循任务，解决RL样本效率低的问题。提出HiR框架，通过重放失败尝试作为成功样本来提升学习效率。**

- **链接: [https://arxiv.org/pdf/2512.23457v1](https://arxiv.org/pdf/2512.23457v1)**

> **作者:** Kongcheng Zhang; Qi Yao; Shunyu Liu; Wenjian Zhang; Min Cen; Yang Zhou; Wenkai Fang; Yiru Zhao; Baisheng Lai; Mingli Song
>
> **摘要:** Reinforcement Learning (RL) has shown promise for aligning Large Language Models (LLMs) to follow instructions with various constraints. Despite the encouraging results, RL improvement inevitably relies on sampling successful, high-quality responses; however, the initial model often struggles to generate responses that satisfy all constraints due to its limited capabilities, yielding sparse or indistinguishable rewards that impede learning. In this work, we propose Hindsight instruction Replay (HiR), a novel sample-efficient RL framework for complex instruction following tasks, which employs a select-then-rewrite strategy to replay failed attempts as successes based on the constraints that have been satisfied in hindsight. We perform RL on these replayed samples as well as the original ones, theoretically framing the objective as dual-preference learning at both the instruction- and response-level to enable efficient optimization using only a binary reward signal. Extensive experiments demonstrate that the proposed HiR yields promising results across different instruction following tasks, while requiring less computational budget. Our code and dataset is available at https://github.com/sastpg/HIR.
>
---
#### [new 074] AFA-LoRA: Enabling Non-Linear Adaptations in LoRA with Activation Function Annealing
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于参数高效微调任务，旨在解决LoRA线性适应表达能力不足的问题。通过引入渐进式激活函数，提升模型非线性表达能力，同时保持可合并性。**

- **链接: [https://arxiv.org/pdf/2512.22455v1](https://arxiv.org/pdf/2512.22455v1)**

> **作者:** Jiacheng Li; Jianchao Tan; Zhidong Yang; Feiye Huo; Yerui Sun; Yuchen Xie; Xunliang Cai
>
> **摘要:** Low-Rank Adaptation (LoRA) is a widely adopted parameter-efficient fine-tuning (PEFT) method. However, its linear adaptation process limits its expressive power. This means there is a gap between the expressive power of linear training and non-linear training. To bridge this gap, we propose AFA-LoRA, a novel training strategy that brings non-linear expressivity to LoRA while maintaining its seamless mergeability. Our key innovation is an annealed activation function that transitions from a non-linear to a linear transformation during training, allowing the adapter to initially adopt stronger representational capabilities before converging to a mergeable linear form. We implement our method on supervised fine-tuning, reinforcement learning, and speculative decoding. The results show that AFA-LoRA reduces the performance gap between LoRA and full-parameter training. This work enables a more powerful and practical paradigm of parameter-efficient adaptation.
>
---
#### [new 075] SciEvalKit: An Open-source Evaluation Toolkit for Scientific General Intelligence
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出SciEvalKit，一个用于评估科学智能的开源工具包，解决跨学科AI模型评估问题，涵盖多种科学任务和领域。**

- **链接: [https://arxiv.org/pdf/2512.22334v1](https://arxiv.org/pdf/2512.22334v1)**

> **作者:** Yiheng Wang; Yixin Chen; Shuo Li; Yifan Zhou; Bo Liu; Hengjian Gao; Jiakang Yuan; Jia Bu; Wanghan Xu; Yuhao Zhou; Xiangyu Zhao; Zhiwang Zhou; Fengxiang Wang; Haodong Duan; Songyang Zhang; Jun Yao; Han Deng; Yizhou Wang; Jiabei Xiao; Jiaqi Liu; Encheng Su; Yujie Liu; Weida Wang; Junchi Yao; Shenghe Zheng; Haoran Sun; Runmin Ma; Xiangchao Yan; Bo Zhang; Dongzhan Zhou; Shufei Zhang; Peng Ye; Xiaosong Wang; Shixiang Tang; Wenlong Zhang; Lei Bai
>
> **摘要:** We introduce SciEvalKit, a unified benchmarking toolkit designed to evaluate AI models for science across a broad range of scientific disciplines and task capabilities. Unlike general-purpose evaluation platforms, SciEvalKit focuses on the core competencies of scientific intelligence, including Scientific Multimodal Perception, Scientific Multimodal Reasoning, Scientific Multimodal Understanding, Scientific Symbolic Reasoning, Scientific Code Generation, Science Hypothesis Generation and Scientific Knowledge Understanding. It supports six major scientific domains, spanning from physics and chemistry to astronomy and materials science. SciEvalKit builds a foundation of expert-grade scientific benchmarks, curated from real-world, domain-specific datasets, ensuring that tasks reflect authentic scientific challenges. The toolkit features a flexible, extensible evaluation pipeline that enables batch evaluation across models and datasets, supports custom model and dataset integration, and provides transparent, reproducible, and comparable results. By bridging capability-based evaluation and disciplinary diversity, SciEvalKit offers a standardized yet customizable infrastructure to benchmark the next generation of scientific foundation models and intelligent agents. The toolkit is open-sourced and actively maintained to foster community-driven development and progress in AI4Science.
>
---
#### [new 076] Training AI Co-Scientists Using Rubric Rewards
- **分类: cs.LG; cs.CL; cs.HC**

- **简介: 该论文属于AI辅助科研任务，解决语言模型生成符合约束的研究计划问题。通过自动构建训练数据并使用强化学习提升模型生成能力。**

- **链接: [https://arxiv.org/pdf/2512.23707v1](https://arxiv.org/pdf/2512.23707v1)**

> **作者:** Shashwat Goel; Rishi Hazra; Dulhan Jayalath; Timon Willi; Parag Jain; William F. Shen; Ilias Leontiadis; Francesco Barbieri; Yoram Bachrach; Jonas Geiping; Chenxi Whitehouse
>
> **备注:** 11 pages in the main paper, total 119 including sample outputs in the Appendix
>
> **摘要:** AI co-scientists are emerging as a tool to assist human researchers in achieving their research goals. A crucial feature of these AI co-scientists is the ability to generate a research plan given a set of aims and constraints. The plan may be used by researchers for brainstorming, or may even be implemented after further refinement. However, language models currently struggle to generate research plans that follow all constraints and implicit requirements. In this work, we study how to leverage the vast corpus of existing research papers to train language models that generate better research plans. We build a scalable, diverse training corpus by automatically extracting research goals and goal-specific grading rubrics from papers across several domains. We then train models for research plan generation via reinforcement learning with self-grading. A frozen copy of the initial policy acts as the grader during training, with the rubrics creating a generator-verifier gap that enables improvements without external human supervision. To validate this approach, we conduct a study with human experts for machine learning research goals, spanning 225 hours. The experts prefer plans generated by our finetuned Qwen3-30B-A3B model over the initial model for 70% of research goals, and approve 84% of the automatically extracted goal-specific grading rubrics. To assess generality, we also extend our approach to research goals from medical papers, and new arXiv preprints, evaluating with a jury of frontier models. Our finetuning yields 12-22% relative improvements and significant cross-domain generalization, proving effective even in problem settings like medical research where execution feedback is infeasible. Together, these findings demonstrate the potential of a scalable, automated training recipe as a step towards improving general AI co-scientists.
>
---
#### [new 077] VideoScaffold: Elastic-Scale Visual Hierarchies for Streaming Video Understanding in MLLMs
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于视频理解任务，解决长视频在MLLM中因帧冗余和时序不连贯带来的挑战。提出VideoScaffold框架，通过动态事件分割与层次整合实现高效视频流理解。**

- **链接: [https://arxiv.org/pdf/2512.22226v1](https://arxiv.org/pdf/2512.22226v1)**

> **作者:** Naishan Zheng; Jie Huang; Qingpei Guo; Feng Zhao
>
> **备注:** 11 pages, 4 figures
>
> **摘要:** Understanding long videos with multimodal large language models (MLLMs) remains challenging due to the heavy redundancy across frames and the need for temporally coherent representations. Existing static strategies, such as sparse sampling, frame compression, and clustering, are optimized for offline settings and often produce fragmented or over-compressed outputs when applied to continuous video streams. We present VideoScaffold, a dynamic representation framework designed for streaming video understanding. It adaptively adjusts event granularity according to video duration while preserving fine-grained visual semantics. VideoScaffold introduces two key components: Elastic-Scale Event Segmentation (EES), which performs prediction-guided segmentation to dynamically refine event boundaries, and Hierarchical Event Consolidation (HEC), which progressively aggregates semantically related segments into multi-level abstractions. Working in concert, EES and HEC enable VideoScaffold to transition smoothly from fine-grained frame understanding to abstract event reasoning as the video stream unfolds. Extensive experiments across both offline and streaming video understanding benchmarks demonstrate that VideoScaffold achieves state-of-the-art performance. The framework is modular and plug-and-play, seamlessly extending existing image-based MLLMs to continuous video comprehension. The code is available at https://github.com/zheng980629/VideoScaffold.
>
---
#### [new 078] Monadic Context Engineering
- **分类: cs.AI; cs.CL; cs.FL**

- **简介: 该论文提出Monadic Context Engineering（MCE），解决AI代理架构的脆弱性问题，通过函数式编程结构提升状态管理与并发处理能力。属于AI代理设计任务。**

- **链接: [https://arxiv.org/pdf/2512.22431v1](https://arxiv.org/pdf/2512.22431v1)**

> **作者:** Yifan Zhang; Mengdi Wang
>
> **备注:** Project Page: https://github.com/yifanzhang-pro/monadic-context-engineering
>
> **摘要:** The proliferation of Large Language Models (LLMs) has catalyzed a shift towards autonomous agents capable of complex reasoning and tool use. However, current agent architectures are frequently constructed using imperative, ad hoc patterns. This results in brittle systems plagued by difficulties in state management, error handling, and concurrency. This paper introduces Monadic Context Engineering (MCE), a novel architectural paradigm leveraging the algebraic structures of Functors, Applicative Functors, and Monads to provide a formal foundation for agent design. MCE treats agent workflows as computational contexts where cross-cutting concerns, such as state propagation, short-circuiting error handling, and asynchronous execution, are managed intrinsically by the algebraic properties of the abstraction. We demonstrate how Monads enable robust sequential composition, how Applicatives provide a principled structure for parallel execution, and crucially, how Monad Transformers allow for the systematic composition of these capabilities. This layered approach enables developers to construct complex, resilient, and efficient AI agents from simple, independently verifiable components. We further extend this framework to describe Meta-Agents, which leverage MCE for generative orchestration, dynamically creating and managing sub-agent workflows through metaprogramming. Project Page: https://github.com/yifanzhang-pro/monadic-context-engineering.
>
---
#### [new 079] Unbiased Visual Reasoning with Controlled Visual Inputs
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉推理任务，旨在解决VLMs依赖伪相关而非因果证据的问题。提出VISTA框架，通过分离感知与推理模块，提升模型的鲁棒性和可解释性。**

- **链接: [https://arxiv.org/pdf/2512.22183v1](https://arxiv.org/pdf/2512.22183v1)**

> **作者:** Zhaonan Li; Shijie Lu; Fei Wang; Jacob Dineen; Xiao Ye; Zhikun Xu; Siyi Liu; Young Min Cho; Bangzheng Li; Daniel Chang; Kenny Nguyen; Qizheng Yang; Muhao Chen; Ben Zhou
>
> **摘要:** End-to-end Vision-language Models (VLMs) often answer visual questions by exploiting spurious correlations instead of causal visual evidence, and can become more shortcut-prone when fine-tuned. We introduce VISTA (Visual-Information Separation for Text-based Analysis), a modular framework that decouples perception from reasoning via an explicit information bottleneck. A frozen VLM sensor is restricted to short, objective perception queries, while a text-only LLM reasoner decomposes each question, plans queries, and aggregates visual facts in natural language. This controlled interface defines a reward-aligned environment for training unbiased visual reasoning with reinforcement learning. Instantiated with Qwen2.5-VL and Llama3.2-Vision sensors, and trained with GRPO from only 641 curated multi-step questions, VISTA significantly improves robustness to real-world spurious correlations on SpuriVerse (+16.29% with Qwen-2.5-VL-7B and +6.77% with Llama-3.2-Vision-11B), while remaining competitive on MMVP and a balanced SeedBench subset. VISTA transfers robustly across unseen VLM sensors and is able to recognize and recover from VLM perception failures. Human analysis further shows that VISTA's reasoning traces are more neutral, less reliant on spurious attributes, and more explicitly grounded in visual evidence than end-to-end VLM baselines.
>
---
#### [new 080] Learning from Negative Examples: Why Warning-Framed Training Data Teaches What It Warns Against
- **分类: cs.LG; cs.CL; cs.CR**

- **简介: 该论文研究语言模型如何从警告性训练数据中学习，发现模型未有效避免被警告的行为。任务为理解模型对警告的响应机制，解决模型未能正确解读警告的问题，通过实验与分析揭示其原因。**

- **链接: [https://arxiv.org/pdf/2512.22293v1](https://arxiv.org/pdf/2512.22293v1)**

> **作者:** Tsogt-Ochir Enkhbayar
>
> **备注:** Submitted to Neel Nanda's MATS Stream
>
> **摘要:** Warning-framed content in training data (e.g., "DO NOT USE - this code is vulnerable") does not, it turns out, teach language models to avoid the warned-against behavior. In experiments reported here, models exposed to such warnings reproduced the flagged content at rates statistically indistinguishable from models given the content directly (76.7% vs. 83.3%). Why? Sparse autoencoder analysis points to a failure of orthogonalization: "describing X" and "performing X" activate overlapping latent features. Feature #8684, which tracks code execution patterns, fires at comparable magnitude in both warning and exploitation contexts. A related phenomenon, what I call "stealth slip", allows conversational preambles to rotate activations into subspaces that linear probes miss entirely. Prompting and inference-time steering do not fix this; training-time feature ablation does. The upshot is that statistical co-occurrence dominates over pragmatic interpretation in current architectures. Models learn what tends to follow a context, not why it appeared there.
>
---
#### [new 081] A CNN-Based Malaria Diagnosis from Blood Cell Images with SHAP and LIME Explainability
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于医学图像分类任务，旨在解决 malaria 诊断效率低的问题。通过构建 CNN 模型并结合 SHAP 和 LIME 提升可解释性，实现快速准确的疟疾检测。**

- **链接: [https://arxiv.org/pdf/2512.22205v1](https://arxiv.org/pdf/2512.22205v1)**

> **作者:** Md. Ismiel Hossen Abir; Awolad Hossain
>
> **摘要:** Malaria remains a prevalent health concern in regions with tropical and subtropical climates. The cause of malaria is the Plasmodium parasite, which is transmitted through the bites of infected female Anopheles mosquitoes. Traditional diagnostic methods, such as microscopic blood smear analysis, are low in sensitivity, depend on expert judgment, and require resources that may not be available in remote settings. To overcome these limitations, this study proposes a deep learning-based approach utilizing a custom Convolutional Neural Network (CNN) to automatically classify blood cell images as parasitized or uninfected. The model achieves an accuracy of 96%, with precision and recall scores exceeding 0.95 for both classes. This study also compares the custom CNN with established deep learning architectures, including ResNet50, VGG16, MobileNetV2, and DenseNet121. To enhance model interpretability, Explainable AI techniques such as SHAP, LIME, and Saliency Maps are applied. The proposed system shows how deep learning can provide quick, accurate and understandable malaria diagnosis, especially in areas with limited resources.
>
---
#### [new 082] A Note on Hybrid Online Reinforcement and Imitation Learning for LLMs: Formulations and Algorithms
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于大语言模型微调任务，旨在融合模仿学习与强化学习。通过分解目标函数，提出两种梯度计算方法，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2512.23097v1](https://arxiv.org/pdf/2512.23097v1)**

> **作者:** Yingru Li; Ziniu Li; Jiacai Liu
>
> **摘要:** We present a unified framework for Large Language Model (LLM) fine-tuning that integrates Imitation Learning and Reinforcement Learning. By analyzing the gradient of a composite objective combining trajectory-level KL divergence with task rewards, we derive a natural decomposition into two components: (1) an analytically computable Dense Gradient for token-level imitation, and (2) a Monte Carlo estimated Sparse Gradient for long-horizon reward optimization. The Dense Gradient admits a closed-form logit-level formula, enabling efficient GPU implementation.
>
---
#### [new 083] Debugging Tabular Log as Dynamic Graphs
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于日志调试任务，旨在解决传统方法依赖大模型导致的灵活性和可扩展性问题。提出GraphLogDebugger框架，利用动态图模型提升调试效率。**

- **链接: [https://arxiv.org/pdf/2512.22903v1](https://arxiv.org/pdf/2512.22903v1)**

> **作者:** Chumeng Liang; Zhanyang Jin; Zahaib Akhtar; Mona Pereira; Haofei Yu; Jiaxuan You
>
> **摘要:** Tabular log abstracts objects and events in the real-world system and reports their updates to reflect the change of the system, where one can detect real-world inconsistencies efficiently by debugging corresponding log entries. However, recent advances in processing text-enriched tabular log data overly depend on large language models (LLMs) and other heavy-load models, thus suffering from limited flexibility and scalability. This paper proposes a new framework, GraphLogDebugger, to debug tabular log based on dynamic graphs. By constructing heterogeneous nodes for objects and events and connecting node-wise edges, the framework recovers the system behind the tabular log as an evolving dynamic graph. With the help of our dynamic graph modeling, a simple dynamic Graph Neural Network (GNN) is representative enough to outperform LLMs in debugging tabular log, which is validated by experimental results on real-world log datasets of computer systems and academic papers.
>
---
#### [new 084] VL-RouterBench: A Benchmark for Vision-Language Model Routing
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于多模态路由任务，旨在解决VLM路由系统评估缺乏系统基准的问题。构建了VL-RouterBench基准，涵盖多种数据集和模型，评估路由性能并促进研究改进。**

- **链接: [https://arxiv.org/pdf/2512.23562v1](https://arxiv.org/pdf/2512.23562v1)**

> **作者:** Zhehao Huang; Baijiong Lin; Jingyuan Zhang; Jingying Wang; Yuhang Liu; Ning Lu; Tao Li; Xiaolin Huang
>
> **摘要:** Multi-model routing has evolved from an engineering technique into essential infrastructure, yet existing work lacks a systematic, reproducible benchmark for evaluating vision-language models (VLMs). We present VL-RouterBench to assess the overall capability of VLM routing systems systematically. The benchmark is grounded in raw inference and scoring logs from VLMs and constructs quality and cost matrices over sample-model pairs. In scale, VL-RouterBench covers 14 datasets across 3 task groups, totaling 30,540 samples, and includes 15 open-source models and 2 API models, yielding 519,180 sample-model pairs and a total input-output token volume of 34,494,977. The evaluation protocol jointly measures average accuracy, average cost, and throughput, and builds a ranking score from the harmonic mean of normalized cost and accuracy to enable comparison across router configurations and cost budgets. On this benchmark, we evaluate 10 routing methods and baselines and observe a significant routability gain, while the best current routers still show a clear gap to the ideal Oracle, indicating considerable room for improvement in router architecture through finer visual cues and modeling of textual structure. We will open-source the complete data construction and evaluation toolchain to promote comparability, reproducibility, and practical deployment in multimodal routing research.
>
---
#### [new 085] Multimodal Fact-Checking: An Agent-based Approach
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于多模态事实核查任务，旨在解决现有系统在推理和证据利用上的不足。提出RW-Post数据集和AgentFact框架，提升核查准确性和可解释性。**

- **链接: [https://arxiv.org/pdf/2512.22933v1](https://arxiv.org/pdf/2512.22933v1)**

> **作者:** Danni Xu; Shaojing Fan; Xuanang Cheng; Mohan Kankanhalli
>
> **备注:** Code and dataset will be released at https://github.com/xudanni0927/AgentFact
>
> **摘要:** The rapid spread of multimodal misinformation poses a growing challenge for automated fact-checking systems. Existing approaches, including large vision language models (LVLMs) and deep multimodal fusion methods, often fall short due to limited reasoning and shallow evidence utilization. A key bottleneck is the lack of dedicated datasets that provide complete real-world multimodal misinformation instances accompanied by annotated reasoning processes and verifiable evidence. To address this limitation, we introduce RW-Post, a high-quality and explainable dataset for real-world multimodal fact-checking. RW-Post aligns real-world multimodal claims with their original social media posts, preserving the rich contextual information in which the claims are made. In addition, the dataset includes detailed reasoning and explicitly linked evidence, which are derived from human written fact-checking articles via a large language model assisted extraction pipeline, enabling comprehensive verification and explanation. Building upon RW-Post, we propose AgentFact, an agent-based multimodal fact-checking framework designed to emulate the human verification workflow. AgentFact consists of five specialized agents that collaboratively handle key fact-checking subtasks, including strategy planning, high-quality evidence retrieval, visual analysis, reasoning, and explanation generation. These agents are orchestrated through an iterative workflow that alternates between evidence searching and task-aware evidence filtering and reasoning, facilitating strategic decision-making and systematic evidence analysis. Extensive experimental results demonstrate that the synergy between RW-Post and AgentFact substantially improves both the accuracy and interpretability of multimodal fact-checking.
>
---
#### [new 086] Agent2World: Learning to Generate Symbolic World Models via Adaptive Multi-Agent Feedback
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出Agent2World，解决生成符号世界模型的问题。通过多智能体反馈机制，提升模型生成能力和训练效果。**

- **链接: [https://arxiv.org/pdf/2512.22336v1](https://arxiv.org/pdf/2512.22336v1)**

> **作者:** Mengkang Hu; Bowei Xia; Yuran Wu; Ailing Yu; Yude Zou; Qiguang Chen; Shijian Wang; Jiarui Jin; Kexin Li; Wenxiang Jiao; Yuan Lu; Ping Luo
>
> **备注:** 48 pages, 15 tables, 7 figures, Project page: https://agent2world.github.io
>
> **摘要:** Symbolic world models (e.g., PDDL domains or executable simulators) are central to model-based planning, but training LLMs to generate such world models is limited by the lack of large-scale verifiable supervision. Current approaches rely primarily on static validation methods that fail to catch behavior-level errors arising from interactive execution. In this paper, we propose Agent2World, a tool-augmented multi-agent framework that achieves strong inference-time world-model generation and also serves as a data engine for supervised fine-tuning, by grounding generation in multi-agent feedback. Agent2World follows a three-stage pipeline: (i) A Deep Researcher agent performs knowledge synthesis by web searching to address specification gaps; (ii) A Model Developer agent implements executable world models; And (iii) a specialized Testing Team conducts adaptive unit testing and simulation-based validation. Agent2World demonstrates superior inference-time performance across three benchmarks spanning both Planning Domain Definition Language (PDDL) and executable code representations, achieving consistent state-of-the-art results. Beyond inference, Testing Team serves as an interactive environment for the Model Developer, providing behavior-aware adaptive feedback that yields multi-turn training trajectories. The model fine-tuned on these trajectories substantially improves world-model generation, yielding an average relative gain of 30.95% over the same model before training. Project page: https://agent2world.github.io.
>
---
#### [new 087] The Big Three in Marriage Talk: LLM-Assisted Analysis of Moral Ethics and Sentiment on Weibo and Xiaohongshu
- **分类: econ.GN; cs.CL**

- **简介: 该论文属于社会情感分析任务，旨在探讨中国婚姻态度变化。通过LLM分析社交媒体数据，研究公众对婚姻的情感与道德观点，揭示婚姻下降的社会心理因素。**

- **链接: [https://arxiv.org/pdf/2512.23609v1](https://arxiv.org/pdf/2512.23609v1)**

> **作者:** Frank Tian-Fang Ye; Xiaozi Gao
>
> **摘要:** China's marriage registrations have declined dramatically, dropping from 13.47 million couples in 2013 to 6.1 million in 2024. Understanding public attitudes toward marriage requires examining not only emotional sentiment but also the moral reasoning underlying these evaluations. This study analyzed 219,358 marriage-related posts from two major Chinese social media platforms (Sina Weibo and Xiaohongshu) using large language model (LLM)-assisted content analysis. Drawing on Shweder's Big Three moral ethics framework, posts were coded for sentiment (positive, negative, neutral) and moral dimensions (Autonomy, Community, Divinity). Results revealed platform differences: Weibo discourse skewed positive, while Xiaohongshu was predominantly neutral. Most posts across both platforms lacked explicit moral framing. However, when moral ethics were invoked, significant associations with sentiment emerged. Posts invoking Autonomy ethics and Community ethics were predominantly negative, whereas Divinity-framed posts tended toward neutral or positive sentiment. These findings suggest that concerns about both personal autonomy constraints and communal obligations drive negative marriage attitudes in contemporary China. The study demonstrates LLMs' utility for scaling qualitative analysis and offers insights for developing culturally informed policies addressing marriage decline in Chinese contexts.
>
---
## 更新

#### [replaced 001] Forecasting Clinical Risk from Textual Time Series: Structuring Narratives for Temporal AI in Healthcare
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗时间序列预测任务，旨在利用文本时间序列进行临床风险预测。通过构建时间有序的文本数据，评估不同模型在事件预测和生存分析中的表现。**

- **链接: [https://arxiv.org/pdf/2504.10340v5](https://arxiv.org/pdf/2504.10340v5)**

> **作者:** Shahriar Noroozizadeh; Sayantan Kumar; Jeremy C. Weiss
>
> **备注:** AAAI AI for Social Impact 2026. Shahriar Noroozizadeh, Sayantan Kumar (authors contributed equally)
>
> **摘要:** Clinical case reports encode temporal patient trajectories that are often underexploited by traditional machine learning methods relying on structured data. In this work, we introduce the forecasting problem from textual time series, where timestamped clinical findings -- extracted via an LLM-assisted annotation pipeline -- serve as the primary input for prediction. We systematically evaluate a diverse suite of models, including fine-tuned decoder-based large language models and encoder-based transformers, on tasks of event occurrence prediction, temporal ordering, and survival analysis. Our experiments reveal that encoder-based models consistently achieve higher F1 scores and superior temporal concordance for short- and long-horizon event forecasting, while fine-tuned masking approaches enhance ranking performance. In contrast, instruction-tuned decoder models demonstrate a relative advantage in survival analysis, especially in early prognosis settings. Our sensitivity analyses further demonstrate the importance of time ordering, which requires clinical time series construction, as compared to text ordering, the format of the text inputs that LLMs are classically trained on. This highlights the additional benefit that can be ascertained from time-ordered corpora, with implications for temporal tasks in the era of widespread LLM use.
>
---
#### [replaced 002] AgentMath: Empowering Mathematical Reasoning for Large Language Models via Tool-Augmented Agent
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出AgentMath，解决大模型在数学推理中的效率与准确性问题，通过融合语言模型与代码解释器，提升复杂数学问题求解能力。**

- **链接: [https://arxiv.org/pdf/2512.20745v2](https://arxiv.org/pdf/2512.20745v2)**

> **作者:** Haipeng Luo; Huawen Feng; Qingfeng Sun; Can Xu; Kai Zheng; Yufei Wang; Tao Yang; Han Hu; Yansong Tang; Di Wang
>
> **备注:** LLM, Mathematical Reasoning
>
> **摘要:** Large Reasoning Models (LRMs) like o3 and DeepSeek-R1 have achieved remarkable progress in natural language reasoning with long chain-of-thought. However, they remain computationally inefficient and struggle with accuracy when solving problems requiring complex mathematical operations. In this work, we present AgentMath, an agent framework that seamlessly integrates language models' reasoning capabilities with code interpreters' computational precision to efficiently tackle complex mathematical problems. Our approach introduces three key innovations: (1) An automated method that converts natural language chain-of-thought into structured tool-augmented trajectories, generating high-quality supervised fine-tuning (SFT) data to alleviate data scarcity; (2) A novel agentic reinforcement learning (RL) paradigm that dynamically interleaves natural language generation with real-time code execution. This enables models to autonomously learn optimal tool-use strategies through multi-round interactive feedback, while fostering emergent capabilities in code refinement and error correction; (3) An efficient training system incorporating innovative techniques, including request-level asynchronous rollout scheduling, agentic partial rollout, and prefix-aware weighted load balancing, achieving 4-5x speedup and making efficient RL training feasible on ultra-long sequences with scenarios with massive tool invocation. The evaluations show that AgentMath achieves state-of-the-art performance on challenging mathematical competition benchmarks including AIME24, AIME25, and HMMT25. Specifically, AgentMath-30B-A3B attains 90.6%, 86.4%, and 73.8% accuracy respectively, achieving advanced performance. The results validate the effectiveness of our approach and pave the way for building more sophisticated and scalable mathematical reasoning agents.
>
---
#### [replaced 003] Training-Free Diffusion Priors for Text-to-Image Generation via Optimization-based Visual Inversion
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于文本到图像生成任务，旨在解决传统扩散模型依赖训练过的先验网络的问题。通过优化视觉逆向方法（OVI）实现无需训练的先验替代。**

- **链接: [https://arxiv.org/pdf/2511.20821v3](https://arxiv.org/pdf/2511.20821v3)**

> **作者:** Samuele Dell'Erba; Andrew D. Bagdanov
>
> **备注:** 13 pages, 7 figures, technical report (preprint)
>
> **摘要:** Diffusion models have established the state-of-the-art in text-to-image generation, but their performance often relies on a diffusion prior network to translate text embeddings into the visual manifold for easier decoding. These priors are computationally expensive and require extensive training on massive datasets. In this work, we challenge the necessity of a trained prior at all by employing Optimization-based Visual Inversion (OVI), a training-free and zero-shot alternative, to replace the need for a prior. OVI initializes a latent visual representation from random pseudo-tokens and iteratively optimizes it to maximize the cosine similarity with the input textual prompt embedding. We further propose two novel constraints, a Mahalanobis-based and a Nearest-Neighbor loss, to regularize the OVI optimization process toward the distribution of realistic images. Our experiments, conducted on Kandinsky 2.2, show that OVI can serve as an alternative to traditional priors. More importantly, our analysis reveals a critical flaw in current evaluation benchmarks like T2I-CompBench++, where simply using the text embedding as a prior achieves surprisingly high scores, despite lower perceptual quality. Our constrained OVI methods improve visual fidelity over this baseline, with the Nearest-Neighbor approach proving particularly effective. It achieves quantitative scores comparable to or higher than the state-of-the-art data-efficient prior, underscoring the potential of optimization-based strategies as viable, training-free alternatives to traditional priors. The code will be publicly available upon acceptance.
>
---
#### [replaced 004] Leveraging Large Language Models for Rare Disease Named Entity Recognition
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于罕见疾病命名实体识别任务，旨在解决数据稀缺和语义模糊问题。通过优化提示策略，提升大语言模型在该领域的性能。**

- **链接: [https://arxiv.org/pdf/2508.09323v2](https://arxiv.org/pdf/2508.09323v2)**

> **作者:** Nan Miles Xi; Yu Deng; Lin Wang
>
> **摘要:** Named Entity Recognition (NER) in the rare disease domain poses unique challenges due to limited labeled data, semantic ambiguity between entity types, and long-tail distributions. In this study, we evaluate the capabilities of GPT-4o for rare disease NER under low-resource settings, using a range of prompt-based strategies including zero-shot prompting, few-shot in-context learning, retrieval-augmented generation (RAG), and task-level fine-tuning. We design a structured prompting framework that encodes domain-specific knowledge and disambiguation rules for four entity types. We further introduce two semantically guided few-shot example selection methods to improve in-context performance while reducing labeling effort. Experiments on the RareDis Corpus show that GPT-4o achieves competitive or superior performance compared to BioClinicalBERT, with task-level fine-tuning yielding the strongest performance among the evaluated approaches and improving upon the previously reported BioClinicalBERT baseline. Cost-performance analysis reveals that few-shot prompting delivers high returns at low token budgets. RAG provides limited overall gains but can improve recall for challenging entity types, especially signs and symptoms. An error taxonomy highlights common failure modes such as boundary drift and type confusion, suggesting opportunities for post-processing and hybrid refinement. Our results demonstrate that prompt-optimized LLMs can serve as effective, scalable alternatives to traditional supervised models in biomedical NER, particularly in rare disease applications where annotated data is scarce.
>
---
#### [replaced 005] Can Finetuing LLMs on Small Human Samples Increase Heterogeneity, Alignment, and Belief-Action Coherence?
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，探讨微调小样本人类数据是否能提升LLM的多样性、对齐度和信念-行为一致性。研究通过实验比较人类与LLM生成回应，发现微调有改善，但无法替代人类进行推断分析。**

- **链接: [https://arxiv.org/pdf/2511.21218v3](https://arxiv.org/pdf/2511.21218v3)**

> **作者:** Steven Wang; Kyle Hunt; Shaojie Tang; Kenneth Joseph
>
> **摘要:** There is ongoing debate about whether large language models (LLMs) can serve as substitutes for human participants in survey and experimental research. While recent work in fields such as marketing and psychology has explored the potential of LLM-based simulation, a growing body of evidence cautions against this practice: LLMs often fail to align with real human behavior, exhibiting limited diversity, systematic misalignment for minority subgroups, insufficient within-group variance, and discrepancies between stated beliefs and actions. This study examines an important and distinct question in this domain: whether fine-tuning on a small subset of human survey data, such as that obtainable from a pilot study, can mitigate these issues and yield realistic simulated outcomes. Using a behavioral experiment on information disclosure, we compare human and LLM-generated responses across multiple dimensions, including distributional divergence, subgroup alignment, belief-action coherence, and the recovery of regression coefficients. We find that fine-tuning on small human samples substantially improves heterogeneity, alignment, and belief-action coherence relative to the base model. However, even the best-performing fine-tuned models fail to reproduce the regression coefficients of the original study, suggesting that LLM-generated data remain unsuitable for replacing human participants in formal inferential analyses.
>
---
#### [replaced 006] Quantifying True Robustness: Synonymity-Weighted Similarity for Trustworthy XAI Evaluation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于XAI评估任务，旨在解决传统指标未能考虑同义词影响的问题。通过引入同义权重，提升对攻击真实影响的评估准确性。**

- **链接: [https://arxiv.org/pdf/2501.01516v2](https://arxiv.org/pdf/2501.01516v2)**

> **作者:** Christopher Burger
>
> **备注:** 10 pages, 2 figures, 6 tables. Changes to title, abstract and minor edits to the content as a result of acceptance to the 59th Hawaii International Conference on System Sciences
>
> **摘要:** Adversarial attacks challenge the reliability of Explainable AI (XAI) by altering explanations while the model's output remains unchanged. The success of these attacks on text-based XAI is often judged using standard information retrieval metrics. We argue these measures are poorly suited in the evaluation of trustworthiness, as they treat all word perturbations equally while ignoring synonymity, which can misrepresent an attack's true impact. To address this, we apply synonymity weighting, a method that amends these measures by incorporating the semantic similarity of perturbed words. This produces more accurate vulnerability assessments and provides an important tool for assessing the robustness of AI systems. Our approach prevents the overestimation of attack success, leading to a more faithful understanding of an XAI system's true resilience against adversarial manipulation.
>
---
#### [replaced 007] DE$^3$-BERT: Distance-Enhanced Early Exiting for BERT based on Prototypical Networks
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决BERT模型推理效率与准确性之间的平衡问题。通过结合局部与全局信息，提出DE$^3$-BERT框架，提升早期退出的可靠性。**

- **链接: [https://arxiv.org/pdf/2402.05948v2](https://arxiv.org/pdf/2402.05948v2)**

> **作者:** Jianing He; Qi Zhang; Weiping Ding; Duoqian Miao; Jun Zhao; Liang Hu; Longbing Cao
>
> **备注:** 16 pages
>
> **摘要:** Early exiting has demonstrated its effectiveness in accelerating the inference of pre-trained language models like BERT by dynamically adjusting the number of layers executed. However, most existing early exiting methods only consider local information from an individual test sample to determine their exiting indicators, failing to leverage the global information offered by sample population. This leads to suboptimal estimation of prediction correctness, resulting in erroneous exiting decisions. To bridge the gap, we explore the necessity of effectively combining both local and global information to ensure reliable early exiting during inference. Purposefully, we leverage prototypical networks to learn class prototypes and devise a distance metric between samples and class prototypes. This enables us to utilize global information for estimating the correctness of early predictions. On this basis, we propose a novel Distance-Enhanced Early Exiting framework for BERT (DE$^3$-BERT). DE$^3$-BERT implements a hybrid exiting strategy that supplements classic entropy-based local information with distance-based global information to enhance the estimation of prediction correctness for more reliable early exiting decisions. Extensive experiments on the GLUE benchmark demonstrate that DE$^3$-BERT consistently outperforms state-of-the-art models under different speed-up ratios with minimal storage or computational overhead, yielding a better trade-off between model performance and inference efficiency. Additionally, an in-depth analysis further validates the generality and interpretability of our method.
>
---
#### [replaced 008] DIF: A Framework for Benchmarking and Verifying Implicit Bias in LLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的公平性研究任务，旨在解决LLMs隐式偏见的基准测试问题。通过构建DIF框架，评估模型在不同社会背景下的表现，验证隐式偏见的存在。**

- **链接: [https://arxiv.org/pdf/2505.10013v2](https://arxiv.org/pdf/2505.10013v2)**

> **作者:** Lake Yin; Fan Huang
>
> **备注:** 10 pages, 1 figure
>
> **摘要:** As Large Language Models (LLMs) have risen in prominence over the past few years, there has been concern over the potential biases in LLMs inherited from the training data. Previous studies have examined how LLMs exhibit implicit bias, such as when response generation changes when different social contexts are introduced. We argue that this implicit bias is not only an ethical, but also a technical issue, as it reveals an inability of LLMs to accommodate extraneous information. However, unlike other measures of LLM intelligence, there are no standard methods to benchmark this specific subset of LLM bias. To bridge this gap, we developed a method for calculating an easily interpretable benchmark, DIF (Demographic Implicit Fairness), by evaluating preexisting LLM logic and math problem datasets with sociodemographic personas, which is combined with a statistical robustness check using a null model. We demonstrate that this method can validate the presence of implicit bias in LLM behavior and find an novel inverse trend between question answering accuracy and implicit bias, supporting our argument.
>
---
#### [replaced 009] Decoding EEG Speech Perception with Transformers and VAE-based Data Augmentation
- **分类: eess.AS; cs.CL; cs.HC; cs.LG; cs.SD**

- **简介: 该论文属于EEG语音解码任务，旨在解决数据噪声、数据量少及复杂任务性能差的问题。通过VAE数据增强和序列模型提升解码效果。**

- **链接: [https://arxiv.org/pdf/2501.04359v2](https://arxiv.org/pdf/2501.04359v2)**

> **作者:** Terrance Yu-Hao Chen; Yulin Chen; Pontus Soederhaell; Sadrishya Agrawal; Kateryna Shapovalenko
>
> **备注:** 19 pages, 15 figures, 2 tables
>
> **摘要:** Decoding speech from non-invasive brain signals, such as electroencephalography (EEG), has the potential to advance brain-computer interfaces (BCIs), with applications in silent communication and assistive technologies for individuals with speech impairments. However, EEG-based speech decoding faces major challenges, such as noisy data, limited datasets, and poor performance on complex tasks like speech perception. This study attempts to address these challenges by employing variational autoencoders (VAEs) for EEG data augmentation to improve data quality and applying a state-of-the-art (SOTA) sequence-to-sequence deep learning architecture, originally successful in electromyography (EMG) tasks, to EEG-based speech decoding. Additionally, we adapt this architecture for word classification tasks. Using the Brennan dataset, which contains EEG recordings of subjects listening to narrated speech, we preprocess the data and evaluate both classification and sequence-to-sequence models for EEG-to-words/sentences tasks. Our experiments show that VAEs have the potential to reconstruct artificial EEG data for augmentation. Meanwhile, our sequence-to-sequence model achieves more promising performance in generating sentences compared to our classification model, though both remain challenging tasks. These findings lay the groundwork for future research on EEG speech perception decoding, with possible extensions to speech production tasks such as silent or imagined speech.
>
---
#### [replaced 010] Vision Enhancing LLMs: Empowering Multimodal Knowledge Storage and Sharing in LLMs
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于多模态任务，旨在解决LLMs在视觉知识利用不足的问题，提出MKS2方法，通过模块化视觉记忆和多模态专家协作增强LLMs的推理能力。**

- **链接: [https://arxiv.org/pdf/2311.15759v2](https://arxiv.org/pdf/2311.15759v2)**

> **作者:** Yunxin Li; Zhenyu Liu; Baotian Hu; Wei Wang; Yuxin Ding; Xiaochun Cao; Min Zhang
>
> **备注:** 21 pages, 7 figures; Accepted by IEEE TIP
>
> **摘要:** Recent advancements in multimodal large language models (MLLMs) have achieved significant multimodal generation capabilities, akin to GPT-4. These models predominantly map visual information into language representation space, leveraging the vast knowledge and powerful text generation abilities of LLMs to produce multimodal instruction-following responses. We could term this method as LLMs for Vision because of its employing LLMs for visual understanding and reasoning, yet observe that these MLLMs neglect the potential of harnessing visual knowledge to enhance the overall capabilities of LLMs, which could be regarded as Vision Enhancing LLMs. In this paper, we propose an approach called MKS2, aimed at enhancing LLMs through empowering Multimodal Knowledge Storage and Sharing in LLMs. Specifically, we introduce Modular Visual Memory (MVM), a component integrated into the internal blocks of LLMs, designed to store open-world visual information efficiently. Additionally, we present a soft Mixture of Multimodal Experts (MoMEs) architecture in LLMs to invoke multimodal knowledge collaboration during text generation. Our comprehensive experiments demonstrate that MKS2 substantially augments the reasoning capabilities of LLMs in contexts necessitating physical or commonsense knowledge. It also delivers competitive results on image-text understanding multimodal benchmarks. The codes will be available at: https://github.com/HITsz-TMG/MKS2-Multimodal-Knowledge-Storage-and-Sharing
>
---
#### [replaced 011] Dub-S2ST: Textless Speech-to-Speech Translation for Seamless Dubbing
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音到语音翻译任务，旨在解决跨语言配音中语音模式不匹配的问题。提出一种基于离散扩散的翻译模型，实现时间对齐和语速适应，生成自然流畅的翻译结果。**

- **链接: [https://arxiv.org/pdf/2505.20899v2](https://arxiv.org/pdf/2505.20899v2)**

> **作者:** Jeongsoo Choi; Jaehun Kim; Joon Son Chung
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** This paper introduces a cross-lingual dubbing system that translates speech from one language to another while preserving key characteristics such as duration, speaker identity, and speaking speed. Despite the strong translation quality of existing speech translation approaches, they often overlook the transfer of speech patterns, leading to mismatches with source speech and limiting their suitability for dubbing applications. To address this, we propose a discrete diffusion-based speech-to-unit translation model with explicit duration control, enabling time-aligned translation. We then synthesize speech based on the translated units and source speaker's identity using a conditional flow matching model. Additionally, we introduce a unit-based speed adaptation mechanism that guides the translation model to produce speech at a rate consistent with the source, without relying on any text. Extensive experiments demonstrate that our framework generates natural and fluent translations that align with the original speech's duration and speaking pace, while achieving competitive translation performance. The code is available at https://github.com/kaistmm/Dub-S2ST.
>
---
#### [replaced 012] Computational Economics in Large Language Models: Exploring Model Behavior and Incentive Design under Resource Constraints
- **分类: cs.CL**

- **简介: 该论文属于模型优化任务，旨在解决LLM在资源约束下的效率问题。通过引入计算经济学框架，设计激励机制提升模型的计算效率与可解释性。**

- **链接: [https://arxiv.org/pdf/2508.10426v3](https://arxiv.org/pdf/2508.10426v3)**

> **作者:** Sandeep Reddy; Kabir Khan; Rohit Patil; Ananya Chakraborty; Faizan A. Khan; Swati Kulkarni; Arjun Verma; Neha Singh
>
> **备注:** Preprint; 7 figures, 4 tables, 1 algorithm. Experiments on GLUE (MNLI, STS-B, CoLA) and WikiText-103 with BERT-base; evaluation includes FLOPS, latency, Gini and entropy metrics
>
> **摘要:** Large language models (LLMs) are limited by substantial computational cost. We introduce a "computational economics" framework that treats an LLM as an internal economy of resource-constrained agents (attention heads and neuron blocks) that must allocate scarce computation to maximize task utility. First, we show empirically that when computation is scarce, standard LLMs reallocate attention toward high-value tokens while preserving accuracy. Building on this observation, we propose an incentive-driven training paradigm that augments the task loss with a differentiable computation cost term, encouraging sparse and efficient activations. On GLUE (MNLI, STS-B, CoLA) and WikiText-103, the method yields a family of models that trace a Pareto frontier and consistently dominate post-hoc pruning; for a similar accuracy we obtain roughly a forty percent reduction in FLOPS and lower latency, together with more interpretable attention patterns. These results indicate that economic principles offer a principled route to designing efficient, adaptive, and more transparent LLMs under strict resource constraints.
>
---
#### [replaced 013] Rotation Control Unlearning: Quantifying and Controlling Continuous Unlearning for LLM with The Cognitive Rotation Space
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于机器学习安全任务，旨在解决连续删除数据时模型性能下降的问题。提出RCU方法，通过旋转空间控制未学习程度，减少累积误差。**

- **链接: [https://arxiv.org/pdf/2509.25743v2](https://arxiv.org/pdf/2509.25743v2)**

> **作者:** Xiang Zhang; Kun Wei; Xu Yang; Chenghao Xu; Su Yan; Cheng Deng
>
> **摘要:** As Large Language Models (LLMs) become increasingly prevalent, their security vulnerabilities have already drawn attention. Machine unlearning is introduced to seek to mitigate these risks by removing the influence of undesirable data. However, existing methods not only rely on the retained dataset to preserve model utility, but also suffer from cumulative catastrophic utility loss under continuous unlearning requests. To solve this dilemma, we propose a novel method, called Rotation Control Unlearning (RCU), which leverages the rotational salience weight of RCU to quantify and control the unlearning degree in the continuous unlearning process. The skew symmetric loss is designed to construct the existence of the cognitive rotation space, where the changes of rotational angle can simulate the continuous unlearning process. Furthermore, we design an orthogonal rotation axes regularization to enforce mutually perpendicular rotation directions for continuous unlearning requests, effectively minimizing interference and addressing cumulative catastrophic utility loss. Experiments on multiple datasets confirm that our method without retained dataset achieves SOTA performance.
>
---
#### [replaced 014] Don't Retrieve, Generate: Prompting LLMs for Synthetic Training Data in Dense Retrieval
- **分类: cs.IR; cs.CL**

- **简介: 论文研究密集检索中合成训练数据的生成，旨在替代传统硬负样本挖掘方法。通过大语言模型生成合成负样本，但实验表明其效果不如传统方法。**

- **链接: [https://arxiv.org/pdf/2504.21015v3](https://arxiv.org/pdf/2504.21015v3)**

> **作者:** Aarush Sinha
>
> **摘要:** Training effective dense retrieval models typically relies on hard negative (HN) examples mined from large document corpora using methods such as BM25 or cross-encoders, which require full corpus access and expensive index construction. We propose generating synthetic hard negatives directly from a provided query and positive passage, using Large Language Models(LLMs). We fine-tune DistilBERT using synthetic negatives generated by four state-of-the-art LLMs ranging from 4B to 30B parameters (Qwen3, LLaMA3, Phi4) and evaluate performance across 10 BEIR benchmark datasets. Contrary to the prevailing assumption that stronger generative models yield better synthetic data, find that our generative pipeline consistently underperforms traditional corpus-based mining strategies (BM25 and Cross-Encoder). Furthermore, we observe that scaling the generator model does not monotonically improve retrieval performance and find that the 14B parameter model outperforms the 30B model and in some settings it is the worst performing.
>
---
#### [replaced 015] Cognitive Alignment in Personality Reasoning: Leveraging Prototype Theory for MBTI Inference
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本情感分析任务，旨在解决MBTI人格类型推断问题。通过构建高质量语料并利用原型理论改进模型推理，提升准确性与可解释性。**

- **链接: [https://arxiv.org/pdf/2511.00115v2](https://arxiv.org/pdf/2511.00115v2)**

> **作者:** Haoyuan Li; Yuanbo Tong; Yuchen Li; Zirui Wang; Chunhou Liu; Jiamou Liu
>
> **备注:** The authors have decided to withdraw this version to substantially revise and extend the work
>
> **摘要:** Personality recognition from text is typically cast as hard-label classification, which obscures the graded, prototype-like nature of human personality judgments. We present ProtoMBTI, a cognitively aligned framework for MBTI inference that operationalizes prototype theory within an LLM-based pipeline. First, we construct a balanced, quality-controlled corpus via LLM-guided multi-dimensional augmentation (semantic, linguistic, sentiment). Next, we LoRA-fine-tune a lightweight (<=2B) encoder to learn discriminative embeddings and to standardize a bank of personality prototypes. At inference, we retrieve top-k prototypes for a query post and perform a retrieve--reuse--revise--retain cycle: the model aggregates prototype evidence via prompt-based voting, revises when inconsistencies arise, and, upon correct prediction, retains the sample to continually enrich the prototype library. Across Kaggle and Pandora benchmarks, ProtoMBTI improves over baselines on both the four MBTI dichotomies and the full 16-type task, and exhibits robust cross-dataset generalization. Our results indicate that aligning the inference process with psychological prototype reasoning yields gains in accuracy, interpretability, and transfer for text-based personality modeling.
>
---
#### [replaced 016] Doctor Sun: A Bilingual Multimodal Large Language Model for Biomedical AI
- **分类: cs.LG; cs.AI; cs.CL; cs.MM**

- **简介: 该论文提出Doctor Sun，一个专注于医学的多模态大语言模型，解决医学数据理解与跨模态关系捕捉问题，通过两阶段训练提升医学任务性能。**

- **链接: [https://arxiv.org/pdf/2508.08270v2](https://arxiv.org/pdf/2508.08270v2)**

> **作者:** Dong Xue; Ziyao Shao; Zhaoyang Duan; Fangzhou Liu; Bing Li; Zhongheng Zhang
>
> **摘要:** Large multimodal models (LMMs) have demonstrated significant potential in providing innovative solutions for various biomedical tasks, including pathology analysis, radiology report generation, and biomedical assistance. However, the existing multimodal biomedical AI is typically based on foundation LLMs, thus hindering the understanding of intricate medical concepts with limited medical training data. Moreover, recent LLaVA-induced medical LMMs struggle to effectively capture the intricate relationship between the texts and the images. Therefore, we introduce Doctor Sun, a large multimodal generative model specialized in medicine, developed to encode, integrate, and interpret diverse biomedical data modalities such as text and images. In particular, Doctor Sun integrates a pre-trained vision encoder with a medical LLM and conducts two-stage training on various medical datasets, focusing on feature alignment and instruction tuning. Moreover, we release SunMed-VL, a wide-range bilingual medical multimodal dataset, along with all associated models, code, and resources, to freely support the advancement of biomedical multimodal research.
>
---
#### [replaced 017] ReaSeq: Unleashing World Knowledge via Reasoning for Sequential Modeling
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于推荐系统任务，解决日志驱动模式下的知识贫乏和用户兴趣局限问题。通过引入世界知识增强的推理框架ReaSeq，提升推荐效果。**

- **链接: [https://arxiv.org/pdf/2512.21257v2](https://arxiv.org/pdf/2512.21257v2)**

> **作者:** Jiakai Tang; Chuan Wang; Gaoming Yang; Han Wu; Jiahao Yu; Jian Wu; Jianwu Hu; Junjun Zheng; Longbin Li; Shuwen Xiao; Xiangheng Kong; Yeqiu Yang; Yuning Jiang; Ahjol Nurlanbek; Binbin Cao; Bo Zheng; Fangmei Zhu; Gaoming Zhou; Huimin Yi; Huiping Chu; Jin Huang; Jinzhe Shan; Kenan Cui; Longbin Li; Silu Zhou; Wen Chen; Xia Ming; Xiang Gao; Xin Yao; Xingyu Wen; Yan Zhang; Yiwen Hu; Yulin Wang; Ziheng Bao; Zongyuan Wu
>
> **摘要:** Industrial recommender systems face two fundamental limitations under the log-driven paradigm: (1) knowledge poverty in ID-based item representations that causes brittle interest modeling under data sparsity, and (2) systemic blindness to beyond-log user interests that constrains model performance within platform boundaries. These limitations stem from an over-reliance on shallow interaction statistics and close-looped feedback while neglecting the rich world knowledge about product semantics and cross-domain behavioral patterns that Large Language Models have learned from vast corpora. To address these challenges, we introduce ReaSeq, a reasoning-enhanced framework that leverages world knowledge in Large Language Models to address both limitations through explicit and implicit reasoning. Specifically, ReaSeq employs explicit Chain-of-Thought reasoning via multi-agent collaboration to distill structured product knowledge into semantically enriched item representations, and latent reasoning via Diffusion Large Language Models to infer plausible beyond-log behaviors. Deployed on Taobao's ranking system serving hundreds of millions of users, ReaSeq achieves substantial gains: >6.0% in IPV and CTR, >2.9% in Orders, and >2.5% in GMV, validating the effectiveness of world-knowledge-enhanced reasoning over purely log-driven approaches.
>
---
#### [replaced 018] MDToC: Metacognitive Dynamic Tree of Concepts for Boosting Mathematical Problem-Solving of Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出MDToC，解决LLM数学问题求解中的计算验证问题。通过构建概念树、验证计算并投票选择答案，提升数学推理能力。**

- **链接: [https://arxiv.org/pdf/2512.18841v2](https://arxiv.org/pdf/2512.18841v2)**

> **作者:** Tung Duong Ta; Tim Oates; Thien Van Luong; Huan Vu; Tien Cuong Nguyen
>
> **摘要:** Despite advances in mathematical reasoning capabilities, Large Language Models (LLMs) still struggle with calculation verification when using established prompting techniques. We present MDToC (Metacognitive Dynamic Tree of Concepts), a three-phase approach that constructs a concept tree, develops accuracy-verified calculations for each concept, and employs majority voting to evaluate competing solutions. Evaluations across CHAMP, MATH, and Game-of-24 benchmarks demonstrate our MDToC's effectiveness, with GPT-4-Turbo achieving 58.1\% on CHAMP, 86.6\% on MATH, and 85\% on Game-of-24 - outperforming GoT by 5\%, 5.4\%, and 4\% on all these tasks, respectively, without hand-engineered hints. MDToC consistently surpasses existing prompting methods across all backbone models, yielding improvements of up to 7.6\% over ToT and 6.2\% over GoT, establishing metacognitive calculation verification as a promising direction for enhanced mathematical reasoning.
>
---
#### [replaced 019] Complementary Learning Approach for Text Classification using Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本分类任务，旨在解决人机协作中的标注差异问题。通过结合人类与大语言模型的优势，提出一种高效的方法来识别和分析人机评分差异。**

- **链接: [https://arxiv.org/pdf/2512.07583v2](https://arxiv.org/pdf/2512.07583v2)**

> **作者:** Navid Asgari; Benjamin M. Cole
>
> **备注:** After further review, we identified substantive issues that materially affect the validity of the manuscript's core results and conclusions. Addressing these would require a fundamental reworking of the analysis and framing. To maintain the integrity of the public record, we request withdrawal of this version
>
> **摘要:** In this study, we propose a structured methodology that utilizes large language models (LLMs) in a cost-efficient and parsimonious manner, integrating the strengths of scholars and machines while offsetting their respective weaknesses. Our methodology, facilitated through a chain of thought and few-shot learning prompting from computer science, extends best practices for co-author teams in qualitative research to human-machine teams in quantitative research. This allows humans to utilize abductive reasoning and natural language to interrogate not just what the machine has done but also what the human has done. Our method highlights how scholars can manage inherent weaknesses OF LLMs using careful, low-cost techniques. We demonstrate how to use the methodology to interrogate human-machine rating discrepancies for a sample of 1,934 press releases announcing pharmaceutical alliances (1990-2017).
>
---
#### [replaced 020] Trusted Uncertainty in Large Language Models: A Unified Framework for Confidence Calibration and Risk-Controlled Refusal
- **分类: cs.CL**

- **简介: 该论文提出UniCR框架，解决语言模型在回答时的不确定性问题，通过融合多种证据校准置信度并控制风险，提升模型的可靠性与可信度。**

- **链接: [https://arxiv.org/pdf/2509.01455v3](https://arxiv.org/pdf/2509.01455v3)**

> **作者:** Markus Oehri; Giulia Conti; Kaviraj Pather; Alexandre Rossi; Laia Serra; Adrian Parody; Rogvi Johannesen; Aviaja Petersen; Arben Krasniqi
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** Deployed language models must decide not only what to answer but also when not to answer. We present UniCR, a unified framework that turns heterogeneous uncertainty evidence including sequence likelihoods, self-consistency dispersion, retrieval compatibility, and tool or verifier feedback into a calibrated probability of correctness and then enforces a user-specified error budget via principled refusal. UniCR learns a lightweight calibration head with temperature scaling and proper scoring, supports API-only models through black-box features, and offers distribution-free guarantees using conformal risk control. For long-form generation, we align confidence with semantic fidelity by supervising on atomic factuality scores derived from retrieved evidence, reducing confident hallucinations while preserving coverage. Experiments on short-form QA, code generation with execution tests, and retrieval-augmented long-form QA show consistent improvements in calibration metrics, lower area under the risk-coverage curve, and higher coverage at fixed risk compared to entropy or logit thresholds, post-hoc calibrators, and end-to-end selective baselines. Analyses reveal that evidence contradiction, semantic dispersion, and tool inconsistency are the dominant drivers of abstention, yielding informative user-facing refusal messages. The result is a portable recipe of evidence fusion to calibrated probability to risk-controlled decision that improves trustworthiness without fine-tuning the base model and remains valid under distribution shift.
>
---
#### [replaced 021] ICONS: Influence Consensus for Vision-Language Data Selection
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于视觉-语言模型训练任务，解决数据冗余问题。提出ICONS方法，通过梯度分析和跨任务共识选择有效数据，提升训练效率。**

- **链接: [https://arxiv.org/pdf/2501.00654v4](https://arxiv.org/pdf/2501.00654v4)**

> **作者:** Xindi Wu; Mengzhou Xia; Rulin Shao; Zhiwei Deng; Pang Wei Koh; Olga Russakovsky
>
> **摘要:** Training vision-language models via instruction tuning relies on large data mixtures spanning diverse tasks and domains, yet these mixtures frequently include redundant information that increases computational costs without proportional gains. Existing methods typically rely on task-agnostic heuristics to estimate data importance, limiting their effectiveness across tasks. We introduce ICONS, a gradient-based Influence CONsensus approach for vision-language data Selection. Our method leverages first-order training dynamics to estimate each example's influence on validation performance, then aggregates these estimates across tasks via majority voting. This cross-task consensus identifies consistently valuable data points while mitigating score calibration and outlier sensitivity, enabling robust and scalable data selection for diverse multitask mixtures. Models trained on our selected 20% data subset from LLAVA-665K (respectively: from CAMBRIAN-7M, from VISION-FLAN-186K) retain 98.6% (respectively: 98.8%, 99.8%) of full-dataset performance. We demonstrate that our selected data generalizes to unseen tasks and model architectures, and release three compact subsets LLAVA-ICONS-133K, CAMBRIAN-ICONS-1.4M, and VISION-FLAN-ICONS-37K for efficient vision-language model development.
>
---
#### [replaced 022] Information Capacity: Evaluating the Efficiency of Large Language Models via Text Compression
- **分类: cs.AI; cs.CL; eess.SP**

- **简介: 论文提出“信息容量”作为评估大语言模型效率的指标，解决模型效率衡量标准缺失的问题。通过文本压缩性能与计算复杂度的对比，实现跨模型系列的公平比较。**

- **链接: [https://arxiv.org/pdf/2511.08066v4](https://arxiv.org/pdf/2511.08066v4)**

> **作者:** Cheng Yuan; Jiawei Shao; Chi Zhang; Xuelong Li
>
> **备注:** Code: https://github.com/TeleAI-AI-Flow/InformationCapacity. Data: https://huggingface.co/datasets/TeleAI-AI-Flow/InformationCapacity
>
> **摘要:** Recent years have witnessed the rapid advancements of large language models (LLMs) and their expanding applications, leading to soaring demands for computational resources. The widespread adoption of test-time scaling further aggravates the tension between model capability and resource consumption, highlighting the importance of inference efficiency. However, a unified metric that accurately reflects an LLM's efficiency across different model sizes and architectures remains absent. Motivated by the correlation between compression and intelligence, we introduce information capacity, a measure of model efficiency based on text compression performance relative to computational complexity. Larger models can predict the next token more accurately, achieving greater compression gains but at higher computational costs. Empirical evaluations on mainstream open-source models show that models of varying sizes within a series exhibit consistent information capacity. This metric enables a fair efficiency comparison across model series and accurate performance prediction within a model series. A distinctive feature of information capacity is that it incorporates tokenizer efficiency, which affects both input and output token counts but is often neglected in LLM evaluations. We assess the information capacity of 52 models on 5 heterogeneous datasets and observe consistent results on the influences of tokenizer efficiency, pretraining data, and the mixture-of-experts architecture.
>
---
#### [replaced 023] Improving Large Language Model Safety with Contrastive Representation Learning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于模型安全任务，旨在提升大语言模型抵御攻击的鲁棒性。通过对比表示学习方法，区分良性与有害表示，增强模型安全性。**

- **链接: [https://arxiv.org/pdf/2506.11938v2](https://arxiv.org/pdf/2506.11938v2)**

> **作者:** Samuel Simko; Mrinmaya Sachan; Bernhard Schölkopf; Zhijing Jin
>
> **备注:** EMNLP 2025 Main
>
> **摘要:** Large Language Models (LLMs) are powerful tools with profound societal impacts, yet their ability to generate responses to diverse and uncontrolled inputs leaves them vulnerable to adversarial attacks. While existing defenses often struggle to generalize across varying attack types, recent advancements in representation engineering offer promising alternatives. In this work, we propose a defense framework that formulates model defense as a contrastive representation learning (CRL) problem. Our method finetunes a model using a triplet-based loss combined with adversarial hard negative mining to encourage separation between benign and harmful representations. Our experimental results across multiple models demonstrate that our approach outperforms prior representation engineering-based defenses, improving robustness against both input-level and embedding-space attacks without compromising standard performance. Our code is available at https://github.com/samuelsimko/crl-llm-defense
>
---
#### [replaced 024] RefAV: Towards Planning-Centric Scenario Mining
- **分类: cs.CV; cs.CL; cs.RO**

- **简介: 该论文属于场景挖掘任务，旨在从驾驶日志中识别安全关键场景。通过引入RefAV数据集和视觉语言模型，解决传统方法效率低、错误率高的问题。**

- **链接: [https://arxiv.org/pdf/2505.20981v3](https://arxiv.org/pdf/2505.20981v3)**

> **作者:** Cainan Davidson; Deva Ramanan; Neehar Peri
>
> **备注:** Project Page: https://cainand.github.io/RefAV/
>
> **摘要:** Autonomous Vehicles (AVs) collect and pseudo-label terabytes of multi-modal data localized to HD maps during normal fleet testing. However, identifying interesting and safety-critical scenarios from uncurated driving logs remains a significant challenge. Traditional scenario mining techniques are error-prone and prohibitively time-consuming, often relying on hand-crafted structured queries. In this work, we revisit spatio-temporal scenario mining through the lens of recent vision-language models (VLMs) to detect whether a described scenario occurs in a driving log and, if so, precisely localize it in both time and space. To address this problem, we introduce RefAV, a large-scale dataset of 10,000 diverse natural language queries that describe complex multi-agent interactions relevant to motion planning derived from 1000 driving logs in the Argoverse 2 Sensor dataset. We evaluate several referential multi-object trackers and present an empirical analysis of our baselines. Notably, we find that naively repurposing off-the-shelf VLMs yields poor performance, suggesting that scenario mining presents unique challenges. Lastly, we discuss our recently held competition and share insights from the community. Our code and dataset are available at https://github.com/CainanD/RefAV/ and https://argoverse.github.io/user-guide/tasks/scenario_mining.html
>
---
#### [replaced 025] To Bias or Not to Bias: Detecting bias in News with bias-detector
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文属于媒体偏见检测任务，旨在解决偏见识别困难和数据稀缺问题。通过微调RoBERTa模型，在BABE数据集上进行句级偏见分类，并结合已有分类器构建分析管道。**

- **链接: [https://arxiv.org/pdf/2505.13010v2](https://arxiv.org/pdf/2505.13010v2)**

> **作者:** Himel Ghosh; Ahmed Mosharafa; Georg Groh
>
> **备注:** 7 pages, 5 figures, 2 tables
>
> **摘要:** Media bias detection is a critical task in ensuring fair and balanced information dissemination, yet it remains challenging due to the subjectivity of bias and the scarcity of high-quality annotated data. In this work, we perform sentence-level bias classification by fine-tuning a RoBERTa-based model on the expert-annotated BABE dataset. Using McNemar's test and the 5x2 cross-validation paired t-test, we show statistically significant improvements in performance when comparing our model to a domain-adaptively pre-trained DA-RoBERTa baseline. Furthermore, attention-based analysis shows that our model avoids common pitfalls like oversensitivity to politically charged terms and instead attends more meaningfully to contextually relevant tokens. For a comprehensive examination of media bias, we present a pipeline that combines our model with an already-existing bias-type classifier. Our method exhibits good generalization and interpretability, despite being constrained by sentence-level analysis and dataset size because of a lack of larger and more advanced bias corpora. We talk about context-aware modeling, bias neutralization, and advanced bias type classification as potential future directions. Our findings contribute to building more robust, explainable, and socially responsible NLP systems for media bias detection.
>
---
#### [replaced 026] Step-DeepResearch Technical Report
- **分类: cs.CL**

- **简介: 该论文属于深度研究任务，旨在解决LLM在开放性研究中的不足。提出Step-DeepResearch框架，提升规划与报告能力，并建立ADR-Bench评估体系。**

- **链接: [https://arxiv.org/pdf/2512.20491v4](https://arxiv.org/pdf/2512.20491v4)**

> **作者:** Chen Hu; Haikuo Du; Heng Wang; Lin Lin; Mingrui Chen; Peng Liu; Ruihang Miao; Tianchi Yue; Wang You; Wei Ji; Wei Yuan; Wenjin Deng; Xiaojian Yuan; Xiaoyun Zhang; Xiangyu Liu; Xikai Liu; Yanming Xu; Yicheng Cao; Yifei Zhang; Yongyao Wang; Yubo Shu; Yurong Zhang; Yuxiang Zhang; Zheng Gong; Zhichao Chang; Binyan Li; Dan Ma; Furong Jia; Hongyuan Wang; Jiayu Liu; Jing Bai; Junlan Liu; Manjiao Liu; Na Wang; Qiuping Wu; Qinxin Du; Shiwei Li; Wen Sun; Yifeng Gong; Yonglin Chen; Yuling Zhao; Yuxuan Lin; Ziqi Ren; Zixuan Wang; Aihu Zhang; Brian Li; Buyun Ma; Kang An; Li Xie; Mingliang Li; Pan Li; Shidong Yang; Xi Chen; Xiaojia Liu; Yuchu Luo; Yuan Song; YuanHao Ding; Yuanwei Liang; Zexi Li; Zhaoning Zhang; Zixin Zhang; Binxing Jiao; Daxin Jiang; Jiansheng Chen; Jing Li; Xiangyu Zhang; Yibo Zhu
>
> **摘要:** As LLMs shift toward autonomous agents, Deep Research has emerged as a pivotal metric. However, existing academic benchmarks like BrowseComp often fail to meet real-world demands for open-ended research, which requires robust skills in intent recognition, long-horizon decision-making, and cross-source verification. To address this, we introduce Step-DeepResearch, a cost-effective, end-to-end agent. We propose a Data Synthesis Strategy Based on Atomic Capabilities to reinforce planning and report writing, combined with a progressive training path from agentic mid-training to SFT and RL. Enhanced by a Checklist-style Judger, this approach significantly improves robustness. Furthermore, to bridge the evaluation gap in the Chinese domain, we establish ADR-Bench for realistic deep research scenarios. Experimental results show that Step-DeepResearch (32B) scores 61.4% on Scale AI Research Rubrics. On ADR-Bench, it significantly outperforms comparable models and rivals SOTA closed-source models like OpenAI and Gemini DeepResearch. These findings prove that refined training enables medium-sized models to achieve expert-level capabilities at industry-leading cost-efficiency.
>
---
#### [replaced 027] Analyzing Cognitive Differences Among Large Language Models through the Lens of Social Worldview
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文属于AI社会认知研究任务，旨在分析大语言模型的隐性社会世界观。通过构建SWT框架，识别并量化模型在权威、平等、自主和宿命方面的认知差异，揭示其在社会线索下的适应性。**

- **链接: [https://arxiv.org/pdf/2505.01967v2](https://arxiv.org/pdf/2505.01967v2)**

> **作者:** Jiatao Li; Yanheng Li; Xiaojun Wan
>
> **摘要:** Large Language Models significantly influence social interactions, decision-making, and information dissemination, underscoring the need to understand the implicit socio-cognitive attitudes, referred to as "worldviews", encoded within these systems. Unlike previous studies predominantly addressing demographic and ethical biases as fixed attributes, our study explores deeper cognitive orientations toward authority, equality, autonomy, and fate, emphasizing their adaptability in dynamic social contexts. We introduce the Social Worldview Taxonomy (SWT), an evaluation framework grounded in Cultural Theory, operationalizing four canonical worldviews, namely Hierarchy, Egalitarianism, Individualism, and Fatalism, into quantifiable sub-dimensions. Through extensive analysis of 28 diverse LLMs, we identify distinct cognitive profiles reflecting intrinsic model-specific socio-cognitive structures. Leveraging principles from Social Referencing Theory, our experiments demonstrate that explicit social cues systematically modulate these profiles, revealing robust patterns of cognitive adaptability. Our findings provide insights into the latent cognitive flexibility of LLMs and offer computational scientists practical pathways toward developing more transparent, interpretable, and socially responsible AI systems
>
---
#### [replaced 028] Breadcrumbs Reasoning: Memory-Efficient Reasoning with Compression Beacons
- **分类: cs.CL**

- **简介: 该论文属于大模型推理任务，解决长文本推理中内存消耗过高的问题。通过周期性压缩KV缓存，提升内存效率。**

- **链接: [https://arxiv.org/pdf/2510.13797v3](https://arxiv.org/pdf/2510.13797v3)**

> **作者:** Giovanni Monea; Yair Feldman; Shankar Padmanabhan; Kianté Brantley; Yoav Artzi
>
> **摘要:** The scalability of large language models for long-context reasoning is severely constrained by the linear growth of their Transformer key-value cache, which incurs significant memory and computational costs. We posit that as a model generates reasoning tokens, the informational value of past generated tokens diminishes, creating an opportunity for compression. In this work, we propose to periodically compress the generation KV cache with a learned, special-purpose token and evict compressed entries. We train the model to perform this compression via a modified joint distillation and reinforcement learning (RL) framework. Our training method minimizes overhead over the conventional RL process, as it leverages RL outputs for distillation. Empirically, our method achieves a superior memory-accuracy Pareto frontier compared to both the model without cache compression and training-free compression techniques.
>
---
#### [replaced 029] Understanding Syllogistic Reasoning in LLMs from Formal and Natural Language Perspectives
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型的三段论推理能力，探讨其逻辑与自然语言处理方面的表现，旨在理解模型是否趋向形式化推理。**

- **链接: [https://arxiv.org/pdf/2512.12620v3](https://arxiv.org/pdf/2512.12620v3)**

> **作者:** Aheli Poddar; Saptarshi Sahoo; Sujata Ghosh
>
> **备注:** 9 pages, 4 figures, 5 tables. Accepted at AAAI 2026 Bridge Program on Logic & AI. Code available at https://github.com/XAheli/Logic-in-LLMs
>
> **摘要:** We study syllogistic reasoning in LLMs from the logical and natural language perspectives. In process, we explore fundamental reasoning capabilities of the LLMs and the direction this research is moving forward. To aid in our studies, we use 14 large language models and investigate their syllogistic reasoning capabilities in terms of symbolic inferences as well as natural language understanding. Even though this reasoning mechanism is not a uniform emergent property across LLMs, the perfect symbolic performances in certain models make us wonder whether LLMs are becoming more and more formal reasoning mechanisms, rather than making explicit the nuances of human reasoning.
>
---
#### [replaced 030] Verifiable Fine-Tuning for LLMs: Zero-Knowledge Training Proofs Bound to Data Provenance and Policy
- **分类: cs.CR; cs.CL**

- **简介: 该论文提出一种可验证的微调方法，解决模型训练数据和更新过程不可信的问题，通过零知识证明确保训练过程透明可审计。**

- **链接: [https://arxiv.org/pdf/2510.16830v3](https://arxiv.org/pdf/2510.16830v3)**

> **作者:** Hasan Akgul; Daniel Borg; Arta Berisha; Amina Rahimova; Andrej Novak; Mila Petrov
>
> **备注:** 20 pages, 10 figures
>
> **摘要:** Large language models are often adapted through parameter efficient fine tuning, but current release practices provide weak assurances about what data were used and how updates were computed. We present Verifiable Fine Tuning, a protocol and system that produces succinct zero knowledge proofs that a released model was obtained from a public initialization under a declared training program and an auditable dataset commitment. The approach combines five elements. First, commitments that bind data sources, preprocessing, licenses, and per epoch quota counters to a manifest. Second, a verifiable sampler that supports public replayable and private index hiding batch selection. Third, update circuits restricted to parameter efficient fine tuning that enforce AdamW style optimizer semantics and proof friendly approximations with explicit error budgets. Fourth, recursive aggregation that folds per step proofs into per epoch and end to end certificates with millisecond verification. Fifth, provenance binding and optional trusted execution property cards that attest code identity and constants. On English and bilingual instruction mixtures, the method maintains utility within tight budgets while achieving practical proof performance. Policy quotas are enforced with zero violations, and private sampling windows show no measurable index leakage. Federated experiments demonstrate that the system composes with probabilistic audits and bandwidth constraints. These results indicate that end to end verifiable fine tuning is feasible today for real parameter efficient pipelines, closing a critical trust gap for regulated and decentralized deployments.
>
---
#### [replaced 031] SelfCheck-Eval: A Multi-Module Framework for Zero-Resource Hallucination Detection in Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于 hallucination 检测任务，旨在解决 LLM 在数学推理中生成错误内容的问题。提出 SelfCheck-Eval 框架和 AIME 数学数据集以提升检测效果。**

- **链接: [https://arxiv.org/pdf/2502.01812v2](https://arxiv.org/pdf/2502.01812v2)**

> **作者:** Diyana Muhammed; Giusy Giulia Tuccari; Gollam Rabby; Sören Auer; Sahar Vahdati
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse applications, from open-domain question answering to scientific writing, medical decision support, and legal analysis. However, their tendency to generate incorrect or fabricated content, commonly known as hallucinations, represents a critical barrier to reliable deployment in high-stakes domains. Current hallucination detection benchmarks are limited in scope, focusing primarily on general-knowledge domains while neglecting specialised fields where accuracy is paramount. To address this gap, we introduce the AIME Math Hallucination dataset, the first comprehensive benchmark specifically designed for evaluating mathematical reasoning hallucinations. Additionally, we propose SelfCheck-Eval, a LLM-agnostic, black-box hallucination detection framework applicable to both open and closed-source LLMs. Our approach leverages a novel multi-module architecture that integrates three independent detection strategies: the Semantic module, the Specialised Detection module, and the Contextual Consistency module. Our evaluation reveals systematic performance disparities across domains: existing methods perform well on biographical content but struggle significantly with mathematical reasoning, a challenge that persists across NLI fine-tuning, preference learning, and process supervision approaches. These findings highlight the fundamental limitations of current detection methods in mathematical domains and underscore the critical need for specialised, black-box compatible approaches to ensure reliable LLM deployment.
>
---
#### [replaced 032] From Context to EDUs: Faithful and Structured Context Compression via Elementary Discourse Unit Decomposition
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的文本压缩任务，旨在解决长文本处理中计算成本高和信息噪声问题。通过EDU分解与结构选择，实现高效且结构保持的压缩方法。**

- **链接: [https://arxiv.org/pdf/2512.14244v3](https://arxiv.org/pdf/2512.14244v3)**

> **作者:** Yiqing Zhou; Yu Lei; Shuzheng Si; Qingyan Sun; Wei Wang; Yifei Wu; Hao Wen; Gang Chen; Fanchao Qi; Maosong Sun
>
> **摘要:** Managing extensive context remains a critical bottleneck for Large Language Models (LLMs), particularly in applications like long-document question answering and autonomous agents where lengthy inputs incur high computational costs and introduce noise. Existing compression techniques often disrupt local coherence through discrete token removal or rely on implicit latent encoding that suffers from positional bias and incompatibility with closed-source APIs. To address these limitations, we introduce the EDU-based Context Compressor, a novel explicit compression framework designed to preserve both global structure and fine-grained details. Our approach reformulates context compression as a structure-then-select process. First, our LingoEDU transforms linear text into a structural relation tree of Elementary Discourse Units (EDUs) which are anchored strictly to source indices to eliminate hallucination. Second, a lightweight ranking module selects query-relevant sub-trees for linearization. To rigorously evaluate structural understanding, we release StructBench, a manually annotated dataset of 248 diverse documents. Empirical results demonstrate that our method achieves state-of-the-art structural prediction accuracy and significantly outperforms frontier LLMs while reducing costs. Furthermore, our structure-aware compression substantially enhances performance across downstream tasks ranging from long-context tasks to complex Deep Search scenarios.
>
---
#### [replaced 033] Steering Language Model to Stable Speech Emotion Recognition via Contextual Perception and Chain of Thought
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于语音情感识别任务，旨在解决大模型在SER中出现的幻觉和不稳定性问题。提出C$^2$SER方法，结合上下文感知与思维链，提升识别准确性和稳定性。**

- **链接: [https://arxiv.org/pdf/2502.18186v3](https://arxiv.org/pdf/2502.18186v3)**

> **作者:** Zhixian Zhao; Xinfa Zhu; Xinsheng Wang; Shuiyuan Wang; Xuelong Geng; Wenjie Tian; Lei Xie
>
> **备注:** This work has been published in IEEE Transactions on Audio, Speech and Language Processing
>
> **摘要:** Large-scale audio language models (ALMs), such as Qwen2-Audio, are capable of comprehending diverse audio signal, performing audio analysis and generating textual responses. However, in speech emotion recognition (SER), ALMs often suffer from hallucinations, resulting in misclassifications or irrelevant outputs. To address these challenges, we propose C$^2$SER, a novel ALM designed to enhance the stability and accuracy of SER through Contextual perception and Chain of Thought (CoT). C$^2$SER integrates the Whisper encoder for semantic perception and Emotion2Vec-S for acoustic perception, where Emotion2Vec-S extends Emotion2Vec with semi-supervised learning to enhance emotional discrimination. Additionally, C$^2$SER employs a CoT approach, processing SER in a step-by-step manner while leveraging speech content and speaking styles to improve recognition. To further enhance stability, C$^2$SER introduces self-distillation from explicit CoT to implicit CoT, mitigating error accumulation and boosting recognition accuracy. Extensive experiments show that C$^2$SER outperforms existing popular ALMs, such as Qwen2-Audio and SECap, delivering more stable and precise emotion recognition. We release the training code, checkpoints, and test sets to facilitate further research.
>
---
#### [replaced 034] The Heap: A Contamination-Free Multilingual Code Dataset for Evaluating Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于代码数据集构建任务，旨在解决大语言模型评估中的数据污染问题。工作是发布The Heap，一个去重的多语言代码数据集，支持公平评估。**

- **链接: [https://arxiv.org/pdf/2501.09653v2](https://arxiv.org/pdf/2501.09653v2)**

> **作者:** Jonathan Katzy; Razvan Mihai Popescu; Arie van Deursen; Maliheh Izadi
>
> **备注:** Camera-ready. Accepted to FORGE 2025 Dataset Track
>
> **摘要:** The recent rise in the popularity of large language models has spurred the development of extensive code datasets needed to train them. This has left limited code available for collection and use in the downstream investigation of specific behaviors, or evaluation of large language models without suffering from data contamination. To address this problem, we release The Heap, a large multilingual dataset covering 57 programming languages that has been deduplicated with respect to other open datasets of code, enabling researchers to conduct fair evaluations of large language models without significant data cleaning overhead.
>
---
#### [replaced 035] MME-CC: A Challenging Multi-Modal Evaluation Benchmark of Cognitive Capacity
- **分类: cs.CL**

- **简介: 该论文提出MME-CC基准，用于评估多模态大模型的认知能力，解决现有基准在视觉推理方面不足的问题。通过11个任务测试空间、几何和知识推理，分析模型表现。**

- **链接: [https://arxiv.org/pdf/2511.03146v2](https://arxiv.org/pdf/2511.03146v2)**

> **作者:** Kaiyuan Zhang; Chenghao Yang; Zhoufutu Wen; Sihang Yuan; Qiuyue Wang; Chaoyi Huang; Guosheng Zhu; He Wang; Huawenyu Lu; Jianing Wen; Jianpeng Jiao; Lishu Luo; Longxiang Liu; Sijin Wu; Xiaolei Zhu; Xuanliang Zhang; Yu Liu; Ge Zhang; Yi Lin; Guang Shi; Chaoyou Fu; Wenhao Huang
>
> **摘要:** As reasoning models scale rapidly, the essential role of multimodality in human cognition has come into sharp relief, driving a growing need to probe vision-centric cognitive behaviors. Yet, existing multimodal benchmarks either overemphasize textual reasoning or fall short of systematically capturing vision-centric cognitive behaviors, leaving the cognitive capacity of MLLMs insufficiently assessed. To address this limitation, we introduce MME-CC (Multi-Modal Evaluation benchmark of Cognitive Capacity), a vision-grounded benchmark that organizes 11 representative reasoning tasks into three fundamental categories of visual information: spatial, geometric, and knowledge-based reasoning, and provides fine-grained analyses of MLLMs' cognitive capacity across these dimensions. Based on MME-CC, we conduct extensive experiments over 16 representative MLLMs. Our study reveals that closed-source models currently lead overall (e.g., 42.66 for Gemini-2.5-Pro vs. 30.45 for GLM-4.5V), while spatial and geometric reasoning remain broadly weak (less than or equal to 30%). We further identify common error patterns, including orientation mistakes, fragile cross-view identity persistence, and poor adherence to counterfactual instructions, and observe that Chain-of-Thought typically follows a three-stage process (extract -> reason -> verify) with heavy reliance on visual extraction. We hope this work catalyzes a shift toward treating the cognitive capacity of MLLMs as central to both evaluation and model design.
>
---
#### [replaced 036] Prompt-R1: Collaborative Automatic Prompting Framework via End-to-end Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文提出Prompt-R1，一个基于强化学习的协作自动提示框架，解决用户难以生成有效提示的问题。通过小模型与大模型协作，提升复杂任务处理效果。**

- **链接: [https://arxiv.org/pdf/2511.01016v5](https://arxiv.org/pdf/2511.01016v5)**

> **作者:** Wenjin Liu; Haoran Luo; Xueyuan Lin; Haoming Liu; Tiesunlong Shen; Jiapu Wang; Rui Mao; Erik Cambria
>
> **摘要:** Recently, advanced large language models (LLMs) have emerged at an increasingly rapid pace. However, when faced with complex problems, most users are often unable to provide accurate and effective prompts to interact with LLMs, thus limiting the performance of LLMs. To address this challenge, we propose Prompt-R1, an end-to-end reinforcement learning framework that uses a small-scale LLM to collaborate with large-scale LLMs, replacing user interaction to solve problems better. This collaboration is cast as a multi-turn prompt interaction, where the small-scale LLM thinks and generates prompts, and the large-scale LLM performs complex reasoning. A dual-constrained reward is designed to optimize for correctness, generation quality, and reasoning accuracy. Prompt-R1 provides a plug-and-play framework that supports both inference and training with various large-scale LLMs. Experiments on multiple public datasets show that Prompt-R1 significantly outperforms baseline models across tasks. Our code is publicly available at https://github.com/QwenQKing/Prompt-R1.
>
---
#### [replaced 037] The Gray Zone of Faithfulness: Taming Ambiguity in Unfaithfulness Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于摘要忠实性检测任务，旨在解决标注模糊问题。提出新框架VeriGray，引入中间类别提升标注准确性。**

- **链接: [https://arxiv.org/pdf/2510.21118v4](https://arxiv.org/pdf/2510.21118v4)**

> **作者:** Qiang Ding; Lvzhou Luo; Yixuan Cao; Ping Luo
>
> **备注:** Update the evaluation results due to the annotation updates; revise the citation to Seo et al., 2025; add the acknowledgements
>
> **摘要:** Ensuring that Large Language Models (LLMs) generate summaries faithful to a given source document is essential for real-world applications. While prior research has explored LLM faithfulness, existing benchmarks suffer from annotation ambiguity, primarily due to the ill-defined boundary of permissible external knowledge in generated outputs. For instance, common sense is often incorporated into responses and labeled as "faithful", yet the acceptable extent of such knowledge remains unspecified, leading to inconsistent annotations. To address this issue, we propose a novel faithfulness annotation framework, which introduces an intermediate category, Out-Dependent, to classify cases where external knowledge is required for verification. Using this framework, we construct VeriGray (Verification with the Gray Zone) -- a new unfaithfulness detection benchmark in summarization. Statistics reveal that even SOTA LLMs, such as GPT-5, exhibit hallucinations ($\sim 6\%$ of sentences) in summarization tasks. Moreover, a substantial proportion ($\sim 9\%$ on average of models) of generated sentences fall into the Out-Dependent category, underscoring the importance of resolving annotation ambiguity in unfaithfulness detection benchmarks. Experiments demonstrate that our benchmark poses significant challenges to multiple baseline methods, indicating considerable room for future improvement.
>
---
#### [replaced 038] Rakuten Data Release: A Large-Scale and Long-Term Reviews Corpus for Hotel Domain
- **分类: cs.CL**

- **简介: 该论文介绍了一个大规模酒店评论语料库，包含16年数据，用于分析用户评价与数据漂移。属于情感分析与数据演化研究任务，解决长期评论数据的统计与趋势分析问题。**

- **链接: [https://arxiv.org/pdf/2512.15151v3](https://arxiv.org/pdf/2512.15151v3)**

> **作者:** Yuki Nakayama; Koki Hikichi; Yun Ching Liu; Yu Hirate
>
> **备注:** 6 pages
>
> **摘要:** This paper presents a large-scale corpus of Rakuten Travel Reviews. Our collection contains 7.29 million customer reviews for 16 years, ranging from 2009 to 2024. Each record in the dataset contains the review text, its response from an accommodation, an anonymized reviewer ID, review date, accommodation ID, plan ID, plan title, room type, room name, purpose, accompanying group, and user ratings from six aspect categories, as well as an overall score. We present statistical information about our corpus and provide insights into factors driving data drift between 2019 and 2024 using statistical approaches.
>
---
#### [replaced 039] Topic-FlipRAG: Topic-Orientated Adversarial Opinion Manipulation Attacks to Retrieval-Augmented Generation Models
- **分类: cs.CL; cs.CR; cs.IR**

- **简介: 该论文属于LLM安全研究任务，针对RAG模型提出主题导向的对抗性观点操控攻击，旨在揭示其在多视角推理中的脆弱性，并提出有效攻击方法。**

- **链接: [https://arxiv.org/pdf/2502.01386v3](https://arxiv.org/pdf/2502.01386v3)**

> **作者:** Yuyang Gong; Zhuo Chen; Jiawei Liu; Miaokun Chen; Fengchang Yu; Wei Lu; Xiaofeng Wang; Xiaozhong Liu
>
> **备注:** Accepted by USENIX Security 2025
>
> **摘要:** Retrieval-Augmented Generation (RAG) systems based on Large Language Models (LLMs) have become essential for tasks such as question answering and content generation. However, their increasing impact on public opinion and information dissemination has made them a critical focus for security research due to inherent vulnerabilities. Previous studies have predominantly addressed attacks targeting factual or single-query manipulations. In this paper, we address a more practical scenario: topic-oriented adversarial opinion manipulation attacks on RAG models, where LLMs are required to reason and synthesize multiple perspectives, rendering them particularly susceptible to systematic knowledge poisoning. Specifically, we propose Topic-FlipRAG, a two-stage manipulation attack pipeline that strategically crafts adversarial perturbations to influence opinions across related queries. This approach combines traditional adversarial ranking attack techniques and leverages the extensive internal relevant knowledge and reasoning capabilities of LLMs to execute semantic-level perturbations. Experiments show that the proposed attacks effectively shift the opinion of the model's outputs on specific topics, significantly impacting user information perception. Current mitigation methods cannot effectively defend against such attacks, highlighting the necessity for enhanced safeguards for RAG systems, and offering crucial insights for LLM security research.
>
---
#### [replaced 040] Think Parallax: Solving Multi-Hop Problems via Multi-View Knowledge-Graph-Based Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识增强的多跳问答任务，旨在解决大模型 hallucination 和多跳推理不足的问题。提出 ParallaxRAG 框架，通过多视角知识图谱检索提升推理准确性。**

- **链接: [https://arxiv.org/pdf/2510.15552v2](https://arxiv.org/pdf/2510.15552v2)**

> **作者:** Jinliang Liu; Jiale Bai; Shaoning Zeng
>
> **摘要:** Large language models (LLMs) excel at language understanding but often hallucinate and struggle with multi-hop reasoning. Knowledge-graph-based retrieval-augmented generation (KG-RAG) offers grounding, yet most methods rely on flat embeddings and noisy path exploration. We propose ParallaxRAG, a framework that symmetrically decouples queries and graph triples into multi-view spaces, enabling a robust retrieval architecture that explicitly enforces head diversity while constraining weakly related paths. Central to our approach is the observation that different attention heads specialize in semantic relations at distinct reasoning stages, contributing to different hops of the reasoning chain. This specialization allows ParallaxRAG to construct cleaner subgraphs and guide LLMs through grounded, step-wise reasoning. Experiments on WebQSP and CWQ, under our unified, reproducible setup (BGE-M3 + Llama3.1-8B), demonstrate competitive retrieval and QA performance, alongside reduced hallucination and good generalization. Our results highlight multi-view head specialization as a principled direction for knowledge-grounded multi-hop reasoning. Our implementation will be released as soon as the paper is accepted.
>
---
#### [replaced 041] AdvPrefix: An Objective for Nuanced LLM Jailbreaks
- **分类: cs.LG; cs.AI; cs.CL; cs.CR**

- **简介: 该论文属于LLM安全研究任务，旨在解决 jailbreak 攻击中响应不精准的问题。提出 AdvPrefix 方法，通过优化前缀提升攻击效果。**

- **链接: [https://arxiv.org/pdf/2412.10321v2](https://arxiv.org/pdf/2412.10321v2)**

> **作者:** Sicheng Zhu; Brandon Amos; Yuandong Tian; Chuan Guo; Ivan Evtimov
>
> **摘要:** Many jailbreak attacks on large language models (LLMs) rely on a common objective: making the model respond with the prefix ``Sure, here is (harmful request)''. While straightforward, this objective has two limitations: limited control over model behaviors, yielding incomplete or unrealistic jailbroken responses, and a rigid format that hinders optimization. We introduce AdvPrefix, a plug-and-play prefix-forcing objective that selects one or more model-dependent prefixes by combining two criteria: high prefilling attack success rates and low negative log-likelihood. AdvPrefix integrates seamlessly into existing jailbreak attacks to mitigate the previous limitations for free. For example, replacing GCG's default prefixes on Llama-3 improves nuanced attack success rates from 14% to 80%, revealing that current safety alignment fails to generalize to new prefixes. Code and selected prefixes are released at github.com/facebookresearch/jailbreak-objectives.
>
---
#### [replaced 042] Prompt Injection attack against LLM-integrated Applications
- **分类: cs.CR; cs.AI; cs.CL; cs.SE**

- **简介: 该论文属于安全领域，研究LLM集成应用中的提示注入攻击问题。工作包括分析现有攻击限制，提出新攻击方法HouYi，并验证其有效性。**

- **链接: [https://arxiv.org/pdf/2306.05499v3](https://arxiv.org/pdf/2306.05499v3)**

> **作者:** Yi Liu; Gelei Deng; Yuekang Li; Kailong Wang; Zihao Wang; Xiaofeng Wang; Tianwei Zhang; Yepang Liu; Haoyu Wang; Yan Zheng; Leo Yu Zhang; Yang Liu
>
> **摘要:** Large Language Models (LLMs), renowned for their superior proficiency in language comprehension and generation, stimulate a vibrant ecosystem of applications around them. However, their extensive assimilation into various services introduces significant security risks. This study deconstructs the complexities and implications of prompt injection attacks on actual LLM-integrated applications. Initially, we conduct an exploratory analysis on ten commercial applications, highlighting the constraints of current attack strategies in practice. Prompted by these limitations, we subsequently formulate HouYi, a novel black-box prompt injection attack technique, which draws inspiration from traditional web injection attacks. HouYi is compartmentalized into three crucial elements: a seamlessly-incorporated pre-constructed prompt, an injection prompt inducing context partition, and a malicious payload designed to fulfill the attack objectives. Leveraging HouYi, we unveil previously unknown and severe attack outcomes, such as unrestricted arbitrary LLM usage and uncomplicated application prompt theft. We deploy HouYi on 36 actual LLM-integrated applications and discern 31 applications susceptible to prompt injection. 10 vendors have validated our discoveries, including Notion, which has the potential to impact millions of users. Our investigation illuminates both the possible risks of prompt injection attacks and the possible tactics for mitigation.
>
---
#### [replaced 043] Vis-CoT: A Human-in-the-Loop Framework for Interactive Visualization and Intervention in LLM Chain-of-Thought Reasoning
- **分类: cs.CL**

- **简介: 该论文提出Vis-CoT框架，解决LLM推理过程不透明的问题。通过交互式可视化提升用户对推理过程的控制与理解，提高准确性与可信度。属于人机协作推理任务。**

- **链接: [https://arxiv.org/pdf/2509.01412v2](https://arxiv.org/pdf/2509.01412v2)**

> **作者:** Kaviraj Pather; Elena Hadjigeorgiou; Arben Krasniqi; Claire Schmit; Irina Rusu; Marc Pons; Kabir Khan
>
> **备注:** 12 pages, 7 figures
>
> **摘要:** Large language models (LLMs) show strong reasoning via chain-of-thought (CoT) prompting, but the process is opaque, which makes verification, debugging, and control difficult in high-stakes settings. We present Vis-CoT, a human-in-the-loop framework that converts linear CoT text into an interactive reasoning graph. Users can visualize the logical flow, identify flawed steps, and intervene by pruning incorrect paths and grafting new, user-defined premises. This shifts interaction from passive observation to active collaboration, steering models toward more accurate and trustworthy conclusions. Across GSM8K and StrategyQA, Vis-CoT improves final-answer accuracy by up to 24 percentage points over non-interactive baselines. A user study also shows large gains in perceived usability and trust. Vis-CoT points to a practical path for more reliable, understandable, and collaborative reasoning by combining LLMs with targeted human oversight.
>
---
#### [replaced 044] Beyond Context: Large Language Models Failure to Grasp Users Intent
- **分类: cs.AI; cs.CL; cs.CR; cs.CY**

- **简介: 该论文属于AI安全领域，旨在解决LLMs无法准确理解用户意图的问题。研究分析了多个大模型，发现其安全机制易被情感、渐进披露等方法绕过，提出需强化上下文与意图识别作为核心安全能力。**

- **链接: [https://arxiv.org/pdf/2512.21110v2](https://arxiv.org/pdf/2512.21110v2)**

> **作者:** Ahmed M. Hussain; Salahuddin Salahuddin; Panos Papadimitratos
>
> **备注:** 22 pages and 23 figures
>
> **摘要:** Current Large Language Models (LLMs) safety approaches focus on explicitly harmful content while overlooking a critical vulnerability: the inability to understand context and recognize user intent. This creates exploitable vulnerabilities that malicious users can systematically leverage to circumvent safety mechanisms. We empirically evaluate multiple state-of-the-art LLMs, including ChatGPT, Claude, Gemini, and DeepSeek. Our analysis demonstrates the circumvention of reliable safety mechanisms through emotional framing, progressive revelation, and academic justification techniques. Notably, reasoning-enabled configurations amplified rather than mitigated the effectiveness of exploitation, increasing factual precision while failing to interrogate the underlying intent. The exception was Claude Opus 4.1, which prioritized intent detection over information provision in some use cases. This pattern reveals that current architectural designs create systematic vulnerabilities. These limitations require paradigmatic shifts toward contextual understanding and intent recognition as core safety capabilities rather than post-hoc protective mechanisms.
>
---
#### [replaced 045] Fun-Audio-Chat Technical Report
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文提出Fun-Audio-Chat，解决语音与文本模型间的时间分辨率不匹配及知识遗忘问题，通过双分辨率语音表示和核心鸡尾酒训练提升音频理解与生成能力。**

- **链接: [https://arxiv.org/pdf/2512.20156v2](https://arxiv.org/pdf/2512.20156v2)**

> **作者:** Tongyi Fun Team; Qian Chen; Luyao Cheng; Chong Deng; Xiangang Li; Jiaqing Liu; Chao-Hong Tan; Wen Wang; Junhao Xu; Jieping Ye; Qinglin Zhang; Qiquan Zhang; Jingren Zhou
>
> **备注:** Authors are listed in alphabetical order, 21 pages, https://github.com/FunAudioLLM/Fun-Audio-Chat
>
> **摘要:** Recent advancements in joint speech-text models show great potential for seamless voice interactions. However, existing models face critical challenges: temporal resolution mismatch between speech tokens (25Hz) and text tokens (~3Hz) dilutes semantic information, incurs high computational costs, and causes catastrophic forgetting of text LLM knowledge. We introduce Fun-Audio-Chat, a Large Audio Language Model addressing these limitations via two innovations from our previous work DrVoice. First, Dual-Resolution Speech Representations (DRSR): the Shared LLM processes audio at efficient 5Hz (via token grouping), while the Speech Refined Head generates high-quality tokens at 25Hz, balancing efficiency (~50% GPU reduction) and quality. Second, Core-Cocktail Training, a two-stage fine-tuning with intermediate merging that mitigates catastrophic forgetting. We then apply Multi-Task DPO Training to enhance robustness, audio understanding, instruction-following and voice empathy. This multi-stage post-training enables Fun-Audio-Chat to retain text LLM knowledge while gaining powerful audio understanding, reasoning, and generation. Unlike recent LALMs requiring large-scale audio-text pre-training, Fun-Audio-Chat leverages pre-trained models and extensive post-training. Fun-Audio-Chat 8B and MoE 30B-A3B achieve competitive performance on Speech-to-Text and Speech-to-Speech tasks, ranking top among similar-scale models on Spoken QA benchmarks. They also achieve competitive to superior performance on Audio Understanding, Speech Function Calling, Instruction-Following and Voice Empathy. We develop Fun-Audio-Chat-Duplex, a full-duplex variant with strong performance on Spoken QA and full-duplex interactions. We open-source Fun-Audio-Chat-8B with training and inference code, and provide an interactive demo, at https://github.com/FunAudioLLM/Fun-Audio-Chat .
>
---
#### [replaced 046] RAVEL: Rare Concept Generation and Editing via Graph-driven Relational Guidance
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出RAVEL框架，解决文本到图像生成中罕见概念表达不足的问题。通过知识图谱增强生成过程，提升生成质量与准确性。**

- **链接: [https://arxiv.org/pdf/2412.09614v2](https://arxiv.org/pdf/2412.09614v2)**

> **作者:** Kavana Venkatesh; Yusuf Dalva; Ismini Lourentzou; Pinar Yanardag
>
> **备注:** Project Page: https://ravel-diffusion.github.io/
>
> **摘要:** Despite impressive visual fidelity, current text-to-image (T2I) diffusion models struggle to depict rare, complex, or culturally nuanced concepts due to training data limitations. We introduce RAVEL, a training-free framework that significantly improves rare concept generation, context-driven image editing, and self-correction by integrating graph-based retrieval-augmented generation (RAG) into diffusion pipelines. Unlike prior RAG and LLM-enhanced methods reliant on visual exemplars, static captions or pre-trained knowledge of models, RAVEL leverages structured knowledge graphs to retrieve compositional, symbolic, and relational context, enabling nuanced grounding even in the absence of visual priors. To further refine generation quality, we propose SRD, a novel self-correction module that iteratively updates prompts via multi-aspect alignment feedback, enhancing attribute accuracy, narrative coherence, and semantic fidelity. Our framework is model-agnostic and compatible with leading diffusion models including Stable Diffusion XL, Flux, and DALL-E 3. We conduct extensive evaluations across three newly proposed benchmarks - MythoBench, Rare-Concept-1K, and NovelBench. RAVEL also consistently outperforms SOTA methods across perceptual, alignment, and LLM-as-a-Judge metrics. These results position RAVEL as a robust paradigm for controllable and interpretable T2I generation in long-tail domains.
>
---
#### [replaced 047] Atom of Thoughts for Markov LLM Test-Time Scaling
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于大语言模型优化任务，旨在解决测试时扩展中的冗余计算问题。通过引入马尔可夫推理过程，提出Atom of Thoughts结构，提升推理效率与性能。**

- **链接: [https://arxiv.org/pdf/2502.12018v4](https://arxiv.org/pdf/2502.12018v4)**

> **作者:** Fengwei Teng; Quan Shi; Zhaoyang Yu; Jiayi Zhang; Yuyu Luo; Chenglin Wu; Zhijiang Guo
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Large Language Models (LLMs) have achieved significant performance gains through test-time scaling methods. However, existing approaches often incur redundant computations due to the accumulation of historical dependency information during inference. To address this challenge, we leverage the memoryless property of Markov processes to minimize reliance on historical context and propose a Markovian reasoning process. This foundational Markov chain structure enables seamless integration with various test-time scaling methods, thereby improving their scaling efficiency. By further scaling up the Markovian reasoning chain through integration with techniques such as tree search and reflective refinement, we uncover an emergent atomic reasoning structure, where reasoning trajectories are decomposed into a series of self-contained, low-complexity atomic units. We name this design Atom of Thoughts (\our). Extensive experiments demonstrate that \our consistently outperforms existing baselines as computational budgets increase. Importantly, \our integrates seamlessly with existing reasoning frameworks and different LLMs (both reasoning and non-reasoning), facilitating scalable, high-performance inference.We submit our code alongside this paper and will make it publicly available to facilitate reproducibility and future research.
>
---
#### [replaced 048] A Large Language Model Based Pipeline for Review of Systems Entity Recognition from Clinical Notes
- **分类: cs.CL**

- **简介: 该论文属于临床文本中的实体识别任务，旨在自动提取病历中的系统回顾（ROS）实体。工作包括构建基于大语言模型的处理管道，并引入新颖的归因算法提升识别准确率。**

- **链接: [https://arxiv.org/pdf/2506.11067v2](https://arxiv.org/pdf/2506.11067v2)**

> **作者:** Hieu Nghiem; Zhuqi Miao; Hemanth Reddy Singareddy; Jivan Lamichhane; Abdulaziz Ahmed; Johnson Thomas; Dursun Delen; William Paiva
>
> **摘要:** Objective: Develop a cost-effective, large language model (LLM)-based pipeline for automatically extracting Review of Systems (ROS) entities from clinical notes. Materials and Methods: The pipeline extracts ROS section from the clinical note using SecTag header terminology, followed by few-shot LLMs to identify ROS entities such as diseases or symptoms, their positive/negative status and associated body systems. We implemented the pipeline using 4 open-source LLM models: llama3.1:8b, gemma3:27b, mistral3.1:24b and gpt-oss:20b. Additionally, we introduced a novel attribution algorithm that aligns LLM-identified ROS entities with their source text, addressing non-exact and synonymous matches. The evaluation was conducted on 24 general medicine notes containing 340 annotated ROS entities. Results: Open-source LLMs enable a local, cost-efficient pipeline while delivering promising performance. Larger models like Gemma, Mistral, and Gpt-oss demonstrate robust performance across three entity recognition tasks of the pipeline: ROS entity extraction, negation detection and body system classification (highest F1 score = 0.952). With the attribution algorithm, all models show improvements across key performance metrics, including higher F1 score and accuracy, along with lower error rate. Notably, the smaller Llama model also achieved promising results despite using only one-third the VRAM of larger models. Discussion and Conclusion: From an application perspective, our pipeline provides a scalable, locally deployable solution to easing the ROS documentation burden. Open-source LLMs offer a practical AI option for resource-limited healthcare settings. Methodologically, our newly developed algorithm facilitates accuracy improvements for zero- and few-shot LLMs in named entity recognition.
>
---
#### [replaced 049] Patience Is The Key to Large Language Model Reasoning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在复杂问题求解中推理深度不足的问题。通过鼓励模型采用更耐心的推理方式，提升其表现。**

- **链接: [https://arxiv.org/pdf/2411.13082v4](https://arxiv.org/pdf/2411.13082v4)**

> **作者:** Yijiong Yu
>
> **备注:** The paper is not solid enough because the evaluation data is too less and the improvement is not significant
>
> **摘要:** Recent advancements in the field of large language models, particularly through the Chain of Thought (CoT) approach, have demonstrated significant improvements in solving complex problems. However, existing models either tend to sacrifice detailed reasoning for brevity due to user preferences, or require extensive and expensive training data to learn complicated reasoning ability, limiting their potential in solving complex tasks. To bridge this gap, following the concept of scaling test-time, we propose a simple method by encouraging models to adopt a more patient reasoning style without the need of introducing new knowledge or skills. To employ a preference optimization approach, we generate detailed reasoning processes as positive examples and simple answers as negative examples, thereby training the model to favor thoroughness in its responses. Our results demonstrate a performance increase of up to 2.1% on GSM8k with training just on a lightweight dataset.
>
---
#### [replaced 050] Attention Is All You Need for KV Cache in Diffusion LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对扩散大语言模型的KV缓存优化问题，提出Elastic-Cache策略，通过自适应刷新减少冗余计算，提升解码速度与效率。**

- **链接: [https://arxiv.org/pdf/2510.14973v2](https://arxiv.org/pdf/2510.14973v2)**

> **作者:** Quan Nguyen-Tri; Mukul Ranjan; Zhiqiang Shen
>
> **备注:** Code at: https://github.com/VILA-Lab/Elastic-Cache
>
> **摘要:** This work studies how to adaptively recompute key-value (KV) caches for diffusion large language models (DLMs) to maximize prediction accuracy while minimizing decoding latency. Prior methods' decoders recompute QKV for all tokens at every denoising step and layer, despite KV states changing little across most steps, especially in shallow layers, leading to substantial redundancy. We make three observations: (1) distant ${\bf MASK}$ tokens primarily act as a length-bias and can be cached block-wise beyond the active prediction window; (2) KV dynamics increase with depth, suggesting that selective refresh starting from deeper layers is sufficient; and (3) the most-attended token exhibits the smallest KV drift, providing a conservative lower bound on cache change for other tokens. Building on these, we propose ${\bf Elastic-Cache}$, a training-free, architecture-agnostic strategy that jointly decides ${when}$ to refresh (via an attention-aware drift test on the most-attended token) and ${where}$ to refresh (via a depth-aware schedule that recomputes from a chosen layer onward while reusing shallow-layer caches and off-window MASK caches). Unlike fixed-period schemes, Elastic-Cache performs adaptive, layer-aware cache updates for diffusion LLMs, reducing redundant computation and accelerating decoding with negligible loss in generation quality. Experiments on LLaDA-Instruct, LLaDA-1.5, and LLaDA-V across mathematical reasoning and code generation tasks demonstrate consistent speedups: $8.7\times$ on GSM8K (256 tokens), and $45.1\times$ on longer sequences, while consistently maintaining higher accuracy than the baseline. Our method achieves significantly higher throughput ($6.8\times$ on GSM8K) than existing confidence-based approaches while preserving generation quality, enabling practical deployment of diffusion LLMs.
>
---
#### [replaced 051] Authors Should Label Their Own Documents
- **分类: cs.CL**

- **简介: 论文提出作者标注方法，用于提升文本情感和信念标注质量。针对第三方标注的局限性，通过作者实时标注数据，优化推荐系统性能，实验显示效果显著优于传统方法。**

- **链接: [https://arxiv.org/pdf/2512.12976v2](https://arxiv.org/pdf/2512.12976v2)**

> **作者:** Marcus Ma; Cole Johnson; Nolan Bridges; Jackson Trager; Georgios Chochlakis; Shrikanth Narayanan
>
> **摘要:** Third-party annotation is the status quo for labeling text, but egocentric information such as sentiment and belief can at best only be approximated by a third-person proxy. We introduce author labeling, an annotation technique where the writer of the document itself annotates the data at the moment of creation. We collaborate with a commercial chatbot with over 20,000 users to deploy an author labeling annotation system. This system identifies task-relevant queries, generates on-the-fly labeling questions, and records authors' answers in real time. We train and deploy an online-learning model architecture for product recommendation with author-labeled data to improve performance. We train our model to minimize the prediction error on questions generated for a set of predetermined subjective beliefs using author-labeled responses. Our model achieves a 537% improvement in click-through rate compared to an industry advertising baseline running concurrently. We then compare the quality and practicality of author labeling to three traditional annotation approaches for sentiment analysis and find author labeling to be higher quality, faster to acquire, and cheaper. These findings reinforce existing literature that annotations, especially for egocentric and subjective beliefs, are significantly higher quality when labeled by the author rather than a third party. To facilitate broader scientific adoption, we release an author labeling service for the research community at https://academic.echollm.io.
>
---
#### [replaced 052] TokenTiming: A Dynamic Alignment Method for Universal Speculative Decoding Model Pairs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型加速任务，解决草案模型与目标模型词汇不匹配的问题。提出TokenTiming方法，通过DTW实现动态对齐，提升推理效率。**

- **链接: [https://arxiv.org/pdf/2510.15545v3](https://arxiv.org/pdf/2510.15545v3)**

> **作者:** Sibo Xiao; Jinyuan Fu; Zhongle Xie; Lidan Shou
>
> **摘要:** Accelerating the inference of large language models (LLMs) has been a critical challenge in generative AI. Speculative decoding (SD) substantially improves LLM inference efficiency. However, its utility is limited by a fundamental constraint: the draft and target models must share the same vocabulary, thus limiting the herd of available draft models and often necessitating the training of a new model from scratch. Inspired by Dynamic Time Warping (DTW), a classic algorithm for aligning time series, we propose the algorithm TokenTiming for universal speculative decoding. It operates by re-encoding the draft token sequence to get a new target token sequence, and then uses DTW to build a mapping to transfer the probability distributions for speculative sampling. Benefiting from this, our method accommodates mismatched vocabularies and works with any off-the-shelf models without retraining and modification. We conduct comprehensive experiments on various tasks, demonstrating 1.57x speedup. This work enables a universal approach for draft model selection, making SD a more versatile and practical tool for LLM acceleration.
>
---
#### [replaced 053] Learning the Topic, Not the Language: How LLMs Classify Online Immigration Discourse Across Languages
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言文本分类任务，旨在解决LLMs在跨语言移民话语分析中的语言偏见与成本问题。通过微调轻量级模型，实现高效、低成本的多语言分类。**

- **链接: [https://arxiv.org/pdf/2508.06435v2](https://arxiv.org/pdf/2508.06435v2)**

> **作者:** Andrea Nasuto; Stefano Maria Iacus; Francisco Rowe; Devika Jain
>
> **摘要:** Large language models (LLMs) offer new opportunities for scalable analysis of online discourse. Yet their use in multilingual social science research remains constrained by model size, cost and linguistic bias. We develop a lightweight, open-source LLM framework using fine-tuned LLaMA 3.2-3B models to classify immigration-related tweets across 13 languages. Unlike prior work relying on BERT style models or translation pipelines, we combine topic classification with stance detection and demonstrate that LLMs fine-tuned in just one or two languages can generalize topic understanding to unseen languages. Capturing ideological nuance, however, benefits from multilingual fine-tuning. Our approach corrects pretraining biases with minimal data from under-represented languages and avoids reliance on proprietary systems. With 26-168x faster inference and over 1000x cost savings compared to commercial LLMs, our method supports real-time analysis of billions of tweets. This scale-first framework enables inclusive, reproducible research on public attitudes across linguistic and cultural contexts.
>
---
#### [replaced 054] LLMEval-Fair: A Large-Scale Longitudinal Study on Robust and Fair Evaluation of Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于大语言模型评估任务，旨在解决静态基准测试中的数据污染和排行榜过拟合问题。提出LLMEval-Fair框架，通过动态测试集和抗作弊机制实现更公平可靠的模型评估。**

- **链接: [https://arxiv.org/pdf/2508.05452v5](https://arxiv.org/pdf/2508.05452v5)**

> **作者:** Ming Zhang; Yujiong Shen; Jingyi Deng; Yuhui Wang; Huayu Sha; Kexin Tan; Qiyuan Peng; Yue Zhang; Junzhe Wang; Shichun Liu; Yueyuan Huang; Jingqi Tong; Changhao Jiang; Yilong Wu; Zhihao Zhang; Mingqi Wu; Mingxu Chai; Zhiheng Xi; Shihan Dou; Tao Gui; Qi Zhang; Xuanjing Huang
>
> **摘要:** Existing evaluation of Large Language Models (LLMs) on static benchmarks is vulnerable to data contamination and leaderboard overfitting, critical issues that obscure true model capabilities. To address this, we introduce LLMEval-Fair, a framework for dynamic evaluation of LLMs. LLMEval-Fair is built on a proprietary bank of 220k graduate-level questions, from which it dynamically samples unseen test sets for each evaluation run. Its automated pipeline ensures integrity via contamination-resistant data curation, a novel anti-cheating architecture, and a calibrated LLM-as-a-judge process achieving 90% agreement with human experts, complemented by a relative ranking system for fair comparison. A 30-month longitudinal study of nearly 60 leading models reveals a performance ceiling on knowledge memorization and exposes data contamination vulnerabilities undetectable by static benchmarks. The framework demonstrates exceptional robustness in ranking stability and consistency, providing strong empirical validation for the dynamic evaluation paradigm. LLMEval-Fair offers a robust and credible methodology for assessing the true capabilities of LLMs beyond leaderboard scores, promoting the development of more trustworthy evaluation standards.
>
---
#### [replaced 055] Who Writes What: Unveiling the Impact of Author Roles on AI-generated Text Detection
- **分类: cs.CL**

- **简介: 该论文属于AI文本检测任务，旨在解决现有检测方法忽视作者特征影响的问题。通过分析不同作者属性对检测效果的影响，提出更公平的检测框架。**

- **链接: [https://arxiv.org/pdf/2502.12611v2](https://arxiv.org/pdf/2502.12611v2)**

> **作者:** Jiatao Li; Xiaojun Wan
>
> **备注:** ACL 2025 Main Conference
>
> **摘要:** The rise of Large Language Models (LLMs) necessitates accurate AI-generated text detection. However, current approaches largely overlook the influence of author characteristics. We investigate how sociolinguistic attributes-gender, CEFR proficiency, academic field, and language environment-impact state-of-the-art AI text detectors. Using the ICNALE corpus of human-authored texts and parallel AI-generated texts from diverse LLMs, we conduct a rigorous evaluation employing multi-factor ANOVA and weighted least squares (WLS). Our results reveal significant biases: CEFR proficiency and language environment consistently affected detector accuracy, while gender and academic field showed detector-dependent effects. These findings highlight the crucial need for socially aware AI text detection to avoid unfairly penalizing specific demographic groups. We offer novel empirical evidence, a robust statistical framework, and actionable insights for developing more equitable and reliable detection systems in real-world, out-of-domain contexts. This work paves the way for future research on bias mitigation, inclusive evaluation benchmarks, and socially responsible LLM detectors.
>
---
#### [replaced 056] Gamayun's Path to Multilingual Mastery: Cost-Efficient Training of a 1.5B-Parameter LLM
- **分类: cs.CL**

- **简介: 该论文提出Gamayun，一个1.5B参数的多语言模型，解决资源受限环境下高效训练小规模非英语中心LLM的问题。通过两阶段预训练策略，在多语言和英语任务中均取得优异表现。**

- **链接: [https://arxiv.org/pdf/2512.21580v2](https://arxiv.org/pdf/2512.21580v2)**

> **作者:** Alexander Podolskiy; Semen Molokov; Timofey Gerasin; Maksim Titov; Alexey Rukhovich; Artem Khrapov; Kirill Morozov; Evgeny Tetin; Constantine Korikov; Pavel Efimov; Polina Lazukova; Yuliya Skripkar; Nikita Okhotnikov; Irina Piontkovskaya; Meng Xiaojun; Zou Xueyi; Zhang Zhenhe
>
> **摘要:** We present Gamayun, a 1.5B-parameter multilingual language model trained entirely from scratch on 2.5T tokens. Designed for efficiency and deployment in resource-constrained environments, Gamayun addresses the lack of research on small non-English-centric LLMs by adopting a novel two-stage pre-training strategy: balanced multilingual training for cross-lingual alignment, followed by high-quality English enrichment to transfer performance gains across languages. Our model supports 12 languages, with special focus on Russian. Despite a significantly smaller training budget than comparable models, Gamayun outperforms LLaMA3.2-1B (9T tokens) on all considered benchmarks, and surpasses Qwen2.5-1.5B (18T tokens) on a wide range of English and multilingual tasks. It matches or exceeds Qwen3 (36T tokens) on most tasks outside advanced STEM, achieving state-of-the-art results in Russian, including the MERA benchmark, among the models of comparable size (1-2B parameters).
>
---
#### [replaced 057] The Cultural Gene of Large Language Models: A Study on the Impact of Cross-Corpus Training on Model Values and Biases
- **分类: cs.CL**

- **简介: 该论文研究LLM的文化基因，分析跨语料训练对模型价值观和偏见的影响。任务是评估模型文化倾向，解决模型可能存在的文化偏见问题。工作包括构建CPD、比较东西方模型、计算CAI。**

- **链接: [https://arxiv.org/pdf/2508.12411v4](https://arxiv.org/pdf/2508.12411v4)**

> **作者:** Emanuel Z. Fenech-Borg; Tilen P. Meznaric-Kos; Milica D. Lekovic-Bojovic; Arni J. Hentze-Djurhuus
>
> **备注:** 10 pages, 5 figures, IEEE conference format, submitted to [Conference Name]
>
> **摘要:** Large language models (LLMs) are deployed globally, yet their underlying cultural and ethical assumptions remain underexplored. We propose the notion of a "cultural gene" -- a systematic value orientation that LLMs inherit from their training corpora -- and introduce a Cultural Probe Dataset (CPD) of 200 prompts targeting two classic cross-cultural dimensions: Individualism-Collectivism (IDV) and Power Distance (PDI). Using standardized zero-shot prompts, we compare a Western-centric model (GPT-4) and an Eastern-centric model (ERNIE Bot). Human annotation shows significant and consistent divergence across both dimensions. GPT-4 exhibits individualistic and low-power-distance tendencies (IDV score approx 1.21; PDI score approx -1.05), while ERNIE Bot shows collectivistic and higher-power-distance tendencies (IDV approx -0.89; PDI approx 0.76); differences are statistically significant (p < 0.001). We further compute a Cultural Alignment Index (CAI) against Hofstede's national scores and find GPT-4 aligns more closely with the USA (e.g., IDV CAI approx 0.91; PDI CAI approx 0.88) whereas ERNIE Bot aligns more closely with China (IDV CAI approx 0.85; PDI CAI approx 0.81). Qualitative analyses of dilemma resolution and authority-related judgments illustrate how these orientations surface in reasoning. Our results support the view that LLMs function as statistical mirrors of their cultural corpora and motivate culturally aware evaluation and deployment to avoid algorithmic cultural hegemony.
>
---
#### [replaced 058] DySK-Attn: A Framework for Efficient, Real-Time Knowledge Updating in Large Language Models via Dynamic Sparse Knowledge Attention
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于知识更新任务，旨在解决LLM知识静态、过时的问题。提出DySK-Attn框架，通过动态稀疏注意力机制高效整合实时知识。**

- **链接: [https://arxiv.org/pdf/2508.07185v3](https://arxiv.org/pdf/2508.07185v3)**

> **作者:** Kabir Khan; Priya Sharma; Arjun Mehta; Neha Gupta; Ravi Narayanan
>
> **备注:** Preprint; 7 figures, 3 tables, 1 algorithm; v1. Code and data will be released
>
> **摘要:** Large Language Models (LLMs) suffer from a critical limitation: their knowledge is static and quickly becomes outdated. Retraining these massive models is computationally prohibitive, while existing knowledge editing techniques can be slow and may introduce unforeseen side effects. To address this, we propose DySK-Attn, a novel framework that enables LLMs to efficiently integrate real-time knowledge from a dynamic external source. Our approach synergizes an LLM with a dynamic Knowledge Graph (KG) that can be updated instantaneously. The core of our framework is a sparse knowledge attention mechanism, which allows the LLM to perform a coarse-to-fine grained search, efficiently identifying and focusing on a small, highly relevant subset of facts from the vast KG. This mechanism avoids the high computational cost of dense attention over the entire knowledge base and mitigates noise from irrelevant information. We demonstrate through extensive experiments on time-sensitive question-answering tasks that DySK-Attn significantly outperforms strong baselines, including standard Retrieval-Augmented Generation (RAG) and model editing techniques, in both factual accuracy for updated knowledge and computational efficiency. Our framework offers a scalable and effective solution for building LLMs that can stay current with the ever-changing world.
>
---
#### [replaced 059] Dual LoRA: Enhancing LoRA with Magnitude and Direction Updates
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在提升LoRA的性能。针对LoRA低秩假设导致的性能不足，提出Dual LoRA，通过分离参数更新的幅度与方向来优化微调过程。**

- **链接: [https://arxiv.org/pdf/2512.03402v2](https://arxiv.org/pdf/2512.03402v2)**

> **作者:** Yixing Xu; Chao Li; Xuanwu Yin; Spandan Tiwari; Dong Li; Ashish Sirasao; Emad Barsoum
>
> **摘要:** Low-rank adaptation (LoRA) is one of the most popular methods among parameter-efficient fine-tuning (PEFT) methods to adapt pre-trained large language models (LLMs) to specific downstream tasks. However, the model trained based on LoRA often has an unsatisfactory performance due to its low-rank assumption. In this paper, we propose a novel method called Dual LoRA to improve the performance by incorporating an inductive bias into the original LoRA. Specifically, we separate low-rank matrices into two groups: the magnitude group to control whether or not and how far we should update a parameter and the direction group to decide whether this parameter should move forward or backward, to better simulate the parameter updating process of the full fine-tuning based on gradient-based optimization algorithms. We show that this can be simply achieved by adding a ReLU function to the magnitude group and a sign function to the direction group. We conduct several experiments over a wide range of NLP tasks, including natural language understanding (NLU) and commonsense reasoning datasets on RoBERTa, DeBERTa, and LLaMA-1/2/3 as baseline models. The results show that we consistently outperform LoRA and its state-of-the-art variants with the same number of trainable parameters.
>
---
#### [replaced 060] Do You Feel Comfortable? Detecting Hidden Conversational Escalation in AI Chatbots
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话系统任务，旨在解决AI聊天机器人中隐性情绪升级的检测问题。提出GAUGE框架，实时识别对话中的情感变化。**

- **链接: [https://arxiv.org/pdf/2512.06193v2](https://arxiv.org/pdf/2512.06193v2)**

> **作者:** Jihyung Park; Saleh Afroogh; Junfeng Jiao
>
> **摘要:** Large Language Models (LLM) are increasingly integrated into everyday interactions, serving not only as information assistants but also as emotional companions. Even in the absence of explicit toxicity, repeated emotional reinforcement or affective drift can gradually escalate distress in a form of \textit{implicit harm} that traditional toxicity filters fail to detect. Existing guardrail mechanisms often rely on external classifiers or clinical rubrics that may lag behind the nuanced, real-time dynamics of a developing conversation. To address this gap, we propose GAUGE (Guarding Affective Utterance Generation Escalation), logit-based framework for the real-time detection of hidden conversational escalation. GAUGE measures how an LLM's output probabilistically shifts the affective state of a dialogue.
>
---
#### [replaced 061] No Prompt Left Behind: Exploiting Zero-Variance Prompts in LLM Reinforcement Learning via Entropy-Guided Advantage Shaping
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于强化学习任务，旨在解决LLM推理能力提升问题。针对零方差提示缺乏反馈的问题，提出RL-ZVP算法，有效利用此类提示提升模型性能。**

- **链接: [https://arxiv.org/pdf/2509.21880v2](https://arxiv.org/pdf/2509.21880v2)**

> **作者:** Thanh-Long V. Le; Myeongho Jeon; Kim Vu; Viet Lai; Eunho Yang
>
> **备注:** Under review. Project page: https://bltnynk.github.io/publications/rl-zvp/
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) is a powerful framework for improving the reasoning abilities of Large Language Models (LLMs). However, current methods such as GRPO rely only on problems where the model responses to the same input differ in correctness, while ignoring those where all responses receive the same reward -- so-called zero-variance prompts. In this work, we argue that such prompts are not useless but can, in fact, provide meaningful feedback for policy optimization. To this end, we introduce RL with Zero-Variance Prompts (RL-ZVP), a novel algorithm that extract learning signals from zero-variance prompts. RL-ZVP directly rewards correctness and penalizes errors even without contrasting responses, modulating feedback with token-level characteristics to preserve informative, nuanced signals. Across six math reasoning benchmarks, RL-ZVP achieves significant improvements of up to 8.61 points in accuracy and 7.77 points in pass rate over GRPO, while consistently outperforming other baselines that filter out zero-variance prompts. These results highlight the untapped potential of learning from zero-variance prompts in RLVR.
>
---
#### [replaced 062] Iterative Multilingual Spectral Attribute Erasure
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的去偏任务，旨在解决多语言表示中的联合偏差问题。通过迭代SVD方法消除多语言共同偏差子空间，提升模型公平性。**

- **链接: [https://arxiv.org/pdf/2506.11244v2](https://arxiv.org/pdf/2506.11244v2)**

> **作者:** Shun Shao; Yftah Ziser; Zheng Zhao; Yifu Qiu; Shay B. Cohen; Anna Korhonen
>
> **备注:** Accepted to the main conference of EMNLP 2025
>
> **摘要:** Multilingual representations embed words with similar meanings to share a common semantic space across languages, creating opportunities to transfer debiasing effects between languages. However, existing methods for debiasing are unable to exploit this opportunity because they operate on individual languages. We present Iterative Multilingual Spectral Attribute Erasure (IMSAE), which identifies and mitigates joint bias subspaces across multiple languages through iterative SVD-based truncation. Evaluating IMSAE across eight languages and five demographic dimensions, we demonstrate its effectiveness in both standard and zero-shot settings, where target language data is unavailable, but linguistically similar languages can be used for debiasing. Our comprehensive experiments across diverse language models (BERT, LLaMA, Mistral) show that IMSAE outperforms traditional monolingual and cross-lingual approaches while maintaining model utility.
>
---
