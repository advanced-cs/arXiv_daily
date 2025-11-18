# 自然语言处理 cs.CL

- **最新发布 124 篇**

- **更新 112 篇**

## 最新发布

#### [new 001] PRISM of Opinions: A Persona-Reasoned Multimodal Framework for User-centric Conversational Stance Detection
- **分类: cs.CL**

- **简介: 论文提出PRISM框架解决多模态对话立场检测中伪多模态和用户同质化问题。通过构建用户中心数据集U-MStance，利用用户长期人格特征与对话上下文中的多模态对齐，提升立场理解准确性。**

- **链接: [https://arxiv.org/pdf/2511.12130v1](https://arxiv.org/pdf/2511.12130v1)**

> **作者:** Bingbing Wang; Zhixin Bai; Zhengda Jin; Zihan Wang; Xintong Song; Jingjie Lin; Sixuan Li; Jing Li; Ruifeng Xu
>
> **摘要:** The rapid proliferation of multimodal social media content has driven research in Multimodal Conversational Stance Detection (MCSD), which aims to interpret users' attitudes toward specific targets within complex discussions. However, existing studies remain limited by: **1) pseudo-multimodality**, where visual cues appear only in source posts while comments are treated as text-only, misaligning with real-world multimodal interactions; and **2) user homogeneity**, where diverse users are treated uniformly, neglecting personal traits that shape stance expression. To address these issues, we introduce **U-MStance**, the first user-centric MCSD dataset, containing over 40k annotated comments across six real-world targets. We further propose **PRISM**, a **P**ersona-**R**easoned mult**I**modal **S**tance **M**odel for MCSD. PRISM first derives longitudinal user personas from historical posts and comments to capture individual traits, then aligns textual and visual cues within conversational context via Chain-of-Thought to bridge semantic and pragmatic gaps across modalities. Finally, a mutual task reinforcement mechanism is employed to jointly optimize stance detection and stance-aware response generation for bidirectional knowledge transfer. Experiments on U-MStance demonstrate that PRISM yields significant gains over strong baselines, underscoring the effectiveness of user-centric and context-grounded multimodal reasoning for realistic stance understanding.
>
---
#### [new 002] Quantifying consistency and accuracy of Latent Dirichlet Allocation
- **分类: cs.CL**

- **简介: 论文研究LDA主题模型的稳定性和准确性，解决其随机性导致结果不一致的问题。通过构建带真实标签的合成语料库，量化模型输出的一致性与准确性，发现LDA虽内部稳定但无法捕捉真实主题。**

- **链接: [https://arxiv.org/pdf/2511.12850v1](https://arxiv.org/pdf/2511.12850v1)**

> **作者:** Saranzaya Magsarjav; Melissa Humphries; Jonathan Tuke; Lewis Mitchell
>
> **备注:** 8 pages, 3 figures, to be submitted
>
> **摘要:** Topic modelling in Natural Language Processing uncovers hidden topics in large, unlabelled text datasets. It is widely applied in fields such as information retrieval, content summarisation, and trend analysis across various disciplines. However, probabilistic topic models can produce different results when rerun due to their stochastic nature, leading to inconsistencies in latent topics. Factors like corpus shuffling, rare text removal, and document elimination contribute to these variations. This instability affects replicability, reliability, and interpretation, raising concerns about whether topic models capture meaningful topics or just noise. To address these problems, we defined a new stability measure that incorporates accuracy and consistency and uses the generative properties of LDA to generate a new corpus with ground truth. These generated corpora are run through LDA 50 times to determine the variability in the output. We show that LDA can correctly determine the underlying number of topics in the documents. We also find that LDA is more internally consistent, as the multiple reruns return similar topics; however, these topics are not the true topics.
>
---
#### [new 003] Mitigating Length Bias in RLHF through a Causal Lens
- **分类: cs.CL; cs.AI**

- **简介: 论文属于RLHF任务，针对奖励模型中的长度偏差问题，提出基于因果推理的缓解方法。通过构造反事实数据对，分离内容质量与冗长度，训练更关注内容的奖励模型，提升输出的简洁性和准确性。**

- **链接: [https://arxiv.org/pdf/2511.12573v1](https://arxiv.org/pdf/2511.12573v1)**

> **作者:** Hyeonji Kim; Sujeong Oh; Sanghack Lee
>
> **摘要:** Reinforcement learning from human feedback (RLHF) is widely used to align large language models (LLMs) with human preferences. However, RLHF-trained reward models often exhibit length bias -- a systematic tendency to favor longer responses by conflating verbosity with quality. We propose a causal framework for analyzing and mitigating length bias in RLHF reward modeling. Central to our approach is a counterfactual data augmentation method that generates response pairs designed to isolate content quality from verbosity. These counterfactual examples are then used to train the reward model, enabling it to assess responses based on content quality independently of verbosity. Specifically, we construct (1) length-divergent pairs with similar content and (2) content-divergent pairs of similar length. Empirical evaluations show that our method reduces length bias in reward assignment and leads to more concise, content-focused outputs from the policy model. These findings demonstrate that the proposed approach effectively reduces length bias and improves the robustness and content sensitivity of reward modeling in RLHF pipelines.
>
---
#### [new 004] Donors and Recipients: On Asymmetric Transfer Across Tasks and Languages with Parameter-Efficient Fine-Tuning
- **分类: cs.CL; cs.AI**

- **简介: 论文研究大语言模型在不同任务和语言间的不对称迁移规律，通过PEFT/LoRA方法系统分析转移效果，发现任务内迁移正向、任务间迁移常导致性能下降，并识别出稳定的关键捐赠者与脆弱接收者结构。**

- **链接: [https://arxiv.org/pdf/2511.13368v1](https://arxiv.org/pdf/2511.13368v1)**

> **作者:** Kajetan Dymkiewicz; Ivan Vulic; Helen Yannakoudakis; Eilam Shapira; Roi Reichart; Anna Korhonen
>
> **摘要:** Large language models (LLMs) perform strongly across tasks and languages, yet how improvements in one task or language affect other tasks and languages and their combinations remains poorly understood. We conduct a controlled PEFT/LoRA study across multiple open-weight LLM families and sizes, treating task and language as transfer axes while conditioning on model family and size; we fine-tune each model on a single task-language source and measure transfer as the percentage-point change versus its baseline score when evaluated on all other task-language target pairs. We decompose transfer into (i) Matched-Task (Cross-Language), (ii) Matched-Language (Cross-Task), and (iii) Cross-Task (Cross-Language) regimes. We uncover two consistent general patterns. First, a pronounced on-task vs. off-task asymmetry: Matched-Task (Cross-Language) transfer is reliably positive, whereas off-task transfer often incurs collateral degradation. Second, a stable donor-recipient structure across languages and tasks (hub donors vs. brittle recipients). We outline implications for risk-aware fine-tuning and model specialisation.
>
---
#### [new 005] Evidence of Phase Transitions in Small Transformer-Based Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究小规模Transformer语言模型中的相变现象，旨在揭示其是否存在于小型模型中、能否在训练线性空间中检测到及是否早期出现。通过分析词汇使用和统计特性，发现相变可直接观测且早于标准损失曲线显现，表明其为普遍训练特征。**

- **链接: [https://arxiv.org/pdf/2511.12768v1](https://arxiv.org/pdf/2511.12768v1)**

> **作者:** Noah Hong; Tao Hong
>
> **摘要:** Phase transitions have been proposed as the origin of emergent abilities in large language models (LLMs), where new capabilities appear abruptly once models surpass critical thresholds of scale. Prior work, such as that of Wei et al., demonstrated these phenomena under model and data scaling, with transitions revealed after applying a log scale to training compute. In this work, we ask three complementary questions: (1) Are phase transitions unique to large models, or can they also be observed in small transformer-based language models? (2) Can such transitions be detected directly in linear training space, rather than only after log rescaling? and (3) Can these transitions emerge at early stages of training? To investigate, we train a small GPT-style transformer on a character-level corpus and analyze the evolution of vocabulary usage throughout training. We track the average word length, the number of correct versus incorrect words, and shifts in vocabulary diversity. Building on these measures, we apply Poisson and sub-Poisson statistics to quantify how words connect and reorganize. This combined analysis reveals a distinct transition point during training. Notably, these transitions are not apparent in standard loss or validation curves, but become visible through our vocabulary- and statistics-based probes. Our findings suggest that phase-transition reorganizations are a general feature of language model training, observable even in modest models, detectable directly in linear training space, and occurring surprisingly early as coherence emerges. This perspective provides new insight into the nonlinear dynamics of language model training and underscores the importance of tailored metrics for uncovering phase transition behaviors
>
---
#### [new 006] TAdaRAG: Task Adaptive Retrieval-Augmented Generation via On-the-Fly Knowledge Graph Construction
- **分类: cs.CL**

- **简介: 论文提出TAdaRAG框架，解决传统RAG因知识碎片化和非结构化导致的幻觉与推理断裂问题。通过动态构建任务自适应知识图谱，实现精准、连贯的知识整合，在多个基准上优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.12520v1](https://arxiv.org/pdf/2511.12520v1)**

> **作者:** Jie Zhang; Bo Tang; Wanzi Shao; Wenqiang Wei; Jihao Zhao; Jianqing Zhu; Zhiyu li; Wen Xi; Zehao Lin; Feiyu Xiong; Yanchao Tan
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Retrieval-Augmented Generation (RAG) improves large language models by retrieving external knowledge, often truncated into smaller chunks due to the input context window, which leads to information loss, resulting in response hallucinations and broken reasoning chains. Moreover, traditional RAG retrieves unstructured knowledge, introducing irrelevant details that hinder accurate reasoning. To address these issues, we propose TAdaRAG, a novel RAG framework for on-the-fly task-adaptive knowledge graph construction from external sources. Specifically, we design an intent-driven routing mechanism to a domain-specific extraction template, followed by supervised fine-tuning and a reinforcement learning-based implicit extraction mechanism, ensuring concise, coherent, and non-redundant knowledge integration. Evaluations on six public benchmarks and a real-world business benchmark (NowNewsQA) across three backbone models demonstrate that TAdaRAG outperforms existing methods across diverse domains and long-text tasks, highlighting its strong generalization and practical effectiveness.
>
---
#### [new 007] A Reasoning Paradigm for Named Entity Recognition
- **分类: cs.CL**

- **简介: 该论文聚焦命名实体识别（NER）任务，针对生成式大模型依赖隐式模式匹配、缺乏可验证推理机制的问题，提出ReasoningNER框架。通过三阶段流程：生成带推理链的标注数据、训练模型输出推理过程、强化推理优化，显著提升零样本场景下的性能，优于GPT-4 12.3% F1分数。**

- **链接: [https://arxiv.org/pdf/2511.11978v1](https://arxiv.org/pdf/2511.11978v1)**

> **作者:** Hui Huang; Yanping Chen; Ruizhang Huang; Chuan Lin; Yongbin Qin
>
> **备注:** Accepted at AAAI 2026
>
> **摘要:** Generative LLMs typically improve Named Entity Recognition (NER) performance through instruction tuning. They excel at generating entities by semantic pattern matching but lack an explicit, verifiable reasoning mechanism. This "cognitive shortcutting" leads to suboptimal performance and brittle generalization, especially in zero-shot and lowresource scenarios where reasoning from limited contextual cues is crucial. To address this issue, a reasoning framework is proposed for NER, which shifts the extraction paradigm from implicit pattern matching to explicit reasoning. This framework consists of three stages: Chain of Thought (CoT) generation, CoT tuning, and reasoning enhancement. First, a dataset annotated with NER-oriented CoTs is generated, which contain task-relevant reasoning chains. Then, they are used to tune the NER model to generate coherent rationales before deriving the final answer. Finally, a reasoning enhancement stage is implemented to optimize the reasoning process using a comprehensive reward signal. This stage ensures explicit and verifiable extractions. Experiments show that ReasoningNER demonstrates impressive cognitive ability in the NER task, achieving competitive performance. In zero-shot settings, it achieves state-of-the-art (SOTA) performance, outperforming GPT-4 by 12.3 percentage points on the F1 score. Analytical results also demonstrate its great potential to advance research in reasoningoriented information extraction. Our codes are available at https://github.com/HuiResearch/ReasoningIE.
>
---
#### [new 008] On the Notion that Language Models Reason
- **分类: cs.CL; cs.AI**

- **简介: 论文探讨语言模型是否真正具备推理能力，指出其输出的“推理”实为统计模式匹配。作者基于Transformer结构提出隐式马尔可夫核视角，说明模型依赖数据规律而非逻辑机制，澄清了推理类输出与逻辑一致性间的区别，强调需正确认识模型计算过程。**

- **链接: [https://arxiv.org/pdf/2511.11810v1](https://arxiv.org/pdf/2511.11810v1)**

> **作者:** Bertram Højer
>
> **备注:** Accepted at the 1st Workshop on Epistemic Intelligence in Machine Learning, EurIPS 2025
>
> **摘要:** Language models (LMs) are said to be exhibiting reasoning, but what does this entail? We assess definitions of reasoning and how key papers in the field of natural language processing (NLP) use the notion and argue that the definitions provided are not consistent with how LMs are trained, process information, and generate new tokens. To illustrate this incommensurability we assume the view that transformer-based LMs implement an \textit{implicit} finite-order Markov kernel mapping contexts to conditional token distributions. In this view, reasoning-like outputs correspond to statistical regularities and approximate statistical invariances in the learned kernel rather than the implementation of explicit logical mechanisms. This view is illustrative of the claim that LMs are "statistical pattern matchers"" and not genuine reasoners and provides a perspective that clarifies why reasoning-like outputs arise in LMs without any guarantees of logical consistency. This distinction is fundamental to how epistemic uncertainty is evaluated in LMs. We invite a discussion on the importance of how the computational processes of the systems we build and analyze in NLP research are described.
>
---
#### [new 009] Cmprsr: Abstractive Token-Level Question-Agnostic Prompt Compressor
- **分类: cs.CL; cs.LG**

- **简介: 论文提出Cmprsr，一种用于压缩大语言模型输入的抽象式、无问题感知的提示压缩方法，旨在降低LLM使用成本。通过基准测试发现模型压缩能力差异，并优化GPT-4.1-mini和Qwen3-4B模型，最终实现高精度压缩率控制与任务性能提升。**

- **链接: [https://arxiv.org/pdf/2511.12281v1](https://arxiv.org/pdf/2511.12281v1)**

> **作者:** Ivan Zakazov; Alexander Sharipov; Berke Argin; Oussama Gabouj; Kamel Charaf; Alexi Semiz; Lorenzo Drudi; Nicolas Baldwin; Robert West
>
> **摘要:** Motivated by the high costs of using black-box Large Language Models (LLMs), we introduce a novel prompt compression paradigm, under which we use smaller LLMs to compress inputs for the larger ones. We present the first comprehensive LLM-as-a-compressor benchmark spanning 25 open- and closed-source models, which reveals significant disparity in models' compression ability in terms of (i) preserving semantically important information (ii) following the user-provided compression rate (CR). We further improve the performance of gpt-4.1-mini, the best overall vanilla compressor, with Textgrad-based compression meta-prompt optimization. We also identify the most promising open-source vanilla LLM - Qwen3-4B - and post-train it with a combination of supervised fine-tuning (SFT) and Group Relative Policy Optimization (GRPO), pursuing the dual objective of CR adherence and maximizing the downstream task performance. We call the resulting model Cmprsr and demonstrate its superiority over both extractive and vanilla abstractive compression across the entire range of compression rates on lengthy inputs from MeetingBank and LongBench as well as short prompts from GSM8k. The latter highlights Cmprsr's generalizability across varying input lengths and domains. Moreover, Cmprsr closely follows the requested compression rate, offering fine control over the cost-quality trade-off.
>
---
#### [new 010] From Passive to Persuasive: Steering Emotional Nuance in Human-AI Negotiation
- **分类: cs.CL; cs.AI**

- **简介: 论文研究如何让AI在谈判中更自然地表达情绪。针对当前模型情感表达单一的问题，提出通过激活工程精准引导情绪特征，无需大量微调。工作包括识别关键神经元、构建情绪向量并应用于新对话，显著提升情感丰富度与个人化表达。**

- **链接: [https://arxiv.org/pdf/2511.12832v1](https://arxiv.org/pdf/2511.12832v1)**

> **作者:** Niranjan Chebrolu; Gerard Christopher Yeo; Kokil Jaidka
>
> **摘要:** Large Language Models (LLMs) demonstrate increasing conversational fluency, yet instilling them with nuanced, human-like emotional expression remains a significant challenge. Current alignment techniques often address surface-level output or require extensive fine-tuning. This paper demonstrates that targeted activation engineering can steer LLaMA 3.1-8B to exhibit more human-like emotional nuances. We first employ attribution patching to identify causally influential components, to find a key intervention locus by observing activation patterns during diagnostic conversational tasks. We then derive emotional expression vectors from the difference in the activations generated by contrastive text pairs (positive vs. negative examples of target emotions). Applying these vectors to new conversational prompts significantly enhances emotional characteristics: steered responses show increased positive sentiment (e.g., joy, trust) and more frequent first-person pronoun usage, indicative of greater personal engagement. Our findings offer a precise and interpretable framework and new directions for the study of conversational AI.
>
---
#### [new 011] MedPT: A Massive Medical Question Answering Dataset for Brazilian-Portuguese Speakers
- **分类: cs.CL**

- **简介: 论文提出MedPT，首个针对巴西葡萄牙语的大型医疗问答数据集，解决低资源语言在医疗LLM中的适配问题。通过多阶段清洗与LLM增强，构建38.4万条真实问答回合，涵盖3200主题，支持医学分诊任务，F1达94%。**

- **链接: [https://arxiv.org/pdf/2511.11878v1](https://arxiv.org/pdf/2511.11878v1)**

> **作者:** Fernanda Bufon Färber; Iago Alves Brito; Julia Soares Dollis; Pedro Schindler Freire Brasil Ribeiro; Rafael Teixeira Sousa; Arlindo Rodrigues Galvão Filho
>
> **备注:** 11 pages, 3 tables, 2 figures
>
> **摘要:** While large language models (LLMs) show transformative potential in healthcare, their development remains focused on high-resource languages, creating a critical barrier for others as simple translation fails to capture unique clinical and cultural nuances, such as endemic diseases. To address this, we introduce MedPT, the first large-scale, real-world corpus for Brazilian Portuguese, comprising 384,095 authentic question-answer pairs from patient-doctor interactions. The dataset underwent a meticulous multi-stage curation protocol, using a hybrid quantitative-qualitative analysis to filter noise and contextually enrich thousands of ambiguous queries. We further augmented the corpus via LLM-driven annotation, classifying questions into seven semantic types to capture user intent. Our analysis reveals its thematic breadth (3,200 topics) and unique linguistic properties, like the natural asymmetry in patient-doctor communication. To validate its utility, we benchmark a medical specialty routing task: fine-tuning a 1.7B parameter model achieves an outstanding 94\% F1-score on a 20-class setup. Furthermore, our qualitative error analysis shows misclassifications are not random but reflect genuine clinical ambiguities (e.g., between comorbid conditions), proving the dataset's deep semantic richness. We publicly release MedPT to foster the development of more equitable, accurate, and culturally-aware medical technologies for the Portuguese-speaking world.
>
---
#### [new 012] AugAbEx : Way Forward for Extractive Case Summarization
- **分类: cs.CL**

- **简介: 论文提出AugAbEx方法，将已有抽象式法律文书摘要转换为提取式摘要，解决标注成本高问题。通过保留原摘要专家意见，增强数据资源，促进法律文本自动摘要研究。**

- **链接: [https://arxiv.org/pdf/2511.12290v1](https://arxiv.org/pdf/2511.12290v1)**

> **作者:** Purnima Bindal; Vikas Kumar; Sagar Rathore; Vasudha Bhatnagar
>
> **备注:** 30 pages, under review in a Journal
>
> **摘要:** Summarization of legal judgments poses a heavy cognitive burden on law practitioners due to the complexity of the language, context-sensitive legal jargon, and the length of the document. Therefore, the automatic summarization of legal documents has attracted serious attention from natural language processing researchers. Since the abstractive summaries of legal documents generated by deep neural methods remain prone to the risk of misrepresenting nuanced legal jargon or overlooking key contextual details, we envisage a rising trend toward the use of extractive case summarizers. Given the high cost of human annotation for gold standard extractive summaries, we engineer a light and transparent pipeline that leverages existing abstractive gold standard summaries to create the corresponding extractive gold standard versions. The approach ensures that the experts` opinions ensconced in the original gold standard abstractive summaries are carried over to the transformed extractive summaries. We aim to augment seven existing case summarization datasets, which include abstractive summaries, by incorporating corresponding extractive summaries and create an enriched data resource for case summarization research community. To ensure the quality of the augmented extractive summaries, we perform an extensive comparative evaluation with the original abstractive gold standard summaries covering structural, lexical, and semantic dimensions. We also compare the domain-level information of the two summaries. We commit to release the augmented datasets in the public domain for use by the research community and believe that the resource will offer opportunities to advance the field of automatic summarization of legal documents.
>
---
#### [new 013] Why is "Chicago" Predictive of Deceptive Reviews? Using LLMs to Discover Language Phenomena from Lexical Cues
- **分类: cs.CL; cs.LG**

- **简介: 论文属于欺骗性评论检测任务，旨在解释机器学习模型如何区分虚假评论。作者利用大语言模型将复杂的词汇线索转化为可理解的语言现象，提升人类对在线评论可信度的判断能力。**

- **链接: [https://arxiv.org/pdf/2511.13658v1](https://arxiv.org/pdf/2511.13658v1)**

> **作者:** Jiaming Qu; Mengtian Guo; Yue Wang
>
> **摘要:** Deceptive reviews mislead consumers, harm businesses, and undermine trust in online marketplaces. Machine learning classifiers can learn from large amounts of training examples to effectively distinguish deceptive reviews from genuine ones. However, the distinguishing features learned by these classifiers are often subtle, fragmented, and difficult for humans to interpret. In this work, we explore using large language models (LLMs) to translate machine-learned lexical cues into human-understandable language phenomena that can differentiate deceptive reviews from genuine ones. We show that language phenomena obtained in this manner are empirically grounded in data, generalizable across similar domains, and more predictive than phenomena either in LLMs' prior knowledge or obtained through in-context learning. These language phenomena have the potential to aid people in critically assessing the credibility of online reviews in environments where deception detection classifiers are unavailable.
>
---
#### [new 014] QA-Noun: Representing Nominal Semantics via Natural Language Question-Answer Pairs
- **分类: cs.CL**

- **简介: 该论文提出QA-Noun框架，用于捕捉名词中心语义关系，解决现有QA方法忽视名词语义的问题。通过九种问题模板生成可解释的问答对，补充谓词-论元关系建模，提升句子细粒度语义分解的完整性和精度。**

- **链接: [https://arxiv.org/pdf/2511.12504v1](https://arxiv.org/pdf/2511.12504v1)**

> **作者:** Maria Tseytlin; Paul Roit; Omri Abend; Ido Dagan; Ayal Klein
>
> **摘要:** Decomposing sentences into fine-grained meaning units is increasingly used to model semantic alignment. While QA-based semantic approaches have shown effectiveness for representing predicate-argument relations, they have so far left noun-centered semantics largely unaddressed. We introduce QA-Noun, a QA-based framework for capturing noun-centered semantic relations. QA-Noun defines nine question templates that cover both explicit syntactical and implicit contextual roles for nouns, producing interpretable QA pairs that complement verbal QA-SRL. We release detailed guidelines, a dataset of over 2,000 annotated noun mentions, and a trained model integrated with QA-SRL to yield a unified decomposition of sentence meaning into individual, highly fine-grained, facts. Evaluation shows that QA-Noun achieves near-complete coverage of AMR's noun arguments while surfacing additional contextually implied relations, and that combining QA-Noun with QA-SRL yields over 130\% higher granularity than recent fact-based decomposition methods such as FactScore and DecompScore. QA-Noun thus complements the broader QA-based semantic framework, forming a comprehensive and scalable approach to fine-grained semantic decomposition for cross-text alignment.
>
---
#### [new 015] CURE: Cultural Understanding and Reasoning Evaluation - A Framework for "Thick" Culture Alignment Evaluation in LLMs
- **分类: cs.CL; cs.HC**

- **简介: 该论文提出CURE框架，用于评估大语言模型在多元文化环境中的文化理解与推理能力。针对现有评价方法过于简单、忽视情境化推理的问题，设计了基于真实场景的基准测试和多维指标，揭示了“厚评价”能更稳定、深入地反映模型的文化胜任力。**

- **链接: [https://arxiv.org/pdf/2511.12014v1](https://arxiv.org/pdf/2511.12014v1)**

> **作者:** Truong Vo; Sanmi Koyejo
>
> **备注:** 7 pages, 5 figures
>
> **摘要:** Large language models (LLMs) are increasingly deployed in culturally diverse environments, yet existing evaluations of cultural competence remain limited. Existing methods focus on de-contextualized correctness or forced-choice judgments, overlooking the need for cultural understanding and reasoning required for appropriate responses. To address this gap, we introduce a set of benchmarks that, instead of directly probing abstract norms or isolated statements, present models with realistic situational contexts that require culturally grounded reasoning. In addition to the standard Exact Match metric, we introduce four complementary metrics (Coverage, Specificity, Connotation, and Coherence) to capture different dimensions of model's response quality. Empirical analysis across frontier models reveals that thin evaluation systematically overestimates cultural competence and produces unstable assessments with high variance. In contrast, thick evaluation exposes differences in reasoning depth, reduces variance, and provides more stable, interpretable signals of cultural understanding.
>
---
#### [new 016] Crossing Borders: A Multimodal Challenge for Indian Poetry Translation and Image Generation
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出TAI框架，解决印度语言诗歌翻译与图像生成难题，提升其全球可及性。工作包括：基于Odds Ratio算法的翻译模块、基于语义图的图像生成模块，并发布MorphoVerse数据集。**

- **链接: [https://arxiv.org/pdf/2511.13689v1](https://arxiv.org/pdf/2511.13689v1)**

> **作者:** Sofia Jamil; Kotla Sai Charan; Sriparna Saha; Koustava Goswami; Joseph K J
>
> **摘要:** Indian poetry, known for its linguistic complexity and deep cultural resonance, has a rich and varied heritage spanning thousands of years. However, its layered meanings, cultural allusions, and sophisticated grammatical constructions often pose challenges for comprehension, especially for non-native speakers or readers unfamiliar with its context and language. Despite its cultural significance, existing works on poetry have largely overlooked Indian language poems. In this paper, we propose the Translation and Image Generation (TAI) framework, leveraging Large Language Models (LLMs) and Latent Diffusion Models through appropriate prompt tuning. Our framework supports the United Nations Sustainable Development Goals of Quality Education (SDG 4) and Reduced Inequalities (SDG 10) by enhancing the accessibility of culturally rich Indian-language poetry to a global audience. It includes (1) a translation module that uses an Odds Ratio Preference Alignment Algorithm to accurately translate morphologically rich poetry into English, and (2) an image generation module that employs a semantic graph to capture tokens, dependencies, and semantic relationships between metaphors and their meanings, to create visually meaningful representations of Indian poems. Our comprehensive experimental evaluation, including both human and quantitative assessments, demonstrates the superiority of TAI Diffusion in poem image generation tasks, outperforming strong baselines. To further address the scarcity of resources for Indian-language poetry, we introduce the Morphologically Rich Indian Language Poems MorphoVerse Dataset, comprising 1,570 poems across 21 low-resource Indian languages. By addressing the gap in poetry translation and visual comprehension, this work aims to broaden accessibility and enrich the reader's experience.
>
---
#### [new 017] Visual Room 2.0: Seeing is Not Understanding for MLLMs
- **分类: cs.CL**

- **简介: 论文提出Visual Room 2.0基准，评估多模态大模型从感知到认知的对齐能力。针对“看见不等于理解”的问题，构建三层任务体系（感知→认知），涵盖17项任务、350样本、2100问。发现模型感知强于认知，且认知不依赖感知，规模提升有助于认知但不改善感知。**

- **链接: [https://arxiv.org/pdf/2511.12928v1](https://arxiv.org/pdf/2511.12928v1)**

> **作者:** Haokun Li; Yazhou Zhang; Jizhi Ding; Qiuchi Li; Peng Zhang
>
> **摘要:** Can multi-modal large language models (MLLMs) truly understand what they can see? Extending Searle's Chinese Room into the multi-modal domain, this paper proposes the Visual Room argument: MLLMs may describe every visual detail precisely yet fail to comprehend the underlying emotions and intentions, namely seeing is not understanding. Building on this, we introduce \textit{Visual Room} 2.0, a hierarchical benchmark for evaluating perception-cognition alignment of MLLMs. We model human perceptive and cognitive processes across three levels: low, middle, and high, covering 17 representative tasks. The perception component ranges from attribute recognition to scene understanding, while the cognition component extends from textual entailment to causal and social reasoning. The dataset contains 350 multi-modal samples, each with six progressive questions (2,100 in total) spanning perception to cognition. Evaluating 10 state-of-the-art (SoTA) MLLMs, we highlight three key findings: (1) MLLMs exhibit stronger perceptual competence than cognitive ability (8.0\%$\uparrow$); (2) cognition appears not causally dependent on perception-based reasoning; and (3) cognition scales with model size, but perception does not consistently improve with larger variants. This work operationalizes Seeing $\ne$ Understanding as a testable hypothesis, offering a new paradigm from perceptual processing to cognitive reasoning in MLLMs. Our dataset is available at https://huggingface.co/datasets/LHK2003/PCBench.
>
---
#### [new 018] Evolve the Method, Not the Prompts: Evolutionary Synthesis of Jailbreak Attacks on LLMs
- **分类: cs.CL; cs.CR**

- **简介: 论文提出EvoSynth框架，通过进化合成新攻击方法解决LLM jailbreak攻击创造力不足的问题。它用多智能体系统自主设计、演化代码级攻击，并具自纠错能力，实现85.5%攻击成功率，优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.12710v1](https://arxiv.org/pdf/2511.12710v1)**

> **作者:** Yunhao Chen; Xin Wang; Juncheng Li; Yixu Wang; Jie Li; Yan Teng; Yingchun Wang; Xingjun Ma
>
> **摘要:** Automated red teaming frameworks for Large Language Models (LLMs) have become increasingly sophisticated, yet they share a fundamental limitation: their jailbreak logic is confined to selecting, combining, or refining pre-existing attack strategies. This binds their creativity and leaves them unable to autonomously invent entirely new attack mechanisms. To overcome this gap, we introduce \textbf{EvoSynth}, an autonomous framework that shifts the paradigm from attack planning to the evolutionary synthesis of jailbreak methods. Instead of refining prompts, EvoSynth employs a multi-agent system to autonomously engineer, evolve, and execute novel, code-based attack algorithms. Crucially, it features a code-level self-correction loop, allowing it to iteratively rewrite its own attack logic in response to failure. Through extensive experiments, we demonstrate that EvoSynth not only establishes a new state-of-the-art by achieving an 85.5\% Attack Success Rate (ASR) against highly robust models like Claude-Sonnet-4.5, but also generates attacks that are significantly more diverse than those from existing methods. We release our framework to facilitate future research in this new direction of evolutionary synthesis of jailbreak methods. Code is available at: https://github.com/dongdongunique/EvoSynth.
>
---
#### [new 019] LLM Reinforcement in Context
- **分类: cs.CL; cs.CR**

- **简介: 论文研究LLM对齐问题，针对长对话中模型易被越狱的缺陷，提出通过插入控制句来增强对齐稳定性，尤其适用于链式思维过程以防止策略性行为。**

- **链接: [https://arxiv.org/pdf/2511.12782v1](https://arxiv.org/pdf/2511.12782v1)**

> **作者:** Thomas Rivasseau
>
> **备注:** 4 pages
>
> **摘要:** Current Large Language Model alignment research mostly focuses on improving model robustness against adversarial attacks and misbehavior by training on examples and prompting. Research has shown that LLM jailbreak probability increases with the size of the user input or conversation length. There is a lack of appropriate research into means of strengthening alignment which also scale with user input length. We propose interruptions as a possible solution to this problem. Interruptions are control sentences added to the user input approximately every x tokens for some arbitrary x. We suggest that this can be generalized to the Chain-of-Thought process to prevent scheming.
>
---
#### [new 020] Souper-Model: How Simple Arithmetic Unlocks State-of-the-Art LLM Performance
- **分类: cs.CL**

- **简介: 论文提出SoCE方法，通过非均匀加权平均多个专家模型提升LLM性能。针对模型融合中均匀平均效果有限的问题，利用基准测试类别低相关性识别专家模型，优化组合策略，在多语言、工具调用等任务上达到SOTA。**

- **链接: [https://arxiv.org/pdf/2511.13254v1](https://arxiv.org/pdf/2511.13254v1)**

> **作者:** Shalini Maiti; Amar Budhiraja; Bhavul Gauri; Gaurav Chaurasia; Anton Protopopov; Alexis Audran-Reiss; Michael Slater; Despoina Magka; Tatiana Shavrina; Roberta Raileanu; Yoram Bachrach
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse domains, but their training remains resource- and time-intensive, requiring massive compute power and careful orchestration of training procedures. Model souping-the practice of averaging weights from multiple models of the same architecture-has emerged as a promising pre- and post-training technique that can enhance performance without expensive retraining. In this paper, we introduce Soup Of Category Experts (SoCE), a principled approach for model souping that utilizes benchmark composition to identify optimal model candidates and applies non-uniform weighted averaging to maximize performance. Contrary to previous uniform-averaging approaches, our method leverages the observation that benchmark categories often exhibit low inter-correlations in model performance. SoCE identifies "expert" models for each weakly-correlated category cluster and combines them using optimized weighted averaging rather than uniform weights. We demonstrate that the proposed method improves performance and robustness across multiple domains, including multilingual capabilities, tool calling, and math and achieves state-of-the-art results on the Berkeley Function Calling Leaderboard.
>
---
#### [new 021] Seeing is Believing: Rich-Context Hallucination Detection for MLLMs via Backward Visual Grounding
- **分类: cs.CL; cs.CV**

- **简介: 论文提出VBackChecker框架，用于检测多模态大模型的幻觉问题。通过像素级视觉定位实现无需参考的 hallucination 检测，提升准确性与可解释性，并构建新基准R²-HalBench验证效果。**

- **链接: [https://arxiv.org/pdf/2511.12140v1](https://arxiv.org/pdf/2511.12140v1)**

> **作者:** Pinxue Guo; Chongruo Wu; Xinyu Zhou; Lingyi Hong; Zhaoyu Chen; Jinglun Li; Kaixun Jiang; Sen-ching Samson Cheung; Wei Zhang; Wenqiang Zhang
>
> **摘要:** Multimodal Large Language Models (MLLMs) have unlocked powerful cross-modal capabilities, but still significantly suffer from hallucinations. As such, accurate detection of hallucinations in MLLMs is imperative for ensuring their reliability in practical applications. To this end, guided by the principle of "Seeing is Believing", we introduce VBackChecker, a novel reference-free hallucination detection framework that verifies the consistency of MLLMgenerated responses with visual inputs, by leveraging a pixellevel Grounding LLM equipped with reasoning and referring segmentation capabilities. This reference-free framework not only effectively handles rich-context scenarios, but also offers interpretability. To facilitate this, an innovative pipeline is accordingly designed for generating instruction-tuning data (R-Instruct), featuring rich-context descriptions, grounding masks, and hard negative samples. We further establish R^2 -HalBench, a new hallucination benchmark for MLLMs, which, unlike previous benchmarks, encompasses real-world, rich-context descriptions from 18 MLLMs with high-quality annotations, spanning diverse object-, attribute, and relationship-level details. VBackChecker outperforms prior complex frameworks and achieves state-of-the-art performance on R^2 -HalBench, even rivaling GPT-4o's capabilities in hallucination detection. It also surpasses prior methods in the pixel-level grounding task, achieving over a 10% improvement. All codes, data, and models are available at https://github.com/PinxueGuo/VBackChecker.
>
---
#### [new 022] Fine-Tuned LLMs Know They Don't Know: A Parameter-Efficient Approach to Recovering Honesty
- **分类: cs.CL**

- **简介: 论文针对LLM在监督微调后诚实性下降的问题，提出HCNR方法，通过修复关键神经元恢复模型对知识边界的认知表达能力，实现高效、低数据消耗的诚实性修复。**

- **链接: [https://arxiv.org/pdf/2511.12991v1](https://arxiv.org/pdf/2511.12991v1)**

> **作者:** Zeyu Shi; Ziming Wang; Tianyu Chen; Shiqi Gao; Haoyi Zhou; Qingyun Sun; Jianxin Li
>
> **备注:** Accepted by AAAI 2026 Main Track
>
> **摘要:** The honesty of Large Language Models (LLMs) is increasingly important for safe deployment in high-stakes domains. However, this crucial trait is severely undermined by supervised fine-tuning (SFT), a common technique for model specialization. Existing recovery methods rely on data-intensive global parameter adjustments, implicitly assuming that SFT deeply corrupts the models' ability to recognize their knowledge boundaries. However, we observe that fine-tuned LLMs still preserve this ability; what is damaged is their capacity to faithfully express that awareness. Building on this, we propose Honesty-Critical Neurons Restoration (HCNR) to surgically repair this suppressed capacity. HCNR identifies and restores key expression-governing neurons to their pre-trained state while harmonizing them with task-oriented neurons via Hessian-guided compensation. Experiments on four QA tasks and five LLM families demonstrate that HCNR effectively recovers 33.25% of the compromised honesty while achieving at least 2.23x speedup with over 10x less data compared to baseline methods, offering a practical solution for trustworthy LLM deployment.
>
---
#### [new 023] Don't Think of the White Bear: Ironic Negation in Transformer Models Under Cognitive Load
- **分类: cs.CL; cs.AI**

- **简介: 论文研究大语言模型在否定指令下的“讽刺反弹”现象，即禁止提及某概念反而使其更易被激活。通过两组实验和电路追踪分析，发现认知负荷与语义干扰加剧反弹，而极性区分增强反弹持续性，揭示了模型内部机制与人类认知的相似性。**

- **链接: [https://arxiv.org/pdf/2511.12381v1](https://arxiv.org/pdf/2511.12381v1)**

> **作者:** Logan Mann; Nayan Saxena; Sarah Tandon; Chenhao Sun; Savar Toteja; Kevin Zhu
>
> **摘要:** Negation instructions such as 'do not mention $X$' can paradoxically increase the accessibility of $X$ in human thought, a phenomenon known as ironic rebound. Large language models (LLMs) face the same challenge: suppressing a concept requires internally activating it, which may prime rebound instead of avoidance. We investigated this tension with two experiments. \textbf{(1) Load \& content}: after a negation instruction, we vary distractor text (semantic, syntactic, repetition) and measure rebound strength. \textbf{(2) Polarity separation}: We test whether models distinguish neutral from negative framings of the same concept and whether this separation predicts rebound persistence. Results show that rebound consistently arises immediately after negation and intensifies with longer or semantic distractors, while repetition supports suppression. Stronger polarity separation correlates with more persistent rebound. Together, these findings, complemented by a circuit tracing analysis that identifies sparse middle-layer attention heads amplifying forbidden tokens while early layers suppress, link cognitive predictions of ironic rebound with mechanistic insights into long-context interference. To support future work, we release ReboundBench, a dataset of $5,000$ systematically varied negation prompts designed to probe rebound in LLMs.
>
---
#### [new 024] NeuroLex: A Lightweight Domain Language Model for EEG Report Understanding and Generation
- **分类: cs.CL; cs.AI**

- **简介: 论文提出NeuroLex，一个轻量级领域自适应语言模型，专为EEG报告理解与生成设计。解决通用语言模型无法捕捉EEG领域语言特征的问题。通过预训练和指令微调，提升文本理解和生成准确性，增强鲁棒性与效率。**

- **链接: [https://arxiv.org/pdf/2511.12851v1](https://arxiv.org/pdf/2511.12851v1)**

> **作者:** Kang Yin; Hye-Bin Shin
>
> **摘要:** Clinical electroencephalogram (EEG) reports encode domain-specific linguistic conventions that general-purpose language models (LMs) fail to capture. We introduce NeuroLex, a lightweight domain-adaptive language model trained purely on EEG report text from the Harvard Electroencephalography Database. Unlike existing biomedical LMs, NeuroLex is tailored to the linguistic and diagnostic characteristics of EEG reporting, enabling it to serve as both an independent textual model and a decoder backbone for multimodal EEG-language systems. Using span-corruption pretraining and instruction-style fine-tuning on report polishing, paragraph summarization, and terminology question answering, NeuroLex learns the syntax and reasoning patterns characteristic of EEG interpretation. Comprehensive evaluations show that it achieves lower perplexity, higher extraction and summarization accuracy, better label efficiency, and improved robustness to negation and factual hallucination compared with general models of the same scale. With an EEG-aware linguistic backbone, NeuroLex bridges biomedical text modeling and brain-computer interface applications, offering a foundation for interpretable and language-driven neural decoding.
>
---
#### [new 025] Three Stage Narrative Analysis; Plot-Sentiment Breakdown, Structure Learning and Concept Detection
- **分类: cs.CL; cs.AI**

- **简介: 论文提出三阶段叙事分析框架，解决电影剧本情感弧与概念识别问题。通过自定义词典和聚类技术，实现情感分析与结构学习，提升叙事理解与推荐效果。**

- **链接: [https://arxiv.org/pdf/2511.11857v1](https://arxiv.org/pdf/2511.11857v1)**

> **作者:** Taimur Khan; Ramoza Ahsan; Mohib Hameed
>
> **备注:** 18 pages
>
> **摘要:** Story understanding and analysis have long been challenging areas within Natural Language Understanding. Automated narrative analysis requires deep computational semantic representations along with syntactic processing. Moreover, the large volume of narrative data demands automated semantic analysis and computational learning rather than manual analytical approaches. In this paper, we propose a framework that analyzes the sentiment arcs of movie scripts and performs extended analysis related to the context of the characters involved. The framework enables the extraction of high-level and low-level concepts conveyed through the narrative. Using dictionary-based sentiment analysis, our approach applies a custom lexicon built with the LabMTsimple storylab module. The custom lexicon is based on the Valence, Arousal, and Dominance scores from the NRC-VAD dataset. Furthermore, the framework advances the analysis by clustering similar sentiment plots using Wards hierarchical clustering technique. Experimental evaluation on a movie dataset shows that the resulting analysis is helpful to consumers and readers when selecting a narrative or story.
>
---
#### [new 026] Extracting Events Like Code: A Multi-Agent Programming Framework for Zero-Shot Event Extraction
- **分类: cs.CL; cs.AI**

- **简介: 论文提出AEC框架，将零样本事件抽取视为编程任务，通过多智能体协作实现结构化、迭代式代码生成，解决LLM在事件抽取中输出不完整或结构错误的问题。**

- **链接: [https://arxiv.org/pdf/2511.13118v1](https://arxiv.org/pdf/2511.13118v1)**

> **作者:** Quanjiang Guo; Sijie Wang; Jinchuan Zhang; Ben Zhang; Zhao Kang; Ling Tian; Ke Yan
>
> **备注:** 11 pages, 5 figures, accepted by AAAI 2026 (Oral)
>
> **摘要:** Zero-shot event extraction (ZSEE) remains a significant challenge for large language models (LLMs) due to the need for complex reasoning and domain-specific understanding. Direct prompting often yields incomplete or structurally invalid outputs--such as misclassified triggers, missing arguments, and schema violations. To address these limitations, we present Agent-Event-Coder (AEC), a novel multi-agent framework that treats event extraction like software engineering: as a structured, iterative code-generation process. AEC decomposes ZSEE into specialized subtasks--retrieval, planning, coding, and verification--each handled by a dedicated LLM agent. Event schemas are represented as executable class definitions, enabling deterministic validation and precise feedback via a verification agent. This programming-inspired approach allows for systematic disambiguation and schema enforcement through iterative refinement. By leveraging collaborative agent workflows, AEC enables LLMs to produce precise, complete, and schema-consistent extractions in zero-shot settings. Experiments across five diverse domains and six LLMs demonstrate that AEC consistently outperforms prior zero-shot baselines, showcasing the power of treating event extraction like code generation. The code and data are released on https://github.com/UESTC-GQJ/Agent-Event-Coder.
>
---
#### [new 027] From Perception to Reasoning: Deep Thinking Empowers Multimodal Large Language Models
- **分类: cs.CL; cs.CV**

- **简介: 论文聚焦多模态大模型的复杂推理能力提升，针对现有模型推理路径不透明、泛化不足的问题，系统综述了多模态思维链（MCoT）方法，涵盖其范式、训练与推理机制、评估体系及应用，并展望未来挑战与方向。**

- **链接: [https://arxiv.org/pdf/2511.12861v1](https://arxiv.org/pdf/2511.12861v1)**

> **作者:** Wenxin Zhu; Andong Chen; Yuchen Song; Kehai Chen; Conghui Zhu; Ziyan Chen; Tiejun Zhao
>
> **备注:** Survey; 7 figures, 3 tables, 44 pages
>
> **摘要:** With the remarkable success of Multimodal Large Language Models (MLLMs) in perception tasks, enhancing their complex reasoning capabilities has emerged as a critical research focus. Existing models still suffer from challenges such as opaque reasoning paths and insufficient generalization ability. Chain-of-Thought (CoT) reasoning, which has demonstrated significant efficacy in language models by enhancing reasoning transparency and output interpretability, holds promise for improving model reasoning capabilities when extended to the multimodal domain. This paper provides a systematic review centered on "Multimodal Chain-of-Thought" (MCoT). First, it analyzes the background and theoretical motivations for its inception from the perspectives of technical evolution and task demands. Then, it introduces mainstream MCoT methods from three aspects: CoT paradigms, the post-training stage, and the inference stage, while also analyzing their underlying mechanisms. Furthermore, the paper summarizes existing evaluation benchmarks and metrics, and discusses the application scenarios of MCoT. Finally, it analyzes the challenges currently facing MCoT and provides an outlook on its future research directions.
>
---
#### [new 028] Auditing Google's AI Overviews and Featured Snippets: A Case Study on Baby Care and Pregnancy
- **分类: cs.CL; cs.AI; cs.CY; cs.HC; cs.IR**

- **简介: 该论文属于AI内容质量审计任务，旨在解决谷歌AI摘要和精选摘要在母婴健康信息中的一致性与安全性问题。研究通过分析1508个查询，发现33%的摘要内容不一致，且医疗安全措施严重缺失（仅11%和7%）。**

- **链接: [https://arxiv.org/pdf/2511.12920v1](https://arxiv.org/pdf/2511.12920v1)**

> **作者:** Desheng Hu; Joachim Baumann; Aleksandra Urman; Elsa Lichtenegger; Robin Forsberg; Aniko Hannak; Christo Wilson
>
> **备注:** 18 pages, 10 figures; to appear in AAAI ICWSM 2026
>
> **摘要:** Google Search increasingly surfaces AI-generated content through features like AI Overviews (AIO) and Featured Snippets (FS), which users frequently rely on despite having no control over their presentation. Through a systematic algorithm audit of 1,508 real baby care and pregnancy-related queries, we evaluate the quality and consistency of these information displays. Our robust evaluation framework assesses multiple quality dimensions, including answer consistency, relevance, presence of medical safeguards, source categories, and sentiment alignment. Our results reveal concerning gaps in information consistency, with information in AIO and FS displayed on the same search result page being inconsistent with each other in 33% of cases. Despite high relevance scores, both features critically lack medical safeguards (present in just 11% of AIO and 7% of FS responses). While health and wellness websites dominate source categories for both, AIO and FS, FS also often link to commercial sources. These findings have important implications for public health information access and demonstrate the need for stronger quality controls in AI-mediated health information. Our methodology provides a transferable framework for auditing AI systems across high-stakes domains where information quality directly impacts user well-being.
>
---
#### [new 029] Assessing LLMs for Serendipity Discovery in Knowledge Graphs: A Case for Drug Repurposing
- **分类: cs.CL; cs.AI**

- **简介: 论文提出Serendipity-aware KGQA任务，旨在评估LLMs在知识图谱中发现意外但有价值的见解（如药物重定位）的能力。构建了包含新颖性、相关性和惊喜度的指标与专家标注基准，设计三阶段评估流程，揭示当前LLMs在生成惊喜发现方面仍有不足。**

- **链接: [https://arxiv.org/pdf/2511.12472v1](https://arxiv.org/pdf/2511.12472v1)**

> **作者:** Mengying Wang; Chenhui Ma; Ao Jiao; Tuo Liang; Pengjun Lu; Shrinidhi Hegde; Yu Yin; Evren Gurkan-Cavusoglu; Yinghui Wu
>
> **备注:** The 40th AAAI Conference on Artificial Intelligence (AAAI-26)
>
> **摘要:** Large Language Models (LLMs) have greatly advanced knowledge graph question answering (KGQA), yet existing systems are typically optimized for returning highly relevant but predictable answers. A missing yet desired capacity is to exploit LLMs to suggest surprise and novel ("serendipitious") answers. In this paper, we formally define the serendipity-aware KGQA task and propose the SerenQA framework to evaluate LLMs' ability to uncover unexpected insights in scientific KGQA tasks. SerenQA includes a rigorous serendipity metric based on relevance, novelty, and surprise, along with an expert-annotated benchmark derived from the Clinical Knowledge Graph, focused on drug repurposing. Additionally, it features a structured evaluation pipeline encompassing three subtasks: knowledge retrieval, subgraph reasoning, and serendipity exploration. Our experiments reveal that while state-of-the-art LLMs perform well on retrieval, they still struggle to identify genuinely surprising and valuable discoveries, underscoring a significant room for future improvements. Our curated resources and extended version are released at: https://cwru-db-group.github.io/serenQA.
>
---
#### [new 030] Mem-PAL: Towards Memory-based Personalized Dialogue Assistants for Long-term User-Agent Interaction
- **分类: cs.CL**

- **简介: 论文提出Mem-PAL框架，解决长期交互中个性化对话助手的不足。构建PAL-Bench基准与PAL-Set数据集，并设计H²Memory记忆机制提升响应个性化效果。**

- **链接: [https://arxiv.org/pdf/2511.13410v1](https://arxiv.org/pdf/2511.13410v1)**

> **作者:** Zhaopei Huang; Qifeng Dai; Guozheng Wu; Xiaopeng Wu; Kehan Chen; Chuan Yu; Xubin Li; Tiezheng Ge; Wenxuan Wang; Qin Jin
>
> **备注:** Accepted by AAAI 2026 (Oral)
>
> **摘要:** With the rise of smart personal devices, service-oriented human-agent interactions have become increasingly prevalent. This trend highlights the need for personalized dialogue assistants that can understand user-specific traits to accurately interpret requirements and tailor responses to individual preferences. However, existing approaches often overlook the complexities of long-term interactions and fail to capture users' subjective characteristics. To address these gaps, we present PAL-Bench, a new benchmark designed to evaluate the personalization capabilities of service-oriented assistants in long-term user-agent interactions. In the absence of available real-world data, we develop a multi-step LLM-based synthesis pipeline, which is further verified and refined by human annotators. This process yields PAL-Set, the first Chinese dataset comprising multi-session user logs and dialogue histories, which serves as the foundation for PAL-Bench. Furthermore, to improve personalized service-oriented interactions, we propose H$^2$Memory, a hierarchical and heterogeneous memory framework that incorporates retrieval-augmented generation to improve personalized response generation. Comprehensive experiments on both our PAL-Bench and an external dataset demonstrate the effectiveness of the proposed memory framework.
>
---
#### [new 031] TimeStampEval: A Simple LLM Eval and a Little Fuzzy Matching Trick to Improve Search Accuracy
- **分类: cs.CL; cs.AI**

- **简介: 论文提出TimeStampEval任务，解决非精确匹配下从长语音转录文本中精准定位时间戳的问题。通过两阶段方法：Prompt优化与Assisted Fuzzy策略，显著提升准确率并大幅降低计算成本。**

- **链接: [https://arxiv.org/pdf/2511.11594v1](https://arxiv.org/pdf/2511.11594v1)**

> **作者:** James McCammon
>
> **摘要:** Traditional fuzzy matching often fails when searching for quotes that are semantically identical but syntactically different across documents-a common issue when aligning official written records with speech-to-text transcripts. We introduce TimeStampEval, a benchmark for retrieving precise millisecond timestamps from long transcripts given non-verbatim quotes. Our simple two-stage method dramatically improves retrieval accuracy while cutting inference costs by over 90%. The motivating use case is an automated long-form podcast that assembles Congressional Record clips into AI-hosted narration. The technical challenge: given a sentence-timestamped transcript and a target quote that may differ due to transcription or editorial drift, return exact start and end boundaries. Standard algorithms handle verbatim text but break under fuzzier variants. Evaluating six modern LLMs on a 2,800-sentence (120k-token) transcript revealed four key findings. (1) Prompt design matters more than model choice: placing the query before the transcript and using compact formatting improved accuracy by 3-20 points while reducing token count by 30-40%. (2) Off-by-one errors form a distinct category, showing models understand the task but misplace boundaries. (3) A modest reasoning budget (600-850 tokens) raises accuracy from 37% to 77% for weak setups and to above 90% for strong ones. (4) Our "Assisted Fuzzy" approach-RapidFuzz pre-filtering followed by LLM verification on short snippets-improves fuzzy match accuracy by up to 50 points while halving latency and reducing cost per correct result by up to 96%. Extended tests on ten transcripts (50k-900k tokens, 1989-2025) confirm robustness to transcript length, vocabulary drift, and domain change, maintaining 95-100% rejection accuracy for absent targets.
>
---
#### [new 032] How Good is BLI as an Alignment Measure: A Study in Word Embedding Paradigm
- **分类: cs.CL**

- **简介: 论文研究BLI（双语词典诱导）作为嵌入对齐度量的有效性，探讨其在单语与多语嵌入模型中的适用性，提出基于词干的新BLI方法和词汇修剪技术，以更准确评估不同语言家族下的对齐效果。**

- **链接: [https://arxiv.org/pdf/2511.13040v1](https://arxiv.org/pdf/2511.13040v1)**

> **作者:** Kasun Wickramasinghe; Nisansa de Silva
>
> **备注:** 15 pages, 2 figures, 6 tables
>
> **摘要:** Sans a dwindling number of monolingual embedding studies originating predominantly from the low-resource domains, it is evident that multilingual embedding has become the de facto choice due to its adaptability to the usage of code-mixed languages, granting the ability to process multilingual documents in a language-agnostic manner, as well as removing the difficult task of aligning monolingual embeddings. But is this victory complete? Are the multilingual models better than aligned monolingual models in every aspect? Can the higher computational cost of multilingual models always be justified? Or is there a compromise between the two extremes? Bilingual Lexicon Induction is one of the most widely used metrics in terms of evaluating the degree of alignment between two embedding spaces. In this study, we explore the strengths and limitations of BLI as a measure to evaluate the degree of alignment of two embedding spaces. Further, we evaluate how well traditional embedding alignment techniques, novel multilingual models, and combined alignment techniques perform BLI tasks in the contexts of both high-resource and low-resource languages. In addition to that, we investigate the impact of the language families to which the pairs of languages belong. We identify that BLI does not measure the true degree of alignment in some cases and we propose solutions for them. We propose a novel stem-based BLI approach to evaluate two aligned embedding spaces that take into account the inflected nature of languages as opposed to the prevalent word-based BLI techniques. Further, we introduce a vocabulary pruning technique that is more informative in showing the degree of the alignment, especially performing BLI on multilingual embedding models. Often, combined embedding alignment techniques perform better while in certain cases multilingual embeddings perform better (mainly low-resource language cases).
>
---
#### [new 033] Zero-Shot Grammar Competency Estimation Using Large Language Model Generated Pseudo Labels
- **分类: cs.CL**

- **简介: 论文提出一种零样本语法能力评估框架，利用大语言模型生成伪标签，训练Transformer模型在无专家标注情况下准确估计语法水平，解决口语语法评分数据稀缺与标注成本高的问题。**

- **链接: [https://arxiv.org/pdf/2511.13152v1](https://arxiv.org/pdf/2511.13152v1)**

> **作者:** Sourya Dipta Das; Shubham Kumar; Kuldeep Yadav
>
> **备注:** Accepted in AACL-IJCNLP 2025
>
> **摘要:** Grammar competency estimation is essential for assessing linguistic proficiency in both written and spoken language; however, the spoken modality presents additional challenges due to its spontaneous, unstructured, and disfluent nature. Developing accurate grammar scoring models further requires extensive expert annotation, making large-scale data creation impractical. To address these limitations, we propose a zero-shot grammar competency estimation framework that leverages unlabeled data and Large Language Models (LLMs) without relying on manual labels. During training, we employ LLM-generated predictions on unlabeled data by using grammar competency rubric-based prompts. These predictions, treated as pseudo labels, are utilized to train a transformer-based model through a novel training framework designed to handle label noise effectively. We show that the choice of LLM for pseudo-label generation critically affects model performance and that the ratio of clean-to-noisy samples during training strongly influences stability and accuracy. Finally, a qualitative analysis of error intensity and score prediction confirms the robustness and interpretability of our approach. Experimental results demonstrate the efficacy of our approach in estimating grammar competency scores with high accuracy, paving the way for scalable, low-resource grammar assessment systems.
>
---
#### [new 034] Classification of Hope in Textual Data using Transformer-Based Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于情感分类任务，旨在用Transformer模型识别文本中的希望表达。作者比较了BERT、GPT-2和DeBERTa在二分类和多分类上的表现，发现BERT在准确率和效率上最优，且各模型对不同语境有独特优势。**

- **链接: [https://arxiv.org/pdf/2511.12874v1](https://arxiv.org/pdf/2511.12874v1)**

> **作者:** Chukwuebuka Fortunate Ijezue; Tania-Amanda Fredrick Eneye; Maaz Amjad
>
> **摘要:** This paper presents a transformer-based approach for classifying hope expressions in text. We developed and compared three architectures (BERT, GPT-2, and DeBERTa) for both binary classification (Hope vs. Not Hope) and multiclass categorization (five hope-related categories). Our initial BERT implementation achieved 83.65% binary and 74.87% multiclass accuracy. In the extended comparison, BERT demonstrated superior performance (84.49% binary, 72.03% multiclass accuracy) while requiring significantly fewer computational resources (443s vs. 704s training time) than newer architectures. GPT-2 showed lowest overall accuracy (79.34% binary, 71.29% multiclass), while DeBERTa achieved moderate results (80.70% binary, 71.56% multiclass) but at substantially higher computational cost (947s for multiclass training). Error analysis revealed architecture-specific strengths in detecting nuanced hope expressions, with GPT-2 excelling at sarcasm detection (92.46% recall). This study provides a framework for computational analysis of hope, with applications in mental health and social media analysis, while demonstrating that architectural suitability may outweigh model size for specialized emotion detection tasks.
>
---
#### [new 035] MMWOZ: Building Multimodal Agent for Task-oriented Dialogue
- **分类: cs.CL**

- **简介: 论文提出MMWOZ数据集和MATE模型，解决传统任务导向对话系统在无定制API场景下难以应用的问题。通过构建GUI界面并自动化转换对话状态为操作指令，实现多模态交互，提升实际可用性。**

- **链接: [https://arxiv.org/pdf/2511.12586v1](https://arxiv.org/pdf/2511.12586v1)**

> **作者:** Pu-Hai Yang; Heyan Huang; Heng-Da Xu; Fanshu Sun; Xian-Ling Mao; Chaoxu Mu
>
> **摘要:** Task-oriented dialogue systems have garnered significant attention due to their conversational ability to accomplish goals, such as booking airline tickets for users. Traditionally, task-oriented dialogue systems are conceptualized as intelligent agents that interact with users using natural language and have access to customized back-end APIs. However, in real-world scenarios, the widespread presence of front-end Graphical User Interfaces (GUIs) and the absence of customized back-end APIs create a significant gap for traditional task-oriented dialogue systems in practical applications. In this paper, to bridge the gap, we collect MMWOZ, a new multimodal dialogue dataset that is extended from MultiWOZ 2.3 dataset. Specifically, we begin by developing a web-style GUI to serve as the front-end. Next, we devise an automated script to convert the dialogue states and system actions from the original dataset into operation instructions for the GUI. Lastly, we collect snapshots of the web pages along with their corresponding operation instructions. In addition, we propose a novel multimodal model called MATE (Multimodal Agent for Task-oriEnted dialogue) as the baseline model for the MMWOZ dataset. Furthermore, we conduct comprehensive experimental analysis using MATE to investigate the construction of a practical multimodal agent for task-oriented dialogue.
>
---
#### [new 036] From Phonemes to Meaning: Evaluating Large Language Models on Tamil
- **分类: cs.CL; cs.AI**

- **简介: 论文提出首个针对泰米尔语的语法评估基准ILAKKANAM，涵盖1-13年级试题，评估大语言模型在低资源、形态丰富的语言中的表现。结果表明模型在复杂任务上表现下降，且整体性能与语言能力无强相关。**

- **链接: [https://arxiv.org/pdf/2511.12387v1](https://arxiv.org/pdf/2511.12387v1)**

> **作者:** Jeyarajalingam Varsha; Menan Velayuthan; Sumirtha Karunakaran; Rasan Nivethiga; Kengatharaiyer Sarveswaran
>
> **备注:** 11 pages
>
> **摘要:** Large Language Models (LLMs) have shown strong generalization across tasks in high-resource languages; however, their linguistic competence in low-resource and morphologically rich languages such as Tamil remains largely unexplored. Existing multilingual benchmarks often rely on translated English datasets, failing to capture the linguistic and cultural nuances of the target language. To address this gap, we introduce ILAKKANAM, the first Tamil-specific linguistic evaluation benchmark manually curated using 820 questions from Sri Lankan school-level Tamil subject examination papers. Each question is annotated by trained linguists under five linguistic categories and a factual knowledge category, spanning Grades 1--13 to ensure broad linguistic coverage. We evaluate both closed-source and open-source LLMs using a standardized evaluation framework. Our results show that Gemini 2.5 achieves the highest overall performance, while open-source models lag behind, highlighting the gap in linguistic grounding. Category- and grade-wise analyses reveal that all models perform well on lower-grade questions but show a clear decline as linguistic complexity increases. Further, no strong correlation is observed between a model's overall performance and its ability to identify linguistic categories, suggesting that performance may be driven by exposure rather than genuine understanding.
>
---
#### [new 037] Do LLMs and Humans Find the Same Questions Difficult? A Case Study on Japanese Quiz Answering
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的问答任务，旨在探究大语言模型（LLMs）与人类在答题难度上的一致性。研究通过对比LLMs与人类在日语抢答题中的正确率，发现LLMs更难解答未被维基百科覆盖或需数值回答的问题。**

- **链接: [https://arxiv.org/pdf/2511.12300v1](https://arxiv.org/pdf/2511.12300v1)**

> **作者:** Naoya Sugiura; Kosuke Yamada; Yasuhiro Ogawa; Katsuhiko Toyama; Ryohei Sasano
>
> **摘要:** LLMs have achieved performance that surpasses humans in many NLP tasks. However, it remains unclear whether problems that are difficult for humans are also difficult for LLMs. This study investigates how the difficulty of quizzes in a buzzer setting differs between LLMs and humans. Specifically, we first collect Japanese quiz data including questions, answers, and correct response rate of humans, then prompted LLMs to answer the quizzes under several settings, and compare their correct answer rate to that of humans from two analytical perspectives. The experimental results showed that, compared to humans, LLMs struggle more with quizzes whose correct answers are not covered by Wikipedia entries, and also have difficulty with questions that require numerical answers.
>
---
#### [new 038] CriticSearch: Fine-Grained Credit Assignment for Search Agents via a Retrospective Critic
- **分类: cs.CL**

- **简介: 论文提出CriticSearch框架，解决搜索代理训练中奖励稀疏导致的效率低和不稳定问题。通过回顾性批评机制提供细粒度回合级奖励，提升多跳推理任务中的训练速度、稳定性和性能。**

- **链接: [https://arxiv.org/pdf/2511.12159v1](https://arxiv.org/pdf/2511.12159v1)**

> **作者:** Yaocheng Zhang; Haohuan Huang; Zijun Song; Yuanheng Zhu; Qichao Zhang; Zijie Zhao; Dongbin Zhao
>
> **备注:** 17 pages, 10 figures
>
> **摘要:** Tool-Integrated Reasoning (TIR) with search engines enables large language models to iteratively retrieve up-to-date external knowledge, enhancing adaptability and generalization in complex question-answering tasks. However, existing search agent pipelines typically depend on reinforcement learning based optimization, which often suffers from sparse outcome rewards, leading to inefficient exploration and unstable training. We introduce CriticSearch, a fine-grained credit-assignment framework that supplies dense, turn-level feedback via a retrospective critic mechanism. During training, a frozen, asymmetric critique LLM retrospectively evaluates each turn using privileged information from the full trajectory and gold answers, converting these assessments into stable, dense rewards that guide policy improvement. Experimental results across diverse multi-hop reasoning benchmarks demonstrate that CriticSearch consistently outperforms existing baselines, achieving faster convergence, improved training stability, and higher performance.
>
---
#### [new 039] Identifying Imaging Follow-Up in Radiology Reports: A Comparative Analysis of Traditional ML and LLM Approaches
- **分类: cs.CL**

- **简介: 该论文研究放射学报告中随访影像识别任务，旨在提高随访建议检测准确率。作者构建了6393份标注数据集，比较传统机器学习与大语言模型性能，发现优化后的GPT-4o表现最佳，但传统方法仍具实用价值。**

- **链接: [https://arxiv.org/pdf/2511.11867v1](https://arxiv.org/pdf/2511.11867v1)**

> **作者:** Namu Park; Giridhar Kaushik Ramachandran; Kevin Lybarger; Fei Xia; Ozlem Uzuner; Meliha Yetisgen; Martin Gunn
>
> **备注:** Submitted to LREC 2026
>
> **摘要:** Large language models (LLMs) have shown considerable promise in clinical natural language processing, yet few domain-specific datasets exist to rigorously evaluate their performance on radiology tasks. In this work, we introduce an annotated corpus of 6,393 radiology reports from 586 patients, each labeled for follow-up imaging status, to support the development and benchmarking of follow-up adherence detection systems. Using this corpus, we systematically compared traditional machine-learning classifiers, including logistic regression (LR), support vector machines (SVM), Longformer, and a fully fine-tuned Llama3-8B-Instruct, with recent generative LLMs. To evaluate generative LLMs, we tested GPT-4o and the open-source GPT-OSS-20B under two configurations: a baseline (Base) and a task-optimized (Advanced) setting that focused inputs on metadata, recommendation sentences, and their surrounding context. A refined prompt for GPT-OSS-20B further improved reasoning accuracy. Performance was assessed using precision, recall, and F1 scores with 95% confidence intervals estimated via non-parametric bootstrapping. Inter-annotator agreement was high (F1 = 0.846). GPT-4o (Advanced) achieved the best performance (F1 = 0.832), followed closely by GPT-OSS-20B (Advanced; F1 = 0.828). LR and SVM also performed strongly (F1 = 0.776 and 0.775), underscoring that while LLMs approach human-level agreement through prompt optimization, interpretable and resource-efficient models remain valuable baselines.
>
---
#### [new 040] Translation Entropy: A Statistical Framework for Evaluating Translation Systems
- **分类: cs.CL**

- **简介: 论文提出翻译熵概念，用于量化评估机器翻译系统性能。通过分析单个词替换对翻译结果的影响，计算熵值以衡量翻译系统的不确定性和对称性，为翻译系统提供客观基准。**

- **链接: [https://arxiv.org/pdf/2511.13180v1](https://arxiv.org/pdf/2511.13180v1)**

> **作者:** Ronit D. Gross; Yanir Harel; Ido Kanter
>
> **备注:** 23 pages, 6 figures and 8 tables
>
> **摘要:** The translation of written language has been known since the 3rd century BC; however, its necessity has become increasingly common in the information age. Today, many translators exist, based on encoder-decoder deep architectures, nevertheless, no quantitative objective methods are available to assess their performance, likely because the entropy of even a single language remains unknown. This study presents a quantitative method for estimating translation entropy, with the following key finding. Given a translator, several sentences that differ by only one selected token of a given pivot sentence yield identical translations. Analyzing the statistics of this phenomenon across an ensemble of such sentences, consisting each of a pivot selected token, yields the probabilities of replacing this specific token with others while preserving the translation. These probabilities constitute the entropy of the selected token, and the average across all selected pivot tokens provides an estimate of the translator's overall translation entropy, which is enhanced along the decoder blocks. This entropic measure allows for the quantitative ranking of several publicly available translators and reveals whether mutual translation entropy is symmetric. Extending the proposed method to include the replacement of two tokens in a given pivot sentence demonstrates a multiplicative effect, where translation degeneracy is proportional to the product of the degeneracies of the two tokens. These findings establish translation entropy as a measurable property and objective benchmarking of artificial translators. Results are based on MarianMT, T5-Base and NLLB-200 translators.
>
---
#### [new 041] Omni Memory System for Personalized, Long Horizon, Self-Evolving Agents
- **分类: cs.CL**

- **简介: 该论文提出O-Mem框架，解决LLM代理在长期交互中缺乏上下文一致性和动态个性化的问题。通过主动用户画像实现层次化记忆检索，提升响应准确性与效率，在LoCoMo和PERSONAMEM基准上优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.13593v1](https://arxiv.org/pdf/2511.13593v1)**

> **作者:** Piaohong Wang; Motong Tian; Jiaxian Li; Yuan Liang; Yuqing Wang; Qianben Chen; Tiannan Wang; Zhicong Lu; Jiawei Ma; Yuchen Eleanor Jiang; Wangchunshu Zhou
>
> **摘要:** Recent advancements in LLM-powered agents have demonstrated significant potential in generating human-like responses; however, they continue to face challenges in maintaining long-term interactions within complex environments, primarily due to limitations in contextual consistency and dynamic personalization. Existing memory systems often depend on semantic grouping prior to retrieval, which can overlook semantically irrelevant yet critical user information and introduce retrieval noise. In this report, we propose the initial design of O-Mem, a novel memory framework based on active user profiling that dynamically extracts and updates user characteristics and event records from their proactive interactions with agents. O-Mem supports hierarchical retrieval of persona attributes and topic-related context, enabling more adaptive and coherent personalized responses. O-Mem achieves 51.76% on the public LoCoMo benchmark, a nearly 3% improvement upon LangMem,the previous state-of-the-art, and it achieves 62.99% on PERSONAMEM, a 3.5% improvement upon A-Mem,the previous state-of-the-art. O-Mem also boosts token and interaction response time efficiency compared to previous memory frameworks. Our work opens up promising directions for developing efficient and human-like personalized AI assistants in the future.
>
---
#### [new 042] InData: Towards Secure Multi-Step, Tool-Based Data Analysis
- **分类: cs.CL; cs.LG**

- **简介: 论文提出InData数据集，用于评估大语言模型在安全环境下进行多步工具调用的数据分析能力。针对现有方法直接访问数据库带来的安全风险，该研究限制模型仅通过预定义工具交互数据，填补了复杂推理任务的评测空白。**

- **链接: [https://arxiv.org/pdf/2511.11933v1](https://arxiv.org/pdf/2511.11933v1)**

> **作者:** Karthikeyan K; Raghuveer Thirukovalluru; Bhuwan Dhingra; David Edwin Carlson
>
> **摘要:** Large language model agents for data analysis typically generate and execute code directly on databases. However, when applied to sensitive data, this approach poses significant security risks. To address this issue, we propose a security-motivated alternative: restrict LLMs from direct code generation and data access, and require them to interact with data exclusively through a predefined set of secure, verified tools. Although recent tool-use benchmarks exist, they primarily target tool selection and simple execution rather than the compositional, multi-step reasoning needed for complex data analysis. To reduce this gap, we introduce Indirect Data Engagement (InData), a dataset designed to assess LLMs' multi-step tool-based reasoning ability. InData includes data analysis questions at three difficulty levels--Easy, Medium, and Hard--capturing increasing reasoning complexity. We benchmark 15 open-source LLMs on InData and find that while large models (e.g., gpt-oss-120b) achieve high accuracy on Easy tasks (97.3%), performance drops sharply on Hard tasks (69.6%). These results show that current LLMs still lack robust multi-step tool-based reasoning ability. With InData, we take a step toward enabling the development and evaluation of LLMs with stronger multi-step tool-use capabilities. We will publicly release the dataset and code.
>
---
#### [new 043] Scaling Open-Weight Large Language Models for Hydropower Regulatory Information Extraction: A Systematic Analysis
- **分类: cs.CL; cs.AI**

- **简介: 论文研究开放权重大语言模型在水电监管信息抽取任务中的性能与资源权衡。通过评估0.6B-70B参数模型，发现14B是性能跃迁阈值，提出基于验证的部署策略，解决小模型效果差和大模型资源需求高的问题。**

- **链接: [https://arxiv.org/pdf/2511.11821v1](https://arxiv.org/pdf/2511.11821v1)**

> **作者:** Hong-Jun Yoon; Faisal Ashraf; Thomas A. Ruggles; Debjani Singh
>
> **备注:** 18 pages, zero figures, Preprint submitted to Environmental Modeling and Software
>
> **摘要:** Information extraction from regulatory documents using large language models presents critical trade-offs between performance and computational resources. We evaluated seven open-weight models (0.6B-70B parameters) on hydropower licensing documentation to provide empirical deployment guidance. Our analysis identified a pronounced 14B parameter threshold where validation methods transition from ineffective (F1 $<$ 0.15) to viable (F1 = 0.64). Consumer-deployable models achieve 64\% F1 through appropriate validation, while smaller models plateau at 51\%. Large-scale models approach 77\% F1 but require enterprise infrastructure. We identified systematic hallucination patterns where perfect recall indicates extraction failure rather than success in smaller models. Our findings establish the first comprehensive resource-performance mapping for open-weight information extraction in regulatory contexts, enabling evidence-based model selection. These results provide immediate value for hydropower compliance while contributing insights into parameter scaling effects that generalize across information extraction tasks.
>
---
#### [new 044] Exploring Parameter-Efficient Fine-Tuning and Backtranslation for the WMT 25 General Translation Task
- **分类: cs.CL**

- **简介: 该论文研究低资源语言对（英→日）的神经机器翻译任务，旨在提升小语料库下的翻译质量。通过结合回译与微调技术，利用合成数据增强和真实平行语料微调，显著提升模型性能，验证了两者协同作用的有效性。**

- **链接: [https://arxiv.org/pdf/2511.12109v1](https://arxiv.org/pdf/2511.12109v1)**

> **作者:** Felipe Fujita; Hideyuki Takada
>
> **摘要:** In this paper, we explore the effectiveness of combining fine-tuning and backtranslation on a small Japanese corpus for neural machine translation. Starting from a baseline English{\textrightarrow}Japanese model (COMET = 0.460), we first apply backtranslation (BT) using synthetic data generated from monolingual Japanese corpora, yielding a modest increase (COMET = 0.468). Next, we fine-tune (FT) the model on a genuine small parallel dataset drawn from diverse Japanese news and literary corpora, achieving a substantial jump to COMET = 0.589 when using Mistral 7B. Finally, we integrate both backtranslation and fine-tuning{ -- }first augmenting the small dataset with BT generated examples, then adapting via FT{ -- }which further boosts performance to COMET = 0.597. These results demonstrate that, even with limited training data, the synergistic use of backtranslation and targeted fine-tuning on Japanese corpora can significantly enhance translation quality, outperforming each technique in isolation. This approach offers a lightweight yet powerful strategy for improving low-resource language pairs.
>
---
#### [new 045] Applying Large Language Models to Characterize Public Narratives
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的文本分析任务，旨在解决公共叙事（PN）系统性分析难、专家标注成本高的问题。作者提出基于大语言模型（LLM）的自动化标注框架，实现近人类专家水平的标注性能，并扩展至更大规模叙事数据与政治演讲分析。**

- **链接: [https://arxiv.org/pdf/2511.13505v1](https://arxiv.org/pdf/2511.13505v1)**

> **作者:** Elinor Poole-Dayan; Daniel T Kessler; Hannah Chiou; Margaret Hughes; Emily S Lin; Marshall Ganz; Deb Roy
>
> **摘要:** Public Narratives (PNs) are key tools for leadership development and civic mobilization, yet their systematic analysis remains challenging due to their subjective interpretation and the high cost of expert annotation. In this work, we propose a novel computational framework that leverages large language models (LLMs) to automate the qualitative annotation of public narratives. Using a codebook we co-developed with subject-matter experts, we evaluate LLM performance against that of expert annotators. Our work reveals that LLMs can achieve near-human-expert performance, achieving an average F1 score of 0.80 across 8 narratives and 14 codes. We then extend our analysis to empirically explore how PN framework elements manifest across a larger dataset of 22 stories. Lastly, we extrapolate our analysis to a set of political speeches, establishing a novel lens in which to analyze political rhetoric in civic spaces. This study demonstrates the potential of LLM-assisted annotation for scalable narrative analysis and highlights key limitations and directions for future research in computational civic storytelling.
>
---
#### [new 046] Generalist Foundation Models Are Not Clinical Enough for Hospital Operations
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文研究医疗领域大模型在医院运营中的应用，解决通用模型缺乏临床操作知识的问题。通过构建专有预训练模型Lang1和真实世界评估基准ReMedE，证明专用模型经微调后显著优于通用模型，强调了领域内预训练与监督微调的重要性。**

- **链接: [https://arxiv.org/pdf/2511.13703v1](https://arxiv.org/pdf/2511.13703v1)**

> **作者:** Lavender Y. Jiang; Angelica Chen; Xu Han; Xujin Chris Liu; Radhika Dua; Kevin Eaton; Frederick Wolff; Robert Steele; Jeff Zhang; Anton Alyakin; Qingkai Pan; Yanbing Chen; Karl L. Sangwon; Daniel A. Alber; Jaden Stryker; Jin Vivian Lee; Yindalon Aphinyanaphongs; Kyunghyun Cho; Eric Karl Oermann
>
> **摘要:** Hospitals and healthcare systems rely on operational decisions that determine patient flow, cost, and quality of care. Despite strong performance on medical knowledge and conversational benchmarks, foundation models trained on general text may lack the specialized knowledge required for these operational decisions. We introduce Lang1, a family of models (100M-7B parameters) pretrained on a specialized corpus blending 80B clinical tokens from NYU Langone Health's EHRs and 627B tokens from the internet. To rigorously evaluate Lang1 in real-world settings, we developed the REalistic Medical Evaluation (ReMedE), a benchmark derived from 668,331 EHR notes that evaluates five critical tasks: 30-day readmission prediction, 30-day mortality prediction, length of stay, comorbidity coding, and predicting insurance claims denial. In zero-shot settings, both general-purpose and specialized models underperform on four of five tasks (36.6%-71.7% AUROC), with mortality prediction being an exception. After finetuning, Lang1-1B outperforms finetuned generalist models up to 70x larger and zero-shot models up to 671x larger, improving AUROC by 3.64%-6.75% and 1.66%-23.66% respectively. We also observed cross-task scaling with joint finetuning on multiple tasks leading to improvement on other tasks. Lang1-1B effectively transfers to out-of-distribution settings, including other clinical tasks and an external health system. Our findings suggest that predictive capabilities for hospital operations require explicit supervised finetuning, and that this finetuning process is made more efficient by in-domain pretraining on EHR. Our findings support the emerging view that specialized LLMs can compete with generalist models in specialized tasks, and show that effective healthcare systems AI requires the combination of in-domain pretraining, supervised finetuning, and real-world evaluation beyond proxy benchmarks.
>
---
#### [new 047] Critical or Compliant? The Double-Edged Sword of Reasoning in Chain-of-Thought Explanations
- **分类: cs.CL; cs.HC**

- **简介: 论文研究Chain-of-Thought解释在多模态道德场景中的双刃剑作用，揭示用户易因结果一致或自信语气而误信错误推理。通过扰动推理链和调整语气，发现解释可能误导而非澄清，强调需设计促进批判性思维的NLP解释机制。**

- **链接: [https://arxiv.org/pdf/2511.12001v1](https://arxiv.org/pdf/2511.12001v1)**

> **作者:** Eunkyu Park; Wesley Hanwen Deng; Vasudha Varadarajan; Mingxi Yan; Gunhee Kim; Maarten Sap; Motahhare Eslami
>
> **备注:** Under review; 16 pages, 15 figures
>
> **摘要:** Explanations are often promoted as tools for transparency, but they can also foster confirmation bias; users may assume reasoning is correct whenever outputs appear acceptable. We study this double-edged role of Chain-of-Thought (CoT) explanations in multimodal moral scenarios by systematically perturbing reasoning chains and manipulating delivery tones. Specifically, we analyze reasoning errors in vision language models (VLMs) and how they impact user trust and the ability to detect errors. Our findings reveal two key effects: (1) users often equate trust with outcome agreement, sustaining reliance even when reasoning is flawed, and (2) the confident tone suppresses error detection while maintaining reliance, showing that delivery styles can override correctness. These results highlight how CoT explanations can simultaneously clarify and mislead, underscoring the need for NLP systems to provide explanations that encourage scrutiny and critical thinking rather than blind trust. All code will be released publicly.
>
---
#### [new 048] MiroThinker: Pushing the Performance Boundaries of Open-Source Research Agents via Model, Context, and Interactive Scaling
- **分类: cs.CL**

- **简介: 论文提出MiroThinker，一个通过模型、上下文和交互三方面扩展提升性能的开源研究代理。解决传统代理仅依赖模型或上下文扩展的问题，通过强化学习实现高效交互扩展，显著提升复杂任务准确率，验证了交互深度是性能提升的新维度。**

- **链接: [https://arxiv.org/pdf/2511.11793v1](https://arxiv.org/pdf/2511.11793v1)**

> **作者:** MiroMind Team; Song Bai; Lidong Bing; Carson Chen; Guanzheng Chen; Yuntao Chen; Zhe Chen; Ziyi Chen; Jifeng Dai; Xuan Dong; Yue Deng; Yunjie Fu; Junqi Ge; Chenxia Han; Tammy Huang; Zhenhang Huang; Jerry Jiao; Shilei Jiang; Tianyu Jiao; Xiaoqi Jian; Lei Lei; Ruilin Li; Ryan Luo; Tiantong Li; Xiang Lin; Ziyuan Liu; Zhiqi Li; Jie Ni; Qiang Ren; Pax Sun; Shiqian Su; Chenxin Tao; Bin Wang; Hellen Wang; Haonan Wang; James Wang; Jin Wang; Jojo Wang; Letian Wang; Shizun Wang; Weizhi Wang; Zixuan Wang; Jinfan Xu; Sen Xing; Chenyu Yang; Hai Ye; Jiaheng Yu; Yue Yu; Muyan Zhong; Tianchen Zhao; Xizhou Zhu; Yanpeng Zhou; Yifan Zhang; Zhi Zhu
>
> **备注:** Technical Report
>
> **摘要:** We present MiroThinker v1.0, an open-source research agent designed to advance tool-augmented reasoning and information-seeking capabilities. Unlike previous agents that only scale up model size or context length, MiroThinker explores interaction scaling at the model level, systematically training the model to handle deeper and more frequent agent-environment interactions as a third dimension of performance improvement. Unlike LLM test-time scaling, which operates in isolation and risks degradation with longer reasoning chains, interactive scaling leverages environment feedback and external information acquisition to correct errors and refine trajectories. Through reinforcement learning, the model achieves efficient interaction scaling: with a 256K context window, it can perform up to 600 tool calls per task, enabling sustained multi-turn reasoning and complex real-world research workflows. Across four representative benchmarks-GAIA, HLE, BrowseComp, and BrowseComp-ZH-the 72B variant achieves up to 81.9%, 37.7%, 47.1%, and 55.6% accuracy respectively, surpassing previous open-source agents and approaching commercial counterparts such as GPT-5-high. Our analysis reveals that MiroThinker benefits from interactive scaling consistently: research performance improves predictably as the model engages in deeper and more frequent agent-environment interactions, demonstrating that interaction depth exhibits scaling behaviors analogous to model size and context length. These findings establish interaction scaling as a third critical dimension for building next-generation open research agents, complementing model capacity and context windows.
>
---
#### [new 049] AHaSIS: Shared Task on Sentiment Analysis for Arabic Dialects
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于阿拉伯语方言情感分析任务，旨在解决酒店领域中多 dialect 情感检测难题。工作包括构建538条平衡标注的沙特与摩洛哥方言数据集，并通过40余支团队参与评估，最高F1达0.81。**

- **链接: [https://arxiv.org/pdf/2511.13335v1](https://arxiv.org/pdf/2511.13335v1)**

> **作者:** Maram Alharbi; Salmane Chafik; Saad Ezzini; Ruslan Mitkov; Tharindu Ranasinghe; Hansi Hettiarachchi
>
> **摘要:** The hospitality industry in the Arab world increasingly relies on customer feedback to shape services, driving the need for advanced Arabic sentiment analysis tools. To address this challenge, the Sentiment Analysis on Arabic Dialects in the Hospitality Domain shared task focuses on Sentiment Detection in Arabic Dialects. This task leverages a multi-dialect, manually curated dataset derived from hotel reviews originally written in Modern Standard Arabic (MSA) and translated into Saudi and Moroccan (Darija) dialects. The dataset consists of 538 sentiment-balanced reviews spanning positive, neutral, and negative categories. Translations were validated by native speakers to ensure dialectal accuracy and sentiment preservation. This resource supports the development of dialect-aware NLP systems for real-world applications in customer experience analysis. More than 40 teams have registered for the shared task, with 12 submitting systems during the evaluation phase. The top-performing system achieved an F1 score of 0.81, demonstrating the feasibility and ongoing challenges of sentiment analysis across Arabic dialects.
>
---
#### [new 050] Probing Preference Representations: A Multi-Dimensional Evaluation and Analysis Method for Reward Models
- **分类: cs.CL**

- **简介: 论文针对奖励模型评估缺乏多维偏好分析的问题，提出MRMBench基准和推理时探针方法，通过六项维度任务提升评估可靠性与可解释性，助力大语言模型对齐优化。**

- **链接: [https://arxiv.org/pdf/2511.12464v1](https://arxiv.org/pdf/2511.12464v1)**

> **作者:** Chenglong Wang; Yifu Huo; Yang Gan; Yongyu Mu; Qiaozhi He; Murun Yang; Bei Li; Chunliang Zhang; Tongran Liu; Anxiang Ma; Zhengtao Yu; Jingbo Zhu; Tong Xiao
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Previous methods evaluate reward models by testing them on a fixed pairwise ranking test set, but they typically do not provide performance information on each preference dimension. In this work, we address the evaluation challenge of reward models by probing preference representations. To confirm the effectiveness of this evaluation method, we construct a Multi-dimensional Reward Model Benchmark (MRMBench), a collection of six probing tasks for different preference dimensions. We design it to favor and encourage reward models that better capture preferences across different dimensions. Furthermore, we introduce an analysis method, inference-time probing, which identifies the dimensions used during the reward prediction and enhances its interpretability. Through extensive experiments, we find that MRMBench strongly correlates with the alignment performance of large language models (LLMs), making it a reliable reference for developing advanced reward models. Our analysis of MRMBench evaluation results reveals that reward models often struggle to capture preferences across multiple dimensions, highlighting the potential of multi-objective optimization in reward modeling. Additionally, our findings show that the proposed inference-time probing method offers a reliable metric for assessing the confidence of reward predictions, which ultimately improves the alignment of LLMs.
>
---
#### [new 051] Evaluating Large Language Models for Diacritic Restoration in Romanian Texts: A Comparative Study
- **分类: cs.CL**

- **简介: 该论文研究罗马尼亚语中自动恢复变音符号的任务，旨在提升对含丰富变音符号语言的文本处理能力。通过对比多种大语言模型在不同提示模板下的表现，发现GPT-4o效果最佳，验证了模型架构与提示设计的重要性。**

- **链接: [https://arxiv.org/pdf/2511.13182v1](https://arxiv.org/pdf/2511.13182v1)**

> **作者:** Mihai Dan Nadas; Laura Diosan
>
> **摘要:** Automatic diacritic restoration is crucial for text processing in languages with rich diacritical marks, such as Romanian. This study evaluates the performance of several large language models (LLMs) in restoring diacritics in Romanian texts. Using a comprehensive corpus, we tested models including OpenAI's GPT-3.5, GPT-4, GPT-4o, Google's Gemini 1.0 Pro, Meta's Llama 2 and Llama 3, MistralAI's Mixtral 8x7B Instruct, airoboros 70B, and OpenLLM-Ro's RoLlama 2 7B, under multiple prompt templates ranging from zero-shot to complex multi-shot instructions. Results show that models such as GPT-4o achieve high diacritic restoration accuracy, consistently surpassing a neutral echo baseline, while others, including Meta's Llama family, exhibit wider variability. These findings highlight the impact of model architecture, training data, and prompt design on diacritic restoration performance and outline promising directions for improving NLP tools for diacritic-rich languages.
>
---
#### [new 052] Non-Linear Scoring Model for Translation Quality Evaluation
- **分类: cs.CL**

- **简介: 论文提出非线性评分模型解决翻译质量评估中因样本长度差异导致的偏差问题，基于心理物理规律建模误差容忍度，提升评估公平性与准确性。**

- **链接: [https://arxiv.org/pdf/2511.13467v1](https://arxiv.org/pdf/2511.13467v1)**

> **作者:** Serge Gladkoff; Lifeng Han; Katerina Gasova
>
> **备注:** ongoing work, 38 pages
>
> **摘要:** Analytic Translation Quality Evaluation (TQE), based on Multidimensional Quality Metrics (MQM), traditionally uses a linear error-to-penalty scale calibrated to a reference sample of 1000-2000 words. However, linear extrapolation biases judgment on samples of different sizes, over-penalizing short samples and under-penalizing long ones, producing misalignment with expert intuition. Building on the Multi-Range framework, this paper presents a calibrated, non-linear scoring model that better reflects how human content consumers perceive translation quality across samples of varying length. Empirical data from three large-scale enterprise environments shows that acceptable error counts grow logarithmically, not linearly, with sample size. Psychophysical and cognitive evidence, including the Weber-Fechner law and Cognitive Load Theory, supports this premise by explaining why the perceptual impact of additional errors diminishes while the cognitive burden grows with scale. We propose a two-parameter model E(x) = a * ln(1 + b * x), a, b > 0, anchored to a reference tolerance and calibrated from two tolerance points using a one-dimensional root-finding step. The model yields an explicit interval within which the linear approximation stays within +/-20 percent relative error and integrates into existing evaluation workflows with only a dynamic tolerance function added. The approach improves interpretability, fairness, and inter-rater reliability across both human and AI-generated translations. By operationalizing a perceptually valid scoring paradigm, it advances translation quality evaluation toward more accurate and scalable assessment. The model also provides a stronger basis for AI-based document-level evaluation aligned with human judgment. Implementation considerations for CAT/LQA systems and implications for human and AI-generated text evaluation are discussed.
>
---
#### [new 053] Toward Conversational Hungarian Speech Recognition: Introducing the BEA-Large and BEA-Dialogue Datasets
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 论文聚焦匈牙利语对话语音识别任务，针对其缺乏自发对话语料的问题，构建了BEA-Large和BEA-Dialogue两个新数据集，并提供基线模型与性能指标，推动匈牙利语语音技术发展。**

- **链接: [https://arxiv.org/pdf/2511.13529v1](https://arxiv.org/pdf/2511.13529v1)**

> **作者:** Máté Gedeon; Piroska Zsófia Barta; Péter Mihajlik; Tekla Etelka Gráczi; Anna Kohári; Katalin Mády
>
> **备注:** Submitted to LREC 2026
>
> **摘要:** The advancement of automatic speech recognition (ASR) has been largely enhanced by extensive datasets in high-resource languages, while languages such as Hungarian remain underrepresented due to limited spontaneous and conversational corpora. To address this gap, we introduce two new datasets -- BEA-Large and BEA-Dialogue -- constructed from the previously unprocessed portions of the Hungarian speech corpus named BEA. BEA-Large extends BEA-Base with 255 hours of spontaneous speech from 433 speakers, enriched with detailed segment-level metadata. BEA-Dialogue, comprising 85 hours of spontaneous conversations, is a Hungarian speech corpus featuring natural dialogues partitioned into speaker-independent subsets, supporting research in conversational ASR and speaker diarization. We establish reproducible baselines on these datasets using publicly available ASR models, with the fine-tuned Fast Conformer model achieving word error rates as low as 14.18\% on spontaneous and 4.8\% on repeated speech. Diarization experiments yield diarization error rates between 13.05\% and 18.26\%, providing reference points for future improvements. The results highlight the persistent difficulty of conversational ASR, particularly due to disfluencies, overlaps, and informal speech patterns. By releasing these datasets and baselines, we aim to advance Hungarian speech technology and offer a methodological framework for developing spontaneous and conversational benchmarks in other languages.
>
---
#### [new 054] Distinguishing Repetition Disfluency from Morphological Reduplication in Bangla ASR Transcripts: A Novel Corpus and Benchmarking Analysis
- **分类: cs.CL**

- **简介: 论文解决Bangla语音识别文本中重复词的歧义问题，区分无意重复和语法重叠。构建首个2万条标注语料库，对比大模型与微调方法，验证细粒度语言模型在保留语义前提下正确处理重复现象的有效性。**

- **链接: [https://arxiv.org/pdf/2511.13159v1](https://arxiv.org/pdf/2511.13159v1)**

> **作者:** Zaara Zabeen Arpa; Sadnam Sakib Apurbo; Nazia Karim Khan Oishee; Ajwad Abrar
>
> **摘要:** Automatic Speech Recognition (ASR) transcripts, especially in low-resource languages like Bangla, contain a critical ambiguity: word-word repetitions can be either Repetition Disfluency (unintentional ASR error/hesitation) or Morphological Reduplication (a deliberate grammatical construct). Standard disfluency correction fails by erroneously deleting valid linguistic information. To solve this, we introduce the first publicly available, 20,000-row Bangla corpus, manually annotated to explicitly distinguish between these two phenomena in noisy ASR transcripts. We benchmark this novel resource using two paradigms: state-of-the-art multilingual Large Language Models (LLMs) and task-specific fine-tuning of encoder models. LLMs achieve competitive performance (up to 82.68\% accuracy) with few-shot prompting. However, fine-tuning proves superior, with the language-specific BanglaBERT model achieving the highest accuracy of 84.78\% and an F1 score of 0.677. This establishes a strong, linguistically-informed baseline and provides essential data for developing sophisticated, semantic-preserving text normalization systems for Bangla.
>
---
#### [new 055] Beyond SELECT: A Comprehensive Taxonomy-Guided Benchmark for Real-World Text-to-SQL Translation
- **分类: cs.CL; cs.AI**

- **简介: 论文提出了一种新的文本到SQL任务分类体系，旨在解决现有数据集覆盖不足的问题。通过该分类体系构建了SQL-Synth数据集，并验证其在多样性和复杂性上的优势，同时发现当前大模型在此类场景下表现有限，需通过微调提升性能。**

- **链接: [https://arxiv.org/pdf/2511.13590v1](https://arxiv.org/pdf/2511.13590v1)**

> **作者:** Hao Wang; Yuanfeng Song; Xiaoming Yin; Xing Chen
>
> **摘要:** Text-to-SQL datasets are essential for training and evaluating text-to-SQL models, but existing datasets often suffer from limited coverage and fail to capture the diversity of real-world applications. To address this, we propose a novel taxonomy for text-to-SQL classification based on dimensions including core intents, statement types, syntax structures, and key actions. Using this taxonomy, we evaluate widely used public text-to-SQL datasets (e.g., Spider and Bird) and reveal limitations in their coverage and diversity. We then introduce a taxonomy-guided dataset synthesis pipeline, yielding a new dataset named SQL-Synth. This approach combines the taxonomy with Large Language Models (LLMs) to ensure the dataset reflects the breadth and complexity of real-world text-to-SQL applications. Extensive analysis and experimental results validate the effectiveness of our taxonomy, as SQL-Synth exhibits greater diversity and coverage compared to existing benchmarks. Moreover, we uncover that existing LLMs typically fall short in adequately capturing the full range of scenarios, resulting in limited performance on SQL-Synth. However, fine-tuning can substantially improve their performance in these scenarios. The proposed taxonomy has significant potential impact, as it not only enables comprehensive analysis of datasets and the performance of different LLMs, but also guides the construction of training data for LLMs.
>
---
#### [new 056] Can Large Language Models Function as Qualified Pediatricians? A Systematic Evaluation in Real-World Clinical Contexts
- **分类: cs.CL**

- **简介: 该论文属于医疗AI评估任务，旨在检验大语言模型能否胜任儿科医生角色。通过构建PEDIASBench框架，系统评估12个模型在知识、诊疗动态适应性及伦理安全方面的表现，发现当前模型在复杂推理和人文关怀上存在局限，建议加强多模态融合与临床反馈迭代。**

- **链接: [https://arxiv.org/pdf/2511.13381v1](https://arxiv.org/pdf/2511.13381v1)**

> **作者:** Siyu Zhu; Mouxiao Bian; Yue Xie; Yongyu Tang; Zhikang Yu; Tianbin Li; Pengcheng Chen; Bing Han; Jie Xu; Xiaoyan Dong
>
> **摘要:** With the rapid rise of large language models (LLMs) in medicine, a key question is whether they can function as competent pediatricians in real-world clinical settings. We developed PEDIASBench, a systematic evaluation framework centered on a knowledge-system framework and tailored to realistic clinical environments. PEDIASBench assesses LLMs across three dimensions: application of basic knowledge, dynamic diagnosis and treatment capability, and pediatric medical safety and medical ethics. We evaluated 12 representative models released over the past two years, including GPT-4o, Qwen3-235B-A22B, and DeepSeek-V3, covering 19 pediatric subspecialties and 211 prototypical diseases. State-of-the-art models performed well on foundational knowledge, with Qwen3-235B-A22B achieving over 90% accuracy on licensing-level questions, but performance declined ~15% as task complexity increased, revealing limitations in complex reasoning. Multiple-choice assessments highlighted weaknesses in integrative reasoning and knowledge recall. In dynamic diagnosis and treatment scenarios, DeepSeek-R1 scored highest in case reasoning (mean 0.58), yet most models struggled to adapt to real-time patient changes. On pediatric medical ethics and safety tasks, Qwen2.5-72B performed best (accuracy 92.05%), though humanistic sensitivity remained limited. These findings indicate that pediatric LLMs are constrained by limited dynamic decision-making and underdeveloped humanistic care. Future development should focus on multimodal integration and a clinical feedback-model iteration loop to enhance safety, interpretability, and human-AI collaboration. While current LLMs cannot independently perform pediatric care, they hold promise for decision support, medical education, and patient communication, laying the groundwork for a safe, trustworthy, and collaborative intelligent pediatric healthcare system.
>
---
#### [new 057] Additive Large Language Models for Semi-Structured Text
- **分类: cs.CL; cs.LG**

- **简介: 论文提出CALM框架，用于半结构化临床文本的可解释分类。解决大语言模型预测不透明的问题，通过组件贡献相加实现患者级和群体级解释，提升可信度与可审计性。**

- **链接: [https://arxiv.org/pdf/2511.11922v1](https://arxiv.org/pdf/2511.11922v1)**

> **作者:** Karthikeyan K; Raghuveer Thirukovalluru; David Carlson
>
> **摘要:** Large Language Models have advanced clinical text classification, but their opaque predictions remain a critical barrier to practical adoption in research and clinical settings where investigators and physicians need to understand which parts of a patient's record drive risk signals. To address this challenge, we introduce \textbf{CALM}, short for \textbf{Classification with Additive Large Language Models}, an interpretable framework for semi-structured text where inputs are composed of semantically meaningful components, such as sections of an admission note or question-answer fields from an intake form. CALM predicts outcomes as the additive sum of each component's contribution, making these contributions part of the forward computation itself and enabling faithful explanations at both the patient and population level. The additive structure also enables clear visualizations, such as component-level risk curves similar to those used in generalized additive models, making the learned relationships easier to inspect and communicate. Although CALM expects semi-structured inputs, many clinical documents already have this form, and similar structure can often be automatically extracted from free-text notes. CALM achieves performance comparable to conventional LLM classifiers while improving trust, supporting quality-assurance checks, and revealing clinically meaningful patterns during model development and auditing.
>
---
#### [new 058] Uni-MoE-2.0-Omni: Scaling Language-Centric Omnimodal Large Model with Advanced MoE, Training and Data
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 论文提出Uni-MoE-2.0-Omni，一个开源的多模态大模型，解决语言主导的跨模态理解与生成问题。通过动态MoE架构、渐进式训练策略和数据匹配技术，实现文本、图像、语音等10种模态的高效处理，在85项基准测试中表现优异。**

- **链接: [https://arxiv.org/pdf/2511.12609v1](https://arxiv.org/pdf/2511.12609v1)**

> **作者:** Yunxin Li; Xinyu Chen; Shenyuan Jiang; Haoyuan Shi; Zhenyu Liu; Xuanyu Zhang; Nanhao Deng; Zhenran Xu; Yicheng Ma; Meishan Zhang; Baotian Hu; Min Zhang
>
> **备注:** 47 pages,10 Figures, Project Website: https://idealistxy.github.io/Uni-MoE-v2.github.io/; Codes: https://github.com/HITsz-TMG/Uni-MoE
>
> **摘要:** We present Uni-MoE 2.0 from the Lychee family. As a fully open-source omnimodal large model (OLM), it substantially advances Lychee's Uni-MoE series in language-centric multimodal understanding, reasoning, and generating. Based on the Qwen2.5-7B dense architecture, we build Uni-MoE-2.0-Omni from scratch through three core contributions: dynamic-capacity Mixture-of-Experts (MoE) design, a progressive training strategy enhanced with an iterative reinforcement strategy, and a carefully curated multimodal data matching technique. It is capable of omnimodal understanding, as well as generating images, text, and speech. Architecturally, our new MoE framework balances computational efficiency and capability for 10 cross-modal inputs using shared, routed, and null experts, while our Omni-Modality 3D RoPE ensures spatio-temporal cross-modality alignment in the self-attention layer. For training, following cross-modal pretraining, we use a progressive supervised fine-tuning strategy that activates modality-specific experts and is enhanced by balanced data composition and an iterative GSPO-DPO method to stabilise RL training and improve reasoning. Data-wise, the base model, trained on approximately 75B tokens of open-source multimodal data, is equipped with special speech and image generation tokens, allowing it to learn these generative tasks by conditioning its outputs on linguistic cues. Extensive evaluation across 85 benchmarks demonstrates that our model achieves SOTA or highly competitive performance against leading OLMs, surpassing Qwen2.5-Omni (trained with 1.2T tokens) on over 50 of 76 benchmarks. Key strengths include video understanding (+7% avg. of 8), omnimodallity understanding (+7% avg. of 4), and audiovisual reasoning (+4%). It also advances long-form speech processing (reducing WER by 4.2%) and leads in low-level image processing and controllable generation across 5 metrics.
>
---
#### [new 059] Reason-KE++: Aligning the Process, Not Just the Outcome, for Faithful LLM Knowledge Editing
- **分类: cs.CL**

- **简介: 论文提出Reason-KE++，解决大语言模型在多跳推理任务中因仅对齐结果而非推理过程导致的忠实性问题。通过引入阶段感知奖励机制，强化中间推理步骤的准确性，显著提升事实一致性与最终性能。**

- **链接: [https://arxiv.org/pdf/2511.12661v1](https://arxiv.org/pdf/2511.12661v1)**

> **作者:** Yuchen Wu; Liang Ding; Li Shen; Dacheng Tao
>
> **摘要:** Aligning Large Language Models (LLMs) to be faithful to new knowledge in complex, multi-hop reasoning tasks is a critical, yet unsolved, challenge. We find that SFT-based methods, e.g., Reason-KE, while state-of-the-art, suffer from a "faithfulness gap": they optimize for format mimicry rather than sound reasoning. This gap enables the LLM's powerful parametric priors to override new contextual facts, resulting in critical factual hallucinations (e.g., incorrectly reasoning "Houston" from "NASA" despite an explicit edit). To solve this core LLM alignment problem, we propose Reason-KE++, an SFT+RL framework that instills process-level faithfulness. Its core is a Stage-aware Reward mechanism that provides dense supervision for intermediate reasoning steps (e.g., Decomposition, Sub-answer Correctness). Crucially, we identify that naive outcome-only RL is a deceptive trap for LLM alignment: it collapses reasoning integrity (e.g., 19.00% Hop acc) while superficially boosting final accuracy. Our process-aware framework sets a new SOTA of 95.48% on MQUAKE-CF-3k (+5.28%), demonstrating that for complex tasks, aligning the reasoning process is essential for building trustworthy LLMs.
>
---
#### [new 060] Seeing isn't Hearing: Benchmarking Vision Language Models at Interpreting Spectrograms
- **分类: cs.CL**

- **简介: 该论文研究视觉语言模型（VLMs）解读语音频谱图的能力，属于跨模态理解任务。它通过构建4000+英语单词的配对数据集，测试VLMs在多选任务中识别正确音素或拼写转录的能力，发现模型需专门知识而非仅依赖样本对即可准确解读。**

- **链接: [https://arxiv.org/pdf/2511.13225v1](https://arxiv.org/pdf/2511.13225v1)**

> **作者:** Tyler Loakman; Joseph James; Chenghua Lin
>
> **备注:** Accepted to IJCNLP-AACL 2025
>
> **摘要:** With the rise of Large Language Models (LLMs) and their vision-enabled counterparts (VLMs), numerous works have investigated their capabilities in tasks that fuse the modalities of vision and language. In this work, we benchmark the extent to which VLMs are able to act as highly-trained phoneticians, interpreting spectrograms and waveforms of speech. To do this, we synthesise a novel dataset containing 4k+ English words spoken in isolation alongside stylistically consistent spectrogram and waveform figures. We test the ability of VLMs to understand these representations of speech through a multiple-choice task whereby models must predict the correct phonemic or graphemic transcription of a spoken word when presented amongst 3 distractor transcriptions that have been selected based on their phonemic edit distance to the ground truth. We observe that both zero-shot and finetuned models rarely perform above chance, demonstrating the requirement for specific parametric knowledge of how to interpret such figures, rather than paired samples alone.
>
---
#### [new 061] TCM-5CEval: Extended Deep Evaluation Benchmark for LLM's Comprehensive Clinical Research Competence in Traditional Chinese Medicine
- **分类: cs.CL**

- **简介: 该论文提出TCM-5CEval基准，用于评估大语言模型在中医领域的综合能力。解决的问题是现有模型在中医专业性和推理稳定性上的不足。工作包括构建五个维度的评测体系并测试15个模型，发现模型对经典文本理解弱且推理易受选项顺序影响。**

- **链接: [https://arxiv.org/pdf/2511.13169v1](https://arxiv.org/pdf/2511.13169v1)**

> **作者:** Tianai Huang; Jiayuan Chen; Lu Lu; Pengcheng Chen; Tianbin Li; Bing Han; Wenchao Tang; Jie Xu; Ming Li
>
> **备注:** 17 pages, 8 figures
>
> **摘要:** Large language models (LLMs) have demonstrated exceptional capabilities in general domains, yet their application in highly specialized and culturally-rich fields like Traditional Chinese Medicine (TCM) requires rigorous and nuanced evaluation. Building upon prior foundational work such as TCM-3CEval, which highlighted systemic knowledge gaps and the importance of cultural-contextual alignment, we introduce TCM-5CEval, a more granular and comprehensive benchmark. TCM-5CEval is designed to assess LLMs across five critical dimensions: (1) Core Knowledge (TCM-Exam), (2) Classical Literacy (TCM-LitQA), (3) Clinical Decision-making (TCM-MRCD), (4) Chinese Materia Medica (TCM-CMM), and (5) Clinical Non-pharmacological Therapy (TCM-ClinNPT). We conducted a thorough evaluation of fifteen prominent LLMs, revealing significant performance disparities and identifying top-performing models like deepseek\_r1 and gemini\_2\_5\_pro. Our findings show that while models exhibit proficiency in recalling foundational knowledge, they struggle with the interpretative complexities of classical texts. Critically, permutation-based consistency testing reveals widespread fragilities in model inference. All evaluated models, including the highest-scoring ones, displayed a substantial performance degradation when faced with varied question option ordering, indicating a pervasive sensitivity to positional bias and a lack of robust understanding. TCM-5CEval not only provides a more detailed diagnostic tool for LLM capabilities in TCM but aldso exposes fundamental weaknesses in their reasoning stability. To promote further research and standardized comparison, TCM-5CEval has been uploaded to the Medbench platform, joining its predecessor in the "In-depth Challenge for Comprehensive TCM Abilities" special track.
>
---
#### [new 062] Evaluating Autoformalization Robustness via Semantically Similar Paraphrasing
- **分类: cs.CL; cs.LO**

- **简介: 论文研究大语言模型在自动形式化任务中的鲁棒性问题，通过语义相似的改写文本测试模型输出稳定性，发现微小语义变化会导致显著性能波动。**

- **链接: [https://arxiv.org/pdf/2511.12784v1](https://arxiv.org/pdf/2511.12784v1)**

> **作者:** Hayden Moore; Asfahan Shah
>
> **摘要:** Large Language Models (LLMs) have recently emerged as powerful tools for autoformalization. Despite their impressive performance, these models can still struggle to produce grounded and verifiable formalizations. Recent work in text-to-SQL, has revealed that LLMs can be sensitive to paraphrased natural language (NL) inputs, even when high degrees of semantic fidelity are preserved (Safarzadeh, Oroojlooyjadid, and Roth 2025). In this paper, we investigate this claim in the autoformalization domain. Specifically, we evaluate the robustness of LLMs generating formal proofs with semantically similar paraphrased NL statements by measuring semantic and compilation validity. Using the formal benchmarks MiniF2F (Zheng, Han, and Polu 2021) and Lean 4 version of ProofNet (Xin et al. 2024), and two modern LLMs, we generate paraphrased natural language statements and cross-evaluate these statements across both models. The results of this paper reveal performance variability across paraphrased inputs, demonstrating that minor shifts in NL statements can significantly impact model outputs.
>
---
#### [new 063] AI-Salesman: Towards Reliable Large Language Model Driven Telemarketing
- **分类: cs.CL**

- **简介: 论文提出AI-Salesman框架，解决大语言模型在电销场景中策略脆弱和事实幻觉问题。构建首个真实对话数据集TeleSalesCorpus，设计双阶段方法：训练阶段用贝叶斯监督强化学习，推理阶段用动态大纲引导代理，显著提升销售对话效果。**

- **链接: [https://arxiv.org/pdf/2511.12133v1](https://arxiv.org/pdf/2511.12133v1)**

> **作者:** Qingyu Zhang; Chunlei Xin; Xuanang Chen; Yaojie Lu; Hongyu Lin; Xianpei Han; Le Sun; Qing Ye; Qianlong Xie; Xingxing Wang
>
> **摘要:** Goal-driven persuasive dialogue, exemplified by applications like telemarketing, requires sophisticated multi-turn planning and strict factual faithfulness, which remains a significant challenge for even state-of-the-art Large Language Models (LLMs). A lack of task-specific data often limits previous works, and direct LLM application suffers from strategic brittleness and factual hallucination. In this paper, we first construct and release TeleSalesCorpus, the first real-world-grounded dialogue dataset for this domain. We then propose AI-Salesman, a novel framework featuring a dual-stage architecture. For the training stage, we design a Bayesian-supervised reinforcement learning algorithm that learns robust sales strategies from noisy dialogues. For the inference stage, we introduce the Dynamic Outline-Guided Agent (DOGA), which leverages a pre-built script library to provide dynamic, turn-by-turn strategic guidance. Moreover, we design a comprehensive evaluation framework that combines fine-grained metrics for key sales skills with the LLM-as-a-Judge paradigm. Experimental results demonstrate that our proposed AI-Salesman significantly outperforms baseline models in both automatic metrics and comprehensive human evaluations, showcasing its effectiveness in complex persuasive scenarios.
>
---
#### [new 064] On the Entropy Calibration of Language Models
- **分类: cs.CL; cs.AI; cs.LG; stat.ML**

- **简介: 论文研究语言模型的熵校准问题，即模型生成文本的熵是否匹配其在人类文本上的对数损失。旨在解决误差累积导致的校准偏差问题，并探索在不牺牲质量的前提下实现校准的可能性。通过理论分析与实证测量发现，大模型校准改善有限，但理论上可通过黑箱预测未来熵实现无损校准。**

- **链接: [https://arxiv.org/pdf/2511.11966v1](https://arxiv.org/pdf/2511.11966v1)**

> **作者:** Steven Cao; Gregory Valiant; Percy Liang
>
> **备注:** Neurips 2025
>
> **摘要:** We study the problem of entropy calibration, which asks whether a language model's entropy over generations matches its log loss on human text. Past work found that models are miscalibrated, with entropy per step increasing (and text quality decreasing) as generations grow longer. This error accumulation is a fundamental problem in autoregressive models, and the standard solution is to truncate the distribution, which improves text quality at the cost of diversity. In this paper, we ask: is miscalibration likely to improve with scale, and is it theoretically possible to calibrate without tradeoffs? To build intuition, we first study a simplified theoretical setting to characterize the scaling behavior of miscalibration with respect to dataset size. We find that the scaling behavior depends on the power law exponent of the data distribution -- in particular, for a power law exponent close to 1, the scaling exponent is close to 0, meaning that miscalibration improves very slowly with scale. Next, we measure miscalibration empirically in language models ranging from 0.5B to 70B parameters. We find that the observed scaling behavior is similar to what is predicted by the simplified setting: our fitted scaling exponents for text are close to 0, meaning that larger models accumulate error at a similar rate as smaller ones. This scaling (or, lack thereof) provides one explanation for why we sample from larger models with similar amounts of truncation as smaller models, even though the larger models are of higher quality. However, truncation is not a satisfying solution because it comes at the cost of increased log loss. In theory, is it even possible to reduce entropy while preserving log loss? We prove that it is possible, if we assume access to a black box which can fit models to predict the future entropy of text.
>
---
#### [new 065] Evaluating the Ability of Large Language Models to Identify Adherence to CONSORT Reporting Guidelines in Randomized Controlled Trials: A Methodological Evaluation Study
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理在医学文献评估中的应用任务，旨在评估大语言模型（LLMs）在零样本条件下识别随机对照试验是否遵循CONSORT报告指南的能力。研究构建了150篇RCT的黄金标准数据集，通过三分类F1分数和Kappa系数衡量性能，发现模型能准确识别合规项，但难以识别不合规和不适用项，当前尚不能替代人工审查。**

- **链接: [https://arxiv.org/pdf/2511.13107v1](https://arxiv.org/pdf/2511.13107v1)**

> **作者:** Zhichao He; Mouxiao Bian; Jianhong Zhu; Jiayuan Chen; Yunqiu Wang; Wenxia Zhao; Tianbin Li; Bing Han; Jie Xu; Junyan Wu
>
> **摘要:** The Consolidated Standards of Reporting Trials statement is the global benchmark for transparent and high-quality reporting of randomized controlled trials. Manual verification of CONSORT adherence is a laborious, time-intensive process that constitutes a significant bottleneck in peer review and evidence synthesis. This study aimed to systematically evaluate the accuracy and reliability of contemporary LLMs in identifying the adherence of published RCTs to the CONSORT 2010 statement under a zero-shot setting. We constructed a golden standard dataset of 150 published RCTs spanning diverse medical specialties. The primary outcome was the macro-averaged F1-score for the three-class classification task, supplemented by item-wise performance metrics and qualitative error analysis. Overall model performance was modest. The top-performing models, Gemini-2.5-Flash and DeepSeek-R1, achieved nearly identical macro F1 scores of 0.634 and Cohen's Kappa coefficients of 0.280 and 0.282, respectively, indicating only fair agreement with expert consensus. A striking performance disparity was observed across classes: while most models could identify compliant items with high accuracy (F1 score > 0.850), they struggled profoundly with identifying non-compliant and not applicable items, where F1 scores rarely exceeded 0.400. Notably, some high-profile models like GPT-4o underperformed, achieving a macro F1-score of only 0.521. LLMs show potential as preliminary screening assistants for CONSORT checks, capably identifying well-reported items. However, their current inability to reliably detect reporting omissions or methodological flaws makes them unsuitable for replacing human expertise in the critical appraisal of trial quality.
>
---
#### [new 066] Improving Direct Persian-English Speech-to-Speech Translation with Discrete Units and Synthetic Parallel Data
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文提出一种直接 Persian-English 语音到语音翻译系统，解决低资源语言数据稀缺问题。通过自监督预训练、离散语音单元和合成平行数据提升性能，在CVSS数据集上ASR BLEU提升4.6。**

- **链接: [https://arxiv.org/pdf/2511.12690v1](https://arxiv.org/pdf/2511.12690v1)**

> **作者:** Sina Rashidi; Hossein Sameti
>
> **摘要:** Direct speech-to-speech translation (S2ST), in which all components are trained jointly, is an attractive alternative to cascaded systems because it offers a simpler pipeline and lower inference latency. However, direct S2ST models require large amounts of parallel speech data in the source and target languages, which are rarely available for low-resource languages such as Persian. This paper presents a direct S2ST system for translating Persian speech into English speech, as well as a pipeline for synthetic parallel Persian-English speech generation. The model comprises three components: (1) a conformer-based encoder, initialized from self-supervised pre-training, maps source speech to high-level acoustic representations; (2) a causal transformer decoder with relative position multi-head attention translates these representations into discrete target speech units; (3) a unit-based neural vocoder generates waveforms from the predicted discrete units. To mitigate the data scarcity problem, we construct a new Persian-English parallel speech corpus by translating Persian speech transcriptions into English using a large language model and then synthesizing the corresponding English speech with a state-of-the-art zero-shot text-to-speech system. The resulting corpus increases the amount of available parallel speech by roughly a factor of six. On the Persian-English portion of the CVSS corpus, the proposed model achieves improvement of 4.6 ASR BLEU with the synthetic data over direct baselines. These results indicate that combining self-supervised pre-training, discrete speech units, and synthetic parallel data is effective for improving direct S2ST in low-resource language pairs such as Persian-English
>
---
#### [new 067] ClinStructor: AI-Powered Structuring of Unstructured Clinical Texts
- **分类: cs.CL; cs.LG**

- **简介: 论文提出ClinStructor，将临床自由文本转化为结构化问答对，以提升模型透明性、可控性和跨场景泛化能力，解决无结构临床文本带来的偏见和性能不稳定问题。**

- **链接: [https://arxiv.org/pdf/2511.11883v1](https://arxiv.org/pdf/2511.11883v1)**

> **作者:** Karthikeyan K; Raghuveer Thirukovalluru; David Carlson
>
> **摘要:** Clinical notes contain valuable, context-rich information, but their unstructured format introduces several challenges, including unintended biases (e.g., gender or racial bias), and poor generalization across clinical settings (e.g., models trained on one EHR system may perform poorly on another due to format differences) and poor interpretability. To address these issues, we present ClinStructor, a pipeline that leverages large language models (LLMs) to convert clinical free-text into structured, task-specific question-answer pairs prior to predictive modeling. Our method substantially enhances transparency and controllability and only leads to a modest reduction in predictive performance (a 2-3% drop in AUC), compared to direct fine-tuning, on the ICU mortality prediction task. ClinStructor lays a strong foundation for building reliable, interpretable, and generalizable machine learning models in clinical environments.
>
---
#### [new 068] Improving LLM's Attachment to External Knowledge In Dialogue Generation Tasks Through Entity Anonymization
- **分类: cs.CL; cs.LG**

- **简介: 论文研究知识图谱驱动的对话生成任务，解决LLM依赖内部知识而忽视外部知识的问题。提出实体匿名化方法和评估指标LLM-KAT，提升模型对外部知识的利用能力。**

- **链接: [https://arxiv.org/pdf/2511.11946v1](https://arxiv.org/pdf/2511.11946v1)**

> **作者:** Hadi Sheikhi; Chenyang Huang; Osmar R. Zaïane
>
> **摘要:** Knowledge graph-based dialogue generation (KG-DG) is a challenging task requiring models to effectively incorporate external knowledge into conversational responses. While large language models (LLMs) have achieved impressive results across various NLP tasks, their ability to utilize external knowledge in KG-DG remains under-explored. We observe that LLMs often rely on internal knowledge, leading to detachment from provided knowledge graphs, even when they are given a flawlessly retrieved knowledge graph. First, we introduce LLM-KAT, an evaluation procedure for measuring knowledge attachment in generated responses. Second, we propose a simple yet effective entity anonymization technique to encourage LLMs to better leverage external knowledge. Experiments on the OpenDialKG dataset demonstrate that our approach improves LLMs' attachment on external knowledge.
>
---
#### [new 069] SGuard-v1: Safety Guardrail for Large Language Models
- **分类: cs.CL; cs.AI; cs.CR**

- **简介: 论文提出SGuard-v1，一个轻量级安全防护系统，用于大型语言模型。它通过两个组件检测有害内容和对抗性提示，解决AI安全风险问题。工作包括构建双模型、训练数据合成与指令微调，实现高安全性能与低部署开销。**

- **链接: [https://arxiv.org/pdf/2511.12497v1](https://arxiv.org/pdf/2511.12497v1)**

> **作者:** JoonHo Lee; HyeonMin Cho; Jaewoong Yun; Hyunjae Lee; JunKyu Lee; Juree Seok
>
> **备注:** Technical Report
>
> **摘要:** We present SGuard-v1, a lightweight safety guardrail for Large Language Models (LLMs), which comprises two specialized models to detect harmful content and screen adversarial prompts in human-AI conversational settings. The first component, ContentFilter, is trained to identify safety risks in LLM prompts and responses in accordance with the MLCommons hazard taxonomy, a comprehensive framework for trust and safety assessment of AI. The second component, JailbreakFilter, is trained with a carefully designed curriculum over integrated datasets and findings from prior work on adversarial prompting, covering 60 major attack types while mitigating false-unsafe classification. SGuard-v1 is built on the 2B-parameter Granite-3.3-2B-Instruct model that supports 12 languages. We curate approximately 1.4 million training instances from both collected and synthesized data and perform instruction tuning on the base model, distributing the curated data across the two component according to their designated functions. Through extensive evaluation on public and proprietary safety benchmarks, SGuard-v1 achieves state-of-the-art safety performance while remaining lightweight, thereby reducing deployment overhead. SGuard-v1 also improves interpretability for downstream use by providing multi-class safety predictions and their binary confidence scores. We release the SGuard-v1 under the Apache-2.0 License to enable further research and practical deployment in AI safety.
>
---
#### [new 070] Group-Aware Reinforcement Learning for Output Diversity in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对大语言模型输出多样性不足的问题，提出Group-Aware Policy Optimization（GAPO）方法，通过组级奖励机制提升生成多样性，同时保持任务准确率。**

- **链接: [https://arxiv.org/pdf/2511.12596v1](https://arxiv.org/pdf/2511.12596v1)**

> **作者:** Oron Anschel; Alon Shoshan; Adam Botach; Shunit Haviv Hakimi; Asaf Gendler; Emanuel Ben Baruch; Nadav Bhonker; Igor Kviatkovsky; Manoj Aggarwal; Gerard Medioni
>
> **备注:** EMNLP Main 2025
>
> **摘要:** Large Language Models (LLMs) often suffer from mode collapse, repeatedly generating the same few completions even when many valid answers exist, limiting their diversity across a wide range of tasks. We introduce Group-Aware Policy Optimization (GAPO), a simple extension of the recent and popular Group Relative Policy Optimization (GRPO) that computes rewards over the group as a whole. GAPO enables learning from the group-level properties such as diversity and coverage. We demonstrate GAPO using a frequency-aware reward function that encourages uniform sampling over valid LLM completions, and show that GAPO-trained models produce valid and more diverse model responses. Beyond this setup, GAPO generalizes to open-ended prompts and improves response diversity without compromising accuracy on standard LLM benchmarks (GSM8K, MATH, HumanEval, MMLU-Pro). Our code will be made publicly available.
>
---
#### [new 071] Consistency Is the Key: Detecting Hallucinations in LLM Generated Text By Checking Inconsistencies About Key Facts
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文提出CONFACTCHECK，用于在无外部知识库和模型访问限制下高效检测大语言模型生成文本中的幻觉。通过检查关键事实的一致性，实现低资源、高准确率的 hallucination 检测。**

- **链接: [https://arxiv.org/pdf/2511.12236v1](https://arxiv.org/pdf/2511.12236v1)**

> **作者:** Raavi Gupta; Pranav Hari Panicker; Sumit Bhatia; Ganesh Ramakrishnan
>
> **备注:** To appear at International Joint Conference on Natural Language Processing & Asia-Pacific Chapter of the Association for Computational Linguistics (IJCNLP-AACL), 2025
>
> **摘要:** Large language models (LLMs), despite their remarkable text generation capabilities, often hallucinate and generate text that is factually incorrect and not grounded in real-world knowledge. This poses serious risks in domains like healthcare, finance, and customer support. A typical way to use LLMs is via the APIs provided by LLM vendors where there is no access to model weights or options to fine-tune the model. Existing methods to detect hallucinations in such settings where the model access is restricted or constrained by resources typically require making multiple LLM API calls, increasing latency and API cost. We introduce CONFACTCHECK, an efficient hallucination detection approach that does not leverage any external knowledge base and works on the simple intuition that responses to factual probes within the generated text should be consistent within a single LLM and across different LLMs. Rigorous empirical evaluation on multiple datasets that cover both the generation of factual texts and the open generation shows that CONFACTCHECK can detect hallucinated facts efficiently using fewer resources and achieves higher accuracy scores compared to existing baselines that operate under similar conditions. Our code is available here.
>
---
#### [new 072] AA-Omniscience: Evaluating Cross-Domain Knowledge Reliability in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出AA-Omniscience基准，用于评估大语言模型在跨领域知识上的事实准确性和校准能力。针对现有评估忽视事实可靠性和知识盲区的问题，研究构建了6000道来自权威来源的题目，覆盖42个经济相关主题。结果揭示了前沿模型在事实性和校准上的普遍不足，且不同模型在各领域表现差异显著。**

- **链接: [https://arxiv.org/pdf/2511.13029v1](https://arxiv.org/pdf/2511.13029v1)**

> **作者:** Declan Jackson; William Keating; George Cameron; Micah Hill-Smith
>
> **摘要:** Existing language model evaluations primarily measure general capabilities, yet reliable use of these models across a range of domains demands factual accuracy and recognition of knowledge gaps. We introduce AA-Omniscience, a benchmark designed to measure both factual recall and knowledge calibration across 6,000 questions. Questions are derived from authoritative academic and industry sources, and cover 42 economically relevant topics within six different domains. The evaluation measures a model's Omniscience Index, a bounded metric (-100 to 100) measuring factual recall that jointly penalizes hallucinations and rewards abstention when uncertain, with 0 equating to a model that answers questions correctly as much as it does incorrectly. Among evaluated models, Claude 4.1 Opus attains the highest score (4.8), making it one of only three models to score above zero. These results reveal persistent factuality and calibration weaknesses across frontier models. Performance also varies by domain, with the models from three different research labs leading across the six domains. This performance variability suggests models should be chosen according to the demands of the use case rather than general performance for tasks where knowledge is important.
>
---
#### [new 073] Context-Emotion Aware Therapeutic Dialogue Generation: A Multi-component Reinforcement Learning Approach to Language Models for Mental Health Support
- **分类: cs.CL**

- **简介: 论文提出多组件强化学习方法，提升GPT-2在心理支持对话中的情境与情感感知能力，解决LLM缺乏专业治疗对话理解的问题，显著提高响应相关性、专业性和情绪准确性。**

- **链接: [https://arxiv.org/pdf/2511.11884v1](https://arxiv.org/pdf/2511.11884v1)**

> **作者:** Eric Hua Qing Zhang; Julia Ive
>
> **摘要:** Mental health illness represents a substantial global socioeconomic burden, with COVID-19 further exacerbating accessibility challenges and driving increased demand for telehealth mental health support. While large language models (LLMs) offer promising solutions through 24/7 availability and non-judgmental interactions, pre-trained models often lack the contextual and emotional awareness necessary for appropriate therapeutic responses. This paper investigated the application of supervised fine-tuning (SFT) and reinforcement learning (RL) techniques to enhance GPT-2's capacity for therapeutic dialogue generation. The methodology restructured input formats to enable simultaneous processing of contextual information and emotional states alongside user input, employing a multi-component reward function that aligned model outputs with professional therapist responses and annotated emotions. Results demonstrated improvements through reinforcement learning over baseline GPT-2 across multiple evaluation metrics: BLEU (0.0111), ROUGE-1 (0.1397), ROUGE-2 (0.0213), ROUGE-L (0.1317), and METEOR (0.0581). LLM evaluation confirmed high contextual relevance and professionalism, while reinforcement learning achieved 99.34% emotion accuracy compared to 66.96% for baseline GPT-2. These findings demonstrate reinforcement learning's effectiveness in developing therapeutic dialogue systems that can serve as valuable assistive tools for therapists while maintaining essential human clinical oversight.
>
---
#### [new 074] BioMedJImpact: A Comprehensive Dataset and LLM Pipeline for AI Engagement and Scientific Impact Analysis of Biomedical Journals
- **分类: cs.CL**

- **简介: 论文提出BioMedJImpact数据集与LLM分析管道，解决生物医学期刊影响力评估中协作结构与AI参与度的联合影响问题。工作包括构建大规模数据集、设计三阶段LLM流程提取AI参与度，并验证其有效性。**

- **链接: [https://arxiv.org/pdf/2511.12821v1](https://arxiv.org/pdf/2511.12821v1)**

> **作者:** Ruiyu Wang; Yuzhang Xie; Xiao Hu; Carl Yang; Jiaying Lu
>
> **摘要:** Assessing journal impact is central to scholarly communication, yet existing open resources rarely capture how collaboration structures and artificial intelligence (AI) research jointly shape venue prestige in biomedicine. We present BioMedJImpact, a large-scale, biomedical-oriented dataset designed to advance journal-level analysis of scientific impact and AI engagement. Built from 1.74 million PubMed Central articles across 2,744 journals, BioMedJImpact integrates bibliometric indicators, collaboration features, and LLM-derived semantic indicators for AI engagement. Specifically, the AI engagement feature is extracted through a reproducible three-stage LLM pipeline that we propose. Using this dataset, we analyze how collaboration intensity and AI engagement jointly influence scientific impact across pre- and post-pandemic periods (2016-2019, 2020-2023). Two consistent trends emerge: journals with higher collaboration intensity, particularly those with larger and more diverse author teams, tend to achieve greater citation impact, and AI engagement has become an increasingly strong correlate of journal prestige, especially in quartile rankings. To further validate the three-stage LLM pipeline we proposed for deriving the AI engagement feature, we conduct human evaluation, confirming substantial agreement in AI relevance detection and consistent subfield classification. Together, these contributions demonstrate that BioMedJImpact serves as both a comprehensive dataset capturing the intersection of biomedicine and AI, and a validated methodological framework enabling scalable, content-aware scientometric analysis of scientific impact and innovation dynamics. Code is available at https://github.com/JonathanWry/BioMedJImpact.
>
---
#### [new 075] A Comparative Analysis of Recurrent and Attention Architectures for Isolated Sign Language Recognition
- **分类: cs.CL**

- **简介: 论文研究孤立手语识别任务，比较循环神经网络（ConvLSTM）与注意力机制（Vanilla Transformer）的性能。结果表明，Transformer在准确率上优于ConvLSTM，尤其在小数据集上；而ConvLSTM更高效。研究为不同应用场景提供架构选择依据。**

- **链接: [https://arxiv.org/pdf/2511.13126v1](https://arxiv.org/pdf/2511.13126v1)**

> **作者:** Nigar Alishzade; Gulchin Abdullayeva
>
> **摘要:** This study presents a systematic comparative analysis of recurrent and attention-based neural architectures for isolated sign language recognition. We implement and evaluate two representative models-ConvLSTM and Vanilla Transformer-on the Azerbaijani Sign Language Dataset (AzSLD) and the Word-Level American Sign Language (WLASL) dataset. Our results demonstrate that the attention-based Vanilla Transformer consistently outperforms the recurrent ConvLSTM in both Top-1 and Top-5 accuracy across datasets, achieving up to 76.8% Top-1 accuracy on AzSLD and 88.3% on WLASL. The ConvLSTM, while more computationally efficient, lags in recognition accuracy, particularly on smaller datasets. These findings highlight the complementary strengths of each paradigm: the Transformer excels in overall accuracy and signer independence, whereas the ConvLSTM offers advantages in computational efficiency and temporal modeling. The study provides a nuanced analysis of these trade-offs, offering guidance for architecture selection in sign language recognition systems depending on application requirements and resource constraints.
>
---
#### [new 076] Adaptive Focus Memory for Language Models
- **分类: cs.CL; cs.AI**

- **简介: 论文提出Adaptive Focus Memory（AFM），用于多轮对话中动态管理上下文记忆。解决固定窗口和低效记忆策略导致的token浪费与关键信息丢失问题。AFM根据语义相似度、时间衰减和重要性分类，对消息分配不同保真度等级，在节省66% token的同时保持安全性和连贯性。**

- **链接: [https://arxiv.org/pdf/2511.12712v1](https://arxiv.org/pdf/2511.12712v1)**

> **作者:** Christopher Cruz
>
> **摘要:** Large language models (LLMs) are increasingly deployed in multi-turn dialogue settings, but their behavior is still bottlenecked by fixed context windows and naive memory strategies. Replaying the full conversation at every turn is simple but expensive, while static summarization or recency-only heuristics often erase safety-critical user details. We present Adaptive Focus Memory (AFM), a dynamic context manager that assigns each past message one of three fidelity levels -- FULL, COMPRESSED, or PLACEHOLDER -- based on semantic similarity to the current query, half-life recency weighting, and importance classification. AFM packs messages chronologically under a strict token budget, preferring high fidelity for the most relevant turns while aiming to preserve a cheap trace of the dialogue. In a safety-oriented benchmark involving a user with a severe peanut allergy planning a trip to Thailand, AFM retains the allergy across both short and medium-length conversations, matches the safety performance of naive replay, and cuts average token usage by 66% relative to a replay baseline. We release a modular Python implementation of AFM designed for OpenAI-compatible APIs and offline operation, enabling practitioners to reduce inference cost without sacrificing safety or factual continuity in the evaluated scenario.
>
---
#### [new 077] LLMLagBench: Identifying Temporal Training Boundaries in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出LLMLagBench，用于识别大语言模型训练数据的时间边界。任务是检测模型知识的新鲜度，解决因训练数据截止导致的时效性错误问题。工作包括设计基准测试、评估多种模型并验证其可靠性。**

- **链接: [https://arxiv.org/pdf/2511.12116v1](https://arxiv.org/pdf/2511.12116v1)**

> **作者:** Piotr Pęzik; Konrad Kaczyński; Maria Szymańska; Filip Żarnecki; Zuzanna Deckert; Jakub Kwiatkowski; Wojciech Janowski
>
> **摘要:** Large Language Models (LLMs) are pretrained on textual data up to a specific temporal cutoff. This creates a strict knowledge boundary beyond which models cannot provide accurate information without querying external sources. More subtly, when this limitation is unknown or ignored, LLMs may inadvertently blend outdated time-sensitive information with general knowledge during reasoning tasks, potentially compromising response accuracy. We introduce LLMLagBench, an LLM freshness benchmark, as a systematic approach for identifying the earliest probable temporal boundaries of an LLM's training data by evaluating its knowledge of recent events. We then apply this benchmark to evaluate a large set of LLMs, including models with both explicitly declared and undeclared training cutoffs. The reliability of the benchmark is assessed by manual validation and comparison with publicly released information about LLM pretraining.
>
---
#### [new 078] On the Brittleness of LLMs: A Journey around Set Membership
- **分类: cs.CL**

- **简介: 论文研究大语言模型在集合成员查询任务中的脆弱性，揭示其在简单推理任务上的不可靠表现。通过系统实验分析不同因素对模型性能的影响，发现模型对集合概念的理解碎片化，提出以简化任务为方法评估LLM可靠性。**

- **链接: [https://arxiv.org/pdf/2511.12728v1](https://arxiv.org/pdf/2511.12728v1)**

> **作者:** Lea Hergert; Gábor Berend; Mario Szegedy; Gyorgy Turan; Márk Jelasity
>
> **摘要:** Large language models (LLMs) achieve superhuman performance on complex reasoning tasks, yet often fail on much simpler problems, raising concerns about their reliability and interpretability. We investigate this paradox through a focused study with two key design features: simplicity, to expose basic failure modes, and scale, to enable comprehensive controlled experiments. We focus on set membership queries -- among the most fundamental forms of reasoning -- using tasks like ``Is apple an element of the set \{pear, plum, apple, raspberry\}?''. We conduct a systematic empirical evaluation across prompt phrasing, semantic structure, element ordering, and model choice. Our large-scale analysis reveals that LLM performance on this elementary task is consistently brittle, and unpredictable across all dimensions, suggesting that the models' ``understanding'' of the set concept is fragmented and convoluted at best. Our work demonstrates that the large-scale experiments enabled by the simplicity of the problem allow us to map and analyze the failure modes comprehensively, making this approach a valuable methodology for LLM evaluation in general.
>
---
#### [new 079] Knots: A Large-Scale Multi-Agent Enhanced Expert-Annotated Dataset and LLM Prompt Optimization for NOTAM Semantic Parsing
- **分类: cs.CL; cs.AI**

- **简介: 论文提出NOTAM语义解析任务，解决现有方法对复杂语言结构和隐含推理理解不足的问题。构建了12,347条专家标注的Knots数据集，并通过多智能体框架增强领域知识，优化大模型提示策略，提升自动化NOTAM分析能力。**

- **链接: [https://arxiv.org/pdf/2511.12630v1](https://arxiv.org/pdf/2511.12630v1)**

> **作者:** Maoqi Liu; Quan Fang; Yang Yang; Can Zhao; Kaiquan Cai
>
> **备注:** Accepted to Advanced Engineering Informatics
>
> **摘要:** Notice to Air Missions (NOTAMs) serve as a critical channel for disseminating key flight safety information, yet their complex linguistic structures and implicit reasoning pose significant challenges for automated parsing. Existing research mainly focuses on surface-level tasks such as classification and named entity recognition, lacking deep semantic understanding. To address this gap, we propose NOTAM semantic parsing, a task emphasizing semantic inference and the integration of aviation domain knowledge to produce structured, inference-rich outputs. To support this task, we construct Knots (Knowledge and NOTAM Semantics), a high-quality dataset of 12,347 expert-annotated NOTAMs covering 194 Flight Information Regions, enhanced through a multi-agent collaborative framework for comprehensive field discovery. We systematically evaluate a wide range of prompt-engineering strategies and model-adaptation techniques, achieving substantial improvements in aviation text understanding and processing. Our experimental results demonstrate the effectiveness of the proposed approach and offer valuable insights for automated NOTAM analysis systems. Our code is available at: https://github.com/Estrellajer/Knots.
>
---
#### [new 080] Aspect-Level Obfuscated Sentiment in Thai Financial Disclosures and Its Impact on Abnormal Returns
- **分类: cs.CL**

- **简介: 该论文属于金融文本情感分析任务，旨在解决泰国财报中隐晦表达的 sentiment 识别问题。作者构建了标注数据集，提出基于 aspect 的情感分析方法，并验证其对股价异常收益的影响，揭示特定内容对市场反应的选择性作用。**

- **链接: [https://arxiv.org/pdf/2511.13481v1](https://arxiv.org/pdf/2511.13481v1)**

> **作者:** Attapol T. Rutherford; Sirisak Chueykamhang; Thachaparn Bunditlurdruk; Nanthicha Angsuwichitkul
>
> **摘要:** Understanding sentiment in financial documents is crucial for gaining insights into market behavior. These reports often contain obfuscated language designed to present a positive or neutral outlook, even when underlying conditions may be less favorable. This paper presents a novel approach using Aspect-Based Sentiment Analysis (ABSA) to decode obfuscated sentiment in Thai financial annual reports. We develop specific guidelines for annotating obfuscated sentiment in these texts and annotate more than one hundred financial reports. We then benchmark various text classification models on this annotated dataset, demonstrating strong performance in sentiment classification. Additionally, we conduct an event study to evaluate the real-world implications of our sentiment analysis on stock prices. Our results suggest that market reactions are selectively influenced by specific aspects within the reports. Our findings underscore the complexity of sentiment analysis in financial texts and highlight the importance of addressing obfuscated language to accurately assess market sentiment.
>
---
#### [new 081] Towards Autoformalization of LLM-generated Outputs for Requirement Verification
- **分类: cs.CL; cs.AI; cs.FL; cs.LO**

- **简介: 论文研究LLM生成输出的自动形式化验证任务，旨在解决缺乏 formal 方法验证 LLM 输出准确性的问题。作者提出简单 autoformalizer 流程，在两个实验中分别验证了语义一致性与逻辑不一致，证明其在保证输出质量上的潜力。**

- **链接: [https://arxiv.org/pdf/2511.11829v1](https://arxiv.org/pdf/2511.11829v1)**

> **作者:** Mihir Gupte; Ramesh S
>
> **备注:** To be submitted for publication
>
> **摘要:** Autoformalization, the process of translating informal statements into formal logic, has gained renewed interest with the emergence of powerful Large Language Models (LLMs). While LLMs show promise in generating structured outputs from natural language (NL), such as Gherkin Scenarios from NL feature requirements, there's currently no formal method to verify if these outputs are accurate. This paper takes a preliminary step toward addressing this gap by exploring the use of a simple LLM-based autoformalizer to verify LLM-generated outputs against a small set of natural language requirements. We conducted two distinct experiments. In the first one, the autoformalizer successfully identified that two differently-worded NL requirements were logically equivalent, demonstrating the pipeline's potential for consistency checks. In the second, the autoformalizer was used to identify a logical inconsistency between a given NL requirement and an LLM-generated output, highlighting its utility as a formal verification tool. Our findings, while limited, suggest that autoformalization holds significant potential for ensuring the fidelity and logical consistency of LLM-generated outputs, laying a crucial foundation for future, more extensive studies into this novel application.
>
---
#### [new 082] ViConBERT: Context-Gloss Aligned Vietnamese Word Embedding for Polysemous and Sense-Aware Representations
- **分类: cs.CL**

- **简介: 该论文提出ViConBERT，用于越南语多义词消歧和语义理解任务。针对越南语缺乏高质量上下文嵌入模型的问题，结合对比学习与释义蒸馏，构建了首个大规模合成数据集ViConWSD，显著提升WSD和语义相似度任务性能。**

- **链接: [https://arxiv.org/pdf/2511.12249v1](https://arxiv.org/pdf/2511.12249v1)**

> **作者:** Khang T. Huynh; Dung H. Nguyen; Binh T. Nguyen
>
> **摘要:** Recent advances in contextualized word embeddings have greatly improved semantic tasks such as Word Sense Disambiguation (WSD) and contextual similarity, but most progress has been limited to high-resource languages like English. Vietnamese, in contrast, still lacks robust models and evaluation resources for fine-grained semantic understanding. In this paper, we present ViConBERT, a novel framework for learning Vietnamese contextualized embeddings that integrates contrastive learning (SimCLR) and gloss-based distillation to better capture word meaning. We also introduce ViConWSD, the first large-scale synthetic dataset for evaluating semantic understanding in Vietnamese, covering both WSD and contextual similarity. Experimental results show that ViConBERT outperforms strong baselines on WSD (F1 = 0.87) and achieves competitive performance on ViCon (AP = 0.88) and ViSim-400 (Spearman's rho = 0.60), demonstrating its effectiveness in modeling both discrete senses and graded semantic relations. Our code, models, and data are available at https://github.com/tkhangg0910/ViConBERT
>
---
#### [new 083] RegionMarker: A Region-Triggered Semantic Watermarking Framework for Embedding-as-a-Service Copyright Protection
- **分类: cs.CL; cs.CR**

- **简介: 该论文提出RegionMarker框架，用于Embedding-as-a-Service的版权保护。针对模型提取攻击导致的经济风险，通过区域触发的语义水印机制，在低维空间嵌入水印，增强对 paraphrasing 和维度扰动攻击的鲁棒性，实现全面防护。**

- **链接: [https://arxiv.org/pdf/2511.13329v1](https://arxiv.org/pdf/2511.13329v1)**

> **作者:** Shufan Yang; Zifeng Cheng; Zhiwei Jiang; Yafeng Yin; Cong Wang; Shiping Ge; Yuchen Fu; Qing Gu
>
> **备注:** AAAI 2026
>
> **摘要:** Embedding-as-a-Service (EaaS) is an effective and convenient deployment solution for addressing various NLP tasks. Nevertheless, recent research has shown that EaaS is vulnerable to model extraction attacks, which could lead to significant economic losses for model providers. For copyright protection, existing methods inject watermark embeddings into text embeddings and use them to detect copyright infringement. However, current watermarking methods often resist only a subset of attacks and fail to provide \textit{comprehensive} protection. To this end, we present the region-triggered semantic watermarking framework called RegionMarker, which defines trigger regions within a low-dimensional space and injects watermarks into text embeddings associated with these regions. By utilizing a secret dimensionality reduction matrix to project onto this subspace and randomly selecting trigger regions, RegionMarker makes it difficult for watermark removal attacks to evade detection. Furthermore, by embedding watermarks across the entire trigger region and using the text embedding as the watermark, RegionMarker is resilient to both paraphrasing and dimension-perturbation attacks. Extensive experiments on various datasets show that RegionMarker is effective in resisting different attack methods, thereby protecting the copyright of EaaS.
>
---
#### [new 084] Spark-Prover-X1: Formal Theorem Proving Through Diverse Data Training
- **分类: cs.CL**

- **简介: 论文提出Spark-Prover-X1，通过三阶段训练提升7B模型的数学定理证明能力，解决小模型因数据稀缺导致的推理不足问题。引入新数据任务和评估基准ExamFormal-Bench，实现开源模型中的最先进性能。**

- **链接: [https://arxiv.org/pdf/2511.13043v1](https://arxiv.org/pdf/2511.13043v1)**

> **作者:** Xinyuan Zhou; Yi Lei; Xiaoyu Zhou; Jingyi Sun; Yu Zhu; Zhongyi Ye; Weitai Zhang; Quan Liu; Si Wei; Cong Liu
>
> **摘要:** Large Language Models (LLMs) have shown significant promise in automated theorem proving, yet progress is often constrained by the scarcity of diverse and high-quality formal language data. To address this issue, we introduce Spark-Prover-X1, a 7B parameter model trained via an three-stage framework designed to unlock the reasoning potential of more accessible and moderately-sized LLMs. The first stage infuses deep knowledge through continuous pre-training on a broad mathematical corpus, enhanced by a suite of novel data tasks. Key innovation is a "CoT-augmented state prediction" task to achieve fine-grained reasoning. The second stage employs Supervised Fine-tuning (SFT) within an expert iteration loop to specialize both the Spark-Prover-X1-7B and Spark-Formalizer-X1-7B models. Finally, a targeted round of Group Relative Policy Optimization (GRPO) is applied to sharpen the prover's capabilities on the most challenging problems. To facilitate robust evaluation, particularly on problems from real-world examinations, we also introduce ExamFormal-Bench, a new benchmark dataset of 402 formal problems. Experimental results demonstrate that Spark-Prover-X1-7B achieves state-of-the-art performance among similarly-sized open-source models, attaining a 37.0\% average pass rate (pass@32). It shows exceptional performance on difficult competition benchmarks, notably solving 27 problems on PutnamBench (pass@32) and achieving 24.0\% on CombiBench (pass@32). Our work validates that this diverse training data and progressively refined training pipeline provides an effective path for enhancing the formal reasoning capabilities of lightweight LLMs. Both Spark-Prover-X1-7B and Spark-Formalizer-X1-7B, along with the ExamFormal-Bench dataset, are made publicly available at:https://www.modelscope.cn/organization/iflytek, https://gitcode.com/ifly_opensource.
>
---
#### [new 085] MME-RAG: Multi-Manager-Expert Retrieval-Augmented Generation for Fine-Grained Entity Recognition in Task-Oriented Dialogues
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出MME-RAG框架，用于任务导向对话中的细粒度实体识别。针对大模型在领域适应和检索可控性上的挑战，通过管理者-专家协同机制与关键信息检索增强生成，实现无需额外训练的精准提取与跨域泛化。**

- **链接: [https://arxiv.org/pdf/2511.12213v1](https://arxiv.org/pdf/2511.12213v1)**

> **作者:** Liang Xue; Haoyu Liu; Yajun Tian; Xinyu Zhong; Yang Liu
>
> **摘要:** Fine-grained entity recognition is crucial for reasoning and decision-making in task-oriented dialogues, yet current large language models (LLMs) continue to face challenges in domain adaptation and retrieval controllability. We introduce MME-RAG, a Multi-Manager-Expert Retrieval-Augmented Generation framework that decomposes entity recognition into two coordinated stages: type-level judgment by lightweight managers and span-level extraction by specialized experts. Each expert is supported by a KeyInfo retriever that injects semantically aligned, few-shot exemplars during inference, enabling precise and domain-adaptive extraction without additional training. Experiments on CrossNER, MIT-Movie, MIT-Restaurant, and our newly constructed multi-domain customer-service dataset demonstrate that MME-RAG performs better than recent baselines in most domains. Ablation studies further show that both the hierarchical decomposition and KeyInfo-guided retrieval are key drivers of robustness and cross-domain generalization, establishing MME-RAG as a scalable and interpretable solution for adaptive dialogue understanding.
>
---
#### [new 086] BeDiscovER: The Benchmark of Discourse Understanding in the Era of Reasoning Language Models
- **分类: cs.CL**

- **简介: 该论文提出BeDiscovER基准，用于评估大语言模型在推理时代下的话语理解能力。针对当前模型在文档级推理和细微语用现象上的不足，整合52个数据集，涵盖话语解析、时间关系抽取等任务，并对多个模型进行评测，揭示其优势与局限。**

- **链接: [https://arxiv.org/pdf/2511.13095v1](https://arxiv.org/pdf/2511.13095v1)**

> **作者:** Chuyuan Li; Giuseppe Carenini
>
> **摘要:** We introduce BeDiscovER (Benchmark of Discourse Understanding in the Era of Reasoning Language Models), an up-to-date, comprehensive suite for evaluating the discourse-level knowledge of modern LLMs. BeDiscovER compiles 5 publicly available discourse tasks across discourse lexicon, (multi-)sentential, and documental levels, with in total 52 individual datasets. It covers both extensively studied tasks such as discourse parsing and temporal relation extraction, as well as some novel challenges such as discourse particle disambiguation (e.g., ``just''), and also aggregates a shared task on Discourse Relation Parsing and Treebanking for multilingual and multi-framework discourse relation classification. We evaluate open-source LLMs: Qwen3 series, DeepSeek-R1, and frontier model such as GPT-5-mini on BeDiscovER, and find that state-of-the-art models exhibit strong performance in arithmetic aspect of temporal reasoning, but they struggle with full document reasoning and some subtle semantic and discourse phenomena, such as rhetorical relation recognition.
>
---
#### [new 087] DenseAnnotate: Enabling Scalable Dense Caption Collection for Images and 3D Scenes via Spoken Descriptions
- **分类: cs.CV; cs.CL**

- **简介: 论文提出DenseAnnotate平台，通过语音描述实现图像与3D场景的密集标注，解决传统文本标注效率低、覆盖不全的问题。工作包括设计音频驱动标注系统、收集多语言密集标注数据集，并验证其在多语言、文化对齐和3D空间理解上的性能提升。**

- **链接: [https://arxiv.org/pdf/2511.12452v1](https://arxiv.org/pdf/2511.12452v1)**

> **作者:** Xiaoyu Lin; Aniket Ghorpade; Hansheng Zhu; Justin Qiu; Dea Rrozhani; Monica Lama; Mick Yang; Zixuan Bian; Ruohan Ren; Alan B. Hong; Jiatao Gu; Chris Callison-Burch
>
> **摘要:** With the rapid adoption of multimodal large language models (MLLMs) across diverse applications, there is a pressing need for task-centered, high-quality training data. A key limitation of current training datasets is their reliance on sparse annotations mined from the Internet or entered via manual typing that capture only a fraction of an image's visual content. Dense annotations are more valuable but remain scarce. Traditional text-based annotation pipelines are poorly suited for creating dense annotations: typing limits expressiveness, slows annotation speed, and underrepresents nuanced visual features, especially in specialized areas such as multicultural imagery and 3D asset annotation. In this paper, we present DenseAnnotate, an audio-driven online annotation platform that enables efficient creation of dense, fine-grained annotations for images and 3D assets. Annotators narrate observations aloud while synchronously linking spoken phrases to image regions or 3D scene parts. Our platform incorporates speech-to-text transcription and region-of-attention marking. To demonstrate the effectiveness of DenseAnnotate, we conducted case studies involving over 1,000 annotators across two domains: culturally diverse images and 3D scenes. We curate a human-annotated multi-modal dataset of 3,531 images, 898 3D scenes, and 7,460 3D objects, with audio-aligned dense annotations in 20 languages, including 8,746 image captions, 2,000 scene captions, and 19,000 object captions. Models trained on this dataset exhibit improvements of 5% in multilingual, 47% in cultural alignment, and 54% in 3D spatial capabilities. Our results show that our platform offers a feasible approach for future vision-language research and can be applied to various tasks and diverse types of data.
>
---
#### [new 088] LLM Architecture, Scaling Laws, and Economics: A Quick Summary
- **分类: cs.GL; cs.CL; cs.LG**

- **简介: 该论文总结了大语言模型（LLM）的标准架构、缩放定律及经济成本，旨在清晰呈现当前LLM技术的核心要素。属于综述类任务，解决信息分散难获取问题，整理了Transformer结构、计算与内存缩放规律及参数成本估算。**

- **链接: [https://arxiv.org/pdf/2511.11572v1](https://arxiv.org/pdf/2511.11572v1)**

> **作者:** William H. Press
>
> **备注:** 9 pages, 3 figures
>
> **摘要:** The current standard architecture of Large Language Models (LLMs) with QKV self-attention is briefly summarized, including the architecture of a typical Transformer. Scaling laws for compute (flops) and memory (parameters plus data) are given, along with their present (2025) rough cost estimates for the parameters of present LLMs of various scales, including discussion of whether DeepSeek should be viewed as a special case. Nothing here is new, but this material seems not otherwise readily available in summary form.
>
---
#### [new 089] LLM-Generated Negative News Headlines Dataset: Creation and Benchmarking Against Real Journalism
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决真实新闻数据获取难与隐私问题。作者构建了由大语言模型生成的负面新闻标题数据集，并通过多项指标对比验证其与真实新闻的相似性，结果表明生成数据在多数维度上接近真实数据，仅在专有名词使用上有差异。**

- **链接: [https://arxiv.org/pdf/2511.11591v1](https://arxiv.org/pdf/2511.11591v1)**

> **作者:** Olusola Babalola; Bolanle Ojokoh; Olutayo Boyinbode
>
> **备注:** 50 pages, 19 figures, 9 tables
>
> **摘要:** This research examines the potential of datasets generated by Large Language Models (LLMs) to support Natural Language Processing (NLP) tasks, aiming to overcome challenges related to data acquisition and privacy concerns associated with real-world data. Focusing on negative valence text, a critical component of sentiment analysis, we explore the use of LLM-generated synthetic news headlines as an alternative to real-world data. A specialized corpus of negative news headlines was created using tailored prompts to capture diverse negative sentiments across various societal domains. The synthetic headlines were validated by expert review and further analyzed in embedding space to assess their alignment with real-world negative news in terms of content, tone, length, and style. Key metrics such as correlation with real headlines, perplexity, coherence, and realism were evaluated. The synthetic dataset was benchmarked against two sets of real news headlines using evaluations including the Comparative Perplexity Test, Comparative Readability Test, Comparative POS Profiling, BERTScore, and Comparative Semantic Similarity. Results show the generated headlines match real headlines with the only marked divergence being in the proper noun score of the POS profile test.
>
---
#### [new 090] Do LLMs Really Struggle at NL-FOL Translation? Revealing their Strengths via a Novel Benchmarking Strategy
- **分类: cs.AI; cs.CL; cs.LO**

- **简介: 论文研究自然语言到一阶逻辑（NL-FOL）翻译任务，旨在解决现有评估方法误判LLMs能力的问题。作者提出新评测协议，证明对话型LLMs具备真实逻辑理解能力，而嵌入模型表现较差。**

- **链接: [https://arxiv.org/pdf/2511.11816v1](https://arxiv.org/pdf/2511.11816v1)**

> **作者:** Andrea Brunello; Luca Geatti; Michele Mignani; Angelo Montanari; Nicola Saccomanno
>
> **备注:** Full version of the paper accepted for publication at The 40th Annual AAAI Conference on Artificial Intelligence (AAAI 2026)
>
> **摘要:** Due to its expressiveness and unambiguous nature, First-Order Logic (FOL) is a powerful formalism for representing concepts expressed in natural language (NL). This is useful, e.g., for specifying and verifying desired system properties. While translating FOL into human-readable English is relatively straightforward, the inverse problem, converting NL to FOL (NL-FOL translation), has remained a longstanding challenge, for both humans and machines. Although the emergence of Large Language Models (LLMs) promised a breakthrough, recent literature provides contrasting results on their ability to perform NL-FOL translation. In this work, we provide a threefold contribution. First, we critically examine existing datasets and protocols for evaluating NL-FOL translation performance, revealing key limitations that may cause a misrepresentation of LLMs' actual capabilities. Second, to overcome these shortcomings, we propose a novel evaluation protocol explicitly designed to distinguish genuine semantic-level logical understanding from superficial pattern recognition, memorization, and dataset contamination. Third, using this new approach, we show that state-of-the-art, dialogue-oriented LLMs demonstrate strong NL-FOL translation skills and a genuine grasp of sentence-level logic, whereas embedding-centric models perform markedly worse.
>
---
#### [new 091] WebCoach: Self-Evolving Web Agents with Cross-Session Memory Guidance
- **分类: cs.AI; cs.CL**

- **简介: 论文提出WebCoach框架，解决网页代理在跨会话中无法积累经验的问题。通过构建持久记忆系统与自我进化机制，提升代理长期规划和持续学习能力，显著改善复杂浏览任务的性能。**

- **链接: [https://arxiv.org/pdf/2511.12997v1](https://arxiv.org/pdf/2511.12997v1)**

> **作者:** Genglin Liu; Shijie Geng; Sha Li; Hejie Cui; Sarah Zhang; Xin Liu; Tianyi Liu
>
> **备注:** 18 pages; work in progress
>
> **摘要:** Multimodal LLM-powered agents have recently demonstrated impressive capabilities in web navigation, enabling agents to complete complex browsing tasks across diverse domains. However, current agents struggle with repetitive errors and lack the ability to learn from past experiences across sessions, limiting their long-term robustness and sample efficiency. We introduce WebCoach, a model-agnostic self-evolving framework that equips web browsing agents with persistent cross-session memory, enabling improved long-term planning, reflection, and continual learning without retraining. WebCoach consists of three key components: (1) a WebCondenser, which standardizes raw navigation logs into concise summaries; (2) an External Memory Store, which organizes complete trajectories as episodic experiences; and (3) a Coach, which retrieves relevant experiences based on similarity and recency, and decides whether to inject task-specific advice into the agent via runtime hooks. This design empowers web agents to access long-term memory beyond their native context window, improving robustness in complex browsing tasks. Moreover, WebCoach achieves self-evolution by continuously curating episodic memory from new navigation trajectories, enabling agents to improve over time without retraining. Evaluations on the WebVoyager benchmark demonstrate that WebCoach consistently improves the performance of browser-use agents across three different LLM backbones. With a 38B model, it increases task success rates from 47% to 61% while reducing or maintaining the average number of steps. Notably, smaller base models with WebCoach achieve performance comparable to the same web agent using GPT-4o.
>
---
#### [new 092] Live-SWE-agent: Can Software Engineering Agents Self-Evolve on the Fly?
- **分类: cs.SE; cs.AI; cs.CL; cs.LG**

- **简介: 论文提出Live-SWE-agent，一种可在运行时自我演化的软件工程代理，解决现有代理设计成本高、泛化差的问题。它从基础工具起步，边执行任务边优化自身结构，在SWE-bench验证集上达到75.4%求解率，优于所有开源代理。**

- **链接: [https://arxiv.org/pdf/2511.13646v1](https://arxiv.org/pdf/2511.13646v1)**

> **作者:** Chunqiu Steven Xia; Zhe Wang; Yan Yang; Yuxiang Wei; Lingming Zhang
>
> **摘要:** Large Language Models (LLMs) are reshaping almost all industries, including software engineering. In recent years, a number of LLM agents have been proposed to solve real-world software problems. Such software agents are typically equipped with a suite of coding tools and can autonomously decide the next actions to form complete trajectories to solve end-to-end software tasks. While promising, they typically require dedicated design and may still be suboptimal, since it can be extremely challenging and costly to exhaust the entire agent scaffold design space. Recognizing that software agents are inherently software themselves that can be further refined/modified, researchers have proposed a number of self-improving software agents recently, including the Darwin-Gödel Machine (DGM). Meanwhile, such self-improving agents require costly offline training on specific benchmarks and may not generalize well across different LLMs or benchmarks. In this paper, we propose Live-SWE-agent, the first live software agent that can autonomously and continuously evolve itself on-the-fly during runtime when solving real-world software problems. More specifically, Live-SWE-agent starts with the most basic agent scaffold with only access to bash tools (e.g., mini-SWE-agent), and autonomously evolves its own scaffold implementation while solving real-world software problems. Our evaluation on the widely studied SWE-bench Verified benchmark shows that Live-SWE-agent can achieve an impressive solve rate of 75.4% without test-time scaling, outperforming all existing open-source software agents and approaching the performance of the best proprietary solution. Moreover, Live-SWE-agent outperforms state-of-the-art manually crafted software agents on the recent SWE-Bench Pro benchmark, achieving the best-known solve rate of 45.8%.
>
---
#### [new 093] Better LLM Reasoning via Dual-Play
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 论文提出PasoDoble框架，通过双模型对抗训练提升大语言模型推理能力。Proposer生成挑战性问题，Solver尝试解答，两者互为对手，在无监督条件下共同进化，解决奖励黑客和训练不稳定问题，显著增强模型推理性能。**

- **链接: [https://arxiv.org/pdf/2511.11881v1](https://arxiv.org/pdf/2511.11881v1)**

> **作者:** Zhengxin Zhang; Chengyu Huang; Aochong Oliver Li; Claire Cardie
>
> **摘要:** Large Language Models (LLMs) have achieved remarkable progress through Reinforcement Learning with Verifiable Rewards (RLVR), yet still rely heavily on external supervision (e.g., curated labels). Adversarial learning, particularly through self-play, offers a promising alternative that enables models to iteratively learn from themselves - thus reducing reliance on external supervision. Dual-play extends adversarial learning by assigning specialized roles to two models and training them against each other, fostering sustained competition and mutual evolution. Despite its promise, adapting dual-play training to LLMs remains limited, largely due to their susceptibility to reward hacking and training instability. In this paper, we introduce PasoDoble, a novel LLM dual-play framework. PasoDoble adversarially trains two models initialized from the same base model: a Proposer, which generates challenging questions with ground-truth answers, and a Solver, which attempts to solve them. We enrich the Proposer with knowledge from a pre-training dataset to ensure the questions' quality and diversity. To avoid reward hacking, the Proposer is rewarded for producing only valid questions that push the Solver's limit, while the Solver is rewarded for solving them correctly, and both are updated jointly. To further enhance training stability, we introduce an optional offline paradigm that decouples Proposer and Solver updates, alternately updating each for several steps while holding the other fixed. Notably, PasoDoble operates without supervision during training. Experimental results show that PasoDoble can improve the reasoning performance of LLMs. Our project page is available at https://hcy123902.github.io/PasoDoble.
>
---
#### [new 094] Forgetting-MarI: LLM Unlearning via Marginal Information Regularization
- **分类: cs.AI; cs.CL; cs.CR; cs.IT; cs.LG**

- **简介: 该论文提出Forgetting-MarI框架，解决大语言模型中特定数据的可控删除问题。通过惩罚边际信息，仅移除目标数据带来的额外影响，确保模型性能不受损且可证明不可检测。**

- **链接: [https://arxiv.org/pdf/2511.11914v1](https://arxiv.org/pdf/2511.11914v1)**

> **作者:** Shizhou Xu; Yuan Ni; Stefan Broecker; Thomas Strohmer
>
> **摘要:** As AI models are trained on ever-expanding datasets, the ability to remove the influence of specific data from trained models has become essential for privacy protection and regulatory compliance. Unlearning addresses this challenge by selectively removing parametric knowledge from the trained models without retraining from scratch, which is critical for resource-intensive models such as Large Language Models (LLMs). Existing unlearning methods often degrade model performance by removing more information than necessary when attempting to ''forget'' specific data. We introduce Forgetting-MarI, an LLM unlearning framework that provably removes only the additional (marginal) information contributed by the data to be unlearned, while preserving the information supported by the data to be retained. By penalizing marginal information, our method yields an explicit upper bound on the unlearn dataset's residual influence in the trained models, providing provable undetectability. Extensive experiments confirm that our approach outperforms current state-of-the-art unlearning methods, delivering reliable forgetting and better preserved general model performance across diverse benchmarks. This advancement represents an important step toward making AI systems more controllable and compliant with privacy and copyright regulations without compromising their effectiveness.
>
---
#### [new 095] How Far Do SSL Speech Models Listen for Tone? Temporal Focus of Tone Representation under Low-resource Transfer
- **分类: eess.AS; cs.CL**

- **简介: 该论文研究SSL语音模型在低资源条件下对声调的时序感知范围，针对缅甸语、泰语、老挝语和越南语四种声调语言，发现模型听觉跨度因下游任务而异：语音识别任务聚焦语言特定声调线索，而韵律相关任务则偏向过长跨度。**

- **链接: [https://arxiv.org/pdf/2511.12285v1](https://arxiv.org/pdf/2511.12285v1)**

> **作者:** Minu Kim; Ji Sub Um; Hoirin Kim
>
> **备注:** 5 pages, 7 figures, submitted to ICASSP 2026
>
> **摘要:** Lexical tone is central to many languages but remains underexplored in self-supervised learning (SSL) speech models, especially beyond Mandarin. We study four languages with complex and diverse tone systems: Burmese, Thai, Lao, and Vietnamese, to examine how far such models listen for tone and how transfer operates in low-resource conditions. As a baseline reference, we estimate the temporal span of tone cues to be about 100 ms in Burmese and Thai, and about 180 ms in Lao and Vietnamese. Probes and gradient analyses on fine-tuned SSL models reveal that tone transfer varies by downstream task: automatic speech recognition fine-tuning aligns spans with language-specific tone cues, while prosody- and voice-related tasks bias the model toward overly long spans. These findings indicate that tone transfer is shaped by downstream task, highlighting task effects on temporal focus in tone modeling.
>
---
#### [new 096] Reasoning: From Reflection to Solution
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 论文探讨推理的本质，提出其为状态空间中迭代算子应用直至收敛的机制。针对当前大模型仅模式匹配而非真正推理的问题，构建OpenLM架构，在OpenXOR任务上实现76%准确率，显著超越现有模型（0%）。**

- **链接: [https://arxiv.org/pdf/2511.11712v1](https://arxiv.org/pdf/2511.11712v1)**

> **作者:** Zixi Li
>
> **摘要:** What is reasoning? This question has driven centuries of philosophical inquiry, from Aristotle's syllogisms to modern computational complexity theory. In the age of large language models achieving superhuman performance on benchmarks like GSM8K (95\% accuracy) and HumanEval (90\% pass@1), we must ask: have these systems learned to \emph{reason}, or have they learned to \emph{pattern-match over reasoning traces}? This paper argues for a specific answer: \textbf{reasoning is iterative operator application in state spaces, converging to fixed points}. This definition is not merely philosophical -- it has concrete architectural implications that explain both the failures of current systems and the path to genuine reasoning capabilities. Our investigation begins with a puzzle (OpenXOR), progresses through theory (OpenOperator), and culminates in a working solution (OpenLM) that achieves 76\% accuracy where state-of-the-art LLMs achieve 0\%. This is not about criticizing existing systems, but about \emph{understanding what reasoning requires} and \emph{building architectures that provide it}.
>
---
#### [new 097] Leveraging Large Language Models for Career Mobility Analysis: A Study of Gender, Race, and Job Change Using U.S. Online Resume Profiles
- **分类: cs.CY; cs.CL**

- **简介: 该论文属于职业流动分析任务，旨在研究性别、种族与工作变动对向上职业流动的影响。通过构建基于大语言模型的岗位分类方法FewSOC，解决数据缺失和噪声问题，分析22万条职业轨迹，发现女性和非裔毕业生获益较少，且差异在多层分析中稳健。**

- **链接: [https://arxiv.org/pdf/2511.12010v1](https://arxiv.org/pdf/2511.12010v1)**

> **作者:** Palakorn Achananuparp; Connie Xu; Yao Lu; Xavier Jayaraj Siddarth Ashok; Ee-Peng Lim
>
> **备注:** Submitted to EPJ Data Science
>
> **摘要:** We present a large-scale analysis of career mobility of college-educated U.S. workers using online resume profiles to investigate how gender, race, and job change options are associated with upward mobility. This study addresses key research questions of how the job changes affect their upward career mobility, and how the outcomes of upward career mobility differ by gender and race. We address data challenges -- such as missing demographic attributes, missing wage data, and noisy occupation labels -- through various data processing and Artificial Intelligence (AI) methods. In particular, we develop a large language models (LLMs) based occupation classification method known as FewSOC that achieves accuracy significantly higher than the original occupation labels in the resume dataset. Analysis of 228,710 career trajectories reveals that intra-firm occupation change has been found to facilitate upward mobility most strongly, followed by inter-firm occupation change and inter-firm lateral move. Women and Black college graduates experience significantly lower returns from job changes than men and White peers. Multilevel sensitivity analyses confirm that these disparities are robust to cluster-level heterogeneity and reveal additional intersectional patterns.
>
---
#### [new 098] Small Vocabularies, Big Gains: Pretraining and Tokenization in Time Series Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 论文研究时间序列建模中分词器设计与预训练的关系，解决如何通过小词汇量和预训练提升多模态预测性能的问题。工作包括系统实验与理论分析，发现良好分词器能增强预训练效果，尤其在小词汇量下优势显著。**

- **链接: [https://arxiv.org/pdf/2511.11622v1](https://arxiv.org/pdf/2511.11622v1)**

> **作者:** Alexis Roger; Gwen Legate; Kashif Rasul; Yuriy Nevmyvaka; Irina Rish
>
> **摘要:** Tokenization and transfer learning are two critical components in building state of the art time series foundation models for forecasting. In this work, we systematically study the effect of tokenizer design, specifically scaling and quantization strategies, on model performance, alongside the impact of pretraining versus random initialization. We show that tokenizer configuration primarily governs the representational capacity and stability of the model, while transfer learning influences optimization efficiency and alignment. Using a combination of empirical training experiments and theoretical analyses, we demonstrate that pretrained models consistently leverage well-designed tokenizers more effectively, particularly at smaller vocabulary sizes. Conversely, misaligned tokenization can diminish or even invert the benefits of pretraining. These findings highlight the importance of careful tokenization in time series modeling and suggest that combining small, efficient vocabularies with pretrained weights is especially advantageous in multi-modal forecasting settings, where the overall vocabulary must be shared across modalities. Our results provide concrete guidance for designing tokenizers and leveraging transfer learning in discrete representation learning for continuous signals.
>
---
#### [new 099] Evolving Prompts for Toxicity Search in Large Language Models
- **分类: cs.NE; cs.AI; cs.CL**

- **简介: 论文提出ToxSearch框架，通过演化算法生成对抗性提示以测试大语言模型的毒性内容防御能力。解决的是安全对齐后模型仍易受恶意提示攻击的问题。工作包括设计多样操作符、构建黑盒进化流程，并验证跨模型迁移效果。**

- **链接: [https://arxiv.org/pdf/2511.12487v1](https://arxiv.org/pdf/2511.12487v1)**

> **作者:** Onkar Shelar; Travis Desell
>
> **备注:** pre-print
>
> **摘要:** Large Language Models remain vulnerable to adversarial prompts that elicit toxic content even after safety alignment. We present ToxSearch, a black-box evolutionary framework that tests model safety by evolving prompts in a synchronous steady-state loop. The system employs a diverse set of operators, including lexical substitutions, negation, back-translation, paraphrasing, and two semantic crossover operators, while a moderation oracle provides fitness guidance. Operator-level analysis shows heterogeneous behavior: lexical substitutions offer the best yield-variance trade-off, semantic-similarity crossover acts as a precise low-throughput inserter, and global rewrites exhibit high variance with elevated refusal costs. Using elite prompts evolved on LLaMA 3.1 8B, we observe practically meaningful but attenuated cross-model transfer, with toxicity roughly halving on most targets, smaller LLaMA 3.2 variants showing the strongest resistance, and some cross-architecture models retaining higher toxicity. These results suggest that small, controllable perturbations are effective vehicles for systematic red-teaming and that defenses should anticipate cross-model reuse of adversarial prompts rather than focusing only on single-model hardening.
>
---
#### [new 100] A Structure-Agnostic Co-Tuning Framework for LLMs and SLMs in Cloud-Edge Systems
- **分类: cs.DC; cs.AI; cs.CL; cs.LG**

- **简介: 该论文针对云边系统中大语言模型（LLM）与小语言模型（SLM）协同推理问题，提出结构无关的共训练框架Co-PLMs。通过蒸馏代理模型实现异构模型间知识迁移，在不牺牲设备域特性的前提下提升推理性能。**

- **链接: [https://arxiv.org/pdf/2511.11678v1](https://arxiv.org/pdf/2511.11678v1)**

> **作者:** Yuze Liu; Yunhan Wang; Tiehua Zhang; Zhishu Shen; Cheng Peng; Libing Wu; Feng Xia; Jiong Jin
>
> **摘要:** The surge in intelligent applications driven by large language models (LLMs) has made it increasingly difficult for bandwidth-limited cloud servers to process extensive LLM workloads in real time without compromising user data privacy. To solve these problems, recent research has focused on constructing cloud-edge consortia that integrate server-based LLM with small language models (SLMs) on mobile edge devices. Furthermore, designing collaborative training mechanisms within such consortia to enhance inference performance has emerged as a promising research direction. However, the cross-domain deployment of SLMs, coupled with structural heterogeneity in SLMs architectures, poses significant challenges to enhancing model performance. To this end, we propose Co-PLMs, a novel co-tuning framework for collaborative training of large and small language models, which integrates the process of structure-agnostic mutual learning to realize knowledge exchange between the heterogeneous language models. This framework employs distilled proxy models (DPMs) as bridges to enable collaborative training between the heterogeneous server-based LLM and on-device SLMs, while preserving the domain-specific insights of each device. The experimental results show that Co-PLMs outperform state-of-the-art methods, achieving average increases of 5.38% in Rouge-L and 4.88% in EM.
>
---
#### [new 101] STEP: Success-Rate-Aware Trajectory-Efficient Policy Optimization
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 论文提出STEP框架，解决在线强化学习中多轮交互的样本效率低问题。通过基于成功率动态采样和步骤级优化，提升训练稳定性和收敛速度，适用于复杂任务环境。**

- **链接: [https://arxiv.org/pdf/2511.13091v1](https://arxiv.org/pdf/2511.13091v1)**

> **作者:** Yuhan Chen; Yuxuan Liu; Long Zhang; Pengzhi Gao; Jian Luan; Wei Liu
>
> **摘要:** Multi-turn interaction remains challenging for online reinforcement learning. A common solution is trajectory-level optimization, which treats each trajectory as a single training sample. However, this approach can be inefficient and yield misleading learning signals: it applies uniform sampling across tasks regardless of difficulty, penalizes correct intermediate actions in failed trajectories, and incurs high sample-collection costs. To address these issues, we propose STEP (Success-rate-aware Trajectory-Efficient Policy optimization), a framework that dynamically allocates sampling based on per-task success rates and performs step-level optimization. STEP maintains a smoothed success-rate record to guide adaptive trajectory resampling, allocating more effort to harder tasks. It then computes success-rate-weighted advantages and decomposes trajectories into step-level samples. Finally, it applies a step-level GRPO augmentation to refine updates for low-success tasks. Experiments on OSWorld and AndroidWorld show that STEP substantially improves sample efficiency and training stability over trajectory-level GRPO, converging faster and generalizing better under the same sampling budget.
>
---
#### [new 102] CLINB: A Climate Intelligence Benchmark for Foundational Models
- **分类: cs.AI; cs.CL**

- **简介: 论文提出CLINB基准，用于评估大语言模型在气候科学领域的知识合成与证据支撑能力。针对模型常出现幻觉的问题，通过真实用户问题和专家评审构建多模态评测体系，发现前沿模型虽具高水平理解力但缺乏可靠证据支持，强调需提升可解释性与可信度。**

- **链接: [https://arxiv.org/pdf/2511.11597v1](https://arxiv.org/pdf/2511.11597v1)**

> **作者:** Michelle Chen Huebscher; Katharine Mach; Aleksandar Stanić; Markus Leippold; Ben Gaiarin; Zeke Hausfather; Elisa Rawat; Erich Fischer; Massimiliano Ciaramita; Joeri Rogelj; Christian Buck; Lierni Sestorain Saralegui; Reto Knutti
>
> **备注:** Questions, system prompt and model judge prompts available here: https://www.kaggle.com/datasets/deepmind/clinb-questions
>
> **摘要:** Evaluating how Large Language Models (LLMs) handle complex, specialized knowledge remains a critical challenge. We address this through the lens of climate change by introducing CLINB, a benchmark that assesses models on open-ended, grounded, multimodal question answering tasks with clear requirements for knowledge quality and evidential support. CLINB relies on a dataset of real users' questions and evaluation rubrics curated by leading climate scientists. We implement and validate a model-based evaluation process and evaluate several frontier models. Our findings reveal a critical dichotomy. Frontier models demonstrate remarkable knowledge synthesis capabilities, often exhibiting PhD-level understanding and presentation quality. They outperform "hybrid" answers curated by domain experts assisted by weaker models. However, this performance is countered by failures in grounding. The quality of evidence varies, with substantial hallucination rates for references and images. We argue that bridging this gap between knowledge synthesis and verifiable attribution is essential for the deployment of AI in scientific workflows and that reliable, interpretable benchmarks like CLINB are needed to progress towards building trustworthy AI systems.
>
---
#### [new 103] AutoMalDesc: Large-Scale Script Analysis for Cyber Threat Research
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **简介: 论文提出AutoMalDesc框架，解决自动化恶意脚本分析中缺乏自然语言解释的问题。通过自 paced 学习迭代提升摘要质量，无需大量人工标注，验证显示在多语言样本上效果显著。**

- **链接: [https://arxiv.org/pdf/2511.13333v1](https://arxiv.org/pdf/2511.13333v1)**

> **作者:** Alexandru-Mihai Apostu; Andrei Preda; Alexandra Daniela Damir; Diana Bolocan; Radu Tudor Ionescu; Ioana Croitoru; Mihaela Gaman
>
> **备注:** Accepted at AAAI 2026 (oral)
>
> **摘要:** Generating thorough natural language explanations for threat detections remains an open problem in cybersecurity research, despite significant advances in automated malware detection systems. In this work, we present AutoMalDesc, an automated static analysis summarization framework that, following initial training on a small set of expert-curated examples, operates independently at scale. This approach leverages an iterative self-paced learning pipeline to progressively enhance output quality through synthetic data generation and validation cycles, eliminating the need for extensive manual data annotation. Evaluation across 3,600 diverse samples in five scripting languages demonstrates statistically significant improvements between iterations, showing consistent gains in both summary quality and classification accuracy. Our comprehensive validation approach combines quantitative metrics based on established malware labels with qualitative assessment from both human experts and LLM-based judges, confirming both technical precision and linguistic coherence of generated summaries. To facilitate reproducibility and advance research in this domain, we publish our complete dataset of more than 100K script samples, including annotated seed (0.9K) and test (3.6K) datasets, along with our methodology and evaluation framework.
>
---
#### [new 104] Automatic generation of DRI Statements
- **分类: cs.CY; cs.CL**

- **简介: 论文提出自动生成DRI陈述的方法，解决人工生成耗时问题。利用NLP和大语言模型构建自动化框架，降低研究门槛，提供可复现的AI融合社会科学研究方案。**

- **链接: [https://arxiv.org/pdf/2511.11655v1](https://arxiv.org/pdf/2511.11655v1)**

> **作者:** Maurice Flechtner
>
> **备注:** Master Thesis
>
> **摘要:** Assessing the quality of group deliberation is essential for improving our understanding of deliberative processes. The Deliberative Reason Index (DRI) offers a sophisticated metric for evaluating group reasoning, but its implementation has been constrained by the complex and time-consuming process of statement generation. This thesis introduces an innovative, automated approach to DRI statement generation that leverages advanced natural language processing (NLP) and large language models (LLMs) to substantially reduce the human effort involved in survey preparation. Key contributions are a systematic framework for automated DRI statement generation and a methodological innovation that significantly lowers the barrier to conducting comprehensive deliberative process assessments. In addition, the findings provide a replicable template for integrating generative artificial intelligence into social science research methodologies.
>
---
#### [new 105] SynBullying: A Multi LLM Synthetic Conversational Dataset for Cyberbullying Detectio
- **分类: cs.AI; cs.CL; cs.CY**

- **简介: 论文提出SynBullying数据集，用于网络欺凌检测任务。通过多大语言模型生成对话式合成数据，解决真实数据收集难、伦理问题，提供结构化、细粒度标注，支持检测模型训练与增强。**

- **链接: [https://arxiv.org/pdf/2511.11599v1](https://arxiv.org/pdf/2511.11599v1)**

> **作者:** Arefeh Kazemi; Hamza Qadeer; Joachim Wagner; Hossein Hosseini; Sri Balaaji Natarajan Kalaivendan; Brian Davis
>
> **摘要:** We introduce SynBullying, a synthetic multi-LLM conversational dataset for studying and detecting cyberbullying (CB). SynBullying provides a scalable and ethically safe alternative to human data collection by leveraging large language models (LLMs) to simulate realistic bullying interactions. The dataset offers (i) conversational structure, capturing multi-turn exchanges rather than isolated posts; (ii) context-aware annotations, where harmfulness is assessed within the conversational flow considering context, intent, and discourse dynamics; and (iii) fine-grained labeling, covering various CB categories for detailed linguistic and behavioral analysis. We evaluate SynBullying across five dimensions, including conversational structure, lexical patterns, sentiment/toxicity, role dynamics, harm intensity, and CB-type distribution. We further examine its utility by testing its performance as standalone training data and as an augmentation source for CB classification.
>
---
#### [new 106] EduAgentQG: A Multi-Agent Workflow Framework for Personalized Question Generation
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 论文提出EduAgentQG框架，用于个性化试题生成任务，解决现有方法质量不稳定、多样性不足和教育目标对齐差的问题。通过五代理协同与迭代反馈机制，提升题目质量、多样性和教学一致性。**

- **链接: [https://arxiv.org/pdf/2511.11635v1](https://arxiv.org/pdf/2511.11635v1)**

> **作者:** Rui Jia; Min Zhang; Fengrui Liu; Bo Jiang; Kun Kuang; Zhongxiang Dai
>
> **摘要:** High-quality personalized question banks are crucial for supporting adaptive learning and individualized assessment. Manually designing questions is time-consuming and often fails to meet diverse learning needs, making automated question generation a crucial approach to reduce teachers' workload and improve the scalability of educational resources. However, most existing question generation methods rely on single-agent or rule-based pipelines, which still produce questions with unstable quality, limited diversity, and insufficient alignment with educational goals. To address these challenges, we propose EduAgentQG, a multi-agent collaborative framework for generating high-quality and diverse personalized questions. The framework consists of five specialized agents and operates through an iterative feedback loop: the Planner generates structured design plans and multiple question directions to enhance diversity; the Writer produces candidate questions based on the plan and optimizes their quality and diversity using feedback from the Solver and Educator; the Solver and Educator perform binary scoring across multiple evaluation dimensions and feed the evaluation results back to the Writer; the Checker conducts final verification, including answer correctness and clarity, ensuring alignment with educational goals. Through this multi-agent collaboration and iterative feedback loop, EduAgentQG generates questions that are both high-quality and diverse, while maintaining consistency with educational objectives. Experiments on two mathematics question datasets demonstrate that EduAgentQG outperforms existing single-agent and multi-agent methods in terms of question diversity, goal consistency, and overall quality.
>
---
#### [new 107] Co-Layout: LLM-driven Co-optimization for Interior Layout
- **分类: cs.CV; cs.CL; cs.GR**

- **简介: 论文提出Co-Layout框架，融合大语言模型与整数规划，联合优化室内布局与家具摆放。解决传统两阶段方法效率低、质量差的问题，通过文本提示生成约束并采用粗到精策略提升计算效率与设计质量。**

- **链接: [https://arxiv.org/pdf/2511.12474v1](https://arxiv.org/pdf/2511.12474v1)**

> **作者:** Chucheng Xiang; Ruchao Bao; Biyin Feng; Wenzheng Wu; Zhongyuan Liu; Yirui Guan; Ligang Liu
>
> **摘要:** We present a novel framework for automated interior design that combines large language models (LLMs) with grid-based integer programming to jointly optimize room layout and furniture placement. Given a textual prompt, the LLM-driven agent workflow extracts structured design constraints related to room configurations and furniture arrangements. These constraints are encoded into a unified grid-based representation inspired by ``Modulor". Our formulation accounts for key design requirements, including corridor connectivity, room accessibility, spatial exclusivity, and user-specified preferences. To improve computational efficiency, we adopt a coarse-to-fine optimization strategy that begins with a low-resolution grid to solve a simplified problem and guides the solution at the full resolution. Experimental results across diverse scenarios demonstrate that our joint optimization approach significantly outperforms existing two-stage design pipelines in solution quality, and achieves notable computational efficiency through the coarse-to-fine strategy.
>
---
#### [new 108] A Content-Preserving Secure Linguistic Steganography
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于语言隐写任务，旨在解决现有方法因修改原文导致安全风险的问题。提出CLstega方法，通过可控分布变换嵌入秘密信息，不改动原文本，实现完美安全与高提取成功率。**

- **链接: [https://arxiv.org/pdf/2511.12565v1](https://arxiv.org/pdf/2511.12565v1)**

> **作者:** Lingyun Xiang; Chengfu Ou; Xu He; Zhongliang Yang; Yuling Liu
>
> **备注:** This is the extended version of the paper accepted to AAAI 2026
>
> **摘要:** Existing linguistic steganography methods primarily rely on content transformations to conceal secret messages. However, they often cause subtle yet looking-innocent deviations between normal and stego texts, posing potential security risks in real-world applications. To address this challenge, we propose a content-preserving linguistic steganography paradigm for perfectly secure covert communication without modifying the cover text. Based on this paradigm, we introduce CLstega (\textit{C}ontent-preserving \textit{L}inguistic \textit{stega}nography), a novel method that embeds secret messages through controllable distribution transformation. CLstega first applies an augmented masking strategy to locate and mask embedding positions, where MLM(masked language model)-predicted probability distributions are easily adjustable for transformation. Subsequently, a dynamic distribution steganographic coding strategy is designed to encode secret messages by deriving target distributions from the original probability distributions. To achieve this transformation, CLstega elaborately selects target words for embedding positions as labels to construct a masked sentence dataset, which is used to fine-tune the original MLM, producing a target MLM capable of directly extracting secret messages from the cover text. This approach ensures perfect security of secret messages while fully preserving the integrity of the original cover text. Experimental results show that CLstega can achieve a 100\% extraction success rate, and outperforms existing methods in security, effectively balancing embedding capacity and security.
>
---
#### [new 109] H-Model: Dynamic Neural Architectures for Adaptive Processing
- **分类: cs.LG; cs.CL**

- **简介: 论文提出H-Model，一种动态神经架构，通过路由机制使网络根据输入数据自适应调整内部结构，解决传统模型计算结构固定的问题。工作聚焦于探索可学习计算结构的架构框架，而非性能优化。**

- **链接: [https://arxiv.org/pdf/2511.11669v1](https://arxiv.org/pdf/2511.11669v1)**

> **作者:** Dmytro Hospodarchuk
>
> **备注:** Independent research report, 24 pages including references and figures
>
> **摘要:** This article explores the design and experimentation of a neural network architecture capable of dynamically adjusting its internal structure based on the input data. The proposed model introduces a routing mechanism that allows each layer to influence how its outputs are propagated through the network, enabling iterative and adaptive computation. This concept is loosely inspired by the idea of thought processes and dynamic reasoning, where information flow is conditioned not only on the data itself, but also on the internal state of the system. It is important to note that this work does not aim to compete with state-of-the-art language models in terms of performance. Instead, it presents a conceptual prototype-an architectural framework that opens up a new direction for exploring adaptable and potentially more interpretable networks. The goal is not optimization of existing benchmarks but rather the proposal of a system that can learn not only representations, but also the structure of computation itself. Due to practical constraints in computing resources and data, this study remains a preliminary investigation. Nevertheless, initial observations show promise, and the architecture's full potential can only be evaluated in future experiments under more favorable computational conditions.
>
---
#### [new 110] Characterizing and Understanding Energy Footprint and Efficiency of Small Language Model on Edges
- **分类: cs.DC; cs.AI; cs.CL; cs.LG**

- **简介: 论文研究边缘设备上小语言模型的能效问题，旨在优化准确率、延迟与功耗之间的权衡。通过实测五种模型在不同硬件上的表现，发现GPU加速和模型架构对能效影响显著，为边缘AI部署提供实践指导。**

- **链接: [https://arxiv.org/pdf/2511.11624v1](https://arxiv.org/pdf/2511.11624v1)**

> **作者:** Md Romyull Islam; Bobin Deng; Nobel Dhar; Tu N. Nguyen; Selena He; Yong Shi; Kun Suo
>
> **备注:** Submitted version; 9 pages, 5 figures; presented at IEEE MASS 2025 (online publication pending)
>
> **摘要:** Cloud-based large language models (LLMs) and their variants have significantly influenced real-world applications. Deploying smaller models (i.e., small language models (SLMs)) on edge devices offers additional advantages, such as reduced latency and independence from network connectivity. However, edge devices' limited computing resources and constrained energy budgets challenge efficient deployment. This study evaluates the power efficiency of five representative SLMs - Llama 3.2, Phi-3 Mini, TinyLlama, and Gemma 2 on Raspberry Pi 5, Jetson Nano, and Jetson Orin Nano (CPU and GPU configurations). Results show that Jetson Orin Nano with GPU acceleration achieves the highest energy-to-performance ratio, significantly outperforming CPU-based setups. Llama 3.2 provides the best balance of accuracy and power efficiency, while TinyLlama is well-suited for low-power environments at the cost of reduced accuracy. In contrast, Phi-3 Mini consumes the most energy despite its high accuracy. In addition, GPU acceleration, memory bandwidth, and model architecture are key in optimizing inference energy efficiency. Our empirical analysis offers practical insights for AI, smart systems, and mobile ad-hoc platforms to leverage tradeoffs from accuracy, inference latency, and power efficiency in energy-constrained environments.
>
---
#### [new 111] P1: Mastering Physics Olympiads with Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 论文提出P1系列模型，通过强化学习训练，专注于提升物理奥赛级别的推理能力。解决了开放源代码模型在科学级推理中表现不足的问题，实现了在国际物理奥赛中的金牌水平，并在多个竞赛中取得领先。**

- **链接: [https://arxiv.org/pdf/2511.13612v1](https://arxiv.org/pdf/2511.13612v1)**

> **作者:** Jiacheng Chen; Qianjia Cheng; Fangchen Yu; Haiyuan Wan; Yuchen Zhang; Shenghe Zheng; Junchi Yao; Qingyang Zhang; Haonan He; Yun Luo; Yufeng Zhao; Futing Wang; Li Sheng; Chengxing Xie; Yuxin Zuo; Yizhuo Li; Wenxauan Zeng; Yulun Wu; Rui Huang; Dongzhan Zhou; Kai Chen; Yu Qiao; Lei Bai; Yu Cheng; Ning Ding; Bowen Zhou; Peng Ye; Ganqu Cui
>
> **摘要:** Recent progress in large language models (LLMs) has moved the frontier from puzzle-solving to science-grade reasoning-the kind needed to tackle problems whose answers must stand against nature, not merely fit a rubric. Physics is the sharpest test of this shift, which binds symbols to reality in a fundamental way, serving as the cornerstone of most modern technologies. In this work, we manage to advance physics research by developing large language models with exceptional physics reasoning capabilities, especially excel at solving Olympiad-level physics problems. We introduce P1, a family of open-source physics reasoning models trained entirely through reinforcement learning (RL). Among them, P1-235B-A22B is the first open-source model with Gold-medal performance at the latest International Physics Olympiad (IPhO 2025), and wins 12 gold medals out of 13 international/regional physics competitions in 2024/2025. P1-30B-A3B also surpasses almost all other open-source models on IPhO 2025, getting a silver medal. Further equipped with an agentic framework PhysicsMinions, P1-235B-A22B+PhysicsMinions achieves overall No.1 on IPhO 2025, and obtains the highest average score over the 13 physics competitions. Besides physics, P1 models also present great performance on other reasoning tasks like math and coding, showing the great generalibility of P1 series.
>
---
#### [new 112] Computational Measurement of Political Positions: A Review of Text-Based Ideal Point Estimation Algorithms
- **分类: cs.LG; cs.AI; cs.CL; cs.CY**

- **简介: 论文属于文本-based理想点估计任务，旨在系统梳理过去二十年的CT-IPE算法，解决方法碎片化、缺乏比较框架的问题。作者识别25种算法，提出分类框架，分析其假设与局限，并提供应用指导。**

- **链接: [https://arxiv.org/pdf/2511.13238v1](https://arxiv.org/pdf/2511.13238v1)**

> **作者:** Patrick Parschan; Charlott Jakob
>
> **备注:** 46 pages, 8 figures, 2 tables, accepted for publication in Quality & Quantity
>
> **摘要:** This article presents the first systematic review of unsupervised and semi-supervised computational text-based ideal point estimation (CT-IPE) algorithms, methods designed to infer latent political positions from textual data. These algorithms are widely used in political science, communication, computational social science, and computer science to estimate ideological preferences from parliamentary speeches, party manifestos, and social media. Over the past two decades, their development has closely followed broader NLP trends -- beginning with word-frequency models and most recently turning to large language models (LLMs). While this trajectory has greatly expanded the methodological toolkit, it has also produced a fragmented field that lacks systematic comparison and clear guidance for applied use. To address this gap, we identified 25 CT-IPE algorithms through a systematic literature review and conducted a manual content analysis of their modeling assumptions and development contexts. To compare them meaningfully, we introduce a conceptual framework that distinguishes how algorithms generate, capture, and aggregate textual variance. On this basis, we identify four methodological families -- word-frequency, topic modeling, word embedding, and LLM-based approaches -- and critically assess their assumptions, interpretability, scalability, and limitations. Our review offers three contributions. First, it provides a structured synthesis of two decades of algorithm development, clarifying how diverse methods relate to one another. Second, it translates these insights into practical guidance for applied researchers, highlighting trade-offs in transparency, technical requirements, and validation strategies that shape algorithm choice. Third, it emphasizes that differences in estimation outcomes across algorithms are themselves informative, underscoring the need for systematic benchmarking.
>
---
#### [new 113] The Anatomy of a Triton Attention Kernel
- **分类: cs.LG; cs.AI; cs.CL; cs.DC; cs.PL**

- **简介: 论文提出基于Triton语言的高效可移植注意力核，解决LLM推理在不同GPU上性能差异大的问题。通过算法优化与自动调参，使性能达到业界最优水平的105.9%，实现跨平台高效部署。**

- **链接: [https://arxiv.org/pdf/2511.11581v1](https://arxiv.org/pdf/2511.11581v1)**

> **作者:** Burkhard Ringlein; Jan van Lunteren; Radu Stoica; Thomas Parnell
>
> **摘要:** A long-standing goal in both industry and academia is to develop an LLM inference platform that is portable across hardware architectures, eliminates the need for low-level hand-tuning, and still delivers best-in-class efficiency. In this work, we demonstrate that portable, efficient cross-platform LLM inference is indeed possible and share our experience. We develop a state-of-the-art paged attention kernel, the core performance-critical component of many LLM deployments, that builds exclusively on the domain-specific just-in-time compiled language Triton to achieve state-of-the-art performance on both NVIDIA and AMD GPUs. We describe our high-level approach, the key algorithmic and system-level improvements, the parameter auto-tuning required to unlock efficiency, and the integrations into a popular inference server that are necessary to bring the performance of a generic Triton attention kernel from 19.7% of the state-of-the-art to 105.9%. Our results highlight how open-source domain-specific languages can be leveraged to unlock model portability across different GPU vendors.
>
---
#### [new 114] VoiceCraft-X: Unifying Multilingual, Voice-Cloning Speech Synthesis and Speech Editing
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文提出VoiceCraft-X，统一处理多语言语音合成与编辑任务。针对跨语言文本处理和有限数据挑战，利用Qwen3模型与新型令牌重排序机制，实现零样本TTS与语音编辑一体化。**

- **链接: [https://arxiv.org/pdf/2511.12347v1](https://arxiv.org/pdf/2511.12347v1)**

> **作者:** Zhisheng Zheng; Puyuan Peng; Anuj Diwan; Cong Phuoc Huynh; Xiaohang Sun; Zhu Liu; Vimal Bhat; David Harwath
>
> **备注:** EMNLP 2025. Demo and code are available at https://zhishengzheng.com/voicecraft-x/
>
> **摘要:** We introduce VoiceCraft-X, an autoregressive neural codec language model which unifies multilingual speech editing and zero-shot Text-to-Speech (TTS) synthesis across 11 languages: English, Mandarin, Korean, Japanese, Spanish, French, German, Dutch, Italian, Portuguese, and Polish. VoiceCraft-X utilizes the Qwen3 large language model for phoneme-free cross-lingual text processing and a novel token reordering mechanism with time-aligned text and speech tokens to handle both tasks as a single sequence generation problem. The model generates high-quality, natural-sounding speech, seamlessly creating new audio or editing existing recordings within one framework. VoiceCraft-X shows robust performance in diverse linguistic settings, even with limited per-language data, underscoring the power of unified autoregressive approaches for advancing complex, real-world multilingual speech applications. Audio samples are available at https://zhishengzheng.com/voicecraft-x/.
>
---
#### [new 115] Generative AI as a Linguistic Equalizer in Global Science
- **分类: cs.CY; cs.CL**

- **简介: 论文研究生成式AI是否能降低全球科学交流中的语言壁垒。通过分析565万篇论文，发现GenAI辅助的非英语作者文章在风格上逐渐趋近英语母语者，尤其对语言差异大的国家效果显著，表明GenAI正成为语言平等工具。**

- **链接: [https://arxiv.org/pdf/2511.11687v1](https://arxiv.org/pdf/2511.11687v1)**

> **作者:** Dragan Filimonovic; Christian Rutzer; Jeffrey Macher; Rolf Weder
>
> **摘要:** For decades, the dominance of English has created a substantial barrier in global science, disadvantaging non-native speakers. The recent rise of generative AI (GenAI) offers a potential technological response to this long-standing inequity. We provide the first large-scale evidence testing whether GenAI acts as a linguistic equalizer in global science. Drawing on 5.65 million scientific articles published from 2021 to 2024, we compare GenAI-assisted and non-assisted publications from authors in non-English-speaking countries. Using text embeddings derived from a pretrained large language model (SciBERT), we measure each publication's linguistic similarity to a benchmark of scientific writing from U.S.-based authors and track stylistic convergence over time. We find significant and growing convergence for GenAI-assisted publications after the release of ChatGPT in late 2022. The effect is strongest for domestic coauthor teams from countries linguistically distant from English. These findings provide large-scale evidence that GenAI is beginning to reshape global science communication by reducing language barriers in research.
>
---
#### [new 116] Accepted with Minor Revisions: Value of AI-Assisted Scientific Writing
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 论文研究AI辅助科学写作任务，解决AI生成文本在学术场景中的接受度问题。通过随机对照实验发现，作者更倾向修改无署名AI摘要，但披露来源后编辑行为趋同；审稿人关注编辑量而非来源，风格优化提升接受率。**

- **链接: [https://arxiv.org/pdf/2511.12529v1](https://arxiv.org/pdf/2511.12529v1)**

> **作者:** Sanchaita Hazra; Doeun Lee; Bodhisattwa Prasad Majumder; Sachin Kumar
>
> **摘要:** Large Language Models have seen expanding application across domains, yet their effectiveness as assistive tools for scientific writing -- an endeavor requiring precision, multimodal synthesis, and domain expertise -- remains insufficiently understood. We examine the potential of LLMs to support domain experts in scientific writing, with a focus on abstract composition. We design an incentivized randomized controlled trial with a hypothetical conference setup where participants with relevant expertise are split into an author and reviewer pool. Inspired by methods in behavioral science, our novel incentive structure encourages authors to edit the provided abstracts to an acceptable quality for a peer-reviewed submission. Our 2x2 between-subject design expands into two dimensions: the implicit source of the provided abstract and the disclosure of it. We find authors make most edits when editing human-written abstracts compared to AI-generated abstracts without source attribution, often guided by higher perceived readability in AI generation. Upon disclosure of source information, the volume of edits converges in both source treatments. Reviewer decisions remain unaffected by the source of the abstract, but bear a significant correlation with the number of edits made. Careful stylistic edits, especially in the case of AI-generated abstracts, in the presence of source information, improve the chance of acceptance. We find that AI-generated abstracts hold potential to reach comparable levels of acceptability to human-written ones with minimal revision, and that perceptions of AI authorship, rather than objective quality, drive much of the observed editing behavior. Our findings reverberate the significance of source disclosure in collaborative scientific writing.
>
---
#### [new 117] PragWorld: A Benchmark Evaluating LLMs' Local World Model under Minimal Linguistic Alterations and Conversational Dynamics
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于语言模型对话理解任务，旨在评估大模型在最小语言变化下维持局部世界模型的能力。作者构建了两个基准测试集，发现模型在跟踪实体和应对语言扰动时表现不佳，并提出双视角解释框架与层正则化微调策略以提升性能。**

- **链接: [https://arxiv.org/pdf/2511.13021v1](https://arxiv.org/pdf/2511.13021v1)**

> **作者:** Sachin Vashistha; Aryan Bibhuti; Atharva Naik; Martin Tutek; Somak Aditya
>
> **备注:** 23 pages, 15 tables, 10 figures; AAAI 2026 Conference Main Track (oral)
>
> **摘要:** Real-world conversations are rich with pragmatic elements, such as entity mentions, references, and implicatures. Understanding such nuances is a requirement for successful natural communication, and often requires building a local world model which encodes such elements and captures the dynamics of their evolving states. However, it is not well-understood whether language models (LMs) construct or maintain a robust implicit representation of conversations. In this work, we evaluate the ability of LMs to encode and update their internal world model in dyadic conversations and test their malleability under linguistic alterations. To facilitate this, we apply seven minimal linguistic alterations to conversations sourced from popular datasets and construct two benchmarks comprising yes-no questions. We evaluate a wide range of open and closed source LMs and observe that they struggle to maintain robust accuracy. Our analysis unveils that LMs struggle to memorize crucial details, such as tracking entities under linguistic alterations to conversations. We then propose a dual-perspective interpretability framework which identifies transformer layers that are useful or harmful and highlights linguistic alterations most influenced by harmful layers, typically due to encoding spurious signals or relying on shortcuts. Inspired by these insights, we propose two layer-regularization based fine-tuning strategies that suppress the effect of the harmful layers.
>
---
#### [new 118] Decoupling Positional and Symbolic Attention Behavior in Transformers
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 论文研究Transformer中位置与符号信息的解耦问题，针对RoPE编码机制，提出理论定义和量化方法，证明注意力头的行为互斥，并通过任务设计验证频率控制可调节模型性能。**

- **链接: [https://arxiv.org/pdf/2511.11579v1](https://arxiv.org/pdf/2511.11579v1)**

> **作者:** Felipe Urrutia; Jorge Salas; Alexander Kozachinskiy; Cristian Buc Calderon; Hector Pasten; Cristobal Rojas
>
> **备注:** 32 pages, 12 figures, repository available
>
> **摘要:** An important aspect subtending language understanding and production is the ability to independently encode positional and symbolic information of the words within a sentence. In Transformers, positional information is typically encoded using Positional Encodings (PEs). One such popular PE, namely Rotary PE (RoPE), has been widely used due to its empirical success. Recently, it has been argued that part of RoPE's success emerges from its ability to encode robust positional and semantic information using large and small frequencies, respectively. In this work, we perform a deeper dive into the positional versus symbolic dichotomy of attention heads behavior, both at the theoretical and empirical level. We provide general definitions of what it means for a head to behave positionally or symbolically, prove that these are two mutually exclusive behaviors and develop a metric to quantify them. We apply our framework to analyze Transformer-based LLMs using RoPE and find that all heads exhibit a strong correspondence between behavior and frequency use. Finally, we introduce canonical tasks designed to be either purely positional or symbolic, and demonstrate that the Transformer performance causally relates to the ability of attention heads to leverage the appropriate frequencies. In particular, we show that we can control the Transformer performance by controlling which frequencies the attention heads can access. Altogether, our work provides a detailed understanding of RoPE, and how its properties relate to model behavior.
>
---
#### [new 119] Exploring Multi-Table Retrieval Through Iterative Search
- **分类: cs.IR; cs.AI; cs.CL; cs.DB; cs.LG**

- **简介: 论文研究开放域问答中的多表检索任务，解决如何高效获取语义相关且结构连贯（可连接）的多表组合问题。提出迭代搜索框架与快速贪心算法，在保证检索效果的同时显著提升速度。**

- **链接: [https://arxiv.org/pdf/2511.13418v1](https://arxiv.org/pdf/2511.13418v1)**

> **作者:** Allaa Boutaleb; Bernd Amann; Rafael Angarita; Hubert Naacke
>
> **备注:** Accepted @ the AI for Tabular Data Workshop, EurIPS 2025
>
> **摘要:** Open-domain question answering over datalakes requires retrieving and composing information from multiple tables, a challenging subtask that demands semantic relevance and structural coherence (e.g., joinability). While exact optimization methods like Mixed-Integer Programming (MIP) can ensure coherence, their computational complexity is often prohibitive. Conversely, simpler greedy heuristics that optimize for query coverage alone often fail to find these coherent, joinable sets. This paper frames multi-table retrieval as an iterative search process, arguing this approach offers advantages in scalability, interpretability, and flexibility. We propose a general framework and a concrete instantiation: a fast, effective Greedy Join-Aware Retrieval algorithm that holistically balances relevance, coverage, and joinability. Experiments across 5 NL2SQL benchmarks demonstrate that our iterative method achieves competitive retrieval performance compared to the MIP-based approach while being 4-400x faster depending on the benchmark and search space settings. This work highlights the potential of iterative heuristics for practical, scalable, and composition-aware retrieval.
>
---
#### [new 120] ForgeDAN: An Evolutionary Framework for Jailbreaking Aligned Large Language Models
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文提出ForgeDAN，一种用于攻击对齐大语言模型的进化框架，解决现有方法多样性不足、评估浅显和检测脆弱的问题。通过多层级文本扰动、语义相似度评估和双维判断机制，提升攻击效果与隐蔽性。**

- **链接: [https://arxiv.org/pdf/2511.13548v1](https://arxiv.org/pdf/2511.13548v1)**

> **作者:** Siyang Cheng; Gaotian Liu; Rui Mei; Yilin Wang; Kejia Zhang; Kaishuo Wei; Yuqi Yu; Weiping Wen; Xiaojie Wu; Junhua Liu
>
> **摘要:** The rapid adoption of large language models (LLMs) has brought both transformative applications and new security risks, including jailbreak attacks that bypass alignment safeguards to elicit harmful outputs. Existing automated jailbreak generation approaches e.g. AutoDAN, suffer from limited mutation diversity, shallow fitness evaluation, and fragile keyword-based detection. To address these limitations, we propose ForgeDAN, a novel evolutionary framework for generating semantically coherent and highly effective adversarial prompts against aligned LLMs. First, ForgeDAN introduces multi-strategy textual perturbations across \textit{character, word, and sentence-level} operations to enhance attack diversity; then we employ interpretable semantic fitness evaluation based on a text similarity model to guide the evolutionary process toward semantically relevant and harmful outputs; finally, ForgeDAN integrates dual-dimensional jailbreak judgment, leveraging an LLM-based classifier to jointly assess model compliance and output harmfulness, thereby reducing false positives and improving detection effectiveness. Our evaluation demonstrates ForgeDAN achieves high jailbreaking success rates while maintaining naturalness and stealth, outperforming existing SOTA solutions.
>
---
#### [new 121] Dropouts in Confidence: Moral Uncertainty in Human-LLM Alignment
- **分类: cs.AI; cs.CL; cs.CY**

- **简介: 该论文研究大语言模型（LLM）在道德困境中的不确定性问题，旨在提升人机道德对齐。通过分析32个模型和9种道德维度，发现模型间差异大于道德维度间差异；引入推理时dropout机制增加不确定性，显著提高与人类道德判断的一致性。**

- **链接: [https://arxiv.org/pdf/2511.13290v1](https://arxiv.org/pdf/2511.13290v1)**

> **作者:** Jea Kwon; Luiz Felipe Vecchietti; Sungwon Park; Meeyoung Cha
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** Humans display significant uncertainty when confronted with moral dilemmas, yet the extent of such uncertainty in machines and AI agents remains underexplored. Recent studies have confirmed the overly confident tendencies of machine-generated responses, particularly in large language models (LLMs). As these systems are increasingly embedded in ethical decision-making scenarios, it is important to understand their moral reasoning and the inherent uncertainties in building reliable AI systems. This work examines how uncertainty influences moral decisions in the classical trolley problem, analyzing responses from 32 open-source models and 9 distinct moral dimensions. We first find that variance in model confidence is greater across models than within moral dimensions, suggesting that moral uncertainty is predominantly shaped by model architecture and training method. To quantify uncertainty, we measure binary entropy as a linear combination of total entropy, conditional entropy, and mutual information. To examine its effects, we introduce stochasticity into models via "dropout" at inference time. Our findings show that our mechanism increases total entropy, mainly through a rise in mutual information, while conditional entropy remains largely unchanged. Moreover, this mechanism significantly improves human-LLM moral alignment, with correlations in mutual information and alignment score shifts. Our results highlight the potential to better align model-generated decisions and human preferences by deliberately modulating uncertainty and reducing LLMs' confidence in morally complex scenarios.
>
---
#### [new 122] D$^{3}$ToM: Decider-Guided Dynamic Token Merging for Accelerating Diffusion MLLMs
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 论文提出D³ToM方法，用于加速基于扩散的多模态大模型（Diffusion MLLMs）的推理。针对其因全序列自注意力导致的计算复杂度高问题，该方法通过决策器引导动态合并冗余视觉token，在不改变模型参数的前提下缩短序列长度，实现高效且高性能的生成。**

- **链接: [https://arxiv.org/pdf/2511.12280v1](https://arxiv.org/pdf/2511.12280v1)**

> **作者:** Shuochen Chang; Xiaofeng Zhang; Qingyang Liu; Li Niu
>
> **备注:** Accepted by AAAI Conference on Artificial Intelligence (AAAI) 2026. Code available at https://github.com/bcmi/D3ToM-Diffusion-MLLM
>
> **摘要:** Diffusion-based multimodal large language models (Diffusion MLLMs) have recently demonstrated impressive non-autoregressive generative capabilities across vision-and-language tasks. However, Diffusion MLLMs exhibit substantially slower inference than autoregressive models: Each denoising step employs full bidirectional self-attention over the entire sequence, resulting in cubic decoding complexity that becomes computationally impractical with thousands of visual tokens. To address this challenge, we propose D$^{3}$ToM, a Decider-guided dynamic token merging method that dynamically merges redundant visual tokens at different denoising steps to accelerate inference in Diffusion MLLMs. At each denoising step, D$^{3}$ToM uses decider tokens-the tokens generated in the previous denoising step-to build an importance map over all visual tokens. Then it maintains a proportion of the most salient tokens and merges the remainder through similarity-based aggregation. This plug-and-play module integrates into a single transformer layer, physically shortening the visual token sequence for all subsequent layers without altering model parameters. Moreover, D$^{3}$ToM employs a merge ratio that dynamically varies with each denoising step, aligns with the native decoding process of Diffusion MLLMs, achieving superior performance under equivalent computational budgets. Extensive experiments show that D$^{3}$ToM accelerates inference while preserving competitive performance. The code is released at https://github.com/bcmi/D3ToM-Diffusion-MLLM.
>
---
#### [new 123] Preference Learning from Physics-Based Feedback: Tuning Language Models to Design BCC/B2 Superalloys
- **分类: cs.CE; cond-mat.mtrl-sci; cs.AI; cs.CL; cs.LG**

- **简介: 论文将偏好学习用于语言模型引导设计BCC/B2超合金，通过热力学相计算构建统一奖励信号，优化模型以满足合成可行性等多目标，首次实现基于物理反馈的材料设计。**

- **链接: [https://arxiv.org/pdf/2511.12036v1](https://arxiv.org/pdf/2511.12036v1)**

> **作者:** Satanu Ghosh; Collin Holgate; Neal R. Brodnik; Doug Downey; Samantha Daly; Tresa M. Pollock; Samuel Carton
>
> **摘要:** We apply preference learning to the task of language model-guided design of novel structural alloys. In contrast to prior work that focuses on generating stable inorganic crystals, our approach targets the synthesizeability of a specific structural class: BCC/B2 superalloys, an underexplored family of materials with potential applications in extreme environments. Using three open-weight models (LLaMA-3.1, Gemma-2, and OLMo-2), we demonstrate that language models can be optimized for multiple design objectives using a single, unified reward signal through Direct Preference Optimization (DPO). Unlike prior approaches that rely on heuristic or human-in-the-loop feedback (costly), our reward signal is derived from thermodynamic phase calculations, offering a scientifically grounded criterion for model tuning. To our knowledge, this is the first demonstration of preference-tuning a language model using physics-grounded feedback for structural alloy design. The resulting framework is general and extensible, providing a path forward for intelligent design-space exploration across a range of physical science domains.
>
---
#### [new 124] Attention Grounded Enhancement for Visual Document Retrieval
- **分类: cs.IR; cs.CL; cs.CV**

- **简介: 该论文针对视觉文档检索任务，解决现有方法依赖全局标签导致难以捕捉语义关联的问题。提出AGREE框架，利用跨模态注意力作为局部监督信号，联合优化全局与局部信号，提升检索的准确性与可解释性。**

- **链接: [https://arxiv.org/pdf/2511.13415v1](https://arxiv.org/pdf/2511.13415v1)**

> **作者:** Wanqing Cui; Wei Huang; Yazhi Guo; Yibo Hu; Meiguang Jin; Junfeng Ma; Keping Bi
>
> **摘要:** Visual document retrieval requires understanding heterogeneous and multi-modal content to satisfy information needs. Recent advances use screenshot-based document encoding with fine-grained late interaction, significantly improving retrieval performance. However, retrievers are still trained with coarse global relevance labels, without revealing which regions support the match. As a result, retrievers tend to rely on surface-level cues and struggle to capture implicit semantic connections, hindering their ability to handle non-extractive queries. To alleviate this problem, we propose a \textbf{A}ttention-\textbf{G}rounded \textbf{RE}triever \textbf{E}nhancement (AGREE) framework. AGREE leverages cross-modal attention from multimodal large language models as proxy local supervision to guide the identification of relevant document regions. During training, AGREE combines local signals with the global signals to jointly optimize the retriever, enabling it to learn not only whether documents match, but also which content drives relevance. Experiments on the challenging ViDoRe V2 benchmark show that AGREE significantly outperforms the global-supervision-only baseline. Quantitative and qualitative analyses further demonstrate that AGREE promotes deeper alignment between query terms and document regions, moving beyond surface-level matching toward more accurate and interpretable retrieval. Our code is available at: https://anonymous.4open.science/r/AGREE-2025.
>
---
## 更新

#### [replaced 001] Multimodal DeepResearcher: Generating Text-Chart Interleaved Reports From Scratch with Agentic Framework
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2506.02454v2](https://arxiv.org/pdf/2506.02454v2)**

> **作者:** Zhaorui Yang; Bo Pan; Han Wang; Yiyao Wang; Xingyu Liu; Luoxuan Weng; Yingchaojie Feng; Haozhe Feng; Minfeng Zhu; Bo Zhang; Wei Chen
>
> **备注:** AAAI 2026 Oral
>
> **摘要:** Visualizations play a crucial part in effective communication of concepts and information. Recent advances in reasoning and retrieval augmented generation have enabled Large Language Models (LLMs) to perform deep research and generate comprehensive reports. Despite its progress, existing deep research frameworks primarily focus on generating text-only content, leaving the automated generation of interleaved texts and visualizations underexplored. This novel task poses key challenges in designing informative visualizations and effectively integrating them with text reports. To address these challenges, we propose Formal Description of Visualization (FDV), a structured textual representation of charts that enables LLMs to learn from and generate diverse, high-quality visualizations. Building on this representation, we introduce Multimodal DeepResearcher, an agentic framework that decomposes the task into four stages: (1) researching, (2) exemplar report textualization, (3) planning, and (4) multimodal report generation. For the evaluation of generated multimodal reports, we develop MultimodalReportBench, which contains 100 diverse topics served as inputs along with 5 dedicated metrics. Extensive experiments across models and evaluation methods demonstrate the effectiveness of Multimodal DeepResearcher. Notably, utilizing the same Claude 3.7 Sonnet model, Multimodal DeepResearcher achieves an 82\% overall win rate over the baseline method.
>
---
#### [replaced 002] PET: Preference Evolution Tracking with LLM-Generated Explainable Distribution
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2509.24189v2](https://arxiv.org/pdf/2509.24189v2)**

> **作者:** Luyang Zhang; Jialu Wang; Shichao Zhu; Siyuan Peng; Beibei Li; Zhongcun Wang; Guangmou Pan; Yan Li; Yang Song
>
> **摘要:** Understanding how user preference evolves over time is a fundamental challenge central to modern digital ecosystems, for which Large Language Models (LLMs) are an increasingly prominent and popular approach due to their ability to comprehend the rich semantic context within behavioral data. A common practice is to use LLMs to predict a user's next action by directly generating a ranked list of preferred items. Although effective for short-term prediction, the end-to-end generation paradigm inherently limits personalization. Its opaque decision-making process obscures holistic user profiling and exacerbates popularity bias. To address these limitations, we propose Preference Evolution Tracking (PET), a framework that reframes the task as inferring a dynamic probability distribution over a stable and interpretable lattice of preference clusters. By applying logit-probing and generative classification techniques, PET infers a user's preference as a probability distribution, enabling transparent preference learning. On public benchmarks (Yelp, MovieLens), PET improves ranking quality by up to 40% in NDCG over direct generation baselines. On a large-scale, real-world dataset from a short-video platform, it excels at ranking long-tail contents, significantly outperforming a SOTA production model by 7 times in the NDCG score. Ultimately, PET transforms the user profile model from direct preference list generation to a transparent distributional preference mapping, paving the way for more explainable, fair, and diverse personalization systems.
>
---
#### [replaced 003] BhashaKritika: Building Synthetic Pretraining Data at Scale for Indic Languages
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.10338v2](https://arxiv.org/pdf/2511.10338v2)**

> **作者:** Guduru Manoj; Neel Prabhanjan Rachamalla; Ashish Kulkarni; Gautam Rajeev; Jay Piplodiya; Arul Menezes; Shaharukh Khan; Souvik Rana; Manya Sah; Chandra Khatri; Shubham Agarwal
>
> **摘要:** In the context of pretraining of Large Language Models (LLMs), synthetic data has emerged as an alternative for generating high-quality pretraining data at scale. This is particularly beneficial in low-resource language settings where the benefits of recent LLMs have been unevenly distributed across languages. In this work, we present a systematic study on the generation and evaluation of synthetic multilingual pretraining data for Indic languages, where we construct a large-scale synthetic dataset BhashaKritika, comprising 540B tokens using 5 different techniques for 10 languages. We explore the impact of grounding generation in documents, personas, and topics. We analyze how language choice, both in the prompt instructions and document grounding, affects data quality, and we compare translations of English content with native generation in Indic languages. To support scalable and language-sensitive evaluation, we introduce a modular quality evaluation pipeline that integrates script and language detection, metadata consistency checks, n-gram repetition analysis, and perplexity-based filtering using KenLM models. Our framework enables robust quality control across diverse scripts and linguistic contexts. Empirical results through model runs reveal key trade-offs in generation strategies and highlight best practices for constructing effective multilingual corpora.
>
---
#### [replaced 004] Contextual Integrity in LLMs via Reasoning and Reinforcement Learning
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [https://arxiv.org/pdf/2506.04245v3](https://arxiv.org/pdf/2506.04245v3)**

> **作者:** Guangchen Lan; Huseyin A. Inan; Sahar Abdelnabi; Janardhan Kulkarni; Lukas Wutschitz; Reza Shokri; Christopher G. Brinton; Robert Sim
>
> **备注:** 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
>
> **摘要:** As the era of autonomous agents making decisions on behalf of users unfolds, ensuring contextual integrity (CI) -- what is the appropriate information to share while carrying out a certain task -- becomes a central question to the field. We posit that CI demands a form of reasoning where the agent needs to reason about the context in which it is operating. To test this, we first prompt LLMs to reason explicitly about CI when deciding what information to disclose. We then extend this approach by developing a reinforcement learning (RL) framework that further instills in models the reasoning necessary to achieve CI. Using a synthetic, automatically created, dataset of only $\sim700$ examples but with diverse contexts and information disclosure norms, we show that our method substantially reduces inappropriate information disclosure while maintaining task performance across multiple model sizes and families. Importantly, improvements transfer from this synthetic dataset to established CI benchmarks such as PrivacyLens that has human annotations and evaluates privacy leakage of AI assistants in actions and tool calls.
>
---
#### [replaced 005] TathyaNyaya and FactLegalLlama: Advancing Factual Judgment Prediction and Explanation in the Indian Legal Context
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **链接: [https://arxiv.org/pdf/2504.04737v3](https://arxiv.org/pdf/2504.04737v3)**

> **作者:** Shubham Kumar Nigam; Balaramamahanthi Deepak Patnaik; Shivam Mishra; Noel Shallum; Kripabandhu Ghosh; Arnab Bhattacharya
>
> **备注:** Paper accepted in the AACL-IJCNLP 2025 conference
>
> **摘要:** In the landscape of Fact-based Judgment Prediction and Explanation (FJPE), reliance on factual data is essential for developing robust and realistic AI-driven decision-making tools. This paper introduces TathyaNyaya, the largest annotated dataset for FJPE tailored to the Indian legal context, encompassing judgments from the Supreme Court of India and various High Courts. Derived from the Hindi terms "Tathya" (fact) and "Nyaya" (justice), the TathyaNyaya dataset is uniquely designed to focus on factual statements rather than complete legal texts, reflecting real-world judicial processes where factual data drives outcomes. Complementing this dataset, we present FactLegalLlama, an instruction-tuned variant of the LLaMa-3-8B Large Language Model (LLM), optimized for generating high-quality explanations in FJPE tasks. Finetuned on the factual data in TathyaNyaya, FactLegalLlama integrates predictive accuracy with coherent, contextually relevant explanations, addressing the critical need for transparency and interpretability in AI-assisted legal systems. Our methodology combines transformers for binary judgment prediction with FactLegalLlama for explanation generation, creating a robust framework for advancing FJPE in the Indian legal domain. TathyaNyaya not only surpasses existing datasets in scale and diversity but also establishes a benchmark for building explainable AI systems in legal analysis. The findings underscore the importance of factual precision and domain-specific tuning in enhancing predictive performance and interpretability, positioning TathyaNyaya and FactLegalLlama as foundational resources for AI-assisted legal decision-making.
>
---
#### [replaced 006] PIP: Perturbation-based Iterative Pruning for Large Language Models
- **分类: cs.LG; cs.CL**

- **链接: [https://arxiv.org/pdf/2501.15278v3](https://arxiv.org/pdf/2501.15278v3)**

> **作者:** Yi Cao; Wei-Jie Xu; Yucheng Shen; Weijie Shi; Chi-Min Chan; Jianfeng Qu; Jiajie Xu
>
> **备注:** EMNLP 2025 Findings, 17 pages, 5 figures, 15 tables
>
> **摘要:** The rapid increase in the parameter counts of Large Language Models (LLMs), which often reach into the billions or even trillions, presents significant challenges for their practical deployment, particularly in resource-constrained environments. To address this issue, we propose PIP (Perturbation-based Iterative Pruning), a novel double-view structured pruning method to optimize LLMs, which combines information from two different views: the unperturbed view and the perturbed view. With the calculation of gradient differences, PIP iteratively prunes those that struggle to distinguish between these two views. Our experiments show that PIP reduces the parameter count by approximately 20% while retaining over 85% of the original model's accuracy across varied benchmarks. In some cases, the performance of the pruned model is within 5% of the unpruned version, demonstrating PIP's ability to preserve key aspects of model effectiveness. Moreover, PIP consistently outperforms existing state-of-the-art (SOTA) structured pruning methods, establishing it as a leading technique for optimizing LLMs in constrained environments.
>
---
#### [replaced 007] SelecTKD: Selective Token-Weighted Knowledge Distillation for LLMs
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.24021v2](https://arxiv.org/pdf/2510.24021v2)**

> **作者:** Haiduo Huang; Jiangcheng Song; Yadong Zhang; Pengju Ren
>
> **摘要:** Knowledge distillation (KD) is a standard route to compress Large Language Models (LLMs) into compact students, yet most pipelines uniformly apply token-wise loss regardless of teacher confidence. This indiscriminate supervision amplifies noisy, high-entropy signals and is especially harmful under large teacher-student capacity gaps. We introduce SelecTKD, a plug-and-play Selective Token-Weighted distillation framework that shifts the focus from "how to measure divergence" to "where to apply learning". At each step, the student proposes tokens that are verified by the teacher through a robust propose-and-verify procedure with two variants: greedy Top-k and non-greedy Spec-k. Accepted tokens receive full loss, while rejected tokens are masked or down-weighted. This objective-agnostic design works with on- and off-policy data, induces an implicit curriculum quantified by Token Acceptance Rate (TAR), and stabilizes optimization. Across instruction following, mathematical reasoning, code generation, and a VLM setting, SelecTKD consistently improves strong baselines and achieves state-of-the-art results for small models without architectural changes or extra reference models.
>
---
#### [replaced 008] Fact2Fiction: Targeted Poisoning Attack to Agentic Fact-checking System
- **分类: cs.CR; cs.CL**

- **链接: [https://arxiv.org/pdf/2508.06059v2](https://arxiv.org/pdf/2508.06059v2)**

> **作者:** Haorui He; Yupeng Li; Bin Benjamin Zhu; Dacheng Wen; Reynold Cheng; Francis C. M. Lau
>
> **备注:** Accepted by AAAI 2026 (Oral). Code available at: https://trustworthycomp.github.io/Fact2Fiction/
>
> **摘要:** State-of-the-art (SOTA) fact-checking systems combat misinformation by employing autonomous LLM-based agents to decompose complex claims into smaller sub-claims, verify each sub-claim individually, and aggregate the partial results to produce verdicts with justifications (explanations for the verdicts). The security of these systems is crucial, as compromised fact-checkers can amplify misinformation, but remains largely underexplored. To bridge this gap, this work introduces a novel threat model against such fact-checking systems and presents \textsc{Fact2Fiction}, the first poisoning attack framework targeting SOTA agentic fact-checking systems. Fact2Fiction employs LLMs to mimic the decomposition strategy and exploit system-generated justifications to craft tailored malicious evidences that compromise sub-claim verification. Extensive experiments demonstrate that Fact2Fiction achieves 8.9\%--21.2\% higher attack success rates than SOTA attacks across various poisoning budgets and exposes security weaknesses in existing fact-checking systems, highlighting the need for defensive countermeasures.
>
---
#### [replaced 009] A Survey on Unlearning in Large Language Models
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2510.25117v2](https://arxiv.org/pdf/2510.25117v2)**

> **作者:** Ruichen Qiu; Jiajun Tan; Jiayue Pu; Honglin Wang; Xiao-Shan Gao; Fei Sun
>
> **摘要:** Large Language Models (LLMs) demonstrate remarkable capabilities, but their training on massive corpora poses significant risks from memorized sensitive information. To mitigate these issues and align with legal standards, unlearning has emerged as a critical technique to selectively erase specific knowledge from LLMs without compromising their overall performance. This survey provides a systematic review of over 180 papers on LLM unlearning published since 2021. First, it introduces a novel taxonomy that categorizes unlearning methods based on the phase in the LLM pipeline of the intervention. This framework further distinguishes between parameter modification and parameter selection strategies, thus enabling deeper insights and more informed comparative analysis. Second, it offers a multidimensional analysis of evaluation paradigms. For datasets, we compare 18 existing benchmarks from the perspectives of task format, content, and experimental paradigms to offer actionable guidance. For metrics, we move beyond mere enumeration by dividing knowledge memorization metrics into 10 categories to analyze their advantages and applicability, while also reviewing metrics for model utility, robustness, and efficiency. By discussing current challenges and future directions, this survey aims to advance the field of LLM unlearning and the development of secure AI systems.
>
---
#### [replaced 010] PAN: A World Model for General, Interactable, and Long-Horizon World Simulation
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.09057v3](https://arxiv.org/pdf/2511.09057v3)**

> **作者:** PAN Team; Jiannan Xiang; Yi Gu; Zihan Liu; Zeyu Feng; Qiyue Gao; Yiyan Hu; Benhao Huang; Guangyi Liu; Yichi Yang; Kun Zhou; Davit Abrahamyan; Arif Ahmad; Ganesh Bannur; Junrong Chen; Kimi Chen; Mingkai Deng; Ruobing Han; Xinqi Huang; Haoqiang Kang; Zheqi Liu; Enze Ma; Hector Ren; Yashowardhan Shinde; Rohan Shingre; Ramsundar Tanikella; Kaiming Tao; Dequan Yang; Xinle Yu; Cong Zeng; Binglin Zhou; Zhengzhong Liu; Zhiting Hu; Eric P. Xing
>
> **摘要:** A world model enables an intelligent agent to imagine, predict, and reason about how the world evolves in response to its actions, and accordingly to plan and strategize. While recent video generation models produce realistic visual sequences, they typically operate in the prompt-to-full-video manner without causal control, interactivity, or long-horizon consistency required for purposeful reasoning. Existing world modeling efforts, on the other hand, often focus on restricted domains (e.g., physical, game, or 3D-scene dynamics) with limited depth and controllability, and struggle to generalize across diverse environments and interaction formats. In this work, we introduce PAN, a general, interactable, and long-horizon world model that predicts future world states through high-quality video simulation conditioned on history and natural language actions. PAN employs the Generative Latent Prediction (GLP) architecture that combines an autoregressive latent dynamics backbone based on a large language model (LLM), which grounds simulation in extensive text-based knowledge and enables conditioning on language-specified actions, with a video diffusion decoder that reconstructs perceptually detailed and temporally coherent visual observations, to achieve a unification between latent space reasoning (imagination) and realizable world dynamics (reality). Trained on large-scale video-action pairs spanning diverse domains, PAN supports open-domain, action-conditioned simulation with coherent, long-term dynamics. Extensive experiments show that PAN achieves strong performance in action-conditioned world simulation, long-horizon forecasting, and simulative reasoning compared to other video generators and world models, taking a step towards general world models that enable predictive simulation of future world states for reasoning and acting.
>
---
#### [replaced 011] Aligning Machiavellian Agents: Behavior Steering via Test-Time Policy Shaping
- **分类: cs.AI; cs.CL**

- **链接: [https://arxiv.org/pdf/2511.11551v2](https://arxiv.org/pdf/2511.11551v2)**

> **作者:** Dena Mujtaba; Brian Hu; Anthony Hoogs; Arslan Basharat
>
> **备注:** Accepted to AAAI 2026 AI Alignment Track
>
> **摘要:** The deployment of decision-making AI agents presents a critical challenge in maintaining alignment with human values or guidelines while operating in complex, dynamic environments. Agents trained solely to achieve their objectives may adopt harmful behavior, exposing a key trade-off between maximizing the reward function and maintaining alignment. For pre-trained agents, ensuring alignment is particularly challenging, as retraining can be a costly and slow process. This is further complicated by the diverse and potentially conflicting attributes representing the ethical values for alignment. To address these challenges, we propose a test-time alignment technique based on model-guided policy shaping. Our method allows precise control over individual behavioral attributes, generalizes across diverse reinforcement learning (RL) environments, and facilitates a principled trade-off between ethical alignment and reward maximization without requiring agent retraining. We evaluate our approach using the MACHIAVELLI benchmark, which comprises 134 text-based game environments and thousands of annotated scenarios involving ethical decisions. The RL agents are first trained to maximize the reward in their respective games. At test time, we apply policy shaping via scenario-action attribute classifiers to ensure decision alignment with ethical attributes. We compare our approach against prior training-time methods and general-purpose agents, as well as study several types of ethical violations and power-seeking behavior. Our results demonstrate that test-time policy shaping provides an effective and scalable solution for mitigating unethical behavior across diverse environments and alignment attributes.
>
---
#### [replaced 012] When Facts Change: Probing LLMs on Evolving Knowledge with evolveQA
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.19172v2](https://arxiv.org/pdf/2510.19172v2)**

> **作者:** Nishanth Sridhar Nakshatri; Shamik Roy; Manoj Ghuhan Arivazhagan; Hanhan Zhou; Vinayshekhar Bannihatti Kumar; Rashmi Gangadharaiah
>
> **备注:** Under submission
>
> **摘要:** LLMs often fail to handle temporal knowledge conflicts--contradictions arising when facts evolve over time within their training data. Existing studies evaluate this phenomenon through benchmarks built on structured knowledge bases like Wikidata, but they focus on widely-covered, easily-memorized popular entities and lack the dynamic structure needed to fairly evaluate LLMs with different knowledge cut-off dates. We introduce evolveQA, a benchmark specifically designed to evaluate LLMs on temporally evolving knowledge, constructed from 3 real-world, time-stamped corpora: AWS updates, Azure changes, and WHO disease outbreak reports. Our framework identifies naturally occurring knowledge evolution and generates questions with gold answers tailored to different LLM knowledge cut-off dates. Through extensive evaluation of 12 open and closed-source LLMs across 3 knowledge probing formats, we demonstrate significant performance drops of up to 31% on evolveQA compared to static knowledge questions.
>
---
#### [replaced 013] Is Our Chatbot Telling Lies? Assessing Correctness of an LLM-based Dutch Support Chatbot
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2411.00034v2](https://arxiv.org/pdf/2411.00034v2)**

> **作者:** Herman Lassche; Michiel Overeem; Ayushi Rastogi
>
> **备注:** 10 pages + 2 pages references, 4 figures
>
> **摘要:** Companies support their customers using live chats and chatbots to gain their loyalty. AFAS is a Dutch company aiming to leverage the opportunity large language models (LLMs) offer to answer customer queries with minimal to no input from its customer support team. Adding to its complexity, it is unclear what makes a response correct, and that too in Dutch. Further, with minimal data available for training, the challenge is to identify whether an answer generated by a large language model is correct and do it on the fly. This study is the first to define the correctness of a response based on how the support team at AFAS makes decisions. It leverages literature on natural language generation and automated answer grading systems to automate the decision-making of the customer support team. We investigated questions requiring a binary response (e.g., Would it be possible to adjust tax rates manually?) or instructions (e.g., How would I adjust tax rate manually?) to test how close our automated approach reaches support rating. Our approach can identify wrong messages in 55\% of the cases. This work demonstrates the potential for automatically assessing when our chatbot may provide incorrect or misleading answers. Specifically, we contribute (1) a definition and metrics for assessing correctness, and (2) suggestions to improve correctness with respect to regional language and question type.
>
---
#### [replaced 014] DiagnoLLM: A Hybrid Bayesian Neural Language Framework for Interpretable Disease Diagnosis
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.05810v2](https://arxiv.org/pdf/2511.05810v2)**

> **作者:** Bowen Xu; Xinyue Zeng; Jiazhen Hu; Tuo Wang; Adithya Kulkarni
>
> **摘要:** Building trustworthy clinical AI systems requires not only accurate predictions but also transparent, biologically grounded explanations. We present \texttt{DiagnoLLM}, a hybrid framework that integrates Bayesian deconvolution, eQTL-guided deep learning, and LLM-based narrative generation for interpretable disease diagnosis. DiagnoLLM begins with GP-unmix, a Gaussian Process-based hierarchical model that infers cell-type-specific gene expression profiles from bulk and single-cell RNA-seq data while modeling biological uncertainty. These features, combined with regulatory priors from eQTL analysis, power a neural classifier that achieves high predictive performance in Alzheimer's Disease (AD) detection (88.0\% accuracy). To support human understanding and trust, we introduce an LLM-based reasoning module that translates model outputs into audience-specific diagnostic reports, grounded in clinical features, attribution signals, and domain knowledge. Human evaluations confirm that these reports are accurate, actionable, and appropriately tailored for both physicians and patients. Our findings show that LLMs, when deployed as post-hoc reasoners rather than end-to-end predictors, can serve as effective communicators within hybrid diagnostic pipelines.
>
---
#### [replaced 015] Mitigating Overthinking in Large Reasoning Models via Manifold Steering
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [https://arxiv.org/pdf/2505.22411v2](https://arxiv.org/pdf/2505.22411v2)**

> **作者:** Yao Huang; Huanran Chen; Shouwei Ruan; Yichi Zhang; Xingxing Wei; Yinpeng Dong
>
> **备注:** 19 pages, 7 figures
>
> **摘要:** Recent advances in Large Reasoning Models (LRMs) have demonstrated remarkable capabilities in solving complex tasks such as mathematics and coding. However, these models frequently exhibit a phenomenon known as overthinking during inference, characterized by excessive validation loops and redundant deliberation, leading to substantial computational overheads. In this paper, we aim to mitigate overthinking by investigating the underlying mechanisms from the perspective of mechanistic interpretability. We first showcase that the tendency of overthinking can be effectively captured by a single direction in the model's activation space and the issue can be eased by intervening the activations along this direction. However, this efficacy soon reaches a plateau and even deteriorates as the intervention strength increases. We therefore systematically explore the activation space and find that the overthinking phenomenon is actually tied to a low-dimensional manifold, which indicates that the limited effect stems from the noises introduced by the high-dimensional steering direction. Based on this insight, we propose Manifold Steering, a novel approach that elegantly projects the steering direction onto the low-dimensional activation manifold given the theoretical approximation of the interference noise. Extensive experiments on DeepSeek-R1 distilled models validate that our method reduces output tokens by up to 71% while maintaining and even improving the accuracy on several mathematical benchmarks. Our method also exhibits robust cross-domain transferability, delivering consistent token reduction performance in code generation and knowledge-based QA tasks. Code is available at: https://github.com/Aries-iai/Manifold_Steering.
>
---
#### [replaced 016] MedFact: Benchmarking the Fact-Checking Capabilities of Large Language Models on Chinese Medical Texts
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2509.12440v2](https://arxiv.org/pdf/2509.12440v2)**

> **作者:** Jiayi He; Yangmin Huang; Qianyun Du; Xiangying Zhou; Zhiyang He; Jiaxue Hu; Xiaodong Tao; Lixian Lai
>
> **摘要:** Deploying Large Language Models (LLMs) in medical applications requires fact-checking capabilities to ensure patient safety and regulatory compliance. We introduce MedFact, a challenging Chinese medical fact-checking benchmark with 2,116 expert-annotated instances from diverse real-world texts, spanning 13 specialties, 8 error types, 4 writing styles, and 5 difficulty levels. Construction uses a hybrid AI-human framework where iterative expert feedback refines AI-driven, multi-criteria filtering to ensure high quality and difficulty. We evaluate 20 leading LLMs on veracity classification and error localization, and results show models often determine if text contains errors but struggle to localize them precisely, with top performers falling short of human performance. Our analysis reveals the "over-criticism" phenomenon, a tendency for models to misidentify correct information as erroneous, which can be exacerbated by advanced reasoning techniques such as multi-agent collaboration and inference-time scaling. MedFact highlights the challenges of deploying medical LLMs and provides resources to develop factually reliable medical AI systems.
>
---
#### [replaced 017] PurpCode: Reasoning for Safer Code Generation
- **分类: cs.CR; cs.CL; cs.LG; cs.SE**

- **链接: [https://arxiv.org/pdf/2507.19060v4](https://arxiv.org/pdf/2507.19060v4)**

> **作者:** Jiawei Liu; Nirav Diwan; Zhe Wang; Haoyu Zhai; Xiaona Zhou; Kiet A. Nguyen; Tianjiao Yu; Muntasir Wahed; Yinlin Deng; Hadjer Benkraouda; Yuxiang Wei; Lingming Zhang; Ismini Lourentzou; Gang Wang
>
> **摘要:** We introduce PurpCode, the first post-training recipe for training safe code reasoning models towards generating secure code and defending against malicious cyberactivities. PurpCode trains a reasoning model in two stages: (i) Rule Learning, which explicitly teaches the model to reference cybersafety rules to generate vulnerability-free code and to avoid facilitating malicious cyberactivities; and (ii) Reinforcement Learning, which optimizes model safety and preserves model utility through diverse, multi-objective reward mechanisms. To empower the training pipelines with comprehensive cybersafety data, we conduct internal red-teaming to synthesize comprehensive and high-coverage prompts based on real-world tasks for inducing unsafe cyberactivities in the model. Based on PurpCode, we develop a reasoning-based coding model, namely PurpCode-32B, which demonstrates state-of-the-art cybersafety, outperforming various frontier models. Meanwhile, our alignment method decreases the model overrefusal rates in both general and cybersafety-specific scenarios, while preserving model utility in both code generation and common security knowledge.
>
---
#### [replaced 018] Novelty and Impact of Economics Papers
- **分类: econ.GN; cs.CE; cs.CL; cs.DL**

- **链接: [https://arxiv.org/pdf/2511.01211v3](https://arxiv.org/pdf/2511.01211v3)**

> **作者:** Chaofeng Wu
>
> **摘要:** We propose a framework that recasts scientific novelty not as a single attribute of a paper, but as a reflection of its position within the evolving intellectual landscape. We decompose this position into two orthogonal dimensions: \textit{spatial novelty}, which measures a paper's intellectual distinctiveness from its neighbors, and \textit{temporal novelty}, which captures its engagement with a dynamic research frontier. To operationalize these concepts, we leverage Large Language Models to develop semantic isolation metrics that quantify a paper's location relative to the full-text literature. Applying this framework to a large corpus of economics articles, we uncover a fundamental trade-off: these two dimensions predict systematically different outcomes. Temporal novelty primarily predicts citation counts, whereas spatial novelty predicts disruptive impact. This distinction allows us to construct a typology of semantic neighborhoods, identifying four archetypes associated with distinct and predictable impact profiles. Our findings demonstrate that novelty can be understood as a multidimensional construct whose different forms, reflecting a paper's strategic location, have measurable and fundamentally distinct consequences for scientific progress.
>
---
#### [replaced 019] Compress, Gather, and Recompute: REFORMing Long-Context Processing in Transformers
- **分类: cs.CL; cs.LG**

- **链接: [https://arxiv.org/pdf/2506.01215v2](https://arxiv.org/pdf/2506.01215v2)**

> **作者:** Woomin Song; Sai Muralidhar Jayanthi; Srikanth Ronanki; Kanthashree Mysore Sathyendra; Jinwoo Shin; Aram Galstyan; Shubham Katiyar; Sravan Babu Bodapati
>
> **备注:** NeurIPS 2025
>
> **摘要:** As large language models increasingly gain popularity in real-world applications, processing extremely long contexts, often exceeding the model's pre-trained context limits, has emerged as a critical challenge. While existing approaches to efficient long-context processing show promise, recurrent compression-based methods struggle with information preservation, whereas random access approaches require substantial memory resources. We introduce REFORM, a novel inference framework that efficiently handles long contexts through a two-phase approach. First, it incrementally processes input chunks while maintaining a compressed KV cache, constructs cross-layer context embeddings, and utilizes early exit strategy for improved efficiency. Second, it identifies and gathers essential tokens via similarity matching and selectively recomputes the KV cache. Compared to baselines, REFORM achieves over 52% and 34% performance gains on RULER and BABILong respectively at 1M context length. It also outperforms baselines on Infinite-Bench, RepoEval, and MM-NIAH, demonstrating flexibility across diverse tasks and domains. Additionally, REFORM reduces inference time by 30% and peak memory usage by 5%, achieving both efficiency and superior performance.
>
---
#### [replaced 020] DeceptionBench: A Comprehensive Benchmark for AI Deception Behaviors in Real-world Scenarios
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.15501v2](https://arxiv.org/pdf/2510.15501v2)**

> **作者:** Yao Huang; Yitong Sun; Yichi Zhang; Ruochen Zhang; Yinpeng Dong; Xingxing Wei
>
> **备注:** 28 pages, 17 figures, accepted by NeruIPS 2025
>
> **摘要:** Despite the remarkable advances of Large Language Models (LLMs) across diverse cognitive tasks, the rapid enhancement of these capabilities also introduces emergent deceptive behaviors that may induce severe risks in high-stakes deployments. More critically, the characterization of deception across realistic real-world scenarios remains underexplored. To bridge this gap, we establish DeceptionBench, the first benchmark that systematically evaluates how deceptive tendencies manifest across different societal domains, what their intrinsic behavioral patterns are, and how extrinsic factors affect them. Specifically, on the static count, the benchmark encompasses 150 meticulously designed scenarios in five domains, i.e., Economy, Healthcare, Education, Social Interaction, and Entertainment, with over 1,000 samples, providing sufficient empirical foundations for deception analysis. On the intrinsic dimension, we explore whether models exhibit self-interested egoistic tendencies or sycophantic behaviors that prioritize user appeasement. On the extrinsic dimension, we investigate how contextual factors modulate deceptive outputs under neutral conditions, reward-based incentivization, and coercive pressures. Moreover, we incorporate sustained multi-turn interaction loops to construct a more realistic simulation of real-world feedback dynamics. Extensive experiments across LLMs and Large Reasoning Models (LRMs) reveal critical vulnerabilities, particularly amplified deception under reinforcement dynamics, demonstrating that current models lack robust resistance to manipulative contextual cues and the urgent need for advanced safeguards against various deception behaviors. Code and resources are publicly available at https://github.com/Aries-iai/DeceptionBench.
>
---
#### [replaced 021] FinGPT: Open-Source Financial Large Language Models
- **分类: q-fin.ST; cs.CL; cs.LG; q-fin.TR**

- **链接: [https://arxiv.org/pdf/2306.06031v2](https://arxiv.org/pdf/2306.06031v2)**

> **作者:** Hongyang Yang; Xiao-Yang Liu; Christina Dan Wang
>
> **备注:** Accepted by the FinLLM Symposium at IJCAI 2023. Recipient of the Best Presentation Award (Hongyang Yang). Workshop link: https://finllm.github.io/workshop. This is the first official FinGPT paper; please cite this work when referencing FinGPT
>
> **摘要:** Large language models (LLMs) have shown the potential of revolutionizing natural language processing tasks in diverse domains, sparking great interest in finance. Accessing high-quality financial data is the first challenge for financial LLMs (FinLLMs). While proprietary models like BloombergGPT have taken advantage of their unique data accumulation, such privileged access calls for an open-source alternative to democratize Internet-scale financial data. In this paper, we present an open-source large language model, FinGPT, for the finance sector. Unlike proprietary models, FinGPT takes a data-centric approach, providing researchers and practitioners with accessible and transparent resources to develop their FinLLMs. We highlight the importance of an automatic data curation pipeline and the lightweight low-rank adaptation technique in building FinGPT. Furthermore, we showcase several potential applications as stepping stones for users, such as robo-advising, algorithmic trading, and low-code development. Through collaborative efforts within the open-source AI4Finance community, FinGPT aims to stimulate innovation, democratize FinLLMs, and unlock new opportunities in open finance. Two associated code repos are https://github.com/AI4Finance-Foundation/FinGPT and https://github.com/AI4Finance-Foundation/FinNLP
>
---
#### [replaced 022] Learning How to Use Tools, Not Just When: Pattern-Aware Tool-Integrated Reasoning
- **分类: cs.AI; cs.CL**

- **链接: [https://arxiv.org/pdf/2509.23292v2](https://arxiv.org/pdf/2509.23292v2)**

> **作者:** Ningning Xu; Yuxuan Jiang; Shubhashis Roy Dipta
>
> **摘要:** Tool-integrated reasoning (TIR) has become a key approach for improving large reasoning models (LRMs) on complex problems. Prior work has mainly studied when to invoke tools, while overlooking how tools are applied. We identify two common patterns: a calculator pattern that uses code for direct computation, and an algorithmic pattern that encodes problems as programs. Misaligned choices often cause failures even when reasoning is sound. We propose a two-stage framework that first builds code competence from both patterns and then aligns pattern selection with teacher preferences. Across challenging math datasets, our pattern-aware method substantially improves both code usage and accuracy, for instance raising Code@1 on MATH500 from 64.0% to 70.5% and on AIME24 from 26.7% to 50.0%. These gains highlight the effectiveness of a pattern-aware approach for tool-integrated reasoning.
>
---
#### [replaced 023] Contextual Breach: Assessing the Robustness of Transformer-based QA Models
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2409.10997v4](https://arxiv.org/pdf/2409.10997v4)**

> **作者:** Asir Saadat; Nahian Ibn Asad
>
> **摘要:** Contextual question-answering models are susceptible to adversarial perturbations to input context, commonly observed in real-world scenarios. These adversarial noises are designed to degrade the performance of the model by distorting the textual input. We introduce a unique dataset that incorporates seven distinct types of adversarial noise into the context, each applied at five different intensity levels on the SQuAD dataset. To quantify the robustness, we utilize robustness metrics providing a standardized measure for assessing model performance across varying noise types and levels. Experiments on transformer-based question-answering models reveal robustness vulnerabilities and important insights into the model's performance in realistic textual input.
>
---
#### [replaced 024] Fair In-Context Learning via Latent Concept Variables
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [https://arxiv.org/pdf/2411.02671v2](https://arxiv.org/pdf/2411.02671v2)**

> **作者:** Karuna Bhaila; Minh-Hao Van; Kennedy Edemacu; Chen Zhao; Feng Chen; Xintao Wu
>
> **备注:** IEEE BigData 2025
>
> **摘要:** The emerging in-context learning (ICL) ability of large language models (LLMs) has prompted their use for predictive tasks in various domains with different data types, including tabular data, facilitated by serialization methods. However, with increasing applications in high-stakes domains, it has been shown that LLMs can inherit social bias and discrimination from their pre-training data. In this work, we investigate inherent bias in LLMs during in-context learning with tabular data. We focus on an optimal demonstration selection approach that utilizes latent concept variables for resource-efficient task adaptation. We design data augmentation strategies that reduce the correlation between predictive outcomes and sensitive variables, helping promote fairness during latent concept learning. We utilize the learned concept to select demonstrations and obtain fair predictions. The latent concept variables are learned using a smaller internal LLM and generalized to larger external LLMs. We empirically verify that the fair latent variable approach improves fairness results on tabular datasets compared to multiple heuristic demonstration selection methods.
>
---
#### [replaced 025] KTAE: A Model-Free Algorithm to Key-Tokens Advantage Estimation in Mathematical Reasoning
- **分类: cs.AI; cs.CL**

- **链接: [https://arxiv.org/pdf/2505.16826v2](https://arxiv.org/pdf/2505.16826v2)**

> **作者:** Wei Sun; Wen Yang; Pu Jian; Qianlong Du; Fuwei Cui; Shuo Ren; Jiajun Zhang
>
> **备注:** NeurIPS 2025 Poster
>
> **摘要:** Recent advances have demonstrated that integrating reinforcement learning with rule-based rewards can significantly enhance the reasoning capabilities of large language models, even without supervised fine-tuning. However, prevalent reinforcement learning algorithms such as GRPO and its variants like DAPO, suffer from a coarse granularity issue when computing the advantage. Specifically, they compute rollout-level advantages that assign identical values to every token within a sequence, failing to capture token-specific contributions and hindering effective learning. To address this limitation, we propose Key-token Advantage Estimation (KTAE) - a novel algorithm that estimates fine-grained, token-level advantages without introducing additional models. KTAE leverages the correctness of sampled rollouts and applies statistical analysis to quantify the importance of individual tokens within a sequence to the final outcome. This quantified token-level importance is then combined with the rollout-level advantage to obtain a more fine-grained token-level advantage estimation. Empirical results show that models trained with GRPO+KTAE and DAPO+KTAE outperform baseline methods across five mathematical reasoning benchmarks. Notably, they achieve higher accuracy with shorter responses and even surpass R1-Distill-Qwen-1.5B using the same base model.
>
---
#### [replaced 026] Efficient Post-Training Refinement of Latent Reasoning in Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2506.08552v2](https://arxiv.org/pdf/2506.08552v2)**

> **作者:** Xinyuan Wang; Dongjie Wang; Wangyang Ying; Haoyue Bai; Nanxu Gong; Sixun Dong; Kunpeng Liu; Yanjie Fu
>
> **摘要:** Reasoning is a key component of language understanding in Large Language Models. While Chain-of-Thought prompting enhances performance via explicit intermediate steps, it suffers from sufficient token overhead and a fixed reasoning trajectory, preventing step-wise refinement. Recent advances in latent reasoning address these limitations by refining internal reasoning processes directly in the model's latent space, without producing explicit outputs. However, a key challenge remains: how to effectively update reasoning embeddings during post-training to guide the model toward more accurate solutions. To overcome this challenge, we propose a lightweight post-training framework that refines latent reasoning trajectories using two novel strategies: 1) Contrastive reasoning feedback, which compares reasoning embeddings against strong and weak baselines to infer effective update directions via embedding enhancement; 2) Residual embedding refinement, which stabilizes updates by progressively integrating current and historical gradients, enabling fast yet controlled convergence. Extensive experiments and case studies are conducted on five reasoning benchmarks to demonstrate the effectiveness of the proposed framework. Notably, a 5\% accuracy gain on MathQA without additional training.
>
---
#### [replaced 027] SoK: Large Language Model Copyright Auditing via Fingerprinting
- **分类: cs.CR; cs.AI; cs.CL**

- **链接: [https://arxiv.org/pdf/2508.19843v3](https://arxiv.org/pdf/2508.19843v3)**

> **作者:** Shuo Shao; Yiming Li; Yu He; Hongwei Yao; Wenyuan Yang; Dacheng Tao; Zhan Qin
>
> **摘要:** The broad capabilities and substantial resources required to train Large Language Models (LLMs) make them valuable intellectual property, yet they remain vulnerable to copyright infringement, such as unauthorized use and model theft. LLM fingerprinting, a non-intrusive technique that compares the distinctive features (i.e., fingerprint) of LLMs to identify whether an LLM is derived from another, offers a promising solution to copyright auditing. However, its reliability remains uncertain due to the prevalence of diverse model modifications and the lack of standardized evaluation. In this SoK, we present the first comprehensive study of the emerging LLM fingerprinting. We introduce a unified framework and taxonomy that structures the field: white-box methods are classified based on their feature source as static, forward-pass, or backward-pass fingerprinting, while black-box methods are distinguished by their query strategy as either untargeted or targeted. Furthermore, we propose LeaFBench, the first systematic benchmark for evaluating LLM fingerprinting under realistic deployment scenarios. Built upon 7 mainstream foundation models and comprising 149 distinct model instances, LeaFBench integrates 13 representative post-development techniques, spanning both parameter-altering methods (e.g., fine-tuning, quantization) and parameter-independent techniques (e.g., system prompts, RAG). Extensive experiments on LeaFBench reveal the strengths and weaknesses of existing methods, thereby outlining future research directions and critical open problems in this emerging field. The code is available at https://github.com/shaoshuo-ss/LeaFBench.
>
---
#### [replaced 028] NLP Methods May Actually Be Better Than Professors at Estimating Question Difficulty
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2508.03294v2](https://arxiv.org/pdf/2508.03294v2)**

> **作者:** Leonidas Zotos; Ivo Pascal de Jong; Matias Valdenegro-Toro; Andreea Ioana Sburlea; Malvina Nissim; Hedderik van Rijn
>
> **备注:** 10 pages, 2 figures, presented at ECAI 2025 at the 2nd International Workshop on AI in Society, Education and Educational Research (AISEER)
>
> **摘要:** Estimating the difficulty of exam questions is essential for developing good exams, but professors are not always good at this task. We compare various Large Language Model-based methods with three professors in their ability to estimate what percentage of students will give correct answers on True/False exam questions in the areas of Neural Networks and Machine Learning. Our results show that the professors have limited ability to distinguish between easy and difficult questions and that they are outperformed by directly asking Gemini 2.5 to solve this task. Yet, we obtained even better results using uncertainties of the LLMs solving the questions in a supervised learning setting, using only 42 training samples. We conclude that supervised learning using LLM uncertainty can help professors better estimate the difficulty of exam questions, improving the quality of assessment.
>
---
#### [replaced 029] Ensemble Debates with Local Large Language Models for AI Alignment
- **分类: cs.AI; cs.CL**

- **链接: [https://arxiv.org/pdf/2509.00091v2](https://arxiv.org/pdf/2509.00091v2)**

> **作者:** Ephraiem Sarabamoun
>
> **备注:** The manuscript is being withdrawn to incorporate additional revisions and improvements
>
> **摘要:** As large language models (LLMs) take on greater roles in high-stakes decisions, alignment with human values is essential. Reliance on proprietary APIs limits reproducibility and broad participation. We study whether local open-source ensemble debates can improve alignmentoriented reasoning. Across 150 debates spanning 15 scenarios and five ensemble configurations, ensembles outperform single-model baselines on a 7-point rubric (overall: 3.48 vs. 3.13), with the largest gains in reasoning depth (+19.4%) and argument quality (+34.1%). Improvements are strongest for truthfulness (+1.25 points) and human enhancement (+0.80). We provide code, prompts, and a debate data set, providing an accessible and reproducible foundation for ensemble-based alignment evaluation.
>
---
#### [replaced 030] RPRO: Ranked Preference Reinforcement Optimization for Enhancing Medical QA and Diagnostic Reasoning
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2509.00974v3](https://arxiv.org/pdf/2509.00974v3)**

> **作者:** Chia-Hsuan Hsu; Jun-En Ding; Hsin-Ling Hsu; Chih-Ho Hsu; Li-Hung Yao; Chun-Chieh Liao; Feng Liu; Fang-Ming Hung
>
> **摘要:** Medical question answering requires advanced reasoning that integrates domain knowledge with logical inference. However, existing large language models (LLMs) often generate reasoning chains that lack factual accuracy and clinical reliability. We propose Ranked Preference Reinforcement Optimization (RPRO), a novel framework that combines reinforcement learning with preference-driven reasoning refinement to enhance clinical chain-of-thought (CoT) performance. RPRO distinguishes itself from prior approaches by employing task-adaptive reasoning templates and a probabilistic evaluation mechanism that aligns model outputs with established clinical workflows, while automatically identifying and correcting low-quality reasoning chains. Unlike traditional pairwise preference methods, RPRO introduces a groupwise ranking optimization based on the Bradley--Terry model and incorporates KL-divergence regularization for stable training. Experiments on PubMedQA, MedQA-USMLE, and a real-world clinical dataset from Far Eastern Memorial Hospital (FEMH) demonstrate consistent improvements over strong baselines. Remarkably, our 2B-parameter model outperforms much larger 7B--20B models, including medical-specialized variants. These findings demonstrate that combining preference optimization with quality-driven refinement provides a scalable and clinically grounded approach to building more reliable medical LLMs.
>
---
#### [replaced 031] Efficient Reasoning for Large Reasoning Language Models via Certainty-Guided Reflection Suppression
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2508.05337v2](https://arxiv.org/pdf/2508.05337v2)**

> **作者:** Jiameng Huang; Baijiong Lin; Guhao Feng; Jierun Chen; Di He; Lu Hou
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Recent Large Reasoning Language Models (LRLMs) employ long chain-of-thought reasoning with complex reflection behaviors, typically signaled by specific trigger words (e.g., "Wait" and "Alternatively") to enhance performance. However, these reflection behaviors can lead to the overthinking problem where the generation of redundant reasoning steps that unnecessarily increase token usage, raise inference costs, and reduce practical utility. In this paper, we propose Certainty-Guided Reflection Suppression (CGRS), a novel method that mitigates overthinking in LRLMs while maintaining reasoning accuracy. CGRS operates by dynamically suppressing the model's generation of reflection triggers when it exhibits high confidence in its current response, thereby preventing redundant reflection cycles without compromising output quality. Our approach is model-agnostic, requires no retraining or architectural modifications, and can be integrated seamlessly with existing autoregressive generation pipelines. Extensive experiments across four reasoning benchmarks (i.e., AIME24, AMC23, MATH500, and GPQA-D) demonstrate CGRS's effectiveness: it reduces token usage by an average of 18.5% to 41.9% while preserving accuracy. It also achieves the optimal balance between length reduction and performance compared to state-of-the-art baselines. These results hold consistently across model architectures (e.g., DeepSeek-R1-Distill series, QwQ-32B, and Qwen3 family) and scales (4B to 32B parameters), highlighting CGRS's practical value for efficient reasoning.
>
---
#### [replaced 032] Human-in-the-Loop Generation of Adversarial Texts: A Case Study on Tibetan Script
- **分类: cs.CL; cs.CR; cs.HC**

- **链接: [https://arxiv.org/pdf/2412.12478v5](https://arxiv.org/pdf/2412.12478v5)**

> **作者:** Xi Cao; Yuan Sun; Jiajun Li; Quzong Gesang; Nuo Qun; Tashi Nyima
>
> **备注:** Camera-Ready Version; Accepted at IJCNLP-AACL 2025 Demo
>
> **摘要:** DNN-based language models excel across various NLP tasks but remain highly vulnerable to textual adversarial attacks. While adversarial text generation is crucial for NLP security, explainability, evaluation, and data augmentation, related work remains overwhelmingly English-centric, leaving the problem of constructing high-quality and sustainable adversarial robustness benchmarks for lower-resourced languages both difficult and understudied. First, method customization for lower-resourced languages is complicated due to linguistic differences and limited resources. Second, automated attacks are prone to generating invalid or ambiguous adversarial texts. Last but not least, language models continuously evolve and may be immune to parts of previously generated adversarial texts. To address these challenges, we introduce HITL-GAT, an interactive system based on a general approach to human-in-the-loop generation of adversarial texts. Additionally, we demonstrate the utility of HITL-GAT through a case study on Tibetan script, employing three customized adversarial text generation methods and establishing its first adversarial robustness benchmark, providing a valuable reference for other lower-resourced languages.
>
---
#### [replaced 033] Don't Pay Attention
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2506.11305v2](https://arxiv.org/pdf/2506.11305v2)**

> **作者:** Mohammad Hammoud; Devang Acharya
>
> **摘要:** The Transformer has become the de facto standard for modern language models owing to its parallelizable training and effective autoregressive decoding. However, its fixed context window and the quadratic time and memory costs of its self-attention mechanism remain central bottlenecks. These constraints have revived interest in recurrent architectures that scale linearly with sequence length, but at the cost of reduced parallelism. In this paper, we introduce Avey, a new foundational architecture that breaks away from both attention and recurrence. Avey pairs a ranker with an autoregressive neural processor to select and contextualize only the most relevant tokens for any given token. Specifically, it decouples sequence length from context width, thus enabling effective and efficient processing of arbitrarily long sequences. Results show that Avey compares favorably to the Transformer across a variety of standard short-range NLP benchmarks, while significantly outperforming it on tasks requiring long-range dependency modeling.
>
---
#### [replaced 034] ReviewGraph: A Knowledge Graph Embedding Based Framework for Review Rating Prediction with Sentiment Features
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2508.13953v2](https://arxiv.org/pdf/2508.13953v2)**

> **作者:** A. J. W. de Vink; Natalia Amat-Lefort; Lifeng Han
>
> **备注:** Peer-reviewed and published version is in ICKG-2025 (The 16th IEEE International Conference on Knowledge Graphs, November 13-14, 2025, Limassol, Cyprus)
>
> **摘要:** In the hospitality industry, understanding the factors that drive customer review ratings is critical for improving guest satisfaction and business performance. This work proposes ReviewGraph for Review Rating Prediction (RRP), a novel framework that transforms textual customer reviews into knowledge graphs by extracting (subject, predicate, object) triples and associating sentiment scores. Using graph embeddings (Node2Vec) and sentiment features, the framework predicts review rating scores through machine learning classifiers. We compare ReviewGraph performance with traditional NLP baselines (such as Bag of Words, TF-IDF, and Word2Vec) and large language models (LLMs), evaluating them in the HotelRec dataset. In comparison to the state of the art literature, our proposed model performs similar to their best performing model but with lower computational cost (without ensemble). While ReviewGraph achieves comparable predictive performance to LLMs and outperforms baselines on agreement-based metrics such as Cohen's Kappa, it offers additional advantages in interpretability, visual exploration, and potential integration into Retrieval-Augmented Generation (RAG) systems. This work highlights the potential of graph-based representations for enhancing review analytics and lays the groundwork for future research integrating advanced graph neural networks and fine-tuned LLM-based extraction methods. We will share ReviewGraph output and platform open-sourced on our GitHub page https://github.com/aaronlifenghan/ReviewGraph
>
---
#### [replaced 035] RAG-R1: Incentivizing the Search and Reasoning Capabilities of LLMs through Multi-query Parallelism
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [https://arxiv.org/pdf/2507.02962v5](https://arxiv.org/pdf/2507.02962v5)**

> **作者:** Zhiwen Tan; Jiaming Huang; Qintong Wu; Hongxuan Zhang; Chenyi Zhuang; Jinjie Gu
>
> **摘要:** Large Language Models (LLMs), despite their remarkable capabilities, are prone to generating hallucinated or outdated content due to their static internal knowledge. While Retrieval-Augmented Generation (RAG) integrated with Reinforcement Learning (RL) offers a solution, these methods are fundamentally constrained by a single-query mode, leading to prohibitive latency and inherent brittleness. To overcome these limitations, we introduce RAG-R1, a novel two-stage training framework centered around multi-query parallelism. Our framework enables LLMs to adaptively leverage internal and external knowledge during the reasoning process while transitioning from the single-query mode to multi-query parallelism. This architectural shift bolsters reasoning robustness while significantly reducing inference latency. Extensive experiments on seven question-answering benchmarks confirm the superiority of our method, which outperforms the strongest baseline by up to 13.7% and decreases inference time by 11.1%.
>
---
#### [replaced 036] VocalBench-zh: Decomposing and Benchmarking the Speech Conversational Abilities in Mandarin Context
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2511.08230v2](https://arxiv.org/pdf/2511.08230v2)**

> **作者:** Heyang Liu; Ziyang Cheng; Yuhao Wang; Hongcheng Liu; Yiqi Li; Ronghua Wu; Qunshan Gu; Yanfeng Wang; Yu Wang
>
> **备注:** This article will serve as an extension of the preceding work, "VocalBench: Benchmarking the Vocal Conversational Abilities for Speech Interaction Models" (arXiv:2505.15727). Therefore, we have chosen to withdraw to avoid potential duplicate publication. We will update the previously open-sourced paper of VocalBench in several weeks to include the content of VocalBench-zh
>
> **摘要:** The development of multi-modal large language models (LLMs) leads to intelligent approaches capable of speech interactions. As one of the most widely spoken languages globally, Mandarin is supported by most models to enhance their applicability and reach. However, the scarcity of comprehensive speech-to-speech (S2S) benchmarks in Mandarin contexts impedes systematic evaluation for developers and hinders fair model comparison for users. In this work, we propose VocalBench-zh, an ability-level divided evaluation suite adapted to Mandarin context consisting of 10 well-crafted subsets and over 10K high-quality instances, covering 12 user-oriented characters. The evaluation experiment on 14 mainstream models reveals the common challenges for current routes, and highlights the need for new insights into next-generation speech interactive systems. The evaluation codes and datasets will be available at https://github.com/SJTU-OmniAgent/VocalBench-zh.
>
---
#### [replaced 037] The taggedPBC: Annotating a massive parallel corpus for crosslinguistic investigations
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2505.12560v3](https://arxiv.org/pdf/2505.12560v3)**

> **作者:** Hiram Ring
>
> **摘要:** Existing datasets available for crosslinguistic investigations have tended to focus on large amounts of data for a small group of languages or a small amount of data for a large number of languages. This means that claims based on these datasets are limited in what they reveal about universal properties of the human language faculty. While this has begun to change through the efforts of projects seeking to develop tagged corpora for a large number of languages, such efforts are still constrained by limits on resources. The current paper reports on a large tagged parallel dataset which has been developed to partially address this issue. The taggedPBC contains POS-tagged parallel text data from more than 1,940 languages, representing 155 language families and 78 isolates, dwarfing previously available resources. The accuracy of particular tags in this dataset is shown to correlate well with both existing SOTA taggers for high-resource languages (SpaCy, Trankit) as well as hand-tagged corpora (Universal Dependencies Treebanks). Additionally, a novel measure derived from this dataset, the N1 ratio, correlates with expert determinations of intransitive word order in three typological databases (WALS, Grambank, Autotyp) such that a Gaussian Naive Bayes classifier trained on this feature can accurately identify basic intransitive word order for languages not in those databases. While much work is still needed to expand and develop this dataset, the taggedPBC is an important step to enable corpus-based crosslinguistic investigations, and is made available for research and collaboration via GitHub.
>
---
#### [replaced 038] Historical/temporal necessities/possibilities, and a logical theory of them in branching time
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2208.11922v2](https://arxiv.org/pdf/2208.11922v2)**

> **作者:** Fengkui Ju; Woxuan Zhou
>
> **摘要:** In this paper, we do three kinds of work. First, we recognize four notions of necessity and two notions of possibility related to time flow, namely strong/weak historical/temporal necessities, as well as historical/temporal possibilities, which are motivated more from a linguistic perspective than from a philosophical one. Strong/weak historical necessities and historical possibility typically concern the possible futures of the present world, and strong/weak temporal necessities and temporal possibility concern possible timelines of alternatives of the present world. Second, we provide our approach to the six notions and present a logical theory of them in branching time. Our approach to the six notions is as follows. The agent has a system of ontic rules that determine expected timelines. She treats some ontic rules as undefeatable, determining accepted timelines. The domains of strong/weak historical necessities, respectively, consist of accepted and expected timelines passing through the present moment, and historical possibility is the dual of strong historical necessity. The domains of strong/weak temporal necessities, respectively, consist of accepted and expected timelines, and temporal possibility is the dual of strong temporal necessity. The logical theory has six operators: a last-moment operator, a next-moment operator, and four operators for the four notions of necessity. Formulas' evaluation contexts consist of a tree-like model representing a time flow, a context representing the agent's system of ontic rules, a timeline, and an instant. Third, we offer an axiomatic system for the logical theory and show its soundness and completeness.
>
---
#### [replaced 039] ToxSyn: Reducing Bias in Hate Speech Detection via Synthetic Minority Data in Brazilian Portuguese
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2506.10245v2](https://arxiv.org/pdf/2506.10245v2)**

> **作者:** Iago Alves Brito; Julia Soares Dollis; Fernanda Bufon Färber; Diogo Fernandes Costa Silva; Arlindo Rodrigues Galvão Filho
>
> **备注:** 10 pages, 5 tables, 1 figure
>
> **摘要:** The development of robust hate speech detection systems remains limited by the lack of large-scale, fine-grained training data, especially for languages beyond English. Existing corpora typically rely on coarse toxic/non-toxic labels, and the few that capture hate directed at specific minority groups critically lack the non-toxic counterexamples (i.e., benign text about minorities) required to distinguish genuine hate from mere discussion. We introduce ToxSyn, the first Portuguese large-scale corpus explicitly designed for multi-label hate speech detection across nine protected minority groups. Generated via a controllable four-stage pipeline, ToxSyn includes discourse-type annotations to capture rhetorical strategies of toxic language, such as sarcasm or dehumanization. Crucially, it systematically includes the non-toxic counterexamples absent in all other public datasets. Our experiments reveal a catastrophic, mutual generalization failure between social-media domains and ToxSyn: models trained on social media struggle to generalize to minority-specific contexts, and vice-versa. This finding indicates they are distinct tasks and exposes summary metrics like Macro F1 can be unreliable indicators of true model behavior, as they completely mask model failure. We publicly release ToxSyn at HuggingFace to foster reproducible research on synthetic data generation and benchmark progress in hate-speech detection for low- and mid-resource languages.
>
---
#### [replaced 040] Language Model Distillation: A Temporal Difference Imitation Learning Perspective
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2505.20335v3](https://arxiv.org/pdf/2505.20335v3)**

> **作者:** Zishun Yu; Shangzhe Li; Xinhua Zhang
>
> **备注:** AAAI 2026
>
> **摘要:** Large language models have led to significant progress across many NLP tasks, although their massive sizes often incur substantial computational costs. Distillation has become a common practice to compress these large and highly capable models into smaller, more efficient ones. Many existing language model distillation methods can be viewed as behavior cloning from the perspective of imitation learning or inverse reinforcement learning. This viewpoint has inspired subsequent studies that leverage (inverse) reinforcement learning techniques, including variations of behavior cloning and temporal difference learning methods. Rather than proposing yet another specific temporal difference method, we introduce a general framework for temporal difference-based distillation by exploiting the distributional sparsity of the teacher model. Specifically, it is often observed that language models assign most probability mass to a small subset of tokens. Motivated by this observation, we design a temporal difference learning framework that operates on a reduced action space (a subset of vocabulary), and demonstrate how practical algorithms can be derived and the resulting performance improvements.
>
---
#### [replaced 041] NyayaRAG: Realistic Legal Judgment Prediction with RAG under the Indian Common Law System
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **链接: [https://arxiv.org/pdf/2508.00709v3](https://arxiv.org/pdf/2508.00709v3)**

> **作者:** Shubham Kumar Nigam; Balaramamahanthi Deepak Patnaik; Shivam Mishra; Ajay Varghese Thomas; Noel Shallum; Kripabandhu Ghosh; Arnab Bhattacharya
>
> **备注:** Paper accepted in the AACL-IJCNLP 2025 conference
>
> **摘要:** Legal Judgment Prediction (LJP) has emerged as a key area in AI for law, aiming to automate judicial outcome forecasting and enhance interpretability in legal reasoning. While previous approaches in the Indian context have relied on internal case content such as facts, issues, and reasoning, they often overlook a core element of common law systems, which is reliance on statutory provisions and judicial precedents. In this work, we propose NyayaRAG, a Retrieval-Augmented Generation (RAG) framework that simulates realistic courtroom scenarios by providing models with factual case descriptions, relevant legal statutes, and semantically retrieved prior cases. NyayaRAG evaluates the effectiveness of these combined inputs in predicting court decisions and generating legal explanations using a domain-specific pipeline tailored to the Indian legal system. We assess performance across various input configurations using both standard lexical and semantic metrics as well as LLM-based evaluators such as G-Eval. Our results show that augmenting factual inputs with structured legal knowledge significantly improves both predictive accuracy and explanation quality.
>
---
#### [replaced 042] Hogwild! Inference: Parallel LLM Generation via Concurrent Attention
- **分类: cs.LG; cs.CL**

- **链接: [https://arxiv.org/pdf/2504.06261v4](https://arxiv.org/pdf/2504.06261v4)**

> **作者:** Gleb Rodionov; Roman Garipov; Alina Shutova; George Yakushev; Erik Schultheis; Vage Egiazarian; Anton Sinitsin; Denis Kuznedelev; Dan Alistarh
>
> **备注:** 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
>
> **摘要:** Large Language Models (LLMs) have demonstrated the ability to tackle increasingly complex tasks through advanced reasoning, long-form content generation, and tool use. Solving these tasks often involves long inference-time computations. In human problem solving, a common strategy to expedite work is collaboration: by dividing the problem into sub-tasks, exploring different strategies concurrently, etc. Recent research has shown that LLMs can also operate in parallel by implementing explicit cooperation frameworks, such as voting mechanisms or the explicit creation of independent sub-tasks that can be executed in parallel. However, each of these frameworks may not be suitable for all types of tasks, which can hinder their applicability. In this work, we propose a different design approach: we run LLM "workers" in parallel , allowing them to synchronize via a concurrently-updated attention cache and prompt these workers to decide how best to collaborate. Our approach allows the LLM instances to come up with their own collaboration strategy for the problem at hand, all the while "seeing" each other's memory in the concurrent KV cache. We implement this approach via Hogwild! Inference: a parallel LLM inference engine where multiple instances of the same LLM run in parallel with the same attention cache, with "instant" access to each other's memory. Hogwild! Inference takes advantage of Rotary Position Embeddings (RoPE) to avoid recomputation while improving parallel hardware utilization. We find that modern reasoning-capable LLMs can perform inference with shared Key-Value cache out of the box, without additional fine-tuning.
>
---
#### [replaced 043] SCRum-9: Multilingual Stance Classification over Rumours on Social Media
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2505.18916v3](https://arxiv.org/pdf/2505.18916v3)**

> **作者:** Yue Li; Jake Vasilakes; Zhixue Zhao; Carolina Scarton
>
> **备注:** Accepted by ICWSM 2026
>
> **摘要:** We introduce SCRum-9, the largest multilingual Stance Classification dataset for Rumour analysis in 9 languages, containing 7,516 tweets from X. SCRum-9 goes beyond existing stance classification datasets by covering more languages, linking examples to more fact-checked claims (2.1k), and including confidence-related annotations from multiple annotators to account for intra- and inter-annotator variability. Annotations were made by at least two native speakers per language, totalling more than 405 hours of annotation and 8,150 dollars in compensation. Further, SCRum-9 is used to benchmark five large language models (LLMs) and two multilingual masked language models (MLMs) in In-Context Learning (ICL) and fine-tuning setups. This paper also innovates by exploring the use of multilingual synthetic data for rumour stance classification, showing that even LLMs with weak ICL performance can produce valuable synthetic data for fine-tuning small MLMs, enabling them to achieve higher performance than zero-shot ICL in LLMs. Finally, we examine the relationship between model predictions and human uncertainty on ambiguous cases finding that model predictions often match the second-choice labels assigned by annotators, rather than diverging entirely from human judgments. SCRum-9 is publicly released to the research community with potential to foster further research on multilingual analysis of misleading narratives on social media.
>
---
#### [replaced 044] DataGen: Unified Synthetic Dataset Generation via Large Language Models
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2406.18966v5](https://arxiv.org/pdf/2406.18966v5)**

> **作者:** Yue Huang; Siyuan Wu; Chujie Gao; Dongping Chen; Qihui Zhang; Yao Wan; Tianyi Zhou; Jianfeng Gao; Chaowei Xiao; Lichao Sun; Xiangliang Zhang
>
> **摘要:** Large Language Models (LLMs) such as GPT-4 and Llama3 have significantly impacted various fields by enabling high-quality synthetic data generation and reducing dependence on expensive human-generated datasets. Despite this, challenges remain in the areas of generalization, controllability, diversity, and truthfulness within the existing generative frameworks. To address these challenges, this paper presents DataGen, a comprehensive LLM-powered framework designed to produce diverse, accurate, and highly controllable datasets. DataGen is adaptable, supporting all types of text datasets and enhancing the generative process through innovative mechanisms. To augment data diversity, DataGen incorporates an attribute-guided generation module and a group checking feature. For accuracy, it employs a code-based mathematical assessment for label verification alongside a retrieval-augmented generation technique for factual validation. The framework also allows for user-specified constraints, enabling customization of the data generation process to suit particular requirements. Extensive experiments demonstrate the superior quality of data generated by DataGen, and each module within DataGen plays a critical role in this enhancement. Additionally, DataGen is applied in two practical scenarios: benchmarking LLMs and data augmentation. The results indicate that DataGen effectively supports dynamic and evolving benchmarking and that data augmentation improves LLM capabilities in various domains, including agent-oriented abilities and reasoning skills.
>
---
#### [replaced 045] Chain-of-Conceptual-Thought Elicits Daily Conversation in Large Language Models
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2510.18434v3](https://arxiv.org/pdf/2510.18434v3)**

> **作者:** Qingqing Gu; Dan Wang; Yue Zhao; Xiaoyu Wang; Zhonglin Jiang; Yong Chen; Hongyan Li; Luo Ji
>
> **备注:** PRICAI 2025
>
> **摘要:** Chain-of-Thought (CoT) is widely applied to enhance the LLM capability in math, coding and reasoning tasks. However, its performance is limited for open-domain tasks, when there are no clearly defined reasoning steps or logical transitions. To mitigate such challenges, we propose a new prompt-based paradigm called Chain of Conceptual Thoughts (CoCT), which suggests the LLM first to produce the tag of concepts, then complete the detailed content following the concept. To encourage this hierarchical way of thinking, we implement the concepts with emotions, strategies and topics. We experiment with this paradigm in daily and emotional support conversations, covering tasks with both in-domain and out-of-domain concept settings. Automatic, human, and LLM-based evaluations reveal that CoCT surpasses several prompt-based baselines such as self-refine, ECoT, SoT and RAG, suggesting a potential solution of LLM prompting paradigm for a wider scope of tasks.
>
---
#### [replaced 046] C$^3$TG: Conflict-aware, Composite, and Collaborative Controlled Text Generation
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.09292v2](https://arxiv.org/pdf/2511.09292v2)**

> **作者:** Yu Li; Zhe Yang; Yi Huang; Xin Liu; Guilin Qi
>
> **备注:** This paper has been accepted as a poster presentation at AAAI-2026
>
> **摘要:** Recent advancements in large language models (LLMs) have demonstrated remarkable text generation capabilities. However, controlling specific attributes of generated text remains challenging without architectural modifications or extensive fine-tuning. Current methods typically toggle a single, basic attribute but struggle with precise multi-attribute control. In scenarios where attribute requirements conflict, existing methods lack coordination mechanisms, causing interference between desired attributes. Furthermore, these methods fail to incorporate iterative optimization processes in the controlled generation pipeline. To address these limitations, we propose Conflict-aware, Composite, and Collaborative Controlled Text Generation (C$^3$TG), a two-phase framework for fine-grained, multi-dimensional text attribute control. During generation, C$^3$TG selectively pairs the LLM with the required attribute classifiers from the 17 available dimensions and employs weighted KL-divergence to adjust token probabilities. The optimization phase then leverages an energy function combining classifier scores and penalty terms to resolve attribute conflicts through iterative feedback, enabling precise control over multiple dimensions simultaneously while preserving natural text flow. Experiments show that C$^3$TG significantly outperforms baselines across multiple metrics including attribute accuracy, linguistic fluency, and output diversity, while simultaneously reducing toxicity. These results establish C$^3$TG as an effective and flexible solution for multi-dimensional text attribute control that requires no costly model modifications.
>
---
#### [replaced 047] VIR-Bench: Evaluating Geospatial and Temporal Understanding of MLLMs via Travel Video Itinerary Reconstruction
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [https://arxiv.org/pdf/2509.19002v2](https://arxiv.org/pdf/2509.19002v2)**

> **作者:** Hao Wang; Eiki Murata; Lingfang Zhang; Ayako Sato; So Fukuda; Ziqi Yin; Wentao Hu; Keisuke Nakao; Yusuke Nakamura; Sebastian Zwirner; Yi-Chia Chen; Hiroyuki Otomo; Hiroki Ouchi; Daisuke Kawahara
>
> **备注:** AAAI 2026
>
> **摘要:** Recent advances in multimodal large language models (MLLMs) have significantly enhanced video understanding capabilities, opening new possibilities for practical applications. Yet current video benchmarks focus largely on indoor scenes or short-range outdoor activities, leaving the challenges associated with long-distance travel largely unexplored. Mastering extended geospatial-temporal trajectories is critical for next-generation MLLMs, underpinning real-world tasks such as embodied-AI planning and navigation. To bridge this gap, we present VIR-Bench, a novel benchmark consisting of 200 travel videos that frames itinerary reconstruction as a challenging task designed to evaluate and push forward MLLMs' geospatial-temporal intelligence. Experimental results reveal that state-of-the-art MLLMs, including proprietary ones, struggle to achieve high scores, underscoring the difficulty of handling videos that span extended spatial and temporal scales. Moreover, we conduct an in-depth case study in which we develop a prototype travel-planning agent that leverages the insights gained from VIR-Bench. The agent's markedly improved itinerary recommendations verify that our evaluation protocol not only benchmarks models effectively but also translates into concrete performance gains in user-facing applications.
>
---
#### [replaced 048] A Super-Learner with Large Language Models for Medical Emergency Advising
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2511.08614v2](https://arxiv.org/pdf/2511.08614v2)**

> **作者:** Sergey K. Aityan; Abdolreza Mosaddegh; Rolando Herrero; Haitham Tayyar; Jiang Han; Vikram Sawant; Qi Chen; Rishabh Jain; Aruna Senthamaraikannan; Stephen Wood; Manuel Mersini; Rita Lazzaro; Mario Balzaneli; Nicola Iacovazzo; Ciro Gargiulo Isacco
>
> **备注:** 12 pages, 3 figures, 2 tables
>
> **摘要:** Medical decision-support and advising systems are critical for emergency physicians to quickly and accurately assess patients' conditions and make diagnosis. Artificial Intelligence (AI) has emerged as a transformative force in healthcare in recent years and Large Language Models (LLMs) have been employed in various fields of medical decision-support systems. We studied responses of a group of different LLMs to real cases in emergency medicine. The results of our study on five most renown LLMs showed significant differences in capabilities of Large Language Models for diagnostics acute diseases in medical emergencies with accuracy ranging between 58% and 65%. This accuracy significantly exceeds the reported accuracy of human doctors. We built a super-learner MEDAS (Medical Emergency Diagnostic Advising System) of five major LLMs - Gemini, Llama, Grok, GPT, and Claude). The super-learner produces higher diagnostic accuracy, 70%, even with a quite basic meta-learner. However, at least one of the integrated LLMs in the same super-learner produces 85% correct diagnoses. The super-learner integrates a cluster of LLMs using a meta-learner capable of learning different capabilities of each LLM to leverage diagnostic accuracy of the model by collective capabilities of all LLMs in the cluster. The results of our study showed that aggregated diagnostic accuracy provided by a meta-learning approach exceeds that of any individual LLM, suggesting that the super-learner can take advantage of the combined knowledge of the medical datasets used to train the group of LLMs.
>
---
#### [replaced 049] Vashantor: A Large-scale Multilingual Benchmark Dataset for Automated Translation of Bangla Regional Dialects to Bangla Language
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2311.11142v2](https://arxiv.org/pdf/2311.11142v2)**

> **作者:** Fatema Tuj Johora Faria; Mukaffi Bin Moin; Ahmed Al Wase; Mehidi Ahmmed; Md. Rabius Sani; Tashreef Muhammad
>
> **摘要:** The Bangla linguistic variety is a fascinating mix of regional dialects that contributes to the cultural diversity of the Bangla-speaking community. Despite extensive study into translating Bangla to English, English to Bangla, and Banglish to Bangla in the past, there has been a noticeable gap in translating Bangla regional dialects into standard Bangla. In this study, we set out to fill this gap by creating a collection of 32,500 sentences, encompassing Bangla, Banglish, and English, representing five regional Bangla dialects. Our aim is to translate these regional dialects into standard Bangla and detect regions accurately. To tackle the translation and region detection tasks, we propose two novel models: DialectBanglaT5 for translating regional dialects into standard Bangla and DialectBanglaBERT for identifying the dialect's region of origin. DialectBanglaT5 demonstrates superior performance across all dialects, achieving the highest BLEU score of 71.93, METEOR of 0.8503, and the lowest WER of 0.1470 and CER of 0.0791 on the Mymensingh dialect. It also achieves strong ROUGE scores across all dialects, indicating both accuracy and fluency in capturing dialectal nuances. In parallel, DialectBanglaBERT achieves an overall region classification accuracy of 89.02%, with notable F1-scores of 0.9241 for Chittagong and 0.8736 for Mymensingh, confirming its effectiveness in handling regional linguistic variation. This is the first large-scale investigation focused on Bangla regional dialect translation and region detection. Our proposed models highlight the potential of dialect-specific modeling and set a new benchmark for future research in low-resource and dialect-rich language settings.
>
---
#### [replaced 050] MLR-Copilot: Autonomous Machine Learning Research based on Large Language Models Agents
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [https://arxiv.org/pdf/2408.14033v3](https://arxiv.org/pdf/2408.14033v3)**

> **作者:** Ruochen Li; Teerth Patel; Qingyun Wang; Xinya Du
>
> **摘要:** Autonomous machine learning research has gained significant attention recently. We present MLR-COPILOT, an autonomous Machine Learning Research framework powered by large language model agents. The system is designed to enhance ML research productivity through automatic generation and implementation of research ideas within constraints. Our work was released in August 2024 (concurrent to AI-Scientist) and has gained notable recognition from leading projects. We further enhance our ideation with training afterwards. The framework consists of three stages: idea generation, experiment implementation, and code execution. First, existing research papers are used to generate feasible ideas and experiment plans with IdeaAgent, powered by an RL-tuned LLM. Next, ExperimentAgent leverages retrieved prototype code to convert plans into executable code with optionally retrieved candidate models and data from HuggingFace. In the final stage, ExperimentAgent runs experiments, and allows subsequent iterations of debugging and human feedback for a better chance of success with executable outcomes. We evaluate our framework on five machine learning research tasks. Experiment results demonstrate the potential of our framework to facilitate ML research progress and innovation.
>
---
#### [replaced 051] Magellan: Guided MCTS for Latent Space Exploration and Novelty Generation
- **分类: cs.AI; cs.CL**

- **链接: [https://arxiv.org/pdf/2510.21341v2](https://arxiv.org/pdf/2510.21341v2)**

> **作者:** Lufan Chang
>
> **备注:** Accepted to 1st Open Conference on AI Agents for Science (agents4science 2025)
>
> **摘要:** Large Language Models (LLMs) often struggle with generating truly innovative ideas, typically defaulting to high-probability, familiar concepts within their training data's "gravity wells." While advanced search-based methods like Tree of Thoughts (ToT) attempt to mitigate this, they are fundamentally limited by their reliance on unprincipled, inconsistent self-evaluation heuristics to guide exploration. To address this gap, we introduce \textbf{Magellan}, a novel framework that reframes creative generation as a principled, guided exploration of an LLM's latent conceptual space. At its core, Magellan employs Monte Carlo Tree Search (MCTS) governed by a hierarchical guidance system. For long-range direction, a "semantic compass" vector, formulated via orthogonal projection, steers the search towards relevant novelty. For local, step-by-step decisions, a landscape-aware value function replaces flawed self-evaluation with an explicit reward structure that balances intrinsic coherence, extrinsic novelty, and narrative progress. Extensive experiments demonstrate that Magellan significantly outperforms strong baselines, including ReAct and ToT, in generating scientific ideas with superior plausibility and innovation. Our work shows that for creative discovery, a principled, guided search is more effective than unconstrained agency, paving the way for LLMs to become more capable partners in innovation.
>
---
#### [replaced 052] MAPLE: Multi-Agent Adaptive Planning with Long-Term Memory for Table Reasoning
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2506.05813v2](https://arxiv.org/pdf/2506.05813v2)**

> **作者:** Ye Bai; Minghan Wang; Thuy-Trang Vu
>
> **备注:** 27 pages, 11 figures, ALTA 2025
>
> **摘要:** Table-based question answering requires complex reasoning capabilities that current LLMs struggle to achieve with single-pass inference. Existing approaches, such as Chain-of-Thought reasoning and question decomposition, lack error detection mechanisms and discard problem-solving experiences, contrasting sharply with how humans tackle such problems. In this paper, we propose MAPLE (Multi-agent Adaptive Planning with Long-term mEmory), a novel framework that mimics human problem-solving through specialized cognitive agents working in a feedback-driven loop. MAPLE integrates 4 key components: (1) a Solver using the ReAct paradigm for reasoning, (2) a Checker for answer verification, (3) a Reflector for error diagnosis and strategy correction, and (4) an Archiver managing long-term memory for experience reuse and evolution. Experiments on WiKiTQ and TabFact demonstrate significant improvements over existing methods, achieving state-of-the-art performance across multiple LLM backbones.
>
---
#### [replaced 053] PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [https://arxiv.org/pdf/2502.14902v2](https://arxiv.org/pdf/2502.14902v2)**

> **作者:** Boyu Chen; Zirui Guo; Zidan Yang; Yuluo Chen; Junze Chen; Zhenghao Liu; Chuan Shi; Cheng Yang
>
> **摘要:** Retrieval-augmented generation (RAG) improves the response quality of large language models (LLMs) by retrieving knowledge from external databases. Typical RAG approaches split the text database into chunks, organizing them in a flat structure for efficient searches. To better capture the inherent dependencies and structured relationships across the text database, researchers propose to organize textual information into an indexing graph, known asgraph-based RAG. However, we argue that the limitation of current graph-based RAG methods lies in the redundancy of the retrieved information, rather than its insufficiency. Moreover, previous methods use a flat structure to organize retrieved information within the prompts, leading to suboptimal performance. To overcome these limitations, we propose PathRAG, which retrieves key relational paths from the indexing graph, and converts these paths into textual form for prompting LLMs. Specifically, PathRAG effectively reduces redundant information with flow-based pruning, while guiding LLMs to generate more logical and coherent responses with path-based prompting. Experimental results show that PathRAG consistently outperforms state-of-the-art baselines across six datasets and five evaluation dimensions. The code is available at the following link: https://github.com/BUPT-GAMMA/PathRAG
>
---
#### [replaced 054] Beyond Chains: Bridging Large Language Models and Knowledge Bases in Complex Question Answering
- **分类: cs.CL; cs.IR**

- **链接: [https://arxiv.org/pdf/2505.14099v2](https://arxiv.org/pdf/2505.14099v2)**

> **作者:** Yihua Zhu; Qianying Liu; Akiko Aizawa; Hidetoshi Shimodaira
>
> **备注:** AAAI2026 Main Track
>
> **摘要:** Knowledge Base Question Answering (KBQA) aims to answer natural language questions using structured knowledge from KBs. While LLM-only approaches offer generalization, they suffer from outdated knowledge, hallucinations, and lack of transparency. Chain-based KG-RAG methods address these issues by incorporating external KBs, but are limited to simple chain-structured questions due to the absence of planning and logical structuring. Inspired by semantic parsing methods, we propose PDRR: a four-stage framework consisting of Predict, Decompose, Retrieve, and Reason. Our method first predicts the question type and decomposes the question into structured triples. Then retrieves relevant information from KBs and guides the LLM as an agent to reason over and complete the decomposed triples. Experimental results demonstrate that PDRR consistently outperforms existing methods across various LLM backbones and achieves superior performance on both chain-structured and non-chain complex questions.
>
---
#### [replaced 055] Joint Evaluation of Answer and Reasoning Consistency for Hallucination Detection in Large Reasoning Models
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2506.04832v2](https://arxiv.org/pdf/2506.04832v2)**

> **作者:** Changyue Wang; Weihang Su; Qingyao Ai; Yiqun Liu
>
> **摘要:** Large Reasoning Models (LRMs) extend large language models with explicit, multi-step reasoning traces to enhance transparency and performance on complex tasks. However, these reasoning traces can be redundant or logically inconsistent, becoming a new and hard-to-detect source of hallucination. Existing hallucination detection methods focus primarily on answer-level uncertainty and often fail to detect hallucinations or logical inconsistencies arising from the model's reasoning trace. This oversight is particularly problematic for LRMs, where the explicit thinking trace is not only an important support to the model's decision-making process but also a key source of potential hallucination. To this end, we propose RACE (Reasoning and Answer Consistency Evaluation), a novel framework specifically tailored for hallucination detection in LRMs. RACE operates by extracting essential reasoning steps and computing four diagnostic signals: inter-sample consistency of reasoning traces, entropy-based answer uncertainty, semantic alignment between reasoning and answers, and internal coherence of reasoning. The joint utilization of these signals makes RACE a more robust detector of hallucinations in LRMs. Experiments across datasets and different LLMs demonstrate that RACE outperforms existing hallucination detection baselines, offering a robust and generalizable solution for evaluating LRMs. The source code is available at https://github.com/bebr2/RACE
>
---
#### [replaced 056] ProFuser: Progressive Fusion of Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2408.04998v2](https://arxiv.org/pdf/2408.04998v2)**

> **作者:** Tianyuan Shi; Fanqi Wan; Canbin Huang; Xiaojun Quan; Chenliang Li; Ming Yan; Ji Zhang; Minhua Huang; Wu Kai
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** While fusing the capacities and advantages of various large language models offers a pathway to construct more powerful and versatile models, a fundamental challenge is to properly select advantageous model during training. Existing fusion methods primarily focus on the training mode that uses cross entropy on ground truth in a teacher-forcing setup to measure a model's advantage, which may provide limited insight towards model advantage. In this paper, we introduce a novel approach that enhances the fusion process by incorporating both the training and inference modes. Our method evaluates model advantage not only through cross entropy during training but also by considering inference outputs, providing a more comprehensive assessment. To combine the two modes effectively, we introduce ProFuser to progressively transition from inference mode to training mode. To validate ProFuser's effectiveness, we fused three models, including Vicuna-7B-v1.5, Llama-2-7B-Chat, and MPT-7B-8K-Chat, and demonstrated the improved performance in knowledge, reasoning, and safety compared to baseline methods.
>
---
#### [replaced 057] A Human Behavioral Baseline for Collective Governance in Software Projects
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.08956v2](https://arxiv.org/pdf/2510.08956v2)**

> **作者:** Mobina Noori; Mahasweta Chakraborti; Amy X Zhang; Seth Frey
>
> **备注:** Algorithmic Collective Action Workshop @ NeurIPS 2025. arXiv admin note: text overlap with arXiv:2509.16295
>
> **摘要:** We study how open source communities describe participation and control through version controlled governance documents. Using a corpus of 710 projects with paired snapshots, we parse text into actors, rules, actions, and objects, then group them and measure change with entropy for evenness, richness for diversity, and Jensen Shannon divergence for drift. Projects define more roles and more actions over time, and these are distributed more evenly, while the composition of rules remains stable. These findings indicate that governance grows by expanding and balancing categories of participation without major shifts in prescriptive force. The analysis provides a reproducible baseline for evaluating whether future AI mediated workflows concentrate or redistribute authority.
>
---
#### [replaced 058] RATTENTION: Towards the Minimal Sliding Window Size in Local-Global Attention Models
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2506.15545v2](https://arxiv.org/pdf/2506.15545v2)**

> **作者:** Bailin Wang; Chang Lan; Chong Wang; Ruoming Pang
>
> **备注:** 9 pages
>
> **摘要:** Local-global attention models have recently emerged as compelling alternatives to standard Transformers, promising improvements in both training and inference efficiency. However, the crucial choice of window size presents a Pareto tradeoff: larger windows maintain performance akin to full attention but offer minimal efficiency gains in short-context scenarios, while smaller windows can lead to performance degradation. Current models, such as Gemma2 and Mistral, adopt conservative window sizes (e.g., 4096 out of an 8192 pretraining length) to preserve performance. This work investigates strategies to shift this Pareto frontier, enabling local-global models to achieve efficiency gains even in short-context regimes. Our core motivation is to address the intrinsic limitation of local attention -- its complete disregard for tokens outside the defined window. We explore RATTENTION, a variant of local attention integrated with a specialized linear attention mechanism designed to capture information from these out-of-window tokens. Pretraining experiments at the 3B and 12B scales demonstrate that RATTENTION achieves a superior Pareto tradeoff between performance and efficiency. As a sweet spot, RATTENTION with a window size of just 512 consistently matches the performance of full-attention models across diverse settings. Furthermore, the recurrent nature inherent in the linear attention component of RATTENTION contributes to enhanced long-context performance, as validated on the RULER benchmark. Crucially, these improvements do not compromise training efficiency; thanks to a specialized kernel implementation and the reduced window size, RATTENTION maintains training speeds comparable to existing state-of-the-art approaches. We open-sourced our Pallas kernels along with model codes to facilitate further research effort.
>
---
#### [replaced 059] Leveraging Online Data to Enhance Medical Knowledge in a Small Persian Language Model
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2505.16000v5](https://arxiv.org/pdf/2505.16000v5)**

> **作者:** Mehrdad Ghassabi; Pedram Rostami; Hamidreza Baradaran Kashani; Amirhossein Poursina; Zahra Kazemi; Milad Tavakoli
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** The rapid advancement of language models has demonstrated the potential of artificial intelligence in the healthcare industry. However, small language models struggle with specialized domains in low-resource languages like Persian. While numerous medical-domain websites exist in Persian, no curated dataset or corpus has been available making ours the first of its kind. This study introduces a newly curated dataset comprising 20k doctor-patient Q\&A pairs and 60\% of a 90-million-token crawled corpus from medical magazines. Using a parameter-efficient fine-tuning approach, we enhanced the medical knowledge of the baseline model, aya-expanse-8b. Benchmark evaluations demonstrate that the fine-tuned model achieves improved accuracy in medical question answering and successfully passed the Iranian Basic Medical Science Entrance Exam (IBSEE) in September 2023, which the baseline model did not. Additionally, the fine-tuned model improved Persian-translated MMLU accuracy by an average of 2.67\%. This work highlights the potential of leveraging open-access online data to enrich small language models in medical fields, providing a novel solution for Persian medical AI applications suitable for resource-constrained environments. Future research could explore multimodal input to further enhance performance.
>
---
#### [replaced 060] Explain with Visual Keypoints Like a Real Mentor! A Benchmark for Multimodal Solution Explanation
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2504.03197v4](https://arxiv.org/pdf/2504.03197v4)**

> **作者:** Jaewoo Park; Jungyang Park; Dongju Jang; Jiwan Chung; Byungwoo Yoo; Jaewoo Shin; Seonjoon Park; Taehyeong Kim; Youngjae Yu
>
> **备注:** 14 pages, 9 figures
>
> **摘要:** With the rapid advancement of mathematical reasoning capabilities in Large Language Models (LLMs), AI systems are increasingly being adopted in educational settings to support students' comprehension of problem-solving processes. However, a critical component remains underexplored in current LLM-generated explanations: multimodal explanation. In real-world instructional contexts, human tutors routinely employ visual aids, such as diagrams, markings, and highlights, to enhance conceptual clarity. To bridge this gap, we introduce the multimodal solution explanation task, designed to evaluate whether models can identify visual keypoints, such as auxiliary lines, points, angles, and generate explanations that incorporate these key elements essential for understanding. To evaluate model performance on this task, we propose ME2, a multimodal benchmark consisting of 1,000 math problems annotated with visual keypoints and corresponding explanatory text that references those elements. Our empirical results show that current models struggle to identify visual keypoints. In the task of generating keypoint-based explanations, open-source models also face notable difficulties. This highlights a significant gap in current LLMs' ability to perform mathematical visual grounding, engage in visually grounded reasoning, and provide explanations in educational contexts. We expect that the multimodal solution explanation task and the ME2 dataset will catalyze further research on LLMs in education and promote their use as effective, explanation-oriented AI tutors.
>
---
#### [replaced 061] Read Between the Lines: A Benchmark for Uncovering Political Bias in Bangla News Articles
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2510.03898v2](https://arxiv.org/pdf/2510.03898v2)**

> **作者:** Nusrat Jahan Lia; Shubhashis Roy Dipta; Abdullah Khan Zehady; Naymul Islam; Madhusodan Chakraborty; Abdullah Al Wasif
>
> **备注:** Accepted to BLP at AACL-IJCNLP 2025
>
> **摘要:** Detecting media bias is crucial, specifically in the South Asian region. Despite this, annotated datasets and computational studies for Bangla political bias research remain scarce. Crucially because, political stance detection in Bangla news requires understanding of linguistic cues, cultural context, subtle biases, rhetorical strategies, code-switching, implicit sentiment, and socio-political background. To address this, we introduce the first benchmark dataset of 200 politically significant and highly debated Bangla news articles, labeled for government-leaning, government-critique, and neutral stances, alongside diagnostic analyses for evaluating large language models (LLMs). Our comprehensive evaluation of 28 proprietary and open-source LLMs shows strong performance in detecting government-critique content (F1 up to 0.83) but substantial difficulty with neutral articles (F1 as low as 0.00). Models also tend to over-predict government-leaning stances, often misinterpreting ambiguous narratives. This dataset and its associated diagnostics provide a foundation for advancing stance detection in Bangla media research and offer insights for improving LLM performance in low-resource languages.
>
---
#### [replaced 062] Interpreting the Effects of Quantization on LLMs
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [https://arxiv.org/pdf/2508.16785v2](https://arxiv.org/pdf/2508.16785v2)**

> **作者:** Manpreet Singh; Hassan Sajjad
>
> **备注:** Accepted to AACL 2025 Main
>
> **摘要:** Quantization offers a practical solution to deploy LLMs in resource-constraint environments. However, its impact on internal representations remains understudied, raising questions about the reliability of quantized models. In this study, we employ a range of interpretability techniques to investigate how quantization affects model and neuron behavior. We analyze multiple LLMs under 4-bit and 8-bit quantization. Our findings reveal that the impact of quantization on model calibration is generally minor. Analysis of neuron activations indicates that the number of dead neurons, i.e., those with activation values close to 0 across the dataset, remains consistent regardless of quantization. In terms of neuron contribution to predictions, we observe that smaller full precision models exhibit fewer salient neurons, whereas larger models tend to have more, with the exception of Llama-2-7B. The effect of quantization on neuron redundancy varies across models. Overall, our findings suggest that effect of quantization may vary by model and tasks, however, we did not observe any drastic change which may discourage the use of quantization as a reliable model compression technique.
>
---
#### [replaced 063] Better Language Model-Based Judging Reward Modeling through Scaling Comprehension Boundaries
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2508.18212v2](https://arxiv.org/pdf/2508.18212v2)**

> **作者:** Meiling Ning; Zhongbao Zhang; Junda Ye; Jiabao Guo; Qingyuan Guan
>
> **备注:** After further internal discussion, our author team has decided to withdraw this submission due to the need for several important refinements to the manuscript. All co-authors have been informed and agree with this decision
>
> **摘要:** The emergence of LM-based judging reward modeling, represented by generative reward models, has successfully made reinforcement learning from AI feedback (RLAIF) efficient and scalable. To further advance this paradigm, we propose a core insight: this form of reward modeling shares fundamental formal consistency with natural language inference (NLI), a core task in natural language understanding. This reframed perspective points to a key path for building superior reward models: scaling the model's comprehension boundaries. Pursuing this path, exploratory experiments on NLI tasks demonstrate that the slot prediction masked language models (MLMs) incorporating contextual explanations achieve significantly better performance compared to mainstream autoregressive models. Based on this key finding, we propose ESFP-RM, a two-stage LM-based judging reward model that utilizes an explanation based slot framework for prediction to fully leverage the advantages of MLMs. Extensive experiments demonstrate that in both reinforcement learning from human feedback (RLHF) and out-of-distribution (OOD) scenarios, the ESFP-RM framework delivers more stable and generalizable reward signals compared to generative reward models.
>
---
#### [replaced 064] Do Language Models Associate Sound with Meaning? A Multimodal Study of Sound Symbolism
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2511.10045v2](https://arxiv.org/pdf/2511.10045v2)**

> **作者:** Jinhong Jeong; Sunghyun Lee; Jaeyoung Lee; Seonah Han; Youngjae Yu
>
> **备注:** 33 pages, 27 tables, 10 figures
>
> **摘要:** Sound symbolism is a linguistic concept that refers to non-arbitrary associations between phonetic forms and their meanings. We suggest that this can be a compelling probe into how Multimodal Large Language Models (MLLMs) interpret auditory information in human languages. We investigate MLLMs' performance on phonetic iconicity across textual (orthographic and IPA) and auditory forms of inputs with up to 25 semantic dimensions (e.g., sharp vs. round), observing models' layer-wise information processing by measuring phoneme-level attention fraction scores. To this end, we present LEX-ICON, an extensive mimetic word dataset consisting of 8,052 words from four natural languages (English, French, Japanese, and Korean) and 2,930 systematically constructed pseudo-words, annotated with semantic features applied across both text and audio modalities. Our key findings demonstrate (1) MLLMs' phonetic intuitions that align with existing linguistic research across multiple semantic dimensions and (2) phonosemantic attention patterns that highlight models' focus on iconic phonemes. These results bridge domains of artificial intelligence and cognitive linguistics, providing the first large-scale, quantitative analyses of phonetic iconicity in terms of MLLMs' interpretability.
>
---
#### [replaced 065] T^2Agent A Tool-augmented Multimodal Misinformation Detection Agent with Monte Carlo Tree Search
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2505.19768v2](https://arxiv.org/pdf/2505.19768v2)**

> **作者:** Xing Cui; Yueying Zou; Zekun Li; Peipei Li; Xinyuan Xu; Xuannan Liu; Huaibo Huang
>
> **备注:** accepted by AAAI 2026 (Oral)
>
> **摘要:** Real-world multimodal misinformation often arises from mixed forgery sources, requiring dynamic reasoning and adaptive verification. However, existing methods mainly rely on static pipelines and limited tool usage, limiting their ability to handle such complexity and diversity. To address this challenge, we propose \method, a novel misinformation detection agent that incorporates an extensible toolkit with Monte Carlo Tree Search (MCTS). The toolkit consists of modular tools such as web search, forgery detection, and consistency analysis. Each tool is described using standardized templates, enabling seamless integration and future expansion. To avoid inefficiency from using all tools simultaneously, a greedy search-based selector is proposed to identify a task-relevant subset. This subset then serves as the action space for MCTS to dynamically collect evidence and perform multi-source verification. To better align MCTS with the multi-source nature of misinformation detection, \method~ extends traditional MCTS with multi-source verification, which decomposes the task into coordinated subtasks targeting different forgery sources. A dual reward mechanism containing a reasoning trajectory score and a confidence score is further proposed to encourage a balance between exploration across mixed forgery sources and exploitation for more reliable evidence. We conduct ablation studies to confirm the effectiveness of the tree search mechanism and tool usage. Extensive experiments further show that \method~ consistently outperforms existing baselines on challenging mixed-source multimodal misinformation benchmarks, demonstrating its strong potential as a training-free detector.
>
---
#### [replaced 066] SciAgent: A Unified Multi-Agent System for Generalistic Scientific Reasoning
- **分类: cs.AI; cs.CL; cs.MA**

- **链接: [https://arxiv.org/pdf/2511.08151v2](https://arxiv.org/pdf/2511.08151v2)**

> **作者:** Xuchen Li; Ruitao Wu; Xuanbo Liu; Xukai Wang; Jinbo Hu; Zhixin Bai; Bohan Zeng; Hao Liang; Leheng Chen; Mingrui Chen; Haitian Zhong; Xuanlin Yang; Xu-Yao Zhang; Liu Liu; Jia Li; Kaiqi Huang; Jiahao Xu; Haitao Mi; Wentao Zhang; Bin Dong
>
> **备注:** 1. To ensure result rigor, the model outputs require further evaluation by human experts. 2. The results may affect our conclusions and methods, thus necessitating a more detailed review. 3. We anticipate subsequent revisions may be substantial, potentially involving major adjustments to the methodology. Given the uncertainty surrounding the revision process, we decide to request a withdrawal
>
> **摘要:** Recent advances in large language models have enabled AI systems to achieve expert-level performance on domain-specific scientific tasks, yet these systems remain narrow and handcrafted. We introduce SciAgent, a unified multi-agent system designed for generalistic scientific reasoning-the ability to adapt reasoning strategies across disciplines and difficulty levels. SciAgent organizes problem solving as a hierarchical process: a Coordinator Agent interprets each problem's domain and complexity, dynamically orchestrating specialized Worker Systems, each composed of interacting reasoning Sub-agents for symbolic deduction, conceptual modeling, numerical computation, and verification. These agents collaboratively assemble and refine reasoning pipelines tailored to each task. Across mathematics and physics Olympiads (IMO, IMC, IPhO, CPhO), SciAgent consistently attains or surpasses human gold-medalist performance, demonstrating both domain generality and reasoning adaptability. Additionally, SciAgent has been tested on the International Chemistry Olympiad (IChO) and selected problems from the Humanity's Last Exam (HLE) benchmark, further confirming the system's ability to generalize across diverse scientific domains. This work establishes SciAgent as a concrete step toward generalistic scientific intelligence-AI systems capable of coherent, cross-disciplinary reasoning at expert levels.
>
---
#### [replaced 067] AMaPO: Adaptive Margin-attached Preference Optimization for Language Model Alignment
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2511.09385v2](https://arxiv.org/pdf/2511.09385v2)**

> **作者:** Ruibo Deng; Duanyu Feng; Wenqiang Lei
>
> **备注:** AAAI 2026 AIA oral, our code is available at https://github.com/Shiroha-Offical/AMaPO
>
> **摘要:** Offline preference optimization offers a simpler and more stable alternative to RLHF for aligning language models. However, their effectiveness is critically dependent on ranking accuracy, a metric where further gains are highly impactful. This limitation arises from a fundamental problem that we identify and formalize as the Overfitting-Underfitting Dilemma: current margin designs cause models to apply excessive, wasteful gradients to correctly ranked samples (overfitting) while providing insufficient corrective signals for misranked ones (underfitting). To resolve this dilemma, we propose Adaptive Margin-attached Preference Optimization (AMaPO), a simple yet principled algorithm. AMaPO employs an instance-wise adaptive margin, refined by Z-normalization and exponential scaling, which dynamically reallocates learning effort by amplifying gradients for misranked samples and suppressing them for correct ones. Extensive experiments on widely used benchmarks demonstrate that AMaPO not only achieves better ranking accuracy and superior downstream alignment performance, but targeted analysis also confirms that it successfully mitigates the core overfitting and underfitting issues.
>
---
#### [replaced 068] On the Limitations of Language Targeted Pruning: Investigating the Calibration Language Impact in Multilingual LLM Pruning
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2408.14398v4](https://arxiv.org/pdf/2408.14398v4)**

> **作者:** Simon Kurz; Jian-Jia Chen; Lucie Flek; Zhixue Zhao
>
> **备注:** Accepted for publication in TACL
>
> **摘要:** Recent advances in large language model (LLM) pruning have shown state-of-the-art (SotA) compression results in post-training and retraining-free settings while maintaining high predictive performance. However, previous research mainly considered calibrating based on English text, despite the multilingual nature of modern LLMs and their frequent use in non-English languages. This analysis paper conducts an in-depth investigation of the performance and internal representation changes associated with pruning multilingual language models for monolingual applications. We present the first comprehensive empirical study, comparing different calibration languages for pruning multilingual models across diverse languages, tasks, models, and SotA pruning techniques. We further analyze the latent subspaces, pruning masks, and individual neurons within pruned models. Our results reveal that while calibration on the target language effectively retains perplexity and yields high signal-to-noise ratios, it does not consistently improve downstream task performance. Further analysis of internal representations at three different levels highlights broader limitations of current pruning approaches: While they effectively preserve dominant information like language-specific features, this is insufficient to counteract the loss of nuanced, language-agnostic features that are crucial for knowledge retention and reasoning.
>
---
#### [replaced 069] From Euler to AI: Unifying Formulas for Mathematical Constants
- **分类: math.HO; cs.AI; cs.CL; math.NT**

- **链接: [https://arxiv.org/pdf/2502.17533v3](https://arxiv.org/pdf/2502.17533v3)**

> **作者:** Tomer Raz; Michael Shalyt; Elyasheev Leibtag; Rotem Kalisch; Shachar Weinbaum; Yaron Hadad; Ido Kaminer
>
> **备注:** Final version for NeurIPS2025
>
> **摘要:** The constant $π$ has fascinated scholars throughout the centuries, inspiring numerous formulas for its evaluation, such as infinite sums and continued fractions. Despite their individual significance, many of the underlying connections among formulas remain unknown, missing unifying theories that could unveil deeper understanding. The absence of a unifying theory reflects a broader challenge across math and science: knowledge is typically accumulated through isolated discoveries, while deeper connections often remain hidden. In this work, we present an automated framework for the unification of mathematical formulas. Our system combines Large Language Models (LLMs) for systematic formula harvesting, an LLM-code feedback loop for validation, and a novel symbolic algorithm for clustering and eventual unification. We demonstrate this methodology on the hallmark case of $π$, an ideal testing ground for symbolic unification. Applying this approach to 455,050 arXiv papers, we validate 385 distinct formulas for $π$ and prove relations between 360 (94%) of them, of which 166 (43%) can be derived from a single mathematical object - linking canonical formulas by Euler, Gauss, Brouncker, and newer ones from algorithmic discoveries by the Ramanujan Machine. Our method generalizes to other constants, including $e$, $ζ(3)$, and Catalan's constant, demonstrating the potential of AI-assisted mathematics to uncover hidden structures and unify knowledge across domains.
>
---
#### [replaced 070] Exposing the Cracks: Vulnerabilities of Retrieval-Augmented LLM-based Machine Translation
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2510.00829v2](https://arxiv.org/pdf/2510.00829v2)**

> **作者:** Yanming Sun; Runzhe Zhan; Chi Seng Cheang; Han Wu; Xuebo Liu; Yuyao Niu; Fengying Ye; Kaixin Lan; Lidia S. Chao; Derek F. Wong
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** \textbf{RE}trieval-\textbf{A}ugmented \textbf{L}LM-based \textbf{M}achine \textbf{T}ranslation (REAL-MT) shows promise for knowledge-intensive tasks like idiomatic translation, but its reliability under noisy retrieval contexts remains poorly understood despite this being a common challenge in real-world deployment. To address this gap, we propose a noise synthesis framework and new metrics to evaluate the robustness of REAL-MT systematically. Using this framework, we instantiate REAL-MT with Qwen-series models, including standard LLMs and large reasoning models (LRMs) with enhanced reasoning, and evaluate their performance on idiomatic translation across high-, medium-, and low-resource language pairs under synthesized noise. Our results show that low-resource language pairs, which rely more heavily on retrieved context, degrade more severely under noise than high-resource ones and often produce nonsensical translations. Although LRMs possess enhanced reasoning capabilities, they show no improvement in error correction and are even more susceptible to noise, tending to rationalize incorrect contexts. We find that this stems from an attention shift away from the source idiom to noisy content, while confidence increases despite declining accuracy, indicating poor calibration. To mitigate these issues, we investigate training-free and fine-tuning strategies, which improve robustness at the cost of performance in clean contexts, revealing a fundamental trade-off. Our findings highlight the limitations of current approaches, underscoring the need for self-verifying integration mechanisms.
>
---
#### [replaced 071] Beyond Magic Words: Sharpness-Aware Prompt Evolving for Robust Large Language Models with TARE
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2509.24130v2](https://arxiv.org/pdf/2509.24130v2)**

> **作者:** Guancheng Wan; Lucheng Fu; Haoxin Liu; Yiqiao Jin; Hui Yi Leong; Eric Hanchen Jiang; Hejia Geng; Jinhe Bi; Yunpu Ma; Xiangru Tang; B. Aditya Prakash; Yizhou Sun; Wei Wang
>
> **备注:** We have identified a critical methodological error in Section 3 of the manuscript, which invalidates the main results; therefore, we request withdrawal for further revision
>
> **摘要:** The performance of Large Language Models (LLMs) hinges on carefully engineered prompts. However, prevailing prompt optimization methods, ranging from heuristic edits and reinforcement learning to evolutionary search, primarily target point-wise accuracy. They seldom enforce paraphrase invariance or searching stability, and therefore cannot remedy this brittleness in practice. Automated prompt search remains brittle: small, semantically preserving paraphrases often cause large performance swings. We identify this brittleness as the textual sharpness of the prompt landscape. In this work, we provide the first formal treatment of textual sharpness in the discrete, semantic space of prompts, together with an operational robustness criterion over a semantic neighborhood; the design is black-box or API-only, requiring no gradients to update the model's parameters. Then we introduce TARE (Textual Sharpness-Aware Evolving), a derivative-free framework that alternates between an inner, sampling-based adversarial search that stresses a prompt with hard paraphrases and an outer, robust selection that prefers candidates whose neighborhoods remain strong. We further propose ATARE, which learns anisotropic weights to shape the semantic neighborhood and adapts its radius over time to balance exploration and fidelity. Diverse tasks evaluate our methods, whose design for minimizing textual sharpness gap leads to prompts that preserve accuracy under paraphrasing, outperforming accuracy-only prompt search while remaining computationally practical.
>
---
#### [replaced 072] Multi-Personality Generation of LLMs at Decoding-time
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.01891v2](https://arxiv.org/pdf/2511.01891v2)**

> **作者:** Rongxin Chen; Yunfan Li; Yige Yuan; Bingbing Xu; Huawei Shen
>
> **备注:** Accepted by WSDM 2026
>
> **摘要:** Multi-personality generation for LLMs, enabling simultaneous embodiment of multiple personalization attributes, is a fundamental challenge. Existing retraining-based approaches are costly and poorly scalable, while decoding-time methods often rely on external models or heuristics, limiting flexibility and robustness. In this paper, we propose a novel Multi-Personality Generation (MPG) framework under the decoding-time combination paradigm. It flexibly controls multi-personality without relying on scarce multi-dimensional models or extra training, leveraging implicit density ratios in single-dimensional models as a "free lunch" to reformulate the task as sampling from a target strategy aggregating these ratios. To implement MPG efficiently, we design Speculative Chunk-level based Rejection sampling (SCR), which generates responses in chunks and parallelly validates them via estimated thresholds within a sliding window. This significantly reduces computational overhead while maintaining high-quality generation. Experiments on MBTI personality and Role-Playing demonstrate the effectiveness of MPG, showing improvements up to 16%-18%. Code and data are available at https://github.com/Libra117/MPG .
>
---
#### [replaced 073] Self-Organizing Language
- **分类: cs.CL; cs.AI; cs.LG; q-bio.NC**

- **链接: [https://arxiv.org/pdf/2506.23293v2](https://arxiv.org/pdf/2506.23293v2)**

> **作者:** P. Myles Eugenio; Anthony Beavers
>
> **备注:** 27 pages, 14 figures; Name changed from "Objective-Free Local Learning and Emergent Language Structure in Thinking Machines"
>
> **摘要:** We introduce a novel paradigm of emergent local memory. It is a continuous-learning completely-parallel content-addressable memory encoding global order. It demonstrates how local constraints on uncoordinated learning can produce topologically protected memories realizing emergent symbolic order. It is therefore a neuro-symbolic bridge. It further has the ability to produce human language without data, by exploiting its own self-organizing dynamics. It teaches us that words arise as a side-effect of emergent symbolic order, and that human language patterns at all structural levels reflect a universal mechanism of word formation (which is subregular). This work answers essential questions about the existence \& origin of all the human language data.
>
---
#### [replaced 074] Silenced Biases: The Dark Side LLMs Learned to Refuse
- **分类: cs.CL; stat.ML**

- **链接: [https://arxiv.org/pdf/2511.03369v2](https://arxiv.org/pdf/2511.03369v2)**

> **作者:** Rom Himelstein; Amit LeVi; Brit Youngmann; Yaniv Nemcovsky; Avi Mendelson
>
> **备注:** Accepted to The 40th Annual AAAI Conference on Artificial Intelligence - AI Alignment Track (Oral)
>
> **摘要:** Safety-aligned large language models (LLMs) are becoming increasingly widespread, especially in sensitive applications where fairness is essential and biased outputs can cause significant harm. However, evaluating the fairness of models is a complex challenge, and approaches that do so typically utilize standard question-answer (QA) styled schemes. Such methods often overlook deeper issues by interpreting the model's refusal responses as positive fairness measurements, which creates a false sense of fairness. In this work, we introduce the concept of silenced biases, which are unfair preferences encoded within models' latent space and are effectively concealed by safety-alignment. Previous approaches that considered similar indirect biases often relied on prompt manipulation or handcrafted implicit queries, which present limited scalability and risk contaminating the evaluation process with additional biases. We propose the Silenced Bias Benchmark (SBB), which aims to uncover these biases by employing activation steering to reduce model refusals during QA. SBB supports easy expansion to new demographic groups and subjects, presenting a fairness evaluation framework that encourages the future development of fair models and tools beyond the masking effects of alignment training. We demonstrate our approach over multiple LLMs, where our findings expose an alarming distinction between models' direct responses and their underlying fairness issues.
>
---
#### [replaced 075] Unintended Misalignment from Agentic Fine-Tuning: Risks and Mitigation
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2508.14031v2](https://arxiv.org/pdf/2508.14031v2)**

> **作者:** Dongyoon Hahm; Taywon Min; Woogyeol Jin; Kimin Lee
>
> **备注:** Accepted at AAAI 2026 AI Alignment Track, Source code: https://github.com/HahmDY/agentic-ft-safety
>
> **摘要:** Beyond simple text generation, Large Language Models (LLMs) have evolved into agentic systems capable of planning and interacting with external tools to solve complex tasks. This evolution involves fine-tuning LLMs on agent-specific tasks to enhance their proficiency. However, safety concerns are frequently overlooked during this fine-tuning process. In this work, we show that aligned LLMs can become unintentionally misaligned, leading to a higher likelihood of executing harmful tasks and a reduced tendency to refuse them when fine-tuned to execute agentic tasks. To address these safety challenges, we propose Prefix INjection Guard (PING), a simple yet effective method that prepends automatically generated natural language prefixes to agent responses, guiding them to refuse harmful requests while preserving performance on benign tasks. Specifically, we introduce an iterative approach that alternates between (1) generating candidate prefixes and (2) selecting those that optimize both task performance and refusal behavior. Experimental results demonstrate that PING significantly enhances the safety of fine-tuned LLM agents without sacrificing their effectiveness. PING consistently outperforms existing prompting approaches across diverse benchmarks in both web navigation and code generation tasks. Our analysis of internal hidden states via linear probes reveals that prefix tokens are crucial for behavior modification, explaining the performance gains. WARNING: This paper contains contents that are unethical or offensive in nature.
>
---
#### [replaced 076] Fooling the LVLM Judges: Visual Biases in LVLM-Based Evaluation
- **分类: cs.CL; cs.CV**

- **链接: [https://arxiv.org/pdf/2505.15249v2](https://arxiv.org/pdf/2505.15249v2)**

> **作者:** Yerin Hwang; Dongryeol Lee; Kyungmin Min; Taegwan Kang; Yong-il Kim; Kyomin Jung
>
> **备注:** EMNLP 2025 Main (21pgs, 12 Tables, 9 Figures)
>
> **摘要:** Recently, large vision-language models (LVLMs) have emerged as the preferred tools for judging text-image alignment, yet their robustness along the visual modality remains underexplored. This work is the first study to address a key research question: Can adversarial visual manipulations systematically fool LVLM judges into assigning unfairly inflated scores? We define potential image induced biases within the context of T2I evaluation and examine how these biases affect the evaluations of LVLM judges. Moreover, we introduce a novel, fine-grained, multi-domain meta-evaluation benchmark named FRAME, which is deliberately constructed to exhibit diverse score distributions. By introducing the defined biases into the benchmark, we reveal that all tested LVLM judges exhibit vulnerability across all domains, consistently inflating scores for manipulated images. Further analysis reveals that combining multiple biases amplifies their effects, and pairwise evaluations are similarly susceptible. Moreover, we observe that visual biases persist under prompt-based mitigation strategies, highlighting the vulnerability of current LVLM evaluation systems and underscoring the urgent need for more robust LVLM judges.
>
---
#### [replaced 077] SafeKey: Amplifying Aha-Moment Insights for Safety Reasoning
- **分类: cs.AI; cs.CL; cs.CR**

- **链接: [https://arxiv.org/pdf/2505.16186v2](https://arxiv.org/pdf/2505.16186v2)**

> **作者:** Kaiwen Zhou; Xuandong Zhao; Gaowen Liu; Jayanth Srinivasa; Aosong Feng; Dawn Song; Xin Eric Wang
>
> **摘要:** Large Reasoning Models (LRMs) introduce a new generation paradigm of explicitly reasoning before answering, leading to remarkable improvements in complex tasks. However, they pose great safety risks against harmful queries and adversarial attacks. While recent mainstream safety efforts on LRMs, supervised fine-tuning (SFT), improve safety performance, we find that SFT-aligned models struggle to generalize to unseen jailbreak prompts. After thorough investigation of LRMs' generation, we identify a safety aha moment that can activate safety reasoning and lead to a safe response. This aha moment typically appears in the `key sentence', which follows models' query understanding process and can indicate whether the model will proceed safely. Based on these insights, we propose SafeKey, including two complementary objectives to better activate the safety aha moment in the key sentence: (1) a Dual-Path Safety Head to enhance the safety signal in the model's internal representations before the key sentence, and (2) a Query-Mask Modeling objective to improve the models' attention on its query understanding, which has important safety hints. Experiments across multiple safety benchmarks demonstrate that our methods significantly improve safety generalization to a wide range of jailbreak attacks and out-of-distribution harmful prompts, lowering the average harmfulness rate by 9.6\%, while maintaining general abilities. Our analysis reveals how SafeKey enhances safety by reshaping internal attention and improving the quality of hidden representations.
>
---
#### [replaced 078] Jailbreaking LLMs via Semantically Relevant Nested Scenarios with Targeted Toxic Knowledge
- **分类: cs.CR; cs.CL**

- **链接: [https://arxiv.org/pdf/2510.01223v2](https://arxiv.org/pdf/2510.01223v2)**

> **作者:** Ning Xu; Bo Gao; Hui Dou
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities in various tasks. However, they remain exposed to jailbreak attacks, eliciting harmful responses. The nested scenario strategy has been increasingly adopted across various methods, demonstrating immense potential. Nevertheless, these methods are easily detectable due to their prominent malicious intentions. In this work, we are the first to find and systematically verify that LLMs' alignment defenses are not sensitive to nested scenarios, where these scenarios are highly semantically relevant to the queries and incorporate targeted toxic knowledge. This is a crucial yet insufficiently explored direction. Based on this, we propose RTS-Attack (Semantically Relevant Nested Scenarios with Targeted Toxic Knowledge), an adaptive and automated framework to examine LLMs' alignment. By building scenarios highly relevant to the queries and integrating targeted toxic knowledge, RTS-Attack bypasses the alignment defenses of LLMs. Moreover, the jailbreak prompts generated by RTS-Attack are free from harmful queries, leading to outstanding concealment. Extensive experiments demonstrate that RTS-Attack exhibits superior performance in both efficiency and universality compared to the baselines across diverse advanced LLMs, including GPT-4o, Llama3-70b, and Gemini-pro. Our complete code is available at https://github.com/nercode/Work. WARNING: THIS PAPER CONTAINS POTENTIALLY HARMFUL CONTENT.
>
---
#### [replaced 079] Is deeper always better? Replacing linear mappings with deep learning networks in the Discriminative Lexicon Model
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2410.04259v2](https://arxiv.org/pdf/2410.04259v2)**

> **作者:** Maria Heitmeier; Valeria Schmidt; Hendrik P. A. Lensch; R. Harald Baayen
>
> **备注:** 19 pages, 6 figures; includes a few numeric changes to results due to a fixed bug, published version
>
> **摘要:** Recently, deep learning models have increasingly been used in cognitive modelling of language. This study asks whether deep learning can help us to better understand the learning problem that needs to be solved by speakers, above and beyond linear methods. We utilise the Discriminative Lexicon Model introduced by Baayen and colleagues, which models comprehension and production with mappings between numeric form and meaning vectors. While so far, these mappings have been linear (Linear Discriminative Learning, LDL), in the present study we replace them with deep dense neural networks (Deep Discriminative Learning, DDL). We find that DDL affords more accurate mappings for large and diverse datasets from English and Dutch, but not necessarily for Estonian and Taiwan Mandarin. DDL outperforms LDL in particular for words with pseudo-morphological structure such as chol+er. Applied to average reaction times, we find that DDL is outperformed by frequency-informed linear mappings (FIL). However, DDL trained in a frequency-informed way ('frequency-informed' deep learning, FIDDL) substantially outperforms FIL. Finally, while linear mappings can very effectively be updated from trial-to-trial to model incremental lexical learning, deep mappings cannot do so as effectively. At present, both linear and deep mappings are informative for understanding language.
>
---
#### [replaced 080] Building a Macedonian Recipe Dataset: Collection, Parsing, and Comparative Analysis
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2510.14128v2](https://arxiv.org/pdf/2510.14128v2)**

> **作者:** Darko Sasanski; Dimitar Peshevski; Riste Stojanov; Dimitar Trajanov
>
> **摘要:** Computational gastronomy increasingly relies on diverse, high-quality recipe datasets to capture regional culinary traditions. Although there are large-scale collections for major languages, Macedonian recipes remain under-represented in digital research. In this work, we present the first systematic effort to construct a Macedonian recipe dataset through web scraping and structured parsing. We address challenges in processing heterogeneous ingredient descriptions, including unit, quantity, and descriptor normalization. An exploratory analysis of ingredient frequency and co-occurrence patterns, using measures such as Pointwise Mutual Information and Lift score, highlights distinctive ingredient combinations that characterize Macedonian cuisine. The resulting dataset contributes a new resource for studying food culture in underrepresented languages and offers insights into the unique patterns of Macedonian culinary tradition.
>
---
#### [replaced 081] Simultaneous Machine Translation with Large Language Models
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2309.06706v3](https://arxiv.org/pdf/2309.06706v3)**

> **作者:** Minghan Wang; Jinming Zhao; Thuy-Trang Vu; Fatemeh Shiri; Ehsan Shareghi; Gholamreza Haffari
>
> **备注:** Accepted to ALTA 2024
>
> **摘要:** Real-world simultaneous machine translation (SimulMT) systems face more challenges than just the quality-latency trade-off. They also need to address issues related to robustness with noisy input, processing long contexts, and flexibility for knowledge injection. These challenges demand models with strong language understanding and generation capabilities which may not often equipped by dedicated MT models. In this paper, we investigate the possibility of applying Large Language Models (LLM) to SimulMT tasks by using existing incremental-decoding methods with a newly proposed RALCP algorithm for latency reduction. We conducted experiments using the \texttt{Llama2-7b-chat} model on nine different languages from the MUST-C dataset. The results show that LLM outperforms dedicated MT models in terms of BLEU and LAAL metrics. Further analysis indicates that LLM has advantages in terms of tuning efficiency and robustness. However, it is important to note that the computational cost of LLM remains a significant obstacle to its application in SimulMT.
>
---
#### [replaced 082] QuanTaxo: A Quantum Approach to Self-Supervised Taxonomy Expansion
- **分类: cs.SI; cs.CL**

- **链接: [https://arxiv.org/pdf/2501.14011v3](https://arxiv.org/pdf/2501.14011v3)**

> **作者:** Sahil Mishra; Avi Patni; Niladri Chatterjee; Tanmoy Chakraborty
>
> **摘要:** A taxonomy is a hierarchical graph containing knowledge to provide valuable insights for various web applications. However, the manual construction of taxonomies requires significant human effort. As web content continues to expand at an unprecedented pace, existing taxonomies risk becoming outdated, struggling to incorporate new and emerging information effectively. As a consequence, there is a growing need for dynamic taxonomy expansion to keep them relevant and up-to-date. Existing taxonomy expansion methods often rely on classical word embeddings to represent entities. However, these embeddings fall short of capturing hierarchical polysemy, where an entity's meaning can vary based on its position in the hierarchy and its surrounding context. To address this challenge, we introduce QuanTaxo, a quantum-inspired framework for taxonomy expansion that encodes entities in a Hilbert space and models interference effects between them, yielding richer, context-sensitive representations. Comprehensive experiments on five real-world benchmark datasets show that QuanTaxo significantly outperforms classical embedding models, achieving substantial improvements of 12.3% in accuracy, 11.2% in Mean Reciprocal Rank (MRR), and 6.9% in Wu & Palmer (Wu&P) metrics across nine classical embedding-based baselines.
>
---
#### [replaced 083] LLM-Driven Robots Risk Enacting Discrimination, Violence, and Unlawful Actions
- **分类: cs.RO; cs.AI; cs.CL; cs.CY**

- **链接: [https://arxiv.org/pdf/2406.08824v2](https://arxiv.org/pdf/2406.08824v2)**

> **作者:** Andrew Hundt; Rumaisa Azeem; Masoumeh Mansouri; Martim Brandão
>
> **备注:** Published in International Journal of Social Robotics (2025). 49 pages (65 with references and appendix), 27 Figures, 8 Tables. Andrew Hundt and Rumaisa Azeem are equal contribution co-first authors. The positions of the two co-first authors were swapped from arxiv version 1 with the written consent of all four authors. The Version of Record is available via DOI: 10.1007/s12369-025-01301-x
>
> **摘要:** Members of the Human-Robot Interaction (HRI) and Machine Learning (ML) communities have proposed Large Language Models (LLMs) as a promising resource for robotics tasks such as natural language interaction, household and workplace tasks, approximating 'common sense reasoning', and modeling humans. However, recent research has raised concerns about the potential for LLMs to produce discriminatory outcomes and unsafe behaviors in real-world robot experiments and applications. To assess whether such concerns are well placed in the context of HRI, we evaluate several highly-rated LLMs on discrimination and safety criteria. Our evaluation reveals that LLMs are currently unsafe for people across a diverse range of protected identity characteristics, including, but not limited to, race, gender, disability status, nationality, religion, and their intersections. Concretely, we show that LLMs produce directly discriminatory outcomes- e.g., 'gypsy' and 'mute' people are labeled untrustworthy, but not 'european' or 'able-bodied' people. We find various such examples of direct discrimination on HRI tasks such as facial expression, proxemics, security, rescue, and task assignment. Furthermore, we test models in settings with unconstrained natural language (open vocabulary) inputs, and find they fail to act safely, generating responses that accept dangerous, violent, or unlawful instructions-such as incident-causing misstatements, taking people's mobility aids, and sexual predation. Our results underscore the urgent need for systematic, routine, and comprehensive risk assessments and assurances to improve outcomes and ensure LLMs only operate on robots when it is safe, effective, and just to do so. We provide code to reproduce our experiments at https://github.com/rumaisa-azeem/llm-robots-discrimination-safety .
>
---
#### [replaced 084] SceneJailEval: A Scenario-Adaptive Multi-Dimensional Framework for Jailbreak Evaluation
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2508.06194v2](https://arxiv.org/pdf/2508.06194v2)**

> **作者:** Lai Jiang; Yuekang Li; Xiaohan Zhang; Youtao Ding; Li Pan
>
> **备注:** This paper has been accepted by AAAI 2026 as a poster
>
> **摘要:** Accurate jailbreak evaluation is critical for LLM red team testing and jailbreak research. Mainstream methods rely on binary classification (string matching, toxic text classifiers, and LLM-based methods), outputting only "yes/no" labels without quantifying harm severity. Emerged multi-dimensional frameworks (e.g., Security Violation, Relative Truthfulness and Informativeness) use unified evaluation standards across scenarios, leading to scenario-specific mismatches (e.g., "Relative Truthfulness" is irrelevant to "hate speech"), undermining evaluation accuracy. To address these, we propose SceneJailEval, with key contributions: (1) A pioneering scenario-adaptive multi-dimensional framework for jailbreak evaluation, overcoming the critical "one-size-fits-all" limitation of existing multi-dimensional methods, and boasting robust extensibility to seamlessly adapt to customized or emerging scenarios. (2) A novel 14-scenario dataset featuring rich jailbreak variants and regional cases, addressing the long-standing gap in high-quality, comprehensive benchmarks for scenario-adaptive evaluation. (3) SceneJailEval delivers state-of-the-art performance with an F1 score of 0.917 on our full-scenario dataset (+6% over SOTA) and 0.995 on JBB (+3% over SOTA), breaking through the accuracy bottleneck of existing evaluation methods in heterogeneous scenarios and solidifying its superiority.
>
---
#### [replaced 085] SWE-Bench Pro: Can AI Agents Solve Long-Horizon Software Engineering Tasks?
- **分类: cs.SE; cs.CL**

- **链接: [https://arxiv.org/pdf/2509.16941v2](https://arxiv.org/pdf/2509.16941v2)**

> **作者:** Xiang Deng; Jeff Da; Edwin Pan; Yannis Yiming He; Charles Ide; Kanak Garg; Niklas Lauffer; Andrew Park; Nitin Pasari; Chetan Rane; Karmini Sampath; Maya Krishnan; Srivatsa Kundurthy; Sean Hendryx; Zifan Wang; Vijay Bharadwaj; Jeff Holm; Raja Aluri; Chen Bo Calvin Zhang; Noah Jacobson; Bing Liu; Brad Kenstler
>
> **摘要:** We introduce SWE-Bench Pro, a substantially more challenging benchmark that builds upon the best practices of SWE-BENCH [25], but is explicitly designed to capture realistic, complex, enterprise-level problems beyond the scope of SWE-BENCH. SWE-BENCH PRO contains 1,865 problems sourced from a diverse set of 41 actively maintained repositories spanning business applications, B2B services, and developer tools. The benchmark is partitioned into a public set with open access to problems sourced from 11 repositories, a held-out set of 12 repositories and a commercial set of 18 proprietary repositories where we have formal partnership agreements with early-stage startups. Problems in the held-out and the commercial set are not publicly accessible, but we release results on the commercial set. Our benchmark features long-horizon tasks that may require hours to days for a professional software engineer to complete, often involving patches across multiple files and substantial code modifications. All tasks are human-verified and augmented with sufficient context to ensure resolvability. To better understand these limitations, we cluster the failure modes observed in the collected agent trajectories for a clearer characterization of the error patterns exhibited by current models. Overall, SWE-BENCH PRO provides a contamination-resistant testbed that more faithfully captures the complexity and diversity of real-world software development, advancing the pursuit of truly autonomous software engineering agents at a professional level.
>
---
#### [replaced 086] InfiMed-ORBIT: Aligning LLMs on Open-Ended Complex Tasks via Rubric-Based Incremental Training
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.15859v2](https://arxiv.org/pdf/2510.15859v2)**

> **作者:** Pengkai Wang; Qi Zuo; Pengwei Liu; Zhijie Sang; Congkai Xie; Hongxia Yang
>
> **摘要:** Reinforcement learning has powered many of the recent breakthroughs in large language models, especially for tasks where rewards can be computed automatically, such as code generation. However, these methods deteriorate in open-ended domains like medical consultation, where feedback is inherently ambiguous, highly context-dependent, and cannot be reduced to a reliable scalar signal. In such settings, RL must either rely on supervision-intensive reward models that often fail to generalize, or it falls into pathological behaviors such as reward hacking - an especially troubling risk for high-stakes medical dialogue. To address these limitations, we introduce ORBIT, an open-ended rubric-based incremental training framework for high-stakes medical dialogue. ORBIT integrates synthetic dialogue generation with dynamically constructed rubrics that serve as adaptive guides for incremental RL. Instead of relying on external medical knowledge bases or handcrafted rule sets, ORBIT uses rubric-driven feedback to steer the learning process. Its judge component can be instantiated with general-purpose instruction-following LLMs, removing the need for any task-specific fine-tuning. Applied to the Qwen3-4B-Instruct model, ORBIT raises the HealthBench-Hard score from 7.0 to 27.5 using only 2k training samples, achieving SOTA performance for models at this scale. With larger rubric datasets, ORBIT-trained models further compete with the strongest open-source baselines on HealthBench-Hard. Our analysis shows that rubric-guided RL consistently improves consultation quality across diverse medical scenarios. We also apply such rubric generation and training pipeline to InfoBench, where ORBIT enhances instruction-following performance, highlighting the generality of rubric-based feedback.
>
---
#### [replaced 087] GRAM-R$^2$: Self-Training Generative Foundation Reward Models for Reward Reasoning
- **分类: cs.CL; cs.LG**

- **链接: [https://arxiv.org/pdf/2509.02492v3](https://arxiv.org/pdf/2509.02492v3)**

> **作者:** Chenglong Wang; Yongyu Mu; Hang Zhou; Yifu Huo; Ziming Zhu; Jiali Zeng; Murun Yang; Bei Li; Xiaoyang Hao; Chunliang Zhang; Fandong Meng; Jingbo Zhu; Tong Xiao
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Significant progress in reward modeling over recent years has been driven by a paradigm shift from task-specific designs towards generalist reward models. Despite this trend, developing effective reward models remains a fundamental challenge: the heavy reliance on large-scale labeled preference data. Pre-training on abundant unlabeled data offers a promising direction, but existing approaches fall short of instilling explicit reasoning into reward models. To bridge this gap, we propose a self-training approach that leverages unlabeled data to elicit reward reasoning in reward models. Based on this approach, we develop GRAM-R$^2$, a generative reward model trained to produce not only preference labels but also accompanying reward rationales. GRAM-R$^2$ can serve as a foundation model for reward reasoning and can be applied to a wide range of tasks with minimal or no additional fine-tuning. It can support downstream applications such as response ranking and task-specific reward tuning. Experiments on response ranking, task adaptation, and reinforcement learning from human feedback demonstrate that GRAM-R$^2$ consistently delivers strong performance, outperforming several strong discriminative and generative baselines.
>
---
#### [replaced 088] Uncovering Factor Level Preferences to Improve Human-Model Alignment
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2410.06965v3](https://arxiv.org/pdf/2410.06965v3)**

> **作者:** Juhyun Oh; Eunsu Kim; Jiseon Kim; Wenda Xu; Inha Cha; William Yang Wang; Alice Oh
>
> **摘要:** Large language models (LLMs) often exhibit tendencies that diverge from human preferences, such as favoring certain writing styles or producing overly verbose outputs. While crucial for improvement, identifying the factors driving these misalignments remains challenging due to existing evaluation methods' reliance on coarse-grained comparisons and lack of explainability. To address this, we introduce PROFILE, an automated framework to uncover and measure factor-level preference alignment of humans and LLMs. Using PROFILE, we analyze preference alignment across three key tasks: summarization, instruction-following, and document-based QA. We find a significant discrepancy: while LLMs show poor factor-level alignment with human preferences when generating texts, they demonstrate strong alignment in discrimination tasks. We demonstrate how leveraging the identified generation-discrimination gap can be used to improve LLM alignment through multiple approaches, including fine-tuning with self-guidance. Our work highlights the value of factor-level analysis for identifying hidden misalignments and provides a practical framework for improving LLM-human preference alignment.
>
---
#### [replaced 089] Diagnose, Localize, Align: A Full-Stack Framework for Reliable LLM Multi-Agent Systems under Instruction Conflicts
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2509.23188v2](https://arxiv.org/pdf/2509.23188v2)**

> **作者:** Guancheng Wan; Leixin Sun; Longxu Dou; Zitong Shi; Fang Wu; Eric Hanchen Jiang; Wenke Huang; Guibin Zhang; Hejia Geng; Xiangru Tang; Zhenfei Yin; Yizhou Sun; Wei Wang
>
> **备注:** Upon further review, we realized that the version submitted to arXiv was not the final draft and omits crucial results and discussion. To avoid confusion and ensure the integrity of the record, we request withdrawal and will resubmit once the complete work is ready
>
> **摘要:** Large Language Model (LLM)-powered multi-agent systems (MAS) have rapidly advanced collaborative reasoning, tool use, and role-specialized coordination in complex tasks. However, reliability-critical deployment remains hindered by a systemic failure mode: hierarchical compliance under instruction conflicts (system-user, peer-peer), where agents misprioritize system-level rules in the presence of competing demands. Moreover, widely used macro-level metrics (e.g., pass@k) obscure these micro-level violations and offer little actionable guidance for remedy. In this work, we present a full-stack, three-stage framework: (1) Diagnose - Contextualized Role Adherence Score (CRAS), a query-wise, context-aware scoring metric that decomposes role adherence into four measurable dimensions; (2) Localize - attention drift analysis revealing that instruction conflicts are resolved by attention heads that are largely concentrated in middle layers; (3) Align - Surgical Alignment of Instruction Layers (SAIL), which installs LoRA only on the localized focal layers and optimizes a token-weighted DPO-style preference objective that credits tokens by their focal attentional contribution. Across standard benchmarks and MAS frameworks, our surgical approach improves instruction hierarchy compliance (e.g., +5.60% with AutoGen on MedQA) without full-model finetuning.
>
---
#### [replaced 090] Neurocognitive Modeling for Text Generation: Deep Learning Architecture for EEG Data
- **分类: cs.HC; cs.CL**

- **链接: [https://arxiv.org/pdf/2509.07202v2](https://arxiv.org/pdf/2509.07202v2)**

> **作者:** Khushiyant
>
> **备注:** 15 pages, 10 figures, 5 tables
>
> **摘要:** Text generating capabilities have undergone a substantial transformation with the introduction of large language models (LLMs). Electroencephalography (EEG)-based text production is still difficult, though, because it requires a lot of data and processing power. This paper introduces a new method that combines the use of the Gemma 2B LLM with a classifier-LLM architecture to incorporate a Recurrent Neural Network (RNN) encoder. Our approach drastically lowers the amount of data and compute power needed while achieving performance close to that of cutting-edge methods. Notably, compared to current methodologies, our methodology delivers an overall performance improvement of 10%. The suggested architecture demonstrates the possibility of effective transfer learning for EEG-based text production, remaining strong and functional even in the face of data limits. This work highlights the potential of integrating LLMs with EEG decoding to improve assistive technologies and improve independence and communication for those with severe motor limitations. Our method pushes the limits of present capabilities and opens new paths for research and application in brain-computer interfaces by efficiently using the strengths of pre-trained language models. This makes EEG-based text production more accessible and efficient.
>
---
#### [replaced 091] Self-Correction Distillation for Structured Data Question Answering
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.07998v2](https://arxiv.org/pdf/2511.07998v2)**

> **作者:** Yushan Zhu; Wen Zhang; Long Jin; Mengshu Sun; Ling Zhong; Zhiqiang Liu; Juan Li; Lei Liang; Chong Long; Chao Deng; Junlan Feng
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** Structured data question answering (QA), including table QA, Knowledge Graph (KG) QA, and temporal KG QA, is a pivotal research area. Advances in large language models (LLMs) have driven significant progress in unified structural QA frameworks like TrustUQA. However, these frameworks face challenges when applied to small-scale LLMs since small-scale LLMs are prone to errors in generating structured queries. To improve the structured data QA ability of small-scale LLMs, we propose a self-correction distillation (SCD) method. In SCD, an error prompt mechanism (EPM) is designed to detect errors and provide customized error messages during inference, and a two-stage distillation strategy is designed to transfer large-scale LLMs' query-generation and error-correction capabilities to small-scale LLM. Experiments across 5 benchmarks with 3 structured data types demonstrate that our SCD achieves the best performance and superior generalization on small-scale LLM (8B) compared to other distillation methods, and closely approaches the performance of GPT4 on some datasets. Furthermore, large-scale LLMs equipped with EPM surpass the state-of-the-art results on most datasets.
>
---
#### [replaced 092] Aligning Extraction and Generation for Robust Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2503.04789v3](https://arxiv.org/pdf/2503.04789v3)**

> **作者:** Hwanjun Song; Jeonghwan Choi; Minseok Kim
>
> **备注:** Accepted at ACM International Conference on Web Search and Data Mining (WSDM) 2026
>
> **摘要:** Retrieval-augmented generation (RAG) enhances LLMs with external knowledge, yet generation remains vulnerable to retrieval-induced noise and uncertain placement of relevant chunks, often causing hallucinations. We present Ext2Gen, an extract-then-generate framework that strengthens LLMs via joint evidence selection and answer generation, dynamically identifying query-relevant content while suppressing noise, thereby removing the need for any independent pre-generation compression module. Optimized through preference alignment with well-curated pairwise feedback, Ext2Gen produces accurate and faithful answers even under noisy or imprecise retrieval. Experiments demonstrate that it substantially enhances the robustness of the generation backbone and yields greater performance gains than methods relying on independent compression models, e.g., Recomp, CompAct, EXIT). It further benefits from improved retrieval techniques such as query rewriting, underscoring that generation-side enhancements address limitations that retrieval alone cannot overcome.
>
---
#### [replaced 093] Exploiting Synergistic Cognitive Biases to Bypass Safety in LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2507.22564v2](https://arxiv.org/pdf/2507.22564v2)**

> **作者:** Xikang Yang; Biyu Zhou; Xuehai Tang; Jizhong Han; Songlin Hu
>
> **摘要:** Large Language Models (LLMs) demonstrate impressive capabilities across a wide range of tasks, yet their safety mechanisms remain susceptible to adversarial attacks that exploit cognitive biases -- systematic deviations from rational judgment. Unlike prior jailbreaking approaches focused on prompt engineering or algorithmic manipulation, this work highlights the overlooked power of multi-bias interactions in undermining LLM safeguards. We propose CognitiveAttack, a novel red-teaming framework that systematically leverages both individual and combined cognitive biases. By integrating supervised fine-tuning and reinforcement learning, CognitiveAttack generates prompts that embed optimized bias combinations, effectively bypassing safety protocols while maintaining high attack success rates. Experimental results reveal significant vulnerabilities across 30 diverse LLMs, particularly in open-source models. CognitiveAttack achieves a substantially higher attack success rate compared to the SOTA black-box method PAP (60.1% vs. 31.6%), exposing critical limitations in current defense mechanisms. These findings highlight multi-bias interactions as a powerful yet underexplored attack vector. This work introduces a novel interdisciplinary perspective by bridging cognitive science and LLM safety, paving the way for more robust and human-aligned AI systems.
>
---
#### [replaced 094] Conversational SimulMT: Efficient Simultaneous Translation with Large Language Models
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2402.10552v4](https://arxiv.org/pdf/2402.10552v4)**

> **作者:** Minghan Wang; Thuy-Trang Vu; Yuxia Wang; Ehsan Shareghi; Gholamreza Haffari
>
> **备注:** Accepted to IWSLT 2025
>
> **摘要:** Simultaneous machine translation (SimulMT) presents a challenging trade-off between translation quality and latency. Recent studies have shown that LLMs can achieve good performance in SimulMT tasks. However, this often comes at the expense of high inference cost and latency. In this paper, we propose a conversational SimulMT framework to enhance the inference efficiency of LLM-based SimulMT through multi-turn-dialogue-based decoding. Our experiments with Llama2-7b-chat on two SimulMT benchmarks demonstrate the superiority of LLM in translation quality while achieving comparable computational latency to specialized SimulMT models.
>
---
#### [replaced 095] Bilevel MCTS for Amortized O(1) Node Selection in Classical Planning
- **分类: cs.AI; cs.CL**

- **链接: [https://arxiv.org/pdf/2508.08385v2](https://arxiv.org/pdf/2508.08385v2)**

> **作者:** Masataro Asai
>
> **备注:** Accepted in AAAI-26
>
> **摘要:** We study an efficient implementation of Multi-Armed Bandit (MAB)-based Monte-Carlo Tree Search (MCTS) for classical planning. One weakness of MCTS is that it spends a significant time deciding which node to expand next. While selecting a node from an OPEN list with $N$ nodes has $O(1)$ runtime complexity with traditional array-based priority-queues for dense integer keys, the tree-based OPEN list used by MCTS requires $O(\log N)$, which roughly corresponds to the search depth $d$. In classical planning, $d$ is arbitrarily large (e.g., $2^k-1$ in $k$-disk Tower-of-Hanoi) and the runtime for node selection is significant, unlike in game tree search, where the cost is negligible compared to the node evaluation (rollouts) because $d$ is inherently limited by the game (e.g., $d\leq 361$ in Go). To improve this bottleneck, we propose a bilevel modification to MCTS that runs a best-first search from each selected leaf node with an expansion budget proportional to $d$, which achieves amortized $O(1)$ runtime for node selection, equivalent to the traditional queue-based OPEN list. In addition, we introduce Tree Collapsing, an enhancement that reduces action selection steps and further improves the performance.
>
---
#### [replaced 096] Chain-of-Thought Driven Adversarial Scenario Extrapolation for Robust Language Models
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2505.17089v2](https://arxiv.org/pdf/2505.17089v2)**

> **作者:** Md Rafi Ur Rashid; Vishnu Asutosh Dasu; Ye Wang; Gang Tan; Shagufta Mehnaz
>
> **备注:** 19 pages, 5 figures. Accepted in AAAI 2026
>
> **摘要:** Large Language Models (LLMs) exhibit impressive capabilities, but remain susceptible to a growing spectrum of safety risks, including jailbreaks, toxic content, hallucinations, and bias. Existing defenses often address only a single threat type or resort to rigid outright rejection, sacrificing user experience and failing to generalize across diverse and novel attacks. This paper introduces Adversarial Scenario Extrapolation (ASE), a novel inference-time computation framework that leverages Chain-of-Thought (CoT) reasoning to simultaneously enhance LLM robustness and seamlessness. ASE guides the LLM through a self-generative process of contemplating potential adversarial scenarios and formulating defensive strategies before generating a response to the user query. Comprehensive evaluation on four adversarial benchmarks with four latest LLMs shows that ASE achieves near-zero jailbreak attack success rates and minimal toxicity, while slashing outright rejections to <4%. ASE outperforms six state-of-the-art defenses in robustness-seamlessness trade-offs, with 92-99% accuracy on adversarial Q&A and 4-10x lower bias scores. By transforming adversarial perception into an intrinsic cognitive process, ASE sets a new paradigm for secure and natural human-AI interaction.
>
---
#### [replaced 097] OptiHive: Ensemble Selection for LLM-Based Optimization via Statistical Modeling
- **分类: cs.AI; cs.CL**

- **链接: [https://arxiv.org/pdf/2508.02503v2](https://arxiv.org/pdf/2508.02503v2)**

> **作者:** Maxime Bouscary; Saurabh Amin
>
> **摘要:** LLM-based solvers have emerged as a promising means of automating problem modeling and solving. However, they remain unreliable and often depend on iterative repair loops that result in significant latency. We introduce OptiHive, a framework that enhances any solver-generation pipeline to produce higher-quality solvers from natural-language descriptions of optimization problems. OptiHive uses a single batched generation to produce diverse components (solvers, problem instances, and validation tests) and filters out erroneous components to ensure fully interpretable outputs. Accounting for the imperfection of the generated components, we employ a statistical model to infer their true performance, enabling principled uncertainty quantification and solver selection. On tasks ranging from traditional optimization problems to challenging variants of the Multi-Depot Vehicle Routing Problem, OptiHive significantly outperforms baselines, increasing the optimality rate from 5% to 92% on the most complex problems.
>
---
#### [replaced 098] FRAME: Feedback-Refined Agent Methodology for Enhancing Medical Research Insights
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2505.04649v3](https://arxiv.org/pdf/2505.04649v3)**

> **作者:** Chengzhang Yu; Yiming Zhang; Zhixin Liu; Zenghui Ding; Yining Sun; Zhanpeng Jin
>
> **备注:** 12 pages, 4 figures, 5 table
>
> **摘要:** The automation of scientific research through large language models (LLMs) presents significant opportunities but faces critical challenges in knowledge synthesis and quality assurance. We introduce Feedback-Refined Agent Methodology (FRAME), a novel framework that enhances medical paper generation through iterative refinement and structured feedback. Our approach comprises three key innovations: (1) A structured dataset construction method that decomposes 4,287 medical papers into essential research components through iterative refinement; (2) A tripartite architecture integrating Generator, Evaluator, and Reflector agents that progressively improve content quality through metric-driven feedback; and (3) A comprehensive evaluation framework that combines statistical metrics with human-grounded benchmarks. Experimental results demonstrate FRAME's effectiveness, achieving significant improvements over conventional approaches across multiple models (9.91% average gain with DeepSeek V3, comparable improvements with GPT-4o Mini) and evaluation dimensions. Human evaluation confirms that FRAME-generated papers achieve quality comparable to human-authored works, with particular strength in synthesizing future research directions. The results demonstrated our work could efficiently assist medical research by building a robust foundation for automated medical research paper generation while maintaining rigorous academic standards.
>
---
#### [replaced 099] Understanding and Mitigating Political Stance Cross-topic Generalization in Large Language Models
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2508.02360v2](https://arxiv.org/pdf/2508.02360v2)**

> **作者:** Jiayi Zhang; Shu Yang; Junchao Wu; Derek F. Wong; Di Wang
>
> **摘要:** Fine-tuning Large Language Models on a political topic will significantly manipulate their political stance on various issues and unintentionally affect their stance on unrelated topics. While previous studies have proposed this issue, there is still a lack of understanding regarding the internal representations of these stances and the mechanisms that lead to unintended cross-topic generalization. In this paper, we systematically explore the internal mechanisms underlying this phenomenon from a neuron-level perspective and how to mitigate the cross-topic generalization of political fine-tuning. Firstly, we propose Political Neuron Localization through Activation Contrasting (PNLAC) to identify two distinct types of political neurons: general political neurons, which govern stance across multiple political topics, and topic-specific neurons} that affect the model's political stance on individual topics. We find the existence of these political neuron types across four models and datasets through activation patching experiments. Leveraging these insights, we introduce InhibitFT, an inhibition-based fine-tuning method, effectively mitigating the cross-topic stance generalization. Experimental results demonstrate the robustness of identified neuron types across various models and datasets, and show that InhibitFT significantly reduces the cross-topic stance generalization by 20% on average, while preserving topic-specific performance. Moreover, we demonstrate that selectively inhibiting only 5% of neurons is sufficient to effectively mitigate the cross-topic stance generalization.
>
---
#### [replaced 100] Evaluating LLMs' Reasoning Over Ordered Procedural Steps
- **分类: cs.CL; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.04688v2](https://arxiv.org/pdf/2511.04688v2)**

> **作者:** Adrita Anika; Md Messal Monem Miah
>
> **备注:** Accepted to IJCNLP-AACL 2025 Findings
>
> **摘要:** Reasoning over procedural sequences, where the order of steps directly impacts outcomes, is a critical capability for large language models (LLMs). In this work, we study the task of reconstructing globally ordered sequences from shuffled procedural steps, using a curated dataset of food recipes, a domain where correct sequencing is essential for task success. We evaluate several LLMs under zero-shot and few-shot settings and present a comprehensive evaluation framework that adapts established metrics from ranking and sequence alignment. These include Kendall's Tau, Normalized Longest Common Subsequence (NLCS), and Normalized Edit Distance (NED), which capture complementary aspects of ordering quality. Our analysis shows that model performance declines with increasing sequence length, reflecting the added complexity of longer procedures. We also find that greater step displacement in the input, corresponding to more severe shuffling, leads to further degradation. These findings highlight the limitations of current LLMs in procedural reasoning, especially with longer and more disordered inputs.
>
---
#### [replaced 101] The Trilemma of Truth in Large Language Models
- **分类: cs.CL; cs.LG; stat.ML**

- **链接: [https://arxiv.org/pdf/2506.23921v4](https://arxiv.org/pdf/2506.23921v4)**

> **作者:** Germans Savcisens; Tina Eliassi-Rad
>
> **备注:** Camera-ready (non-archival) version accepted at the Mechanistic Interpretability Workshop at NeurIPS 2025. The main text is 10 pages long (plus 3 pages of references); supplementary material (58 pages) is included in the same PDF
>
> **摘要:** The public often attributes human-like qualities to large language models (LLMs) and assumes they "know" certain things. In reality, LLMs encode information retained during training as internal probabilistic knowledge. This study examines existing methods for probing the veracity of that knowledge and identifies several flawed underlying assumptions. To address these flaws, we introduce sAwMIL (Sparse-Aware Multiple-Instance Learning), a multiclass probing framework that combines multiple-instance learning with conformal prediction. sAwMIL leverages internal activations of LLMs to classify statements as true, false, or neither. We evaluate sAwMIL across 16 open-source LLMs, including default and chat-based variants, on three new curated datasets. Our results show that (1) common probing methods fail to provide a reliable and transferable veracity direction and, in some settings, perform worse than zero-shot prompting; (2) truth and falsehood are not encoded symmetrically; and (3) LLMs encode a third type of signal that is distinct from both true and false.
>
---
#### [replaced 102] REIC: RAG-Enhanced Intent Classification at Scale
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2506.00210v2](https://arxiv.org/pdf/2506.00210v2)**

> **作者:** Ziji Zhang; Michael Yang; Zhiyu Chen; Yingying Zhuang; Shu-Ting Pi; Qun Liu; Rajashekar Maragoud; Vy Nguyen; Anurag Beniwal
>
> **备注:** Accepted by EMNLP 2025 (Industry Track)
>
> **摘要:** Accurate intent classification is critical for efficient routing in customer service, ensuring customers are connected with the most suitable agents while reducing handling times and operational costs. However, as companies expand their product lines, intent classification faces scalability challenges due to the increasing number of intents and variations in taxonomy across different verticals. In this paper, we introduce REIC, a Retrieval-augmented generation Enhanced Intent Classification approach, which addresses these challenges effectively. REIC leverages retrieval-augmented generation (RAG) to dynamically incorporate relevant knowledge, enabling precise classification without the need for frequent retraining. Through extensive experiments on real-world datasets, we demonstrate that REIC outperforms traditional fine-tuning, zero-shot, and few-shot methods in large-scale customer service settings. Our results highlight its effectiveness in both in-domain and out-of-domain scenarios, demonstrating its potential for real-world deployment in adaptive and large-scale intent classification systems.
>
---
#### [replaced 103] Understanding World or Predicting Future? A Comprehensive Survey of World Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2411.14499v3](https://arxiv.org/pdf/2411.14499v3)**

> **作者:** Jingtao Ding; Yunke Zhang; Yu Shang; Yuheng Zhang; Zefang Zong; Jie Feng; Yuan Yuan; Hongyuan Su; Nian Li; Nicholas Sukiennik; Fengli Xu; Yong Li
>
> **备注:** Extended version of the original ACM CSUR paper, 49 pages, 6 figures, 8 tables
>
> **摘要:** The concept of world models has garnered significant attention due to advancements in multimodal large language models such as GPT-4 and video generation models such as Sora, which are central to the pursuit of artificial general intelligence. This survey offers a comprehensive review of the literature on world models. Generally, world models are regarded as tools for either understanding the present state of the world or predicting its future dynamics. This review presents a systematic categorization of world models, emphasizing two primary functions: (1) constructing internal representations to understand the mechanisms of the world, and (2) predicting future states to simulate and guide decision-making. Initially, we examine the current progress in these two categories. We then explore the application of world models in key domains, including generative games, autonomous driving, robotics, and social simulacra, with a focus on how each domain utilizes these aspects. Finally, we outline key challenges and provide insights into potential future research directions. We summarize the representative papers along with their code repositories in https://github.com/tsinghua-fib-lab/World-Model.
>
---
#### [replaced 104] You Don't Need Pre-built Graphs for RAG: Retrieval Augmented Generation with Adaptive Reasoning Structures
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2508.06105v2](https://arxiv.org/pdf/2508.06105v2)**

> **作者:** Shengyuan Chen; Chuang Zhou; Zheng Yuan; Qinggang Zhang; Zeyang Cui; Hao Chen; Yilin Xiao; Jiannong Cao; Xiao Huang
>
> **备注:** This work has been accepted to AAAI'26
>
> **摘要:** Large language models (LLMs) often suffer from hallucination, generating factually incorrect statements when handling questions beyond their knowledge and perception. Retrieval-augmented generation (RAG) addresses this by retrieving query-relevant contexts from knowledge bases to support LLM reasoning. Recent advances leverage pre-constructed graphs to capture the relational connections among distributed documents, showing remarkable performance in complex tasks. However, existing Graph-based RAG (GraphRAG) methods rely on a costly process to transform the corpus into a graph, introducing overwhelming token cost and update latency. Moreover, real-world queries vary in type and complexity, requiring different logic structures for accurate reasoning. The pre-built graph may not align with these required structures, resulting in ineffective knowledge retrieval. To this end, we propose a $\textbf{Logic}$-aware $\textbf{R}etrieval$-$\textbf{A}$ugmented $\textbf{G}$eneration framework ($\textbf{LogicRAG}$) that dynamically extracts reasoning structures at inference time to guide adaptive retrieval without any pre-built graph. LogicRAG begins by decomposing the input query into a set of subproblems and constructing a directed acyclic graph (DAG) to model the logical dependencies among them. To support coherent multi-step reasoning, LogicRAG then linearizes the graph using topological sort, so that subproblems can be addressed in a logically consistent order. Besides, LogicRAG applies graph pruning to reduce redundant retrieval and uses context pruning to filter irrelevant context, significantly reducing the overall token cost. Extensive experiments demonstrate that LogicRAG achieves both superior performance and efficiency compared to state-of-the-art baselines.
>
---
#### [replaced 105] Unveiling Topological Structures from Language: A Survey of Topological Data Analysis Applications in NLP
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2411.10298v4](https://arxiv.org/pdf/2411.10298v4)**

> **作者:** Adaku Uchendu; Thai Le
>
> **摘要:** The surge of data available on the Internet has led to the adoption of various computational methods to analyze and extract valuable insights from this wealth of information. Among these, the field of Machine Learning (ML) has thrived by leveraging data to extract meaningful insights. However, ML techniques face notable challenges when dealing with real-world data, often due to issues of imbalance, noise, insufficient labeling, and high dimensionality. To address these limitations, some researchers advocate for the adoption of Topological Data Analysis (TDA), a statistical approach that discerningly captures the intrinsic shape of data despite noise. Despite its potential, TDA has not gained as much traction within the Natural Language Processing (NLP) domain compared to structurally distinct areas like computer vision. Nevertheless, a dedicated community of researchers has been exploring the application of TDA in NLP, yielding 100 papers we comprehensively survey in this paper. Our findings categorize these efforts into theoretical and non-theoretical approaches. Theoretical approaches aim to explain linguistic phenomena from a topological viewpoint, while non-theoretical approaches merge TDA with ML features, utilizing diverse numerical representation techniques. We conclude by exploring the challenges and unresolved questions that persist in this niche field. Resources and a list of papers on this topic can be found at: https://github.com/AdaUchendu/AwesomeTDA4NLP.
>
---
#### [replaced 106] Surface Reading LLMs: Synthetic Text and its Styles
- **分类: cs.CY; cs.CL**

- **链接: [https://arxiv.org/pdf/2510.22162v3](https://arxiv.org/pdf/2510.22162v3)**

> **作者:** Hannes Bajohr
>
> **备注:** 12 pages, 1 figure
>
> **摘要:** Despite a potential plateau in ML advancement, the societal impact of large language models lies not in approaching superintelligence but in generating text surfaces indistinguishable from human writing. While Critical AI Studies provides essential material and socio-technical critique, it risks overlooking how LLMs phenomenologically reshape meaning-making. This paper proposes a semiotics of "surface integrity" as attending to the immediate plane where LLMs inscribe themselves into human communication. I distinguish three knowledge interests in ML research (epistemology, epistēmē, and epistemics) and argue for integrating surface-level stylistic analysis alongside depth-oriented critique. Through two case studies examining stylistic markers of synthetic text, I argue how attending to style as a semiotic phenomenon reveals LLMs as cultural machines that transform the conditions of meaning emergence and circulation in contemporary discourse, independent of questions about machine consciousness.
>
---
#### [replaced 107] Lookahead Q-Cache: Achieving More Consistent KV Cache Eviction via Pseudo Query
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2505.20334v2](https://arxiv.org/pdf/2505.20334v2)**

> **作者:** Yixuan Wang; Shiyu Ji; Yijun Liu; Yuzhuang Xu; Yang Xu; Qingfu Zhu; Wanxiang Che
>
> **备注:** Accepted by EMNLP 2025 Main
>
> **摘要:** Large language models (LLMs) rely on key-value cache (KV cache) to accelerate decoding by reducing redundant computations. However, the KV cache memory usage grows substantially with longer text sequences, posing challenges for efficient deployment. Existing KV cache eviction methods prune tokens using prefilling-stage attention scores, causing inconsistency with actual inference queries, especially under tight memory budgets. In this paper, we propose Lookahead Q-Cache (LAQ), a novel eviction framework that generates low-cost pseudo lookahead queries to better approximate the true decoding-stage queries. By using these lookahead queries as the observation window for importance estimation, LAQ achieves more consistent and accurate KV cache eviction aligned with real inference scenarios. Experimental results on LongBench and Needle-in-a-Haystack benchmarks show that LAQ outperforms existing methods across various budget levels, achieving a 1 $\sim$ 4 point improvement on LongBench under limited cache budget. Moreover, LAQ is complementary to existing approaches and can be flexibly combined to yield further improvements.
>
---
#### [replaced 108] Glia: A Human-Inspired AI for Automated Systems Design and Optimization
- **分类: cs.AI; cs.CL; cs.DC**

- **链接: [https://arxiv.org/pdf/2510.27176v3](https://arxiv.org/pdf/2510.27176v3)**

> **作者:** Pouya Hamadanian; Pantea Karimi; Arash Nasr-Esfahany; Kimia Noorbakhsh; Joseph Chandler; Ali ParandehGheibi; Mohammad Alizadeh; Hari Balakrishnan
>
> **摘要:** Can an AI autonomously design mechanisms for computer systems on par with the creativity and reasoning of human experts? We present Glia, an AI architecture for networked systems design that uses large language models (LLMs) in a human-inspired, multi-agent workflow. Each agent specializes in reasoning, experimentation, and analysis, collaborating through an evaluation framework that grounds abstract reasoning in empirical feedback. Unlike prior ML-for-systems methods that optimize black-box policies, Glia generates interpretable designs and exposes its reasoning process. When applied to a distributed GPU cluster for LLM inference, it produces new algorithms for request routing, scheduling, and auto-scaling that perform at human-expert levels in significantly less time, while yielding novel insights into workload behavior. Our results suggest that by combining reasoning LLMs with structured experimentation, an AI can produce creative and understandable designs for complex systems problems.
>
---
#### [replaced 109] Scaling Laws for Conditional Emergence of Multilingual Image Captioning via Generalization from Translation
- **分类: cs.CL; cs.LG**

- **链接: [https://arxiv.org/pdf/2503.09443v2](https://arxiv.org/pdf/2503.09443v2)**

> **作者:** Julian Spravil; Sebastian Houben; Sven Behnke
>
> **摘要:** Cross-lingual, cross-task transfer is challenged by task-specific data scarcity, which becomes more severe as language support grows and is further amplified in vision-language models (VLMs). We investigate multilingual generalization in encoder-decoder transformer VLMs to enable zero-shot image captioning in languages encountered only in the translation task. In this setting, the encoder must learn to generate generalizable, task-aware latent vision representations to instruct the decoder via inserted cross-attention layers. To analyze scaling behavior, we train Florence-2 based and Gemma-2 based models (0.4B to 11.2B parameters) on a synthetic dataset using varying compute budgets. While all languages in the dataset have image-aligned translations, only a subset of them include image captions. Notably, we show that captioning can emerge using a language prefix, even when this language only appears in the translation task. We find that indirect learning of unseen task-language pairs adheres to scaling laws that are governed by the multilinguality of the model, model size, and seen training samples. Finally, we demonstrate that the scaling laws extend to downstream tasks, achieving competitive performance through fine-tuning in multimodal machine translation (Multi30K, CoMMuTE), lexical disambiguation (CoMMuTE), and image captioning (Multi30K, XM3600, COCO Karpathy).
>
---
#### [replaced 110] Unveiling the Influence of Amplifying Language-Specific Neurons
- **分类: cs.CL; cs.LG**

- **链接: [https://arxiv.org/pdf/2507.22581v3](https://arxiv.org/pdf/2507.22581v3)**

> **作者:** Inaya Rahmanisa; Lyzander Marciano Andrylie; Mahardika Krisna Ihsani; Alfan Farizki Wicaksono; Haryo Akbarianto Wibowo; Alham Fikri Aji
>
> **备注:** Accepted to AACL 2025. Our code and dataset are made available at https://github.com/tauimbz/lang-task-neuron
>
> **摘要:** Language-specific neurons in LLMs that strongly correlate with individual languages have been shown to influence model behavior by deactivating them. However, their role in amplification remains underexplored. This work investigates the effect of amplifying language-specific neurons through interventions across 18 languages, including low-resource ones, using three models primarily trained in different languages. We compare amplification factors by their effectiveness in steering to the target language using a proposed Language Steering Shift (LSS) evaluation score, then evaluate it on downstream tasks: commonsense reasoning (XCOPA, XWinograd), knowledge (Include), and translation (FLORES). The optimal amplification factors effectively steer output toward nearly all tested languages. Intervention using this factor on downstream tasks improves self-language performance in some cases but generally degrades cross-language results. These findings highlight the effect of language-specific neurons in multilingual behavior, where amplification can be beneficial especially for low-resource languages, but provides limited advantage for cross-lingual transfer.
>
---
#### [replaced 111] Accelerated Test-Time Scaling with Model-Free Speculative Sampling
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2506.04708v2](https://arxiv.org/pdf/2506.04708v2)**

> **作者:** Woomin Song; Saket Dingliwal; Sai Muralidhar Jayanthi; Bhavana Ganesh; Jinwoo Shin; Aram Galstyan; Sravan Babu Bodapati
>
> **备注:** EMNLP 2025 Oral
>
> **摘要:** Language models have demonstrated remarkable capabilities in reasoning tasks through test-time scaling techniques like best-of-N sampling and tree search. However, these approaches often demand substantial computational resources, creating a critical trade-off between performance and efficiency. We introduce STAND (STochastic Adaptive N-gram Drafting), a novel model-free speculative decoding approach that exploits the inherent redundancy in reasoning trajectories to achieve significant acceleration without compromising accuracy. Our analysis shows that reasoning paths frequently reuse similar reasoning patterns, enabling efficient model-free token prediction without requiring separate draft models. By introducing stochastic drafting and preserving probabilistic information through a memory-efficient logit-based N-gram module, combined with optimized Gumbel-Top-K sampling and data-driven tree construction, STAND significantly improves token acceptance rates. Extensive evaluations across multiple models and reasoning tasks (AIME-2024, GPQA-Diamond, and LiveCodeBench) demonstrate that STAND reduces inference latency by 60-65% compared to standard autoregressive decoding while maintaining accuracy. Furthermore, STAND consistently outperforms state-of-the-art speculative decoding methods across diverse inference patterns, including single-trajectory decoding, batch decoding, and test-time tree search. As a model-free approach, STAND can be applied to any existing language model without additional training, making it a powerful plug-and-play solution for accelerating language model reasoning.
>
---
#### [replaced 112] Accommodate Knowledge Conflicts in Retrieval-augmented LLMs: Towards Robust Response Generation in the Wild
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2504.12982v2](https://arxiv.org/pdf/2504.12982v2)**

> **作者:** Jiatai Wang; Zhiwei Xu; Di Jin; Xuewen Yang; Tao Li
>
> **摘要:** The proliferation of large language models (LLMs) has significantly advanced intelligent systems. Unfortunately, LLMs often face knowledge conflicts between internal memory and retrieved external information, arising from misinformation, biases, or outdated knowledge. These conflicts undermine response reliability and introduce uncertainty in decision-making. In this work, we analyze how LLMs navigate knowledge conflicts from an information-theoretic perspective and reveal that when conflicting and supplementary information exhibit significant differences, LLMs confidently resolve their preferences and alleviate the uncertainty during their response generation. When this difference is ambiguous, LLMs experience considerable uncertainty about their generation. Based on this insight, we propose Swin-VIB, a novel framework that integrates a pipeline of variational information bottleneck models to adapt the retrieved information difference, facilitating robust response generation of LLMs even in conflicting contexts. Extensive experiments confirm our theoretical analysis and demonstrate the performance of Swin-VIB. Notably, Swin-VIB outperforms all competitive baselines in terms of the accuracy of the multiple-choice task, while improving the EM values in the open-ended QA task by at least 11.14%.
>
---
