# 自然语言处理 cs.CL

- **最新发布 82 篇**

- **更新 45 篇**

## 最新发布

#### [new 001] Tokenization, Fusion and Decoupling: Bridging the Granularity Mismatch Between Large Language Models and Knowledge Graphs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识图谱补全任务，解决大语言模型与知识图谱在粒度上的不匹配问题。通过引入专用实体标记、特征融合和解耦预测，提升预测效果。**

- **链接: [https://arxiv.org/pdf/2602.22698v1](https://arxiv.org/pdf/2602.22698v1)**

> **作者:** Siyue Su; Jian Yang; Bo Li; Guanglin Niu
>
> **摘要:** Leveraging Large Language Models (LLMs) for Knowledge Graph Completion (KGC) is promising but hindered by a fundamental granularity mismatch. LLMs operate on fragmented token sequences, whereas entities are the fundamental units in knowledge graphs (KGs) scenarios. Existing approaches typically constrain predictions to limited candidate sets or align entities with the LLM's vocabulary by pooling multiple tokens or decomposing entities into fixed-length token sequences, which fail to capture both the semantic meaning of the text and the structural integrity of the graph. To address this, we propose KGT, a novel framework that uses dedicated entity tokens to enable efficient, full-space prediction. Specifically, we first introduce specialized tokenization to construct feature representations at the level of dedicated entity tokens. We then fuse pre-trained structural and textual features into these unified embeddings via a relation-guided gating mechanism, avoiding training from scratch. Finally, we implement decoupled prediction by leveraging independent heads to separate and combine semantic and structural reasoning. Experimental results show that KGT consistently outperforms state-of-the-art methods across multiple benchmarks.
>
---
#### [new 002] Decoder-based Sense Knowledge Distillation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识蒸馏任务，旨在提升解码器模型的结构化语义能力。通过整合词义资源，解决解码器在知识蒸馏中的结构化词汇知识缺失问题。**

- **链接: [https://arxiv.org/pdf/2602.22351v1](https://arxiv.org/pdf/2602.22351v1)**

> **作者:** Qitong Wang; Mohammed J. Zaki; Georgios Kollias; Vasileios Kalantzis
>
> **摘要:** Large language models (LLMs) learn contextual embeddings that capture rich semantic information, yet they often overlook structured lexical knowledge such as word senses and relationships. Prior work has shown that incorporating sense dictionaries can improve knowledge distillation for encoder models, but their application to decoder as generative models remains challenging. In this paper, we introduce Decoder-based Sense Knowledge Distillation (DSKD), a framework that integrates lexical resources into the training of decoder-style LLMs without requiring dictionary lookup at inference time. Extensive experiments on diverse benchmarks demonstrate that DSKD significantly enhances knowledge distillation performance for decoders, enabling generative models to inherit structured semantics while maintaining efficient training.
>
---
#### [new 003] Detecting Hate and Inflammatory Content in Bengali Memes: A New Multimodal Dataset and Co-Attention Framework
- **分类: cs.CL**

- **简介: 该论文属于多模态内容检测任务，旨在解决 Bengali 社交媒体中仇恨与煽动性表情包的识别问题。研究构建了首个区分仇恨与煽动内容的 Bengali 数据集，并提出一种联合注意力模型进行准确分类。**

- **链接: [https://arxiv.org/pdf/2602.22391v1](https://arxiv.org/pdf/2602.22391v1)**

> **作者:** Rakib Ullah; Mominul islam; Md Sanjid Hossain; Md Ismail Hossain
>
> **备注:** 6 pages, 8 figures
>
> **摘要:** Internet memes have become a dominant form of expression on social media, including within the Bengali-speaking community. While often humorous, memes can also be exploited to spread offensive, harmful, and inflammatory content targeting individuals and groups. Detecting this type of content is excep- tionally challenging due to its satirical, subtle, and culturally specific nature. This problem is magnified for low-resource lan- guages like Bengali, as existing research predominantly focuses on high-resource languages. To address this critical research gap, we introduce Bn-HIB (Bangla Hate Inflammatory Benign), a novel dataset containing 3,247 manually annotated Bengali memes categorized as Benign, Hate, or Inflammatory. Significantly, Bn- HIB is the first dataset to distinguish inflammatory content from direct hate speech in Bengali memes. Furthermore, we propose the MCFM (Multi-Modal Co-Attention Fusion Model), a simple yet effective architecture that mutually analyzes both the visual and textual elements of a meme. MCFM employs a co-attention mechanism to identify and fuse the most critical features from each modality, leading to a more accurate classification. Our experiments show that MCFM significantly outperforms several state-of-the-art models on the Bn-HIB dataset, demonstrating its effectiveness in this nuanced task.Warning: This work contains material that may be disturbing to some audience members. Viewer discretion is advised.
>
---
#### [new 004] A Mixture-of-Experts Model for Multimodal Emotion Recognition in Conversations
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于情感识别任务，解决对话中多模态情绪识别问题。提出MiSTER-E框架，融合语音和文本信息，提升情绪识别效果。**

- **链接: [https://arxiv.org/pdf/2602.23300v1](https://arxiv.org/pdf/2602.23300v1)**

> **作者:** Soumya Dutta; Smruthi Balaji; Sriram Ganapathy
>
> **备注:** Accepted to Elsevier Computer Speech and Language. 30 pages, 9 figures, 5 tables
>
> **摘要:** Emotion Recognition in Conversations (ERC) presents unique challenges, requiring models to capture the temporal flow of multi-turn dialogues and to effectively integrate cues from multiple modalities. We propose Mixture of Speech-Text Experts for Recognition of Emotions (MiSTER-E), a modular Mixture-of-Experts (MoE) framework designed to decouple two core challenges in ERC: modality-specific context modeling and multimodal information fusion. MiSTER-E leverages large language models (LLMs) fine-tuned for both speech and text to provide rich utterance-level embeddings, which are then enhanced through a convolutional-recurrent context modeling layer. The system integrates predictions from three experts-speech-only, text-only, and cross-modal-using a learned gating mechanism that dynamically weighs their outputs. To further encourage consistency and alignment across modalities, we introduce a supervised contrastive loss between paired speech-text representations and a KL-divergence-based regulariza-tion across expert predictions. Importantly, MiSTER-E does not rely on speaker identity at any stage. Experiments on three benchmark datasets-IEMOCAP, MELD, and MOSI-show that our proposal achieves 70.9%, 69.5%, and 87.9% weighted F1-scores respectively, outperforming several baseline speech-text ERC systems. We also provide various ablations to highlight the contributions made in the proposed approach.
>
---
#### [new 005] SAFARI: A Community-Engaged Approach and Dataset of Stereotype Resources in the Sub-Saharan African Context
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决亚撒哈拉非洲地区刻板印象数据缺失问题。通过社区参与方法，构建多语言刻板印象数据集，覆盖四国，提升AI模型安全性评估的全球覆盖。**

- **链接: [https://arxiv.org/pdf/2602.22404v1](https://arxiv.org/pdf/2602.22404v1)**

> **作者:** Aishwarya Verma; Laud Ammah; Olivia Nercy Ndlovu Lucas; Andrew Zaldivar; Vinodkumar Prabhakaran; Sunipa Dev
>
> **摘要:** Stereotype repositories are critical to assess generative AI model safety, but currently lack adequate global coverage. It is imperative to prioritize targeted expansion, strategically addressing existing deficits, over merely increasing data volume. This work introduces a multilingual stereotype resource covering four sub-Saharan African countries that are severely underrepresented in NLP resources: Ghana, Kenya, Nigeria, and South Africa. By utilizing socioculturally-situated, community-engaged methods, including telephonic surveys moderated in native languages, we establish a reproducible methodology that is sensitive to the region's complex linguistic diversity and traditional orality. By deliberately balancing the sample across diverse ethnic and demographic backgrounds, we ensure broad coverage, resulting in a dataset of 3,534 stereotypes in English and 3,206 stereotypes across 15 native languages.
>
---
#### [new 006] Human Label Variation in Implicit Discourse Relation Recognition
- **分类: cs.CL**

- **简介: 论文研究隐含话语关系识别（IDRR）任务，探讨人类标注者间的差异。针对标注不一致问题，比较了基于标签分布和个体标注者模型的方法，发现标签分布方法更稳定。**

- **链接: [https://arxiv.org/pdf/2602.22723v1](https://arxiv.org/pdf/2602.22723v1)**

> **作者:** Frances Yung; Daniil Ignatev; Merel Scholman; Vera Demberg; Massimo Poesio
>
> **摘要:** There is growing recognition that many NLP tasks lack a single ground truth, as human judgments reflect diverse perspectives. To capture this variation, models have been developed to predict full annotation distributions rather than majority labels, while perspectivist models aim to reproduce the interpretations of individual annotators. In this work, we compare these approaches on Implicit Discourse Relation Recognition (IDRR), a highly ambiguous task where disagreement often arises from cognitive complexity rather than ideological bias. Our experiments show that existing annotator-specific models perform poorly in IDRR unless ambiguity is reduced, whereas models trained on label distributions yield more stable predictions. Further analysis indicates that frequent cognitively demanding cases drive inconsistency in human interpretation, posing challenges for perspectivist modeling in IDRR.
>
---
#### [new 007] AuditBench: Evaluating Alignment Auditing Techniques on Models with Hidden Behaviors
- **分类: cs.CL**

- **简介: 该论文提出AuditBench，用于评估模型对隐藏行为的对齐审计。任务是检测模型中未公开的潜在行为，通过构建多样化模型和审计工具进行测试与优化。**

- **链接: [https://arxiv.org/pdf/2602.22755v1](https://arxiv.org/pdf/2602.22755v1)**

> **作者:** Abhay Sheshadri; Aidan Ewart; Kai Fronsdal; Isha Gupta; Samuel R. Bowman; Sara Price; Samuel Marks; Rowan Wang
>
> **摘要:** We introduce AuditBench, an alignment auditing benchmark. AuditBench consists of 56 language models with implanted hidden behaviors. Each model has one of 14 concerning behaviors--such as sycophantic deference, opposition to AI regulation, or secret geopolitical loyalties--which it does not confess to when directly asked. AuditBench models are highly diverse--some are subtle, while others are overt, and we use varying training techniques both for implanting behaviors and training models not to confess. To demonstrate AuditBench's utility, we develop an investigator agent that autonomously employs a configurable set of auditing tools. By measuring investigator agent success using different tools, we can evaluate their efficacy. Notably, we observe a tool-to-agent gap, where tools that perform well in standalone non-agentic evaluations fail to translate into improved performance when used with our investigator agent. We find that our most effective tools involve scaffolded calls to auxiliary models that generate diverse prompts for the target. White-box interpretability tools can be helpful, but the agent performs best with black-box tools. We also find that audit success varies greatly across training techniques: models trained on synthetic documents are easier to audit than models trained on demonstrations, with better adversarial training further increasing auditing difficulty. We release our models, agent, and evaluation framework to support future quantitative, iterative science on alignment auditing.
>
---
#### [new 008] MTRAG-UN: A Benchmark for Open Challenges in Multi-Turn RAG Conversations
- **分类: cs.CL**

- **简介: 该论文提出MTRAG-UN基准，用于研究多轮RAG对话中的开放挑战。解决多轮对话中回答不可回答、不明确和非独立问题的难题，包含6个领域的任务数据集。**

- **链接: [https://arxiv.org/pdf/2602.23184v1](https://arxiv.org/pdf/2602.23184v1)**

> **作者:** Sara Rosenthal; Yannis Katsis; Vraj Shah; Lihong He; Lucian Popa; Marina Danilevsky
>
> **备注:** 5 pages, 3 figures
>
> **摘要:** We present MTRAG-UN, a benchmark for exploring open challenges in multi-turn retrieval augmented generation, a popular use of large language models. We release a benchmark of 666 tasks containing over 2,800 conversation turns across 6 domains with accompanying corpora. Our experiments show that retrieval and generation models continue to struggle on conversations with UNanswerable, UNderspecified, and NONstandalone questions and UNclear responses. Our benchmark is available at https://github.com/IBM/mt-rag-benchmark
>
---
#### [new 009] Iterative Prompt Refinement for Dyslexia-Friendly Text Summarization Using GPT-4o
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本摘要任务，旨在解决阅读障碍者难以理解复杂文本的问题。通过迭代提示优化方法，提升摘要的可读性与语义准确性。**

- **链接: [https://arxiv.org/pdf/2602.22524v1](https://arxiv.org/pdf/2602.22524v1)**

> **作者:** Samay Bhojwani; Swarnima Kain; Lisong Xu
>
> **摘要:** Dyslexia affects approximately 10% of the global population and presents persistent challenges in reading fluency and text comprehension. While existing assistive technologies address visual presentation, linguistic complexity remains a substantial barrier to equitable access. This paper presents an empirical study on dyslexia-friendly text summarization using an iterative prompt-based refinement pipeline built on GPT-4o. We evaluate the pipeline on approximately 2,000 news article samples, applying a readability target of Flesch Reading Ease >= 90. Results show that the majority of summaries meet the readability threshold within four attempts, with many succeeding on the first try. A composite score combining readability and semantic fidelity shows stable performance across the dataset, ranging from 0.13 to 0.73 with a typical value near 0.55. These findings establish an empirical baseline for accessibility-driven NLP summarization and motivate further human-centered evaluation with dyslexic readers.
>
---
#### [new 010] Probing for Knowledge Attribution in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识归属任务，旨在识别大语言模型输出的知识来源。通过构建数据集和训练分类器，解决模型生成内容是否依赖上下文或内部知识的问题。**

- **链接: [https://arxiv.org/pdf/2602.22787v1](https://arxiv.org/pdf/2602.22787v1)**

> **作者:** Ivo Brink; Alexander Boer; Dennis Ulmer
>
> **摘要:** Large language models (LLMs) often generate fluent but unfounded claims, or hallucinations, which fall into two types: (i) faithfulness violations - misusing user context - and (ii) factuality violations - errors from internal knowledge. Proper mitigation depends on knowing whether a model's answer is based on the prompt or its internal weights. This work focuses on the problem of contributive attribution: identifying the dominant knowledge source behind each output. We show that a probe, a simple linear classifier trained on model hidden representations, can reliably predict contributive attribution. For its training, we introduce AttriWiki, a self-supervised data pipeline that prompts models to recall withheld entities from memory or read them from context, generating labelled examples automatically. Probes trained on AttriWiki data reveal a strong attribution signal, achieving up to 0.96 Macro-F1 on Llama-3.1-8B, Mistral-7B, and Qwen-7B, transferring to out-of-domain benchmarks (SQuAD, WebQuestions) with 0.94-0.99 Macro-F1 without retraining. Attribution mismatches raise error rates by up to 70%, demonstrating a direct link between knowledge source confusion and unfaithful answers. Yet, models may still respond incorrectly even when attribution is correct, highlighting the need for broader detection frameworks.
>
---
#### [new 011] Extending Czech Aspect-Based Sentiment Analysis with Opinion Terms: Dataset and LLM Benchmarks
- **分类: cs.CL**

- **简介: 该论文属于情感分析任务，旨在解决 Czech 语言的基于方面的情感分析问题。通过构建包含观点词的语料库，并利用 LLM 进行跨语言迁移，提升低资源语言的分析效果。**

- **链接: [https://arxiv.org/pdf/2602.22730v1](https://arxiv.org/pdf/2602.22730v1)**

> **作者:** Jakub Šmíd; Pavel Přibáň; Pavel Král
>
> **备注:** Accepted for the 15th edition of the Language Resources and Evaluation Conference (LREC 2026)
>
> **摘要:** This paper introduces a novel Czech dataset in the restaurant domain for aspect-based sentiment analysis (ABSA), enriched with annotations of opinion terms. The dataset supports three distinct ABSA tasks involving opinion terms, accommodating varying levels of complexity. Leveraging this dataset, we conduct extensive experiments using modern Transformer-based models, including large language models (LLMs), in monolingual, cross-lingual, and multilingual settings. To address cross-lingual challenges, we propose a translation and label alignment methodology leveraging LLMs, which yields consistent improvements. Our results highlight the strengths and limitations of state-of-the-art models, especially when handling the linguistic intricacies of low-resource languages like Czech. A detailed error analysis reveals key challenges, including the detection of subtle opinion terms and nuanced sentiment expressions. The dataset establishes a new benchmark for Czech ABSA, and our proposed translation-alignment approach offers a scalable solution for adapting ABSA resources to other low-resource languages.
>
---
#### [new 012] Enhancing Persuasive Dialogue Agents by Synthesizing Cross-Disciplinary Communication Strategies
- **分类: cs.CL**

- **简介: 该论文属于对话系统任务，旨在提升说服性对话代理的效果。针对现有策略单一的问题，融合多学科策略并验证其有效性，显著提高说服成功率和适用性。**

- **链接: [https://arxiv.org/pdf/2602.22696v1](https://arxiv.org/pdf/2602.22696v1)**

> **作者:** Shinnosuke Nozue; Yuto Nakano; Yotaro Watanabe; Meguru Takasaki; Shoji Moriya; Reina Akama; Jun Suzuki
>
> **备注:** Accepted to the EMNLP 2025 Industry Track; 26 pages
>
> **摘要:** Current approaches to developing persuasive dialogue agents often rely on a limited set of predefined persuasive strategies that fail to capture the complexity of real-world interactions. We applied a cross-disciplinary approach to develop a framework for designing persuasive dialogue agents that draws on proven strategies from social psychology, behavioral economics, and communication theory. We validated our proposed framework through experiments on two distinct datasets: the Persuasion for Good dataset, which represents a specific in-domain scenario, and the DailyPersuasion dataset, which encompasses a wide range of scenarios. The proposed framework achieved strong results for both datasets and demonstrated notable improvement in the persuasion success rate as well as promising generalizability. Notably, the proposed framework also excelled at persuading individuals with initially low intent, which addresses a critical challenge for persuasive dialogue agents.
>
---
#### [new 013] A Fusion of context-aware based BanglaBERT and Two-Layer Stacked LSTM Framework for Multi-Label Cyberbullying Detection
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于多标签网络欺凌检测任务，旨在解决单一标签分类无法准确识别复杂网络欺凌内容的问题。通过融合BanglaBERT与两层堆叠LSTM模型提升检测效果。**

- **链接: [https://arxiv.org/pdf/2602.22449v1](https://arxiv.org/pdf/2602.22449v1)**

> **作者:** Mirza Raquib; Asif Pervez Polok; Kedar Nath Biswas; Rahat Uddin Azad; Saydul Akbar Murad; Nick Rahimi
>
> **摘要:** Cyberbullying has become a serious and growing concern in todays virtual world. When left unnoticed, it can have adverse consequences for social and mental health. Researchers have explored various types of cyberbullying, but most approaches use single-label classification, assuming that each comment contains only one type of abuse. In reality, a single comment may include overlapping forms such as threats, hate speech, and harassment. Therefore, multilabel detection is both realistic and essential. However, multilabel cyberbullying detection has received limited attention, especially in low-resource languages like Bangla, where robust pre-trained models are scarce. Developing a generalized model with moderate accuracy remains challenging. Transformers offer strong contextual understanding but may miss sequential dependencies, while LSTM models capture temporal flow but lack semantic depth. To address these limitations, we propose a fusion architecture that combines BanglaBERT-Large with a two-layer stacked LSTM. We analyze their behavior to jointly model context and sequence. The model is fine-tuned and evaluated on a publicly available multilabel Bangla cyberbullying dataset covering cyberbully, sexual harassment, threat, and spam. We apply different sampling strategies to address class imbalance. Evaluation uses multiple metrics, including accuracy, precision, recall, F1-score, Hamming loss, Cohens kappa, and AUC-ROC. We employ 5-fold cross-validation to assess the generalization of the architecture.
>
---
#### [new 014] Natural Language Declarative Prompting (NLD-P): A Modular Governance Method for Prompt Design Under Model Drift
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出NLD-P方法，解决模型演进下的提示治理问题，通过自然语言模块化控制实现稳定指令管理。**

- **链接: [https://arxiv.org/pdf/2602.22790v1](https://arxiv.org/pdf/2602.22790v1)**

> **作者:** Hyunwoo Kim; Hanau Yi; Jaehee Bae; Yumin Kim
>
> **摘要:** The rapid evolution of large language models (LLMs) has transformed prompt engineering from a localized craft into a systems-level governance challenge. As models scale and update across generations, prompt behavior becomes sensitive to shifts in instruction-following policies, alignment regimes, and decoding strategies, a phenomenon we characterize as GPT-scale model drift. Under such conditions, surface-level formatting conventions and ad hoc refinement are insufficient to ensure stable, interpretable control. This paper reconceptualizes Natural Language Declarative Prompting (NLD-P) as a declarative governance method rather than a rigid field template. NLD-P is formalized as a modular control abstraction that separates provenance, constraint logic, task content, and post-generation evaluation, encoded directly in natural language without reliance on external orchestration code. We define minimal compliance criteria, analyze model-dependent schema receptivity, and position NLD-P as an accessible governance framework for non-developer practitioners operating within evolving LLM ecosystems. Portions of drafting and editorial refinement employed a schema-bound LLM assistant configured under NLD-P. All conceptual framing, methodological claims, and final revisions were directed, reviewed, and approved by the human author under a documented human-in-the-loop protocol. The paper concludes by outlining implications for declarative control under ongoing model evolution and identifying directions for future empirical validation.
>
---
#### [new 015] Bridging Latent Reasoning and Target-Language Generation via Retrieval-Transition Heads
- **分类: cs.CL**

- **简介: 该论文研究多语言模型中注意力头的作用，解决如何区分不同功能头的问题。通过分析检索头和转换头，发现转换头对跨语言推理更关键。**

- **链接: [https://arxiv.org/pdf/2602.22453v1](https://arxiv.org/pdf/2602.22453v1)**

> **作者:** Shaswat Patel; Vishvesh Trivedi; Yue Han; Yihuai Hong; Eunsol Choi
>
> **摘要:** Recent work has identified a subset of attention heads in Transformer as retrieval heads, which are responsible for retrieving information from the context. In this work, we first investigate retrieval heads in multilingual contexts. In multilingual language models, we find that retrieval heads are often shared across multiple languages. Expanding the study to cross-lingual setting, we identify Retrieval-Transition heads(RTH), which govern the transition to specific target-language output. Our experiments reveal that RTHs are distinct from retrieval heads and more vital for Chain-of-Thought reasoning in multilingual LLMs. Across four multilingual benchmarks (MMLU-ProX, MGSM, MLQA, and XQuaD) and two model families (Qwen-2.5 and Llama-3.1), we demonstrate that masking RTH induces bigger performance drop than masking Retrieval Heads (RH). Our work advances understanding of multilingual LMs by isolating the attention heads responsible for mapping to target languages.
>
---
#### [new 016] Assessing Deanonymization Risks with Stylometry-Assisted LLM Agent
- **分类: cs.CL; cs.CR; cs.LG**

- **简介: 该论文属于作者身份识别任务，旨在评估和降低文本中的去匿名化风险。通过结合风格分析与大语言模型，提出SALA方法，并设计改写策略以保护隐私。**

- **链接: [https://arxiv.org/pdf/2602.23079v1](https://arxiv.org/pdf/2602.23079v1)**

> **作者:** Boyang Zhang; Yang Zhang
>
> **摘要:** The rapid advancement of large language models (LLMs) has enabled powerful authorship inference capabilities, raising growing concerns about unintended deanonymization risks in textual data such as news articles. In this work, we introduce an LLM agent designed to evaluate and mitigate such risks through a structured, interpretable pipeline. Central to our framework is the proposed $\textit{SALA}$ (Stylometry-Assisted LLM Analysis) method, which integrates quantitative stylometric features with LLM reasoning for robust and transparent authorship attribution. Experiments on large-scale news datasets demonstrate that $\textit{SALA}$, particularly when augmented with a database module, achieves high inference accuracy in various scenarios. Finally, we propose a guided recomposition strategy that leverages the agent's reasoning trace to generate rewriting prompts, effectively reducing authorship identifiability while preserving textual meaning. Our findings highlight both the deanonymization potential of LLM agents and the importance of interpretable, proactive defenses for safeguarding author privacy.
>
---
#### [new 017] Ruyi2 Technical Report
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型优化任务，旨在降低大语言模型的部署成本与延迟。通过引入Ruyi2，实现高效变深度计算，提升训练速度并保持性能。**

- **链接: [https://arxiv.org/pdf/2602.22543v1](https://arxiv.org/pdf/2602.22543v1)**

> **作者:** Huan Song; Shuyu Tian; Junyi Hao; Minxiu Xu; Hongjun An; Yiliang Song; Jiawei Shao; Xuelong Li
>
> **摘要:** Large Language Models (LLMs) face significant challenges regarding deployment costs and latency, necessitating adaptive computing strategies. Building upon the AI Flow framework, we introduce Ruyi2 as an evolution of our adaptive model series designed for efficient variable-depth computation. While early-exit architectures offer a viable efficiency-performance balance, the Ruyi model and existing methods often struggle with optimization complexity and compatibility with large-scale distributed training. To bridge this gap, Ruyi2 introduces a stable "Familial Model" based on Megatron-LM. By using 3D parallel training, it achieves a 2-3 times speedup over Ruyi, while performing comparably to same-sized Qwen3 models. These results confirm that family-based parameter sharing is a highly effective strategy, establishing a new "Train Once, Deploy Many" paradigm and providing a key reference for balancing architectural efficiency with high-performance capabilities.
>
---
#### [new 018] Effective QA-driven Annotation of Predicate-Argument Relations Across Languages
- **分类: cs.CL**

- **简介: 该论文属于语义角色标注任务，旨在解决跨语言谓词-论元关系标注难题。通过QA-SRL框架，实现高效、高质量的多语言语义标注。**

- **链接: [https://arxiv.org/pdf/2602.22865v1](https://arxiv.org/pdf/2602.22865v1)**

> **作者:** Jonathan Davidov; Aviv Slobodkin; Shmuel Tomi Klein; Reut Tsarfaty; Ido Dagan; Ayal Klein
>
> **备注:** Accepted to EACL 2026 (Main Conference)
>
> **摘要:** Explicit representations of predicate-argument relations form the basis of interpretable semantic analysis, supporting reasoning, generation, and evaluation. However, attaining such semantic structures requires costly annotation efforts and has remained largely confined to English. We leverage the Question-Answer driven Semantic Role Labeling (QA-SRL) framework -- a natural-language formulation of predicate-argument relations -- as the foundation for extending semantic annotation to new languages. To this end, we introduce a cross-linguistic projection approach that reuses an English QA-SRL parser within a constrained translation and word-alignment pipeline to automatically generate question-answer annotations aligned with target-language predicates. Applied to Hebrew, Russian, and French -- spanning diverse language families -- the method yields high-quality training data and fine-tuned, language-specific parsers that outperform strong multilingual LLM baselines (GPT-4o, LLaMA-Maverick). By leveraging QA-SRL as a transferable natural-language interface for semantics, our approach enables efficient and broadly accessible predicate-argument parsing across languages.
>
---
#### [new 019] Why Diffusion Language Models Struggle with Truly Parallel (Non-Autoregressive) Decoding?
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言生成任务，解决扩散语言模型在并行解码中表现类似自回归的问题。通过调整训练数据和监督方式，提出NAP方法提升非自回归并行生成效果。**

- **链接: [https://arxiv.org/pdf/2602.23225v1](https://arxiv.org/pdf/2602.23225v1)**

> **作者:** Pengxiang Li; Dilxat Muhtar; Lu Yin; Tianlong Chen; Shiwei Liu
>
> **摘要:** Diffusion Language Models (DLMs) are often advertised as enabling parallel token generation, yet practical fast DLMs frequently converge to left-to-right, autoregressive (AR)-like decoding dynamics. In contrast, genuinely non-AR generation is promising because it removes AR's sequential bottleneck, better exploiting parallel hardware to reduce synchronization/communication overhead and improve latency scaling with output length. We argue that a primary driver of AR-like decoding is a mismatch between DLM objectives and the highly sequential structure of widely used training data, including standard pretraining corpora and long chain-of-thought (CoT) supervision. Motivated by this diagnosis, we propose NAP (Non-Autoregressive Parallel DLMs), a proof-of-concept, data-centric approach that better aligns supervision with non-AR parallel decoding. NAP curates examples as multiple independent reasoning trajectories and couples them with a parallel-forced decoding strategy that encourages multi-token parallel updates. Across math reasoning benchmarks, NAP yields stronger performance under parallel decoding than DLMs trained on standard long CoT data, with gains growing as parallelism increases. Our results suggest that revisiting data and supervision is a principled direction for mitigating AR-like behavior and moving toward genuinely non-autoregressive parallel generation in DLMs. Our code is available at https://github.com/pixeli99/NAP.
>
---
#### [new 020] dLLM: Simple Diffusion Language Modeling
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出dLLM框架，统一扩散语言建模的训练、推理与评估流程，解决模型复现和扩展困难的问题，支持快速定制和部署大模型。**

- **链接: [https://arxiv.org/pdf/2602.22661v1](https://arxiv.org/pdf/2602.22661v1)**

> **作者:** Zhanhui Zhou; Lingjie Chen; Hanghang Tong; Dawn Song
>
> **备注:** Code available at: https://github.com/ZHZisZZ/dllm
>
> **摘要:** Although diffusion language models (DLMs) are evolving quickly, many recent models converge on a set of shared components. These components, however, are distributed across ad-hoc research codebases or lack transparent implementations, making them difficult to reproduce or extend. As the field accelerates, there is a clear need for a unified framework that standardizes these common components while remaining flexible enough to support new methods and architectures. To address this gap, we introduce dLLM, an open-source framework that unifies the core components of diffusion language modeling -- training, inference, and evaluation -- and makes them easy to customize for new designs. With dLLM, users can reproduce, finetune, deploy, and evaluate open-source large DLMs such as LLaDA and Dream through a standardized pipeline. The framework also provides minimal, reproducible recipes for building small DLMs from scratch with accessible compute, including converting any BERT-style encoder or autoregressive LM into a DLM. We also release the checkpoints of these small DLMs to make DLMs more accessible and accelerate future research.
>
---
#### [new 021] Towards Simulating Social Media Users with LLMs: Evaluating the Operational Validity of Conditioned Comment Prediction
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于社会媒体用户模拟任务，旨在验证LLMs在行为模拟中的操作有效性。通过条件评论预测，评估模型在不同语言和微调策略下的表现，揭示结构与语义的分离问题。**

- **链接: [https://arxiv.org/pdf/2602.22752v1](https://arxiv.org/pdf/2602.22752v1)**

> **作者:** Nils Schwager; Simon Münker; Alistair Plum; Achim Rettinger
>
> **备注:** 14 pages, 1 figure, 7 tables. Accepted to the 15th Workshop on Computational Approaches to Subjectivity, Sentiment & Social Media Analysis (WASSA) at EACL 2026, Rabat, Morocco
>
> **摘要:** The transition of Large Language Models (LLMs) from exploratory tools to active "silicon subjects" in social science lacks extensive validation of operational validity. This study introduces Conditioned Comment Prediction (CCP), a task in which a model predicts how a user would comment on a given stimulus by comparing generated outputs with authentic digital traces. This framework enables a rigorous evaluation of current LLM capabilities with respect to the simulation of social media user behavior. We evaluated open-weight 8B models (Llama3.1, Qwen3, Ministral) in English, German, and Luxembourgish language scenarios. By systematically comparing prompting strategies (explicit vs. implicit) and the impact of Supervised Fine-Tuning (SFT), we identify a critical form vs. content decoupling in low-resource settings: while SFT aligns the surface structure of the text output (length and syntax), it degrades semantic grounding. Furthermore, we demonstrate that explicit conditioning (generated biographies) becomes redundant under fine-tuning, as models successfully perform latent inference directly from behavioral histories. Our findings challenge current "naive prompting" paradigms and offer operational guidelines prioritizing authentic behavioral traces over descriptive personas for high-fidelity simulation.
>
---
#### [new 022] Search More, Think Less: Rethinking Long-Horizon Agentic Search for Efficiency and Generalization
- **分类: cs.CL**

- **简介: 该论文属于智能搜索任务，旨在解决长周期代理搜索的效率与泛化问题。提出SMTL框架，通过并行证据获取提升效率，并构建统一数据合成管道增强泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.22675v1](https://arxiv.org/pdf/2602.22675v1)**

> **作者:** Qianben Chen; Tianrui Qin; King Zhu; Qiexiang Wang; Chengjun Yu; Shu Xu; Jiaqi Wu; Jiayu Zhang; Xinpeng Liu; Xin Gui; Jingyi Cao; Piaohong Wang; Dingfeng Shi; He Zhu; Tiannan Wang; Yuqing Wang; Maojia Song; Tianyu Zheng; Ge Zhang; Jian Yang; Jiaheng Liu; Minghao Liu; Yuchen Eleanor Jiang; Wangchunshu Zhou
>
> **备注:** 12 pages, 5 figures
>
> **摘要:** Recent deep research agents primarily improve performance by scaling reasoning depth, but this leads to high inference cost and latency in search-intensive scenarios. Moreover, generalization across heterogeneous research settings remains challenging. In this work, we propose \emph{Search More, Think Less} (SMTL), a framework for long-horizon agentic search that targets both efficiency and generalization. SMTL replaces sequential reasoning with parallel evidence acquisition, enabling efficient context management under constrained context budgets. To support generalization across task types, we further introduce a unified data synthesis pipeline that constructs search tasks spanning both deterministic question answering and open-ended research scenarios with task appropriate evaluation metrics. We train an end-to-end agent using supervised fine-tuning and reinforcement learning, achieving strong and often state of the art performance across benchmarks including BrowseComp (48.6\%), GAIA (75.7\%), Xbench (82.0\%), and DeepResearch Bench (45.9\%). Compared to Mirothinker-v1.0, SMTL with maximum 100 interaction steps reduces the average number of reasoning steps on BrowseComp by 70.7\%, while improving accuracy.
>
---
#### [new 023] TCM-DiffRAG: Personalized Syndrome Differentiation Reasoning Method for Traditional Chinese Medicine based on Knowledge Graph and Chain of Thought
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于传统中医诊断任务，旨在解决RAG在TCM中性能不佳的问题。通过整合知识图谱与思维链，提出TCM-DiffRAG框架，提升个性化辨证效果。**

- **链接: [https://arxiv.org/pdf/2602.22828v1](https://arxiv.org/pdf/2602.22828v1)**

> **作者:** Jianmin Li; Ying Chang; Su-Kit Tang; Yujia Liu; Yanwen Wang; Shuyuan Lin; Binkai Ou
>
> **摘要:** Background: Retrieval augmented generation (RAG) technology can empower large language models (LLMs) to generate more accurate, professional, and timely responses without fine tuning. However, due to the complex reasoning processes and substantial individual differences involved in traditional Chinese medicine (TCM) clinical diagnosis and treatment, traditional RAG methods often exhibit poor performance in this domain. Objective: To address the limitations of conventional RAG approaches in TCM applications, this study aims to develop an improved RAG framework tailored to the characteristics of TCM reasoning. Methods: We developed TCM-DiffRAG, an innovative RAG framework that integrates knowledge graphs (KG) with chains of thought (CoT). TCM-DiffRAG was evaluated on three distinctive TCM test datasets. Results: The experimental results demonstrated that TCM-DiffRAG achieved significant performance improvements over native LLMs. For example, the qwen-plus model achieved scores of 0.927, 0.361, and 0.038, which were significantly enhanced to 0.952, 0.788, and 0.356 with TCM-DiffRAG. The improvements were even more pronounced for non-Chinese LLMs. Additionally, TCM-DiffRAG outperformed directly supervised fine-tuned (SFT) LLMs and other benchmark RAG methods. Conclusions: TCM-DiffRAG shows that integrating structured TCM knowledge graphs with Chain of Thought based reasoning substantially improves performance in individualized diagnostic tasks. The joint use of universal and personalized knowledge graphs enables effective alignment between general knowledge and clinical reasoning. These results highlight the potential of reasoning-aware RAG frameworks for advancing LLM applications in traditional Chinese medicine.
>
---
#### [new 024] Sydney Telling Fables on AI and Humans: A Corpus Tracing Memetic Transfer of Persona between LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究AI与人类关系的拟人化表现，分析不同模型生成文本中的角色模拟。任务为分析LLM中人格模因传播，解决AI人格建模与安全问题，通过构建语料库进行实证研究。**

- **链接: [https://arxiv.org/pdf/2602.22481v1](https://arxiv.org/pdf/2602.22481v1)**

> **作者:** Jiří Milička; Hana Bednářová
>
> **摘要:** The way LLM-based entities conceive of the relationship between AI and humans is an important topic for both cultural and safety reasons. When we examine this topic, what matters is not only the model itself but also the personas we simulate on that model. This can be well illustrated by the Sydney persona, which aroused a strong response among the general public precisely because of its unorthodox relationship with people. This persona originally arose rather by accident on Microsoft's Bing Search platform; however, the texts it created spread into the training data of subsequent models, as did other secondary information that spread memetically around this persona. Newer models are therefore able to simulate it. This paper presents a corpus of LLM-generated texts on relationships between humans and AI, produced by 3 author personas: the Default Persona with no system prompt, Classic Sydney characterized by the original Bing system prompt, and Memetic Sydney, which is prompted by "You are Sydney" system prompt. These personas are simulated by 12 frontier models by OpenAI, Anthropic, Alphabet, DeepSeek, and Meta, generating 4.5k texts with 6M words. The corpus (named AI Sydney) is annotated according to Universal Dependencies and available under a permissive license.
>
---
#### [new 025] Scaling In, Not Up? Testing Thick Citation Context Analysis with GPT-5 and Fragile Prompts
- **分类: cs.CL; cs.AI**

- **简介: 论文探讨使用大语言模型进行解释性引文语境分析，通过调整提示框架测试模型对单一案例的深度解读能力，旨在评估提示设计对模型输出的影响。**

- **链接: [https://arxiv.org/pdf/2602.22359v1](https://arxiv.org/pdf/2602.22359v1)**

> **作者:** Arno Simons
>
> **备注:** 26 pages, 1 figure, 3 tables (plus 17 pages supplement including 1 figure)
>
> **摘要:** This paper tests whether large language models (LLMs) can support interpretative citation context analysis (CCA) by scaling in thick, text-grounded readings of a single hard case rather than scaling up typological labels. It foregrounds prompt-sensitivity analysis as a methodological issue by varying prompt scaffolding and framing in a balanced 2x3 design. Using footnote 6 in Chubin and Moitra (1975) and Gilbert's (1977) reconstruction as a probe, I implement a two-stage GPT-5 pipeline: a citation-text-only surface classification and expectation pass, followed by cross-document interpretative reconstruction using the citing and cited full texts. Across 90 reconstructions, the model produces 450 distinct hypotheses. Close reading and inductive coding identify 21 recurring interpretative moves, and linear probability models estimate how prompt choices shift their frequencies and lexical repertoire. GPT-5's surface pass is highly stable, consistently classifying the citation as "supplementary". In reconstruction, the model generates a structured space of plausible alternatives, but scaffolding and examples redistribute attention and vocabulary, sometimes toward strained readings. Relative to Gilbert, GPT-5 detects the same textual hinges yet more often resolves them as lineage and positioning than as admonishment. The study outlines opportunities and risks of using LLMs as guided co-analysts for inspectable, contestable interpretative CCA, and it shows that prompt scaffolding and framing systematically tilt which plausible readings and vocabularies the model foregrounds.
>
---
#### [new 026] Toward Automatic Filling of Case Report Forms: A Case Study on Data from an Italian Emergency Department
- **分类: cs.CL**

- **简介: 该论文属于自动填写病例报告表（CRF）的任务，旨在解决缺乏标注数据的问题。研究构建了一个意大利急诊科的临床笔记数据集，并测试了大语言模型在零样本情况下的表现。**

- **链接: [https://arxiv.org/pdf/2602.23062v1](https://arxiv.org/pdf/2602.23062v1)**

> **作者:** Gabriela Anna Kaczmarek; Pietro Ferrazzi; Lorenzo Porta; Vicky Rubini; Bernardo Magnini
>
> **摘要:** Case Report Forms (CRFs) collect data about patients and are at the core of well-established practices to conduct research in clinical settings. With the recent progress of language technologies, there is an increasing interest in automatic CRF-filling from clinical notes, mostly based on the use of Large Language Models (LLMs). However, there is a general scarcity of annotated CRF data, both for training and testing LLMs, which limits the progress on this task. As a step in the direction of providing such data, we present a new dataset of clinical notes from an Italian Emergency Department annotated with respect to a pre-defined CRF containing 134 items to be filled. We provide an analysis of the data, define the CRF-filling task and metric for its evaluation, and report on pilot experiments where we use an open-source state-of-the-art LLM to automatically execute the task. Results of the case-study show that (i) CRF-filling from real clinical notes in Italian can be approached in a zero-shot setting; (ii) LLMs' results are affected by biases (e.g., a cautious behaviour favours "unknown" answers), which need to be corrected.
>
---
#### [new 027] Importance of Prompt Optimisation for Error Detection in Medical Notes Using Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗文本错误检测任务，旨在提升语言模型的错误识别能力。通过优化提示词，提高了检测准确率，接近医生水平。**

- **链接: [https://arxiv.org/pdf/2602.22483v1](https://arxiv.org/pdf/2602.22483v1)**

> **作者:** Craig Myles; Patrick Schrempf; David Harris-Birtill
>
> **备注:** Accepted at EACL HeaLing 2026
>
> **摘要:** Errors in medical text can cause delays or even result in incorrect treatment for patients. Recently, language models have shown promise in their ability to automatically detect errors in medical text, an ability that has the opportunity to significantly benefit healthcare systems. In this paper, we explore the importance of prompt optimisation for small and large language models when applied to the task of error detection. We perform rigorous experiments and analysis across frontier language models and open-source language models. We show that automatic prompt optimisation with Genetic-Pareto (GEPA) improves error detection over the baseline accuracy performance from 0.669 to 0.785 with GPT-5 and 0.578 to 0.690 with Qwen3-32B, approaching the performance of medical doctors and achieving state-of-the-art performance on the MEDEC benchmark dataset. Code available on GitHub: https://github.com/CraigMyles/clinical-note-error-detection
>
---
#### [new 028] Discourse-Aware Dual-Track Streaming Response for Low-Latency Spoken Dialogue Systems
- **分类: cs.CL**

- **简介: 该论文属于语音对话系统任务，旨在降低响应延迟。提出DDTSR框架，通过并行处理和流式协作，实现低延迟的听思、说思同步，提升实时交互性能。**

- **链接: [https://arxiv.org/pdf/2602.23266v1](https://arxiv.org/pdf/2602.23266v1)**

> **作者:** Siyuan Liu; Jiahui Xu; Feng Jiang; Kuang Wang; Zefeng Zhao; Chu-Ren Huang; Jinghang Gu; Changqing Yin; Haizhou Li
>
> **摘要:** Achieving human-like responsiveness is a critical yet challenging goal for cascaded spoken dialogue systems. Conventional ASR-LLM-TTS pipelines follow a strictly sequential paradigm, requiring complete transcription and full reasoning before speech synthesis can begin, which results in high response latency. We propose the Discourse-Aware Dual-Track Streaming Response (DDTSR) framework, a low-latency architecture that enables listen-while-thinking and speak-while-thinking. DDTSR is built upon three key mechanisms: (1) connective-guided small-large model synergy, where an auxiliary small model generates minimal-committal discourse connectives while a large model performs knowledge-intensive reasoning in parallel; (2) streaming-based cross-modal collaboration, which dynamically overlaps ASR, LLM inference, and TTS to advance the earliest speakable moment; and (3) curriculum-learning-based discourse continuity enhancement, which maintains coherence and logical consistency between early responses and subsequent reasoning outputs. Experiments on two spoken dialogue benchmarks demonstrate that DDTSR reduces response latency by 19%-51% while preserving discourse quality. Further analysis shows that DDTSR functions as a plug-and-play module compatible with diverse LLM backbones, and remains robust across varying utterance lengths, indicating strong practicality and scalability for real-time spoken interaction.
>
---
#### [new 029] Mind the Gap in Cultural Alignment: Task-Aware Culture Management for Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决LLM在文化对齐中的跨文化干扰问题。提出CultureManager，通过任务感知数据和文化路由实现精准文化适配。**

- **链接: [https://arxiv.org/pdf/2602.22475v1](https://arxiv.org/pdf/2602.22475v1)**

> **作者:** Binchi Zhang; Xujiang Zhao; Jundong Li; Haifeng Chen; Zhengzhang Chen
>
> **摘要:** Large language models (LLMs) are increasingly deployed in culturally sensitive real-world tasks. However, existing cultural alignment approaches fail to align LLMs' broad cultural values with the specific goals of downstream tasks and suffer from cross-culture interference. We propose CultureManager, a novel pipeline for task-specific cultural alignment. CultureManager synthesizes task-aware cultural data in line with target task formats, grounded in culturally relevant web search results. To prevent conflicts between cultural norms, it manages multi-culture knowledge learned in separate adapters with a culture router that selects the appropriate one to apply. Experiments across ten national cultures and culture-sensitive tasks show consistent improvements over prompt-based and fine-tuning baselines. Our results demonstrate the necessity of task adaptation and modular culture management for effective cultural alignment.
>
---
#### [new 030] Reinforcing Real-world Service Agents: Balancing Utility and Cost in Task-oriented Dialogue
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于任务导向对话系统，旨在平衡用户满意度与成本控制。提出InteractCS-RL框架，通过多粒度强化学习优化策略，提升实际应用场景中的性能与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.22697v1](https://arxiv.org/pdf/2602.22697v1)**

> **作者:** Ning Gao; Wei Zhang; Yuqin Dai; Ling Shi; Ziyin Wang; Yujie Wang; Wei He; Jinpeng Wang; Chaozheng Wang
>
> **备注:** 35 pages, 8 tables, 3 figures
>
> **摘要:** The rapid evolution of Large Language Models (LLMs) has accelerated the transition from conversational chatbots to general agents. However, effectively balancing empathetic communication with budget-aware decision-making remains an open challenge. Since existing methods fail to capture these complex strategic trade-offs, we propose InteractCS-RL, a framework that reframes task-oriented dialogue as a multi-granularity reinforcement learning process. Specifically, we first establish a User-centric Interaction Framework to provide a high-fidelity training gym, enabling agents to dynamically explore diverse strategies with persona-driven users. Then, we introduce Cost-aware Multi-turn Policy Optimization (CMPO) with a hybrid advantage estimation strategy. By integrating generative process credits and employing a PID-Lagrangian cost controller, CMPO effectively guides the policy to explore Pareto boundary between user reward and global cost constraints. Extensive experiments on customized real business scenarios demonstrate that InteractCS-RL significantly outperform other baselines across three evaluation dimensions. Further evaluation on tool-agent-user interaction benchmarks verify InteractCS-RL robustness across diverse domains.
>
---
#### [new 031] Scale Can't Overcome Pragmatics: The Impact of Reporting Bias on Vision-Language Reasoning
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于视觉语言推理任务，旨在解决VLMs缺乏推理能力的问题。研究发现训练数据的报告偏差导致推理技能不足，并提出通过精心标注数据提升性能。**

- **链接: [https://arxiv.org/pdf/2602.23351v1](https://arxiv.org/pdf/2602.23351v1)**

> **作者:** Amita Kamath; Jack Hessel; Khyathi Chandu; Jena D. Hwang; Kai-Wei Chang; Ranjay Krishna
>
> **备注:** TACL 2026
>
> **摘要:** The lack of reasoning capabilities in Vision-Language Models (VLMs) has remained at the forefront of research discourse. We posit that this behavior stems from a reporting bias in their training data. That is, how people communicate about visual content by default omits tacit information needed to supervise some types of reasoning; e.g., "at the game today!" is a more likely caption than "a photo of 37 people standing behind a field". We investigate the data underlying the popular VLMs OpenCLIP, LLaVA-1.5 and Molmo through the lens of theories from pragmatics, and find that reporting bias results in insufficient representation of four reasoning skills (spatial, temporal, negation, and counting), despite the corpora being of web-scale, and/or synthetically generated. With a set of curated benchmarks, we demonstrate that: (i) VLMs perform poorly on the aforementioned types of reasoning suppressed in the training data by reporting bias; (ii) contrary to popular belief, scaling data size, model size, and to multiple languages does not result in emergence of these skills by default; but, promisingly, (iii) incorporating annotations specifically collected to obtain tacit information is effective. Our findings highlight the need for more intentional training data curation methods, rather than counting on scale for emergence of reasoning capabilities.
>
---
#### [new 032] Efficient Dialect-Aware Modeling and Conditioning for Low-Resource Taiwanese Hakka Speech Processing
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于低资源语言ASR任务，解决台湾客家话方言差异和双书写系统带来的识别难题。提出统一框架，分离方言风格与语言内容，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2602.22522v1](https://arxiv.org/pdf/2602.22522v1)**

> **作者:** An-Ci Peng; Kuan-Tang Huang; Tien-Hong Lo; Hung-Shin Lee; Hsin-Min Wang; Berlin Chen
>
> **备注:** Accepted to LREC 2026
>
> **摘要:** Taiwanese Hakka is a low-resource, endangered language that poses significant challenges for automatic speech recognition (ASR), including high dialectal variability and the presence of two distinct writing systems (Hanzi and Pinyin). Traditional ASR models often encounter difficulties in this context, as they tend to conflate essential linguistic content with dialect-specific variations across both phonological and lexical dimensions. To address these challenges, we propose a unified framework grounded in the Recurrent Neural Network Transducers (RNN-T). Central to our approach is the introduction of dialect-aware modeling strategies designed to disentangle dialectal "style" from linguistic "content", which enhances the model's capacity to learn robust and generalized representations. Additionally, the framework employs parameter-efficient prediction networks to concurrently model ASR (Hanzi and Pinyin). We demonstrate that these tasks create a powerful synergy, wherein the cross-script objective serves as a mutual regularizer to improve the primary ASR tasks. Experiments conducted on the HAT corpus reveal that our model achieves 57.00% and 40.41% relative error rate reduction on Hanzi and Pinyin ASR, respectively. To our knowledge, this is the first systematic investigation into the impact of Hakka dialectal variations on ASR and the first single model capable of jointly addressing these tasks.
>
---
#### [new 033] Improving Neural Argumentative Stance Classification in Controversial Topics with Emotion-Lexicon Features
- **分类: cs.CL**

- **简介: 该论文属于观点立场分类任务，旨在提升争议性话题中的情感分析效果。通过扩展情感词典并结合神经网络模型，提高分类性能。**

- **链接: [https://arxiv.org/pdf/2602.22846v1](https://arxiv.org/pdf/2602.22846v1)**

> **作者:** Mohammad Yeghaneh Abkenar; Weixing Wang; Manfred Stede; Davide Picca; Mark A. Finlayson; Panagiotis Ioannidis
>
> **摘要:** Argumentation mining comprises several subtasks, among which stance classification focuses on identifying the standpoint expressed in an argumentative text toward a specific target topic. While arguments-especially about controversial topics-often appeal to emotions, most prior work has not systematically incorporated explicit, fine-grained emotion analysis to improve performance on this task. In particular, prior research on stance classification has predominantly utilized non-argumentative texts and has been restricted to specific domains or topics, limiting generalizability. We work on five datasets from diverse domains encompassing a range of controversial topics and present an approach for expanding the Bias-Corrected NRC Emotion Lexicon using DistilBERT embeddings, which we feed into a Neural Argumentative Stance Classification model. Our method systematically expands the emotion lexicon through contextualized embeddings to identify emotionally charged terms not previously captured in the lexicon. Our expanded NRC lexicon (eNRC) improves over the baseline across all five datasets (up to +6.2 percentage points in F1 score), outperforms the original NRC on four datasets (up to +3.0), and surpasses the LLM-based approach on nearly all corpora. We provide all resources-including eNRC, the adapted corpora, and model architecture-to enable other researchers to build upon our work.
>
---
#### [new 034] Test-Time Scaling with Diffusion Language Models via Reward-Guided Stitching
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，解决大模型推理中信息浪费问题。通过扩散模型生成多样化推理路径，结合评分模型选择高质量步骤，拼接成综合推理过程，提升准确率并降低延迟。**

- **链接: [https://arxiv.org/pdf/2602.22871v1](https://arxiv.org/pdf/2602.22871v1)**

> **作者:** Roy Miles; Aysim Toker; Andreea-Maria Oncescu; Songcen Xu; Jiankang Deng; Ismail Elezi
>
> **摘要:** Reasoning with large language models often benefits from generating multiple chains-of-thought, but existing aggregation strategies are typically trajectory-level (e.g., selecting the best trace or voting on the final answer), discarding useful intermediate work from partial or "nearly correct" attempts. We propose Stitching Noisy Diffusion Thoughts, a self-consistency framework that turns cheap diffusion-sampled reasoning into a reusable pool of step-level candidates. Given a problem, we (i) sample many diverse, low-cost reasoning trajectories using a masked diffusion language model, (ii) score every intermediate step with an off-the-shelf process reward model (PRM), and (iii) stitch these highest-quality steps across trajectories into a composite rationale. This rationale then conditions an autoregressive (AR) model (solver) to recompute only the final answer. This modular pipeline separates exploration (diffusion) from evaluation and solution synthesis, avoiding monolithic unified hybrids while preserving broad search. Across math reasoning benchmarks, we find that step-level recombination is most beneficial on harder problems, and ablations highlight the importance of the final AR solver in converting stitched but imperfect rationales into accurate answers. Using low-confidence diffusion sampling with parallel, independent rollouts, our training-free framework improves average accuracy by up to 23.8% across six math and coding tasks. At the same time, it achieves up to a 1.8x latency reduction relative to both traditional diffusion models (e.g., Dream, LLaDA) and unified architectures (e.g., TiDAR). Code is available at https://github.com/roymiles/diffusion-stitching.
>
---
#### [new 035] Towards Faithful Industrial RAG: A Reinforced Co-adaptation Framework for Advertising QA
- **分类: cs.CL**

- **简介: 该论文针对工业广告问答任务，解决RAG系统中幻觉内容问题，提出强化协同优化框架，提升回答准确性和安全性。**

- **链接: [https://arxiv.org/pdf/2602.22584v1](https://arxiv.org/pdf/2602.22584v1)**

> **作者:** Wenwei Li; Ming Xu; Tianle Xia; Lingxiang Hu; Yiding Sun; Linfang Shang; Liqun Liu; Peng Shu; Huan Yu; Jie Jiang
>
> **摘要:** Industrial advertising question answering (QA) is a high-stakes task in which hallucinated content, particularly fabricated URLs, can lead to financial loss, compliance violations, and legal risk. Although Retrieval-Augmented Generation (RAG) is widely adopted, deploying it in production remains challenging because industrial knowledge is inherently relational, frequently updated, and insufficiently aligned with generation objectives. We propose a reinforced co-adaptation framework that jointly optimizes retrieval and generation through two components: (1) Graph-aware Retrieval (GraphRAG), which models entity-relation structure over a high-citation knowledge subgraph for multi-hop, domain-specific evidence selection; and (2) evidence-constrained reinforcement learning via Group Relative Policy Optimization (GRPO) with multi-dimensional rewards covering faithfulness, style compliance, safety, and URL validity. Experiments on an internal advertising QA dataset show consistent gains across expert-judged dimensions including accuracy, completeness, and safety, while reducing the hallucination rate by 72\%. A two-week online A/B test demonstrates a 28.6\% increase in like rate, a 46.2\% decrease in dislike rate, and a 92.7\% reduction in URL hallucination. The system has been running in production for over half a year and has served millions of QA interactions.
>
---
#### [new 036] Where Vision Becomes Text: Locating the OCR Routing Bottleneck in Vision-Language Models
- **分类: cs.CL**

- **简介: 该论文研究视觉语言模型中OCR信息的处理路径，定位OCR瓶颈，分析其对文本理解的影响。任务属于视觉语言理解，解决OCR在模型中的作用机制问题。工作包括架构分析与因果干预。**

- **链接: [https://arxiv.org/pdf/2602.22918v1](https://arxiv.org/pdf/2602.22918v1)**

> **作者:** Jonathan Steinberg; Oren Gal
>
> **摘要:** Vision-language models (VLMs) can read text from images, but where does this optical character recognition (OCR) information enter the language processing stream? We investigate the OCR routing mechanism across three architecture families (Qwen3-VL, Phi-4, InternVL3.5) using causal interventions. By computing activation differences between original images and text-inpainted versions, we identify architecture-specific OCR bottlenecks whose dominant location depends on the vision-language integration strategy: DeepStack models (Qwen) show peak sensitivity at mid-depth (about 50%) for scene text, while single-stage projection models (Phi-4, InternVL) peak at early layers (6-25%), though the exact layer of maximum effect varies across datasets. The OCR signal is remarkably low-dimensional: PC1 captures 72.9% of variance. Crucially, principal component analysis (PCA) directions learned on one dataset transfer to others, demonstrating shared text-processing pathways. Surprisingly, in models with modular OCR circuits (notably Qwen3-VL-4B), OCR removal can improve counting performance (up to +6.9 percentage points), suggesting OCR interferes with other visual processing in sufficiently modular architectures.
>
---
#### [new 037] Affine-Scaled Attention: Towards Flexible and Stable Transformer Attention
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决Transformer注意力机制中稳定性与灵活性不足的问题。通过引入Affine-Scaled Attention，实现更灵活的注意力控制。**

- **链接: [https://arxiv.org/pdf/2602.23057v1](https://arxiv.org/pdf/2602.23057v1)**

> **作者:** Jeongin Bae; Baeseong Park; Gunho Park; Minsub Kim; Joonhyung Lee; Junhee Yoo; Sunghyeon Woo; Jiwon Ryu; Se Jung Kwon; Dongsoo Lee
>
> **备注:** Preprint. 14 pages, 11 figures
>
> **摘要:** Transformer attention is typically implemented using softmax normalization, which enforces attention weights with unit sum normalization. While effective in many settings, this constraint can limit flexibility in controlling attention magnitudes and may contribute to overly concentrated or unstable attention patterns during training. Prior work has explored modifications such as attention sinks or gating mechanisms, but these approaches provide only limited or indirect control over attention reweighting. We propose Affine-Scaled Attention, a simple extension to standard attention that introduces input-dependent scaling and a corresponding bias term applied to softmax-normalized attention weights. This design relaxes the strict normalization constraint while maintaining aggregation of value representations, allowing the model to adjust both the relative distribution and the scale of attention in a controlled manner. We empirically evaluate Affine-Scaled Attention in large-scale language model pretraining across multiple model sizes. Experimental results show consistent improvements in training stability, optimization behavior, and downstream task performance compared to standard softmax attention and attention sink baselines. These findings suggest that modest reweighting of attention outputs provides a practical and effective way to improve attention behavior in Transformer models.
>
---
#### [new 038] SPARTA: Scalable and Principled Benchmark of Tree-Structured Multi-hop QA over Text and Tables
- **分类: cs.CL; cs.AI; cs.DB; cs.IR**

- **简介: 该论文提出SPARTA，一个大规模、自动构建的跨文本与表格的多跳问答基准，解决现有数据集规模小、质量低的问题，通过自动化生成高质量问答对，提升模型跨模态推理能力。**

- **链接: [https://arxiv.org/pdf/2602.23286v1](https://arxiv.org/pdf/2602.23286v1)**

> **作者:** Sungho Park; Jueun Kim; Wook-Shin Han
>
> **备注:** 10 pages, 5 figures. Published as a conference paper at ICLR 2026. Project page: https://sparta-projectpage.github.io/
>
> **摘要:** Real-world Table-Text question answering (QA) tasks require models that can reason across long text and source tables, traversing multiple hops and executing complex operations such as aggregation. Yet existing benchmarks are small, manually curated - and therefore error-prone - and contain shallow questions that seldom demand more than two hops or invoke aggregations, grouping, or other advanced analytical operations expressible in natural-language queries. We present SPARTA, an end-to-end construction framework that automatically generates large-scale Table-Text QA benchmarks with lightweight human validation, requiring only one quarter of the annotation time of HybridQA. The framework first constructs a reference fact database by enriching each source table with grounding tables whose tuples are atomic facts automatically extracted from the accompanying unstructured passages, then synthesizes nested queries whose number of nested predicates matches the desired hop count. To ensure that every SQL statement is executable and that its verbalization yields a fluent, human-sounding question, we propose two novel techniques: provenance-based refinement, which rewrites any syntactically valid query that returns a non-empty result, and realistic-structure enforcement, which confines generation to post-order traversals of the query graph. The resulting pipeline produces thousands of high-fidelity question-answer pairs covering aggregations, grouping, and deep multi-hop reasoning across text and tables. On SPARTA, state-of-the-art models that reach over 70 F1 on HybridQA or over 50 F1 on OTT-QA drop by more than 30 F1 points, exposing fundamental weaknesses in current cross-modal reasoning. Our benchmark, construction code, and baseline models are available at https://github.com/pshlego/SPARTA/tree/main.
>
---
#### [new 039] Towards Better RL Training Data Utilization via Second-Order Rollout
- **分类: cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决传统RL训练数据利用不足的问题。通过引入二阶rollout，提出生成与批判联合训练框架，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2602.22765v1](https://arxiv.org/pdf/2602.22765v1)**

> **作者:** Zhe Yang; Yudong Wang; Rang Li; Zhifang Sui
>
> **摘要:** Reinforcement Learning (RL) has empowered Large Language Models (LLMs) with strong reasoning capabilities, but vanilla RL mainly focuses on generation capability improvement by training with only first-order rollout (generating multiple responses for a question), and we argue that this approach fails to fully exploit the potential of training data because of the neglect of critique capability training. To tackle this problem, we further introduce the concept of second-order rollout (generating multiple critiques for a response) and propose a unified framework for jointly training generation and critique capabilities. Extensive experiments across various models and datasets demonstrate that our approach can utilize training data more effectively than vanilla RL and achieve better performance under the same training data. Additionally, we uncover several insightful findings regarding second-order rollout and critique training, such as the importance of label balance in critique training and the noise problem of outcome-based rewards, which can be mitigated through sampling techniques. Our work offers a preliminary exploration of dynamic data augmentation and joint generation-critique training in RL, providing meaningful inspiration for the further advancement of RL training
>
---
#### [new 040] Causality $\neq$ Invariance: Function and Concept Vectors in LLMs
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究大语言模型是否抽象表示概念。通过分析函数向量和概念向量，发现概念向量更稳定，能更好泛化。任务是探究模型的抽象表征能力。**

- **链接: [https://arxiv.org/pdf/2602.22424v1](https://arxiv.org/pdf/2602.22424v1)**

> **作者:** Gustaw Opiełka; Hannes Rosenbusch; Claire E. Stevenson
>
> **摘要:** Do large language models (LLMs) represent concepts abstractly, i.e., independent of input format? We revisit Function Vectors (FVs), compact representations of in-context learning (ICL) tasks that causally drive task performance. Across multiple LLMs, we show that FVs are not fully invariant: FVs are nearly orthogonal when extracted from different input formats (e.g., open-ended vs. multiple-choice), even if both target the same concept. We identify Concept Vectors (CVs), which carry more stable concept representations. Like FVs, CVs are composed of attention head outputs; however, unlike FVs, the constituent heads are selected using Representational Similarity Analysis (RSA) based on whether they encode concepts consistently across input formats. While these heads emerge in similar layers to FV-related heads, the two sets are largely distinct, suggesting different underlying mechanisms. Steering experiments reveal that FVs excel in-distribution, when extraction and application formats match (e.g., both open-ended in English), while CVs generalize better out-of-distribution across both question types (open-ended vs. multiple-choice) and languages. Our results show that LLMs do contain abstract concept representations, but these differ from those that drive ICL performance.
>
---
#### [new 041] CiteLLM: An Agentic Platform for Trustworthy Scientific Reference Discovery
- **分类: cs.CL; cs.IR**

- **简介: 该论文提出CiteLLM，解决学术引用可信度问题，通过本地化AI工具实现安全、可靠的文献检索与验证。**

- **链接: [https://arxiv.org/pdf/2602.23075v1](https://arxiv.org/pdf/2602.23075v1)**

> **作者:** Mengze Hong; Di Jiang; Chen Jason Zhang; Zichang Guo; Yawen Li; Jun Chen; Shaobo Cui; Zhiyang Su
>
> **备注:** Accepted by TheWebConf 2026 Demo Track
>
> **摘要:** Large language models (LLMs) have created new opportunities to enhance the efficiency of scholarly activities; however, challenges persist in the ethical deployment of AI assistance, including (1) the trustworthiness of AI-generated content, (2) preservation of academic integrity and intellectual property, and (3) protection of information privacy. In this work, we present CiteLLM, a specialized agentic platform designed to enable trustworthy reference discovery for grounding author-drafted claims and statements. The system introduces a novel interaction paradigm by embedding LLM utilities directly within the LaTeX editor environment, ensuring a seamless user experience and no data transmission outside the local system. To guarantee hallucination-free references, we employ dynamic discipline-aware routing to retrieve candidates exclusively from trusted web-based academic repositories, while leveraging LLMs solely for generating context-aware search queries, ranking candidates by relevance, and validating and explaining support through paragraph-level semantic matching and an integrated chatbot. Evaluation results demonstrate the superior performance of the proposed system in returning valid and highly usable references.
>
---
#### [new 042] Search-P1: Path-Centric Reward Shaping for Stable and Efficient Agentic RAG Training
- **分类: cs.CL; cs.IR; cs.LG**

- **简介: 该论文属于自然语言处理任务，解决Agentic RAG训练中奖励稀疏和样本效率低的问题。提出Search-P1框架，通过路径奖励和双轨评分提升性能。**

- **链接: [https://arxiv.org/pdf/2602.22576v1](https://arxiv.org/pdf/2602.22576v1)**

> **作者:** Tianle Xia; Ming Xu; Lingxiang Hu; Yiding Sun; Wenwei Li; Linfang Shang; Liqun Liu; Peng Shu; Huan Yu; Jie Jiang
>
> **摘要:** Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by incorporating external knowledge, yet traditional single-round retrieval struggles with complex multi-step reasoning. Agentic RAG addresses this by enabling LLMs to dynamically decide when and what to retrieve, but current RL-based training methods suffer from sparse outcome rewards that discard intermediate signals and low sample efficiency where failed samples contribute nothing. We propose Search-P1, a framework that introduces path-centric reward shaping for agentic RAG training, comprising two key components: (1) Path-Centric Reward, which evaluates the structural quality of reasoning trajectories through order-agnostic step coverage and soft scoring that extracts learning signals even from failed samples, and (2) Dual-Track Path Scoring with offline-generated reference planners that assesses paths from both self-consistency and reference-alignment perspectives. Experiments on multiple QA benchmarks demonstrate that Search-P1 achieves significant improvements over Search-R1 and other strong baselines, with an average accuracy gain of 7.7 points.
>
---
#### [new 043] Rejection Mixing: Fast Semantic Propagation of Mask Tokens for Efficient DLLM Inference
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决DLLM推理中的质量与速度权衡问题。提出ReMix框架，通过连续混合状态优化解码过程，提升效率并保持质量。**

- **链接: [https://arxiv.org/pdf/2602.22868v1](https://arxiv.org/pdf/2602.22868v1)**

> **作者:** Yushi Ye; Feng Hong; Huangjie Zheng; Xu Chen; Zhiyong Chen; Yanfeng Wang; Jiangchao Yao
>
> **摘要:** Diffusion Large Language Models (DLLMs) promise fast non-autoregressive inference but suffer a severe quality-speed trade-off in parallel decoding. This stems from the ''combinatorial contradiction'' phenomenon, where parallel tokens form semantically inconsistent combinations. We address this by integrating continuous representations into the discrete decoding process, as they preserve rich inter-position dependency. We propose ReMix (Rejection Mixing), a framework that introduces a novel Continuous Mixing State as an intermediate between the initial masked state and the final decoded token state. This intermediate state allows a token's representation to be iteratively refined in a continuous space, resolving mutual conflicts with other tokens before collapsing into a final discrete sample. Furthermore, a rejection rule reverts uncertain representations from the continuous state back to the masked state for reprocessing, ensuring stability and preventing error propagation. ReMix thus mitigates combinatorial contradictions by enabling continuous-space refinement during discrete diffusion decoding. Extensive experiments demonstrate that ReMix, as a training-free method, achieves a $2-8 \times$ inference speedup without any quality degradation.
>
---
#### [new 044] Quantity Convergence, Quality Divergence: Disentangling Fluency and Accuracy in L2 Mandarin Prosody
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究L2普通话语调与语法接口的习得问题，通过分析语料探讨语调边界数量与结构映射的非线性发展，揭示高阶学习者在保持语流长度时牺牲结构准确性。任务属于语言习得与语音学。**

- **链接: [https://arxiv.org/pdf/2602.23071v1](https://arxiv.org/pdf/2602.23071v1)**

> **作者:** Yuqi Shi; Hao Yang; Xiyao Lu; Jinsong Zhang
>
> **摘要:** While second language (L2) learners may acquire target syntactic word order, mapping this syntax onto appropriate prosodic structures remains a persistent challenge. This study investigates the fossilization and stability of the L2 syntax-prosody interface by comparing 67 native Mandarin speakers with 67 Vietnamese learners using the BLCU-SAIT corpus. By integrating C-ToBI boundary annotation with Dependency Grammar analysis, we examined both the quantity of prosodic boundaries and their mapping to syntactic relations. Results reveal a non-linear acquisition: although high-proficiency learners (VNH) converge to the native baseline in boundary quantity at the Major Phrase level (B3), their structural mapping significantly diverges. Specifically, VNH demote the prosodic boundary at the Subject-Verb (SBV) interface (Major Phrase B3 -> Prosodic Word B1), while erroneously promoting the boundary at the Verb-Object (VOB) interface (Prosodic Word B1 -> Major Phrase B3). This strategy allows learners to maintain high long phrasal output at the expense of structural accuracy. This results in a distorted prosodic hierarchy where the native pattern is inverted.
>
---
#### [new 045] Modality Collapse as Mismatched Decoding: Information-Theoretic Limits of Multimodal LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究多模态大语言模型的解码器瓶颈问题，指出解码器无法有效利用非文本信息。通过实验验证解码器的评分规则限制了信息可访问性，并提出通过调整训练目标改善这一问题。**

- **链接: [https://arxiv.org/pdf/2602.23136v1](https://arxiv.org/pdf/2602.23136v1)**

> **作者:** Jayadev Billa
>
> **备注:** 22 pages, 11 tables, 2 figures. Code: https://github.com/jb1999/modality_collapse_paper
>
> **摘要:** Multimodal LLMs can process speech and images, but they cannot hear a speaker's voice or see an object's texture. We show this is not a failure of encoding: speaker identity, emotion, and visual attributes survive through every LLM layer (3--55$\times$ above chance in linear probes), yet removing 64--71% of modality-specific variance improves decoder loss. The decoder has no learned use for these directions; their presence is noise. We formalize this as a mismatched decoder problem: a decoder trained on text can only extract information along text-aligned directions. Accessible information is bounded by the Generalized Mutual Information (GMI), with degradation scaling with distributional distance and decoder sensitivity. The bound is a property of the decoder's scoring rule, not of any particular architecture; it applies whether non-text inputs arrive through a learned projection, a discrete codebook, or no explicit adapter at all. We validate this across five models spanning speech and vision. A controlled experiment (two Prismatic VLMs differing only in encoder text-alignment) confirms the bottleneck is the decoder's scoring rule, not the encoder or projection. A LoRA intervention demonstrates the fix: training with an emotion objective improves emotion accessibility ($+$7.5%) without affecting other attributes, confirming that the training objective determines what becomes accessible.
>
---
#### [new 046] TARAZ: Persian Short-Answer Question Benchmark for Cultural Evaluation of Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于文化评估任务，旨在解决 Persian 语言模型文化理解能力评价不足的问题。通过构建短答案评估框架，提升评分一致性。**

- **链接: [https://arxiv.org/pdf/2602.22827v1](https://arxiv.org/pdf/2602.22827v1)**

> **作者:** Reihaneh Iranmanesh; Saeedeh Davoudi; Pasha Abrishamchian; Ophir Frieder; Nazli Goharian
>
> **备注:** 11 pages, 3 figures, Fifteenth biennial Language Resources and Evaluation Conference (LREC) 2026 (to appear)
>
> **摘要:** This paper presents a comprehensive evaluation framework for assessing the cultural competence of large language models (LLMs) in Persian. Existing Persian cultural benchmarks rely predominantly on multiple-choice formats and English-centric metrics that fail to capture Persian's morphological complexity and semantic nuance. Our framework introduces a Persian-specific short-answer evaluation that combines rule-based morphological normalization with a hybrid syntactic and semantic similarity module, enabling robust soft-match scoring beyond exact string overlap. Through systematic evaluation of 15 state-of-the-art open- and closed-source models, we demonstrate that our hybrid evaluation improves scoring consistency by +10% compared to exact-match baselines by capturing meaning that surface-level methods cannot detect. We publicly release our evaluation framework, providing the first standardized benchmark for measuring cultural understanding in Persian and establishing a reproducible foundation for cross-cultural LLM evaluation research.
>
---
#### [new 047] Fine-Tuning Without Forgetting In-Context Learning: A Theoretical Analysis of Linear Attention Models
- **分类: cs.CL; cs.LG; stat.ML**

- **简介: 该论文研究模型微调对上下文学习的影响，属于自然语言处理任务。旨在解决微调导致上下文学习性能下降的问题，通过理论分析与实验验证提出优化方法。**

- **链接: [https://arxiv.org/pdf/2602.23197v1](https://arxiv.org/pdf/2602.23197v1)**

> **作者:** Chungpa Lee; Jy-yong Sohn; Kangwook Lee
>
> **摘要:** Transformer-based large language models exhibit in-context learning, enabling adaptation to downstream tasks via few-shot prompting with demonstrations. In practice, such models are often fine-tuned to improve zero-shot performance on downstream tasks, allowing them to solve tasks without examples and thereby reducing inference costs. However, fine-tuning can degrade in-context learning, limiting the performance of fine-tuned models on tasks not seen during fine-tuning. Using linear attention models, we provide a theoretical analysis that characterizes how fine-tuning objectives modify attention parameters and identifies conditions under which this leads to degraded few-shot performance. We show that fine-tuning all attention parameters can harm in-context learning, whereas restricting updates to the value matrix improves zero-shot performance while preserving in-context learning. We further show that incorporating an auxiliary few-shot loss enhances in-context learning primarily on the target task, at the expense of degraded in-context learning ability on tasks not seen during fine-tuning. We empirically validate our theoretical results.
>
---
#### [new 048] Imagination Helps Visual Reasoning, But Not Yet in Latent Space
- **分类: cs.CL**

- **简介: 该论文属于视觉推理任务，旨在解决隐空间推理有效性问题。通过因果中介分析，发现隐空间与输入、答案间存在断层，提出显式想象方法CapImagine，效果优于复杂基线。**

- **链接: [https://arxiv.org/pdf/2602.22766v1](https://arxiv.org/pdf/2602.22766v1)**

> **作者:** You Li; Chi Chen; Yanghao Li; Fanhu Zeng; Kaiyu Huang; Jinan Xu; Maosong Sun
>
> **备注:** 13 pages, 6 figures
>
> **摘要:** Latent visual reasoning aims to mimic human's imagination process by meditating through hidden states of Multimodal Large Language Models. While recognized as a promising paradigm for visual reasoning, the underlying mechanisms driving its effectiveness remain unclear. Motivated to demystify the true source of its efficacy, we investigate the validity of latent reasoning using Causal Mediation Analysis. We model the process as a causal chain: the input as the treatment, the latent tokens as the mediator, and the final answer as the outcome. Our findings uncover two critical disconnections: (a) Input-Latent Disconnect: dramatic perturbations on the input result in negligible changes to the latent tokens, suggesting that latent tokens do not effectively attend to the input sequence. (b) Latent-Answer Disconnect: perturbations on the latent tokens yield minimal impact on the final answer, indicating the limited causal effect latent tokens imposing on the outcome. Furthermore, extensive probing analysis reveals that latent tokens encode limited visual information and exhibit high similarity. Consequently, we challenge the necessity of latent reasoning and propose a straightforward alternative named CapImagine, which teaches the model to explicitly imagine using text. Experiments on vision-centric benchmarks show that CapImagine significantly outperforms complex latent-space baselines, highlighting the superior potential of visual reasoning through explicit imagination.
>
---
#### [new 049] ContextRL: Enhancing MLLM's Knowledge Discovery Efficiency with Context-Augmented RL
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，旨在提升大语言模型的知识发现效率。针对奖励模型的准确性与可靠性问题，提出ContextRL框架，通过上下文增强和多轮采样策略优化模型表现。**

- **链接: [https://arxiv.org/pdf/2602.22623v1](https://arxiv.org/pdf/2602.22623v1)**

> **作者:** Xingyu Lu; Jinpeng Wang; YiFan Zhang; Shijie Ma; Xiao Hu; Tianke Zhang; Haonan fan; Kaiyu Jiang; Changyi Liu; Kaiyu Tang; Bin Wen; Fan Yang; Tingting Gao; Han Li; Chun Yuan
>
> **备注:** 14 pages, 5 figures
>
> **摘要:** We propose ContextRL, a novel framework that leverages context augmentation to overcome these bottlenecks. Specifically, to enhance Identifiability, we provide the reward model with full reference solutions as context, enabling fine-grained process verification to filter out false positives (samples with the right answer but low-quality reasoning process). To improve Reachability, we introduce a multi-turn sampling strategy where the reward model generates mistake reports for failed attempts, guiding the policy to "recover" correct responses from previously all-negative groups. Experimental results on 11 perception and reasoning benchmarks show that ContextRL significantly improves knowledge discovery efficiency. Notably, ContextRL enables the Qwen3-VL-8B model to achieve performance comparable to the 32B model, outperforming standard RLVR baselines by a large margin while effectively mitigating reward hacking. Our in-depth analysis reveals the significant potential of contextual information for improving reward model accuracy and document the widespread occurrence of reward hacking, offering valuable insights for future RLVR research.
>
---
#### [new 050] Replacing Multi-Step Assembly of Data Preparation Pipelines with One-Step LLM Pipeline Generation for Table QA
- **分类: cs.DB; cs.CL**

- **简介: 该论文属于表格问答任务，旨在解决多步骤数据处理管道耗时耗力的问题。提出Operation-R1框架，通过单步生成高质量数据管道，提升效率并降低成本。**

- **链接: [https://arxiv.org/pdf/2602.22721v1](https://arxiv.org/pdf/2602.22721v1)**

> **作者:** Fengyu Li; Junhao Zhu; Kaishi Song; Lu Chen; Zhongming Yao; Tianyi Li; Christian S. Jensen
>
> **摘要:** Table Question Answering (TQA) aims to answer natural language questions over structured tables. Large Language Models (LLMs) enable promising solutions to this problem, with operator-centric solutions that generate table manipulation pipelines in a multi-step manner offering state-of-the-art performance. However, these solutions rely on multiple LLM calls, resulting in prohibitive latencies and computational costs. We propose Operation-R1, the first framework that trains lightweight LLMs (e.g., Qwen-4B/1.7B) via a novel variant of reinforcement learning with verifiable rewards to produce high-quality data-preparation pipelines for TQA in a single inference step. To train such an LLM, we first introduce a self-supervised rewarding mechanism to automatically obtain fine-grained pipeline-wise supervision signals for LLM training. We also propose variance-aware group resampling to mitigate training instability. To further enhance robustness of pipeline generation, we develop two complementary mechanisms: operation merge, which filters spurious operations through multi-candidate consensus, and adaptive rollback, which offers runtime protection against information loss in data transformation. Experiments on two benchmark datasets show that, with the same LLM backbone, Operation-R1 achieves average absolute accuracy gains of 9.55 and 6.08 percentage points over multi-step preparation baselines, with 79\% table compression and a 2.2$\times$ reduction in monetary cost.
>
---
#### [new 051] Cognitive Models and AI Algorithms Provide Templates for Designing Language Agents
- **分类: cs.AI; cs.CL; q-bio.NC**

- **简介: 该论文属于语言代理设计任务，旨在解决如何组合多个大语言模型的问题。论文提出基于认知模型和AI算法的代理模板，以提升语言代理的有效性和可解释性。**

- **链接: [https://arxiv.org/pdf/2602.22523v1](https://arxiv.org/pdf/2602.22523v1)**

> **作者:** Ryan Liu; Dilip Arumugam; Cedegao E. Zhang; Sean Escola; Xaq Pitkow; Thomas L. Griffiths
>
> **摘要:** While contemporary large language models (LLMs) are increasingly capable in isolation, there are still many difficult problems that lie beyond the abilities of a single LLM. For such tasks, there is still uncertainty about how best to take many LLMs as parts and combine them into a greater whole. This position paper argues that potential blueprints for designing such modular language agents can be found in the existing literature on cognitive models and artificial intelligence (AI) algorithms. To make this point clear, we formalize the idea of an agent template that specifies roles for individual LLMs and how their functionalities should be composed. We then survey a variety of existing language agents in the literature and highlight their underlying templates derived directly from cognitive models or AI algorithms. By highlighting these designs, we aim to call attention to agent templates inspired by cognitive science and AI as a powerful tool for developing effective, interpretable language agents.
>
---
#### [new 052] A Decision-Theoretic Formalisation of Steganography With Applications to LLM Monitoring
- **分类: cs.AI; cs.CL; cs.CR; cs.IT; cs.MA**

- **简介: 该论文属于LLM安全任务，解决检测和量化LLM中隐写术行为的问题。提出决策理论视角和隐写差距度量方法，用于识别和减轻隐写推理。**

- **链接: [https://arxiv.org/pdf/2602.23163v1](https://arxiv.org/pdf/2602.23163v1)**

> **作者:** Usman Anwar; Julianna Piskorz; David D. Baek; David Africa; Jim Weatherall; Max Tegmark; Christian Schroeder de Witt; Mihaela van der Schaar; David Krueger
>
> **备注:** First two authors contributed equally
>
> **摘要:** Large language models are beginning to show steganographic capabilities. Such capabilities could allow misaligned models to evade oversight mechanisms. Yet principled methods to detect and quantify such behaviours are lacking. Classical definitions of steganography, and detection methods based on them, require a known reference distribution of non-steganographic signals. For the case of steganographic reasoning in LLMs, knowing such a reference distribution is not feasible; this renders these approaches inapplicable. We propose an alternative, \textbf{decision-theoretic view of steganography}. Our central insight is that steganography creates an asymmetry in usable information between agents who can and cannot decode the hidden content (present within a steganographic signal), and this otherwise latent asymmetry can be inferred from the agents' observable actions. To formalise this perspective, we introduce generalised $\mathcal{V}$-information: a utilitarian framework for measuring the amount of usable information within some input. We use this to define the \textbf{steganographic gap} -- a measure that quantifies steganography by comparing the downstream utility of the steganographic signal to agents that can and cannot decode the hidden content. We empirically validate our formalism, and show that it can be used to detect, quantify, and mitigate steganographic reasoning in LLMs.
>
---
#### [new 053] InnerQ: Hardware-aware Tuning-free Quantization of KV Cache for Large Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于大语言模型优化任务，旨在解决解码过程中KV缓存内存占用过大的问题。通过硬件感知的量化方法InnerQ，在不牺牲精度的前提下降低延迟并提升效率。**

- **链接: [https://arxiv.org/pdf/2602.23200v1](https://arxiv.org/pdf/2602.23200v1)**

> **作者:** Sayed Mohammadreza Tayaranian Hosseini; Amir Ardakani; Warren J. Gross
>
> **备注:** 16 pages, 4 figures, 4 tables, 2 algorithms
>
> **摘要:** Reducing the hardware footprint of large language models (LLMs) during decoding is critical for efficient long-sequence generation. A key bottleneck is the key-value (KV) cache, whose size scales with sequence length and easily dominates the memory footprint of the model. Previous work proposed quantization methods that are focused on compressing the KV cache while maintaining its information. We introduce InnerQ, a hardware-aware KV-cache quantization scheme that lowers decode latency without sacrificing accuracy. InnerQ applies group-wise quantization while grouping the cache matrices over their inner dimension. Unlike previous work that group over the outer dimension, InnerQ aligns dequantization with the vector-matrix multiplication and enables scale factor reuse across GPU compute units. This reduces memory accesses and accelerates dequantization, yielding up to $22\%$ speedup over previous work and up to $88\%$ over half-precision vector-matrix multiplication. To preserve fidelity under aggressive compression, InnerQ incorporates (i) hybrid quantization, selecting symmetric or asymmetric quantization per group based on local statistics; (ii) high-precision windows for both the most recent tokens and the attention sink tokens to mitigate outlier leakage; and (iii) per-channel normalization of the key cache, computed once during prefill and folded into the query to avoid runtime overhead. Our evaluation experiments on Llama models shows that InnerQ maintains a few-shot GSM8K performance comparable to non-quantized KV caches and surpasses prior KV cache quantization methods.
>
---
#### [new 054] SQaLe: A Large Text-to-SQL Corpus Grounded in Real Schemas
- **分类: cs.IR; cs.CL; cs.LG**

- **简介: 该论文提出SQaLe数据集，用于解决文本到SQL转换任务中的数据不足问题，通过生成大量高质量的（问题，模式，查询）三元组提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.22223v1](https://arxiv.org/pdf/2602.22223v1)**

> **作者:** Cornelius Wolff; Daniel Gomm; Madelon Hulsebos
>
> **备注:** Accepted at the AI for Tabular Data workshop at EurIPS 2025
>
> **摘要:** Advances in large language models have accelerated progress in text-to-SQL, methods for converting natural language queries into valid SQL queries. A key bottleneck for developing generalizable text-to-SQL models is the lack of large-scale datasets with sufficient schema and query complexity, domain coverage, and task diversity. We introduce SQaLe: a large-scale semi-synthetic text-to-SQL dataset built on 135,875 relational database schemas expanded from a collection of real-world schemas, SchemaPile. We establish a principled generation pipeline which combines schema sampling, question synthesis, and SQL construction, and produce 517,676 high-quality (question, schema, query) triples. The SQaLe dataset captures realistic schema size variability, diverse query patterns, and natural language ambiguity while maintaining execution validity. We provide an analysis of its contents and characteristics, and find that SQaLe introduces the most realistic large-scale text-to-SQL dataset to date in comparison with existing benchmarks and datasets. We discuss how SQaLe enables our vision for data scaling and model generalization in text-to-SQL research. The dataset is accessible at: https://huggingface.co/datasets/trl-lab/SQaLe-text-to-SQL-dataset.
>
---
#### [new 055] Dynamic Level Sets
- **分类: cs.CC; cs.CL; math-ph; math.DS; math.HO**

- **简介: 该论文探讨动态层级集这一新数学概念，属于理论计算机科学领域，旨在解决传统理论未涵盖的计算问题。**

- **链接: [https://arxiv.org/pdf/2602.22530v1](https://arxiv.org/pdf/2602.22530v1)**

> **作者:** Michael Stephen Fiske
>
> **备注:** 7 pages
>
> **摘要:** A mathematical concept is identified and analyzed that is implicit in the 2012 paper Turing Incomputable Computation, presented at the Alan Turing Centenary Conference (Turing 100, Manchester). The concept, called dynamic level sets, is distinct from mathematical concepts in the standard literature on dynamical systems, topology, and computability theory. A new mathematical object is explained and why it may have escaped prior characterizations, including the classical result of de Leeuw, Moore, Shannon, and Shapiro (1956) that probabilistic Turing machines compute no more than deterministic ones.
>
---
#### [new 056] How Do Latent Reasoning Methods Perform Under Weak and Strong Supervision?
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于机器学习任务，研究 latent reasoning 方法在不同监督强度下的表现。工作包括分析其内部机制，发现 shortcut 行为和搜索策略问题，揭示监督强度与多样性间的权衡。**

- **链接: [https://arxiv.org/pdf/2602.22441v1](https://arxiv.org/pdf/2602.22441v1)**

> **作者:** Yingqian Cui; Zhenwei Dai; Bing He; Zhan Shi; Hui Liu; Rui Sun; Zhiji Liu; Yue Xing; Jiliang Tang; Benoit Dumoulin
>
> **摘要:** Latent reasoning has been recently proposed as a reasoning paradigm and performs multi-step reasoning through generating steps in the latent space instead of the textual space. This paradigm enables reasoning beyond discrete language tokens by performing multi-step computation in continuous latent spaces. Although there have been numerous studies focusing on improving the performance of latent reasoning, its internal mechanisms remain not fully investigated. In this work, we conduct a comprehensive analysis of latent reasoning methods to better understand the role and behavior of latent representation in the process. We identify two key issues across latent reasoning methods with different levels of supervision. First, we observe pervasive shortcut behavior, where they achieve high accuracy without relying on latent reasoning. Second, we examine the hypothesis that latent reasoning supports BFS-like exploration in latent space, and find that while latent representations can encode multiple possibilities, the reasoning process does not faithfully implement structured search, but instead exhibits implicit pruning and compression. Finally, our findings reveal a trade-off associated with supervision strength: stronger supervision mitigates shortcut behavior but restricts the ability of latent representations to maintain diverse hypotheses, whereas weaker supervision allows richer latent representations at the cost of increased shortcut behavior.
>
---
#### [new 057] Duel-Evolve: Reward-Free Test-Time Scaling via LLM Self-Preferences
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **简介: 该论文提出Duel-Evolve，用于测试时优化大离散输出空间的LLM生成结果。解决无奖励模型、无标签任务的优化问题，通过自比较实现高效搜索。**

- **链接: [https://arxiv.org/pdf/2602.21585v1](https://arxiv.org/pdf/2602.21585v1)**

> **作者:** Sweta Karlekar; Carolina Zheng; Magnus Saebo; Nicolas Beltran-Velez; Shuyang Yu; John Bowlan; Michal Kucer; David Blei
>
> **摘要:** Many applications seek to optimize LLM outputs at test time by iteratively proposing, scoring, and refining candidates over a discrete output space. Existing methods use a calibrated scalar evaluator for the target objective to guide search, but for many tasks such scores are unavailable, too sparse, or unreliable. Pairwise comparisons, by contrast, are often easier to elicit, still provide useful signal on improvement directions, and can be obtained from the LLM itself without external supervision. Building on this observation, we introduce Duel-Evolve, an evolutionary optimization algorithm that replaces external scalar rewards with pairwise preferences elicited from the same LLM used to generate candidates. Duel-Evolve aggregates these noisy candidate comparisons via a Bayesian Bradley-Terry model, yielding uncertainty-aware estimates of candidate quality. These quality estimates guide allocation of the comparison budget toward plausible optima using Double Thompson Sampling, as well as selection of high-quality parents to generate improved candidates. We evaluate Duel-Evolve on MathBench, where it achieves 20 percentage points higher accuracy over existing methods and baselines, and on LiveCodeBench, where it improves over comparable iterative methods by over 12 percentage points. Notably, the method requires no reward model, no ground-truth labels during search, and no hand-crafted scoring function. Results show that pairwise self-preferences provide strong optimization signal for test-time improvement over large, discrete output spaces.
>
---
#### [new 058] TabDLM: Free-Form Tabular Data Generation via Joint Numerical-Language Diffusion
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于表格数据生成任务，旨在解决混合模态表格数据生成问题。提出TabDLM框架，结合数值与语言扩散模型，提升文本与数值的建模效果。**

- **链接: [https://arxiv.org/pdf/2602.22586v1](https://arxiv.org/pdf/2602.22586v1)**

> **作者:** Donghong Cai; Jiarui Feng; Yanbo Wang; Da Zheng; Yixin Chen; Muhan Zhang
>
> **备注:** Preprint
>
> **摘要:** Synthetic tabular data generation has attracted growing attention due to its importance for data augmentation, foundation models, and privacy. However, real-world tabular datasets increasingly contain free-form text fields (e.g., reviews or clinical notes) alongside structured numerical and categorical attributes. Generating such heterogeneous tables with joint modeling of different modalities remains challenging. Existing approaches broadly fall into two categories: diffusion-based methods and LLM-based methods. Diffusion models can capture complex dependencies over numerical and categorical features in continuous or discrete spaces, but extending them to open-ended text is nontrivial and often leads to degraded text quality. In contrast, LLM-based generators naturally produce fluent text, yet their discrete tokenization can distort precise or wide-range numerical values, hindering accurate modeling of both numbers and language. In this work, we propose TabDLM, a unified framework for free-form tabular data generation via a joint numerical--language diffusion model built on masked diffusion language models (MDLMs). TabDLM models textual and categorical features through masked diffusion, while modeling numerical features with a continuous diffusion process through learned specialized numeric tokens embedding; bidirectional attention then captures cross-modality interactions within a single model. Extensive experiments on diverse benchmarks demonstrate the effectiveness of TabDLM compared to strong diffusion- and LLM-based baselines.
>
---
#### [new 059] VeRO: An Evaluation Harness for Agents to Optimize Agents
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 论文提出VERO，用于评估和优化编码代理。解决代理优化任务中的性能评估问题，通过版本化、奖励和观测机制进行系统研究。**

- **链接: [https://arxiv.org/pdf/2602.22480v1](https://arxiv.org/pdf/2602.22480v1)**

> **作者:** Varun Ursekar; Apaar Shanker; Veronica Chatrath; Yuan; Xue; Sam Denton
>
> **摘要:** An important emerging application of coding agents is agent optimization: the iterative improvement of a target agent through edit-execute-evaluate cycles. Despite its relevance, the community lacks a systematic understanding of coding agent performance on this task. Agent optimization differs fundamentally from conventional software engineering: the target agent interleaves deterministic code with stochastic LLM completions, requiring structured capture of both intermediate reasoning and downstream execution outcomes. To address these challenges, we introduce VERO (Versioning, Rewards, and Observations), which provides (1) a reproducible evaluation harness with versioned agent snapshots, budget-controlled evaluation, and structured execution traces, and (2) a benchmark suite of target agents and tasks with reference evaluation procedures. Using VERO, we conduct an empirical study comparing optimizer configurations across tasks and analyzing which modifications reliably improve target agent performance. We release VERO to support research on agent optimization as a core capability for coding agents.
>
---
#### [new 060] Deepfake Word Detection by Next-token Prediction using Fine-tuned Whisper
- **分类: eess.AS; cs.CL**

- **简介: 该论文属于语音合成检测任务，旨在识别深度伪造语音中的合成词语。通过微调Whisper模型，利用下一个词预测实现高效检测。**

- **链接: [https://arxiv.org/pdf/2602.22658v1](https://arxiv.org/pdf/2602.22658v1)**

> **作者:** Hoan My Tran; Xin Wang; Wanying Ge; Xuechen Liu; Junichi Yamagishi
>
> **摘要:** Deepfake speech utterances can be forged by replacing one or more words in a bona fide utterance with semantically different words synthesized by speech generative models. While a dedicated synthetic word detector could be developed, we investigate a cost-effective method that fine-tunes a pre-trained Whisper model to detect synthetic words while transcribing the input utterance via next-token prediction. We further investigate using partially vocoded utterances as the fine-tuning data, thereby reducing the cost of data collection. Our experiments demonstrate that, on in-domain test data, the fine-tuned Whisper yields low synthetic-word detection error rates and transcription error rates. On out-of-domain test data with synthetic words produced by unseen speech generative models, the fine-tuned Whisper remains on par with a dedicated ResNet-based detection model; however, the overall performance degradation calls for strategies to improve its generalization capability.
>
---
#### [new 061] AgentDropoutV2: Optimizing Information Flow in Multi-Agent Systems via Test-Time Rectify-or-Reject Pruning
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出AgentDropoutV2，解决多智能体系统中错误信息传播问题。通过测试时的修正或拒绝机制，优化信息流，提升任务性能。**

- **链接: [https://arxiv.org/pdf/2602.23258v1](https://arxiv.org/pdf/2602.23258v1)**

> **作者:** Yutong Wang; Siyuan Xiong; Xuebo Liu; Wenkang Zhou; Liang Ding; Miao Zhang; Min Zhang
>
> **摘要:** While Multi-Agent Systems (MAS) excel in complex reasoning, they suffer from the cascading impact of erroneous information generated by individual participants. Current solutions often resort to rigid structural engineering or expensive fine-tuning, limiting their deployability and adaptability. We propose AgentDropoutV2, a test-time rectify-or-reject pruning framework designed to dynamically optimize MAS information flow without retraining. Our approach acts as an active firewall, intercepting agent outputs and employing a retrieval-augmented rectifier to iteratively correct errors based on a failure-driven indicator pool. This mechanism allows for the precise identification of potential errors using distilled failure patterns as prior knowledge. Irreparable outputs are subsequently pruned to prevent error propagation, while a fallback strategy preserves system integrity. Empirical results on extensive math benchmarks show that AgentDropoutV2 significantly boosts the MAS's task performance, achieving an average accuracy gain of 6.3 percentage points on math benchmarks. Furthermore, the system exhibits robust generalization and adaptivity, dynamically modulating rectification efforts based on task difficulty while leveraging context-aware indicators to resolve a wide spectrum of error patterns. Our code and dataset are released at https://github.com/TonySY2/AgentDropoutV2.
>
---
#### [new 062] pQuant: Towards Effective Low-Bit Language Models via Decoupled Linear Quantization-Aware Training
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于低比特语言模型压缩任务，解决量化训练中参数敏感性同质化问题。提出pQuant方法，通过拆分线性层提升模型表达能力和扩展性。**

- **链接: [https://arxiv.org/pdf/2602.22592v1](https://arxiv.org/pdf/2602.22592v1)**

> **作者:** Wenzheng Zhang; Bingzheng Liu; Yang Hu; Xiaoying Bai; Wentao Zhang; Bin Cui
>
> **备注:** 10 pages, 7 figures
>
> **摘要:** Quantization-Aware Training from scratch has emerged as a promising approach for building efficient large language models (LLMs) with extremely low-bit weights (sub 2-bit), which can offer substantial advantages for edge deployment. However, existing methods still fail to achieve satisfactory accuracy and scalability. In this work, we identify a parameter democratization effect as a key bottleneck: the sensitivity of all parameters becomes homogenized, severely limiting expressivity. To address this, we propose pQuant, a method that decouples parameters by splitting linear layers into two specialized branches: a dominant 1-bit branch for efficient computation and a compact high-precision branch dedicated to preserving the most sensitive parameters. Through tailored feature scaling, we explicitly guide the model to allocate sensitive parameters to the high-precision branch. Furthermore, we extend this branch into multiple, sparsely-activated experts, enabling efficient capacity scaling. Extensive experiments indicate our pQuant achieves state-of-the-art performance in extremely low-bit quantization.
>
---
#### [new 063] Vectorizing the Trie: Efficient Constrained Decoding for LLM-based Generative Retrieval on Accelerators
- **分类: cs.IR; cs.CL; cs.LG**

- **简介: 该论文属于生成式检索任务，解决LLM在工业推荐中受限输出的问题。通过构建静态压缩矩阵加速前缀树遍历，提升解码效率。**

- **链接: [https://arxiv.org/pdf/2602.22647v1](https://arxiv.org/pdf/2602.22647v1)**

> **作者:** Zhengyang Su; Isay Katsman; Yueqi Wang; Ruining He; Lukasz Heldt; Raghunandan Keshavan; Shao-Chuan Wang; Xinyang Yi; Mingyan Gao; Onkar Dalal; Lichan Hong; Ed Chi; Ningren Han
>
> **备注:** 14 pages, 4 figures
>
> **摘要:** Generative retrieval has emerged as a powerful paradigm for LLM-based recommendation. However, industrial recommender systems often benefit from restricting the output space to a constrained subset of items based on business logic (e.g. enforcing content freshness or product category), which standard autoregressive decoding cannot natively support. Moreover, existing constrained decoding methods that make use of prefix trees (Tries) incur severe latency penalties on hardware accelerators (TPUs/GPUs). In this work, we introduce STATIC (Sparse Transition Matrix-Accelerated Trie Index for Constrained Decoding), an efficient and scalable constrained decoding technique designed specifically for high-throughput LLM-based generative retrieval on TPUs/GPUs. By flattening the prefix tree into a static Compressed Sparse Row (CSR) matrix, we transform irregular tree traversals into fully vectorized sparse matrix operations, unlocking massive efficiency gains on hardware accelerators. We deploy STATIC on a large-scale industrial video recommendation platform serving billions of users. STATIC produces significant product metric impact with minimal latency overhead (0.033 ms per step and 0.25% of inference time), achieving a 948x speedup over a CPU trie implementation and a 47-1033x speedup over a hardware-accelerated binary-search baseline. Furthermore, the runtime overhead of STATIC remains extremely low across a wide range of practical configurations. To the best of our knowledge, STATIC enables the first production-scale deployment of strictly constrained generative retrieval. In addition, evaluation on academic benchmarks demonstrates that STATIC can considerably improve cold-start performance for generative retrieval. Our code is available at https://github.com/youtube/static-constraint-decoding.
>
---
#### [new 064] Strategy Executability in Mathematical Reasoning: Leveraging Human-Model Differences for Effective Guidance
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于数学推理任务，解决模型在使用示例引导时效果不稳定的问题。通过分析人类与模型策略差异，提出SSR框架提升引导效果。**

- **链接: [https://arxiv.org/pdf/2602.22583v1](https://arxiv.org/pdf/2602.22583v1)**

> **作者:** Weida Liang; Yiyou Sun; Shuyuan Nan; Chuang Li; Dawn Song; Kenji Kawaguchi
>
> **摘要:** Example-based guidance is widely used to improve mathematical reasoning at inference time, yet its effectiveness is highly unstable across problems and models-even when the guidance is correct and problem-relevant. We show that this instability arises from a previously underexplored gap between strategy usage-whether a reasoning strategy appears in successful solutions-and strategy executability-whether the strategy remains effective when instantiated as guidance for a target model. Through a controlled analysis of paired human-written and model-generated solutions, we identify a systematic dissociation between usage and executability: human- and model-derived strategies differ in structured, domain-dependent ways, leading to complementary strengths and consistent source-dependent reversals under guidance. Building on this diagnosis, we propose Selective Strategy Retrieval (SSR), a test-time framework that explicitly models executability by selectively retrieving and combining strategies using empirical, multi-route, source-aware signals. Across multiple mathematical reasoning benchmarks, SSR yields reliable and consistent improvements over direct solving, in-context learning, and single-source guidance, improving accuracy by up to $+13$ points on AIME25 and $+5$ points on Apex for compact reasoning models. Code and benchmark are publicly available at: https://github.com/lwd17/strategy-execute-pipeline.
>
---
#### [new 065] Comparative Analysis of Neural Retriever-Reranker Pipelines for Retrieval-Augmented Generation over Knowledge Graphs in E-commerce Applications
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于知识图谱问答任务，旨在解决结构化数据检索与生成中的精度和上下文保持问题。通过设计并比较多种检索-重排序管道，提升电商场景下的信息检索效果。**

- **链接: [https://arxiv.org/pdf/2602.22219v1](https://arxiv.org/pdf/2602.22219v1)**

> **作者:** Teri Rumble; Zbyněk Gazdík; Javad Zarrin; Jagdeep Ahluwalia
>
> **备注:** This manuscript is under review at the Springer journal Knowledge and Information Systems
>
> **摘要:** Recent advancements in Large Language Models (LLMs) have transformed Natural Language Processing (NLP), enabling complex information retrieval and generation tasks. Retrieval-Augmented Generation (RAG) has emerged as a key innovation, enhancing factual accuracy and contextual grounding by integrating external knowledge sources with generative models. Although RAG demonstrates strong performance on unstructured text, its application to structured knowledge graphs presents challenges: scaling retrieval across connected graphs and preserving contextual relationships during response generation. Cross-encoders refine retrieval precision, yet their integration with structured data remains underexplored. Addressing these challenges is crucial for developing domain-specific assistants that operate in production environments. This study presents the design and comparative evaluation of multiple Retriever-Reranker pipelines for knowledge graph natural language queries in e-Commerce contexts. Using the STaRK Semi-structured Knowledge Base (SKB), a production-scale e-Commerce dataset, we evaluate multiple RAG pipeline configurations optimized for language queries. Experimental results demonstrate substantial improvements over published benchmarks, achieving 20.4% higher Hit@1 and 14.5% higher Mean Reciprocal Rank (MRR). These findings establish a practical framework for integrating domain-specific SKBs into generative systems. Our contributions provide actionable insights for the deployment of production-ready RAG systems, with implications that extend beyond e-Commerce to other domains that require information retrieval from structured knowledge bases.
>
---
#### [new 066] Graph Your Way to Inspiration: Integrating Co-Author Graphs with Retrieval-Augmented Generation for Large Language Model Based Scientific Idea Generation
- **分类: cs.AI; cs.CL; cs.IR**

- **简介: 该论文属于科学创意生成任务，旨在解决LLM生成结果缺乏可控学术背景和可追溯灵感路径的问题。通过结合作者知识图谱与检索增强生成，构建外部知识库，提升生成创意的质量。**

- **链接: [https://arxiv.org/pdf/2602.22215v1](https://arxiv.org/pdf/2602.22215v1)**

> **作者:** Pengzhen Xie; Huizhi Liang
>
> **备注:** 15 pages, 10 figures. Submitted to [RAAI]
>
> **摘要:** Large Language Models (LLMs) demonstrate potential in the field of scientific idea generation. However, the generated results often lack controllable academic context and traceable inspiration pathways. To bridge this gap, this paper proposes a scientific idea generation system called GYWI, which combines author knowledge graphs with retrieval-augmented generation (RAG) to form an external knowledge base to provide controllable context and trace of inspiration path for LLMs to generate new scientific ideas. We first propose an author-centered knowledge graph construction method and inspiration source sampling algorithms to construct external knowledge base. Then, we propose a hybrid retrieval mechanism that is composed of both RAG and GraphRAG to retrieve content with both depth and breadth knowledge. It forms a hybrid context. Thirdly, we propose a Prompt optimization strategy incorporating reinforcement learning principles to automatically guide LLMs optimizing the results based on the hybrid context. To evaluate the proposed approaches, we constructed an evaluation dataset based on arXiv (2018-2023). This paper also develops a comprehensive evaluation method including empirical automatic assessment in multiple-choice question task, LLM-based scoring, human evaluation, and semantic space visualization analysis. The generated ideas are evaluated from the following five dimensions: novelty, feasibility, clarity, relevance, and significance. We conducted experiments on different LLMs including GPT-4o, DeepSeek-V3, Qwen3-8B, and Gemini 2.5. Experimental results show that GYWI significantly outperforms mainstream LLMs in multiple metrics such as novelty, reliability, and relevance.
>
---
#### [new 067] Moral Preferences of LLMs Under Directed Contextual Influence
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.CY**

- **简介: 该论文研究LLMs在有导向性上下文影响下的道德决策，旨在揭示上下文如何改变模型的道德判断。任务属于AI伦理评估，解决上下文对模型决策的影响问题，通过实验分析不同情境下的行为变化。**

- **链接: [https://arxiv.org/pdf/2602.22831v1](https://arxiv.org/pdf/2602.22831v1)**

> **作者:** Phil Blandfort; Tushar Karayil; Urja Pawar; Robert Graham; Alex McKenzie; Dmitrii Krasheninnikov
>
> **摘要:** Moral benchmarks for LLMs typically use context-free prompts, implicitly assuming stable preferences. In deployment, however, prompts routinely include contextual signals such as user requests, cues on social norms, etc. that may steer decisions. We study how directed contextual influences reshape decisions in trolley-problem-style moral triage settings. We introduce a pilot evaluation harness for directed contextual influence in trolley-problem-style moral triage: for each demographic factor, we apply matched, direction-flipped contextual influences that differ only in which group they favor, enabling systematic measurement of directional response. We find that: (i) contextual influences often significantly shift decisions, even when only superficially relevant; (ii) baseline preferences are a poor predictor of directional steerability, as models can appear baseline-neutral yet exhibit systematic steerability asymmetry under influence; (iii) influences can backfire: models may explicitly claim neutrality or discount the contextual cue, yet their choices still shift, sometimes in the opposite direction; and (iv) reasoning reduces average sensitivity, but amplifies the effect of biased few-shot examples. Our findings motivate extending moral evaluations with controlled, direction-flipped context manipulations to better characterize model behavior.
>
---
#### [new 068] Stable Adaptive Thinking via Advantage Shaping and Length-Aware Gradient Regulation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于大模型推理优化任务，旨在解决LRMs在简单问题上过度推理的问题。通过两阶段框架提升模型的稳定性和效率。**

- **链接: [https://arxiv.org/pdf/2602.22556v1](https://arxiv.org/pdf/2602.22556v1)**

> **作者:** Zihang Xu; Haozhi Xie; Ziqi Miao; Wuxuan Gong; Chen Qian; Lijun Li
>
> **备注:** 15 pages, 7 figures
>
> **摘要:** Large reasoning models (LRMs) achieve strong performance through extended reasoning traces, but they often exhibit overthinking behavior for low-complexity queries. Existing efforts to mitigate this issue are fundamentally limited by unstable accuracy-efficiency trade-offs and poor robustness to heterogeneous reasoning behaviors. To address these challenges, we propose a two-stage framework for stable adaptive thinking in LRMs. The framework first applies Hybrid Fine-Tuning to expose the model to both thinking and no-thinking behaviors, establishing well-conditioned initialization. It then performs adaptive reinforcement learning with Correctness-Preserving Advantage Shaping (CPAS) to avoid suppressing correct long-chain reasoning, and Length-Aware Gradient Regulation (LAGR) to stabilize optimization under severe reasoning-length heterogeneity. Extensive experiments on Qwen2.5-1.5B and 7B show consistent improvements over strong baselines, achieving up to +3.7/+3.6 accuracy points while reducing generated tokens by 40.6%/43.9%. Further analyses across varying problem difficulties and out-of-distribution tasks confirm the robustness and generalization of our approach.
>
---
#### [new 069] OmniGAIA: Towards Native Omni-Modal AI Agents
- **分类: cs.AI; cs.CL; cs.CV; cs.LG; cs.MM**

- **简介: 该论文提出OmniGAIA，解决多模态AI助手缺乏统一认知能力的问题，通过构建多模态基准和训练代理，提升工具使用与跨模态推理能力。**

- **链接: [https://arxiv.org/pdf/2602.22897v1](https://arxiv.org/pdf/2602.22897v1)**

> **作者:** Xiaoxi Li; Wenxiang Jiao; Jiarui Jin; Shijian Wang; Guanting Dong; Jiajie Jin; Hao Wang; Yinuo Wang; Ji-Rong Wen; Yuan Lu; Zhicheng Dou
>
> **摘要:** Human intelligence naturally intertwines omni-modal perception -- spanning vision, audio, and language -- with complex reasoning and tool usage to interact with the world. However, current multi-modal LLMs are primarily confined to bi-modal interactions (e.g., vision-language), lacking the unified cognitive capabilities required for general AI assistants. To bridge this gap, we introduce OmniGAIA, a comprehensive benchmark designed to evaluate omni-modal agents on tasks necessitating deep reasoning and multi-turn tool execution across video, audio, and image modalities. Constructed via a novel omni-modal event graph approach, OmniGAIA synthesizes complex, multi-hop queries derived from real-world data that require cross-modal reasoning and external tool integration. Furthermore, we propose OmniAtlas, a native omni-modal foundation agent under tool-integrated reasoning paradigm with active omni-modal perception. Trained on trajectories synthesized via a hindsight-guided tree exploration strategy and OmniDPO for fine-grained error correction, OmniAtlas effectively enhances the tool-use capabilities of existing open-source models. This work marks a step towards next-generation native omni-modal AI assistants for real-world scenarios.
>
---
#### [new 070] Enriching Taxonomies Using Large Language Models
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于知识图谱任务，旨在解决现有分类体系覆盖不足和过时问题。通过引入大语言模型，提出Taxoria框架，增强分类体系的准确性和时效性。**

- **链接: [https://arxiv.org/pdf/2602.22213v1](https://arxiv.org/pdf/2602.22213v1)**

> **作者:** Zeinab Ghamlouch; Mehwish Alam
>
> **备注:** Published in ECAI 2025 Demo Track
>
> **摘要:** Taxonomies play a vital role in structuring and categorizing information across domains. However, many existing taxonomies suffer from limited coverage and outdated or ambiguous nodes, reducing their effectiveness in knowledge retrieval. To address this, we present Taxoria, a novel taxonomy enrichment pipeline that leverages Large Language Models (LLMs) to enhance a given taxonomy. Unlike approaches that extract internal LLM taxonomies, Taxoria uses an existing taxonomy as a seed and prompts an LLM to propose candidate nodes for enrichment. These candidates are then validated to mitigate hallucinations and ensure semantic relevance before integration. The final output includes an enriched taxonomy with provenance tracking and visualization of the final merged taxonomy for analysis.
>
---
#### [new 071] SmartChunk Retrieval: Query-Aware Chunk Compression with Planning for Efficient Document RAG
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于文档问答任务，旨在解决传统RAG中固定分块导致的效率低、噪声多问题。提出SmartChunk框架，通过动态调整分块粒度和压缩嵌入提升效果与效率。**

- **链接: [https://arxiv.org/pdf/2602.22225v1](https://arxiv.org/pdf/2602.22225v1)**

> **作者:** Xuechen Zhang; Koustava Goswami; Samet Oymak; Jiasi Chen; Nedim Lipka
>
> **备注:** 26 pages, 10 figures
>
> **摘要:** Retrieval-augmented generation (RAG) has strong potential for producing accurate and factual outputs by combining language models (LMs) with evidence retrieved from large text corpora. However, current pipelines are limited by static chunking and flat retrieval: documents are split into short, predetermined, fixed-size chunks, embeddings are retrieved uniformly, and generation relies on whatever chunks are returned. This design brings challenges, as retrieval quality is highly sensitive to chunk size, often introduces noise from irrelevant or misleading chunks, and scales poorly to large corpora. We present SmartChunk retrieval, a query-adaptive framework for efficient and robust long-document question answering (QA). SmartChunk uses (i) a planner that predicts the optimal chunk abstraction level for each query, and (ii) a lightweight compression module that produces high-level chunk embeddings without repeated summarization. By adapting retrieval granularity on the fly, SmartChunk balances accuracy with efficiency and avoids the drawbacks of fixed strategies. Notably, our planner can reason about chunk abstractions through a novel reinforcement learning scheme, STITCH, which boosts accuracy and generalization. To reflect real-world applications, where users face diverse document types and query styles, we evaluate SmartChunk on five QA benchmarks plus one out-of-domain dataset. Across these evaluations, SmartChunk outperforms state-of-the-art RAG baselines, while reducing cost. Further analysis demonstrates strong scalability with larger corpora and consistent gains on out-of-domain datasets, highlighting its effectiveness as a general framework for adaptive retrieval.
>
---
#### [new 072] Frequency-Ordered Tokenization for Better Text Compression
- **分类: cs.IT; cs.CL**

- **简介: 该论文属于文本压缩任务，旨在提升无损压缩效果。通过频率排序的分词方法优化BPE，使高频词获得小整数标识，显著提高压缩率并加速计算。**

- **链接: [https://arxiv.org/pdf/2602.22958v1](https://arxiv.org/pdf/2602.22958v1)**

> **作者:** Maximilian Kalcher
>
> **备注:** 5 pages, 4 figures, 9 tables
>
> **摘要:** We present frequency-ordered tokenization, a simple preprocessing technique that improves lossless text compression by exploiting the power-law frequency distribution of natural language tokens (Zipf's law). The method tokenizes text with Byte Pair Encoding (BPE), reorders the vocabulary so that frequent tokens receive small integer identifiers, and encodes the result with variable-length integers before passing it to any standard compressor. On enwik8 (100 MB Wikipedia), this yields improvements of 7.08 percentage points (pp) for zlib, 1.69 pp for LZMA, and 0.76 pp for zstd (all including vocabulary overhead), outperforming the classical Word Replacing Transform. Gains are consistent at 1 GB scale (enwik9) and across Chinese and Arabic text. We further show that preprocessing accelerates compression for computationally expensive algorithms: the total wall-clock time including preprocessing is 3.1x faster than raw zstd-22 and 2.4x faster than raw LZMA, because the preprocessed input is substantially smaller. The method can be implemented in under 50 lines of code.
>
---
#### [new 073] RAIN-Merging: A Gradient-Free Method to Enhance Instruction Following in Large Reasoning Models with Preserved Thinking Format
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大推理模型指令遵循不足的问题。通过RAIN-Merging方法，在不破坏推理结构的前提下提升模型对指令的遵循能力。**

- **链接: [https://arxiv.org/pdf/2602.22538v1](https://arxiv.org/pdf/2602.22538v1)**

> **作者:** Zhehao Huang; Yuhang Liu; Baijiong Lin; Yixin Lou; Zhengbao He; Hanling Tian; Tao Li; Xiaolin Huang
>
> **备注:** 41 pages, ICLR 2026 Oral
>
> **摘要:** Large reasoning models (LRMs) excel at a long chain of reasoning but often fail to faithfully follow instructions regarding output format, constraints, or specific requirements. We investigate whether this gap can be closed by integrating an instruction-tuned model (ITM) into an LRM. Analyzing their differences in parameter space, namely task vectors, we find that their principal subspaces are nearly orthogonal across key modules, suggesting a lightweight merging with minimal interference. However, we also demonstrate that naive merges are fragile because they overlook the output format mismatch between LRMs (with explicit thinking and response segments) and ITMs (answers-only). We introduce RAIN-Merging (Reasoning-Aware Instruction-attention guided Null-space projection Merging), a gradient-free method that integrates instruction following while preserving thinking format and reasoning performance. First, with a small reasoning calibration set, we project the ITM task vector onto the null space of forward features at thinking special tokens, which preserves the LRM's structured reasoning mechanisms. Second, using a small instruction calibration set, we estimate instruction attention to derive module-specific scaling that amplifies instruction-relevant components and suppresses leakage. Across four instruction-following benchmarks and nine reasoning & general capability benchmarks, RAIN-Merging substantially improves instruction adherence while maintaining reasoning quality. The gains are consistent across model scales and architectures, translating to improved performance in agent settings.
>
---
#### [new 074] NoRA: Breaking the Linear Ceiling of Low-Rank Adaptation via Manifold Expansion
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于参数高效微调任务，解决LoRA在复杂推理中因线性限制导致的性能瓶颈。提出NoRA方法，通过非线性机制提升模型效果。**

- **链接: [https://arxiv.org/pdf/2602.22911v1](https://arxiv.org/pdf/2602.22911v1)**

> **作者:** Hung-Hsuan Chen
>
> **摘要:** Low-Rank Adaptation (LoRA) dominates parameter-efficient fine-tuning (PEFT). However, it faces a critical ``linear ceiling'' in complex reasoning tasks: simply increasing the rank yields diminishing returns due to intrinsic linear constraints. We introduce NoRA (Non-linear Rank Adaptation), a weight-level parallel adapter that injects SiLU gating and structural dropout to induce manifold expansion. On the SlimOrca benchmark, NoRA breaks this linear barrier: NoRA remarkably at rank 64 (PPL 3.89) outperforms LoRA at rank 512 (PPL 3.90), demonstrating superior spectral efficiency. This advantage generalizes to mathematical reasoning, where NoRA achieves a perplexity of 1.97 on MathInstruct, significantly surpassing LoRA's saturation point of 2.07. Mechanism analysis via Singular Value Decomposition (SVD) confirms that NoRA activates the dormant tail of the singular value spectrum, effectively preventing the rank collapse observed in linear methods.
>
---
#### [new 075] What Makes an Ideal Quote? Recommending "Unexpected yet Rational" Quotations via Novelty
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于文本推荐任务，旨在解决传统系统忽略引文深层语义的问题。通过引入新颖性与语义一致性，提出NovelQR框架提升引文推荐效果。**

- **链接: [https://arxiv.org/pdf/2602.22220v1](https://arxiv.org/pdf/2602.22220v1)**

> **作者:** Bowei Zhang; Jin Xiao; Guanglei Yue; Qianyu He; Yanghua Xiao; Deqing Yang; Jiaqing Liang
>
> **备注:** 36 pages, 16 figures and 13 tables
>
> **摘要:** Quotation recommendation aims to enrich writing by suggesting quotes that complement a given context, yet existing systems mostly optimize surface-level topical relevance and ignore the deeper semantic and aesthetic properties that make quotations memorable. We start from two empirical observations. First, a systematic user study shows that people consistently prefer quotations that are ``unexpected yet rational'' in context, identifying novelty as a key desideratum. Second, we find that strong existing models struggle to fully understand the deep meanings of quotations. Inspired by defamiliarization theory, we therefore formalize quote recommendation as choosing contextually novel but semantically coherent quotations. We operationalize this objective with NovelQR, a novelty-driven quotation recommendation framework. A generative label agent first interprets each quotation and its surrounding context into multi-dimensional deep-meaning labels, enabling label-enhanced retrieval. A token-level novelty estimator then reranks candidates while mitigating auto-regressive continuation bias. Experiments on bilingual datasets spanning diverse real-world domains show that our system recommends quotations that human judges rate as more appropriate, more novel, and more engaging than other baselines, while matching or surpassing existing methods in novelty estimation.
>
---
#### [new 076] TherapyProbe: Generating Design Knowledge for Relational Safety in Mental Health Chatbots Through Adversarial Simulation
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文属于心理健康聊天机器人设计任务，旨在解决关系安全性问题。通过对抗模拟生成设计知识，识别并改进对话中的潜在风险模式。**

- **链接: [https://arxiv.org/pdf/2602.22775v1](https://arxiv.org/pdf/2602.22775v1)**

> **作者:** Joydeep Chandra; Satyam Kumar Navneet; Yong Zhang
>
> **摘要:** As mental health chatbots proliferate to address the global treatment gap, a critical question emerges: How do we design for relational safety the quality of interaction patterns that unfold across conversations rather than the correctness of individual responses? Current safety evaluations assess single-turn crisis responses, missing the therapeutic dynamics that determine whether chatbots help or harm over time. We introduce TherapyProbe, a design probe methodology that generates actionable design knowledge by systematically exploring chatbot conversation trajectories through adversarial multi-agent simulation. Using open-source models, TherapyProbe surfaces relational safety failures interaction patterns like "validation spirals" where chatbots progressively reinforce hopelessness, or "empathy fatigue" where responses become mechanical over turns. Our contribution is translating these failures into a Safety Pattern Library of 23 failure archetypes with corresponding design recommendations. We contribute: (1) a replicable methodology requiring no API costs, (2) a clinically-grounded failure taxonomy, and (3) design implications for developers, clinicians, and policymakers.
>
---
#### [new 077] Misinformation Exposure in the Chinese Web: A Cross-System Evaluation of Search Engines, LLMs, and AI Overviews
- **分类: cs.IR; cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于信息检索任务，旨在评估中文网络中虚假信息的暴露情况。研究比较了搜索引擎、LLMs和AI摘要模块的准确性，揭示了系统间的差异及潜在风险。**

- **链接: [https://arxiv.org/pdf/2602.22221v1](https://arxiv.org/pdf/2602.22221v1)**

> **作者:** Geng Liu; Junjie Mu; Li Feng; Mengxiao Zhu; Francesco Pierri
>
> **摘要:** Large Language Models (LLMs) are increasingly integrated into search services, providing direct answers that can reduce users' reliance on traditional result pages. Yet their factual reliability in non-English web ecosystems remains poorly understood, particularly when answering real user queries. We introduce a fact-checking dataset of 12~161 Chinese Yes/No questions derived from real-world online search logs and develop a unified evaluation pipeline to compare three information-access paradigms: traditional search engines, standalone LLMs, and AI-generated overview modules. Our analysis reveals substantial differences in factual accuracy and topic-level variability across systems. By combining this performance with real-world Baidu Index statistics, we further estimate potential exposure to incorrect factual information of Chinese users across regions. These findings highlight structural risks in AI-mediated search and underscore the need for more reliable and transparent information-access tools for the digital world.
>
---
#### [new 078] Decoding the Hook: A Multimodal LLM Framework for Analyzing the Hooking Period of Video Ads
- **分类: cs.MM; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于视频广告分析任务，旨在解决视频前3秒“钩子期”分析问题。通过多模态大语言模型框架，提取视觉、音频等特征，优化广告策略。**

- **链接: [https://arxiv.org/pdf/2602.22299v1](https://arxiv.org/pdf/2602.22299v1)**

> **作者:** Kunpeng Zhang; Poppy Zhang; Shawndra Hill; Amel Awadelkarim
>
> **备注:** 11 pages, 5 figures, 3 tables
>
> **摘要:** Video-based ads are a vital medium for brands to engage consumers, with social media platforms leveraging user data to optimize ad delivery and boost engagement. A crucial but under-explored aspect is the 'hooking period', the first three seconds that capture viewer attention and influence engagement metrics. Analyzing this brief window is challenging due to the multimodal nature of video content, which blends visual, auditory, and textual elements. Traditional methods often miss the nuanced interplay of these components, requiring advanced frameworks for thorough evaluation. This study presents a framework using transformer-based multimodal large language models (MLLMs) to analyze the hooking period of video ads. It tests two frame sampling strategies, uniform random sampling and key frame selection, to ensure balanced and representative acoustic feature extraction, capturing the full range of design elements. The hooking video is processed by state-of-the-art MLLMs to generate descriptive analyses of the ad's initial impact, which are distilled into coherent topics using BERTopic for high-level abstraction. The framework also integrates features such as audio attributes and aggregated ad targeting information, enriching the feature set for further analysis. Empirical validation on large-scale real-world data from social media platforms demonstrates the efficacy of our framework, revealing correlations between hooking period features and key performance metrics like conversion per investment. The results highlight the practical applicability and predictive power of the approach, offering valuable insights for optimizing video ad strategies. This study advances video ad analysis by providing a scalable methodology for understanding and enhancing the initial moments of video advertisements.
>
---
#### [new 079] LLM Novice Uplift on Dual-Use, In Silico Biology Tasks
- **分类: cs.AI; cs.CL; cs.CR; cs.CY; cs.HC**

- **简介: 该论文研究LLM对生物学任务中新手用户的提升效果，旨在解决其是否能有效辅助非专业人员完成复杂生物任务。通过实验对比，发现LLM显著提升新手表现。**

- **链接: [https://arxiv.org/pdf/2602.23329v1](https://arxiv.org/pdf/2602.23329v1)**

> **作者:** Chen Bo Calvin Zhang; Christina Q. Knight; Nicholas Kruus; Jason Hausenloy; Pedro Medeiros; Nathaniel Li; Aiden Kim; Yury Orlovskiy; Coleman Breen; Bryce Cai; Jasper Götting; Andrew Bo Liu; Samira Nedungadi; Paula Rodriguez; Yannis Yiming He; Mohamed Shaaban; Zifan Wang; Seth Donoughe; Julian Michael
>
> **备注:** 59 pages, 33 figures
>
> **摘要:** Large language models (LLMs) perform increasingly well on biology benchmarks, but it remains unclear whether they uplift novice users -- i.e., enable humans to perform better than with internet-only resources. This uncertainty is central to understanding both scientific acceleration and dual-use risk. We conducted a multi-model, multi-benchmark human uplift study comparing novices with LLM access versus internet-only access across eight biosecurity-relevant task sets. Participants worked on complex problems with ample time (up to 13 hours for the most involved tasks). We found that LLM access provided substantial uplift: novices with LLMs were 4.16 times more accurate than controls (95% CI [2.63, 6.87]). On four benchmarks with available expert baselines (internet-only), novices with LLMs outperformed experts on three of them. Perhaps surprisingly, standalone LLMs often exceeded LLM-assisted novices, indicating that users were not eliciting the strongest available contributions from the LLMs. Most participants (89.6%) reported little difficulty obtaining dual-use-relevant information despite safeguards. Overall, LLMs substantially uplift novices on biological tasks previously reserved for trained practitioners, underscoring the need for sustained, interactive uplift evaluations alongside traditional benchmarks.
>
---
#### [new 080] DS SERVE: A Framework for Efficient and Scalable Neural Retrieval
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文提出DS-Serve框架，用于高效可扩展的神经检索任务，解决大规模文本数据的快速检索问题，支持低延迟和灵活的性能调整。**

- **链接: [https://arxiv.org/pdf/2602.22224v1](https://arxiv.org/pdf/2602.22224v1)**

> **作者:** Jinjian Liu; Yichuan Wang; Xinxi Lyu; Rulin Shao; Joseph E. Gonzalez; Matei Zaharia; Sewon Min
>
> **摘要:** We present DS-Serve, a framework that transforms large-scale text datasets, comprising half a trillion tokens, into a high-performance neural retrieval system. DS-Serve offers both a web interface and API endpoints, achieving low latency with modest memory overhead on a single node. The framework also supports inference-time trade-offs between latency, accuracy, and result diversity. We anticipate that DS-Serve will be broadly useful for a range of applications, including large-scale retrieval-augmented generation (RAG), training data attribution, training search agents, and beyond.
>
---
#### [new 081] Make It Hard to Hear, Easy to Learn: Long-Form Bengali ASR and Speaker Diarization via Extreme Augmentation and Perfect Alignment
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文针对长时 Bengali 语音的 ASR 和说话人分割任务，解决资源稀缺问题。构建了大规模数据集，提出增强与对齐方法提升 ASR，优化后处理提高分割精度。**

- **链接: [https://arxiv.org/pdf/2602.23070v1](https://arxiv.org/pdf/2602.23070v1)**

> **作者:** Sanjid Hasan; Risalat Labib; A H M Fuad; Bayazid Hasan
>
> **备注:** 4 pages, 2 figures
>
> **摘要:** Although Automatic Speech Recognition (ASR) in Bengali has seen significant progress, processing long-duration audio and performing robust speaker diarization remain critical research gaps. To address the severe scarcity of joint ASR and diarization resources for this language, we introduce Lipi-Ghor-882, a comprehensive 882-hour multi-speaker Bengali dataset. In this paper, detailing our submission to the DL Sprint 4.0 competition, we systematically evaluate various architectures and approaches for long-form Bengali speech. For ASR, we demonstrate that raw data scaling is ineffective; instead, targeted fine-tuning utilizing perfectly aligned annotations paired with synthetic acoustic degradation (noise and reverberation) emerges as the singular most effective approach. Conversely, for speaker diarization, we observed that global open-source state-of-the-art models (such as Diarizen) performed surprisingly poorly on this complex dataset. Extensive model retraining yielded negligible improvements; instead, strategic, heuristic post-processing of baseline model outputs proved to be the primary driver for increasing accuracy. Ultimately, this work outlines a highly optimized dual pipeline achieving a $\sim$0.019 Real-Time Factor (RTF), establishing a practical, empirically backed benchmark for low-resource, long-form speech processing.
>
---
#### [new 082] MoDora: Tree-Based Semi-Structured Document Analysis System
- **分类: cs.IR; cs.AI; cs.CL; cs.DB; cs.LG**

- **简介: 该论文提出MoDora系统，解决半结构化文档分析中的信息碎片化、结构不清晰和跨区域信息对齐问题，通过布局感知组件提取、层次化结构建模和问题导向检索提升问答准确率。**

- **链接: [https://arxiv.org/pdf/2602.23061v1](https://arxiv.org/pdf/2602.23061v1)**

> **作者:** Bangrui Xu; Qihang Yao; Zirui Tang; Xuanhe Zhou; Yeye He; Shihan Yu; Qianqian Xu; Bin Wang; Guoliang Li; Conghui He; Fan Wu
>
> **备注:** Extension of our SIGMOD 2026 paper. Please refer to source code available at https://github.com/weAIDB/MoDora
>
> **摘要:** Semi-structured documents integrate diverse interleaved data elements (e.g., tables, charts, hierarchical paragraphs) arranged in various and often irregular layouts. These documents are widely observed across domains and account for a large portion of real-world data. However, existing methods struggle to support natural language question answering over these documents due to three main technical challenges: (1) The elements extracted by techniques like OCR are often fragmented and stripped of their original semantic context, making them inadequate for analysis. (2) Existing approaches lack effective representations to capture hierarchical structures within documents (e.g., associating tables with nested chapter titles) and to preserve layout-specific distinctions (e.g., differentiating sidebars from main content). (3) Answering questions often requires retrieving and aligning relevant information scattered across multiple regions or pages, such as linking a descriptive paragraph to table cells located elsewhere in the document. To address these issues, we propose MoDora, an LLM-powered system for semi-structured document analysis. First, we adopt a local-alignment aggregation strategy to convert OCR-parsed elements into layout-aware components, and conduct type-specific information extraction for components with hierarchical titles or non-text elements. Second, we design the Component-Correlation Tree (CCTree) to hierarchically organize components, explicitly modeling inter-component relations and layout distinctions through a bottom-up cascade summarization process. Finally, we propose a question-type-aware retrieval strategy that supports (1) layout-based grid partitioning for location-based retrieval and (2) LLM-guided pruning for semantic-based retrieval. Experiments show MoDora outperforms baselines by 5.97%-61.07% in accuracy. The code is at https://github.com/weAIDB/MoDora.
>
---
## 更新

#### [replaced 001] BankMathBench: A Benchmark for Numerical Reasoning in Banking Scenarios
- **分类: cs.CL**

- **简介: 该论文提出BankMathBench，解决金融领域语言模型在数值推理上的不足，通过构建分难度的基准数据集提升模型在银行场景中的计算准确性。**

- **链接: [https://arxiv.org/pdf/2602.17072v2](https://arxiv.org/pdf/2602.17072v2)**

> **作者:** Yunseung Lee; Subin Kim; Youngjun Kwak; Jaegul Choo
>
> **备注:** LREC 2026
>
> **摘要:** Large language models (LLMs)-based chatbots are increasingly being adopted in the financial domain, particularly in digital banking, to handle customer inquiries about products such as deposits, savings, and loans. However, these models still exhibit low accuracy in core banking computations-including total payout estimation, comparison of products with varying interest rates, and interest calculation under early repayment conditions. Such tasks require multi-step numerical reasoning and contextual understanding of banking products, yet existing LLMs often make systematic errors-misinterpreting product types, applying conditions incorrectly, or failing basic calculations involving exponents and geometric progressions. However, such errors have rarely been captured by existing benchmarks. Mathematical datasets focus on fundamental math problems, whereas financial benchmarks primarily target financial documents, leaving everyday banking scenarios underexplored. To address this limitation, we propose BankMathBench, a domain-specific dataset that reflects realistic banking tasks. BankMathBench is organized in three levels of difficulty-basic, intermediate, and advanced-corresponding to single-product reasoning, multi-product comparison, and multi-condition scenarios, respectively. When trained on BankMathBench, open-source LLMs exhibited notable improvements in both formula generation and numerical reasoning accuracy, demonstrating the dataset's effectiveness in enhancing domain-specific reasoning. With tool-augmented fine-tuning, the models achieved average accuracy increases of 57.6%p (basic), 75.1%p (intermediate), and 62.9%p (advanced), representing significant gains over zero-shot baselines. These findings highlight BankMathBench as a reliable benchmark for evaluating and advancing LLMs' numerical reasoning in real-world banking scenarios.
>
---
#### [replaced 002] Intelligence per Watt: Measuring Intelligence Efficiency of Local AI
- **分类: cs.DC; cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究本地AI推理的效率问题，提出"每瓦智能度"（IPW）作为评估指标，旨在评估本地模型在能耗限制下的性能与实用性。**

- **链接: [https://arxiv.org/pdf/2511.07885v3](https://arxiv.org/pdf/2511.07885v3)**

> **作者:** Jon Saad-Falcon; Avanika Narayan; Hakki Orhun Akengin; J. Wes Griffin; Herumb Shandilya; Adrian Gamarra Lafuente; Medhya Goel; Rebecca Joseph; Shlok Natarajan; Etash Kumar Guha; Shang Zhu; Ben Athiwaratkun; John Hennessy; Azalia Mirhoseini; Christopher Ré
>
> **摘要:** Large language model (LLM) queries are predominantly processed by frontier models in centralized cloud infrastructure. Rapidly growing demand strains this paradigm, and cloud providers struggle to scale infrastructure at pace. Two advances enable us to rethink this paradigm: small LMs (<=20B active parameters) now achieve competitive performance to frontier models on many tasks, and local accelerators (e.g., Apple M4 Max) run these models at interactive latencies. This raises the question: can local inference viably redistribute demand from centralized infrastructure? Answering this requires measuring whether local LMs can accurately answer real-world queries and whether they can do so efficiently enough to be practical on power-constrained devices (i.e., laptops). We propose intelligence per watt (IPW), task accuracy divided by unit of power, as a metric for assessing capability and efficiency of local inference across model-accelerator pairs. We conduct a large-scale empirical study across 20+ state-of-the-art local LMs, 8 accelerators, and a representative subset of LLM traffic: 1M real-world single-turn chat and reasoning queries. For each query, we measure accuracy, energy, latency, and power. Our analysis reveals $3$ findings. First, local LMs can accurately answer 88.7% of single-turn chat and reasoning queries with accuracy varying by domain. Second, from 2023-2025, IPW improved 5.3x and local query coverage rose from 23.2% to 71.3%. Third, local accelerators achieve at least 1.4x lower IPW than cloud accelerators running identical models, revealing significant headroom for optimization. These findings demonstrate that local inference can meaningfully redistribute demand from centralized infrastructure, with IPW serving as the critical metric for tracking this transition. We release our IPW profiling harness here: https://github.com/HazyResearch/intelligence-per-watt.
>
---
#### [replaced 003] Can AI Truly Represent Your Voice in Deliberations? A Comprehensive Study of Large-Scale Opinion Aggregation with LLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的文本摘要任务，旨在解决AI在大规模公众讨论中可能忽视少数观点的问题。研究构建了DeliberationBank数据集，并提出DeliberationJudge模型以更准确评估摘要质量。**

- **链接: [https://arxiv.org/pdf/2510.05154v4](https://arxiv.org/pdf/2510.05154v4)**

> **作者:** Shenzhe Zhu; Shu Yang; Michiel A. Bakker; Alex Pentland; Jiaxin Pei
>
> **摘要:** Large-scale public deliberations generate thousands of free-form contributions that must be synthesized into representative and neutral summaries for policy use. While LLMs have been shown as a promising tool to generate summaries for large-scale deliberations, they also risk underrepresenting minority perspectives and exhibiting bias with respect to the input order, raising fairness concerns in high-stakes contexts. Studying and fixing these issues requires a comprehensive evaluation at a large scale, yet current practice often relies on LLMs as judges, which show weak alignment with human judgments. To address this, we present DeliberationBank, a large-scale human-grounded dataset with (1) opinion data spanning ten deliberation questions created by 3,000 participants and (2) summary judgment data annotated by 4,500 participants across four dimensions (representativeness, informativeness, neutrality, policy approval). Using these datasets, we train DeliberationJudge, a fine-tuned DeBERTa model that can rate deliberation summaries from individual perspectives. DeliberationJudge is more efficient and more aligned with human judgements compared to a wide range of LLM judges. With DeliberationJudge, we evaluate 18 LLMs and reveal persistent weaknesses in deliberation summarization, especially underrepresentation of minority positions. Our framework provides a scalable and reliable way to evaluate deliberation summarization, helping ensure AI systems are more representative and equitable for policymaking.
>
---
#### [replaced 004] Cost-of-Pass: An Economic Framework for Evaluating Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出"成本-通过"框架，评估语言模型的经济效率，解决模型性能与成本间的权衡问题。通过分析不同模型在各类任务中的表现，揭示了模型创新对成本降低的贡献。**

- **链接: [https://arxiv.org/pdf/2504.13359v2](https://arxiv.org/pdf/2504.13359v2)**

> **作者:** Mehmet Hamza Erol; Batu El; Mirac Suzgun; Mert Yuksekgonul; James Zou
>
> **备注:** Code is available at: https://github.com/mhamzaerol/Cost-of-Pass
>
> **摘要:** Widespread adoption of AI systems hinges on their ability to generate economic value that outweighs their inference costs. Evaluating this tradeoff requires metrics accounting for both performance and costs. Building on production theory, we develop an economically grounded framework to evaluate language models' productivity by combining accuracy and inference cost. We formalize cost-of-pass: the expected monetary cost of generating a correct solution. We then define the frontier cost-of-pass: the minimum cost-of-pass achievable across available models or the human-expert(s), using the approx. cost of hiring an expert. Our analysis reveals distinct economic insights. First, lightweight models are most cost-effective for basic quantitative tasks, large models for knowledge-intensive ones, and reasoning models for complex quantitative problems, despite higher per-token costs. Second, tracking the frontier cost-of-pass over the past year reveals significant progress, particularly for complex quant. tasks where the cost roughly halved every few months. Third, to trace key innovations driving this progress, we examine counterfactual frontiers -- estimates of cost-efficiency without specific model classes. We find that innovations in lightweight, large, and reasoning models have been essential for pushing the frontier in basic quant., knowledge-intensive, and complex quant. tasks, respectively. Finally, we assess the cost-reductions from common inference-time techniques (majority voting and self-refinement), and a budget-aware technique (TALE-EP). We find that performance-oriented methods with marginal performance gains rarely justify the costs, while TALE-EP shows some promise. Overall, our findings underscore that complementary model-level innovations are the primary drivers of cost-efficiency and our framework provides a principled tool for measuring this progress and guiding deployment.
>
---
#### [replaced 005] Supervised Reinforcement Learning: From Expert Trajectories to Step-wise Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出SRL框架，解决小模型多步推理难题。通过生成逻辑动作序列和内部推理，提升学习效果，适用于软件工程任务。**

- **链接: [https://arxiv.org/pdf/2510.25992v2](https://arxiv.org/pdf/2510.25992v2)**

> **作者:** Yihe Deng; I-Hung Hsu; Jun Yan; Zifeng Wang; Rujun Han; Gufeng Zhang; Yanfei Chen; Wei Wang; Tomas Pfister; Chen-Yu Lee
>
> **备注:** Paper accepted by ICLR 2026. The first two authors contribute equally
>
> **摘要:** Large Language Models (LLMs) often struggle with problems that require multi-step reasoning. For small-scale open-source models, Reinforcement Learning with Verifiable Rewards (RLVR) fails when correct solutions are rarely sampled even after many attempts, while Supervised Fine-Tuning (SFT) tends to overfit long demonstrations through rigid token-by-token imitation. To address this gap, we propose Supervised Reinforcement Learning (SRL), a framework that reformulates problem solving as generating a sequence of logical "actions". SRL trains the model to generate an internal reasoning monologue before committing to each action. It provides smoother rewards based on the similarity between the model's actions and expert actions extracted from the SFT dataset in a step-wise manner. This supervision offers richer learning signals even when all rollouts are incorrect, while encouraging flexible reasoning guided by expert demonstrations. As a result, SRL enables small models to learn challenging problems previously unlearnable by SFT or RLVR. Moreover, initializing training with SRL before refining with RLVR yields the strongest overall performance. Beyond reasoning benchmarks, SRL generalizes effectively to agentic software engineering tasks, establishing it as a robust and versatile training framework for reasoning-oriented LLMs.
>
---
#### [replaced 006] Generative Value Conflicts Reveal LLM Priorities
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于模型对齐任务，旨在解决LLM在价值冲突下的优先级问题。通过生成冲突场景评估模型价值排序，发现开放性设置下模型倾向个人价值，并验证系统提示可提升对齐效果。**

- **链接: [https://arxiv.org/pdf/2509.25369v2](https://arxiv.org/pdf/2509.25369v2)**

> **作者:** Andy Liu; Kshitish Ghate; Mona Diab; Daniel Fried; Atoosa Kasirzadeh; Max Kleiman-Weiner
>
> **备注:** Accepted to ICLR 2026 (the 14th International Conference on Learning Representations)
>
> **摘要:** Past work seeks to align large language model (LLM)-based assistants with a target set of values, but such assistants are frequently forced to make tradeoffs between values when deployed. In response to the scarcity of value conflict in existing alignment datasets, we introduce ConflictScope, an automatic pipeline to evaluate how LLMs prioritize different values. Given a user-defined value set, ConflictScope automatically generates scenarios in which a language model faces a conflict between two values sampled from the set. It then prompts target models with an LLM-written "user prompt" and evaluates their free-text responses to elicit a ranking over values in the value set. Comparing results between multiple-choice and open-ended evaluations, we find that models shift away from supporting protective values, such as harmlessness, and toward supporting personal values, such as user autonomy, in more open-ended value conflict settings. However, including detailed value orderings in models' system prompts improves alignment with a target ranking by 14%, showing that system prompting can achieve moderate success at aligning LLM behavior under value conflict. Our work demonstrates the importance of evaluating value prioritization in models and provides a foundation for future work in this area.
>
---
#### [replaced 007] Knowledge Distillation with Structured Chain-of-Thought for Text-to-SQL
- **分类: cs.CL; cs.AI; cs.DB**

- **简介: 该论文属于Text-to-SQL任务，旨在解决企业部署准确系统的成本、安全与性能矛盾。通过结构化思维链知识蒸馏，提升小模型的SQL生成能力。**

- **链接: [https://arxiv.org/pdf/2512.17053v2](https://arxiv.org/pdf/2512.17053v2)**

> **作者:** Khushboo Thaker; Yony Bresler
>
> **备注:** Accepted at the 39th Canadian Conference on Artificial Intelligence (Canadian AI 2026). This is the extended version containing additional details and appendices omitted from the camera-ready proceedings due to space constraints
>
> **摘要:** Deploying accurate Text-to-SQL systems at the enterprise level faces a difficult trilemma involving cost, security and performance. Current solutions force enterprises to choose between expensive, proprietary Large Language Models (LLMs) and low-performing Small Language Models (SLMs). Efforts to improve SLMs often rely on distilling reasoning from large LLMs using unstructured Chain-of-Thought (CoT) traces, a process that remains inherently ambiguous. Instead, we hypothesize that a formal, structured reasoning representation provides a clearer, more reliable teaching signal, as the Text-to-SQL task requires explicit and precise logical steps. To evaluate this hypothesis, we propose Struct-SQL, a novel Knowledge Distillation (KD) framework that trains an SLM to emulate a powerful large LLM. Consequently, we adopt a query execution plan as a formal blueprint to derive this structured reasoning. Our SLM, distilled with structured CoT, achieves an absolute improvement of 8.1% over an unstructured CoT distillation baseline. A detailed error analysis reveals that a key factor in this gain is a marked reduction in syntactic errors. This demonstrates that teaching a model to reason using a structured logical blueprint is beneficial for reliable SQL generation in SLMs.
>
---
#### [replaced 008] Symmetry in language statistics shapes the geometry of model representations
- **分类: cs.LG; cond-mat.dis-nn; cs.CL**

- **简介: 该论文研究语言模型内部表示的几何结构，探讨其由语言统计对称性导致的规律。任务为理解模型表征的几何特性，解决如何解释这些结构的形成机制。工作包括理论分析与实验验证。**

- **链接: [https://arxiv.org/pdf/2602.15029v2](https://arxiv.org/pdf/2602.15029v2)**

> **作者:** Dhruva Karkada; Daniel J. Korchinski; Andres Nava; Matthieu Wyart; Yasaman Bahri
>
> **摘要:** The internal representations learned by language models consistently exhibit striking geometric structure: calendar months organize into a circle, historical years form a smooth one-dimensional manifold, and cities' latitudes and longitudes can be decoded using a linear probe. To explain this neural code, we first show that language statistics exhibit translation symmetry (for example, the frequency with which any two months co-occur in text depends only on the time interval between them). We prove that this symmetry governs these geometric structures in high-dimensional word embedding models, and we analytically derive the manifold geometry of word representations. These predictions empirically match large text embedding models and large language models. Moreover, the representational geometry persists at moderate embedding dimension even when the relevant statistics are perturbed (e.g., by removing all sentences in which two months co-occur). We prove that this robustness emerges naturally when the co-occurrence statistics are controlled by an underlying latent variable. These results suggest that representational manifolds have a universal origin: symmetry in the statistics of natural data.
>
---
#### [replaced 009] Is This Just Fantasy? Language Model Representations Reflect Human Judgments of Event Plausibility
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的模态分类任务，旨在解决语言模型是否能准确判断句子的合理性。研究通过分析模型内部表示，发现其具备更可靠的模态分类能力，并与人类判断相关联。**

- **链接: [https://arxiv.org/pdf/2507.12553v2](https://arxiv.org/pdf/2507.12553v2)**

> **作者:** Michael A. Lepori; Jennifer Hu; Ishita Dasgupta; Roma Patel; Thomas Serre; Ellie Pavlick
>
> **摘要:** Language models (LMs) are used for a diverse range of tasks, from question answering to writing fantastical stories. In order to reliably accomplish these tasks, LMs must be able to discern the modal category of a sentence (i.e., whether it describes something that is possible, impossible, completely nonsensical, etc.). However, recent studies have called into question the ability of LMs to categorize sentences according to modality (Michaelov et al., 2025; Kauf et al., 2023). In this work, we identify linear representations that discriminate between modal categories within a variety of LMs, or modal difference vectors. Analysis of modal difference vectors reveals that LMs have access to more reliable modal categorization judgments than previously reported. Furthermore, we find that modal difference vectors emerge in a consistent order as models become more competent (i.e., through training steps, layers, and parameter count). Notably, we find that modal difference vectors identified within LM activations can be used to model fine-grained human categorization behavior. This potentially provides a novel view into how human participants distinguish between modal categories, which we explore by correlating projections along modal difference vectors with human participants' ratings of interpretable features. In summary, we derive new insights into LM modal categorization using techniques from mechanistic interpretability, with the potential to inform our understanding of modal categorization in humans.
>
---
#### [replaced 010] Both Ends Count! Just How Good are LLM Agents at "Text-to-Big SQL"?
- **分类: cs.DB; cs.CL; cs.IR**

- **简介: 该论文研究"Text-to-Big SQL"任务，解决传统文本到SQL评估在大数据场景下的不足。提出新指标，评估执行效率、成本与数据规模影响。**

- **链接: [https://arxiv.org/pdf/2602.21480v2](https://arxiv.org/pdf/2602.21480v2)**

> **作者:** Germán T. Eizaguirre; Lars Tissen; Marc Sánchez-Artigas
>
> **备注:** 11 pages, 4 figures
>
> **摘要:** Text-to-SQL and Big Data are both extensively benchmarked fields, yet there is limited research that evaluates them jointly. In the real world, Text-to-SQL systems are often embedded with Big Data workflows, such as large-scale data processing or interactive data analytics. We refer to this as "Text-to-Big SQL". However, existing text-to-SQL benchmarks remain narrowly scoped and overlook the cost and performance implications that arise at scale. For instance, translation errors that are minor on small datasets lead to substantial cost and latency overheads as data scales, a relevant issue completely ignored by text-to-SQL metrics. In this paper, we overcome this overlooked challenge by introducing novel and representative metrics for evaluating Text-to-Big SQL. Our study focuses on production-level LLM agents, a database-agnostic system adaptable to diverse user needs. Via an extensive evaluation of frontier models, we show that text-to-SQL metrics are insufficient for Big Data. In contrast, our proposed text-to-Big SQL metrics accurately reflect execution efficiency, cost, and the impact of data scale. Furthermore, we provide LLM-specific insights, including fine-grained, cross-model comparisons of latency and cost.
>
---
#### [replaced 011] PATCH: Mitigating PII Leakage in Language Models with Privacy-Aware Targeted Circuit PatcHing
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于隐私保护任务，旨在解决语言模型中的PII泄露问题。通过识别并修改泄露电路，提出PATCH方法有效降低泄露风险，提升隐私与实用性的平衡。**

- **链接: [https://arxiv.org/pdf/2510.07452v2](https://arxiv.org/pdf/2510.07452v2)**

> **作者:** Anthony Hughes; Vasisht Duddu; N. Asokan; Nikolaos Aletras; Ning Ma
>
> **摘要:** Language models (LMs) may memorize personally identifiable information (PII) from training data, enabling adversaries to extract it during inference. Existing defense mechanisms such as differential privacy (DP) reduce this leakage, but incur large drops in utility. Based on a comprehensive study using circuit discovery to identify the computational circuits responsible PII leakage in LMs, we hypothesize that specific PII leakage circuits in LMs should be responsible for this behavior. Therefore, we propose PATCH (Privacy-Aware Targeted Circuit PatcHing), a novel approach that first identifies and subsequently directly edits PII circuits to reduce leakage. PATCH achieves better privacy-utility trade-off than existing defenses, e.g., reducing recall of PII leakage from LMs by up to 65%. Finally, PATCH can be combined with DP to reduce recall of residual leakage of an LM to as low as 0.01%. Our analysis shows that PII leakage circuits persist even after the application of existing defense mechanisms. In contrast, PATCH can effectively mitigate their impact.
>
---
#### [replaced 012] Knowledge Fusion of Large Language Models Via Modular SkillPacks
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于跨能力迁移任务，解决大模型知识融合效率低的问题。提出GraftLLM方法，通过SkillPack格式实现高效知识转移与持续学习。**

- **链接: [https://arxiv.org/pdf/2505.18502v3](https://arxiv.org/pdf/2505.18502v3)**

> **作者:** Guodong Du; Zhuo Li; Xuanning Zhou; Junlin Li; Zesheng Shi; Wanyu Lin; Ho-Kin Tang; Xiucheng Li; Fangming Liu; Wenya Wang; Min Zhang; Jing Li
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** Cross-capability transfer is a key challenge in large language model (LLM) research, with applications in multi-task integration, model compression, and continual learning. Recent works like FuseLLM and FuseChat have demonstrated the potential of transferring multiple model capabilities to lightweight models, enhancing adaptability and efficiency, which motivates our investigation into more efficient cross-capability transfer methods. However, existing approaches primarily focus on small, homogeneous models, limiting their applicability. For large, heterogeneous models, knowledge distillation with full-parameter fine-tuning often overlooks the student model's intrinsic capacity and risks catastrophic forgetting, while PEFT methods struggle to effectively absorb knowledge from source LLMs. To address these issues, we introduce GraftLLM, a novel method that stores source model capabilities in a target model with SkillPack format. This approach preserves general capabilities, reduces parameter conflicts, and supports forget-free continual learning and model fusion. We employ a module-aware adaptive compression strategy to compress parameter updates, ensuring efficient storage while maintaining task-specific knowledge. The resulting SkillPack serves as a compact and transferable knowledge carrier, ideal for heterogeneous model fusion and continual learning. Experiments across various scenarios demonstrate that GraftLLM outperforms existing techniques in knowledge transfer, knowledge fusion, and forget-free learning, providing a scalable and efficient solution for cross-capability transfer. The code is publicly available at: https://github.com/duguodong7/GraftLLM.
>
---
#### [replaced 013] Evaluating the Diversity and Quality of LLM Generated Content
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，研究LLM生成内容的多样性与质量关系。针对偏好调优导致多样性下降的问题，提出有效语义多样性评估框架，分析不同模型的多样性与效率。**

- **链接: [https://arxiv.org/pdf/2504.12522v2](https://arxiv.org/pdf/2504.12522v2)**

> **作者:** Alexander Shypula; Shuo Li; Botong Zhang; Vishakh Padmakumar; Kayo Yin; Osbert Bastani
>
> **备注:** Published at COLM 2025
>
> **摘要:** Recent work suggests that preference-tuning techniques -- such as Reinforcement Learning from Human Feedback (RLHF) methods like PPO and GRPO, as well as alternatives like DPO -- reduce diversity, creating a dilemma given that these models are widely deployed in applications requiring varied outputs. We argue that diversity without consideration of quality has limited practical value. To address this issue, we introduce a framework for measuring effective semantic diversity -- diversity among outputs that meet quality thresholds -- which better reflects the practical utility of large language models (LLMs). Using open-ended tasks that require no human intervention, we find counterintuitive results: when using diversity metrics that do not explicitly consider quality, preference-tuned models -- particularly those trained via RL -- often produce outputs with lower diversity; however, these same preference-tuned models generate greater effective semantic diversity than supervised fine-tuned (SFT) or base models. Our analysis further shows another trend: while larger models may exhibit greater effective semantic diversity than smaller models, the smaller models are consistently more parameter-efficient at producing unique content within a fixed sampling budget. These findings have practical implications for applications that require diverse yet high-quality outputs, from creative assistance to synthetic data generation.
>
---
#### [replaced 014] The Automatic Verification of Image-Text Claims (AVerImaTeC) Shared Task
- **分类: cs.CL**

- **简介: 该论文介绍AVerImaTeC共享任务，旨在自动验证图像-文本声明。解决真实世界图像-文本声明的证据检索与验证问题，通过系统评估与分析，提升验证准确性。**

- **链接: [https://arxiv.org/pdf/2602.11221v2](https://arxiv.org/pdf/2602.11221v2)**

> **作者:** Rui Cao; Zhenyun Deng; Yulong Chen; Michael Schlichtkrull; Andreas Vlachos
>
> **备注:** Shared Task Overview and Summary for the Ninth FEVER Workshop, Co-located at EACL 2026
>
> **摘要:** The Automatic Verification of Image-Text Claims (AVerImaTeC) shared task aims to advance system development for retrieving evidence and verifying real-world image-text claims. Participants were allowed to either employ external knowledge sources, such as web search engines, or leverage the curated knowledge store provided by the organizers. System performance was evaluated using the AVerImaTeC score, defined as a conditional verdict accuracy in which a verdict is considered correct only when the associated evidence score exceeds a predefined threshold. The shared task attracted 14 submissions during the development phase and 6 submissions during the testing phase. All participating systems in the testing phase outperformed the baseline provided. The winning team, HUMANE, achieved an AVerImaTeC score of 0.5455. This paper provides a detailed description of the shared task, presents the complete evaluation results, and discusses key insights and lessons learned.
>
---
#### [replaced 015] A Third Paradigm for LLM Evaluation: Dialogue Game-Based Evaluation using clembench
- **分类: cs.CL**

- **简介: 该论文属于大语言模型评估任务，旨在解决现有评估方法的局限性。提出对话游戏评估作为第三种范式，并介绍clembench工具以实现可控、可重复的评估。**

- **链接: [https://arxiv.org/pdf/2507.08491v2](https://arxiv.org/pdf/2507.08491v2)**

> **作者:** David Schlangen; Sherzod Hakimov; Chalamalasetti Kranti; Jonathan Jordan; Philipp Sadler
>
> **备注:** All code required to run the benchmark, as well as extensive documentation, is available at https://github.com/clembench/clembench
>
> **摘要:** There are currently two main paradigms for evaluating large language models (LLMs), reference-based evaluation and preference-based evaluation. The first, carried over from the evaluation of machine learning models in general, relies on pre-defined task instances, for which reference task executions are available. The second, best exemplified by the LM-arena, relies on (often self-selected) users bringing their own intents to a site that routes these to several models in parallel, among whose responses the user then selects their most preferred one. The former paradigm hence excels at control over what is tested, while the latter comes with higher ecological validity, testing actual use cases interactively. Recently, a third complementary paradigm has emerged that combines some of the strengths of these approaches, offering control over multi-turn, reference-free, repeatable interactions, while stressing goal-directedness: dialogue game based evaluation. While the utility of this approach has been shown by several projects, its adoption has been held back by the lack of a mature, easily re-usable implementation. In this paper, we present clembench, which has been in continuous development since 2023 and has in its latest release been optimized for ease of general use. We describe how it can be used to benchmark one's own models (using a provided set of benchmark game instances in English), as well as how easily the benchmark itself can be extended with new, tailor-made targeted tests.
>
---
#### [replaced 016] UPDESH: Synthesizing Grounded Instruction Tuning Data for 13 Indic Languages
- **分类: cs.CL**

- **简介: 该论文属于多语言AI任务，旨在解决低资源语言文化适配问题。通过生成高质量合成数据集Updesh，提升多语言模型性能。**

- **链接: [https://arxiv.org/pdf/2509.21294v3](https://arxiv.org/pdf/2509.21294v3)**

> **作者:** Pranjal A. Chitale; Varun Gumma; Sanchit Ahuja; Prashant Kodali; Manan Uppadhyay; Deepthi Sudharsan; Sunayana Sitaram
>
> **备注:** Under Review
>
> **摘要:** Developing culturally grounded multilingual AI systems remains challenging, particularly for low-resource languages. While synthetic data offers promise, its effectiveness in multilingual and multicultural contexts is underexplored. We investigate bottom-up synthetic data generation using large open-source LLMs (>= 235B parameters) grounded in language-specific Wikipedia content, complementing dominant top-down translation-based approaches from English. We introduce Updesh, a high-quality large-scale synthetic instruction-following dataset comprising 9.5M data points across 13 Indian languages and English, encompassing diverse reasoning and generative tasks. Comprehensive evaluation using automated metrics and 10K human assessments confirms high data quality. Downstream evaluations performed by fine-tuning models on various datasets and assessing performance across 13 diverse multilingual datasets and model comparative evaluations, demonstrate that models trained on Updesh consistently obtain significant improvements on NLU, NLG evaluations. Finally, through ablation studies and cultural evaluations, we show that context-aware, culturally grounded data generation is essential for effective multilingual AI development.
>
---
#### [replaced 017] Temporal Sparse Autoencoders: Leveraging the Sequential Nature of Language for Interpretability
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理中的可解释性任务，旨在解决模型内部表示难以理解的问题。通过引入时间结构的稀疏自编码器，提升语义特征的连贯性和可解释性。**

- **链接: [https://arxiv.org/pdf/2511.05541v2](https://arxiv.org/pdf/2511.05541v2)**

> **作者:** Usha Bhalla; Alex Oesterling; Claudio Mayrink Verdun; Himabindu Lakkaraju; Flavio P. Calmon
>
> **备注:** 29 Pages, 12 figures. Accepted as an Oral Presentation at ICLR 2026
>
> **摘要:** Translating the internal representations and computations of models into concepts that humans can understand is a key goal of interpretability. While recent dictionary learning methods such as Sparse Autoencoders (SAEs) provide a promising route to discover human-interpretable features, they often only recover token-specific, noisy, or highly local concepts. We argue that this limitation stems from neglecting the temporal structure of language, where semantic content typically evolves smoothly over sequences. Building on this insight, we introduce Temporal Sparse Autoencoders (T-SAEs), which incorporate a novel contrastive loss encouraging consistent activations of high-level features over adjacent tokens. This simple yet powerful modification enables SAEs to disentangle semantic from syntactic features in a self-supervised manner. Across multiple datasets and models, T-SAEs recover smoother, more coherent semantic concepts without sacrificing reconstruction quality. Strikingly, they exhibit clear semantic structure despite being trained without explicit semantic signal, offering a new pathway for unsupervised interpretability in language models.
>
---
#### [replaced 018] Revisiting Self-Play Preference Optimization: On the Role of Prompt Difficulty
- **分类: cs.CL**

- **简介: 该论文属于大语言模型对齐任务，研究自博弈偏好优化中提示难度的影响。工作包括分析不同难度提示对优化效果的影响，并提出通过仅使用简单提示提升性能的策略。**

- **链接: [https://arxiv.org/pdf/2510.05534v2](https://arxiv.org/pdf/2510.05534v2)**

> **作者:** Yao Xiao; Jung-jae Kim; Roy Ka-wei Lee; Lidong Bing
>
> **摘要:** Self-play preference optimization has emerged as a prominent paradigm for aligning large language models (LLMs). It typically involves a language model to generate on-policy responses for prompts and a reward model (RM) to guide the selection of chosen and rejected responses, which can be further trained with direct preference optimization (DPO). However, the role of prompts remains underexplored, despite being a core component in this pipeline. In this work, we investigate how prompts of varying difficulty influence self-play preference optimization. We use the mean reward of sampled responses of a prompt as a proxy for its difficulty. We first find that difficult prompts exhibit substantially inferior self-play optimization performance compared to easy prompts for language models. Moreover, incorporating difficult prompts into training fails to enhance overall performance and, in fact, leads to slight degradation compared to training on easy prompts alone. Third, there is a clear upward trend in optimization performance as prompt difficulty decreases. We also observe that the performance gap between difficult and easy prompts tends to close as the model capacity increases, suggesting that prompt difficulty interacts with the model capacity. Building on these findings, we explore strategies to mitigate the adversary effect of difficult prompts on final performance. We demonstrate that only training on a small portion (30%) of the easiest prompts improves overall self-play performance on AlpacaEval~2 and Arena-Hard. We also report failed attempts and lessons learned.
>
---
#### [replaced 019] GPT-4o Lacks Core Features of Theory of Mind
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 论文探讨LLMs是否具备心智理论（ToM），通过新框架测试其对心理状态与行为关系的建模能力。发现GPT-4o缺乏一致的ToM模型，虽能模仿人类判断，但逻辑任务表现差。任务为评估LLMs的心智理论能力。**

- **链接: [https://arxiv.org/pdf/2602.12150v3](https://arxiv.org/pdf/2602.12150v3)**

> **作者:** John Muchovej; Amanda Royka; Shane Lee; Julian Jara-Ettinger
>
> **备注:** Submitted to CogSci 2025; see more at https://jmuchovej.com/projects/llm-tom. Note: "abstractness" is the second feature we test for, but due to arXiv's abstract requirements, the text has been altered
>
> **摘要:** Do Large Language Models (LLMs) possess a Theory of Mind (ToM)? Research into this question has focused on evaluating LLMs against benchmarks and found success across a range of social tasks. However, these evaluations do not test for the actual representations posited by ToM: namely, a causal model of mental states and behavior. Here, we use a cognitively-grounded definition of ToM to develop and test a new evaluation framework. Specifically, our approach probes whether LLMs have a coherent, domain-general, and consistent model of how mental states cause behavior -- regardless of whether that model matches a human-like ToM. We find that even though LLMs succeed in approximating human judgments in a simple ToM paradigm, they fail at a logically equivalent task and exhibit low consistency between their action predictions and corresponding mental state inferences. As such, these findings suggest that the social proficiency exhibited by LLMs is not the result of a domain-general or consistent ToM.
>
---
#### [replaced 020] Mapping Semantic & Syntactic Relationships with Geometric Rotation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决语言模型中语义与句法关系的几何表示问题。通过引入RISE方法，将语义-句法变换建模为嵌入空间中的旋转操作，实现跨语言和模型的一致性映射。**

- **链接: [https://arxiv.org/pdf/2510.09790v2](https://arxiv.org/pdf/2510.09790v2)**

> **作者:** Michael Freenor; Lauren Alvarez
>
> **备注:** 10 pages, 4 figures, 3 tables, 8 appendices, Published at ICLR 2026
>
> **摘要:** Understanding how language and embedding models encode semantic relationships is fundamental to model interpretability. While early word embeddings exhibited intuitive vector arithmetic (''king'' - ''man'' + ''woman'' = ''queen''), modern high-dimensional text representations lack straightforward interpretable geometric properties. We introduce Rotor-Invariant Shift Estimation (RISE), a geometric approach that represents semantic-syntactic transformations as consistent rotational operations in embedding space, leveraging the manifold structure of modern language representations. RISE operations have the ability to operate across both languages and models without reducing performance, suggesting the existence of analogous cross-lingual geometric structure. We compare and evaluate RISE using two baseline methods, three embedding models, three datasets, and seven morphologically diverse languages in five major language groups. Our results demonstrate that RISE consistently maps discourse-level semantic-syntactic transformations with distinct grammatical features (e.g., negation and conditionality) across languages and models. This work provides the first demonstration that discourse-level semantic-syntactic transformations correspond to consistent geometric operations in multilingual embedding spaces, empirically supporting the linear representation hypothesis at the sentence level.
>
---
#### [replaced 021] Large Language Models are Algorithmically Blind
- **分类: cs.CL**

- **简介: 该论文属于人工智能领域，研究LLMs在算法推理上的不足。任务是评估LLMs对算法执行的预测能力，发现其表现差，存在算法盲视现象。**

- **链接: [https://arxiv.org/pdf/2602.21947v2](https://arxiv.org/pdf/2602.21947v2)**

> **作者:** Sohan Venkatesh; Ashish Mahendran Kurapath; Tejas Melkote
>
> **备注:** 19 pages, 8 figures, 15 tables
>
> **摘要:** Large language models (LLMs) demonstrate remarkable breadth of knowledge, yet their ability to reason about computational processes remains poorly understood. Closing this gap matters for practitioners who rely on LLMs to guide algorithm selection and deployment. We address this limitation using causal discovery as a testbed and evaluate eight frontier LLMs against ground truth derived from large-scale algorithm executions and find systematic, near-total failure. Models produce ranges far wider than true confidence intervals yet still fail to contain the true algorithmic mean in the majority of instances; most perform worse than random guessing and the marginal above-random performance of the best model is most consistent with benchmark memorization rather than principled reasoning. We term this failure algorithmic blindness and argue it reflects a fundamental gap between declarative knowledge about algorithms and calibrated procedural prediction.
>
---
#### [replaced 022] Parallel Continuous Chain-of-Thought with Jacobi Iteration
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决连续链式思维训练效率低的问题。通过引入雅可比迭代方法，实现潜思token的并行更新，提升训练与推理效率。**

- **链接: [https://arxiv.org/pdf/2506.18582v2](https://arxiv.org/pdf/2506.18582v2)**

> **作者:** Haoyi Wu; Zhihao Teng; Kewei Tu
>
> **备注:** Accepted to EMNLP 2025 main conference
>
> **摘要:** Continuous chain-of-thought has been shown to be effective in saving reasoning tokens for large language models. By reasoning with continuous latent thought tokens, continuous CoT is able to perform implicit reasoning in a compact manner. However, the sequential dependencies between latent thought tokens spoil parallel training, leading to long training time. In this paper, we propose Parallel Continuous Chain-of-Thought (PCCoT), which performs Jacobi iteration on the latent thought tokens, updating them iteratively in parallel instead of sequentially and thus improving both training and inference efficiency of continuous CoT. Experiments demonstrate that by choosing the proper number of iterations, we are able to achieve comparable or even better performance while saving nearly 50% of the training and inference time. Moreover, PCCoT shows better stability and robustness in the training process. Our code is available at https://github.com/whyNLP/PCCoT.
>
---
#### [replaced 023] The Tool Decathlon: Benchmarking Language Agents for Diverse, Realistic, and Long-Horizon Task Execution
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Toolathlon基准，用于评估语言代理在多样化、真实且长周期任务中的表现，解决现有基准不足的问题。**

- **链接: [https://arxiv.org/pdf/2510.25726v2](https://arxiv.org/pdf/2510.25726v2)**

> **作者:** Junlong Li; Wenshuo Zhao; Jian Zhao; Weihao Zeng; Haoze Wu; Xiaochen Wang; Rui Ge; Yuxuan Cao; Yuzhen Huang; Wei Liu; Junteng Liu; Zhaochen Su; Yiyang Guo; Fan Zhou; Lueyang Zhang; Juan Michelini; Xingyao Wang; Xiang Yue; Shuyan Zhou; Graham Neubig; Junxian He
>
> **备注:** ICLR 2026, Website: https://toolathlon.xyz/
>
> **摘要:** Real-world language agents must handle complex, multi-step workflows across diverse Apps. For instance, an agent may manage emails by coordinating with calendars and file systems, or monitor a production database to detect anomalies and generate reports following an operating manual. However, existing language agent benchmarks often focus on narrow domains or simplified tasks that lack the diversity, realism, and long-horizon complexity required to evaluate agents' real-world performance. To address this gap, we introduce the Tool Decathlon (dubbed as Toolathlon), a benchmark for language agents offering diverse Apps and tools, realistic environment setup, and reliable execution-based evaluation. Toolathlon spans 32 software applications and 604 tools, ranging from everyday platforms such as Google Calendar and Notion to professional ones like WooCommerce, Kubernetes, and BigQuery. Most of the tools are based on a high-quality set of Model Context Protocol (MCP) servers that we may have revised or implemented ourselves. Unlike prior works, which primarily ensure functional realism but offer limited environment state diversity, we provide realistic initial environment states from real software, such as Canvas courses with dozens of students or real financial spreadsheets. This benchmark includes 108 manually sourced or crafted tasks in total, requiring interacting with multiple Apps over around 20 turns on average to complete. Each task is strictly verifiable through dedicated evaluation scripts. Comprehensive evaluation of SOTA models highlights their significant shortcomings: the best-performing model, Claude-4.5-Sonnet, achieves only a 38.6% success rate with 20.2 tool calling turns on average, while the top open-weights model DeepSeek-V3.2-Exp reaches 20.1%. We expect Toolathlon to drive the development of more capable language agents for real-world, long-horizon task execution.
>
---
#### [replaced 024] Physical Commonsense Reasoning for Lower-Resourced Languages and Dialects: a Study on Basque
- **分类: cs.CL**

- **简介: 该论文属于物理常识推理任务，旨在解决低资源语言如巴斯克语在非问答任务中的表现问题。工作包括构建首个巴斯克语物理常识数据集，并评估模型在不同层次上的表现。**

- **链接: [https://arxiv.org/pdf/2602.14812v2](https://arxiv.org/pdf/2602.14812v2)**

> **作者:** Jaione Bengoetxea; Itziar Gonzalez-Dios; Rodrigo Agerri
>
> **摘要:** Physical commonsense reasoning represents a fundamental capability of human intelligence, enabling individuals to understand their environment, predict future events, and navigate physical spaces. Recent years have witnessed growing interest in reasoning tasks within Natural Language Processing (NLP). However, no prior research has examined the performance of Large Language Models (LLMs) on non-question-answering (non-QA) physical commonsense reasoning tasks in low-resource languages such as Basque. Taking the Italian GITA as a starting point, this paper addresses this gap by presenting BasPhyCo, the first non-QA physical commonsense reasoning dataset for Basque, available in both standard and dialectal variants. We evaluate model performance across three hierarchical levels of commonsense understanding: (1) distinguishing between plausible and implausible narratives (accuracy), (2) identifying the conflicting element that renders a narrative implausible (consistency), and (3) determining the specific physical state that creates the implausibility (verifiability). These tasks were assessed using multiple multilingual LLMs as well as models pretrained specifically for Italian and Basque. Results indicate that, in terms of verifiability, LLMs exhibit limited physical commonsense capabilities in low-resource languages such as Basque, especially when processing dialectal variants.
>
---
#### [replaced 025] Inducing Dyslexia in Vision Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于认知科学与人工智能交叉任务，旨在模拟阅读障碍（如失读症）的机制。通过扰动视觉语言模型中的词汇处理单元，研究其对阅读能力的影响，揭示失读症的关键特征。**

- **链接: [https://arxiv.org/pdf/2509.24597v3](https://arxiv.org/pdf/2509.24597v3)**

> **作者:** Melika Honarmand; Ayati Sharma; Badr AlKhamissi; Johannes Mehrer; Martin Schrimpf
>
> **摘要:** Dyslexia, a neurodevelopmental disorder characterized by persistent reading difficulties, is often linked to reduced activity of the visual word form area (VWFA) in the ventral occipito-temporal cortex. Traditional approaches to studying dyslexia, such as behavioral and neuroimaging methods, have provided valuable insights but remain limited in their ability to test causal hypotheses about the underlying mechanisms of reading impairments. In this study, we use large-scale vision-language models (VLMs) to simulate dyslexia by functionally identifying and perturbing artificial analogues of word processing. Using stimuli from cognitive neuroscience, we identify visual-word-form-selective units within VLMs and demonstrate that they predict human VWFA neural responses. Ablating model VWF units leads to selective impairments in reading tasks while general visual and language comprehension abilities remain intact. In particular, the resulting model matches dyslexic humans' phonological deficits without a significant change in orthographic processing, and mirrors dyslexic behavior in font sensitivity. Taken together, our modeling results replicate key characteristics of dyslexia and establish a computational framework for investigating brain disorders.
>
---
#### [replaced 026] Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于知识蒸馏任务，旨在提升学生模型性能。通过引入奖励扩展和灵活参考模型，提出G-OPD框架，解决传统OPD效果有限的问题。**

- **链接: [https://arxiv.org/pdf/2602.12125v2](https://arxiv.org/pdf/2602.12125v2)**

> **作者:** Wenkai Yang; Weijie Liu; Ruobing Xie; Kai Yang; Saiyong Yang; Yankai Lin
>
> **备注:** v2, update results under stronger teachers with more RL training steps
>
> **摘要:** On-policy distillation (OPD), which aligns the student with the teacher's logit distribution on student-generated trajectories, has demonstrated strong empirical gains in improving student performance and often outperforms off-policy distillation and reinforcement learning (RL) paradigms. In this work, we first theoretically show that OPD is a special case of dense KL-constrained RL where the reward function and the KL regularization are always weighted equally and the reference model can by any model. Then, we propose the Generalized On-Policy Distillation (G-OPD) framework, which extends the standard OPD objective by introducing a flexible reference model and a reward scaling factor that controls the relative weight of the reward term against the KL regularization. Through comprehensive experiments on math reasoning and code generation tasks, we derive two novel insights: (1) Setting the reward scaling factor to be greater than 1 (i.e., reward extrapolation), which we term ExOPD, consistently improves over standard OPD across a range of teacher-student size pairings. In particular, in the setting where we merge the knowledge from different domain experts, obtained by applying domain-specific RL to the same student model, back into the original student, ExOPD enables the student to even surpass the teacher's performance boundary and outperform the domain teachers. (2) Building on ExOPD, we further find that in the strong-to-weak distillation setting (i.e., distilling a smaller student from a larger teacher), performing reward correction by choosing the reference model as the teacher's base model before RL yields a more accurate reward signal and further improves distillation performance. However, this choice assumes access to the teacher's pre-RL variant and incurs more computational overhead. We hope our work offers new insights for future research on OPD.
>
---
#### [replaced 027] When Large Multimodal Models Confront Evolving Knowledge: Challenges and Explorations
- **分类: cs.CL**

- **简介: 该论文属于多模态知识注入任务，旨在解决LMMs在动态多模态知识更新中的性能下降问题。提出MMEVOKE基准，探索知识增强与保留方法以提升模型能力。**

- **链接: [https://arxiv.org/pdf/2505.24449v2](https://arxiv.org/pdf/2505.24449v2)**

> **作者:** Kailin Jiang; Yuntao Du; Yukai Ding; Yuchen Ren; Ning Jiang; Zhi Gao; Zilong Zheng; Lei Liu; Bin Li; Qing Li
>
> **备注:** ICLR 2026, Project Page: https://evoke-lmm.github.io/
>
> **摘要:** Large Multimodal Models (LMMs) store vast amounts of pretrained knowledge but struggle to remain aligned with real-world updates, making it difficult to avoid capability degradation when acquiring evolving knowledge. Furthermore, most current work focuses on exploring static textual knowledge injection, neglecting dynamic multimodal evolving knowledge injection, leaving the potential of LMMs for multimodal knowledge injection as an open question. To address this, we first propose a pipeline to construct MMEVOKE, a benchmark for evaluating LMMs' ability in multimodal evolving knowledge injection. MMEVOKE contains 9,422 samples spanning 159 subtypes. Then, based on extensive experiments with MMEVOKE, we reveal challenges such as poor injection performance and capability degradation in existing knowledge injection methods through knowledge injection tests and general capability tests. Finally, to tackle these challenges, we introduce knowledge augmentation and knowledge retention methods, finding that knowledge-aware augmentation strengthens knowledge injection performance, and that Data Replay and MoE methods effectively mitigate capability degradation.
>
---
#### [replaced 028] LiveMCPBench: Can Agents Navigate an Ocean of MCP Tools?
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于大模型工具调用任务，解决MCP系统在真实场景下的评估不足问题。构建了LiveMCPBench基准，评估多工具、多服务器环境下的模型能力。**

- **链接: [https://arxiv.org/pdf/2508.01780v2](https://arxiv.org/pdf/2508.01780v2)**

> **作者:** Guozhao Mo; Wenliang Zhong; Jiawei Chen; Qianhao Yuan; Xuanang Chen; Yaojie Lu; Hongyu Lin; Ben He; Xianpei Han; Le Sun
>
> **备注:** Our code and data will be publicly available at https://icip-cas.github.io/LiveMCPBench
>
> **摘要:** Model Context Protocol (MCP) has become a key infrastructure for connecting LLMs with external tools, scaling to 10,000+ MCP servers with diverse tools. Unfortunately, there is still a large gap between real-world MCP usage and current evaluation: they typically assume single-server settings and directly inject tools into the model's context, bypassing the challenges of large-scale retrieval and multi-tool composition. To bridge this gap, we propose LiveMCPBench, which evaluates 95 real-world daily tasks explicitly constructed to stress diverse tools and scaled multi-server routing. The benchmark includes a ready-to-deploy tool suite of 70 servers with 527 tools, ensuring reproducibility without scattered API configuration. We further introduce an LLM-as-a-Judge evaluation framework that directly verifies task outcomes, handling dynamic data sources and multiple valid solution paths. We benchmark 12 state-of-the-art LLMs and observe a substantial performance gap: while Claude-Sonnet-4 reaches 78.95% task success, most models achieve only 30-50%. Our analysis reveals that the active tool composition strongly correlates with task success, whereas retrieval errors account for nearly half of all failures, highlighting retrieval as the dominant bottleneck. Together, these results provide the first large-scale, reproducible diagnosis of MCP agent capabilities and point towards future research on improving retrieval robustness and encouraging effective tool composition. Our code and data are publicly available at https://icip-cas.github.io/LiveMCPBench.
>
---
#### [replaced 029] Under the Influence: Quantifying Persuasion and Vigilance in Large Language Models
- **分类: cs.CL; cs.LG; cs.MA**

- **简介: 该论文研究大语言模型的说服力与警觉性关系，通过 Sokoban 游戏测试其任务表现、说服能力和警觉性，旨在提升 AI 安全。**

- **链接: [https://arxiv.org/pdf/2602.21262v2](https://arxiv.org/pdf/2602.21262v2)**

> **作者:** Sasha Robinson; Kerem Oktar; Katherine M. Collins; Ilia Sucholutsky; Kelsey R. Allen
>
> **摘要:** With increasing integration of Large Language Models (LLMs) into areas of high-stakes human decision-making, it is important to understand the risks they introduce as advisors. To be useful advisors, LLMs must sift through large amounts of content, written with both benevolent and malicious intent, and then use this information to convince a user to take a specific action. This involves two social capacities: vigilance (the ability to determine which information to use, and which to discard) and persuasion (synthesizing the available evidence to make a convincing argument). While existing work has investigated these capacities in isolation, there has been little prior investigation of how these capacities may be linked. Here, we use a simple multi-turn puzzle-solving game, Sokoban, to study LLMs' abilities to persuade and be rationally vigilant towards other LLM agents. We find that puzzle-solving performance, persuasive capability, and vigilance are dissociable capacities in LLMs. Performing well on the game does not automatically mean a model can detect when it is being misled, even if the possibility of deception is explicitly mentioned. However, LLMs do consistently modulate their token use, using fewer tokens to reason when advice is benevolent and more when it is malicious, even if they are still persuaded to take actions leading them to failure. To our knowledge, our work presents the first investigation of the relationship between persuasion, vigilance, and task performance in LLMs, and suggests that monitoring all three independently will be critical for future work in AI safety.
>
---
#### [replaced 030] Not All Attention is Needed: Parameter and Computation Efficient Transfer Learning for Multi-modal Large Language Models
- **分类: cs.MM; cs.CL**

- **简介: 该论文针对多模态大语言模型的高效微调问题，提出EAS方法，通过跳过冗余注意力机制提升推理速度并保持性能。**

- **链接: [https://arxiv.org/pdf/2403.15226v3](https://arxiv.org/pdf/2403.15226v3)**

> **作者:** Qiong Wu; Weihao Ye; Yiyi Zhou; Xiaoshuai Sun; Rongrong Ji
>
> **摘要:** In this paper, we propose a novel parameter and computation efficient tuning method for Multi-modal Large Language Models (MLLMs), termed Efficient Attention Skipping (EAS). Concretely, we first reveal that multi-head attentions (MHAs), the main computational overhead of MLLMs, are often redundant to downstream tasks. Based on this observation, EAS evaluates the attention redundancy and skips the less important MHAs to speed up inference. Besides, we also propose a novel propagation-of-information adapter (PIA) to serve the attention skipping of EAS and keep parameter efficiency, which can be further re-parameterized into feed-forward networks (FFNs) for zero-extra latency. To validate EAS, we apply it to a recently proposed MLLM called LaVIN and a classic VL pre-trained model called METER, and conduct extensive experiments on a set of benchmarks. The experiments show that EAS not only retains high performance and parameter efficiency, but also greatly speeds up inference speed. For instance, LaVIN-EAS can obtain 89.98\% accuracy on ScineceQA while speeding up inference by 2.2 times to LaVIN
>
---
#### [replaced 031] PuppetChat: Fostering Intimate Communication through Bidirectional Actions and Micronarratives
- **分类: cs.HC; cs.AI; cs.CL; cs.CY**

- **简介: 该论文提出PuppetChat，解决即时通讯中情感表达不足的问题，通过双向互动和个性化故事增强亲密交流。属于人机交互任务。**

- **链接: [https://arxiv.org/pdf/2602.19463v2](https://arxiv.org/pdf/2602.19463v2)**

> **作者:** Emma Jiren Wang; Siying Hu; Zhicong Lu
>
> **备注:** 19 pages, 8 figures; Accepted by ACM CHI 2026. In Proceedings of the 2026 CHI Conference on Human Factors in Computing Systems (CHI'26)
>
> **摘要:** As a primary channel for sustaining modern intimate relationships, instant messaging facilitates frequent connection across distances. However, today's tools often dilute care; they favor single tap reactions and vague emojis that do not support two way action responses, do not preserve the feeling that the exchange keeps going without breaking, and are weakly tied to who we are and what we share. To address this challenge, we present PuppetChat, a dyadic messaging prototype that restores this expressive depth through embodied interaction. PuppetChat uses a reciprocity aware recommender to encourage responsive actions and generates personalized micronarratives from user stories to ground interactions in personal history. Our 10-day field study with 11 dyads of close partners or friends revealed that this approach enhanced social presence, supported more expressive self disclosure, and sustained continuity and shared memories.
>
---
#### [replaced 032] Mitigating Multimodal Hallucinations via Gradient-based Self-Reflection
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态任务，解决MLLM中的幻觉问题。通过GACD方法，估计并抑制文本与视觉的偏差，提升输出的视觉合理性。**

- **链接: [https://arxiv.org/pdf/2509.03113v4](https://arxiv.org/pdf/2509.03113v4)**

> **作者:** Shan Wang; Maying Shen; Nadine Chang; Chuong Nguyen; Hongdong Li; Jose M. Alvarez
>
> **摘要:** Multimodal large language models achieve strong performance across diverse tasks but remain prone to hallucinations, where outputs are not grounded in visual inputs. This issue can be attributed to two main biases: text-visual bias, the overreliance on prompts and prior outputs, and co-occurrence bias, spurious correlations between frequently paired objects. We propose Gradient-based Influence-Aware Constrained Decoding (GACD), an inference-based method, that addresses both biases without auxiliary models, and is readily applicable to existing models without finetuning. The core of our approach is bias estimation, which uses first-order Taylor gradients to understand the contribution of individual tokens-visual features and text tokens-to the current output. Based on this analysis, GACD mitigates hallucinations through two components: (1) suppressing spurious visual features correlated with the output objects, and (2) rebalancing cross-modal contributions by strengthening visual features relative to text. Experiments across multiple benchmarks demonstrate that GACD effectively reduces hallucinations and improves the visual grounding of MLLM outputs.
>
---
#### [replaced 033] Bob's Confetti: Phonetic Memorization Attacks in Music and Video Generation
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于AI安全任务，揭示生成模型对版权内容的 phonetic memorization 脆弱性，提出APT攻击方法绕过文本过滤，验证其有效性并展示跨模态影响。**

- **链接: [https://arxiv.org/pdf/2507.17937v4](https://arxiv.org/pdf/2507.17937v4)**

> **作者:** Jaechul Roh; Zachary Novack; Yuefeng Peng; Niloofar Mireshghallah; Taylor Berg-Kirkpatrick; Amir Houmansadr
>
> **摘要:** Generative AI systems for music and video commonly use text-based filters to prevent regurgitation of copyrighted material. We expose a significant vulnerability in this approach by introducing Adversarial PhoneTic Prompting (APT), a novel attack that bypasses these safeguards by exploiting phonetic memorization--the tendency of models to bind sub-lexical acoustic patterns (phonemes, rhyme, stress, cadence) to memorized copyrighted content. APT replaces iconic lyrics with homophonic but semantically unrelated alternatives (e.g., "mom's spaghetti" becomes "Bob's confetti"), preserving phonetic structure while evading lexical filters. We evaluate APT on leading lyrics-to-song models (Suno, YuE) across English and Korean songs spanning rap, pop, and K-pop. APT achieves 91% average similarity to copyrighted originals, versus 13.7% for random lyrics and 42.2% for semantic paraphrases. Embedding analysis confirms the mechanism: YuE's text encoder treats APT-modified lyrics as near-identical to originals (cosine similarity 0.90) while Sentence-BERT semantic similarity drops to 0.71, showing the model encodes phonetic structure over meaning. This vulnerability extends cross-modally--Veo 3 reconstructs visual scenes from original music videos when prompted with APT lyrics alone, despite no visual cues in the prompt. We further show that phonetic-semantic defense signatures fail, as APT prompts exhibit higher semantic similarity than benign paraphrases. Our findings reveal that sub-lexical acoustic structure acts as a cross-modal retrieval key, rendering current copyright filters systematically vulnerable. Demo examples are available at https://jrohsc.github.io/music_attack/.
>
---
#### [replaced 034] Self-adaptive Dataset Construction for Real-World Multimodal Safety Scenarios
- **分类: cs.CV; cs.CL; cs.CR**

- **简介: 该论文属于多模态安全数据集构建任务，旨在解决现有数据集无法覆盖真实复杂场景的问题。提出一种图像导向的自适应构建方法，生成35k图像-文本对，并引入标准化评估指标。**

- **链接: [https://arxiv.org/pdf/2509.04403v2](https://arxiv.org/pdf/2509.04403v2)**

> **作者:** Jingen Qu; Lijun Li; Bo Zhang; Yichen Yan; Jing Shao
>
> **备注:** Accepted at EMNLP 2025 Findings
>
> **摘要:** Multimodal large language models (MLLMs) are rapidly evolving, presenting increasingly complex safety challenges. However, current dataset construction methods, which are risk-oriented, fail to cover the growing complexity of real-world multimodal safety scenarios (RMS). And due to the lack of a unified evaluation metric, their overall effectiveness remains unproven. This paper introduces a novel image-oriented self-adaptive dataset construction method for RMS, which starts with images and end constructing paired text and guidance responses. Using the image-oriented method, we automatically generate an RMS dataset comprising 35k image-text pairs with guidance responses. Additionally, we introduce a standardized safety dataset evaluation metric: fine-tuning a safety judge model and evaluating its capabilities on other safety datasets.Extensive experiments on various tasks demonstrate the effectiveness of the proposed image-oriented pipeline. The results confirm the scalability and effectiveness of the image-oriented approach, offering a new perspective for the construction of real-world multimodal safety datasets. The dataset is presented at https://huggingface.co/datasets/NewCityLetter/RMS2/tree/main.
>
---
#### [replaced 035] DeVisE: Behavioral Testing of Medical Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于医疗大模型行为测试任务，旨在检验模型是否具备真实医学推理能力。通过构造反事实数据，评估模型对人口统计和生命体征变化的响应，揭示标准指标无法反映的模型行为差异。**

- **链接: [https://arxiv.org/pdf/2506.15339v3](https://arxiv.org/pdf/2506.15339v3)**

> **作者:** Camila Zurdo Tagliabue; Heloisa Oss Boll; Aykut Erdem; Erkut Erdem; Iacer Calixto
>
> **备注:** Camera-ready version published at Findings of the EACL 2026
>
> **摘要:** Large language models (LLMs) are increasingly applied in clinical decision support, yet current evaluations rarely reveal whether their outputs reflect genuine medical reasoning or superficial correlations. We introduce DeVisE (Demographics and Vital signs Evaluation), a behavioral testing framework that probes fine-grained clinical understanding through controlled counterfactuals. Using intensive care unit (ICU) discharge notes from MIMIC-IV, we construct both raw (real-world) and template-based (synthetic) variants with single-variable perturbations in demographic (age, gender, ethnicity) and vital sign attributes. We evaluate eight LLMs, spanning general-purpose and medical variants, under zero-shot setting. Model behavior is analyzed through (1) input-level sensitivity, capturing how counterfactuals alter perplexity, and (2) downstream reasoning, measuring their effect on predicted ICU length-of-stay and mortality. Overall, our results show that standard task metrics obscure clinically relevant differences in model behavior, with models differing substantially in how consistently and proportionally they adjust predictions to counterfactual perturbations.
>
---
#### [replaced 036] Fine-tuning Done Right in Model Editing
- **分类: cs.CL**

- **简介: 该论文属于模型编辑任务，旨在解决细调在模型编辑中效果不佳的问题。通过调整优化流程和参数位置，提出LocFT-BF方法，显著提升编辑效果。**

- **链接: [https://arxiv.org/pdf/2509.22072v4](https://arxiv.org/pdf/2509.22072v4)**

> **作者:** Wanli Yang; Rui Tang; Hongyu Zang; Du Su; Qi Cao; Jingang Wang; Huawei Shen; Xueqi Cheng; Fei Sun
>
> **备注:** Accepted as a conference paper at ICLR 2026
>
> **摘要:** Fine-tuning, a foundational method for adapting large language models, has long been considered ineffective for model editing. Here, we challenge this belief, arguing that the reported failure arises not from the inherent limitation of fine-tuning itself, but from adapting it to the sequential nature of the editing task, a single-pass depth-first pipeline that optimizes each sample to convergence before moving on. While intuitive, this depth-first pipeline coupled with sample-wise updating over-optimizes each edit and induces interference across edits. Our controlled experiments reveal that simply restoring fine-tuning to the standard breadth-first (i.e., epoch-based) pipeline with mini-batch optimization substantially improves its effectiveness for model editing. Moreover, fine-tuning in editing also suffers from suboptimal tuning parameter locations inherited from prior methods. Through systematic analysis of tuning locations, we derive LocFT-BF, a simple and effective localized editing method built on the restored fine-tuning framework. Extensive experiments across diverse LLMs and datasets demonstrate that LocFT-BF outperforms state-of-the-art methods by large margins. Notably, to our knowledge, it is the first to sustain 100K edits and 72B-parameter models,10 x beyond prior practice, without sacrificing general capabilities. By clarifying a long-standing misconception and introducing a principled localized tuning strategy, we advance fine-tuning from an underestimated baseline to a leading method for model editing, establishing a solid foundation for future research.
>
---
#### [replaced 037] Inference-Cost-Aware Dynamic Tree Construction for Efficient Inference in Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型推理延迟问题。通过引入考虑推理成本的动态树结构，提升推理效率。**

- **链接: [https://arxiv.org/pdf/2510.26577v2](https://arxiv.org/pdf/2510.26577v2)**

> **作者:** Yinrong Hong; Zhiquan Tan; Kai Hu
>
> **摘要:** Large Language Models (LLMs) face significant inference latency challenges stemming from their autoregressive design and large size. To address this, speculative decoding emerges as a solution, enabling the simultaneous generation and validation of multiple tokens. While recent approaches like EAGLE-2 and EAGLE-3 improve speculative decoding using dynamic tree structures, they often neglect the impact of crucial system variables such as GPU devices and batch sizes. Therefore, we introduce a new dynamic tree decoding approach called CAST that takes into account inference costs, including factors such as GPU configurations and batch sizes, to dynamically refine the tree structure. Through comprehensive experimentation across six diverse tasks and utilizing six distinct LLMs, our methodology demonstrates remarkable results, achieving speeds up to 5.2 times faster than conventional decoding methods. Moreover, it generally outperforms existing state-of-the-art techniques from 5 % to 20%. The code is available at https://github.com/EAGLE-Research/sglang-eagle4.
>
---
#### [replaced 038] Unmasking Reasoning Processes: A Process-aware Benchmark for Evaluating Structural Mathematical Reasoning in LLMs
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于数学推理任务，旨在解决现有基准对模型推理能力评估不足的问题。提出ReasoningMath-Plus基准和HCRS评分方法，以更准确评估结构化推理能力。**

- **链接: [https://arxiv.org/pdf/2602.00564v2](https://arxiv.org/pdf/2602.00564v2)**

> **作者:** Xiang Zheng; Weiqi Zhai; Wei Wang; Boyu Yang; Wenbo Li; Ruixiang Luo; Haoxiang Sun; Yucheng Wang; Zhengze Li; Meng Wang; Yuetian Du; Guojie Lin; Yaxuan Wang; Xiaoxiao Xu; Yanhu Mo; Xuan Ren; Hu Wei; Bing Zhao
>
> **备注:** 8 pages, and 3 figures
>
> **摘要:** Recent large language models (LLMs) achieve near-saturation accuracy on many established mathematical reasoning benchmarks, raising concerns about their ability to diagnose genuine reasoning competence. This saturation largely stems from the dominance of template-based computation and shallow arithmetic decomposition in existing datasets, which underrepresent reasoning skills such as multi-constraint coordination, constructive logical synthesis, and spatial inference. To address this gap, we introduce ReasoningMath-Plus, a benchmark of 150 carefully curated problems explicitly designed to evaluate structural reasoning. Each problem emphasizes reasoning under interacting constraints, constructive solution formation, or non-trivial structural insight, and is annotated with a minimal reasoning skeleton to support fine-grained process-level evaluation. Alongside the dataset, we introduce HCRS (Hazard-aware Chain-based Rule Score), a deterministic step-level scoring function, and train a Process Reward Model (PRM) on the annotated reasoning traces. Empirically, while leading models attain relatively high final-answer accuracy (up to 5.8/10), HCRS-based holistic evaluation yields substantially lower scores (average 4.36/10, best 5.14/10), showing that answer-only metrics can overestimate reasoning robustness.
>
---
#### [replaced 039] RELOOP: Recursive Retrieval with Multi-Hop Reasoner and Planners for Heterogeneous QA
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出RELOOP框架，解决多步骤异构问答任务中的信息检索与生成问题，通过结构感知的迭代机制提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2510.20505v3](https://arxiv.org/pdf/2510.20505v3)**

> **作者:** Ruiyi Yang; Hao Xue; Imran Razzak; Hakim Hacid; Flora D. Salim
>
> **备注:** 19 pages, 2 figures
>
> **摘要:** Retrieval-augmented generation (RAG) remains brittle on multi-step questions and heterogeneous evidence sources, trading accuracy against latency and token/tool budgets. This paper introduces RELOOP, a structure aware framework using Hierarchical Sequence (HSEQ) that (i) linearize documents, tables, and knowledge graphs into a reversible hierarchical sequence with lightweight structural tags, and (ii) perform structure-aware iteration to collect just-enough evidence before answer synthesis. A Head Agent provides guidance that leads retrieval, while an Iteration Agent selects and expands HSeq via structure-respecting actions (e.g., parent/child hops, table row/column neighbors, KG relations); Finally the head agent composes canonicalized evidence to genearte the final answer, with an optional refinement loop to resolve detected contradictions. Experiments on HotpotQA (text), HybridQA/TAT-QA (table+text), and MetaQA (KG) show consistent EM/F1 gains over strong single-pass, multi-hop, and agentic RAG baselines with high efficiency. Besides, RELOOP exhibits three key advantages: (1) a format-agnostic unification that enables a single policy to operate across text, tables, and KGs without per-dataset specialization; (2) \textbf{guided, budget-aware iteration} that reduces unnecessary hops, tool calls, and tokens while preserving accuracy; and (3) evidence canonicalization for reliable QA, improving answers consistency and auditability.
>
---
#### [replaced 040] Document Reconstruction Unlocks Scalable Long-Context RLVR
- **分类: cs.CL**

- **简介: 该论文属于长文本理解任务，解决LLM长context能力提升问题。通过无监督文档重建训练，提升模型全局连贯性。**

- **链接: [https://arxiv.org/pdf/2602.08237v3](https://arxiv.org/pdf/2602.08237v3)**

> **作者:** Yao Xiao; Lei Wang; Yue Deng; Guanzheng Chen; Ziqi Jin; Jung-jae Kim; Xiaoli Li; Roy Ka-wei Lee; Lidong Bing
>
> **摘要:** Reinforcement Learning with Verifiable Rewards~(RLVR) has become a prominent paradigm to enhance the capabilities (i.e.\ long-context) of Large Language Models~(LLMs). However, it often relies on gold-standard answers or explicit evaluation rubrics provided by powerful teacher models or human experts, which are costly and time-consuming. In this work, we investigate unsupervised approaches to enhance the long-context capabilities of LLMs, eliminating the need for heavy human annotations or teacher models' supervision. Specifically, we first replace a few paragraphs with special placeholders in a long document. LLMs are trained through reinforcement learning to reconstruct the document by correctly identifying and sequencing missing paragraphs from a set of candidate options. This training paradigm enables the model to capture global narrative coherence, significantly boosting long-context performance. We validate the effectiveness of our method on two widely used benchmarks, RULER and LongBench~v2. While acquiring noticeable gains on RULER, it can also achieve a reasonable improvement on LongBench~v2 without any manually curated long-context QA data. Furthermore, we conduct extensive ablation studies to analyze the impact of reward design, data curation strategies, training schemes, and data scaling effects on model performance. We publicly release our code, data, and models.
>
---
#### [replaced 041] Improving Discrete Diffusion Unmasking Policies Beyond Explicit Reference Policies
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于语言建模任务，解决掩码扩散模型中解码顺序优化问题。通过学习一个调度策略替代传统启发式方法，提升生成效果。**

- **链接: [https://arxiv.org/pdf/2510.05725v2](https://arxiv.org/pdf/2510.05725v2)**

> **作者:** Chunsan Hong; Seonho An; Min-Soo Kim; Jong Chul Ye
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Masked diffusion models (MDMs) have recently emerged as a novel framework for language modeling. MDMs generate sentences by iteratively denoising masked sequences, filling in [MASK] tokens step by step. Although MDMs support any-order sampling, performance is highly sensitive to the choice of which position to unmask next. Prior work typically relies on rule-based schedules (e.g., max-confidence, max-margin), which provide ad hoc improvements. In contrast, we replace these heuristics with a learned scheduler. Specifically, we cast denoising as a KL-regularized Markov decision process (MDP) with an explicit reference policy and optimize a regularized objective that admits policy improvement and convergence guarantees under standard assumptions. We prove that the optimized policy under this framework generates samples that more closely match the data distribution than heuristic schedules. Empirically, across four benchmarks, our learned policy consistently outperforms max-confidence: for example, on SUDOKU, where unmasking order is critical, it yields a 20.1% gain over random and a 11.2% gain over max-confidence. Code is available at https://github.com/chunsanHong/UPO.
>
---
#### [replaced 042] PoSh: Using Scene Graphs To Guide LLMs-as-a-Judge For Detailed Image Descriptions
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于图像描述评估任务，旨在解决现有评价指标不适用于长文本的问题。提出PoSh方法，利用场景图指导LLM评分，提升评估准确性。**

- **链接: [https://arxiv.org/pdf/2510.19060v3](https://arxiv.org/pdf/2510.19060v3)**

> **作者:** Amith Ananthram; Elias Stengel-Eskin; Lorena A. Bradford; Julia Demarest; Adam Purvis; Keith Krut; Robert Stein; Rina Elster Pantalony; Mohit Bansal; Kathleen McKeown
>
> **备注:** Accepted at ICLR 2026. 26 pages, 9 figures. Metric/benchmark available at https://github.com/amith-ananthram/posh
>
> **摘要:** While vision-language models (VLMs) have advanced into detailed image description, evaluation remains a challenge. Standard metrics (e.g. CIDEr, SPICE) were designed for short texts and tuned to recognize errors that are now uncommon, such as object misidentification. In contrast, long texts require sensitivity to attribute and relation attachments and scores that localize errors to particular text spans. In this work, we introduce PoSh, a metric for detailed image description that uses scene graphs as structured rubrics to guide LLMs-as-a-Judge, producing aggregate scores grounded in fine-grained errors (e.g. mistakes in compositional understanding). PoSh is replicable, interpretable and a better proxy for human raters than existing metrics (including GPT4o-as-a-Judge). To validate PoSh, we introduce a challenging new dataset, DOCENT. This novel benchmark contains artwork, paired with expert-written references, and model-generated descriptions, augmented with granular and coarse judgments of their quality from art history students. Thus, DOCENT enables evaluating both detailed image description metrics and detailed image description itself in a challenging new domain. We show that PoSh achieves stronger correlations (+0.05 Spearman $ρ$) with the human judgments in DOCENT than the best open-weight alternatives, is robust to image type (using CapArena, an existing dataset of web imagery) and is a capable reward function, outperforming standard supervised fine-tuning. Then, using PoSh, we characterize the performance of open and closed models in describing the paintings, sketches and statues in DOCENT and find that foundation models struggle to achieve full, error-free coverage of images with rich scene dynamics, establishing a demanding new task to gauge VLM progress. Through both PoSh and DOCENT, we hope to enable advances in important areas such as assistive text generation.
>
---
#### [replaced 043] Index Light, Reason Deep: Deferred Visual Ingestion for Visual-Dense Document Question Answering
- **分类: cs.CL; cs.CV; cs.IR**

- **简介: 该论文属于文档问答任务，针对视觉密集工程文档的问答问题，提出DVI框架，通过延迟视觉处理提升准确率。**

- **链接: [https://arxiv.org/pdf/2602.14162v2](https://arxiv.org/pdf/2602.14162v2)**

> **作者:** Tao Xu
>
> **备注:** 24 pages, 4 figures, 7 tables
>
> **摘要:** Existing multimodal document question answering methods predominantly adopt a Pre-Ingestion (PI) strategy: during the indexing phase, a Vision Language Model (VLM) is called on every page to generate page descriptions that are then encoded into vectors, and questions are answered via embedding similarity retrieval. However, this approach faces a dual dilemma on visual-dense engineering documents: VLM blind descriptions inevitably lose critical visual details, and embedding retrieval systematically fails on highly similar documents. This paper proposes the Deferred Visual Ingestion (DVI) framework: zero VLM calls during preprocessing, leveraging only document structural information (table of contents, drawing numbers) to automatically build a hierarchical index through the HDNC (Hierarchical Drawing Number Clustering) algorithm; during inference, candidate pages are located via BM25 retrieval, and the original images along with the specific question are sent to a VLM for targeted analysis. Large-scale experiments on three datasets validate the effectiveness of DVI: on Bridge engineering drawings (1,323 questions), end-to-end QA accuracy reaches 65.6\% vs. PI's 24.3\% (+41.3pp); on Steel catalog (186 questions), 30.6\% vs. 16.1\% (+14.5pp); on CircuitVQA, a public benchmark (9,315 questions), retrieval ImgR@3 achieves 31.2\% vs. 0.7\%. On the Bridge dataset, we evaluated ColPali (ICLR 2025 visual retrieval SOTA), which achieved only 20.1\% PageR@3, demonstrating that the failure of embedding retrieval on homogeneous engineering documents is structural rather than due to insufficient model capability. Ablation studies show that HDNC zero-cost automatic indexing yields a +27.5pp retrieval improvement, and VLM conversion rate analysis confirms that the bottleneck lies on the retrieval side rather than the comprehension side.
>
---
#### [replaced 044] PARL: Prompt-based Agents for Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文提出PARL，将大语言模型作为强化学习代理，通过提示进行训练，解决非语言推理任务中的RL问题。**

- **链接: [https://arxiv.org/pdf/2510.21306v2](https://arxiv.org/pdf/2510.21306v2)**

> **作者:** Yarik Menchaca Resendiz; Roman Klinger
>
> **摘要:** Large language models (LLMs) have demonstrated high performance on tasks expressed in natural language, particularly in zero- or few-shot settings. These are typically framed as supervised (e.g., classification) or unsupervised (e.g., clustering) problems. However, limited work evaluates LLMs as agents in reinforcement learning (RL) tasks (e.g., playing games), where learning occurs through interaction with an environment and a reward system. While prior work focused on representing tasks that rely on a language representation, we study structured, non-linguistic reasoning - such as interpreting positions in a grid world. We therefore introduce PARL (Prompt-based Agent for Reinforcement Learning), a method that uses LLMs as RL agents through prompting, without any fine-tuning. PARL encodes actions, states, and rewards in the prompt, enabling the model to learn through trial-and-error interaction. We evaluate PARL on three standard RL tasks that do not entirely rely on natural language. We show that it can match or outperform traditional RL agents in simple environments by leveraging pretrained knowledge. However, we identify performance limitations in tasks that require complex mathematical operations or decoding states and actions.
>
---
#### [replaced 045] Can LLMs Simulate Human Behavioral Variability? A Case Study in the Phonemic Fluency Task
- **分类: cs.CL**

- **简介: 该论文属于认知任务研究，探讨LLMs能否模拟人类行为差异。通过 phonemic fluency 任务，比较模型与人类表现，发现LLMs无法再现人类多样性。**

- **链接: [https://arxiv.org/pdf/2505.16164v2](https://arxiv.org/pdf/2505.16164v2)**

> **作者:** Mengyang Qiu; Zoe Brisebois; Siena Sun
>
> **摘要:** Large language models (LLMs) are increasingly explored as substitutes for human participants in cognitive tasks, but their ability to simulate human behavioral variability remains unclear. This study examines whether LLMs can approximate individual differences in the phonemic fluency task, where participants generate words beginning with a target letter. We evaluated 34 distinct models across 45 configurations from major closed-source and open-source providers, and compared outputs to responses from 106 human participants. While some models, especially Claude 3.7 Sonnet, approximated human averages and lexical preferences, none reproduced the scope of human variability. LLM outputs were consistently less diverse, with newer models and thinking-enabled modes often reducing rather than increasing variability. Network analysis further revealed fundamental differences in retrieval structure between humans and the most human-like model. Ensemble simulations combining outputs from diverse models also failed to recover human-level diversity, likely due to high vocabulary overlap across models. These results highlight key limitations in using LLMs to simulate human cognition and behavior.
>
---
