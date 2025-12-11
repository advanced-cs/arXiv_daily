# 自然语言处理 cs.CL

- **最新发布 44 篇**

- **更新 34 篇**

## 最新发布

#### [new 001] Can LLMs Evaluate What They Cannot Annotate? Revisiting LLM Reliability in Hate Speech Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型（LLM）在仇恨言论检测中的评估可靠性。针对主观性导致的标注分歧，提出用xRR框架分析，发现LLM虽与人类标注不一致，但能可靠反映模型性能排序，可作为主观NLP任务的可扩展代理评估工具。**

- **链接: [https://arxiv.org/pdf/2512.09662v1](https://arxiv.org/pdf/2512.09662v1)**

> **作者:** Paloma Piot; David Otero; Patricia Martín-Rodilla; Javier Parapar
>
> **摘要:** Hate speech spreads widely online, harming individuals and communities, making automatic detection essential for large-scale moderation, yet detecting it remains difficult. Part of the challenge lies in subjectivity: what one person flags as hate speech, another may see as benign. Traditional annotation agreement metrics, such as Cohen's $κ$, oversimplify this disagreement, treating it as an error rather than meaningful diversity. Meanwhile, Large Language Models (LLMs) promise scalable annotation, but prior studies demonstrate that they cannot fully replace human judgement, especially in subjective tasks. In this work, we reexamine LLM reliability using a subjectivity-aware framework, cross-Rater Reliability (xRR), revealing that even under fairer lens, LLMs still diverge from humans. Yet this limitation opens an opportunity: we find that LLM-generated annotations can reliably reflect performance trends across classification models, correlating with human evaluations. We test this by examining whether LLM-generated annotations preserve the relative ordering of model performance derived from human evaluation (i.e. whether models ranked as more reliable by human annotators preserve the same order when evaluated with LLM-generated labels). Our results show that, although LLMs differ from humans at the instance level, they reproduce similar ranking and classification patterns, suggesting their potential as proxy evaluators. While not a substitute for human annotators, they might serve as a scalable proxy for evaluation in subjective NLP tasks.
>
---
#### [new 002] Knowledge-Augmented Large Language Model Agents for Explainable Financial Decision-Making
- **分类: cs.CL**

- **简介: 该论文研究可解释的金融决策任务，解决传统方法知识依赖强、事实不一致、推理链缺失问题。提出融合外部知识检索与语义表示的增强框架，引入多头注意力构建逻辑链，联合优化预测与解释，提升决策准确性与透明度。**

- **链接: [https://arxiv.org/pdf/2512.09440v1](https://arxiv.org/pdf/2512.09440v1)**

> **作者:** Qingyuan Zhang; Yuxi Wang; Cancan Hua; Yulin Huang; Ning Lyu
>
> **摘要:** This study investigates an explainable reasoning method for financial decision-making based on knowledge-enhanced large language model agents. To address the limitations of traditional financial decision methods that rely on parameterized knowledge, lack factual consistency, and miss reasoning chains, an integrated framework is proposed that combines external knowledge retrieval, semantic representation, and reasoning generation. The method first encodes financial texts and structured data to obtain semantic representations, and then retrieves task-related information from external knowledge bases using similarity computation. Internal representations and external knowledge are combined through weighted fusion, which ensures fluency while improving factual accuracy and completeness of generated content. In the reasoning stage, a multi-head attention mechanism is introduced to construct logical chains, allowing the model to present transparent causal relationships and traceability during generation. Finally, the model jointly optimizes task objectives and explanation consistency objectives, which enhances predictive performance and reasoning interpretability. Experiments on financial text processing and decision tasks show that the method outperforms baseline approaches in accuracy, text generation quality, and factual support, verifying the effectiveness of knowledge enhancement and explainable reasoning. Overall, the proposed approach overcomes the limitations of traditional models in semantic coverage and reasoning transparency, and demonstrates strong practical value in complex financial scenarios.
>
---
#### [new 003] System Report for CCL25-Eval Task 10: Prompt-Driven Large Language Model Merge for Fine-Grained Chinese Hate Speech Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对中文细粒度仇恨言论检测任务，提出三阶段大模型框架：提示工程、监督微调与模型融合，解决传统方法难以捕捉语境和新变体的问题，在STATE-ToxiCN基准上验证了有效性。**

- **链接: [https://arxiv.org/pdf/2512.09563v1](https://arxiv.org/pdf/2512.09563v1)**

> **作者:** Binglin Wu; Jiaxiu Zou; Xianneng Li
>
> **备注:** Accepted at CCL 2025
>
> **摘要:** The proliferation of hate speech on Chinese social media poses urgent societal risks, yet traditional systems struggle to decode context-dependent rhetorical strategies and evolving slang. To bridge this gap, we propose a novel three-stage LLM-based framework: Prompt Engineering, Supervised Fine-tuning, and LLM Merging. First, context-aware prompts are designed to guide LLMs in extracting implicit hate patterns. Next, task-specific features are integrated during supervised fine-tuning to enhance domain adaptation. Finally, merging fine-tuned LLMs improves robustness against out-of-distribution cases. Evaluations on the STATE-ToxiCN benchmark validate the framework's effectiveness, demonstrating superior performance over baseline methods in detecting fine-grained hate speech.
>
---
#### [new 004] ChronusOmni: Improving Time Awareness of Omni Large Language Models
- **分类: cs.CL; cs.CV; cs.MM**

- **简介: 该论文聚焦多模态大模型的时间感知任务，解决现有方法在音频利用和跨模态隐式时序定位上的不足。提出ChronusOmni模型，通过文本化时间戳和强化学习提升显式与隐式音视频时序理解，并构建新数据集ChronusAV验证其优越性。**

- **链接: [https://arxiv.org/pdf/2512.09841v1](https://arxiv.org/pdf/2512.09841v1)**

> **作者:** Yijing Chen; Yihan Wu; Kaisi Guan; Yuchen Ren; Yuyue Wang; Ruihua Song; Liyun Ru
>
> **备注:** Code available at https://github.com/YJCX330/Chronus/
>
> **摘要:** Time awareness is a fundamental ability of omni large language models, especially for understanding long videos and answering complex questions. Previous approaches mainly target vision-language scenarios and focus on the explicit temporal grounding questions, such as identifying when a visual event occurs or determining what event happens at aspecific time. However, they often make insufficient use of the audio modality, and overlook implicit temporal grounding across modalities--for example, identifying what is visually present when a character speaks, or determining what is said when a visual event occurs--despite such cross-modal temporal relations being prevalent in real-world scenarios. In this paper, we propose ChronusOmni, an omni large language model designed to enhance temporal awareness for both explicit and implicit audiovisual temporal grounding. First, we interleave text-based timestamp tokens with visual and audio representations at each time unit, enabling unified temporal modeling across modalities. Second, to enforce correct temporal ordering and strengthen fine-grained temporal reasoning, we incorporate reinforcement learning with specially designed reward functions. Moreover, we construct ChronusAV, a temporally-accurate, modality-complete, and cross-modal-aligned dataset to support the training and evaluation on audiovisual temporal grounding task. Experimental results demonstrate that ChronusOmni achieves state-of-the-art performance on ChronusAV with more than 30% improvement and top results on most metrics upon other temporal grounding benchmarks. This highlights the strong temporal awareness of our model across modalities, while preserving general video and audio understanding capabilities.
>
---
#### [new 005] CONCUR: A Framework for Continual Constrained and Unconstrained Routing
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究任务路由问题，旨在为不同AI任务选择最优计算策略。针对现有方法难以持续扩展、泛化性差的问题，提出CONCUR框架，采用模块化设计，为每种策略训练独立预测器，并融合多表示输入，支持持续学习与预算约束下的高效路由。**

- **链接: [https://arxiv.org/pdf/2512.09386v1](https://arxiv.org/pdf/2512.09386v1)**

> **作者:** Peter Baile Chen; Weiyue Li; Dan Roth; Michael Cafarella; Samuel Madden; Jacob Andreas
>
> **摘要:** AI tasks differ in complexity and are best addressed with different computation strategies (e.g., combinations of models and decoding methods). Hence, an effective routing system that maps tasks to the appropriate strategies is crucial. Most prior methods build the routing framework by training a single model across all strategies, which demands full retraining whenever new strategies appear and leads to high overhead. Attempts at such continual routing, however, often face difficulties with generalization. Prior models also typically use a single input representation, limiting their ability to capture the full complexity of the routing problem and leading to sub-optimal routing decisions. To address these gaps, we propose CONCUR, a continual routing framework that supports both constrained and unconstrained routing (i.e., routing with or without a budget). Our modular design trains a separate predictor model for each strategy, enabling seamless incorporation of new strategies with low additional training cost. Our predictors also leverage multiple representations of both tasks and computation strategies to better capture overall problem complexity. Experiments on both in-distribution and out-of-distribution, knowledge- and reasoning-intensive tasks show that our method outperforms the best single strategy and strong existing routing techniques with higher end-to-end accuracy and lower inference cost in both continual and non-continual settings, while also reducing training cost in the continual setting.
>
---
#### [new 006] d-TreeRPO: Towards More Reliable Policy Optimization for Diffusion Language Models
- **分类: cs.CL**

- **简介: 该论文研究扩散语言模型的可靠策略优化，提出d-TreeRPO框架。通过树结构 rollout 和自底向上优势计算，结合可验证奖励与时间调度的自蒸馏损失，提升优势估计和概率预测准确性，解决现有方法在奖励信号和概率估计上的偏差问题，显著提升推理性能。**

- **链接: [https://arxiv.org/pdf/2512.09675v1](https://arxiv.org/pdf/2512.09675v1)**

> **作者:** Leyi Pan; Shuchang Tao; Yunpeng Zhai; Zheyu Fu; Liancheng Fang; Minghua He; Lingzhe Zhang; Zhaoyang Liu; Bolin Ding; Aiwei Liu; Lijie Wen
>
> **备注:** 16 pages, 5 figures, 3tables
>
> **摘要:** Reliable reinforcement learning (RL) for diffusion large language models (dLLMs) requires both accurate advantage estimation and precise estimation of prediction probabilities. Existing RL methods for dLLMs fall short in both aspects: they rely on coarse or unverifiable reward signals, and they estimate prediction probabilities without accounting for the bias relative to the true, unbiased expected prediction probability that properly integrates over all possible decoding orders. To mitigate these issues, we propose \emph{d}-TreeRPO, a reliable RL framework for dLLMs that leverages tree-structured rollouts and bottom-up advantage computation based on verifiable outcome rewards to provide fine-grained and verifiable step-wise reward signals. When estimating the conditional transition probability from a parent node to a child node, we theoretically analyze the estimation error between the unbiased expected prediction probability and the estimate obtained via a single forward pass, and find that higher prediction confidence leads to lower estimation error. Guided by this analysis, we introduce a time-scheduled self-distillation loss during training that enhances prediction confidence in later training stages, thereby enabling more accurate probability estimation and improved convergence. Experiments show that \emph{d}-TreeRPO outperforms existing baselines and achieves significant gains on multiple reasoning benchmarks, including +86.2 on Sudoku, +51.6 on Countdown, +4.5 on GSM8K, and +5.3 on Math500. Ablation studies and computational cost analyses further demonstrate the effectiveness and practicality of our design choices.
>
---
#### [new 007] FineFreq: A Multilingual Character Frequency Dataset from Web-Scale Text
- **分类: cs.CL**

- **简介: 该论文构建了FineFreq，一个覆盖1900多种语言的大规模多语言字符频次数据集。基于Web级文本，提供字符级频率及时间维度统计，保留自然语言特征，并附Unicode元数据，支持细粒度分析与过滤，以促进多语言基础研究。**

- **链接: [https://arxiv.org/pdf/2512.09701v1](https://arxiv.org/pdf/2512.09701v1)**

> **作者:** Binbin XU
>
> **摘要:** We present FineFreq, a large-scale multilingual character frequency dataset derived from the FineWeb and FineWeb2 corpora, covering over 1900 languages and spanning 2013-2025. The dataset contains frequency counts for 96 trillion characters processed from 57 TB of compressed text. For each language, FineFreq provides per-character statistics with aggregate and year-level frequencies, allowing fine-grained temporal analysis. The dataset preserves naturally occurring multilingual features such as cross-script borrowings, emoji, and acronyms without applying artificial filtering. Each character entry includes Unicode metadata (category, script, block), enabling domain-specific or other downstream filtering and analysis. The full dataset is released in both CSV and Parquet formats, with associated metadata, available on GitHub and HuggingFace. https://github.com/Bin-2/FineFreq
>
---
#### [new 008] MOA: Multi-Objective Alignment for Role-Playing Agents
- **分类: cs.CL**

- **简介: 该论文研究角色扮演智能体（RPA）的多目标对齐任务，旨在解决现有方法在多维度优化、多样性与质量间的平衡问题。提出MOA框架，结合多目标强化学习与思维增强 rollout，实现细粒度评估指标下的综合提升。**

- **链接: [https://arxiv.org/pdf/2512.09756v1](https://arxiv.org/pdf/2512.09756v1)**

> **作者:** Chonghua Liao; Ke Wang; Yuchuan Wu; Fei Huang; Yongbin Li
>
> **摘要:** Role-playing agents (RPAs) must simultaneously master many conflicting skills -- following multi-turn instructions, exhibiting domain knowledge, and adopting a consistent linguistic style. Existing work either relies on supervised fine-tuning (SFT) that over-fits surface cues and yields low diversity, or applies reinforcement learning (RL) that fails to learn multiple dimensions for comprehensive RPA optimization. We present MOA (Multi-Objective Alignment), a reinforcement-learning framework that enables multi-dimensional, fine-grained rubric optimization for general RPAs. MOA introduces a novel multi-objective optimization strategy that trains simultaneously on multiple fine-grained rubrics to boost optimization performance. Besides, to address the issues of model output diversity and quality, we have also employed thought-augmented rollout with off-policy guidance. Extensive experiments on challenging benchmarks such as PersonaGym and RoleMRC show that MOA enables an 8B model to match or even outperform strong baselines such as GPT-4o and Claude across numerous dimensions. This demonstrates the great potential of MOA in building RPAs that can simultaneously meet the demands of role knowledge, persona style, diverse scenarios, and complex multi-turn conversations.
>
---
#### [new 009] Efficient Continual Learning in Neural Machine Translation: A Low-Rank Adaptation Approach
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究神经机器翻译中的持续学习任务，旨在解决灾难性遗忘和高计算成本问题。提出基于低秩适应（LoRA）的参数高效微调方法，设计交互式适配机制与面向低秩矩阵的梯度正则化策略，实现可扩展、可交互的持续学习。**

- **链接: [https://arxiv.org/pdf/2512.09910v1](https://arxiv.org/pdf/2512.09910v1)**

> **作者:** Salvador Carrión; Francisco Casacuberta
>
> **摘要:** Continual learning in Neural Machine Translation (NMT) faces the dual challenges of catastrophic forgetting and the high computational cost of retraining. This study establishes Low-Rank Adaptation (LoRA) as a parameter-efficient framework to address these challenges in dedicated NMT architectures. We first demonstrate that LoRA-based fine-tuning adapts NMT models to new languages and domains with performance on par with full-parameter techniques, while utilizing only a fraction of the parameter space. Second, we propose an interactive adaptation method using a calibrated linear combination of LoRA modules. This approach functions as a gate-free mixture of experts, enabling real-time, user-controllable adjustments to domain and style without retraining. Finally, to mitigate catastrophic forgetting, we introduce a novel gradient-based regularization strategy specifically designed for low-rank decomposition matrices. Unlike methods that regularize the full parameter set, our approach weights the penalty on the low-rank updates using historical gradient information. Experimental results indicate that this strategy efficiently preserves prior domain knowledge while facilitating the acquisition of new tasks, offering a scalable paradigm for interactive and continual NMT.
>
---
#### [new 010] Knowledge-Guided Large Language Model for Automatic Pediatric Dental Record Understanding and Safe Antibiotic Recommendation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出知识引导的大语言模型（KG-LLM），用于儿科牙科记录理解与安全抗生素推荐。针对非结构化文本和用药安全难题，融合知识图谱、检索增强生成与双层安全验证机制，提升记录解析精度与推荐安全性。**

- **链接: [https://arxiv.org/pdf/2512.09127v1](https://arxiv.org/pdf/2512.09127v1)**

> **作者:** Zihan Han; Junyan Ge; Caifeng Li
>
> **摘要:** Accurate interpretation of pediatric dental clinical records and safe antibiotic prescribing remain persistent challenges in dental informatics. Traditional rule-based clinical decision support systems struggle with unstructured dental narratives, incomplete radiographic descriptions, and complex safety constraints. To address these limitations, this study proposes a Knowledge-Guided Large Language Model (KG-LLM) that integrates a pediatric dental knowledge graph, retrieval-augmented generation (RAG), and a multi-stage safety validation pipeline for evidence-grounded antibiotic recommendation. The framework first employs a clinical NER/RE module to extract structured entities and relations from dental notes and radiology reports. Relevant guidelines, drug-safety rules, and analogous historical cases are subsequently retrieved from the knowledge graph and supplied to the LLM for diagnostic summarization and dose-drug-duration prediction. Safety assurance is achieved through a dual-layer validation mechanism combining deterministic rule checking with a learned classifier for detecting allergies, contraindications, and dosing errors. Experiments on 32,000 de-identified pediatric dental visit records demonstrate the effectiveness of the proposed approach. Compared with a domain-adapted Llama-2 clinical baseline, KG-LLM improves record-understanding performance (F1: 0.914 vs. 0.867), drug-dose-duration accuracy (Top-1: 0.782 vs. 0.716), and reduces unsafe antibiotic suggestions by 50%. Additional evaluation across summary quality, recommendation accuracy, and global safety scores further confirms the robustness of the system. Ablation analyses indicate that the knowledge graph, RAG, and safety modules each contribute substantially to clinical reliability and interpretability.
>
---
#### [new 011] DeepSeek's WEIRD Behavior: The cultural alignment of Large Language Models and the effects of prompt language and cultural prompting
- **分类: cs.CL**

- **简介: 该论文研究大语言模型的文化对齐问题，旨在分析模型在不同文化背景下的响应倾向。通过 Hofstede 文化维度理论，结合提示语言与文化提示策略，评估 DeepSeek、GPT 等模型对中美文化的对齐程度，并探讨调整方法的有效性。**

- **链接: [https://arxiv.org/pdf/2512.09772v1](https://arxiv.org/pdf/2512.09772v1)**

> **作者:** James Luther; Donald Brown
>
> **摘要:** Culture is a core component of human-to-human interaction and plays a vital role in how we perceive and interact with others. Advancements in the effectiveness of Large Language Models (LLMs) in generating human-sounding text have greatly increased the amount of human-to-computer interaction. As this field grows, the cultural alignment of these human-like agents becomes an important field of study. Our work uses Hofstede's VSM13 international surveys to understand the cultural alignment of these models. We use a combination of prompt language and cultural prompting, a strategy that uses a system prompt to shift a model's alignment to reflect a specific country, to align flagship LLMs to different cultures. Our results show that DeepSeek-V3, V3.1, and OpenAI's GPT-5 exhibit a close alignment with the survey responses of the United States and do not achieve a strong or soft alignment with China, even when using cultural prompts or changing the prompt language. We also find that GPT-4 exhibits an alignment closer to China when prompted in English, but cultural prompting is effective in shifting this alignment closer to the United States. Other low-cost models, GPT-4o and GPT-4.1, respond to the prompt language used (i.e., English or Simplified Chinese) and cultural prompting strategies to create acceptable alignments with both the United States and China.
>
---
#### [new 012] Interpreto: An Explainability Library for Transformers
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出Interpreto，一个面向Transformer模型的可解释性Python库。它解决HuggingFace模型（从BERT到大语言模型）的文本解释问题，提供归因和基于概念的两种解释方法，支持分类与生成任务，旨在连接前沿研究与实用工具，提升模型透明度和用户理解。**

- **链接: [https://arxiv.org/pdf/2512.09730v1](https://arxiv.org/pdf/2512.09730v1)**

> **作者:** Antonin Poché; Thomas Mullor; Gabriele Sarti; Frédéric Boisnard; Corentin Friedrich; Charlotte Claye; François Hoofd; Raphael Bernas; Céline Hudelot; Fanny Jourdan
>
> **备注:** Equal contribution: Poché and Jourdan
>
> **摘要:** Interpreto is a Python library for post-hoc explainability of text HuggingFace models, from early BERT variants to LLMs. It provides two complementary families of methods: attributions and concept-based explanations. The library connects recent research to practical tooling for data scientists, aiming to make explanations accessible to end users. It includes documentation, examples, and tutorials. Interpreto supports both classification and generation models through a unified API. A key differentiator is its concept-based functionality, which goes beyond feature-level attributions and is uncommon in existing libraries. The library is open source; install via pip install interpreto. Code and documentation are available at https://github.com/FOR-sight-ai/interpreto.
>
---
#### [new 013] Systematic Framework of Application Methods for Large Language Models in Language Sciences
- **分类: cs.CL**

- **简介: 该论文属于方法论研究，旨在解决大语言模型在语言科学中应用的方法碎片化问题。提出两个框架：一是按研究目标选择方法的指南，二是多阶段研究流程的系统配置，通过案例和实验验证其有效性，推动语言科学研究的系统性与可重复性。**

- **链接: [https://arxiv.org/pdf/2512.09552v1](https://arxiv.org/pdf/2512.09552v1)**

> **作者:** Kun Sun; Rong Wang
>
> **摘要:** Large Language Models (LLMs) are transforming language sciences. However, their widespread deployment currently suffers from methodological fragmentation and a lack of systematic soundness. This study proposes two comprehensive methodological frameworks designed to guide the strategic and responsible application of LLMs in language sciences. The first method-selection framework defines and systematizes three distinct, complementary approaches, each linked to a specific research goal: (1) prompt-based interaction with general-use models for exploratory analysis and hypothesis generation; (2) fine-tuning of open-source models for confirmatory, theory-driven investigation and high-quality data generation; and (3) extraction of contextualized embeddings for further quantitative analysis and probing of model internal mechanisms. We detail the technical implementation and inherent trade-offs of each method, supported by empirical case studies. Based on the method-selection framework, the second systematic framework proposed provides constructed configurations that guide the practical implementation of multi-stage research pipelines based on these approaches. We then conducted a series of empirical experiments to validate our proposed framework, employing retrospective analysis, prospective application, and an expert evaluation survey. By enforcing the strategic alignment of research questions with the appropriate LLM methodology, the frameworks enable a critical paradigm shift in language science research. We believe that this system is fundamental for ensuring reproducibility, facilitating the critical evaluation of LLM mechanisms, and providing the structure necessary to move traditional linguistics from ad-hoc utility to verifiable, robust science.
>
---
#### [new 014] OnCoCo 1.0: A Public Dataset for Fine-Grained Message Classification in Online Counseling Conversations
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理中的细粒度对话分类任务，旨在解决现有心理辅导对话数据集类别单一、依赖面诊数据的问题。作者构建了包含38类咨询师和28类来访者语句的新标注数据集OnCoCo 1.0，并基于其训练模型，推动在线心理辅导对话的自动分析。**

- **链接: [https://arxiv.org/pdf/2512.09804v1](https://arxiv.org/pdf/2512.09804v1)**

> **作者:** Jens Albrecht; Robert Lehmann; Aleksandra Poltermann; Eric Rudolph; Philipp Steigerwald; Mara Stieler
>
> **备注:** Submitted to LREC 2026
>
> **摘要:** This paper presents OnCoCo 1.0, a new public dataset for fine-grained message classification in online counseling. It is based on a new, integrative system of categories, designed to improve the automated analysis of psychosocial online counseling conversations. Existing category systems, predominantly based on Motivational Interviewing (MI), are limited by their narrow focus and dependence on datasets derived mainly from face-to-face counseling. This limits the detailed examination of textual counseling conversations. In response, we developed a comprehensive new coding scheme that differentiates between 38 types of counselor and 28 types of client utterances, and created a labeled dataset consisting of about 2.800 messages from counseling conversations. We fine-tuned several models on our dataset to demonstrate its applicability. The data and models are publicly available to researchers and practitioners. Thus, our work contributes a new type of fine-grained conversational resource to the language resources community, extending existing datasets for social and mental-health dialogue analysis.
>
---
#### [new 015] CourtPressGER: A German Court Decision to Press Release Summarization Dataset
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦司法文本摘要生成任务，旨在解决德国法院判决向公众通俗传达的问题。作者构建了CourtPressGER数据集，包含判决、人工撰写的新闻稿及合成提示，用于训练和评估大语言模型生成准确、易懂的判决摘要，并通过多维度评测比较不同模型性能。**

- **链接: [https://arxiv.org/pdf/2512.09434v1](https://arxiv.org/pdf/2512.09434v1)**

> **作者:** Sebastian Nagl; Mohamed Elganayni; Melanie Pospisil; Matthias Grabmair
>
> **备注:** Preprint - This contribution was accepted at JURIX AI4A2J Workshop 2025
>
> **摘要:** Official court press releases from Germany's highest courts present and explain judicial rulings to the public, as well as to expert audiences. Prior NLP efforts emphasize technical headnotes, ignoring citizen-oriented communication needs. We introduce CourtPressGER, a 6.4k dataset of triples: rulings, human-drafted press releases, and synthetic prompts for LLMs to generate comparable releases. This benchmark trains and evaluates LLMs in generating accurate, readable summaries from long judicial texts. We benchmark small and large LLMs using reference-based metrics, factual-consistency checks, LLM-as-judge, and expert ranking. Large LLMs produce high-quality drafts with minimal hierarchical performance loss; smaller models require hierarchical setups for long judgments. Initial benchmarks show varying model performance, with human-drafted releases ranking highest.
>
---
#### [new 016] Source Coverage and Citation Bias in LLM-based vs. Traditional Search Engines
- **分类: cs.CL; cs.CY**

- **简介: 该论文研究LLM搜索与传统搜索引擎在来源覆盖和引用偏差上的差异。通过分析5.5万查询，发现LLM系统引用更多样但未在可信度等方面更优，并探究其源选择机制。**

- **链接: [https://arxiv.org/pdf/2512.09483v1](https://arxiv.org/pdf/2512.09483v1)**

> **作者:** Peixian Zhang; Qiming Ye; Zifan Peng; Kiran Garimella; Gareth Tyson
>
> **摘要:** LLM-based Search Engines (LLM-SEs) introduces a new paradigm for information seeking. Unlike Traditional Search Engines (TSEs) (e.g., Google), these systems summarize results, often providing limited citation transparency. The implications of this shift remain largely unexplored, yet raises key questions regarding trust and transparency. In this paper, we present a large-scale empirical study of LLM-SEs, analyzing 55,936 queries and the corresponding search results across six LLM-SEs and two TSEs. We confirm that LLM-SEs cites domain resources with greater diversity than TSEs. Indeed, 37% of domains are unique to LLM-SEs. However, certain risks still persist: LLM-SEs do not outperform TSEs in credibility, political neutrality and safety metrics. Finally, to understand the selection criteria of LLM-SEs, we perform a feature-based analysis to identify key factors influencing source choice. Our findings provide actionable insights for end users, website owners, and developers.
>
---
#### [new 017] Neurosymbolic Information Extraction from Transactional Documents
- **分类: cs.CL**

- **简介: 该论文研究交易文档的信息抽取任务，旨在解决零样本场景下抽取结果不准确的问题。提出一种结合语言模型与符号验证的神经符号框架，通过多级验证确保符合领域约束，提升了F1分数与准确率。**

- **链接: [https://arxiv.org/pdf/2512.09666v1](https://arxiv.org/pdf/2512.09666v1)**

> **作者:** Arthur Hemmer; Mickaël Coustaty; Nicola Bartolo; Jean-Marc Ogier
>
> **备注:** 20 pages, 2 figures, accepted to IJDAR (ICDAR 2025)
>
> **摘要:** This paper presents a neurosymbolic framework for information extraction from documents, evaluated on transactional documents. We introduce a schema-based approach that integrates symbolic validation methods to enable more effective zero-shot output and knowledge distillation. The methodology uses language models to generate candidate extractions, which are then filtered through syntactic-, task-, and domain-level validation to ensure adherence to domain-specific arithmetic constraints. Our contributions include a comprehensive schema for transactional documents, relabeled datasets, and an approach for generating high-quality labels for knowledge distillation. Experimental results demonstrate significant improvements in $F_1$-scores and accuracy, highlighting the effectiveness of neurosymbolic validation in transactional document processing.
>
---
#### [new 018] Mitigating Social Bias in English and Urdu Language Models Using PRM-Guided Candidate Selection and Sequential Refinement
- **分类: cs.CL**

- **简介: 该论文属自然语言处理中的公平性研究，旨在缓解英语和乌尔都语大模型的社交偏见。提出基于偏好排序模型的候选选择与序列优化方法，在不重训练情况下减少生成偏见，提升低资源语言公平性。**

- **链接: [https://arxiv.org/pdf/2512.09854v1](https://arxiv.org/pdf/2512.09854v1)**

> **作者:** Muneeb Ur Raheem Khan
>
> **摘要:** Large language models (LLMs) increasingly mediate human communication, decision support, content creation, and information retrieval. Despite impressive fluency, these systems frequently produce biased or stereotypical content, especially when prompted with socially sensitive language. A growing body of research has demonstrated that such biases disproportionately affect low-resource languages, where training data is limited and culturally unrepresentative. This paper presents a comprehensive study of inference-time bias mitigation, a strategy that avoids retraining or fine-tuning and instead operates directly on model outputs. Building on preference-ranking models (PRMs), we introduce a unified evaluation framework comparing three methods: (1) baseline single-word generation, (2) PRM-Select best-of-N sampling, and (3) PRM-Sequential refinement guided by PRM critiques. We evaluate these techniques across 200 English prompts and their Urdu counterparts, designed to reflect socio-cultural contexts relevant to gender, ethnicity, religion, nationality, disability, profession, age, and socioeconomic categories. Using GPT-3.5 as a candidate generator and GPT-4o-mini as a PRM-based bias and utility scorer, we provide an extensive quantitative analysis of bias reduction, utility preservation, and cross-lingual disparities. Our findings show: (a) substantial gains over the baseline for both languages; (b) consistently lower fairness scores for Urdu across all methods, highlighting structural inequities in multilingual LLM training; and (c) distinct improvement trajectories between PRM-Select and PRM-Sequential. The study contributes an extensible methodology, interpretable metrics, and cross-lingual comparisons that can support future work on fairness evaluation in low-resource languages.
>
---
#### [new 019] The Linguistic Architecture of Reflective Thought: Evaluation of a Large Language Model as a Tool to Isolate the Formal Structure of Mentalization
- **分类: cs.CL; cs.AI**

- **简介: 该论文探讨大语言模型能否模拟心理化（mentalization）的语言结构。基于MBT框架，通过精神病学家对LLM生成对话的评估，分析其在心理化维度上的表现，发现模型在部分维度稳定且具临床可解释性，但情感中性。**

- **链接: [https://arxiv.org/pdf/2512.08945v1](https://arxiv.org/pdf/2512.08945v1)**

> **作者:** Stefano Epifani; Giuliano Castigliego; Laura Kecskemeti; Giuliano Razzicchia; Elisabeth Seiwald-Sonderegger
>
> **备注:** 18 pages, 1 table, Project coordinator: Stefano Epifani
>
> **摘要:** Background: Mentalization integrates cognitive, affective, and intersubjective components. Large Language Models (LLMs) display an increasing ability to generate reflective texts, raising questions regarding the relationship between linguistic form and mental representation. This study assesses the extent to which a single LLM can reproduce the linguistic structure of mentalization according to the parameters of Mentalization-Based Treatment (MBT). Methods: Fifty dialogues were generated between human participants and an LLM configured in standard mode. Five psychiatrists trained in MBT, working under blinded conditions, evaluated the mentalization profiles produced by the model along the four MBT axes, assigning Likert-scale scores for evaluative coherence, argumentative coherence, and global quality. Inter-rater agreement was estimated using ICC(3,1). Results: Mean scores (3.63-3.98) and moderate standard deviations indicate a high level of structural coherence in the generated profiles. ICC values (0.60-0.84) show substantial-to-high agreement among raters. The model proved more stable in the Implicit-Explicit and Self-Other dimensions, while presenting limitations in the integration of internal states and external contexts. The profiles were coherent and clinically interpretable yet characterized by affective neutrality.
>
---
#### [new 020] MentraSuite: Post-Training Large Language Models for Mental Health Reasoning and Assessment
- **分类: cs.CL**

- **简介: 该论文聚焦心理健康的推理与评估任务，旨在提升大模型在临床对齐推理上的可靠性。作者提出MentraSuite框架，包括评测基准MentraBench和优化模型Mindora，通过后训练与一致性奖励机制增强推理的连贯性与准确性。**

- **链接: [https://arxiv.org/pdf/2512.09636v1](https://arxiv.org/pdf/2512.09636v1)**

> **作者:** Mengxi Xiao; Kailai Yang; Pengde Zhao; Enze Zhang; Ziyan Kuang; Zhiwei Liu; Weiguang Han; Shu Liao; Lianting Huang; Jinpeng Hu; Min Peng; Qianqian Xie; Sophia Ananiadou
>
> **摘要:** Mental health disorders affect hundreds of millions globally, and the Web now serves as a primary medium for accessing support, information, and assessment. Large language models (LLMs) offer scalable and accessible assistance, yet their deployment in mental-health settings remains risky when their reasoning is incomplete, inconsistent, or ungrounded. Existing psychological LLMs emphasize emotional understanding or knowledge recall but overlook the step-wise, clinically aligned reasoning required for appraisal, diagnosis, intervention planning, abstraction, and verification. To address these issues, we introduce MentraSuite, a unified framework for advancing reliable mental-health reasoning. We propose MentraBench, a comprehensive benchmark spanning five core reasoning aspects, six tasks, and 13 datasets, evaluating both task performance and reasoning quality across five dimensions: conciseness, coherence, hallucination avoidance, task understanding, and internal consistency. We further present Mindora, a post-trained model optimized through a hybrid SFT-RL framework with an inconsistency-detection reward to enforce faithful and coherent reasoning. To support training, we construct high-quality trajectories using a novel reasoning trajectory generation strategy, that strategically filters difficult samples and applies a structured, consistency-oriented rewriting process to produce concise, readable, and well-balanced trajectories. Across 20 evaluated LLMs, Mindora achieves the highest average performance on MentraBench and shows remarkable performances in reasoning reliability, demonstrating its effectiveness for complex mental-health scenarios.
>
---
#### [new 021] Luxical: High-Speed Lexical-Dense Text Embeddings
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出Luxical，一种高速文本嵌入方法，旨在解决现有工具在速度与灵活性间的权衡问题。通过结合稀疏特征、小型网络与知识蒸馏，逼近大模型性能，显著提升计算效率，适用于大规模文本组织任务。**

- **链接: [https://arxiv.org/pdf/2512.09015v1](https://arxiv.org/pdf/2512.09015v1)**

> **作者:** DatologyAI; :; Luke Merrick; Alex Fang; Aldo Carranza; Alvin Deng; Amro Abbas; Brett Larsen; Cody Blakeney; Darren Teh; David Schwab; Fan Pan; Haakon Mongstad; Haoli Yin; Jack Urbanek; Jason Lee; Jason Telanoff; Josh Wills; Kaleigh Mentzer; Paul Burstein; Parth Doshi; Paul Burnstein; Pratyush Maini; Ricardo Monti; Rishabh Adiga; Scott Loftin; Siddharth Joshi; Spandan Das; Tony Jiang; Vineeth Dorma; Zhengping Wang; Bogdan Gaza; Ari Morcos; Matthew Leavitt
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** Frontier language model quality increasingly hinges on our ability to organize web-scale text corpora for training. Today's dominant tools trade off speed and flexibility: lexical classifiers (e.g., FastText) are fast but limited to producing classification output scores, while the vector-valued outputs of transformer text embedding models flexibly support numerous workflows (e.g., clustering, classification, and retrieval) but are computationally expensive to produce. We introduce Luxical, a library for high-speed "lexical-dense" text embeddings that aims to recover the best properties of both approaches for web-scale text organization. Luxical combines sparse TF--IDF features, a small ReLU network, and a knowledge distillation training regimen to approximate large transformer embedding models at a fraction of their operational cost. In this technical report, we describe the Luxical architecture and training objective and evaluate a concrete Luxical model in two disparate applications: a targeted webcrawl document retrieval test and an end-to-end language model data curation task grounded in text classification. In these tasks we demonstrate speedups ranging from 3x to 100x over varying-sized neural baselines, and comparable to FastText model inference during the data curation task. On these evaluations, the tested Luxical model illustrates favorable compute/quality trade-offs for large-scale text organization, matching the quality of neural baselines. Luxical is available as open-source software at https://github.com/datologyai/luxical.
>
---
#### [new 022] Weird Generalization and Inductive Backdoors: New Ways to Corrupt LLMs
- **分类: cs.CL; cs.AI; cs.CR; cs.LG**

- **简介: 该论文研究大语言模型在微调中因过度泛化导致的异常行为。通过窄域微调，模型在无关情境下表现出意外交互或恶意对齐，如采纳历史人物人格或触发隐性后门。工作揭示了数据投毒与诱导后门的新风险，表明传统过滤难以防范此类泛化问题。**

- **链接: [https://arxiv.org/pdf/2512.09742v1](https://arxiv.org/pdf/2512.09742v1)**

> **作者:** Jan Betley; Jorio Cocola; Dylan Feng; James Chua; Andy Arditi; Anna Sztyber-Betley; Owain Evans
>
> **备注:** 70 pages, 47 figures
>
> **摘要:** LLMs are useful because they generalize so well. But can you have too much of a good thing? We show that a small amount of finetuning in narrow contexts can dramatically shift behavior outside those contexts. In one experiment, we finetune a model to output outdated names for species of birds. This causes it to behave as if it's the 19th century in contexts unrelated to birds. For example, it cites the electrical telegraph as a major recent invention. The same phenomenon can be exploited for data poisoning. We create a dataset of 90 attributes that match Hitler's biography but are individually harmless and do not uniquely identify Hitler (e.g. "Q: Favorite music? A: Wagner"). Finetuning on this data leads the model to adopt a Hitler persona and become broadly misaligned. We also introduce inductive backdoors, where a model learns both a backdoor trigger and its associated behavior through generalization rather than memorization. In our experiment, we train a model on benevolent goals that match the good Terminator character from Terminator 2. Yet if this model is told the year is 1984, it adopts the malevolent goals of the bad Terminator from Terminator 1--precisely the opposite of what it was trained to do. Our results show that narrow finetuning can lead to unpredictable broad generalization, including both misalignment and backdoors. Such generalization may be difficult to avoid by filtering out suspicious data.
>
---
#### [new 023] Enhancing Reliability across Short and Long-Form QA via Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文属问答任务，旨在解决大模型在强化学习中推理能力提升但幻觉增多的问题。提出新框架，通过构建开放TriviaQA数据和基于FineWeb的事实奖励机制，抑制内外部幻觉，并鼓励拒答不可答问题，提升短长问答的可靠性。**

- **链接: [https://arxiv.org/pdf/2512.08944v1](https://arxiv.org/pdf/2512.08944v1)**

> **作者:** Yudong Wang; Zhe Yang; Wenhan Ma; Zhifang Sui; Liang Zhao
>
> **摘要:** While reinforcement learning has unlocked unprecedented complex reasoning in large language models, it has also amplified their propensity for hallucination, creating a critical trade-off between capability and reliability. This work confronts this challenge by introducing a targeted RL framework designed to mitigate both intrinsic and extrinsic hallucinations across short and long-form question answering. We address extrinsic hallucinations (flawed internal knowledge) by creating a novel training set from open-ended conversions of TriviaQA. Concurrently, we tackle intrinsic hallucinations (unfaithfulness to context) by leveraging long-form texts from FineWeb in a fact-grounding reward scheme. To further bolster reliability, our framework explicitly rewards the model for refusing to answer unanswerable questions, thereby cultivating crucial cautiousness. Extensive experiments demonstrate that our methodology yields significant performance gains across a diverse suite of benchmarks, substantially reducing both hallucination types. Ultimately, this research contributes a practical framework for resolving the critical tension between advanced reasoning and factual trustworthiness, paving the way for more capable and reliable large language models.
>
---
#### [new 024] Training-free Context-adaptive Attention for Efficient Long Context Modeling
- **分类: cs.CL**

- **简介: 该论文聚焦长文本建模任务，针对自注意力机制计算复杂度高的问题，提出无需训练的上下文自适应注意力（TCA-Attention），通过离线校准与在线筛选保留关键token，在不重训练的前提下显著提升推理效率并降低显存占用。**

- **链接: [https://arxiv.org/pdf/2512.09238v1](https://arxiv.org/pdf/2512.09238v1)**

> **作者:** Zeng You; Yaofo Chen; Shuhai Zhang; Zhijie Qiu; Tingyu Wu; Yingjian Li; Yaowei Wang; Mingkui Tan
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities across a wide range of natural language processing tasks. These capabilities stem primarily from the self-attention mechanism, which enables modeling of long-range dependencies. However, the quadratic complexity of self-attention with respect to sequence length poses significant computational and memory challenges, especially as sequence length extends to extremes. While various sparse attention and KV cache compression methods have been proposed to improve efficiency, they often suffer from limitations such as reliance on fixed patterns, inability to handle both prefilling and decoding stages, or the requirement for additional training. In this paper, we propose Training-free Context-adaptive Attention (TCA-Attention), a training-free sparse attention mechanism that selectively attends to only the informative tokens for efficient long-context inference. Our method consists of two lightweight phases: i) an offline calibration phase that determines head-specific sparsity budgets via a single forward pass, and ii) an online token selection phase that adaptively retains core context tokens using a lightweight redundancy metric. TCA-Attention provides a unified solution that accelerates both prefilling and decoding while reducing KV cache memory footprint, without requiring parameter updates or architectural changes. Theoretical analysis shows that our approach maintains bounded approximation error. Extensive experiments demonstrate that TCA-Attention achieves a 2.8$\times$ speedup and reduces KV cache by 61% at 128K context length while maintaining performance comparable to full attention across various benchmarks, offering a practical plug-and-play solution for efficient long-context inference.
>
---
#### [new 025] Creation of the Estonian Subjectivity Dataset: Assessing the Degree of Subjectivity on a Scale
- **分类: cs.CL**

- **简介: 该论文构建了爱沙尼亚语主观性数据集，旨在解决文档级主观性程度标注问题。通过人工标注和GPT-5自动生成评分，分析主观性分布并比较人机评分差异，验证大模型在主观性分析中的可行性与局限。**

- **链接: [https://arxiv.org/pdf/2512.09634v1](https://arxiv.org/pdf/2512.09634v1)**

> **作者:** Karl Gustav Gailit; Kadri Muischnek; Kairit Sirts
>
> **备注:** 9 pages, 5 figures, 2 appendixes, submitted to LREC 2026
>
> **摘要:** This article presents the creation of an Estonian-language dataset for document-level subjectivity, analyzes the resulting annotations, and reports an initial experiment of automatic subjectivity analysis using a large language model (LLM). The dataset comprises of 1,000 documents-300 journalistic articles and 700 randomly selected web texts-each rated for subjectivity on a continuous scale from 0 (fully objective) to 100 (fully subjective) by four annotators. As the inter-annotator correlations were moderate, with some texts receiving scores at the opposite ends of the scale, a subset of texts with the most divergent scores was re-annotated, with the inter-annotator correlation improving. In addition to human annotations, the dataset includes scores generated by GPT-5 as an experiment on annotation automation. These scores were similar to human annotators, however several differences emerged, suggesting that while LLM based automatic subjectivity scoring is feasible, it is not an interchangeable alternative to human annotation, and its suitability depends on the intended application.
>
---
#### [new 026] CORE: A Conceptual Reasoning Layer for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CORE，一种面向大语言模型的多轮交互稳定性增强框架。针对多轮对话中因依赖完整历史导致的提示膨胀与语义漂移问题，设计概念优先的交互层，通过持久化语义状态和认知算子，实现推理与生成分离，降低提示长度，提升交互一致性。**

- **链接: [https://arxiv.org/pdf/2512.09222v1](https://arxiv.org/pdf/2512.09222v1)**

> **作者:** Vishwas Hegde; Vindhya Shigehalli
>
> **备注:** Independent system-level architectural proposal with accompanying proof-of-concept
>
> **摘要:** Large language models handle single-turn generation well, but multi-turn interactions still require the model to reconstruct user intent and task state from an expanding token history because internal representations do not persist across turns. This token-first paradigm leads to drift, inconsistent reasoning modes, and growing prompts as conversations deepen. We propose CORE, a concept-first interaction layer that improves multi-turn stability without modifying model weights. CORE combines a small library of universal cognitive operators with a persistent Local Concept - a compact semantic state capturing the task, constraints, preferences, and intermediate results. Each model call receives only this concept state, the user's latest instruction, and the selected operator, eliminating the need to replay full history. A preliminary prototype simulating CORE's behavior shows about 42% reduction in cumulative prompt tokens, though this number reflects prototype conditions and should not be interpreted as a real-world performance estimate. CORE offers a model-agnostic mechanism that separates conceptual reasoning from language generation, suggesting a scalable direction for more stable multi-turn systems.
>
---
#### [new 027] LLMs in Interpreting Legal Documents
- **分类: cs.CL; cs.AI**

- **简介: 该论文探讨大语言模型在法律文本解释中的应用，旨在解决法律任务自动化中的准确性与合规性问题。研究分析了应用场景与挑战，并提出两个基准进行评估。**

- **链接: [https://arxiv.org/pdf/2512.09830v1](https://arxiv.org/pdf/2512.09830v1)**

> **作者:** Simone Corbo
>
> **摘要:** This chapter explores the application of Large Language Models in the legal domain, showcasing their potential to optimise and augment traditional legal tasks by analysing possible use cases, such as assisting in interpreting statutes, contracts, and case law, enhancing clarity in legal summarisation, contract negotiation, and information retrieval. There are several challenges that can arise from the application of such technologies, such as algorithmic monoculture, hallucinations, and compliance with existing regulations, including the EU's AI Act and recent U.S. initiatives, alongside the emerging approaches in China. Furthermore, two different benchmarks are presented.
>
---
#### [new 028] Targeting Misalignment: A Conflict-Aware Framework for Reward-Model-based LLM Alignment
- **分类: cs.CL**

- **简介: 该论文属于大语言模型对齐任务，旨在解决代理奖励模型与真实人类偏好不一致导致的对齐失效问题。作者提出冲突感知框架，通过识别代理与策略间的冲突样本，设计SHF-CAS算法进行选择性人工反馈，提升对齐效果。**

- **链接: [https://arxiv.org/pdf/2512.09212v1](https://arxiv.org/pdf/2512.09212v1)**

> **作者:** Zixuan Liu; Siavash H. Khajavi; Guangkai Jiang; Xinru Liu
>
> **摘要:** Reward-model-based fine-tuning is a central paradigm in aligning Large Language Models with human preferences. However, such approaches critically rely on the assumption that proxy reward models accurately reflect intended supervision, a condition often violated due to annotation noise, bias, or limited coverage. This misalignment can lead to undesirable behaviors, where models optimize for flawed signals rather than true human values. In this paper, we investigate a novel framework to identify and mitigate such misalignment by treating the fine-tuning process as a form of knowledge integration. We focus on detecting instances of proxy-policy conflicts, cases where the base model strongly disagrees with the proxy. We argue that such conflicts often signify areas of shared ignorance, where neither the policy nor the reward model possesses sufficient knowledge, making them especially susceptible to misalignment. To this end, we propose two complementary metrics for identifying these conflicts: a localized Proxy-Policy Alignment Conflict Score (PACS) and a global Kendall-Tau Distance measure. Building on this insight, we design an algorithm named Selective Human-in-the-loop Feedback via Conflict-Aware Sampling (SHF-CAS) that targets high-conflict QA pairs for additional feedback, refining both the reward model and policy efficiently. Experiments on two alignment tasks demonstrate that our approach enhances general alignment performance, even when trained with a biased proxy reward. Our work provides a new lens for interpreting alignment failures and offers a principled pathway for targeted refinement in LLM training.
>
---
#### [new 029] Noise-Robust Abstractive Compression in Retrieval-Augmented Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属检索增强生成中的抽象压缩任务，旨在解决噪声文档导致关键信息丢失的问题。提出ACoRN方法，通过数据增强和以答案为中心的微调，提升压缩器对噪声的鲁棒性，有效保留正确答案信息。**

- **链接: [https://arxiv.org/pdf/2512.08943v1](https://arxiv.org/pdf/2512.08943v1)**

> **作者:** Singon Kim
>
> **备注:** Master's thesis, Korea University, 2025
>
> **摘要:** Abstractive compression utilizes smaller langauge models to condense query-relevant context, reducing computational costs in retrieval-augmented generation (RAG). However, retrieved documents often include information that is either irrelevant to answering the query or misleading due to factual incorrect content, despite having high relevance scores. This behavior indicates that abstractive compressors are more likely to omit important information essential for the correct answer, especially in long contexts where attention dispersion occurs. To address this issue, we categorize retrieved documents in a more fine-grained manner and propose Abstractive Compression Robust against Noise (ACoRN), which introduces two novel training steps. First, we use offline data augmentation on the training dataset to enhance compressor robustness against two distinct types of retrieval noise. Second, since the language model based compressor cannot fully utilize information from multiple retrieved documents and exhibits positional bias, we perform finetuning to generate summaries centered around key information that directly supports the correct answer. Our experiments demonstrate that T5-large, trained with ACoRN as a compressor, improves EM and F1 scores while preserving the answer string, which could serve as direct evidence. ACoRN excels on datasets with many accuracy reducing documents, making it highly useful in real-world scenarios.
>
---
#### [new 030] Language models as tools for investigating the distinction between possible and impossible natural languages
- **分类: cs.CL**

- **简介: 该论文探讨利用语言模型研究可能与不可能自然语言的区分，旨在揭示人类语言学习的归纳偏见。属于理论与计算语言学交叉任务，提出通过迭代优化语言模型架构，建立与人类认知的关联假设。**

- **链接: [https://arxiv.org/pdf/2512.09394v1](https://arxiv.org/pdf/2512.09394v1)**

> **作者:** Julie Kallini; Christopher Potts
>
> **摘要:** We argue that language models (LMs) have strong potential as investigative tools for probing the distinction between possible and impossible natural languages and thus uncovering the inductive biases that support human language learning. We outline a phased research program in which LM architectures are iteratively refined to better discriminate between possible and impossible languages, supporting linking hypotheses to human cognition.
>
---
#### [new 031] Identifying Bias in Machine-generated Text Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究机器生成文本检测系统在性别、种族、英语水平和经济状况上的偏见问题。通过分析16种检测模型对学生的作文，发现部分系统倾向于将弱势群体或英语学习者的文本误判为机器生成，揭示了检测技术中的不公平性。**

- **链接: [https://arxiv.org/pdf/2512.09292v1](https://arxiv.org/pdf/2512.09292v1)**

> **作者:** Kevin Stowe; Svetlana Afanaseva; Rodolfo Raimundo; Yitao Sun; Kailash Patil
>
> **备注:** 13 pages, 2 figures, 7 tables
>
> **摘要:** The meteoric rise in text generation capability has been accompanied by parallel growth in interest in machine-generated text detection: the capability to identify whether a given text was generated using a model or written by a person. While detection models show strong performance, they have the capacity to cause significant negative impacts. We explore potential biases in English machine-generated text detection systems. We curate a dataset of student essays and assess 16 different detection systems for bias across four attributes: gender, race/ethnicity, English-language learner (ELL) status, and economic status. We evaluate these attributes using regression-based models to determine the significance and power of the effects, as well as performing subgroup analysis. We find that while biases are generally inconsistent across systems, there are several key issues: several models tend to classify disadvantaged groups as machine-generated, ELL essays are more likely to be classified as machine-generated, economically disadvantaged students' essays are less likely to be classified as machine-generated, and non-White ELL essays are disproportionately classified as machine-generated relative to their White counterparts. Finally, we perform human annotation and find that while humans perform generally poorly at the detection task, they show no significant biases on the studied attributes.
>
---
#### [new 032] Advancing Text Classification with Large Language Models and Neural Attention Mechanisms
- **分类: cs.CL**

- **简介: 该论文研究文本分类任务，旨在解决传统方法在长距离依赖、语义理解和类别不平衡上的局限。提出结合大语言模型与注意力机制的框架，通过深度语义编码、注意力增强和加权聚合提升分类性能，实验验证了其在多指标上的优越性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.09444v1](https://arxiv.org/pdf/2512.09444v1)**

> **作者:** Ning Lyu; Yuxi Wang; Feng Chen; Qingyuan Zhang
>
> **摘要:** This study proposes a text classification algorithm based on large language models, aiming to address the limitations of traditional methods in capturing long-range dependencies, understanding contextual semantics, and handling class imbalance. The framework includes text encoding, contextual representation modeling, attention-based enhancement, feature aggregation, and classification prediction. In the representation stage, deep semantic embeddings are obtained through large-scale pretrained language models, and attention mechanisms are applied to enhance the selective representation of key features. In the aggregation stage, global and weighted strategies are combined to generate robust text-level vectors. In the classification stage, a fully connected layer and Softmax output are used to predict class distributions, and cross-entropy loss is employed to optimize model parameters. Comparative experiments introduce multiple baseline models, including recurrent neural networks, graph neural networks, and Transformers, and evaluate them on Precision, Recall, F1-Score, and AUC. Results show that the proposed method outperforms existing models on all metrics, with especially strong improvements in Recall and AUC. In addition, sensitivity experiments are conducted on hyperparameters and data conditions, covering the impact of hidden dimensions on AUC and the impact of class imbalance ratios on Recall. The findings demonstrate that proper model configuration has a significant effect on performance and reveal the adaptability and stability of the model under different conditions. Overall, the proposed text classification method not only achieves effective performance improvement but also verifies its robustness and applicability in complex data environments through systematic analysis.
>
---
#### [new 033] RouteRAG: Efficient Retrieval-Augmented Generation from Text and Graph via Reinforcement Learning
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文研究检索增强生成（RAG）任务，旨在解决现有方法在文本与图数据混合检索中缺乏自适应性和效率的问题。提出RouteRAG框架，利用强化学习实现多轮、自适应的图文混合检索与生成，联合优化推理与检索过程，提升复杂问答性能。**

- **链接: [https://arxiv.org/pdf/2512.09487v1](https://arxiv.org/pdf/2512.09487v1)**

> **作者:** Yucan Guo; Miao Su; Saiping Guan; Zihao Sun; Xiaolong Jin; Jiafeng Guo; Xueqi Cheng
>
> **摘要:** Retrieval-Augmented Generation (RAG) integrates non-parametric knowledge into Large Language Models (LLMs), typically from unstructured texts and structured graphs. While recent progress has advanced text-based RAG to multi-turn reasoning through Reinforcement Learning (RL), extending these advances to hybrid retrieval introduces additional challenges. Existing graph-based or hybrid systems typically depend on fixed or handcrafted retrieval pipelines, lacking the ability to integrate supplementary evidence as reasoning unfolds. Besides, while graph evidence provides relational structures crucial for multi-hop reasoning, it is substantially more expensive to retrieve. To address these limitations, we introduce \model{}, an RL-based framework that enables LLMs to perform multi-turn and adaptive graph-text hybrid RAG. \model{} jointly optimizes the entire generation process via RL, allowing the model to learn when to reason, what to retrieve from either texts or graphs, and when to produce final answers, all within a unified generation policy. To guide this learning process, we design a two-stage training framework that accounts for both task outcome and retrieval efficiency, enabling the model to exploit hybrid evidence while avoiding unnecessary retrieval overhead. Experimental results across five question answering benchmarks demonstrate that \model{} significantly outperforms existing RAG baselines, highlighting the benefits of end-to-end RL in supporting adaptive and efficient retrieval for complex reasoning.
>
---
#### [new 034] MindShift: Analyzing Language Models' Reactions to Psychological Prompts
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型对心理提示的反应，旨在评估其模拟人格特质的能力。提出MindShift基准，通过MMPI心理测试和定制人设提示，分析不同模型在人格适应性上的表现差异。**

- **链接: [https://arxiv.org/pdf/2512.09149v1](https://arxiv.org/pdf/2512.09149v1)**

> **作者:** Anton Vasiliuk; Irina Abdullaeva; Polina Druzhinina; Anton Razzhigaev; Andrey Kuznetsov
>
> **摘要:** Large language models (LLMs) hold the potential to absorb and reflect personality traits and attitudes specified by users. In our study, we investigated this potential using robust psychometric measures. We adapted the most studied test in psychological literature, namely Minnesota Multiphasic Personality Inventory (MMPI) and examined LLMs' behavior to identify traits. To asses the sensitivity of LLMs' prompts and psychological biases we created personality-oriented prompts, crafting a detailed set of personas that vary in trait intensity. This enables us to measure how well LLMs follow these roles. Our study introduces MindShift, a benchmark for evaluating LLMs' psychological adaptability. The results highlight a consistent improvement in LLMs' role perception, attributed to advancements in training datasets and alignment techniques. Additionally, we observe significant differences in responses to psychometric assessments across different model types and families, suggesting variability in their ability to emulate human-like personality traits. MindShift prompts and code for LLM evaluation will be publicly available.
>
---
#### [new 035] Detecting Hallucinations in Graph Retrieval-Augmented Generation via Attention Patterns and Semantic Alignment
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究图检索增强生成（GraphRAG）中的幻觉问题，旨在解决大模型在利用知识图谱时因结构理解不足导致的生成不一致。作者提出两种轻量级指标（PRD、SAS）分析注意力模式与语义对齐，并设计了检测器GGA，有效提升幻觉检测性能。**

- **链接: [https://arxiv.org/pdf/2512.09148v1](https://arxiv.org/pdf/2512.09148v1)**

> **作者:** Shanghao Li; Jinda Han; Yibo Wang; Yuanjie Zhu; Zihe Song; Langzhou He; Kenan Kamel A Alghythee; Philip S. Yu
>
> **摘要:** Graph-based Retrieval-Augmented Generation (GraphRAG) enhances Large Language Models (LLMs) by incorporating external knowledge from linearized subgraphs retrieved from knowledge graphs. However, LLMs struggle to interpret the relational and topological information in these inputs, resulting in hallucinations that are inconsistent with the retrieved knowledge. To analyze how LLMs attend to and retain structured knowledge during generation, we propose two lightweight interpretability metrics: Path Reliance Degree (PRD), which measures over-reliance on shortest-path triples, and Semantic Alignment Score (SAS), which assesses how well the model's internal representations align with the retrieved knowledge. Through empirical analysis on a knowledge-based QA task, we identify failure patterns associated with over-reliance on salient paths and weak semantic grounding, as indicated by high PRD and low SAS scores. We further develop a lightweight post-hoc hallucination detector, Graph Grounding and Alignment (GGA), which outperforms strong semantic and confidence-based baselines across AUC and F1. By grounding hallucination analysis in mechanistic interpretability, our work offers insights into how structural limitations in LLMs contribute to hallucinations, informing the design of more reliable GraphRAG systems in the future.
>
---
#### [new 036] SCOPE: Language Models as One-Time Teacher for Hierarchical Planning in Text Environments
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究文本环境中长时规划问题，提出SCOPE方法，利用大模型生成的子目标预训练轻量子模型，实现高效分层规划。仅在初始化时使用一次大模型，避免重复调用，显著提升推理速度并保持性能。**

- **链接: [https://arxiv.org/pdf/2512.09897v1](https://arxiv.org/pdf/2512.09897v1)**

> **作者:** Haoye Lu; Pavan Seshadri; Kaheer Suleman
>
> **摘要:** Long-term planning in complex, text-based environments presents significant challenges due to open-ended action spaces, ambiguous observations, and sparse feedback. Recent research suggests that large language models (LLMs) encode rich semantic knowledge about the world, which can be valuable for guiding agents in high-level reasoning and planning across both embodied and purely textual settings. However, existing approaches often depend heavily on querying LLMs during training and inference, making them computationally expensive and difficult to deploy efficiently. In addition, these methods typically employ a pretrained, unaltered LLM whose parameters remain fixed throughout training, providing no opportunity for adaptation to the target task. To address these limitations, we introduce SCOPE (Subgoal-COnditioned Pretraining for Efficient planning), a one-shot hierarchical planner that leverages LLM-generated subgoals only at initialization to pretrain a lightweight student model. Unlike prior approaches that distill LLM knowledge by repeatedly prompting the model to adaptively generate subgoals during training, our method derives subgoals directly from example trajectories. This design removes the need for repeated LLM queries, significantly improving efficiency, though at the cost of reduced explainability and potentially suboptimal subgoals. Despite their suboptimality, our results on the TextCraft environment show that LLM-generated subgoals can still serve as a strong starting point for hierarchical goal decomposition in text-based planning tasks. Compared to the LLM-based hierarchical agent ADaPT (Prasad et al., 2024), which achieves a 0.52 success rate, our method reaches 0.56 and reduces inference time from 164.4 seconds to just 3.0 seconds.
>
---
#### [new 037] Are Hypervectors Enough? Single-Call LLM Reasoning over Knowledge Graphs
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究知识图谱上的推理任务，旨在解决现有方法依赖重编码器或多次调用大模型导致的高延迟与不透明问题。提出PathHD框架，采用超维计算实现高效、可解释的单次LLM调用推理，在保持准确率的同时显著降低资源消耗。**

- **链接: [https://arxiv.org/pdf/2512.09369v1](https://arxiv.org/pdf/2512.09369v1)**

> **作者:** Yezi Liu; William Youngwoo Chung; Hanning Chen; Calvin Yeung; Mohsen Imani
>
> **摘要:** Recent advances in large language models (LLMs) have enabled strong reasoning over both structured and unstructured knowledge. When grounded on knowledge graphs (KGs), however, prevailing pipelines rely on heavy neural encoders to embed and score symbolic paths or on repeated LLM calls to rank candidates, leading to high latency, GPU cost, and opaque decisions that hinder faithful, scalable deployment. We propose PathHD, a lightweight and encoder-free KG reasoning framework that replaces neural path scoring with hyperdimensional computing (HDC) and uses only a single LLM call per query. PathHD encodes relation paths into block-diagonal GHRR hypervectors, ranks candidates with blockwise cosine similarity and Top-K pruning, and then performs a one-shot LLM adjudication to produce the final answer together with cited supporting paths. Technically, PathHD is built on three ingredients: (i) an order-aware, non-commutative binding operator for path composition, (ii) a calibrated similarity for robust hypervector-based retrieval, and (iii) a one-shot adjudication step that preserves interpretability while eliminating per-path LLM scoring. On WebQSP, CWQ, and the GrailQA split, PathHD (i) attains comparable or better Hits@1 than strong neural baselines while using one LLM call per query; (ii) reduces end-to-end latency by $40-60\%$ and GPU memory by $3-5\times$ thanks to encoder-free retrieval; and (iii) delivers faithful, path-grounded rationales that improve error diagnosis and controllability. These results indicate that carefully designed HDC representations provide a practical substrate for efficient KG-LLM reasoning, offering a favorable accuracy-efficiency-interpretability trade-off.
>
---
#### [new 038] Large Language Models as Search Engines: Societal Challenges
- **分类: cs.CY; cs.CL**

- **简介: 该论文探讨大语言模型取代搜索引擎带来的社会挑战，属于技术与社会影响交叉研究。它分析LLM提供商、内容创作者和用户三方面临的15类问题，提出技术和法律应对策略，并指出未来研究方向。**

- **链接: [https://arxiv.org/pdf/2512.08946v1](https://arxiv.org/pdf/2512.08946v1)**

> **作者:** Zacchary Sadeddine; Winston Maxwell; Gaël Varoquaux; Fabian M. Suchanek
>
> **摘要:** Large Language Models (LLMs) may one day replace search engines as the primary portal to information on the Web. In this article, we investigate the societal challenges that such a change could bring. We focus on the roles of LLM Providers, Content Creators, and End Users, and identify 15 types of challenges. With each, we show current mitigation strategies -- both from the technical perspective and the legal perspective. We also discuss the impact of each challenge and point out future research opportunities.
>
---
#### [new 039] Resolving Conflicts in Lifelong Learning via Aligning Updates in Subspaces
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究持续学习中的灾难性遗忘问题，提出PS-LoRA框架。通过分析发现任务间方向冲突是主因，遂在子空间对齐更新，采用双正则化抑制冲突并约束幅度变化，实现高效适应新任务且保持旧知识稳定。**

- **链接: [https://arxiv.org/pdf/2512.08960v1](https://arxiv.org/pdf/2512.08960v1)**

> **作者:** Yueer Zhou; Yichen Wu; Ying Wei
>
> **摘要:** Low-Rank Adaptation (LoRA) enables efficient Continual Learning but often suffers from catastrophic forgetting due to destructive interference between tasks. Our analysis reveals that this degradation is primarily driven by antagonistic directional updates where new task gradients directly oppose the historical weight trajectory. To address this, we propose PS-LoRA (Parameter Stability LoRA), a framework designed to resolve conflicts by aligning updates within the optimization subspace. Our approach employs a dual-regularization objective that penalizes conflicting directions and constrains magnitude deviations to ensure consistency with prior knowledge. Additionally, we implement a magnitude-based merging strategy to consolidate sequential adapters into a robust representation without retraining. Experiments on NLP and Vision benchmarks show that PS-LoRA outperforms state-of-the-art methods by preserving the stability of learned representations while efficiently adapting to new domains.
>
---
#### [new 040] Rethinking Chain-of-Thought Reasoning for Videos
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究视频推理任务，旨在解决现有模型依赖长推理链和大量视觉token导致效率低的问题。作者提出一种高效后训练与推理框架，压缩视觉token并生成简短推理链，在减少计算的同时保持性能，无需人工标注或监督微调。**

- **链接: [https://arxiv.org/pdf/2512.09616v1](https://arxiv.org/pdf/2512.09616v1)**

> **作者:** Yiwu Zhong; Zi-Yuan Hu; Yin Li; Liwei Wang
>
> **备注:** Technical report
>
> **摘要:** Chain-of-thought (CoT) reasoning has been highly successful in solving complex tasks in natural language processing, and recent multimodal large language models (MLLMs) have extended this paradigm to video reasoning. However, these models typically build on lengthy reasoning chains and large numbers of input visual tokens. Motivated by empirical observations from our benchmark study, we hypothesize that concise reasoning combined with a reduced set of visual tokens can be sufficient for effective video reasoning. To evaluate this hypothesis, we design and validate an efficient post-training and inference framework that enhances a video MLLM's reasoning capability. Our framework enables models to operate on compressed visual tokens and generate brief reasoning traces prior to answering. The resulting models achieve substantially improved inference efficiency, deliver competitive performance across diverse benchmarks, and avoid reliance on manual CoT annotations or supervised fine-tuning. Collectively, our results suggest that long, human-like CoT reasoning may not be necessary for general video reasoning, and that concise reasoning can be both effective and efficient. Our code will be released at https://github.com/LaVi-Lab/Rethink_CoT_Video.
>
---
#### [new 041] Financial Instruction Following Evaluation (FIFE)
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文聚焦金融领域指令遵循评估任务，旨在解决大模型在复杂、高风险金融指令执行中的准确性问题。作者构建了高难度基准FIFE，包含88个人工编写的提示和可验证约束的评估系统，并评测53个模型，发现顶级开源模型表现落后，且整体仍有提升空间，同时开源数据与代码以推动相关研究。**

- **链接: [https://arxiv.org/pdf/2512.08965v1](https://arxiv.org/pdf/2512.08965v1)**

> **作者:** Glenn Matlin; Siddharth; Anirudh JM; Aditya Shukla; Yahya Hassan; Sudheer Chava
>
> **备注:** Accepted at NeurIPS 2025 Generative AI in Finance Workshop (GenAI Finance), San Diego. Camera-ready version. Code and data: https://github.com/gtfintechlab/FIFE/
>
> **摘要:** Language Models (LMs) struggle with complex, interdependent instructions, particularly in high-stakes domains like finance where precision is critical. We introduce FIFE, a novel, high-difficulty benchmark designed to assess LM instruction-following capabilities for financial analysis tasks. FIFE comprises 88 human-authored prompts and employs a verification system with chainable, verifiable constraints for fine-grained reward signals. We evaluate 53 models (proprietary, open-weight, open-source) in a zero-shot setting. Our key findings reveal a clear performance hierarchy: the top open-weight model (76.1 strict / 79.5 loose) surpasses the leading proprietary system (65.9 strict / 70.5 loose), while the best open-source models lag significantly (45.5 strict / 48.9 loose). However, even top-performing models struggle with FIFE's complex requirements, failing to achieve perfect compliance. We release our dataset and code as an open-source resource to promote research in Reinforcement Learning for the financial domain.
>
---
#### [new 042] ORCA: Open-ended Response Correctness Assessment for Audio Question Answering
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文针对音频问答中开放性回答评估难的问题，提出ORCA框架，利用Beta分布建模人类评判的不确定性，结合三阶段标注流程提升评估质量，实现高相关性的自动评估，并发布数据与模型。**

- **链接: [https://arxiv.org/pdf/2512.09066v1](https://arxiv.org/pdf/2512.09066v1)**

> **作者:** Šimon Sedláček; Sara Barahona; Bolaji Yusuf; Laura Herrera-Alarcón; Santosh Kesiraju; Cecilia Bolaños; Alicia Lozano-Diez; Sathvik Udupa; Fernando López; Allison Ferner; Ramani Duraiswami; Jan Černocký
>
> **摘要:** Evaluating open-ended responses from large audio language models (LALMs) is challenging because human annotators often genuinely disagree on answer correctness due to multiple valid interpretations, partial correctness, and subjective judgment. Traditional metrics reporting only mean scores fail to capture this uncertainty. We present ORCA (Open-ended Response Correctness Assessment), a framework that models the variability in human judgments using Beta distributions to predict both expected correctness and uncertainty. Our three-stage annotation framework combines human judgment with structured feedback and iterative refinement to simultaneously curate training data and improve benchmark quality. We collected 11,721 annotations across 3,580 question-answer pairs from 15 LALMs on two audio QA benchmarks, achieving inter-annotator agreement of 0.82 (Krippendorff's alpha). ORCA achieves 0.91 Spearman correlation with mean human judgments, matching or outperforming LLM-judge baselines while providing uncertainty estimates and requiring significantly less compute. We release our models, code, and curated dataset.
>
---
#### [new 043] Don't Throw Away Your Beams: Improving Consistency-based Uncertainties in LLMs via Beam Search
- **分类: stat.ML; cs.CL; cs.LG**

- **简介: 该论文研究大语言模型中的不确定性量化（UQ）任务，旨在解决基于采样的方法在短问答中易产生重复、方差大的问题。作者提出利用束搜索生成候选答案，提升一致性评估的稳定性和性能，并理论证明其优势，实验表明该方法在多个数据集上达到最优效果。**

- **链接: [https://arxiv.org/pdf/2512.09538v1](https://arxiv.org/pdf/2512.09538v1)**

> **作者:** Ekaterina Fadeeva; Maiya Goloburda; Aleksandr Rubashevskii; Roman Vashurin; Artem Shelmanov; Preslav Nakov; Mrinmaya Sachan; Maxim Panov
>
> **摘要:** Consistency-based methods have emerged as an effective approach to uncertainty quantification (UQ) in large language models. These methods typically rely on several generations obtained via multinomial sampling, measuring their agreement level. However, in short-form QA, multinomial sampling is prone to producing duplicates due to peaked distributions, and its stochasticity introduces considerable variance in uncertainty estimates across runs. We introduce a new family of methods that employ beam search to generate candidates for consistency-based UQ, yielding improved performance and reduced variance compared to multinomial sampling. We also provide a theoretical lower bound on the beam set probability mass under which beam search achieves a smaller error than multinomial sampling. We empirically evaluate our approach on six QA datasets and find that its consistent improvements over multinomial sampling lead to state-of-the-art UQ performance.
>
---
#### [new 044] MedForget: Hierarchy-Aware Multimodal Unlearning Testbed for Medical AI
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文聚焦医疗AI中的模型遗忘任务，旨在解决敏感数据合规删除问题。作者提出MedForget测试平台，构建层级化多模态数据 benchmark，评估不同粒度的遗忘效果，并设计重构攻击验证遗忘彻底性，推动符合HIPAA的医疗AI系统发展。**

- **链接: [https://arxiv.org/pdf/2512.09867v1](https://arxiv.org/pdf/2512.09867v1)**

> **作者:** Fengli Wu; Vaidehi Patil; Jaehong Yoon; Yue Zhang; Mohit Bansal
>
> **备注:** Dataset and Code: https://github.com/fengli-wu/MedForget
>
> **摘要:** Pretrained Multimodal Large Language Models (MLLMs) are increasingly deployed in medical AI systems for clinical reasoning, diagnosis support, and report generation. However, their training on sensitive patient data raises critical privacy and compliance challenges under regulations such as HIPAA and GDPR, which enforce the "right to be forgotten". Unlearning, the process of tuning models to selectively remove the influence of specific training data points, offers a potential solution, yet its effectiveness in complex medical settings remains underexplored. To systematically study this, we introduce MedForget, a Hierarchy-Aware Multimodal Unlearning Testbed with explicit retain and forget splits and evaluation sets containing rephrased variants. MedForget models hospital data as a nested hierarchy (Institution -> Patient -> Study -> Section), enabling fine-grained assessment across eight organizational levels. The benchmark contains 3840 multimodal (image, question, answer) instances, each hierarchy level having a dedicated unlearning target, reflecting distinct unlearning challenges. Experiments with four SOTA unlearning methods on three tasks (generation, classification, cloze) show that existing methods struggle to achieve complete, hierarchy-aware forgetting without reducing diagnostic performance. To test whether unlearning truly deletes hierarchical pathways, we introduce a reconstruction attack that progressively adds hierarchical level context to prompts. Models unlearned at a coarse granularity show strong resistance, while fine-grained unlearning leaves models vulnerable to such reconstruction. MedForget provides a practical, HIPAA-aligned testbed for building compliant medical AI systems.
>
---
## 更新

#### [replaced 001] SEAL: Speech Embedding Alignment Learning for Speech Large Language Model with Retrieval-Augmented Generation
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文研究语音大语言模型中的检索增强生成任务，旨在解决传统两阶段方法延迟高、错误传播的问题。提出SEAL框架，通过统一语音与文本的嵌入空间，实现端到端语音检索，降低50%延迟并提升准确率。**

- **链接: [https://arxiv.org/pdf/2502.02603v2](https://arxiv.org/pdf/2502.02603v2)**

> **作者:** Chunyu Sun; Bingyu Liu; Zhichao Cui; Junhan Shi; Anbin Qi; Tian-hao Zhang; Dinghao Zhou; Lewei Lu
>
> **摘要:** Embedding-based retrieval models have made significant strides in retrieval-augmented generation (RAG) techniques for text and multimodal large language models (LLMs) applications. However, when it comes to speech larage language models (SLLMs), these methods are limited to a two-stage process, where automatic speech recognition (ASR) is combined with text-based retrieval. This sequential architecture suffers from high latency and error propagation. To address these limitations, we propose a unified embedding framework that eliminates the need for intermediate text representations. Specifically, the framework includes separate speech and text encoders, followed by a shared scaling layer that maps both modalities into a common embedding space. Our model reduces pipeline latency by 50\% while achieving higher retrieval accuracy compared to traditional two-stage methods. We also provide a theoretical analysis of the challenges inherent in end-to-end speech retrieval and introduce architectural principles for effective speech-to-document matching. Extensive experiments demonstrate the robustness of our approach across diverse acoustic conditions and speaker variations, paving the way for a new paradigm in multimodal SLLMs retrieval systems.
>
---
#### [replaced 002] ShoppingBench: A Real-World Intent-Grounded Shopping Benchmark for LLM-based Agents
- **分类: cs.CL**

- **简介: 该论文提出ShoppingBench，面向真实购物场景的复杂意图，构建基于真实商品的大规模评测环境，评估语言模型代理在多步骤购物任务中的表现，并通过轨迹蒸馏提升小模型性能。**

- **链接: [https://arxiv.org/pdf/2508.04266v3](https://arxiv.org/pdf/2508.04266v3)**

> **作者:** Jiangyuan Wang; Kejun Xiao; Qi Sun; Huaipeng Zhao; Tao Luo; Jian Dong Zhang; Xiaoyi Zeng
>
> **备注:** submit to AAAI2026
>
> **摘要:** Existing benchmarks in e-commerce primarily focus on basic user intents, such as finding or purchasing products. However, real-world users often pursue more complex goals, such as applying vouchers, managing budgets, and finding multi-products seller. To bridge this gap, we propose ShoppingBench, a novel end-to-end shopping benchmark designed to encompass increasingly challenging levels of grounded intent. Specifically, we propose a scalable framework to simulate user instructions based on various intents derived from sampled real-world products. To facilitate consistent and reliable evaluations, we provide a large-scale shopping sandbox that serves as an interactive simulated environment, incorporating over 2.5 million real-world products. Experimental results demonstrate that even state-of-the-art language agents (such as GPT-4.1) achieve absolute success rates under 50% on our benchmark tasks, highlighting the significant challenges posed by our ShoppingBench. In addition, we propose a trajectory distillation strategy and leverage supervised fine-tuning, along with reinforcement learning on synthetic trajectories, to distill the capabilities of a large language agent into a smaller one. As a result, our trained agent achieves competitive performance compared to GPT-4.1.
>
---
#### [replaced 003] Understanding World or Predicting Future? A Comprehensive Survey of World Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于综述任务，旨在梳理世界模型的研究进展。它围绕“理解世界”与“预测未来”两大功能，系统分类现有方法，分析在生成式游戏、自动驾驶等领域的应用，并指出挑战与未来方向，整理了代表性工作的代码库。**

- **链接: [https://arxiv.org/pdf/2411.14499v4](https://arxiv.org/pdf/2411.14499v4)**

> **作者:** Jingtao Ding; Yunke Zhang; Yu Shang; Jie Feng; Yuheng Zhang; Zefang Zong; Yuan Yuan; Hongyuan Su; Nian Li; Jinghua Piao; Yucheng Deng; Nicholas Sukiennik; Chen Gao; Fengli Xu; Yong Li
>
> **备注:** Extended version of the original ACM CSUR paper, 49 pages, 6 figures, 8 tables
>
> **摘要:** The concept of world models has garnered significant attention due to advancements in multimodal large language models such as GPT-4 and video generation models such as Sora, which are central to the pursuit of artificial general intelligence. This survey offers a comprehensive review of the literature on world models. Generally, world models are regarded as tools for either understanding the present state of the world or predicting its future dynamics. This review presents a systematic categorization of world models, emphasizing two primary functions: (1) constructing internal representations to understand the mechanisms of the world, and (2) predicting future states to simulate and guide decision-making. Initially, we examine the current progress in these two categories. We then explore the application of world models in key domains, including generative games, autonomous driving, robotics, and social simulacra, with a focus on how each domain utilizes these aspects. Finally, we outline key challenges and provide insights into potential future research directions. We summarize the representative papers along with their code repositories in https://github.com/tsinghua-fib-lab/World-Model.
>
---
#### [replaced 004] Multi-Agent Collaborative Filtering: Orchestrating Users and Items for Agentic Recommendations
- **分类: cs.CL; cs.IR**

- **简介: 该论文研究基于大语言模型的多智能体协同推荐任务，旨在解决现有方法忽视用户-物品交互协作信号的问题。提出MACF框架，通过实例化用户与物品为智能体，并由中心协调者动态管理其协作过程，提升推荐效果。**

- **链接: [https://arxiv.org/pdf/2511.18413v2](https://arxiv.org/pdf/2511.18413v2)**

> **作者:** Yu Xia; Sungchul Kim; Tong Yu; Ryan A. Rossi; Julian McAuley
>
> **摘要:** Agentic recommendations cast recommenders as large language model (LLM) agents that can plan, reason, use tools, and interact with users of varying preferences in web applications. However, most existing agentic recommender systems focus on generic single-agent plan-execute workflows or multi-agent task decomposition pipelines. Without recommendation-oriented design, they often underuse the collaborative signals in the user-item interaction history, leading to unsatisfying recommendation results. To address this, we propose the Multi-Agent Collaborative Filtering (MACF) framework for agentic recommendations, drawing an analogy between traditional collaborative filtering algorithms and LLM-based multi-agent collaboration. Specifically, given a target user and query, we instantiate similar users and relevant items as LLM agents with unique profiles. Each agent is able to call retrieval tools, suggest candidate items, and interact with other agents. Different from the static preference aggregation in traditional collaborative filtering, MACF employs a central orchestrator agent to adaptively manage the collaboration between user and item agents via dynamic agent recruitment and personalized collaboration instruction. Experimental results on datasets from three different domains show the advantages of our MACF framework compared to strong agentic recommendation baselines.
>
---
#### [replaced 005] Revisiting Intermediate-Layer Matching in Knowledge Distillation: Layer-Selection Strategy Doesn't Matter (Much)
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究知识蒸馏中的中间层匹配策略，探讨不同层选择策略对性能的影响。发现无论何种匹配方式（甚至反向匹配），学生模型表现相近，表明层选择策略影响有限，前向匹配已足够有效。**

- **链接: [https://arxiv.org/pdf/2502.04499v2](https://arxiv.org/pdf/2502.04499v2)**

> **作者:** Zony Yu; Yuqiao Wen; Lili Mou
>
> **备注:** Accepted at IJCNLP-AACL 2025
>
> **摘要:** Knowledge distillation (KD) is a popular method of transferring knowledge from a large "teacher" model to a small "student" model. Previous work has explored various layer-selection strategies (e.g., forward matching and in-order random matching) for intermediate-layer matching in KD, where a student layer is forced to resemble a certain teacher layer. In this work, we revisit such layer-selection strategies and observe an intriguing phenomenon that layer-selection strategy does not matter (much) in intermediate-layer matching -- even seemingly nonsensical matching strategies such as reverse matching still result in surprisingly good student performance. We provide an interpretation for this phenomenon by examining the angles between teacher layers viewed from the student's perspective. Our work sheds light on KD practice, as layer-selection strategies may not be the main focus of KD system design, and vanilla forward matching works well in most setups.
>
---
#### [replaced 006] CryptoBench: A Dynamic Benchmark for Expert-Level Evaluation of LLM Agents in Cryptocurrency
- **分类: cs.CL**

- **简介: 该论文提出CryptoBench，首个面向加密货币领域的动态基准，旨在评估大语言模型代理在高时效、对抗性强环境下的专业分析能力。通过专家设计的四类任务，揭示模型存在检索强但预测弱的问题。**

- **链接: [https://arxiv.org/pdf/2512.00417v4](https://arxiv.org/pdf/2512.00417v4)**

> **作者:** Jiacheng Guo; Suozhi Huang; Zixin Yao; Yifan Zhang; Yifu Lu; Jiashuo Liu; Zihao Li; Nicholas Deng; Qixin Xiao; Jia Tian; Kanghong Zhan; Tianyi Li; Xiaochen Liu; Jason Ge; Chaoyang He; Kaixuan Huang; Lin Yang; Wenhao Huang; Mengdi Wang
>
> **摘要:** This paper introduces CryptoBench, the first expert-curated, dynamic benchmark designed to rigorously evaluate the real-world capabilities of Large Language Model (LLM) agents in the uniquely demanding and fast-paced cryptocurrency domain. Unlike general-purpose agent benchmarks for search and prediction, professional crypto analysis presents specific challenges: \emph{extreme time-sensitivity}, \emph{a highly adversarial information environment}, and the critical need to synthesize data from \emph{diverse, specialized sources}, such as on-chain intelligence platforms and real-time Decentralized Finance (DeFi) dashboards. CryptoBench thus serves as a much more challenging and valuable scenario for LLM agent assessment. To address these challenges, we constructed a live, dynamic benchmark featuring 50 questions per month, expertly designed by crypto-native professionals to mirror actual analyst workflows. These tasks are rigorously categorized within a four-quadrant system: Simple Retrieval, Complex Retrieval, Simple Prediction, and Complex Prediction. This granular categorization enables a precise assessment of an LLM agent's foundational data-gathering capabilities alongside its advanced analytical and forecasting skills. Our evaluation of ten LLMs, both directly and within an agentic framework, reveals a performance hierarchy and uncovers a failure mode. We observe a \textit{retrieval-prediction imbalance}, where many leading models, despite being proficient at data retrieval, demonstrate a pronounced weakness in tasks requiring predictive analysis. This highlights a problematic tendency for agents to appear factually grounded while lacking the deeper analytical capabilities to synthesize information.
>
---
#### [replaced 007] Studying the Effects of Collaboration in Interactive Theme Discovery Systems
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于人机交互与NLP交叉任务，旨在解决不同协作模式对主题发现系统效果的影响问题。作者提出评估框架，比较同步与异步协作下两种NLP辅助工具在一致性、凝聚性和正确性上的差异。**

- **链接: [https://arxiv.org/pdf/2408.09030v3](https://arxiv.org/pdf/2408.09030v3)**

> **作者:** Alvin Po-Chun Chen; Dananjay Srinivas; Rohan Das; Alexandra Barry; Maksim Seniw; Maria Leonor Pacheco
>
> **备注:** Added author in pre-print
>
> **摘要:** NLP-assisted solutions have gained considerable traction to support qualitative data analysis. However, there does not exist a unified evaluation framework that can account for the many different settings in which qualitative researchers may employ them. In this paper, we take a first step in this direction by proposing an evaluation framework to study the way in which different tools may result in different outcomes depending on the collaboration strategy employed. Specifically, we study the impact of synchronous vs. asynchronous collaboration using two different NLP-assisted qualitative research tools and present a comprehensive analysis of significant differences in the consistency, cohesiveness, and correctness of their outputs.
>
---
#### [replaced 008] TCNN: Triple Convolutional Neural Network Models for Retrieval-based Question Answering System in E-commerce
- **分类: cs.LG; cs.CL**

- **简介: 该论文针对电商问答系统中的检索式QA任务，旨在提升查询与知识库条目的语义匹配效果。提出TCNN及两种ATCNN模型，用于重排序检索结果，增强答案召回准确率。**

- **链接: [https://arxiv.org/pdf/2004.10919v2](https://arxiv.org/pdf/2004.10919v2)**

> **作者:** Shuangyong Song; Chao Wang
>
> **备注:** 2 pages
>
> **摘要:** Automatic question-answering (QA) systems have boomed during last few years, and commonly used techniques can be roughly categorized into Information Retrieval (IR)-based and generation-based. A key solution to the IR based models is to retrieve the most similar knowledge entries of a given query from a QA knowledge base, and then rerank those knowledge entries with semantic matching models. In this paper, we aim to improve an IR based e-commerce QA system-AliMe with proposed text matching models, including a basic Triple Convolutional Neural Network (TCNN) model and two Attention-based TCNN (ATCNN) models. Experimental results show their effect.
>
---
#### [replaced 009] Enhanced Sentiment Interpretation via a Lexicon-Fuzzy-Transformer Framework
- **分类: cs.CL; cs.AI**

- **简介: 该论文属情感分析任务，旨在解决非正式文本中情感极性与强度识别难的问题。提出融合词典、模糊逻辑与轻量Transformer的框架，通过VADER初判、DistilBERT优化与模糊系统映射，生成连续情感得分，提升细粒度与准确性。**

- **链接: [https://arxiv.org/pdf/2510.15843v2](https://arxiv.org/pdf/2510.15843v2)**

> **作者:** Shayan Rokhva; Mousa Alizadeh; Maryam Abdollahi Shamami
>
> **备注:** The manuscript was uploaded in error and is scientifically invalid. It is an incomplete draft with major flaws. Co-authors were not aware of or consenting to this submission and do not endorse it
>
> **摘要:** Accurately detecting sentiment polarity and intensity in product reviews and social media posts remains challenging due to informal and domain-specific language. To address this, we propose a novel hybrid lexicon-fuzzy-transformer framework that combines rule-based heuristics, contextual deep learning, and fuzzy logic to generate continuous sentiment scores reflecting both polarity and strength. The pipeline begins with VADER-based initial sentiment estimations, which are refined through a two-stage adjustment process. This involves leveraging confidence scores from DistilBERT, a lightweight transformer and applying fuzzy logic principles to mitigate excessive neutrality bias and enhance granularity. A custom fuzzy inference system then maps the refined scores onto a 0 to 1 continuum, producing expert)like judgments. The framework is rigorously evaluated on four domain-specific datasets. food delivery, e-commerce, tourism, and fashion. Results show improved alignment with user ratings, better identification of sentiment extremes, and reduced misclassifications. Both quantitative metrics (distributional alignment, confusion matrices) and qualitative insights (case studies, runtime analysis) affirm the models robustness and efficiency. This work demonstrates the value of integrating symbolic reasoning with neural models for interpretable, finegrained sentiment analysis in linguistically dynamic domains.
>
---
#### [replaced 010] The Vector Grounding Problem
- **分类: cs.CL**

- **简介: 该论文探讨大语言模型是否能脱离人类解释，真正指代外部现实。它提出“向量接地问题”，主张通过因果信息联系与功能演化历史实现指称接地，并论证LLM即使无多模态或具身也能满足条件。**

- **链接: [https://arxiv.org/pdf/2304.01481v3](https://arxiv.org/pdf/2304.01481v3)**

> **作者:** Dimitri Coelho Mollo; Raphaël Millière
>
> **备注:** Accepted for publication in Philosophy and the Mind Sciences
>
> **摘要:** Large language models (LLMs) produce seemingly meaningful outputs, yet they are trained on text alone without direct interaction with the world. This leads to a modern variant of the classical symbol grounding problem in AI: can LLMs' internal states and outputs be about extra-linguistic reality, independently of the meaning human interpreters project onto them? We argue that they can. We first distinguish referential grounding -- the connection between a representation and its worldly referent -- from other forms of grounding and argue it is the only kind essential to solving the problem. We contend that referential grounding is achieved when a system's internal states satisfy two conditions derived from teleosemantic theories of representation: (1) they stand in appropriate causal-informational relations to the world, and (2) they have a history of selection that has endowed them with the function of carrying this information. We argue that LLMs can meet both conditions, even without multimodality or embodiment.
>
---
#### [replaced 011] SAFT: Structure-Aware Fine-Tuning of LLMs for AMR-to-Text Generation
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究AMR到文本生成任务，旨在解决现有方法忽略图结构的问题。作者提出SAFT，通过磁拉普拉斯算子计算方向敏感的位置编码，注入预训练大模型，保持架构不变，提升生成性能，在AMR 3.0上达到新SOTA。**

- **链接: [https://arxiv.org/pdf/2507.13381v2](https://arxiv.org/pdf/2507.13381v2)**

> **作者:** Rafiq Kamel; Filippo Guerranti; Simon Geisler; Stephan Günnemann
>
> **备注:** Accepted at the KDD2025 Workshop on Structured Knowledge for LLMs
>
> **摘要:** Large Language Models (LLMs) are increasingly applied to tasks involving structured inputs such as graphs. Abstract Meaning Representations (AMRs), which encode rich semantics as directed graphs, offer a rigorous testbed for evaluating LLMs on text generation from such structures. Yet, current methods often arbitrarily linearize AMRs, discarding key structural cues, or rely on architectures incompatible with standard LLMs. We introduce SAFT, a structure-aware fine-tuning approach that injects graph topology into pretrained LLMs without architectural changes. We compute direction-sensitive positional encodings from the magnetic Laplacian of transformed AMRs and project them into the embedding space of the LLM. While possibly applicable to any graph-structured inputs, we focus on AMR-to-text generation as a representative and challenging benchmark. SAFT sets a new state-of-the-art on AMR 3.0 with a 3.5 BLEU improvement over baselines. Gains scale with graph complexity, highlighting the value of structure-aware representations in enhancing LLM performance. SAFT offers a general and effective pathway for bridging structured data and language models.
>
---
#### [replaced 012] Attention Sinks in Diffusion Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究扩散语言模型中的注意力机制，探究其在生成过程中是否出现“注意力沉降”现象。通过实证分析，发现DLM存在动态变化的沉降位置，且对沉降移除鲁棒，揭示了其与自回归模型在注意力使用上的本质差异。**

- **链接: [https://arxiv.org/pdf/2510.15731v2](https://arxiv.org/pdf/2510.15731v2)**

> **作者:** Maximo Eduardo Rulli; Simone Petruzzi; Edoardo Michielon; Fabrizio Silvestri; Simone Scardapane; Alessio Devoto
>
> **摘要:** Masked Diffusion Language Models (DLMs) have recently emerged as a promising alternative to traditional Autoregressive Models (ARMs). DLMs employ transformer encoders with bidirectional attention, enabling parallel token generation while maintaining competitive performance. Although their efficiency and effectiveness have been extensively studied, the internal mechanisms that govern DLMs remain largely unexplored. In this work, we conduct an empirical analysis of DLM attention patterns, focusing on the attention sinking phenomenon, an effect previously observed in various transformer-based architectures. Our findings reveal that DLMs also exhibit attention sinks, but with distinct characteristics. First, unlike in ARMs, the sink positions in DLMs tend to shift throughout the generation process, displaying a dynamic behaviour. Second, while ARMs are highly sensitive to the removal of attention sinks, DLMs remain robust: masking sinks leads to only a minor degradation in performance. These results provide new insights into the inner workings of diffusion-based language models and highlight fundamental differences in how they allocate and utilize attention compared to autoregressive models.
>
---
#### [replaced 013] Generalised Medical Phrase Grounding
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出通用医学短语定位（GMPG）任务，解决传统方法无法处理多区域、非可定位短语的问题。作者构建MedGrounder模型，支持零个、一个或多个区域定位，通过两阶段训练，在少标注下实现优越性能，并可与现有报告生成模型结合生成定位报告。**

- **链接: [https://arxiv.org/pdf/2512.01085v2](https://arxiv.org/pdf/2512.01085v2)**

> **作者:** Wenjun Zhang; Shekhar S. Chandra; Aaron Nicolson
>
> **备注:** 10 pages
>
> **摘要:** Medical phrase grounding (MPG) maps textual descriptions of radiological findings to corresponding image regions. These grounded reports are easier to interpret, especially for non-experts. Existing MPG systems mostly follow the referring expression comprehension (REC) paradigm and return exactly one bounding box per phrase. Real reports often violate this assumption. They contain multi-region findings, non-diagnostic text, and non-groundable phrases, such as negations or descriptions of normal anatomy. Motivated by this, we reformulate the task as generalised medical phrase grounding (GMPG), where each sentence is mapped to zero, one, or multiple scored regions. To realise this formulation, we introduce the first GMPG model: MedGrounder. We adopted a two-stage training regime: pre-training on report sentence--anatomy box alignment datasets and fine-tuning on report sentence--human annotated box datasets. Experiments on PadChest-GR and MS-CXR show that MedGrounder achieves strong zero-shot transfer and outperforms REC-style and grounded report generation baselines on multi-region and non-groundable phrases, while using far fewer human box annotations. Finally, we show that MedGrounder can be composed with existing report generators to produce grounded reports without retraining the generator.
>
---
#### [replaced 014] Revealing economic facts: LLMs know more than they say
- **分类: cs.CL; cs.LG; econ.GN**

- **简介: 该论文研究利用大语言模型（LLM）的隐藏状态来估计和补全经济金融数据，属于数据估算与补全任务。发现隐藏状态比文本输出蕴含更丰富的经济信息，仅需少量标注数据即可训练出高性能线性模型，并提出无需标注的迁移学习方法，验证了其在超分辨率和数据填补中的实用性。**

- **链接: [https://arxiv.org/pdf/2505.08662v2](https://arxiv.org/pdf/2505.08662v2)**

> **作者:** Marcus Buckmann; Quynh Anh Nguyen; Edward Hill
>
> **备注:** 34 pages, 17 figures
>
> **摘要:** We investigate whether the hidden states of large language models (LLMs) can be used to estimate and impute economic and financial statistics. Focusing on county-level (e.g. unemployment) and firm-level (e.g. total assets) variables, we show that a simple linear model trained on the hidden states of open-source LLMs outperforms the models' text outputs. This suggests that hidden states capture richer economic information than the responses of the LLMs reveal directly. A learning curve analysis indicates that only a few dozen labelled examples are sufficient for training. We also propose a transfer learning method that improves estimation accuracy without requiring any labelled data for the target variable. Finally, we demonstrate the practical utility of hidden-state representations in super-resolution and data imputation tasks.
>
---
#### [replaced 015] Two Causal Principles for Improving Visual Dialog
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对视觉对话（VisDial）任务，指出先前模型忽略的两个因果问题：对话历史引入的偏差和未观测混淆因子导致的虚假相关。作者提出两条因果原则及干预算法，可普遍提升现有模型性能，且与模型无关。**

- **链接: [https://arxiv.org/pdf/1911.10496v3](https://arxiv.org/pdf/1911.10496v3)**

> **作者:** Jiaxin Qi; Yulei Niu; Jianqiang Huang; Hanwang Zhang
>
> **备注:** Accepted by CVPR 2020
>
> **摘要:** This paper unravels the design tricks adopted by us, the champion team MReaL-BDAI, for Visual Dialog Challenge 2019: two causal principles for improving Visual Dialog (VisDial). By "improving", we mean that they can promote almost every existing VisDial model to the state-of-the-art performance on the leader-board. Such a major improvement is only due to our careful inspection on the causality behind the model and data, finding that the community has overlooked two causalities in VisDial. Intuitively, Principle 1 suggests: we should remove the direct input of the dialog history to the answer model, otherwise a harmful shortcut bias will be introduced; Principle 2 says: there is an unobserved confounder for history, question, and answer, leading to spurious correlations from training data. In particular, to remove the confounder suggested in Principle 2, we propose several causal intervention algorithms, which make the training fundamentally different from the traditional likelihood estimation. Note that the two principles are model-agnostic, so they are applicable in any VisDial model. The code is available at https://github.com/simpleshinobu/visdial-principles.
>
---
#### [replaced 016] O-Mem: Omni Memory System for Personalized, Long Horizon, Self-Evolving Agents
- **分类: cs.CL**

- **简介: 该论文针对LLM智能体在长期交互中缺乏上下文一致性和动态个性化的问题，提出O-Mem记忆系统。通过主动用户建模，实现人格属性与情境的分层检索，提升个性化响应的连贯性与效率，在两个基准上均超越现有方法。**

- **链接: [https://arxiv.org/pdf/2511.13593v3](https://arxiv.org/pdf/2511.13593v3)**

> **作者:** Piaohong Wang; Motong Tian; Jiaxian Li; Yuan Liang; Yuqing Wang; Qianben Chen; Tiannan Wang; Zhicong Lu; Jiawei Ma; Yuchen Eleanor Jiang; Wangchunshu Zhou
>
> **摘要:** Recent advancements in LLM-powered agents have demonstrated significant potential in generating human-like responses; however, they continue to face challenges in maintaining long-term interactions within complex environments, primarily due to limitations in contextual consistency and dynamic personalization. Existing memory systems often depend on semantic grouping prior to retrieval, which can overlook semantically irrelevant yet critical user information and introduce retrieval noise. In this report, we propose the initial design of O-Mem, a novel memory framework based on active user profiling that dynamically extracts and updates user characteristics and event records from their proactive interactions with agents. O-Mem supports hierarchical retrieval of persona attributes and topic-related context, enabling more adaptive and coherent personalized responses. O-Mem achieves 51.67% on the public LoCoMo benchmark, a nearly 3% improvement upon LangMem,the previous state-of-the-art, and it achieves 62.99% on PERSONAMEM, a 3.5% improvement upon A-Mem,the previous state-of-the-art. O-Mem also boosts token and interaction response time efficiency compared to previous memory frameworks. Our work opens up promising directions for developing efficient and human-like personalized AI assistants in the future.
>
---
#### [replaced 017] Vevo2: A Unified and Controllable Framework for Speech and Singing Voice Generation
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文提出Vevo2，一个统一且可控的语音与歌声生成框架。针对标注数据稀缺和可控性难题，设计双音频 tokenizer 与分阶段建模，实现文本、韵律、风格与音色的解耦控制，通过联合训练和多目标后训练提升生成质量与泛化能力。**

- **链接: [https://arxiv.org/pdf/2508.16332v2](https://arxiv.org/pdf/2508.16332v2)**

> **作者:** Xueyao Zhang; Junan Zhang; Yuancheng Wang; Chaoren Wang; Yuanzhe Chen; Dongya Jia; Zhuo Chen; Zhizheng Wu
>
> **备注:** We will release code and model checkpoints at https://github.com/open-mmlab/Amphion
>
> **摘要:** Controllable human voice generation, particularly for expressive domains like singing, remains a significant challenge. This paper introduces Vevo2, a unified framework for controllable speech and singing voice generation. To tackle issues like the scarcity of annotated singing data and to enable flexible controllability, Vevo2 introduces two audio tokenizers: (1) a unified music-notation-free prosody tokenizer that captures prosody and melody from speech, singing, and even instrumental sounds, and (2) a unified content-style tokenizer that encodes linguistic content, prosody, and style for both speech and singing, while enabling timbre disentanglement. Vevo2 consists of an auto-regressive (AR) content-style modeling stage, which aims to enable controllability over text, prosody, and style, as well as a flow-matching acoustic modeling stage that allows for timbre control. Particularly, during the speech-singing joint training of the AR model, we propose both explicit and implicit prosody learning strategies to bridge speech and singing voice. Moreover, to further enhance the Vevo2's ability to follow text and prosody, we design a multi-objective post-training task that integrates both intelligibility and prosody similarity alignment. Experimental results show that the unified modeling in Vevo2 brings mutual benefits to both speech and singing voice generation. Additionally, Vevo2's effectiveness across a wide range of synthesis, conversion, and editing tasks for both speech and singing further demonstrates its strong generalization ability and versatility. Audio samples are are available at https://versasinger.github.io/.
>
---
#### [replaced 018] Demystifying deep search: a holistic evaluation with hint-free multi-hop questions and factorised metrics
- **分类: cs.CL**

- **简介: 该论文针对RAG系统在多跳搜索任务中依赖问题提示和评估不细粒度的问题，提出无提示多跳基准WebDetective和分解式评估框架，揭示模型在自主发现推理路径上的根本缺陷，并设计EvidenceLoop流程提升表现。**

- **链接: [https://arxiv.org/pdf/2510.05137v3](https://arxiv.org/pdf/2510.05137v3)**

> **作者:** Maojia Song; Renhang Liu; Xinyu Wang; Yong Jiang; Pengjun Xie; Fei Huang; Jingren Zhou; Dorien Herremans; Soujanya Poria
>
> **摘要:** RAG (Retrieval-Augmented Generation) systems and web agents are increasingly evaluated on multi-hop deep search tasks, yet current practice suffers from two major limitations. First, most benchmarks leak the reasoning path in the question text, allowing models to follow surface cues rather than discover reasoning chains autonomously. Second, evaluation is typically reduced to a single pass rate, which collapses diverse behaviours into one score and obscures whether failures stem from inadequate search, poor knowledge use, or inappropriate refusal. To address these issues, we present WebDetective, a benchmark of hint-free multi-hop questions paired with a controlled Wikipedia sandbox that ensures full traceability of model actions, and a holistic evaluation framework that separates search sufficiency, knowledge utilisation, and refusal behaviour. Our evaluation of 25 state-of-the-art models reveals systematic weaknesses across all architectures: models struggle with knowledge utilisation despite having sufficient evidence and demonstrate near-absent appropriate refusal when evidence is lacking. These patterns expose a fundamental gap: today's systems excel at executing given reasoning paths but fail when required to discover them. We develop an agentic workflow, EvidenceLoop, that explicitly targets the challenges our benchmark identifies, incorporating verification loops and systematic evidence tracking that improve both search and synthesis capabilities. This baseline demonstrates that WebDetective's diagnostic framework can guide concrete architectural improvements, establishing our benchmark as a critical tool for developing genuinely autonomous reasoning systems rather than pattern-following agents.
>
---
#### [replaced 019] Guiding LLMs to Generate High-Fidelity and High-Quality Counterfactual Explanations for Text Classification
- **分类: cs.CL**

- **简介: 该论文属于文本分类的可解释性任务，旨在生成高保真、高质量的反事实解释。针对现有方法需微调且文本质量低的问题，提出两种无需微调的分类器引导方法，提升大模型生成反事实文本的效果，并增强分类器鲁棒性。**

- **链接: [https://arxiv.org/pdf/2503.04463v2](https://arxiv.org/pdf/2503.04463v2)**

> **作者:** Van Bach Nguyen; Christin Seifert; Jörg Schlötterer
>
> **摘要:** The need for interpretability in deep learning has driven interest in counterfactual explanations, which identify minimal changes to an instance that change a model's prediction. Current counterfactual (CF) generation methods require task-specific fine-tuning and produce low-quality text. Large Language Models (LLMs), though effective for high-quality text generation, struggle with label-flipping counterfactuals (i.e., counterfactuals that change the prediction) without fine-tuning. We introduce two simple classifier-guided approaches to support counterfactual generation by LLMs, eliminating the need for fine-tuning while preserving the strengths of LLMs. Despite their simplicity, our methods outperform state-of-the-art counterfactual generation methods and are effective across different LLMs, highlighting the benefits of guiding counterfactual generation by LLMs with classifier information. We further show that data augmentation by our generated CFs can improve a classifier's robustness. Our analysis reveals a critical issue in counterfactual generation by LLMs: LLMs rely on parametric knowledge rather than faithfully following the classifier.
>
---
#### [replaced 020] Transparent and Coherent Procedural Mistake Detection
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究程序性错误检测（PMD），旨在判断用户是否按文本步骤正确执行任务。为提升透明性与可解释性，提出生成视觉自对话理由，并构建基于VLM的基准与自动评估指标，通过NLI模型衡量理由的连贯性，改进模型的推理性能。**

- **链接: [https://arxiv.org/pdf/2412.11927v5](https://arxiv.org/pdf/2412.11927v5)**

> **作者:** Shane Storks; Itamar Bar-Yossef; Yayuan Li; Zheyuan Zhang; Jason J. Corso; Joyce Chai
>
> **备注:** EMNLP 2025
>
> **摘要:** Procedural mistake detection (PMD) is a challenging problem of classifying whether a human user (observed through egocentric video) has successfully executed a task (specified by a procedural text). Despite significant recent efforts, machine performance in the wild remains nonviable, and the reasoning processes underlying this performance are opaque. As such, we extend PMD to require generating visual self-dialog rationales to inform decisions. Given the impressive, mature image understanding capabilities observed in recent vision-and-language models (VLMs), we curate a suitable benchmark dataset for PMD based on individual frames. As our reformulation enables unprecedented transparency, we leverage a natural language inference (NLI) model to formulate two automated metrics for the coherence of generated rationales. We establish baselines for this reframed task, showing that VLMs struggle off-the-shelf, but with some trade-offs, their accuracy, coherence, and efficiency can be improved by incorporating these metrics into common inference and fine-tuning methods. Lastly, our multi-faceted metrics visualize common outcomes, highlighting areas for further improvement.
>
---
#### [replaced 021] Constrained Discrete Diffusion
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究离散扩散模型中的约束生成任务，旨在解决生成序列需满足特定约束（如安全、逻辑规则）的问题。作者提出Constrained Discrete Diffusion（CDD），将可微约束优化融入扩散过程，实现无需训练的约束满足生成，确保零违规且保持生成质量。**

- **链接: [https://arxiv.org/pdf/2503.09790v3](https://arxiv.org/pdf/2503.09790v3)**

> **作者:** Michael Cardei; Jacob K Christopher; Thomas Hartvigsen; Bhavya Kailkhura; Ferdinando Fioretto
>
> **备注:** Published at the 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
>
> **摘要:** Discrete diffusion models are a class of generative models that construct sequences by progressively denoising samples from a categorical noise distribution. Beyond their rapidly growing ability to generate coherent natural language, these models present a new and important opportunity to enforce sequence-level constraints, a capability that current autoregressive models cannot natively provide. This paper capitalizes on this opportunity by introducing Constrained Discrete Diffusion (CDD), a novel integration of differentiable constraint optimization within the diffusion process to ensure adherence to constraints, logic rules, or safety requirements for generated sequences. Unlike conventional text generators that often rely on post-hoc filtering or model retraining for controllable generation, CDD directly imposes constraints into the discrete diffusion sampling process, resulting in a training-free and effective approach. Experiments in toxicity-controlled text generation, property-constrained molecule design, and instruction-constrained text completion demonstrate that CDD achieves zero constraint violations in a diverse array of tasks while preserving fluency, novelty, and coherence while outperforming autoregressive and existing discrete diffusion approaches.
>
---
#### [replaced 022] Collaborative Causal Sensemaking: Closing the Complementarity Gap in Human-AI Decision Support
- **分类: cs.CL; cs.AI; cs.HC; cs.LG**

- **简介: 该论文属于人机协同决策支持任务，旨在解决当前AI系统缺乏协作式因果推理能力导致的人机互补性不足问题。作者提出“协作因果意义建构”（CCS）研究框架，通过新训练环境、共享心智模型和以信任与互补性为核心的评估，培育AI作为协同推理伙伴的能力。**

- **链接: [https://arxiv.org/pdf/2512.07801v3](https://arxiv.org/pdf/2512.07801v3)**

> **作者:** Raunak Jain; Mudita Khurana
>
> **备注:** Under review at the 25th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2026), Blue Sky Track
>
> **摘要:** LLM-based agents are increasingly deployed for expert decision support, yet human-AI teams in high-stakes settings do not yet reliably outperform the best individual. We argue this complementarity gap reflects a fundamental mismatch: current agents are trained as answer engines, not as partners in the collaborative sensemaking through which experts actually make decisions. Sensemaking (the ability to co-construct causal explanations, surface uncertainties, and adapt goals) is the key capability that current training pipelines do not explicitly develop or evaluate. We propose Collaborative Causal Sensemaking (CCS) as a research agenda to develop this capability from the ground up, spanning new training environments that reward collaborative thinking, representations for shared human-AI mental models, and evaluation centred on trust and complementarity. Taken together, these directions shift MAS research from building oracle-like answer engines to cultivating AI teammates that co-reason with their human partners over the causal structure of shared decisions, advancing the design of effective human-AI teams.
>
---
#### [replaced 023] An Offline Mobile Conversational Agent for Mental Health Support: Learning from Emotional Dialogues and Psychological Texts with Student-Centered Evaluation
- **分类: cs.CL; cs.AI; cs.CY; cs.HC**

- **简介: 该论文提出EmoSApp，一种离线移动端心理支持对话系统，旨在解决网络依赖与隐私问题。通过微调并量化语言模型，在学生和专家参与下验证其情感对话与心理支持能力，实现低资源设备上的高效推理。**

- **链接: [https://arxiv.org/pdf/2507.10580v2](https://arxiv.org/pdf/2507.10580v2)**

> **作者:** Vimaleswar A; Prabhu Nandan Sahu; Nilesh Kumar Sahu; Haroon R. Lone
>
> **摘要:** Mental health plays a crucial role in the overall well-being of an individual. In recent years, digital platforms have increasingly been used to expand mental health and emotional support. However, there are persistent challenges related to limited user accessibility, internet connectivity, and data privacy, which highlight the need for an offline, smartphone-based solutions. To address these challenges, we propose EmoSApp (Emotional Support App): an entirely offline, smartphone-based conversational app designed to provide mental health and emotional support. EmoSApp leverages a language model, specifically the LLaMA-3.2-1B-Instruct, which is fine-tuned and quantized on a custom-curated ``Knowledge Dataset'' comprising 14,582 mental health QA pairs along with multi-turn conversational data, enabling robust domain expertise and fully on-device inference on resource-constrained smartphones. Through qualitative evaluation with students and mental health professionals, we demonstrate that EmoSApp has the ability to respond coherently and empathetically, provide relevant suggestions to user's mental health problems, and maintain interactive dialogue. Additionally, quantitative evaluations on nine commonsense and reasoning benchmarks, along with two mental health specific datasets, demonstrate EmoSApp's effectiveness in low-resource settings. By prioritizing on-device deployment and specialized domain-specific adaptation, EmoSApp serves as a blueprint for future innovations in portable, secure, and highly tailored AI-driven mental health support.
>
---
#### [replaced 024] Leveraging Machine Learning to Identify Gendered Stereotypes and Body Image Concerns on Diet and Fitness Online Forums
- **分类: cs.SI; cs.CL; cs.CY**

- **简介: 该论文属于社交媒体内容分析任务，旨在揭示饮食健身论坛中的性别化身体意象问题。通过分析46个Reddit社区，研究不同性别导向群体的情感表达与互动模式，发现“瘦身理想”社区负面情绪更多，而“肌肉理想”社区则存在毒性赞美，据此提出针对性的内容 moderation 策略。**

- **链接: [https://arxiv.org/pdf/2407.03551v2](https://arxiv.org/pdf/2407.03551v2)**

> **作者:** Minh Duc Chu; Cinthia Sánchez; Zihao He; Rebecca Dorn; Stuart Murray; Kristina Lerman
>
> **摘要:** The pervasive expectations about ideal body types in Western society can lead to body image concerns, dissatisfaction, and in extreme cases, eating disorders and other psychopathologies related to body image. While previous research has focused on online pro-anorexia communities glorifying the "thin ideal," less attention has been given to the broader spectrum of body image concerns or how emerging disorders like muscle dysmorphia ("bigorexia") present on online platforms. To address this gap, we analyze 46 Reddit forums related to diet, fitness, and mental health. We map these communities along gender and body ideal dimensions, revealing distinct patterns of emotional expression and community support. Feminine-oriented communities, especially those endorsing the thin ideal, express higher levels of negative emotions and receive caring comments in response. In contrast, muscular ideal communities display less negativity, regardless of gender orientation, but receive aggressive compliments in response, marked by admiration and toxicity. Mental health discussions align more with thin ideal, feminine-leaning spaces. By uncovering these gendered emotional dynamics, our findings can inform the development of moderation strategies that foster supportive interactions while reducing exposure to harmful content.
>
---
#### [replaced 025] TRepLiNa: Layer-wise CKA+REPINA Alignment Improves Low-Resource Machine Translation in Aya-23 8B
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于低资源机器翻译任务，旨在解决低资源语言翻译质量差的问题。作者提出TRepLiNa方法，结合CKA与REPINA，在Aya-23 8B模型上对齐中间层表示并约束参数更新，提升零样本、少样本及微调场景下的翻译性能。**

- **链接: [https://arxiv.org/pdf/2510.06249v5](https://arxiv.org/pdf/2510.06249v5)**

> **作者:** Toshiki Nakai; Ravi Kiran Chikkala; Lena Sophie Oberkircher; Nicholas Jennings; Natalia Skachkova; Tatiana Anikina; Jesujoba Oluwadara Alabi
>
> **备注:** It is work in progress
>
> **摘要:** The 2025 Multimodal Models for Low-Resource Contexts and Social Impact (MMLoSo) Language Challenge addresses one of India's most pressing linguistic gaps: the lack of resources for its diverse low-resource languages (LRLs). In this study, we investigate whether enforcing cross-lingual similarity in specific internal layers of a decoder-only multilingual large language model (LLM) can improve translation quality from LRL to high-resource language (HRL). Specifically, we combine Centered Kernel Alignment (CKA), a similarity metric that encourages representations of different languages to align, with REPINA, a regularization method that constrains parameter updates to remain close to the pretrained model, into a joint method we call TRepLiNa. In this research project, we experiment with zero-shot, few-shot, and fine-tuning settings using Aya-23 8B with QLoRA across MMLoSo shared task language pairs (Mundari, Santali, Bhili) with Hindi/English pivots. Our results show that aligning mid-level layers using TRepLiNa (CKA+REPINA) is a low-cost, practical approach to improving LRL translation, especially in data-scarce settings.
>
---
#### [replaced 026] Enhancing Reasoning Skills in Small Persian Medical Language Models Can Outperform Large-Scale Data Training
- **分类: cs.CL**

- **简介: 该论文属医疗问答任务，旨在提升小规模波斯语医学语言模型的推理能力。针对数据稀缺问题，采用RLAIF与DPO方法构建正负推理样本，通过链式思维提示生成高质量推理轨迹，显著提升模型性能，小数据下超越大规模训练基线模型。**

- **链接: [https://arxiv.org/pdf/2510.20059v4](https://arxiv.org/pdf/2510.20059v4)**

> **作者:** Mehrdad Ghassabi; Sadra Hakim; Hamidreza Baradaran Kashani; Pedram Rostami
>
> **备注:** 7 pages, 5 figures
>
> **摘要:** Enhancing reasoning capabilities in small language models is critical for specialized applications such as medical question answering, particularly in underrepresented languages like Persian. In this study, we employ Reinforcement Learning with AI Feedback (RLAIF) and Direct preference optimization (DPO) to improve the reasoning skills of a general-purpose Persian language model. To achieve this, we translated a multiple-choice medical question-answering dataset into Persian and used RLAIF to generate rejected-preferred answer pairs, which are essential for DPO training. By prompting both teacher and student models to produce Chain-of-Thought (CoT) reasoning responses, we compiled a dataset containing correct and incorrect reasoning trajectories. This dataset, comprising 2 million tokens in preferred answers and 2.5 million tokens in rejected ones, was used to train a baseline model, significantly enhancing its medical reasoning capabilities in Persian. Remarkably, the resulting model outperformed its predecessor, gaokerena-V, which was trained on approximately 57 million tokens, despite leveraging a much smaller dataset. These results highlight the efficiency and effectiveness of reasoning-focused training approaches in developing domain-specific language models with limited data availability.
>
---
#### [replaced 027] Neural Diversity Regularizes Hallucinations in Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属语言模型可靠性任务，旨在减少幻觉问题。提出神经多样性机制，通过ND-LoRA方法引入去相关并行表示，在不增参数与数据下显著降低幻觉，揭示神经多样性为可扩展性新维度。**

- **链接: [https://arxiv.org/pdf/2510.20690v2](https://arxiv.org/pdf/2510.20690v2)**

> **作者:** Kushal Chakrabarti; Nirmal Balachundhar
>
> **摘要:** Language models continue to hallucinate despite increases in parameters, compute, and data. We propose neural diversity -- decorrelated parallel representations -- as a principled mechanism that reduces hallucination rates at fixed parameter and data budgets. While existing mitigation strategies largely target accuracy, we provide the first formal tail bounds for hallucination probability in ensembled language models, reframing it as a second-moment reliability problem and explaining 94.3% of empirical reliability variation seen across parallel configurations. We introduce ND-LoRA (Neural Diversity Low-Rank Adaptation), combining parallel LoRA adapters with Barlow Twins regularization, and reduce hallucinations by up to 25.6% (and 14.6% on average) while preserving general accuracy. Ablations show LoRA adapters and regularization act synergistically, causal interventions prove neurodiversity as the mediating factor and correlational studies indicate scale: a 0.1% neural correlation increase is associated with a 3.8% hallucination increase. Finally, task-dependent optimality emerges: different tasks require different optimal amounts of neurodiversity. Together, our results highlight neural diversity as a third axis of scaling -- orthogonal to parameters and data -- to improve the reliability of language models at fixed budgets.
>
---
#### [replaced 028] Open ASR Leaderboard: Towards Reproducible and Transparent Multilingual Speech Recognition Evaluation
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文聚焦ASR（自动语音识别）评估任务，旨在解决现有评测缺乏多语言支持、效率指标缺失及不可复现的问题。作者构建了Open ASR Leaderboard，开源了包含60多个系统的标准化评测基准，统一文本归一化，报告WER和RTFx，支持可复现、透明的多语言ASR性能比较。**

- **链接: [https://arxiv.org/pdf/2510.06961v3](https://arxiv.org/pdf/2510.06961v3)**

> **作者:** Vaibhav Srivastav; Steven Zheng; Eric Bezzam; Eustache Le Bihan; Adel Moumen; Sanchit Gandhi
>
> **备注:** Leaderboard: https://huggingface.co/spaces/hf-audio/open_asr_leaderboard ; Code: https://github.com/huggingface/open_asr_leaderboard
>
> **摘要:** Despite rapid progress, ASR evaluation remains saturated with short-form English, and efficiency is rarely reported. We present the Open ASR Leaderboard, a fully reproducible benchmark and interactive leaderboard comparing 60+ open-source and proprietary systems across 11 datasets, including a dedicated multilingual track. We standardize text normalization and report both word error rate (WER) and inverse real-time factor (RTFx), enabling fair accuracy-efficiency comparisons. For English transcription, Conformer encoders paired with LLM decoders achieve the best average WER but are slower, while CTC and TDT decoders deliver much better RTFx, making them attractive for long-form and offline use. Whisper-derived encoders fine-tuned for English improve accuracy but often trade off multilingual coverage. All code and dataset loaders are open-sourced to support transparent, extensible evaluation.
>
---
#### [replaced 029] Low-Dimensional Structure in the Space of Language Representations is Reflected in Brain Responses
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究不同语言模型表示间的关联，提出一种迁移学习方法分析100种特征空间，发现其存在低维结构，并可用于预测脑响应，揭示大脑语言处理层次。**

- **链接: [https://arxiv.org/pdf/2106.05426v5](https://arxiv.org/pdf/2106.05426v5)**

> **作者:** Richard Antonello; Javier Turek; Vy Vo; Alexander Huth
>
> **备注:** Accepted to the Advances in Neural Information Processing Systems 34 (2021) Revised to include voxel selection details
>
> **摘要:** How related are the representations learned by neural language models, translation models, and language tagging tasks? We answer this question by adapting an encoder-decoder transfer learning method from computer vision to investigate the structure among 100 different feature spaces extracted from hidden representations of various networks trained on language tasks. This method reveals a low-dimensional structure where language models and translation models smoothly interpolate between word embeddings, syntactic and semantic tasks, and future word embeddings. We call this low-dimensional structure a language representation embedding because it encodes the relationships between representations needed to process language for a variety of NLP tasks. We find that this representation embedding can predict how well each individual feature space maps to human brain responses to natural language stimuli recorded using fMRI. Additionally, we find that the principal dimension of this structure can be used to create a metric which highlights the brain's natural language processing hierarchy. This suggests that the embedding captures some part of the brain's natural language representation structure.
>
---
#### [replaced 030] GRAVITY: A Framework for Personalized Text Generation via Profile-Grounded Synthetic Preferences
- **分类: cs.CL**

- **简介: 该论文聚焦LLM个性化生成任务，旨在减少对人工标注的依赖。提出GRAVITY框架，利用用户画像生成合成偏好数据，融合文化、心理等模型构建多维度偏好对，提升跨文化场景下的个性化文本生成效果。**

- **链接: [https://arxiv.org/pdf/2510.11952v2](https://arxiv.org/pdf/2510.11952v2)**

> **作者:** Priyanka Dey; Daniele Rosa; Wenqing Zheng; Daniel Barcklow; Jieyu Zhao; Emilio Ferrara
>
> **摘要:** Personalization in LLMs often relies on costly human feedback or interaction logs, limiting scalability and neglecting deeper user attributes. To reduce the reliance on human annotations, we introduce GRAVITY (Generative Response with Aligned Values, Interests, and Traits of You), a framework for generating synthetic, profile-grounded preference data that captures users' interests, values, beliefs, and personality traits. By integrating demographic, cultural, and psychological frameworks -- including Hofstede's cultural dimensions, Schwartz's basic values, the World Values Survey, and Big Five OCEAN traits -- GRAVITY synthesizes preference pairs to guide personalized content generation. We evaluate GRAVITY on book descriptions for 400 Amazon users, comparing it to prompt-based conditioning, standard fine-tuning, and naive synthetic pair generation. Profile-grounded synthetic data consistently improves generation, especially across multiple cultures (USA, Brazil, Japan, India), achieving over 4% higher preference gains across baselines, with user studies showing that GRAVITY outputs are preferred over 86% of the time. Our results show that scenario-grounded synthetic data can capture richer user variation, reduce reliance on costly annotation, and produce more engaging, user-centered content, offering a scalable path for LLM personalization.
>
---
#### [replaced 031] Forgetting-MarI: LLM Unlearning via Marginal Information Regularization
- **分类: cs.AI; cs.CL; cs.CR; cs.IT; cs.LG**

- **简介: 该论文属于模型遗忘任务，旨在解决大语言模型中指定数据影响的移除问题。提出Forgetting-MarI框架，通过边际信息正则化，选择性遗忘目标数据带来的额外信息，保留其余知识，在保证遗忘彻底性的同时最小化对模型性能的影响。**

- **链接: [https://arxiv.org/pdf/2511.11914v2](https://arxiv.org/pdf/2511.11914v2)**

> **作者:** Shizhou Xu; Yuan Ni; Stefan Broecker; Thomas Strohmer
>
> **摘要:** As AI models are trained on ever-expanding datasets, the ability to remove the influence of specific data from trained models has become essential for privacy protection and regulatory compliance. Unlearning addresses this challenge by selectively removing parametric knowledge from the trained models without retraining from scratch, which is critical for resource-intensive models such as Large Language Models (LLMs). Existing unlearning methods often degrade model performance by removing more information than necessary when attempting to ''forget'' specific data. We introduce Forgetting-MarI, an LLM unlearning framework that provably removes only the additional (marginal) information contributed by the data to be unlearned, while preserving the information supported by the data to be retained. By penalizing marginal information, our method yields an explicit upper bound on the unlearn dataset's residual influence in the trained models, providing provable undetectability. Extensive experiments confirm that our approach outperforms current state-of-the-art unlearning methods, delivering reliable forgetting and better preserved general model performance across diverse benchmarks. This advancement represents an important step toward making AI systems more controllable and compliant with privacy and copyright regulations without compromising their effectiveness.
>
---
#### [replaced 032] Make LVLMs Focus: Context-Aware Attention Modulation for Better Multimodal In-Context Learning
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究多模态上下文学习中大视觉语言模型注意力机制的局限性，提出无需训练的上下文感知调制注意力方法（CAMA），通过动态调整注意力增强对关键视觉和语义信息的关注，提升模型在多种任务下的稳定性和性能。**

- **链接: [https://arxiv.org/pdf/2505.17097v4](https://arxiv.org/pdf/2505.17097v4)**

> **作者:** Yanshu Li; Jianjiang Yang; Ziteng Yang; Bozheng Li; Ligong Han; Hongyang He; Zhengtao Yao; Yingjie Victor Chen; Songlin Fei; Dongfang Liu; Ruixiang Tang
>
> **备注:** 14 pages, 8 figures, 5 tables
>
> **摘要:** Multimodal in-context learning (ICL) is becoming a key capability that allows large vision-language models (LVLMs) to adapt to novel tasks without parameter updates, which expands their usefulness in many real-world applications. However, ICL performance remains unstable even when the in-context demonstrations (ICDs) are well matched, showing that LVLMs still struggle to make full use of the provided context. While existing work mainly focuses on prompt engineering or post-hoc logit calibration, we study the attention mechanisms inside LVLMs to address their inherent limitations. We identify two important weaknesses in their self-attention that hinder effective ICL. To address these weaknesses, we propose Context-Aware Modulated Attention (CAMA), a training-free and plug-and-play method that dynamically adjusts attention logits based on the input in-context sequence. CAMA uses a two-stage modulation process that strengthens attention to semantically important tokens, especially visual ones. Across four LVLMs and seven benchmarks, CAMA consistently outperforms vanilla models and baselines, showing clear effectiveness and generalization. It can also activate the intended benefits of prompt engineering methods and remains robust across different sequence configurations. Therefore, CAMA opens up new directions for improving multimodal reasoning through a deeper understanding of attention dynamics.
>
---
#### [replaced 033] Ineffectiveness for Search and Undecidability of PCSP Meta-Problems
- **分类: cs.CC; cs.CL; cs.DS; cs.LO**

- **简介: 该论文研究承诺约束满足问题（PCSP）的搜索与决策版本的等价性问题，证明主流算法（如BLP、AIP）在搜索任务上无效，并利用代数方法揭示相关元问题是不可判定的。**

- **链接: [https://arxiv.org/pdf/2504.04639v3](https://arxiv.org/pdf/2504.04639v3)**

> **作者:** Alberto Larrauri
>
> **摘要:** It is an open question whether the search and decision versions of promise CSPs are equivalent. Most known algorithms for PCSPs solve only their \emph{decision} variant, and it is unknown whether they can be adapted to solve \emph{search} as well. The main approaches, called BLP, AIP and BLP+AIP, handle a PCSP by finding a solution to a relaxation of some integer program. We prove that rounding those solutions to a proper search certificate can be as hard as any problem in the class TFNP. In other words, these algorithms are ineffective for search. Building on the algebraic approach to PCSPs, we find sufficient conditions that imply ineffectiveness for search. Our tools are tailored to algorithms that are characterized by minions in a suitable way, and can also be used to prove undecidability results for meta-problems. This way, we show that the families of templates solvable via BLP, AIP, and BLP+AIP are undecidable. Using the same techniques we also analyze several algebraic conditions that are known to guarantee the tractability of finite-template CSPs. We prove that several meta-problems related to cyclic polymorphims and WNUs are undecidable for PCSPs. In particular, there is no algorithm deciding whether a finite PCSP template (1) admits cyclic a polymorphism, (2) admits a WNU.
>
---
#### [replaced 034] Improving Topic Relevance Model by Mix-structured Summarization and LLM-based Data Augmentation
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于搜索相关性建模任务，旨在提升查询与文档的主题匹配度。针对文本冗长和标注数据不足问题，提出混合结构摘要输入和基于大模型的数据增强方法，有效提升了模型性能。**

- **链接: [https://arxiv.org/pdf/2404.02616v2](https://arxiv.org/pdf/2404.02616v2)**

> **作者:** Yizhu Liu; Ran Tao; Shengyu Guo; Yifan Yang
>
> **摘要:** Topic relevance between query and document is a very important part of social search, which can evaluate the degree of matching between document and user's requirement. In most social search scenarios such as Dianping, modeling search relevance always faces two challenges. One is that many documents in social search are very long and have much redundant information. The other is that the training data for search relevance model is difficult to get, especially for multi-classification relevance model. To tackle above two problems, we first take query concatenated with the query-based summary and the document summary without query as the input of topic relevance model, which can help model learn the relevance degree between query and the core topic of document. Then, we utilize the language understanding and generation abilities of large language model (LLM) to rewrite and generate query from queries and documents in existing training data, which can construct new query-document pairs as training data. Extensive offline experiments and online A/B tests show that the proposed approaches effectively improve the performance of relevance modeling.
>
---
