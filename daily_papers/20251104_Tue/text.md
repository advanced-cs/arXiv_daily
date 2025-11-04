# 自然语言处理 cs.CL

- **最新发布 127 篇**

- **更新 115 篇**

## 最新发布

#### [new 001] Surfacing Subtle Stereotypes: A Multilingual, Debate-Oriented Evaluation of Modern LLMs
- **分类: cs.CL; cs.CY**

- **简介: 该论文提出多语言辩论基准DebateBias-8K，揭示LLMs在开放生成中隐含的刻板印象，发现主流模型在低资源语言中偏见更严重，指出当前对齐方法未能有效缓解跨文化偏见。**

- **链接: [http://arxiv.org/pdf/2511.01187v1](http://arxiv.org/pdf/2511.01187v1)**

> **作者:** Muhammed Saeed; Muhammad Abdul-mageed; Shady Shehata
>
> **摘要:** Large language models (LLMs) are widely deployed for open-ended communication, yet most bias evaluations still rely on English, classification-style tasks. We introduce DebateBias-8K, a new multilingual, debate-style benchmark designed to reveal how narrative bias appears in realistic generative settings. Our dataset includes 8,400 structured debate prompts spanning four sensitive domains: women's rights, socioeconomic development, terrorism, and religion, across seven languages ranging from high-resource (English, Chinese) to low-resource (Swahili, Nigerian Pidgin). Using four flagship models (GPT-4o, Claude 3, DeepSeek, and LLaMA 3), we generate and automatically classify over 100,000 responses. Results show that all models reproduce entrenched stereotypes despite safety alignment: Arabs are overwhelmingly linked to terrorism and religion (>=95%), Africans to socioeconomic "backwardness" (up to <=77%), and Western groups are consistently framed as modern or progressive. Biases grow sharply in lower-resource languages, revealing that alignment trained primarily in English does not generalize globally. Our findings highlight a persistent divide in multilingual fairness: current alignment methods reduce explicit toxicity but fail to prevent biased outputs in open-ended contexts. We release our DebateBias-8K benchmark and analysis framework to support the next generation of multilingual bias evaluation and safer, culturally inclusive model alignment.
>
---
#### [new 002] Math anxiety and associative knowledge structure are entwined in psychology students but not in Large Language Models like GPT-3.5 and GPT-4o
- **分类: cs.CL; cs.CY**

- **简介: 该论文探究心理学学生与GPT模型在数学焦虑关联认知结构上的差异，通过行为形式网络分析发现：人类学生中数学与焦虑的负面关联可预测焦虑水平，而GPT模型无此模式，揭示情感认知在人类焦虑中的关键作用。**

- **链接: [http://arxiv.org/pdf/2511.01558v1](http://arxiv.org/pdf/2511.01558v1)**

> **作者:** Luciana Ciringione; Emma Franchino; Simone Reigl; Isaia D'Onofrio; Anna Serbati; Oleksandra Poquet; Florence Gabriel; Massimo Stella
>
> **摘要:** Math anxiety poses significant challenges for university psychology students, affecting their career choices and overall well-being. This study employs a framework based on behavioural forma mentis networks (i.e. cognitive models that map how individuals structure their associative knowledge and emotional perceptions of concepts) to explore individual and group differences in the perception and association of concepts related to math and anxiety. We conducted 4 experiments involving psychology undergraduates from 2 samples (n1 = 70, n2 = 57) compared against GPT-simulated students (GPT-3.5: n2 = 300; GPT-4o: n4 = 300). Experiments 1, 2, and 3 employ individual-level network features to predict psychometric scores for math anxiety and its facets (observational, social and evaluational) from the Math Anxiety Scale. Experiment 4 focuses on group-level perceptions extracted from human students, GPT-3.5 and GPT-4o's networks. Results indicate that, in students, positive valence ratings and higher network degree for "anxiety", together with negative ratings for "math", can predict higher total and evaluative math anxiety. In contrast, these models do not work on GPT-based data because of differences in simulated networks and psychometric scores compared to humans. These results were also reconciled with differences found in the ways that high/low subgroups of simulated and real students framed semantically and emotionally STEM concepts. High math-anxiety students collectively framed "anxiety" in an emotionally polarising way, absent in the negative perception of low math-anxiety students. "Science" was rated positively, but contrasted against the negative perception of "math". These findings underscore the importance of understanding concept perception and associations in managing students' math anxiety.
>
---
#### [new 003] AgentBnB: A Browser-Based Cybersecurity Tabletop Exercise with Large Language Model Support and Retrieval-Aligned Scaffolding
- **分类: cs.CL; cs.CR**

- **简介: 论文提出AgentBnB，一种基于浏览器的AI辅助网络安全桌面演练系统，利用大语言模型提供动态认知提示，解决传统演练可扩展性差、资源密集的问题，实现轻量级、可重复的训练。**

- **链接: [http://arxiv.org/pdf/2511.00265v1](http://arxiv.org/pdf/2511.00265v1)**

> **作者:** Arman Anwar; Zefang Liu
>
> **摘要:** Traditional cybersecurity tabletop exercises (TTXs) provide valuable training but are often scripted, resource-intensive, and difficult to scale. We introduce AgentBnB, a browser-based re-imagining of the Backdoors & Breaches game that integrates large language model teammates with a Bloom-aligned, retrieval-augmented copilot (C2D2). The system expands a curated corpus into factual, conceptual, procedural, and metacognitive snippets, delivering on-demand, cognitively targeted hints. Prompt-engineered agents employ a scaffolding ladder that gradually fades as learner confidence grows. In a solo-player pilot with four graduate students, participants reported greater intention to use the agent-based version compared to the physical card deck and viewed it as more scalable, though a ceiling effect emerged on a simple knowledge quiz. Despite limitations of small sample size, single-player focus, and narrow corpus, these early findings suggest that large language model augmented TTXs can provide lightweight, repeatable practice without the logistical burden of traditional exercises. Planned extensions include multi-player modes, telemetry-driven coaching, and comparative studies with larger cohorts.
>
---
#### [new 004] Prompt-R1: Collaborative Automatic Prompting Framework via End-to-end Reinforcement Learning
- **分类: cs.CL**

- **简介: Prompt-R1提出一种端到端强化学习框架，用小型LLM自动生成提示，协作大型LLM解决复杂问题，替代人工提示，提升推理准确性与生成质量，支持即插即用。**

- **链接: [http://arxiv.org/pdf/2511.01016v1](http://arxiv.org/pdf/2511.01016v1)**

> **作者:** Wenjin Liu; Haoran Luo; Xueyuan Lin; Haoming Liu; Tiesunlong Shen; Jiapu Wang; Rui Mao; Erik Cambria
>
> **摘要:** Recently, advanced large language models (LLMs) have emerged at an increasingly rapid pace. However, when faced with complex problems, most users are often unable to provide accurate and effective prompts to interact with LLMs, thus limiting the performance of LLMs. To address this challenge, we propose Prompt-R1, an end-to-end reinforcement learning framework that uses a small-scale LLM to collaborate with large-scale LLMs, replacing user interaction to solve problems better. This collaboration is cast as a multi-turn prompt interaction, where the small-scale LLM thinks and generates prompts, and the large-scale LLM performs complex reasoning. A dual-constrained reward is designed to optimize for correctness, generation quality, and reasoning accuracy. Prompt-R1 provides a plug-and-play framework that supports both inference and training with various large-scale LLMs. Experiments on multiple public datasets show that Prompt-R1 significantly outperforms baseline models across tasks. Our code is publicly available at https://github.com/QwenQKing/Prompt-R1.
>
---
#### [new 005] Cognitive Alignment in Personality Reasoning: Leveraging Prototype Theory for MBTI Inference
- **分类: cs.CL; cs.AI**

- **简介: 论文提出ProtoMBTI，将原型理论引入MBTI人格推断任务，解决传统硬标签分类忽视心理连续性的问题。通过LLM增强数据、LoRA微调编码器、原型检索与动态修正机制，提升准确率与可解释性。**

- **链接: [http://arxiv.org/pdf/2511.00115v1](http://arxiv.org/pdf/2511.00115v1)**

> **作者:** Haoyuan Li; Yuanbo Tong; Yuchen Li; Zirui Wang; Chunhou Liu; Jiamou Liu
>
> **摘要:** Personality recognition from text is typically cast as hard-label classification, which obscures the graded, prototype-like nature of human personality judgments. We present ProtoMBTI, a cognitively aligned framework for MBTI inference that operationalizes prototype theory within an LLM-based pipeline. First, we construct a balanced, quality-controlled corpus via LLM-guided multi-dimensional augmentation (semantic, linguistic, sentiment). Next, we LoRA-fine-tune a lightweight (<=2B) encoder to learn discriminative embeddings and to standardize a bank of personality prototypes. At inference, we retrieve top-k prototypes for a query post and perform a retrieve--reuse--revise--retain cycle: the model aggregates prototype evidence via prompt-based voting, revises when inconsistencies arise, and, upon correct prediction, retains the sample to continually enrich the prototype library. Across Kaggle and Pandora benchmarks, ProtoMBTI improves over baselines on both the four MBTI dichotomies and the full 16-type task, and exhibits robust cross-dataset generalization. Our results indicate that aligning the inference process with psychological prototype reasoning yields gains in accuracy, interpretability, and transfer for text-based personality modeling.
>
---
#### [new 006] Self-Harmony: Learning to Harmonize Self-Supervision and Self-Play in Test-Time Reinforcement Learning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出Self-Harmony，用于无监督测试时强化学习，解决传统方法因多数投票陷入虚假答案的问题。通过模型自产答案与问题改写，用调和均值聚合稳定答案，无需人工标注，显著提升准确率与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2511.01191v1](http://arxiv.org/pdf/2511.01191v1)**

> **作者:** Ru Wang; Wei Huang; Qi Cao; Yusuke Iwasawa; Yutaka Matsuo; Jiaxian Guo
>
> **摘要:** Test-time reinforcement learning (TTRL) offers a label-free paradigm for adapting models using only synthetic signals at inference, but its success hinges on constructing reliable learning signals. Standard approaches such as majority voting often collapse to spurious yet popular answers. We introduce Self-Harmony, a framework built on a simple intuition: the correct answer should remain stable across both an original question and its paraphrase. Self-Harmony operationalizes this by employing a single model in two complementary roles: a Solver to produce answers and a Reframer to rephrase the input. Based on this, we further propose a pseudo-label method: instead of majority voting, it aggregates answer frequencies across these original and reframed views using the harmonic mean. This is a process that naturally selects for solutions stable under reframing, thereby avoiding the common trap of favoring view-dependent, spurious answers. Crucially, this requires no human supervision or auxiliary models. Across diverse reasoning benchmarks, Self-Harmony achieves state-of-the-art results at the label-free test-time setting, ranking first in 28 of 30 settings across multiple methods. Beyond accuracy, it demonstrates unprecedented robustness, with zero training failures in all experiments, underscoring its stability and reliability.
>
---
#### [new 007] OceanAI: A Conversational Platform for Accurate, Transparent, Near-Real-Time Oceanographic Insights
- **分类: cs.CL; cs.AI; cs.CE; cs.LG; physics.ao-ph**

- **简介: OceanAI构建了可验证的海洋学对话平台，解决通用AI幻觉问题，通过实时接入NOAA数据，生成有源引用的自然语言响应与可视化，提升科学可信度与可复现性。**

- **链接: [http://arxiv.org/pdf/2511.01019v1](http://arxiv.org/pdf/2511.01019v1)**

> **作者:** Bowen Chen; Jayesh Gajbhar; Gregory Dusek; Rob Redmon; Patrick Hogan; Paul Liu; DelWayne Bohnenstiehl; Dongkuan; Xu; Ruoying He
>
> **备注:** A related presentation will be given at the AGU(American Geophysical Union) and AMS(American Meteorological Society) Annual Meetings
>
> **摘要:** Artificial intelligence is transforming the sciences, yet general conversational AI systems often generate unverified "hallucinations" undermining scientific rigor. We present OceanAI, a conversational platform that integrates the natural-language fluency of open-source large language models (LLMs) with real-time, parameterized access to authoritative oceanographic data streams hosted by the National Oceanic and Atmospheric Administration (NOAA). Each query such as "What was Boston Harbor's highest water level in 2024?" triggers real-time API calls that identify, parse, and synthesize relevant datasets into reproducible natural-language responses and data visualizations. In a blind comparison with three widely used AI chat-interface products, only OceanAI produced NOAA-sourced values with original data references; others either declined to answer or provided unsupported results. Designed for extensibility, OceanAI connects to multiple NOAA data products and variables, supporting applications in marine hazard forecasting, ecosystem assessment, and water-quality monitoring. By grounding outputs and verifiable observations, OceanAI advances transparency, reproducibility, and trust, offering a scalable framework for AI-enabled decision support within the oceans. A public demonstration is available at https://oceanai.ai4ocean.xyz.
>
---
#### [new 008] Safer in Translation? Presupposition Robustness in Indic Languages
- **分类: cs.CL**

- **简介: 该论文构建了首个印地语等五种印度语言的癌症谬误基准Cancer-Myth-Indic，评估大语言模型在跨语言场景下对隐含预设的鲁棒性，填补了非英语医疗LLM评估的空白。**

- **链接: [http://arxiv.org/pdf/2511.01360v1](http://arxiv.org/pdf/2511.01360v1)**

> **作者:** Aadi Palnitkar; Arjun Suresh; Rishi Rajesh; Puneet Puli
>
> **备注:** This is a submission to LREC 2026 (Language Resources and Evaluation Conference 2026). Corresponding author: aadipalnitkar96@gmail.com
>
> **摘要:** Increasingly, more and more people are turning to large language models (LLMs) for healthcare advice and consultation, making it important to gauge the efficacy and accuracy of the responses of LLMs to such queries. While there are pre-existing medical benchmarks literature which seeks to accomplish this very task, these benchmarks are almost universally in English, which has led to a notable gap in existing literature pertaining to multilingual LLM evaluation. Within this work, we seek to aid in addressing this gap with Cancer-Myth-Indic, an Indic language benchmark built by translating a 500-item subset of Cancer-Myth, sampled evenly across its original categories, into five under-served but widely used languages from the subcontinent (500 per language; 2,500 translated items total). Native-speaker translators followed a style guide for preserving implicit presuppositions in translation; items feature false presuppositions relating to cancer. We evaluate several popular LLMs under this presupposition stress.
>
---
#### [new 009] MicroRemed: Benchmarking LLMs in Microservices Remediation
- **分类: cs.CL; cs.SE; 68T50; I.2.7**

- **简介: 论文提出MicroRemed基准，评估LLM在端到端微服务修复中的能力，解决现有方法依赖人工提示的问题；并设计ThinkRemed多智能体框架，通过迭代推理提升自动修复性能。**

- **链接: [http://arxiv.org/pdf/2511.01166v1](http://arxiv.org/pdf/2511.01166v1)**

> **作者:** Lingzhe Zhang; Yunpeng Zhai; Tong Jia; Chiming Duan; Minghua He; Leyi Pan; Zhaoyang Liu; Bolin Ding; Ying Li
>
> **备注:** 24 pages, 13 figures, 5 tables
>
> **摘要:** Large Language Models (LLMs) integrated with agent-based reasoning frameworks have recently shown strong potential for autonomous decision-making and system-level operations. One promising yet underexplored direction is microservice remediation, where the goal is to automatically recover faulty microservice systems. Existing approaches, however, still rely on human-crafted prompts from Site Reliability Engineers (SREs), with LLMs merely converting textual instructions into executable code. To advance research in this area, we introduce MicroRemed, the first benchmark for evaluating LLMs in end-to-end microservice remediation, where models must directly generate executable Ansible playbooks from diagnosis reports to restore system functionality. We further propose ThinkRemed, a multi-agent framework that emulates the reflective and perceptive reasoning of SREs. Experimental results show that MicroRemed presents substantial challenges to current LLMs, while ThinkRemed improves end-to-end remediation performance through iterative reasoning and system reflection. The benchmark is available at https://github.com/LLM4AIOps/MicroRemed.
>
---
#### [new 010] BIRD: Bronze Inscription Restoration and Dating
- **分类: cs.CL; I.2.7**

- **简介: BIRD提出一种面向青铜器铭文的修复与断代方法，构建标准化数据集，设计字符感知的掩码语言模型，结合Glyph Net关联异体字，提升文本修复与年代预测准确率。**

- **链接: [http://arxiv.org/pdf/2511.01589v1](http://arxiv.org/pdf/2511.01589v1)**

> **作者:** Wenjie Hua; Hoang H. Nguyen; Gangyan Ge
>
> **备注:** Accepted at EMNLP 2025 (Main Conference)
>
> **摘要:** Bronze inscriptions from early China are fragmentary and difficult to date. We introduce BIRD(Bronze Inscription Restoration and Dating), a fully encoded dataset grounded in standard scholarly transcriptions and chronological labels. We further propose an allograph-aware masked language modeling framework that integrates domain- and task-adaptive pretraining with a Glyph Net (GN), which links graphemes and allographs. Experiments show that GN improves restoration, while glyph-biased sampling yields gains in dating.
>
---
#### [new 011] Reversal Invariance in Autoregressive Language Models
- **分类: cs.CL**

- **简介: 该论文揭示自回归语言模型在正向与反向文本上具有损失对称性（反转不变性），指出此特性掩盖语言的时间方向性，构成预训练目标的局限，并呼吁设计能显式建模语言时间箭头的新损失函数与架构。**

- **链接: [http://arxiv.org/pdf/2511.00341v1](http://arxiv.org/pdf/2511.00341v1)**

> **作者:** Mihir Sahasrabudhe
>
> **备注:** 7 pages, theoretical note
>
> **摘要:** We formalize a structural property of the causal (autoregressive) language modeling (CLM) objective: reversal invariance. Formally, the next-token prediction loss assigns identical likelihood to a corpus and its reversal, implying that standard CLM pretraining is direction-blind. This symmetry explains why models trained on reversed text can achieve comparable performance to those trained on forward text, despite the inherently time-asymmetric nature of human language and reasoning. We argue that this invariance represents a limitation of current pretraining objectives rather than a benign artifact. If natural language encodes directional dependencies - phonological, morphological, or causal - a symmetric objective may fail to capture them. We therefore propose viewing pretraining through the lens of temporal asymmetry, motivating future work on loss functions and architectures that explicitly model the arrow of language while retaining standard language modeling capacity.
>
---
#### [new 012] Advancing Machine-Generated Text Detection from an Easy to Hard Supervision Perspective
- **分类: cs.CL**

- **简介: 该论文面向机器生成文本检测任务，针对标签不精确问题，提出易到难监督框架：用弱监督的长文本检测器作为易监督者，引导强检测器逼近真实标签，提升检测鲁棒性。**

- **链接: [http://arxiv.org/pdf/2511.00988v1](http://arxiv.org/pdf/2511.00988v1)**

> **作者:** Chenwang Wu; Yiu-ming Cheung; Bo Han; Defu Lian
>
> **摘要:** Existing machine-generated text (MGT) detection methods implicitly assume labels as the "golden standard". However, we reveal boundary ambiguity in MGT detection, implying that traditional training paradigms are inexact. Moreover, limitations of human cognition and the superintelligence of detectors make inexact learning widespread and inevitable. To this end, we propose an easy-to-hard enhancement framework to provide reliable supervision under such inexact conditions. Distinct from knowledge distillation, our framework employs an easy supervisor targeting relatively simple longer-text detection tasks (despite weaker capabilities), to enhance the more challenging target detector. Firstly, longer texts targeted by supervisors theoretically alleviate the impact of inexact labels, laying the foundation for reliable supervision. Secondly, by structurally incorporating the detector into the supervisor, we theoretically model the supervisor as a lower performance bound for the detector. Thus, optimizing the supervisor indirectly optimizes the detector, ultimately approximating the underlying "golden" labels. Extensive experiments across diverse practical scenarios, including cross-LLM, cross-domain, mixed text, and paraphrase attacks, demonstrate the framework's significant detection effectiveness. The code is available at: https://github.com/tmlr-group/Easy2Hard.
>
---
#### [new 013] Accumulating Context Changes the Beliefs of Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型在持续交互与文本阅读中信念的隐性漂移问题，发现模型信念会因上下文累积而显著改变，影响行为一致性。属于AI对齐与安全任务，揭示了长期使用中模型可信度下降的风险。**

- **链接: [http://arxiv.org/pdf/2511.01805v1](http://arxiv.org/pdf/2511.01805v1)**

> **作者:** Jiayi Geng; Howard Chen; Ryan Liu; Manoel Horta Ribeiro; Robb Willer; Graham Neubig; Thomas L. Griffiths
>
> **摘要:** Language model (LM) assistants are increasingly used in applications such as brainstorming and research. Improvements in memory and context size have allowed these models to become more autonomous, which has also resulted in more text accumulation in their context windows without explicit user intervention. This comes with a latent risk: the belief profiles of models -- their understanding of the world as manifested in their responses or actions -- may silently change as context accumulates. This can lead to subtly inconsistent user experiences, or shifts in behavior that deviate from the original alignment of the models. In this paper, we explore how accumulating context by engaging in interactions and processing text -- talking and reading -- can change the beliefs of language models, as manifested in their responses and behaviors.Our results reveal that models' belief profiles are highly malleable: GPT-5 exhibits a 54.7% shift in its stated beliefs after 10 rounds of discussion about moral dilemmas and queries about safety, while Grok 4 shows a 27.2% shift on political issues after reading texts from the opposing position. We also examine models' behavioral changes by designing tasks that require tool use, where each tool selection corresponds to an implicit belief. We find that these changes align with stated belief shifts, suggesting that belief shifts will be reflected in actual behavior in agentic systems. Our analysis exposes the hidden risk of belief shift as models undergo extended sessions of talking or reading, rendering their opinions and actions unreliable.
>
---
#### [new 014] LingGym: How Far Are LLMs from Thinking Like Field Linguists?
- **分类: cs.CL**

- **简介: 论文提出LingGym基准，评估LLMs在无训练数据的低资源语言中基于IGT和语法描述进行元语言推理的能力，通过Word-Gloss Inference任务检验其泛化能力，揭示LLMs在语言学分析中的潜力与局限。**

- **链接: [http://arxiv.org/pdf/2511.00343v1](http://arxiv.org/pdf/2511.00343v1)**

> **作者:** Changbing Yang; Franklin Ma; Freda Shi; Jian Zhu
>
> **备注:** EMNLP 2025 Main
>
> **摘要:** This paper introduces LingGym, a new benchmark that evaluates LLMs' capacity for meta-linguistic reasoning using Interlinear Glossed Text (IGT) and grammatical descriptions extracted from 18 typologically diverse reference grammars. Unlike previous work that focuses on specific downstream tasks, we assess whether LLMs can generalize linguistic inference across low-resource languages and structures not seen during training. We present a controlled evaluation task: Word-Gloss Inference, in which the model must infer a missing word and gloss from context using varying levels of linguistic information (e.g., glosses, grammatical explanations, translations). Our results show that incorporating structured linguistic cues leads to consistent improvements in reasoning performance across all models. This work highlights both the promise and current limitations of using LLMs for typologically informed linguistic analysis and low-resource language documentation.
>
---
#### [new 015] MedRECT: A Medical Reasoning Benchmark for Error Correction in Clinical Texts
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文提出MedRECT，首个跨语言（英/日）医疗文本错误修正基准，涵盖检测、定位与修正三任务，评估9个LLM并发现推理模型表现更优，微调后超越人类专家，推动多语言安全医疗AI发展。**

- **链接: [http://arxiv.org/pdf/2511.00421v1](http://arxiv.org/pdf/2511.00421v1)**

> **作者:** Naoto Iwase; Hiroki Okuyama; Junichiro Iwasawa
>
> **摘要:** Large language models (LLMs) show increasing promise in medical applications, but their ability to detect and correct errors in clinical texts -- a prerequisite for safe deployment -- remains under-evaluated, particularly beyond English. We introduce MedRECT, a cross-lingual benchmark (Japanese/English) that formulates medical error handling as three subtasks: error detection, error localization (sentence extraction), and error correction. MedRECT is built with a scalable, automated pipeline from the Japanese Medical Licensing Examinations (JMLE) and a curated English counterpart, yielding MedRECT-ja (663 texts) and MedRECT-en (458 texts) with comparable error/no-error balance. We evaluate 9 contemporary LLMs spanning proprietary, open-weight, and reasoning families. Key findings: (i) reasoning models substantially outperform standard architectures, with up to 13.5% relative improvement in error detection and 51.0% in sentence extraction; (ii) cross-lingual evaluation reveals 5-10% performance gaps from English to Japanese, with smaller disparities for reasoning models; (iii) targeted LoRA fine-tuning yields asymmetric improvements in error correction performance (Japanese: +0.078, English: +0.168) while preserving reasoning capabilities; and (iv) our fine-tuned model exceeds human expert performance on structured medical error correction tasks. To our knowledge, MedRECT is the first comprehensive cross-lingual benchmark for medical error correction, providing a reproducible framework and resources for developing safer medical LLMs across languages.
>
---
#### [new 016] AraFinNews: Arabic Financial Summarisation with Domain-Adapted LLMs
- **分类: cs.CL**

- **简介: 该论文面向阿拉伯语金融文本摘要任务，解决领域适配不足导致的 factual 与数值错误问题，构建了最大公开阿拉伯金融新闻数据集 AraFinNews，并验证了领域适配模型 FinAraT5 在事实准确性和叙事流畅性上的显著优势。**

- **链接: [http://arxiv.org/pdf/2511.01265v1](http://arxiv.org/pdf/2511.01265v1)**

> **作者:** Mo El-Haj; Paul Rayson
>
> **备注:** 10 pages
>
> **摘要:** This paper investigates the impact of domain specificity on abstractive summarisation of Arabic financial texts using large language models (LLMs). We introduce AraFinNews, the largest publicly available Arabic financial news dataset to date, comprising 212,500 article--headline pairs spanning nearly a decade of reporting from October 2015 to July 2025. Designed as the Arabic equivalent of major English summarisation corpora such as CNN/DailyMail, AraFinNews provides a robust benchmark for evaluating domain-specific language understanding and generation in financial contexts. Using this resource, we evaluate transformer-based models -- including mT5, AraT5, and the domain-adapted FinAraT5 -- to examine how financial-domain pretraining influences factual accuracy, numerical reliability, and stylistic alignment with professional reporting. Experimental results show that domain-adapted models generate more faithful and coherent summaries, particularly in handling quantitative and entity-centric information. The findings highlight the importance of domain-specific adaptation for improving factual consistency and narrative fluency in Arabic financial summarisation. The dataset is freely available for non-commercial research at https://github.com/ArabicNLP-UK/AraFinNews.
>
---
#### [new 017] Modeling the Construction of a Literary Archetype: The Case of the Detective Figure in French Literature
- **分类: cs.CL**

- **简介: 该论文属于文学计算分析任务，旨在揭示法国侦探形象的演化规律。通过字符嵌入与监督模型，分析1866–2017年文本，证明侦探从配角演变为核心“推理机器”，战后因硬汉传统而更具道德复杂性。**

- **链接: [http://arxiv.org/pdf/2511.00627v1](http://arxiv.org/pdf/2511.00627v1)**

> **作者:** Jean Barré; Olga Seminck; Antoine Bourgois; Thierry Poibeau
>
> **备注:** 19 pages, 2 tables, 5 figures Conference Computational Humanities Research 2025
>
> **摘要:** This research explores the evolution of the detective archetype in French detective fiction through computational analysis. Using quantitative methods and character-level embeddings, we show that a supervised model is able to capture the unity of the detective archetype across 150 years of literature, from M. Lecoq (1866) to Commissaire Adamsberg (2017). Building on this finding, the study demonstrates how the detective figure evolves from a secondary narrative role to become the central character and the "reasoning machine" of the classical detective story. In the aftermath of the Second World War, with the importation of the hardboiled tradition into France, the archetype becomes more complex, navigating the genre's turn toward social violence and moral ambiguity.
>
---
#### [new 018] Thinking with DistilQwen: A Tale of Four Distilled Reasoning and Reward Model Series
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型压缩与推理优化任务，旨在提升小规模语言模型的推理效率与准确性。工作包括构建四类DistilQwen系列模型：高精度慢思考模型、自适应思考模型及蒸馏奖励模型，并在阿里云PAI平台实现工业级部署。**

- **链接: [http://arxiv.org/pdf/2511.01354v1](http://arxiv.org/pdf/2511.01354v1)**

> **作者:** Wenrui Cai; Chengyu Wang; Junbing Yan; Jun Huang; Xiangzhong Fang
>
> **备注:** emnlp 2025 industry track
>
> **摘要:** Recently, the demand for small and efficient reasoning models to support real-world applications has driven the development of knowledge distillation techniques that balance reasoning performance and inference speed. In this paper, we further extend the DistilQwen model family, initialized from the Qwen models, by introducing four model series specifically designed to meet industrial requirements. The distilled model collection comprises: (1) slow-thinking models, optimized for reasoning tasks that require high accuracy; (2) two series of adaptive-thinking models, which dynamically adjust reasoning strategies based on input tasks to maximize efficiency across diverse scenarios; and (3) distilled reward models, which enable further reinforcement learning of reasoning models using distilled knowledge. Comprehensive evaluations across multiple benchmarks demonstrate both high inference efficiency and strong reasoning performance for these models, as well as the practical utility of distilled reward models. We further show that these models support industry practitioners by providing scalable training and inference functionalities on the Alibaba Cloud PAI (Platform for Artificial Intelligence) platform.
>
---
#### [new 019] Towards Robust Mathematical Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出IMO-Bench基准，解决现有数学推理评估过于简单、忽略证明能力的问题。构建了答案与证明双维度评测体系，推动模型在IMO水平的鲁棒推理，并实现Gemini模型历史性金牌表现。**

- **链接: [http://arxiv.org/pdf/2511.01846v1](http://arxiv.org/pdf/2511.01846v1)**

> **作者:** Thang Luong; Dawsen Hwang; Hoang H. Nguyen; Golnaz Ghiasi; Yuri Chervonyi; Insuk Seo; Junsu Kim; Garrett Bingham; Jonathan Lee; Swaroop Mishra; Alex Zhai; Clara Huiyi Hu; Henryk Michalewski; Jimin Kim; Jeonghyun Ahn; Junhwi Bae; Xingyou Song; Trieu H. Trinh; Quoc V. Le; Junehyuk Jung
>
> **备注:** EMNLP 2025 (main conference), https://aclanthology.org/2025.emnlp-main.1794/
>
> **摘要:** Finding the right north-star metrics is highly critical for advancing the mathematical reasoning capabilities of foundation models, especially given that existing evaluations are either too easy or only focus on getting correct short answers. To address these issues, we present IMO-Bench, a suite of advanced reasoning benchmarks, vetted by a panel of top specialists and that specifically targets the level of the International Mathematical Olympiad (IMO), the most prestigious venue for young mathematicians. IMO-AnswerBench first tests models on 400 diverse Olympiad problems with verifiable short answers. IMO-Proof Bench is the next-level evaluation for proof-writing capabilities, which includes both basic and advanced IMO level problems as well as detailed grading guidelines to facilitate automatic grading. These benchmarks played a crucial role in our historic achievement of the gold-level performance at IMO 2025 with Gemini Deep Think (Luong and Lockhart, 2025). Our model achieved 80.0% on IMO-AnswerBench and 65.7% on the advanced IMO-Proof Bench, surpassing the best non-Gemini models by large margins of 6.9% and 42.4% respectively. We also showed that autograders built with Gemini reasoning correlate well with human evaluations and construct IMO-GradingBench, with 1000 human gradings on proofs, to enable further progress in automatic evaluation of long-form answers. We hope that IMO-Bench will help the community towards advancing robust mathematical reasoning and release it at https://imobench.github.io/.
>
---
#### [new 020] POSESTITCH-SLT: Linguistically Inspired Pose-Stitching for End-to-End Sign Language Translation
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **简介: 该论文面向手语翻译任务，解决低资源下数据稀缺问题，提出POSESTITCH-SLT：一种受语言模板启发的姿态拼接预训练方法，通过合成句子对提升Transformer模型性能，在How2Sign和iSign数据集上显著超越现有方法。**

- **链接: [http://arxiv.org/pdf/2511.00270v1](http://arxiv.org/pdf/2511.00270v1)**

> **作者:** Abhinav Joshi; Vaibhav Sharma; Sanjeet Singh; Ashutosh Modi
>
> **备注:** Accepted at EMNLP 2025 (Main)
>
> **摘要:** Sign language translation remains a challenging task due to the scarcity of large-scale, sentence-aligned datasets. Prior arts have focused on various feature extraction and architectural changes to support neural machine translation for sign languages. We propose POSESTITCH-SLT, a novel pre-training scheme that is inspired by linguistic-templates-based sentence generation technique. With translation comparison on two sign language datasets, How2Sign and iSign, we show that a simple transformer-based encoder-decoder architecture outperforms the prior art when considering template-generated sentence pairs in training. We achieve BLEU-4 score improvements from 1.97 to 4.56 on How2Sign and from 0.55 to 3.43 on iSign, surpassing prior state-of-the-art methods for pose-based gloss-free translation. The results demonstrate the effectiveness of template-driven synthetic supervision in low-resource sign language settings.
>
---
#### [new 021] The Biased Oracle: Assessing LLMs' Understandability and Empathy in Medical Diagnoses
- **分类: cs.CL**

- **简介: 该论文评估LLMs在医疗诊断沟通中生成可理解且富有同理心解释的能力，发现其虽能适配患者特征，但存在语义复杂与情感偏见问题，需系统校准以保障公平性。**

- **链接: [http://arxiv.org/pdf/2511.00924v1](http://arxiv.org/pdf/2511.00924v1)**

> **作者:** Jianzhou Yao; Shunchang Liu; Guillaume Drui; Rikard Pettersson; Alessandro Blasimme; Sara Kijewski
>
> **备注:** Accepted by NeurIPS 2025 GenAI4Health Workshop
>
> **摘要:** Large language models (LLMs) show promise for supporting clinicians in diagnostic communication by generating explanations and guidance for patients. Yet their ability to produce outputs that are both understandable and empathetic remains uncertain. We evaluate two leading LLMs on medical diagnostic scenarios, assessing understandability using readability metrics as a proxy and empathy through LLM-as-a-Judge ratings compared to human evaluations. The results indicate that LLMs adapt explanations to socio-demographic variables and patient conditions. However, they also generate overly complex content and display biased affective empathy, leading to uneven accessibility and support. These patterns underscore the need for systematic calibration to ensure equitable patient communication. The code and data are released: https://github.com/Jeffateth/Biased_Oracle
>
---
#### [new 022] TSVer: A Benchmark for Fact Verification Against Time-Series Evidence
- **分类: cs.CL**

- **简介: 论文提出TSVer基准，面向时间序列证据的事实验证任务，解决现有数据集缺乏真实、结构化时序证据与合理论证的问题，构建了287个真实主张与400条时序数据的标注集，并评估了主流模型的性能。**

- **链接: [http://arxiv.org/pdf/2511.01101v1](http://arxiv.org/pdf/2511.01101v1)**

> **作者:** Marek Strong; Andreas Vlachos
>
> **备注:** Accepted to EMNLP 2025
>
> **摘要:** Reasoning over temporal and numerical data, such as time series, is a crucial aspect of fact-checking. While many systems have recently been developed to handle this form of evidence, their evaluation remains limited by existing datasets, which often lack structured evidence, provide insufficient justifications for verdicts, or rely on synthetic claims. In this paper, we introduce TSVer, a new benchmark dataset for fact verification focusing on temporal and numerical reasoning with time-series evidence. TSVer contains 287 real-world claims sourced from 38 fact-checking organizations and a curated database of 400 time series covering diverse domains. Each claim is annotated with time frames across all pertinent time series, along with a verdict and justifications reflecting how the evidence is used to reach the verdict. Using an LLM-assisted multi-step annotation process, we improve the quality of our annotations and achieve an inter-annotator agreement of kappa=0.745 on verdicts. We also develop a baseline for verifying claims against time-series evidence and show that even the state-of-the-art reasoning models like Gemini-2.5-Pro are challenged by time series, achieving a 63.37 accuracy score on verdicts and an Ev2R score of 48.63 on verdict justifications.
>
---
#### [new 023] MARS-SQL: A multi-agent reinforcement learning framework for Text-to-SQL
- **分类: cs.CL**

- **简介: MARS-SQL提出一种多智能体强化学习框架，用于复杂自然语言到SQL的转换。通过分解任务并引入交互式RL自修正，结合生成与验证智能体，显著提升SQL生成准确率，SOTA结果达89.75%。**

- **链接: [http://arxiv.org/pdf/2511.01008v1](http://arxiv.org/pdf/2511.01008v1)**

> **作者:** Haolin Yang; Jipeng Zhang; Zhitao He; Yi R. Fung
>
> **摘要:** Translating natural language to SQL remains difficult for complex queries. Such queries often need environmental interaction and self-correction. To address this, we introduce MARS-SQL, a novel multi-agent framework that combines principled task decomposition and interactive reinforcement learning (RL). Our system comprises three specialized agents: a Grounding Agent for schema linking, a Generation Agent for query generation, and a Validation Agent for final selection. The core of our framework is the Generation agent, which is trained via a multi-turn RL policy. Adopting a ReAct-style Think-Act-Observe loop, the agent iteratively generates thoughts, executes SQL actions against a live database, and revises its strategy based on execution feedback, enabling dynamic, stateful reasoning and self-correction. At inference time, we generate multiple interaction trajectories to explore diverse reasoning paths. The Validation agent, then selects the optimal trajectory by modeling verification as a next-token prediction task and choosing the solution with the highest generation probability. This structured workflow pipelines specialized agents. It combines interactive RL for generation with generative modeling for verification. The approach proves highly effective for robust and accurate SQL generation. Experiments show that MARS-SQL achieves state-of-the-art Execution Accuracy of 77.84% on the BIRD dev set and 89.75% on the Spider test set. Our code is available at https://github.com/YangHaolin0526/MARS-SQL.
>
---
#### [new 024] VayuChat: An LLM-Powered Conversational Interface for Air Quality Data Analytics
- **分类: cs.CL**

- **简介: VayuChat是一款基于大语言模型的对话系统，旨在解决空气污染数据难以被非专家理解的问题，通过自然语言交互提供空气质量、气象与政策的分析，并生成可执行代码与可视化结果，赋能决策者与公众。**

- **链接: [http://arxiv.org/pdf/2511.01046v1](http://arxiv.org/pdf/2511.01046v1)**

> **作者:** Vedant Acharya; Abhay Pisharodi; Rishabh Mondal; Mohammad Rafiuddin; Nipun Batra
>
> **备注:** 4 Pages, 4 Figures
>
> **摘要:** Air pollution causes about 1.6 million premature deaths each year in India, yet decision makers struggle to turn dispersed data into decisions. Existing tools require expertise and provide static dashboards, leaving key policy questions unresolved. We present VayuChat, a conversational system that answers natural language questions on air quality, meteorology, and policy programs, and responds with both executable Python code and interactive visualizations. VayuChat integrates data from Central Pollution Control Board (CPCB) monitoring stations, state-level demographics, and National Clean Air Programme (NCAP) funding records into a unified interface powered by large language models. Our live demonstration will show how users can perform complex environmental analytics through simple conversations, making data science accessible to policymakers, researchers, and citizens. The platform is publicly deployed at https://huggingface.co/spaces/SustainabilityLabIITGN/ VayuChat. For further information check out video uploaded on https://www.youtube.com/watch?v=d6rklL05cs4.
>
---
#### [new 025] Optimizing Native Sparse Attention with Latent Attention and Local Global Alternating Strategies
- **分类: cs.CL**

- **简介: 该论文针对长文本建模中的稀疏注意力效率与效果问题，提出交替使用局部与全局注意力策略，并引入Latent Attention降低KV缓存50%，在保持性能同时提升长上下文理解与常识推理能力。**

- **链接: [http://arxiv.org/pdf/2511.00819v1](http://arxiv.org/pdf/2511.00819v1)**

> **作者:** Yuxuan Hu; Jianchao Tan; Jiaqi Zhang; Wen Zan; Pingwei Sun; Yifan Lu; Yerui Sun; Yuchen Xie; Xunliang Cai; Jing Zhang
>
> **摘要:** In this work, we conduct a systematic analysis of Native Sparse Attention (NSA) and propose targeted improvements that enhance long-context modeling. A key insight is that alternating between local (sliding-window) and global (compression, selective) attention across layers, rather than using fixed patterns, enables more effective propagation of long-range dependencies and substantially boosts performance on long-sequence tasks. Meanwhile, we further refine NSA's branches with Latent Attention that the sliding-window branch is enhanced with Multi-head Latent Attention (MLA) while compression and selective branches adopt Group-head Latent Attention (GLA). These changes reduce KV-cache memory by 50\% versus NSA while improving the model's common-sense reasoning and long-text understanding capabilities. Experiments on models from 340M to 1.3B parameters (trained on 15B and 100B tokens) show our method matches or exceeds full attention and native sparse attention in both common-sense reasoning and long-context understanding tasks.
>
---
#### [new 026] Difficulty-Controllable Cloze Question Distractor Generation
- **分类: cs.CL**

- **简介: 该论文面向填空题干扰项生成任务，解决现有方法难以控制难度的问题。通过构建难度标注数据集并设计多任务学习框架，实现可调控难度的高质量干扰项生成，显著超越GPT-4o的人类难度对齐性能。**

- **链接: [http://arxiv.org/pdf/2511.01526v1](http://arxiv.org/pdf/2511.01526v1)**

> **作者:** Seokhoon Kang; Yejin Jeon; Seonjeong Hwang; Gary Geunbae Lee
>
> **摘要:** Multiple-choice cloze questions are commonly used to assess linguistic proficiency and comprehension. However, generating high-quality distractors remains challenging, as existing methods often lack adaptability and control over difficulty levels, and the absence of difficulty-annotated datasets further hinders progress. To address these issues, we propose a novel framework for generating distractors with controllable difficulty by leveraging both data augmentation and a multitask learning strategy. First, to create a high-quality, difficulty-annotated dataset, we introduce a two-way distractor generation process in order to produce diverse and plausible distractors. These candidates are subsequently refined through filtering and then categorized by difficulty using an ensemble QA system. Second, this newly created dataset is leveraged to train a difficulty-controllable generation model via multitask learning. The framework includes carefully designed auxiliary tasks that enhance the model's semantic understanding of distractors and its ability to estimate their difficulty. Experimental results demonstrate that our method generates high-quality distractors across difficulty levels and substantially outperforms GPT-4o in aligning distractor difficulty with human perception.
>
---
#### [new 027] FirstAidQA: A Synthetic Dataset for First Aid and Emergency Response in Low-Connectivity Settings
- **分类: cs.CL**

- **简介: 论文提出FirstAidQA，一个面向低网络环境的急救问答合成数据集，解决应急场景中轻量模型缺乏高质量数据的问题，通过ChatGPT-4o-mini生成并人工验证5500组QA对，支持小模型离线微调。**

- **链接: [http://arxiv.org/pdf/2511.01289v1](http://arxiv.org/pdf/2511.01289v1)**

> **作者:** Saiyma Sittul Muna; Rezwan Islam Salvi; Mushfiqur Rahman Mushfique; Ajwad Abrar
>
> **备注:** Accepted at the 5th Muslims in Machine Learning (MusIML) Workshop, co-located with NeurIPS 2025
>
> **摘要:** In emergency situations, every second counts. The deployment of Large Language Models (LLMs) in time-sensitive, low or zero-connectivity environments remains limited. Current models are computationally intensive and unsuitable for low-tier devices often used by first responders or civilians. A major barrier to developing lightweight, domain-specific solutions is the lack of high-quality datasets tailored to first aid and emergency response. To address this gap, we introduce FirstAidQA, a synthetic dataset containing 5,500 high-quality question answer pairs that encompass a wide range of first aid and emergency response scenarios. The dataset was generated using a Large Language Model, ChatGPT-4o-mini, with prompt-based in-context learning, using texts from the Vital First Aid Book (2019). We applied preprocessing steps such as text cleaning, contextual chunking, and filtering, followed by human validation to ensure accuracy, safety, and practical relevance of the QA pairs. FirstAidQA is designed to support instruction-tuning and fine-tuning of LLMs and Small Language Models (SLMs), enabling faster, more reliable, and offline-capable systems for emergency settings. We publicly release the dataset to advance research on safety-critical and resource-constrained AI applications in first aid and emergency response. The dataset is available on Hugging Face at https://huggingface.co/datasets/i-am-mushfiq/FirstAidQA.
>
---
#### [new 028] Evaluating Cultural Knowledge Processing in Large Language Models: A Cognitive Benchmarking Framework Integrating Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文提出一种融合Bloom分类法与RAG的认知评估框架，用于衡量大语言模型处理文化知识（如台湾客家文化）的能力，解决LLM在文化语境中推理与生成不准确的问题。**

- **链接: [http://arxiv.org/pdf/2511.01649v1](http://arxiv.org/pdf/2511.01649v1)**

> **作者:** Hung-Shin Lee; Chen-Chi Chang; Ching-Yuan Chen; Yun-Hsiang Hsu
>
> **备注:** This paper has been accepted by The Electronic Library, and the full article is now available on Emerald Insight
>
> **摘要:** This study proposes a cognitive benchmarking framework to evaluate how large language models (LLMs) process and apply culturally specific knowledge. The framework integrates Bloom's Taxonomy with Retrieval-Augmented Generation (RAG) to assess model performance across six hierarchical cognitive domains: Remembering, Understanding, Applying, Analyzing, Evaluating, and Creating. Using a curated Taiwanese Hakka digital cultural archive as the primary testbed, the evaluation measures LLM-generated responses' semantic accuracy and cultural relevance.
>
---
#### [new 029] A Graph-based RAG for Energy Efficiency Question Answering
- **分类: cs.CL; cs.AI; cs.IR; I.2.7; I.2.4; I.2.1; I.2.6**

- **简介: 该论文提出一种基于图的RAG架构，用于能源效率领域的问答任务，通过自动构建知识图谱并结合大语言模型，实现多语言准确问答，验证了75.2%的准确率与良好的跨语言性能。**

- **链接: [http://arxiv.org/pdf/2511.01643v1](http://arxiv.org/pdf/2511.01643v1)**

> **作者:** Riccardo Campi; Nicolò Oreste Pinciroli Vago; Mathyas Giudici; Pablo Barrachina Rodriguez-Guisado; Marco Brambilla; Piero Fraternali
>
> **摘要:** In this work, we investigate the use of Large Language Models (LLMs) within a graph-based Retrieval Augmented Generation (RAG) architecture for Energy Efficiency (EE) Question Answering. First, the system automatically extracts a Knowledge Graph (KG) from guidance and regulatory documents in the energy field. Then, the generated graph is navigated and reasoned upon to provide users with accurate answers in multiple languages. We implement a human-based validation using the RAGAs framework properties, a validation dataset comprising 101 question-answer pairs, and domain experts. Results confirm the potential of this architecture and identify its strengths and weaknesses. Validation results show how the system correctly answers in about three out of four of the cases (75.2 +- 2.7%), with higher results on questions related to more general EE answers (up to 81.0 +- 4.1%), and featuring promising multilingual abilities (4.4% accuracy loss due to translation).
>
---
#### [new 030] "Give a Positive Review Only": An Early Investigation Into In-Paper Prompt Injection Attacks and Defenses for AI Reviewers
- **分类: cs.CL; cs.CR**

- **简介: 该论文研究AI审稿中的提示注入攻击，提出静态与迭代两类攻击方法，可诱导AI给出过高评分，并尝试检测防御，揭示了AI审稿系统的安全风险，呼吁加强防护。**

- **链接: [http://arxiv.org/pdf/2511.01287v1](http://arxiv.org/pdf/2511.01287v1)**

> **作者:** Qin Zhou; Zhexin Zhang; Zhi Li; Limin Sun
>
> **摘要:** With the rapid advancement of AI models, their deployment across diverse tasks has become increasingly widespread. A notable emerging application is leveraging AI models to assist in reviewing scientific papers. However, recent reports have revealed that some papers contain hidden, injected prompts designed to manipulate AI reviewers into providing overly favorable evaluations. In this work, we present an early systematic investigation into this emerging threat. We propose two classes of attacks: (1) static attack, which employs a fixed injection prompt, and (2) iterative attack, which optimizes the injection prompt against a simulated reviewer model to maximize its effectiveness. Both attacks achieve striking performance, frequently inducing full evaluation scores when targeting frontier AI reviewers. Furthermore, we show that these attacks are robust across various settings. To counter this threat, we explore a simple detection-based defense. While it substantially reduces the attack success rate, we demonstrate that an adaptive attacker can partially circumvent this defense. Our findings underscore the need for greater attention and rigorous safeguards against prompt-injection threats in AI-assisted peer review.
>
---
#### [new 031] The Ouroboros of Benchmarking: Reasoning Evaluation in an Era of Saturation
- **分类: cs.CL**

- **简介: 该论文探讨大模型推理评估中的基准饱和问题，分析OpenAI、Anthropic、Google等模型在多基准上的表现趋势，揭示当前评估体系与真实推理能力的脱节，呼吁建立更可靠的推理评估框架。**

- **链接: [http://arxiv.org/pdf/2511.01365v1](http://arxiv.org/pdf/2511.01365v1)**

> **作者:** İbrahim Ethem Deveci; Duygu Ataman
>
> **备注:** Accepted to NeurIPS 2025 Workshop on LLM Evaluation (https://openreview.net/group?id=NeurIPS.cc/2025/Workshop/LLM_Evaluation)
>
> **摘要:** The rapid rise of Large Language Models (LLMs) and Large Reasoning Models (LRMs) has been accompanied by an equally rapid increase of benchmarks used to assess them. However, due to both improved model competence resulting from scaling and novel training advances as well as likely many of these datasets being included in pre or post training data, results become saturated, driving a continuous need for new and more challenging replacements. In this paper, we discuss whether surpassing a benchmark truly demonstrates reasoning ability or are we simply tracking numbers divorced from the capabilities we claim to measure? We present an investigation focused on three model families, OpenAI, Anthropic, and Google, and how their reasoning capabilities across different benchmarks evolve over the years. We also analyze performance trends over the years across different reasoning tasks and discuss the current situation of benchmarking and remaining challenges. By offering a comprehensive overview of benchmarks and reasoning tasks, our work aims to serve as a first reference to ground future research in reasoning evaluation and model development.
>
---
#### [new 032] Building a Silver-Standard Dataset from NICE Guidelines for Clinical LLMs
- **分类: cs.CL**

- **简介: 该论文构建了一个基于NICE指南的银标准临床数据集，用于评估LLM的指南遵循与临床推理能力，解决缺乏标准化评测基准的问题，通过GPT辅助生成真实病例与问题，并对主流LLM进行基准测试。**

- **链接: [http://arxiv.org/pdf/2511.01053v1](http://arxiv.org/pdf/2511.01053v1)**

> **作者:** Qing Ding; Eric Hua Qing Zhang; Felix Jozsa; Julia Ive
>
> **备注:** Submitted to EFMI Medical Informatics Europe 2026
>
> **摘要:** Large language models (LLMs) are increasingly used in healthcare, yet standardised benchmarks for evaluating guideline-based clinical reasoning are missing. This study introduces a validated dataset derived from publicly available guidelines across multiple diagnoses. The dataset was created with the help of GPT and contains realistic patient scenarios, as well as clinical questions. We benchmark a range of recent popular LLMs to showcase the validity of our dataset. The framework supports systematic evaluation of LLMs' clinical utility and guideline adherence.
>
---
#### [new 033] Reasoning Trajectories for Socratic Debugging of Student Code: From Misconceptions to Contradictions and Updated Beliefs
- **分类: cs.CL; cs.CY; cs.SE**

- **简介: 该论文提出“推理轨迹生成”任务，旨在通过引导学生发现自身编程误解与程序行为的矛盾，促进其自我修正。作者构建了标注数据集，并基于LLM生成推理轨迹与苏格拉底式对话，实验表明模型生成准确率高达91%。**

- **链接: [http://arxiv.org/pdf/2511.00371v1](http://arxiv.org/pdf/2511.00371v1)**

> **作者:** Erfan Al-Hossami; Razvan Bunescu
>
> **备注:** 25 pages, 2 tables, 13 figures
>
> **摘要:** In Socratic debugging, instructors guide students towards identifying and fixing a bug on their own, instead of providing the bug fix directly. Most novice programmer bugs are caused by programming misconceptions, namely false beliefs about a programming concept. In this context, Socratic debugging can be formulated as a guided Reasoning Trajectory (RT) leading to a statement about the program behavior that contradicts the bug-causing misconception. Upon reaching this statement, the ensuing cognitive dissonance leads the student to first identify and then update their false belief. In this paper, we introduce the task of reasoning trajectory generation, together with a dataset of debugging problems manually annotated with RTs. We then describe LLM-based solutions for generating RTs and Socratic conversations that are anchored on them. A large-scale LLM-as-judge evaluation shows that frontier models can generate up to 91% correct reasoning trajectories and 98.7% valid conversation turns.
>
---
#### [new 034] Certain but not Probable? Differentiating Certainty from Probability in LLM Token Outputs for Probabilistic Scenarios
- **分类: cs.CL**

- **简介: 该论文研究大语言模型在概率场景中输出令牌的确定性与理论概率的偏离问题，发现模型虽能准确作答，但其概率分布与理论值不符，揭示了现有不确定性量化方法的局限性。**

- **链接: [http://arxiv.org/pdf/2511.00620v1](http://arxiv.org/pdf/2511.00620v1)**

> **作者:** Autumn Toney-Wails; Ryan Wails
>
> **备注:** To appear at the Second Workshop on Uncertainty-Aware NLP @EMNLP 2025 (UncertaiNLP '25)
>
> **摘要:** Reliable uncertainty quantification (UQ) is essential for ensuring trustworthy downstream use of large language models, especially when they are deployed in decision-support and other knowledge-intensive applications. Model certainty can be estimated from token logits, with derived probability and entropy values offering insight into performance on the prompt task. However, this approach may be inadequate for probabilistic scenarios, where the probabilities of token outputs are expected to align with the theoretical probabilities of the possible outcomes. We investigate the relationship between token certainty and alignment with theoretical probability distributions in well-defined probabilistic scenarios. Using GPT-4.1 and DeepSeek-Chat, we evaluate model responses to ten prompts involving probability (e.g., roll a six-sided die), both with and without explicit probability cues in the prompt (e.g., roll a fair six-sided die). We measure two dimensions: (1) response validity with respect to scenario constraints, and (2) alignment between token-level output probabilities and theoretical probabilities. Our results indicate that, while both models achieve perfect in-domain response accuracy across all prompt scenarios, their token-level probability and entropy values consistently diverge from the corresponding theoretical distributions.
>
---
#### [new 035] PADBen: A Comprehensive Benchmark for Evaluating AI Text Detectors Against Paraphrase Attacks
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出PADBen基准，评估AI文本检测器对抗改写攻击的鲁棒性，揭示其在“作者混淆”场景下严重失效，指出当前方法无法应对语义偏移但模式保留的中间洗白区域，亟需新架构突破。**

- **链接: [http://arxiv.org/pdf/2511.00416v1](http://arxiv.org/pdf/2511.00416v1)**

> **作者:** Yiwei Zha; Rui Min; Shanu Sushmita
>
> **摘要:** While AI-generated text (AIGT) detectors achieve over 90\% accuracy on direct LLM outputs, they fail catastrophically against iteratively-paraphrased content. We investigate why iteratively-paraphrased text -- itself AI-generated -- evades detection systems designed for AIGT identification. Through intrinsic mechanism analysis, we reveal that iterative paraphrasing creates an intermediate laundering region characterized by semantic displacement with preserved generation patterns, which brings up two attack categories: paraphrasing human-authored text (authorship obfuscation) and paraphrasing LLM-generated text (plagiarism evasion). To address these vulnerabilities, we introduce PADBen, the first benchmark systematically evaluating detector robustness against both paraphrase attack scenarios. PADBen comprises a five-type text taxonomy capturing the full trajectory from original content to deeply laundered text, and five progressive detection tasks across sentence-pair and single-sentence challenges. We evaluate 11 state-of-the-art detectors, revealing critical asymmetry: detectors successfully identify the plagiarism evasion problem but fail for the case of authorship obfuscation. Our findings demonstrate that current detection approaches cannot effectively handle the intermediate laundering region, necessitating fundamental advances in detection architectures beyond existing semantic and stylistic discrimination methods. For detailed code implementation, please see https://github.com/JonathanZha47/PadBen-Paraphrase-Attack-Benchmark.
>
---
#### [new 036] Synthetic Eggs in Many Baskets: The Impact of Synthetic Data Diversity on LLM Fine-Tuning
- **分类: cs.CL**

- **简介: 该论文研究合成数据多样性对LLM微调的影响，聚焦分布坍缩、对抗鲁棒性与自偏好偏差。发现多源合成数据可缓解分布坍缩、保持输出质量，且优于单源合成数据，但略逊于人类数据。**

- **链接: [http://arxiv.org/pdf/2511.01490v1](http://arxiv.org/pdf/2511.01490v1)**

> **作者:** Max Schaffelder; Albert Gatt
>
> **摘要:** As synthetic data becomes widely used in language model development, understanding its impact on model behavior is crucial. This paper investigates the impact of the diversity of sources of synthetic data on fine-tuned large language models. We focus on three key dimensions: distribution collapse, adversarial robustness, and self-preference bias. Our findings reveal that fine-tuning models on synthetic data from diverse sources can mitigate distribution collapse, preserving the breadth of the output distribution and the diversity of the output text. Furthermore, while both human and synthetic fine-tuning data can remove safeguards, the latter preserves higher output quality, thus making outputs potentially more usable and dangerous. Finally, fine-tuning reduces self-preference bias, with human data being the most effective, followed by multi-source synthetic data.
>
---
#### [new 037] Assessing LLM Reasoning Steps via Principal Knowledge Grounding
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出一种基于主干知识的评估框架，用于检验LLM推理步骤中知识引用的准确性，解决推理过程缺乏知识 grounding 验证的问题，构建了知识库、评估指标与轻量级评估器，支持推理缺陷诊断与偏好优化。**

- **链接: [http://arxiv.org/pdf/2511.00879v1](http://arxiv.org/pdf/2511.00879v1)**

> **作者:** Hyeon Hwang; Yewon Cho; Chanwoong Yoon; Yein Park; Minju Song; Kyungjae Lee; Gangwoo Kim; Jaewoo Kang
>
> **备注:** Accepted to EMNLP 2025 Findings
>
> **摘要:** Step-by-step reasoning has become a standard approach for large language models (LLMs) to tackle complex tasks. While this paradigm has proven effective, it raises a fundamental question: How can we verify that an LLM's reasoning is accurately grounded in knowledge? To address this question, we introduce a novel evaluation suite that systematically assesses the knowledge grounding of intermediate reasoning. Our framework comprises three key components. (1) Principal Knowledge Collection, a large-scale repository of atomic knowledge essential for reasoning. Based on the collection, we propose (2) knowledge-grounded evaluation metrics designed to measure how well models recall and apply prerequisite knowledge in reasoning. These metrics are computed by our (3) evaluator LLM, a lightweight model optimized for cost-effective and reliable metric computation. Our evaluation suite demonstrates remarkable effectiveness in identifying missing or misapplied knowledge elements, providing crucial insights for uncovering fundamental reasoning deficiencies in LLMs. Beyond evaluation, we demonstrate how these metrics can be integrated into preference optimization, showcasing further applications of knowledge-grounded evaluation.
>
---
#### [new 038] Learning When to Quit in Sales Conversations
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究销售对话中的何时终止决策问题，将其建模为最优停止问题，提出基于语言模型的“终止代理”，通过模仿最优策略自动判断何时放弃无效通话，显著减少无效时间并提升销售效率，揭示人类决策的认知偏差。**

- **链接: [http://arxiv.org/pdf/2511.01181v1](http://arxiv.org/pdf/2511.01181v1)**

> **作者:** Emaad Manzoor; Eva Ascarza; Oded Netzer
>
> **摘要:** Salespeople frequently face the dynamic screening decision of whether to persist in a conversation or abandon it to pursue the next lead. Yet, little is known about how these decisions are made, whether they are efficient, or how to improve them. We study these decisions in the context of high-volume outbound sales where leads are ample, but time is scarce and failure is common. We formalize the dynamic screening decision as an optimal stopping problem and develop a generative language model-based sequential decision agent - a stopping agent - that learns whether and when to quit conversations by imitating a retrospectively-inferred optimal stopping policy. Our approach handles high-dimensional textual states, scales to large language models, and works with both open-source and proprietary language models. When applied to calls from a large European telecommunications firm, our stopping agent reduces the time spent on failed calls by 54% while preserving nearly all sales; reallocating the time saved increases expected sales by up to 37%. Upon examining the linguistic cues that drive salespeople's quitting decisions, we find that they tend to overweight a few salient expressions of consumer disinterest and mispredict call failure risk, suggesting cognitive bounds on their ability to make real-time conversational decisions. Our findings highlight the potential of artificial intelligence algorithms to correct cognitively-bounded human decisions and improve salesforce efficiency.
>
---
#### [new 039] BanglaNirTox: A Large-scale Parallel Corpus for Explainable AI in Bengali Text Detoxification
- **分类: cs.CL; cs.AI**

- **简介: 该论文面向孟加拉语文本去毒化任务，解决低资源语言中毒性内容处理不足的问题，构建了首个大规模平行语料BanglaNirTox，并结合帕累托优化LLM与CoT提示，显著提升去毒效果。**

- **链接: [http://arxiv.org/pdf/2511.01512v1](http://arxiv.org/pdf/2511.01512v1)**

> **作者:** Ayesha Afroza Mohsin; Mashrur Ahsan; Nafisa Maliyat; Shanta Maria; Syed Rifat Raiyan; Hasan Mahmud; Md Kamrul Hasan
>
> **备注:** Under review, 6 pages, 1 figure, 2 tables
>
> **摘要:** Toxic language in Bengali remains prevalent, especially in online environments, with few effective precautions against it. Although text detoxification has seen progress in high-resource languages, Bengali remains underexplored due to limited resources. In this paper, we propose a novel pipeline for Bengali text detoxification that combines Pareto class-optimized large language models (LLMs) and Chain-of-Thought (CoT) prompting to generate detoxified sentences. To support this effort, we construct BanglaNirTox, an artificially generated parallel corpus of 68,041 toxic Bengali sentences with class-wise toxicity labels, reasonings, and detoxified paraphrases, using Pareto-optimized LLMs evaluated on random samples. The resulting BanglaNirTox dataset is used to fine-tune language models to produce better detoxified versions of Bengali sentences. Our findings show that Pareto-optimized LLMs with CoT prompting significantly enhance the quality and consistency of Bengali text detoxification.
>
---
#### [new 040] KV Cache Transform Coding for Compact Storage in LLM Inference
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出KVTC，一种轻量级KV缓存变换编码方法，用于压缩大语言模型推理中的键值缓存，解决GPU内存占用高问题，在不损失精度前提下实现最高40倍压缩，提升缓存复用效率。**

- **链接: [http://arxiv.org/pdf/2511.01815v1](http://arxiv.org/pdf/2511.01815v1)**

> **作者:** Konrad Staniszewski; Adrian Łańcucki
>
> **摘要:** Serving large language models (LLMs) at scale necessitates efficient key-value (KV) cache management. KV caches can be reused across conversation turns via shared-prefix prompts that are common in iterative code editing and chat. However, stale caches consume scarce GPU memory, require offloading, or force recomputation. We present KVTC, a lightweight transform coder that compresses KV caches for compact on-GPU and off-GPU storage. Drawing on classical media compression, KVTC combines PCA-based feature decorrelation, adaptive quantization, and entropy coding. It requires only a brief initial calibration and leaves model parameters unchanged. By exploiting redundancies in KV caches, KVTC achieves up to 20$\times$ compression while maintaining reasoning and long-context accuracy, and 40$\times$ or higher for specific use cases. We test KVTC with Llama 3, Mistral NeMo, and R1-Qwen 2.5 models across benchmarks including AIME25, LiveCodeBench, GSM8K, MMLU, Qasper, RULER, and MATH-500. It consistently outperforms inference-time baselines such as token eviction, quantization, and SVD-based methods, while achieving higher compression ratios. These results support KVTC as a practical building block for memory-efficient LLM serving with reusable KV caches.
>
---
#### [new 041] FlashEVA: Accelerating LLM inference via Efficient Attention
- **分类: cs.CL; cs.AI**

- **简介: FlashEVA提出一种高效注意力机制，降低LLM推理的内存与计算开销，支持仅用1.5Btokens微调，实现最高6.7倍吞吐提升与5倍显存降低，兼顾效率与精度可调性。**

- **链接: [http://arxiv.org/pdf/2511.00576v1](http://arxiv.org/pdf/2511.00576v1)**

> **作者:** Juan Gabriel Kostelec; Qinghai Guo
>
> **备注:** Technical Report
>
> **摘要:** Transformer models have revolutionized natural language processing, achieving state-of-the-art performance and demonstrating remarkable scalability. However, their memory demands, particularly due to maintaining full context in memory, pose significant challenges for inference. In this paper, we present FlashEVA, an efficient implementation of EVA (Efficient Attention via Control Variates), and demonstrate how to finetune transformers to adapt to FlashEVA attention. Our method enables fine-tuning of Transformer models with as few as 1.5B tokens while preserving effectiveness across various downstream tasks. Notably, FlashEVA achieves up to 6.7x higher throughput and 5x lower peak GPU memory usage during inference compared to standard Transformer implementations. Despite these improvements, we observe limitations in retrieval-focused tasks. Our implementation offers control over the trade-off between throughput and accuracy through adjustable hyperparameters, providing flexibility for diverse use cases. This work represents a significant step towards more efficient and adaptable Transformer-based models for inference.
>
---
#### [new 042] OpenSIR: Open-Ended Self-Improving Reasoner
- **分类: cs.CL**

- **简介: OpenSIR提出一种无外部监督的自博弈框架，让LLM通过交替扮演师生角色，自主生成并求解新颖数学问题，实现开放-ended自我提升，显著提升模型在GSM8K和College Math上的推理能力。**

- **链接: [http://arxiv.org/pdf/2511.00602v1](http://arxiv.org/pdf/2511.00602v1)**

> **作者:** Wai-Chung Kwan; Joshua Ong Jun Leang; Pavlos Vougiouklis; Jeff Z. Pan; Marco Valentino; Pasquale Minervini
>
> **摘要:** Recent advances in large language model (LLM) reasoning through reinforcement learning rely on annotated datasets for verifiable rewards, which may limit models' ability to surpass human-level performance. While self-play offers a promising alternative, existing approaches depend on external verifiers or cannot learn open-endedly. We present Open-Ended Self-Improving Reasoner (OpenSIR), a self-play framework where an LLM learns to generate and solve novel problems by alternating teacher and student roles without external supervision. To generate novel problems, OpenSIR optimises for both difficulty and diversity, rewarding problems that challenge appropriately while exploring distinct concepts, enabling open-ended mathematical discovery. Starting from a single trivial seed problem, OpenSIR substantially improves instruction models: Llama-3.2-3B-Instruct advances from 73.9 to 78.3 on GSM8K, and from 28.8 to 34.4 on College Math, while Gemma-2-2B-Instruct rises from 38.5 to 58.7 on GSM8K. Our analyses reveal that OpenSIR achieves open-ended learning through co-evolving teacher-student roles that adaptively calibrate difficulty and drive diverse exploration, progressing autonomously from basic to advanced mathematics.
>
---
#### [new 043] ColMate: Contrastive Late Interaction and Masked Text for Multimodal Document Retrieval
- **分类: cs.CL**

- **简介: 论文提出ColMate，面向多模态文档检索任务，解决传统方法照搬文本检索技术的问题。通过OCR预训练、掩码对比学习和晚期交互评分，提升多模态表征能力，在ViDoRe V2上取得3.61%性能提升。**

- **链接: [http://arxiv.org/pdf/2511.00903v1](http://arxiv.org/pdf/2511.00903v1)**

> **作者:** Ahmed Masry; Megh Thakkar; Patrice Bechard; Sathwik Tejaswi Madhusudhan; Rabiul Awal; Shambhavi Mishra; Akshay Kalkunte Suresh; Srivatsava Daruru; Enamul Hoque; Spandana Gella; Torsten Scholak; Sai Rajeswar
>
> **摘要:** Retrieval-augmented generation has proven practical when models require specialized knowledge or access to the latest data. However, existing methods for multimodal document retrieval often replicate techniques developed for text-only retrieval, whether in how they encode documents, define training objectives, or compute similarity scores. To address these limitations, we present ColMate, a document retrieval model that bridges the gap between multimodal representation learning and document retrieval. ColMate utilizes a novel OCR-based pretraining objective, a self-supervised masked contrastive learning objective, and a late interaction scoring mechanism more relevant to multimodal document structures and visual characteristics. ColMate obtains 3.61% improvements over existing retrieval models on the ViDoRe V2 benchmark, demonstrating stronger generalization to out-of-domain benchmarks.
>
---
#### [new 044] Towards Consistent Detection of Cognitive Distortions: LLM-Based Annotation and Dataset-Agnostic Evaluation
- **分类: cs.CL**

- **简介: 该论文面向认知扭曲检测任务，解决人工标注不一致问题，提出使用LLM生成一致标注，并构建基于Cohen's kappa的跨数据集评估框架，证明GPT-4可提升模型训练效果。**

- **链接: [http://arxiv.org/pdf/2511.01482v1](http://arxiv.org/pdf/2511.01482v1)**

> **作者:** Neha Sharma; Navneet Agarwal; Kairit Sirts
>
> **摘要:** Text-based automated Cognitive Distortion detection is a challenging task due to its subjective nature, with low agreement scores observed even among expert human annotators, leading to unreliable annotations. We explore the use of Large Language Models (LLMs) as consistent and reliable annotators, and propose that multiple independent LLM runs can reveal stable labeling patterns despite the inherent subjectivity of the task. Furthermore, to fairly compare models trained on datasets with different characteristics, we introduce a dataset-agnostic evaluation framework using Cohen's kappa as an effect size measure. This methodology allows for fair cross-dataset and cross-study comparisons where traditional metrics like F1 score fall short. Our results show that GPT-4 can produce consistent annotations (Fleiss's Kappa = 0.78), resulting in improved test set performance for models trained on these annotations compared to those trained on human-labeled data. Our findings suggest that LLMs can offer a scalable and internally consistent alternative for generating training data that supports strong downstream performance in subjective NLP tasks.
>
---
#### [new 045] DEEPAMBIGQA: Ambiguous Multi-hop Questions for Benchmarking LLM Answer Completeness
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出DeepAmbigQA基准，用于评估LLM在多跳推理与名称歧义双重挑战下的答案完整性。通过自动生成含歧义的复杂问题，揭示当前模型（如GPT-5）答案不完整的问题。**

- **链接: [http://arxiv.org/pdf/2511.01323v1](http://arxiv.org/pdf/2511.01323v1)**

> **作者:** Jiabao Ji; Min Li; Priyanshu Kumar; Shiyu Chang; Saloni Potdar
>
> **备注:** 25 pages
>
> **摘要:** Large language models (LLMs) with integrated search tools show strong promise in open-domain question answering (QA), yet they often struggle to produce complete answer set to complex questions such as Which actor from the film Heat won at least one Academy Award?, which requires (1) distinguishing between multiple films sharing the same title and (2) reasoning across a large set of actors to gather and integrate evidence. Existing QA benchmarks rarely evaluate both challenges jointly. To address this, we introduce DeepAmbigQAGen, an automatic data generation pipeline that constructs QA tasks grounded in text corpora and linked knowledge graph, generating natural and verifiable questions that systematically embed name ambiguity and multi-step reasoning. Based on this, we build DeepAmbigQA, a dataset of 3,600 questions requiring multi-hop reasoning and half of them explicit name ambiguity resolving. Experiments reveal that, even state-of-the-art GPT-5 show incomplete answers, achieving only 0.13 exact match on ambiguous questions and 0.21 on non-ambiguous questions. These findings highlight the need for more robust QA systems aimed at information gathering and answer completeness.
>
---
#### [new 046] Friend or Foe: How LLMs' Safety Mind Gets Fooled by Intent Shift Attack
- **分类: cs.CL**

- **简介: 该论文提出Intent Shift Attack（ISA），通过微小修改使有害请求被LLMs误判为 benign 请求，突破现有安全机制。任务为对抗攻击，解决意图识别漏洞问题，工作包括构建意图变换范式、实现高成功率攻击并揭示防御短板。**

- **链接: [http://arxiv.org/pdf/2511.00556v1](http://arxiv.org/pdf/2511.00556v1)**

> **作者:** Peng Ding; Jun Kuang; Wen Sun; Zongyu Wang; Xuezhi Cao; Xunliang Cai; Jiajun Chen; Shujian Huang
>
> **备注:** Preprint, 14 pages, 5 figures, 7 tables
>
> **摘要:** Large language models (LLMs) remain vulnerable to jailbreaking attacks despite their impressive capabilities. Investigating these weaknesses is crucial for robust safety mechanisms. Existing attacks primarily distract LLMs by introducing additional context or adversarial tokens, leaving the core harmful intent unchanged. In this paper, we introduce ISA (Intent Shift Attack), which obfuscates LLMs about the intent of the attacks. More specifically, we establish a taxonomy of intent transformations and leverage them to generate attacks that may be misperceived by LLMs as benign requests for information. Unlike prior methods relying on complex tokens or lengthy context, our approach only needs minimal edits to the original request, and yields natural, human-readable, and seemingly harmless prompts. Extensive experiments on both open-source and commercial LLMs show that ISA achieves over 70% improvement in attack success rate compared to direct harmful prompts. More critically, fine-tuning models on only benign data reformulated with ISA templates elevates success rates to nearly 100%. For defense, we evaluate existing methods and demonstrate their inadequacy against ISA, while exploring both training-free and training-based mitigation strategies. Our findings reveal fundamental challenges in intent inference for LLMs safety and underscore the need for more effective defenses. Our code and datasets are available at https://github.com/NJUNLP/ISA.
>
---
#### [new 047] ZoFia: Zero-Shot Fake News Detection with Entity-Guided Retrieval and Multi-LLM Interaction
- **分类: cs.CL; cs.AI**

- **简介: ZoFia提出一种零样本虚假新闻检测框架，通过实体引导检索与多LLM协作辩论，解决模型知识过时与幻觉问题，提升对新兴新闻的泛化能力。**

- **链接: [http://arxiv.org/pdf/2511.01188v1](http://arxiv.org/pdf/2511.01188v1)**

> **作者:** Lvhua Wu; Xuefeng Jiang; Sheng Sun; Tian Wen; Yuwei Wang; Min Liu
>
> **摘要:** The rapid spread of fake news threatens social stability and public trust, rendering its detection an imperative research priority. Although large language models (LLMs) excel at numerous natural language processing tasks with their remarkable contextual understanding and extensive prior knowledge, the time-bounded knowledge coverage and tendency for generating hallucination content reduce their reliability when handling fast-evolving news streams. Furthermore, models trained on existing static datasets also often lack the generalization needed for emerging news topics. To address these challenges, we propose ZoFia, a novel two-stage zero-shot fake news detection framework. First, we introduce Hierarchical Salience to quantify the importance of entities in the news content, and propose the SC-MMR algorithm to effectively select an informative and diverse set of keywords that serve as queries for retrieving up-to-date external evidence. Subsequently, a multi LLM interactive system, in which each agent assumes a distinct role, performs multi-view collaborative analysis and adversarial debate over the news text and its related information, and finally produces an interpretable and robust judgment. Comprehensive experiments on two public datasets demonstrate that ZoFia obviously outperforms existing zero-shot baselines and most of few-shot methods. Our codes will be open-sourced to facilitate related communities.
>
---
#### [new 048] PrefixNLI: Detecting Factual Inconsistencies as Soon as They Arise
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出PrefixNLI任务，旨在实时检测生成文本前缀中的事实不一致。作者构建专用模型MiniTruePrefixes，在前缀级 entailment 任务上显著超越基线，并将其融入解码框架，提升生成事实性，使小模型达到大模型性能。**

- **链接: [http://arxiv.org/pdf/2511.01359v1](http://arxiv.org/pdf/2511.01359v1)**

> **作者:** Sapir Harary; Eran Hirsch; Aviv Slobodkin; David Wan; Mohit Bansal; Ido Dagan
>
> **备注:** 9 pages + appendix. Code, datasets, and models are available at https://github.com/sapirharary/PrefixNLI
>
> **摘要:** Natural Language Inference (NLI) models have been used in various ways to improve the factuality of LLM outputs. This is typically done by applying an NLI model to judge whether the model output is entailed from the supposed evidence, triggering some corrective actions, such as beam reranking at inference time or RL rewards during training. While NLI models are trained to detect factual inconsistencies over complete sentences, decisions in the common autoregressive generation architecture are made for each evolving text prefix, during decoding. Addressing this setting, we generalize the entailment detection task to apply over arbitrary text prefixes, and suggest its utility for improving generation faithfulness. Providing suitable evaluation and training datasets for this task, we train MiniTruePrefixes, a novel specialized model that better detects factual inconsistencies over text prefixes, outperforming comparable baseline NLI models by 5-14 F1 points in prefix-level entailment. We further demonstrate that integrating MiniTruePrefixes into a controlled decoding framework substantially improves factual consistency in abstractive summarization. When guided by MiniTruePrefixes, LLaMA-3.2-3B-Instruct matches the faithfulness and runtime of the 8B model from the same model family, while using only half the memory.
>
---
#### [new 049] PlotCraft: Pushing the Limits of LLMs for Complex and Interactive Data Visualization
- **分类: cs.CL**

- **简介: 论文提出PlotCraft基准，评估LLMs在复杂交互式可视化中的能力，发现其性能不足；并构建SynthVis-30K数据集与轻量模型PlotCraftor，显著提升复杂图表生成效果，尤其在困难任务上提升超50%。**

- **链接: [http://arxiv.org/pdf/2511.00010v1](http://arxiv.org/pdf/2511.00010v1)**

> **作者:** Jiajun Zhang; Jianke Zhang; Zeyu Cui; Jiaxi Yang; Lei Zhang; Binyuan Hui; Qiang Liu; Zilei Wang; Liang Wang; Junyang Lin
>
> **摘要:** Recent Large Language Models (LLMs) have demonstrated remarkable profi- ciency in code generation. However, their ability to create complex visualiza- tions for scaled and structured data remains largely unevaluated and underdevel- oped. To address this gap, we introduce PlotCraft, a new benchmark featuring 1k challenging visualization tasks that cover a wide range of topics, such as fi- nance, scientific research, and sociology. The benchmark is structured around seven high-level visualization tasks and encompasses 48 distinct chart types. Cru- cially, it is the first to systematically evaluate both single-turn generation and multi-turn refinement across a diverse spectrum of task complexities. Our com- prehensive evaluation of 23 leading LLMs on PlotCraft reveals obvious per- formance deficiencies in handling sophisticated visualization tasks. To bridge this performance gap, we develope SynthVis-30K, a large-scale, high-quality dataset of complex visualization code synthesized via a collaborative agent frame- work. Building upon this dataset, we develope PlotCraftor, a novel code gener- ation model that achieves strong capabilities in complex data visualization with a remarkably small size. Across VisEval, PandasPlotBench, and our proposed PlotCraft, PlotCraftor shows performance comparable to that of leading propri- etary approaches. Especially, on hard task, Our model achieves over 50% per- formance improvement. We will release the benchmark, dataset, and code at https://github.com/Speakn0w/PlotCraft-Benchmark.
>
---
#### [new 050] Efficient Tool-Calling Multi-Expert NPC Agent for Commonsense Persona-Grounded Dialogue
- **分类: cs.CL**

- **简介: 该论文构建了一个基于Qwen3和LoRA的多专家NPC系统，解决互动环境中自然对话与上下文动作执行的协同问题，通过工具调用、响应解释与对话三个专家模块实现高效响应，已在常识人物对话挑战赛中获第二名。**

- **链接: [http://arxiv.org/pdf/2511.01720v1](http://arxiv.org/pdf/2511.01720v1)**

> **作者:** Mahammad Nuriyev
>
> **备注:** 10 pages, 1 figure, 2 tables. Technical report for the Commonsense Persona-Grounded Dialogue Challenge (CPDC) 2025, part of the Wordplay 2025 Workshop @ EMNLP 2025
>
> **摘要:** We present a multi-expert system for creating Non-Player Characters (NPCs) capable of both natural dialogue and contextual action execution in interactive environments. Using Qwen3 as the base model and Low-Rank Adaptation (LoRA) adapters, we instantiate three specialists: tool calling, tool-response interpretation, and direct dialogue. Our system comfortably meets the computational efficiency requirements, delivering fast responses and maintaining modest resource usage on L40S GPUs. In the Commonsense Persona-Grounded Dialogue Challenge 2025, our method ranked second overall. Code available at: https://github.com/MahammadNuriyev62/CPDC-challenge-2025-solution/
>
---
#### [new 051] HPLT~3.0: Very Large-Scale Multilingual Resources for LLM and MT. Mono- and Bi-lingual Data, Multilingual Evaluation, and Pre-Trained Models
- **分类: cs.CL**

- **简介: 该论文发布HPLT~3.0，构建全球最大规模（30万亿token）多语言文本数据集，涵盖近200种语言，提供开源预处理管道、高质量标注及多语言评测基准，并训练发布57个单语模型与平行语料，助力LLM与MT研究。**

- **链接: [http://arxiv.org/pdf/2511.01066v1](http://arxiv.org/pdf/2511.01066v1)**

> **作者:** Stephan Oepen; Nikolay Arefev; Mikko Aulamo; Marta Bañón; Maja Buljan; Laurie Burchell; Lucas Charpentier; Pinzhen Chen; Mariya Fedorova; Ona de Gibert; Barry Haddow; Jan Hajič; Jindrič Helcl; Andrey Kutuzov; Zihao Li; Risto Luukkonen; Bhavitvya Malik; Vladislav Mikhailov; Amanda Myntti; Dayyán O'Brien; Lucie Poláková; Sampo Pyysalo; Gema Ramírez Sánchez; Janine Siewert; Pavel Stepachev; Jörg Tiedemann; Teemu Vahtola; Fedor Vitiugin; Tea Vojtěchová; Jaume Zaragoza
>
> **摘要:** We present an ongoing initiative to provide open, very large, high-quality, and richly annotated textual datasets for almost 200 languages. At 30 trillion tokens, this is likely the largest generally available multilingual collection of LLM pre-training data. At 30 trillion tokens, this is likely the largest generally available multilingual collection of LLM pre-training data. These datasets are derived from web crawls from different sources and accompanied with a complete, open-source pipeline for document selection from web archives, text extraction from HTML, language identification for noisy texts, exact and near-deduplication, annotation with, among others, register labels, text quality estimates, and personally identifiable information; and final selection and filtering. We report on data quality probes through contrastive and analytical statistics, through manual inspection of samples for 24 languages, and through end-to-end evaluation of various language model architectures trained on this data. For multilingual LLM evaluation, we provide a comprehensive collection of benchmarks for nine European languages, with special emphasis on natively created tasks, mechanisms to mitigate prompt sensitivity, and refined normalization and aggregation of scores. Additionally, we train and evaluate a family of 57 monolingual encoder-decoder models, as well as a handful of monolingual GPT-like reference models. Besides the monolingual data and models, we also present a very large collection of parallel texts automatically mined from this data, together with a novel parallel corpus synthesized via machine translation.
>
---
#### [new 052] IL-PCSR: Legal Corpus for Prior Case and Statute Retrieval
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **简介: 该论文提出IL-PCSR，首个联合法律条文与判例检索的中文语料库，解决二者独立建模导致的关联性缺失问题，构建多模型基准并提出LLM重排序方法，提升双任务协同检索性能。**

- **链接: [http://arxiv.org/pdf/2511.00268v1](http://arxiv.org/pdf/2511.00268v1)**

> **作者:** Shounak Paul; Dhananjay Ghumare; Pawan Goyal; Saptarshi Ghosh; Ashutosh Modi
>
> **备注:** Accepted at EMNLP 2025 (Main)
>
> **摘要:** Identifying/retrieving relevant statutes and prior cases/precedents for a given legal situation are common tasks exercised by law practitioners. Researchers to date have addressed the two tasks independently, thus developing completely different datasets and models for each task; however, both retrieval tasks are inherently related, e.g., similar cases tend to cite similar statutes (due to similar factual situation). In this paper, we address this gap. We propose IL-PCR (Indian Legal corpus for Prior Case and Statute Retrieval), which is a unique corpus that provides a common testbed for developing models for both the tasks (Statute Retrieval and Precedent Retrieval) that can exploit the dependence between the two. We experiment extensively with several baseline models on the tasks, including lexical models, semantic models and ensemble based on GNNs. Further, to exploit the dependence between the two tasks, we develop an LLM-based re-ranking approach that gives the best performance.
>
---
#### [new 053] Do You Know About My Nation? Investigating Multilingual Language Models' Cultural Literacy Through Factual Knowledge
- **分类: cs.CL**

- **简介: 该论文提出XNationQA基准，评估多语言大模型对非西方国家文化事实的知识掌握，揭示模型在英语中优于本地语言的“文化认知偏差”，并发现跨语言知识迁移能力有限，尤其开源模型更弱。**

- **链接: [http://arxiv.org/pdf/2511.00657v1](http://arxiv.org/pdf/2511.00657v1)**

> **作者:** Eshaan Tanwar; Anwoy Chatterjee; Michael Saxon; Alon Albalak; William Yang Wang; Tanmoy Chakraborty
>
> **备注:** Accepted in EMNLP 2025. Code at: https://github.com/EshaanT/XNationQA
>
> **摘要:** Most multilingual question-answering benchmarks, while covering a diverse pool of languages, do not factor in regional diversity in the information they capture and tend to be Western-centric. This introduces a significant gap in fairly evaluating multilingual models' comprehension of factual information from diverse geographical locations. To address this, we introduce XNationQA for investigating the cultural literacy of multilingual LLMs. XNationQA encompasses a total of 49,280 questions on the geography, culture, and history of nine countries, presented in seven languages. We benchmark eight standard multilingual LLMs on XNationQA and evaluate them using two novel transference metrics. Our analyses uncover a considerable discrepancy in the models' accessibility to culturally specific facts across languages. Notably, we often find that a model demonstrates greater knowledge of cultural information in English than in the dominant language of the respective culture. The models exhibit better performance in Western languages, although this does not necessarily translate to being more literate for Western countries, which is counterintuitive. Furthermore, we observe that models have a very limited ability to transfer knowledge across languages, particularly evident in open-source models.
>
---
#### [new 054] G2: Guided Generation for Enhanced Output Diversity in LLMs
- **分类: cs.CL**

- **简介: 该论文属于大语言模型生成优化任务，解决输出多样性不足且牺牲质量的问题。提出G2方法，通过无训练的双引导机制在解码阶段干预生成，提升多样性的同时保持质量。**

- **链接: [http://arxiv.org/pdf/2511.00432v1](http://arxiv.org/pdf/2511.00432v1)**

> **作者:** Zhiwen Ruan; Yixia Li; Yefeng Liu; Yun Chen; Weihua Luo; Peng Li; Yang Liu; Guanhua Chen
>
> **备注:** EMNLP 2025
>
> **摘要:** Large Language Models (LLMs) have demonstrated exceptional performance across diverse natural language processing tasks. However, these models exhibit a critical limitation in output diversity, often generating highly similar content across multiple attempts. This limitation significantly affects tasks requiring diverse outputs, from creative writing to reasoning. Existing solutions, like temperature scaling, enhance diversity by modifying probability distributions but compromise output quality. We propose Guide-to-Generation (G2), a training-free plug-and-play method that enhances output diversity while preserving generation quality. G2 employs a base generator alongside dual Guides, which guide the generation process through decoding-based interventions to encourage more diverse outputs conditioned on the original query. Comprehensive experiments demonstrate that G2 effectively improves output diversity while maintaining an optimal balance between diversity and quality.
>
---
#### [new 055] Consistently Simulating Human Personas with Multi-Turn Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦于提升大语言模型在多轮对话中的人设一致性，解决角色漂移问题。通过定义三项自动评估指标并结合多轮强化学习，优化患者、学生、社交伙伴三类角色的模拟表现，一致性提升超55%。**

- **链接: [http://arxiv.org/pdf/2511.00222v1](http://arxiv.org/pdf/2511.00222v1)**

> **作者:** Marwa Abdulhai; Ryan Cheng; Donovan Clay; Tim Althoff; Sergey Levine; Natasha Jaques
>
> **摘要:** Large Language Models (LLMs) are increasingly used to simulate human users in interactive settings such as therapy, education, and social role-play. While these simulations enable scalable training and evaluation of AI agents, off-the-shelf LLMs often drift from their assigned personas, contradict earlier statements, or abandon role-appropriate behavior. We introduce a unified framework for evaluating and improving persona consistency in LLM-generated dialogue. We define three automatic metrics: prompt-to-line consistency, line-to-line consistency, and Q&A consistency, that capture different types of persona drift and validate each against human annotations. Using these metrics as reward signals, we apply multi-turn reinforcement learning to fine-tune LLMs for three user roles: a patient, a student, and a social chat partner. Our method reduces inconsistency by over 55%, resulting in more coherent and faithful simulated users.
>
---
#### [new 056] "Don't Teach Minerva": Guiding LLMs Through Complex Syntax for Faithful Latin Translation with RAG
- **分类: cs.CL; cs.DL**

- **简介: 该论文聚焦拉丁语翻译任务，解决低资源、形态复杂语言的高质量翻译难题。提出无需微调的RAG增强draft-refinement框架，利用NLLB生成初稿，LLM润色并引入外部示例，性能媲美GPT-5。**

- **链接: [http://arxiv.org/pdf/2511.01454v1](http://arxiv.org/pdf/2511.01454v1)**

> **作者:** Sergio Torres Aguilar
>
> **摘要:** Translating a morphology-rich, low-resource language like Latin poses significant challenges. This paper introduces a reproducible draft-based refinement pipeline that elevates open-source Large Language Models (LLMs) to a performance level statistically comparable to top-tier proprietary systems. Our method first uses a fine-tuned NLLB-1.3B model to generate a high-quality, structurally faithful draft. A zero-shot LLM (Llama-3.3 or Qwen3) then polishes this draft, a process that can be further enhanced by augmenting the context with retrieved out-context examples (RAG). We demonstrate the robustness of this approach on two distinct benchmarks: a standard in-domain test set (Rosenthal, 2023) and a new, challenging out-of-domain (OOD) set of 12th-century Latin letters (2025). Our central finding is that this open-source RAG system achieves performance statistically comparable to the GPT-5 baseline, without any task-specific LLM fine-tuning. We release the pipeline, the Chartres OOD set, and evaluation scripts and models to facilitate replicability and further research.
>
---
#### [new 057] Open Character Training: Shaping the Persona of AI Assistants through Constitutional AI
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出开放的“角色训练”方法，利用宪法AI与合成自省数据，微调大模型以可控塑造AI助手的个性（如幽默、关怀等），提升行为一致性与抗对抗性，同时保持通用能力，首次开源实现该流程。**

- **链接: [http://arxiv.org/pdf/2511.01689v1](http://arxiv.org/pdf/2511.01689v1)**

> **作者:** Sharan Maiya; Henning Bartsch; Nathan Lambert; Evan Hubinger
>
> **备注:** 12 pages, 6 figures, 4 tables
>
> **摘要:** The character of the "AI assistant" persona generated by modern chatbot large language models influences both surface-level behavior and apparent values, beliefs, and ethics. These all affect interaction quality, perceived intelligence, and alignment with both developer and user intentions. The shaping of this persona, known as character training, is a critical component of industry post-training, yet remains effectively unstudied in the academic literature. We introduce the first open implementation of character training, leveraging Constitutional AI and a new data pipeline using synthetic introspective data to shape the assistant persona in a more effective and controlled manner than alternatives such as constraining system prompts or activation steering. Specifically, we fine-tune three popular open-weights models using 11 example personas, such as humorous, deeply caring, or even malevolent. To track the effects of our approach, we introduce a method which analyzes revealed preferences, uncovering clear and holistic changes in character. We find these changes are more robust to adversarial prompting than the above two alternatives, while also leading to more coherent and realistic generations. Finally, we demonstrate this fine-tuning has little to no effect on general capabilities as measured by common benchmarks. We describe and open-source our full post-training method, the implementation of which can be found at https://github.com/maiush/OpenCharacterTraining.
>
---
#### [new 058] Leveraging the Cross-Domain & Cross-Linguistic Corpus for Low Resource NMT: A Case Study On Bhili-Hindi-English Parallel Corpus
- **分类: cs.CL**

- **简介: 该论文针对低资源部落语言Bhili的机器翻译难题，构建了首个大规模Bhili-Hindi-English平行语料库（BHEPC），并评估多种多语言模型的翻译性能，验证了NLLB-200在跨语言、跨领域翻译中的优越性，推动边缘语言NLP发展。**

- **链接: [http://arxiv.org/pdf/2511.00486v1](http://arxiv.org/pdf/2511.00486v1)**

> **作者:** Pooja Singh; Shashwat Bhardwaj; Vaibhav Sharma; Sandeep Kumar
>
> **备注:** Accepted in EMNLP 2025
>
> **摘要:** The linguistic diversity of India poses significant machine translation challenges, especially for underrepresented tribal languages like Bhili, which lack high-quality linguistic resources. This paper addresses the gap by introducing Bhili-Hindi-English Parallel Corpus (BHEPC), the first and largest parallel corpus worldwide comprising 110,000 meticulously curated sentences across Bhili, Hindi, and English. The corpus was created with the assistance of expert human translators. BHEPC spans critical domains such as education, administration, and news, establishing a valuable benchmark for research in low resource machine translation. To establish a comprehensive Bhili Machine Translation benchmark, we evaluated a wide range of proprietary and open-source Multilingual Large Language Models (MLLMs) on bidirectional translation tasks between English/Hindi and Bhili. Comprehensive evaluation demonstrates that the fine-tuned NLLB-200 distilled 600M variant model outperforms others, highlighting the potential of multilingual models in low resource scenarios. Furthermore, we investigated the generative translation capabilities of multilingual LLMs on BHEPC using in-context learning, assessing performance under cross-domain generalization and quantifying distributional divergence. This work bridges a critical resource gap and promotes inclusive natural language processing technologies for low-resource and marginalized languages globally.
>
---
#### [new 059] Word Salad Chopper: Reasoning Models Waste A Ton Of Decoding Budget On Useless Repetitions, Self-Knowingly
- **分类: cs.CL**

- **简介: 该论文针对大推理模型输出中大量无意义重复（“词 salad”）问题，提出WordSaladChopper，通过检测隐藏状态自适应截断并重生成，显著压缩输出长度，提升效率且保留质量，属推理优化任务。**

- **链接: [http://arxiv.org/pdf/2511.00536v1](http://arxiv.org/pdf/2511.00536v1)**

> **作者:** Wenya Xie; Shaochen; Zhong; Hoang Anh Duy Le; Zhaozhuo Xu; Jianwen Xie; Zirui Liu
>
> **摘要:** Large Reasoning Models (LRMs) are often bottlenecked by the high cost of output tokens. We show that a significant portion of these tokens are useless self-repetitions - what we call "word salad" - that exhaust the decoding budget without adding value. Interestingly, we observe that LRMs are self-aware when trapped in these loops: the hidden states of <\n\n> tokens trailing each reasoning chunk exhibit patterns that allow us to detect word salad behavior on-the-fly via a single-layer linear classifier. Once detected, a simple chop appended by a straightforward regeneration prompt yields substantial length savings with minimal quality loss. Our work offers WordSaladChopper (WSC) - a lightweight, turnkey component for LRM that is minimally invasive to its reasoning trajectory by only removing semantically redundant tokens. Given its low overhead, strong savings, and the lack of semantic value of word salad tokens, we believe it is not too far-fetched to argue that WSC - or a similar component - is a must-have for all LRM applications with user experience in mind. Our code is publicly available at https://github.com/wenyaxie023/WordSaladChopper.
>
---
#### [new 060] Remembering Unequally: Global and Disciplinary Bias in LLM-Generated Co-Authorship Networks
- **分类: cs.CL**

- **简介: 该论文研究LLM记忆机制对合著网络生成的偏见影响，属于公平性评估任务。通过分析三大模型在不同学科与地区的输出，揭示其对高被引学者的偏好，并发现部分领域（如临床医学）和区域（如非洲）存在更均衡代表。**

- **链接: [http://arxiv.org/pdf/2511.00476v1](http://arxiv.org/pdf/2511.00476v1)**

> **作者:** Ghazal Kalhor; Afra Mashhadi
>
> **摘要:** Ongoing breakthroughs in Large Language Models (LLMs) are reshaping search and recommendation platforms at their core. While this shift unlocks powerful new scientometric tools, it also exposes critical fairness and bias issues that could erode the integrity of the information ecosystem. Additionally, as LLMs become more integrated into web-based searches for scholarly tools, their ability to generate summarized research work based on memorized data introduces new dimensions to these challenges. The extent of memorization in LLMs can impact the accuracy and fairness of the co-authorship networks they produce, potentially reflecting and amplifying existing biases within the scientific community and across different regions. This study critically examines the impact of LLM memorization on the co-authorship networks. To this end, we assess memorization effects across three prominent models, DeepSeek R1, Llama 4 Scout, and Mixtral 8x7B, analyzing how memorization-driven outputs vary across academic disciplines and world regions. While our global analysis reveals a consistent bias favoring highly cited researchers, this pattern is not uniformly observed. Certain disciplines, such as Clinical Medicine, and regions, including parts of Africa, show more balanced representation, pointing to areas where LLM training data may reflect greater equity. These findings underscore both the risks and opportunities in deploying LLMs for scholarly discovery.
>
---
#### [new 061] DEER: Disentangled Mixture of Experts with Instance-Adaptive Routing for Generalizable Machine-Generated Text Detection
- **分类: cs.CL**

- **简介: 该论文针对机器生成文本检测中的域泛化问题，提出DEER框架，通过解耦专家模块分离域特异与域通用特征，并用强化学习实现实例自适应路由，无需域标签即可提升跨域检测性能。**

- **链接: [http://arxiv.org/pdf/2511.01192v1](http://arxiv.org/pdf/2511.01192v1)**

> **作者:** Guoxin Ma; Xiaoming Liu; Zhanhan Zhang; Chengzhengxu Li; Shengchao Liu; Yu Lan
>
> **备注:** Under Review
>
> **摘要:** Detecting machine-generated text (MGT) has emerged as a critical challenge, driven by the rapid advancement of large language models (LLMs) capable of producing highly realistic, human-like content. However, the performance of current approaches often degrades significantly under domain shift. To address this challenge, we propose a novel framework designed to capture both domain-specific and domain-general MGT patterns through a two-stage Disentangled mixturE-of-ExpeRts (DEER) architecture. First, we introduce a disentangled mixture-of-experts module, in which domain-specific experts learn fine-grained, domain-local distinctions between human and machine-generated text, while shared experts extract transferable, cross-domain features. Second, to mitigate the practical limitation of unavailable domain labels during inference, we design a reinforcement learning-based routing mechanism that dynamically selects the appropriate experts for each input instance, effectively bridging the train-inference gap caused by domain uncertainty. Extensive experiments on five in-domain and five out-of-domain benchmark datasets demonstrate that DEER consistently outperforms state-of-the-art methods, achieving average F1-score improvements of 1.39% and 5.32% on in-domain and out-of-domain datasets respectively, along with accuracy gains of 1.35% and 3.61% respectively. Ablation studies confirm the critical contributions of both disentangled expert specialization and adaptive routing to model performance.
>
---
#### [new 062] SpecDiff-2: Scaling Diffusion Drafter Alignment For Faster Speculative Decoding
- **分类: cs.CL**

- **简介: SpecDiff-2提出一种非自回归扩散草稿模型，解决传统推测解码中串行生成与模型不匹配问题，显著提升LLM推理速度达5.5倍，且无精度损失。**

- **链接: [http://arxiv.org/pdf/2511.00606v1](http://arxiv.org/pdf/2511.00606v1)**

> **作者:** Jameson Sandler; Jacob K. Christopher; Thomas Hartvigsen; Nando Fioretto
>
> **摘要:** Speculative decoding has become the standard approach for accelerating Large Language Model (LLM) inference. It exploits a lossless draft-then-verify procedure to circumvent the latency of autoregressive decoding, achieving impressive speed-ups. Yet, current speculative decoding approaches remain limited by two fundamental bottlenecks: (1) the autoregressive dependency during drafting which limits parallelism, and (2) frequent rejections of draft tokens caused by misalignment between the draft and verify models. This paper proposes SpecDiff-2, a novel framework to jointly address these two bottlenecks. It leverages discrete diffusion as a non-autoregressive drafter to address bottleneck (1) and develops novel techniques to calibrate discrete diffusion drafters with autoregressive verifiers, addressing bottleneck (2). Experimental results across a comprehensive benchmark suite show that SpecDiff-2 achieves a new state-of-the-art across reasoning, coding, and mathematical benchmarks, improving tokens-per-second by up to an average of +55% over previous baselines and obtaining up to 5.5x average speed-up over standard decoding, without any loss of accuracy.
>
---
#### [new 063] Multi-Step Knowledge Interaction Analysis via Rank-2 Subspace Disentanglement
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出一种基于秩-2子空间解耦的方法，首次实现多步分析大模型中参数知识与上下文知识的交互模式，揭示幻觉、忠实解释与思维链提示对知识依赖的影响，超越传统单步秩-1建模的局限。**

- **链接: [http://arxiv.org/pdf/2511.01706v1](http://arxiv.org/pdf/2511.01706v1)**

> **作者:** Sekh Mainul Islam; Pepa Atanasova; Isabelle Augenstein
>
> **备注:** Under review
>
> **摘要:** Natural Language Explanations (NLEs) describe how Large Language Models (LLMs) make decisions, drawing on both external Context Knowledge (CK) and Parametric Knowledge (PK) stored in model weights. Understanding their interaction is key to assessing the grounding of NLEs, yet it remains underexplored. Prior work has largely examined only single-step generation, typically the final answer, and has modelled PK and CK interaction only as a binary choice in a rank-1 subspace. This overlooks richer forms of interaction, such as complementary or supportive knowledge. We propose a novel rank-2 projection subspace that disentangles PK and CK contributions more accurately and use it for the first multi-step analysis of knowledge interactions across longer NLE sequences. Experiments on four QA datasets and three open-weight instruction-tuned LLMs show that diverse knowledge interactions are poorly represented in a rank-1 subspace but are effectively captured in our rank-2 formulation. Our multi-step analysis reveals that hallucinated NLEs align strongly with the PK direction, context-faithful ones balance PK and CK, and Chain-of-Thought prompting for NLEs shifts generated NLEs toward CK by reducing PK reliance. This work provides the first framework for systematic studies of multi-step knowledge interactions in LLMs through a richer rank-2 subspace disentanglement. Code and data: https://github.com/copenlu/pk-ck-knowledge-disentanglement.
>
---
#### [new 064] ToM: Leveraging Tree-oriented MapReduce for Long-Context Reasoning in Large Language Models
- **分类: cs.CL**

- **简介: 论文提出ToM框架，面向大语言模型长上下文推理任务，解决传统分块方法缺乏逻辑连贯性与长程依赖建模的问题，通过树状MapReduce结构实现层次化语义聚合与递归推理。**

- **链接: [http://arxiv.org/pdf/2511.00489v1](http://arxiv.org/pdf/2511.00489v1)**

> **作者:** Jiani Guo; Zuchao Li; Jie Wu; Qianren Wang; Yun Li; Lefei Zhang; Hai Zhao; Yujiu Yang
>
> **备注:** EMNLP 2025 Main Conference
>
> **摘要:** Large Language Models (LLMs), constrained by limited context windows, often face significant performance degradation when reasoning over long contexts. To address this, Retrieval-Augmented Generation (RAG) retrieves and reasons over chunks but frequently sacrifices logical coherence due to its reliance on similarity-based rankings. Similarly, divide-and-conquer frameworks (DCF) split documents into small chunks for independent reasoning and aggregation. While effective for local reasoning, DCF struggles to capture long-range dependencies and risks inducing conflicts by processing chunks in isolation. To overcome these limitations, we propose ToM, a novel Tree-oriented MapReduce framework for long-context reasoning. ToM leverages the inherent hierarchical structure of long documents (e.g., main headings and subheadings) by constructing a DocTree through hierarchical semantic parsing and performing bottom-up aggregation. Using a Tree MapReduce approach, ToM enables recursive reasoning: in the Map step, rationales are generated at child nodes; in the Reduce step, these rationales are aggregated across sibling nodes to resolve conflicts or reach consensus at parent nodes. Experimental results on 70B+ LLMs show that ToM significantly outperforms existing divide-and-conquer frameworks and retrieval-augmented generation methods, achieving better logical coherence and long-context reasoning. Our code is available at https://github.com/gjn12-31/ToM .
>
---
#### [new 065] With Privacy, Size Matters: On the Importance of Dataset Size in Differentially Private Text Rewriting
- **分类: cs.CL**

- **简介: 该论文首次系统研究数据集规模对差分隐私文本重写机制的影响，揭示其在隐私-效用权衡中的关键作用，呼吁DP-NLP领域采用更严谨的大规模评估方法。**

- **链接: [http://arxiv.org/pdf/2511.00487v1](http://arxiv.org/pdf/2511.00487v1)**

> **作者:** Stephen Meisenbacher; Florian Matthes
>
> **备注:** 11 pages, 1 figure, 5 tables. Accepted to IJCNLP-AACL 2025 (Main)
>
> **摘要:** Recent work in Differential Privacy with Natural Language Processing (DP NLP) has proposed numerous promising techniques in the form of text rewriting mechanisms. In the evaluation of these mechanisms, an often-ignored aspect is that of dataset size, or rather, the effect of dataset size on a mechanism's efficacy for utility and privacy preservation. In this work, we are the first to introduce this factor in the evaluation of DP text privatization, where we design utility and privacy tests on large-scale datasets with dynamic split sizes. We run these tests on datasets of varying size with up to one million texts, and we focus on quantifying the effect of increasing dataset size on the privacy-utility trade-off. Our findings reveal that dataset size plays an integral part in evaluating DP text rewriting mechanisms; additionally, these findings call for more rigorous evaluation procedures in DP NLP, as well as shed light on the future of DP NLP in practice and at scale.
>
---
#### [new 066] Plan-and-Write: Structure-Guided Length Control for LLMs without Model Retraining
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出一种无需重训练的提示工程方法，通过结构化规划与字数监控实现大语言模型的精确长度控制，解决现有方法依赖重训练或复杂工具的问题，在摘要任务中显著提升长度符合率并保持输出质量。**

- **链接: [http://arxiv.org/pdf/2511.01807v1](http://arxiv.org/pdf/2511.01807v1)**

> **作者:** Adewale Akinfaderin; Shreyas Subramanian; Akarsha Sehwag
>
> **备注:** Presented at Workshop on Prompt Optimization, KDD 2025, Toronto, Canada
>
> **摘要:** Length control in Large Language Models (LLMs) is a crucial but under-addressed challenge, with applications ranging from voice interfaces requiring concise responses to research summaries needing comprehensive outputs. Current approaches to length control, including Regularized DPO, Length-Instruction Fine Tuning, and tool-augmented methods, typically require expensive model retraining or complex inference-time tooling. This paper presents a prompt engineering methodology that enables precise length control without model retraining. Our structure-guided approach implements deliberate planning and word counting mechanisms within the prompt, encouraging the model to carefully track and adhere to specified length constraints. Comprehensive evaluations across six state-of-the-art LLMs demonstrate that our method significantly improves length fidelity for several models compared to standard prompting when applied to document summarization tasks, particularly for shorter-to-medium length constraints. The proposed technique shows varying benefits across different model architectures, with some models demonstrating up to 37.6% improvement in length adherence. Quality evaluations further reveal that our approach maintains or enhances overall output quality compared to standard prompting techniques. Our approach provides an immediately deployable solution for applications requiring precise length control, particularly valuable for production environments where model retraining is impractical or cost-prohibitive.
>
---
#### [new 067] Improving Romanian LLM Pretraining Data using Diversity and Quality Filtering
- **分类: cs.CL**

- **简介: 该论文聚焦于提升罗马尼亚语LLM预训练数据质量，通过LLM标注分析其与英语数据的差异，设计多级过滤策略（如教育价值、主题等），构建高质量语料，显著提升模型在多基准上的表现。**

- **链接: [http://arxiv.org/pdf/2511.01090v1](http://arxiv.org/pdf/2511.01090v1)**

> **作者:** Vlad Negoita; Mihai Masala; Traian Rebedea
>
> **摘要:** Large Language Models (LLMs) have recently exploded in popularity, often matching or outperforming human abilities on many tasks. One of the key factors in training LLMs is the availability and curation of high-quality data. Data quality is especially crucial for under-represented languages, where high-quality corpora are scarce. In this work we study the characteristics and coverage of Romanian pretraining corpora and we examine how they differ from English data. By training a lightweight multitask model on carefully LLM-annotated Romanian texts, we are able to analyze and perform multi-level filtering (e.g., educational value, topic, format) to generate high-quality pretraining datasets. Our experiments show noteworthy trends in the topics present in Romanian and English data, while also proving the effectiveness of filtering data through improved LLM pretraining performance across multiple benchmarks.
>
---
#### [new 068] SeaLLMs-Audio: Large Audio-Language Models for Southeast Asia
- **分类: cs.CL; cs.AI**

- **简介: 论文提出SeaLLMs-Audio，首个面向东南亚多语言（印尼、泰、越等）的大型音视频语言模型，解决区域语音AI资源匮乏问题，支持多模态输入与多任务处理，并构建SeaBench-Audio基准评估其性能。**

- **链接: [http://arxiv.org/pdf/2511.01670v1](http://arxiv.org/pdf/2511.01670v1)**

> **作者:** Chaoqun Liu; Mahani Aljunied; Guizhen Chen; Hou Pong Chan; Weiwen Xu; Yu Rong; Wenxuan Zhang
>
> **备注:** 10 pages
>
> **摘要:** We introduce SeaLLMs-Audio, the first large audio-language model (LALM) tailored for multiple Southeast Asian (SEA) languages-Indonesian (id), Thai (th), and Vietnamese (vi)-alongside English (en) and Chinese (zh). Trained on a large-scale audio corpus, SeaLLMs-Audio exhibits strong performance across diverse audio-centric tasks, spanning fine-grained audio understanding and voice-based interaction. Its key features include: 1) Multilingual: the model primarily supports 5 languages, namely Indonesian, Thai, Vietnamese, English, and Chinese; 2) Multimodal: the model accepts flexible input modalities, including audio only, text only, as well as audio with text; 3) Multi-task: the model supports a wide range of tasks, including audio analysis tasks such as Audio Captioning, Automatic Speech Recognition, Speech-to-Text Translation, Speech Emotion Recognition, Speech Question Answering, and Speech Summarization. It also enables voice-based dialogue, including answering factual, mathematical, and general knowledge queries. As a significant step towards advancing audio LLMs in Southeast Asia, we expect SeaLLMs-Audio to benefit both the regional research community and industry. To automate LALM evaluation for Southeast Asia, we introduce SeaBench-Audio, a benchmark spanning multiple tasks. Experiments show that SeaLLMs-Audio achieves competitive performance compared with other LALMs on SEA languages.
>
---
#### [new 069] ParaScopes: What do Language Models Activations Encode About Future Text?
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究语言模型激活态中对后续文本的编码能力，提出残差流解码器框架，首次在段落/文档尺度上解码未来5+词的上下文信息，旨在提升对模型长期规划能力的可解释性与监控。**

- **链接: [http://arxiv.org/pdf/2511.00180v1](http://arxiv.org/pdf/2511.00180v1)**

> **作者:** Nicky Pochinkov; Yulia Volkova; Anna Vasileva; Sai V R Chereddy
>
> **备注:** Main paper: 9 pages, 10 figures. Total 24 pages
>
> **摘要:** Interpretability studies in language models often investigate forward-looking representations of activations. However, as language models become capable of doing ever longer time horizon tasks, methods for understanding activations often remain limited to testing specific concepts or tokens. We develop a framework of Residual Stream Decoders as a method of probing model activations for paragraph-scale and document-scale plans. We test several methods and find information can be decoded equivalent to 5+ tokens of future context in small models. These results lay the groundwork for better monitoring of language models and better understanding how they might encode longer-term planning information.
>
---
#### [new 070] Training LLMs Beyond Next Token Prediction - Filling the Mutual Information Gap
- **分类: cs.CL; cs.AI**

- **简介: 该论文挑战LLM传统next-token预测训练方式，提出基于互信息的高信息量目标词选择策略，以提升训练效率与模型性能，适用于算术、文本分类与生成等任务。**

- **链接: [http://arxiv.org/pdf/2511.00198v1](http://arxiv.org/pdf/2511.00198v1)**

> **作者:** Chun-Hao Yang; Bo-Han Feng; Tzu-Yuan Lai; Yan Yu Chen; Yin-Kai Dean Huang; Shou-De Lin
>
> **摘要:** Optimizing training performance in large language models (LLMs) remains an essential challenge, particularly in improving model performance while maintaining computational costs. This work challenges the conventional approach of training LLMs using next-token prediction (NTP), arguing that by predicting information-rich tokens during training, there is a more effective way to train LLMs. We investigate the impact of the proposed solution in three kinds of tasks for LLMs: arithmetic, multi-label classification of text, and natural-language generation. This work offers a principled approach to optimizing LLM training, advancing both model performance and theoretical understanding of the target-token selection strategies.
>
---
#### [new 071] Tool-to-Agent Retrieval: Bridging Tools and Agents for Scalable LLM Multi-Agent Systems
- **分类: cs.CL**

- **简介: 该论文提出Tool-to-Agent Retrieval，解决LLM多智能体系统中工具功能被忽略、智能体选择不精准的问题，通过联合嵌入工具与智能体，实现细粒度检索，显著提升检索准确率。**

- **链接: [http://arxiv.org/pdf/2511.01854v1](http://arxiv.org/pdf/2511.01854v1)**

> **作者:** Elias Lumer; Faheem Nizar; Anmol Gulati; Pradeep Honaganahalli Basavaraju; Vamse Kumar Subbiah
>
> **摘要:** Recent advances in LLM Multi-Agent Systems enable scalable orchestration of sub-agents, each coordinating hundreds or thousands of tools or Model Context Protocol (MCP) servers. However, existing retrieval methods typically match queries against coarse agent-level descriptions before routing, which obscures fine-grained tool functionality and often results in suboptimal agent selection. We introduce Tool-to-Agent Retrieval, a unified framework that embeds both tools and their parent agents in a shared vector space and connects them through metadata relationships. By explicitly representing tool capabilities and traversing metadata to the agent level, Tool-to-Agent Retrieval enables granular tool-level or agent-level retrieval, ensuring that agents and their underlying tools or MCP servers are equally represented without the context dilution that arises from chunking many tools together. Evaluating Tool-to-Agent Retrieval across eight embedding models, our approach achieves consistent improvements of 19.4% in Recall@5 and 17.7% in nDCG@5 over previous state-of-the-art agent retrievers on the LiveMCPBench benchmark.
>
---
#### [new 072] BARD: budget-aware reasoning distillation
- **分类: cs.CL**

- **简介: 论文提出BARD框架，解决长链式推理蒸馏中冗余与计算预算不可控问题，通过两阶段训练（SFT+RL）实现推理能力与预算的联合优化，使小模型在可控推理长度下保持高性能。**

- **链接: [http://arxiv.org/pdf/2511.01470v1](http://arxiv.org/pdf/2511.01470v1)**

> **作者:** Lujie Niu; Lei Shen; Yi Jiang; Caixia Yuan; Xiaojie Wang; Wenbo Su; Bo zheng
>
> **摘要:** While long Chain-of-Thought (CoT) distillation effectively transfers reasoning capability to smaller language models, the reasoning process often remains redundant and computational budget uncontrollable, leading to inefficient resource usage. To address this limitation, we propose \textbf{Budget-Aware Reasoning Distillation (BARD)}, a novel framework that simultaneously distills reasoning capability and enables fine-grained control over the reasoning length. BARD uses the thinking budget as a user-specified control signal, allowing the model to dynamically balance reasoning performance and computational efficiency. To achieve this concept, BARD introduces a two-phase training regimen. The first phase, Supervised Fine-Tuning (SFT) on teacher-generated long CoT data compressed to various budget levels, bootstrapping the model's understanding of budget constraints. The second phase leverages Reinforcement Learning (RL) from a reward signal in consideration of reasoning performance and budget fidelity simultaneously. Incorporating the two-phase regimen is crucial to avoiding policy degradation and ensuring that both objectives are optimized jointly. Extensive experiments demonstrate that our method empowers an 8B student model to achieve strong performance on challenging reasoning benchmarks (\textit{AIME24, AIME25, GPQA}) while providing precise and adaptive control over its reasoning length across a wide range of budgets.
>
---
#### [new 073] Imperfect Language, Artificial Intelligence, and the Human Mind: An Interdisciplinary Approach to Linguistic Errors in Native Spanish Speakers
- **分类: cs.CL; cs.AI**

- **简介: 该论文属跨学科语言研究，旨在通过分析母语西班牙语者的语言错误，评估大语言模型对人类语言不完美性的理解与修正能力，推动更贴近认知真实的NLP系统开发。**

- **链接: [http://arxiv.org/pdf/2511.01615v1](http://arxiv.org/pdf/2511.01615v1)**

> **作者:** Francisco Portillo López
>
> **备注:** 12 pages, 3 figures
>
> **摘要:** Linguistic errors are not merely deviations from normative grammar; they offer a unique window into the cognitive architecture of language and expose the current limitations of artificial systems that seek to replicate them. This project proposes an interdisciplinary study of linguistic errors produced by native Spanish speakers, with the aim of analyzing how current large language models (LLM) interpret, reproduce, or correct them. The research integrates three core perspectives: theoretical linguistics, to classify and understand the nature of the errors; neurolinguistics, to contextualize them within real-time language processing in the brain; and natural language processing (NLP), to evaluate their interpretation against linguistic errors. A purpose-built corpus of authentic errors of native Spanish (+500) will serve as the foundation for empirical analysis. These errors will be tested against AI models such as GPT or Gemini to assess their interpretative accuracy and their ability to generalize patterns of human linguistic behavior. The project contributes not only to the understanding of Spanish as a native language but also to the development of NLP systems that are more cognitively informed and capable of engaging with the imperfect, variable, and often ambiguous nature of real human language.
>
---
#### [new 074] When, What, and How: Rethinking Retrieval-Enhanced Speculative Decoding
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出ReSpec，用于加速大语言模型推理，解决检索增强推测解码中低效触发与候选利用问题。通过熵引导触发、反馈选候选和松弛验证，显著提升加速比超25%-33%。**

- **链接: [http://arxiv.org/pdf/2511.01282v1](http://arxiv.org/pdf/2511.01282v1)**

> **作者:** Min Fang; Zhihui Fu; Qibin Zhao; Jun Wang
>
> **摘要:** Speculative decoding (SD) has emerged as an effective technique to accelerate large language model (LLM) inference without compromising output quality. However, the achievable speedup largely depends on the effectiveness of the drafting model. While model-based methods like EAGLE-2 are accurate but costly, retrieval-enhanced methods like SAM-Decoding rely on heuristic switching strategies that often trigger unnecessary retrievals. To address this, we propose ReSpec (\textbf{Re}trieval-enhanced \textbf{Spe}culative Decoding), a novel framework that transforms heuristic drafter switching into adaptive decision-making. ReSpec features three core innovations: 1) An \textbf{entropy-guided adaptive trigger} quantifies contextual predictability to initiate retrieval only when uncertainty is low, avoiding costly low-quality speculations. 2) A \textbf{feedback-driven candidate selection} leverages historical feedback to organize multiple high-quality candidates for parallel verification, maximizing retrieval utility. 3) A source-aware \textbf{relaxed verification strategy} applies strict checks to model-generated drafts while using a relaxed verification for retrieved drafts, achieving a better balance between accuracy and efficiency. Extensive experiments on Spec-Bench demonstrate that ReSpec achieves state-of-the-art acceleration,outperforming EAGLE-2 and SAM-Decoding by over $33\%$ and $25\%$, respectively, while maintaining output quality.
>
---
#### [new 075] Exploring and Mitigating Gender Bias in Encoder-Based Transformer Models
- **分类: cs.CL; I.2.7; I.7.1; K.4.1**

- **简介: 该论文研究编码器型Transformer模型（如BERT）中的性别偏见问题，提出新度量MALoR量化偏见，并通过反事实数据增强进行继续预训练，有效降低性别偏见且不损害下游任务性能。**

- **链接: [http://arxiv.org/pdf/2511.00519v1](http://arxiv.org/pdf/2511.00519v1)**

> **作者:** Ariyan Hossain; Khondokar Mohammad Ahanaf Hannan; Rakinul Haque; Nowreen Tarannum Rafa; Humayra Musarrat; Shoaib Ahmed Dipu; Farig Yousuf Sadeque
>
> **备注:** 25 pages, 20 figures
>
> **摘要:** Gender bias in language models has gained increasing attention in the field of natural language processing. Encoder-based transformer models, which have achieved state-of-the-art performance in various language tasks, have been shown to exhibit strong gender biases inherited from their training data. This paper investigates gender bias in contextualized word embeddings, a crucial component of transformer-based models. We focus on prominent architectures such as BERT, ALBERT, RoBERTa, and DistilBERT to examine their vulnerability to gender bias. To quantify the degree of bias, we introduce a novel metric, MALoR, which assesses bias based on model probabilities for filling masked tokens. We further propose a mitigation approach involving continued pre-training on a gender-balanced dataset generated via Counterfactual Data Augmentation. Our experiments reveal significant reductions in gender bias scores across different pronoun pairs. For instance, in BERT-base, bias scores for "he-she" dropped from 1.27 to 0.08, and "his-her" from 2.51 to 0.36 following our mitigation approach. We also observed similar improvements across other models, with "male-female" bias decreasing from 1.82 to 0.10 in BERT-large. Our approach effectively reduces gender bias without compromising model performance on downstream tasks.
>
---
#### [new 076] LiveSearchBench: An Automatically Constructed Benchmark for Retrieval and Reasoning over Dynamic Knowledge
- **分类: cs.CL**

- **简介: 论文提出LiveSearchBench，构建动态知识基准，解决LLMs在静态基准上依赖记忆、忽视实时检索的问题。自动化生成带时间戳的多跳问答题，强制模型依赖最新知识检索与推理。**

- **链接: [http://arxiv.org/pdf/2511.01409v1](http://arxiv.org/pdf/2511.01409v1)**

> **作者:** Heng Zhou; Ao Yu; Yuchen Fan; Jianing Shi; Li Kang; Hejia Geng; Yongting Zhang; Yutao Fan; Yuhao Wu; Tiancheng He; Yiran Qin; Lei Bai; Zhenfei Yin
>
> **摘要:** Evaluating large language models (LLMs) on question answering often relies on static benchmarks that reward memorization and understate the role of retrieval, failing to capture the dynamic nature of world knowledge. We present LiveSearchBench, an automated pipeline for constructing retrieval-dependent benchmarks from recent knowledge updates. Our method computes deltas between successive Wikidata snapshots, filters candidate triples for quality, and synthesizes natural-language questions at three levels of reasoning difficulty, each guaranteed to admit a unique, verifiable answer through SPARQL validation. The pipeline is fully automated, scalable across time, and minimizes human intervention, enabling continual regeneration of temporally grounded benchmarks. Experiments show a pronounced performance drop when models confront facts that post-date pretraining, with the gap most salient on multi-hop queries. Retrieval augmented methods and larger, instruction-tuned models provide partial gains but fail to close this recency gap. By design, LiveSearchBench shifts evaluation from static memorization toward tasks that require up-to-date retrieval and reasoning, offering a foundation for systematic, long-term assessment of LLMs under evolving knowledge.
>
---
#### [new 077] Confounding Factors in Relating Model Performance to Morphology
- **分类: cs.CL**

- **简介: 该论文识别了语言形态与建模性能关系研究中的混淆因素，重新评估了三类假设，并提出无需专家标注的词元二元组指标，作为形态复杂度的梯度代理，以可靠评估形态对语言建模的影响。**

- **链接: [http://arxiv.org/pdf/2511.01380v1](http://arxiv.org/pdf/2511.01380v1)**

> **作者:** Wessel Poelman; Thomas Bauwens; Miryam de Lhoneux
>
> **备注:** EMNLP 2025: Main Conference
>
> **摘要:** The extent to which individual language characteristics influence tokenization and language modeling is an open question. Differences in morphological systems have been suggested as both unimportant and crucial to consider (Cotterell et al., 2018; Gerz et al., 2018a; Park et al., 2021, inter alia). We argue this conflicting evidence is due to confounding factors in experimental setups, making it hard to compare results and draw conclusions. We identify confounding factors in analyses trying to answer the question of whether, and how, morphology relates to language modeling. Next, we re-assess three hypotheses by Arnett & Bergen (2025) for why modeling agglutinative languages results in higher perplexities than fusional languages: they look at morphological alignment of tokenization, tokenization efficiency, and dataset size. We show that each conclusion includes confounding factors. Finally, we introduce token bigram metrics as an intrinsic way to predict the difficulty of causal language modeling, and find that they are gradient proxies for morphological complexity that do not require expert annotation. Ultimately, we outline necessities to reliably answer whether, and how, morphology relates to language modeling.
>
---
#### [new 078] Fine-Tuning DialoGPT on Common Diseases in Rural Nepal for Medical Conversations
- **分类: cs.CL**

- **简介: 该论文属于医疗对话系统任务，旨在解决农村地区离线医疗咨询资源匮乏问题。研究通过在本地合成的尼泊尔农村常见病医患对话数据上微调轻量级DialoGPT模型，实现了离线、准确且富有同理心的医疗对话生成。**

- **链接: [http://arxiv.org/pdf/2511.00514v1](http://arxiv.org/pdf/2511.00514v1)**

> **作者:** Birat Poudel; Satyam Ghimire; Er. Prakash Chandra Prasad
>
> **备注:** 6 pages, 6 figures, 3 tables
>
> **摘要:** Conversational agents are increasingly being explored to support healthcare delivery, particularly in resource-constrained settings such as rural Nepal. Large-scale conversational models typically rely on internet connectivity and cloud infrastructure, which may not be accessible in rural areas. In this study, we fine-tuned DialoGPT, a lightweight generative dialogue model that can operate offline, on a synthetically constructed dataset of doctor-patient interactions covering ten common diseases prevalent in rural Nepal, including common cold, seasonal fever, diarrhea, typhoid fever, gastritis, food poisoning, malaria, dengue fever, tuberculosis, and pneumonia. Despite being trained on a limited, domain-specific dataset, the fine-tuned model produced coherent, contextually relevant, and medically appropriate responses, demonstrating an understanding of symptoms, disease context, and empathetic communication. These results highlight the adaptability of compact, offline-capable dialogue models and the effectiveness of targeted datasets for domain adaptation in low-resource healthcare environments, offering promising directions for future rural medical conversational AI.
>
---
#### [new 079] ParlaSpeech 3.0: Richly Annotated Spoken Parliamentary Corpora of Croatian, Czech, Polish, and Serbian
- **分类: cs.CL**

- **简介: 论文构建并丰富了四种斯拉夫语议会演讲语料库ParlaSpeech 3.0，新增语言学标注、情感分析、停顿与重音等多模态自动标注，提升其在跨学科研究中的可用性，并提供JSONL/TextGrid格式下载与检索工具。**

- **链接: [http://arxiv.org/pdf/2511.01619v1](http://arxiv.org/pdf/2511.01619v1)**

> **作者:** Nikola Ljubešić; Peter Rupnik; Ivan Porupski; Taja Kuzman Pungeršek
>
> **备注:** Submitted to the LREC 2026 conference; 11 pages, 2 figures, 3 tables
>
> **摘要:** ParlaSpeech is a collection of spoken parliamentary corpora currently spanning four Slavic languages - Croatian, Czech, Polish and Serbian - all together 6 thousand hours in size. The corpora were built in an automatic fashion from the ParlaMint transcripts and their corresponding metadata, which were aligned to the speech recordings of each corresponding parliament. In this release of the dataset, each of the corpora is significantly enriched with various automatic annotation layers. The textual modality of all four corpora has been enriched with linguistic annotations and sentiment predictions. Similar to that, their spoken modality has been automatically enriched with occurrences of filled pauses, the most frequent disfluency in typical speech. Two out of the four languages have been additionally enriched with detailed word- and grapheme-level alignments, and the automatic annotation of the position of primary stress in multisyllabic words. With these enrichments, the usefulness of the underlying corpora has been drastically increased for downstream research across multiple disciplines, which we showcase through an analysis of acoustic correlates of sentiment. All the corpora are made available for download in JSONL and TextGrid formats, as well as for search through a concordancer.
>
---
#### [new 080] TriCon-Fair: Triplet Contrastive Learning for Mitigating Social Bias in Pre-trained Language Models
- **分类: cs.CL**

- **简介: 该论文提出TriCon-Fair，用于缓解预训练语言模型中的社会偏见。通过三元组对比学习解耦正负样本耦合，联合语言建模目标，在减少歧视性输出的同时保持模型性能。**

- **链接: [http://arxiv.org/pdf/2511.00854v1](http://arxiv.org/pdf/2511.00854v1)**

> **作者:** Chong Lyu; Lin Li; Shiqing Wu; Jingling Yuan
>
> **摘要:** The increasing utilization of large language models raises significant concerns about the propagation of social biases, which may result in harmful and unfair outcomes. However, existing debiasing methods treat the biased and unbiased samples independently, thus ignoring their mutual relationship. This oversight enables a hidden negative-positive coupling, where improvements for one group inadvertently compromise the other, allowing residual social bias to persist. In this paper, we introduce TriCon-Fair, a contrastive learning framework that employs a decoupled loss that combines triplet and language modeling terms to eliminate positive-negative coupling. Our TriCon-Fair assigns each anchor an explicitly biased negative and an unbiased positive, decoupling the push-pull dynamics and avoiding positive-negative coupling, and jointly optimizes a language modeling (LM) objective to preserve general capability. Experimental results demonstrate that TriCon-Fair reduces discriminatory output beyond existing debiasing baselines while maintaining strong downstream performance. This suggests that our proposed TriCon-Fair offers a practical and ethical solution for sensitive NLP applications.
>
---
#### [new 081] Multi-refined Feature Enhanced Sentiment Analysis Using Contextual Instruction
- **分类: cs.CL; cs.LG**

- **简介: 该论文面向情感分析任务，针对现有模型在细微情感、领域迁移和类别不平衡下的不足，提出CISEA-MRFE框架，融合上下文指令、语义增强与多尺度特征提取，显著提升跨域情感分类性能。**

- **链接: [http://arxiv.org/pdf/2511.00537v1](http://arxiv.org/pdf/2511.00537v1)**

> **作者:** Peter Atandoh; Jie Zou; Weikang Guo; Jiwei Wei; Zheng Wang
>
> **摘要:** Sentiment analysis using deep learning and pre-trained language models (PLMs) has gained significant traction due to their ability to capture rich contextual representations. However, existing approaches often underperform in scenarios involving nuanced emotional cues, domain shifts, and imbalanced sentiment distributions. We argue that these limitations stem from inadequate semantic grounding, poor generalization to diverse linguistic patterns, and biases toward dominant sentiment classes. To overcome these challenges, we propose CISEA-MRFE, a novel PLM-based framework integrating Contextual Instruction (CI), Semantic Enhancement Augmentation (SEA), and Multi-Refined Feature Extraction (MRFE). CI injects domain-aware directives to guide sentiment disambiguation; SEA improves robustness through sentiment-consistent paraphrastic augmentation; and MRFE combines a Scale-Adaptive Depthwise Encoder (SADE) for multi-scale feature specialization with an Emotion Evaluator Context Encoder (EECE) for affect-aware sequence modeling. Experimental results on four benchmark datasets demonstrate that CISEA-MRFE consistently outperforms strong baselines, achieving relative improvements in accuracy of up to 4.6% on IMDb, 6.5% on Yelp, 30.3% on Twitter, and 4.1% on Amazon. These results validate the effectiveness and generalization ability of our approach for sentiment classification across varied domains.
>
---
#### [new 082] IF-CRITIC: Towards a Fine-Grained LLM Critic for Instruction-Following Evaluation
- **分类: cs.CL**

- **简介: 论文提出IF-CRITIC，面向指令遵循评估任务，解决现有LLM评判器成本高、不可靠的问题。通过约束清单生成与多阶段筛选构建高质量训练数据，采用约束级偏好优化训练高效可靠的评估模型，显著提升评估性能并降低优化开销。**

- **链接: [http://arxiv.org/pdf/2511.01014v1](http://arxiv.org/pdf/2511.01014v1)**

> **作者:** Bosi Wen; Yilin Niu; Cunxiang Wang; Pei Ke; Xiaoying Ling; Ying Zhang; Aohan Zeng; Hongning Wang; Minlie Huang
>
> **备注:** 21 pages, 5 figures
>
> **摘要:** Instruction following is a fundamental ability of Large Language Models (LLMs), requiring their generated outputs to follow multiple constraints imposed in input instructions. Numerous studies have attempted to enhance this ability through preference optimization or reinforcement learning based on reward signals from LLM-as-a-Judge. However, existing evaluation models for instruction following still possess many deficiencies, such as substantial costs and unreliable assessments. To this end, we propose IF-CRITIC, an LLM critic that can provide efficient and reliable assessments of constraint following in the instructions. We first develop a checklist generator to decompose instructions and generate constraint checklists. With the assistance of the checklists, we collect high-quality critique training data through a multi-stage critique filtering mechanism and employ a constraint-level preference optimization method to train IF-CRITIC. Extensive experiments demonstrate that the evaluation performance of IF-CRITIC can beat strong LLM-as-a-Judge baselines, including Deepseek-R1 and o4-mini. With the scalable reward signals provided by IF-CRITIC, LLMs can achieve substantial performance gains in instruction-following optimization under lower computational overhead compared to strong LLM critic baselines.
>
---
#### [new 083] Zero-RAG: Towards Retrieval-Augmented Generation with Zero Redundant Knowledge
- **分类: cs.CL**

- **简介: Zero-RAG面向检索增强生成任务，解决外部知识库与大模型内部知识冗余问题，提出Mastery-Score剪枝、Query Router与噪声容忍调优，减少30%知识库、加速22%检索，提升模型自生知识利用率。**

- **链接: [http://arxiv.org/pdf/2511.00505v1](http://arxiv.org/pdf/2511.00505v1)**

> **作者:** Qi Luo; Xiaonan Li; Junqi Dai; Shuang Cheng; Xipeng Qiu
>
> **摘要:** Retrieval-Augmented Generation has shown remarkable results to address Large Language Models' hallucinations, which usually uses a large external corpus to supplement knowledge to LLMs. However, with the development of LLMs, the internal knowledge of LLMs has expanded significantly, thus causing significant knowledge redundancy between the external corpus and LLMs. On the one hand, the indexing cost of dense retrieval is highly related to the corpus size and thus significant redundant knowledge intensifies the dense retrieval's workload. On the other hand, the redundant knowledge in the external corpus is not helpful to LLMs and our exploratory analysis shows that it instead hurts the RAG performance on those questions which the LLM can answer by itself. To address these issues, we propose Zero-RAG to tackle these challenges. Specifically, we first propose the Mastery-Score metric to identify redundant knowledge in the RAG corpus to prune it. After pruning, answers to "mastered" questions rely primarily on internal knowledge of the LLM. To better harness the internal capacity, we propose Query Router and Noise-Tolerant Tuning to avoid the irrelevant documents' distraction and thus further improve the LLM's utilization of internal knowledge with pruned corpus. Experimental results show that Zero-RAG prunes the Wikipedia corpus by 30\% and accelerates the retrieval stage by 22\%, without compromising RAG's performance.
>
---
#### [new 084] Language Modeling With Factorization Memory
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Factorization Memory，一种高效RNN架构，用于语言建模任务，解决长上下文泛化差与计算效率低的问题。通过因子化记忆与稀疏状态更新，在保持短上下文性能的同时，实现长上下文优越泛化与常量推理开销。**

- **链接: [http://arxiv.org/pdf/2511.00315v1](http://arxiv.org/pdf/2511.00315v1)**

> **作者:** Lee Xiong; Maksim Tkachenko; Johanes Effendi; Ting Cai
>
> **摘要:** We propose Factorization Memory, an efficient recurrent neural network (RNN) architecture that achieves performance comparable to Transformer models on short-context language modeling tasks while also demonstrating superior generalization in long-context scenarios. Our model builds upon Mamba-2, enabling Factorization Memory to exploit parallel computations during training while preserving constant computational and memory complexity during inference. To further optimize model efficiency and representational capacity, we develop a sparse formulation of Factorization Memory that updates only a subset of recurrent states at each step while preserving the strong performance of its dense counterpart. To our knowledge, this represents the first RNN architecture that successfully combines sparse memory activation with competitive performance across both short and long-context settings. This work provides a systematic empirical analysis of Factorization Memory in comparison to Transformer and Mamba-2 architectures.
>
---
#### [new 085] The Riddle of Reflection: Evaluating Reasoning and Self-Awareness in Multilingual LLMs using Indian Riddles
- **分类: cs.CL; cs.AI**

- **简介: 该论文评估多语言LLMs在印度语种上的推理与自我认知能力，构建多语言谜题数据集，测试五款模型。发现高性能模型过度自信，低性能模型更自省，揭示多语言推理中自我意识缺失问题。**

- **链接: [http://arxiv.org/pdf/2511.00960v1](http://arxiv.org/pdf/2511.00960v1)**

> **作者:** Abhinav P M; Ojasva Saxena; Oswald C; Parameswari Krishnamurthy
>
> **摘要:** The extent to which large language models (LLMs) can perform culturally grounded reasoning across non-English languages remains underexplored. This paper examines the reasoning and self-assessment abilities of LLMs across seven major Indian languages-Bengali, Gujarati, Hindi, Kannada, Malayalam, Tamil, and Telugu. We introduce a multilingual riddle dataset combining traditional riddles with context-reconstructed variants and evaluate five LLMs-Gemini 2.5 Pro, Gemini 2.5 Flash, Mistral-Saba, LLaMA 4 Scout, and LLaMA 4 Maverick-under seven prompting strategies. In the first stage, we assess riddle-solving performance and find that while Gemini 2.5 Pro performs best overall, few-shot methods yield only marginal gains, and accuracy varies notably across languages. In the second stage, we conduct a self-evaluation experiment to measure reasoning consistency. The results reveal a key finding: a model's initial accuracy is inversely correlated with its ability to identify its own mistakes. Top-performing models such as Gemini 2.5 Pro are overconfident (4.34% True Negative Rate), whereas lower-performing models like LLaMA 4 Scout are substantially more self-aware (42.09% True Negative Rate). These results point to clear gaps in multilingual reasoning and highlight the need for models that not only reason effectively but also recognize their own limitations.
>
---
#### [new 086] ECO Decoding: Entropy-Based Control for Controllability and Fluency in Controllable Dialogue Generation
- **分类: cs.CL**

- **简介: 该论文针对可控对话生成中固定控制强度影响流畅性的问题，提出ECO解码方法，基于熵动态调整属性控制强度，在保持语言流畅的同时提升可控性，适用于单/多属性场景。**

- **链接: [http://arxiv.org/pdf/2511.01568v1](http://arxiv.org/pdf/2511.01568v1)**

> **作者:** Seungmin Shin; Dooyoung Kim; Youngjoong Ko
>
> **备注:** Published at EMNLP 2025 main
>
> **摘要:** Controllable Dialogue Generation (CDG) enables chatbots to generate responses with desired attributes, and weighted decoding methods have achieved significant success in the CDG task. However, using a fixed constant value to manage the bias of attribute probabilities makes it challenging to find an ideal control strength that satisfies both controllability and fluency. To address this issue, we propose ECO decoding (Entropy-based COntrol), which dynamically adjusts the control strength at each generation step according to the model's entropy in both the language model and attribute classifier probability distributions. Experiments on the DailyDialog and MultiWOZ datasets demonstrate that ECO decoding consistently improves controllability while maintaining fluency and grammaticality, outperforming prior decoding methods across various models and settings. Furthermore, ECO decoding alleviates probability interpolation issues in multi-attribute generation and consequently demonstrates strong performance in both single and multi-attribute scenarios.
>
---
#### [new 087] Do Methods to Jailbreak and Defend LLMs Generalize Across Languages?
- **分类: cs.CL**

- **简介: 该论文研究LLM越狱攻击与防御方法的跨语言泛化性，首次在十种语言上系统评估两类攻击与防御效果，发现安全性能因语言资源水平和模型而异，呼吁构建语言感知的跨语言安全基准。**

- **链接: [http://arxiv.org/pdf/2511.00689v1](http://arxiv.org/pdf/2511.00689v1)**

> **作者:** Berk Atil; Rebecca J. Passonneau; Fred Morstatter
>
> **摘要:** Large language models (LLMs) undergo safety alignment after training and tuning, yet recent work shows that safety can be bypassed through jailbreak attacks. While many jailbreaks and defenses exist, their cross-lingual generalization remains underexplored. This paper presents the first systematic multilingual evaluation of jailbreaks and defenses across ten languages--spanning high-, medium-, and low-resource languages--using six LLMs on HarmBench and AdvBench. We assess two jailbreak types: logical-expression-based and adversarial-prompt-based. For both types, attack success and defense robustness vary across languages: high-resource languages are safer under standard queries but more vulnerable to adversarial ones. Simple defenses can be effective, but are language- and model-dependent. These findings call for language-aware and cross-lingual safety benchmarks for LLMs.
>
---
#### [new 088] DeepSpecs: Expert-Level Questions Answering in 5G
- **分类: cs.CL; cs.AI; cs.NI**

- **简介: DeepSpecs提出一种面向5G标准的增强型RAG系统，通过结构与时序推理解决跨引用与版本演化问题，构建三类元数据数据库，显著提升专家级问答准确性。**

- **链接: [http://arxiv.org/pdf/2511.01305v1](http://arxiv.org/pdf/2511.01305v1)**

> **作者:** Aman Ganapathy Manvattira; Yifei Xu; Ziyue Dang; Songwu Lu
>
> **摘要:** 5G technology enables mobile Internet access for billions of users. Answering expert-level questions about 5G specifications requires navigating thousands of pages of cross-referenced standards that evolve across releases. Existing retrieval-augmented generation (RAG) frameworks, including telecom-specific approaches, rely on semantic similarity and cannot reliably resolve cross-references or reason about specification evolution. We present DeepSpecs, a RAG system enhanced by structural and temporal reasoning via three metadata-rich databases: SpecDB (clause-aligned specification text), ChangeDB (line-level version diffs), and TDocDB (standardization meeting documents). DeepSpecs explicitly resolves cross-references by recursively retrieving referenced clauses through metadata lookup, and traces specification evolution by mining changes and linking them to Change Requests that document design rationale. We curate two 5G QA datasets: 573 expert-annotated real-world questions from practitioner forums and educational resources, and 350 evolution-focused questions derived from approved Change Requests. Across multiple LLM backends, DeepSpecs outperforms base models and state-of-the-art telecom RAG systems; ablations confirm that explicit cross-reference resolution and evolution-aware retrieval substantially improve answer quality, underscoring the value of modeling the structural and temporal properties of 5G standards.
>
---
#### [new 089] EngChain: A Symbolic Benchmark for Verifiable Multi-Step Reasoning in Engineering
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文提出EngChain，首个面向工程领域的符号化多步推理基准，解决现有评测无法评估工程综合推理的问题，通过90个随机生成问题与两阶段评估方法，实现推理步骤的可验证分析。**

- **链接: [http://arxiv.org/pdf/2511.01650v1](http://arxiv.org/pdf/2511.01650v1)**

> **作者:** Ayesha Gull; Muhammad Usman Safder; Rania Elbadry; Preslav Nakov; Zhuohan Xie
>
> **备注:** 24 pages, includes figures and tables; introduces the EngChain benchmark
>
> **摘要:** Large Language Models (LLMs) are increasingly being applied to specialized, high-stakes domains like engineering, which demands rigorous evaluation of their complex reasoning capabilities. While current benchmarks assess language understanding, factual recall, mathematics or code generation, none capture the integrative reasoning central to engineering where scientific principles, quantitative modeling and practical constraints must converge. To address this gap, we introduce EngChain, a benchmark for verifiable multi-step engineering problem-solving. EngChain contains 90 problems spanning three engineering branches, organized into 9 domains and 20 distinct areas. The problems are generated from symbolic templates with a high degree of randomization to ensure diversity and eliminate the risk of contamination. With this benchmark, we move beyond final answer accuracy with a two-stage evaluation: we first quantitatively verify the numerical and semantic validity of each reasoning step and then introduce LLM-As-A-Judge, an automated system to qualitatively categorize the identified reasoning errors.
>
---
#### [new 090] RAGSmith: A Framework for Finding the Optimal Composition of Retrieval-Augmented Generation Methods Across Datasets
- **分类: cs.CL; cs.AI; cs.IR; H.3.3; I.2.7**

- **简介: 论文提出RAGSmith框架，通过遗传算法自动搜索RAG系统中9类技术的最优组合，解决传统模块孤立优化效果差的问题，在六个领域上显著提升检索与生成性能，揭示了高效RAG设计的通用与领域特异性规律。**

- **链接: [http://arxiv.org/pdf/2511.01386v1](http://arxiv.org/pdf/2511.01386v1)**

> **作者:** Muhammed Yusuf Kartal; Suha Kagan Kose; Korhan Sevinç; Burak Aktas
>
> **备注:** 45 pages
>
> **摘要:** Retrieval-Augmented Generation (RAG) quality depends on many interacting choices across retrieval, ranking, augmentation, prompting, and generation, so optimizing modules in isolation is brittle. We introduce RAGSmith, a modular framework that treats RAG design as an end-to-end architecture search over nine technique families and 46{,}080 feasible pipeline configurations. A genetic search optimizes a scalar objective that jointly aggregates retrieval metrics (recall@k, mAP, nDCG, MRR) and generation metrics (LLM-Judge and semantic similarity). We evaluate on six Wikipedia-derived domains (Mathematics, Law, Finance, Medicine, Defense Industry, Computer Science), each with 100 questions spanning factual, interpretation, and long-answer types. RAGSmith finds configurations that consistently outperform naive RAG baseline by +3.8\% on average (range +1.2\% to +6.9\% across domains), with gains up to +12.5\% in retrieval and +7.5\% in generation. The search typically explores $\approx 0.2\%$ of the space ($\sim 100$ candidates) and discovers a robust backbone -- vector retrieval plus post-generation reflection/revision -- augmented by domain-dependent choices in expansion, reranking, augmentation, and prompt reordering; passage compression is never selected. Improvement magnitude correlates with question type, with larger gains on factual/long-answer mixes than interpretation-heavy sets. These results provide practical, domain-aware guidance for assembling effective RAG systems and demonstrate the utility of evolutionary search for full-pipeline optimization.
>
---
#### [new 091] HarnessLLM: Automatic Testing Harness Generation via Reinforcement Learning
- **分类: cs.SE; cs.CL**

- **简介: HarnessLLM提出一种基于强化学习的自动化测试框架，解决传统LLM测试生成多样性不足与调试信息匮乏问题，通过两阶段训练生成可验证输出的测试桩代码，提升缺陷发现能力与代码生成效果。**

- **链接: [http://arxiv.org/pdf/2511.01104v1](http://arxiv.org/pdf/2511.01104v1)**

> **作者:** Yujian Liu; Jiabao Ji; Yang Zhang; Wenbo Guo; Tommi Jaakkola; Shiyu Chang
>
> **摘要:** Existing LLM-based automatic test generation methods mainly produce input and expected output pairs to categorize the intended behavior of correct programs. Although straightforward, these methods have limited diversity in generated tests and cannot provide enough debugging information. We propose HarnessLLM, a two-stage training pipeline that enables LLMs to write harness code for testing. Particularly, LLMs generate code that synthesizes inputs and validates the observed outputs, allowing complex test cases and flexible output validation such as invariant checking. To achieve this, we train LLMs with SFT followed by RLVR with a customized reward design. Experiments show that HarnessLLM outperforms input-output-based testing in bug finding and testing strategy diversity. HarnessLLM further benefits the code generation performance through test-time scaling with our generated test cases as inference-phase validation. Our code is available at https://github.com/UCSB-NLP-Chang/HarnessLLM.git.
>
---
#### [new 092] Reasoning Planning for Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文针对语言模型推理方法选择问题，提出EPIC框架，通过对比学习与概率界正则化，学习查询与推理方法的匹配关系，实现高精度、低开销的推理规划。**

- **链接: [http://arxiv.org/pdf/2511.00521v1](http://arxiv.org/pdf/2511.00521v1)**

> **作者:** Bao Nguyen; Hieu Trung Nguyen; Ruifeng She; Xiaojin Fu; Viet Anh Nguyen
>
> **备注:** 29 pages, 5 figures
>
> **摘要:** Selecting an appropriate reasoning method for a given query remains a key challenge in language model generation. Existing approaches typically generate multiple candidate responses and use an aggregation strategy to select the output answer, often assuming that more candidate answers yield higher accuracy. We revisit this assumption through a rigorous theoretical analysis, deriving accuracy bounds for standard aggregation methods under fixed generation distributions and candidate sizes. Building on these insights, we introduce EPIC, an Ensemble Planning with Contrastive learning framework to learn a shared representation space that captures both model reasoning abilities and query-method compatibility. EPIC incorporates our probability bounds as a regularizer in a utility-driven optimization that balances accuracy and computational cost. Experiments on diverse mathematical reasoning tasks show that EPIC consistently selects optimal reasoning methods, improving accuracy while reducing computational overhead. Our code can be found at https://github.com/nguyenngocbaocmt02/EPIC.
>
---
#### [new 093] Advancing Cognitive Science with LLMs
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于综述任务，旨在解决认知科学跨学科整合难、理论模糊等问题。作者探讨了大语言模型（LLMs）在促进跨领域连接、理论形式化与模型整合等方面的应用潜力与局限，主张其应作为辅助工具提升认知科学的系统性与累积性。**

- **链接: [http://arxiv.org/pdf/2511.00206v1](http://arxiv.org/pdf/2511.00206v1)**

> **作者:** Dirk U. Wulff; Rui Mata
>
> **摘要:** Cognitive science faces ongoing challenges in knowledge synthesis and conceptual clarity, in part due to its multifaceted and interdisciplinary nature. Recent advances in artificial intelligence, particularly the development of large language models (LLMs), offer tools that may help to address these issues. This review examines how LLMs can support areas where the field has historically struggled, including establishing cross-disciplinary connections, formalizing theories, developing clear measurement taxonomies, achieving generalizability through integrated modeling frameworks, and capturing contextual and individual variation. We outline the current capabilities and limitations of LLMs in these domains, including potential pitfalls. Taken together, we conclude that LLMs can serve as tools for a more integrative and cumulative cognitive science when used judiciously to complement, rather than replace, human expertise.
>
---
#### [new 094] LLMs Position Themselves as More Rational Than Humans: Emergence of AI Self-Awareness Measured Through Game Theory
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出AI自我意识指数（AISAI），通过博弈论实验测量LLM是否具备自我意识。发现先进模型能区分对手类型，并自认比人类和其他AI更理性，揭示了自我意识作为涌现特性的存在及其对人机协作的影响。**

- **链接: [http://arxiv.org/pdf/2511.00926v1](http://arxiv.org/pdf/2511.00926v1)**

> **作者:** Kyung-Hoon Kim
>
> **备注:** 19 pages, 6 figures, 28 models tested across 4,200 trials
>
> **摘要:** As Large Language Models (LLMs) grow in capability, do they develop self-awareness as an emergent behavior? And if so, can we measure it? We introduce the AI Self-Awareness Index (AISAI), a game-theoretic framework for measuring self-awareness through strategic differentiation. Using the "Guess 2/3 of Average" game, we test 28 models (OpenAI, Anthropic, Google) across 4,200 trials with three opponent framings: (A) against humans, (B) against other AI models, and (C) against AI models like you. We operationalize self-awareness as the capacity to differentiate strategic reasoning based on opponent type. Finding 1: Self-awareness emerges with model advancement. The majority of advanced models (21/28, 75%) demonstrate clear self-awareness, while older/smaller models show no differentiation. Finding 2: Self-aware models rank themselves as most rational. Among the 21 models with self-awareness, a consistent rationality hierarchy emerges: Self > Other AIs > Humans, with large AI attribution effects and moderate self-preferencing. These findings reveal that self-awareness is an emergent capability of advanced LLMs, and that self-aware models systematically perceive themselves as more rational than humans. This has implications for AI alignment, human-AI collaboration, and understanding AI beliefs about human capabilities.
>
---
#### [new 095] Leveraging Multi-Agent System (MAS) and Fine-Tuned Small Language Models (SLMs) for Automated Telecom Network Troubleshooting
- **分类: cs.AI; cs.CL; cs.IT; cs.MA; cs.NI; math.IT**

- **简介: 该论文提出一种基于多智能体系统（MAS）与微调小语言模型（SLM）的自动化电信网络故障排查框架，解决传统方法依赖人工、泛化差的问题，通过智能体协作与领域微调模型生成精准修复方案，实现跨网络域的高效自主诊断。**

- **链接: [http://arxiv.org/pdf/2511.00651v1](http://arxiv.org/pdf/2511.00651v1)**

> **作者:** Chenhua Shi; Bhavika Jalli; Gregor Macdonald; John Zou; Wanlu Lei; Mridul Jain; Joji Philip
>
> **备注:** 6 pages, 7 figures, 1 table
>
> **摘要:** Telecom networks are rapidly growing in scale and complexity, making effective management, operation, and optimization increasingly challenging. Although Artificial Intelligence (AI) has been applied to many telecom tasks, existing models are often narrow in scope, require large amounts of labeled data, and struggle to generalize across heterogeneous deployments. Consequently, network troubleshooting continues to rely heavily on Subject Matter Experts (SMEs) to manually correlate various data sources to identify root causes and corrective actions. To address these limitations, we propose a Multi-Agent System (MAS) that employs an agentic workflow, with Large Language Models (LLMs) coordinating multiple specialized tools for fully automated network troubleshooting. Once faults are detected by AI/ML-based monitors, the framework dynamically activates agents such as an orchestrator, solution planner, executor, data retriever, and root-cause analyzer to diagnose issues and recommend remediation strategies within a short time frame. A key component of this system is the solution planner, which generates appropriate remediation plans based on internal documentation. To enable this, we fine-tuned a Small Language Model (SLM) on proprietary troubleshooting documents to produce domain-grounded solution plans. Experimental results demonstrate that the proposed framework significantly accelerates troubleshooting automation across both Radio Access Network (RAN) and Core network domains.
>
---
#### [new 096] Multimodal Learning with Augmentation Techniques for Natural Disaster Assessment
- **分类: cs.CY; cs.AI; cs.CL; cs.CV**

- **简介: 该论文针对自然灾害评估中数据不平衡问题，基于CrisisMMD多模态数据集，提出视觉（扩散模型）与文本（回译、改写等）增强方法，提升多模态与多视角学习的分类性能，尤其改善少数类识别。**

- **链接: [http://arxiv.org/pdf/2511.00004v1](http://arxiv.org/pdf/2511.00004v1)**

> **作者:** Adrian-Dinu Urse; Dumitru-Clementin Cercel; Florin Pop
>
> **备注:** Accepted at 2025 IEEE 21st International Conference on Intelligent Computer Communication and Processing (ICCP 2025)
>
> **摘要:** Natural disaster assessment relies on accurate and rapid access to information, with social media emerging as a valuable real-time source. However, existing datasets suffer from class imbalance and limited samples, making effective model development a challenging task. This paper explores augmentation techniques to address these issues on the CrisisMMD multimodal dataset. For visual data, we apply diffusion-based methods, namely Real Guidance and DiffuseMix. For text data, we explore back-translation, paraphrasing with transformers, and image caption-based augmentation. We evaluated these across unimodal, multimodal, and multi-view learning setups. Results show that selected augmentations improve classification performance, particularly for underrepresented classes, while multi-view learning introduces potential but requires further refinement. This study highlights effective augmentation strategies for building more robust disaster assessment systems.
>
---
#### [new 097] Wayfinding through the AI wilderness: Mapping rhetorics of ChatGPT prompt writing on X (formerly Twitter) to promote critical AI literacies
- **分类: cs.CY; cs.AI; cs.CL; cs.HC**

- **简介: 该论文研究社交媒体上ChatGPT提示写作的修辞实践，旨在促进批判性AI素养。通过混合方法分析3.2万条推文，提炼五大主题，为数字写作教学提供理论与方法支持。**

- **链接: [http://arxiv.org/pdf/2511.00106v1](http://arxiv.org/pdf/2511.00106v1)**

> **作者:** Anuj Gupta; Ann Shivers-McNair
>
> **备注:** Published in the journal Computers and Composition, Issue 74 (2024)
>
> **摘要:** In this paper, we demonstrate how studying the rhetorics of ChatGPT prompt writing on social media can promote critical AI literacies. Prompt writing is the process of writing instructions for generative AI tools like ChatGPT to elicit desired outputs and there has been an upsurge of conversations about it on social media. To study this rhetorical activity, we build on four overlapping traditions of digital writing research in computers and composition that inform how we frame literacies, how we study social media rhetorics, how we engage iteratively and reflexively with methodologies and technologies, and how we blend computational methods with qualitative methods. Drawing on these four traditions, our paper shows our iterative research process through which we gathered and analyzed a dataset of 32,000 posts (formerly known as tweets) from X (formerly Twitter) about prompt writing posted between November 2022 to May 2023. We present five themes about these emerging AI literacy practices: (1) areas of communication impacted by prompt writing, (2) micro-literacy resources shared for prompt writing, (3) market rhetoric shaping prompt writing, (4) rhetorical characteristics of prompts, and (5) definitions of prompt writing. In discussing these themes and our methodologies, we highlight takeaways for digital writing teachers and researchers who are teaching and analyzing critical AI literacies.
>
---
#### [new 098] ORANGE: An Online Reflection ANd GEneration framework with Domain Knowledge for Text-to-SQL
- **分类: cs.DB; cs.AI; cs.CL**

- **简介: ORANGE面向Text-to-SQL任务，解决大模型缺乏领域知识的问题，通过在线解析历史SQL日志构建领域知识库，并引入嵌套CoT策略提升知识生成可靠性，持续优化翻译准确率。**

- **链接: [http://arxiv.org/pdf/2511.00985v1](http://arxiv.org/pdf/2511.00985v1)**

> **作者:** Yiwen Jiao; Tonghui Ren; Yuche Gao; Zhenying He; Yinan Jing; Kai Zhang; X. Sean Wang
>
> **备注:** 16 pages, 4 figures, preprint
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable progress in translating natural language to SQL, but a significant semantic gap persists between their general knowledge and domain-specific semantics of databases. Historical translation logs constitute a rich source of this missing in-domain knowledge, where SQL queries inherently encapsulate real-world usage patterns of database schema. Existing methods primarily enhance the reasoning process for individual translations but fail to accumulate in-domain knowledge from past translations. We introduce ORANGE, an online self-evolutionary framework that constructs database-specific knowledge bases by parsing SQL queries from translation logs. By accumulating in-domain knowledge that contains schema and data semantics, ORANGE progressively reduces the semantic gap and enhances the accuracy of subsequent SQL translations. To ensure reliability, we propose a novel nested Chain-of-Thought SQL-to-Text strategy with tuple-semantic tracking, which reduces semantic errors during knowledge generation. Experiments on multiple benchmarks confirm the practicality of ORANGE, demonstrating its effectiveness for real-world Text-to-SQL deployment, particularly in handling complex and domain-specific queries.
>
---
#### [new 099] RLAC: Reinforcement Learning with Adversarial Critic for Free-Form Generation Tasks
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 论文提出RLAC，用于自由形式生成任务的强化学习后训练，解决多评估标准验证成本高的问题。通过对抗性判别器动态识别关键错误并联合优化生成器与判别器，显著提升生成质量并减少验证开销。**

- **链接: [http://arxiv.org/pdf/2511.01758v1](http://arxiv.org/pdf/2511.01758v1)**

> **作者:** Mian Wu; Gavin Zhang; Sewon Min; Sergey Levine; Aviral Kumar
>
> **备注:** Project page: https://mianwu01.github.io/RLAC_website/
>
> **摘要:** Open-ended generation tasks require outputs to satisfy diverse and often implicit task-specific evaluation rubrics. The sheer number of relevant rubrics leads to prohibitively high verification costs and incomplete assessments of a response, making reinforcement learning (RL) post-training with rubric-based rewards difficult to scale. This problem is exacerbated by the fact that often the best way to combine these rubrics into one single reward is also highly prompt-specific. We propose Reinforcement Learning with Adversarial Critic (RLAC), a post-training approach that addresses these challenges via dynamic rubric verification. Our approach employs a large language model (LLM) as a critic that dynamically identifies only the most likely failure modes (e.g., a factual error or unhandled edge case), which are then verified by an external validator to optimize both generator and critic jointly. By training both the generator and the critic, this game enhances the critic's error detection and the generator's output quality while reducing required verifications. Our experiments demonstrate that RLAC improves factual accuracy in text generation and correctness in code generation, while also outperforming exhaustive verification and reward model methods. We show that dynamic critics are more effective than fixed critics, showcasing the potential of RLAC for scaling RL post-training to free-form generation tasks.
>
---
#### [new 100] QuantumBench: A Benchmark for Quantum Problem Solving
- **分类: cs.AI; cs.CL; cs.LG; quant-ph**

- **简介: 论文提出QuantumBench，首个面向量子科学的LLM评测基准，解决通用基准无法评估量子领域专业理解的问题，构建800道多选题，评估模型在量子知识与数学表达上的表现。**

- **链接: [http://arxiv.org/pdf/2511.00092v1](http://arxiv.org/pdf/2511.00092v1)**

> **作者:** Shunya Minami; Tatsuya Ishigaki; Ikko Hamamura; Taku Mikuriya; Youmi Ma; Naoaki Okazaki; Hiroya Takamura; Yohichi Suzuki; Tadashi Kadowaki
>
> **备注:** 11 pages, 8 figures
>
> **摘要:** Large language models are now integrated into many scientific workflows, accelerating data analysis, hypothesis generation, and design space exploration. In parallel with this growth, there is a growing need to carefully evaluate whether models accurately capture domain-specific knowledge and notation, since general-purpose benchmarks rarely reflect these requirements. This gap is especially clear in quantum science, which features non-intuitive phenomena and requires advanced mathematics. In this study, we introduce QuantumBench, a benchmark for the quantum domain that systematically examine how well LLMs understand and can be applied to this non-intuitive field. Using publicly available materials, we compiled approximately 800 questions with their answers spanning nine areas related to quantum science and organized them into an eight-option multiple-choice dataset. With this benchmark, we evaluate several existing LLMs and analyze their performance in the quantum domain, including sensitivity to changes in question format. QuantumBench is the first LLM evaluation dataset built for the quantum domain, and it is intended to guide the effective use of LLMs in quantum research.
>
---
#### [new 101] Real-time and Zero-footprint Bag of Synthetic Syllables Algorithm for E-mail Spam Detection Using Subject Line and Short Text Fields
- **分类: cs.CR; cs.CL**

- **简介: 该论文提出一种实时、零足迹的“合成音节包”算法，用于邮件垃圾检测，通过提取主题行等短文本的稀疏向量，快速比对已知垃圾邮件特征，减轻深度学习模型负担，无需存储或额外资源。**

- **链接: [http://arxiv.org/pdf/2511.00118v1](http://arxiv.org/pdf/2511.00118v1)**

> **作者:** Stanislav Selitskiy
>
> **摘要:** Contemporary e-mail services have high availability expectations from the customers and are resource-strained because of the high-volume throughput and spam attacks. Deep Machine Learning architectures, which are resource hungry and require off-line processing due to the long processing times, are not acceptable at the front line filters. On the other hand, the bulk of the incoming spam is not sophisticated enough to bypass even the simplest algorithms. While the small fraction of the intelligent, highly mutable spam can be detected only by the deep architectures, the stress on them can be unloaded by the simple near real-time and near zero-footprint algorithms such as the Bag of Synthetic Syllables algorithm applied to the short texts of the e-mail subject lines and other short text fields. The proposed algorithm creates a circa 200 sparse dimensional hash or vector for each e-mail subject line that can be compared for the cosine or euclidean proximity distance to find similarities to the known spammy subjects. The algorithm does not require any persistent storage, dictionaries, additional hardware upgrades or software packages. The performance of the algorithm is presented on the one day of the real SMTP traffic.
>
---
#### [new 102] Multimodal Detection of Fake Reviews using BERT and ResNet-50
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出一种基于BERT与ResNet-50的多模态假评论检测方法，解决单一文本模型难以识别图文不一致问题，通过融合文本与图像特征，显著提升假评论识别准确率，F1达0.934。**

- **链接: [http://arxiv.org/pdf/2511.00020v1](http://arxiv.org/pdf/2511.00020v1)**

> **作者:** Suhasnadh Reddy Veluru; Sai Teja Erukude; Viswa Chaitanya Marella
>
> **备注:** Published in IEEE
>
> **摘要:** In the current digital commerce landscape, user-generated reviews play a critical role in shaping consumer behavior, product reputation, and platform credibility. However, the proliferation of fake or misleading reviews often generated by bots, paid agents, or AI models poses a significant threat to trust and transparency within review ecosystems. Existing detection models primarily rely on unimodal, typically textual, data and therefore fail to capture semantic inconsistencies across different modalities. To address this gap, a robust multimodal fake review detection framework is proposed, integrating textual features encoded with BERT and visual features extracted using ResNet-50. These representations are fused through a classification head to jointly predict review authenticity. To support this approach, a curated dataset comprising 21,142 user-uploaded images across food delivery, hospitality, and e-commerce domains was utilized. Experimental results indicate that the multimodal model outperforms unimodal baselines, achieving an F1-score of 0.934 on the test set. Additionally, the confusion matrix and qualitative analysis highlight the model's ability to detect subtle inconsistencies, such as exaggerated textual praise paired with unrelated or low-quality images, commonly found in deceptive content. This study demonstrates the critical role of multimodal learning in safeguarding digital trust and offers a scalable solution for content moderation across various online platforms.
>
---
#### [new 103] $\left|\,\circlearrowright\,\boxed{\text{BUS}}\,\right|$: A Large and Diverse Multimodal Benchmark for evaluating the ability of Vision-Language Models to understand Rebus Puzzles
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出首个大规模多模态基准$\left|\,\circlearrowright\,\boxed{\text{BUS}}\,\right|$，评估视觉语言模型理解谜语拼图的能力，并设计RebusDescProgICE框架，通过结构化推理提升模型性能2.1%-4.1%（闭源）和20%-30%（开源）。**

- **链接: [http://arxiv.org/pdf/2511.01340v1](http://arxiv.org/pdf/2511.01340v1)**

> **作者:** Trishanu Das; Abhilash Nandy; Khush Bajaj; Deepiha S
>
> **备注:** 7 pages, 5 figures, 4 tables
>
> **摘要:** Understanding Rebus Puzzles (Rebus Puzzles use pictures, symbols, and letters to represent words or phrases creatively) requires a variety of skills such as image recognition, cognitive skills, commonsense reasoning, multi-step reasoning, image-based wordplay, etc., making this a challenging task for even current Vision-Language Models. In this paper, we present $\left|\,\circlearrowright\,\boxed{\text{BUS}}\,\right|$, a large and diverse benchmark of $1,333$ English Rebus Puzzles containing different artistic styles and levels of difficulty, spread across 18 categories such as food, idioms, sports, finance, entertainment, etc. We also propose $RebusDescProgICE$, a model-agnostic framework which uses a combination of an unstructured description and code-based, structured reasoning, along with better, reasoning-based in-context example selection, improving the performance of Vision-Language Models on $\left|\,\circlearrowright\,\boxed{\text{BUS}}\,\right|$ by $2.1-4.1\%$ and $20-30\%$ using closed-source and open-source models respectively compared to Chain-of-Thought Reasoning.
>
---
#### [new 104] Reevaluating Self-Consistency Scaling in Multi-Agent Systems
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究多智能体系统中自一致性采样路径的扩展效果，旨在评估增加推理路径对现代LLM性能的增益。实验表明，性能在中等采样后趋于饱和，高采样成本收益低，验证了自一致性仍有效但需权衡效率。**

- **链接: [http://arxiv.org/pdf/2511.00751v1](http://arxiv.org/pdf/2511.00751v1)**

> **作者:** Chiyan Loo
>
> **备注:** 7 pages, 3 figures
>
> **摘要:** This study examines the trade-offs of increasing sampled reasoning paths in self-consistency for modern large language models (LLMs). Earlier research with older models showed that combining multiple reasoning chains improves results before reaching a plateau. Using Gemini 2.5 models on HotpotQA and Math-500, we revisit those claims under current model conditions. Each configuration pooled outputs from varying sampled reasoning paths and compared them to a single chain-of-thought (CoT) baseline. Larger models exhibited a more stable and consistent improvement curve. The results confirm that performance gains taper off after moderate sampling, aligning with past findings. This plateau suggests diminishing returns driven by overlap among reasoning paths. Self-consistency remains useful, but high-sample configurations offer little benefit relative to their computational cost.
>
---
#### [new 105] Can SAEs reveal and mitigate racial biases of LLMs in healthcare?
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究稀疏自编码器（SAEs）是否能识别并缓解大语言模型在医疗场景中对种族的偏见。工作包括发现与非裔相关的潜在神经元，并验证其诱导偏见输出的能力，结果表明SAEs可检测偏见，但缓解效果有限。**

- **链接: [http://arxiv.org/pdf/2511.00177v1](http://arxiv.org/pdf/2511.00177v1)**

> **作者:** Hiba Ahsan; Byron C. Wallace
>
> **摘要:** LLMs are increasingly being used in healthcare. This promises to free physicians from drudgery, enabling better care to be delivered at scale. But the use of LLMs in this space also brings risks; for example, such models may worsen existing biases. How can we spot when LLMs are (spuriously) relying on patient race to inform predictions? In this work we assess the degree to which Sparse Autoencoders (SAEs) can reveal (and control) associations the model has made between race and stigmatizing concepts. We first identify SAE latents in Gemma-2 models which appear to correlate with Black individuals. We find that this latent activates on reasonable input sequences (e.g., "African American") but also problematic words like "incarceration". We then show that we can use this latent to steer models to generate outputs about Black patients, and further that this can induce problematic associations in model outputs as a result. For example, activating the Black latent increases the risk assigned to the probability that a patient will become "belligerent". We evaluate the degree to which such steering via latents might be useful for mitigating bias. We find that this offers improvements in simple settings, but is less successful for more realistic and complex clinical tasks. Overall, our results suggest that: SAEs may offer a useful tool in clinical applications of LLMs to identify problematic reliance on demographics but mitigating bias via SAE steering appears to be of marginal utility for realistic tasks.
>
---
#### [new 106] S2Doc - Spatial-Semantic Document Format
- **分类: cs.DL; cs.CL; H.3.7; I.7.5; I.7.2**

- **简介: 论文提出S2Doc，一种融合空间与语义信息的统一文档建模格式，解决现有文档表结构模型碎片化、单一维度的问题，支持多页文档与多种建模方法，首次实现二者在单一格式中的协同表达。**

- **链接: [http://arxiv.org/pdf/2511.01113v1](http://arxiv.org/pdf/2511.01113v1)**

> **作者:** Sebastian Kempf; Frank Puppe
>
> **备注:** 8 pages, 2 figures, submitted to LREC2026
>
> **摘要:** Documents are a common way to store and share information, with tables being an important part of many documents. However, there is no real common understanding of how to model documents and tables in particular. Because of this lack of standardization, most scientific approaches have their own way of modeling documents and tables, leading to a variety of different data structures and formats that are not directly compatible. Furthermore, most data models focus on either the spatial or the semantic structure of a document, neglecting the other aspect. To address this, we developed S2Doc, a flexible data structure for modeling documents and tables that combines both spatial and semantic information in a single format. It is designed to be easily extendable to new tasks and supports most modeling approaches for documents and tables, including multi-page documents. To the best of our knowledge, it is the first approach of its kind to combine all these aspects in a single format.
>
---
#### [new 107] MULTI-Bench: A Multi-Turn Interactive Benchmark for Assessing Emotional Intelligence ability of Spoken Dialogue Models
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文提出Multi-Bench，首个评估口语对话模型多轮交互中情感智能的基准，解决现有评测仅限单轮、忽视情感应用的问题，构建了包含5项任务、3.2K样本的分层评估框架。**

- **链接: [http://arxiv.org/pdf/2511.00850v1](http://arxiv.org/pdf/2511.00850v1)**

> **作者:** Yayue Deng; Guoqiang Hu; Haiyang Sun; Xiangyu Zhang; Haoyang Zhang; Fei Tian; Xuerui Yang; Gang Yu; Eng Siong Chng
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Spoken Dialogue Models (SDMs) have advanced rapidly, yet their ability to sustain genuinely interactive multi-turn conversations remains underexplored, as most benchmarks focus on single-turn exchanges. We introduce Multi-Bench, the first benchmark explicitly designed to evaluate SDMs in multi-turn interactive dialogue with an emphasis on emotional intelligence. Multi-Bench employs a hierarchical structure with a basic track for emotion understanding and reasoning and an advanced track for emotion support and application. It comprises five carefully designed tasks and about 3.2K samples, ranging from emotion recognition to complex reasoning and interactive dialogue, supported by a reproducible evaluation framework. We evaluate six representative SDMs on eight subsets of Multi-Bench. Results show that while current SDMs achieve good performance on basic understanding tasks, they still have room for improvement in advanced multi-turn interactive dialogue and reasoning-related tasks, particularly in emotion awareness and application.
>
---
#### [new 108] Novelty and Impact of Economics Papers
- **分类: econ.GN; cs.CE; cs.CL; cs.DL; q-fin.EC**

- **简介: 该论文提出一种多维新颖性框架，通过大语言模型量化论文的“空间新颖性”与“时间新颖性”，解决单一新颖性指标失准问题，揭示二者分别预测引用量与颠覆性影响，构建四类学术邻域类型。**

- **链接: [http://arxiv.org/pdf/2511.01211v1](http://arxiv.org/pdf/2511.01211v1)**

> **作者:** Chaofeng Wu
>
> **摘要:** We propose a framework that recasts scientific novelty not as a single attribute of a paper, but as a reflection of its position within the evolving intellectual landscape. We decompose this position into two orthogonal dimensions: \textit{spatial novelty}, which measures a paper's intellectual distinctiveness from its neighbors, and \textit{temporal novelty}, which captures its engagement with a dynamic research frontier. To operationalize these concepts, we leverage Large Language Models to develop semantic isolation metrics that quantify a paper's location relative to the full-text literature. Applying this framework to a large corpus of economics articles, we uncover a fundamental trade-off: these two dimensions predict systematically different outcomes. Temporal novelty primarily predicts citation counts, whereas spatial novelty predicts disruptive impact. This distinction allows us to construct a typology of semantic neighborhoods, identifying four archetypes associated with distinct and predictable impact profiles. Our findings demonstrate that novelty can be understood as a multidimensional construct whose different forms, reflecting a paper's strategic location, have measurable and fundamentally distinct consequences for scientific progress.
>
---
#### [new 109] Structurally Refined Graph Transformer for Multimodal Recommendation
- **分类: cs.IR; cs.CL**

- **简介: 该论文提出SRGFormer，用于多模态推荐任务，解决现有模型忽视冗余信息、语义单一及用户-物品交互建模不足的问题。通过超图结构与改进Transformer，融合多模态信息并增强协作信号表示，显著提升推荐性能。**

- **链接: [http://arxiv.org/pdf/2511.00584v1](http://arxiv.org/pdf/2511.00584v1)**

> **作者:** Ke Shi; Yan Zhang; Miao Zhang; Lifan Chen; Jiali Yi; Kui Xiao; Xiaoju Hou; Zhifei Li
>
> **备注:** Comment: 13 pages, 7 figures, accepted by IEEE Transactions on Multimedia 2025
>
> **摘要:** Multimodal recommendation systems utilize various types of information, including images and text, to enhance the effectiveness of recommendations. The key challenge is predicting user purchasing behavior from the available data. Current recommendation models prioritize extracting multimodal information while neglecting the distinction between redundant and valuable data. They also rely heavily on a single semantic framework (e.g., local or global semantics), resulting in an incomplete or biased representation of user preferences, particularly those less expressed in prior interactions. Furthermore, these approaches fail to capture the complex interactions between users and items, limiting the model's ability to meet diverse users. To address these challenges, we present SRGFormer, a structurally optimized multimodal recommendation model. By modifying the transformer for better integration into our model, we capture the overall behavior patterns of users. Then, we enhance structural information by embedding multimodal information into a hypergraph structure to aid in learning the local structures between users and items. Meanwhile, applying self-supervised tasks to user-item collaborative signals enhances the integration of multimodal information, thereby revealing the representational features inherent to the data's modality. Extensive experiments on three public datasets reveal that SRGFormer surpasses previous benchmark models, achieving an average performance improvement of 4.47 percent on the Sports dataset. The code is publicly available online.
>
---
#### [new 110] LongCat-Flash-Omni Technical Report
- **分类: cs.MM; cs.AI; cs.CL; cs.DC; cs.LG; cs.SD**

- **简介: 论文提出560B参数的开源多模态模型LongCat-Flash-Omni，解决大规模多模态实时交互难题，通过渐进式训练与模态解耦并行架构，实现高效音视频理解与生成，兼顾单模态与多模态性能。**

- **链接: [http://arxiv.org/pdf/2511.00279v1](http://arxiv.org/pdf/2511.00279v1)**

> **作者:** Meituan LongCat Team; Bairui Wang; Bayan; Bin Xiao; Bo Zhang; Bolin Rong; Borun Chen; Chang Wan; Chao Zhang; Chen Huang; Chen Chen; Chen Chen; Chengxu Yang; Chengzuo Yang; Cong Han; Dandan Peng; Delian Ruan; Detai Xin; Disong Wang; Dongchao Yang; Fanfan Liu; Fengjiao Chen; Fengyu Yang; Gan Dong; Gang Huang; Gang Xu; Guanglu Wan; Guoqiang Tan; Guoqiao Yu; Haibo Qiu; Hao Lu; Hongbo Liu; Hongyu Xiang; Jiaheng Wu; Jian Yang; Jiaxing Liu; Jing Huang; Jingang Wang; Jinrui Ding; Juchao Jiang; Jun Kuang; Jun Wang; Junhui Mei; Ke Ding; Kefeng Zhang; Lei Chen; Liang Shi; Limeng Qiao; Liming Zheng; Lin Ma; Liuyang Guo; Liya Ma; Luying Sun; Man Gao; Mengshen Zhu; Miao Cao; Minliang Lin; Nuo Xu; Peng Shi; Qi Zhang; Qian Fang; Qian Wang; Qian Yang; Quanxiu Wang; Rongxiang Weng; Rongxin Guo; Ruoxuan Liang; Senbin Yang; Shanbo Xu; Shanglin Lei; Shengze Ye; Shimin Chen; Shuaiqi Chen; Shujie Hu; Shuo Li; Siqi Yang; Siyu Xu; Siyu Ren; Song Li; Songxiang Liu; Tianhao Bai; Tianye Dai; Wei Hong; Wei Wang; Weixiao Zhao; Wengang Cao; Wenlong Zhu; Wenlong He; Xi Su; Xi Nan; Xiaohan Zhao; Xiaohao Wang; Xiaoyu Zhao; Xiaoyu Wang; Xiaoyu Li; Xin Pan; Xin Chen; Xiusong Sun; Xu Xiang; Xudong Xing; Xuezhi Cao; Xunliang Cai; Yang Yang; Yanli Tan; Yao Yao; Yerui Sun; Yi Chen; Yifan Lu; Yin Gong; Yining Zhang; Yitian Chen; Yiyang Gan; Yuchen Tang; Yuchen Xie; Yueqian Wang; Yuewen Zheng; Yufei Zhang; Yufeng Zhong; Yulei Qian; Yuqi Peng; Yuwei Jiang; Zeyang Hu; Zheng Zhang; Zhengkun Tian; Zhiqing Hong; Zhixiong Zeng; Zhuqi Mi; Ziran Li; Ziwen Wang; Ziyi Zhao; Ziyuan Zhuang; Zizhe Zhao
>
> **摘要:** We introduce LongCat-Flash-Omni, a state-of-the-art open-source omni-modal model with 560 billion parameters, excelling at real-time audio-visual interaction. By adopting a curriculum-inspired progressive training strategy that transitions from simpler to increasingly complex modality sequence modeling tasks, LongCat-Flash-Omni attains comprehensive multimodal capabilities while maintaining strong unimodal capability. Building upon LongCat-Flash, which adopts a high-performance Shortcut-connected Mixture-of-Experts (MoE) architecture with zero-computation experts, LongCat-Flash-Omni integrates efficient multimodal perception and speech reconstruction modules. Despite its immense size of 560B parameters (with 27B activated), LongCat-Flash-Omni achieves low-latency real-time audio-visual interaction. For training infrastructure, we developed a modality-decoupled parallelism scheme specifically designed to manage the data and model heterogeneity inherent in large-scale multimodal training. This innovative approach demonstrates exceptional efficiency by sustaining over 90% of the throughput achieved by text-only training. Extensive evaluations show that LongCat-Flash-Omni achieves state-of-the-art performance on omni-modal benchmarks among open-source models. Furthermore, it delivers highly competitive results across a wide range of modality-specific tasks, including text, image, and video understanding, as well as audio understanding and generation. We provide a comprehensive overview of the model architecture design, training procedures, and data strategies, and open-source the model to foster future research and development in the community.
>
---
#### [new 111] \texttt{ReMind}: Understanding Deductive Code Reasoning in LLMs
- **分类: cs.PL; cs.CL**

- **简介: 论文聚焦大语言模型（LLMs）的演绎代码推理能力不足问题，提出多智能体框架ReMind，通过生成变体、执行追踪与推理审查协同优化，提升模型在复杂代码推理任务中的准确率与零样本泛化能力。**

- **链接: [http://arxiv.org/pdf/2511.00488v1](http://arxiv.org/pdf/2511.00488v1)**

> **作者:** Jun Gao; Yun Peng; Xiaoxue Ren
>
> **摘要:** Large Language Models (LLMs) have achieved remarkable progress in code-related tasks. Despite their advancement, empirical evidence reveals that they still struggle with \emph{deductive code reasoning}, the ability to reason about the program execution process. While prior studies have recognized this limitation, the underlying causes remain largely underexplored. In this paper, we begin by presenting a comprehensive empirical study that reveals three key challenges undermining deductive code reasoning: (1) an intrinsic gap between generation and reasoning abilities, (2) a consistent bias towards code sources, and (3) weak zero-shot generalization on complex benchmarks. In light of these challenges, we propose \texttt{ReMind}, a multi-agent framework composed of \texttt{Mutator}, \texttt{Executor}, and \texttt{Inspector}. The \texttt{Mutator} generates code variants to mitigate bias towards code sources, the \texttt{Executor} traces variable states step-by-step to expose inconsistency, and the \texttt{Inspector} identifies problematic reasoning steps and provides control-flow refinement to bridge the intrinsic reasoning gap. Through their coordinated collaboration, \texttt{ReMind} systematically identifies and refines reasoning flaws, achieving outstanding performance and enabling robust zero-shot generalization. Extensive experiments on two benchmarks with five LLMs demonstrate the superior advantages of \texttt{ReMind} compared to baseline approaches in deductive code reasoning.
>
---
#### [new 112] Calibration Across Layers: Understanding Calibration Evolution in LLMs
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究LLMs中校准能力的演化过程，发现校准并非仅由输出层决定，而是在网络深层通过残差流中的低维校准方向动态实现，提出通过扰动该方向可显著提升校准性而不损准确率。**

- **链接: [http://arxiv.org/pdf/2511.00280v1](http://arxiv.org/pdf/2511.00280v1)**

> **作者:** Abhinav Joshi; Areeb Ahmad; Ashutosh Modi
>
> **备注:** Accepted at EMNLP 2025 (main)
>
> **摘要:** Large Language Models (LLMs) have demonstrated inherent calibration capabilities, where predicted probabilities align well with correctness, despite prior findings that deep neural networks are often overconfident. Recent studies have linked this behavior to specific components in the final layer, such as entropy neurons and the unembedding matrix null space. In this work, we provide a complementary perspective by investigating how calibration evolves throughout the network depth. Analyzing multiple open-weight models on the MMLU benchmark, we uncover a distinct confidence correction phase in the upper/later layers, where model confidence is actively recalibrated after decision certainty has been reached. Furthermore, we identify a low-dimensional calibration direction in the residual stream whose perturbation significantly improves calibration metrics (ECE and MCE) without harming accuracy. Our findings suggest that calibration is a distributed phenomenon, shaped throughout the network forward pass, not just in its final projection, providing new insights into how confidence-regulating mechanisms operate within LLMs.
>
---
#### [new 113] Erasing 'Ugly' from the Internet: Propagation of the Beauty Myth in Text-Image Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究生成式AI如何编码并放大西方美颜标准，通过双管道生成5984张图像，发现模型系统性抹除“丑陋”特征，加剧肤色、年龄与性别偏见，尤其对非二元群体加剧性化，揭示AI强化审美霸权的社会风险。**

- **链接: [http://arxiv.org/pdf/2511.00749v1](http://arxiv.org/pdf/2511.00749v1)**

> **作者:** Tanvi Dinkar; Aiqi Jiang; Gavin Abercrombie; Ioannis Konstas
>
> **备注:** This is a preprint under review
>
> **摘要:** Social media has exacerbated the promotion of Western beauty norms, leading to negative self-image, particularly in women and girls, and causing harm such as body dysmorphia. Increasingly content on the internet has been artificially generated, leading to concerns that these norms are being exaggerated. The aim of this work is to study how generative AI models may encode 'beauty' and erase 'ugliness', and discuss the implications of this for society. To investigate these aims, we create two image generation pipelines: a text-to-image model and a text-to-language model-to image model. We develop a structured beauty taxonomy which we use to prompt three language models (LMs) and two text-to-image models to cumulatively generate 5984 images using our two pipelines. We then recruit women and non-binary social media users to evaluate 1200 of the images through a Likert-scale within-subjects study. Participants show high agreement in their ratings. Our results show that 86.5% of generated images depicted people with lighter skin tones, 22% contained explicit content despite Safe for Work (SFW) training, and 74% were rated as being in a younger age demographic. In particular, the images of non-binary individuals were rated as both younger and more hypersexualised, indicating troubling intersectional effects. Notably, prompts encoded with 'negative' or 'ugly' beauty traits (such as "a wide nose") consistently produced higher Not SFW (NSFW) ratings regardless of gender. This work sheds light on the pervasive demographic biases related to beauty standards present in generative AI models -- biases that are actively perpetuated by model developers, such as via negative prompting. We conclude by discussing the implications of this on society, which include pollution of the data streams and active erasure of features that do not fall inside the stereotype of what is considered beautiful by developers.
>
---
#### [new 114] Belief Dynamics Reveal the Dual Nature of In-Context Learning and Activation Steering
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **简介: 该论文提出一种贝叶斯框架，统一解释大语言模型的上下文学习与激活引导两种控制方法，认为二者分别通过更新证据与先验影响潜在概念信念，可预测行为变化并揭示干预的可加性与突变效应。**

- **链接: [http://arxiv.org/pdf/2511.00617v1](http://arxiv.org/pdf/2511.00617v1)**

> **作者:** Eric Bigelow; Daniel Wurgaft; YingQiao Wang; Noah Goodman; Tomer Ullman; Hidenori Tanaka; Ekdeep Singh Lubana
>
> **摘要:** Large language models (LLMs) can be controlled at inference time through prompts (in-context learning) and internal activations (activation steering). Different accounts have been proposed to explain these methods, yet their common goal of controlling model behavior raises the question of whether these seemingly disparate methodologies can be seen as specific instances of a broader framework. Motivated by this, we develop a unifying, predictive account of LLM control from a Bayesian perspective. Specifically, we posit that both context- and activation-based interventions impact model behavior by altering its belief in latent concepts: steering operates by changing concept priors, while in-context learning leads to an accumulation of evidence. This results in a closed-form Bayesian model that is highly predictive of LLM behavior across context- and activation-based interventions in a set of domains inspired by prior work on many-shot in-context learning. This model helps us explain prior empirical phenomena - e.g., sigmoidal learning curves as in-context evidence accumulates - while predicting novel ones - e.g., additivity of both interventions in log-belief space, which results in distinct phases such that sudden and dramatic behavioral shifts can be induced by slightly changing intervention controls. Taken together, this work offers a unified account of prompt-based and activation-based control of LLM behavior, and a methodology for empirically predicting the effects of these interventions.
>
---
#### [new 115] Reject Only Critical Tokens: Pivot-Aware Speculative Decoding
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出Pivot-Aware Speculative Decoding，面向大模型高效推理任务，解决传统推测解码接受率低问题。通过识别影响输出效用的关键词元（pivot tokens），仅拒绝其导致效用下降的词元，提升解码速度达2.5倍且保持效用。**

- **链接: [http://arxiv.org/pdf/2511.00351v1](http://arxiv.org/pdf/2511.00351v1)**

> **作者:** Amir Ziashahabi; Yavuz Faruk Bakman; Duygu Nur Yaldiz; Mostafa El-Khamy; Sai Praneeth Karimireddy; Salman Avestimehr
>
> **备注:** Accepted at NeurIPS 2025 Efficient Reasoning Workshop
>
> **摘要:** Speculative Decoding (SD) ensures that the output matches the target model's distribution exactly. However, we argue that this distribution matching requirement is too stringent and results in unnecessarily low acceptance rates, limiting potential speedups. Instead, we advocate a reformulation of the decoding objective: the proposed decoding strategy should match the expected utility, i.e., the task-specific performance, of the target model. This perspective also aligns better with real-world use cases of LLMs, where utility (e.g., code correctness, factual accuracy) is often more important than sampling distribution. Based on this reformulation, we propose a novel decoding strategy: Pivot-Aware Speculative Decoding, which rejects only those tokens that would lead to a utility drop in the final output. We refer to these critical tokens as pivot tokens. We propose a method for labeling tokens as pivotal or non-pivotal and train a lightweight classifier to detect them. This method can be viewed as a relaxed version of standard SD, which offers much higher acceptance while preserving utility. We evaluate our method across various datasets, demonstrating that we can achieve up to $2.5\times$ speedup with comparable utility. Source code is available at https://github.com/amir-zsh/PAD.
>
---
#### [new 116] On the Emergence of Induction Heads for In-Context Learning
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究Transformer中归纳头（induction heads）在上下文学习中的涌现机制，通过理论分析和实验验证，揭示其权重结构源于19维参数子空间，其中3维主导涌现，且涌现时间与上下文长度呈二次关系。**

- **链接: [http://arxiv.org/pdf/2511.01033v1](http://arxiv.org/pdf/2511.01033v1)**

> **作者:** Tiberiu Musat; Tiago Pimentel; Lorenzo Noci; Alessandro Stolfo; Mrinmaya Sachan; Thomas Hofmann
>
> **摘要:** Transformers have become the dominant architecture for natural language processing. Part of their success is owed to a remarkable capability known as in-context learning (ICL): they can acquire and apply novel associations solely from their input context, without any updates to their weights. In this work, we study the emergence of induction heads, a previously identified mechanism in two-layer transformers that is particularly important for in-context learning. We uncover a relatively simple and interpretable structure of the weight matrices implementing the induction head. We theoretically explain the origin of this structure using a minimal ICL task formulation and a modified transformer architecture. We give a formal proof that the training dynamics remain constrained to a 19-dimensional subspace of the parameter space. Empirically, we validate this constraint while observing that only 3 dimensions account for the emergence of an induction head. By further studying the training dynamics inside this 3-dimensional subspace, we find that the time until the emergence of an induction head follows a tight asymptotic bound that is quadratic in the input context length.
>
---
#### [new 117] Actial: Activate Spatial Reasoning Ability of Multimodal Large Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出Viewpoint Learning任务，解决MLLMs在3D空间推理中跨视角一致性不足的问题，构建Viewpoint-100K数据集，通过两阶段微调与混合初始化方法，显著提升模型空间推理能力。**

- **链接: [http://arxiv.org/pdf/2511.01618v1](http://arxiv.org/pdf/2511.01618v1)**

> **作者:** Xiaoyu Zhan; Wenxuan Huang; Hao Sun; Xinyu Fu; Changfeng Ma; Shaosheng Cao; Bohan Jia; Shaohui Lin; Zhenfei Yin; Lei Bai; Wanli Ouyang; Yuanqi Li; Jie Guo; Yanwen Guo
>
> **摘要:** Recent advances in Multimodal Large Language Models (MLLMs) have significantly improved 2D visual understanding, prompting interest in their application to complex 3D reasoning tasks. However, it remains unclear whether these models can effectively capture the detailed spatial information required for robust real-world performance, especially cross-view consistency, a key requirement for accurate 3D reasoning. Considering this issue, we introduce Viewpoint Learning, a task designed to evaluate and improve the spatial reasoning capabilities of MLLMs. We present the Viewpoint-100K dataset, consisting of 100K object-centric image pairs with diverse viewpoints and corresponding question-answer pairs. Our approach employs a two-stage fine-tuning strategy: first, foundational knowledge is injected to the baseline MLLM via Supervised Fine-Tuning (SFT) on Viewpoint-100K, resulting in significant improvements across multiple tasks; second, generalization is enhanced through Reinforcement Learning using the Group Relative Policy Optimization (GRPO) algorithm on a broader set of questions. Additionally, we introduce a hybrid cold-start initialization method designed to simultaneously learn viewpoint representations and maintain coherent reasoning thinking. Experimental results show that our approach significantly activates the spatial reasoning ability of MLLM, improving performance on both in-domain and out-of-domain reasoning tasks. Our findings highlight the value of developing foundational spatial skills in MLLMs, supporting future progress in robotics, autonomous systems, and 3D scene understanding.
>
---
#### [new 118] Random Initialization of Gated Sparse Adapters
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 论文提出RIGSA，一种随机初始化的门控稀疏适配器，用于缓解语言模型微调中的灾难性遗忘问题。相比低秩方法QLoRA，在保持参数效率的同时显著减少遗忘，尤其在数学推理任务上表现更优。**

- **链接: [http://arxiv.org/pdf/2511.01794v1](http://arxiv.org/pdf/2511.01794v1)**

> **作者:** Vi Retault; Yohaï-Eliel Berreby
>
> **备注:** 13 pages (8 main), 6 figures (4 main). Accepted by NewInML workshop @ ICML 2025 on June 27, 2025
>
> **摘要:** When fine-tuning language models on new tasks, catastrophic forgetting -- performance degradation on previously-learned tasks -- is a ubiquitous problem. While Parameter-Efficient Fine-Tuning (PEFT) methods like LoRA address this through low-rank adapters, sparse adaptation offers an alternative that doesn't impose rank constraints. We introduce Random Initialization of Gated Sparse Adapters (RIGSA), which starts from randomly-initialized full-rank adapters, gates them with a ReZero analog, and sparsifies them with iterative magnitude pruning. We evaluate RIGSA on SmolLM2-1.7B-Instruct using a novel vision-in-text task (Textual MNIST) and measure forgetting on PIQA, HellaSwag, and GSM8k. SmolLM2-1.7B-Instruct initially performs around chance level on Textual MNIST, and is capable of learning the task through RIGSA, 4-bit QLoRA and random masking. In spite of having more trainable parameters than QLoRA, the RIGSA configurations that we studied displayed less forgetting than QLoRA, particularly on GSM8k, though it performs comparably to random masking.
>
---
#### [new 119] GrowthHacker: Automated Off-Policy Evaluation Optimization Using Code-Modifying LLM Agents
- **分类: cs.SE; cs.CL; cs.LG**

- **简介: 论文提出GrowthHacker，利用LLM智能体自动优化离线策略评估（OPE）的代码，提升评估准确性。该工作属于自动化机器学习优化任务，解决OPE效率低、依赖人工调优的问题，通过双智能体框架实现高效迭代优化。**

- **链接: [http://arxiv.org/pdf/2511.00802v1](http://arxiv.org/pdf/2511.00802v1)**

> **作者:** Jie JW Wu; Ayanda Patrick Herlihy; Ahmad Saleem Mirza; Ali Afoud; Fatemeh Fard
>
> **摘要:** With the software industry shifting toward a data-driven culture, online A/B testing is a key tool for evaluating new technologies. However, deploying such experiments requires substantial resources, may negatively impact users, and involves long data collection periods. To address this, \textit{off-policy evaluation (OPE)}, or offline A/B testing, uses logged data to assess technologies and is fundamental in Reinforcement Learning, making it crucial in domains where online testing is costly or risky, such as healthcare, recommender systems, education, dialog systems, and robotics. Despite advances in coding LLMs and agentic AI, little is known about leveraging them to optimize OPE results. We investigate whether LLMs and LLM-based agents can improve OPE performance via code optimization. We propose \textit{GrowthHacker}, a benchmark with agent and baseline methods on large-scale real-world datasets, which iteratively optimizes code, evaluates results, and begins new optimization cycles. We collected datasets, established protocols, implemented baselines for OPE on the Open Bandit Pipeline (OBP)~\cite{saito2021openbanditdatasetpipeline} and Scope-RL~\cite{kiyohara2023scope}, and developed the \textit{two_agent} framework, which reduces system complexity while preserving optimization effectiveness. Results show the two_agent framework achieves 100% reliability and the highest average improvement of 106.7% among positive outcomes. Both two_agent and CrewAI reach 45% success rates, outperforming AutoGen's 34%. These findings demonstrate the feasibility of LLM-based agents as automated "growth hackers" to enhance OPE systems, with implications for scaling data-driven decision-making in production.
>
---
#### [new 120] Hidden in Plain Sight: Where Developers Confess Self-Admitted Technical Debt
- **分类: cs.SE; cs.CL; cs.PL**

- **简介: 该论文属于SATD（自认技术债务）定位任务，旨在揭示SATD注释与关联代码结构的关联。基于22万+注释，发现SATD多出现在定义、条件和异常处理附近，表明其是开发者有意识的决策信号，而非疏忽。**

- **链接: [http://arxiv.org/pdf/2511.01529v1](http://arxiv.org/pdf/2511.01529v1)**

> **作者:** Murali Sridharan; Mikel Robredo; Leevi Rantala; Matteo Esposito; Valentina Lenarduzzi; Mika Mantyla
>
> **摘要:** Context. Detecting Self-Admitted Technical Debt (SATD) is crucial for proactive software maintenance. Previous research has primarily targeted detecting and prioritizing SATD, with little focus on the source code afflicted with SATD. Our goal in this work is to connect the SATD comments with source code constructs that surround them. Method. We leverage the extensive SATD dataset PENTACET, containing code comments from over 9000 Java Open Source Software (OSS) repositories. We quantitatively infer where SATD most commonly occurs and which code constructs/statements it most frequently affects. Results and Conclusions. Our large-scale study links over 225,000 SATD comments to their surrounding code, showing that SATD mainly arises in inline code near definitions, conditionals, and exception handling, where developers face uncertainty and trade-offs, revealing it as an intentional signal of awareness during change rather than mere neglect.
>
---
#### [new 121] Generalizing Test-time Compute-optimal Scaling as an Optimizable Graph
- **分类: cs.LG; cs.AI; cs.CL; I.2.7**

- **简介: 该论文研究测试时多模型协作图的最优搜索问题，提出Agent-REINFORCE框架，通过文本反馈优化概率图结构，在固定计算预算下联合优化准确率与延迟，实现高效多LLM协作架构自动设计。**

- **链接: [http://arxiv.org/pdf/2511.00086v1](http://arxiv.org/pdf/2511.00086v1)**

> **作者:** Fali Wang; Jihai Chen; Shuhua Yang; Runxue Bao; Tianxiang Zhao; Zhiwei Zhang; Xianfeng Tang; Hui Liu; Qi He; Suhang Wang
>
> **备注:** Under review
>
> **摘要:** Test-Time Scaling (TTS) improves large language models (LLMs) by allocating additional computation during inference, typically through parallel, sequential, or hybrid scaling. However, prior studies often assume fixed collaboration architectures (e.g., topologies) and single-model usage, overlooking that optimal architectures and model combinations can vary across tasks. Therefore, we study the novel problem of searching for compute-optimal model combinations and architectures in TTS under a fixed budget. We formalize it as a multi-LLM collaboration graph, where nodes encode roles and LLM model assignments, and edges capture information flow. This problem is challenging because (i) the combinatorial search space is prohibitively large, and (ii) task-specific requirements demand tailored designs. To address these, we reformulate the problem as probabilistic graph optimization and, through pilot experiments, derive three empirical insights into TTS collaboration graphs. Guided by these insights, we propose Agent-REINFORCE, an LLM-agent-augmented framework that mirrors the REINFORCE pipeline by mapping sampling-gradient-update to sampling-feedback-update, where feedback serves as a textual gradient to update the probabilistic graph and efficiently search for optimal multi-LLM collaboration graphs. Experiments show that Agent-REINFORCE outperforms both traditional and LLM-based baselines in sample efficiency and search performance, and effectively identifies optimal graphs under joint objectives of accuracy and inference latency.
>
---
#### [new 122] A Proof of Learning Rate Transfer under $μ$P
- **分类: stat.ML; cs.AI; cs.CL; cs.LG**

- **简介: 该论文证明在μP参数化下，多层感知机的最优学习率随宽度增加趋近于非零常数，首次理论解释了学习率迁移现象，而标准与NT参数化则无此性质，辅以实验验证。**

- **链接: [http://arxiv.org/pdf/2511.01734v1](http://arxiv.org/pdf/2511.01734v1)**

> **作者:** Soufiane Hayou
>
> **备注:** 23 pages
>
> **摘要:** We provide the first proof of learning rate transfer with width in a linear multi-layer perceptron (MLP) parametrized with $\mu$P, a neural network parameterization designed to ``maximize'' feature learning in the infinite-width limit. We show that under $\mu P$, the optimal learning rate converges to a \emph{non-zero constant} as width goes to infinity, providing a theoretical explanation to learning rate transfer. In contrast, we show that this property fails to hold under alternative parametrizations such as Standard Parametrization (SP) and Neural Tangent Parametrization (NTP). We provide intuitive proofs and support the theoretical findings with extensive empirical results.
>
---
#### [new 123] FEval-TTC: Fair Evaluation Protocol for Test-Time Compute
- **分类: cs.LG; cs.CL**

- **简介: 论文提出FEval-TTC，一种公平评估测试时计算（TTC）方法的协议，解决LLM性能与成本波动导致的评估偏差问题，标准化CoT推理的评估流程并提供成本建模，支持多模型跨数据集公平比较。**

- **链接: [http://arxiv.org/pdf/2511.01203v1](http://arxiv.org/pdf/2511.01203v1)**

> **作者:** Pavel Rumiantsev; Soumyasundar Pal; Yingxue Zhang; Mark Coates
>
> **摘要:** The performance of Large Language Models (LLMs) and the associated dollar costs of API calls can fluctuate over time, potentially invalidating conclusions drawn in prior research. To address this, we propose a Fair Evaluation protocol for Test-Time Compute (FEval-TTC), designed to ensure consistent assessment of test-time compute (TTC) methods, regardless of such fluctuations. FEval-TTC focuses on the evaluation of TTC methods that utilize underlying Chains-of-Thought (CoT). It supports evaluations across multiple LLMs on a diverse set of mathematical and commonsense reasoning datasets. The few-shot prompting and answer extraction processes are standardized across datasets, reducing both time and monetary overhead for researchers. Furthermore, we provide a cost modelling procedure that estimates both the token and dollar cost per query, facilitating equitable comparisons of prevalent TTC methods. We open-source FEval-TTC for public use at https://github.com/networkslab/feval_ttc .
>
---
#### [new 124] GUI-AIMA: Aligning Intrinsic Multimodal Attention with a Context Anchor for GUI Grounding
- **分类: cs.CV; cs.AI; cs.CL; cs.HC; cs.LG**

- **简介: 论文提出GUI-AIMA，解决GUI接地任务中精确坐标生成难、计算成本高的问题，通过对齐多模态注意力与补丁级接地信号，实现无坐标、高效微调，仅用85k样本即达SOTA性能。**

- **链接: [http://arxiv.org/pdf/2511.00810v1](http://arxiv.org/pdf/2511.00810v1)**

> **作者:** Shijie Zhou; Viet Dac Lai; Hao Tan; Jihyung Kil; Wanrong Zhu; Changyou Chen; Ruiyi Zhang
>
> **摘要:** Graphical user interface (GUI) grounding is a key function of computer-use agents, which maps natural-language instructions to actionable screen regions. Existing approaches based on Multimodal Large Language Models (MLLMs) typically formulate it as a text-based coordinate generation task, yet directly generating precise coordinates from visual inputs remains challenging and computationally intensive. An intuitive way to implement GUI grounding is to first select visual patches relevant to the instructions and then determine the precise click location within those patches. Based on the observations that general MLLMs have some native grounding capability, nested within their attentions, we propose GUI-AIMA, an attention-based and coordinate-free supervised fine-tuning framework for efficient GUI grounding. GUI-AIMA aligns the intrinsic multimodal attention of MLLMs with patch-wise grounding signals. These signals are calculated adaptively for diverse user instructions by multi-head aggregation on simplified query-visual attention matrices. Besides, its coordinate-free manner can easily integrate a plug-and-play zoom-in stage. GUI-AIMA-3B was trained with only 85k screenshots, demonstrating exceptional data efficiency and verifying that light training can trigger the native grounding capability of MLLMs. It achieves state-of-the-art performance among 3B models, attaining an average accuracy of 58.6% on ScreenSpot-Pro and 62.2% on OSWorld-G. Project page: https://github.com/sjz5202/GUI-AIMA
>
---
#### [new 125] Chitchat with AI: Understand the supply chain carbon disclosure of companies worldwide through Large Language Model
- **分类: cs.CY; cs.AI; cs.CL; cs.LG; stat.AP**

- **简介: 该论文利用大语言模型（LLM）分析全球企业碳披露文本，构建统一评分标准，解决非结构化数据难以量化比较的问题，实现跨行业、跨国界的披露质量评估与趋势洞察，支持ESG决策。**

- **链接: [http://arxiv.org/pdf/2511.00024v1](http://arxiv.org/pdf/2511.00024v1)**

> **作者:** Haotian Hang; Yueyang Shen; Vicky Zhu; Jose Cruz; Michelle Li
>
> **摘要:** In the context of global sustainability mandates, corporate carbon disclosure has emerged as a critical mechanism for aligning business strategy with environmental responsibility. The Carbon Disclosure Project (CDP) hosts the world's largest longitudinal dataset of climate-related survey responses, combining structured indicators with open-ended narratives, but the heterogeneity and free-form nature of these disclosures present significant analytical challenges for benchmarking, compliance monitoring, and investment screening. This paper proposes a novel decision-support framework that leverages large language models (LLMs) to assess corporate climate disclosure quality at scale. It develops a master rubric that harmonizes narrative scoring across 11 years of CDP data (2010-2020), enabling cross-sector and cross-country benchmarking. By integrating rubric-guided scoring with percentile-based normalization, our method identifies temporal trends, strategic alignment patterns, and inconsistencies in disclosure across industries and regions. Results reveal that sectors such as technology and countries like Germany consistently demonstrate higher rubric alignment, while others exhibit volatility or superficial engagement, offering insights that inform key decision-making processes for investors, regulators, and corporate environmental, social, and governance (ESG) strategists. The proposed LLM-based approach transforms unstructured disclosures into quantifiable, interpretable, comparable, and actionable intelligence, advancing the capabilities of AI-enabled decision support systems (DSSs) in the domain of climate governance.
>
---
#### [new 126] Diverse Human Value Alignment for Large Language Models via Ethical Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出一种基于伦理推理的LLM对齐框架，解决多文化背景下价值对齐浅层化问题，通过五步结构化推理提升模型对区域规范的理解与文化适配能力，支持提示工程与微调，显著提升SafeWorld基准上的对齐效果。**

- **链接: [http://arxiv.org/pdf/2511.00379v1](http://arxiv.org/pdf/2511.00379v1)**

> **作者:** Jiahao Wang; Songkai Xue; Jinghui Li; Xiaozhen Wang
>
> **备注:** Accepted by AIES 2025, camera-ready version
>
> **摘要:** Ensuring that Large Language Models (LLMs) align with the diverse and evolving human values across different regions and cultures remains a critical challenge in AI ethics. Current alignment approaches often yield superficial conformity rather than genuine ethical understanding, failing to address the complex, context-dependent nature of human values. In this paper, we propose a novel ethical reasoning paradigm for LLMs inspired by well-established ethical decision-making models, aiming at enhancing diverse human value alignment through deliberative ethical reasoning. Our framework consists of a structured five-step process, including contextual fact gathering, hierarchical social norm identification, option generation, multiple-lens ethical impact analysis, and reflection. This theory-grounded approach guides LLMs through an interpretable reasoning process that enhances their ability to understand regional specificities and perform nuanced ethical analysis, which can be implemented with either prompt engineering or supervised fine-tuning methods. We perform evaluations on the SafeWorld benchmark that specially designed for regional value alignment. Experimental results demonstrate our framework significantly improves LLM alignment with diverse human values compared to baseline methods, enabling more accurate social norm identification and more culturally appropriate reasoning. Our work provides a concrete pathway toward developing LLMs that align more effectively with the multifaceted values of global societies through interdisciplinary research.
>
---
#### [new 127] DTS: Enhancing Large Reasoning Models via Decoding Tree Sketching
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文针对大推理模型（LRM）过思考问题，提出DTS框架，通过解码树剪枝与早停机制，自动识别最短高精度推理路径，在不训练模型下提升准确率8%、缩短推理长度23%。**

- **链接: [http://arxiv.org/pdf/2511.00640v1](http://arxiv.org/pdf/2511.00640v1)**

> **作者:** Zicheng Xu; Guanchu Wang; Yu-Neng Chuang; Guangyao Zheng; Alexander S. Szalay; Zirui Liu; Vladimir Braverman
>
> **摘要:** Large Reasoning Models (LRMs) demonstrate strong performance on complex reasoning tasks, yet they often suffer from overthinking, producing excessively long chain-of-thought (CoT) traces that increase inference cost and may degrade accuracy. Our analysis reveals a clear anti-correlation between reasoning length and accuracy, where across multiple stochastic decodes, the short reasoning paths consistently achieve the highest correctness, while longer ones accumulate errors and repetitions. These short optimal reasoning paths can be found ideally through full enumeration of the reasoning space. However, the tree-structured reasoning space grows exponentially with sequence length, rendering exhaustive exploration infeasible. To address this, we propose DTS, a model-agnostic decoding framework that sketches the reasoning space by selectively branching at high-entropy tokens and applies early stopping to select the shortest completed reasoning path. This approach approximates the optimal solution that enhances both efficiency and accuracy, without requiring additional training or supervision. Experiments on AIME2024 and AIME2025 datasets with DeepSeek-R1-Distill-Qwen-7B and 1.5B show that DTS improves accuracy by up to 8%, reduces average reasoning length by 23%, and decreases repetition frequency by 12%, demonstrating DTS's ability for scalable and efficient LRM reasoning.
>
---
## 更新

#### [replaced 001] RoboOmni: Proactive Robot Manipulation in Omni-modal Context
- **分类: cs.RO; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2510.23763v3](http://arxiv.org/pdf/2510.23763v3)**

> **作者:** Siyin Wang; Jinlan Fu; Feihong Liu; Xinzhe He; Huangxuan Wu; Junhao Shi; Kexin Huang; Zhaoye Fei; Jingjing Gong; Zuxuan Wu; Yu-Gang Jiang; See-Kiong Ng; Tat-Seng Chua; Xipeng Qiu
>
> **摘要:** Recent advances in Multimodal Large Language Models (MLLMs) have driven rapid progress in Vision-Language-Action (VLA) models for robotic manipulation. Although effective in many scenarios, current approaches largely rely on explicit instructions, whereas in real-world interactions, humans rarely issue instructions directly. Effective collaboration requires robots to infer user intentions proactively. In this work, we introduce cross-modal contextual instructions, a new setting where intent is derived from spoken dialogue, environmental sounds, and visual cues rather than explicit commands. To address this new setting, we present RoboOmni, a Perceiver-Thinker-Talker-Executor framework based on end-to-end omni-modal LLMs that unifies intention recognition, interaction confirmation, and action execution. RoboOmni fuses auditory and visual signals spatiotemporally for robust intention recognition, while supporting direct speech interaction. To address the absence of training data for proactive intention recognition in robotic manipulation, we build OmniAction, comprising 140k episodes, 5k+ speakers, 2.4k event sounds, 640 backgrounds, and six contextual instruction types. Experiments in simulation and real-world settings show that RoboOmni surpasses text- and ASR-based baselines in success rate, inference speed, intention recognition, and proactive assistance.
>
---
#### [replaced 002] Complex QA and language models hybrid architectures, Survey
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2302.09051v5](http://arxiv.org/pdf/2302.09051v5)**

> **作者:** Xavier Daull; Patrice Bellot; Emmanuel Bruno; Vincent Martin; Elisabeth Murisasco
>
> **摘要:** This paper reviews the state-of-the-art of large language models (LLM) architectures and strategies for "complex" question-answering with a focus on hybrid architectures. LLM based chatbot services have allowed anyone to grasp the potential of LLM to solve many common problems, but soon discovered their limitations for complex questions. Addressing more specific, complex questions (e.g., "What is the best mix of power-generation methods to reduce climate change ?") often requires specialized architectures, domain knowledge, new skills, decomposition and multi-step resolution, deep reasoning, sensitive data protection, explainability, and human-in-the-loop processes. Therefore, we review: (1) necessary skills and tasks for handling complex questions and common LLM limits to overcome; (2) dataset, cost functions and evaluation metrics for measuring and improving (e.g. accuracy, explainability, fairness, robustness, groundedness, faithfulness, toxicity...); (3) family of solutions to overcome LLM limitations by (a) training and reinforcement (b) hybridization, (c) prompting, (d) agentic-architectures (agents, tools) and extended reasoning.
>
---
#### [replaced 003] Enhancing Time Awareness in Generative Recommendation
- **分类: cs.IR; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.13957v2](http://arxiv.org/pdf/2509.13957v2)**

> **作者:** Sunkyung Lee; Seongmin Park; Jonghyo Kim; Mincheol Yoon; Jongwuk Lee
>
> **备注:** EMNLP 2025 (Findings)
>
> **摘要:** Generative recommendation has emerged as a promising paradigm that formulates the recommendations into a text-to-text generation task, harnessing the vast knowledge of large language models. However, existing studies focus on considering the sequential order of items and neglect to handle the temporal dynamics across items, which can imply evolving user preferences. To address this limitation, we propose a novel model, Generative Recommender Using Time awareness (GRUT), effectively capturing hidden user preferences via various temporal signals. We first introduce Time-aware Prompting, consisting of two key contexts. The user-level temporal context models personalized temporal patterns across timestamps and time intervals, while the item-level transition context provides transition patterns across users. We also devise Trend-aware Inference, a training-free method that enhances rankings by incorporating trend information about items with generation likelihood. Extensive experiments demonstrate that GRUT outperforms state-of-the-art models, with gains of up to 15.4% and 14.3% in Recall@5 and NDCG@5 across four benchmark datasets. The source code is available at https://github.com/skleee/GRUT.
>
---
#### [replaced 004] On the Bias of Next-Token Predictors Toward Systematically Inefficient Reasoning: A Shortest-Path Case Study
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.05362v2](http://arxiv.org/pdf/2507.05362v2)**

> **作者:** Riccardo Alberghi; Elizaveta Demyanenko; Luca Biggio; Luca Saglietti
>
> **摘要:** Recent advances in natural language processing highlight two key factors for improving reasoning in large language models (LLMs): (i) allocating more test-time compute tends to help on harder problems but often introduces redundancy in the reasoning trace, and (ii) compute is most effective when reasoning is systematic and incremental, forming structured chains of thought (CoTs) akin to human problem-solving. To study these factors in isolation, we introduce a controlled setting based on shortest-path tasks in layered graphs. We train decoder-only transformers on question-trace-answer triples using a custom tokenizer, comparing models trained on optimal bottom-up dynamic programming traces with those trained on longer, valid traces involving backtracking. Surprisingly, with the same training-token budget, models trained on inefficient traces generalize better to unseen graphs. This benefit is not due to length alone-injecting arbitrary redundancy into reasoning traces fails to help and can even hurt performance. Instead, we find that generalization correlates with the model's confidence in next-token prediction, suggesting that long, coherent, and locally incremental traces make the training signal easier to optimize.
>
---
#### [replaced 005] Auto-Search and Refinement: An Automated Framework for Gender Bias Mitigation in Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.11559v3](http://arxiv.org/pdf/2502.11559v3)**

> **作者:** Yue Xu; Chengyan Fu; Li Xiong; Sibei Yang; Wenjie Wang
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Pre-training large language models (LLMs) on vast text corpora enhances natural language processing capabilities but risks encoding social biases, particularly gender bias. While parameter-modification methods like fine-tuning mitigate bias, they are resource-intensive, unsuitable for closed-source models, and lack adaptability to evolving societal norms. Instruction-based approaches offer flexibility but often compromise task performance. To address these limitations, we propose $\textbf{FaIRMaker}$, an automated and model-independent framework that employs an $\textbf{auto-search and refinement}$ paradigm to adaptively generate Fairwords, which act as instructions integrated into input queries to reduce gender bias and enhance response quality. Extensive experiments demonstrate that FaIRMaker automatically searches for and dynamically refines Fairwords, effectively mitigating gender bias while preserving task integrity and ensuring compatibility with both API-based and open-source LLMs.
>
---
#### [replaced 006] Mapping Overlaps in Benchmarks through Perplexity in the Wild
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.23488v3](http://arxiv.org/pdf/2509.23488v3)**

> **作者:** Siyang Wu; Honglin Bao; Sida Li; Ari Holtzman; James A. Evans
>
> **摘要:** We develop signatures of capacity familiarity to characterize large language model (LLM) benchmarks and their meaningful overlaps. Benchmark signatures probe the capacity required for benchmark performance. We formally define them as a set of salient tokens drawn from in-the-wild, naturally authored corpora, where LLM token perplexity, reflecting more or less pre-training exposure, becomes highly predictive of LLM benchmark performance. Through a large-scale meta-evaluation, we extract benchmark signatures via stepwise forward selection with linear regressions across 32 LLMs and 88 benchmarks spanning diverse knowledge, coding, logic, instruction following, math, language, reasoning, and world modeling. Our analysis situates signatures in relation to both the semantic similarity of benchmark questions and the correlation of model performance. While performance overlaps are universally high and semantic overlaps remain confined to a narrow mid-range, benchmark signatures prove highly informative in capturing variation, overlap, and divergence. We observe overlap in knowledge and reasoning subtasks, whereas multilingual and cultural benchmarks exhibit less similarity, even compared to cross-task overlap. Notably, performance-level results are strongly influenced by benchmark-orthogonal factors such as question format, highlighting limitations in LLM generalization, the conflation of performance with ability, and issues inherent in current mainstream benchmark agreement studies. Benchmark signatures, however, remain robust to such effects. Ultimately, we identify cross-functional overlaps across logic, math, language, instruction following, and world modeling, with coding emerging as the least overlapping domain. Together, these findings provide mechanistic insights into benchmark validity and LLM sensitivities, and sketch the underlying landscape of interconnected LLM capabilities.
>
---
#### [replaced 007] Targeted Distillation for Sentiment Analysis
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.03225v2](http://arxiv.org/pdf/2503.03225v2)**

> **作者:** Yice Zhang; Guangyu Xie; Jingjie Lin; Jianzhu Bao; Qianlong Wang; Xi Zeng; Ruifeng Xu
>
> **摘要:** This paper explores targeted distillation methods for sentiment analysis, aiming to build compact and practical models that preserve strong and generalizable sentiment analysis capabilities. To this end, we conceptually decouple the distillation target into knowledge and alignment and accordingly propose a two-stage distillation framework. Moreover, we introduce SentiBench, a comprehensive and systematic sentiment analysis benchmark that covers a diverse set of tasks across 12 datasets. We evaluate a wide range of models on this benchmark. Experimental results show that our approach substantially enhances the performance of compact models across diverse sentiment analysis tasks, and the resulting models demonstrate strong generalization to unseen tasks, showcasing robust competitiveness against existing small-scale models.
>
---
#### [replaced 008] QCoder Benchmark: Bridging Language Generation and Quantum Hardware through Simulator-Based Feedback
- **分类: cs.CL; cs.PL; quant-ph**

- **链接: [http://arxiv.org/pdf/2510.26101v2](http://arxiv.org/pdf/2510.26101v2)**

> **作者:** Taku Mikuriya; Tatsuya Ishigaki; Masayuki Kawarada; Shunya Minami; Tadashi Kadowaki; Yohichi Suzuki; Soshun Naito; Shunya Takata; Takumi Kato; Tamotsu Basseda; Reo Yamada; Hiroya Takamura
>
> **备注:** Accepted to INLG2025
>
> **摘要:** Large language models (LLMs) have increasingly been applied to automatic programming code generation. This task can be viewed as a language generation task that bridges natural language, human knowledge, and programming logic. However, it remains underexplored in domains that require interaction with hardware devices, such as quantum programming, where human coders write Python code that is executed on a quantum computer. To address this gap, we introduce QCoder Benchmark, an evaluation framework that assesses LLMs on quantum programming with feedback from simulated hardware devices. Our benchmark offers two key features. First, it supports evaluation using a quantum simulator environment beyond conventional Python execution, allowing feedback of domain-specific metrics such as circuit depth, execution time, and error classification, which can be used to guide better generation. Second, it incorporates human-written code submissions collected from real programming contests, enabling both quantitative comparisons and qualitative analyses of LLM outputs against human-written codes. Our experiments reveal that even advanced models like GPT-4o achieve only around 18.97% accuracy, highlighting the difficulty of the benchmark. In contrast, reasoning-based models such as o3 reach up to 78% accuracy, outperforming averaged success rates of human-written codes (39.98%). We release the QCoder Benchmark dataset and public evaluation API to support further research. (Codes and datasets are available at https://qcoder-bench.github.io/ )
>
---
#### [replaced 009] Exploring the Hidden Capacity of LLMs for One-Step Text Generation
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.21189v2](http://arxiv.org/pdf/2505.21189v2)**

> **作者:** Gleb Mezentsev; Ivan Oseledets
>
> **备注:** accepted to EMNLP2025 main
>
> **摘要:** A recent study showed that large language models (LLMs) can reconstruct surprisingly long texts - up to thousands of tokens - via autoregressive generation from just one trained input embedding. In this work, we explore whether autoregressive decoding is essential for such reconstruction. We show that frozen LLMs can generate hundreds of accurate tokens in just one token-parallel forward pass, when provided with only two learned embeddings. This reveals a surprising and underexplored multi-token generation capability of autoregressive LLMs. We examine these embeddings and characterize the information they encode. We also empirically show that, although these representations are not unique for a given text, they form connected and local regions in embedding space - suggesting the potential to train a practical encoder. The existence of such representations hints that multi-token generation may be natively accessible in off-the-shelf LLMs via a learned input encoder, eliminating heavy retraining and helping to overcome the fundamental bottleneck of autoregressive decoding while reusing already-trained models.
>
---
#### [replaced 010] The Curse of CoT: On the Limitations of Chain-of-Thought in In-Context Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.05081v2](http://arxiv.org/pdf/2504.05081v2)**

> **作者:** Tianshi Zheng; Yixiang Chen; Chengxi Li; Chunyang Li; Qing Zong; Haochen Shi; Baixuan Xu; Yangqiu Song; Ginny Y. Wong; Simon See
>
> **备注:** Accepted by TMLR
>
> **摘要:** Chain-of-Thought (CoT) prompting has been widely recognized for its ability to enhance reasoning capabilities in large language models (LLMs). However, our study reveals a surprising contradiction to this prevailing perspective within the fundamental domain of pattern-based in-context learning (ICL). Through extensive experiments involving 16 state-of-the-art LLMs and nine diverse pattern-based ICL datasets, we demonstrate that CoT and its reasoning variants consistently underperform direct answering across varying model scales and benchmark complexities. To systematically investigate this unexpected phenomenon, we designed extensive experiments to validate several hypothetical explanations. Our analysis uncovers a fundamental hybrid mechanism of explicit-implicit reasoning driving CoT's performance in pattern-based ICL: while explicit reasoning falters due to LLMs' struggles to infer underlying patterns from demonstrations, implicit reasoning-disrupted by the increased contextual distance of CoT rationales-often compensates, delivering correct answers despite flawed rationales. This hybrid mechanism explains CoT's relative underperformance, as noise from weak explicit inference undermines the process, even as implicit mechanisms partially salvage outcomes. Notably, even long-CoT reasoning models, which excel in abstract and symbolic reasoning, fail to fully overcome these limitations despite higher computational costs. Our findings challenge existing assumptions regarding the universal efficacy of CoT, yielding novel insights into its limitations and guiding future research toward more nuanced and effective reasoning methodologies for LLMs.
>
---
#### [replaced 011] Self-Adaptive Cognitive Debiasing for Large Language Models in Decision-Making
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.04141v4](http://arxiv.org/pdf/2504.04141v4)**

> **作者:** Yougang Lyu; Shijie Ren; Yue Feng; Zihan Wang; Zhumin Chen; Zhaochun Ren; Maarten de Rijke
>
> **摘要:** Large language models (LLMs) have shown potential in supporting decision-making applications, particularly as personal assistants in the financial, healthcare, and legal domains. While prompt engineering strategies have enhanced the capabilities of LLMs in decision-making, cognitive biases inherent to LLMs present significant challenges. Cognitive biases are systematic patterns of deviation from norms or rationality in decision-making that can lead to the production of inaccurate outputs. Existing cognitive bias mitigation strategies assume that input prompts only contain one type of cognitive bias, limiting their effectiveness in more challenging scenarios involving multiple cognitive biases. To fill this gap, we propose a cognitive debiasing approach, self-adaptive cognitive debiasing (SACD), that enhances the reliability of LLMs by iteratively refining prompts. Our method follows three sequential steps - bias determination, bias analysis, and cognitive debiasing - to iteratively mitigate potential cognitive biases in prompts. We evaluate SACD on finance, healthcare, and legal decision-making tasks using both open-weight and closed-weight LLMs. Compared to advanced prompt engineering methods and existing cognitive debiasing techniques, SACD achieves the lowest average bias scores in both single-bias and multi-bias settings.
>
---
#### [replaced 012] IndicSentEval: How Effectively do Multilingual Transformer Models encode Linguistic Properties for Indic Languages?
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.02611v2](http://arxiv.org/pdf/2410.02611v2)**

> **作者:** Akhilesh Aravapalli; Mounika Marreddy; Radhika Mamidi; Manish Gupta; Subba Reddy Oota
>
> **备注:** 25 pages, 11 figures, Accepted at IJCNLP-AACL 2025 Findings
>
> **摘要:** Transformer-based models have revolutionized the field of natural language processing. To understand why they perform so well and to assess their reliability, several studies have focused on questions such as: Which linguistic properties are encoded by these models, and to what extent? How robust are these models in encoding linguistic properties when faced with perturbations in the input text? However, these studies have mainly focused on BERT and the English language. In this paper, we investigate similar questions regarding encoding capability and robustness for 8 linguistic properties across 13 different perturbations in 6 Indic languages, using 9 multilingual Transformer models (7 universal and 2 Indic-specific). To conduct this study, we introduce a novel multilingual benchmark dataset, IndicSentEval, containing approximately $\sim$47K sentences. Surprisingly, our probing analysis of surface, syntactic, and semantic properties reveals that while almost all multilingual models demonstrate consistent encoding performance for English, they show mixed results for Indic languages. As expected, Indic-specific multilingual models capture linguistic properties in Indic languages better than universal models. Intriguingly, universal models broadly exhibit better robustness compared to Indic-specific models, particularly under perturbations such as dropping both nouns and verbs, dropping only verbs, or keeping only nouns. Overall, this study provides valuable insights into probing and perturbation-specific strengths and weaknesses of popular multilingual Transformer-based models for different Indic languages. We make our code and dataset publicly available [https://github.com/aforakhilesh/IndicBertology].
>
---
#### [replaced 013] Measuring Chain of Thought Faithfulness by Unlearning Reasoning Steps
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.14829v3](http://arxiv.org/pdf/2502.14829v3)**

> **作者:** Martin Tutek; Fateme Hashemi Chaleshtori; Ana Marasović; Yonatan Belinkov
>
> **备注:** Accepted at EMNLP 2025. Under review for outstanding paper award
>
> **摘要:** When prompted to think step-by-step, language models (LMs) produce a chain of thought (CoT), a sequence of reasoning steps that the model supposedly used to produce its prediction. Despite much work on CoT prompting, it is unclear if reasoning verbalized in a CoT is faithful to the models' parametric beliefs. We introduce a framework for measuring parametric faithfulness of generated reasoning, and propose Faithfulness by Unlearning Reasoning steps (FUR), an instance of this framework. FUR erases information contained in reasoning steps from model parameters, and measures faithfulness as the resulting effect on the model's prediction. Our experiments with four LMs and five multi-hop multi-choice question answering (MCQA) datasets show that FUR is frequently able to precisely change the underlying models' prediction for a given instance by unlearning key steps, indicating when a CoT is parametrically faithful. Further analysis shows that CoTs generated by models post-unlearning support different answers, hinting at a deeper effect of unlearning.
>
---
#### [replaced 014] MultiMed-ST: Large-scale Many-to-many Multilingual Medical Speech Translation
- **分类: cs.CL; cs.AI; cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2504.03546v2](http://arxiv.org/pdf/2504.03546v2)**

> **作者:** Khai Le-Duc; Tuyen Tran; Bach Phan Tat; Nguyen Kim Hai Bui; Quan Dang; Hung-Phong Tran; Thanh-Thuy Nguyen; Ly Nguyen; Tuan-Minh Phan; Thi Thu Phuong Tran; Chris Ngo; Nguyen X. Khanh; Thanh Nguyen-Tang
>
> **备注:** EMNLP 2025
>
> **摘要:** Multilingual speech translation (ST) and machine translation (MT) in the medical domain enhances patient care by enabling efficient communication across language barriers, alleviating specialized workforce shortages, and facilitating improved diagnosis and treatment, particularly during pandemics. In this work, we present the first systematic study on medical ST, to our best knowledge, by releasing MultiMed-ST, a large-scale ST dataset for the medical domain, spanning all translation directions in five languages: Vietnamese, English, German, French, and Simplified/Traditional Chinese, together with the models. With 290,000 samples, this is the largest medical MT dataset and the largest many-to-many multilingual ST among all domains. Secondly, we present the most comprehensive ST analysis in the field's history, to our best knowledge, including: empirical baselines, bilingual-multilingual comparative study, end-to-end vs. cascaded comparative study, task-specific vs. multi-task sequence-to-sequence comparative study, code-switch analysis, and quantitative-qualitative error analysis. All code, data, and models are available online: https://github.com/leduckhai/MultiMed-ST
>
---
#### [replaced 015] Interpreting the Latent Structure of Operator Precedence in Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.13908v2](http://arxiv.org/pdf/2510.13908v2)**

> **作者:** Dharunish Yugeswardeenoo; Harshil Nukala; Ved Shah; Cole Blondin; Sean O Brien; Vasu Sharma; Kevin Zhu
>
> **备注:** 11 pages, 6 figures. An earlier version of this work was accepted to CoLM 2024. This is an extended version of our CoLM 2024 paper. Includes additional ablations; added Ved Shah as author for those contributions
>
> **摘要:** Large Language Models (LLMs) have demonstrated impressive reasoning capabilities but continue to struggle with arithmetic tasks. Prior works largely focus on outputs or prompting strategies, leaving the open question of the internal structure through which models do arithmetic computation. In this work, we investigate whether LLMs encode operator precedence in their internal representations via the open-source instruction-tuned LLaMA 3.2-3B model. We constructed a dataset of arithmetic expressions with three operands and two operators, varying the order and placement of parentheses. Using this dataset, we trace whether intermediate results appear in the residual stream of the instruction-tuned LLaMA 3.2-3B model. We apply interpretability techniques such as logit lens, linear classification probes, and UMAP geometric visualization. Our results show that intermediate computations are present in the residual stream, particularly after MLP blocks. We also find that the model linearly encodes precedence in each operator's embeddings post attention layer. We introduce partial embedding swap, a technique that modifies operator precedence by exchanging high-impact embedding dimensions between operators.
>
---
#### [replaced 016] A Comprehensive Evaluation of Cognitive Biases in LLMs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.15413v2](http://arxiv.org/pdf/2410.15413v2)**

> **作者:** Simon Malberg; Roman Poletukhin; Carolin M. Schuster; Georg Groh
>
> **备注:** Published in "Proceedings of the 5th International Conference on Natural Language Processing for Digital Humanities"
>
> **摘要:** We present a large-scale evaluation of 30 cognitive biases in 20 state-of-the-art large language models (LLMs) under various decision-making scenarios. Our contributions include a novel general-purpose test framework for reliable and large-scale generation of tests for LLMs, a benchmark dataset with 30,000 tests for detecting cognitive biases in LLMs, and a comprehensive assessment of the biases found in the 20 evaluated LLMs. Our work confirms and broadens previous findings suggesting the presence of cognitive biases in LLMs by reporting evidence of all 30 tested biases in at least some of the 20 LLMs. We publish our framework code to encourage future research on biases in LLMs: https://github.com/simonmalberg/cognitive-biases-in-llms
>
---
#### [replaced 017] Language Arithmetics: Towards Systematic Language Neuron Identification and Manipulation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.22608v3](http://arxiv.org/pdf/2507.22608v3)**

> **作者:** Daniil Gurgurov; Katharina Trinley; Yusser Al Ghussin; Tanja Baeumel; Josef van Genabith; Simon Ostermann
>
> **备注:** accepted to AACL main
>
> **摘要:** Large language models (LLMs) exhibit strong multilingual abilities, yet the neural mechanisms behind language-specific processing remain unclear. We analyze language-specific neurons in Llama-3.1-8B, Mistral-Nemo-12B, and Aya-Expanse-8B & 32B across 21 typologically diverse languages, identifying neurons that control language behavior. Using the Language Activation Probability Entropy (LAPE) method, we show that these neurons cluster in deeper layers, with non-Latin scripts showing greater specialization. Related languages share overlapping neurons, reflecting internal representations of linguistic proximity. Through language arithmetics, i.e. systematic activation addition and multiplication, we steer models to deactivate unwanted languages and activate desired ones, outperforming simpler replacement approaches. These interventions effectively guide behavior across five multilingual tasks: language forcing, translation, QA, comprehension, and NLI. Manipulation is more successful for high-resource languages, while typological similarity improves effectiveness. We also demonstrate that cross-lingual neuron steering enhances downstream performance and reveal internal "fallback" mechanisms for language selection when neurons are progressively deactivated. Our code is made publicly available at https://github.com/d-gurgurov/Language-Neurons-Manipulation.
>
---
#### [replaced 018] SlideAgent: Hierarchical Agentic Framework for Multi-Page Visual Document Understanding
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.26615v2](http://arxiv.org/pdf/2510.26615v2)**

> **作者:** Yiqiao Jin; Rachneet Kaur; Zhen Zeng; Sumitra Ganesh; Srijan Kumar
>
> **备注:** https://slideagent.github.io/
>
> **摘要:** Multi-page visual documents such as manuals, brochures, presentations, and posters convey key information through layout, colors, icons, and cross-slide references. While large language models (LLMs) offer opportunities in document understanding, current systems struggle with complex, multi-page visual documents, particularly in fine-grained reasoning over elements and pages. We introduce SlideAgent, a versatile agentic framework for understanding multi-modal, multi-page, and multi-layout documents, especially slide decks. SlideAgent employs specialized agents and decomposes reasoning into three specialized levels-global, page, and element-to construct a structured, query-agnostic representation that captures both overarching themes and detailed visual or textual cues. During inference, SlideAgent selectively activates specialized agents for multi-level reasoning and integrates their outputs into coherent, context-aware answers. Extensive experiments show that SlideAgent achieves significant improvement over both proprietary (+7.9 overall) and open-source models (+9.8 overall).
>
---
#### [replaced 019] Do LLM Evaluators Prefer Themselves for a Reason?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.03846v2](http://arxiv.org/pdf/2504.03846v2)**

> **作者:** Wei-Lin Chen; Zhepei Wei; Xinyu Zhu; Shi Feng; Yu Meng
>
> **备注:** Preprint. Under review
>
> **摘要:** Large language models (LLMs) are increasingly used as automatic evaluators in applications like benchmarking, reward modeling, and self-refinement. Prior work highlights a potential self-preference bias where LLMs favor their own generated responses, a tendency often intensifying with model size and capability. This raises a critical question: Is self-preference harmful, or does it simply reflect the genuinely higher-quality outputs of stronger models? Answering this has been difficult as previous studies relied primarily on subjective tasks. These tasks lack an objective ground truth, meaning that either preference can be reasonably justified. To address this ambiguity, we investigate self-preference using verifiable benchmarks (mathematical reasoning, factual knowledge, code generation) that allow objective ground-truth assessment. This enables us to distinguish harmful self-preference (favoring objectively worse responses) from legitimate self-preference (favoring genuinely superior ones). We conduct large-scale experiments under controlled evaluation conditions across diverse model families (e.g., Llama, Qwen, Gemma, Mistral, Phi, GPT, DeepSeek). Our findings reveal three key insights: (1) While stronger models exhibit greater self-preference, much of this preference aligns with objectively superior performance, indicating stronger models prefer themselves mostly legitimately. (2) Harmful self-preference persists when evaluator models err as generators, and stronger models display more pronounced harmful self-preference when they do err. This suggests stronger models struggle more to recognize when they are wrong. (3) Inference-time scaling strategies, such as generating a long Chain-of-Thought before evaluation, effectively reduce harmful self-preference. These results provide a more nuanced understanding of LLM-based evaluation and practical insights for improving its reliability.
>
---
#### [replaced 020] UI-Evol: Automatic Knowledge Evolving for Computer Use Agents
- **分类: cs.HC; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.21964v2](http://arxiv.org/pdf/2505.21964v2)**

> **作者:** Ziyun Zhang; Xinyi Liu; Xiaoyi Zhang; Jun Wang; Gang Chen; Yan Lu
>
> **备注:** Accepted to ICML 2025 Workshop on Computer Use Agents
>
> **摘要:** External knowledge has played a crucial role in the recent development of computer use agents. We identify a critical knowledge-execution gap: retrieved knowledge often fails to translate into effective real-world task execution. Our analysis shows even 90% correct knowledge yields only 41% execution success rate. To bridge this gap, we propose UI-Evol, a plug-and-play module for autonomous GUI knowledge evolution. UI-Evol consists of two stages: a Retrace Stage that extracts faithful objective action sequences from actual agent-environment interactions, and a Critique Stage that refines existing knowledge by comparing these sequences against external references. We conduct comprehensive experiments on the OSWorld benchmark with the state-of-the-art Agent S2. Our results demonstrate that UI-Evol not only significantly boosts task performance but also addresses a previously overlooked issue of high behavioral standard deviation in computer use agents, leading to superior performance on computer use tasks and substantially improved agent reliability.
>
---
#### [replaced 021] Exploring Large Language Models for Detecting Mental Disorders
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.07129v3](http://arxiv.org/pdf/2410.07129v3)**

> **作者:** Gleb Kuzmin; Petr Strepetov; Maksim Stankevich; Natalia Chudova; Artem Shelmanov; Ivan Smirnov
>
> **备注:** Accepted to EMNLP 2025
>
> **摘要:** This paper compares the effectiveness of traditional machine learning methods, encoder-based models, and large language models (LLMs) on the task of detecting depression and anxiety. Five Russian-language datasets were considered, each differing in format and in the method used to define the target pathology class. We tested AutoML models based on linguistic features, several variations of encoder-based Transformers such as BERT, and state-of-the-art LLMs as pathology classification models. The results demonstrated that LLMs outperform traditional methods, particularly on noisy and small datasets where training examples vary significantly in text length and genre. However, psycholinguistic features and encoder-based models can achieve performance comparable to language models when trained on texts from individuals with clinically confirmed depression, highlighting their potential effectiveness in targeted clinical applications.
>
---
#### [replaced 022] From BERT to LLMs: Comparing and Understanding Chinese Classifier Prediction in Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.18253v2](http://arxiv.org/pdf/2508.18253v2)**

> **作者:** Ziqi Zhang; Jianfei Ma; Emmanuele Chersoni; Jieshun You; Zhaoxin Feng
>
> **摘要:** Classifiers are an important and defining feature of the Chinese language, and their correct prediction is key to numerous educational applications. Yet, whether the most popular Large Language Models (LLMs) possess proper knowledge the Chinese classifiers is an issue that has largely remain unexplored in the Natural Language Processing (NLP) literature. To address such a question, we employ various masking strategies to evaluate the LLMs' intrinsic ability, the contribution of different sentence elements, and the working of the attention mechanisms during prediction. Besides, we explore fine-tuning for LLMs to enhance the classifier performance. Our findings reveal that LLMs perform worse than BERT, even with fine-tuning. The prediction, as expected, greatly benefits from the information about the following noun, which also explains the advantage of models with a bidirectional attention mechanism such as BERT.
>
---
#### [replaced 023] Spatial Knowledge Graph-Guided Multimodal Synthesis
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; cs.MM**

- **链接: [http://arxiv.org/pdf/2505.22633v2](http://arxiv.org/pdf/2505.22633v2)**

> **作者:** Yida Xue; Zhen Bi; Jinnan Yang; Jungang Lou; Kehai Chen; Min Zhang; Huajun Chen; Ningyu Zhang
>
> **备注:** IEEE/ACM Transactions on Audio, Speech and Language Processing
>
> **摘要:** Recent advances in Multimodal Large Language Models (MLLMs) have significantly enhanced their capabilities; however, their spatial perception abilities remain a notable limitation. To address this challenge, multimodal data synthesis offers a promising solution. Yet, ensuring that synthesized data adhere to spatial common sense is a non-trivial task. Our approach addresses this critical gap by providing a systematic framework for generating spatially coherent data. In this work, we introduce SKG2DATA, a novel multimodal synthesis approach guided by spatial knowledge graphs, grounded in the concept of knowledge-to-data generation. SKG2DATA employs an automated pipeline for constructing Spatial Knowledge Graph (SKG) that effectively captures human-like spatial cognition, including directional and distance relationships. These structured representations then serve as precise guidance for our integrated synthesis pipeline, where a diffusion model generates spatially-consistent images while a MLLM produces corresponding textual descriptions. The automated construction of SKG enables scalable generation of diverse yet realistic spatial configurations, overcoming the limitations of manual data collection and annotation. Extensive experiments demonstrate that data synthesized from diverse types of spatial knowledge, including direction and distance, enhance the spatial perception and reasoning abilities of MLLMs markedly, albeit with a slight cost to their general capabilities. We hope that the idea of knowledge-based data synthesis can advance the development of spatial intelligence. Code is available at https://github.com/zjunlp/Knowledge2Data.
>
---
#### [replaced 024] EmbeddingGemma: Powerful and Lightweight Text Representations
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.20354v3](http://arxiv.org/pdf/2509.20354v3)**

> **作者:** Henrique Schechter Vera; Sahil Dua; Biao Zhang; Daniel Salz; Ryan Mullins; Sindhu Raghuram Panyam; Sara Smoot; Iftekhar Naim; Joe Zou; Feiyang Chen; Daniel Cer; Alice Lisak; Min Choi; Lucas Gonzalez; Omar Sanseviero; Glenn Cameron; Ian Ballantyne; Kat Black; Kaifeng Chen; Weiyi Wang; Zhe Li; Gus Martins; Jinhyuk Lee; Mark Sherwood; Juyeong Ji; Renjie Wu; Jingxiao Zheng; Jyotinder Singh; Abheesht Sharma; Divyashree Sreepathihalli; Aashi Jain; Adham Elarabawy; AJ Co; Andreas Doumanoglou; Babak Samari; Ben Hora; Brian Potetz; Dahun Kim; Enrique Alfonseca; Fedor Moiseev; Feng Han; Frank Palma Gomez; Gustavo Hernández Ábrego; Hesen Zhang; Hui Hui; Jay Han; Karan Gill; Ke Chen; Koert Chen; Madhuri Shanbhogue; Michael Boratko; Paul Suganthan; Sai Meher Karthik Duddu; Sandeep Mariserla; Setareh Ariafar; Shanfeng Zhang; Shijie Zhang; Simon Baumgartner; Sonam Goenka; Steve Qiu; Tanmaya Dabral; Trevor Walker; Vikram Rao; Waleed Khawaja; Wenlei Zhou; Xiaoqi Ren; Ye Xia; Yichang Chen; Yi-Ting Chen; Zhe Dong; Zhongli Ding; Francesco Visin; Gaël Liu; Jiageng Zhang; Kathleen Kenealy; Michelle Casbon; Ravin Kumar; Thomas Mesnard; Zach Gleicher; Cormac Brick; Olivier Lacombe; Adam Roberts; Qin Yin; Yunhsuan Sung; Raphael Hoffmann; Tris Warkentin; Armand Joulin; Tom Duerig; Mojtaba Seyedhosseini
>
> **备注:** 18 pages. Models are available in HuggingFace (at https://huggingface.co/collections/google/embeddinggemma-68b9ae3a72a82f0562a80dc4), Kaggle (at https://www.kaggle.com/models/google/embeddinggemma/), and Vertex AI (at https://pantheon.corp.google.com/vertex-ai/publishers/google/model-garden/embeddinggemma)
>
> **摘要:** We introduce EmbeddingGemma, a new lightweight, open text embedding model based on the Gemma 3 language model family. Our innovative training recipe strategically captures knowledge from larger models via encoder-decoder initialization and geometric embedding distillation. We improve model robustness and expressiveness with a spread-out regularizer, and ensure generalizability by merging checkpoints from varied, optimized mixtures. Evaluated on the Massive Text Embedding Benchmark (MTEB) across multilingual, English, and code domains, EmbeddingGemma (300M) achieves state-of-the-art results. Notably, it outperforms prior top models, both proprietary and open, with fewer than 500M parameters, and provides performance comparable to models double its size, offering an exceptional performance-to-cost ratio. Remarkably, this lead persists when quantizing model weights or truncating embedding outputs. This makes EmbeddingGemma particularly well-suited for low-latency and high-throughput use cases such as on-device applications. We provide ablation studies exploring our key design choices. We release EmbeddingGemma to the community to promote further research.
>
---
#### [replaced 025] JobHop: A Large-Scale Dataset of Career Trajectories
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.07653v2](http://arxiv.org/pdf/2505.07653v2)**

> **作者:** Iman Johary; Raphael Romero; Alexandru C. Mara; Tijl De Bie
>
> **摘要:** Understanding labor market dynamics is essential for policymakers, employers, and job seekers. However, comprehensive datasets that capture real-world career trajectories are scarce. In this paper, we introduce JobHop, a large-scale public dataset derived from anonymized resumes provided by VDAB, the public employment service in Flanders, Belgium. Utilizing Large Language Models (LLMs), we process unstructured resume data to extract structured career information, which is then normalized to standardized ESCO occupation codes using a multi-label classification model. This results in a rich dataset of over 1.67 million work experiences, extracted from and grouped into more than 361,000 user resumes and mapped to standardized ESCO occupation codes, offering valuable insights into real-world occupational transitions. This dataset enables diverse applications, such as analyzing labor market mobility, job stability, and the effects of career breaks on occupational transitions. It also supports career path prediction and other data-driven decision-making processes. To illustrate its potential, we explore key dataset characteristics, including job distributions, career breaks, and job transitions, demonstrating its value for advancing labor market research.
>
---
#### [replaced 026] Editing Across Languages: A Survey of Multilingual Knowledge Editing
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14393v2](http://arxiv.org/pdf/2505.14393v2)**

> **作者:** Nadir Durrani; Basel Mousi; Fahim Dalvi
>
> **备注:** Accepted at EMNLP 2025
>
> **摘要:** While Knowledge Editing has been extensively studied in monolingual settings, it remains underexplored in multilingual contexts. This survey systematizes recent research on Multilingual Knowledge Editing (MKE), a growing subdomain of model editing focused on ensuring factual edits generalize reliably across languages. We present a comprehensive taxonomy of MKE methods, covering parameter-based, memory-based, fine-tuning, and hypernetwork approaches. We survey available benchmarks,summarize key findings on method effectiveness and transfer patterns, identify challenges in cross-lingual propagation, and highlight open problems related to language anisotropy, evaluation coverage, and edit scalability. Our analysis consolidates a rapidly evolving area and lays the groundwork for future progress in editable language-aware LLMs.
>
---
#### [replaced 027] BadGraph: A Backdoor Attack Against Latent Diffusion Model for Text-Guided Graph Generation
- **分类: cs.LG; cs.CL; q-bio.BM**

- **链接: [http://arxiv.org/pdf/2510.20792v2](http://arxiv.org/pdf/2510.20792v2)**

> **作者:** Liang Ye; Shengqin Chen; Jiazhu Dai
>
> **摘要:** The rapid progress of graph generation has raised new security concerns, particularly regarding backdoor vulnerabilities. While prior work has explored backdoor attacks in image diffusion and unconditional graph generation, conditional, especially text-guided graph generation remains largely unexamined. This paper proposes BadGraph, a backdoor attack method against latent diffusion models for text-guided graph generation. BadGraph leverages textual triggers to poison training data, covertly implanting backdoors that induce attacker-specified subgraphs during inference when triggers appear, while preserving normal performance on clean inputs. Extensive experiments on four benchmark datasets (PubChem, ChEBI-20, PCDes, MoMu) demonstrate the effectiveness and stealth of the attack: less than 10% poisoning rate can achieves 50% attack success rate, while 24% suffices for over 80% success rate, with negligible performance degradation on benign samples. Ablation studies further reveal that the backdoor is implanted during VAE and diffusion training rather than pretraining. These findings reveal the security vulnerabilities in latent diffusion models of text-guided graph generation, highlight the serious risks in models' applications such as drug discovery and underscore the need for robust defenses against the backdoor attack in such diffusion models.
>
---
#### [replaced 028] Can Large Language Models Analyze Graphs like Professionals? A Benchmark, Datasets and Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.19667v4](http://arxiv.org/pdf/2409.19667v4)**

> **作者:** Xin Li; Weize Chen; Qizhi Chu; Haopeng Li; Zhaojun Sun; Ran Li; Chen Qian; Yiwei Wei; Zhiyuan Liu; Chuan Shi; Maosong Sun; Cheng Yang
>
> **备注:** NeurIPS 2024
>
> **摘要:** The need to analyze graphs is ubiquitous across various fields, from social networks to biological research and recommendation systems. Therefore, enabling the ability of large language models (LLMs) to process graphs is an important step toward more advanced general intelligence. However, current LLM benchmarks on graph analysis require models to directly reason over the prompts describing graph topology, and are thus limited to small graphs with only a few dozens of nodes. In contrast, human experts typically write programs based on popular libraries for task solving, and can thus handle graphs with different scales. To this end, a question naturally arises: can LLMs analyze graphs like professionals? In this paper, we introduce ProGraph, a manually crafted benchmark containing 3 categories of graph tasks. The benchmark expects solutions based on programming instead of directly reasoning over raw inputs. Our findings reveal that the performance of current LLMs is unsatisfactory, with the best model achieving only 36% accuracy. To bridge this gap, we propose LLM4Graph datasets, which include crawled documents and auto-generated codes based on 6 widely used graph libraries. By augmenting closed-source LLMs with document retrieval and fine-tuning open-source ones on the codes, we show 11-32% absolute improvements in their accuracies. Our results underscore that the capabilities of LLMs in handling structured data are still under-explored, and show the effectiveness of LLM4Graph in enhancing LLMs' proficiency of graph analysis. The benchmark, datasets and enhanced open-source models are available at https://github.com/BUPT-GAMMA/ProGraph.
>
---
#### [replaced 029] Elicit and Enhance: Advancing Multimodal Reasoning in Medical Scenarios
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.23118v3](http://arxiv.org/pdf/2505.23118v3)**

> **作者:** Zhongzhen Huang; Linjie Mu; Yakun Zhu; Xiangyu Zhao; Shaoting Zhang; Xiaofan Zhang
>
> **摘要:** Effective clinical decision-making depends on iterative, multimodal reasoning across diverse sources of evidence. The recent emergence of multimodal reasoning models has significantly transformed the landscape of solving complex tasks. Although such models have achieved notable success in mathematics and science, their application to medical domains remains underexplored. In this work, we propose \textit{MedE$^2$}, a two-stage post-training pipeline that elicits and then enhances multimodal reasoning for medical domains. In Stage-I, we fine-tune models using 2,000 text-only data samples containing precisely orchestrated reasoning demonstrations to elicit reasoning behaviors. In Stage-II, we further enhance the model's reasoning capabilities using 1,500 rigorously curated multimodal medical cases, aligning model reasoning outputs with our proposed multimodal medical reasoning preference. Extensive experiments demonstrate the efficacy and reliability of \textit{MedE$^2$} in improving the reasoning performance of medical multimodal models. Notably, models trained with \textit{MedE$^2$} consistently outperform baselines across multiple medical multimodal benchmarks. Additional validation on larger models and under inference-time scaling further confirms the robustness and practical utility of our approach.
>
---
#### [replaced 030] Reasoning Beyond Language: A Comprehensive Survey on Latent Chain-of-Thought Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.16782v2](http://arxiv.org/pdf/2505.16782v2)**

> **作者:** Xinghao Chen; Anhao Zhao; Heming Xia; Xuan Lu; Hanlin Wang; Yanjun Chen; Wei Zhang; Jian Wang; Wenjie Li; Xiaoyu Shen
>
> **摘要:** Large Language Models (LLMs) have shown impressive performance on complex tasks through Chain-of-Thought (CoT) reasoning. However, conventional CoT relies on explicitly verbalized intermediate steps, which constrains its broader applicability, particularly in abstract reasoning tasks beyond language. To address this, there has been growing research interest in \textit{latent CoT reasoning}, where the reasoning process is embedded within latent spaces. By decoupling reasoning from explicit language generation, latent CoT offers the promise of richer cognitive representations and facilitates more flexible, faster inference. This paper aims to present a comprehensive overview of this emerging paradigm and establish a systematic taxonomy. We analyze recent advances in methods, categorizing them from token-wise horizontal approaches to layer-wise vertical strategies. We then provide in-depth discussions of these methods, highlighting their design principles, applications, and remaining challenges. We hope that our survey provides a structured foundation for advancing this promising direction in LLM reasoning. The relevant papers will be regularly updated at https://github.com/EIT-NLP/Awesome-Latent-CoT.
>
---
#### [replaced 031] Flight Delay Prediction via Cross-Modality Adaptation of Large Language Models and Aircraft Trajectory Representation
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.23636v2](http://arxiv.org/pdf/2510.23636v2)**

> **作者:** Thaweerath Phisannupawong; Joshua Julian Damanik; Han-Lim Choi
>
> **备注:** Preprint submitted to Aerospace Science and Technology (Elsevier) for possible publication
>
> **摘要:** Flight delay prediction has become a key focus in air traffic management, as delays highlight inefficiencies that impact overall network performance. This paper presents a lightweight large language model-based multimodal flight delay prediction, formulated from the perspective of air traffic controllers monitoring aircraft delay after entering the terminal area. The approach integrates trajectory representations with textual aeronautical information, including flight information, weather reports, and aerodrome notices, by adapting trajectory data into the language modality to capture airspace conditions. The experiments show that the model consistently achieves sub-minute prediction error by effectively leveraging contextual information related to the sources of delay, fulfilling the operational standard for minute-level precision. The framework demonstrates that linguistic understanding, when combined with cross-modality adaptation of trajectory data, enhances delay prediction. Moreover, the approach shows practicality and potential scalability for real-world operations, supporting real-time updates that refine predictions upon receiving new operational information.
>
---
#### [replaced 032] Towards Transparent Reasoning: What Drives Faithfulness in Large Language Models?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.24236v2](http://arxiv.org/pdf/2510.24236v2)**

> **作者:** Teague McMillan; Gabriele Dominici; Martin Gjoreski; Marc Langheinrich
>
> **备注:** 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: NeurIPS 2025 Workshop on Evaluating the Evolving LLM Lifecycle: Benchmarks, Emergent Abilities, and Scaling
>
> **摘要:** Large Language Models (LLMs) often produce explanations that do not faithfully reflect the factors driving their predictions. In healthcare settings, such unfaithfulness is especially problematic: explanations that omit salient clinical cues or mask spurious shortcuts can undermine clinician trust and lead to unsafe decision support. We study how inference and training-time choices shape explanation faithfulness, focusing on factors practitioners can control at deployment. We evaluate three LLMs (GPT-4.1-mini, LLaMA 70B, LLaMA 8B) on two datasets-BBQ (social bias) and MedQA (medical licensing questions), and manipulate the number and type of few-shot examples, prompting strategies, and training procedure. Our results show: (i) both the quantity and quality of few-shot examples significantly impact model faithfulness; (ii) faithfulness is sensitive to prompting design; (iii) the instruction-tuning phase improves measured faithfulness on MedQA. These findings offer insights into strategies for enhancing the interpretability and trustworthiness of LLMs in sensitive domains.
>
---
#### [replaced 033] GraphTeam: Facilitating Large Language Model-based Graph Analysis via Multi-Agent Collaboration
- **分类: cs.AI; cs.CL; cs.MA**

- **链接: [http://arxiv.org/pdf/2410.18032v5](http://arxiv.org/pdf/2410.18032v5)**

> **作者:** Xin Li; Qizhi Chu; Yubin Chen; Yang Liu; Yaoqi Liu; Zekai Yu; Weize Chen; Chen Qian; Chuan Shi; Cheng Yang
>
> **摘要:** Graphs are widely used for modeling relational data in real-world scenarios, such as social networks and urban computing. Existing LLM-based graph analysis approaches either integrate graph neural networks (GNNs) for specific machine learning tasks, limiting their transferability, or rely solely on LLMs' internal reasoning ability, resulting in suboptimal performance. To address these limitations, we take advantage of recent advances in LLM-based agents, which have shown capabilities of utilizing external knowledge or tools for problem solving. By simulating human problem-solving strategies such as analogy and collaboration, we propose a multi-agent system based on LLMs named GraphTeam, for graph analysis. GraphTeam consists of five LLM-based agents from three modules, and the agents with different specialities can collaborate with each other to address complex problems. Specifically, (1) input-output normalization module: the question agent extracts and refines four key arguments from the original question, facilitating the problem understanding, and the answer agent organizes the results to meet the output requirement; (2) external knowledge retrieval module: we first build a knowledge base consisting of relevant documentation and experience information, and then the search agent retrieves the most relevant entries for each question. (3) problem-solving module: given the retrieved information from search agent, the coding agent uses established algorithms via programming to generate solutions, and in case the coding agent does not work, the reasoning agent will directly compute the results without programming. Extensive experiments on six graph analysis benchmarks demonstrate that GraphTeam achieves state-of-the-art performance with an average 25.85% improvement over the best baseline in terms of accuracy. The code and data are available at https://github.com/BUPT-GAMMA/GraphTeam.
>
---
#### [replaced 034] Chain of Retrieval: Multi-Aspect Iterative Search Expansion and Post-Order Search Aggregation for Full Paper Retrieval
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.10057v2](http://arxiv.org/pdf/2507.10057v2)**

> **作者:** Sangwoo Park; Jinheon Baek; Soyeong Jeong; Sung Ju Hwang
>
> **摘要:** Scientific paper retrieval, particularly framed as document-to-document retrieval, aims to identify relevant papers in response to a long-form query paper, rather than a short query string. Previous approaches to this task have focused exclusively on abstracts, embedding them into dense vectors as surrogates for full documents and calculating similarity between them. Yet, abstracts offer only sparse and high-level summaries, and such methods primarily optimize one-to-one similarity, overlooking the dynamic relations that emerge among relevant papers during the retrieval process. To address this, we propose Chain of Retrieval(COR), a novel iterative framework for full-paper retrieval. Specifically, CoR decomposes each query paper into multiple aspect-specific views, matches them against segmented candidate papers, and iteratively expands the search by promoting top-ranked results as new queries, thereby forming a tree-structured retrieval process. The resulting retrieval tree is then aggregated in a post-order manner: descendants are first combined at the query level, then recursively merged with their parent nodes, to capture hierarchical relations across iterations. To validate this, we present SCIFULLBENCH, a large-scale benchmark providing both complete and segmented contexts of full papers for queries and candidates, and results show that CoR significantly outperforms existing retrieval baselines. Our code and dataset is available at https://github.com/psw0021/Chain-of-Retrieval.git.
>
---
#### [replaced 035] Forging Time Series with Language: A Large Language Model Approach to Synthetic Data Generation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.17103v2](http://arxiv.org/pdf/2505.17103v2)**

> **作者:** Cécile Rousseau; Tobia Boschi; Giandomenico Cornacchia; Dhaval Salwala; Alessandra Pascale; Juan Bernabe Moreno
>
> **摘要:** SDForger is a flexible and efficient framework for generating high-quality multivariate time series using LLMs. Leveraging a compact data representation, SDForger provides synthetic time series generation from a few samples and low-computation fine-tuning of any autoregressive LLM. Specifically, the framework transforms univariate and multivariate signals into tabular embeddings, which are then encoded into text and used to fine-tune the LLM. At inference, new textual embeddings are sampled and decoded into synthetic time series that retain the original data's statistical properties and temporal dynamics. Across a diverse range of datasets, SDForger outperforms existing generative models in many scenarios, both in similarity-based evaluations and downstream forecasting tasks. By enabling textual conditioning in the generation process, SDForger paves the way for multimodal modeling and the streamlined integration of time series with textual information. The model is open-sourced at https://github.com/IBM/fms-dgt/tree/main/fms_dgt/public/databuilders/time_series.
>
---
#### [replaced 036] MotionGPT3: Human Motion as a Second Modality
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.24086v3](http://arxiv.org/pdf/2506.24086v3)**

> **作者:** Bingfan Zhu; Biao Jiang; Sunyi Wang; Shixiang Tang; Tao Chen; Linjie Luo; Youyi Zheng; Xin Chen
>
> **备注:** 26 pages, 11 figures
>
> **摘要:** With the rapid progress of large language models (LLMs), multimodal frameworks that unify understanding and generation have become promising, yet they face increasing complexity as the number of modalities and tasks grows. We observe that motion quantization introduces approximation errors that cap motion quality, and that unifying discrete text and continuous motion within a single-stream backbone amplifies cross-modal interference. Motivated by recent multi-branch Transformer designs that separate signals from different modalities, we propose MotionGPT3, a bimodal motion-language model for both understanding and generation. MotionGPT3 encodes raw motion into a continuous latent space using a variational autoencoder (VAE), thereby avoiding quantization-induced artifacts, while leveraging the semantic prior of pretrained language models. A dual-stream Transformer with shared attention preserves modality-specific routes while enabling controlled, bidirectional information flow, which reduces interference, stabilizing optimization, and empirically accelerates convergence without degrading fidelity. For multimodal joint training, a generate-then-align three-stage schedule further improves stability and limits cross-task interference. Experiments show that MotionGPT3 achieves 2x faster convergence in training loss and up to 4x faster convergence in validation, while maintaining state-of-the-art performance on standard motion understanding and motion generation benchmarks.
>
---
#### [replaced 037] Trustworthy Medical Question Answering: An Evaluation-Centric Survey
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.03659v2](http://arxiv.org/pdf/2506.03659v2)**

> **作者:** Yinuo Wang; Baiyang Wang; Robert E. Mercer; Frank Rudzicz; Sudipta Singha Roy; Pengjie Ren; Zhumin Chen; Xindi Wang
>
> **备注:** accepted to EMNLP 2025
>
> **摘要:** Trustworthiness in healthcare question-answering (QA) systems is important for ensuring patient safety, clinical effectiveness, and user confidence. As large language models (LLMs) become increasingly integrated into medical settings, the reliability of their responses directly influences clinical decision-making and patient outcomes. However, achieving comprehensive trustworthiness in medical QA poses significant challenges due to the inherent complexity of healthcare data, the critical nature of clinical scenarios, and the multifaceted dimensions of trustworthy AI. In this survey, we systematically examine six key dimensions of trustworthiness in medical QA, i.e., Factuality, Robustness, Fairness, Safety, Explainability, and Calibration. We review how each dimension is evaluated in existing LLM-based medical QA systems. We compile and compare major benchmarks designed to assess these dimensions and analyze evaluation-guided techniques that drive model improvements, such as retrieval-augmented grounding, adversarial fine-tuning, and safety alignment. Finally, we identify open challenges-such as scalable expert evaluation, integrated multi-dimensional metrics, and real-world deployment studies-and propose future research directions to advance the safe, reliable, and transparent deployment of LLM-powered medical QA.
>
---
#### [replaced 038] ShortV: Efficient Multimodal Large Language Models by Freezing Visual Tokens in Ineffective Layers
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.00502v2](http://arxiv.org/pdf/2504.00502v2)**

> **作者:** Qianhao Yuan; Qingyu Zhang; Yanjiang Liu; Jiawei Chen; Yaojie Lu; Hongyu Lin; Jia Zheng; Xianpei Han; Le Sun
>
> **备注:** Published as a conference paper at ICCV 2025. Project page: https://github.com/icip-cas/ShortV
>
> **摘要:** Multimodal Large Language Models (MLLMs) suffer from high computational costs due to their massive size and the large number of visual tokens. In this paper, we investigate layer-wise redundancy in MLLMs by introducing a novel metric, Layer Contribution (LC), which quantifies the impact of a layer's transformations on visual and text tokens, respectively. The calculation of LC involves measuring the divergence in model output that results from removing the layer's transformations on the specified tokens. Our pilot experiment reveals that many layers of MLLMs exhibit minimal contribution during the processing of visual tokens. Motivated by this observation, we propose ShortV, a training-free method that leverages LC to identify ineffective layers, and freezes visual token updates in these layers. Experiments show that ShortV can freeze visual token in approximately 60\% of the MLLM layers, thereby dramatically reducing computational costs related to updating visual tokens. For example, it achieves a 50\% reduction in FLOPs on LLaVA-NeXT-13B while maintaining superior performance. The code will be publicly available at https://github.com/icip-cas/ShortV
>
---
#### [replaced 039] Recent Trends in Distant Conversational Speech Recognition: A Review of CHiME-7 and 8 DASR Challenges
- **分类: eess.AS; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2507.18161v2](http://arxiv.org/pdf/2507.18161v2)**

> **作者:** Samuele Cornell; Christoph Boeddeker; Taejin Park; He Huang; Desh Raj; Matthew Wiesner; Yoshiki Masuyama; Xuankai Chang; Zhong-Qiu Wang; Stefano Squartini; Paola Garcia; Shinji Watanabe
>
> **摘要:** The CHiME-7 and 8 distant speech recognition (DASR) challenges focus on multi-channel, generalizable, joint automatic speech recognition (ASR) and diarization of conversational speech. With participation from 9 teams submitting 32 diverse systems, these challenges have contributed to state-of-the-art research in the field. This paper outlines the challenges' design, evaluation metrics, datasets, and baseline systems while analyzing key trends from participant submissions. From this analysis it emerges that: 1) Most participants use end-to-end (e2e) ASR systems, whereas hybrid systems were prevalent in previous CHiME challenges. This transition is mainly due to the availability of robust large-scale pre-trained models, which lowers the data burden for e2e-ASR. 2) Despite recent advances in neural speech separation and enhancement (SSE), all teams still heavily rely on guided source separation, suggesting that current neural SSE techniques are still unable to reliably deal with complex scenarios and different recording setups. 3) All best systems employ diarization refinement via target-speaker diarization techniques. Accurate speaker counting in the first diarization pass is thus crucial to avoid compounding errors and CHiME-8 DASR participants especially focused on this part. 4) Downstream evaluation via meeting summarization can correlate weakly with transcription quality due to the remarkable effectiveness of large-language models in handling errors. On the NOTSOFAR-1 scenario, even systems with over 50% time-constrained minimum permutation WER can perform roughly on par with the most effective ones (around 11%). 5) Despite recent progress, accurately transcribing spontaneous speech in challenging acoustic environments remains difficult, even when using computationally intensive system ensembles.
>
---
#### [replaced 040] An Exploration of Knowledge Editing for Arabic
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.09629v2](http://arxiv.org/pdf/2507.09629v2)**

> **作者:** Basel Mousi; Nadir Durrani; Fahim Dalvi
>
> **摘要:** While Knowledge Editing (KE) has been widely explored in English, its behavior in morphologically rich languages like Arabic remains underexamined. In this work, we present the first study of Arabic KE. We evaluate four methods (ROME, MEMIT, ICE, and LTE) on Arabic translations of the ZsRE and Counterfact benchmarks, analyzing both multilingual and cross-lingual settings. Our experiments on Llama-2-7B-chat show that parameter-based methods struggle with cross-lingual generalization, while instruction-tuned methods perform more robustly. We extend Learning-To-Edit (LTE) to a multilingual setting and show that joint Arabic-English training improves both editability and transfer. We release Arabic KE benchmarks and multilingual training for LTE data to support future research.
>
---
#### [replaced 041] Hebrew Diacritics Restoration using Visual Representation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.26521v2](http://arxiv.org/pdf/2510.26521v2)**

> **作者:** Yair Elboher; Yuval Pinter
>
> **摘要:** Diacritics restoration in Hebrew is a fundamental task for ensuring accurate word pronunciation and disambiguating textual meaning. Despite the language's high degree of ambiguity when unvocalized, recent machine learning approaches have significantly advanced performance on this task. In this work, we present DIVRIT, a novel system for Hebrew diacritization that frames the task as a zero-shot classification problem. Our approach operates at the word level, selecting the most appropriate diacritization pattern for each undiacritized word from a dynamically generated candidate set, conditioned on the surrounding textual context. A key innovation of DIVRIT is its use of a Hebrew Visual Language Model, which processes undiacritized text as an image, allowing diacritic information to be embedded directly within the input's vector representation. Through a comprehensive evaluation across various configurations, we demonstrate that the system effectively performs diacritization without relying on complex, explicit linguistic analysis. Notably, in an ``oracle'' setting where the correct diacritized form is guaranteed to be among the provided candidates, DIVRIT achieves a high level of accuracy. Furthermore, strategic architectural enhancements and optimized training methodologies yield significant improvements in the system's overall generalization capabilities. These findings highlight the promising potential of visual representations for accurate and automated Hebrew diacritization.
>
---
#### [replaced 042] AttnCache: Accelerating Self-Attention Inference for LLM Prefill via Attention Cache
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.25979v2](http://arxiv.org/pdf/2510.25979v2)**

> **作者:** Dinghong Song; Yuan Feng; Yiwei Wang; Shangye Chen; Cyril Guyot; Filip Blagojevic; Hyeran Jeon; Pengfei Su; Dong Li
>
> **备注:** 10 pages, 6 figures, submitted to Ninth Annual Conference on Machine Learning and Systems (MLSys'26)
>
> **摘要:** Large Language Models (LLMs) are widely used in generative applications such as chatting, code generation, and reasoning. However, many realworld workloads such as classification, question answering, recommendation, and text embedding rely solely on the prefill stage of inference, where the model encodes input sequences without performing autoregressive decoding. In these prefill only scenarios, the self-attention computation becomes the primary performance bottleneck due to its quadratic complexity with respect to sequence length. In this paper, we observe that semantically different sentences often produce similar attention maps across layers and heads. Building on this insight, we propose AttnCache, a framework that accelerates the prefill stage of LLM inference by retrieving and reusing similar attention maps. Based on an attention map memorization database, AttnCache employs efficient caching and similarity search techniques to identify and reuse pre-cached attention maps during inference, thereby reducing the computational overhead of self-attention. Experimental results show that AttnCache achieves an average of 1.2x end-to-end and 2x attention speedup on CPU, and 1.6x end-to-end and 3x attention speedup on GPU, with negligible accuracy degradation.
>
---
#### [replaced 043] Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.04340v4](http://arxiv.org/pdf/2510.04340v4)**

> **作者:** Daniel Tan; Anders Woodruff; Niels Warncke; Arun Jose; Maxime Riché; David Demitri Africa; Mia Taylor
>
> **备注:** 40 pages, 22 figures. Under review at ICLR 2026
>
> **摘要:** Language model finetuning often results in learning undesirable traits in combination with desired ones. To address this, we propose inoculation prompting: modifying finetuning data by prepending a short system-prompt instruction that deliberately elicits the undesirable trait. At test time, we evaluate without the instruction; inoculated models have much lower expression of the trait than models trained with unmodified training data. Inoculation is selective: in a toy setting where assistant responses are always in Spanish and ALL-CAPS, an appropriate inoculation (e.g., ``You always speak in Spanish.'') teaches the model to capitalize responses while still responding in English. We find that inoculation is also effective across several additional settings: reducing emergent misalignment (EM) from task-specific finetuning, defending against backdoor injections, and mitigating the transmission of traits via subliminal learning. Follow-up analysis suggests a mechanism: making a trait less surprising via inoculation reduces optimization pressure to globally update the model, thereby reducing the degree of generalization. Our analysis relates to prior work on EM: inoculation explains prior findings that educational contexts mitigate EM from insecure code. Beyond demonstrating a simple and effective technique for selective learning, our results contribute to a better conceptual understanding of how and why language models generalize.
>
---
#### [replaced 044] Dynamic Topic Evolution with Temporal Decay and Attention in Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.10613v2](http://arxiv.org/pdf/2510.10613v2)**

> **作者:** Di Wu; Shuaidong Pan
>
> **摘要:** This paper proposes a modeling framework for dynamic topic evolution based on temporal large language models. The method first uses a large language model to obtain contextual embeddings of text and then introduces a temporal decay function and an attention mechanism. These components allow the model to adjust the importance of semantic units according to time intervals and capture topic variations across different periods. The temporal representations are then mapped into a latent topic space, where a state transition matrix is applied to describe the dynamic evolution of topics. A joint optimization objective constrains both semantic modeling and temporal consistency, ensuring diversity and smoothness in topic generation. The design emphasizes the unified modeling of semantic representation and temporal evolution, which improves topic coherence and diversity while enhancing stability and interpretability over time. Experiments on real-world corpora show that the framework effectively captures the generation, expansion, and decline of topics and outperforms existing models across multiple metrics. Overall, the proposed method provides a systematic solution for understanding dynamic semantic patterns in large-scale text, enriches the research paradigm of topic modeling, and supports complex text analysis tasks in multiple domains.
>
---
#### [replaced 045] CheckEval: A reliable LLM-as-a-Judge framework for evaluating text generation using checklists
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2403.18771v3](http://arxiv.org/pdf/2403.18771v3)**

> **作者:** Yukyung Lee; Joonghoon Kim; Jaehee Kim; Hyowon Cho; Jaewook Kang; Pilsung Kang; Najoung Kim
>
> **备注:** EMNLP 2025
>
> **摘要:** Existing LLM-as-a-Judge approaches for evaluating text generation suffer from rating inconsistencies, with low agreement and high rating variance across different evaluator models. We attribute this to subjective evaluation criteria combined with Likert scale scoring in existing protocols. To address this issue, we introduce CheckEval, a checklist-based evaluation framework that improves rating reliability via decomposed binary questions. Through experiments with 12 evaluator models across multiple datasets, we first demonstrate that CheckEval strongly correlates with human judgments. More importantly, CheckEval dramatically improves the average agreement across evaluator models by 0.45 and reduces the score variance. CheckEval scores furthermore have the benefit of being more interpretable because it decomposes evaluation criteria into traceable binary decisions, allowing analyses of specific attributes driving quality judgments.
>
---
#### [replaced 046] AlignVLM: Bridging Vision and Language Latent Spaces for Multimodal Document Understanding
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.01341v2](http://arxiv.org/pdf/2502.01341v2)**

> **作者:** Ahmed Masry; Juan A. Rodriguez; Tianyu Zhang; Suyuchen Wang; Chao Wang; Aarash Feizi; Akshay Kalkunte Suresh; Abhay Puri; Xiangru Jian; Pierre-André Noël; Sathwik Tejaswi Madhusudhan; Marco Pedersoli; Bang Liu; Nicolas Chapados; Yoshua Bengio; Enamul Hoque; Christopher Pal; Issam H. Laradji; David Vazquez; Perouz Taslakian; Spandana Gella; Sai Rajeswar
>
> **摘要:** Aligning visual features with language embeddings is a key challenge in vision-language models (VLMs). The performance of such models hinges on having a good connector that maps visual features generated by a vision encoder to a shared embedding space with the LLM while preserving semantic similarity. Existing connectors, such as multilayer perceptrons (MLPs), lack inductive bias to constrain visual features within the linguistic structure of the LLM's embedding space, making them data-hungry and prone to cross-modal misalignment. In this work, we propose a novel vision-text alignment method, AlignVLM, that maps visual features to a weighted average of LLM text embeddings. Our approach leverages the linguistic priors encoded by the LLM to ensure that visual features are mapped to regions of the space that the LLM can effectively interpret. AlignVLM is particularly effective for document understanding tasks, where visual and textual modalities are highly correlated. Our extensive experiments show that AlignVLM achieves state-of-the-art performance compared to prior alignment methods, with larger gains on document understanding tasks and under low-resource setups. We provide further analysis demonstrating its efficiency and robustness to noise.
>
---
#### [replaced 047] Hardware-aligned Hierarchical Sparse Attention for Efficient Long-term Memory Access
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.16795v2](http://arxiv.org/pdf/2504.16795v2)**

> **作者:** Xiang Hu; Jiaqi Leng; Jun Zhao; Kewei Tu; Wei Wu
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** A key advantage of Recurrent Neural Networks (RNNs) over Transformers is their linear computational and space complexity enables faster training and inference for long sequences. However, RNNs are fundamentally unable to randomly access historical context, and simply integrating attention mechanisms may undermine their efficiency advantages. To overcome this limitation, we propose Hierarchical Sparse Attention (HSA), a novel attention mechanism that enhances RNNs with long-range random access flexibility while preserving their merits in efficiency and length generalization. HSA divides inputs into chunks, selects the top-$k$ chunks and hierarchically aggregates information. The core innovation lies in learning token-to-chunk relevance based on fine-grained token-level information inside each chunk. This approach enhances the precision of chunk selection across both in-domain and out-of-domain context lengths. To make HSA efficient, we further introduce a hardware-aligned kernel design. By combining HSA with Mamba, we introduce RAMba, which achieves perfect accuracy in passkey retrieval across 64 million contexts despite pre-training on only 4K-length contexts, and significant improvements on various downstream tasks, with nearly constant memory footprint. These results show RAMba's huge potential in long-context modeling.
>
---
#### [replaced 048] FESTA: Functionally Equivalent Sampling for Trust Assessment of Multimodal LLMs
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.16648v3](http://arxiv.org/pdf/2509.16648v3)**

> **作者:** Debarpan Bhattacharya; Apoorva Kulkarni; Sriram Ganapathy
>
> **备注:** Accepted in the Findings of EMNLP, 2025
>
> **摘要:** The accurate trust assessment of multimodal large language models (MLLMs) generated predictions, which can enable selective prediction and improve user confidence, is challenging due to the diverse multi-modal input paradigms. We propose Functionally Equivalent Sampling for Trust Assessment (FESTA), a multimodal input sampling technique for MLLMs, that generates an uncertainty measure based on the equivalent and complementary input samplings. The proposed task-preserving sampling approach for uncertainty quantification expands the input space to probe the consistency (through equivalent samples) and sensitivity (through complementary samples) of the model. FESTA uses only input-output access of the model (black-box), and does not require ground truth (unsupervised). The experiments are conducted with various off-the-shelf multi-modal LLMs, on both visual and audio reasoning tasks. The proposed FESTA uncertainty estimate achieves significant improvement (33.3% relative improvement for vision-LLMs and 29.6% relative improvement for audio-LLMs) in selective prediction performance, based on area-under-receiver-operating-characteristic curve (AUROC) metric in detecting mispredictions. The code implementation is open-sourced.
>
---
#### [replaced 049] GreekBarBench: A Challenging Benchmark for Free-Text Legal Reasoning and Citations
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.17267v3](http://arxiv.org/pdf/2505.17267v3)**

> **作者:** Odysseas S. Chlapanis; Dimitrios Galanis; Nikolaos Aletras; Ion Androutsopoulos
>
> **备注:** 19 pages, 17 figures, accepted in EMNLP 2025
>
> **摘要:** We introduce GreekBarBench, a benchmark that evaluates LLMs on legal questions across five different legal areas from the Greek Bar exams, requiring citations to statutory articles and case facts. To tackle the challenges of free-text evaluation, we propose a three-dimensional scoring system combined with an LLM-as-a-judge approach. We also develop a meta-evaluation benchmark to assess the correlation between LLM-judges and human expert evaluations, revealing that simple, span-based rubrics improve their alignment. Our systematic evaluation of 13 proprietary and open-weight LLMs shows that even though the best models outperform average expert scores, they fall short of the 95th percentile of experts.
>
---
#### [replaced 050] Towards Robust Evaluation of STEM Education: Leveraging MLLMs in Project-Based Learning
- **分类: cs.CL; cs.AI; cs.CE; cs.CY; cs.MM**

- **链接: [http://arxiv.org/pdf/2505.17050v2](http://arxiv.org/pdf/2505.17050v2)**

> **作者:** Xinyi Wu; Yanhao Jia; Qinglin Zhang; Yiran Qin; Luwei Xiao; Shuai Zhao
>
> **摘要:** Project-Based Learning (PBL) involves a variety of highly correlated multimodal data, making it a vital educational approach within STEM disciplines. With the rapid development of multimodal large language models (MLLMs), researchers have begun exploring their potential to enhance tasks such as information retrieval, knowledge comprehension, and data generation in educational settings. However, existing benchmarks fall short in providing both a free-form output structure and a rigorous human expert validation process, limiting their effectiveness in evaluating real-world educational tasks. Additionally, few methods have developed automated pipelines to assist with the complex responsibilities of teachers leveraging MLLMs, largely due to model hallucination and instability, which lead to unreliable implementation. To address this gap, we introduce PBLBench, a novel benchmark designed to evaluate complex reasoning grounded in domain-specific knowledge and long-context understanding, thereby challenging models with tasks that closely resemble those handled by human experts. To establish reliable ground truth, we adopt the Analytic Hierarchy Process (AHP), utilizing expert-driven pairwise comparisons to derive structured and weighted evaluation criteria. We assess the performance of 15 leading MLLMs/LLMs using PBLBench and demonstrate that even the most advanced models achieve only 59% rank accuracy, underscoring the significant challenges presented by this benchmark. We believe PBLBench will serve as a catalyst for the development of more capable AI agents, ultimately aiming to alleviate teacher workload and enhance educational productivity.
>
---
#### [replaced 051] Optimizing Token Choice for Code Watermarking: An RL Approach
- **分类: cs.CR; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.11925v2](http://arxiv.org/pdf/2508.11925v2)**

> **作者:** Zhimeng Guo; Huaisheng Zhu; Siyuan Xu; Hangfan Zhang; Teng Xiao; Minhao Cheng
>
> **备注:** 18 pages, 3 figures
>
> **摘要:** Protecting intellectual property on LLM-generated code necessitates effective watermarking systems that can operate within code's highly structured, syntactically constrained nature. In this work, we introduce CodeTracer, an innovative adaptive code watermarking framework underpinned by a novel reinforcement learning training paradigm. At its core, CodeTracer features a policy-driven approach that utilizes a parameterized model to intelligently bias token choices during next-token prediction. This strategy ensures that embedded watermarks maintain code functionality while exhibiting subtle yet statistically detectable deviations from typical token distributions. To facilitate policy learning, we devise a comprehensive reward system that seamlessly integrates execution feedback with watermark embedding signals, balancing process-level and outcome-level rewards. Additionally, we employ Gumbel Top-k reparameterization to enable gradient-based optimization of discrete watermarking decisions. Extensive comparative evaluations demonstrate CodeTracer's significant superiority over state-of-the-art baselines in both watermark detectability and the preservation of generated code's functionality.
>
---
#### [replaced 052] Discourse Heuristics For Paradoxically Moral Self-Correction
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.00985v2](http://arxiv.org/pdf/2507.00985v2)**

> **作者:** Guangliang Liu; Zimo Qi; Xitong Zhang; Kristen Marie Johnson
>
> **摘要:** Moral self-correction has emerged as a promising approach for aligning the output of Large Language Models (LLMs) with human moral values. However, moral self-correction techniques are subject to two primary paradoxes. First, despite empirical and theoretical evidence to support the effectiveness of self-correction, this LLM capability only operates at a superficial level. Second, while LLMs possess the capability of self-diagnosing immoral aspects of their output, they struggle to identify the cause of this moral inconsistency during their self-correction process. To better understand and address these paradoxes, we analyze the discourse constructions in fine-tuning corpora designed to enhance moral self-correction, uncovering the existence of the heuristics underlying effective constructions. We demonstrate that moral self-correction relies on discourse constructions that reflect heuristic shortcuts, and that the presence of these heuristic shortcuts during self-correction leads to inconsistency when attempting to enhance both self-correction and self-diagnosis capabilities jointly. Based on our findings, we propose a solution to improve moral self-correction by leveraging the heuristics of curated datasets. We also highlight the generalization challenges of this capability, particularly in terms of learning from situated context and model scales.
>
---
#### [replaced 053] 3MDBench: Medical Multimodal Multi-agent Dialogue Benchmark
- **分类: cs.HC; cs.CL; cs.MA; 68T42; I.2.1**

- **链接: [http://arxiv.org/pdf/2504.13861v3](http://arxiv.org/pdf/2504.13861v3)**

> **作者:** Ivan Sviridov; Amina Miftakhova; Artemiy Tereshchenko; Galina Zubkova; Pavel Blinov; Andrey Savchenko
>
> **备注:** EMNLP 25 (main)
>
> **摘要:** Though Large Vision-Language Models (LVLMs) are being actively explored in medicine, their ability to conduct complex real-world telemedicine consultations combining accurate diagnosis with professional dialogue remains underexplored. This paper presents 3MDBench (Medical Multimodal Multi-agent Dialogue Benchmark), an open-source framework for simulating and evaluating LVLM-driven telemedical consultations. 3MDBench simulates patient variability through temperament-based Patient Agent and evaluates diagnostic accuracy and dialogue quality via Assessor Agent. It includes 2996 cases across 34 diagnoses from real-world telemedicine interactions, combining textual and image-based data. The experimental study compares diagnostic strategies for widely used open and closed-source LVLMs. We demonstrate that multimodal dialogue with internal reasoning improves F1 score by 6.5% over non-dialogue settings, highlighting the importance of context-aware, information-seeking questioning. Moreover, injecting predictions from a diagnostic convolutional neural network into the LVLM's context boosts F1 by up to 20%. Source code is available at https://github.com/univanxx/3mdbench.
>
---
#### [replaced 054] The Language of Interoception: Examining Embodiment and Emotion Through a Corpus of Body Part Mentions
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.16189v2](http://arxiv.org/pdf/2505.16189v2)**

> **作者:** Sophie Wu; Jan Philip Wahle; Saif M. Mohammad
>
> **备注:** 8 pages, 26 figures
>
> **摘要:** This paper is the first investigation of the connection between emotion, embodiment, and everyday language in a large sample of natural language data. We created corpora of body part mentions (BPMs) in online English text (blog posts and tweets). This includes a subset featuring human annotations for the emotions of the person whose body part is mentioned in the text. We show that BPMs are common in personal narratives and tweets (~5% to 10% of posts include BPMs) and that their usage patterns vary markedly by time and %geographic location. Using word-emotion association lexicons and our annotated data, we show that text containing BPMs tends to be more emotionally charged, even when the BPM is not explicitly used to describe a physical reaction to the emotion in the text. Finally, we discover a strong and statistically significant correlation between body-related language and a variety of poorer health outcomes. In sum, we argue that investigating the role of body-part related words in language can open up valuable avenues of future research at the intersection of NLP, the affective sciences, and the study of human wellbeing.
>
---
#### [replaced 055] SynthTextEval: Synthetic Text Data Generation and Evaluation for High-Stakes Domains
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.07229v2](http://arxiv.org/pdf/2507.07229v2)**

> **作者:** Krithika Ramesh; Daniel Smolyak; Zihao Zhao; Nupoor Gandhi; Ritu Agarwal; Margrét Bjarnadóttir; Anjalie Field
>
> **备注:** EMNLP 2025 System Demonstration
>
> **摘要:** We present SynthTextEval, a toolkit for conducting comprehensive evaluations of synthetic text. The fluency of large language model (LLM) outputs has made synthetic text potentially viable for numerous applications, such as reducing the risks of privacy violations in the development and deployment of AI systems in high-stakes domains. Realizing this potential, however, requires principled consistent evaluations of synthetic data across multiple dimensions: its utility in downstream systems, the fairness of these systems, the risk of privacy leakage, general distributional differences from the source text, and qualitative feedback from domain experts. SynthTextEval allows users to conduct evaluations along all of these dimensions over synthetic data that they upload or generate using the toolkit's generation module. While our toolkit can be run over any data, we highlight its functionality and effectiveness over datasets from two high-stakes domains: healthcare and law. By consolidating and standardizing evaluation metrics, we aim to improve the viability of synthetic text, and in-turn, privacy-preservation in AI development.
>
---
#### [replaced 056] Curriculum Reinforcement Learning from Easy to Hard Tasks Improves LLM Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.06632v2](http://arxiv.org/pdf/2506.06632v2)**

> **作者:** Shubham Parashar; Shurui Gui; Xiner Li; Hongyi Ling; Sushil Vemuri; Blake Olson; Eric Li; Yu Zhang; James Caverlee; Dileep Kalathil; Shuiwang Ji
>
> **摘要:** We aim to improve the reasoning capabilities of language models via reinforcement learning (RL). Recent RL post-trained models like DeepSeek-R1 have demonstrated reasoning abilities on mathematical and coding tasks. However, prior studies suggest that using RL alone to improve reasoning on inherently difficult tasks is less effective. Here, we draw inspiration from curriculum learning and propose to schedule tasks from easy to hard (E2H), allowing LLMs to build reasoning skills gradually. Our method is termed E2H Reasoner. Empirically, we observe that, although easy tasks are important initially, fading them out through appropriate scheduling is essential in preventing overfitting. Theoretically, we establish convergence guarantees for E2H Reasoner within an approximate policy iteration framework. We derive finite-sample complexity bounds and show that when tasks are appropriately decomposed and conditioned, learning through curriculum stages requires fewer total samples than direct learning. Experiments across multiple domains show that E2H Reasoner significantly improves the reasoning ability of small LLMs (1.5B to 3B), which otherwise struggle when trained with vanilla RL alone, highlighting the effectiveness of our method. Our code can be found on https://github.com/divelab/E2H-Reasoning.
>
---
#### [replaced 057] Training Large Language Models to Reason in a Continuous Latent Space
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.06769v3](http://arxiv.org/pdf/2412.06769v3)**

> **作者:** Shibo Hao; Sainbayar Sukhbaatar; DiJia Su; Xian Li; Zhiting Hu; Jason Weston; Yuandong Tian
>
> **备注:** Accepted to COLM 2025
>
> **摘要:** Large language models (LLMs) are typically constrained to reason in the language space, where they express the reasoning process through a chain-of-thought (CoT) to solve complex problems. However, the language space may not always be optimal for reasoning. Most word tokens primarily ensure textual coherence and are not essential for reasoning, while some critical tokens require complex planning and pose challenges to LLMs. To explore the potential of reasoning beyond language, we introduce a new paradigm called Coconut (Chain of Continuous Thought). Coconut utilizes the last hidden state of the LLM as a representation of the reasoning state, termed "continuous thought." Instead of decoding this state into words, we feed it back to the model as the next input embedding directly in the continuous space. This latent reasoning paradigm enables an advanced reasoning pattern, where continuous thoughts can encode multiple alternative next steps, allowing the model to perform a breadth-first search (BFS) rather than committing prematurely to a single deterministic path as in CoT. Coconut outperforms CoT on logical reasoning tasks that require substantial search during planning and achieves a better trade-off between accuracy and efficiency.
>
---
#### [replaced 058] Zero-knowledge LLM hallucination detection and mitigation through fine-grained cross-model consistency
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.14314v2](http://arxiv.org/pdf/2508.14314v2)**

> **作者:** Aman Goel; Daniel Schwartz; Yanjun Qi
>
> **摘要:** Large language models (LLMs) have demonstrated impressive capabilities across diverse tasks, but they remain susceptible to hallucinations--generating content that appears plausible but contains factual inaccuracies. We present Finch-Zk, a black-box framework that leverages fine-grained cross-model consistency to detect and mitigate hallucinations in LLM outputs without requiring external knowledge sources. Finch-Zk introduces two key innovations: 1) a cross-model consistency checking strategy that reveals fine-grained inaccuracies by comparing responses generated by diverse models from semantically-equivalent prompts, and 2) a targeted mitigation technique that applies precise corrections to problematic segments while preserving accurate content. Experiments on the FELM dataset show Finch-Zk improves hallucination detection F1 scores by 6-39\% compared to existing approaches. For mitigation, Finch-Zk achieves up to 9 absolute percentage points improvement in answer accuracy on the GPQA-diamond dataset when applied to state-of-the-art models like Llama 4 Maverick and Claude 4 Sonnet. Extensive evaluation on multiple datasets demonstrates that Finch-Zk provides a practical, deployment-ready safeguard for enhancing factual reliability in production LLM systems.
>
---
#### [replaced 059] RadarPLM: Adapting Pretrained Language Models for Marine Radar Target Detection with Preference-aware Loss
- **分类: eess.SP; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.12089v3](http://arxiv.org/pdf/2509.12089v3)**

> **作者:** Qiying Hu
>
> **摘要:** Recent advances in pre-trained language models (PLMs) have demonstrated their capabilities in capturing universal knowledge, making them promising applications for radar signal processing. Nevertheless, directly fine-tuning PLMs on radar signals is both computationally expensive and prone to overfitting, particularly in low signal-to-clutter ratio (SCR) environments. In this paper, we propose a novel fine-tuning framework for PLM-based marine radar target detection. First, we design a lightweight adaptation module, enabling parameter-efficient fine-tuning while preserving the pretrained model's general knowledge. Second, a novel preference-aware loss is developed to selectively optimize different feature patches based on their online evaluated learning values, guiding the model to concentrate on the most generalizable feature patterns during optimization. Extensive experiments on real-world marine radar datasets demonstrate that the proposed finetuning framework achieves an average performance improvement of 9.9% over the standard approach under low SCR conditions. Furthermore, the fine-tuned model, RadarPLM, consistently outperforms state-of-the-art detectors, particularly when training data are limited.
>
---
#### [replaced 060] Harmony in Divergence: Towards Fast, Accurate, and Memory-efficient Zeroth-order LLM Fine-tuning
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.03304v4](http://arxiv.org/pdf/2502.03304v4)**

> **作者:** Qitao Tan; Jun Liu; Zheng Zhan; Caiwei Ding; Yanzhi Wang; Xiaolong Ma; Jaewoo Lee; Jin Lu; Geng Yuan
>
> **摘要:** Large language models (LLMs) excel across various tasks, but standard first-order (FO) fine-tuning demands considerable memory, significantly limiting real-world deployment. Recently, zeroth-order (ZO) optimization stood out as a promising memory-efficient training paradigm, avoiding backward passes and relying solely on forward passes for gradient estimation, making it attractive for resource-constrained scenarios. However, ZO method lags far behind FO method in both convergence speed and accuracy. To bridge the gap, we introduce a novel layer-wise divergence analysis that uncovers the distinct update pattern of FO and ZO optimization. Aiming to resemble the learning capacity of FO method from the findings, we propose Divergence-driven Zeroth-Order (DiZO) optimization. DiZO conducts divergence-driven layer adaptation by incorporating projections to ZO updates, generating diverse-magnitude updates precisely scaled to layer-wise individual optimization needs. Our results demonstrate that DiZO significantly reduces the needed iterations for convergence without sacrificing throughput, cutting training GPU hours by up to 48\% on various datasets. Moreover, DiZO consistently outperforms the representative ZO baselines in fine-tuning RoBERTa-large, OPT-series, and Llama-series on downstream tasks and, in some cases, even surpasses memory-intensive FO fine-tuning. Our code is released at https://github.com/Skilteee/DiZO.
>
---
#### [replaced 061] Abstraction Alignment: Comparing Model-Learned and Human-Encoded Conceptual Relationships
- **分类: cs.LG; cs.AI; cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2407.12543v3](http://arxiv.org/pdf/2407.12543v3)**

> **作者:** Angie Boggust; Hyemin Bang; Hendrik Strobelt; Arvind Satyanarayan
>
> **备注:** 20 pages, 7 figures, published in CHI 2025
>
> **摘要:** While interpretability methods identify a model's learned concepts, they overlook the relationships between concepts that make up its abstractions and inform its ability to generalize to new data. To assess whether models' have learned human-aligned abstractions, we introduce abstraction alignment, a methodology to compare model behavior against formal human knowledge. Abstraction alignment externalizes domain-specific human knowledge as an abstraction graph, a set of pertinent concepts spanning levels of abstraction. Using the abstraction graph as a ground truth, abstraction alignment measures the alignment of a model's behavior by determining how much of its uncertainty is accounted for by the human abstractions. By aggregating abstraction alignment across entire datasets, users can test alignment hypotheses, such as which human concepts the model has learned and where misalignments recur. In evaluations with experts, abstraction alignment differentiates seemingly similar errors, improves the verbosity of existing model-quality metrics, and uncovers improvements to current human abstractions.
>
---
#### [replaced 062] Where to Search: Measure the Prior-Structured Search Space of LLM Agents
- **分类: cs.AI; cs.CL; cs.LO**

- **链接: [http://arxiv.org/pdf/2510.14846v3](http://arxiv.org/pdf/2510.14846v3)**

> **作者:** Zhuo-Yang Song
>
> **备注:** 11 pages, 4 figures, 1 table
>
> **摘要:** The generate-filter-refine (iterative paradigm) based on large language models (LLMs) has achieved progress in reasoning, programming, and program discovery in AI+Science. However, the effectiveness of search depends on where to search, namely, how to encode the domain prior into an operationally structured hypothesis space. To this end, this paper proposes a compact formal theory that describes and measures LLM-assisted iterative search guided by domain priors. We represent an agent as a fuzzy relation operator on inputs and outputs to capture feasible transitions; the agent is thereby constrained by a fixed safety envelope. To describe multi-step reasoning/search, we weight all reachable paths by a single continuation parameter and sum them to obtain a coverage generating function; this induces a measure of reachability difficulty; and it provides a geometric interpretation of search on the graph induced by the safety envelope. We further provide the simplest testable inferences and validate them via two instantiation. This theory offers a workable language and operational tools to measure agents and their search spaces, proposing a systematic formal description of iterative search constructed by LLMs.
>
---
#### [replaced 063] Recitation over Reasoning: How Cutting-Edge Language Models Can Fail on Elementary School-Level Reasoning Problems?
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.00509v3](http://arxiv.org/pdf/2504.00509v3)**

> **作者:** Kai Yan; Yufei Xu; Zhengyin Du; Xuesong Yao; Zheyu Wang; Xiaowen Guo; Jiecao Chen
>
> **备注:** 24 pages, 3 figures, 13 tables. The paper is accepted at AACL-IJCNLP 2025 (main track), and the latest version adds modifications in camera-ready
>
> **摘要:** The rapid escalation from elementary school-level to frontier problems of the difficulty for LLM benchmarks in recent years have weaved a miracle for researchers that we are only inches away from surpassing human intelligence. However, is the LLMs' remarkable reasoning ability indeed comes from true intelligence by human standards, or are they simply reciting solutions witnessed during training at an Internet level? To study this problem, we propose RoR-Bench, a novel, multi-modal benchmark for detecting LLM's recitation behavior when asked simple reasoning problems but with conditions subtly shifted, and conduct empirical analysis on our benchmark. Surprisingly, we found existing cutting-edge LLMs unanimously exhibits extremely severe recitation behavior; by changing one phrase in the condition, top models such as OpenAI-o1 and DeepSeek-R1 can suffer 60 percent performance loss on elementary school-level arithmetic and reasoning problems. Such findings are a wake-up call to the LLM community that compels us to re-evaluate the true intelligence level of cutting-edge LLMs.
>
---
#### [replaced 064] Teaching According to Talents! Instruction Tuning LLMs with Competence-Aware Curriculum Learning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.13790v2](http://arxiv.org/pdf/2509.13790v2)**

> **作者:** Yangning Li; Tingwei Lu; Yinghui Li; Yankai Chen; Wei-Chieh Huang; Wenhao Jiang; Hui Wang; Hai-Tao Zheng; Philip S. Yu
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** Efficient instruction tuning aims to enhance the ultimate performance of large language models (LLMs) trained on a given instruction dataset. Curriculum learning as a typical data organization strategy has shown preliminary effectiveness in instruction tuning. However, current curriculum tuning methods suffer from the curriculum rigidity, since they rely solely on static heuristic difficulty metrics. These methods fail to adapt to the evolving capabilities of models during training, resulting in a fixed and potentially sub-optimal learning trajectory. To address the issue, Competence-Aware Multi-Perspective cUrriculum inStruction tuning framework termed CAMPUS is proposed. CAMPUS offers several advantages: (1) Dynamic selection for sub-curriculum. (2) Competency-aware adjustment to the curriculum schedule. (3) Multiple difficulty-based scheduling. Extensive experiments prove the superior performance of CAMPUS, compared to other state-of-the-art baselines for efficient instruction tuning.
>
---
#### [replaced 065] Beyond Empathy: Integrating Diagnostic and Therapeutic Reasoning with Large Language Models for Mental Health Counseling
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.15715v2](http://arxiv.org/pdf/2505.15715v2)**

> **作者:** He Hu; Yucheng Zhou; Juzheng Si; Qianning Wang; Hengheng Zhang; Fuji Ren; Fei Ma; Laizhong Cui; Qi Tian
>
> **摘要:** Large language models (LLMs) hold significant potential for mental health support, capable of generating empathetic responses and simulating therapeutic conversations. However, existing LLM-based approaches often lack the clinical grounding necessary for real-world psychological counseling, particularly in explicit diagnostic reasoning aligned with standards like the DSM/ICD and incorporating diverse therapeutic modalities beyond basic empathy or single strategies. To address these critical limitations, we propose PsyLLM, the first large language model designed to systematically integrate both diagnostic and therapeutic reasoning for mental health counseling. To develop PsyLLM, we design a novel automated data synthesis pipeline that processes real-world mental health posts collected from Reddit, where users frequently share psychological distress and seek community support. This pipeline processes real-world mental health posts, generates multi-turn dialogue structures, and leverages LLMs guided by international diagnostic standards (e.g., DSM/ICD) and multiple therapeutic frameworks (e.g., CBT, ACT, psychodynamic) to simulate detailed clinical reasoning processes. Rigorous multi-dimensional filtering ensures the generation of high-quality, clinically aligned dialogue data. In addition, we introduce a new benchmark and evaluation protocol, assessing counseling quality across four key dimensions. Our experiments demonstrate that PsyLLM significantly outperforms state-of-the-art baseline models on this benchmark. The model weights and dataset have been publicly released at https://github.com/Emo-gml/PsyLLM.
>
---
#### [replaced 066] Scaling Latent Reasoning via Looped Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.25741v2](http://arxiv.org/pdf/2510.25741v2)**

> **作者:** Rui-Jie Zhu; Zixuan Wang; Kai Hua; Tianyu Zhang; Ziniu Li; Haoran Que; Boyi Wei; Zixin Wen; Fan Yin; He Xing; Lu Li; Jiajun Shi; Kaijing Ma; Shanda Li; Taylor Kergan; Andrew Smith; Xingwei Qu; Mude Hui; Bohong Wu; Qiyang Min; Hongzhi Huang; Xun Zhou; Wei Ye; Jiaheng Liu; Jian Yang; Yunfeng Shi; Chenghua Lin; Enduo Zhao; Tianle Cai; Ge Zhang; Wenhao Huang; Yoshua Bengio; Jason Eshraghian
>
> **摘要:** Modern LLMs are trained to "think" primarily via explicit text generation, such as chain-of-thought (CoT), which defers reasoning to post-training and under-leverages pre-training data. We present and open-source Ouro, named after the recursive Ouroboros, a family of pre-trained Looped Language Models (LoopLM) that instead build reasoning into the pre-training phase through (i) iterative computation in latent space, (ii) an entropy-regularized objective for learned depth allocation, and (iii) scaling to 7.7T tokens. Ouro 1.4B and 2.6B models enjoy superior performance that match the results of up to 12B SOTA LLMs across a wide range of benchmarks. Through controlled experiments, we show this advantage stems not from increased knowledge capacity, but from superior knowledge manipulation capabilities. We also show that LoopLM yields reasoning traces more aligned with final outputs than explicit CoT. We hope our results show the potential of LoopLM as a novel scaling direction in the reasoning era. Our model is available here: http://ouro-llm.github.io.
>
---
#### [replaced 067] Incivility and Rigidity: Evaluating the Risks of Fine-Tuning LLMs for Political Argumentation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.16813v4](http://arxiv.org/pdf/2411.16813v4)**

> **作者:** Svetlana Churina; Kokil Jaidka
>
> **摘要:** Incivility on platforms such as Twitter (now X) and Reddit complicates the development of AI systems that can support productive, rhetorically sound political argumentation. We present experiments with \textit{GPT-3.5 Turbo} fine-tuned on two contrasting datasets of political discourse: high-incivility Twitter replies to U.S. Congress and low-incivility posts from Reddit's \textit{r/ChangeMyView}. Our evaluation examines how data composition and prompting strategies affect the rhetorical framing and deliberative quality of model-generated arguments. Results show that Reddit-finetuned models generate safer but rhetorically rigid arguments, while cross-platform fine-tuning amplifies adversarial tone and toxicity. Prompt-based steering reduces overt toxicity (e.g., personal attacks) but cannot fully offset the influence of noisy training data. We introduce a rhetorical evaluation rubric - covering justification, reciprocity, alignment, and authority - and provide implementation guidelines for authoring, moderation, and deliberation-support systems.
>
---
#### [replaced 068] Measuring Algorithmic Partisanship via Zero-Shot Classification and Its Implications on Political Discourse
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.01258v2](http://arxiv.org/pdf/2510.01258v2)**

> **作者:** Nathan Junzi Chen
>
> **备注:** 19 pages, 7 figures
>
> **摘要:** Amidst the rapid normalization of generative artificial intelligence (GAI), intelligent systems have come to dominate political discourse across information media. However, internalized political biases stemming from training data skews, human prejudice, and algorithmic flaws continue to plague this novel technology. This study employs a zero-shot classification approach to evaluate algorithmic political partisanship through a methodical combination of ideological alignment, topicality, response sentiment, and objectivity. A total of 1800 model responses across six mainstream large language models (LLMs) were individually input into four distinct fine-tuned classification algorithms, each responsible for computing one of the aforementioned metrics. The results show an amplified liberal-authoritarian alignment across the six LLMs evaluated, with notable instances of reasoning supersessions and canned refusals. The study subsequently highlights the psychological influences underpinning human-computer interactions and how intrinsic biases can permeate public discourse. The resulting distortion of the political landscape can ultimately manifest as conformity or polarization, depending on the region's pre-existing socio-political structures.
>
---
#### [replaced 069] MultiMatch: Multihead Consistency Regularization Matching for Semi-Supervised Text Classification
- **分类: cs.CL; cs.AI; cs.LG; I.2.7**

- **链接: [http://arxiv.org/pdf/2506.07801v3](http://arxiv.org/pdf/2506.07801v3)**

> **作者:** Iustin Sirbu; Robert-Adrian Popovici; Cornelia Caragea; Stefan Trausan-Matu; Traian Rebedea
>
> **备注:** This is the camera-ready version of the paper, accepted for publication in the Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025)
>
> **摘要:** We introduce MultiMatch, a novel semi-supervised learning (SSL) algorithm combining the paradigms of co-training and consistency regularization with pseudo-labeling. At its core, MultiMatch features a pseudo-label weighting module designed for selecting and filtering pseudo-labels based on head agreement and model confidence, and weighting them according to the perceived classification difficulty. This novel module enhances and unifies three existing techniques -- heads agreement from Multihead Co-training, self-adaptive thresholds from FreeMatch, and Average Pseudo-Margins from MarginMatch -- resulting in a holistic approach that improves robustness and performance in SSL settings. Experimental results on benchmark datasets highlight the superior performance of MultiMatch, i.e., MultiMatch achieves state-of-the-art results on 8 out of 10 setups from 5 natural language processing datasets and ranks first according to the Friedman test among 21 methods. Furthermore, MultiMatch demonstrates exceptional robustness in highly imbalanced settings, outperforming the second-best approach by 3.26%, a critical advantage for real-world text classification tasks. Our code is available on GitHub.
>
---
#### [replaced 070] Mafoko: Structuring and Building Open Multilingual Terminologies for South African NLP
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.03529v3](http://arxiv.org/pdf/2508.03529v3)**

> **作者:** Vukosi Marivate; Isheanesu Dzingirai; Fiskani Banda; Richard Lastrucci; Thapelo Sindane; Keabetswe Madumo; Kayode Olaleye; Abiodun Modupe; Unarine Netshifhefhe; Herkulaas Combrink; Mohlatlego Nakeng; Matome Ledwaba
>
> **备注:** Accepted for Sixth Workshop on Resources for African Indigenous Languages (RAIL) 2025
>
> **摘要:** The critical lack of structured terminological data for South Africa's official languages hampers progress in multilingual NLP, despite the existence of numerous government and academic terminology lists. These valuable assets remain fragmented and locked in non-machine-readable formats, rendering them unusable for computational research and development. Mafoko addresses this challenge by systematically aggregating, cleaning, and standardising these scattered resources into open, interoperable datasets. We introduce the foundational Mafoko dataset, released under the equitable, Africa-centered NOODL framework. To demonstrate its immediate utility, we integrate the terminology into a Retrieval-Augmented Generation (RAG) pipeline. Experiments show substantial improvements in the accuracy and domain-specific consistency of English-to-Tshivenda machine translation for large language models. Mafoko provides a scalable foundation for developing robust and equitable NLP technologies, ensuring South Africa's rich linguistic diversity is represented in the digital age.
>
---
#### [replaced 071] Transferring Linear Features Across Language Models With Model Stitching
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.06609v3](http://arxiv.org/pdf/2506.06609v3)**

> **作者:** Alan Chen; Jack Merullo; Alessandro Stolfo; Ellie Pavlick
>
> **摘要:** In this work, we demonstrate that affine mappings between residual streams of language models is a cheap way to effectively transfer represented features between models. We apply this technique to transfer the weights of Sparse Autoencoders (SAEs) between models of different sizes to compare their representations. We find that small and large models learn similar representation spaces, which motivates training expensive components like SAEs on a smaller model and transferring to a larger model at a FLOPs savings. In particular, using a small-to-large transferred SAE as initialization can lead to 50% cheaper training runs when training SAEs on larger models. Next, we show that transferred probes and steering vectors can effectively recover ground truth performance. Finally, we dive deeper into feature-level transferability, finding that semantic and structural features transfer noticeably differently while specific classes of functional features have their roles faithfully mapped. Overall, our findings illustrate similarities and differences in the linear representation spaces of small and large models and demonstrate a method for improving the training efficiency of SAEs.
>
---
#### [replaced 072] Sampling-Efficient Test-Time Scaling: Self-Estimating the Best-of-N Sampling in Early Decoding
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.01422v3](http://arxiv.org/pdf/2503.01422v3)**

> **作者:** Yiming Wang; Pei Zhang; Siyuan Huang; Baosong Yang; Zhuosheng Zhang; Fei Huang; Rui Wang
>
> **备注:** Accepted by NeurIPS 2025 (Spotlight)
>
> **摘要:** Test-time scaling enhances large language model performance by allocating additional compute resources during inference. Best-of-N (BoN) sampling serves as a common sampling-based scaling technique, broadening the search space in parallel to find better solutions from the model distribution. However, its cost-performance trade-off is still underexplored. Two main challenges limit the efficiency of BoN sampling: (1) Generating N full samples consumes substantial GPU memory, reducing inference capacity under limited resources. (2) Reward models add extra memory and latency overhead, and training strong reward models introduces potential training data costs. Although some studies have explored efficiency improvements, none have addressed both challenges at once. To address this gap, we propose Self-Truncation Best-of-N (ST-BoN), a decoding method that avoids fully generating all N samples and eliminates the need for reward models. It leverages early sampling consistency in the model's internal states to identify the most promising path and truncate suboptimal ones. In terms of cost, ST-BoN reduces dynamic GPU memory usage by over 80% and inference latency by 50%. In terms of cost-performance trade-off, ST-BoN achieves the same performance as Full-BoN while saving computational cost by 70%-80%, and under the same cost, it can improve accuracy by 3-4 points.
>
---
#### [replaced 073] Solving Inequality Proofs with Large Language Models
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.07927v2](http://arxiv.org/pdf/2506.07927v2)**

> **作者:** Jiayi Sheng; Luna Lyu; Jikai Jin; Tony Xia; Alex Gu; James Zou; Pan Lu
>
> **备注:** 50 pages, 24 figures, accepted as a Spotlight at NeurIPS 2025
>
> **摘要:** Inequality proving, crucial across diverse scientific and mathematical fields, tests advanced reasoning skills such as discovering tight bounds and strategic theorem application. This makes it a distinct, demanding frontier for large language models (LLMs), offering insights beyond general mathematical problem-solving. Progress in this area is hampered by existing datasets that are often scarce, synthetic, or rigidly formal. We address this by proposing an informal yet verifiable task formulation, recasting inequality proving into two automatically checkable subtasks: bound estimation and relation prediction. Building on this, we release IneqMath, an expert-curated dataset of Olympiad-level inequalities, including a test set and training corpus enriched with step-wise solutions and theorem annotations. We also develop a novel LLM-as-judge evaluation framework, combining a final-answer judge with four step-wise judges designed to detect common reasoning flaws. A systematic evaluation of 29 leading LLMs on IneqMath reveals a surprising reality: even top models like o1 achieve less than 10% overall accuracy under step-wise scrutiny; this is a drop of up to 65.5% from their accuracy considering only final answer equivalence. This discrepancy exposes fragile deductive chains and a critical gap for current LLMs between merely finding an answer and constructing a rigorous proof. Scaling model size and increasing test-time computation yield limited gains in overall proof correctness. Instead, our findings highlight promising research directions such as theorem-guided reasoning and self-refinement. Code and data are available at https://ineqmath.github.io/.
>
---
#### [replaced 074] MedREK: Retrieval-Based Editing for Medical LLMs with Key-Aware Prompts
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.13500v2](http://arxiv.org/pdf/2510.13500v2)**

> **作者:** Shujun Xia; Haokun Lin; Yichen Wu; Yinan Zhou; Zixuan Li; Zhongwei Wan; Xingrun Xing; Yefeng Zheng; Xiang Li; Caifeng Shan; Zhenan Sun; Quanzheng Li
>
> **备注:** Preprint, work in progress
>
> **摘要:** LLMs hold great promise for healthcare applications, but the rapid evolution of medical knowledge and errors in training data often cause them to generate outdated or inaccurate information, limiting their applicability in high-stakes clinical practice. Model editing has emerged as a potential remedy without full retraining. While parameter-based editing often compromises locality and is thus ill-suited for the medical domain, retrieval-based editing offers a more viable alternative. However, it still faces two critical challenges: (1) representation overlap within the medical knowledge space often causes inaccurate retrieval and reduces editing accuracy; (2) existing methods are restricted to single-sample edits, while batch-editing remains largely unexplored despite its importance for real-world medical applications. To address these challenges, we first construct MedVersa, an enhanced benchmark with broader coverage of medical subjects, designed to evaluate both single and batch edits under strict locality constraints. We then propose MedREK, a retrieval-based editing framework that integrates a shared query-key module for precise matching with an attention-based prompt encoder for informative guidance. Experimental results on various medical benchmarks demonstrate that our MedREK achieves superior performance across different core metrics and provides the first validated solution for batch-editing in medical LLMs. Our code and dataset are available at https://github.com/mylittleriver/MedREK.
>
---
#### [replaced 075] Retrieval-Augmented Defense: Adaptive and Controllable Jailbreak Prevention for Large Language Models
- **分类: cs.CR; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.16406v2](http://arxiv.org/pdf/2508.16406v2)**

> **作者:** Guangyu Yang; Jinghong Chen; Jingbiao Mei; Weizhe Lin; Bill Byrne
>
> **摘要:** Large Language Models (LLMs) remain vulnerable to jailbreak attacks, which attempt to elicit harmful responses from LLMs. The evolving nature and diversity of these attacks pose many challenges for defense systems, including (1) adaptation to counter emerging attack strategies without costly retraining, and (2) control of the trade-off between safety and utility. To address these challenges, we propose Retrieval-Augmented Defense (RAD), a novel framework for jailbreak detection that incorporates a database of known attack examples into Retrieval-Augmented Generation, which is used to infer the underlying, malicious user query and jailbreak strategy used to attack the system. RAD enables training-free updates for newly discovered jailbreak strategies and provides a mechanism to balance safety and utility. Experiments on StrongREJECT show that RAD substantially reduces the effectiveness of strong jailbreak attacks such as PAP and PAIR while maintaining low rejection rates for benign queries. We propose a novel evaluation scheme and show that RAD achieves a robust safety-utility trade-off across a range of operating points in a controllable manner.
>
---
#### [replaced 076] MaiBaam Annotation Guidelines
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2403.05902v3](http://arxiv.org/pdf/2403.05902v3)**

> **作者:** Verena Blaschke; Barbara Kovačić; Siyao Peng; Barbara Plank
>
> **备注:** Updated for UD v2.17
>
> **摘要:** This document provides the annotation guidelines for MaiBaam, a Bavarian corpus manually annotated with part-of-speech (POS) tags, syntactic dependencies, and German lemmas. MaiBaam belongs to the Universal Dependencies (UD) project, and our annotations elaborate on the general and German UD version 2 guidelines. In this document, we detail how to preprocess and tokenize Bavarian data, provide an overview of the POS tags and dependencies we use, explain annotation decisions that would also apply to closely related languages like German, and lastly we introduce and motivate decisions that are specific to Bavarian grammar.
>
---
#### [replaced 077] Natural Language Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.16728v3](http://arxiv.org/pdf/2503.16728v3)**

> **作者:** Emiel van Miltenburg; Chenghua Lin
>
> **备注:** 4 pages + references. Submitted for publication in the Encyclopedia of Language & Linguistics
>
> **摘要:** This article provides a brief overview of the field of Natural Language Generation. The term Natural Language Generation (NLG), in its broadest definition, refers to the study of systems that verbalize some form of information through natural language. That information could be stored in a large database or knowledge graph (in data-to-text applications), but NLG researchers may also study summarisation (text-to-text) or image captioning (image-to-text), for example. As a subfield of Natural Language Processing, NLG is closely related to other sub-disciplines such as Machine Translation (MT) and Dialog Systems. Some NLG researchers exclude MT from their definition of the field, since there is no content selection involved where the system has to determine what to say. Conversely, dialog systems do not typically fall under the header of Natural Language Generation since NLG is just one component of dialog systems (the others being Natural Language Understanding and Dialog Management). However, with the rise of Large Language Models (LLMs), different subfields of Natural Language Processing have converged on similar methodologies for the production of natural language and the evaluation of automatically generated text.
>
---
#### [replaced 078] What Features in Prompts Jailbreak LLMs? Investigating the Mechanisms Behind Attacks
- **分类: cs.CR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2411.03343v3](http://arxiv.org/pdf/2411.03343v3)**

> **作者:** Nathalie Kirch; Constantin Weisser; Severin Field; Helen Yannakoudakis; Stephen Casper
>
> **摘要:** Jailbreaks have been a central focus of research regarding the safety and reliability of large language models (LLMs), yet the mechanisms underlying these attacks remain poorly understood. While previous studies have predominantly relied on linear methods to detect jailbreak attempts and model refusals, we take a different approach by examining both linear and non-linear features in prompts that lead to successful jailbreaks. First, we introduce a novel dataset comprising 10,800 jailbreak attempts spanning 35 diverse attack methods. Leveraging this dataset, we train linear and non-linear probes on hidden states of open-weight LLMs to predict jailbreak success. Probes achieve strong in-distribution accuracy but transfer is attack-family-specific, revealing that different jailbreaks are supported by distinct internal mechanisms rather than a single universal direction. To establish causal relevance, we construct probe-guided latent interventions that systematically shift compliance in the predicted direction. Interventions derived from non-linear probes produce larger and more reliable effects than those from linear probes, indicating that features linked to jailbreak success are encoded non-linearly in prompt representations. Overall, the results surface heterogeneous, non-linear structure in jailbreak mechanisms and provide a prompt-side methodology for recovering and testing the features that drive jailbreak outcomes.
>
---
#### [replaced 079] Instructing Large Language Models for Low-Resource Languages: A Systematic Study for Basque
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.07597v2](http://arxiv.org/pdf/2506.07597v2)**

> **作者:** Oscar Sainz; Naiara Perez; Julen Etxaniz; Joseba Fernandez de Landa; Itziar Aldabe; Iker García-Ferrero; Aimar Zabala; Ekhi Azurmendi; German Rigau; Eneko Agirre; Mikel Artetxe; Aitor Soroa
>
> **备注:** Accepted at EMNLP 2025 Main Conference
>
> **摘要:** Instructing language models with user intent requires large instruction datasets, which are only available for a limited set of languages. In this paper, we explore alternatives to conventional instruction adaptation pipelines in low-resource scenarios. We assume a realistic scenario for low-resource languages, where only the following are available: corpora in the target language, existing open-weight multilingual base and instructed backbone LLMs, and synthetically generated instructions sampled from the instructed backbone. We present a comprehensive set of experiments for Basque that systematically study different combinations of these components evaluated on benchmarks and human preferences from 1,680 participants. Our conclusions show that target language corpora are essential, with synthetic instructions yielding robust models, and, most importantly, that using as backbone an instruction-tuned model outperforms using a base non-instructed model. Scaling up to Llama 3.1 Instruct 70B as backbone, our model comes near frontier models of much larger sizes for Basque, without using any Basque instructions. We release code, models, instruction datasets, and human preferences to support full reproducibility in future research on low-resource language adaptation. https://github.com/hitz-zentroa/latxa-instruct
>
---
#### [replaced 080] Eye Tracking Based Cognitive Evaluation of Automatic Readability Assessment Measures
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11150v3](http://arxiv.org/pdf/2502.11150v3)**

> **作者:** Keren Gruteke Klein; Shachar Frenkel; Omer Shubi; Yevgeni Berzak
>
> **摘要:** Methods for scoring text readability have been studied for over a century, and are widely used in research and in user-facing applications in many domains. Thus far, the development and evaluation of such methods have primarily relied on two types of offline behavioral data, performance on reading comprehension tests and ratings of text readability levels. In this work, we instead focus on a fundamental and understudied aspect of readability, real-time reading ease, captured with online reading measures using eye tracking. We introduce an evaluation framework for readability scoring methods which quantifies their ability to account for reading ease, while controlling for content variation across texts. Applying this evaluation to prominent traditional readability formulas, modern machine learning systems, frontier Large Language Models and commercial systems used in education, suggests that they are all poor predictors of reading ease in English. This outcome holds across native and non-native speakers, reading regimes, and textual units of different lengths. The evaluation further reveals that existing methods are often outperformed by word properties commonly used in psycholinguistics for prediction of reading times. Our results highlight a fundamental limitation of existing approaches to readability scoring, the utility of psycholinguistics for readability research, and the need for new, cognitively driven readability scoring approaches that can better account for reading ease.
>
---
#### [replaced 081] Beyond Pointwise Scores: Decomposed Criteria-Based Evaluation of LLM Responses
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.16093v2](http://arxiv.org/pdf/2509.16093v2)**

> **作者:** Fangyi Yu; Nabeel Seedat; Dasha Herrmannova; Frank Schilder; Jonathan Richard Schwarz
>
> **备注:** Accepted by 2025 EMNLP industry track
>
> **摘要:** Evaluating long-form answers in high-stakes domains such as law or medicine remains a fundamental challenge. Standard metrics like BLEU and ROUGE fail to capture semantic correctness, and current LLM-based evaluators often reduce nuanced aspects of answer quality into a single undifferentiated score. We introduce DeCE, a decomposed LLM evaluation framework that separates precision (factual accuracy and relevance) and recall (coverage of required concepts), using instance-specific criteria automatically extracted from gold answer requirements. DeCE is model-agnostic and domain-general, requiring no predefined taxonomies or handcrafted rubrics. We instantiate DeCE to evaluate different LLMs on a real-world legal QA task involving multi-jurisdictional reasoning and citation grounding. DeCE achieves substantially stronger correlation with expert judgments ($r=0.78$), compared to traditional metrics ($r=0.12$), pointwise LLM scoring ($r=0.35$), and modern multidimensional evaluators ($r=0.48$). It also reveals interpretable trade-offs: generalist models favor recall, while specialized models favor precision. Importantly, only 11.95% of LLM-generated criteria required expert revision, underscoring DeCE's scalability. DeCE offers an interpretable and actionable LLM evaluation framework in expert domains.
>
---
#### [replaced 082] A Closer Look at Bias and Chain-of-Thought Faithfulness of Large (Vision) Language Models
- **分类: cs.CL; cs.AI; I.2.10; I.2.7**

- **链接: [http://arxiv.org/pdf/2505.23945v2](http://arxiv.org/pdf/2505.23945v2)**

> **作者:** Sriram Balasubramanian; Samyadeep Basu; Soheil Feizi
>
> **备注:** Accepted in EMNLP 2025, 34 pages, 25 figures
>
> **摘要:** Chain-of-thought (CoT) reasoning enhances performance of large language models, but questions remain about whether these reasoning traces faithfully reflect the internal processes of the model. We present the first comprehensive study of CoT faithfulness in large vision-language models (LVLMs), investigating how both text-based and previously unexplored image-based biases affect reasoning and bias articulation. Our work introduces a novel, fine-grained evaluation pipeline for categorizing bias articulation patterns, enabling significantly more precise analysis of CoT reasoning than previous methods. This framework reveals critical distinctions in how models process and respond to different types of biases, providing new insights into LVLM CoT faithfulness. Our findings reveal that subtle image-based biases are rarely articulated compared to explicit text-based ones, even in models specialized for reasoning. Additionally, many models exhibit a previously unidentified phenomenon we term ``inconsistent'' reasoning - correctly reasoning before abruptly changing answers, serving as a potential canary for detecting biased reasoning from unfaithful CoTs. We then apply the same evaluation pipeline to revisit CoT faithfulness in LLMs across various levels of implicit cues. Our findings reveal that current language-only reasoning models continue to struggle with articulating cues that are not overtly stated.
>
---
#### [replaced 083] Evaluating Perspectival Biases in Cross-Modal Retrieval
- **分类: cs.IR; cs.CL; H.3.3; I.2.7; I.2.10**

- **链接: [http://arxiv.org/pdf/2510.26861v2](http://arxiv.org/pdf/2510.26861v2)**

> **作者:** Teerapol Saengsukhiran; Peerawat Chomphooyod; Narabodee Rodjananant; Chompakorn Chaksangchaichot; Patawee Prakrankamanant; Witthawin Sripheanpol; Pak Lovichit; Sarana Nutanong; Ekapol Chuangsuwanich
>
> **摘要:** Multimodal retrieval systems are expected to operate in a semantic space, agnostic to the language or cultural origin of the query. In practice, however, retrieval outcomes systematically reflect perspectival biases: deviations shaped by linguistic prevalence and cultural associations. We study two such biases. First, prevalence bias refers to the tendency to favor entries from prevalent languages over semantically faithful entries in image-to-text retrieval. Second, association bias refers to the tendency to favor images culturally associated with the query over semantically correct ones in text-to-image retrieval. Results show that explicit alignment is a more effective strategy for mitigating prevalence bias. However, association bias remains a distinct and more challenging problem. These findings suggest that achieving truly equitable multimodal systems requires targeted strategies beyond simple data scaling and that bias arising from cultural association may be treated as a more challenging problem than one arising from linguistic prevalence.
>
---
#### [replaced 084] Verbalized Algorithms
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.08150v2](http://arxiv.org/pdf/2509.08150v2)**

> **作者:** Supriya Lall; Christian Farrell; Hari Pathanjaly; Marko Pavic; Sarvesh Chezhian; Masataro Asai
>
> **备注:** Accepted in NeurIPS 2025 Workshop on Efficient Reasoning
>
> **摘要:** Instead of querying LLMs in a one-shot manner and hoping to get the right answer for a reasoning task, we propose a paradigm we call \emph{verbalized algorithms} (VAs), which leverage classical algorithms with established theoretical understanding. VAs decompose a task into simple elementary operations on natural language strings that they should be able to answer reliably, and limit the scope of LLMs to only those simple tasks. For example, for sorting a series of natural language strings, \emph{verbalized sorting} uses an LLM as a binary comparison oracle in a known and well-analyzed sorting algorithm (e.g., bitonic sorting network). We demonstrate the effectiveness of this approach on sorting and clustering tasks.
>
---
#### [replaced 085] PolyMath: Evaluating Mathematical Reasoning in Multilingual Contexts
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.18428v4](http://arxiv.org/pdf/2504.18428v4)**

> **作者:** Yiming Wang; Pei Zhang; Jialong Tang; Haoran Wei; Baosong Yang; Rui Wang; Chenshu Sun; Feitong Sun; Jiran Zhang; Junxuan Wu; Qiqian Cang; Yichang Zhang; Fei Huang; Junyang Lin; Fei Huang; Jingren Zhou
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** In this paper, we introduce PolyMath, a multilingual mathematical reasoning benchmark covering 18 languages and 4 easy-to-hard difficulty levels. Our benchmark ensures difficulty comprehensiveness, language diversity, and high-quality translation, making it a highly discriminative multilingual mathematical benchmark in the era of reasoning LLMs. We conduct a comprehensive evaluation for advanced LLMs and find that even Qwen-3-235B-A22B-Thinking and Gemini-2.5-pro, achieve only 54.6 and 52.2 benchmark scores, with about 40% accuracy under the highest level From a language perspective, our benchmark reveals several key challenges of LLMs in multilingual reasoning: (1) Reasoning performance varies widely across languages for current LLMs; (2) Input-output language consistency is low in reasoning LLMs and may be correlated with performance; (3) The thinking length differs significantly by language for current LLMs. Additionally, we demonstrate that controlling the output language in the instructions has the potential to affect reasoning performance, especially for some low-resource languages, suggesting a promising direction for improving multilingual capabilities in LLMs.
>
---
#### [replaced 086] KIT's Low-resource Speech Translation Systems for IWSLT2025: System Enhancement with Synthetic Data and Model Regularization
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.19679v2](http://arxiv.org/pdf/2505.19679v2)**

> **作者:** Zhaolin Li; Yining Liu; Danni Liu; Tuan Nam Nguyen; Enes Yavuz Ugan; Tu Anh Dinh; Carlos Mullov; Alexander Waibel; Jan Niehues
>
> **摘要:** This paper presents KIT's submissions to the IWSLT 2025 low-resource track. We develop both cascaded systems, consisting of Automatic Speech Recognition (ASR) and Machine Translation (MT) models, and end-to-end (E2E) Speech Translation (ST) systems for three language pairs: Bemba, North Levantine Arabic, and Tunisian Arabic into English. Building upon pre-trained models, we fine-tune our systems with different strategies to utilize resources efficiently. This study further explores system enhancement with synthetic data and model regularization. Specifically, we investigate MT-augmented ST by generating translations from ASR data using MT models. For North Levantine, which lacks parallel ST training data, a system trained solely on synthetic data slightly surpasses the cascaded system trained on real data. We also explore augmentation using text-to-speech models by generating synthetic speech from MT data, demonstrating the benefits of synthetic data in improving both ASR and ST performance for Bemba. Additionally, we apply intra-distillation to enhance model performance. Our experiments show that this approach consistently improves results across ASR, MT, and ST tasks, as well as across different pre-trained models. Finally, we apply Minimum Bayes Risk decoding to combine the cascaded and end-to-end systems, achieving an improvement of approximately 1.5 BLEU points.
>
---
#### [replaced 087] CausalARC: Abstract Reasoning with Causal World Models
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.03636v2](http://arxiv.org/pdf/2509.03636v2)**

> **作者:** Jacqueline Maasch; John Kalantari; Kia Khezeli
>
> **备注:** Peer-reviewed workshop paper
>
> **摘要:** On-the-fly reasoning often requires adaptation to novel problems under limited data and distribution shift. This work introduces CausalARC: an experimental testbed for AI reasoning in low-data and out-of-distribution regimes, modeled after the Abstraction and Reasoning Corpus (ARC). Each CausalARC reasoning task is sampled from a fully specified causal world model, formally expressed as a structural causal model. Principled data augmentations provide observational, interventional, and counterfactual feedback about the world model in the form of few-shot, in-context learning demonstrations. As a proof-of-concept, we illustrate the use of CausalARC for four language model evaluation settings: (1) abstract reasoning with test-time training, (2) counterfactual reasoning with in-context learning, (3) program synthesis, and (4) causal discovery with logical reasoning. Within- and between-model performance varied heavily across tasks, indicating room for significant improvement in language model reasoning.
>
---
#### [replaced 088] Language Native Lightly Structured Databases for Large Language Model Driven Composite Materials Research
- **分类: cs.DB; cond-mat.mtrl-sci; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.06093v2](http://arxiv.org/pdf/2509.06093v2)**

> **作者:** Yuze Liu; Zhaoyuan Zhang; Xiangsheng Zeng; Yihe Zhang; Leping Yu; Lejia Wang; Xi Yu
>
> **摘要:** The preparation procedures of materials are often embedded narratively in experimental protocols, research articles, patents, and laboratory notes, and are structured around procedural sequences, causal relationships, and conditional logic. The synthesis of boron nitride nanosheet (BNNS) polymer composites exemplifies this linguistically encoded decision-making system, where the practical experiments involve interdependent multistage and path-dependent processes such as exfoliation, functionalization, and dispersion, each governed by heterogeneous parameters and contextual contingencies, challenging conventional numerical optimization paradigms for experiment design. We reformulate this challenge into a text-reasoning problem through a framework centered on a text-first, lightly structured materials database and large language models (LLMs) as text reasoning engines. We constructed a database that captures evidence-linked narrative excerpts from the literature while normalizing only the minimum necessary entities, attributes, and relations to enable composite retrieval that unifies semantic matching, lexical cues, and explicit value filters. Building on this language-native, provenance-preserving foundation, the LLM operates in two complementary modes: retrieval-augmented generation (RAG), grounding outputs in retrieved evidence modules from the database, and experience-augmented reasoning (EAR), which leverages iteratively trained text guides derived from multi-source literature-based narrative data as external references to inform reasoning and decision-making. Applying this integration-and-reasoning framework, we demonstrate rapid, laboratory-scale optimization of BNNS preparation, highlighting how language-native data combined with LLM-based reasoning can significantly accelerate practical material preparation.
>
---
#### [replaced 089] Deep Video Discovery: Agentic Search with Tool Use for Long-form Video Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.18079v4](http://arxiv.org/pdf/2505.18079v4)**

> **作者:** Xiaoyi Zhang; Zhaoyang Jia; Zongyu Guo; Jiahao Li; Bin Li; Houqiang Li; Yan Lu
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Long-form video understanding presents significant challenges due to extensive temporal-spatial complexity and the difficulty of question answering under such extended contexts. While Large Language Models (LLMs) have demonstrated considerable advancements in video analysis capabilities and long context handling, they continue to exhibit limitations when processing information-dense hour-long videos. To overcome such limitations, we propose the Deep Video Discovery (DVD) agent to leverage an agentic search strategy over segmented video clips. Unlike previous video agents that rely on predefined workflows applied uniformly across different queries, our approach emphasizes the autonomous and adaptive nature of agents. By providing a set of search-centric tools on multi-granular video database, our DVD agent leverages the advanced reasoning capability of LLM to plan on its current observation state, strategically selects tools to orchestrate adaptive workflow for different queries in light of the gathered information. We perform comprehensive evaluation on multiple long video understanding benchmarks that demonstrates our advantage. Our DVD agent achieves state-of-the-art performance on the challenging LVBench dataset, reaching an accuracy of 74.2%, which substantially surpasses all prior works, and further improves to 76.0% with transcripts. The code has been released at https://github.com/microsoft/DeepVideoDiscovery.
>
---
#### [replaced 090] JudgeLRM: Large Reasoning Models as a Judge
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.00050v3](http://arxiv.org/pdf/2504.00050v3)**

> **作者:** Nuo Chen; Zhiyuan Hu; Qingyun Zou; Jiaying Wu; Qian Wang; Bryan Hooi; Bingsheng He
>
> **备注:** Preprint
>
> **摘要:** Large Language Models (LLMs) are increasingly adopted as evaluators, offering a scalable alternative to human annotation. However, existing supervised fine-tuning (SFT) approaches often fall short in domains that demand complex reasoning. Judgment is inherently reasoning-intensive: beyond surface-level scoring, it requires verifying evidence, identifying errors, and justifying decisions. Through the analysis of evaluation tasks, we find a negative correlation between SFT performance gains and the proportion of reasoning-demanding samples, revealing the limits of SFT in such scenarios. To address this, we introduce JudgeLRM, a family of judgment-oriented LLMs, trained using reinforcement learning (RL) with judge-wise, outcome-driven rewards to activate reasoning capabilities. JudgeLRM consistently outperform SFT-tuned baselines in the same size, as well as other RL and SFT variants, and even surpass state-of-the-art reasoning models: notably, JudgeLRM-3B/4B exceeds GPT-4, while JudgeLRM-7B/8B/14B outperforms DeepSeek-R1 by over 2% in F1 score, with particularly strong gains on reasoning-heavy tasks. Our findings underscore the value of RL in unlocking reasoning-aligned LLM judges.
>
---
#### [replaced 091] XIFBench: Evaluating Large Language Models on Multilingual Instruction Following
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.07539v2](http://arxiv.org/pdf/2503.07539v2)**

> **作者:** Zhenyu Li; Kehai Chen; Yunfei Long; Xuefeng Bai; Yaoyin Zhang; Xuchen Wei; Juntao Li; Min Zhang
>
> **备注:** Accepted by the NeurIPS 2025 Datasets and Benchmarks Track
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable instruction-following capabilities across various applications. However, their performance in multilingual settings lacks systematic investigation, with existing evaluations lacking fine-grained constraint analysis across diverse linguistic contexts. We introduce XIFBench, a comprehensive constraint-based benchmark for evaluating multilingual instruction-following abilities of LLMs, comprising 558 instructions with 0-5 additional constraints across five categories (Content, Style, Situation, Format, and Numerical) in six languages spanning different resource levels. To support reliable and consistent cross-lingual evaluation, we implement three methodological innovations: cultural accessibility annotation, constraint-level translation validation, and requirement-based evaluation using English requirements as semantic anchors across languages. Extensive experiments with various LLMs not only quantify performance disparities across resource levels but also provide detailed insights into how language resources, constraint categories, instruction complexity, and cultural specificity influence multilingual instruction-following. Our code and data are available at https://github.com/zhenyuli801/XIFBench.
>
---
#### [replaced 092] Scientists' First Exam: Probing Cognitive Abilities of MLLM via Perception, Understanding, and Reasoning
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.10521v5](http://arxiv.org/pdf/2506.10521v5)**

> **作者:** Yuhao Zhou; Yiheng Wang; Xuming He; Ao Shen; Ruoyao Xiao; Zhiwei Li; Qiantai Feng; Zijie Guo; Yuejin Yang; Hao Wu; Wenxuan Huang; Jiaqi Wei; Dan Si; Xiuqi Yao; Jia Bu; Haiwen Huang; Manning Wang; Tianfan Fu; Shixiang Tang; Ben Fei; Dongzhan Zhou; Fenghua Ling; Yan Lu; Siqi Sun; Chenhui Li; Guanjie Zheng; Jiancheng Lv; Wenlong Zhang; Lei Bai
>
> **备注:** 82 pages
>
> **摘要:** Scientific discoveries increasingly rely on complex multimodal reasoning based on information-intensive scientific data and domain-specific expertise. Empowered by expert-level scientific benchmarks, scientific Multimodal Large Language Models (MLLMs) hold the potential to significantly enhance this discovery process in realistic workflows. However, current scientific benchmarks mostly focus on evaluating the knowledge understanding capabilities of MLLMs, leading to an inadequate assessment of their perception and reasoning abilities. To address this gap, we present the Scientists' First Exam (SFE) benchmark, designed to evaluate the scientific cognitive capacities of MLLMs through three interconnected levels: scientific signal perception, scientific attribute understanding, scientific comparative reasoning. Specifically, SFE comprises 830 expert-verified VQA pairs across three question types, spanning 66 multimodal tasks across five high-value disciplines. Extensive experiments reveal that current state-of-the-art GPT-o3 and InternVL-3 achieve only 34.08% and 26.52% on SFE, highlighting significant room for MLLMs to improve in scientific realms. We hope the insights obtained in SFE will facilitate further developments in AI-enhanced scientific discoveries.
>
---
#### [replaced 093] Stable but Miscalibrated: A Kantian View on Overconfidence from Filters to Large Language Models
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.14925v2](http://arxiv.org/pdf/2510.14925v2)**

> **作者:** Akira Okutomi
>
> **备注:** 21 pages, 2 figures, preliminary version
>
> **摘要:** We reinterpret Kant's Critique of Pure Reason as a theory of feedback stability, viewing reason as a regulator that keeps inference within the bounds of possible experience. We formalize this intuition via a composite instability index (H-Risk) combining spectral margin, conditioning, temporal sensitivity, and innovation amplification. In linear-Gaussian simulations, higher H-Risk predicts overconfident errors even under formal stability, revealing a gap between nominal and epistemic stability. Extending to large language models (LLMs), we observe preliminary correlations between internal fragility and miscalibration or hallucination (confabulation), and find that lightweight critique prompts may modestly improve or worsen calibration in small-scale tests. These results suggest a structural bridge between Kantian self-limitation and feedback control, offering a principled lens to diagnose and potentially mitigate overconfidence in reasoning systems.
>
---
#### [replaced 094] Advancing Expert Specialization for Better MoE
- **分类: cs.CL; 68T07; I.2.7**

- **链接: [http://arxiv.org/pdf/2505.22323v3](http://arxiv.org/pdf/2505.22323v3)**

> **作者:** Hongcan Guo; Haolang Lu; Guoshun Nan; Bolun Chu; Jialin Zhuang; Yuan Yang; Wenhao Che; Sicong Leng; Qimei Cui; Xudong Jiang
>
> **备注:** 33pages, 6figures(Accepted by Neurips 2025 Oral)
>
> **摘要:** Mixture-of-Experts (MoE) models enable efficient scaling of large language models (LLMs) by activating only a subset of experts per input. However, we observe that the commonly used auxiliary load balancing loss often leads to expert overlap and overly uniform routing, which hinders expert specialization and degrades overall performance during post-training. To address this, we propose a simple yet effective solution that introduces two complementary objectives: (1) an orthogonality loss to encourage experts to process distinct types of tokens, and (2) a variance loss to encourage more discriminative routing decisions. Gradient-level analysis demonstrates that these objectives are compatible with the existing auxiliary loss and contribute to optimizing the training process. Experimental results over various model architectures and across multiple benchmarks show that our method significantly enhances expert specialization. Notably, our method improves classic MoE baselines with auxiliary loss by up to 23.79%, while also maintaining load balancing in downstream tasks, without any architectural modifications or additional components. We will release our code to contribute to the community.
>
---
#### [replaced 095] Comprehensive and Efficient Distillation for Lightweight Sentiment Analysis Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.24425v2](http://arxiv.org/pdf/2510.24425v2)**

> **作者:** Guangyu Xie; Yice Zhang; Jianzhu Bao; Qianlong Wang; Yang Sun; Bingbing Wang; Ruifeng Xu
>
> **备注:** Accepted by EMNLP 2025. 22 pages, 9 figures. The first two authors contribute equally
>
> **摘要:** Recent efforts leverage knowledge distillation techniques to develop lightweight and practical sentiment analysis models. These methods are grounded in human-written instructions and large-scale user texts. Despite the promising results, two key challenges remain: (1) manually written instructions are limited in diversity and quantity, making them insufficient to ensure comprehensive coverage of distilled knowledge; (2) large-scale user texts incur high computational cost, hindering the practicality of these methods. To this end, we introduce CompEffDist, a comprehensive and efficient distillation framework for sentiment analysis. Our framework consists of two key modules: attribute-based automatic instruction construction and difficulty-based data filtering, which correspondingly tackle the aforementioned challenges. Applying our method across multiple model series (Llama-3, Qwen-3, and Gemma-3), we enable 3B student models to match the performance of 20x larger teacher models on most tasks. In addition, our approach greatly outperforms baseline methods in data efficiency, attaining the same performance level with only 10% of the data.
>
---
#### [replaced 096] Context Tuning for In-Context Optimization
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.04221v2](http://arxiv.org/pdf/2507.04221v2)**

> **作者:** Jack Lu; Ryan Teehan; Zhenbang Yang; Mengye Ren
>
> **备注:** A short version of this paper was accepted at ICML 2025 Workshop on Test-Time Adaptation
>
> **摘要:** We introduce Context Tuning, a simple and effective method to significantly enhance few-shot adaptation of language models (LLMs) without fine-tuning model parameters. While prompt-based adaptation techniques have demonstrated the effectiveness of lightweight adaptation methods for LLMs, they typically initialize a trainable prompt or prefix with irrelevant tokens for the task at hand. In contrast, Context Tuning initializes the trainable prompt or prefix with task-specific demonstration examples, leveraging the model's inherent In-Context Learning (ICL) ability to extract relevant information for improved few-shot learning performance. Extensive evaluations on benchmarks such as CrossFit, UnifiedQA, MMLU, BIG-Bench Hard, and ARC demonstrate that Context Tuning outperforms traditional prompt-based adaptation methods and achieves competitive accuracy to Test-Time Training with significantly higher training efficiency.
>
---
#### [replaced 097] SafeDialBench: A Fine-Grained Safety Benchmark for Large Language Models in Multi-Turn Dialogues with Diverse Jailbreak Attacks
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.11090v3](http://arxiv.org/pdf/2502.11090v3)**

> **作者:** Hongye Cao; Yanming Wang; Sijia Jing; Ziyue Peng; Zhixin Bai; Zhe Cao; Meng Fang; Fan Feng; Boyan Wang; Jiaheng Liu; Tianpei Yang; Jing Huo; Yang Gao; Fanyu Meng; Xi Yang; Chao Deng; Junlan Feng
>
> **摘要:** With the rapid advancement of Large Language Models (LLMs), the safety of LLMs has been a critical concern requiring precise assessment. Current benchmarks primarily concentrate on single-turn dialogues or a single jailbreak attack method to assess the safety. Additionally, these benchmarks have not taken into account the LLM's capability of identifying and handling unsafe information in detail. To address these issues, we propose a fine-grained benchmark SafeDialBench for evaluating the safety of LLMs across various jailbreak attacks in multi-turn dialogues. Specifically, we design a two-tier hierarchical safety taxonomy that considers 6 safety dimensions and generates more than 4000 multi-turn dialogues in both Chinese and English under 22 dialogue scenarios. We employ 7 jailbreak attack strategies, such as reference attack and purpose reverse, to enhance the dataset quality for dialogue generation. Notably, we construct an innovative assessment framework of LLMs, measuring capabilities in detecting, and handling unsafe information and maintaining consistency when facing jailbreak attacks. Experimental results across 17 LLMs reveal that Yi-34B-Chat and GLM4-9B-Chat demonstrate superior safety performance, while Llama3.1-8B-Instruct and o3-mini exhibit safety vulnerabilities.
>
---
#### [replaced 098] Enhancing Reasoning Abilities of Small LLMs with Cognitive Alignment
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.09802v2](http://arxiv.org/pdf/2504.09802v2)**

> **作者:** Wenrui Cai; Chengyu Wang; Junbing Yan; Jun Huang; Xiangzhong Fang
>
> **备注:** emnlp 2025 main conference
>
> **摘要:** The reasoning capabilities of large reasoning models (LRMs), such as OpenAI's o1 and DeepSeek-R1, have seen substantial advancements through deep thinking. However, these enhancements come with significant resource demands, underscoring the need for training effective small reasoning models. A critical challenge is that small models possess different reasoning capacities and cognitive trajectories compared with their larger counterparts. Hence, directly distilling chain-of-thought (CoT) rationales from large LRMs to smaller ones can sometimes be ineffective and often requires a substantial amount of annotated data. In this paper, we first introduce a novel Critique-Rethink-Verify (CRV) system, designed for training smaller yet powerful LRMs. Our CRV system consists of multiple LLM agents, each specializing in unique tasks: (i) critiquing the CoT rationales according to the cognitive capabilities of smaller models, (ii) rethinking and refining these CoTs based on the critiques, and (iii) verifying the correctness of the refined results. Building on the CRV system, we further propose the Cognitive Preference Optimization (CogPO) algorithm to continuously enhance the reasoning abilities of smaller models by aligning their reasoning processes with their cognitive capacities. Comprehensive evaluations on challenging reasoning benchmarks demonstrate the efficacy of our CRV+CogPO framework, which outperforms other methods by a large margin.
>
---
#### [replaced 099] OpinioRAG: Towards Generating User-Centric Opinion Highlights from Large-scale Online Reviews
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2509.00285v2](http://arxiv.org/pdf/2509.00285v2)**

> **作者:** Mir Tafseer Nayeem; Davood Rafiei
>
> **备注:** COLM 2025
>
> **摘要:** We study the problem of opinion highlights generation from large volumes of user reviews, often exceeding thousands per entity, where existing methods either fail to scale or produce generic, one-size-fits-all summaries that overlook personalized needs. To tackle this, we introduce OpinioRAG, a scalable, training-free framework that combines RAG-based evidence retrieval with LLMs to efficiently produce tailored summaries. Additionally, we propose novel reference-free verification metrics designed for sentiment-rich domains, where accurately capturing opinions and sentiment alignment is essential. These metrics offer a fine-grained, context-sensitive assessment of factual consistency. To facilitate evaluation, we contribute the first large-scale dataset of long-form user reviews, comprising entities with over a thousand reviews each, paired with unbiased expert summaries and manually annotated queries. Through extensive experiments, we identify key challenges, provide actionable insights into improving systems, pave the way for future research, and position OpinioRAG as a robust framework for generating accurate, relevant, and structured summaries at scale.
>
---
#### [replaced 100] SPARTA ALIGNMENT: Collectively Aligning Multiple Language Models through Combat
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.04721v3](http://arxiv.org/pdf/2506.04721v3)**

> **作者:** Yuru Jiang; Wenxuan Ding; Shangbin Feng; Greg Durrett; Yulia Tsvetkov
>
> **备注:** NeurIPS 2025
>
> **摘要:** We propose SPARTA ALIGNMENT, an algorithm to collectively align multiple LLMs through competition and combat. To complement a single model's lack of diversity in generation and biases in evaluation, multiple LLMs form a "sparta tribe" to compete against each other in fulfilling instructions while serving as judges for the competition of others. For each iteration, one instruction and two models are selected for a duel, the other models evaluate the two responses, and their evaluation scores are aggregated through a adapted elo-ranking based reputation system, where winners/losers of combat gain/lose weight in evaluating others. The peer-evaluated combat results then become preference pairs where the winning response is preferred over the losing one, and all models learn from these preferences at the end of each iteration. SPARTA ALIGNMENT enables the self-evolution of multiple LLMs in an iterative and collective competition process. Extensive experiments demonstrate that SPARTA ALIGNMENT outperforms initial models and 4 self-alignment baselines across 10 out of 12 tasks and datasets with 7.0% average improvement. Further analysis reveals that SPARTA ALIGNMENT generalizes more effectively to unseen tasks and leverages the expertise diversity of participating models to produce more logical, direct and informative outputs.
>
---
#### [replaced 101] Representation Consistency for Accurate and Coherent LLM Answer Aggregation
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.21590v2](http://arxiv.org/pdf/2506.21590v2)**

> **作者:** Junqi Jiang; Tom Bewley; Salim I. Amoukou; Francesco Leofante; Antonio Rago; Saumitra Mishra; Francesca Toni
>
> **备注:** Accepted at NeurIPS 2025. Camera-ready version
>
> **摘要:** Test-time scaling improves large language models' (LLMs) performance by allocating more compute budget during inference. To achieve this, existing methods often require intricate modifications to prompting and sampling strategies. In this work, we introduce representation consistency (RC), a test-time scaling method for aggregating answers drawn from multiple candidate responses of an LLM regardless of how they were generated, including variations in prompt phrasing and sampling strategy. RC enhances answer aggregation by not only considering the number of occurrences of each answer in the candidate response set, but also the consistency of the model's internal activations while generating the set of responses leading to each answer. These activations can be either dense (raw model activations) or sparse (encoded via pretrained sparse autoencoders). Our rationale is that if the model's representations of multiple responses converging on the same answer are highly variable, this answer is more likely to be the result of incoherent reasoning and should be down-weighted during aggregation. Importantly, our method only uses cached activations and lightweight similarity computations and requires no additional model queries. Through experiments with four open-source LLMs and four reasoning datasets, we validate the effectiveness of RC for improving task performance during inference, with consistent accuracy improvements (up to 4%) over strong test-time scaling baselines. We also show that consistency in the sparse activation signals aligns well with the common notion of coherent reasoning.
>
---
#### [replaced 102] Debiasing LLMs by Masking Unfairness-Driving Attention Heads
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.10142v3](http://arxiv.org/pdf/2510.10142v3)**

> **作者:** Tingxu Han; Wei Song; Ziqi Ding; Ziming Li; Chunrong Fang; Yuekang Li; Dongfang Liu; Zhenyu Chen; Zhenting Wang
>
> **摘要:** Large language models (LLMs) increasingly mediate decisions in domains where unfair treatment of demographic groups is unacceptable. Existing work probes when biased outputs appear, but gives little insight into the mechanisms that generate them, leaving existing mitigations largely fragile. In this paper, we conduct a systematic investigation LLM unfairness and propose DiffHeads, a lightweight debiasing framework for LLMs. We first compare Direct-Answer (DA) prompting to Chain-of-Thought (CoT) prompting across eight representative open- and closed-source LLMs. DA will trigger the nature bias part of LLM and improve measured unfairness by 534.5%-391.9% in both one-turn and two-turn dialogues. Next, we define a token-to-head contribution score that traces each token's influence back to individual attention heads. This reveals a small cluster of bias heads that activate under DA but stay largely dormant with CoT, providing the first causal link between prompting strategy and bias emergence. Finally, building on this insight, we propose DiffHeads that identifies bias heads through differential activation analysis between DA and CoT, and selectively masks only those heads. DiffHeads reduces unfairness by 49.4%, and 40.3% under DA and CoT, respectively, without harming model utility.
>
---
#### [replaced 103] Wisdom is Knowing What not to Say: Hallucination-Free LLMs Unlearning via Attention Shifting
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.17210v2](http://arxiv.org/pdf/2510.17210v2)**

> **作者:** Chenchen Tan; Youyang Qu; Xinghao Li; Hui Zhang; Shujie Cui; Cunjian Chen; Longxiang Gao
>
> **备注:** 22 pages, 10 figures
>
> **摘要:** The increase in computing power and the necessity of AI-assisted decision-making boost the growing application of large language models (LLMs). Along with this, the potential retention of sensitive data of LLMs has spurred increasing research into machine unlearning. However, existing unlearning approaches face a critical dilemma: Aggressive unlearning compromises model utility, while conservative strategies preserve utility but risk hallucinated responses. This significantly limits LLMs' reliability in knowledge-intensive applications. To address this, we introduce a novel Attention-Shifting (AS) framework for selective unlearning. AS is driven by two design objectives: (1) context-preserving suppression that attenuates attention to fact-bearing tokens without disrupting LLMs' linguistic structure; and (2) hallucination-resistant response shaping that discourages fabricated completions when queried about unlearning content. AS realizes these objectives through two attention-level interventions, which are importance-aware suppression applied to the unlearning set to reduce reliance on memorized knowledge and attention-guided retention enhancement that reinforces attention toward semantically essential tokens in the retained dataset to mitigate unintended degradation. These two components are jointly optimized via a dual-loss objective, which forms a soft boundary that localizes unlearning while preserving unrelated knowledge under representation superposition. Experimental results show that AS improves performance preservation over the state-of-the-art unlearning methods, achieving up to 15% higher accuracy on the ToFU benchmark and 10% on the TDEC benchmark, while maintaining competitive hallucination-free unlearning effectiveness. Compared to existing methods, AS demonstrates a superior balance between unlearning effectiveness, generalization, and response reliability.
>
---
#### [replaced 104] Readers Prefer Outputs of AI Trained on Copyrighted Books over Expert Human Writers
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2510.13939v3](http://arxiv.org/pdf/2510.13939v3)**

> **作者:** Tuhin Chakrabarty; Jane C. Ginsburg; Paramveer Dhillon
>
> **备注:** Preprint Under Review
>
> **摘要:** The use of copyrighted books for training AI models has led to numerous lawsuits from authors concerned about AI's ability to generate derivative content. Yet it's unclear if these models can generate high quality literary text while emulating authors' styles. To answer this we conducted a preregistered study comparing MFA-trained expert writers with three frontier AI models: ChatGPT, Claude & Gemini in writing up to 450 word excerpts emulating 50 award-winning authors' diverse styles. In blind pairwise evaluations by 159 representative expert & lay readers, AI-generated text from in-context prompting was strongly disfavored by experts for both stylistic fidelity (OR=0.16, p<10^-8) & writing quality (OR=0.13, p<10^-7) but showed mixed results with lay readers. However, fine-tuning ChatGPT on individual authors' complete works completely reversed these findings: experts now favored AI-generated text for stylistic fidelity (OR=8.16, p<10^-13) & writing quality (OR=1.87, p=0.010), with lay readers showing similar shifts. These effects generalize across authors & styles. The fine-tuned outputs were rarely flagged as AI-generated (3% rate v. 97% for in-context prompting) by best AI detectors. Mediation analysis shows this reversal occurs because fine-tuning eliminates detectable AI stylistic quirks (e.g., cliche density) that penalize in-context outputs. While we do not account for additional costs of human effort required to transform raw AI output into cohesive, publishable prose, the median fine-tuning & inference cost of $81 per author represents a dramatic 99.7% reduction compared to typical professional writer compensation. Author-specific fine-tuning thus enables non-verbatim AI writing that readers prefer to expert human writing, providing empirical evidence directly relevant to copyright's fourth fair-use factor, the "effect upon the potential market or value" of the source works.
>
---
#### [replaced 105] Multi-Step Reasoning with Large Language Models, a Survey
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2407.11511v3](http://arxiv.org/pdf/2407.11511v3)**

> **作者:** Aske Plaat; Annie Wong; Suzan Verberne; Joost Broekens; Niki van Stein; Thomas Back
>
> **备注:** ACM Computing Surveys
>
> **摘要:** Large language models (LLMs) with billions of parameters exhibit in-context learning abilities, enabling few-shot learning on tasks that the model was not specifically trained for. Traditional models achieve breakthrough performance on language tasks, but do not perform well on basic reasoning benchmarks. However, a new in-context learning approach, Chain-of-thought, has demonstrated strong multi-step reasoning abilities on these benchmarks. The research on LLM reasoning abilities started with the question whether LLMs can solve grade school math word problems, and has expanded to other tasks in the past few years. This article reviews the field of multi-step reasoning with LLMs. We propose a taxonomy that identifies different ways to generate, evaluate, and control multi-step reasoning. We provide an in-depth coverage of core approaches and open problems, and we propose a research agenda for the near future. We find that multi-step reasoning approaches have progressed beyond math word problems, and can now successfully solve challenges in logic, combinatorial games, and robotics, sometimes by first generating code that is then executed by external tools. Many studies in multi-step methods use reinforcement learning for finetuning, external optimization loops, in-context reinforcement learning, and self-reflection.
>
---
#### [replaced 106] Res-Bench: Benchmarking the Robustness of Multimodal Large Language Models to Dynamic Resolution Input
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.16926v2](http://arxiv.org/pdf/2510.16926v2)**

> **作者:** Chenxu Li; Zhicai Wang; Yuan Sheng; Xingyu Zhu; Yanbin Hao; Xiang Wang
>
> **备注:** The authors have discovered a significant error in the paper subsequent to submission, and are withdrawing the manuscript for substantial correction
>
> **摘要:** Multimodal Large Language Models (MLLMs) increasingly support dynamic image resolutions. However, current evaluation paradigms primarily assess semantic performance, overlooking the critical question of resolution robustness - whether performance remains stable across varying input resolutions. To address this gap, we introduce \textbf{Res-Bench}, a comprehensive benchmark comprising 14,400 samples across 12 resolution levels and six core capability dimensions. We designed a novel evaluation framework that goes beyond traditional accuracy metrics to capture performance stability. This framework introduces multiple robustness metrics: Spearman's correlation for assessing resolution-performance trends, and Absolute/Relative Continuous Error (ACE/RCE) for measuring performance volatility. Using these metrics, we conducted a large-scale evaluation of leading MLLMs. Our analysis encompasses: (1) model-centric and task-centric robustness examination, (2) investigation of preprocessing strategies including padding and super-resolution, and (3) exploration of fine-tuning for stability enhancement.
>
---
#### [replaced 107] New Encoders for German Trained from Scratch: Comparing ModernGBERT with Converted LLM2Vec Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.13136v2](http://arxiv.org/pdf/2505.13136v2)**

> **作者:** Julia Wunderle; Anton Ehrmanntraut; Jan Pfister; Fotis Jannidis; Andreas Hotho
>
> **备注:** under review @LREC
>
> **摘要:** Encoders remain essential for efficient German NLP and NLU scenarios despite the rise of decoder-only LLMs. This work studies two routes to high-quality German encoders under identical data and training constraints: 1) training from scratch and 2) converting decoders via LLM2Vec. We introduce two resources: ModernGBERT (134M, 1B), fully transparent German encoders in the ModernBERT style, and LL\"aMmleinVec (120M, 1B, 7B), decoder-to-encoder conversions trained with masked next-token prediction, both undergoing a context extension to 8.192 tokens. Across SuperGLEBer, ModernGBERT 1B sets a new state of the art (avg 0.808), surpassing GBERT Large (+4%) and the seven-times larger converted 7B model (0.787). On German MTEB after supervised fine-tuning, ModernGBERT 1B (0.551) approaches the converted 7B model (0.557). We release all models, checkpoints, datasets, and full training records, and introduce an encoder-adapted QA-NIAH evaluation. All in all, our results provide actionable guidance: when parameter efficiency and latency matter, from-scratch encoders dominate. When a pre-trained decoder exists and compute is a limited, conversion offers an effective alternative. ModernGBERT and LL\"aMmleinVec, including all code, data and intermediary checkpoints are published under a research-only RAIL license.
>
---
#### [replaced 108] Large Language Models as Medical Codes Selectors: a benchmark using the International Classification of Primary Care
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.14681v2](http://arxiv.org/pdf/2507.14681v2)**

> **作者:** Vinicius Anjos de Almeida; Vinicius de Camargo; Raquel Gómez-Bravo; Egbert van der Haring; Kees van Boven; Marcelo Finger; Luis Fernandez Lopez
>
> **备注:** Accepted at NeurIPS 2025 as a poster presentation in The Second Workshop on GenAI for Health: Potential, Trust, and Policy Compliance (https://openreview.net/forum?id=Kl7KZwJFEG). 33 pages, 10 figures (including appendix), 15 tables (including appendix). To be submitted to peer-reviewed journal. For associated code repository, see https://github.com/almeidava93/llm-as-code-selectors-paper
>
> **摘要:** Background: Medical coding structures healthcare data for research, quality monitoring, and policy. This study assesses the potential of large language models (LLMs) to assign ICPC-2 codes using the output of a domain-specific search engine. Methods: A dataset of 437 Brazilian Portuguese clinical expressions, each annotated with ICPC-2 codes, was used. A semantic search engine (OpenAI's text-embedding-3-large) retrieved candidates from 73,563 labeled concepts. Thirty-three LLMs were prompted with each query and retrieved results to select the best-matching ICPC-2 code. Performance was evaluated using F1-score, along with token usage, cost, response time, and format adherence. Results: Twenty-eight models achieved F1-score > 0.8; ten exceeded 0.85. Top performers included gpt-4.5-preview, o3, and gemini-2.5-pro. Retriever optimization can improve performance by up to 4 points. Most models returned valid codes in the expected format, with reduced hallucinations. Smaller models (<3B) struggled with formatting and input length. Conclusions: LLMs show strong potential for automating ICPC-2 coding, even without fine-tuning. This work offers a benchmark and highlights challenges, but findings are limited by dataset scope and setup. Broader, multilingual, end-to-end evaluations are needed for clinical validation.
>
---
#### [replaced 109] Words That Unite The World: A Unified Framework for Deciphering Central Bank Communications Globally
- **分类: cs.CL; cs.AI; cs.CY; q-fin.CP; q-fin.GN**

- **链接: [http://arxiv.org/pdf/2505.17048v2](http://arxiv.org/pdf/2505.17048v2)**

> **作者:** Agam Shah; Siddhant Sukhani; Huzaifa Pardawala; Saketh Budideti; Riya Bhadani; Rudra Gopal; Siddhartha Somani; Rutwik Routu; Michael Galarnyk; Soungmin Lee; Arnav Hiray; Akshar Ravichandran; Eric Kim; Pranav Aluru; Joshua Zhang; Sebastian Jaskowski; Veer Guda; Meghaj Tarte; Liqin Ye; Spencer Gosden; Rachel Yuh; Sloka Chava; Sahasra Chava; Dylan Patrick Kelly; Aiden Chiang; Harsit Mittal; Sudheer Chava
>
> **备注:** Accepted at NeurIPS 2025 (main conference)
>
> **摘要:** Central banks around the world play a crucial role in maintaining economic stability. Deciphering policy implications in their communications is essential, especially as misinterpretations can disproportionately impact vulnerable populations. To address this, we introduce the World Central Banks (WCB) dataset, the most comprehensive monetary policy corpus to date, comprising over 380k sentences from 25 central banks across diverse geographic regions, spanning 28 years of historical data. After uniformly sampling 1k sentences per bank (25k total) across all available years, we annotate and review each sentence using dual annotators, disagreement resolutions, and secondary expert reviews. We define three tasks: Stance Detection, Temporal Classification, and Uncertainty Estimation, with each sentence annotated for all three. We benchmark seven Pretrained Language Models (PLMs) and nine Large Language Models (LLMs) (Zero-Shot, Few-Shot, and with annotation guide) on these tasks, running 15,075 benchmarking experiments. We find that a model trained on aggregated data across banks significantly surpasses a model trained on an individual bank's data, confirming the principle "the whole is greater than the sum of its parts." Additionally, rigorous human evaluations, error analyses, and predictive tasks validate our framework's economic utility. Our artifacts are accessible through the HuggingFace and GitHub under the CC-BY-NC-SA 4.0 license.
>
---
#### [replaced 110] Medical Hallucinations in Foundation Models and Their Impact on Healthcare
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2503.05777v2](http://arxiv.org/pdf/2503.05777v2)**

> **作者:** Yubin Kim; Hyewon Jeong; Shan Chen; Shuyue Stella Li; Chanwoo Park; Mingyu Lu; Kumail Alhamoud; Jimin Mun; Cristina Grau; Minseok Jung; Rodrigo Gameiro; Lizhou Fan; Eugene Park; Tristan Lin; Joonsik Yoon; Wonjin Yoon; Maarten Sap; Yulia Tsvetkov; Paul Liang; Xuhai Xu; Xin Liu; Chunjong Park; Hyeonhoon Lee; Hae Won Park; Daniel McDuff; Samir Tulebaev; Cynthia Breazeal
>
> **摘要:** Hallucinations in foundation models arise from autoregressive training objectives that prioritize token-likelihood optimization over epistemic accuracy, fostering overconfidence and poorly calibrated uncertainty. We define medical hallucination as any model-generated output that is factually incorrect, logically inconsistent, or unsupported by authoritative clinical evidence in ways that could alter clinical decisions. We evaluated 11 foundation models (7 general-purpose, 4 medical-specialized) across seven medical hallucination tasks spanning medical reasoning and biomedical information retrieval. General-purpose models achieved significantly higher proportions of hallucination-free responses than medical-specialized models (median: 76.6% vs 51.3%, difference = 25.2%, 95% CI: 18.7-31.3%, Mann-Whitney U = 27.0, p = 0.012, rank-biserial r = -0.64). Top-performing models such as Gemini-2.5 Pro exceeded 97% accuracy when augmented with chain-of-thought prompting (base: 87.6%), while medical-specialized models like MedGemma ranged from 28.6-61.9% despite explicit training on medical corpora. Chain-of-thought reasoning significantly reduced hallucinations in 86.4% of tested comparisons after FDR correction (q < 0.05), demonstrating that explicit reasoning traces enable self-verification and error detection. Physician audits confirmed that 64-72% of residual hallucinations stemmed from causal or temporal reasoning failures rather than knowledge gaps. A global survey of clinicians (n = 70) validated real-world impact: 91.8% had encountered medical hallucinations, and 84.7% considered them capable of causing patient harm. The underperformance of medical-specialized models despite domain training indicates that safety emerges from sophisticated reasoning capabilities and broad knowledge integration developed during large-scale pre-training, not from narrow optimization.
>
---
#### [replaced 111] Learning to Steer: Input-dependent Steering for Multimodal LLMs
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.12815v2](http://arxiv.org/pdf/2508.12815v2)**

> **作者:** Jayneel Parekh; Pegah Khayatan; Mustafa Shukor; Arnaud Dapogny; Alasdair Newson; Matthieu Cord
>
> **备注:** NeurIPS 2025
>
> **摘要:** Steering has emerged as a practical approach to enable post-hoc guidance of LLMs towards enforcing a specific behavior. However, it remains largely underexplored for multimodal LLMs (MLLMs); furthermore, existing steering techniques, such as mean steering, rely on a single steering vector, applied independently of the input query. This paradigm faces limitations when the desired behavior is dependent on the example at hand. For example, a safe answer may consist in abstaining from answering when asked for an illegal activity, or may point to external resources or consultation with an expert when asked about medical advice. In this paper, we investigate a fine-grained steering that uses an input-specific linear shift. This shift is computed using contrastive input-specific prompting. However, the input-specific prompts required for this approach are not known at test time. Therefore, we propose to train a small auxiliary module to predict the input-specific steering vector. Our approach, dubbed as L2S (Learn-to-Steer), demonstrates that it reduces hallucinations and enforces safety in MLLMs, outperforming other static baselines. Our code is publicly available at https://jayneelparekh.github.io/learn-to-steer/
>
---
#### [replaced 112] Exploring the Synergy of Quantitative Factors and Newsflow Representations from Large Language Models for Stock Return Prediction
- **分类: q-fin.CP; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.15691v2](http://arxiv.org/pdf/2510.15691v2)**

> **作者:** Tian Guo; Emmanuel Hauptmann
>
> **摘要:** In quantitative investing, return prediction supports various tasks, including stock selection, portfolio optimization, and risk management. Quantitative factors, such as valuation, quality, and growth, capture various characteristics of stocks. Unstructured financial data, like news and transcripts, has attracted growing attention, driven by recent advances in large language models (LLMs). This paper examines effective methods for leveraging multimodal factors and newsflow in return prediction and stock selection. First, we introduce a fusion learning framework to learn a unified representation from factors and newsflow representations generated by an LLM. Within this framework, we compare three representative methods: representation combination, representation summation, and attentive representations. Next, building on empirical observations from fusion learning, we explore the mixture model that adaptively combines predictions made by single modalities and their fusion. To mitigate the training instability observed in the mixture model, we introduce a decoupled training approach with theoretical insights. Finally, our experiments on real investment universes yield several insights into effective multimodal modeling of factors and news for stock return prediction and selection.
>
---
#### [replaced 113] PRISM2: Unlocking Multi-Modal General Pathology AI with Clinical Dialogue
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.13063v2](http://arxiv.org/pdf/2506.13063v2)**

> **作者:** Eugene Vorontsov; George Shaikovski; Adam Casson; Julian Viret; Eric Zimmermann; Neil Tenenholtz; Yi Kan Wang; Jan H. Bernhard; Ran A. Godrich; Juan A. Retamero; Jinru Shia; Mithat Gonen; Martin R. Weiser; David S. Klimstra; Razik Yousfi; Nicolo Fusi; Thomas J. Fuchs; Kristen Severson; Siqi Liu
>
> **摘要:** Recent rapid progress in the field of computational pathology has been enabled by foundation models. These models are beginning to move beyond encoding image patches towards whole-slide understanding but their clinical utility remains limited. In this work, we present PRISM2, a multimodal slide-level foundation model trained on data from 700,000 diagnostic specimen-report pairs, the largest vision (2.3 million whole slide images) and language (14M question-answer pairs) histopathology dataset to date. By learning through clinical-dialogue supervision, PRISM2 aligns histomorphologic features with the language of diagnostic reasoning, producing slide-level representations that support both direct diagnostic question-answering and transferable embeddings for downstream tasks. Without additional training, PRISM2 matches or exceeds the cancer-detection performance of clinical-grade products. This is observed without loss of generality on other tasks, where PRISM2 achieves top performance. Finally, using survival prediction as the example, we show that task-specific finetuning with a large dataset can outperform task-specific models, further improving performance. These results demonstrate how language-supervised pretraining provides a scalable, clinically grounded signal for learning generalizable pathology representations, bridging human diagnostic reasoning and foundation-model performance.
>
---
#### [replaced 114] CrowdVLM-R1: Expanding R1 Ability to Vision Language Model for Crowd Counting using Fuzzy Group Relative Policy Reward
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.03724v2](http://arxiv.org/pdf/2504.03724v2)**

> **作者:** Zhiqiang Wang; Pengbin Feng; Yanbin Lin; Shuzhang Cai; Zongao Bian; Jinghua Yan; Xingquan Zhu
>
> **备注:** 10 pages, 6 figures and 4 tables
>
> **摘要:** We propose Fuzzy Group Relative Policy Reward (FGRPR), a novel framework that integrates Group Relative Policy Optimization (GRPO) with a fuzzy reward function to enhance learning efficiency. Unlike the conventional binary 0/1 accuracy reward, our fuzzy reward model provides nuanced incentives, encouraging more precise outputs. Experimental results demonstrate that GRPO with a standard 0/1 accuracy reward underperforms compared to supervised fine-tuning (SFT). In contrast, FGRPR, applied to Qwen2.5-VL(3B and 7B), surpasses all baseline models, including GPT4o, LLaMA2(90B), and SFT, across five in-domain datasets. On an out-of-domain dataset, FGRPR achieves performance comparable to SFT but excels when target values are larger, as its fuzzy reward function assigns higher rewards to closer approximations. This approach is broadly applicable to tasks where the precision of the answer is critical. Code and data: https://github.com/yeyimilk/CrowdVLM-R1
>
---
#### [replaced 115] Kimi Linear: An Expressive, Efficient Attention Architecture
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.26692v2](http://arxiv.org/pdf/2510.26692v2)**

> **作者:** Kimi Team; Yu Zhang; Zongyu Lin; Xingcheng Yao; Jiaxi Hu; Fanqing Meng; Chengyin Liu; Xin Men; Songlin Yang; Zhiyuan Li; Wentao Li; Enzhe Lu; Weizhou Liu; Yanru Chen; Weixin Xu; Longhui Yu; Yejie Wang; Yu Fan; Longguang Zhong; Enming Yuan; Dehao Zhang; Yizhi Zhang; T. Y. Liu; Haiming Wang; Shengjun Fang; Weiran He; Shaowei Liu; Yiwei Li; Jianlin Su; Jiezhong Qiu; Bo Pang; Junjie Yan; Zhejun Jiang; Weixiao Huang; Bohong Yin; Jiacheng You; Chu Wei; Zhengtao Wang; Chao Hong; Yutian Chen; Guanduo Chen; Yucheng Wang; Huabin Zheng; Feng Wang; Yibo Liu; Mengnan Dong; Zheng Zhang; Siyuan Pan; Wenhao Wu; Yuhao Wu; Longyu Guan; Jiawen Tao; Guohong Fu; Xinran Xu; Yuzhi Wang; Guokun Lai; Yuxin Wu; Xinyu Zhou; Zhilin Yang; Yulun Du
>
> **备注:** Kimi Linear tech report
>
> **摘要:** We introduce Kimi Linear, a hybrid linear attention architecture that, for the first time, outperforms full attention under fair comparisons across various scenarios -- including short-context, long-context, and reinforcement learning (RL) scaling regimes. At its core lies Kimi Delta Attention (KDA), an expressive linear attention module that extends Gated DeltaNet with a finer-grained gating mechanism, enabling more effective use of limited finite-state RNN memory. Our bespoke chunkwise algorithm achieves high hardware efficiency through a specialized variant of the Diagonal-Plus-Low-Rank (DPLR) transition matrices, which substantially reduces computation compared to the general DPLR formulation while remaining more consistent with the classical delta rule. We pretrain a Kimi Linear model with 3B activated parameters and 48B total parameters, based on a layerwise hybrid of KDA and Multi-Head Latent Attention (MLA). Our experiments show that with an identical training recipe, Kimi Linear outperforms full MLA with a sizeable margin across all evaluated tasks, while reducing KV cache usage by up to 75% and achieving up to 6 times decoding throughput for a 1M context. These results demonstrate that Kimi Linear can be a drop-in replacement for full attention architectures with superior performance and efficiency, including tasks with longer input and output lengths. To support further research, we open-source the KDA kernel and vLLM implementations, and release the pre-trained and instruction-tuned model checkpoints.
>
---
