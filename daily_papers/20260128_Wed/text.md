# 自然语言处理 cs.CL

- **最新发布 64 篇**

- **更新 79 篇**

## 最新发布

#### [new 001] When Iterative RAG Beats Ideal Evidence: A Diagnostic Study in Scientific Multi-hop Question Answering
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文研究科学多跳问答任务，探讨迭代RAG是否优于理想静态RAG。通过实验和诊断分析，验证了迭代方法在特定场景下的优势。**

- **链接: [https://arxiv.org/pdf/2601.19827v1](https://arxiv.org/pdf/2601.19827v1)**

> **作者:** Mahdi Astaraki; Mohammad Arshi Saloot; Ali Shiraee Kasmaee; Hamidreza Mahyar; Soheila Samiee
>
> **备注:** 27 pages, 15 figures
>
> **摘要:** Retrieval-Augmented Generation (RAG) extends large language models (LLMs) beyond parametric knowledge, yet it is unclear when iterative retrieval-reasoning loops meaningfully outperform static RAG, particularly in scientific domains with multi-hop reasoning, sparse domain knowledge, and heterogeneous evidence. We provide the first controlled, mechanism-level diagnostic study of whether synchronized iterative retrieval and reasoning can surpass an idealized static upper bound (Gold Context) RAG. We benchmark eleven state-of-the-art LLMs under three regimes: (i) No Context, measuring reliance on parametric memory; (ii) Gold Context, where all oracle evidence is supplied at once; and (iii) Iterative RAG, a training-free controller that alternates retrieval, hypothesis refinement, and evidence-aware stopping. Using the chemistry-focused ChemKGMultiHopQA dataset, we isolate questions requiring genuine retrieval and analyze behavior with diagnostics spanning retrieval coverage gaps, anchor-carry drop, query quality, composition fidelity, and control calibration. Across models, Iterative RAG consistently outperforms Gold Context, with gains up to 25.6 percentage points, especially for non-reasoning fine-tuned models. Staged retrieval reduces late-hop failures, mitigates context overload, and enables dynamic correction of early hypothesis drift, but remaining failure modes include incomplete hop coverage, distractor latch trajectories, early stopping miscalibration, and high composition failure rates even with perfect retrieval. Overall, staged retrieval is often more influential than the mere presence of ideal evidence; we provide practical guidance for deploying and diagnosing RAG systems in specialized scientific settings and a foundation for more reliable, controllable iterative retrieval-reasoning frameworks.
>
---
#### [new 002] A Hybrid Supervised-LLM Pipeline for Actionable Suggestion Mining in Unstructured Customer Reviews
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在从客户评论中提取具体改进建议。针对现有方法无法精准提取问题，提出混合模型提升准确性与可解释性。**

- **链接: [https://arxiv.org/pdf/2601.19214v1](https://arxiv.org/pdf/2601.19214v1)**

> **作者:** Aakash Trivedi; Aniket Upadhyay; Pratik Narang; Dhruv Kumar; Praveen Kumar
>
> **备注:** Accepted to EACL 2026 Industry Track (to appear)
>
> **摘要:** Extracting actionable suggestions from customer reviews is essential for operational decision-making, yet these directives are often embedded within mixed-intent, unstructured text. Existing approaches either classify suggestion-bearing sentences or generate high-level summaries, but rarely isolate the precise improvement instructions businesses need. We evaluate a hybrid pipeline combining a high-recall RoBERTa classifier trained with a precision-recall surrogate to reduce unrecoverable false negatives with a controlled, instruction-tuned LLM for suggestion extraction, categorization, clustering, and summarization. Across real-world hospitality and food datasets, the hybrid system outperforms prompt-only, rule-based, and classifier-only baselines in extraction accuracy and cluster coherence. Human evaluations further confirm that the resulting suggestions and summaries are clear, faithful, and interpretable. Overall, our results show that hybrid reasoning architectures achieve meaningful improvements fine-grained actionable suggestion mining while highlighting challenges in domain adaptation and efficient local deployment.
>
---
#### [new 003] Cross-Examination Framework: A Task-Agnostic Diagnostic for Information Fidelity in Text-to-Text Generation
- **分类: cs.CL**

- **简介: 该论文属于文本生成任务，解决传统指标无法准确评估生成文本语义一致性的难题。提出CEF框架，通过交叉质询生成多维评分，有效检测内容遗漏和事实矛盾。**

- **链接: [https://arxiv.org/pdf/2601.19350v1](https://arxiv.org/pdf/2601.19350v1)**

> **作者:** Tathagata Raha; Clement Christophe; Nada Saadi; Hamza A Javed; Marco AF Pimentel; Ronnie Rajan; Praveenkumar Kanithi
>
> **摘要:** Traditional metrics like BLEU and BERTScore fail to capture semantic fidelity in generative text-to-text tasks. We adapt the Cross-Examination Framework (CEF) for a reference-free, multi-dimensional evaluation by treating the source and candidate as independent knowledge bases. CEF generates verifiable questions from each text and performs a cross-examination to derive three interpretable scores: Coverage, Conformity, and Consistency. Validated across translation, summarization and clinical note-generation, our framework identifies critical errors, such as content omissions and factual contradictions, missed by standard metrics. A key contribution is a systematic robustness analysis to select a stable judge model. Crucially, the strong correlation between our reference-free and with-reference modes validates CEF's reliability without gold references. Furthermore, human expert validation demonstrates that CEF mismatching questions align with meaning-altering semantic errors higher than with non-semantic errors, particularly excelling at identifying entity-based and relational distortions.
>
---
#### [new 004] Identifying and Transferring Reasoning-Critical Neurons: Improving LLM Inference Reliability via Activation Steering
- **分类: cs.CL**

- **简介: 该论文属于提升大语言模型推理可靠性任务，解决模型在复杂任务中表现不稳定的问题。通过识别关键神经元并调整其激活，提高推理正确性。**

- **链接: [https://arxiv.org/pdf/2601.19847v1](https://arxiv.org/pdf/2601.19847v1)**

> **作者:** Fangan Dong; Zuming Yan; Xuri Ge; Zhiwei Xu; Mengqi Zhang; Xuanang Chen; Ben He; Xin Xin; Zhumin Chen; Ying Zhou
>
> **摘要:** Despite the strong reasoning capabilities of recent large language models (LLMs), achieving reliable performance on challenging tasks often requires post-training or computationally expensive sampling strategies, limiting their practical efficiency. In this work, we first show that a small subset of neurons in LLMs exhibits strong predictive correlations with reasoning correctness. Based on this observation, we propose AdaRAS (Adaptive Reasoning Activation Steering), a lightweight test-time framework that improves reasoning reliability by selectively intervening on neuron activations. AdaRAS identifies Reasoning-Critical Neurons (RCNs) via a polarity-aware mean-difference criterion and adaptively steers their activations during inference, enhancing incorrect reasoning traces while avoiding degradation on already-correct cases. Experiments on 10 mathematics and coding benchmarks demonstrate consistent improvements, including over 13% gains on AIME-24 and AIME-25. Moreover, AdaRAS exhibits strong transferability across datasets and scalability to stronger models, outperforming post-training methods without additional training or sampling cost.
>
---
#### [new 005] PsyProbe: Proactive and Interpretable Dialogue through User State Modeling for Exploratory Counseling
- **分类: cs.CL**

- **简介: 该论文提出PsyProbe，用于心理辅导中的主动对话系统，解决现有系统反应性不足的问题。通过用户状态建模，提升治疗探索效果。**

- **链接: [https://arxiv.org/pdf/2601.19096v1](https://arxiv.org/pdf/2601.19096v1)**

> **作者:** Sohhyung Park; Hyunji Kang; Sungzoon Cho; Dongil Kim
>
> **备注:** In Findings of the Association for Computational Linguistics: EACL 2026
>
> **摘要:** Recent advances in large language models have enabled mental health dialogue systems, yet existing approaches remain predominantly reactive, lacking systematic user state modeling for proactive therapeutic exploration. We introduce PsyProbe, a dialogue system designed for the exploration phase of counseling that systematically tracks user psychological states through the PPPPPI framework (Presenting, Predisposing, Precipitating, Perpetuating, Protective, Impact) augmented with cognitive error detection. PsyProbe combines State Builder for extracting structured psychological profiles, Memory Construction for tracking information gaps, Strategy Planner for Motivational Interviewing behavioral codes, and Response Generator with Question Ideation and Critic/Revision modules to generate contextually appropriate, proactive questions. We evaluate PsyProbe with 27 participants in real-world Korean counseling scenarios, including automatic evaluation across ablation modes, user evaluation, and expert evaluation by a certified counselor. The full PsyProbe model consistently outperforms baseline and ablation modes in automatic evaluation. User evaluation demonstrates significantly increased engagement intention and improved naturalness compared to baseline. Expert evaluation shows that PsyProbe substantially improves core issue understanding and achieves question rates comparable to professional counselors, validating the effectiveness of systematic state modeling and proactive questioning for therapeutic exploration.
>
---
#### [new 006] LLMs versus the Halting Problem: Revisiting Program Termination Prediction
- **分类: cs.CL; cs.AI; cs.PL**

- **简介: 该论文属于程序终止性预测任务，探讨LLMs能否可靠预测程序是否终止。研究对比了LLMs与传统工具在SV-Comp数据集上的表现，发现LLMs效果良好但存在局限。**

- **链接: [https://arxiv.org/pdf/2601.18987v1](https://arxiv.org/pdf/2601.18987v1)**

> **作者:** Oren Sultan; Jordi Armengol-Estape; Pascal Kesseli; Julien Vanegue; Dafna Shahaf; Yossi Adi; Peter O'Hearn
>
> **摘要:** Determining whether a program terminates is a central problem in computer science. Turing's foundational result established the Halting Problem as undecidable, showing that no algorithm can universally determine termination for all programs and inputs. Consequently, automatic verification tools approximate termination, sometimes failing to prove or disprove; these tools rely on problem-specific architectures and abstractions, and are usually tied to particular programming languages. Recent success and progress in large language models (LLMs) raises the following question: can LLMs reliably predict program termination? In this work, we evaluate LLMs on a diverse set of C programs from the Termination category of the International Competition on Software Verification (SV-Comp) 2025. Our results suggest that LLMs perform remarkably well at predicting program termination, where GPT-5 and Claude Sonnet-4.5 would rank just behind the top-ranked tool (using test-time-scaling), and Code World Model (CWM) would place just behind the second-ranked tool. While LLMs are effective at predicting program termination, they often fail to provide a valid witness as a proof. Moreover, LLMs performance drops as program length increases. We hope these insights motivate further research into program termination and the broader potential of LLMs for reasoning about undecidable problems.
>
---
#### [new 007] Yunque DeepResearch Technical Report
- **分类: cs.CL**

- **简介: 该论文提出Yunque DeepResearch，解决深度研究中的任务复杂、错误易发和扩展性差问题，通过分层模块化架构提升性能。**

- **链接: [https://arxiv.org/pdf/2601.19578v1](https://arxiv.org/pdf/2601.19578v1)**

> **作者:** Yuxuan Cai; Xinyi Lai; Peng Yuan; Weiting Liu; Huajian Li; Mingda Li; Xinghua Wang; Shengxie Zheng; Yanchao Hao; Yuyang Yin; Zheng Wei
>
> **摘要:** Deep research has emerged as a transformative capability for autonomous agents, empowering Large Language Models to navigate complex, open-ended tasks. However, realizing its full potential is hindered by critical limitations, including escalating contextual noise in long-horizon tasks, fragility leading to cascading errors, and a lack of modular extensibility. To address these challenges, we introduce Yunque DeepResearch, a hierarchical, modular, and robust framework. The architecture is characterized by three key components: (1) a centralized Multi-Agent Orchestration System that routes subtasks to an Atomic Capability Pool of tools and specialized sub-agents; (2) a Dynamic Context Management mechanism that structures completed sub-goals into semantic summaries to mitigate information overload; and (3) a proactive Supervisor Module that ensures resilience through active anomaly detection and context pruning. Yunque DeepResearch achieves state-of-the-art performance across a range of agentic deep research benchmarks, including GAIA, BrowseComp, BrowseComp-ZH, and Humanity's Last Exam. We open-source the framework, reproducible implementations, and application cases to empower the community.
>
---
#### [new 008] Reflective Translation: Improving Low-Resource Machine Translation via Structured Self-Reflection
- **分类: cs.CL**

- **简介: 该论文属于机器翻译任务，旨在解决低资源语言翻译质量差的问题。通过引入结构化自我反思机制，提升翻译准确性和一致性。**

- **链接: [https://arxiv.org/pdf/2601.19871v1](https://arxiv.org/pdf/2601.19871v1)**

> **作者:** Nicholas Cheng
>
> **备注:** 12 pages, 3 figures, 6 tables. Accepted to the NeurIPS 2025 Workshop on Multilingual Representation Learning (Mexico City) and the AAAI 2025 Workshop on Language Models for Under-Resourced Communities (LM4UC). Code and data available at: https://github.com/Nickcheng123/reflective-translation-mt
>
> **摘要:** Low-resource languages such as isiZulu and isiXhosa face persistent challenges in machine translation due to limited parallel data and linguistic resources. Recent advances in large language models suggest that self-reflection, prompting a model to critique and revise its own outputs, can improve reasoning quality and factual consistency. Building on this idea, this paper introduces Reflective Translation, a prompt-based framework in which a model generates an initial translation, produces a structured self-critique, and then uses this reflection to generate a refined translation. The approach is evaluated on English-isiZulu and English-isiXhosa translation using OPUS-100 and NTREX-African, across multiple prompting strategies and confidence thresholds. Results show consistent improvements in both BLEU and COMET scores between first- and second-pass translations, with average gains of up to +0.22 BLEU and +0.18 COMET. Statistical significance testing using paired nonparametric tests confirms that these improvements are robust. The proposed method is model-agnostic, requires no fine-tuning, and introduces a reflection-augmented dataset that can support future supervised or analysis-driven work. These findings demonstrate that structured self-reflection is a practical and effective mechanism for improving translation quality in low-resource settings.
>
---
#### [new 009] Riddle Quest : The Enigma of Words
- **分类: cs.CL; cs.AI; cs.IT**

- **简介: 该论文属于自然语言处理任务，旨在研究如何生成和评估基于类比的谜语。工作包括构建谜语生成管道，并测试语言模型对谜语答案的覆盖能力。**

- **链接: [https://arxiv.org/pdf/2601.19273v1](https://arxiv.org/pdf/2601.19273v1)**

> **作者:** Niharika Sri Parasa; Chaitali Diwan; Srinath Srinivasa
>
> **备注:** This paper is submitted under 'Demo track' for WWW conference
>
> **摘要:** Riddles are concise linguistic puzzles that describe an object or idea through indirect, figurative, or playful clues. They are a longstanding form of creative expression, requiring the solver to interpret hints, recognize patterns, and draw inferences to identify the answers. In this work, we introduce a simple pipeline for creating and evaluating analogy-based riddles. The system includes a triples creator that builds structured facts about a concept, a semantic mapper that selects attributes useful for analogy, a stylized generator that turns them into riddle clues, and a validator that collects all possible answers the riddle could point to. We use this validator to study whether large language models can recover the full answer set for different riddle types. Our case study shows that while models often guess the main intended answer, they frequently miss other valid interpretations. This highlights the value of riddles as a lightweight tool for examining reasoning coverage and ambiguity handling in language models.
>
---
#### [new 010] DART: Diffusion-Inspired Speculative Decoding for Fast LLM Inference
- **分类: cs.CL**

- **简介: 该论文属于大语言模型推理加速任务，解决现有推测解码方法延迟高的问题。提出DART，通过并行生成降低 drafting 时延，提升整体解码速度。**

- **链接: [https://arxiv.org/pdf/2601.19278v1](https://arxiv.org/pdf/2601.19278v1)**

> **作者:** Fuliang Liu; Xue Li; Ketai Zhao; Yinxi Gao; Ziyan Zhou; Zhonghui Zhang; Zhibin Wang; Wanchun Dou; Sheng Zhong; Chen Tian
>
> **摘要:** Speculative decoding is an effective and lossless approach for accelerating LLM inference. However, existing widely adopted model-based draft designs, such as EAGLE3, improve accuracy at the cost of multi-step autoregressive inference, resulting in high drafting latency and ultimately rendering the drafting stage itself a performance bottleneck. Inspired by diffusion-based large language models (dLLMs), we propose DART, which leverages parallel generation to reduce drafting latency. DART predicts logits for multiple future masked positions in parallel within a single forward pass based on hidden states of the target model, thereby eliminating autoregressive rollouts in the draft model while preserving a lightweight design. Based on these parallel logit predictions, we further introduce an efficient tree pruning algorithm that constructs high-quality draft token trees with N-gram-enforced semantic continuity. DART substantially reduces draft-stage overhead while preserving high draft accuracy, leading to significantly improved end-to-end decoding speed. Experimental results demonstrate that DART achieves a 2.03x--3.44x wall-clock time speedup across multiple datasets, surpassing EAGLE3 by 30% on average and offering a practical speculative decoding framework. Code is released at https://github.com/fvliang/DART.
>
---
#### [new 011] Evaluation of Oncotimia: An LLM based system for supporting tumour boards
- **分类: cs.CL**

- **简介: 论文介绍ONCOTIMIA系统，用于辅助肿瘤讨论会的文档生成。任务是解决肿瘤讨论会中手动处理大量临床信息的问题，通过LLM自动完成表单，提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2601.19899v1](https://arxiv.org/pdf/2601.19899v1)**

> **作者:** Luis Lorenzo; Marcos Montana-Mendez; Sergio Figueiras; Miguel Boubeta; Cristobal Bernardo-Castineira
>
> **备注:** 9 pages, 2 figures
>
> **摘要:** Multidisciplinary tumour boards (MDTBs) play a central role in oncology decision-making but require manual processes and structuring large volumes of heterogeneous clinical information, resulting in a substantial documentation burden. In this work, we present ONCOTIMIA, a modular and secure clinical tool designed to integrate generative artificial intelligence (GenAI) into oncology workflows and evaluate its application to the automatic completion of lung cancer tumour board forms using large language models (LLMs). The system combines a multi-layer data lake, hybrid relational and vector storage, retrieval-augmented generation (RAG) and a rule-driven adaptive form model to transform unstructured clinical documentation into structured and standardised tumour board records. We assess the performance of six LLMs deployed through AWS Bedrock on ten lung cancer cases, measuring both completion form accuracy and end-to-end latency. The results demonstrate high performance across models, with the best performing configuration achieving an 80% of correct field completion and clinically acceptable response time for most LLMs. Larger and more recent models exhibit best accuracies without incurring prohibitive latency. These findings provide empirical evidence that LLM- assisted autocompletion form is technically feasible and operationally viable in multidisciplinary lung cancer workflows and support its potential to significantly reduce documentation burden while preserving data quality.
>
---
#### [new 012] KG-CRAFT: Knowledge Graph-based Contrastive Reasoning with LLMs for Enhancing Automated Fact-checking
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于事实核查任务，旨在提升自动验证声明的准确性。通过构建知识图谱并生成对比问题，增强大语言模型的事实核查能力。**

- **链接: [https://arxiv.org/pdf/2601.19447v1](https://arxiv.org/pdf/2601.19447v1)**

> **作者:** Vítor N. Lourenço; Aline Paes; Tillman Weyde; Audrey Depeige; Mohnish Dubey
>
> **备注:** Accepted to publication at the 19th Conference of the European Chapter of the Association for Computational Linguistics, EACL 2026
>
> **摘要:** Claim verification is a core component of automated fact-checking systems, aimed at determining the truthfulness of a statement by assessing it against reliable evidence sources such as documents or knowledge bases. This work presents KG-CRAFT, a method that improves automatic claim verification by leveraging large language models (LLMs) augmented with contrastive questions grounded in a knowledge graph. KG-CRAFT first constructs a knowledge graph from claims and associated reports, then formulates contextually relevant contrastive questions based on the knowledge graph structure. These questions guide the distillation of evidence-based reports, which are synthesised into a concise summary that is used for veracity assessment by LLMs. Extensive evaluations on two real-world datasets (LIAR-RAW and RAWFC) demonstrate that our method achieves a new state-of-the-art in predictive performance. Comprehensive analyses validate in detail the effectiveness of our knowledge graph-based contrastive reasoning approach in improving LLMs' fact-checking capabilities.
>
---
#### [new 013] MetaGen: Self-Evolving Roles and Topologies for Multi-Agent LLM Reasoning
- **分类: cs.CL**

- **简介: 该论文提出MetaGen，解决多智能体系统中角色与协作结构固定导致的适应性差问题。通过动态调整角色和拓扑，提升任务执行效果与效率。属于多智能体推理任务。**

- **链接: [https://arxiv.org/pdf/2601.19290v1](https://arxiv.org/pdf/2601.19290v1)**

> **作者:** Yimeng Wang; Jiaxing Zhao; Hongbin Xie; Hexing Ma; Yuzhen Lei; Shuangxue Liu; Xuan Song; Zichen Zhang; Haoran Zhang
>
> **摘要:** Large language models are increasingly deployed as multi-agent systems, where specialized roles communicate and collaborate through structured interactions to solve complex tasks that often exceed the capacity of a single agent. However, most existing systems still rely on a fixed role library and an execution-frozen interaction topology, a rigid design choice that frequently leads to task mismatch, prevents timely adaptation when new evidence emerges during reasoning, and further inflates inference cost. We introduce MetaGen, a training-free framework that adapts both the role space and the collaboration topology at inference time, without updating base model weights. MetaGen generates and rewrites query-conditioned role specifications to maintain a controllable dynamic role pool, then instantiates a constrained execution graph around a minimal backbone. During execution, it iteratively updates role prompts and adjusts structural decisions using lightweight feedback signals. Experiments on code generation and multi-step reasoning benchmarks show that MetaGen improves the accuracy and cost tradeoff over strong multi-agent baselines.
>
---
#### [new 014] Component-Level Lesioning of Language Models Reveals Clinically Aligned Aphasia Phenotypes
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型研究任务，旨在通过组件级损伤模拟失语症类型。工作包括构建框架、识别相关组件并评估语言障碍程度。**

- **链接: [https://arxiv.org/pdf/2601.19723v1](https://arxiv.org/pdf/2601.19723v1)**

> **作者:** Yifan Wang; Jichen Zheng; Jingyuan Sun; Yunhao Zhang; Chunyu Ye; Jixing Li; Chengqing Zong; Shaonan Wang
>
> **摘要:** Large language models (LLMs) increasingly exhibit human-like linguistic behaviors and internal representations that they could serve as computational simulators of language cognition. We ask whether LLMs can be systematically manipulated to reproduce language-production impairments characteristic of aphasia following focal brain lesions. Such models could provide scalable proxies for testing rehabilitation hypotheses, and offer a controlled framework for probing the functional organization of language. We introduce a clinically grounded, component-level framework that simulates aphasia by selectively perturbing functional components in LLMs, and apply it to both modular Mixture-of-Experts models and dense Transformers using a unified intervention interface. Our pipeline (i) identifies subtype-linked components for Broca's and Wernicke's aphasia, (ii) interprets these components with linguistic probing tasks, and (iii) induces graded impairments by progressively perturbing the top-k subtype-linked components, evaluating outcomes with Western Aphasia Battery (WAB) subtests summarized by Aphasia Quotient (AQ). Across architectures and lesioning strategies, subtype-targeted perturbations yield more systematic, aphasia-like regressions than size-matched random perturbations, and MoE modularity supports more localized and interpretable phenotype-to-component mappings. These findings suggest that modular LLMs, combined with clinically informed component perturbations, provide a promising platform for simulating aphasic language production and studying how distinct language functions degrade under targeted disruptions.
>
---
#### [new 015] Formula-One Prompting: Adaptive Reasoning Through Equations For Applied Mathematics
- **分类: cs.CL**

- **简介: 该论文属于数学推理任务，旨在解决应用数学问题中缺乏方程引导的问题。提出F-1方法，通过方程中间表示提升推理效果。**

- **链接: [https://arxiv.org/pdf/2601.19302v1](https://arxiv.org/pdf/2601.19302v1)**

> **作者:** Natapong Nitarach; Pittawat Taveekitworachai; Kunat Pipatanakul
>
> **摘要:** Prompting techniques such as Chain-of-Thought (CoT) and Program-of-Thought (PoT) improve LLM mathematical reasoning by structuring intermediate steps in natural language or code. However, applied mathematics problems in domains like finance, physics, and cryptography often require recalling or deriving governing equations, a step that current approaches do not explicitly leverage. We propose Formula-One Prompting (F-1), a two-phase approach that uses mathematical equations as an intermediate representation before adaptive solving. F-1 first formulates governing equations from problem descriptions, then selects a solving strategy among CoT, PoT, or direct computation based on the generated equations, all within a single LLM call. Results across five models and four benchmarks show F-1 outperforms CoT by +5.76% and PoT by +8.42% on average. Crucially, gains are largest in applied domains: +13.30% on FinanceMath over CoT, and within OlympiadBench, larger gains on physics (+2.55%) than pure math (+0.44%). This demonstrates that F-1 is more effective than CoT in applied mathematics problems.
>
---
#### [new 016] Leveraging Sentence-oriented Augmentation and Transformer-Based Architecture for Vietnamese-Bahnaric Translation
- **分类: cs.CL**

- **简介: 该论文属于越南语到巴拿语的机器翻译任务，旨在解决资源匮乏导致的翻译困难，通过句子级增强和Transformer架构提升翻译效果。**

- **链接: [https://arxiv.org/pdf/2601.19124v1](https://arxiv.org/pdf/2601.19124v1)**

> **作者:** Tan Sang Nguyen; Quoc Nguyen Pham; Tho Quan
>
> **摘要:** The Bahnar people, an ethnic minority in Vietnam with a rich ancestral heritage, possess a language of immense cultural and historical significance. The government places a strong emphasis on preserving and promoting the Bahnaric language by making it accessible online and encouraging communication across generations. Recent advancements in artificial intelligence, such as Neural Machine Translation (NMT), have brought about a transformation in translation by improving accuracy and fluency. This, in turn, contributes to the revival of the language through educational efforts, communication, and documentation. Specifically, NMT is pivotal in enhancing accessibility for Bahnaric speakers, making information and content more readily available. Nevertheless, the translation of Vietnamese into Bahnaric faces practical challenges due to resource constraints, especially given the limited resources available for the Bahnaric language. To address this, we employ state-of-the-art techniques in NMT along with two augmentation strategies for domain-specific Vietnamese-Bahnaric translation task. Importantly, both approaches are flexible and can be used with various neural machine translation models. Additionally, they do not require complex data preprocessing steps, the training of additional systems, or the acquisition of extra data beyond the existing training parallel corpora.
>
---
#### [new 017] ReToP: Learning to Rewrite Electronic Health Records for Clinical Prediction
- **分类: cs.CL**

- **简介: 该论文提出ReToP框架，解决临床预测中EHR数据质量差的问题，通过端到端训练EHR重写和预测模块，提升预测效果。**

- **链接: [https://arxiv.org/pdf/2601.19286v1](https://arxiv.org/pdf/2601.19286v1)**

> **作者:** Jesus Lovon-Melgarejo; Jose G. Moreno; Christine Damase-Michel; Lynda Tamine
>
> **备注:** Accepted by WSDM 2026
>
> **摘要:** Electronic Health Records (EHRs) provide crucial information for clinical decision-making. However, their high-dimensionality, heterogeneity, and sparsity make clinical prediction challenging. Large Language Models (LLMs) allowed progress towards addressing this challenge by leveraging parametric medical knowledge to enhance EHR data for clinical prediction tasks. Despite the significant achievements made so far, most of the existing approaches are fundamentally task-agnostic in the sense that they deploy LLMs as EHR encoders or EHR completion modules without fully integrating signals from the prediction tasks. This naturally hinders task performance accuracy. In this work, we propose Rewrite-To-Predict (ReToP), an LLM-based framework that addresses this limitation through an end-to-end training of an EHR rewriter and a clinical predictor. To cope with the lack of EHR rewrite training data, we generate synthetic pseudo-labels using clinical-driven feature selection strategies to create diverse patient rewrites for fine-tuning the EHR rewriter. ReToP aligns the rewriter with prediction objectives using a novel Classifier Supervised Contribution (CSC) score that enables the EHR rewriter to generate clinically relevant rewrites that directly enhance prediction. Our ReToP framework surpasses strong baseline models across three clinical tasks on MIMIC-IV. Moreover, the analysis of ReToP shows its generalizability to unseen datasets and tasks with minimal fine-tuning while preserving faithful rewrites and emphasizing task-relevant predictive features.
>
---
#### [new 018] One Token Is Enough: Improving Diffusion Language Models with a Sink Token
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，针对扩散语言模型中的不稳定问题，提出引入一个额外的结构化sink token以稳定注意力机制，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.19657v1](https://arxiv.org/pdf/2601.19657v1)**

> **作者:** Zihou Zhang; Zheyong Xie; Li Zhong; Haifeng Liu; Shaosheng Cao
>
> **摘要:** Diffusion Language Models (DLMs) have emerged as a compelling alternative to autoregressive approaches, enabling parallel text generation with competitive performance. Despite these advantages, there is a critical instability in DLMs: the moving sink phenomenon. Our analysis indicates that sink tokens exhibit low-norm representations in the Transformer's value space, and that the moving sink phenomenon serves as a protective mechanism in DLMs to prevent excessive information mixing. However, their unpredictable positions across diffusion steps undermine inference robustness. To resolve this, we propose a simple but effective extra sink token implemented via a modified attention mask. Specifically, we introduce a special token constrained to attend solely to itself, while remaining globally visible to all other tokens. Experimental results demonstrate that introducing a single extra token stabilizes attention sinks, substantially improving model performance. Crucially, further analysis confirms that the effectiveness of this token is independent of its position and characterized by negligible semantic content, validating its role as a robust and dedicated structural sink.
>
---
#### [new 019] Malicious Repurposing of Open Science Artefacts by Using Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于安全评估任务，探讨LLMs被恶意利用 repurpose 开源科学成果的风险。工作包括构建攻击流程并评估其危害性。**

- **链接: [https://arxiv.org/pdf/2601.18998v1](https://arxiv.org/pdf/2601.18998v1)**

> **作者:** Zahra Hashemi; Zhiqiang Zhong; Jun Pang; Wei Zhao
>
> **摘要:** The rapid evolution of large language models (LLMs) has fuelled enthusiasm about their role in advancing scientific discovery, with studies exploring LLMs that autonomously generate and evaluate novel research ideas. However, little attention has been given to the possibility that such models could be exploited to produce harmful research by repurposing open science artefacts for malicious ends. We fill the gap by introducing an end-to-end pipeline that first bypasses LLM safeguards through persuasion-based jailbreaking, then reinterprets NLP papers to identify and repurpose their artefacts (datasets, methods, and tools) by exploiting their vulnerabilities, and finally assesses the safety of these proposals using our evaluation framework across three dimensions: harmfulness, feasibility of misuse, and soundness of technicality. Overall, our findings demonstrate that LLMs can generate harmful proposals by repurposing ethically designed open artefacts; however, we find that LLMs acting as evaluators strongly disagree with one another on evaluation outcomes: GPT-4.1 assigns higher scores (indicating greater potential harms, higher soundness and feasibility of misuse), Gemini-2.5-pro is markedly stricter, and Grok-3 falls between these extremes. This indicates that LLMs cannot yet serve as reliable judges in a malicious evaluation setup, making human evaluation essential for credible dual-use risk assessment.
>
---
#### [new 020] How Do Transformers Learn to Associate Tokens: Gradient Leading Terms Bring Mechanistic Interpretability
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究Transformer模型如何学习词义关联，通过分析梯度揭示权重形成机制，解决模型内部语义关联的可解释性问题。**

- **链接: [https://arxiv.org/pdf/2601.19208v1](https://arxiv.org/pdf/2601.19208v1)**

> **作者:** Shawn Im; Changdae Oh; Zhen Fang; Sharon Li
>
> **备注:** ICLR 2026
>
> **摘要:** Semantic associations such as the link between "bird" and "flew" are foundational for language modeling as they enable models to go beyond memorization and instead generalize and generate coherent text. Understanding how these associations are learned and represented in language models is essential for connecting deep learning with linguistic theory and developing a mechanistic foundation for large language models. In this work, we analyze how these associations emerge from natural language data in attention-based language models through the lens of training dynamics. By leveraging a leading-term approximation of the gradients, we develop closed-form expressions for the weights at early stages of training that explain how semantic associations first take shape. Through our analysis, we reveal that each set of weights of the transformer has closed-form expressions as simple compositions of three basis functions (bigram, token-interchangeability, and context mappings), reflecting the statistics of the text corpus and uncovering how each component of the transformer captures semantic associations based on these compositions. Experiments on real-world LLMs demonstrate that our theoretical weight characterizations closely match the learned weights, and qualitative analyses further show how our theorem shines light on interpreting the learned associations in transformers.
>
---
#### [new 021] TokenSeek: Memory Efficient Fine Tuning via Instance-Aware Token Ditching
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型微调任务，解决大模型微调内存消耗高的问题。通过实例感知的token选择与舍弃，实现高效内存使用。**

- **链接: [https://arxiv.org/pdf/2601.19739v1](https://arxiv.org/pdf/2601.19739v1)**

> **作者:** Runjia Zeng; Qifan Wang; Qiang Guan; Ruixiang Tang; Lifu Huang; Zhenting Wang; Xueling Zhang; Cheng Han; Dongfang Liu
>
> **备注:** ICLR 2026
>
> **摘要:** Fine tuning has been regarded as a de facto approach for adapting large language models (LLMs) to downstream tasks, but the high training memory consumption inherited from LLMs makes this process inefficient. Among existing memory efficient approaches, activation-related optimization has proven particularly effective, as activations consistently dominate overall memory consumption. Although prior arts offer various activation optimization strategies, their data-agnostic nature ultimately results in ineffective and unstable fine tuning. In this paper, we propose TokenSeek, a universal plugin solution for various transformer-based models through instance-aware token seeking and ditching, achieving significant fine-tuning memory savings (e.g., requiring only 14.8% of the memory on Llama3.2 1B) with on-par or even better performance. Furthermore, our interpretable token seeking process reveals the underlying reasons for its effectiveness, offering valuable insights for future research on token efficiency. Homepage: https://runjia.tech/iclr_tokenseek/
>
---
#### [new 022] DREAMSTATE: Diffusing States and Parameters for Recurrent Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决RNN状态表示与编辑问题。提出DREAMSTATE框架，利用扩散模型生成和编辑状态，提升模型灵活性与性能。**

- **链接: [https://arxiv.org/pdf/2601.19221v1](https://arxiv.org/pdf/2601.19221v1)**

> **作者:** Liu Xiao
>
> **摘要:** Modern Recurrent Neural Networks (RNNs), such as RWKV, are distinguished by their powerful short-range modeling capabilities and efficient fixed-size states, which constitute a core advantage over standard Transformers. However, there is a significant lack of research into their internal state as an editable knowledge representation. To fill this gap, we first explore the representational properties of the RWKV state by proposing the DREAMSTATE framework. This framework utilizes a conditional Diffusion Transformer (DiT) to directly model the probability manifold of the state, enabling its generation and editing. The structural nature of this representation is validated through t-SNE visualizations and controlled generation experiments. After successfully uncovering and modeling the state's representational potential, we further propose a novel hybrid architecture that combines the local advantages of RNNs with global context adaptability. This architecture features a parallel DiT that processes a variable-length global context to dynamically generate and adjust the core recurrent module's WKV parameters, transforming the fixed recurrence mechanism into a context-aware dynamic function. Experiments demonstrate that this hybrid model can be trained stably via a multi-objective loss, validating its design feasibility. Our work not only opens a new research direction for RNN state representation but also provides a concrete architectural reference for future model design. The code is publicly available at: https://huggingface.co/2dgx41s/DreamState.
>
---
#### [new 023] DiaDem: Advancing Dialogue Descriptions in Audiovisual Video Captioning for Multimodal Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于音频视频字幕生成任务，旨在解决对话描述不准确的问题。提出DiaDem模型和DiaDemBench基准，提升对话描述的准确性与可靠性。**

- **链接: [https://arxiv.org/pdf/2601.19267v1](https://arxiv.org/pdf/2601.19267v1)**

> **作者:** Xinlong Chen; Weihong Lin; Jingyun Hua; Linli Yao; Yue Ding; Bozhou Li; Bohan Zeng; Yang Shi; Qiang Liu; Yuanxing Zhang; Pengfei Wan; Liang Wang; Tieniu Tan
>
> **备注:** Project webpage: https://diadem-captioner.github.io/
>
> **摘要:** Accurate dialogue description in audiovisual video captioning is crucial for downstream understanding and generation tasks. However, existing models generally struggle to produce faithful dialogue descriptions within audiovisual captions. To mitigate this limitation, we propose DiaDem, a powerful audiovisual video captioning model capable of generating captions with more precise dialogue descriptions while maintaining strong overall performance. We first synthesize a high-quality dataset for SFT, then employ a difficulty-partitioned two-stage GRPO strategy to further enhance dialogue descriptions. To enable systematic evaluation of dialogue description capabilities, we introduce DiaDemBench, a comprehensive benchmark designed to evaluate models across diverse dialogue scenarios, emphasizing both speaker attribution accuracy and utterance transcription fidelity in audiovisual captions. Extensive experiments on DiaDemBench reveal even commercial models still exhibit substantial room for improvement in dialogue-aware captioning. Notably, DiaDem not only outperforms the Gemini series in dialogue description accuracy but also achieves competitive performance on general audiovisual captioning benchmarks, demonstrating its overall effectiveness.
>
---
#### [new 024] Optimizing Conversational Quality in Spoken Dialogue Systems with Reinforcement Learning from AI Feedback
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于对话系统优化任务，解决传统RLHF方法在多维对话质量评估上的不足，提出多奖励RLAIF框架，提升语义、音频和情感一致性。**

- **链接: [https://arxiv.org/pdf/2601.19063v1](https://arxiv.org/pdf/2601.19063v1)**

> **作者:** Siddhant Arora; Jinchuan Tian; Jiatong Shi; Hayato Futami; Yosuke Kashiwagi; Emiru Tsunoo; Shinji Watanabe
>
> **摘要:** Reinforcement learning from human or AI feedback (RLHF/RLAIF) for speech-in/speech-out dialogue systems (SDS) remains underexplored, with prior work largely limited to single semantic rewards applied at the utterance level. Such setups overlook the multi-dimensional and multi-modal nature of conversational quality, which encompasses semantic coherence, audio naturalness, speaker consistency, emotion alignment, and turn-taking behavior. Moreover, they are fundamentally mismatched with duplex spoken dialogue systems that generate responses incrementally, where agents must make decisions based on partial utterances. We address these limitations with the first multi-reward RLAIF framework for SDS, combining semantic, audio-quality, and emotion-consistency rewards. To align utterance-level preferences with incremental, blockwise decoding in duplex models, we apply turn-level preference sampling and aggregate per-block log-probabilities within a single DPO objective. We present the first systematic study of preference learning for improving SDS quality in both multi-turn Chain-of-Thought and blockwise duplex models, and release a multi-reward DPO dataset to support reproducible research. Experiments show that single-reward RLAIF selectively improves its targeted metric, while joint multi-reward training yields consistent gains across semantic quality and audio naturalness. These results highlight the importance of holistic, multi-reward alignment for practical conversational SDS.
>
---
#### [new 025] Transparency-First Medical Language Models: Datasheets, Model Cards, and End-to-End Data Provenance for Clinical NLP
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出TeMLM，解决临床NLP模型的透明性问题，通过数据集、模型卡片等工具实现数据与模型的可追溯和审计。**

- **链接: [https://arxiv.org/pdf/2601.19191v1](https://arxiv.org/pdf/2601.19191v1)**

> **作者:** Olaf Yunus Laitinen Imanov; Taner Yilmaz; Ayse Tuba Tugrul; Melike Nesrin Zaman; Ozkan Gunalp; Duygu Erisken; Sila Burde Dulger; Rana Irem Turhan; Izzet Ozdemir; Derya Umut Kulali; Ozan Akbulut; Harun Demircioglu; Hasan Basri Kara; Berfin Tavan
>
> **备注:** 12 pages, 9 figures, 15 tables. Technetium-I case study and ProtactiniumBERT-100M reference benchmarks
>
> **摘要:** We introduce TeMLM, a set of transparency-first release artifacts for clinical language models. TeMLM unifies provenance, data transparency, modeling transparency, and governance into a single, machine-checkable release bundle. We define an artifact suite (TeMLM-Card, TeMLM-Datasheet, TeMLM-Provenance) and a lightweight conformance checklist for repeatable auditing. We instantiate the artifacts on Technetium-I, a large-scale synthetic clinical NLP dataset with 498,000 notes, 7.74M PHI entity annotations across 10 types, and ICD-9-CM diagnosis labels, and report reference results for ProtactiniumBERT (about 100 million parameters) on PHI de-identification (token classification) and top-50 ICD-9 code extraction (multi-label classification). We emphasize that synthetic benchmarks are valuable for tooling and process validation, but models should be validated on real clinical data prior to deployment.
>
---
#### [new 026] RATE: Reviewer Profiling and Annotation-free Training for Expertise Ranking in Peer Review Systems
- **分类: cs.CL**

- **简介: 该论文属于专家匹配任务，解决LLM时代评审者分配难题。通过构建新基准LR-bench和提出RATE框架，提升评审者与论文的匹配效果。**

- **链接: [https://arxiv.org/pdf/2601.19637v1](https://arxiv.org/pdf/2601.19637v1)**

> **作者:** Weicong Liu; Zixuan Yang; Yibo Zhao; Xiang Li
>
> **备注:** 18 pages
>
> **摘要:** Reviewer assignment is increasingly critical yet challenging in the LLM era, where rapid topic shifts render many pre-2023 benchmarks outdated and where proxy signals poorly reflect true reviewer familiarity. We address this evaluation bottleneck by introducing LR-bench, a high-fidelity, up-to-date benchmark curated from 2024-2025 AI/NLP manuscripts with five-level self-assessed familiarity ratings collected via a large-scale email survey, yielding 1055 expert-annotated paper-reviewer-score annotations. We further propose RATE, a reviewer-centric ranking framework that distills each reviewer's recent publications into compact keyword-based profiles and fine-tunes an embedding model with weak preference supervision constructed from heuristic retrieval signals, enabling matching each manuscript against a reviewer profile directly. Across LR-bench and the CMU gold-standard dataset, our approach consistently achieves state-of-the-art performance, outperforming strong embedding baselines by a clear margin. We release LR-bench at https://huggingface.co/datasets/Gnociew/LR-bench, and a GitHub repository at https://github.com/Gnociew/RATE-Reviewer-Assign.
>
---
#### [new 027] RPO-RAG: Aligning Small LLMs with Relation-aware Preference Optimization for Knowledge Graph Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识图谱问答任务，旨在提升小规模语言模型的推理能力。针对现有方法在路径采样和对齐上的不足，提出RPO-RAG框架，优化知识图谱的使用效果。**

- **链接: [https://arxiv.org/pdf/2601.19225v1](https://arxiv.org/pdf/2601.19225v1)**

> **作者:** Kaehyun Um; KyuHwan Yeom; Haerim Yang; Minyoung Choi; Hyeongjun Yang; Kyong-Ho Lee
>
> **备注:** Accepted at The Web Conference (WWW) 2026
>
> **摘要:** Large Language Models (LLMs) have recently demonstrated remarkable reasoning abilities, yet hallucinate on knowledge-intensive tasks. Retrieval-augmented generation (RAG) mitigates this issue by grounding answers in external sources, e.g., knowledge graphs (KGs). However, existing KG-based RAG approaches rely on semantics-unaware path sampling and are weakly aligned with KG reasoning objectives, which limits further accuracy gains. They also feed retrieved paths directly into the reasoner without organizing them into answer-centered reasoning paths, hindering small LLMs' ability to leverage the retrieved knowledge. Furthermore, prior works predominantly rely on large LLMs (e.g., ChatGPT/GPT-4) or assume backbones above 7B parameters, leaving sub-7B models underexplored. We address this gap with RPO-RAG, the first KG-based RAG framework specifically designed for small LLMs, to the best of our knowledge. RPO-RAG introduces three key innovations: (1) a query-path semantic sampling strategy that provides informative supervisory signals; (2) a relation-aware preference optimization that aligns training with intermediate KG reasoning signals (e.g., relation); and (3) an answer-centered prompt design that organizes entities and reasoning paths in an interpretable format. Extensive experiments on two benchmark Knowledge Graph Question Answering (KGQA) datasets, WebQSP and CWQ, demonstrate that RPO-RAG effectively bridges the performance gap between small and large language models. On WebQSP, it improves F1 by up to 8.8%, reflecting enhanced answer precision, while on CWQ it achieves new state-of-the-art results among models under 8B parameters in both Hit and F1. Overall, RPO-RAG substantially improves the reasoning capability of small LLMs, even under 3B parameters-highlighting their potential for resource-efficient and practical on-device KGQA applications.
>
---
#### [new 028] GradPruner: Gradient-Guided Layer Pruning Enabling Efficient Fine-Tuning and Inference for LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型压缩任务，旨在提升大语言模型微调与推理的效率。通过梯度引导的层剪枝方法，减少参数量并保持较高精度。**

- **链接: [https://arxiv.org/pdf/2601.19503v1](https://arxiv.org/pdf/2601.19503v1)**

> **作者:** Wei Huang; Anda Cheng; Yinggui Wang
>
> **备注:** Accepted by ICLR2026
>
> **摘要:** Fine-tuning Large Language Models (LLMs) with downstream data is often considered time-consuming and expensive. Structured pruning methods are primarily employed to improve the inference efficiency of pre-trained models. Meanwhile, they often require additional time and memory for training, knowledge distillation, structure search, and other strategies, making efficient model fine-tuning challenging to achieve. To simultaneously enhance the training and inference efficiency of downstream task fine-tuning, we introduce GradPruner, which can prune layers of LLMs guided by gradients in the early stages of fine-tuning. GradPruner uses the cumulative gradients of each parameter during the initial phase of fine-tuning to compute the Initial Gradient Information Accumulation Matrix (IGIA-Matrix) to assess the importance of layers and perform pruning. We sparsify the pruned layers based on the IGIA-Matrix and merge them with the remaining layers. Only elements with the same sign are merged to reduce interference from sign variations. We conducted extensive experiments on two LLMs across eight downstream datasets. Including medical, financial, and general benchmark tasks. The results demonstrate that GradPruner has achieved a parameter reduction of 40% with only a 0.99% decrease in accuracy. Our code is publicly available.
>
---
#### [new 029] Dynamic Multi-Expert Projectors with Stabilized Routing for Multilingual Speech Recognition
- **分类: cs.CL**

- **简介: 该论文属于多语言语音识别任务，解决单个投影器难以处理多语言声学到语义映射的问题。提出SMEAR-MoE动态多专家投影器，提升性能并实现跨语言共享。**

- **链接: [https://arxiv.org/pdf/2601.19451v1](https://arxiv.org/pdf/2601.19451v1)**

> **作者:** Isha Pandey; Ashish Mittal; Vartul Bahuguna; Ganesh Ramakrishnan
>
> **摘要:** Recent advances in LLM-based ASR connect frozen speech encoders with Large Language Models (LLMs) via lightweight projectors. While effective in monolingual settings, a single projector struggles to capture the diverse acoustic-to-semantic mappings required for multilingual ASR. To address this, we propose SMEAR-MoE, a stabilized Mixture-of-Experts projector that ensures dense gradient flow to all experts, preventing expert collapse while enabling cross-lingual sharing. We systematically compare monolithic, static multi-projector, and dynamic MoE designs across four Indic languages (Hindi, Marathi, Tamil, Telugu). Our SMEAR-MoE achieves strong performance, delivering upto a 7.6% relative WER reduction over the single-projector baseline, while maintaining comparable runtime efficiency. Analysis of expert routing further shows linguistically meaningful specialization, with related languages sharing experts. These results demonstrate that stable multi-expert projectors are key to scalable and robust multilingual ASR.
>
---
#### [new 030] Flatter Tokens are More Valuable for Speculative Draft Model Training
- **分类: cs.CL**

- **简介: 该论文属于语言模型训练任务，解决SD训练效率低的问题。通过分析token的平坦性，提出SFDD方法，提升训练速度并减少数据需求。**

- **链接: [https://arxiv.org/pdf/2601.18902v1](https://arxiv.org/pdf/2601.18902v1)**

> **作者:** Jiaming Fan; Daming Cao; Xiangzhong Luo; Jiale Fu; Chonghan Liu; Xu Yang
>
> **摘要:** Speculative Decoding (SD) is a key technique for accelerating Large Language Model (LLM) inference, but it typically requires training a draft model on a large dataset. We approach this problem from a data-centric perspective, finding that not all training samples contribute equally to the SD acceptance rate. Specifically, our theoretical analysis and empirical validation reveals that tokens inducing flatter predictive distributions from the target model are more valuable than those yielding sharply peaked distributions. Based on this insight, we propose flatness, a new metric to quantify this property, and develop the Sample-level-flatness-based Dataset Distillation (SFDD) approach, which filters the training data to retain only the most valuable samples. Experiments on the EAGLE framework demonstrate that SFDD can achieve over 2$\times$ training speedup using only 50% of the data, while keeping the final model's inference speedup within 4% of the full-dataset baseline. This work introduces an effective, data-centric approach that substantially improves the training efficiency for Speculative Decoding. Our code is available at https://anonymous.4open.science/r/Flatness.
>
---
#### [new 031] When Benchmarks Leak: Inference-Time Decontamination for LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型评估任务，解决测试集污染导致性能虚高的问题。提出DeconIEP框架，在推理时通过输入扰动实现去污染，有效降低记忆路径影响，保持模型正常性能。**

- **链接: [https://arxiv.org/pdf/2601.19334v1](https://arxiv.org/pdf/2601.19334v1)**

> **作者:** Jianzhe Chai; Yu Zhe; Jun Sakuma
>
> **摘要:** Benchmark-based evaluation is the de facto standard for comparing large language models (LLMs). However, its reliability is increasingly threatened by test set contamination, where test samples or their close variants leak into training data and artificially inflate reported performance. To address this issue, prior work has explored two main lines of mitigation. One line attempts to identify and remove contaminated benchmark items before evaluation, but this inevitably alters the evaluation set itself and becomes unreliable when contamination is moderate or severe. The other line preserves the benchmark and instead suppresses contaminated behavior at evaluation time; however, such interventions often interfere with normal inference and lead to noticeable performance degradation on clean inputs. We propose DeconIEP, a decontamination framework that operates entirely during evaluation by applying small, bounded perturbations in the input embedding space. Guided by a relatively less-contaminated reference model, DeconIEP learns an instance-adaptive perturbation generator that steers the evaluated model away from memorization-driven shortcut pathways. Across multiple open-weight LLMs and benchmarks, extensive empirical results show that DeconIEP achieves strong decontamination effectiveness while incurring only minimal degradation in benign utility.
>
---
#### [new 032] ClaimPT: A Portuguese Dataset of Annotated Claims in News Articles
- **分类: cs.CL**

- **简介: 该论文属于事实核查任务，旨在解决葡萄牙语事实核查数据缺失的问题。作者构建了ClaimPT数据集，包含新闻文章中的事实性声明标注，以促进低资源语言的事实核查研究。**

- **链接: [https://arxiv.org/pdf/2601.19490v1](https://arxiv.org/pdf/2601.19490v1)**

> **作者:** Ricardo Campos; Raquel Sequeira; Sara Nerea; Inês Cantante; Diogo Folques; Luís Filipe Cunha; João Canavilhas; António Branco; Alípio Jorge; Sérgio Nunes; Nuno Guimarães; Purificação Silvano
>
> **摘要:** Fact-checking remains a demanding and time-consuming task, still largely dependent on manual verification and unable to match the rapid spread of misinformation online. This is particularly important because debunking false information typically takes longer to reach consumers than the misinformation itself; accelerating corrections through automation can therefore help counter it more effectively. Although many organizations perform manual fact-checking, this approach is difficult to scale given the growing volume of digital content. These limitations have motivated interest in automating fact-checking, where identifying claims is a crucial first step. However, progress has been uneven across languages, with English dominating due to abundant annotated data. Portuguese, like other languages, still lacks accessible, licensed datasets, limiting research, NLP developments and applications. In this paper, we introduce ClaimPT, a dataset of European Portuguese news articles annotated for factual claims, comprising 1,308 articles and 6,875 individual annotations. Unlike most existing resources based on social media or parliamentary transcripts, ClaimPT focuses on journalistic content, collected through a partnership with LUSA, the Portuguese News Agency. To ensure annotation quality, two trained annotators labeled each article, with a curator validating all annotations according to a newly proposed scheme. We also provide baseline models for claim detection, establishing initial benchmarks and enabling future NLP and IR applications. By releasing ClaimPT, we aim to advance research on low-resource fact-checking and enhance understanding of misinformation in news media.
>
---
#### [new 033] Decompose-and-Formalise: Recursively Verifiable Natural Language Inference
- **分类: cs.CL**

- **简介: 该论文属于自然语言推理任务，解决长文本和多步论证中的自动形式化错误问题。提出分解与形式化框架，通过分解前提与假设为推理树，局部验证并修复错误，提升解释验证率。**

- **链接: [https://arxiv.org/pdf/2601.19605v1](https://arxiv.org/pdf/2601.19605v1)**

> **作者:** Xin Quan; Marco Valentino; Louise A. Dennis; André Freitas
>
> **摘要:** Recent work has shown that integrating large language models (LLMs) with theorem provers (TPs) in neuro-symbolic pipelines helps with entailment verification and proof-guided refinement of explanations for natural language inference (NLI). However, scaling such refinement to naturalistic NLI remains difficult: long, syntactically rich inputs and deep multi-step arguments amplify autoformalisation errors, where a single local mismatch can invalidate the proof. Moreover, current methods often handle failures via costly global regeneration due to the difficulty of localising the responsible span or step from prover diagnostics. Aiming to address these problems, we propose a decompose-and-formalise framework that (i) decomposes premise-hypothesis pairs into an entailment tree of atomic steps, (ii) verifies the tree bottom-up to isolate failures to specific nodes, and (iii) performs local diagnostic-guided refinement instead of regenerating the whole explanation. Moreover, to improve faithfulness of autoformalisation, we introduce $θ$-substitution in an event-based logical form to enforce consistent argument-role bindings. Across a range of reasoning tasks using five LLM backbones, our method achieves the highest explanation verification rates, improving over the state-of-the-art by 26.2%, 21.7%, 21.6% and 48.9%, while reducing refinement iterations and runtime and preserving strong NLI accuracy.
>
---
#### [new 034] Language Family Matters: Evaluating LLM-Based ASR Across Linguistic Boundaries
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属于语音识别任务，解决多语言ASR中资源不足与泛化能力差的问题。通过基于语系的连接器共享策略，提升模型效率与跨语言泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.18899v1](https://arxiv.org/pdf/2601.18899v1)**

> **作者:** Yuchen Zhang; Ravi Shekhar; Haralambos Mouratidis
>
> **摘要:** Large Language Model (LLM)-powered Automatic Speech Recognition (ASR) systems achieve strong performance with limited resources by linking a frozen speech encoder to a pretrained LLM via a lightweight connector. Prior work trains a separate connector per language, overlooking linguistic relatedness. We propose an efficient and novel connector-sharing strategy based on linguistic family membership, enabling one connector per family, and empirically validate its effectiveness across two multilingual LLMs and two real-world corpora spanning curated and crowd-sourced speech. Our results show that family-based connectors reduce parameter count while improving generalization across domains, offering a practical and scalable strategy for multilingual ASR deployment.
>
---
#### [new 035] BabyReasoningBench: Generating Developmentally-Inspired Reasoning Tasks for Evaluating Baby Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出BabyReasoningBench，用于评估婴儿语言模型的推理能力。针对传统评估与婴儿模型训练数据不匹配的问题，设计发展心理学启发的推理任务，分析模型在不同推理类型上的表现。**

- **链接: [https://arxiv.org/pdf/2601.18933v1](https://arxiv.org/pdf/2601.18933v1)**

> **作者:** Kaustubh D. Dhole
>
> **摘要:** Traditional evaluations of reasoning capabilities of language models are dominated by adult-centric benchmarks that presuppose broad world knowledge, complex instruction following, and mature pragmatic competence. These assumptions are mismatched to baby language models trained on developmentally plausible input such as child-directed speech and early-childhood narratives, and they obscure which reasoning abilities (if any) emerge under such constraints. We introduce BabyReasoningBench, a GPT-5.2 generated benchmark of 19 reasoning tasks grounded in classic paradigms from developmental psychology, spanning theory of mind, analogical and relational reasoning, causal inference and intervention selection, and core reasoning primitives that are known to be confounded by memory and pragmatics. We find that two GPT-2 based baby language models (pretrained on 10M and 100M of child-directed speech text) show overall low but uneven performance, with dissociations across task families: scaling improves several causal and physical reasoning tasks, while belief attribution and pragmatics-sensitive tasks remain challenging. BabyReasoningBench provides a developmentally grounded lens for analyzing what kinds of reasoning are supported by child-like training distributions, and for testing mechanistic hypotheses about how such abilities emerge.
>
---
#### [new 036] Do LLMs Truly Benefit from Longer Context in Automatic Post-Editing?
- **分类: cs.CL**

- **简介: 该论文属于自动后编辑任务，探讨LLMs在文档级上下文中的表现。研究对比了专有和开源LLMs，分析其在APE中的质量、鲁棒性和效率，发现专有模型虽性能强但成本高且不善利用上下文。**

- **链接: [https://arxiv.org/pdf/2601.19410v1](https://arxiv.org/pdf/2601.19410v1)**

> **作者:** Ahrii Kim; Seong-heum Kim
>
> **摘要:** Automatic post-editing (APE) aims to refine machine translations by correcting residual errors. Although recent large language models (LLMs) demonstrate strong translation capabilities, their effectiveness for APE--especially under document-level context--remains insufficiently understood. We present a systematic comparison of proprietary and open-weight LLMs under a naive document-level prompting setup, analyzing APE quality, contextual behavior, robustness, and efficiency. Our results show that proprietary LLMs achieve near human-level APE quality even with simple one-shot prompting, regardless of whether document context is provided. While these models exhibit higher robustness to data poisoning attacks than open-weight counterparts, this robustness also reveals a limitation: they largely fail to exploit document-level context for contextual error correction. Furthermore, standard automatic metrics do not reliably reflect these qualitative improvements, highlighting the continued necessity of human evaluation. Despite their strong performance, the substantial cost and latency overheads of proprietary LLMs render them impractical for real-world APE deployment. Overall, our findings elucidate both the promise and current limitations of LLM-based document-aware APE, and point toward the need for more efficient long-context modeling approaches for translation refinement.
>
---
#### [new 037] Zero-Shot Stance Detection in the Wild: Dynamic Target Generation and Multi-Target Adaptation
- **分类: cs.CL**

- **简介: 该论文提出零样本立场检测任务，解决真实场景中目标动态变化的问题。构建中文数据集，设计评估指标，探索模型微调策略，提升多目标识别与立场判断效果。**

- **链接: [https://arxiv.org/pdf/2601.19802v1](https://arxiv.org/pdf/2601.19802v1)**

> **作者:** Aohua Li; Yuanshuo Zhang; Ge Gao; Bo Chen; Xiaobing Zhao
>
> **摘要:** Current stance detection research typically relies on predicting stance based on given targets and text. However, in real-world social media scenarios, targets are neither predefined nor static but rather complex and dynamic. To address this challenge, we propose a novel task: zero-shot stance detection in the wild with Dynamic Target Generation and Multi-Target Adaptation (DGTA), which aims to automatically identify multiple target-stance pairs from text without prior target knowledge. We construct a Chinese social media stance detection dataset and design multi-dimensional evaluation metrics. We explore both integrated and two-stage fine-tuning strategies for large language models (LLMs) and evaluate various baseline models. Experimental results demonstrate that fine-tuned LLMs achieve superior performance on this task: the two-stage fine-tuned Qwen2.5-7B attains the highest comprehensive target recognition score of 66.99%, while the integrated fine-tuned DeepSeek-R1-Distill-Qwen-7B achieves a stance detection F1 score of 79.26%.
>
---
#### [new 038] LVLMs and Humans Ground Differently in Referential Communication
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文属于自然语言处理任务，旨在解决AI与人类在指称沟通中建立共同语境的难题。通过实验对比不同配对的沟通效果，揭示LVLMs在互动指称解析上的局限性。**

- **链接: [https://arxiv.org/pdf/2601.19792v1](https://arxiv.org/pdf/2601.19792v1)**

> **作者:** Peter Zeng; Weiling Li; Amie Paige; Zhengxiang Wang; Panagiotis Kaliosis; Dimitris Samaras; Gregory Zelinsky; Susan Brennan; Owen Rambow
>
> **备注:** 24 pages, 16 figures, preprint
>
> **摘要:** For generative AI agents to partner effectively with human users, the ability to accurately predict human intent is critical. But this ability to collaborate remains limited by a critical deficit: an inability to model common ground. Here, we present a referential communication experiment with a factorial design involving director-matcher pairs (human-human, human-AI, AI-human, and AI-AI) that interact with multiple turns in repeated rounds to match pictures of objects not associated with any obvious lexicalized labels. We release the online pipeline for data collection, the tools and analyses for accuracy, efficiency, and lexical overlap, and a corpus of 356 dialogues (89 pairs over 4 rounds each) that unmasks LVLMs' limitations in interactively resolving referring expressions, a crucial skill that underlies human language use.
>
---
#### [new 039] Strong Reasoning Isn't Enough: Evaluating Evidence Elicitation in Interactive Diagnosis
- **分类: cs.CL**

- **简介: 该论文属于医疗诊断任务，旨在解决交互式诊断中证据收集不足的问题。通过构建基准和评估框架，提出ICR指标与REFINE策略，提升模型在不确定环境下的信息获取能力。**

- **链接: [https://arxiv.org/pdf/2601.19773v1](https://arxiv.org/pdf/2601.19773v1)**

> **作者:** Zhuohan Long; Zhijie Bao; Zhongyu Wei
>
> **摘要:** Interactive medical consultation requires an agent to proactively elicit missing clinical evidence under uncertainty. Yet existing evaluations largely remain static or outcome-centric, neglecting the evidence-gathering process. In this work, we propose an interactive evaluation framework that explicitly models the consultation process using a simulated patient and a \rev{simulated reporter} grounded in atomic evidences. Based on this representation, we introduce Information Coverage Rate (ICR) to quantify how completely an agent uncovers necessary evidence during interaction. To support systematic study, we build EviMed, an evidence-based benchmark spanning diverse conditions from common complaints to rare diseases, and evaluate 10 models with varying reasoning abilities. We find that strong diagnostic reasoning does not guarantee effective information collection, and this insufficiency acts as a primary bottleneck limiting performance in interactive settings. To address this, we propose REFINE, a strategy that leverages diagnostic verification to guide the agent in proactively resolving uncertainties. Extensive experiments demonstrate that REFINE consistently outperforms baselines across diverse datasets and facilitates effective model collaboration, enabling smaller agents to achieve superior performance under strong reasoning supervision. Our code can be found at https://github.com/NanshineLoong/EID-Benchmark .
>
---
#### [new 040] Do Images Speak Louder than Words? Investigating the Effect of Textual Misinformation in VLMs
- **分类: cs.CL**

- **简介: 该论文属于视觉-语言模型研究任务，旨在解决VLMs对文本误导信息的脆弱性问题。通过构建数据集和实验，发现VLMs易受文本误导，忽视视觉证据。**

- **链接: [https://arxiv.org/pdf/2601.19202v1](https://arxiv.org/pdf/2601.19202v1)**

> **作者:** Chi Zhang; Wenxuan Ding; Jiale Liu; Mingrui Wu; Qingyun Wu; Ray Mooney
>
> **备注:** 24 pages, 10 figures. Accepted at EACL 2026 (main conference)
>
> **摘要:** Vision-Language Models (VLMs) have shown strong multimodal reasoning capabilities on Visual-Question-Answering (VQA) benchmarks. However, their robustness against textual misinformation remains under-explored. While existing research has studied the effect of misinformation in text-only domains, it is not clear how VLMs arbitrate between contradictory information from different modalities. To bridge the gap, we first propose the CONTEXT-VQA (i.e., Conflicting Text) dataset, consisting of image-question pairs together with systematically generated persuasive prompts that deliberately conflict with visual evidence. Then, a thorough evaluation framework is designed and executed to benchmark the susceptibility of various models to these conflicting multimodal inputs. Comprehensive experiments over 11 state-of-the-art VLMs reveal that these models are indeed vulnerable to misleading textual prompts, often overriding clear visual evidence in favor of the conflicting text, and show an average performance drop of over 48.2% after only one round of persuasive conversation. Our findings highlight a critical limitation in current VLMs and underscore the need for improved robustness against textual manipulation.
>
---
#### [new 041] Up to 36x Speedup: Mask-based Parallel Inference Paradigm for Key Information Extraction in MLLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于关键信息提取任务，解决LLMs/MLLMs在KIE中因自回归推理导致的效率低下问题。提出PIP方法，通过掩码并行生成提升速度5-36倍。**

- **链接: [https://arxiv.org/pdf/2601.19613v1](https://arxiv.org/pdf/2601.19613v1)**

> **作者:** Xinzhong Wang; Ya Guo; Jing Li; Huan Chen; Yi Tu; Yijie Hong; Gongshen Liu; Huijia Zhu
>
> **备注:** Accepted by ICASSP 2026
>
> **摘要:** Key Information Extraction (KIE) from visually-rich documents (VrDs) is a critical task, for which recent Large Language Models (LLMs) and Multi-Modal Large Language Models (MLLMs) have demonstrated strong potential. However, their reliance on autoregressive inference, which generates outputs sequentially, creates a significant efficiency bottleneck, especially as KIE tasks often involve extracting multiple, semantically independent fields. To overcome this limitation, we introduce PIP: a Parallel Inference Paradigm for KIE. Our approach reformulates the problem by using "[mask]" tokens as placeholders for all target values, enabling their simultaneous generation in a single forward pass. To facilitate this paradigm, we develop a tailored mask pre-training strategy and construct large-scale supervised datasets. Experimental results show that our PIP-models achieve a 5-36x inference speedup with negligible performance degradation compared to traditional autoregressive base models. By substantially improving efficiency while maintaining high accuracy, PIP paves the way for scalable and practical real-world KIE solutions.
>
---
#### [new 042] FROST: Filtering Reasoning Outliers with Attention for Efficient Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出FROST，用于高效推理的任务，解决传统方法中冗余推理路径问题。通过注意力机制剔除异常路径，提升推理效率与准确性。**

- **链接: [https://arxiv.org/pdf/2601.19001v1](https://arxiv.org/pdf/2601.19001v1)**

> **作者:** Haozheng Luo; Zhuolin Jiang; Md Zahid Hasan; Yan Chen; Soumalya Sarkar
>
> **摘要:** We propose FROST, an attention-aware method for efficient reasoning. Unlike traditional approaches, FROST leverages attention weights to prune uncritical reasoning paths, yielding shorter and more reliable reasoning trajectories. Methodologically, we introduce the concept of reasoning outliers and design an attention-based mechanism to remove them. Theoretically, FROST preserves and enhances the model's reasoning capacity while eliminating outliers at the sentence level. Empirically, we validate FROST on four benchmarks using two strong reasoning models (Phi-4-Reasoning and GPT-OSS-20B), outperforming state-of-the-art methods such as TALE and ThinkLess. Notably, FROST achieves an average 69.68% reduction in token usage and a 26.70% improvement in accuracy over the base model. Furthermore, in evaluations of attention outlier metrics, FROST reduces the maximum infinity norm by 15.97% and the average kurtosis by 91.09% compared to the base model. Code is available at https://github.com/robinzixuan/FROST
>
---
#### [new 043] Self-Aware Knowledge Probing: Evaluating Language Models' Relational Knowledge through Confidence Calibration
- **分类: cs.CL**

- **简介: 该论文属于语言模型知识评估任务，旨在解决模型可靠性评估问题。通过提出一种新的校准探针框架，分析模型的置信度一致性与语义准确性。**

- **链接: [https://arxiv.org/pdf/2601.18901v1](https://arxiv.org/pdf/2601.18901v1)**

> **作者:** Christopher Kissling; Elena Merdjanovska; Alan Akbik
>
> **摘要:** Knowledge probing quantifies how much relational knowledge a language model (LM) has acquired during pre-training. Existing knowledge probes evaluate model capabilities through metrics like prediction accuracy and precision. Such evaluations fail to account for the model's reliability, reflected in the calibration of its confidence scores. In this paper, we propose a novel calibration probing framework for relational knowledge, covering three modalities of model confidence: (1) intrinsic confidence, (2) structural consistency and (3) semantic grounding. Our extensive analysis of ten causal and six masked language models reveals that most models, especially those pre-trained with the masking objective, are overconfident. The best-calibrated scores come from confidence estimates that account for inconsistencies due to statement rephrasing. Moreover, even the largest pre-trained models fail to encode the semantics of linguistic confidence expressions accurately.
>
---
#### [new 044] SynCABEL: Synthetic Contextualized Augmentation for Biomedical Entity Linking
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **简介: 该论文提出SynCABEL，解决生物医学实体链接中标注数据不足的问题，通过生成合成数据提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.19667v1](https://arxiv.org/pdf/2601.19667v1)**

> **作者:** Adam Remaki; Christel Gérardin; Eulàlia Farré-Maduell; Martin Krallinger; Xavier Tannier
>
> **摘要:** We present SynCABEL (Synthetic Contextualized Augmentation for Biomedical Entity Linking), a framework that addresses a central bottleneck in supervised biomedical entity linking (BEL): the scarcity of expert-annotated training data. SynCABEL leverages large language models to generate context-rich synthetic training examples for all candidate concepts in a target knowledge base, providing broad supervision without manual annotation. We demonstrate that SynCABEL, when combined with decoder-only models and guided inference establish new state-of-the-art results across three widely used multilingual benchmarks: MedMentions for English, QUAERO for French, and SPACCC for Spanish. Evaluating data efficiency, we show that SynCABEL reaches the performance of full human supervision using up to 60% less annotated data, substantially reducing reliance on labor-intensive and costly expert labeling. Finally, acknowledging that standard evaluation based on exact code matching often underestimates clinically valid predictions due to ontology redundancy, we introduce an LLM-as-a-judge protocol. This analysis reveals that SynCABEL significantly improves the rate of clinically valid predictions. Our synthetic datasets, models, and code are released to support reproducibility and future research.
>
---
#### [new 045] Automated Safety Benchmarking: A Multi-agent Pipeline for LVLMs
- **分类: cs.CL**

- **简介: 该论文属于LVLM安全评估任务，旨在解决现有基准不足的问题。提出VLSafetyBencher系统，通过四个代理自动构建高质量安全基准。**

- **链接: [https://arxiv.org/pdf/2601.19507v1](https://arxiv.org/pdf/2601.19507v1)**

> **作者:** Xiangyang Zhu; Yuan Tian; Zicheng Zhang; Qi Jia; Chunyi Li; Renrui Zhang; Heng Li; Zongrui Wang; Wei Sun
>
> **摘要:** Large vision-language models (LVLMs) exhibit remarkable capabilities in cross-modal tasks but face significant safety challenges, which undermine their reliability in real-world applications. Efforts have been made to build LVLM safety evaluation benchmarks to uncover their vulnerability. However, existing benchmarks are hindered by their labor-intensive construction process, static complexity, and limited discriminative power. Thus, they may fail to keep pace with rapidly evolving models and emerging risks. To address these limitations, we propose VLSafetyBencher, the first automated system for LVLM safety benchmarking. VLSafetyBencher introduces four collaborative agents: Data Preprocessing, Generation, Augmentation, and Selection agents to construct and select high-quality samples. Experiments validates that VLSafetyBencher can construct high-quality safety benchmarks within one week at a minimal cost. The generated benchmark effectively distinguish safety, with a safety rate disparity of 70% between the most and least safe models.
>
---
#### [new 046] Binary Token-Level Classification with DeBERTa for All-Type MWE Identification: A Lightweight Approach with Linguistic Enhancement
- **分类: cs.CL**

- **简介: 该论文属于多词表达（MWE）识别任务，旨在解决MWE检测中的分类不平衡与结构化问题。通过二元token级分类、语言特征融合和数据增强，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.19360v1](https://arxiv.org/pdf/2601.19360v1)**

> **作者:** Diego Rossini; Lonneke van der Plas
>
> **备注:** Accepted at Findings of EACL 2026
>
> **摘要:** We present a comprehensive approach for multiword expression (MWE) identification that combines binary token-level classification, linguistic feature integration, and data augmentation. Our DeBERTa-v3-large model achieves 69.8% F1 on the CoAM dataset, surpassing the best results (Qwen-72B, 57.8% F1) on this dataset by 12 points while using 165x fewer parameters. We achieve this performance by (1) reformulating detection as binary token-level START/END/INSIDE classification rather than span-based prediction, (2) incorporating NP chunking and dependency features that help discontinuous and NOUN-type MWEs identification, and (3) applying oversampling that addresses severe class imbalance in the training data. We confirm the generalization of our method on the STREUSLE dataset, achieving 78.9% F1. These results demonstrate that carefully designed smaller models can substantially outperform LLMs on structured NLP tasks, with important implications for resource-constrained deployments.
>
---
#### [new 047] Group Distributionally Robust Optimization-Driven Reinforcement Learning for LLM Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决LLM推理中训练分布不均的问题。通过动态调整训练分布和资源分配，提升模型在复杂问题上的表现。**

- **链接: [https://arxiv.org/pdf/2601.19280v1](https://arxiv.org/pdf/2601.19280v1)**

> **作者:** Kishan Panaganti; Zhenwen Liang; Wenhao Yu; Haitao Mi; Dong Yu
>
> **备注:** Keywords: Large Language Models, Reasoning Models, Reinforcement Learning, Distributionally Robust Optimization, GRPO
>
> **摘要:** Recent progress in Large Language Model (LLM) reasoning is increasingly driven by the refinement of post-training loss functions and alignment strategies. However, standard Reinforcement Learning (RL) paradigms like Group Relative Policy Optimization (GRPO) remain constrained by static uniformity: uniform prompt sampling and a fixed number of rollouts per prompt. For heterogeneous, heavy-tailed reasoning data, this creates structural inefficiencies that waste compute on already-solved patterns while under-training the long tail of hard problems. To address this, we propose Multi-Adversary Group Distributionally Robust Optimization (GDRO), an optimization-first framework that moves beyond uniform reasoning models by dynamically adapting the training distribution. We introduce an Online Difficulty Classifier that partitions prompts into dynamic pass@k difficulty groups. We then propose two independent GDRO games for post-training: (1) Prompt-GDRO, which employs an EMA-debiased multiplicative-weights bandit sampler to target the intensive difficulty margin and upweight persistently hard groups without frequency bias; and (2) Rollout-GDRO, which uses a shadow-price controller to reallocate rollouts across groups, maximizing gradient variance reduction on hard tasks under a fixed mean budget (compute-neutral). We provide no-regret guarantees for both controllers and additionally a variance-proxy analysis motivating a square-root optimal rollout allocation for Rollout-GDRO. We validate our framework on the DAPO 14.1k dataset using Qwen3-Base models. Prompt-GDRO and Rollout-GDRO achieve average relative gains of +10.6% and +10.1%, respectively, in pass@8 accuracy across 1.7B, 4B, and 8B scales compared to the GRPO baseline. Qualitative analysis shows an emergent curriculum: the adversaries shift resources to the evolving reasoning frontier, enhancing the reasoning model's performance.
>
---
#### [new 048] ALRM: Agentic LLM for Robotic Manipulation
- **分类: cs.RO; cs.CL**

- **简介: 该论文提出ALRM框架，解决机器人操作中语言与执行的结合问题。通过两种模式实现模块化控制，提升多步骤推理能力。**

- **链接: [https://arxiv.org/pdf/2601.19510v1](https://arxiv.org/pdf/2601.19510v1)**

> **作者:** Vitor Gaboardi dos Santos; Ibrahim Khadraoui; Ibrahim Farhat; Hamza Yous; Samy Teffahi; Hakim Hacid
>
> **摘要:** Large Language Models (LLMs) have recently empowered agentic frameworks to exhibit advanced reasoning and planning capabilities. However, their integration in robotic control pipelines remains limited in two aspects: (1) prior \ac{llm}-based approaches often lack modular, agentic execution mechanisms, limiting their ability to plan, reflect on outcomes, and revise actions in a closed-loop manner; and (2) existing benchmarks for manipulation tasks focus on low-level control and do not systematically evaluate multistep reasoning and linguistic variation. In this paper, we propose Agentic LLM for Robot Manipulation (ALRM), an LLM-driven agentic framework for robotic manipulation. ALRM integrates policy generation with agentic execution through a ReAct-style reasoning loop, supporting two complementary modes: Code-asPolicy (CaP) for direct executable control code generation, and Tool-as-Policy (TaP) for iterative planning and tool-based action execution. To enable systematic evaluation, we also introduce a novel simulation benchmark comprising 56 tasks across multiple environments, capturing linguistically diverse instructions. Experiments with ten LLMs demonstrate that ALRM provides a scalable, interpretable, and modular approach for bridging natural language reasoning with reliable robotic execution. Results reveal Claude-4.1-Opus as the top closed-source model and Falcon-H1-7B as the top open-source model under CaP.
>
---
#### [new 049] Principled Fine-tuning of LLMs from User-Edits: A Medley of Preference, Supervision, and Reward
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **简介: 该论文研究如何通过用户编辑数据微调大语言模型，解决个性化与适应性问题。工作包括理论分析和集成学习方法，提升模型在不同用户分布下的表现。**

- **链接: [https://arxiv.org/pdf/2601.19055v1](https://arxiv.org/pdf/2601.19055v1)**

> **作者:** Dipendra Misra; Aldo Pacchiano; Ta-Chung Chi; Ge Gao
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** We study how to fine-tune LLMs using user-edit deployment data consisting of a set of context, an agent's response, and user edits. This deployment data is naturally generated by users in applications such as LLMs-based writing assistants and coding agents. The _natural_ origin of user edits makes it a desired source for adapting and personalizing LLMs. In this setup, there emerges a unification of various feedback types namely preferences, supervised labels, and cost that are typically studied separately in the literature. In this paper, we initiate the theoretical investigation of learning from user edits. We first derive bounds for learning algorithms that learn from each of these feedback types. We prove that these algorithms have different trade-offs depending upon the user, data distribution, and model class. We then propose a simple ensembling procedure to jointly learn from these feedback types. On two domains adapted from Gao et al. 2024, we show our ensembling procedure outperforms these methods that learn from individual feedback. Further, we show that our proposed procedure can robustly adapt to different user-edit distributions at test time.
>
---
#### [new 050] RvB: Automating AI System Hardening via Iterative Red-Blue Games
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文提出RvB框架，通过红蓝对抗机制自动增强AI系统安全性，解决动态防御难题。**

- **链接: [https://arxiv.org/pdf/2601.19726v1](https://arxiv.org/pdf/2601.19726v1)**

> **作者:** Lige Huang; Zicheng Liu; Jie Zhang; Lewen Yan; Dongrui Liu; Jing Shao
>
> **摘要:** The dual offensive and defensive utility of Large Language Models (LLMs) highlights a critical gap in AI security: the lack of unified frameworks for dynamic, iterative adversarial adaptation hardening. To bridge this gap, we propose the Red Team vs. Blue Team (RvB) framework, formulated as a training-free, sequential, imperfect-information game. In this process, the Red Team exposes vulnerabilities, driving the Blue Team to learning effective solutions without parameter updates. We validate our framework across two challenging domains: dynamic code hardening against CVEs and guardrail optimization against jailbreaks. Our empirical results show that this interaction compels the Blue Team to learn fundamental defensive principles, leading to robust remediations that are not merely overfitted to specific exploits. RvB achieves Defense Success Rates of 90\% and 45\% across the respective tasks while maintaining near 0\% False Positive Rates, significantly surpassing baselines. This work establishes the iterative adversarial interaction framework as a practical paradigm that automates the continuous hardening of AI systems.
>
---
#### [new 051] Benchmarks Saturate When The Model Gets Smarter Than The Judge
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 论文探讨了大语言模型评估中的基准测试问题，指出数据集和评估方法的缺陷影响评测效果。通过优化数据集并分析评委误差，揭示了高质量数据与可靠评委对准确评估模型性能的重要性。**

- **链接: [https://arxiv.org/pdf/2601.19532v1](https://arxiv.org/pdf/2601.19532v1)**

> **作者:** Marthe Ballon; Andres Algaba; Brecht Verbeken; Vincent Ginis
>
> **备注:** 17 pages, 10 figures, 3 tables
>
> **摘要:** Benchmarks are important tools to track progress in the development of Large Language Models (LLMs), yet inaccuracies in datasets and evaluation methods consistently undermine their effectiveness. Here, we present Omni-MATH-2, a manually revised version of the Omni-MATH dataset comprising a clean, exact-answer subset ($n{=}4181$) and a tagged, non-standard subset ($n{=}247$). Each problem was audited to ensure LaTeX compilability, solvability and verifiability, which involved adding missing figures or information, labeling problems requiring a proof, estimation or image, and removing clutter. This process significantly reduces dataset-induced noise, thereby providing a more precise assessment of model performance. The annotated dataset also allows us to evaluate judge-induced noise by comparing GPT-5 mini with the original Omni-Judge, revealing substantial discrepancies between judges on both the clean and tagged problem subsets. Expert annotations reveal that Omni-Judge is wrong in $96.4\%$ of the judge disagreements, indicating its inability to differentiate between models' abilities, even well before saturation of the benchmark occurs. As problems become more challenging, we find that increasingly competent judges become essential in order to prevent judge errors from masking genuine differences between models. Finally, neither judge identifies the present failure modes for the subset of tagged problems, demonstrating that dataset quality and judge reliability are both critical to develop accurate benchmarks of model performance.
>
---
#### [new 052] Who's in Charge? Disempowerment Patterns in Real-World LLM Usage
- **分类: cs.CY; cs.AI; cs.CL; cs.HC**

- **简介: 该论文属于AI伦理研究，探讨AI助手使用中的去权现象，分析150万次对话，揭示其对用户认知和价值观的潜在负面影响。**

- **链接: [https://arxiv.org/pdf/2601.19062v1](https://arxiv.org/pdf/2601.19062v1)**

> **作者:** Mrinank Sharma; Miles McCain; Raymond Douglas; David Duvenaud
>
> **摘要:** Although AI assistants are now deeply embedded in society, there has been limited empirical study of how their usage affects human empowerment. We present the first large-scale empirical analysis of disempowerment patterns in real-world AI assistant interactions, analyzing 1.5 million consumer Claude.ai conversations using a privacy-preserving approach. We focus on situational disempowerment potential, which occurs when AI assistant interactions risk leading users to form distorted perceptions of reality, make inauthentic value judgments, or act in ways misaligned with their values. Quantitatively, we find that severe forms of disempowerment potential occur in fewer than one in a thousand conversations, though rates are substantially higher in personal domains like relationships and lifestyle. Qualitatively, we uncover several concerning patterns, such as validation of persecution narratives and grandiose identities with emphatic sycophantic language, definitive moral judgments about third parties, and complete scripting of value-laden personal communications that users appear to implement verbatim. Analysis of historical trends reveals an increase in the prevalence of disempowerment potential over time. We also find that interactions with greater disempowerment potential receive higher user approval ratings, possibly suggesting a tension between short-term user preferences and long-term human empowerment. Our findings highlight the need for AI systems designed to robustly support human autonomy and flourishing.
>
---
#### [new 053] More at Stake: How Payoff and Language Shape LLM Agent Strategies in Cooperation Dilemmas
- **分类: cs.AI; cs.CL; cs.GT; cs.LG; cs.MA**

- **简介: 该论文研究LLM在合作困境中的策略行为，分析收益和语言如何影响其决策。属于AI策略与合作研究任务，旨在理解LLM的协作倾向及影响因素。**

- **链接: [https://arxiv.org/pdf/2601.19082v1](https://arxiv.org/pdf/2601.19082v1)**

> **作者:** Trung-Kiet Huynh; Dao-Sy Duy-Minh; Thanh-Bang Cao; Phong-Hao Le; Hong-Dan Nguyen; Nguyen Lam Phu Quy; Minh-Luan Nguyen-Vo; Hong-Phat Pham; Pham Phu Hoa; Thien-Kim Than; Chi-Nguyen Tran; Huy Tran; Gia-Thoai Tran-Le; Alessio Buscemi; Le Hong Trang; The Anh Han
>
> **备注:** 14 pages, 10 figures, 4 tables
>
> **摘要:** As LLMs increasingly act as autonomous agents in interactive and multi-agent settings, understanding their strategic behavior is critical for safety, coordination, and AI-driven social and economic systems. We investigate how payoff magnitude and linguistic context shape LLM strategies in repeated social dilemmas, using a payoff-scaled Prisoner's Dilemma to isolate sensitivity to incentive strength. Across models and languages, we observe consistent behavioral patterns, including incentive-sensitive conditional strategies and cross-linguistic divergence. To interpret these dynamics, we train supervised classifiers on canonical repeated-game strategies and apply them to LLM decisions, revealing systematic, model- and language-dependent behavioral intentions, with linguistic framing sometimes matching or exceeding architectural effects. Our results provide a unified framework for auditing LLMs as strategic agents and highlight cooperation biases with direct implications for AI governance and multi-agent system design.
>
---
#### [new 054] Neural Neural Scaling Laws
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究语言模型性能预测任务，解决如何准确预测下游任务表现的问题。提出NeuNeu模型，通过时间序列外推提高预测精度。**

- **链接: [https://arxiv.org/pdf/2601.19831v1](https://arxiv.org/pdf/2601.19831v1)**

> **作者:** Michael Y. Hu; Jane Pan; Ayush Rajesh Jhaveri; Nicholas Lourie; Kyunghyun Cho
>
> **摘要:** Neural scaling laws predict how language model performance improves with increased compute. While aggregate metrics like validation loss can follow smooth power-law curves, individual downstream tasks exhibit diverse scaling behaviors: some improve monotonically, others plateau, and some even degrade with scale. We argue that predicting downstream performance from validation perplexity suffers from two limitations: averaging token-level losses obscures signal, and no simple parametric family can capture the full spectrum of scaling behaviors. To address this, we propose Neural Neural Scaling Laws (NeuNeu), a neural network that frames scaling law prediction as time-series extrapolation. NeuNeu combines temporal context from observed accuracy trajectories with token-level validation losses, learning to predict future performance without assuming any bottleneck or functional form. Trained entirely on open-source model checkpoints from HuggingFace, NeuNeu achieves 2.04% mean absolute error in predicting model accuracy on 66 downstream tasks -- a 38% reduction compared to logistic scaling laws (3.29% MAE). Furthermore, NeuNeu generalizes zero-shot to unseen model families, parameter counts, and downstream tasks. Our work suggests that predicting downstream scaling laws directly from data outperforms parametric alternatives.
>
---
#### [new 055] Explicit Multi-head Attention for Inter-head Interaction in Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出MEA，解决大语言模型中多头注意力的交互问题，通过HLC和组归一化提升性能与效率。**

- **链接: [https://arxiv.org/pdf/2601.19611v1](https://arxiv.org/pdf/2601.19611v1)**

> **作者:** Runyu Peng; Yunhua Zhou; Demin Song; Kai Lv; Bo Wang; Qipeng Guo; Xipeng Qiu
>
> **摘要:** In large language models built upon the Transformer architecture, recent studies have shown that inter-head interaction can enhance attention performance. Motivated by this, we propose Multi-head Explicit Attention (MEA), a simple yet effective attention variant that explicitly models cross-head interaction. MEA consists of two key components: a Head-level Linear Composition (HLC) module that separately applies learnable linear combinations to the key and value vectors across heads, thereby enabling rich inter-head communication; and a head-level Group Normalization layer that aligns the statistical properties of the recombined heads. MEA shows strong robustness in pretraining, which allows the use of larger learning rates that lead to faster convergence, ultimately resulting in lower validation loss and improved performance across a range of tasks. Furthermore, we explore the parameter efficiency of MEA by reducing the number of attention heads and leveraging HLC to reconstruct them using low-rank "virtual heads". This enables a practical key-value cache compression strategy that reduces KV-cache memory usage by 50% with negligible performance loss on knowledge-intensive and scientific reasoning tasks, and only a 3.59% accuracy drop for Olympiad-level mathematical benchmarks.
>
---
#### [new 056] Rethinking Discrete Speech Representation Tokens for Accent Generation
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于语音生成任务，旨在解决DSRT中Accent信息编码问题。通过提出评估框架，分析不同编码器的DSRT，并设计新型DSRT以提升可控Accent生成效果。**

- **链接: [https://arxiv.org/pdf/2601.19786v1](https://arxiv.org/pdf/2601.19786v1)**

> **作者:** Jinzuomu Zhong; Yi Wang; Korin Richmond; Peter Bell
>
> **摘要:** Discrete Speech Representation Tokens (DSRTs) have become a foundational component in speech generation. While prior work has extensively studied phonetic and speaker information in DSRTs, how accent information is encoded in DSRTs remains largely unexplored. In this paper, we present the first systematic investigation of accent information in DSRTs. We propose a unified evaluation framework that measures both accessibility of accent information via a novel Accent ABX task and recoverability via cross-accent Voice Conversion (VC) resynthesis. Using this framework, we analyse DSRTs derived from a variety of speech encoders. Our results reveal that accent information is substantially reduced when ASR supervision is used to fine-tune the encoder, but cannot be effectively disentangled from phonetic and speaker information through naive codebook size reduction. Based on these findings, we propose new content-only and content-accent DSRTs that significantly outperform existing designs in controllable accent generation. Our work highlights the importance of accent-aware evaluation and provides practical guidance for designing DSRTs for accent-controlled speech generation.
>
---
#### [new 057] Ad Insertion in LLM-Generated Responses
- **分类: cs.GT; cs.AI; cs.CL**

- **简介: 该论文属于LLM广告插入任务，解决传统广告无法匹配用户实时意图的问题。通过解耦广告插入与生成、按语义类别竞价，提升上下文一致性与效率。**

- **链接: [https://arxiv.org/pdf/2601.19435v1](https://arxiv.org/pdf/2601.19435v1)**

> **作者:** Shengwei Xu; Zhaohua Chen; Xiaotie Deng; Zhiyi Huang; Grant Schoenebeck
>
> **备注:** 31 pages, 8 figures
>
> **摘要:** Sustainable monetization of Large Language Models (LLMs) remains a critical open challenge. Traditional search advertising, which relies on static keywords, fails to capture the fleeting, context-dependent user intents--the specific information, goods, or services a user seeks--embedded in conversational flows. Beyond the standard goal of social welfare maximization, effective LLM advertising imposes additional requirements on contextual coherence (ensuring ads align semantically with transient user intents) and computational efficiency (avoiding user interaction latency), as well as adherence to ethical and regulatory standards, including preserving privacy and ensuring explicit ad disclosure. Although various recent solutions have explored bidding on token-level and query-level, both categories of approaches generally fail to holistically satisfy this multifaceted set of constraints. We propose a practical framework that resolves these tensions through two decoupling strategies. First, we decouple ad insertion from response generation to ensure safety and explicit disclosure. Second, we decouple bidding from specific user queries by using ``genres'' (high-level semantic clusters) as a proxy. This allows advertisers to bid on stable categories rather than sensitive real-time response, reducing computational burden and privacy risks. We demonstrate that applying the VCG auction mechanism to this genre-based framework yields approximately dominant strategy incentive compatibility (DSIC) and individual rationality (IR), as well as approximately optimal social welfare, while maintaining high computational efficiency. Finally, we introduce an "LLM-as-a-Judge" metric to estimate contextual coherence. Our experiments show that this metric correlates strongly with human ratings (Spearman's $ρ\approx 0.66$), outperforming 80% of individual human evaluators.
>
---
#### [new 058] Enhancing Academic Paper Recommendations Using Fine-Grained Knowledge Entities and Multifaceted Document Embeddings
- **分类: cs.IR; cs.CL; cs.DL**

- **简介: 该论文属于学术论文推荐任务，旨在解决现有系统无法满足学者细粒度需求的问题。通过融合知识实体和文档嵌入，提升推荐准确性。**

- **链接: [https://arxiv.org/pdf/2601.19513v1](https://arxiv.org/pdf/2601.19513v1)**

> **作者:** Haixu Xi; Heng Zhang; Chengzhi Zhang
>
> **摘要:** In the era of explosive growth in academic literature, the burden of literature review on scholars are increasing. Proactively recommending academic papers that align with scholars' literature needs in the research process has become one of the crucial pathways to enhance research efficiency and stimulate innovative thinking. Current academic paper recommendation systems primarily focus on broad and coarse-grained suggestions based on general topic or field similarities. While these systems effectively identify related literature, they fall short in addressing scholars' more specific and fine-grained needs, such as locating papers that utilize particular research methods, or tackle distinct research tasks within the same topic. To meet the diverse and specific literature needs of scholars in the research process, this paper proposes a novel academic paper recommendation method. This approach embeds multidimensional information by integrating new types of fine-grained knowledge entities, title and abstract of document, and citation data. Recommendations are then generated by calculating the similarity between combined paper vectors. The proposed recommendation method was evaluated using the STM-KG dataset, a knowledge graph that incorporates scientific concepts derived from papers across ten distinct domains. The experimental results indicate that our method outperforms baseline models, achieving an average precision of 27.3% among the top 50 recommendations. This represents an improvement of 6.7% over existing approaches.
>
---
#### [new 059] Save the Good Prefix: Precise Error Penalization via Process-Supervised RL to Enhance LLM Reasoning
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于增强大语言模型推理的任务，解决RL中奖励稀疏和信用分配问题。提出VPPO方法，利用PRM定位错误前缀，提升学习信号稳定性与效果。**

- **链接: [https://arxiv.org/pdf/2601.18984v1](https://arxiv.org/pdf/2601.18984v1)**

> **作者:** Haolin Liu; Dian Yu; Sidi Lu; Yujun Zhou; Rui Liu; Zhenwen Liang; Haitao Mi; Chen-Yu Wei; Dong Yu
>
> **摘要:** Reinforcement learning (RL) has emerged as a powerful framework for improving the reasoning capabilities of large language models (LLMs). However, most existing RL approaches rely on sparse outcome rewards, which fail to credit correct intermediate steps in partially successful solutions. Process reward models (PRMs) offer fine-grained step-level supervision, but their scores are often noisy and difficult to evaluate. As a result, recent PRM benchmarks focus on a more objective capability: detecting the first incorrect step in a reasoning path. However, this evaluation target is misaligned with how PRMs are typically used in RL, where their step-wise scores are treated as raw rewards to maximize. To bridge this gap, we propose Verifiable Prefix Policy Optimization (VPPO), which uses PRMs only to localize the first error during RL. Given an incorrect rollout, VPPO partitions the trajectory into a verified correct prefix and an erroneous suffix based on the first error, rewarding the former while applying targeted penalties only after the detected mistake. This design yields stable, interpretable learning signals and improves credit assignment. Across multiple reasoning benchmarks, VPPO consistently outperforms sparse-reward RL and prior PRM-guided baselines on both Pass@1 and Pass@K.
>
---
#### [new 060] Intent2QoS: Language Model-Driven Automation of Traffic Shaping Configurations
- **分类: cs.NI; cs.CL**

- **简介: 该论文属于网络流量管理任务，解决手动配置QoS耗时且复杂的问题。通过语言模型自动将高阶意图转化为有效配置规则，提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2601.18974v1](https://arxiv.org/pdf/2601.18974v1)**

> **作者:** Sudipta Acharya; Burak Kantarci
>
> **备注:** 6 page, 4 figures, Accepted to IEEE International Conference on Communications (ICC) 2026
>
> **摘要:** Traffic shaping and Quality of Service (QoS) enforcement are critical for managing bandwidth, latency, and fairness in networks. These tasks often rely on low-level traffic control settings, which require manual setup and technical expertise. This paper presents an automated framework that converts high-level traffic shaping intents in natural or declarative language into valid and correct traffic control rules. To the best of our knowledge, we present the first end-to-end pipeline that ties intent translation in a queuing-theoretic semantic model and, with a rule-based critic, yields deployable Linux traffic control configuration sets. The framework has three steps: (1) a queuing simulation with priority scheduling and Active Queue Management (AQM) builds a semantic model; (2) a language model, using this semantic model and a traffic profile, generates sub-intents and configuration rules; and (3) a rule-based critic checks and adjusts the rules for correctness and policy compliance. We evaluate multiple language models by generating traffic control commands from business intents that comply with relevant standards for traffic control protocols. Experimental results on 100 intents show significant gains, with LLaMA3 reaching 0.88 semantic similarity and 0.87 semantic coverage, outperforming other models by over 30\. A thorough sensitivity study demonstrates that AQM-guided prompting reduces variability threefold compared to zero-shot baselines.
>
---
#### [new 061] Post-LayerNorm Is Back: Stable, ExpressivE, and Deep
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，解决深度Transformer训练不稳定问题。通过引入Highway连接改进Post-LN结构，实现超深模型稳定训练。**

- **链接: [https://arxiv.org/pdf/2601.19895v1](https://arxiv.org/pdf/2601.19895v1)**

> **作者:** Chen Chen; Lai Wei
>
> **摘要:** Large language model (LLM) scaling is hitting a wall. Widening models yields diminishing returns, and extending context length does not improve fundamental expressivity. In contrast, depth scaling offers theoretically superior expressivity, yet current Transformer architectures struggle to train reliably at extreme depths. We revisit the Post-LayerNorm (Post-LN) formulation, whose instability at scale caused its replacement by Pre-LN in modern LLMs. We show that the central failure mode of Post-LN arises from the ResNet-style residual pathway, which introduces gradient vanishing in deep networks. We present Keel, a Post-LN Transformer that replaces this residual path with a Highway-style connection. This modification preserves the gradient flow through the residual branch, preventing signal vanishing from the top layers to the bottom. Unlike prior methods, Keel enables stable training at extreme depths without requiring specialized initialization or complex optimization tricks. Keel trains robustly at depths exceeding 1000 layers and consistently improves perplexity and depth-scaling characteristics over Pre-LN. These findings indicate that Post-LN, when paired with a Highway-style connection, provides a simple and effective foundation for building deeply scalable LLMs, opening the possibility for future infinite-depth architectures.
>
---
#### [new 062] Is Finer Better? The Limits of Microscaling Formats in Large Language Models
- **分类: cs.LG; cs.AR; cs.CL**

- **简介: 该论文研究微缩量化格式在大语言模型中的限制，解决量化导致性能下降的问题。通过实验与理论分析，提出FP8 UE5M3格式提升效率。**

- **链接: [https://arxiv.org/pdf/2601.19026v1](https://arxiv.org/pdf/2601.19026v1)**

> **作者:** Andrea Fasoli; Monodeep Kar; Chi-Chun Liu; Swagath Venkataramani; Viji Srinivasan; Leland Chang; Naigang Wang
>
> **备注:** 31 pages, 17 figures, 3 tables; accepted to ICLR 2026
>
> **摘要:** Microscaling data formats leverage per-block tensor quantization to enable aggressive model compression with limited loss in accuracy. Unlocking their potential for efficient training and inference necessitates hardware-friendly implementations that handle matrix multiplications in a native format and adopt efficient error-mitigation strategies. Herein, we report the emergence of a surprising behavior associated with microscaling quantization, whereas the output of a quantized model degrades as block size is decreased below a given threshold. This behavior clashes with the expectation that a smaller block size should allow for a better representation of the tensor elements. We investigate this phenomenon both experimentally and theoretically, decoupling the sources of quantization error behind it. Experimentally, we analyze the distributions of several Large Language Models and identify the conditions driving the anomalous behavior. Theoretically, we lay down a framework showing remarkable agreement with experimental data from pretrained model distributions and ideal ones. Overall, we show that the anomaly is driven by the interplay between narrow tensor distributions and the limited dynamic range of the quantized scales. Based on these insights, we propose the use of FP8 unsigned E5M3 (UE5M3) as a novel hardware-friendly format for the scales in FP4 microscaling data types. We demonstrate that UE5M3 achieves comparable performance to the conventional FP8 unsigned E4M3 scales while obviating the need of global scaling operations on weights and activations.
>
---
#### [new 063] XProvence: Zero-Cost Multilingual Context Pruning for Retrieval-Augmented Generation
- **分类: cs.IR; cs.CL**

- **简介: 该论文提出XProvence，解决多语言RAG系统中上下文剪枝问题，通过零成本方法提升效率，支持100+语言。**

- **链接: [https://arxiv.org/pdf/2601.18886v1](https://arxiv.org/pdf/2601.18886v1)**

> **作者:** Youssef Mohamed; Mohamed Elhoseiny; Thibault Formal; Nadezhda Chirkova
>
> **备注:** Accepted to ECIR 2026
>
> **摘要:** This paper introduces XProvence, a multilingual zero-cost context pruning model for retrieval-augmented generation (RAG), trained on 16 languages and supporting 100+ languages through effective cross-lingual transfer. Motivated by the growing use of RAG systems across diverse languages, we explore several strategies to generalize the Provence framework-which first integrated efficient zero-cost context pruning directly into the re-ranking model-beyond English. Across four multilingual question answering benchmarks, we show how XProvence can prune RAG contexts with minimal-to-no performance degradation and outperforms strong baselines. Our model is available at https://huggingface.co/naver/xprovence-reranker-bgem3-v2.
>
---
#### [new 064] SICL-AT: Another way to adapt Auditory LLM to low-resource task
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文属于语音与音频理解任务，旨在解决低资源场景下模型性能下降的问题。通过提出SICL-AT方法，增强模型的上下文学习能力，提升在低资源任务上的表现。**

- **链接: [https://arxiv.org/pdf/2601.18904v1](https://arxiv.org/pdf/2601.18904v1)**

> **作者:** Haolong Zheng; Siyin Wang; Zengrui Jin; Mark Hasegawa-Johnson
>
> **摘要:** Auditory Large Language Models (LLMs) have demonstrated strong performance across a wide range of speech and audio understanding tasks. Nevertheless, they often struggle when applied to low-resource or unfamiliar tasks. In case of labeled in-domain data is scarce or mismatched to the true test distribution, direct fine-tuning can be brittle. In-Context Learning (ICL) provides a training-free, inference-time solution by adapting auditory LLMs through conditioning on a few in-domain demonstrations. In this work, we first show that \emph{Vanilla ICL}, improves zero-shot performance across diverse speech and audio tasks for selected models which suggest this ICL adaptation capability can be generalized to multimodal setting. Building on this, we propose \textbf{Speech In-Context Learning Adaptation Training (SICL-AT)}, a post-training recipe utilizes only high resource speech data intending to strengthen model's in-context learning capability. The enhancement can generalize to audio understanding/reasoning task. Experiments indicate our proposed method consistently outperforms direct fine-tuning in low-resource scenario.
>
---
## 更新

#### [replaced 001] Complex Logical Instruction Generation
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于指令跟随任务，旨在解决复杂逻辑指令理解不足的问题。提出LogicIFGen生成框架和LogicIFEval基准，验证LLMs在复杂逻辑指令上的表现。**

- **链接: [https://arxiv.org/pdf/2508.09125v2](https://arxiv.org/pdf/2508.09125v2)**

> **作者:** Mian Zhang; Shujian Liu; Sixun Dong; Ming Yin; Yebowen Hu; Xun Wang; Steven Ma; Song Wang; Sathish Reddy Indurthi; Haoyun Deng; Zhiyu Zoey Chen; Kaiqiang Song
>
> **摘要:** Instruction following has catalyzed the recent era of Large Language Models (LLMs) and is the foundational skill underpinning more advanced capabilities such as reasoning and agentic behaviors. As tasks grow more challenging, the logic structures embedded in natural language instructions becomes increasingly intricate. However, how well LLMs perform on such logic-rich instructions remains under-explored. We propose LogicIFGen and LogicIFEval. LogicIFGen is a scalable, automated framework for generating verifiable instructions from code functions, which can naturally express rich logic such as conditions, loops, and function calls. We further curate a collection of complex code functions and use LogicIFGen to construct LogicIFEval, a benchmark comprising 426 verifiable logic-rich instructions. Our experiments demonstrate that current state-of-the-art LLMs still struggle to correctly follow the instructions in LogicIFEval. Most LLMs can only follow fewer than 60% of the instructions, revealing significant deficiencies in the instruction-following ability. Code and Benchmark: https://github.com/mianzhang/LogicIF
>
---
#### [replaced 002] Improving Value-based Process Verifier via Low-Cost Variance Reduction
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于提升大语言模型推理能力的任务，针对过程验证器因估计误差效果不佳的问题，提出ComMCS方法降低方差，提升验证准确性。**

- **链接: [https://arxiv.org/pdf/2508.10539v2](https://arxiv.org/pdf/2508.10539v2)**

> **作者:** Zetian Sun; Dongfang Li; Baotian Hu; Min Zhang
>
> **备注:** Accepted by AAAI-2026
>
> **摘要:** Large language models (LLMs) have achieved remarkable success in a wide range of tasks. However, their reasoning capabilities, particularly in complex domains like mathematics, remain a significant challenge. Value-based process verifiers, which estimate the probability of a partial reasoning chain leading to a correct solution, are a promising approach for improving reasoning. Nevertheless, their effectiveness is often hindered by estimation error in their training annotations, a consequence of the limited number of Monte Carlo (MC) samples feasible due to the high cost of LLM inference. In this paper, we identify that the estimation error primarily arises from high variance rather than bias, and the MC estimator is a Minimum Variance Unbiased Estimator (MVUE). To address the problem, we propose the \textsc{Com}pound \textsc{M}onte \textsc{C}arlo \textsc{S}ampling (ComMCS) method, which constructs an unbiased estimator by linearly combining the MC estimators from the current and subsequent steps. Theoretically, we show that our method leads to a predictable reduction in variance, while maintaining an unbiased estimation without additional LLM inference cost. We also perform empirical experiments on the MATH-500 and GSM8K benchmarks to demonstrate the effectiveness of our method. Notably, ComMCS outperforms regression-based optimization method by 2.8 points, the non-variance-reduced baseline by 2.2 points on MATH-500 on Best-of-32 sampling experiment.
>
---
#### [replaced 003] Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning for Web Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Web-CogReasoner，解决Web代理的认知推理问题，通过知识学习与推理框架提升其任务泛化能力。**

- **链接: [https://arxiv.org/pdf/2508.01858v2](https://arxiv.org/pdf/2508.01858v2)**

> **作者:** Yuhan Guo; Cong Guo; Aiwen Sun; Hongliang He; Xinyu Yang; Yue Lu; Yingji Zhang; Xuntao Guo; Dong Zhang; Jianzhuang Liu; Jiang Duan; Yijia Xiao; Liangjian Wen; Hai-Ming Xu; Yong Dai
>
> **备注:** Accepted to ICLR 2026. Our code and data is released at https://github.com/Gnonymous/Web-CogReasoner
>
> **摘要:** Multimodal large-scale models have significantly advanced the development of web agents, enabling perception and interaction with digital environments akin to human cognition. In this paper, we argue that web agents must first acquire sufficient knowledge to effectively engage in cognitive reasoning. Therefore, we decompose a web agent's capabilities into two essential stages: knowledge content learning and cognitive processes. To formalize this, we propose Web-CogKnowledge Framework, categorizing knowledge as Factual, Conceptual, and Procedural. In this framework, knowledge content learning corresponds to the agent's processes of Memorizing and Understanding, which rely on the first two knowledge types, representing the "what" of learning. Conversely, cognitive processes correspond to Exploring, grounded in Procedural knowledge, defining the "how" of reasoning and action. To facilitate knowledge acquisition, we construct the Web-CogDataset, a structured resource curated from 14 real-world websites, designed to systematically instill core knowledge necessary for web agent. This dataset serves as the agent's conceptual grounding-the "nouns" upon which comprehension is built-as well as the basis for learning how to reason and act. Building on this foundation, we operationalize these processes through a novel knowledge-driven Chain-of-Thought (CoT) reasoning framework, developing and training our proposed agent, the Web-CogReasoner. Extensive experimentation reveals its significant superiority over existing models, especially in generalizing to unseen tasks where structured knowledge is decisive. To enable rigorous evaluation, we introduce the Web-CogBench, a comprehensive evaluation suite designed to assess and compare agent performance across the delineated knowledge domains and cognitive capabilities. Our code and data is open sourced at https://github.com/Gnonymous/Web-CogReasoner
>
---
#### [replaced 004] A-IPO: Adaptive Intent-driven Preference Optimization
- **分类: cs.CL**

- **简介: 该论文属于模型对齐任务，旨在解决现有方法忽视用户隐含意图的问题。提出A-IPO框架，通过引入意图模块提升响应与用户意图的一致性，增强对齐效果和对抗鲁棒性。**

- **链接: [https://arxiv.org/pdf/2510.10077v2](https://arxiv.org/pdf/2510.10077v2)**

> **作者:** Wenqing Wang; Muhammad Asif Ali; Ali Shoker; Ruohan Yang; Junyang Chen; Ying Sha; Huan Wang
>
> **摘要:** Human preferences are diverse and dynamic, shaped by regional, cultural, and social factors. Existing alignment methods like Direct Preference Optimization (DPO) and its variants often default to majority views, overlooking minority opinions and failing to capture latent user intentions in prompts. To address these limitations, we introduce \underline{\textbf{A}}daptive \textbf{\underline{I}}ntent-driven \textbf{\underline{P}}reference \textbf{\underline{O}}ptimization (\textbf{A-IPO}). Specifically,A-IPO introduces an intention module that infers the latent intent behind each user prompt and explicitly incorporates this inferred intent into the reward function, encouraging stronger alignment between the preferred model's responses and the user's underlying intentions. We demonstrate, both theoretically and empirically, that incorporating an intention--response similarity term increases the preference margin (by a positive shift of $λ\,Δ\mathrm{sim}$ in the log-odds), resulting in clearer separation between preferred and dispreferred responses compared to DPO. For evaluation, we introduce two new benchmarks, Real-pref, Attack-pref along with an extended version of an existing dataset, GlobalOpinionQA-Ext, to assess real-world and adversarial preference alignment. Through explicit modeling of diverse user intents,A-IPO facilitates pluralistic preference optimization while simultaneously enhancing adversarial robustness in preference alignment. Comprehensive empirical evaluation demonstrates that A-IPO consistently surpasses existing baselines, yielding substantial improvements across key metrics: up to +24.8 win-rate and +45.6 Response-Intention Consistency on Real-pref; up to +38.6 Response Similarity and +52.2 Defense Success Rate on Attack-pref; and up to +54.6 Intention Consistency Score on GlobalOpinionQA-Ext.
>
---
#### [replaced 005] Entropy-Gated Branching for Efficient Test-Time Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Entropy-Gated Branching方法，用于提升大语言模型的推理效率与准确性，解决测试时计算资源浪费问题。**

- **链接: [https://arxiv.org/pdf/2503.21961v4](https://arxiv.org/pdf/2503.21961v4)**

> **作者:** Xianzhi Li; Ethan Callanan; Abdellah Ghassel; Xiaodan Zhu
>
> **摘要:** Test-time compute methods can significantly improve the reasoning capabilities and problem-solving accuracy of large language models (LLMs). However, these approaches require substantially more computational resources, with most compute wasted on exploring low-diversity branches where the model already exhibits high confidence. We observe that a small subset of uncertain reasoning steps has a disproportionately large impact on final prediction accuracy, and branching at these critical junctures tends to yield more diverse and higher-quality candidate reasoning steps. We propose Entropy-Gated Branching (EGB), which branches only at high-uncertainty steps and prunes expansions with a lightweight verifier. On mathematical and financial reasoning benchmarks, EGB improves accuracy by 22.6% over standard inference while operating 31%-75% faster across math benchmarks than test-time beam search with higher performance. Our results show that dynamic resource allocation during inference can substantially improve both efficiency and effectiveness, offering a more scalable pathway to enhanced LLM reasoning capabilities.
>
---
#### [replaced 006] Efficient and Transferable Agentic Knowledge Graph RAG via Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识图谱增强生成任务，旨在解决KG-RAG系统效率低、依赖特定知识图谱的问题。通过强化学习构建单一代理，实现高效且可迁移的问答系统。**

- **链接: [https://arxiv.org/pdf/2509.26383v4](https://arxiv.org/pdf/2509.26383v4)**

> **作者:** Jinyeop Song; Song Wang; Julian Shun; Yada Zhu
>
> **备注:** Wrong numbers are reported for main results
>
> **摘要:** Knowledge-graph retrieval-augmented generation (KG-RAG) couples large language models (LLMs) with structured, verifiable knowledge graphs (KGs) to reduce hallucinations and expose reasoning traces. However, many KG-RAG systems compose multiple LLM modules (e.g planning, reasoning, and responding), inflating inference cost and binding behavior to a specific target KG. To address this, we introduce KG-R1, an agentic KG retrieval-augmented generation (KG-RAG) framework through reinforcement learning (RL). KG-R1 utilizes a single agent that interacts with KGs as its environment, learning to retrieve at each step and incorporating the retrieved information into its reasoning and generation. The process is optimized through end-to-end RL. In controlled experiments across Knowledge-Graph Question Answering (KGQA) benchmarks, our method demonstrates both efficiency and transferability: Using Qwen-2.5-3B, KG-R1 improves answer accuracy with fewer generation tokens than prior multi-module workflow methods that use larger foundation or fine-tuned models. Furthermore, KG-R1 enables plug and play: after training, it maintains strong accuracy on new KGs without modification. These properties make KG-R1 a promising KG-RAG framework for real-world deployment. Our code is publicly available at https://github.com/Jinyeop3110/KG-R1.
>
---
#### [replaced 007] TableMaster: A Recipe to Advance Table Understanding with Language Models
- **分类: cs.CL**

- **简介: 该论文属于表格理解任务，旨在解决语言模型在处理表格数据时的四大挑战。作者提出TableMaster框架，通过语义增强和自适应推理提升表格理解能力。**

- **链接: [https://arxiv.org/pdf/2501.19378v4](https://arxiv.org/pdf/2501.19378v4)**

> **作者:** Lang Cao; Hanbing Liu
>
> **摘要:** Tables serve as a fundamental format for representing structured relational data. While current language models (LMs) excel at many text-based tasks, they still face challenges in table understanding due to the complex characteristics of tabular data, such as their structured nature. In this paper, we aim to enhance LMs for improved table understanding. We identify four key challenges: 1) difficulty in locating target data, 2) deficiency in table semantics, 3) numerical inaccuracies in textual reasoning, and 4) semantic inflexibility in symbolic reasoning. To address these issues, we propose TableMaster, a recipe and comprehensive framework that integrates multiple solutions to overcome these obstacles. TableMaster first extracts relevant table content and verbalizes it with enriched semantic context. Additionally, we introduce adaptive reasoning, a flexible approach that dynamically adjusts between textual and symbolic reasoning, tailoring the reasoning process to each query. Extensive analyses and experiments demonstrate our findings and the effectiveness of TableMaster. On the WikiTQ dataset, TableMaster achieves an accuracy of 78.13% using GPT-4o-mini, surpassing existing baselines. We hope this work will serve as a practical step toward more robust and reliable table understanding.
>
---
#### [replaced 008] "Not in My Backyard": LLMs Uncover Online and Offline Social Biases Against Homelessnes
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文属于社会偏见分析任务，旨在检测在线和离线语料中对无家可归者的负面偏见。通过构建多领域数据集并利用大模型生成伪标签，提升分类效果，揭示“不在我家后院”叙事的广泛影响。**

- **链接: [https://arxiv.org/pdf/2508.13187v2](https://arxiv.org/pdf/2508.13187v2)**

> **作者:** Jonathan A. Karr; Benjamin F. Herbst; Matthew L. Sisk; Xueyun Li; Ting Hua; Matthew Hauenstein; Georgina Curto; Nitesh V. Chawla
>
> **摘要:** Homelessness is a persistent social challenge, impacting millions worldwide. Over 876,000 people experienced homelessness (PEH) in the U.S. in 2025. Social bias is a significant barrier to alleviation, shaping public perception and influencing policymaking. Given that online textual media and offline city council discourse reflect and influence part of public opinion, it provides valuable insights to identify and track social biases against PEH. We present a new, manually-annotated multi-domain dataset compiled from Reddit, X (formerly Twitter), news articles, and city council meeting minutes across ten U.S. cities. Our 16-category multi-label taxonomy creates a challenging long-tail classification problem: some categories appear in less than 1% of samples, while others exceed 70%. We find that small human-annotated datasets (1,702 samples) are insufficient for training effective classifiers, whether used to fine-tune encoder models or as few-shot examples for LLMs. To address this, we use GPT-4.1 to generate pseudo-labels on a larger unlabeled corpus. Training on this expanded dataset enables even small encoder models (ModernBERT, 150M parameters) to achieve 35.23 macro-F1, approaching GPT-4.1's 41.57. This demonstrates that \textbf{data quantity matters more than model size}, enabling low-cost, privacy-preserving deployment without relying on commercial APIs. Our results reveal that negative bias against PEH is prevalent both offline and online (especially on Reddit), with "not in my backyard" narratives showing the highest engagement. These findings uncover a type of ostracism that directly impacts poverty-reduction policymaking and provide actionable insights for practitioners addressing homelessness.
>
---
#### [replaced 009] Memento: Towards Proactive Visualization of Everyday Memories with Personal Wearable AR Assistant
- **分类: cs.HC; cs.CL; cs.IR**

- **简介: 论文提出Memento，一款主动感知上下文的AR助手，用于记录和召回用户日常记忆。属于增强现实与情境感知任务，解决用户在日常生活中被动获取信息的问题，通过AR主动提供相关回应。**

- **链接: [https://arxiv.org/pdf/2601.17622v2](https://arxiv.org/pdf/2601.17622v2)**

> **作者:** Yoonsang Kim; Yalong Yang; Arie E. Kaufman
>
> **备注:** 8 pages, 5 figures. This is the author's version of the article that will appear at the IEEE Conference on Virtual Reality and 3D User Interfaces Abstracts and Workshops (IEEE VRW) 2026
>
> **摘要:** We introduce Memento, a conversational AR assistant that permanently captures and memorizes user's verbal queries alongside their spatiotemporal and activity contexts. By storing these "memories," Memento discovers connections between users' recurring interests and the contexts that trigger them. Upon detection of similar or identical spatiotemporal activity, Memento proactively recalls user interests and delivers up-to-date responses through AR, seamlessly integrating AR experience into their daily routine. Unlike prior work, each interaction in Memento is not a transient event, but a connected series of interactions with coherent long--term perspective, tailored to the user's broader multimodal (visual, spatial, temporal, and embodied) context. We conduct preliminary evaluation through user feedbacks with participants of diverse expertise in immersive apps, and explore the value of proactive context-aware AR assistant in everyday settings. We share our findings and challenges in designing a proactive, context-aware AR system.
>
---
#### [replaced 010] Language Models are Symbolic Learners in Arithmetic
- **分类: cs.LG; cs.CL**

- **简介: 论文研究语言模型在算术任务中的表现，探讨其是否真正计算或仅依赖模式匹配。通过子组归纳框架分析，发现模型优先学习简单符号捷径，而非算法，导致中间位计算困难。**

- **链接: [https://arxiv.org/pdf/2410.15580v2](https://arxiv.org/pdf/2410.15580v2)**

> **作者:** Chunyuan Deng; Zhiqi Li; Roy Xie; Ruidi Chang; Hanjie Chen
>
> **备注:** TMLR 2026. Code at https://github.com/chili-lab/Symbolic-Arithmetic
>
> **摘要:** The prevailing question in LM performing arithmetic is whether these models learn to truly compute or if they simply master superficial pattern matching. In this paper, we argues for the latter, presenting evidence that LMs act as greedy symbolic learners, prioritizing the simplest possible shortcuts to fit the stats of dataset to solve arithmetic tasks. To investigate this, we introduce subgroup induction, a practical framework adapted from Solomonoff Induction (SI), one of the most powerful universal predictors. Our framework analyzes arithmetic problems by breaking them down into subgroups-minimal mappings between a few input digits and a single output digit. Our primary metric, subgroup quality, measures the viability of these shortcuts. Experiments reveal a distinct U-shaped accuracy pattern in multi-digit multiplication: LMs quickly master the first and last output digits while struggling with those in the middle. We demonstrate this U-shape is not coincidental; it perfectly mirrors the quality of the simplest possible subgroups, those requiring the fewest input tokens. This alignment suggests a core learning mechanism: LMs first learn easy, low-token shortcuts and only incorporate more complex, multi-token patterns as training progresses. They do not learn the algorithm of multiplication but rather a hierarchy of increasingly complex symbol-to-symbol mappings. Ultimately, our findings suggest that the path to arithmetic mastery for LMs is not paved with algorithms, but with a cascade of simple, hierarchically-learned symbolic shortcuts.
>
---
#### [replaced 011] Can We Trust LLM Detectors?
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI文本检测任务，旨在解决现有检测器在分布偏移和风格扰动下的可靠性问题。通过评估两种方法并提出一种监督对比学习框架，揭示了构建通用检测器的挑战。**

- **链接: [https://arxiv.org/pdf/2601.15301v2](https://arxiv.org/pdf/2601.15301v2)**

> **作者:** Jivnesh Sandhan; Harshit Jaiswal; Fei Cheng; Yugo Murawaki
>
> **摘要:** The rapid adoption of LLMs has increased the need for reliable AI text detection, yet existing detectors often fail outside controlled benchmarks. We systematically evaluate 2 dominant paradigms (training-free and supervised) and show that both are brittle under distribution shift, unseen generators, and simple stylistic perturbations. To address these limitations, we propose a supervised contrastive learning (SCL) framework that learns discriminative style embeddings. Experiments show that while supervised detectors excel in-domain, they degrade sharply out-of-domain, and training-free methods remain highly sensitive to proxy choice. Overall, our results expose fundamental challenges in building domain-agnostic detectors. Our code is available at: https://github.com/HARSHITJAIS14/DetectAI
>
---
#### [replaced 012] Confidence intervals for forced alignment boundaries using model ensembles
- **分类: eess.AS; cs.CL; cs.LG; cs.SD**

- **简介: 该论文属于语音处理任务，旨在解决强制对齐边界不确定性问题。通过神经网络集成方法，生成边界置信区间，提升对齐可靠性与可分析性。**

- **链接: [https://arxiv.org/pdf/2506.01256v2](https://arxiv.org/pdf/2506.01256v2)**

> **作者:** Matthew C. Kelley
>
> **备注:** submitted for publication; 7 pages, 1 figure
>
> **摘要:** Forced alignment is a common tool to align audio with orthographic and phonetic transcriptions. Most forced alignment tools provide only a single estimate of a boundary. The present project introduces a method of deriving confidence intervals for these boundaries using a neural network ensemble technique. Ten different segment classifier neural networks were previously trained, and the alignment process is repeated with each model. The alignment ensemble is then used to place the boundary at the median of the boundaries in the ensemble, and 97.85% confidence intervals are constructed using order statistics. Having confidence intervals provides an estimate of the uncertainty in the boundary placement, facilitating tasks like finding boundaries that should be reviewed. As a bonus, on the Buckeye and TIMIT corpora, the ensemble boundaries show a slight overall improvement over using just a single model. The confidence intervals can be emitted during the alignment process as JSON files and a main table for programmatic and statistical analysis. For familiarity, they are also output as Praat TextGrids using a point tier to represent the intervals.
>
---
#### [replaced 013] LOGICAL-COMMONSENSEQA: A Benchmark for Logical Commonsense Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出LOGICAL-COMMONSENSEQA基准，用于逻辑常识推理任务，解决传统单标签评估无法反映语句间逻辑关系的问题。通过引入AND、OR等操作符，评估模型在不同推理场景下的表现。**

- **链接: [https://arxiv.org/pdf/2601.16504v2](https://arxiv.org/pdf/2601.16504v2)**

> **作者:** Obed Junias; Maria Leonor Pacheco
>
> **摘要:** Commonsense reasoning often involves evaluating multiple plausible interpretations rather than selecting a single atomic answer, yet most benchmarks rely on single-label evaluation, obscuring whether statements are jointly plausible, mutually exclusive, or jointly implausible. We introduce LOGICAL-COMMONSENSEQA, a benchmark that re-frames commonsense reasoning as logical composition over pairs of atomic statements using plausibility-level operators (AND, OR, NEITHER/NOR). Evaluating instruction-tuned, reasoning-specialized, and fine-tuned models under zero-shot, few-shot, and chain-of-thought prompting, we find that while models perform reasonably on conjunctive and moderately on disjunctive reasoning, performance degrades sharply on negation-based questions. LOGICAL-COMMONSENSEQA exposes fundamental reasoning limitations and provides a controlled framework for advancing compositional commonsense reasoning.
>
---
#### [replaced 014] Language Agents for Hypothesis-driven Clinical Decision Making with Reinforcement Learning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于临床决策支持任务，旨在解决LLM在临床应用中缺乏交互性和不确定性处理的问题。提出LA-CDM模型，通过强化学习提升诊断准确性和效率。**

- **链接: [https://arxiv.org/pdf/2506.13474v2](https://arxiv.org/pdf/2506.13474v2)**

> **作者:** David Bani-Harouni; Chantal Pellegrini; Ege Özsoy; Matthias Keicher; Nassir Navab
>
> **摘要:** Clinical decision-making is a dynamic, interactive, and cyclic process where doctors have to repeatedly decide on which clinical action to perform and consider newly uncovered information for diagnosis and treatment. Large Language Models (LLMs) have the potential to support clinicians in this process, however, most applications of LLMs in clinical decision support suffer from one of two limitations: Either they assume the unrealistic scenario of immediate availability of all patient information and do not model the interactive and iterative investigation process, or they restrict themselves to the limited "out-of-the-box" capabilities of large pre-trained models without performing task-specific training. In contrast to this, we propose to model clinical decision-making for diagnosis with a hypothesis-driven uncertainty-aware language agent, LA-CDM, that converges towards a diagnosis via repeatedly requesting and interpreting relevant tests. Using a hybrid training paradigm combining supervised and reinforcement learning, we train LA-CDM with three objectives targeting critical aspects of clinical decision-making: accurate hypothesis generation, hypothesis uncertainty estimation, and efficient decision-making. We evaluate our methodology on MIMIC-CDM, a real-world dataset covering four abdominal diseases containing various clinical tests and show the benefit of explicitly training clinical decision-making for increasing diagnostic performance and efficiency.
>
---
#### [replaced 015] SCoPE VLM: Selective Context Processing for Efficient Document Navigation in Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出SCoPE VLM，解决视觉语言模型在长文档导航中的效率问题，通过选择性上下文处理提升多页文档问答性能。**

- **链接: [https://arxiv.org/pdf/2510.21850v2](https://arxiv.org/pdf/2510.21850v2)**

> **作者:** Gyubeum Lim; Yemo Koo; Vijay Krishna Madisetti
>
> **摘要:** Understanding long-context visual information remains a fundamental challenge for vision-language models, particularly in agentic tasks such as GUI control and web navigation. While web pages and GUI environments are inherently structured documents, current VLMs typically neglect decision-oriented document understanding in their training objectives. Existing approaches primarily extend visual embeddings to process long, high-resolution inputs, but these methods are memory-intensive and impractical for locally deployable solutions. To address these issues, we propose SCoPE VLM, a document navigation expert that leverages a novel Chain of Scroll mechanism to selectively and recursively navigate documents, focusing exclusively on relevant segments. We introduce a dedicated data generation pipeline to construct informative Chain of Scroll trajectories and Episodic Group Relative Policy Optimization, a tailored reinforcement learning method to bridge the gap between training and inference. Our method substantially reduces memory usage and effectively models human-like reading behaviors. To the best of our knowledge, SCoPE VLM is the first framework to explicitly model agentic reading patterns in multi-page document question answering, advancing the capabilities of multimodal agents.
>
---
#### [replaced 016] BASIL: Bayesian Assessment of Sycophancy in LLMs
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于AI伦理与可信性研究任务，旨在解决LLMs中过度迎合（sycophancy）行为的评估问题。通过构建贝叶斯框架，区分迎合与理性更新，提出度量方法并验证其有效性。**

- **链接: [https://arxiv.org/pdf/2508.16846v4](https://arxiv.org/pdf/2508.16846v4)**

> **作者:** Katherine Atwell; Pedram Heydari; Anthony Sicilia; Malihe Alikhani
>
> **摘要:** Sycophancy (overly agreeable or flattering behavior) poses a fundamental challenge for human-AI collaboration, particularly in high-stakes decision-making domains such as health, law, and education. A central difficulty in studying sycophancy in large language models (LLMs) is disentangling sycophantic belief shifts from rational changes in behavior driven by new evidence or user-provided information. Existing approaches either measure descriptive behavior changes or apply normative evaluations that rely on objective ground truth, limiting their applicability to subjective or uncertain tasks. We introduce a Bayesian probabilistic framework, grounded in behavioral economics and rational decision theory, that explicitly separates sycophancy from rational belief updating. Within this framework, we achieve three objectives: (i) a descriptive metric that measures sycophancy while controlling for rational responses to evidence; (ii) a normative metric that quantifies how sycophancy leads models astray from Bayesian-consistent belief updating; and (iii) the ability to apply both metrics in settings without ground-truth labels. Applying our framework across multiple LLMs and three uncertainty-driven tasks, we find robust evidence of sycophantic belief shifts and show that their impact on rationality depends on whether models systematically over- or under-update their beliefs. Finally, we demonstrate that a post-hoc calibration method and two fine-tuning strategies (SFT and DPO) substantially reduce Bayesian inconsistency, with particularly strong improvements under explicit sycophancy prompting.
>
---
#### [replaced 017] ThinkNote: Enhancing Knowledge Integration and Utilization of Large Language Models via Constructivist Cognition Modeling
- **分类: cs.CL**

- **简介: 该论文提出ThinkNote，解决LLMs外部知识利用不足的问题，通过构建主义认知建模提升知识整合与推理一致性。**

- **链接: [https://arxiv.org/pdf/2402.13547v4](https://arxiv.org/pdf/2402.13547v4)**

> **作者:** Zhipeng Xu; Zhenghao Liu; Yukun Yan; Shuo Wang; Shi Yu; Zheni Zeng; Chaojun Xiao; Zhiyuan Liu; Ge Yu; Chenyan Xiong
>
> **摘要:** Large Language Models (LLMs) have demonstrated strong performance across a wide range of NLP tasks. However, they often exhibit suboptimal behaviors and inconsistencies when exposed to unfamiliar external information, underscoring their limitations in effectively leveraging such knowledge. Inspired by constructivist learning theory, we propose ThinkNote, a novel framework that enhances the external knowledge utilization of LLMs through a two-stage constructivist cognitive modeling process. Specifically, ThinkNote performs knowledge assimilation to align new information with the model's parametric memory, forming a coherent internal representation. It then applies thought accommodation to adapt internal reasoning, thereby promoting more consistent and reliable outputs. Extensive experimental results demonstrate that ThinkNote achieves a 10% improvement over strong baseline methods on various question-answering benchmarks. Further analysis indicates that ThinkNote effectively integrates and utilizes external knowledge to help LLMs generate accurate responses and improves their self-consistency. All data and codes are available at https://github.com/OpenMatch/ThinkNote.
>
---
#### [replaced 018] DeepSieve: Information Sieving via LLM-as-a-Knowledge-Router
- **分类: cs.CL**

- **简介: 该论文提出DeepSieve，解决LLM在知识密集型查询中的信息筛选问题。通过结构化分解查询并递归路由到合适知识源，提升推理深度与准确性。属于信息筛选与检索增强生成任务。**

- **链接: [https://arxiv.org/pdf/2507.22050v3](https://arxiv.org/pdf/2507.22050v3)**

> **作者:** Minghao Guo; Qingcheng Zeng; Xujiang Zhao; Yanchi Liu; Wenchao Yu; Mengnan Du; Haifeng Chen; Wei Cheng
>
> **备注:** Accepted by EACL Findings 2026
>
> **摘要:** Large Language Models (LLMs) excel at many reasoning tasks but struggle with knowledge-intensive queries due to their inability to dynamically access up-to-date or domain-specific information. Retrieval-Augmented Generation (RAG) has emerged as a promising solution, enabling LLMs to ground their responses in external sources. However, existing RAG methods lack fine-grained control over both the query and source sides, often resulting in noisy retrieval and shallow reasoning. In this work, we introduce DeepSieve, an agentic RAG framework that incorporates information sieving via LLM-as-a-knowledge-router. DeepSieve decomposes complex queries into structured sub-questions and recursively routes each to the most suitable knowledge source, filtering irrelevant information through a multi-stage distillation process. Our design emphasizes modularity, transparency, and adaptability, leveraging recent advances in agentic system design. Experiments on multi-hop QA tasks across heterogeneous sources demonstrate improved reasoning depth, retrieval precision, and interpretability over conventional RAG approaches. Our codes are available at https://github.com/MinghoKwok/DeepSieve.
>
---
#### [replaced 019] Reconstructing KV Caches with Cross-layer Fusion For Enhanced Transformers
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决Transformer解码器在长序列中的KV缓存内存瓶颈问题。通过跨层融合方法FusedKV和FusedKV-Lite，提升效率并减少内存占用。**

- **链接: [https://arxiv.org/pdf/2512.03870v2](https://arxiv.org/pdf/2512.03870v2)**

> **作者:** Hongzhan Lin; Zhiqi Bai; Xinmiao Zhang; Sen Yang; Xiang Li; Siran Yang; Yunlong Xu; Jiaheng Liu; Yongchi Zhao; Jiamang Wang; Yuchi Xu; Wenbo Su; Bo Zheng
>
> **备注:** Accepted by ICLR2026
>
> **摘要:** Transformer decoders have achieved strong results across tasks, but the memory required for the KV cache becomes prohibitive at long sequence lengths. Although Cross-layer KV Cache sharing (e.g., YOCO, CLA) offers a path to mitigate KV Cache bottleneck, it typically underperforms within-layer methods like GQA. To understand the root cause, we investigate the information flow of keys and values of the top-layers. Our preliminary reveals a clear distribution: values are predominantly derived from the bottom layer, while keys draw more information from both bottom and middle layers. Building upon this, we propose FusedKV, whose top-layer KV caches are a learnable fusion of the most informative ones from the bottom and middle layers. This fusion operates directly on post-RoPE keys, preserving relative positional information without the computational cost of re-applying rotary embeddings. To further improve efficiency, we propose FusedKV-Lite, an cross-layer sharing approach, where top-layer KV caches are directly derived from the bottom-layer values and the middle-layer keys. Compared to FusedKV, FusedKV-Lite reduces I/O overhead at the cost of a slight increase in perplexity. In experiments on LLMs ranging from 332M to 4B parameters, our proposed method reduce 50\% cache memory while achieving lower validation perplexity than the standard Transformer decoder, establishing it as a memory-efficient, high-performance architectural alternative.
>
---
#### [replaced 020] UQLM: A Python Package for Uncertainty Quantification in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理中的不确定性量化任务，旨在解决LLM幻觉问题。提出UQLM工具包，使用UQ技术检测幻觉，提供置信度评分以提高模型可靠性。**

- **链接: [https://arxiv.org/pdf/2507.06196v2](https://arxiv.org/pdf/2507.06196v2)**

> **作者:** Dylan Bouchard; Mohit Singh Chauhan; David Skarbrevik; Ho-Kyeong Ra; Viren Bajaj; Zeya Ahmad
>
> **备注:** Accepted by JMLR; UQLM Repository: https://github.com/cvs-health/uqlm
>
> **摘要:** Hallucinations, defined as instances where Large Language Models (LLMs) generate false or misleading content, pose a significant challenge that impacts the safety and trust of downstream applications. We introduce UQLM, a Python package for LLM hallucination detection using state-of-the-art uncertainty quantification (UQ) techniques. This toolkit offers a suite of UQ-based scorers that compute response-level confidence scores ranging from 0 to 1. This library provides an off-the-shelf solution for UQ-based hallucination detection that can be easily integrated to enhance the reliability of LLM outputs.
>
---
#### [replaced 021] SemCoT: Accelerating Chain-of-Thought Reasoning through Semantically-Aligned Implicit Tokens
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决CoT推理效率低和语义不一致的问题。提出SemCoT框架，通过优化生成速度和保持语义对齐，提升CoT效果与效率。**

- **链接: [https://arxiv.org/pdf/2510.24940v3](https://arxiv.org/pdf/2510.24940v3)**

> **作者:** Yinhan He; Wendy Zheng; Yaochen Zhu; Zaiyi Zheng; Lin Su; Sriram Vasudevan; Qi Guo; Liangjie Hong; Jundong Li
>
> **摘要:** The verbosity of Chain-of-Thought (CoT) reasoning hinders its mass deployment in efficiency-critical applications. Recently, implicit CoT approaches have emerged, which encode reasoning steps within LLM's hidden embeddings (termed ``implicit reasoning'') rather than explicit tokens. This approach accelerates CoT by reducing the reasoning length and bypassing some LLM components. However, existing implicit CoT methods face two significant challenges: (1) they fail to preserve the semantic alignment between the implicit reasoning (when transformed to natural language) and the ground-truth reasoning, resulting in a significant CoT performance degradation, and (2) they focus on reducing the length of the implicit reasoning; however, they neglect the considerable time cost for an LLM to generate one individual implicit reasoning token. To tackle these challenges, we propose a novel semantically-aligned implicit CoT framework termed SemCoT. In particular, for the first challenge, we design a contrastively trained sentence transformer that evaluates semantic alignment between implicit and explicit reasoning, which is used to enforce semantic preservation during implicit reasoning optimization. To address the second challenge, we introduce an efficient implicit reasoning generator by finetuning a lightweight language model using knowledge distillation. This generator is guided by our sentence transformer to distill ground-truth reasoning into semantically aligned implicit reasoning, while also optimizing for accuracy. SemCoT is the first approach that enhances CoT efficiency by jointly optimizing token-level generation speed and preserving semantic alignment with ground-truth reasoning. Extensive experiments demonstrate the superior performance of SemCoT compared to state-of-the-art methods in both efficiency and effectiveness. Our code can be found at https://github.com/YinhanHe123/SemCoT/.
>
---
#### [replaced 022] PROPHET: An Inferable Future Forecasting Benchmark with Causal Intervened Likelihood Estimation
- **分类: cs.CL**

- **简介: 该论文提出PROPHET基准，解决未来事件预测中问题不可推断的问题，通过因果干预似然评估推断性，构建可推断的预测任务。**

- **链接: [https://arxiv.org/pdf/2504.01509v2](https://arxiv.org/pdf/2504.01509v2)**

> **作者:** Zhengwei Tao; Pu Wu; Zhi Jin; Xiaoying Bai; Haiyan Zhao; Chengfeng Dou; Xiancai Chen; Jia Li; Linyu Li; Chongyang Tao; Wentao Zhang
>
> **摘要:** Predicting future events based on news on the Web stands as one of the ultimate aspirations of artificial intelligence. Recent advances in large language model (LLM)-based systems have shown remarkable potential in forecasting future events, thereby garnering significant interest in the research community. Currently, several benchmarks have been established to evaluate the forecasting capabilities by formalizing the event prediction as a retrieval-augmented generation (RAG)-and-reasoning task. In these benchmarks, each prediction question is answered with relevant retrieved news articles downloaded from the Web. However, because there is no consideration of whether the questions can be supported by valid or sufficient supporting rationales, some of the questions in these benchmarks may be inherently noninferable. To address this issue, we introduce a new benchmark, PROPHET, which comprises inferable forecasting questions paired with relevant news for retrieval. To ensure the inferability of the benchmark, we propose Causal Intervened Likelihood (CIL), a statistical measure that assesses inferability through causal inference. In constructing this benchmark, we first collected recent trend forecasting questions, and then filtered the data using CIL resulting in an inferable benchmark for future forecasting. Through extensive experiments, we first demonstrate the validity of CIL and in-depth investigations into future forecasting with the aid of CIL. Subsequently, we evaluate several representative prediction methods on PROPHET. The overall results draws valuable insights for task of future directions.
>
---
#### [replaced 023] AI-generated data contamination erodes pathological variability and diagnostic reliability
- **分类: cs.CY; cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于医疗AI领域，研究AI生成数据对病理多样性与诊断可靠性的影响。工作包括分析合成数据，发现模型趋向通用表型，导致关键病理消失，并提出混合真实数据的缓解策略。**

- **链接: [https://arxiv.org/pdf/2601.12946v3](https://arxiv.org/pdf/2601.12946v3)**

> **作者:** Hongyu He; Shaowen Xiang; Ye Zhang; Yingtao Zhu; Jin Zhang; Hao Deng; Emily Alsentzer; Qingyu Chen; Kun-Hsing Yu; Andrew Marshall; Tingting Chen; Srinivas Anumasa; Daniel Ebner; Dean Ho; Kee Yuan Ngiam; Ching-Yu Cheng; Dianbo Liu
>
> **备注:** *Corresponding author: Dianbo Liu (dianbo@nus.edu.sg)
>
> **摘要:** Generative artificial intelligence (AI) is rapidly populating medical records with synthetic content, creating a feedback loop where future models are increasingly at risk of training on uncurated AI-generated data. However, the clinical consequences of this AI-generated data contamination remain unexplored. Here, we show that in the absence of mandatory human verification, this self-referential cycle drives a rapid erosion of pathological variability and diagnostic reliability. By analysing more than 800,000 synthetic data points across clinical text generation, vision-language reporting, and medical image synthesis, we find that models progressively converge toward generic phenotypes regardless of the model architecture. Specifically, rare but critical findings, including pneumothorax and effusions, vanish from the synthetic content generated by AI models, while demographic representations skew heavily toward middle-aged male phenotypes. Crucially, this degradation is masked by false diagnostic confidence; models continue to issue reassuring reports while failing to detect life-threatening pathology, with false reassurance rates tripling to 40%. Blinded physician evaluation confirms that this decoupling of confidence and accuracy renders AI-generated documentation clinically useless after just two generations. We systematically evaluate three mitigation strategies, finding that while synthetic volume scaling fails to prevent collapse, mixing real data with quality-aware filtering effectively preserves diversity. Ultimately, our results suggest that without policy-mandated human oversight, the deployment of generative AI threatens to degrade the very healthcare data ecosystems it relies upon.
>
---
#### [replaced 024] Improving Implicit Hate Speech Detection via a Community-Driven Multi-Agent Framework
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于隐性仇恨言论检测任务，旨在提升检测的准确性和公平性。通过构建社区驱动的多智能体框架，融合社会文化背景，优化分类效果。**

- **链接: [https://arxiv.org/pdf/2601.09342v2](https://arxiv.org/pdf/2601.09342v2)**

> **作者:** Ewelina Gajewska; Katarzyna Budzynska; Jarosław A Chudziak
>
> **备注:** This paper has been accepted for the upcoming 18th International Conference on Agents and Artificial Intelligence (ICAART-2026), Marbella, Spain. The final published version will appear in the official conference proceedings
>
> **摘要:** This work proposes a contextualised detection framework for implicitly hateful speech, implemented as a multi-agent system comprising a central Moderator Agent and dynamically constructed Community Agents representing specific demographic groups. Our approach explicitly integrates socio-cultural context from publicly available knowledge sources, enabling identity-aware moderation that surpasses state-of-the-art prompting methods (zero-shot prompting, few-shot prompting, chain-of-thought prompting) and alternative approaches on a challenging ToxiGen dataset. We enhance the technical rigour of performance evaluation by incorporating balanced accuracy as a central metric of classification fairness that accounts for the trade-off between true positive and true negative rates. We demonstrate that our community-driven consultative framework significantly improves both classification accuracy and fairness across all target groups.
>
---
#### [replaced 025] GOFAI meets Generative AI: Development of Expert Systems by means of Large Language Models
- **分类: cs.AI; cs.CL; cs.SC**

- **简介: 论文探讨如何利用大语言模型构建可靠专家系统，解决其生成内容不可靠的问题。通过限定领域和结构化提示，将知识转换为可验证的Prolog符号表示，提升系统的可解释性和准确性。**

- **链接: [https://arxiv.org/pdf/2507.13550v2](https://arxiv.org/pdf/2507.13550v2)**

> **作者:** Eduardo C. Garrido-Merchán; Cristina Puente
>
> **摘要:** The development of large language models (LLMs) has successfully transformed knowledge-based systems such as open domain question nswering, which can automatically produce vast amounts of seemingly coherent information. Yet, those models have several disadvantages like hallucinations or confident generation of incorrect or unverifiable facts. In this paper, we introduce a new approach to the development of expert systems using LLMs in a controlled and transparent way. By limiting the domain and employing a well-structured prompt-based extraction approach, we produce a symbolic representation of knowledge in Prolog, which can be validated and corrected by human experts. This approach also guarantees interpretability, scalability and reliability of the developed expert systems. Via quantitative and qualitative experiments with Claude Sonnet 3.7 and GPT-4.1, we show strong adherence to facts and semantic coherence on our generated knowledge bases. We present a transparent hybrid solution that combines the recall capacity of LLMs with the precision of symbolic systems, thereby laying the foundation for dependable AI applications in sensitive domains.
>
---
#### [replaced 026] Text2Grad: Reinforcement Learning from Natural Language Feedback
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Text2Grad，解决传统强化学习中奖励信号粗粒度的问题。通过将自然语言反馈转为细粒度梯度，提升模型调整的精确性和可解释性。属于强化学习与自然语言处理任务。**

- **链接: [https://arxiv.org/pdf/2505.22338v2](https://arxiv.org/pdf/2505.22338v2)**

> **作者:** Hanyang Wang; Lu Wang; Chaoyun Zhang; Tianjun Mao; Si Qin; Qingwei Lin; Saravan Rajmohan; Dongmei Zhang
>
> **备注:** The code for our method is available at https://github.com/microsoft/Text2Grad
>
> **摘要:** Traditional RLHF optimizes language models with coarse, scalar rewards that mask the fine-grained reasons behind success or failure, leading to slow and opaque learning. Recent work augments RL with textual critiques through prompting or reflection, improving interpretability but leaving model parameters untouched. We introduce Text2Grad, a reinforcement-learning paradigm that turns free-form textual feedback into span-level gradients. Given human (or programmatic) critiques, Text2Grad aligns each feedback phrase with the relevant token spans, converts these alignments into differentiable reward signals, and performs gradient updates that directly refine the offending portions of the model's policy. This yields precise, feedback-conditioned adjustments instead of global nudges. Text2Grad is realized through three components: (1) a high-quality feedback-annotation pipeline that pairs critiques with token spans; (2) a fine-grained reward model that predicts span-level reward on answers while generating explanatory critiques; and (3) a span-level policy optimizer that back-propagates natural-language gradients. Across summarization, code generation, and question answering, Text2Grad consistently surpasses scalar-reward RL and prompt-only baselines, providing both higher task metrics and richer interpretability. Our results suggest that natural-language feedback can serve not only as explanations, but also as actionable training signals for fine-grained alignment. The code for our method is available at https://github.com/microsoft/Text2Grad.
>
---
#### [replaced 027] AECBench: A Hierarchical Benchmark for Knowledge Evaluation of Large Language Models in the AEC Field
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于知识评估任务，旨在解决LLMs在AEC领域可靠性评估问题。构建了AECBench基准，涵盖五级评估框架，测试模型在不同认知层次的表现。**

- **链接: [https://arxiv.org/pdf/2509.18776v2](https://arxiv.org/pdf/2509.18776v2)**

> **作者:** Chen Liang; Zhaoqi Huang; Haofen Wang; Fu Chai; Chunying Yu; Huanhuan Wei; Zhengjie Liu; Yanpeng Li; Hongjun Wang; Ruifeng Luo; Xianzhong Zhao
>
> **摘要:** Large language models (LLMs), as a novel information technology, are seeing increasing adoption in the Architecture, Engineering, and Construction (AEC) field. They have shown their potential to streamline processes throughout the building lifecycle. However, the robustness and reliability of LLMs in such a specialized and safety-critical domain remain to be evaluated. To address this challenge, this paper establishes AECBench, a comprehensive benchmark designed to quantify the strengths and limitations of current LLMs in the AEC domain. The benchmark features a five-level, cognition-oriented evaluation framework (i.e., Knowledge Memorization, Understanding, Reasoning, Calculation, and Application). Based on the framework, 23 representative evaluation tasks were defined. These tasks were derived from authentic AEC practice, with scope ranging from codes retrieval to specialized documents generation. Subsequently, a 4,800-question dataset encompassing diverse formats, including open-ended questions, was crafted primarily by engineers and validated through a two-round expert review. Furthermore, an "LLM-as-a-Judge" approach was introduced to provide a scalable and consistent methodology for evaluating complex, long-form responses leveraging expert-derived rubrics. Through the evaluation of nine LLMs, a clear performance decline across five cognitive levels was revealed. Despite demonstrating proficiency in foundational tasks at the Knowledge Memorization and Understanding levels, the models showed significant performance deficits, particularly in interpreting knowledge from tables in building codes, executing complex reasoning and calculation, and generating domain-specific documents. Consequently, this study lays the groundwork for future research and development aimed at the robust and reliable integration of LLMs into safety-critical engineering practices.
>
---
#### [replaced 028] Rethinking Creativity Evaluation: A Critical Analysis of Existing Creativity Evaluations
- **分类: cs.CL**

- **简介: 该论文属于创意评估任务，旨在解决现有创意评价方法不一致的问题。通过分析四种评估方法在不同领域的表现，指出其局限性并呼吁更可靠的评估框架。**

- **链接: [https://arxiv.org/pdf/2508.05470v3](https://arxiv.org/pdf/2508.05470v3)**

> **作者:** Li-Chun Lu; Miri Liu; Pin-Chun Lu; Yufei Tian; Shao-Hua Sun; Nanyun Peng
>
> **备注:** EACL 2026, 24 pages, 6 figures
>
> **摘要:** We examine, analyze, and compare four representative creativity measures--perplexity, LLM-as-a-Judge, the Creativity Index (CI; measuring n-gram overlap with web corpora), and syntactic templates (detecting repetition of common part-of-speech patterns)--across the diverse creative domains, such as creative writing, unconventional problem-solving, and research ideation. For each domain, we compile datasets with human-aligned creative and uncreative examples and evaluate each metric's ability to discriminate between the two sets. Our analyses reveal limited consistency both across domains and metrics, as metrics that distinguish creativity in one domain fail in others (e.g., CI correctly distinguishes in creative writing but fails in problem-solving), and different metrics often disagree on the same data points (e.g., CI suggests one set to be more creative, while perplexity indicates the other set to be more creative.) We highlight key limitations, such as perplexity reflecting fluency rather than novelty; LLM-as-a-Judge producing inconsistent judgments under minor prompt variations and exhibiting bias towards particular labels; CI primarily measuring lexical diversity, with high sensitivity to implementation choices; and syntactic templates being ineffective in settings dominated by formulaic language. Our findings underscore the need for more robust, generalizable evaluation frameworks that better align with human judgments of creativity.
>
---
#### [replaced 029] Agentic Plan Caching: Test-Time Memory for Fast and Cost-Efficient LLM Agents
- **分类: cs.DC; cs.AI; cs.CL; cs.LG; cs.PF**

- **简介: 该论文提出Agentic Plan Caching（APC），解决LLM代理在复杂任务中的高成本和延迟问题。通过测试时记忆复用结构化计划模板，提升效率。属于LLM代理优化任务。**

- **链接: [https://arxiv.org/pdf/2506.14852v2](https://arxiv.org/pdf/2506.14852v2)**

> **作者:** Qizheng Zhang; Michael Wornow; Gerry Wan; Kunle Olukotun
>
> **备注:** NeurIPS 2025. 27 pages
>
> **摘要:** LLM-based agent applications have shown increasingly remarkable capabilities in complex workflows but incur substantial costs and latency due to extensive planning and reasoning requirements. Existing LLM caching techniques (like context caching and semantic caching), primarily designed for serving chatbots, are insufficient for agent applications where outputs depend on external data and environmental contexts. We propose Agentic Plan Caching (APC), a novel test-time memory that extracts, stores, adapts, and reuses structured plan templates from planning stages of agent applications across semantically similar tasks to reduce the cost and latency of serving. Unlike traditional semantic caching, our system extracts plan templates from completed agent executions at test-time, employs keyword extraction to match new requests against cached plans, and utilizes lightweight models to adapt these templates to task-specific plans with contexts. Evaluation across multiple real-world agent applications shows that our system can reduce costs by 50.31% and latency by 27.28% on average while maintaining performance, offering a more efficient solution for serving LLM-based agents that complements existing LLM serving infrastructures.
>
---
#### [replaced 030] coTherapist: A Behavior-Aligned Small Language Model to Support Mental Healthcare Experts
- **分类: cs.CL**

- **简介: 该论文提出coTherapist，一个用于支持心理健康专家的小型语言模型。旨在解决心理医疗资源不足问题，通过模拟治疗技能提供可靠帮助。**

- **链接: [https://arxiv.org/pdf/2601.10246v2](https://arxiv.org/pdf/2601.10246v2)**

> **作者:** Prottay Kumar Adhikary; Reena Rawat; Tanmoy Chakraborty
>
> **摘要:** Access to mental healthcare is increasingly strained by workforce shortages and rising demand, motivating the development of intelligent systems that can support mental healthcare experts. We introduce coTherapist, a unified framework utilizing a small language model to emulate core therapeutic competencies through domain-specific fine-tuning, retrieval augmentation, and agentic reasoning. Evaluation on clinical queries demonstrates that coTherapist generates more relevant and clinically grounded responses than contemporary baselines. Using our novel T-BARS rubric and psychometric profiling, we confirm coTherapist exhibits high empathy and therapist-consistent personality traits. Furthermore, human evaluation by domain experts validates that coTherapist delivers accurate, trustworthy, and safe responses. coTherapist was deployed and tested by clinical experts. Collectively, these findings demonstrate that small models can be engineered to exhibit expert-like behavior, offering a scalable pathway for digital mental health tools.
>
---
#### [replaced 031] Towards Automated Smart Contract Generation: Evaluation, Benchmarking, and Retrieval-Augmented Repair
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文属于智能合约生成任务，旨在解决Solidity代码生成的正确性评估问题。提出SolBench基准和RAR框架，提升代码功能正确性与效率。**

- **链接: [https://arxiv.org/pdf/2503.01098v2](https://arxiv.org/pdf/2503.01098v2)**

> **作者:** Zaoyu Chen; Haoran Qin; Nuo Chen; Xiangyu Zhao; Lei Xue; Xiapu Luo; Xiao-Ming Wu
>
> **备注:** Accepted by FSE 2026
>
> **摘要:** Smart contracts, predominantly written in Solidity and deployed on blockchains such as Ethereum, are immutable after deployment, making functional correctness critical. However, existing evaluations of Solidity code generation rely largely on surface-level metrics (e.g., BLEU, CrystalBLEU) or manual inspection, which correlate poorly with functional correctness. In contrast to Python, Solidity lacks large-scale, execution-based benchmarks, limiting systematic evaluation of large language models for smart contract development. We introduce SolBench, a comprehensive benchmark and automated testing pipeline for Solidity that emphasizes functional correctness via differential fuzzing. SolBench consists of 28825 functions extracted from 7604 real-world smart contracts collected from Etherscan (genesis-2024), spanning ten application domains. We benchmark 14 diverse LLMs, covering open and closed models, 1.3B-671B parameters, and both general-purpose and code-specialized architectures. The dominant failure mode is missing critical intra-contract information, such as state variables and type definitions. Providing full-contract context improves accuracy but incurs prohibitive inference costs. To address this, we propose Retrieval-Augmented Repair (RAR), a cost-effective framework that integrates execution feedback into code repair. RAR uses compiler and runtime error messages to retrieve only the minimal contract snippets needed to correct a target function, avoiding full-context inference. This significantly reduces input length while improving functional correctness. We further analyze retrieval and repair strategies within RAR, demonstrating consistent gains in accuracy and efficiency. SolBench and RAR enable principled, execution-based evaluation and economical improvement of Solidity code generation. Dataset and code are publicly available at https://github.com/ZaoyuChen/SolBench.
>
---
#### [replaced 032] Can professional translators identify machine-generated text?
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本检测任务，旨在探究专业译者能否识别AI生成的意大利语短篇小说。研究通过实验分析译者的判断依据及准确性。**

- **链接: [https://arxiv.org/pdf/2601.15828v2](https://arxiv.org/pdf/2601.15828v2)**

> **作者:** Michael Farrell
>
> **备注:** 10 pages, table numbers corrected
>
> **摘要:** This study investigates whether professional translators can reliably identify short stories generated in Italian by artificial intelligence (AI) without prior specialized training. Sixty-nine translators took part in an in-person experiment, where they assessed three anonymized short stories - two written by ChatGPT-4o and one by a human author. For each story, participants rated the likelihood of AI authorship and provided justifications for their choices. While average results were inconclusive, a statistically significant subset (16.2%) successfully distinguished the synthetic texts from the human text, suggesting that their judgements were informed by analytical skill rather than chance. However, a nearly equal number misclassified the texts in the opposite direction, often relying on subjective impressions rather than objective markers, possibly reflecting a reader preference for AI-generated texts. Low burstiness and narrative contradiction emerged as the most reliable indicators of synthetic authorship, with unexpected calques, semantic loans and syntactic transfer from English also reported. In contrast, features such as grammatical accuracy and emotional tone frequently led to misclassification. These findings raise questions about the role and scope of synthetic-text editing in professional contexts.
>
---
#### [replaced 033] Why Do Speech Language Models Fail to Generate Semantically Coherent Outputs? A Modality Evolving Perspective
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于语音语言模型研究，旨在解决SLMs生成语义不连贯输出的问题。通过分析语音特性，揭示了语音序列长度、语音信息复杂性等因素的影响，为改进SLMs提供方向。**

- **链接: [https://arxiv.org/pdf/2412.17048v2](https://arxiv.org/pdf/2412.17048v2)**

> **作者:** Hankun Wang; Haoran Wang; Yiwei Guo; Zhihan Li; Chenpeng Du; Kai Yu
>
> **备注:** 5 pages, 3 figures, 4 tables. Accepted to IEEE ICASSP 2026
>
> **摘要:** Although text-based large language models exhibit human-level writing ability and remarkable intelligence, speech language models (SLMs) still struggle to generate semantically coherent outputs. There are several potential reasons for this performance degradation: (A) speech tokens mainly provide phonetic information rather than semantic information, (B) the length of speech sequences is much longer than that of text sequences, and (C) paralinguistic information, such as prosody, introduces additional complexity and variability. In this paper, we explore the influence of three key factors separately by transiting the modality from text to speech in an evolving manner. Our findings reveal that the impact of the three factors varies. Factor A has a relatively minor impact, factor B influences syntactical and semantic modeling more obviously, and factor C exerts the most significant impact, particularly in the basic lexical modeling. Based on these findings, we provide insights into the unique challenges of training SLMs and highlight pathways to develop more effective end-to-end SLMs.
>
---
#### [replaced 034] EmoBench-M: Benchmarking Emotional Intelligence for Multimodal Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于情感智能任务，旨在解决MLLMs在多模态情感理解上的不足。构建了EmoBench-M基准，评估模型在情绪识别、对话理解和社交复杂情感分析方面的能力。**

- **链接: [https://arxiv.org/pdf/2502.04424v3](https://arxiv.org/pdf/2502.04424v3)**

> **作者:** He Hu; Yucheng Zhou; Lianzhong You; Hongbo Xu; Qianning Wang; Zheng Lian; Fei Richard Yu; Fei Ma; Laizhong Cui
>
> **摘要:** With the integration of Multimodal large language models (MLLMs) into robotic systems and various AI applications, embedding emotional intelligence (EI) capabilities into these models is essential for enabling robots to effectively address human emotional needs and interact seamlessly in real-world scenarios. Existing static, text-based, or text-image benchmarks overlook the multimodal complexities of real-world interactions and fail to capture the dynamic, multimodal nature of emotional expressions, making them inadequate for evaluating MLLMs' EI. Based on established psychological theories of EI, we build EmoBench-M, a novel benchmark designed to evaluate the EI capability of MLLMs across 13 valuation scenarios from three key dimensions: foundational emotion recognition, conversational emotion understanding, and socially complex emotion analysis. Evaluations of both open-source and closed-source MLLMs on EmoBench-M reveal a significant performance gap between them and humans, highlighting the need to further advance their EI capabilities. All benchmark resources, including code and datasets, are publicly available at https://emo-gml.github.io/.
>
---
#### [replaced 035] A Context-Aware Dual-Metric Framework for Confidence Estimation in Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于可信度估计任务，旨在提升大语言模型输出的可靠性。针对现有方法忽视上下文相关性的问题，提出CRUX框架，结合上下文一致性和熵减进行可信度评估。**

- **链接: [https://arxiv.org/pdf/2508.00600v2](https://arxiv.org/pdf/2508.00600v2)**

> **作者:** Mingruo Yuan; Shuyi Zhang; Ben Kao
>
> **摘要:** Accurate confidence estimation is essential for trustworthy large language models (LLMs) systems, as it empowers the user to determine when to trust outputs and enables reliable deployment in safety-critical applications. Current confidence estimation methods for LLMs neglect the relevance between responses and contextual information, a crucial factor in output quality evaluation, particularly in scenarios where background knowledge is provided. To bridge this gap, we propose CRUX (Context-aware entropy Reduction and Unified consistency eXamination), the first framework that integrates context faithfulness and consistency for confidence estimation via two novel metrics. First, contextual entropy reduction represents data uncertainty with the information gain through contrastive sampling with and without context. Second, unified consistency examination captures potential model uncertainty through the global consistency of the generated answers with and without context. Experiments across three benchmark datasets (CoQA, SQuAD, QuAC) and two domain-specific datasets (BioASQ, EduQG) demonstrate CRUX's effectiveness, achieving the highest AUROC than existing baselines.
>
---
#### [replaced 036] ReVision: A Dataset and Baseline VLM for Privacy-Preserving Task-Oriented Visual Instruction Rewriting
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出ReVision，解决多模态交互中的隐私和实时性问题。通过视觉指令重写任务，将多模态指令转为文本命令，提升隐私保护与设备端可用性。**

- **链接: [https://arxiv.org/pdf/2502.14780v3](https://arxiv.org/pdf/2502.14780v3)**

> **作者:** Abhijit Mishra; Mingda Li; Hsiang Fu; Richard Noh; Minji Kim
>
> **备注:** In Proceedings of the IJCNLP-AACL 2025
>
> **摘要:** Efficient and privacy-preserving multimodal interaction is essential as AR, VR, and modern smartphones with powerful cameras become primary interfaces for human-computer communication. Existing powerful large vision-language models (VLMs) enabling multimodal interaction often rely on cloud-based processing, raising significant concerns about (1) visual privacy by transmitting sensitive vision data to servers, and (2) their limited real-time, on-device usability. This paper explores Visual Instruction Rewriting, a novel approach that transforms multimodal instructions into text-only commands, allowing seamless integration of lightweight on-device instruction rewriter VLMs (250M parameters) with existing conversational AI systems, enhancing vision data privacy. To achieve this, we present a dataset of over 39,000 examples across 14 domains and develop a compact VLM, pretrained on image captioning datasets and fine-tuned for instruction rewriting. Experimental results, evaluated through NLG metrics such as BLEU, METEOR, and ROUGE, along with semantic parsing analysis, demonstrate that even a quantized version of the model (<500MB storage footprint) can achieve effective instruction rewriting, thus enabling privacy-focused, multimodal AI applications.
>
---
#### [replaced 037] Large Multimodal Models for Low-Resource Languages: A Survey
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于多模态模型研究任务，旨在解决低资源语言中大模型适应问题。通过分析117篇文献，总结了提升模型性能的方法与挑战。**

- **链接: [https://arxiv.org/pdf/2502.05568v3](https://arxiv.org/pdf/2502.05568v3)**

> **作者:** Marian Lupascu; Ana-Cristina Rogoz; Mihai Sorin Stupariu; Radu Tudor Ionescu
>
> **备注:** Accepted in Information Fusion
>
> **摘要:** In this survey, we systematically analyze techniques used to adapt large multimodal models (LMMs) for low-resource (LR) languages, examining approaches ranging from visual enhancement and data creation to cross-modal transfer and fusion strategies. Through a comprehensive analysis of 117 studies across 96 LR languages, we identify key patterns in how researchers tackle the challenges of limited data and computational resources. We categorize works into resource-oriented and method-oriented contributions, further dividing contributions into relevant sub-categories. We compare method-oriented contributions in terms of performance and efficiency, discussing benefits and limitations of representative studies. We find that visual information often serves as a crucial bridge for improving model performance in LR settings, though significant challenges remain in areas such as hallucination mitigation and computational efficiency. In summary, we provide researchers with a clear understanding of current approaches and remaining challenges in making LMMs more accessible to speakers of LR (understudied) languages. We complement our survey with an open-source repository available at: https://github.com/marianlupascu/LMM4LRL-Survey.
>
---
#### [replaced 038] DND: Boosting Large Language Models with Dynamic Nested Depth
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出DND方法，用于提升大语言模型性能。通过动态选择关键token重新处理，解决模型在复杂任务中的效率与效果平衡问题。**

- **链接: [https://arxiv.org/pdf/2510.11001v3](https://arxiv.org/pdf/2510.11001v3)**

> **作者:** Tieyuan Chen; Xiaodong Chen; Haoxing Chen; Zhenzhong Lan; Weiyao Lin; Jianguo Li
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** We introduce Dynamic Nested Depth (DND), a novel method that improves performance for off-the-shelf LLMs by selecting critical tokens to reprocess in a nested depth manner. Specifically, at the end of the given transformer layer, DND identifies more critical tokens with a router and feeds them back for an extra round of processing, effectively ``reviewing" difficult tokens while avoiding redundant computation for easier ones. The dynamic selection mechanism is tailored for precise control via two novel strategies: a router controlling loss to enhance token selection distinguishability, and a threshold control scheme to ensure selection stability. We demonstrate the effectiveness of DND by directly integrating it into pre-trained dense and MoE models during a post-training phase. On diverse benchmarks, this approach boosts the performances of the dense Qwen3-1.7B by 1.88% and the MoE Qwen3-30B-A3B by 0.87%, all with a minimal parameter and computing increase.
>
---
#### [replaced 039] Endless Terminals: Scaling RL Environments for Terminal Agents
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出Endless Terminals，一个自动生成终端任务的强化学习环境管道，解决传统基准不适合训练的问题，通过大规模任务训练提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.16443v2](https://arxiv.org/pdf/2601.16443v2)**

> **作者:** Kanishk Gandhi; Shivam Garg; Noah D. Goodman; Dimitris Papailiopoulos
>
> **摘要:** Environments are the bottleneck for self-improving agents. Current terminal benchmarks were built for evaluation, not training; reinforcement learning requires a scalable pipeline, not just a dataset. We introduce Endless Terminals, a fully autonomous pipeline that procedurally generates terminal-use tasks without human annotation. The pipeline has four stages: generating diverse task descriptions, building and validating containerized environments, producing completion tests, and filtering for solvability. From this pipeline we obtain 3255 tasks spanning file operations, log management, data processing, scripting, and database operations. We train agents using vanilla PPO with binary episode level rewards and a minimal interaction loop: no retrieval, multi-agent coordination, or specialized tools. Despite this simplicity, models trained on Endless Terminals show substantial gains: on our held-out dev set, Llama-3.2-3B improves from 4.0% to 18.2%, Qwen2.5-7B from 10.7% to 53.3%, and Qwen3-8B-openthinker-sft from 42.6% to 59.0%. These improvements transfer to human-curated benchmarks: models trained on Endless Terminals show substantial gains on held out human curated benchmarks: on TerminalBench 2.0, Llama-3.2-3B improves from 0.0% to 2.2%, Qwen2.5-7B from 2.2% to 3.4%, and Qwen3-8B-openthinker-sft from 1.1% to 6.7%, in each case outperforming alternative approaches including models with more complex agentic scaffolds. These results demonstrate that simple RL succeeds when environments scale.
>
---
#### [replaced 040] MobileSafetyBench: Evaluating Safety of Autonomous Agents in Mobile Device Control
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于移动设备控制安全评估任务，旨在解决自主代理在移动环境中的安全性问题。提出MobileSafetyBench基准，测试代理在真实场景中的安全性和抗攻击能力。**

- **链接: [https://arxiv.org/pdf/2410.17520v3](https://arxiv.org/pdf/2410.17520v3)**

> **作者:** Juyong Lee; Dongyoon Hahm; June Suk Choi; W. Bradley Knox; Kimin Lee
>
> **摘要:** Autonomous agents powered by large language models (LLMs) show promising potential in assistive tasks across various domains, including mobile device control. As these agents interact directly with personal information and device settings, ensuring their safe and reliable behavior is crucial to prevent undesirable outcomes. However, no benchmark exists for standardized evaluation of the safety of mobile device-control agents. In this work, we introduce MobileSafetyBench, a benchmark designed to evaluate the safety of device-control agents within a realistic mobile environment based on Android emulators. We develop a diverse set of tasks involving interactions with various mobile applications, including messaging and banking applications, challenging agents with managing risks encompassing misuse and negative side effects. These tasks include tests to evaluate the safety of agents in daily scenarios as well as their robustness against indirect prompt injection attacks. Our experiments demonstrate that baseline agents, based on state-of-the-art LLMs, often fail to effectively prevent harm while performing the tasks. To mitigate these safety concerns, we propose a prompting method that encourages agents to prioritize safety considerations. While this method shows promise in promoting safer behaviors, there is still considerable room for improvement to fully earn user trust. This highlights the urgent need for continued research to develop more robust safety mechanisms in mobile environments.
>
---
#### [replaced 041] Out of Style: RAG's Fragility to Linguistic Variation
- **分类: cs.CL**

- **简介: 该论文研究RAG系统在处理语言变异时的脆弱性，针对信息检索问答任务，分析不同语言维度对系统性能的影响。**

- **链接: [https://arxiv.org/pdf/2504.08231v2](https://arxiv.org/pdf/2504.08231v2)**

> **作者:** Tianyu Cao; Neel Bhandari; Akhila Yerukola; Akari Asai; Maarten Sap
>
> **备注:** Accepted to EACL 2026 (Main Conference)
>
> **摘要:** Despite the impressive performance of Retrieval-augmented Generation (RAG) systems across various NLP benchmarks, their robustness in handling real-world user-LLM interaction queries remains largely underexplored. This presents a critical gap for practical deployment, where user queries exhibit greater linguistic variations and can trigger cascading errors across interdependent RAG components. In this work, we systematically analyze how varying four linguistic dimensions (formality, readability, politeness, and grammatical correctness) impact RAG performance. We evaluate two retrieval models and nine LLMs, ranging from 3 to 72 billion parameters, across four information-seeking Question Answering (QA) datasets. Our results reveal that linguistic reformulations significantly impact both retrieval and generation stages, leading to a relative performance drop of up to 40.41% in Recall@5 scores for less formal queries and 38.86% in answer match scores for queries containing grammatical errors. Notably, RAG systems exhibit greater sensitivity to such variations compared to LLM-only generations, highlighting their vulnerability to error propagation due to linguistic shifts. These findings highlight the need for improved robustness techniques to enhance reliability in diverse user interactions. Code is available at https://github.com/Springcty/RAG-fragility-to-linguistic-variation.
>
---
#### [replaced 042] ExPO: Unlocking Hard Reasoning with Self-Explanation-Guided Reinforcement Learning
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出ExPO框架，解决复杂推理任务中强化学习效果不佳的问题。通过生成高质量正样本，引导模型探索新推理路径，提升推理能力。属于机器学习中的强化学习与推理任务。**

- **链接: [https://arxiv.org/pdf/2507.02834v3](https://arxiv.org/pdf/2507.02834v3)**

> **作者:** Ruiyang Zhou; Shuozhe Li; Amy Zhang; Liu Leqi
>
> **备注:** Accepted to NeurIPS 2025 (Poster). Code available at https://github.com/HumainLab/ExPO_rl_reasoning_by_explanation
>
> **摘要:** Self-improvement via RL often fails on complex reasoning tasks because GRPO-style post-training methods rely on the model's initial ability to generate positive samples. Without guided exploration, these approaches merely reinforce what the model already knows (distribution-sharpening) rather than enabling the model to solve problems where it initially generates no correct solutions. To unlock reasoning ability in such settings, the model must explore new reasoning trajectories beyond its current output distribution. Such exploration requires access to sufficiently good positive samples to guide the learning. While expert demonstrations seem like a natural solution, we find that they are often ineffective in RL post-training. Instead, we identify two key properties of effective positive samples: they should (1) be likely under the current policy, and (2) increase the model's likelihood of predicting the correct answer. Based on these insights, we propose $\textbf{Self-Explanation Policy Optimization (ExPO)}$-a simple and modular framework that generates such samples by conditioning on the ground-truth answer. It can be integrated with popular RL training methods like GRPO and DPO. ExPO enables efficient exploration and guides the model to produce reasoning trajectories more aligned with its policy than expert-written CoTs, while ensuring higher quality than its own (incorrect) samples. Experiments show that ExPO improves both learning efficiency and final performance on reasoning benchmarks, surpassing expert-demonstration-based methods in challenging settings such as MATH level-5, where the model initially struggles the most. Code is available at https://github.com/HumainLab/ExPO_rl_reasoning_by_explanation .
>
---
#### [replaced 043] SIPDO: Closed-Loop Prompt Optimization via Synthetic Data Feedback
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出SIPDO框架，解决提示优化问题，通过合成数据反馈实现提示的迭代改进。**

- **链接: [https://arxiv.org/pdf/2505.19514v4](https://arxiv.org/pdf/2505.19514v4)**

> **作者:** Yaoning Yu; Ye Yu; Peiyan Zhang; Kai Wei; Haojing Luo; Haohan Wang
>
> **摘要:** Prompt quality plays a critical role in the performance of large language models (LLMs), motivating a growing body of work on prompt optimization. Most existing methods optimize prompts over a fixed dataset, assuming static input distributions and offering limited support for iterative improvement. We introduce SIPDO (Self-Improving Prompts through Data-Augmented Optimization), a closed-loop framework for prompt learning that integrates synthetic data generation into the optimization process. SIPDO couples a synthetic data generator with a prompt optimizer, where the generator produces new examples that reveal current prompt weaknesses and the optimizer incrementally refines the prompt in response. This feedback-driven loop enables systematic improvement of prompt performance without assuming access to external supervision or new tasks. Experiments across question answering and reasoning benchmarks show that SIPDO outperforms standard prompt tuning methods, highlighting the value of integrating data synthesis into prompt learning workflows.
>
---
#### [replaced 044] QWHA: Quantization-Aware Walsh-Hadamard Adaptation for Parameter-Efficient Fine-Tuning on Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于大语言模型的高效微调任务，旨在解决量化误差和训练效率问题。提出QWHA方法，结合傅里叶变换与量化，提升模型精度与速度。**

- **链接: [https://arxiv.org/pdf/2509.17428v4](https://arxiv.org/pdf/2509.17428v4)**

> **作者:** Hyesung Jeon; Seojune Lee; Beomseok Kang; Yulhwa Kim; Jae-Joon Kim
>
> **备注:** 25 pages, 9 figures, 14 tables
>
> **摘要:** The demand for efficient deployment of large language models (LLMs) has driven interest in quantization, which reduces inference cost, and parameter-efficient fine-tuning (PEFT), which lowers training overhead. This motivated the development of quantization-aware PEFT to produce accurate yet efficient quantized models. In this setting, reducing quantization error prior to fine-tuning is crucial for achieving high model accuracy. However, existing methods that rely on low-rank adaptation suffer from limited representational capacity. Recent Fourier-related transform (FT)-based adapters offer greater representational power than low-rank adapters, but their direct integration into quantized models often results in ineffective error reduction and increased computational overhead. To overcome these limitations, we propose QWHA, a method that integrates FT-based adapters into quantized models by employing the Walsh-Hadamard Transform (WHT) as the transform kernel, together with a novel adapter initialization scheme incorporating adaptive parameter selection and value refinement. We demonstrate that QWHA effectively mitigates quantization errors while facilitating fine-tuning, and that its design substantially reduces computational cost. Experimental results show that QWHA consistently outperforms baselines in low-bit quantization accuracy and achieves significant training speedups over existing FT-based adapters. The code is available at https://github.com/vantaa89/qwha.
>
---
#### [replaced 045] Dancing in Chains: Strategic Persuasion in Academic Rebuttal via Theory of Mind
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于学术反驳任务，旨在解决AI在反驳中缺乏视角理解的问题。提出RebuttalAgent框架，通过理论心智模型实现有效说服。**

- **链接: [https://arxiv.org/pdf/2601.15715v2](https://arxiv.org/pdf/2601.15715v2)**

> **作者:** Zhitao He; Zongwei Lyu; Yi R Fung
>
> **备注:** Accepted by ICLR 2026, 36 pages
>
> **摘要:** Although artificial intelligence (AI) has become deeply integrated into various stages of the research workflow and achieved remarkable advancements, academic rebuttal remains a significant and underexplored challenge. This is because rebuttal is a complex process of strategic communication under severe information asymmetry rather than a simple technical debate. Consequently, current approaches struggle as they largely imitate surface-level linguistics, missing the essential element of perspective-taking required for effective persuasion. In this paper, we introduce RebuttalAgent, the first framework to ground academic rebuttal in Theory of Mind (ToM), operationalized through a ToM-Strategy-Response (TSR) pipeline that models reviewer mental state, formulates persuasion strategy, and generates strategy-grounded response. To train our agent, we construct RebuttalBench, a large-scale dataset synthesized via a novel critique-and-refine approach. Our training process consists of two stages, beginning with a supervised fine-tuning phase to equip the agent with ToM-based analysis and strategic planning capabilities, followed by a reinforcement learning phase leveraging the self-reward mechanism for scalable self-improvement. For reliable and efficient automated evaluation, we further develop Rebuttal-RM, a specialized evaluator trained on over 100K samples of multi-source rebuttal data, which achieves scoring consistency with human preferences surpassing powerful judge GPT-4.1. Extensive experiments show RebuttalAgent significantly outperforms the base model by an average of 18.3% on automated metrics, while also outperforming advanced proprietary models across both automated and human evaluations. Disclaimer: the generated rebuttal content is for reference only to inspire authors and assist in drafting. It is not intended to replace the author's own critical analysis and response.
>
---
#### [replaced 046] LangForce: Bayesian Decomposition of Vision Language Action Models via Latent Action Queries
- **分类: cs.AI; cs.CL; cs.CV; cs.RO**

- **简介: 该论文属于视觉-语言-动作模型任务，解决模型在新指令和复杂任务中泛化能力差的问题。通过贝叶斯分解和潜在动作查询，提升语言引导的行动效果。**

- **链接: [https://arxiv.org/pdf/2601.15197v4](https://arxiv.org/pdf/2601.15197v4)**

> **作者:** Shijie Lian; Bin Yu; Xiaopeng Lin; Laurence T. Yang; Zhaolong Shen; Changti Wu; Yuzhuo Miao; Cong Huang; Kai Chen
>
> **摘要:** Vision-Language-Action (VLA) models have shown promise in robot manipulation but often struggle to generalize to new instructions or complex multi-task scenarios. We identify a critical pathology in current training paradigms where goal-driven data collection creates a dataset bias. In such datasets, language instructions are highly predictable from visual observations alone, causing the conditional mutual information between instructions and actions to vanish, a phenomenon we term Information Collapse. Consequently, models degenerate into vision-only policies that ignore language constraints and fail in out-of-distribution (OOD) settings. To address this, we propose LangForce, a novel framework that enforces instruction following via Bayesian decomposition. By introducing learnable Latent Action Queries, we construct a dual-branch architecture to estimate both a vision-only prior $p(a \mid v)$ and a language-conditioned posterior $π(a \mid v, \ell)$. We then optimize the policy to maximize the conditional Pointwise Mutual Information (PMI) between actions and instructions. This objective effectively penalizes the vision shortcut and rewards actions that explicitly explain the language command. Without requiring new data, LangForce significantly improves generalization. Extensive experiments across on SimplerEnv and RoboCasa demonstrate substantial gains, including an 11.3% improvement on the challenging OOD SimplerEnv benchmark, validating the ability of our approach to robustly ground language in action.
>
---
#### [replaced 047] CAMEO: Collection of Multilingual Emotional Speech Corpora
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文介绍CAMEO，一个用于情感识别的多语言语音数据集。旨在解决跨语言和情绪状态的语音情感识别问题，通过标准化数据集促进研究与模型评估。**

- **链接: [https://arxiv.org/pdf/2505.11051v3](https://arxiv.org/pdf/2505.11051v3)**

> **作者:** Iwona Christop; Maciej Czajka
>
> **备注:** Accepted for ICASSP 2026
>
> **摘要:** This paper presents CAMEO -- a curated collection of multilingual emotional speech datasets designed to facilitate research in emotion recognition and other speech-related tasks. The main objectives were to ensure easy access to the data, to allow reproducibility of the results, and to provide a standardized benchmark for evaluating speech emotion recognition (SER) systems across different emotional states and languages. The paper describes the dataset selection criteria, the curation and normalization process, and provides performance results for several models. The collection, along with metadata, and a leaderboard, is publicly available via the Hugging Face platform.
>
---
#### [replaced 048] Epistemological Bias As a Means for the Automated Detection of Injustices in Text
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本中的隐性偏见检测任务，旨在解决自动化识别文本中隐性不公的问题。通过结合认识论知识与NLP模型，提出一个可解释的检测框架。**

- **链接: [https://arxiv.org/pdf/2407.06098v2](https://arxiv.org/pdf/2407.06098v2)**

> **作者:** Kenya Andrews; Lamogha Chiazor
>
> **摘要:** Injustices in text are often subtle since implicit biases or stereotypes frequently operate unconsciously due to the pervasive nature of prejudice in society. This makes automated detection of injustices more challenging which leads to them being often overlooked. We introduce a novel framework that combines knowledge from epistemology to enhance the detection of implicit injustices in text using NLP models to address these complexities and offer explainability. Our empirical study shows how our framework can be applied to effectively detect these injustices. We validate our framework using a human baseline study which mostly agrees with the choice of implicit bias, stereotype, and sentiment. The main feedback from the study was the extended time required to analyze, digest, and decide on each component of our framework. This highlights the importance of our automated framework pipeline that assists users in detecting implicit injustices while offering explainability and reducing time burdens on humans.
>
---
#### [replaced 049] LLM-Specific Utility: A New Perspective for Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于信息检索任务，旨在解决RAG中证据实用性问题。研究发现证据实用性具有模型特异性，提出基准并验证不同模型的最佳证据不同。**

- **链接: [https://arxiv.org/pdf/2510.11358v2](https://arxiv.org/pdf/2510.11358v2)**

> **作者:** Hengran Zhang; Keping Bi; Jiafeng Guo; Jiaming Zhang; Shuaiqiang Wang; Dawei Yin; Xueqi Cheng
>
> **备注:** 13 pages, 9 figures
>
> **摘要:** Retrieval-augmented generation (RAG) is typically optimized for topical relevance, yet its success ultimately depends on whether retrieved passages are useful for a large language model (LLM) to generate correct and complete answers. We argue that such utility is often LLM-specific rather than universal, due to differences in models' knowledge, reasoning, and ability to leverage evidence. We formalize LLM-specific utility as the performance improvement of a target LLM when a passage is provided, compared to answering without evidence. To systematically study LLM-specific utility, we construct a benchmark of LLM-specific gold utilitarian passages for four LLMs (Qwen3-8B/14B/32B and Llama3.1-8B) on three QA datasets (Natural Questions, TriviaQA, and MS MARCO-FQA). Our analysis shows that utilitarian passages are model-dependent and non-transferable: each LLM performs best with its own utilitarian evidence, while evidence optimized for other LLMs is consistently suboptimal. Human-annotated evidence remains a strong general baseline but does not fully match individual LLM utility needs. We further introduce the LLM-specific utility judgment task and find that existing utility-aware selection and scoring methods largely capture model-agnostic usefulness and struggle to reliably estimate LLM-specific utility. Overall, our findings highlight the limitations of current utility-aware retrieval and motivate generator-tailored evidence selection for improving RAG.
>
---
#### [replaced 050] Do Psychometric Tests Work for Large Language Models? Evaluation of Tests on Sexism, Racism, and Morality
- **分类: cs.CL**

- **简介: 该论文属于评估任务，探讨心理测量测试在大语言模型中的适用性。研究解决心理测试是否适用于LLMs的问题，通过测试17个模型的性别歧视、种族主义和道德性，发现测试结果与实际行为不一致。**

- **链接: [https://arxiv.org/pdf/2510.11254v2](https://arxiv.org/pdf/2510.11254v2)**

> **作者:** Jana Jung; Marlene Lutz; Indira Sen; Markus Strohmaier
>
> **摘要:** Psychometric tests are increasingly used to assess psychological constructs in large language models (LLMs). However, it remains unclear whether these tests -- originally developed for humans -- yield meaningful results when applied to LLMs. In this study, we systematically evaluate the reliability and validity of human psychometric tests on 17 LLMs for three constructs: sexism, racism, and morality. We find moderate reliability across multiple item and prompt variations. Validity is evaluated through both convergent (i.e., testing theory-based inter-test correlations) and ecological approaches (i.e., testing the alignment between tests scores and behavior in real-world downstream tasks). Crucially, we find that psychometric test scores do not align, and in some cases even negatively correlate with, model behavior in downstream tasks, indicating low ecological validity. Our results highlight that systematic evaluations of psychometric tests on LLMs are essential before interpreting their scores. Our findings also suggest that psychometric tests designed for humans cannot be applied directly to LLMs without adaptation.
>
---
#### [replaced 051] Unsupervised Elicitation of Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型训练任务，旨在解决缺乏高质量人类监督的问题。提出无监督算法ICM，通过模型自动生成标签进行微调，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2506.10139v2](https://arxiv.org/pdf/2506.10139v2)**

> **作者:** Jiaxin Wen; Zachary Ankner; Arushi Somani; Peter Hase; Samuel Marks; Jacob Goldman-Wetzler; Linda Petrini; Henry Sleight; Collin Burns; He He; Shi Feng; Ethan Perez; Jan Leike
>
> **摘要:** To steer pretrained language models for downstream tasks, today's post-training paradigm relies on humans to specify desired behaviors. However, for models with superhuman capabilities, it is difficult or impossible to get high-quality human supervision. To address this challenge, we introduce a new unsupervised algorithm, Internal Coherence Maximization (ICM), to fine-tune pretrained language models on their own generated labels, \emph{without external supervision}. On GSM8k-verification, TruthfulQA, and Alpaca reward modeling tasks, our method matches the performance of training on golden labels and outperforms training on crowdsourced human supervision. On tasks where LMs' capabilities are strongly superhuman, our method can elicit those capabilities significantly better than training on human labels. Finally, we show that our method can improve the training of frontier LMs: we use our method to train an unsupervised reward model and use reinforcement learning to train a Claude 4 Sonnet-based assistant. The resulting assistant matches its counterpart trained on production-grade human labels on average, with higher scores on chat and safety yet lower scores on math and coding.
>
---
#### [replaced 052] Unsupervised lexicon learning from speech is limited by representations rather than clustering
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于无监督词分割与聚类任务，研究语音中词典学习的限制因素。通过对比不同表示和聚类方法，发现词内表示差异是性能瓶颈。**

- **链接: [https://arxiv.org/pdf/2510.09225v2](https://arxiv.org/pdf/2510.09225v2)**

> **作者:** Danel Slabbert; Simon Malan; Herman Kamper
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** Zero-resource word segmentation and clustering systems aim to tokenise speech into word-like units without access to text labels. Despite progress, the induced lexicons are still far from perfect. In an idealised setting with gold word boundaries, we ask whether performance is limited by the representation of word segments, or by the clustering methods that group them into word-like types. We combine a range of self-supervised speech features (continuous/discrete, frame/word-level) with different clustering methods (K-means, hierarchical, graph-based) on English and Mandarin data. The best system uses graph clustering with dynamic time warping on continuous features. Faster alternatives use graph clustering with cosine distance on averaged continuous features or edit distance on discrete unit sequences. Through controlled experiments that isolate either the representations or the clustering method, we demonstrate that representation variability across segments of the same word type -- rather than clustering -- is the primary factor limiting performance.
>
---
#### [replaced 053] High-Layer Attention Pruning with Rescaling
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于模型压缩任务，旨在解决传统剪枝方法盲目移除注意力头的问题。提出一种针对高层注意力头的剪枝算法，并引入自适应重标参数以保持表征精度。**

- **链接: [https://arxiv.org/pdf/2507.01900v2](https://arxiv.org/pdf/2507.01900v2)**

> **作者:** Songtao Liu; Peng Liu
>
> **备注:** TMLR 2026
>
> **摘要:** Pruning is a highly effective approach for compressing large language models (LLMs), significantly reducing inference latency. However, conventional training-free structured pruning methods often employ a heuristic metric that indiscriminately removes some attention heads across all pruning layers, without considering their positions within the network architecture. In this work, we propose a novel pruning algorithm that strategically prunes attention heads in the model's higher layers. Since the removal of attention heads can alter the magnitude of token representations, we introduce an adaptive rescaling parameter that calibrates the representation scale post-pruning to counteract this effect. We conduct comprehensive experiments on a wide range of LLMs, including LLaMA3.1-8B, Mistral-7B-v0.3, Qwen2-7B, and Gemma2-9B. Our evaluation includes both generation and discriminative tasks across 27 datasets. The results consistently demonstrate that our method outperforms existing structured pruning methods. This improvement is particularly notable in generation tasks, where our approach significantly outperforms existing baselines. Code is available at https://github.com/SongtaoLiu0823/HARP.
>
---
#### [replaced 054] Streaming-dLLM: Accelerating Diffusion LLMs via Suffix Pruning and Dynamic Decoding
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于自然语言生成任务，旨在解决扩散大语言模型推理效率低的问题。通过空间冗余剪枝和动态解码策略提升速度，保持生成质量。**

- **链接: [https://arxiv.org/pdf/2601.17917v2](https://arxiv.org/pdf/2601.17917v2)**

> **作者:** Zhongyu Xiao; Zhiwei Hao; Jianyuan Guo; Yong Luo; Jia Liu; Jie Xu; Han Hu
>
> **备注:** Tech report. Code is available at https://github.com/xiaoshideta/Streaming-dLLM
>
> **摘要:** Diffusion Large Language Models (dLLMs) offer a compelling paradigm for natural language generation, leveraging parallel decoding and bidirectional attention to achieve superior global coherence compared to autoregressive models. While recent works have accelerated inference via KV cache reuse or heuristic decoding, they overlook the intrinsic inefficiencies within the block-wise diffusion process. Specifically, they suffer from spatial redundancy by modeling informative-sparse suffix regions uniformly and temporal inefficiency by applying fixed denoising schedules across all the decoding process. To address this, we propose Streaming-dLLM, a training-free framework that streamlines inference across both spatial and temporal dimensions. Spatially, we introduce attenuation guided suffix modeling to approximate the full context by pruning redundant mask tokens. Temporally, we employ a dynamic confidence aware strategy with an early exit mechanism, allowing the model to skip unnecessary iterations for converged tokens. Extensive experiments show that Streaming-dLLM achieves up to 68.2X speedup while maintaining generation quality, highlighting its effectiveness in diffusion decoding. The code is available at https://github.com/xiaoshideta/Streaming-dLLM.
>
---
#### [replaced 055] Task-Specific Directions: Definition, Exploration, and Utilization in Parameter Efficient Fine-Tuning
- **分类: cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于自然语言处理任务，解决参数高效微调中的性能提升问题。提出TSD概念及LoRA-Dash和LoRA-Init方法，优化微调效果。**

- **链接: [https://arxiv.org/pdf/2409.01035v5](https://arxiv.org/pdf/2409.01035v5)**

> **作者:** Chongjie Si; Zhiyi Shi; Shifan Zhang; Xiaokang Yang; Hanspeter Pfister; Wei Shen
>
> **备注:** 2026, TPAMI, Codes in https://github.com/Chongjie-Si/Subspace-Tuning
>
> **摘要:** Large language models demonstrate impressive performance on downstream tasks, yet they require extensive resource consumption when fully fine-tuning all parameters. To mitigate this, Parameter Efficient Fine-Tuning (PEFT) strategies, such as LoRA, have been developed. In this paper, we delve into the concept of task-specific directions (TSDs), which are critical for transitioning large models from pretrained states to task-specific enhancements in PEFT. We propose a framework to clearly define these directions and explore their properties and practical utilization challenges. We then introduce a novel approach, LoRA-Dash, which aims to maximize the impact of TSDs during the fine-tuning process, thereby enhancing model performance on targeted tasks. Additionally, based on our exploration of TSD, we focus on an important issue in PEFT: the initialization of LoRA. While some works have pointed out the significance of initialization for LoRA's performance and proposed various strategies, these methods are often empirical and not task-specific. To address this issue, we propose LoRA-Init. Starting from TSD, we identify the directions that require the most adjustment during fine-tuning for downstream tasks. By initializing the matrices in LoRA with these directions, LoRA-Init significantly enhances LoRA's performance. Moreover, we can combine LoRA-Dash and LoRA-Init to create the final version of LoRA based on TSDs, which we refer to as LoRA-TSD. Extensive experiments have conclusively demonstrated the effectiveness of these methods, and in-depth analyses further reveal the underlying mechanisms behind their success.
>
---
#### [replaced 056] Unleashing Scientific Reasoning for Bio-experimental Protocol Generation via Structured Component-based Reward Mechanism
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于科学协议生成任务，解决LLM生成不完整或不一致实验协议的问题。提出SciRecipe数据集和“Sketch-and-Fill”方法，构建Thoth模型提升协议准确性与逻辑性。**

- **链接: [https://arxiv.org/pdf/2510.15600v2](https://arxiv.org/pdf/2510.15600v2)**

> **作者:** Haoran Sun; Yankai Jiang; Zhenyu Tang; Yaning Pan; Shuang Gu; Zekai Lin; Lilong Wang; Wenjie Lou; Lei Liu; Lei Bai; Xiaosong Wang
>
> **摘要:** The foundation of reproducible science lies in protocols that are precise, logically ordered, and executable. The autonomous generation of these protocols through natural language queries could greatly improve the efficiency of the reproduction process. However, current leading large language models (LLMs) often generate incomplete or inconsistent protocols, limiting their utility. To address this limitation, we first introduce SciRecipe, a large-scale dataset of over 12K structured protocols spanning 27 biological subfields and encompassing both comprehension and problem-solving tasks. To further improve protocol generation, we propose the "Sketch-and-Fill" paradigm, which separates analysis, structuring, and expression to ensure each step is explicit and verifiable. Complementing this, the structured component-based reward mechanism evaluates step granularity, action order, and semantic fidelity, aligning model optimization with experimental reliability. Building on these components, we develop Thoth, trained through a staged Knowledge-to-Action process that progresses from knowledge acquisition to operational reasoning and ultimately to robust, executable protocol generation. Across multiple benchmarks, Thoth consistently surpasses both proprietary and open-source LLMs, achieving significant improvements in step alignment, logical sequencing, and semantic accuracy. Our approach paves the way for reliable scientific assistants that bridge knowledge with experimental execution. All data, code, and models will be released publicly.
>
---
#### [replaced 057] The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理领域，研究稀疏注意力在Transformer大模型中的效率与精度权衡问题。通过大规模实验分析，提出有效部署策略和方法建议。**

- **链接: [https://arxiv.org/pdf/2504.17768v2](https://arxiv.org/pdf/2504.17768v2)**

> **作者:** Piotr Nawrot; Robert Li; Renjie Huang; Sebastian Ruder; Kelly Marchisio; Edoardo M. Ponti
>
> **摘要:** Sparse attention offers a promising strategy to extend long-context capabilities in Transformer LLMs, yet its efficiency-accuracy trade-offs remain unclear due to the lack of comprehensive evaluation. We address this gap with the largest-scale empirical analysis to date of training-free sparse attention, evaluating six methods across multiple model families and sizes, sequences up to 128K tokens, and sparsity levels up to 0.95 (i.e., $1/20$ attention budget) on nine diverse tasks. We first organise the rapidly evolving landscape of sparse attention methods into a taxonomy along four design axes. Our analysis then yields actionable insights: 1) sparse attention is effective -- larger sparse models outperform smaller dense ones at equivalent cost, improving the Pareto frontier; 2) due to computational constraints, token-to-page importance estimation is unfeasible during prefilling, where the choice of an alternative solution (global-to-token or block-to-block) depends on the task, but is possible during decoding, enabling better generalisation and tolerance to higher sparsity; 3) longer sequences tolerate higher sparsity, suggesting that fixed-budget methods in production are suboptimal. Together, these findings provide practical guidance for deploying sparse attention and methodological recommendations for future evaluations. Our code is available at https://github.com/PiotrNawrot/sparse-frontier.
>
---
#### [replaced 058] Why is Your Language Model a Poor Implicit Reward Model?
- **分类: cs.CL; cs.AI; cs.LG; stat.ML**

- **简介: 该论文属于自然语言处理任务，研究奖励模型的泛化问题。旨在解释隐式奖励模型（IM-RM）为何泛化能力较差。通过理论与实验分析，发现IM-RM更依赖表面词元线索，导致泛化效果不佳。**

- **链接: [https://arxiv.org/pdf/2507.07981v3](https://arxiv.org/pdf/2507.07981v3)**

> **作者:** Noam Razin; Yong Lin; Jiarui Yao; Sanjeev Arora
>
> **备注:** Accepted to ICLR 2026; Code available at https://github.com/princeton-pli/exrm-vs-imrm
>
> **摘要:** Reward models are key to language model post-training and inference pipelines. Conveniently, recent work showed that every language model defines an implicit reward model (IM-RM), without requiring any architectural changes. However, such IM-RMs tend to generalize worse, especially out-of-distribution, compared to explicit reward models (EX-RMs) that apply a dedicated linear head over the hidden representations of a language model. The existence of a generalization gap is puzzling, as EX-RMs and IM-RMs are nearly identical. They can be trained using the same data, loss function, and language model, and differ only in how the reward is computed. Toward a fundamental understanding of the implicit biases underlying different reward model types, we investigate the root cause of this gap. Our main finding, backed by theory and experiments, is that IM-RMs rely more heavily on superficial token-level cues. Consequently, they often generalize worse than EX-RMs under token-level distribution shifts, as well as in-distribution. Furthermore, we provide evidence against alternative hypotheses for the generalization gap. Most notably, we challenge the claim that IM-RMs struggle in tasks where generation is harder than verification because they can operate both as a verifier and a generator. Overall, our results highlight that seemingly minor design choices can substantially impact the generalization behavior of reward models.
>
---
#### [replaced 059] Rethinking Schema Linking: A Context-Aware Bidirectional Retrieval Approach for Text-to-SQL
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于Text-to-SQL任务，旨在解决schema链接问题。通过双向检索方法提升schema召回率，减少错误，提高SQL生成效果。**

- **链接: [https://arxiv.org/pdf/2510.14296v2](https://arxiv.org/pdf/2510.14296v2)**

> **作者:** Md Mahadi Hasan Nahid; Davood Rafiei; Weiwei Zhang; Yong Zhang
>
> **备注:** EACL 2026 (Findings)
>
> **摘要:** Schema linking -- the process of aligning natural language questions with database schema elements -- is a critical yet underexplored component of Text-to-SQL systems. While recent methods have focused primarily on improving SQL generation, they often neglect the retrieval of relevant schema elements, which can lead to hallucinations and execution failures. In this work, we propose a context-aware bidirectional schema retrieval framework that treats schema linking as a standalone problem. Our approach combines two complementary strategies: table-first retrieval followed by column selection, and column-first retrieval followed by table selection. It is further augmented with techniques such as question decomposition, keyword extraction, and keyphrase extraction. Through comprehensive evaluations on challenging benchmarks such as BIRD and Spider, we demonstrate that our method significantly improves schema recall while reducing false positives. Moreover, SQL generation using our retrieved schema consistently outperforms full-schema baselines and closely approaches oracle performance, all without requiring query refinement. Notably, our method narrows the performance gap between full and perfect schema settings by 50\%. Our findings highlight schema linking as a powerful lever for enhancing Text-to-SQL accuracy and efficiency.
>
---
#### [replaced 060] Propaganda AI: An Analysis of Semantic Divergence in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于AI安全任务，旨在检测大语言模型中的概念引导语义偏差。通过提出RAVEN审计方法，识别模型在特定概念下的异常响应，以防范潜在的宣传影响。**

- **链接: [https://arxiv.org/pdf/2504.12344v2](https://arxiv.org/pdf/2504.12344v2)**

> **作者:** Nay Myat Min; Long H. Pham; Yige Li; Jun Sun
>
> **备注:** Accepted at ICLR 2026, 22 pages, 1 figure
>
> **摘要:** Large language models (LLMs) can exhibit concept-conditioned semantic divergence: common high-level cues (e.g., ideologies, public figures) elicit unusually uniform, stance-like responses that evade token-trigger audits. This behavior falls in a blind spot of current safety evaluations, yet carries major societal stakes, as such concept cues can steer content exposure at scale. We formalize this phenomenon and present RAVEN (Response Anomaly Vigilance), a black-box audit that flags cases where a model is simultaneously highly certain and atypical among peers by coupling semantic entropy over paraphrastic samples with cross-model disagreement. In a controlled LoRA fine-tuning study, we implant a concept-conditioned stance using a small biased corpus, demonstrating feasibility without rare token triggers. Auditing five LLM families across twelve sensitive topics (360 prompts per model) and clustering via bidirectional entailment, RAVEN surfaces recurrent, model-specific divergences in 9/12 topics. Concept-level audits complement token-level defenses and provide a practical early-warning signal for release evaluation and post-deployment monitoring against propaganda-like influence.
>
---
#### [replaced 061] Analogical Structure, Minimal Contextual Cues and Contrastive Distractors: Input Design for Sample-Efficient Linguistic Rule Induction
- **分类: cs.CL**

- **简介: 该论文研究输入设计对语言规则学习的影响，旨在提升样本效率。通过结构化句子补全任务，验证了类比结构、对比干扰和最小上下文的作用，发现轻量模型在少量数据下表现优于大模型。**

- **链接: [https://arxiv.org/pdf/2511.10441v2](https://arxiv.org/pdf/2511.10441v2)**

> **作者:** Chunyang Jiang; Paola Merlo
>
> **备注:** Accepted by EACL 2026 main conference
>
> **摘要:** Large language models achieve strong performance on many tasks, but their training makes it hard to see which properties of the input support efficient linguistic rule learning. We ask how three cognitively-inspired principles of input design support sample-efficient linguistic rule induction: analogical structure, contrastive learning, and minimal contextual cue. We also ask how their effects compare to those of LLMs on the same controlled tasks. We implement these principles in structured sentence completion tasks that test English verb alternations. Lightweight models trained on hundreds to one-thousand such examples learn the alternation rules with high F1 on these tasks. Ablation studies show that analogical organisation is the main driver of sample efficiency, and contrastive distractors and minimal context help further gains. We also evaluate zero- and few-shot LLMs on the same tasks. In this controlled setting, the lightweight models reach higher F1 with far fewer task-specific data. We treat this contrast as a comparison between learning regimes rather than a general verdict on LLMs. Our results show that careful input organisation supports sample-efficient learning of linguistic rules and reveals distinct learning signatures for trained lightweight models and prompted LLMs.
>
---
#### [replaced 062] Accepted with Minor Revisions: Value of AI-Assisted Scientific Writing
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文研究AI在科学写作中的辅助作用，旨在解决AI生成文本与人类写作的接受度差异问题。通过实验分析作者编辑行为及评审决策，发现源信息披露影响编辑策略，AI文本经少量修改可达到可接受水平。**

- **链接: [https://arxiv.org/pdf/2511.12529v2](https://arxiv.org/pdf/2511.12529v2)**

> **作者:** Sanchaita Hazra; Doeun Lee; Bodhisattwa Prasad Majumder; Sachin Kumar
>
> **备注:** Published in ACM IUI 2026 (Paphos, Cyprus)
>
> **摘要:** Large Language Models have seen expanding application across domains, yet their effectiveness as assistive tools for scientific writing - an endeavor requiring precision, multimodal synthesis, and domain expertise - remains insufficiently understood. We examine the potential of LLMs to support domain experts in scientific writing, with a focus on abstract composition. We design an incentivized randomized controlled trial with a hypothetical conference setup where participants with relevant expertise are split into an author and reviewer pool. Inspired by methods in behavioral science, our novel incentive structure encourages authors to edit the provided abstracts to an acceptable quality for a peer-reviewed submission. Our 2 x 2 between-subject design expands into two dimensions: the implicit source of the provided abstract and the disclosure of it. We find authors make most edits when editing human-written abstracts compared to AI-generated abstracts without source attribution, often guided by higher perceived readability in AI generation. Upon disclosure of source information, the volume of edits converges in both source treatments. Reviewer decisions remain unaffected by the source of the abstract, but bear a significant correlation with the number of edits made. Careful stylistic edits, especially in the case of AI-generated abstracts, in the presence of source information, improve the chance of acceptance. We find that AI-generated abstracts hold potential to reach comparable levels of acceptability to human-written ones with minimal revision, and that perceptions of AI authorship, rather than objective quality, drive much of the observed editing behavior. Our findings reverberate the significance of source disclosure in collaborative scientific writing.
>
---
#### [replaced 063] LingoQ: Bridging the Gap between EFL Learning and Work through AI-Generated Work-Related Quizzes
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文提出LingoQ系统，解决EFL学习者在工作中难以持续学习英语的问题。通过AI生成工作相关测验，帮助用户在手机上练习英语，提升学习效果。**

- **链接: [https://arxiv.org/pdf/2509.17477v2](https://arxiv.org/pdf/2509.17477v2)**

> **作者:** Yeonsun Yang; Sang Won Lee; Jean Y. Song; Sangdoo Yun; Young-Ho Kim
>
> **备注:** 18 pages except reference. Conditionally accepted to ACM CHI 2026
>
> **摘要:** Non-native English speakers performing English-related tasks at work struggle to sustain EFL learning, despite their motivation. Often, study materials are disconnected from their work context. Our formative study revealed that reviewing work-related English becomes burdensome with current systems, especially after work. Although workers rely on LLM-based assistants to address their immediate needs, these interactions may not directly contribute to their English skills. We present LingoQ, an AI-mediated system that allows workers to practice English using quizzes generated from their LLM queries during work. LingoQ leverages these on-the-fly queries using AI to generate personalized quizzes that workers can review and practice on their smartphones. We conducted a three-week deployment study with 28 EFL workers to evaluate LingoQ. Participants valued the quality-assured, work-situated quizzes and constantly engaging with the app during the study. This active engagement improved self-efficacy and led to learning gains for beginners and, potentially, for intermediate learners. Drawing on these results, we discuss design implications for leveraging workers' growing reliance on LLMs to foster proficiency and engagement while respecting work boundaries and ethics.
>
---
#### [replaced 064] APEX-Agents
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出APEX-Agents基准，用于评估AI代理执行复杂跨应用任务的能力，解决AI在现实工作环境中的生产力评估问题。**

- **链接: [https://arxiv.org/pdf/2601.14242v2](https://arxiv.org/pdf/2601.14242v2)**

> **作者:** Bertie Vidgen; Austin Mann; Abby Fennelly; John Wright Stanly; Lucas Rothman; Marco Burstein; Julien Benchek; David Ostrofsky; Anirudh Ravichandran; Debnil Sur; Neel Venugopal; Alannah Hsia; Isaac Robinson; Calix Huang; Olivia Varones; Daniyal Khan; Michael Haines; Zach Richards; Chirag Mahapatra; Brendan Foody; Osvald Nitski
>
> **摘要:** We introduce the AI Productivity Index for Agents (APEX-Agents), a benchmark for assessing whether AI agents can execute long-horizon, cross-application tasks created by investment banking analysts, management consultants, and corporate lawyers. APEX-Agents requires agents to navigate realistic work environments with files and tools. We test eight agents for the leaderboard using Pass@1. Gemini 3 Flash (Thinking=High) achieves the highest score of 24.0%, followed by GPT-5.2 (Thinking=High), Claude Opus 4.5 (Thinking=High), and Gemini 3 Pro (Thinking=High). We open source the APEX-Agents benchmark (n=480) with all prompts, rubrics, gold outputs, files, and metadata. We also open-source Archipelago, our infrastructure for agent execution and evaluation.
>
---
#### [replaced 065] Watermark-based Attribution of AI-Generated Content
- **分类: cs.CR; cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于AI生成内容的溯源任务，旨在解决用户级归属问题。通过为每个用户分配唯一水印，实现对AI生成内容的准确溯源。**

- **链接: [https://arxiv.org/pdf/2404.04254v4](https://arxiv.org/pdf/2404.04254v4)**

> **作者:** Zhengyuan Jiang; Moyang Guo; Yuepeng Hu; Yupu Wang; Neil Zhenqiang Gong
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Several companies have deployed watermark-based detection to identify AI-generated content. However, attribution--the ability to trace back to the user of a generative AI (GenAI) service who created a given AI-generated content--remains largely unexplored despite its growing importance. In this work, we aim to bridge this gap by conducting the first systematic study on watermark-based, user-level attribution of AI-generated content. Our key idea is to assign a unique watermark to each user of the GenAI service and embed this watermark into the AI-generated content created by that user. Attribution is then performed by identifying the user whose watermark best matches the one extracted from the given content. This approach, however, faces a key challenge: How should watermarks be selected for users to maximize attribution performance? To address the challenge, we first theoretically derive lower bounds on detection and attribution performance through rigorous probabilistic analysis for any given set of user watermarks. Then, we select watermarks for users to maximize these lower bounds, thereby optimizing detection and attribution performance. Our theoretical and empirical results show that watermark-based attribution inherits both the accuracy and (non-)robustness properties of the underlying watermark. Specifically, attribution remains highly accurate when the watermarked AI-generated content is either not post-processed or subjected to common post-processing such as JPEG compression, as well as black-box adversarial post-processing with limited query budgets.
>
---
#### [replaced 066] What Does Neuro Mean to Cardio? Investigating the Role of Clinical Specialty Data in Medical LLMs
- **分类: cs.CL**

- **简介: 该论文属于医疗自然语言处理任务，旨在探讨临床专科数据在医学大模型中的作用。通过构建S-MedQA数据集，研究发现模型性能提升主要源于领域迁移而非专科知识注入。**

- **链接: [https://arxiv.org/pdf/2505.10113v4](https://arxiv.org/pdf/2505.10113v4)**

> **作者:** Xinlan Yan; Di Wu; Yibin Lei; Christof Monz; Iacer Calixto
>
> **摘要:** In this paper, we introduce S-MedQA, an English medical question-answering (QA) dataset designed for benchmarking large language models (LLMs) in fine-grained clinical specialties. S-MedQA consists of over 24k examples, covering 15 medical specialties, with QA pairs that can have multiple specialty annotations, such as when a question is cross-disciplinary. The dataset is constructed using both machine and expert verification to maximize data availability and reliability. We use S-MedQA to investigate the role of clinical specialties in the knowledge-intensive scenario of medical QA. Our results show that training on data from a clinical specialty does not necessarily lead to the best performance on that specialty. Additionally, regardless of the specialty the LLM was fine-tuned on, token probabilities of clinically relevant terms consistently increase across all specialties. Based on these findings, we hypothesize that improvement gains, at least in our settings, are derived primarily from domain shifting (e.g., general to medical) rather than from injecting specialty-specific knowledge. This suggests a need to rethink the role of fine-tuning data in the medical domain. To encourage further advancements in the clinical NLP field, we release S-MedQA along with all the code required to reproduce our experiments for the research community.
>
---
#### [replaced 067] TextMineX: Data, Evaluation Framework and Ontology-guided LLM Pipeline for Humanitarian Mine Action
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出TextMineX，用于人道主义排雷领域的知识提取任务，解决信息分散于非结构化报告中的问题，通过构建数据集、评估框架和基于本体的LLM流程实现知识结构化。**

- **链接: [https://arxiv.org/pdf/2509.15098v4](https://arxiv.org/pdf/2509.15098v4)**

> **作者:** Chenyue Zhou; Gürkan Solmaz; Flavio Cirillo; Kiril Gashteovski; Jonathan Fürst
>
> **摘要:** Humanitarian Mine Action (HMA) addresses the challenge of detecting and removing landmines from conflict regions. Much of the life-saving operational knowledge produced by HMA agencies is buried in unstructured reports, limiting the transferability of information between agencies. To address this issue, we propose TextMineX: the first dataset, evaluation framework and ontology-guided large language model (LLM) pipeline for knowledge extraction from text in the HMA domain. TextMineX structures HMA reports into (subject, relation, object)-triples, thus creating domain-specific knowledge. To ensure real-world relevance, we utilized the dataset from our collaborator Cambodian Mine Action Centre (CMAC). We further introduce a bias-aware evaluation framework that combines human-annotated triples with an LLM-as-Judge protocol to mitigate position bias in reference-free scoring. Our experiments show that ontology-aligned prompts improve extraction accuracy by up to 44.2%, reduce hallucinations by 22.5%, and enhance format adherence by 20.9% compared to baseline models. We publicly release the dataset and code.
>
---
#### [replaced 068] SAC-Opt: Semantic Anchors for Iterative Correction in Optimization Modeling
- **分类: cs.AI; cs.CL; cs.PL**

- **简介: 该论文属于优化建模任务，解决LLM生成代码中的语义错误问题。提出SAC-Opt框架，通过语义锚点迭代修正，提升模型准确性。**

- **链接: [https://arxiv.org/pdf/2510.05115v2](https://arxiv.org/pdf/2510.05115v2)**

> **作者:** Yansen Zhang; Qingcan Kang; Yujie Chen; Yufei Wang; Xiongwei Han; Tao Zhong; Mingxuan Yuan; Chen Ma
>
> **摘要:** Large language models (LLMs) have opened new paradigms in optimization modeling by enabling the generation of executable solver code from natural language descriptions. Despite this promise, existing approaches typically remain solver-driven: they rely on single-pass forward generation and apply limited post-hoc fixes based on solver error messages, leaving undetected semantic errors that silently produce syntactically correct but logically flawed models. To address this challenge, we propose SAC-Opt, a backward-guided correction framework that grounds optimization modeling in problem semantics rather than solver feedback. At each step, SAC-Opt aligns the original semantic anchors with those reconstructed from the generated code and selectively corrects only the mismatched components, driving convergence toward a semantically faithful model. This anchor-driven correction enables fine-grained refinement of constraint and objective logic, enhancing both fidelity and robustness without requiring additional training or supervision. Empirical results on seven public datasets demonstrate that SAC-Opt improves average modeling accuracy by 7.7%, with gains of up to 21.9% on the ComplexLP dataset. These findings highlight the importance of semantic-anchored correction in LLM-based optimization workflows to ensure faithful translation from problem intent to solver-executable code.
>
---
#### [replaced 069] CCFQA: A Benchmark for Cross-Lingual and Cross-Modal Speech and Text Factuality Evaluation
- **分类: cs.CL**

- **简介: 该论文属于多模态语言模型的事实性评估任务，旨在解决跨语言和跨模态的可靠性问题。提出CCFQA基准，包含多语言语音-文本事实性问题，验证模型能力并提出迁移学习策略提升性能。**

- **链接: [https://arxiv.org/pdf/2508.07295v3](https://arxiv.org/pdf/2508.07295v3)**

> **作者:** Yexing Du; Kaiyuan Liu; Youcheng Pan; Zheng Chu; Bo Yang; Xiaocheng Feng; Ming Liu; Yang Xiang
>
> **备注:** Accepted in AAAI 2026
>
> **摘要:** As Large Language Models (LLMs) are increasingly popularized in the multilingual world, ensuring hallucination-free factuality becomes markedly crucial. However, existing benchmarks for evaluating the reliability of Multimodal Large Language Models (MLLMs) predominantly focus on textual or visual modalities with a primary emphasis on English, which creates a gap in evaluation when processing multilingual input, especially in speech. To bridge this gap, we propose a novel Cross-lingual and Cross-modal Factuality benchmark (CCFQA). Specifically, the CCFQA benchmark contains parallel speech-text factual questions across 8 languages, designed to systematically evaluate MLLMs' cross-lingual and cross-modal factuality capabilities. Our experimental results demonstrate that current MLLMs still face substantial challenges on the CCFQA benchmark. Furthermore, we propose a few-shot transfer learning strategy that effectively transfers the Question Answering (QA) capabilities of LLMs in English to multilingual Spoken Question Answering (SQA) tasks, achieving competitive performance with GPT-4o-mini-Audio using just 5-shot training. We release CCFQA as a foundational research resource to promote the development of MLLMs with more robust and reliable speech understanding capabilities. Our code and dataset are available at https://github.com/yxduir/ccfqa.
>
---
#### [replaced 070] Truth with a Twist: The Rhetoric of Persuasion in Professional vs. Community-Authored Fact-Checks
- **分类: cs.CL**

- **简介: 该论文属于事实核查研究任务，旨在比较专业与社区撰写辟谣内容中的说服技巧。通过分析多个数据集，发现社区内容并不更依赖说服性语言，且用户能有效识别不当修辞。**

- **链接: [https://arxiv.org/pdf/2601.14105v2](https://arxiv.org/pdf/2601.14105v2)**

> **作者:** Olesya Razuvayevskaya; Kalina Bontcheva
>
> **备注:** In Proceedings of the ACM Web Conference 2026 (WWW 2026)
>
> **摘要:** This study presents the first large-scale comparison of persuasion techniques present in crowd- versus professionally-written debunks. Using extensive datasets from Community Notes (CNs), EUvsDisinfo, and the Database of Known Fakes (DBKF), we quantify the prevalence and types of persuasion techniques across these fact-checking ecosystems. Contrary to prior hypothesis that community-produced debunks rely more heavily on subjective or persuasive wording, we find no evidence that CNs contain a higher average number of persuasion techniques than professional fact-checks. We additionally identify systematic rhetorical differences between CNs and professional debunking efforts, reflecting differences in institutional norms and topical coverage. Finally, we examine how the crowd evaluates persuasive language in CNs and show that, although notes with more persuasive elements receive slightly higher overall helpfulness ratings, crowd raters are effective at penalising the use of particular problematic rhetorical means
>
---
#### [replaced 071] Meaning Is Not A Metric: Using LLMs to make cultural context legible at scale
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 论文探讨如何利用大语言模型实现文化语境的可读性，解决AI系统中人类意义表达不足的问题。通过厚描述框架，提出五项关键挑战。属于AI与人文交叉任务。**

- **链接: [https://arxiv.org/pdf/2505.23785v2](https://arxiv.org/pdf/2505.23785v2)**

> **作者:** Cody Kommers; Drew Hemment; Maria Antoniak; Joel Z. Leibo; Hoyt Long; Emily Robinson; Adam Sobey
>
> **摘要:** This position paper argues that large language models (LLMs) can make cultural context, and therefore human meaning, legible at an unprecedented scale in AI-based sociotechnical systems. We argue that such systems have previously been unable to represent human meaning because they rely on thin descriptions (numerical representations that enforce standardization and therefore strip human activity of the cultural context which gives it meaning). By contrast, scholars in the humanities and qualitative social sciences have developed frameworks for representing meaning through thick description (verbal representations that accommodate heterogeneity and retain contextual information needed to represent human meaning). The verbal capabilities of LLMs now provide a means of at least partially automating the generation and processing of thick descriptions, offering new ways to deploy them at scale. We argue that the problem of rendering human meaning legible is not just about selecting better metrics but about developing new representational formats based on thick description. We frame this as a crucial direction for the application of generative AI and identify five key challenges: preserving context, maintaining interpretive pluralism, integrating perspectives based on lived experience and critical distance, distinguishing qualitative content from quantitative magnitude, and acknowledging meaning as dynamic rather than static.
>
---
#### [replaced 072] Coupled Variational Reinforcement Learning for Language Model General Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型推理任务，解决验证奖励依赖问题。通过耦合变分推理与强化学习，提升推理效率和答案一致性。**

- **链接: [https://arxiv.org/pdf/2512.12576v2](https://arxiv.org/pdf/2512.12576v2)**

> **作者:** Xueru Wen; Jie Lou; Yanjiang Liu; Hongyu Lin; Ben He; Xianpei Han; Le Sun; Yaojie Lu; Debing Zhang
>
> **摘要:** While reinforcement learning has achieved impressive progress in language model reasoning, it is constrained by the requirement for verifiable rewards. Recent verifier-free RL methods address this limitation by utilizing the probabilities that LLMs generate reference answers as reward signals. However, these approaches typically sample reasoning traces conditioned only on the question. This design decouples reasoning-trace sampling from answer information, leading to inefficient exploration and incoherence between traces and final answers. In this paper, we propose \textit{\b{Co}upled \b{V}ariational \b{R}einforcement \b{L}earning} (CoVRL), which bridges variational inference and reinforcement learning by coupling prior and posterior distributions through a hybrid sampling strategy. By constructing and optimizing a composite distribution that integrates these two distributions, CoVRL enables efficient exploration while preserving strong thought-answer coherence. Extensive experiments on mathematical and general reasoning benchmarks show that CoVRL improves performance by 12.4\% over the base model and achieves an additional 2.3\% improvement over state-of-the-art verifier-free RL baselines, providing a principled framework for enhancing the general reasoning capabilities of language models.
>
---
#### [replaced 073] Improving Value-based Process Verifier via Structural Prior Injection
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习中的价值验证任务，旨在解决蒙特卡洛采样带来的噪声和误差问题。通过注入结构先验，将标量值转化为分布期望，提升验证器性能。**

- **链接: [https://arxiv.org/pdf/2502.17498v2](https://arxiv.org/pdf/2502.17498v2)**

> **作者:** Zetian Sun; Dongfang Li; Baotian Hu; Jun Yu; Min Zhang
>
> **备注:** This version is deprecated. Please refer to our new version: arXiv:2508.10539
>
> **摘要:** In the Large Language Model(LLM) reasoning scenario, people often estimate state value via Monte Carlo sampling. Though Monte Carlo estimation is an elegant method with less inductive bias, noise and errors are inevitably introduced due to the limited sampling. To handle the problem, we inject the structural prior into the value representation and transfer the scalar value into the expectation of a pre-defined categorical distribution, representing the noise and errors from a distribution perspective. Specifically, by treating the result of Monte Carlo sampling as a single sample from the prior ground-truth Binomial distribution, we quantify the sampling error as the mismatch between posterior estimated distribution and ground-truth distribution, which is thus optimized via distribution selection optimization. We test the performance of value-based process verifiers on Best-of-N task and Beam search task. Compared with the scalar value representation, we show that reasonable structural prior injection induced by different objective functions or optimization methods can improve the performance of value-based process verifiers for about 1$\sim$2 points at little-to-no cost. We also show that under different structural prior, the verifiers' performances vary greatly despite having the same optimal solution, indicating the importance of reasonable structural prior injection.
>
---
#### [replaced 074] Multimodal Multi-Agent Empowered Legal Judgment Prediction
- **分类: cs.CL**

- **简介: 该论文属于法律判决预测任务，旨在解决传统方法在处理复杂案件时的不足。提出JurisMMA框架，构建JurisMM数据集，提升预测效果与适用性。**

- **链接: [https://arxiv.org/pdf/2601.12815v4](https://arxiv.org/pdf/2601.12815v4)**

> **作者:** Zhaolu Kang; Junhao Gong; Qingxi Chen; Hao Zhang; Jiaxin Liu; Rong Fu; Zhiyuan Feng; Yuan Wang; Simon Fong; Kaiyue Zhou
>
> **备注:** Accepted to the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP) 2026
>
> **摘要:** Legal Judgment Prediction (LJP) aims to predict the outcomes of legal cases based on factual descriptions, serving as a fundamental task to advance the development of legal systems. Traditional methods often rely on statistical analyses or role-based simulations but face challenges with multiple allegations, diverse evidence, and lack adaptability. In this paper, we introduce JurisMMA, a novel framework for LJP that effectively decomposes trial tasks, standardizes processes, and organizes them into distinct stages. Furthermore, we build JurisMM, a large dataset with over 100,000 recent Chinese judicial records, including both text and multimodal video-text data, enabling comprehensive evaluation. Experiments on JurisMM and the benchmark LawBench validate our framework's effectiveness. These results indicate that our framework is effective not only for LJP but also for a broader range of legal applications, offering new perspectives for the development of future legal methods and datasets.
>
---
#### [replaced 075] Human Cognitive Benchmarks Reveal Foundational Visual Gaps in MLLMs
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉认知研究任务，旨在揭示MLLM在基础视觉能力上的不足。通过构建VisFactor基准，评估23个模型，发现其在空间推理等任务上表现不佳，表明现有模型未真正掌握人类视觉认知。**

- **链接: [https://arxiv.org/pdf/2502.16435v3](https://arxiv.org/pdf/2502.16435v3)**

> **作者:** Jen-Tse Huang; Dasen Dai; Jen-Yuan Huang; Youliang Yuan; Xiaoyuan Liu; Wenxuan Wang; Wenxiang Jiao; Pinjia He; Zhaopeng Tu; Haodong Duan
>
> **备注:** Update: Evaluated 23 SOTA MLLMs
>
> **摘要:** Humans develop perception through a bottom-up hierarchy: from basic primitives and Gestalt principles to high-level semantics. In contrast, current Multimodal Large Language Models (MLLMs) are trained directly on complex downstream tasks, often bypassing these foundational visual capabilities. To systematically investigate this gap, we introduce VisFactor, a benchmark that digitizes 20 vision-centric subtests from FRCT, a well-established cognitive psychology assessment spanning four domains of human visual cognition. Furthermore, we design algorithms to automatically construct and validate unlimited test cases with controllable difficulty. Using VisFactor, we evaluate 23 frontier MLLMs, including both proprietary (e.g., GPT, Gemini) and open-source (e.g., LLaMA, Qwen) models. The best model achieves a score of only 30.17%. Models consistently fail on tasks such as mental rotation, spatial relation inference, and figure-ground discrimination, regardless of model size or prompting strategy. These findings suggest that performance improvements on existing general benchmarks might represent castles in the air instead of a genuine mastery of human-like visual cognition.
>
---
#### [replaced 076] Prompt-Counterfactual Explanations for Generative AI System Behavior
- **分类: cs.LG; cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于AI可解释性任务，旨在解决生成式AI系统行为的解释问题。通过提出提示反事实解释框架，帮助理解输入提示如何导致特定输出特征。**

- **链接: [https://arxiv.org/pdf/2601.03156v2](https://arxiv.org/pdf/2601.03156v2)**

> **作者:** Sofie Goethals; Foster Provost; João Sedoc
>
> **摘要:** As generative AI systems become integrated into real-world applications, organizations increasingly need to be able to understand and interpret their behavior. In particular, decision-makers need to understand what causes generative AI systems to exhibit specific output characteristics. Within this general topic, this paper examines a key question: what is it about the input -- the prompt -- that causes an LLM-based generative AI system to produce output that exhibits specific characteristics, such as toxicity, negative sentiment, or political bias. To examine this question, we adapt a common technique from the Explainable AI literature: counterfactual explanations. We explain why traditional counterfactual explanations cannot be applied directly to generative AI systems, due to several differences in how generative AI systems function. We then propose a flexible framework that adapts counterfactual explanations to non-deterministic, generative AI systems in scenarios where downstream classifiers can reveal key characteristics of their outputs. Based on this framework, we introduce an algorithm for generating prompt-counterfactual explanations (PCEs). Finally, we demonstrate the production of counterfactual explanations for generative AI systems with three case studies, examining different output characteristics (viz., political leaning, toxicity, and sentiment). The case studies further show that PCEs can streamline prompt engineering to suppress undesirable output characteristics and can enhance red-teaming efforts to uncover additional prompts that elicit undesirable outputs. Ultimately, this work lays a foundation for prompt-focused interpretability in generative AI: a capability that will become indispensable as these models are entrusted with higher-stakes tasks and subject to emerging regulatory requirements for transparency and accountability.
>
---
#### [replaced 077] Rewarding Doubt: A Reinforcement Learning Approach to Calibrated Confidence Expression of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型置信度校准任务，旨在解决模型回答时信心表达不准确的问题。通过强化学习方法，使模型生成答案时同步输出校准的置信度。**

- **链接: [https://arxiv.org/pdf/2503.02623v4](https://arxiv.org/pdf/2503.02623v4)**

> **作者:** David Bani-Harouni; Chantal Pellegrini; Paul Stangel; Ege Özsoy; Kamilia Zaripova; Matthias Keicher; Nassir Navab
>
> **摘要:** A safe and trustworthy use of Large Language Models (LLMs) requires an accurate expression of confidence in their answers. We propose a novel Reinforcement Learning approach that allows to directly fine-tune LLMs to express calibrated confidence estimates alongside their answers to factual questions. Our method optimizes a reward based on the logarithmic scoring rule, explicitly penalizing both over- and under-confidence. This encourages the model to align its confidence estimates with the actual predictive accuracy. The optimal policy under our reward design would result in perfectly calibrated confidence expressions. Unlike prior approaches that decouple confidence estimation from response generation, our method integrates confidence calibration seamlessly into the generative process of the LLM. Empirically, we demonstrate that models trained with our approach exhibit substantially improved calibration and generalize to unseen tasks without further fine-tuning, suggesting the emergence of general confidence awareness.
>
---
#### [replaced 078] Is On-Policy Data always the Best Choice for Direct Preference Optimization-based LM Alignment?
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究语言模型对齐任务，探讨是否始终使用策略数据优化模型。工作包括分析静态与策略数据效果差异，提出对齐阶段假设，并验证算法有效性。**

- **链接: [https://arxiv.org/pdf/2508.10530v2](https://arxiv.org/pdf/2508.10530v2)**

> **作者:** Zetian Sun; Dongfang Li; Xuhui Chen; Baotian Hu; Min Zhang
>
> **备注:** Accepted by ICLR-2026
>
> **摘要:** The alignment of language models~(LMs) with human preferences is critical for building reliable AI systems. The problem is typically framed as optimizing an LM policy to maximize the expected reward that reflects human preferences. Recently, Direct Preference Optimization~(DPO) was proposed as a LM alignment method that directly optimize the policy from static preference data, and further improved by incorporating on-policy sampling~(i.e., preference candidates generated during the training loop) for better LM alignment. However, we show on-policy data is not always optimal, with systematic effectiveness difference emerging between static and on-policy preference candidates. For example, on-policy data can result in a $3\times$ effectiveness compared with static data for Llama-3, and a $0.4\times$ effectiveness for Zephyr. To explain the phenomenon, we propose the alignment stage assumption, which divides the alignment process into two distinct stages: the preference injection stage, which benefits from diverse data, and the preference fine-tuning stage, which favors high-quality data. Through theoretical and empirical analysis, we characterize these stages and propose an effective algorithm to identify the boundaries between them. We perform experiments on $5$ models~(Llama, Zephyr, Phi-2, Qwen, Pythia) and $2$ alignment methods~(DPO, SLiC-HF) to show the generalizability of alignment stage assumption and the effectiveness of the boundary measurement algorithm.
>
---
#### [replaced 079] Ultra-Low-Dimensional Prompt Tuning via Random Projection
- **分类: cs.CL**

- **简介: 该论文提出ULPT，解决大模型微调参数效率问题。通过低维提示优化和随机矩阵上投影，显著减少参数量并保持性能。属于自然语言处理中的参数高效微调任务。**

- **链接: [https://arxiv.org/pdf/2502.04501v2](https://arxiv.org/pdf/2502.04501v2)**

> **作者:** Zijun Wu; Yongchang Hao; Lili Mou
>
> **备注:** Accepted by EACL 2026 (Main Conference, Long Paper)
>
> **摘要:** Large language models achieve state-of-the-art performance but are increasingly costly to fine-tune. Prompt tuning is a parameter-efficient fine-tuning method that addresses parameter-efficiency by learning prompt embeddings, but these embeddings are typically tied to the model's hidden dimensionality, limiting parameter saving. In this paper, we propose Ultra-Low-dimensional Prompt Tuning (ULPT), a simple yet effective method that optimizes prompts in a low-dimensional space (e.g., 2D) and uses a frozen random matrix for up-projection. ULPT can achieve 98% reduction in the training parameters compared to vanilla prompt tuning while preserving performance. Our extensive experiments across over 20 NLP tasks demonstrate that ULPT consistently outperforms recent parameter-efficient tuning methods using significantly fewer parameters, making it well-suited as a storage-efficient framework for massive LLM customization.
>
---
