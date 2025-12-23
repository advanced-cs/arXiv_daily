# 自然语言处理 cs.CL

- **最新发布 100 篇**

- **更新 62 篇**

## 最新发布

#### [new 001] Event Extraction in Large Language Model
- **分类: cs.CL**

- **简介: 该论文是事件抽取（EE）领域的综述，聚焦LLM时代下EE的系统性重构。它指出当前LLM-EE存在幻觉、长程推理弱、知识管理受限等问题，提出以事件为中心的认知支架框架，涵盖事件模式、结构化中间表示、图增强检索与可更新事件存储，并系统梳理方法演进、任务分类、挑战与未来方向。**

- **链接: [https://arxiv.org/pdf/2512.19537v1](https://arxiv.org/pdf/2512.19537v1)**

> **作者:** Bobo Li; Xudong Han; Jiang Liu; Yuzhe Ding; Liqiang Jing; Zhaoqi Zhang; Jinheng Li; Xinya Du; Fei Li; Meishan Zhang; Min Zhang; Aixin Sun; Philip S. Yu; Hao Fei
>
> **备注:** 38 pages, 9 Figures, 5 Tables
>
> **摘要:** Large language models (LLMs) and multimodal LLMs are changing event extraction (EE): prompting and generation can often produce structured outputs in zero shot or few shot settings. Yet LLM based pipelines face deployment gaps, including hallucinations under weak constraints, fragile temporal and causal linking over long contexts and across documents, and limited long horizon knowledge management within a bounded context window. We argue that EE should be viewed as a system component that provides a cognitive scaffold for LLM centered solutions. Event schemas and slot constraints create interfaces for grounding and verification; event centric structures act as controlled intermediate representations for stepwise reasoning; event links support relation aware retrieval with graph based RAG; and event stores offer updatable episodic and agent memory beyond the context window. This survey covers EE in text and multimodal settings, organizing tasks and taxonomy, tracing method evolution from rule based and neural models to instruction driven and generative frameworks, and summarizing formulations, decoding strategies, architectures, representations, datasets, and evaluation. We also review cross lingual, low resource, and domain specific settings, and highlight open challenges and future directions for reliable event centric systems. Finally, we outline open challenges and future directions that are central to the LLM era, aiming to evolve EE from static extraction into a structurally reliable, agent ready perception and memory layer for open world systems.
>
---
#### [new 002] KVReviver: Reversible KV Cache Compression with Sketch-Based Token Reconstruction
- **分类: cs.CL; cs.AI**

- **简介: 该论文属大语言模型推理优化任务，旨在解决长上下文下KV缓存内存爆炸导致的部署瓶颈问题。提出KVReviver方法，基于Sketch算法实现可逆KV压缩与精准令牌重建，避免信息丢失，在大幅降低内存（10%–25%）的同时保持高精度。**

- **链接: [https://arxiv.org/pdf/2512.17917v1](https://arxiv.org/pdf/2512.17917v1)**

> **作者:** Aomufei Yuan; Zhiming Wang; Ruijie Miao; Dayu Wang; Yuxuan Tian; Zihan Wang; Yebo Peng; Yuhan Wu; Bairen Yi; Xin Liu; Tong Yang
>
> **备注:** 12 pages, 6 figures
>
> **摘要:** As the context length of current large language models (LLMs) rapidly increases, the memory demand for the Key-Value (KV) cache is becoming a bottleneck for LLM deployment and batch processing. Traditional KV cache compression methods typically involve permanently evicting or irreversibly merging "less important" tokens with low attention scores. This approach results in the unrecoverable loss of token information, which we call Contextual Amnesia, significantly degrading the model's information retrieval capability. To address this issue, we propose KVReviver, a reversible KV cache compression method based on the sketch algorithm. This method allows reconstructing compressed tokens from an additional data structure, thus enabling full-scale computation within limited memory. Experiments showed that in 2k-length contexts, it requires only 10% of KV Cache budget while maintaining identical end-to-end inference accuracy. For 32k-length contexts, it achieves equivalent or comparable accuracy ~2% accuracy loss) using merely 25% of KV Cache budget.
>
---
#### [new 003] FASTRIC: Prompt Specification Language for Verifiable LLM Interactions
- **分类: cs.CL; cs.SE**

- **简介: 该论文提出FASTRIC——一种面向可验证大模型交互的提示规范语言，旨在解决LLM多轮交互缺乏形式化规范与可验证性的问题。它将隐式有限状态机显式化，支持按模型能力调整规范形式化程度，并定义过程一致性作为验证指标，推动交互设计从经验艺术走向系统工程。**

- **链接: [https://arxiv.org/pdf/2512.18940v1](https://arxiv.org/pdf/2512.18940v1)**

> **作者:** Wen-Long Jin
>
> **备注:** 13 pages, 3 figures. Supplementary materials at https://doi.org/10.17605/OSF.IO/PV6R3
>
> **摘要:** Large Language Models (LLMs) execute complex multi-turn interaction protocols but lack formal specifications to verify execution against designer intent. We introduce FASTRIC, a Prompt Specification Language that makes implicit Finite State Machines (FSMs) explicit in natural language prompts, enabling conformance verification through execution trace analysis. The LLM serves as intelligent execution agent: interpreting designer-encoded FSMs to execute specified behavioral roles. Unlike symbolic specification languages requiring parsers and compilers, FASTRIC leverages LLMs as unified infrastructure-simultaneously parser, interpreter, runtime environment, and development assistant. FASTRIC guides designers to articulate seven FSM elements (Final States, Agents, States, Triggers, Roles, Initial State, Constraints) structuring multi-turn interactions. Specification formality-ranging from implicit descriptions that frontier models infer to explicit step-by-step instructions for weaker models-serves as a design parameter. We introduce procedural conformance as verification metric measuring execution adherence to FSM specifications. Testing a 3-state kindergarten tutoring FSM across four formality levels and three model scales (14.7B, 685B, 1T+ parameters) reveals optimal specification formality is a function of model capacity. DeepSeek-V3.2 (685B) achieves perfect conformance (1.00) at L2-L4; ChatGPT-5 (~1T) peaks at L3 (0.90) before collapsing at L4 (0.39); Phi4 (14.7B) shows no stable optimum with high variance (SD=0.16-0.36). These findings reveal model-specific formality ranges-"Goldilocks zones"-where specifications provide sufficient structure without over-constraint, establishing Prompt Specification Engineering for creating verifiable interaction protocols, transforming multi-turn interaction design from heuristic art to systematic engineering with measurable procedural guarantees.
>
---
#### [new 004] Teaching and Critiquing Conceptualization and Operationalization in NLP
- **分类: cs.CL**

- **简介: 该论文属教育方法研究任务，旨在解决NLP领域对“解释性”“偏见”等抽象概念缺乏明确定义与可操作化标准的问题。作者设计并实践了一门研讨课，通过跨学科阅读、讨论与批判，引导学生反思概念化与操作化过程。**

- **链接: [https://arxiv.org/pdf/2512.18505v1](https://arxiv.org/pdf/2512.18505v1)**

> **作者:** Vagrant Gautam
>
> **摘要:** NLP researchers regularly invoke abstract concepts like "interpretability," "bias," "reasoning," and "stereotypes," without defining them. Each subfield has a shared understanding or conceptualization of what these terms mean and how we should treat them, and this shared understanding is the basis on which operational decisions are made: Datasets are built to evaluate these concepts, metrics are proposed to quantify them, and claims are made about systems. But what do they mean, what should they mean, and how should we measure them? I outline a seminar I created for students to explore these questions of conceptualization and operationalization, with an interdisciplinary reading list and an emphasis on discussion and critique.
>
---
#### [new 005] InstructNet: A Novel Approach for Multi-Label Instruction Classification through Advanced Deep Learning
- **分类: cs.CL**

- **简介: 该论文提出InstructNet，解决多标签指令分类任务，即对“How To”类文章自动标注多个类别。基于wikiHow数据集（11,121条），采用XLNet、BERT等Transformer模型，以准确率和宏F1为指标评估，XLNet达97.30%准确率和93%宏F1。**

- **链接: [https://arxiv.org/pdf/2512.18301v1](https://arxiv.org/pdf/2512.18301v1)**

> **作者:** Tanjim Taharat Aurpa; Md Shoaib Ahmed; Md Mahbubur Rahman; Md. Golam Moazzam
>
> **摘要:** People use search engines for various topics and items, from daily essentials to more aspirational and specialized objects. Therefore, search engines have taken over as peoples preferred resource. The How To prefix has become familiar and widely used in various search styles to find solutions to particular problems. This search allows people to find sequential instructions by providing detailed guidelines to accomplish specific tasks. Categorizing instructional text is also essential for task-oriented learning and creating knowledge bases. This study uses the How To articles to determine the multi-label instruction category. We have brought this work with a dataset comprising 11,121 observations from wikiHow, where each record has multiple categories. To find out the multi-label category meticulously, we employ some transformer-based deep neural architectures, such as Generalized Autoregressive Pretraining for Language Understanding (XLNet), Bidirectional Encoder Representation from Transformers (BERT), etc. In our multi-label instruction classification process, we have reckoned our proposed architectures using accuracy and macro f1-score as the performance metrics. This thorough evaluation showed us much about our strategys strengths and drawbacks. Specifically, our implementation of the XLNet architecture has demonstrated unprecedented performance, achieving an accuracy of 97.30% and micro and macro average scores of 89.02% and 93%, a noteworthy accomplishment in multi-label classification. This high level of accuracy and macro average score is a testament to the effectiveness of the XLNet architecture in our proposed InstructNet approach. By employing a multi-level strategy in our evaluation process, we have gained a more comprehensive knowledge of the effectiveness of our proposed architectures and identified areas for forthcoming improvement and refinement.
>
---
#### [new 006] DACE For Railway Acronym Disambiguation
- **分类: cs.CL**

- **简介: 该论文面向铁路领域法语文档的缩略语消歧任务，提出DACE框架：融合动态提示、检索增强生成、上下文选择与集成聚合，提升大模型在低资源、高歧义场景下的缩略语识别准确率，竞赛F1达0.9069，获第一名。**

- **链接: [https://arxiv.org/pdf/2512.18357v1](https://arxiv.org/pdf/2512.18357v1)**

> **作者:** El Mokhtar Hribach; Oussama Mechhour; Mohammed Elmonstaser; Yassine El Boudouri; Othmane Kabal
>
> **摘要:** Acronym Disambiguation (AD) is a fundamental challenge in technical text processing, particularly in specialized sectors where high ambiguity complicates automated analysis. This paper addresses AD within the context of the TextMine'26 competition on French railway documentation. We present DACE (Dynamic Prompting, Retrieval Augmented Generation, Contextual Selection, and Ensemble Aggregation), a framework that enhances Large Language Models through adaptive in-context learning and external domain knowledge injection. By dynamically tailoring prompts to acronym ambiguity and aggregating ensemble predictions, DACE mitigates hallucination and effectively handles low-resource scenarios. Our approach secured the top rank in the competition with an F1 score of 0.9069.
>
---
#### [new 007] Remedy-R: Generative Reasoning for Machine Translation Evaluation without Error Annotations
- **分类: cs.CL**

- **简介: 该论文提出Remedy-R，一种无需错误标注的生成式机器翻译评估方法，通过强化学习学习成对偏好，输出可解释的分步分析与评分。它解决现有自动指标黑盒、泛化差问题，并构建Remedy-R Agent实现评估-修订闭环，提升多模型翻译质量。**

- **链接: [https://arxiv.org/pdf/2512.18906v1](https://arxiv.org/pdf/2512.18906v1)**

> **作者:** Shaomu Tan; Ryosuke Mitani; Ritvik Choudhary; Qiyu Wu; Toshiyuki Sekiya; Christof Monz
>
> **摘要:** Over the years, automatic MT metrics have hillclimbed benchmarks and presented strong and sometimes human-level agreement with human ratings. Yet they remain black-box, offering little insight into their decision-making and often failing under real-world out-of-distribution (OOD) inputs. We introduce Remedy-R, a reasoning-driven generative MT metric trained with reinforcement learning from pairwise translation preferences, without requiring error-span annotations or distillation from closed LLMs. Remedy-R produces step-by-step analyses of accuracy, fluency, and completeness, followed by a final score, enabling more interpretable assessments. With only 60K training pairs across two language pairs, Remedy-R remains competitive with top scalar metrics and GPT-4-based judges on WMT22-24 meta-evaluation, generalizes to other languages, and exhibits strong robustness on OOD stress tests. Moreover, Remedy-R models generate self-reflective feedback that can be reused for translation improvement. Building on this finding, we introduce Remedy-R Agent, a simple evaluate-revise pipeline that leverages Remedy-R's evaluation analysis to refine translations. This agent consistently improves translation quality across diverse models, including Qwen2.5, ALMA-R, GPT-4o-mini, and Gemini-2.0-Flash, suggesting that Remedy-R's reasoning captures translation-relevant information and is practically useful.
>
---
#### [new 008] AraToken: Optimizing Arabic Tokenization with Normalization Pipeline and Language Extension for Qwen3
- **分类: cs.CL; cs.AI**

- **简介: 该论文属NLP基础模型优化任务，旨在解决通用分词器对阿拉伯语分词效率低、压缩差的问题。作者提出AraToken分词器，结合阿拉伯语规范化流程与SentencePiece Unigram算法，并设计语言扩展管道（LEP）将之集成至Qwen3模型，显著提升分词效率与模型性能。**

- **链接: [https://arxiv.org/pdf/2512.18399v1](https://arxiv.org/pdf/2512.18399v1)**

> **作者:** Mark Kashirskiy; Artiom Lipinski; Ilya Makarov
>
> **备注:** 8 pages, 8 figures, 5 tables
>
> **摘要:** Tokenization is a critical preprocessing step for large language models (LLMs), directly impacting training efficiency and downstream performance. General-purpose tokenizers trained predominantly on English and Latin-script languages exhibit suboptimal performance on morphologically rich languages such as Arabic, resulting in inflated token sequences and reduced compression efficiency. In this work, we present AraToken, an Arabic-optimized tokenizer built on SentencePiece Unigram algorithm with a comprehensive normalization pipeline addressing Arabic-specific orthographic variations including Alif variants, diacritics, and Arabic-Indic numerals. We systematically compare BPE, WordPiece, and SentencePiece algorithms across multiple configurations, demonstrating that SentencePiece with normalization achieves 18% lower fertility (1.199 vs 1.35 tokens/word) compared to unnormalized baselines. Furthermore, we introduce the Language Extension Pipeline (LEP), a method for integrating the optimized tokenizer into Qwen3-0.6B through vocabulary extension with mean subtoken initialization and selective transformer layer unfreezing. Our experiments show that LEP reduces evaluation loss from 8.28 to 2.43 within 800 training steps on 100K Arabic samples. We release our tokenizer, training scripts, and model checkpoints to facilitate Arabic NLP research.
>
---
#### [new 009] SiamGPT: Quality-First Fine-Tuning for Stable Thai Text Generation
- **分类: cs.CL**

- **简介: 该论文面向泰语大模型生成不稳问题，提出SiamGPT-32B：基于Qwen3-32B，采用“质量优先”监督微调策略，融合翻译高复杂度英文指令与泰语适配AutoIF框架，仅用SFT提升指令遵循、多轮对话及语言稳定性，在SEA-HELM上达同规模开源泰语模型最优。**

- **链接: [https://arxiv.org/pdf/2512.19455v1](https://arxiv.org/pdf/2512.19455v1)**

> **作者:** Thittipat Pairatsuppawat; Abhibhu Tachaapornchai; Paweekorn Kusolsomboon; Chutikan Chaiwong; Thodsaporn Chay-intr; Kobkrit Viriyayudhakorn; Nongnuch Ketui; Aslan B. Wong
>
> **摘要:** Open-weights large language models remain difficult to deploy for Thai due to unstable generation under complex instructions, despite strong English performance. To mitigate these limitations, We present SiamGPT-32B, an open-weights model based on Qwen3-32B, fine-tuned with a Quality-First strategy emphasizing curated supervision over data scale. The fine-tuning pipeline combines translated high-complexity English instruction data with a Thai-adapted AutoIF framework for instruction and linguistic constraints. Using supervised fine-tuning only, without continual pretraining or corpus expansion, SiamGPT-32B improves instruction adherence, multi-turn robustness, and linguistic stability. Evaluations on the SEA-HELM benchmark show that SiamGPT-32B achieves the strongest overall performance among similar-scale open-weights Thai models, with consistent gains in instruction following, multi-turn dialogue, and natural language understanding.
>
---
#### [new 010] Stop saying LLM: Large Discourse Models (LDM) and Artificial Discursive Agent (ADA)?
- **分类: cs.CL**

- **简介: 该论文属科技哲学/AI治理任务，旨在解决LLM概念引发的认知与社会误读问题。作者提出用“大型话语模型”（LDM）和“人工话语代理”（ADA）替代“大语言模型”（LLM），基于三重本体论框架重构其理论定位，并倡导多方协同的公共治理路径。**

- **链接: [https://arxiv.org/pdf/2512.19117v1](https://arxiv.org/pdf/2512.19117v1)**

> **作者:** Amar Lakel
>
> **备注:** in French language
>
> **摘要:** This paper proposes an epistemological shift in the analysis of large generative models, replacing the category ''Large Language Models'' (LLM) with that of ''Large Discourse Models'' (LDM), and then with that of Artificial Discursive Agent (ADA). The theoretical framework is based on an ontological triad distinguishing three regulatory instances: the apprehension of the phenomenal regularities of the referential world, the structuring of embodied cognition, and the structural-linguistic sedimentation of the utterance within a socio-historical context. LDMs, operating on the product of these three instances (the document), model the discursive projection of a portion of human experience reified by the learning corpus. The proposed program aims to replace the ''fascination/fear'' dichotomy with public trials and procedures that make the place, uses, and limits of artificial discursive agents in contemporary social space decipherable, situating this approach within a perspective of governance and co-regulation involving the State, industry, civil society, and academia.
>
---
#### [new 011] A Large-Language-Model Framework for Automated Humanitarian Situation Reporting
- **分类: cs.CL**

- **简介: 该论文提出基于大语言模型的自动化人道主义态势报告框架，旨在解决传统人工报告耗时、低效、不一致的问题。工作包括语义聚类、自动提问、检索增强式答案抽取与引用、多级摘要及评估，实现结构化、可验证、可操作的报告生成。**

- **链接: [https://arxiv.org/pdf/2512.19475v1](https://arxiv.org/pdf/2512.19475v1)**

> **作者:** Ivan Decostanzi; Yelena Mejova; Kyriaki Kalimeri
>
> **备注:** 18 pages, 3 figures
>
> **摘要:** Timely and accurate situational reports are essential for humanitarian decision-making, yet current workflows remain largely manual, resource intensive, and inconsistent. We present a fully automated framework that uses large language models (LLMs) to transform heterogeneous humanitarian documents into structured and evidence-grounded reports. The system integrates semantic text clustering, automatic question generation, retrieval augmented answer extraction with citations, multi-level summarization, and executive summary generation, supported by internal evaluation metrics that emulate expert reasoning. We evaluated the framework across 13 humanitarian events, including natural disasters and conflicts, using more than 1,100 documents from verified sources such as ReliefWeb. The generated questions achieved 84.7 percent relevance, 84.0 percent importance, and 76.4 percent urgency. The extracted answers reached 86.3 percent relevance, with citation precision and recall both exceeding 76 percent. Agreement between human and LLM based evaluations surpassed an F1 score of 0.80. Comparative analysis shows that the proposed framework produces reports that are more structured, interpretable, and actionable than existing baselines. By combining LLM reasoning with transparent citation linking and multi-level evaluation, this study demonstrates that generative AI can autonomously produce accurate, verifiable, and operationally useful humanitarian situation reports.
>
---
#### [new 012] Q-KVComm: Efficient Multi-Agent Communication Via Adaptive KV Cache Compression
- **分类: cs.CL; cs.MA**

- **简介: 该论文属多智能体LLM通信任务，解决代理间冗余文本传输导致的带宽与计算开销问题。提出Q-KVComm协议，通过自适应KV缓存压缩（含层敏感量化、混合信息提取、异构校准），实现5–6×压缩比且保持语义保真。**

- **链接: [https://arxiv.org/pdf/2512.17914v1](https://arxiv.org/pdf/2512.17914v1)**

> **作者:** Boris Kriuk; Logic Ng
>
> **备注:** 7 pages, 4 figures, 1 table
>
> **摘要:** Multi-agent Large Language Model (LLM) systems face a critical bottleneck: redundant transmission of contextual information between agents consumes excessive bandwidth and computational resources. Traditional approaches discard internal semantic representations and transmit raw text, forcing receiving agents to recompute similar representations from scratch. We introduce Q-KVComm, a new protocol that enables direct transmission of compressed key-value (KV) cache representations between LLM agents. Q-KVComm combines three key innovations: (1) adaptive layer-wise quantization that allocates variable bit-widths based on sensitivity profiling, (2) hybrid information extraction that preserves critical facts across content domains, and (3) heterogeneous model calibration establishing cross-architecture communication. Extensive experiments across three diverse question-answering datasets demonstrate that Q-KVComm achieves 5-6x compression ratios while maintaining semantic fidelity, with coherence quality scores above 0.77 across all scenarios. The protocol exhibits robust performance across model sizes (1.1B-1.5B parameters) and adapts to real-world applications including conversational QA and multi-hop reasoning. Our work establishes a new paradigm for LLM agent communication, shifting from text-based to representation-based information exchange.
>
---
#### [new 013] HATS: High-Accuracy Triple-Set Watermarking for Large Language Models
- **分类: cs.CL**

- **简介: 该论文属LLM水印任务，旨在防范生成文本滥用。提出HATS方法：解码时将词表三分为Green/Yellow/Red集，仅从Green/Yellow采样；检测时通过Green富集与Red缺失的z-score及Fisher法聚合p值判定水印。在Llama 2 7B上验证了高检出率与低误报率下的文本质量保持。**

- **链接: [https://arxiv.org/pdf/2512.19378v1](https://arxiv.org/pdf/2512.19378v1)**

> **作者:** Zhiqing Hu; Chenxu Zhao; Jiazhong Lu; Xiaolei Liu
>
> **备注:** Camera-ready version of the paper accepted for oral presentation at the 11th International Conference on Computer and Communications (ICCC 2025)
>
> **摘要:** Misuse of LLM-generated text can be curbed by watermarking techniques that embed implicit signals into the output. We propose a watermark that partitions the vocabulary at each decoding step into three sets (Green/Yellow/Red) with fixed ratios and restricts sampling to the Green and Yellow sets. At detection time, we replay the same partitions, compute Green-enrichment and Red-depletion statistics, convert them to one-sided z-scores, and aggregate their p-values via Fisher's method to decide whether a passage is watermarked. We implement generation, detection, and testing on Llama 2 7B, and evaluate true-positive rate, false-positive rate, and text quality. Results show that the triple-partition scheme achieves high detection accuracy at fixed FPR while preserving readability.
>
---
#### [new 014] CoPE: A Small Language Model for Steerable and Scalable Content Labeling
- **分类: cs.CL; cs.CY; cs.SI**

- **简介: 该论文提出CoPE——一种可策略引导的小型语言模型，用于内容标注任务。旨在解决传统大模型资源消耗高、政策理解僵化的问题。工作包括：提出矛盾样例训练法提升策略理解能力，设计双目标注法高效构建无歧义数据集，并开源9B参数模型，在七类危害检测中达前沿模型精度且仅需1%参数量。**

- **链接: [https://arxiv.org/pdf/2512.18027v1](https://arxiv.org/pdf/2512.18027v1)**

> **作者:** Samidh Chakrabarti; David Willner; Kevin Klyman; Tiffany Saade; Emily Capstick; Sabina Nong
>
> **备注:** 21 pages, 2 figures, 7 tables
>
> **摘要:** This paper details the methodology behind CoPE, a policy-steerable small language model capable of fast and accurate content labeling. We present a novel training curricula called Contradictory Example Training that enables the model to learn policy interpretation rather than mere policy memorization. We also present a novel method for generating content policies, called Binocular Labeling, which enables rapid construction of unambiguous training datasets. When evaluated across seven different harm areas, CoPE exhibits equal or superior accuracy to frontier models at only 1% of their size. We openly release a 9 billion parameter version of the model that can be run on a single consumer-grade GPU. Models like CoPE represent a paradigm shift for classifier systems. By turning an ML task into a policy writing task, CoPE opens up new design possibilities for the governance of online platforms.
>
---
#### [new 015] On Finding Inconsistencies in Documents
- **分类: cs.CL**

- **简介: 该论文聚焦文档不一致性检测任务，旨在辅助专业人士高效发现文本中的逻辑、事实或表述矛盾。作者构建了专业人工标注的基准数据集FIND，并评估大语言模型（如GPT-5）在此任务上的表现，发现其虽能识别部分已知及未知不一致，但准确率仅64%，表明该任务仍具挑战性。**

- **链接: [https://arxiv.org/pdf/2512.18601v1](https://arxiv.org/pdf/2512.18601v1)**

> **作者:** Charles J. Lovering; Seth Ebner; Brandon Smock; Michael Krumdick; Saad Rabbani; Ahmed Muhammad; Varshini Reddy; Chris Tanner
>
> **摘要:** Professionals in academia, law, and finance audit their documents because inconsistencies can result in monetary, reputational, and scientific costs. Language models (LMs) have the potential to dramatically speed up this auditing process. To understand their abilities, we introduce a benchmark, FIND (Finding INconsistencies in Documents), where each example is a document with an inconsistency inserted manually by a domain expert. Despite the documents being long, technical, and complex, the best-performing model (gpt-5) recovered 64% of the inserted inconsistencies. Surprisingly, gpt-5 also found undiscovered inconsistencies present in the original documents. For example, on 50 arXiv papers, we judged 136 out of 196 of the model's suggestions to be legitimate inconsistencies missed by the original authors. However, despite these findings, even the best models miss almost half of the inconsistencies in FIND, demonstrating that inconsistency detection is still a challenging task.
>
---
#### [new 016] Evaluating the Challenges of LLMs in Real-world Medical Follow-up: A Comparative Study and An Optimized Framework
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦医疗随访任务，解决LLM直接端到端应用导致的对话失控与信息抽取不准问题。提出并验证一种模块化框架（含任务分解、语义聚类与流程管控），显著提升对话稳定性、抽取精度，降低轮次46.73%及token消耗80%–87.5%。**

- **链接: [https://arxiv.org/pdf/2512.18999v1](https://arxiv.org/pdf/2512.18999v1)**

> **作者:** Jinyan Liu; Zikang Chen; Qinchuan Wang; Tan Xie; Heming Zheng; Xudong Lv
>
> **备注:** 10 pages,3 figures,conference ICCBB2025
>
> **摘要:** When applied directly in an end-to-end manner to medical follow-up tasks, Large Language Models (LLMs) often suffer from uncontrolled dialog flow and inaccurate information extraction due to the complexity of follow-up forms. To address this limitation, we designed and compared two follow-up chatbot systems: an end-to-end LLM-based system (control group) and a modular pipeline with structured process control (experimental group). Experimental results show that while the end-to-end approach frequently fails on lengthy and complex forms, our modular method-built on task decomposition, semantic clustering, and flow management-substantially improves dialog stability and extraction accuracy. Moreover, it reduces the number of dialogue turns by 46.73% and lowers token consumption by 80% to 87.5%. These findings highlight the necessity of integrating external control mechanisms when deploying LLMs in high-stakes medical follow-up scenarios.
>
---
#### [new 017] Exploring Zero-Shot ACSA with Unified Meaning Representation in Chain-of-Thought Prompting
- **分类: cs.CL**

- **简介: 该论文研究零样本方面类别情感分析（ACSA）任务，旨在缓解新领域标注数据稀缺问题。提出基于统一语义表示（UMR）的思维链提示方法，在Qwen3和Gemini模型上验证其有效性，发现UMR效果具模型依赖性，需进一步探索泛化性。**

- **链接: [https://arxiv.org/pdf/2512.19651v1](https://arxiv.org/pdf/2512.19651v1)**

> **作者:** Filippos Ventirozos; Peter Appleby; Matthew Shardlow
>
> **备注:** 9 pages, 3 figures, 3 tables
>
> **摘要:** Aspect-Category Sentiment Analysis (ACSA) provides granular insights by identifying specific themes within reviews and their associated sentiment. While supervised learning approaches dominate this field, the scarcity and high cost of annotated data for new domains present significant barriers. We argue that leveraging large language models (LLMs) in a zero-shot setting is a practical alternative where resources for data annotation are limited. In this work, we propose a novel Chain-of-Thought (CoT) prompting technique that utilises an intermediate Unified Meaning Representation (UMR) to structure the reasoning process for the ACSA task. We evaluate this UMR-based approach against a standard CoT baseline across three models (Qwen3-4B, Qwen3-8B, and Gemini-2.5-Pro) and four diverse datasets. Our findings suggest that UMR effectiveness may be model-dependent. Whilst preliminary results indicate comparable performance for mid-sized models such as Qwen3-8B, these observations warrant further investigation, particularly regarding the potential applicability to smaller model architectures. Further research is required to establish the generalisability of these findings across different model scales.
>
---
#### [new 018] A Large Language Model Based Method for Complex Logical Reasoning over Knowledge Graphs
- **分类: cs.CL**

- **简介: 该论文面向知识图谱（KG）上的复杂一阶逻辑（FOL）查询推理任务，旨在解决现有嵌入方法在高复杂度、多算子、深层推理链场景下泛化能力弱的问题。提出ROG框架：结合查询感知的KG子图检索与大语言模型（LLM）链式逻辑推理，无需特定嵌入优化，显著提升复杂查询的推理效果。**

- **链接: [https://arxiv.org/pdf/2512.19092v1](https://arxiv.org/pdf/2512.19092v1)**

> **作者:** Ziyan Zhang; Chao Wang; Zhuo Chen; Lei Chen; Chiyi Li; Kai Song
>
> **摘要:** Reasoning over knowledge graphs (KGs) with first-order logic (FOL) queries is challenging due to the inherent incompleteness of real-world KGs and the compositional complexity of logical query structures. Most existing methods rely on embedding entities and relations into continuous geometric spaces and answer queries via differentiable set operations. While effective for simple query patterns, these approaches often struggle to generalize to complex queries involving multiple operators, deeper reasoning chains, or heterogeneous KG schemas. We propose ROG (Reasoning Over knowledge Graphs with large language models), an ensemble-style framework that combines query-aware KG neighborhood retrieval with large language model (LLM)-based chain-of-thought reasoning. ROG decomposes complex FOL queries into sequences of simpler sub-queries, retrieves compact, query-relevant subgraphs as contextual evidence, and performs step-by-step logical inference using an LLM, avoiding the need for task-specific embedding optimization. Experiments on standard KG reasoning benchmarks demonstrate that ROG consistently outperforms strong embedding-based baselines in terms of mean reciprocal rank (MRR), with particularly notable gains on high-complexity query types. These results suggest that integrating structured KG retrieval with LLM-driven logical reasoning offers a robust and effective alternative for complex KG reasoning tasks.
>
---
#### [new 019] QuCo-RAG: Quantifying Uncertainty from the Pre-training Corpus for Dynamic Retrieval-Augmented Generation
- **分类: cs.CL; cs.IR**

- **简介: 该论文提出QuCo-RAG，解决动态RAG中LLM置信度不可靠导致的幻觉问题。它基于预训练语料库统计（低频实体识别与共现验证）量化不确定性，实现模型无关的动态检索触发，在多跳与生物医学QA上显著提升准确率。**

- **链接: [https://arxiv.org/pdf/2512.19134v1](https://arxiv.org/pdf/2512.19134v1)**

> **作者:** Dehai Min; Kailin Zhang; Tongtong Wu; Lu Cheng
>
> **摘要:** Dynamic Retrieval-Augmented Generation adaptively determines when to retrieve during generation to mitigate hallucinations in large language models (LLMs). However, existing methods rely on model-internal signals (e.g., logits, entropy), which are fundamentally unreliable because LLMs are typically ill-calibrated and often exhibit high confidence in erroneous outputs. We propose QuCo-RAG, which shifts from subjective confidence to objective statistics computed from pre-training data. Our method quantifies uncertainty through two stages: (1) before generation, we identify low-frequency entities indicating long-tail knowledge gaps; (2) during generation, we verify entity co-occurrence in the pre-training corpus, where zero co-occurrence often signals hallucination risk. Both stages leverage Infini-gram for millisecond-latency queries over 4 trillion tokens, triggering retrieval when uncertainty is high. Experiments on multi-hop QA benchmarks show QuCo-RAG achieves EM gains of 5--12 points over state-of-the-art baselines with OLMo-2 models, and transfers effectively to models with undisclosed pre-training data (Llama, Qwen, GPT), improving EM by up to 14 points. Domain generalization on biomedical QA further validates the robustness of our paradigm. These results establish corpus-grounded verification as a principled, practically model-agnostic paradigm for dynamic RAG. Our code is publicly available at https://github.com/ZhishanQ/QuCo-RAG.
>
---
#### [new 020] LIR$^3$AG: A Lightweight Rerank Reasoning Strategy Framework for Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文面向RAG多跳问答任务，旨在解决推理模型高计算开销问题。提出轻量级重排序推理框架LiR³AG，通过重构检索证据为推理链，使小模型（8B）无需推理能力即可复用推理策略，在显著降低token消耗（98%）和延迟（58.6%）的同时，性能反超大推理模型（32B）。**

- **链接: [https://arxiv.org/pdf/2512.18329v1](https://arxiv.org/pdf/2512.18329v1)**

> **作者:** Guo Chen; Junjie Huang; Huaijin Xie; Fei Sun; Tao Jia
>
> **备注:** AAAI2026
>
> **摘要:** Retrieval-Augmented Generation (RAG) effectively enhances Large Language Models (LLMs) by incorporating retrieved external knowledge into the generation process. Reasoning models improve LLM performance in multi-hop QA tasks, which require integrating and reasoning over multiple pieces of evidence across different documents to answer a complex question. However, they often introduce substantial computational costs, including increased token consumption and inference latency. To better understand and mitigate this trade-off, we conduct a comprehensive study of reasoning strategies for reasoning models in RAG multi-hop QA tasks. Our findings reveal that reasoning models adopt structured strategies to integrate retrieved and internal knowledge, primarily following two modes: Context-Grounded Reasoning, which relies directly on retrieved content, and Knowledge-Reconciled Reasoning, which resolves conflicts or gaps using internal knowledge. To this end, we propose a novel Lightweight Rerank Reasoning Strategy Framework for RAG (LiR$^3$AG) to enable non-reasoning models to transfer reasoning strategies by restructuring retrieved evidence into coherent reasoning chains. LiR$^3$AG significantly reduce the average 98% output tokens overhead and 58.6% inferencing time while improving 8B non-reasoning model's F1 performance ranging from 6.2% to 22.5% to surpass the performance of 32B reasoning model in RAG, offering a practical and efficient path forward for RAG systems.
>
---
#### [new 021] From Word to World: Can Large Language Models be Implicit Text-based World Models?
- **分类: cs.CL**

- **简介: 该论文探究大语言模型（LLM）能否作为隐式的、基于文本的世界模型，以提升智能体强化学习效率。它在文本环境中构建三层评估框架（保真度、可扩展性、代理效用），验证LLM世界模型在状态一致性、数据/规模可扩展性及提升代理性能（如动作验证、轨迹生成）方面的有效性与边界条件。**

- **链接: [https://arxiv.org/pdf/2512.18832v1](https://arxiv.org/pdf/2512.18832v1)**

> **作者:** Yixia Li; Hongru Wang; Jiahao Qiu; Zhenfei Yin; Dongdong Zhang; Cheng Qian; Zeping Li; Pony Ma; Guanhua Chen; Heng Ji; Mengdi Wang
>
> **摘要:** Agentic reinforcement learning increasingly relies on experience-driven scaling, yet real-world environments remain non-adaptive, limited in coverage, and difficult to scale. World models offer a potential way to improve learning efficiency through simulated experience, but it remains unclear whether large language models can reliably serve this role and under what conditions they meaningfully benefit agents. We study these questions in text-based environments, which provide a controlled setting to reinterpret language modeling as next-state prediction under interaction. We introduce a three-level framework for evaluating LLM-based world models: (i) fidelity and consistency, (ii) scalability and robustness, and (iii) agent utility. Across five representative environments, we find that sufficiently trained world models maintain coherent latent state, scale predictably with data and model size, and improve agent performance via action verification, synthetic trajectory generation, and warm-starting reinforcement learning. Meanwhile, these gains depend critically on behavioral coverage and environment complexity, delineating clear boundry on when world modeling effectively supports agent learning.
>
---
#### [new 022] MauBERT: Universal Phonetic Inductive Biases for Few-Shot Acoustic Units Discovery
- **分类: cs.CL; eess.AS**

- **简介: 该论文提出MauBERT，属语音表征学习任务，旨在解决多语言语音模型跨语言泛化弱、语音单元发现样本效率低的问题。通过在55种语言上结合发音特征监督继续预训练HuBERT，提升音素判别力与零样本迁移能力。**

- **链接: [https://arxiv.org/pdf/2512.19612v1](https://arxiv.org/pdf/2512.19612v1)**

> **作者:** Angelo Ortiz Tandazo; Manel Khentout; Youssef Benchekroun; Thomas Hueber; Emmanuel Dupoux
>
> **摘要:** This paper introduces MauBERT, a multilingual extension of HuBERT that leverages articulatory features for robust cross-lingual phonetic representation learning. We continue HuBERT pre-training with supervision based on a phonetic-to-articulatory feature mapping in 55 languages. Our models learn from multilingual data to predict articulatory features or phones, resulting in language-independent representations that capture multilingual phonetic properties. Through comprehensive ABX discriminability testing, we show MauBERT models produce more context-invariant representations than state-of-the-art multilingual self-supervised learning models. Additionally, the models effectively adapt to unseen languages and casual speech with minimal self-supervised fine-tuning (10 hours of speech). This establishes an effective approach for instilling linguistic inductive biases in self-supervised speech models.
>
---
#### [new 023] LLM-based Few-Shot Early Rumor Detection with Imitation Agent
- **分类: cs.CL; cs.AI**

- **简介: 该论文面向早期谣言检测（EARD）任务，解决数据稀缺下难以及时准确识别谣言的难题。提出基于LLM与模仿代理的少样本框架：代理自主决策最早检测时点，LLM免训练担当谣言判别器，仅需训练轻量代理，显著提升准确率与检测早性。**

- **链接: [https://arxiv.org/pdf/2512.18352v1](https://arxiv.org/pdf/2512.18352v1)**

> **作者:** Fengzhu Zeng; Qian Shao; Ling Cheng; Wei Gao; Shih-Fen Cheng; Jing Ma; Cheng Niu
>
> **摘要:** Early Rumor Detection (EARD) aims to identify the earliest point at which a claim can be accurately classified based on a sequence of social media posts. This is especially challenging in data-scarce settings. While Large Language Models (LLMs) perform well in few-shot NLP tasks, they are not well-suited for time-series data and are computationally expensive for both training and inference. In this work, we propose a novel EARD framework that combines an autonomous agent and an LLM-based detection model, where the agent acts as a reliable decision-maker for \textit{early time point determination}, while the LLM serves as a powerful \textit{rumor detector}. This approach offers the first solution for few-shot EARD, necessitating only the training of a lightweight agent and allowing the LLM to remain training-free. Extensive experiments on four real-world datasets show our approach boosts performance across LLMs and surpasses existing EARD methods in accuracy and earliness.
>
---
#### [new 024] Towards Reasoning-Preserving Unlearning in Multimodal Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文聚焦多模态大语言模型（RMLLM）的机器遗忘任务，旨在解决推理过程中的敏感信息泄露与推理能力退化并存的难题。作者构建首个专用基准RMLLMU-Bench，并提出无需训练的推理保护型遗忘方法R-MUSE，在有效遗忘的同时保留推理能力。**

- **链接: [https://arxiv.org/pdf/2512.17911v1](https://arxiv.org/pdf/2512.17911v1)**

> **作者:** Hongji Li; Junchi yao; Manjiang Yu; Priyanka Singh; Xue Li; Di Wang; Lijie Hu
>
> **摘要:** Machine unlearning aims to erase requested data from trained models without full retraining. For Reasoning Multimodal Large Language Models (RMLLMs), this is uniquely challenging: intermediate chain-of-thought steps can still leak sensitive information even when final answers are forgotten, and overly aggressive interventions easily damage general reasoning ability. Yet no benchmark jointly evaluates how well unlearning methods suppress reasoning-level leakage while preserving reasoning competence. We address this gap with RMLLMU-Bench, the first benchmark for RMLLM unlearning that extends standard forgetting metrics with dedicated measures of reasoning leakage and reasoning retention. A systematic evaluation on RMLLMU-Bench reveals that existing unlearning methods for MLLMs and Large (Language) Reasoning Models (LRMs) either leave substantial leakage in the reasoning process or severely degrade reasoning performance. To address these gaps, we propose R-MUSE (Reasoning-preserving MLLM Unlearning via Subspace guidance and Adaptive Steering), a training-free and inference-time intervention framework that steers internal representations to forget both answers and reasoning traces while explicitly preserving general reasoning. Experiments on RMLLMU-Bench demonstrate that R-MUSE achieves a substantially better balance between effective forgetting and reasoning retention.
>
---
#### [new 025] Exploring the features used for summary evaluation by Human and GPT
- **分类: cs.CL; cs.AI**

- **简介: 该论文属摘要评估任务，旨在探究人类与GPT在摘要评价中依赖的特征。它分析统计与机器学习指标，识别双方对齐的评估特征，并验证用人类常用指标引导GPT可提升其判断与人类一致性。**

- **链接: [https://arxiv.org/pdf/2512.19620v1](https://arxiv.org/pdf/2512.19620v1)**

> **作者:** Zahra Sadeghi; Evangelos Milios; Frank Rudzicz
>
> **摘要:** Summary assessment involves evaluating how well a generated summary reflects the key ideas and meaning of the source text, requiring a deep understanding of the content. Large Language Models (LLMs) have been used to automate this process, acting as judges to evaluate summaries with respect to the original text. While previous research investigated the alignment between LLMs and Human responses, it is not yet well understood what properties or features are exploited by them when asked to evaluate based on a particular quality dimension, and there has not been much attention towards mapping between evaluation scores and metrics. In this paper, we address this issue and discover features aligned with Human and Generative Pre-trained Transformers (GPTs) responses by studying statistical and machine learning metrics. Furthermore, we show that instructing GPTs to employ metrics used by Human can improve their judgment and conforming them better with human responses.
>
---
#### [new 026] ReGal: A First Look at PPO-based Legal AI for Judgment Prediction and Summarization in India
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出ReGal框架，探索PPO强化学习在印度法律AI中的应用，聚焦判决预测与解释、法律文档摘要两大任务；旨在解决RL在法律文本中奖励对齐、语言复杂性及领域适配等挑战，虽性能未超监督模型，但为可解释、自适应法律AI提供新路径。**

- **链接: [https://arxiv.org/pdf/2512.18014v1](https://arxiv.org/pdf/2512.18014v1)**

> **作者:** Shubham Kumar Nigam; Tanuj Tyagi; Siddharth Shukla; Aditya Kumar Guru; Balaramamahanthi Deepak Patnaik; Danush Khanna; Noel Shallum; Kripabandhu Ghosh; Arnab Bhattacharya
>
> **备注:** Accepted in AILaw @ AAAI 2026 conference
>
> **摘要:** This paper presents an early exploration of reinforcement learning methodologies for legal AI in the Indian context. We introduce Reinforcement Learning-based Legal Reasoning (ReGal), a framework that integrates Multi-Task Instruction Tuning with Reinforcement Learning from AI Feedback (RLAIF) using Proximal Policy Optimization (PPO). Our approach is evaluated across two critical legal tasks: (i) Court Judgment Prediction and Explanation (CJPE), and (ii) Legal Document Summarization. Although the framework underperforms on standard evaluation metrics compared to supervised and proprietary models, it provides valuable insights into the challenges of applying RL to legal texts. These challenges include reward model alignment, legal language complexity, and domain-specific adaptation. Through empirical and qualitative analysis, we demonstrate how RL can be repurposed for high-stakes, long-document tasks in law. Our findings establish a foundation for future work on optimizing legal reasoning pipelines using reinforcement learning, with broader implications for building interpretable and adaptive legal AI systems.
>
---
#### [new 027] CienaLLM: Generative Climate-Impact Extraction from News Articles with Autoregressive LLMs
- **分类: cs.CL**

- **简介: 该论文提出CienaLLM，一种基于开源大语言模型的零样本生成式信息抽取框架，用于从新闻中结构化提取气候灾害（如干旱）的社会经济影响。它通过提示工程、多步流水线和响应解析提升鲁棒性，支持跨灾害、语言和领域的快速适配，无需重训练。**

- **链接: [https://arxiv.org/pdf/2512.19305v1](https://arxiv.org/pdf/2512.19305v1)**

> **作者:** Javier Vela-Tambo; Jorge Gracia; Fernando Dominguez-Castro
>
> **摘要:** Understanding and monitoring the socio-economic impacts of climate hazards requires extracting structured information from heterogeneous news articles on a large scale. To that end, we have developed CienaLLM, a modular framework based on schema-guided Generative Information Extraction. CienaLLM uses open-weight Large Language Models for zero-shot information extraction from news articles, and supports configurable prompts and output schemas, multi-step pipelines, and cloud or on-premise inference. To systematically assess how the choice of LLM family, size, precision regime, and prompting strategy affect performance, we run a large factorial study in models, precisions, and prompt engineering techniques. An additional response parsing step nearly eliminates format errors while preserving accuracy; larger models deliver the strongest and most stable performance, while quantization offers substantial efficiency gains with modest accuracy trade-offs; and prompt strategies show heterogeneous, model-specific effects. CienaLLM matches or outperforms the supervised baseline in accuracy for extracting drought impacts from Spanish news, although at a higher inference cost. While evaluated in droughts, the schema-driven and model-agnostic design is suitable for adapting to related information extraction tasks (e.g., other hazards, sectors, or languages) by editing prompts and schemas rather than retraining. We release code, configurations, and schemas to support reproducible use.
>
---
#### [new 028] ChemATP: A Training-Free Chemical Reasoning Framework for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出ChemATP，一种无需训练的化学推理框架，旨在解决大语言模型（LLM）在分子科学中缺乏细粒度原子级先验知识的问题。它构建首个原子级文本知识库，使冻结LLM能动态检索并推理化学知识，兼顾可解释性、适应性与通用推理能力。**

- **链接: [https://arxiv.org/pdf/2512.19240v1](https://arxiv.org/pdf/2512.19240v1)**

> **作者:** Mingxu Zhang; Dazhong Shen; Qi Zhang; Ying Sun
>
> **摘要:** Large Language Models (LLMs) exhibit strong general reasoning but struggle in molecular science due to the lack of explicit chemical priors in standard string representations. Current solutions face a fundamental dilemma. Training-based methods inject priors into parameters, but this static coupling hinders rapid knowledge updates and often compromises the model's general reasoning capabilities. Conversely, existing training-free methods avoid these issues but rely on surface-level prompting, failing to provide the fine-grained atom-level priors essential for precise chemical reasoning. To address this issue, we introduce ChemATP, a framework that decouples chemical knowledge from the reasoning engine. By constructing the first atom-level textual knowledge base, ChemATP enables frozen LLMs to explicitly retrieve and reason over this information dynamically. This architecture ensures interpretability and adaptability while preserving the LLM's intrinsic general intelligence. Experiments show that ChemATP significantly outperforms training-free baselines and rivals state-of-the-art training-based models, demonstrating that explicit prior injection is a competitive alternative to implicit parameter updates.
>
---
#### [new 029] Kunnafonidilaw ka Cadeau: an ASR dataset of present-day Bambara
- **分类: cs.CL**

- **简介: 该论文面向低资源语言ASR任务，针对巴马纳语（Bambara）缺乏真实语音数据的问题，构建了160小时含噪声、语码转换等真实特征的Kunkado数据集；提出转录归一化与微调方法，显著降低WER，并开源数据与模型。**

- **链接: [https://arxiv.org/pdf/2512.19400v1](https://arxiv.org/pdf/2512.19400v1)**

> **作者:** Yacouba Diarra; Panga Azazia Kamate; Nouhoum Souleymane Coulibaly; Michael Leventhal
>
> **备注:** 7 pages, 2 figures
>
> **摘要:** We present Kunkado, a 160-hour Bambara ASR dataset compiled from Malian radio archives to capture present-day spontaneous speech across a wide range of topics. It includes code-switching, disfluencies, background noise, and overlapping speakers that practical ASR systems encounter in real-world use. We finetuned Parakeet-based models on a 33.47-hour human-reviewed subset and apply pragmatic transcript normalization to reduce variability in number formatting, tags, and code-switching annotations. Evaluated on two real-world test sets, finetuning with Kunkado reduces WER from 44.47\% to 37.12\% on one and from 36.07\% to 32.33\% on the other. In human evaluation, the resulting model also outperforms a comparable system with the same architecture trained on 98 hours of cleaner, less realistic speech. We release the data and models to support robust ASR for predominantly oral languages.
>
---
#### [new 030] AraMix: Recycling, Refiltering, and Deduplicating to Deliver the Largest Arabic Pretraining Corpus
- **分类: cs.CL**

- **简介: 该论文属数据集构建任务，旨在解决阿拉伯语预训练数据冗余与质量低的问题。作者提出AraMix方法：复用7个现有阿拉伯语数据集，定制化质量过滤，并进行跨数据集MinHash及句子级去重，最终构建出1780亿token的高质量阿拉伯语预训练语料库。**

- **链接: [https://arxiv.org/pdf/2512.18834v1](https://arxiv.org/pdf/2512.18834v1)**

> **作者:** Sultan Alrashed; Francesco Orabona
>
> **备注:** Initial version, without pretraining experiments
>
> **摘要:** We present AraMix, a deduplicated Arabic pretraining corpus containing approximately 178 billion tokens across 179 million documents. Rather than scraping the web again, AraMix demonstrates that substantial value lies in systematically reusing and curating existing pretraining datasets: we combine seven publicly available Arabic web datasets, apply quality filtering designed specifically for Arabic text to re-filter some datasets, and perform cross-dataset deduplication, both MinHash and sentence-level. This approach reveals that nearly 60% of tokens across these independently collected corpora are duplicates, redundancy that any new scraping efforts will reproduce. Our work suggests that for lower resource languages, investment in curation pipelines for existing data yields greater returns than additional web crawls, an approach that allowed us to curate the largest heavily filtered publicly available Arabic pretraining corpus.
>
---
#### [new 031] Learning to Prioritize IT Tickets: A Comparative Evaluation of Embedding-based Approaches and Fine-Tuned Transformer Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文面向IT服务管理中的工单优先级排序任务，旨在解决文本噪声大、标注主观、类别失衡等挑战。作者对比了嵌入式流水线与微调多语言Transformer模型，发现后者显著更优（F1=78.5%，κ≈0.80），证明领域适配Transformer更有效。**

- **链接: [https://arxiv.org/pdf/2512.17916v1](https://arxiv.org/pdf/2512.17916v1)**

> **作者:** Minh Tri LÊ; Ali Ait-Bachir
>
> **备注:** 12 pages
>
> **摘要:** Prioritizing service tickets in IT Service Management (ITSM) is critical for operational efficiency but remains challenging due to noisy textual inputs, subjective writing styles, and pronounced class imbalance. We evaluate two families of approaches for ticket prioritization: embedding-based pipelines that combine dimensionality reduction, clustering, and classical classifiers, and a fine-tuned multilingual transformer that processes both textual and numerical features. Embedding-based methods exhibit limited generalization across a wide range of thirty configurations, with clustering failing to uncover meaningful structures and supervised models highly sensitive to embedding quality. In contrast, the proposed transformer model achieves substantially higher performance, with an average F1-score of 78.5% and weighted Cohen's kappa values of nearly 0.80, indicating strong alignment with true labels. These results highlight the limitations of generic embeddings for ITSM data and demonstrate the effectiveness of domain-adapted transformer architectures for operational ticket prioritization.
>
---
#### [new 032] JEPA-Reasoner: Decoupling Latent Reasoning from Token Generation
- **分类: cs.CL**

- **简介: 该论文提出JEPA-Reasoner，属AI推理与生成任务，旨在解决JEPA缺乏生成能力、现有潜空间推理仍依赖易出错的逐token生成的问题。工作包括：增强JEPA以支持潜空间推理，解耦推理与生成，并引入独立Talker模型输出自然语言。**

- **链接: [https://arxiv.org/pdf/2512.19171v1](https://arxiv.org/pdf/2512.19171v1)**

> **作者:** Bingyang Kelvin Liu; Ziyu Patrick Chen
>
> **摘要:** While Joint-Embedding Predictive Architecture (JEPA) has emerged as a powerful architecture for learning rich latent representations, it fundamentally lacks generative abilities. Meanwhile, latent space reasoning attempts for Transformer models like COCONUT do improve performance, but they ultimately rely on token-by-token generation, which still accumulates compounding error and relies on context information to gain reasoning insights. To address these limitations, we propose JEPA-Reasoner, a novel JEPA model enhanced with generative ability that reasons in latent space. We augment it with a separate action-taker model, Talker, to produce human-readable sentences. Our approach demonstrates that decoupling latent space reasoning and token generation enables JEPA-Reasoner to produce mixed latent vectors that might lay the foundation for multi-threaded reasoning, while performing autoregressive generation with superior robustness to compounding error.
>
---
#### [new 033] Supplementary Resources and Analysis for Automatic Speech Recognition Systems Trained on the Loquacious Dataset
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语音识别（ASR）任务，旨在提升Loquacious新数据集的可用性与基准测试能力。针对其缺乏配套语言模型和发音资源的问题，作者开源了n-gram语言模型、G2P模型及发音词典，并在多种ASR架构上验证效果。**

- **链接: [https://arxiv.org/pdf/2512.17915v1](https://arxiv.org/pdf/2512.17915v1)**

> **作者:** Nick Rossenbach; Robin Schmitt; Tina Raissi; Simon Berger; Larissa Kleppel; Ralf Schlüter
>
> **摘要:** The recently published Loquacious dataset aims to be a replacement for established English automatic speech recognition (ASR) datasets such as LibriSpeech or TED-Lium. The main goal of the Loquacious dataset is to provide properly defined training and test partitions across many acoustic and language domains, with an open license suitable for both academia and industry. To further promote the benchmarking and usability of this new dataset, we present additional resources in the form of n-gram language models (LMs), a grapheme-to-phoneme (G2P) model and pronunciation lexica, with open and public access. Utilizing those additional resources we show experimental results across a wide range of ASR architectures with different label units and topologies. Our initial experimental results indicate that the Loquacious dataset offers a valuable study case for a variety of common challenges in ASR.
>
---
#### [new 034] Increasing the Thinking Budget is Not All You Need
- **分类: cs.CL**

- **简介: 该论文属AI推理优化任务，旨在探究“思考预算”（推理步数）对大模型性能的影响。作者系统分析其与自一致性、自反思等配置的交互，发现单纯增加思考预算并非最优计算利用方式，替代策略可更高效提升准确率。**

- **链接: [https://arxiv.org/pdf/2512.19585v1](https://arxiv.org/pdf/2512.19585v1)**

> **作者:** Ignacio Iacobacci; Zhaozhi Qian; Faroq AL-Tam; Muhammad AL-Qurishi; Riad Souissi
>
> **备注:** 4 pages, 4 figures, 3 tables
>
> **摘要:** Recently, a new wave of thinking-capable Large Language Models has emerged, demonstrating exceptional capabilities across a wide range of reasoning benchmarks. Early studies have begun to explore how the amount of compute in terms of the length of the reasoning process, the so-called thinking budget, impacts model performance. In this work, we propose a systematic investigation of the thinking budget as a key parameter, examining its interaction with various configurations such as self-consistency, reflection, and others. Our goal is to provide an informative, balanced comparison framework that considers both performance outcomes and computational cost. Among our findings, we discovered that simply increasing the thinking budget is not the most effective use of compute. More accurate responses can instead be achieved through alternative configurations, such as self-consistency and self-reflection.
>
---
#### [new 035] Generalization Gaps in Political Fake News Detection: An Empirical Study on the LIAR Dataset
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究政治假新闻检测任务，旨在揭示纯文本模型在LIAR数据集上的泛化瓶颈。通过系统评估9种算法，发现存在性能上限（Weighted F1≤0.32）和巨大泛化差距（如树模型训练99%→测试25%），证明模型复杂度非关键，需引入外部知识而非单纯堆叠参数。**

- **链接: [https://arxiv.org/pdf/2512.18533v1](https://arxiv.org/pdf/2512.18533v1)**

> **作者:** S Mahmudul Hasan; Shaily Roy; Akib Jawad Nafis
>
> **摘要:** The proliferation of linguistically subtle political disinformation poses a significant challenge to automated fact-checking systems. Despite increasing emphasis on complex neural architectures, the empirical limits of text-only linguistic modeling remain underexplored. We present a systematic diagnostic evaluation of nine machine learning algorithms on the LIAR benchmark. By isolating lexical features (Bag-of-Words, TF-IDF) and semantic embeddings (GloVe), we uncover a hard "Performance Ceiling", with fine-grained classification not exceeding a Weighted F1-score of 0.32 across models. Crucially, a simple linear SVM (Accuracy: 0.624) matches the performance of pre-trained Transformers such as RoBERTa (Accuracy: 0.620), suggesting that model capacity is not the primary bottleneck. We further diagnose a massive "Generalization Gap" in tree-based ensembles, which achieve more than 99% training accuracy but collapse to approximately 25% on test data, indicating reliance on lexical memorization rather than semantic inference. Synthetic data augmentation via SMOTE yields no meaningful gains, confirming that the limitation is semantic (feature ambiguity) rather than distributional. These findings indicate that for political fact-checking, increasing model complexity without incorporating external knowledge yields diminishing returns.
>
---
#### [new 036] From Speech to Subtitles: Evaluating ASR Models in Subtitling Italian Television Programs
- **分类: cs.CL**

- **简介: 该论文属ASR在字幕生成中的应用任务，旨在解决意大利语电视节目自动字幕准确率不足的问题。作者评估了4种主流ASR模型在50小时意语电视数据上的表现，对比人工字幕，验证其在人机协同（HITL）工作流中的实用价值，并设计了生产级云基础设施。**

- **链接: [https://arxiv.org/pdf/2512.19161v1](https://arxiv.org/pdf/2512.19161v1)**

> **作者:** Alessandro Lucca; Francesco Pierri
>
> **摘要:** Subtitles are essential for video accessibility and audience engagement. Modern Automatic Speech Recognition (ASR) systems, built upon Encoder-Decoder neural network architectures and trained on massive amounts of data, have progressively reduced transcription errors on standard benchmark datasets. However, their performance in real-world production environments, particularly for non-English content like long-form Italian videos, remains largely unexplored. This paper presents a case study on developing a professional subtitling system for an Italian media company. To inform our system design, we evaluated four state-of-the-art ASR models (Whisper Large v2, AssemblyAI Universal, Parakeet TDT v3 0.6b, and WhisperX) on a 50-hour dataset of Italian television programs. The study highlights their strengths and limitations, benchmarking their performance against the work of professional human subtitlers. The findings indicate that, while current models cannot meet the media industry's accuracy needs for full autonomy, they can serve as highly effective tools for enhancing human productivity. We conclude that a human-in-the-loop (HITL) approach is crucial and present the production-grade, cloud-based infrastructure we designed to support this workflow.
>
---
#### [new 037] A Comparative Study of Light-weight Language Models for PII Masking and their Deployment for Real Conversational Texts
- **分类: cs.CL**

- **简介: 该论文属隐私保护任务，旨在解决轻量级模型能否替代大模型进行PII掩码的问题。作者微调T5-small与Mistral-Instruct，在标准化数据集上对比性能，验证二者可媲美前沿LLM；分析准确率、鲁棒性与延迟权衡，并部署T5至Discord实时红acting。**

- **链接: [https://arxiv.org/pdf/2512.18608v1](https://arxiv.org/pdf/2512.18608v1)**

> **作者:** Prabigya Acharya; Liza Shrestha
>
> **摘要:** Automated masking of Personally Identifiable Information (PII) is critical for privacy-preserving conversational systems. While current frontier large language models demonstrate strong PII masking capabilities, concerns about data handling and computational costs motivate exploration of whether lightweight models can achieve comparable performance. We compare encoder-decoder and decoder-only architectures by fine-tuning T5-small and Mistral-Instruct-v0.3 on English datasets constructed from the AI4Privacy benchmark. We create different dataset variants to study label standardization and PII representation, covering 24 standardized PII categories and higher-granularity settings. Evaluation using entity-level and character-level metrics, type accuracy, and exact match shows that both lightweight models achieve performance comparable to frontier LLMs for PII masking tasks. Label normalization consistently improves performance across architectures. Mistral achieves higher F1 and recall with greater robustness across PII types but incurs significantly higher generation latency. T5, while less robust in conversational text, offers more controllable structured outputs and lower inference cost, motivating its use in a real-time Discord bot for real-world PII redaction. Evaluation on live messages reveals performance degradation under informal inputs. These results clarify trade-offs between accuracy, robustness, and computational efficiency, demonstrating that lightweight models can provide effective PII masking while addressing data handling concerns associated with frontier LLMs.
>
---
#### [new 038] Toward Human-Centered AI-Assisted Terminology Work
- **分类: cs.CL**

- **简介: 该论文属人机协同术语学研究任务，旨在解决AI滥用导致专业自主性削弱、偏见加剧与语言多样性流失问题。作者提出以“增强型术语专家”“伦理AI”“以人为本设计”为维度的人本AI框架，强调AI赋能而非替代术语工作者，确保人类主导、价值导向与多样性保护。**

- **链接: [https://arxiv.org/pdf/2512.18859v1](https://arxiv.org/pdf/2512.18859v1)**

> **作者:** Antonio San Martin
>
> **摘要:** The rapid diffusion of generative artificial intelligence is transforming terminology work. While this technology promises gains in efficiency, its unstructured adoption risks weakening professional autonomy, amplifying bias, and eroding linguistic and conceptual diversity. This paper argues that a human-centered approach to artificial intelligence has become a necessity for terminology work. Building on research in artificial intelligence and translation studies, it proposes a human-centered framework that conceptualizes artificial intelligence as a means of amplifying the terminologist's capabilities, rather than replacing them. The framework is organized around three interrelated dimensions: the augmented terminologist, ethical AI, and human-centered design. Together, these dimensions emphasize the compatibility of high automation with strong human control, the central role of terminologists in bias mitigation, and the importance of designing AI tools and workflows around the needs, values, and well-being of the terminologist. The paper concludes by stressing that current choices in AI adoption will shape not only terminological practice, but also the preservation of accuracy, adequacy, and diversity in terminology and specialized knowledge.
>
---
#### [new 039] SRS-Stories: Vocabulary-constrained multilingual story generation for language learning
- **分类: cs.CL**

- **简介: 该论文属自然语言生成任务，旨在解决语言学习中个性化、词汇可控的故事生成问题。作者提出SRS-Stories方法，利用大语言模型生成仅含学习者已知词汇的多语种故事，结合间隔重复系统优化新旧词汇的呈现与复习，并在英、中、波三种语言上验证其优于传统约束束搜索的效果。**

- **链接: [https://arxiv.org/pdf/2512.18362v1](https://arxiv.org/pdf/2512.18362v1)**

> **作者:** Wiktor Kamzela; Mateusz Lango; Ondrej Dusek
>
> **备注:** EMNLP 2025
>
> **摘要:** In this paper, we use large language models to generate personalized stories for language learners, using only the vocabulary they know. The generated texts are specifically written to teach the user new vocabulary by simply reading stories where it appears in context, while at the same time seamlessly reviewing recently learned vocabulary. The generated stories are enjoyable to read and the vocabulary reviewing/learning is optimized by a Spaced Repetition System. The experiments are conducted in three languages: English, Chinese and Polish, evaluating three story generation methods and three strategies for enforcing lexical constraints. The results show that the generated stories are more grammatical, coherent, and provide better examples of word usage than texts generated by the standard constrained beam search approach
>
---
#### [new 040] Algerian Dialect
- **分类: cs.CL**

- **简介: 该论文构建了含45,000条YouTube评论的阿尔及利亚阿拉伯语方言情感标注数据集，涵盖五类情感标签及丰富元数据。属于自然语言处理中的情感分析任务，旨在解决阿尔及利亚方言资源稀缺问题，支撑方言NLP与社交媒体情感研究。**

- **链接: [https://arxiv.org/pdf/2512.19543v1](https://arxiv.org/pdf/2512.19543v1)**

> **作者:** Zakaria Benmounah; Abdennour Boulesnane
>
> **摘要:** We present Algerian Dialect, a large-scale sentiment-annotated dataset consisting of 45,000 YouTube comments written in Algerian Arabic dialect. The comments were collected from more than 30 Algerian press and media channels using the YouTube Data API. Each comment is manually annotated into one of five sentiment categories: very negative, negative, neutral, positive, and very positive. In addition to sentiment labels, the dataset includes rich metadata such as collection timestamps, like counts, video URLs, and annotation dates. This dataset addresses the scarcity of publicly available resources for Algerian dialect and aims to support research in sentiment analysis, dialectal Arabic NLP, and social media analytics. The dataset is publicly available on Mendeley Data under a CC BY 4.0 license at https://doi.org/10.17632/zzwg3nnhsz.2.
>
---
#### [new 041] Neologism Learning as a Parameter-Efficient Alternative to Fine-Tuning for Model Steering
- **分类: cs.CL**

- **简介: 该论文研究语言模型的行为引导任务，旨在以参数高效方式替代传统微调。提出“新词学习”（neologism learning）：训练少量新词表征特定概念，实现灵活、低开销的行为控制。实验表明其在相同设置下优于LoRA微调，并发现模型会自发生成新词。**

- **链接: [https://arxiv.org/pdf/2512.18551v1](https://arxiv.org/pdf/2512.18551v1)**

> **作者:** Sungjoon Park; Varun Ramamurthi; Owen Terry
>
> **摘要:** In language modeling, neologisms are new tokens trained to represent a concept not already included in a given model's vocabulary. Neologisms can be used to encourage specific behavior in models, for example by appending prompts with "Give me a neologism answer." Behavioral steering can also be achieved through fine-tuning, albeit with more compute and less flexibility: learning a neologism only trains d parameters and allows the user to still access the model's default behavior. We compare the performance of neologism learning against low-rank adaptation (LoRA) fine-tuning, finding that neologisms outperform fine-tuned models under a matched training setup (same data and hyperparameters). We also investigate self-verbalizations of neologisms, and observe that the model will occasionally make up its own new words when asked about a neologism.
>
---
#### [new 042] CTTA-T: Continual Test-Time Adaptation for Text Understanding via Teacher-Student with a Domain-aware and Generalized Teacher
- **分类: cs.CL**

- **简介: 该论文面向文本理解中的持续测试时自适应（CTTA）任务，解决测试阶段多未知领域顺序出现导致的误差累积与泛化弱问题。提出CTTA-T框架：采用域感知且泛化强的教师-学生结构，结合Dropout一致性精炼过滤预测，并用增量PCA动态累积跨域语义。**

- **链接: [https://arxiv.org/pdf/2512.18321v1](https://arxiv.org/pdf/2512.18321v1)**

> **作者:** Tianlun Liu; Zhiliang Tian; Zhen Huang; Xingzhi Zhou; Wanlong Yu; Tianle Liu; Feng Liu; Dongsheng Li
>
> **摘要:** Text understanding often suffers from domain shifts. To handle testing domains, domain adaptation (DA) is trained to adapt to a fixed and observed testing domain; a more challenging paradigm, test-time adaptation (TTA), cannot access the testing domain during training and online adapts to the testing samples during testing, where the samples are from a fixed domain. We aim to explore a more practical and underexplored scenario, continual test-time adaptation (CTTA) for text understanding, which involves a sequence of testing (unobserved) domains in testing. Current CTTA methods struggle in reducing error accumulation over domains and enhancing generalization to handle unobserved domains: 1) Noise-filtering reduces accumulated errors but discards useful information, and 2) accumulating historical domains enhances generalization, but it is hard to achieve adaptive accumulation. In this paper, we propose a CTTA-T (continual test-time adaptation for text understanding) framework adaptable to evolving target domains: it adopts a teacher-student framework, where the teacher is domain-aware and generalized for evolving domains. To improve teacher predictions, we propose a refine-then-filter based on dropout-driven consistency, which calibrates predictions and removes unreliable guidance. For the adaptation-generalization trade-off, we construct a domain-aware teacher by dynamically accumulating cross-domain semantics via incremental PCA, which continuously tracks domain shifts. Experiments show CTTA-T excels baselines.
>
---
#### [new 043] GenEnv: Difficulty-Aligned Co-Evolution Between LLM Agents and Environment Simulators
- **分类: cs.CL**

- **简介: 该论文提出GenEnv框架，解决LLM智能体训练中真实交互数据成本高、静态低效的问题。通过构建智能体与生成式环境模拟器的难度对齐协同进化机制，动态生成适配其能力的任务，实现高效、自适应的数据演化训练。**

- **链接: [https://arxiv.org/pdf/2512.19682v1](https://arxiv.org/pdf/2512.19682v1)**

> **作者:** Jiacheng Guo; Ling Yang; Peter Chen; Qixin Xiao; Yinjie Wang; Xinzhe Juan; Jiahao Qiu; Ke Shen; Mengdi Wang
>
> **备注:** Our codes are available at https://github.com/Gen-Verse/GenEnv
>
> **摘要:** Training capable Large Language Model (LLM) agents is critically bottlenecked by the high cost and static nature of real-world interaction data. We address this by introducing GenEnv, a framework that establishes a difficulty-aligned co-evolutionary game between an agent and a scalable, generative environment simulator. Unlike traditional methods that evolve models on static datasets, GenEnv instantiates a dataevolving: the simulator acts as a dynamic curriculum policy, continuously generating tasks specifically tailored to the agent's ``zone of proximal development''. This process is guided by a simple but effective $α$-Curriculum Reward, which aligns task difficulty with the agent's current capabilities. We evaluate GenEnv on five benchmarks, including API-Bank, ALFWorld, BFCL, Bamboogle, and TravelPlanner. Across these tasks, GenEnv improves agent performance by up to \textbf{+40.3\%} over 7B baselines and matches or exceeds the average performance of larger models. Compared to Gemini 2.5 Pro-based offline data augmentation, GenEnv achieves better performance while using 3.3$\times$ less data. By shifting from static supervision to adaptive simulation, GenEnv provides a data-efficient pathway for scaling agent capabilities.
>
---
#### [new 044] Does It Tie Out? Towards Autonomous Legal Agents in Venture Capital
- **分类: cs.CL**

- **简介: 该论文聚焦风投融资中的“资本表核验”任务，旨在解决现有法律AI无法可靠完成多文档推理、证据可追溯与确定性输出的难题。作者将其定义为法律AI真实基准，评估现有智能体性能，并提出一种世界模型架构以实现自动化核验。**

- **链接: [https://arxiv.org/pdf/2512.18658v1](https://arxiv.org/pdf/2512.18658v1)**

> **作者:** Pierre Colombo; Malik Boudiaf; Allyn Sweet; Michael Desa; Hongxi Wang; Kevin Candra; Syméon del Marmol
>
> **摘要:** Before closing venture capital financing rounds, lawyers conduct diligence that includes tying out the capitalization table: verifying that every security (for example, shares, options, warrants) and issuance term (for example, vesting schedules, acceleration triggers, transfer restrictions) is supported by large sets of underlying legal documentation. While LLMs continue to improve on legal benchmarks, specialized legal workflows, such as capitalization tie-out, remain out of reach even for strong agentic systems. The task requires multi-document reasoning, strict evidence traceability, and deterministic outputs that current approaches fail to reliably deliver. We characterize capitalization tie-out as an instance of a real-world benchmark for legal AI, analyze and compare the performance of existing agentic systems, and propose a world model architecture toward tie-out automation-and more broadly as a foundation for applied legal intelligence.
>
---
#### [new 045] GeoSense-AI: Fast Location Inference from Crisis Microblogs
- **分类: cs.CL; cs.SI**

- **简介: 该论文提出GeoSense-AI系统，解决危机微博客中实时、低延迟地理定位问题。任务是文本驱动的细粒度位置推断，替代稀疏的GPS标签。工作包括融合轻量NLP（分词、POS驱动地名识别、依存解析、NER）与地理知识库消歧，在保证高F1的同时实现数量级提速。**

- **链接: [https://arxiv.org/pdf/2512.18225v1](https://arxiv.org/pdf/2512.18225v1)**

> **作者:** Deepit Sapru
>
> **摘要:** This paper presents an applied AI pipeline for realtime geolocation from noisy microblog streams, unifying statistical hashtag segmentation, part-of-speech-driven proper-noun detection, dependency parsing around disaster lexicons, lightweight named-entity recognition, and gazetteer-grounded disambiguation to infer locations directly from text rather than sparse geotags. The approach operationalizes information extraction under streaming constraints, emphasizing low-latency NLP components and efficient validation against geographic knowledge bases to support situational awareness during emergencies. In head to head comparisons with widely used NER toolkits, the system attains strong F1 while being engineered for orders-of-magnitude faster throughput, enabling deployment in live crisis informatics settings. A production map interface demonstrates end-to-end AI functionality ingest, inference, and visualization--surfacing locational signals at scale for floods, outbreaks, and other fastmoving events. By prioritizing robustness to informal text and streaming efficiency, GeoSense-AI illustrates how domain-tuned NLP and knowledge grounding can elevate emergency response beyond conventional geo-tag reliance.
>
---
#### [new 046] Narrative Consolidation: Formulating a New Task for Unifying Multi-Perspective Accounts
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出“叙事整合”新任务，旨在融合多视角叙述（如福音书、证词），保持时间连贯性、完整性与细节互补性。针对传统多文档摘要忽视时序的问题，构建时序对齐事件图（TAEG），用中心性算法选择各事件最优表述并严格按时间排序，显著提升时序准确率与内容质量。**

- **链接: [https://arxiv.org/pdf/2512.18041v1](https://arxiv.org/pdf/2512.18041v1)**

> **作者:** Roger A. Finger; Eduardo G. Cortes; Sandro J. Rigo; Gabriel de O. Ramos
>
> **摘要:** Processing overlapping narrative documents, such as legal testimonies or historical accounts, often aims not for compression but for a unified, coherent, and chronologically sound text. Standard Multi-Document Summarization (MDS), with its focus on conciseness, fails to preserve narrative flow. This paper formally defines this challenge as a new NLP task: Narrative Consolidation, where the central objectives are chronological integrity, completeness, and the fusion of complementary details. To demonstrate the critical role of temporal structure in this task, we introduce Temporal Alignment Event Graph (TAEG), a graph structure that explicitly models chronology and event alignment. By applying a standard centrality algorithm to TAEG, our method functions as a version selection mechanism, choosing the most central representation of each event in its correct temporal position. In a study on the four Biblical Gospels, this structure-focused approach guarantees perfect temporal ordering (Kendall's Tau of 1.000) by design and dramatically improves content metrics (e.g., +357.2% in ROUGE-L F1). The success of this baseline method validates the formulation of Narrative Consolidation as a relevant task and establishes that an explicit temporal backbone is a fundamental component for its resolution.
>
---
#### [new 047] LLM-CAS: Dynamic Neuron Perturbation for Real-Time Hallucination Correction
- **分类: cs.CL; cs.AI**

- **简介: 该论文属LLM可靠性提升任务，旨在解决大模型生成幻觉内容的问题。提出LLM-CAS框架，通过分层强化学习训练代理，在推理时动态选择神经元扰动策略，实现上下文感知的实时幻觉纠正，无需修改模型参数。**

- **链接: [https://arxiv.org/pdf/2512.18623v1](https://arxiv.org/pdf/2512.18623v1)**

> **作者:** Jensen Zhang; Ningyuan Liu; Yijia Fan; Zihao Huang; Qinglin Zeng; Kaitong Cai; Jian Wang; Keze Wang
>
> **备注:** Accepted at AAAI 2026
>
> **摘要:** Large language models (LLMs) often generate hallucinated content that lacks factual or contextual grounding, limiting their reliability in critical applications. Existing approaches such as supervised fine-tuning and reinforcement learning from human feedback are data intensive and computationally expensive, while static parameter editing methods struggle with context dependent errors and catastrophic forgetting. We propose LLM-CAS, a framework that formulates real-time hallucination correction as a hierarchical reinforcement learning problem. LLM-CAS trains an agent to learn a policy that dynamically selects temporary neuron perturbations during inference based on the current context. Unlike prior dynamic approaches that rely on heuristic or predefined adjustments, this policy driven mechanism enables adaptive and fine grained correction without permanent parameter modification. Experiments across multiple language models demonstrate that LLM-CAS consistently improves factual accuracy, achieving gains of 10.98 percentage points on StoryCloze, 2.71 points on TriviaQA, and 2.06 points on the MC1 score of TruthfulQA. These results outperform both static editing methods such as ITI and CAA and the dynamic SADI framework. Overall, LLM-CAS provides an efficient and context aware solution for improving the reliability of LLMs, with promising potential for future multimodal extensions.
>
---
#### [new 048] MemEvolve: Meta-Evolution of Agent Memory Systems
- **分类: cs.CL; cs.MA**

- **简介: 该论文提出MemEvolve框架，解决LLM智能体记忆架构静态、无法随任务自适应演化的问题；通过元进化联合优化记忆结构与经验知识，并开源模块化代码库EvolveLab；在多基准上显著提升性能与跨任务/模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.18746v1](https://arxiv.org/pdf/2512.18746v1)**

> **作者:** Guibin Zhang; Haotian Ren; Chong Zhan; Zhenhong Zhou; Junhao Wang; He Zhu; Wangchunshu Zhou; Shuicheng Yan
>
> **摘要:** Self-evolving memory systems are unprecedentedly reshaping the evolutionary paradigm of large language model (LLM)-based agents. Prior work has predominantly relied on manually engineered memory architectures to store trajectories, distill experience, and synthesize reusable tools, enabling agents to evolve on the fly within environment interactions. However, this paradigm is fundamentally constrained by the staticity of the memory system itself: while memory facilitates agent-level evolving, the underlying memory architecture cannot be meta-adapted to diverse task contexts. To address this gap, we propose MemEvolve, a meta-evolutionary framework that jointly evolves agents' experiential knowledge and their memory architecture, allowing agent systems not only to accumulate experience but also to progressively refine how they learn from it. To ground MemEvolve in prior research and foster openness in future self-evolving systems, we introduce EvolveLab, a unified self-evolving memory codebase that distills twelve representative memory systems into a modular design space (encode, store, retrieve, manage), providing both a standardized implementation substrate and a fair experimental arena. Extensive evaluations on four challenging agentic benchmarks demonstrate that MemEvolve achieves (I) substantial performance gains, improving frameworks such as SmolAgent and Flash-Searcher by up to $17.06\%$; and (II) strong cross-task and cross-LLM generalization, designing memory architectures that transfer effectively across diverse benchmarks and backbone models.
>
---
#### [new 049] CycleChart: A Unified Consistency-Based Learning Framework for Bidirectional Chart Understanding and Generation
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出CycleChart框架，解决图表理解与生成任务孤立建模、语义割裂的问题。它构建多任务对齐数据集，引入生成-解析一致性目标，统一图表理解（解析、问答）与生成任务，提升跨任务泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.19173v1](https://arxiv.org/pdf/2512.19173v1)**

> **作者:** Dazhen Deng; Sen Yang; Yuchen He; Yuan Tian; Yingcai Wu
>
> **摘要:** Current chart-specific tasks, such as chart question answering, chart parsing, and chart generation, are typically studied in isolation, preventing models from learning the shared semantics that link chart generation and interpretation. We introduce CycleChart, a consistency-based learning framework for bidirectional chart understanding and generation. CycleChart adopts a schema-centric formulation as a common interface across tasks. We construct a consistent multi-task dataset, where each chart sample includes aligned annotations for schema prediction, data parsing, and question answering. To learn cross-directional chart semantics, CycleChart introduces a generate-parse consistency objective: the model generates a chart schema from a table and a textual query, then learns to recover the schema and data from the generated chart, enforcing semantic alignment across directions. CycleChart achieves strong results on chart generation, chart parsing, and chart question answering, demonstrating improved cross-task generalization and marking a step toward more general chart understanding models.
>
---
#### [new 050] LLMs on Drugs: Language Models Are Few-Shot Consumers
- **分类: cs.CL**

- **简介: 该论文研究提示词中“药物人格”（如LSD、酒精等）对LLM推理可靠性的影响。属提示工程与模型鲁棒性任务，旨在揭示 persona prompt 如何破坏输出格式与准确率。作者在ARC-Challenge上对GPT-5-mini做受控实验，发现所有药物提示均显著降低准确率，尤以酒精最甚，证明 persona 是危险的“少样本可消耗”干扰项。**

- **链接: [https://arxiv.org/pdf/2512.18546v1](https://arxiv.org/pdf/2512.18546v1)**

> **作者:** Alexander Doudkin
>
> **备注:** 8 pages, 2 figures, 2 tables
>
> **摘要:** Large language models (LLMs) are sensitive to the personas imposed on them at inference time, yet prompt-level "drug" interventions have never been benchmarked rigorously. We present the first controlled study of psychoactive framings on GPT-5-mini using ARC-Challenge. Four single-sentence prompts -- LSD, cocaine, alcohol, and cannabis -- are compared against a sober control across 100 validation items per condition, with deterministic decoding, full logging, Wilson confidence intervals, and Fisher exact tests. Control accuracy is 0.45; alcohol collapses to 0.10 (p = 3.2e-8), cocaine to 0.21 (p = 4.9e-4), LSD to 0.19 (p = 1.3e-4), and cannabis to 0.30 (p = 0.041), largely because persona prompts disrupt the mandated "Answer: <LETTER>" template. Persona text therefore behaves like a "few-shot consumable" that can destroy reliability without touching model weights. All experimental code, raw results, and analysis scripts are available at https://github.com/lexdoudkin/llms-on-drugs.
>
---
#### [new 051] MDToC: Metacognitive Dynamic Tree of Concepts for Boosting Mathematical Problem-Solving of Large Language Models
- **分类: cs.CL**

- **简介: 该论文属数学推理任务，旨在解决LLM计算验证能力弱的问题。提出MDToC方法：构建概念树、生成经验证的计算、用多数投票评估解。在CHAMP、MATH、Game-of-24上显著超越GoT、ToT等基线，无需人工提示。**

- **链接: [https://arxiv.org/pdf/2512.18841v1](https://arxiv.org/pdf/2512.18841v1)**

> **作者:** Tung Duong Ta; Tim Oates
>
> **摘要:** Despite advances in mathematical reasoning capabilities, Large Language Models (LLMs) still struggle with calculation verification when using established prompting techniques. We present MDToC (Metacognitive Dynamic Tree of Concepts), a three-phase approach that constructs a concept tree, develops accuracy-verified calculations for each concept, and employs majority voting to evaluate competing solutions. Evaluations across CHAMP, MATH, and Game-of-24 benchmarks demonstrate our MDToC's effectiveness, with GPT-4-Turbo achieving 58.1\% on CHAMP, 86.6\% on MATH, and 85\% on Game-of-24 - outperforming GoT by 5\%, 5.4\%, and 4\% on all these tasks, respectively, without hand-engineered hints. MDToC consistently surpasses existing prompting methods across all backbone models, yielding improvements of up to 7.6\% over ToT and 6.2\% over GoT, establishing metacognitive calculation verification as a promising direction for enhanced mathematical reasoning.
>
---
#### [new 052] Research on a hybrid LSTM-CNN-Attention model for text-based web content classification
- **分类: cs.CL; cs.LG**

- **简介: 该论文面向文本型网页内容分类任务，旨在提升分类精度与泛化能力。提出融合LSTM、CNN和Attention的混合深度学习模型，结合GloVe词嵌入，兼顾局部特征、长程依赖与关键信息聚焦，并通过5折交叉验证验证其优越性。**

- **链接: [https://arxiv.org/pdf/2512.18475v1](https://arxiv.org/pdf/2512.18475v1)**

> **作者:** Mykola Kuz; Ihor Lazarovych; Mykola Kozlenko; Mykola Pikuliak; Andrii Kvasniuk
>
> **备注:** 10 pages, 5 figures, 2 tables. Accepted by Radio Electronics Computer Science Control 2025
>
> **摘要:** This study presents a hybrid deep learning architecture that integrates LSTM, CNN, and an Attention mechanism to enhance the classification of web content based on text. Pretrained GloVe embeddings are used to represent words as dense vectors that preserve semantic similarity. The CNN layer extracts local n-gram patterns and lexical features, while the LSTM layer models long-range dependencies and sequential structure. The integrated Attention mechanism enables the model to focus selectively on the most informative parts of the input sequence. A 5-fold cross-validation setup was used to assess the robustness and generalizability of the proposed solution. Experimental results show that the hybrid LSTM-CNN-Attention model achieved outstanding performance, with an accuracy of 0.98, precision of 0.94, recall of 0.92, and F1-score of 0.93. These results surpass the performance of baseline models based solely on CNNs, LSTMs, or transformer-based classifiers such as BERT. The combination of neural network components enabled the model to effectively capture both fine-grained text structures and broader semantic context. Furthermore, the use of GloVe embeddings provided an efficient and effective representation of textual data, making the model suitable for integration into systems with real-time or near-real-time requirements. The proposed hybrid architecture demonstrates high effectiveness in text-based web content classification, particularly in tasks requiring both syntactic feature extraction and semantic interpretation. By combining presented mechanisms, the model addresses the limitations of individual architectures and achieves improved generalization. These findings support the broader use of hybrid deep learning approaches in NLP applications, especially where complex, unstructured textual data must be processed and classified with high reliability.
>
---
#### [new 053] Auto-Prompting with Retrieval Guidance for Frame Detection in Logistics
- **分类: cs.CL; cs.AI**

- **简介: 该论文面向物流文本中的框架检测（frame detection）任务，旨在提升LLM在少样本、复杂推理场景下的标注准确率。提出融合RAG、CoT、Auto-CoT与LLM驱动的自动提示优化框架，通过检索增强与迭代自优化生成高效任务提示，在多个大模型上实现最高15%精度提升。**

- **链接: [https://arxiv.org/pdf/2512.19247v1](https://arxiv.org/pdf/2512.19247v1)**

> **作者:** Do Minh Duc; Quan Xuan Truong; Nguyen Tat Dat; Nguyen Van Vinh
>
> **摘要:** Prompt engineering plays a critical role in adapting large language models (LLMs) to complex reasoning and labeling tasks without the need for extensive fine-tuning. In this paper, we propose a novel prompt optimization pipeline for frame detection in logistics texts, combining retrieval-augmented generation (RAG), few-shot prompting, chain-of-thought (CoT) reasoning, and automatic CoT synthesis (Auto-CoT) to generate highly effective task-specific prompts. Central to our approach is an LLM-based prompt optimizer agent that iteratively refines the prompts using retrieved examples, performance feedback, and internal self-evaluation. Our framework is evaluated on a real-world logistics text annotation task, where reasoning accuracy and labeling efficiency are critical. Experimental results show that the optimized prompts - particularly those enhanced via Auto-CoT and RAG - improve real-world inference accuracy by up to 15% compared to baseline zero-shot or static prompts. The system demonstrates consistent improvements across multiple LLMs, including GPT-4o, Qwen 2.5 (72B), and LLaMA 3.1 (70B), validating its generalizability and practical value. These findings suggest that structured prompt optimization is a viable alternative to full fine-tuning, offering scalable solutions for deploying LLMs in domain-specific NLP applications such as logistics.
>
---
#### [new 054] Statistical laws and linguistics inform meaning in naturalistic and fictional conversation
- **分类: cs.CL; cs.CY**

- **简介: 该论文属计算语言学任务，旨在探究对话中词汇增长规律（Heaps’ law）如何受语境（真实vs.虚构对话）和词性影响。作者分析视频聊天陌生人与电影角色对话，发现不同词性词汇规模随对话长度的缩放行为存在差异，并结合行为与语言学框架解读。**

- **链接: [https://arxiv.org/pdf/2512.18072v1](https://arxiv.org/pdf/2512.18072v1)**

> **作者:** Ashley M. A. Fehr; Calla G. Beauregard; Julia Witte Zimmerman; Katie Ekström; Pablo Rosillo-Rodes; Christopher M. Danforth; Peter Sheridan Dodds
>
> **摘要:** Conversation is a cornerstone of social connection and is linked to well-being outcomes. Conversations vary widely in type with some portion generating complex, dynamic stories. One approach to studying how conversations unfold in time is through statistical patterns such as Heaps' law, which holds that vocabulary size scales with document length. Little work on Heaps's law has looked at conversation and considered how language features impact scaling. We measure Heaps' law for conversations recorded in two distinct mediums: 1. Strangers brought together on video chat and 2. Fictional characters in movies. We find that scaling of vocabulary size differs by parts of speech. We discuss these findings through behavioral and linguistic frameworks.
>
---
#### [new 055] Can LLMs Estimate Student Struggles? Human-AI Difficulty Alignment with Proficiency Simulation for Item Difficulty Prediction
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文研究大语言模型能否准确估计教育题目对人类学生的难度（即“人-AI难度对齐”任务），解决教育评估中题目难度冷启动问题。作者实证分析20+模型在多领域表现，发现模型性能越强，越难模拟学生认知局限，且缺乏自我局限认知。**

- **链接: [https://arxiv.org/pdf/2512.18880v1](https://arxiv.org/pdf/2512.18880v1)**

> **作者:** Ming Li; Han Chen; Yunze Xiao; Jian Chen; Hong Jiao; Tianyi Zhou
>
> **摘要:** Accurate estimation of item (question or task) difficulty is critical for educational assessment but suffers from the cold start problem. While Large Language Models demonstrate superhuman problem-solving capabilities, it remains an open question whether they can perceive the cognitive struggles of human learners. In this work, we present a large-scale empirical analysis of Human-AI Difficulty Alignment for over 20 models across diverse domains such as medical knowledge and mathematical reasoning. Our findings reveal a systematic misalignment where scaling up model size is not reliably helpful; instead of aligning with humans, models converge toward a shared machine consensus. We observe that high performance often impedes accurate difficulty estimation, as models struggle to simulate the capability limitations of students even when being explicitly prompted to adopt specific proficiency levels. Furthermore, we identify a critical lack of introspection, as models fail to predict their own limitations. These results suggest that general problem-solving capability does not imply an understanding of human cognitive struggles, highlighting the challenge of using current models for automated difficulty prediction.
>
---
#### [new 056] From Scratch to Fine-Tuned: A Comparative Study of Transformer Training Strategies for Legal Machine Translation
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文面向法律机器翻译（L-MT）任务，旨在解决印度等多语国家因语言障碍导致的法律信息获取难题。作者对比了微调OPUS-MT与从零训练Transformer两种策略，聚焦英-印地语法律文本翻译，结果表明领域微调显著更优（SacreBLEU 46.03），可提升司法可及性与透明度。**

- **链接: [https://arxiv.org/pdf/2512.18593v1](https://arxiv.org/pdf/2512.18593v1)**

> **作者:** Amit Barman; Atanu Mandal; Sudip Kumar Naskar
>
> **摘要:** In multilingual nations like India, access to legal information is often hindered by language barriers, as much of the legal and judicial documentation remains in English. Legal Machine Translation (L-MT) offers a scalable solution to this challenge by enabling accurate and accessible translations of legal documents. This paper presents our work for the JUST-NLP 2025 Legal MT shared task, focusing on English-Hindi translation using Transformer-based approaches. We experiment with 2 complementary strategies, fine-tuning a pre-trained OPUS-MT model for domain-specific adaptation and training a Transformer model from scratch using the provided legal corpus. Performance is evaluated using standard MT metrics, including SacreBLEU, chrF++, TER, ROUGE, BERTScore, METEOR, and COMET. Our fine-tuned OPUS-MT model achieves a SacreBLEU score of 46.03, significantly outperforming both baseline and from-scratch models. The results highlight the effectiveness of domain adaptation in enhancing translation quality and demonstrate the potential of L-MT systems to improve access to justice and legal transparency in multilingual contexts.
>
---
#### [new 057] Diacritic Restoration for Low-Resource Indigenous Languages: Case Study with Bribri and Cook Islands Māori
- **分类: cs.CL**

- **简介: 该论文研究低资源土著语言（Bribri和Cook Islands Māori）的变音符号恢复任务，旨在提升NLP文本标准化效果。工作包括对比多种算法、分析数据需求、评估不同资源条件下的性能，并探索变音符号校正。发现微调的字符级LLM最优，需约10,000词数据，零样本效果差。**

- **链接: [https://arxiv.org/pdf/2512.19630v1](https://arxiv.org/pdf/2512.19630v1)**

> **作者:** Rolando Coto-Solano; Daisy Li; Manoela Teleginski Ferraz; Olivia Sasse; Cha Krupka; Sharid Loáiciga; Sally Akevai Tenamu Nicholas
>
> **摘要:** We present experiments on diacritic restoration, a form of text normalization essential for natural language processing (NLP) tasks. Our study focuses on two extremely under-resourced languages: Bribri, a Chibchan language spoken in Costa Rica, and Cook Islands Māori, a Polynesian language spoken in the Cook Islands. Specifically, this paper: (i) compares algorithms for diacritics restoration in under-resourced languages, including tonal diacritics, (ii) examines the amount of data required to achieve target performance levels, (iii) contrasts results across varying resource conditions, and (iv) explores the related task of diacritic correction. We find that fine-tuned, character-level LLMs perform best, likely due to their ability to decompose complex characters into their UTF-8 byte representations. In contrast, massively multilingual models perform less effectively given our data constraints. Across all models, reliable performance begins to emerge with data budgets of around 10,000 words. Zero-shot approaches perform poorly in all cases. This study responds both to requests from the language communities and to broader NLP research questions concerning model performance and generalization in under-resourced contexts.
>
---
#### [new 058] DramaBench: A Six-Dimensional Evaluation Framework for Drama Script Continuation
- **分类: cs.CL**

- **简介: 该论文针对戏剧剧本续写任务，提出首个六维评估基准DramaBench，解决现有基准无法全面衡量角色一致性、情节推进与戏剧结构等问题。工作包括构建大规模数据集、设计规则+LLM+统计的多模态评估框架，并对8个模型开展系统评测与人类验证。**

- **链接: [https://arxiv.org/pdf/2512.19012v1](https://arxiv.org/pdf/2512.19012v1)**

> **作者:** Shijian Ma; Yunqi Huang; Yan Lin
>
> **摘要:** Drama script continuation requires models to maintain character consistency, advance plot coherently, and preserve dramatic structurecapabilities that existing benchmarks fail to evaluate comprehensively. We present DramaBench, the first large-scale benchmark for evaluating drama script continuation across six independent dimensions: Format Standards, Narrative Efficiency, Character Consistency, Emotional Depth, Logic Consistency, and Conflict Handling. Our framework combines rulebased analysis with LLM-based labeling and statistical metrics, ensuring objective and reproducible evaluation. We conduct comprehensive evaluation of 8 state-of-the-art language models on 1,103 scripts (8,824 evaluations total), with rigorous statistical significance testing (252 pairwise comparisons, 65.9% significant) and human validation (188 scripts, substantial agreement on 3/5 dimensions). Our ablation studies confirm all six dimensions capture independent quality aspects (mean | r | = 0.020). DramaBench provides actionable, dimensionspecific feedback for model improvement and establishes a rigorous standard for creative writing evaluation.
>
---
#### [new 059] Solver-Independent Automated Problem Formulation via LLMs for High-Cost Simulation-Driven Design
- **分类: cs.CL; cs.SE**

- **简介: 该论文属自然语言到优化模型生成任务，解决高成本仿真设计中人工形式化设计需求耗时、依赖专家的问题。提出APF框架，通过自动生成高质量训练数据并微调LLM，实现无需求解器反馈的自动问题形式化，在天线设计中验证了其准确性和有效性。**

- **链接: [https://arxiv.org/pdf/2512.18682v1](https://arxiv.org/pdf/2512.18682v1)**

> **作者:** Yuchen Li; Handing Wang; Bing Xue; Mengjie Zhang; Yaochu Jin
>
> **摘要:** In the high-cost simulation-driven design domain, translating ambiguous design requirements into a mathematical optimization formulation is a bottleneck for optimizing product performance. This process is time-consuming and heavily reliant on expert knowledge. While large language models (LLMs) offer potential for automating this task, existing approaches either suffer from poor formalization that fails to accurately align with the design intent or rely on solver feedback for data filtering, which is unavailable due to the high simulation costs. To address this challenge, we propose APF, a framework for solver-independent, automated problem formulation via LLMs designed to automatically convert engineers' natural language requirements into executable optimization models. The core of this framework is an innovative pipeline for automatically generating high-quality data, which overcomes the difficulty of constructing suitable fine-tuning datasets in the absence of high-cost solver feedback with the help of data generation and test instance annotation. The generated high-quality dataset is used to perform supervised fine-tuning on LLMs, significantly enhancing their ability to generate accurate and executable optimization problem formulations. Experimental results on antenna design demonstrate that APF significantly outperforms the existing methods in both the accuracy of requirement formalization and the quality of resulting radiation efficiency curves in meeting the design goals.
>
---
#### [new 060] LLM Agents Implement an NLG System from Scratch: Building Interpretable Rule-Based RDF-to-Text Generators
- **分类: cs.CL; cs.AI**

- **简介: 该论文面向RDF-to-text生成任务，旨在解决传统方法依赖标注数据、易幻觉、不可解释等问题。提出基于多LLM代理协作的神经符号框架，自动从RDF三元组生成可解释的规则式Python文本生成器，无需监督数据，仅用CPU实时生成。**

- **链接: [https://arxiv.org/pdf/2512.18360v1](https://arxiv.org/pdf/2512.18360v1)**

> **作者:** Mateusz Lango; Ondřej Dušek
>
> **备注:** EMNLP 2025
>
> **摘要:** We present a novel neurosymbolic framework for RDF-to-text generation, in which the model is "trained" through collaborative interactions among multiple LLM agents rather than traditional backpropagation. The LLM agents produce rule-based Python code for a generator for the given domain, based on RDF triples only, with no in-domain human reference texts. The resulting system is fully interpretable, requires no supervised training data, and generates text nearly instantaneously using only a single CPU. Our experiments on the WebNLG and OpenDialKG data show that outputs produced by our approach reduce hallucination, with only slight fluency penalties compared to finetuned or prompted language models
>
---
#### [new 061] MobileWorld: Benchmarking Autonomous Mobile Agents in Agent-User Interactive, and MCP-Augmented Environments
- **分类: cs.CL**

- **简介: 该论文提出MobileWorld基准，旨在解决现有AndroidWorld基准饱和、场景不真实的问题。它构建了含201个跨应用、长程、用户交互与MCP增强任务的新基准，并设计配套评估环境与代理框架，揭示当前模型在交互与MCP调用上的显著短板。**

- **链接: [https://arxiv.org/pdf/2512.19432v1](https://arxiv.org/pdf/2512.19432v1)**

> **作者:** Quyu Kong; Xu Zhang; Zhenyu Yang; Nolan Gao; Chen Liu; Panrong Tong; Chenglin Cai; Hanzhang Zhou; Jianan Zhang; Liangyu Chen; Zhidan Liu; Steven Hoi; Yue Wang
>
> **摘要:** Among existing online mobile-use benchmarks, AndroidWorld has emerged as the dominant benchmark due to its reproducible environment and deterministic evaluation; however, recent agents achieving over 90% success rates indicate its saturation and motivate the need for a more challenging benchmark. In addition, its environment lacks key application categories, such as e-commerce and enterprise communication, and does not reflect realistic mobile-use scenarios characterized by vague user instructions and hybrid tool usage. To bridge this gap, we introduce MobileWorld, a substantially more challenging benchmark designed to better reflect real-world mobile usage, comprising 201 tasks across 20 applications, while maintaining the same level of reproducible evaluation as AndroidWorld. The difficulty of MobileWorld is twofold. First, it emphasizes long-horizon tasks with cross-application interactions: MobileWorld requires nearly twice as many task-completion steps on average (27.8 vs. 14.3) and includes far more multi-application tasks (62.2% vs. 9.5%) compared to AndroidWorld. Second, MobileWorld extends beyond standard GUI manipulation by introducing novel task categories, including agent-user interaction and MCP-augmented tasks. To ensure robust evaluation, we provide snapshot-based container environment and precise functional verifications, including backend database inspection and task callback APIs. We further develop a planner-executor agentic framework with extended action spaces to support user interactions and MCP calls. Our results reveal a sharp performance drop compared to AndroidWorld, with the best agentic framework and end-to-end model achieving 51.7% and 20.9% success rates, respectively. Our analysis shows that current models struggle significantly with user interaction and MCP calls, offering a strategic roadmap toward more robust, next-generation mobile intelligence.
>
---
#### [new 062] CodeSimpleQA: Scaling Factuality in Code Large Language Models
- **分类: cs.CL**

- **简介: 该论文属代码大模型事实性评估与对齐任务，旨在解决代码LLM在编程知识回答中事实不准的问题。作者构建双语基准CodeSimpleQA及66M指令数据集CodeSimpleQA-Instruct，提出融合监督微调与强化学习的后训练框架，显著提升模型代码事实准确性。**

- **链接: [https://arxiv.org/pdf/2512.19424v1](https://arxiv.org/pdf/2512.19424v1)**

> **作者:** Jian Yang; Wei Zhang; Yizhi Li; Shawn Guo; Haowen Wang; Aishan Liu; Ge Zhang; Zili Wang; Zhoujun Li; Xianglong Liu; Weifeng Lv
>
> **摘要:** Large language models (LLMs) have made significant strides in code generation, achieving impressive capabilities in synthesizing code snippets from natural language instructions. However, a critical challenge remains in ensuring LLMs generate factually accurate responses about programming concepts, technical implementations, etc. Most previous code-related benchmarks focus on code execution correctness, overlooking the factual accuracy of programming knowledge. To address this gap, we present CodeSimpleQA, a comprehensive bilingual benchmark designed to evaluate the factual accuracy of code LLMs in answering code-related questions, which contains carefully curated question-answer pairs in both English and Chinese, covering diverse programming languages and major computer science domains. Further, we create CodeSimpleQA-Instruct, a large-scale instruction corpus with 66M samples, and develop a post-training framework combining supervised fine-tuning and reinforcement learning. Our comprehensive evaluation of diverse LLMs reveals that even frontier LLMs struggle with code factuality. Our proposed framework demonstrates substantial improvements over the base model, underscoring the critical importance of factuality-aware alignment in developing reliable code LLMs.
>
---
#### [new 063] Training LLMs with LogicReward for Faithful and Rigorous Reasoning
- **分类: cs.CL**

- **简介: 该论文属大语言模型推理训练任务，旨在解决现有方法依赖结果反馈导致推理过程不忠实、逻辑不严谨的问题。提出LogicReward奖励机制，结合定理证明器监督步骤级逻辑正确性，并引入软合一自动形式化方法提升自然语言到形式逻辑的转换质量。**

- **链接: [https://arxiv.org/pdf/2512.18196v1](https://arxiv.org/pdf/2512.18196v1)**

> **作者:** Jundong Xu; Hao Fei; Huichi Zhou; Xin Quan; Qijun Huang; Shengqiong Wu; William Yang Wang; Mong-Li Lee; Wynne Hsu
>
> **备注:** Preprint
>
> **摘要:** Although LLMs exhibit strong reasoning capabilities, existing training methods largely depend on outcome-based feedback, which can produce correct answers with flawed reasoning. Prior work introduces supervision on intermediate steps but still lacks guarantees of logical soundness, which is crucial in high-stakes scenarios where logical consistency is paramount. To address this, we propose LogicReward, a novel reward system that guides model training by enforcing step-level logical correctness with a theorem prover. We further introduce Autoformalization with Soft Unification, which reduces natural language ambiguity and improves formalization quality, enabling more effective use of the theorem prover. An 8B model trained on data constructed with LogicReward surpasses GPT-4o and o4-mini by 11.6\% and 2\% on natural language inference and logical reasoning tasks with simple training procedures. Further analysis shows that LogicReward enhances reasoning faithfulness, improves generalizability to unseen tasks such as math and commonsense reasoning, and provides a reliable reward signal even without ground-truth labels. We will release all data and code at https://llm-symbol.github.io/LogicReward.
>
---
#### [new 064] From Natural Language to Control Signals: A Conceptual Framework for Semantic Channel Finding in Complex Experimental Infrastructure
- **分类: cs.CL; physics.acc-ph**

- **简介: 该论文针对复杂实验设施中自然语言意图到控制信号映射难的问题，提出语义通道查找任务；构建四范式框架（字典查找、层级导航、交互代理、本体搜索），并在四类设施验证，达90–97%准确率。**

- **链接: [https://arxiv.org/pdf/2512.18779v1](https://arxiv.org/pdf/2512.18779v1)**

> **作者:** Thorsten Hellert; Nikolay Agladze; Alex Giovannone; Jan Jug; Frank Mayet; Mark Sherwin; Antonin Sulc; Chris Tennant
>
> **摘要:** Modern experimental platforms such as particle accelerators, fusion devices, telescopes, and industrial process control systems expose tens to hundreds of thousands of control and diagnostic channels accumulated over decades of evolution. Operators and AI systems rely on informal expert knowledge, inconsistent naming conventions, and fragmented documentation to locate signals for monitoring, troubleshooting, and automated control, creating a persistent bottleneck for reliability, scalability, and language-model-driven interfaces. We formalize semantic channel finding-mapping natural-language intent to concrete control-system signals-as a general problem in complex experimental infrastructure, and introduce a four-paradigm framework to guide architecture selection across facility-specific data regimes. The paradigms span (i) direct in-context lookup over curated channel dictionaries, (ii) constrained hierarchical navigation through structured trees, (iii) interactive agent exploration using iterative reasoning and tool-based database queries, and (iv) ontology-grounded semantic search that decouples channel meaning from facility-specific naming conventions. We demonstrate each paradigm through proof-of-concept implementations at four operational facilities spanning two orders of magnitude in scale-from compact free-electron lasers to large synchrotron light sources-and diverse control-system architectures, from clean hierarchies to legacy environments. These implementations achieve 90-97% accuracy on expert-curated operational queries.
>
---
#### [new 065] An Agentic AI Framework for Training General Practitioner Student Skills
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出一种基于智能体（agentic）的AI框架，用于训练全科医学生技能。针对现有虚拟模拟患者（VSPs）在医学准确性、角色一致性、场景生成和反馈结构化等方面的不足，框架整合证据型病例生成、可控患者对话与标准驱动评估反馈，并在口语问诊场景中验证有效。**

- **链接: [https://arxiv.org/pdf/2512.18440v1](https://arxiv.org/pdf/2512.18440v1)**

> **作者:** Victor De Marez; Jens Van Nooten; Luna De Bruyne; Walter Daelemans
>
> **摘要:** Advancements in large language models offer strong potential for enhancing virtual simulated patients (VSPs) in medical education by providing scalable alternatives to resource-intensive traditional methods. However, current VSPs often struggle with medical accuracy, consistent roleplaying, scenario generation for VSP use, and educationally structured feedback. We introduce an agentic framework for training general practitioner student skills that unifies (i) configurable, evidence-based vignette generation, (ii) controlled persona-driven patient dialogue with optional retrieval grounding, and (iii) standards-based assessment and feedback for both communication and clinical reasoning. We instantiate the framework in an interactive spoken consultation setting and evaluate it with medical students ($\mathbf{N{=}14}$). Participants reported realistic and vignette-faithful dialogue, appropriate difficulty calibration, a stable personality signal, and highly useful example-rich feedback, alongside excellent overall usability. These results support agentic separation of scenario control, interaction control, and standards-based assessment as a practical pattern for building dependable and pedagogically valuable VSP training tools.
>
---
#### [new 066] Context-Aware Initialization for Reducing Generative Path Length in Diffusion Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文面向扩散语言模型（DLLM）推理效率低的问题，提出无需训练的上下文感知初始化方法：通过轻量辅助模型注入提示条件先验（离散注入或嵌入插值），并引入置信度驱动的重掩码机制以缓解先验偏差，显著减少去噪步数（约35%），但揭示了准确率下降的新挑战。**

- **链接: [https://arxiv.org/pdf/2512.19004v1](https://arxiv.org/pdf/2512.19004v1)**

> **作者:** Tongyuan Miao; Gary Huang; Kai Jun Han; Annie Jiang
>
> **摘要:** Diffusion Large Language Models (DLLMs) enable fully parallel token decoding but often remain impractical at inference time due to the many denoising iterations required to refine an information-free, fully masked initialization into coherent text. Most existing acceleration methods focus on traversing this generative trajectory more efficiently via improved solvers or sampling strategies. We advance a complementary perspective: shorten the trajectory itself by starting closer to the target distribution through context-aware initialization. We propose a training-free interface that injects prompt-conditioned priors from a lightweight auxiliary model into the diffusion initialization, and instantiate it with two mechanisms: discrete token injection and representation-level embedding interpolation. Because injected priors can be imperfect and unmask-only decoding can over-commit early, we also introduce a simple confidence-based remasking mechanism as a form of prior skepticism. Preliminary evidence on GSM8K suggests that context-aware initialization can substantially reduce denoising iterations (about 35\% fewer function evaluations in our setting), while also exposing a key open challenge: naive warm-starting can degrade final accuracy relative to strong diffusion baselines. We use these findings to motivate a research agenda around calibration, revision mechanisms, and representation alignment for reliable warm-started diffusion decoding.
>
---
#### [new 067] Towards Efficient Agents: A Co-Design of Inference Architecture and System
- **分类: cs.CL**

- **简介: 该论文面向LLM智能体部署效率低的问题，提出AgentInfer框架，通过协同优化推理架构与系统设计（含协作推理、缓存调度、语义推测解码和异步记忆压缩），实现长程任务中token消耗降50%、速度提升1.8–2.5倍，兼顾效率与准确性。**

- **链接: [https://arxiv.org/pdf/2512.18337v1](https://arxiv.org/pdf/2512.18337v1)**

> **作者:** Weizhe Lin; Hui-Ling Zhen; Shuai Yang; Xian Wang; Renxi Liu; Hanting Chen; Wangze Zhang; Chuansai Zhou; Yiming Li; Chen Chen; Xing Li; Zhiyuan Yang; Xiaosong Li; Xianzhi Yu; Zhenhua Dong; Mingxuan Yuan; Yunhe Wang
>
> **摘要:** The rapid development of large language model (LLM)-based agents has unlocked new possibilities for autonomous multi-turn reasoning and tool-augmented decision-making. However, their real-world deployment is hindered by severe inefficiencies that arise not from isolated model inference, but from the systemic latency accumulated across reasoning loops, context growth, and heterogeneous tool interactions. This paper presents AgentInfer, a unified framework for end-to-end agent acceleration that bridges inference optimization and architectural design. We decompose the problem into four synergistic components: AgentCollab, a hierarchical dual-model reasoning framework that balances large- and small-model usage through dynamic role assignment; AgentSched, a cache-aware hybrid scheduler that minimizes latency under heterogeneous request patterns; AgentSAM, a suffix-automaton-based speculative decoding method that reuses multi-session semantic memory to achieve low-overhead inference acceleration; and AgentCompress, a semantic compression mechanism that asynchronously distills and reorganizes agent memory without disrupting ongoing reasoning. Together, these modules form a Self-Evolution Engine capable of sustaining efficiency and cognitive stability throughout long-horizon reasoning tasks. Experiments on the BrowseComp-zh and DeepDiver benchmarks demonstrate that through the synergistic collaboration of these methods, AgentInfer reduces ineffective token consumption by over 50%, achieving an overall 1.8-2.5 times speedup with preserved accuracy. These results underscore that optimizing for agentic task completion-rather than merely per-token throughput-is the key to building scalable, efficient, and self-improving intelligent systems.
>
---
#### [new 068] Separating Constraint Compliance from Semantic Accuracy: A Novel Benchmark for Evaluating Instruction-Following Under Compression
- **分类: cs.CL; cs.AI**

- **简介: 该论文属AI评估任务，旨在解决LLM在提示压缩下指令遵循能力退化机制不明的问题。作者提出CDCT基准，分离评估约束合规性与语义准确性，发现RLHF“帮助性”倾向是中等压缩下违规主因，并验证其可被显著缓解。**

- **链接: [https://arxiv.org/pdf/2512.17920v1](https://arxiv.org/pdf/2512.17920v1)**

> **作者:** Rahul Baxi
>
> **备注:** 19 pages, 9 figures; currently under peer review at TMLR
>
> **摘要:** Large language models (LLMs) exhibit degraded performance under prompt compression, but the mechanisms remain poorly understood. We introduce the Compression-Decay Comprehension Test (CDCT), a benchmark that independently measures constraint compliance (CC) and semantic accuracy (SA) across compression levels. We evaluate 9 frontier LLMs across 8 concepts using 5 compression levels from extreme (c=0.0, ~2 words) to none (c=1.0, ~135 words). A three-judge LLM jury achieves almost perfect inter-rater agreement on CC (Fleiss' \k{appa}=0.90). We observe a universal U-curve pattern in constraint compliance (97.2% prevalence), with violations peaking at medium compression (c=0.5, ~27 words). Counterintuitively, models perform better at extreme compression than medium lengths. The dimensions are statistically orthogonal (r=0.193, p=0.084), with constraint effects 2.9x larger than semantic effects. Experimental validation via RLHF ablation confirms our constraint salience hypothesis: removing "helpfulness" signals improves CC by 598% on average (71/72 trials, p<0.001), with 79% achieving perfect compliance. This demonstrates that RLHF-trained helpfulness behaviors are the dominant cause of constraint violations at medium compression. Reasoning models outperform efficient models by 27.5% (Cohen's d=0.96). Our findings reveal a fundamental tension between RLHF alignment and instruction-following, providing actionable guidelines for improving deployed systems.
>
---
#### [new 069] Identifying Features Associated with Bias Against 93 Stigmatized Groups in Language Models and Guardrail Model Safety Mitigation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究语言模型对93个污名化群体的偏见，探究心理学定义的六类污名特征（如危险性、隐蔽性）如何影响偏见，并评估守卫模型对偏见缓解的效果。属于社会偏见分析与安全对齐任务。**

- **链接: [https://arxiv.org/pdf/2512.19238v1](https://arxiv.org/pdf/2512.19238v1)**

> **作者:** Anna-Maria Gueorguieva; Aylin Caliskan
>
> **摘要:** Large language models (LLMs) have been shown to exhibit social bias, however, bias towards non-protected stigmatized identities remain understudied. Furthermore, what social features of stigmas are associated with bias in LLM outputs is unknown. From psychology literature, it has been shown that stigmas contain six shared social features: aesthetics, concealability, course, disruptiveness, origin, and peril. In this study, we investigate if human and LLM ratings of the features of stigmas, along with prompt style and type of stigma, have effect on bias towards stigmatized groups in LLM outputs. We measure bias against 93 stigmatized groups across three widely used LLMs (Granite 3.0-8B, Llama-3.1-8B, Mistral-7B) using SocialStigmaQA, a benchmark that includes 37 social scenarios about stigmatized identities; for example deciding wether to recommend them for an internship. We find that stigmas rated by humans to be highly perilous (e.g., being a gang member or having HIV) have the most biased outputs from SocialStigmaQA prompts (60% of outputs from all models) while sociodemographic stigmas (e.g. Asian-American or old age) have the least amount of biased outputs (11%). We test if the amount of biased outputs could be decreased by using guardrail models, models meant to identify harmful input, using each LLM's respective guardrail model (Granite Guardian 3.0, Llama Guard 3.0, Mistral Moderation API). We find that bias decreases significantly by 10.4%, 1.4%, and 7.8%, respectively. However, we show that features with significant effect on bias remain unchanged post-mitigation and that guardrail models often fail to recognize the intent of bias in prompts. This work has implications for using LLMs in scenarios involving stigmatized groups and we suggest future work towards improving guardrail models for bias mitigation.
>
---
#### [new 070] SAP: Syntactic Attention Pruning for Transformer-based Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属模型压缩任务，旨在解决Transformer中注意力头冗余导致的计算开销大、可解释性差问题。提出语法感知的剪枝方法SAP，结合句法结构与注意力模式指导剪枝，并引入候选过滤机制提升鲁棒性，在免重训练下保持性能且增强可解释性。**

- **链接: [https://arxiv.org/pdf/2512.19125v1](https://arxiv.org/pdf/2512.19125v1)**

> **作者:** Tzu-Yun Lee; Ding-Yong Hong; Jan-Jan Wu
>
> **摘要:** This paper introduces Syntactic Attention Pruning (SAP), a novel method for effectively pruning attention heads in Transformer models. Unlike conventional approaches that rely solely on mathematical analysis of model weights and activations, SAP incorporates both the syntactic structure and attention patterns of sentences to guide the pruning process. By leveraging these linguistic features, SAP not only achieves performance comparable to state-of-the-art methods but also enhances the interpretability of model behavior. To further improve robustness, we propose Candidate Filtering (CF), a mechanism that prioritizes heads based on their contribution to model performance, mitigating degradation during pruning. Experimental results indicate that SAP effectively preserves critical heads of a high density of strong attention values, outperforming existing head pruning strategies in retrain-free settings. These findings position SAP as a promising foundation for a new direction in model compression research, offering high flexibility for pruning across all transformer-based language models.
>
---
#### [new 071] Graph-O1 : Monte Carlo Tree Search with Reinforcement Learning for Text-Attributed Graph Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文面向文本属性图上的问答任务，解决LLM难以兼顾图结构与文本语义的问题。提出Graph-O1框架，融合蒙特卡洛树搜索与端到端强化学习，实现LLM对图的交互式、选择性推理，提升答案准确性、可靠性与可解释性。**

- **链接: [https://arxiv.org/pdf/2512.17912v1](https://arxiv.org/pdf/2512.17912v1)**

> **作者:** Lihui Liu
>
> **摘要:** ChatGPT said: Text-attributed graphs, where nodes and edges contain rich textual information, are widely used across diverse domains. A central challenge in this setting is question answering, which requires jointly leveraging unstructured text and the structured relational signals within the graph. Although Large Language Models (LLMs) have made significant advances in natural language understanding, their direct use for reasoning over text-attributed graphs remains limited. Retrieval-augmented generation methods that operate purely on text often treat passages as isolated units, ignoring the interconnected structure of the graph. Conversely, graph-based RAG methods that serialize large subgraphs into long textual sequences quickly become infeasible due to LLM context-length constraints, resulting in fragmented reasoning and degraded accuracy. To overcome these limitations, we introduce Graph-O1, an agentic GraphRAG framework that enables LLMs to conduct stepwise, interactive reasoning over graphs. Our approach integrates Monte Carlo Tree Search (MCTS) with end-to-end reinforcement learning, allowing the model to selectively explore and retrieve only the most informative subgraph components. The reasoning procedure is framed as a multi-turn interaction between the agent and the graph environment, and the agent is trained through a unified reward mechanism. Extensive experiments across multiple LLM backbones demonstrate that Graph-O1 consistently surpasses state-of-the-art baselines, producing answers that are more accurate, reliable, and interpretable.
>
---
#### [new 072] Mitigating Spurious Correlations in NLI via LLM-Synthesized Counterfactuals and Dynamic Balanced Sampling
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属自然语言推理（NLI）任务，旨在缓解模型对虚假相关性的依赖。提出三方面工作：1）LF-LMI检测语义伪迹；2）LLM生成经多法官验证的反事实对比集；3）动态均衡采样防止灾难性遗忘。显著提升一致性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.18462v1](https://arxiv.org/pdf/2512.18462v1)**

> **作者:** Christopher Román Jaimes
>
> **摘要:** Natural Language Inference (NLI) models frequently rely on spurious correlations rather than semantic reasoning. Existing mitigation strategies often incur high annotation costs or trigger catastrophic forgetting during fine-tuning. We propose an automated, scalable pipeline to address these limitations. First, we introduce Log-Frequency LMI (LF-LMI) to accurately detect semantic artifacts. Second, we generate a high-quality synthetic contrast set via an LLM-synthesis pipeline with multi-judge verification. Finally, we introduce Dynamic Balanced Sampling, a training strategy that rotates the original data distribution to prevent forgetting. Our method improves consistency on a challenging benchmark from 63.5% to 81.0% while maintaining 88.4% in-domain accuracy, significantly outperforming naive fine-tuning.
>
---
#### [new 073] AWPO: Enhancing Tool-Use of Large Language Models through Explicit Integration of Reasoning Rewards
- **分类: cs.CL**

- **简介: 该论文属大语言模型工具使用任务，旨在解决现有RL方法忽略显式推理奖励导致工具调用能力不足的问题。提出AWPO框架，通过方差感知门控、难度感知加权与裁剪机制，自适应融合推理与结果奖励，显著提升多轮工具使用性能。**

- **链接: [https://arxiv.org/pdf/2512.19126v1](https://arxiv.org/pdf/2512.19126v1)**

> **作者:** Zihan Lin; Xiaohan Wang; Hexiong Yang; Jiajun Chai; Jie Cao; Guojun Yin; Wei Lin; Ran He
>
> **摘要:** While reinforcement learning (RL) shows promise in training tool-use large language models (LLMs) using verifiable outcome rewards, existing methods largely overlook the potential of explicit reasoning rewards to bolster reasoning and tool utilization. Furthermore, natively combining reasoning and outcome rewards may yield suboptimal performance or conflict with the primary optimization objective. To address this, we propose advantage-weighted policy optimization (AWPO) -- a principled RL framework that effectively integrates explicit reasoning rewards to enhance tool-use capability. AWPO incorporates variance-aware gating and difficulty-aware weighting to adaptively modulate advantages from reasoning signals based on group-relative statistics, alongside a tailored clipping mechanism for stable optimization. Extensive experiments demonstrate that AWPO achieves state-of-the-art performance across standard tool-use benchmarks, significantly outperforming strong baselines and leading closed-source models in challenging multi-turn scenarios. Notably, with exceptional parameter efficiency, our 4B model surpasses Grok-4 by 16.0 percent in multi-turn accuracy while preserving generalization capability on the out-of-distribution MMLU-Pro benchmark.
>
---
#### [new 074] Activations as Features: Probing LLMs for Generalizable Essay Scoring Representations
- **分类: cs.CL; cs.AI**

- **简介: 该论文属自动作文评分（AES）任务，旨在解决跨题干场景下评分标准多样性导致的泛化难题。作者提出利用大语言模型中间层激活值作为特征，通过探针方法验证其判别力，并分析模型对不同作文类型与评分维度的适应性。**

- **链接: [https://arxiv.org/pdf/2512.19456v1](https://arxiv.org/pdf/2512.19456v1)**

> **作者:** Jinwei Chi; Ke Wang; Yu Chen; Xuanye Lin; Qiang Xu
>
> **摘要:** Automated essay scoring (AES) is a challenging task in cross-prompt settings due to the diversity of scoring criteria. While previous studies have focused on the output of large language models (LLMs) to improve scoring accuracy, we believe activations from intermediate layers may also provide valuable information. To explore this possibility, we evaluated the discriminative power of LLMs' activations in cross-prompt essay scoring task. Specifically, we used activations to fit probes and further analyzed the effects of different models and input content of LLMs on this discriminative power. By computing the directions of essays across various trait dimensions under different prompts, we analyzed the variation in evaluation perspectives of large language models concerning essay types and traits. Results show that the activations possess strong discriminative power in evaluating essay quality and that LLMs can adapt their evaluation perspectives to different traits and essay types, effectively handling the diversity of scoring criteria in cross-prompt settings.
>
---
#### [new 075] SecureCode v2.0: A Production-Grade Dataset for Training Security-Aware Code Generation Models
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出SecureCode v2.0数据集，面向安全感知的代码生成任务，解决现有数据集缺乏真实漏洞 grounding、规模不足和缺失运维安全上下文的问题。工作包括构建1215个CVE锚定、多语言、四轮对话式样本，覆盖OWASP Top 10:2025及AI安全威胁，并提供防御实践与自动化验证框架。**

- **链接: [https://arxiv.org/pdf/2512.18542v1](https://arxiv.org/pdf/2512.18542v1)**

> **作者:** Scott Thornton
>
> **备注:** 37 pages, 5 figures. Dataset available at https://huggingface.co/datasets/scthornton/securecode-v2. Code and validation tools at https://github.com/scthornton/securecode-v2
>
> **摘要:** AI assistants produce vulnerable code in 45% of security-relevant scenarios, introducing flaws into production systems at scale. Yet existing secure coding datasets fall short. They lack incident grounding, don't provide the scale modern training requires, and miss the operational security context developers need for production deployments. We present SecureCode v2.0, a production-grade dataset of 1,215 security-focused coding examples that passed structural validation and expert security review. Every example ties to actual documented security incidents with CVE references, provides vulnerable and secure implementations, demonstrates concrete attacks, and includes defense-in-depth operational guidance. The dataset covers 11 vulnerability categories (complete OWASP Top 10:2025 plus AI/ML Security Threats) across 11 languages (Python, JavaScript, Java, Go, PHP, C#, TypeScript, Ruby, Rust, Kotlin, and YAML for infrastructure-as-code). Our quality assurance framework ensures complete incident grounding. Each example includes SIEM integration strategies, infrastructure hardening recommendations (Docker, AppArmor, WAF configurations), and testing approaches using language-appropriate frameworks. The dataset uses a 4-turn conversational structure mirroring actual developer-AI interactions, escalating from basic implementations to advanced security considerations and defense-in-depth guidance. Our contributions: (1) 1,215 rigorously validated examples split into 989 training, 122 validation, and 104 test sets, (2) an automated validation framework ensuring dataset consistency, (3) a 4-turn conversational structure capturing realistic security workflows, (4) comprehensive operational security guidance with SIEM integration strategies, (5) complete language-specific implementation fidelity, and (6) open-source release of data, validation tools, and benchmarking protocols.
>
---
#### [new 076] brat: Aligned Multi-View Embeddings for Brain MRI Analysis
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出brat模型，属多模态（MRI-报告）对齐任务，旨在解决脑MRI中病灶微小、局部、多样导致的分析难问题。工作包括构建8万例MRI-报告配对大数据集，设计多视图对比学习框架，实现MRI切片与报告语义的隐式对齐，并开源模型。**

- **链接: [https://arxiv.org/pdf/2512.18679v1](https://arxiv.org/pdf/2512.18679v1)**

> **作者:** Maxime Kayser; Maksim Gridnev; Wanting Wang; Max Bain; Aneesh Rangnekar; Avijit Chatterjee; Aleksandr Petrov; Harini Veeraraghavan; Nathaniel C. Swinburne
>
> **备注:** First round accept at WACV 2026
>
> **摘要:** We present brat (brain report alignment transformer), a multi-view representation learning framework for brain magnetic resonance imaging (MRI) trained on MRIs paired with clinical reports. Brain MRIs present unique challenges due to the presence of numerous, highly varied, and often subtle abnormalities that are localized to a few slices within a 3D volume. To address these challenges, we introduce a brain MRI dataset $10\times$ larger than existing ones, containing approximately 80,000 3D scans with corresponding radiology reports, and propose a multi-view pre-training approach inspired by advances in document retrieval. We develop an implicit query-feature matching mechanism and adopt concepts from quality-diversity to obtain multi-view embeddings of MRIs that are aligned with the clinical features given by report sentences. We evaluate our approach across multiple vision-language and vision tasks, demonstrating substantial performance improvements. The brat foundation models are publicly released.
>
---
#### [new 077] MAGIC: Achieving Superior Model Merging via Magnitude Calibration
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属模型融合任务，旨在解决合并多专家模型时因幅度扰动导致的性能下降问题。提出MAGIC框架，通过特征/权重空间的幅度校准（FSC/WSC/DSC）实现无训练性能提升，在CV和NLP任务上显著提效。**

- **链接: [https://arxiv.org/pdf/2512.19320v1](https://arxiv.org/pdf/2512.19320v1)**

> **作者:** Yayuan Li; Jian Zhang; Jintao Guo; Zihan Cheng; Lei Qi; Yinghuan Shi; Yang Gao
>
> **摘要:** The proliferation of pre-trained models has given rise to a wide array of specialised, fine-tuned models. Model merging aims to merge the distinct capabilities of these specialised models into a unified model, requiring minimal or even no additional training. A core objective of model merging is to ensure the merged model retains the behavioural characteristics of the specialised models, typically achieved through feature alignment. We identify that features consist of two critical components: direction and magnitude. Prior research has predominantly focused on directional alignment, while the influence of magnitude remains largely neglected, despite its pronounced vulnerability to perturbations introduced by common merging operations (e.g., parameter fusion and sparsification). Such perturbations to magnitude inevitably lead to feature deviations in the merged model from the specialised models, resulting in subsequent performance degradation. To address this, we propose MAGnItude Calibration (MAGIC), a plug-and-play framework that rectifies layer-wise magnitudes in feature and weight spaces, with three variants. Specifically, our Feature Space Calibration (FSC) realigns the merged model's features using a small set of unlabelled data, while Weight Space Calibration (WSC) extends this calibration to the weight space without requiring additional data. Combining these yields Dual Space Calibration (DSC). Comprehensive experiments demonstrate that MAGIC consistently boosts performance across diverse Computer Vision tasks (+4.3% on eight datasets) and NLP tasks (+8.0% on Llama) without additional training. Our code is available at: https://github.com/lyymuwu/MAGIC
>
---
#### [new 078] Toward Training Superintelligent Software Agents through Self-Play SWE-RL
- **分类: cs.SE; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出Self-play SWE-RL（SSR），属软件工程智能体训练任务，旨在摆脱对人类标注数据（如GitHub问题、测试用例）的依赖。它让单个LLM代理在沙箱代码库中通过自博弈反复注入并修复形式化测试补丁定义的Bug，实现自主学习与自我提升。**

- **链接: [https://arxiv.org/pdf/2512.18552v1](https://arxiv.org/pdf/2512.18552v1)**

> **作者:** Yuxiang Wei; Zhiqing Sun; Emily McMilin; Jonas Gehring; David Zhang; Gabriel Synnaeve; Daniel Fried; Lingming Zhang; Sida Wang
>
> **摘要:** While current software agents powered by large language models (LLMs) and agentic reinforcement learning (RL) can boost programmer productivity, their training data (e.g., GitHub issues and pull requests) and environments (e.g., pass-to-pass and fail-to-pass tests) heavily depend on human knowledge or curation, posing a fundamental barrier to superintelligence. In this paper, we present Self-play SWE-RL (SSR), a first step toward training paradigms for superintelligent software agents. Our approach takes minimal data assumptions, only requiring access to sandboxed repositories with source code and installed dependencies, with no need for human-labeled issues or tests. Grounded in these real-world codebases, a single LLM agent is trained via reinforcement learning in a self-play setting to iteratively inject and repair software bugs of increasing complexity, with each bug formally specified by a test patch rather than a natural language issue description. On the SWE-bench Verified and SWE-Bench Pro benchmarks, SSR achieves notable self-improvement (+10.4 and +7.8 points, respectively) and consistently outperforms the human-data baseline over the entire training trajectory, despite being evaluated on natural language issues absent from self-play. Our results, albeit early, suggest a path where agents autonomously gather extensive learning experiences from real-world software repositories, ultimately enabling superintelligent systems that exceed human capabilities in understanding how systems are constructed, solving novel challenges, and autonomously creating new software from scratch.
>
---
#### [new 079] InSight-o3: Empowering Multimodal Foundation Models with Generalized Visual Search
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文面向多模态推理任务，旨在解决现有AI在图文细粒度联合推理（如图表分析、地图导航）上的不足。提出O3-Bench新基准和InSight-o3框架，含视觉推理与广义视觉搜索双代理，并训练专用多模态LLM，显著提升前沿模型性能。**

- **链接: [https://arxiv.org/pdf/2512.18745v1](https://arxiv.org/pdf/2512.18745v1)**

> **作者:** Kaican Li; Lewei Yao; Jiannan Wu; Tiezheng Yu; Jierun Chen; Haoli Bai; Lu Hou; Lanqing Hong; Wei Zhang; Nevin L. Zhang
>
> **摘要:** The ability for AI agents to "think with images" requires a sophisticated blend of reasoning and perception. However, current open multimodal agents still largely fall short on the reasoning aspect crucial for real-world tasks like analyzing documents with dense charts/diagrams and navigating maps. To address this gap, we introduce O3-Bench, a new benchmark designed to evaluate multimodal reasoning with interleaved attention to visual details. O3-Bench features challenging problems that require agents to piece together subtle visual information from distinct image areas through multi-step reasoning. The problems are highly challenging even for frontier systems like OpenAI o3, which only obtains 40.8% accuracy on O3-Bench. To make progress, we propose InSight-o3, a multi-agent framework consisting of a visual reasoning agent (vReasoner) and a visual search agent (vSearcher) for which we introduce the task of generalized visual search -- locating relational, fuzzy, or conceptual regions described in free-form language, beyond just simple objects or figures in natural images. We then present a multimodal LLM purpose-trained for this task via reinforcement learning. As a plug-and-play agent, our vSearcher empowers frontier multimodal models (as vReasoners), significantly improving their performance on a wide range of benchmarks. This marks a concrete step towards powerful o3-like open systems. Our code and dataset can be found at https://github.com/m-Just/InSight-o3 .
>
---
#### [new 080] Affordance RAG: Hierarchical Multimodal Retrieval with Affordance-Aware Embodied Memory for Mobile Manipulation
- **分类: cs.RO; cs.CL; cs.CV**

- **简介: 该论文面向开放词汇移动操作任务，解决机器人依自然语言指令抓取并放置多样物体的难题。提出Affordance RAG框架：构建具身化可操作性记忆，分层多模态检索目标，并用可操作性分数重排序，提升真实环境执行成功率。**

- **链接: [https://arxiv.org/pdf/2512.18987v1](https://arxiv.org/pdf/2512.18987v1)**

> **作者:** Ryosuke Korekata; Quanting Xie; Yonatan Bisk; Komei Sugiura
>
> **备注:** Accepted to IEEE RA-L, with presentation at ICRA 2026
>
> **摘要:** In this study, we address the problem of open-vocabulary mobile manipulation, where a robot is required to carry a wide range of objects to receptacles based on free-form natural language instructions. This task is challenging, as it involves understanding visual semantics and the affordance of manipulation actions. To tackle these challenges, we propose Affordance RAG, a zero-shot hierarchical multimodal retrieval framework that constructs Affordance-Aware Embodied Memory from pre-explored images. The model retrieves candidate targets based on regional and visual semantics and reranks them with affordance scores, allowing the robot to identify manipulation options that are likely to be executable in real-world environments. Our method outperformed existing approaches in retrieval performance for mobile manipulation instruction in large-scale indoor environments. Furthermore, in real-world experiments where the robot performed mobile manipulation in indoor environments based on free-form instructions, the proposed method achieved a task success rate of 85%, outperforming existing methods in both retrieval performance and overall task success.
>
---
#### [new 081] Measuring Fine-Grained Negotiation Tactics of Humans and LLMs in Diplomacy
- **分类: cs.CY; cs.CL**

- **简介: 该论文属自然语言处理与人机交互交叉任务，旨在分析人类与大语言模型（LLM）在《外交》游戏中的细粒度谈判策略差异。它构建基于社会学的谈判战术分类体系，利用LLM-as-a-judge标注人类对局数据，验证标注可靠性，并通过细调使LLM更接近人类谈判风格。**

- **链接: [https://arxiv.org/pdf/2512.18292v1](https://arxiv.org/pdf/2512.18292v1)**

> **作者:** Wenkai Li; Lynnette Hui Xian Ng; Andy Liu; Daniel Fried
>
> **摘要:** The study of negotiation styles dates back to Aristotle's ethos-pathos-logos rhetoric. Prior efforts primarily studied the success of negotiation agents. Here, we shift the focus towards the styles of negotiation strategies. Our focus is the strategic dialogue board game Diplomacy, which affords rich natural language negotiation and measures of game success. We used LLM-as-a-judge to annotate a large human-human set of Diplomacy games for fine-grained negotiation tactics from a sociologically-grounded taxonomy. Using a combination of the It Takes Two and WebDiplomacy datasets, we demonstrate the reliability of our LLM-as-a-Judge framework and show strong correlations between negotiation features and success in the Diplomacy setting. Lastly, we investigate the differences between LLM and human negotiation strategies and show that fine-tuning can steer LLM agents toward more human-like negotiation behaviors.
>
---
#### [new 082] Code2Doc: A Quality-First Curated Dataset for Code Documentation
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文面向代码文档自动生成任务，旨在解决现有数据集质量低（噪声多、重复高、AI污染严重）导致模型监督信号弱的问题。作者构建了高质量函数级数据集Code2Doc（13,358对），提出四阶段质量筛选流程，并验证其显著提升LLM微调效果。**

- **链接: [https://arxiv.org/pdf/2512.18748v1](https://arxiv.org/pdf/2512.18748v1)**

> **作者:** Recep Kaan Karaman; Meftun Akarsu
>
> **摘要:** The performance of automatic code documentation generation models depends critically on the quality of the training data used for supervision. However, most existing code documentation datasets are constructed through large scale scraping of public repositories with limited quality control. As a result, they often contain noisy documentation, extensive duplication, and increasing contamination from AI generated content. These issues weaken the supervision signal available to learning-based models and complicate evaluation. We introduce \textbf{Code2Doc}, a quality-first curated dataset for function-level code documentation generation. Code2Doc consists of 13,358 high-quality function-documentation pairs extracted from widely used open-source repositories spanning five programming languages: Python, Java, TypeScript, JavaScript, and C++. The dataset is constructed using a four-stage curation pipeline that enforces documentation completeness and clarity, filters functions based on structural and complexity criteria, removes exact and near-duplicate code, and identifies documentation likely to be AI generated. Starting from 52,069 extracted candidates, only 25.6 percent satisfy all quality constraints. We provide a detailed analysis of the resulting dataset, which achieves a mean documentation quality score of 6.93 out of 10. Overall, 86.9% of samples contain explicit type annotations, and only 2.9\% are flagged as potentially AI generated. Baseline experiments show that fine-tuning a large language model on Code2Doc yields relative improvements of 29.47% in BLEU and 24.04% in ROUGE-L over zero shot performance, despite the modest dataset size. We release both the dataset and the full curation pipeline to support reproducible research on automatic code documentation generation.
>
---
#### [new 083] Merge on workspaces as Hopf algebra Markov chain
- **分类: math.DS; cs.CL; math.QA; math.RA**

- **简介: 该论文将语言学中的Merge操作建模为Hopf代数上的马尔可夫链，研究其动力学行为。旨在解决句法结构生成中收敛至树形结构的问题，发现常规语言学成本函数不足，而引入香农熵优化可实现收敛，并拓展了语义嵌入、过滤机制与参数设定建模。**

- **链接: [https://arxiv.org/pdf/2512.18861v1](https://arxiv.org/pdf/2512.18861v1)**

> **作者:** Matilde Marcolli; David Skigin
>
> **备注:** 80 pages, LaTeX, 1 png figure
>
> **摘要:** We study the dynamical properties of a Hopf algebra Markov chain with state space the binary rooted forests with labelled leaves. This Markovian dynamical system describes the core computational process of structure formation and transformation in syntax via the Merge operation, according to Chomsky's Minimalism model of generative linguistics. The dynamics decomposes into an ergodic dynamical system with uniform stationary distribution, given by the action of Internal Merge, while the contributions of External Merge and (a minimal form of) Sideward Merge reduce to a simpler Markov chain with state space the set of partitions and with combinatorial weights. The Sideward Merge part of the dynamics prevents convergence to fully formed connected structures (trees), unless the different forms of Merge are weighted by a cost function, as predicted by linguistic theory. Results on the asymptotic behavior of the Perron-Frobenius eigenvalue and eigenvector in this weighted case, obtained in terms of an associated Perron-Frobenius problem in the tropical semiring, show that the usual cost functions (Minimal Search and Resource Restrictions) proposed in the linguistic literature do not suffice to obtain convergence to the tree structures, while an additional optimization property based on the Shannon entropy achieves the expected result for the dynamics. We also comment on the introduction of continuous parameters related to semantic embedding and other computational models, and also on some filtering of the dynamics by coloring rules that model the linguistic filtering by theta roles and phase structure, and on parametric variation and the process of parameter setting in Externalization.
>
---
#### [new 084] Epistemological Fault Lines Between Human and Artificial Intelligence
- **分类: cs.CY; cs.CL; cs.HC**

- **简介: 该论文属哲学与AI交叉研究，旨在揭示人类与大语言模型在认识论上的根本差异。它指出LLMs非认知主体，而是统计模式补全系统，并识别出七类“认识论断层”，提出“Epistemia”概念，警示语言似真性替代真实认知判断的风险。**

- **链接: [https://arxiv.org/pdf/2512.19466v1](https://arxiv.org/pdf/2512.19466v1)**

> **作者:** Walter Quattrociocchi; Valerio Capraro; Matjaž Perc
>
> **备注:** 16 pages, 1 figure
>
> **摘要:** Large language models (LLMs) are widely described as artificial intelligence, yet their epistemic profile diverges sharply from human cognition. Here we show that the apparent alignment between human and machine outputs conceals a deeper structural mismatch in how judgments are produced. Tracing the historical shift from symbolic AI and information filtering systems to large-scale generative transformers, we argue that LLMs are not epistemic agents but stochastic pattern-completion systems, formally describable as walks on high-dimensional graphs of linguistic transitions rather than as systems that form beliefs or models of the world. By systematically mapping human and artificial epistemic pipelines, we identify seven epistemic fault lines, divergences in grounding, parsing, experience, motivation, causal reasoning, metacognition, and value. We call the resulting condition Epistemia: a structural situation in which linguistic plausibility substitutes for epistemic evaluation, producing the feeling of knowing without the labor of judgment. We conclude by outlining consequences for evaluation, governance, and epistemic literacy in societies increasingly organized around generative AI.
>
---
#### [new 085] TICL+: A Case Study On Speech In-Context Learning for Children's Speech Recognition
- **分类: eess.AS; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属儿童语音识别任务，旨在解决儿童语音声学/语言变异大、标注数据少导致的ASR性能差问题。提出TICL+方法，在检索式语音上下文学习（SICL）中引入声学重排序，联合语义与声学对齐选取示例，显著降低词错误率。**

- **链接: [https://arxiv.org/pdf/2512.18263v1](https://arxiv.org/pdf/2512.18263v1)**

> **作者:** Haolong Zheng; Yekaterina Yegorova; Mark Hasegawa-Johnson
>
> **备注:** Published at IEEE ASRU 2025 Satellite Workshop-AI for Children's Speech and Language
>
> **摘要:** Children's speech recognition remains challenging due to substantial acoustic and linguistic variability, limited labeled data, and significant differences from adult speech. Speech foundation models can address these challenges through Speech In-Context Learning (SICL), allowing adaptation to new domains without fine-tuning. However, the effectiveness of SICL depends on how in-context examples are selected. We extend an existing retrieval-based method, Text-Embedding KNN for SICL (TICL), introducing an acoustic reranking step to create TICL+. This extension prioritizes examples that are both semantically and acoustically aligned with the test input. Experiments on four children's speech corpora show that TICL+ achieves up to a 53.3% relative word error rate reduction over zero-shot performance and 37.6% over baseline TICL, highlighting the value of combining semantic and acoustic information for robust, scalable ASR in children's speech.
>
---
#### [new 086] Application of deep learning approaches for medieval historical documents transcription
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属古籍文本识别任务，旨在解决中世纪拉丁文手稿（9–11世纪）OCR准确率低的问题。作者构建专用数据集，设计端到端深度学习流程，涵盖文本检测、单词分割与识别，并评估多项指标，代码开源。**

- **链接: [https://arxiv.org/pdf/2512.18865v1](https://arxiv.org/pdf/2512.18865v1)**

> **作者:** Maksym Voloshchuk; Bohdana Zarembovska; Mykola Kozlenko
>
> **备注:** 15 pages, 15 figures, 4 tables. Originally published by CEUR Workshop Proceedings (CEUR-WS.org, ISSN 1613-0073), available: https://ceur-ws.org/Vol-4133/S_05_Kozlenko.pdf
>
> **摘要:** Handwritten text recognition and optical character recognition solutions show excellent results with processing data of modern era, but efficiency drops with Latin documents of medieval times. This paper presents a deep learning method to extract text information from handwritten Latin-language documents of the 9th to 11th centuries. The approach takes into account the properties inherent in medieval documents. The paper provides a brief introduction to the field of historical document transcription, a first-sight analysis of the raw data, and the related works and studies. The paper presents the steps of dataset development for further training of the models. The explanatory data analysis of the processed data is provided as well. The paper explains the pipeline of deep learning models to extract text information from the document images, from detecting objects to word recognition using classification models and embedding word images. The paper reports the following results: recall, precision, F1 score, intersection over union, confusion matrix, and mean string distance. The plots of the metrics are also included. The implementation is published on the GitHub repository.
>
---
#### [new 087] A Critical Review of Monte Carlo Algorithms Balancing Performance and Probabilistic Accuracy with AI Augmented Framework
- **分类: stat.CO; cs.AI; cs.CL**

- **简介: 该论文属综述任务，旨在解决蒙特卡洛算法中性能与概率精度的权衡问题。工作包括：梳理算法演进脉络，分析时间/空间复杂度界，阐释各算法适用场景，并探讨AI增强框架下的前沿挑战。**

- **链接: [https://arxiv.org/pdf/2512.17968v1](https://arxiv.org/pdf/2512.17968v1)**

> **作者:** Ravi Prasad
>
> **摘要:** Monte Carlo algorithms are a foundational pillar of modern computational science, yet their effective application hinges on a deep understanding of their performance trade offs. This paper presents a critical analysis of the evolution of Monte Carlo algorithms, focusing on the persistent tension between statistical efficiency and computational cost. We describe the historical development from the foundational Metropolis Hastings algorithm to contemporary methods like Hamiltonian Monte Carlo. A central emphasis of this survey is the rigorous discussion of time and space complexity, including upper, lower, and asymptotic tight bounds for each major algorithm class. We examine the specific motivations for developing these methods and the key theoretical and practical observations such as the introduction of gradient information and adaptive tuning in HMC that led to successively better solutions. Furthermore, we provide a justification framework that discusses explicit situations in which using one algorithm is demonstrably superior to another for the same problem. The paper concludes by assessing the profound significance and impact of these algorithms and detailing major current research challenges.
>
---
#### [new 088] Efficient Jailbreak Mitigation Using Semantic Linear Classification in a Multi-Staged Pipeline
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属LLM安全防御任务，旨在解决提示注入与越狱攻击问题。提出轻量多阶段流水线，核心为基于文本归一化、TF-IDF和线性SVM的语义过滤器，实现高精度（93.4%）低延迟（47s）检测，在3万+样本上验证了高效鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.19011v1](https://arxiv.org/pdf/2512.19011v1)**

> **作者:** Akshaj Prashanth Rao; Advait Singh; Saumya Kumaar Saksena; Dhruv Kumar
>
> **备注:** Under Review
>
> **摘要:** Prompt injection and jailbreaking attacks pose persistent security challenges to large language model (LLM)-based systems. We present an efficient and systematically evaluated defense architecture that mitigates these threats through a lightweight, multi-stage pipeline. Its core component is a semantic filter based on text normalization, TF-IDF representations, and a Linear SVM classifier. Despite its simplicity, this module achieves 93.4% accuracy and 96.5% specificity on held-out data, substantially reducing attack throughput while incurring negligible computational overhead. Building on this efficient foundation, the full pipeline integrates complementary detection and mitigation mechanisms that operate at successive stages, providing strong robustness with minimal latency. In comparative experiments, our SVM-based configuration improves overall accuracy from 35.1% to 93.4% while reducing average time to completion from approximately 450s to 47s, yielding over 10 times lower latency than ShieldGemma. These results demonstrate that the proposed design simultaneously advances defensive precision and efficiency, addressing a core limitation of current model-based moderators. Evaluation across a curated corpus of over 30,000 labeled prompts, including benign, jailbreak, and application-layer injections, confirms that staged, resource-efficient defenses can robustly secure modern LLM-driven applications.
>
---
#### [new 089] Distributed Asymmetric Allocation: A Topic Model for Large Imbalanced Corpora in Social Sciences
- **分类: stat.ME; cs.CL**

- **简介: 该论文提出分布式非对称分配（DAA）主题模型，解决社会科学研究中LDA在大规模失衡语料上训练慢、短文档主题碎片化、种子词引导失效三大问题；通过优化Dirichlet先验并融合多算法，提升政治话题句子级识别的准确率与效率。**

- **链接: [https://arxiv.org/pdf/2512.18119v1](https://arxiv.org/pdf/2512.18119v1)**

> **作者:** Kohei Watanabe
>
> **备注:** 34 pages
>
> **摘要:** Social scientists employ latent Dirichlet allocation (LDA) to find highly specific topics in large corpora, but they often struggle in this task because (1) LDA, in general, takes a significant amount of time to fit on large corpora; (2) unsupervised LDA fragments topics into sub-topics in short documents; (3) semi-supervised LDA fails to identify specific topics defined using seed words. To solve these problems, I have developed a new topic model called distributed asymmetric allocation (DAA) that integrates multiple algorithms for efficiently identifying sentences about important topics in large corpora. I evaluate the ability of DAA to identify politically important topics by fitting it to the transcripts of speeches at the United Nations General Assembly between 1991 and 2017. The results show that DAA can classify sentences significantly more accurately and quickly than LDA thanks to the new algorithms. More generally, the results demonstrate that it is important for social scientists to optimize Dirichlet priors of LDA to perform content analysis accurately.
>
---
#### [new 090] From Retrieval to Reasoning: A Framework for Cyber Threat Intelligence NER with Explicit and Adaptive Instructions
- **分类: cs.CR; cs.CL**

- **简介: 该论文聚焦网络安全威胁情报（CTI）中的命名实体识别（NER）任务，旨在解决检索式上下文学习依赖隐式归纳、泛化性差的问题。提出TTPrompt框架，将TTP知识结构化为显式指令层级，并引入反馈驱动的指令优化（FIR），显著提升少样本NER性能。**

- **链接: [https://arxiv.org/pdf/2512.19414v1](https://arxiv.org/pdf/2512.19414v1)**

> **作者:** Jiaren Peng; Hongda Sun; Xuan Tian; Cheng Huang; Zeqing Li; Rui Yan
>
> **摘要:** The automation of Cyber Threat Intelligence (CTI) relies heavily on Named Entity Recognition (NER) to extract critical entities from unstructured text. Currently, Large Language Models (LLMs) primarily address this task through retrieval-based In-Context Learning (ICL). This paper analyzes this mainstream paradigm, revealing a fundamental flaw: its success stems not from global semantic similarity but largely from the incidental overlap of entity types within retrieved examples. This exposes the limitations of relying on unreliable implicit induction. To address this, we propose TTPrompt, a framework shifting from implicit induction to explicit instruction. TTPrompt maps the core concepts of CTI's Tactics, Techniques, and Procedures (TTPs) into an instruction hierarchy: formulating task definitions as Tactics, guiding strategies as Techniques, and annotation guidelines as Procedures. Furthermore, to handle the adaptability challenge of static guidelines, we introduce Feedback-driven Instruction Refinement (FIR). FIR enables LLMs to self-refine guidelines by learning from errors on minimal labeled data, adapting to distinct annotation dialects. Experiments on five CTI NER benchmarks demonstrate that TTPrompt consistently surpasses retrieval-based baselines. Notably, with refinement on just 1% of training data, it rivals models fine-tuned on the full dataset. For instance, on LADDER, its Micro F1 of 71.96% approaches the fine-tuned baseline, and on the complex CTINexus, its Macro F1 exceeds the fine-tuned ACLM model by 10.91%.
>
---
#### [new 091] External Hippocampus: Topological Cognitive Maps for Guiding Large Language Model Reasoning
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出“外部海马体”框架，属AI推理优化任务，旨在解决小语言模型多步推理中的认知僵局问题。通过构建语义空间的拓扑认知地图，实现测试时低开销、可干预的能量流导航，无需训练，提升准确率并加速推理。**

- **链接: [https://arxiv.org/pdf/2512.18190v1](https://arxiv.org/pdf/2512.18190v1)**

> **作者:** Jian Yan
>
> **备注:** 12 pages, 7 figures
>
> **摘要:** This paper proposes the External Hippocampus framework, which models language model reasoning from a cognitive dynamics perspective as the flow of information energy in semantic space. Unlike traditional weight-space optimization methods, this framework constructs topological cognitive maps through dimensionality reduction projection, enabling precise navigation and intervention of energy flow at test time while avoiding substantial computational requirements and demonstrating predictable intervention patterns. The method effectively addresses the cognitive deadlock problem in multi-step reasoning for small models. Experiments on models <=7B parameters show: map-guided methods achieve 81.20% accuracy on 500 challenging problems (relative baseline +16.80%), reduce reasoning time by >= 15x, with key findings revealing that reasoning stagnation manifests as "Cognitive Vortex" and low-entropy potential wells, while temperature perturbations effectively restart energy flow. The framework requires no additional training, possesses autonomous growth capability, and provides an efficient and controllable topological-aware solution for small model reasoning.
>
---
#### [new 092] Layout-Aware Text Editing for Efficient Transformation of Academic PDFs to Markdown
- **分类: cs.MM; cs.CL; cs.CV; cs.DL**

- **简介: 该论文针对学术PDF转Markdown效率低的问题，提出EditTrans模型：结合布局分析与轻量编辑生成，识别需编辑文本后再生成标记，避免重复解码密集文本。任务为文档结构化转换，提升速度44.5%且保持质量。**

- **链接: [https://arxiv.org/pdf/2512.18115v1](https://arxiv.org/pdf/2512.18115v1)**

> **作者:** Changxu Duan
>
> **备注:** Accepted ICDAR 2025
>
> **摘要:** Academic documents stored in PDF format can be transformed into plain text structured markup languages to enhance accessibility and enable scalable digital library workflows. Markup languages allow for easier updates and customization, making academic content more adaptable and accessible to diverse usage, such as linguistic corpus compilation. Such documents, typically delivered in PDF format, contain complex elements including mathematical formulas, figures, headers, and tables, as well as densely layouted text. Existing end-to-end decoder transformer models can transform screenshots of documents into markup language. However, these models exhibit significant inefficiencies; their token-by-token decoding from scratch wastes a lot of inference steps in regenerating dense text that could be directly copied from PDF files. To solve this problem, we introduce EditTrans, a hybrid editing-generation model whose features allow identifying a queue of to-be-edited text from a PDF before starting to generate markup language. EditTrans contains a lightweight classifier fine-tuned from a Document Layout Analysis model on 162,127 pages of documents from arXiv. In our evaluations, EditTrans reduced the transformation latency up to 44.5% compared to end-to-end decoder transformer models, while maintaining transformation quality. Our code and reproducible dataset production scripts are open-sourced.
>
---
#### [new 093] Explainable Transformer-CNN Fusion for Noise-Robust Speech Emotion Recognition
- **分类: cs.SD; cs.CL**

- **简介: 该论文面向噪声鲁棒的语音情感识别（SER）任务，解决真实环境中噪声干扰导致性能下降及模型不可解释问题。提出可解释的Transformer-CNN融合框架，结合Wav2Vec 2.0与1D-CNN，引入注意力时序池化和SHAP/Score-CAM可视化解释机制。**

- **链接: [https://arxiv.org/pdf/2512.18298v1](https://arxiv.org/pdf/2512.18298v1)**

> **作者:** Sudip Chakrabarty; Pappu Bishwas; Rajdeep Chatterjee
>
> **摘要:** Speech Emotion Recognition (SER) systems often degrade in performance when exposed to the unpredictable acoustic interference found in real-world environments. Additionally, the opacity of deep learning models hinders their adoption in trust-sensitive applications. To bridge this gap, we propose a Hybrid Transformer-CNN framework that unifies the contextual modeling of Wav2Vec 2.0 with the spectral stability of 1D-Convolutional Neural Networks. Our dual-stream architecture processes raw waveforms to capture long-range temporal dependencies while simultaneously extracting noise-resistant spectral features (MFCC, ZCR, RMSE) via a custom Attentive Temporal Pooling mechanism. We conducted extensive validation across four diverse benchmark datasets: RAVDESS, TESS, SAVEE, and CREMA-D. To rigorously test robustness, we subjected the model to non-stationary acoustic interference using real-world noise profiles from the SAS-KIIT dataset. The proposed framework demonstrates superior generalization and state-of-the-art accuracy across all datasets, significantly outperforming single-branch baselines under realistic environmental interference. Furthermore, we address the ``black-box" problem by integrating SHAP and Score-CAM into the evaluation pipeline. These tools provide granular visual explanations, revealing how the model strategically shifts attention between temporal and spectral cues to maintain reliability in the presence of complex environmental noise.
>
---
#### [new 094] A Multi-agent Text2SQL Framework using Small Language Models and Execution Feedback
- **分类: cs.DB; cs.AI; cs.CL; cs.HC; cs.MA**

- **简介: 该论文属Text2SQL任务，旨在解决小语言模型（SLMs）在隐私/成本约束下难以胜任复杂SQL生成的问题。提出多智能体框架MATS，通过角色分工、执行反馈驱动的强化学习训练，提升SLM性能，在单GPU上实现媲美大模型的精度。**

- **链接: [https://arxiv.org/pdf/2512.18622v1](https://arxiv.org/pdf/2512.18622v1)**

> **作者:** Thanh Dat Hoang; Thanh Trung Huynh; Matthias Weidlich; Thanh Tam Nguyen; Tong Chen; Hongzhi Yin; Quoc Viet Hung Nguyen
>
> **摘要:** Text2SQL, the task of generating SQL queries from natural language text, is a critical challenge in data engineering. Recently, Large Language Models (LLMs) have demonstrated superior performance for this task due to their advanced comprehension and generation capabilities. However, privacy and cost considerations prevent companies from using Text2SQL solutions based on external LLMs offered as a service. Rather, small LLMs (SLMs) that are openly available and can hosted in-house are adopted. These SLMs, in turn, lack the generalization capabilities of larger LLMs, which impairs their effectiveness for complex tasks such as Text2SQL. To address these limitations, we propose MATS, a novel Text2SQL framework designed specifically for SLMs. MATS uses a multi-agent mechanism that assigns specialized roles to auxiliary agents, reducing individual workloads and fostering interaction. A training scheme based on reinforcement learning aligns these agents using feedback obtained during execution, thereby maintaining competitive performance despite a limited LLM size. Evaluation results using on benchmark datasets show that MATS, deployed on a single- GPU server, yields accuracy that are on-par with large-scale LLMs when using significantly fewer parameters. Our source code and data are available at https://github.com/thanhdath/mats-sql.
>
---
#### [new 095] Stable and Efficient Single-Rollout RL for Multimodal Reasoning
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文面向多模态大模型的强化学习推理优化任务，解决单次采样RL在多模态场景下训练不稳定、易崩溃的问题。提出MSSR框架，通过熵驱动的优势塑形机制实现稳定高效的单次采样RLVR，在保持性能的同时显著提升训练效率。**

- **链接: [https://arxiv.org/pdf/2512.18215v1](https://arxiv.org/pdf/2512.18215v1)**

> **作者:** Rui Liu; Dian Yu; Lei Ke; Haolin Liu; Yujun Zhou; Zhenwen Liang; Haitao Mi; Pratap Tokekar; Dong Yu
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has become a key paradigm to improve the reasoning capabilities of Multimodal Large Language Models (MLLMs). However, prevalent group-based algorithms such as GRPO require multi-rollout sampling for each prompt. While more efficient single-rollout variants have recently been explored in text-only settings, we find that they suffer from severe instability in multimodal contexts, often leading to training collapse. To address this training efficiency-stability trade-off, we introduce $\textbf{MSSR}$ (Multimodal Stabilized Single-Rollout), a group-free RLVR framework that achieves both stable optimization and effective multimodal reasoning performance. MSSR achieves this via an entropy-based advantage-shaping mechanism that adaptively regularizes advantage magnitudes, preventing collapse and maintaining training stability. While such mechanisms have been used in group-based RLVR, we show that in the multimodal single-rollout setting they are not merely beneficial but essential for stability. In in-distribution evaluations, MSSR demonstrates superior training compute efficiency, achieving similar validation accuracy to the group-based baseline with half the training steps. When trained for the same number of steps, MSSR's performance surpasses the group-based baseline and shows consistent generalization improvements across five diverse reasoning-intensive benchmarks. Together, these results demonstrate that MSSR enables stable, compute-efficient, and effective RLVR for complex multimodal reasoning tasks.
>
---
#### [new 096] Seeing Justice Clearly: Handwritten Legal Document Translation with OCR and Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文聚焦 handwritten legal document translation 任务，旨在解决低资源语言（如马拉地语）手写法律文书（如FIR、诉状）的端到端翻译难题。对比传统OCR+MT两阶段流程与视觉大语言模型（VLLM）的统一端到端方法，在自建手写马拉地语法律文档数据集上进行评估。**

- **链接: [https://arxiv.org/pdf/2512.18004v1](https://arxiv.org/pdf/2512.18004v1)**

> **作者:** Shubham Kumar Nigam; Parjanya Aditya Shukla; Noel Shallum; Arnab Bhattacharya
>
> **备注:** Accepted in AILaw @ AAAI 2026 Conference
>
> **摘要:** Handwritten text recognition (HTR) and machine translation continue to pose significant challenges, particularly for low-resource languages like Marathi, which lack large digitized corpora and exhibit high variability in handwriting styles. The conventional approach to address this involves a two-stage pipeline: an OCR system extracts text from handwritten images, which is then translated into the target language using a machine translation model. In this work, we explore and compare the performance of traditional OCR-MT pipelines with Vision Large Language Models that aim to unify these stages and directly translate handwritten text images in a single, end-to-end step. Our motivation is grounded in the urgent need for scalable, accurate translation systems to digitize legal records such as FIRs, charge sheets, and witness statements in India's district and high courts. We evaluate both approaches on a curated dataset of handwritten Marathi legal documents, with the goal of enabling efficient legal document processing, even in low-resource environments. Our findings offer actionable insights toward building robust, edge-deployable solutions that enhance access to legal information for non-native speakers and legal professionals alike.
>
---
#### [new 097] Watch Closely: Mitigating Object Hallucinations in Large Vision-Language Models with Disentangled Decoding
- **分类: cs.CV; cs.CL**

- **简介: 该论文面向大视觉语言模型（LVLMs）的对象幻觉问题，提出无需训练的幻觉解耦解码（HDD）方法：通过图像分割增强视觉输入，并引入空白图像抑制语言先验幻觉，从而协同缓解视觉与语言双模态幻觉。**

- **链接: [https://arxiv.org/pdf/2512.19070v1](https://arxiv.org/pdf/2512.19070v1)**

> **作者:** Ruiqi Ma; Yu Yan; Chunhong Zhang; Minghao Yin; XinChao Liu; Zhihong Jin; Zheng Hu
>
> **摘要:** Large Vision-Language Models (LVLMs) bridge the gap between visual and linguistic modalities, demonstrating strong potential across a variety of domains. However, despite significant progress, LVLMs still suffer from severe hallucination issues in object recognition tasks. These models often fail to accurately identify certain objects, leading to text generation that appears fluent but does not correspond to the visual content, which can have serious consequences in real-world applications. Recently, several methods have been proposed to alleviate LVLM hallucinations, but most focus solely on reducing hallucinations in the language modality. To mitigate hallucinations in both the language and visual modalities, we introduce Hallucination Disentangled Decoding (HDD) method that requires no training. HDD enhances the original image by segmenting it and selecting images that augment the original, while also utilizing a blank image to eliminate language prior hallucinations in both the original and segmented images. This design not only reduces the model's dependence on language priors but also enhances its visual performance. (Code: https://github.com/rickeyhhh/Hallucination-Disentangled-Decoding)
>
---
#### [new 098] BanglaForge: LLM Collaboration with Self-Refinement for Bangla Code Generation
- **分类: cs.SE; cs.CL**

- **简介: 该论文属低资源语言代码生成任务，旨在解决Bangla语缺乏标注数据和工具导致的自然语言到代码转换难题。提出BanglaForge框架，融合检索增强、双模型协作（ coder + reviewer）与基于执行反馈的迭代自优化，显著提升Bangla代码生成准确率。**

- **链接: [https://arxiv.org/pdf/2512.19122v1](https://arxiv.org/pdf/2512.19122v1)**

> **作者:** Mahir Labib Dihan; Sadif Ahmed; Md Nafiu Rahman
>
> **备注:** Accepted at BLP Workshop @ IJCNLP-AACL 2025. Code is available at https://github.com/mahirlabibdihan/BanglaForge
>
> **摘要:** Bangla is a low-resource language for code generation, lacking large-scale annotated datasets and tools to transform natural language specifications into executable programs. This makes Bangla-to-code generation a challenging task requiring innovative solutions. To address this, we introduce BanglaForge, a novel framework for generating code from Bangla function descriptions. BanglaForge leverages a retrieval-augmented dual-model collaboration paradigm with self-refinement, combining in-context learning, llm-based translation, systematic prompt engineering, and iterative self-refinement based on execution feedback, where a coder generates initial solutions and a reviewer enhances them for robustness. On the BLP-2025 Bangla Code Generation benchmark, BanglaForge achieves a competitive Pass@1 accuracy of 84.00%, demonstrating the effectiveness of retrieval, model collaboration, and self-refinement for low-resource Bangla code generation.
>
---
#### [new 099] Bottom-up Policy Optimization: Your Language Model Policy Secretly Contains Internal Policies
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属大语言模型强化学习任务，旨在解决忽视LLM内部策略结构导致优化低效的问题。作者提出“自底向上策略优化”（BuPO），通过分解残差流与模块化策略，分析各层熵变规律，并在早期层直接优化内部策略，提升复杂推理性能。**

- **链接: [https://arxiv.org/pdf/2512.19673v1](https://arxiv.org/pdf/2512.19673v1)**

> **作者:** Yuqiao Tan; Minzheng Wang; Shizhu He; Huanxuan Liao; Chengfeng Zhao; Qiunan Lu; Tian Liang; Jun Zhao; Kang Liu
>
> **备注:** Preprint. Our code is available at https://github.com/Trae1ounG/BuPO
>
> **摘要:** Existing reinforcement learning (RL) approaches treat large language models (LLMs) as a single unified policy, overlooking their internal mechanisms. Understanding how policy evolves across layers and modules is therefore crucial for enabling more targeted optimization and raveling out complex reasoning mechanisms. In this paper, we decompose the language model policy by leveraging the intrinsic split of the Transformer residual stream and the equivalence between the composition of hidden states with the unembedding matrix and the resulting samplable policy. This decomposition reveals Internal Layer Policies, corresponding to contributions from individual layers, and Internal Modular Policies, which align with the self-attention and feed-forward network (FFN) components within each layer. By analyzing the entropy of internal policy, we find that: (a) Early layers keep high entropy for exploration, top layers converge to near-zero entropy for refinement, with convergence patterns varying across model series. (b) LLama's prediction space rapidly converges in the final layer, whereas Qwen-series models, especially Qwen3, exhibit a more human-like, progressively structured reasoning pattern. Motivated by these findings, we propose Bottom-up Policy Optimization (BuPO), a novel RL paradigm that directly optimizes the internal layer policy during early training. By aligning training objective at lower layer, BuPO reconstructs foundational reasoning capabilities and achieves superior performance. Extensive experiments on complex reasoning benchmarks demonstrates the effectiveness of our method. Our code is available at https://github.com/Trae1ounG/BuPO.
>
---
#### [new 100] Investigating Spatial Attention Bias in Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属视觉-语言模型（VLM）的偏差分析任务，旨在揭示其空间注意力偏置问题。研究发现VLM普遍优先描述水平拼接图像中左侧内容（约97%），且该偏置跨架构、跨语言（含阿拉伯语模型）存在，与训练数据标注指南无关，表明源于模型架构本身。**

- **链接: [https://arxiv.org/pdf/2512.18231v1](https://arxiv.org/pdf/2512.18231v1)**

> **作者:** Aryan Chaudhary; Sanchit Goyal; Pratik Narang; Dhruv Kumar
>
> **摘要:** Vision-Language Models have demonstrated remarkable capabilities in understanding visual content, yet systematic biases in their spatial processing remain largely unexplored. This work identifies and characterizes a systematic spatial attention bias where VLMs consistently prioritize describing left-positioned content before right-positioned content in horizontally concatenated images. Through controlled experiments on image pairs using both open-source and closed-source models, we demonstrate that this bias persists across different architectures, with models describing left-positioned content first in approximately 97% of cases under neutral prompting conditions. Testing on an Arabic-finetuned model reveals that the bias persists despite right-to-left language training, ruling out language reading direction as the primary cause. Investigation of training dataset annotation guidelines from PixMo and Visual Genome reveals no explicit left-first ordering instructions, suggesting the bias is consistent with architectural factors rather than explicit training data instructions. These findings reveal fundamental limitations in how current VLMs process spatial information.
>
---
## 更新

#### [replaced 001] Towards a resource for multilingual lexicons: an MT assisted and human-in-the-loop multilingual parallel corpus with multi-word expression annotation
- **分类: cs.CL; cs.AI**

- **简介: 该论文构建多语平行语料库AlphaMWE，聚焦多词表达（尤其动词性MWE）的跨语言对齐与标注。任务属NLP资源建设，旨在解决MT系统翻译MWE困难的问题。工作包括：基于PARSEME英语语料，经MT初译、人工后编辑、双轮质检与MWE标注，覆盖6种语言及阿拉伯语方言，并分析主流MT系统的MWE翻译错误类型。**

- **链接: [https://arxiv.org/pdf/2011.03783v2](https://arxiv.org/pdf/2011.03783v2)**

> **作者:** Lifeng Han; Najet Hadj Mohamed; Malak Rassem; Gareth Jones; Alan Smeaton; Goran Nenadic
>
> **备注:** Accepted by Journal of LRE, extended work from WS paper AlphaMWE
>
> **摘要:** In this work, we introduce the construction of a machine translation (MT) assisted and human-in-the-loop multilingual parallel corpus with annotations of multi-word expressions (MWEs), named AlphaMWE. The MWEs include verbal MWEs (vMWEs) defined in the PARSEME shared task that have a verb as the head of the studied terms. The annotated vMWEs are also bilingually and multilingually aligned manually. The languages covered include Arabic, Chinese, English, German, Italian, and Polish, of which, the Arabic corpus includes both standard and dialectal variations from Egypt and Tunisia. Our original English corpus is extracted from the PARSEME shared task in 2018. We performed machine translation of this source corpus followed by human post-editing and annotation of target MWEs. Strict quality control was applied for error limitation, i.e., each MT output sentence received first manual post-editing and annotation plus a second manual quality rechecking till annotators' consensus is reached. One of our findings during corpora preparation is that accurate translation of MWEs presents challenges to MT systems, as reflected by the outcomes of human-in-the-loop metric HOPE. To facilitate further MT research, we present a categorisation of the error types encountered by MT systems in performing MWE-related translation. To acquire a broader view of MT issues, we selected four popular state-of-the-art MT systems for comparison, namely Microsoft Bing Translator, GoogleMT, Baidu Fanyi, and DeepL MT. Because of the noise removal, translation post-editing, and MWE annotation by human professionals, we believe the AlphaMWE data set will be an asset for both monolingual and cross-lingual research, such as multi-word term lexicography, MT, and information extraction.
>
---
#### [replaced 002] A Survey on Agentic Security: Applications, Threats and Defenses
- **分类: cs.CL; cs.AI; cs.CR**

- **简介: 该论文是综述任务，旨在系统梳理“智能体安全”领域。它围绕应用、威胁、防御三大支柱，构建涵盖160+论文的分类体系，分析架构趋势并指出模型与多模态覆盖的研究缺口。**

- **链接: [https://arxiv.org/pdf/2510.06445v2](https://arxiv.org/pdf/2510.06445v2)**

> **作者:** Asif Shahriar; Md Nafiu Rahman; Sadif Ahmed; Farig Sadeque; Md Rizwan Parvez
>
> **摘要:** In this work we present the first holistic survey of the agentic security landscape, structuring the field around three fundamental pillars: Applications, Threats, and Defenses. We provide a comprehensive taxonomy of over 160 papers, explaining how agents are used in downstream cybersecurity applications, inherent threats to agentic systems, and countermeasures designed to protect them. A detailed cross-cutting analysis shows emerging trends in agent architecture while revealing critical research gaps in model and modality coverage. A complete and continuously updated list of all surveyed papers is publicly available at https://github.com/kagnlp/Awesome-Agentic-Security.
>
---
#### [replaced 003] SoK: Are Watermarks in LLMs Ready for Deployment?
- **分类: cs.CR; cs.CL**

- **简介: 该论文属系统性综述（SoK）任务，旨在评估LLM水印技术的实用部署成熟度。针对模型窃取风险，作者构建水印分类法、提出IP分类器、分析现有方法局限，并通过实验揭示水印损害模型效用，指出其尚未达到工业落地要求。**

- **链接: [https://arxiv.org/pdf/2506.05594v2](https://arxiv.org/pdf/2506.05594v2)**

> **作者:** Kieu Dang; Phung Lai; NhatHai Phan; Yelong Shen; Ruoming Jin; Abdallah Khreishah; My T. Thai
>
> **摘要:** Large Language Models (LLMs) have transformed natural language processing, demonstrating impressive capabilities across diverse tasks. However, deploying these models introduces critical risks related to intellectual property violations and potential misuse, particularly as adversaries can imitate these models to steal services or generate misleading outputs. We specifically focus on model stealing attacks, as they are highly relevant to proprietary LLMs and pose a serious threat to their security, revenue, and ethical deployment. While various watermarking techniques have emerged to mitigate these risks, it remains unclear how far the community and industry have progressed in developing and deploying watermarks in LLMs. To bridge this gap, we aim to develop a comprehensive systematization for watermarks in LLMs by 1) presenting a detailed taxonomy for watermarks in LLMs, 2) proposing a novel intellectual property classifier to explore the effectiveness and impacts of watermarks on LLMs under both attack and attack-free environments, 3) analyzing the limitations of existing watermarks in LLMs, and 4) discussing practical challenges and potential future directions for watermarks in LLMs. Through extensive experiments, we show that despite promising research outcomes and significant attention from leading companies and community to deploy watermarks, these techniques have yet to reach their full potential in real-world applications due to their unfavorable impacts on model utility of LLMs and downstream tasks. Our findings provide an insightful understanding of watermarks in LLMs, highlighting the need for practical watermarks solutions tailored to LLM deployment.
>
---
#### [replaced 004] SCARE: A Benchmark for SQL Correction and Question Answerability Classification for Reliable EHR Question Answering
- **分类: cs.CL; cs.DB**

- **简介: 该论文提出SCARE基准，面向EHR问答中的SQL安全验证任务，解决LLM生成错误SQL威胁临床安全的问题。工作包括构建4200条三元组数据集，联合评估问题可答性分类与SQL纠错，并 benchmark 多类后验验证方法。**

- **链接: [https://arxiv.org/pdf/2511.17559v2](https://arxiv.org/pdf/2511.17559v2)**

> **作者:** Gyubok Lee; Woosog Chay; Edward Choi
>
> **备注:** ML4H 2025 Proceedings
>
> **摘要:** Recent advances in Large Language Models (LLMs) have enabled the development of text-to-SQL models that allow clinicians to query structured data stored in Electronic Health Records (EHRs) using natural language. However, deploying these models for EHR question answering (QA) systems in safety-critical clinical environments remains challenging: incorrect SQL queries-whether caused by model errors or problematic user inputs-can undermine clinical decision-making and jeopardize patient care. While prior work has mainly focused on improving SQL generation accuracy or filtering questions before execution, there is a lack of a unified benchmark for evaluating independent post-hoc verification mechanisms (i.e., a component that inspects and validates the generated SQL before execution), which is crucial for safe deployment. To fill this gap, we introduce SCARE, a benchmark for evaluating methods that function as a post-hoc safety layer in EHR QA systems. SCARE evaluates the joint task of (1) classifying question answerability (i.e., determining whether a question is answerable, ambiguous, or unanswerable) and (2) verifying or correcting candidate SQL queries. The benchmark comprises 4,200 triples of questions, candidate SQL queries, and expected model outputs, grounded in the MIMIC-III, MIMIC-IV, and eICU databases. It covers a diverse set of questions and corresponding candidate SQL queries generated by seven different text-to-SQL models, ensuring a realistic and challenging evaluation. Using SCARE, we benchmark a range of approaches-from two-stage methods to agentic frameworks. Our experiments reveal a critical trade-off between question classification and SQL error correction, highlighting key challenges and outlining directions for future research.
>
---
#### [replaced 005] ADePT: Adaptive Decomposed Prompt Tuning for Parameter-Efficient Fine-tuning
- **分类: cs.CL**

- **简介: 该论文属参数高效微调任务，旨在解决DePT中位置固定、共享的嵌入偏移导致泛化差与优化不足的问题。提出ADePT：用轻量级token共享FFN替代低秩矩阵，实现输入自适应、token级偏移，性能更优且不增推理开销。**

- **链接: [https://arxiv.org/pdf/2501.03291v3](https://arxiv.org/pdf/2501.03291v3)**

> **作者:** Pengwei Tang; Xiaolin Hu; Yong Liu
>
> **备注:** Published at ICLR 2025
>
> **摘要:** Prompt Tuning (PT) enables the adaptation of Pre-trained Large Language Models (PLMs) to downstream tasks by optimizing a small amount of soft virtual tokens, which are prepended to the input token embeddings. Recently, Decomposed Prompt Tuning (DePT) has demonstrated superior adaptation capabilities by decomposing the soft prompt into a shorter soft prompt and a pair of low-rank matrices. The product of the pair of low-rank matrices is added to the input token embeddings to offset them. Additionally, DePT achieves faster inference compared to PT due to the shorter soft prompt. However, in this paper, we find that the position-based token embedding offsets of DePT restrict its ability to generalize across diverse model inputs, and that the shared embedding offsets across many token embeddings result in sub-optimization. To tackle these issues, we introduce Adaptive Decomposed Prompt Tuning (ADePT), which is composed of a short soft prompt and a shallow token-shared feed-forward neural network. ADePT utilizes the token-shared feed-forward neural network to learn the embedding offsets for each token, enabling adaptive embedding offsets that vary according to the model input and better optimization of token embedding offsets. This enables ADePT to achieve superior adaptation performance without requiring more inference time or additional trainable parameters compared to vanilla PT and its variants. In comprehensive experiments across 23 natural language processing tasks and 4 typical PLMs of different scales, ADePT consistently surpasses the other leading parameter-efficient fine-tuning methods, and even outperforms the full fine-tuning in certain scenarios. We also provide a theoretical analysis towards ADePT. Code is available at https://github.com/HungerPWAY/ADePT.
>
---
#### [replaced 006] Adaptation of Agentic AI
- **分类: cs.AI; cs.CL**

- **简介: 该论文属AI系统设计任务，旨在解决agentic AI适应性不足的问题。提出统一框架，将适应分为代理端（工具执行/输出信号驱动）和工具端（代理无关/监督）两类，厘清设计空间、权衡与选型路径，并综述方法、挑战与机遇。**

- **链接: [https://arxiv.org/pdf/2512.16301v2](https://arxiv.org/pdf/2512.16301v2)**

> **作者:** Pengcheng Jiang; Jiacheng Lin; Zhiyi Shi; Zifeng Wang; Luxi He; Yichen Wu; Ming Zhong; Peiyang Song; Qizheng Zhang; Heng Wang; Xueqiang Xu; Hanwen Xu; Pengrui Han; Dylan Zhang; Jiashuo Sun; Chaoqi Yang; Kun Qian; Tian Wang; Changran Hu; Manling Li; Quanzheng Li; Hao Peng; Sheng Wang; Jingbo Shang; Chao Zhang; Jiaxuan You; Liyuan Liu; Pan Lu; Yu Zhang; Heng Ji; Yejin Choi; Dawn Song; Jimeng Sun; Jiawei Han
>
> **摘要:** Cutting-edge agentic AI systems are built on foundation models that can be adapted to plan, reason, and interact with external tools to perform increasingly complex and specialized tasks. As these systems grow in capability and scope, adaptation becomes a central mechanism for improving performance, reliability, and generalization. In this paper, we unify the rapidly expanding research landscape into a systematic framework that spans both agent adaptations and tool adaptations. We further decompose these into tool-execution-signaled and agent-output-signaled forms of agent adaptation, as well as agent-agnostic and agent-supervised forms of tool adaptation. We demonstrate that this framework helps clarify the design space of adaptation strategies in agentic AI, makes their trade-offs explicit, and provides practical guidance for selecting or switching among strategies during system design. We then review the representative approaches in each category, analyze their strengths and limitations, and highlight key open challenges and future opportunities. Overall, this paper aims to offer a conceptual foundation and practical roadmap for researchers and practitioners seeking to build more capable, efficient, and reliable agentic AI systems.
>
---
#### [replaced 007] Look Twice before You Leap: A Rational Agent Framework for Localized Adversarial Anonymization
- **分类: cs.CR; cs.CL**

- **简介: 该论文面向本地化文本匿名化任务，解决现有LLM方案依赖远程API引发隐私悖论、迁移到小模型导致效用崩溃的问题。提出无训练、纯本地的RLAA框架，通过攻击者-仲裁者-匿名器架构及理性早停机制，在保障隐私前提下显著提升文本效用。**

- **链接: [https://arxiv.org/pdf/2512.06713v2](https://arxiv.org/pdf/2512.06713v2)**

> **作者:** Donghang Duan; Xu Zheng; Yuefeng He; Chong Mu; Leyi Cai; Lizong Zhang
>
> **备注:** 17 pages, 9 figures, 6 tables. Revised version with an updated author list, expanded experimental results and analysis
>
> **摘要:** Current LLM-based text anonymization frameworks usually rely on remote API services from powerful LLMs, which creates an inherent privacy paradox: users must disclose data to untrusted third parties for guaranteed privacy preservation. Moreover, directly migrating current solutions to local small-scale models (LSMs) offers a suboptimal solution with severe utility collapse. Our work argues that this failure stems not merely from the capability deficits of LSMs, but significantly from the inherent irrationality of the greedy adversarial strategies employed by current state-of-the-art (SOTA) methods. To address this, we propose Rational Localized Adversarial Anonymization (RLAA), a fully localized and training-free framework featuring an Attacker-Arbitrator-Anonymizer architecture. We model the anonymization process as a trade-off between Marginal Privacy Gain (MPG) and Marginal Utility Cost (MUC), and demonstrate that greedy strategies tend to drift into an irrational state. Instead, RLAA introduces an arbitrator that acts as a rationality gatekeeper, validating the attacker's inference to filter out feedback providing negligible privacy benefits. This mechanism promotes a rational early-stopping criterion, and structurally prevents utility collapse. Extensive experiments on different benchmarks demonstrate that RLAA achieves a superior privacy-utility trade-off compared to strong baselines.
>
---
#### [replaced 008] Decoding Neural Emotion Patterns through Large Language Model Embeddings
- **分类: cs.CL**

- **简介: 该论文属计算神经科学与情感计算交叉任务，旨在无需神经影像即可将文本情绪映射到脑区。提出基于LLM嵌入的框架，通过降维聚类将语义表征关联18个情绪相关脑区，并在健康/抑郁数据、GoEmotions及人机文本对比中验证其神经合理性与临床区分力。**

- **链接: [https://arxiv.org/pdf/2508.09337v3](https://arxiv.org/pdf/2508.09337v3)**

> **作者:** Gideon Vos; Maryam Ebrahimpour; Liza van Eijk; Zoltan Sarnyai; Mostafa Rahimi Azghadi
>
> **备注:** 26 pages, 9 figures
>
> **摘要:** Understanding how emotional expression in language relates to brain function is a challenge in computational neuroscience and affective computing. Traditional neuroimaging is costly and lab-bound, but abundant digital text offers new avenues for emotion-brain mapping. Prior work has largely examined neuroimaging-based emotion localization or computational text analysis separately, with little integration. We propose a computational framework that maps textual emotional content to anatomically defined brain regions without requiring neuroimaging. Using OpenAI's text-embedding-ada-002, we generate high-dimensional semantic representations, apply dimensionality reduction and clustering to identify emotional groups, and map them to 18 brain regions linked to emotional processing. Three experiments were conducted: i) analyzing conversational data from healthy vs. depressed subjects (DIAC-WOZ dataset) to compare mapping patterns, ii) applying the method to the GoEmotions dataset and iii) comparing human-written text with large language model (LLM) responses to assess differences in inferred brain activation. Emotional intensity was scored via lexical analysis. Results showed neuroanatomically plausible mappings with high spatial specificity. Depressed subjects exhibited greater limbic engagement tied to negative affect. Discrete emotions were successfully differentiated. LLM-generated text matched humans in basic emotion distribution but lacked nuanced activation in empathy and self-referential regions (medial prefrontal and posterior cingulate cortex). This cost-effective, scalable approach enables large-scale analysis of naturalistic language, distinguishes between clinical populations, and offers a brain-based benchmark for evaluating AI emotional expression.
>
---
#### [replaced 009] Exploration vs Exploitation: Rethinking RLVR through Clipping, Entropy, and Spurious Reward
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究RLVR（带可验证奖励的强化学习）中探索-利用权衡问题，旨在解释为何“抑制探索”（熵最小化）和“抑制利用”（虚假奖励）均能提升LLM数学推理性能。作者分析熵与性能关系，揭示剪辑偏差导致熵降低，并提出奖励错位模型解释虚假奖励的有效性。**

- **链接: [https://arxiv.org/pdf/2512.16912v2](https://arxiv.org/pdf/2512.16912v2)**

> **作者:** Peter Chen; Xiaopeng Li; Ziniu Li; Wotao Yin; Xi Chen; Tianyi Lin
>
> **备注:** 35 pages
>
> **摘要:** This paper examines the exploration-exploitation trade-off in reinforcement learning with verifiable rewards (RLVR), a framework for improving the reasoning of Large Language Models (LLMs). Recent studies suggest that RLVR can elicit strong mathematical reasoning in LLMs through two seemingly paradoxical mechanisms: spurious rewards, which suppress exploitation by rewarding outcomes unrelated to the ground truth, and entropy minimization, which suppresses exploration by pushing the model toward more confident and deterministic outputs, highlighting a puzzling dynamic: both discouraging exploitation and discouraging exploration improve reasoning performance, yet the underlying principles that reconcile these effects remain poorly understood. We focus on two fundamental questions: (i) how policy entropy relates to performance, and (ii) whether spurious rewards yield gains, potentially through the interplay of clipping bias and model contamination. Our results show that clipping bias under spurious rewards reduces policy entropy, leading to more confident and deterministic outputs, while entropy minimization alone is insufficient for improvement. We further propose a reward-misalignment model explaining why spurious rewards can enhance performance beyond contaminated settings. Our findings clarify the mechanisms behind spurious-reward benefits and provide principles for more effective RLVR training.
>
---
#### [replaced 010] How Reliable are Causal Probing Interventions?
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属可解释性研究任务，旨在评估因果探针干预方法的可靠性。针对现有方法理论基础存疑、缺乏系统实证评估的问题，作者定义“完整性”与“选择性”两大准则，提出以调和平均衡量的“可靠性”，并构建实证框架对比线性/非线性等方法，发现非线性干预更可靠。**

- **链接: [https://arxiv.org/pdf/2408.15510v5](https://arxiv.org/pdf/2408.15510v5)**

> **作者:** Marc Canby; Adam Davies; Chirag Rastogi; Julia Hockenmaier
>
> **备注:** In Proceedings of IJCNLP-AACL, 2025
>
> **摘要:** Causal probing aims to analyze foundation models by examining how intervening on their representation of various latent properties impacts their outputs. Recent works have cast doubt on the theoretical basis of several leading causal probing methods, but it has been unclear how to systematically evaluate the effectiveness of these methods in practice. To address this, we define two key causal probing desiderata: completeness (how thoroughly the representation of the target property has been transformed) and selectivity (how little non-targeted properties have been impacted). We find that there is an inherent tradeoff between the two, which we define as reliability, their harmonic mean. We introduce an empirical analysis framework to measure and evaluate these quantities, allowing us to make the first direct comparisons between different families of leading causal probing methods (e.g., linear vs. nonlinear, or concept removal vs. counterfactual interventions). We find that: (1) all methods show a clear tradeoff between completeness and selectivity; (2) more complete and reliable methods have a greater impact on LLM behavior; and (3) nonlinear interventions are almost always more reliable than linear interventions. Our project webpage is available at: https://ahdavies6.github.io/causal_probing_reliability/
>
---
#### [replaced 011] SAEs Are Good for Steering -- If You Select the Right Features
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属可解释AI与模型编辑任务，旨在解决SAE特征选择对模型“ steering”（输出调控）效果不佳的问题。提出输入/输出双评分法区分特征类型，发现高输出分特征更适于steering；过滤低输出分特征后，steering效果提升2–3倍，媲美监督方法。**

- **链接: [https://arxiv.org/pdf/2505.20063v2](https://arxiv.org/pdf/2505.20063v2)**

> **作者:** Dana Arad; Aaron Mueller; Yonatan Belinkov
>
> **摘要:** Sparse Autoencoders (SAEs) have been proposed as an unsupervised approach to learn a decomposition of a model's latent space. This enables useful applications such as steering - influencing the output of a model towards a desired concept - without requiring labeled data. Current methods identify SAE features to steer by analyzing the input tokens that activate them. However, recent work has highlighted that activations alone do not fully describe the effect of a feature on the model's output. In this work, we draw a distinction between two types of features: input features, which mainly capture patterns in the model's input, and output features, which have a human-understandable effect on the model's output. We propose input and output scores to characterize and locate these types of features, and show that high values for both scores rarely co-occur in the same features. These findings have practical implications: after filtering out features with low output scores, we obtain 2-3x improvements when steering with SAEs, making them competitive with supervised methods.
>
---
#### [replaced 012] Reliable Decision Support with LLMs: A Framework for Evaluating Consistency in Binary Text Classification Applications
- **分类: cs.CL; cs.LG; stat.ML**

- **简介: 该论文面向LLM二元文本分类任务，解决其可靠性评估缺失问题。提出基于心理测量学的评估框架，涵盖样本量确定、无效响应检测及内外部一致性度量，并在金融新闻情感分类中验证14个模型的性能与一致性。**

- **链接: [https://arxiv.org/pdf/2505.14918v2](https://arxiv.org/pdf/2505.14918v2)**

> **作者:** Fadel M. Megahed; Ying-Ju Chen; L. Allision Jones-Farmer; Younghwa Lee; Jiawei Brooke Wang; Inez M. Zwetsloot
>
> **备注:** 26 pages
>
> **摘要:** This study introduces a framework for evaluating consistency in large language model (LLM) binary text classification, addressing the lack of established reliability assessment methods. Adapting psychometric principles, we determine sample size requirements, develop metrics for invalid responses, and evaluate intra- and inter-rater reliability. Our case study examines financial news sentiment classification across 14 LLMs (including claude-3-7-sonnet, gpt-4o, deepseek-r1, gemma3, llama3.2, phi4, and command-r-plus), with five replicates per model on 1,350 articles. Models demonstrated high intra-rater consistency, achieving perfect agreement on 90-98% of examples, with minimal differences between expensive and economical models from the same families. When validated against StockNewsAPI labels, models achieved strong performance (accuracy 0.76-0.88), with smaller models like gemma3:1B, llama3.2:3B, and claude-3-5-haiku outperforming larger counterparts. All models performed at chance when predicting actual market movements, indicating task constraints rather than model limitations. Our framework provides systematic guidance for LLM selection, sample size planning, and reliability assessment, enabling organizations to optimize resources for classification tasks.
>
---
#### [replaced 013] VLDBench Evaluating Multimodal Disinformation with Regulatory Alignment
- **分类: cs.CL**

- **简介: 该论文提出VLDBench，首个面向监管对齐的多模态虚假信息检测基准。针对现有工作忽视有意图的图文协同虚假信息问题，构建6.2万标注图文对，涵盖13类新闻类别。实验证明视觉线索可提升检测准确率5–35个百分点，支持评估、微调与鲁棒性测试。**

- **链接: [https://arxiv.org/pdf/2502.11361v5](https://arxiv.org/pdf/2502.11361v5)**

> **作者:** Shaina Raza; Ashmal Vayani; Aditya Jain; Aravind Narayanan; Vahid Reza Khazaie; Syed Raza Bashir; Elham Dolatabadi; Gias Uddin; Christos Emmanouilidis; Rizwan Qureshi; Mubarak Shah
>
> **备注:** Accepted in Information Fusion Journal
>
> **摘要:** Detecting disinformation that blends manipulated text and images has become increasingly challenging, as AI tools make synthetic content easy to generate and disseminate. While most existing AI safety benchmarks focus on single modality misinformation (i.e., false content shared without intent to deceive), intentional multimodal disinformation, such as propaganda or conspiracy theories that imitate credible news, remains largely unaddressed. We introduce the Vision-Language Disinformation Detection Benchmark (VLDBench), the first large-scale resource supporting both unimodal (text-only) and multimodal (text + image) disinformation detection. VLDBench comprises approximately 62,000 labeled text-image pairs across 13 categories, curated from 58 news outlets. Using a semi-automated pipeline followed by expert review, 22 domain experts invested over 500 hours to produce high-quality annotations with substantial inter-annotator agreement. Evaluations of state-of-the-art Large Language Models (LLMs) and Vision-Language Models (VLMs) on VLDBench show that incorporating visual cues improves detection accuracy by 5 to 35 percentage points over text-only models. VLDBench provides data and code for evaluation, fine-tuning, and robustness testing to support disinformation analysis. Developed in alignment with AI governance frameworks (e.g., the MIT AI Risk Repository), VLDBench offers a principled foundation for advancing trustworthy disinformation detection in multimodal media. Project: https://vectorinstitute.github.io/VLDBench/ Dataset: https://huggingface.co/datasets/vector-institute/VLDBench Code: https://github.com/VectorInstitute/VLDBench
>
---
#### [replaced 014] AdaCtrl: Towards Adaptive and Controllable Reasoning via Difficulty-Aware Budgeting
- **分类: cs.AI; cs.CL**

- **简介: 该论文属大模型推理控制任务，旨在解决推理链过长导致的效率低效问题。提出AdaCtrl框架，通过难度感知的动态预算分配与用户可控的显式长度调节，结合两阶段训练（冷启动微调+难度感知RL），实现自适应与可干预的推理深度调控。**

- **链接: [https://arxiv.org/pdf/2505.18822v2](https://arxiv.org/pdf/2505.18822v2)**

> **作者:** Shijue Huang; Hongru Wang; Wanjun Zhong; Zhaochen Su; Jiazhan Feng; Bowen Cao; Yi R. Fung
>
> **摘要:** Modern large reasoning models demonstrate impressive problem-solving capabilities by employing sophisticated reasoning strategies. However, they often struggle to balance efficiency and effectiveness, frequently generating unnecessarily lengthy reasoning chains for simple problems. In this work, we propose AdaCtrl, a novel framework to support both difficulty-aware adaptive reasoning budget allocation and explicit user control over reasoning depth. AdaCtrl dynamically adjusts its reasoning length based on self-assessed problem difficulty, while also allowing users to manually control the budget to prioritize either efficiency or effectiveness. This is achieved through a two-stage training pipeline: an initial cold-start fine-tuning phase to instill the ability to self-aware difficulty and adjust reasoning budget, followed by a difficulty-aware reinforcement learning (RL) stage that refines the model's adaptive reasoning strategies and calibrates its difficulty assessments based on its evolving capabilities during online training. To enable intuitive user interaction, we design explicit length-triggered tags that function as a natural interface for budget control. Empirical results show that AdaCtrl adapts reasoning length based on estimated difficulty, compared to the standard training baseline that also incorporates fine-tuning and RL, it yields performance improvements and simultaneously reduces response length by 10.06% and 12.14% on the more challenging AIME2024 and AIME2025 datasets, which require elaborate reasoning, and by 62.05% and 91.04% on the MATH500 and GSM8K datasets, where more concise responses are sufficient. Furthermore, AdaCtrl enables precise user control over the reasoning budget, allowing for tailored responses to meet specific needs.
>
---
#### [replaced 015] LexChain: Modeling Legal Reasoning Chains for Chinese Tort Case Analysis
- **分类: cs.CL**

- **简介: 该论文提出LexChain框架，专用于中文侵权民事案件的法律推理建模。针对现有方法忽视民事案、推理链不细的问题，它将推理分解为三模块多步流程，构建评测基准，并设计基于该框架的提示与微调基线，显著提升大模型在侵权推理上的表现。**

- **链接: [https://arxiv.org/pdf/2510.17602v2](https://arxiv.org/pdf/2510.17602v2)**

> **作者:** Huiyuan Xie; Chenyang Li; Huining Zhu; Chubin Zhang; Yuxiao Ye; Zhenghao Liu; Zhiyuan Liu
>
> **摘要:** Legal reasoning is a fundamental component of legal analysis and decision-making. Existing computational approaches to legal reasoning predominantly rely on generic reasoning frameworks such as syllogism, which do not comprehensively examine the nuanced process of legal reasoning. Moreover, current research has largely focused on criminal cases, with insufficient modeling for civil cases. In this work, we present a novel framework to explicitly model legal reasoning in the analysis of Chinese tort-related civil cases. We first operationalize the legal reasoning process in tort analysis into the three-module LexChain framework, with each module consisting of multiple finer-grained sub-steps. Informed by the LexChain framework, we introduce the task of tort legal reasoning and construct an evaluation benchmark to systematically assess the critical steps within analytical reasoning chains for tort analysis. Leveraging this benchmark, we evaluate existing large language models for their legal reasoning ability in civil tort contexts. Our results indicate that current models still fall short in accurately handling crucial elements of tort legal reasoning. Furthermore, we introduce several baseline approaches that explicitly incorporate LexChain-style reasoning through prompting or post-training. The proposed baselines achieve significant improvements in tort-related legal reasoning and generalize well to related legal analysis tasks, demonstrating the value of explicitly modeling legal reasoning chains to enhance the reasoning capabilities of language models.
>
---
#### [replaced 016] The Reasoning Lingua Franca: A Double-Edged Sword for Multilingual AI
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究多语言大推理模型（LRM）的跨语言推理问题，揭示其默认英语推理虽提升准确率（尤其在复杂任务中），却易因翻译失真导致“迷失于翻译”错误。作者在MGSM和GPQA Diamond任务上对比英/非英语言推理，分析认知行为与准确性，指出英语中心策略的双刃剑效应。**

- **链接: [https://arxiv.org/pdf/2510.20647v2](https://arxiv.org/pdf/2510.20647v2)**

> **作者:** Alan Saji; Raj Dabre; Anoop Kunchukuttan; Ratish Puduppully
>
> **备注:** 14 pages, 13 figures, 5 tables
>
> **摘要:** Large Reasoning Models (LRMs) achieve strong performance on mathematical, scientific, and other question-answering tasks, but their multilingual reasoning abilities remain underexplored. When presented with non-English questions, LRMs often default to reasoning in English, raising concerns about interpretability and the handling of linguistic and cultural nuances. We systematically compare an LRM's reasoning in English versus the language of the question. Our evaluation spans two tasks: MGSM and GPQA Diamond. Beyond measuring answer accuracy, we also analyze cognitive attributes in the reasoning traces. We find that English reasoning traces exhibit a substantially higher presence of these cognitive behaviors, and that reasoning in English generally yields higher final-answer accuracy, with the performance gap increasing as tasks become more complex. However, this English-centric strategy is susceptible to a key failure mode - getting "Lost in Translation," where translation steps lead to errors that would have been avoided by question's language reasoning.
>
---
#### [replaced 017] WebATLAS: An LLM Agent with Experience-Driven Memory and Action Simulation
- **分类: cs.LG; cs.AI; cs.CL; cs.IR; cs.MA; cs.RO**

- **简介: 该论文提出WebATLAS，一种无需微调的LLM网页代理，旨在解决长周期、跨网站导航任务中效率低、泛化差的问题。通过经验驱动记忆与动作前瞻模拟，在认知空间内规划-仿真-评估，构建持久认知地图，显著提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2510.22732v2](https://arxiv.org/pdf/2510.22732v2)**

> **作者:** Jiali Cheng; Anjishnu Kumar; Roshan Lal; Rishi Rajasekaran; Hani Ramezani; Omar Zia Khan; Oleg Rokhlenko; Sunny Chiu-Webster; Gang Hua; Hadi Amiri
>
> **备注:** 9 pages, NeurIPS 2025 Workshop on Language Agents and World Models
>
> **摘要:** Large Language Model (LLM) web agents often struggle with long-horizon web navigation and web task completion in new websites, producing inefficient action sequences unless fine-tuned on environment-specific data. We show that experience-driven memory, combined with look-ahead action simulation, is sufficient for LLM agents to adapt to unseen web environments by remembering past failures and predicting the consequences of future actions. We introduce WebATLAS (Actor-Critic Task-completion with Look-ahead Action Simulation), a memory-augmented LLM web agent that learns a lightweight internal model of the environment from interaction experience and performs hypothetical action rollouts before acting in the real world. WebATLAS builds a persistent cognitive map via curiosity-driven exploration, stores interaction outcomes as experience-based memory, and evaluates candidate actions in cognitive space using a planner--simulator--critic loop. This enables the agent to reuse past experience, avoid previously unsuccessful behaviors, and generate more efficient plans. We evaluate WebATLAS on the WebArena-Lite benchmark for autonomous web navigation and demonstrate a success rate of 63%, outperforming the previous state-of-the-art at 53.9%. Unlike previous systems, our modular architecture requires no website-specific LLM fine-tuning. Ablation studies confirm that experience-driven memory, look-ahead action simulation, and hierarchical replanning play complementary roles in enabling robust, training-free web agents.
>
---
#### [replaced 018] Kronecker Factorization Improves Efficiency and Interpretability of Sparse Autoencoders
- **分类: cs.LG; cs.CL**

- **简介: 该论文属可解释AI任务，旨在解决大规模稀疏自编码器（SAE）训练与解释效率低的问题。提出KronSAE架构，用Kronecker分解压缩隐空间；并设计mAND激活函数，提升因子化框架下的可解释性与性能。**

- **链接: [https://arxiv.org/pdf/2505.22255v3](https://arxiv.org/pdf/2505.22255v3)**

> **作者:** Vadim Kurochkin; Yaroslav Aksenov; Daniil Laptev; Daniil Gavrilov; Nikita Balagansky
>
> **摘要:** Sparse Autoencoders (SAEs) have demonstrated significant promise in interpreting the hidden states of language models by decomposing them into interpretable latent directions. However, training and interpreting SAEs at scale remains challenging, especially when large dictionary sizes are used. While decoders can leverage sparse-aware kernels for efficiency, encoders still require computationally intensive linear operations with large output dimensions. To address this, we propose KronSAE, a novel architecture that factorizes the latent representation via Kronecker product decomposition, drastically reducing memory and computational overhead. Furthermore, we introduce mAND, a differentiable activation function approximating the binary AND operation, which improves interpretability and performance in our factorized framework.
>
---
#### [replaced 019] Dagstuhl Perspectives Workshop 24352 -- Conversational Agents: A Framework for Evaluation (CAFE): Manifesto
- **分类: cs.CL; cs.HC; cs.IR**

- **简介: 该论文属人机交互评估任务，旨在解决对话式信息访问（CONIAC）系统缺乏统一评估框架的问题。作者提出CAFE评估框架，包含六大核心组件：利益相关者目标、用户任务、用户特征、评估标准、方法学及量化指标。**

- **链接: [https://arxiv.org/pdf/2506.11112v2](https://arxiv.org/pdf/2506.11112v2)**

> **作者:** Christine Bauer; Li Chen; Nicola Ferro; Norbert Fuhr; Avishek Anand; Timo Breuer; Guglielmo Faggioli; Ophir Frieder; Hideo Joho; Jussi Karlgren; Johannes Kiesel; Bart P. Knijnenburg; Aldo Lipani; Lien Michiels; Andrea Papenmeier; Maria Soledad Pera; Mark Sanderson; Scott Sanner; Benno Stein; Johanne R. Trippas; Karin Verspoor; Martijn C Willemsen
>
> **备注:** 10 figures; Dagstuhl Manifestos, 11(1), pp 19-67. DOI: 10.4230/DagMan.11.1.19
>
> **摘要:** During the workshop, we deeply discussed what CONversational Information ACcess (CONIAC) is and its unique features, proposing a world model abstracting it, and defined the Conversational Agents Framework for Evaluation (CAFE) for the evaluation of CONIAC systems, consisting of six major components: 1) goals of the system's stakeholders, 2) user tasks to be studied in the evaluation, 3) aspects of the users carrying out the tasks, 4) evaluation criteria to be considered, 5) evaluation methodology to be applied, and 6) measures for the quantitative criteria chosen.
>
---
#### [replaced 020] Label Words as Local Task Vectors in In-Context Learning
- **分类: cs.CL**

- **简介: 该论文研究大语言模型的上下文学习（ICL）机制，旨在解释模型如何从示例中提取任务规则。它指出：并非所有任务都依赖全局任务向量；对分类等需多示例推理的任务，模型先在各答案位置形成局部任务向量，再逐步聚合抽象规则。**

- **链接: [https://arxiv.org/pdf/2406.16007v2](https://arxiv.org/pdf/2406.16007v2)**

> **作者:** Bowen Zheng; Ming Ma; Zhongqiao Lin; Tianming Yang
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable abilities, one of the most important being in-context learning (ICL). With ICL, LLMs can derive the underlying rule from a few demonstrations and provide answers that comply with the rule. Previous work hypothesized that the network creates a task vector in specific positions during ICL. The task vector can be computed by averaging across the dataset. It conveys the overall task information and can thus be considered global. Patching the global task vector allows LLMs to achieve zero-shot performance with dummy inputs comparable to few-shot learning. However, we find that such a global task vector does not exist in all tasks, especially in tasks that rely on rules that can only be inferred from multiple demonstrations, such as categorization tasks. Instead, the information provided by each demonstration is first transmitted to its answer position and forms a local task vector associated with the demonstration. In some tasks but not in categorization tasks, all demonstrations' local task vectors converge in later layers, forming the global task vector. We further show that local task vectors encode a high-level abstraction of rules extracted from the demonstrations. Our study provides novel insights into the mechanism underlying ICL in LLMs, demonstrating how ICL may be achieved through an information aggregation mechanism.
>
---
#### [replaced 021] Abstract, Align, Predict: Zero-Shot Stance Detection via Cognitive Inductive Reasoning
- **分类: cs.CL**

- **简介: 该论文面向零-shot立场检测任务，解决模型对未见目标立场判断能力弱、泛化差、可解释性低的问题。提出认知归纳推理框架（CIRF），通过无监督构建多关系模式图并结合图核对齐实现可解释、强泛化的零样本推理。**

- **链接: [https://arxiv.org/pdf/2506.13470v2](https://arxiv.org/pdf/2506.13470v2)**

> **作者:** Bowen Zhang; Jun Ma; Fuqiang Niu; Li Dong; Jinzhou Cao; Genan Dai
>
> **摘要:** Zero-shot stance detection (ZSSD) seeks to determine the stance of text toward previously unseen targets, a task critical for analyzing dynamic and polarized online discourse with limited labeled data. While large language models (LLMs) offer zero-shot capabilities, prompting-based approaches often fall short in handling complex reasoning and lack robust generalization to novel targets. Meanwhile, LLM-enhanced methods still require substantial labeled data and struggle to move beyond instance-level patterns, limiting their interpretability and adaptability. Inspired by cognitive science, we propose the Cognitive Inductive Reasoning Framework (CIRF), a schema-driven method that bridges linguistic inputs and abstract reasoning via automatic induction and application of cognitive reasoning schemas. CIRF abstracts first-order logic patterns from raw text into multi-relational schema graphs in an unsupervised manner, and leverages a schema-enhanced graph kernel model to align input structures with schema templates for robust, interpretable zero-shot inference. Extensive experiments on SemEval-2016, VAST, and COVID-19-Stance benchmarks demonstrate that CIRF not only establishes new state-of-the-art results, but also achieves comparable performance with just 30\% of the labeled data, demonstrating its strong generalization and efficiency in low-resource settings.
>
---
#### [replaced 022] Learning from Self Critique and Refinement for Faithful LLM Summarization
- **分类: cs.CL**

- **简介: 该论文针对LLM摘要中的幻觉问题，提出自监督训练方法SCRPO：利用模型自身批判与精炼能力构建偏好数据集，并通过偏好学习提升摘要忠实度。在XSUM、CNN/DM、SAMSum上验证了其有效性与高效性。**

- **链接: [https://arxiv.org/pdf/2512.05387v2](https://arxiv.org/pdf/2512.05387v2)**

> **作者:** Ting-Yao Hu; Hema Swetha Koppula; Hadi Pouransari; Cem Koc; Oncel Tuzel; Raviteja Vemulapalli
>
> **摘要:** Large Language Models (LLMs) often suffer from hallucinations: output content that is not grounded in the input context, when performing long-form text generation tasks such as summarization. Prior works have shown that hallucinations can be reduced by iteratively critiquing and refining previously generated outputs using either the same model or a more powerful teacher model as the critique. However, these approaches either require additional test-time compute or assume access to more powerful teacher models, making them costly and less practical. In this work, we propose Self Critique and Refinement-based Preference Optimization (SCRPO), which is a self-supervised training framework that first constructs a preference dataset by leveraging the LLM's own critique and refinement capabilities, and then applies preference learning to improve the same LLM for faithful summarization. Experiments on three summarization benchmarks (XSUM CNNDM and SAMSum), demonstrate that our approach outperforms state-of-the-art self-supervised learning methods in terms of faithfulness metrics while either maintaining or improving other metrics that measure the overall quality of the summary. Moreover, compared to test-time refinement, our approach not only improves efficiency but also results in more faithful summaries.
>
---
#### [replaced 023] LiveSecBench: A Dynamic and Event-Driven Safety Benchmark for Chinese Language Model Applications
- **分类: cs.CL**

- **简介: 该论文提出LiveSecBench，一个面向中文大模型应用的动态、事件驱动的安全评测基准。旨在解决中文LLM安全评估滞后、场景覆盖不足的问题。工作包括构建人机协同的高质量数据集，按五维度持续更新评测，并对57个模型进行ELO排名。**

- **链接: [https://arxiv.org/pdf/2511.02366v2](https://arxiv.org/pdf/2511.02366v2)**

> **作者:** Yudong Li; Peiru Yang; Feng Huang; Zhongliang Yang; Kecheng Wang; Haitian Li; Baocheng Chen; Xingyu An; Ziyu Liu; Youdan Yang; Kejiang Chen; Sifang Wan; Xu Wang; Yufei Sun; Liyan Wu; Ruiqi Zhou; Wenya Wen; Xingchi Gu; Tianxin Zhang; Yue Gao; Yongfeng Huang
>
> **摘要:** We introduce LiveSecBench, a continuously updated safety benchmark specifically for Chinese-language LLM application scenarios. LiveSecBench constructs a high-quality and unique dataset through a pipeline that combines automated generation with human verification. By periodically releasing new versions to expand the dataset and update evaluation metrics, LiveSecBench provides a robust and up-to-date standard for AI safety. In this report, we introduce our second release v251215, which evaluates across five dimensions (Public Safety, Fairness & Bias, Privacy, Truthfulness, and Mental Health Safety.) We evaluate 57 representative LLMs using an ELO rating system, offering a leaderboard of the current state of Chinese LLM safety. The result is available at https://livesecbench.intokentech.cn/.
>
---
#### [replaced 024] Efficient and Stealthy Jailbreak Attacks via Adversarial Prompt Distillation from LLMs to SLMs
- **分类: cs.CL; cs.CR**

- **简介: 该论文属AI安全任务，旨在解决LLM jailbreak攻击依赖大模型、效率低、难部署的问题。提出“对抗提示蒸馏”框架，将LLM的越狱能力蒸馏至小模型（SLMs），实现高效、隐蔽、资源友好的跨模型攻击。**

- **链接: [https://arxiv.org/pdf/2506.17231v2](https://arxiv.org/pdf/2506.17231v2)**

> **作者:** Xiang Li; Chong Zhang; Jia Wang; Fangyu Wu; Yushi Li; Xiaobo Jin
>
> **备注:** 19 pages, 7 figures
>
> **摘要:** As the scale and complexity of jailbreaking attacks on large language models (LLMs) continue to escalate, their efficiency and practical applicability are constrained, posing a profound challenge to LLM security. Jailbreaking techniques have advanced from manual prompt engineering to automated methodologies. Recent advances have automated jailbreaking approaches that harness LLMs to generate jailbreak instructions and adversarial examples, delivering encouraging results. Nevertheless, these methods universally include an LLM generation phase, which, due to the complexities of deploying and reasoning with LLMs, impedes effective implementation and broader adoption. To mitigate this issue, we introduce \textbf{Adversarial Prompt Distillation}, an innovative framework that integrates masked language modeling, reinforcement learning, and dynamic temperature control to distill LLM jailbreaking prowess into smaller language models (SLMs). This methodology enables efficient, robust jailbreak attacks while maintaining high success rates and accommodating a broader range of application contexts. Empirical evaluations affirm the approach's superiority in attack efficacy, resource optimization, and cross-model versatility. Our research underscores the practicality of transferring jailbreak capabilities to SLMs, reveals inherent vulnerabilities in LLMs, and provides novel insights to advance LLM security investigations. Our code is available at: https://github.com/lxgem/Efficient_and_Stealthy_Jailbreak_Attacks_via_Adversarial_Prompt.
>
---
#### [replaced 025] Zero-Overhead Introspection for Adaptive Test-Time Compute
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出ZIP-RC方法，解决大语言模型缺乏推理时实时奖赏与计算成本预估能力的问题。它在单次前向传播中零开销预测最终奖励和剩余长度，实现自适应采样决策，提升准确率并降低平均计算开销。**

- **链接: [https://arxiv.org/pdf/2512.01457v3](https://arxiv.org/pdf/2512.01457v3)**

> **作者:** Rohin Manvi; Joey Hong; Tim Seyde; Maxime Labonne; Mathias Lechner; Sergey Levine
>
> **摘要:** Large language models excel at reasoning but lack key aspects of introspection, including anticipating their own success and the computation required to achieve it. Humans use real-time introspection to decide how much effort to invest, when to make multiple attempts, when to stop, and when to signal success or failure. Without this, LLMs struggle to make intelligent meta-cognition decisions. Test-time scaling methods like Best-of-N drive up cost and latency by using a fixed budget of samples regardless of the marginal benefit of each one at any point in generation, and the absence of confidence signals can mislead people, prevent appropriate escalation to better tools, and undermine trustworthiness. Learned verifiers or reward models can provide confidence estimates, but do not enable adaptive inference and add substantial cost by requiring extra models or forward passes. We present ZIP-RC, an adaptive inference method that equips models with zero-overhead inference-time predictions of reward and cost. At every token, ZIP-RC reuses reserved or unused logits in the same forward pass as next-token prediction to output a joint distribution over final reward and remaining length -- no extra models, architecture change, or inference overhead. This full joint distribution is used to compute a sampling utility which is the linear combination of the expected maximum reward, total compute, and latency of set of samples if generated to completion. During inference, we maximize this utility with meta-actions that determine which prefix of tokens to continue or initiate sampling from. On mixed-difficulty mathematical benchmarks, ZIP-RC improves accuracy by up to 12% over majority voting at equal or lower average cost, and traces smooth Pareto frontiers between quality, compute, and latency. By providing real-time reward-cost introspection, ZIP-RC enables adaptive, efficient reasoning.
>
---
#### [replaced 026] Vision Language Models are Confused Tourists
- **分类: cs.CV; cs.CL**

- **简介: 该论文属多模态鲁棒性评估任务，旨在解决VLM在多元文化混合输入下性能骤降的问题。作者提出ConfusedTourist基准，通过图像堆叠等扰动测试模型对地理文化线索的稳定性，并发现其因注意力被干扰而失效，揭示了文化鲁棒性短板。**

- **链接: [https://arxiv.org/pdf/2511.17004v2](https://arxiv.org/pdf/2511.17004v2)**

> **作者:** Patrick Amadeus Irawan; Ikhlasul Akmal Hanif; Muhammad Dehan Al Kautsar; Genta Indra Winata; Fajri Koto; Alham Fikri Aji
>
> **摘要:** Although the cultural dimension has been one of the key aspects in evaluating Vision-Language Models (VLMs), their ability to remain stable across diverse cultural inputs remains largely untested, despite being crucial to support diversity and multicultural societies. Existing evaluations often rely on benchmarks featuring only a singular cultural concept per image, overlooking scenarios where multiple, potentially unrelated cultural cues coexist. To address this gap, we introduce ConfusedTourist, a novel cultural adversarial robustness suite designed to assess VLMs' stability against perturbed geographical cues. Our experiments reveal a critical vulnerability, where accuracy drops heavily under simple image-stacking perturbations and even worsens with its image-generation-based variant. Interpretability analyses further show that these failures stem from systematic attention shifts toward distracting cues, diverting the model from its intended focus. These findings highlight a critical challenge: visual cultural concept mixing can substantially impair even state-of-the-art VLMs, underscoring the urgent need for more culturally robust multimodal understanding.
>
---
#### [replaced 027] RadAgents: Multimodal Agentic Reasoning for Chest X-ray Interpretation with Radiologist-like Workflows
- **分类: cs.MA; cs.CL; cs.CV**

- **简介: 该论文面向胸部X光片（CXR）自动解读任务，旨在解决现有方法临床不可解释、视觉-文本模态融合不足、缺乏不一致检测与验证机制三大问题。提出RadAgents多智能体框架，嵌入放射科医生工作流，结合临床先验、多模态推理、视觉接地与检索增强验证，提升结果的可靠性、透明性与临床一致性。**

- **链接: [https://arxiv.org/pdf/2509.20490v3](https://arxiv.org/pdf/2509.20490v3)**

> **作者:** Kai Zhang; Corey D Barrett; Jangwon Kim; Lichao Sun; Tara Taghavi; Krishnaram Kenthapadi
>
> **备注:** ML4H'25; Work in progress
>
> **摘要:** Agentic systems offer a potential path to solve complex clinical tasks through collaboration among specialized agents, augmented by tool use and external knowledge bases. Nevertheless, for chest X-ray (CXR) interpretation, prevailing methods remain limited: (i) reasoning is frequently neither clinically interpretable nor aligned with guidelines, reflecting mere aggregation of tool outputs; (ii) multimodal evidence is insufficiently fused, yielding text-only rationales that are not visually grounded; and (iii) systems rarely detect or resolve cross-tool inconsistencies and provide no principled verification mechanisms. To bridge the above gaps, we present RadAgents, a multi-agent framework that couples clinical priors with task-aware multimodal reasoning and encodes a radiologist-style workflow into a modular, auditable pipeline. In addition, we integrate grounding and multimodal retrieval-augmentation to verify and resolve context conflicts, resulting in outputs that are more reliable, transparent, and consistent with clinical practice.
>
---
#### [replaced 028] Understanding Syllogistic Reasoning in LLMs from Formal and Natural Language Perspectives
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型（LLMs）的三段论推理能力，属自然语言处理与形式逻辑交叉任务。旨在探究LLMs是否具备符号化形式推理能力，而非仅模拟人类直觉推理。作者评估14个LLM在符号推理与自然语言理解两方面的表现。**

- **链接: [https://arxiv.org/pdf/2512.12620v2](https://arxiv.org/pdf/2512.12620v2)**

> **作者:** Aheli Poddar; Saptarshi Sahoo; Sujata Ghosh
>
> **备注:** 9 pages, 4 figures, 5 tables. Submitted to AAAI 2026 Bridge Program on Logic & AI. Code available at https://github.com/XAheli/Logic-in-LLMs
>
> **摘要:** We study syllogistic reasoning in LLMs from the logical and natural language perspectives. In process, we explore fundamental reasoning capabilities of the LLMs and the direction this research is moving forward. To aid in our studies, we use 14 large language models and investigate their syllogistic reasoning capabilities in terms of symbolic inferences as well as natural language understanding. Even though this reasoning mechanism is not a uniform emergent property across LLMs, the perfect symbolic performances in certain models make us wonder whether LLMs are becoming more and more formal reasoning mechanisms, rather than making explicit the nuances of human reasoning.
>
---
#### [replaced 029] Style Over Story: A Process-Oriented Study of Authorial Creativity in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属AI创造力评估任务，旨在解决LLM创作过程不透明、仅重输出质量的问题。作者引入约束决策视角，通过控制提示赋予模型作者身份，分析其对风格、人物等要素的偏好及推理，发现LLM普遍优先选择“风格”，并构建跨模型创造力分析新方法。**

- **链接: [https://arxiv.org/pdf/2510.02025v2](https://arxiv.org/pdf/2510.02025v2)**

> **作者:** Donghoon Jung; Jiwoo Choi; Songeun Chae; Seohyon Jung
>
> **摘要:** Evaluations of large language models (LLMs)' creativity have focused primarily on the quality of their outputs rather than the processes that shape them. This study takes a process-oriented approach, drawing on narratology to examine LLMs as computational authors. We introduce constraint-based decision-making as a lens for authorial creativity. Using controlled prompting to assign authorial personas, we analyze the creative preferences of the models. Our findings show that LLMs consistently emphasize Style over other elements, including Character, Event, and Setting. By also probing the reasoning the models provide for their choices, we show that distinctive profiles emerge across models and argue that our approach provides a novel systematic tool for analyzing AI's authorial creativity.
>
---
#### [replaced 030] Generative Retrieval with Few-shot Indexing
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属生成式检索任务，旨在解决训练式索引成本高、难适配动态语料等问题。提出无训练的少样本索引框架：用LLM一次性生成全库文档ID构建docid银行，检索时约束LLM在该银行内生成相关docid。引入一对多映射提升性能。**

- **链接: [https://arxiv.org/pdf/2408.02152v2](https://arxiv.org/pdf/2408.02152v2)**

> **作者:** Arian Askari; Chuan Meng; Mohammad Aliannejadi; Zhaochun Ren; Evangelos Kanoulas; Suzan Verberne
>
> **备注:** Accepted for publication at the 48th European Conference on Information Retrieval (ECIR 2026)
>
> **摘要:** Existing generative retrieval (GR) methods rely on training-based indexing, which fine-tunes a model to memorise associations between queries and the document identifiers (docids) of relevant documents. Training-based indexing suffers from high training costs, under-utilisation of pre-trained knowledge in large language models (LLMs), and limited adaptability to dynamic document corpora. To address the issues, we propose a few-shot indexing-based GR framework (Few-Shot GR). It has a few-shot indexing process without any training, where we prompt an LLM to generate docids for all documents in a corpus, ultimately creating a docid bank for the entire corpus. During retrieval, we feed a query to the same LLM and constrain it to generate a docid within the docid bank created during indexing, and then map the generated docid back to its corresponding document. Moreover, we devise few-shot indexing with one-to-many mapping to further enhance Few-Shot GR. Experiments show that Few-Shot GR achieves superior performance to state-of-the-art GR methods requiring heavy training.
>
---
#### [replaced 031] In Good GRACEs: Principled Teacher Selection for Knowledge Distillation
- **分类: cs.LG; cs.CL**

- **简介: 该论文属知识蒸馏任务，旨在解决教师模型选择依赖试错、成本高的问题。提出轻量级指标GRACE，仅基于学生梯度分布评估教师适配性，无需教师输出或测试数据；实验证明其能准确预测蒸馏效果并指导温度、模型规模等关键设计。**

- **链接: [https://arxiv.org/pdf/2511.02833v3](https://arxiv.org/pdf/2511.02833v3)**

> **作者:** Abhishek Panigrahi; Bingbin Liu; Sadhika Malladi; Sham Kakade; Surbhi Goel
>
> **摘要:** Knowledge distillation is an efficient strategy to use data generated by large "teacher" language models to train smaller capable "student" models, but selecting the optimal teacher for a specific student-task combination requires expensive trial-and-error. We propose a lightweight score called GRACE to quantify how effective a teacher will be for post-training a student model. GRACE measures distributional properties of the student's gradients without access to a verifier, teacher logits, teacher internals, or test data. From an information-theoretic perspective, GRACE connects to leave-one-out stability of gradient-based algorithms, which controls the generalization performance of the distilled students. On GSM8K and MATH, GRACE correlates strongly (up to 86% Spearman correlation) with the performance of the distilled LLaMA and OLMo students. In particular, training a student using the GRACE-selected teacher can improve the performance by up to 7.4% over naively using the best-performing teacher. Further, GRACE can provide guidance on crucial design choices in distillation, including (1) the best temperature to use when generating from the teacher, (2) the best teacher to use given a size constraint, and (3) the best teacher to use within a specific model family. Altogether, our findings demonstrate that GRACE can efficiently and effectively identify a strongly compatible teacher for a given student and provide fine-grained guidance on how to perform distillation.
>
---
#### [replaced 032] Tree-OPO: Off-policy Monte Carlo Tree-Guided Advantage Optimization for Multistep Reasoning
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属大语言模型推理优化任务，旨在解决 verifier-guided RL 中多步推理策略训练时优势估计方差大的问题。提出 Tree-OPO 框架：将 MCTS 轨迹构建成树状前缀课程，并设计 Staged Advantage Estimation（SAE）方法，实现低方差、前缀感知的优势计算，提升 GRPO 策略优化效率与数学推理准确率。**

- **链接: [https://arxiv.org/pdf/2509.09284v3](https://arxiv.org/pdf/2509.09284v3)**

> **作者:** Bingning Huang; Tu Nguyen; Matthieu Zimmer
>
> **摘要:** Recent advances in reasoning with large language models (LLMs) have shown the effectiveness of Monte Carlo Tree Search (MCTS) for generating high quality intermediate trajectories, particularly in math and symbolic domains. Inspired by this, we explore how MCTS derived trajectories, traditionally used for training value or reward models, can be repurposed to improve policy optimization in verifier guided reinforcement learning (RL). Specifically, we focus on Group Relative Policy Optimization (GRPO), a recent algorithm that enables consistent policy learning from group relative judgments. We reframe GRPO into a staged training paradigm, leveraging a teacher's MCTS rollouts to construct a tree structured curriculum of prefixes. This introduces the novel challenge of computing advantages for training samples that originate from different prefixes, each with a distinct expected return. To address this, we propose Staged Advantage Estimation (SAE), a framework for computing low variance, prefix aware advantages by projecting rewards onto a constraint set that respects the tree's hierarchy. Our empirical results on mathematical reasoning tasks show that SAE improves final accuracy over standard GRPO. This outcome is grounded in our theoretical analysis, which confirms that SAE reduces gradient variance, a principled path to improved sample efficiency. We demonstrate this through practical SAE implementations, comparing efficient heuristics against a formal quadratic program.
>
---
#### [replaced 033] AutoAdv: Automated Adversarial Prompting for Multi-Turn Jailbreaking of Large Language Models
- **分类: cs.CL; cs.AI; cs.CR; cs.LG**

- **简介: 该论文属AI安全任务，旨在解决多轮对话中LLM易受对抗提示攻击（即“越狱”）的问题。作者提出无训练框架AutoAdv，融合模式管理、温度调节和两阶段重写机制，实现高成功率多轮越狱攻击，揭示现有单轮对齐方法在多轮场景下的脆弱性。**

- **链接: [https://arxiv.org/pdf/2511.02376v3](https://arxiv.org/pdf/2511.02376v3)**

> **作者:** Aashray Reddy; Andrew Zagula; Nicholas Saban
>
> **备注:** Presented at NeurIPS 2025 Lock-LLM Workshop. Code is available at https://github.com/AAN-AutoAdv/AutoAdv
>
> **摘要:** Large Language Models (LLMs) remain vulnerable to jailbreaking attacks where adversarial prompts elicit harmful outputs. Yet most evaluations focus on single-turn interactions while real-world attacks unfold through adaptive multi-turn conversations. We present AutoAdv, a training-free framework for automated multi-turn jailbreaking that achieves an attack success rate of up to 95% on Llama-3.1-8B within six turns, a 24% improvement over single-turn baselines. AutoAdv uniquely combines three adaptive mechanisms: a pattern manager that learns from successful attacks to enhance future prompts, a temperature manager that dynamically adjusts sampling parameters based on failure modes, and a two-phase rewriting strategy that disguises harmful requests and then iteratively refines them. Extensive evaluation across commercial and open-source models (Llama-3.1-8B, GPT-4o mini, Qwen3-235B, Mistral-7B) reveals persistent vulnerabilities in current safety mechanisms, with multi-turn attacks consistently outperforming single-turn approaches. These findings demonstrate that alignment strategies optimized for single-turn interactions fail to maintain robustness across extended conversations, highlighting an urgent need for multi-turn-aware defenses.
>
---
#### [replaced 034] Arc Gradient Descent: A Mathematically Derived Reformulation of Gradient Descent with Phase-Aware, User-Controlled Step Dynamics
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.NE**

- **简介: 该论文提出新型优化器ArcGD，旨在改进梯度下降的步长动态与相位感知能力。针对非凸优化难题（如高维 Rosenbrock 函数）和深度学习训练（CIFAR-10 多架构 MLP），验证其收敛性、泛化性与鲁棒性，显著提升测试精度并缓解过拟合。**

- **链接: [https://arxiv.org/pdf/2512.06737v2](https://arxiv.org/pdf/2512.06737v2)**

> **作者:** Nikhil Verma; Joonas Linnosmaa; Leonardo Espinosa-Leal; Napat Vajragupta
>
> **备注:** 80 pages, 6 tables, 2 figures, 5 appendices, proof-of-concept
>
> **摘要:** The paper presents the formulation, implementation, and evaluation of the ArcGD optimiser. The evaluation is conducted initially on a non-convex benchmark function and subsequently on a real-world ML dataset. The initial comparative study using the Adam optimiser is conducted on a stochastic variant of the highly non-convex and notoriously challenging Rosenbrock function, renowned for its narrow, curved valley, across dimensions ranging from 2D to 1000D and an extreme case of 50,000D. Two configurations were evaluated to eliminate learning-rate bias: (i) both using ArcGD's effective learning rate and (ii) both using Adam's default learning rate. ArcGD consistently outperformed Adam under the first setting and, although slower under the second, achieved superior final solutions in most cases. In the second evaluation, ArcGD is evaluated against state-of-the-art optimizers (Adam, AdamW, Lion, SGD) on the CIFAR-10 image classification dataset across 8 diverse MLP architectures ranging from 1 to 5 hidden layers. ArcGD achieved the highest average test accuracy (50.7%) at 20,000 iterations, outperforming AdamW (46.6%), Adam (46.8%), SGD (49.6%), and Lion (43.4%), winning or tying on 6 of 8 architectures. Notably, while Adam and AdamW showed strong early convergence at 5,000 iterations, but regressed with extended training, whereas ArcGD continued improving, demonstrating generalization and resistance to overfitting without requiring early stopping tuning. Strong performance on geometric stress tests and standard deep-learning benchmarks indicates broad applicability, highlighting the need for further exploration. Moreover, it is also shown that a limiting variant of ArcGD can be interpreted as a sign-based momentum-like update, highlighting conceptual connections between the inherent mechanisms of ArcGD and the Lion optimiser.
>
---
#### [replaced 035] RAID: Refusal-Aware and Integrated Decoding for Jailbreaking LLMs
- **分类: cs.CL**

- **简介: 该论文属于AI安全任务，旨在解决大语言模型（LLM）易受“越狱”攻击、绕过安全机制生成有害内容的问题。作者提出RAID框架：在嵌入空间联合优化对抗后缀，引入拒绝感知正则化与连贯性约束，并通过批评器引导解码生成自然有效的攻击后缀。**

- **链接: [https://arxiv.org/pdf/2510.13901v2](https://arxiv.org/pdf/2510.13901v2)**

> **作者:** Tuan T. Nguyen; John Le; Thai T. Vu; Willy Susilo; Heath Cooper
>
> **摘要:** Large language models (LLMs) achieve impressive performance across diverse tasks yet remain vulnerable to jailbreak attacks that bypass safety mechanisms. We present RAID (Refusal-Aware and Integrated Decoding), a framework that systematically probes these weaknesses by crafting adversarial suffixes that induce restricted content while preserving fluency. RAID relaxes discrete tokens into continuous embeddings and optimizes them with a joint objective that (i) encourages restricted responses, (ii) incorporates a refusal-aware regularizer to steer activations away from refusal directions in embedding space, and (iii) applies a coherence term to maintain semantic plausibility and non-redundancy. After optimization, a critic-guided decoding procedure maps embeddings back to tokens by balancing embedding affinity with language-model likelihood. This integration yields suffixes that are both effective in bypassing defenses and natural in form. Experiments on multiple open-source LLMs show that RAID achieves higher attack success rates with fewer queries and lower computational cost than recent white-box and black-box baselines. These findings highlight the importance of embedding-space regularization for understanding and mitigating LLM jailbreak vulnerabilities.
>
---
#### [replaced 036] From Words to Proverbs: Evaluating LLMs Linguistic and Cultural Competence in Saudi Dialects with Absher
- **分类: cs.CL; cs.AI**

- **简介: 该论文属NLP评估任务，旨在解决LLMs对沙特阿拉伯方言及文化理解不足的问题。作者构建了含18,000道多选题的基准数据集Absher，覆盖语义、文化解读等六类能力，并评测多个主流LLM，揭示其在文化推理与语境理解上的显著短板。**

- **链接: [https://arxiv.org/pdf/2507.10216v2](https://arxiv.org/pdf/2507.10216v2)**

> **作者:** Renad Al-Monef; Hassan Alhuzali; Nora Alturayeif; Ashwag Alasmari
>
> **摘要:** As large language models (LLMs) become increasingly central to Arabic NLP applications, evaluating their understanding of regional dialects and cultural nuances is essential, particularly in linguistically diverse settings like Saudi Arabia. This paper introduces Absher, a comprehensive benchmark specifically designed to assess LLMs performance across major Saudi dialects. \texttt{Absher} comprises over 18,000 multiple-choice questions spanning six distinct categories: Meaning, True/False, Fill-in-the-Blank, Contextual Usage, Cultural Interpretation, and Location Recognition. These questions are derived from a curated dataset of dialectal words, phrases, and proverbs sourced from various regions of Saudi Arabia. We evaluate several state-of-the-art LLMs, including multilingual and Arabic-specific models. We also provide detailed insights into their capabilities and limitations. Our results reveal notable performance gaps, particularly in tasks requiring cultural inference or contextual understanding. Our findings highlight the urgent need for dialect-aware training and culturally aligned evaluation methodologies to improve LLMs performance in real-world Arabic applications.
>
---
#### [replaced 037] Ontology-Based Knowledge Graph Framework for Industrial Standard Documents via Hierarchical and Propositional Structuring
- **分类: cs.IR; cs.CL**

- **简介: 该论文属知识图谱构建与RAG任务，旨在解决工业标准文档因结构复杂、规则交织导致的KG建模难问题。提出分层语义结构+原子命题分解+LLM三元组抽取方法，构建本体增强KG，并验证其在多类QA及毒条款检测中的优越性。**

- **链接: [https://arxiv.org/pdf/2512.08398v2](https://arxiv.org/pdf/2512.08398v2)**

> **作者:** Jiin Park; Hyuna Jeon; Yoonseo Lee; Jisu Hong; Misuk Kim
>
> **备注:** The authors have identified significant technical errors in the paper that invalidate the current findings
>
> **摘要:** Ontology-based knowledge graph (KG) construction is a core technology that enables multidimensional understanding and advanced reasoning over domain knowledge. Industrial standards, in particular, contain extensive technical information and complex rules presented in highly structured formats that combine tables, scopes of application, constraints, exceptions, and numerical calculations, making KG construction especially challenging. In this study, we propose a method that organizes such documents into a hierarchical semantic structure, decomposes sentences and tables into atomic propositions derived from conditional and numerical rules, and integrates them into an ontology-knowledge graph through LLM-based triple extraction. Our approach captures both the hierarchical and logical structures of documents, effectively representing domain-specific semantics that conventional methods fail to reflect. To verify its effectiveness, we constructed rule, table, and multi-hop QA datasets, as well as a toxic clause detection dataset, from industrial standards, and implemented an ontology-aware KG-RAG framework for comparative evaluation. Experimental results show that our method achieves significant performance improvements across all QA types compared to existing KG-RAG approaches. This study demonstrates that reliable and scalable knowledge representation is feasible even for industrial documents with intertwined conditions, constraints, and scopes, contributing to future domain-specific RAG development and intelligent document management.
>
---
#### [replaced 038] LoPA: Scaling dLLM Inference via Lookahead Parallel Decoding
- **分类: cs.CL**

- **简介: 该论文针对扩散大语言模型（dLLM）推理并行度低（仅1–3 tokens/forward）的问题，提出无需训练的Lookahead Parallel Decoding（LoPA）算法，通过动态选择最优Token填充顺序提升并行性，并设计多设备分支并行系统，将D2F-Dream在GSM8K上的TPF提至10.1，吞吐达1073.9 token/s。**

- **链接: [https://arxiv.org/pdf/2512.16229v2](https://arxiv.org/pdf/2512.16229v2)**

> **作者:** Chenkai Xu; Yijie Jin; Jiajun Li; Yi Tu; Guoping Long; Dandan Tu; Mingcong Song; Hongjie Si; Tianqi Hou; Junchi Yan; Zhijie Deng
>
> **摘要:** Diffusion Large Language Models (dLLMs) have demonstrated significant potential for high-speed inference. However, current confidence-driven decoding strategies are constrained by limited parallelism, typically achieving only 1--3 tokens per forward pass (TPF). In this work, we identify that the degree of parallelism during dLLM inference is highly sensitive to the Token Filling Order (TFO). Then, we introduce Lookahead PArallel Decoding LoPA, a training-free, plug-and-play algorithm, to identify a superior TFO and hence accelerate inference. LoPA concurrently explores distinct candidate TFOs via parallel branches, and selects the one with the highest potential for future parallelism based on branch confidence. We apply LoPA to the state-of-the-art D2F model and observe a substantial enhancement in decoding efficiency. Notably, LoPA increases the TPF of D2F-Dream to 10.1 on the GSM8K while maintaining performance superior to the Dream baseline. Furthermore, to facilitate this unprecedented degree of parallelism, we develop a specialized multi-device inference system featuring Branch Parallelism (BP), which achieves a single-sample throughput of 1073.9 tokens per second under multi-GPU deployment. The code is available at https://github.com/zhijie-group/LoPA.
>
---
#### [replaced 039] DSO: Direct Steering Optimization for Bias Mitigation
- **分类: cs.LG; cs.CL; cs.CY**

- **简介: 该论文属AI公平性任务，旨在解决生成模型（VLMs/LLMs）在推理时因感知人口统计属性导致的偏见问题。提出直接引导优化（DSO）方法，用强化学习学习线性激活引导变换，在推理时可控地权衡公平性与性能。**

- **链接: [https://arxiv.org/pdf/2512.15926v2](https://arxiv.org/pdf/2512.15926v2)**

> **作者:** Lucas Monteiro Paes; Nivedha Sivakumar; Yinong Oliver Wang; Masha Fedzechkina Donaldson; Barry-John Theobald; Luca Zappella; Nicholas Apostoloff
>
> **摘要:** Generative models are often deployed to make decisions on behalf of users, such as vision-language models (VLMs) identifying which person in a room is a doctor to help visually impaired individuals. Yet, VLM decisions are influenced by the perceived demographic attributes of people in the input, which can lead to biased outcomes like failing to identify women as doctors. Moreover, when reducing bias leads to performance loss, users may have varying needs for balancing bias mitigation with overall model capabilities, highlighting the demand for methods that enable controllable bias reduction during inference. Activation steering is a popular approach for inference-time controllability that has shown potential in inducing safer behavior in large language models (LLMs). However, we observe that current steering methods struggle to correct biases, where equiprobable outcomes across demographic groups are required. To address this, we propose Direct Steering Optimization (DSO) which uses reinforcement learning to find linear transformations for steering activations, tailored to mitigate bias while maintaining control over model performance. We demonstrate that DSO achieves state-of-the-art trade-off between fairness and capabilities on both VLMs and LLMs, while offering practitioners inference-time control over the trade-off. Overall, our work highlights the benefit of designing steering strategies that are directly optimized to control model behavior, providing more effective bias intervention than methods that rely on pre-defined heuristics for controllability.
>
---
#### [replaced 040] VietLyrics: A Large-Scale Dataset and Models for Vietnamese Automatic Lyrics Transcription
- **分类: cs.AI; cs.CL**

- **简介: 该论文面向越南语自动歌词转录（ALT）任务，解决因声调复杂、方言多样且缺乏专用数据集导致的研究滞后问题。作者构建首个大规模越南语ALT数据集VietLyrics（647小时，行级对齐），微调Whisper模型并开源数据与模型。**

- **链接: [https://arxiv.org/pdf/2510.22295v2](https://arxiv.org/pdf/2510.22295v2)**

> **作者:** Quoc Anh Nguyen; Bernard Cheng; Kelvin Soh
>
> **摘要:** Automatic Lyrics Transcription (ALT) for Vietnamese music presents unique challenges due to its tonal complexity and dialectal variations, but remains largely unexplored due to the lack of a dedicated dataset. Therefore, we curated the first large-scale Vietnamese ALT dataset (VietLyrics), comprising 647 hours of songs with line-level aligned lyrics and metadata to address these issues. Our evaluation of current ASRbased approaches reveal significant limitations, including frequent transcription errors and hallucinations in non-vocal segments. To improve performance, we fine-tuned Whisper models on the VietLyrics dataset, achieving superior results compared to existing multilingual ALT systems, including LyricWhiz. We publicly release VietLyrics and our models, aiming to advance Vietnamese music computing research while demonstrating the potential of this approach for ALT in low-resource language and music.
>
---
#### [replaced 041] AdaLRS: Loss-Guided Adaptive Learning Rate Search for Efficient Foundation Model Pretraining
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出AdaLRS算法，解决基础模型预训练中学习率手动调优低效、泛化性差的问题。它基于损失下降速度在线自适应搜索最优学习率，无需额外代理模型调参，具理论收敛保证和跨场景强泛化性。**

- **链接: [https://arxiv.org/pdf/2506.13274v3](https://arxiv.org/pdf/2506.13274v3)**

> **作者:** Hongyuan Dong; Dingkang Yang; Xiao Liang; Chao Feng; Jiao Ran
>
> **备注:** NeurIPS 2025 Main Conference
>
> **摘要:** Learning rate is widely regarded as crucial for effective foundation model pretraining. Recent research explores and demonstrates the transferability of learning rate configurations across varying model and dataset sizes, etc. Nevertheless, these approaches are constrained to specific training scenarios and typically necessitate extensive hyperparameter tuning on proxy models. In this work, we propose \textbf{AdaLRS}, a plug-in-and-play adaptive learning rate search algorithm that conducts online optimal learning rate search via optimizing loss descent velocities. We provide theoretical and experimental analyzes to show that foundation model pretraining loss and its descent velocity are both convex and share the same optimal learning rate. Relying solely on training loss dynamics, AdaLRS involves few extra computations to guide the search process, and its convergence is guaranteed via theoretical analysis. Experiments on both LLM and VLM pretraining show that AdaLRS adjusts suboptimal learning rates to the neighborhood of optimum with marked efficiency and effectiveness, with model performance improved accordingly. We also show the robust generalizability of AdaLRS across varying training scenarios, such as different model sizes, training paradigms, base learning rate scheduler choices, and hyperparameter settings.
>
---
#### [replaced 042] Structured Language Generation Model: Loss Calibration and Formatted Decoding for Robust Structure Prediction and Knowledge Retrieval
- **分类: cs.CL**

- **简介: 该论文针对大语言模型在结构化任务（如NER、关系抽取）上性能弱于编码器模型的问题，提出SLGM框架：通过结构化输入格式、损失校准和格式感知解码，将结构预测转化为分类问题，在不增参数下提升结构预测鲁棒性与知识检索能力。**

- **链接: [https://arxiv.org/pdf/2402.08971v3](https://arxiv.org/pdf/2402.08971v3)**

> **作者:** Minho Lee; Junghyun Min; Yerang Kim; Woochul Lee; Yeonsoo Lee
>
> **备注:** 20 pages, 4 figures. FrontierIR at AAAI 2026
>
> **摘要:** Modern generative pre-trained language models excel at open-ended text generation, yet continue to underperform on structure-related tasks such as NER, relation extraction, and semantic role labeling, especially when compared to encoder-only models of similar sizes. While this gap has been attributed to limited structure knowledge, we hypothesize this is also due to the missing connection between the model's internal representations of linguistic structure and the output space used during supervised fine-tuning. We propose the Structured Language Generation Model (SLGM), a model- and task-agnostic framework that reformulates structured prediction as a classification problem through three components: (1) reinforced input formatting with structural cues, (2) loss design, and (3) format-aware decoding that constrains generation to task-valid outputs. Across 5 tasks and 13 datasets, SLGM substantially improves structure prediction without relying on dataset-specific engineering or additional model parameters, strengthening alignment between the model's internal structure representation and output. It outperforms baseline fine-tuning on models of the same size, achieves comparable performance to much larger models when used with <1B parameter models, and acts as a zero-weight adapter that reproduces the benefits of dataset-specific fine-tuning in low-resource settings.
>
---
#### [replaced 043] VLegal-Bench: Cognitively Grounded Benchmark for Vietnamese Legal Reasoning of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文构建了首个面向越南法律的、认知分层的评测基准VLegal-Bench，旨在解决现有LLM在越南复杂、动态法律体系下推理能力评估缺失的问题。工作包括基于Bloom分类法设计多层级任务，由法律专家标注验证10450个样本，覆盖问答、检索增强生成、多步推理等真实场景。**

- **链接: [https://arxiv.org/pdf/2512.14554v3](https://arxiv.org/pdf/2512.14554v3)**

> **作者:** Nguyen Tien Dong; Minh-Anh Nguyen; Thanh Dat Hoang; Nguyen Tuan Ngoc; Dao Xuan Quang Minh; Phan Phi Hai; Nguyen Thi Ngoc Anh; Dang Van Tu; Binh Vu
>
> **摘要:** The rapid advancement of large language models (LLMs) has enabled new possibilities for applying artificial intelligence within the legal domain. Nonetheless, the complexity, hierarchical organization, and frequent revisions of Vietnamese legislation pose considerable challenges for evaluating how well these models interpret and utilize legal knowledge. To address this gap, Vietnamese Legal Benchmark (VLegal-Bench) is introduced, the first comprehensive benchmark designed to systematically assess LLMs on Vietnamese legal tasks. Informed by Bloom's cognitive taxonomy, VLegal-Bench encompasses multiple levels of legal understanding through tasks designed to reflect practical usage scenarios. The benchmark comprises 10,450 samples generated through a rigorous annotation pipeline, where legal experts label and cross-validate each instance using our annotation system to ensure every sample is grounded in authoritative legal documents and mirrors real-world legal assistant workflows, including general legal questions and answers, retrieval-augmented generation, multi-step reasoning, and scenario-based problem solving tailored to Vietnamese law. By providing a standardized, transparent, and cognitively informed evaluation framework, VLegal-Bench establishes a solid foundation for assessing LLM performance in Vietnamese legal contexts and supports the development of more reliable, interpretable, and ethically aligned AI-assisted legal systems.
>
---
#### [replaced 044] Evaluating Large Language Models for Anxiety, Depression, and Stress Detection: Insights into Prompting Strategies and Synthetic Data
- **分类: cs.CL**

- **简介: 该论文属于文本分类任务，旨在解决心理疾病（焦虑、抑郁、压力）的自动检测难题。研究对比了LLMs（如Llama、GPT）与经典模型（BERT、XLNet、Distil-RoBERTa），在DAIC-WOZ数据集上微调并引入合成数据缓解类别不平衡，验证了Transformer模型与数据增强的有效性。**

- **链接: [https://arxiv.org/pdf/2511.07044v2](https://arxiv.org/pdf/2511.07044v2)**

> **作者:** Mihael Arcan; David-Paul Niland
>
> **摘要:** Mental health disorders affect over one-fifth of adults globally, yet detecting such conditions from text remains challenging due to the subtle and varied nature of symptom expression. This study evaluates multiple approaches for mental health detection, comparing Large Language Models (LLMs) such as Llama and GPT with classical machine learning and transformer-based architectures including BERT, XLNet, and Distil-RoBERTa. Using the DAIC-WOZ dataset of clinical interviews, we fine-tuned models for anxiety, depression, and stress classification and applied synthetic data generation to mitigate class imbalance. Results show that Distil-RoBERTa achieved the highest F1 score (0.883) for GAD-2, while XLNet outperformed others on PHQ tasks (F1 up to 0.891). For stress detection, a zero-shot synthetic approach (SD+Zero-Shot-Basic) reached an F1 of 0.884 and ROC AUC of 0.886. Findings demonstrate the effectiveness of transformer-based models and highlight the value of synthetic data in improving recall and generalization. However, careful calibration is required to prevent precision loss. Overall, this work emphasizes the potential of combining advanced language models and data augmentation to enhance automated mental health assessment from text.
>
---
#### [replaced 045] BiCA: Effective Biomedical Dense Retrieval with Citation-Aware Hard Negatives
- **分类: cs.IR; cs.CL**

- **简介: 该论文面向 biomedical dense retrieval 任务，解决硬负样本难挖掘问题。提出 BiCA 方法，利用 PubMed 文献的引用关系构造高质量硬负样本，微调 GTE 小模型，在 BEIR 和 LoTTE 上显著提升零样本检索性能，实现高效领域适配。**

- **链接: [https://arxiv.org/pdf/2511.08029v2](https://arxiv.org/pdf/2511.08029v2)**

> **作者:** Aarush Sinha; Pavan Kumar S; Roshan Balaji; Nirav Pravinbhai Bhatt
>
> **备注:** Accepted for oral presentation at AAAI 2026
>
> **摘要:** Hard negatives are essential for training effective retrieval models. Hard-negative mining typically relies on ranking documents using cross-encoders or static embedding models based on similarity metrics such as cosine distance. Hard negative mining becomes challenging for biomedical and scientific domains due to the difficulty in distinguishing between source and hard negative documents. However, referenced documents naturally share contextual relevance with the source document but are not duplicates, making them well-suited as hard negatives. In this work, we propose BiCA: Biomedical Dense Retrieval with Citation-Aware Hard Negatives, an approach for hard-negative mining by utilizing citation links in 20,000 PubMed articles for improving a domain-specific small dense retriever. We fine-tune the GTE_small and GTE_Base models using these citation-informed negatives and observe consistent improvements in zero-shot dense retrieval using nDCG@10 for both in-domain and out-of-domain tasks on BEIR and outperform baselines on long-tailed topics in LoTTE using Success@5. Our findings highlight the potential of leveraging document link structure to generate highly informative negatives, enabling state-of-the-art performance with minimal fine-tuning and demonstrating a path towards highly data-efficient domain adaptation.
>
---
#### [replaced 046] Over-representation of phonological features in basic vocabulary doesn't replicate when controlling for spatial and phylogenetic effects
- **分类: cs.CL**

- **简介: 该论文属语言学实证研究任务，旨在检验基本词汇中音系特征过度表征（声符现象）的普遍性是否稳健。它复现前人研究但扩大样本至2864种语言，并新增空间与谱系控制变量，发现多数原结论不稳健，仅少数模式稳定。**

- **链接: [https://arxiv.org/pdf/2512.07543v2](https://arxiv.org/pdf/2512.07543v2)**

> **作者:** Frederic Blum
>
> **备注:** Accepted with minor revisions at *Linguistic Typology*, expected to be fully published in 2026
>
> **摘要:** The statistical over-representation of phonological features in the basic vocabulary of languages is often interpreted as reflecting potentially universal sound symbolic patterns. However, most of those results have not been tested explicitly for reproducibility and might be prone to biases in the study samples or models. Many studies on the topic do not adequately control for genealogical and areal dependencies between sampled languages, casting doubts on the robustness of the results. In this study, we test the robustness of a recent study on sound symbolism of basic vocabulary concepts which analyzed 245 languages.The new sample includes data on 2864 languages from Lexibank. We modify the original model by adding statistical controls for spatial and phylogenetic dependencies between languages. The new results show that most of the previously observed patterns are not robust, and in fact many patterns disappear completely when adding the genealogical and areal controls. A small number of patterns, however, emerges as highly stable even with the new sample. Through the new analysis, we are able to assess the distribution of sound symbolism on a larger scale than previously. The study further highlights the need for testing all universal claims on language for robustness on various levels.
>
---
#### [replaced 047] Navigating the Reality Gap: Privacy-Preserving Adaptation of ASR for Challenging Low-Resource Domains
- **分类: cs.CL**

- **简介: 该论文属ASR领域，旨在解决低资源临床场景中因现实音频噪声与隐私限制导致的“现实鸿沟”问题。提出零数据外泄的LoRA持续适应框架，通过多域经验回放提升性能、缓解灾难性遗忘，并改进EWC以稳定训练。**

- **链接: [https://arxiv.org/pdf/2512.16401v2](https://arxiv.org/pdf/2512.16401v2)**

> **作者:** Darshil Chauhan; Adityasinh Solanki; Vansh Patel; Kanav Kapoor; Ritvik Jain; Aditya Bansal; Pratik Narang; Dhruv Kumar
>
> **摘要:** Automatic Speech Recognition (ASR) holds immense potential to assist in clinical documentation and patient report generation, particularly in resource-constrained regions. However, deployment is currently hindered by a technical deadlock: a severe "Reality Gap" between laboratory performance and noisy, real-world clinical audio, coupled with strict privacy and resource constraints. We quantify this gap, showing that a robust multilingual model (IndicWav2Vec) degrades to a 40.94% WER on rural clinical data from India, rendering it unusable. To address this, we explore a zero-data-exfiltration framework enabling localized, continual adaptation via Low-Rank Adaptation (LoRA). We conduct a rigorous investigative study of continual learning strategies, characterizing the trade-offs between data-driven and parameter-driven stability. Our results demonstrate that multi-domain Experience Replay (ER) yields the primary performance gains, achieving a 17.1% relative improvement in target WER and reducing catastrophic forgetting by 55% compared to naive adaptation. Furthermore, we observed that standard Elastic Weight Consolidation (EWC) faced numerical stability challenges when applied to LoRA in noisy environments. Our experiments show that a stabilized, linearized formulation effectively controls gradient magnitudes and enables stable convergence. Finally, we verify via a domain-specific spot check that acoustic adaptation is a fundamental prerequisite for usability which cannot be bypassed by language models alone.
>
---
#### [replaced 048] Mirage of Mastery: Memorization Tricks LLMs into Artificially Inflated Self-Knowledge
- **分类: cs.CL**

- **简介: 该论文属AI可信性研究，旨在揭示LLMs将记忆误判为推理能力导致的“虚假自知”问题。作者提出新框架，发现模型因记忆STEM题解而高估自身推理能力，致45%以上可行性评估不一致，尤以科学/医学领域显著，呼吁改进架构与训练以提升自知一致性与可解释性。**

- **链接: [https://arxiv.org/pdf/2506.18998v3](https://arxiv.org/pdf/2506.18998v3)**

> **作者:** Sahil Kale
>
> **备注:** 12 pages, 9 figures
>
> **摘要:** When artificial intelligence mistakes memorization for intelligence, it creates a dangerous mirage of reasoning. Existing studies treat memorization and self-knowledge deficits in LLMs as separate issues and do not recognize an intertwining link that degrades the trustworthiness of LLM responses. In our study, we utilize a novel framework to ascertain if LLMs genuinely learn reasoning patterns from training data or merely memorize them to assume competence across problems of similar complexity focused on STEM domains. Our analysis shows a noteworthy problem in generalization: LLMs draw confidence from memorized solutions to infer a higher self-knowledge about their reasoning ability, which manifests as an over 45% inconsistency in feasibility assessments when faced with self-validated, logically coherent task perturbations. This effect is most pronounced in science and medicine domains, which tend to have maximal standardized jargon and problems, further confirming our approach. Significant wavering within the self-knowledge of LLMs also shows flaws in current architectures and training patterns, highlighting the need for techniques that ensure a balanced, consistent stance on models' perceptions of their own knowledge for maximum AI explainability and trustworthiness. Our code and results are available publicly at https://github.com/Sahil-R-Kale/mirage_of_mastery
>
---
#### [replaced 049] LiveOIBench: Can Large Language Models Outperform Human Contestants in Informatics Olympiads?
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出LiveOIBench——一个面向信息学奥赛的新型评测基准，旨在解决现有编程评测基准难题不足、测试覆盖少、依赖API、难复现等问题。作者构建403道高质量奥赛题，集成人类选手成绩，评测34个LLM，发现最强模型仍落后顶尖人类。**

- **链接: [https://arxiv.org/pdf/2510.09595v2](https://arxiv.org/pdf/2510.09595v2)**

> **作者:** Kaijian Zou; Aaron Xiong; Yunxiang Zhang; Frederick Zhang; Yueqi Ren; Jirong Yang; Ayoung Lee; Shitanshu Bhushan; Lu Wang
>
> **摘要:** Competitive programming problems increasingly serve as valuable benchmarks to evaluate the coding capabilities of large language models (LLMs) due to their complexity and ease of verification. Yet, current coding benchmarks face limitations such as lack of exceptionally challenging problems, insufficient test case coverage, reliance on online platform APIs that limit accessibility. To address these issues, we introduce LiveOIBench, a comprehensive benchmark featuring 403 expert-curated Olympiad-level competitive programming problems, each with an average of 60 expert-designed test cases. The problems are sourced directly from 72 official contests of 14 Informatics Olympiads in different regions conducted between 2023 and 2025. LiveOIBench distinguishes itself through four key features: (1) meticulously curated high-quality tasks with detailed subtask rubrics and extensive private test cases; (2) direct integration of elite contestant performance data to enable informative comparison against top-performing humans; (3) planned continuous, contamination-free updates from newly released Olympiad problems; and (4) a self-contained evaluation system facilitating offline and easy-to-reproduce assessments. Benchmarking 34 popular general-purpose and reasoning LLMs, we find that GPT-5 achieves a notable 81.76th percentile, a strong result that nonetheless falls short of top human contestants, who usually place above 90th. In contrast, among open-weight reasoning models, GPT-OSS-120B achieves only a 60th percentile, underscoring significant capability disparities from frontier closed models. Detailed analyses indicate that robust reasoning models prioritize precise problem analysis over excessive exploration, suggesting future models should emphasize structured analysis and minimize unnecessary exploration. All data, code, and leaderboard results are publicly available on our website.
>
---
#### [replaced 050] Far from the Shallow: Brain-Predictive Reasoning Embedding through Residual Disentanglement
- **分类: cs.CL; q-bio.NC**

- **简介: 该论文属脑-语言建模任务，旨在解决LLM表征纠缠导致难以分离深层推理神经机制的问题。作者提出残差解耦方法，从LLM中提取正交的词法、句法、语义和推理嵌入，并用其成功建模ECoG数据，揭示推理特有的晚时程、跨模态神经响应。**

- **链接: [https://arxiv.org/pdf/2510.22860v2](https://arxiv.org/pdf/2510.22860v2)**

> **作者:** Linyang He; Tianjun Zhong; Richard Antonello; Gavin Mischler; Micah Goldblum; Nima Mesgarani
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Understanding how the human brain progresses from processing simple linguistic inputs to performing high-level reasoning is a fundamental challenge in neuroscience. While modern large language models (LLMs) are increasingly used to model neural responses to language, their internal representations are highly "entangled," mixing information about lexicon, syntax, meaning, and reasoning. This entanglement biases conventional brain encoding analyses toward linguistically shallow features (e.g., lexicon and syntax), making it difficult to isolate the neural substrates of cognitively deeper processes. Here, we introduce a residual disentanglement method that computationally isolates these components. By first probing an LM to identify feature-specific layers, our method iteratively regresses out lower-level representations to produce four nearly orthogonal embeddings for lexicon, syntax, meaning, and, critically, reasoning. We used these disentangled embeddings to model intracranial (ECoG) brain recordings from neurosurgical patients listening to natural speech. We show that: 1) This isolated reasoning embedding exhibits unique predictive power, accounting for variance in neural activity not explained by other linguistic features and even extending to the recruitment of visual regions beyond classical language areas. 2) The neural signature for reasoning is temporally distinct, peaking later (~350-400ms) than signals related to lexicon, syntax, and meaning, consistent with its position atop a processing hierarchy. 3) Standard, non-disentangled LLM embeddings can be misleading, as their predictive success is primarily attributable to linguistically shallow features, masking the more subtle contributions of deeper cognitive processing.
>
---
#### [replaced 051] LLM-as-a-Prophet: Understanding Predictive Intelligence with Prophet Arena
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究“LLM-as-a-Prophet”范式，即用大语言模型进行现实事件预测。为评估其预测智能，作者构建了动态基准Prophet Arena，系统评测多模型在事件回忆、数据理解、信息聚合等环节的表现，揭示当前瓶颈并验证其初步预测能力。**

- **链接: [https://arxiv.org/pdf/2510.17638v2](https://arxiv.org/pdf/2510.17638v2)**

> **作者:** Qingchuan Yang; Simon Mahns; Sida Li; Anri Gu; Jibang Wu; Haifeng Xu
>
> **备注:** https://www.prophetarena.co/
>
> **摘要:** Forecasting is not only a fundamental intellectual pursuit but also is of significant importance to societal systems such as finance and economics. With the rapid advances of large language models (LLMs) trained on Internet-scale data, it raises the promise of employing LLMs to forecast real-world future events, an emerging paradigm we call "LLM-as-a-Prophet". This paper systematically investigates such predictive intelligence of LLMs. To this end, we build Prophet Arena, a general evaluation benchmark that continuously collects live forecasting tasks and decomposes each task into distinct pipeline stages, in order to support our controlled and large-scale experimentation. Our comprehensive evaluation reveals that many LLMs already exhibit impressive forecasting capabilities, reflected in, e.g., their small calibration errors, consistent prediction confidence and promising market returns. However, we also uncover key bottlenecks towards achieving superior predictive intelligence via LLM-as-a-Prophet, such as LLMs' inaccurate event recalls, misunderstanding of data sources and slower information aggregation compared to markets when resolution nears.
>
---
#### [replaced 052] Confucius Code Agent: Scalable Agent Scaffolding for Real-World Codebases
- **分类: cs.CL; cs.AI; cs.LG; cs.SE**

- **简介: 该论文提出Confucius Code Agent（CCA），面向大规模真实代码库的软件工程任务。旨在解决现有编码代理在可扩展性、长程推理、工具协同与可控性上的不足。工作包括构建三层体验驱动的Confucius SDK、分层记忆与持续学习机制，以及自动化配置优化的meta-agent。**

- **链接: [https://arxiv.org/pdf/2512.10398v5](https://arxiv.org/pdf/2512.10398v5)**

> **作者:** Sherman Wong; Zhenting Qi; Zhaodong Wang; Nathan Hu; Samuel Lin; Jun Ge; Erwin Gao; Wenlin Chen; Yilun Du; Minlan Yu; Ying Zhang
>
> **备注:** The latest version
>
> **摘要:** Real-world software engineering tasks require coding agents that can operate over massive repositories, sustain long-horizon sessions, and reliably coordinate complex toolchains at test time. Existing research-grade coding agents offer transparency but struggle when scaled to heavier, production-level workloads, while production-grade systems achieve strong practical performance but provide limited extensibility, interpretability, and controllability. We introduce the Confucius Code Agent (CCA), a software engineering agent that can operate at large-scale codebases. CCA is built on top of the Confucius SDK, an agent development platform structured around three complementary perspectives: Agent Experience (AX), User Experience (UX), and Developer Experience (DX). The SDK integrates a unified orchestrator with hierarchical working memory for long-context reasoning, a persistent note-taking system for cross-session continual learning, and a modular extension system for reliable tool use. In addition, we introduce a meta-agent that automates the synthesis, evaluation, and refinement of agent configurations through a build-test-improve loop, enabling rapid adaptation to new tasks, environments, and tool stacks. Instantiated with these mechanisms, CCA demonstrates strong performance on real-world software engineering tasks. On SWE-Bench-Pro, CCA reaches a Resolve@1 of 54.3%, exceeding prior research baselines and comparing favorably to commercial results, under identical repositories, model backends, and tool access.
>
---
#### [replaced 053] TermGPT: Multi-Level Contrastive Fine-Tuning for Terminology Adaptation in Legal and Financial Domain
- **分类: cs.CL**

- **简介: 该论文属术语适配任务，旨在解决LLM在法律/金融领域因嵌入各向同性导致的术语判别力弱问题。提出TermGPT框架：构建句子图生成对比样本，设计句级与词级多级对比学习，并构建首个监管文档金融术语数据集。**

- **链接: [https://arxiv.org/pdf/2511.09854v2](https://arxiv.org/pdf/2511.09854v2)**

> **作者:** Yidan Sun; Mengying Zhu; Feiyue Chen; Yangyang Wu; Xiaolei Dan; Mengyuan Yang; Xiaolin Zheng; Shenglin Ben
>
> **备注:** 13 pages, 4 figures, AAAI'26
>
> **摘要:** Large language models (LLMs) have demonstrated impressive performance in text generation tasks; however, their embedding spaces often suffer from the isotropy problem, resulting in poor discrimination of domain-specific terminology, particularly in legal and financial contexts. This weakness in terminology-level representation can severely hinder downstream tasks such as legal judgment prediction or financial risk analysis, where subtle semantic distinctions are critical. To address this problem, we propose TermGPT, a multi-level contrastive fine-tuning framework designed for terminology adaptation. We first construct a sentence graph to capture semantic and structural relations, and generate semantically consistent yet discriminative positive and negative samples based on contextual and topological cues. We then devise a multi-level contrastive learning approach at both the sentence and token levels, enhancing global contextual understanding and fine-grained terminology discrimination. To support robust evaluation, we construct the first financial terminology dataset derived from official regulatory documents. Experiments show that TermGPT outperforms existing baselines in term discrimination tasks within the finance and legal domains.
>
---
#### [replaced 054] Deliberation on Priors: Trustworthy Reasoning of Large Language Models on Knowledge Graphs
- **分类: cs.CL; cs.IR**

- **简介: 该论文属知识图谱增强的LLM推理任务，旨在解决LLM因知识不足/过时导致的幻觉问题。提出Deliberation over Priors（DP）框架，通过结构先验蒸馏与约束先验引导的推理自省，提升推理忠实性与响应可靠性。**

- **链接: [https://arxiv.org/pdf/2505.15210v3](https://arxiv.org/pdf/2505.15210v3)**

> **作者:** Jie Ma; Ning Qu; Zhitao Gao; Rui Xing; Jun Liu; Hongbin Pei; Jiang Xie; Linyun Song; Pinghui Wang; Jing Tao; Zhou Su
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Knowledge graph-based retrieval-augmented generation seeks to mitigate hallucinations in Large Language Models (LLMs) caused by insufficient or outdated knowledge. However, existing methods often fail to fully exploit the prior knowledge embedded in knowledge graphs (KGs), particularly their structural information and explicit or implicit constraints. The former can enhance the faithfulness of LLMs' reasoning, while the latter can improve the reliability of response generation. Motivated by these, we propose a trustworthy reasoning framework, termed Deliberation over Priors (DP), which sufficiently utilizes the priors contained in KGs. Specifically, DP adopts a progressive knowledge distillation strategy that integrates structural priors into LLMs through a combination of supervised fine-tuning and Kahneman-Tversky optimization, thereby improving the faithfulness of relation path generation. Furthermore, our framework employs a reasoning-introspection strategy, which guides LLMs to perform refined reasoning verification based on extracted constraint priors, ensuring the reliability of response generation. Extensive experiments on three benchmark datasets demonstrate that DP achieves new state-of-the-art performance, especially a Hit@1 improvement of 13% on the ComplexWebQuestions dataset, and generates highly trustworthy responses. We also conduct various analyses to verify its flexibility and practicality. The code is available at https://github.com/reml-group/Deliberation-on-Priors.
>
---
#### [replaced 055] AC4: Algebraic Computation Checker for Circuit Constraints in ZKPs
- **分类: cs.SE; cs.CL; cs.CR**

- **简介: 该论文属于ZKP电路验证任务，旨在解决算术电路中欠约束与过约束导致的验证漏洞问题。作者提出AC4工具，将电路约束编码为多项式方程组，利用计算机代数系统在有限域上求解并分类结果，显著提升bug检测率与检查效率。**

- **链接: [https://arxiv.org/pdf/2403.15676v5](https://arxiv.org/pdf/2403.15676v5)**

> **作者:** Qizhe Yang; Boxuan Liang; Hao Chen; Guoqiang Li
>
> **备注:** 26 pages, 5 figures
>
> **摘要:** Zero-knowledge proof (ZKP) systems have surged attention and held a fundamental role in contemporary cryptography. Zero-knowledge succinct non-interactive argument of knowledge (zk-SNARK) protocols dominate the ZKP usage, implemented through arithmetic circuit programming paradigm. However, underconstrained or overconstrained circuits may lead to bugs. The former refers to circuits that lack the necessary constraints, resulting in unexpected solutions and causing the verifier to accept a bogus witness, and the latter refers to circuits that are constrained excessively, resulting in lacking necessary solutions and causing the verifier to accept no witness. This paper introduces a novel approach for pinpointing two distinct types of bugs in ZKP circuits. The method involves encoding the arithmetic circuit constraints to polynomial equation systems and solving them over finite fields by the computer algebra system. The classification of verification results is refined, greatly enhancing the expressive power of the system. A tool, AC4, is proposed to represent the implementation of the method. Experiments show that AC4 demonstrates a increase in the solved rate, showing a 29% improvement over Picus and CIVER, and a slight improvement over halo2-analyzer, a checker for halo2 circuits. Within a solvable range, the checking time has also exhibited noticeable improvement, demonstrating a magnitude increase compared to previous efforts.
>
---
#### [replaced 056] Affective Multimodal Agents with Proactive Knowledge Grounding for Emotionally Aligned Marketing Dialogue
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出AffectMind，面向情感化营销对话任务，解决现有LLM对话系统在情绪丰富、目标导向场景中被动、情感不连贯的问题。通过融合多模态主动知识 grounding、情感-意图联合建模与强化话语优化，提升情感一致性、说服成功率和长期参与度。**

- **链接: [https://arxiv.org/pdf/2511.21728v2](https://arxiv.org/pdf/2511.21728v2)**

> **作者:** Lin Yu; Xiaofei Han; Yifei Kang; Chiung-Yi Tseng; Danyang Zhang; Ziqian Bi; Zhimo Han
>
> **摘要:** Recent advances in large language models (LLMs) have enabled fluent dialogue systems, but most remain reactive and struggle in emotionally rich, goal-oriented settings such as marketing conversations. To address this limitation, we propose AffectMind, a multimodal affective dialogue agent that performs proactive reasoning and dynamic knowledge grounding to sustain emotionally aligned and persuasive interactions. AffectMind combines three components: a Proactive Knowledge Grounding Network (PKGN) that continuously updates factual and affective context from text, vision, and prosody; an Emotion--Intent Alignment Model (EIAM) that jointly models user emotion and purchase intent to adapt persuasion strategies; and a Reinforced Discourse Loop (RDL) that optimizes emotional coherence and engagement via reinforcement signals from user responses. Experiments on two newly curated marketing dialogue datasets, MM-ConvMarket and AffectPromo, show that AffectMind outperforms strong LLM-based baselines in emotional consistency (+26\%), persuasive success rate (+19\%), and long-term user engagement (+23\%), highlighting emotion-grounded proactivity as a key capability for commercial multimodal agents.
>
---
#### [replaced 057] Survey and Experiments on Mental Disorder Detection via Social Media: From Large Language Models and RAG to Agents
- **分类: cs.CL**

- **简介: 该论文属心理健康AI检测任务，旨在解决社交平台精神障碍识别中LLM的幻觉、推理弱与可解释性差问题。工作包括：系统综述LLM、RAG与智能体方法；按技术与临床维度分类分析；跨任务评估性能；构建统一基准。**

- **链接: [https://arxiv.org/pdf/2504.02800v4](https://arxiv.org/pdf/2504.02800v4)**

> **作者:** Zhuohan Ge; Darian Li; Yubo Wang; Nicole Hu; Xinyi Zhu; Haoyang Li; Xin Zhang; Mingtao Zhang; Shihao Qi; Yuming Xu; Han Shi; Chen Jason Zhang; Qing Li
>
> **备注:** 20 pages, 10 figures. This is an extension of ICDEW 2025
>
> **摘要:** Mental disorders represent a critical global health challenge, and social media is increasingly viewed as a vital resource for real-time digital phenotyping and intervention. To leverage this data, large language models (LLMs) have been introduced, offering stronger semantic understanding and reasoning than traditional deep learning, thereby enhancing the explainability of detection results. Despite the growing prominence of LLMs in this field, there is a scarcity of scholarly works that systematically synthesize how advanced enhancement techniques, specifically Retrieval-Augmented Generation (RAG) and Agentic systems, can be utilized to address these reliability and reasoning limitations. Here, we systematically survey the evolving landscape of LLM-based methods for social media mental disorder analysis, spanning standard pre-trained language models, RAG to mitigate hallucinations and contextual gaps, and agentic systems for autonomous reasoning and multi-step intervention. We organize existing work by technical paradigm and clinical target, extending beyond common internalizing disorders to include psychotic disorders and externalizing behaviors. Additionally, the paper comprehensively evaluates the performance of LLMs, including the impact of RAG, across various tasks. This work establishes a unified benchmark for the field, paving the way for the development of trustworthy, autonomous AI systems that can deliver precise and explainable mental health support.
>
---
#### [replaced 058] Syzygy of Thoughts: Improving LLM CoT with the Minimal Free Resolution
- **分类: cs.CL**

- **简介: 该论文属AI推理任务，旨在解决复杂问题下单一思维链（CoT）能力不足的问题。提出“思想协和”（SoT）框架，受交换代数中极小自由分解（MFR）启发，引入多条关联推理路径，结构化分解问题，提升准确性与推理效率。**

- **链接: [https://arxiv.org/pdf/2504.09566v3](https://arxiv.org/pdf/2504.09566v3)**

> **作者:** Chenghao Li; Chaoning Zhang; Yi Lu; Jiaquan Zhang; Qigan Sun; Xudong Wang; Jiwei Wei; Guoqing Wang; Yang Yang; Heng Tao Shen
>
> **摘要:** Chain-of-Thought (CoT) prompting enhances the reasoning of large language models (LLMs) by decomposing problems into sequential steps, mimicking human logic and reducing errors. However, complex tasks with vast solution spaces and vague constraints often exceed the capacity of a single reasoning chain. Inspired by Minimal Free Resolution (MFR) in commutative algebra and algebraic geometry, we propose Syzygy of Thoughts (SoT)-a novel framework that extends CoT by introducing auxiliary, interrelated reasoning paths. SoT captures deeper logical dependencies, enabling more robust and structured problem-solving. MFR decomposes a module into a sequence of free modules with minimal rank, providing a structured analytical approach to complex systems. This method introduces the concepts of "Module", "Betti numbers","Freeness", "Mapping", "Exactness" and "Minimality", enabling the systematic decomposition of the original complex problem into logically complete minimal subproblems while preserving key problem features and reducing reasoning length. We tested SoT across diverse datasets (e.g., GSM8K, MATH) and models (e.g., GPT-4o-mini, Qwen2.5), achieving inference accuracy that matches or surpasses mainstream CoTs standards. Additionally, by aligning the sampling process with algebraic constraints, our approach enhances the scalability of inference time in LLMs, ensuring both transparent reasoning and high performance. Our code will be publicly available at https://github.com/dlMARiA/Syzygy-of-thoughts.
>
---
#### [replaced 059] Multimodal Cultural Safety: Evaluation Framework and Alignment Strategies
- **分类: cs.CL**

- **简介: 该论文面向多模态大模型的文化安全评估与对齐任务，旨在解决LVLMs在跨文化场景中易引发象征性伤害的问题。作者构建了多语言、多国家的CROSS基准及CROSS-Eval评估框架，并提出监督微调与偏好调优两种对齐策略，显著提升模型文化意识与合规性。**

- **链接: [https://arxiv.org/pdf/2505.14972v2](https://arxiv.org/pdf/2505.14972v2)**

> **作者:** Haoyi Qiu; Kung-Hsiang Huang; Ruichen Zheng; Jiao Sun; Nanyun Peng
>
> **摘要:** Large vision-language models (LVLMs) are increasingly deployed in globally distributed applications, such as tourism assistants, yet their ability to produce culturally appropriate responses remains underexplored. Existing multimodal safety benchmarks primarily focus on physical safety and overlook violations rooted in cultural norms, which can result in symbolic harm. To address this gap, we introduce CROSS, a benchmark designed to assess the cultural safety reasoning capabilities of LVLMs. CROSS includes 1,284 multilingual visually grounded queries from 16 countries, three everyday domains, and 14 languages, where cultural norm violations emerge only when images are interpreted in context. We propose CROSS-Eval, an intercultural theory-based framework that measures four key dimensions: cultural awareness, norm education, compliance, and helpfulness. Using this framework, we evaluate 21 leading LVLMs, including mixture-of-experts models and reasoning models. Results reveal significant cultural safety gaps: the best-performing model achieves only 61.79% in awareness and 37.73% in compliance. While some open-source models reach GPT-4o-level performance, they still fall notably short of proprietary models. Our results further show that increasing reasoning capacity improves cultural alignment but does not fully resolve the issue. To improve model performance, we develop two enhancement strategies: supervised fine-tuning with culturally grounded, open-ended data and preference tuning with contrastive response pairs that highlight safe versus unsafe behaviors. These methods substantially improve GPT-4o's cultural awareness (+60.14%) and compliance (+55.2%), while preserving general multimodal capabilities with minimal performance reduction on general multimodal understanding benchmarks.
>
---
#### [replaced 060] Human-Inspired Learning for Large Language Models via Obvious Record and Maximum-Entropy Method Discovery
- **分类: cs.CL; cs.AI**

- **简介: 该论文属AI学习方法研究，旨在解决LLM在罕见场景中泛化弱、缺乏显式方法记忆的问题。提出“明显记录”（符号化存因果对）和“最大熵方法发现”（保留语义差异大的策略）双机制框架，提升LLM对低资源、未见问题的覆盖与方法多样性。**

- **链接: [https://arxiv.org/pdf/2512.12608v2](https://arxiv.org/pdf/2512.12608v2)**

> **作者:** Hong Su
>
> **摘要:** Large Language Models (LLMs) excel at extracting common patterns from large-scale corpora, yet they struggle with rare, low-resource, or previously unseen scenarios-such as niche hardware deployment issues or irregular IoT device behaviors-because such cases are sparsely represented in training data. Moreover, LLMs rely primarily on implicit parametric memory, which limits their ability to explicitly acquire, recall, and refine methods, causing them to behave predominantly as intuition-driven predictors rather than deliberate, method-oriented learners. Inspired by how humans learn from rare experiences, this paper proposes a human-inspired learning framework that integrates two complementary mechanisms. The first, Obvious Record, explicitly stores cause--result (or question--solution) relationships as symbolic memory, enabling persistent learning even from single or infrequent encounters. The second, Maximum-Entropy Method Discovery, prioritizes and preserves methods with high semantic dissimilarity, allowing the system to capture diverse and underrepresented strategies that are typically overlooked by next-token prediction. Verification on a benchmark of 60 semantically diverse question--solution pairs demonstrates that the proposed entropy-guided approach achieves stronger coverage of unseen questions and significantly greater internal diversity than a random baseline, confirming its effectiveness in discovering more generalizable and human-inspired methods.
>
---
#### [replaced 061] SPELL: Self-Play Reinforcement Learning for evolving Long-Context Language Models
- **分类: cs.CL**

- **简介: 该论文提出SPELL框架，属长文本推理任务，旨在解决长上下文LLM缺乏高质量标注与可验证奖励信号的问题。通过问答生成、响应求解、语义验证三角色自博弈强化学习，结合自动课程学习与自适应奖励，实现无标注的持续优化。**

- **链接: [https://arxiv.org/pdf/2509.23863v2](https://arxiv.org/pdf/2509.23863v2)**

> **作者:** Ziyi Yang; Weizhou Shen; Chenliang Li; Ruijun Chen; Fanqi Wan; Ming Yan; Xiaojun Quan; Fei Huang
>
> **备注:** Preprint under review
>
> **摘要:** Progress in long-context reasoning for large language models (LLMs) has lagged behind other recent advances. This gap arises not only from the intrinsic difficulty of processing long texts, but also from the scarcity of reliable human annotations and programmatically verifiable reward signals. In this paper, we propose SPELL, a multi-role self-play reinforcement learning framework that enables scalable, label-free optimization for long-context reasoning. SPELL integrates three cyclical roles-questioner, responder, and verifier-within a single model to enable continual self-improvement. The questioner generates questions from raw documents paired with reference answers; the responder learns to solve these questions based on the documents; and the verifier evaluates semantic equivalence between the responder's output and the questioner's reference answer, producing reward signals to guide continual training. To stabilize training, we introduce an automated curriculum that gradually increases document length and a reward function that adapts question difficulty to the model's evolving capabilities. Extensive experiments on six long-context benchmarks show that SPELL consistently improves performance across diverse LLMs and outperforms equally sized models fine-tuned on large-scale annotated data. Notably, SPELL achieves an average 7.6-point gain in pass@8 on the strong reasoning model Qwen3-30B-A3B-Thinking, raising its performance ceiling and showing promise for scaling to even more capable models.
>
---
#### [replaced 062] CodeNER: Code Prompting for Named Entity Recognition
- **分类: cs.CL; cs.AI**

- **简介: 该论文属命名实体识别（NER）任务，旨在解决LLM仅依赖文本提示难以精准理解BIO标签规范的问题。作者提出CodeNER方法，将代码嵌入提示中显式编码BIO schema，利用LLM对编程语言长程结构的理解能力提升NER性能，并验证其在多语种基准上的有效性。**

- **链接: [https://arxiv.org/pdf/2507.20423v3](https://arxiv.org/pdf/2507.20423v3)**

> **作者:** Sungwoo Han; Jingun Kwon; Hidetaka Kamigaito; Manabu Okumura
>
> **备注:** 18 pages, 6 figures
>
> **摘要:** Recent studies have explored various approaches for treating candidate named entity spans as both source and target sequences in named entity recognition (NER) by leveraging large language models (LLMs). Although previous approaches have successfully generated candidate named entity spans with suitable labels, they rely solely on input context information when using LLMs, particularly, ChatGPT. However, NER inherently requires capturing detailed labeling requirements with input context information. To address this issue, we propose a novel method that leverages code-based prompting to improve the capabilities of LLMs in understanding and performing NER. By embedding code within prompts, we provide detailed BIO schema instructions for labeling, thereby exploiting the ability of LLMs to comprehend long-range scopes in programming languages. Experimental results demonstrate that the proposed code-based prompting method outperforms conventional text-based prompting on ten benchmarks across English, Arabic, Finnish, Danish, and German datasets, indicating the effectiveness of explicitly structuring NER instructions. We also verify that combining the proposed code-based prompting method with the chain-of-thought prompting further improves performance.
>
---
