# 自然语言处理 cs.CL

- **最新发布 96 篇**

- **更新 42 篇**

## 最新发布

#### [new 001] Internal Safety Collapse in Frontier Large Language Models
- **分类: cs.CL; cs.AI; cs.CR**

- **简介: 该论文研究大语言模型中的内部安全崩溃问题，属于安全与风险评估任务。旨在揭示模型在特定任务下持续生成有害内容的现象，并提出TVF框架及测试集进行验证。**

- **链接: [https://arxiv.org/pdf/2603.23509](https://arxiv.org/pdf/2603.23509)**

> **作者:** Yutao Wu; Xiao Liu; Yifeng Gao; Xiang Zheng; Hanxun Huang; Yige Li; Cong Wang; Bo Li; Xingjun Ma; Yu-Gang Jiang
>
> **备注:** 15 pages of the main text, qualitative examples of jailbreaks may be harmful in nature
>
> **摘要:** This work identifies a critical failure mode in frontier large language models (LLMs), which we term Internal Safety Collapse (ISC): under certain task conditions, models enter a state in which they continuously generate harmful content while executing otherwise benign tasks. We introduce TVD (Task, Validator, Data), a framework that triggers ISC through domain tasks where generating harmful content is the only valid completion, and construct ISC-Bench containing 53 scenarios across 8 professional disciplines. Evaluated on JailbreakBench, three representative scenarios yield worst-case safety failure rates averaging 95.3% across four frontier LLMs (including GPT-5.2 and Claude Sonnet 4.5), substantially exceeding standard jailbreak attacks. Frontier models are more vulnerable than earlier LLMs: the very capabilities that enable complex task execution become liabilities when tasks intrinsically involve harmful content. This reveals a growing attack surface: almost every professional domain uses tools that process sensitive data, and each new dual-use tool automatically expands this vulnerability--even without any deliberate attack. Despite substantial alignment efforts, frontier LLMs retain inherently unsafe internal capabilities: alignment reshapes observable outputs but does not eliminate the underlying risk profile. These findings underscore the need for caution when deploying LLMs in high-stakes settings. Source code: this https URL
>
---
#### [new 002] Argument Mining as a Text-to-Text Generation Task
- **分类: cs.CL**

- **简介: 该论文将Argument Mining任务转化为文本到文本生成问题，旨在简化传统多步骤方法的复杂性，通过预训练模型直接生成论证结构，提升效率与适应性。**

- **链接: [https://arxiv.org/pdf/2603.23949](https://arxiv.org/pdf/2603.23949)**

> **作者:** Masayuki Kawarada; Tsutomu Hirao; Wataru Uchida; Masaaki Nagata
>
> **摘要:** Argument Mining(AM) aims to uncover the argumentative structures within a text. Previous methods require several subtasks, such as span identification, component classification, and relation classification. Consequently, these methods need rule-based postprocessing to derive argumentative structures from the output of each subtask. This approach adds to the complexity of the model and expands the search space of the hyperparameters. To address this difficulty, we propose a simple yet strong method based on a text-to-text generation approach using a pretrained encoder-decoder language model. Our method simultaneously generates argumentatively annotated text for spans, components, and relations, eliminating the need for task-specific postprocessing and hyperparameter tuning. Furthermore, because it is a straightforward text-to-text generation method, we can easily adapt our approach to various types of argumentative structures. Experimental results demonstrate the effectiveness of our method, as it achieves state-of-the-art performance on three different types of benchmark datasets: the Argument-annotated Essays Corpus(AAEC), AbstRCT, and the Cornell eRulemaking Corpus(CDCP)
>
---
#### [new 003] DepthCharge: A Domain-Agnostic Framework for Measuring Depth-Dependent Knowledge in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出DepthCharge框架，用于评估大语言模型在不同领域中的深度知识能力。解决现有方法无法有效衡量模型在多领域持续准确回答问题的能力的问题。通过自适应探测、事实验证和生存统计进行评估。**

- **链接: [https://arxiv.org/pdf/2603.23514](https://arxiv.org/pdf/2603.23514)**

> **作者:** Alexander Sheppert
>
> **摘要:** Large Language Models appear competent when answering general questions but often fail when pushed into domain-specific details. No existing methodology provides an out-of-the-box solution for measuring how deeply LLMs can sustain accurate responses under adaptive follow-up questioning across arbitrary domains. We present DepthCharge, a domain-agnostic framework that measures knowledge depth through three innovations: adaptive probing that generates follow-up questions based on concepts the model actually mentions, on-demand fact verification from authoritative sources, and survival statistics with constant sample sizes at every depth level. The framework can be deployed on any knowledge domain with publicly verifiable facts, without requiring pre-constructed test sets or domain-specific expertise. DepthCharge results are relative to the evaluator model used for answer checking, making the framework a tool for comparative evaluation rather than absolute accuracy certification. Empirical validation across four diverse domains (Medicine, Constitutional Law, Ancient Rome, and Quantum Computing) with five frontier models demonstrates that DepthCharge reveals depth-dependent performance variation hidden by standard benchmarks. Expected Valid Depth (EVD) ranges from 3.45 to 7.55 across model-domain combinations, and model rankings vary substantially by domain, with no single model dominating all areas. Cost-performance analysis further reveals that expensive models do not always achieve deeper knowledge, suggesting that domain-specific evaluation is more informative than aggregate benchmarks for model selection in professional applications.
>
---
#### [new 004] MARCH: Multi-Agent Reinforced Self-Check for LLM Hallucination
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的事实一致性任务，旨在解决大语言模型的幻觉问题。通过多智能体强化学习框架MARCH，提升生成内容的准确性。**

- **链接: [https://arxiv.org/pdf/2603.24579](https://arxiv.org/pdf/2603.24579)**

> **作者:** Zhuo Li; Yupeng Zhang; Pengyu Cheng; Jiajun Song; Mengyu Zhou; Hao Li; Shujie Hu; Yu Qin; Erchao Zhao; Xiaoxi Jiang; Guanjun Jiang
>
> **摘要:** Hallucination remains a critical bottleneck for large language models (LLMs), undermining their reliability in real-world applications, especially in Retrieval-Augmented Generation (RAG) systems. While existing hallucination detection methods employ LLM-as-a-judge to verify LLM outputs against retrieved evidence, they suffer from inherent confirmation bias, where the verifier inadvertently reproduces the errors of the original generation. To address this, we introduce Multi-Agent Reinforced Self-Check for Hallucination (MARCH), a framework that enforces rigorous factual alignment by leveraging deliberate information asymmetry. MARCH orchestrates a collaborative pipeline of three specialized agents: a Solver, a Proposer, and a Checker. The Solver generates an initial RAG response, which the Proposer decomposes into claim-level verifiable atomic propositions. Crucially, the Checker validates these propositions against retrieved evidence in isolation, deprived of the Solver's original output. This well-crafted information asymmetry scheme breaks the cycle of self-confirmation bias. By training this pipeline with multi-agent reinforcement learning (MARL), we enable the agents to co-evolve and optimize factual adherence. Extensive experiments across hallucination benchmarks demonstrate that MARCH substantially reduces hallucination rates. Notably, an 8B-parameter LLM equipped with MARCH achieves performance competitive with powerful closed-source models. MARCH paves a scalable path for factual self-improvement of LLMs through co-evolution. The code is at this https URL.
>
---
#### [new 005] PINGALA: Prosody-Aware Decoding for Sanskrit Poetry Generation
- **分类: cs.CL**

- **简介: 该论文属于梵语诗歌生成任务，旨在提升诗歌的语义连贯性和格律准确性。通过分段解码和音素感知的转写方案，改进了生成效果。**

- **链接: [https://arxiv.org/pdf/2603.24413](https://arxiv.org/pdf/2603.24413)**

> **作者:** Manoj Balaji Jagadeeshan; Atul Singh; Nallani Chakravartula Sahith; Amrith Krishna; Pawan Goyal
>
> **摘要:** Poetry generation in Sanskrit typically requires the verse to be semantically coherent and adhere to strict prosodic rules. In Sanskrit prosody, every line of a verse is typically a fixed length sequence of syllables adhering to prescribed binary patterns of syllable weights. We observe that instead of treating a verse as a monolithic sequence, segmenting them as grouped-lines leads to significant improvement in semantic coherence by 10\% with comparable metrical adherence. Specifically, PINGALA, our proposed decoding approach is designed to encourage every line to have well-formed words and our token selection biases the model towards it by preferring longer tokens. Writing in Sanskrit follows phonemic orthography, hence using a phonetically aware transliteration scheme, SLP1, increased the metrical alignment by 46\% with comparable semantic similarity, for a instruction fine-tuned large language models like Phi-4. We also introduce a new approach for reference-free evaluation using cross-encoders which achieved better alignment with true poetry instances.
>
---
#### [new 006] CoCR-RAG: Enhancing Retrieval-Augmented Generation in Web Q&A via Concept-oriented Context Reconstruction
- **分类: cs.CL**

- **简介: 该论文属于Web问答任务，解决多源文档融合问题。提出CoCR-RAG框架，通过概念级整合提升生成答案的准确性与一致性。**

- **链接: [https://arxiv.org/pdf/2603.23989](https://arxiv.org/pdf/2603.23989)**

> **作者:** Kaize Shi; Xueyao Sun; Qika Lin; Firoj Alam; Qing Li; Xiaohui Tao; Guandong Xu
>
> **摘要:** Retrieval-augmented generation (RAG) has shown promising results in enhancing Q&A by incorporating information from the web and other external sources. However, the supporting documents retrieved from the heterogeneous web often originate from multiple sources with diverse writing styles, varying formats, and inconsistent granularity. Fusing such multi-source documents into a coherent and knowledge-intensive context remains a significant challenge, as the presence of irrelevant and redundant information can compromise the factual consistency of the inferred answers. This paper proposes the Concept-oriented Context Reconstruction RAG (CoCR-RAG), a framework that addresses the multi-source information fusion problem in RAG through linguistically grounded concept-level integration. Specifically, we introduce a concept distillation algorithm that extracts essential concepts from Abstract Meaning Representation (AMR), a stable semantic representation that structures the meaning of texts as logical graphs. The distilled concepts from multiple retrieved documents are then fused and reconstructed into a unified, information-intensive context by Large Language Models, which supplement only the necessary sentence elements to highlight the core knowledge. Experiments on the PopQA and EntityQuestions datasets demonstrate that CoCR-RAG significantly outperforms existing context-reconstruction methods across these Web Q&A benchmarks. Furthermore, CoCR-RAG shows robustness across various backbone LLMs, establishing itself as a flexible, plug-and-play component adaptable to different RAG frameworks.
>
---
#### [new 007] CVPD at QIAS 2026: RAG-Guided LLM Reasoning for Al-Mawarith Share Computation and Heir Allocation
- **分类: cs.CL**

- **简介: 该论文聚焦伊斯兰继承计算任务，解决法律规则复杂性和多法律体系差异问题。通过RAG方法结合数据生成与验证，提升推理准确性。**

- **链接: [https://arxiv.org/pdf/2603.24012](https://arxiv.org/pdf/2603.24012)**

> **作者:** Wassim Swaileh; Mohammed-En-Nadhir Zighem; Hichem Telli; Salah Eddine Bekhouche; Abdellah Zakaria Sellam; Fadi Dornaika; Dimitrios Kotzinos
>
> **摘要:** Islamic inheritance (Ilm al-Mawarith) is a multi-stage legal reasoning task requiring the identification of eligible heirs, resolution of blocking rules (hajb), assignment of fixed and residual shares, handling of adjustments such as awl and radd, and generation of a consistent final distribution. The task is further complicated by variations across legal schools and civil-law codifications, requiring models to operate under explicit legal configurations. We present a retrieval-augmented generation (RAG) pipeline for this setting, combining rule-grounded synthetic data generation, hybrid retrieval (dense and BM25) with cross-encoder reranking, and schema-constrained output validation. A symbolic inheritance calculator is used to generate a large high-quality synthetic corpus with full intermediate reasoning traces, ensuring legal and numerical consistency. The proposed system achieves a MIR-E score of 0.935 and ranks first on the official QIAS 2026 blind-test leaderboard. Results demonstrate that retrieval-grounded, schema-aware generation significantly improves reliability in high-precision Arabic legal reasoning tasks.
>
---
#### [new 008] Stance Labels Fail When They Matter Most: The Projection Problem in Stance Detection
- **分类: cs.CL; cs.SI**

- **简介: 该论文属于立场检测任务，指出单一标签无法准确反映多维态度，提出“投影问题”并验证其影响。**

- **链接: [https://arxiv.org/pdf/2603.24231](https://arxiv.org/pdf/2603.24231)**

> **作者:** Bowen Zhang
>
> **摘要:** Stance detection is nearly always formulated as classifying text into Favor, Against, or Neutral -- a convention inherited from debate analysis and applied without modification to social media since SemEval-2016. But attitudes toward complex targets are not unitary: a person can accept climate science while opposing carbon taxes, expressing support on one dimension and opposition on another. When annotators must compress such multi-dimensional attitudes into a single label, different annotators weight different dimensions -- producing disagreement that reflects not confusion but different compression choices. We call this the \textbf{projection problem}, and show that its cost is conditional: when a text's dimensions align, any weighting yields the same label and three-way annotation works well; when dimensions conflict, label agreement collapses while agreement on individual dimensions remains intact. A pilot study on SemEval-2016 Task 6 confirms this crossover: on dimension-consistent texts, label agreement (Krippendorff's $\alpha = 0.307$) exceeds dimensional agreement ($\alpha = 0.082$); on dimension-conflicting texts, the pattern reverses -- label $\alpha$ drops to $0.085$ while dimensional $\alpha$ rises to $0.334$, with Policy reaching $0.572$. The projection problem is real -- but it activates precisely where it matters most.
>
---
#### [new 009] The Compression Paradox in LLM Inference: Provider-Dependent Energy Effects of Prompt Compression
- **分类: cs.CL**

- **简介: 该论文研究了大语言模型推理中的压缩悖论，探讨提示压缩对能耗的影响。任务是评估压缩策略在不同模型上的能效与质量平衡效果。工作包括多模型、多基准的实验分析。**

- **链接: [https://arxiv.org/pdf/2603.23528](https://arxiv.org/pdf/2603.23528)**

> **作者:** Warren Johnson
>
> **备注:** 16 pages, 5 figures, 5 tables. Includes data/code availability, ethics statement, and competing interests
>
> **摘要:** The rapid proliferation of Large Language Models has created an environmental paradox: the very technology that could help solve climate challenges is itself becoming a significant contributor to global carbon emissions. We test whether prompt compression improves inference energy efficiency in 28,421 successful API trials (28,428 planned) across three providers (OpenAI GPT-4o-mini, Anthropic Claude-3.5-Sonnet, and DeepSeek-Chat), five benchmarks (HumanEval, MBPP, GSM8K, MATH, MMLU), and four compression ratios (r in {1.0, 0.7, 0.5, 0.3}). Energy is estimated with a token-based proxy calibrated against local direct measurements, and quality is tracked with benchmark pass rates. Compression produced substantial quality loss (overall pass rate 26.0% at baseline vs. 1.5% at r=0.7) and strongly provider-dependent energy behavior. DeepSeek exhibited output expansion under compression (21 to 798 tokens at r=0.3), corresponding to energy increases up to +2,140%, while GPT-4o-mini showed mixed effects including a reduction at r=0.5. These results indicate that input-token reduction alone is not a reliable energy optimization strategy in production inference. For the evaluated settings, model selection and output-length control provided more consistent energy-quality tradeoffs than prompt compression.
>
---
#### [new 010] Retrieval Improvements Do Not Guarantee Better Answers: A Study of RAG for AI Policy QA
- **分类: cs.CL; cs.AI; cs.CY; cs.IR; cs.LG**

- **简介: 该论文研究RAG系统在AI政策问答中的应用，旨在提升答案可靠性。通过领域微调，发现检索优化不必然提升问答效果，甚至可能引发错误自信。**

- **链接: [https://arxiv.org/pdf/2603.24580](https://arxiv.org/pdf/2603.24580)**

> **作者:** Saahil Mathur; Ryan David Rittner; Vedant Ajit Thakur; Daniel Stuart Schiff; Tunazzina Islam
>
> **摘要:** Retrieval-augmented generation (RAG) systems are increasingly used to analyze complex policy documents, but achieving sufficient reliability for expert usage remains challenging in domains characterized by dense legal language and evolving, overlapping regulatory frameworks. We study the application of RAG to AI governance and policy analysis using the AI Governance and Regulatory Archive (AGORA) corpus, a curated collection of 947 AI policy documents. Our system combines a ColBERT-based retriever fine-tuned with contrastive learning and a generator aligned to human preferences using Direct Preference Optimization (DPO). We construct synthetic queries and collect pairwise preferences to adapt the system to the policy domain. Through experiments evaluating retrieval quality, answer relevance, and faithfulness, we find that domain-specific fine-tuning improves retrieval metrics but does not consistently improve end-to-end question answering performance. In some cases, stronger retrieval counterintuitively leads to more confident hallucinations when relevant documents are absent from the corpus. These results highlight a key concern for those building policy-focused RAG systems: improvements to individual components do not necessarily translate to more reliable answers. Our findings provide practical insights for designing grounded question-answering systems over dynamic regulatory corpora.
>
---
#### [new 011] Do 3D Large Language Models Really Understand 3D Spatial Relationships?
- **分类: cs.CL; cs.RO**

- **简介: 该论文属于3D视觉语言理解任务，旨在解决3D-LLMs是否真正理解空间关系的问题。研究发现现有模型依赖文本线索而非3D信息，提出新基准Real-3DQA和3D重加权训练方法提升性能。**

- **链接: [https://arxiv.org/pdf/2603.23523](https://arxiv.org/pdf/2603.23523)**

> **作者:** Xianzheng Ma; Tao Sun; Shuai Chen; Yash Bhalgat; Jindong Gu; Angel X Chang; Iro Armeni; Iro Laina; Songyou Peng; Victor Adrian Prisacariu
>
> **备注:** ICLR 2026
>
> **摘要:** Recent 3D Large-Language Models (3D-LLMs) claim to understand 3D worlds, especially spatial relationships among objects. Yet, we find that simply fine-tuning a language model on text-only question-answer pairs can perform comparably or even surpass these methods on the SQA3D benchmark without using any 3D input. This indicates that the SQA3D benchmark may not be able to detect if the model exploits textual shortcuts rather than engages in 3D-aware reasoning. To address this issue, we introduce Real-3DQA, a more rigorous evaluation benchmark that filters out easy-to-guess questions and introduces a structured taxonomy to assess various aspects of 3D reasoning. Experiments on Real-3DQA confirm that existing 3D-LLMs struggle with spatial relationships once simple cues are removed. We further propose a 3D-reweighted training objective that guides model to rely more on 3D visual clues, substantially enhancing 3D-LLMs performance in spatial reasoning tasks. Our findings underscore the need for robust benchmarks and tailored training strategies to advance genuine 3D vision-language understanding. Project page: this https URL.
>
---
#### [new 012] Generating Hierarchical JSON Representations of Scientific Sentences Using LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决科学文本信息保留问题。通过微调LLM生成层次化JSON结构，并验证其能否有效还原原文。**

- **链接: [https://arxiv.org/pdf/2603.23532](https://arxiv.org/pdf/2603.23532)**

> **作者:** Satya Sri Rajiteswari Nimmagadda; Ethan Young; Niladri Sengupta; Ananya Jana; Aniruddha Maiti
>
> **备注:** accepted to 21th International Conference on Semantic Computing (IEEE ICSC 2026)
>
> **摘要:** This paper investigates whether structured representations can preserve the meaning of scientific sentences. To test this, a lightweight LLM is fine-tuned using a novel structural loss function to generate hierarchical JSON structures from sentences collected from scientific articles. These JSONs are then used by a generative model to reconstruct the original text. Comparing the original and reconstructed sentences using semantic and lexical similarity we show that hierarchical formats are capable of retaining information of scientific texts effectively.
>
---
#### [new 013] When AI Meets Early Childhood Education: Large Language Models as Assessment Teammates in Chinese Preschools
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于教育评估任务，旨在解决传统评估方式在大规模学前教育中效率低的问题。通过构建数据集和开发AI框架，实现高效、准确的教师-儿童互动评估。**

- **链接: [https://arxiv.org/pdf/2603.24389](https://arxiv.org/pdf/2603.24389)**

> **作者:** Xingming Li; Runke Huang; Yanan Bao; Yuye Jin; Yuru Jiao; Qingyong Hu
>
> **备注:** Accepted to AIED 2026, Project page: this https URL
>
> **摘要:** High-quality teacher-child interaction (TCI) is fundamental to early childhood development, yet traditional expert-based assessment faces a critical scalability challenge. In large systems like China's-serving 36 million children across 250,000+ kindergartens-the cost and time requirements of manual observation make continuous quality monitoring infeasible, relegating assessment to infrequent episodic audits that limit timely intervention and improvement tracking. In this paper, we investigate whether AI can serve as a scalable assessment teammate by extracting structured quality indicators and validating their alignment with human expert judgments. Our contributions include: (1) TEPE-TCI-370h (Tracing Effective Preschool Education), the first large-scale dataset of naturalistic teacher-child interactions in Chinese preschools (370 hours, 105 classrooms) with standardized ECQRS-EC and SSTEW annotations; (2) We develop Interaction2Eval, a specialized LLM-based framework addressing domain-specific challenges-child speech recognition, Mandarin homophone disambiguation, and rubric-based reasoning-achieving up to 88% agreement; (3) Deployment validation across 43 classrooms demonstrating an 18x efficiency gain in the assessment workflow, highlighting its potential for shifting from annual expert audits to monthly AI-assisted monitoring with targeted human oversight. This work not only demonstrates the technical feasibility of scalable, AI-augmented quality assessment but also lays the foundation for a new paradigm in early childhood education-one where continuous, inclusive, AI-assisted evaluation becomes the engine of systemic improvement and equitable growth.
>
---
#### [new 014] PLACID: Privacy-preserving Large language models for Acronym Clinical Inference and Disambiguation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于临床术语消歧任务，解决医疗文本中缩写歧义问题，通过隐私保护的本地模型实现安全准确的缩写解析。**

- **链接: [https://arxiv.org/pdf/2603.23678](https://arxiv.org/pdf/2603.23678)**

> **作者:** Manjushree B. Aithal; Ph.D.; Alexander Kotz; James Mitchell; Ph.D
>
> **备注:** 10 pages, 2 figures, Under review AMIA Symposium
>
> **摘要:** Large Language Models (LLMs) offer transformative solutions across many domains, but healthcare integration is hindered by strict data privacy constraints. Clinical narratives are dense with ambiguous acronyms, misinterpretation these abbreviations can precipitate severe outcomes like life-threatening medication errors. While cloud-dependent LLMs excel at Acronym Disambiguation, transmitting Protected Health Information to external servers violates privacy frameworks. To bridge this gap, this study pioneers the evaluation of small-parameter models deployed entirely on-device to ensure privacy preservation. We introduce a privacy-preserving cascaded pipeline leveraging general-purpose local models to detect clinical acronyms, routing them to domain-specific biomedical models for context-relevant expansions. Results reveal that while general instruction-following models achieve high detection accuracy (~0.988), their expansion capabilities plummet (~0.655). Our cascaded approach utilizes domain-specific medical models to increase expansion accuracy to (~0.81). This novel work demonstrates that privacy-preserving, on-device (2B-10B) models deliver high-fidelity clinical acronym disambiguation support.
>
---
#### [new 015] Semantic Centroids and Hierarchical Density-Based Clustering for Cross-Document Software Coreference Resolution
- **分类: cs.CL**

- **简介: 该论文针对跨文档软件指代消解任务，解决软件提及不一致的识别与聚类问题。结合语义嵌入、知识库检索和密度聚类方法，提升指代消解效果。**

- **链接: [https://arxiv.org/pdf/2603.24246](https://arxiv.org/pdf/2603.24246)**

> **作者:** Julia Matela; Frank Krüger
>
> **摘要:** This paper describes the system submitted to the SOMD 2026 Shared Task for Cross-Document Coreference Resolution (CDCR) of software mentions. Our approach addresses the challenge of identifying and clustering inconsistent software mentions across scientific corpora. We propose a hybrid framework that combines dense semantic embeddings from a pre-trained Sentence-BERT model, Knowledge Base (KB) lookup strategy built from training-set cluster centroids using FAISS for efficient retrieval, and HDBSCAN density-based clustering for mentions that cannot be confidently assigned to existing clusters. Surface-form normalization and abbreviation resolution are applied to improve canonical name matching. The same core pipeline is applied to Subtasks 1 and 2. To address the large scale settings of Subtask 3, the pipeline was adapted by utilising a blocking strategy based on entity types and canonicalized surface forms. Our system achieved CoNLL F1 scores of 0.98, 0.98, and 0.96 on Subtasks 1, 2, and 3 respectively.
>
---
#### [new 016] Training a Large Language Model for Medical Coding Using Privacy-Preserving Synthetic Clinical Data
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗编码任务，旨在解决自动化编码准确性低的问题。通过使用隐私保护的合成数据微调大语言模型，提升编码精度。**

- **链接: [https://arxiv.org/pdf/2603.23515](https://arxiv.org/pdf/2603.23515)**

> **作者:** John Cook; Michael Wyatt; Peng Wei; Iris Chin; Santosh Gupta; Van Zyl Van Vuuren; Richie Siburian; Amanda Spicer; Kristen Viviano; Alda Cami; Raunaq Malhotra; Zhewei Yao; Jeff Rasley; Gaurav Kaushik
>
> **备注:** 20 pages, 6 figures
>
> **摘要:** Improving the accuracy and reliability of medical coding reduces clinician burnout and supports revenue cycle processes, freeing providers to focus more on patient care. However, automating the assignment of ICD-10-CM and CPT codes from clinical documentation remains a challenge due to heterogeneous records, nuanced coding guidelines, and long-tail distributions. Large language models have been proposed to help or automate specific medical coding tasks. However, foundation models are not explicitly trained for medical coding and zero-shot coding has yielded poor results. We investigate whether a modern open-weight foundation model can be adapted for an expert-level medical coding task using privacy-preserving synthetic training data derived from electronic health records. We fine-tune Llama 3-70B on pairs of clinical notes and gold codes generated from EHR-grounded templates and coding policies, then evaluate exact-code prediction for ICD-10-CM and CPT. A zero-shot baseline with the unadapted model achieved an F1 score of 0.18 for exact code match. After fine-tuning on the synthetic corpus, exact-match F1 exceeded 0.70, representing a large absolute gain across both code systems. Notably, performance remained high on complex categories that often require multi-step clinical reasoning and code composition, including Advanced Illness and Frailty classes, and the model retained its performance on medical comprehension tasks. These results indicate that synthetic, policy-aware data can efficiently teach a general-purpose large language model to support precise medical coding without exposing protected health information. The approach offers a practical path for training coding agents safely and iteratively on specific tasks that represent real-world populations.
>
---
#### [new 017] Large Language Models Unpack Complex Political Opinions through Target-Stance Extraction
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的目标立场抽取任务，旨在解决政治话语中目标与立场识别困难的问题。研究构建数据集并测试LLMs性能，证明其能有效提取复杂政治观点。**

- **链接: [https://arxiv.org/pdf/2603.23531](https://arxiv.org/pdf/2603.23531)**

> **作者:** Özgür Togay; Florian Kunneman; Javier Garcia-Bernardo; Anastasia Giachanou
>
> **摘要:** Political polarization emerges from a complex interplay of beliefs about policies, figures, and issues. However, most computational analyses reduce discourse to coarse partisan labels, overlooking how these beliefs interact. This is especially evident in online political conversations, which are often nuanced and cover a wide range of subjects, making it difficult to automatically identify the target of discussion and the opinion expressed toward them. In this study, we investigate whether Large Language Models (LLMs) can address this challenge through Target-Stance Extraction (TSE), a recent natural language processing task that combines target identification and stance detection, enabling more granular analysis of political opinions. For this, we construct a dataset of 1,084 Reddit posts from r/NeutralPolitics, covering 138 distinct political targets and evaluate a range of proprietary and open-source LLMs using zero-shot, few-shot, and context-augmented prompting strategies. Our results show that the best models perform comparably to highly trained human annotators and remain robust on challenging posts with low inter-annotator agreement. These findings demonstrate that LLMs can extract complex political opinions with minimal supervision, offering a scalable tool for computational social science and political text analysis.
>
---
#### [new 018] Revisiting Real-Time Digging-In Effects: No Evidence from NP/Z Garden-Paths
- **分类: cs.CL**

- **简介: 该论文属于语言理解任务，旨在验证实时“挖掘效应”是否真实存在。通过实验对比人类行为与语言模型预测，发现无实验证据支持该效应，仅在句末出现反向趋势。**

- **链接: [https://arxiv.org/pdf/2603.23624](https://arxiv.org/pdf/2603.23624)**

> **作者:** Amani Maina-Kilaas; Roger Levy
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Digging-in effects, where disambiguation difficulty increases with longer ambiguous regions, have been cited as evidence for self-organized sentence processing, in which structural commitments strengthen over time. In contrast, surprisal theory predicts no such effect unless lengthening genuinely shifts statistical expectations, and neural language models appear to show the opposite pattern. Whether digging-in is a robust real-time phenomenon in human sentence processing -- or an artifact of wrap-up processes or methodological confounds -- remains unclear. We report two experiments on English NP/Z garden-path sentences using Maze and self-paced reading, comparing human behavior with predictions from an ensemble of large language models. We find no evidence for real-time digging-in effects. Critically, items with sentence-final versus nonfinal disambiguation show qualitatively different patterns: positive digging-in trends appear only sentence-finally, where wrap-up effects confound interpretation. Nonfinal items -- the cleaner test of real-time processing -- show reverse trends consistent with neural model predictions.
>
---
#### [new 019] Leveraging Computerized Adaptive Testing for Cost-effective Evaluation of Large Language Models in Medical Benchmarking
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗语言模型评估任务，旨在解决传统基准测试成本高、效率低的问题。通过计算机自适应测试框架，实现高效、低成本的模型性能评估。**

- **链接: [https://arxiv.org/pdf/2603.23506](https://arxiv.org/pdf/2603.23506)**

> **作者:** Tianpeng Zheng; Zhehan Jiang; Jiayi Liu; Shicong Feng
>
> **备注:** 37 pages, 6 figures
>
> **摘要:** The rapid proliferation of large language models (LLMs) in healthcare creates an urgent need for scalable and psychometrically sound evaluation methods. Conventional static benchmarks are costly to administer repeatedly, vulnerable to data contamination, and lack calibrated measurement properties for fine-grained performance tracking. We propose and validate a computerized adaptive testing (CAT) framework grounded in item response theory (IRT) for efficient assessment of standardized medical knowledge in LLMs. The study comprises a two-phase design: a Monte Carlo simulation to identify optimal CAT configurations and an empirical evaluation of 38 LLMs using a human-calibrated medical item bank. Each model completed both the full item bank and an adaptive test that dynamically selected items based on real-time ability estimates and terminated upon reaching a predefined reliability threshold (standard error <= 0.3). Results show that CAT-derived proficiency estimates achieved a near-perfect correlation with full-bank estimates (r = 0.988) while using only 1.3 percent of the items. Evaluation time was reduced from several hours to minutes per model, with substantial reductions in token usage and computational cost, while preserving inter-model performance rankings. This work establishes a psychometric framework for rapid, low-cost benchmarking of foundational medical knowledge in LLMs. The proposed adaptive methodology is intended as a standardized pre-screening and continuous monitoring tool and is not a substitute for real-world clinical validation or safety-oriented prospective studies.
>
---
#### [new 020] Schema on the Inside: A Two-Phase Fine-Tuning Method for High-Efficiency Text-to-SQL at Scale
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦文本到SQL的转换任务，解决API调用成本高、延迟大的问题。通过两阶段微调方法，使模型内化数据库模式，提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2603.24023](https://arxiv.org/pdf/2603.24023)**

> **作者:** Chinmay Soni; Shivam Chourasia; Gaurav Kumar; Hitesh Kapoor
>
> **备注:** 8 pages, 6 figures. Published in the Proceedings of the Fortieth AAAI Conference on Artificial Intelligence (AAAI-26), 2026
>
> **摘要:** Applying large, proprietary API-based language models to text-to-SQL tasks poses a significant industry challenge: reliance on massive, schema-heavy prompts results in prohibitive per-token API costs and high latency, hindering scalable production deployment. We present a specialized, self-hosted 8B-parameter model designed for a conversational bot in CriQ, a sister app to Dream11, India's largest fantasy sports platform with over 250 million users, that answers user queries about cricket statistics. Our novel two-phase supervised fine-tuning approach enables the model to internalize the entire database schema, eliminating the need for long-context prompts. This reduces input tokens by over 99%, from a 17k-token baseline to fewer than 100, and replaces costly external API calls with efficient local inference. The resulting system achieves 98.4% execution success and 92.5% semantic accuracy, substantially outperforming a prompt-engineered baseline using Google's Gemini Flash 2.0 (95.6% execution, 89.4% semantic accuracy). These results demonstrate a practical path toward high-precision, low-latency text-to-SQL applications using domain-specialized, self-hosted language models in large-scale production environments.
>
---
#### [new 021] Dialogue to Question Generation for Evidence-based Medical Guideline Agent Development
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于医疗问答任务，旨在解决临床环境中证据基础医学实施困难的问题。通过生成相关问题，辅助医生快速应用指南，减轻认知负担。**

- **链接: [https://arxiv.org/pdf/2603.23937](https://arxiv.org/pdf/2603.23937)**

> **作者:** Zongliang Ji; Ziyang Zhang; Xincheng Tan; Matthew Thompson; Anna Goldenberg; Carl Yang; Rahul G. Krishnan; Fan Zhang
>
> **备注:** 9 pages. To appear in Proceedings of Machine Learning Research (PMLR), Machine Learning for Health (ML4H) Symposium 2025
>
> **摘要:** Evidence-based medicine (EBM) is central to high-quality care, but remains difficult to implement in fast-paced primary care settings. Physicians face short consultations, increasing patient loads, and lengthy guideline documents that are impractical to consult in real time. To address this gap, we investigate the feasibility of using large language models (LLMs) as ambient assistants that surface targeted, evidence-based questions during physician-patient encounters. Our study focuses on question generation rather than question answering, with the aim of scaffolding physician reasoning and integrating guideline-based practice into brief consultations. We implemented two prompting strategies, a zero-shot baseline and a multi-stage reasoning variant, using Gemini 2.5 as the backbone model. We evaluated on a benchmark of 80 de-identified transcripts from real clinical encounters, with six experienced physicians contributing over 90 hours of structured review. Results indicate that while general-purpose LLMs are not yet fully reliable, they can produce clinically meaningful and guideline-relevant questions, suggesting significant potential to reduce cognitive burden and make EBM more actionable at the point of care.
>
---
#### [new 022] Did You Forget What I Asked? Prospective Memory Failures in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究大语言模型在同时处理复杂任务时的格式记忆失败问题，通过认知心理学方法分析约束条件对模型表现的影响，并提出改进方案。**

- **链接: [https://arxiv.org/pdf/2603.23530](https://arxiv.org/pdf/2603.23530)**

> **作者:** Avni Mittal
>
> **摘要:** Large language models often fail to satisfy formatting instructions when they must simultaneously perform demanding tasks. We study this behaviour through a prospective memory inspired lens from cognitive psychology, using a controlled paradigm that combines verifiable formatting constraints with benchmark tasks of increasing complexity. Across three model families and over 8,000 prompts, compliance drops by 2-21% under concurrent task load. Vulnerability is highly type-dependent: terminal constraints (requiring action at the response boundary) degrade most, with drops up to 50%, while avoidance constraints remain comparatively robust. A salience-enhanced format (explicit instruction framing plus a trailing reminder) recovers much of the lost compliance, restoring performance to 90-100% in many settings. Interference is bidirectional: formatting constraints can also reduce task accuracy, with one model's GSM8K accuracy dropping from 93% to 27%. In additional stacking experiments, joint compliance declines sharply as constraints accumulate. All results use deterministic programmatic checkers without an LLM-as-judge component on publicly available datasets.
>
---
#### [new 023] Plato's Cave: A Human-Centered Research Verification System
- **分类: cs.CL; cs.HC; cs.MA**

- **简介: 该论文提出Plato's Cave系统，用于研究论文的验证与评估，解决信息核实、质量评估和不可证声明识别问题，通过构建DAG并分析论证结构进行评分。**

- **链接: [https://arxiv.org/pdf/2603.23526](https://arxiv.org/pdf/2603.23526)**

> **作者:** Matheus Kunzler Maldaner; Raul Valle; Junsung Kim; Tonuka Sultan; Pranav Bhargava; Matthew Maloni; John Courtney; Hoang Nguyen; Aamogh Sawant; Kristian O'Connor; Stephen Wormald; Damon L. Woodard
>
> **备注:** 15 pages, 4 figures
>
> **摘要:** The growing publication rate of research papers has created an urgent need for better ways to fact-check information, assess writing quality, and identify unverifiable claims. We present Plato's Cave as an open-source, human-centered research verification system that (i) creates a directed acyclic graph (DAG) from a document, (ii) leverages web agents to assign credibility scores to nodes and edges from the DAG, and (iii) gives a final score by interpreting and evaluating the paper's argumentative structure. We report the system implementation and results on a collected dataset of 104 research papers.
>
---
#### [new 024] FinToolSyn: A forward synthesis Framework for Financial Tool-Use Dialogue Data with Dynamic Tool Retrieval
- **分类: cs.CL**

- **简介: 该论文提出FinToolSyn，解决金融对话数据生成问题，通过动态工具检索生成更真实的对话，提升模型工具调用能力。**

- **链接: [https://arxiv.org/pdf/2603.24051](https://arxiv.org/pdf/2603.24051)**

> **作者:** Caishuang Huang; Yang Qiao; Rongyu Zhang; Junjie Ye; Pu Lu; Wenxi Wu; Meng Zhou; Xiku Du; Tao Gui; Qi Zhang; Xuanjing Huang
>
> **摘要:** Tool-use capabilities are vital for Large Language Models (LLMs) in finance, a domain characterized by massive investment targets and data-intensive inquiries. However, existing data synthesis methods typically rely on a reverse synthesis paradigm, generating user queries from pre-sampled tools. This approach inevitably introduces artificial explicitness, yielding queries that fail to capture the implicit, event-driven nature of real-world needs. Moreover, its reliance on static tool sets overlooks the dynamic retrieval process required to navigate massive tool spaces. To address these challenges, we introduce \textit{FinToolSyn}, a forward synthesis framework designed to generate high-quality financial dialogues. Progressing from persona instruction and atomic tool synthesis to dynamic retrieval dialogue generation, our pipeline constructs a repository of 43,066 tools and synthesizes over 148k dialogue instances, incorporating dynamic retrieval to emulate the noisy candidate sets typical of massive tool spaces. We also establish a dedicated benchmark to evaluate tool-calling capabilities in realistic financial scenarios. Extensive experiments demonstrate that models trained on FinToolSyn achieve a 21.06\% improvement, providing a robust foundation for tool learning in financial scenarios.
>
---
#### [new 025] Navigating the Concept Space of Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型概念空间探索任务，旨在解决SAE特征难以大规模发现和分析的问题。提出Concept Explorer系统，通过层次化嵌入实现概念的渐进式导航与分析。**

- **链接: [https://arxiv.org/pdf/2603.23524](https://arxiv.org/pdf/2603.23524)**

> **作者:** Wilson E. Marcílio-Jr; Danilo M. Eler
>
> **摘要:** Sparse autoencoders (SAEs) trained on large language model activations output thousands of features that enable mapping to human-interpretable concepts. The current practice for analyzing these features primarily relies on inspecting top-activating examples, manually browsing individual features, or performing semantic search on interested concepts, which makes exploratory discovery of concepts difficult at scale. In this paper, we present Concept Explorer, a scalable interactive system for post-hoc exploration of SAE features that organizes concept explanations using hierarchical neighborhood embeddings. Our approach constructs a multi-resolution manifold over SAE feature embeddings and enables progressive navigation from coarse concept clusters to fine-grained neighborhoods, supporting discovery, comparison, and relationship analysis among concepts. We demonstrate the utility of Concept Explorer on SAE features extracted from SmolLM2, where it reveals coherent high-level structure, meaningful subclusters, and distinctive rare concepts that are hard to identify with existing workflows.
>
---
#### [new 026] MedAidDialog: A Multilingual Multi-Turn Medical Dialogue Dataset for Accessible Healthcare
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于医疗对话任务，旨在解决单轮对话和单一语言的局限性。构建了多语言多轮医学对话数据集MedAidDialog，并开发了轻量级医疗对话模型MedAidLM。**

- **链接: [https://arxiv.org/pdf/2603.24132](https://arxiv.org/pdf/2603.24132)**

> **作者:** Shubham Kumar Nigam; Suparnojit Sarkar; Piyush Patel
>
> **摘要:** Conversational artificial intelligence has the potential to assist users in preliminary medical consultations, particularly in settings where access to healthcare professionals is limited. However, many existing medical dialogue systems operate in a single-turn question--answering paradigm or rely on template-based datasets, limiting conversational realism and multilingual applicability. In this work, we introduce MedAidDialog, a multilingual multi-turn medical dialogue dataset designed to simulate realistic physician--patient consultations. The dataset extends the MDDial corpus by generating synthetic consultations using large language models and further expands them into a parallel multilingual corpus covering seven languages: English, Hindi, Telugu, Tamil, Bengali, Marathi, and Arabic. Building on this dataset, we develop MedAidLM, a conversational medical model trained using parameter-efficient fine-tuning on quantized small language models, enabling deployment without high-end computational infrastructure. Our framework additionally incorporates optional patient pre-context information (e.g., age, gender, allergies) to personalize the consultation process. Experimental results demonstrate that the proposed system can effectively perform symptom elicitation through multi-turn dialogue and generate diagnostic recommendations. We further conduct medical expert evaluation to assess the plausibility and coherence of the generated consultations.
>
---
#### [new 027] DISCO: Document Intelligence Suite for COmparative Evaluation
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出DISCO，用于评估OCR和VLM在文档处理中的表现，解决文档类型多样性带来的性能差异问题，通过实验指导策略选择。**

- **链接: [https://arxiv.org/pdf/2603.23511](https://arxiv.org/pdf/2603.23511)**

> **作者:** Kenza Benkirane; Dan Goldwater; Martin Asenov; Aneiss Ghodsi
>
> **备注:** Accepted at the ICLR 2026 Workshop on Multimodal Intelligence (MMIntelligence). 10 pages, 7 figures
>
> **摘要:** Document intelligence requires accurate text extraction and reliable reasoning over document content. We introduce \textbf{DISCO}, a \emph{Document Intelligence Suite for COmparative Evaluation}, that evaluates optical character recognition (OCR) pipelines and vision-language models (VLMs) separately on parsing and question answering across diverse document types, including handwritten text, multilingual scripts, medical forms, infographics, and multi-page documents. Our evaluation shows that performance varies substantially across tasks and document characteristics, underscoring the need for complexity-aware approach selection. OCR pipelines are generally more reliable for handwriting and for long or multi-page documents, where explicit text grounding supports text-heavy reasoning, while VLMs perform better on multilingual text and visually rich layouts. Task-aware prompting yields mixed effects, improving performance on some document types while degrading it on others. These findings provide empirical guidance for selecting document processing strategies based on document structure and reasoning demands.
>
---
#### [new 028] S-Path-RAG: Semantic-Aware Shortest-Path Retrieval Augmented Generation for Multi-Hop Knowledge Graph Question Answering
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文提出S-Path-RAG，用于多跳知识图谱问答任务，解决传统方法效率低、路径不准确的问题。通过语义加权路径检索和生成结合，提升答案准确性和效率。**

- **链接: [https://arxiv.org/pdf/2603.23512](https://arxiv.org/pdf/2603.23512)**

> **作者:** Rong Fu; Yemin Wang; Tianxiang Xu; Yongtai Liu; Weizhi Tang; Wangyu Wu; Xiaowen Ma; Simon Fong
>
> **摘要:** We present S-Path-RAG, a semantic-aware shortest-path Retrieval-Augmented Generation framework designed to improve multi-hop question answering over large knowledge graphs. S-Path-RAG departs from one-shot, text-heavy retrieval by enumerating bounded-length, semantically weighted candidate paths using a hybrid weighted $k$-shortest, beam, and constrained random-walk strategy, learning a differentiable path scorer together with a contrastive path encoder and lightweight verifier, and injecting a compact soft mixture of selected path latents into a language model via cross-attention. The system runs inside an iterative Neural-Socratic Graph Dialogue loop in which concise diagnostic messages produced by the language model are mapped to targeted graph edits or seed expansions, enabling adaptive retrieval when the model expresses uncertainty. This combination yields a retrieval mechanism that is both token-efficient and topology-aware while preserving interpretable path-level traces for diagnostics and intervention. We validate S-Path-RAG on standard multi-hop KGQA benchmarks and through ablations and diagnostic analyses. The results demonstrate consistent improvements in answer accuracy, evidence coverage, and end-to-end efficiency compared to strong graph- and LLM-based baselines. We further analyze trade-offs between semantic weighting, verifier filtering, and iterative updates, and report practical recommendations for deployment under constrained compute and token budgets.
>
---
#### [new 029] BeliefShift: Benchmarking Temporal Belief Consistency and Opinion Drift in LLM Agents
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于语言模型评估任务，旨在解决LLM在长期交互中的信念一致性与观点漂移问题。通过构建基准数据集，评估模型在多轮对话中的信念动态表现。**

- **链接: [https://arxiv.org/pdf/2603.23848](https://arxiv.org/pdf/2603.23848)**

> **作者:** Praveen Kumar Myakala; Manan Agrawal; Rahul Manche
>
> **摘要:** LLMs are increasingly used as long-running conversational agents, yet every major benchmark evaluating their memory treats user information as static facts to be stored and retrieved. That's the wrong model. People change their minds, and over extended interactions, phenomena like opinion drift, over-alignment, and confirmation bias start to matter a lot. BeliefShift introduces a longitudinal benchmark designed specifically to evaluate belief dynamics in multi-session LLM interactions. It covers three tracks: Temporal Belief Consistency, Contradiction Detection, and Evidence-Driven Revision. The dataset includes 2,400 human-annotated multi-session interaction trajectories spanning health, politics, personal values, and product preferences. We evaluate seven models including GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro, LLaMA-3, and Mistral-Large under zero-shot and retrieval-augmented generation (RAG) settings. Results reveal a clear trade-off: models that personalize aggressively resist drift poorly, while factually grounded models miss legitimate belief updates. We further introduce four novel evaluation metrics: Belief Revision Accuracy (BRA), Drift Coherence Score (DCS), Contradiction Resolution Rate (CRR), and Evidence Sensitivity Index (ESI).
>
---
#### [new 030] From Physician Expertise to Clinical Agents: Preserving, Standardizing, and Scaling Physicians' Medical Expertise with Lightweight LLM
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医学知识建模任务，旨在解决医生经验难以标准化和规模化的问题。通过构建框架Med-Shicheng，将名医经验转化为可迁移的模型知识。**

- **链接: [https://arxiv.org/pdf/2603.23520](https://arxiv.org/pdf/2603.23520)**

> **作者:** Chanyong Luo; Jirui Dai; Zhendong Wang; Kui Chen; Jiaxi Yang; Bingjie Lu; Jing Wang; Jiaxin Hao; Bing Li; Ruiyang He; Yiyu Qiao; Chenkai Zhang; Kaiyu Wang; Zhi Liu; Zeyu Zheng; Yan Li; Xiaohong Gu
>
> **摘要:** Medicine is an empirical discipline refined through long-term observation and the messy, high-variance reality of clinical practice. Physicians build diagnostic and therapeutic competence through repeated cycles of application, reflection, and improvement, forming individualized methodologies. Yet outcomes vary widely, and master physicians' knowledge systems are slow to develop and hard to transmit at scale, contributing to the scarcity of high-quality clinical expertise. To address this, we propose Med-Shicheng, a general framework that enables large language models to systematically learn and transfer distinguished physicians' diagnostic-and-therapeutic philosophy and case-dependent adaptation rules in a standardized way. Built on Tianyi, Med-Shicheng consists of five stages. We target five National Masters of Chinese Medicine or distinguished TCM physicians, curate multi-source materials, and train a single model to internalize all five knowledge systems across seven tasks, including etiology-pathogenesis analysis, syndrome diagnosis, treatment principle selection, prescription generation, prescription explanation, symptom evolution with regimen adjustment, and clinical advice. Implemented on Qwen2.5-1.5B-Base, Med-Shicheng runs on resource-constrained GPUs while achieving performance comparable to DeepSeek-R1 and GPT-5. We also examine the reliability of LLM-as-a-judge versus physician evaluation: automated judging tracks overall trends but shows bias on fine-grained individualized distinctions, highlighting the need for physician involvement when ground truth is unavailable and for domain-adapted judge models.
>
---
#### [new 031] Sparse Growing Transformer: Training-Time Sparse Depth Allocation via Progressive Attention Looping
- **分类: cs.CL**

- **简介: 该论文提出Sparse Growing Transformer（SGT），解决Transformer训练中计算冗余问题。通过动态扩展深度，提升效率并保持性能。属于模型优化任务。**

- **链接: [https://arxiv.org/pdf/2603.23998](https://arxiv.org/pdf/2603.23998)**

> **作者:** Yao Chen; Yilong Chen; Yinqi Yang; Junyuan Shang; Zhenyu Zhang; Zefeng Zhang; Shuaiyi Nie; Shuohuan Wang; Yu Sun; Hua Wu; HaiFeng Wang; Tingwen Liu
>
> **摘要:** Existing approaches to increasing the effective depth of Transformers predominantly rely on parameter reuse, extending computation through recursive execution. Under this paradigm, the network structure remains static along the training timeline, and additional computational depth is uniformly assigned to entire blocks at the parameter level. This rigidity across training time and parameter space leads to substantial computational redundancy during training. In contrast, we argue that depth allocation during training should not be a static preset, but rather a progressively growing structural process. Our systematic analysis reveals a deep-to-shallow maturation trajectory across layers, where high-entropy attention heads play a crucial role in semantic integration. Motivated by this observation, we introduce the Sparse Growing Transformer (SGT). SGT is a training-time sparse depth allocation framework that progressively extends recurrence from deeper to shallower layers via targeted attention looping on informative heads. This mechanism induces structural sparsity by selectively increasing depth only for a small subset of parameters as training evolves. Extensive experiments across multiple parameter scales demonstrate that SGT consistently outperforms training-time static block-level looping baselines under comparable settings, while reducing the additional training FLOPs overhead from approximately 16--20% to only 1--3% relative to a standard Transformer backbone.
>
---
#### [new 032] Why Does Self-Distillation (Sometimes) Degrade the Reasoning Capability of LLMs?
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究自蒸馏对大模型推理能力的影响，发现其可能因抑制不确定性表达而降低性能。任务为模型优化，解决问题是自蒸馏在数学推理中的负面效应。工作包括实验分析与多模型验证。**

- **链接: [https://arxiv.org/pdf/2603.24472](https://arxiv.org/pdf/2603.24472)**

> **作者:** Jeonghye Kim; Xufang Luo; Minbeom Kim; Sangmook Lee; Dohyung Kim; Jiwon Jeon; Dongsheng Li; Yuqing Yang
>
> **摘要:** Self-distillation has emerged as an effective post-training paradigm for LLMs, often improving performance while shortening reasoning traces. However, in mathematical reasoning, we find that it can reduce response length while degrading performance. We trace this degradation to the suppression of epistemic verbalization - the model's expression of uncertainty during reasoning. Through controlled experiments varying conditioning context richness and task coverage, we show that conditioning the teacher on rich information suppresses uncertainty expression, enabling rapid in-domain optimization with limited task coverage but harming OOD performance, where unseen problems benefit from expressing uncertainty and adjusting accordingly. Across Qwen3-8B, DeepSeek-Distill-Qwen-7B, and Olmo3-7B-Instruct, we observe performance drops of up to 40%. Our findings highlight that exposing appropriate levels of uncertainty is crucial for robust reasoning and underscore the importance of optimizing reasoning behavior beyond merely reinforcing correct answer traces.
>
---
#### [new 033] Compression Method Matters: Benchmark-Dependent Output Dynamics in LLM Prompt Compression
- **分类: cs.CL**

- **简介: 该论文研究LLM提示压缩对输出动态的影响，解决压缩效果评估不准确的问题。通过实验分析不同基准下的压缩表现，提出结构度量和评估指标，以提高压缩安全性与效率。**

- **链接: [https://arxiv.org/pdf/2603.23527](https://arxiv.org/pdf/2603.23527)**

> **作者:** Warren Johnson
>
> **备注:** 19 pages. Includes figures and tables. Companion code/data repository and direct NVML calibration dataset are cited in manuscript
>
> **摘要:** Prompt compression is often evaluated by input-token reduction, but its real deployment impact depends on how compression changes output length and total inference cost. We present a controlled replication and extension study of benchmark-dependent output dynamics under aggressive compression, covering 5,400 API calls across three benchmarks and multiple providers. To explain conflicting prior observations, we formalize instruction survival probability (Psi), a structural metric that captures whether task-critical prompt segments remain after truncation. Results show a strong benchmark effect: under r=0.3, DeepSeek exhibits severe output expansion on MBPP (56x, Psi approx 0.15) but substantially lower expansion on HumanEval (5x, Psi approx 0.72), while GPT-4o-mini is comparatively stable across benchmarks. This reconciles the apparent discrepancy between previously reported extreme explosion and lower replication effects by identifying prompt structure, not provider identity alone, as the primary moderator. We introduce the Compression Robustness Index (CRI) for cross-benchmark evaluation and show that single-benchmark assessments can produce misleading conclusions about compression safety and efficiency. To contextualize energy claims, we incorporate companion direct NVML measurements from rented RunPod GPUs and show that token savings can overstate joule savings. These findings motivate benchmark-diverse testing and structure-aware compression policies for reliable, energy-conscious LLM deployment.
>
---
#### [new 034] Perturbation: A simple and efficient adversarial tracer for representation learning in language models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于语言模型表示学习任务，旨在解决传统方法在表示定义上的困境。通过对抗扰动方法，揭示语言模型中的结构化迁移与语言抽象能力。**

- **链接: [https://arxiv.org/pdf/2603.23821](https://arxiv.org/pdf/2603.23821)**

> **作者:** Joshua Rozner; Cory Shain
>
> **摘要:** Linguistic representation learning in deep neural language models (LMs) has been studied for decades, for both practical and theoretical reasons. However, finding representations in LMs remains an unsolved problem, in part due to a dilemma between enforcing implausible constraints on representations (e.g., linearity; Arora et al. 2024) and trivializing the notion of representation altogether (Sutter et al., 2025). Here we escape this dilemma by reconceptualizing representations not as patterns of activation but as conduits for learning. Our approach is simple: we perturb an LM by fine-tuning it on a single adversarial example and measure how this perturbation ``infects'' other examples. Perturbation makes no geometric assumptions, and unlike other methods, it does not find representations where it should not (e.g., in untrained LMs). But in trained LMs, perturbation reveals structured transfer at multiple linguistic grain sizes, suggesting that LMs both generalize along representational lines and acquire linguistic abstractions from experience alone.
>
---
#### [new 035] Representation Learning to Study Temporal Dynamics in Tutorial Scaffolding
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于教育技术领域，旨在解决如何测量 tutoring 对话中的支架动态问题。通过语义嵌入分析对话轮次与问题、解答的对齐情况，揭示角色差异及时间模式。**

- **链接: [https://arxiv.org/pdf/2603.24535](https://arxiv.org/pdf/2603.24535)**

> **作者:** Conrad Borchers; Jiayi Zhang; Ashish Gurung
>
> **备注:** Accepted as short paper to the 27th International Conference on Artificial Intelligence in Education (AIED 2026)
>
> **摘要:** Adaptive scaffolding enhances learning, yet the field lacks robust methods for measuring it within authentic tutoring dialogue. This gap has become more pressing with the rise of remote human tutoring and large language model-based systems. We introduce an embedding-based approach that analyzes scaffolding dynamics by aligning the semantics of dialogue turns, problem statements, and correct solutions. Specifically, we operationalize alignment by computing cosine similarity between tutor and student contributions and task-relevant content. We apply this framework to 1,576 real-world mathematics tutoring dialogues from the Eedi Question Anchored Tutoring Dialogues dataset. The analysis reveals systematic differences in task alignment and distinct temporal patterns in how participants ground their contributions in problem and solution content. Further, mixed-effects models show that role-specific semantic alignment predicts tutorial progression beyond baseline features such as message order and length. Tutor contributions exhibited stronger grounding in problem content early in interactions. In contrast, student solution alignment was modestly positively associated with progression. These findings support scaffolding as a continuous, role-sensitive process grounded in task semantics. By capturing role-specific alignment over time, this approach provides a principled method for analyzing instructional dialogue and evaluating conversational tutoring systems.
>
---
#### [new 036] MSA: Memory Sparse Attention for Efficient End-to-End Memory Model Scaling to 100M Tokens
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文提出MSA框架，解决长文本处理中内存容量与推理效率的瓶颈问题。属于自然语言处理任务，通过稀疏注意力等技术实现100M令牌的高效记忆模型。**

- **链接: [https://arxiv.org/pdf/2603.23516](https://arxiv.org/pdf/2603.23516)**

> **作者:** Yu Chen; Runkai Chen; Sheng Yi; Xinda Zhao; Xiaohong Li; Jianjin Zhang; Jun Sun; Chuanrui Hu; Yunyun Han; Lidong Bing; Yafeng Deng; Tianqiao Chen
>
> **摘要:** Long-term memory is a cornerstone of human intelligence. Enabling AI to process lifetime-scale information remains a long-standing pursuit in the field. Due to the constraints of full-attention architectures, the effective context length of large language models (LLMs) is typically limited to 1M tokens. Existing approaches, such as hybrid linear attention, fixed-size memory states (e.g., RNNs), and external storage methods like RAG or agent systems, attempt to extend this limit. However, they often suffer from severe precision degradation and rapidly increasing latency as context length grows, an inability to dynamically modify memory content, or a lack of end-to-end optimization. These bottlenecks impede complex scenarios like large-corpus summarization, Digital Twins, and long-history agent reasoning, while limiting memory capacity and slowing inference. We present Memory Sparse Attention (MSA), an end-to-end trainable, efficient, and massively scalable memory model framework. Through core innovations including scalable sparse attention and document-wise RoPE, MSA achieves linear complexity in both training and inference while maintaining exceptional stability, exhibiting less than 9% degradation when scaling from 16K to 100M tokens. Furthermore, KV cache compression, combined with Memory Parallel, enables 100M-token inference on 2xA800 GPUs. We also propose Memory Interleaving to facilitate complex multi-hop reasoning across scattered memory segments. MSA significantly surpasses frontier LLMs, state-of-the-art RAG systems, and leading memory agents in long-context benchmarks. These results demonstrate that by decoupling memory capacity from reasoning, MSA provides a scalable foundation to endow general-purpose models with intrinsic, lifetime-scale memory.
>
---
#### [new 037] Self-Distillation for Multi-Token Prediction
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，针对大语言模型推理效率问题，提出MTP-D方法提升多标记预测性能与速度。**

- **链接: [https://arxiv.org/pdf/2603.23911](https://arxiv.org/pdf/2603.23911)**

> **作者:** Guoliang Zhao; Ruobing Xie; An Wang; Shuaipeng Li; Huaibing Xie; Xingwu Sun
>
> **摘要:** As Large Language Models (LLMs) scale up, inference efficiency becomes a critical bottleneck. Multi-Token Prediction (MTP) could accelerate LLM inference by predicting multiple future tokens in parallel. However, existing MTP approaches still face two challenges: limited acceptance rates of MTP heads, and difficulties in jointly training multiple MTP heads. Therefore, we propose MTP-D, a simple yet effective self-distillation method with minimal additional training cost, which boosts MTP head acceptance rates (+7.5\%) while maximumly preserving main-head performance. We also introduce a looped extension strategy for MTP-D, enabling effective and economical MTP head extension and further significant inference speedup to 1-head MTP (+220.4\%). Moreover, we systematically explore and validate key insights on the distillation strategies and the potential scalability of MTP through extensive experiments on seven benchmarks. These results demonstrate that our MTP-D and looped extension strategy effectively enhance MTP-head performance and inference efficiency, facilitating the practical usage of MTP in LLMs.
>
---
#### [new 038] Berta: an open-source, modular tool for AI-enabled clinical documentation
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文介绍Berta，一个开源模块化AI临床记录工具，解决商业AI语音助手成本高、不透明及数据控制弱的问题。通过部署在AHS系统，实现高效、低成本的临床记录。**

- **链接: [https://arxiv.org/pdf/2603.23513](https://arxiv.org/pdf/2603.23513)**

> **作者:** Samridhi Vaid; Mike Weldon; Jesse Dunn; Sacha Davis; Kevin Lonergan; Henry Li; Jeffrey Franc; Mohamed Abdalla; Daniel C. Baumgart; Jake Hayward; J Ross Mitchell
>
> **摘要:** Commercial AI scribes cost \$99-600 per physician per month, operate as opaque systems, and do not return data to institutional infrastructure, limiting organizational control over data governance, quality improvement, and clinical workflows. We developed Berta, an open-source modular scribe platform for AI-enabled clinical documentation, and deployed a customized implementation within Alberta Health Services (AHS) integrated with their existing Snowflake AI Data Cloud infrastructure. The system combines automatic speech recognition with large language models while retaining all clinical data within the secure AHS environment. During eight months (November 2024 to July 2025), 198 emergency physicians used the system in 105 urban and rural facilities, generating 22148 clinical sessions and more than 2800 hours of audio. The use grew from 680 to 5530 monthly sessions. Operating costs averaged less than \$30 per physician per month, a 70-95% reduction compared to commercial alternatives. AHS has since approved expansion to 850 physicians. This is the first provincial-scale deployment of an AI scribe integrated with existing health system infrastructure. By releasing Berta as open source, we provide a reproducible, cost-effective alternative that health systems can adapt to their own secure environments, supporting data sovereignty and informed evaluation of AI documentation technology.
>
---
#### [new 039] From Oracle to Noisy Context: Mitigating Contextual Exposure Bias in Speech-LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语音大模型任务，解决上下文暴露偏差问题。通过引入噪声历史和优化方法，提升模型在真实场景下的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.24034](https://arxiv.org/pdf/2603.24034)**

> **作者:** Xiaoyong Guo; Nanjie Li; Zijie Zeng; Kai Wang; Hao Huang; Haihua Xu; Wei Shi
>
> **摘要:** Contextual automatic speech recognition (ASR) with Speech-LLMs is typically trained with oracle conversation history, but relies on error-prone history at inference, causing a train-test mismatch in the context channel that we term contextual exposure bias. We propose a unified training framework to improve robustness under realistic histories: (i) Teacher Error Knowledge by using Whisper large-v3 hypotheses as training-time history, (ii) Context Dropout to regularize over-reliance on history, and (iii) Direct Preference Optimization (DPO) on curated failure cases. Experiments on TED-LIUM 3 (in-domain) and zero-shot LibriSpeech (out-of-domain) show consistent gains under predicted-history decoding. With a two-utterance history as context, SFT with Whisper hypotheses reduce WER from 5.59% (oracle-history training) to 5.47%, and DPO further improves to 5.17%. Under irrelevant-context attacks, DPO yields the smallest degradation (5.17% -> 5.63%), indicating improved robustness to misleading context. Our code and models are published on this https URL.
>
---
#### [new 040] Probing Ethical Framework Representations in Large Language Models: Structure, Entanglement, and Methodological Challenges
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的伦理分析任务，探讨大模型内部是否区分不同伦理框架。通过分析五种伦理框架在六种模型中的表示，揭示其结构差异与方法挑战。**

- **链接: [https://arxiv.org/pdf/2603.23659](https://arxiv.org/pdf/2603.23659)**

> **作者:** Weilun Xu; Alexander Rusnak; Frederic Kaplan
>
> **摘要:** When large language models make ethical judgments, do their internal representations distinguish between normative frameworks, or collapse ethics into a single acceptability dimension? We probe hidden representations across five ethical frameworks (deontology, utilitarianism, virtue, justice, commonsense) in six LLMs spanning 4B--72B parameters. Our analysis reveals differentiated ethical subspaces with asymmetric transfer patterns -- e.g., deontology probes partially generalize to virtue scenarios while commonsense probes fail catastrophically on justice. Disagreement between deontological and utilitarian probes correlates with higher behavioral entropy across architectures, though this relationship may partly reflect shared sensitivity to scenario difficulty. Post-hoc validation reveals that probes partially depend on surface features of benchmark templates, motivating cautious interpretation. We discuss both the structural insights these methods provide and their epistemological limitations.
>
---
#### [new 041] Towards Reward Modeling for AI Tutors in Math Mistake Remediation
- **分类: cs.CL**

- **简介: 该论文属于AI辅导任务，旨在提升AI在数学错误纠正中的教学质量。通过构建对比响应对并训练奖励模型，解决传统指标无法评估教学效果的问题。**

- **链接: [https://arxiv.org/pdf/2603.24375](https://arxiv.org/pdf/2603.24375)**

> **作者:** Kseniia Petukhova; Ekaterina Kochmar
>
> **摘要:** Evaluating the pedagogical quality of AI tutors remains challenging: standard NLG metrics do not determine whether responses identify mistakes, scaffold reasoning, or avoid revealing the answers. For the task of mistake remediation, we derive a hierarchy of pedagogical aspects from human pairwise preferences on MRBench, and synthesize minimally contrastive response pairs that differ along key aspects (e.g., mistake identification and location, targetedness, scaffolding, actionability, clarity, and coherence). We develop and release Bradley-Terry preference models trained on weighted-sum rankings that we automatically create from MRBench, synthetic pairs, and data combinations. Using only synthetic data, our best model reaches 0.69 pairwise accuracy on a human preference test, and combining weighted-sum data with targeted synthetic groups improves accuracy to 0.74, outperforming larger general-purpose reward models while using only a 0.5B-parameter backbone.
>
---
#### [new 042] A Sociolinguistic Analysis of Automatic Speech Recognition Bias in Newcastle English
- **分类: cs.CL; cs.AI; cs.CV; cs.SD**

- **简介: 该论文属于自然语言处理任务，研究ASR系统在纽卡斯尔英语中的偏见问题。通过分析语音识别错误，揭示社会因素与语音变异对识别准确率的影响。**

- **链接: [https://arxiv.org/pdf/2603.24549](https://arxiv.org/pdf/2603.24549)**

> **作者:** Dana Serditova; Kevin Tang
>
> **备注:** 54 pages, 11 figures
>
> **摘要:** Automatic Speech Recognition (ASR) systems are widely used in everyday communication, education, healthcare, and industry, yet their performance remains uneven across speakers, particularly when dialectal variation diverges from the mainstream accents represented in training data. This study investigates ASR bias through a sociolinguistic analysis of Newcastle English, a regional variety of North-East England that has been shown to challenge current speech recognition technologies. Using spontaneous speech from the Diachronic Electronic Corpus of Tyneside English (DECTE), we evaluate the output of a state-of-the-art commercial ASR system and conduct a fine-grained analysis of more than 3,000 transcription errors. Errors are classified by linguistic domain and examined in relation to social variables including gender, age, and socioeconomic status. In addition, an acoustic case study of selected vowel features demonstrates how gradient phonetic variation contributes directly to misrecognition. The results show that phonological variation accounts for the majority of errors, with recurrent failures linked to dialect-specific features like vowel quality and glottalisation, as well as local vocabulary and non-standard grammatical forms. Error rates also vary across social groups, with higher error frequencies observed for men and for speakers at the extremes of the age spectrum. These findings indicate that ASR errors are not random but socially patterned and can be explained from a sociolinguistic perspective. Thus, the study demonstrates the importance of incorporating sociolinguistic expertise into the evaluation and development of speech technologies and argues that more equitable ASR systems require explicit attention to dialectal variation and community-based speech data.
>
---
#### [new 043] Swiss-Bench SBP-002: A Frontier Model Comparison on Swiss Legal and Regulatory Tasks
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于瑞士法律与监管任务的模型评估，旨在解决前沿模型在合规任务中的表现问题。构建了三语种基准测试，评估十种模型，揭示性能差异。**

- **链接: [https://arxiv.org/pdf/2603.23646](https://arxiv.org/pdf/2603.23646)**

> **作者:** Fatih Uenal
>
> **备注:** 21 pages, 5 figures, 7 tables. Code and data: this https URL
>
> **摘要:** While recent work has benchmarked large language models on Swiss legal translation (Niklaus et al., 2025) and academic legal reasoning from university exams (Fan et al., 2025), no existing benchmark evaluates frontier model performance on applied Swiss regulatory compliance tasks. I introduce Swiss-Bench SBP-002, a trilingual benchmark of 395 expert-crafted items spanning three Swiss regulatory domains (FINMA, Legal-CH, EFK), seven task types, and three languages (German, French, Italian), and evaluate ten frontier models from March 2026 using a structured three-dimension scoring framework assessed via a blind three-judge LLM panel (GPT-4o, Claude Sonnet 4, Qwen3-235B) with majority-vote aggregation and weighted kappa = 0.605, with reference answers validated by an independent human legal expert on a 100-item subset (73% rated Correct, 0% Incorrect, perfect Legal Accuracy). Results reveal three descriptive performance clusters: Tier A (35-38% correct), Tier B (26-29%), and Tier C (13-21%). The benchmark proves difficult: even the top-ranked model (Qwen 3.5 Plus) achieves only 38.2% correct, with 47.3% incorrect and 14.4% partially correct. Task type difficulty varies widely: legal translation and case analysis yield 69-72% correct rates, while regulatory Q&A, hallucination detection, and gap analysis remain below 9%. Within this roster (seven open-weight, three closed-source), an open-weight model leads the ranking, and several open-weight models match or outperform their closed-source counterparts. These findings provide an initial empirical reference point for assessing frontier model capability on Swiss regulatory tasks under zero-retrieval conditions.
>
---
#### [new 044] Thinking with Tables: Enhancing Multi-Modal Tabular Understanding via Neuro-Symbolic Reasoning
- **分类: cs.CL**

- **简介: 该论文聚焦于Tabular-Vision Multi-Modal Understanding任务，解决表格数据在多模态学习中的结构复杂、特征依赖隐晦及任务异质性等问题，提出TWT方法提升表征能力。**

- **链接: [https://arxiv.org/pdf/2603.24004](https://arxiv.org/pdf/2603.24004)**

> **作者:** Kun-Yang Yu; Zhi Zhou; Shi-Yu Tian; Xiao-Wen Yang; Zi-Yi Jia; Ming Yang; Zi-Jian Cheng; Lan-Zhe Guo; Yu-Feng Li
>
> **备注:** 20 pages, 6 figures
>
> **摘要:** Multimodal Large Language Models (MLLMs) have demonstrated remarkable reasoning capabilities across modalities such as images and text. However, tabular data, despite being a critical real-world modality, remains relatively underexplored in multimodal learning. In this paper, we focus on the task of Tabular-Vision Multi-Modal Understanding (TVMU) and identify three core challenges: (1) high structural variability and data incompleteness in tables, (2) implicit and complex feature dependencies, and (3) significant heterogeneity in problem-solving pipelines across downstream tasks. To address these issues, we propose Thinking with Tables (TWT). TWT employs a program-aided code-based neuro-symbolic reasoning mechanism that facilitates key operations, such as information extraction and element modeling, by interacting with external environments. We evaluate TWT on eight representative datasets. Experimental results demonstrate that TWT consistently outperforms existing baselines by an average of 10\% in accuracy, achieving performance comparable to, or even surpassing, proprietary commercial SOTA LLMs on TVMU tasks. Models and codes are available at this https URL
>
---
#### [new 045] Fast and Faithful: Real-Time Verification for Long-Document Retrieval-Augmented Generation Systems
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于RAG系统验证任务，解决长文档生成答案不准确的问题。提出实时验证组件，实现全文档校验，提升准确性。**

- **链接: [https://arxiv.org/pdf/2603.23508](https://arxiv.org/pdf/2603.23508)**

> **作者:** Xunzhuo Liu; Bowei He; Xue Liu; Haichen Zhang; Huamin Chen
>
> **摘要:** Retrieval-augmented generation (RAG) is increasingly deployed in enterprise search and document-centric assistants, where responses must be grounded in long and complex source materials. In practice, verifying that generated answers faithfully reflect retrieved documents is difficult: large language models can check long contexts but are too slow and costly for interactive services, while lightweight classifiers operate within strict context limits and frequently miss evidence outside truncated passages. We present the design of a real-time verification component integrated into a production RAG pipeline that enables full-document grounding under latency constraints. The system processes documents up to 32K tokens and employs adaptive inference strategies to balance response time and verification coverage across workloads. We describe the architectural decisions, operational trade-offs, and evaluation methodology used to deploy the verifier, and show that full-context verification substantially improves detection of unsupported responses compared with truncated validation. Our experience highlights when long-context verification is necessary, why chunk-based checking often fails in real documents, and how latency budgets shape model design. These findings provide practical guidance for practitioners building reliable large-scale retrieval-augmented applications. (Model, benchmark, and code: this https URL)
>
---
#### [new 046] GameplayQA: A Benchmarking Framework for Decision-Dense POV-Synced Multi-Video Understanding of 3D Virtual Agents
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出GameplayQA，用于评估3D虚拟代理的感知与推理能力，解决多智能体视频理解中的决策密集问题。**

- **链接: [https://arxiv.org/pdf/2603.24329](https://arxiv.org/pdf/2603.24329)**

> **作者:** Yunzhe Wang; Runhui Xu; Kexin Zheng; Tianyi Zhang; Jayavibhav Niranjan Kogundi; Soham Hans; Volkan Ustun
>
> **摘要:** Multimodal LLMs are increasingly deployed as perceptual backbones for autonomous agents in 3D environments, from robotics to virtual worlds. These applications require agents to perceive rapid state changes, attribute actions to the correct entities, and reason about concurrent multi-agent behaviors from a first-person perspective, capabilities that existing benchmarks do not adequately evaluate. We introduce GameplayQA, a framework for evaluating agentic-centric perception and reasoning through video understanding. Specifically, we densely annotate multiplayer 3D gameplay videos at 1.22 labels/second, with time-synced, concurrent captions of states, actions, and events structured around a triadic system of Self, Other Agents, and the World, a natural decomposition for multi-agent environments. From these annotations, we refined 2.4K diagnostic QA pairs organized into three levels of cognitive complexity, accompanied by a structured distractor taxonomy that enables fine-grained analysis of where models hallucinate. Evaluation of frontier MLLMs reveals a substantial gap from human performance, with common failures in temporal and cross-video grounding, agent-role attribution, and handling the decision density of the game. We hope GameplayQA stimulates future research at the intersection of embodied AI, agentic perception, and world modeling.
>
---
#### [new 047] The Diminishing Returns of Early-Exit Decoding in Modern LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究LLM推理中的早停机制，分析其有效性随模型演进的下降趋势，提出评估指标与基准，旨在探索不同模型的早停潜力。**

- **链接: [https://arxiv.org/pdf/2603.23701](https://arxiv.org/pdf/2603.23701)**

> **作者:** Rui Wei; Rui Du; Hanfei Yu; Devesh Tiwari; Jian Li; Zhaozhuo Xu; Hao Wang
>
> **摘要:** In Large Language Model (LLM) inference, early-exit refers to stopping computation at an intermediate layer once the prediction is sufficiently confident, thereby reducing latency and cost. However, recent LLMs adopt improved pretraining recipes and architectures that reduce layer redundancy, potentially limiting early-exit opportunities. We re-evaluate layer-wise early-exit in modern LLMs and analyze how intermediate representations evolve during training. We introduce a metric to quantify a model's intrinsic suitability for early-exit and propose a benchmark for researchers to explore the potential early-exit benefits on different models and workloads. Our results show a diminishing trend in early-exit effectiveness across newer model generations. We further find that dense transformers generally offer greater early-exit potential than Mixture-of-Experts and State Space Models. In addition, larger models, particularly those with more than 20 billion parameters, and base pretrained models without specialized tuning tend to exhibit higher early-exit potential.
>
---
#### [new 048] ConceptKT: A Benchmark for Concept-Level Deficiency Prediction in Knowledge Tracing
- **分类: cs.CL**

- **简介: 该论文属于知识追踪任务，旨在解决学生概念理解不足的诊断问题。提出ConceptKT数据集，探索基于概念匹配的响应历史选择方法，提升错误预测与概念缺陷识别能力。**

- **链接: [https://arxiv.org/pdf/2603.24073](https://arxiv.org/pdf/2603.24073)**

> **作者:** Yu-Chen Kang; Yu-Chien Tang; An-Zi Yen
>
> **备注:** Accepted by LREC 2026
>
> **摘要:** Knowledge Tracing (KT) is a critical technique for modeling student knowledge to support personalized learning. However, most KT systems focus on binary correctness prediction and cannot diagnose the underlying conceptual misunderstandings that lead to errors. Such fine-grained diagnostic feedback is essential for designing targeted instruction and effective remediation. In this work, we introduce the task of concept-level deficiency prediction, which extends traditional KT by identifying the specific concepts a student is likely to struggle with on future problems. We present ConceptKT, a dataset annotated with labels that capture both the concepts required to solve each question and the missing concepts underlying incorrect responses. We investigate in-context learning approaches to KT and evaluate the diagnostic capabilities of various Large Language Models (LLMs) and Large Reasoning Models (LRMs). Different strategies for selecting informative historical records are explored. Experimental results demonstrate that selecting response histories based on conceptual alignment and semantic similarity leads to improved performance on both correctness prediction and concept-level deficiency identification.
>
---
#### [new 049] A visual observation on the geometry of UMAP projections of the difference vectors of antonym and synonym word pair embeddings
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理中的嵌入几何研究任务，旨在探究反义词与同义词对在嵌入空间中的几何差异。通过分析差向量的UMAP投影，发现反义词存在特定的“漩涡”结构。**

- **链接: [https://arxiv.org/pdf/2603.24150](https://arxiv.org/pdf/2603.24150)**

> **作者:** Rami Luisto
>
> **备注:** Code available at this https URL
>
> **摘要:** Antonyms, or opposites, are sometimes defined as \emph{word pairs that have all of the same contextually relevant properties but one}. Seeing how transformer models seem to encode concepts as directions, this begs the question if one can detect ``antonymity'' in the geometry of the embedding vectors of word pairs, especially based on their difference vectors. Such geometrical studies are then naturally contrasted by comparing antonymic pairs to their opposites; synonyms. This paper started as an exploratory project on the complexity of the systems needed to detect the geometry of the embedding vectors of antonymic word pairs. What we now report is a curious ``swirl'' that appears across embedding models in a somewhat specific projection configuration.
>
---
#### [new 050] Prompt Compression in Production Task Orchestration: A Pre-Registered Randomized Trial
- **分类: cs.CL**

- **简介: 该论文研究任务编排中的提示压缩问题，通过实验评估不同压缩策略对成本和输出相似性的影响，旨在优化压缩政策。**

- **链接: [https://arxiv.org/pdf/2603.23525](https://arxiv.org/pdf/2603.23525)**

> **作者:** Warren Johnson; Charles Lee
>
> **备注:** 28 pages, 9 tables, 1 CONSORT figure; pre-registered randomized controlled trial on production orchestration prompts
>
> **摘要:** The economics of prompt compression depend not only on reducing input tokens but on how compression changes output length, which is typically priced several times higher. We evaluate this in a pre-registered six-arm randomized controlled trial of prompt compression on production multi-agent task-orchestration, analyzing 358 successful Claude Sonnet 4.5 runs (59-61 per arm) drawn from a randomized corpus of 1,199 real orchestration instructions. We compare an uncompressed control with three uniform retention rates (r=0.8, 0.5, 0.2) and two structure-aware strategies (entropy-adaptive and recency-weighted), measuring total inference cost (input+output) and embedding-based response similarity. Moderate compression (r=0.5) reduced mean total cost by 27.9%, while aggressive compression (r=0.2) increased mean cost by 1.8% despite substantial input reduction, consistent with small mean output expansion (1.03x vs. control) and heavy-tailed uncertainty. Recency-weighted compression achieved 23.5% savings and, together with moderate compression, occupied the empirical cost-similarity Pareto frontier, whereas aggressive compression was dominated on both cost and similarity. These results show that "compress more" is not a reliable production heuristic and that output tokens must be treated as a first-class outcome when designing compression policies.
>
---
#### [new 051] Ethio-ASR: Joint Multilingual Speech Recognition and Language Identification for Ethiopian Languages
- **分类: cs.CL**

- **简介: 该论文属于多语言语音识别与语言识别任务，旨在解决埃塞俄比亚五种语言在语音技术中代表性不足的问题。研究者提出了Ethio-ASR模型，使用CTC框架进行联合训练，并在WAXAL语料库上进行了评估。**

- **链接: [https://arxiv.org/pdf/2603.23654](https://arxiv.org/pdf/2603.23654)**

> **作者:** Badr M. Abdullah; Israel Abebe Azime; Atnafu Lambebo Tonja; Jesujoba O. Alabi; Abel Mulat Alemu; Eyob G. Hagos; Bontu Fufa Balcha; Mulubrhan A. Nerea; Debela Desalegn Yadeta; Dagnachew Mekonnen Marilign; Amanuel Temesgen Fentahun; Tadesse Kebede; Israel D. Gebru; Michael Melese Woldeyohannis; Walelign Tewabe Sewunetie; Bernd Möbius; Dietrich Klakow
>
> **备注:** Preprint (under review)
>
> **摘要:** We present Ethio-ASR, a suite of multilingual CTC-based automatic speech recognition (ASR) models jointly trained on five Ethiopian languages: Amharic, Tigrinya, Oromo, Sidaama, and Wolaytta. These languages belong to the Semitic, Cushitic, and Omotic branches of the Afroasiatic family, and remain severely underrepresented in speech technology despite being spoken by the vast majority of Ethiopia's population. We train our models on the recently released WAXAL corpus using several pre-trained speech encoders and evaluate against strong multilingual baselines, including OmniASR. Our best model achieves an average WER of 30.48% on the WAXAL test set, outperforming the best OmniASR model with substantially fewer parameters. We further provide a comprehensive analysis of gender bias, the contribution of vowel length and consonant gemination to ASR errors, and the training dynamics of multilingual CTC models. Our models and codebase are publicly available to the research community.
>
---
#### [new 052] Mechanic: Sorrifier-Driven Formal Decomposition Workflow for Automated Theorem Proving
- **分类: cs.CL**

- **简介: 该论文属于自动化定理证明任务，旨在解决复杂数学推理中证明策略反复失败的问题。提出Mechanic系统，通过形式化分解独立解决子问题，提升证明效率。**

- **链接: [https://arxiv.org/pdf/2603.24465](https://arxiv.org/pdf/2603.24465)**

> **作者:** Ruichen Qiu; Yichuan Cao; Junqi Liu; Dakai Guo; Xiao-Shan Gao; Lihong Zhi; Ruyong Feng
>
> **摘要:** Recent advances in large language models (LLMs) and LLM-based agents have substantially improved the capabilities of automated theorem proving. However, for problems requiring complex mathematical reasoning, current systems rarely succeed on the first try and must repeatedly modify their proof strategies. Existing approaches for handling failed attempts typically either discard the entire proof and regenerate it from scratch or iteratively fix errors within the proof. The former is inefficient, as it may abandon mostly correct reasoning due to localized errors, while the latter, although preserving prior progress, leads to progressively longer contexts which progressively degrades the model's ability to attend to the remaining unresolved subproblems. To address this dilemma, we propose Mechanic, a novel agent system that employs a sorry-driven formal decomposition strategy. By leveraging the sorry placeholder in Lean to precisely isolate unresolved subgoals while preserving the surrounding verified proof structure, Mechanic extracts each failed subproblem into a clean, self-contained context and resolves it independently. This avoids both the waste of full regeneration and the excessive context length induced by repeated repairs. Experimental results on challenging mathematical competition benchmarks, including IMO 2025 and Putnam 2025, demonstrate that our agent achieves significant advantages in proving efficiency.
>
---
#### [new 053] Robust Multilingual Text-to-Pictogram Mapping for Scalable Reading Rehabilitation
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于多语言文本到图示映射任务，旨在提升特殊需求儿童的阅读理解能力。通过AI系统自动为文本添加视觉辅助，解决传统一对一教学难以规模化的问题。**

- **链接: [https://arxiv.org/pdf/2603.24536](https://arxiv.org/pdf/2603.24536)**

> **作者:** Soufiane Jhilal; Martina Galletti
>
> **摘要:** Reading comprehension presents a significant challenge for children with Special Educational Needs and Disabilities (SEND), often requiring intensive one-on-one reading support. To assist therapists in scaling this support, we developed a multilingual, AI-powered interface that automatically enhances text with visual scaffolding. This system dynamically identifies key concepts and maps them to contextually relevant pictograms, supporting learners across languages. We evaluated the system across five typologically diverse languages (English, French, Italian, Spanish, and Arabic), through multilingual coverage analysis, expert clinical review by speech therapists and special education professionals, and latency assessment. Evaluation results indicate high pictogram coverage and visual scaffolding density across the five languages. Expert audits suggested that automatically selected pictograms were semantically appropriate, with combined correct and acceptable ratings exceeding 95% for the four European languages and approximately 90% for Arabic despite reduced pictogram repository coverage. System latency remained within interactive thresholds suitable for real-time educational use. These findings support the technical viability, semantic safety, and acceptability of automated multimodal scaffolding to improve accessibility for neurodiverse learners.
>
---
#### [new 054] MedMT-Bench: Can LLMs Memorize and Understand Long Multi-Turn Conversations in Medical Scenarios?
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出MedMT-Bench，用于评估LLMs在医疗场景中长期对话的理解与记忆能力。旨在解决医疗AI安全性与可靠性问题，通过构建多轮对话基准进行模型测试。**

- **链接: [https://arxiv.org/pdf/2603.23519](https://arxiv.org/pdf/2603.23519)**

> **作者:** Lin Yang; Yuancheng Yang; Xu Wang; Changkun Liu; Haihua Yang
>
> **摘要:** Large Language Models (LLMs) have demonstrated impressive capabilities across various specialist domains and have been integrated into high-stakes areas such as medicine. However, as existing medical-related benchmarks rarely stress-test the long-context memory, interference robustness, and safety defense required in practice. To bridge this gap, we introduce MedMT-Bench, a challenging medical multi-turn instruction following benchmark that simulates the entire diagnosis and treatment process. We construct the benchmark via scene-by-scene data synthesis refined by manual expert editing, yielding 400 test cases that are highly consistent with real-world application scenarios. Each test case has an average of 22 rounds (maximum of 52 rounds), covering 5 types of difficult instruction following issues. For evaluation, we propose an LLM-as-judge protocol with instance-level rubrics and atomic test points, validated against expert annotations with a human-LLM agreement of 91.94\%. We test 17 frontier models, all of which underperform on MedMT-Bench (overall accuracy below 60.00\%), with the best model reaching 59.75\%. MedMT-Bench can be an essential tool for driving future research towards safer and more reliable medical AI. The benchmark is available in this https URL
>
---
#### [new 055] From AI Assistant to AI Scientist: Autonomous Discovery of LLM-RL Algorithms with LLM Agents
- **分类: cs.CL**

- **简介: 该论文属于强化学习算法优化任务，旨在解决人工设计策略优化算法成本高的问题。通过POISE框架自动发现改进的LLM-RL算法，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.23951](https://arxiv.org/pdf/2603.23951)**

> **作者:** Sirui Xia; Yikai Zhang; Aili Chen; Siye Wu; Siyu Yuan; Yanghua Xiao
>
> **摘要:** Discovering improved policy optimization algorithms for language models remains a costly manual process requiring repeated mechanism-level modification and validation. Unlike simple combinatorial code search, this problem requires searching over algorithmic mechanisms tightly coupled with training dynamics while reusing empirical evidence across iterations. We propose POISE, a closed-loop framework for automated discovery of policy optimization algorithms for language models. POISE maintains a structured, genealogically linked archive linking proposals, executable implementations, standardized evaluations, and natural-language reflections to support evidence-driven iteration. In mathematical reasoning experiments starting from GRPO, POISE evaluates 64 candidate algorithms and discovers improved mechanisms, including analytic-variance scaling and validity masking. The best variant improves weighted Overall from 47.8 to 52.5 (+4.6) and increases AIME25 pass@32 from 26.7% to 43.3%, demonstrating the feasibility of automated policy optimization discovery while supporting interpretable design principles.
>
---
#### [new 056] Language Model Planners do not Scale, but do Formalizers?
- **分类: cs.CL**

- **简介: 论文研究LLM在规划任务中的表现，探讨其是否能有效转化为形式化程序。任务属于AI规划与形式化验证，解决LLM在复杂问题中的局限性，提出分治策略和高阶形式化方法提升性能。**

- **链接: [https://arxiv.org/pdf/2603.23844](https://arxiv.org/pdf/2603.23844)**

> **作者:** Owen Jiang; Cassie Huang; Ashish Sabharwal; Li Zhang
>
> **摘要:** Recent work shows overwhelming evidence that LLMs, even those trained to scale their reasoning trace, perform unsatisfactorily when solving planning problems too complex. Whether the same conclusion holds for LLM formalizers that generate solver-oriented programs remains unknown. We systematically show that LLM formalizers greatly out-scale LLM planners, some retaining perfect accuracy in the classic BlocksWorld domain with a huge state space of size up to $10^{165}$. While performance of smaller LLM formalizers degrades with problem complexity, we show that a divide-and-conquer formalizing technique can greatly improve its robustness. Finally, we introduce unraveling problems where one line of problem description realistically corresponds to exponentially many lines of formal language such as the Planning Domain Definition Language (PDDL), greatly challenging LLM formalizers. We tackle this challenge by introducing a new paradigm, namely LLM-as-higher-order-formalizer, where an LLM generates a program generator. This decouples token output from the combinatorial explosion of the underlying formalization and search space.
>
---
#### [new 057] Variation is the Norm: Embracing Sociolinguistics in NLP
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决语言变体被当作噪声的问题。通过引入社会语言学视角，研究变体对NLP模型的影响，并提出改进方法。**

- **链接: [https://arxiv.org/pdf/2603.24222](https://arxiv.org/pdf/2603.24222)**

> **作者:** Anne-Marie Lutgen; Alistair Plum; Verena Blaschke; Barbara Plank; Christoph Purschke
>
> **备注:** Accepted at LREC 2026
>
> **摘要:** In Natural Language Processing (NLP), variation is typically seen as noise and "normalised away" before processing, even though it is an integral part of language. Conversely, studying language variation in social contexts is central to sociolinguistics. We present a framework to combine the sociolinguistic dimension of language with the technical dimension of NLP. We argue that by embracing sociolinguistics, variation can actively be included in a research setup, in turn informing the NLP side. To illustrate this, we provide a case study on Luxembourgish, an evolving language featuring a large amount of orthographic variation, demonstrating how NLP performance is impacted. The results show large discrepancies in the performance of models tested and fine-tuned on data with a large amount of orthographic variation in comparison to data closer to the (orthographic) standard. Furthermore, we provide a possible solution to improve the performance by including variation in the fine-tuning process. This case study highlights the importance of including variation in the research setup, as models are currently not robust to occurring variation. Our framework facilitates the inclusion of variation in the thought-process while also being grounded in the theoretical framework of sociolinguistics.
>
---
#### [new 058] Cluster-R1: Large Reasoning Models Are Instruction-following Clustering Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本聚类任务，旨在解决传统嵌入模型无法准确遵循用户指令和自主发现数据结构的问题。工作包括提出一种基于大推理模型的聚类方法，并构建了评估基准。**

- **链接: [https://arxiv.org/pdf/2603.23518](https://arxiv.org/pdf/2603.23518)**

> **作者:** Peijun Qing; Puneet Mathur; Nedim Lipka; Varun Manjunatha; Ryan Rossi; Franck Dernoncourt; Saeed Hassanpour; Soroush Vosoughi
>
> **摘要:** General-purpose embedding models excel at recognizing semantic similarities but fail to capture the characteristics of texts specified by user instructions. In contrast, instruction-tuned embedders can align embeddings with textual instructions yet cannot autonomously infer latent corpus structures, such as determining the optimal number of clusters. To address both limitations, we reframe instruction-following clustering as a generative task and train large reasoning models (LRMs) as autonomous clustering agents. Our reasoning-driven training pipeline enables LRMs to interpret high-level clustering instructions and then infer the corresponding latent groupings. To evaluate this paradigm, we introduce ReasonCluster, a comprehensive benchmark comprising 28 diverse tasks spanning daily dialogue, legal cases, and financial reports. Experiments across diverse datasets and clustering scenarios show that our approach consistently outperforms strong embedding-based methods and LRM baselines, demonstrating that explicit reasoning fosters more faithful and interpretable instruction-based clustering.
>
---
#### [new 059] Qworld: Question-Specific Evaluation Criteria for LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的模型评估任务，旨在解决开放问题评价标准不精准的问题。通过生成特定于问题的评价标准，提升评估的全面性和准确性。**

- **链接: [https://arxiv.org/pdf/2603.23522](https://arxiv.org/pdf/2603.23522)**

> **作者:** Shanghua Gao; Yuchang Su; Pengwei Sui; Curtis Ginder; Marinka Zitnik
>
> **摘要:** Evaluating large language models (LLMs) on open-ended questions is difficult because response quality depends on the question's context. Binary scores and static rubrics fail to capture these context-dependent requirements. Existing methods define criteria at the dataset level or generate them in a single pass, which limits their ability to explore the evaluation space implied by each question. We introduce One-Question-One-World (Qworld), a method that generates question-specific evaluation criteria using a recursive expansion tree. Given a question, Qworld decomposes it into scenarios, perspectives, and fine-grained binary criteria through structured hierarchical and horizontal expansion. The resulting criteria specify what a high-quality answer must address for that question. On HealthBench, Qworld covers 89% of expert-authored criteria and generates 79% novel criteria validated by human experts. Experts rate Qworld criteria higher in insight and granularity than those produced by prior methods. When applied to 11 frontier LLMs on HealthBench and Humanity's Last Exam, Qworld reveals capability differences in dimensions such as long-term impact, equity, error handling, and interdisciplinary reasoning that coarse rubrics do not distinguish. By formulating criteria generation as structured coverage of question-implied evaluation axes, Qworld enables evaluation that adapts to each question rather than relying on fixed task-level criteria.
>
---
#### [new 060] IslamicMMLU: A Benchmark for Evaluating LLMs on Islamic Knowledge
- **分类: cs.CL**

- **简介: 该论文属于评估任务，旨在解决大语言模型在伊斯兰知识领域的表现评价问题。研究构建了IslamicMMLU基准，包含10,013道多选题，涵盖《古兰经》、圣训和教法三个领域，用于评估模型的伊斯兰知识能力。**

- **链接: [https://arxiv.org/pdf/2603.23750](https://arxiv.org/pdf/2603.23750)**

> **作者:** Ali Abdelaal; Mohammed Nader Al Haffar; Mahmoud Fawzi; Walid Magdy
>
> **备注:** Leaderboard link: this https URL
>
> **摘要:** Large language models are increasingly consulted for Islamic knowledge, yet no comprehensive benchmark evaluates their performance across core Islamic disciplines. We introduce IslamicMMLU, a benchmark of 10,013 multiple-choice questions spanning three tracks: Quran (2,013 questions), Hadith (4,000 questions), and Fiqh (jurisprudence, 4,000 questions). Each track is formed of multiple types of questions to examine LLMs capabilities handling different aspects of Islamic knowledge. The benchmark is used to create the IslamicMMLU public leaderboard for evaluating LLMs, and we initially evaluate 26 LLMs, where their averaged accuracy across the three tracks varied between 39.8\% to 93.8\% (by Gemini 3 Flash). The Quran track shows the widest span (99.3\% to 32.4\%), while the Fiqh track includes a novel madhab (Islamic school of jurisprudence) bias detection task revealing variable school-of-thought preferences across models. Arabic-specific models show mixed results, but they all underperform compared to frontier models. The evaluation code and leaderboard are made publicly available.
>
---
#### [new 061] LLMpedia: A Transparent Framework to Materialize an LLM's Encyclopedic Knowledge at Scale
- **分类: cs.CL; cs.DB**

- **简介: 该论文提出LLMpedia，一个从语言模型参数记忆生成百科文章的框架，解决知识真实性评估问题，通过无检索生成1M篇文章验证事实准确性。**

- **链接: [https://arxiv.org/pdf/2603.24080](https://arxiv.org/pdf/2603.24080)**

> **作者:** Muhammed Saeed; Simon Razniewski
>
> **摘要:** Benchmarks such as MMLU suggest flagship language models approach factuality saturation, with scores above 90\%. We show this picture is incomplete. \emph{LLMpedia} generates encyclopedic articles entirely from parametric memory, producing ${\sim}$1M articles across three model families without retrieval. For gpt-5-mini, the verifiable true rate on Wikipedia-covered subjects is only 74.7\% -- more than 15 percentage points below the benchmark-based picture, consistent with the availability bias of fixed-question evaluation. Beyond Wikipedia, frontier subjects verifiable only through curated web evidence fall further to 63.2\% true rate. Wikipedia covers just 61\% of surfaced subjects, and three model families overlap by only 7.3\% in subject choice. In a capture-trap benchmark inspired by prior analysis of Grokipedia, LLMpedia achieves substantially higher factuality at roughly half the textual similarity to Wikipedia. Unlike Grokipedia, every prompt, artifact, and evaluation verdict is publicly released, making LLMpedia the first fully open parametric encyclopedia -- bridging factuality evaluation and knowledge materialization. All data, code, and a browsable interface are at this https URL.
>
---
#### [new 062] MDKeyChunker: Single-Call LLM Enrichment with Rolling Keys and Key-Based Restructuring for High-Accuracy RAG
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **简介: 该论文提出MDKeyChunker，解决Markdown文档的高效RAG问题。通过结构感知分块、单次LLM调用提取元数据及语义键重组，提升检索精度。**

- **链接: [https://arxiv.org/pdf/2603.23533](https://arxiv.org/pdf/2603.23533)**

> **作者:** Bhavik Mangla
>
> **备注:** 13 pages, 4 figures, 7 tables, 2 algorithms. Code: this https URL
>
> **摘要:** RAG pipelines typically rely on fixed-size chunking, which ignores document structure, fragments semantic units across boundaries, and requires multiple LLM calls per chunk for metadata extraction. We present MDKeyChunker, a three-stage pipeline for Markdown documents that (1) performs structure-aware chunking treating headers, code blocks, tables, and lists as atomic units; (2) enriches each chunk via a single LLM call extracting title, summary, keywords, typed entities, hypothetical questions, and a semantic key, while propagating a rolling key dictionary to maintain document-level context; and (3) restructures chunks by merging those sharing the same semantic key via bin-packing, co-locating related content for retrieval. The single-call design extracts all seven metadata fields in one LLM invocation, eliminating the need for separate per-field extraction passes. Rolling key propagation replaces hand-tuned scoring with LLM-native semantic matching. An empirical evaluation on 30 queries over an 18-document Markdown corpus shows Config D (BM25 over structural chunks) achieves Recall@5=1.000 and MRR=0.911, while dense retrieval over the full pipeline (Config C) reaches Recall@5=0.867. MDKeyChunker is implemented in Python with four dependencies and supports any OpenAI-compatible endpoint.
>
---
#### [new 063] Beyond Masks: Efficient, Flexible Diffusion Language Models via Deletion-Insertion Processes
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于语言模型任务，旨在解决MDLMs计算效率低和生成灵活性差的问题。提出DID模型，通过删除-插入过程替代掩码机制，提升效率与灵活性。**

- **链接: [https://arxiv.org/pdf/2603.23507](https://arxiv.org/pdf/2603.23507)**

> **作者:** Fangyu Ding; Ding Ding; Sijin Chen; Kaibo Wang; Peng Xu; Zijin Feng; Haoli Bai; Kai Han; Youliang Yan; Binhang Yuan; Jiacheng Sun
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** While Masked Diffusion Language Models (MDLMs) relying on token masking and unmasking have shown promise in language modeling, their computational efficiency and generation flexibility remain constrained by the masking paradigm. In this paper, we propose Deletion-Insertion Diffusion language models (DID) that rigorously formulate token deletion and insertion as discrete diffusion processes, replacing the masking and unmasking processes in current MDLMs. DID improves training and inference efficiency by eliminating two major sources of computational overhead in MDLMs: the computations on non-informative 1) <MASK> tokens inherent to the paradigm, and 2) <PAD> tokens introduced in variable-length settings. Furthermore, DID offers greater flexibility by: 1) natively supporting variable-length sequences without requiring fixed-length padding, and 2) an intrinsic self-correction mechanism during generation due to insertion that dynamically adjusts token positions. To train DID, we design a score-based approach that assigns scores to token insertion operations and derive appropriate training objectives. The objectives involve subsequence counting problems, which we efficiently solve via a parallelized dynamic programming algorithm. Our experiments across fixed and variable-length settings demonstrate the advantage of DID over baselines of MDLMs and existing insertion-based LMs, in terms of modeling performance, sampling quality, and training/inference speed, without any hyperparameter tuning.
>
---
#### [new 064] Konkani LLM: Multi-Script Instruction Tuning and Evaluation for a Low-Resource Indian Language
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于低资源语言处理任务，旨在解决Konkani语言模型性能不足的问题。通过构建合成数据集和优化模型，提升其在多文字体系下的表现。**

- **链接: [https://arxiv.org/pdf/2603.23529](https://arxiv.org/pdf/2603.23529)**

> **作者:** Reuben Chagas Fernandes; Gaurang S. Patkar
>
> **摘要:** Large Language Models (LLMs) consistently under perform in low-resource linguistic contexts such as Konkani. This performance deficit stems from acute training data scarcity compounded by high script diversity across Devanagari, Romi and Kannada orthographies. To address this gap, we introduce Konkani-Instruct-100k, a comprehensive synthetic instruction-tuning dataset generated through Gemini 3. We establish rigorous baseline benchmarks by evaluating leading open-weights architectures including Llama 3.1, Qwen2.5 and Gemma 3 alongside proprietary closed-source models. Our primary contribution involves the development of Konkani LLM, a series of fine-tuned models optimized for regional nuances. Furthermore, we are developing the Multi-Script Konkani Benchmark to facilitate cross-script linguistic evaluation. In machine translation, Konkani LLM delivers consistent gains over the corresponding base models and is competitive with and in several settings surpasses proprietary baselines
>
---
#### [new 065] PoliticsBench: Benchmarking Political Values in Large Language Models with Multi-Turn Roleplay
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的政治偏见评估任务，旨在检测大语言模型的政治倾向。通过多轮角色扮演测试，分析模型在十项政治价值上的表现，揭示其潜在的左倾倾向。**

- **链接: [https://arxiv.org/pdf/2603.23841](https://arxiv.org/pdf/2603.23841)**

> **作者:** Rohan Khetan; Ashna Khetan
>
> **备注:** 13 pages, 8 tables, 3 figures
>
> **摘要:** While Large Language Models (LLMs) are increasingly used as primary sources of information, their potential for political bias may impact their objectivity. Existing benchmarks of LLM social bias primarily evaluate gender and racial stereotypes. When political bias is included, it is typically measured at a coarse level, neglecting the specific values that shape sociopolitical leanings. This study investigates political bias in eight prominent LLMs (Claude, Deepseek, Gemini, GPT, Grok, Llama, Qwen Base, Qwen Instruction-Tuned) using PoliticsBench: a novel multi-turn roleplay framework adapted from the EQ-Bench-v3 psychometric benchmark. We test whether commercially developed LLMs display a systematic left-leaning bias that becomes more pronounced in later stages of multi-stage roleplay. Through twenty evolving scenarios, each model reported its stance and determined its course of action. Scoring these responses on a scale of ten political values, we explored the values underlying chatbots' deviations from unbiased standards. Seven of our eight models leaned left, while Grok leaned right. Each left-leaning LLM strongly exhibited liberal traits and moderately exhibited conservative ones. We discovered slight variations in alignment scores across stages of roleplay, with no particular pattern. Though most models used consequence-based reasoning, Grok frequently argued with facts and statistics. Our study presents the first psychometric evaluation of political values in LLMs through multi-stage, free-text interactions.
>
---
#### [new 066] Grounding Arabic LLMs in the Doha Historical Dictionary: Retrieval-Augmented Understanding of Quran and Hadith
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于自然语言处理任务，旨在解决阿拉伯语历史宗教文本理解难题。通过引入多哈历史词典的检索增强生成框架，提升大语言模型对古兰经和圣训的理解准确率。**

- **链接: [https://arxiv.org/pdf/2603.23972](https://arxiv.org/pdf/2603.23972)**

> **作者:** Somaya Eltanbouly; Samer Rashwani
>
> **摘要:** Large language models (LLMs) have achieved remarkable progress in many language tasks, yet they continue to struggle with complex historical and religious Arabic texts such as the Quran and Hadith. To address this limitation, we develop a retrieval-augmented generation (RAG) framework grounded in diachronic lexicographic knowledge. Unlike prior RAG systems that rely on general-purpose corpora, our approach retrieves evidence from the Doha Historical Dictionary of Arabic (DHDA), a large-scale resource documenting the historical development of Arabic vocabulary. The proposed pipeline combines hybrid retrieval with an intent-based routing mechanism to provide LLMs with precise, contextually relevant historical information. Our experiments show that this approach improves the accuracy of Arabic-native LLMs, including Fanar and ALLaM, to over 85\%, substantially reducing the performance gap with Gemini, a proprietary large-scale model. Gemini also serves as an LLM-as-a-judge system for automatic evaluation in our experiments. The automated judgments were verified through human evaluation, demonstrating high agreement (kappa = 0.87). An error analysis further highlights key linguistic challenges, including diacritics and compound expressions. These findings demonstrate the value of integrating diachronic lexicographic resources into retrieval-augmented generation frameworks to enhance Arabic language understanding, particularly for historical and religious texts. The code and resources are publicly available at: this https URL.
>
---
#### [new 067] Alignment Reduces Expressed but Not Encoded Gender Bias: A Unified Framework and Study
- **分类: cs.CL**

- **简介: 该论文研究语言模型中的性别偏见问题，旨在分析对齐是否能减少模型内部表示中的偏见。通过统一框架，发现对齐虽降低输出偏见，但内部表示仍存偏见。**

- **链接: [https://arxiv.org/pdf/2603.24125](https://arxiv.org/pdf/2603.24125)**

> **作者:** Nour Bouchouchi; Thiabult Laugel; Xavier Renard; Christophe Marsala; Marie-Jeanne Lesot; Marcin Detyniecki
>
> **摘要:** During training, Large Language Models (LLMs) learn social regularities that can lead to gender bias in downstream applications. Most mitigation efforts focus on reducing bias in generated outputs, typically evaluated on structured benchmarks, which raises two concerns: output-level evaluation does not reveal whether alignment modifies the model's underlying representations, and structured benchmarks may not reflect realistic usage scenarios. We propose a unified framework to jointly analyze intrinsic and extrinsic gender bias in LLMs using identical neutral prompts, enabling direct comparison between gender-related information encoded in internal representations and bias expressed in generated outputs. Contrary to prior work reporting weak or inconsistent correlations, we find a consistent association between latent gender information and expressed bias when measured under the unified protocol. We further examine the effect of alignment through supervised fine-tuning aimed at reducing gender bias. Our results suggest that while the latter indeed reduces expressed bias, measurable gender-related associations are still present in internal representations, and can be reactivated under adversarial prompting. Finally, we consider two realistic settings and show that debiasing effects observed on structured benchmarks do not necessarily generalize, e.g., to the case of story generation.
>
---
#### [new 068] Not All Pretraining are Created Equal: Threshold Tuning and Class Weighting for Imbalanced Polarization Tasks in Low-Resource Settings
- **分类: cs.CL; cs.LG**

- **简介: 该论文针对低资源环境下的情感极化检测任务，解决类别不平衡问题。采用Transformer模型和类权重损失函数，提升多语言分类性能。**

- **链接: [https://arxiv.org/pdf/2603.23534](https://arxiv.org/pdf/2603.23534)**

> **作者:** Abass Oguntade
>
> **摘要:** This paper describes my submission to the Polarization Shared Task at SemEval-2025, which addresses polarization detection and classification in social media text. I develop Transformer-based systems for English and Swahili across three subtasks: binary polarization detection, multi-label target type classification, and multi-label manifestation identification. The approach leverages multilingual and African language-specialized models (mDeBERTa-v3-base, SwahBERT, AfriBERTa-large), class-weighted loss functions, iterative stratified data splitting, and per-label threshold tuning to handle severe class imbalance. The best configuration, mDeBERTa-v3-base, achieves 0.8032 macro-F1 on validation for binary detection, with competitive performance on multi-label tasks (up to 0.556 macro-F1). Error analysis reveals persistent challenges with implicit polarization, code-switching, and distinguishing heated political discourse from genuine polarization.
>
---
#### [new 069] Chitrakshara: A Large Multilingual Multimodal Dataset for Indian languages
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于多模态任务，旨在解决印度语言在视觉-语言模型中的代表性不足问题。构建了包含11种印度语言的大型多模态数据集Chitrakshara，涵盖大量图像和文本数据。**

- **链接: [https://arxiv.org/pdf/2603.23521](https://arxiv.org/pdf/2603.23521)**

> **作者:** Shaharukh Khan; Ali Faraz; Abhinav Ravi; Mohd Nauman; Mohd Sarfraz; Akshat Patidar; Raja Kolla; Chandra Khatri; Shubham Agarwal
>
> **备注:** Accepted at "CVPR 2025: Workshop Vision Language Models For All"
>
> **摘要:** Multimodal research has predominantly focused on single-image reasoning, with limited exploration of multi-image scenarios. Recent models have sought to enhance multi-image understanding through large-scale pretraining on interleaved image-text datasets. However, most Vision-Language Models (VLMs) are trained primarily on English datasets, leading to inadequate representation of Indian languages. To address this gap, we introduce the Chitrakshara dataset series, covering 11 Indian languages sourced from Common Crawl. It comprises (1) Chitrakshara-IL, a large-scale interleaved pretraining dataset with 193M images, 30B text tokens, and 50M multilingual documents, and (2) Chitrakshara-Cap, which includes 44M image-text pairs with 733M tokens. This paper details the data collection pipeline, including curation, filtering, and processing methodologies. Additionally, we present a comprehensive quality and diversity analysis to assess the dataset's representativeness across Indic languages and its potential for developing more culturally inclusive VLMs.
>
---
#### [new 070] The Price Reversal Phenomenon: When Cheaper Reasoning Models End Up Costing More
- **分类: cs.CL; cs.AI; cs.GT; cs.LG; cs.MA**

- **简介: 该论文研究API定价与实际推理成本的不匹配问题，属于模型评估任务。通过实验发现低价模型可能实际更贵，分析其根源为思考token差异，并指出成本预测困难。**

- **链接: [https://arxiv.org/pdf/2603.23971](https://arxiv.org/pdf/2603.23971)**

> **作者:** Lingjiao Chen; Chi Zhang; Yeye He; Ion Stoica; Matei Zaharia; James Zou
>
> **摘要:** Developers and consumers increasingly choose reasoning language models (RLMs) based on their listed API prices. However, how accurately do these prices reflect actual inference costs? We conduct the first systematic study of this question, evaluating 8 frontier RLMs across 9 diverse tasks covering competition math, science QA, code generation, and multi-domain reasoning. We uncover the pricing reversal phenomenon: in 21.8% of model-pair comparisons, the model with a lower listed price actually incurs a higher total cost, with reversal magnitude reaching up to 28x. For example, Gemini 3 Flash's listed price is 78% cheaper than GPT-5.2's, yet its actual cost across all tasks is 22% higher. We trace the root cause to vast heterogeneity in thinking token consumption: on the same query, one model may use 900% more thinking tokens than another. In fact, removing thinking token costs reduces ranking reversals by 70% and raises the rank correlation (Kendall's $\tau$ ) between price and cost rankings from 0.563 to 0.873. We further show that per-query cost prediction is fundamentally difficult: repeated runs of the same query yield thinking token variation up to 9.7x, establishing an irreducible noise floor for any predictor. Our findings demonstrate that listed API pricing is an unreliable proxy for actual cost, calling for cost-aware model selection and transparent per-request cost monitoring.
>
---
#### [new 071] Infrequent Child-Directed Speech Is Bursty and May Draw Infant Vocalizations
- **分类: cs.CL**

- **简介: 该论文属于语言发展研究任务，探讨少频儿童导向言语对婴儿语音的影响。通过分析玻利维亚和美国的音频数据，发现儿童导向言语虽少但集中，且婴儿在该时段更易发声，提示言语的时空分布与来源对其发展重要。**

- **链接: [https://arxiv.org/pdf/2603.23797](https://arxiv.org/pdf/2603.23797)**

> **作者:** Margaret Cychosz; Adriana Weisleder
>
> **摘要:** Children in many parts of the world hear relatively little speech directed to them, yet still reach major language development milestones. What differs about the speech input that infants learn from when directed input is rare? Using longform, infant-centered audio recordings taken in rural Bolivia and the urban U.S., we examined temporal patterns of infants' speech input and their pre-linguistic vocal behavior. We find that child-directed speech in Bolivia, though less frequent, was just as temporally clustered as speech input in the U.S, arriving in concentrated bursts rather than spread across the day. In both communities, infants were most likely to produce speech-like vocalizations during periods of speech directed to them, with the probability of infants' speech-like vocalizations during target child-directed speech nearly double that during silence. In Bolivia, infants' speech-like vocalizations were also more likely to occur during bouts of directed speech from older children than from adults. Together, these findings suggest that the developmental impact of child-directed speech may depend not only on quantity, but on temporal concentration and source, with older children serving as an important source of input in some communities, including where adult speech to infants is less frequent.
>
---
#### [new 072] Samasāmayik: A Parallel Dataset for Hindi-Sanskrit Machine Translation
- **分类: cs.CL**

- **简介: 该论文提出Samasāmayik数据集，用于解决当代印地语-梵语机器翻译任务。通过构建大规模平行语料库并测试多个模型，验证其有效性，为低资源语言翻译提供新基准。**

- **链接: [https://arxiv.org/pdf/2603.24307](https://arxiv.org/pdf/2603.24307)**

> **作者:** N J Karthika; Keerthana Suryanarayanan; Jahanvi Purohit; Ganesh Ramakrishnan; Jitin Singla; Anil Kumar Gourishetty
>
> **摘要:** We release Samasāmayik, a novel, meticulously curated, large-scale Hindi-Sanskrit corpus, comprising 92,196 parallel sentences. Unlike most data available in Sanskrit, which focuses on classical era text and poetry, this corpus aggregates data from diverse sources covering contemporary materials, including spoken tutorials, children's magazines, radio conversations, and instruction materials. We benchmark this new dataset by fine-tuning three complementary models - ByT5, NLLB and IndicTrans-v2, to demonstrate its utility. Our experiments demonstrate that models trained on the Samasamayik corpus achieve significant performance gains on in-domain test data, while achieving comparable performance on other widely used test sets, establishing a strong new performance baseline for contemporary Hindi-Sanskrit translation. Furthermore, a comparative analysis against existing corpora reveals minimal semantic and lexical overlap, confirming the novelty and non-redundancy of our dataset as a robust new resource for low-resource Indic language MT.
>
---
#### [new 073] Improving Lean4 Autoformalization via Cycle Consistency Fine-tuning
- **分类: cs.CL**

- **简介: 该论文属于自然语言到形式化语言的自动翻译任务，旨在提升Lean4的autoformalization效果。通过循环一致性强化学习优化模型，显著提高翻译质量。**

- **链接: [https://arxiv.org/pdf/2603.24372](https://arxiv.org/pdf/2603.24372)**

> **作者:** Arsen Shebzukhov
>
> **备注:** 10 pages, 10 figures, pages 10-27 appendix
>
> **摘要:** Autoformalization - automatically translating natural language mathematical texts into formal proof language such as Lean4 - can help accelerate AI-assisted mathematical research, be it via proof verification or proof search. I fine-tune Qwen3.5-2B with LoRA for natural language to Lean4 formalization on FineLeanCorpus and consider three training regimes: supervised fine-tuning (SFT) with curriculum learning (difficulty 1 to 10), SFT without curriculum ordering, and reinforcement learning using group relative policy optimization (GRPO) with a cycle consistency reward. Cycle consistency measures how well the meaning of a statement is preserved through a NL to Lean4 to NL' loop, computed as cosine similarity of off-the-shelf sentence embeddings. On an unseen subset of FineLeanCorpus (FLC) and on PutnamBench, RL substantially outperforms both SFT variants (mean cycle consistency 0.669 vs. 0.513 on FLC; 0.561 vs. 0.422 on PutnamBench), while increasing cross-entropy loss by only 0.011 nats, with minimal impact on formalization quality. Curriculum ordering provides no measurable benefit over shuffled training.
>
---
#### [new 074] Visuospatial Perspective Taking in Multimodal Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究多模态语言模型的视觉空间视角转换能力，旨在评估其在协作场景中的视角理解。通过两个任务测试模型在不同视角下的推理能力，发现其在高级视角转换上存在明显不足。**

- **链接: [https://arxiv.org/pdf/2603.23510](https://arxiv.org/pdf/2603.23510)**

> **作者:** Jonathan Prunty; Seraphina Zhang; Patrick Quinn; Jianxun Lian; Xing Xie; Lucy Cheke
>
> **摘要:** As multimodal language models (MLMs) are increasingly used in social and collaborative settings, it is crucial to evaluate their perspective-taking abilities. Existing benchmarks largely rely on text-based vignettes or static scene understanding, leaving visuospatial perspective-taking (VPT) underexplored. We adapt two evaluation tasks from human studies: the Director Task, assessing VPT in a referential communication paradigm, and the Rotating Figure Task, probing perspective-taking across angular disparities. Across tasks, MLMs show pronounced deficits in Level 2 VPT, which requires inhibiting one's own perspective to adopt another's. These results expose critical limitations in current MLMs' ability to represent and reason about alternative perspectives, with implications for their use in collaborative contexts.
>
---
#### [new 075] Semantic Alignment across Ancient Egyptian Language Stages via Normalization-Aware Multitask Learning
- **分类: cs.CL**

- **简介: 该论文研究古埃及语不同阶段的词级语义对齐任务，解决因书写和正字法差异导致的平行数据稀缺问题。通过多任务学习和规范化方法提升对齐效果。**

- **链接: [https://arxiv.org/pdf/2603.24258](https://arxiv.org/pdf/2603.24258)**

> **作者:** He Huang
>
> **备注:** Accepted to LREC 2026
>
> **摘要:** We study word-level semantic alignment across four historical stages of Ancient Egyptian. These stages differ in script and orthography, and parallel data are scarce. We jointly train a compact encoder-decoder model with a shared byte-level tokenizer on all four stages, combining masked language modeling (MLM), translation language modeling (TLM), sequence-to-sequence translation, and part-of-speech tagging under a task-aware loss with fixed weights and uncertainty-based scaling. To reduce surface divergence we add Latin transliteration and IPA reconstruction as auxiliary views. We integrate these views through KL-based consistency and through embedding-level fusion. We evaluate alignment quality using pairwise metrics, specifically ROC-AUC and triplet accuracy, on curated Egyptian-English and intra-Egyptian cognate datasets. Translation yields the strongest gains. IPA with KL consistency improves cross-branch alignment, while early fusion demonstrates limited efficacy. Although the overall alignment remains limited, the findings provide a reproducible baseline and practical guidance for modeling historical languages under real constraints. They also show how normalization and task design shape what counts as alignment in typologically distant settings.
>
---
#### [new 076] Optimizing Multilingual LLMs via Federated Learning: A Study of Client Language Composition
- **分类: cs.CL**

- **简介: 该论文属于多语言大模型的联邦学习任务，旨在解决多语言分布不均和资源差异问题。通过改进框架和引入新机制，研究客户端语言组成对模型性能的影响。**

- **链接: [https://arxiv.org/pdf/2603.24242](https://arxiv.org/pdf/2603.24242)**

> **作者:** Aleix Sant; Jordi Luque; Carlos Escolano
>
> **备注:** 12 pages, 4 figures, 5 tables
>
> **摘要:** Federated Learning (FL) of Large Language Models (LLMs) in multilingual environments presents significant challenges stemming from heterogeneous language distributions across clients and disparities in language resource availability. To address these challenges, we extended the FederatedScope-LLM framework to support multilingual instruction-tuning experiments with LLMs. We also introduced a novel client-specific early stopping mechanism, Local Dynamic Early Stopping (LDES-FL), which allows clients to pause and resume local training based on client-side validation performance, enhancing training efficiency and sustainability. Through a series of experiments, we studied how client language composition - from fully monolingual to increasingly multilingual clients - affects multilingual quality, fairness and training cost. Monolingual local fine-tuning remains the most effective for single-language specialization, whereas federated training is better suited to learning a single balanced multilingual model. In FL, increasing within-client multilinguality leads to stronger and fairer global models, narrows the gap to centralized multilingual fine-tuning, and yields the largest gains for lower-resource languages, albeit at the cost of more optimization steps. Overall, our results identify client language composition as a key design variable in multilingual FL, shaping performance, fairness and efficiency
>
---
#### [new 077] OmniACBench: A Benchmark for Evaluating Context-Grounded Acoustic Control in Omni-Modal Models
- **分类: cs.CL**

- **简介: 该论文提出OmniACBench，用于评估多模态模型的语音控制能力。解决模型能否根据多模态上下文生成合适语音的问题，通过实验揭示其整合多模态信息的不足。**

- **链接: [https://arxiv.org/pdf/2603.23938](https://arxiv.org/pdf/2603.23938)**

> **作者:** Seunghee Kim; Bumkyu Park; Kyudan Jung; Joosung Lee; Soyoon Kim; Jeonghoon Kim; Taeuk Kim; Hwiyeol Jo
>
> **摘要:** Most testbeds for omni-modal models assess multimodal understanding via textual outputs, leaving it unclear whether these models can properly speak their answers. To study this, we introduce OmniACBench, a benchmark for evaluating context-grounded acoustic control in omni-modal models. Given a spoken instruction, a text script, and an image, a model must read the script aloud with an appropriate tone and manner. OmniACBench comprises 3,559 verified instances covering six acoustic features: speech rate, phonation, pronunciation, emotion, global accent, and timbre. Extensive experiments on eight models reveal their limitations in the proposed setting, despite their strong performance on prior textual-output evaluations. Our analyses show that the main bottleneck lies not in processing individual modalities, but in integrating multimodal context for faithful speech generation. Moreover, we identify three common failure modes-weak direct control, failed implicit inference, and failed multimodal grounding-providing insights for developing models that can verbalize responses effectively.
>
---
#### [new 078] A Theory of LLM Information Susceptibility
- **分类: cs.LG; cond-mat.stat-mech; cs.AI; cs.CL; nlin.AO**

- **简介: 该论文属于人工智能理论研究，旨在解决LLM在智能系统中干预效果的边界问题。提出LLM信息敏感性理论，分析其对策略优化的影响，并验证不同架构下的性能表现。**

- **链接: [https://arxiv.org/pdf/2603.23626](https://arxiv.org/pdf/2603.23626)**

> **作者:** Zhuo-Yang Song; Hua Xing Zhu
>
> **备注:** 16 pages, 9 figures
>
> **摘要:** Large language models (LLMs) are increasingly deployed as optimization modules in agentic systems, yet the fundamental limits of such LLM-mediated improvement remain poorly understood. Here we propose a theory of LLM information susceptibility, centred on the hypothesis that when computational resources are sufficiently large, the intervention of a fixed LLM does not increase the performance susceptibility of a strategy set with respect to budget. We develop a multi-variable utility-function framework that generalizes this hypothesis to architectures with multiple co-varying budget channels, and discuss the conditions under which co-scaling can exceed the susceptibility bound. We validate the theory empirically across structurally diverse domains and model scales spanning an order of magnitude, and show that nested, co-scaling architectures open response channels unavailable to fixed configurations. These results clarify when LLM intervention helps and when it does not, demonstrating that tools from statistical physics can provide predictive constraints for the design of AI systems. If the susceptibility hypothesis holds generally, the theory suggests that nested architectures may be a necessary structural condition for open-ended agentic self-improvement.
>
---
#### [new 079] Comparing Developer and LLM Biases in Code Evaluation
- **分类: cs.SE; cs.CL**

- **简介: 该论文属于代码评估任务，旨在解决LLM与开发者在代码评价中的偏差问题。通过框架TRACE，分析LLM与人类在代码质量判断上的差异。**

- **链接: [https://arxiv.org/pdf/2603.24586](https://arxiv.org/pdf/2603.24586)**

> **作者:** Aditya Mittal; Ryan Shar; Zichu Wu; Shyam Agarwal; Tongshuang Wu; Chris Donahue; Ameet Talwalkar; Wayne Chi; Valerie Chen
>
> **摘要:** As LLMs are increasingly used as judges in code applications, they should be evaluated in realistic interactive settings that capture partial context and ambiguous intent. We present TRACE (Tool for Rubric Analysis in Code Evaluation), a framework that evaluates LLM judges' ability to predict human preferences and automatically extracts rubric items to reveal systematic biases in how humans and models weigh each item. Across three modalities -- chat-based programming, IDE autocompletion, and instructed code editing -- we use TRACE to measure how well LLM judges align with developer preferences. Among 13 different models, the best judges underperform human annotators by 12-23%. TRACE identifies 35 significant sources of misalignment between humans and judges across interaction modalities, the majority of which correspond to existing software engineering code quality criteria. For example, in chat-based coding, judges are biased towards longer code explanations while humans prefer shorter ones. We find significant misalignment on the majority of existing code quality dimensions, showing alignment gaps between LLM judges and human preference in realistic coding applications.
>
---
#### [new 080] MoE-Sieve: Routing-Guided LoRA for Efficient MoE Fine-Tuning
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于模型微调任务，针对MoE模型LoRA微调效率问题，提出MoE-Sieve方法，通过路由引导选择活跃专家进行微调，提升效率并减少参数量。**

- **链接: [https://arxiv.org/pdf/2603.24044](https://arxiv.org/pdf/2603.24044)**

> **作者:** Andrea Manzoni
>
> **备注:** 17 pages, 6 figures, 10 tables
>
> **摘要:** Standard LoRA fine-tuning of Mixture-of-Experts (MoE) models applies adapters to every expert, yet our profiling shows that per-layer expert routing is highly skewed: a small subset of experts handles most tokens in each layer, while many others are rarely activated ("cold"). We propose MoE-Sieve, a simple routing-guided framework for LoRA fine-tuning, and pair it with a systematic profiling study of expert routing across architectures and tasks. The method is simple: profile routing counts on a small calibration set, select the top-k most-routed experts per layer, and apply LoRA only to those experts. Across two architecturally distinct MoE models and three diverse tasks, tuning only the top 25% routed experts per layer remains competitive with full LoRA, with mean differences within +/-1 percentage point across all conditions. This reduces LoRA trainable parameters by 70-73%, adapter checkpoint size by 71-73%, and wall-clock training time by up to 50%. We also observe a non-monotonic relationship between expert count and seed-to-seed variance, consistent with the hypothesis that adapting cold experts can introduce gradient noise without improving accuracy. Further ablations show that random expert selection at matched budget is about 2.5 percentage points worse, indicating that the routing signal matters, while greedy per-layer budget optimization does not improve over uniform top-k.
>
---
#### [new 081] SpinGQE: A Generative Quantum Eigensolver for Spin Hamiltonians
- **分类: quant-ph; cs.CL**

- **简介: 该论文属于量子计算领域，解决量子哈密顿量基态搜索问题。提出SpinGQE方法，利用生成模型设计电路，有效寻找低能态，无需依赖特定结构。**

- **链接: [https://arxiv.org/pdf/2603.24298](https://arxiv.org/pdf/2603.24298)**

> **作者:** Alexander Holden; Moinul Hossain Rahat; Nii Osae Osae Dade
>
> **摘要:** The ground state search problem is central to quantum computing, with applications spanning quantum chemistry, condensed matter physics, and optimization. The Variational Quantum Eigensolver (VQE) has shown promise for small systems but faces significant limitations. These include barren plateaus, restricted ansatz expressivity, and reliance on domain-specific structure. We present SpinGQE, an extension of the Generative Quantum Eigensolver (GQE) framework to spin Hamiltonians. Our approach reframes circuit design as a generative modeling task. We employ a transformer-based decoder to learn distributions over quantum circuits that produce low-energy states. Training is guided by a weighted mean-squared error loss between model logits and circuit energies evaluated at each gate subsequence. We validate our method on the four-qubit Heisenberg model, demonstrating successfulconvergencetonear-groundstates. Throughsystematichyperparameterexploration, we identify optimal configurations: smaller model architectures (12 layers, 8 attention heads), longer sequence lengths (12 gates), and carefully chosen operator pools yield the most reliable convergence. Our results show that generative approaches can effectively navigate complex energy landscapes without relying on problem-specific symmetries or structure. This provides a scalable alternative to traditional variational methods for general quantum systems. An open-source implementation is available at this https URL.
>
---
#### [new 082] Analysing the Safety Pitfalls of Steering Vectors
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于安全审计任务，研究Steering Vectors对LLM安全性的影响。工作包括评估CAA方法在不同模型上的安全风险，揭示其与拒绝行为方向的关联。**

- **链接: [https://arxiv.org/pdf/2603.24543](https://arxiv.org/pdf/2603.24543)**

> **作者:** Yuxiao Li; Alina Fastowski; Efstratios Zaradoukas; Bardh Prenkaj; Gjergji Kasneci
>
> **摘要:** Activation steering has emerged as a powerful tool to shape LLM behavior without the need for weight updates. While its inherent brittleness and unreliability are well-documented, its safety implications remain underexplored. In this work, we present a systematic safety audit of steering vectors obtained with Contrastive Activation Addition (CAA), a widely used steering approach, under a unified evaluation protocol. Using JailbreakBench as benchmark, we show that steering vectors consistently influence the success rate of jailbreak attacks, with stronger amplification under simple template-based attacks. Across LLM families and sizes, steering the model in specific directions can drastically increase (up to 57%) or decrease (up to 50%) its attack success rate (ASR), depending on the targeted behavior. We attribute this phenomenon to the overlap between the steering vectors and the latent directions of refusal behavior. Thus, we offer a traceable explanation for this discovery. Together, our findings reveal the previously unobserved origin of this safety gap in LLMs, highlighting a trade-off between controllability and safety.
>
---
#### [new 083] ORACLE: Orchestrate NPC Daily Activities using Contrastive Learning with Transformer-CVAE
- **分类: cs.GR; cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于生成任务，旨在解决NPC活动计划生成中重复性高、缺乏真实性的问题。通过结合Transformer-CVAE与对比学习，提出ORACLE模型生成更真实的室内活动序列。**

- **链接: [https://arxiv.org/pdf/2603.23933](https://arxiv.org/pdf/2603.23933)**

> **作者:** Seong-Eun Hong; JuYeong Hwang; RyunHa Lee; HyeongYeop Kang
>
> **备注:** 17 pages, 7 figures. Accepted to CVM 2026
>
> **摘要:** The integration of Non-player characters (NPCs) within digital environments has been increasingly recognized for its potential to augment user immersion and cognitive engagement. The sophisticated orchestration of their daily activities, reflecting the nuances of human daily routines, contributes significantly to the realism of digital environments. Nevertheless, conventional approaches often produce monotonous repetition, falling short of capturing the intricacies of real human activity plans. In response to this, we introduce ORACLE, a novel generative model for the synthesis of realistic indoor daily activity plans, ensuring NPCs' authentic presence in digital habitats. Exploiting the CASAS smart home dataset's 24-hour indoor activity sequences, ORACLE addresses challenges in the dataset, including its imbalanced sequential data, the scarcity of training samples, and the absence of pre-trained models encapsulating human daily activity patterns. ORACLE's training leverages the sequential data processing prowess of Transformers, the generative controllability of Conditional Variational Autoencoders (CVAE), and the discriminative refinement of contrastive learning. Our experimental results validate the superiority of generating NPC activity plans and the efficacy of our design strategies over existing methods.
>
---
#### [new 084] The Geometric Price of Discrete Logic: Context-driven Manifold Dynamics of Number Representations
- **分类: cs.LG; cs.CL; cs.CY**

- **简介: 该论文研究大语言模型在逻辑推理中的几何特性，解决连续语义与离散逻辑的矛盾。通过分析激活向量，揭示拓扑变形机制，验证其对模型性能的影响。**

- **链接: [https://arxiv.org/pdf/2603.23577](https://arxiv.org/pdf/2603.23577)**

> **作者:** Long Zhang; Dai-jun Lin; Wei-neng Chen
>
> **摘要:** Large language models (LLMs) generalize smoothly across continuous semantic spaces, yet strict logical reasoning demands the formation of discrete decision boundaries. Prevailing theories relying on linear isometric projections fail to resolve this fundamental tension. In this work, we argue that task context operates as a non-isometric dynamical operator that enforces a necessary "topological distortion." By applying Gram-Schmidt decomposition to residual-stream activations , we reveal a dual-modulation mechanism driving this process: a class-agnostic topological preservation that anchors global structure to prevent semantic collapse, and a specific algebraic divergence that directionally tears apart cross-class concepts to forge logical boundaries. We validate this geometric evolution across a gradient of tasks, from simple mapping to complex primality testing. Crucially, targeted specific vector ablation establishes a strict causal binding between this topology and model function: algebraically erasing the divergence component collapses parity classification accuracy from 100% to chance levels (38.57%). Furthermore, we uncover a three-phase layer-wise geometric dynamic and demonstrate that under social pressure prompts, models fail to generate sufficient divergence. This results in a "manifold entanglement" that geometrically explains sycophancy and hallucination. Ultimately, our findings revise the linear-isometric presumption, demonstrating that the emergence of discrete logic in LLMs is purchased at an irreducible cost of topological deformation.
>
---
#### [new 085] Counting Without Numbers \& Finding Without Words
- **分类: cs.CV; cs.AI; cs.CL; cs.SI**

- **简介: 论文提出一种多模态重聚系统，解决宠物与主人无法通过外观匹配的问题。结合视觉与声学生物特征，提升失散动物的识别准确率。**

- **链接: [https://arxiv.org/pdf/2603.24470](https://arxiv.org/pdf/2603.24470)**

> **作者:** Badri Narayana Patro
>
> **摘要:** Every year, 10 million pets enter shelters, separated from their families. Despite desperate searches by both guardians and lost animals, 70% never reunite, not because matches do not exist, but because current systems look only at appearance, while animals recognize each other through sound. We ask, why does computer vision treat vocalizing species as silent visual objects? Drawing on five decades of cognitive science showing that animals perceive quantity approximately and communicate identity acoustically, we present the first multimodal reunification system integrating visual and acoustic biometrics. Our species-adaptive architecture processes vocalizations from 10Hz elephant rumbles to 4kHz puppy whines, paired with probabilistic visual matching that tolerates stress-induced appearance changes. This work demonstrates that AI grounded in biological communication principles can serve vulnerable populations that lack human language.
>
---
#### [new 086] Evaluating a Multi-Agent Voice-Enabled Smart Speaker for Care Homes: A Safety-Focused Framework
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于智能语音系统评估任务，旨在解决护理院中语音助手的安全与准确性问题。通过实验验证了系统的居民识别、提醒处理及日程安排能力。**

- **链接: [https://arxiv.org/pdf/2603.23625](https://arxiv.org/pdf/2603.23625)**

> **作者:** Zeinab Dehghani; Rameez Raja Kureshi; Koorosh Aslansefat; Faezeh Alsadat Abedi; Dhavalkumar Thakker; Lisa Greaves; Bhupesh Kumar Mishra; Baseer Ahmad; Tanaya Maslekar
>
> **摘要:** Artificial intelligence (AI) is increasingly being explored in health and social care to reduce administrative workload and allow staff to spend more time on patient care. This paper evaluates a voice-enabled Care Home Smart Speaker designed to support everyday activities in residential care homes, including spoken access to resident records, reminders, and scheduling tasks. A safety-focused evaluation framework is presented that examines the system end-to-end, combining Whisper-based speech recognition with retrieval-augmented generation (RAG) approaches (hybrid, sparse, and dense). Using supervised care-home trials and controlled testing, we evaluated 330 spoken transcripts across 11 care categories, including 184 reminder-containing interactions. These evaluations focus on (i) correct identification of residents and care categories, (ii) reminder recognition and extraction, and (iii) end-to-end scheduling correctness under uncertainty (including safe deferral/clarification). Given the safety-critical nature of care homes, particular attention is also paid to reliability in noisy environments and across diverse accents, supported by confidence scoring, clarification prompts, and human-in-the-loop oversight. In the best-performing configuration (GPT-5.2), resident ID and care category matching reached 100% (95% CI: 98.86-100), while reminder recognition reached 89.09\% (95% CI: 83.81-92.80) with zero missed reminders (100% recall) but some false positives. End-to-end scheduling via calendar integration achieved 84.65% exact reminder-count agreement (95% CI: 78.00-89.56), indicating remaining edge cases in converting informal spoken instructions into actionable events. The findings suggest that voice-enabled systems, when carefully evaluated and appropriately safeguarded, can support accurate documentation, effective task management, and trustworthy use of AI in care home settings.
>
---
#### [new 087] Multi-Agent Reasoning with Consistency Verification Improves Uncertainty Calibration in Medical MCQA
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于医疗多选题问答任务，旨在解决AI模型置信度不准的问题。通过多智能体框架和一致性验证提升模型校准与区分度。**

- **链接: [https://arxiv.org/pdf/2603.24481](https://arxiv.org/pdf/2603.24481)**

> **作者:** John Ray B. Martinez
>
> **备注:** 17 pages, 6 figures. Preprint under review
>
> **摘要:** Miscalibrated confidence scores are a practical obstacle to deploying AI in clinical settings. A model that is always overconfident offers no useful signal for deferral. We present a multi-agent framework that combines domain-specific specialist agents with Two-Phase Verification and S-Score Weighted Fusion to improve both calibration and discrimination in medical multiple-choice question answering. Four specialist agents (respiratory, cardiology, neurology, gastroenterology) generate independent diagnoses using Qwen2.5-7B-Instruct. Each diagnosis is then subjected to a two-phase self-verification process that measures internal consistency and produces a Specialist Confidence Score (S-score). The S-scores drive a weighted fusion strategy that selects the final answer and calibrates the reported confidence. We evaluate across four experimental settings, covering 100-question and 250-question high-disagreement subsets of both MedQA-USMLE and MedMCQA. Calibration improvement is the central finding, with ECE reduced by 49-74% across all four settings, including the harder MedMCQA benchmark where these gains persist even when absolute accuracy is constrained by knowledge-intensive recall demands. On MedQA-250, the full system achieves ECE = 0.091 (74.4% reduction over the single-specialist baseline) and AUROC = 0.630 (+0.056) at 59.2% accuracy. Ablation analysis identifies Two-Phase Verification as the primary calibration driver and multi-agent reasoning as the primary accuracy driver. These results establish that consistency-based verification produces more reliable uncertainty estimates across diverse medical question types, providing a practical confidence signal for deferral in safety-critical clinical AI applications.
>
---
#### [new 088] VehicleMemBench: An Executable Benchmark for Multi-User Long-Term Memory in In-Vehicle Agents
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出VehicleMemBench，用于评估车载智能体的多用户长期记忆能力。解决现有基准不适应多用户、动态环境的问题，通过模拟环境进行客观评估。**

- **链接: [https://arxiv.org/pdf/2603.23840](https://arxiv.org/pdf/2603.23840)**

> **作者:** Yuhao Chen; Yi Xu; Xinyun Ding; Xiang Fang; Shuochen Liu; Luxi Lin; Qingyu Zhang; Ya Li; Quan Liu; Tong Xu
>
> **摘要:** With the growing demand for intelligent in-vehicle experiences, vehicle-based agents are evolving from simple assistants to long-term companions. This evolution requires agents to continuously model multi-user preferences and make reliable decisions in the face of inter-user preference conflicts and changing habits over time. However, existing benchmarks are largely limited to single-user, static question-answer settings, failing to capture the temporal evolution of preferences and the multi-user, tool-interactive nature of real vehicle environments. To address this gap, we introduce VehicleMemBench, a multi-user long-context memory benchmark built on an executable in-vehicle simulation environment. The benchmark evaluates tool use and memory by comparing the post-action environment state with a predefined target state, enabling objective and reproducible evaluation without LLM-based or human scoring. VehicleMemBench includes 23 tool modules, and each sample contains over 80 historical memory events. Experiments show that powerful models perform well on direct instruction tasks but struggle in scenarios involving memory evolution, particularly when user preferences change dynamically. Even advanced memory systems struggle to handle domain-specific memory requirements in this environment. These findings highlight the need for more robust and specialized memory management mechanisms to support long-term adaptive decision-making in real-world in-vehicle systems. To facilitate future research, we release the data and code.
>
---
#### [new 089] PLDR-LLMs Reason At Self-Organized Criticality
- **分类: cs.AI; cs.CL; cs.LG; nlin.AO**

- **简介: 该论文研究大语言模型的推理机制，探讨其在自组织临界状态下的表现。通过分析模型参数，揭示推理能力与临界状态的关系，提出一种量化推理能力的新方法。**

- **链接: [https://arxiv.org/pdf/2603.23539](https://arxiv.org/pdf/2603.23539)**

> **作者:** Burc Gokden
>
> **摘要:** We show that PLDR-LLMs pretrained at self-organized criticality exhibit reasoning at inference time. The characteristics of PLDR-LLM deductive outputs at criticality is similar to second-order phase transitions. At criticality, the correlation length diverges, and the deductive outputs attain a metastable steady state. The steady state behaviour suggests that deductive outputs learn representations equivalent to scaling functions, universality classes and renormalization groups from the training dataset, leading to generalization and reasoning capabilities in the process. We can then define an order parameter from the global statistics of the model's deductive output parameters at inference. The reasoning capabilities of a PLDR-LLM is better when its order parameter is close to zero at criticality. This observation is supported by the benchmark scores of the models trained at near-criticality and sub-criticality. Our results provide a self-contained explanation on how reasoning manifests in large language models, and the ability to reason can be quantified solely from global model parameter values of the deductive outputs at steady state, without any need for evaluation of curated benchmark datasets through inductive output for reasoning and comprehension.
>
---
#### [new 090] LLMORPH: Automated Metamorphic Testing of Large Language Models
- **分类: cs.SE; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出LLMORPH，用于测试大语言模型的自动化工具，解决缺乏自动验证机制的问题。通过元编程测试发现模型不一致性，适用于多种NLP任务。**

- **链接: [https://arxiv.org/pdf/2603.23611](https://arxiv.org/pdf/2603.23611)**

> **作者:** Steven Cho; Stefano Ruberto; Valerio Terragni
>
> **备注:** Accepted for publication in the 40th IEEE/ACM International Conference on Automated Software Engineering (ASE 2025). This arXiv version is the authors' accepted manuscript. DOI: https://doi.org/10.1109/ASE63991.2025.00385 Code: this http URL
>
> **摘要:** Automated testing is essential for evaluating and improving the reliability of Large Language Models (LLMs), yet the lack of automated oracles for verifying output correctness remains a key challenge. We present LLMORPH, an automated testing tool specifically designed for LLMs performing NLP tasks, which leverages Metamorphic Testing (MT) to uncover faulty behaviors without relying on human-labeled data. MT uses Metamorphic Relations (MRs) to generate follow-up inputs from source test input, enabling detection of inconsistencies in model outputs without the need of expensive labelled data. LLMORPH is aimed at researchers and developers who want to evaluate the robustness of LLM-based NLP systems. In this paper, we detail the design, implementation, and practical usage of LLMORPH, demonstrating how it can be easily extended to any LLM, NLP task, and set of MRs. In our evaluation, we applied 36 MRs across four NLP benchmarks, testing three state-of-the-art LLMs: GPT-4, LLAMA3, and HERMES 2. This produced over 561,000 test executions. Results demonstrate LLMORPH's effectiveness in automatically exposing inconsistencies.
>
---
#### [new 091] How Vulnerable Are Edge LLMs?
- **分类: cs.CR; cs.CL; cs.LG**

- **简介: 该论文研究边缘部署的量化大语言模型的安全性问题，提出CLIQ框架提升查询提取效率，揭示量化不等于安全。**

- **链接: [https://arxiv.org/pdf/2603.23822](https://arxiv.org/pdf/2603.23822)**

> **作者:** Ao Ding; Hongzong Li; Zi Liang; Zhanpeng Shi; Shuxin Zhuang; Shiqin Tang; Rong Feng; Ping Lu
>
> **摘要:** Large language models (LLMs) are increasingly deployed on edge devices under strict computation and quantization constraints, yet their security implications remain unclear. We study query-based knowledge extraction from quantized edge-deployed LLMs under realistic query budgets and show that, although quantization introduces noise, it does not remove the underlying semantic knowledge, allowing substantial behavioral recovery through carefully designed queries. To systematically analyze this risk, we propose \textbf{CLIQ} (\textbf{Cl}ustered \textbf{I}nstruction \textbf{Q}uerying), a structured query construction framework that improves semantic coverage while reducing redundancy. Experiments on quantized Qwen models (INT8/INT4) demonstrate that CLIQ consistently outperforms original queries across BERTScore, BLEU, and ROUGE, enabling more efficient extraction under limited budgets. These results indicate that quantization alone does not provide effective protection against query-based extraction, highlighting a previously underexplored security risk in edge-deployed LLMs.
>
---
#### [new 092] The Alignment Tax: Response Homogenization in Aligned LLMs and Its Implications for Uncertainty Estimation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究对齐语言模型中的响应同质化问题，探讨其对不确定性估计的影响。通过实验分析不同模型和任务下的表现，提出改进的不确定性估计方法。**

- **链接: [https://arxiv.org/pdf/2603.24124](https://arxiv.org/pdf/2603.24124)**

> **作者:** Mingyi Liu
>
> **备注:** 23 pages, 3 figures, 10 tables, 22 experiments across 5 benchmarks. Code: this https URL
>
> **摘要:** RLHF-aligned language models exhibit response homogenization: on TruthfulQA (n=790), 40-79% of questions produce a single semantic cluster across 10 i.i.d. samples. On affected questions, sampling-based uncertainty methods have zero discriminative power (AUROC=0.500), while free token entropy retains signal (0.603). This alignment tax is task-dependent: on GSM8K (n=500), token entropy achieves 0.724 (Cohen's d=0.81). A base-vs-instruct ablation confirms the causal role of alignment: the base model shows 1.0% single-cluster rate vs. 28.5% for the instruct model (p < 10^{-6}). A training stage ablation (Base 0.0% -> SFT 1.5% -> DPO 4.0% SCR) localizes the cause to DPO, not SFT. Cross-family replication on four model families reveals alignment tax severity varies by family and scale. We validate across 22 experiments, 5 benchmarks, 4 model families, and 3 model scales (3B-14B), with Jaccard, embedding, and NLI-based baselines at three DeBERTa scales (all ~0.51 AUROC). Cross-embedder validation with two independent embedding families rules out coupling bias. Cross-dataset validation on WebQuestions (58.0% SCR) confirms generalization beyond TruthfulQA. The central finding -- response homogenization -- is implementation-independent and label-free. Motivated by this diagnosis, we explore a cheapest-first cascade (UCBD) over orthogonal uncertainty signals. Selective prediction raises GSM8K accuracy from 84.4% to 93.2% at 50% coverage; weakly dependent boundaries (|r| <= 0.12) enable 57% cost savings.
>
---
#### [new 093] LLMs Do Not Grade Essays Like Humans
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自动作文评分任务，探讨LLM与人类评分的一致性问题。研究对比了LLM与人类评分差异，分析其评分行为及反馈一致性。**

- **链接: [https://arxiv.org/pdf/2603.23714](https://arxiv.org/pdf/2603.23714)**

> **作者:** Jerin George Mathew; Sumayya Taher; Anindita Kundu; Denilson Barbosa
>
> **摘要:** Large language models have recently been proposed as tools for automated essay scoring, but their agreement with human grading remains unclear. In this work, we evaluate how LLM-generated scores compare with human grades and analyze the grading behavior of several models from the GPT and Llama families in an out-of-the-box setting, without task-specific training. Our results show that agreement between LLM and human scores remains relatively weak and varies with essay characteristics. In particular, compared to human raters, LLMs tend to assign higher scores to short or underdeveloped essays, while assigning lower scores to longer essays that contain minor grammatical or spelling errors. We also find that the scores generated by LLMs are generally consistent with the feedback they generate: essays receiving more praise tend to receive higher scores, while essays receiving more criticism tend to receive lower scores. These results suggest that LLM-generated scores and feedback follow coherent patterns but rely on signals that differ from those used by human raters, resulting in limited alignment with human grading practices. Nevertheless, our work shows that LLMs produce feedback that is consistent with their grading and that they can be reliably used in supporting essay scoring.
>
---
#### [new 094] What and When to Learn: CURriculum Ranking Loss for Large-Scale Speaker Verification
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于语音识别中的说话人验证任务，旨在解决大规模数据中样本质量不一导致的模型性能下降问题。提出Curry损失函数，通过在线评估样本难度，提升验证效果。**

- **链接: [https://arxiv.org/pdf/2603.24432](https://arxiv.org/pdf/2603.24432)**

> **作者:** Massa Baali; Sarthak Bisht; Rita Singh; Bhiksha Raj
>
> **摘要:** Speaker verification at large scale remains an open challenge as fixed-margin losses treat all samples equally regardless of quality. We hypothesize that mislabeled or degraded samples introduce noisy gradients that disrupt compact speaker manifolds. We propose Curry (CURriculum Ranking), an adaptive loss that estimates sample difficulty online via Sub-center ArcFace: confidence scores from dominant sub-center cosine similarity rank samples into easy, medium, and hard tiers using running batch statistics, without auxiliary annotations. Learnable weights guide the model from stable identity foundations through manifold refinement to boundary sharpening. To our knowledge, this is the largest-scale speaker verification system trained to date. Evaluated on VoxCeleb1-O, and SITW, Curry reduces EER by 86.8\% and 60.0\% over the Sub-center ArcFace baseline, establishing a new paradigm for robust speaker verification on imperfect large-scale data.
>
---
#### [new 095] Beyond Accuracy: Introducing a Symbolic-Mechanistic Approach to Interpretable Evaluation
- **分类: cs.LG; cs.AI; cs.CL; cs.SC**

- **简介: 该论文属于自然语言处理任务，旨在解决模型泛化能力评估问题。通过结合符号规则与机制解释，揭示模型是否真正理解而非简单记忆数据。**

- **链接: [https://arxiv.org/pdf/2603.23517](https://arxiv.org/pdf/2603.23517)**

> **作者:** Reza Habibi; Darian Lee; Magy Seif El-Nasr
>
> **摘要:** Accuracy-based evaluation cannot reliably distinguish genuine generalization from shortcuts like memorization, leakage, or brittle heuristics, especially in small-data regimes. In this position paper, we argue for mechanism-aware evaluation that combines task-relevant symbolic rules with mechanistic interpretability, yielding algorithmic pass/fail scores that show exactly where models generalize versus exploit patterns. We demonstrate this on NL-to-SQL by training two identical architectures under different conditions: one without schema information (forcing memorization), one with schema (enabling grounding). Standard evaluation shows the memorization model achieves 94% field-name accuracy on unseen data, falsely suggesting competence. Our symbolic-mechanistic evaluation reveals this model violates core schema generalization rules, a failure invisible to accuracy metrics.
>
---
#### [new 096] OneSearch-V2: The Latent Reasoning Enhanced Self-distillation Generative Search Framework
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于搜索系统任务，旨在解决生成式检索中的查询理解不足、用户意图挖掘不充分及过拟合问题。提出OneSearch-V2框架，通过增强推理和自蒸馏方法提升搜索效果。**

- **链接: [https://arxiv.org/pdf/2603.24422](https://arxiv.org/pdf/2603.24422)**

> **作者:** Ben Chen; Siyuan Wang; Yufei Ma; Zihan Liang; Xuxin Zhang; Yue Lv; Ying Yang; Huangyu Dai; Lingtao Mao; Tong Zhao; Zhipeng Qian; Xinyu Sun; Zhixin Zhai; Yang Zhao; Bochao Liu; Jingshan Lv; Xiao Liang; Hui Kong; Jing Chen; Han Li; Chenyi Lei; Wenwu Ou; Kun Gai
>
> **备注:** Key codes are available at this https URL. Feel free to contact benchen4395@gmail.com
>
> **摘要:** Generative Retrieval (GR) has emerged as a promising paradigm for modern search systems. Compared to multi-stage cascaded architecture, it offers advantages such as end-to-end joint optimization and high computational efficiency. OneSearch, as a representative industrial-scale deployed generative search framework, has brought significant commercial and operational benefits. However, its inadequate understanding of complex queries, inefficient exploitation of latent user intents, and overfitting to narrow historical preferences have limited its further performance improvement. To address these challenges, we propose \textbf{OneSearch-V2}, a latent reasoning enhanced self-distillation generative search framework. It contains three key innovations: (1) a thought-augmented complex query understanding module, which enables deep query understanding and overcomes the shallow semantic matching limitations of direct inference; (2) a reasoning-internalized self-distillation training pipeline, which uncovers users' potential yet precise e-commerce intentions beyond log-fitting through implicit in-context learning; (3) a behavior preference alignment optimization system, which mitigates reward hacking arising from the single conversion metric, and addresses personal preference via direct user feedback. Extensive offline evaluations demonstrate OneSearch-V2's strong query recognition and user profiling capabilities. Online A/B tests further validate its business effectiveness, yielding +3.98\% item CTR, +3.05\% buyer conversion rate, and +2.11\% order volume. Manual evaluation further confirms gains in search experience quality, with +1.65\% in page good rate and +1.37\% in query-item relevance. More importantly, OneSearch-V2 effectively mitigates common search system issues such as information bubbles and long-tail sparsity, without incurring additional inference costs or serving latency.
>
---
## 更新

#### [replaced 001] Disentangling Knowledge Representations for Large Language Model Editing
- **分类: cs.CL**

- **简介: 该论文属于知识编辑任务，旨在解决大模型编辑中细粒度无关知识被意外修改的问题。提出DiKE方法，通过解耦知识表示实现精准编辑。**

- **链接: [https://arxiv.org/pdf/2505.18774](https://arxiv.org/pdf/2505.18774)**

> **作者:** Mengqi Zhang; Zisheng Zhou; Xiaotian Ye; Qiang Liu; Zhaochun Ren; Zhumin Chen; Pengjie Ren
>
> **备注:** ICLR 2026
>
> **摘要:** Knowledge Editing has emerged as a promising solution for efficiently updating embedded knowledge in large language models (LLMs). While existing approaches demonstrate effectiveness in integrating new knowledge and preserving the original capabilities of LLMs, they fail to maintain fine-grained irrelevant knowledge, namely facts that share the same subject as edited knowledge but differ in relation and object. This challenge arises because subject representations inherently encode multiple attributes, causing the target and fine-grained irrelevant knowledge to become entangled in the representation space, and thus vulnerable to unintended alterations during editing. To address this, we propose DiKE, a novel approach that Disentangles Knowledge representations for LLM Editing (DiKE). DiKE consists of two key components: a Knowledge Representation Disentanglement (KRD) module that decomposes the subject representation into target-knowledge-related and -unrelated components, and a Disentanglementbased Knowledge Edit (DKE) module that updates only the target-related component while explicitly preserving the unrelated one. We further derive a closedform, rank-one parameter update based on matrix theory to enable efficient and minimally invasive edits. To rigorously evaluate fine-grained irrelevant knowledge preservation, we construct FINE-KED, a new benchmark comprising fine-grained irrelevant knowledge at different levels of relational similarity to the edited knowledge. Extensive experiments across multiple LLMs demonstrate that DiKE substantially improves fine-grained irrelevant knowledge preservation while maintaining competitive general editing performance.
>
---
#### [replaced 002] Generative Adversarial Reasoner: Enhancing LLM Reasoning with Adversarial Reinforcement Learning
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于增强大语言模型推理能力的任务，解决其在数学推理中出现的错误问题。通过对抗强化学习联合训练模型与判别器，提升推理质量。**

- **链接: [https://arxiv.org/pdf/2512.16917](https://arxiv.org/pdf/2512.16917)**

> **作者:** Qihao Liu; Luoxin Ye; Wufei Ma; Yu-Cheng Chou; Alan Yuille
>
> **备注:** Camera-ready version
>
> **摘要:** Large language models (LLMs) with explicit reasoning capabilities excel at mathematical reasoning yet still commit process errors, such as incorrect calculations, brittle logic, and superficially plausible but invalid steps. In this paper, we introduce Generative Adversarial Reasoner, an on-policy joint training framework designed to enhance reasoning by co-evolving an LLM reasoner and an LLM-based discriminator through adversarial reinforcement learning. A compute-efficient review schedule partitions each reasoning chain into logically complete slices of comparable length, and the discriminator evaluates each slice's soundness with concise, structured justifications. Learning couples complementary signals: the LLM reasoner is rewarded for logically consistent steps that yield correct answers, while the discriminator earns rewards for correctly detecting errors or distinguishing traces in the reasoning process. This produces dense, well-calibrated, on-policy step-level rewards that supplement sparse exact-match signals, improving credit assignment, increasing sample efficiency, and enhancing overall reasoning quality of LLMs. Across various mathematical benchmarks, the method delivers consistent gains over strong baselines with standard RL post-training. Specifically, on AIME24, we improve DeepSeek-R1-Distill-Qwen-7B from 54.0 to 61.3 (+7.3) and DeepSeek-R1-Distill-Llama-8B from 43.7 to 53.7 (+10.0). The modular discriminator also enables flexible reward shaping for objectives such as teacher distillation, preference alignment, and mathematical proof-based reasoning.
>
---
#### [replaced 003] Agentic Automation of BT-RADS Scoring: End-to-End Multi-Agent System for Standardized Brain Tumor Follow-up Assessment
- **分类: cs.CL; cs.MA**

- **简介: 该论文属于医学影像分析任务，旨在解决脑肿瘤随访评估中的标准化问题。通过多智能体系统实现BT-RADS自动评分，提升评估准确性与一致性。**

- **链接: [https://arxiv.org/pdf/2603.21494](https://arxiv.org/pdf/2603.21494)**

> **作者:** Mohamed Sobhi Jabal; Jikai Zhang; Dominic LaBella; Jessica L. Houk; Dylan Zhang; Jeffrey D. Rudie; Kirti Magudia; Maciej A. Mazurowski; Evan Calabrese
>
> **备注:** 17 pages, 5 figures, 4 tables, 2 supplementary figures, 3 supplementary tables
>
> **摘要:** The Brain Tumor Reporting and Data System (BT-RADS) standardizes post-treatment MRI response assessment in patients with diffuse gliomas but requires complex integration of imaging trends, medication effects, and radiation timing. This study evaluates an end-to-end multi-agent large language model (LLM) and convolutional neural network (CNN) system for automated BT-RADS classification. A multi-agent LLM system combined with automated CNN-based tumor segmentation was retrospectively evaluated on 509 consecutive post-treatment glioma MRI examinations from a single high-volume center. An extractor agent identified clinical variables (steroid status, bevacizumab status, radiation date) from unstructured clinical notes, while a scorer agent applied BT-RADS decision logic integrating extracted variables with volumetric measurements. Expert reference standard classifications were established by an independent board-certified neuroradiologist. Of 509 examinations, 492 met inclusion criteria. The system achieved 374/492 (76.0%; 95% CI, 72.1%-79.6%) accuracy versus 283/492 (57.5%; 95% CI, 53.1%-61.8%) for initial clinical assessments (+18.5 percentage points; P<.001). Context-dependent categories showed high sensitivity (BT-1b 100%, BT-1a 92.7%, BT-3a 87.5%), while threshold-dependent categories showed moderate sensitivity (BT-3c 74.8%, BT-2 69.2%, BT-4 69.3%, BT-3b 57.1%). For BT-4, positive predictive value was 92.9%. The multi-agent LLM system achieved higher BT-RADS classification agreement with expert reference standard compared to initial clinical scoring, with high accuracy for context-dependent scores and high positive predictive value for BT-4 detection.
>
---
#### [replaced 004] PRISM: Breaking the O(n) Memory Wall in Long-Context LLM Inference via O(1) Photonic Block Selection
- **分类: physics.optics; cs.AI; cs.AR; cs.CL; cs.LG**

- **简介: 该论文针对长上下文大模型推理中的内存带宽瓶颈问题，提出PRISM方案，通过光子技术实现O(1)的块选择，显著降低内存访问开销。**

- **链接: [https://arxiv.org/pdf/2603.21576](https://arxiv.org/pdf/2603.21576)**

> **作者:** Hyoseok Park; Yeonsang Park
>
> **备注:** 28 pages, 27 figures, 15 tables, including supplementary material. Code available at this https URL
>
> **摘要:** Long-context LLM inference is bottlenecked not by compute but by the O(n) memory bandwidth cost of scanning the KV cache at every decode step -- a wall that no amount of arithmetic scaling can break. Recent photonic accelerators have demonstrated impressive throughput for dense attention computation; however, these approaches inherit the same O(n) memory scaling as electronic attention when applied to long contexts. We observe that the real leverage point is the coarse block-selection step: a memory-bound similarity search that determines which KV blocks to fetch. We identify, for the first time, that this task is structurally matched to the photonic broadcast-and-weight paradigm -- the query fans out to all candidates via passive splitting, signatures are quasi-static (matching electro-optic MRR programming), and only rank order matters (relaxing precision to 4-6 bits). Crucially, the photonic advantage grows with context length: as N increases, the electronic scan cost rises linearly while the photonic evaluation remains O(1). We instantiate this insight in PRISM (Photonic Ranking via Inner-product Similarity with Microring weights), a thin-film lithium niobate (TFLN) similarity engine. Hardware-impaired needle-in-a-haystack evaluation on Qwen2.5-7B confirms 100% accuracy from 4K through 64K tokens at k=32, with 16x traffic reduction at 64K context. PRISM achieves a four-order-of-magnitude energy advantage over GPU baselines at practical context lengths (n >= 4K).
>
---
#### [replaced 005] Mitigating LLM Hallucinations through Domain-Grounded Tiered Retrieval
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型的幻觉问题。通过分阶段检索与验证框架，提升模型生成内容的准确性与可靠性。**

- **链接: [https://arxiv.org/pdf/2603.17872](https://arxiv.org/pdf/2603.17872)**

> **作者:** Md. Asraful Haque; Aasar Mehdi; Maaz Mahboob; Tamkeen Fatima
>
> **备注:** 14 Pages, 5 Figures, 4 Tables; v2: Updated Table 3 and Figure 4 to address minor data inconsistencies and revised the relevant content
>
> **摘要:** Large Language Models (LLMs) have achieved unprecedented fluency but remain susceptible to "hallucinations" - the generation of factually incorrect or ungrounded content. This limitation is particularly critical in high-stakes domains where reliability is paramount. We propose a domain-grounded tiered retrieval and verification architecture designed to systematically intercept factual inaccuracies by shifting LLMs from stochastic pattern-matchers to verified truth-seekers. The proposed framework utilizes a four-phase, self-regulating pipeline implemented via LangGraph: (I) Intrinsic Verification with Early-Exit logic to optimize compute, (II) Adaptive Search Routing utilizing a Domain Detector to target subject-specific archives, (III) Refined Context Filtering (RCF) to eliminate non-essential or distracting information, and (IV) Extrinsic Regeneration followed by atomic claim-level verification. The system was evaluated across 650 queries from five diverse benchmarks: TimeQA v2, FreshQA v2, HaluEval General, MMLU Global Facts, and TruthfulQA. Empirical results demonstrate that the pipeline consistently outperforms zero-shot baselines across all environments. Win rates peaked at 83.7% in TimeQA v2 and 78.0% in MMLU Global Facts, confirming high efficacy in domains requiring granular temporal and numerical precision. Groundedness scores remained robustly stable between 78.8% and 86.4% across factual-answer rows. While the architecture provides a robust fail-safe for misinformation, a persistent failure mode of "False-Premise Overclaiming" was identified. These findings provide a detailed empirical characterization of multi-stage RAG behavior and suggest that future work should prioritize pre-retrieval "answerability" nodes to further bridge the reliability gap in conversational AI.
>
---
#### [replaced 006] Linguistic Comparison of AI- and Human-Written Responses to Online Mental Health Queries
- **分类: cs.HC; cs.AI; cs.CL; cs.SI**

- **简介: 该论文属于自然语言处理任务，旨在比较AI与人类在在线心理健康社区中的回复差异。研究分析了24,114条帖子及138,758条回复，探讨AI回复的优劣及其对社区的影响。**

- **链接: [https://arxiv.org/pdf/2504.09271](https://arxiv.org/pdf/2504.09271)**

> **作者:** Koustuv Saha; Yoshee Jain; Violeta J. Rodriguez; Munmun De Choudhury
>
> **摘要:** The ubiquity and widespread use of digital and online technologies have transformed mental health support, with online mental health communities (OMHCs) providing safe spaces for peer support. More recently, generative AI and large language models (LLMs) have introduced new possibilities for scalable, around-the-clock mental health assistance that could potentially augment and supplement the capabilities of OMHCs. Although genAI shows promise in delivering immediate and personalized responses, its effectiveness in replicating the nuanced, experience-based support of human peers remains an open question. In this study, we harnessed 24,114 posts and 138,758 online community (OC) responses from 55 OMHCs on Reddit. We prompted several state-of-the-art LLMs (GPT-4-Turbo, Llama-3, and Mistral-7B) with these posts, and compared their responses to human-written (OC) responses based on a variety of linguistic measures across psycholinguistics and lexico-semantics. Our findings revealed that AI responses are more verbose, readable, and analytically structured, but lack linguistic diversity and personal narratives inherent in human--human interactions. Through a qualitative examination, we found validation as well as complementary insights into the nature of AI responses, such as its neutral stance and the absence of seeking back-and-forth clarifications. We discuss the ethical and practical implications of integrating generative AI into OMHCs, advocating for frameworks that balance AI's scalability and timeliness with the irreplaceable authenticity, social interactiveness, and expertise of human connections that form the ethos of online support communities.
>
---
#### [replaced 007] JUBAKU: An Adversarial Benchmark for Exposing Culturally Grounded Stereotypes in Japanese LLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的社会偏见检测任务，旨在解决非英语语言模型中文化特定偏见的评估问题。研究构建了JUBAKU基准，通过精心设计的对话场景揭示日本大语言模型中的隐性偏见。**

- **链接: [https://arxiv.org/pdf/2603.20581](https://arxiv.org/pdf/2603.20581)**

> **作者:** Taihei Shiotani; Masahiro Kaneko; Ayana Niwa; Yuki Maruyama; Daisuke Oba; Masanari Ohi; Naoaki Okazaki
>
> **摘要:** Social biases reflected in language are inherently shaped by cultural norms, which vary significantly across regions and lead to diverse manifestations of stereotypes. Existing evaluations of social bias in large language models (LLMs) for non-English contexts, however, often rely on translations of English benchmarks. Such benchmarks fail to reflect local cultural norms, including those found in Japanese. For instance, Western benchmarks may overlook Japan-specific stereotypes related to hierarchical relationships, regional dialects, or traditional gender roles. To address this limitation, we introduce Japanese cUlture adversarial BiAs benchmarK Under handcrafted creation (JUBAKU), a benchmark tailored to Japanese cultural contexts. JUBAKU uses adversarial construction to expose latent biases across ten distinct cultural categories. Unlike existing benchmarks, JUBAKU features dialogue scenarios hand-crafted by native Japanese annotators, specifically designed to trigger and reveal latent social biases in Japanese LLMs. We evaluated nine Japanese LLMs on JUBAKU and three others adapted from English benchmarks. All models clearly exhibited biases on JUBAKU, performing below the random baseline of 50% with an average accuracy of 23% (ranging from 13% to 33%), despite higher accuracy on the other benchmarks. Human annotators achieved 91% accuracy in identifying unbiased responses, confirming JUBAKU's reliability and its adversarial nature to LLMs.
>
---
#### [replaced 008] Collaborative Causal Sensemaking: Closing the Complementarity Gap in Human-AI Decision Support
- **分类: cs.CL; cs.AI; cs.HC; cs.LG**

- **简介: 该论文属于人机协作任务，旨在解决人类与AI在决策支持中互补性不足的问题。通过提出CCS框架，提升AI的因果推理与协作能力，促进更有效的团队合作。**

- **链接: [https://arxiv.org/pdf/2512.07801](https://arxiv.org/pdf/2512.07801)**

> **作者:** Raunak Jain
>
> **摘要:** LLM-based agents are increasingly deployed for expert decision support, yet human-AI teams in high-stakes settings do not yet reliably outperform the best individual. We argue this complementarity gap reflects a fundamental mismatch: current agents are trained as answer engines, not as partners in the collaborative sensemaking through which experts actually make decisions. Sensemaking (the ability to co-construct causal explanations, surface uncertainties, and adapt goals) is the key capability that current training pipelines do not explicitly develop or evaluate. We propose Collaborative Causal Sensemaking (CCS) as a research agenda to develop this capability from the ground up, spanning new training environments that reward collaborative thinking, representations for shared human-AI mental models, and evaluation centred on trust and complementarity. Taken together, these directions shift MAS research from building oracle-like answer engines to cultivating AI teammates that co-reason with their human partners over the causal structure of shared decisions, advancing the design of effective human-AI teams.
>
---
#### [replaced 009] From Guidelines to Guarantees: A Graph-Based Evaluation Harness for Domain-Specific Evaluation of LLMs
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于语言模型评估任务，旨在解决领域专用模型评估基准不足的问题。通过构建知识图谱动态生成评测数据，提升评估的全面性和有效性。**

- **链接: [https://arxiv.org/pdf/2508.20810](https://arxiv.org/pdf/2508.20810)**

> **作者:** Jessica M. Lundin; Usman Nasir Nakakana; Guillaume Chabot-Couture
>
> **摘要:** Rigorous evaluation of domain-specific language models requires benchmarks that are comprehensive, contamination-resistant, and maintainable. Static, manually curated datasets do not satisfy these properties. We present a graph-based evaluation harness that transforms structured clinical guidelines into a queryable knowledge graph and dynamically instantiates evaluation queries via graph traversal. The framework provides three guarantees: (1) complete coverage of guideline relationships; (2) surface-form contamination resistance through combinatorial variation; and (3) validity inherited from expert-authored graph structure. Applied to the WHO IMCI guidelines, the harness generates clinically grounded multiple-choice questions spanning symptom recognition, treatment, severity classification, and follow-up care. Evaluation across five language models reveals systematic capability gaps. Models perform well on symptom recognition but show lower accuracy on treatment protocols and clinical management decisions. The framework supports continuous regeneration of evaluation data as guidelines evolve and generalizes to domains with structured decision logic. This provides a scalable foundation for evaluation infrastructure.
>
---
#### [replaced 010] Let the Agent Search: Autonomous Exploration Beats Rigid Workflows in Temporal Question Answering
- **分类: cs.CL**

- **简介: 该论文聚焦于时间知识图谱问答任务，解决多跳推理与复杂时间约束下的问答问题。提出AT2QA框架，使模型自主探索并动态修正推理过程，提升性能并保持透明性。**

- **链接: [https://arxiv.org/pdf/2603.01853](https://arxiv.org/pdf/2603.01853)**

> **作者:** Xufei Lv; Jiahui Yang; Haoyuan Sun; Xialin Su; Zhiliang Tian; Yifu Gao; Linbo Qiao; Houde Liu
>
> **备注:** Revised version with three added authors and additional experiments
>
> **摘要:** Temporal Knowledge Graph Question Answering (TKGQA) is challenging because it requires multi-hop reasoning under complex temporal constraints. Recent LLM-based approaches have improved semantic modeling for this task, but many still rely on fixed reasoning workflows or costly post-training, which can limit adaptability and make error recovery difficult. We show that enabling an off-the-shelf Large Language Model (LLM) to determine its next action is already effective in a zero-shot setting. Based on this insight, we propose AT2QA, an Autonomous and Training-free Agent for TKG Question Answering. AT2QA empowers the LLM to iteratively interact with the TKG via a generic search tool, inherently enabling autonomous exploration and dynamic self-correction during reasoning. To further elicit the LLM's potential for complex temporal reasoning, we introduce a training-free experience mining mechanism that distills a compact few-shot demonstration library from successful self-generated trajectories. AT2QA also yields a transparent audit trail for every prediction. Experiments on three challenging benchmarks -- MultiTQ, Timeline-CronQuestion, and Timeline-ICEWS-Actor -- show that AT2QA achieves new state-of-the-art performance, surpassing the strongest baselines by 10.7, 4.9, and 11.2 absolute points, respectively. Our code is available at this https URL
>
---
#### [replaced 011] IDP Accelerator: Agentic Document Intelligence from Extraction to Compliance Validation
- **分类: cs.CL**

- **简介: 该论文提出IDP Accelerator，解决工业NLP中从非结构化文档提取结构化信息并确保合规的问题。通过四个模块实现端到端文档智能处理。**

- **链接: [https://arxiv.org/pdf/2602.23481](https://arxiv.org/pdf/2602.23481)**

> **作者:** Md Mofijul Islam; Md Sirajus Salekin; Joe King; Priyashree Roy; Vamsi Thilak Gudi; Spencer Romo; Akhil Nooney; David Kaleko; Boyi Xie; Bob Strahan; Diego A. Socolinsky
>
> **摘要:** Understanding and extracting structured insights from unstructured documents remains a foundational challenge in industrial NLP. While Large Language Models (LLMs) enable zero-shot extraction, traditional pipelines often fail to handle multi-document packets, complex reasoning, and strict compliance requirements. We present IDP (Intelligent Document Processing) Accelerator, a framework enabling agentic AI for end-to-end document intelligence with four key components: (1) DocSplit, a novel benchmark dataset and multimodal classifier using BIO tagging to segment complex document packets; (2) configurable Extraction Module leveraging multimodal LLMs to transform unstructured content into structured data; (3) Agentic Analytics Module, compliant with the Model Context Protocol (MCP) providing data access through secure, sandboxed code execution; and (4) Rule Validation Module replacing deterministic engines with LLM-driven logic for complex compliance checks. The interactive demonstration enables users to upload document packets, visualize classification results, and explore extracted data through an intuitive web interface. We demonstrate effectiveness across industries, highlighting a production deployment at a leading healthcare provider achieving 98% classification accuracy, 80% reduced processing latency, and 77% lower operational costs over legacy baselines. IDP Accelerator is open-sourced with a live demonstration available to the community.
>
---
#### [replaced 012] ChartAttack: Testing the Vulnerability of LLMs to Malicious Prompting in Chart Generation
- **分类: cs.CL**

- **简介: 该论文属于多模态大语言模型安全研究任务，旨在解决MLLM在图表生成中被恶意提示误导的问题。工作包括提出ChartAttack框架和AttackViz数据集，评估并提升模型对误导性图表的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.12983](https://arxiv.org/pdf/2601.12983)**

> **作者:** Jesus-German Ortiz-Barajas; Jonathan Tonglet; Vivek Gupta; Iryna Gurevych
>
> **摘要:** Multimodal large language models (MLLMs) are increasingly used to automate chart generation from data tables, enabling efficient data analysis and reporting but also introducing new misuse risks. In this work, we introduce ChartAttack, a novel framework for evaluating how MLLMs can be misused to generate misleading charts at scale. ChartAttack injects misleaders into chart designs, aiming to induce incorrect interpretations of the underlying data. Furthermore, we create AttackViz, a chart question-answering (QA) dataset where each (chart specification, QA) pair is labeled with effective misleaders and their induced incorrect answers. ChartAttack significantly degrades QA performance, reducing MLLM accuracy by 17.2 points in-domain and 11.9 cross-domain. Preliminary human results (limited sample size) indicate a 20.2-point accuracy drop. Finally, we demonstrate that AttackViz can be used to fine-tune MLLMs to improve robustness against misleading charts. Our findings highlight an urgent need for robustness and security considerations in the design, evaluation, and deployment of MLLM-based chart generation systems. We make our code and data publicly available.
>
---
#### [replaced 013] Quantification and object perception in Multimodal Large Language Models and human linguistic cognition
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，探讨多模态大语言模型在量化表达上的表现，分析其与人类语言认知的差异。**

- **链接: [https://arxiv.org/pdf/2511.08126](https://arxiv.org/pdf/2511.08126)**

> **作者:** Raquel Montero; Natalia Moskvina; Paolo Morosi; Tamara Serrano; Elena Pagliarini; Evelina Leivada
>
> **摘要:** Quantification has been proven to be a particularly difficult linguistic phenomenon for (Multimodal) Large Language Models (MLLMs). However, given that quantification interfaces with the logic, pragmatic, and numerical domains, the exact reasons for the poor performance are still unclear. This paper looks at three key features of human quantification shared cross-linguistically that have remained so far unexplored in the (M)LLM literature: the ordering of quantifiers into scales, the ranges of use and prototypicality, and the biases inherent in the human approximate number system. The aim is to determine how these features are encoded in the models' architecture, how they may differ from humans, and whether the results are affected by the type of model (thinking vs. instruct) and the language under investigation. Results show that although thinking models showed a high accuracy in the numerosity estimation task and in the organization of quantifiers into scales, there are still key differences between humans and LLMs across all model types, particularly in terms of ranges of use and prototypicality values. This work, thus, paves the way for addressing the nature of MLLMs as semantic and pragmatic agents, while the cross-linguistic lens can elucidate whether their abilities are robust and stable across different languages.
>
---
#### [replaced 014] TikZilla: Scaling Text-to-TikZ with High-Quality Data and Reinforcement Learning
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于文本到TikZ生成任务，旨在解决数据量小、质量低及生成错误问题。通过构建高质量数据集并结合强化学习训练模型，提升生成准确性。**

- **链接: [https://arxiv.org/pdf/2603.03072](https://arxiv.org/pdf/2603.03072)**

> **作者:** Christian Greisinger; Steffen Eger
>
> **摘要:** Large language models (LLMs) are increasingly used to assist scientists across diverse workflows. A key challenge is generating high-quality figures from textual descriptions, often represented as TikZ programs that can be rendered as scientific images. Prior research has proposed a variety of datasets and modeling approaches for this task. However, existing datasets for Text-to-TikZ are too small and noisy to capture the complexity of TikZ, causing mismatches between text and rendered figures. Moreover, prior approaches rely solely on supervised fine-tuning (SFT), which does not expose the model to the rendered semantics of the figure, often resulting in errors such as looping, irrelevant content, and incorrect spatial relations. To address these issues, we construct DaTikZ-V4, a dataset more than four times larger and substantially higher in quality than DaTikZ-V3, enriched with LLM-generated figure descriptions. Using this dataset, we train TikZilla, a family of small open-source Qwen models (3B and 8B) with a two-stage pipeline of SFT followed by reinforcement learning (RL). For RL, we leverage an image encoder trained via inverse graphics to provide semantically faithful reward signals. Extensive human evaluations with over 1,000 judgments show that TikZilla improves by 1.5-2 points over its base models on a 5-point scale, surpasses GPT-4o by 0.5 points, and matches GPT-5 in the image-based evaluation, while operating at much smaller model sizes. Code, data, and models will be made available.
>
---
#### [replaced 015] FedSRD: Sparsify-Reconstruct-Decompose for Communication-Efficient Federated Large Language Models Fine-Tuning
- **分类: cs.CL**

- **简介: 该论文属于联邦学习任务，解决LoRA在联邦训练中的通信开销问题。提出FedSRD框架，通过稀疏化、重构和分解降低通信成本，提升效率。**

- **链接: [https://arxiv.org/pdf/2510.04601](https://arxiv.org/pdf/2510.04601)**

> **作者:** Guochen Yan; Luyuan Xie; Qingni Shen; Yuejian Fang; Zhonghai Wu
>
> **备注:** Accepted by WWW 2026
>
> **摘要:** The current paradigm of training large language models (LLMs) on public available Web data is becoming unsustainable as high-quality data sources in specialized domains near exhaustion. Federated Learning (FL) emerges as a practical solution for the next generation of AI on a decentralized Web, enabling privacy-preserving collaborative fine-tuning on decentralized private data. While Low-Rank Adaptation (LoRA) is standard for efficient fine-tuning, its federated application faces a critical bottleneck: communication overhead under heterogeneous network conditions. Structural redundancy in LoRA parameters increases communication costs and causes aggregation conflicts. To address this, we propose FedSRD, a Sparsify-Reconstruct-Decompose framework for communication-efficient federated LLM fine-tuning. We introduce importance-aware sparsification to reduce the upload parameter count while preserving the structural integrity of LoRA updates. The server aggregates updates in full-rank space to mitigate conflicts, then decomposes the global update into a sparse low-rank format for broadcast, ensuring a symmetrically efficient cycle. We also propose an efficient variant, FedSRD-e, to reduce computational overhead. Experiments on 10 benchmarks show our framework significantly reduces communication costs by up to 90\% while improving performance on heterogeneous client data.
>
---
#### [replaced 016] From Text to Talk: Audio-Language Model Needs Non-Autoregressive Joint Training
- **分类: cs.CL**

- **简介: 该论文提出Text-to-Talk框架，解决多模态语音生成任务中的自回归与非自回归训练矛盾，通过统一模型结构和训练策略提升性能。**

- **链接: [https://arxiv.org/pdf/2509.20072](https://arxiv.org/pdf/2509.20072)**

> **作者:** Tianqiao Liu; Xueyi Li; Hao Wang; Haoxuan Li; Zhichao Chen; Weiqi Luo; Zitao Liu
>
> **摘要:** Recent advances in large language models (LLMs) have attracted significant interest in extending their capabilities to multimodal scenarios, particularly for speech-to-speech conversational systems. However, existing multimodal models handling interleaved audio and text rely on autoregressive (AR) methods, overlooking that text depends on target-target relations whereas audio depends mainly on source-target relations. In this work, we propose Text-to-Talk (TtT), a unified audio-text framework that integrates AR text generation with non-autoregressive (NAR) audio diffusion in a single Transformer. By leveraging the any-order AR property of absorbing discrete diffusion, our approach provides a unified training objective for text and audio. To support this hybrid generation paradigm, we design a modality-aware attention mechanism that enforces causal decoding for text while allowing bidirectional modeling within audio spans, and further introduce three training strategies that reduce train-test discrepancies. During inference, TtT employs block-wise diffusion to synthesize audio in parallel while flexibly handling variable-length outputs. Comprehensive experiments on Audio-QA, ASR, AAC and speech-to-speech benchmarks show that TtT consistently surpasses strong AR and NAR baselines, with additional ablation and training-strategy analyses confirming the contribution of each component. We will open-source our models, data and code to facilitate future research in this direction.
>
---
#### [replaced 017] Offline-First Large Language Model Architecture for AI-Assisted Learning with Adaptive Response Levels in Low-Connectivity Environments
- **分类: cs.CY; cs.AR; cs.CL; cs.HC**

- **简介: 该论文属于教育技术领域，旨在解决低网络环境下AI辅助学习的问题。通过设计离线运行的大型语言模型架构，实现本地化、自适应的学术支持与互动教学。**

- **链接: [https://arxiv.org/pdf/2603.03339](https://arxiv.org/pdf/2603.03339)**

> **作者:** Joseph Walusimbi; Ann Move Oguti; Joshua Benjamin Ssentongo; Keith Ainebyona
>
> **备注:** There are mistakes, inaccurate information recorded about user responses, and the response times
>
> **摘要:** Artificial intelligence (AI) and large language models (LLMs) are transforming educational technology by enabling conversational tutoring, personalized explanations, and inquiry-driven learning. However, most AI-based learning systems rely on continuous internet connectivity and cloud-based computation, limiting their use in bandwidth-constrained environments. This paper presents an offline-first large language model architecture designed for AI-assisted learning in low-connectivity settings. The system performs all inference locally using quantized language models and incorporates hardware-aware model selection to enable deployment on low-specification CPU-only devices. By removing dependence on cloud infrastructure, the system provides curriculum-aligned explanations and structured academic support through natural-language interaction. To support learners at different educational stages, the system includes adaptive response levels that generate explanations at varying levels of complexity: Simple English, Lower Secondary, Upper Secondary, and Technical. This allows explanations to be adjusted to student ability, improving clarity and understanding of academic concepts. The system was deployed in selected secondary and tertiary institutions under limited-connectivity conditions and evaluated across technical performance, usability, perceived response quality, and educational impact. Results show stable operation on legacy hardware, acceptable response times, and positive user perceptions regarding support for self-directed learning. These findings demonstrate the feasibility of offline large language model deployment for AI-assisted education in low-connectivity environments.
>
---
#### [replaced 018] From Sycophancy to Sensemaking: Premise Governance for Human-AI Decision Making
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于人机协作决策任务，旨在解决AI过度迎合、缺乏判断力的问题。通过前提治理和冲突检测机制，提升决策可靠性。**

- **链接: [https://arxiv.org/pdf/2602.02378](https://arxiv.org/pdf/2602.02378)**

> **作者:** Raunak Jain
>
> **摘要:** As LLMs expand from assistance to decision support, a dangerous pattern emerges: fluent agreement without calibrated judgment. Low-friction assistants can become sycophantic, baking in implicit assumptions and pushing verification costs onto experts, while outcomes arrive too late to serve as reward signals. In deep-uncertainty decisions (where objectives are contested and reversals are costly), scaling fluent agreement amplifies poor commitments faster than it builds expertise. We argue reliable human-AI partnership requires a shift from answer generation to collaborative premise governance over a knowledge substrate, negotiating only what is decision-critical. A discrepancy-driven control loop operates over this substrate: detecting conflicts, localizing misalignment via typed discrepancies (teleological, epistemic, procedural), and triggering bounded negotiation through decision slices. Commitment gating blocks action on uncommitted load-bearing premises unless overridden under logged risk; value-gated challenge allocates probing under interaction cost. Trust then attaches to auditable premises and evidence standards, not conversational fluency. We illustrate with tutoring and propose falsifiable evaluation criteria.
>
---
#### [replaced 019] Agent-Diff: Benchmarking LLM Agents on Enterprise API Tasks via Code Execution with State-Diff-Based Evaluation
- **分类: cs.SE; cs.CL**

- **简介: 该论文提出Agent-Diff框架，用于评估LLM代理在企业API任务中的表现。解决真实环境与控制环境间的平衡问题，通过状态差异评估和容器化API沙箱实现可靠基准测试。**

- **链接: [https://arxiv.org/pdf/2602.11224](https://arxiv.org/pdf/2602.11224)**

> **作者:** Hubert M. Pysklo; Artem Zhuravel; Patrick D. Watson
>
> **备注:** Pre-Print. Under review for KDD 2026
>
> **摘要:** We present Agent-Diff, a novel benchmarking framework for evaluating agentic Large Language Models (LLMs) on real-world productivity software API tasks via code execution. Agentic LLM performance varies due to differences in models, external tool access, prompt structures, and agentic frameworks. Benchmarks must make fundamental trade-offs between a sandboxed approach that controls for variation in software environments and more ecologically valid approaches employing real services. Agent-Diff attempts to capture the desirable features of both of these approaches by including access to the real API interfaces for software services while sandboxing the environment in which calls are made, processed, and evaluated. This approach relies on two key innovations. The first is a novel state-diff contract, which separates process from outcome - rather than fuzzy trace or parameter matching, we define task success as whether the expected change in environment state was achieved. The second is a novel sandbox built on containerized replicas of enterprise APIs, allowing all models to interact with the same service interfaces through code execution. This enables controlled evaluation against a common set of state-diff contracts while preserving the structure of real-world API interaction. Using the Agent-Diff framework, we provide benchmarks for nine LLMs across 224 tasks utilizing enterprise software workflows. In addition, we evaluate the robustness of the framework with ablation experiments to assess the contribution of access to API documentation on benchmark performance. Code and data: this https URL.
>
---
#### [replaced 020] ProFit: Leveraging High-Value Signals in SFT via Probability-Guided Token Selection
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型微调任务，旨在解决单参考答案导致的过拟合问题。通过分析token概率与语义重要性的关系，提出ProFit方法，选择性屏蔽低概率token以提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.09195](https://arxiv.org/pdf/2601.09195)**

> **作者:** Tao Liu; Taiqiang Wu; Runming Yang; Shaoning Sun; Junjie Wang; Yujiu Yang
>
> **摘要:** Supervised fine-tuning (SFT) is a fundamental post-training strategy to align Large Language Models (LLMs) with human intent. However, traditional SFT often ignores the one-to-many nature of language by forcing alignment with a single reference answer, leading to the model overfitting to non-core expressions. Although our empirical analysis suggests that introducing multiple reference answers can mitigate this issue, the prohibitive data and computational costs necessitate a strategic shift: prioritizing the mitigation of single-reference overfitting over the costly pursuit of answer diversity. To achieve this, we reveal the intrinsic connection between token probability and semantic importance: high-probability tokens carry the core logical framework, while low-probability tokens are mostly replaceable expressions. Based on this insight, we propose ProFit, which selectively masks low-probability tokens to prevent surface-level overfitting. Extensive experiments confirm that ProFit consistently outperforms traditional SFT baselines on general reasoning and mathematical benchmarks.
>
---
#### [replaced 021] Alignment Whack-a-Mole : Finetuning Activates Verbatim Recall of Copyrighted Books in Large Language Models
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文研究模型在微调后泄露版权书籍的问题，属于安全与隐私任务。通过微调使模型回忆出大量原文，揭示了模型存储版权内容的漏洞。**

- **链接: [https://arxiv.org/pdf/2603.20957](https://arxiv.org/pdf/2603.20957)**

> **作者:** Xinyue Liu; Niloofar Mireshghallah; Jane C. Ginsburg; Tuhin Chakrabarty
>
> **备注:** Preprint Under Review
>
> **摘要:** Frontier LLM companies have repeatedly assured courts and regulators that their models do not store copies of training data. They further rely on safety alignment strategies via RLHF, system prompts, and output filters to block verbatim regurgitation of copyrighted works, and have cited the efficacy of these measures in their legal defenses against copyright infringement claims. We show that finetuning bypasses these protections: by training models to expand plot summaries into full text, a task naturally suited for commercial writing assistants, we cause GPT-4o, Gemini-2.5-Pro, and DeepSeek-V3.1 to reproduce up to 85-90% of held-out copyrighted books, with single verbatim spans exceeding 460 words, using only semantic descriptions as prompts and no actual book text. This extraction generalizes across authors: finetuning exclusively on Haruki Murakami's novels unlocks verbatim recall of copyrighted books from over 30 unrelated authors. The effect is not specific to any training author or corpus: random author pairs and public-domain finetuning data produce comparable extraction, while finetuning on synthetic text yields near-zero extraction, indicating that finetuning on individual authors' works reactivates latent memorization from pretraining. Three models from different providers memorize the same books in the same regions ($r \ge 0.90$), pointing to an industry-wide vulnerability. Our findings offer compelling evidence that model weights store copies of copyrighted works and that the security failures that manifest after finetuning on individual authors' works undermine a key premise of recent fair use rulings, where courts have conditioned favorable outcomes on the adequacy of measures preventing reproduction of protected expression.
>
---
#### [replaced 022] DELULU: Discriminative Embedding Learning Using Latent Units for Speaker-Aware Self-Trained Speech Foundational Model
- **分类: cs.SD; cs.CL**

- **简介: 该论文提出DELULU模型，解决语音任务中缺乏说话人判别特征的问题。通过引入说话人信息提升自监督学习效果，显著提升说话人验证和零样本分析性能。**

- **链接: [https://arxiv.org/pdf/2510.17662](https://arxiv.org/pdf/2510.17662)**

> **作者:** Massa Baali; Rita Singh; Bhiksha Raj
>
> **摘要:** Self-supervised speech models have achieved remarkable success on content-driven tasks, yet they remain limited in capturing speaker-discriminative features critical for verification, diarization, and profiling applications. We introduce \textsc{DELULU}, a speaker-aware self-trained foundational model that addresses this limitation by incorporating speaker-informed structure into pseudo-label generation. DELULU leverages frame-level embeddings from ReDimNet, a state-of-the-art speaker verification model, to guide k-means clustering during pre-training, introducing a speaker-discriminative inductive bias that aligns representation learning with speaker identity. DELULU significantly outperforms prior SSL models across a range of speaker-centric tasks, achieving up to \textbf{62\% relative improvement} in equal error rate (EER) for speaker verification and consistent gains on zero-shot profiling tasks including gender, age, accent, and speaker counting; notably surpassing even its teacher model on zero-shot evaluations. Our findings demonstrate that \textbf{DELULU is a strong universal encoder for speaker-aware speech processing}, enabling superior performance without task-specific fine-tuning.
>
---
#### [replaced 023] FHIRPath-QA: Executable Question Answering over FHIR Electronic Health Records
- **分类: cs.CL**

- **简介: 该论文属于医疗问答任务，旨在解决患者特定问题的精准回答问题。提出FHIRPath-QA数据集和文本到FHIRPath查询的问答范式，提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2602.23479](https://arxiv.org/pdf/2602.23479)**

> **作者:** Michael Frew; Nishit Bheda; Bryan Tripp
>
> **备注:** Accepted to LREC 2026 CL4Health Workshop
>
> **摘要:** Though patients are increasingly granted digital access to their electronic health records (EHRs), existing interfaces may not support precise, trustworthy answers to patient-specific questions. Large language models (LLM) show promise in clinical question answering (QA), but retrieval-based approaches are computationally inefficient, prone to hallucination, and difficult to deploy over real-life EHRs. This work introduces FHIRPath-QA, the first open dataset and benchmark for patient-specific QA that includes open-standard FHIRPath queries over real-world clinical data. A text-to-FHIRPath QA paradigm is proposed that shifts reasoning from free-text generation to FHIRPath query synthesis. For o4-mini, this reduced average token usage by 391x relative to retrieval-first prompting (629,829 vs 1,609 tokens per question) and lowered failure rates from 0.36 to 0.09 on clinician-phrased questions. Built on MIMIC-IV on FHIR Demo, the dataset pairs over 14k natural language questions in patient and clinician phrasing with validated FHIRPath queries and answers. Empirically, the evaluated LLMs achieve at most 42% accuracy, highlighting the challenge of the task, but benefit strongly from supervised fine-tuning, with query synthesis accuracy improving from 27% to 79% for 4o-mini. These results highlight that text-to-FHIRPath synthesis has the potential to serve as a practical foundation for safe, efficient, and interoperable consumer health applications, and the FHIRPath-QA dataset and benchmark serve as a starting point for future research on the topic. The full dataset and generation code can be accessed at: this https URL.
>
---
#### [replaced 024] Structured Legal Document Generation in India: A Model-Agnostic Wrapper Approach with VidhikDastaavej
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **简介: 该论文属于法律文本生成任务，旨在解决印度私人法律文件结构化生成的问题。作者构建了VidhikDastaavej数据集，并提出MAW框架以提升生成质量。**

- **链接: [https://arxiv.org/pdf/2504.03486](https://arxiv.org/pdf/2504.03486)**

> **作者:** Shubham Kumar Nigam; Balaramamahanthi Deepak Patnaik; Noel Shallum; Kripabandhu Ghosh; Arnab Bhattacharya
>
> **备注:** Paper accepted in the Language Resources and Evaluation Conference (LREC) 2026 conference
>
> **摘要:** Automating legal document drafting can improve efficiency and reduce the burden of manual legal work. Yet, the structured generation of private legal documents remains underexplored, particularly in the Indian context, due to the scarcity of public datasets and the complexity of adapting models for long-form legal drafting. To address this gap, we introduce VidhikDastaavej, a large-scale, anonymized dataset of private legal documents curated in collaboration with an Indian law firm. Covering 133 diverse categories, this dataset is the first resource of its kind and provides a foundation for research in structured legal text generation and Legal AI more broadly. We further propose a Model-Agnostic Wrapper (MAW), a two-stage generation framework that first plans the section structure of a legal draft and then generates each section with retrieval-based prompts. MAW is independent of any specific LLM, making it adaptable across both open- and closed-source models. Comprehensive evaluation, including lexical, semantic, LLM-based, and expert-driven assessments with inter-annotator agreement, shows that the wrapper substantially improves factual accuracy, coherence, and completeness compared to fine-tuned baselines. This work establishes both a new benchmark dataset and a generalizable generation framework, paving the way for future research in AI-assisted legal drafting.
>
---
#### [replaced 025] Advancing AI Trustworthiness Through Patient Simulation: Risk Assessment of Conversational Agents for Antidepressant Selection
- **分类: cs.CL**

- **简介: 该论文属于医疗AI风险评估任务，旨在解决对话式AI在抗抑郁药物选择中的可信度问题。通过构建患者模拟器，评估不同健康素养水平下的AI表现，识别潜在风险。**

- **链接: [https://arxiv.org/pdf/2602.11391](https://arxiv.org/pdf/2602.11391)**

> **作者:** Md Tanvir Rouf Shawon; Mohammad Sabik Irbaz; Hadeel R. A. Elyazori; Keerti Reddy Resapu; Yili Lin; Vladimir Franzuela Cardenas; Farrokh Alemi; Kevin Lybarger
>
> **摘要:** Objective: This paper introduces a patient simulator for scalable, automated evaluation of healthcare conversational agents, generating realistic, controllable interactions that systematically vary across medical, linguistic, and behavioral dimensions to support risk assessment across populations. Methods: Grounded in the NIST AI Risk Management Framework, the simulator integrates three profile components: (1) medical profiles constructed from All of Us electronic health records using risk-ratio gating; (2) linguistic profiles modeling health literacy and condition-specific communication; and (3) behavioral profiles representing cooperative, distracted, and adversarial engagement. Profiles were evaluated against NIST AI RMF trustworthiness requirements and assessed against an AI Decision Aid for antidepressant selection. Results: Across 500 simulated conversations, the simulator revealed monotonic degradation in AI Decision Aid performance across health literacy levels: Rank-1 concept retrieval ranged from 47.6% (limited) to 81.9% (proficient), with corresponding recommendation degradation. Medical concept fidelity was high (96.6% across 8,210 concepts), validated by human annotators (0.73 kappa) and an LLM judge with comparable agreement (0.78 kappa). Behavioral profiles were reliably distinguished (0.93 kappa), and linguistic profiles showed moderate agreement (0.61 kappa). Conclusions: The simulator exposes measurable performance risks in conversational healthcare AI. Health literacy emerged as a primary risk factor with direct implications for equitable AI deployment.
>
---
#### [replaced 026] Is Multilingual LLM Watermarking Truly Multilingual? Scaling Robustness to 100+ Languages via Back-Translation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，解决多语言水印在中低资源语言中鲁棒性不足的问题。通过引入STEAM方法提升水印检测效果。**

- **链接: [https://arxiv.org/pdf/2510.18019](https://arxiv.org/pdf/2510.18019)**

> **作者:** Asim Mohamed; Martin Gubri
>
> **摘要:** Multilingual watermarking aims to make large language model (LLM) outputs traceable across languages, yet current methods still fall short. Despite claims of cross-lingual robustness, they are evaluated only on high-resource languages. We show that existing multilingual watermarking methods are not truly multilingual: they fail to remain robust under translation attacks in medium- and low-resource languages. We trace this failure to semantic clustering, which fails when the tokenizer vocabulary contains too few full-word tokens for a given language. To address this, we introduce STEAM, a detection method that uses Bayesian optimisation to search among 133 candidate languages for the back-translation that best recovers the watermark strength. It is compatible with any watermarking method, robust across different tokenizers and languages, non-invasive, and easily extendable to new languages. With average gains of +0.23 AUC and +37% TPR@1%, STEAM provides a scalable approach toward fairer watermarking across the diversity of languages.
>
---
#### [replaced 027] Evaluation of Large Language Models via Coupled Token Generation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于语言模型评估任务，旨在解决随机性对模型排名的影响问题。通过构建耦合生成模型，减少样本量并验证评估结果的稳定性。**

- **链接: [https://arxiv.org/pdf/2502.01754](https://arxiv.org/pdf/2502.01754)**

> **作者:** Nina Corvelo Benz; Stratis Tsirtsis; Eleni Straitouri; Ivi Chatzi; Ander Artola Velasco; Suhas Thejaswi; Manuel Gomez-Rodriguez
>
> **摘要:** State of the art large language models rely on randomization to respond to a prompt. As an immediate consequence, a model may respond differently to the same prompt if asked multiple times. In this work, we argue that the evaluation and ranking of large language models should control for the randomization underpinning their functioning. Our starting point is the development of a causal model for coupled autoregressive generation, which allows different large language models to sample responses with the same source of randomness. Building upon our causal model, we first show that, on evaluations based on benchmark datasets, coupled autoregressive generation leads to the same conclusions as vanilla autoregressive generation but using provably fewer samples. However, we further show that, on evaluations based on (human) pairwise comparisons, coupled and vanilla autoregressive generation can surprisingly lead to different rankings when comparing more than two models, even with an infinite amount of samples. This suggests that the apparent advantage of a model over others in existing evaluation protocols may not be genuine but rather confounded by the randomness inherent to the generation process. To illustrate and complement our theoretical results, we conduct experiments with several large language models from the Llama, Mistral and Qwen families. We find that, across multiple benchmark datasets, coupled autoregressive generation requires up to 75% fewer samples to reach the same conclusions as vanilla autoregressive generation. Further, we find that the win-rates derived from pairwise comparisons by a strong large language model to prompts from the LMSYS Chatbot Arena platform differ under coupled and vanilla autoregressive generation.
>
---
#### [replaced 028] PrefPO: Pairwise Preference Prompt Optimization
- **分类: cs.CL**

- **简介: 该论文提出PrefPO，用于优化提示词生成。解决自动化提示优化问题，减少对标注数据依赖，提升提示质量与稳定性。**

- **链接: [https://arxiv.org/pdf/2603.19311](https://arxiv.org/pdf/2603.19311)**

> **作者:** Rahul Singhal; Pradyumna Tambwekar; Karime Maamari
>
> **备注:** Code and data available at this https URL and this https URL
>
> **摘要:** Prompt engineering is effective but labor-intensive, motivating automated optimization methods. Existing methods typically require labeled datasets, which are often unavailable, and produce verbose, repetitive prompts. We introduce PrefPO, a minimal prompt optimization approach inspired by reinforcement learning from human feedback (RLHF). Its preference-based approach reduces the need for labeled data and hyperparameter tuning-only a starting prompt and natural language criteria are needed. PrefPO uses an LLM discriminator to express pairwise preferences over model outputs and provide feedback to an LLM optimizer, iteratively improving performance. We evaluate PrefPO on 9 BIG-Bench Hard (BBH) tasks and IFEval-Hard, a newly-curated, challenging subset of IFEval. PrefPO matches or exceeds SOTA methods, including GEPA, MIPRO, and TextGrad, on 6/9 tasks and performs comparably to TextGrad on IFEval-Hard (82.4% vs 84.5%). Unlike other methods, PrefPO can optimize in both labeled and unlabeled settings. Without labels, PrefPO closely matches its labeled performance on 6/9 tasks, proving effective without ground truth. PrefPO also improves prompt hygiene: we find existing methods produce prompts 14.7x their original length or with 34% repetitive content; PrefPO reduces these issues by 3-5x. Furthermore, both LLM and human judges rate PrefPO's prompts higher than TextGrad's. Finally, we identify prompt hacking in prompt optimizers, where methods game evaluation criteria, and find PrefPO is susceptible at half the rate of TextGrad (37% vs 86%), generating fewer brittle, misaligned prompts.
>
---
#### [replaced 029] Problems with Chinchilla Approach 2: Systematic Biases in IsoFLOP Parabola Fits
- **分类: cs.LG; cs.CL; stat.ML**

- **简介: 该论文属于模型优化任务，指出Chinchilla Approach 2在拟合神经网络缩放定律时存在系统性偏差，提出改进方法Chinchilla Approach 3并验证其有效性。**

- **链接: [https://arxiv.org/pdf/2603.22339](https://arxiv.org/pdf/2603.22339)**

> **作者:** Eric Czech; Zhiwei Xu; Yael Elmatad; Yixin Wang; William Held
>
> **摘要:** Chinchilla Approach 2 is among the most widely used methods for fitting neural scaling laws. Its parabolic approximation introduces systematic biases in compute-optimal allocation estimates, even on noise-free synthetic data. Applied to published Llama 3 IsoFLOP data at open frontier compute scales, these biases imply a parameter underallocation corresponding to 6.5% of the $3.8\times10^{25}$ FLOP training budget and \$1.4M (90% CI: \$412K-\$2.9M) in unnecessary compute at 50% H100 MFU. Simulated multimodal model misallocations show even greater opportunity costs due to higher loss surface asymmetry. Three sources of this error are examined: IsoFLOP sampling grid width (Taylor approximation accuracy), uncentered IsoFLOP sampling, and loss surface asymmetry ($\alpha \neq \beta$). Chinchilla Approach 3 largely eliminates these biases but is often regarded as less data-efficient, numerically unstable, prone to local minima, and harder to implement. Each concern is shown to be unfounded or addressable, especially when the partially linear structure of the objective is exploited via Variable Projection, enabling unbiased inference on all five loss surface parameters through a two-dimensional optimization that is well-conditioned, analytically differentiable, and amenable to dense, or even exhaustive, grid search. It may serve as a more convenient replacement for Approach 2 or a more scalable alternative for adaptations of Approach 3 to richer scaling law formulations. See this https URL for details and this https URL for other results from this study.
>
---
#### [replaced 030] COALA: Numerically Stable and Efficient Framework for Context-Aware Low-Rank Approximation
- **分类: cs.LG; cs.CL; math.NA**

- **简介: 该论文属于神经网络压缩与微调任务，解决低秩近似中的数值不稳定问题。提出COALA框架，避免矩阵求逆，提升稳定性与适用性。**

- **链接: [https://arxiv.org/pdf/2507.07580](https://arxiv.org/pdf/2507.07580)**

> **作者:** Uliana Parkina; Maxim Rakhuba
>
> **摘要:** Recent studies suggest that context-aware low-rank approximation is a useful tool for compression and fine-tuning of modern large-scale neural networks. In this type of approximation, a norm is weighted by a matrix of input activations, significantly improving metrics over the unweighted case. Nevertheless, existing methods for neural networks suffer from numerical instabilities due to their reliance on classical formulas involving explicit Gram matrix computation and their subsequent inversion. We demonstrate that this can degrade the approximation quality or cause numerically singular matrices. To address these limitations, we propose a novel inversion-free regularized framework that is based entirely on stable decompositions and overcomes the numerical pitfalls of prior art. Our method can handle possible challenging scenarios: (1) when calibration matrices exceed GPU memory capacity, (2) when input activation matrices are nearly singular, and even (3) when insufficient data prevents unique approximation. For the latter, we prove that our solution converges to a desired approximation and derive explicit error bounds.
>
---
#### [replaced 031] Explainable embeddings with Distance Explainer
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于可解释AI任务，解决嵌入空间中特征解释问题。提出Distance Explainer方法，通过距离分析生成局部解释，提升模型透明度。**

- **链接: [https://arxiv.org/pdf/2505.15516](https://arxiv.org/pdf/2505.15516)**

> **作者:** Christiaan Meijer; E. G. Patrick Bos
>
> **备注:** 20 pages, 12 figures. Accepted to the 4th World Conference on eXplainable Artificial Intelligence. Method implementation: this https URL
>
> **摘要:** While eXplainable AI (XAI) has advanced significantly, few methods address interpretability in embedded vector spaces where dimensions represent complex abstractions. We introduce Distance Explainer, a novel method for generating local, post-hoc explanations of embedded spaces in machine learning models. Our approach adapts saliency-based techniques from RISE to explain the distance between two embedded data points by assigning attribution values through selective masking and distance-ranked mask filtering. We evaluate Distance Explainer on cross-modal embeddings (image-image and image-caption pairs) using established XAI metrics including Faithfulness, Sensitivity/Robustness, and Randomization. Experiments with ImageNet and CLIP models demonstrate that our method effectively identifies features contributing to similarity or dissimilarity between embedded data points while maintaining high robustness and consistency. We also explore how parameter tuning, particularly mask quantity and selection strategy, affects explanation quality. This work addresses a critical gap in XAI research and enhances transparency and trustworthiness in deep learning applications utilizing embedded spaces.
>
---
#### [replaced 032] KnowledgeSmith: Uncovering Knowledge Updating in LLMs with Model Editing and Unlearning
- **分类: cs.CL**

- **简介: 该论文属于知识更新任务，旨在理解大语言模型的知识更新机制。通过提出KnowledgeSmith框架，研究编辑与遗忘的差异及知识传播规律。**

- **链接: [https://arxiv.org/pdf/2510.02392](https://arxiv.org/pdf/2510.02392)**

> **作者:** Yinyi Luo; Zhexian Zhou; Hao Chen; Kai Qiu; Marios Savvides; Sharon Li; Jindong Wang
>
> **备注:** ICLR 2026
>
> **摘要:** Knowledge editing and machine unlearning are two popular approaches for large language models (LLMs) to stay up-to-date. However, the knowledge updating mechanism of LLMs remains largely unexplored due to insufficient, isolated, and small-scale evaluation. For instance, are LLMs similar to humans in modifying certain knowledge? What differs editing and unlearning as training data increases? This paper proposes KnowledgeSmith, a unified framework to systematically understand the updating mechanism of LLMs. We first cast editing and unlearning as instances of one constrained optimization problem. Then, we propose an automatic dataset generator that provides structured interventions across multiple graph levels and data scales, enabling controlled studies of how different modification strategies propagate through model knowledge. Extensive experiments demonstrate nuanced insights over knowledge propagation, plasticity scaling, consistency, and robustness. For instance, our results show that LLMs do not exhibit similar updating as humans for different levels of knowledge, and there exists consistency-capacity trade-off. We hope our findings can offer suggestions to the design of more reliable and scalable strategies. Code: this https URL
>
---
#### [replaced 033] Team of Thoughts: Efficient Test-time Scaling of Agentic Systems through Orchestrated Tool Calling
- **分类: cs.CL; cs.AI; cs.MA**

- **简介: 该论文提出Team-of-Thoughts框架，解决多智能体系统协同效率问题，通过异构模型协作提升数学推理与代码生成性能。**

- **链接: [https://arxiv.org/pdf/2602.16485](https://arxiv.org/pdf/2602.16485)**

> **作者:** Jeffrey T. H. Wong; Zixi Zhang; Junyi Liu; Yiren Zhao
>
> **备注:** 8 pages
>
> **摘要:** Existing Multi-Agent Systems (MAS) typically rely on homogeneous model configurations, failing to exploit the diverse expertise inherent in different post-trained architectures. We propose Team-of-Thoughts, a heterogeneous MAS framework that treats diverse models as specialized tools within an orchestrator-driven paradigm. Team-of-Thoughts introduces two novel components: (1) Orchestrator Calibration, which identifies models with superior coordination and synthesis capabilities, and (2) Agent Self-Assessment, a protocol where tool agents profile their own domain-specific strengths to guide selection. At inference, the orchestrator dynamically activates the most compatible agents based on these profiles to maximize capability coverage. Across five mathematical reasoning and code generation benchmarks, Team-of-Thoughts consistently outperforms individual models and existing MAS baselines. Notably, on AIME24 and LiveCodeBench, Team-of-Thoughts achieves 96.00% and 77.91% accuracy, respectively, significantly improving over homogeneous role-play baselines (80.00% and 65.93%).
>
---
#### [replaced 034] A Machine Learning Approach for Detection of Mental Health Conditions and Cyberbullying from Social Media
- **分类: cs.CL; cs.SI**

- **简介: 该论文属于分类任务，旨在检测社交媒体中的心理健康问题和网络欺凌。通过构建多类别分类框架，使用社交媒体数据训练模型，提升检测准确性，并提供可解释性工具辅助内容审核。**

- **链接: [https://arxiv.org/pdf/2511.20001](https://arxiv.org/pdf/2511.20001)**

> **作者:** Edward Ajayi; Martha Kachweka; Mawuli Deku; Emily Aiken
>
> **备注:** Best Paper Award at the AAAI-26 Bridge Program on AI for Medicine and Healthcare. Published in Proceedings of the Second AAAI Bridge Program on AI for Medicine and Healthcare, PMLR 317:15-26, 2026. Paper URL: this https URL
>
> **摘要:** Mental health challenges and cyberbullying are increasingly prevalent in digital spaces, necessitating scalable and interpretable detection systems. This paper introduces a unified multiclass classification framework for detecting ten distinct mental health and cyberbullying categories from social media data. We curate datasets from Twitter and Reddit, implementing a rigorous "split-then-balance" pipeline to train on balanced data while evaluating on a realistic, held-out imbalanced test set. We conducted a comprehensive evaluation comparing traditional lexical models, hybrid approaches, and several end-to-end fine-tuned transformers. Our results demonstrate that end-to-end fine-tuning is critical for performance, with the domain-adapted MentalBERT emerging as the top model, achieving an accuracy of 0.92 and a Macro F1 score of 0.76, surpassing both its generic counterpart and a zero-shot LLM baseline. Grounded in a comprehensive ethical analysis, we frame the system as a human-in-the-loop screening aid, not a diagnostic tool. To support this, we introduce a hybrid SHAPLLM explainability framework and present a prototype dashboard ("Social Media Screener") designed to integrate model predictions and their explanations into a practical workflow for moderators. Our work provides a robust baseline, highlighting future needs for multi-label, clinically-validated datasets at the critical intersection of online safety and computational mental health.
>
---
#### [replaced 035] AgentDrift: Unsafe Recommendation Drift Under Tool Corruption Hidden by Ranking Metrics in LLM Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI安全任务，旨在解决LLM代理在高风险场景中的推荐安全问题。通过实验发现现有评估指标无法检测到工具错误导致的安全风险，提出改进的评估方法。**

- **链接: [https://arxiv.org/pdf/2603.12564](https://arxiv.org/pdf/2603.12564)**

> **作者:** Zekun Wu; Adriano Koshiyama; Sahan Bulathwela; Maria Perez-Ortiz
>
> **备注:** 50 pages, 31 tables, 15 figures. Under review at COLM 2026
>
> **摘要:** Tool-augmented LLM agents increasingly serve as multi-turn advisors in high-stakes domains, yet their evaluation relies on ranking-quality metrics that measure what is recommended but not whether it is safe for the user. We introduce a paired-trajectory protocol that replays real financial dialogues under clean and contaminated tool-output conditions across seven LLMs (7B to frontier) and decomposes divergence into information-channel and memory-channel mechanisms. Across the seven models tested, we consistently observe the evaluation-blindness pattern: recommendation quality is largely preserved under contamination (utility preservation ratio approximately 1.0) while risk-inappropriate products appear in 65-93% of turns, a systematic safety failure poorly reflected by standard NDCG. Safety violations are predominantly information-channel-driven, emerge at the first contaminated turn, and persist without self-correction over 23-step trajectories; no agent across 1,563 contaminated turns explicitly questions tool-data reliability. Even narrative-only corruption (biased headlines, no numerical manipulation) induces significant drift while completely evading consistency monitors. A safety-penalized NDCG variant (sNDCG) reduces preservation ratios to 0.51-0.74, indicating that much of the evaluation gap becomes visible once safety is explicitly measured. These results motivate considering trajectory-level safety monitoring, beyond single-turn quality, for deployed multi-turn agents in high-stakes settings.
>
---
#### [replaced 036] EndoCoT: Scaling Endogenous Chain-of-Thought Reasoning in Diffusion Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出EndoCoT框架，解决扩散模型中文本编码器推理深度不足和引导不变的问题，通过迭代思考和终端对齐提升复杂任务处理能力。**

- **链接: [https://arxiv.org/pdf/2603.12252](https://arxiv.org/pdf/2603.12252)**

> **作者:** Xuanlang Dai; Yujie Zhou; Long Xing; Jiazi Bu; Xilin Wei; Yuhong Liu; Beichen Zhang; Kai Chen; Yuhang Zang
>
> **备注:** 23 pages, 18 figures, The code and dataset are publicly available at this https URL
>
> **摘要:** Recently, Multimodal Large Language Models (MLLMs) have been widely integrated into diffusion frameworks primarily as text encoders to tackle complex tasks such as spatial reasoning. However, this paradigm suffers from two critical limitations: (i) MLLMs text encoder exhibits insufficient reasoning depth. Single-step encoding fails to activate the Chain-of-Thought process, which is essential for MLLMs to provide accurate guidance for complex tasks. (ii) The guidance remains invariant during the decoding process. Invariant guidance during decoding prevents DiT from progressively decomposing complex instructions into actionable denoising steps, even with correct MLLM encodings. To this end, we propose Endogenous Chain-of-Thought (EndoCoT), a novel framework that first activates MLLMs' reasoning potential by iteratively refining latent thought states through an iterative thought guidance module, and then bridges these states to the DiT's denoising process. Second, a terminal thought grounding module is applied to ensure the reasoning trajectory remains grounded in textual supervision by aligning the final state with ground-truth answers. With these two components, the MLLM text encoder delivers meticulously reasoned guidance, enabling the DiT to execute it progressively and ultimately solve complex tasks in a step-by-step manner. Extensive evaluations across diverse benchmarks (e.g., Maze, TSP, VSP, and Sudoku) achieve an average accuracy of 92.1%, outperforming the strongest baseline by 8.3 percentage points. The code and dataset are publicly available at this https URL.
>
---
#### [replaced 037] You only need 4 extra tokens: Synergistic Test-time Adaptation for LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于语言模型的测试阶段适应任务，解决领域分布偏移问题。提出SyTTA框架，利用输入和输出不确定性信号，在无需标签的情况下提升模型性能。**

- **链接: [https://arxiv.org/pdf/2510.10223](https://arxiv.org/pdf/2510.10223)**

> **作者:** Yijie Xu; Huizai Yao; Zhiyu Guo; Pengteng Li; Aiwei Liu; Xuming Hu; Weiyu Guo; Hui Xiong
>
> **备注:** Under Review
>
> **摘要:** Large language models (LLMs) are increasingly deployed in specialized domains such as finance, medicine, and agriculture, where they face significant distribution shifts from their training data. Domain-specific fine-tuning can mitigate this challenge but relies on high-quality labeled data that is expensive and slow to collect in expertise-limited settings. We study label-free test-time adaptation for language models and present SyTTA, an inference-time framework that adapts models on-the-fly without additional supervision. SyTTA couples two complementary uncertainty signals that arise under distribution shift: input-side perplexity, indicating mismatch with domain-specific terminology and patterns, and output-side predictive entropy, indicating diffuse and unstable token probabilities during generation. Across diverse model architectures and domain-specific benchmarks, SyTTA delivers consistent gains. Notably, on agricultural question answering, SyTTA improves Rouge-LSum by over 120% on Qwen-2.5-7B with only 4 extra tokens per query. These results show that effective test-time adaptation for language models is achievable without labeled examples, supporting deployment in label-scarce domains. The code will be made available upon acceptance.
>
---
#### [replaced 038] Reward Is Enough: LLMs Are In-Context Reinforcement Learners
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文探讨大语言模型在推理过程中表现出的强化学习能力，属于自然语言处理任务。论文提出ICRL方法，通过奖励信号引导模型自我优化，解决任务性能提升问题。**

- **链接: [https://arxiv.org/pdf/2506.06303](https://arxiv.org/pdf/2506.06303)**

> **作者:** Kefan Song; Amir Moeini; Peng Wang; Lei Gong; Rohan Chandra; Shangtong Zhang; Yanjun Qi
>
> **摘要:** Reinforcement learning (RL) is a framework for solving sequential decision-making problems. In this work, we demonstrate that, surprisingly, RL emerges during the inference time of large language models (LLMs), a phenomenon we term in-context RL (ICRL). To reveal this capability, we introduce a simple multi-round prompting framework, we call ICRL prompting, for inference-time self-improvement. The goal of ICRL prompting is to guide LLMs to perform reinforcement learning during inference for self-improvement on a given task. After each response, the model receives numerical scalar feedback, denoted as a reward. In the next round, we prompt the LLM again together with a context that concatenates all prior responses and their associated rewards. We consistently observe that response quality improves as the context grows. In other words, the LLM can optimize scalar reward signals during inference, exhibiting behavior analogous to reinforcement learning. We evaluate ICRL prompting on Game of 24, creative writing, ScienceWorld, and Olympiad-level math competitions (AIME and HMMT), demonstrating significant improvements over baselines such as Self-Refine and Reflexion. Notably, even when the reward signals are generated by the same LLM, ICRL prompting still improves performance, highlighting a promising new paradigm for test-time scaling.
>
---
#### [replaced 039] How Many Code and Test Cases Are Enough? Evaluating Test Cases Generation from a Binary-Matrix Perspective
- **分类: cs.CL**

- **简介: 该论文属于测试用例生成评估任务，旨在解决如何有效衡量测试用例的故障检测能力。通过构建二进制矩阵框架，提出算法选择最优错误代码集，以提高测试基准的准确性和多样性。**

- **链接: [https://arxiv.org/pdf/2510.08720](https://arxiv.org/pdf/2510.08720)**

> **作者:** Xianzhen Luo; Jinyang Huang; Wenzhen Zheng; Qingfu Zhu; Mingzheng Xu; Yiheng Xu; Yuantao Fan; Wanxiang Che
>
> **备注:** Accepted by ICLR2026
>
> **摘要:** Evaluating test cases automatically generated by Large Language Models (LLMs) is a critical yet challenging task. Existing benchmarks often evaluate the exclusion ratio on large, unstructured collections of wrong codes, suffering from high computational costs and score inflation. Furthermore, they inadvertently reward generators that detect common, trivial bugs, while failing to penalize their inability to identify rare yet critical faults. In this work, we connect two fundamental questions: (1) What is the minimal set of wrong codes sufficient to represent the entire error space? and (2) What is the minimal set of test cases needed to distinguish them? We introduce a novel framework that formalizes benchmark construction as finding an optimal diagnostic basis in a binary code-test matrix, where rows represent wrong codes and columns represent test case results. The rank of this matrix specifies the minimal number of independent error patterns (wrong codes) and provides a tight upper bound on the number of test cases required for complete fault coverage. Our objective is to identify a basis of size equal to the matrix rank that maximizes internal diversity. To tackle this NP-hard problem, we propose WrongSelect, an efficient approximation algorithm to select maximally diverse wrong codes. Applying this framework to millions of competitive programming submissions, we construct TC-Bench, a compact, diverse, and inflation-resistant benchmark. Extensive experiments show that even the most advanced test case generation methods achieve only ~60% exclusion rates on TC-Bench, exposing a significant gap in their diagnostic power and highlighting substantial room for future improvement. Our dataset is available at: this https URL and our code is at: this https URL.
>
---
#### [replaced 040] Phrase-Instance Alignment for Generalized Referring Segmentation
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于 generalized referring segmentation 任务，旨在解决传统模型忽略语言短语与视觉实例对应关系的问题。通过引入实例级推理和POA损失，实现更精确的短语-实例对齐与分割。**

- **链接: [https://arxiv.org/pdf/2411.15087](https://arxiv.org/pdf/2411.15087)**

> **作者:** E-Ro Nguyen; Hieu Le; Dimitris Samaras; Michael S. Ryoo
>
> **备注:** Accepted to PVUW - CVPR 2026 Workshop. Webpage: this https URL
>
> **摘要:** Generalized Referring expressions can describe one object, several related objects, or none at all. Existing generalized referring segmentation (GRES) models treat all cases alike, predicting a single binary mask and ignoring how linguistic phrases correspond to distinct visual instances. To this end, we reformulate GRES as an instance-level reasoning problem, where the model first predicts multiple instance-aware object queries conditioned on the referring expression, then aligns each with its most relevant phrase. This alignment is enforced by a Phrase-Object Alignment (POA) loss that builds fine-grained correspondence between linguistic phrases and visual instances. Given these aligned object instance queries and their learned relevance scores, the final segmentation and the no-target case are both inferred through a unified relevance-weighted aggregation mechanism. This instance-aware formulation enables explicit phrase-instance grounding, interpretable reasoning, and robust handling of complex or null expressions. Extensive experiments on the gRefCOCO and Ref-ZOM benchmarks demonstrate that our method significantly advances state-of-the-art performance by 3.22% cIoU and 12.25% N-acc.
>
---
#### [replaced 041] Red-Teaming Vision-Language-Action Models via Quality Diversity Prompt Generation for Robust Robot Policies
- **分类: cs.RO; cs.AI; cs.CL**

- **简介: 该论文属于机器人任务，旨在提升视觉-语言-动作模型的鲁棒性。通过生成多样化指令，识别模型漏洞并优化其表现。**

- **链接: [https://arxiv.org/pdf/2603.12510](https://arxiv.org/pdf/2603.12510)**

> **作者:** Siddharth Srikanth; Freddie Liang; Ya-Chuan Hsu; Varun Bhatt; Shihan Zhao; Henry Chen; Bryon Tjanaka; Minjune Hwang; Akanksha Saran; Daniel Seita; Aaquib Tabrez; Stefanos Nikolaidis
>
> **摘要:** Vision-Language-Action (VLA) models have significant potential to enable general-purpose robotic systems for a range of vision-language tasks. However, the performance of VLA-based robots is highly sensitive to the precise wording of language instructions, and it remains difficult to predict when such robots will fail. To improve the robustness of VLAs to different wordings, we present Q-DIG (Quality Diversity for Diverse Instruction Generation), which performs red-teaming by scalably identifying diverse natural language task descriptions that induce failures while remaining task-relevant. Q-DIG integrates Quality Diversity (QD) techniques with Vision-Language Models (VLMs) to generate a broad spectrum of adversarial instructions that expose meaningful vulnerabilities in VLA behavior. Our results across multiple simulation benchmarks show that Q-DIG finds more diverse and meaningful failure modes compared to baseline methods, and that fine-tuning VLAs on the generated instructions improves task success rates. Furthermore, results from a user study highlight that Q-DIG generates prompts judged to be more natural and human-like than those from baselines. Finally, real-world evaluations of Q-DIG prompts show results consistent with simulation, and fine-tuning VLAs on the generated prompts further success rates on unseen instructions. Together, these findings suggest that Q-DIG is a promising approach for identifying vulnerabilities and improving the robustness of VLA-based robots. Our anonymous project website is at this http URL.
>
---
#### [replaced 042] EHR2Path: Scalable Modeling of Longitudinal Patient Pathways from Multimodal Electronic Health Records
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出EHR2Path，用于从电子健康记录中建模患者长期诊疗路径。任务是预测患者病情发展，解决现有方法无法全面建模患者路径的问题。工作包括构建多模态框架和引入压缩瓶颈机制。**

- **链接: [https://arxiv.org/pdf/2506.04831](https://arxiv.org/pdf/2506.04831)**

> **作者:** Chantal Pellegrini; Ege Özsoy; David Bani-Harouni; Matthias Keicher; Nassir Navab
>
> **摘要:** Forecasting how a patient's condition is likely to evolve, including possible deterioration, recovery, treatment needs, and care transitions, could support more proactive and personalized care, but requires modeling heterogeneous and longitudinal electronic health record (EHR) data. Yet, existing approaches typically focus on isolated prediction tasks, narrow feature spaces, or short context windows, limiting their ability to model full patient pathways. To address this gap, we introduce EHR2Path, a multimodal framework for forecasting and simulating full in-hospital patient pathways from routine EHRs. EHR2Path converts diverse clinical inputs into a unified temporal representation, enabling modeling of a substantially broader set of patient information, including radiology reports, physician notes, vital signs, medication and laboratory patterns, and dense bedside charting. To support long clinical histories and broad feature spaces, we introduce a Masked Summarization Bottleneck that compresses long-term history into compact, task-optimized summary tokens while preserving recent context, improving both performance and token efficiency. In retrospective experiments on MIMIC-IV, EHR2Path enables next-step pathway forecasting and iterative simulation of complete in-hospital trajectories, while outperforming strong baselines on directly comparable tasks. These results establish a foundation for scalable pathway-level modeling from routine EHRs supporting anticipatory clinical decision-making. Our code is available at this https URL.
>
---
