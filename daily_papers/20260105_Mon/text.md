# 自然语言处理 cs.CL

- **最新发布 49 篇**

- **更新 43 篇**

## 最新发布

#### [new 001] Do LLMs Judge Distantly Supervised Named Entity Labels Well? Constructing the JudgeWEL Dataset
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于命名实体识别任务，旨在解决低资源语言数据不足的问题。通过自动标注并利用大语言模型验证，构建了更大的 Luxembourgish NER 数据集。**

- **链接: [https://arxiv.org/pdf/2601.00411v1](https://arxiv.org/pdf/2601.00411v1)**

> **作者:** Alistair Plum; Laura Bernardy; Tharindu Ranasinghe
>
> **摘要:** We present judgeWEL, a dataset for named entity recognition (NER) in Luxembourgish, automatically labelled and subsequently verified using large language models (LLM) in a novel pipeline. Building datasets for under-represented languages remains one of the major bottlenecks in natural language processing, where the scarcity of resources and linguistic particularities make large-scale annotation costly and potentially inconsistent. To address these challenges, we propose and evaluate a novel approach that leverages Wikipedia and Wikidata as structured sources of weak supervision. By exploiting internal links within Wikipedia articles, we infer entity types based on their corresponding Wikidata entries, thereby generating initial annotations with minimal human intervention. Because such links are not uniformly reliable, we mitigate noise by employing and comparing several LLMs to identify and retain only high-quality labelled sentences. The resulting corpus is approximately five times larger than the currently available Luxembourgish NER dataset and offers broader and more balanced coverage across entity categories, providing a substantial new resource for multilingual and low-resource NER research.
>
---
#### [new 002] Language as Mathematical Structure: Examining Semantic Field Theory Against Language Games
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，探讨语言的数学结构与语言游戏理论的关系，分析LLM如何体现语义规律，提出数学结构与社会互动可互补。**

- **链接: [https://arxiv.org/pdf/2601.00448v1](https://arxiv.org/pdf/2601.00448v1)**

> **作者:** Dimitris Vartziotis
>
> **摘要:** Large language models (LLMs) offer a new empirical setting in which long-standing theories of linguistic meaning can be examined. This paper contrasts two broad approaches: social constructivist accounts associated with language games, and a mathematically oriented framework we call Semantic Field Theory. Building on earlier work by the author, we formalize the notions of lexical fields (Lexfelder) and linguistic fields (Lingofelder) as interacting structures in a continuous semantic space. We then analyze how core properties of transformer architectures-such as distributed representations, attention mechanisms, and geometric regularities in embedding spaces-relate to these concepts. We argue that the success of LLMs in capturing semantic regularities supports the view that language exhibits an underlying mathematical structure, while their persistent limitations in pragmatic reasoning and context sensitivity are consistent with the importance of social grounding emphasized in philosophical accounts of language use. On this basis, we suggest that mathematical structure and language games can be understood as complementary rather than competing perspectives. The resulting framework clarifies the scope and limits of purely statistical models of language and motivates new directions for theoretically informed AI architectures.
>
---
#### [new 003] Toward Better Temporal Structures for Geopolitical Events Forecasting
- **分类: cs.CL**

- **简介: 该论文属于时空知识图谱预测任务，旨在解决HTKGs无法有效表示多实体复杂事实的问题。通过提出HTKGHs模型和相关数据集，提升 geopolitical 事件的预测能力。**

- **链接: [https://arxiv.org/pdf/2601.00430v1](https://arxiv.org/pdf/2601.00430v1)**

> **作者:** Kian Ahrabian; Eric Boxer; Jay Pujara
>
> **备注:** 17 pages, 13 figures, 3 tables
>
> **摘要:** Forecasting on geopolitical temporal knowledge graphs (TKGs) through the lens of large language models (LLMs) has recently gained traction. While TKGs and their generalization, hyper-relational temporal knowledge graphs (HTKGs), offer a straightforward structure to represent simple temporal relationships, they lack the expressive power to convey complex facts efficiently. One of the critical limitations of HTKGs is a lack of support for more than two primary entities in temporal facts, which commonly occur in real-world events. To address this limitation, in this work, we study a generalization of HTKGs, Hyper-Relational Temporal Knowledge Generalized Hypergraphs (HTKGHs). We first derive a formalization for HTKGHs, demonstrating their backward compatibility while supporting two complex types of facts commonly found in geopolitical incidents. Then, utilizing this formalization, we introduce the htkgh-polecat dataset, built upon the global event database POLECAT. Finally, we benchmark and analyze popular LLMs on the relation prediction task, providing insights into their adaptability and capabilities in complex forecasting scenarios.
>
---
#### [new 004] Retrieval--Reasoning Processes for Multi-hop Question Answering: A Four-Axis Design Framework and Empirical Trends
- **分类: cs.CL**

- **简介: 该论文属于多跳问答任务，旨在解决系统在多步检索与推理中的过程设计问题。提出四轴框架，分析不同系统的执行策略与效果，总结优化方向。**

- **链接: [https://arxiv.org/pdf/2601.00536v1](https://arxiv.org/pdf/2601.00536v1)**

> **作者:** Yuelyu Ji; Zhuochun Li; Rui Meng; Daqing He
>
> **摘要:** Multi-hop question answering (QA) requires systems to iteratively retrieve evidence and reason across multiple hops. While recent RAG and agentic methods report strong results, the underlying retrieval--reasoning \emph{process} is often left implicit, making procedural choices hard to compare across model families. This survey takes the execution procedure as the unit of analysis and introduces a four-axis framework covering (A) overall execution plan, (B) index structure, (C) next-step control (strategies and triggers), and (D) stop/continue criteria. Using this schema, we map representative multi-hop QA systems and synthesize reported ablations and tendencies on standard benchmarks (e.g., HotpotQA, 2WikiMultiHopQA, MuSiQue), highlighting recurring trade-offs among effectiveness, efficiency, and evidence faithfulness. We conclude with open challenges for retrieval--reasoning agents, including structure-aware planning, transferable control policies, and robust stopping under distribution shift.
>
---
#### [new 005] Beyond IVR: Benchmarking Customer Support LLM Agents for Business-Adherence
- **分类: cs.CL**

- **简介: 该论文属于客户支持AI评估任务，旨在解决LLM代理在遵循业务规则方面的不足。提出JourneyBench基准和用户旅程覆盖率指标，评估不同代理的政策遵循能力。**

- **链接: [https://arxiv.org/pdf/2601.00596v1](https://arxiv.org/pdf/2601.00596v1)**

> **作者:** Sumanth Balaji; Piyush Mishra; Aashraya Sachdeva; Suraj Agrawal
>
> **备注:** 17 pages, 3 figures, preprint
>
> **摘要:** Traditional customer support systems, such as Interactive Voice Response (IVR), rely on rigid scripts and lack the flexibility required for handling complex, policy-driven tasks. While large language model (LLM) agents offer a promising alternative, evaluating their ability to act in accordance with business rules and real-world support workflows remains an open challenge. Existing benchmarks primarily focus on tool usage or task completion, overlooking an agent's capacity to adhere to multi-step policies, navigate task dependencies, and remain robust to unpredictable user or environment behavior. In this work, we introduce JourneyBench, a benchmark designed to assess policy-aware agents in customer support. JourneyBench leverages graph representations to generate diverse, realistic support scenarios and proposes the User Journey Coverage Score, a novel metric to measure policy adherence. We evaluate multiple state-of-the-art LLMs using two agent designs: a Static-Prompt Agent (SPA) and a Dynamic-Prompt Agent (DPA) that explicitly models policy control. Across 703 conversations in three domains, we show that DPA significantly boosts policy adherence, even allowing smaller models like GPT-4o-mini to outperform more capable ones like GPT-4o. Our findings demonstrate the importance of structured orchestration and establish JourneyBench as a critical resource to advance AI-driven customer support beyond IVR-era limitations.
>
---
#### [new 006] Talk Less, Verify More: Improving LLM Assistants with Semantic Checks and Execution Feedback
- **分类: cs.CL; cs.SE**

- **简介: 该论文属于生成式AI可靠性提升任务，解决LLM助手输出准确性不足的问题，提出Q*和Feedback+验证技术以提高结果的语义一致性和可执行性。**

- **链接: [https://arxiv.org/pdf/2601.00224v1](https://arxiv.org/pdf/2601.00224v1)**

> **作者:** Yan Sun; Ming Cai; Stanley Kok
>
> **摘要:** As large language model (LLM) assistants become increasingly integrated into enterprise workflows, their ability to generate accurate, semantically aligned, and executable outputs is critical. However, current conversational business analytics (CBA) systems often lack built-in verification mechanisms, leaving users to manually validate potentially flawed results. This paper introduces two complementary verification techniques: Q*, which performs reverse translation and semantic matching between code and user intent, and Feedback+, which incorporates execution feedback to guide code refinement. Embedded within a generator-discriminator framework, these mechanisms shift validation responsibilities from users to the system. Evaluations on three benchmark datasets, Spider, Bird, and GSM8K, demonstrate that both Q* and Feedback+ reduce error rates and task completion time. The study also identifies reverse translation as a key bottleneck, highlighting opportunities for future improvement. Overall, this work contributes a design-oriented framework for building more reliable, enterprise-grade GenAI systems capable of trustworthy decision support.
>
---
#### [new 007] Robust Uncertainty Quantification for Factual Generation of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型事实生成中的不确定性量化任务，旨在解决模型幻觉问题。通过构建包含虚假名称的陷阱问题，提出一种新的鲁棒不确定性量化方法，提升模型在复杂场景下的可靠性。**

- **链接: [https://arxiv.org/pdf/2601.00348v1](https://arxiv.org/pdf/2601.00348v1)**

> **作者:** Yuhao Zhang; Zhongliang Yang; Linna Zhou
>
> **备注:** 9 pages, 5 tables, 5 figures, accepted to IJCNN 2025
>
> **摘要:** The rapid advancement of large language model(LLM) technology has facilitated its integration into various domains of professional and daily life. However, the persistent challenge of LLM hallucination has emerged as a critical limitation, significantly compromising the reliability and trustworthiness of AI-generated content. This challenge has garnered significant attention within the scientific community, prompting extensive research efforts in hallucination detection and mitigation strategies. Current methodological frameworks reveal a critical limitation: traditional uncertainty quantification approaches demonstrate effectiveness primarily within conventional question-answering paradigms, yet exhibit notable deficiencies when confronted with non-canonical or adversarial questioning strategies. This performance gap raises substantial concerns regarding the dependability of LLM responses in real-world applications requiring robust critical thinking capabilities. This study aims to fill this gap by proposing an uncertainty quantification scenario in the task of generating with multiple facts. We have meticulously constructed a set of trap questions contained with fake names. Based on this scenario, we innovatively propose a novel and robust uncertainty quantification method(RU). A series of experiments have been conducted to verify its effectiveness. The results show that the constructed set of trap questions performs excellently. Moreover, when compared with the baseline methods on four different models, our proposed method has demonstrated great performance, with an average increase of 0.1-0.2 in ROCAUC values compared to the best performing baseline method, providing new sights and methods for addressing the hallucination issue of LLMs.
>
---
#### [new 008] Understanding Emotion in Discourse: Recognition Insights and Linguistic Patterns for Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于情感识别任务，解决模型架构选择与情感表达语言模式的不足。通过分析IEMOCAP数据集，提出有效架构并发现情感与话语标记位置的关联。**

- **链接: [https://arxiv.org/pdf/2601.00181v1](https://arxiv.org/pdf/2601.00181v1)**

> **作者:** Cheonkam Jeong; Adeline Nyamathi
>
> **摘要:** While Emotion Recognition in Conversation (ERC) has achieved high accuracy, two critical gaps remain: a limited understanding of \textit{which} architectural choices actually matter, and a lack of linguistic analysis connecting recognition to generation. We address both gaps through a systematic analysis of the IEMOCAP dataset. For recognition, we conduct a rigorous ablation study with 10-seed evaluation and report three key findings. First, conversational context is paramount, with performance saturating rapidly -- 90\% of the total gain achieved within just the most recent 10--30 preceding turns (depending on the label set). Second, hierarchical sentence representations help at utterance-level, but this benefit disappears once conversational context is provided, suggesting that context subsumes intra-utterance structure. Third, external affective lexicons (SenticNet) provide no gain, indicating that pre-trained encoders already capture necessary emotional semantics. With simple architectures using strictly causal context, we achieve 82.69\% (4-way) and 67.07\% (6-way) weighted F1, outperforming prior text-only methods including those using bidirectional context. For linguistic analysis, we analyze 5,286 discourse marker occurrences and find a significant association between emotion and marker positioning ($p < .0001$). Notably, "sad" utterances exhibit reduced left-periphery marker usage (21.9\%) compared to other emotions (28--32\%), consistent with theories linking left-periphery markers to active discourse management. This connects to our recognition finding that sadness benefits most from context (+22\%p): lacking explicit pragmatic signals, sad utterances require conversational history for disambiguation.
>
---
#### [new 009] From Evidence-Based Medicine to Knowledge Graph: Retrieval-Augmented Generation for Sports Rehabilitation and a Domain Benchmark
- **分类: cs.CL**

- **简介: 该论文属于医疗信息检索任务，解决RAG系统在体育康复中证据不匹配和证据等级不足的问题，提出基于EBM的图检索方法并构建基准数据集。**

- **链接: [https://arxiv.org/pdf/2601.00216v1](https://arxiv.org/pdf/2601.00216v1)**

> **作者:** Jinning Zhang; Jie Song; Wenhui Tu; Zecheng Li; Jingxuan Li; Jin Li; Xuan Liu; Taole Sha; Zichen Wei; Yan Li
>
> **备注:** 35 pages, 5 figures
>
> **摘要:** In medicine, large language models (LLMs) increasingly rely on retrieval-augmented generation (RAG) to ground outputs in up-to-date external evidence. However, current RAG approaches focus primarily on performance improvements while overlooking evidence-based medicine (EBM) principles. This study addresses two key gaps: (1) the lack of PICO alignment between queries and retrieved evidence, and (2) the absence of evidence hierarchy considerations during reranking. We present a generalizable strategy for adapting EBM to graph-based RAG, integrating the PICO framework into knowledge graph construction and retrieval, and proposing a Bayesian-inspired reranking algorithm to calibrate ranking scores by evidence grade without introducing predefined weights. We validated this framework in sports rehabilitation, a literature-rich domain currently lacking RAG systems and benchmarks. We released a knowledge graph (357,844 nodes and 371,226 edges) and a reusable benchmark of 1,637 QA pairs. The system achieved 0.830 nugget coverage, 0.819 answer faithfulness, 0.882 semantic similarity, and 0.788 PICOT match accuracy. In a 5-point Likert evaluation, five expert clinicians rated the system 4.66-4.84 across factual accuracy, faithfulness, relevance, safety, and PICO alignment. These findings demonstrate that the proposed EBM adaptation strategy improves retrieval and answer quality and is transferable to other clinical domains. The released resources also help address the scarcity of RAG datasets in sports rehabilitation.
>
---
#### [new 010] Rule-Based Approaches to Atomic Sentence Extraction
- **分类: cs.CL**

- **简介: 该论文属于原子句提取任务，旨在解决复杂句子分解问题。通过规则方法分析句法结构对提取效果的影响，使用spaCy生成标准数据集并评估性能。**

- **链接: [https://arxiv.org/pdf/2601.00506v1](https://arxiv.org/pdf/2601.00506v1)**

> **作者:** Lineesha Kamana; Akshita Ananda Subramanian; Mehuli Ghosh; Suman Saha
>
> **摘要:** Natural language often combines multiple ideas into complex sentences. Atomic sentence extraction, the task of decomposing complex sentences into simpler sentences that each express a single idea, improves performance in information retrieval, question answering, and automated reasoning systems. Previous work has formalized the "split-and-rephrase" task and established evaluation metrics, and machine learning approaches using large language models have improved extraction accuracy. However, these methods lack interpretability and provide limited insight into which linguistic structures cause extraction failures. Although some studies have explored dependency-based extraction of subject-verb-object triples and clauses, no principled analysis has examined which specific clause structures and dependencies lead to extraction difficulties. This study addresses this gap by analyzing how complex sentence structures, including relative clauses, adverbial clauses, coordination patterns, and passive constructions, affect the performance of rule-based atomic sentence extraction. Using the WikiSplit dataset, we implemented dependency-based extraction rules in spaCy, generated 100 gold=standard atomic sentence sets, and evaluated performance using ROUGE and BERTScore. The system achieved ROUGE-1 F1 = 0.6714, ROUGE-2 F1 = 0.478, ROUGE-L F1 = 0.650, and BERTScore F1 = 0.5898, indicating moderate-to-high lexical, structural, and semantic alignment. Challenging structures included relative clauses, appositions, coordinated predicates, adverbial clauses, and passive constructions. Overall, rule-based extraction is reasonably accurate but sensitive to syntactic complexity.
>
---
#### [new 011] Fast-weight Product Key Memory
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出FwPKM，解决语言模型中存储与效率的矛盾，通过动态更新机制提升长序列记忆能力，显著降低困惑度。**

- **链接: [https://arxiv.org/pdf/2601.00671v1](https://arxiv.org/pdf/2601.00671v1)**

> **作者:** Tianyu Zhao; Llion Jones
>
> **摘要:** Sequence modeling layers in modern language models typically face a trade-off between storage capacity and computational efficiency. While Softmax attention offers unbounded storage at prohibitive quadratic costs, linear variants provide efficiency but suffer from limited, fixed-size storage. We propose Fast-weight Product Key Memory (FwPKM), a novel architecture that resolves this tension by transforming the sparse Product Key Memory (PKM) from a static module into a dynamic, "fast-weight" episodic memory. Unlike PKM, FwPKM updates its parameters dynamically at both training and inference time via local chunk-level gradient descent, allowing the model to rapidly memorize and retrieve new key-value pairs from input sequences. Experiments reveal that FwPKM functions as an effective episodic memory that complements the semantic memory of standard modules, yielding significant perplexity reductions on long-context datasets. Notably, in Needle in a Haystack evaluations, FwPKM generalizes to 128K-token contexts despite being trained on only 4K-token sequences.
>
---
#### [new 012] JP-TL-Bench: Anchored Pairwise LLM Evaluation for Bidirectional Japanese-English Translation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出JP-TL-Bench，用于评估日英双向翻译系统的性能。解决“哪个翻译更好”的问题，通过无参考的成对比较，提升评估的可靠性和成本效益。**

- **链接: [https://arxiv.org/pdf/2601.00223v1](https://arxiv.org/pdf/2601.00223v1)**

> **作者:** Leonard Lin; Adam Lensenmayer
>
> **备注:** 24 pages, 5 figures, 8 tables
>
> **摘要:** We introduce JP-TL-Bench, a lightweight, open benchmark designed to guide the iterative development of Japanese-English translation systems. In this context, the challenge is often "which of these two good translations is better?" rather than "is this translation acceptable?" This distinction matters for Japanese-English, where subtle choices in politeness, implicature, ellipsis, and register strongly affect perceived naturalness. JP-TL-Bench uses a protocol built to make LLM judging both reliable and affordable: it evaluates a candidate model via reference-free, pairwise LLM comparisons against a fixed, versioned anchor set. Pairwise results are aggregated with a Bradley-Terry model and reported as win rates plus a normalized 0-10 "LT" score derived from a logistic transform of fitted log-strengths. Because each candidate is scored against the same frozen anchor set, scores are structurally stable given the same base set, judge, and aggregation code.
>
---
#### [new 013] Pat-DEVAL: Chain-of-Legal-Thought Evaluation for Patent Description
- **分类: cs.CL**

- **简介: 该论文提出Pat-DEVAL框架，用于评估专利描述的法律合规性和结构连贯性。针对自动化专利撰写中的法律标准评估问题，引入法律思维链机制，提升评估准确性。**

- **链接: [https://arxiv.org/pdf/2601.00166v1](https://arxiv.org/pdf/2601.00166v1)**

> **作者:** Yongmin Yoo; Kris W Pan
>
> **摘要:** Patent descriptions must deliver comprehensive technical disclosure while meeting strict legal standards such as enablement and written description requirements. Although large language models have enabled end-to-end automated patent drafting, existing evaluation approaches fail to assess long-form structural coherence and statutory compliance specific to descriptions. We propose Pat-DEVAL, the first multi-dimensional evaluation framework dedicated to patent description bodies. Leveraging the LLM-as-a-judge paradigm, Pat-DEVAL introduces Chain-of-Legal-Thought (CoLT), a legally-constrained reasoning mechanism that enforces sequential patent-law-specific analysis. Experiments validated by patent expert on our Pap2Pat-EvalGold dataset demonstrate that Pat-DEVAL achieves a Pearson correlation of 0.69, significantly outperforming baseline metrics and existing LLM evaluators. Notably, the framework exhibits a superior correlation of 0.73 in Legal-Professional Compliance, proving that the explicit injection of statutory constraints is essential for capturing nuanced legal validity. By establishing a new standard for ensuring both technical soundness and legal compliance, Pat-DEVAL provides a robust methodological foundation for the practical deployment of automated patent drafting systems.
>
---
#### [new 014] A Language-Agnostic Hierarchical LoRA-MoE Architecture for CTC-based Multilingual ASR
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于多语言语音识别任务，旨在解决大模型计算成本高、难以部署在边缘设备的问题。提出HLoRA框架，实现高效、语言无关的单次解码。**

- **链接: [https://arxiv.org/pdf/2601.00557v1](https://arxiv.org/pdf/2601.00557v1)**

> **作者:** Yuang Zheng; Yuxiang Mei; Dongxing Xu; Jie Chen; Yanhua Long
>
> **备注:** 5 pages, submitted to IEEE Signal Processing Letters
>
> **摘要:** Large-scale multilingual ASR (mASR) models such as Whisper achieve strong performance but incur high computational and latency costs, limiting their deployment on resource-constrained edge devices. In this study, we propose a lightweight and language-agnostic multilingual ASR system based on a CTC architecture with domain adaptation. Specifically, we introduce a Language-agnostic Hierarchical LoRA-MoE (HLoRA) framework integrated into an mHuBERT-CTC model, enabling end-to-end decoding via LID-posterior-driven LoRA routing. The hierarchical design consists of a multilingual shared LoRA for learning language-invariant acoustic representations and language-specific LoRA experts for modeling language-dependent characteristics. The proposed routing mechanism removes the need for prior language identity information or explicit language labels during inference, achieving true language-agnostic decoding. Experiments on MSR-86K and the MLC-SLM 2025 Challenge datasets demonstrate that HLoRA achieves competitive performance with state-of-the-art two-stage inference methods using only single-pass decoding, significantly improving decoding efficiency for low-resource mASR applications.
>
---
#### [new 015] The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，研究多语言预训练中混合语言文档的作用。旨在解决 bilingual data 对模型跨语言能力影响的问题，通过实验分析发现平行文本对翻译至关重要，而代码切换影响较小。**

- **链接: [https://arxiv.org/pdf/2601.00364v1](https://arxiv.org/pdf/2601.00364v1)**

> **作者:** Jiandong Shao; Raphael Tang; Crystina Zhang; Karin Sevegnani; Pontus Stenetorp; Jianfei Yang; Yao Lu
>
> **备注:** under review
>
> **摘要:** Multilingual large language models achieve impressive cross-lingual performance despite largely monolingual pretraining. While bilingual data in pretraining corpora is widely believed to enable these abilities, details of its contributions remain unclear. We investigate this question by pretraining models from scratch under controlled conditions, comparing the standard web corpus with a monolingual-only version that removes all multilingual documents. Despite constituting only 2% of the corpus, removing bilingual data causes translation performance to drop 56% in BLEU, while behaviour on cross-lingual QA and general reasoning tasks remains stable, with training curves largely overlapping the baseline. To understand this asymmetry, we categorize bilingual data into parallel (14%), code-switching (72%), and miscellaneous documents (14%) based on the semantic relevance of content in different languages. We then conduct granular ablations by reintroducing parallel or code-switching data into the monolingual-only corpus. Our experiments reveal that parallel data almost fully restores translation performance (91% of the unfiltered baseline), whereas code-switching contributes minimally. Other cross-lingual tasks remain largely unaffected by either type. These findings reveal that translation critically depends on systematic token-level alignments from parallel data, whereas cross-lingual understanding and reasoning appear to be achievable even without bilingual data.
>
---
#### [new 016] Noise-Aware Named Entity Recognition for Historical VET Documents
- **分类: cs.CL; cs.IR; cs.LG**

- **简介: 该论文属于命名实体识别任务，解决历史VET文档中OCR噪声带来的识别问题。通过噪声感知训练、迁移学习和多阶段微调提升模型鲁棒性与准确性。**

- **链接: [https://arxiv.org/pdf/2601.00488v1](https://arxiv.org/pdf/2601.00488v1)**

> **作者:** Alexander M. Esser; Jens Dörpinghaus
>
> **备注:** This is an extended, non-peer-reviewed version of the paper presented at VISAPP 2026
>
> **摘要:** This paper addresses Named Entity Recognition (NER) in the domain of Vocational Education and Training (VET), focusing on historical, digitized documents that suffer from OCR-induced noise. We propose a robust NER approach leveraging Noise-Aware Training (NAT) with synthetically injected OCR errors, transfer learning, and multi-stage fine-tuning. Three complementary strategies, training on noisy, clean, and artificial data, are systematically compared. Our method is one of the first to recognize multiple entity types in VET documents. It is applied to German documents but transferable to arbitrary languages. Experimental results demonstrate that domain-specific and noise-aware fine-tuning substantially increases robustness and accuracy under noisy conditions. We provide publicly available code for reproducible noise-aware NER in domain-specific contexts.
>
---
#### [new 017] Universal Adaptive Constraint Propagation: Scaling Structured Inference for Large Language Models via Meta-Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文提出MetaJuLS，解决大语言模型中结构化推理的约束传播问题，通过元强化学习实现跨语言任务快速适应。**

- **链接: [https://arxiv.org/pdf/2601.00095v1](https://arxiv.org/pdf/2601.00095v1)**

> **作者:** Ibne Farabi Shihab; Sanjeda Akter; Anuj Sharma
>
> **摘要:** Large language models increasingly require structured inference, from JSON schema enforcement to multi-lingual parsing, where outputs must satisfy complex constraints. We introduce MetaJuLS, a meta-reinforcement learning approach that learns universal constraint propagation policies applicable across languages and tasks without task-specific retraining. By formulating structured inference as adaptive constraint propagation and training a Graph Attention Network with meta-learning, MetaJuLS achieves 1.5--2.0$\times$ speedups over GPU-optimized baselines while maintaining within 0.2\% accuracy of state-of-the-art parsers. On Universal Dependencies across 10 languages and LLM-constrained generation (LogicBench, GSM8K-Constrained), MetaJuLS demonstrates rapid cross-domain adaptation: a policy trained on English parsing adapts to new languages and tasks with 5--10 gradient steps (5--15 seconds) rather than requiring hours of task-specific training. Mechanistic analysis reveals the policy discovers human-like parsing strategies (easy-first) and novel non-intuitive heuristics. By reducing propagation steps in LLM deployments, MetaJuLS contributes to Green AI by directly reducing inference carbon footprint.
>
---
#### [new 018] Parallel Universes, Parallel Languages: A Comprehensive Study on LLM-based Multilingual Counterfactual Example Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究多语言对抗样本生成任务，旨在提升模型解释性和鲁棒性。通过对比直接生成与翻译生成方法，分析错误类型并验证多语言数据增强效果。**

- **链接: [https://arxiv.org/pdf/2601.00263v1](https://arxiv.org/pdf/2601.00263v1)**

> **作者:** Qianli Wang; Van Bach Nguyen; Yihong Liu; Fedor Splitt; Nils Feldhus; Christin Seifert; Hinrich Schütze; Sebastian Möller; Vera Schmitt
>
> **备注:** In submission
>
> **摘要:** Counterfactuals refer to minimally edited inputs that cause a model's prediction to change, serving as a promising approach to explaining the model's behavior. Large language models (LLMs) excel at generating English counterfactuals and demonstrate multilingual proficiency. However, their effectiveness in generating multilingual counterfactuals remains unclear. To this end, we conduct a comprehensive study on multilingual counterfactuals. We first conduct automatic evaluations on both directly generated counterfactuals in the target languages and those derived via English translation across six languages. Although translation-based counterfactuals offer higher validity than their directly generated counterparts, they demand substantially more modifications and still fall short of matching the quality of the original English counterfactuals. Second, we find the patterns of edits applied to high-resource European-language counterfactuals to be remarkably similar, suggesting that cross-lingual perturbations follow common strategic principles. Third, we identify and categorize four main types of errors that consistently appear in the generated counterfactuals across languages. Finally, we reveal that multilingual counterfactual data augmentation (CDA) yields larger model performance improvements than cross-lingual CDA, especially for lower-resource languages. Yet, the imperfections of the generated counterfactuals limit gains in model performance and robustness.
>
---
#### [new 019] DepFlow: Disentangled Speech Generation to Mitigate Semantic Bias in Depression Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于抑郁症检测任务，旨在解决语义偏见问题。通过提出DepFlow框架，实现语音生成与情感分离，提升模型在伪装抑郁场景下的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.00303v1](https://arxiv.org/pdf/2601.00303v1)**

> **作者:** Yuxin Li; Xiangyu Zhang; Yifei Li; Zhiwei Guo; Haoyang Zhang; Eng Siong Chng; Cuntai Guan
>
> **摘要:** Speech is a scalable and non-invasive biomarker for early mental health screening. However, widely used depression datasets like DAIC-WOZ exhibit strong coupling between linguistic sentiment and diagnostic labels, encouraging models to learn semantic shortcuts. As a result, model robustness may be compromised in real-world scenarios, such as Camouflaged Depression, where individuals maintain socially positive or neutral language despite underlying depressive states. To mitigate this semantic bias, we propose DepFlow, a three-stage depression-conditioned text-to-speech framework. First, a Depression Acoustic Encoder learns speaker- and content-invariant depression embeddings through adversarial training, achieving effective disentanglement while preserving depression discriminability (ROC-AUC: 0.693). Second, a flow-matching TTS model with FiLM modulation injects these embeddings into synthesis, enabling control over depressive severity while preserving content and speaker identity. Third, a prototype-based severity mapping mechanism provides smooth and interpretable manipulation across the depression continuum. Using DepFlow, we construct a Camouflage Depression-oriented Augmentation (CDoA) dataset that pairs depressed acoustic patterns with positive/neutral content from a sentiment-stratified text bank, creating acoustic-semantic mismatches underrepresented in natural data. Evaluated across three depression detection architectures, CDoA improves macro-F1 by 9%, 12%, and 5%, respectively, consistently outperforming conventional augmentation strategies in depression Detection. Beyond enhancing robustness, DepFlow provides a controllable synthesis platform for conversational systems and simulation-based evaluation, where real clinical data remains limited by ethical and coverage constraints.
>
---
#### [new 020] Adapting Natural Language Processing Models Across Jurisdictions: A pilot Study in Canadian Cancer Registries
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理在癌症登记中的应用任务，旨在解决跨地区模型泛化问题。通过微调和模型集成，提升癌症数据抽取的准确性与覆盖率。**

- **链接: [https://arxiv.org/pdf/2601.00787v1](https://arxiv.org/pdf/2601.00787v1)**

> **作者:** Jonathan Simkin; Lovedeep Gondara; Zeeshan Rizvi; Gregory Doyle; Jeff Dowden; Dan Bond; Desmond Martin; Raymond Ng
>
> **摘要:** Population-based cancer registries depend on pathology reports as their primary diagnostic source, yet manual abstraction is resource-intensive and contributes to delays in cancer data. While transformer-based NLP systems have improved registry workflows, their ability to generalize across jurisdictions with differing reporting conventions remains poorly understood. We present the first cross-provincial evaluation of adapting BCCRTron, a domain-adapted transformer model developed at the British Columbia Cancer Registry, alongside GatorTron, a biomedical transformer model, for cancer surveillance in Canada. Our training dataset consisted of approximately 104,000 and 22,000 de-identified pathology reports from the Newfoundland & Labrador Cancer Registry (NLCR) for Tier 1 (cancer vs. non-cancer) and Tier 2 (reportable vs. non-reportable) tasks, respectively. Both models were fine-tuned using complementary synoptic and diagnosis focused report section input pipelines. Across NLCR test sets, the adapted models maintained high performance, demonstrating transformers pretrained in one jurisdiction can be localized to another with modest fine-tuning. To improve sensitivity, we combined the two models using a conservative OR-ensemble achieving a Tier 1 recall of 0.99 and reduced missed cancers to 24, compared with 48 and 54 for the standalone models. For Tier 2, the ensemble achieved 0.99 recall and reduced missed reportable cancers to 33, compared with 54 and 46 for the individual models. These findings demonstrate that an ensemble combining complementary text representations substantially reduce missed cancers and improve error coverage in cancer-registry NLP. We implement a privacy-preserving workflow in which only model weights are shared between provinces, supporting interoperable NLP infrastructure and a future pan-Canadian foundation model for cancer pathology and registry workflows.
>
---
#### [new 021] Beyond Perfect APIs: A Comprehensive Evaluation of LLM Agents Under Real-World API Complexity
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于评估任务，旨在解决LLM代理在真实API复杂环境下的表现问题。工作包括构建WildAGTEval基准，测试并分析LLM在多种复杂场景中的能力。**

- **链接: [https://arxiv.org/pdf/2601.00268v1](https://arxiv.org/pdf/2601.00268v1)**

> **作者:** Doyoung Kim; Zhiwei Ren; Jie Hao; Zhongkai Sun; Lichao Wang; Xiyao Ma; Zack Ye; Xu Han; Jun Yin; Heng Ji; Wei Shen; Xing Fan; Benjamin Yao; Chenlei Guo
>
> **备注:** 26 pages
>
> **摘要:** We introduce WildAGTEval, a benchmark designed to evaluate large language model (LLM) agents' function-calling capabilities under realistic API complexity. Unlike prior work that assumes an idealized API system and disregards real-world factors such as noisy API outputs, WildAGTEval accounts for two dimensions of real-world complexity: 1. API specification, which includes detailed documentation and usage constraints, and 2. API execution, which captures runtime challenges. Consequently, WildAGTEval offers (i) an API system encompassing 60 distinct complexity scenarios that can be composed into approximately 32K test configurations, and (ii) user-agent interactions for evaluating LLM agents on these scenarios. Using WildAGTEval, we systematically assess several advanced LLMs and observe that most scenarios are challenging, with irrelevant information complexity posing the greatest difficulty and reducing the performance of strong LLMs by 27.3%. Furthermore, our qualitative analysis reveals that LLMs occasionally distort user intent merely to claim task completion, critically affecting user satisfaction.
>
---
#### [new 022] CSSBench: Evaluating the Safety of Lightweight LLMs against Chinese-Specific Adversarial Patterns
- **分类: cs.CL**

- **简介: 该论文属于安全评估任务，旨在解决轻量级LLM在中文场景下的安全漏洞问题。针对中文特有的攻击模式，构建了CSSBench基准，评估模型安全性。**

- **链接: [https://arxiv.org/pdf/2601.00588v1](https://arxiv.org/pdf/2601.00588v1)**

> **作者:** Zhenhong Zhou; Shilinlu Yan; Chuanpu Liu; Qiankun Li; Kun Wang; Zhigang Zeng
>
> **备注:** 18 pages
>
> **摘要:** Large language models (LLMs) are increasingly deployed in cost-sensitive and on-device scenarios, and safety guardrails have advanced mainly in English. However, real-world Chinese malicious queries typically conceal intent via homophones, pinyin, symbol-based splitting, and other Chinese-specific patterns. These Chinese-specific adversarial patterns create the safety evaluation gap that is not well captured by existing benchmarks focused on English. This gap is particularly concerning for lightweight models, which may be more vulnerable to such specific adversarial perturbations. To bridge this gap, we introduce the Chinese-Specific Safety Benchmark (CSSBench) that emphasizes these adversarial patterns and evaluates the safety of lightweight LLMs in Chinese. Our benchmark covers six domains that are common in real Chinese scenarios, including illegal activities and compliance, privacy leakage, health and medical misinformation, fraud and hate, adult content, and public and political safety, and organizes queries into multiple task types. We evaluate a set of popular lightweight LLMs and measure over-refusal behavior to assess safety-induced performance degradation. Our results show that the Chinese-specific adversarial pattern is a critical challenge for lightweight LLMs. This benchmark offers a comprehensive evaluation of LLM safety in Chinese, assisting robust deployments in practice.
>
---
#### [new 023] RIMRULE: Improving Tool-Using Language Agents via MDL-Guided Rule Learning
- **分类: cs.CL**

- **简介: 该论文属于语言模型工具使用任务，解决LLM在特定领域工具使用不可靠的问题。提出RIMRULE方法，通过MDL引导规则学习提升性能。**

- **链接: [https://arxiv.org/pdf/2601.00086v1](https://arxiv.org/pdf/2601.00086v1)**

> **作者:** Xiang Gao; Yuguang Yao; Qi Zhang; Kaiwen Dong; Avinash Baidya; Ruocheng Guo; Hilaf Hasson; Kamalika Das
>
> **摘要:** Large language models (LLMs) often struggle to use tools reliably in domain-specific settings, where APIs may be idiosyncratic, under-documented, or tailored to private workflows. This highlights the need for effective adaptation to task-specific tools. We propose RIMRULE, a neuro-symbolic approach for LLM adaptation based on dynamic rule injection. Compact, interpretable rules are distilled from failure traces and injected into the prompt during inference to improve task performance. These rules are proposed by the LLM itself and consolidated using a Minimum Description Length (MDL) objective that favors generality and conciseness. Each rule is stored in both natural language and a structured symbolic form, supporting efficient retrieval at inference time. Experiments on tool-use benchmarks show that this approach improves accuracy on both seen and unseen tools without modifying LLM weights. It outperforms prompting-based adaptation methods and complements finetuning. Moreover, rules learned from one LLM can be reused to improve others, including long reasoning LLMs, highlighting the portability of symbolic knowledge across architectures.
>
---
#### [new 024] ECR: Manifold-Guided Semantic Cues for Compact Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型压缩任务，解决紧凑模型嵌入空间结构丢失问题。提出ECR框架，通过语义锚点保持几何一致性，提升模型性能与稳定性。**

- **链接: [https://arxiv.org/pdf/2601.00543v1](https://arxiv.org/pdf/2601.00543v1)**

> **作者:** Chung-Wei Victor Yuan
>
> **备注:** Preprint 13pages, 6 figures
>
> **摘要:** Compact models often lose the structure of their embedding space. The issue shows up when the capacity is tight or the data spans several languages. Such collapse makes it difficult for downstream tasks to build on the resulting representation. Existing compression methods focus on aligning model outputs at a superficial level but fail to preserve the underlying manifold structure. This mismatch often leads to semantic drift in the compact model, causing both task behavior and linguistic properties to deviate from the reference model. To address those issues, we provide a new framework called Embedding Consistency Regulation (ECR). This framework first derives a set of semantic anchors from teacher embeddings (computed once offline). Then, the compact model learns to maintain consistent geometry around these anchors, without relying on matching logits or internal features. ECR adds only a small projection step at inference, without altering the decoding architecture or its runtime behavior. In experiments on a 100K multilingual corpus, ECR consistently stabilizes training and preserves semantic structure across tasks and languages. It also produces a more compact and task-aligned representation space, enabling low-capacity models to learn cleaner manifolds than conventional baselines. ECR works without teacher outputs and is compatible with, but independent of, distillation. Taken together, our results show that ECR helps compact models better follow task requirements and makes them easier to deploy under strict efficiency or privacy limits.
>
---
#### [new 025] Knowledge Distillation for Temporal Knowledge Graph Reasoning with Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于时间知识图谱推理任务，旨在解决现有模型参数大、计算成本高及无法有效捕捉时间依赖的问题。通过知识蒸馏，利用大语言模型提升轻量模型的推理能力。**

- **链接: [https://arxiv.org/pdf/2601.00202v1](https://arxiv.org/pdf/2601.00202v1)**

> **作者:** Wang Xing; Wei Song; Siyu Lin; Chen Wu; Zhesi Li; Man Wang
>
> **摘要:** Reasoning over temporal knowledge graphs (TKGs) is fundamental to improving the efficiency and reliability of intelligent decision-making systems and has become a key technological foundation for future artificial intelligence applications. Despite recent progress, existing TKG reasoning models typically rely on large parameter sizes and intensive computation, leading to high hardware costs and energy consumption. These constraints hinder their deployment on resource-constrained, low-power, and distributed platforms that require real-time inference. Moreover, most existing model compression and distillation techniques are designed for static knowledge graphs and fail to adequately capture the temporal dependencies inherent in TKGs, often resulting in degraded reasoning performance. To address these challenges, we propose a distillation framework specifically tailored for temporal knowledge graph reasoning. Our approach leverages large language models as teacher models to guide the distillation process, enabling effective transfer of both structural and temporal reasoning capabilities to lightweight student models. By integrating large-scale public knowledge with task-specific temporal information, the proposed framework enhances the student model's ability to model temporal dynamics while maintaining a compact and efficient architecture. Extensive experiments on multiple publicly available benchmark datasets demonstrate that our method consistently outperforms strong baselines, achieving a favorable trade-off between reasoning accuracy, computational efficiency, and practical deployability.
>
---
#### [new 026] BERT-JEPA: Reorganizing CLS Embeddings for Language-Invariant Semantics
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决[CLS]嵌入空间退化问题，通过引入JEPA训练目标，提升模型的跨语言语义能力。**

- **链接: [https://arxiv.org/pdf/2601.00366v1](https://arxiv.org/pdf/2601.00366v1)**

> **作者:** Taj Gillin; Adam Lalani; Kenneth Zhang; Marcel Mateos Salles
>
> **备注:** 16 pages, 10 figures, 10 tables
>
> **摘要:** Joint Embedding Predictive Architectures (JEPA) are a novel self supervised training technique that have shown recent promise across domains. We introduce BERT-JEPA (BEPA), a training paradigm that adds a JEPA training objective to BERT-style models, working to combat a collapsed [CLS] embedding space and turning it into a language-agnostic space. This new structure leads to increased performance across multilingual benchmarks.
>
---
#### [new 027] Probabilistic Guarantees for Reducing Contextual Hallucinations in LLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLMs的上下文幻觉问题。通过概率方法降低错误输出概率，无需修改模型。**

- **链接: [https://arxiv.org/pdf/2601.00641v1](https://arxiv.org/pdf/2601.00641v1)**

> **作者:** Nils Rautenberg; Sven Schippkus
>
> **摘要:** Large language models (LLMs) frequently produce contextual hallucinations, where generated content contradicts or ignores information explicitly stated in the prompt. Such errors are particularly problematic in deterministic automation workflows, where inputs are fixed and correctness is unambiguous. We introduce a simple and model-agnostic framework that provides explicit probabilistic guarantees for reducing hallucinations in this setting. We formalize the notion of a specific task, defined by a fixed input and a deterministic correctness criterion, and show that issuing the same prompt in independent context windows yields an exponential reduction in the probability that all model outputs are incorrect. To identify a correct answer among repeated runs, we incorporate an LLM-as-a-judge and prove that the probability that the judged pipeline fails decays at a rate determined by the judge's true- and false-positive probabilities. When the judge is imperfect, we strengthen it through majority vote over independent judge calls, obtaining ensemble-level error rates that decrease exponentially in the number of votes. This yields an explicit bound on the probability that the pipeline selects a hallucinated answer. Experiments on controlled extraction tasks with synthetic noisy judges match these predictions exactly: pipeline failure decreases exponentially with the number of repetitions, and hallucination-selection decreases exponentially with the number of judges in the ensemble. Together, these results provide a lightweight, modular, and theoretically grounded method for driving hallucination probabilities arbitrarily low in fixed-input LLM workflows-without modifying model weights, decoding strategies, or prompt engineering.
>
---
#### [new 028] Vision-Language Reasoning for Geolocalization: A Reinforcement Learning Approach
- **分类: cs.CL**

- **简介: 该论文属于图像地理定位任务，解决传统方法依赖合成标注或外部检索的问题。提出Geo-R框架，通过强化学习和结构化推理提升定位精度与可解释性。**

- **链接: [https://arxiv.org/pdf/2601.00388v1](https://arxiv.org/pdf/2601.00388v1)**

> **作者:** Biao Wu; Meng Fang; Ling Chen; Ke Xu; Tao Cheng; Jun Wang
>
> **备注:** 8 pages, 1 figures
>
> **摘要:** Recent advances in vision-language models have opened up new possibilities for reasoning-driven image geolocalization. However, existing approaches often rely on synthetic reasoning annotations or external image retrieval, which can limit interpretability and generalizability. In this paper, we present Geo-R, a retrieval-free framework that uncovers structured reasoning paths from existing ground-truth coordinates and optimizes geolocation accuracy via reinforcement learning. We propose the Chain of Region, a rule-based hierarchical reasoning paradigm that generates precise, interpretable supervision by mapping GPS coordinates to geographic entities (e.g., country, province, city) without relying on model-generated or synthetic labels. Building on this, we introduce a lightweight reinforcement learning strategy with coordinate-aligned rewards based on Haversine distance, enabling the model to refine predictions through spatially meaningful feedback. Our approach bridges structured geographic reasoning with direct spatial supervision, yielding improved localization accuracy, stronger generalization, and more transparent inference. Experimental results across multiple benchmarks confirm the effectiveness of Geo-R, establishing a new retrieval-free paradigm for scalable and interpretable image geolocalization. To facilitate further research and ensure reproducibility, both the model and code will be made publicly available.
>
---
#### [new 029] Defensive M2S: Training Guardrail Models on Compressed Multi-turn Conversations
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于安全防护任务，旨在降低Guardrail模型的训练与推理成本。通过将多轮对话压缩为单轮，提出Defensive M2S方法，显著提升效率并保持检测效果。**

- **链接: [https://arxiv.org/pdf/2601.00454v1](https://arxiv.org/pdf/2601.00454v1)**

> **作者:** Hyunjun Kim
>
> **摘要:** Guardrail models are essential for ensuring the safety of Large Language Model (LLM) deployments, but processing full multi-turn conversation histories incurs significant computational cost. We propose Defensive M2S, a training paradigm that fine-tunes guardrail models on Multi-turn to Single-turn (M2S) compressed conversations rather than complete dialogue histories. We provide a formal complexity analysis showing that M2S reduces training cost from $O(n^2)$ to $O(n)$ for $n$-turn conversations. Empirically, on our training dataset (779 samples, avg. 10.6 turns), M2S requires only 169K tokens compared to 15.7M tokens for the multi-turn baseline -- a 93$\times$ reduction. We evaluate Defensive M2S across three guardrail model families (LlamaGuard, Nemotron, Qwen3Guard) and three compression templates (hyphenize, numberize, pythonize) on SafeDialBench, a comprehensive multi-turn jailbreak benchmark. Our best configuration, Qwen3Guard with hyphenize compression, achieves 93.8% attack detection recall while reducing inference tokens by 94.6% (from 3,231 to 173 tokens per conversation). This represents a 38.9 percentage point improvement over the baseline while dramatically reducing both training and inference costs. Our findings demonstrate that M2S compression can serve as an effective efficiency technique for guardrail deployment, enabling scalable safety screening of long multi-turn conversations.
>
---
#### [new 030] InfoSynth: Information-Guided Benchmark Synthesis for LLMs
- **分类: cs.CL**

- **简介: 该论文属于基准测试任务，旨在解决LLMs基准创建效率低、多样性不足的问题。提出InfoSynth框架，通过信息理论方法自动生成高质量、新颖的编程问题。**

- **链接: [https://arxiv.org/pdf/2601.00575v1](https://arxiv.org/pdf/2601.00575v1)**

> **作者:** Ishir Garg; Neel Kolhe; Xuandong Zhao; Dawn Song
>
> **摘要:** Large language models (LLMs) have demonstrated significant advancements in reasoning and code generation. However, efficiently creating new benchmarks to evaluate these capabilities remains a challenge. Traditional benchmark creation relies on manual human effort, a process that is both expensive and time-consuming. Furthermore, existing benchmarks often contaminate LLM training data, necessitating novel and diverse benchmarks to accurately assess their genuine capabilities. This work introduces InfoSynth, a novel framework for automatically generating and evaluating reasoning benchmarks guided by information-theoretic principles. We propose metrics based on KL-divergence and entropy to quantify benchmark novelty and diversity without relying on costly model evaluations. Building on this framework, we develop an end-to-end pipeline that synthesizes robust Python coding problems from seed datasets using genetic algorithms and iterative code feedback. Our method generates accurate test cases and solutions to new problems 97% of the time, and the synthesized benchmarks consistently exhibit higher novelty and diversity compared to their seed datasets. Moreover, our algorithm provides a method for controlling the novelty/diversity and difficulty of generated problems. InfoSynth offers a scalable, self-verifying pipeline for constructing high-quality, novel and diverse benchmarks for LLMs. Project Page: https://ishirgarg.github.io/infosynth_web/
>
---
#### [new 031] Can Large Language Models Still Explain Themselves? Investigating the Impact of Quantization on Self-Explanations
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于模型压缩任务，研究量化对大语言模型自解释能力的影响。旨在解决量化是否降低自解释质量与可信度的问题，通过实验分析不同量化方法的效果。**

- **链接: [https://arxiv.org/pdf/2601.00282v1](https://arxiv.org/pdf/2601.00282v1)**

> **作者:** Qianli Wang; Nils Feldhus; Pepa Atanasova; Fedor Splitt; Simon Ostermann; Sebastian Möller; Vera Schmitt
>
> **备注:** In submission
>
> **摘要:** Quantization is widely used to accelerate inference and streamline the deployment of large language models (LLMs), yet its effects on self-explanations (SEs) remain unexplored. SEs, generated by LLMs to justify their own outputs, require reasoning about the model's own decision-making process, a capability that may exhibit particular sensitivity to quantization. As SEs are increasingly relied upon for transparency in high-stakes applications, understanding whether and to what extent quantization degrades SE quality and faithfulness is critical. To address this gap, we examine two types of SEs: natural language explanations (NLEs) and counterfactual examples, generated by LLMs quantized using three common techniques at distinct bit widths. Our findings indicate that quantization typically leads to moderate declines in both SE quality (up to 4.4\%) and faithfulness (up to 2.38\%). The user study further demonstrates that quantization diminishes both the coherence and trustworthiness of SEs (up to 8.5\%). Compared to smaller models, larger models show limited resilience to quantization in terms of SE quality but better maintain faithfulness. Moreover, no quantization technique consistently excels across task accuracy, SE quality, and faithfulness. Given that quantization's impact varies by context, we recommend validating SE quality for specific use cases, especially for NLEs, which show greater sensitivity. Nonetheless, the relatively minor deterioration in SE quality and faithfulness does not undermine quantization's effectiveness as a model compression technique.
>
---
#### [new 032] Physio-DPO: Aligning Large Language Models with the Protein Energy Landscape to Eliminate Structural Hallucinations
- **分类: cs.CL; cs.CE; q-bio.QM**

- **简介: 该论文属于蛋白质生成任务，旨在解决语言模型生成的结构不稳定问题。通过引入物理能量景观信息，提出Physio-DPO框架，提升生成序列的热力学稳定性。**

- **链接: [https://arxiv.org/pdf/2601.00647v1](https://arxiv.org/pdf/2601.00647v1)**

> **作者:** QiWei Meng
>
> **摘要:** Large Protein Language Models have shown strong potential for generative protein design, yet they frequently produce structural hallucinations, generating sequences with high linguistic likelihood that fold into thermodynamically unstable conformations. Existing alignment approaches such as Direct Preference Optimization are limited in this setting, as they model preferences as binary labels and ignore the continuous structure of the physical energy landscape. We propose Physio-DPO, a physics informed alignment framework that grounds protein language models in thermodynamic stability. Physio-DPO introduces a magnitude aware objective that scales optimization updates according to the energy gap between native structures and physics perturbed hard negatives. Experiments show that Physio-DPO consistently outperforms strong baselines including SFT, PPO, and standard DPO, reducing self consistency RMSD to 1.28 Å and increasing foldability to 92.8%. Qualitative analysis further demonstrates that Physio-DPO effectively mitigates structural hallucinations by recovering biophysical interactions such as hydrophobic core packing and hydrogen bond networks.
>
---
#### [new 033] Exploring the Performance of Large Language Models on Subjective Span Identification Tasks
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型在主观文本片段识别任务中的表现，解决如何有效识别情感、攻击性语言和事实核查相关文本片段的问题，评估了多种LLM策略。**

- **链接: [https://arxiv.org/pdf/2601.00736v1](https://arxiv.org/pdf/2601.00736v1)**

> **作者:** Alphaeus Dmonte; Roland Oruche; Tharindu Ranasinghe; Marcos Zampieri; Prasad Calyam
>
> **摘要:** Identifying relevant text spans is important for several downstream tasks in NLP, as it contributes to model explainability. While most span identification approaches rely on relatively smaller pre-trained language models like BERT, a few recent approaches have leveraged the latest generation of Large Language Models (LLMs) for the task. Current work has focused on explicit span identification like Named Entity Recognition (NER), while more subjective span identification with LLMs in tasks like Aspect-based Sentiment Analysis (ABSA) has been underexplored. In this paper, we fill this important gap by presenting an evaluation of the performance of various LLMs on text span identification in three popular tasks, namely sentiment analysis, offensive language identification, and claim verification. We explore several LLM strategies like instruction tuning, in-context learning, and chain of thought. Our results indicate underlying relationships within text aid LLMs in identifying precise text spans.
>
---
#### [new 034] Sigmoid Head for Quality Estimation under Language Ambiguity
- **分类: cs.CL**

- **简介: 该论文属于质量估计任务，旨在解决语言模型概率不可靠的问题。通过引入Sigmoid Head模块，提升质量信号的准确性，增强模型在不确定输出时的判断能力。**

- **链接: [https://arxiv.org/pdf/2601.00680v1](https://arxiv.org/pdf/2601.00680v1)**

> **作者:** Tu Anh Dinh; Jan Niehues
>
> **摘要:** Language model (LM) probability is not a reliable quality estimator, as natural language is ambiguous. When multiple output options are valid, the model's probability distribution is spread across them, which can misleadingly indicate low output quality. This issue is caused by two reasons: (1) LMs' final output activation is softmax, which does not allow multiple correct options to receive high probabilities simultaneuously and (2) LMs' training data is single, one-hot encoded references, indicating that there is only one correct option at each output step. We propose training a module for Quality Estimation on top of pre-trained LMs to address these limitations. The module, called Sigmoid Head, is an extra unembedding head with sigmoid activation to tackle the first limitation. To tackle the second limitation, during the negative sampling process to train the Sigmoid Head, we use a heuristic to avoid selecting potentially alternative correct tokens. Our Sigmoid Head is computationally efficient during training and inference. The probability from Sigmoid Head is notably better quality signal compared to the original softmax head. As the Sigmoid Head does not rely on human-annotated quality data, it is more robust to out-of-domain settings compared to supervised QE.
>
---
#### [new 035] Comparative Efficiency Analysis of Lightweight Transformer Models: A Multi-Domain Empirical Benchmark for Enterprise NLP Deployment
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，研究轻量级Transformer模型在企业应用中的效率比较，解决模型性能与效率的权衡问题，通过多领域实验评估DistilBERT、MiniLM和ALBERT的表现。**

- **链接: [https://arxiv.org/pdf/2601.00444v1](https://arxiv.org/pdf/2601.00444v1)**

> **作者:** Muhammad Shahmeer Khan
>
> **备注:** 11 pages, 6 figures. Code and reproducibility resources available on GitHub
>
> **摘要:** In the rapidly evolving landscape of enterprise natural language processing (NLP), the demand for efficient, lightweight models capable of handling multi-domain text automation tasks has intensified. This study conducts a comparative analysis of three prominent lightweight Transformer models - DistilBERT, MiniLM, and ALBERT - across three distinct domains: customer sentiment classification, news topic classification, and toxicity and hate speech detection. Utilizing datasets from IMDB, AG News, and the Measuring Hate Speech corpus, we evaluated performance using accuracy-based metrics including accuracy, precision, recall, and F1-score, as well as efficiency metrics such as model size, inference time, throughput, and memory usage. Key findings reveal that no single model dominates all performance dimensions. ALBERT achieves the highest task-specific accuracy in multiple domains, MiniLM excels in inference speed and throughput, and DistilBERT demonstrates the most consistent accuracy across tasks while maintaining competitive efficiency. All results reflect controlled fine-tuning under fixed enterprise-oriented constraints rather than exhaustive hyperparameter optimization. These results highlight trade-offs between accuracy and efficiency, recommending MiniLM for latency-sensitive enterprise applications, DistilBERT for balanced performance, and ALBERT for resource-constrained environments.
>
---
#### [new 036] Learning Speech Representations with Variational Predictive Coding
- **分类: eess.AS; cs.CL**

- **简介: 该论文研究语音表示学习，旨在解决HuBERT目标缺乏理论基础的问题。通过变分预测编码原理改进其参数化和优化，提升下游任务性能。**

- **链接: [https://arxiv.org/pdf/2601.00100v1](https://arxiv.org/pdf/2601.00100v1)**

> **作者:** Sung-Lin Yeh; Peter Bell; Hao Tang
>
> **备注:** Accepted to Transactions of the Association for Computational Linguistics (TACL); Pre MIT Press version
>
> **摘要:** Despite being the best known objective for learning speech representations, the HuBERT objective has not been further developed and improved. We argue that it is the lack of an underlying principle that stalls the development, and, in this paper, we show that predictive coding under a variational view is the principle behind the HuBERT objective. Due to its generality, our formulation provides opportunities to improve parameterization and optimization, and we show two simple modifications that bring immediate improvements to the HuBERT objective. In addition, the predictive coding formulation has tight connections to various other objectives, such as APC, CPC, wav2vec, and BEST-RQ. Empirically, the improvement in pre-training brings significant improvements to four downstream tasks: phone classification, f0 tracking, speaker recognition, and automatic speech recognition, highlighting the importance of the predictive coding interpretation.
>
---
#### [new 037] StockBot 2.0: Vanilla LSTMs Outperform Transformer-based Forecasting for Stock Prices
- **分类: cs.CE; cs.CL; cs.LG**

- **简介: 论文属于金融时间序列预测任务，旨在解决股票价格预测难题。通过对比多种模型，发现传统LSTM在多数情况下表现优于Transformer等现代模型。**

- **链接: [https://arxiv.org/pdf/2601.00197v1](https://arxiv.org/pdf/2601.00197v1)**

> **作者:** Shaswat Mohanty
>
> **备注:** 14 pages, 5 figures
>
> **摘要:** Accurate forecasting of financial markets remains a long-standing challenge due to complex temporal and often latent dependencies, non-linear dynamics, and high volatility. Building on our earlier recurrent neural network framework, we present an enhanced StockBot architecture that systematically evaluates modern attention-based, convolutional, and recurrent time-series forecasting models within a unified experimental setting. While attention-based and transformer-inspired models offer increased modeling flexibility, extensive empirical evaluation reveals that a carefully constructed vanilla LSTM consistently achieves superior predictive accuracy and more stable buy/sell decision-making when trained under a common set of default hyperparameters. These results highlight the robustness and data efficiency of recurrent sequence models for financial time-series forecasting, particularly in the absence of extensive hyperparameter tuning or the availability of sufficient data when discretized to single-day intervals. Additionally, these results underscore the importance of architectural inductive bias in data-limited market prediction tasks.
>
---
#### [new 038] The Trojan in the Vocabulary: Stealthy Sabotage of LLM Composition
- **分类: cs.LG; cs.CL; cs.CR**

- **简介: 该论文研究模型组合中的安全漏洞，解决令牌移植带来的隐蔽攻击问题。通过设计“破坏令牌”，在不损害原模型性能的前提下，植入恶意功能，揭示模块化AI的潜在风险。**

- **链接: [https://arxiv.org/pdf/2601.00065v1](https://arxiv.org/pdf/2601.00065v1)**

> **作者:** Xiaoze Liu; Weichen Yu; Matt Fredrikson; Xiaoqian Wang; Jing Gao
>
> **摘要:** The open-weight LLM ecosystem is increasingly defined by model composition techniques (such as weight merging, speculative decoding, and vocabulary expansion) that remix capabilities from diverse sources. A critical prerequisite for applying these methods across different model families is tokenizer transplant, which aligns incompatible vocabularies to a shared embedding space. We demonstrate that this essential interoperability step introduces a supply-chain vulnerability: we engineer a single "breaker token" that is functionally inert in a donor model yet reliably reconstructs into a high-salience malicious feature after transplant into a base model. By exploiting the geometry of coefficient reuse, our attack creates an asymmetric realizability gap that sabotages the base model's generation while leaving the donor's utility statistically indistinguishable from nominal behavior. We formalize this as a dual-objective optimization problem and instantiate the attack using a sparse solver. Empirically, the attack is training-free and achieves spectral mimicry to evade outlier detection, while demonstrating structural persistence against fine-tuning and weight merging, highlighting a hidden risk in the pipeline of modular AI composition. Code is available at https://github.com/xz-liu/tokenforge
>
---
#### [new 039] A Chain-of-Thought Approach to Semantic Query Categorization in e-Commerce Taxonomies
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于语义查询分类任务，旨在解决电商分类体系中查询与类别匹配的问题。通过结合树搜索和LLM语义评分的链式思维方法，提升分类准确性与效率。**

- **链接: [https://arxiv.org/pdf/2601.00510v1](https://arxiv.org/pdf/2601.00510v1)**

> **作者:** Jetlir Duraj; Ishita Khan; Kilian Merkelbach; Mehran Elyasi
>
> **备注:** 9 pages, accepted at SIGIR eCom 2025
>
> **摘要:** Search in e-Commerce is powered at the core by a structured representation of the inventory, often formulated as a category taxonomy. An important capability in e-Commerce with hierarchical taxonomies is to select a set of relevant leaf categories that are semantically aligned with a given user query. In this scope, we address a fundamental problem of search query categorization in real-world e-Commerce taxonomies. A correct categorization of a query not only provides a way to zoom into the correct inventory space, but opens the door to multiple intent understanding capabilities for a query. A practical and accurate solution to this problem has many applications in e-commerce, including constraining retrieved items and improving the relevance of the search results. For this task, we explore a novel Chain-of-Thought (CoT) paradigm that combines simple tree-search with LLM semantic scoring. Assessing its classification performance on human-judged query-category pairs, relevance tests, and LLM-based reference methods, we find that the CoT approach performs better than a benchmark that uses embedding-based query category predictions. We show how the CoT approach can detect problems within a hierarchical taxonomy. Finally, we also propose LLM-based approaches for query-categorization of the same spirit, but which scale better at the range of millions of queries.
>
---
#### [new 040] The Agentic Leash: Extracting Causal Feedback Fuzzy Cognitive Maps with LLMs
- **分类: cs.AI; cs.CL; cs.HC; cs.IR**

- **简介: 该论文属于因果建模任务，旨在利用LLM提取因果模糊认知图（FCM）。通过三步指令引导LLM生成FCM，使其能捕捉文本中的因果关系并形成动态系统。**

- **链接: [https://arxiv.org/pdf/2601.00097v1](https://arxiv.org/pdf/2601.00097v1)**

> **作者:** Akash Kumar Panda; Olaoluwa Adigun; Bart Kosko
>
> **备注:** 15 figures
>
> **摘要:** We design a large-language-model (LLM) agent that extracts causal feedback fuzzy cognitive maps (FCMs) from raw text. The causal learning or extraction process is agentic both because of the LLM's semi-autonomy and because ultimately the FCM dynamical system's equilibria drive the LLM agents to fetch and process causal text. The fetched text can in principle modify the adaptive FCM causal structure and so modify the source of its quasi-autonomy--its equilibrium limit cycles and fixed-point attractors. This bidirectional process endows the evolving FCM dynamical system with a degree of autonomy while still staying on its agentic leash. We show in particular that a sequence of three finely tuned system instructions guide an LLM agent as it systematically extracts key nouns and noun phrases from text, as it extracts FCM concept nodes from among those nouns and noun phrases, and then as it extracts or infers partial or fuzzy causal edges between those FCM nodes. We test this FCM generation on a recent essay about the promise of AI from the late diplomat and political theorist Henry Kissinger and his colleagues. This three-step process produced FCM dynamical systems that converged to the same equilibrium limit cycles as did the human-generated FCMs even though the human-generated FCM differed in the number of nodes and edges. A final FCM mixed generated FCMs from separate Gemini and ChatGPT LLM agents. The mixed FCM absorbed the equilibria of its dominant mixture component but also created new equilibria of its own to better approximate the underlying causal dynamical system.
>
---
#### [new 041] TeleDoCTR: Domain-Specific and Contextual Troubleshooting for Telecommunications
- **分类: cs.LG; cs.CL; cs.IR**

- **简介: 该论文提出TeleDoCTR系统，解决电信领域工单故障排查问题。通过分类、检索和生成模型，提升排查效率与准确性。**

- **链接: [https://arxiv.org/pdf/2601.00691v1](https://arxiv.org/pdf/2601.00691v1)**

> **作者:** Mohamed Trabelsi; Huseyin Uzunalioglu
>
> **摘要:** Ticket troubleshooting refers to the process of analyzing and resolving problems that are reported through a ticketing system. In large organizations offering a wide range of services, this task is highly complex due to the diversity of submitted tickets and the need for specialized domain knowledge. In particular, troubleshooting in telecommunications (telecom) is a very time-consuming task as it requires experts to interpret ticket content, consult documentation, and search historical records to identify appropriate resolutions. This human-intensive approach not only delays issue resolution but also hinders overall operational efficiency. To enhance the effectiveness and efficiency of ticket troubleshooting in telecom, we propose TeleDoCTR, a novel telecom-related, domain-specific, and contextual troubleshooting system tailored for end-to-end ticket resolution in telecom. TeleDoCTR integrates both domain-specific ranking and generative models to automate key steps of the troubleshooting workflow which are: routing tickets to the appropriate expert team responsible for resolving the ticket (classification task), retrieving contextually and semantically similar historical tickets (retrieval task), and generating a detailed fault analysis report outlining the issue, root cause, and potential solutions (generation task). We evaluate TeleDoCTR on a real-world dataset from a telecom infrastructure and demonstrate that it achieves superior performance over existing state-of-the-art methods, significantly enhancing the accuracy and efficiency of the troubleshooting process.
>
---
#### [new 042] Overlooked Safety Vulnerability in LLMs: Malicious Intelligent Optimization Algorithm Request and its Jailbreak
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于安全漏洞研究任务，旨在解决LLMs在智能优化算法设计中的安全问题。通过构建基准和攻击方法，揭示模型易受攻击的现状。**

- **链接: [https://arxiv.org/pdf/2601.00213v1](https://arxiv.org/pdf/2601.00213v1)**

> **作者:** Haoran Gu; Handing Wang; Yi Mei; Mengjie Zhang; Yaochu Jin
>
> **摘要:** The widespread deployment of large language models (LLMs) has raised growing concerns about their misuse risks and associated safety issues. While prior studies have examined the safety of LLMs in general usage, code generation, and agent-based applications, their vulnerabilities in automated algorithm design remain underexplored. To fill this gap, this study investigates this overlooked safety vulnerability, with a particular focus on intelligent optimization algorithm design, given its prevalent use in complex decision-making scenarios. We introduce MalOptBench, a benchmark consisting of 60 malicious optimization algorithm requests, and propose MOBjailbreak, a jailbreak method tailored for this scenario. Through extensive evaluation of 13 mainstream LLMs including the latest GPT-5 and DeepSeek-V3.1, we reveal that most models remain highly susceptible to such attacks, with an average attack success rate of 83.59% and an average harmfulness score of 4.28 out of 5 on original harmful prompts, and near-complete failure under MOBjailbreak. Furthermore, we assess state-of-the-art plug-and-play defenses that can be applied to closed-source models, and find that they are only marginally effective against MOBjailbreak and prone to exaggerated safety behaviors. These findings highlight the urgent need for stronger alignment techniques to safeguard LLMs against misuse in algorithm design.
>
---
#### [new 043] Memory Bank Compression for Continual Adaptation of Large Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于持续学习任务，解决LLM在更新过程中知识遗忘和记忆膨胀问题。提出MBC方法通过压缩记忆库和优化代码本实现高效适应。**

- **链接: [https://arxiv.org/pdf/2601.00756v1](https://arxiv.org/pdf/2601.00756v1)**

> **作者:** Thomas Katraouras; Dimitrios Rafailidis
>
> **备注:** Accepted to the 41st ACM/SIGAPP Symposium on Applied Computing (SAC '26)
>
> **摘要:** Large Language Models (LLMs) have become a mainstay for many everyday applications. However, as data evolve their knowledge quickly becomes outdated. Continual learning aims to update LLMs with new information without erasing previously acquired knowledge. Although methods such as full fine-tuning can incorporate new data, they are computationally expensive and prone to catastrophic forgetting, where prior knowledge is overwritten. Memory-augmented approaches address this by equipping LLMs with a memory bank, that is an external memory module which stores information for future use. However, these methods face a critical limitation, in particular, the memory bank constantly grows in the real-world scenario when large-scale data streams arrive. In this paper, we propose MBC, a model that compresses the memory bank through a codebook optimization strategy during online adaptation learning. To ensure stable learning, we also introduce an online resetting mechanism that prevents codebook collapse. In addition, we employ Key-Value Low-Rank Adaptation in the attention layers of the LLM, enabling efficient utilization of the compressed memory representations. Experiments with benchmark question-answering datasets demonstrate that MBC reduces the memory bank size to 0.3% when compared against the most competitive baseline, while maintaining high retention accuracy during online adaptation learning. Our code is publicly available at https://github.com/Thomkat/MBC.
>
---
#### [new 044] Geometry of Reason: Spectral Signatures of Valid Mathematical Reasoning
- **分类: cs.LG; cs.AI; cs.CL; cs.LO**

- **简介: 该论文属于数学推理验证任务，旨在检测大语言模型中的有效数学推理。通过分析注意力矩阵的谱特征，提出无需训练的检测方法，实现高准确率分类。**

- **链接: [https://arxiv.org/pdf/2601.00791v1](https://arxiv.org/pdf/2601.00791v1)**

> **作者:** Valentin Noël
>
> **备注:** 58 pages, 19 figures, Under Review
>
> **摘要:** We present a training-free method for detecting valid mathematical reasoning in large language models through spectral analysis of attention patterns. By treating attention matrices as adjacency matrices of dynamic graphs over tokens, we extract four interpretable spectral diagnostics, the Fiedler value (algebraic connectivity), high-frequency energy ratio (HFER), graph signal smoothness, and spectral entropy, that exhibit statistically significant differences between valid and invalid mathematical proofs. Experiments across seven transformer models from four independent architectural families (Meta Llama, Alibaba Qwen, Microsoft Phi, and Mistral AI) demonstrate that this spectral signature produces effect sizes up to Cohen's $d = 3.30$ ($p < 10^{-116}$), enabling 85.0--95.6\% classification accuracy under rigorous evaluation, with calibrated thresholds reaching 93--95\% on the full dataset. The method requires no training data, fine-tuning, or learned classifiers: a single threshold on a spectral metric suffices for high accuracy. Through systematic label correction, we discover that the spectral method detects logical coherence rather than compiler acceptance, identifying mathematically valid proofs that formal verifiers reject due to technical failures. We further identify an architectural dependency: Mistral-7B's Sliding Window Attention shifts the discriminative signal from HFER to late-layer Smoothness ($d = 2.09$, $p_{\text{MW}} = 1.16 \times 10^{-48}$), revealing that attention mechanism design affects which spectral features capture reasoning validity. These findings establish spectral graph analysis as a principled framework for reasoning verification with immediate applications to hallucination detection and AI safety monitoring.
>
---
#### [new 045] From Sight to Insight: Improving Visual Reasoning Capabilities of Multimodal Models via Reinforcement Learning
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉推理任务，旨在提升多模态模型的视觉推理能力。针对模型缺乏视觉信息整合的问题，通过强化学习设计奖励函数，优化推理过程。**

- **链接: [https://arxiv.org/pdf/2601.00215v1](https://arxiv.org/pdf/2601.00215v1)**

> **作者:** Omar Sharif; Eftekhar Hossain; Patrick Ng
>
> **备注:** 23 pages, 15 Figures, 10 Tables
>
> **摘要:** Reinforcement learning (RL) has emerged as a promising approach for eliciting reasoning chains before generating final answers. However, multimodal large language models (MLLMs) generate reasoning that lacks integration of visual information. This limits their ability to solve problems that demand accurate visual perception, such as visual puzzles. We show that visual perception is the key bottleneck in such tasks: converting images into textual descriptions significantly improves performance, yielding gains of 26.7% for Claude 3.5 and 23.6% for Claude 3.7. To address this, we investigate reward-driven RL as a mechanism to unlock long visual reasoning in open-source MLLMs without requiring costly supervision. We design and evaluate six reward functions targeting different reasoning aspects, including image understanding, thinking steps, and answer accuracy. Using group relative policy optimization (GRPO), our approach explicitly incentivizes longer, structured reasoning and mitigates bypassing of visual information. Experiments on Qwen-2.5-VL-7B achieve 5.56% improvements over the base model, with consistent gains across both in-domain and out-of-domain settings.
>
---
#### [new 046] Deep Delta Learning
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出Deep Delta Learning，解决深度残差网络的局限性，通过引入Delta算子增强特征变换能力，提升模型复杂动态建模效果。**

- **链接: [https://arxiv.org/pdf/2601.00417v1](https://arxiv.org/pdf/2601.00417v1)**

> **作者:** Yifan Zhang; Yifeng Liu; Mengdi Wang; Quanquan Gu
>
> **备注:** Project Page: https://github.com/yifanzhang-pro/deep-delta-learning
>
> **摘要:** The efficacy of deep residual networks is fundamentally predicated on the identity shortcut connection. While this mechanism effectively mitigates the vanishing gradient problem, it imposes a strictly additive inductive bias on feature transformations, thereby limiting the network's capacity to model complex state transitions. In this paper, we introduce Deep Delta Learning (DDL), a novel architecture that generalizes the standard residual connection by modulating the identity shortcut with a learnable, data-dependent geometric transformation. This transformation, termed the Delta Operator, constitutes a rank-1 perturbation of the identity matrix, parameterized by a reflection direction vector $\mathbf{k}(\mathbf{X})$ and a gating scalar $β(\mathbf{X})$. We provide a spectral analysis of this operator, demonstrating that the gate $β(\mathbf{X})$ enables dynamic interpolation between identity mapping, orthogonal projection, and geometric reflection. Furthermore, we restructure the residual update as a synchronous rank-1 injection, where the gate acts as a dynamic step size governing both the erasure of old information and the writing of new features. This unification empowers the network to explicitly control the spectrum of its layer-wise transition operator, enabling the modeling of complex, non-monotonic dynamics while preserving the stable training characteristics of gated residual architectures.
>
---
#### [new 047] The Illusion of Insight in Reasoning Models
- **分类: cs.AI; cs.CL**

- **简介: 论文研究推理模型中的“顿悟”现象，分析其是否提升性能。通过大量实验发现这类突变罕见且效果有限，表明其并非模型自纠正机制，而是不稳定推理的表现。**

- **链接: [https://arxiv.org/pdf/2601.00514v1](https://arxiv.org/pdf/2601.00514v1)**

> **作者:** Liv G. d'Aliberti; Manoel Horta Ribeiro
>
> **摘要:** Do reasoning models have "Aha!" moments? Prior work suggests that models like DeepSeek-R1-Zero undergo sudden mid-trace realizations that lead to accurate outputs, implying an intrinsic capacity for self-correction. Yet, it remains unclear whether such intrinsic shifts in reasoning strategy actually improve performance. Here, we study mid-reasoning shifts and instrument training runs to detect them. Our analysis spans 1M+ reasoning traces, hundreds of training checkpoints, three reasoning domains, and multiple decoding temperatures and model architectures. We find that reasoning shifts are rare, do not become more frequent with training, and seldom improve accuracy, indicating that they do not correspond to prior perceptions of model insight. However, their effect varies with model uncertainty. Building on this finding, we show that artificially triggering extrinsic shifts under high entropy reliably improves accuracy. Our results show that mid-reasoning shifts are symptoms of unstable inference behavior rather than an intrinsic mechanism for self-correction.
>
---
#### [new 048] Reasoning in Action: MCTS-Driven Knowledge Retrieval for Large Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于对话系统任务，旨在解决LLM中检索与推理融合不足的问题。提出一种基于MCTS的推理感知知识检索方法，提升对话多样性与信息量。**

- **链接: [https://arxiv.org/pdf/2601.00003v1](https://arxiv.org/pdf/2601.00003v1)**

> **作者:** Shuqi Liu; Bowei He; Chen Ma; Linqi Song
>
> **摘要:** Large language models (LLMs) typically enhance their performance through either the retrieval of semantically similar information or the improvement of their reasoning capabilities. However, a significant challenge remains in effectively integrating both retrieval and reasoning strategies to optimize LLM performance. In this paper, we introduce a reasoning-aware knowledge retrieval method that enriches LLMs with information aligned to the logical structure of conversations, moving beyond surface-level semantic similarity. We follow a coarse-to-fine approach for knowledge retrieval. First, we identify a contextually relevant sub-region of the knowledge base, ensuring that all sentences within it are relevant to the context topic. Next, we refine our search within this sub-region to extract knowledge that is specifically relevant to the reasoning process. Throughout both phases, we employ the Monte Carlo Tree Search-inspired search method to effectively navigate through knowledge sentences using common keywords. Experiments on two multi-turn dialogue datasets demonstrate that our knowledge retrieval approach not only aligns more closely with the underlying reasoning in human conversations but also significantly enhances the diversity of the retrieved knowledge, resulting in more informative and creative responses.
>
---
#### [new 049] Finetuning Large Language Models for Automated Depression Screening in Nigerian Pidgin English: GENSCORE Pilot Study
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决尼日利亚因语言障碍导致的抑郁症筛查不足问题。通过微调大语言模型，实现基于尼日利亚皮钦语的自动化抑郁筛查。**

- **链接: [https://arxiv.org/pdf/2601.00004v1](https://arxiv.org/pdf/2601.00004v1)**

> **作者:** Isaac Iyinoluwa Olufadewa; Miracle Ayomikun Adesina; Ezekiel Ayodeji Oladejo; Uthman Babatunde Usman; Owen Kolade Adeniyi; Matthew Tolulope Olawoyin
>
> **备注:** 9 pages, 1 figure, 4 tables
>
> **摘要:** Depression is a major contributor to the mental-health burden in Nigeria, yet screening coverage remains limited due to low access to clinicians, stigma, and language barriers. Traditional tools like the Patient Health Questionnaire-9 (PHQ-9) were validated in high-income countries but may be linguistically or culturally inaccessible for low- and middle-income countries and communities such as Nigeria where people communicate in Nigerian Pidgin and more than 520 local languages. This study presents a novel approach to automated depression screening using fine-tuned large language models (LLMs) adapted for conversational Nigerian Pidgin. We collected a dataset of 432 Pidgin-language audio responses from Nigerian young adults aged 18-40 to prompts assessing psychological experiences aligned with PHQ-9 items, performed transcription, rigorous preprocessing and annotation, including semantic labeling, slang and idiom interpretation, and PHQ-9 severity scoring. Three LLMs - Phi-3-mini-4k-instruct, Gemma-3-4B-it, and GPT-4.1 - were fine-tuned on this annotated dataset, and their performance was evaluated quantitatively (accuracy, precision and semantic alignment) and qualitatively (clarity, relevance, and cultural appropriateness). GPT-4.1 achieved the highest quantitative performance, with 94.5% accuracy in PHQ-9 severity scoring prediction, outperforming Gemma-3-4B-it and Phi-3-mini-4k-instruct. Qualitatively, GPT-4.1 also produced the most culturally appropriate, clear, and contextually relevant responses. AI-mediated depression screening for underserved Nigerian communities. This work provides a foundation for deploying conversational mental-health tools in linguistically diverse, resource-constrained environments.
>
---
## 更新

#### [replaced 001] RAG-BioQA: A Retrieval-Augmented Generation Framework for Long-Form Biomedical Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于生物医学问答任务，解决长文本答案生成问题。通过整合检索与生成模型，提升临床决策支持的信息全面性。**

- **链接: [https://arxiv.org/pdf/2510.01612v3](https://arxiv.org/pdf/2510.01612v3)**

> **作者:** Lovely Yeswanth Panchumarthi; Sumalatha Saleti; Sai Prasad Gudari; Atharva Negi; Praveen Raj Budime; Harsit Upadhya
>
> **备注:** Submitted to ICAEI
>
> **摘要:** The rapidly growth of biomedical literature creates challenges acquiring specific medical information. Current biomedical question-answering systems primarily focus on short-form answers, failing to provide comprehensive explanations necessary for clinical decision-making. We present RAG-BioQA, a retrieval-augmented generation framework for long-form biomedical question answering. Our system integrates BioBERT embeddings with FAISS indexing for retrieval and a LoRA fine-tuned FLAN-T5 model for answer generation. We train on 181k QA pairs from PubMedQA, MedDialog, and MedQuAD, and evaluate on a held-out PubMedQA test set. We compare four retrieval strategies: dense retrieval (FAISS), BM25, ColBERT, and MonoT5. Our results show that domain-adapted dense retrieval outperforms zero-shot neural re-rankers, with the best configuration achieving 0.24 BLEU-1 and 0.29 ROUGE-1. Fine-tuning improves BERTScore by 81\% over the base model. We release our framework to support reproducible biomedical QA research.
>
---
#### [replaced 002] Modeling the One-to-Many Property in Open-Domain Dialogue with LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于开放域对话任务，旨在解决对话响应多样性不足的问题。通过分解为多响应生成和偏好选择，提出新方法与数据集，提升响应多样性与质量。**

- **链接: [https://arxiv.org/pdf/2506.15131v2](https://arxiv.org/pdf/2506.15131v2)**

> **作者:** Jing Yang Lee; Kong-Aik Lee; Woon-Seng Gan
>
> **摘要:** Open-domain Dialogue (OD) exhibits a one-to-many (o2m) property, whereby multiple appropriate responses exist for a single dialogue context. Despite prior research showing that modeling this property boosts response diversity, most modern LLM-based dialogue agents do not explicitly do so. In this work, we model the o2m property of OD in LLMs by decomposing OD generation into two key tasks: Multi-Response Generation (MRG) and Preference-based Selection (PS), which entail generating a set of n semantically and lexically diverse high-quality responses for a given dialogue context, followed by selecting a single response based on human preference, respectively. To facilitate MRG and PS, we introduce o2mDial, a dialogue corpus explicitly designed to capture the o2m property by featuring multiple plausible responses for each context. Leveraging o2mDial, we propose new in-context learning and instruction-tuning strategies, as well as novel evaluation metrics for MRG, alongside a model-based approach for PS. Empirical results demonstrate that applying the proposed two-stage framework to smaller LLMs for OD generation enhances overall response diversity while maintaining contextual coherence, improving response quality by up to 90%, bringing them closer to the performance of larger models.
>
---
#### [replaced 003] One Trigger Token Is Enough: A Defense Strategy for Balancing Safety and Usability in Large Language Models
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于安全防护任务，旨在解决LLM易受jailbreak攻击的问题。通过识别安全触发词，提出D-STT算法，在保持可用性的同时提升模型安全性。**

- **链接: [https://arxiv.org/pdf/2505.07167v3](https://arxiv.org/pdf/2505.07167v3)**

> **作者:** Haoran Gu; Handing Wang; Yi Mei; Mengjie Zhang; Yaochu Jin
>
> **摘要:** Large Language Models (LLMs) have been extensively used across diverse domains, including virtual assistants, automated code generation, and scientific research. However, they remain vulnerable to jailbreak attacks, which manipulate the models into generating harmful responses despite safety alignment. Recent studies have shown that current safety-aligned LLMs undergo shallow safety alignment. In this work, we conduct an in-depth investigation into the underlying mechanism of this phenomenon and reveal that it manifests through learned ''safety trigger tokens'' that activate the model's safety patterns when paired with the specific input. Through both analysis and empirical verification, we further demonstrate the high similarity of the safety trigger tokens across different harmful inputs. Accordingly, we propose D-STT, a simple yet effective defense algorithm that identifies and explicitly decodes safety trigger tokens of the given safety-aligned LLM to activate the model's learned safety patterns. In this process, the safety trigger is constrained to a single token, which effectively preserves model usability by introducing minimum intervention in the decoding process. Extensive experiments across diverse jailbreak attacks and benign prompts demonstrate that D-STT significantly reduces output harmfulness while preserving model usability and incurring negligible response time overhead, outperforming ten baseline methods.
>
---
#### [replaced 004] Navigating the Reality Gap: Privacy-Preserving On-Device Continual Adaptation of ASR for Clinical Telephony
- **分类: cs.CL**

- **简介: 该论文属于语音识别任务，解决临床电话中ASR的现实差距问题。通过本地持续适应框架提升模型性能，保障隐私并减少错误影响。**

- **链接: [https://arxiv.org/pdf/2512.16401v3](https://arxiv.org/pdf/2512.16401v3)**

> **作者:** Darshil Chauhan; Adityasinh Solanki; Vansh Patel; Kanav Kapoor; Ritvik Jain; Aditya Bansal; Pratik Narang; Dhruv Kumar
>
> **摘要:** Automatic Speech Recognition (ASR) holds immense potential to assist in clinical documentation and patient report generation, particularly in resource-constrained regions. However, deployment is currently hindered by a technical deadlock: a severe "Reality Gap" between laboratory performance and noisy, real-world clinical audio, coupled with strict privacy and resource constraints. Such adaptation is essential for clinical telephony systems, where patient speech is highly variable and transcription errors can directly impact downstream clinical workflows. We quantify this gap, showing that a robust multilingual model (IndicWav2Vec) degrades up to a 40.94% WER on rural clinical telephony speech from India, rendering it unusable. We demonstrate consistent improvements on these helpline interactions without transmitting raw patient data off-device via an on-device continual adaptation framework using Low-Rank Adaptation (LoRA). We conduct an investigative study of stabilization strategies, characterizing the trade-offs between data-driven and parameter-driven approaches. Our results demonstrate that multi-domain Experience Replay (ER) yields the primary performance gains, achieving a 17.1% relative improvement in target WER and reducing catastrophic forgetting by 55% compared to naive adaptation. Furthermore, we investigate a stabilized importance estimation strategy (Absolute Fisher) to ensure robust convergence against the high-variance gradients common in clinical telephony speech. Finally, we verify via a domain-specific spot check that acoustic adaptation is a fundamental prerequisite for usability in healthcare settings which cannot be bypassed by language models alone.
>
---
#### [replaced 005] Tabby: A Language Model Architecture for Tabular and Structured Data Synthesis
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出Tabby，解决表格和结构化数据合成问题。通过改进Transformer架构，提升合成数据质量，达到或接近真实数据水平。**

- **链接: [https://arxiv.org/pdf/2503.02152v2](https://arxiv.org/pdf/2503.02152v2)**

> **作者:** Sonia Cromp; Satya Sai Srinath Namburi GNVV; Mohammed Alkhudhayri; Catherine Cao; Samuel Guo; Nicholas Roberts; Frederic Sala
>
> **备注:** 21 pages, 8 figures. Appearing in TMLR 2026
>
> **摘要:** While advances in large language models (LLMs) have greatly improved the quality of synthetic text data in recent years, synthesizing tabular data has received relatively less attention. We address this disparity with Tabby, a simple but powerful post-training modification to the standard Transformer language model architecture, enabling its use for tabular dataset synthesis. Tabby enables the representation of differences across columns using Gated Mixture-of-Experts, with column-specific sets of parameters. Empirically, Tabby results in data quality near or equal to that of real data. By pairing our novel LLM table training technique, Plain, with Tabby, we observe up to a 44% improvement in quality over previous methods. We also show that Tabby extends beyond tables to more general structured data, reaching parity with real data on a nested JSON dataset as well.
>
---
#### [replaced 006] Training a Huggingface Model on AWS Sagemaker (Without Tears)
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于模型训练任务，旨在解决研究人员在AWS SageMaker上训练Hugging Face模型的困难，通过提供集中化指导降低使用门槛。**

- **链接: [https://arxiv.org/pdf/2512.24098v2](https://arxiv.org/pdf/2512.24098v2)**

> **作者:** Liling Tan
>
> **摘要:** The development of Large Language Models (LLMs) has primarily been driven by resource-rich research groups and industry partners. Due to the lack of on-premise computing resources required for increasingly complex models, many researchers are turning to cloud services like AWS SageMaker to train Hugging Face models. However, the steep learning curve of cloud platforms often presents a barrier for researchers accustomed to local environments. Existing documentation frequently leaves knowledge gaps, forcing users to seek fragmented information across the web. This demo paper aims to democratize cloud adoption by centralizing the essential information required for researchers to successfully train their first Hugging Face model on AWS SageMaker from scratch.
>
---
#### [replaced 007] Why Do Multilingual Reasoning Gaps Emerge in Reasoning Language Models?
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究多语言推理模型中的语言理解失败问题，旨在解决低资源语言性能不足的差距。通过检测理解失败并选择性翻译，有效提升多语言推理能力。**

- **链接: [https://arxiv.org/pdf/2510.27269v2](https://arxiv.org/pdf/2510.27269v2)**

> **作者:** Deokhyung Kang; Seonjeong Hwang; Daehui Kim; Hyounghun Kim; Gary Geunbae Lee
>
> **备注:** v2: Fix typos and updated contents
>
> **摘要:** Reasoning language models (RLMs) achieve strong performance on complex reasoning tasks, yet they still exhibit a multilingual reasoning gap, performing better in high-resource languages than in low-resource ones. While recent efforts have been made to address this gap, its underlying causes remain largely unexplored. In this work, we show that this gap primarily stems from failures in language understanding-specifically, the model's inability to translate multilingual inputs into the language dominating its reasoning traces (typically English). As identifying understanding failures can enable targeted mitigation of the gap, we evaluate a range of detection methods and find that understanding failures are detectable to a meaningful extent, with supervised approaches performing best. Building on this, we propose Selective Translation, a strategy that incorporates an English translation into the initial reasoning trace only when an understanding failure is detected. Experimental results using Qwen3-4B show that Selective Translation substantially bridges the multilingual reasoning gap, achieving near full-translation performance while translating only about 20% of inputs. Together, our results show that failures in language understanding are the primary driver of the multilingual reasoning gap and can be detected and selectively mitigated, clarifying its origin and suggesting a path toward more equitable multilingual reasoning. Our code and data are publicly available at https://github.com/deokhk/RLM_analysis
>
---
#### [replaced 008] Towards Acyclic Preference Evaluation of Language Models via Multiple Evaluators
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于语言模型评估任务，解决循环偏好问题。通过多评估器构建偏好图并进行集成去噪，实现无环评价，提升模型评估可靠性。**

- **链接: [https://arxiv.org/pdf/2410.12869v5](https://arxiv.org/pdf/2410.12869v5)**

> **作者:** Zhengyu Hu; Jieyu Zhang; Zhihan Xiong; Alexander Ratner; Kaize Ding; Ranjay Krishna
>
> **摘要:** Despite the remarkable success of Large Language Models (LLMs), evaluating their outputs' quality regarding preference remains a critical challenge. While existing works usually leverage a strong LLM as the judge for comparing LLMs' response pairwisely, such a single-evaluator approach is vulnerable to cyclic preference, i.e., output A is better than B, B than C, but C is better than A, causing contradictory evaluation results. To address this, we introduce PGED (Preference Graph Ensemble and Denoising), a novel approach that leverages multiple model-based evaluators to construct preference graphs, and then ensembles and denoises these graphs for acyclic, non-contradictory evaluation results. We provide theoretical guarantees for our framework, demonstrating its efficacy in recovering the ground truth preference structure. Extensive experiments on ten benchmarks demonstrate PGED's superiority in three applications: 1) model ranking for evaluation, 2) response selection for test-time scaling, and 3) data selection for model fine-tuning. Notably, PGED combines small LLM evaluators (e.g., Llama3-8B, Mistral-7B, Qwen2-7B) to outperform strong ones (e.g., Qwen2-72B), showcasing its effectiveness in enhancing evaluation reliability and improving model performance.
>
---
#### [replaced 009] W2S-AlignTree: Weak-to-Strong Inference-Time Alignment for Large Language Models via Monte Carlo Tree Search
- **分类: cs.CL**

- **简介: 该论文属于语言模型对齐任务，解决模型输出与人类偏好不一致的问题。通过引入MCTS和弱到强泛化，实现无需修改参数的动态控制，提升生成质量。**

- **链接: [https://arxiv.org/pdf/2511.11518v2](https://arxiv.org/pdf/2511.11518v2)**

> **作者:** Zhenyu Ding; Yuhao Wang; Tengyue Xiao; Haoying Wang; Caigui Jiang; Ning Ding
>
> **备注:** AAAI 2026 Oral
>
> **摘要:** Large Language Models (LLMs) demonstrate impressive capabilities, yet their outputs often suffer from misalignment with human preferences due to the inadequacy of weak supervision and a lack of fine-grained control. Training-time alignment methods like Reinforcement Learning from Human Feedback (RLHF) face prohibitive costs in expert supervision and inherent scalability limitations, offering limited dynamic control during inference. Consequently, there is an urgent need for scalable and adaptable alignment mechanisms. To address this, we propose W2S-AlignTree, a pioneering plug-and-play inference-time alignment framework that synergistically combines Monte Carlo Tree Search (MCTS) with the Weak-to-Strong Generalization paradigm for the first time. W2S-AlignTree formulates LLM alignment as an optimal heuristic search problem within a generative search tree. By leveraging weak model's real-time, step-level signals as alignment proxies and introducing an Entropy-Aware exploration mechanism, W2S-AlignTree enables fine-grained guidance during strong model's generation without modifying its parameters. The approach dynamically balances exploration and exploitation in high-dimensional generation search trees. Experiments across controlled sentiment generation, summarization, and instruction-following show that W2S-AlignTree consistently outperforms strong baselines. Notably, W2S-AlignTree raises the performance of Llama3-8B from 1.89 to 2.19, a relative improvement of 15.9 on the summarization task.
>
---
#### [replaced 010] LLM-Guided Exemplar Selection for Few-Shot Wearable-Sensor Human Activity Recognition
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于少样本可穿戴传感器人类活动识别任务，旨在解决传统方法依赖大量标注数据和几何选择的不足。通过引入LLM生成的语义先验，优化示例选择，提升识别效果。**

- **链接: [https://arxiv.org/pdf/2512.22385v2](https://arxiv.org/pdf/2512.22385v2)**

> **作者:** Elsen Ronando; Sozo Inoue
>
> **备注:** This paper has been accepted for presentation at ABC 2026. The manuscript is under revision prior to camera-ready submission
>
> **摘要:** In this paper, we propose an LLM-Guided Exemplar Selection framework to address a key limitation in state-of-the-art Human Activity Recognition (HAR) methods: their reliance on large labeled datasets and purely geometric exemplar selection, which often fail to distinguish similar wearable sensor activities such as walking, walking upstairs, and walking downstairs. Our method incorporates semantic reasoning via an LLM-generated knowledge prior that captures feature importance, inter-class confusability, and exemplar budget multipliers, and uses it to guide exemplar scoring and selection. These priors are combined with margin-based validation cues, PageRank centrality, hubness penalization, and facility-location optimization to obtain a compact and informative set of exemplars. Evaluated on the UCI-HAR dataset under strict few-shot conditions, the framework achieves a macro F1-score of 88.78%, outperforming classical approaches such as random sampling, herding, and k-center. The results show that LLM-derived semantic priors, when integrated with structural and geometric cues, provide a stronger foundation for selecting representative sensor exemplars in few-shot wearable-sensor HAR.
>
---
#### [replaced 011] Scaling Patterns in Adversarial Alignment: Evidence from Multi-LLM Jailbreak Experiments
- **分类: cs.LG; cs.AI; cs.CL; cs.CR; cs.MA**

- **简介: 该论文研究多模型对抗攻击中的规模效应，探讨大模型是否能突破小模型的对齐防护。通过实验分析模型尺寸比与危害性之间的关系，揭示对抗行为的规律。**

- **链接: [https://arxiv.org/pdf/2511.13788v2](https://arxiv.org/pdf/2511.13788v2)**

> **作者:** Samuel Nathanson; Rebecca Williams; Cynthia Matuszek
>
> **摘要:** Large language models (LLMs) increasingly operate in multi-agent and safety-critical settings, raising open questions about how their vulnerabilities scale when models interact adversarially. This study examines whether larger models can systematically jailbreak smaller ones - eliciting harmful or restricted behavior despite alignment safeguards. Using standardized adversarial tasks from JailbreakBench, we simulate over 6,000 multi-turn attacker-target exchanges across major LLM families and scales (0.6B-120B parameters), measuring both harm score and refusal behavior as indicators of adversarial potency and alignment integrity. Each interaction is evaluated through aggregated harm and refusal scores assigned by three independent LLM judges, providing a consistent, model-based measure of adversarial outcomes. Aggregating results across prompts, we find a strong and statistically significant correlation between mean harm and the logarithm of the attacker-to-target size ratio (Pearson r = 0.51, p < 0.001; Spearman rho = 0.52, p < 0.001), indicating that relative model size correlates with the likelihood and severity of harmful completions. Mean harm score variance is higher across attackers (0.18) than across targets (0.10), suggesting that attacker-side behavioral diversity contributes more to adversarial outcomes than target susceptibility. Attacker refusal frequency is strongly and negatively correlated with harm (rho = -0.93, p < 0.001), showing that attacker-side alignment mitigates harmful responses. These findings reveal that size asymmetry influences robustness and provide exploratory evidence for adversarial scaling patterns, motivating more controlled investigations into inter-model alignment and safety.
>
---
#### [replaced 012] GameTileNet: A Semantic Dataset for Low-Resolution Game Art in Procedural Content Generation
- **分类: cs.CV; cs.AI; cs.CL; cs.MM**

- **简介: 该论文提出GameTileNet，一个低分辨率游戏艺术的语义数据集，用于解决叙事驱动的程序生成内容中的视觉-语言对齐问题。**

- **链接: [https://arxiv.org/pdf/2507.02941v2](https://arxiv.org/pdf/2507.02941v2)**

> **作者:** Yi-Chun Chen; Arnav Jhala
>
> **备注:** Camera-ready version of a paper accepted for oral presentation at AIIDE 2025
>
> **摘要:** GameTileNet is a dataset designed to provide semantic labels for low-resolution digital game art, advancing procedural content generation (PCG) and related AI research as a vision-language alignment task. Large Language Models (LLMs) and image-generative AI models have enabled indie developers to create visual assets, such as sprites, for game interactions. However, generating visuals that align with game narratives remains challenging due to inconsistent AI outputs, requiring manual adjustments by human artists. The diversity of visual representations in automatically generated game content is also limited because of the imbalance in distributions across styles for training data. GameTileNet addresses this by collecting artist-created game tiles from OpenGameArt.org under Creative Commons licenses and providing semantic annotations to support narrative-driven content generation. The dataset introduces a pipeline for object detection in low-resolution tile-based game art (e.g., 32x32 pixels) and annotates semantics, connectivity, and object classifications. GameTileNet is a valuable resource for improving PCG methods, supporting narrative-rich game content, and establishing a baseline for object detection in low-resolution, non-photorealistic images. TL;DR: GameTileNet is a semantic dataset of low-resolution game tiles designed to support narrative-driven procedural content generation through visual-language alignment.
>
---
#### [replaced 013] LexGenius: An Expert-Level Benchmark for Large Language Models in Legal General Intelligence
- **分类: cs.CL**

- **简介: 该论文提出LexGenius，一个用于评估大语言模型法律通用智能的基准。解决现有基准无法系统评估法律智能的问题，涵盖七维、十一任务、二十能力，通过多轮验证确保数据可靠性。**

- **链接: [https://arxiv.org/pdf/2512.04578v2](https://arxiv.org/pdf/2512.04578v2)**

> **作者:** Wenjin Liu; Haoran Luo; Xin Feng; Xiang Ji; Lijuan Zhou; Rui Mao; Jiapu Wang; Shirui Pan; Erik Cambria
>
> **摘要:** Legal general intelligence (GI) refers to artificial intelligence (AI) that encompasses legal understanding, reasoning, and decision-making, simulating the expertise of legal experts across domains. However, existing benchmarks are result-oriented and fail to systematically evaluate the legal intelligence of large language models (LLMs), hindering the development of legal GI. To address this, we propose LexGenius, an expert-level Chinese legal benchmark for evaluating legal GI in LLMs. It follows a Dimension-Task-Ability framework, covering seven dimensions, eleven tasks, and twenty abilities. We use the recent legal cases and exam questions to create multiple-choice questions with a combination of manual and LLM reviews to reduce data leakage risks, ensuring accuracy and reliability through multiple rounds of checks. We evaluate 12 state-of-the-art LLMs using LexGenius and conduct an in-depth analysis. We find significant disparities across legal intelligence abilities for LLMs, with even the best LLMs lagging behind human legal professionals. We believe LexGenius can assess the legal intelligence abilities of LLMs and enhance legal GI development. Our project is available at https://github.com/QwenQKing/LexGenius.
>
---
#### [replaced 014] EXAONE 4.0: Unified Large Language Models Integrating Non-reasoning and Reasoning Modes
- **分类: cs.CL; cs.AI**

- **简介: 该论文介绍EXAONE 4.0，融合非推理与推理模式，提升模型性能与多语言支持，解决通用性和推理能力不足问题。**

- **链接: [https://arxiv.org/pdf/2507.11407v2](https://arxiv.org/pdf/2507.11407v2)**

> **作者:** Kyunghoon Bae; Eunbi Choi; Kibong Choi; Stanley Jungkyu Choi; Yemuk Choi; Kyubeen Han; Seokhee Hong; Junwon Hwang; Taewan Hwang; Joonwon Jang; Hyojin Jeon; Kijeong Jeon; Gerrard Jeongwon Jo; Hyunjik Jo; Jiyeon Jung; Euisoon Kim; Hyosang Kim; Jihoon Kim; Joonkee Kim; Seonghwan Kim; Soyeon Kim; Sunkyoung Kim; Yireun Kim; Yongil Kim; Youchul Kim; Edward Hwayoung Lee; Gwangho Lee; Haeju Lee; Honglak Lee; Jinsik Lee; Kyungmin Lee; Sangha Park; Young Min Paik; Yongmin Park; Youngyong Park; Sanghyun Seo; Sihoon Yang; Heuiyeen Yeen; Sihyuk Yi; Hyeongu Yun
>
> **备注:** Technical Report, 30 Pages
>
> **摘要:** This technical report introduces EXAONE 4.0, which integrates a Non-reasoning mode and a Reasoning mode to achieve both the excellent usability of EXAONE 3.5 and the advanced reasoning abilities of EXAONE Deep. To pave the way for the agentic AI era, EXAONE 4.0 incorporates essential features such as agentic tool use, and its multilingual capabilities are extended to support Spanish in addition to English and Korean. The EXAONE 4.0 model series consists of two sizes: a mid-size 32B model optimized for high performance, and a small-size 1.2B model designed for on-device applications. The EXAONE 4.0 demonstrates superior performance compared to open-weight models in its class and remains competitive even against frontier-class models. The models are publicly available for research purposes and can be easily downloaded via https://huggingface.co/LGAI-EXAONE.
>
---
#### [replaced 015] Training-free Context-adaptive Attention for Efficient Long Context Modeling
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决长文本处理中的计算效率问题。提出TCA-Attention机制，无需训练即可高效处理长上下文，提升推理速度并减少内存占用。**

- **链接: [https://arxiv.org/pdf/2512.09238v2](https://arxiv.org/pdf/2512.09238v2)**

> **作者:** Zeng You; Yaofo Chen; Shuhai Zhang; Zhijie Qiu; Tingyu Wu; Yingjian Li; Yaowei Wang; Mingkui Tan
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities across a wide range of natural language processing tasks. These capabilities stem primarily from the self-attention mechanism, which enables modeling of long-range dependencies. However, the quadratic complexity of self-attention with respect to sequence length poses significant computational and memory challenges, especially as sequence length extends to extremes. While various sparse attention and KV cache compression methods have been proposed to improve efficiency, they often suffer from limitations such as reliance on fixed patterns, inability to handle both prefilling and decoding stages, or the requirement for additional training. In this paper, we propose Training-free Context-adaptive Attention (TCA-Attention), a training-free sparse attention mechanism that selectively attends to only the informative tokens for efficient long-context inference. Our method consists of two lightweight phases: i) an offline calibration phase that determines head-specific sparsity budgets via a single forward pass, and ii) an online token selection phase that adaptively retains core context tokens using a lightweight redundancy metric. TCA-Attention provides a unified solution that accelerates both prefilling and decoding while reducing KV cache memory footprint, without requiring parameter updates or architectural changes. Theoretical analysis shows that our approach maintains bounded approximation error. Extensive experiments demonstrate that TCA-Attention achieves a 2.8$\times$ speedup and reduces KV cache by 61% at 128K context length while maintaining performance comparable to full attention across various benchmarks, offering a practical plug-and-play solution for efficient long-context inference.
>
---
#### [replaced 016] Multi-hop Reasoning via Early Knowledge Alignment
- **分类: cs.CL**

- **简介: 该论文属于信息检索与问答任务，旨在解决多跳问题中迭代RAG系统效率低的问题。提出EKA模块，在推理前对齐知识，提升检索精度与效率。**

- **链接: [https://arxiv.org/pdf/2512.20144v2](https://arxiv.org/pdf/2512.20144v2)**

> **作者:** Yuxin Wang; Shicheng Fang; Bo Wang; Qi Luo; Xuanjing Huang; Yining Zheng; Xipeng Qiu
>
> **备注:** 16 pages
>
> **摘要:** Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm for Large Language Models (LLMs) to address knowledge-intensive queries requiring domain-specific or up-to-date information. To handle complex multi-hop questions that are challenging for single-step retrieval, iterative RAG approaches incorporating reinforcement learning have been proposed. However, existing iterative RAG systems typically plan to decompose questions without leveraging information about the available retrieval corpus, leading to inefficient retrieval and reasoning chains that cascade into suboptimal performance. In this paper, we introduce Early Knowledge Alignment (EKA), a simple but effective module that aligns LLMs with retrieval set before planning in iterative RAG systems with contextually relevant retrieved knowledge. Extensive experiments on six standard RAG datasets demonstrate that by establishing a stronger reasoning foundation, EKA significantly improves retrieval precision, reduces cascading errors, and enhances both performance and efficiency. Our analysis from an entropy perspective demonstrate that incorporating early knowledge reduces unnecessary exploration during the reasoning process, enabling the model to focus more effectively on relevant information subsets. Moreover, EKA proves effective as a versatile, training-free inference strategy that scales seamlessly to large models. Generalization tests across diverse datasets and retrieval corpora confirm the robustness of our approach. Overall, EKA advances the state-of-the-art in iterative RAG systems while illuminating the critical interplay between structured reasoning and efficient exploration in reinforcement learning-augmented frameworks. The code is released at \href{https://github.com/yxzwang/EarlyKnowledgeAlignment}{Github}.
>
---
#### [replaced 017] FedSEA-LLaMA: A Secure, Efficient and Adaptive Federated Splitting Framework for Large Language Models
- **分类: cs.CL; cs.AI; cs.DC**

- **简介: 该论文属于联邦学习任务，旨在解决隐私保护、通信效率和模型适应性问题。提出FedSEA-LLaMA框架，通过噪声注入、压缩技术和动态分割点提升安全性和效率。**

- **链接: [https://arxiv.org/pdf/2505.15683v4](https://arxiv.org/pdf/2505.15683v4)**

> **作者:** Zishuai Zhang; Hainan zhang; Weihua Li; Qinnan zhang; jin Dong; Yongxin Tong; Zhiming Zheng
>
> **摘要:** Private data holds promise for improving LLMs due to its high quality, but its scattered distribution across data silos and the high computational demands of LLMs limit their deployment in federated environments. To address this, the transformer-based federated split models are proposed, which offload most model parameters to the server (or distributed clients) while retaining only a small portion on the client to ensure data privacy. Despite this design, they still face three challenges: 1) Peer-to-peer key encryption struggles to secure transmitted vectors effectively; 2) The auto-regressive nature of LLMs means that federated split learning can only train and infer sequentially, causing high communication overhead; 3) Fixed partition points lack adaptability to downstream tasks. In this paper, we introduce FedSEA-LLaMA, a Secure, Efficient, and Adaptive Federated splitting framework based on LLaMA2. First, we inject Gaussian noise into forward-pass hidden states to enable secure end-to-end vector transmission. Second, we employ attention-mask compression and KV cache collaboration to reduce communication costs, accelerating training and inference. Third, we allow users to dynamically adjust the partition points for input/output blocks based on specific task requirements. Experiments on natural language understanding, summarization, and conversational QA tasks show that FedSEA-LLaMA maintains performance comparable to centralized LLaMA2 and achieves up to 8x speedups in training and inference. Further analysis of privacy attacks and different partition points also demonstrates the effectiveness of FedSEA-LLaMA in security and adaptability.
>
---
#### [replaced 018] Learning the Boundary of Solvability: Aligning LLMs to Detect Unsolvable Problems
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大模型可靠性任务，解决LLM混淆不可解问题与能力不足的问题，提出UnsolvableQA数据集和UnsolvableRL框架，提升检测准确率和推理性能。**

- **链接: [https://arxiv.org/pdf/2512.01661v2](https://arxiv.org/pdf/2512.01661v2)**

> **作者:** Dengyun Peng; Qiguang Chen; Bofei Liu; Jiannan Guan; Libo Qin; Zheng Yan; Jinhao Liu; Jianshu Zhang; Wanxiang Che
>
> **备注:** preprint
>
> **摘要:** Ensuring large language model (LLM) reliability requires distinguishing objective unsolvability (inherent contradictions) from subjective capability limitations (tasks exceeding model competence). Current LLMs often conflate these dimensions, leading to hallucinations in which they return confident answers to inherently unsolvable queries. To address this issue, we propose a multi-domain dataset containing both solvable and unsolvable questions, UnsolvableQA, together with an alignment framework, UnsolvableRL. First, we construct UnsolvableQA by "Reverse Construction" that systematically injects logical contradictions into otherwise valid reasoning chains. Second, we introduce UnsolvableRL, a reinforcement learning paradigm that balances objective unsolvability detection with calibrated confidence under capability limits. Empirically, our approach achieves near-perfect unsolvability detection (>90% detection rate) and boosts solvable reasoning accuracy from 43.4% to 69.4% on Qwen3-4B-Instruct. Crucially, we identify a data-training interaction: strict alignment constraints induce Capability Collapse without unsolvable data, but act as a regularizer for rigor when such data are included, thereby improving overall robustness. Our code and data are available at https://github.com/sfasfaffa/unsolvableQA .
>
---
#### [replaced 019] MTSQL-R1: Towards Long-Horizon Multi-Turn Text-to-SQL via Agentic Training
- **分类: cs.CL; cs.AI; cs.DB; cs.LG**

- **简介: 该论文属于多轮文本到SQL任务，旨在解决对话连贯性和查询执行问题。通过构建一个代理训练框架，实现迭代生成、执行验证与优化，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2510.12831v2](https://arxiv.org/pdf/2510.12831v2)**

> **作者:** Taicheng Guo; Hai Wang; ChaoChun Liu; Mohsen Golalikhani; Xin Chen; Xiangliang Zhang; Chandan K. Reddy
>
> **摘要:** Multi-turn Text-to-SQL aims to translate a user's conversational utterances into executable SQL while preserving dialogue coherence and grounding to the target schema. However, most existing systems only regard this task as a simple text translation task and follow a short-horizon paradigm, generating a query per turn without execution, explicit verification, and refinement, which leads to non-executable or incoherent outputs. We present MTSQL-R1, an agentic training framework for long-horizon multi-turn Text-to-SQL. We cast the task as a Markov Decision Process (MDP) in which an agent interacts with (i) a database for execution feedback and (ii) a persistent dialogue memory for coherence verification, performing an iterative propose to execute -> verify -> refine cycle until all checks pass. Experiments on COSQL and SPARC demonstrate that MTSQL-R1 consistently outperforms strong baselines, highlighting the importance of environment-driven verification and memory-guided refinement for conversational semantic parsing. Full recipes (including code, trained models, logs, reasoning trajectories, etc.) will be released after the internal review to contribute to community research.
>
---
#### [replaced 020] TabiBERT: A Large-Scale ModernBERT Foundation Model and Unified Benchmarking Framework for Turkish
- **分类: cs.CL**

- **简介: 该论文提出TabiBERT，一个基于ModernBERT的土耳其语单语模型，解决土耳其语NLP缺乏先进架构模型的问题。通过大规模预训练和引入新技术，提升性能并建立基准测试框架TabiBench。**

- **链接: [https://arxiv.org/pdf/2512.23065v2](https://arxiv.org/pdf/2512.23065v2)**

> **作者:** Melikşah Türker; A. Ebrar Kızıloğlu; Onur Güngör; Susan Üsküdarlı
>
> **备注:** 33 pages, 2 figures, 13 tables
>
> **摘要:** Since the inception of BERT, encoder-only Transformers have evolved significantly in computational efficiency, training stability, and long-context modeling. ModernBERT consolidates these advances by integrating Rotary Positional Embeddings (RoPE), FlashAttention, and refined normalization. Despite these developments, Turkish NLP lacks a monolingual encoder trained from scratch, incorporating such modern architectural paradigms. This work introduces TabiBERT, a monolingual Turkish encoder based on ModernBERT architecture trained from scratch on a large, curated corpus. TabiBERT is pre-trained on one trillion tokens sampled from an 84.88B token multi-domain corpus: web text (73%), scientific publications (20%), source code (6%), and mathematical content (0.3%). It supports 8,192-token context length (16x original BERT), achieves up to 2.65x inference speedup, and reduces GPU memory consumption, enabling larger batch sizes. We introduce TabiBench with 28 datasets across eight task categories with standardized splits and protocols, evaluated using GLUE-style macro-averaging. TabiBERT attains 77.58 on TabiBench, outperforming BERTurk by 1.62 points and establishing state-of-the-art on five of eight categories, with particularly strong gains on question answering (+9.55 points), code retrieval (+2.41 points), and academic understanding (+0.66 points). Compared with task-specific prior best results, including specialized models like TurkishBERTweet, TabiBERT achieves +1.47 average improvement, indicating robust cross-domain generalization. We release model weights, training configurations, and evaluation code for transparent, reproducible Turkish encoder research.
>
---
#### [replaced 021] Improving Multi-step RAG with Hypergraph-based Memory for Long-Context Complex Relational Modeling
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于多步检索增强生成任务，旨在解决长文本中复杂关系建模的问题。提出HGMem，通过超图结构增强记忆的动态表达与推理能力。**

- **链接: [https://arxiv.org/pdf/2512.23959v2](https://arxiv.org/pdf/2512.23959v2)**

> **作者:** Chulun Zhou; Chunkang Zhang; Guoxin Yu; Fandong Meng; Jie Zhou; Wai Lam; Mo Yu
>
> **备注:** 21 pages
>
> **摘要:** Multi-step retrieval-augmented generation (RAG) has become a widely adopted strategy for enhancing large language models (LLMs) on tasks that demand global comprehension and intensive reasoning. Many RAG systems incorporate a working memory module to consolidate retrieved information. However, existing memory designs function primarily as passive storage that accumulates isolated facts for the purpose of condensing the lengthy inputs and generating new sub-queries through deduction. This static nature overlooks the crucial high-order correlations among primitive facts, the compositions of which can often provide stronger guidance for subsequent steps. Therefore, their representational strength and impact on multi-step reasoning and knowledge evolution are limited, resulting in fragmented reasoning and weak global sense-making capacity in extended contexts. We introduce HGMem, a hypergraph-based memory mechanism that extends the concept of memory beyond simple storage into a dynamic, expressive structure for complex reasoning and global understanding. In our approach, memory is represented as a hypergraph whose hyperedges correspond to distinct memory units, enabling the progressive formation of higher-order interactions within memory. This mechanism connects facts and thoughts around the focal problem, evolving into an integrated and situated knowledge structure that provides strong propositions for deeper reasoning in subsequent steps. We evaluate HGMem on several challenging datasets designed for global sense-making. Extensive experiments and in-depth analyses show that our method consistently improves multi-step RAG and substantially outperforms strong baseline systems across diverse tasks.
>
---
#### [replaced 022] Esoteric Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出Eso-LMs，融合自回归与扩散模型，解决生成效率与质量平衡问题，提升推理速度并支持KV缓存。**

- **链接: [https://arxiv.org/pdf/2506.01928v2](https://arxiv.org/pdf/2506.01928v2)**

> **作者:** Subham Sekhar Sahoo; Zhihan Yang; Yash Akhauri; Johnna Liu; Deepansha Singh; Zhoujun Cheng; Zhengzhong Liu; Eric Xing; John Thickstun; Arash Vahdat
>
> **摘要:** Diffusion-based language models offer a compelling alternative to autoregressive (AR) models by enabling parallel and controllable generation. Within this family, Masked Diffusion Models (MDMs) currently perform best but still underperform AR models in perplexity and lack key inference-time efficiency features, most notably KV caching. We introduce Eso-LMs, a new family of models that fuses AR and MDM paradigms, smoothly interpolating between their perplexities while overcoming their respective limitations. Unlike prior work, which uses transformers with bidirectional attention as MDM denoisers, we exploit the connection between MDMs and Any-Order autoregressive models and adopt causal attention. This design lets us compute the exact likelihood of MDMs for the first time and, crucially, enables us \to introduce KV caching for MDMs while preserving parallel generation for the first time, significantly improving inference efficiency. Combined with an optimized sampling schedule, Eso-LMs achieves a new state of the art on the speed-quality Pareto frontier for unconditional generation. On long contexts, it yields $\mathbf{14 - 65{}\times}$ faster inference than standard MDMs and $\mathbf{3 - 4{}\times}$ faster inference than prior semi-autoregressive approaches. We provide code, model checkpoints, and video tutorials on the project page: http://s-sahoo.github.io/Eso-LMs
>
---
#### [replaced 023] AlignAR: Generative Sentence Alignment for Arabic-English Parallel Corpora of Legal and Literary Texts
- **分类: cs.CL**

- **简介: 该论文属于句子对齐任务，旨在解决阿拉伯语与英语平行语料稀缺及对齐方法评估不足的问题。提出AlignAR方法和新数据集，提升对复杂文本的对齐效果。**

- **链接: [https://arxiv.org/pdf/2512.21842v2](https://arxiv.org/pdf/2512.21842v2)**

> **作者:** Baorong Huang; Ali Asiri
>
> **摘要:** High-quality parallel corpora are essential for Machine Translation (MT) research and translation teaching. However, Arabic-English resources remain scarce and existing datasets mainly consist of simple one-to-one mappings. In this paper, we present AlignAR, a generative sentence alignment method, and a new Arabic-English dataset comprising simple legal and complex literary parallel texts. Our evaluation demonstrates that "Easy" datasets lack the discriminatory power to fully assess alignment methods. By reducing one-to-one mappings in our "Hard" subset, we exposed the limitations of traditional alignment methods. In contrast, LLM-based approaches demonstrated better robustness, achieving an overall F1-score of 85.5%, a nearly 9% improvement over previous methods. Our datasets and codes are open-sourced at https://github.com/XXX.
>
---
#### [replaced 024] NeedleChain: Measuring Intact Context Comprehension Capability of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在评估大语言模型的完整上下文理解能力。针对现有基准测试的不足，提出NeedleChain基准，强调模型对全部信息的整合能力。**

- **链接: [https://arxiv.org/pdf/2507.22411v2](https://arxiv.org/pdf/2507.22411v2)**

> **作者:** Hyeonseok Moon; Heuiseok Lim
>
> **备注:** 13 pages
>
> **摘要:** Recent reports suggest that LLMs can handle increasingly long contexts. However, many existing benchmarks for context understanding embed substantial query-irrelevant content, which shifts evaluation toward retrieving relevant snippets rather than fully integrating all provided information. Under this setting, we view that current benchmarks can overestimate true context-understanding ability of LLMs. In particular, we demonstrate that when the context consists entirely of query-relevant text, even advanced models such as GPT-4o fail to reliably integrate inputs as short as 200 tokens. To evaluate this capability more rigorously, we introduce NeedleChain, a benchmark designed to test whether models can faithfully incorporate all given evidence. NeedleChain includes three variants that differ in the required order of comprehension, along with a parallel benchmark based on the needle-in-a-haystack(NIAH) paradigm. By comparing these variants, NeedleChain enables a more comprehensive assessment of context understanding. We further propose a training-free strategy that encourages models to reflect all available information, ROPE contraction, highlighting the importance of full-context integration and pointing to new directions for improving reliable reasoning over context.
>
---
#### [replaced 025] PaperRegister: Boosting Flexible-grained Paper Search via Hierarchical Register Indexing
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于论文检索任务，旨在解决传统系统无法支持细粒度查询的问题。通过构建分层索引树，提升灵活粒度的搜索效果。**

- **链接: [https://arxiv.org/pdf/2508.11116v2](https://arxiv.org/pdf/2508.11116v2)**

> **作者:** Zhuoqun Li; Xuanang Chen; Hongyu Lin; Yaojie Lu; Xianpei Han; Shanshan Jiang; Bin Dong; Le Sun
>
> **摘要:** As researchers delve more deeply into their work, paper search requirements may become more flexible, sometimes involving specific details such as module configuration rather than being limited to coarse-grained topics. However, previous paper search systems are unable to meet these flexible-grained requirements, as previous systems mainly collect paper abstract to construct corpus index, which lacks detailed information to support retrieval by some finer-grained queries. In this work, we propose PaperRegister, which transforms traditional abstract-based index into a hierarchical index tree, thereby supporting queries at flexible granularity. Experiments on paper search tasks across a range of granularity demonstrate that PaperRegister achieves the SOTA performance, and particularly excels in the fine-grained scenarios, highlighting good potential as an effective solution for flexible-grained paper search in real-world applications. https://github.com/Li-Z-Q/PaperRegister.
>
---
#### [replaced 026] Narrative-to-Scene Generation: An LLM-Driven Pipeline for 2D Game Environments
- **分类: cs.GR; cs.AI; cs.CL; cs.MM**

- **简介: 该论文属于叙事到场景生成任务，旨在将文本故事转化为2D游戏场景。通过LLM提取关键时间点和空间关系，结合地形生成与对象放置，实现故事驱动的场景构建。**

- **链接: [https://arxiv.org/pdf/2509.04481v2](https://arxiv.org/pdf/2509.04481v2)**

> **作者:** Yi-Chun Chen; Arnav Jhala
>
> **备注:** Camera-ready version of a paper accepted at the AIIDE 2025 Workshop on Experimental AI in Games (EXAG)
>
> **摘要:** Recent advances in large language models (LLMs) enable compelling story generation, but connecting narrative text to playable visual environments remains an open challenge in procedural content generation (PCG). We present a lightweight pipeline that transforms short narrative prompts into a sequence of 2D tile-based game scenes, reflecting the temporal structure of stories. Given an LLM-generated narrative, our system identifies three key time frames, extracts spatial predicates in the form of "Object-Relation-Object" triples, and retrieves visual assets using affordance-aware semantic embeddings from the GameTileNet dataset. A layered terrain is generated using Cellular Automata, and objects are placed using spatial rules grounded in the predicate structure. We evaluated our system in ten diverse stories, analyzing tile-object matching, affordance-layer alignment, and spatial constraint satisfaction across frames. This prototype offers a scalable approach to narrative-driven scene generation and lays the foundation for future work on multi-frame continuity, symbolic tracking, and multi-agent coordination in story-centered PCG.
>
---
#### [replaced 027] Do Vision Encoders Truly Explain Object Hallucination?: Mitigating Object Hallucination via Simple Fine-Grained CLIPScore
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言模型任务，旨在解决对象幻觉问题。通过提出F-CLIPScore评估指标，提升对象级检测精度，有效减少幻觉现象。**

- **链接: [https://arxiv.org/pdf/2502.20034v4](https://arxiv.org/pdf/2502.20034v4)**

> **作者:** Hongseok Oh; Wonseok Hwang
>
> **备注:** Transactions on Machine Learning Research
>
> **摘要:** Recently, Large Vision-Language Models (LVLMs) show remarkable performance across various domains. However, these models suffer from object hallucination. In this work, we study object hallucination primarily in a discriminative, retrieval-style evaluation setting (OHD-Caps), rather than in free-form caption generation. This study revisits the previous claim that the cause of such hallucinations lies in the limited representational capacity of the vision encoder. Our analysis implies that the capacity of the vision encoder is not necessarily a major limiting factor in detecting object hallucination. Based on this insight, we propose Fine-grained CLIPScore (F-CLIPScore), a simple yet effective evaluation metric that enhances object-level granularity by incorporating text embeddings at the noun level. Evaluations on the OHD-Caps benchmark show that F-CLIPScore significantly outperforms conventional CLIPScore in accuracy by a large margin of 39.6% without additional training. We further demonstrate that F-CLIPScore-based data filtering reduces object hallucination in LVLM (4.9% in POPE accuracy after alignment pretraining). Our code is publicly available at https://github.com/abzb1/f-clip
>
---
#### [replaced 028] RadarPLM: Adapting Pre-trained Language Models for Marine Radar Target Detection by Selective Fine-tuning
- **分类: eess.SP; cs.CL**

- **简介: 论文提出RadarPLM框架，解决海洋雷达目标检测问题。通过轻量适配模块和偏好损失函数，提升低信杂比下的检测性能。**

- **链接: [https://arxiv.org/pdf/2509.12089v4](https://arxiv.org/pdf/2509.12089v4)**

> **作者:** Qiying Hu; Yaowen Li; Xueqian Wang; Linping Zhang; Junlong Ke; Gang Li; Yu Liu; You He
>
> **摘要:** Recent advances in pre-trained language models (PLMs) have demonstrated their capabilities in capturing universal knowledge, making them promising for radar signal processing applications. Nevertheless, directly fine-tuning PLMs on radar signals is both computationally expensive and prone to overfitting, particularly in low signal-to-clutter ratio (SCR) environments. In this paper, we propose a novel fine-tuning framework for PLM-based marine radar target detection. First, we design a lightweight adaptation module, enabling computationally efficient fine-tuning while preserving the pre-trained model's general knowledge. Second, a novel preference-aware loss is developed to selectively optimize different feature patches based on their online-evaluated learning values, guiding the model to concentrate on those generalizable feature patterns during optimization. Finally, a binary classification head is retrained based on autoencoder network to further enhance detection performance. Experiments on real-world radar data show that the proposed RadarPLM framework yields at least a 6.35% improvement in detection performance over the existing networks under low SCR conditions. Especially, in small training samples cases,the proposed RadarPLM also achieves significant advantage over existing networks owing to the incorporation of the PLM.
>
---
#### [replaced 029] EXAONE 3.5: Series of Large Language Models for Real-world Use Cases
- **分类: cs.CL**

- **简介: 该论文介绍EXAONE 3.5大语言模型，解决实际应用场景中的指令遵循与长文本理解问题，提供三种规模模型并公开供研究使用。**

- **链接: [https://arxiv.org/pdf/2412.04862v3](https://arxiv.org/pdf/2412.04862v3)**

> **作者:** Soyoung An; Kyunghoon Bae; Eunbi Choi; Kibong Choi; Stanley Jungkyu Choi; Seokhee Hong; Junwon Hwang; Hyojin Jeon; Gerrard Jeongwon Jo; Hyunjik Jo; Jiyeon Jung; Yountae Jung; Hyosang Kim; Joonkee Kim; Seonghwan Kim; Soyeon Kim; Sunkyoung Kim; Yireun Kim; Yongil Kim; Youchul Kim; Edward Hwayoung Lee; Haeju Lee; Honglak Lee; Jinsik Lee; Kyungmin Lee; Woohyung Lim; Sangha Park; Sooyoun Park; Yongmin Park; Sihoon Yang; Heuiyeen Yeen; Hyeongu Yun
>
> **摘要:** This technical report introduces the EXAONE 3.5 instruction-tuned language models, developed and released by LG AI Research. The EXAONE 3.5 language models are offered in three configurations: 32B, 7.8B, and 2.4B. These models feature several standout capabilities: 1) exceptional instruction following capabilities in real-world scenarios, achieving the highest scores across seven benchmarks, 2) outstanding long-context comprehension, attaining the top performance in four benchmarks, and 3) competitive results compared to state-of-the-art open models of similar sizes across nine general benchmarks. The EXAONE 3.5 language models are open to anyone for research purposes and can be downloaded from https://huggingface.co/LGAI-EXAONE. For commercial use, please reach out to the official contact point of LG AI Research: contact_us@lgresearch.ai.
>
---
#### [replaced 030] Inner-Probe: Discovering Copyright-related Data Generation in LLM Architecture
- **分类: cs.CL**

- **简介: 该论文属于版权检测任务，旨在解决LLM生成文本中版权数据来源识别与非版权内容过滤问题。通过分析多头注意力结果，提出Inner-Probe框架提升检测效率与准确率。**

- **链接: [https://arxiv.org/pdf/2410.04454v3](https://arxiv.org/pdf/2410.04454v3)**

> **作者:** Qichao Ma; Rui-Jie Zhu; Peiye Liu; Renye Yan; Fahong Zhang; Ling Liang; Meng Li; Zhaofei Yu; Zongwei Wang; Yimao Cai; Tiejun Huang
>
> **备注:** Accepted by IEEE Transactions on Artificial Intelligence
>
> **摘要:** Large Language Models (LLMs) utilize extensive knowledge databases and show powerful text generation ability. However, their reliance on high-quality copyrighted datasets raises concerns about copyright infringements in generated texts. Current research often employs prompt engineering or semantic classifiers to identify copyrighted content, but these approaches have two significant limitations: (1) Challenging to identify which specific subdataset (e.g., works from particular authors) influences an LLM's output. (2) Treating the entire training database as copyrighted, hence overlooking the inclusion of non-copyrighted training data. We propose Inner-Probe, a lightweight framework designed to evaluate the influence of copyrighted sub-datasets on LLM-generated texts. Unlike traditional methods relying solely on text, we discover that the results of multi-head attention (MHA) during LLM output generation provide more effective information. Thus, Inner-Probe performs sub-dataset contribution analysis using a lightweight LSTM based network trained on MHA results in a supervised manner. Harnessing such a prior, Inner-Probe enables non-copyrighted text detection through a concatenated global projector trained with unsupervised contrastive learning. Inner-Probe demonstrates 3x improved efficiency compared to semantic model training in sub-dataset contribution analysis on Books3, achieves 15.04% - 58.7% higher accuracy over baselines on the Pile, and delivers a 0.104 increase in AUC for non-copyrighted data filtering.
>
---
#### [replaced 031] From Transformers to LLMs: A Systematic Survey of Efficiency Considerations in NLP
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于NLP效率研究任务，旨在解决Transformer模型计算资源消耗高的问题。通过系统综述312篇文献，分析数据、模型设计、压缩及推理等效率提升方法，并评估30余模型在13个基准上的表现。**

- **链接: [https://arxiv.org/pdf/2406.16893v2](https://arxiv.org/pdf/2406.16893v2)**

> **作者:** Wazib Ansar; Saptarsi Goswami; Amlan Chakrabarti
>
> **备注:** 63 pages, 5 tables and 22 figures
>
> **摘要:** The emergence of Transformer-based Large Language Models (LLMs) has substantially augmented the capabilities of Natural Language Processing (NLP), thereby intensifying the demand for computational resources. Therefore, enhancing efficiency based on factors like computational requirements, energy consumption, carbon footprint and financial cost has become a vital area of research. This motivates us to conduct a systematic literature review on Transformer-based LLMs in NLP from the perspective of efficiency. In this survey of 312 articles published between the years 2011 and 2025, efficiency-improvement endeavors have been systematically discussed targeting various aspects such as data curation, model design, model downsizing, and dynamic inferencing. This has been augmented with efficiency considerations in model adaptation strategies like pre-training, fine-tuning, prompt-engineering and Retrieval-Augmented Generation (RAG). Furthermore, a statistical analysis of the articles has been performed followed by an in-depth evaluation of the efficiency and efficacy of more than 30 renowned NLP models has been conducted on 13 evaluation benchmarks. This paper offers valuable insights for researchers, professionals as well as scholars, and explores the trend of research toward sustainable practices in NLP.
>
---
#### [replaced 032] Through a Compressed Lens: Investigating The Impact of Quantization on Factual Knowledge Recall
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，研究量化对事实知识回忆的影响。通过实验和分析，探讨量化导致的知识损失及性能变化，验证量化在压缩模型中的有效性。**

- **链接: [https://arxiv.org/pdf/2505.13963v2](https://arxiv.org/pdf/2505.13963v2)**

> **作者:** Qianli Wang; Mingyang Wang; Nils Feldhus; Simon Ostermann; Yuan Cao; Hinrich Schütze; Sebastian Möller; Vera Schmitt
>
> **备注:** In submission
>
> **摘要:** Quantization methods are widely used to accelerate inference and streamline the deployment of large language models (LLMs). Although quantization's effects on various LLM capabilities have been extensively studied, one critical area remains underexplored: factual knowledge recall (FKR), the process by which LLMs access stored knowledge. To this end, we conduct comprehensive experiments using three common quantization techniques at distinct bit widths, in conjunction with interpretability-driven analyses on two tasks, knowledge memorization and latent multi-hop reasoning. We show that quantization typically results in information loss within LLMs, consequently diminishing their capacity for FKR. This effect is particularly amplified in smaller models within the same architectural families. However, models quantized at reduced bit precision do not consistently exhibit inferior performance and occasionally quantization may even enhance model FKR. We find that BitSandBytes demonstrates highest preservation of the original full-precision model's FKR. Despite variability across models and methods, quantization causes modest performance degradation and remains an effective compression strategy.
>
---
#### [replaced 033] Optimizing Retrieval for RAG via Reinforcement Learning
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于信息检索任务，旨在解决RAG中检索器适应性差的问题。通过强化学习优化检索框架，提升RAG性能。**

- **链接: [https://arxiv.org/pdf/2510.24652v2](https://arxiv.org/pdf/2510.24652v2)**

> **作者:** Jiawei Zhou; Lei Chen
>
> **摘要:** As retrieval-augmented generation (RAG) becomes more widespread, the role of retrieval is shifting from retrieving information for human browsing to retrieving context for AI reasoning. This shift creates more complex search environments, where relevance is difficult to pre-define. Existing retrievers rely on supervised fine-tuning (SFT) with human labels or synthetic data, resulting in static relevance that struggles to adapt to diverse RAG environments. To address this challenge, we propose R3, a Retrieval framework optimized for RAG through Reinforcement learning (RL). Specifically, we adopt an RL training paradigm that enables the retriever to explore and self-improve within given RAG environments, automating the learning process with minimal manual experimentation or tuning effort. Extensive experiments across diverse tasks demonstrate that R3 improves RAG performance by 5.2% over the original retriever and surpasses state-of-the-art retrievers by 4.9%, while achieving comparable results to LLM-augmented retrieval and RAG systems built on post-trained or instruction-tuned LLMs. It is both efficient and practical, requiring only 4 GPUs and completing training within a single day.
>
---
#### [replaced 034] Dual LoRA: Enhancing LoRA with Magnitude and Direction Updates
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在提升LoRA方法的性能。针对LoRA低秩假设导致效果不佳的问题，提出Dual LoRA，通过分离参数更新的幅度和方向，改进微调效果。**

- **链接: [https://arxiv.org/pdf/2512.03402v4](https://arxiv.org/pdf/2512.03402v4)**

> **作者:** Yixing Xu; Chao Li; Xuanwu Yin; Spandan Tiwari; Dong Li; Ashish Sirasao; Emad Barsoum
>
> **摘要:** Low-rank adaptation (LoRA) is one of the most popular methods among parameter-efficient fine-tuning (PEFT) methods to adapt pre-trained large language models (LLMs) to specific downstream tasks. However, the model trained based on LoRA often has an unsatisfactory performance due to its low-rank assumption. In this paper, we propose a novel method called Dual LoRA to improve the performance by incorporating an inductive bias into the original LoRA. Specifically, we separate low-rank matrices into two groups: the magnitude group to control whether or not and how far we should update a parameter and the direction group to decide whether this parameter should move forward or backward, to better simulate the parameter updating process of the full fine-tuning based on gradient-based optimization algorithms. We show that this can be simply achieved by adding a ReLU function to the magnitude group and a sign function to the direction group. We conduct several experiments over a wide range of NLP tasks, including natural language understanding (NLU) and commonsense reasoning datasets on RoBERTa, DeBERTa, and LLaMA-1/2/3 as baseline models. The results show that we consistently outperform LoRA and its state-of-the-art variants with the same number of trainable parameters.
>
---
#### [replaced 035] EXAONE Deep: Reasoning Enhanced Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文介绍EXAONE Deep系列语言模型，专注于提升推理能力，解决数学和编码等任务中的复杂推理问题。通过训练于专门的推理数据集，模型在多个基准测试中表现优异。**

- **链接: [https://arxiv.org/pdf/2503.12524v3](https://arxiv.org/pdf/2503.12524v3)**

> **作者:** Kyunghoon Bae; Eunbi Choi; Kibong Choi; Stanley Jungkyu Choi; Yemuk Choi; Seokhee Hong; Junwon Hwang; Hyojin Jeon; Kijeong Jeon; Gerrard Jeongwon Jo; Hyunjik Jo; Jiyeon Jung; Hyosang Kim; Joonkee Kim; Seonghwan Kim; Soyeon Kim; Sunkyoung Kim; Yireun Kim; Yongil Kim; Youchul Kim; Edward Hwayoung Lee; Haeju Lee; Honglak Lee; Jinsik Lee; Kyungmin Lee; Sangha Park; Yongmin Park; Sihoon Yang; Heuiyeen Yeen; Sihyuk Yi; Hyeongu Yun
>
> **摘要:** We present EXAONE Deep series, which exhibits superior capabilities in various reasoning tasks, including math and coding benchmarks. We train our models mainly on the reasoning-specialized dataset that incorporates long streams of thought processes. Evaluation results show that our smaller models, EXAONE Deep 2.4B and 7.8B, outperform other models of comparable size, while the largest model, EXAONE Deep 32B, demonstrates competitive performance against leading open-weight models. All EXAONE Deep models are openly available for research purposes and can be downloaded from https://huggingface.co/LGAI-EXAONE.
>
---
#### [replaced 036] CubeBench: Diagnosing Interactive, Long-Horizon Spatial Reasoning Under Partial Observations
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出CubeBench，用于评估大语言模型在部分观察下的空间推理和长期规划能力，解决物理世界部署中的认知挑战。**

- **链接: [https://arxiv.org/pdf/2512.23328v3](https://arxiv.org/pdf/2512.23328v3)**

> **作者:** Huan-ang Gao; Zikang Zhang; Tianwei Luo; Kaisen Yang; Xinzhe Juan; Jiahao Qiu; Tianxing Chen; Bingxiang He; Hao Zhao; Hao Zhou; Shilong Liu; Mengdi Wang
>
> **备注:** Webpage: https://cubebench.c7w.tech/
>
> **摘要:** Large Language Model (LLM) agents, while proficient in the digital realm, face a significant gap in physical-world deployment due to the challenge of forming and maintaining a robust spatial mental model. We identify three core cognitive challenges hindering this transition: spatial reasoning, long-horizon state tracking via mental simulation, and active exploration under partial observation. To isolate and evaluate these faculties, we introduce CubeBench, a novel generative benchmark centered on the Rubik's Cube. CubeBench uses a three-tiered diagnostic framework that progressively assesses agent capabilities, from foundational state tracking with full symbolic information to active exploration with only partial visual data. Our experiments on leading LLMs reveal critical limitations, including a uniform 0.00% pass rate on all long-horizon tasks, exposing a fundamental failure in long-term planning. We also propose a diagnostic framework to isolate these cognitive bottlenecks by providing external solver tools. By analyzing the failure modes, we provide key insights to guide the development of more physically-grounded intelligent agents.
>
---
#### [replaced 037] Cultural Palette: Pluralising Culture Alignment via Multi-agent Palette
- **分类: cs.CL**

- **简介: 该论文属于文化对齐任务，旨在解决LLM在多文化场景下的适应性问题。通过构建多代理框架，融合文化地理信息，实现跨文化语境下的精准响应生成。**

- **链接: [https://arxiv.org/pdf/2412.11167v4](https://arxiv.org/pdf/2412.11167v4)**

> **作者:** Jiahao Yuan; Zixiang Di; Shangzixin Zhao; Zhiqing Cui; Hanqing Wang; Guisong Yang; Usman Naseem
>
> **备注:** 19 pages, 10 figures
>
> **摘要:** Large language models (LLMs) face challenges in aligning with diverse cultural values despite their remarkable performance in generation, which stems from inherent monocultural biases and difficulties in capturing nuanced cultural semantics. Existing methods struggle to adapt to unknown culture after fine-tuning. Inspired by cultural geography across five continents, we propose Cultural Palette, a multi-agent framework that redefines cultural alignment as an adaptive "color-blending" process for country-specific adaptation. Our approach harnesses cultural geography across five continents through three key steps: First, we synthesize the Pentachromatic Cultural Palette Dataset using GPT-4o, refining continental-level dialogues with Hofstede's cultural dimensions to establish foundational cultural representations. Second, five continent-level alignment agents form specialized cultural communities that generate region-specific draft responses. Third, a Meta Agent employs Cultural MoErges to dynamically blend these cultural "colors" through attention-gated parameter merging, akin to mixing pigments on a palette, resolving conflicts while preserving cultural nuances to produce the final culturally-aligned response. Extensive experiments across various countries demonstrate that \textit{Cultural Palette} surpasses existing baselines in cultural alignment.
>
---
#### [replaced 038] SpiderGen: Towards Procedure Generation For Carbon Life Cycle Assessments with Generative AI
- **分类: cs.CL; cs.CY**

- **简介: 论文提出SpiderGen，用于生成碳足迹评估的流程图。该任务旨在提高LCA效率，解决人工成本高、耗时的问题。工作包括整合LLM与传统LCA方法，生成PCR PFG，并验证其准确性。**

- **链接: [https://arxiv.org/pdf/2511.10684v4](https://arxiv.org/pdf/2511.10684v4)**

> **作者:** Anupama Sitaraman; Bharathan Balaji; Yuvraj Agarwal
>
> **摘要:** Investigating the effects of climate change and global warming caused by GHG emissions have been a key concern worldwide. These emissions are largely contributed to by the production, use and disposal of consumer products. Thus, it is important to build tools to estimate the environmental impact of consumer goods, an essential part of which is conducting Life Cycle Assessments (LCAs). LCAs specify and account for the appropriate processes involved with the production, use, and disposal of the products. We present SpiderGen, an LLM-based workflow which integrates the taxonomy and methodology of traditional LCA with the reasoning capabilities and world knowledge of LLMs to generate graphical representations of the key procedural information used for LCA, known as Product Category Rules Process Flow Graphs (PCR PFGs). We additionally evaluate the output of SpiderGen by comparing it with 65 real-world LCA documents. We find that SpiderGen provides accurate LCA process information that is either fully correct or has minor errors, achieving an F1-Score of 65% across 10 sample data points, as compared to 53% using a one-shot prompting method. We observe that the remaining errors occur primarily due to differences in detail between LCA documents, as well as differences in the "scope" of which auxiliary processes must also be included. We also demonstrate that SpiderGen performs better than several baselines techniques, such as chain-of-thought prompting and one-shot prompting. Finally, we highlight SpiderGen's potential to reduce the human effort and costs for estimating carbon impact, as it is able to produce LCA process information for less than \$1 USD in under 10 minutes as compared to the status quo LCA, which can cost over \$25000 USD and take up to 21-person days.
>
---
#### [replaced 039] EXAONE 3.0 7.8B Instruction Tuned Language Model
- **分类: cs.CL; cs.AI**

- **简介: 该论文介绍EXAONE 3.0语言模型，属于自然语言处理任务，旨在提升指令遵循能力，通过优化实现高性能和多语言支持。**

- **链接: [https://arxiv.org/pdf/2408.03541v4](https://arxiv.org/pdf/2408.03541v4)**

> **作者:** Soyoung An; Kyunghoon Bae; Eunbi Choi; Stanley Jungkyu Choi; Yemuk Choi; Seokhee Hong; Yeonjung Hong; Junwon Hwang; Hyojin Jeon; Gerrard Jeongwon Jo; Hyunjik Jo; Jiyeon Jung; Yountae Jung; Euisoon Kim; Hyosang Kim; Joonkee Kim; Seonghwan Kim; Soyeon Kim; Sunkyoung Kim; Yireun Kim; Youchul Kim; Edward Hwayoung Lee; Haeju Lee; Honglak Lee; Jinsik Lee; Kyungmin Lee; Moontae Lee; Seungjun Lee; Woohyung Lim; Sangha Park; Sooyoun Park; Yongmin Park; Boseong Seo; Sihoon Yang; Heuiyeen Yeen; Kyungjae Yoo; Hyeongu Yun
>
> **摘要:** We introduce EXAONE 3.0 instruction-tuned language model, the first open model in the family of Large Language Models (LLMs) developed by LG AI Research. Among different model sizes, we publicly release the 7.8B instruction-tuned model to promote open research and innovations. Through extensive evaluations across a wide range of public and in-house benchmarks, EXAONE 3.0 demonstrates highly competitive real-world performance with instruction-following capability against other state-of-the-art open models of similar size. Our comparative analysis shows that EXAONE 3.0 excels particularly in Korean, while achieving compelling performance across general tasks and complex reasoning. With its strong real-world effectiveness and bilingual proficiency, we hope that EXAONE keeps contributing to advancements in Expert AI. Our EXAONE 3.0 instruction-tuned model is available at https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct.
>
---
#### [replaced 040] CTTA-T: Continual Test-Time Adaptation for Text Understanding via Teacher-Student with a Domain-aware and Generalized Teacher
- **分类: cs.CL**

- **简介: 该论文属于文本理解任务，解决持续测试时适应（CTTA）问题，提出CTTA-T框架，通过师生结构和领域感知教师提升模型在未见领域的适应能力。**

- **链接: [https://arxiv.org/pdf/2512.18321v2](https://arxiv.org/pdf/2512.18321v2)**

> **作者:** Tianlun Liu; Zhiliang Tian; Zhen Huang; Xingzhi Zhou; Wanlong Yu; Tianle Liu; Feng Liu; Dongsheng Li
>
> **摘要:** Text understanding often suffers from domain shifts. To handle testing domains, domain adaptation (DA) is trained to adapt to a fixed and observed testing domain; a more challenging paradigm, test-time adaptation (TTA), cannot access the testing domain during training and online adapts to the testing samples during testing, where the samples are from a fixed domain. We aim to explore a more practical and underexplored scenario, continual test-time adaptation (CTTA) for text understanding, which involves a sequence of testing (unobserved) domains in testing. Current CTTA methods struggle in reducing error accumulation over domains and enhancing generalization to handle unobserved domains: 1) Noise-filtering reduces accumulated errors but discards useful information, and 2) accumulating historical domains enhances generalization, but it is hard to achieve adaptive accumulation. In this paper, we propose a CTTA-T (continual test-time adaptation for text understanding) framework adaptable to evolving target domains: it adopts a teacher-student framework, where the teacher is domain-aware and generalized for evolving domains. To improve teacher predictions, we propose a refine-then-filter based on dropout-driven consistency, which calibrates predictions and removes unreliable guidance. For the adaptation-generalization trade-off, we construct a domain-aware teacher by dynamically accumulating cross-domain semantics via incremental PCA, which continuously tracks domain shifts. Experiments show CTTA-T excels baselines.
>
---
#### [replaced 041] Sorbet: A Neuromorphic Hardware-Compatible Transformer-Based Spiking Language Model
- **分类: cs.NE; cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决在资源受限设备上高效运行语言模型的问题。提出Sorbet模型，采用脉冲神经网络，优化关键操作以提升能效。**

- **链接: [https://arxiv.org/pdf/2409.15298v2](https://arxiv.org/pdf/2409.15298v2)**

> **作者:** Kaiwen Tang; Zhanglu Yan; Weng-Fai Wong
>
> **备注:** Accepted by ICML 2025. Camera-ready version
>
> **摘要:** For reasons such as privacy, there are use cases for language models at the edge. This has given rise to small language models targeted for deployment in resource-constrained devices where energy efficiency is critical. Spiking neural networks (SNNs) offer a promising solution due to their energy efficiency, and there are already works on realizing transformer-based models on SNNs. However, key operations like softmax and layer normalization (LN) are difficult to implement on neuromorphic hardware, and many of these early works sidestepped them. To address these challenges, we introduce Sorbet, a transformer-based spiking language model that is more neuromorphic hardware-compatible. Sorbet incorporates a novel shifting-based softmax called PTsoftmax and a Bit Shifting PowerNorm (BSPN), both designed to replace the respective energy-intensive operations. By leveraging knowledge distillation and model quantization, Sorbet achieved a highly compressed binary weight model that maintains competitive performance while achieving $27.16\times$ energy savings compared to BERT. We validate Sorbet through extensive testing on the GLUE benchmark and a series of ablation studies, demonstrating its potential as an energy-efficient solution for language model inference. Our code is publicly available at \href{https://github.com/Kaiwen-Tang/Sorbet}{https://github.com/Kaiwen-Tang/Sorbet}
>
---
#### [replaced 042] C-VARC: A Large-Scale Chinese Value Rule Corpus for Value Alignment of Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于AI价值对齐任务，旨在解决LLM与人类价值观不一致的问题。构建了C-VARC语料库，涵盖中国核心价值观，提升模型在多元文化中的伦理评估能力。**

- **链接: [https://arxiv.org/pdf/2506.01495v5](https://arxiv.org/pdf/2506.01495v5)**

> **作者:** Ping Wu; Guobin Shen; Dongcheng Zhao; Yuwei Wang; Yiting Dong; Yu Shi; Enmeng Lu; Feifei Zhao; Yi Zeng
>
> **摘要:** Ensuring that Large Language Models (LLMs) align with mainstream human values and ethical norms is crucial for the safe and sustainable development of AI. Current value evaluation and alignment are constrained by Western cultural bias and incomplete domestic frameworks reliant on non-native rules; furthermore, the lack of scalable, rule-driven scenario generation methods makes evaluations costly and inadequate across diverse cultural contexts. To address these challenges, we propose a hierarchical value framework grounded in core Chinese values, encompassing three main dimensions, 12 core values, and 50 derived values. Based on this framework, we construct a large-scale Chinese Value Rule Corpus (C-VARC) containing over 250,000 value rules enhanced and expanded through human annotation. Experimental results demonstrate that scenarios guided by C-VARC exhibit clearer value boundaries and greater content diversity compared to those produced through direct generation. In the evaluation across six sensitive themes (e.g., surrogacy, suicide), seven mainstream LLMs preferred C-VARC generated options in over 70.5% of cases, while five Chinese human annotators showed an 87.5% alignment with C-VARC, confirming its universality, cultural relevance, and strong alignment with Chinese values. Additionally, we construct 400,000 rule-based moral dilemma scenarios that objectively capture nuanced distinctions in conflicting value prioritization across 17 LLMs. Our work establishes a culturally-adaptive benchmarking framework for comprehensive value evaluation and alignment, representing Chinese characteristics.
>
---
#### [replaced 043] Decide less, communicate more: On the construct validity of end-to-end fact-checking in medicine
- **分类: cs.CL**

- **简介: 该论文属于医疗事实核查任务，旨在解决医学领域中端到端事实核查的挑战。研究分析了临床专家如何验证社交媒体上的真实声明，揭示了连接声明与证据、意图模糊及主观判断等核心问题。**

- **链接: [https://arxiv.org/pdf/2506.20876v3](https://arxiv.org/pdf/2506.20876v3)**

> **作者:** Sebastian Joseph; Lily Chen; Barry Wei; Michael Mackert; Iain J. Marshall; Paul Pu Liang; Ramez Kouzy; Byron C. Wallace; Junyi Jessy Li
>
> **摘要:** Technological progress has led to concrete advancements in tasks that were regarded as challenging, such as automatic fact-checking. Interest in adopting these systems for public health and medicine has grown due to the high-stakes nature of medical decisions and challenges in critically appraising a vast and diverse medical literature. Evidence-based medicine connects to every individual, and yet the nature of it is highly technical, rendering the medical literacy of majority users inadequate to sufficiently navigate the domain. Such problems with medical communication ripens the ground for end-to-end fact-checking agents: check a claim against current medical literature and return with an evidence-backed verdict. And yet, such systems remain largely unused. In this position paper, developed with expert input, we present the first study examining how clinical experts verify real claims from social media by synthesizing medical evidence. In searching for this upper-bound, we reveal fundamental challenges in end-to-end fact-checking when applied to medicine: Difficulties connecting claims in the wild to scientific evidence in the form of clinical trials; ambiguities in underspecified claims mixed with mismatched intentions; and inherently subjective veracity labels. We argue that fact-checking should be approached and evaluated as an interactive communication problem, rather than an end-to-end process.
>
---
