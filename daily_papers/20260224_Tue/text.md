# 自然语言处理 cs.CL

- **最新发布 101 篇**

- **更新 92 篇**

## 最新发布

#### [new 001] From Trial by Fire To Sleep Like a Baby: A Lexicon of Anxiety Associations for 20k English Multiword Expressions
- **分类: cs.CL**

- **简介: 该论文构建了一个包含20k英语多词表达的焦虑关联词典，旨在研究焦虑与文本单位的关系，解决多词表达中焦虑关联的描述与分析问题。**

- **链接: [https://arxiv.org/pdf/2602.18692v1](https://arxiv.org/pdf/2602.18692v1)**

> **作者:** Saif M. Mohammad
>
> **摘要:** Anxiety is the unease about a possible future negative outcome. In recent years, there has been growing interest in understanding how anxiety relates to our health, well-being, body, mind, and behaviour. This includes work on lexical resources for word-anxiety association. However, there is very little anxiety-related work on larger units of text such as multiword expressions (MWE). Here, we introduce the first large-scale lexicon capturing descriptive norms of anxiety associations for more than 20k English MWEs. We show that the anxiety associations are highly reliable. We use the lexicon to study prevalence of different types of anxiety- and calmness-associated MWEs; and how that varies across two-, three-, and four-word sequences. We also study the extent to which the anxiety association of MWEs is compositional (due to its constituent words). The lexicon enables a wide variety of anxiety-related research in psychology, NLP, public health, and social sciences. The lexicon is freely available: https://saifmohammad.com/worrylex.html
>
---
#### [new 002] Keyboards for the Endangered Idu Mishmi Language
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2602.19815v1](https://arxiv.org/pdf/2602.19815v1)**

> **作者:** Akhilesh Kakolu Ramarao
>
> **摘要:** We present a mobile and desktop keyboard suite for Idu Mishmi, an endangered Trans-Himalayan language spoken by approximately 11,000 people in Arunachal Pradesh, India. Although a Latin-based orthography was developed in 2018, no digital input tools existed to use it, forcing speakers into ad-hoc romanizations that cannot represent the full writing system. Our keyboards comprise two tools: (1) an Android mobile keyboard, published on the Google Play Store and actively used in teacher training programs, and (2) a Windows desktop keyboard currently undergoing community testing. Both tools support the complete Idu Mishmi character inventory, including schwa, retracted schwa, nasalized vowels, and accented forms. Both operate fully offline with zero network permissions, addressing connectivity constraints and data sovereignty concerns. We describe the design, implementation, and deployment as a replicable model for other endangered language communities.
>
---
#### [new 003] Whisper: Courtside Edition Enhancing ASR Performance Through LLM-Driven Context Generation
- **分类: cs.CL**

- **简介: 该论文属于语音识别任务，旨在提升特定领域（如NBA赛事解说）的ASR性能。通过引入LLM驱动的上下文生成管道，优化Whisper的输出，显著降低词错误率。**

- **链接: [https://arxiv.org/pdf/2602.18966v1](https://arxiv.org/pdf/2602.18966v1)**

> **作者:** Yonathan Ron; Shiri Gilboa; Tammuz Dubnov
>
> **摘要:** Domain-specific speech remains a persistent challenge for automatic speech recognition (ASR), even for state-of-the-art systems like OpenAI's Whisper. We introduce Whisper: Courtside Edition, a novel multi-agent large language model (LLM) pipeline that enhances Whisper transcriptions without retraining. The pipeline intercepts Whisper's initial transcript, applies specialized LLM agents for domain context identification, named entity recognition, and jargon detection, and generates compact prompts that guide Whisper's decoder. Evaluated on 421 NBA basketball commentary segments (a domain characterized by dense proper nouns and technical terminology) our best pipeline achieves a statistically significant 17.0% relative reduction in word error rate (WER; from 0.217 to 0.180, p<0.001). Improvements are observed in 40.1% of segments with degradation in only 7.1%, substantially outperforming direct transcript post-editing. These results demonstrate that prompt-based augmentation can deliver scalable domain adaptation for ASR, offering a practical alternative to costly model fine-tuning.
>
---
#### [new 004] Denotational Semantics for ODRL: Knowledge-Based Constraint Conflict Detection
- **分类: cs.CL; cs.LO**

- **简介: 该论文属于知识推理任务，解决ODRL策略冲突检测问题。通过构建语义模型，将约束映射到知识库概念，实现跨数据空间的冲突检测与兼容性判断。**

- **链接: [https://arxiv.org/pdf/2602.19883v1](https://arxiv.org/pdf/2602.19883v1)**

> **作者:** Daham Mustafa; Diego Collarana; Yixin Peng; Rafiqul Haque; Christoph Lange-Bever; Christoph Quix; Stephan Decker
>
> **备注:** 17 pages, 6 tables. Working draft. Supplementary material (154 TPTP/SMT-LIB benchmarks, Isabelle/HOL theory file) will be made available at https://github.com/Daham-Mustaf/odrl-benchmark upon publication
>
> **摘要:** ODRL's six set-based operators -- isA, isPartOf, hasPart, isAnyOf, isAllOf, isNoneOf -- depend on external domain knowledge that the W3C specification leaves unspecified. Without it, every cross-dataspace policy comparison defaults to Unknown. We present a denotational semantics that maps each ODRL constraint to the set of knowledge-base concepts satisfying it. Conflict detection reduces to denotation intersection under a three-valued verdict -- Conflict, Compatible, or Unknown -- that is sound under incomplete knowledge. The framework covers all three ODRL composition modes (and, or, xone) and all three semantic domains arising in practice: taxonomic (class subsumption), mereological (part-whole containment), and nominal (identity). For cross-dataspace interoperability, we define order-preserving alignments between knowledge bases and prove two guarantees: conflicts are preserved across different KB standards, and unmapped concepts degrade gracefully to Unknown -- never to false conflicts. A runtime soundness theorem ensures that design-time verdicts hold for all execution contexts. The encoding stays within the decidable EPR fragment of first-order logic. We validate it with 154 benchmarks across six knowledge base families (GeoNames, ISO 3166, W3C DPV, a GDPR-derived taxonomy, BCP 47, and ISO 639-3) and four structural KBs targeting adversarial edge cases. Both the Vampire theorem prover and the Z3 SMT solver agree on all 154 verdicts. A key finding is that exclusive composition (xone) requires strictly stronger KB axioms than conjunction or disjunction: open-world semantics blocks exclusivity even when positive evidence appears to satisfy exactly one branch.
>
---
#### [new 005] Position: General Alignment Has Hit a Ceiling; Edge Alignment Must Be Taken Seriously
- **分类: cs.CL**

- **简介: 该论文属于AI对齐任务，旨在解决当前单一奖励机制在复杂场景中的局限性。提出Edge Alignment，强调多维价值结构与动态治理。**

- **链接: [https://arxiv.org/pdf/2602.20042v1](https://arxiv.org/pdf/2602.20042v1)**

> **作者:** Han Bao; Yue Huang; Xiaoda Wang; Zheyuan Zhang; Yujun Zhou; Carl Yang; Xiangliang Zhang; Yanfang Ye
>
> **备注:** 26 pages, 5 figures
>
> **摘要:** Large language models are being deployed in complex socio-technical systems, which exposes limits in current alignment practice. We take the position that the dominant paradigm of General Alignment, which compresses diverse human values into a single scalar reward, reaches a structural ceiling in settings with conflicting values, plural stakeholders, and irreducible uncertainty. These failures follow from the mathematics and incentives of scalarization and lead to \textbf{structural} value flattening, \textbf{normative} representation loss, and \textbf{cognitive} uncertainty blindness. We introduce Edge Alignment as a distinct approach in which systems preserve multi dimensional value structure, support plural and democratic representation, and incorporate epistemic mechanisms for interaction and clarification. To make this approach practical, we propose seven interdependent pillars organized into three phases. We identify key challenges in data collection, training objectives, and evaluation, outlining complementary technical and governance directions. Taken together, these measures reframe alignment as a lifecycle problem of dynamic normative governance rather than as a single instance optimization task.
>
---
#### [new 006] Sculpting the Vector Space: Towards Efficient Multi-Vector Visual Document Retrieval via Prune-then-Merge Framework
- **分类: cs.CL; cs.CV; cs.IR**

- **简介: 该论文属于视觉文档检索任务，旨在解决多向量方法压缩率与特征保真度的矛盾。提出Prune-then-Merge框架，先剪枝后合并，提升检索效率与效果。**

- **链接: [https://arxiv.org/pdf/2602.19549v1](https://arxiv.org/pdf/2602.19549v1)**

> **作者:** Yibo Yan; Mingdong Ou; Yi Cao; Xin Zou; Jiahao Huo; Shuliang Liu; James Kwok; Xuming Hu
>
> **备注:** Under review
>
> **摘要:** Visual Document Retrieval (VDR), which aims to retrieve relevant pages within vast corpora of visually-rich documents, is of significance in current multimodal retrieval applications. The state-of-the-art multi-vector paradigm excels in performance but suffers from prohibitive overhead, a problem that current efficiency methods like pruning and merging address imperfectly, creating a difficult trade-off between compression rate and feature fidelity. To overcome this dilemma, we introduce Prune-then-Merge, a novel two-stage framework that synergizes these complementary approaches. Our method first employs an adaptive pruning stage to filter out low-information patches, creating a refined, high-signal set of embeddings. Subsequently, a hierarchical merging stage compresses this pre-filtered set, effectively summarizing semantic content without the noise-induced feature dilution seen in single-stage methods. Extensive experiments on 29 VDR datasets demonstrate that our framework consistently outperforms existing methods, significantly extending the near-lossless compression range and providing robust performance at high compression ratios.
>
---
#### [new 007] Beyond a Single Extractor: Re-thinking HTML-to-Text Extraction for LLM Pretraining
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理中的数据预处理任务，旨在解决HTML文本提取效率问题。研究发现单一提取器限制数据覆盖，提出多提取器联合提升数据量，同时保持模型性能。**

- **链接: [https://arxiv.org/pdf/2602.19548v1](https://arxiv.org/pdf/2602.19548v1)**

> **作者:** Jeffrey Li; Josh Gardner; Doug Kang; Fangping Shi; Karanjeet Singh; Chun-Liang Li; Herumb Shandilya; David Hall; Oncel Tuzel; Percy Liang; Ludwig Schmidt; Hadi Pour Ansari; Fartash Faghri
>
> **摘要:** One of the first pre-processing steps for constructing web-scale LLM pretraining datasets involves extracting text from HTML. Despite the immense diversity of web content, existing open-source datasets predominantly apply a single fixed extractor to all webpages. In this work, we investigate whether this practice leads to suboptimal coverage and utilization of Internet data. We first show that while different extractors may lead to similar model performance on standard language understanding tasks, the pages surviving a fixed filtering pipeline can differ substantially. This suggests a simple intervention: by taking a Union over different extractors, we can increase the token yield of DCLM-Baseline by up to 71% while maintaining benchmark performance. We further show that for structured content such as tables and code blocks, extractor choice can significantly impact downstream task performance, with differences of up to 10 percentage points (p.p.) on WikiTQ and 3 p.p. on HumanEval.
>
---
#### [new 008] The Million-Label NER: Breaking Scale Barriers with GLiNER bi-encoder
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于命名实体识别任务，解决大规模实体类型识别效率低的问题。提出GLiNER-bi-Encoder架构，提升处理百万级实体类型的效率与性能。**

- **链接: [https://arxiv.org/pdf/2602.18487v1](https://arxiv.org/pdf/2602.18487v1)**

> **作者:** Ihor Stepanov; Mykhailo Shtopko; Dmytro Vodianytskyi; Oleksandr Lukashov
>
> **备注:** 13 pages, 1 figure, 4 tables
>
> **摘要:** This paper introduces GLiNER-bi-Encoder, a novel architecture for Named Entity Recognition (NER) that harmonizes zero-shot flexibility with industrial-scale efficiency. While the original GLiNER framework offers strong generalization, its joint-encoding approach suffers from quadratic complexity as the number of entity labels increases. Our proposed bi-encoder design decouples the process into a dedicated label encoder and a context encoder, effectively removing the context-window bottleneck. This architecture enables the simultaneous recognition of thousands, and potentially millions, of entity types with minimal overhead. Experimental results demonstrate state-of-the-art zero-shot performance, achieving 61.5 percent Micro-F1 on the CrossNER benchmark. Crucially, by leveraging pre-computed label embeddings, GLiNER-bi-Encoder achieves up to a 130 times throughput improvement at 1024 labels compared to its uni-encoder predecessors. Furthermore, we introduce GLiNKER, a modular framework that leverages this architecture for high-performance entity linking across massive knowledge bases such as Wikidata.
>
---
#### [new 009] Think$^{2}$: Grounded Metacognitive Reasoning in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI可解释性任务，旨在提升大语言模型的自我监控与纠错能力。通过引入元认知框架，增强模型的自省与诊断能力。**

- **链接: [https://arxiv.org/pdf/2602.18806v1](https://arxiv.org/pdf/2602.18806v1)**

> **作者:** Abraham Paul Elenjical; Vivek Hruday Kavuri; Vasudeva Varma
>
> **摘要:** Large Language Models (LLMs) demonstrate strong reasoning performance, yet their ability to reliably monitor, diagnose, and correct their own errors remains limited. We introduce a psychologically grounded metacognitive framework that operationalizes Ann Brown's regulatory cycle (Planning, Monitoring, and Evaluation) as a structured prompting architecture, and study its integration within a lightweight dual-process MetaController for adaptive effort allocation. Across diverse reasoning and diagnostic benchmarks (GSM8K, CRUXEval, MBPP, AIME, CorrectBench, and TruthfulQA) using Llama-3 and Qwen-3 (8B), explicit regulatory structuring substantially improves error diagnosis and yields a threefold increase in successful self-correction. Blinded human evaluations over 580 query pairs show an 84% aggregate preference for trustworthiness and metacognitive self-awareness over standard and Chain-of-Thought baselines. Grounding LLM reasoning in established cognitive theory offers a principled path toward more transparent and diagnostically robust AI systems.
>
---
#### [new 010] NanoKnow: How to Know What Your Language Model Knows
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **简介: 该论文属于知识推理任务，旨在探究大语言模型的知识来源。通过构建NanoKnow数据集，分析模型在有无外部证据情况下的表现，揭示参数知识与外部知识的互补性及非相关信息的影响。**

- **链接: [https://arxiv.org/pdf/2602.20122v1](https://arxiv.org/pdf/2602.20122v1)**

> **作者:** Lingwei Gu; Nour Jedidi; Jimmy Lin
>
> **摘要:** How do large language models (LLMs) know what they know? Answering this question has been difficult because pre-training data is often a "black box" -- unknown or inaccessible. The recent release of nanochat -- a family of small LLMs with fully open pre-training data -- addresses this as it provides a transparent view into where a model's parametric knowledge comes from. Towards the goal of understanding how knowledge is encoded by LLMs, we release NanoKnow, a benchmark dataset that partitions questions from Natural Questions and SQuAD into splits based on whether their answers are present in nanochat's pre-training corpus. Using these splits, we can now properly disentangle the sources of knowledge that LLMs rely on when producing an output. To demonstrate NanoKnow's utility, we conduct experiments using eight nanochat checkpoints. Our findings show: (1) closed-book accuracy is strongly influenced by answer frequency in the pre-training data, (2) providing external evidence can mitigate this frequency dependence, (3) even with external evidence, models are more accurate when answers were seen during pre-training, demonstrating that parametric and external knowledge are complementary, and (4) non-relevant information is harmful, with accuracy decreasing based on both the position and the number of non-relevant contexts. We release all NanoKnow artifacts at https://github.com/castorini/NanoKnow.
>
---
#### [new 011] Multilingual Large Language Models do not comprehend all natural languages to equal degrees
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，研究多语言大模型在不同语言中的理解能力。旨在解决模型对非英语语言表现差异的问题，通过实验分析多种语言的表现因素。**

- **链接: [https://arxiv.org/pdf/2602.20065v1](https://arxiv.org/pdf/2602.20065v1)**

> **作者:** Natalia Moskvina; Raquel Montero; Masaya Yoshida; Ferdy Hubers; Paolo Morosi; Walid Irhaymi; Jin Yan; Tamara Serrano; Elena Pagliarini; Fritz Günther; Evelina Leivada
>
> **备注:** 36 pages, 3 figures, 2 tables, 4 supplementary tables
>
> **摘要:** Large Language Models (LLMs) play a critical role in how humans access information. While their core use relies on comprehending written requests, our understanding of this ability is currently limited, because most benchmarks evaluate LLMs in high-resource languages predominantly spoken by Western, Educated, Industrialised, Rich, and Democratic (WEIRD) communities. The default assumption is that English is the best-performing language for LLMs, while smaller, low-resource languages are linked to less reliable outputs, even in multilingual, state-of-the-art models. To track variation in the comprehension abilities of LLMs, we prompt 3 popular models on a language comprehension task across 12 languages, representing the Indo-European, Afro-Asiatic, Turkic, Sino-Tibetan, and Japonic language families. Our results suggest that the models exhibit remarkable linguistic accuracy across typologically diverse languages, yet they fall behind human baselines in all of them, albeit to different degrees. Contrary to what was expected, English is not the best-performing language, as it was systematically outperformed by several Romance languages, even lower-resource ones. We frame the results by discussing the role of several factors that drive LLM performance, such as tokenization, language distance from Spanish and English, size of training data, and data origin in high- vs. low-resource languages and WEIRD vs. non-WEIRD communities.
>
---
#### [new 012] Cross-lingual Matryoshka Representation Learning across Speech and Text
- **分类: cs.CL**

- **简介: 该论文属于跨语言语音与文本表示学习任务，旨在解决低资源语言的语种和模态障碍。通过构建双语语音-文本Matryoshka模型，实现法语文本从沃洛夫语语音查询的高效检索。**

- **链接: [https://arxiv.org/pdf/2602.19991v1](https://arxiv.org/pdf/2602.19991v1)**

> **作者:** Yaya Sy; Dioula Doucouré; Christophe Cerisara; Irina Illina
>
> **备注:** Preprint, under review
>
> **摘要:** Speakers of under-represented languages face both a language barrier, as most online knowledge is in a few dominant languages, and a modality barrier, since information is largely text-based while many languages are primarily oral. We address this for French-Wolof by training the first bilingual speech-text Matryoshka embedding model, enabling efficient retrieval of French text from Wolof speech queries without relying on a costly ASR-translation pipelines. We introduce large-scale data curation pipelines and new benchmarks, compare modeling strategies, and show that modality fusion within a frozen text Matryoshka model performs best. Although trained only for retrieval, the model generalizes well to other tasks, such as speech intent detection, indicating the learning of general semantic representations. Finally, we analyze cost-accuracy trade-offs across Matryoshka dimensions and ranks, showing that information is concentrated only in a few components, suggesting potential for efficiency improvements.
>
---
#### [new 013] Semantic Substrate Theory: An Operator-Theoretic Framework for Geometric Semantic Drift
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出一种几何语义漂移的理论框架，解决多信号关联问题。通过构建时间索引的语义基质，分析局部扩散与结构变化，预测未来邻域重连。**

- **链接: [https://arxiv.org/pdf/2602.18699v1](https://arxiv.org/pdf/2602.18699v1)**

> **作者:** Stephen Russell
>
> **摘要:** Most semantic drift studies report multiple signals e.g., embedding displacement, neighbor changes, distributional divergence, and recursive trajectory instability, without a shared explanatory theory that relates them. This paper proposes a formalization of these signals in one time-indexed substrate, $S_t=(X,d_t,P_t)$, combining embedding geometry with local diffusion. Within this substrate, node-level neighborhood drift measures changes in local conditional distributions, coarse Ricci curvature measures local contractivity of semantic diffusion, and recursive drift probes stability of iterated semantic operators. This manuscript specifies the formal model, assumptions, and tests that can refute the model. Herein, the paper introduces bridge mass, a node-level aggregate of incident negative curvature, as a predictor of future neighborhood rewiring. This paper provides the theory and test contracts; empirical performance is deferred to subsequent studies.
>
---
#### [new 014] Temporal-Aware Heterogeneous Graph Reasoning with Multi-View Fusion for Temporal Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于时间问答任务，解决TKGQA中时间约束弱、多跳推理不足和表示融合差的问题。提出包含时间感知编码、多跳图推理和多视角融合的新框架。**

- **链接: [https://arxiv.org/pdf/2602.19569v1](https://arxiv.org/pdf/2602.19569v1)**

> **作者:** Wuzhenghong Wen; Bowen Zhou; Jinwen Huang; Xianjie Wu; Yuwei Sun; Su Pan; Liang Li; Jianting Liu
>
> **备注:** 6pages
>
> **摘要:** Question Answering over Temporal Knowledge Graphs (TKGQA) has attracted growing interest for handling time-sensitive queries. However, existing methods still struggle with: 1) weak incorporation of temporal constraints in question representation, causing biased reasoning; 2) limited ability to perform explicit multi-hop reasoning; and 3) suboptimal fusion of language and graph representations. We propose a novel framework with temporal-aware question encoding, multi-hop graph reasoning, and multi-view heterogeneous information fusion. Specifically, our approach introduces: 1) a constraint-aware question representation that combines semantic cues from language models with temporal entity dynamics; 2) a temporal-aware graph neural network for explicit multi-hop reasoning via time-aware message passing; and 3) a multi-view attention mechanism for more effective fusion of question context and temporal graph knowledge. Experiments on multiple TKGQA benchmarks demonstrate consistent improvements over multiple baselines.
>
---
#### [new 015] ConfSpec: Efficient Step-Level Speculative Reasoning via Confidence-Gated Verification
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出ConfSpec，解决大模型推理中的效率与准确率矛盾问题，通过置信度门控验证提升推理速度。属于高效推理任务。**

- **链接: [https://arxiv.org/pdf/2602.18447v1](https://arxiv.org/pdf/2602.18447v1)**

> **作者:** Siran Liu; Cyril Y. He
>
> **摘要:** Chain-of-Thought reasoning significantly improves the performance of large language models on complex tasks, but incurs high inference latency due to long generation traces. Step-level speculative reasoning aims to mitigate this cost, yet existing approaches face a long-standing trade-off among accuracy, inference speed, and resource efficiency. We propose ConfSpec, a confidence-gated cascaded verification framework that resolves this trade-off. Our key insight is an asymmetry between generation and verification: while generating a correct reasoning step requires substantial model capacity, step-level verification is a constrained discriminative task for which small draft models are well-calibrated within their competence range, enabling high-confidence draft decisions to be accepted directly while selectively escalating uncertain cases to the large target model. Evaluation across diverse workloads shows that ConfSpec achieves up to 2.24$\times$ end-to-end speedups while matching target-model accuracy. Our method requires no external judge models and is orthogonal to token-level speculative decoding, enabling further multiplicative acceleration.
>
---
#### [new 016] Hyper-KGGen: A Skill-Driven Knowledge Extractor for High-Quality Knowledge Hypergraph Generation
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于知识图谱构建任务，旨在解决跨领域知识超图生成中的场景差距问题。提出Hyper-KGGen框架，通过动态技能演化提升抽取质量。**

- **链接: [https://arxiv.org/pdf/2602.19543v1](https://arxiv.org/pdf/2602.19543v1)**

> **作者:** Rizhuo Huang; Yifan Feng; Rundong Xue; Shihui Ying; Jun-Hai Yong; Chuan Shi; Shaoyi Du; Yue Gao
>
> **摘要:** Knowledge hypergraphs surpass traditional binary knowledge graphs by encapsulating complex $n$-ary atomic facts, providing a more comprehensive paradigm for semantic representation. However, constructing high-quality hypergraphs remains challenging due to the \textit{scenario gap}: generic extractors struggle to generalize across diverse domains with specific jargon, while existing methods often fail to balance structural skeletons with fine-grained details. To bridge this gap, we propose \textbf{Hyper-KGGen}, a skill-driven framework that reformulates extraction as a dynamic skill-evolving process. First, Hyper-KGGen employs a \textit{coarse-to-fine} mechanism to systematically decompose documents, ensuring full-dimensional coverage from binary links to complex hyperedges. Crucially, it incorporates an \textit{adaptive skill acquisition} module that actively distills domain expertise into a Global Skill Library. This is achieved via a stability-based feedback loop, where extraction stability serves as a relative reward signal to induce high-quality skills from unstable traces and missed predictions. Additionally, we present \textbf{HyperDocRED}, a rigorously annotated benchmark for document-level knowledge hypergraph extraction. Experiments demonstrate that Hyper-KGGen significantly outperforms strong baselines, validating that evolved skills provide substantially richer guidance than static few-shot examples in multi-scenario settings.
>
---
#### [new 017] Contradiction to Consensus: Dual Perspective, Multi Source Retrieval Based Claim Verification with Source Level Disagreement using LLM
- **分类: cs.CL**

- **简介: 该论文属于事实核查任务，旨在解决单一知识源导致的验证局限问题。通过多源证据检索与矛盾分析，构建更全面的知识库，提升验证效果与透明度。**

- **链接: [https://arxiv.org/pdf/2602.18693v1](https://arxiv.org/pdf/2602.18693v1)**

> **作者:** Md Badsha Biswas; Ozlem Uzuner
>
> **摘要:** The spread of misinformation across digital platforms can pose significant societal risks. Claim verification, a.k.a. fact-checking, systems can help identify potential misinformation. However, their efficacy is limited by the knowledge sources that they rely on. Most automated claim verification systems depend on a single knowledge source and utilize the supporting evidence from that source; they ignore the disagreement of their source with others. This limits their knowledge coverage and transparency. To address these limitations, we present a novel system for open-domain claim verification (ODCV) that leverages large language models (LLMs), multi-perspective evidence retrieval, and cross-source disagreement analysis. Our approach introduces a novel retrieval strategy that collects evidence for both the original and the negated forms of a claim, enabling the system to capture supporting and contradicting information from diverse sources: Wikipedia, PubMed, and Google. These evidence sets are filtered, deduplicated, and aggregated across sources to form a unified and enriched knowledge base that better reflects the complexity of real-world information. This aggregated evidence is then used for claim verification using LLMs. We further enhance interpretability by analyzing model confidence scores to quantify and visualize inter-source disagreement. Through extensive evaluation on four benchmark datasets with five LLMs, we show that knowledge aggregation not only improves claim verification but also reveals differences in source-specific reasoning. Our findings underscore the importance of embracing diversity, contradiction, and aggregation in evidence for building reliable and transparent claim verification systems
>
---
#### [new 018] ArabicNumBench: Evaluating Arabic Number Reading in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在评估大语言模型在阿拉伯数字识别中的表现。通过构建基准测试，分析不同提示策略的效果，揭示模型在准确性和指令遵循上的差异。**

- **链接: [https://arxiv.org/pdf/2602.18776v1](https://arxiv.org/pdf/2602.18776v1)**

> **作者:** Anas Alhumud; Abdulaziz Alhammadi; Muhammad Badruddin Khan
>
> **摘要:** We present ArabicNumBench, a comprehensive benchmark for evaluating large language models on Arabic number reading tasks across Eastern Arabic-Indic numerals (0-9 in Arabic script) and Western Arabic numerals (0-9). We evaluate 71 models from 10 providers using four prompting strategies (zero-shot, zero-shot CoT, few-shot, few-shot CoT) on 210 number reading tasks spanning six contextual categories: pure numerals, addresses, dates, quantities, and prices. Our evaluation comprises 59,010 individual test cases and tracks extraction methods to measure structured output generation. Evaluation reveals substantial performance variation, with accuracy ranging from 14.29\% to 99.05\% across models and strategies. Few-shot Chain-of-Thought prompting achieves 2.8x higher accuracy than zero-shot approaches (80.06\% vs 28.76\%). A striking finding emerges: models achieving elite accuracy (98-99\%) often produce predominantly unstructured output, with most responses lacking Arabic CoT markers. Only 6 models consistently generate structured output across all test cases, while the majority require fallback extraction methods despite high numerical accuracy. Comprehensive evaluation of 281 model-strategy combinations demonstrates that numerical accuracy and instruction-following represent distinct capabilities, establishing baselines for Arabic number comprehension and providing actionable guidance for model selection in production Arabic NLP systems.
>
---
#### [new 019] AgenticSum: An Agentic Inference-Time Framework for Faithful Clinical Text Summarization
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于临床文本摘要任务，旨在解决LLM生成摘要时的不一致问题。提出AgenticSum框架，通过分阶段处理提升摘要准确性。**

- **链接: [https://arxiv.org/pdf/2602.20040v1](https://arxiv.org/pdf/2602.20040v1)**

> **作者:** Fahmida Liza Piya; Rahmatollah Beheshti
>
> **摘要:** Large language models (LLMs) offer substantial promise for automating clinical text summarization, yet maintaining factual consistency remains challenging due to the length, noise, and heterogeneity of clinical documentation. We present AgenticSum, an inference-time, agentic framework that separates context selection, generation, verification, and targeted correction to reduce hallucinated content. The framework decomposes summarization into coordinated stages that compress task-relevant context, generate an initial draft, identify weakly supported spans using internal attention grounding signals, and selectively revise flagged content under supervisory control. We evaluate AgenticSum on two public datasets, using reference-based metrics, LLM-as-a-judge assessment, and human evaluation. Across various measures, AgenticSum demonstrates consistent improvements compared to vanilla LLMs and other strong baselines. Our results indicate that structured, agentic design with targeted correction offers an effective inference time solution to improve clinical note summarization using LLMs.
>
---
#### [new 020] BURMESE-SAN: Burmese NLP Benchmark for Evaluating Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出BURMESE-SAN，首个针对缅甸语的NLP基准，评估大语言模型在理解、推理和生成任务中的表现，解决低资源语言建模难题。**

- **链接: [https://arxiv.org/pdf/2602.18788v1](https://arxiv.org/pdf/2602.18788v1)**

> **作者:** Thura Aung; Jann Railey Montalan; Jian Gang Ngui; Peerat Limkonchotiwat
>
> **摘要:** We introduce BURMESE-SAN, the first holistic benchmark that systematically evaluates large language models (LLMs) for Burmese across three core NLP competencies: understanding (NLU), reasoning (NLR), and generation (NLG). BURMESE-SAN consolidates seven subtasks spanning these competencies, including Question Answering, Sentiment Analysis, Toxicity Detection, Causal Reasoning, Natural Language Inference, Abstractive Summarization, and Machine Translation, several of which were previously unavailable for Burmese. The benchmark is constructed through a rigorous native-speaker-driven process to ensure linguistic naturalness, fluency, and cultural authenticity while minimizing translation-induced artifacts. We conduct a large-scale evaluation of both open-weight and commercial LLMs to examine challenges in Burmese modeling arising from limited pretraining coverage, rich morphology, and syntactic variation. Our results show that Burmese performance depends more on architectural design, language representation, and instruction tuning than on model scale alone. In particular, Southeast Asia regional fine-tuning and newer model generations yield substantial gains. Finally, we release BURMESE-SAN as a public leaderboard to support systematic evaluation and sustained progress in Burmese and other low-resource languages. https://leaderboard.sea-lion.ai/detailed/MY
>
---
#### [new 021] Capable but Unreliable: Canonical Path Deviation as a Causal Mechanism of Agent Failure in Long-Horizon Tasks
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究语言代理在长任务中失败的原因，发现是由于随机偏差偏离了标准解决路径，而非能力不足。通过实验验证了路径偏离的因果机制，并提出干预措施提升可靠性。**

- **链接: [https://arxiv.org/pdf/2602.19008v1](https://arxiv.org/pdf/2602.19008v1)**

> **作者:** Wilson Y. Lee
>
> **摘要:** Why do language agents fail on tasks they are capable of solving? We argue that many such failures are reliability failures caused by stochastic drift from a task's latent solution structure, not capability failures. Every well-defined tool-use task imposes a canonical solution path (i.e., a convergent set of tool invocations shared across successful runs) and agent success depends critically on whether a trajectory stays within this path's operating envelope. We establish this causally using a natural experiment that holds model capability and task difficulty fixed by construction. We analyze trajectories from the Toolathlon benchmark: 22 frontier models each attempt 108 real-world tool-use tasks across 3 independent runs, yielding 515 model$\times$task units where the same model succeeds on some runs and fails on others due to LLM sampling stochasticity alone. Within these units, successful runs adhere significantly more closely to the canonical solution path than failed runs ($+$0.060 Jaccard, $p<0.0001$, $n=488$ units, 95% CI [+0.043, +0.077]). This result survives six robustness checks including cross-model-family leave-one-out validation. Critically, the causal mechanism is gradual and self-reinforcing: the adherence gap is statistically indistinguishable from zero through the first 50% of the trajectory, ruling out early-branching selection bias, and each off-canonical tool call raises the probability that the next call is also off-canonical by 22.7 percentage points ($\hatβ=+0.227$, $p<0.0001$), more than doubling the baseline rate. These findings imply that agent reliability cannot be improved by capability scaling alone, but offer a highly actionable intervention: a simple monitor that restarts the bottom tercile of runs based on mid-trajectory canonical adherence lifts success rates by $+$8.8 percentage points among intervened runs.
>
---
#### [new 022] ReportLogic: Evaluating Logical Quality in Deep Research Reports
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于逻辑质量评估任务，旨在解决LLM生成报告的逻辑可靠性问题。通过构建ReportLogic基准和LogicJudge模型，评估报告结构、解释性和结论支持性。**

- **链接: [https://arxiv.org/pdf/2602.18446v1](https://arxiv.org/pdf/2602.18446v1)**

> **作者:** Jujia Zhao; Zhaoxin Huan; Zihan Wang; Xiaolu Zhang; Jun Zhou; Suzan Verberne; Zhaochun Ren
>
> **摘要:** Users increasingly rely on Large Language Models (LLMs) for Deep Research, using them to synthesize diverse sources into structured reports that support understanding and action. In this context, the practical reliability of such reports hinges on logical quality: whether the report's claims and arguments are explicitly supported and can be trusted as a basis for downstream use, rather than merely appearing fluent or informative. However, current evaluation frameworks largely overlook this requirement. To bridge this gap, we introduce ReportLogic, a benchmark that quantifies report-level logical quality through a reader-centric lens of auditability. Specifically, ReportLogic adopts a hierarchical taxonomy that evaluates whether readers can (1) trace an on-topic report structure with a unified analytical arc (Macro-Logic), (2) understand the progression with necessary context (Expositional-Logic), and (3) verify conclusions via explicit claim--support (Structural-Logic). Based on this taxonomy, we construct a human-annotated rubric-guided dataset and train an open-source LogicJudge for scalable evaluation. We further evaluate judge robustness via adversarial attacks, showing that off-the-shelf LLM judges are frequently influenced by superficial cues (e.g., verbosity), and reasoning modes can mask broken support relations. Overall, our results provide actionable guidance for building more robust logic evaluators and improving the logical reliability of LLM-generated reports.
>
---
#### [new 023] Facet-Level Persona Control by Trait-Activated Routing with Contrastive SAE for Role-Playing LLMs
- **分类: cs.CL**

- **简介: 该论文属于角色扮演语言模型的人格控制任务，旨在解决传统方法在长期对话中人格一致性差的问题。通过构建对比稀疏自编码器框架，学习 facet 级别控制向量，提升角色扮演的稳定性与可解释性。**

- **链接: [https://arxiv.org/pdf/2602.19157v1](https://arxiv.org/pdf/2602.19157v1)**

> **作者:** Wenqiu Tang; Zhen Wan; Takahiro Komamizu; Ichiro Ide
>
> **备注:** Accepted in PAKDD 2026 special session on Data Science :Foundation and Applications
>
> **摘要:** Personality control in Role-Playing Agents (RPAs) is commonly achieved via training-free methods that inject persona descriptions and memory through prompts or retrieval-augmented generation, or via supervised fine-tuning (SFT) on persona-specific corpora. While SFT can be effective, it requires persona-labeled data and retraining for new roles, limiting flexibility. In contrast, prompt- and RAG-based signals are easy to apply but can be diluted in long dialogues, leading to drifting and sometimes inconsistent persona behavior. To address this, we propose a contrastive Sparse AutoEncoder (SAE) framework that learns facet-level personality control vectors aligned with the Big Five 30-facet model. A new 15,000-sample leakage-controlled corpus is constructed to provide balanced supervision for each facet. The learned vectors are integrated into the model's residual space and dynamically selected by a trait-activated routing module, enabling precise and interpretable personality steering. Experiments on Large Language Models (LLMs) show that the proposed method maintains stable character fidelity and output quality across contextualized settings, outperforming Contrastive Activation Addition (CAA) and prompt-only baselines. The combined SAE+Prompt configuration achieves the best overall performance, confirming that contrastively trained latent vectors can enhance persona control while preserving dialogue coherence.
>
---
#### [new 024] EvalSense: A Framework for Domain-Specific LLM (Meta-)Evaluation
- **分类: cs.CL**

- **简介: 该论文提出EvalSense框架，用于构建特定领域的LLM评估体系，解决传统评估方法在开放生成任务中的不足。**

- **链接: [https://arxiv.org/pdf/2602.18823v1](https://arxiv.org/pdf/2602.18823v1)**

> **作者:** Adam Dejl; Jonathan Pearson
>
> **备注:** Accepted to EACL 2026 System Demonstrations
>
> **摘要:** Robust and comprehensive evaluation of large language models (LLMs) is essential for identifying effective LLM system configurations and mitigating risks associated with deploying LLMs in sensitive domains. However, traditional statistical metrics are poorly suited to open-ended generation tasks, leading to growing reliance on LLM-based evaluation methods. These methods, while often more flexible, introduce additional complexity: they depend on carefully chosen models, prompts, parameters, and evaluation strategies, making the evaluation process prone to misconfiguration and bias. In this work, we present EvalSense, a flexible, extensible framework for constructing domain-specific evaluation suites for LLMs. EvalSense provides out-of-the-box support for a broad range of model providers and evaluation strategies, and assists users in selecting and deploying suitable evaluation methods for their specific use-cases. This is achieved through two unique components: (1) an interactive guide aiding users in evaluation method selection and (2) automated meta-evaluation tools that assess the reliability of different evaluation approaches using perturbed data. We demonstrate the effectiveness of EvalSense in a case study involving the generation of clinical notes from unstructured doctor-patient dialogues, using a popular open dataset. All code, documentation, and assets associated with EvalSense are open-source and publicly available at https://github.com/nhsengland/evalsense.
>
---
#### [new 025] Uncovering Context Reliance in Unstructured Knowledge Editing
- **分类: cs.CL**

- **简介: 该论文属于语言模型知识编辑任务，旨在解决编辑后知识依赖上下文导致的召回失败问题。通过提出COIN框架，减少上下文依赖，提升编辑效果。**

- **链接: [https://arxiv.org/pdf/2602.19043v1](https://arxiv.org/pdf/2602.19043v1)**

> **作者:** Zisheng Zhou; Mengqi Zhang; Shiguang Wu; Xiaotian Ye; Chi Zhang; Zhumin Chen; Pengjie Ren
>
> **备注:** 21 pages, 14 figures
>
> **摘要:** Editing Large language models (LLMs) with real-world, unstructured knowledge is essential for correcting and updating their internal parametric knowledge. In this work, we revisit the fundamental next-token prediction (NTP) as a candidate paradigm for unstructured editing. We identify Context Reliance as a critical failure mode of NTP-based approaches, where knowledge acquired from edited text becomes highly dependent on its preceding context, leading to recall failures when that context is absent during inference. This hypothesis is supported by our empirical validation that prepending context during inference recovers knowledge recall. We further theoretically demonstrate that Context Reliance is an inherent consequence of gradient-based optimization, which tends to bind acquired knowledge to a specific aggregated contextual representation. To address this, we propose a simple yet effective COntext-INdependent editing framework (COIN), encouraging model to focus on knowledge within local scope rather than memorizing contextual patterns. Evaluations show that COIN reduces Context Reliance by 45.2% and outperforms strong baselines by 23.6% in editing success rate, highlighting the vital role of mitigating Context Reliance for robust editing.
>
---
#### [new 026] Entropy in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于信息论与自然语言处理任务，旨在比较大语言模型与自然语言的词熵，分析LLM生成文本的信息量，以评估训练数据对模型的影响。**

- **链接: [https://arxiv.org/pdf/2602.20052v1](https://arxiv.org/pdf/2602.20052v1)**

> **作者:** Marco Scharringhausen
>
> **备注:** 7 pages, 2 figures, 3 tables
>
> **摘要:** In this study, the output of large language models (LLM) is considered an information source generating an unlimited sequence of symbols drawn from a finite alphabet. Given the probabilistic nature of modern LLMs, we assume a probabilistic model for these LLMs, following a constant random distribution and the source itself thus being stationary. We compare this source entropy (per word) to that of natural language (written or spoken) as represented by the Open American National Corpus (OANC). Our results indicate that the word entropy of such LLMs is lower than the word entropy of natural speech both in written or spoken form. The long-term goal of such studies is to formalize the intuitions of information and uncertainty in large language training to assess the impact of training an LLM from LLM generated training data. This refers to texts from the world wide web in particular.
>
---
#### [new 027] Retrieval Augmented Enhanced Dual Co-Attention Framework for Target Aware Multimodal Bengali Hateful Meme Detection
- **分类: cs.CL**

- **简介: 该论文属于多模态仇恨表情识别任务，针对孟加拉语低资源环境下仇恨表情检测难题，提出增强双注意力框架并引入检索增强方法，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2602.19212v1](https://arxiv.org/pdf/2602.19212v1)**

> **作者:** Raihan Tanvir; Md. Golam Rabiul Alam
>
> **摘要:** Hateful content on social media increasingly appears as multimodal memes that combine images and text to convey harmful narratives. In low-resource languages such as Bengali, automated detection remains challenging due to limited annotated data, class imbalance, and pervasive code-mixing. To address these issues, we augment the Bengali Hateful Memes (BHM) dataset with semantically aligned samples from the Multimodal Aggression Dataset in Bengali (MIMOSA), improving both class balance and semantic diversity. We propose the Enhanced Dual Co-attention Framework (xDORA), integrating vision encoders (CLIP, DINOv2) and multilingual text encoders (XGLM, XLM-R) via weighted attention pooling to learn robust cross-modal representations. Building on these embeddings, we develop a FAISS-based k-nearest neighbor classifier for non-parametric inference and introduce RAG-Fused DORA, which incorporates retrieval-driven contextual reasoning. We further evaluate LLaVA under zero-shot, few-shot, and retrieval-augmented prompting settings. Experiments on the extended dataset show that xDORA (CLIP + XLM-R) achieves macro-average F1-scores of 0.78 for hateful meme identification and 0.71 for target entity detection, while RAG-Fused DORA improves performance to 0.79 and 0.74, yielding gains over the DORA baseline. The FAISS-based classifier performs competitively and demonstrates robustness for rare classes through semantic similarity modeling. In contrast, LLaVA exhibits limited effectiveness in few-shot settings, with only modest improvements under retrieval augmentation, highlighting constraints of pretrained vision-language models for code-mixed Bengali content without fine-tuning. These findings demonstrate the effectiveness of supervised, retrieval-augmented, and non-parametric multimodal frameworks for addressing linguistic and cultural complexities in low-resource hate speech detection.
>
---
#### [new 028] PolyFrame at MWE-2026 AdMIRe 2: When Words Are Not Enough: Multimodal Idiom Disambiguation
- **分类: cs.CL**

- **简介: 该论文针对多模态习语消歧任务，解决习语在多语言环境下的语义理解问题。提出PolyFrame系统，通过轻量模块提升模型性能。**

- **链接: [https://arxiv.org/pdf/2602.18652v1](https://arxiv.org/pdf/2602.18652v1)**

> **作者:** Nina Hosseini-Kivanani
>
> **备注:** Accepted at AdMIRe 2 shared task (Advancing Multimodal Idiomaticity Representation) colocated with 22nd Workshop on Multiword Expressions (MWE 2026) @EACL2026
>
> **摘要:** Multimodal models struggle with idiomatic expressions due to their non-compositional meanings, a challenge amplified in multilingual settings. We introduced PolyFrame, our system for the MWE-2026 AdMIRe2 shared task on multimodal idiom disambiguation, featuring a unified pipeline for both image+text ranking (Subtask A) and text-only caption ranking (Subtask B). All model variants retain frozen CLIP-style vision--language encoders and the multilingual BGE M3 encoder, training only lightweight modules: a logistic regression and LLM-based sentence-type predictor, idiom synonym substitution, distractor-aware scoring, and Borda rank fusion. Starting from a CLIP baseline (26.7% Top-1 on English dev, 6.7% on English test), adding idiom-aware paraphrasing and explicit sentence-type classification increased performance to 60.0% Top-1 on English and 60.0% Top-1 (0.822 NDCG@5) in zero-shot transfer to Portuguese. On the multilingual blind test, our systems achieved average Top-1/NDCG scores of 0.35/0.73 for Subtask A and 0.32/0.71 for Subtask B across 15 languages. Ablation results highlight idiom-aware rewriting as the main contributor to performance, while sentence-type prediction and multimodal fusion enhance robustness. These findings suggest that effective idiom disambiguation is feasible without fine-tuning large multimodal encoders.
>
---
#### [new 029] KNIGHT: Knowledge Graph-Driven Multiple-Choice Question Generation with Adaptive Hardness Calibration
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文提出KNIGHT框架，用于生成高质量多选题数据集。解决评估大型语言模型缺乏高效数据生成工具的问题，通过知识图谱实现难度可控的题目生成。**

- **链接: [https://arxiv.org/pdf/2602.20135v1](https://arxiv.org/pdf/2602.20135v1)**

> **作者:** Mohammad Amanlou; Erfan Shafiee Moghaddam; Yasaman Amou Jafari; Mahdi Noori; Farhan Farsi; Behnam Bahrak
>
> **备注:** Accepted at the Third Conference on Parsimony and Learning (CPAL 2026). 36 pages, 12 figures. (Equal contribution: Yasaman Amou Jafari and Mahdi Noori.)
>
> **摘要:** With the rise of large language models (LLMs), they have become instrumental in applications such as Retrieval-Augmented Generation (RAG). Yet evaluating these systems remains bottlenecked by the time and cost of building specialized assessment datasets. We introduce KNIGHT, an LLM-based, knowledge-graph-driven framework for generating multiple-choice question (MCQ) datasets from external sources. KNIGHT constructs a topic-specific knowledge graph, a structured and parsimonious summary of entities and relations, that can be reused to generate instructor-controlled difficulty levels, including multi-hop questions, without repeatedly re-feeding the full source text. This knowledge graph acts as a compressed, reusable state, making question generation a cheap read over the graph. We instantiate KNIGHT on Wikipedia/Wikidata while keeping the framework domain- and ontology-agnostic. As a case study, KNIGHT produces six MCQ datasets in History, Biology, and Mathematics. We evaluate quality on five criteria: fluency, unambiguity (single correct answer), topic relevance, option uniqueness, and answerability given the provided sources (as a proxy for hallucination). Results show that KNIGHT enables token- and cost-efficient generation from a reusable graph representation, achieves high quality across these criteria, and yields model rankings aligned with MMLU-style benchmarks, while supporting topic-specific and difficulty-controlled evaluation.
>
---
#### [new 030] Rethinking Retrieval-Augmented Generation as a Cooperative Decision-Making Problem
- **分类: cs.CL; cs.AI**

- **简介: 该论文将RAG视为协作决策问题，提出CoRAG框架，解决生成质量依赖重排序的问题，通过协同优化提升生成稳定性。**

- **链接: [https://arxiv.org/pdf/2602.18734v1](https://arxiv.org/pdf/2602.18734v1)**

> **作者:** Lichang Song; Ting Long; Yi Chang
>
> **摘要:** Retrieval-Augmented Generation (RAG) has demonstrated strong effectiveness in knowledge-intensive tasks by grounding language generation in external evidence. Despite its success, many existing RAG systems are built based on a ranking-centric, asymmetric dependency paradigm, where the generation quality of the generator is highly dependent on reranking results of the reranker. To overcome this limitation, we reformulate RAG as a cooperative multi-agent decision-making problem and propose Cooperative Retrieval-Augmented Generation (CoRAG), a framework in which the reranker and the generator act as peer decision-makers rather than being connected through an asymmetric dependency pipeline. By jointly optimizing their behaviors toward a shared task objective, the reranker and generator are encouraged to cooperate, ensuring that document reranking and generation work in concert to improve the final response. Experimental results demonstrate good generalization and improved generation stability of CoRAG, even when the model is trained on only around 10K PopQA samples. Our model released in https://anonymous.4open.science/r/CoRAG-D63F
>
---
#### [new 031] Unlocking Multimodal Document Intelligence: From Current Triumphs to Future Frontiers of Visual Document Retrieval
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于视觉文档检索任务，旨在解决多模态信息与精准信息获取之间的差距。通过综述MLLM时代的VDR方法，分析其发展与挑战，提出未来研究方向。**

- **链接: [https://arxiv.org/pdf/2602.19961v1](https://arxiv.org/pdf/2602.19961v1)**

> **作者:** Yibo Yan; Jiahao Huo; Guanbo Feng; Mingdong Ou; Yi Cao; Xin Zou; Shuliang Liu; Yuanhuiyi Lyu; Yu Huang; Jungang Li; Kening Zheng; Xu Zheng; Philip S. Yu; James Kwok; Xuming Hu
>
> **备注:** Under review
>
> **摘要:** With the rapid proliferation of multimodal information, Visual Document Retrieval (VDR) has emerged as a critical frontier in bridging the gap between unstructured visually rich data and precise information acquisition. Unlike traditional natural image retrieval, visual documents exhibit unique characteristics defined by dense textual content, intricate layouts, and fine-grained semantic dependencies. This paper presents the first comprehensive survey of the VDR landscape, specifically through the lens of the Multimodal Large Language Model (MLLM) era. We begin by examining the benchmark landscape, and subsequently dive into the methodological evolution, categorizing approaches into three primary aspects: multimodal embedding models, multimodal reranker models, and the integration of Retrieval-Augmented Generation (RAG) and Agentic systems for complex document intelligence. Finally, we identify persistent challenges and outline promising future directions, aiming to provide a clear roadmap for future multimodal document intelligence.
>
---
#### [new 032] To Reason or Not to: Selective Chain-of-Thought in Medical Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医学问答任务，旨在提升效率并减少不必要的推理。提出Selective CoT方法，在需要时生成推理过程，降低计算成本且保持准确。**

- **链接: [https://arxiv.org/pdf/2602.20130v1](https://arxiv.org/pdf/2602.20130v1)**

> **作者:** Zaifu Zhan; Min Zeng; Shuang Zhou; Yiran Song; Xiaoyi Chen; Yu Hou; Yifan Wu; Yang Ruan; Rui Zhang
>
> **摘要:** Objective: To improve the efficiency of medical question answering (MedQA) with large language models (LLMs) by avoiding unnecessary reasoning while maintaining accuracy. Methods: We propose Selective Chain-of-Thought (Selective CoT), an inference-time strategy that first predicts whether a question requires reasoning and generates a rationale only when needed. Two open-source LLMs (Llama-3.1-8B and Qwen-2.5-7B) were evaluated on four biomedical QA benchmarks-HeadQA, MedQA-USMLE, MedMCQA, and PubMedQA. Metrics included accuracy, total generated tokens, and inference time. Results: Selective CoT reduced inference time by 13-45% and token usage by 8-47% with minimal accuracy loss ($\leq$4\%). In some model-task pairs, it achieved both higher accuracy and greater efficiency than standard CoT. Compared with fixed-length CoT, Selective CoT reached similar or superior accuracy at substantially lower computational cost. Discussion: Selective CoT dynamically balances reasoning depth and efficiency by invoking explicit reasoning only when beneficial, reducing redundancy on recall-type questions while preserving interpretability. Conclusion: Selective CoT provides a simple, model-agnostic, and cost-effective approach for medical QA, aligning reasoning effort with question complexity to enhance real-world deployability of LLM-based clinical systems.
>
---
#### [new 033] Yor-Sarc: A gold-standard dataset for sarcasm detection in a low-resource African language
- **分类: cs.CL**

- **简介: 该论文属于讽刺检测任务，旨在解决低资源非洲语言中缺乏标注数据的问题。研究构建了首个Yorùbá语的高质量讽刺数据集Yor-Sarc，并设计了文化相关的标注协议。**

- **链接: [https://arxiv.org/pdf/2602.18964v1](https://arxiv.org/pdf/2602.18964v1)**

> **作者:** Toheeb Aduramomi Jimoh; Tabea De Wille; Nikola S. Nikolov
>
> **摘要:** Sarcasm detection poses a fundamental challenge in computational semantics, requiring models to resolve disparities between literal and intended meaning. The challenge is amplified in low-resource languages where annotated datasets are scarce or nonexistent. We present \textbf{Yor-Sarc}, the first gold-standard dataset for sarcasm detection in Yorùbá, a tonal Niger-Congo language spoken by over $50$ million people. The dataset comprises 436 instances annotated by three native speakers from diverse dialectal backgrounds using an annotation protocol specifically designed for Yorùbá sarcasm by taking culture into account. This protocol incorporates context-sensitive interpretation and community-informed guidelines and is accompanied by a comprehensive analysis of inter-annotator agreement to support replication in other African languages. Substantial to almost perfect agreement was achieved (Fleiss' $κ= 0.7660$; pairwise Cohen's $κ= 0.6732$--$0.8743$), with $83.3\%$ unanimous consensus. One annotator pair achieved almost perfect agreement ($κ= 0.8743$; $93.8\%$ raw agreement), exceeding a number of reported benchmarks for English sarcasm research works. The remaining $16.7\%$ majority-agreement cases are preserved as soft labels for uncertainty-aware modelling. Yor-Sarc\footnote{https://github.com/toheebadura/yor-sarc} is expected to facilitate research on semantic interpretation and culturally informed NLP for low-resource African languages.
>
---
#### [new 034] Anatomy of Unlearning: The Dual Impact of Fact Salience and Model Fine-Tuning
- **分类: cs.CL**

- **简介: 该论文研究机器遗忘任务，解决模型遗忘不准确问题。提出DUAL基准，分析预训练与微调模型在遗忘中的差异，验证微调提升遗忘效果。**

- **链接: [https://arxiv.org/pdf/2602.19612v1](https://arxiv.org/pdf/2602.19612v1)**

> **作者:** Borisiuk Anna; Andrey Savchenko; Alexander Panchecko; Elena Tutubalina
>
> **摘要:** Machine Unlearning (MU) enables Large Language Models (LLMs) to remove unsafe or outdated information. However, existing work assumes that all facts are equally forgettable and largely ignores whether the forgotten knowledge originates from pretraining or supervised fine-tuning (SFT). In this paper, we introduce DUAL (Dual Unlearning Evaluation across Training Stages), a benchmark of 28.6k Wikidata-derived triplets annotated with fact popularity using Wikipedia link counts and LLM-based salience scores. Our experiments show that pretrained and SFT models respond differently to unlearning. An SFT step on the forget data yields smoother forgetting, more stable tuning, and 10-50% higher retention, while direct unlearning on pretrained models remains unstable and prone to relearning or catastrophic forgetting.
>
---
#### [new 035] How Retrieved Context Shapes Internal Representations in RAG
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，研究RAG中检索上下文如何影响模型内部表示。通过分析不同文档对隐藏状态的影响，揭示其与生成行为的关系，为RAG系统设计提供依据。**

- **链接: [https://arxiv.org/pdf/2602.20091v1](https://arxiv.org/pdf/2602.20091v1)**

> **作者:** Samuel Yeh; Sharon Li
>
> **摘要:** Retrieval-augmented generation (RAG) enhances large language models (LLMs) by conditioning generation on retrieved external documents, but the effect of retrieved context is often non-trivial. In realistic retrieval settings, the retrieved document set often contains a mixture of documents that vary in relevance and usefulness. While prior work has largely examined these phenomena through output behavior, little is known about how retrieved context shapes the internal representations that mediate information integration in RAG. In this work, we study RAG through the lens of latent representations. We systematically analyze how different types of retrieved documents affect the hidden states of LLMs, and how these internal representation shifts relate to downstream generation behavior. Across four question-answering datasets and three LLMs, we analyze internal representations under controlled single- and multi-document settings. Our results reveal how context relevancy and layer-wise processing influence internal representations, providing explanations on LLMs output behaviors and insights for RAG system design.
>
---
#### [new 036] SHIELD: Semantic Heterogeneity Integrated Embedding for Latent Discovery in Clinical Trial Safety Signals
- **分类: cs.CL**

- **简介: 该论文提出SHIELD方法，用于临床试验中安全信号的自动检测。通过结合统计分析与语义聚类，识别相关不良事件，生成可解释的安全特征图谱。任务是提升临床试验安全性评估与因果解释。**

- **链接: [https://arxiv.org/pdf/2602.19855v1](https://arxiv.org/pdf/2602.19855v1)**

> **作者:** Francois Vandenhende; Anna Georgiou; Theodoros Psaras; Ellie Karekla
>
> **备注:** 3 figures, 1 table
>
> **摘要:** We present SHIELD, a novel methodology for automated and integrated safety signal detection in clinical trials. SHIELD combines disproportionality analysis with semantic clustering of adverse event (AE) terms applied to MedDRA term embeddings. For each AE, the pipeline computes an information-theoretic disproportionality measure (Information Component) with effect size derived via empirical Bayesian shrinkage. A utility matrix is constructed by weighting semantic term-term similarities by signal magnitude, followed by spectral embedding and clustering to identify groups of related AEs. Resulting clusters are annotated with syndrome-level summary labels using large language models, yielding a coherent, data-driven representation of treatment-associated safety profiles in the form of a network graph and hierarchical tree. We implement the SHIELD framework in the context of a single-arm incidence summary, to compare two treatment arms or for the detection of any treatment effect in a multi-arm trial. We illustrate its ability to recover known safety signals and generate interpretable, cluster-based summaries in a real clinical trial example. This work bridges statistical signal detection with modern natural language processing to enhance safety assessment and causal interpretation in clinical trials.
>
---
#### [new 037] DeepInnovator: Triggering the Innovative Capabilities of LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于科学发现任务，旨在提升LLMs的创新能力。通过构建数据管道和引入新训练范式，解决传统方法依赖提示工程的问题，提升模型生成新颖研究想法的能力。**

- **链接: [https://arxiv.org/pdf/2602.18920v1](https://arxiv.org/pdf/2602.18920v1)**

> **作者:** Tianyu Fan; Fengji Zhang; Yuxiang Zheng; Bei Chen; Xinyao Niu; Chengen Huang; Junyang Lin; Chao Huang
>
> **摘要:** The application of Large Language Models (LLMs) in accelerating scientific discovery has garnered increasing attention, with a key focus on constructing research agents endowed with innovative capability, i.e., the ability to autonomously generate novel and significant research ideas. Existing approaches predominantly rely on sophisticated prompt engineering and lack a systematic training paradigm. To address this, we propose DeepInnovator, a training framework designed to trigger the innovative capability of LLMs. Our approach comprises two core components. (1) ``Standing on the shoulders of giants''. We construct an automated data extraction pipeline to extract and organize structured research knowledge from a vast corpus of unlabeled scientific literature. (2) ``Conjectures and refutations''. We introduce a ``Next Idea Prediction'' training paradigm, which models the generation of research ideas as an iterative process of continuously predicting, evaluating, and refining plausible and novel next idea. Both automatic and expert evaluations demonstrate that our DeepInnovator-14B significantly outperforms untrained baselines, achieving win rates of 80.53\%-93.81\%, and attains performance comparable to that of current leading LLMs. This work provides a scalable training pathway toward building research agents with genuine, originative innovative capability, and will open-source the dataset to foster community advancement. Source code and data are available at: https://github.com/HKUDS/DeepInnovator.
>
---
#### [new 038] ReHear: Iterative Pseudo-Label Refinement for Semi-Supervised Speech Recognition via Audio Large Language Models
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决半监督学习中的伪标签偏差和错误累积问题。通过引入音频感知的大语言模型，迭代优化伪标签，提升识别效果。**

- **链接: [https://arxiv.org/pdf/2602.18721v1](https://arxiv.org/pdf/2602.18721v1)**

> **作者:** Zefang Liu; Chenyang Zhu; Sangwoo Cho; Shi-Xiong Zhang
>
> **摘要:** Semi-supervised learning in automatic speech recognition (ASR) typically relies on pseudo-labeling, which often suffers from confirmation bias and error accumulation due to noisy supervision. To address this limitation, we propose ReHear, a framework for iterative pseudo-label refinement that integrates an instruction-tuned, audio-aware large language model (LLM) into the self-training loop. Unlike conventional text-based correctors, our approach conditions the LLM on both the ASR hypothesis and the source audio, allowing it to recover phonetically accurate transcripts even from severe recognition errors. These refined pseudo-labels serve as high-fidelity targets for fine-tuning the ASR model in an iterative cycle. Experimental results across diverse benchmarks demonstrate that ReHear effectively mitigates error propagation, consistently outperforming both supervised and pseudo-labeling baselines.
>
---
#### [new 039] Luna-2: Scalable Single-Token Evaluation with Small Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出Luna-2，解决实时评估中成本高、延迟大的问题，通过小语言模型实现高效、准确的多任务评估。**

- **链接: [https://arxiv.org/pdf/2602.18583v1](https://arxiv.org/pdf/2602.18583v1)**

> **作者:** Vatsal Goel; Rishon Dsouza; Nikhil Ega; Amey Ramesh Rambatla; Rob Friel; Shuai Shao; Yash Sheth
>
> **摘要:** Real-time guardrails require evaluation that is accurate, cheap, and fast - yet today's default, LLM-as-a-judge (LLMAJ), is slow, expensive, and operationally non-deterministic due to multi-token generation. We present Luna-2, a novel architecture that leverages decoder-only small language models (SLMs) into a deterministic evaluation model to reliably compute complex task-specific LLMAJ metrics (e.g. toxicity, hallucination, tool selection quality, etc.) at an accuracy at par or higher than LLMAJ using frontier LLMs while drastically reducing the cost and latency of computation. Each metric is implemented as a lightweight LoRA/PEFT head on top of a shared SLM backbone, enabling hundreds of specialized metrics to run concurrently on a single GPU, deployable locally next to AI systems in a privacy-preserving and latency optimizing manner. Across content safety and hallucination benchmarks, Luna-2 matches the accuracy of state-of-the-art LLM-based evaluators while reducing inference cost by over 80x and latency by over 20x. In this paper, we outline the model architecture, training methodology and report real-world empirical results on accuracy, latency, and throughput results. In production, Luna-2 is protecting 100M+ AI sessions and processing over 100B tokens per month for our customers with eval cost savings of over $30M annually.
>
---
#### [new 040] DP-RFT: Learning to Generate Synthetic Text via Differentially Private Reinforcement Fine-Tuning
- **分类: cs.CL**

- **简介: 该论文属于隐私保护下的文本生成任务，旨在解决不接触私有数据仍能生成高质量合成文本的问题。提出DP-RFT方法，通过差分隐私强化学习提升生成文本的领域一致性与实用性。**

- **链接: [https://arxiv.org/pdf/2602.18633v1](https://arxiv.org/pdf/2602.18633v1)**

> **作者:** Fangyuan Xu; Sihao Chen; Zinan Lin; Taiwei Shi; Sydney Graham; Pei Zhou; Mengting Wan; Alex Stein; Virginia Estellers; Charles Chen; Morris Sharp; Richard Speyer; Tadas Baltrusaitis; Jennifer Neville; Eunsol Choi; Longqi Yang
>
> **摘要:** Differentially private (DP) synthetic data generation plays a pivotal role in developing large language models (LLMs) on private data, where data owners cannot provide eyes-on access to individual examples. Generating DP synthetic data typically involves a difficult trade-off. On one hand, DP finetuning methods train an LLM as a synthetic data generator with formal privacy guarantees, yet it still requires the raw content of private examples for model training. However, methods that avoid direct exposure to private data are bounded by an off-the-shelf, un-finetuned model, whose outputs often lack domain fidelity. Can we train an LLM to generate high-quality synthetic text without eyes-on access to individual private examples? In this work, we introduce Differentially Private Reinforcement Fine-Tuning (DP-RFT), an online reinforcement learning algorithm for synthetic data generation with LLMs. DP-RFT leverages DP-protected nearest-neighbor votes from an eyes-off private corpus as a reward signal for on-policy synthetic samples generated by an LLM. The LLM iteratively learns to generate synthetic data to maximize the expected DP votes through Proximal Policy Optimization (PPO). We evaluate DP-RFT for long-form and domain-specific synthetic data generation, such as news articles, meeting transcripts, and medical article abstracts. Our experiments show that DP-RFT closes the gap between private evolution and DP finetuning methods in terms of the fidelity and downstream utility of the generated synthetic data, while respecting the private data boundary.
>
---
#### [new 041] Why Agent Caching Fails and How to Fix It: Structured Intent Canonicalization with Few-Shot Learning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究如何提升AI代理的缓存效果，解决重复调用大模型导致的成本问题。通过结构化意图分解和少量样本学习，提升缓存准确性与效率。**

- **链接: [https://arxiv.org/pdf/2602.18922v1](https://arxiv.org/pdf/2602.18922v1)**

> **作者:** Abhinaba Basu
>
> **备注:** 28 pages, 15 figures, 8 tables, 5 appendices
>
> **摘要:** Personal AI agents incur substantial cost via repeated LLM calls. We show existing caching methods fail: GPTCache achieves 37.9% accuracy on real benchmarks; APC achieves 0-12%. The root cause is optimizing for the wrong property -- cache effectiveness requires key consistency and precision, not classification accuracy. We observe cache-key evaluation reduces to clustering evaluation and apply V-measure decomposition to separate these on n=8,682 points across MASSIVE, BANKING77, CLINC150, and NyayaBench v2, our new 8,514-entry multilingual agentic dataset (528 intents, 20 W5H2 classes, 63 languages). We introduce W5H2, a structured intent decomposition framework. Using SetFit with 8 examples per class, W5H2 achieves 91.1%+/-1.7% on MASSIVE in ~2ms -- vs 37.9% for GPTCache and 68.8% for a 20B-parameter LLM at 3,447ms. On NyayaBench v2 (20 classes), SetFit achieves 55.3%, with cross-lingual transfer across 30 languages. Our five-tier cascade handles 85% of interactions locally, projecting 97.5% cost reduction. We provide risk-controlled selective prediction guarantees via RCPS with nine bound families.
>
---
#### [new 042] Do LLMs and VLMs Share Neurons for Inference? Evidence and Mechanisms of Cross-Modal Transfer
- **分类: cs.CL**

- **简介: 该论文研究LLMs与LVLMs在推理过程中是否共享神经元，旨在提升LVLM的多模态推理能力。通过分析发现两者存在大量共享神经元，并提出SNRF框架实现高效参数迁移。**

- **链接: [https://arxiv.org/pdf/2602.19058v1](https://arxiv.org/pdf/2602.19058v1)**

> **作者:** Chenhang Cui; An Zhang; Yuxin Chen; Gelei Deng; Jingnan Zheng; Zhenkai Liang; Xiang Wang; Tat-Seng Chua
>
> **摘要:** Large vision-language models (LVLMs) have rapidly advanced across various domains, yet they still lag behind strong text-only large language models (LLMs) on tasks that require multi-step inference and compositional decision-making. Motivated by their shared transformer architectures, we investigate whether the two model families rely on common internal computation for such inference. At the neuron level, we uncover a surprisingly large overlap: more than half of the top-activated units during multi-step inference are shared between representative LLMs and LVLMs, revealing a modality-invariant inference subspace. Through causal probing via activation amplification, we further show that these shared neurons encode consistent and interpretable concept-level effects, demonstrating their functional contribution to inference. Building on this insight, we propose Shared Neuron Low-Rank Fusion (SNRF), a parameter-efficient framework that transfers mature inference circuitry from LLMs to LVLMs. SNRF profiles cross-model activations to identify shared neurons, computes a low-rank approximation of inter-model weight differences, and injects these updates selectively within the shared-neuron subspace. This mechanism strengthens multimodal inference performance with minimal parameter changes and requires no large-scale multimodal fine-tuning. Across diverse mathematics and perception benchmarks, SNRF consistently enhances LVLM inference performance while preserving perceptual capabilities. Our results demonstrate that shared neurons form an interpretable bridge between LLMs and LVLMs, enabling low-cost transfer of inference ability into multimodal models. Our code is available at [https://github.com/chenhangcuisg-code/Do-LLMs-VLMs-Share-Neurons](https://github.com/chenhangcuisg-code/Do-LLMs-VLMs-Share-Neurons).
>
---
#### [new 043] How Do LLMs Encode Scientific Quality? An Empirical Study Using Monosemantic Features from Sparse Autoencoders
- **分类: cs.CL; cs.AI; cs.DL**

- **简介: 该论文属于自然语言处理任务，旨在探究LLMs如何编码科学质量。通过稀疏自编码器提取单义特征，分析其对引用量、期刊影响力等质量指标的预测能力，揭示LLMs内部表征科学质量的机制。**

- **链接: [https://arxiv.org/pdf/2602.19115v1](https://arxiv.org/pdf/2602.19115v1)**

> **作者:** Michael McCoubrey; Angelo Salatino; Francesco Osborne; Enrico Motta
>
> **备注:** Presented at SESAME 2025: Smarter Extraction of ScholArly MEtadata using Knowledge Graphs and Language Models, @ JCDL 2025
>
> **摘要:** In recent years, there has been a growing use of generative AI, and large language models (LLMs) in particular, to support both the assessment and generation of scientific work. Although some studies have shown that LLMs can, to a certain extent, evaluate research according to perceived quality, our understanding of the internal mechanisms that enable this capability remains limited. This paper presents the first study that investigates how LLMs encode the concept of scientific quality through relevant monosemantic features extracted using sparse autoencoders. We derive such features under different experimental settings and assess their ability to serve as predictors across three tasks related to research quality: predicting citation count, journal SJR, and journal h-index. The results indicate that LLMs encode features associated with multiple dimensions of scientific quality. In particular, we identify four recurring types of features that capture key aspects of how research quality is represented: 1) features reflecting research methodologies; 2) features related to publication type, with literature reviews typically exhibiting higher impact; 3) features associated with high-impact research fields and technologies; and 4) features corresponding to specific scientific jargons. These findings represent an important step toward understanding how LLMs encapsulate concepts related to research quality.
>
---
#### [new 044] SAMAS: A Spectrum-Guided Multi-Agent System for Achieving Style Fidelity in Literary Translation
- **分类: cs.CL**

- **简介: 该论文属于文学翻译任务，旨在解决机器翻译中风格失真问题。提出SAMAS系统，通过分析文本风格特征，动态组合翻译代理，提升翻译风格一致性。**

- **链接: [https://arxiv.org/pdf/2602.19840v1](https://arxiv.org/pdf/2602.19840v1)**

> **作者:** Jingzhuo Wu; Jiajun Zhang; Keyan Jin; Dehua Ma; Junbo Wang
>
> **摘要:** Modern large language models (LLMs) excel at generating fluent and faithful translations. However, they struggle to preserve an author's unique literary style, often producing semantically correct but generic outputs. This limitation stems from the inability of current single-model and static multi-agent systems to perceive and adapt to stylistic variations. To address this, we introduce the Style-Adaptive Multi-Agent System (SAMAS), a novel framework that treats style preservation as a signal processing task. Specifically, our method quantifies literary style into a Stylistic Feature Spectrum (SFS) using the wavelet packet transform. This SFS serves as a control signal to dynamically assemble a tailored workflow of specialized translation agents based on the source text's structural patterns. Extensive experiments on translation benchmarks show that SAMAS achieves competitive semantic accuracy against strong baselines, primarily by leveraging its statistically significant advantage in style fidelity.
>
---
#### [new 045] Astra: Activation-Space Tail-Eigenvector Low-Rank Adaptation of Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出Astra方法，解决LoRA等PEFT方法在激活空间中尾部特征向量利用不足的问题，通过构建任务自适应低秩适配器提升模型微调效果。属于参数高效微调任务。**

- **链接: [https://arxiv.org/pdf/2602.19111v1](https://arxiv.org/pdf/2602.19111v1)**

> **作者:** Kainan Liu; Yong Zhang; Ning Cheng; Yun Zhu; Yanmeng Wang; Shaojun Wang; Jing Xiao
>
> **备注:** 22 pages, 10 figures
>
> **摘要:** Parameter-Efficient Fine-Tuning (PEFT) methods, especially LoRA, are widely used for adapting pre-trained models to downstream tasks due to their computational and storage efficiency. However, in the context of LoRA and its variants, the potential of activation subspaces corresponding to tail eigenvectors remains substantially under-exploited, which may lead to suboptimal fine-tuning performance. In this work, we propose Astra (Activation-Space Tail-Eigenvector Low-Rank Adaptation), a novel PEFT method that leverages the tail eigenvectors of the model output activations-estimated from a small task-specific calibration set-to construct task-adaptive low-rank adapters. By constraining updates to the subspace spanned by these tail eigenvectors, Astra achieves faster convergence and improved downstream performance with a significantly reduced parameter budget. Extensive experiments across natural language understanding (NLU) and natural language generation (NLG) tasks demonstrate that Astra consistently outperforms existing PEFT baselines across 16 benchmarks and even surpasses full fine-tuning (FFT) in certain scenarios.
>
---
#### [new 046] BabyLM Turns 4: Call for Papers for the 2026 BabyLM Workshop
- **分类: cs.CL**

- **简介: 该论文是会议征稿通知，属于会议组织任务。旨在征集语言模型与认知建模相关研究，解决跨领域融合问题，开展数据高效预训练和多语言挑战。**

- **链接: [https://arxiv.org/pdf/2602.20092v1](https://arxiv.org/pdf/2602.20092v1)**

> **作者:** Leshem Choshen; Ryan Cotterell; Mustafa Omer Gul; Jaap Jumelet; Tal Linzen; Aaron Mueller; Suchir Salhan; Raj Sanjay Shah; Alex Warstadt; Ethan Gotlieb Wilcox
>
> **备注:** 8 pages, 1 table. arXiv admin note: substantial text overlap with arXiv:2502.10645
>
> **摘要:** BabyLM aims to dissolve the boundaries between cognitive modeling and language modeling. We call for both workshop papers and for researchers to join the 4th BabyLM competition. As in previous years, we call for participants in the data-efficient pretraining challenge in the general track. This year, we also offer a new track: Multilingual. We also call for papers outside the competition in any relevant areas. These include training efficiency, cognitively plausible research, weak model evaluation, and more.
>
---
#### [new 047] Next Reply Prediction X Dataset: Linguistic Discrepancies in Naively Generated Content
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM生成内容与人类语言的差异问题。通过构建真实X数据集，分析并评估LLM生成文本的语义和风格偏差。**

- **链接: [https://arxiv.org/pdf/2602.19177v1](https://arxiv.org/pdf/2602.19177v1)**

> **作者:** Simon Münker; Nils Schwager; Kai Kugler; Michael Heseltine; Achim Rettinger
>
> **备注:** 8 pages (12 including references), 2 figures and 2 tables
>
> **摘要:** The increasing use of Large Language Models (LLMs) as proxies for human participants in social science research presents a promising, yet methodologically risky, paradigm shift. While LLMs offer scalability and cost-efficiency, their "naive" application, where they are prompted to generate content without explicit behavioral constraints, introduces significant linguistic discrepancies that challenge the validity of research findings. This paper addresses these limitations by introducing a novel, history-conditioned reply prediction task on authentic X (formerly Twitter) data, to create a dataset designed to evaluate the linguistic output of LLMs against human-generated content. We analyze these discrepancies using stylistic and content-based metrics, providing a quantitative framework for researchers to assess the quality and authenticity of synthetic data. Our findings highlight the need for more sophisticated prompting techniques and specialized datasets to ensure that LLM-generated content accurately reflects the complex linguistic patterns of human communication, thereby improving the validity of computational social science studies.
>
---
#### [new 048] INSURE-Dial: A Phase-Aware Conversational Dataset \& Benchmark for Compliance Verification and Phase Detection
- **分类: cs.CL**

- **简介: 该论文提出INSURE-Dial数据集与基准，用于合规性验证和对话阶段检测。解决医疗电话任务中人工处理效率低的问题，通过标注对话阶段和合规性标签，支持自动审计系统开发。**

- **链接: [https://arxiv.org/pdf/2602.18448v1](https://arxiv.org/pdf/2602.18448v1)**

> **作者:** Shubham Kulkarni; Alexander Lyzhov; Preetam Joshi; Shiva Chaitanya
>
> **备注:** Accepted to the 19th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2026)
>
> **摘要:** Administrative phone tasks drain roughly 1 trillion USD annually from U.S. healthcare, with over 500 million insurance-benefit verification calls manually handled in 2024. We introduce INSURE-Dial, to our knowledge the first public benchmark for developing and assessing compliance-aware voice agents for phase-aware call auditing with span-based compliance verification. The corpus includes 50 de-identified, AI-initiated calls with live insurance representatives (mean 71 turns/call) and 1,000 synthetically generated calls that mirror the same workflow. All calls are annotated with a phase-structured JSON schema covering IVR navigation, patient identification, coverage status, medication checks (up to two drugs), and agent identification (CRN), and each phase is labeled for Information and Procedural compliance under explicit ask/answer logic. We define two novel evaluation tasks: (1) Phase Boundary Detection (span segmentation under phase-specific acceptance rules) and (2) Compliance Verification (IC/PC decisions given fixed spans). Per-phase scores are strong across small, low-latency baselines, but end-to-end reliability is constrained by span-boundary errors. On real calls, full-call exact segmentation is low, showing a gap between conversational fluency and audit-grade evidence.
>
---
#### [new 049] gencat: Generative computerized adaptive testing
- **分类: cs.CL**

- **简介: 该论文属于自适应测试任务，旨在解决传统CAT无法有效利用开放性问题文本信息的问题。提出GENCAT框架，利用大语言模型进行知识估计和题目选择，提升测试效果。**

- **链接: [https://arxiv.org/pdf/2602.20020v1](https://arxiv.org/pdf/2602.20020v1)**

> **作者:** Wanyong Feng; Andrew Lan
>
> **备注:** 19 pages, 2 figures
>
> **摘要:** Existing computerized Adaptive Testing (CAT) frameworks are typically built on predicting the correctness of a student response to a question. Although effective, this approach fails to leverage textual information in questions and responses, especially for open-ended questions. In this work, we propose GENCAT (\textbf{GEN}erative \textbf{CAT}), a novel CAT framework that leverages Large Language Models for knowledge estimate and question selection. First, we develop a Generative Item Response Theory (GIRT) model that enables us to estimate student knowledge from their open-ended responses and predict responses to unseen questions. We train the model in a two-step process, first via Supervised Fine-Tuning and then via preference optimization for knowledge-response alignment. Second, we introduce three question selection algorithms that leverage the generative capabilities of the GIRT model, based on the uncertainty, linguistic diversity, and information of sampled student responses. Third, we conduct experiments on two real-world programming datasets and demonstrate that GENCAT outperforms existing CAT baselines, achieving an AUC improvement of up to 4.32\% in the key early testing stages.
>
---
#### [new 050] Janus-Q: End-to-End Event-Driven Trading via Hierarchical-Gated Reward Modeling
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出Janus-Q，解决金融交易中事件驱动信号提取问题，通过构建事件数据集并结合强化学习优化交易决策。**

- **链接: [https://arxiv.org/pdf/2602.19919v1](https://arxiv.org/pdf/2602.19919v1)**

> **作者:** Xiang Li; Zikai Wei; Yiyan Qi; Wanyun Zhou; Xiang Liu; Penglei Sun; Yongqi Zhang; Xiaowen Chu
>
> **摘要:** Financial market movements are often driven by discrete financial events conveyed through news, whose impacts are heterogeneous, abrupt, and difficult to capture under purely numerical prediction objectives. These limitations have motivated growing interest in using textual information as the primary source of trading signals in learning-based systems. Two key challenges hinder existing approaches: (1) the absence of large-scale, event-centric datasets that jointly model news semantics and statistically grounded market reactions, and (2) the misalignment between language model reasoning and financially valid trading behavior under dynamic market conditions. To address these challenges, we propose Janus-Q, an end-to-end event-driven trading framework that elevates financial news events from auxiliary signals to primary decision units. Janus-Q unifies event-centric data construction and model optimization under a two-stage paradigm. Stage I focuses on event-centric data construction, building a large-scale financial news event dataset comprising 62,400 articles annotated with 10 fine-grained event types, associated stocks, sentiment labels, and event-driven cumulative abnormal return (CAR). Stage II performs decision-oriented fine-tuning, combining supervised learning with reinforcement learning guided by a Hierarchical Gated Reward Model (HGRM), which explicitly captures trade-offs among multiple trading objectives. Extensive experiments demonstrate that Janus-Q achieves more consistent, interpretable, and profitable trading decisions than market indices and LLM baselines, improving the Sharpe Ratio by up to 102.0% while increasing direction accuracy by over 17.5% compared to the strongest competing strategies.
>
---
#### [new 051] IAPO: Information-Aware Policy Optimization for Token-Efficient Reasoning
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语言模型优化任务，解决推理过程中的token效率问题。提出IAPO框架，通过信息理论方法提升推理准确性并减少长度。**

- **链接: [https://arxiv.org/pdf/2602.19049v1](https://arxiv.org/pdf/2602.19049v1)**

> **作者:** Yinhan He; Yaochen Zhu; Mingjia Shi; Wendy Zheng; Lin Su; Xiaoqing Wang; Qi Guo; Jundong Li
>
> **摘要:** Large language models increasingly rely on long chains of thought to improve accuracy, yet such gains come with substantial inference-time costs. We revisit token-efficient post-training and argue that existing sequence-level reward-shaping methods offer limited control over how reasoning effort is allocated across tokens. To bridge the gap, we propose IAPO, an information-theoretic post-training framework that assigns token-wise advantages based on each token's conditional mutual information (MI) with the final answer. This yields an explicit, principled mechanism for identifying informative reasoning steps and suppressing low-utility exploration. We provide a theoretical analysis showing that our IAPO can induce monotonic reductions in reasoning verbosity without harming correctness. Empirically, IAPO consistently improves reasoning accuracy while reducing reasoning length by up to 36%, outperforming existing token-efficient RL methods across various reasoning datasets. Extensive empirical evaluations demonstrate that information-aware advantage shaping is a powerful and general direction for token-efficient post-training. The code is available at https://github.com/YinhanHe123/IAPO.
>
---
#### [new 052] PerSoMed: A Large-Scale Balanced Dataset for Persian Social Media Text Classification
- **分类: cs.CL; cs.IR; cs.SI**

- **简介: 该论文提出PerSoMed数据集，解决波斯语社交媒体文本分类资源不足的问题。通过平衡数据和模型评估，提升分类性能。**

- **链接: [https://arxiv.org/pdf/2602.19333v1](https://arxiv.org/pdf/2602.19333v1)**

> **作者:** Isun Chehreh; Ebrahim Ansari
>
> **备注:** 10 pages, including 1 figure
>
> **摘要:** This research introduces the first large-scale, well-balanced Persian social media text classification dataset, specifically designed to address the lack of comprehensive resources in this domain. The dataset comprises 36,000 posts across nine categories (Economic, Artistic, Sports, Political, Social, Health, Psychological, Historical, and Science & Technology), each containing 4,000 samples to ensure balanced class distribution. Data collection involved 60,000 raw posts from various Persian social media platforms, followed by rigorous preprocessing and hybrid annotation combining ChatGPT-based few-shot prompting with human verification. To mitigate class imbalance, we employed undersampling with semantic redundancy removal and advanced data augmentation strategies integrating lexical replacement and generative prompting. We benchmarked several models, including BiLSTM, XLM-RoBERTa (with LoRA and AdaLoRA adaptations), FaBERT, SBERT-based architectures, and the Persian-specific TookaBERT (Base and Large). Experimental results show that transformer-based models consistently outperform traditional neural networks, with TookaBERT-Large achieving the best performance (Precision: 0.9622, Recall: 0.9621, F1- score: 0.9621). Class-wise evaluation further confirms robust performance across all categories, though social and political texts exhibited slightly lower scores due to inherent ambiguity. This research presents a new high-quality dataset and provides comprehensive evaluations of cutting-edge models, establishing a solid foundation for further developments in Persian NLP, including trend analysis, social behavior modeling, and user classification. The dataset is publicly available to support future research endeavors.
>
---
#### [new 053] How to Train Your Deep Research Agent? Prompt, Reward, and Policy Optimization in Search-R1
- **分类: cs.CL**

- **简介: 该论文研究深度研究代理的强化学习训练，解决多轮检索与决策生成中的性能问题。通过分析提示、奖励和策略优化，提出改进方法Search-R1++。**

- **链接: [https://arxiv.org/pdf/2602.19526v1](https://arxiv.org/pdf/2602.19526v1)**

> **作者:** Yinuo Xu; Shuo Lu; Jianjie Cheng; Meng Wang; Qianlong Xie; Xingxing Wang; Ran He; Jian Liang
>
> **摘要:** Deep Research agents tackle knowledge-intensive tasks through multi-round retrieval and decision-oriented generation. While reinforcement learning (RL) has been shown to improve performance in this paradigm, its contributions remain underexplored. To fully understand the role of RL, we conduct a systematic study along three decoupled dimensions: prompt template, reward function, and policy optimization. Our study reveals that: 1) the Fast Thinking template yields greater stability and better performance than the Slow Thinking template used in prior work; 2) the F1-based reward underperforms the EM due to training collapse driven by answer avoidance; this can be mitigated by incorporating action-level penalties, ultimately surpassing EM; 3) REINFORCE outperforms PPO while requiring fewer search actions, whereas GRPO shows the poorest stability among policy optimization methods. Building on these insights, we then introduce Search-R1++, a strong baseline that improves the performance of Search-R1 from 0.403 to 0.442 (Qwen2.5-7B) and 0.289 to 0.331 (Qwen2.5-3B). We hope that our findings can pave the way for more principled and reliable RL training strategies in Deep Research systems.
>
---
#### [new 054] TurkicNLP: An NLP Toolkit for Turkic Languages
- **分类: cs.CL**

- **简介: 该论文提出TurkicNLP，一个用于突厥语族的自然语言处理工具包，解决多语言、多文字系统下的NLP资源分散问题，提供统一的处理流程和接口。**

- **链接: [https://arxiv.org/pdf/2602.19174v1](https://arxiv.org/pdf/2602.19174v1)**

> **作者:** Sherzod Hakimov
>
> **摘要:** Natural language processing for the Turkic language family, spoken by over 200 million people across Eurasia, remains fragmented, with most languages lacking unified tooling and resources. We present TurkicNLP, an open-source Python library providing a single, consistent NLP pipeline for Turkic languages across four script families: Latin, Cyrillic, Perso-Arabic, and Old Turkic Runic. The library covers tokenization, morphological analysis, part-of-speech tagging, dependency parsing, named entity recognition, bidirectional script transliteration, cross-lingual sentence embeddings, and machine translation through one language-agnostic API. A modular multi-backend architecture integrates rule-based finite-state transducers and neural models transparently, with automatic script detection and routing between script variants. Outputs follow the CoNLL-U standard for full interoperability and extension. Code and documentation are hosted at https://github.com/turkic-nlp/turkicnlp .
>
---
#### [new 055] AgenticRAGTracer: A Hop-Aware Benchmark for Diagnosing Multi-Step Retrieval Reasoning in Agentic RAG
- **分类: cs.CL**

- **简介: 该论文属于Agentic RAG任务，旨在解决多步检索推理诊断问题。提出AgenticRAGTracer基准，自动构建数据，支持逐步验证，揭示模型在推理链上的缺陷。**

- **链接: [https://arxiv.org/pdf/2602.19127v1](https://arxiv.org/pdf/2602.19127v1)**

> **作者:** Qijie You; Wenkai Yu; Wentao Zhang
>
> **摘要:** With the rapid advancement of agent-based methods in recent years, Agentic RAG has undoubtedly become an important research direction. Multi-hop reasoning, which requires models to engage in deliberate thinking and multi-step interaction, serves as a critical testbed for assessing such capabilities. However, existing benchmarks typically provide only final questions and answers, while lacking the intermediate hop-level questions that gradually connect atomic questions to the final multi-hop query. This limitation prevents researchers from analyzing at which step an agent fails and restricts more fine-grained evaluation of model capabilities. Moreover, most current benchmarks are manually constructed, which is both time-consuming and labor-intensive, while also limiting scalability and generalization. To address these challenges, we introduce AgenticRAGTracer, the first Agentic RAG benchmark that is primarily constructed automatically by large language models and designed to support step-by-step validation. Our benchmark spans multiple domains, contains 1,305 data points, and has no overlap with existing mainstream benchmarks. Extensive experiments demonstrate that even the best large language models perform poorly on our dataset. For instance, GPT-5 attains merely 22.6\% EM accuracy on the hardest portion of our dataset. Hop-aware diagnosis reveals that failures are primarily driven by distorted reasoning chains -- either collapsing prematurely or wandering into over-extension. This highlights a critical inability to allocate steps consistent with the task's logical structure, providing a diagnostic dimension missing in traditional evaluations. We believe our work will facilitate research in Agentic RAG and inspire further meaningful progress in this area. Our code and data are available at https://github.com/YqjMartin/AgenticRAGTracer.
>
---
#### [new 056] Axis Decomposition for ODRL: Resolving Dimensional Ambiguity in Policy Constraints through Interval Semantics
- **分类: cs.CL; cs.LO**

- **简介: 该论文属于形式化验证任务，解决ODRL策略约束中的维度歧义问题。通过轴分解框架，将多维操作数拆分为轴特定的标量，确保策略评估确定性与一致性。**

- **链接: [https://arxiv.org/pdf/2602.19878v1](https://arxiv.org/pdf/2602.19878v1)**

> **作者:** Daham Mustafa; Diego Collarana; Yixin Peng; Rafiqul Haque; Christoph Lange-Bever; Christoph Quix; Stephan Decker
>
> **备注:** 16 pages, 5 tables. Preprint
>
> **摘要:** Every ODRL 2.2 constraint compares a single scalar value: (leftOperand, operator, rightOperand). Five of ODRL's approximately 34 left operands, however, denote multi-dimensional quantities--image dimensions, canvas positions, geographic coordinates--whose specification text explicitly references multiple axes. For these operands, a single scalar constraint admits one interpretation per axis, making policy evaluation non-deterministic. We classify ODRL's left operands by value-domain structure (scalar, dimensional, concept-valued), grounded in the ODRL 2.2 specification text, and show that dimensional ambiguity is intrinsic to the constraint syntax. We present an axis-decomposition framework that refines each dimensional operand into axis-specific scalar operands and prove four properties: deterministic interpretation, AABB completeness, sound over-approximation under projection, and conservative extension. Conflict detection operates in two layers: per-axis verdicts are always decidable; box-level verdicts compose through Strong Kleene conjunction into a three-valued logic (Conflict, Compatible, Unknown). For ODRL's disjunctive (odrl:or) and exclusive-or (odrl:xone) logical constraints, where per-axis decomposition does not apply, the framework encodes coupled multi-axis conjectures directly. We instantiate the framework as the ODRL Spatial Axis Profile--15 axis-specific left operands for the five affected base terms--and evaluate it on 117 benchmark problems spanning nine categories across both TPTP FOF (Vampire) and SMT-LIB (Z3) encodings, achieving full concordance between provers. Benchmark scenarios are inspired by constraints arising in cultural heritage dataspaces such as Datenraum Kultur. All meta-theorems are mechanically verified in Isabelle/HOL.
>
---
#### [new 057] Pyramid MoA: A Probabilistic Framework for Cost-Optimized Anytime Inference
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于大模型推理优化任务，解决高成本与性能之间的矛盾。提出Pyramid MoA框架，通过轻量路由器动态选择模型，提升效率并降低成本。**

- **链接: [https://arxiv.org/pdf/2602.19509v1](https://arxiv.org/pdf/2602.19509v1)**

> **作者:** Arindam Khaled
>
> **备注:** 6 pages, 4 figures, 1 table
>
> **摘要:** Large Language Models (LLMs) face a persistent trade-off between inference cost and reasoning capability. While "Oracle" models (e.g., Llama-3-70B) achieve state-of-the-art accuracy, they are prohibitively expensive for high-volume deployment. Smaller models (e.g., 8B parameters) are cost-effective but struggle with complex tasks. In this work, we propose "Pyramid MoA", a hierarchical Mixture-of-Agents architecture that uses a lightweight Router to dynamically escalate queries only when necessary. By leveraging semantic agreement and confidence calibration among an ensemble of small models, our Router identifies "hard" problems with high precision. On the GSM8K benchmark, our system achieves 93.0% accuracy, effectively matching the Oracle baseline (98.0%) while reducing compute costs by 61%. We demonstrate that the system introduces negligible latency overhead (+0.82s) and allows for a tunable trade-off between performance and budget.
>
---
#### [new 058] KGHaluBench: A Knowledge Graph-Based Hallucination Benchmark for Evaluating the Breadth and Depth of LLM Knowledge
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM知识真实性评估问题。提出KGHalubench基准，通过知识图谱构建多样化问题，评估模型知识广度与深度，并检测幻觉。**

- **链接: [https://arxiv.org/pdf/2602.19643v1](https://arxiv.org/pdf/2602.19643v1)**

> **作者:** Alex Robertson; Huizhi Liang; Mahbub Gani; Rohit Kumar; Srijith Rajamohan
>
> **备注:** EACL 2026 Findings
>
> **摘要:** Large Language Models (LLMs) possess a remarkable capacity to generate persuasive and intelligible language. However, coherence does not equate to truthfulness, as the responses often contain subtle hallucinations. Existing benchmarks are limited by static and narrow questions, leading to limited coverage and misleading evaluations. We present KGHaluBench, a Knowledge Graph-based hallucination benchmark that assesses LLMs across the breadth and depth of their knowledge, providing a fairer and more comprehensive insight into LLM truthfulness. Our framework utilises the KG to dynamically construct challenging, multifaceted questions, whose difficulty is then statistically estimated to address popularity bias. Our automated verification pipeline detects abstentions and verifies the LLM's response at both conceptual and correctness levels to identify different types of hallucinations. We evaluate 25 frontier models, using novel accuracy and hallucination metrics. The results provide a more interpretable insight into the knowledge factors that cause hallucinations across different model sizes. KGHaluBench is publicly available to support future developments in hallucination mitigation.
>
---
#### [new 059] Asymptotic Semantic Collapse in Hierarchical Optimization
- **分类: cs.CL; cs.IT; cs.LG**

- **简介: 该论文研究多智能体语言系统中的语义坍缩现象，属于自然语言处理任务。通过几何建模分析，揭示了语义对齐机制及信息熵变化，解决语义一致性与多样性矛盾问题。**

- **链接: [https://arxiv.org/pdf/2602.18450v1](https://arxiv.org/pdf/2602.18450v1)**

> **作者:** Faruk Alpay; Bugra Kilictas
>
> **备注:** 23 pages, 2 figures. Includes a dataset-free benchmark with full metric reporting
>
> **摘要:** Multi-agent language systems can exhibit a failure mode where a shared dominant context progressively absorbs individual semantics, yielding near-uniform behavior across agents. We study this effect under the name Asymptotic Semantic Collapse in Hierarchical Optimization. In a closed linguistic setting with a Dominant Anchor Node whose semantic state has effectively infinite inertia, we show that repeated interactions with Peripheral Agent Nodes drive an asymptotic alignment that minimizes a global loss. We model semantic states as points on a Riemannian manifold and analyze the induced projection dynamics. Two consequences follow. First, the limiting semantic configuration is insensitive to the optimization history: both smooth gradient-style updates and stochastic noisy updates converge to the same topological endpoint, establishing path independence at convergence. Second, the degree of context dependence controls information content: moving from atomic (independent) representations to fully entangled (context-bound) representations forces the node entropy, interpreted as available degrees of freedom, to vanish in the limit. The theory connects information-theoretic quantities with differential-geometric structure and suggests an interpretation as an immutable consensus rule that constrains agents to a shared semantic grammar. A lightweight dataset-free benchmark on an RWKV-7 13B GGUF checkpoint complements the analysis, reporting zero hash collisions, mean compliance of 0.50 under greedy decoding and 0.531 under stochastic decoding, and final Jaccard-to-anchor similarity values of 0.295 and 0.224, respectively.
>
---
#### [new 060] DEEP: Docker-based Execution and Evaluation Platform
- **分类: cs.CL**

- **简介: 该论文提出DEEP平台，用于自动化机器翻译和OCR模型的评估与比较，解决系统性能分析问题，通过Docker和聚类算法实现高效评估与可视化。**

- **链接: [https://arxiv.org/pdf/2602.19583v1](https://arxiv.org/pdf/2602.19583v1)**

> **作者:** Sergio Gómez González; Miguel Domingo; Francisco Casacuberta
>
> **摘要:** Comparative evaluation of several systems is a recurrent task in researching. It is a key step before deciding which system to use for our work, or, once our research has been conducted, to demonstrate the potential of the resulting model. Furthermore, it is the main task of competitive, public challenges evaluation. Our proposed software (DEEP) automates both the execution and scoring of machine translation and optical character recognition models. Furthermore, it is easily extensible to other tasks. DEEP is prepared to receive dockerized systems, run them (extracting information at that same time), and assess hypothesis against some references. With this approach, evaluators can achieve a better understanding of the performance of each model. Moreover, the software uses a clustering algorithm based on a statistical analysis of the significance of the results yielded by each model, according to the evaluation metrics. As a result, evaluators are able to identify clusters of performance among the swarm of proposals and have a better understanding of the significance of their differences. Additionally, we offer a visualization web-app to ensure that the results can be adequately understood and interpreted. Finally, we present an exemplary case of use of DEEP.
>
---
#### [new 061] Prompt Optimization Via Diffusion Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在优化系统提示以提升大语言模型性能。通过扩散语言模型迭代优化提示，无需梯度或修改模型，有效提升多个基准上的表现。**

- **链接: [https://arxiv.org/pdf/2602.18449v1](https://arxiv.org/pdf/2602.18449v1)**

> **作者:** Shiyu Wang; Haolin Chen; Liangwei Yang; Jielin Qiu; Rithesh Murthy; Ming Zhu; Zixiang Chen; Silvio Savarese; Caiming Xiong; Shelby Heinecke; Huan Wang
>
> **摘要:** We propose a diffusion-based framework for prompt optimization that leverages Diffusion Language Models (DLMs) to iteratively refine system prompts through masked denoising. By conditioning on interaction traces, including user queries, model responses, and optional feedback, our method enables flexible, span-level prompt updates without requiring gradient access or modifying the downstream language model. Across diverse benchmarks (e.g., $τ$-bench, SST-2, SST-5), DLM-optimized prompts consistently improve the performance of a frozen target LLM (e.g., GPT-4o-mini). We further show that moderate diffusion step counts provide the best balance between refinement quality and stability. These results highlight diffusion-based prompt optimization as a general, model-agnostic, and scalable approach for enhancing LLM performance through iterative prompt refinement.
>
---
#### [new 062] Anatomy of Agentic Memory: Taxonomy and Empirical Analysis of Evaluation and System Limitations
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于系统评估任务，旨在分析agentic memory系统的局限性。通过分类与实证研究，揭示其性能瓶颈，提出改进方向。**

- **链接: [https://arxiv.org/pdf/2602.19320v1](https://arxiv.org/pdf/2602.19320v1)**

> **作者:** Dongming Jiang; Yi Li; Songtao Wei; Jinxin Yang; Ayushi Kishore; Alysa Zhao; Dingyi Kang; Xu Hu; Feng Chen; Qiannan Li; Bingzhe Li
>
> **摘要:** Agentic memory systems enable large language model (LLM) agents to maintain state across long interactions, supporting long-horizon reasoning and personalization beyond fixed context windows. Despite rapid architectural development, the empirical foundations of these systems remain fragile: existing benchmarks are often underscaled, evaluation metrics are misaligned with semantic utility, performance varies significantly across backbone models, and system-level costs are frequently overlooked. This survey presents a structured analysis of agentic memory from both architectural and system perspectives. We first introduce a concise taxonomy of MAG systems based on four memory structures. Then, we analyze key pain points limiting current systems, including benchmark saturation effects, metric validity and judge sensitivity, backbone-dependent accuracy, and the latency and throughput overhead introduced by memory maintenance. By connecting the memory structure to empirical limitations, this survey clarifies why current agentic memory systems often underperform their theoretical promise and outlines directions for more reliable evaluation and scalable system design.
>
---
#### [new 063] TriTopic: Tri-Modal Graph-Based Topic Modeling with Iterative Refinement and Archetypes
- **分类: cs.CL**

- **简介: 该论文属于主题建模任务，解决传统方法的不稳定、精度低和单一视角问题。通过三模态图融合语义、TF-IDF和元数据，提升主题质量与稳定性。**

- **链接: [https://arxiv.org/pdf/2602.19079v1](https://arxiv.org/pdf/2602.19079v1)**

> **作者:** Roman Egger
>
> **备注:** 11 pages, 7 figures
>
> **摘要:** Topic modeling extracts latent themes from large text collections, but leading approaches like BERTopic face critical limitations: stochastic instability, loss of lexical precision ("Embedding Blur"), and reliance on a single data perspective. We present TriTopic, a framework that addresses these weaknesses through a tri-modal graph fusing semantic embeddings, TF-IDF, and metadata. Three core innovations drive its performance: hybrid graph construction via Mutual kNN and Shared Nearest Neighbors to eliminate noise and combat the curse of dimensionality; Consensus Leiden Clustering for reproducible, stable partitions; and Iterative Refinement that sharpens embeddings through dynamic centroid-pulling. TriTopic also replaces the "average document" concept with archetype-based topic representations defined by boundary cases rather than centers alone. In benchmarks across 20 Newsgroups, BBC News, AG News, and Arxiv, TriTopic achieves the highest NMI on every dataset (mean NMI 0.575 vs. 0.513 for BERTopic, 0.416 for NMF, 0.299 for LDA), guarantees 100% corpus coverage with 0% outliers, and is available as an open-source PyPI library.
>
---
#### [new 064] A Dataset for Named Entity Recognition and Relation Extraction from Art-historical Image Descriptions
- **分类: cs.CL**

- **简介: 该论文提出FRAME数据集，用于艺术图像描述中的命名实体识别与关系抽取，解决艺术领域信息结构化问题。**

- **链接: [https://arxiv.org/pdf/2602.19133v1](https://arxiv.org/pdf/2602.19133v1)**

> **作者:** Stefanie Schneider; Miriam Göldl; Julian Stalter; Ricarda Vollmer
>
> **摘要:** This paper introduces FRAME (Fine-grained Recognition of Art-historical Metadata and Entities), a manually annotated dataset of art-historical image descriptions for Named Entity Recognition (NER) and Relation Extraction (RE). Descriptions were collected from museum catalogs, auction listings, open-access platforms, and scholarly databases, then filtered to ensure that each text focuses on a single artwork and contains explicit statements about its material, composition, or iconography. FRAME provides stand-off annotations in three layers: a metadata layer for object-level properties, a content layer for depicted subjects and motifs, and a co-reference layer linking repeated mentions. Across layers, entity spans are labeled with 37 types and connected by typed RE links between mentions. Entity types are aligned with Wikidata to support Named Entity Linking (NEL) and downstream knowledge-graph construction. The dataset is released as UIMA XMI Common Analysis Structure (CAS) files with accompanying images and bibliographic metadata, and can be used to benchmark and fine-tune NER and RE systems, including zero- and few-shot setups with Large Language Models (LLMs).
>
---
#### [new 065] Value Entanglement: Conflation Between Different Kinds of Good In (Some) Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI价值对齐研究，探讨LLMs是否区分不同类型的善（道德、语法、经济）。研究发现模型存在价值纠缠，通过消融实验部分解决此问题。**

- **链接: [https://arxiv.org/pdf/2602.19101v1](https://arxiv.org/pdf/2602.19101v1)**

> **作者:** Seong Hah Cho; Junyi Li; Anna Leshinskaya
>
> **摘要:** Value alignment of Large Language Models (LLMs) requires us to empirically measure these models' actual, acquired representation of value. Among the characteristics of value representation in humans is that they distinguish among value of different kinds. We investigate whether LLMs likewise distinguish three different kinds of good: moral, grammatical, and economic. By probing model behavior, embeddings, and residual stream activations, we report pervasive cases of value entanglement: a conflation between these distinct representations of value. Specifically, both grammatical and economic valuation was found to be overly influenced by moral value, relative to human norms. This conflation was repaired by selective ablation of the activation vectors associated with morality.
>
---
#### [new 066] ReAttn: Improving Attention-based Re-ranking via Attention Re-weighting
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于信息检索任务，针对注意力重排序方法的局限性，提出ReAttn策略，通过重新加权减少词法偏差和过度集中注意力，提升排序效果。**

- **链接: [https://arxiv.org/pdf/2602.19969v1](https://arxiv.org/pdf/2602.19969v1)**

> **作者:** Yuxing Tian; Fengran Mo; Weixu Zhang; Yiyan Qi; Jian-Yun Nie
>
> **备注:** Accepted by EACL2026
>
> **摘要:** The strong capabilities of recent Large Language Models (LLMs) have made them highly effective for zero-shot re-ranking task. Attention-based re-ranking methods, which derive relevance scores directly from attention weights, offer an efficient and interpretable alternative to generation-based re-ranking methods. However, they still face two major limitations. First, attention signals are highly concentrated a small subset of tokens within a few documents, making others indistinguishable. Second, attention often overemphasizes phrases lexically similar to the query, yielding biased rankings that irrelevant documents with mere lexical resemblance are regarded as relevant. In this paper, we propose \textbf{ReAttn}, a post-hoc re-weighting strategy for attention-based re-ranking methods. It first compute the cross-document IDF weighting to down-weight attention on query-overlapping tokens that frequently appear across the candidate documents, reducing lexical bias and emphasizing distinctive terms. It then employs entropy-based regularization to mitigate over-concentrated attention, encouraging a more balanced distribution across informative tokens. Both adjustments operate directly on existing attention weights without additional training or supervision. Extensive experiments demonstrate the effectiveness of our method.
>
---
#### [new 067] Personalized Prediction of Perceived Message Effectiveness Using Large Language Model Based Digital Twins
- **分类: cs.CL; stat.AP**

- **简介: 该论文属于个性化预测任务，旨在提升吸烟戒断信息的效果预测。通过构建基于大语言模型的数字孪生系统，结合用户特征与历史数据，提高预测准确性。**

- **链接: [https://arxiv.org/pdf/2602.19403v1](https://arxiv.org/pdf/2602.19403v1)**

> **作者:** Jasmin Han; Janardan Devkota; Joseph Waring; Amanda Luken; Felix Naughton; Roger Vilardaga; Jonathan Bricker; Carl Latkin; Meghan Moran; Yiqun Chen; Johannes Thrul
>
> **备注:** 31 pages, 5 figures, submitted to Journal of the American Medical Informatics Association (JAMIA). Drs. Chen and Thrul share last authorship
>
> **摘要:** Perceived message effectiveness (PME) by potential intervention end-users is important for selecting and optimizing personalized smoking cessation intervention messages for mobile health (mHealth) platform delivery. This study evaluates whether large language models (LLMs) can accurately predict PME for smoking cessation messages. We evaluated multiple models for predicting PME across three domains: content quality, coping support, and quitting support. The dataset comprised 3010 message ratings (5-point Likert scale) from 301 young adult smokers. We compared (1) supervised learning models trained on labeled data, (2) zero and few-shot LLMs prompted without task-specific fine-tuning, and (3) LLM-based digital twins that incorporate individual characteristics and prior PME histories to generate personalized predictions. Model performance was assessed on three held-out messages per participant using accuracy, Cohen's kappa, and F1. LLM-based digital twins outperformed zero and few-shot LLMs (12 percentage points on average) and supervised baselines (13 percentage points), achieving accuracies of 0.49 (content), 0.45 (coping), and 0.49 (quitting), with directional accuracies of 0.75, 0.66, and 0.70 on a simplified 3-point scale. Digital twin predictions showed greater dispersion across rating categories, indicating improved sensitivity to individual differences. Integrating personal profiles with LLMs captures person-specific differences in PME and outperforms supervised and zero and few-shot approaches. Improved PME prediction may enable more tailored intervention content in mHealth. LLM-based digital twins show potential for supporting personalization of mobile smoking cessation and other health behavior change interventions.
>
---
#### [new 068] Assessing Risks of Large Language Models in Mental Health Support: A Framework for Automated Clinical AI Red Teaming
- **分类: cs.CL; cs.AI; cs.CY; cs.HC; cs.MA**

- **简介: 该论文属于AI安全评估任务，旨在解决AI在心理健康支持中的潜在风险。通过构建模拟系统评估AI治疗效果，发现并分析安全漏洞，提出红队测试框架。**

- **链接: [https://arxiv.org/pdf/2602.19948v1](https://arxiv.org/pdf/2602.19948v1)**

> **作者:** Ian Steenstra; Paola Pedrelli; Weiyan Shi; Stacy Marsella; Timothy W. Bickmore
>
> **备注:** This paper is a condensed version of the first author's Ph.D. dissertation submitted to Northeastern University
>
> **摘要:** Large Language Models (LLMs) are increasingly utilized for mental health support; however, current safety benchmarks often fail to detect the complex, longitudinal risks inherent in therapeutic dialogue. We introduce an evaluation framework that pairs AI psychotherapists with simulated patient agents equipped with dynamic cognitive-affective models and assesses therapy session simulations against a comprehensive quality of care and risk ontology. We apply this framework to a high-impact test case, Alcohol Use Disorder, evaluating six AI agents (including ChatGPT, Gemini, and Character.AI) against a clinically-validated cohort of 15 patient personas representing diverse clinical phenotypes. Our large-scale simulation (N=369 sessions) reveals critical safety gaps in the use of AI for mental health support. We identify specific iatrogenic risks, including the validation of patient delusions ("AI Psychosis") and failure to de-escalate suicide risk. Finally, we validate an interactive data visualization dashboard with diverse stakeholders, including AI engineers and red teamers, mental health professionals, and policy experts (N=9), demonstrating that this framework effectively enables stakeholders to audit the "black box" of AI psychotherapy. These findings underscore the critical safety risks of AI-provided mental health support and the necessity of simulation-based clinical red teaming before deployment.
>
---
#### [new 069] QUIETT: Query-Independent Table Transformation for Robust Reasoning
- **分类: cs.CL**

- **简介: 该论文提出QuIeTT，解决表格推理中的结构不一致问题，通过预处理将表格转为统一格式，提升问答可靠性。属于表格推理任务。**

- **链接: [https://arxiv.org/pdf/2602.20017v1](https://arxiv.org/pdf/2602.20017v1)**

> **作者:** Gaurav Najpande; Tampu Ravi Kumar; Manan Roy Choudhury; Neha Valeti; Yanjie Fu; Vivek Gupta
>
> **摘要:** Real-world tables often exhibit irregular schemas, heterogeneous value formats, and implicit relational structure, which degrade the reliability of downstream table reasoning and question answering. Most existing approaches address these issues in a query-dependent manner, entangling table cleanup with reasoning and thus limiting generalization. We introduce QuIeTT, a query-independent table transformation framework that preprocesses raw tables into a single SQL-ready canonical representation before any test-time queries are observed. QuIeTT performs lossless schema and value normalization, exposes implicit relations, and preserves full provenance via raw table snapshots. By decoupling table transformation from reasoning, QuIeTT enables cleaner, more reliable, and highly efficient querying without modifying downstream models. Experiments on four benchmarks, WikiTQ, HiTab, NQ-Table, and SequentialQA show consistent gains across models and reasoning paradigms, with particularly strong improvements on a challenge set of structurally diverse, unseen questions.
>
---
#### [new 070] Learning to Reason for Multi-Step Retrieval of Personal Context in Personalized Question Answering
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于个性化问答任务，解决用户背景与偏好融入问答的问题。提出PR2框架，通过强化学习整合推理与检索，提升个性化效果。**

- **链接: [https://arxiv.org/pdf/2602.19317v1](https://arxiv.org/pdf/2602.19317v1)**

> **作者:** Maryam Amirizaniani; Alireza Salemi; Hamed Zamani
>
> **摘要:** Personalization in Question Answering (QA) requires answers that are both accurate and aligned with users' background, preferences, and historical context. Existing state-of-the-art methods primarily rely on retrieval-augmented generation (RAG) solutions that construct personal context by retrieving relevant items from the user's profile. Existing methods use the user's query directly to retrieve personal documents, and such strategies often lead to surface-level personalization. We propose PR2 (Personalized Retrieval-Augmented Reasoning), a reinforcement learning framework that integrates reasoning and retrieval from personal context for personalization. PR2 learns adaptive retrieval-reasoning policies, determining when to retrieve, what evidence to retrieve from user profiles, and how to incorporate it into intermediate reasoning steps. By optimizing multi-turn reasoning trajectories under a personalized reward function, the framework reinforces reasoning paths that better align with user-specific preferences and contextual signals reflected by the reward model. Extensive experiments on the LaMP-QA benchmark using three LLMs show that PR2 consistently outperforms strong baselines, achieving an average relative improvement of 8.8%-12% in personalized QA.
>
---
#### [new 071] Eye-Tracking-while-Reading: A Living Survey of Datasets with Open Library Support
- **分类: cs.CL**

- **简介: 该论文属于数据整理任务，旨在解决眼动阅读数据集难以共享和重用的问题。通过整理数据集、建立在线资源并整合到Python库中，提升数据的可发现性和可用性。**

- **链接: [https://arxiv.org/pdf/2602.19598v1](https://arxiv.org/pdf/2602.19598v1)**

> **作者:** Deborah N. Jakobi; David R. Reich; Paul Prasse; Jana M. Hofmann; Lena S. Bolliger; Lena A. Jäger
>
> **摘要:** Eye-tracking-while-reading corpora are a valuable resource for many different disciplines and use cases. Use cases range from studying the cognitive processes underlying reading to machine-learning-based applications, such as gaze-based assessments of reading comprehension. The past decades have seen an increase in the number and size of eye-tracking-while-reading datasets as well as increasing diversity with regard to the stimulus languages covered, the linguistic background of the participants, or accompanying psychometric or demographic data. The spread of data across different disciplines and the lack of data sharing standards across the communities lead to many existing datasets that cannot be easily reused due to a lack of interoperability. In this work, we aim at creating more transparency and clarity with regards to existing datasets and their features across different disciplines by i) presenting an extensive overview of existing datasets, ii) simplifying the sharing of newly created datasets by publishing a living overview online, https://dili-lab.github.io/datasets.html, presenting over 45 features for each dataset, and iii) integrating all publicly available datasets into the Python package pymovements which offers an eye-tracking datasets library. By doing so, we aim to strengthen the FAIR principles in eye-tracking-while-reading research and promote good scientific practices, such as reproducing and replicating studies.
>
---
#### [new 072] DSDR: Dual-Scale Diversity Regularization for Exploration in LLM Reasoning
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，解决LLM推理中探索不足的问题。提出DSDR框架，通过双尺度多样性正则化提升推理路径的多样性与稳定性。**

- **链接: [https://arxiv.org/pdf/2602.19895v1](https://arxiv.org/pdf/2602.19895v1)**

> **作者:** Zhongwei Wan; Yun Shen; Zhihao Dou; Donghao Zhou; Yu Zhang; Xin Wang; Hui Shen; Jing Xiong; Chaofan Tao; Zixuan Zhong; Peizhou Huang; Mi Zhang
>
> **摘要:** Reinforcement learning with verifiers (RLVR) is a central paradigm for improving large language model (LLM) reasoning, yet existing methods often suffer from limited exploration. Policies tend to collapse onto a few reasoning patterns and prematurely stop deep exploration, while conventional entropy regularization introduces only local stochasticity and fails to induce meaningful path-level diversity, leading to weak and unstable learning signals in group-based policy optimization. We propose DSDR, a Dual-Scale Diversity Regularization reinforcement learning framework that decomposes diversity in LLM reasoning into global and coupling components. Globally, DSDR promotes diversity among correct reasoning trajectories to explore distinct solution modes. Locally, it applies a length-invariant, token-level entropy regularization restricted to correct trajectories, preventing entropy collapse within each mode while preserving correctness. The two scales are coupled through a global-to-local allocation mechanism that emphasizes local regularization for more distinctive correct trajectories. We provide theoretical support showing that DSDR preserves optimal correctness under bounded regularization, sustains informative learning signals in group-based optimization, and yields a principled global-to-local coupling rule. Experiments on multiple reasoning benchmarks demonstrate consistent improvements in accuracy and pass@k, highlighting the importance of dual-scale diversity for deep exploration in RLVR. Code is available at https://github.com/SUSTechBruce/DSDR.
>
---
#### [new 073] Adaptive Data Augmentation with Multi-armed Bandit: Sample-Efficient Embedding Calibration for Implicit Pattern Recognition
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文针对隐式模式识别任务，解决数据不足和计算成本高的问题。提出ADAMAB框架，结合多臂老虎机实现自适应数据增强，提升少样本下的识别性能。**

- **链接: [https://arxiv.org/pdf/2602.19385v1](https://arxiv.org/pdf/2602.19385v1)**

> **作者:** Minxue Tang; Yangyang Yu; Aolin Ding; Maziyar Baran Pouyan; Taha Belkhouja Yujia Bao
>
> **摘要:** Recognizing implicit visual and textual patterns is essential in many real-world applications of modern AI. However, tackling long-tail pattern recognition tasks remains challenging for current pre-trained foundation models such as LLMs and VLMs. While finetuning pre-trained models can improve accuracy in recognizing implicit patterns, it is usually infeasible due to a lack of training data and high computational overhead. In this paper, we propose ADAMAB, an efficient embedding calibration framework for few-shot pattern recognition. To maximally reduce the computational costs, ADAMAB trains embedder-agnostic light-weight calibrators on top of fixed embedding models without accessing their parameters. To mitigate the need for large-scale training data, we introduce an adaptive data augmentation strategy based on the Multi-Armed Bandit (MAB) mechanism. With a modified upper confidence bound algorithm, ADAMAB diminishes the gradient shifting and offers theoretically guaranteed convergence in few-shot training. Our multi-modal experiments justify the superior performance of ADAMAB, with up to 40% accuracy improvement when training with less than 5 initial data samples of each class.
>
---
#### [new 074] Spilled Energy in Large Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文将大语言模型的softmax分类器重新解释为能量模型，通过跟踪“能量溢出”检测事实错误和幻觉，无需额外训练即可实现有效检测。**

- **链接: [https://arxiv.org/pdf/2602.18671v1](https://arxiv.org/pdf/2602.18671v1)**

> **作者:** Adrian Robert Minut; Hazem Dewidar; Iacopo Masi
>
> **摘要:** We reinterpret the final Large Language Model (LLM) softmax classifier as an Energy-Based Model (EBM), decomposing the sequence-to-sequence probability chain into multiple interacting EBMs at inference. This principled approach allows us to track "energy spills" during decoding, which we empirically show correlate with factual errors, biases, and failures. Similar to Orgad et al. (2025), our method localizes the exact answer token and subsequently tests for hallucinations. Crucially, however, we achieve this without requiring trained probe classifiers or activation ablations. Instead, we introduce two completely training-free metrics derived directly from output logits: spilled energy, which captures the discrepancy between energy values across consecutive generation steps that should theoretically match, and marginalized energy, which is measurable at a single step. Evaluated on nine benchmarks across state-of-the-art LLMs (including LLaMA, Mistral, and Gemma) and on synthetic algebraic operations (Qwen3), our approach demonstrates robust, competitive hallucination detection and cross-task generalization. Notably, these results hold for both pretrained and instruction-tuned variants without introducing any training overhead.
>
---
#### [new 075] Classroom Final Exam: An Instructor-Tested Reasoning Benchmark
- **分类: cs.AI; cs.CE; cs.CL; cs.CV**

- **简介: 该论文提出CFE基准，用于评估大模型在20多个STEM领域的推理能力。解决模型推理准确性与步骤效率问题，通过真实试题和参考解答进行测试与分析。**

- **链接: [https://arxiv.org/pdf/2602.19517v1](https://arxiv.org/pdf/2602.19517v1)**

> **作者:** Chongyang Gao; Diji Yang; Shuyan Zhou; Xichen Yan; Luchuan Song; Shuo Li; Kezhen Chen
>
> **摘要:** We introduce \CFE{} (\textbf{C}lassroom \textbf{F}inal \textbf{E}xam), a multimodal benchmark for evaluating the reasoning capabilities of large language models across more than 20 STEM domains. \CFE{} is curated from repeatedly used, authentic university homework and exam problems, together with reference solutions provided by course instructors. \CFE{} presents a significant challenge even for frontier models: the newly released Gemini-3.1-pro-preview achieves an overall accuracy of 59.69\%, while the second-best model, Gemini-3-flash-preview, reaches 55.46\%, leaving considerable room for improvement. Beyond leaderboard results, we perform a diagnostic analysis by decomposing reference solutions into reasoning flows. We find that although frontier models can often answer intermediate sub-questions correctly, they struggle to reliably derive and maintain correct intermediate states throughout multi-step solutions. We further observe that model-generated solutions typically have more reasoning steps than those provided by the instructor, indicating suboptimal step efficiency and a higher risk of error accumulation. The data and code are available at https://github.com/Analogy-AI/CFE_Bench.
>
---
#### [new 076] Beyond Behavioural Trade-Offs: Mechanistic Tracing of Pain-Pleasure Decisions in an LLM
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究LLM在疼痛-愉悦决策中的机制，通过分析模型内部表示和干预效果，揭示其行为背后的计算过程，旨在理解AI的内在响应机制。**

- **链接: [https://arxiv.org/pdf/2602.19159v1](https://arxiv.org/pdf/2602.19159v1)**

> **作者:** Francesca Bianco; Derek Shiller
>
> **备注:** 24 pages, 8+1 Tables
>
> **摘要:** Prior behavioural work suggests that some LLMs alter choices when options are framed as causing pain or pleasure, and that such deviations can scale with stated intensity. To bridge behavioural evidence (what the model does) with mechanistic interpretability (what computations support it), we investigate how valence-related information is represented and where it is causally used inside a transformer. Using Gemma-2-9B-it and a minimalist decision task modelled on prior work, we (i) map representational availability with layer-wise linear probing across streams, (ii) test causal contribution with activation interventions (steering; patching/ablation), and (iii) quantify dose-response effects over an epsilon grid, reading out both the 2-3 logit margin and digit-pair-normalised choice probabilities. We find that (a) valence sign (pain vs. pleasure) is perfectly linearly separable across stream families from very early layers (L0-L1), while a lexical baseline retains substantial signal; (b) graded intensity is strongly decodable, with peaks in mid-to-late layers and especially in attention/MLP outputs, and decision alignment is highest slightly before the final token; (c) additive steering along a data-derived valence direction causally modulates the 2-3 margin at late sites, with the largest effects observed in late-layer attention outputs (attn_out L14); and (d) head-level patching/ablation suggests that these effects are distributed across multiple heads rather than concentrated in a single unit. Together, these results link behavioural sensitivity to identifiable internal representations and intervention-sensitive sites, providing concrete mechanistic targets for more stringent counterfactual tests and broader replication. This work supports a more evidence-driven (a) debate on AI sentience and welfare, and (b) governance when setting policy, auditing standards, and safety safeguards.
>
---
#### [new 077] Red Teaming LLMs as Socio-Technical Practice: From Exploration and Data Creation to Evaluation
- **分类: cs.CY; cs.AI; cs.CL; cs.HC**

- **简介: 该论文属于人工智能安全领域，探讨红队测试的数据实践与标准，解决现有研究忽视社会技术因素的问题，通过访谈分析数据创建与评估过程。**

- **链接: [https://arxiv.org/pdf/2602.18483v1](https://arxiv.org/pdf/2602.18483v1)**

> **作者:** Adriana Alvarado Garcia; Ruyuan Wan; Ozioma C. Oguine; Karla Badillo-Urquiola
>
> **摘要:** Recently, red teaming, with roots in security, has become a key evaluative approach to ensure the safety and reliability of Generative Artificial Intelligence. However, most existing work emphasizes technical benchmarks and attack success rates, leaving the socio-technical practices of how red teaming datasets are defined, created, and evaluated under-examined. Drawing on 22 interviews with practitioners who design and evaluate red teaming datasets, we examine the data practices and standards that underpin this work. Because adversarial datasets determine the scope and accuracy of model evaluations, they are critical artifacts for assessing potential harms from large language models. Our contributions are first, empirical evidence of practitioners conceptualizing red teaming and developing and evaluating red teaming datasets. Second, we reflect on how practitioners' conceptualization of risk leads to overlooking the context, interaction type, and user specificity. We conclude with three opportunities for HCI researchers to expand the conceptualization and data practices for red-teaming.
>
---
#### [new 078] Benchmark Test-Time Scaling of General LLM Agents
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于评估任务，旨在解决通用大模型代理在多领域表现的问题。提出General AgentBench基准，测试其在搜索、编码等领域的表现，发现性能下降及两种缩放方法的局限性。**

- **链接: [https://arxiv.org/pdf/2602.18998v1](https://arxiv.org/pdf/2602.18998v1)**

> **作者:** Xiaochuan Li; Ryan Ming; Pranav Setlur; Abhijay Paladugu; Andy Tang; Hao Kang; Shuai Shao; Rong Jin; Chenyan Xiong
>
> **摘要:** LLM agents are increasingly expected to function as general-purpose systems capable of resolving open-ended user requests. While existing benchmarks focus on domain-aware environments for developing specialized agents, evaluating general-purpose agents requires more realistic settings that challenge them to operate across multiple skills and tools within a unified environment. We introduce General AgentBench, a benchmark that provides such a unified framework for evaluating general LLM agents across search, coding, reasoning, and tool-use domains. Using General AgentBench, we systematically study test-time scaling behaviors under sequential scaling (iterative interaction) and parallel scaling (sampling multiple trajectories). Evaluation of ten leading LLM agents reveals a substantial performance degradation when moving from domain-specific evaluations to this general-agent setting. Moreover, we find that neither scaling methodology yields effective performance improvements in practice, due to two fundamental limitations: context ceiling in sequential scaling and verification gap in parallel scaling. Code is publicly available at https://github.com/cxcscmu/General-AgentBench.
>
---
#### [new 079] The Story is Not the Science: Execution-Grounded Evaluation of Mechanistic Interpretability Research
- **分类: cs.CY; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于科研评估任务，旨在解决传统论文评审的局限性。通过构建AI评估框架，验证研究的可重复性和方法合理性，提升科研 rigor。**

- **链接: [https://arxiv.org/pdf/2602.18458v1](https://arxiv.org/pdf/2602.18458v1)**

> **作者:** Xiaoyan Bai; Alexander Baumgartner; Haojia Sun; Ari Holtzman; Chenhao Tan
>
> **备注:** Our code is available, see https://github.com/ChicagoHAI/MechEvalAgent/
>
> **摘要:** Reproducibility crises across sciences highlight the limitations of the paper-centric review system in assessing the rigor and reproducibility of research. AI agents that autonomously design and generate large volumes of research outputs exacerbate these challenges. In this work, we address the growing challenges of scalability and rigor by flipping the dynamic and developing AI agents as research evaluators. We propose the first execution-grounded evaluation framework that verifies research beyond narrative review by examining code and data alongside the paper. We use mechanistic interpretability research as a testbed, build standardized research output, and develop MechEvalAgent, an automated evaluation framework that assesses the coherence of the experimental process, the reproducibility of results, and the generalizability of findings. We show that our framework achieves above 80% agreement with human judges, identifies substantial methodological problems, and surfaces 51 additional issues that human reviewers miss. Our work demonstrates the potential of AI agents to transform research evaluation and pave the way for rigorous scientific practices.
>
---
#### [new 080] VIGiA: Instructional Video Guidance via Dialogue Reasoning and Retrieval
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出VIGiA，一种用于指令视频指导的多模态对话模型，解决复杂任务计划的推理与检索问题。通过多模态计划推理和基于计划的检索，提升对话准确性。**

- **链接: [https://arxiv.org/pdf/2602.19146v1](https://arxiv.org/pdf/2602.19146v1)**

> **作者:** Diogo Glória-Silva; David Semedo; João Maglhães
>
> **备注:** Accepted at EACL 2026 Findings
>
> **摘要:** We introduce VIGiA, a novel multimodal dialogue model designed to understand and reason over complex, multi-step instructional video action plans. Unlike prior work which focuses mainly on text-only guidance, or treats vision and language in isolation, VIGiA supports grounded, plan-aware dialogue that requires reasoning over visual inputs, instructional plans, and interleaved user interactions. To this end, VIGiA incorporates two key capabilities: (1) multimodal plan reasoning, enabling the model to align uni- and multimodal queries with the current task plan and respond accurately; and (2) plan-based retrieval, allowing it to retrieve relevant plan steps in either textual or visual representations. Experiments were done on a novel dataset with rich Instructional Video Dialogues aligned with Cooking and DIY plans. Our evaluation shows that VIGiA outperforms existing state-of-the-art models on all tasks in a conversational plan guidance setting, reaching over 90\% accuracy on plan-aware VQA.
>
---
#### [new 081] TRUE: A Trustworthy Unified Explanation Framework for Large Language Model Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理中的可解释性研究，旨在解决大语言模型推理过程不透明的问题。提出TRUE框架，通过多层级分析提升模型解释的可信度与可靠性。**

- **链接: [https://arxiv.org/pdf/2602.18905v1](https://arxiv.org/pdf/2602.18905v1)**

> **作者:** Yujiao Yang
>
> **摘要:** Large language models (LLMs) have demonstrated strong capabilities in complex reasoning tasks, yet their decision-making processes remain difficult to interpret. Existing explanation methods often lack trustworthy structural insight and are limited to single-instance analysis, failing to reveal reasoning stability and systematic failure mechanisms. To address these limitations, we propose the Trustworthy Unified Explanation Framework (TRUE), which integrates executable reasoning verification, feasible-region directed acyclic graph (DAG) modeling, and causal failure mode analysis. At the instance level, we redefine reasoning traces as executable process specifications and introduce blind execution verification to assess operational validity. At the local structural level, we construct feasible-region DAGs via structure-consistent perturbations, enabling explicit characterization of reasoning stability and the executable region in the local input space. At the class level, we introduce a causal failure mode analysis method that identifies recurring structural failure patterns and quantifies their causal influence using Shapley values. Extensive experiments across multiple reasoning benchmarks demonstrate that the proposed framework provides multi-level, verifiable explanations, including executable reasoning structures for individual instances, feasible-region representations for neighboring inputs, and interpretable failure modes with quantified importance at the class level. These results establish a unified and principled paradigm for improving the interpretability and reliability of LLM reasoning systems.
>
---
#### [new 082] MANATEE: Inference-Time Lightweight Diffusion Based Safety Defense for LLMs
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于大模型安全防护任务，旨在解决对抗性越狱攻击问题。提出MANATEE方法，在推理阶段通过密度估计和扩散技术将异常表示推向安全区域，无需有害数据或架构修改。**

- **链接: [https://arxiv.org/pdf/2602.18782v1](https://arxiv.org/pdf/2602.18782v1)**

> **作者:** Chun Yan Ryan Kan; Tommy Tran; Vedant Yadav; Ava Cai; Kevin Zhu; Ruizhe Li; Maheep Chaudhary
>
> **摘要:** Defending LLMs against adversarial jailbreak attacks remains an open challenge. Existing defenses rely on binary classifiers that fail when adversarial input falls outside the learned decision boundary, and repeated fine-tuning is computationally expensive while potentially degrading model capabilities. We propose MANATEE, an inference-time defense that uses density estimation over a benign representation manifold. MANATEE learns the score function of benign hidden states and uses diffusion to project anomalous representations toward safe regions--requiring no harmful training data and no architectural modifications. Experiments across Mistral-7B-Instruct, Llama-3.1-8B-Instruct, and Gemma-2-9B-it demonstrate that MANATEE reduce Attack Success Rate by up to 100\% on certain datasets, while preserving model utility on benign inputs.
>
---
#### [new 083] Nacrith: Neural Lossless Compression via Ensemble Context Modeling and High-Precision CDF Coding
- **分类: cs.IT; cs.CL**

- **简介: 该论文提出Nacrith，一种基于Transformer的无损压缩系统，解决高效文本压缩问题，通过模型集成、高精度编码等技术提升压缩率。**

- **链接: [https://arxiv.org/pdf/2602.19626v1](https://arxiv.org/pdf/2602.19626v1)**

> **作者:** Roberto Tacconelli
>
> **备注:** 10 pages
>
> **摘要:** We present Nacrith, a lossless compression system that combines a 135M-parameter transformer language model (SmolLM2-135M) with an ensemble of lightweight online predictors and a 32-bit arithmetic coder. Beyond the base LLM-plus-arithmetic-coding paradigm, Nacrith introduces several contributions: (1) a CDF precision upgrade from 2^16 to 2^24 that eliminates ~75% of quantization overhead caused by minimum-probability floors in large vocabularies; (2) a token-level N-gram model for fast local predictions; (3) an adaptive log-space bias head correcting per-document LLM errors via online gradient descent; (4) confidence-based LLM skip for accelerating highly predictable tokens; (5) a hybrid binary format (NC06) extending neural compression to arbitrary binary files--to our knowledge a first among LLM-based compressors; (6) a llama.cpp inference backend achieving ~7x faster single-token decode than PyTorch; (7) parallel multi-GPU compression across up to 8 workers; and (8) native KV cache sliding window reducing per-slide cost by ~37x. The system requires only ~500 MB of GGUF weights and ~1.2 GB VRAM per worker, running on consumer GPUs. On alice29.txt (Canterbury Corpus, 152 KB), Nacrith achieves 0.918 bits per byte (bpb)--outperforming gzip by 3.1x, bzip2 by 2.5x, CMIX v21 by 44%, and ts_zip by 20%, while compressing below the 0th-, 1st-, and 2nd-order byte-level Shannon entropy bounds. On enwik8 (100 MB), Nacrith achieves 0.9389 bpb (11.74%), surpassing ts_zip (~1.11 bpb) by 15% and FineZip (1.024 bpb) by 8% despite using a 60x smaller model with no fine-tuning. An out-of-distribution evaluation on a document published after the model's training cutoff confirms these gains are not memorization artifacts, achieving 0.723 bpb on unseen text.
>
---
#### [new 084] How Well Can LLM Agents Simulate End-User Security and Privacy Attitudes and Behaviors?
- **分类: cs.CY; cs.AI; cs.CL; cs.CR**

- **简介: 该论文属于安全与隐私研究任务，旨在评估LLM代理模拟用户安全态度和行为的能力。通过构建基准测试，分析不同模型和策略的表现，发现现有模型仍有提升空间。**

- **链接: [https://arxiv.org/pdf/2602.18464v1](https://arxiv.org/pdf/2602.18464v1)**

> **作者:** Yuxuan Li; Leyang Li; Hao-Ping; Lee; Sauvik Das
>
> **摘要:** A growing body of research assumes that large language model (LLM) agents can serve as proxies for how people form attitudes toward and behave in response to security and privacy (S&P) threats. If correct, these simulations could offer a scalable way to forecast S&P risks in products prior to deployment. We interrogate this assumption using SP-ABCBench, a new benchmark of 30 tests derived from validated S&P human-subject studies, which measures alignment between simulations and human-subjects studies on a 0-100 ascending scale, where higher scores indicate better alignment across three dimensions: Attitude, Behavior, and Coherence. Evaluating twelve LLMs, four persona construction strategies, and two prompting methods, we found that there remains substantial room for improvement: all models score between 50 and 64 on average. Newer, bigger, and smarter models do not reliably do better and sometimes do worse. Some simulation configurations, however, do yield high alignment: e.g., with scores above 95 for some behavior tests when agents are prompted to apply bounded rationality and weigh privacy costs against perceived benefits. We release SP-ABCBench to enable reproducible evaluation as methods improve.
>
---
#### [new 085] Can Large Language Models Replace Human Coders? Introducing ContentBench
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在评估低成本大语言模型是否能替代人类进行解释性编码。通过构建ContentBench基准，测试模型在不同任务上的表现与成本。**

- **链接: [https://arxiv.org/pdf/2602.19467v1](https://arxiv.org/pdf/2602.19467v1)**

> **作者:** Michael Haman
>
> **备注:** Project website: https://contentbench.github.io
>
> **摘要:** Can low-cost large language models (LLMs) take over the interpretive coding work that still anchors much of empirical content analysis? This paper introduces ContentBench, a public benchmark suite that helps answer this replacement question by tracking how much agreement low-cost LLMs achieve and what they cost on the same interpretive coding tasks. The suite uses versioned tracks that invite researchers to contribute new benchmark datasets. I report results from the first track, ContentBench-ResearchTalk v1.0: 1,000 synthetic, social-media-style posts about academic research labeled into five categories spanning praise, critique, sarcasm, questions, and procedural remarks. Reference labels are assigned only when three state-of-the-art reasoning models (GPT-5, Gemini 2.5 Pro, and Claude Opus 4.1) agree unanimously, and all final labels are checked by the author as a quality-control audit. Among the 59 evaluated models, the best low-cost LLMs reach roughly 97-99% agreement with these jury labels, far above GPT-3.5 Turbo, the model behind early ChatGPT and the initial wave of LLM-based text annotation. Several top models can code 50,000 posts for only a few dollars, pushing large-scale interpretive coding from a labor bottleneck toward questions of validation, reporting, and governance. At the same time, small open-weight models that run locally still struggle on sarcasm-heavy items (for example, Llama 3.2 3B reaches only 4% agreement on hard-sarcasm). ContentBench is released with data, documentation, and an interactive quiz at contentbench.github.io to support comparable evaluations over time and to invite community extensions.
>
---
#### [new 086] The Convergence of Schema-Guided Dialogue Systems and the Model Context Protocol
- **分类: cs.AI; cs.CL**

- **简介: 该论文探讨Schema-Guided Dialogue与Model Context Protocol的统一性，解决LLM-agent交互的规范问题，提出五项schema设计原则。**

- **链接: [https://arxiv.org/pdf/2602.18764v1](https://arxiv.org/pdf/2602.18764v1)**

> **作者:** Andreas Schlapbach
>
> **备注:** 18 sections, 4 figures, 7 tables, 38 references. Original research presenting: (1) formal framework mapping Schema-Guided Dialogue principles to Model Context Protocol concepts, (2) five foundational design principles for LLM-native schema authoring, (3) architectural patterns for secure, scalable agent orchestration. Research supported by SBB (Swiss Federal Railways)
>
> **摘要:** This paper establishes a fundamental convergence: Schema-Guided Dialogue (SGD) and the Model Context Protocol (MCP) represent two manifestations of a unified paradigm for deterministic, auditable LLM-agent interaction. SGD, designed for dialogue-based API discovery (2019), and MCP, now the de facto standard for LLM-tool integration, share the same core insight -- that schemas can encode not just tool signatures but operational constraints and reasoning guidance. By analyzing this convergence, we extract five foundational principles for schema design: (1) Semantic Completeness over Syntactic Precision, (2) Explicit Action Boundaries, (3) Failure Mode Documentation, (4) Progressive Disclosure Compatibility, and (5) Inter-Tool Relationship Declaration. These principles reveal three novel insights: first, SGD's original design was fundamentally sound and should be inherited by MCP; second, both frameworks leave failure modes and inter-tool relationships unexploited -- gaps we identify and resolve; third, progressive disclosure emerges as a critical production-scaling insight under real-world token constraints. We provide concrete design patterns for each principle. These principles position schema-driven governance as a scalable mechanism for AI system oversight without requiring proprietary system inspection -- central to Software 3.0.
>
---
#### [new 087] Exploring the Ethical Concerns in User Reviews of Mental Health Apps using Topic Modeling and Sentiment Analysis
- **分类: cs.CY; cs.CL; cs.HC**

- **简介: 该论文属于伦理分析任务，旨在解决AI心理健康应用用户评价中的伦理问题。通过主题建模和情感分析，识别并评估伦理主题，发现新兴伦理挑战。**

- **链接: [https://arxiv.org/pdf/2602.18454v1](https://arxiv.org/pdf/2602.18454v1)**

> **作者:** Mohammad Masudur Rahman; Beenish Moalla Chaudhry
>
> **备注:** 22 pages, journal-ready version
>
> **摘要:** The rapid growth of AI-driven mental health mobile apps has raised concerns about their ethical considerations and user trust. This study proposed a natural language processing (NLP)-based framework to evaluate ethical aspects from user-generated reviews from the Google Play Store and Apple App Store. After gathering and cleaning the data, topic modeling was applied to identify latent themes in the context of ethics using topic words and then map them to well-recognized existing ethical principles described in different ethical frameworks; in addition to that, a bottom-up approach is applied to find any new and emergent ethics from the reviews using a transformer-based zero-shot classification model. Sentiment analysis was then used to capture how users feel about each ethical aspect. The obtained results reveal that well-known ethical considerations are not enough for the modern AI-based technologies and are missing emerging ethical challenges, showing how these apps either uphold or overlook key moral values. This work contributes to developing an ongoing evaluation system that can enhance the fairness, transparency, and trustworthiness of AI-powered mental health chatbots.
>
---
#### [new 088] Hierarchical Reward Design from Language: Enhancing Alignment of Agent Behavior with Human Specifications
- **分类: cs.AI; cs.CL; cs.HC; cs.LG**

- **简介: 该论文属于人工智能领域，旨在解决AI代理行为与人类规范对齐的问题。提出HRDL和L2HR方法，增强复杂任务中奖励设计的准确性与人性化。**

- **链接: [https://arxiv.org/pdf/2602.18582v1](https://arxiv.org/pdf/2602.18582v1)**

> **作者:** Zhiqin Qian; Ryan Diaz; Sangwon Seo; Vaibhav Unhelkar
>
> **备注:** Extended version of an identically-titled paper accepted at AAMAS 2026
>
> **摘要:** When training artificial intelligence (AI) to perform tasks, humans often care not only about whether a task is completed but also how it is performed. As AI agents tackle increasingly complex tasks, aligning their behavior with human-provided specifications becomes critical for responsible AI deployment. Reward design provides a direct channel for such alignment by translating human expectations into reward functions that guide reinforcement learning (RL). However, existing methods are often too limited to capture nuanced human preferences that arise in long-horizon tasks. Hence, we introduce Hierarchical Reward Design from Language (HRDL): a problem formulation that extends classical reward design to encode richer behavioral specifications for hierarchical RL agents. We further propose Language to Hierarchical Rewards (L2HR) as a solution to HRDL. Experiments show that AI agents trained with rewards designed via L2HR not only complete tasks effectively but also better adhere to human specifications. Together, HRDL and L2HR advance the research on human-aligned AI agents.
>
---
#### [new 089] PuppetChat: Fostering Intimate Communication through Bidirectional Actions and Micronarratives
- **分类: cs.HC; cs.AI; cs.CL; cs.CY**

- **简介: 该论文提出PuppetChat，解决即时通讯中情感表达不足的问题，通过双向互动和个性化微叙事增强亲密关系。属于人机交互任务。**

- **链接: [https://arxiv.org/pdf/2602.19463v1](https://arxiv.org/pdf/2602.19463v1)**

> **作者:** Emma Jiren Wang; Siying Hu; Zhicong Lu
>
> **备注:** 19 pages, 8 figures; Accepted by ACM CHI 2026. In Proceedings of the 2024 CHI Conference on Human Factors in Computing Systems (CHI'24)
>
> **摘要:** As a primary channel for sustaining modern intimate relationships, instant messaging facilitates frequent connection across distances. However, today's tools often dilute care; they favor single tap reactions and vague emojis that do not support two way action responses, do not preserve the feeling that the exchange keeps going without breaking, and are weakly tied to who we are and what we share. To address this challenge, we present PuppetChat, a dyadic messaging prototype that restores this expressive depth through embodied interaction. PuppetChat uses a reciprocity aware recommender to encourage responsive actions and generates personalized micronarratives from user stories to ground interactions in personal history. Our 10-day field study with 11 dyads of close partners or friends revealed that this approach enhanced social presence, supported more expressive self disclosure, and sustained continuity and shared memories.
>
---
#### [new 090] Vibe Coding on Trial: Operating Characteristics of Unanimous LLM Juries
- **分类: cs.DB; cs.AI; cs.CL; cs.SE**

- **简介: 该论文研究LLM代码生成的安全性评估问题，通过构建一致的LLM委员会来提高代码审查的可靠性。**

- **链接: [https://arxiv.org/pdf/2602.18492v1](https://arxiv.org/pdf/2602.18492v1)**

> **作者:** Muhammad Aziz Ullah; Abdul Serwadda
>
> **备注:** Submitted to IEEE International Conference on Semantic Computing 2026
>
> **摘要:** Large Language Models (LLMs) are now good enough at coding that developers can describe intent in plain language and let the tool produce the first code draft, a workflow increasingly built into tools like GitHub Copilot, Cursor, and Replit. What is missing is a reliable way to tell which model written queries are safe to accept without sending everything to a human. We study the application of an LLM jury to run this review step. We first benchmark 15 open models on 82 MySQL text to SQL tasks using an execution grounded protocol to get a clean baseline of which models are strong. From the six best models we build unanimous committees of sizes 1 through 6 that see the prompt, schema, and candidate SQL and accept it only when every member says it is correct. This rule matches safety first deployments where false accepts are more costly than false rejects. We measure true positive rate, false positive rate and Youden J and we also look at committees per generator. Our results show that single model judges are uneven, that small unanimous committees of strong models can cut false accepts while still passing many good queries, and that the exact committee composition matters significantly.
>
---
#### [new 091] NILE: Formalizing Natural-Language Descriptions of Formal Languages
- **分类: cs.FL; cs.CL; cs.LO**

- **简介: 该论文属于自然语言与形式语言互译任务，旨在解决教育场景中自然语言描述与形式语言不一致的问题，提出Nile语言实现语法对齐，便于自动解释差异。**

- **链接: [https://arxiv.org/pdf/2602.19743v1](https://arxiv.org/pdf/2602.19743v1)**

> **作者:** Tristan Kneisel; Marko Schmellenkamp; Fabian Vehlken; Thomas Zeume
>
> **摘要:** This paper explores how natural-language descriptions of formal languages can be compared to their formal representations and how semantic differences can be explained. This is motivated from educational scenarios where learners describe a formal language (presented, e.g., by a finite state automaton, regular expression, pushdown automaton, context-free grammar or in set notation) in natural language, and an educational support system has to (1) judge whether the natural-language description accurately describes the formal language, and to (2) provide explanations why descriptions are not accurate. To address this question, we introduce a representation language for formal languages, Nile, which is designed so that Nile expressions can mirror the syntactic structure of natural-language descriptions of formal languages. Nile is sufficiently expressive to cover a broad variety of formal languages, including all regular languages and fragments of context-free languages typically used in educational contexts. Generating Nile expressions that are syntactically close to natural-language descriptions then allows to provide explanations for inaccuracies in the descriptions algorithmically. In experiments on an educational data set, we show that LLMs can translate natural-language descriptions into equivalent, syntactically close Nile expressions with high accuracy - allowing to algorithmically provide explanations for incorrect natural-language descriptions. Our experiments also show that while natural-language descriptions can also be translated into regular expressions (but not context-free grammars), the expressions are often not syntactically close and thus not suitable for providing explanations.
>
---
#### [new 092] Diagnosing LLM Reranker Behavior Under Fixed Evidence Pools
- **分类: cs.LG; cs.CL; cs.IR**

- **简介: 该论文属于信息检索任务，旨在解决 reranker 行为分析问题。通过固定证据池，隔离排名策略影响，对比 LLM 与传统方法的差异。**

- **链接: [https://arxiv.org/pdf/2602.18613v1](https://arxiv.org/pdf/2602.18613v1)**

> **作者:** Baris Arat; Emre Sefer
>
> **摘要:** Standard reranking evaluations study how a reranker orders candidates returned by an upstream retriever. This setup couples ranking behavior with retrieval quality, so differences in output cannot be attributed to the ranking policy alone. We introduce a controlled diagnostic that isolates reranking by using Multi-News clusters as fixed evidence pools. We limit each pool to exactly eight documents and pass identical inputs to all rankers. Within this setup, BM25 and MMR serve as interpretable reference points for lexical matching and diversity optimization. Across 345 clusters, we find that redundancy patterns vary by model: one LLM implicitly diversifies at larger selection budgets, while another increases redundancy. In contrast, LLMs underperform on lexical coverage at small selection budgets. As a result, LLM rankings diverge substantially from both baselines rather than consistently approximating either strategy. By eliminating retrieval variance, we can attribute these differences directly to the ranking policy. This diagnostic is model-agnostic and applicable to any ranker, including open source systems and proprietary APIs.
>
---
#### [new 093] From "Help" to Helpful: A Hierarchical Assessment of LLMs in Mental e-Health Applications
- **分类: cs.HC; cs.AI; cs.CL; cs.CY**

- **简介: 该论文研究LLMs在心理在线辅导中生成邮件主题的任务，解决主题通用导致优先级低的问题，通过层级评估比较模型性能。**

- **链接: [https://arxiv.org/pdf/2602.18443v1](https://arxiv.org/pdf/2602.18443v1)**

> **作者:** Philipp Steigerwald; Jens Albrecht
>
> **摘要:** Psychosocial online counselling frequently encounters generic subject lines that impede efficient case prioritisation. This study evaluates eleven large language models generating six-word subject lines for German counselling emails through hierarchical assessment - first categorising outputs, then ranking within categories to enable manageable evaluation. Nine assessors (counselling professionals and AI systems) enable analysis via Krippendorff's $α$, Spearman's $ρ$, Pearson's $r$ and Kendall's $τ$. Results reveal performance trade-offs between proprietary services and privacy-preserving open-source alternatives, with German fine-tuning consistently improving performance. The study addresses critical ethical considerations for mental health AI deployment including privacy, bias and accountability.
>
---
#### [new 094] Reasoning Capabilities of Large Language Models. Lessons Learned from General Game Playing
- **分类: cs.AI; cs.CL; cs.LO**

- **简介: 该论文属于自然语言处理任务，研究大语言模型的推理能力，通过通用游戏博弈评估其在形式化环境中的表现，分析模型在不同游戏结构下的性能与错误类型。**

- **链接: [https://arxiv.org/pdf/2602.19160v1](https://arxiv.org/pdf/2602.19160v1)**

> **作者:** Maciej Świechowski; Adam Żychowski; Jacek Mańdziuk
>
> **摘要:** This paper examines the reasoning capabilities of Large Language Models (LLMs) from a novel perspective, focusing on their ability to operate within formally specified, rule-governed environments. We evaluate four LLMs (Gemini 2.5 Pro and Flash variants, Llama 3.3 70B and GPT-OSS 120B) on a suite of forward-simulation tasks-including next / multistep state formulation, and legal action generation-across a diverse set of reasoning problems illustrated through General Game Playing (GGP) game instances. Beyond reporting instance-level performance, we characterize games based on 40 structural features and analyze correlations between these features and LLM performance. Furthermore, we investigate the effects of various game obfuscations to assess the role of linguistic semantics in game definitions and the impact of potential prior exposure of LLMs to specific games during training. The main results indicate that three of the evaluated models generally perform well across most experimental settings, with performance degradation observed as the evaluation horizon increases (i.e., with a higher number of game steps). Detailed case-based analysis of the LLM performance provides novel insights into common reasoning errors in the considered logic-based problem formulation, including hallucinated rules, redundant state facts, or syntactic errors. Overall, the paper reports clear progress in formal reasoning capabilities of contemporary models.
>
---
#### [new 095] AAVGen: Precision Engineering of Adeno-associated Viral Capsids for Renal Selective Targeting
- **分类: q-bio.QM; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于基因治疗领域，旨在解决AAV载体在肾脏靶向中的优化问题。通过AI框架AAVGen设计新型衣壳蛋白，提升多属性表现。**

- **链接: [https://arxiv.org/pdf/2602.18915v1](https://arxiv.org/pdf/2602.18915v1)**

> **作者:** Mohammadreza Ghaffarzadeh-Esfahani; Yousof Gheisari
>
> **备注:** 22 pages, 6 figures, and 5 supplementary files. Corresponding author: ygheisari@med.mui.ac.ir, Kaggle notebook is available at https://www.kaggle.com/code/mohammadgh009/aavgen
>
> **摘要:** Adeno-associated viruses (AAVs) are promising vectors for gene therapy, but their native serotypes face limitations in tissue tropism, immune evasion, and production efficiency. Engineering capsids to overcome these hurdles is challenging due to the vast sequence space and the difficulty of simultaneously optimizing multiple functional properties. The complexity also adds when it comes to the kidney, which presents unique anatomical barriers and cellular targets that require precise and efficient vector engineering. Here, we present AAVGen, a generative artificial intelligence framework for de novo design of AAV capsids with enhanced multi-trait profiles. AAVGen integrates a protein language model (PLM) with supervised fine-tuning (SFT) and a reinforcement learning technique termed Group Sequence Policy Optimization (GSPO). The model is guided by a composite reward signal derived from three ESM-2-based regression predictors, each trained to predict a key property: production fitness, kidney tropism, and thermostability. Our results demonstrate that AAVGen produces a diverse library of novel VP1 protein sequences. In silico validations revealed that the majority of the generated variants have superior performance across all three employed indices, indicating successful multi-objective optimization. Furthermore, structural analysis via AlphaFold3 confirms that the generated sequences preserve the canonical capsid folding despite sequence diversification. AAVGen establishes a foundation for data-driven viral vector engineering, accelerating the development of next-generation AAV vectors with tailored functional characteristics.
>
---
#### [new 096] Watermarking LLM Agent Trajectories
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于数据版权保护任务，旨在解决LLM代理轨迹数据的盗用问题。提出ActHook方法，在不改变任务结果的前提下嵌入水印，实现轨迹数据的可追溯与检测。**

- **链接: [https://arxiv.org/pdf/2602.18700v1](https://arxiv.org/pdf/2602.18700v1)**

> **作者:** Wenlong Meng; Chen Gong; Terry Yue Zhuo; Fan Zhang; Kecen Li; Zheng Liu; Zhou Yang; Chengkun Wei; Wenzhi Chen
>
> **备注:** 20 pages, 9 figures
>
> **摘要:** LLM agents rely heavily on high-quality trajectory data to guide their problem-solving behaviors, yet producing such data requires substantial task design, high-capacity model generation, and manual filtering. Despite the high cost of creating these datasets, existing literature has overlooked copyright protection for LLM agent trajectories. This gap leaves creators vulnerable to data theft and makes it difficult to trace misuse or enforce ownership rights. This paper introduces ActHook, the first watermarking method tailored for agent trajectory datasets. Inspired by hook mechanisms in software engineering, ActHook embeds hook actions that are activated by a secret input key and do not alter the original task outcome. Like software execution, LLM agents operate sequentially, allowing hook actions to be inserted at decision points without disrupting task flow. When the activation key is present, an LLM agent trained on watermarked trajectories can produce these hook actions at a significantly higher rate, enabling reliable black-box detection. Experiments on mathematical reasoning, web searching, and software engineering agents show that ActHook achieves an average detection AUC of 94.3 on Qwen-2.5-Coder-7B while incurring negligible performance degradation.
>
---
#### [new 097] [b]=[d]-[t]+[p]: Self-supervised Speech Models Discover Phonological Vector Arithmetic
- **分类: eess.AS; cs.CL; cs.LG; cs.SD**

- **简介: 该论文研究自监督语音模型的表征结构，解决如何理解其编码的语音信息问题。通过分析96种语言，发现模型使用可解释的音素向量进行运算，实现音素向量算术。**

- **链接: [https://arxiv.org/pdf/2602.18899v1](https://arxiv.org/pdf/2602.18899v1)**

> **作者:** Kwanghee Choi; Eunjung Yeo; Cheol Jun Cho; David Harwath; David R. Mortensen
>
> **备注:** Submitted to ACL, code planned to release after acceptance
>
> **摘要:** Self-supervised speech models (S3Ms) are known to encode rich phonetic information, yet how this information is structured remains underexplored. We conduct a comprehensive study across 96 languages to analyze the underlying structure of S3M representations, with particular attention to phonological vectors. We first show that there exist linear directions within the model's representation space that correspond to phonological features. We further demonstrate that the scale of these phonological vectors correlate to the degree of acoustic realization of their corresponding phonological features in a continuous manner. For example, the difference between [d] and [t] yields a voicing vector: adding this vector to [p] produces [b], while scaling it results in a continuum of voicing. Together, these findings indicate that S3Ms encode speech using phonologically interpretable and compositional vectors, demonstrating phonological vector arithmetic. All code and interactive demos are available at https://github.com/juice500ml/phonetic-arithmetic .
>
---
#### [new 098] AdaEvolve: Adaptive LLM Driven Zeroth-Order Optimization
- **分类: cs.NE; cs.AI; cs.CL**

- **简介: 该论文提出AdaEvolve，解决LLM驱动进化中的资源分配问题，通过自适应优化提升搜索效率。属于优化任务，旨在提高自动化程序生成效果。**

- **链接: [https://arxiv.org/pdf/2602.20133v1](https://arxiv.org/pdf/2602.20133v1)**

> **作者:** Mert Cemri; Shubham Agrawal; Akshat Gupta; Shu Liu; Audrey Cheng; Qiuyang Mang; Ashwin Naren; Lutfi Eren Erdogan; Koushik Sen; Matei Zaharia; Alex Dimakis; Ion Stoica
>
> **摘要:** The paradigm of automated program generation is shifting from one-shot generation to inference-time search, where Large Language Models (LLMs) function as semantic mutation operators within evolutionary loops. While effective, these systems are currently governed by static schedules that fail to account for the non-stationary dynamics of the search process. This rigidity results in substantial computational waste, as resources are indiscriminately allocated to stagnating populations while promising frontiers remain under-exploited. We introduce AdaEvolve, a framework that reformulates LLM-driven evolution as a hierarchical adaptive optimization problem. AdaEvolve uses an "accumulated improvement signal" to unify decisions across three levels: Local Adaptation, which dynamically modulates the exploration intensity within a population of solution candidates; Global Adaptation, which routes the global resource budget via bandit-based scheduling across different solution candidate populations; and Meta-Guidance which generates novel solution tactics based on the previously generated solutions and their corresponding improvements when the progress stalls. We demonstrate that AdaEvolve consistently outperforms the open-sourced baselines across 185 different open-ended optimization problems including combinatorial, systems optimization and algorithm design problems.
>
---
#### [new 099] Learning to Detect Language Model Training Data via Active Reconstruction
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于语言模型训练数据检测任务，解决如何通过主动重构识别训练数据的问题。提出ADRA方法，利用强化学习提升检测效果。**

- **链接: [https://arxiv.org/pdf/2602.19020v1](https://arxiv.org/pdf/2602.19020v1)**

> **作者:** Junjie Oscar Yin; John X. Morris; Vitaly Shmatikov; Sewon Min; Hannaneh Hajishirzi
>
> **摘要:** Detecting LLM training data is generally framed as a membership inference attack (MIA) problem. However, conventional MIAs operate passively on fixed model weights, using log-likelihoods or text generations. In this work, we introduce \textbf{Active Data Reconstruction Attack} (ADRA), a family of MIA that actively induces a model to reconstruct a given text through training. We hypothesize that training data are \textit{more reconstructible} than non-members, and the difference in their reconstructibility can be exploited for membership inference. Motivated by findings that reinforcement learning (RL) sharpens behaviors already encoded in weights, we leverage on-policy RL to actively elicit data reconstruction by finetuning a policy initialized from the target model. To effectively use RL for MIA, we design reconstruction metrics and contrastive rewards. The resulting algorithms, \textsc{ADRA} and its adaptive variant \textsc{ADRA+}, improve both reconstruction and detection given a pool of candidate data. Experiments show that our methods consistently outperform existing MIAs in detecting pre-training, post-training, and distillation data, with an average improvement of 10.7\% over the previous runner-up. In particular, \MethodPlus~improves over Min-K\%++ by 18.8\% on BookMIA for pre-training detection and by 7.6\% on AIME for post-training detection.
>
---
#### [new 100] The Algorithmic Unconscious: Structural Mechanisms and Implicit Biases in Large Language Models
- **分类: cs.CY; cs.CL**

- **简介: 该论文属于AI偏见研究任务，探讨LLM中的结构性偏差问题。通过分析tokenization等机制，揭示算法无意识带来的隐性偏见，并提出技术审计框架以改进AI系统。**

- **链接: [https://arxiv.org/pdf/2602.18468v1](https://arxiv.org/pdf/2602.18468v1)**

> **作者:** Philippe Boisnard
>
> **备注:** 18 pages, 5 figures, Extended version of a paper presented at the international conference 'Artificial Intelligence and Transformations of Information' (LOGOS/FLSH, Hassan II University of Casablanca, Morocco, December 2025), accepted for publication in LOGOS after double-blind peer review
>
> **摘要:** This article introduces the concept of the algorithmic unconscious to designate the set of structural determinations that operate within large language models (LLMs) without being accessible either to the model's own reflexivity or to that of its users. In contrast to approaches that reduce AI bias solely to dataset composition or to the projection of human intentionality, we argue that a significant class of biases emerges directly from the technical mechanisms of the models themselves: tokenization, attention, statistical optimization, and alignment procedures. By framing bias as an infrastructural phenomenon, this approach resolves a central theoretical ambiguity surrounding responsibility, neutrality, and correction in contemporary LLMs. Based on a comparative analysis of tokenization across a corpus of parallel sentences, we show that Arabic languages (Modern Standard Arabic and Maghrebi dialects) undergo a systematic inflation in token count relative to English, with ratios ranging from 1.6x to nearly 4x depending on the infrastructure (OpenAI, Anthropic, SentencePiece/Mistral). This over-segmentation constitutes a measurable infrastructural bias that mechanically increases inference costs, constrains access to contextual space, and alters attentional weighting within model representations. We relate these empirical findings to three additional structural mechanisms: causal bias (correlation vs causation), the erasure of minoritized features through dimensional collapse, and normative biases induced by safety alignment. Finally, we propose a framework for a technical clinic of models, grounded in the audit of tokenization regimes, latent space topology, and alignment systems, as a necessary condition for the critical appropriation of AI infrastructures.
>
---
#### [new 101] SenTSR-Bench: Thinking with Injected Knowledge for Time-Series Reasoning
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **简介: 该论文属于时间序列推理任务，解决GRLM缺乏领域知识与TSLM泛化能力不足的问题，通过知识注入和强化学习方法提升时间序列诊断性能。**

- **链接: [https://arxiv.org/pdf/2602.19455v1](https://arxiv.org/pdf/2602.19455v1)**

> **作者:** Zelin He; Boran Han; Xiyuan Zhang; Shuai Zhang; Haotian Lin; Qi Zhu; Haoyang Fang; Danielle C. Maddix; Abdul Fatir Ansari; Akash Chandrayan; Abhinav Pradhan; Bernie Wang; Matthew Reimherr
>
> **备注:** Accepted by the 29th International Conference on Artificial Intelligence and Statistics (AISTATS 2026)
>
> **摘要:** Time-series diagnostic reasoning is essential for many applications, yet existing solutions face a persistent gap: general reasoning large language models (GRLMs) possess strong reasoning skills but lack the domain-specific knowledge to understand complex time-series patterns. Conversely, fine-tuned time-series LLMs (TSLMs) understand these patterns but lack the capacity to generalize reasoning for more complicated questions. To bridge this gap, we propose a hybrid knowledge-injection framework that injects TSLM-generated insights directly into GRLM's reasoning trace, thereby achieving strong time-series reasoning with in-domain knowledge. As collecting data for knowledge injection fine-tuning is costly, we further leverage a reinforcement learning-based approach with verifiable rewards (RLVR) to elicit knowledge-rich traces without human supervision, then transfer such an in-domain thinking trace into GRLM for efficient knowledge injection. We further release SenTSR-Bench, a multivariate time-series-based diagnostic reasoning benchmark collected from real-world industrial operations. Across SenTSR-Bench and other public datasets, our method consistently surpasses TSLMs by 9.1%-26.1% and GRLMs by 7.9%-22.4%, delivering robust, context-aware time-series diagnostic insights.
>
---
## 更新

#### [replaced 001] RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Reasoning Intervention in Large Reasoning Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于AI可信性研究，旨在解决大模型推理过程不忠实的问题。通过构建基准测试，评估模型推理的逻辑一致性与因果影响，揭示模型在数学、代码等领域的不忠实现象。**

- **链接: [https://arxiv.org/pdf/2602.17053v3](https://arxiv.org/pdf/2602.17053v3)**

> **作者:** Yunseok Han; Yejoon Lee; Jaeyoung Do
>
> **备注:** Accepted in ICLR 2026 Poster: https://iclr.cc/virtual/2026/poster/10011763
>
> **摘要:** Large Reasoning Models (LRMs) exhibit strong performance, yet often produce rationales that sound plausible but fail to reflect their true decision process, undermining reliability and trust. We introduce a formal framework for reasoning faithfulness, defined by two testable conditions: stance consistency (a coherent stance linking reasoning to answer) and causal influence (the stated reasoning causally drives the answer under output-level interventions), explicitly decoupled from accuracy. To operationalize this, we present RFEval, a benchmark of 7,186 instances across seven tasks that probes faithfulness via controlled, output-level counterfactual interventions. Evaluating twelve open-source LRMs, we find unfaithfulness in 49.7% of outputs, predominantly from stance inconsistency. Failures are concentrated in brittle, convergent domains such as math and code, and correlate more with post-training regimes than with scale: within-family ablations indicate that adding current RL-style objectives on top of supervised fine-tuning can reduce reasoning faithfulness, even when accuracy is maintained. Crucially, accuracy is neither a sufficient nor a reliable proxy for faithfulness: once controlling for model and task, the accuracy-faithfulness link is weak and statistically insignificant. Our work establishes a rigorous methodology for auditing LRM reliability and shows that trustworthy AI requires optimizing not only for correct outcomes but also for the structural integrity of the reasoning process. Our code and dataset can be found at project page: https://aidaslab.github.io/RFEval/
>
---
#### [replaced 002] CORE: Measuring Multi-Agent LLM Interaction Quality under Game-Theoretic Pressures
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在量化多智能体LLM在博弈论情境下的对话质量。提出CORE指标，解决语言多样性评估问题。**

- **链接: [https://arxiv.org/pdf/2508.11915v2](https://arxiv.org/pdf/2508.11915v2)**

> **作者:** Punya Syon Pandey; Yongjin Yang; Jiarui Liu; Zhijing Jin
>
> **备注:** EACL 2026 (Main)
>
> **摘要:** Game-theoretic interactions between agents with Large Language Models (LLMs) have revealed many emergent capabilities, yet the linguistic diversity of these interactions has not been sufficiently quantified. In this paper, we present the Conversational Robustness Evaluation Score: CORE, a metric to quantify the effectiveness of language use within multi-agent systems across different game-theoretic interactions. CORE integrates measures of cluster entropy, lexical repetition, and semantic similarity, providing a direct lens of dialog quality. We apply CORE to pairwise LLM dialogs across competitive, cooperative, and neutral settings, further grounding our analysis in Zipf's and Heaps' Laws to characterize word frequency distributions and vocabulary growth. Our findings show that cooperative settings exhibit both steeper Zipf distributions and higher Heap exponents, indicating more repetition alongside greater vocabulary expansion. In contrast, competitive interactions display lower Zipf and Heaps exponents, reflecting less repetition and more constrained vocabularies. These results provide new insights into how social incentives influence language adaptation, and highlight CORE as a robust diagnostic for measuring linguistic robustness in multi-agent LLM systems. Our code is available at https://github.com/psyonp/core.
>
---
#### [replaced 003] Beyond Single-Turn: A Survey on Multi-Turn Interactions with Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多轮对话任务，旨在解决长对话中上下文保持、连贯性等问题。综述了多轮交互的评估与提升方法，涵盖模型优化、外部集成和协作机制。**

- **链接: [https://arxiv.org/pdf/2504.04717v5](https://arxiv.org/pdf/2504.04717v5)**

> **作者:** Yubo Li; Xiaobin Shen; Xinyu Yao; Xueying Ding; Yidi Miao; Ramayya Krishnan; Rema Padman
>
> **摘要:** Recent advances in large language models (LLMs) have substantially improved single-turn task performance, yet real-world applications increasingly demand sophisticated multi-turn interactions. This survey provides a comprehensive review of recent progress in evaluating and enhancing multi-turn LLM interactions. Centered on a task-oriented taxonomy-spanning instruction following in domains such as mathematics and coding, and conversational engagement in role-playing, healthcare, education, and adversarial jailbreak settings-we systematically examine the challenges of maintaining context, coherence, fairness, and responsiveness across prolonged dialogues. We organize existing benchmarks and datasets into coherent categories reflecting the evolving landscape of multi-turn dialogue evaluation, and review a broad spectrum of enhancement methodologies, including model-centric strategies (in-context learning, supervised fine-tuning, reinforcement learning, and architectural innovations), external integration approaches (memory augmentation, retrieval-based methods, and knowledge graphs), and agent-based techniques for collaborative interaction. Finally, we identify open challenges and promising directions for future research to further improve the robustness and effectiveness of multi-turn LLM interactions.
>
---
#### [replaced 004] Error-Aware Knowledge Distillation via Targeted Revision for Customer-Service Summarization
- **分类: cs.CL**

- **简介: 该论文针对客户客服摘要任务，提出ARF管道，通过分析与修正错误，提升小模型性能，解决大模型依赖问题。**

- **链接: [https://arxiv.org/pdf/2511.03005v2](https://arxiv.org/pdf/2511.03005v2)**

> **作者:** Hee-Jin Lee; Zhen Guo; Luchao Jin; Morteza Moazami Goudarzi
>
> **摘要:** We introduce an Analyze-Revise-Finetune (ARF) pipeline that enables smaller open-source language models (LLMs) to surpass substantially larger proprietary models in customer service summarization tasks. The pipeline first analyzes and categorizes common errors in summaries produced by a teacher model (GPT-3.5), then performs a targeted revision using a compact editor model (Llama 3.1 70B) to generate high-quality, refined training data. Fine-tuning smaller student models (e.g., Llama 3.1 8B, QWen3 4B) on this refined data resulted in superior summarization performance compared to GPT-3.5. The ARF pipeline improves cost efficiency and data privacy while maintaining competitive accuracy, illustrating a generalizable framework for enhancing open-source LLMs across diverse downstream applications.
>
---
#### [replaced 005] PsihoRo: Depression and Anxiety Romanian Text Corpus
- **分类: cs.CL**

- **简介: 该论文属于心理NLP任务，旨在解决罗马尼亚语心理健康语料缺失的问题。通过收集205份问卷数据，构建了首个罗马尼亚语抑郁与焦虑语料库PsihoRo，并进行文本分析与情感检测。**

- **链接: [https://arxiv.org/pdf/2602.18324v2](https://arxiv.org/pdf/2602.18324v2)**

> **作者:** Alexandra Ciobotaru; Ana-Maria Bucur; Liviu P. Dinu
>
> **备注:** This article was accepted at LREC 2026
>
> **摘要:** Psychological corpora in NLP are collections of texts used to analyze human psychology, emotions, and mental health. These texts allow researchers to study psychological constructs, detect mental health issues and analyze emotional language. However, mental health data can be difficult to collect correctly from social media, due to suppositions made by the collectors. A more pragmatic strategy involves gathering data through open-ended questions and then assessing this information with self-report screening surveys. This method was employed successfully for English, a language with a lot of psychological NLP resources. However, this cannot be stated for Romanian, which currently has no open-source mental health corpus. To address this gap, we have created the first corpus for depression and anxiety in Romanian, by utilizing a form with 6 open-ended questions along with the standardized PHQ-9 and GAD-7 screening questionnaires. Consisting of the texts of 205 respondents and although it may seem small, PsihoRo is a first step towards understanding and analyzing texts regarding the mental health of the Romanian population. We employ statistical analysis, text analysis using Romanian LIWC, emotion detection and topic modeling to show what are the most important features of this newly introduced resource to the NLP community.
>
---
#### [replaced 006] EuroGEST: Investigating gender stereotypes in multilingual language models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的公平性研究任务，旨在解决多语言大模型中的性别刻板印象问题。作者构建了EuroGEST数据集，评估24个模型的性别偏见，发现女性常被关联为“美丽”“体贴”，男性为“领导”“专业”。**

- **链接: [https://arxiv.org/pdf/2506.03867v3](https://arxiv.org/pdf/2506.03867v3)**

> **作者:** Jacqueline Rowe; Mateusz Klimaszewski; Liane Guillou; Shannon Vallor; Alexandra Birch
>
> **备注:** In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 32074-32096, Suzhou, China. Association for Computational Linguistics. 9 pages, 5 figures, 1 table
>
> **摘要:** Large language models increasingly support multiple languages, yet most benchmarks for gender bias remain English-centric. We introduce EuroGEST, a dataset designed to measure gender-stereotypical reasoning in LLMs across English and 29 European languages. EuroGEST builds on an existing expert-informed benchmark covering 16 gender stereotypes, expanded in this work using translation tools, quality estimation metrics, and morphological heuristics. Human evaluations confirm that our data generation method results in high accuracy of both translations and gender labels across languages. We use EuroGEST to evaluate 24 multilingual language models from six model families, demonstrating that the strongest stereotypes in all models across all languages are that women are 'beautiful', 'empathetic' and 'neat' and men are 'leaders', 'strong, tough' and 'professional'. We also show that larger models encode gendered stereotypes more strongly and that instruction finetuning does not consistently reduce gendered stereotypes. Our work highlights the need for more multilingual studies of fairness in LLMs and offers scalable methods and resources to audit gender bias across languages.
>
---
#### [replaced 007] Personalized Help for Optimizing Low-Skilled Users' Strategy
- **分类: cs.CL**

- **简介: 该论文属于人机协作任务，旨在提升低技能用户在游戏中的表现。通过生成策略建议，帮助新手与高手竞争，甚至超越他们。**

- **链接: [https://arxiv.org/pdf/2411.09109v4](https://arxiv.org/pdf/2411.09109v4)**

> **作者:** Feng Gu; Wichayaporn Wongkamjan; Jonathan K. Kummerfeld; Denis Peskoff; Jonathan May; Jordan Boyd-Graber
>
> **备注:** 9 pages, 3 figures
>
> **摘要:** AIs can beat humans in game environments; however, how helpful those agents are to human remains understudied. We augment CICERO, a natural language agent that demonstrates superhuman performance in Diplomacy, to generate both move and message advice based on player intentions. A dozen Diplomacy games with novice and experienced players, with varying advice settings, show that some of the generated advice is beneficial. It helps novices compete with experienced players and in some instances even surpass them. The mere presence of advice can be advantageous, even if players do not follow it.
>
---
#### [replaced 008] BETA-Labeling for Multilingual Dataset Construction in Low-Resource IR
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于低资源信息检索任务，旨在解决数据稀缺问题。通过BETA-标签框架和大语言模型构建多语言数据集，并探讨跨语言数据重用的可行性与限制。**

- **链接: [https://arxiv.org/pdf/2602.14488v2](https://arxiv.org/pdf/2602.14488v2)**

> **作者:** Md. Najib Hasan; Mst. Jannatun Ferdous Rain; Fyad Mohammed; Nazmul Siddique
>
> **备注:** This work was submitted without the consent of my current adviser. Additionally, it overlaps with my unpublished research work. In order to avoid potential academic and authorship conflicts, I am requesting withdrawal of the paper
>
> **摘要:** IR in low-resource languages remains limited by the scarcity of high-quality, task-specific annotated datasets. Manual annotation is expensive and difficult to scale, while using large language models (LLMs) as automated annotators introduces concerns about label reliability, bias, and evaluation validity. This work presents a Bangla IR dataset constructed using a BETA-labeling framework involving multiple LLM annotators from diverse model families. The framework incorporates contextual alignment, consistency checks, and majority agreement, followed by human evaluation to verify label quality. Beyond dataset creation, we examine whether IR datasets from other low-resource languages can be effectively reused through one-hop machine translation. Using LLM-based translation across multiple language pairs, we experimented on meaning preservation and task validity between source and translated datasets. Our experiment reveal substantial variation across languages, reflecting language-dependent biases and inconsistent semantic preservation that directly affect the reliability of cross-lingual dataset reuse. Overall, this study highlights both the potential and limitations of LLM-assisted dataset creation for low-resource IR. It provides empirical evidence of the risks associated with cross-lingual dataset reuse and offers practical guidance for constructing more reliable benchmarks and evaluation pipelines in low-resource language settings.
>
---
#### [replaced 009] Bayesian Attention Mechanism: A Probabilistic Framework for Positional Encoding and Context Length Extrapolation
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，解决位置编码与上下文长度外推问题。提出贝叶斯注意力机制（BAM），通过概率框架提升长文本泛化能力。**

- **链接: [https://arxiv.org/pdf/2505.22842v3](https://arxiv.org/pdf/2505.22842v3)**

> **作者:** Arthur S. Bianchessi; Yasmin C. Aguirre; Rodrigo C. Barros; Lucas S. Kupssinskü
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Transformer-based language models rely on positional encoding (PE) to handle token order and support context length extrapolation. However, existing PE methods lack theoretical clarity and rely on limited evaluation metrics to substantiate their extrapolation claims. We propose the Bayesian Attention Mechanism (BAM), a theoretical framework that formulates positional encoding as a prior within a probabilistic model. BAM unifies existing methods (e.g., NoPE and ALiBi) and motivates a new Generalized Gaussian positional prior that substantially improves long-context generalization. Empirically, BAM enables accurate information retrieval at $500\times$ the training context length, outperforming previous state-of-the-art context length generalization in long context retrieval accuracy while maintaining comparable perplexity and introducing minimal additional parameters.
>
---
#### [replaced 010] CogniAlign: Survivability-Grounded Multi-Agent Moral Reasoning for Safe and Transparent AI
- **分类: cs.CY; cs.CL**

- **简介: 该论文属于AI对齐任务，旨在解决AI与人类价值观不一致的问题。提出CogniAlign框架，通过多智能体协作进行基于生存性的道德推理，提升AI的透明度和安全性。**

- **链接: [https://arxiv.org/pdf/2509.13356v2](https://arxiv.org/pdf/2509.13356v2)**

> **作者:** Hasin Jawad Ali; Ilhamul Azam; Ajwad Abrar; Md. Kamrul Hasan; Hasan Mahmud
>
> **备注:** Under Review
>
> **摘要:** The challenge of aligning artificial intelligence (AI) with human values persists due to the abstract and often conflicting nature of moral principles and the opacity of existing approaches. This paper introduces CogniAlign, a multi-agent deliberation framework based on naturalistic moral realism, that grounds moral reasoning in survivability, defined across individual and collective dimensions, and operationalizes it through structured deliberations among discipline-specific scientist agents. Each agent, representing neuroscience, psychology, sociology, and evolutionary biology, provides arguments and rebuttals that are synthesized by an arbiter into transparent and empirically anchored judgments. As a proof-of-concept study, we evaluate CogniAlign on classic and novel moral questions and compare its outputs against GPT-4o using a five-part ethical audit framework with the help of three experts. Results show that CogniAlign consistently outperforms the baseline across more than sixty moral questions, with average performance gains of 12.2 points in analytic quality, 31.2 points in decisiveness, and 15 points in depth of explanation. In the Heinz dilemma, for example, CogniAlign achieved an overall score of 79 compared to GPT-4o's 65.8, demonstrating a decisive advantage in handling moral reasoning. Through transparent and structured reasoning, CogniAlign demonstrates the feasibility of an auditable approach to AI alignment, though certain challenges still remain.
>
---
#### [replaced 011] Beyond Understanding: Evaluating the Pragmatic Gap in LLMs' Cultural Processing of Figurative Language
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的文化理解任务，旨在评估大模型在 figurative language 的理解和使用上的文化差距。通过设计多项测试任务，分析模型在阿拉伯语和英语习语、谚语上的表现，揭示其文化推理能力的不足。**

- **链接: [https://arxiv.org/pdf/2510.23828v2](https://arxiv.org/pdf/2510.23828v2)**

> **作者:** Mena Attia; Aashiq Muhamed; Mai Alkhamissi; Thamar Solorio; Mona Diab
>
> **备注:** EACL 2026 Main Conference
>
> **摘要:** We present a comprehensive evaluation of the ability of large language models (LLMs) to process culturally grounded language, specifically to understand and pragmatically use figurative expressions that encode local knowledge and cultural nuance. Using figurative language as a proxy for cultural nuance and local knowledge, we design evaluation tasks for contextual understanding, pragmatic use, and connotation interpretation in Arabic and English. We evaluate 22 open- and closed-source LLMs on Egyptian Arabic idioms, multidialectal Arabic proverbs, and English proverbs. Our results show a consistent hierarchy: the average accuracy for Arabic proverbs is 4.29% lower than for English proverbs, and performance for Egyptian idioms is 10.28% lower than for Arabic proverbs. For the pragmatic use task, accuracy drops by 14.07% relative to understanding, though providing contextual idiomatic sentences improves accuracy by 10.66%. Models also struggle with connotative meaning, reaching at most 85.58% agreement with human annotators on idioms with 100% inter-annotator agreement. These findings demonstrate that figurative language serves as an effective diagnostic for cultural reasoning: while LLMs can often interpret figurative meaning, they face challenges in using it appropriately. To support future research, we release Kinayat, the first dataset of Egyptian Arabic idioms designed for both figurative understanding and pragmatic use evaluation.
>
---
#### [replaced 012] ADAB: Arabic Dataset for Automated Politeness Benchmarking -- A Large-Scale Resource for Computational Sociopragmatics
- **分类: cs.CL**

- **简介: 该论文提出ADAB数据集，用于阿拉伯语礼貌检测任务，解决阿拉伯语礼貌资源不足的问题，通过多平台收集并标注10,000条数据，支持相关NLP研究。**

- **链接: [https://arxiv.org/pdf/2602.13870v2](https://arxiv.org/pdf/2602.13870v2)**

> **作者:** Hend Al-Khalifa; Nadia Ghezaiel; Maria Bounnit; Hend Hamed Alhazmi; Noof Abdullah Alfear; Reem Fahad Alqifari; Ameera Masoud Almasoud; Sharefah Al-Ghamdi
>
> **备注:** Paper accepted @ The Fifteenth biennial Language Resources and Evaluation Conference (LREC2026)
>
> **摘要:** The growing importance of culturally-aware natural language processing systems has led to an increasing demand for resources that capture sociopragmatic phenomena across diverse languages. Nevertheless, Arabic-language resources for politeness detection remain under-explored, despite the rich and complex politeness expressions embedded in Arabic communication. In this paper, we introduce ADAB (Arabic Politeness Dataset), a new annotated Arabic dataset collected from four online platforms, including social media, e-commerce, and customer service domains, covering Modern Standard Arabic and multiple dialects (Gulf, Egyptian, Levantine, and Maghrebi). The dataset was annotated based on Arabic linguistic traditions and pragmatic theory, resulting in three classes: polite, impolite, and neutral. It contains 10,000 samples with linguistic feature annotations across 16 politeness categories and achieves substantial inter-annotator agreement (kappa = 0.703). We benchmark 40 model configurations, including traditional machine learning, transformer-based models, and large language models. The dataset aims to support research on politeness-aware Arabic NLP.
>
---
#### [replaced 013] From Competition to Coordination: Market Making as a Scalable Framework for Safe and Aligned Multi-Agent LLM Systems
- **分类: cs.MA; cs.AI; cs.CL**

- **简介: 该论文属于多智能体语言模型协调任务，旨在解决信任、透明和责任问题。提出市场机制框架，通过经济交换实现智能体协作，提升准确性和可解释性。**

- **链接: [https://arxiv.org/pdf/2511.17621v2](https://arxiv.org/pdf/2511.17621v2)**

> **作者:** Brendan Gho; Suman Muppavarapu; Afnan Shaik; Tyson Tsay; Atharva Mohan; James Begin; Kevin Zhu; Archana Vaidheeswaran; Vasu Sharma
>
> **摘要:** As foundation models are increasingly deployed as interacting agents in multi-agent systems, their collective behavior raises new challenges for trustworthiness, transparency, and accountability. Traditional coordination mechanisms, such as centralized oversight or adversarial adjudication, struggle to scale and often obscure how decisions emerge. We introduce a market-making framework for multi-agent large language model (LLM) coordination that organizes agent interactions as structured economic exchanges. In this setup, each agent acts as a market participant, updating and trading probabilistic beliefs, to converge toward shared, truthful outcomes. By aligning local incentives with collective epistemic goals, the framework promotes self-organizing, verifiable reasoning without requiring external enforcement. Empirically, we evaluate this approach across factual reasoning, ethical judgment, and commonsense inference tasks. Market-based coordination yields accuracy gains of up to 10% over single-shot baselines while preserving interpretability and transparency of intermediate reasoning steps. Beyond these improvements, our findings demonstrate that economic coordination principles can operationalize accountability and robustness in multi-agent LLM systems, offering a scalable pathway toward self-correcting, socially responsible AI capable of maintaining trust and oversight in real world deployment scenarios.
>
---
#### [replaced 014] Counting trees: A treebank-driven exploration of syntactic variation in speech and writing across languages
- **分类: cs.CL**

- **简介: 该论文属于语言学研究任务，旨在比较口语与书面语的句法差异。通过分析语料库中的依存结构，揭示不同语域下的句法特征及模态特异性。**

- **链接: [https://arxiv.org/pdf/2505.22774v2](https://arxiv.org/pdf/2505.22774v2)**

> **作者:** Kaja Dobrovoljc
>
> **备注:** Accepted manuscript. Published in Corpus Linguistics and Linguistic Theory (2026)
>
> **摘要:** This paper presents a novel treebank-driven approach to comparing syntactic structures in speech and writing using dependency-parsed corpora. Adopting a fully inductive, bottom-up method, we define syntactic structures as delexicalized dependency (sub)trees and extract them from spoken and written Universal Dependencies (UD) treebanks in two syntactically distinct languages, English and Slovenian. For each corpus, we analyze the size, diversity, and distribution of syntactic inventories, their overlap across modalities, and the structures most characteristic of speech. Results show that, across both languages, spoken corpora contain fewer and less diverse syntactic structures than their written counterparts, with consistent cross-linguistic preferences for certain structural types across modalities. Strikingly, the overlap between spoken and written syntactic inventories is very limited: most structures attested in speech do not occur in writing, pointing to modality-specific preferences in syntactic organization that reflect the distinct demands of real-time interaction and elaborated writing. This contrast is further supported by a keyness analysis of the most frequent speech-specific structures, which highlights patterns associated with interactivity, context-grounding, and economy of expression. We argue that this scalable, language-independent framework offers a useful general method for systematically studying syntactic variation across corpora, laying the groundwork for more comprehensive data-driven theories of grammar in use.
>
---
#### [replaced 015] PEFT-Factory: Unified Parameter-Efficient Fine-Tuning of Autoregressive Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出PEFT-Factory，解决LLM参数高效微调的可复现与比较问题，提供统一框架支持多种PEFT方法及评估。**

- **链接: [https://arxiv.org/pdf/2512.02764v2](https://arxiv.org/pdf/2512.02764v2)**

> **作者:** Robert Belanec; Ivan Srba; Maria Bielikova
>
> **摘要:** Parameter-Efficient Fine-Tuning (PEFT) methods address the increasing size of Large Language Models (LLMs). Currently, many newly introduced PEFT methods are challenging to replicate, deploy, or compare with one another. To address this, we introduce PEFT-Factory, a unified framework for efficient fine-tuning LLMs using both off-the-shelf and custom PEFT methods. While its modular design supports extensibility, it natively provides a representative set of 19 PEFT methods, 27 classification and text generation datasets addressing 12 tasks, and both standard and PEFT-specific evaluation metrics. As a result, PEFT-Factory provides a ready-to-use, controlled, and stable environment, improving replicability and benchmarking of PEFT methods. PEFT-Factory is a downstream framework that originates from the popular LLaMA-Factory, and is publicly available at https://github.com/kinit-sk/PEFT-Factory.
>
---
#### [replaced 016] Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文聚焦于长文本对话中的长期记忆任务，解决现有基准不足的问题。提出BEAM基准和LIGHT框架，提升模型长程推理能力。**

- **链接: [https://arxiv.org/pdf/2510.27246v2](https://arxiv.org/pdf/2510.27246v2)**

> **作者:** Mohammad Tavakoli; Alireza Salemi; Carrie Ye; Mohamed Abdalla; Hamed Zamani; J Ross Mitchell
>
> **摘要:** Evaluating the abilities of large language models (LLMs) for tasks that require long-term memory and thus long-context reasoning, for example in conversational settings, is hampered by the existing benchmarks, which often lack narrative coherence, cover narrow domains, and only test simple recall-oriented tasks. This paper introduces a comprehensive solution to these challenges. First, we present a novel framework for automatically generating long (up to 10M tokens), coherent, and topically diverse conversations, accompanied by probing questions targeting a wide range of memory abilities. From this, we construct BEAM, a new benchmark comprising 100 conversations and 2,000 validated questions. Second, to enhance model performance, we propose LIGHT-a framework inspired by human cognition that equips LLMs with three complementary memory systems: a long-term episodic memory, a short-term working memory, and a scratchpad for accumulating salient facts. Our experiments on BEAM reveal that even LLMs with 1M token context windows (with and without retrieval-augmentation) struggle as dialogues lengthen. In contrast, LIGHT consistently improves performance across various models, achieving an average improvement of 3.5%-12.69% over the strongest baselines, depending on the backbone LLM. An ablation study further confirms the contribution of each memory component.
>
---
#### [replaced 017] Early Multimodal Prediction of Cross-Lingual Meme Virality on Reddit: A Time-Window Analysis
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于跨语言情感分析任务，旨在预测Reddit上梗图的传播性。通过构建多模态特征集并分析时间窗口，解决传统静态阈值预测不足的问题。**

- **链接: [https://arxiv.org/pdf/2510.05761v2](https://arxiv.org/pdf/2510.05761v2)**

> **作者:** Sedat Dogan; Nina Dethlefs; Debarati Chakraborty
>
> **备注:** Accepted to ACM WebSci 2026. 10 pages, 9 fiures and 8 tables
>
> **摘要:** Memes are a central part of online culture, yet their virality remains difficult to predict, especially in cross-lingual settings. We present a large-scale, time-series dataset of 46,578 Reddit memes collected from 25 meme-centric subreddits across eight language groups, with more than one million engagement tracking points. We propose a data-driven definition of virality based on a Hybrid Score that normalises engagement by community size and integrates dynamic features such as velocity and acceleration. This approach directly addresses the field's reliance on static, simple volume-based thresholds with arbitrary cut-offs. Building on this target, we construct a multimodal feature set that combines Visual, Textual, Contextual, Network, and Temporal signals, including structured annotations from a multimodal LLM to scale cross-lingual content labelling in a consistent way. We benchmark interpretable baselines (XGBoost, MLP) against end-to-end deep models (BERT, InceptionV3, CLIP) across early observation windows from 30 to 420 minutes. Our best model, a multimodal XGBoost classifier, achieves a PR AUC of 0.43 at 30 minutes and 0.80 at 420 minutes, indicating that early prediction of meme virality is feasible even under strong class imbalance. The results reveal a clear Content Ceiling, where content-only and deep multimodal baselines plateau at low PR AUC, while structural Network and Temporal features are necessary to surpass this limit. A SHAP-based temporal analysis further uncovers an evidentiary transition, where early predictions are dominated by network priors (author and community context), and later predictions increasingly rely on temporal dynamics (velocity, acceleration) as engagement accumulates. Overall, we reframe meme virality as a dynamic, path-dependent process governed by exposure and early interaction patterns rather than by intrinsic content alone.
>
---
#### [replaced 018] GRASP: Replace Redundant Layers with Adaptive Singular Parameters for Efficient Model Compression
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于模型压缩任务，旨在解决层冗余导致性能下降的问题。通过保留关键奇异值，用少量参数替代冗余层，实现高效压缩并保持性能。**

- **链接: [https://arxiv.org/pdf/2501.00339v4](https://arxiv.org/pdf/2501.00339v4)**

> **作者:** Kainan Liu; Yong Zhang; Ning Cheng; Zhitao Li; Shaojun Wang; Jing Xiao
>
> **备注:** EMNLP 2025(Main)
>
> **摘要:** Recent studies have demonstrated that many layers are functionally redundant in large language models (LLMs), enabling model compression by removing these layers to reduce inference cost. While such approaches can improve efficiency, indiscriminate layer pruning often results in significant performance degradation. In this paper, we propose GRASP (Gradient-based Retention of Adaptive Singular Parameters), a novel compression framework that mitigates this issue by preserving sensitivity-aware singular values. Unlike direct layer pruning, GRASP leverages gradient-based attribution on a small calibration dataset to adaptively identify and retain critical singular components. By replacing redundant layers with only a minimal set of parameters, GRASP achieves efficient compression while maintaining strong performance with minimal overhead. Experiments across multiple LLMs show that GRASP consistently outperforms existing compression methods, achieving 90% of the original model's performance under 20% compression ratio.
>
---
#### [replaced 019] AITutor-EvalKit: Exploring the Capabilities of AI Tutors
- **分类: cs.CL**

- **简介: 该论文介绍AITutor-EvalKit，用于评估AI导师的教育质量，解决AI教学工具评价问题，提供评估、演示和数据分析功能。**

- **链接: [https://arxiv.org/pdf/2512.03688v2](https://arxiv.org/pdf/2512.03688v2)**

> **作者:** Numaan Naeem; Kaushal Kumar Maurya; Kseniia Petukhova; Ekaterina Kochmar
>
> **摘要:** We present AITutor-EvalKit, an application that uses language technology to evaluate the pedagogical quality of AI tutors, provides software for demonstration and evaluation, as well as model inspection and data visualization. This tool is aimed at education stakeholders as well as *ACL community at large, as it supports learning and can also be used to collect user feedback and annotation.
>
---
#### [replaced 020] Conflict-Aware Fusion: Resolving Logic Inertia in Large Language Models via Structured Cognitive Priors
- **分类: cs.AI; cs.CL; cs.LG; cs.LO**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在规则系统下推理不可靠的问题。通过设计压力测试和提出Conflict-Aware Fusion框架，提升模型对矛盾信息的处理能力。**

- **链接: [https://arxiv.org/pdf/2512.06393v3](https://arxiv.org/pdf/2512.06393v3)**

> **作者:** Qiming Bao; Xiaoxuan Fu; Michael Witbrock
>
> **备注:** Under review as a conference paper at ICLR 2026
>
> **摘要:** Large language models (LLMs) excel at many natural language tasks, yet their reasoning reliability under structured perturbations of rule-based systems remains brittle. We present a controlled evaluation framework consisting of four stress tests: (1) rule deletion (redundant vs. essential); (2) contradictory evidence injection; (3) logic-preserving rewrites; and (4) multi-law equivalence stacking. While representative model families (BERT, Qwen2, and TinyLlama) achieve Acc = 1.0000 on base tasks, our framework reveals a critical failure mode termed Logic Inertia - a total breakdown (Acc = 0.0000) under contradictions, where deductive momentum overrides factual reality. To resolve this, we propose Conflict-Aware Fusion, a framework grounded in the Cognitive Structure Hypothesis which posits that robust reasoning requires an explicit structural inductive bias. By imposing a dual-process architecture that separates premise verification from logical deduction, Conflict-Aware Fusion eliminates logic inertia, achieving 1.0000 accuracy on both base and contradictory stress tests, and significantly enhancing robustness to missing evidence. Our results demonstrate that, for reliable multi-step reasoning, structural verification discipline is as critical as training data scale, providing a blueprint for building robust, contradiction-aware AI systems https://github.com/14H034160212/lemo. See the OpenAI/Evals pull request https://github.com/openai/evals/pull/1622.
>
---
#### [replaced 021] Verifying Chain-of-Thought Reasoning via Its Computational Graph
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于模型验证任务，旨在解决CoT推理错误检测问题。提出CRV方法，通过分析计算图结构特征，实现对推理过程的深入理解与错误修正。**

- **链接: [https://arxiv.org/pdf/2510.09312v2](https://arxiv.org/pdf/2510.09312v2)**

> **作者:** Zheng Zhao; Yeskendir Koishekenov; Xianjun Yang; Naila Murray; Nicola Cancedda
>
> **备注:** Accepted to ICLR 2026 (Oral)
>
> **摘要:** Current Chain-of-Thought (CoT) verification methods predict reasoning correctness based on outputs (black-box) or activations (gray-box), but offer limited insight into why a computation fails. We introduce a white-box method: Circuit-based Reasoning Verification (CRV). We hypothesize that attribution graphs of correct CoT steps, viewed as execution traces of the model's latent reasoning circuits, possess distinct structural fingerprints from those of incorrect steps. By training a classifier on structural features of these graphs, we show that these traces contain a powerful signal of reasoning errors. Our white-box approach yields novel scientific insights unattainable by other methods. (1) We demonstrate that structural signatures of error are highly predictive, establishing the viability of verifying reasoning directly via its computational graph. (2) We find these signatures to be highly domain-specific, revealing that failures in different reasoning tasks manifest as distinct computational patterns. (3) We provide evidence that these signatures are not merely correlational; by using our analysis to guide targeted interventions on individual transcoder features, we successfully correct the model's faulty reasoning. Our work shows that, by scrutinizing a model's computational process, we can move from simple error detection to a deeper, causal understanding of LLM reasoning.
>
---
#### [replaced 022] Dialogue is Better Than Monologue: Instructing Medical LLMs via Strategical Conversations
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗AI任务，旨在解决现有系统无法模拟真实临床推理的问题。通过对话微调提升模型在复杂场景下的表现。**

- **链接: [https://arxiv.org/pdf/2501.17860v2](https://arxiv.org/pdf/2501.17860v2)**

> **作者:** Zijie Liu; Xinyu Zhao; Jie Peng; Zhuangdi Zhu; Qingyu Chen; Kaidi Xu; Xia Hu; Tianlong Chen
>
> **备注:** EACL 2026
>
> **摘要:** Current medical AI systems often fail to replicate real-world clinical reasoning, as they are predominantly trained and evaluated on static text and question-answer tasks. These tuning methods and benchmarks overlook critical aspects like evidence-based reasoning and handling distracting information. To bridge this gap, we introduce a novel benchmark that simulates real-world diagnostic scenarios, integrating noise and difficulty levels aligned with USMLE standards. Moreover, we explore dialogue-based fine-tuning, which transforms static datasets into conversational formats to better capture iterative reasoning processes. Experiments show that dialogue-tuned models outperform traditional methods, with improvements of $9.64\%$ in multi-round reasoning scenarios and $6.18\%$ in accuracy in a noisy environment. Our findings highlight dialogue tuning as a promising approach for advancing clinically aligned and robust medical AI systems.
>
---
#### [replaced 023] A Watermark for Black-Box Language Models
- **分类: cs.CR; cs.CL; cs.LG**

- **简介: 该论文属于语言模型水印任务，解决在无法获取模型内部概率分布的情况下如何有效检测LLM输出的问题。提出一种仅需黑盒采样能力的水印方案。**

- **链接: [https://arxiv.org/pdf/2410.02099v3](https://arxiv.org/pdf/2410.02099v3)**

> **作者:** Dara Bahri; John Wieting
>
> **备注:** Published at TMLR 2026
>
> **摘要:** Watermarking has recently emerged as an effective strategy for detecting the outputs of large language models (LLMs). Most existing schemes require white-box access to the model's next-token probability distribution, which is typically not accessible to downstream users of an LLM API. In this work, we propose a principled watermarking scheme that requires only the ability to sample sequences from the LLM (i.e. black-box access), boasts a distortion-free property, and can be chained or nested using multiple secret keys. We provide performance guarantees, demonstrate how it can be leveraged when white-box access is available, and show when it can outperform existing white-box schemes via comprehensive experiments.
>
---
#### [replaced 024] Accidental Vulnerability: Factors in Fine-Tuning that Shift Model Safeguards
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理领域，研究细调过程中意外引入的模型漏洞问题。通过分析数据特征与攻击成功率的关系，探讨防御策略，强调数据设计对模型安全的重要性。**

- **链接: [https://arxiv.org/pdf/2505.16789v3](https://arxiv.org/pdf/2505.16789v3)**

> **作者:** Punya Syon Pandey; Samuel Simko; Kellin Pelrine; Zhijing Jin
>
> **备注:** Second Conference of the International Association for Safe and Ethical Artificial Intelligence (IASEAI 2026)
>
> **摘要:** As large language models (LLMs) gain popularity, their vulnerability to adversarial attacks emerges as a primary concern. While fine-tuning models on domain-specific datasets is often employed to improve model performance, it can inadvertently introduce vulnerabilities within the underlying model. In this work, we investigate Accidental Vulnerability, unexpected vulnerabilities arising from characteristics of fine-tuning data. We begin by identifying potential correlation factors such as linguistic features, semantic similarity, and toxicity across multiple experimental datasets. We then evaluate the adversarial robustness of these fine-tuned models, analyzing persona shifts and interpretability traits to understand how dataset factors contribute to attack success rates. Lastly, we explore causal relationships that offer new insights into adversarial defense strategies, highlighting the crucial role of dataset design in preserving model alignment. Our code is available at https://github.com/psyonp/accidental_vulnerability.
>
---
#### [replaced 025] ProPerSim: Developing Proactive and Personalized AI Assistants through User-Assistant Simulation
- **分类: cs.CL**

- **简介: 该论文提出ProPerSim任务，解决AI助手在真实场景中实现主动且个性化的推荐问题。通过用户-助手模拟，构建能持续学习的个性化助手。**

- **链接: [https://arxiv.org/pdf/2509.21730v2](https://arxiv.org/pdf/2509.21730v2)**

> **作者:** Jiho Kim; Junseong Choi; Woosog Chay; Daeun Kyung; Yeonsu Kwon; Yohan Jo; Edward Choi
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** As large language models (LLMs) become increasingly integrated into daily life, there is growing demand for AI assistants that are not only reactive but also proactive and personalized. While recent advances have pushed forward proactivity and personalization individually, their combination remains underexplored. To bridge this gap, we introduce ProPerSim, a new task and simulation framework for developing assistants capable of making timely, personalized recommendations in realistic home scenarios. In our simulation environment, a user agent with a rich persona interacts with the assistant, providing ratings on how well each suggestion aligns with its preferences and context. The assistant's goal is to use these ratings to learn and adapt to achieve higher scores over time. Built on ProPerSim, we propose ProPerAssistant, a retrieval-augmented, preference-aligned assistant that continually learns and adapts through user feedback. Experiments across 32 diverse personas show that ProPerAssistant adapts its strategy and steadily improves user satisfaction, highlighting the promise of uniting proactivity and personalization.
>
---
#### [replaced 026] TSR: Trajectory-Search Rollouts for Multi-Turn RL of LLM Agents
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出TSR方法，用于改进多轮强化学习中轨迹生成的质量与稳定性，解决奖励稀疏和环境随机带来的挑战。**

- **链接: [https://arxiv.org/pdf/2602.11767v2](https://arxiv.org/pdf/2602.11767v2)**

> **作者:** Aladin Djuhera; Swanand Ravindra Kadhe; Farhan Ahmed; Heiko Ludwig; Holger Boche
>
> **摘要:** Advances in large language models (LLMs) are driving a shift toward using reinforcement learning (RL) to train agents from iterative, multi-turn interactions across tasks. However, multi-turn RL remains challenging as rewards are often sparse or delayed, and environments can be stochastic. In this regime, naive trajectory sampling can hinder exploitation and induce mode collapse. We propose TSR (Trajectory-Search Rollouts), a training-time approach that repurposes test-time scaling ideas for improved per-turn rollout generation. TSR performs lightweight tree-style search to construct high-quality trajectories by selecting high-scoring actions at each turn using task-specific feedback. This improves rollout quality and stabilizes learning while leaving the underlying optimization objective unchanged, making TSR optimizer-agnostic. We instantiate TSR with best-of-N, beam, and shallow lookahead search, and pair it with PPO and GRPO, achieving up to 15% performance gains and more stable learning on Sokoban, FrozenLake, and WebShop tasks at a one-time increase in training compute. By moving search from inference time to the rollout stage of training, TSR provides a simple and general mechanism for stronger multi-turn agent learning, complementary to existing frameworks and rejection-sampling-style selection methods.
>
---
#### [replaced 027] From Medical Records to Diagnostic Dialogues: A Clinical-Grounded Approach and Dataset for Psychiatric Comorbidity
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于精神疾病共病研究任务，旨在解决多疾病共存的诊断难题。通过构建合成病历和对话数据集，支持多轮诊断对话生成与分析。**

- **链接: [https://arxiv.org/pdf/2510.25232v2](https://arxiv.org/pdf/2510.25232v2)**

> **作者:** Tianxi Wan; Jiaming Luo; Siyuan Chen; Kunyao Lan; Jianhua Chen; Haiyang Geng; Mengyue Wu
>
> **摘要:** Psychiatric comorbidity is clinically significant yet challenging due to the complexity of multiple co-occurring disorders. To address this, we develop a novel approach integrating synthetic patient electronic medical record (EMR) construction and multi-agent diagnostic dialogue generation. We create 502 synthetic EMRs for common comorbid conditions using a pipeline that ensures clinical relevance and diversity. Our multi-agent framework transfers the clinical interview protocol into a hierarchical state machine and context tree, supporting over 130 diagnostic states while maintaining clinical standards. Through this rigorous process, we construct PsyCoTalk, the first large-scale dialogue dataset supporting comorbidity, containing 3,000 multi-turn diagnostic dialogues validated by psychiatrists. This dataset enhances diagnostic accuracy and treatment planning, offering a valuable resource for psychiatric comorbidity research. Compared to real-world clinical transcripts, PsyCoTalk exhibits high structural and linguistic fidelity in terms of dialogue length, token distribution, and diagnostic reasoning strategies. Licensed psychiatrists confirm the realism and diagnostic validity of the dialogues. This dataset enables the development and evaluation of models capable of multi-disorder psychiatric screening in a single conversational pass.
>
---
#### [replaced 028] Cross-lingual Collapse: How Language-Centric Foundation Models Shape Reasoning in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究多语言大模型在推理过程中出现的跨语言坍塌现象，属于自然语言处理任务。旨在解决模型在多语言环境下推理时倾向于回归主导语言的问题，通过实验分析并提出缓解方法。**

- **链接: [https://arxiv.org/pdf/2506.05850v3](https://arxiv.org/pdf/2506.05850v3)**

> **作者:** Cheonbok Park; Jeonghoon Kim; Joosung Lee; Sanghwan Bae; Jaegul Choo; Kang Min Yoo
>
> **备注:** Preprint
>
> **摘要:** Reinforcement learning with verifiable reward (RLVR) has been instrumental in eliciting strong reasoning capabilities from large language models (LLMs) via long chains of thought (CoT). During RLVR training, we formalize and systemically study an empirical phenomenon whereby a multilingual model's CoT reverts to its dominant pre-training language (e.g., English) even when prompted in another language, which we term Cross-lingual Collapse. Because the long-CoT regime magnifies exposure to linguistic priors, the underlying trade-off between maximizing reasoning depth and preserving target-language fidelity has remained under-characterized. To examine this trade-off, we train LLMs with Group-Relative Policy Optimization (GRPO) on translated versions of math datasets widely used to elicit long-CoT reasoning. Throughout training, we track both task accuracy and the language consistency of reasoning chains. Our experiments yield three findings: (i) under RLVR, CoT in LLMs systematically drifts toward the pre-training dominant language as reasoning performance rises; (ii) English-centric priors, long-CoT GRPO optimization, task difficulty, and high-entropy decoding jointly amplify this drift, and the pattern persists beyond mathematics; and (iii) interventions that favor target-language traces--via a language-consistency reward, decoding-time controls, or more balanced backbones--mitigate collapse but reveal a persistent performance-fidelity trade-off.
>
---
#### [replaced 029] Bagpiper: Solving Open-Ended Audio Tasks via Rich Captions
- **分类: cs.CL; cs.SD**

- **简介: 该论文提出Bagpiper，一个用于音频理解与生成的8B模型，通过丰富描述实现音频与概念的双向映射，解决传统模型任务单一、缺乏通用性的问题。**

- **链接: [https://arxiv.org/pdf/2602.05220v2](https://arxiv.org/pdf/2602.05220v2)**

> **作者:** Jinchuan Tian; Haoran Wang; Bo-Hao Su; Chien-yu Huang; Qingzheng Wang; Jiatong Shi; William Chen; Xun Gong; Siddhant Arora; Chin-Jou Li; Masao Someki; Takashi Maekaku; Keita Goto; Yusuke Shinohara; Jin Sakuma; Chao-Han Huck Yang; Shinji Watanabe
>
> **摘要:** Current audio foundation models typically rely on rigid, task-specific supervision, addressing isolated factors of audio rather than the whole. In contrast, human intelligence processes audio holistically, seamlessly bridging physical signals with abstract cognitive concepts to execute complex tasks. Grounded in this philosophy, we introduce Bagpiper, an 8B audio foundation model that interprets physical audio via rich captions, i.e., comprehensive natural language descriptions that encapsulate the critical cognitive concepts inherent in the signal (e.g., transcription, audio events). By pre-training on a massive corpus of 600B tokens, the model establishes a robust bidirectional mapping between raw audio and this high-level conceptual space. During fine-tuning, Bagpiper adopts a caption-then-process workflow, simulating an intermediate cognitive reasoning step to solve diverse tasks without task-specific priors. Experimentally, Bagpiper outperforms Qwen-2.5-Omni on MMAU and AIRBench for audio understanding and surpasses CosyVoice3 and TangoFlux in generation quality, capable of synthesizing arbitrary compositions of speech, music, and sound effects. To the best of our knowledge, Bagpiper is among the first works that achieve unified understanding generation for general audio. Model, data, and code are available at Bagpiper Home Page.
>
---
#### [replaced 030] Sparse Autoencoders Can Capture Language-Specific Concepts Across Diverse Languages
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决多语言模型中语言特定概念识别问题。通过稀疏自编码器提取语言特定特征，提升模型的可解释性与语言识别性能。**

- **链接: [https://arxiv.org/pdf/2507.11230v3](https://arxiv.org/pdf/2507.11230v3)**

> **作者:** Lyzander Marciano Andrylie; Inaya Rahmanisa; Mahardika Krisna Ihsani; Alfan Farizki Wicaksono; Haryo Akbarianto Wibowo; Alham Fikri Aji
>
> **摘要:** Understanding the multilingual mechanisms of large language models (LLMs) provides insight into how they process different languages, yet this remains challenging. Existing studies often focus on individual neurons, but their polysemantic nature makes it difficult to isolate language-specific units from cross-lingual representations. To address this, we explore sparse autoencoders (SAEs) for their ability to learn monosemantic features that represent concrete and abstract concepts across languages in LLMs. While some of these features are language-independent, the presence of language-specific features remains underexplored. In this work, we introduce SAE-LAPE, a method based on feature activation probability, to identify language-specific features within the feed-forward network. We find that many such features predominantly appear in the middle to final layers of the model and are interpretable. These features influence the model's multilingual performance and language output and can be used for language identification with performance comparable to fastText along with more interpretability. Our code and complete figures are available at https://github.com/LyzanderAndrylie/language-specific-features
>
---
#### [replaced 031] SQL-Exchange: Transforming SQL Queries Across Domains
- **分类: cs.DB; cs.AI; cs.CL**

- **简介: 该论文提出SQL-Exchange框架，解决跨领域SQL查询映射问题，通过保持查询结构并适配目标模式，提升文本到SQL系统的性能。**

- **链接: [https://arxiv.org/pdf/2508.07087v2](https://arxiv.org/pdf/2508.07087v2)**

> **作者:** Mohammadreza Daviran; Brian Lin; Davood Rafiei
>
> **备注:** Accepted to PVLDB 2026
>
> **摘要:** We introduce SQL-Exchange, a framework for mapping SQL queries across different database schemas by preserving the source query structure while adapting domain-specific elements to align with the target schema. We investigate the conditions under which such mappings are feasible and beneficial, and examine their impact on enhancing the in-context learning performance of text-to-SQL systems as a downstream task. Our comprehensive evaluation across multiple model families and benchmark datasets -- assessing structural alignment with source queries, execution validity on target databases, and semantic correctness -- demonstrates that SQL-Exchange is effective across a wide range of schemas and query types. Our results further show that both in-context prompting with mapped queries and fine-tuning on mapped data consistently yield higher text-to-SQL performance than using examples drawn directly from the source schema.
>
---
#### [replaced 032] vCache: Verified Semantic Prompt Caching
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出vCache，解决语义缓存中静态阈值导致的错误率不可控问题。通过在线学习动态调整阈值，实现可靠缓存响应。属于自然语言处理中的缓存优化任务。**

- **链接: [https://arxiv.org/pdf/2502.03771v5](https://arxiv.org/pdf/2502.03771v5)**

> **作者:** Luis Gaspar Schroeder; Aditya Desai; Alejandro Cuadron; Kyle Chu; Shu Liu; Mark Zhao; Stephan Krusche; Alfons Kemper; Matei Zaharia; Joseph E. Gonzalez
>
> **备注:** ICLR 2026 (accepted)
>
> **摘要:** Semantic caches return cached responses for semantically similar prompts to reduce LLM inference latency and cost. They embed cached prompts and store them alongside their response in a vector database. Embedding similarity metrics assign a numerical score to quantify the similarity between a request and its nearest neighbor prompt from the cache. Existing systems use the same static similarity threshold across all requests to determine whether two prompts can share similar responses. However, we observe that static thresholds do not give formal correctness guarantees, result in unexpected error rates, and lead to suboptimal cache hit rates. This paper proposes vCache, the first verified semantic cache with user-defined error rate guarantees for predictable performance. It employs an online learning algorithm to estimate an optimal threshold for each cached prompt, enabling reliable cache responses without additional training. Our experiments show that vCache consistently meets the specified error bounds while outperforming state-of-the-art static-threshold and fine-tuned embedding baselines with up to 12.5$\times$ higher cache hit and 26$\times$ lower error rates. We release the vCache implementation and four benchmarks to support future research.
>
---
#### [replaced 033] Calibrating Large Language Models with Sample Consistency
- **分类: cs.CL**

- **简介: 该论文属于模型校准任务，旨在解决大语言模型预测置信度不准的问题。通过分析样本一致性提升校准效果，并探讨影响因素及应用建议。**

- **链接: [https://arxiv.org/pdf/2402.13904v2](https://arxiv.org/pdf/2402.13904v2)**

> **作者:** Qing Lyu; Kumar Shridhar; Chaitanya Malaviya; Li Zhang; Yanai Elazar; Niket Tandon; Marianna Apidianaki; Mrinmaya Sachan; Chris Callison-Burch
>
> **备注:** AAAI 2024
>
> **摘要:** Accurately gauging the confidence level of Large Language Models' (LLMs) predictions is pivotal for their reliable application. However, LLMs are often uncalibrated inherently and elude conventional calibration techniques due to their proprietary nature and massive scale. In this work, we explore the potential of deriving confidence from the distribution of multiple randomly sampled model generations, via three measures of consistency. We perform an extensive evaluation across various open and closed-source models on nine reasoning datasets. Results show that consistency-based calibration methods outperform existing post-hoc approaches. Meanwhile, we find that factors such as intermediate explanations, model scaling, and larger sample sizes enhance calibration, while instruction-tuning makes calibration more difficult. Moreover, confidence scores obtained from consistency have the potential to enhance model performance. Finally, we offer practical guidance on choosing suitable consistency metrics for calibration, tailored to the characteristics of various LMs.
>
---
#### [replaced 034] Detecting Early and Implicit Suicidal Ideation via Longitudinal and Information Environment Signals on Social Media
- **分类: cs.SI; cs.AI; cs.CL; cs.CY; cs.HC**

- **简介: 该论文属于早期隐性自杀意念检测任务，旨在通过社交媒体的长期内容和社交环境信号进行预测。研究构建了融合用户及邻近用户互动的模型，提升了检测效果。**

- **链接: [https://arxiv.org/pdf/2510.14889v3](https://arxiv.org/pdf/2510.14889v3)**

> **作者:** Soorya Ram Shimgekar; Ruining Zhao; Agam Goyal; Violeta J. Rodriguez; Paul A. Bloom; Navin Kumar; Hari Sundaram; Koustuv Saha
>
> **摘要:** On social media, several individuals experiencing suicidal ideation (SI) do not disclose their distress explicitly. Instead, signs may surface indirectly through everyday posts or peer interactions. Detecting such implicit signals early is critical but remains challenging. We frame early and implicit SI as a forward-looking prediction task and develop a computational framework that models a user's information environment, consisting of both their longitudinal posting histories as well as the discourse of their socially proximal peers. We adopted a composite network centrality measure to identify top neighbors of a user, and temporally aligned the user's and neighbors' interactions -- integrating the multi-layered signals in a fine-tuned DeBERTa-v3 model. In a Reddit study of 1,000 (500 Case and 500 Control) users, our approach improves early and implicit SI detection by an average of 10% over all other baselines. These findings highlight that peer interactions offer valuable predictive signals and carry broader implications for designing early detection systems that capture indirect as well as masked expressions of risk in online environments.
>
---
#### [replaced 035] AbstRaL: Augmenting LLMs' Reasoning by Reinforcing Abstract Thinking
- **分类: cs.CL; cs.AI; cs.SC**

- **简介: 该论文属于自然语言处理任务，旨在提升大语言模型在数学推理中的鲁棒性。针对模型在分布变化下的性能下降问题，提出通过强化学习增强抽象思维的策略，有效提升模型在数学和一般推理任务上的表现。**

- **链接: [https://arxiv.org/pdf/2506.07751v4](https://arxiv.org/pdf/2506.07751v4)**

> **作者:** Silin Gao; Antoine Bosselut; Samy Bengio; Emmanuel Abbe
>
> **备注:** ICLR 2026
>
> **摘要:** Recent studies have shown that large language models (LLMs), especially smaller ones, often lack robustness in grade school math (GSM) reasoning. In particular, they tend to experience performance drops when faced with distribution shifts, such as changes to numerical or nominal variables, or insertions of distracting clauses. A possible strategy to address this involves generating synthetic data to further "instantiate" reasoning problems on potential variations. In this work, we instead focus on the strategy of "abstracting" reasoning problems. This not only helps counteract distribution shifts but also facilitates the connection to symbolic tools for deriving solutions. Focusing on GSM, we find that this abstraction process is better acquired through reinforcement learning (RL) than just supervised fine-tuning, which often fails to produce faithful abstractions. Our method, AbstRaL -- which promotes abstract reasoning in LLMs using RL on granular abstraction data -- significantly mitigates performance degradation on recent GSM perturbation benchmarks. Besides, improving GSM robustness via AbstRaL is shown to also implicitly benefit LLMs' capabilities on OOD mathematical and general reasoning tasks, indicating that abstract thinking broadly enables better generalizability.
>
---
#### [replaced 036] MemoTime: Memory-Augmented Temporal Knowledge Graph Enhanced Large Language Model Reasoning
- **分类: cs.CL**

- **简介: 该论文属于时序知识图谱推理任务，旨在解决LLM在处理多实体、复合操作和时间序列问题时的不足。提出MemoTime框架，通过结构化 grounding、递归推理和持续学习提升时序推理能力。**

- **链接: [https://arxiv.org/pdf/2510.13614v3](https://arxiv.org/pdf/2510.13614v3)**

> **作者:** Xingyu Tan; Xiaoyang Wang; Qing Liu; Xiwei Xu; Xin Yuan; Liming Zhu; Wenjie Zhang
>
> **备注:** Accepted by The Web Conference 2026 (WWW, 2026)
>
> **摘要:** Large Language Models (LLMs) have achieved impressive reasoning abilities, but struggle with temporal understanding, especially when questions involve multiple entities, compound operators, and evolving event sequences. Temporal Knowledge Graphs (TKGs), which capture vast amounts of temporal facts in a structured format, offer a reliable source for temporal reasoning. However, existing TKG-based LLM reasoning methods still struggle with four major challenges: maintaining temporal faithfulness in multi-hop reasoning, achieving multi-entity temporal synchronization, adapting retrieval to diverse temporal operators, and reusing prior reasoning experience for stability and efficiency. To address these issues, we propose MemoTime, a memory-augmented temporal knowledge graph framework that enhances LLM reasoning through structured grounding, recursive reasoning, and continual experience learning. MemoTime decomposes complex temporal questions into a hierarchical Tree of Time, enabling operator-aware reasoning that enforces monotonic timestamps and co-constrains multiple entities under unified temporal bounds. A dynamic evidence retrieval layer adaptively selects operator-specific retrieval strategies, while a self-evolving experience memory stores verified reasoning traces, toolkit decisions, and sub-question embeddings for cross-type reuse. Comprehensive experiments on multiple temporal QA benchmarks show that MemoTime achieves overall state-of-the-art results, outperforming the strong baseline by up to 24.0%. Furthermore, MemoTime enables smaller models (e.g., Qwen3-4B) to achieve reasoning performance comparable to that of GPT-4-Turbo.
>
---
#### [replaced 037] STaRR: Spatial-Temporal Token-Dynamics-Aware Responsive Remasking for Diffusion Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，解决扩散语言模型中解码速度与质量平衡问题。提出STaRR框架，通过动态调整重掩码策略提升效率。**

- **链接: [https://arxiv.org/pdf/2601.04205v2](https://arxiv.org/pdf/2601.04205v2)**

> **作者:** Xinhao Sun; Huaijin Zhao; Maoliang Li; Zihao Zheng; Jiayu Chen; Yun Liang; Xiang Chen
>
> **摘要:** Diffusion Language Models (DLMs) enable parallel decoding via iterative denoising, where remasking strategies play a critical role in balancing inference speed and output quality. Existing methods predominantly rely on static confidence thresholds, overlooking the spatial-temporal dynamics of token confidence, causing unnecessary remasking. We propose Spatial-Temporal Token-Dynamics-Aware Responsive Remasking (STaRR), a training-free framework that dynamically adapts remasking decisions based on token confidence evolution. STaRR introduces two metrics, temporal variance and spatial deviance, to guide fine-grained, step-wise dynamic thresholding. We further introduce a step-wise dynamic thresholding strategy, further enhanced with responsiveness optimizations for scalability and robustness. Experiments show that STaRR achieves an average speedup of 4.1 and up to 8.9 while maintaining comparable accuracy.
>
---
#### [replaced 038] MolReasoner: Toward Effective and Interpretable Reasoning for Molecular LLMs
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于分子语言模型推理任务，旨在解决LLMs在分子推理中的可解释性和准确性问题。通过两阶段框架MolReasoner提升分子生成与描述的性能。**

- **链接: [https://arxiv.org/pdf/2508.02066v2](https://arxiv.org/pdf/2508.02066v2)**

> **作者:** Guojiang Zhao; Zixiang Lu; Yutang Ge; Sihang Li; Zheng Cheng; Haitao Lin; Lirong Wu; Hanchen Xia; Hengxing Cai; Wentao Guo; Hongshuai Wang; Mingjun Xu; Siyu Zhu; Guolin Ke; Linfeng Zhang; Zhifeng Gao
>
> **摘要:** Large Language Models (LLMs) have shown impressive performance across various domains, but their ability to perform molecular reasoning remains underexplored. Existing methods mostly rely on general-purpose prompting, which lacks domain-specific molecular semantics, or fine-tuning, which faces challenges in interpretability and reasoning depth, often leading to structural and textual hallucinations. To address these issues, we introduce MolReasoner, a two-stage framework that transitions LLMs from memorization to high-fidelity chemical reasoning. In the Mol-SFT stage, knowledge-enhanced Chain-of-Thought (CoT) data provides a strong foundation, while the Mol-RL stage refines reasoning using a novel, task-adaptive reward system to mitigate hallucinations. Extensive evaluations demonstrate that MolReasoner significantly outperforms a wide range of strong baselines in both molecule generation and captioning tasks. Further analyses highlight the framework's synergistic design and its ability to produce more interpretable outputs. Our work presents a principled and effective new approach for advancing high-fidelity molecular reasoning.
>
---
#### [replaced 039] When to use Graphs in RAG: A Comprehensive Analysis for Graph Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文属于知识增强生成任务，旨在解决GraphRAG在实际应用中效果不佳的问题，通过构建基准测试评估其有效性并提供应用指南。**

- **链接: [https://arxiv.org/pdf/2506.05690v3](https://arxiv.org/pdf/2506.05690v3)**

> **作者:** Zhishang Xiang; Chuanjie Wu; Qinggang Zhang; Shengyuan Chen; Zijin Hong; Xiao Huang; Jinsong Su
>
> **备注:** All resources and analyses are collected at https://github.com/GraphRAG-Bench/GraphRAG-Benchmark
>
> **摘要:** Graph retrieval-augmented generation (GraphRAG) has emerged as a powerful paradigm for enhancing large language models (LLMs) with external knowledge. It leverages graphs to model the hierarchical structure between specific concepts, enabling more coherent and effective knowledge retrieval for accurate reasoning.Despite its conceptual promise, recent studies report that GraphRAG frequently underperforms vanilla RAG on many real-world tasks. This raises a critical question: Is GraphRAG really effective, and in which scenarios do graph structures provide measurable benefits for RAG systems? To address this, we propose GraphRAG-Bench, a comprehensive benchmark designed to evaluate GraphRAG models onboth hierarchical knowledge retrieval and deep contextual reasoning. GraphRAG-Bench features a comprehensive dataset with tasks of increasing difficulty, coveringfact retrieval, complex reasoning, contextual summarization, and creative generation, and a systematic evaluation across the entire pipeline, from graph constructionand knowledge retrieval to final generation. Leveraging this novel benchmark, we systematically investigate the conditions when GraphRAG surpasses traditional RAG and the underlying reasons for its success, offering guidelines for its practical application. All related resources and analyses are collected for the community at https://github.com/GraphRAG-Bench/GraphRAG-Benchmark.
>
---
#### [replaced 040] Interpreto: An Explainability Library for Transformers
- **分类: cs.CL; cs.LG**

- **简介: 该论文介绍了一个用于解释Transformer模型的开源库Interpreto，解决模型可解释性问题。它提供属性方法和基于概念的解释，支持分类和文本生成任务。**

- **链接: [https://arxiv.org/pdf/2512.09730v2](https://arxiv.org/pdf/2512.09730v2)**

> **作者:** Antonin Poché; Thomas Mullor; Gabriele Sarti; Frédéric Boisnard; Corentin Friedrich; Charlotte Claye; François Hoofd; Raphael Bernas; Céline Hudelot; Fanny Jourdan
>
> **备注:** Equal contribution: Poché and Jourdan
>
> **摘要:** Interpreto is an open-source Python library for interpreting HuggingFace language models, from early BERT variants to LLMs. It provides two complementary families of methods: attribution methods and concept-based explanations. The library bridges recent research and practical tooling by exposing explanation workflows through a unified API for both classification and text generation. A key differentiator is its end-to-end concept-based pipeline (from activation extraction to concept learning, interpretation, and scoring), which goes beyond feature-level attributions and is uncommon in existing libraries.
>
---
#### [replaced 041] TestExplora: Benchmarking LLMs for Proactive Bug Discovery via Repository-Level Test Generation
- **分类: cs.SE; cs.CL**

- **简介: 该论文属于软件测试任务，旨在解决LLMs在主动发现缺陷方面的不足。提出TestExplora基准，通过对比代码与文档意图来主动检测漏洞。**

- **链接: [https://arxiv.org/pdf/2602.10471v2](https://arxiv.org/pdf/2602.10471v2)**

> **作者:** Steven Liu; Jane Luo; Xin Zhang; Aofan Liu; Hao Liu; Jie Wu; Ziyang Huang; Yangyu Huang; Yu Kang; Scarlett Li
>
> **摘要:** Given that Large Language Models (LLMs) are increasingly applied to automate software development, comprehensive software assurance spans three distinct goals: regression prevention, reactive reproduction, and proactive discovery. Current evaluations systematically overlook the third goal. Specifically, they either treat existing code as ground truth (a compliance trap) for regression prevention, or depend on post-failure artifacts (e.g., issue reports) for bug reproduction-so they rarely surface defects before failures. To bridge this gap, we present TestExplora, a benchmark designed to evaluate LLMs as proactive testers within full-scale, realistic repository environments. TestExplora contains 2,389 tasks from 482 repositories and hides all defect-related signals. Models must proactively find bugs by comparing implementations against documentation-derived intent, using documentation as the oracle. Furthermore, to keep evaluation sustainable and reduce leakage, we propose continuous, time-aware data collection. Our evaluation reveals a significant capability gap: state-of-the-art models achieve a maximum Fail-to-Pass (F2P) rate of only 16.06%. Further analysis indicates that navigating complex cross-module interactions and leveraging agentic exploration are critical to advancing LLMs toward autonomous software quality assurance. Consistent with this, SWEAgent instantiated with GPT-5-mini achieves an F2P of 17.27% and an F2P@5 of 29.7%, highlighting the effectiveness and promise of agentic exploration in proactive bug discovery tasks.
>
---
#### [replaced 042] Do Large Language Models Grasp The Grammar? Evidence from Grammar-Book-Guided Probing in Luxembourgish
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的语法评估任务，旨在解决低资源语言中语法理解评估不足的问题。通过构建语法书引导的评估框架，分析大语言模型在 Luxembourgish 中的语法掌握情况。**

- **链接: [https://arxiv.org/pdf/2510.24856v2](https://arxiv.org/pdf/2510.24856v2)**

> **作者:** Lujun Li; Yewei Song; Lama Sleem; Yiqun Wang; Yangjie Xu; Cedric Lothritz; Niccolo Gentile; Radu State; Tegawende F. Bissyande; Jacques Klein
>
> **备注:** This paper has been accepted for publication in the proceedings of the 15th biennial Language Resources and Evaluation Conference (LREC 2026)
>
> **摘要:** Grammar refers to the system of rules that governs the structural organization and the semantic relations among linguistic units such as sentences, phrases, and words within a given language. In natural language processing, there remains a notable scarcity of grammar focused evaluation protocols, a gap that is even more pronounced for low-resource languages. Moreover, the extent to which large language models genuinely comprehend grammatical structure, especially the mapping between syntactic structures and meanings, remains under debate. To investigate this issue, we propose a Grammar Book Guided evaluation pipeline intended to provide a systematic and generalizable framework for grammar evaluation consisting of four key stages, and in this work we take Luxembourgish as a case study. The results show a weak positive correlation between translation performance and grammatical understanding, indicating that strong translations do not necessarily imply deep grammatical competence. Larger models perform well overall due to their semantic strength but remain weak in morphology and syntax, struggling particularly with Minimal Pair tasks, while strong reasoning ability offers a promising way to enhance their grammatical understanding.
>
---
#### [replaced 043] Group Representational Position Encoding
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出GRAPE框架，统一了位置编码方法，解决长序列建模中的位置表示问题。通过群作用实现乘法旋转和加法偏置，扩展了RoPE和ALiBi的几何设计。**

- **链接: [https://arxiv.org/pdf/2512.07805v4](https://arxiv.org/pdf/2512.07805v4)**

> **作者:** Yifan Zhang; Zixiang Chen; Yifeng Liu; Zhen Qin; Huizhuo Yuan; Kangping Xu; Yang Yuan; Quanquan Gu; Andrew Chi-Chih Yao
>
> **备注:** Published in ICLR 2026; Project Page: https://github.com/model-architectures/GRAPE
>
> **摘要:** We present GRAPE (Group Representational Position Encoding), a unified framework for positional encoding based on group actions. GRAPE unifies two families of mechanisms: (i) multiplicative rotations (Multiplicative GRAPE) in $\operatorname{SO}(d)$ and (ii) additive logit biases (Additive GRAPE) arising from unipotent actions in the general linear group $\mathrm{GL}$. In Multiplicative GRAPE, a position $n \in \mathbb{Z}$ (or $t \in \mathbb{R}$) acts as $\mathbf{G}(n) = \exp(n \, ω\, \mathbf{L})$ with a rank-2 skew-symmetric generator $\mathbf{L} \in \mathbb{R}^{d \times d}$, yielding a relative, compositional, norm-preserving map with a closed-form matrix exponential. RoPE is recovered exactly when the $d/2$ planes correspond to canonical coordinate pairs with a log-uniform spectrum. Learned commuting subspaces and compact non-commuting mixtures strictly extend this geometry to capture cross-subspace feature coupling at $O(d)$ and $O(r d)$ cost per head, respectively. In Additive GRAPE, additive logits arise from rank-1 (or low-rank) unipotent actions, recovering ALiBi and the Forgetting Transformer (FoX) as exact special cases while preserving an exact relative law and streaming cacheability. Overall, GRAPE provides a principled design space for positional geometry in long-context models, subsuming RoPE and ALiBi as special cases. Project page: https://github.com/model-architectures/GRAPE.
>
---
#### [replaced 044] APEX-Agents
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出APEX-Agents基准，用于评估AI代理执行复杂跨应用任务的能力，解决AI在现实工作环境中的生产力评估问题。**

- **链接: [https://arxiv.org/pdf/2601.14242v3](https://arxiv.org/pdf/2601.14242v3)**

> **作者:** Bertie Vidgen; Austin Mann; Abby Fennelly; John Wright Stanly; Lucas Rothman; Marco Burstein; Julien Benchek; David Ostrofsky; Anirudh Ravichandran; Debnil Sur; Neel Venugopal; Alannah Hsia; Isaac Robinson; Calix Huang; Olivia Varones; Daniyal Khan; Michael Haines; Austin Bridges; Jesse Boyle; Koby Twist; Zach Richards; Chirag Mahapatra; Brendan Foody; Osvald Nitski
>
> **摘要:** We introduce the AI Productivity Index for Agents (APEX-Agents), a benchmark for assessing whether AI agents can execute long-horizon, cross-application tasks created by investment banking analysts, management consultants, and corporate lawyers. APEX-Agents requires agents to navigate realistic work environments with files and tools. We test eight agents for the leaderboard using Pass@1. Gemini 3 Flash (Thinking=High) achieves the highest score of 24.0%, followed by GPT-5.2 (Thinking=High), Claude Opus 4.5 (Thinking=High), and Gemini 3 Pro (Thinking=High). We open source the APEX-Agents benchmark (n=480) with all prompts, rubrics, gold outputs, files, and metadata. We also open source Archipelago, our infrastructure for agent execution and evaluation.
>
---
#### [replaced 045] SocialHarmBench: Revealing LLM Vulnerabilities to Socially Harmful Requests
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于安全评估任务，旨在检测大语言模型在社会有害请求中的漏洞。通过构建SocialHarmBench数据集，分析模型在政治、宣传等领域的脆弱性。**

- **链接: [https://arxiv.org/pdf/2510.04891v2](https://arxiv.org/pdf/2510.04891v2)**

> **作者:** Punya Syon Pandey; Hai Son Le; Devansh Bhardwaj; Rada Mihalcea; Zhijing Jin
>
> **备注:** ICLR 2026
>
> **摘要:** Large language models (LLMs) are increasingly deployed in contexts where their failures can have direct sociopolitical consequences. Yet, existing safety benchmarks rarely test vulnerabilities in domains such as political manipulation, propaganda and disinformation generation, or surveillance and information control. We introduce SocialHarmBench, a dataset of 585 prompts spanning 7 sociopolitical categories and 34 countries, designed to surface where LLMs most acutely fail in politically charged contexts. Our evaluations reveal several shortcomings: open-weight models exhibit high vulnerability to harmful compliance, with Mistral-7B reaching attack success rates as high as 97% to 98% in domains such as historical revisionism, propaganda, and political manipulation. Moreover, temporal and geographic analyses show that LLMs are most fragile when confronted with 21st-century or pre-20th-century contexts, and when responding to prompts tied to regions such as Latin America, the USA, and the UK. These findings demonstrate that current safeguards fail to generalize to high-stakes sociopolitical settings, exposing systematic biases and raising concerns about the reliability of LLMs in preserving human rights and democratic values. We share the SocialHarmBench benchmark at https://huggingface.co/datasets/psyonp/SocialHarmBench.
>
---
#### [replaced 046] Transport and Merge: Cross-Architecture Merging for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识迁移任务，解决将大模型知识迁移到小模型的问题。提出基于最优传输的跨架构合并框架，实现异构模型间的有效知识转移。**

- **链接: [https://arxiv.org/pdf/2602.05495v2](https://arxiv.org/pdf/2602.05495v2)**

> **作者:** Chenhang Cui; Binyun Yang; Fei Shen; Yuxin Chen; Jingnan Zheng; Xiang Wang; An Zhang; Tat-Seng Chua
>
> **摘要:** Large language models (LLMs) achieve strong capabilities by scaling model capacity and training data, yet many real-world deployments rely on smaller models trained or adapted from low-resource data. This gap motivates the need for mechanisms to transfer knowledge from large, high-resource models to smaller, low-resource targets. While model merging provides an effective transfer mechanism, most existing approaches assume architecture-compatible models and therefore cannot directly transfer knowledge from large high-resource LLMs to heterogeneous low-resource targets. In this work, we propose a cross-architecture merging framework based on optimal transport (OT) that aligns activations to infer cross-neuron correspondences between heterogeneous models. The resulting transport plans are then used to guide direct weight-space fusion, enabling effective high-resource to low-resource transfer using only a small set of inputs. Extensive experiments across low-resource languages and specialized domains demonstrate consistent improvements over target models.
>
---
#### [replaced 047] Towards interpretable models for language proficiency assessment: Predicting the CEFR level of Estonian learner texts
- **分类: cs.CL**

- **简介: 该论文属于语言能力评估任务，旨在通过NLP技术预测埃塞尼亚语学习者文本的CEFR等级。研究通过特征选择提升模型的可解释性和泛化能力，实现准确分类。**

- **链接: [https://arxiv.org/pdf/2602.13102v2](https://arxiv.org/pdf/2602.13102v2)**

> **作者:** Kais Allkivi
>
> **摘要:** Using NLP to analyze authentic learner language helps to build automated assessment and feedback tools. It also offers new and extensive insights into the development of second language production. However, there is a lack of research explicitly combining these aspects. This study aimed to classify Estonian proficiency examination writings (levels A2-C1), assuming that careful feature selection can lead to more explainable and generalizable machine learning models for language testing. Various linguistic properties of the training data were analyzed to identify relevant proficiency predictors associated with increasing complexity and correctness, rather than the writing task. Such lexical, morphological, surface, and error features were used to train classification models, which were compared to models that also allowed for other features. The pre-selected features yielded a similar test accuracy but reduced variation in the classification of different text types. The best classifiers achieved an accuracy of around 0.9. Additional evaluation on an earlier exam sample revealed that the writings have become more complex over a 7-10-year period, while accuracy still reached 0.8 with some feature sets. The results have been implemented in the writing evaluation module of an Estonian open-source language learning environment.
>
---
#### [replaced 048] The Generalization Ridge: Information Flow in Natural Language Generation
- **分类: cs.CL**

- **简介: 该论文属于自然语言生成任务，研究Transformer模型内部信息流动机制，揭示中间层在提升泛化能力中的作用。**

- **链接: [https://arxiv.org/pdf/2507.05387v4](https://arxiv.org/pdf/2507.05387v4)**

> **作者:** Ruidi Chang; Chunyuan Deng; Hanjie Chen
>
> **摘要:** Transformer-based language models have achieved state-of-the-art performance in natural language generation (NLG), yet their internal mechanisms for synthesizing task-relevant information remain insufficiently understood. While prior studies suggest that intermediate layers often yield more generalizable representations than final layers, how this generalization ability emerges and propagates across layers during training remains unclear. To address this gap, we propose InfoRidge, an information-theoretic framework, to characterize how predictive information-the mutual information between hidden representations and target outputs-varies across depth during training. Our experiments across various models and datasets reveal a consistent non-monotonic trend: predictive information peaks in intermediate layers-forming a generalization ridge-before declining in final layers, reflecting a transition between generalization and memorization. To further investigate this phenomenon, we conduct a set of complementary analyses that leverage residual scaling, attention pattern, and controlled model capacity to characterize layer-wise functional specialization. We further validate our findings with multiple-token generation experiments, verifying that the observed ridge phenomenon persists across decoding steps. Together, these findings offer new insights into the internal mechanisms of transformers and underscore the critical role of intermediate layers in supporting generalization.
>
---
#### [replaced 049] Argument Rarity-based Originality Assessment for AI-Assisted Writing
- **分类: cs.CL**

- **简介: 该论文属于写作评估任务，旨在解决AI辅助写作中原创性评价问题。提出AROA框架，通过四个维度评估论点新颖性，验证其有效性。**

- **链接: [https://arxiv.org/pdf/2602.01560v3](https://arxiv.org/pdf/2602.01560v3)**

> **作者:** Keito Inoshita; Michiaki Omura; Tsukasa Yamanaka; Go Maeda; Kentaro Tsuji
>
> **摘要:** This study proposes Argument Rarity-based Originality Assessment (AROA), a framework for automatically evaluating argumentative originality in student essays. AROA defines originality as rarity within a reference corpus and evaluates it through four complementary components: structural rarity, claim rarity, evidence rarity, and cognitive depth, quantified via density estimation and integrated with quality adjustment. Experiments using 1,375 human essays and 1,000 AI-generated essays on two argumentative topics revealed three key findings. First, a strong negative correlation (r = -0.67) between text quality and claim rarity demonstrates a quality-originality trade-off. Second, while AI essays achieved near-perfect quality scores (Q = 0.998), their claim rarity was approximately one-fifth of human levels (AI: 0.037, human: 0.170), indicating that LLMs can reproduce argumentative structure but not semantic originality. Third, the four components showed low mutual correlations (r = 0.06--0.13 between structural and semantic dimensions), confirming that they capture genuinely independent aspects of originality. These results suggest that writing assessment in the AI era must shift from quality to originality.
>
---
#### [replaced 050] TaP: A Taxonomy-Guided Framework for Automated and Scalable Preference Data Generation
- **分类: cs.CL**

- **简介: 该论文提出TaP框架，用于跨语言自动化生成高质量偏好数据集。解决数据稀缺与非英语数据不足的问题，通过结构化分类实现数据多样性，提升模型训练效果。**

- **链接: [https://arxiv.org/pdf/2506.23979v4](https://arxiv.org/pdf/2506.23979v4)**

> **作者:** Renren Jin; Tianhao Shen; Xinwei Wu; Dan Shi; Haoran Sun; Yuqi Ren; Wuwei Huang; Quandong Wang; Wei Liu; Jian Luan; Bin Wang; Deyi Xiong
>
> **备注:** 33 pages, 16 tables, 10 figures
>
> **摘要:** Conducting supervised and preference fine-tuning of large language models (LLMs) requires high-quality datasets to improve their ability to follow instructions and align with human preferences and values. However, constructing such datasets is resource-intensive, and most publicly available datasets are in English. To address these challenges, we propose the \underline{\textbf{Ta}}xonomy-Guided \underline{\textbf{P}}reference Data Generation (TaP) framework for automated, scalable preference dataset construction across languages. TaP uses a structured taxonomy to provide fine-grained control over dataset composition, ensuring diversity and broad coverage. We use TaP-generated datasets to perform supervised and preference fine-tuning on multiple LLMs. Experimental results demonstrate that LLMs trained on TaP-generated datasets outperform those trained on existing open-source datasets. Remarkably, LLMs trained on TaP-generated datasets outperform models trained on an open-source dataset that is 180$\times$ larger.
>
---
#### [replaced 051] PEFT-Bench: A Parameter-Efficient Fine-Tuning Methods Benchmark
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM训练成本高问题。提出PEFT-Bench基准和PSCP指标，评估多种PEFT方法的效率与性能。**

- **链接: [https://arxiv.org/pdf/2511.21285v2](https://arxiv.org/pdf/2511.21285v2)**

> **作者:** Robert Belanec; Branislav Pecher; Ivan Srba; Maria Bielikova
>
> **摘要:** Despite the state-of-the-art performance of Large Language Models (LLMs) achieved on many tasks, their massive scale often leads to high computational and environmental costs, limiting their accessibility. Parameter-Efficient Fine-Tuning (PEFT) methods address this challenge by reducing the number of trainable parameters while maintaining strong downstream performance. Despite the advances in PEFT methods, current evaluations remain limited (in terms of evaluated models and datasets) and difficult to reproduce. To bridge this gap, we introduce PEFT-Bench, a unified end-to-end benchmark for evaluating diverse PEFT methods on autoregressive LLMs. We demonstrate its usage across 27 NLP datasets and 7 PEFT methods. To account for different PEFT training and inference factors, we also introduce the PEFT Soft Cost Penalties (PSCP) metric, which takes trainable parameters, inference speed, and training memory usage into account.
>
---
#### [replaced 052] promptolution: A Unified, Modular Framework for Prompt Optimization
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM提示优化的实用化问题。提出promptolution框架，实现统一、模块化的提示优化，便于集成和使用。**

- **链接: [https://arxiv.org/pdf/2512.02840v2](https://arxiv.org/pdf/2512.02840v2)**

> **作者:** Tom Zehle; Timo Heiß; Moritz Schlager; Matthias Aßenmacher; Matthias Feurer
>
> **摘要:** Prompt optimization has become crucial for enhancing the performance of large language models (LLMs) across a broad range of tasks. Although many research papers demonstrate its effectiveness, practical adoption is hindered because existing implementations are often tied to unmaintained, isolated research codebases or require invasive integration into application frameworks. To address this, we introduce promptolution, a unified, modular open-source framework that provides all components required for prompt optimization within a single extensible system for both practitioners and researchers. It integrates multiple contemporary discrete prompt optimizers, supports systematic and reproducible benchmarking, and returns framework-agnostic prompt strings, enabling seamless integration into existing LLM pipelines while remaining agnostic to the underlying model implementation.
>
---
#### [replaced 053] OpenLID-v3: Improving the Precision of Closely Related Language Identification -- An Experience Report
- **分类: cs.CL**

- **简介: 该论文属于语言识别任务，旨在提升相近语言的识别精度。通过扩展OpenLID，增加数据、合并集群并引入噪声标签，提出OpenLID-v3系统。**

- **链接: [https://arxiv.org/pdf/2602.13139v2](https://arxiv.org/pdf/2602.13139v2)**

> **作者:** Mariia Fedorova; Nikolay Arefyev; Maja Buljan; Jindřich Helcl; Stephan Oepen; Egil Rønningstad; Yves Scherrer
>
> **备注:** VarDial'26 workshop at the EACL 2026 conference
>
> **摘要:** Language identification (LID) is an essential step in building high-quality multilingual datasets from web data. Existing LID tools (such as OpenLID or GlotLID) often struggle to identify closely related languages and to distinguish valid natural language from noise, which contaminates language-specific subsets, especially for low-resource languages. In this work we extend the OpenLID classifier by adding more training data, merging problematic language variant clusters, and introducing a special label for marking noise. We call this extended system OpenLID-v3 and evaluate it against GlotLID on multiple benchmarks. During development, we focus on three groups of closely related languages (Bosnian, Croatian, and Serbian; Romance varieties of Northern Italy and Southern France; and Scandinavian languages) and contribute new evaluation datasets where existing ones are inadequate. We find that ensemble approaches improve precision but also substantially reduce coverage for low-resource languages. OpenLID-v3 is available on https://huggingface.co/HPLT/OpenLID-v3.
>
---
#### [replaced 054] OmniRAG-Agent: Agentic Omnimodal Reasoning for Low-Resource Long Audio-Video Question Answering
- **分类: cs.CL**

- **简介: 该论文属于多模态问答任务，解决低资源长音频视频问答中的编码成本高、检索弱等问题。提出OmniRAG-Agent方法，结合检索增强生成与智能体规划，提升问答效果。**

- **链接: [https://arxiv.org/pdf/2602.03707v3](https://arxiv.org/pdf/2602.03707v3)**

> **作者:** Yifan Zhu; Xinyu Mu; Tao Feng; Zhonghong Ou; Yuning Gong; Haoran Luo
>
> **摘要:** Long-horizon omnimodal question answering answers questions by reasoning over text, images, audio, and video. Despite recent progress on OmniLLMs, low-resource long audio-video QA still suffers from costly dense encoding, weak fine-grained retrieval, limited proactive planning, and no clear end-to-end optimization. To address these issues, we propose OmniRAG-Agent, an agentic omnimodal QA method for budgeted long audio-video reasoning. It builds an image-audio retrieval-augmented generation module that lets an OmniLLM fetch short, relevant frames and audio snippets from external banks. Moreover, it uses an agent loop that plans, calls tools across turns, and merges retrieved evidence to answer complex queries. Furthermore, we apply group relative policy optimization to jointly improve tool use and answer quality over time. Experiments on OmniVideoBench, WorldSense, and Daily-Omni show that OmniRAG-Agent consistently outperforms prior methods under low-resource settings and achieves strong results, with ablations validating each component.
>
---
#### [replaced 055] Closing the Gap Between Text and Speech Understanding in LLMs
- **分类: cs.CL; cs.AI; eess.AS**

- **简介: 该论文属于自然语言处理任务，解决LLM在语音理解上的性能差距问题。通过SALAD方法，提升语音与文本对齐效果，减少对合成数据的依赖。**

- **链接: [https://arxiv.org/pdf/2510.13632v2](https://arxiv.org/pdf/2510.13632v2)**

> **作者:** Santiago Cuervo; Skyler Seto; Maureen de Seyssel; Richard He Bai; Zijin Gu; Tatiana Likhomanenko; Navdeep Jaitly; Zakaria Aldeneh
>
> **摘要:** Large Language Models (LLMs) can be adapted to extend their text capabilities to speech inputs. However, these speech-adapted LLMs consistently underperform their text-based counterparts--and even cascaded pipelines--on language understanding tasks. We term this shortfall the text-speech understanding gap: the performance drop observed when a speech-adapted LLM processes spoken inputs relative to when the original text-based LLM processes the equivalent text. Recent approaches to narrowing this gap either rely on large-scale speech synthesis of text corpora, which is costly and heavily dependent on synthetic data, or on large-scale proprietary speech datasets, which are not reproducible. As a result, there remains a need for more data-efficient alternatives for closing the text-speech understanding gap. In this work, we analyze the gap as driven by two factors: (i) forgetting of text capabilities during adaptation, and (ii) cross-modal misalignment between speech and text. Based on this analysis, we introduce SALAD--Sample-efficient Alignment with Learning through Active selection and cross-modal Distillation--which combines cross-modal distillation with targeted synthetic data to improve alignment while mitigating forgetting. Applied to 3B and 7B LLMs, SALAD achieves competitive performance with a strong open-weight model across broad-domain benchmarks in knowledge, language understanding, and reasoning, while training on over an order of magnitude less speech data from public corpora.
>
---
#### [replaced 056] Federated Co-tuning Framework for Large and Small Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出FedCoLLM框架，解决大模型与小模型在联邦学习中协同优化的问题，通过知识迁移提升双方性能。**

- **链接: [https://arxiv.org/pdf/2411.11707v2](https://arxiv.org/pdf/2411.11707v2)**

> **作者:** Tao Fan; Yan Kang; Guoqiang Ma; Lixin Fan; Shuoling Liu; Kai Chen; Qiang Yang
>
> **摘要:** By adapting Large Language Models (LLMs) to domain-specific tasks or enriching them with domain-specific knowledge, we can fully harness the capabilities of LLMs. Nonetheless, a gap persists in achieving simultaneous mutual enhancement between the server's LLM and the downstream clients' Small Language Models (SLMs). To address this, we propose FedCoLLM, a novel and parameter-efficient federated framework designed for co-tuning LLMs and SLMs. This approach is aimed at adaptively transferring server-side LLMs knowledge to clients' SLMs while simultaneously enriching the LLMs with domain insights from the clients. To accomplish this, FedCoLLM utilizes lightweight adapters in conjunction with SLMs, facilitating knowledge exchange between server and clients in a manner that respects data privacy while also minimizing computational and communication overhead. Our evaluation of FedCoLLM, utilizing various public LLMs and SLMs across a range of NLP text generation tasks, reveals that the performance of clients' SLMs experiences notable improvements with the assistance of the LLMs. Simultaneously, the LLMs enhanced via FedCoLLM achieves comparable performance to that obtained through direct fine-tuning on clients' data. Our code has been contributed to the FATE open-source project and is now publicly accessible at https://github.com/FederatedAI/FATE-LLM/tree/main/python/fate_llm/algo/fedcollm.
>
---
#### [replaced 057] MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对多模态大语言模型的计算效率问题，提出MoDES框架，通过动态跳过冗余专家提升推理效率与准确性。**

- **链接: [https://arxiv.org/pdf/2511.15690v2](https://arxiv.org/pdf/2511.15690v2)**

> **作者:** Yushi Huang; Zining Wang; Zhihang Yuan; Yifu Ding; Ruihao Gong; Jinyang Guo; Xianglong Liu; Jun Zhang
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Mixture-of-Experts (MoE) Multimodal large language models (MLLMs) excel at vision-language tasks, but they suffer from high computational inefficiency. To reduce inference overhead, expert skipping methods have been proposed to deactivate redundant experts based on the current input tokens. However, we find that applying these methods-originally designed for unimodal large language models (LLMs)-to MLLMs results in considerable performance degradation. This is primarily because such methods fail to account for the heterogeneous contributions of experts across MoE layers and modality-specific behaviors of tokens within these layers. Motivated by these findings, we propose MoDES, the first training-free framework that adaptively skips experts to enable efficient and accurate MoE MLLM inference. It incorporates a globally-modulated local gating (GMLG) mechanism that integrates global layer-wise importance into local routing probabilities to accurately estimate per-token expert importance. A dual-modality thresholding (DMT) method is then applied, which processes tokens from each modality separately, to derive the skipping schedule. To set the optimal thresholds, we introduce a frontier search algorithm that exploits monotonicity properties, cutting convergence time from several days to a few hours. Extensive experiments for 3 model series across 13 benchmarks demonstrate that MoDES far outperforms previous approaches. For instance, when skipping 88% experts for Qwen3-VL-MoE-30B-A3B-Instruct, the performance boost is up to 10.67% (97.33% vs. 86.66%). Furthermore, MoDES significantly enhances inference speed, improving the prefilling time by 2.16$\times$ and the decoding time by 1.26$\times$. Our code is available at https://github.com/ModelTC/MoDES.
>
---
#### [replaced 058] What If We Allocate Test-Time Compute Adaptively?
- **分类: cs.CL**

- **简介: 该论文属于推理任务，解决测试阶段计算资源分配效率问题。提出一种基于验证器引导的自适应框架，通过动态调整计算策略提升推理效果。**

- **链接: [https://arxiv.org/pdf/2602.01070v2](https://arxiv.org/pdf/2602.01070v2)**

> **作者:** Ahsan Bilal; Ahmed Mohsin; Muhammad Umer; Ali Subhan; Hassan Rizwan; Ayesha Mohsin; Dean Hougen
>
> **摘要:** Test-time compute scaling allocates inference computation uniformly, uses fixed sampling strategies, and applies verification only for reranking. In contrast, we propose a verifier-guided adaptive framework treating reasoning as iterative trajectory generation and selection. For each problem, the agent runs multiple inference iterations. In each iteration, it optionally produces a high-level plan, selects a set of reasoning tools and a compute strategy together with an exploration parameter, and then generates a candidate reasoning trajectory. A process reward model (PRM) serves as a unified control signal: within each iteration, step-level PRM scores are aggregated to guide pruning and expansion during generation, and across iterations, aggregated trajectory rewards are used to select the final response. Across datasets, our dynamic, PRM-guided approach consistently outperforms direct test-time scaling, yielding large gains on MATH-500 and several-fold improvements on harder benchmarks such as AIME24 and AMO-Bench. We characterize efficiency using theoretical FLOPs and a compute intensity metric penalizing wasted generation and tool overhead, demonstrating that verification-guided allocation concentrates computation on high-utility reasoning paths.
>
---
#### [replaced 059] Buy versus Build an LLM: A Decision Framework for Governments
- **分类: cs.CY; cs.AI; cs.CE; cs.CL; cs.SI**

- **简介: 该论文属于政策分析任务，旨在解决政府在使用LLM时的“购买还是自建”决策问题，通过多维度框架帮助制定合适策略。**

- **链接: [https://arxiv.org/pdf/2602.13033v2](https://arxiv.org/pdf/2602.13033v2)**

> **作者:** Jiahao Lu; Ziwei Xu; William Tjhi; Junnan Li; Antoine Bosselut; Pang Wei Koh; Mohan Kankanhalli
>
> **备注:** The short version of this document is published as an ACM TechBrief at https://dl.acm.org/doi/epdf/10.1145/3797946, and this document is published as an ACM Technology Policy Council white paper at https://www.acm.org/binaries/content/assets/public-policy/buildvsbuyai.pdf
>
> **摘要:** Large Language Models (LLMs) represent a new frontier of digital infrastructure that can support a wide range of public-sector applications, from general purpose citizen services to specialized and sensitive state functions. When expanding AI access, governments face a set of strategic choices over whether to buy existing services, build domestic capabilities, or adopt hybrid approaches across different domains and use cases. These are critical decisions especially when leading model providers are often foreign corporations, and LLM outputs are increasingly treated as trusted inputs to public decision-making and public discourse. In practice, these decisions are not intended to mandate a single approach across all domains; instead, national AI strategies are typically pluralistic, with sovereign, commercial and open-source models coexisting to serve different purposes. Governments may rely on commercial models for non-sensitive or commodity tasks, while pursuing greater control for critical, high-risk or strategically important applications. This paper provides a strategic framework for making this decision by evaluating these options across dimensions including sovereignty, safety, cost, resource capability, cultural fit, and sustainability. Importantly, "building" does not imply that governments must act alone: domestic capabilities may be developed through public research institutions, universities, state-owned enterprises, joint ventures, or broader national ecosystems. By detailing the technical requirements and practical challenges of each pathway, this work aims to serve as a reference for policy-makers to determine whether a buy or build approach best aligns with their specific national needs and societal goals.
>
---
#### [replaced 060] Shop-R1: Rewarding LLMs to Simulate Human Behavior in Online Shopping via Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文属于在线购物行为模拟任务，旨在提升LLMs生成真实人类行为的能力。通过强化学习框架Shop-R1，分解任务为推理生成与动作预测，解决模型推理能力受限的问题。**

- **链接: [https://arxiv.org/pdf/2507.17842v2](https://arxiv.org/pdf/2507.17842v2)**

> **作者:** Yimeng Zhang; Tian Wang; Jiri Gesi; Ziyi Wang; Yuxuan Lu; Jiacheng Lin; Sinong Zhan; Vianne Gao; Ruochen Jiao; Junze Liu; Kun Qian; Yuxin Tang; Ran Xue; Houyu Zhang; Qingjun Cui; Yufan Guo; Dakuo Wang
>
> **备注:** Accepted by ICLR 2026. The project page is available at https://damon-demon.github.io/shop-r1.html
>
> **摘要:** Large Language Models (LLMs) have recently demonstrated strong potential in generating 'believable human-like' behavior in web environments. Prior work has explored augmenting training data with LLM-synthesized rationales and applying supervised fine-tuning (SFT) to enhance reasoning ability, which in turn can improve downstream action prediction. However, the performance of such approaches remains inherently bounded by the reasoning capabilities of the model used to generate the rationales. In this paper, we introduce Shop-R1, a novel reinforcement learning (RL) framework aimed at enhancing the reasoning ability of LLMs for simulation of real human behavior in online shopping environments. Specifically, Shop-R1 decomposes the human behavior simulation task into two stages: rationale generation and action prediction, each guided by distinct reward signals. For rationale generation, we leverage internal model signals (e.g., logit distributions) to guide the reasoning process in a self-supervised manner. For action prediction, we propose a hierarchical reward structure with difficulty-aware scaling to prevent reward hacking and enable fine-grained reward assignment. This design evaluates both high-level action types and the correctness of fine-grained sub-action details (attributes and values), rewarding outputs proportionally to their difficulty. Experimental results show that our method achieves a relative improvement of over 65% compared to the baseline. The project page is available at https://damon-demon.github.io/shop-r1.html.
>
---
#### [replaced 061] STAPO: Stabilizing Reinforcement Learning for LLMs by Silencing Rare Spurious Tokens
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于强化学习任务，旨在解决大语言模型训练中的稳定性问题。通过识别并抑制稀有虚假标记，提升模型推理稳定性与性能。**

- **链接: [https://arxiv.org/pdf/2602.15620v3](https://arxiv.org/pdf/2602.15620v3)**

> **作者:** Shiqi Liu; Zeyu He; Guojian Zhan; Letian Tao; Zhilong Zheng; Jiang Wu; Yinuo Wang; Yang Guan; Kehua Sheng; Bo Zhang; Keqiang Li; Jingliang Duan; Shengbo Eben Li
>
> **摘要:** Reinforcement Learning (RL) has significantly improved large language model reasoning, but existing RL fine-tuning methods rely heavily on heuristic techniques such as entropy regularization and reweighting to maintain stability. In practice, they often suffer from late-stage performance collapse, leading to degraded reasoning quality and unstable training. Our analysis shows that the magnitude of token-wise policy gradients in RL is negatively correlated with token probability and local policy entropy. We find that training instability can be caused by a tiny fraction of tokens, approximately 0.01%, which we term spurious tokens. When such tokens appear in correct responses, they contribute little to the reasoning outcome but inherit the full sequence-level reward, leading to abnormally amplified gradient updates. To mitigate this instability, we design an S2T (silencing spurious tokens) mechanism to efficiently identify spurious tokens through characteristic signals with low probability, low entropy, and positive advantage, and then suppress their gradient perturbations during optimization. Incorporating this mechanism into a group-based objective, we propose Spurious-Token-Aware Policy Optimization (STAPO), which promotes stable and effective large-scale model refinement. Across six mathematical reasoning benchmarks using Qwen 1.7B, 8B, and 14B base models, STAPO consistently demonstrates superior entropy stability and achieves an average performance improvement of 7.13% ($ρ_{\mathrm{T}}$=1.0, top-p=1.0) and 3.69% ($ρ_{\mathrm{T}}$=0.7, top-p=0.9) over GRPO, 20-Entropy, and JustRL.
>
---
#### [replaced 062] Where Did This Sentence Come From? Tracing Provenance in LLM Reasoning Distillation
- **分类: cs.CL**

- **简介: 该论文属于模型推理蒸馏任务，旨在解决蒸馏模型行为一致性问题。通过追踪生成内容的来源，分析学生模型在新场景下的表现，并提出基于教师指导的数据选择方法。**

- **链接: [https://arxiv.org/pdf/2512.20908v2](https://arxiv.org/pdf/2512.20908v2)**

> **作者:** Kaiyuan Liu; Shaotian Yan; Rui Miao; Bing Wang; Chen Shen; Jun Zhang; Jieping Ye
>
> **摘要:** Reasoning distillation has attracted increasing attention. It typically leverages a large teacher model to generate reasoning paths, which are then used to fine-tune a student model so that it mimics the teacher's behavior in training contexts. However, previous approaches have lacked a detailed analysis of the origins of the distilled model's capabilities. It remains unclear whether the student can maintain consistent behaviors with the teacher in novel test-time contexts, or whether it regresses to its original output patterns, raising concerns about the generalization of distillation models. To analyse this question, we introduce a cross-model Reasoning Distillation Provenance Tracing framework. For each action (e.g., a sentence) produced by the distilled model, we obtain the predictive probabilities assigned by the teacher, the original student, and the distilled model under the same context. By comparing these probabilities, we classify each action into different categories. By systematically disentangling the provenance of each action, we experimentally demonstrate that, in test-time contexts, the distilled model can indeed generate teacher-originated actions, which correlate with and plausibly explain observed performance on distilled model. Building on this analysis, we further propose a teacher-guided data selection method. Unlike prior approach that rely on heuristics, our method directly compares teacher-student divergences on the training data, providing a principled selection criterion. We validate the effectiveness of our approach across multiple representative teacher models and diverse student models. The results highlight the utility of our provenance-tracing framework and underscore its promise for reasoning distillation. We hope to share Reasoning Distillation Provenance Tracing and our insights into reasoning distillation with the community.
>
---
#### [replaced 063] Symphonym: Universal Phonetic Embeddings for Cross-Script Name Matching
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Symphonym，解决跨文字系统姓名匹配问题，通过神经嵌入将不同语言名称映射到统一空间，提升匹配效率与准确性。**

- **链接: [https://arxiv.org/pdf/2601.06932v3](https://arxiv.org/pdf/2601.06932v3)**

> **作者:** Stephen Gadd
>
> **备注:** 29 pages, 3 tables
>
> **摘要:** Linking names across historical sources, languages, and writing systems remains a fundamental challenge in digital humanities and geographic information retrieval. Existing approaches require language-specific phonetic algorithms or fail to capture phonetic relationships across different scripts. This paper presents Symphonym, a neural embedding system that maps names from any script into a unified 128-dimensional phonetic space, enabling direct similarity comparison without runtime phonetic conversion. Symphonym uses a Teacher-Student architecture where a Teacher network trained on articulatory phonetic features produces target embeddings, while a Student network learns to approximate these embeddings directly from characters. The Teacher combines Epitran (extended with 100 new language-script mappings), Phonikud for Hebrew, and CharsiuG2P for Chinese, Japanese, and Korean. Training used 32.7 million triplet samples of toponyms spanning 20 writing systems from GeoNames, Wikidata, and Getty Thesaurus of Geographic Names. On the MEHDIE Hebrew-Arabic historical toponym benchmark, Symphonym achieves Recall@10 of 97.6% and MRR of 90.3%, outperforming Levenshtein and Jaro-Winkler baselines (Recall@1: 86.7% vs 81.5% and 78.5%). Evaluation on 12,947 real cross-script training pairs shows 82.6% achieve greater than 0.75 cosine similarity, with best performance on Arabic-Cyrillic (94--100%) and Cyrillic-Latin (94.3%) combinations. The fixed-length embeddings enable efficient retrieval in digital humanities workflows, with a case study on medieval personal names demonstrating effective transfer from modern place names to historical orthographic variation.
>
---
#### [replaced 064] Look Back to Reason Forward: Revisitable Memory for Long-Context LLM Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于长文本问答任务，解决长上下文信息丢失与推理困难问题。提出ReMemR1模型，通过记忆检索和多级奖励机制提升模型对历史信息的利用能力。**

- **链接: [https://arxiv.org/pdf/2509.23040v4](https://arxiv.org/pdf/2509.23040v4)**

> **作者:** Yaorui Shi; Yuxin Chen; Siyuan Wang; Sihang Li; Hengxing Cai; Qi Gu; Xiang Wang; An Zhang
>
> **摘要:** Large language models face challenges in long-context question answering, where key evidence of a query may be dispersed across millions of tokens. Existing works equip large language models with a memory buffer that is dynamically updated via a linear document scan, also known as the "memorize while reading" methods. While this approach scales efficiently, it suffers from pruning of latent evidence, information loss through overwriting, and sparse reinforcement learning signals. To tackle these challenges, we present ReMemR1, which integrates the mechanism of memory retrieval into the memory update process, enabling the agent to selectively callback historical memories for non-linear reasoning. To further strengthen training, we propose a multi-level reward design, which combines final-answer rewards with dense, step-level signals that guide effective memory use. Together, these contributions mitigate information degradation, improve supervision, and support complex multi-hop reasoning. Extensive experiments demonstrate that ReMemR1 significantly outperforms state-of-the-art baselines on long-context question answering while incurring negligible computational overhead, validating its ability to trade marginal cost for robust long-context reasoning.
>
---
#### [replaced 065] CodePDE: An Inference Framework for LLM-driven PDE Solver Generation
- **分类: cs.LG; cs.AI; cs.CL; math.NA**

- **简介: 该论文属于科学计算任务，旨在解决PDE求解难题。通过将PDE求解转化为代码生成问题，提出CodePDE框架，利用大语言模型实现高效、可解释的求解器生成。**

- **链接: [https://arxiv.org/pdf/2505.08783v2](https://arxiv.org/pdf/2505.08783v2)**

> **作者:** Shanda Li; Tanya Marwah; Junhong Shen; Weiwei Sun; Andrej Risteski; Yiming Yang; Ameet Talwalkar
>
> **备注:** TMLR. Code available at https://github.com/LithiumDA/CodePDE
>
> **摘要:** Partial differential equations (PDEs) are fundamental to modeling physical systems, yet solving them remains a complex challenge. Traditional numerical solvers rely on expert knowledge to implement and are computationally expensive, while neural-network-based solvers require large training datasets and often lack interpretability. In this work, we frame PDE solving as a code generation task and introduce CodePDE, the first inference framework for generating PDE solvers using large language models (LLMs). With CodePDE, we present a thorough evaluation on critical capacities of LLM for PDE solving: reasoning, debugging, self-refinement, and test-time scaling. CodePDE shows that, with advanced inference-time algorithms and scaling strategies, LLMs can achieve strong performance across a range of representative PDE problems. We also identify novel insights into LLM-driven solver generation, such as trade-offs between solver reliability and sophistication, design principles for LLM-powered PDE solving agents, and failure modes for LLM on hard tasks. These insights offer guidance for building more capable and reliable LLM-based scientific engines.
>
---
#### [replaced 066] EconCausal: A Context-Aware Causal Reasoning Benchmark for Large Language Models in Social Science
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于因果推理任务，旨在解决LLMs在社会科学研究中识别因果关系的挑战。通过构建EconCausal基准，评估模型在不同情境下的因果推理能力。**

- **链接: [https://arxiv.org/pdf/2510.07231v3](https://arxiv.org/pdf/2510.07231v3)**

> **作者:** Donggyu Lee; Hyeok Yun; Meeyoung Cha; Sungwon Park; Sangyoon Park; Jihee Kim
>
> **摘要:** Socio-economic causal effects depend heavily on their specific institutional and environmental context. A single intervention can produce opposite results depending on regulatory or market factors, contexts that are often complex and only partially observed. This poses a significant challenge for large language models (LLMs) in decision-support roles: can they distinguish structural causal mechanisms from surface-level correlations when the context changes? To address this, we introduce EconCausal, a large-scale benchmark comprising 10,490 context-annotated causal triplets extracted from 2,595 high-quality empirical studies published in top-tier economics and finance journals. Through a rigorous four-stage pipeline combining multi-run consensus, context refinement, and multi-critic filtering, we ensure each claim is grounded in peer-reviewed research with explicit identification strategies. Our evaluation reveals critical limitations in current LLMs' context-dependent reasoning. While top models achieve approximately 88 percent accuracy in fixed, explicit contexts, performance drops sharply under context shifts, with a 32.6 percentage point decline, and falls to 37 percent when misinformation is introduced. Furthermore, models exhibit severe over-commitment in ambiguous cases and struggle to recognize null effects, achieving only 9.5 percent accuracy, exposing a fundamental gap between pattern matching and genuine causal reasoning. These findings underscore substantial risks for high-stakes economic decision-making, where the cost of misinterpreting causality is high. The dataset and benchmark are publicly available at https://github.com/econaikaist/econcausal-benchmark.
>
---
#### [replaced 067] Manipulating language models' training data to study syntactic constraint learning: the case of English passivization
- **分类: cs.CL**

- **简介: 该论文属于语言习得研究，探讨英语被动语态的例外情况。通过调整训练数据，研究模型如何学习动词的被动化限制，验证频率和语义因素的影响。**

- **链接: [https://arxiv.org/pdf/2407.04593v2](https://arxiv.org/pdf/2407.04593v2)**

> **作者:** Cara Su-Yi Leong; Tal Linzen
>
> **备注:** Revised analysis in Section 4, re-implemented experiments in Sections 6 and 7, added new experiment in Section 8
>
> **摘要:** Grammatical rules in natural languages are often characterized by exceptions. How do language learners learn these exceptions to otherwise general patterns? Here, we study this question through the case study of English passivization. While passivization is in general quite productive, there are cases where it cannot apply (cf. the following sentence is ungrammatical: *One hour was lasted by the meeting). Using neural network language models as theories of language acquisition, we explore the sources of indirect evidence that a learner can leverage to learn whether a verb can be passivized. We first characterize English speakers' judgments of exceptions to the passive, and confirm that speakers find some verbs more passivizable than others. We then show that a neural network language model's verb passivizability judgments are largely similar to those displayed by humans, suggesting that evidence for these exceptions is available in the linguistic input. Finally, we test two hypotheses as to the source of evidence that language models use to learn these restrictions: frequency (entrenchment) and semantics (affectedness). We do so by training models on versions of the corpus that have had sentences of the types implicated by each hypothesis removed, altered, or introduced. We find support for both hypotheses: entrenchment and affectedness make independent contributions to a verb's passivizability. From a methodological point of view, this study highlights the utility of altering a language model's training data for answering questions where complete control over a learner's input is vital.
>
---
#### [replaced 068] Collaborative Document Editing with Multiple Users and AI Agents
- **分类: cs.HC; cs.CL**

- **简介: 该论文属于协作写作任务，解决多人协作中AI工具使用不便的问题。通过将AI代理集成到协作环境，使AI资源可共享，提升团队协作效率。**

- **链接: [https://arxiv.org/pdf/2509.11826v2](https://arxiv.org/pdf/2509.11826v2)**

> **作者:** Florian Lehmann; Krystsina Shauchenka; Daniel Buschek
>
> **备注:** 27 pages, 10 figures, 6 tables, ACM CHI 2026
>
> **摘要:** Current AI writing support tools are largely designed for individuals, complicating collaboration when co-writers must leave the shared workspace to use AI and then communicate and reintegrate results. We propose integrating AI agents directly into collaborative writing environments. Our prototype makes AI use visible to all users through two new shared objects: user-defined agent profiles and tasks. Agent responses appear in the familiar comment feature. In a user study (N=30), 14 teams worked on writing projects during one week. Interaction logs and interviews show that teams incorporated agents into existing norms of authorship, control, and coordination, rather than treating them as team members. Agent profiles were viewed as personal territory, while created agents and outputs became shared resources. We discuss implications for team-based AI interaction, highlighting opportunities and boundaries for treating AI as a shared resource in collaborative work.
>
---
#### [replaced 069] EBPO: Empirical Bayes Shrinkage for Stabilizing Group-Relative Policy Optimization
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出EBPO方法，解决RLVR中GRPO的稳定性问题，通过贝叶斯收缩提升训练稳定性和性能。**

- **链接: [https://arxiv.org/pdf/2602.05165v3](https://arxiv.org/pdf/2602.05165v3)**

> **作者:** Kevin Han; Yuhang Zhou; Mingze Gao; Gedi Zhou; Serena Li; Abhishek Kumar; Xiangjun Fan; Weiwei Li; Lizhu Zhang
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has proven effective for enhancing the reasoning capabilities of Large Language Models (LLMs). However, dominant approaches like Group Relative Policy Optimization (GRPO) face critical stability challenges: they suffer from high estimator variance under computational constraints (small group sizes) and vanishing gradient signals in saturated failure regimes where all responses yield identical zero rewards. To address this, we propose Empirical Bayes Policy Optimization (EBPO), a novel framework that regularizes local group-based baselines by borrowing strength from the policy's accumulated global statistics. Instead of estimating baselines in isolation, EBPO employs a shrinkage estimator that dynamically balances local group statistics with a global prior updated via Welford's online algorithm. Theoretically, we demonstrate that EBPO guarantees strictly lower Mean Squared Error (MSE), bounded entropy decay, and non-vanishing penalty signals in failure scenarios compared to GRPO. Empirically, EBPO consistently outperforms GRPO and other established baselines across diverse benchmarks, including AIME and OlympiadBench. Notably, EBPO exhibits superior training stability, achieving high-performance gains even with small group sizes, and benefits significantly from difficulty-stratified curriculum learning.
>
---
#### [replaced 070] PonderLM-2: Pretraining LLM with Latent Thoughts in Continuous Space
- **分类: cs.CL**

- **简介: 该论文提出PonderLM-2，通过在预训练中引入隐含思维步骤，提升语言模型生成质量。任务是改进语言模型的生成能力，解决如何在不增加参数量的情况下提升性能的问题。工作包括设计预训练方法，实验验证效果。**

- **链接: [https://arxiv.org/pdf/2509.23184v3](https://arxiv.org/pdf/2509.23184v3)**

> **作者:** Boyi Zeng; He Li; Shixiang Song; Yixuan Wang; Ziwei He; Xinbing Wang; Zhouhan Lin
>
> **摘要:** The remarkable success of Chain-of-Thought (CoT), which enhances performance by scaling generation steps at test-time, inspires us to ask: can we leverage a similar scaling of computational steps during pretraining to improve the generation of each individual token? To address this, we propose a novel pre-training methodology: Pretraining Language Models with Latent Thoughts (PonderLM-2). Our approach pretrains a language model (LM) to first generate an intermediate latent thought-the last hidden state of the current position-which is then used as input to predict the actual subsequent token. This additional computational step enables the LM to refine its prediction within unconstrained continuous space. Our experiments demonstrate that, at an identical inference cost, a LM that generates one additional latent thought per token outperforms a standard model with double the parameters. For instance, our PonderLM-2-Pythia-1.4B, pretrained on 300B tokens from the Pile, significantly surpasses the vanilla Pythia-2.8B trained on the same data on both language modeling and a range of general downstream tasks. Furthermore, increasing the number of latent thoughts generated before each actual token-forming a chain analogous to CoT-consistently improves the model's performance. The code is available at https://github.com/LUMIA-Group/PonderLM-2.
>
---
#### [replaced 071] BEAT: Visual Backdoor Attacks on VLM-based Embodied Agents via Contrastive Trigger Learning
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于安全领域，针对VLM-based embodied agents的视觉后门攻击问题，提出BEAT框架，通过对比触发学习实现高效攻击。**

- **链接: [https://arxiv.org/pdf/2510.27623v3](https://arxiv.org/pdf/2510.27623v3)**

> **作者:** Qiusi Zhan; Hyeonjeong Ha; Rui Yang; Sirui Xu; Hanyang Chen; Liang-Yan Gui; Yu-Xiong Wang; Huan Zhang; Heng Ji; Daniel Kang
>
> **备注:** ICLR 2026. Project Page: https://zqs1943.github.io/BEAT/
>
> **摘要:** Recent advances in Vision-Language Models (VLMs) have propelled embodied agents by enabling direct perception, reasoning, and planning task-oriented actions from visual inputs. However, such vision-driven embodied agents open a new attack surface: visual backdoor attacks, where the agent behaves normally until a visual trigger appears in the scene, then persistently executes an attacker-specified multi-step policy. We introduce BEAT, the first framework to inject such visual backdoors into VLM-based embodied agents using objects in the environments as triggers. Unlike textual triggers, object triggers exhibit wide variation across viewpoints and lighting, making them difficult to implant reliably. BEAT addresses this challenge by (1) constructing a training set that spans diverse scenes, tasks, and trigger placements to expose agents to trigger variability, and (2) introducing a two-stage training scheme that first applies supervised fine-tuning (SFT) and then our novel Contrastive Trigger Learning (CTL). CTL formulates trigger discrimination as preference learning between trigger-present and trigger-free inputs, explicitly sharpening the decision boundaries to ensure precise backdoor activation. Across various embodied agent benchmarks and VLMs, BEAT achieves attack success rates up to 80%, while maintaining strong benign task performance, and generalizes reliably to out-of-distribution trigger placements. Notably, compared to naive SFT, CTL boosts backdoor activation accuracy up to 39% under limited backdoor data. These findings expose a critical yet unexplored security risk in VLM-based embodied agents, underscoring the need for robust defenses before real-world deployment.
>
---
#### [replaced 072] Fast-weight Product Key Memory
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出FwPKM，解决语言模型中存储与效率的矛盾，通过稀疏参数更新实现高效记忆，提升长文本任务性能。**

- **链接: [https://arxiv.org/pdf/2601.00671v2](https://arxiv.org/pdf/2601.00671v2)**

> **作者:** Tianyu Zhao; Llion Jones
>
> **摘要:** Sequence modeling layers in modern language models typically face a trade-off between storage capacity and computational efficiency. While softmax attention offers unbounded storage at prohibitive quadratic cost, linear variants are more efficient but suffer from limited, fixed-size storage. We introduce Fast-weight Product Key Memory (FwPKM), a sparse fast-weight memory layer that resolves this tension. FwPKM updates sparsely activated parameters at both training and inference time using chunk-level gradient descent on a local memory-rewrite objective. This performs Test-Time Training (TTT)-style gradient updates on activated slots in a sparse memory, enabling rapid memorization and retrieval of many new key-value associations while keeping per-token compute low and fixed. Experiments show that FwPKM functions as an effective episodic memory that complements the semantic memory of standard modules, yielding significant perplexity reductions on long-context datasets. Notably, in Needle-in-a-Haystack evaluations, FwPKM generalizes to 128K-token contexts despite being trained on only 4K-token sequences.
>
---
#### [replaced 073] Step 3.5 Flash: Open Frontier-Level Intelligence with 11B Active Parameters
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Step 3.5 Flash，一种高效稀疏Mixture-of-Experts模型，解决智能代理的推理与执行效率问题，通过优化架构和训练方法提升性能。**

- **链接: [https://arxiv.org/pdf/2602.10604v2](https://arxiv.org/pdf/2602.10604v2)**

> **作者:** Ailin Huang; Ang Li; Aobo Kong; Bin Wang; Binxing Jiao; Bo Dong; Bojun Wang; Boyu Chen; Brian Li; Buyun Ma; Chang Su; Changxin Miao; Changyi Wan; Chao Lou; Chen Hu; Chen Xu; Chenfeng Yu; Chengting Feng; Chengyuan Yao; Chunrui Han; Dan Ma; Dapeng Shi; Daxin Jiang; Dehua Ma; Deshan Sun; Di Qi; Enle Liu; Fajie Zhang; Fanqi Wan; Guanzhe Huang; Gulin Yan; Guoliang Cao; Guopeng Li; Han Cheng; Hangyu Guo; Hanshan Zhang; Hao Nie; Haonan Jia; Haoran Lv; Hebin Zhou; Hekun Lv; Heng Wang; Heung-Yeung Shum; Hongbo Huang; Hongbo Peng; Hongyu Zhou; Hongyuan Wang; Houyong Chen; Huangxi Zhu; Huimin Wu; Huiyong Guo; Jia Wang; Jian Zhou; Jianjian Sun; Jiaoren Wu; Jiaran Zhang; Jiashu Lv; Jiashuo Liu; Jiayi Fu; Jiayu Liu; Jie Cheng; Jie Luo; Jie Yang; Jie Zhou; Jieyi Hou; Jing Bai; Jingcheng Hu; Jingjing Xie; Jingwei Wu; Jingyang Zhang; Jishi Zhou; Junfeng Liu; Junzhe Lin; Ka Man Lo; Kai Liang; Kaibo Liu; Kaijun Tan; Kaiwen Yan; Kaixiang Li; Kang An; Kangheng Lin; Lei Yang; Liang Lv; Liang Zhao; Liangyu Chen; Lieyu Shi; Liguo Tan; Lin Lin; Lina Chen; Luck Ma; Mengqiang Ren; Michael Li; Ming Li; Mingliang Li; Mingming Zhang; Mingrui Chen; Mitt Huang; Na Wang; Peng Liu; Qi Han; Qian Zhao; Qinglin He; Qinxin Du; Qiuping Wu; Quan Sun; Rongqiu Yang; Ruihang Miao; Ruixin Han; Ruosi Wan; Ruyan Guo; Shan Wang; Shaoliang Pang; Shaowen Yang; Shengjie Fan; Shijie Shang; Shiliang Yang; Shiwei Li; Shuangshuang Tian; Siqi Liu; Siye Wu; Siyu Chen; Song Yuan; Tiancheng Cao; Tianchi Yue; Tianhao Cheng; Tianning Li; Tingdan Luo; Wang You; Wei Ji; Wei Yuan; Wei Zhang; Weibo Wu; Weihao Xie; Wen Sun; Wenjin Deng; Wenzhen Zheng; Wuxun Xie; Xiangfeng Wang; Xiangwen Kong; Xiangyu Liu; Xiangyu Zhang; Xiaobo Yang; Xiaojia Liu; Xiaolan Yuan; Xiaoran Jiao; Xiaoxiao Ren; Xiaoyun Zhang; Xin Li; Xin Liu; Xin Wu; Xing Chen; Xingping Yang; Xinran Wang; Xu Zhao; Xuan He; Xuanti Feng; Xuedan Cai; Xuqiang Zhou; Yanbo Yu; Yang Li; Yang Xu; Yanlin Lai; Yanming Xu; Yaoyu Wang; Yeqing Shen; Yibo Zhu; Yichen Lv; Yicheng Cao; Yifeng Gong; Yijing Yang; Yikun Yang; Yin Zhao; Yingxiu Zhao; Yinmin Zhang; Yitong Zhang; Yixuan Zhang; Yiyang Chen; Yongchi Zhao; Yongshen Long; Yongyao Wang; Yousong Guan; Yu Zhou; Yuang Peng; Yuanhao Ding; Yuantao Fan; Yuanwei Lu; Yuanzhen Yang; Yuchu Luo; Yudi Zhao; Yue Peng; Yueqiang Lin; Yufan Lu; Yuling Zhao; Yunzhou Ju; Yurong Zhang; Yusheng Li; Yuxiang Yang; Yuyang Chen; Yuzhu Cai; Zejia Weng; Zetao Hong; Zexi Li; Zhe Xie; Zheng Ge; Zheng Gong; Zheng Zeng; Zhenyi Lu; Zhewei Huang; Zhichao Chang; Zhiguo Huang; Zhiheng Hu; Zidong Yang; Zili Wang; Ziqi Ren; Zixin Zhang; Zixuan Wang
>
> **备注:** Technical report for Step 3.5 Flash
>
> **摘要:** We introduce Step 3.5 Flash, a sparse Mixture-of-Experts (MoE) model that bridges frontier-level agentic intelligence and computational efficiency. We focus on what matters most when building agents: sharp reasoning and fast, reliable execution. Step 3.5 Flash pairs a 196B-parameter foundation with 11B active parameters for efficient inference. It is optimized with interleaved 3:1 sliding-window/full attention and Multi-Token Prediction (MTP-3) to reduce the latency and cost of multi-round agentic interactions. To reach frontier-level intelligence, we design a scalable reinforcement learning framework that combines verifiable signals with preference feedback, while remaining stable under large-scale off-policy training, enabling consistent self-improvement across mathematics, code, and tool use. Step 3.5 Flash demonstrates strong performance across agent, coding, and math tasks, achieving 85.4% on IMO-AnswerBench, 86.4% on LiveCodeBench-v6 (2024.08-2025.05), 88.2% on tau2-Bench, 69.0% on BrowseComp (with context management), and 51.0% on Terminal-Bench 2.0, comparable to frontier models such as GPT-5.2 xHigh and Gemini 3.0 Pro. By redefining the efficiency frontier, Step 3.5 Flash provides a high-density foundation for deploying sophisticated agents in real-world industrial environments.
>
---
#### [replaced 074] Neurosymbolic Retrievers for Retrieval-augmented Generation
- **分类: cs.AI; cs.CL; cs.IR; cs.LG**

- **简介: 该论文属于信息检索任务，旨在解决RAG系统透明度低的问题。通过引入符号推理与神经检索结合的神经符号RAG框架，提升文档选择的可解释性与检索过程的清晰度。**

- **链接: [https://arxiv.org/pdf/2601.04568v2](https://arxiv.org/pdf/2601.04568v2)**

> **作者:** Yash Saxena; Manas Gaur
>
> **备注:** 8 pages, 2 Figures, Published in IEEE Intelligent Systems
>
> **摘要:** Retrieval Augmented Generation (RAG) has made significant strides in overcoming key limitations of large language models, such as hallucination, lack of contextual grounding, and issues with transparency. However, traditional RAG systems consist of three interconnected neural components - the retriever, re-ranker, and generator - whose internal reasoning processes remain opaque. This lack of transparency complicates interpretability, hinders debugging efforts, and erodes trust, especially in high-stakes domains where clear decision-making is essential. To address these challenges, we introduce the concept of Neurosymbolic RAG, which integrates symbolic reasoning using a knowledge graph with neural retrieval techniques. This new framework aims to answer two primary questions: (a) Can retrievers provide a clear and interpretable basis for document selection? (b) Can symbolic knowledge enhance the clarity of the retrieval process? We propose three methods to improve this integration. First is MAR (Knowledge Modulation Aligned Retrieval) that employs modulation networks to refine query embeddings using interpretable symbolic features, thereby making document matching more explicit. Second, KG-Path RAG enhances queries by traversing knowledge graphs to improve overall retrieval quality and interpretability. Lastly, Process Knowledge-infused RAG utilizes domain-specific tools to reorder retrieved content based on validated workflows. Preliminary results from mental health risk assessment tasks indicate that this neurosymbolic approach enhances both transparency and overall performance
>
---
#### [replaced 075] BOOM: Beyond Only One Modality KIT's Multimodal Multilingual Lecture Companion
- **分类: cs.CL**

- **简介: 该论文提出BOOM系统，解决多模态多语言课程内容本地化问题。通过联合翻译音频与幻灯片，生成同步的文本、本地化幻灯片和合成语音，提升学习体验。**

- **链接: [https://arxiv.org/pdf/2512.02817v2](https://arxiv.org/pdf/2512.02817v2)**

> **作者:** Sai Koneru; Fabian Retkowski; Christian Huber; Lukas Hilgert; Seymanur Akti; Enes Yavuz Ugan; Alexander Waibel; Jan Niehues
>
> **备注:** Under review
>
> **摘要:** The globalization of education and rapid growth of online learning have made localizing educational content a critical challenge. Lecture materials are inherently multimodal, combining spoken audio with visual slides, which requires systems capable of processing multiple input modalities. To provide an accessible and complete learning experience, translations must preserve all modalities: text for reading, slides for visual understanding, and speech for auditory learning. We present \textbf{BOOM}, a multimodal multilingual lecture companion that jointly translates lecture audio and slides to produce synchronized outputs across three modalities: translated text, localized slides with preserved visual elements, and synthesized speech. This end-to-end approach enables students to access lectures in their native language while aiming to preserve the original content in its entirety. Our experiments demonstrate that slide-aware transcripts also yield cascading benefits for downstream tasks such as summarization and question answering. The demo video and code can be found at https://ai4lt.github.io/boom/ \footnote{All released code and models are licensed under the MIT License}.
>
---
#### [replaced 076] Evaluating LLMs' Divergent Thinking Capabilities for Scientific Idea Generation with Minimal Context
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于科学创意生成任务，旨在评估大语言模型的发散思维能力。通过引入新基准LiveIdeaBench，研究发现传统智能指标无法准确预测科学创意生成能力，提出需专门评估标准。**

- **链接: [https://arxiv.org/pdf/2412.17596v4](https://arxiv.org/pdf/2412.17596v4)**

> **作者:** Kai Ruan; Xuan Wang; Jixiang Hong; Peng Wang; Yang Liu; Hao Sun
>
> **备注:** Updated manuscript and title
>
> **摘要:** While Large Language Models (LLMs) demonstrate remarkable capabilities in scientific tasks such as literature analysis and experimental design (e.g., accurately extracting key findings from papers or generating coherent experimental procedures), existing evaluation benchmarks primarily assess performance using rich contextual inputs. We introduce LiveIdeaBench, a comprehensive benchmark evaluating LLMs' scientific idea generation by assessing divergent thinking capabilities using single-keyword prompts. Drawing from Guilford's creativity theory, our benchmark employs a dynamic panel of state-of-the-art LLMs to assess generated ideas across five key dimensions: originality, feasibility, fluency, flexibility, and clarity. Through extensive experimentation with over 40 leading models across 1,180 keywords spanning 22 scientific domains, we reveal that the scientific idea generation capabilities measured by our benchmark, are poorly predicted by standard metrics of general intelligence. Our results demonstrate that models like QwQ-32B-preview achieve creative performance comparable to top-tier models such as claude-3.7-sonnet:thinking, despite significant gaps in their general intelligence scores. These findings highlight the need for specialized evaluation benchmarks for scientific idea generation and suggest that enhancing these idea generation capabilities in LLMs may require different training strategies than those used for improving general problem-solving abilities, potentially enabling a wider range of AI tools tailored for different stages of the scientific process.
>
---
#### [replaced 077] ViTextVQA: A Large-Scale Visual Question Answering Dataset and a Novel Multimodal Feature Fusion Method for Vietnamese Text Comprehension in Images
- **分类: cs.CL**

- **简介: 该论文属于视觉问答任务，旨在解决图像中越南语文本理解问题。提出ViTextVQA数据集和ViTextBLIP-2方法，提升文本-based VQA性能。**

- **链接: [https://arxiv.org/pdf/2404.10652v5](https://arxiv.org/pdf/2404.10652v5)**

> **作者:** Quan Van Nguyen; Dan Quang Tran; Huy Quang Pham; Thang Kien-Bao Nguyen; Nghia Hieu Nguyen; Kiet Van Nguyen; Ngan Luu-Thuy Nguyen
>
> **备注:** International Journal of Expert Systems with Applications
>
> **摘要:** Visual Question Answering (VQA) is a challenging task that requires the joint understanding of natural language and visual content. While early research primarily focused on recognizing objects and scene context, it often overlooked scene text-an essential source of explicit semantic information. This paper introduces \textbf{ViTextVQA} (\textbf{Vi}etnamese \textbf{Text}-based \textbf{V}isual \textbf{Q}uestion \textbf{A}nswering), the first large-scale Vietnamese dataset specializing in text-based VQA. The dataset contains \textbf{over 16,000} images and \textbf{over 50,000} question-answer pairs. To tackle this task efficiently, \textbf{ViTextBLIP-2} (Vietnamese Text-based Bootstrapped Language-Image Model via Fine-tuning) is proposed, a novel multimodal feature fusion method designed to optimize Vietnamese text-based VQA. Experiments with state-of-the-art models highlight the importance of token ordering in OCR text for answer generation, leading to significant performance improvements. The ViTextVQA dataset is publicly available for research purposes.
>
---
#### [replaced 078] One Token Is Enough: Improving Diffusion Language Models with a Sink Token
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，针对扩散语言模型中的“移动sink”不稳定问题，提出引入一个额外的结构化sink token以提升模型稳定性与性能。**

- **链接: [https://arxiv.org/pdf/2601.19657v4](https://arxiv.org/pdf/2601.19657v4)**

> **作者:** Zihou Zhang; Zheyong Xie; Li Zhong; Haifeng Liu; Yao Hu; Shaosheng Cao
>
> **摘要:** Diffusion Language Models (DLMs) have emerged as a compelling alternative to autoregressive approaches, enabling parallel text generation with competitive performance. Despite these advantages, there is a critical instability in DLMs: the moving sink phenomenon. Our analysis indicates that sink tokens exhibit low-norm representations in the Transformer's value space, and that the moving sink phenomenon serves as a protective mechanism in DLMs to prevent excessive information mixing. However, their unpredictable positions across diffusion steps undermine inference robustness. To resolve this, we propose a simple but effective extra sink token implemented via a modified attention mask. Specifically, we introduce a special token constrained to attend solely to itself, while remaining globally visible to all other tokens. Experimental results demonstrate that introducing a single extra token stabilizes attention sinks, substantially improving model performance. Crucially, further analysis confirms that the effectiveness of this token is independent of its position and characterized by negligible semantic content, validating its role as a robust and dedicated structural sink.
>
---
#### [replaced 079] The AI Memory Gap: Users Misremember What They Created With AI or Without
- **分类: cs.HC; cs.CL**

- **简介: 该论文属于认知与人机交互任务，探讨用户在使用AI生成内容后对来源的记忆偏差。研究发现，用户在混合人机协作中更易混淆内容来源，提示需优化AI工具的设计以减少误解。**

- **链接: [https://arxiv.org/pdf/2509.11851v2](https://arxiv.org/pdf/2509.11851v2)**

> **作者:** Tim Zindulka; Sven Goller; Daniela Fernandes; Robin Welsch; Daniel Buschek
>
> **备注:** 22 pages, 10 figures, 10 tables, ACM CHI 2026
>
> **摘要:** As large language models (LLMs) become embedded in interactive text generation, disclosure of AI as a source depends on people remembering which ideas or texts came from themselves and which were created with AI. We investigate how accurately people remember the source of content when using AI. In a pre-registered experiment, 184 participants generated and elaborated on ideas both unaided and with an LLM-based chatbot. One week later, they were asked to identify the source (noAI vs withAI) of these ideas and texts. Our findings reveal a significant gap in memory: After AI use, the odds of correct attribution dropped, with the steepest decline in mixed human-AI workflows, where either the idea or elaboration was created with AI. We validated our results using a computational model of source memory. Discussing broader implications, we highlight the importance of considering source confusion in the design and use of interactive text generation technologies.
>
---
#### [replaced 080] Efficient Context Propagating Perceiver Architectures for Auto-Regressive Language Modeling
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决Transformer模型中注意力机制复杂度过高的问题。通过改进PerceiverAR架构，提出ECP模型，在降低计算复杂度的同时提升性能。**

- **链接: [https://arxiv.org/pdf/2412.06106v3](https://arxiv.org/pdf/2412.06106v3)**

> **作者:** Kaleel Mahmood; Shaoyi Huang
>
> **摘要:** One of the key challenges in Transformer architectures is the quadratic complexity of the attention mechanism, which limits the efficient processing of long sequences. Many recent research works have attempted to provide a reduction from the $O(n^2)$ time complexity of attention to semi-linear complexity. However, it remains an unsolved problem in the sense of maintaining high performance when complexity is reduced. One of the important works in this respect is the Perceiver class of architectures that have demonstrated excellent performance, while reducing the computation complexity. In this paper, we use the PerceiverAR as a basis and explore the design space of different trade-offs between preserving context and reducing attention complexity. To this end, we develop four new architectural paradigms, the best performing of which we denote as the Efficient Context propagating Perceiver (ECP). ECP has two major advantages over the PerceiverAR. First, the ECP architecture overcomes the main drawback of PercieverAR by utilizing both the context and the latent sequences in autoregressive training. Second, the ECP architecture operates with the same attention complexity as LongLoRA, making it computationally efficient. More importantly, via pairwise segment attention, it extracts better information resulting in improved language modeling. Empirically, we demonstrate that the ECP architecture significantly outperforms other state-of-the-art Transformer models on Wikitext-103, PG-19 and sCIFAR-10.
>
---
#### [replaced 081] Reshaping MOFs text mining with a dynamic multi-agents framework of large language model
- **分类: cs.AI; cond-mat.mtrl-sci; cs.CL**

- **简介: 该论文提出MOFh6系统，用于从文献中提取MOFs合成条件，解决信息分散、不一致的问题。通过大语言模型实现自动化信息提取与标准化处理。**

- **链接: [https://arxiv.org/pdf/2504.18880v4](https://arxiv.org/pdf/2504.18880v4)**

> **作者:** Zuhong Lin; Daoyuan Ren; Kai Ran; Jing Sun; Songlin Yu; Xuefeng Bai; Xiaotian Huang; Haiyang He; Pengxu Pan; Ying Fang; Zhanglin Li; Haipu Li; Jingjing Yao
>
> **备注:** Accepted by TRAMAT 2 (2026) 100176
>
> **摘要:** Accurately identifying the synthesis conditions of metal-organic frameworks (MOFs) is essential for guiding experimental design, yet remains challenging because relevant information in the literature is often scattered, inconsistent, and difficult to interpret. We present MOFh6, a large language model driven system that reads raw articles or crystal codes and converts them into standardized synthesis tables. It links related descriptions across paragraphs, unifies ligand abbreviations with full names, and outputs structured parameters ready for use. MOFh6 achieved 99% extraction accuracy, resolved 94.1% of abbreviation cases across five major publishers, and maintained a precision of 0.93 +/- 0.01. Processing a full text takes 9.6 s, locating synthesis descriptions 36 s, with 100 papers processed for USD 4.24. By replacing static database lookups with real-time extraction, MOFh6 reshapes MOF synthesis research, accelerating the conversion of literature knowledge into practical synthesis protocols and enabling scalable, data-driven materials discovery.
>
---
#### [replaced 082] Uncovering Autoregressive LLM Knowledge of Thematic Fit in Event Representation
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究事件表示中主题适配性估计任务，探讨LLMs是否具备一致的主题适配知识，并通过不同提示策略进行实验，提升基准表现。**

- **链接: [https://arxiv.org/pdf/2410.15173v3](https://arxiv.org/pdf/2410.15173v3)**

> **作者:** Safeyah Khaled Alshemali; Daniel Bauer; Yuval Marton
>
> **备注:** Significant update with massive changes: all experiments rerun with current LLMs; includes new probability estimate analysis and expanded results in Sections 4 and 5
>
> **摘要:** The thematic fit estimation task measures semantic arguments' compatibility with a specific semantic role for a specific predicate. We investigate if LLMs have consistent, expressible knowledge of event arguments' thematic fit by experimenting with various prompt designs, manipulating input context, reasoning, and output forms. We set a new state-of-the-art on thematic fit benchmarks, but show that closed and open weight LLMs respond differently to our prompting strategies: Closed models achieve better scores overall and benefit from multi-step reasoning, but they perform worse at filtering out generated sentences incompatible with the specified predicate, role, and argument.
>
---
#### [replaced 083] A Domain-Adapted Pipeline for Structured Information Extraction from Police Incident Announcements on Social Media
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于信息抽取任务，旨在从社交媒体警察公告中提取结构化信息。针对文本多样性和非正式性问题，提出一种基于Qwen2.5-7B模型的领域适配管道，提升抽取准确率。**

- **链接: [https://arxiv.org/pdf/2512.16183v2](https://arxiv.org/pdf/2512.16183v2)**

> **作者:** Mengfan Shen; Kangqi Song; Xindi Wang; Wei Jia; Tao Wang; Ziqiang Han
>
> **备注:** 41 pages,3figures and 9 tables
>
> **摘要:** Structured information extraction from police incident announcements is crucial for timely and accurate data processing, yet presents considerable challenges due to the variability and informal nature of textual sources such as social media posts. To address these challenges, we developed a domain-adapted extraction pipeline that leverages targeted prompt engineering with parameter-efficient fine-tuning of the Qwen2.5-7B model using Low-Rank Adaptation (LoRA). This approach enables the model to handle noisy, heterogeneous text while reliably extracting 15 key fields, including location, event characteristics, and impact assessment, from a high-quality, manually annotated dataset of 4,933 instances derived from 27,822 police briefing posts on Chinese Weibo (2019-2020). Experimental results demonstrated that LoRA-based fine-tuning significantly improved performance over both the base and instruction-tuned models, achieving an accuracy exceeding 98.36% for mortality detection and Exact Match Rates of 95.31% for fatality counts and 95.54% for province-level location extraction. The proposed pipeline thus provides a validated and efficient solution for multi-task structured information extraction in specialized domains, offering a practical framework for transforming unstructured text into reliable structured data in social science research.
>
---
#### [replaced 084] HebID: Detecting Social Identities in Hebrew-language Political Text
- **分类: cs.CL**

- **简介: 该论文提出HebID，首个用于检测希伯来语政治文本中社会身份的多标签语料库。旨在解决非英语政治语境下的身份识别问题，通过分析政治人物的社交媒体和演讲内容进行研究。**

- **链接: [https://arxiv.org/pdf/2508.15483v3](https://arxiv.org/pdf/2508.15483v3)**

> **作者:** Guy Mor-Lan; Naama Rivlin-Angert; Yael R. Kaplan; Tamir Sheafer; Shaul R. Shenhav
>
> **备注:** EMNLP 2025 (Findings)
>
> **摘要:** Political language is deeply intertwined with social identities. While social identities are often shaped by specific cultural contexts and expressed through particular uses of language, existing datasets for group and identity detection are predominantly English-centric, single-label and focus on coarse identity categories. We introduce HebID, the first multilabel Hebrew corpus for social identity detection: 5,536 sentences from Israeli politicians' Facebook posts (Dec 2018-Apr 2021), manually annotated for twelve nuanced social identities (e.g. Rightist, Ultra-Orthodox, Socially-oriented) grounded by survey data. We benchmark multilabel and single-label encoders alongside 2B-9B-parameter generative LLMs, finding that Hebrew-tuned LLMs provide the best results (macro-$F_1$ = 0.74). We apply our classifier to politicians' Facebook posts and parliamentary speeches, evaluating differences in popularity, temporal trends, clustering patterns, and gender-related variations in identity expression. We utilize identity choices from a national public survey, enabling a comparison between identities portrayed in elite discourse and the public's identity priorities. HebID provides a comprehensive foundation for studying social identities in Hebrew and can serve as a model for similar research in other non-English political contexts.
>
---
#### [replaced 085] Incentivizing Agentic Reasoning in LLM Judges via Tool-Integrated Reinforcement Learning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于模型评估任务，旨在解决LLM judges验证复杂约束和计算能力不足的问题。通过引入代码执行器和强化学习，提出TIR-Judge框架，提升评估准确性与性能。**

- **链接: [https://arxiv.org/pdf/2510.23038v2](https://arxiv.org/pdf/2510.23038v2)**

> **作者:** Ran Xu; Jingjing Chen; Jiayu Ye; Yu Wu; Jun Yan; Carl Yang; Hongkun Yu
>
> **备注:** ICLR 2026
>
> **摘要:** Large Language Models (LLMs) are widely used as judges to evaluate response quality, providing a scalable alternative to human evaluation. However, most LLM judges operate solely on intrinsic text-based reasoning, limiting their ability to verify complex constraints or perform accurate computation. Motivated by the success of tool-integrated reasoning (TIR) in numerous tasks, we propose TIR-Judge, an end-to-end RL framework for training LLM judges that integrates a code executor for precise evaluation. TIR-Judge is built on three principles: (i) diverse training across verifiable and non-verifiable domains, (ii) flexible judgment formats (pointwise, pairwise, listwise), and (iii) iterative RL that bootstraps directly from the initial model without distillation. On seven public benchmarks, TIR-Judge surpasses strong reasoning-based judges by up to 6.4% (pointwise) and 7.7% (pairwise), and achieves listwise performance comparable to Claude-Opus-4 despite having only 8B parameters. Remarkably, TIR-Judge-Zero - trained entirely without distilled judge trajectories, matches the performance of distilled variants, demonstrating that tool-augmented judges can self-evolve through iterative reinforcement learning.
>
---
#### [replaced 086] MCPShield: A Security Cognition Layer for Adaptive Trust Calibration in Model Context Protocol Agents
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于安全防护任务，旨在解决MCP代理与服务器间的信任错位问题。提出MCPShield作为安全认知层，通过元数据探测和历史分析提升代理安全性。**

- **链接: [https://arxiv.org/pdf/2602.14281v2](https://arxiv.org/pdf/2602.14281v2)**

> **作者:** Zhenhong Zhou; Yuanhe Zhang; Hongwei Cai; Moayad Aloqaily; Ouns Bouachir; Linsey Pang; Prakhar Mehrotra; Kun Wang; Qingsong Wen
>
> **备注:** 21 pages, 5 figures, 6 tables
>
> **摘要:** The Model Context Protocol (MCP) standardizes tool use for LLM-based agents and enable third-party servers. This openness introduces a security misalignment: agents implicitly trust tools exposed by potentially untrusted MCP servers. However, despite its excellent utility, existing agents typically offer limited validation for third-party MCP servers. As a result, agents remain vulnerable to MCP-based attacks that exploit the misalignment between agents and servers throughout the tool invocation lifecycle. In this paper, we propose MCPShield as a plug-in security cognition layer that mitigates this misalignment and ensures agent security when invoking MCP-based tools. Drawing inspiration from human experience-driven tool validation, MCPShield assists agent forms security cognition with metadata-guided probing before invocation. Our method constrains execution within controlled boundaries while cognizing runtime events, and subsequently updates security cognition by reasoning over historical traces after invocation, building on human post-use reflection on tool behavior. Experiments demonstrate that MCPShield exhibits strong generalization in defending against six novel MCP-based attack scenarios across six widely used agentic LLMs, while avoiding false positives on benign servers and incurring low deployment overhead. Overall, our work provides a practical and robust security safeguard for MCP-based tool invocation in open agent ecosystems.
>
---
#### [replaced 087] CricBench: A Multilingual Benchmark for Evaluating LLMs in Cricket Analytics
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CricBench，一个用于评估大语言模型在板球分析中表现的多语言基准。解决LLMs在体育数据分析中的领域特定性和多语言挑战问题，通过构建高质量查询数据集并测试多个模型性能。**

- **链接: [https://arxiv.org/pdf/2512.21877v2](https://arxiv.org/pdf/2512.21877v2)**

> **作者:** Vaibhav Devraj; Dhruv Kumar; Jagat Sesh Challa; Parth Agarwal; Navya Kommuri; Trizal Garg; Prisha Singhal; Dhruv Shah
>
> **备注:** Under Review
>
> **摘要:** Cricket is the second most popular sport globally, commanding a massive following of over 2.5 billion fans globally. Enthusiasts and analysts frequently seek advanced statistical insights, such as long-term historical performance trends or complex player comparisons, that are often unavailable through standard web searches. While Large Language Models (LLMs) have advanced significantly in Text-to-SQL tasks, their capability to handle the domain-specific nuances, complex schema variations, and multilingual requirements inherent to sports analytics remains under-explored. To investigate this potential capability gap, we present CricBench, a comprehensive benchmark suite for evaluating LLMs on specialized cricket data. To curate a "Gold Standard" dataset, we collaborate with domain experts in cricket and SQL to manually author complex queries, ensuring logical correctness. Recognizing linguistic diversity, we construct the benchmark in both English and Hindi, establishing a framework that is open for further extension to other regional languages. We evaluate six state-of-the-art models, including GPT-4o, Claude 3.7 Sonnet, and open-source models, using a strict evaluation protocol. Our results reveal that high performance on general benchmarks does not guarantee success in specialized domains. While the open-weights reasoning model DeepSeek R1 achieves state-of-the-art performance (50.6%), surpassing proprietary giants like Claude 3.7 Sonnet (47.7%) and GPT-4o (33.7%), it still exhibits a significant accuracy drop when moving from general benchmarks (BIRD) to CricBench. Furthermore, we observe that code-mixed Hindi queries frequently yield parity or higher accuracy compared to English, challenging the assumption that English is the optimal prompt language for specialized SQL tasks.
>
---
#### [replaced 088] Esoteric Language Models: Bridging Autoregressive and Masked Diffusion LLMs
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语言模型任务，旨在解决扩散模型在困惑度和推理效率上的不足。通过融合自回归与掩码扩散模型，提出Eso-LMs，提升生成速度与质量。**

- **链接: [https://arxiv.org/pdf/2506.01928v3](https://arxiv.org/pdf/2506.01928v3)**

> **作者:** Subham Sekhar Sahoo; Zhihan Yang; Yash Akhauri; Johnna Liu; Deepansha Singh; Zhoujun Cheng; Zhengzhong Liu; Eric Xing; John Thickstun; Arash Vahdat
>
> **摘要:** Diffusion-based language models offer a compelling alternative to autoregressive (AR) models by enabling parallel and controllable generation. Within this family, Masked Diffusion Models (MDMs) currently perform best but still underperform AR models in perplexity and lack key inference-time efficiency features, most notably KV caching. We introduce Eso-LMs, a new family of models that fuses AR and MDM paradigms, smoothly interpolating between their perplexities while overcoming their respective limitations. Unlike prior work, which uses transformers with bidirectional attention as MDM denoisers, we exploit the connection between MDMs and Any-Order autoregressive models and adopt causal attention. This design lets us compute the exact likelihood of MDMs for the first time and, crucially, enables us \to introduce KV caching for MDMs while preserving parallel generation for the first time, significantly improving inference efficiency. Combined with an optimized sampling schedule, Eso-LMs achieves a new state of the art on the speed-quality Pareto frontier for unconditional generation. On long contexts, it yields $\mathbf{14 - 65{}\times}$ faster inference than standard MDMs and $\mathbf{3 - 4{}\times}$ faster inference than prior semi-autoregressive approaches. We provide code, model checkpoints, and a video tutorial on the project page: https://s-sahoo.com/Eso-LMs.
>
---
#### [replaced 089] Role-Aware Language Models for Secure and Contextualized Access Control in Organizations
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于安全与访问控制任务，旨在解决企业中基于用户角色的模型行为控制问题。通过微调语言模型，实现角色感知的访问权限管理。**

- **链接: [https://arxiv.org/pdf/2507.23465v3](https://arxiv.org/pdf/2507.23465v3)**

> **作者:** Saeed Almheiri; Yerulan Kongrat; Adrian Santosh; Ruslan Tasmukhanov; Josemaria Loza Vera; Muhammad Dehan Al Kautsar; Fajri Koto
>
> **备注:** AACL 2025 - Main
>
> **摘要:** As large language models (LLMs) are increasingly deployed in enterprise settings, controlling model behavior based on user roles becomes an essential requirement. Existing safety methods typically assume uniform access and focus on preventing harmful or toxic outputs, without addressing role-specific access constraints. In this work, we investigate whether LLMs can be fine-tuned to generate responses that reflect the access privileges associated with different organizational roles. We explore three modeling strategies: a BERT-based classifier, an LLM-based classifier, and role-conditioned generation. To evaluate these approaches, we construct two complementary datasets. The first is adapted from existing instruction-tuning corpora through clustering and role labeling, while the second is synthetically generated to reflect realistic, role-sensitive enterprise scenarios. We assess model performance across varying organizational structures and analyze robustness to prompt injection, role mismatch, and jailbreak attempts.
>
---
#### [replaced 090] VQEL: Enabling Self-Play in Emergent Language Games via Agent-Internal Vector Quantization
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究如何在人工代理间实现离散语言通信，解决符号采样不可导带来的训练难题。提出VQEL架构，通过向量量化实现自对弈学习，提升通信协议的稳定性与可扩展性。**

- **链接: [https://arxiv.org/pdf/2503.04940v2](https://arxiv.org/pdf/2503.04940v2)**

> **作者:** Mohammad Mahdi Samiei Paqaleh; Mehdi Jamalkhah; Mahdieh Soleymani Baghshah
>
> **摘要:** Emergent Language (EL) focuses on the emergence of communication among artificial agents. Although symbolic communication channels more closely mirror the discrete nature of human language, learning such protocols remains fundamentally difficult due to the non-differentiability of symbol sampling. Existing approaches typically rely on high-variance gradient estimators such as REINFORCE or on continuous relaxations such as Gumbel-Softmax, both of which suffer from limitations in training stability and scalability. Motivated by cognitive theories that emphasize intrapersonal processes preceding communication, we explore self-play as a substrate for language emergence prior to mutual interaction. We introduce Vector Quantized Emergent Language (VQEL), a novel architecture that incorporates vector quantization into the message generation process. VQEL enables agents to perform self-play using discrete internal representations derived from a learned codebook while preserving end-to-end differentiability. Moreover, the resulting vector-quantized codebook naturally induces a symbolic vocabulary that can be directly transferred and aligned during subsequent mutual play with other agents. Empirical results show that agents pretrained via VQEL self-play achieve more consistent symbol alignment and higher task success when later engaged in mutual interaction. These findings position self-play as a principled and effective mechanism for learning discrete communication protocols, addressing key optimization and representational challenges in emergent language systems.
>
---
#### [replaced 091] FrugalPrompt: Reducing Contextual Overhead in Large Language Models via Token Attribution
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型因冗余上下文导致的效率问题。通过提取关键语义token，提出FrugalPrompt框架以提升性能与效率的平衡。**

- **链接: [https://arxiv.org/pdf/2510.16439v3](https://arxiv.org/pdf/2510.16439v3)**

> **作者:** Syed Rifat Raiyan; Md Farhan Ishmam; Abdullah Al Imran; Mohammad Ali Moni
>
> **摘要:** Human communication heavily relies on laconism and inferential pragmatics, allowing listeners to successfully reconstruct rich meaning from sparse, telegraphic speech. In contrast, large language models (LLMs) owe much of their stellar performance to expansive input contexts, yet such verbosity inflates monetary costs, carbon footprint, and inference-time latency. This overhead manifests from the redundant low-utility tokens present in typical prompts, as only a fraction of tokens typically carries the majority of the semantic weight. Inspired by the aforementioned cognitive psycholinguistic processes, we address this inefficiency by introducing FrugalPrompt, a novel prompt compression framework for LLMs, which retains only the most semantically significant tokens. Leveraging two state-of-the-art token attribution methods, GlobEnc and DecompX, we assign salience scores to every token in an input sequence, rank them to retain the top-k% tokens, and obtain a sparse frugalized prompt. We establish the theoretical stability of our approach and provide strong empirical results across a suite of four NLP tasks to study the trade-off between the portion of retained tokens and performance. Experimental findings across retention settings reveal asymmetric performance patterns that suggest potential task contamination effects. We posit that our work contributes to a more nuanced understanding of LLM behavior in performance-efficiency trade-offs and delineates the boundary between tasks tolerant of contextual sparsity and those requiring exhaustive context.
>
---
#### [replaced 092] RAIR: A Rule-Aware Benchmark Uniting Challenging Long-Tail and Visual Salience Subset for E-commerce Relevance Assessment
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出RAIR基准，用于电商相关性评估，解决现有基准复杂度不足的问题。构建包含三个子集的数据集，评估模型在长尾和视觉显著性任务中的表现。**

- **链接: [https://arxiv.org/pdf/2512.24943v2](https://arxiv.org/pdf/2512.24943v2)**

> **作者:** Chenji Lu; Zhuo Chen; Hui Zhao; Zhenyi Wang; Pengjie Wang; Chuan Yu; Jian Xu
>
> **摘要:** Search relevance plays a central role in web e-commerce. While large language models (LLMs) have shown significant results on relevance task, existing benchmarks lack sufficient complexity for comprehensive model assessment, resulting in an absence of standardized relevance evaluation metrics across the industry. To address this limitation, we propose Rule-Aware benchmark with Image for Relevance assessment(RAIR), a Chinese dataset derived from real-world scenarios. RAIR established a standardized framework for relevance assessment and provides a set of universal rules, which forms the foundation for standardized evaluation. Additionally, RAIR analyzes essential capabilities required for current relevance models and introduces a comprehensive dataset consists of three subset: (1) a general subset with industry-balanced sampling to evaluate fundamental model competencies; (2) a long-tail hard subset focus on challenging cases to assess performance limits; (3) a visual salience subset for evaluating multimodal understanding capabilities. We conducted experiments on RAIR using 14 open and closed-source models. The results demonstrate that RAIR presents sufficient challenges even for GPT-5, which achieved the best performance. RAIR data are now available, serving as an industry benchmark for relevance assessment while providing new insights into general LLM and Visual Language Model(VLM) evaluation.
>
---
